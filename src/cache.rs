//! Texture cache: avoid regenerating identical configs across spawns.
//!
//! Every texture generated through [`build_procedural_material_async`] takes
//! tens to hundreds of milliseconds at common resolutions.  Re-rolling the
//! same `(generator kind, config, width, height)` tuple wastes that time —
//! the cache stores the resulting [`GeneratedHandles`] (cheap `Handle<Image>`
//! clones, **not** raw pixel buffers) and short-circuits subsequent requests.
//!
//! Two storage backends ship with the crate:
//!
//! * [`MemoryStore`] — process-local `HashMap`.  Bounded by an LRU-trimmed
//!   `max_entries`; entries that exceed the cap are dropped in insertion
//!   order.  Use this for per-app caches (default biomes warm-up, hot
//!   parameter sweeps).
//! * [`FileStore`] — disk-backed key/value store keyed by SipHash-13 of
//!   the cache key.  Survives process restarts and lets a CLI tool warm
//!   the cache from a manifest before the application launches.  Stored
//!   blobs are raw RGBA8 levels (albedo + normal + ORM) and are re-uploaded
//!   into [`Assets<Image>`] on the first hit.
//!
//! Cache invalidation is the user's responsibility: bump the
//! [`TextureCache::manifest_version`] when generator output changes (new
//! noise weights, fixed bugs, …) and stale entries become unreachable.
//!
//! [`build_procedural_material_async`]: crate::material::build_procedural_material_async
//! [`GeneratedHandles`]: crate::generator::GeneratedHandles

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::asset::Assets;
use bevy::ecs::resource::Resource;
use bevy::image::Image;

use crate::generator::{GeneratedHandles, TextureMap, map_to_images, map_to_images_card};

/// Default maximum number of entries kept in [`MemoryStore`].
///
/// At common resolutions each entry costs three [`Image`] handles + their
/// pixel buffers (a few hundred kilobytes).  256 entries hovers around 100 MB
/// of GPU memory and covers most building/biome palettes without thrashing.
pub const DEFAULT_MEMORY_CACHE_ENTRIES: usize = 256;

/// Stable identifier for a cached texture set.
///
/// Combines the generator kind (e.g. `"Bark"`), a fingerprint of the config
/// (`TextureConfig::fingerprint`), and the requested resolution.  `kind` is
/// stored as `&'static str` so cloning a key is `Copy`-cheap; only the
/// fingerprint and dimensions allocate.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TextureCacheKey {
    /// Generator kind label — `TextureConfig::label()`.
    pub kind: &'static str,
    /// `TextureConfig::fingerprint()` — opaque u64.
    pub fingerprint: u64,
    pub width: u32,
    pub height: u32,
}

/// Trait implemented by texture cache backends.
///
/// Implementations must be `Send + Sync` — Bevy's resource lookup hands
/// `&mut TextureCache` to systems on the main scheduling thread, but the
/// trait object can be queried from any thread that holds a reference.
pub trait TextureCacheStore: Send + Sync {
    /// Returns the handles previously stored under `key`, or `None` on miss.
    ///
    /// Implementations that load lazily (e.g. [`FileStore`]) should perform
    /// the I/O and image upload here, returning fully-populated handles ready
    /// to be assigned to `StandardMaterial` slots.
    fn get(
        &mut self,
        key: &TextureCacheKey,
        images: &mut Assets<Image>,
    ) -> Option<Arc<GeneratedHandles>>;

    /// Stores `handles` under `key`, evicting older entries if needed.
    ///
    /// `is_card` lets disk-backed implementations record the upload mode so
    /// the next cold start can choose between
    /// [`map_to_images`](crate::generator::map_to_images) and
    /// [`map_to_images_card`](crate::generator::map_to_images_card).
    fn put(
        &mut self,
        key: TextureCacheKey,
        handles: Arc<GeneratedHandles>,
        is_card: bool,
        map: Option<&TextureMap>,
    );

    /// Optional fast path used by
    /// [`TextureCache::get_handles`] when no `Assets<Image>` is on hand.
    ///
    /// Backends that materialise handles purely from RAM (e.g. [`MemoryStore`])
    /// should override this to return them directly.  Backends that need to
    /// upload pixels to GPU on first hit (e.g. [`FileStore`]) leave the
    /// default — `None` — and the consumer falls back to the regular `get`
    /// path on the system thread that has `Assets<Image>` available.
    fn peek_memory_only(&self, _key: &TextureCacheKey) -> Option<Arc<GeneratedHandles>> {
        None
    }
}

/// Bevy resource wrapper for any [`TextureCacheStore`] implementation.
///
/// Insert this resource before adding [`SymbiosTexturePlugin`](crate::SymbiosTexturePlugin)
/// (or before the first call to
/// [`build_procedural_material_async`](crate::material::build_procedural_material_async))
/// to enable caching:
///
/// ```rust,ignore
/// app.insert_resource(TextureCache::memory(DEFAULT_MEMORY_CACHE_ENTRIES));
/// ```
///
/// `manifest_version` is mixed into the on-disk filename of [`FileStore`]
/// blobs so bumping it invalidates every prior entry without manual cleanup.
#[derive(Resource)]
pub struct TextureCache {
    pub manifest_version: u32,
    inner: Mutex<Box<dyn TextureCacheStore>>,
}

impl TextureCache {
    /// Wrap any [`TextureCacheStore`] in a [`TextureCache`] resource.
    pub fn new(store: Box<dyn TextureCacheStore>, manifest_version: u32) -> Self {
        Self {
            manifest_version,
            inner: Mutex::new(store),
        }
    }

    /// Convenience: in-memory cache with the default capacity.
    pub fn memory(max_entries: usize) -> Self {
        Self::new(Box::new(MemoryStore::new(max_entries)), 0)
    }

    /// Convenience: file-backed cache rooted at `dir`.
    ///
    /// The directory is created if missing.  Each entry produces one
    /// `<hash>-<manifest_version>.bin` file containing the three RGBA8
    /// pixel buffers concatenated; `images: &mut Assets<Image>` is required
    /// at lookup time to upload the blobs into Bevy's asset system.
    pub fn file(dir: impl Into<PathBuf>, manifest_version: u32) -> std::io::Result<Self> {
        Ok(Self::new(
            Box::new(FileStore::new(dir.into())?),
            manifest_version,
        ))
    }

    /// Look up a key without touching `Assets<Image>`.
    ///
    /// Used by the synchronous fast path in
    /// [`build_procedural_material_async`](crate::material::build_procedural_material_async),
    /// where the helper has only the materials store on hand and the polling
    /// system later patches the textures in.  Returns `None` for backends
    /// that need image upload to materialise handles ([`FileStore`] is one
    /// such: a separate full lookup happens in the polling system).
    pub fn get_handles(&self, key: &TextureCacheKey) -> Option<Arc<GeneratedHandles>> {
        self.inner.lock().ok()?.peek_memory_only(key)
    }

    /// Insert handles for `key`.  Mirrors `TextureCacheStore::put` without
    /// the texture-map (memory backends never need it).
    pub fn insert(&mut self, key: TextureCacheKey, handles: Arc<GeneratedHandles>) {
        if let Ok(mut store) = self.inner.lock() {
            store.put(key, handles, false, None);
        }
    }
}

/// In-memory LRU-style cache with bounded capacity.
///
/// Eviction is FIFO on insertion order — simpler than full LRU and adequate
/// for the access pattern (palettes loaded in bulk, hits clustered around
/// hot configs).  When the cap is reached the oldest entry is dropped.
pub struct MemoryStore {
    max_entries: usize,
    entries: HashMap<TextureCacheKey, Arc<GeneratedHandles>>,
    insertion_order: VecDeque<TextureCacheKey>,
}

impl MemoryStore {
    pub fn new(max_entries: usize) -> Self {
        let cap = max_entries.max(1);
        Self {
            max_entries: cap,
            entries: HashMap::with_capacity(cap),
            insertion_order: VecDeque::with_capacity(cap),
        }
    }
}

impl TextureCacheStore for MemoryStore {
    fn get(
        &mut self,
        key: &TextureCacheKey,
        _images: &mut Assets<Image>,
    ) -> Option<Arc<GeneratedHandles>> {
        self.entries.get(key).cloned()
    }

    fn put(
        &mut self,
        key: TextureCacheKey,
        handles: Arc<GeneratedHandles>,
        _is_card: bool,
        _map: Option<&TextureMap>,
    ) {
        // Replace path: keep insertion order untouched, just refresh the value.
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.entries.entry(key.clone())
        {
            e.insert(handles);
            return;
        }
        if self.entries.len() >= self.max_entries
            && let Some(oldest) = self.insertion_order.pop_front()
        {
            self.entries.remove(&oldest);
        }
        self.insertion_order.push_back(key.clone());
        self.entries.insert(key, handles);
    }

    fn peek_memory_only(&self, key: &TextureCacheKey) -> Option<Arc<GeneratedHandles>> {
        self.entries.get(key).cloned()
    }
}

/// On-disk binary blob layout:
///
/// ```text
/// magic:        b"BSTX"        (4 bytes)
/// version:      u32 LE
/// is_card:      u8
/// width:        u32 LE
/// height:       u32 LE
/// albedo_len:   u32 LE
/// normal_len:   u32 LE
/// roughness_len:u32 LE
/// albedo:       albedo_len bytes
/// normal:       normal_len bytes
/// roughness:    roughness_len bytes
/// ```
const FILE_MAGIC: &[u8; 4] = b"BSTX";
const FILE_FORMAT_VERSION: u32 = 1;

/// Disk-backed cache.  Each entry is a single binary blob in `dir`.
///
/// The on-disk filename is `<sip-hash-13(key)>.bin`.  Entries created with a
/// different `manifest_version` are still readable but won't be selected by
/// [`TextureCache::file`] (which mixes the version into the cache key
/// indirectly via the consumer's choice of [`TextureCacheKey::fingerprint`]).
pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub fn new(root: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path_for(&self, key: &TextureCacheKey) -> PathBuf {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        self.root.join(format!("{:016x}.bin", h.finish()))
    }
}

impl TextureCacheStore for FileStore {
    fn get(
        &mut self,
        key: &TextureCacheKey,
        images: &mut Assets<Image>,
    ) -> Option<Arc<GeneratedHandles>> {
        let path = self.path_for(key);
        let mut file = fs::File::open(&path).ok()?;
        let mut header = [0u8; 4 + 4 + 1 + 4 + 4 + 4 + 4 + 4];
        file.read_exact(&mut header).ok()?;
        if &header[0..4] != FILE_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if version != FILE_FORMAT_VERSION {
            return None;
        }
        let is_card = header[8] != 0;
        let width = u32::from_le_bytes(header[9..13].try_into().unwrap());
        let height = u32::from_le_bytes(header[13..17].try_into().unwrap());
        let albedo_len = u32::from_le_bytes(header[17..21].try_into().unwrap()) as usize;
        let normal_len = u32::from_le_bytes(header[21..25].try_into().unwrap()) as usize;
        let roughness_len = u32::from_le_bytes(header[25..29].try_into().unwrap()) as usize;

        let mut albedo = vec![0u8; albedo_len];
        let mut normal = vec![0u8; normal_len];
        let mut roughness = vec![0u8; roughness_len];
        file.read_exact(&mut albedo).ok()?;
        file.read_exact(&mut normal).ok()?;
        file.read_exact(&mut roughness).ok()?;

        let map = TextureMap {
            albedo,
            normal,
            roughness,
            width,
            height,
        };
        let handles = if is_card {
            map_to_images_card(map, images)
        } else {
            map_to_images(map, images)
        };
        Some(Arc::new(handles))
    }

    fn put(
        &mut self,
        key: TextureCacheKey,
        _handles: Arc<GeneratedHandles>,
        is_card: bool,
        map: Option<&TextureMap>,
    ) {
        let Some(map) = map else {
            // No raw pixels to persist — likely an in-process insertion from
            // a memory-only path.  Skipping is correct: we'll re-cache when
            // the next generation pass produces a fresh `TextureMap`.
            return;
        };
        let path = self.path_for(&key);
        if let Err(e) = (|| -> std::io::Result<()> {
            let mut file = fs::File::create(&path)?;
            file.write_all(FILE_MAGIC)?;
            file.write_all(&FILE_FORMAT_VERSION.to_le_bytes())?;
            file.write_all(&[is_card as u8])?;
            file.write_all(&map.width.to_le_bytes())?;
            file.write_all(&map.height.to_le_bytes())?;
            file.write_all(&(map.albedo.len() as u32).to_le_bytes())?;
            file.write_all(&(map.normal.len() as u32).to_le_bytes())?;
            file.write_all(&(map.roughness.len() as u32).to_le_bytes())?;
            file.write_all(&map.albedo)?;
            file.write_all(&map.normal)?;
            file.write_all(&map.roughness)?;
            Ok(())
        })() {
            bevy::log::warn!("FileStore::put failed for {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_handles() -> Arc<GeneratedHandles> {
        Arc::new(GeneratedHandles {
            albedo: Default::default(),
            normal: Default::default(),
            roughness: Default::default(),
        })
    }

    fn key(kind: &'static str, fp: u64) -> TextureCacheKey {
        TextureCacheKey {
            kind,
            fingerprint: fp,
            width: 64,
            height: 64,
        }
    }

    #[test]
    fn memory_store_round_trips_handles() {
        let mut store = MemoryStore::new(8);
        let k = key("Bark", 42);
        assert!(store.peek_memory_only(&k).is_none());
        store.put(k.clone(), dummy_handles(), false, None);
        assert!(store.peek_memory_only(&k).is_some());
    }

    #[test]
    fn memory_store_evicts_oldest_at_capacity() {
        let mut store = MemoryStore::new(2);
        store.put(key("Bark", 1), dummy_handles(), false, None);
        store.put(key("Bark", 2), dummy_handles(), false, None);
        store.put(key("Bark", 3), dummy_handles(), false, None);
        // First entry should have been evicted.
        assert!(store.peek_memory_only(&key("Bark", 1)).is_none());
        assert!(store.peek_memory_only(&key("Bark", 2)).is_some());
        assert!(store.peek_memory_only(&key("Bark", 3)).is_some());
    }

    #[test]
    fn memory_store_treats_replace_as_no_evict() {
        let mut store = MemoryStore::new(2);
        store.put(key("Bark", 1), dummy_handles(), false, None);
        store.put(key("Bark", 2), dummy_handles(), false, None);
        // Re-insert existing key — should not trigger eviction.
        store.put(key("Bark", 1), dummy_handles(), false, None);
        assert!(store.peek_memory_only(&key("Bark", 1)).is_some());
        assert!(store.peek_memory_only(&key("Bark", 2)).is_some());
    }
}
