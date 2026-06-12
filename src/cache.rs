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
//! * [`MemoryStore`] — process-local `HashMap` bounded by `max_entries`.
//!   When the cap is reached, the oldest entry (by insertion order) is
//!   dropped.  Re-inserting an existing key updates in place without
//!   evicting.  Use this for per-app caches (default biomes warm-up, hot
//!   parameter sweeps).
//! * [`FileStore`] — disk-backed key/value store keyed by the standard
//!   library hash (currently SipHash-1-3 via `DefaultHasher`) of the cache
//!   key.  Survives process restarts and lets a CLI tool warm the cache
//!   from a manifest before the application launches.  Stored blobs are
//!   raw RGBA8 base levels (albedo + normal + ORM) — mipmaps are
//!   regenerated on upload by [`map_to_images`] / [`map_to_images_card`].
//!
//! Cache invalidation is driven by [`TextureConfig::fingerprint`]: any change
//! to a config field rolls the fingerprint and therefore the
//! [`TextureCacheKey`], so previously-cached entries become unreachable
//! automatically.  When generator *internals* change (new noise weights, bug
//! fixes, etc.) without a config-field change, callers should rotate the
//! cache directory or bump [`TextureCache::manifest_version`] and act on it
//! externally — the field is stored on the resource for application use but
//! is not currently mixed into on-disk filenames.
//!
//! [`TextureConfig::fingerprint`]: crate::material::TextureConfig::fingerprint
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
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
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
    /// the next cold start can choose between [`map_to_images`] and
    /// [`map_to_images_card`].
    fn put(
        &mut self,
        key: TextureCacheKey,
        handles: Arc<GeneratedHandles>,
        is_card: bool,
        map: Option<&TextureMap>,
    );

    /// Persist the raw pixel buffers for `key` while they are still
    /// available — i.e. **before** the upload consumes the [`TextureMap`].
    ///
    /// [`patch_procedural_material_textures`] calls this on the cache-miss
    /// path with the freshly generated map, then registers the uploaded
    /// handles via [`put`](TextureCacheStore::put) (with `map = None`)
    /// immediately afterwards.  Disk-backed implementations ([`FileStore`])
    /// write their blob here; memory-only backends keep the default no-op.
    ///
    /// [`patch_procedural_material_textures`]: crate::material::patch_procedural_material_textures
    fn put_pixels(&mut self, _key: &TextureCacheKey, _map: &TextureMap, _is_card: bool) {
        // Default: memory-only backends serve handles straight from RAM and
        // have no pixel persistence — the subsequent `put` carries the data
        // they need.  Only disk-backed stores override this.
    }

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
#[derive(Resource)]
pub struct TextureCache {
    /// Application-supplied schema version for the cached blobs.
    ///
    /// Not currently consumed by the built-in stores — entries are keyed on
    /// [`TextureCacheKey`] alone — but exposed so callers can rotate caches
    /// out-of-band when generator internals change without a config-field
    /// change (e.g. delete the cache directory when `manifest_version` differs
    /// from the value baked into a previous build).
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
    /// `<sip-hash-of-key>.bin` file containing a short header (see
    /// [`FileStore`]) followed by the three RGBA8 pixel buffers concatenated.
    /// `images: &mut Assets<Image>` is required at lookup time to upload the
    /// blobs into Bevy's asset system.
    ///
    /// `manifest_version` is recorded on the resource for application use; the
    /// built-in [`FileStore`] does not currently mix it into the on-disk key.
    pub fn file(dir: impl Into<PathBuf>, manifest_version: u32) -> std::io::Result<Self> {
        Ok(Self::new(
            Box::new(FileStore::new(dir.into())?),
            manifest_version,
        ))
    }

    /// Full cache lookup.  Backends that load lazily ([`FileStore`]) read
    /// their blob and upload it into `images` here, so a hit returns handles
    /// ready to assign to `StandardMaterial` slots.
    ///
    /// [`build_procedural_material_async`](crate::material::build_procedural_material_async)
    /// uses this, which is what makes disk-backed caches short-circuit
    /// generation exactly like memory-backed ones.
    pub fn get(
        &self,
        key: &TextureCacheKey,
        images: &mut Assets<Image>,
    ) -> Option<Arc<GeneratedHandles>> {
        self.inner.lock().ok()?.get(key, images)
    }

    /// Look up a key without touching `Assets<Image>`.
    ///
    /// Only backends that can materialise handles from RAM ([`MemoryStore`])
    /// return hits here; disk-backed stores need an image upload and return
    /// `None`.  Prefer [`get`](TextureCache::get) whenever an
    /// `Assets<Image>` is on hand.
    pub fn get_handles(&self, key: &TextureCacheKey) -> Option<Arc<GeneratedHandles>> {
        self.inner.lock().ok()?.peek_memory_only(key)
    }

    /// Persist raw pixels for `key` ahead of the upload that consumes them.
    /// No-op for memory-only backends; see [`TextureCacheStore::put_pixels`].
    pub fn persist_pixels(&self, key: &TextureCacheKey, map: &TextureMap, is_card: bool) {
        if let Ok(mut store) = self.inner.lock() {
            store.put_pixels(key, map, is_card);
        }
    }

    /// Insert handles for `key`.  Mirrors `TextureCacheStore::put` without
    /// the texture-map (memory backends never need it).
    pub fn insert(&mut self, key: TextureCacheKey, handles: Arc<GeneratedHandles>) {
        if let Ok(mut store) = self.inner.lock() {
            store.put(key, handles, false, None);
        }
    }
}

/// In-memory cache with bounded capacity and FIFO eviction.
///
/// Eviction is FIFO on insertion order — simpler than full LRU and adequate
/// for the typical access pattern (palettes loaded in bulk, hits clustered
/// around hot configs).  When the cap is reached the oldest entry is
/// dropped; re-inserting an existing key updates in place without evicting.
pub struct MemoryStore {
    max_entries: usize,
    entries: HashMap<TextureCacheKey, Arc<GeneratedHandles>>,
    insertion_order: VecDeque<TextureCacheKey>,
}

impl MemoryStore {
    /// Build a memory store bounded by `max_entries`.  Values below `1` are
    /// rounded up — a zero-sized cache is never useful.
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
/// The on-disk filename is `<DefaultHasher(key)>.bin` (Rust's
/// `std::hash::DefaultHasher`, currently SipHash-1-3, applied to the entire
/// [`TextureCacheKey`]).  Entries from older versions of this crate may be
/// unreadable when the on-disk format version (`FILE_FORMAT_VERSION` in the
/// blob header) changes; the loader skips entries that fail magic / version
/// checks, so stale files are inert rather than fatal.
pub struct FileStore {
    root: PathBuf,
}

impl FileStore {
    /// Open or create a file-backed store rooted at `root`.
    ///
    /// The directory is created if it does not exist; any I/O error is
    /// returned unchanged so callers can decide whether to fall back to an
    /// in-memory store or abort startup.
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

    /// Serialise `map` into the blob file for `key` (see the layout above).
    /// I/O failures are logged and swallowed — a broken cache write must not
    /// fail texture generation.
    fn write_blob(&self, key: &TextureCacheKey, map: &TextureMap, is_card: bool) {
        let path = self.path_for(key);
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
            bevy::log::warn!("FileStore write failed for {}: {e}", path.display());
        }
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
            // No raw pixels to persist.  The plugin flow persists via
            // `put_pixels` *before* the upload consumes the map, so a
            // handles-only `put` (e.g. `TextureCache::insert`) has nothing
            // left to write — handles cannot be serialised to disk.
            return;
        };
        self.write_blob(&key, map, is_card);
    }

    fn put_pixels(&mut self, key: &TextureCacheKey, map: &TextureMap, is_card: bool) {
        self.write_blob(key, map, is_card);
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

    fn tiny_map(w: u32, h: u32) -> TextureMap {
        let n = (w * h * 4) as usize;
        TextureMap {
            albedo: vec![10u8; n],
            normal: vec![128u8; n],
            roughness: vec![200u8; n],
            width: w,
            height: h,
        }
    }

    /// Unique per-test scratch directory under the system temp dir.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("bst-cache-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn file_store_round_trips_pixels_via_put_pixels() {
        let dir = scratch_dir("roundtrip");
        let mut store = FileStore::new(dir.clone()).expect("create store dir");
        let mut images = Assets::<Image>::default();

        let k = key("Bark", 7);
        assert!(store.get(&k, &mut images).is_none(), "cold store must miss");

        store.put_pixels(&k, &tiny_map(4, 4), false);
        let handles = store.get(&k, &mut images).expect("hit after put_pixels");
        let img = images.get(&handles.albedo).expect("albedo uploaded");
        assert_eq!(img.texture_descriptor.size.width, 4);
        assert_eq!(img.texture_descriptor.size.height, 4);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_preserves_card_mode_across_restart() {
        use bevy::image::{ImageAddressMode, ImageSampler};

        let dir = scratch_dir("cardmode");
        let k = key("Leaf", 9);
        {
            let mut store = FileStore::new(dir.clone()).expect("create store dir");
            store.put_pixels(&k, &tiny_map(2, 2), true);
        }
        // Fresh store over the same directory — simulates a process restart.
        let mut store = FileStore::new(dir.clone()).expect("reopen store dir");
        let mut images = Assets::<Image>::default();
        let handles = store.get(&k, &mut images).expect("hit after restart");
        let img = images.get(&handles.albedo).expect("albedo uploaded");
        match &img.sampler {
            ImageSampler::Descriptor(d) => {
                assert_eq!(
                    d.address_mode_u,
                    ImageAddressMode::ClampToEdge,
                    "is_card=true must restore a clamp-to-edge sampler"
                );
            }
            _ => panic!("expected a descriptor sampler"),
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
