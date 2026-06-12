//! Structural fingerprinting for config types.
//!
//! [`TextureConfig::fingerprint`] needs a hash that identifies a config
//! *value* for cache keys.  Hashing the `Debug` string (the pre-0.6
//! approach) allocated a `String` per call and coupled cache identity to
//! formatting details.  Instead, [`hash_value`] drives a config's
//! `serde::Serialize` impl with a serializer that feeds every primitive
//! straight into a hasher — structural, allocation-free, and automatically
//! covering any field added to a config in the future.
//!
//! # Stability contract
//!
//! [`Fnv1a`] is a fixed, dependency-free algorithm and every primitive is
//! written as little-endian bytes, so fingerprints are stable across runs,
//! Rust versions, **and** platforms.  A fingerprint only changes when the
//! config's field values, field set, or serde representation change.
//! Generator-internal changes (noise weights, blend factors) do **not**
//! roll fingerprints — bump
//! [`TextureCache::manifest_version`](crate::cache::TextureCache::manifest_version)
//! for those.
//!
//! [`TextureConfig::fingerprint`]: crate::material::TextureConfig::fingerprint

use std::fmt::Display;
use std::hash::Hasher;

use serde::ser::{self, Serialize};

/// FNV-1a, 64-bit.  Chosen over `DefaultHasher` because the std SipHash
/// implementation is documented as unstable across Rust releases.
pub(crate) struct Fnv1a(u64);

impl Fnv1a {
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv1a {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Feed `value`'s serde representation into `hasher`.
///
/// Never fails for the plain-old-data config types in this crate; the
/// serializer below has no fallible paths of its own.
pub(crate) fn hash_value<T: Serialize, H: Hasher>(value: &T, hasher: &mut H) {
    value
        .serialize(HashSerializer { hasher })
        .expect("hashing serializer is infallible for plain-old-data configs");
}

/// Error type required by the `Serializer` trait; never constructed by the
/// serializer itself (only reachable through a `Serialize` impl calling
/// `Error::custom`, which the derive-generated config impls never do).
#[derive(Debug)]
pub(crate) struct HashError;

impl Display for HashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("custom serialisation error during fingerprint hashing")
    }
}

impl std::error::Error for HashError {}

impl ser::Error for HashError {
    fn custom<T: Display>(_msg: T) -> Self {
        HashError
    }
}

/// Serializer that writes every primitive into the wrapped hasher as
/// little-endian bytes.  Lengths and enum variant indices are mixed in so
/// differently-shaped values cannot collide by concatenation.
struct HashSerializer<'a, H: Hasher> {
    hasher: &'a mut H,
}

impl<H: Hasher> HashSerializer<'_, H> {
    #[inline]
    fn reborrow(&mut self) -> HashSerializer<'_, H> {
        HashSerializer {
            hasher: self.hasher,
        }
    }

    #[inline]
    fn put(&mut self, bytes: &[u8]) {
        self.hasher.write(bytes);
    }
}

macro_rules! hash_le {
    ($($method:ident: $ty:ty),* $(,)?) => {
        $(
            fn $method(mut self, v: $ty) -> Result<(), HashError> {
                self.put(&v.to_le_bytes());
                Ok(())
            }
        )*
    };
}

impl<'a, H: Hasher> ser::Serializer for HashSerializer<'a, H> {
    type Ok = ();
    type Error = HashError;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    hash_le! {
        serialize_i8: i8,
        serialize_i16: i16,
        serialize_i32: i32,
        serialize_i64: i64,
        serialize_u8: u8,
        serialize_u16: u16,
        serialize_u32: u32,
        serialize_u64: u64,
    }

    fn serialize_bool(mut self, v: bool) -> Result<(), HashError> {
        self.put(&[v as u8]);
        Ok(())
    }

    fn serialize_f32(mut self, v: f32) -> Result<(), HashError> {
        self.put(&v.to_bits().to_le_bytes());
        Ok(())
    }

    fn serialize_f64(mut self, v: f64) -> Result<(), HashError> {
        self.put(&v.to_bits().to_le_bytes());
        Ok(())
    }

    fn serialize_char(mut self, v: char) -> Result<(), HashError> {
        self.put(&(v as u32).to_le_bytes());
        Ok(())
    }

    fn serialize_str(mut self, v: &str) -> Result<(), HashError> {
        self.put(&(v.len() as u64).to_le_bytes());
        self.put(v.as_bytes());
        Ok(())
    }

    fn serialize_bytes(mut self, v: &[u8]) -> Result<(), HashError> {
        self.put(&(v.len() as u64).to_le_bytes());
        self.put(v);
        Ok(())
    }

    fn serialize_none(mut self) -> Result<(), HashError> {
        self.put(&[0]);
        Ok(())
    }

    fn serialize_some<T: Serialize + ?Sized>(mut self, value: &T) -> Result<(), HashError> {
        self.put(&[1]);
        value.serialize(self)
    }

    fn serialize_unit(mut self) -> Result<(), HashError> {
        self.put(&[0]);
        Ok(())
    }

    fn serialize_unit_struct(self, name: &'static str) -> Result<(), HashError> {
        self.serialize_str(name)
    }

    fn serialize_unit_variant(
        mut self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
    ) -> Result<(), HashError> {
        self.put(&variant_index.to_le_bytes());
        Ok(())
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), HashError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        mut self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<(), HashError> {
        self.put(&variant_index.to_le_bytes());
        value.serialize(self)
    }

    fn serialize_seq(mut self, len: Option<usize>) -> Result<Self, HashError> {
        self.put(&(len.unwrap_or(0) as u64).to_le_bytes());
        Ok(self)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self, HashError> {
        Ok(self)
    }

    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self, HashError> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        mut self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self, HashError> {
        self.put(&variant_index.to_le_bytes());
        Ok(self)
    }

    fn serialize_map(mut self, len: Option<usize>) -> Result<Self, HashError> {
        self.put(&(len.unwrap_or(0) as u64).to_le_bytes());
        Ok(self)
    }

    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self, HashError> {
        Ok(self)
    }

    fn serialize_struct_variant(
        mut self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self, HashError> {
        self.put(&variant_index.to_le_bytes());
        Ok(self)
    }
}

impl<H: Hasher> ser::SerializeSeq for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeTuple for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeTupleStruct for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeTupleVariant for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeMap for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), HashError> {
        key.serialize(self.reborrow())
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeStruct for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

impl<H: Hasher> ser::SerializeStructVariant for HashSerializer<'_, H> {
    type Ok = ();
    type Error = HashError;

    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), HashError> {
        value.serialize(self.reborrow())
    }

    fn end(self) -> Result<(), HashError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_of<T: Serialize>(value: &T) -> u64 {
        let mut h = Fnv1a::new();
        hash_value(value, &mut h);
        h.finish()
    }

    #[test]
    fn equal_values_hash_equal() {
        #[derive(serde::Serialize)]
        struct S {
            a: f64,
            b: u32,
            c: [f32; 3],
        }
        let x = S {
            a: 1.5,
            b: 7,
            c: [0.1, 0.2, 0.3],
        };
        let y = S {
            a: 1.5,
            b: 7,
            c: [0.1, 0.2, 0.3],
        };
        assert_eq!(hash_of(&x), hash_of(&y));
    }

    #[test]
    fn field_bit_changes_roll_the_hash() {
        // -0.0 and 0.0 compare equal as floats but are different bit
        // patterns — the structural hash must distinguish them, like any
        // other bit-level change.
        assert_ne!(hash_of(&0.0_f64), hash_of(&-0.0_f64));
        assert_ne!(hash_of(&1.0_f64), hash_of(&1.0000000000000002_f64));
    }

    #[test]
    fn enum_variants_are_distinguished() {
        #[derive(serde::Serialize)]
        enum E {
            A,
            B,
        }
        assert_ne!(hash_of(&E::A), hash_of(&E::B));
    }
}
