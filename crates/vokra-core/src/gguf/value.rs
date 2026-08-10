//! GGUF metadata value types and values.
//!
//! Mirrors the `gguf_metadata_value_type` enum from the upstream GGUF
//! specification (ggml-org/ggml `docs/gguf.md`). The integer tags are part of
//! the on-disk format and MUST match the spec exactly:
//!
//! | tag | type    | tag | type    |
//! |-----|---------|-----|---------|
//! | 0   | UINT8   | 7   | BOOL    |
//! | 1   | INT8    | 8   | STRING  |
//! | 2   | UINT16  | 9   | ARRAY   |
//! | 3   | INT16   | 10  | UINT64  |
//! | 4   | UINT32  | 11  | INT64   |
//! | 5   | INT32   | 12  | FLOAT64 |
//! | 6   | FLOAT32 |     |         |
//!
//! Source: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>.

use super::GgufError;
// M5-03-T05: `String` / `Vec` are `alloc` types (core-clean, no `std::`); the
// no_std subset imports them (inert under std, where they are in the prelude).
#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// Discriminant of a GGUF metadata value, matching the on-disk `uint32` tag.
///
/// The numeric values are load-bearing: they are written to and read from the
/// file verbatim, so they must never be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufValueType {
    /// Unsigned 8-bit integer (tag `0`).
    U8 = 0,
    /// Signed 8-bit integer (tag `1`).
    I8 = 1,
    /// Unsigned 16-bit integer (tag `2`).
    U16 = 2,
    /// Signed 16-bit integer (tag `3`).
    I16 = 3,
    /// Unsigned 32-bit integer (tag `4`).
    U32 = 4,
    /// Signed 32-bit integer (tag `5`).
    I32 = 5,
    /// IEEE-754 32-bit float (tag `6`).
    F32 = 6,
    /// Boolean stored as a single byte (tag `7`).
    Bool = 7,
    /// UTF-8 string, `u64` length-prefixed (tag `8`).
    String = 8,
    /// Homogeneous array: element type, `u64` length, then elements (tag `9`).
    Array = 9,
    /// Unsigned 64-bit integer (tag `10`).
    U64 = 10,
    /// Signed 64-bit integer (tag `11`).
    I64 = 11,
    /// IEEE-754 64-bit float (tag `12`).
    F64 = 12,
}

impl GgufValueType {
    /// Converts an on-disk `uint32` tag to a [`GgufValueType`].
    ///
    /// Returns [`GgufError::UnsupportedValueType`] for any tag outside `0..=12`.
    pub fn from_tag(tag: u32) -> Result<Self, GgufError> {
        Ok(match tag {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::UnsupportedValueType(other)),
        })
    }

    /// Returns the on-disk `uint32` tag for this value type.
    pub fn tag(self) -> u32 {
        self as u32
    }
}

/// A homogeneous GGUF array: an element type plus its elements.
///
/// GGUF arrays are typed and may nest (an element type of
/// [`GgufValueType::Array`] yields nested [`GgufArray`] values).
#[derive(Debug, Clone, PartialEq)]
pub struct GgufArray {
    /// Declared element type of every entry in [`GgufArray::values`].
    pub element_type: GgufValueType,
    /// The array elements, each matching `element_type`.
    pub values: Vec<GgufMetadataValue>,
}

/// A single decoded GGUF metadata value.
///
/// Numeric text parsing is never used here: values are read as fixed-width
/// little-endian binary, so the locale-dependent `strtod` trap (NFR-RL-01)
/// does not apply to this path.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// IEEE-754 32-bit float.
    F32(f32),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous, possibly nested array.
    Array(GgufArray),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// IEEE-754 64-bit float.
    F64(f64),
}

impl GgufMetadataValue {
    /// Returns the [`GgufValueType`] discriminant of this value.
    pub fn value_type(&self) -> GgufValueType {
        match self {
            Self::U8(_) => GgufValueType::U8,
            Self::I8(_) => GgufValueType::I8,
            Self::U16(_) => GgufValueType::U16,
            Self::I16(_) => GgufValueType::I16,
            Self::U32(_) => GgufValueType::U32,
            Self::I32(_) => GgufValueType::I32,
            Self::F32(_) => GgufValueType::F32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::U64(_) => GgufValueType::U64,
            Self::I64(_) => GgufValueType::I64,
            Self::F64(_) => GgufValueType::F64,
        }
    }

    /// Returns the string payload, or `None` for any non-string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the boolean payload, or `None` for any non-boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the array payload, or `None` for any non-array value.
    pub fn as_array(&self) -> Option<&GgufArray> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Returns any unsigned-integer payload widened to `u64`.
    ///
    /// Accepts [`Self::U8`], [`Self::U16`], [`Self::U32`] and [`Self::U64`];
    /// returns `None` for every other variant (including signed integers).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(u64::from(*v)),
            Self::U16(v) => Some(u64::from(*v)),
            Self::U32(v) => Some(u64::from(*v)),
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns any float payload widened to `f64`.
    ///
    /// Accepts [`Self::F32`] and [`Self::F64`]; returns `None` otherwise.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(f64::from(*v)),
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }
}

// Audit 2026-08-10 (Rank 13, test-coverage-audit workflow): value.rs public
// API had zero direct tests — coverage was incidental via reader.rs /
// writer.rs, so an accidental widening in as_str/bool/array or a broken
// tag() ↔ from_tag() pair could reach production without a red build. The
// tag values are load-bearing on-disk contract (GGUF spec, §upstream).
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::{string::String, vec, vec::Vec};

    #[test]
    fn from_tag_roundtrips_every_defined_discriminant() {
        // Tags 0..=12 are spec-defined and must round-trip. This pins the
        // wire contract: renaming a variant without updating the numeric
        // repr would fail here immediately.
        for tag in 0u32..=12 {
            let v = GgufValueType::from_tag(tag)
                .unwrap_or_else(|e| panic!("spec-defined tag {tag} rejected: {e:?}"));
            assert_eq!(v.tag(), tag, "round-trip mismatch at tag {tag}");
        }
    }

    #[test]
    fn from_tag_rejects_first_unassigned_tag() {
        // Tag 13 is the smallest unassigned tag. If upstream ever assigns
        // it, this test fails and forces a conscious extension of the enum.
        let err = GgufValueType::from_tag(13).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedValueType(13)));
    }

    #[test]
    fn from_tag_rejects_max_u32() {
        let err = GgufValueType::from_tag(u32::MAX).unwrap_err();
        assert!(matches!(err, GgufError::UnsupportedValueType(t) if t == u32::MAX));
    }

    #[test]
    fn value_type_matches_wrapped_variant_for_all_scalars() {
        // Every variant must report its own discriminant — a wire-format
        // regression if any pair diverges. Array covered separately below
        // to avoid nested-array boilerplate in this list.
        let cases: [(GgufMetadataValue, GgufValueType); 12] = [
            (GgufMetadataValue::U8(0), GgufValueType::U8),
            (GgufMetadataValue::I8(0), GgufValueType::I8),
            (GgufMetadataValue::U16(0), GgufValueType::U16),
            (GgufMetadataValue::I16(0), GgufValueType::I16),
            (GgufMetadataValue::U32(0), GgufValueType::U32),
            (GgufMetadataValue::I32(0), GgufValueType::I32),
            (GgufMetadataValue::F32(0.0), GgufValueType::F32),
            (GgufMetadataValue::Bool(false), GgufValueType::Bool),
            (
                GgufMetadataValue::String(String::new()),
                GgufValueType::String,
            ),
            (GgufMetadataValue::U64(0), GgufValueType::U64),
            (GgufMetadataValue::I64(0), GgufValueType::I64),
            (GgufMetadataValue::F64(0.0), GgufValueType::F64),
        ];
        for (v, expected) in &cases {
            assert_eq!(v.value_type(), *expected);
        }
    }

    #[test]
    fn value_type_matches_wrapped_variant_for_array() {
        let arr = GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: Vec::new(),
        });
        assert_eq!(arr.value_type(), GgufValueType::Array);
    }

    #[test]
    fn as_str_only_matches_string_variant() {
        assert_eq!(
            GgufMetadataValue::String(String::from("hi")).as_str(),
            Some("hi"),
        );
        // Non-string variants MUST return None. A silent widening would
        // let a U32-tagged field satisfy a String-typed reader — silent
        // corruption of any caller in vokra-models.
        assert_eq!(GgufMetadataValue::U8(0).as_str(), None);
        assert_eq!(GgufMetadataValue::Bool(false).as_str(), None);
        assert_eq!(GgufMetadataValue::F32(0.0).as_str(), None);
    }

    #[test]
    fn as_bool_only_matches_bool_variant() {
        assert_eq!(GgufMetadataValue::Bool(true).as_bool(), Some(true));
        assert_eq!(GgufMetadataValue::Bool(false).as_bool(), Some(false));
        assert_eq!(GgufMetadataValue::U8(1).as_bool(), None);
        assert_eq!(
            GgufMetadataValue::String(String::from("true")).as_bool(),
            None
        );
    }

    #[test]
    fn as_array_only_matches_array_variant() {
        let arr = GgufArray {
            element_type: GgufValueType::U8,
            values: vec![GgufMetadataValue::U8(1), GgufMetadataValue::U8(2)],
        };
        let wrapped = GgufMetadataValue::Array(arr);
        let got = wrapped.as_array().expect("array variant should match");
        assert_eq!(got.element_type, GgufValueType::U8);
        assert_eq!(got.values.len(), 2);
        assert_eq!(GgufMetadataValue::U8(0).as_array(), None);
    }

    #[test]
    fn as_u64_widens_unsigned_variants_only() {
        assert_eq!(GgufMetadataValue::U8(255).as_u64(), Some(255));
        assert_eq!(GgufMetadataValue::U16(65_535).as_u64(), Some(65_535));
        assert_eq!(
            GgufMetadataValue::U32(u32::MAX).as_u64(),
            Some(u64::from(u32::MAX)),
        );
        assert_eq!(GgufMetadataValue::U64(u64::MAX).as_u64(), Some(u64::MAX));
        // Signed variants MUST NOT widen — a -1 silently promoted to
        // u64::MAX corrupts every caller inspecting a signed-declared
        // field.
        assert_eq!(GgufMetadataValue::I8(-1).as_u64(), None);
        assert_eq!(GgufMetadataValue::I64(-1).as_u64(), None);
        assert_eq!(GgufMetadataValue::F64(0.0).as_u64(), None);
        assert_eq!(GgufMetadataValue::String(String::from("0")).as_u64(), None);
    }

    #[test]
    fn as_f64_widens_float_variants_only() {
        assert_eq!(GgufMetadataValue::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(
            GgufMetadataValue::F64(core::f64::consts::PI).as_f64(),
            Some(core::f64::consts::PI),
        );
        assert_eq!(GgufMetadataValue::U32(1).as_f64(), None);
        assert_eq!(GgufMetadataValue::I64(-1).as_f64(), None);
    }
}
