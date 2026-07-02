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
