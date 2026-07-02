//! A minimal, dependency-free safetensors reader.
//!
//! safetensors layout: `u64` little-endian header length, then that many bytes
//! of a JSON object mapping tensor name to `{ "dtype", "shape",
//! "data_offsets": [begin, end] }`, then the raw tensor bytes. `begin`/`end`
//! are offsets into the data region (which starts immediately after the
//! header). See <https://github.com/huggingface/safetensors>.
//!
//! Only the fields Vokra needs are read (dtype / shape / data_offsets); this is
//! the offline tool, so no external crate is pulled in.

use std::fmt;

use crate::json::{self, JsonValue};
use vokra_core::gguf::GgmlType;

/// Error while reading a safetensors buffer.
#[derive(Debug)]
pub(crate) enum SafetensorsError {
    /// The buffer was too short to hold the declared header.
    Truncated,
    /// The header JSON was malformed.
    Json(json::JsonError),
    /// A tensor entry was missing a required field or had a bad shape.
    BadEntry(String),
    /// A dtype outside the M0 range (F32 / F16) was encountered.
    UnsupportedDtype(String),
    /// A tensor's declared byte range fell outside the data region.
    OutOfBounds(String),
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "safetensors buffer truncated"),
            Self::Json(e) => write!(f, "safetensors header: {e}"),
            Self::BadEntry(m) => write!(f, "safetensors entry: {m}"),
            Self::UnsupportedDtype(d) => {
                write!(
                    f,
                    "unsupported safetensors dtype `{d}` (M0 accepts F32, F16)"
                )
            }
            Self::OutOfBounds(name) => write!(f, "tensor `{name}` data out of bounds"),
        }
    }
}

impl std::error::Error for SafetensorsError {}

/// One tensor described by a safetensors header.
#[derive(Debug, Clone)]
pub(crate) struct SafeTensor {
    /// Tensor name (the header key).
    pub(crate) name: String,
    /// Element type mapped into the GGUF dtype space.
    pub(crate) dtype: GgmlType,
    /// Shape, outermost dimension first (as stored by safetensors).
    pub(crate) shape: Vec<u64>,
    /// Absolute byte range `[start, end)` into the backing buffer.
    range: (usize, usize),
}

/// A parsed safetensors file: the backing buffer plus tensor descriptors.
pub(crate) struct SafeTensors {
    data: Vec<u8>,
    tensors: Vec<SafeTensor>,
}

impl SafeTensors {
    /// Parses a safetensors buffer, validating every tensor's byte range.
    pub(crate) fn parse(data: Vec<u8>) -> Result<Self, SafetensorsError> {
        if data.len() < 8 {
            return Err(SafetensorsError::Truncated);
        }
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let header_start: usize = 8;
        let header_end = header_start
            .checked_add(header_len)
            .ok_or(SafetensorsError::Truncated)?;
        if header_end > data.len() {
            return Err(SafetensorsError::Truncated);
        }
        let data_base = header_end;

        let header =
            json::parse(&data[header_start..header_end]).map_err(SafetensorsError::Json)?;
        let entries = header
            .as_object()
            .ok_or_else(|| SafetensorsError::BadEntry("header is not an object".to_owned()))?;

        let mut tensors = Vec::new();
        for (name, entry) in entries {
            // `__metadata__` is an optional free-form string map, not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = entry
                .get("dtype")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: missing dtype")))?;
            let dtype = map_dtype(dtype_str)?;

            let shape = entry
                .get("shape")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: missing shape")))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: bad shape dim")))
                })
                .collect::<Result<Vec<u64>, _>>()?;

            let offsets = entry
                .get("data_offsets")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| {
                    SafetensorsError::BadEntry(format!("{name}: missing data_offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(SafetensorsError::BadEntry(format!(
                    "{name}: data_offsets must have 2 elements"
                )));
            }
            let begin = offsets[0]
                .as_u64()
                .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: bad data_offsets")))?
                as usize;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: bad data_offsets")))?
                as usize;

            let abs_start = data_base
                .checked_add(begin)
                .ok_or_else(|| SafetensorsError::OutOfBounds(name.clone()))?;
            let abs_end = data_base
                .checked_add(end)
                .ok_or_else(|| SafetensorsError::OutOfBounds(name.clone()))?;
            if begin > end || abs_end > data.len() {
                return Err(SafetensorsError::OutOfBounds(name.clone()));
            }

            // Cross-check the byte span against shape * element size.
            let elems: u64 = shape.iter().product();
            let expected = elems * dtype.element_size() as u64;
            if (end - begin) as u64 != expected {
                return Err(SafetensorsError::BadEntry(format!(
                    "{name}: byte span {} does not match shape/dtype {expected}",
                    end - begin
                )));
            }

            tensors.push(SafeTensor {
                name: name.clone(),
                dtype,
                shape,
                range: (abs_start, abs_end),
            });
        }

        Ok(Self { data, tensors })
    }

    /// The parsed tensor descriptors, in header order.
    pub(crate) fn tensors(&self) -> &[SafeTensor] {
        &self.tensors
    }

    /// The raw little-endian payload bytes for a tensor.
    pub(crate) fn tensor_bytes(&self, t: &SafeTensor) -> &[u8] {
        &self.data[t.range.0..t.range.1]
    }
}

/// Maps a safetensors dtype string into the GGUF dtype space (M0: F32 / F16).
fn map_dtype(dtype: &str) -> Result<GgmlType, SafetensorsError> {
    match dtype {
        "F32" => Ok(GgmlType::F32),
        "F16" => Ok(GgmlType::F16),
        other => Err(SafetensorsError::UnsupportedDtype(other.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a synthetic 2-tensor safetensors buffer in memory.
    fn synthetic() -> Vec<u8> {
        // t_a: F32 shape [2] -> 8 bytes; t_b: F16 shape [3] -> 6 bytes.
        let payload_a: [f32; 2] = [1.0, 2.0];
        let payload_b: [u16; 3] = [0x3C00, 0x4000, 0x4200]; // 1.0, 2.0, 3.0 in f16
        let mut data_region = Vec::new();
        for v in payload_a {
            data_region.extend_from_slice(&v.to_le_bytes());
        }
        for v in payload_b {
            data_region.extend_from_slice(&v.to_le_bytes());
        }
        let header = r#"{"t_a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"t_b":{"dtype":"F16","shape":[3],"data_offsets":[8,14]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data_region);
        out
    }

    #[test]
    fn enumerates_tensors_and_bytes() {
        let st = SafeTensors::parse(synthetic()).expect("parse");
        assert_eq!(st.tensors().len(), 2);

        let a = &st.tensors()[0];
        assert_eq!(a.name, "t_a");
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.shape, vec![2]);
        assert_eq!(
            st.tensor_bytes(a),
            1.0f32
                .to_le_bytes()
                .into_iter()
                .chain(2.0f32.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice()
        );

        let b = &st.tensors()[1];
        assert_eq!(b.dtype, GgmlType::F16);
        assert_eq!(st.tensor_bytes(b).len(), 6);
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let header = r#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8, 0u8]);
        assert!(matches!(
            SafeTensors::parse(out),
            Err(SafetensorsError::UnsupportedDtype(_))
        ));
    }

    #[test]
    fn rejects_out_of_bounds_offsets() {
        let header = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        // Only 4 bytes of data, but the entry claims 8.
        out.extend_from_slice(&[0u8; 4]);
        assert!(matches!(
            SafeTensors::parse(out),
            Err(SafetensorsError::OutOfBounds(_))
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(matches!(
            SafeTensors::parse(vec![0u8; 4]),
            Err(SafetensorsError::Truncated)
        ));
    }
}
