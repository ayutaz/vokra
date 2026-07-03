//! A minimal, dependency-free safetensors reader for **runtime** direct-load.
//!
//! safetensors (a Hugging Face on-disk format, Apache-2.0) layout: a `u64`
//! little-endian header length, then that many bytes of a JSON object mapping
//! tensor name to `{ "dtype", "shape", "data_offsets": [begin, end] }`, then the
//! raw tensor bytes. `begin` / `end` are offsets into the data region (which
//! starts immediately after the header). See
//! <https://github.com/huggingface/safetensors>.
//!
//! # Runtime weight provider (FR-LD-04 / IF-06)
//!
//! [`SafetensorsFile`] mirrors [`GgufFile`](crate::gguf::GgufFile): parse a
//! buffer, then lend tensor payloads as zero-copy `&[u8]` slices or decode them
//! to owned `f32`. It is a **weight provider only** — safetensors carries no
//! `vokra.*` metadata, so hyperparameters / frontend spec must come from a GGUF
//! sidecar or a shape-derivation helper.
//!
//! # No `unsafe`, no mmap
//!
//! `vokra-core` is `unsafe_code = "deny"`, so — exactly like the GGUF reader —
//! this reads the whole file into an owned buffer and lends slices into it. The
//! access API is genuinely copy-free, but true *lazy* `mmap` (FR-LD-01 /
//! NFR-PF-11) needs an `unsafe`-allowed home and stays a documented follow-up.
//! Only the dense float dtypes `F32` / `F16` are accepted (safetensors stores
//! dense tensors; `BF16` → `f32` is a trivial future extension).

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::gguf::GgmlType;
use crate::gguf::quant;
use crate::json::{self, JsonValue};

/// Error while reading a safetensors buffer.
#[derive(Debug)]
#[non_exhaustive]
pub enum SafetensorsError {
    /// The buffer was too short to hold the declared header.
    Truncated,
    /// The header JSON was malformed.
    Json(json::JsonError),
    /// A tensor entry was missing a required field or had a bad shape.
    BadEntry(String),
    /// A dtype outside the accepted range (`F32` / `F16`) was encountered.
    UnsupportedDtype(String),
    /// A tensor's declared byte range fell outside the data region.
    OutOfBounds(String),
    /// An underlying I/O error (only from [`SafetensorsFile::open`]).
    Io(std::io::Error),
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
                    "unsupported safetensors dtype `{d}` (accepted: F32, F16)"
                )
            }
            Self::OutOfBounds(name) => write!(f, "tensor `{name}` data out of bounds"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for SafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<SafetensorsError> for crate::VokraError {
    fn from(e: SafetensorsError) -> Self {
        match e {
            // Preserve the I/O source chain at the public boundary.
            SafetensorsError::Io(io) => crate::VokraError::Io(io),
            other => crate::VokraError::ModelLoad(other.to_string()),
        }
    }
}

/// Descriptor for one tensor in a safetensors file (mirrors
/// [`GgufTensorInfo`](crate::gguf::GgufTensorInfo)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    /// Tensor name (the header key).
    pub name: String,
    /// Element type mapped into the GGUF dtype space (`F32` / `F16`).
    pub dtype: GgmlType,
    /// Shape, outermost dimension first (as stored by safetensors).
    pub shape: Vec<u64>,
    /// Absolute byte range `[start, end)` into the backing buffer.
    range: (usize, usize),
}

impl SafeTensorInfo {
    /// Total number of elements (product of all dimensions).
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// A parsed safetensors file: the backing buffer plus tensor descriptors.
pub struct SafetensorsFile {
    data: Vec<u8>,
    tensors: Vec<SafeTensorInfo>,
    index: HashMap<String, usize>,
}

impl fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately omit the (potentially large) backing buffer.
        f.debug_struct("SafetensorsFile")
            .field("tensors", &self.tensors.len())
            .field("file_len", &self.data.len())
            .finish()
    }
}

impl SafetensorsFile {
    /// Opens and parses a safetensors file from disk.
    ///
    /// Reads the whole file into memory (see the module docs on the zero-copy
    /// strategy). Returns [`SafetensorsError::Io`] on I/O failure or a parse
    /// error variant for malformed content.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let data = std::fs::read(path).map_err(SafetensorsError::Io)?;
        Self::parse(data)
    }

    /// Parses a safetensors buffer, validating every tensor's byte range.
    pub fn parse(data: Vec<u8>) -> Result<Self, SafetensorsError> {
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

            // Cross-check the byte span against shape * element size. safetensors
            // is always dense (block_size 1), so `elements * type_size` is exact.
            let elems: u64 = shape.iter().product();
            let expected = elems * dtype.type_size() as u64;
            if (end - begin) as u64 != expected {
                return Err(SafetensorsError::BadEntry(format!(
                    "{name}: byte span {} does not match shape/dtype {expected}",
                    end - begin
                )));
            }

            tensors.push(SafeTensorInfo {
                name: name.clone(),
                dtype,
                shape,
                range: (abs_start, abs_end),
            });
        }

        let mut index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            index.insert(t.name.clone(), i);
        }

        Ok(Self {
            data,
            tensors,
            index,
        })
    }

    /// All tensor descriptors, in header order.
    pub fn tensors(&self) -> &[SafeTensorInfo] {
        &self.tensors
    }

    /// Looks up a tensor descriptor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Lends a known tensor's raw little-endian payload as a zero-copy slice.
    pub fn tensor_bytes(&self, t: &SafeTensorInfo) -> &[u8] {
        &self.data[t.range.0..t.range.1]
    }

    /// Lends a tensor's raw payload by name, or `None` if it is absent.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        Some(self.tensor_bytes(info))
    }

    /// Decodes a tensor's payload into owned `f32` (mirrors
    /// [`GgufFile::tensor_f32`](crate::gguf::GgufFile::tensor_f32)).
    ///
    /// Resolves the `F32` / `F16` payload through the shared
    /// [`quant::dequantize`] path. Returns [`SafetensorsError::BadEntry`] if no
    /// tensor has that name.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, SafetensorsError> {
        let info = self
            .tensor_info(name)
            .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: not found")))?;
        let n = info.element_count() as usize;
        quant::dequantize(info.dtype, self.tensor_bytes(info), n)
            .map_err(|e| SafetensorsError::BadEntry(format!("{name}: {e}")))
    }
}

/// Maps a safetensors dtype string into the GGUF dtype space (`F32` / `F16`).
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
        let st = SafetensorsFile::parse(synthetic()).expect("parse");
        assert_eq!(st.tensors().len(), 2);

        let a = st.tensor_info("t_a").expect("present");
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.shape, vec![2]);
        assert_eq!(a.element_count(), 2);
        assert_eq!(
            st.tensor_bytes(a),
            1.0f32
                .to_le_bytes()
                .into_iter()
                .chain(2.0f32.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice()
        );

        let b = st.tensor_info("t_b").unwrap();
        assert_eq!(b.dtype, GgmlType::F16);
        assert_eq!(st.tensor_data("t_b").unwrap().len(), 6);
    }

    #[test]
    fn tensor_f32_decodes_f32_and_f16_through_shared_dequant() {
        let st = SafetensorsFile::parse(synthetic()).unwrap();
        assert_eq!(st.tensor_f32("t_a").unwrap(), vec![1.0, 2.0]);
        // f16 0x3C00/0x4000/0x4200 = 1.0/2.0/3.0.
        assert_eq!(st.tensor_f32("t_b").unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn tensor_f32_missing_name_is_bad_entry() {
        let st = SafetensorsFile::parse(synthetic()).unwrap();
        assert!(matches!(
            st.tensor_f32("nope"),
            Err(SafetensorsError::BadEntry(_))
        ));
    }

    #[test]
    fn rejects_unsupported_dtype() {
        let header = r#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8, 0u8]);
        assert!(matches!(
            SafetensorsFile::parse(out),
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
            SafetensorsFile::parse(out),
            Err(SafetensorsError::OutOfBounds(_))
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(matches!(
            SafetensorsFile::parse(vec![0u8; 4]),
            Err(SafetensorsError::Truncated)
        ));
    }

    #[test]
    fn safetensors_error_maps_to_vokra_error_at_boundary() {
        let mapped = crate::VokraError::from(SafetensorsError::Truncated);
        assert!(matches!(mapped, crate::VokraError::ModelLoad(_)));
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "x");
        assert!(matches!(
            crate::VokraError::from(SafetensorsError::Io(io)),
            crate::VokraError::Io(_)
        ));
    }
}
