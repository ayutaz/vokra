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
//! Only the dense float dtypes `F32` / `F16` / `BF16` are accepted
//! (safetensors stores dense tensors; `BF16` graduated from "future
//! extension" to supported in M4-06 — the moshiko and raw Voxtral releases
//! are all-BF16).

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
    /// A dtype outside the accepted range (`F32` / `F16` / `BF16`) was
    /// encountered.
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
                    "unsupported safetensors dtype `{d}` (accepted: F32, F16, BF16)"
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
        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let header_end = header_len
            .checked_add(8)
            .ok_or(SafetensorsError::Truncated)?;
        if header_end > data.len() as u64 {
            return Err(SafetensorsError::Truncated);
        }
        let data_base = header_end;
        let tensors =
            parse_header_entries(&data[8..header_end as usize], data_base, data.len() as u64)?;
        let index = build_index(&tensors);
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

/// A safetensors file opened for **windowed** (bounded-memory) tensor
/// reads: only the JSON header is parsed into memory; payloads are read
/// on demand with `seek + read_exact` into a caller-supplied buffer.
///
/// This is the converter-side complement to [`SafetensorsFile`]: a
/// multi-GB checkpoint (the 14 GiB `kyutai/moshiko-pytorch-bf16` single
/// file) streams tensor-by-tensor instead of being materialized whole,
/// holding at most one tensor payload at a time. The descriptor surface
/// ([`Self::tensors`] / [`Self::tensor_info`]) is identical to the
/// in-memory reader, and both run the **same** header parser, so shape
/// derivation code works over either unchanged.
pub struct SafetensorsFileReader {
    file: std::fs::File,
    tensors: Vec<SafeTensorInfo>,
    index: HashMap<String, usize>,
}

impl fmt::Debug for SafetensorsFileReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SafetensorsFileReader")
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl SafetensorsFileReader {
    /// Opens `path` and parses **only** the header (the 8-byte length
    /// prefix and the JSON table). Every tensor's byte range is validated
    /// against the real file length up front, so later reads cannot run
    /// past EOF.
    ///
    /// # Errors
    ///
    /// [`SafetensorsError::Io`] on open/read failure; the same parse
    /// error variants as [`SafetensorsFile::parse`] for malformed content.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(SafetensorsError::Io)?;
        let file_len = file.metadata().map_err(SafetensorsError::Io)?.len();
        if file_len < 8 {
            return Err(SafetensorsError::Truncated);
        }
        let mut len8 = [0u8; 8];
        file.read_exact(&mut len8).map_err(SafetensorsError::Io)?;
        let header_len = u64::from_le_bytes(len8);
        let header_end = header_len
            .checked_add(8)
            .ok_or(SafetensorsError::Truncated)?;
        if header_end > file_len {
            return Err(SafetensorsError::Truncated);
        }
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| SafetensorsError::Truncated)?;
        let mut header = vec![0u8; header_len_usize];
        file.read_exact(&mut header).map_err(SafetensorsError::Io)?;

        let tensors = parse_header_entries(&header, header_end, file_len)?;
        let index = build_index(&tensors);
        Ok(Self {
            file,
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

    /// Reads the named tensor's raw little-endian payload into `buf`
    /// (resized to the exact payload length; capacity is reused across
    /// calls, which is what keeps a whole-checkpoint streaming pass at
    /// one-tensor peak memory).
    ///
    /// # Errors
    ///
    /// [`SafetensorsError::BadEntry`] if no tensor has that name;
    /// [`SafetensorsError::Io`] on seek/read failure.
    pub fn read_tensor_into(
        &mut self,
        name: &str,
        buf: &mut Vec<u8>,
    ) -> Result<(), SafetensorsError> {
        use std::io::{Read, Seek, SeekFrom};

        let Some(&i) = self.index.get(name) else {
            return Err(SafetensorsError::BadEntry(format!("{name}: not found")));
        };
        let (start, end) = self.tensors[i].range;
        buf.clear();
        buf.resize(end - start, 0);
        self.file
            .seek(SeekFrom::Start(start as u64))
            .map_err(SafetensorsError::Io)?;
        self.file.read_exact(buf).map_err(SafetensorsError::Io)?;
        Ok(())
    }
}

/// Parses the safetensors JSON header into validated descriptors.
///
/// `data_base` is the absolute offset where the data region starts (8 +
/// header length) and `total_len` the total byte length the ranges must
/// fit inside — the buffer length for [`SafetensorsFile::parse`], the
/// file length for [`SafetensorsFileReader::open`]. Both readers run this
/// **same** function, so windowed and in-memory descriptor sets can never
/// disagree.
fn parse_header_entries(
    header_bytes: &[u8],
    data_base: u64,
    total_len: u64,
) -> Result<Vec<SafeTensorInfo>, SafetensorsError> {
    let header = json::parse(header_bytes).map_err(SafetensorsError::Json)?;
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
            .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(SafetensorsError::BadEntry(format!(
                "{name}: data_offsets must have 2 elements"
            )));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: bad data_offsets")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| SafetensorsError::BadEntry(format!("{name}: bad data_offsets")))?;

        let abs_start = data_base
            .checked_add(begin)
            .ok_or_else(|| SafetensorsError::OutOfBounds(name.clone()))?;
        let abs_end = data_base
            .checked_add(end)
            .ok_or_else(|| SafetensorsError::OutOfBounds(name.clone()))?;
        if begin > end || abs_end > total_len {
            return Err(SafetensorsError::OutOfBounds(name.clone()));
        }
        let abs_start =
            usize::try_from(abs_start).map_err(|_| SafetensorsError::OutOfBounds(name.clone()))?;
        let abs_end =
            usize::try_from(abs_end).map_err(|_| SafetensorsError::OutOfBounds(name.clone()))?;

        // Cross-check the byte span against shape * element size. safetensors
        // is always dense (block_size 1), so `elements * type_size` is exact.
        let elems: u64 = shape.iter().product();
        let expected = elems * dtype.type_size() as u64;
        if (end - begin) != expected {
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
    Ok(tensors)
}

/// Name → position index over a parsed descriptor list.
fn build_index(tensors: &[SafeTensorInfo]) -> HashMap<String, usize> {
    let mut index = HashMap::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        index.insert(t.name.clone(), i);
    }
    index
}

/// Maps a safetensors dtype string into the GGUF dtype space
/// (`F32` / `F16` / `BF16` — the latter added for the all-BF16
/// `kyutai/moshiko-pytorch-bf16` checkpoint, M4-06).
fn map_dtype(dtype: &str) -> Result<GgmlType, SafetensorsError> {
    match dtype {
        "F32" => Ok(GgmlType::F32),
        "F16" => Ok(GgmlType::F16),
        "BF16" => Ok(GgmlType::BF16),
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
        // BF16 graduated to a supported dtype in M4-06 (the moshiko
        // checkpoint is all-BF16), so the negative example is now F64 —
        // still deliberately unsupported (no dense-f64 weights exist in
        // any Vokra model family).
        let header = r#"{"x":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 8]);
        assert!(matches!(
            SafetensorsFile::parse(out),
            Err(SafetensorsError::UnsupportedDtype(_))
        ));
    }

    #[test]
    fn bf16_decodes_exactly_through_the_shared_dequant() {
        // BF16 = the top 16 bits of the f32 pattern — decode is exact
        // (M4-06; the converter relies on this to write F32 losslessly).
        let values: [f32; 3] = [1.0, -2.5, 0.15625];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let header = r#"{"x":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&bf16);
        let st = SafetensorsFile::parse(out).unwrap();
        assert_eq!(st.tensor_f32("x").unwrap(), values);
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

    // --- SafetensorsFileReader (windowed / bounded-memory) ----------------

    /// A unique temp path per test (shared pid across the test binary).
    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-safetensors-reader-{tag}-{}.safetensors",
            std::process::id()
        ));
        p
    }

    #[test]
    fn windowed_reader_matches_in_memory_parse() {
        // Same header parser, same descriptors, same payload bytes — the
        // windowed reader must be indistinguishable from the owned path.
        let blob = synthetic();
        let path = tmp_path("identity");
        std::fs::write(&path, &blob).unwrap();

        let owned = SafetensorsFile::parse(blob).unwrap();
        let mut windowed = SafetensorsFileReader::open(&path).unwrap();

        assert_eq!(owned.tensors(), windowed.tensors());
        let names: Vec<String> = owned.tensors().iter().map(|t| t.name.clone()).collect();
        let mut buf = Vec::new();
        for name in &names {
            windowed.read_tensor_into(name, &mut buf).unwrap();
            assert_eq!(
                buf.as_slice(),
                owned.tensor_data(name).unwrap(),
                "payload bytes for `{name}`"
            );
        }
        // Buffer capacity is reused (bounded-memory contract): re-reading
        // a smaller tensor shrinks len, not correctness.
        let smallest = names
            .iter()
            .min_by_key(|n| owned.tensor_data(n).unwrap().len())
            .unwrap();
        windowed.read_tensor_into(smallest, &mut buf).unwrap();
        assert_eq!(buf.as_slice(), owned.tensor_data(smallest).unwrap());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn windowed_reader_rejects_missing_name_and_bad_files() {
        let blob = synthetic();
        let path = tmp_path("errors");
        std::fs::write(&path, &blob).unwrap();
        let mut windowed = SafetensorsFileReader::open(&path).unwrap();
        let mut buf = Vec::new();
        assert!(matches!(
            windowed.read_tensor_into("nope", &mut buf),
            Err(SafetensorsError::BadEntry(_))
        ));
        std::fs::remove_file(&path).ok();

        // Truncated: header length prefix larger than the file.
        let path = tmp_path("truncated");
        std::fs::write(&path, [0xFFu8; 8]).unwrap();
        assert!(matches!(
            SafetensorsFileReader::open(&path),
            Err(SafetensorsError::Truncated)
        ));
        std::fs::remove_file(&path).ok();

        // Out-of-bounds tensor range against the real file length.
        let header = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 4]); // 4 bytes present, 8 claimed
        let path = tmp_path("oob");
        std::fs::write(&path, &out).unwrap();
        assert!(matches!(
            SafetensorsFileReader::open(&path),
            Err(SafetensorsError::OutOfBounds(_))
        ));
        std::fs::remove_file(&path).ok();

        // Missing file is an I/O error, not a panic.
        assert!(matches!(
            SafetensorsFileReader::open("/no/such/vokra/checkpoint.safetensors"),
            Err(SafetensorsError::Io(_))
        ));
    }
}
