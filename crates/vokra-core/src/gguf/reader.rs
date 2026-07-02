//! GGUF reader: parse a file into metadata and tensor descriptors, then lend
//! tensor payloads as zero-copy `&[u8]` slices.
//!
//! # zero-copy strategy (std-I/O substitute for mmap)
//!
//! `vokra-core` forbids `unsafe` (workspace lint `unsafe_code = "deny"`), so a
//! true `mmap` (which requires `unsafe` or an external crate such as
//! `memmap2`) cannot live here. [`GgufFile::open`] therefore reads the whole
//! file into an owned buffer with `std::io` and lends `&[u8]` slices into that
//! buffer — the tensor-access API is genuinely copy-free, but the *lazy*
//! cold-start property of `mmap` (FR-LD-01 / NFR-PF-11) is a follow-up that
//! needs either a dedicated `unsafe`-allowed crate or `memmap2`.
//!
//! All offsets and lengths are bounds-checked at parse time, so the slice
//! accessors never index out of range and never panic (NFR-RL-07).

use std::collections::HashMap;
use std::path::Path;

use super::tensor::{GgmlType, GgufTensorInfo, MAX_TENSOR_DIMS};
use super::value::{GgufArray, GgufMetadataValue, GgufValueType};
use super::{DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgufError, align_up, chunks};

/// Maximum array nesting depth accepted while decoding metadata.
///
/// A guard against stack exhaustion from adversarial deeply-nested arrays
/// (NFR-RL-07). Real models nest at most one level.
const MAX_ARRAY_DEPTH: usize = 64;

/// A parsed GGUF file: decoded header, metadata and tensor infos, plus the
/// backing byte buffer that tensor slices borrow from.
///
/// Construct with [`GgufFile::open`] (from a path) or [`GgufFile::parse`]
/// (from an in-memory buffer, used by the writer round-trip tests).
pub struct GgufFile {
    data: Vec<u8>,
    version: u32,
    alignment: u64,
    /// Metadata in file order (`vokra.*` keys included).
    metadata: Vec<(String, GgufMetadataValue)>,
    metadata_index: HashMap<String, usize>,
    tensors: Vec<GgufTensorInfo>,
    tensor_index: HashMap<String, usize>,
    /// Absolute byte offset where the tensor-data region begins.
    tensor_data_offset: u64,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Deliberately omit the (potentially large) backing buffer.
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("metadata_keys", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("tensor_data_offset", &self.tensor_data_offset)
            .field("file_len", &self.data.len())
            .finish()
    }
}

impl GgufFile {
    /// Opens and parses a GGUF file from disk.
    ///
    /// Reads the whole file into memory (see the module docs on the zero-copy
    /// strategy). Returns [`GgufError::Io`] on I/O failure or a parse error
    /// variant for malformed content.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let data = std::fs::read(path).map_err(GgufError::Io)?;
        Self::parse(data)
    }

    /// Parses a GGUF file from an owned in-memory buffer.
    pub fn parse(data: Vec<u8>) -> Result<Self, GgufError> {
        let parsed = {
            let mut r = ByteReader::new(&data);
            parse_all(&mut r, data.len())?
        };
        let Parsed {
            version,
            alignment,
            metadata,
            tensors,
            tensor_data_offset,
        } = parsed;

        let mut metadata_index = HashMap::with_capacity(metadata.len());
        for (i, (k, _)) in metadata.iter().enumerate() {
            // Last occurrence wins (the writer never emits duplicate keys).
            metadata_index.insert(k.clone(), i);
        }
        let mut tensor_index = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            tensor_index.insert(t.name.clone(), i);
        }

        Ok(Self {
            data,
            version,
            alignment,
            metadata,
            metadata_index,
            tensors,
            tensor_index,
            tensor_data_offset,
        })
    }

    /// GGUF format version (always 3 for files this reader accepts).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Tensor-data alignment in bytes (`general.alignment`, default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// All metadata entries in file order.
    pub fn metadata(&self) -> &[(String, GgufMetadataValue)] {
        &self.metadata
    }

    /// Looks up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata_index.get(key).map(|&i| &self.metadata[i].1)
    }

    /// All tensor descriptors in file order.
    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    /// Looks up a tensor descriptor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Lends a tensor's raw payload as a zero-copy slice into the backing
    /// buffer, or `None` if no tensor has that name.
    ///
    /// The returned range was bounds-checked during parsing, so this never
    /// panics.
    pub fn tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_info(name)?;
        Some(self.tensor_bytes(info))
    }

    /// Lends the payload for a known tensor descriptor (see
    /// [`GgufFile::tensor_data`]).
    pub fn tensor_bytes(&self, info: &GgufTensorInfo) -> &[u8] {
        // Bounds were validated in `parse_all`; recompute the checked range.
        let start = (self.tensor_data_offset + info.offset) as usize;
        let len = info.byte_len().expect("byte_len validated during parse") as usize;
        &self.data[start..start + len]
    }
}

/// Intermediate parse result (owns everything; borrows nothing).
struct Parsed {
    version: u32,
    alignment: u64,
    metadata: Vec<(String, GgufMetadataValue)>,
    tensors: Vec<GgufTensorInfo>,
    tensor_data_offset: u64,
}

/// A bounds-checked, little-endian cursor over an in-memory GGUF buffer.
struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize, ctx: &'static str) -> Result<&'a [u8], GgufError> {
        if self.remaining() < n {
            return Err(GgufError::Truncated(ctx));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self, ctx: &'static str) -> Result<u8, GgufError> {
        Ok(self.take(1, ctx)?[0])
    }

    fn u16(&mut self, ctx: &'static str) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.take(2, ctx)?.try_into().unwrap()))
    }

    fn u32(&mut self, ctx: &'static str) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4, ctx)?.try_into().unwrap()))
    }

    fn u64(&mut self, ctx: &'static str) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8, ctx)?.try_into().unwrap()))
    }

    /// Reads a GGUF string: `u64` byte length followed by validated UTF-8.
    fn gguf_string(&mut self, ctx: &'static str) -> Result<String, GgufError> {
        let len = self.u64(ctx)?;
        // A length exceeding the remaining bytes is malformed; this also
        // bounds the allocation to the file size.
        if len > self.remaining() as u64 {
            return Err(GgufError::Truncated(ctx));
        }
        let bytes = self.take(len as usize, ctx)?;
        let s = std::str::from_utf8(bytes).map_err(GgufError::InvalidString)?;
        Ok(s.to_owned())
    }
}

/// Parses the full file from `r` (whose buffer is `file_len` bytes).
fn parse_all(r: &mut ByteReader<'_>, file_len: usize) -> Result<Parsed, GgufError> {
    let magic = r.take(4, "magic")?;
    if magic != GGUF_MAGIC {
        let mut m = [0u8; 4];
        m.copy_from_slice(magic);
        return Err(GgufError::BadMagic(m));
    }
    let version = r.u32("version")?;
    if version != GGUF_VERSION {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = r.u64("tensor_count")?;
    let kv_count = r.u64("metadata_kv_count")?;
    // Each entry consumes at least one byte, so a count larger than the
    // remaining bytes is definitely malformed. This caps loop iterations and
    // prevents count-driven resource exhaustion (NFR-RL-07).
    if tensor_count > r.remaining() as u64 || kv_count > r.remaining() as u64 {
        return Err(GgufError::Truncated("declared count exceeds file size"));
    }

    let mut metadata = Vec::new();
    for _ in 0..kv_count {
        let key = r.gguf_string("metadata key")?;
        let value = read_kv_value(r, 0)?;
        metadata.push((key, value));
    }

    let alignment = resolve_alignment(&metadata)?;

    let mut tensors = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..tensor_count {
        let name = r.gguf_string("tensor name")?;
        let n_dims = r.u32("tensor n_dims")? as usize;
        if n_dims > MAX_TENSOR_DIMS {
            return Err(GgufError::TooManyDimensions(n_dims));
        }
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(r.u64("tensor dim")?);
        }
        let dtype = GgmlType::from_tag(r.u32("tensor dtype")?)?;
        let offset = r.u64("tensor offset")?;
        if !seen.insert(name.clone()) {
            return Err(GgufError::DuplicateTensor(name));
        }
        tensors.push(GgufTensorInfo {
            name,
            dimensions,
            dtype,
            offset,
        });
    }

    // The tensor-data region starts at the next alignment boundary after the
    // tensor infos. With zero tensors there is no data region, so a file that
    // ends right after the tensor infos (before any alignment padding) is
    // still valid.
    let tensor_data_offset = align_up(r.position() as u64, alignment)?;
    if !tensors.is_empty() && tensor_data_offset > file_len as u64 {
        return Err(GgufError::OffsetOutOfBounds(
            "tensor-data region starts past end of file".to_owned(),
        ));
    }

    // Validate every tensor's offset+size fits inside the file so the
    // zero-copy slice accessors are always in range.
    for t in &tensors {
        if t.offset % alignment != 0 {
            return Err(GgufError::UnalignedTensorOffset {
                tensor: t.name.clone(),
                offset: t.offset,
                alignment,
            });
        }
        let start = tensor_data_offset
            .checked_add(t.offset)
            .ok_or(GgufError::Overflow)?;
        let end = start
            .checked_add(t.byte_len()?)
            .ok_or(GgufError::Overflow)?;
        if end > file_len as u64 {
            return Err(GgufError::OffsetOutOfBounds(t.name.clone()));
        }
    }

    Ok(Parsed {
        version,
        alignment,
        metadata,
        tensors,
        tensor_data_offset,
    })
}

/// Resolves the tensor-data alignment from `general.alignment`, defaulting to
/// [`DEFAULT_ALIGNMENT`] and rejecting non-power-of-two values.
fn resolve_alignment(metadata: &[(String, GgufMetadataValue)]) -> Result<u64, GgufError> {
    let Some((_, value)) = metadata
        .iter()
        .find(|(k, _)| k == chunks::KEY_GENERAL_ALIGNMENT)
    else {
        return Ok(DEFAULT_ALIGNMENT);
    };
    let align = value.as_u64().ok_or_else(|| {
        GgufError::OffsetOutOfBounds("general.alignment is not an integer".to_owned())
    })?;
    if align == 0 || !align.is_power_of_two() {
        return Err(GgufError::InvalidAlignment(align));
    }
    Ok(align)
}

/// Reads a top-level metadata value (type tag followed by payload).
fn read_kv_value(r: &mut ByteReader<'_>, depth: usize) -> Result<GgufMetadataValue, GgufError> {
    let vt = GgufValueType::from_tag(r.u32("value type")?)?;
    read_payload(r, vt, depth)
}

/// Reads the payload of a value of known type `vt` (no leading type tag).
fn read_payload(
    r: &mut ByteReader<'_>,
    vt: GgufValueType,
    depth: usize,
) -> Result<GgufMetadataValue, GgufError> {
    Ok(match vt {
        GgufValueType::U8 => GgufMetadataValue::U8(r.u8("u8")?),
        GgufValueType::I8 => GgufMetadataValue::I8(r.u8("i8")? as i8),
        GgufValueType::U16 => GgufMetadataValue::U16(r.u16("u16")?),
        GgufValueType::I16 => GgufMetadataValue::I16(r.u16("i16")? as i16),
        GgufValueType::U32 => GgufMetadataValue::U32(r.u32("u32")?),
        GgufValueType::I32 => GgufMetadataValue::I32(r.u32("i32")? as i32),
        GgufValueType::F32 => GgufMetadataValue::F32(f32::from_bits(r.u32("f32")?)),
        GgufValueType::Bool => {
            let b = r.u8("bool")?;
            match b {
                0 => GgufMetadataValue::Bool(false),
                1 => GgufMetadataValue::Bool(true),
                other => return Err(GgufError::InvalidBool(other)),
            }
        }
        GgufValueType::String => GgufMetadataValue::String(r.gguf_string("string")?),
        GgufValueType::U64 => GgufMetadataValue::U64(r.u64("u64")?),
        GgufValueType::I64 => GgufMetadataValue::I64(r.u64("i64")? as i64),
        GgufValueType::F64 => GgufMetadataValue::F64(f64::from_bits(r.u64("f64")?)),
        GgufValueType::Array => GgufMetadataValue::Array(read_array(r, depth)?),
    })
}

/// Reads a homogeneous array: element type tag, `u64` count, then elements.
fn read_array(r: &mut ByteReader<'_>, depth: usize) -> Result<GgufArray, GgufError> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(GgufError::ArrayTooDeep(depth));
    }
    let element_type = GgufValueType::from_tag(r.u32("array element type")?)?;
    let len = r.u64("array length")?;
    // Every element consumes at least one byte, so a count beyond the
    // remaining bytes is malformed; also bounds the allocation.
    if len > r.remaining() as u64 {
        return Err(GgufError::Truncated("array length exceeds file size"));
    }
    let mut values = Vec::new();
    for _ in 0..len {
        values.push(read_payload(r, element_type, depth + 1)?);
    }
    Ok(GgufArray {
        element_type,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::super::writer::{GgufBuilder, demo_builder};
    use super::*;

    /// A minimal, valid, hand-built GGUF header (version 3, no tensors, no KV)
    /// used to pin the exact on-disk byte layout independent of the writer.
    fn hand_built_empty() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF"); // magic
        v.extend_from_slice(&3u32.to_le_bytes()); // version
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        v.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        v
    }

    #[test]
    fn parses_hand_built_empty_header() {
        let file = GgufFile::parse(hand_built_empty()).expect("valid empty gguf");
        assert_eq!(file.version(), 3);
        assert_eq!(file.alignment(), DEFAULT_ALIGNMENT);
        assert!(file.metadata().is_empty());
        assert!(file.tensors().is_empty());
    }

    #[test]
    fn reads_vokra_prefixed_key_like_any_other() {
        let mut b = GgufBuilder::new();
        b.add_u32(chunks::KEY_FRONTEND_N_FFT, 400);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_FRONTEND_N_FFT),
            Some(&GgufMetadataValue::U32(400))
        );
    }

    #[test]
    fn tensor_info_and_offsets_resolve() {
        let file = GgufFile::parse(demo_builder().to_bytes().unwrap()).unwrap();
        let f32 = file.tensor_info("t.f32").expect("present");
        assert_eq!(f32.dtype, GgmlType::F32);
        assert_eq!(f32.dimensions, vec![2, 3]);
        assert_eq!(f32.element_count().unwrap(), 6);
        assert_eq!(f32.byte_len().unwrap(), 24);
        assert_eq!(f32.offset % file.alignment(), 0);
    }

    // --- malformed input safety (M0-03-T06) ------------------------------

    #[test]
    fn bad_magic_is_rejected() {
        let mut v = hand_built_empty();
        v[0] = b'X';
        assert!(matches!(GgufFile::parse(v), Err(GgufError::BadMagic(_))));
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let mut v = hand_built_empty();
        v[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn truncated_header_is_rejected() {
        let v = b"GGUF".to_vec(); // magic only, no version/counts
        assert!(matches!(GgufFile::parse(v), Err(GgufError::Truncated(_))));
    }

    #[test]
    fn oversized_kv_count_is_rejected_without_allocating() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        v.extend_from_slice(&u64::MAX.to_le_bytes()); // absurd kv_count
        assert!(matches!(GgufFile::parse(v), Err(GgufError::Truncated(_))));
    }

    #[test]
    fn oversized_string_length_is_rejected() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        v.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1
        v.extend_from_slice(&u64::MAX.to_le_bytes()); // key length = absurd
        assert!(matches!(GgufFile::parse(v), Err(GgufError::Truncated(_))));
    }

    #[test]
    fn unsupported_dtype_is_rejected() {
        // Build a header with one tensor declaring ggml type 2 (Q4_0).
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        v.extend_from_slice(&3u64.to_le_bytes()); // name length
        v.extend_from_slice(b"bad");
        v.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        v.extend_from_slice(&4u64.to_le_bytes()); // dim[0] = 4
        v.extend_from_slice(&2u32.to_le_bytes()); // dtype tag 2 = Q4_0 (unsupported)
        v.extend_from_slice(&0u64.to_le_bytes()); // offset
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::UnsupportedDtype(2))
        ));
    }

    #[test]
    fn tensor_offset_out_of_bounds_is_rejected() {
        // One F32 tensor of 4 elements (16 bytes) but its offset points far
        // past the (empty) tensor-data region.
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        v.extend_from_slice(&1u64.to_le_bytes()); // name length
        v.extend_from_slice(b"t");
        v.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        v.extend_from_slice(&4u64.to_le_bytes()); // dim[0] = 4
        v.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        v.extend_from_slice(&4096u64.to_le_bytes()); // offset far past EOF
        // No tensor data written at all.
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::OffsetOutOfBounds(_))
        ));
    }

    #[test]
    fn duplicate_tensor_name_is_rejected() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&2u64.to_le_bytes()); // tensor_count = 2
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        for _ in 0..2 {
            v.extend_from_slice(&3u64.to_le_bytes());
            v.extend_from_slice(b"dup");
            v.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            v.extend_from_slice(&1u64.to_le_bytes()); // dim
            v.extend_from_slice(&0u32.to_le_bytes()); // F32
            v.extend_from_slice(&0u64.to_le_bytes()); // offset
        }
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::DuplicateTensor(_))
        ));
    }

    #[test]
    fn invalid_utf8_key_is_rejected() {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        v.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1
        v.extend_from_slice(&2u64.to_le_bytes()); // key length = 2
        v.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::InvalidString(_))
        ));
    }

    #[test]
    fn unaligned_tensor_offset_is_rejected() {
        // Default alignment 32, offset 8 is not a multiple of 32.
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        v.extend_from_slice(&1u64.to_le_bytes());
        v.extend_from_slice(b"t");
        v.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        v.extend_from_slice(&1u64.to_le_bytes()); // dim = 1
        v.extend_from_slice(&0u32.to_le_bytes()); // F32
        v.extend_from_slice(&8u64.to_le_bytes()); // offset 8 (misaligned)
        // Provide enough trailing bytes that only the alignment check fails.
        v.resize(v.len().next_multiple_of(32) + 64, 0);
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::UnalignedTensorOffset { .. })
        ));
    }

    // --- malformed metadata value / alignment safety ---------------------

    /// Emits `magic + version 3 + tensor_count 0 + kv_count 1` and a
    /// length-prefixed `key`, leaving the caller to append the raw value bytes
    /// (type tag + payload). Pins the reader's handling of malformed metadata
    /// independently of the writer, which cannot emit these files.
    fn gguf_header_one_kv(key: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        v.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count = 1
        v.extend_from_slice(&(key.len() as u64).to_le_bytes());
        v.extend_from_slice(key.as_bytes());
        v
    }

    #[test]
    fn unsupported_value_type_tag_is_rejected() {
        // A value-type tag of 13 is one past the spec's maximum (12 = FLOAT64).
        let mut v = gguf_header_one_kv("k");
        v.extend_from_slice(&13u32.to_le_bytes());
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::UnsupportedValueType(13))
        ));
    }

    #[test]
    fn invalid_bool_byte_is_rejected() {
        // BOOL (tag 7) accepts only 0 or 1; a byte of 2 is non-canonical
        // (NFR-RL-07 strict-bool hardening).
        let mut v = gguf_header_one_kv("b");
        v.extend_from_slice(&7u32.to_le_bytes()); // value type: Bool
        v.push(2); // illegal bool payload
        assert!(matches!(GgufFile::parse(v), Err(GgufError::InvalidBool(2))));
    }

    #[test]
    fn general_alignment_zero_is_rejected_without_panic() {
        // SAFETY: a file-supplied alignment of 0 must be rejected by
        // `resolve_alignment` *before* `align_up` computes `value % align`,
        // which would divide by zero. The builder strips any user
        // `general.alignment`, so this file can only be hand-built.
        let mut v = gguf_header_one_kv(chunks::KEY_GENERAL_ALIGNMENT);
        v.extend_from_slice(&4u32.to_le_bytes()); // value type: U32
        v.extend_from_slice(&0u32.to_le_bytes()); // alignment = 0
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::InvalidAlignment(0))
        ));
    }

    #[test]
    fn general_alignment_non_power_of_two_is_rejected() {
        let mut v = gguf_header_one_kv(chunks::KEY_GENERAL_ALIGNMENT);
        v.extend_from_slice(&4u32.to_le_bytes()); // value type: U32
        v.extend_from_slice(&3u32.to_le_bytes()); // alignment = 3
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::InvalidAlignment(3))
        ));
    }

    #[test]
    fn general_alignment_non_integer_is_rejected() {
        // A string-typed `general.alignment` cannot widen to `u64`, so
        // `resolve_alignment` reports the "not an integer" OffsetOutOfBounds.
        let mut v = gguf_header_one_kv(chunks::KEY_GENERAL_ALIGNMENT);
        v.extend_from_slice(&8u32.to_le_bytes()); // value type: String
        v.extend_from_slice(&1u64.to_le_bytes()); // string length
        v.extend_from_slice(b"x");
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::OffsetOutOfBounds(_))
        ));
    }

    #[test]
    fn too_many_dimensions_is_rejected_by_reader() {
        // GGUF caps tensor rank at MAX_TENSOR_DIMS (4); n_dims = 5 is malformed.
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        v.extend_from_slice(&1u64.to_le_bytes()); // name length
        v.extend_from_slice(b"t");
        v.extend_from_slice(&5u32.to_le_bytes()); // n_dims = 5 (> 4)
        for _ in 0..5 {
            v.extend_from_slice(&1u64.to_le_bytes()); // five dims of 1
        }
        v.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        v.extend_from_slice(&0u64.to_le_bytes()); // offset
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::TooManyDimensions(5))
        ));
    }

    #[test]
    fn tensor_bytelen_overflow_is_rejected() {
        // A single F32 tensor whose lone dimension is u64::MAX: element_count
        // is u64::MAX and byte_len = u64::MAX * 4 overflows u64. The reader
        // must surface Overflow, never panic or slice out of bounds.
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
        v.extend_from_slice(&0u64.to_le_bytes()); // kv_count = 0
        v.extend_from_slice(&1u64.to_le_bytes()); // name length
        v.extend_from_slice(b"t");
        v.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        v.extend_from_slice(&u64::MAX.to_le_bytes()); // dim[0] = u64::MAX
        v.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        v.extend_from_slice(&0u64.to_le_bytes()); // offset = 0 (aligned)
        // Pad so the aligned tensor-data region fits inside the file, letting
        // parsing reach the byte_len overflow check rather than OffsetOutOfBounds.
        let aligned = v.len().next_multiple_of(32);
        v.resize(aligned, 0);
        assert!(matches!(GgufFile::parse(v), Err(GgufError::Overflow)));
    }

    #[test]
    fn array_nested_too_deep_is_rejected() {
        // An array KV nested past MAX_ARRAY_DEPTH (64). The decoder must bail
        // with ArrayTooDeep rather than recurse until the stack overflows.
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        v.extend_from_slice(&1u64.to_le_bytes()); // kv_count = 1
        v.extend_from_slice(&1u64.to_le_bytes()); // key length
        v.extend_from_slice(b"a");
        v.extend_from_slice(&9u32.to_le_bytes()); // KV value type: Array
        // 65 nested array headers (element_type Array, count 1); parsing stops
        // at depth 64 before consuming them all.
        for _ in 0..65 {
            v.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
            v.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        }
        assert!(matches!(
            GgufFile::parse(v),
            Err(GgufError::ArrayTooDeep(_))
        ));
    }
}
