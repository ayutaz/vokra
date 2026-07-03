//! GGUF writer: serialize metadata and tensor data to the on-disk format.
//!
//! Used by the offline conversion tool (FR-TL-01, a separate binary crate that
//! depends on `vokra-core` — the reverse direction is forbidden by FR-LD-05)
//! and by the round-trip tests in this crate. The writer emits GGUF version 3,
//! little-endian, with the layout described in [`super`].

use std::collections::HashSet;

use super::tensor::GgmlType;
use super::value::{GgufArray, GgufMetadataValue};
use super::{DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgufError, align_up, chunks};

/// A tensor queued for writing: descriptor plus its raw little-endian payload.
#[derive(Debug)]
struct TensorEntry {
    name: String,
    dtype: GgmlType,
    dimensions: Vec<u64>,
    data: Vec<u8>,
}

/// Builder that accumulates metadata and tensors and serializes them to GGUF.
///
/// ```
/// use vokra_core::gguf::{GgufBuilder, GgmlType, GgufFile};
///
/// let mut b = GgufBuilder::new();
/// b.add_string("vokra.model.arch", "whisper");
/// b.add_tensor("enc.0.weight", GgmlType::F32, vec![2, 2], vec![0u8; 16])?;
/// let bytes = b.to_bytes()?;
///
/// let file = GgufFile::parse(bytes)?;
/// assert_eq!(file.tensors().len(), 1);
/// # Ok::<(), vokra_core::gguf::GgufError>(())
/// ```
#[derive(Debug)]
pub struct GgufBuilder {
    version: u32,
    alignment: u64,
    metadata: Vec<(String, GgufMetadataValue)>,
    tensors: Vec<TensorEntry>,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// Creates an empty builder targeting GGUF version 3 with the default
    /// alignment of 32 bytes.
    pub fn new() -> Self {
        Self {
            version: GGUF_VERSION,
            alignment: DEFAULT_ALIGNMENT,
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Sets the tensor-data alignment (must be a power of two).
    ///
    /// When the alignment differs from the default of 32, a
    /// `general.alignment` metadata key is emitted automatically so the file
    /// is self-describing. Returns [`GgufError::InvalidAlignment`] if `align`
    /// is zero or not a power of two.
    pub fn set_alignment(&mut self, align: u64) -> Result<&mut Self, GgufError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(align));
        }
        self.alignment = align;
        Ok(self)
    }

    /// Inserts or replaces a metadata key with an arbitrary typed value.
    ///
    /// If `key` is already present its value is overwritten in place, so the
    /// output never contains duplicate keys.
    pub fn add_metadata(&mut self, key: &str, value: GgufMetadataValue) -> &mut Self {
        if let Some(slot) = self.metadata.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.metadata.push((key.to_owned(), value));
        }
        self
    }

    /// Adds a `UINT32` metadata value.
    pub fn add_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.add_metadata(key, GgufMetadataValue::U32(value))
    }

    /// Adds a `FLOAT32` metadata value.
    pub fn add_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.add_metadata(key, GgufMetadataValue::F32(value))
    }

    /// Adds a `BOOL` metadata value.
    pub fn add_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.add_metadata(key, GgufMetadataValue::Bool(value))
    }

    /// Adds a `STRING` metadata value.
    pub fn add_string(&mut self, key: &str, value: &str) -> &mut Self {
        self.add_metadata(key, GgufMetadataValue::String(value.to_owned()))
    }

    /// Queues a tensor for writing.
    ///
    /// `data` must be the little-endian payload whose length equals
    /// [`dtype.payload_size(product(dimensions))`](GgmlType::payload_size) —
    /// `elements * type_size` for dense dtypes, or `(elements / 256) *
    /// type_size` for K-quants; otherwise [`GgufError::TensorSizeMismatch`] is
    /// returned. A K-quant element count that is not a whole number of
    /// super-blocks yields [`GgufError::BlockSizeMisaligned`], and a duplicate
    /// tensor name yields [`GgufError::DuplicateTensor`].
    pub fn add_tensor(
        &mut self,
        name: &str,
        dtype: GgmlType,
        dimensions: Vec<u64>,
        data: Vec<u8>,
    ) -> Result<&mut Self, GgufError> {
        if dimensions.len() > super::tensor::MAX_TENSOR_DIMS {
            return Err(GgufError::TooManyDimensions(dimensions.len()));
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(GgufError::DuplicateTensor(name.to_owned()));
        }
        let mut elems: u64 = 1;
        for &d in &dimensions {
            elems = elems.checked_mul(d).ok_or(GgufError::Overflow)?;
        }
        let expected = dtype.payload_size(elems)?;
        if data.len() as u64 != expected {
            return Err(GgufError::TensorSizeMismatch {
                name: name.to_owned(),
                expected,
                actual: data.len() as u64,
            });
        }
        self.tensors.push(TensorEntry {
            name: name.to_owned(),
            dtype,
            dimensions,
            data,
        });
        Ok(self)
    }

    /// Number of tensors queued for writing.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Number of metadata entries that will be written, including any
    /// auto-injected `general.alignment`.
    pub fn metadata_count(&self) -> usize {
        self.effective_metadata().len()
    }

    /// Serializes the accumulated content to an in-memory GGUF byte buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, GgufError> {
        let metadata = self.effective_metadata();

        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for (key, value) in &metadata {
            write_gguf_string(&mut out, key);
            write_value(&mut out, value);
        }

        // Tensor-data offsets are relative to the tensor-data region and each
        // one is aligned; compute them up front so the tensor infos can carry
        // them.
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut cursor: u64 = 0;
        for t in &self.tensors {
            offsets.push(cursor);
            let size = t.data.len() as u64;
            cursor = align_up(
                cursor.checked_add(size).ok_or(GgufError::Overflow)?,
                self.alignment,
            )?;
        }

        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            write_gguf_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dimensions.len() as u32).to_le_bytes());
            for &dim in &t.dimensions {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&t.dtype.tag().to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }

        // Pad the header/metadata/tensor-info block up to the alignment
        // boundary, then lay out the tensor payloads with per-tensor padding.
        pad_to(&mut out, self.alignment)?;
        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            // Invariant: `out.len()` is at the tensor-data base plus `offset`.
            out.extend_from_slice(&t.data);
            let next = align_up(offset + t.data.len() as u64, self.alignment)?;
            let target_len = (out.len() as u64 + (next - (offset + t.data.len() as u64))) as usize;
            out.resize(target_len, 0);
        }

        Ok(out)
    }

    /// Returns the metadata that will actually be written, injecting
    /// `general.alignment` when the alignment is non-default and stripping any
    /// user-supplied copy so the two never disagree.
    fn effective_metadata(&self) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata: Vec<(String, GgufMetadataValue)> = self
            .metadata
            .iter()
            .filter(|(k, _)| k != chunks::KEY_GENERAL_ALIGNMENT)
            .cloned()
            .collect();
        if self.alignment != DEFAULT_ALIGNMENT {
            metadata.push((
                chunks::KEY_GENERAL_ALIGNMENT.to_owned(),
                GgufMetadataValue::U32(self.alignment as u32),
            ));
        }
        debug_assert_eq!(
            metadata
                .iter()
                .map(|(k, _)| k)
                .collect::<HashSet<_>>()
                .len(),
            metadata.len(),
            "writer must not emit duplicate metadata keys"
        );
        metadata
    }
}

/// Writes a GGUF string: a `u64` little-endian byte length followed by the
/// UTF-8 bytes.
fn write_gguf_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Writes a metadata value: its type tag (for array elements the tag is
/// written by the caller) followed by the payload.
fn write_value(out: &mut Vec<u8>, value: &GgufMetadataValue) {
    out.extend_from_slice(&value.value_type().tag().to_le_bytes());
    write_value_payload(out, value);
}

/// Writes only the payload bytes of a value (no leading type tag).
fn write_value_payload(out: &mut Vec<u8>, value: &GgufMetadataValue) {
    match value {
        GgufMetadataValue::U8(v) => out.push(*v),
        GgufMetadataValue::I8(v) => out.push(*v as u8),
        GgufMetadataValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::Bool(v) => out.push(u8::from(*v)),
        GgufMetadataValue::String(s) => write_gguf_string(out, s),
        GgufMetadataValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufMetadataValue::Array(arr) => write_array(out, arr),
    }
}

/// Writes an array: element type tag, `u64` element count, then each payload.
fn write_array(out: &mut Vec<u8>, arr: &GgufArray) {
    out.extend_from_slice(&arr.element_type.tag().to_le_bytes());
    out.extend_from_slice(&(arr.values.len() as u64).to_le_bytes());
    for v in &arr.values {
        debug_assert_eq!(v.value_type(), arr.element_type);
        write_value_payload(out, v);
    }
}

/// Zero-pads `out` until its length is a multiple of `align`.
fn pad_to(out: &mut Vec<u8>, align: u64) -> Result<(), GgufError> {
    let padded = align_up(out.len() as u64, align)?;
    out.resize(padded as usize, 0);
    Ok(())
}

/// Metadata value used only to exercise the U8-array path, so the value fits
/// in a byte array without loss.
#[cfg(test)]
pub(crate) fn demo_builder() -> GgufBuilder {
    use super::value::GgufValueType;

    let mut b = GgufBuilder::new();
    b.add_metadata("kv.u8", GgufMetadataValue::U8(0x12));
    b.add_metadata("kv.i8", GgufMetadataValue::I8(-5));
    b.add_metadata("kv.u16", GgufMetadataValue::U16(0xBEEF));
    b.add_metadata("kv.i16", GgufMetadataValue::I16(-1234));
    b.add_metadata("kv.u32", GgufMetadataValue::U32(0xDEAD_BEEF));
    b.add_metadata("kv.i32", GgufMetadataValue::I32(-70_000));
    b.add_metadata("kv.f32", GgufMetadataValue::F32(1.5));
    b.add_metadata("kv.bool", GgufMetadataValue::Bool(true));
    b.add_metadata("kv.string", GgufMetadataValue::String("héllo".to_owned()));
    b.add_metadata("kv.u64", GgufMetadataValue::U64(0x0102_0304_0506_0708));
    b.add_metadata("kv.i64", GgufMetadataValue::I64(-9_000_000_000));
    b.add_metadata("kv.f64", GgufMetadataValue::F64(-2.25));
    b.add_metadata(
        "kv.array",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: vec![
                GgufMetadataValue::U32(10),
                GgufMetadataValue::U32(20),
                GgufMetadataValue::U32(30),
            ],
        }),
    );
    b.add_metadata(
        "kv.nested_array",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::Array,
            values: vec![GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![GgufMetadataValue::String("a".to_owned())],
            })],
        }),
    );
    b.add_tensor("t.f32", GgmlType::F32, vec![2, 3], vec![0u8; 24])
        .expect("valid f32 tensor");
    b.add_tensor("t.f16", GgmlType::F16, vec![5], vec![0u8; 10])
        .expect("valid f16 tensor");
    b
}

#[cfg(test)]
mod tests {
    use super::super::GgufFile;
    use super::super::value::GgufValueType;
    use super::*;

    #[test]
    fn roundtrip_all_value_types_and_tensors() {
        let bytes = demo_builder().to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");

        assert_eq!(file.version(), GGUF_VERSION);
        assert_eq!(file.alignment(), DEFAULT_ALIGNMENT);
        assert_eq!(file.get("kv.u8"), Some(&GgufMetadataValue::U8(0x12)));
        assert_eq!(file.get("kv.i8"), Some(&GgufMetadataValue::I8(-5)));
        assert_eq!(file.get("kv.u16"), Some(&GgufMetadataValue::U16(0xBEEF)));
        assert_eq!(file.get("kv.i16"), Some(&GgufMetadataValue::I16(-1234)));
        assert_eq!(
            file.get("kv.u32"),
            Some(&GgufMetadataValue::U32(0xDEAD_BEEF))
        );
        assert_eq!(file.get("kv.i32"), Some(&GgufMetadataValue::I32(-70_000)));
        assert_eq!(file.get("kv.f32"), Some(&GgufMetadataValue::F32(1.5)));
        assert_eq!(file.get("kv.bool"), Some(&GgufMetadataValue::Bool(true)));
        assert_eq!(
            file.get("kv.string").and_then(|v| v.as_str()),
            Some("héllo")
        );
        assert_eq!(
            file.get("kv.u64"),
            Some(&GgufMetadataValue::U64(0x0102_0304_0506_0708))
        );
        assert_eq!(
            file.get("kv.i64"),
            Some(&GgufMetadataValue::I64(-9_000_000_000))
        );
        assert_eq!(file.get("kv.f64"), Some(&GgufMetadataValue::F64(-2.25)));

        let arr = file.get("kv.array").and_then(|v| v.as_array()).unwrap();
        assert_eq!(arr.element_type, GgufValueType::U32);
        assert_eq!(arr.values.len(), 3);
        assert_eq!(arr.values[2], GgufMetadataValue::U32(30));

        let nested = file
            .get("kv.nested_array")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(nested.element_type, GgufValueType::Array);
        let inner = nested.values[0].as_array().unwrap();
        assert_eq!(inner.values[0].as_str(), Some("a"));

        assert_eq!(file.tensors().len(), 2);
        let f16 = file.tensor_info("t.f16").expect("tensor present");
        assert_eq!(f16.dtype, GgmlType::F16);
        assert_eq!(f16.dimensions, vec![5]);
    }

    #[test]
    fn tensor_bytes_survive_roundtrip_with_custom_alignment() {
        let mut b = GgufBuilder::new();
        b.set_alignment(64).expect("power of two");
        let payload_a: Vec<u8> = (0..12u8).collect();
        let payload_b: Vec<u8> = (100..108u8).collect();
        b.add_tensor("a", GgmlType::F32, vec![3], payload_a.clone())
            .unwrap();
        b.add_tensor("b", GgmlType::F16, vec![4], payload_b.clone())
            .unwrap();

        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(file.alignment(), 64);
        assert_eq!(file.tensor_data("a").unwrap(), payload_a.as_slice());
        assert_eq!(file.tensor_data("b").unwrap(), payload_b.as_slice());
        // Every tensor offset must respect the declared alignment.
        for t in file.tensors() {
            assert_eq!(t.offset % 64, 0);
        }
    }

    #[test]
    fn kquant_tensor_bytes_survive_roundtrip() {
        // A correctly-sized Q5_K super-block (176 bytes for 256 elements) is
        // accepted and its raw bytes come back byte-identical.
        let mut b = GgufBuilder::new();
        let payload: Vec<u8> = (0..176u32).map(|i| (i % 251) as u8).collect();
        b.add_tensor("q", GgmlType::Q5K, vec![256], payload.clone())
            .expect("valid one-block Q5_K payload");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(file.tensor_data("q").unwrap(), payload.as_slice());
    }

    #[test]
    fn kquant_wrong_payload_length_is_rejected() {
        // One Q6_K block is 210 bytes; a 200-byte payload is a size mismatch.
        let mut b = GgufBuilder::new();
        let err = b
            .add_tensor("q", GgmlType::Q6K, vec![256], vec![0u8; 200])
            .unwrap_err();
        assert!(matches!(err, GgufError::TensorSizeMismatch { .. }));
    }

    #[test]
    fn kquant_partial_block_dims_are_rejected() {
        // 300 is not a whole number of 256-element super-blocks.
        let mut b = GgufBuilder::new();
        let err = b
            .add_tensor("q", GgmlType::Q4K, vec![300], vec![0u8; 144])
            .unwrap_err();
        assert!(matches!(
            err,
            GgufError::BlockSizeMisaligned {
                elements: 300,
                block_size: 256,
                ..
            }
        ));
    }

    #[test]
    fn tensor_size_mismatch_is_rejected() {
        let mut b = GgufBuilder::new();
        let err = b
            .add_tensor("bad", GgmlType::F32, vec![2, 2], vec![0u8; 8])
            .unwrap_err();
        assert!(matches!(err, GgufError::TensorSizeMismatch { .. }));
    }

    #[test]
    fn duplicate_tensor_name_is_rejected() {
        let mut b = GgufBuilder::new();
        b.add_tensor("dup", GgmlType::F32, vec![1], vec![0u8; 4])
            .unwrap();
        let err = b
            .add_tensor("dup", GgmlType::F32, vec![1], vec![0u8; 4])
            .unwrap_err();
        assert!(matches!(err, GgufError::DuplicateTensor(_)));
    }

    #[test]
    fn non_power_of_two_alignment_is_rejected() {
        let mut b = GgufBuilder::new();
        assert!(matches!(
            b.set_alignment(48),
            Err(GgufError::InvalidAlignment(48))
        ));
    }

    #[test]
    fn duplicate_metadata_key_overwrites() {
        let mut b = GgufBuilder::new();
        b.add_u32("k", 1);
        b.add_u32("k", 2);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(file.get("k"), Some(&GgufMetadataValue::U32(2)));
        assert_eq!(file.metadata().len(), 1);
    }

    #[test]
    fn too_many_dimensions_is_rejected() {
        // The builder caps rank at MAX_TENSOR_DIMS (4); five dims is malformed
        // and is rejected before the payload-length check runs.
        let mut b = GgufBuilder::new();
        let err = b
            .add_tensor("t", GgmlType::F32, vec![1, 1, 1, 1, 1], vec![0u8; 4])
            .unwrap_err();
        assert!(matches!(err, GgufError::TooManyDimensions(5)));
    }
}
