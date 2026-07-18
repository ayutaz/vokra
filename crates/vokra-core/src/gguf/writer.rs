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

/// A tensor declaration for [`GgufStreamWriter`]: descriptor now, payload
/// later (streamed in declaration order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorDecl {
    /// Tensor name (must be unique across the declaration set).
    pub name: String,
    /// Element type.
    pub dtype: GgmlType,
    /// Dimensions (at most [`super::tensor::MAX_TENSOR_DIMS`]).
    pub dimensions: Vec<u64>,
}

/// Streaming GGUF writer: the whole header / metadata / tensor-info block
/// is written up front from *declarations*, then each tensor payload
/// streams through [`Self::write_tensor`] in declaration order — so a
/// multi-GB model converts while at most **one** tensor payload is held
/// in memory (the Moshi full-7B converter previously materialized the
/// entire model ~3× ≈ 97 GiB; this is the fix's core primitive).
///
/// GGUF needs every tensor's data offset in the header, but offsets are
/// fully determined by the declarations (payload sizes derive from
/// `dtype` × `dimensions`; per-tensor alignment padding is deterministic),
/// so no backpatching pass is needed: a single forward write suffices.
///
/// The produced bytes are **identical** to
/// [`GgufBuilder::to_bytes`] over the same metadata + tensors (pinned by
/// `stream_writer_matches_builder_bytes`), so readers cannot tell the two
/// paths apart.
///
/// # Contract (loud — FR-EX-08)
///
/// - `metadata` supplies metadata (and alignment) only; a builder with
///   queued tensors is rejected ([`GgufError::InvalidStreamUse`]) rather
///   than silently merged.
/// - Payloads must arrive exactly once each, in declaration order, sized
///   exactly `dtype.payload_size(product(dimensions))`.
/// - [`Self::finish`] fails unless every declared payload was written (a
///   truncated tensor-data region must never look like success).
pub struct GgufStreamWriter<W: std::io::Write> {
    out: W,
    alignment: u64,
    /// Per declaration: (name, expected payload bytes, data-region offset).
    plan: Vec<(String, u64, u64)>,
    /// Next `plan` index [`Self::write_tensor`] expects.
    next: usize,
}

impl<W: std::io::Write> std::fmt::Debug for GgufStreamWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `W` need not be `Debug`; the progress fields are what matters.
        f.debug_struct("GgufStreamWriter")
            .field("alignment", &self.alignment)
            .field("declared", &self.plan.len())
            .field("next", &self.next)
            .finish()
    }
}

impl<W: std::io::Write> GgufStreamWriter<W> {
    /// Writes the complete prelude (header + metadata + tensor infos +
    /// alignment padding) into `out` and returns the writer positioned at
    /// the tensor-data region.
    ///
    /// `metadata` contributes metadata entries and the alignment (its
    /// `set_alignment` is honored, including the auto-injected
    /// `general.alignment` key); `decls` declares every tensor in payload
    /// order.
    ///
    /// # Errors
    ///
    /// [`GgufError::InvalidStreamUse`] if `metadata` has queued tensors;
    /// [`GgufError::DuplicateTensor`] / [`GgufError::TooManyDimensions`] /
    /// [`GgufError::Overflow`] / block-size errors for invalid
    /// declarations; [`GgufError::Io`] on write failure.
    pub fn begin(
        mut out: W,
        metadata: &GgufBuilder,
        decls: &[GgufTensorDecl],
    ) -> Result<Self, GgufError> {
        if metadata.tensor_count() != 0 {
            return Err(GgufError::InvalidStreamUse(format!(
                "the metadata builder carries {} queued tensor(s); declare \
                 streamed tensors through `decls` only (no silent merge)",
                metadata.tensor_count()
            )));
        }
        let alignment = metadata.alignment;
        // Validate declarations and compute payload sizes + offsets with
        // exactly the `to_bytes` arithmetic (byte-identity contract).
        let mut seen = HashSet::new();
        let mut plan = Vec::with_capacity(decls.len());
        let mut cursor: u64 = 0;
        for d in decls {
            if d.dimensions.len() > super::tensor::MAX_TENSOR_DIMS {
                return Err(GgufError::TooManyDimensions(d.dimensions.len()));
            }
            if !seen.insert(d.name.as_str()) {
                return Err(GgufError::DuplicateTensor(d.name.clone()));
            }
            let mut elems: u64 = 1;
            for &dim in &d.dimensions {
                elems = elems.checked_mul(dim).ok_or(GgufError::Overflow)?;
            }
            let size = d.dtype.payload_size(elems)?;
            plan.push((d.name.clone(), size, cursor));
            cursor = align_up(
                cursor.checked_add(size).ok_or(GgufError::Overflow)?,
                alignment,
            )?;
        }

        // Serialize the prelude into one buffer (metadata is small even
        // with an embedded tokenizer blob; only tensor payloads are big).
        let effective = metadata.effective_metadata();
        let mut head = Vec::new();
        head.extend_from_slice(&GGUF_MAGIC);
        head.extend_from_slice(&metadata.version.to_le_bytes());
        head.extend_from_slice(&(decls.len() as u64).to_le_bytes());
        head.extend_from_slice(&(effective.len() as u64).to_le_bytes());
        for (key, value) in &effective {
            write_gguf_string(&mut head, key);
            write_value(&mut head, value);
        }
        for (d, (_, _, offset)) in decls.iter().zip(&plan) {
            write_gguf_string(&mut head, &d.name);
            head.extend_from_slice(&(d.dimensions.len() as u32).to_le_bytes());
            for &dim in &d.dimensions {
                head.extend_from_slice(&dim.to_le_bytes());
            }
            head.extend_from_slice(&d.dtype.tag().to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
        }
        pad_to(&mut head, alignment)?;
        out.write_all(&head).map_err(GgufError::Io)?;

        Ok(Self {
            out,
            alignment,
            plan,
            next: 0,
        })
    }

    /// Streams the next declared tensor's complete payload (plus its
    /// trailing alignment padding).
    ///
    /// # Errors
    ///
    /// [`GgufError::InvalidStreamUse`] out of declaration order or past
    /// the last declaration; [`GgufError::TensorSizeMismatch`] on a wrong
    /// payload length; [`GgufError::Io`] on write failure.
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> Result<(), GgufError> {
        let Some((want_name, want_size, offset)) = self.plan.get(self.next) else {
            return Err(GgufError::InvalidStreamUse(format!(
                "all {} declared tensor payloads were already written; \
                 unexpected extra payload `{name}`",
                self.plan.len()
            )));
        };
        if want_name != name {
            return Err(GgufError::InvalidStreamUse(format!(
                "payloads must arrive in declaration order: expected \
                 `{want_name}` (index {}), got `{name}`",
                self.next
            )));
        }
        if data.len() as u64 != *want_size {
            return Err(GgufError::TensorSizeMismatch {
                name: name.to_owned(),
                expected: *want_size,
                actual: data.len() as u64,
            });
        }
        self.out.write_all(data).map_err(GgufError::Io)?;
        // Trailing padding up to the next aligned offset (the last tensor
        // also pads to the alignment boundary — `to_bytes` parity).
        let end = offset + *want_size;
        let padded_end = align_up(end, self.alignment)?;
        let mut pad = (padded_end - end) as usize;
        let zeros = [0u8; 64];
        while pad > 0 {
            let n = pad.min(zeros.len());
            self.out.write_all(&zeros[..n]).map_err(GgufError::Io)?;
            pad -= n;
        }
        self.next += 1;
        Ok(())
    }

    /// Declared payloads not yet written (observability / progress).
    pub fn remaining(&self) -> usize {
        self.plan.len() - self.next
    }

    /// Flushes and returns the inner writer.
    ///
    /// # Errors
    ///
    /// [`GgufError::InvalidStreamUse`] if any declared payload was never
    /// written (a truncated data region must never pass as success);
    /// [`GgufError::Io`] on flush failure.
    pub fn finish(mut self) -> Result<W, GgufError> {
        if self.next != self.plan.len() {
            return Err(GgufError::InvalidStreamUse(format!(
                "only {} of {} declared tensor payloads were written; \
                 refusing to finish a truncated tensor-data region",
                self.next,
                self.plan.len()
            )));
        }
        self.out.flush().map_err(GgufError::Io)?;
        Ok(self.out)
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

    // --- GgufStreamWriter -------------------------------------------------

    /// The tensors both stream-writer parity tests exercise: mixed dtypes,
    /// deliberately unaligned payload lengths (12 / 10 / 176 bytes) so the
    /// per-tensor padding arithmetic is actually covered.
    fn stream_parity_tensors() -> Vec<(String, GgmlType, Vec<u64>, Vec<u8>)> {
        vec![
            ("a.f32".into(), GgmlType::F32, vec![3], (0..12u8).collect()),
            ("b.f16".into(), GgmlType::F16, vec![5], (50..60u8).collect()),
            (
                "c.q5k".into(),
                GgmlType::Q5K,
                vec![256],
                (0..176u32).map(|i| (i % 251) as u8).collect(),
            ),
            (
                "d.bf16".into(),
                GgmlType::BF16,
                vec![2, 2],
                vec![0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40],
            ),
        ]
    }

    fn stream_metadata() -> GgufBuilder {
        let mut m = GgufBuilder::new();
        m.add_string("vokra.model.arch", "moshi");
        m.add_u32("vokra.test.answer", 42);
        m
    }

    fn decls_of(tensors: &[(String, GgmlType, Vec<u64>, Vec<u8>)]) -> Vec<GgufTensorDecl> {
        tensors
            .iter()
            .map(|(name, dtype, dims, _)| GgufTensorDecl {
                name: name.clone(),
                dtype: *dtype,
                dimensions: dims.clone(),
            })
            .collect()
    }

    #[test]
    fn stream_writer_matches_builder_bytes() {
        // The byte-identity contract: same metadata + tensors through
        // GgufBuilder::to_bytes and GgufStreamWriter must be identical, so
        // no reader can tell the paths apart.
        let tensors = stream_parity_tensors();

        let mut builder = stream_metadata();
        for (name, dtype, dims, data) in &tensors {
            builder
                .add_tensor(name, *dtype, dims.clone(), data.clone())
                .unwrap();
        }
        let via_builder = builder.to_bytes().unwrap();

        let mut w =
            GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &decls_of(&tensors)).unwrap();
        for (name, _, _, data) in &tensors {
            w.write_tensor(name, data).unwrap();
        }
        let via_stream = w.finish().unwrap();

        assert_eq!(via_builder, via_stream, "byte-identical outputs");
        // And the result parses with the identical content.
        let file = GgufFile::parse(via_stream).unwrap();
        assert_eq!(file.tensors().len(), tensors.len());
        for (name, _, _, data) in &tensors {
            assert_eq!(file.tensor_data(name).unwrap(), data.as_slice());
        }
    }

    #[test]
    fn stream_writer_matches_builder_bytes_with_custom_alignment() {
        let tensors = stream_parity_tensors();
        let custom = || {
            let mut m = stream_metadata();
            m.set_alignment(64).unwrap();
            m
        };

        let mut builder = custom();
        for (name, dtype, dims, data) in &tensors {
            builder
                .add_tensor(name, *dtype, dims.clone(), data.clone())
                .unwrap();
        }
        let via_builder = builder.to_bytes().unwrap();

        let mut w = GgufStreamWriter::begin(Vec::new(), &custom(), &decls_of(&tensors)).unwrap();
        for (name, _, _, data) in &tensors {
            w.write_tensor(name, data).unwrap();
        }
        let via_stream = w.finish().unwrap();
        assert_eq!(via_builder, via_stream);
        assert_eq!(GgufFile::parse(via_stream).unwrap().alignment(), 64);
    }

    #[test]
    fn stream_writer_rejects_metadata_builder_with_queued_tensors() {
        let mut m = stream_metadata();
        m.add_tensor("stray", GgmlType::F32, vec![1], vec![0u8; 4])
            .unwrap();
        let err = GgufStreamWriter::begin(Vec::new(), &m, &[]).unwrap_err();
        assert!(matches!(err, GgufError::InvalidStreamUse(_)), "{err}");
    }

    #[test]
    fn stream_writer_rejects_out_of_order_wrong_size_and_extras() {
        let tensors = stream_parity_tensors();
        let decls = decls_of(&tensors);

        // Out of order.
        let mut w = GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &decls).unwrap();
        let err = w.write_tensor("b.f16", &tensors[1].3).unwrap_err();
        assert!(matches!(err, GgufError::InvalidStreamUse(_)), "{err}");

        // Wrong size.
        let mut w = GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &decls).unwrap();
        let err = w.write_tensor("a.f32", &[0u8; 8]).unwrap_err();
        assert!(matches!(err, GgufError::TensorSizeMismatch { .. }), "{err}");

        // Extra payload past the declaration set.
        let mut w = GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &decls[..1]).unwrap();
        w.write_tensor("a.f32", &tensors[0].3).unwrap();
        let err = w.write_tensor("a.f32", &tensors[0].3).unwrap_err();
        assert!(matches!(err, GgufError::InvalidStreamUse(_)), "{err}");
    }

    #[test]
    fn stream_writer_finish_requires_every_payload() {
        let tensors = stream_parity_tensors();
        let mut w =
            GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &decls_of(&tensors)).unwrap();
        w.write_tensor("a.f32", &tensors[0].3).unwrap();
        assert_eq!(w.remaining(), tensors.len() - 1);
        let err = w.finish().unwrap_err();
        assert!(matches!(err, GgufError::InvalidStreamUse(_)), "{err}");
    }

    #[test]
    fn stream_writer_rejects_duplicate_and_overlong_declarations() {
        let dup = vec![
            GgufTensorDecl {
                name: "t".into(),
                dtype: GgmlType::F32,
                dimensions: vec![1],
            },
            GgufTensorDecl {
                name: "t".into(),
                dtype: GgmlType::F32,
                dimensions: vec![1],
            },
        ];
        let err = GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &dup).unwrap_err();
        assert!(matches!(err, GgufError::DuplicateTensor(_)), "{err}");

        let overlong = vec![GgufTensorDecl {
            name: "t".into(),
            dtype: GgmlType::F32,
            dimensions: vec![1, 1, 1, 1, 1],
        }];
        let err = GgufStreamWriter::begin(Vec::new(), &stream_metadata(), &overlong).unwrap_err();
        assert!(matches!(err, GgufError::TooManyDimensions(5)), "{err}");
    }
}
