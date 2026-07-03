//! GGUF model container: direct loading (reader) and serialization (writer),
//! plus the `vokra.*` metadata chunk namespace (M0-03).
//!
//! # What this module is
//!
//! Vokra loads models from **GGUF** directly (FR-LD-01), never from ONNX at
//! runtime (FR-LD-05; ONNX is handled only by the offline conversion tool).
//! This module provides:
//!
//! - [`GgufFile`] — parse a GGUF file and lend tensor payloads as zero-copy
//!   `&[u8]` slices (see the [`reader`] module docs for the std-I/O-vs-mmap
//!   trade-off in a crate that forbids `unsafe`);
//! - [`GgufBuilder`] — serialize metadata and tensors back to GGUF, used by
//!   the offline conversion tool and by round-trip tests;
//! - [`FrontendSpec`] — the typed view of the `vokra.frontend.*` chunk;
//! - [`chunks`] — the `vokra.*` key namespace specification (M0-03-T08).
//!
//! # On-disk format (GGUF v3, little-endian)
//!
//! Verified against ggml-org/ggml `docs/gguf.md`:
//!
//! 1. magic `GGUF` (`0x47 0x47 0x55 0x46`), then `version: u32` (must be 3),
//!    `tensor_count: u64`, `metadata_kv_count: u64`;
//! 2. `metadata_kv_count` entries of `{ key: gguf_string, value }` where a
//!    `gguf_string` is `{ len: u64, bytes: [u8; len] }` (UTF-8) and a value is
//!    `{ type: u32 }` followed by its payload (see [`value`]);
//! 3. `tensor_count` entries of
//!    `{ name: gguf_string, n_dims: u32, dims: [u64; n_dims], type: u32,
//!    offset: u64 }`;
//! 4. zero padding up to [`DEFAULT_ALIGNMENT`] (or `general.alignment`), then
//!    the tensor data, each tensor starting at an alignment multiple.
//!
//! # Scope (M0)
//!
//! Dense `F32`/`F16` tensors only; K-quant direct load is FR-LD-07 (M1-02).
//! `frontend_spec` is read/written here but **not inspected** — the bit-exact
//! match check is FR-LD-03 (M1-03). See [`chunks`] for the full scope note.

mod reader;
mod writer;

pub mod chunks;
pub mod frontend_spec;
pub mod quant;
pub mod tensor;
pub mod value;

pub use frontend_spec::FrontendSpec;
pub use reader::GgufFile;
pub use tensor::{GgmlType, GgufTensorInfo};
pub use value::{GgufArray, GgufMetadataValue, GgufValueType};
pub use writer::GgufBuilder;

use std::fmt;

/// GGUF magic bytes: the ASCII string `GGUF`.
pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// The only GGUF format version this runtime reads or writes.
///
/// Versions 1 and 2 (which differ in count widths / endianness support) are
/// rejected with [`GgufError::UnsupportedVersion`]; adding them is a
/// deliberate future decision, not an accident.
pub const GGUF_VERSION: u32 = 3;

/// Default tensor-data alignment when `general.alignment` is absent (per the
/// GGUF spec).
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` must be a power of two. Returns [`GgufError::Overflow`] if rounding
/// would overflow `u64`.
pub(crate) fn align_up(value: u64, align: u64) -> Result<u64, GgufError> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let rem = value % align;
    if rem == 0 {
        Ok(value)
    } else {
        value.checked_add(align - rem).ok_or(GgufError::Overflow)
    }
}

/// Errors produced while reading or writing GGUF files.
///
/// Every variant is a *recoverable* error: the loader never panics on
/// malformed input (NFR-RL-07). At the public API boundary this converts into
/// [`VokraError::ModelLoad`](crate::VokraError::ModelLoad) (or
/// [`VokraError::Io`](crate::VokraError::Io) for the I/O case).
#[derive(Debug)]
#[non_exhaustive]
pub enum GgufError {
    /// The first four bytes were not the `GGUF` magic.
    BadMagic([u8; 4]),
    /// The format version is not [`GGUF_VERSION`].
    UnsupportedVersion(u32),
    /// The file ended before a required field could be read (`ctx` names it).
    Truncated(&'static str),
    /// A string field was not valid UTF-8.
    InvalidString(std::str::Utf8Error),
    /// A boolean field held a byte other than 0 or 1.
    InvalidBool(u8),
    /// A metadata value type tag was outside the range `0..=12`.
    UnsupportedValueType(u32),
    /// A tensor declared a ggml type tag Vokra does not load: the accepted set
    /// is `F32` (0), `F16` (1) and the K-quants `Q4_K` (12) / `Q5_K` (13) /
    /// `Q6_K` (14). Other quantized families (IQ2, Q2_K, Q8_0, …) are
    /// intentionally unsupported.
    UnsupportedDtype(u32),
    /// A quantized tensor's element count was not a whole multiple of its
    /// block size (a K-quant row not divisible by [`tensor::QK_K`] = 256).
    /// K-quants are stored as fixed-size super-blocks, so a partial block is
    /// malformed.
    BlockSizeMisaligned {
        /// The ggml dtype tag whose block size was violated.
        dtype: u32,
        /// The element count that was not a whole number of blocks.
        elements: u64,
        /// The dtype's block size in elements.
        block_size: usize,
    },
    /// A tensor requested by name (e.g. via [`GgufFile::tensor_f32`]) was not
    /// present in the file.
    MissingTensor(String),
    /// A tensor declared more than [`tensor::MAX_TENSOR_DIMS`] dimensions.
    TooManyDimensions(usize),
    /// Two tensors shared the same name.
    DuplicateTensor(String),
    /// A tensor's data range fell outside the file (payload names it).
    OffsetOutOfBounds(String),
    /// A tensor offset was not a multiple of the file alignment.
    UnalignedTensorOffset {
        /// Name of the offending tensor.
        tensor: String,
        /// The declared offset.
        offset: u64,
        /// The file alignment it violated.
        alignment: u64,
    },
    /// An arithmetic computation on sizes/offsets overflowed `u64`.
    Overflow,
    /// An alignment value was zero or not a power of two.
    InvalidAlignment(u64),
    /// Array metadata nested deeper than the decoder accepts.
    ArrayTooDeep(usize),
    /// A tensor's supplied payload length did not match its shape and dtype.
    TensorSizeMismatch {
        /// Name of the offending tensor.
        name: String,
        /// Byte length implied by shape and dtype.
        expected: u64,
        /// Byte length actually supplied.
        actual: u64,
    },
    /// A required metadata key was absent (e.g. a `vokra.frontend.*` field).
    MissingKey(String),
    /// A metadata key held a value of an unexpected type.
    WrongType {
        /// The metadata key.
        key: String,
        /// The value type that was expected.
        expected: value::GgufValueType,
    },
    /// An underlying I/O error (only from [`GgufFile::open`]).
    Io(std::io::Error),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(m) => write!(f, "bad GGUF magic: {m:02x?} (expected \"GGUF\")"),
            Self::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported GGUF version {v} (only {GGUF_VERSION} is read)"
                )
            }
            Self::Truncated(ctx) => write!(f, "file truncated while reading {ctx}"),
            Self::InvalidString(e) => write!(f, "invalid UTF-8 in GGUF string: {e}"),
            Self::InvalidBool(b) => write!(f, "invalid GGUF bool byte {b} (expected 0 or 1)"),
            Self::UnsupportedValueType(t) => write!(f, "unsupported metadata value type tag {t}"),
            Self::UnsupportedDtype(t) => {
                write!(
                    f,
                    "unsupported tensor dtype tag {t} \
                     (accepted: F32=0, F16=1, Q4_K=12, Q5_K=13, Q6_K=14)"
                )
            }
            Self::BlockSizeMisaligned {
                dtype,
                elements,
                block_size,
            } => write!(
                f,
                "tensor element count {elements} is not a multiple of block size \
                 {block_size} (dtype tag {dtype})"
            ),
            Self::MissingTensor(name) => write!(f, "tensor `{name}` not found in GGUF"),
            Self::TooManyDimensions(n) => write!(f, "tensor has too many dimensions: {n}"),
            Self::DuplicateTensor(name) => write!(f, "duplicate tensor name: {name}"),
            Self::OffsetOutOfBounds(name) => {
                write!(f, "tensor data out of bounds: {name}")
            }
            Self::UnalignedTensorOffset {
                tensor,
                offset,
                alignment,
            } => write!(
                f,
                "tensor `{tensor}` offset {offset} is not a multiple of alignment {alignment}"
            ),
            Self::Overflow => write!(f, "size/offset computation overflowed"),
            Self::InvalidAlignment(a) => {
                write!(f, "invalid alignment {a} (must be a non-zero power of two)")
            }
            Self::ArrayTooDeep(d) => write!(f, "metadata array nested too deep ({d} levels)"),
            Self::TensorSizeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor `{name}` payload is {actual} bytes but shape/dtype imply {expected}"
            ),
            Self::MissingKey(k) => write!(f, "missing required metadata key `{k}`"),
            Self::WrongType { key, expected } => {
                write!(
                    f,
                    "metadata key `{key}` has wrong type (expected {expected:?})"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::InvalidString(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GgufError> for crate::VokraError {
    fn from(e: GgufError) -> Self {
        match e {
            // Preserve the I/O source chain at the public boundary.
            GgufError::Io(io) => crate::VokraError::Io(io),
            other => crate::VokraError::ModelLoad(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VokraError;
    use std::io;

    #[test]
    fn gguf_error_maps_to_vokra_error_at_boundary() {
        // Every non-Io variant collapses to ModelLoad(string) (FR-API-02)...
        let mapped = VokraError::from(GgufError::BadMagic([0; 4]));
        assert!(matches!(mapped, VokraError::ModelLoad(_)));
        // ...but Io is routed to Io so callers keep the io::ErrorKind source.
        let io_err = io::Error::new(io::ErrorKind::NotFound, "x");
        let mapped = VokraError::from(GgufError::Io(io_err));
        assert!(matches!(mapped, VokraError::Io(_)));
    }

    #[test]
    fn open_nonexistent_path_is_io_error() {
        // `open` is the only Io-producing entry point; a missing file must
        // surface GgufError::Io rather than a parse error.
        let err = GgufFile::open("/no/such/vokra/file.gguf").unwrap_err();
        assert!(matches!(err, GgufError::Io(_)));
    }
}
