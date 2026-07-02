//! GGUF tensor dtype and tensor-info descriptors.
//!
//! Only the dense float types `F32` and `F16` are accepted in M0. The GGUF /
//! ggml type tags are part of the on-disk format: `GGML_TYPE_F32 = 0`,
//! `GGML_TYPE_F16 = 1` (source:
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>).
//!
//! K-quant / i-quant dense-block types (`Q4_K`, `Q5_K`, `Q6_K`, …) are **out
//! of scope for M0**: their GGUF direct-load path is FR-LD-07, owned by
//! M1-02. A tensor declaring any tag other than `0` or `1` is rejected with
//! [`GgufError::UnsupportedDtype`] rather than silently mishandled.

use super::GgufError;

/// Maximum tensor rank accepted by the loader.
///
/// The GGUF spec states tensors currently have at most 4 dimensions; a
/// declaration exceeding this is rejected as malformed input (NFR-RL-07).
pub const MAX_TENSOR_DIMS: usize = 4;

/// Tensor element type, restricted to the dense float types supported in M0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// IEEE-754 32-bit float, ggml type tag `0`. 4 bytes per element.
    F32 = 0,
    /// IEEE-754 16-bit float, ggml type tag `1`. 2 bytes per element.
    F16 = 1,
}

impl GgmlType {
    /// Converts an on-disk ggml type tag to a [`GgmlType`].
    ///
    /// Returns [`GgufError::UnsupportedDtype`] for any tag other than the two
    /// dense float types supported in M0 (this includes every quantized type,
    /// deferred to M1-02 / FR-LD-07).
    pub fn from_tag(tag: u32) -> Result<Self, GgufError> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => Err(GgufError::UnsupportedDtype(other)),
        }
    }

    /// Returns the on-disk ggml type tag for this dtype.
    pub fn tag(self) -> u32 {
        self as u32
    }

    /// Size in bytes of a single element of this dtype.
    pub fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Descriptor for one tensor in a GGUF file.
///
/// [`offset`](Self::offset) is relative to the start of the tensor-data region
/// (after the header, metadata, tensor infos and alignment padding) and is a
/// multiple of the file alignment, exactly as stored on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// Tensor name (UTF-8).
    pub name: String,
    /// Dimensions, innermost first, as stored on disk (`n_dims` entries).
    pub dimensions: Vec<u64>,
    /// Element type.
    pub dtype: GgmlType,
    /// Byte offset from the start of the tensor-data region.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements (product of all dimensions).
    ///
    /// A rank-0 tensor (no dimensions) has a single element, matching ggml.
    /// Returns [`GgufError::Overflow`] if the product overflows `u64`.
    pub fn element_count(&self) -> Result<u64, GgufError> {
        let mut count: u64 = 1;
        for &dim in &self.dimensions {
            count = count.checked_mul(dim).ok_or(GgufError::Overflow)?;
        }
        Ok(count)
    }

    /// Total size of the tensor payload in bytes for the current dense dtype.
    ///
    /// Returns [`GgufError::Overflow`] if the computation overflows `u64`.
    pub fn byte_len(&self) -> Result<u64, GgufError> {
        let elems = self.element_count()?;
        let elem_size = self.dtype.element_size() as u64;
        elems.checked_mul(elem_size).ok_or(GgufError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, GgufTensorInfo};
    use crate::gguf::{GgufBuilder, GgufFile};

    #[test]
    fn rank0_scalar_element_count_and_byte_len() {
        // A rank-0 (no-dimension) tensor is a single scalar element, matching
        // ggml: element_count is the empty product 1, byte_len is one F32.
        let scalar = GgufTensorInfo {
            name: "s".to_owned(),
            dimensions: vec![],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(scalar.element_count().unwrap(), 1);
        assert_eq!(scalar.byte_len().unwrap(), 4);
    }

    #[test]
    fn rank0_scalar_tensor_roundtrips() {
        // Writing and reading a scalar F32 tensor preserves both its 4 payload
        // bytes and its empty-dimensions shape.
        let mut b = GgufBuilder::new();
        b.add_tensor("s", GgmlType::F32, vec![], vec![7, 8, 9, 10])
            .expect("scalar is a valid 4-byte F32 payload");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(file.tensor_data("s").unwrap(), [7u8, 8, 9, 10].as_slice());
        assert!(file.tensor_info("s").unwrap().dimensions.is_empty());
    }
}
