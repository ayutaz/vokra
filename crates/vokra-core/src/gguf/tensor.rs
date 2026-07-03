//! GGUF tensor dtype and tensor-info descriptors.
//!
//! The dense float types `F32` and `F16` (M0) plus the K-quant super-block
//! types `Q4_K`, `Q5_K`, `Q6_K` (M1-02, FR-LD-07 / FR-QT-01) are accepted.
//! The GGUF / ggml type tags are part of the on-disk format: `GGML_TYPE_F32 =
//! 0`, `GGML_TYPE_F16 = 1`, `GGML_TYPE_Q4_K = 12`, `GGML_TYPE_Q5_K = 13`,
//! `GGML_TYPE_Q6_K = 14` (source: ggml-org/ggml `ggml.h` `enum ggml_type` and
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>).
//!
//! The K-quant *on-disk block layout* is transcribed from ggml `k_quants`
//! (ggml / llama.cpp are MIT); this is a data-format spec, not a code copy —
//! the whisper.cpp-style native re-implementation pattern (CLAUDE.md). Dequant
//! lives in the scalar, `unsafe`-free [`quant`](super::quant) module; SIMD
//! acceleration is a documented follow-up in `vokra-backend-cpu`.
//!
//! IQ2 / other i-quant families remain out of scope (FR-QT-01 marks them
//! 極小デバイス用). A tensor declaring any tag other than `0`, `1`, `12`,
//! `13`, `14` is rejected with [`GgufError::UnsupportedDtype`] rather than
//! silently mishandled.

use super::GgufError;

/// Maximum tensor rank accepted by the loader.
///
/// The GGUF spec states tensors currently have at most 4 dimensions; a
/// declaration exceeding this is rejected as malformed input (NFR-RL-07).
pub const MAX_TENSOR_DIMS: usize = 4;

/// K-quant super-block size in elements (`QK_K` in ggml `k_quants.h`).
///
/// Every K-quant type packs its quants into fixed super-blocks of this many
/// elements, so a K-quant tensor's element count must be a whole multiple of
/// `QK_K`.
pub const QK_K: usize = 256;

/// Tensor element type: the dense float types plus the K-quant super-block
/// types (`Q4_K` / `Q5_K` / `Q6_K`).
///
/// Discriminants are the on-disk ggml type tags and are load-bearing (written
/// to and read from the file verbatim); they must never be reordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// IEEE-754 32-bit float, ggml type tag `0`. 4 bytes per element.
    F32 = 0,
    /// IEEE-754 16-bit float, ggml type tag `1`. 2 bytes per element.
    F16 = 1,
    /// 4-bit K-quant, ggml type tag `12`. 256-element super-block, 144 bytes.
    Q4K = 12,
    /// 5-bit K-quant, ggml type tag `13`. 256-element super-block, 176 bytes.
    Q5K = 13,
    /// 6-bit K-quant, ggml type tag `14`. 256-element super-block, 210 bytes.
    Q6K = 14,
}

impl GgmlType {
    /// Converts an on-disk ggml type tag to a [`GgmlType`].
    ///
    /// Returns [`GgufError::UnsupportedDtype`] for any tag other than the dense
    /// float types (`0`, `1`) and the K-quant types (`12`, `13`, `14`). Other
    /// quantized families (IQ2, Q2_K, Q8_0, …) are intentionally unsupported.
    pub fn from_tag(tag: u32) -> Result<Self, GgufError> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            other => Err(GgufError::UnsupportedDtype(other)),
        }
    }

    /// Returns the on-disk ggml type tag for this dtype.
    pub fn tag(self) -> u32 {
        self as u32
    }

    /// Number of elements in one storage block.
    ///
    /// `1` for the dense float types (each element is independently sized);
    /// [`QK_K`] (256) for every K-quant, which stores quants in fixed
    /// super-blocks.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4K | Self::Q5K | Self::Q6K => QK_K,
        }
    }

    /// Size in bytes of one storage block of this dtype.
    ///
    /// For the dense types this is the element size (4 / 2 bytes). For the
    /// K-quants it is the exact `block_q*_K` struct size from ggml
    /// `k_quants.h`: Q4_K = 144, Q5_K = 176, Q6_K = 210 bytes per 256-element
    /// super-block. These are pinned by the dequant round-trip tests.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// Byte length needed to store `n_elements` values of this dtype.
    ///
    /// For dense types this is `n_elements * type_size`. For K-quants it is
    /// `(n_elements / block_size) * type_size`, and the element count MUST be a
    /// whole number of super-blocks: a partial block is malformed and yields
    /// [`GgufError::BlockSizeMisaligned`]. Returns [`GgufError::Overflow`] on
    /// `u64` overflow.
    pub fn payload_size(self, n_elements: u64) -> Result<u64, GgufError> {
        let block = self.block_size() as u64;
        if n_elements % block != 0 {
            return Err(GgufError::BlockSizeMisaligned {
                dtype: self.tag(),
                elements: n_elements,
                block_size: self.block_size(),
            });
        }
        (n_elements / block)
            .checked_mul(self.type_size() as u64)
            .ok_or(GgufError::Overflow)
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

    /// Total size of the tensor payload in bytes for this dtype.
    ///
    /// Block-aware (see [`GgmlType::payload_size`]): dense types are
    /// `elements * type_size`, K-quants are `(elements / block_size) *
    /// type_size`. Returns [`GgufError::BlockSizeMisaligned`] if a K-quant
    /// element count is not a whole number of super-blocks, or
    /// [`GgufError::Overflow`] on `u64` overflow.
    pub fn byte_len(&self) -> Result<u64, GgufError> {
        self.dtype.payload_size(self.element_count()?)
    }
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, GgufTensorInfo};
    use crate::gguf::{GgufBuilder, GgufError, GgufFile};

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
    fn kquant_block_sizes_match_ggml_k_quants() {
        // Block struct sizes transcribed from ggml k_quants.h; pinned here so a
        // typo in type_size is caught independently of the dequant tests.
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q5K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.type_size(), 144);
        assert_eq!(GgmlType::Q5K.type_size(), 176);
        assert_eq!(GgmlType::Q6K.type_size(), 210);
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F16.block_size(), 1);
    }

    #[test]
    fn kquant_byte_len_is_block_aware() {
        // A 256-element Q4_K tensor is exactly one super-block = 144 bytes; a
        // 512-element one is two blocks = 288. F32/F16 stay byte-identical to
        // the old flat formula (block_size == 1).
        let one_block = GgufTensorInfo {
            name: "q".to_owned(),
            dimensions: vec![256],
            dtype: GgmlType::Q4K,
            offset: 0,
        };
        assert_eq!(one_block.element_count().unwrap(), 256);
        assert_eq!(one_block.byte_len().unwrap(), 144);

        let two_block = GgufTensorInfo {
            name: "q".to_owned(),
            dimensions: vec![2, 256],
            dtype: GgmlType::Q6K,
            offset: 0,
        };
        assert_eq!(two_block.byte_len().unwrap(), 2 * 210);
    }

    #[test]
    fn kquant_partial_block_is_block_size_misaligned() {
        // 100 is not a multiple of QK_K (256): a partial K-quant block is
        // malformed and must not silently truncate.
        let bad = GgufTensorInfo {
            name: "q".to_owned(),
            dimensions: vec![100],
            dtype: GgmlType::Q5K,
            offset: 0,
        };
        assert!(matches!(
            bad.byte_len(),
            Err(GgufError::BlockSizeMisaligned {
                block_size: 256,
                elements: 100,
                ..
            })
        ));
    }

    #[test]
    fn dtype_tag_roundtrips_for_all_supported_types() {
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
        ] {
            assert_eq!(GgmlType::from_tag(ty.tag()).unwrap(), ty);
        }
        // A neighbouring quantized tag we do NOT support (Q3_K = 11).
        assert!(matches!(
            GgmlType::from_tag(11),
            Err(GgufError::UnsupportedDtype(11))
        ));
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
