//! Scalar, `unsafe`-free dequantization of GGUF tensor payloads to `f32`.
//!
//! This is the single canonical decode path (FR-LD-07 / FR-QT-01): dense
//! `F32` / `F16` and the K-quant super-block types `Q4_K` / `Q5_K` / `Q6_K`
//! all resolve through [`dequantize`], so native models decode weights once
//! through one place instead of open-coding per-dtype byte loops.
//!
//! # Placement (deliberate)
//!
//! `vokra-core` is `unsafe_code = "deny"`, so this module is a *scalar, safe*
//! reference implementation using bounds-checked slice indexing. K-quant
//! dequant is memory-bound, so scalar is adequate for M1 RTF; a SIMD-accelerated
//! path is a documented follow-up in `vokra-backend-cpu` (the `unsafe`-allowed
//! crate) and MUST stay bit-identical to this reference.
//!
//! # On-disk layout provenance
//!
//! The K-quant `block_q*_K` super-block layouts are transcribed from ggml
//! `k_quants.h` / `dequantize_row_q*_K` (ggml / llama.cpp are MIT). This is a
//! data-format specification, not a code copy — the exact byte layout is what
//! the format *is*, and the per-format modules pin it with in-crate analytic
//! oracles (closed-form super-blocks) so correctness never depends on an
//! external reference file.

use super::GgufError;
use super::tensor::{GgmlType, QK_K};

mod q4_k;
mod q5_k;
mod q6_k;

/// Decodes a tensor payload of the given dtype into owned `f32` values.
///
/// `bytes` must be exactly [`GgmlType::payload_size`] for `n_elements` of
/// `dtype` (the GGUF reader guarantees this for parsed tensors); a mismatch is
/// rejected with [`GgufError::TensorSizeMismatch`] rather than panicking. The
/// returned vector has exactly `n_elements` entries.
pub fn dequantize(dtype: GgmlType, bytes: &[u8], n_elements: usize) -> Result<Vec<f32>, GgufError> {
    let expected = dtype.payload_size(n_elements as u64)?;
    if bytes.len() as u64 != expected {
        return Err(GgufError::TensorSizeMismatch {
            name: format!("<dequant {}>", dtype.tag()),
            expected,
            actual: bytes.len() as u64,
        });
    }
    Ok(match dtype {
        GgmlType::F32 => decode_f32(bytes),
        GgmlType::F16 => decode_f16(bytes),
        GgmlType::Q4K => q4_k::dequantize(bytes, n_elements),
        GgmlType::Q5K => q5_k::dequantize(bytes, n_elements),
        GgmlType::Q6K => q6_k::dequantize(bytes, n_elements),
    })
}

/// Decodes a little-endian `F32` payload (length already validated).
fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decodes a little-endian `F16` payload (length already validated).
fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// IEEE-754 half → single precision (safe, exact; handles subnormals, inf and
/// NaN). Shared by the dense `F16` path and the K-quant super-block scales,
/// which store `d` / `dmin` as `ggml_half` (`f16`).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };
    match exp {
        0 => sign_f * (mant as f32) * 2.0f32.powi(-24), // subnormal / zero
        0x1f => {
            if mant == 0 {
                sign_f * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign_f * (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15),
    }
}

/// Number of super-blocks spanning `n_elements` K-quant values.
///
/// The caller guarantees `n_elements` is a whole multiple of [`QK_K`] (the
/// dispatch validates the byte length against [`GgmlType::payload_size`], which
/// enforces exactly this), so integer division is exact.
#[inline]
fn n_blocks(n_elements: usize) -> usize {
    n_elements / QK_K
}

/// Unpacks the 6-bit sub-scale `d` and sub-min `m` for sub-block `j` (0..8)
/// from a 12-byte `scales` array — ggml `get_scale_min_k4`. Shared by the
/// `Q4_K` and `Q5_K` decoders, which pack their sub-scales identically.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_matches_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn dense_f32_roundtrips_through_dispatch() {
        let vals = [1.0f32, -2.5, 0.0, 3.25];
        let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(dequantize(GgmlType::F32, &bytes, 4).unwrap(), vals);
    }

    #[test]
    fn dense_f16_decodes_through_dispatch() {
        // 1.0, 2.0, -2.0 in half precision.
        let bytes: Vec<u8> = [0x3C00u16, 0x4000, 0xC000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        assert_eq!(
            dequantize(GgmlType::F16, &bytes, 3).unwrap(),
            vec![1.0, 2.0, -2.0]
        );
    }

    #[test]
    fn wrong_payload_length_is_rejected_not_panicked() {
        // One Q4_K block needs 144 bytes; hand a short buffer and require a
        // clean error rather than an out-of-bounds slice.
        let err = dequantize(GgmlType::Q4K, &[0u8; 100], 256).unwrap_err();
        assert!(matches!(err, GgufError::TensorSizeMismatch { .. }));
    }

    #[test]
    fn partial_block_element_count_is_rejected() {
        let err = dequantize(GgmlType::Q6K, &[0u8; 210], 200).unwrap_err();
        assert!(matches!(err, GgufError::BlockSizeMisaligned { .. }));
    }

    #[test]
    fn all_zero_kquant_blocks_decode_to_zeros() {
        // d/dmin/scales/quants all zero => y = 0 for every K-quant format.
        for (dtype, tsize) in [
            (GgmlType::Q4K, 144),
            (GgmlType::Q5K, 176),
            (GgmlType::Q6K, 210),
        ] {
            let out = dequantize(dtype, &vec![0u8; tsize], 256).unwrap();
            assert_eq!(out.len(), 256);
            assert!(out.iter().all(|&v| v == 0.0), "{dtype:?} not all zero");
        }
    }
}
