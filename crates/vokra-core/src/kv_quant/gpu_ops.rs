//! GPU seam for fused KV cache dequantization + GEMV (M3-04 phase 2).
//!
//! # Purpose
//!
//! [`KvQuantDequantGemvOps`] is the trait every GPU backend (Metal, CUDA)
//! implements to consume a quantized KV block matrix *directly* — dequantizing
//! per-block on the fly inside the same kernel that computes the attention-score
//! `y = A · x` GEMV. The CPU differential oracle for this trait is
//! [`dequant_gemv_scalar`](crate::kv_quant::dequant_gemm::dequant_gemv_scalar),
//! and the GPU implementations must produce results within FP32 GEMV rounding
//! of it (see the parity tests in each backend crate).
//!
//! # Placement
//!
//! Kept in `vokra-core` (not the backend crates) so:
//! - The type surface is single-sourced — every backend implements the same
//!   trait against the same [`KvQuant`] enum and the same on-wire block layout
//!   ([`crate::kv_quant::dequantize_bytes`] uses the same input bytes).
//! - The `KvQuantDequantGemvOps` name is namable in a downstream generic
//!   function even when neither Metal nor CUDA is compiled in (the trait
//!   surface is unconditional; only the impls are backend-gated).
//! - Silent CPU fallback stays forbidden (FR-EX-08): a caller who wants CPU
//!   dispatch explicitly calls
//!   [`dequant_gemv_scalar`](crate::kv_quant::dequant_gemm::dequant_gemv_scalar);
//!   the GPU trait is *never* implemented for CPU.
//!
//! # Zero-dep + safe
//!
//! The trait itself is safe Rust with no dependencies (NFR-DS-02). Backend
//! implementations use the same `unsafe` FFI boundary as the rest of their
//! kernel surface — validated in the backend crate's `unsafe` policy, not
//! here.

use super::KvQuant;
use crate::error::Result;

/// GPU-side fused dequantization + row-wise GEMV.
///
/// # Contract (mirrors [`dequant_gemv_scalar`])
///
/// - `mode` selects `Q4_0` / `Q5_0` / `Q8_0`. `KvQuant::Fp32` is an explicit
///   [`crate::VokraError::InvalidArgument`] — the FP32 path uses
///   [`crate::kv_quant::dequant_gemm::dense_gemv_f32`] directly and never
///   routes through this trait.
/// - `blocks_bytes` is a row-major byte payload of size
///   `n_rows * n_blocks_per_row * mode.block_bytes()`. Each row is
///   `n_blocks_per_row` contiguous on-wire quant blocks in the same layout
///   [`crate::kv_quant::dequantize_bytes`] consumes; the GPU kernel dequantizes
///   one block at a time inside its GEMV row loop.
/// - `x.len() == n_blocks_per_row * 32` (equivalently `per_slot`, the FP32
///   query vector).
/// - Output is `Vec<f32>` of length `n_rows`, computed as
///   `y[i] = Σ_k dequant(row_i)[k] · x[k]`.
///
/// # Precision
///
/// The GPU output matches the CPU differential oracle within the FP32 GEMV
/// rounding bound. Backend parity tests pin this to `atol = 1e-4` for
/// `n_blocks_per_row <= 8` (typical Whisper `d_head = 64` = 2 blocks/row);
/// looser bounds are permitted for longer rows but must be documented on the
/// backend impl.
///
/// # Errors
///
/// - [`crate::VokraError::InvalidArgument`] on `mode == KvQuant::Fp32`, any
///   shape mismatch, or a byte-length that is not a whole multiple of
///   `n_rows * n_blocks_per_row * block_bytes`.
/// - [`crate::VokraError::BackendUnavailable`] on a device launch / allocation
///   failure (backends propagate their own driver errors).
pub trait KvQuantDequantGemvOps {
    /// GPU-side fused dequant + GEMV. See the trait docs for the contract.
    fn fused_dequant_gemv(
        &self,
        mode: KvQuant,
        blocks_bytes: &[u8],
        n_rows: usize,
        n_blocks_per_row: usize,
        x: &[f32],
    ) -> Result<Vec<f32>>;
}

/// Shape validation shared by every backend implementation. Returns
/// `(block_bytes, per_row_len)` on success so the backend does not have to
/// recompute them.
///
/// # Errors
///
/// Returns [`crate::VokraError::InvalidArgument`] on `mode == KvQuant::Fp32`,
/// `x.len() != n_blocks_per_row * 32`, or
/// `blocks_bytes.len() != n_rows * n_blocks_per_row * mode.block_bytes()`.
///
/// Kept here so the CPU oracle, the CUDA impl, and the Metal impl share the
/// exact same rejection rules — a shape a backend accepts is a shape the
/// oracle also accepts and vice versa.
pub fn validate_dequant_gemv(
    mode: KvQuant,
    blocks_bytes: &[u8],
    n_rows: usize,
    n_blocks_per_row: usize,
    x: &[f32],
) -> Result<(usize, usize)> {
    if matches!(mode, KvQuant::Fp32) {
        return Err(crate::VokraError::InvalidArgument(
            "fused_dequant_gemv: mode=Fp32 has no on-wire block layout; use dense_gemv_f32 instead"
                .into(),
        ));
    }
    let per_row_len = n_blocks_per_row * super::KV_QUANT_BLOCK_SIZE;
    if x.len() != per_row_len {
        return Err(crate::VokraError::InvalidArgument(format!(
            "fused_dequant_gemv: x.len()={} != n_blocks_per_row({n_blocks_per_row})*32 = {per_row_len}",
            x.len()
        )));
    }
    let block_bytes = mode.block_bytes();
    let expected_bytes = n_rows
        .checked_mul(n_blocks_per_row)
        .and_then(|v| v.checked_mul(block_bytes))
        .ok_or_else(|| {
            crate::VokraError::InvalidArgument(
                "fused_dequant_gemv: byte-length overflow".to_owned(),
            )
        })?;
    if blocks_bytes.len() != expected_bytes {
        return Err(crate::VokraError::InvalidArgument(format!(
            "fused_dequant_gemv: blocks_bytes.len()={} != n_rows({n_rows})*n_blocks_per_row({n_blocks_per_row})*block_bytes({block_bytes}) = {expected_bytes}",
            blocks_bytes.len()
        )));
    }
    Ok((block_bytes, per_row_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VokraError;

    #[test]
    fn validate_rejects_fp32_mode() {
        let err = validate_dequant_gemv(KvQuant::Fp32, &[], 0, 0, &[]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_bad_x_len() {
        let bytes = vec![0u8; 4 * 2 * 34]; // Q8_0: 4 rows × 2 bpr × 34 bytes.
        let x = vec![0.0f32; 63]; // wrong (should be 64).
        let err = validate_dequant_gemv(KvQuant::Q8_0, &bytes, 4, 2, &x).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_bad_bytes_len() {
        let bytes = vec![0u8; 100]; // wrong (should be 272).
        let x = vec![0.0f32; 64];
        let err = validate_dequant_gemv(KvQuant::Q8_0, &bytes, 4, 2, &x).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_accepts_matching_shapes() {
        // Q4_0: 4 rows × 2 bpr × 18 bytes = 144.
        let bytes = vec![0u8; 144];
        let x = vec![0.0f32; 64];
        let (bb, prl) = validate_dequant_gemv(KvQuant::Q4_0, &bytes, 4, 2, &x).unwrap();
        assert_eq!(bb, 18);
        assert_eq!(prl, 64);

        // Q5_0: 4 rows × 2 bpr × 22 bytes = 176.
        let bytes = vec![0u8; 176];
        let (bb, prl) = validate_dequant_gemv(KvQuant::Q5_0, &bytes, 4, 2, &x).unwrap();
        assert_eq!(bb, 22);
        assert_eq!(prl, 64);

        // Q8_0: 4 rows × 2 bpr × 34 bytes = 272.
        let bytes = vec![0u8; 272];
        let (bb, prl) = validate_dequant_gemv(KvQuant::Q8_0, &bytes, 4, 2, &x).unwrap();
        assert_eq!(bb, 34);
        assert_eq!(prl, 64);
    }

    /// Confirms the trait object is object-safe (compilable behind `&dyn Trait`).
    /// The purpose is to future-proof for a `Box<dyn KvQuantDequantGemvOps>` in
    /// a downstream Session builder without pulling either backend in.
    #[test]
    fn trait_is_object_safe() {
        fn takes_dyn(_ops: &dyn KvQuantDequantGemvOps) {}
        // No real backend on the CPU CI leg, so we just prove the trait can be
        // named behind `dyn`. If this line fails to compile, the trait has
        // become non-object-safe (a design regression).
        let _ = takes_dyn;
    }
}
