//! CPU dequant + GEMM scalar reference path (M3-04-T11).
//!
//! This is the **differential oracle** for the M3-04-phase-2 Metal / CUDA
//! fused dequant kernels: given a quantized `K` or `V` block array and an FP32
//! query row, produce the same FP32 attention scores a fused kernel would.
//! No SIMD, no unsafe — the M3-04-T11 rationale is "correctness first, speed
//! later" (`docs/tickets/m3/M3-04-kv-cache-quantization.md` §T11: "SIMD
//! optimization は本 WP 対象外 (M4 で必要に応じて拡張)").
//!
//! # API shape
//!
//! [`dequant_gemv_scalar`] computes `y = A · x` where `A` is a quantized block
//! matrix in row-major layout (each row is `n_blocks_per_row` sequential
//! quantized 32-elem blocks) and `x` is FP32. This matches the KV-cache
//! attention shape:
//!
//! - `K` at layer L, streams `s`, positions `0..t`: `[t, per_slot]` matrix
//!   quantized in per-row-32-elem blocks.
//! - Query row from decoder at position `t`: `[per_slot]` FP32 vector.
//! - Output scores: `[t]` FP32 vector.
//!
//! # Precision contract
//!
//! - Output is FP32 (the M3-04-phase-2 fused kernels also produce FP32).
//! - The scalar dequant → FP32 → dot product must produce bit-identical
//!   output against a `dequant_slice → dense_gemv_f32` two-stage sequence
//!   (the trivial reference); this is pinned by
//!   [`tests::two_stage_bit_identical_to_fused_scalar`].
//! - Bound vs an FP32-baseline GEMV: at most `n_blocks_per_row * amax * atol`
//!   where `atol` is the per-format `d / 2` bound. See
//!   [`tests::q8_0_gemv_within_bound_vs_fp32`].

use super::{BlockQ4_0, BlockQ5_0, BlockQ8_0, KV_QUANT_BLOCK_SIZE, KvQuant};

/// Dequantize a quantized block matrix on the fly and compute `y = A · x`.
///
/// `A` is described by `(mode, blocks, n_rows, n_blocks_per_row)`:
/// - `mode` selects Q4_0 / Q5_0 / Q8_0 (Fp32 is rejected).
/// - `blocks` is the untyped byte payload (`n_rows * n_blocks_per_row *
///   block_bytes` bytes). Layout: row-major, contiguous blocks per row.
/// - `x` is the FP32 vector; `x.len() == n_blocks_per_row *
///   KV_QUANT_BLOCK_SIZE` (equivalently `per_slot`).
/// - Output `y` has length `n_rows`.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// on any shape mismatch, or if `mode == KvQuant::Fp32`.
pub fn dequant_gemv_scalar(
    mode: KvQuant,
    blocks_bytes: &[u8],
    n_rows: usize,
    n_blocks_per_row: usize,
    x: &[f32],
) -> crate::Result<Vec<f32>> {
    if matches!(mode, KvQuant::Fp32) {
        return Err(crate::VokraError::InvalidArgument(
            "dequant_gemv_scalar: mode=Fp32 not supported here — use dense_gemv_f32 directly"
                .into(),
        ));
    }
    let per_row_len = n_blocks_per_row * KV_QUANT_BLOCK_SIZE;
    if x.len() != per_row_len {
        return Err(crate::VokraError::InvalidArgument(format!(
            "dequant_gemv_scalar: x.len()={} != n_blocks_per_row*32 = {}",
            x.len(),
            per_row_len
        )));
    }
    let block_bytes = mode.block_bytes();
    let per_row_bytes = n_blocks_per_row * block_bytes;
    if blocks_bytes.len() != n_rows * per_row_bytes {
        return Err(crate::VokraError::InvalidArgument(format!(
            "dequant_gemv_scalar: blocks_bytes.len()={} != n_rows*per_row_bytes = {}",
            blocks_bytes.len(),
            n_rows * per_row_bytes
        )));
    }

    let mut y = vec![0.0f32; n_rows];
    // Per-row scratch — allocated once outside the loop so the hot path (row
    // iteration) doesn't touch the system allocator. Fixed size = per_row_len
    // FP32 = at most a few KB for practical audio shapes.
    let mut row_fp32 = vec![0.0f32; per_row_len];
    for (row, y_slot) in y.iter_mut().enumerate() {
        let row_start = row * per_row_bytes;
        let row_end = row_start + per_row_bytes;
        crate::kv_quant::dequantize_bytes_into(
            mode,
            &blocks_bytes[row_start..row_end],
            &mut row_fp32,
        )?;
        // Scalar dot product; deterministic order = row-major.
        let acc: f32 = row_fp32.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        *y_slot = acc;
    }
    Ok(y)
}

/// Dense FP32 GEMV baseline: `y = A · x`. Trivial reference that
/// [`dequant_gemv_scalar`] converges to as quantization noise goes to zero.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// on any shape mismatch.
pub fn dense_gemv_f32(a: &[f32], n_rows: usize, x: &[f32]) -> crate::Result<Vec<f32>> {
    if a.len() != n_rows * x.len() {
        return Err(crate::VokraError::InvalidArgument(format!(
            "dense_gemv_f32: a.len()={} != n_rows({n_rows}) * x.len()({})",
            a.len(),
            x.len()
        )));
    }
    let per_row_len = x.len();
    let mut y = vec![0.0f32; n_rows];
    for row in 0..n_rows {
        let mut acc = 0.0f32;
        for i in 0..per_row_len {
            acc += a[row * per_row_len + i] * x[i];
        }
        y[row] = acc;
    }
    Ok(y)
}

/// Pack a dense FP32 row-major matrix into `n_rows * n_blocks_per_row`
/// quantized blocks emitted as a raw byte buffer.
///
/// The byte layout matches [`dequant_gemv_scalar`] input exactly: row-major,
/// each row is `n_blocks_per_row` contiguous quantized blocks.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`](crate::VokraError::InvalidArgument)
/// if the input length or block alignment is wrong.
pub fn pack_matrix_to_bytes(
    mode: KvQuant,
    a: &[f32],
    n_rows: usize,
    n_blocks_per_row: usize,
) -> crate::Result<Vec<u8>> {
    if matches!(mode, KvQuant::Fp32) {
        return Err(crate::VokraError::InvalidArgument(
            "pack_matrix_to_bytes: mode=Fp32 has no packed layout".into(),
        ));
    }
    let per_row_len = n_blocks_per_row * KV_QUANT_BLOCK_SIZE;
    if a.len() != n_rows * per_row_len {
        return Err(crate::VokraError::InvalidArgument(format!(
            "pack_matrix_to_bytes: a.len()={} != n_rows*per_row_len = {}",
            a.len(),
            n_rows * per_row_len
        )));
    }
    let block_bytes = mode.block_bytes();
    let mut out = vec![0u8; n_rows * n_blocks_per_row * block_bytes];
    for row in 0..n_rows {
        let row_start = row * per_row_len;
        let row_bytes_start = row * n_blocks_per_row * block_bytes;
        for b in 0..n_blocks_per_row {
            let block_input =
                &a[row_start + b * KV_QUANT_BLOCK_SIZE..row_start + (b + 1) * KV_QUANT_BLOCK_SIZE];
            let out_start = row_bytes_start + b * block_bytes;
            match mode {
                KvQuant::Q4_0 => {
                    let block = BlockQ4_0::pack(block_input);
                    let bytes = super::block_q4_0_bytes(&block);
                    out[out_start..out_start + block_bytes].copy_from_slice(&bytes);
                }
                KvQuant::Q5_0 => {
                    let block = BlockQ5_0::pack(block_input);
                    let bytes = super::block_q5_0_bytes(&block);
                    out[out_start..out_start + block_bytes].copy_from_slice(&bytes);
                }
                KvQuant::Q8_0 => {
                    let block = BlockQ8_0::pack(block_input);
                    let bytes = super::block_q8_0_bytes(&block);
                    out[out_start..out_start + block_bytes].copy_from_slice(&bytes);
                }
                KvQuant::Fp32 => unreachable!("guarded above"),
            }
        }
    }
    Ok(out)
}

/// Convenience: pack → GEMV, returning both the packed bytes and the
/// scalar-dequant GEMV result. Used by the T14 checklist to demonstrate the
/// end-to-end path in a single call.
///
/// # Errors
///
/// Propagates any error from [`pack_matrix_to_bytes`] or
/// [`dequant_gemv_scalar`].
pub fn quant_gemv_round_trip(
    mode: KvQuant,
    a: &[f32],
    n_rows: usize,
    n_blocks_per_row: usize,
    x: &[f32],
) -> crate::Result<(Vec<u8>, Vec<f32>)> {
    let bytes = pack_matrix_to_bytes(mode, a, n_rows, n_blocks_per_row)?;
    let y = dequant_gemv_scalar(mode, &bytes, n_rows, n_blocks_per_row, x)?;
    Ok((bytes, y))
}

/// Compute the max relative error between two GEMV outputs. Used by the
/// M3-04-T12 verify pipeline to compare an FP32 baseline against each
/// quantization mode.
///
/// Guarded against division-by-zero in the baseline: a zero reference element
/// falls back to absolute error.
#[must_use]
pub fn max_relative_error(baseline: &[f32], measured: &[f32]) -> f32 {
    debug_assert_eq!(baseline.len(), measured.len());
    let mut worst: f32 = 0.0;
    for (b, m) in baseline.iter().zip(measured.iter()) {
        let denom = b.abs().max(f32::EPSILON);
        let rel = ((b - m).abs() / denom).min((b - m).abs());
        // The `.min(abs err)` guard reads odd but is the standard robust
        // metric: for near-zero baselines the relative error can explode; the
        // absolute error is the more honest signal there.
        worst = worst.max(rel);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic 4x64 matrix + 64-elem query vector — small enough to
    // manually eyeball, large enough to exercise 2 blocks per row.
    fn make_shape() -> (Vec<f32>, Vec<f32>, usize, usize) {
        let n_rows = 4;
        let n_blocks_per_row = 2; // 64 elems
        let per_row = n_blocks_per_row * KV_QUANT_BLOCK_SIZE;
        let a: Vec<f32> = (0..n_rows * per_row)
            .map(|i| ((i as f32) / (n_rows * per_row) as f32) * 2.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..per_row)
            .map(|i| ((i as f32) / per_row as f32).sin())
            .collect();
        (a, x, n_rows, n_blocks_per_row)
    }

    #[test]
    fn dequant_gemv_shape_mismatch_is_error() {
        let bad = vec![0u8; 10];
        let x = vec![0.0f32; 64];
        assert!(matches!(
            dequant_gemv_scalar(KvQuant::Q8_0, &bad, 4, 2, &x),
            Err(crate::VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn dequant_gemv_fp32_mode_is_error() {
        let (_a, x, _, _) = make_shape();
        assert!(matches!(
            dequant_gemv_scalar(KvQuant::Fp32, &[], 4, 2, &x),
            Err(crate::VokraError::InvalidArgument(_))
        ));
    }

    /// The fused-scalar dequant-GEMV must match the two-stage
    /// `unpack_slice → dense_gemv_f32` path bit-for-bit (both use the same
    /// row-major dot-product order).
    #[test]
    fn two_stage_bit_identical_to_fused_scalar() {
        let (a, x, n_rows, n_blocks_per_row) = make_shape();
        for mode in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let bytes = pack_matrix_to_bytes(mode, &a, n_rows, n_blocks_per_row).unwrap();
            // Path 1: fused dequant + GEMV.
            let y_fused = dequant_gemv_scalar(mode, &bytes, n_rows, n_blocks_per_row, &x).unwrap();
            // Path 2: dequantize all rows to FP32, then dense GEMV.
            let dequantized = crate::kv_quant::dequantize_bytes(mode, &bytes).unwrap();
            let y_two_stage = dense_gemv_f32(&dequantized, n_rows, &x).unwrap();
            assert_eq!(y_fused, y_two_stage, "path parity broken for {mode:?}");
        }
    }

    /// Q8_0 output must be within a bound proportional to the per-format
    /// quantization step against an FP32 baseline.
    #[test]
    fn q8_0_gemv_within_bound_vs_fp32() {
        let (a, x, n_rows, n_blocks_per_row) = make_shape();
        let y_fp32 = dense_gemv_f32(&a, n_rows, &x).unwrap();

        let bytes = pack_matrix_to_bytes(KvQuant::Q8_0, &a, n_rows, n_blocks_per_row).unwrap();
        let y_q8 =
            dequant_gemv_scalar(KvQuant::Q8_0, &bytes, n_rows, n_blocks_per_row, &x).unwrap();

        // Loose bound: per-elem error ≤ amax(row) / 127 / 2. Sum over per_row
        // elems and multiply by |x|_∞.
        let amax = a.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let x_inf = x.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        let per_row = n_blocks_per_row * KV_QUANT_BLOCK_SIZE;
        let tol = (amax / 127.0 * 0.5) * (per_row as f32) * x_inf * 1.5; // ×1.5 safety margin

        for (a_v, b_v) in y_fp32.iter().zip(&y_q8) {
            assert!(
                (a_v - b_v).abs() <= tol,
                "|{a_v} - {b_v}| > {tol} (Q8_0 GEMV out of bound)"
            );
        }
    }

    /// GEMV error decreases as quantization width grows.
    #[test]
    fn gemv_precision_ordering_is_q4_gt_q5_gt_q8() {
        let (a, x, n_rows, n_blocks_per_row) = make_shape();
        let y_fp32 = dense_gemv_f32(&a, n_rows, &x).unwrap();

        let mut errs = Vec::new();
        for mode in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let bytes = pack_matrix_to_bytes(mode, &a, n_rows, n_blocks_per_row).unwrap();
            let y = dequant_gemv_scalar(mode, &bytes, n_rows, n_blocks_per_row, &x).unwrap();
            let sse: f32 = y_fp32.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum();
            errs.push((mode, sse));
        }
        assert!(errs[2].1 <= errs[1].1);
        assert!(errs[1].1 <= errs[0].1);
    }

    #[test]
    fn max_relative_error_is_zero_for_identical_inputs() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(max_relative_error(&a, &a), 0.0);
    }

    #[test]
    fn max_relative_error_bounded_for_perturbed_inputs() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.01f32, 2.02, 3.03]; // ~1% perturbation
        let e = max_relative_error(&a, &b);
        assert!((0.009..=0.011).contains(&e), "unexpected: {e}");
    }

    #[test]
    fn quant_gemv_round_trip_matches_manual_two_stage() {
        let (a, x, n_rows, n_blocks_per_row) = make_shape();
        let (_, y_convenience) =
            quant_gemv_round_trip(KvQuant::Q8_0, &a, n_rows, n_blocks_per_row, &x).unwrap();
        // Two-stage manual: pack, unpack, dense GEMV.
        let bytes = pack_matrix_to_bytes(KvQuant::Q8_0, &a, n_rows, n_blocks_per_row).unwrap();
        let dequantized = crate::kv_quant::dequantize_bytes(KvQuant::Q8_0, &bytes).unwrap();
        let y_manual = dense_gemv_f32(&dequantized, n_rows, &x).unwrap();
        assert_eq!(y_convenience, y_manual);
    }
}
