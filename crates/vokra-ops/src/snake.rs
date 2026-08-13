//! Snake activation (Ziyin et al. 2020) — per-channel closed-form periodic
//! activation used by every audio-dialect vocoder that follows the BigVGAN /
//! HiFTNet / Kokoro-82M lineage.
//!
//! # Semantics
//!
//! For a `[channels, time]` row-major (channel-outer) FP32 tensor `x`, and a
//! per-channel `alpha` vector of length `channels`, the closed form
//! (upstream `snake` in
//! [`activations.py:7-59`](https://github.com/NVIDIA/BigVGAN) L52-54,
//! CosyVoice2 `cosyvoice.transformer.activation.Snake`, StyleTTS 2 iSTFTNet
//! generator) is:
//!
//! ```text
//! y[c, t] = x[c, t] + (1 / (alpha[c] + eps)) * sin(alpha[c] * x[c, t])^2
//! ```
//!
//! with `eps = 1e-9` (upstream `no_div_by_zero`), matching
//! [`crate::hiftnet::Snake::forward_in_place`] under `alpha_logscale = false`
//! and the private `kokoro::nn::snake_activation` helper in vokra-models
//! **bit-for-bit** (same ordering, same eps, same primitives).
//!
//! This module exposes the **stateless out-of-place** free function
//! [`snake_activation_f32`], which is the shape the
//! [`vokra_models::compute::Compute`] seam dispatches through (mirroring the
//! silu / gelu / softmax family — read `x`, write `out`). The existing
//! stateful [`crate::hiftnet::Snake`] type (with optional `alpha_logscale`,
//! plus SnakeBeta in `bigvgan_generator`) is unchanged — this module is a
//! narrower, lower-level entry point for a GPU dispatch (Metal / CUDA / etc.)
//! that never rewrites x in place and never keeps an owned `alpha` around.
//!
//! # Not `alpha_logscale`, not `SnakeBeta`
//!
//! - `alpha_logscale = true` (HiFTNet `Snake(..., True)`) is an
//!   **upstream-side transformation** (`alpha_eff = exp(alpha_raw)`) applied
//!   before the same core formula. The converter can pre-exponentiate stored
//!   log-α weights and hand this function the already-effective vector; no
//!   need to branch in the hot path.
//! - `SnakeBeta` (`y = x + (1/(beta+eps)) * sin(alpha*x)^2`, two per-channel
//!   vectors) is a different closed form and a distinct op signature. It is
//!   provided by [`crate::bigvgan_generator::SnakeBeta`] and is intentionally
//!   NOT wired through this module — the two are separate consumers with
//!   different weight shapes.
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Every shape mismatch raises [`VokraError::InvalidArgument`] — a wrong
//! `alpha.len()` or `x.len()` / `out.len()` is loud, not silently clamped.
//! `channels == 0` or `time == 0` is accepted as a no-op (empty tensor —
//! upstream models never call with a zero-shape input, but the empty case is
//! well-defined and cheap to allow so callers do not need a special-case
//! branch upstream).

use vokra_core::{Result, VokraError};

/// Upstream `no_div_by_zero` constant (`activations.py:97` and every
/// downstream port: HiFTNet, BigVGAN, Kokoro-82M). Kept as a private module
/// constant so the CPU and GPU implementations pin the same value.
pub(crate) const EPS_SNAKE: f32 = 1e-9;

/// Snake activation — out-of-place, stateless.
///
/// Applies `y[c, t] = x[c, t] + (1 / (alpha[c] + eps)) * sin(alpha[c] · x[c, t])^2`
/// with `eps = 1e-9` per element of a `[channels, time]` row-major (channel-
/// outer) FP32 tensor. `alpha` is length-`channels`; `x` and `out` are both
/// length `channels · time`. `x` is read-only; `out` receives the result.
///
/// The loop order (outer channel, inner time) matches
/// [`crate::hiftnet::Snake::forward_in_place`] and the private
/// `kokoro::nn::snake_activation` helper in vokra-models, so a CPU dispatch
/// through this function is **bit-for-bit** consistent with the pre-seam
/// scalar loops those two call (same eps, same `sin` / multiply / add
/// primitives, same reduction order — trivial per-element, no reduction).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on:
/// - `alpha.len() != channels`;
/// - `x.len() != channels * time`;
/// - `out.len() != channels * time`;
/// - `channels * time` overflows `usize` (only reachable on impossibly large
///   shapes on 32-bit hosts; still a loud error, not a silent truncation).
///
/// `channels == 0` or `time == 0` is a no-op (the buffers are already
/// empty — verified by the length checks above — and no work is dispatched).
pub fn snake_activation_f32(
    x: &[f32],
    alpha: &[f32],
    channels: usize,
    time: usize,
    out: &mut [f32],
) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "snake_activation_f32: channels ({channels}) * time ({time}) overflows usize"
        ))
    })?;
    if alpha.len() != channels {
        return Err(VokraError::InvalidArgument(format!(
            "snake_activation_f32: alpha.len() ({}) != channels ({channels})",
            alpha.len()
        )));
    }
    if x.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "snake_activation_f32: x.len() ({}) != channels * time ({expected})",
            x.len()
        )));
    }
    if out.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "snake_activation_f32: out.len() ({}) != channels * time ({expected})",
            out.len()
        )));
    }
    if channels == 0 || time == 0 {
        return Ok(());
    }
    for (c, &a) in alpha.iter().enumerate() {
        let inv_a = 1.0 / (a + EPS_SNAKE);
        let row_start = c * time;
        let row_in = &x[row_start..row_start + time];
        let row_out = &mut out[row_start..row_start + time];
        for (dst, &v) in row_out.iter_mut().zip(row_in.iter()) {
            let s = (a * v).sin();
            *dst = v + inv_a * s * s;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hiftnet::Snake;

    /// Bit-for-bit vs `hiftnet::Snake::forward_in_place(..., alpha_logscale=false)`
    /// on a small deterministic input — same closed form, same eps, same
    /// reduction order, so the two must agree exactly.
    #[test]
    fn matches_hiftnet_snake_no_logscale_bit_identical() {
        let channels = 4;
        let time = 7;
        let alpha: Vec<f32> = (0..channels).map(|c| 0.3 + c as f32 * 0.17).collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.11).sin() * 1.7)
            .collect();

        let mut got = vec![0.0f32; channels * time];
        snake_activation_f32(&x, &alpha, channels, time, &mut got).unwrap();

        let snake = Snake::new(alpha.clone(), false).unwrap();
        let mut expected = x.clone();
        snake
            .forward_in_place(&mut expected, channels, time)
            .unwrap();

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "index {i}: snake_activation_f32 = {g} vs hiftnet::Snake = {e} (bit pattern must match)"
            );
        }
    }

    /// `snake(0) = 0` regardless of α (deterministic identity at the origin).
    #[test]
    fn zero_input_is_zero_regardless_of_alpha() {
        let channels = 3;
        let time = 5;
        let alpha = vec![0.7f32, 1.3, 2.1];
        let x = vec![0.0f32; channels * time];
        let mut out = vec![f32::NAN; channels * time]; // ensure we do NOT leave NaN
        snake_activation_f32(&x, &alpha, channels, time, &mut out).unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "expected all zeros, got {out:?}"
        );
    }

    /// α = 1 gives the closed form `x + sin(x)^2 / (1 + eps)`. Manual check
    /// at a few sample points using `f32::sin` (self-consistency — any
    /// change to the port would still trip this).
    #[test]
    fn alpha_one_matches_closed_form() {
        let inputs = [-2.0f32, -0.5, 0.0, 0.5, 1.7];
        let mut out = vec![0.0f32; inputs.len()];
        snake_activation_f32(&inputs, &[1.0], 1, inputs.len(), &mut out).unwrap();
        for (i, &x0) in inputs.iter().enumerate() {
            let s = x0.sin();
            let expected = x0 + s * s / (1.0 + EPS_SNAKE);
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "snake({x0}) = {} but expected {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn rejects_wrong_alpha_length() {
        let x = vec![0.0f32; 3 * 4];
        let mut out = vec![0.0f32; 3 * 4];
        let err = snake_activation_f32(&x, &[1.0, 1.0], 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("alpha.len()"), "{msg}");
    }

    #[test]
    fn rejects_wrong_x_length() {
        let x = vec![0.0f32; 3 * 4 - 1];
        let alpha = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4];
        let err = snake_activation_f32(&x, &alpha, 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("x.len()"), "{msg}");
    }

    #[test]
    fn rejects_wrong_out_length() {
        let x = vec![0.0f32; 3 * 4];
        let alpha = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4 + 1];
        let err = snake_activation_f32(&x, &alpha, 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("out.len()"), "{msg}");
    }

    #[test]
    fn empty_channels_or_time_is_noop() {
        // channels = 0
        let mut out: Vec<f32> = Vec::new();
        snake_activation_f32(&[], &[], 0, 5, &mut out).unwrap();
        assert!(out.is_empty());
        // time = 0
        let mut out: Vec<f32> = Vec::new();
        snake_activation_f32(&[], &[1.0f32, 2.0, 3.0], 3, 0, &mut out).unwrap();
        assert!(out.is_empty());
    }

    /// Per-channel alpha spreads correctly across the row: with channels = 2
    /// and time = 3, alpha[0] applies to indices 0..3 and alpha[1] to
    /// indices 3..6 (channel-outer layout).
    #[test]
    fn per_channel_alpha_applies_row_wise() {
        let channels = 2;
        let time = 3;
        let alpha = vec![0.5f32, 2.0];
        let x = vec![0.4f32, -0.4, 0.9, 0.4, -0.4, 0.9];
        let mut out = vec![0.0f32; 6];
        snake_activation_f32(&x, &alpha, channels, time, &mut out).unwrap();

        // Channel 0 (α = 0.5)
        for (i, &v) in x[0..3].iter().enumerate() {
            let s = (0.5f32 * v).sin();
            let expected = v + (1.0 / (0.5 + EPS_SNAKE)) * s * s;
            assert!((out[i] - expected).abs() < 1e-6, "ch0 idx {i}");
        }
        // Channel 1 (α = 2.0)
        for (i, &v) in x[3..6].iter().enumerate() {
            let s = (2.0f32 * v).sin();
            let expected = v + (1.0 / (2.0 + EPS_SNAKE)) * s * s;
            assert!((out[3 + i] - expected).abs() < 1e-6, "ch1 idx {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// SnakeBeta free function (Vocoder Metal wave — GPU-dispatch adapter)
// ---------------------------------------------------------------------------
//
// The stateful [`crate::bigvgan_generator::SnakeBeta`] type carries an owned
// `alpha` / `beta` and an `alpha_logscale` bool that the CPU forward
// exponentiates on the fly. This module exposes a **stateless out-of-place**
// free function [`snake_beta_f32`] that mirrors the shape of
// [`snake_activation_f32`] above so a GPU dispatch (Metal / CUDA / etc.) can
// route through the same seam pattern as the plain-Snake variant. The
// `alpha_logscale = true` case is expected to have `exp(alpha_raw)` /
// `exp(beta_raw)` applied on the host side by the converter or by the caller
// — the free function consumes the **already-effective** vectors so there is
// no branch in the hot path (same convention as [`snake_activation_f32`]).

/// SnakeBeta activation — out-of-place, stateless.
///
/// Applies `y[c, t] = x[c, t] + (1 / (beta[c] + eps)) · sin(alpha[c] · x[c, t])²`
/// with `eps = 1e-9` per element of a `[channels, time]` row-major (channel-
/// outer) FP32 tensor. `alpha` and `beta` are length-`channels`; `x` and
/// `out` are both length `channels · time`.
///
/// # Semantics vs. [`snake_activation_f32`]
///
/// The plain-Snake closed form ties frequency and magnitude to a single
/// per-channel `alpha` (`(1 / (α + eps)) · sin(α · x)²`). SnakeBeta separates
/// the two: `alpha` still scales the sinusoid argument (frequency), but
/// `beta` now scales the reciprocal in front of the squared sine
/// (magnitude). Two per-channel weight vectors → distinct op signature, so
/// the GPU seam gets its own dispatch method.
///
/// # `alpha_logscale`
///
/// - `alpha_logscale = true` (upstream `bigvgan_generator::SnakeBeta` with
///   `snake_logscale = true`) is an upstream-side transformation
///   (`alpha_eff = exp(alpha_raw)`, `beta_eff = exp(beta_raw)`) applied
///   before the same core formula. The converter can pre-exponentiate stored
///   log-scale weights and hand this function the already-effective vectors;
///   no branch in the hot path.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on:
/// - `alpha.len() != channels`;
/// - `beta.len() != channels`;
/// - `x.len() != channels * time`;
/// - `out.len() != channels * time`;
/// - `channels * time` overflows `usize` (only reachable on impossibly large
///   shapes on 32-bit hosts; still a loud error, not a silent truncation).
///
/// `channels == 0` or `time == 0` is a no-op (the buffers are already
/// empty — verified by the length checks above — and no work is dispatched).
pub fn snake_beta_f32(
    x: &[f32],
    alpha: &[f32],
    beta: &[f32],
    channels: usize,
    time: usize,
    out: &mut [f32],
) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "snake_beta_f32: channels ({channels}) * time ({time}) overflows usize"
        ))
    })?;
    if alpha.len() != channels {
        return Err(VokraError::InvalidArgument(format!(
            "snake_beta_f32: alpha.len() ({}) != channels ({channels})",
            alpha.len()
        )));
    }
    if beta.len() != channels {
        return Err(VokraError::InvalidArgument(format!(
            "snake_beta_f32: beta.len() ({}) != channels ({channels})",
            beta.len()
        )));
    }
    if x.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "snake_beta_f32: x.len() ({}) != channels * time ({expected})",
            x.len()
        )));
    }
    if out.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "snake_beta_f32: out.len() ({}) != channels * time ({expected})",
            out.len()
        )));
    }
    if channels == 0 || time == 0 {
        return Ok(());
    }
    for c in 0..channels {
        let a = alpha[c];
        let b = beta[c];
        let inv_b = 1.0 / (b + EPS_SNAKE);
        let row_start = c * time;
        let row_in = &x[row_start..row_start + time];
        let row_out = &mut out[row_start..row_start + time];
        for (dst, &v) in row_out.iter_mut().zip(row_in.iter()) {
            let s = (a * v).sin();
            *dst = v + inv_b * s * s;
        }
    }
    Ok(())
}

#[cfg(test)]
mod snake_beta_tests {
    use super::*;
    use crate::bigvgan_generator::SnakeBeta;

    /// Bit-for-bit vs `bigvgan_generator::SnakeBeta::forward_in_place(..., alpha_logscale=false)`
    /// on a small deterministic input — same closed form, same eps, same
    /// reduction order (trivially per-element), so the two must agree exactly.
    #[test]
    fn snake_beta_matches_bigvgan_snake_beta_no_logscale_bit_identical() {
        let channels = 4;
        let time = 7;
        let alpha: Vec<f32> = (0..channels).map(|c| 0.3 + c as f32 * 0.17).collect();
        let beta: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.11).collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.11).sin() * 1.7)
            .collect();

        let mut got = vec![0.0f32; channels * time];
        snake_beta_f32(&x, &alpha, &beta, channels, time, &mut got).unwrap();

        let sb = SnakeBeta::new(alpha.clone(), beta.clone(), false).unwrap();
        let mut expected = x.clone();
        sb.forward_in_place(&mut expected, channels, time).unwrap();

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "index {i}: snake_beta_f32 = {g} vs bigvgan_generator::SnakeBeta = {e} (bit pattern must match)"
            );
        }
    }

    /// `snake_beta(0) = 0` regardless of α, β (deterministic identity at origin).
    #[test]
    fn snake_beta_zero_input_is_zero_regardless_of_alpha_beta() {
        let channels = 3;
        let time = 5;
        let alpha = vec![0.7f32, 1.3, 2.1];
        let beta = vec![0.5f32, 0.9, 1.5];
        let x = vec![0.0f32; channels * time];
        let mut out = vec![f32::NAN; channels * time];
        snake_beta_f32(&x, &alpha, &beta, channels, time, &mut out).unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "expected all zeros, got {out:?}"
        );
    }

    /// α = β = 1 gives the closed form `x + sin(x)^2 / (1 + eps)` — identical
    /// to the plain-Snake case with α = 1. This pins the reciprocal target
    /// (β, not α) — swapping α ↔ β in the formula would still pass α=β=1 but
    /// break the β = 2 test below.
    #[test]
    fn snake_beta_alpha_beta_one_matches_snake_alpha_one() {
        let inputs = [-2.0f32, -0.5, 0.0, 0.5, 1.7];
        let mut got = vec![0.0f32; inputs.len()];
        snake_beta_f32(&inputs, &[1.0], &[1.0], 1, inputs.len(), &mut got).unwrap();
        for (i, &x0) in inputs.iter().enumerate() {
            let s = x0.sin();
            let expected = x0 + s * s / (1.0 + EPS_SNAKE);
            assert!(
                (got[i] - expected).abs() < 1e-6,
                "snake_beta(α=1, β=1, x={x0}) = {} but expected {}",
                got[i],
                expected
            );
        }
    }

    /// β = 2 (≠ α) discriminates SnakeBeta from Snake and pins the reciprocal
    /// target to β (not α): a variant that computed `1 / (α + eps)` would
    /// produce a distinctly different magnitude and fail here.
    #[test]
    fn snake_beta_beta_two_scales_reciprocal_not_alpha() {
        let inputs = [-1.5f32, -0.3, 0.0, 0.4, 1.2];
        let alpha = 1.0f32;
        let beta = 2.0f32;
        let mut got = vec![0.0f32; inputs.len()];
        snake_beta_f32(&inputs, &[alpha], &[beta], 1, inputs.len(), &mut got).unwrap();
        for (i, &x0) in inputs.iter().enumerate() {
            let s = (alpha * x0).sin();
            let expected_beta_reciprocal = x0 + s * s / (beta + EPS_SNAKE);
            // Sibling alpha-reciprocal alternative — must NOT match at β ≠ α:
            let alt_alpha_reciprocal = x0 + s * s / (alpha + EPS_SNAKE);
            assert!(
                (got[i] - expected_beta_reciprocal).abs() < 1e-6,
                "snake_beta(α=1, β=2, x={x0}) = {} but expected {} \
                 (β-reciprocal form)",
                got[i],
                expected_beta_reciprocal,
            );
            // At x = 0 both reciprocals collapse to 0 — skip that slot for
            // the discriminating check.
            if x0 != 0.0 {
                assert!(
                    (got[i] - alt_alpha_reciprocal).abs() > 1e-6,
                    "snake_beta(α=1, β=2, x={x0}) accidentally matches the \
                     α-reciprocal form ({alt_alpha_reciprocal}) — β vs α \
                     swapped in the reciprocal?"
                );
            }
        }
    }

    #[test]
    fn snake_beta_rejects_wrong_alpha_length() {
        let x = vec![0.0f32; 3 * 4];
        let beta = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4];
        let err = snake_beta_f32(&x, &[1.0, 1.0], &beta, 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("alpha.len()"), "{msg}");
    }

    #[test]
    fn snake_beta_rejects_wrong_beta_length() {
        let x = vec![0.0f32; 3 * 4];
        let alpha = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4];
        let err = snake_beta_f32(&x, &alpha, &[1.0, 1.0], 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("beta.len()"), "{msg}");
    }

    #[test]
    fn snake_beta_rejects_wrong_x_length() {
        let x = vec![0.0f32; 3 * 4 - 1];
        let alpha = vec![1.0f32; 3];
        let beta = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4];
        let err = snake_beta_f32(&x, &alpha, &beta, 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("x.len()"), "{msg}");
    }

    #[test]
    fn snake_beta_rejects_wrong_out_length() {
        let x = vec![0.0f32; 3 * 4];
        let alpha = vec![1.0f32; 3];
        let beta = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 3 * 4 + 1];
        let err = snake_beta_f32(&x, &alpha, &beta, 3, 4, &mut out).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("out.len()"), "{msg}");
    }

    #[test]
    fn snake_beta_empty_channels_or_time_is_noop() {
        // channels = 0
        let mut out: Vec<f32> = Vec::new();
        snake_beta_f32(&[], &[], &[], 0, 5, &mut out).unwrap();
        assert!(out.is_empty());
        // time = 0
        let mut out: Vec<f32> = Vec::new();
        snake_beta_f32(
            &[],
            &[1.0f32, 2.0, 3.0],
            &[1.0f32, 1.0, 1.0],
            3,
            0,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty());
    }
}
