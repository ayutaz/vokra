//! Polyphase anti-aliased upsampling primitive (Vocoder Metal wave —
//! common vocoder primitive shared by BigVGAN's `UpSample1d` and HiFTNet-family
//! alias-free activation chains).
//!
//! # Op contract
//!
//! Given a `[channels, time_in]` row-major FP32 tensor `x`, a scalar integer
//! `ratio` (the upsample factor), and a caller-supplied low-pass filter
//! `kernel` (Kaiser-window taps, pre-designed on the host from the desired
//! `cutoff` and `periodicity` — see the module docstring below), this op
//! writes a `[channels, time_in * ratio]` row-major FP32 tensor `out` where
//!
//! ```text
//! out[c, t_out] = Σ_j kernel[j * ratio + r] * x[c, t - j]
//! ```
//!
//! with `t = t_out / ratio`, `r = t_out % ratio` (the polyphase branch),
//! and the sum running over every `j` where `t - j ∈ [0, time_in)`.
//!
//! This is the standard **polyphase decomposition** of "upsample by `N` (insert
//! `N-1` zeros between samples) then convolve with a low-pass FIR": mathematically
//! `out = up_N(x) * kernel`, but the polyphase form skips the multiplies by
//! the inserted zeros. Each output position depends on `⌈taps / ratio⌉` input
//! samples — the per-output work is `O(taps / ratio)`.
//!
//! # Where the Kaiser design lives (attributes: `cutoff`, `filter_kernel`, `periodicity`)
//!
//! The audit's attribute list — `cutoff`, `filter_kernel`, `periodicity` — is
//! Kaiser-window filter design metadata: given a target low-pass `cutoff`
//! (in units of the Nyquist rate) and a `periodicity` / kernel length, a
//! Kaiser window sinc filter produces the `filter_kernel` taps. That design
//! step is **host-side and once per model load** (the kernel is a config-time
//! constant); this compute op consumes the already-designed taps and does the
//! per-timestep multiply-add.
//!
//! Keeping the Kaiser design out of the hot path (a) matches upstream BigVGAN
//! (`alias_free_activation.torch.act.UpSample1d.__init__` builds the filter
//! once and stores it as a module buffer), (b) lets a caller substitute a
//! different low-pass filter (Hamming, Blackman-Harris) without touching the
//! kernel, and (c) keeps the runtime op signature narrow (three tensor
//! inputs + one scalar `ratio`) — a good fit for a GPU dispatch.
//!
//! # Layout
//!
//! - `x`      — `[channels, time_in]` row-major FP32 (channel-outer).
//! - `kernel` — `[taps]` FP32; every entry may be zero (a length-`taps` all-
//!   zero kernel produces an all-zero output). The kernel is
//!   conceptually **causal**: `kernel[0]` multiplies `x[c, t]`,
//!   `kernel[1]` multiplies `x[c, t-1]`, and so on. Callers that
//!   want a symmetric zero-phase kernel should pre-shift the
//!   output on the host (or pre-pad the input) — the op is
//!   deliberately narrow and does not embed a phase convention.
//! - `out`    — `[channels, time_in * ratio]` row-major FP32 (channel-outer).
//!
//! `kernel.len()` does NOT have to be a multiple of `ratio`; when it is not, the
//! per-branch polyphase filter for branches `r >= (kernel.len() % ratio)` is
//! one tap shorter (the last "aligned" tap `⌈taps / ratio⌉ * ratio + r`
//! would sit past `kernel.len()` and is skipped — the same behaviour as
//! zero-padding the kernel out to a multiple of `ratio`).
//!
//! # FP32, no fast-math on the CPU
//!
//! The CPU op runs the multiply-add in scalar FP32. Under a Rust `for` loop
//! `acc += x * k` does NOT fuse to FMA (rustc does not emit `fmadd` in
//! debug or release for that pattern without `--target-feature=+fma` and
//! explicit `.mul_add`), so the reduction is `((acc + x0*k0) + x1*k1) + ...`
//! in strict left-fold order. The GPU MSL kernel will compile with default
//! fast-math on and may fuse to `fma()` — the parity bound accepts that
//! divergence (`atol ≤ 1e-4`; see `crates/vokra-models/tests/
//! anti_aliased_upsample_metal_bit_identical.rs`).
//!
//! # Zero third-party deps (NFR-DS-02)
//!
//! Pure scalar Rust; no BLAS, no `signal-crate`, no Kaiser designer (the
//! designer is the caller's problem, keeping this op narrow).

use vokra_core::{Result, VokraError};

/// Polyphase anti-aliased upsample.
///
/// See the module docstring for the exact contract, layout, and rounding
/// notes. Fails loudly on any shape disagreement; `channels == 0` or
/// `time_in == 0` is a no-op (both buffers are empty per the length checks).
///
/// # Parameters
///
/// - `x`         — `[channels, time_in]` row-major FP32 input.
/// - `kernel`    — `[taps]` FP32 causal low-pass filter taps
///   (pre-designed on the host).
/// - `ratio`     — integer upsample factor (`>= 1`).
/// - `channels`  — number of channels in `x` / `out`.
/// - `time_in`   — number of input timesteps.
/// - `out`       — `[channels, time_in * ratio]` row-major FP32 output.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on:
/// - `ratio == 0`;
/// - `kernel.is_empty()`;
/// - `x.len() != channels * time_in`;
/// - `out.len() != channels * time_in * ratio`;
/// - `channels * time_in` or `channels * time_in * ratio` overflows `usize`.
///
/// `channels == 0` or `time_in == 0` is a no-op — every buffer is already
/// empty per the length checks above, and nothing is dispatched.
pub fn anti_aliased_upsample_f32(
    x: &[f32],
    kernel: &[f32],
    ratio: usize,
    channels: usize,
    time_in: usize,
    out: &mut [f32],
) -> Result<()> {
    if ratio == 0 {
        return Err(VokraError::InvalidArgument(
            "anti_aliased_upsample_f32: ratio must be >= 1".to_owned(),
        ));
    }
    if kernel.is_empty() {
        return Err(VokraError::InvalidArgument(
            "anti_aliased_upsample_f32: kernel must not be empty".to_owned(),
        ));
    }
    let expected_x = channels.checked_mul(time_in).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "anti_aliased_upsample_f32: channels ({channels}) * time_in ({time_in}) overflows usize"
        ))
    })?;
    let time_out = time_in.checked_mul(ratio).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "anti_aliased_upsample_f32: time_in ({time_in}) * ratio ({ratio}) overflows usize"
        ))
    })?;
    let expected_out = channels.checked_mul(time_out).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "anti_aliased_upsample_f32: channels ({channels}) * time_in * ratio \
             ({time_out}) overflows usize"
        ))
    })?;
    if x.len() != expected_x {
        return Err(VokraError::InvalidArgument(format!(
            "anti_aliased_upsample_f32: x.len() ({}) != channels * time_in ({expected_x})",
            x.len()
        )));
    }
    if out.len() != expected_out {
        return Err(VokraError::InvalidArgument(format!(
            "anti_aliased_upsample_f32: out.len() ({}) != channels * time_in * ratio \
             ({expected_out})",
            out.len()
        )));
    }
    if channels == 0 || time_in == 0 {
        return Ok(());
    }

    let taps = kernel.len();
    // Per-branch tap count is `ceil(taps / ratio)`; a branch `r` uses taps
    // at kernel indices `r, r+ratio, r+2*ratio, ...` up to but not including
    // `taps`. The `if k_idx >= taps { break; }` guard below is what makes the
    // ragged branches one tap shorter without needing a per-branch count
    // pre-computation.
    for c in 0..channels {
        let x_row_off = c * time_in;
        let out_row_off = c * time_out;
        for t_out in 0..time_out {
            let t = t_out / ratio;
            let r = t_out % ratio;
            let mut acc = 0.0f32;
            let mut j = 0usize;
            loop {
                let k_idx = j * ratio + r;
                if k_idx >= taps {
                    break;
                }
                // `src = t - j`; skip out-of-range (t - j < 0) — same
                // zero-padded convention as upstream `nn.functional.conv1d`
                // with `padding = 0` on the causal end.
                if j > t {
                    break;
                }
                let src = t - j;
                acc += x[x_row_off + src] * kernel[k_idx];
                j += 1;
            }
            out[out_row_off + t_out] = acc;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ratio = 1` with a length-1 unit kernel is the identity — every input
    /// sample survives unchanged. Pins the outer shape without exercising
    /// the polyphase math.
    #[test]
    fn ratio_one_unit_kernel_is_identity() {
        let x = vec![0.5f32, -0.3, 1.7, -2.1];
        let mut out = vec![0.0f32; 4];
        anti_aliased_upsample_f32(&x, &[1.0], 1, 1, 4, &mut out).unwrap();
        for (i, (&g, &e)) in out.iter().zip(x.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "idx {i}: got {g}, expected {e}");
        }
    }

    /// A single non-zero tap at index 0 with `ratio = 2` reproduces the
    /// naive "insert zeros between samples" upsample: even outputs are
    /// `x[t/2]`, odd outputs are zero.
    #[test]
    fn ratio_two_impulse_kernel_produces_upsample_by_zero_insertion() {
        let x = vec![1.0f32, 2.0, 3.0];
        // taps = 2, ratio = 2 → branches: kernel[0] applied on r=0,
        // kernel[1] applied on r=1. Setting kernel[1]=0 zeros the odd
        // branch, kernel[0]=1 copies x through the even branch.
        let kernel = vec![1.0f32, 0.0];
        let mut out = vec![0.0f32; 6];
        anti_aliased_upsample_f32(&x, &kernel, 2, 1, 3, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
    }

    /// A length-`ratio` symmetric "box average" kernel with all taps set to
    /// `1.0` — every output branch gets exactly one tap in play and every
    /// output sample equals its corresponding input sample. Pins the
    /// per-branch tap iteration (branch `r` uses tap `r`, so if branch
    /// dispatch is broken this fails immediately).
    #[test]
    fn box_kernel_per_branch_dispatch_is_correct() {
        let x = vec![7.0f32, 4.0, -3.0];
        let ratio = 3;
        // taps = ratio = 3 → each branch r has exactly one tap
        // (kernel[r * 1 + r*0] = kernel[r]).
        let kernel = vec![1.0f32; ratio];
        let mut out = vec![0.0f32; 9];
        anti_aliased_upsample_f32(&x, &kernel, ratio, 1, 3, &mut out).unwrap();
        // out[0] = k[0] * x[0], out[1] = k[1] * x[0], out[2] = k[2] * x[0],
        // then out[3..6] = k[0..3] * x[1], out[6..9] = k[0..3] * x[2].
        assert_eq!(out, vec![7.0, 7.0, 7.0, 4.0, 4.0, 4.0, -3.0, -3.0, -3.0]);
    }

    /// A length-4 causal kernel with `ratio = 2` walks two taps per branch.
    /// Hand-computed output to pin the polyphase index math.
    #[test]
    fn ratio_two_length_four_kernel_hand_computed() {
        // x = [a, b, c] with a = 1, b = 2, c = 3.
        let x = vec![1.0f32, 2.0, 3.0];
        // kernel = [k0, k1, k2, k3] = [0.5, 0.25, 0.1, 0.05].
        let k = [0.5f32, 0.25, 0.1, 0.05];
        let kernel = k.to_vec();
        let mut out = vec![0.0f32; 6];
        anti_aliased_upsample_f32(&x, &kernel, 2, 1, 3, &mut out).unwrap();
        // Branch r=0 uses kernel[0], kernel[2]; branch r=1 uses kernel[1], kernel[3].
        // out[0] (t=0, r=0) = k0 * a
        // out[1] (t=0, r=1) = k1 * a
        // out[2] (t=1, r=0) = k0 * b + k2 * a
        // out[3] (t=1, r=1) = k1 * b + k3 * a
        // out[4] (t=2, r=0) = k0 * c + k2 * b
        // out[5] (t=2, r=1) = k1 * c + k3 * b
        let expected = [
            k[0] * x[0],
            k[1] * x[0],
            k[0] * x[1] + k[2] * x[0],
            k[1] * x[1] + k[3] * x[0],
            k[0] * x[2] + k[2] * x[1],
            k[1] * x[2] + k[3] * x[1],
        ];
        for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "idx {i}: got {g}, expected {e}");
        }
    }

    /// Ragged kernel length: `taps = 5`, `ratio = 2` → branch r=0 gets 3
    /// taps (kernel[0], kernel[2], kernel[4]), branch r=1 gets 2 taps
    /// (kernel[1], kernel[3]). Verifies the `k_idx >= taps { break; }`
    /// guard drops the missing tap cleanly rather than reading past the
    /// buffer.
    #[test]
    fn ragged_kernel_length_drops_missing_branch_taps() {
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let kernel = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0f32; 8];
        anti_aliased_upsample_f32(&x, &kernel, 2, 1, 4, &mut out).unwrap();
        // Branch r=0 (indices 0, 2, 4, 6): 3 taps
        // - out[0] (t=0): only x[0] valid → 1 * 1 = 1
        // - out[2] (t=1): x[1] + x[0] valid → 2 taps in range = 2
        // - out[4] (t=2): x[2] + x[1] + x[0] = 3
        // - out[6] (t=3): x[3] + x[2] + x[1] = 3 (t=3 - j=3 = 0 also valid,
        //   but that would be the 4th tap and branch r=0 only has 3)
        // Actually: branch r=0 has kernel[0], kernel[2], kernel[4] — 3 taps.
        // For t=3, valid j = 0, 1, 2 (all in [0, t+1)), so acc = 1+1+1 = 3.
        // Branch r=1 (indices 1, 3): 2 taps
        // - out[1] (t=0): only x[0] valid → 1
        // - out[3] (t=1): x[1] + x[0] → 2
        // - out[5] (t=2): x[2] + x[1] → 2
        // - out[7] (t=3): x[3] + x[2] → 2
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 2.0]);
    }

    /// Multi-channel with distinct row payloads — pins the row-outer layout
    /// (channel-outer, time-inner) on both `x` and `out`.
    #[test]
    fn multi_channel_rows_are_independent() {
        let channels = 2;
        let time_in = 3;
        let x = vec![
            1.0f32, 2.0, 3.0, // channel 0
            10.0, 20.0, 30.0, // channel 1
        ];
        let ratio = 2;
        let kernel = vec![1.0f32, 0.0]; // identity zero-insertion (see the
        // impulse test above) — makes the assertion easy to eyeball.
        let mut out = vec![0.0f32; channels * time_in * ratio];
        anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut out).unwrap();
        assert_eq!(
            out,
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, // channel 0
                10.0, 0.0, 20.0, 0.0, 30.0, 0.0, // channel 1
            ]
        );
    }

    #[test]
    fn rejects_zero_ratio() {
        let mut out: Vec<f32> = vec![0.0f32; 4];
        let err = anti_aliased_upsample_f32(&[0.0f32; 4], &[1.0], 0, 1, 4, &mut out).unwrap_err();
        assert!(err.to_string().contains("ratio must be >= 1"), "{err}");
    }

    #[test]
    fn rejects_empty_kernel() {
        let mut out: Vec<f32> = vec![0.0f32; 4];
        let err = anti_aliased_upsample_f32(&[0.0f32; 4], &[], 1, 1, 4, &mut out).unwrap_err();
        assert!(
            err.to_string().contains("kernel must not be empty"),
            "{err}"
        );
    }

    #[test]
    fn rejects_wrong_x_length() {
        let x = vec![0.0f32; 3]; // channels=2, time_in=2 → expected 4
        let mut out = vec![0.0f32; 8];
        let err = anti_aliased_upsample_f32(&x, &[1.0], 2, 2, 2, &mut out).unwrap_err();
        assert!(err.to_string().contains("x.len()"), "{err}");
    }

    #[test]
    fn rejects_wrong_out_length() {
        let x = vec![0.0f32; 4]; // channels=2, time_in=2
        let mut out = vec![0.0f32; 7]; // ratio=2 → expected 8
        let err = anti_aliased_upsample_f32(&x, &[1.0], 2, 2, 2, &mut out).unwrap_err();
        assert!(err.to_string().contains("out.len()"), "{err}");
    }

    #[test]
    fn empty_channels_or_time_is_noop() {
        let mut out: Vec<f32> = Vec::new();
        anti_aliased_upsample_f32(&[], &[1.0], 2, 0, 5, &mut out).unwrap();
        assert!(out.is_empty());
        let mut out: Vec<f32> = Vec::new();
        anti_aliased_upsample_f32(&[], &[1.0], 2, 3, 0, &mut out).unwrap();
        assert!(out.is_empty());
    }
}
