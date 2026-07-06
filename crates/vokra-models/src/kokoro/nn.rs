//! Compute primitives for the Kokoro-82M native TTS (M2-07-T09/T10).
//!
//! Kokoro is a StyleTTS 2 派生 iSTFTNet vocoder; it shares the same NN
//! primitive surface as the piper-plus decoder (1-D dilated / grouped /
//! transposed convolutions in `[channels, time]` layout, plus activations and
//! layer-norm). The primitives here are a **copy** of
//! [`crate::piper_plus::nn`] rather than a re-export, so a divergence in either
//! module does not accidentally couple them (per the M2-07 module-independence
//! call); a differential test pins the shipped im2col+GEMM `conv1d` to a
//! reference scalar oracle inside the piper module and that guarantee holds
//! independently for this copy.
//!
//! In addition, this module hosts the private [`adain`] helper: StyleTTS 2's
//! Adaptive Instance Normalisation implemented as a **composition of existing
//! ops** (instance-norm then affine), not as a new first-class op. `EPS = 1e-5`
//! is the standard `nn.InstanceNorm1d` default (aligned with
//! [`LAYER_NORM_EPS`] in the piper config).

use crate::compute::Compute;

/// LeakyReLU slope used throughout the Kokoro decoder — same value as the
/// piper-plus decoder's `mb_istft.py` `LRELU_SLOPE`. Held here so the Kokoro
/// module does not need to reach into [`crate::piper_plus`] internals.
#[allow(dead_code)] // consumed by the T16 decoder wiring
pub(crate) const LRELU_SLOPE: f32 = 0.1;

/// `nn.InstanceNorm1d` / AdaIN epsilon (PyTorch default).
pub(crate) const EPS: f32 = 1e-5;

/// 1-D convolution with stride / padding / dilation / groups, lowered to
/// im2col + GEMM, dispatched through the [`Compute`] seam. Mirrors
/// [`crate::piper_plus::nn::conv1d`] verbatim — see that module for the
/// rationale and the shipping-vs-oracle differential test.
///
/// `x` is `[in_ch, in_len]`, `weight` is `[out_ch, in_ch/groups, kernel]`
/// (PyTorch / ONNX layout), `bias` (when `Some`) is `[out_ch]`. Returns
/// `[out_ch, out_len]`.
#[allow(clippy::too_many_arguments, dead_code)] // consumed by the T12–T17 forward path
pub(crate) fn conv1d(
    compute: &Compute,
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
    dilation: usize,
    groups: usize,
) -> (Vec<f32>, usize) {
    let eff = dilation * (kernel - 1) + 1;
    let out_len = (in_len + 2 * pad - eff) / stride + 1;
    let in_g = in_ch / groups;
    let out_g = out_ch / groups;
    let k = in_g * kernel;
    let mut out = vec![0.0f32; out_ch * out_len];
    let mut col = vec![0.0f32; k * out_len];
    let mut og = vec![0.0f32; out_g * out_len];
    for g in 0..groups {
        col.fill(0.0);
        for ic in 0..in_g {
            let xrow = (g * in_g + ic) * in_len;
            for kk in 0..kernel {
                let crow = (ic * kernel + kk) * out_len;
                for ot in 0..out_len {
                    let it = ot * stride + kk * dilation;
                    if it >= pad && it - pad < in_len {
                        col[crow + ot] = x[xrow + (it - pad)];
                    }
                }
            }
        }
        let wbase = g * out_g * k;
        compute
            .gemm_f32(
                out_g,
                out_len,
                k,
                &weight[wbase..wbase + out_g * k],
                &col,
                None,
                &mut og,
            )
            .expect("kokoro conv1d gemm: internally-consistent shapes");
        for oc in 0..out_g {
            let out_channel = g * out_g + oc;
            let b = bias.map_or(0.0, |b| b[out_channel]);
            let dst = &mut out[out_channel * out_len..out_channel * out_len + out_len];
            for (d, &s) in dst.iter_mut().zip(&og[oc * out_len..(oc + 1) * out_len]) {
                *d = s + b;
            }
        }
    }
    (out, out_len)
}

/// Transposed 1-D convolution with stride / padding / groups (no dilation).
/// Mirrors [`crate::piper_plus::nn::conv_transpose1d`].
#[allow(clippy::too_many_arguments, dead_code)] // consumed by the T16 decoder upsample stack
pub(crate) fn conv_transpose1d(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
    groups: usize,
) -> (Vec<f32>, usize) {
    let out_len = (in_len - 1) * stride + kernel - 2 * pad;
    let in_g = in_ch / groups;
    let out_g = out_ch / groups;
    let mut out = vec![0.0f32; out_ch * out_len];
    for in_channel in 0..in_ch {
        let g = in_channel / in_g;
        let xrow = in_channel * in_len;
        for it in 0..in_len {
            let xv = x[xrow + it];
            if xv == 0.0 {
                continue;
            }
            for oc in 0..out_g {
                let out_channel = g * out_g + oc;
                let wrow = (in_channel * out_g + oc) * kernel;
                for kk in 0..kernel {
                    let pos = it * stride + kk;
                    if pos >= pad {
                        let ot = pos - pad;
                        if ot < out_len {
                            out[out_channel * out_len + ot] += xv * weight[wrow + kk];
                        }
                    }
                }
            }
        }
    }
    if let Some(bias) = bias {
        for (oc, &bv) in bias.iter().enumerate() {
            for v in &mut out[oc * out_len..oc * out_len + out_len] {
                *v += bv;
            }
        }
    }
    (out, out_len)
}

/// Logistic sigmoid `1/(1 + e^-x)`.
#[allow(dead_code)] // consumed by the T13 prosody / T16 decoder wiring
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place LeakyReLU (`x < 0 → slope·x`).
#[allow(dead_code)] // consumed by the T16 decoder wiring
pub(crate) fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// Exact (erf-based) GELU, matching PyTorch `F.gelu` default.
#[allow(dead_code)] // consumed by the T12 text encoder wiring
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26; ~1e-7 max error).
#[allow(clippy::excessive_precision, dead_code)] // A&S reference coefficients
pub(crate) fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_43 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// StyleTTS 2's Adaptive Instance Normalisation, applied in place to a
/// `[channels, time]` signal (M2-07-T16).
///
/// Composition of two existing ops (FR-EX-08 permits composition; a new
/// first-class `adain` op is deliberately NOT introduced — the design synthesis
/// in `docs/adr/0007-kokoro-native.md` records why):
///
///   1. `nn.InstanceNorm1d`-style normalisation over the time axis per channel
///      (mean / variance, `sqrt(var + eps)`);
///   2. affine `y = gamma · x + beta` broadcast per channel.
///
/// `gamma`, `beta` are `[channels]`; `x` is `[channels · time]` channel-major
/// (row-major with the channel as the outer axis, matching every other
/// piper / kokoro tensor).
#[allow(dead_code)] // consumed by the T16 decoder wiring
pub(crate) fn adain(x: &mut [f32], gamma: &[f32], beta: &[f32], channels: usize, time: usize) {
    debug_assert_eq!(x.len(), channels * time, "adain: len mismatch");
    debug_assert_eq!(gamma.len(), channels, "adain: gamma len");
    debug_assert_eq!(beta.len(), channels, "adain: beta len");
    if time == 0 {
        return;
    }
    let inv_t = 1.0 / time as f32;
    for c in 0..channels {
        let row = &mut x[c * time..c * time + time];
        // Mean over the time axis.
        let mut mean = 0.0f32;
        for &v in row.iter() {
            mean += v;
        }
        mean *= inv_t;
        // Variance over the time axis (biased, as PyTorch's InstanceNorm1d).
        let mut var = 0.0f32;
        for &v in row.iter() {
            let d = v - mean;
            var += d * d;
        }
        var *= inv_t;
        let inv = 1.0 / (var + EPS).sqrt();
        let g = gamma[c];
        let b = beta[c];
        for v in row.iter_mut() {
            *v = (*v - mean) * inv * g + b;
        }
    }
}

/// Expands per-phoneme features to frame resolution by repeating each phoneme
/// column `durations[j]` times — the monotonic, search-free length regulator
/// used by StyleTTS 2 / Kokoro-82M's prosody predictor output → decoder input
/// bridge. `encoded` is `[hidden, t_in]` channel-major; returns
/// `([hidden, t_out], t_out)` with `t_out = sum(durations[..t_in])`.
///
/// This is an identity duplication of [`crate::piper_plus::mod::length_regulate`]
/// (see `crates/vokra-models/src/piper_plus/mod.rs:734`) — the compute is
/// domain-independent (a repeat-index copy), and keeping a private Kokoro copy
/// preserves the M2-07 module-independence rule (`docs/adr/0007-kokoro-native.md`).
/// A divergence in either implementation is caught by the differential test
/// [`regulates_and_computes_t_out`] against a scalar oracle.
#[allow(dead_code)] // consumed by the T18 e2e wiring
pub(crate) fn length_regulate(
    encoded: &[f32],
    hidden: usize,
    t_in: usize,
    durations: &[usize],
) -> (Vec<f32>, usize) {
    let t_out: usize = durations.iter().take(t_in).sum();
    let mut out = vec![0.0f32; hidden * t_out];
    let mut tf = 0;
    for (j, &reps) in durations.iter().take(t_in).enumerate() {
        for _ in 0..reps {
            for c in 0..hidden {
                out[c * t_out + tf] = encoded[c * t_in + j];
            }
            tf += 1;
        }
    }
    (out, t_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regulates_and_computes_t_out() {
        // hidden=2, t_in=3, channel-major [2,3]:
        //   ch0 = [1,2,3], ch1 = [4,5,6];  durations = [2,1,3].
        let encoded = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let durations = [2usize, 1, 3];
        let (out, t_out) = length_regulate(&encoded, 2, 3, &durations);

        // t_out = 2+1+3 = 6.
        assert_eq!(t_out, 6);
        // ch0 → [1,1,2,3,3,3], ch1 → [4,4,5,6,6,6] (channel-major).
        assert_eq!(
            out,
            [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0]
        );

        // A trailing duration past t_in is ignored (take(t_in)); the resulting
        // frames and t_out remain unchanged.
        let durations_over = [2usize, 1, 3, 99];
        let (out2, t_out2) = length_regulate(&encoded, 2, 3, &durations_over);
        assert_eq!(t_out2, 6);
        assert_eq!(out2, out);

        // Zero-duration phonemes collapse: durations = [0,0,0] → t_out = 0, empty.
        let (out_zero, t_out_zero) = length_regulate(&encoded, 2, 3, &[0, 0, 0]);
        assert_eq!(t_out_zero, 0);
        assert!(out_zero.is_empty());
    }

    #[test]
    fn adain_matches_scalar_oracle_on_4_samples() {
        // 2 channels, 4 time steps; distinct values per (channel, time) so any
        // mis-index is caught, and the two channels have very different means so
        // a shared-scratch bug would visibly bleed across channels.
        let mut x = vec![
            1.0, 2.0, 3.0, 4.0, // ch0
            100.0, 200.0, 300.0, 400.0, // ch1
        ];
        let gamma = [2.0, 0.5];
        let beta = [10.0, -1.0];
        let channels = 2;
        let time = 4;

        // Scalar oracle: for each channel, normalise (mean/std with EPS) then
        // affine.
        let mut want = x.clone();
        for c in 0..channels {
            let row = &mut want[c * time..c * time + time];
            let mean = row.iter().sum::<f32>() / time as f32;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / time as f32;
            let inv = 1.0 / (var + EPS).sqrt();
            for v in row.iter_mut() {
                *v = (*v - mean) * inv * gamma[c] + beta[c];
            }
        }

        adain(&mut x, &gamma, &beta, channels, time);
        for (i, (&g, &w)) in x.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "index {i}: {g} vs {w}");
        }
    }

    #[test]
    fn adain_with_gamma_1_beta_0_yields_zero_mean_unit_var() {
        // Channels of any pre-affine distribution normalise to ~zero mean and
        // ~unit variance when gamma=1, beta=0.
        let mut x = vec![5.0, 10.0, 15.0, 20.0];
        adain(&mut x, &[1.0], &[0.0], 1, 4);
        let mean = x.iter().sum::<f32>() / 4.0;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean = {mean}");
        // With EPS = 1e-5, var(post) ≈ var(pre)/(var(pre) + eps) → ~1 for var ≫ eps.
        assert!((var - 1.0).abs() < 1e-4, "var = {var}");
    }

    #[test]
    fn sigmoid_and_gelu_reference_points() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!(gelu(0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_345).abs() < 1e-4);
    }

    #[test]
    fn leaky_relu_slope_applied_to_negatives_only() {
        let mut x = [-2.0, -1.0, 0.0, 1.0, 2.0];
        leaky_relu(&mut x, 0.1);
        assert_eq!(x, [-0.2, -0.1, 0.0, 1.0, 2.0]);
    }
}
