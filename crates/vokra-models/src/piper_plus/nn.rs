//! Scalar compute primitives for the piper-plus native TTS (M0-07).
//!
//! MB-iSTFT-VITS2 needs dilated, grouped and transposed 1-D convolutions plus
//! per-position linear layers. The M0-08 CPU kernel set (`vokra-backend-cpu`)
//! does not cover the convolution variants this model needs — its `conv1d_f32`
//! has no dilation/groups and there is no transposed conv — so these ops are
//! self-contained scalar f32 here (NFR-QL-01 parity is FP32). Routing the
//! decoder/flow convolutions and the small attention matmuls through
//! `vokra-backend-cpu` SIMD kernels (once it grows dilation/groups/transpose
//! support, per ADR-0002) is an **M1 optimization follow-up** — M0 has no RTF
//! gate (milestones.md §4.2 note 1) and the scalar path is correct.
//!
//! Tensors are plain row-major `Vec<f32>` with explicit shapes; 1-D signals use
//! the `[channels, time]` layout PyTorch/ONNX convolutions expect.

/// 1-D convolution with stride / padding / dilation / groups.
///
/// `x` is `[in_ch, in_len]`, `weight` is `[out_ch, in_ch/groups, kernel]`
/// (PyTorch/ONNX layout), `bias` (when `Some`) is `[out_ch]`. Returns
/// `[out_ch, out_len]` with `out_len = (in_len + 2·pad − dilation·(kernel−1) −
/// 1) / stride + 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d(
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
    let mut out = vec![0.0f32; out_ch * out_len];
    for g in 0..groups {
        for oc in 0..out_g {
            let out_channel = g * out_g + oc;
            let wbase = out_channel * in_g * kernel;
            for ot in 0..out_len {
                let mut acc = bias.map_or(0.0, |b| b[out_channel]);
                let start = ot * stride;
                for ic in 0..in_g {
                    let in_channel = g * in_g + ic;
                    let xrow = in_channel * in_len;
                    let wrow = wbase + ic * kernel;
                    for kk in 0..kernel {
                        let it = start + kk * dilation;
                        if it >= pad && it - pad < in_len {
                            acc += x[xrow + (it - pad)] * weight[wrow + kk];
                        }
                    }
                }
                out[out_channel * out_len + ot] = acc;
            }
        }
    }
    (out, out_len)
}

/// Transposed 1-D convolution with stride / padding / groups (no dilation —
/// unused by MB-iSTFT-VITS2).
///
/// `x` is `[in_ch, in_len]`, `weight` is `[in_ch, out_ch/groups, kernel]`
/// (PyTorch `ConvTranspose1d` layout). Returns `[out_ch, out_len]` with
/// `out_len = (in_len − 1)·stride − 2·pad + kernel` (output_padding = 0).
#[allow(clippy::too_many_arguments)]
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
                    // ot = it·stride + kk − pad
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

/// Layer norm over the channel axis of a `[channels, time]` signal (VITS
/// `attentions.LayerNorm`: normalise the channel vector at each time step, then
/// affine with `gamma`/`beta`).
pub(crate) fn layer_norm_channels(
    x: &[f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for t in 0..time {
        let mut mean = 0.0f32;
        for c in 0..channels {
            mean += x[c * time + t];
        }
        mean /= channels as f32;
        let mut var = 0.0f32;
        for c in 0..channels {
            let d = x[c * time + t] - mean;
            var += d * d;
        }
        var /= channels as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..channels {
            out[c * time + t] = (x[c * time + t] - mean) * inv * gamma[c] + beta[c];
        }
    }
    out
}

/// In-place LeakyReLU (`x < 0 → slope·x`).
pub(crate) fn leaky_relu(x: &mut [f32], slope: f32) {
    for v in x {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// Exact (erf-based) GELU, matching PyTorch `F.gelu` default.
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26; ~1e-7 max error — well inside
/// the FP32 parity bound).
#[allow(clippy::excessive_precision)] // A&S reference coefficients kept verbatim
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

/// Softplus `ln(1 + eˣ)` with the large-`x` guard PyTorch uses.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Row-wise softmax over the innermost axis of a `rows × cols` buffer,
/// stabilised by the row max (in place).
pub(crate) fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..r * cols + cols];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_matches_hand_fixture() {
        // 1 in-ch, len 5 = 1..5, weight [1,1,3]=[1,1,1], stride 1, pad 0, dil 1.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0, 1.0, 1.0];
        let (out, tout) = conv1d(&x, 1, 5, &w, 1, 3, None, 1, 0, 1, 1);
        assert_eq!(tout, 3);
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn conv1d_dilation_and_padding() {
        // len 5, kernel 3, dilation 2, pad 2 → same length 5.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0, 0.0, -1.0]; // taps at t-2 and t+2 (dilation 2)
        let (out, tout) = conv1d(&x, 1, 5, &w, 1, 3, None, 1, 2, 2, 1);
        assert_eq!(tout, 5);
        // out[t] = x[t-2]*1 + x[t+2]*(-1) (zero-padded).
        // t0: 0 - x2 = -3; t1: 0 - x3 = -4; t2: x0 - x4 = 1-5=-4;
        // t3: x1 - 0 = 2; t4: x2 - 0 = 3.
        assert_eq!(out, [-3.0, -4.0, -4.0, 2.0, 3.0]);
    }

    #[test]
    fn conv1d_depthwise_groups() {
        // 2 channels, groups=2 (depthwise): each channel its own kernel.
        let x = [1.0, 2.0, 3.0, /* ch1 */ 10.0, 20.0, 30.0];
        let w = [1.0, 1.0, /* ch1 */ 2.0, 2.0]; // [2,1,2]
        let (out, tout) = conv1d(&x, 2, 3, &w, 2, 2, None, 1, 0, 1, 2);
        assert_eq!(tout, 2);
        // ch0: 1+2, 2+3 = 3,5; ch1: 2*(10+20), 2*(20+30) = 60,100.
        assert_eq!(out, [3.0, 5.0, 60.0, 100.0]);
    }

    #[test]
    fn conv_transpose_upsamples() {
        // in 1ch len 2, weight [1,1,2]=[1,1], stride 2, pad 0 → len 4.
        let x = [1.0, 2.0];
        let w = [1.0, 1.0];
        let (out, tout) = conv_transpose1d(&x, 1, 2, &w, 1, 2, None, 2, 0, 1);
        assert_eq!(tout, 4);
        // x0 spreads to ot 0,1; x1 to ot 2,3.
        assert_eq!(out, [1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn conv_transpose_groups_independent() {
        // 2 subbands, groups=2, stride 2, weight [2,1,2].
        let x = [1.0, 0.0, /* ch1 */ 0.0, 3.0];
        let w = [1.0, 1.0, /* ch1 */ 2.0, 2.0];
        let (out, tout) = conv_transpose1d(&x, 2, 2, &w, 2, 2, None, 2, 0, 2);
        assert_eq!(tout, 4);
        // ch0: x0=1 → ot0,1 = 1,1; x1=0 → nothing. => [1,1,0,0]
        // ch1: x0=0 → nothing; x1=3 → ot2,3 = 6,6. => [0,0,6,6]
        assert_eq!(out, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0]);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        // 2 channels, 1 time step: values [1, 3] → mean 2, normalised ±1.
        let x = [1.0, 3.0];
        let out = layer_norm_channels(&x, 2, 1, &[1.0, 1.0], &[0.0, 0.0], 1e-5);
        assert!((out[0] + 1.0).abs() < 1e-3);
        assert!((out[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn erf_and_gelu_reference_points() {
        assert!(erf(0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
        assert!(gelu(0.0).abs() < 1e-6);
        // GELU(1) ≈ 0.8413.
        assert!((gelu(1.0) - 0.841_345).abs() < 1e-4);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0];
        softmax_rows(&mut x, 2, 3);
        assert!((x[0] + x[1] + x[2] - 1.0).abs() < 1e-6);
        assert!((x[3] - 1.0 / 3.0).abs() < 1e-6);
    }
}
