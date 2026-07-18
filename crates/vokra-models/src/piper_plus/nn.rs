//! Compute primitives for the piper-plus native TTS (M0-07 / M1-01-D).
//!
//! MB-iSTFT-VITS2 needs dilated, grouped and transposed 1-D convolutions plus
//! per-position linear layers, in the `[channels, time]` layout PyTorch/ONNX
//! convolutions expect. Tensors are plain row-major `Vec<f32>` with explicit
//! shapes; parity is FP32 (NFR-QL-01).
//!
//! # RTF hot path (M1-01-D, ADR-0002 follow-up)
//!
//! [`conv1d`] — the decoder/flow work-horse (conv_pre, the six dilated ResBlock
//! convs, subband_conv_post, the flow WN dilated convs and the 1×1 pre/post
//! projections) — no longer runs a scalar triple loop. It lowers to **im2col +
//! [`Compute::gemm_f32`]**, so the dominant matmuls ride the selected backend's
//! dispatched GEMM (CPU SIMD AVX2 / NEON, or the Metal GPU — M2-01 Phase 3) at
//! run time. The M0-08 `conv1d_f32` kernel has no
//! dilation/groups, so the im2col lives here (per the design: keep the backend
//! API unchanged; M1-05 may later absorb dilation/groups into the backend and
//! this can call it directly). The FP32 reduction order differs from the scalar
//! loop, so results match within the FP32 parity bound, not bit-for-bit — a
//! differential test pins `conv1d` to the scalar oracle ([`conv1d_scalar`]).
//!
//! [`conv_transpose1d`] (the two decoder upsamples + PQMF synthesis) stays
//! scalar for now: the MRF ResBlock stack dominates the FLOPs, and whether to
//! route the transposed convs through GEMM+col2im is decided from the first RTF
//! measurement (M1-01-F), not up front.

use crate::compute::Compute;
use crate::tls_scratch::{with_col_scratch, with_col_scratch2};

/// 1-D convolution with stride / padding / dilation / groups, lowered to
/// im2col + GEMM (M1-01-D), dispatched through the [`Compute`] seam so the model
/// GEMM hot path can run on the CPU kernels or the Metal GPU (M2-01 Phase 3).
///
/// `x` is `[in_ch, in_len]`, `weight` is `[out_ch, in_ch/groups, kernel]`
/// (PyTorch/ONNX layout), `bias` (when `Some`) is `[out_ch]`. Returns
/// `[out_ch, out_len]` with `out_len = (in_len + 2·pad − dilation·(kernel−1) −
/// 1) / stride + 1`.
///
/// # M5-14 Wave-2 glue rework (T16/T17)
///
/// Wave-0 measured 82% of the piper wall OUTSIDE the dispatched kernels,
/// dominated by this function's per-call `vec![0.0; k·out_len]` im2col
/// buffer + element-wise gather + separate scatter. The rework keeps the
/// **identical GEMM operands and the identical output arithmetic** (so the
/// results are bit-for-bit unchanged and the scalar-oracle differential
/// below still pins it), but:
///
/// - the `col` / per-group GEMM buffers come from the grow-only thread-local
///   scratch ([`crate::tls_scratch`]) instead of fresh zeroed allocations;
/// - each im2col row is filled by RANGE: explicit zero-fill of the two pad
///   ranges + a contiguous `copy_from_slice` of the interior for the
///   `stride == 1` shapes this model actually uses (strided scalar loop
///   otherwise) — every element is written, so buffer reuse can never leak
///   stale values;
/// - a pointwise conv (`kernel == 1`, `stride == 1`, `pad == 0`,
///   `groups == 1`) skips im2col entirely — its `col` matrix IS `x`;
/// - for `groups == 1` the GEMM writes straight into `out` and the bias is
///   added in place (one add per element, exactly the old `s + b`).
///
/// The GEMM shapes are derived here and always consistent, so a GEMM shape
/// error would be an internal bug — it panics rather than being threaded as a
/// data error (unlike the `istft` op, whose runtime error the decoder
/// propagates, M1-01-C).
#[allow(clippy::too_many_arguments)]
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
    let k = in_g * kernel; // GEMM reduction dim (im2col rows)
    let mut out = vec![0.0f32; out_ch * out_len];

    // Pointwise fast path: the im2col matrix is exactly `x` — no gather.
    if kernel == 1 && stride == 1 && pad == 0 && groups == 1 {
        compute
            .gemm_f32(out_ch, out_len, in_ch, weight, x, None, &mut out)
            .expect("piper conv1d gemm: internally-consistent shapes");
        add_channel_bias(&mut out, out_len, bias);
        return (out, out_len);
    }

    if groups == 1 {
        with_col_scratch(k * out_len, |col| {
            fill_im2col(
                col, x, 0, in_g, in_len, kernel, out_len, stride, pad, dilation,
            );
            compute
                .gemm_f32(out_ch, out_len, k, weight, col, None, &mut out)
                .expect("piper conv1d gemm: internally-consistent shapes");
        });
        add_channel_bias(&mut out, out_len, bias);
        return (out, out_len);
    }

    // groups > 1 (defensive; no shipping piper conv routes here).
    with_col_scratch2(k * out_len, out_g * out_len, |col, og| {
        for g in 0..groups {
            fill_im2col(
                col,
                x,
                g * in_g,
                in_g,
                in_len,
                kernel,
                out_len,
                stride,
                pad,
                dilation,
            );
            let wbase = g * out_g * k;
            compute
                .gemm_f32(
                    out_g,
                    out_len,
                    k,
                    &weight[wbase..wbase + out_g * k],
                    col,
                    None,
                    og,
                )
                .expect("piper conv1d gemm: internally-consistent shapes");
            // Scatter into `out`, adding the per-output-channel bias.
            for oc in 0..out_g {
                let out_channel = g * out_g + oc;
                let b = bias.map_or(0.0, |b| b[out_channel]);
                let dst = &mut out[out_channel * out_len..out_channel * out_len + out_len];
                for (d, &s) in dst.iter_mut().zip(&og[oc * out_len..(oc + 1) * out_len]) {
                    *d = s + b;
                }
            }
        }
    });
    (out, out_len)
}

/// Adds the per-output-channel bias in place (one add per element — the same
/// single `+ b` the pre-rework scatter applied).
fn add_channel_bias(out: &mut [f32], out_len: usize, bias: Option<&[f32]>) {
    if let Some(bias) = bias {
        for (row, &b) in out.chunks_exact_mut(out_len).zip(bias) {
            for v in row {
                *v += b;
            }
        }
    }
}

/// Fills the im2col matrix `col[(ic·kernel + kk), ot] = x[ic0+ic, ot·stride +
/// kk·dil − pad]` (zero where the tap falls in the padding), writing **every**
/// element of `col[..in_g·kernel·out_len]` so a reused buffer never leaks
/// stale values. Bit-identical contents to the pre-rework `fill(0.0)` +
/// partial-overwrite loop; the interior of each row is one contiguous
/// `copy_from_slice` when `stride == 1` (all shipping piper convs).
#[allow(clippy::too_many_arguments)]
fn fill_im2col(
    col: &mut [f32],
    x: &[f32],
    ic0: usize,
    in_g: usize,
    in_len: usize,
    kernel: usize,
    out_len: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
) {
    for ic in 0..in_g {
        let xrow = (ic0 + ic) * in_len;
        for kk in 0..kernel {
            let crow = (ic * kernel + kk) * out_len;
            let row = &mut col[crow..crow + out_len];
            let kd = kk * dilation;
            // Valid ot range: pad <= ot·stride + kd < pad + in_len.
            let ot_lo = if kd >= pad {
                0
            } else {
                (pad - kd).div_ceil(stride)
            };
            let last_it = pad + in_len - 1;
            let ot_hi = if kd > last_it {
                0
            } else {
                (((last_it - kd) / stride) + 1).min(out_len)
            };
            if ot_lo >= ot_hi {
                row.fill(0.0);
                continue;
            }
            row[..ot_lo].fill(0.0);
            row[ot_hi..].fill(0.0);
            if stride == 1 {
                let src0 = ot_lo + kd - pad;
                row[ot_lo..ot_hi].copy_from_slice(&x[xrow + src0..xrow + src0 + (ot_hi - ot_lo)]);
            } else {
                for (ot, v) in row[ot_lo..ot_hi].iter_mut().enumerate() {
                    let it = (ot_lo + ot) * stride + kd;
                    *v = x[xrow + (it - pad)];
                }
            }
        }
    }
}

/// Reference scalar 1-D convolution — the differential oracle `conv1d` (the
/// im2col + GEMM path) is pinned against. Same signature/semantics as
/// [`conv1d`]; kept test-only so the shipping path is the SIMD one.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_scalar(
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
            // Valid kk range for this input tap: 0 <= it·stride + kk − pad
            // < out_len (M5-14 Wave-2: hoisting the two per-tap guards out of
            // the inner loop turns it into a contiguous, auto-vectorizable
            // axpy over `out` — the iteration order over the surviving
            // (in_channel, it, oc, kk) tuples is UNCHANGED, so every output
            // element's accumulation chain is bit-identical to the guarded
            // loop).
            let base = it * stride;
            let kk_lo = pad.saturating_sub(base);
            let kk_hi = kernel.min((out_len + pad).saturating_sub(base));
            if kk_lo >= kk_hi {
                continue;
            }
            let ot0 = base + kk_lo - pad;
            for oc in 0..out_g {
                let out_channel = g * out_g + oc;
                let wrow = (in_channel * out_g + oc) * kernel;
                let w_seg = &weight[wrow + kk_lo..wrow + kk_hi];
                let dst = &mut out
                    [out_channel * out_len + ot0..out_channel * out_len + ot0 + (kk_hi - kk_lo)];
                for (d, &wv) in dst.iter_mut().zip(w_seg) {
                    *d += xv * wv;
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

/// Per-position linear layer `y = W·x + b` — `W` is `[out, in]` row-major, `x`
/// is `[in]`, `b` is `[out]`, returns `[out]`. The building block of the speaker
/// projection (`spk_proj`) and the prosody / FiLM conditioning heads.
pub(crate) fn linear(weight: &[f32], bias: &[f32], x: &[f32]) -> Vec<f32> {
    let out = bias.len();
    let inn = x.len();
    debug_assert_eq!(
        weight.len(),
        out * inn,
        "linear: weight len {} != out {out} · in {inn}",
        weight.len()
    );
    let mut y = bias.to_vec();
    #[allow(clippy::needless_range_loop)] // row-major matrix indexing
    for o in 0..out {
        let wrow = o * inn;
        let mut acc = y[o];
        for i in 0..inn {
            acc += weight[wrow + i] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// Logistic sigmoid `1/(1 + e^-x)`.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

    /// Deterministic xorshift `[-1, 1)` noise (no external RNG — NFR-DS-02).
    fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
                bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn conv1d_gemm_matches_scalar_oracle() {
        // Each tuple is (in_ch, out_ch, in_len, kernel, stride, pad, dilation,
        // groups), covering every conv shape the decoder / flow route through
        // conv1d (conv_pre & subband_conv_post k7 p3; the ResBlock dilated k3/5/7
        // same-padding; the flow WN k5 d1 p2; the 1×1 pre/post/res_skip) plus a
        // stride>1 and a depthwise (groups>1) case for defensive coverage.
        let cases = [
            (8, 16, 12, 7, 1, 3, 1, 1),   // conv_pre / subband_conv_post shape
            (16, 16, 12, 3, 1, 1, 1, 1),  // ResBlock k3 d1
            (16, 16, 12, 3, 1, 2, 2, 1),  // ResBlock k3 d2 (same padding)
            (16, 16, 14, 5, 1, 4, 2, 1),  // ResBlock k5 d2
            (16, 16, 20, 7, 1, 18, 6, 1), // ResBlock k7 d6 (pad = d*(k-1)/2)
            (12, 24, 10, 5, 1, 2, 1, 1),  // flow WN in_layers k5 d1 p2
            (16, 32, 10, 1, 1, 0, 1, 1),  // 1×1 projection
            (8, 8, 12, 3, 2, 1, 1, 1),    // stride 2
            (8, 8, 12, 3, 1, 1, 1, 2),    // depthwise groups=2
        ];
        for (i, &(in_ch, out_ch, in_len, k, stride, pad, dil, groups)) in cases.iter().enumerate() {
            let x = rand_vec(1 + i as u64, in_ch * in_len);
            let w = rand_vec(101 + i as u64, out_ch * (in_ch / groups) * k);
            let bias = rand_vec(201 + i as u64, out_ch);
            for b in [None, Some(bias.as_slice())] {
                let (got, tg) = conv1d(
                    &Compute::cpu(),
                    &x,
                    in_ch,
                    in_len,
                    &w,
                    out_ch,
                    k,
                    b,
                    stride,
                    pad,
                    dil,
                    groups,
                );
                let (want, ts) = conv1d_scalar(
                    &x, in_ch, in_len, &w, out_ch, k, b, stride, pad, dil, groups,
                );
                assert_eq!(tg, ts, "case {i}: out_len mismatch");
                let d = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(d < 1e-3, "case {i} (bias={}): max|Δ|={d}", b.is_some());
            }
        }
    }

    /// M5-14 Wave-2 bit-identity pin: the reworked conv1d (TLS scratch +
    /// range fills + pointwise / direct-out fast paths) must reproduce the
    /// pre-rework im2col + GEMM + `s + b` scatter **bit-for-bit** — the GEMM
    /// operands are byte-identical and the bias add is the same single `+ b`,
    /// so `==` (not a tolerance) is the correct assertion.
    #[test]
    fn conv1d_bitwise_matches_reference_im2col_path() {
        let cases = [
            (8, 16, 12, 7, 1, 3, 1, 1),
            (16, 16, 12, 3, 1, 2, 2, 1),
            (16, 16, 20, 7, 1, 18, 6, 1),
            (16, 32, 10, 1, 1, 0, 1, 1), // pointwise fast path
            (8, 8, 12, 3, 2, 1, 1, 1),   // stride 2 (strided fill loop)
            (8, 8, 12, 3, 1, 1, 1, 2),   // groups=2 (scatter path)
            (4, 4, 11, 5, 3, 1, 2, 1),   // ragged stride/dilation corner
        ];
        let compute = Compute::cpu();
        for (i, &(in_ch, out_ch, in_len, k, stride, pad, dil, groups)) in cases.iter().enumerate() {
            let x = rand_vec(11 + i as u64, in_ch * in_len);
            let w = rand_vec(111 + i as u64, out_ch * (in_ch / groups) * k);
            let bias = rand_vec(211 + i as u64, out_ch);
            for b in [None, Some(bias.as_slice())] {
                let (got, got_len) = conv1d(
                    &compute, &x, in_ch, in_len, &w, out_ch, k, b, stride, pad, dil, groups,
                );

                // Pre-rework reference: fresh zeroed col + guarded writes +
                // GEMM into a per-group buffer + `s + b` scatter.
                let eff = dil * (k - 1) + 1;
                let out_len = (in_len + 2 * pad - eff) / stride + 1;
                let in_g = in_ch / groups;
                let out_g = out_ch / groups;
                let kk_dim = in_g * k;
                let mut want = vec![0.0f32; out_ch * out_len];
                let mut col = vec![0.0f32; kk_dim * out_len];
                let mut og = vec![0.0f32; out_g * out_len];
                for g in 0..groups {
                    col.fill(0.0);
                    for ic in 0..in_g {
                        let xrow = (g * in_g + ic) * in_len;
                        for kk in 0..k {
                            let crow = (ic * k + kk) * out_len;
                            for ot in 0..out_len {
                                let it = ot * stride + kk * dil;
                                if it >= pad && it - pad < in_len {
                                    col[crow + ot] = x[xrow + (it - pad)];
                                }
                            }
                        }
                    }
                    let wbase = g * out_g * kk_dim;
                    compute
                        .gemm_f32(
                            out_g,
                            out_len,
                            kk_dim,
                            &w[wbase..wbase + out_g * kk_dim],
                            &col,
                            None,
                            &mut og,
                        )
                        .unwrap();
                    for oc in 0..out_g {
                        let out_channel = g * out_g + oc;
                        let bv = b.map_or(0.0, |b| b[out_channel]);
                        let dst = &mut want[out_channel * out_len..out_channel * out_len + out_len];
                        for (d, &s) in dst.iter_mut().zip(&og[oc * out_len..(oc + 1) * out_len]) {
                            *d = s + bv;
                        }
                    }
                }
                assert_eq!(got_len, out_len, "case {i}");
                assert_eq!(
                    got,
                    want,
                    "case {i} (bias={}): not bit-identical",
                    b.is_some()
                );
            }
        }
    }

    #[test]
    fn conv1d_matches_hand_fixture() {
        // 1 in-ch, len 5 = 1..5, weight [1,1,3]=[1,1,1], stride 1, pad 0, dil 1.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0, 1.0, 1.0];
        let (out, tout) = conv1d(&Compute::cpu(), &x, 1, 5, &w, 1, 3, None, 1, 0, 1, 1);
        assert_eq!(tout, 3);
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn conv1d_dilation_and_padding() {
        // len 5, kernel 3, dilation 2, pad 2 → same length 5.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0, 0.0, -1.0]; // taps at t-2 and t+2 (dilation 2)
        let (out, tout) = conv1d(&Compute::cpu(), &x, 1, 5, &w, 1, 3, None, 1, 2, 2, 1);
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
        let (out, tout) = conv1d(&Compute::cpu(), &x, 2, 3, &w, 2, 2, None, 1, 0, 1, 2);
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
    fn conv_transpose_with_padding_trims_symmetrically() {
        // x=[1,2] (1ch,len2), weight [1,1,4]=[1,2,3,4], stride 2, pad 1.
        // out_len = (2-1)·2 + 4 - 2·1 = 4.
        // ConvTranspose1d scatters out[i·stride + k − pad] += x[i]·w[k], dropping
        // taps outside [0,out_len). The full (pad=0) output is [1,2,5,8,6,8];
        // trimming pad=1 from BOTH ends leaves the middle four = [2,5,8,6].
        // (The audit's proposed [1,4,7,6] mixed the pad convention between the
        //  two input taps and is not what this — standard — kernel produces.)
        let x = [1.0, 2.0];
        let w = [1.0, 2.0, 3.0, 4.0];
        let (out, tout) = conv_transpose1d(&x, 1, 2, &w, 1, 4, None, 2, 1, 1);
        assert_eq!(tout, 4);
        assert_eq!(out, [2.0, 5.0, 8.0, 6.0]);
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
    fn softplus_reference_point_and_large_x_guard() {
        // softplus(0) = ln(1+1) = ln 2.
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        // Large-x guard (x > 20): returns x exactly, avoiding exp(x) overflow.
        assert_eq!(softplus(40.0), 40.0);
        // Large negative x: softplus is ln of a value >= 1, so it is
        // non-negative and here underflows to ~0. (The audit's `> 0.0` is not
        // reachable in f32: e^-40 vanishes in the `1.0 + ..` add, giving exactly
        // 0.0 — so the sound bound is 0 <= softplus(-40) < 1e-6.)
        let neg = softplus(-40.0);
        assert!((0.0..1e-6).contains(&neg), "softplus(-40) = {neg}");
    }

    #[test]
    fn linear_matches_hand_fixture() {
        // W = [[1,2,3],[4,5,6]] (out=2,in=3), b=[10,20], x=[1,0,-1].
        // y0 = 10 + (1-3) = 8; y1 = 20 + (4-6) = 18.
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [10.0, 20.0];
        let x = [1.0, 0.0, -1.0];
        assert_eq!(linear(&w, &b, &x), [8.0, 18.0]);
    }

    #[test]
    fn sigmoid_reference_points() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        // sigmoid(2) ≈ 0.880797.
        assert!((sigmoid(2.0) - 0.880_797).abs() < 1e-5);
        // Odd symmetry: sigmoid(-x) = 1 - sigmoid(x).
        assert!((sigmoid(-2.0) - (1.0 - sigmoid(2.0))).abs() < 1e-6);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0];
        softmax_rows(&mut x, 2, 3);
        assert!((x[0] + x[1] + x[2] - 1.0).abs() < 1e-6);
        assert!((x[3] - 1.0 / 3.0).abs() < 1e-6);
    }
}
