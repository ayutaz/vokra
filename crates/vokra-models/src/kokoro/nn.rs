//! Compute primitives for the Kokoro-82M native TTS (M2-07-T09/T10/T16).
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
//! In addition, this module hosts three private composition helpers that avoid
//! introducing new first-class `vokra-ops` ops (FR-EX-08 permits composition;
//! `docs/adr/0007-kokoro-native.md` §"Op gap analysis" records why):
//!
//! * [`adain`] — StyleTTS 2 Adaptive Instance Normalisation as
//!   instance-norm + affine (used by the decoder body).
//! * [`weight_norm_reconstruct_1d`] — reconstructs `w = g · v / ||v||_2` from
//!   the two `weight_g` / `weight_v` tensors PyTorch's
//!   `torch.nn.utils.weight_norm` splits a Conv1d weight into. Consumed by
//!   the text encoder, prosody predictor, and decoder (M2-07-T13/T14/T15).
//! * [`BiLstm1d`] — native bidirectional LSTM forward with the PyTorch gate
//!   layout (`weight_ih_l0` = `[4·H, I]`, stacked `i | f | g | o`).
//! * [`adaln_1d`] — AdaLayerNorm (StyleTTS 2 派生) as `Linear(style → 2·C)`
//!   composed with instance-norm + per-channel affine, on a
//!   `[t, channels]` row-major buffer. The AdaLN naming distinguishes it
//!   from the channel-major [`adain`] used by the decoder body — same math,
//!   different layout.
//!
//! `EPS = 1e-5` is the standard `nn.InstanceNorm1d` default (aligned with
//! `LAYER_NORM_EPS` in the piper config).

use vokra_core::{Result, VokraError};

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

/// Reconstructs `w = g · v / ||v||_2` from the two tensors
/// `torch.nn.utils.weight_norm` splits a Conv1d weight into (M2-07-T16).
///
/// PyTorch's `torch.nn.utils.weight_norm(module, dim=0)` re-parameterises a
/// weight `w` of shape `[out_ch, in_ch, kernel_size]` as
/// `w = g · v / ||v||_axis0` where the norm is a per-output-channel L2 over
/// the remaining axes (`in_ch × kernel_size`), and `g` is broadcast as
/// `[out_ch, 1, 1]`. This helper accepts the flattened `[out_ch]` `g` and the
/// row-major `[out_ch · in_ch · kernel_size]` `v` and returns the row-major
/// reconstructed `w` in the identical layout the Kokoro converter (M2-07-T07)
/// writes and every downstream `conv1d` call in this module expects.
///
/// Zero-norm channels degrade gracefully to a zero row rather than producing
/// a NaN — an all-zero `v[oc, :, :]` is a degenerate but not invalid input
/// (the smoke fixture in `mod.rs::synthesize_smoke_produces_expected_shape`
/// leans on it) and PyTorch's own re-parameterisation would divide-by-zero
/// there. The runtime path never sees this in a real Kokoro weight, so the
/// choice does not affect parity.
#[allow(dead_code)] // consumed by the T13-alpha text encoder rewrite
pub(crate) fn weight_norm_reconstruct_1d(
    g: &[f32],
    v: &[f32],
    out_ch: usize,
    in_ch: usize,
    kernel_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(g.len(), out_ch, "weight_norm: g len != out_ch");
    debug_assert_eq!(
        v.len(),
        out_ch * in_ch * kernel_size,
        "weight_norm: v len != out_ch·in_ch·kernel_size"
    );
    let plane = in_ch * kernel_size;
    let mut w = vec![0.0f32; out_ch * plane];
    for (oc, &g_oc) in g.iter().enumerate().take(out_ch) {
        let base = oc * plane;
        let vslice = &v[base..base + plane];
        let mut sq = 0.0f32;
        for &x in vslice {
            sq += x * x;
        }
        let norm = sq.sqrt();
        // Zero-norm → zero row (see fn doc). A well-trained checkpoint never
        // hits this path; degenerate synthetic fixtures do.
        let scale = if norm > 0.0 { g_oc / norm } else { 0.0 };
        let dst = &mut w[base..base + plane];
        for (d, &s) in dst.iter_mut().zip(vslice) {
            *d = s * scale;
        }
    }
    w
}

/// Native bidirectional LSTM forward (M2-07-T16).
///
/// PyTorch's `nn.LSTM(input_size, hidden_size, bidirectional=True)` stores its
/// weights as `weight_ih_l0[4·H, I]` / `weight_hh_l0[4·H, H]` and biases as
/// `bias_ih_l0[4·H]` / `bias_hh_l0[4·H]` (mirrored by `..._reverse` for the
/// backward direction). The `4·H` first-axis is a stack of the four gates in
/// **`i | f | g | o` order** — verified against the manifest at
/// `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv` (the Kokoro
/// text encoder's `lstm.weight_ih_l0` has shape `(1024, 512)` = `4·256, 512`).
///
/// The forward output layout is `[seq_len, 2·H]` row-major, with the forward
/// direction's `h_t` in the first `H` columns and the reverse direction's `h_t`
/// in the second `H` columns — the exact layout PyTorch's
/// `nn.LSTM(batch_first=True)` returns.
#[allow(dead_code)] // consumed by the T13-alpha text encoder rewrite
#[derive(Debug)]
pub(crate) struct BiLstm1d {
    input_dim: usize,
    hidden_dim: usize,
    w_ih: [Vec<f32>; 2], // [fwd, rev] each row-major [4·H, I]
    w_hh: [Vec<f32>; 2], // [fwd, rev] each row-major [4·H, H]
    b_ih: [Vec<f32>; 2], // [fwd, rev] each [4·H]
    b_hh: [Vec<f32>; 2], // [fwd, rev] each [4·H]
}

impl BiLstm1d {
    /// Builds the bi-directional LSTM from validated weights.
    ///
    /// Every buffer length is cross-checked against
    /// `(input_dim, hidden_dim)`; a length mismatch is a loud
    /// [`VokraError::InvalidArgument`] naming the specific tensor so the caller
    /// can trace back to the offending `weight_ih_l0` / `weight_hh_l0` /
    /// `bias_ih_l0` / `bias_hh_l0` (+ `_reverse`) tensor in the GGUF
    /// (FR-EX-08 — never a silent shape truncation).
    #[allow(dead_code, clippy::too_many_arguments)] // consumed by the T13-alpha text encoder rewrite
    pub(crate) fn new(
        input_dim: usize,
        hidden_dim: usize,
        w_ih_fwd: Vec<f32>,
        w_hh_fwd: Vec<f32>,
        b_ih_fwd: Vec<f32>,
        b_hh_fwd: Vec<f32>,
        w_ih_rev: Vec<f32>,
        w_hh_rev: Vec<f32>,
        b_ih_rev: Vec<f32>,
        b_hh_rev: Vec<f32>,
    ) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "BiLstm1d: input_dim ({input_dim}) and hidden_dim ({hidden_dim}) must be > 0"
            )));
        }
        let g = 4 * hidden_dim;
        let want_ih = g * input_dim;
        let want_hh = g * hidden_dim;
        Self::check_len("weight_ih_l0", &w_ih_fwd, want_ih)?;
        Self::check_len("weight_hh_l0", &w_hh_fwd, want_hh)?;
        Self::check_len("bias_ih_l0", &b_ih_fwd, g)?;
        Self::check_len("bias_hh_l0", &b_hh_fwd, g)?;
        Self::check_len("weight_ih_l0_reverse", &w_ih_rev, want_ih)?;
        Self::check_len("weight_hh_l0_reverse", &w_hh_rev, want_hh)?;
        Self::check_len("bias_ih_l0_reverse", &b_ih_rev, g)?;
        Self::check_len("bias_hh_l0_reverse", &b_hh_rev, g)?;
        Ok(Self {
            input_dim,
            hidden_dim,
            w_ih: [w_ih_fwd, w_ih_rev],
            w_hh: [w_hh_fwd, w_hh_rev],
            b_ih: [b_ih_fwd, b_ih_rev],
            b_hh: [b_hh_fwd, b_hh_rev],
        })
    }

    fn check_len(tensor: &str, buf: &[f32], want: usize) -> Result<()> {
        if buf.len() != want {
            return Err(VokraError::InvalidArgument(format!(
                "BiLstm1d: `{tensor}` length {}, expected {want}",
                buf.len()
            )));
        }
        Ok(())
    }

    /// Bidirectional LSTM forward.
    ///
    /// `input` is `[seq_len, input_dim]` row-major, output is
    /// `[seq_len, 2·hidden_dim]` row-major with the forward direction's `h_t`
    /// occupying columns `0..hidden_dim` and the reverse direction's `h_t`
    /// occupying columns `hidden_dim..2·hidden_dim` (matching
    /// `nn.LSTM(batch_first=True, bidirectional=True)`).
    ///
    /// Cell + hidden states start at zero (PyTorch default when no initial
    /// state is provided). No peephole connections.
    #[allow(dead_code)] // consumed by the T13-alpha text encoder rewrite
    pub(crate) fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        debug_assert_eq!(
            input.len(),
            seq_len * self.input_dim,
            "BiLstm1d::forward: input len != seq_len·input_dim"
        );
        let h = self.hidden_dim;
        let mut output = vec![0.0f32; seq_len * 2 * h];
        if seq_len == 0 {
            return output;
        }

        // Forward direction.
        let mut hs = vec![0.0f32; h];
        let mut cs = vec![0.0f32; h];
        let mut gates = vec![0.0f32; 4 * h];
        for t in 0..seq_len {
            let x_t = &input[t * self.input_dim..(t + 1) * self.input_dim];
            self.step(0, x_t, &mut hs, &mut cs, &mut gates);
            let dst = &mut output[t * 2 * h..t * 2 * h + h];
            dst.copy_from_slice(&hs);
        }

        // Reverse direction — reset state and iterate t = seq_len-1..0.
        hs.fill(0.0);
        cs.fill(0.0);
        for t in (0..seq_len).rev() {
            let x_t = &input[t * self.input_dim..(t + 1) * self.input_dim];
            self.step(1, x_t, &mut hs, &mut cs, &mut gates);
            let dst = &mut output[t * 2 * h + h..(t + 1) * 2 * h];
            dst.copy_from_slice(&hs);
        }

        output
    }

    /// One LSTM cell step: mutates `h` / `c` in place.
    ///
    /// Formula (PyTorch `nn.LSTMCell` standard, no peephole):
    ///
    /// ```text
    /// i = σ(W_ii·x + b_ii + W_hi·h + b_hi)
    /// f = σ(W_if·x + b_if + W_hf·h + b_hf)
    /// g = tanh(W_ig·x + b_ig + W_hg·h + b_hg)
    /// o = σ(W_io·x + b_io + W_ho·h + b_ho)
    /// c' = f · c + i · g
    /// h' = o · tanh(c')
    /// ```
    ///
    /// The gate stack is `i | f | g | o` at rows `[0..H, H..2H, 2H..3H, 3H..4H]`
    /// of `w_ih` / `w_hh` — this is the PyTorch layout, cross-referenced
    /// against the Kokoro tensor manifest.
    fn step(&self, dir: usize, x: &[f32], h: &mut [f32], c: &mut [f32], gates: &mut [f32]) {
        let hd = self.hidden_dim;
        let idim = self.input_dim;
        let w_ih = &self.w_ih[dir];
        let w_hh = &self.w_hh[dir];
        let b_ih = &self.b_ih[dir];
        let b_hh = &self.b_hh[dir];

        // gates[i] = b_ih[i] + b_hh[i] + Σ_j W_ih[i,j]·x[j] + Σ_j W_hh[i,j]·h[j]
        for i in 0..(4 * hd) {
            let ih_row = &w_ih[i * idim..(i + 1) * idim];
            let hh_row = &w_hh[i * hd..(i + 1) * hd];
            let mut acc = b_ih[i] + b_hh[i];
            for j in 0..idim {
                acc += ih_row[j] * x[j];
            }
            for j in 0..hd {
                acc += hh_row[j] * h[j];
            }
            gates[i] = acc;
        }
        // Apply activations + update h, c.
        for j in 0..hd {
            let ig = sigmoid(gates[j]);
            let fg = sigmoid(gates[hd + j]);
            let gg = gates[2 * hd + j].tanh();
            let og = sigmoid(gates[3 * hd + j]);
            let new_c = fg * c[j] + ig * gg;
            let new_h = og * new_c.tanh();
            c[j] = new_c;
            h[j] = new_h;
        }
    }
}

/// StyleTTS 2 派生 AdaLayerNorm on a `[t, channels]` row-major buffer
/// (M2-07-T16).
///
/// Composition of two existing ops (FR-EX-08 permits composition; the ADR
/// records why no new first-class `adaln` op is added):
///
/// 1. Project the style vector `[style_dim]` through a Linear
///    `fc_w[2·channels, style_dim]` + `fc_b[2·channels]` to get
///    `(γ, β)` split at index `channels`.
/// 2. Normalise `x` across the time axis per channel
///    (`(x - mean) / sqrt(var + EPS)`, matching PyTorch's `InstanceNorm1d`).
/// 3. Affine `y = γ · normalised + β` broadcast per channel.
///
/// This differs from [`adain`] only in the input layout: [`adain`] operates
/// on channel-major `[channels, time]` buffers (the decoder body convention)
/// while [`adaln_1d`] operates on row-major `[t, channels]` buffers (the
/// AdaLayerNorm convention the prosody predictor and text-encoder-adjacent
/// paths use). The math is identical.
#[allow(dead_code, clippy::too_many_arguments)] // consumed by the T13/T14 prosody rewrite
pub(crate) fn adaln_1d(
    x: &[f32],
    t: usize,
    channels: usize,
    fc_w: &[f32],
    fc_b: &[f32],
    style: &[f32],
    style_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), t * channels, "adaln_1d: x len mismatch");
    debug_assert_eq!(out.len(), t * channels, "adaln_1d: out len mismatch");
    debug_assert_eq!(style.len(), style_dim, "adaln_1d: style len mismatch");
    debug_assert_eq!(
        fc_w.len(),
        2 * channels * style_dim,
        "adaln_1d: fc_w len mismatch"
    );
    debug_assert_eq!(fc_b.len(), 2 * channels, "adaln_1d: fc_b len mismatch");
    if t == 0 || channels == 0 {
        return;
    }

    // 1. Project style → (γ, β) via a row-major Linear.
    let mut gamma_beta = vec![0.0f32; 2 * channels];
    for i in 0..(2 * channels) {
        let mut acc = fc_b[i];
        let row = &fc_w[i * style_dim..(i + 1) * style_dim];
        for j in 0..style_dim {
            acc += row[j] * style[j];
        }
        gamma_beta[i] = acc;
    }
    let (gamma, beta) = gamma_beta.split_at(channels);

    // 2. Per-channel InstanceNorm across time, then affine.
    let inv_t = 1.0 / t as f32;
    for c in 0..channels {
        let mut mean = 0.0f32;
        for ti in 0..t {
            mean += x[ti * channels + c];
        }
        mean *= inv_t;
        let mut var = 0.0f32;
        for ti in 0..t {
            let d = x[ti * channels + c] - mean;
            var += d * d;
        }
        var *= inv_t;
        let inv = 1.0 / (var + EPS).sqrt();
        let g = gamma[c];
        let b = beta[c];
        for ti in 0..t {
            let src = x[ti * channels + c];
            out[ti * channels + c] = g * (src - mean) * inv + b;
        }
    }
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

    /// [`weight_norm_reconstruct_1d`] must reproduce
    /// `w = g · v / ||v||_2` per output channel with the L2 norm computed
    /// over the `(in_ch, kernel_size)` plane. The scalar oracle uses
    /// out_ch = 2, in_ch = 1, kernel = 3 with hand-chosen norms `||v[0]|| = 3`
    /// and `||v[1]|| = 5` so the expected weight rows are integer-valued and
    /// a mis-index across the plane is instantly visible.
    #[test]
    fn weight_norm_reconstruct_matches_scalar_oracle() {
        // v[0, 0, :] = [1, 2, 2] → ||v[0]|| = √(1+4+4) = 3
        // v[1, 0, :] = [3, 4, 0] → ||v[1]|| = √(9+16+0) = 5
        let v = vec![1.0, 2.0, 2.0, 3.0, 4.0, 0.0];
        let g = vec![3.0, 10.0]; // g[oc]
        let w = weight_norm_reconstruct_1d(&g, &v, 2, 1, 3);
        // Expected per-channel scaling: 3/3 = 1 (ch 0), 10/5 = 2 (ch 1).
        // ch 0: [1·1, 1·2, 1·2] = [1, 2, 2]
        // ch 1: [2·3, 2·4, 2·0] = [6, 8, 0]
        let want = vec![1.0, 2.0, 2.0, 6.0, 8.0, 0.0];
        assert_eq!(w.len(), want.len());
        for (i, (&a, &b)) in w.iter().zip(&want).enumerate() {
            assert!((a - b).abs() < 1e-6, "idx {i}: got {a}, want {b}");
        }
    }

    /// A zero-norm `v` row degrades to a zero output row rather than NaN
    /// (documented in the fn doc). A well-trained checkpoint never hits this
    /// path, but the synthetic smoke fixture in `mod.rs` does.
    #[test]
    fn weight_norm_reconstruct_zero_norm_yields_zero_row() {
        // v[0] all zeros, v[1] non-zero.
        let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let g = vec![5.0, 7.0];
        let w = weight_norm_reconstruct_1d(&g, &v, 2, 1, 3);
        // Row 0: all zeros; row 1: 7·[1,0,0]/1 = [7,0,0].
        assert_eq!(w, vec![0.0, 0.0, 0.0, 7.0, 0.0, 0.0]);
    }

    /// [`BiLstm1d::forward`] must implement the PyTorch gate stack
    /// `i | f | g | o` and match the standard `nn.LSTMCell` formula. The
    /// reference reads each gate's weights from row-major
    /// `w_ih[gate * I..]` / `w_hh[gate * H..]` explicitly, so a gate-order
    /// swap in the implementation would visibly diverge.
    #[test]
    fn bilstm_1d_matches_scalar_oracle_two_step() {
        let input_dim = 1;
        let hidden_dim = 1;
        let seq_len = 2;

        // Distinct per-gate values so any gate-order swap in the impl is
        // caught. Both directions share the same weights to keep the oracle
        // compact — the reverse-direction correctness comes from processing
        // the sequence in reverse, not from a different weight set.
        let w_ih = [1.0f32, 0.5, 1.0, 0.25]; // i, f, g, o rows (each length I=1)
        let w_hh = [0.1f32, 0.05, 0.1, 0.025];
        let b_ih = [0.7f32, 0.3, 0.4, 0.2];
        let b_hh = [0.0f32; 4];

        let lstm = BiLstm1d::new(
            input_dim,
            hidden_dim,
            w_ih.to_vec(),
            w_hh.to_vec(),
            b_ih.to_vec(),
            b_hh.to_vec(),
            w_ih.to_vec(),
            w_hh.to_vec(),
            b_ih.to_vec(),
            b_hh.to_vec(),
        )
        .expect("valid weights build the LSTM");

        let input = [1.0f32, 2.0]; // seq_len=2, input_dim=1
        let out = lstm.forward(&input, seq_len);
        assert_eq!(out.len(), seq_len * 2 * hidden_dim);

        // Reference: one LSTM cell step following PyTorch's `nn.LSTMCell`.
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let step = |x: f32, h: f32, c: f32| -> (f32, f32) {
            // Read gate weights per PyTorch layout: gate order (i, f, g, o).
            let pre_i = w_ih[0] * x + w_hh[0] * h + b_ih[0] + b_hh[0];
            let pre_f = w_ih[1] * x + w_hh[1] * h + b_ih[1] + b_hh[1];
            let pre_g = w_ih[2] * x + w_hh[2] * h + b_ih[2] + b_hh[2];
            let pre_o = w_ih[3] * x + w_hh[3] * h + b_ih[3] + b_hh[3];
            let ig = sig(pre_i);
            let fg = sig(pre_f);
            let gg = pre_g.tanh();
            let og = sig(pre_o);
            let new_c = fg * c + ig * gg;
            let new_h = og * new_c.tanh();
            (new_h, new_c)
        };

        // Forward direction: t = 0, 1.
        let (fwd_h0, fwd_c0) = step(input[0], 0.0, 0.0);
        let (fwd_h1, _) = step(input[1], fwd_h0, fwd_c0);
        // Reverse direction: t = 1, 0.
        let (rev_h1, rev_c1) = step(input[1], 0.0, 0.0);
        let (rev_h0, _) = step(input[0], rev_h1, rev_c1);

        // Output layout `[seq_len, 2·H] = [2, 2]` row-major:
        //   [fwd_h0, rev_h0,
        //    fwd_h1, rev_h1]
        let want = [fwd_h0, rev_h0, fwd_h1, rev_h1];
        for (i, (&g, &w)) in out.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "idx {i}: got {g}, want {w} (fwd_h0={fwd_h0}, rev_h0={rev_h0}, fwd_h1={fwd_h1}, rev_h1={rev_h1})"
            );
        }
    }

    /// A wrong-length `weight_ih_l0` must fail loudly at [`BiLstm1d::new`]
    /// with a message naming the offending tensor (FR-EX-08 — never a silent
    /// truncation).
    #[test]
    fn bilstm_1d_rejects_wrong_shape_weights() {
        let input_dim = 2;
        let hidden_dim = 3;
        let g = 4 * hidden_dim;
        // w_ih_fwd has the WRONG length (need g·input_dim = 24, supply 23).
        let bad = vec![0.0f32; g * input_dim - 1];
        let ok = |n: usize| vec![0.0f32; n];
        let err = BiLstm1d::new(
            input_dim,
            hidden_dim,
            bad,
            ok(g * hidden_dim),
            ok(g),
            ok(g),
            ok(g * input_dim),
            ok(g * hidden_dim),
            ok(g),
            ok(g),
        )
        .expect_err("wrong-length w_ih_fwd must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("weight_ih_l0"),
                    "error should name the offending tensor; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// [`adaln_1d`] must reproduce
    /// `γ · (x - mean)/sqrt(var + eps) + β` per channel, with `(γ, β)`
    /// projected from `style` via the Linear. The scalar oracle explicitly
    /// runs `Linear` + `InstanceNorm1d` + affine as three steps and compares
    /// pointwise — a mis-index in the split or the affine broadcast is
    /// instantly visible.
    #[test]
    fn adaln_1d_matches_scalar_oracle() {
        // channels=2, t=3, style_dim=1; hand-chosen weights so
        //   fc([1]) = [2, 0.5, 10, -1] → γ = [2, 0.5], β = [10, -1].
        let t = 3;
        let channels = 2;
        let style_dim = 1;
        let x = vec![
            // row-major [t, channels]:
            // t=0: (c=0)=1, (c=1)=10
            // t=1: (c=0)=2, (c=1)=20
            // t=2: (c=0)=3, (c=1)=30
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0,
        ];
        let fc_w = vec![2.0, 0.5, 10.0, -1.0]; // shape [4, 1]
        let fc_b = vec![0.0f32; 4];
        let style = vec![1.0f32];

        let mut out = vec![0.0f32; t * channels];
        adaln_1d(&x, t, channels, &fc_w, &fc_b, &style, style_dim, &mut out);

        // Oracle: InstanceNorm per channel + affine with hand-computed γ / β.
        let gammas = [2.0f32, 0.5];
        let betas = [10.0f32, -1.0];
        let mut want = vec![0.0f32; t * channels];
        for c in 0..channels {
            let vals: Vec<f32> = (0..t).map(|ti| x[ti * channels + c]).collect();
            let mean = vals.iter().sum::<f32>() / t as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t as f32;
            let inv = 1.0 / (var + EPS).sqrt();
            for ti in 0..t {
                let src = x[ti * channels + c];
                want[ti * channels + c] = gammas[c] * (src - mean) * inv + betas[c];
            }
        }
        for (i, (&g, &w)) in out.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "idx {i}: got {g}, want {w}");
        }
    }

    /// Setting `fc(style) = [1s..., 0s...]` (γ=1, β=0) must reduce
    /// [`adaln_1d`] to a plain InstanceNorm per channel — the resulting
    /// output has ≈0 mean and ≈1 variance per channel (with EPS negligible
    /// vs the input variance).
    #[test]
    fn adaln_1d_gamma_1_beta_0_yields_zero_mean_unit_var() {
        let t = 4;
        let channels = 2;
        let style_dim = 3;
        // Two channels with very different means so a shared-scratch bug
        // would visibly bleed across channels.
        let x = vec![
            1.0, 100.0, // t=0
            2.0, 200.0, // t=1
            3.0, 300.0, // t=2
            4.0, 400.0, // t=3
        ];
        // fc_w = zeros; fc_b = [γ=1, γ=1, β=0, β=0] — irrespective of style,
        // fc(style) = fc_b = [1, 1, 0, 0].
        let fc_w = vec![0.0f32; 2 * channels * style_dim];
        let fc_b = vec![1.0, 1.0, 0.0, 0.0];
        let style = vec![0.5f32, -0.3, 0.7]; // arbitrary — must not leak.

        let mut out = vec![0.0f32; t * channels];
        adaln_1d(&x, t, channels, &fc_w, &fc_b, &style, style_dim, &mut out);

        for c in 0..channels {
            let vals: Vec<f32> = (0..t).map(|ti| out[ti * channels + c]).collect();
            let mean = vals.iter().sum::<f32>() / t as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t as f32;
            assert!(mean.abs() < 1e-5, "channel {c}: mean = {mean}");
            assert!((var - 1.0).abs() < 1e-3, "channel {c}: var = {var}");
        }
    }
}
