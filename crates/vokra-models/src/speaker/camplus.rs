//! Native CAM++ (3D-Speaker) speaker-encoder forward pass (M0-08).
//!
//! Maps an 80-d Kaldi fbank `[t, 80]` to a 192-d speaker embedding, reproducing
//! the verified `campplus.onnx` topology exactly (FR-MD-03, whisper.cpp-style
//! self-reimplementation — no ONNX at runtime):
//!
//! ```text
//! fbank[t,80] ─transpose→ [1,80,t]
//!   FCM: conv1(1→32,3×3) →ReLU
//!        layer1[downsample, identity]  (freq 80→40)
//!        layer2[downsample, identity]  (freq 40→20)
//!        conv2(32→32,3×3, stride(2,1)) →ReLU   (freq 20→10)
//!        reshape [32,10,t] → [320,t]
//!   xvector.tdnn: conv1d(320→128,k5,stride2,pad2) →ReLU       (t→t/2)
//!   block1 (12 layers, dil1) →transit1 (512→256)
//!   block2 (24 layers, dil2) →transit2 (1024→512)
//!   block3 (16 layers, dil2) →transit3 (1024→512, folded out-BN bias)
//!   out_nonlinear: ReLU
//!   stats: [mean; Bessel-std] over time → 1024
//!   dense: conv1d(1024→192,k1,no bias)
//!   affine-free BN → embedding[192]   (NOT L2-normalized)
//! ```
//!
//! Each `CAMDenseTDNNLayer` is `BN→ReLU→linear1(→128)→ReLU→CAM`, where the CAM
//! module gates a dilated local conv `y = linear_local(x)` by a context mask
//! `m = σ(linear2(ReLU(linear1(mean_t(x) + segpool(x)))))`, returning `y·m`
//! dense-concatenated onto the block state (+32 channels).
//!
//! All convolutions are lowered to im2col + GEMM (the dispatched SIMD GEMM, or
//! the Metal GPU GEMM when a Metal backend is selected — the convs route through
//! the [`Compute`] seam, M2-01 Phase 3); this module is `unsafe`-free (workspace
//! `unsafe_code = "deny"`).

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, Result, VokraError};

use super::weights::{Bn, CamPlusWeights, Conv1dW, Conv2dW, ResBlockW};
use crate::compute::{Compute, HotOp};

/// Output speaker-embedding dimension of the supported CAM++ voice.
pub const EMBED_DIM: usize = 192;

/// The backend hot ops CAM++ dispatches: **GEMM only**. Every convolution is
/// lowered to im2col + GEMM here, and the ReLU / sigmoid / BatchNorm / stats glue
/// is model-internal scalar work (not a backend op). So the Metal backend, which
/// covers GEMM, runs the whole forward on the GPU (M2-01 Phase 3).
const CAMPLUS_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

/// A native CAM++ speaker encoder: fbank → 192-d embedding.
///
/// Load once with [`SpeakerEncoder::from_gguf`] / [`SpeakerEncoder::from_path`],
/// then call [`SpeakerEncoder::embed`] per reference utterance. The forward is
/// stateless and `Send + Sync`, so one instance can be shared across threads.
/// The [`BackendKind`] it holds is `Copy` (never a live `!Send` backend), so a
/// Metal-selected encoder stays `Send + Sync`; the `!Send` GPU context is built
/// on the stack inside [`embed`](Self::embed) (M2-01 Phase 3).
pub struct SpeakerEncoder {
    weights: CamPlusWeights,
    backend_kind: BackendKind,
}

impl SpeakerEncoder {
    /// Binds the encoder from a parsed CAM++ GGUF (FR-LD-01). The backend
    /// defaults to [`BackendKind::Cpu`]; select another with
    /// [`with_backend`](Self::with_backend).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            weights: CamPlusWeights::from_gguf(gguf)?,
            backend_kind: BackendKind::Cpu,
        })
    }

    /// Selects the backend the forward runs on (default [`BackendKind::Cpu`]).
    ///
    /// CAM++ dispatches GEMM only, so a GEMM-covering backend (Metal) runs the
    /// whole forward on that backend. Selecting a backend that does not cover the
    /// GEMM hot op — or that has no device — is an explicit error at
    /// [`embed`](Self::embed) time (FR-EX-08), never a silent CPU fall back.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend_kind = backend;
        self
    }

    /// Builds the [`Compute`] dispatcher for the selected backend (GEMM
    /// coverage), on the stack — the `!Send` Metal context never outlives it.
    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend_kind, CAMPLUS_HOT_OPS)
    }

    /// Opens and binds a CAM++ GGUF from `path`.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Computes the 192-d speaker embedding for a Kaldi fbank feature matrix.
    ///
    /// `fbank` is row-major `[t, 80]` (frame-major: `t` frames of 80 mel bins),
    /// so `fbank.len() == t * feat_dim`. Returns the raw (non-L2-normalized)
    /// embedding, ready to feed the piper v7 `spk_proj`. A wrong-length input or
    /// a non-192 voice is an [`VokraError::InvalidArgument`].
    pub fn embed(&self, fbank: &[f32], t: usize) -> Result<[f32; EMBED_DIM]> {
        let emb = self.run(fbank, t, |_, _| {})?;
        if emb.len() != EMBED_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "CAM++: embedding dim {} != {EMBED_DIM}",
                emb.len()
            )));
        }
        let mut out = [0.0f32; EMBED_DIM];
        out.copy_from_slice(&emb);
        Ok(out)
    }

    /// Full forward pass on the encoder's selected backend, returning the raw
    /// 192-d embedding. Builds the [`Compute`] from `self.backend_kind` on the
    /// stack and forwards to [`run_with`](Self::run_with).
    ///
    /// `capture(stage_name, activation)` is invoked at each parity checkpoint
    /// (`post_fcm_reshape`, `post_tdnn`, `post_block1/2/3`, `post_stats`,
    /// `embedding`); production callers pass a no-op closure (zero cost — it
    /// inlines away), while the parity harness collects the intermediates to
    /// localize any divergence from the onnxruntime reference.
    pub(crate) fn run<F: FnMut(&str, &[f32])>(
        &self,
        fbank: &[f32],
        t: usize,
        capture: F,
    ) -> Result<Vec<f32>> {
        self.run_with(&self.compute()?, fbank, t, capture)
    }

    /// Full forward pass on an explicit [`Compute`] (the backend-parity entry:
    /// the CAM++ Metal-vs-CPU test drives the same encoder under both). The CPU
    /// dispatcher reproduces the pre-seam kernel calls bit-for-bit.
    pub(crate) fn run_with<F: FnMut(&str, &[f32])>(
        &self,
        compute: &Compute,
        fbank: &[f32],
        t: usize,
        mut capture: F,
    ) -> Result<Vec<f32>> {
        let w = &self.weights;
        let feat = w.cfg.feat_dim;
        if fbank.len()
            != t.checked_mul(feat)
                .ok_or_else(|| VokraError::InvalidArgument("CAM++: t * feat_dim overflow".into()))?
        {
            return Err(VokraError::InvalidArgument(format!(
                "CAM++: fbank len {} != t({t}) * feat_dim({feat})",
                fbank.len()
            )));
        }
        if t == 0 {
            return Err(VokraError::InvalidArgument("CAM++: empty fbank".into()));
        }

        // --- FCM 2-D residual front-end -----------------------------------
        // Transpose [t, 80] → [1, 80, t] (channel=1, H=freq=80, W=time=t).
        let mut in_map = vec![0.0f32; feat * t];
        for (frame, chunk) in fbank.chunks_exact(feat).enumerate() {
            for (bin, &v) in chunk.iter().enumerate() {
                in_map[bin * t + frame] = v;
            }
        }
        // FCM output is [32, out_freq, t] contiguous == [32·out_freq, t]; that
        // reshape (320 = 32×10) is `post_fcm_reshape`.
        let (post_fcm_reshape, out_freq, w_t) = self.fcm(compute, &in_map, t)?;
        debug_assert_eq!(w_t, t);
        let fcm_ch = w.fcm.conv2.c_out * out_freq; // 32 × 10 = 320
        debug_assert_eq!(post_fcm_reshape.len(), fcm_ch * t);
        capture("post_fcm_reshape", &post_fcm_reshape);

        // --- xvector.tdnn: conv1d(320→128, k5, stride2, pad2) + ReLU ------
        let mut x = conv1d(compute, &post_fcm_reshape, fcm_ch, t, &w.tdnn, 2, 2, 1)?;
        let t_net = x.len() / w.tdnn.c_out;
        relu(&mut x);
        capture("post_tdnn", &x);

        // --- D-TDNN dense blocks + transitions ----------------------------
        let mut channels = w.tdnn.c_out;
        for (bi, block) in w.blocks.iter().enumerate() {
            for layer in &block.layers {
                let cam_out = dtdnn_layer(
                    compute,
                    &x,
                    channels,
                    t_net,
                    layer,
                    block.dilation,
                    w.cfg.cam_seg_len,
                )?;
                // Dense-concat: append the 32-channel CAM output as new rows.
                x.extend_from_slice(&cam_out);
                channels += cam_out.len() / t_net;
            }
            capture(&format!("post_block{}", bi + 1), &x);

            let tr = &w.transitions[bi];
            bn_apply(&mut x, channels, t_net, &tr.bn);
            relu(&mut x);
            x = conv1d(compute, &x, channels, t_net, &tr.linear, 1, 0, 1)?;
            channels = tr.linear.c_out;
        }

        // --- out_nonlinear (ReLU) + statistics pooling --------------------
        relu(&mut x);
        let post_stats = stats_pool(&x, channels, t_net);
        capture("post_stats", &post_stats);

        // --- dense (1024→192, k1, no bias) + affine-free BN ---------------
        let mut emb = conv1d(compute, &post_stats, post_stats.len(), 1, &w.dense, 1, 0, 1)?;
        bn_apply(&mut emb, w.dense.c_out, 1, &w.final_bn);
        capture("embedding", &emb);
        Ok(emb)
    }

    /// Runs the FCM front-end, returning `([32,10,t] map, out_freq=10, t)`.
    fn fcm(&self, compute: &Compute, x: &[f32], t: usize) -> Result<(Vec<f32>, usize, usize)> {
        let f = &self.weights.fcm;
        // conv1: 1→32, 3×3, stride(1,1), pad(1,1); freq stays 80.
        let mut h = conv2d(
            compute,
            x,
            1,
            self.weights.cfg.feat_dim,
            t,
            &f.conv1,
            (1, 1),
            (1, 1),
        )?;
        relu(&mut h);
        let mut freq = self.weights.cfg.feat_dim;

        // layer1: downsample (freq→40) then identity.
        let (h1, fr1) = res_block(compute, &h, 32, freq, t, &f.layer1[0], 2)?;
        let (h2, fr2) = res_block(compute, &h1, 32, fr1, t, &f.layer1[1], 1)?;
        // layer2: downsample (freq→20) then identity.
        let (h3, fr3) = res_block(compute, &h2, 32, fr2, t, &f.layer2[0], 2)?;
        let (h4, fr4) = res_block(compute, &h3, 32, fr3, t, &f.layer2[1], 1)?;
        freq = fr4;

        // conv2: 32→32, 3×3, stride(2,1), pad(1,1); freq→10.
        let mut out = conv2d(compute, &h4, 32, freq, t, &f.conv2, (2, 1), (1, 1))?;
        relu(&mut out);
        let out_freq = (freq + 2 - 3) / 2 + 1;
        Ok((out, out_freq, t))
    }
}

// SpeakerEncoder holds only owned Vec<f32> weights, so it is Send + Sync.
// (Explicit trait objects are not required; this documents the guarantee.)
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SpeakerEncoder>();
};

/// One D-TDNN dense layer → its 32-channel CAM output `[32, t]`.
fn dtdnn_layer(
    compute: &Compute,
    x_in: &[f32],
    c_in: usize,
    t: usize,
    layer: &super::weights::DtdnnLayerW,
    dilation: usize,
    seg_len: usize,
) -> Result<Vec<f32>> {
    // nonlinear1: BN → ReLU.
    let mut h = x_in.to_vec();
    bn_apply(&mut h, c_in, t, &layer.bn1);
    relu(&mut h);
    // linear1 (→128, with folded nonlinear2 bias) → ReLU  ⇒ CAM input `xc`.
    let mut xc = conv1d(compute, &h, c_in, t, &layer.linear1, 1, 0, 1)?;
    relu(&mut xc);
    let bn_ch = layer.linear1.c_out; // 128

    // CAM value branch: dilated local conv (pad = dilation ⇒ same length).
    let y = conv1d(
        compute,
        &xc,
        bn_ch,
        t,
        &layer.cam.linear_local,
        1,
        dilation,
        dilation,
    )?;

    // CAM context: global time-mean (broadcast) + segment-pool (upsampled).
    let mean = time_mean(&xc, t);
    let seg = seg_pool(&xc, bn_ch, t, seg_len);
    let mut ctx = vec![0.0f32; bn_ch * t];
    for (c, &m) in mean.iter().enumerate() {
        let row = c * t;
        for k in 0..t {
            ctx[row + k] = m + seg[row + k];
        }
    }
    // linear1 (→64) → ReLU → linear2 (→32) → Sigmoid ⇒ gate `m`.
    let mut ctx = conv1d(compute, &ctx, bn_ch, t, &layer.cam.linear1, 1, 0, 1)?;
    relu(&mut ctx);
    let cam_ctx = layer.cam.linear1.c_out; // 64
    let mut gate = conv1d(compute, &ctx, cam_ctx, t, &layer.cam.linear2, 1, 0, 1)?;
    for v in &mut gate {
        *v = sigmoid(*v);
    }
    // y · m (element-wise; both [32, t]).
    let mut out = y;
    for (o, g) in out.iter_mut().zip(&gate) {
        *o *= *g;
    }
    Ok(out)
}

/// One FCM `BasicResBlock` → `(out map, out_freq)`.
///
/// `stride` is the frequency-axis stride of `conv1` and the shortcut (2 for the
/// downsampling blocks, 1 for the identity blocks); the time axis is never
/// strided.
fn res_block(
    compute: &Compute,
    x: &[f32],
    ch: usize,
    freq: usize,
    t: usize,
    rb: &ResBlockW,
    stride: usize,
) -> Result<(Vec<f32>, usize)> {
    let mut c1 = conv2d(compute, x, ch, freq, t, &rb.conv1, (stride, 1), (1, 1))?;
    relu(&mut c1);
    let out_freq = (freq + 2 - 3) / stride + 1;
    let mut out = conv2d(compute, &c1, ch, out_freq, t, &rb.conv2, (1, 1), (1, 1))?;

    // Shortcut: 1×1 projection (downsample) or identity.
    match &rb.shortcut {
        Some(sc) => {
            let proj = conv2d(compute, x, ch, freq, t, sc, (stride, 1), (0, 0))?;
            for (o, s) in out.iter_mut().zip(&proj) {
                *o += *s;
            }
        }
        None => {
            for (o, s) in out.iter_mut().zip(x) {
                *o += *s;
            }
        }
    }
    relu(&mut out);
    Ok((out, out_freq))
}

/// 1-D convolution `[c_in, t] → [c_out, t_out]` via im2col + [`Compute::gemm_f32`].
///
/// `weight` is `[c_out, c_in, k]`; the optional per-channel bias is added after
/// the GEMM. `t_out = (t + 2·pad − dil·(k−1) − 1)/stride + 1`.
#[allow(clippy::too_many_arguments)] // conv parameter set + the backend dispatcher
fn conv1d(
    compute: &Compute,
    input: &[f32],
    c_in: usize,
    t: usize,
    w: &Conv1dW,
    stride: usize,
    pad: usize,
    dil: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(c_in, w.c_in, "conv1d input channels != weight c_in");
    debug_assert_eq!(input.len(), c_in * t, "conv1d input len != c_in * t");
    let (c_out, k) = (w.c_out, w.k);
    let eff = dil * (k - 1) + 1;
    if t + 2 * pad < eff {
        return Err(VokraError::InvalidArgument(format!(
            "CAM++ conv1d: padded len {} < effective kernel {eff}",
            t + 2 * pad
        )));
    }
    let t_out = (t + 2 * pad - eff) / stride + 1;

    // im2col patch matrix `col[c_in·k, t_out]`.
    let mut col = vec![0.0f32; c_in * k * t_out];
    for ci in 0..c_in {
        for kk in 0..k {
            let row = (ci * k + kk) * t_out;
            let src = ci * t;
            for to in 0..t_out {
                let pos = (to * stride + kk * dil) as isize - pad as isize;
                if pos >= 0 && (pos as usize) < t {
                    col[row + to] = input[src + pos as usize];
                }
            }
        }
    }

    let mut out = vec![0.0f32; c_out * t_out];
    compute.gemm_f32(c_out, t_out, c_in * k, &w.weight, &col, None, &mut out)?;
    if let Some(bias) = &w.bias {
        for (&b, row) in bias.iter().zip(out.chunks_exact_mut(t_out)) {
            for v in row {
                *v += b;
            }
        }
    }
    Ok(out)
}

/// 2-D convolution `[c_in, h, w] → [c_out, h_out, w_out]` via im2col +
/// [`Compute::gemm_f32`]; `weight` is `[c_out, c_in, kh, kw]` with a mandatory bias.
#[allow(clippy::too_many_arguments)] // conv parameter set + the backend dispatcher
fn conv2d(
    compute: &Compute,
    input: &[f32],
    c_in: usize,
    h: usize,
    w_dim: usize,
    cw: &Conv2dW,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Result<Vec<f32>> {
    debug_assert_eq!(c_in, cw.c_in, "conv2d input channels != weight c_in");
    debug_assert_eq!(
        input.len(),
        c_in * h * w_dim,
        "conv2d input len != c_in * h * w"
    );
    let (c_out, kh, kw) = (cw.c_out, cw.kh, cw.kw);
    let (sh, sw) = stride;
    let (ph, pw) = pad;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w_dim + 2 * pw - kw) / sw + 1;
    let spatial = h_out * w_out;
    let patch = c_in * kh * kw;

    // im2col `col[c_in·kh·kw, h_out·w_out]`.
    let mut col = vec![0.0f32; patch * spatial];
    for ci in 0..c_in {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = ((ci * kh + ky) * kw + kx) * spatial;
                let plane = ci * h * w_dim;
                for ho in 0..h_out {
                    let ih = (ho * sh + ky) as isize - ph as isize;
                    if ih < 0 || ih as usize >= h {
                        continue;
                    }
                    let irow = plane + ih as usize * w_dim;
                    let orow = row + ho * w_out;
                    for wo in 0..w_out {
                        let iw = (wo * sw + kx) as isize - pw as isize;
                        if iw >= 0 && (iw as usize) < w_dim {
                            col[orow + wo] = input[irow + iw as usize];
                        }
                    }
                }
            }
        }
    }

    let mut out = vec![0.0f32; c_out * spatial];
    compute.gemm_f32(c_out, spatial, patch, &cw.weight, &col, None, &mut out)?;
    for (&b, row) in cw.bias.iter().zip(out.chunks_exact_mut(spatial)) {
        for v in row {
            *v += b;
        }
    }
    Ok(out)
}

/// Per-channel BatchNorm affine `y = x·scale + shift` on a `[c, t]` map.
fn bn_apply(x: &mut [f32], c: usize, t: usize, bn: &Bn) {
    for ci in 0..c {
        let (s, sh) = (bn.scale[ci], bn.shift[ci]);
        let base = ci * t;
        for v in &mut x[base..base + t] {
            *v = *v * s + sh;
        }
    }
}

/// In-place ReLU.
fn relu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-channel arithmetic mean over time → one value per `[.., t]` row.
fn time_mean(x: &[f32], t: usize) -> Vec<f32> {
    let inv = 1.0 / t as f32;
    x.chunks_exact(t)
        .map(|row| row.iter().sum::<f32>() * inv)
        .collect()
}

/// CAM `seg_pooling`: `AvgPool1d(k = s = seg_len, ceil_mode)` then
/// nearest-neighbor upsample ×`seg_len` and slice to `t` — i.e. each frame is
/// replaced by the mean of its length-`seg_len` segment. For `t ≤ seg_len` this
/// is exactly the global time-mean.
fn seg_pool(x: &[f32], c: usize, t: usize, seg_len: usize) -> Vec<f32> {
    let n_seg = t.div_ceil(seg_len);
    // Per-segment means (divisor = real element count, matching PyTorch
    // avg_pool1d with ceil_mode and no padding).
    let mut seg = vec![0.0f32; c * n_seg];
    for ci in 0..c {
        let src = ci * t;
        for j in 0..n_seg {
            let start = j * seg_len;
            let end = (start + seg_len).min(t);
            let mut s = 0.0f32;
            for &v in &x[src + start..src + end] {
                s += v;
            }
            seg[ci * n_seg + j] = s / (end - start) as f32;
        }
    }
    // Nearest upsample: frame `k` takes segment `k / seg_len`.
    let mut out = vec![0.0f32; c * t];
    for ci in 0..c {
        let (drow, srow) = (ci * t, ci * n_seg);
        for k in 0..t {
            out[drow + k] = seg[srow + k / seg_len];
        }
    }
    out
}

/// Statistics pooling: concatenated per-channel `[mean; std]` over time, with
/// Bessel-corrected variance `Σ(x − μ)² / (N − 1)` (verified against the graph).
/// Output length is `2·c` (`mean` in `[0, c)`, `std` in `[c, 2c)`).
fn stats_pool(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let n = t as f32;
    let mut out = vec![0.0f32; 2 * c];
    for (ci, chunk) in x.chunks_exact(t).enumerate() {
        let mean = chunk.iter().sum::<f32>() / n;
        let ss: f32 = chunk.iter().map(|&v| (v - mean) * (v - mean)).sum();
        let var = if t > 1 { ss / (n - 1.0) } else { 0.0 };
        out[ci] = mean;
        out[c + ci] = var.max(0.0).sqrt();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seg_pool_le_seglen_is_global_mean() {
        // t = 4 ≤ seg_len = 100 ⇒ one segment ⇒ every frame is the global mean.
        let x = [1.0f32, 2.0, 3.0, 4.0]; // one channel, t = 4.
        let out = seg_pool(&x, 1, 4, 100);
        assert_eq!(out, vec![2.5, 2.5, 2.5, 2.5]);
    }

    #[test]
    fn seg_pool_tiles_and_upsamples() {
        // t = 5, seg_len = 2 ⇒ segments [0,1],[2,3],[4] with means 0.5,2.5,4.0,
        // each broadcast across its 2 (or trailing 1) frames.
        let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let out = seg_pool(&x, 1, 5, 2);
        assert_eq!(out, vec![0.5, 0.5, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn stats_pool_uses_bessel_variance() {
        // x = [0,2,4]: mean 2, Σ(x−μ)² = 8, Bessel var = 8/2 = 4, std = 2.
        let x = [0.0f32, 2.0, 4.0];
        let out = stats_pool(&x, 1, 3);
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn conv1d_identity_pointwise() {
        // 2→2 k=1 identity weight, no pad/stride ⇒ output == input.
        let w = Conv1dW {
            weight: vec![1.0, 0.0, 0.0, 1.0],
            bias: None,
            c_out: 2,
            c_in: 2,
            k: 1,
        };
        let x = [1.0f32, 2.0, 3.0, 4.0]; // [2 ch, t=2]
        let out = conv1d(&Compute::cpu(), &x, 2, 2, &w, 1, 0, 1).unwrap();
        assert_eq!(out, x);
    }
}
