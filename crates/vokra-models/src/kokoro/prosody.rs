//! Kokoro-82M prosody predictor (M2-07-T14) — rewritten against the upstream
//! `predictor.module.*` tensor manifest at
//! `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`.
//!
//! # Architecture
//!
//! Upstream Kokoro's `predictor` is a StyleTTS 2 派生 `ProsodyPredictor`
//! (`docs/adr/0007-kokoro-native.md` §"T02 upstream inspection findings"):
//!
//! ```text
//! # Text-encoder / duration branch (phoneme rate T):
//! text_features [T, d_model]  ⊕  style [style_dim]  → [T, d_model + style_dim]
//! → 3× [ BiLSTM(d_model+style_dim → d_model)
//!        → AdaLayerNorm(x, style)  (LayerNorm across channels + `(1+γ)·x + β`)
//!        → concat(x, style)  → [T, d_model + style_dim] ]
//! → BiLSTM(d_model+style_dim → d_model)        # `predictor.lstm`
//! → Linear(d_model → max_dur=50)               # `duration_proj`
//! → sigmoid.sum(axis=1).round.clamp(≥1)        # per-phoneme integer duration
//!
//! # Alignment (length regulation): [T, d_model+style_dim] → [T_frames, d_model+style_dim]
//!
//! # Frame-rate branch:
//! en [T_frames, d_model+style_dim]
//! → BiLSTM(d_model+style_dim → d_model)        # `predictor.shared`
//! → hidden [T_frames, d_model]
//!
//! # F0 branch (channel-major [d_model, T_frames]):
//! → AdainResBlk(d_model, d_model, no upsample)      # F0.0
//! → AdainResBlk(d_model, d_model/2, upsample=True)  # F0.1  (out len 2·T_frames)
//! → AdainResBlk(d_model/2, d_model/2, no upsample)  # F0.2
//! → Conv1d(d_model/2 → 1, kernel=1)                 # F0_proj  (no weight-norm)
//! → f0 [2·T_frames]
//!
//! # N branch: same shape as F0.
//! ```
//!
//! # Load-time contract (FR-EX-08)
//!
//! Every `store.tensor_shaped(...)` call binds a tensor at a name that appears
//! **verbatim** in the upstream manifest (`.module.` `nn.DataParallel` prefix
//! preserved), with the exact shape from that manifest. A missing tensor or a
//! shape mismatch surfaces as [`VokraError::InvalidArgument`] with the
//! offending tensor name; no silent zero-fill, no fabricated names, no
//! half-loaded state.
//!
//! # Text-encoder / bert seam (assumption flag)
//!
//! Upstream Kokoro feeds the ALBERT-encoded phoneme features
//! (`bert.module.*` → `bert_encoder.module.*`, 768 → 512) as the
//! `[T, 512]` input to `predictor.text_encoder`. The Vokra scaffold at
//! T13-alpha does NOT wire the bert branch yet (open at T13-beta); until it
//! does, this rewrite consumes the **text-encoder output** (also `[T, 512]`)
//! as a shape-compatible stand-in. The `predictor.text_encoder.lstms.0`
//! input dim is `d_model + style_dim = 640` — this rewrite performs the
//! style concat here at the module boundary, so the seam contract is
//! `encoded: [d_model, T]` channel-major + `style: [style_dim]`, agnostic to
//! whether the source is bert or the T13-alpha text encoder. When the bert
//! branch lands, the caller swaps the source without a signature change.
//!
//! Determinism: no RNG. Two calls with identical inputs produce bit-identical
//! outputs.
//!
//! # Backward compatibility
//!
//! The old [`ProsodyPredictor::forward`] signature
//! `(encoded, style, t, deterministic) -> (log_dur, f0, energy)` is kept as
//! a thin adapter that runs the new upstream forward and downgrades its
//! [`ProsodyOutput`] to three per-phoneme streams — the wiring agent (M2-07
//! phase 3) migrates `mod.rs` to call [`ProsodyPredictor::forward_upstream`]
//! directly and drops the adapter.

use vokra_core::{Result, VokraError};

use super::config::KokoroConfig;
use super::nn::{
    BiLstm1d, EPS, LRELU_SLOPE, adain_conditioned_residual, adaln_layernorm_1d,
    conv_transpose1d_ext, leaky_relu, length_regulate, sigmoid, weight_norm_reconstruct_1d,
};
use super::weights::TensorStore;
use crate::compute::Compute;

/// Bit-identical mirror of [`super::nn::conv1d`] (im2col + GEMM) but with the
/// GEMM inner product accumulated in `f64` before rounding back to `f32`.
///
/// # Rationale (T17-fixup #5, 2026-07-08)
///
/// After [`super::nn::adain_conditioned_residual`] (T17-fixup #4) closed the
/// `norm1` / `norm2` + `F0_proj` / `N_proj` GEMV seams under f64, the only
/// remaining f32 accumulators inside `AdainResBlk::forward` were the two
/// k=3 convs (`conv1`: `dim_in → dim_out`, `conv2`: `dim_out → dim_out`).
/// Their reduction dimension is `kernel · in_ch` (up to `3·256 = 768`) — big
/// enough that the sum-of-products drift against pure f64 truth measurably
/// contributes to the propagated hidden delta that F0_proj later amplifies.
///
/// Groups / dilation are unused by the prosody F0 / N branches (all
/// `groups = 1`, `dilation = 1`, `bias = Some(...)`), so those parameters
/// are elided from the signature to keep the call sites readable — the
/// generic form would just add complexity for no consumer.
///
/// Layout matches [`super::nn::conv1d`]:
/// * `x`: `[in_ch · in_len]` channel-major
/// * `weight`: `[out_ch · in_ch · kernel]` (PyTorch / ONNX layout)
/// * `bias`: `[out_ch]`
/// * returns `[out_ch · out_len]`, `out_len`
#[allow(clippy::too_many_arguments)] // matches nn::conv1d's operand set minus groups / dilation
fn conv1d_f64_acc(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
) -> (Vec<f32>, usize) {
    debug_assert_eq!(x.len(), in_ch * in_len);
    debug_assert_eq!(weight.len(), out_ch * in_ch * kernel);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_ch);
    }
    let out_len = (in_len + 2 * pad - kernel) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_len];
    for oc in 0..out_ch {
        let w_row = &weight[oc * in_ch * kernel..(oc + 1) * in_ch * kernel];
        let b64 = bias.map_or(0.0f64, |b| b[oc] as f64);
        let dst = &mut out[oc * out_len..(oc + 1) * out_len];
        for (ot, dst_ot) in dst.iter_mut().enumerate() {
            let mut acc64: f64 = b64;
            for ic in 0..in_ch {
                for kk in 0..kernel {
                    let it_signed = (ot * stride) as isize + kk as isize - pad as isize;
                    if it_signed < 0 || it_signed >= in_len as isize {
                        continue;
                    }
                    let it = it_signed as usize;
                    acc64 += w_row[ic * kernel + kk] as f64 * x[ic * in_len + it] as f64;
                }
            }
            *dst_ot = acc64 as f32;
        }
    }
    (out, out_len)
}

/// Number of BiLSTM stacks in `predictor.text_encoder.lstms` at positions
/// `.0`, `.2`, `.4` — pinned by the upstream manifest.
const N_DE_BILSTMS: usize = 3;

/// Number of AdaLN stacks in `predictor.text_encoder.lstms` at positions
/// `.1`, `.3`, `.5` — pinned by the upstream manifest.
const N_DE_ADALNS: usize = 3;

/// Number of AdainResBlk stages in `predictor.F0.{0,1,2}` and
/// `predictor.N.{0,1,2}` — pinned by the upstream manifest.
const N_F0N_BLOCKS: usize = 3;

/// Weight-norm split names for a Conv1d parameter.
const WEIGHT_G: &str = "weight_g";
const WEIGHT_V: &str = "weight_v";

// --- AdaLN / AdaIN affine parameters -----------------------------------------

/// Style-conditioned affine parameters — `fc: Linear(style_dim → 2·channels)`
/// yielding `(γ, β)` at inference time. Consumed both by the LayerNorm-based
/// AdaLN in `predictor.text_encoder` and the InstanceNorm-based AdaIN1d in
/// `predictor.F0` / `predictor.N` (the two variants share the same fc split;
/// only the pre-affine normalisation axis differs — see [`super::nn`]).
struct AdaLnParams {
    /// Row-major `[2·channels, style_dim]`.
    fc_w: Vec<f32>,
    /// `[2·channels]`.
    fc_b: Vec<f32>,
    /// Output channel count `C` (the fc output is `2·C`). Kept as a load-time
    /// invariant even though the forward path derives `channels` from the
    /// caller's tensor shape (T17-fixup #4 moved the forward to
    /// [`super::nn::adain_conditioned_residual`]).
    #[allow(dead_code)]
    channels: usize,
}

impl AdaLnParams {
    /// Loads `predictor.<parent>.fc.{weight,bias}` at the expected shapes.
    fn load(store: &TensorStore, prefix: &str, channels: usize, style_dim: usize) -> Result<Self> {
        let w_name = format!("{prefix}.fc.weight");
        let b_name = format!("{prefix}.fc.bias");
        let fc_w = store.tensor_shaped(&w_name, &[2 * channels, style_dim])?;
        let fc_b = store.tensor_shaped(&b_name, &[2 * channels])?;
        Ok(Self {
            fc_w,
            fc_b,
            channels,
        })
    }

    /// Projects `style` through `fc` and splits into `(γ_raw, β)` each of
    /// length `channels`. `γ_raw` is NOT yet shifted by `+1` — callers that
    /// need the residual `(1+γ)` form (StyleTTS 2 AdaIN1d) add the shift.
    ///
    /// **Retained for load-time / diagnostic use only** after T17-fixup #4:
    /// `AdainResBlk::forward` now calls [`super::nn::adain_conditioned_residual`]
    /// directly with `self.fc_w` / `self.fc_b`, fusing this Linear projection
    /// with the InstanceNorm + affine step under f64 accumulators. This method
    /// is kept as documentation of the Linear layout the loader expects and to
    /// preserve a scalar oracle for future unit tests.
    #[allow(dead_code)]
    fn project(&self, style: &[f32], style_dim: usize) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(style.len(), style_dim);
        let two_c = 2 * self.channels;
        let mut gb = vec![0.0f32; two_c];
        for (i, g) in gb.iter_mut().enumerate().take(two_c) {
            let row = &self.fc_w[i * style_dim..(i + 1) * style_dim];
            let mut acc = self.fc_b[i];
            for j in 0..style_dim {
                acc += row[j] * style[j];
            }
            *g = acc;
        }
        let beta = gb[self.channels..].to_vec();
        gb.truncate(self.channels);
        (gb, beta)
    }
}

// --- AdainResBlk (F0/N branch stage) -----------------------------------------

/// One StyleTTS 2 `AdainResBlk1d` — the residual conv block used by the F0
/// and N heads (`predictor.F0.i` / `predictor.N.i`). The forward is
/// `(residual + shortcut) / sqrt(2)` per the upstream reference.
///
/// Layout convention: input / output tensors are channel-major `[C, T]`, per
/// the [`super::nn::conv1d`] contract. WeightNorm-split convs are reconstructed
/// at load time; conv1x1 has no bias in the upstream manifest and is loaded
/// with `bias: None`.
struct AdainResBlk {
    dim_in: usize,
    dim_out: usize,
    /// [`weight_norm_reconstruct_1d`] output `[dim_out, dim_in, 3]`.
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    /// `[dim_out, dim_out, 3]`.
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    /// Present when `dim_in != dim_out` (learned shortcut). No bias — upstream
    /// manifest does not carry `conv1x1.bias`.
    conv1x1_w: Option<Vec<f32>>,
    /// Present when `upsample`. Depthwise `ConvTranspose1d(dim_in → dim_in,
    /// kernel=3, stride=2, groups=dim_in, padding=1, output_padding=1)`
    /// weight `[dim_in, 1, 3]` in PyTorch's `ConvTranspose1d` layout.
    pool_w: Option<Vec<f32>>,
    pool_b: Option<Vec<f32>>,
    /// AdaIN1d over `[dim_in, T]` channel-major (pre-conv1).
    norm1: AdaLnParams,
    /// AdaIN1d over `[dim_out, T]` channel-major (post-conv1, pre-conv2).
    norm2: AdaLnParams,
    /// Nearest-neighbor scale=2 upsample on the shortcut path when true.
    upsample: bool,
}

impl AdainResBlk {
    /// Reconstructs one `predictor.<branch>.i.*` block at the given
    /// (`dim_in`, `dim_out`, `upsample`) shape. All tensors are looked up at
    /// their verbatim upstream names — a missing or wrong-shape tensor fails
    /// loudly at this call.
    fn load(
        store: &TensorStore,
        prefix: &str,
        dim_in: usize,
        dim_out: usize,
        upsample: bool,
        style_dim: usize,
    ) -> Result<Self> {
        let learned_sc = dim_in != dim_out;

        // conv1: WeightNormedConv1d(dim_in → dim_out, k=3).
        let conv1_w = load_wn_conv1d(store, &format!("{prefix}.conv1"), dim_out, dim_in, 3)?;
        let conv1_b = store.tensor_shaped(&format!("{prefix}.conv1.bias"), &[dim_out])?;

        // conv2: WeightNormedConv1d(dim_out → dim_out, k=3).
        let conv2_w = load_wn_conv1d(store, &format!("{prefix}.conv2"), dim_out, dim_out, 3)?;
        let conv2_b = store.tensor_shaped(&format!("{prefix}.conv2.bias"), &[dim_out])?;

        // conv1x1 shortcut (only when dim_in != dim_out). NO bias in manifest.
        let conv1x1_w = if learned_sc {
            Some(load_wn_conv1d(
                store,
                &format!("{prefix}.conv1x1"),
                dim_out,
                dim_in,
                1,
            )?)
        } else {
            None
        };

        // Depthwise ConvTranspose1d pool (only when upsample). Weight layout
        // is PyTorch's `ConvTranspose1d`: `[in_channels, out_channels/groups,
        // kernel_size]` = `[dim_in, 1, 3]` with groups=dim_in (depthwise).
        let (pool_w, pool_b) = if upsample {
            let g = store.tensor_shaped(&format!("{prefix}.pool.{WEIGHT_G}"), &[dim_in, 1, 1])?;
            let v = store.tensor_shaped(&format!("{prefix}.pool.{WEIGHT_V}"), &[dim_in, 1, 3])?;
            let w = weight_norm_reconstruct_1d(&g, &v, dim_in, 1, 3);
            let b = store.tensor_shaped(&format!("{prefix}.pool.bias"), &[dim_in])?;
            (Some(w), Some(b))
        } else {
            (None, None)
        };

        let norm1 = AdaLnParams::load(store, &format!("{prefix}.norm1"), dim_in, style_dim)?;
        let norm2 = AdaLnParams::load(store, &format!("{prefix}.norm2"), dim_out, style_dim)?;

        Ok(Self {
            dim_in,
            dim_out,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            conv1x1_w,
            pool_w,
            pool_b,
            norm1,
            norm2,
            upsample,
        })
    }

    /// Runs one AdainResBlk forward. `x` is channel-major `[dim_in, t_in]`;
    /// returns `[dim_out, t_out]` with `t_out = 2·t_in` when `upsample`,
    /// otherwise `t_out = t_in`.
    fn forward(
        &self,
        compute: &Compute,
        x: &[f32],
        t_in: usize,
        style: &[f32],
        style_dim: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim_in * t_in);

        // Length regulation: residual and shortcut paths must produce the same
        // t_out (added element-wise). Both are 2·t_in when upsample, else t_in.
        let t_out = if self.upsample { 2 * t_in } else { t_in };

        // --- Shortcut path ---------------------------------------------------
        // upsample nearest scale=2 (if applicable), then conv1x1 (if learned).
        let mut sc = if self.upsample {
            interp_nearest_scale2(x, self.dim_in, t_in)
        } else {
            x.to_vec()
        };
        // Length after upsample.
        let sc_len = if self.upsample { 2 * t_in } else { t_in };
        if let Some(w) = &self.conv1x1_w {
            // T17-fixup #6 (2026-07-08): same f64-accumulator conv1d as the
            // residual `conv1` / `conv2` path (T17-fixup #5). This is the
            // Conv1d(dim_in → dim_out, k=1) shortcut projection used only
            // when `dim_in != dim_out` (F0.1 stage of the F0 branch). Its
            // reduction dimension is `in_ch` (up to 256) — same order of
            // magnitude as `conv1`'s `k · in_ch = 768`, so the same
            // f32-sum-of-products drift applies. Routing through
            // `conv1d_f64_acc` (with `bias=None`; conv1x1 has no bias per
            // the upstream manifest) preserves the composition-only
            // design constraint (D6/D7).
            let (out, out_len) = conv1d_f64_acc(
                &sc,
                self.dim_in,
                sc_len,
                w,
                self.dim_out,
                /*kernel*/ 1,
                None,
                /*stride*/ 1,
                /*pad*/ 0,
            );
            debug_assert_eq!(out_len, sc_len);
            debug_assert_eq!(out.len(), self.dim_out * sc_len);
            sc = out;
        }
        // At this point sc has shape `[dim_out, sc_len]`. sc_len == t_out.
        debug_assert_eq!(sc_len, t_out);

        // --- Residual path ---------------------------------------------------
        // norm1(x, style) — AdaIN1d over `[dim_in, t_in]` channel-major
        // (InstanceNorm across time + `(1+γ)·x + β`). T17-fixup #4 (2026-07-08):
        // route through the f64-accumulator helper that fused the Linear
        // projection and the InstanceNorm reductions, replacing the previous
        // `self.norm1.project(...) + adain(...)` f32 sequence. See
        // `docs/adr/0007-kokoro-native.md` §"T17-fixup #4" for the rationale
        // (prosody f0 honest negative closure via decoder-precedent pattern).
        let mut r = x.to_vec();
        adain_conditioned_residual(
            &mut r,
            self.dim_in,
            t_in,
            &self.norm1.fc_w,
            &self.norm1.fc_b,
            style,
            style_dim,
        );

        // LeakyReLU(0.1).
        leaky_relu(&mut r, LRELU_SLOPE);

        // pool: depthwise ConvTranspose1d(dim_in → dim_in, k=3, stride=2,
        // groups=dim_in, padding=1, output_padding=1) when upsample; else
        // Identity. After pool, shape is `[dim_in, t_out]`.
        let (r_after_pool, len_after_pool) =
            if let (Some(pw), Some(pb)) = (&self.pool_w, &self.pool_b) {
                conv_transpose1d_ext(
                    &r,
                    self.dim_in,
                    t_in,
                    pw,
                    self.dim_in,
                    /*kernel*/ 3,
                    Some(pb),
                    /*stride*/ 2,
                    /*pad*/ 1,
                    /*output_padding*/ 1,
                    /*groups*/ self.dim_in,
                )
            } else {
                (r, t_in)
            };
        debug_assert_eq!(len_after_pool, t_out);

        // conv1(k=3, pad=1): dim_in → dim_out.
        //
        // T17-fixup #5 (2026-07-08): drop through the f64-accumulator path
        // instead of the shared `nn::conv1d`. The prosody parity acid test
        // (`parity_kokoro`) shows the F0 tail delta = 2.619e-2 after
        // T17-fixup #4 promoted `norm1`/`norm2` + `F0_proj` GEMV to f64;
        // the remaining f32 seams inside `AdainResBlk` are the two k=3
        // convs. Their accumulator dimension is `k · dim_in` (up to
        // 3·256 = 768) which is large enough that the f32 dot product
        // measurably drifts against the pure-f64 truth. Promoting both
        // convs preserves the design constraint of "no new first-class
        // op" (D6/D7) — this is a composition helper local to
        // `AdainResBlk`, not a new `vokra-ops` entry.
        let (r_after_conv1, len_after_conv1) = conv1d_f64_acc(
            &r_after_pool,
            self.dim_in,
            len_after_pool,
            &self.conv1_w,
            self.dim_out,
            /*kernel*/ 3,
            Some(&self.conv1_b),
            /*stride*/ 1,
            /*pad*/ 1,
        );
        debug_assert_eq!(len_after_conv1, t_out);

        // norm2(x, style) — AdaIN1d over `[dim_out, t_out]` channel-major.
        // T17-fixup #4: same f64-accumulator promotion as norm1 above.
        let mut r = r_after_conv1;
        adain_conditioned_residual(
            &mut r,
            self.dim_out,
            t_out,
            &self.norm2.fc_w,
            &self.norm2.fc_b,
            style,
            style_dim,
        );

        // LeakyReLU(0.1).
        leaky_relu(&mut r, LRELU_SLOPE);

        // conv2(k=3, pad=1): dim_out → dim_out. Same T17-fixup #5 f64
        // promotion as conv1 above.
        let (r_final, r_final_len) = conv1d_f64_acc(
            &r,
            self.dim_out,
            t_out,
            &self.conv2_w,
            self.dim_out,
            3,
            Some(&self.conv2_b),
            1,
            1,
        );
        debug_assert_eq!(r_final_len, t_out);

        let _ = compute; // `compute` was previously required for `conv1d`; the
        // T17-fixup #5 f64 path is CPU-only (the AdainResBlk hot path is
        // small, and prosody parity is measured against the CPU forward),
        // so we drop the compute-dispatch cost here.

        // (residual + shortcut) / sqrt(2)
        let mut out = r_final;
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        for (o, s) in out.iter_mut().zip(sc.iter()) {
            *o = (*o + *s) * inv_sqrt2;
        }
        out
    }
}

// --- Prosody predictor -------------------------------------------------------

/// Output of the upstream-shape prosody predictor. Consumed by the T15
/// decoder rewrite + the M2-07 phase 3 wiring agent.
///
/// * `durations` — integer per-phoneme duration counts (`[T]`), from
///   `duration_proj`'s `round(sigmoid.sum)` scheme. Sum over `T` equals the
///   `T_frames` used by every subsequent field.
/// * `f0` — F0 contour at 2×`T_frames` resolution (`[2·T_frames]`); the
///   F0.1 block upsamples the frame-rate hidden by nearest-2, matching the
///   iSTFTNet decoder's expected upsample-friendly F0 rate.
/// * `n` — energy contour at 2×`T_frames` resolution (`[2·T_frames]`).
/// * `hidden` — frame-rate features from `predictor.shared` (`[d_model,
///   T_frames]` channel-major); consumed by the decoder input-processing
///   stage as the ASR-features residual.
#[derive(Debug, Clone)]
#[allow(dead_code)] // consumed by the wiring agent + T15 decoder rewrite
pub(crate) struct ProsodyOutput {
    pub durations: Vec<usize>,
    pub f0: Vec<f32>,
    pub n: Vec<f32>,
    pub hidden: Vec<f32>,
}

/// The Kokoro-82M prosody predictor bound to `predictor.module.*`.
pub(crate) struct ProsodyPredictor {
    d_model: usize,
    style_dim: usize,
    max_dur: usize,

    // Duration-encoder text-encoder stack (`predictor.text_encoder.lstms`).
    de_bilstms: Vec<BiLstm1d>,
    de_adalns: Vec<AdaLnParams>,

    // Main phoneme-rate LSTM (`predictor.lstm`).
    main_lstm: BiLstm1d,

    // Duration projection (`predictor.duration_proj.linear_layer`).
    dur_proj_w: Vec<f32>,
    dur_proj_b: Vec<f32>,

    // Frame-rate BiLSTM (`predictor.shared`).
    shared_lstm: BiLstm1d,

    // F0 / N branches.
    f0_blocks: Vec<AdainResBlk>,
    f0_proj_w: Vec<f32>, // [1, d_model/2, 1] = d_model/2 floats
    f0_proj_b: Vec<f32>, // [1]

    n_blocks: Vec<AdainResBlk>,
    n_proj_w: Vec<f32>,
    n_proj_b: Vec<f32>,
}

impl ProsodyPredictor {
    /// Loads the prosody predictor from a Kokoro voice GGUF.
    ///
    /// All 122 upstream `predictor.module.*` tensors are bound at the exact
    /// name from `data/upstream_tensors_v1_0.tsv`; any missing / wrong-shape
    /// tensor is a loud [`VokraError::InvalidArgument`] (FR-EX-08 red line
    /// R4).
    ///
    /// Requires `config.hidden_dim` to be even (BiLSTM hidden = `d_model/2`)
    /// and `> 1` (F0/N branch shrinks channels by 2 at F0.1). `max_dur` is
    /// **derived from** the `duration_proj.linear_layer.weight` shape's first
    /// dim so a differently-sized checkpoint (e.g. `max_dur=100`) still loads
    /// as long as the two `duration_proj` tensors are consistent.
    #[allow(dead_code)] // called from `KokoroTts::from_gguf_with_policy`
    pub(crate) fn load(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        Self::new(store, config)
    }

    /// Explicit constructor — the same as [`Self::load`], with an alias so
    /// tests can call the "new" name matching the T13-alpha text encoder.
    pub(crate) fn new(store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        let d_model = config.hidden_dim;
        let style_dim = config.style_dim;
        if d_model == 0 || d_model % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: config.hidden_dim ({d_model}) must be even and > 0"
            )));
        }
        if d_model < 2 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: config.hidden_dim ({d_model}) must be ≥ 2 (F0/N block \
                 shrinks channels to d_model/2)"
            )));
        }
        if style_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "kokoro prosody: config.style_dim is 0".to_owned(),
            ));
        }
        let lstm_hidden = d_model / 2;
        let d_te_in = d_model + style_dim;

        // --- Duration-encoder stack -----------------------------------------
        let mut de_bilstms: Vec<BiLstm1d> = Vec::with_capacity(N_DE_BILSTMS);
        let mut de_adalns: Vec<AdaLnParams> = Vec::with_capacity(N_DE_ADALNS);
        // BiLSTMs at `.0`, `.2`, `.4`; AdaLN at `.1`, `.3`, `.5`.
        for i in 0..N_DE_BILSTMS {
            let bilstm_idx = 2 * i;
            let adaln_idx = 2 * i + 1;
            let prefix_lstm = format!("predictor.module.text_encoder.lstms.{bilstm_idx}");
            de_bilstms.push(load_bilstm(store, &prefix_lstm, d_te_in, lstm_hidden)?);
            let prefix_adaln = format!("predictor.module.text_encoder.lstms.{adaln_idx}");
            de_adalns.push(AdaLnParams::load(store, &prefix_adaln, d_model, style_dim)?);
        }

        // --- Main phoneme-rate LSTM -----------------------------------------
        let main_lstm = load_bilstm(store, "predictor.module.lstm", d_te_in, lstm_hidden)?;

        // --- Duration projection --------------------------------------------
        // Shape-derive max_dur from the weight's first dim so a differently
        // sized checkpoint still loads. The tensor name is the upstream
        // `linear_layer` sub-object.
        let dur_w_name = "predictor.module.duration_proj.linear_layer.weight";
        let dur_b_name = "predictor.module.duration_proj.linear_layer.bias";
        let dur_shape = store.shape(dur_w_name)?;
        if dur_shape.len() != 2 || dur_shape[1] != d_model {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: `{dur_w_name}` shape {dur_shape:?}, \
                 expected [max_dur, {d_model}]"
            )));
        }
        let max_dur = dur_shape[0];
        if max_dur == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: `{dur_w_name}` first dim (max_dur) is 0"
            )));
        }
        let dur_proj_w = store.tensor_shaped(dur_w_name, &[max_dur, d_model])?;
        let dur_proj_b = store.tensor_shaped(dur_b_name, &[max_dur])?;

        // --- Frame-rate BiLSTM ----------------------------------------------
        let shared_lstm = load_bilstm(store, "predictor.module.shared", d_te_in, lstm_hidden)?;

        // --- F0 branch ------------------------------------------------------
        // The three blocks (dim_in, dim_out, upsample):
        //   F0.0: (d_model,          d_model,          false)
        //   F0.1: (d_model,          d_model / 2,      true)
        //   F0.2: (d_model / 2,      d_model / 2,      false)
        let half = lstm_hidden; // d_model / 2
        let f0_blocks = load_branch_blocks(store, "predictor.module.F0", d_model, half, style_dim)?;
        // F0_proj: Conv1d(half → 1, k=1) — NOT weight-normed. Weight shape
        // `[1, half, 1]` = `half` floats.
        let f0_proj_w = store.tensor_shaped("predictor.module.F0_proj.weight", &[1, half, 1])?;
        let f0_proj_b = store.tensor_shaped("predictor.module.F0_proj.bias", &[1])?;

        // --- N branch (mirror of F0) ----------------------------------------
        let n_blocks = load_branch_blocks(store, "predictor.module.N", d_model, half, style_dim)?;
        let n_proj_w = store.tensor_shaped("predictor.module.N_proj.weight", &[1, half, 1])?;
        let n_proj_b = store.tensor_shaped("predictor.module.N_proj.bias", &[1])?;

        Ok(Self {
            d_model,
            style_dim,
            max_dur,
            de_bilstms,
            de_adalns,
            main_lstm,
            dur_proj_w,
            dur_proj_b,
            shared_lstm,
            f0_blocks,
            f0_proj_w,
            f0_proj_b,
            n_blocks,
            n_proj_w,
            n_proj_b,
        })
    }

    /// The `d_model` (`= config.hidden_dim`) resolved at load time.
    #[allow(dead_code)] // consumed by tests + wiring agent
    pub(crate) fn d_model(&self) -> usize {
        self.d_model
    }

    /// The `max_dur` shape-derived from `duration_proj.linear_layer.weight`.
    #[cfg(test)]
    pub(crate) fn max_dur(&self) -> usize {
        self.max_dur
    }

    /// The number of F0/N branch blocks (statically 3 in Kokoro-82M).
    #[cfg(test)]
    pub(crate) fn f0_block_count(&self) -> usize {
        self.f0_blocks.len()
    }

    /// The runs the full upstream prosody predictor forward
    /// `encoded [d_model, T] channel-major + style [style_dim] → ProsodyOutput`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch (FR-EX-08).
    #[allow(dead_code)] // consumed by wiring agent
    pub(crate) fn forward_upstream(
        &self,
        encoded: &[f32],
        style: &[f32],
        t: usize,
    ) -> Result<ProsodyOutput> {
        let d = self.d_model;
        let sd = self.style_dim;
        if encoded.len() != d * t {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: encoded len {} != d_model ({}) · t ({})",
                encoded.len(),
                d,
                t
            )));
        }
        if style.len() != sd {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro prosody: style len {} != style_dim ({})",
                style.len(),
                sd
            )));
        }
        if t == 0 {
            return Ok(ProsodyOutput {
                durations: Vec::new(),
                f0: Vec::new(),
                n: Vec::new(),
                hidden: Vec::new(),
            });
        }

        let compute = Compute::cpu();
        let d_te_in = d + sd;

        // Convert encoded [d, t] channel-major → [t, d] row-major (BiLSTM
        // consumes row-major).
        let mut x_row = vec![0.0f32; t * d];
        for c in 0..d {
            for ti in 0..t {
                x_row[ti * d + c] = encoded[c * t + ti];
            }
        }

        // Concat style at the tail: [t, d + sd] row-major.
        let mut x_cat = concat_style_row(&x_row, style, t, d, sd);

        // --- Duration-encoder stack: 3 × (BiLSTM → AdaLN → concat style) ---
        for i in 0..N_DE_BILSTMS {
            // BiLSTM(d_te_in → d_model): output [t, d].
            let lstm_out = self.de_bilstms[i].forward(&x_cat, t);
            debug_assert_eq!(lstm_out.len(), t * d);
            // AdaLN(x, style): LayerNorm across channels + (1+γ)·x + β.
            let mut norm_out = vec![0.0f32; t * d];
            adaln_layernorm_1d(
                &lstm_out,
                t,
                d,
                &self.de_adalns[i].fc_w,
                &self.de_adalns[i].fc_b,
                style,
                sd,
                &mut norm_out,
            );
            // Concat style: [t, d + sd].
            x_cat = concat_style_row(&norm_out, style, t, d, sd);
        }
        // At this point `x_cat` is `d` in the upstream nomenclature — the
        // duration-encoder output feeding both the main LSTM and the alignment
        // matrix product.
        let d_features = x_cat;

        // --- Main LSTM ------------------------------------------------------
        let main_out = self.main_lstm.forward(&d_features, t);
        debug_assert_eq!(main_out.len(), t * d);

        // --- Duration projection --------------------------------------------
        // Per-phoneme sigmoid.sum.round.clamp. GEMM main_out @ dur_proj_w.T:
        // we use gemv per phoneme to avoid a transposed second operand.
        let mut durations: Vec<usize> = Vec::with_capacity(t);
        let mut dur_row = vec![0.0f32; self.max_dur];
        for ti in 0..t {
            let x_ti = &main_out[ti * d..(ti + 1) * d];
            compute.gemv_f32(
                self.max_dur,
                d,
                &self.dur_proj_w,
                x_ti,
                Some(&self.dur_proj_b),
                &mut dur_row,
            )?;
            let sum: f32 = dur_row.iter().map(|&v| sigmoid(v)).sum();
            // round + clamp to [1, 1024] (upper bound guards against a
            // pathological +inf saturation blowing the frame buffer).
            let d_int = if sum.is_finite() {
                (sum.round() as i64).clamp(1, 1024) as usize
            } else {
                1
            };
            durations.push(d_int);
        }
        let t_frames: usize = durations.iter().sum();

        if t_frames == 0 {
            return Ok(ProsodyOutput {
                durations,
                f0: Vec::new(),
                n: Vec::new(),
                hidden: Vec::new(),
            });
        }

        // --- Length regulation: expand d_features [t, d_te_in] → [T_frames, d_te_in]
        // Go via channel-major so the shared length_regulate helper (which
        // consumes `[hidden, t_in]` channel-major) works.
        let mut d_ch = vec![0.0f32; d_te_in * t];
        for c in 0..d_te_in {
            for ti in 0..t {
                d_ch[c * t + ti] = d_features[ti * d_te_in + c];
            }
        }
        let (en_ch, t_frames_actual) = length_regulate(&d_ch, d_te_in, t, &durations);
        debug_assert_eq!(t_frames_actual, t_frames);

        // Convert en to row-major [T_frames, d_te_in] for the shared BiLSTM.
        let mut en_row = vec![0.0f32; t_frames * d_te_in];
        for c in 0..d_te_in {
            for ti in 0..t_frames {
                en_row[ti * d_te_in + c] = en_ch[c * t_frames + ti];
            }
        }

        // --- Frame-rate shared BiLSTM: [T_frames, d_te_in] → [T_frames, d]
        let shared_out = self.shared_lstm.forward(&en_row, t_frames);
        debug_assert_eq!(shared_out.len(), t_frames * d);

        // Convert to channel-major [d, T_frames] for F0/N convs.
        let mut hidden_ch = vec![0.0f32; d * t_frames];
        for c in 0..d {
            for ti in 0..t_frames {
                hidden_ch[c * t_frames + ti] = shared_out[ti * d + c];
            }
        }

        // --- F0 branch ------------------------------------------------------
        let f0 = self.run_branch(
            &compute,
            &hidden_ch,
            t_frames,
            style,
            sd,
            &self.f0_blocks,
            &self.f0_proj_w,
            &self.f0_proj_b,
        );

        // --- N branch -------------------------------------------------------
        let n = self.run_branch(
            &compute,
            &hidden_ch,
            t_frames,
            style,
            sd,
            &self.n_blocks,
            &self.n_proj_w,
            &self.n_proj_b,
        );

        Ok(ProsodyOutput {
            durations,
            f0,
            n,
            hidden: hidden_ch,
        })
    }

    /// Runs one F0-or-N branch: 3× AdainResBlk → 1×1 Conv1d → squeeze channel.
    #[allow(clippy::too_many_arguments)]
    fn run_branch(
        &self,
        compute: &Compute,
        hidden_ch: &[f32],
        t_frames: usize,
        style: &[f32],
        style_dim: usize,
        blocks: &[AdainResBlk],
        proj_w: &[f32],
        proj_b: &[f32],
    ) -> Vec<f32> {
        let mut cur = hidden_ch.to_vec();
        let mut cur_ch = self.d_model;
        let mut cur_len = t_frames;
        for blk in blocks {
            debug_assert_eq!(blk.dim_in, cur_ch);
            let out = blk.forward(compute, &cur, cur_len, style, style_dim);
            cur_len = if blk.upsample { 2 * cur_len } else { cur_len };
            cur_ch = blk.dim_out;
            cur = out;
        }
        // Final projection Conv1d(cur_ch → 1, k=1) → [1, cur_len] → squeeze.
        //
        // T17-fixup #4 (2026-07-08): inline an f64-accumulator GEMV for this
        // specific 1×k×in_ch case instead of routing through the generic
        // `conv1d` → `compute.gemm_f32` (which uses an f32 dot product per
        // output element). Rationale: `F0_proj` / `N_proj` is a linear
        // combination of all `cur_ch` = `d_model/2` = 256 hidden channels per
        // time step; the f32 sum-of-products at k=256 accumulates ULP drift on
        // the order of `sqrt(k) · ULP · ||x||` which stacks on top of the ~9×
        // amplification of the upstream ~3e-3 hidden delta. See
        // `docs/adr/0007-kokoro-native.md` §"T17-fixup #4" for the full
        // analysis + decoder-precedent link (`adain_conditioned` in nn.rs).
        debug_assert_eq!(proj_w.len(), cur_ch);
        debug_assert_eq!(proj_b.len(), 1);
        let bias0 = proj_b[0];
        let mut proj_out = vec![0.0f32; cur_len];
        for ti in 0..cur_len {
            let mut acc64: f64 = bias0 as f64;
            for c in 0..cur_ch {
                acc64 += proj_w[c] as f64 * cur[c * cur_len + ti] as f64;
            }
            proj_out[ti] = acc64 as f32;
        }
        let proj_len = cur_len;
        debug_assert_eq!(proj_len, cur_len);
        // proj_out is [1, cur_len]; already flat.
        proj_out
    }

    /// Backward-compatible adapter matching the pre-T14 signature.
    ///
    /// Runs [`Self::forward_upstream`] and downgrades the [`ProsodyOutput`]
    /// to `(log_dur[T], f0[T], energy[T])`:
    /// * `log_dur[ti] = ln(max(1, durations[ti]))` — the exp/round the caller
    ///   applies (per the piper-convention length regulator) inverts back to
    ///   `durations[ti]`, matching the pre-rewrite contract.
    /// * `f0[ti]` — the mean of the 2×`T_frames` F0 contour over the frames
    ///   assigned to phoneme `ti`. This is a lossy downgrade; the wiring
    ///   agent (M2-07 phase 3) drops it and consumes `ProsodyOutput.f0`
    ///   directly.
    /// * `energy[ti]` — the analogous mean of the N contour.
    ///
    /// The old `deterministic` flag is preserved for API compatibility; the
    /// stochastic branch is not part of the upstream forward and returns
    /// [`VokraError::NotImplemented`] loudly.
    #[allow(dead_code)] // called by the existing mod.rs wire-up until phase 3
    pub(crate) fn forward(
        &self,
        encoded: &[f32],
        style: &[f32],
        t: usize,
        deterministic: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if !deterministic {
            return Err(VokraError::NotImplemented(
                "kokoro stochastic prosody path not part of the upstream reference",
            ));
        }
        let out = self.forward_upstream(encoded, style, t)?;

        // Downgrade durations → log_dur.
        let log_dur: Vec<f32> = out
            .durations
            .iter()
            .map(|&d| (d.max(1) as f32).ln())
            .collect();

        // Downgrade F0 [2·T_frames] and N [2·T_frames] to per-phoneme means.
        // Each phoneme occupies `durations[ti]` frame-rate positions, i.e.
        // `2·durations[ti]` positions in the 2× F0/N stream.
        let mut f0 = vec![0.0f32; t];
        let mut energy = vec![0.0f32; t];
        let mut off = 0usize;
        for ti in 0..t {
            let width = 2 * out.durations[ti];
            if width > 0 && off + width <= out.f0.len() {
                let f0_slice = &out.f0[off..off + width];
                let n_slice = &out.n[off..off + width];
                let inv = 1.0f32 / width as f32;
                f0[ti] = f0_slice.iter().sum::<f32>() * inv;
                energy[ti] = n_slice.iter().sum::<f32>() * inv;
            }
            off += width;
        }
        Ok((log_dur, f0, energy))
    }
}

// --- Free helpers ------------------------------------------------------------

/// Loads a `predictor.module.<...>.lstm` (or `text_encoder.lstms.i`) as a
/// [`BiLstm1d`] with the given (input_dim, hidden_dim).
fn load_bilstm(
    store: &TensorStore,
    prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
) -> Result<BiLstm1d> {
    let four_h = 4 * hidden_dim;
    let w_ih_fwd = store.tensor_shaped(&format!("{prefix}.weight_ih_l0"), &[four_h, input_dim])?;
    let w_hh_fwd = store.tensor_shaped(&format!("{prefix}.weight_hh_l0"), &[four_h, hidden_dim])?;
    let b_ih_fwd = store.tensor_shaped(&format!("{prefix}.bias_ih_l0"), &[four_h])?;
    let b_hh_fwd = store.tensor_shaped(&format!("{prefix}.bias_hh_l0"), &[four_h])?;
    let w_ih_rev = store.tensor_shaped(
        &format!("{prefix}.weight_ih_l0_reverse"),
        &[four_h, input_dim],
    )?;
    let w_hh_rev = store.tensor_shaped(
        &format!("{prefix}.weight_hh_l0_reverse"),
        &[four_h, hidden_dim],
    )?;
    let b_ih_rev = store.tensor_shaped(&format!("{prefix}.bias_ih_l0_reverse"), &[four_h])?;
    let b_hh_rev = store.tensor_shaped(&format!("{prefix}.bias_hh_l0_reverse"), &[four_h])?;
    BiLstm1d::new(
        input_dim, hidden_dim, w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd, w_ih_rev, w_hh_rev,
        b_ih_rev, b_hh_rev,
    )
}

/// Loads a weight-normed Conv1d weight (`<prefix>.{weight_g, weight_v}`) and
/// reconstructs the full weight `[out_ch, in_ch, kernel]`.
fn load_wn_conv1d(
    store: &TensorStore,
    prefix: &str,
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    let g = store.tensor_shaped(&format!("{prefix}.{WEIGHT_G}"), &[out_ch, 1, 1])?;
    let v = store.tensor_shaped(&format!("{prefix}.{WEIGHT_V}"), &[out_ch, in_ch, kernel])?;
    Ok(weight_norm_reconstruct_1d(&g, &v, out_ch, in_ch, kernel))
}

/// Loads the three AdainResBlk stages of one F0/N branch — the shape schedule
/// is `(d_model, d_model, no upsample)` → `(d_model, d_model/2, upsample)`
/// → `(d_model/2, d_model/2, no upsample)`, pinned by the upstream manifest.
fn load_branch_blocks(
    store: &TensorStore,
    prefix: &str,
    d_model: usize,
    half: usize,
    style_dim: usize,
) -> Result<Vec<AdainResBlk>> {
    let mut out = Vec::with_capacity(N_F0N_BLOCKS);
    // Block 0: (d_model, d_model, false).
    out.push(AdainResBlk::load(
        store,
        &format!("{prefix}.0"),
        d_model,
        d_model,
        false,
        style_dim,
    )?);
    // Block 1: (d_model, half, true).
    out.push(AdainResBlk::load(
        store,
        &format!("{prefix}.1"),
        d_model,
        half,
        true,
        style_dim,
    )?);
    // Block 2: (half, half, false).
    out.push(AdainResBlk::load(
        store,
        &format!("{prefix}.2"),
        half,
        half,
        false,
        style_dim,
    )?);
    Ok(out)
}

/// Concatenates a `style` vector broadcast across time into `x`'s tail:
/// `x [t, d] row-major → out [t, d + style_dim] row-major`. Column `[d..]`
/// of each row is a copy of `style`.
fn concat_style_row(x: &[f32], style: &[f32], t: usize, d: usize, style_dim: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), t * d);
    debug_assert_eq!(style.len(), style_dim);
    let stride = d + style_dim;
    let mut out = vec![0.0f32; t * stride];
    for ti in 0..t {
        out[ti * stride..ti * stride + d].copy_from_slice(&x[ti * d..(ti + 1) * d]);
        out[ti * stride + d..ti * stride + stride].copy_from_slice(style);
    }
    out
}

/// Nearest-neighbour scale-2 upsample of a channel-major `[c, t]` tensor.
/// Each column is duplicated once: `out[c, 2·i] = out[c, 2·i + 1] = x[c, i]`.
/// Mirrors PyTorch's `F.interpolate(scale_factor=2, mode='nearest')` on 1-D
/// tensors — the exact op the StyleTTS 2 AdainResBlk1d shortcut uses on
/// upsample stages.
fn interp_nearest_scale2(x: &[f32], channels: usize, t: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), channels * t);
    let mut out = vec![0.0f32; channels * 2 * t];
    for c in 0..channels {
        for i in 0..t {
            let v = x[c * t + i];
            out[c * 2 * t + 2 * i] = v;
            out[c * 2 * t + 2 * i + 1] = v;
        }
    }
    // Suppress unused-EPS lint when this fn happens to be the only import site
    // in a future test-only build (the const is exported by nn.rs for other
    // helpers).
    let _ = EPS;
    out
}

#[cfg(test)]
mod tests {
    use super::super::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    // --- Synthetic GGUF fixture ----------------------------------------------
    // The fixture uses tiny shapes derived by config:
    //   d_model = 4 (BiLSTM hidden = 2), style_dim = 2, max_dur = 3.
    // Weights are all-zero (loader smoke) or deterministic ramps (forward
    // shape / determinism / permutation tests).

    fn zeros_bytes(n: usize) -> Vec<u8> {
        vec![0u8; n * 4]
    }

    fn ramp_bytes(n: usize, seed: f32, step: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed + i as f32 * step).to_le_bytes())
            .collect()
    }

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// Build a synthetic GGUF carrying every `predictor.module.*` tensor the
    /// [`ProsodyPredictor`] loader binds, at the shapes derived from
    /// `d_model = 4`, `style_dim = 2`, `max_dur = 3`. All non-config tensors
    /// are either zero (loader smoke) or ramp (forward shape / determinism).
    fn build_synthetic_gguf(ramp: bool) -> Vec<u8> {
        let d_model = 4usize;
        let style_dim = 2usize;
        let max_dur = 3usize;
        let lstm_h = d_model / 2; // 2
        let four_h = 4 * lstm_h; // 8
        let d_te_in = d_model + style_dim; // 6
        let half = d_model / 2; // 2

        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 1);
        b.add_u32(KEY_HIDDEN_DIM, d_model as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 8);
        b.add_u32(KEY_ISTFT_HOP, 2);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 8);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a", "b", "c"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));

        // Helper to add F32 tensor.
        let add_zero = |b: &mut GgufBuilder, name: &str, dims: Vec<u64>, size: usize| {
            let bytes = if ramp {
                // Bias, gamma, beta etc. — use a small ramp so the forward
                // produces distinguishable outputs.
                ramp_bytes(size, 0.01, 0.01)
            } else {
                zeros_bytes(size)
            };
            b.add_tensor(name, GgmlType::F32, dims, bytes).expect("add");
        };

        // Text encoder stack: 3× BiLSTM(d_te_in → d_model) + 3× AdaLN(d_model, style_dim).
        for i in 0..N_DE_BILSTMS {
            let bi = 2 * i;
            let ai = 2 * i + 1;
            let pfx = format!("predictor.module.text_encoder.lstms.{bi}");
            for suffix in ["", "_reverse"] {
                add_zero(
                    &mut b,
                    &format!("{pfx}.weight_ih_l0{suffix}"),
                    vec![four_h as u64, d_te_in as u64],
                    four_h * d_te_in,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.weight_hh_l0{suffix}"),
                    vec![four_h as u64, lstm_h as u64],
                    four_h * lstm_h,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.bias_ih_l0{suffix}"),
                    vec![four_h as u64],
                    four_h,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.bias_hh_l0{suffix}"),
                    vec![four_h as u64],
                    four_h,
                );
            }
            let apfx = format!("predictor.module.text_encoder.lstms.{ai}");
            add_zero(
                &mut b,
                &format!("{apfx}.fc.weight"),
                vec![2 * d_model as u64, style_dim as u64],
                2 * d_model * style_dim,
            );
            add_zero(
                &mut b,
                &format!("{apfx}.fc.bias"),
                vec![2 * d_model as u64],
                2 * d_model,
            );
        }

        // Main LSTM.
        for suffix in ["", "_reverse"] {
            add_zero(
                &mut b,
                &format!("predictor.module.lstm.weight_ih_l0{suffix}"),
                vec![four_h as u64, d_te_in as u64],
                four_h * d_te_in,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.lstm.weight_hh_l0{suffix}"),
                vec![four_h as u64, lstm_h as u64],
                four_h * lstm_h,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.lstm.bias_ih_l0{suffix}"),
                vec![four_h as u64],
                four_h,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.lstm.bias_hh_l0{suffix}"),
                vec![four_h as u64],
                four_h,
            );
        }

        // duration_proj.
        add_zero(
            &mut b,
            "predictor.module.duration_proj.linear_layer.weight",
            vec![max_dur as u64, d_model as u64],
            max_dur * d_model,
        );
        add_zero(
            &mut b,
            "predictor.module.duration_proj.linear_layer.bias",
            vec![max_dur as u64],
            max_dur,
        );

        // Shared LSTM.
        for suffix in ["", "_reverse"] {
            add_zero(
                &mut b,
                &format!("predictor.module.shared.weight_ih_l0{suffix}"),
                vec![four_h as u64, d_te_in as u64],
                four_h * d_te_in,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.shared.weight_hh_l0{suffix}"),
                vec![four_h as u64, lstm_h as u64],
                four_h * lstm_h,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.shared.bias_ih_l0{suffix}"),
                vec![four_h as u64],
                four_h,
            );
            add_zero(
                &mut b,
                &format!("predictor.module.shared.bias_hh_l0{suffix}"),
                vec![four_h as u64],
                four_h,
            );
        }

        // F0 and N branches (shapes: (d, d, false), (d, half, true), (half, half, false)).
        for branch in ["F0", "N"] {
            // Block 0: (d_model, d_model, no upsample, no learned_sc).
            {
                let pfx = format!("predictor.module.{branch}.0");
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_g"),
                    vec![d_model as u64, 1, 1],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_v"),
                    vec![d_model as u64, d_model as u64, 3],
                    d_model * d_model * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.bias"),
                    vec![d_model as u64],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_g"),
                    vec![d_model as u64, 1, 1],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_v"),
                    vec![d_model as u64, d_model as u64, 3],
                    d_model * d_model * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.bias"),
                    vec![d_model as u64],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.weight"),
                    vec![2 * d_model as u64, style_dim as u64],
                    2 * d_model * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.bias"),
                    vec![2 * d_model as u64],
                    2 * d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.weight"),
                    vec![2 * d_model as u64, style_dim as u64],
                    2 * d_model * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.bias"),
                    vec![2 * d_model as u64],
                    2 * d_model,
                );
            }
            // Block 1: (d_model, half, upsample=True, learned_sc=True).
            {
                let pfx = format!("predictor.module.{branch}.1");
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_g"),
                    vec![half as u64, 1, 1],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_v"),
                    vec![half as u64, d_model as u64, 3],
                    half * d_model * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.bias"),
                    vec![half as u64],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1x1.weight_g"),
                    vec![half as u64, 1, 1],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1x1.weight_v"),
                    vec![half as u64, d_model as u64, 1],
                    half * d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_g"),
                    vec![half as u64, 1, 1],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_v"),
                    vec![half as u64, half as u64, 3],
                    half * half * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.bias"),
                    vec![half as u64],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.pool.weight_g"),
                    vec![d_model as u64, 1, 1],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.pool.weight_v"),
                    vec![d_model as u64, 1, 3],
                    d_model * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.pool.bias"),
                    vec![d_model as u64],
                    d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.weight"),
                    vec![2 * d_model as u64, style_dim as u64],
                    2 * d_model * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.bias"),
                    vec![2 * d_model as u64],
                    2 * d_model,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.weight"),
                    vec![2 * half as u64, style_dim as u64],
                    2 * half * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.bias"),
                    vec![2 * half as u64],
                    2 * half,
                );
            }
            // Block 2: (half, half, no upsample, no learned_sc).
            {
                let pfx = format!("predictor.module.{branch}.2");
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_g"),
                    vec![half as u64, 1, 1],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.weight_v"),
                    vec![half as u64, half as u64, 3],
                    half * half * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv1.bias"),
                    vec![half as u64],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_g"),
                    vec![half as u64, 1, 1],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.weight_v"),
                    vec![half as u64, half as u64, 3],
                    half * half * 3,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.conv2.bias"),
                    vec![half as u64],
                    half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.weight"),
                    vec![2 * half as u64, style_dim as u64],
                    2 * half * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm1.fc.bias"),
                    vec![2 * half as u64],
                    2 * half,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.weight"),
                    vec![2 * half as u64, style_dim as u64],
                    2 * half * style_dim,
                );
                add_zero(
                    &mut b,
                    &format!("{pfx}.norm2.fc.bias"),
                    vec![2 * half as u64],
                    2 * half,
                );
            }
            // Branch proj.
            let proj = format!("predictor.module.{branch}_proj");
            add_zero(
                &mut b,
                &format!("{proj}.weight"),
                vec![1, half as u64, 1],
                half,
            );
            add_zero(&mut b, &format!("{proj}.bias"), vec![1], 1);
        }

        b.to_bytes().expect("serialize")
    }

    fn build_predictor(ramp: bool) -> ProsodyPredictor {
        let bytes = build_synthetic_gguf(ramp);
        let file = GgufFile::parse(bytes).expect("parse synthetic Kokoro GGUF");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        ProsodyPredictor::new(&store, &config).expect("valid synthetic tensors")
    }

    /// The load path binds every `predictor.module.*` tensor at its shape;
    /// a synthetic GGUF that carries them all builds successfully and the
    /// resolved dims match config.
    #[test]
    fn loads_all_tensors_from_synthetic_gguf() {
        let p = build_predictor(false);
        assert_eq!(p.d_model(), 4);
        assert_eq!(p.max_dur(), 3);
        assert_eq!(p.f0_block_count(), N_F0N_BLOCKS);
        assert_eq!(p.n_blocks.len(), N_F0N_BLOCKS);
    }

    /// The upstream forward on all-zero weights must produce well-shaped
    /// outputs with finite entries (LSTM + conv + AdaIN + LeakyReLU zeros are
    /// bounded).
    #[test]
    fn forward_upstream_returns_expected_shapes_on_zeros() {
        let p = build_predictor(false);
        let t = 3;
        let d = p.d_model();
        let sd = p.style_dim;
        let encoded = vec![0.0f32; d * t];
        let style = vec![0.0f32; sd];
        let out = p
            .forward_upstream(&encoded, &style, t)
            .expect("forward should succeed");
        // Duration count matches phoneme count.
        assert_eq!(out.durations.len(), t);
        let t_frames: usize = out.durations.iter().sum();
        assert_eq!(out.hidden.len(), d * t_frames);
        assert_eq!(out.f0.len(), 2 * t_frames);
        assert_eq!(out.n.len(), 2 * t_frames);
        for v in out.f0.iter().chain(out.n.iter()).chain(out.hidden.iter()) {
            assert!(v.is_finite(), "prosody output must be finite: {v}");
        }
    }

    /// The forward is deterministic — same input → bit-identical output. Uses
    /// ramp weights so the output is non-trivial (not identically zero).
    #[test]
    fn forward_upstream_is_deterministic_across_two_calls() {
        let p = build_predictor(true);
        let t = 3;
        let d = p.d_model();
        let sd = p.style_dim;
        let encoded: Vec<f32> = (0..d * t).map(|i| 0.01 + i as f32 * 0.01).collect();
        let style: Vec<f32> = (0..sd).map(|i| 0.05 + i as f32 * 0.05).collect();
        let a = p.forward_upstream(&encoded, &style, t).expect("first call");
        let b = p
            .forward_upstream(&encoded, &style, t)
            .expect("second call");
        assert_eq!(a.durations, b.durations, "durations differ across calls");
        assert_eq!(a.f0, b.f0, "f0 differs across calls");
        assert_eq!(a.n, b.n, "n differs across calls");
        assert_eq!(a.hidden, b.hidden, "hidden differs across calls");
    }

    /// A permutation of the style vector must reach the output — the style is
    /// consumed in three places (AdaLN in the DE stack, AdaIN in F0/N,
    /// concat before each BiLSTM). If any of these silently drops style, the
    /// permuted call would produce identical output.
    #[test]
    fn style_permutation_changes_outputs() {
        let p = build_predictor(true);
        let t = 3;
        let d = p.d_model();
        let sd = p.style_dim;
        assert!(sd >= 2, "test needs style_dim >= 2 for a permutation");
        let encoded: Vec<f32> = (0..d * t).map(|i| 0.02 + i as f32 * 0.01).collect();
        let a = p
            .forward_upstream(&encoded, &[0.1, 0.2], t)
            .expect("first ok");
        let b = p
            .forward_upstream(&encoded, &[0.2, 0.1], t)
            .expect("permuted ok");
        assert_ne!(
            (a.f0, a.n, a.hidden),
            (b.f0, b.n, b.hidden),
            "permuted style must reach the F0/N/hidden outputs"
        );
    }

    /// FR-EX-08: an encoded vector whose length is not `d_model · t` is a
    /// loud [`VokraError::InvalidArgument`], never silently truncated.
    #[test]
    fn rejects_encoded_shape_mismatch() {
        let p = build_predictor(false);
        let d = p.d_model();
        let sd = p.style_dim;
        let t = 3;
        let encoded = vec![0.0f32; d * t - 1];
        let style = vec![0.0f32; sd];
        match p.forward_upstream(&encoded, &style, t) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("encoded"), "error must name `encoded`: {msg}");
            }
            other => panic!("expected InvalidArgument for encoded, got {other:?}"),
        }
    }

    /// FR-EX-08: a style vector whose length is not `style_dim` is a loud
    /// [`VokraError::InvalidArgument`], never zero-padded.
    #[test]
    fn rejects_style_shape_mismatch() {
        let p = build_predictor(false);
        let d = p.d_model();
        let sd = p.style_dim;
        let t = 3;
        let encoded = vec![0.0f32; d * t];
        let style = vec![0.0f32; sd + 1];
        match p.forward_upstream(&encoded, &style, t) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("style"), "error must name `style`: {msg}");
            }
            other => panic!("expected InvalidArgument for style, got {other:?}"),
        }
    }

    /// The backward-compat [`ProsodyPredictor::forward`] adapter (kept for
    /// the pre-phase-3 mod.rs wire-up) must preserve the pre-rewrite
    /// signature and return three `[T]`-length streams.
    #[test]
    fn forward_adapter_returns_per_phoneme_streams() {
        let p = build_predictor(true);
        let t = 3;
        let d = p.d_model();
        let sd = p.style_dim;
        let encoded: Vec<f32> = (0..d * t).map(|i| 0.01 + i as f32 * 0.01).collect();
        let style: Vec<f32> = (0..sd).map(|i| 0.05 + i as f32 * 0.05).collect();
        let (log_dur, f0, en) = p.forward(&encoded, &style, t, true).expect("adapter ok");
        assert_eq!(log_dur.len(), t);
        assert_eq!(f0.len(), t);
        assert_eq!(en.len(), t);
    }

    /// FR-EX-08: the stochastic path is not silently degraded to a
    /// deterministic zero — it is a loud [`VokraError::NotImplemented`].
    #[test]
    fn adapter_rejects_non_deterministic_path() {
        let p = build_predictor(false);
        let d = p.d_model();
        let sd = p.style_dim;
        let t = 2;
        let encoded = vec![0.0f32; d * t];
        let style = vec![0.0f32; sd];
        assert!(matches!(
            p.forward(&encoded, &style, t, false),
            Err(VokraError::NotImplemented(_))
        ));
    }

    /// FR-EX-08: a config-only GGUF (no predictor tensors) fails at the very
    /// first `tensor_shaped` call — the loader must name the offending
    /// tensor, never half-load.
    #[test]
    fn new_reports_first_missing_tensor() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 2);
        b.add_u32(KEY_NUM_VOICES, 1);
        b.add_u32(KEY_HIDDEN_DIM, 4);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 8);
        b.add_u32(KEY_ISTFT_HOP, 2);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 8);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        // Use `match` on the Result rather than `expect_err` to avoid a
        // `Debug` requirement on `ProsodyPredictor` (which owns non-Debug
        // internal buffers).
        match ProsodyPredictor::new(&store, &config) {
            Ok(_) => panic!("missing tensors must fail"),
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("predictor.module."),
                    "error should name a predictor tensor; got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// FR-EX-08: odd `hidden_dim` is a loud
    /// [`VokraError::InvalidArgument`] — the BiLSTM hidden width is
    /// `hidden_dim / 2` and an odd `hidden_dim` would silently truncate.
    #[test]
    fn new_rejects_odd_hidden_dim() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 2);
        b.add_u32(KEY_NUM_VOICES, 1);
        b.add_u32(KEY_HIDDEN_DIM, 5);
        b.add_u32(KEY_N_TEXT_LAYERS, 3);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 8);
        b.add_u32(KEY_ISTFT_HOP, 2);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 8);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af"]));
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let config = KokoroConfig::from_gguf(&file).expect("valid config");
        let store = TensorStore::new(file);
        // Use `match` on the Result rather than `expect_err` to avoid a
        // `Debug` requirement on `ProsodyPredictor`.
        match ProsodyPredictor::new(&store, &config) {
            Ok(_) => panic!("odd hidden must fail"),
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("even"),
                    "error should mention 'even'; got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
