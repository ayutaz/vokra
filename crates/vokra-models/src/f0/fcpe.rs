//! FCPE — **Fast Context-based Pitch Estimator** (CFNaiveMelPEInfer
//! topology, upstream `CNChTu/FCPE`).
//!
//! - Upstream: <https://github.com/CNChTu/FCPE>
//! - License: **MIT** (Permissive; no runtime-side attribution obligation —
//!   `docs/license-audit.md` §3.1 sign-off 2026-07-30 yousan).
//!
//! FCPE is one of the reference F0 candidates for the Vokra `f0::*` op
//! surface (FR-OP-83 / CLAUDE.md 音声特化オペレータ §"F0 / Pitch 抽出").
//! Compared with RMVPE it is a leaner, "context-based" pitch classifier
//! (a shallow Conv1D stem into a stack of 6 GLU-conv encoder blocks that
//! emit 360 log-frequency pitch class **sigmoids**, decoded with a
//! 9-window centroid).
//!
//! # Architecture — matches the released `fcpe_c_v001.pt` verbatim
//!
//! Rewritten 2026-07-30 (residual-wave3 FCPE publish fixup, CC ADR): the
//! prior module claimed an attention-based Conformer topology
//! (`vokra_ops::conformer::ConformerEncoder`), but the released
//! `torchfcpe/assets/fcpe_c_v001.pt` (bundled inside every
//! `torchfcpe-0.0.{1..4}-py3-none-any.whl`) is a `CFNaiveMelPEInfer` with
//! `conv_only=True` — a **GLU-conv-only** Sequential encoder with no
//! attention weights. The synthetic-weight unit tests passed with the
//! attention-shaped tensors but a real checkpoint failed at load with
//! "missing `net.layer_stack.0.attn.wq`". The module now mirrors the
//! upstream `torchfcpe/models.py::CFNaiveMelPE` +
//! `torchfcpe/model_conformer_naive.py::CFNEncoderLayer(conv_only=True)`
//! exactly:
//!
//! ```text
//! mel[T, 128]
//!   → transpose to [128, T]                              (Conv1d channel-first)
//!   → input_stack.0  Conv1d(128 → 512, k=3, pad=1)       + bias
//!   → input_stack.1  GroupNorm(groups=4, ch=512)
//!   → LeakyReLU(0.01)                                    (no params)
//!   → input_stack.3  Conv1d(512 → 512, k=3, pad=1)       + bias
//!   → transpose back to [T, 512]
//!
//!   for each of 6 encoder layers (conv_only=True → attention skipped):
//!     residual = x
//!     y = LayerNorm(x, γ=conformer.net.0.w, β=conformer.net.0.b)
//!     y = transpose to [512, T]
//!     y = Conv1d(512 → 2048, k=1)                        + bias    (conformer.net.2)
//!     y = GLU(dim=1) → [1024, T]                                   (conformer.net.3, no params)
//!     y = DepthwiseConv1d(1024, k=31, pad=15, groups=1024) + bias  (conformer.net.4.conv)
//!     y = SiLU(y)                                                   (conformer.net.5, no params)
//!     y = Conv1d(1024 → 512, k=1)                        + bias    (conformer.net.6)
//!     y = transpose back to [T, 512]
//!     x = residual + y
//!
//!   → norm  LayerNorm(x, γ=norm.w, β=norm.b)             (top-level, before output head)
//!   → output_proj  Linear(x)                              (weight-norm folded to plain [360, 512] + bias)
//!   → sigmoid                                             (per-bin, NOT softmax)
//!
//!   for each frame t:
//!     max_val, max_idx = argmax(y[t])
//!     window = y[t, (max_idx-4)..(max_idx+4)]             (9 elements, clamped to [0, 359])
//!     cents  = sum(cent_table[window_idx] * y[window_idx]) / sum(y[window_idx])
//!     voiced = max_val > threshold                        (0.05 default — upstream)
//!     f0     = 10 * 2^(cents / 1200) if voiced else 0
//! ```
//!
//! The **weight-norm reparametrisation** on `output_proj` (upstream stores
//! `weight_g` and `weight_v` and computes
//! `w = weight_g * (weight_v / ||weight_v||_dim=1)` at every forward) is
//! folded into a plain `weight` tensor at prep time
//! (`tools/parity/fcpe_prepare_checkpoint.py`) so the runtime does one
//! Linear pass. `cent_table` and `gaussian_blurred_cent_mask` are upstream
//! *buffers*, not learned; the runtime re-computes the cent grid from
//! `fmin` / `fmax` / `n_pitch_bins` so the prep script drops them.
//!
//! # GGUF schema (`vokra.f0.fcpe.*`)
//!
//! | Key                                          | Type | Default  | Description                                       |
//! | -------------------------------------------- | ---- | -------- | ------------------------------------------------- |
//! | `vokra.f0.fcpe.hop`                          | u32  | 160      | Mel-frame hop, samples                            |
//! | `vokra.f0.fcpe.fmin`                         | f32  | 32.7     | Lowest tracked pitch, Hz                          |
//! | `vokra.f0.fcpe.fmax`                         | f32  | 1975.5   | Highest tracked pitch, Hz                         |
//! | `vokra.f0.fcpe.sample_rate`                  | u32  | 16000    | Expected PCM sample rate                          |
//! | `vokra.f0.fcpe.n_mels`                       | u32  | 128      | Mel channel count (== input_channels upstream)    |
//! | `vokra.f0.fcpe.n_fft`                        | u32  | 1024     | STFT FFT / window size                            |
//! | `vokra.f0.fcpe.n_pitch_bins`                 | u32  | 360      | Output class count (`out_dims`)                   |
//! | `vokra.f0.fcpe.confidence_threshold`         | f32  | 0.05     | V/UV threshold on `max(sigmoid(logits))` — upstream default (was mistakenly 0.006 before the 2026-07-30 rewrite) |
//! | `vokra.f0.fcpe.d_model`                      | u32  | 512      | Hidden width                                      |
//! | `vokra.f0.fcpe.ffn_dim`                      | u32  | 2048     | Pre-GLU pointwise expansion (halves to 1024 after GLU) |
//! | `vokra.f0.fcpe.n_layers`                     | u32  | 6        | Encoder block count                               |
//! | `vokra.f0.fcpe.conv_kernel`                  | u32  | 31       | Depthwise conv kernel (must be odd; upstream default) |
//! | `vokra.f0.fcpe.stem_kernel`                  | u32  | 3        | Input-stack Conv1d kernel (both convs; must be odd, `pad = k/2`) |
//! | `vokra.f0.fcpe.stem_groups`                  | u32  | 4        | Input-stack GroupNorm group count                 |
//!
//! Tensor names (verbatim from upstream state dict, no rename — the prep
//! script just extracts and drops the two buffers `cent_table` and
//! `gaussian_blurred_cent_mask`, folds `output_proj.weight_g/weight_v`
//! into `output_proj.weight`, and writes safetensors):
//!
//! - `input_stack.0.weight` `[d_model, n_mels, stem_kernel]` + `input_stack.0.bias` `[d_model]`
//! - `input_stack.1.weight` `[d_model]` + `input_stack.1.bias` `[d_model]` (GroupNorm)
//! - `input_stack.3.weight` `[d_model, d_model, stem_kernel]` + `input_stack.3.bias` `[d_model]`
//! - `net.encoder_layers.{i}.conformer.net.0.weight` `[d_model]` + `.bias` `[d_model]` (LayerNorm)
//! - `net.encoder_layers.{i}.conformer.net.2.weight` `[ffn_dim, d_model, 1]` + `.bias` `[ffn_dim]`
//! - `net.encoder_layers.{i}.conformer.net.4.conv.weight` `[ffn_dim/2, 1, conv_kernel]` + `.conv.bias` `[ffn_dim/2]`
//! - `net.encoder_layers.{i}.conformer.net.6.weight` `[d_model, ffn_dim/2, 1]` + `.bias` `[d_model]`
//! - `norm.weight` `[d_model]` + `norm.bias` `[d_model]` (top-level LayerNorm)
//! - `output_proj.weight` `[n_pitch_bins, d_model]` + `output_proj.bias` `[n_pitch_bins]`
//!
//! The upstream state dict also carries `net.encoder_layers.{i}.norm.weight/bias`
//! (a LayerNorm declared in `CFNEncoderLayer.__init__` for the attention
//! branch — `self.norm`) that is **never called** at inference when
//! `conv_only=True`. The prep script may drop these; if a caller emits a
//! GGUF that carries them, this loader **ignores** them (their presence is
//! not part of the trigger set).
//!
//! # Forward numerics
//!
//! - `GroupNorm(groups=4, eps=1e-5)` — upstream `torch.nn.GroupNorm` default.
//! - `LeakyReLU(0.01)` — upstream default.
//! - `LayerNorm(eps=1e-5)` — upstream `nn.LayerNorm` default.
//! - `SiLU(x) = x * sigmoid(x)` — upstream `nn.SiLU`.
//! - `GLU(dim=1)`: split channel dim into first half + second half, then
//!   `first_half * sigmoid(second_half)` — upstream `nn.GLU`.
//! - Same-padding for stem Conv1d(k=3, pad=1) and depthwise Conv1d(k=31,
//!   pad=15): upstream `calc_same_padding(k) = (k//2, k//2 - (k+1)%2)`
//!   collapses to a symmetric `(k//2, k//2)` when `k` is odd.
//! - Local-argmax decoder: 9-element window `[max_idx-4, max_idx+4]`
//!   clamped to `[0, out_dims-1]`, weighted centroid over the sigmoid
//!   values.
//!
//! # No ONNX ever
//!
//! Reconfirmed 2026-07-30. The upstream `.pt` is a torch pickle bundled
//! inside the `torchfcpe` wheel; the prep script (`tools/parity/
//! fcpe_prepare_checkpoint.py`) extracts, flattens, and folds weight-norm
//! into a safetensors artifact under upstream state-dict names. No pickle
//! and no ONNX ever enters the runtime (FR-LD-05).

use std::path::Path;

use vokra_core::gguf::{GgufError, GgufFile, GgufMetadataValue};
use vokra_core::ir::graph::{MelAttrs, StftAttrs};
use vokra_ops::mel::MelFilterbank;
use vokra_ops::stft::stft;

use super::{F0Frame, LoadError};

// -- Defaults (see the module doc for the source of each value) --------------

/// Default hop between output frames, in samples (10 ms at 16 kHz — the
/// upstream FCPE / RMVPE / CREPE default).
const DEFAULT_HOP: u32 = 160;
/// Default lower pitch bound in Hz (C1 ≈ 32.7 Hz — the torchfcpe / RMVPE
/// canonical cent-grid zero anchor).
const DEFAULT_FMIN: f32 = 32.7;
/// Default upper pitch bound in Hz (torchfcpe / RMVPE convention — ~ B6).
const DEFAULT_FMAX: f32 = 1975.5;
/// Default expected PCM sample rate.
const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// Default mel-channel count (matches upstream FCPE_v001).
const DEFAULT_N_MELS: u32 = 128;
/// Default STFT window / FFT size (matches upstream FCPE_v001).
const DEFAULT_N_FFT: u32 = 1024;
/// Default pitch class count on the log-frequency grid (RMVPE / torchfcpe
/// convention).
const DEFAULT_N_PITCH_BINS: u32 = 360;
/// Default V/UV threshold on `max(sigmoid(logits))` — **upstream torchfcpe
/// default** (`latent2cents_local_decoder(threshold=0.05)`). The prior
/// value 0.006 pre-dated the 2026-07-30 CFNaiveMelPEInfer rewrite and did
/// not match any upstream default.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.05;
/// Default model hidden width (`CFNaiveMelPE(hidden_dims=512)`).
const DEFAULT_D_MODEL: u32 = 512;
/// Default pre-GLU pointwise expansion width (the Conv1d `inner_dim * 2`
/// = `dim * expansion_factor * 2` in upstream `ConformerConvModule`).
/// Upstream uses `expansion_factor=2`, so with `d_model=512` the pre-GLU
/// width is `2*512*2 = 2048`. After `GLU(dim=1)` it halves to 1024.
const DEFAULT_FFN_DIM: u32 = 2048;
/// Default encoder block count (`CFNaiveMelPE(n_layers=6)`).
const DEFAULT_N_LAYERS: u32 = 6;
/// Default depthwise convolution kernel (upstream `ConformerConvModule(
/// kernel_size=31)`). Must be odd for symmetric same-padding
/// (`calc_same_padding` collapses to `(k//2, k//2)` when `k` is odd).
const DEFAULT_CONV_KERNEL: u32 = 31;
/// Default input-stack Conv1d kernel (upstream:
/// `nn.Conv1d(input_channels, hidden_dims, 3, 1, 1)` = kernel 3, stride 1,
/// padding 1). Both convs in the stack share this kernel size.
const DEFAULT_STEM_KERNEL: u32 = 3;
/// Default input-stack GroupNorm group count (upstream:
/// `nn.GroupNorm(4, hidden_dims)`). Requires `d_model % stem_groups == 0`.
const DEFAULT_STEM_GROUPS: u32 = 4;
/// LeakyReLU negative slope (upstream `nn.LeakyReLU()` default — 0.01).
const LEAKY_SLOPE: f32 = 0.01;
/// LayerNorm / GroupNorm eps (upstream `nn.LayerNorm` / `nn.GroupNorm`
/// default — 1e-5).
const NORM_EPS: f32 = 1e-5;

// -- Config ------------------------------------------------------------------

/// FCPE hyperparameters read from `vokra.f0.fcpe.*` metadata (see the module
/// doc for the schema table + defaults).
#[derive(Debug, Clone, Copy)]
pub struct FcpeConfig {
    /// Mel-frame hop, in samples.
    pub hop: u32,
    /// Lowest tracked pitch, Hz.
    pub fmin: f32,
    /// Highest tracked pitch, Hz.
    pub fmax: f32,
    /// Expected PCM sample rate.
    pub sample_rate: u32,
    /// Number of mel channels emitted by the front-end.
    pub n_mels: u32,
    /// STFT FFT / window size (samples).
    pub n_fft: u32,
    /// Number of log-frequency pitch classes.
    pub n_pitch_bins: u32,
    /// V/UV threshold on `max(sigmoid(logits))` — below this a frame is
    /// marked unvoiced (`hz=0.0`).
    pub confidence_threshold: f32,
    /// Encoder hidden width.
    pub d_model: u32,
    /// Pre-GLU pointwise expansion width (halves after `GLU(dim=1)`).
    pub ffn_dim: u32,
    /// Number of encoder blocks.
    pub n_layers: u32,
    /// Depthwise convolution kernel size (must be odd for symmetric
    /// same-padding).
    pub conv_kernel: u32,
    /// Input-stack Conv1d kernel size (both convs in the stack).
    pub stem_kernel: u32,
    /// Input-stack GroupNorm group count.
    pub stem_groups: u32,
}

impl FcpeConfig {
    /// The canonical default FCPE topology (matches `torchfcpe/assets/
    /// fcpe_c_v001.pt` shipped with `torchfcpe-0.0.{1..4}`).
    pub const fn default_v001() -> Self {
        Self {
            hop: DEFAULT_HOP,
            fmin: DEFAULT_FMIN,
            fmax: DEFAULT_FMAX,
            sample_rate: DEFAULT_SAMPLE_RATE,
            n_mels: DEFAULT_N_MELS,
            n_fft: DEFAULT_N_FFT,
            n_pitch_bins: DEFAULT_N_PITCH_BINS,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            d_model: DEFAULT_D_MODEL,
            ffn_dim: DEFAULT_FFN_DIM,
            n_layers: DEFAULT_N_LAYERS,
            conv_kernel: DEFAULT_CONV_KERNEL,
            stem_kernel: DEFAULT_STEM_KERNEL,
            stem_groups: DEFAULT_STEM_GROUPS,
        }
    }

    /// Reads FCPE config from `vokra.f0.fcpe.*` metadata, defaulting each
    /// absent key to [`Self::default_v001`].
    ///
    /// Returns [`LoadError::Gguf`] if a key is present with a non-numeric or
    /// out-of-range type. Missing keys are honored silently (the design has
    /// canonical defaults for the FCPE_v001 shape).
    pub fn from_gguf(file: &GgufFile) -> Result<Self, LoadError> {
        let hop = read_opt_u32(file, "vokra.f0.fcpe.hop")?.unwrap_or(DEFAULT_HOP);
        let fmin = read_opt_f32(file, "vokra.f0.fcpe.fmin")?.unwrap_or(DEFAULT_FMIN);
        let fmax = read_opt_f32(file, "vokra.f0.fcpe.fmax")?.unwrap_or(DEFAULT_FMAX);
        let sample_rate =
            read_opt_u32(file, "vokra.f0.fcpe.sample_rate")?.unwrap_or(DEFAULT_SAMPLE_RATE);
        let n_mels = read_opt_u32(file, "vokra.f0.fcpe.n_mels")?.unwrap_or(DEFAULT_N_MELS);
        let n_fft = read_opt_u32(file, "vokra.f0.fcpe.n_fft")?.unwrap_or(DEFAULT_N_FFT);
        let n_pitch_bins =
            read_opt_u32(file, "vokra.f0.fcpe.n_pitch_bins")?.unwrap_or(DEFAULT_N_PITCH_BINS);
        let confidence_threshold = read_opt_f32(file, "vokra.f0.fcpe.confidence_threshold")?
            .unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD);
        let d_model = read_opt_u32(file, "vokra.f0.fcpe.d_model")?.unwrap_or(DEFAULT_D_MODEL);
        let ffn_dim = read_opt_u32(file, "vokra.f0.fcpe.ffn_dim")?.unwrap_or(DEFAULT_FFN_DIM);
        let n_layers = read_opt_u32(file, "vokra.f0.fcpe.n_layers")?.unwrap_or(DEFAULT_N_LAYERS);
        let conv_kernel =
            read_opt_u32(file, "vokra.f0.fcpe.conv_kernel")?.unwrap_or(DEFAULT_CONV_KERNEL);
        let stem_kernel =
            read_opt_u32(file, "vokra.f0.fcpe.stem_kernel")?.unwrap_or(DEFAULT_STEM_KERNEL);
        let stem_groups =
            read_opt_u32(file, "vokra.f0.fcpe.stem_groups")?.unwrap_or(DEFAULT_STEM_GROUPS);
        Ok(Self {
            hop,
            fmin,
            fmax,
            sample_rate,
            n_mels,
            n_fft,
            n_pitch_bins,
            confidence_threshold,
            d_model,
            ffn_dim,
            n_layers,
            conv_kernel,
            stem_kernel,
            stem_groups,
        })
    }
}

// -- Weights -----------------------------------------------------------------

/// One encoder block's tensors (upstream `CFNEncoderLayer` with
/// `conv_only=True` — attention weights are absent). All tensors are stored
/// in row-major flat `Vec<f32>` matching the upstream layout.
#[derive(Debug, Clone)]
struct FcpeLayerWeights {
    /// LayerNorm applied at the start of the conv module
    /// (`conformer.net.0`). Shape: gamma/beta = `[d_model]`.
    ln_gamma: Vec<f32>,
    ln_beta: Vec<f32>,
    /// Pointwise Conv1d expansion `d_model → ffn_dim`
    /// (`conformer.net.2`). Weight shape (upstream): `[ffn_dim, d_model, 1]`
    /// which flattens to `[ffn_dim, d_model]` for a per-frame matmul.
    /// Bias shape: `[ffn_dim]`.
    pw1_w: Vec<f32>,
    pw1_b: Vec<f32>,
    /// Depthwise Conv1d applied to the post-GLU stream
    /// (`conformer.net.4.conv`). Weight shape: `[ffn_dim/2, 1, conv_kernel]`
    /// which flattens to `[ffn_dim/2, conv_kernel]` for per-channel
    /// convolution. Bias shape: `[ffn_dim/2]`.
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    /// Pointwise Conv1d projection back `ffn_dim/2 → d_model`
    /// (`conformer.net.6`). Weight shape (upstream): `[d_model, ffn_dim/2, 1]`
    /// which flattens to `[d_model, ffn_dim/2]`. Bias shape: `[d_model]`.
    pw2_w: Vec<f32>,
    pw2_b: Vec<f32>,
}

/// FCPE learned parameters bound from a GGUF.
#[derive(Debug, Clone)]
pub struct FcpeWeights {
    /// Input stack 1st conv (`input_stack.0`). Shape: `[d_model, n_mels,
    /// stem_kernel]` flattened to `[d_model, n_mels * stem_kernel]`.
    stem_w1: Vec<f32>,
    stem_b1: Vec<f32>,
    /// Input stack GroupNorm (`input_stack.1`). Shape: gamma/beta =
    /// `[d_model]`.
    stem_gn_gamma: Vec<f32>,
    stem_gn_beta: Vec<f32>,
    /// Input stack 2nd conv (`input_stack.3`). Shape: `[d_model, d_model,
    /// stem_kernel]` flattened to `[d_model, d_model * stem_kernel]`.
    stem_w2: Vec<f32>,
    stem_b2: Vec<f32>,
    /// The 6 encoder layers.
    layers: Vec<FcpeLayerWeights>,
    /// Top-level LayerNorm applied before the output head
    /// (`self.norm` on `CFNaiveMelPE`). Shape: `[d_model]`.
    head_norm_gamma: Vec<f32>,
    head_norm_beta: Vec<f32>,
    /// Output projection weight (`output_proj`, weight-norm folded at prep
    /// time). Shape: `[n_pitch_bins, d_model]`.
    head_w: Vec<f32>,
    /// Output projection bias. Shape: `[n_pitch_bins]`.
    head_b: Vec<f32>,
}

impl FcpeWeights {
    /// Tries to bind the canonical FCPE tensor set from `file` under `cfg`.
    ///
    /// Returns:
    /// - `Ok(Some(weights))` — every canonical tensor is present and has
    ///   the expected shape.
    /// - `Ok(None)` — the GGUF is metadata-only (no tensor named
    ///   `input_stack.0.weight` or `output_proj.weight`). The handle still
    ///   loads, but [`FCPE::extract`] then refuses it by name; only
    ///   [`FCPE::frame_times`] works on such a handle.
    /// - `Err(LoadError::Gguf)` — a canonical tensor is partially present
    ///   or has a wrong shape (loud, per FR-EX-08 — never a silent
    ///   fallback).
    pub fn try_from_gguf(file: &GgufFile, cfg: &FcpeConfig) -> Result<Option<Self>, LoadError> {
        // Trigger tensors — if either is missing, treat the GGUF as
        // metadata-only (skeleton mode). If ONE is present and the other
        // is not, that is inconsistent and is a loud error below.
        let has_stem = file.tensor_info("input_stack.0.weight").is_some();
        let has_head = file.tensor_info("output_proj.weight").is_some();
        if !has_stem && !has_head {
            return Ok(None);
        }
        if has_stem != has_head {
            return Err(LoadError::Gguf(format!(
                "fcpe: partially populated weight set \
                 (input_stack.0.weight={has_stem}, output_proj.weight={has_head}) — \
                 either both trigger tensors must be present (real weights) or \
                 both absent (metadata-only skeleton)."
            )));
        }

        let d_model = cfg.d_model as usize;
        let n_mels = cfg.n_mels as usize;
        let n_pitch_bins = cfg.n_pitch_bins as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        let inner_dim = ffn_dim / 2;
        let conv_k = cfg.conv_kernel as usize;
        let stem_k = cfg.stem_kernel as usize;

        // Shape invariants (fail loud on config drift).
        if cfg.ffn_dim % 2 != 0 {
            return Err(LoadError::Gguf(format!(
                "fcpe: ffn_dim ({ffn_dim}) must be even (GLU halves it)"
            )));
        }
        if cfg.d_model % cfg.stem_groups != 0 {
            return Err(LoadError::Gguf(format!(
                "fcpe: d_model ({d_model}) must be divisible by stem_groups ({})",
                cfg.stem_groups
            )));
        }
        if cfg.conv_kernel % 2 == 0 {
            return Err(LoadError::Gguf(format!(
                "fcpe: conv_kernel ({conv_k}) must be odd for symmetric \
                 same-padding"
            )));
        }
        if cfg.stem_kernel % 2 == 0 {
            return Err(LoadError::Gguf(format!(
                "fcpe: stem_kernel ({stem_k}) must be odd for symmetric \
                 same-padding"
            )));
        }

        // Stem (`input_stack.[0,1,3]`).
        let stem_w1 = load_vec(file, "input_stack.0.weight", d_model * n_mels * stem_k)?;
        let stem_b1 = load_vec(file, "input_stack.0.bias", d_model)?;
        let stem_gn_gamma = load_vec(file, "input_stack.1.weight", d_model)?;
        let stem_gn_beta = load_vec(file, "input_stack.1.bias", d_model)?;
        let stem_w2 = load_vec(file, "input_stack.3.weight", d_model * d_model * stem_k)?;
        let stem_b2 = load_vec(file, "input_stack.3.bias", d_model)?;

        // Encoder layers (`net.encoder_layers.{i}.conformer.net.[0,2,4.conv,6]`).
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            let ln_gamma = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.0.weight"),
                d_model,
            )?;
            let ln_beta = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.0.bias"),
                d_model,
            )?;
            let pw1_w = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.2.weight"),
                ffn_dim * d_model,
            )?;
            let pw1_b = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.2.bias"),
                ffn_dim,
            )?;
            let dw_w = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.4.conv.weight"),
                inner_dim * conv_k,
            )?;
            let dw_b = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.4.conv.bias"),
                inner_dim,
            )?;
            let pw2_w = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.6.weight"),
                d_model * inner_dim,
            )?;
            let pw2_b = load_vec(
                file,
                &format!("net.encoder_layers.{i}.conformer.net.6.bias"),
                d_model,
            )?;
            layers.push(FcpeLayerWeights {
                ln_gamma,
                ln_beta,
                pw1_w,
                pw1_b,
                dw_w,
                dw_b,
                pw2_w,
                pw2_b,
            });
        }

        // Top-level norm + head.
        let head_norm_gamma = load_vec(file, "norm.weight", d_model)?;
        let head_norm_beta = load_vec(file, "norm.bias", d_model)?;
        let head_w = load_vec(file, "output_proj.weight", n_pitch_bins * d_model)?;
        let head_b = load_vec(file, "output_proj.bias", n_pitch_bins)?;

        Ok(Some(Self {
            stem_w1,
            stem_b1,
            stem_gn_gamma,
            stem_gn_beta,
            stem_w2,
            stem_b2,
            layers,
            head_norm_gamma,
            head_norm_beta,
            head_w,
            head_b,
        }))
    }
}

fn load_vec(file: &GgufFile, name: &str, expected: usize) -> Result<Vec<f32>, LoadError> {
    let v = file
        .tensor_f32(name)
        .map_err(|e| LoadError::Gguf(format!("fcpe: `{name}`: {e:?}")))?;
    if v.len() != expected {
        return Err(LoadError::Gguf(format!(
            "fcpe: `{name}`: expected {expected} elements, got {}",
            v.len()
        )));
    }
    Ok(v)
}

// -- Public API --------------------------------------------------------------

/// FCPE inference handle bound to a Vokra GGUF.
#[derive(Debug, Clone)]
pub struct FCPE {
    cfg: FcpeConfig,
    weights: Option<FcpeWeights>,
    /// Pre-computed cent grid (`out_dims` entries, cent-linear).
    cent_grid: Vec<f32>,
}

impl FCPE {
    /// Binds an FCPE from a Vokra GGUF checkpoint.
    ///
    /// Reads the `vokra.f0.fcpe.*` config chunk (see the module doc for the
    /// schema table + defaults) and — if the file also carries the
    /// canonical tensor set — binds the real weights.
    ///
    /// # Errors
    ///
    /// - [`LoadError::FileNotFound`] if the path cannot be opened.
    /// - [`LoadError::Gguf`] on any other GGUF parse / bind failure,
    ///   including partial / mis-shaped tensor sets (never a silent
    ///   fallback to skeleton — FR-EX-08).
    pub fn from_gguf(path: &Path) -> Result<Self, LoadError> {
        let gguf = GgufFile::open(path).map_err(|e| map_gguf_err(path, e))?;
        let cfg = FcpeConfig::from_gguf(&gguf)?;
        let weights = FcpeWeights::try_from_gguf(&gguf, &cfg)?;
        let cent_grid = build_cent_grid(cfg.fmin, cfg.fmax, cfg.n_pitch_bins as usize);
        Ok(Self {
            cfg,
            weights,
            cent_grid,
        })
    }

    /// Extracts an F0 track from PCM samples by running the real FCPE
    /// forward.
    ///
    /// This is a straight delegation to [`extract_real`](Self::extract_real)
    /// — identical behaviour, identical errors. It exists so the obvious
    /// name on a loaded model is the one that measures pitch, matching
    /// [`super::rmvpe::RMVPE::extract`] and [`super::crepe::CREPE::extract`].
    ///
    /// # History
    ///
    /// Before 2026-08-15 this name returned `Vec<F0Frame>` and answered two
    /// different failures with the same all-zero track: no weights bound,
    /// and a `compute_mel` error swallowed by an `Err(_) =>` arm — the
    /// latter directly under a comment claiming "no silent success on
    /// garbage weights". Downstream, a frame-count-correct zero track reads
    /// as "this audio is entirely unvoiced" rather than "no measurement was
    /// made". The timebase-only half now lives in
    /// [`frame_times`](Self::frame_times).
    ///
    /// # Errors
    ///
    /// Propagates [`extract_real`](Self::extract_real)'s errors verbatim —
    /// see that method for the list.
    pub fn extract(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, vokra_core::VokraError> {
        self.extract_real(pcm, sample_rate)
    }

    /// Runs the **real** FCPE forward on `pcm` and returns a per-hop F0
    /// track.
    ///
    /// The output has exactly `pcm.len() / hop` frames (integer truncation;
    /// tail samples that do not fill a hop are dropped — the frame-count
    /// contract callers align buffers against).
    ///
    /// Reachable both under this name and as [`extract`](Self::extract),
    /// which delegates here verbatim. It is fallible on purpose: every
    /// failure below used to be answered with a frame-count-correct all-zero
    /// track, and silently wrong pitch flows straight into a vocoder or a VC
    /// pipeline (FR-EX-08). Call [`frame_times`](Self::frame_times) when
    /// only the per-hop timestamps are wanted.
    ///
    /// # Errors
    ///
    /// - [`vokra_core::VokraError::ModelLoad`] when no weights were bound
    ///   (a metadata-only GGUF — [`has_real_weights`](Self::has_real_weights)
    ///   reports `false`), or when the bound config is unusable
    ///   (`hop == 0` / `sample_rate == 0`, only reachable from a hand-forged
    ///   GGUF).
    /// - [`vokra_core::VokraError::InvalidArgument`] when `sample_rate` is
    ///   not the rate this checkpoint declares in
    ///   `vokra.f0.fcpe.sample_rate`; the error names both. Vokra never
    ///   silently resamples — resample offline and call again.
    /// - Whatever the STFT / mel front-end raises (propagated verbatim) when
    ///   the front-end cannot run on this PCM. That error used to be
    ///   discarded by an `Err(_) =>` arm that returned zeros.
    pub fn extract_real(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<F0Frame>, vokra_core::VokraError> {
        let Some(weights) = &self.weights else {
            return Err(vokra_core::VokraError::ModelLoad(
                "fcpe: no weight tensors were bound from this GGUF (metadata-only \
                 artifact — neither `input_stack.0.weight` nor `output_proj.weight` \
                 is present), so there is nothing to run; convert a real CNChTu/FCPE \
                 checkpoint, or call `frame_times` if only the per-hop timestamps \
                 are wanted (FR-EX-08: never a zero-filled track)"
                    .to_owned(),
            ));
        };
        let hop = self.cfg.hop as usize;
        if hop == 0 {
            return Err(vokra_core::VokraError::ModelLoad(
                "fcpe: `vokra.f0.fcpe.hop` is 0, so the per-frame timebase is \
                 undefined and no track can be produced (FR-EX-08)"
                    .to_owned(),
            ));
        }
        if self.cfg.sample_rate == 0 {
            return Err(vokra_core::VokraError::ModelLoad(
                "fcpe: `vokra.f0.fcpe.sample_rate` is 0, so the mel filterbank is \
                 undefined and no track can be produced (FR-EX-08)"
                    .to_owned(),
            ));
        }
        if sample_rate != self.cfg.sample_rate {
            return Err(vokra_core::VokraError::InvalidArgument(format!(
                "fcpe: got {sample_rate} Hz PCM but this checkpoint declares \
                 {} Hz in `vokra.f0.fcpe.sample_rate` (its mel filterbank edges and \
                 cent grid are anchored to that rate) — resample offline and call \
                 again; Vokra never silently resamples (FR-EX-08)",
                self.cfg.sample_rate,
            )));
        }

        let n_frames = pcm.len() / hop;
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let sr = sample_rate as f32;
        // Compute mel-spectrogram — the FCPE front-end. A failure here is
        // propagated verbatim: it means the STFT could not run on this PCM
        // (buffer shorter than one window, or a hand-forged GGUF with an
        // unusable n_fft / hop), which is a fact the caller needs, not one
        // to paper over with zeros.
        let mel = self.compute_mel(pcm, n_frames)?;
        let latents = self.forward(&mel, n_frames, weights);
        Ok(self.decode(&latents, n_frames, sr, hop))
    }

    /// Returns the analysis timestamps [`extract`](Self::extract) will emit
    /// for a PCM buffer of `pcm_len` samples, in seconds from the start of
    /// the buffer.
    ///
    /// `result.len()` is the frame-count contract (`pcm_len / hop`,
    /// integer-truncated); `result[i]` is the hop-aligned left edge of frame
    /// `i`. A `hop` of 0 yields an empty slice (the timebase is undefined),
    /// and a `sample_rate` of `0` is clamped to `1` so the column stays
    /// finite rather than `NaN` / `±inf`.
    ///
    /// This runs no weights and cannot fail: it is pure arithmetic over the
    /// config, for callers that need to size or align a buffer before (or
    /// without) running the forward — including holders of a metadata-only
    /// GGUF. It deliberately does **not** return [`F0Frame`]: a frame carries
    /// `hz` / `voiced` / `confidence` columns this method has no evidence
    /// for, and emitting zeros there is exactly the fabricated track the
    /// 2026-08-15 fix removed. Mirrors [`super::rmvpe::RMVPE::frame_times`].
    pub fn frame_times(&self, pcm_len: usize, sample_rate: u32) -> Vec<f32> {
        let hop = self.cfg.hop as usize;
        if hop == 0 {
            return Vec::new();
        }
        let sr = sample_rate.max(1) as f32;
        (0..pcm_len / hop).map(|i| (i * hop) as f32 / sr).collect()
    }

    /// Returns the configured hop length (samples per frame).
    pub fn hop(&self) -> u32 {
        self.cfg.hop
    }
    /// Returns the configured minimum-detectable pitch in Hz.
    pub fn fmin(&self) -> f32 {
        self.cfg.fmin
    }
    /// Returns the configured maximum-detectable pitch in Hz.
    pub fn fmax(&self) -> f32 {
        self.cfg.fmax
    }
    /// Returns `true` when real weights are bound — i.e.
    /// [`extract`](Self::extract) / [`extract_real`](Self::extract_real) can
    /// actually run rather than refusing with a "nothing bound" error.
    ///
    /// Callers that want to branch rather than handle the error can gate on
    /// this first; [`frame_times`](Self::frame_times) is the entry point that
    /// works either way.
    pub fn has_real_weights(&self) -> bool {
        self.weights.is_some()
    }
    /// Immutable access to the FCPE config.
    pub fn config(&self) -> &FcpeConfig {
        &self.cfg
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Runs the CFNaiveMelPEInfer forward and returns the sigmoid latents
    /// `[t_frames, n_pitch_bins]` (row-major).
    fn forward(&self, mel: &[f32], t_frames: usize, w: &FcpeWeights) -> Vec<f32> {
        let d_model = self.cfg.d_model as usize;
        let n_mels = self.cfg.n_mels as usize;
        let ffn_dim = self.cfg.ffn_dim as usize;
        let inner_dim = ffn_dim / 2;
        let stem_k = self.cfg.stem_kernel as usize;
        let stem_pad = stem_k / 2;
        let conv_k = self.cfg.conv_kernel as usize;
        let conv_pad = conv_k / 2;
        let n_bins = self.cfg.n_pitch_bins as usize;
        let gn_groups = self.cfg.stem_groups as usize;

        // --- Input stack ---
        // 1. Transpose mel: input mel is stored as `[t_frames, n_mels]`
        //    (row-major). Conv1d wants channel-first `[n_mels, t_frames]`.
        let mel_ch = transpose_flat(mel, t_frames, n_mels);
        // 2. Conv1d(n_mels → d_model, k=3, pad=1) + bias.
        let mut x = conv1d_same(
            &mel_ch, n_mels, d_model, t_frames, stem_k, stem_pad, &w.stem_w1, &w.stem_b1,
        );
        // 3. GroupNorm(groups=4, ch=d_model).
        group_norm_inplace(
            &mut x,
            d_model,
            t_frames,
            gn_groups,
            &w.stem_gn_gamma,
            &w.stem_gn_beta,
        );
        // 4. LeakyReLU(0.01).
        leaky_relu_inplace(&mut x, LEAKY_SLOPE);
        // 5. Conv1d(d_model → d_model, k=3, pad=1) + bias.
        let mut x = conv1d_same(
            &x, d_model, d_model, t_frames, stem_k, stem_pad, &w.stem_w2, &w.stem_b2,
        );
        // 6. Transpose back to `[t_frames, d_model]`.
        let mut x_tf = transpose_flat(&x, d_model, t_frames);
        // Free the channel-first buffer.
        x.clear();
        x.shrink_to_fit();

        // --- Encoder stack ---
        // Scratch buffers reused across layers.
        let mut normed = vec![0.0f32; t_frames * d_model];
        let mut pw1_out_ch = vec![0.0f32; ffn_dim * t_frames];
        let mut gated_ch = vec![0.0f32; inner_dim * t_frames];
        let mut dw_out_ch = vec![0.0f32; inner_dim * t_frames];
        let mut pw2_out_ch = vec![0.0f32; d_model * t_frames];

        for layer in &w.layers {
            // (a) Pre-norm (LayerNorm on `[t_frames, d_model]`).
            layer_norm_all(
                &x_tf,
                &layer.ln_gamma,
                &layer.ln_beta,
                d_model,
                t_frames,
                &mut normed,
            );
            // (b) Transpose normed to `[d_model, t_frames]`.
            let normed_ch = transpose_flat(&normed, t_frames, d_model);
            // (c) Pointwise conv `d_model → ffn_dim` (Conv1d k=1). Same as
            //     a per-frame matmul: for each frame, `y = W @ x + b`.
            //     `pw1_w` stores `[ffn_dim, d_model, 1]` = `[ffn_dim, d_model]`.
            conv1d_pointwise(
                &normed_ch,
                d_model,
                ffn_dim,
                t_frames,
                &layer.pw1_w,
                &layer.pw1_b,
                &mut pw1_out_ch,
            );
            // (d) GLU(dim=1): split channel dim into first half + second
            //     half, output = first_half * sigmoid(second_half). This
            //     halves the channel count.
            glu_channel(&pw1_out_ch, ffn_dim, t_frames, &mut gated_ch);
            // (e) DepthwiseConv1d(inner_dim, k=31, pad=15, groups=inner_dim).
            depthwise_conv1d_same(
                &gated_ch,
                inner_dim,
                t_frames,
                conv_k,
                conv_pad,
                &layer.dw_w,
                &layer.dw_b,
                &mut dw_out_ch,
            );
            // (f) SiLU: y = y * sigmoid(y).
            silu_inplace(&mut dw_out_ch);
            // (g) Pointwise conv `inner_dim → d_model` (Conv1d k=1).
            conv1d_pointwise(
                &dw_out_ch,
                inner_dim,
                d_model,
                t_frames,
                &layer.pw2_w,
                &layer.pw2_b,
                &mut pw2_out_ch,
            );
            // (h) Transpose back to `[t_frames, d_model]` and residual-add.
            residual_add_transposed(&mut x_tf, &pw2_out_ch, t_frames, d_model);
        }

        // --- Top LayerNorm + output head + sigmoid ---
        let mut latents = vec![0.0f32; t_frames * n_bins];
        let mut normed_t = vec![0.0f32; d_model];
        for t in 0..t_frames {
            // Top-level LayerNorm on this frame.
            let row = &x_tf[t * d_model..(t + 1) * d_model];
            layer_norm_row(row, &w.head_norm_gamma, &w.head_norm_beta, &mut normed_t);
            // Output projection: y[o] = sum(head_w[o, d] * normed[d]) + head_b[o]
            let out_row = &mut latents[t * n_bins..(t + 1) * n_bins];
            for (o, slot) in out_row.iter_mut().enumerate() {
                let w_row = &w.head_w[o * d_model..(o + 1) * d_model];
                let mut acc = w.head_b[o];
                for (dv, wv) in normed_t.iter().zip(w_row.iter()) {
                    acc += dv * wv;
                }
                // Sigmoid activation (per-bin, NOT softmax).
                *slot = sigmoid(acc);
            }
        }
        latents
    }

    /// Local-argmax decoder — mirrors upstream
    /// `latent2cents_local_decoder(threshold=confidence_threshold)`.
    fn decode(&self, latents: &[f32], t_frames: usize, sr: f32, hop: usize) -> Vec<F0Frame> {
        let n_bins = self.cfg.n_pitch_bins as usize;
        let mut out = Vec::with_capacity(t_frames);
        for t in 0..t_frames {
            let time_sec = (t * hop) as f32 / sr;
            let row = &latents[t * n_bins..(t + 1) * n_bins];
            let (peak_idx, peak_val) =
                row.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bp), (i, &p)| {
                        if p > bp { (i, p) } else { (bi, bp) }
                    });
            let voiced = peak_val > self.cfg.confidence_threshold;
            let hz = if voiced {
                let cent = local_argmax_centroid(row, &self.cent_grid, peak_idx);
                cent_to_hz(cent)
            } else {
                0.0
            };
            out.push(F0Frame {
                time_sec,
                hz,
                voiced,
                confidence: peak_val.clamp(0.0, 1.0),
            });
        }
        out
    }

    /// Log-mel-spectrogram front-end (STFT power → Mel filterbank → log).
    ///
    /// Errors carry the reason rather than the `()` this used to return: the
    /// sole caller now propagates them to the user, and "the STFT refused
    /// this n_fft" and "the buffer is shorter than one window" are different
    /// facts a caller needs to tell apart.
    fn compute_mel(
        &self,
        pcm: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>, vokra_core::VokraError> {
        let stft_attrs = StftAttrs::new(self.cfg.n_fft as usize, self.cfg.hop as usize);
        let spec = stft(pcm, &stft_attrs)?;
        if spec.frames == 0 {
            return Err(vokra_core::VokraError::InvalidArgument(format!(
                "fcpe: the STFT produced 0 frames from {} PCM sample(s) at n_fft={} / \
                 hop={} — the buffer is shorter than one analysis window, so there is \
                 no mel to run the encoder on (FR-EX-08)",
                pcm.len(),
                self.cfg.n_fft,
                self.cfg.hop,
            )));
        }
        let mel_attrs = MelAttrs::new(
            self.cfg.sample_rate,
            self.cfg.n_fft as usize,
            self.cfg.n_mels as usize,
        );
        let fb = MelFilterbank::new(&mel_attrs);
        let power = spec.power();
        let mel_full = fb.apply(&power, spec.frames);
        // Convert to log-mel (dynamic-range floor 1e-5 = torchfcpe default).
        let n_mels = self.cfg.n_mels as usize;
        let mut mel_log = vec![0.0f32; spec.frames * n_mels];
        for (dst, &src) in mel_log.iter_mut().zip(mel_full.iter()) {
            *dst = src.max(1e-5).ln();
        }
        // Trim / left-pad to n_frames rows.
        Ok(align_frames(&mel_log, spec.frames, n_frames, n_mels))
    }
}

// -- Forward helpers ---------------------------------------------------------

/// Flat-buffer transpose. `src` is `[rows, cols]` row-major; returns
/// `[cols, rows]` row-major.
fn transpose_flat(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

/// Same-padded Conv1d over a channel-first `[in_ch, t]` buffer, producing
/// a channel-first `[out_ch, t]` buffer (padding preserves the time axis).
/// `weight` is `[out_ch, in_ch, kernel]` row-major, `bias` is `[out_ch]`.
#[allow(clippy::too_many_arguments)]
fn conv1d_same(
    src: &[f32],
    in_ch: usize,
    out_ch: usize,
    t: usize,
    kernel: usize,
    pad: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut dst = vec![0.0f32; out_ch * t];
    for oc in 0..out_ch {
        let w_oc = &weight[oc * in_ch * kernel..(oc + 1) * in_ch * kernel];
        let b_oc = bias[oc];
        let dst_oc = &mut dst[oc * t..(oc + 1) * t];
        for (ti, slot) in dst_oc.iter_mut().enumerate() {
            let mut acc = b_oc;
            for ki in 0..kernel {
                // Input position for this kernel tap (with same-padding).
                let src_idx = ti as isize + ki as isize - pad as isize;
                if src_idx < 0 || src_idx >= t as isize {
                    continue;
                }
                let src_pos = src_idx as usize;
                for ic in 0..in_ch {
                    let w = w_oc[ic * kernel + ki];
                    let x = src[ic * t + src_pos];
                    acc += w * x;
                }
            }
            *slot = acc;
        }
    }
    dst
}

/// Pointwise Conv1d (kernel=1) over a channel-first `[in_ch, t]` buffer.
/// Equivalent to a per-frame Linear projection.
/// `weight` is `[out_ch, in_ch]` row-major, `bias` is `[out_ch]`.
/// `dst` is `[out_ch, t]`, must be pre-sized by the caller.
fn conv1d_pointwise(
    src: &[f32],
    in_ch: usize,
    out_ch: usize,
    t: usize,
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
) {
    for oc in 0..out_ch {
        let w_oc = &weight[oc * in_ch..(oc + 1) * in_ch];
        let b_oc = bias[oc];
        for ti in 0..t {
            let mut acc = b_oc;
            for ic in 0..in_ch {
                acc += w_oc[ic] * src[ic * t + ti];
            }
            dst[oc * t + ti] = acc;
        }
    }
}

/// Depthwise same-padded Conv1d — each input channel convolves with its
/// own kernel (`groups == in_ch`), producing an equal-channel output.
/// `weight` is `[ch, kernel]` row-major (upstream `[ch, 1, kernel]`
/// flattened), `bias` is `[ch]`. `dst` is `[ch, t]`.
#[allow(clippy::too_many_arguments)]
fn depthwise_conv1d_same(
    src: &[f32],
    ch: usize,
    t: usize,
    kernel: usize,
    pad: usize,
    weight: &[f32],
    bias: &[f32],
    dst: &mut [f32],
) {
    for c in 0..ch {
        let w_c = &weight[c * kernel..(c + 1) * kernel];
        let b_c = bias[c];
        let src_c = &src[c * t..(c + 1) * t];
        let dst_c = &mut dst[c * t..(c + 1) * t];
        for (ti, slot) in dst_c.iter_mut().enumerate() {
            let mut acc = b_c;
            for (ki, &wv) in w_c.iter().enumerate() {
                let src_idx = ti as isize + ki as isize - pad as isize;
                if src_idx < 0 || src_idx >= t as isize {
                    continue;
                }
                acc += wv * src_c[src_idx as usize];
            }
            *slot = acc;
        }
    }
}

/// `GLU(dim=channel)`: splits `src` `[full_ch, t]` into first-half + second
/// half along channel dim (each `[full_ch/2, t]`), outputs
/// `first_half * sigmoid(second_half)` into `dst` `[full_ch/2, t]`.
fn glu_channel(src: &[f32], full_ch: usize, t: usize, dst: &mut [f32]) {
    let half = full_ch / 2;
    let split = half * t;
    let (first, second) = src.split_at(split);
    for i in 0..split {
        dst[i] = first[i] * sigmoid(second[i]);
    }
}

/// SiLU in-place: `y = y * sigmoid(y)`.
fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= sigmoid(*v);
    }
}

/// LeakyReLU in-place: `y = max(x, 0) + slope * min(x, 0)`.
fn leaky_relu_inplace(x: &mut [f32], slope: f32) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v *= slope;
        }
    }
}

/// GroupNorm applied in-place on a channel-first `[ch, t]` buffer.
/// `groups` divides `ch`; per-group mean/var are computed over the
/// (group_ch * t) elements. `gamma` / `beta` are per-channel `[ch]`.
fn group_norm_inplace(
    x: &mut [f32],
    ch: usize,
    t: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
) {
    let ch_per_group = ch / groups;
    let group_len = ch_per_group * t;
    for g in 0..groups {
        let start = g * group_len;
        let end = start + group_len;
        let slice = &mut x[start..end];
        let n = slice.len() as f32;
        let mean = slice.iter().sum::<f32>() / n;
        let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let inv = 1.0 / (var + NORM_EPS).sqrt();
        // Apply per-channel affine.
        for ic in 0..ch_per_group {
            let c = g * ch_per_group + ic;
            let gm = gamma[c];
            let bt = beta[c];
            let chan_slice = &mut slice[ic * t..(ic + 1) * t];
            for v in chan_slice.iter_mut() {
                *v = (*v - mean) * inv * gm + bt;
            }
        }
    }
}

/// LayerNorm applied per-row on `[t_frames, d_model]` row-major, writing
/// into `dst` (same shape). Each row is normalized independently over its
/// `d_model` channels, then affine-scaled by `gamma` / `beta` `[d_model]`.
fn layer_norm_all(
    src: &[f32],
    gamma: &[f32],
    beta: &[f32],
    d_model: usize,
    t_frames: usize,
    dst: &mut [f32],
) {
    for t in 0..t_frames {
        let row = &src[t * d_model..(t + 1) * d_model];
        let out = &mut dst[t * d_model..(t + 1) * d_model];
        layer_norm_row(row, gamma, beta, out);
    }
}

/// LayerNorm `y = (x - mean) / sqrt(var + eps) * γ + β` with `eps = 1e-5`
/// on a single row.
fn layer_norm_row(row: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + NORM_EPS).sqrt();
    for i in 0..row.len() {
        out[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

/// Adds the channel-first `[d_model, t_frames]` residual back into the
/// time-first `[t_frames, d_model]` main tensor in-place.
fn residual_add_transposed(x_tf: &mut [f32], residual_ch: &[f32], t_frames: usize, d_model: usize) {
    for t in 0..t_frames {
        for d in 0..d_model {
            x_tf[t * d_model + d] += residual_ch[d * t_frames + t];
        }
    }
}

/// Numerically-stable sigmoid.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// -- Decoding helpers --------------------------------------------------------

/// Local-argmax centroid — mirrors upstream
/// `latent2cents_local_decoder`. Takes the 9-element window
/// `[peak-4, peak+4]` (clamped to `[0, out_dims-1]`) and returns the
/// centroid in cents `sum(cent[i] * probs[i]) / sum(probs[i])`.
fn local_argmax_centroid(probs: &[f32], cent_grid: &[f32], peak_idx: usize) -> f32 {
    const HALF_WINDOW: isize = 4;
    let out_dims = probs.len() as isize;
    let mut num = 0.0f32;
    let mut denom = 0.0f32;
    for offset in -HALF_WINDOW..=HALF_WINDOW {
        let raw = peak_idx as isize + offset;
        // Upstream clamps out-of-range indices to the boundary, so
        // multiple offsets can collapse to the same index — matches
        // `torch.gather` semantics with clamped indices.
        let idx = raw.clamp(0, out_dims - 1) as usize;
        num += probs[idx] * cent_grid[idx];
        denom += probs[idx];
    }
    if denom <= 0.0 {
        cent_grid[peak_idx]
    } else {
        num / denom
    }
}

// -- Front-end frame alignment + cent grid -----------------------------------

/// Best-effort mel-frame count alignment: crops if the STFT emitted extra
/// frames, right-pads with the last available frame if it emitted fewer.
fn align_frames(src: &[f32], src_frames: usize, want_frames: usize, n_mels: usize) -> Vec<f32> {
    if src_frames == want_frames {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; want_frames * n_mels];
    let copy_len = src_frames.min(want_frames);
    out[..copy_len * n_mels].copy_from_slice(&src[..copy_len * n_mels]);
    if src_frames < want_frames && src_frames > 0 {
        let last = &src[(src_frames - 1) * n_mels..src_frames * n_mels];
        for t in src_frames..want_frames {
            let dst = &mut out[t * n_mels..(t + 1) * n_mels];
            dst.copy_from_slice(last);
        }
    }
    out
}

/// Builds the cent-linear log-frequency grid over `[fmin, fmax]` at
/// `n_bins` centers — matches upstream `torch.linspace(f0_to_cent(fmin),
/// f0_to_cent(fmax), out_dims)` where `f0_to_cent(f) = 1200 * log2(f / 10)`.
fn build_cent_grid(fmin: f32, fmax: f32, n_bins: usize) -> Vec<f32> {
    if n_bins == 0 {
        return Vec::new();
    }
    let cent_low = hz_to_cent(fmin);
    let cent_high = hz_to_cent(fmax);
    if n_bins == 1 {
        return vec![cent_low];
    }
    let step = (cent_high - cent_low) / (n_bins as f32 - 1.0);
    (0..n_bins).map(|i| cent_low + step * i as f32).collect()
}

/// `cent → Hz` under `cent(f) = 1200 log₂(f / 10)` (torchfcpe / RMVPE /
/// world all agree on the 10 Hz anchor).
fn cent_to_hz(cent: f32) -> f32 {
    10.0 * (cent / 1200.0).exp2()
}

/// `Hz → cent` under `cent(f) = 1200 log₂(f / 10)`.
fn hz_to_cent(hz: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    1200.0 * (hz / 10.0).log2()
}

// -- Metadata helpers --------------------------------------------------------

fn read_opt_u32(file: &GgufFile, key: &str) -> Result<Option<u32>, LoadError> {
    match file.get(key) {
        Some(v) => match value_as_u64(v).and_then(|n| u32::try_from(n).ok()) {
            Some(n) => Ok(Some(n)),
            None => Err(LoadError::Gguf(format!(
                "fcpe: `{key}` metadata is not a u32 (value type: {v:?})"
            ))),
        },
        None => Ok(None),
    }
}

fn read_opt_f32(file: &GgufFile, key: &str) -> Result<Option<f32>, LoadError> {
    match file.get(key) {
        Some(v) => match value_as_f64(v).map(|n| n as f32) {
            Some(n) => Ok(Some(n)),
            None => Err(LoadError::Gguf(format!(
                "fcpe: `{key}` metadata is not a numeric value (value type: {v:?})"
            ))),
        },
        None => Ok(None),
    }
}

fn value_as_u64(v: &GgufMetadataValue) -> Option<u64> {
    v.as_u64()
}

fn value_as_f64(v: &GgufMetadataValue) -> Option<f64> {
    v.as_f64().or_else(|| v.as_u64().map(|n| n as f64))
}

/// Maps a [`GgufError`] into the local [`LoadError`], collapsing an I/O
/// "not found" into the dedicated [`LoadError::FileNotFound`] variant.
fn map_gguf_err(path: &Path, e: GgufError) -> LoadError {
    match e {
        GgufError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            LoadError::FileNotFound(path.to_path_buf())
        }
        other => LoadError::Gguf(format!("{other:?}")),
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    #[test]
    fn fcpe_load_stub_reports_load_error() {
        let missing = Path::new("/nonexistent/vokra-fcpe-does-not-exist.gguf");
        let err = FCPE::from_gguf(missing).expect_err("missing GGUF must be LoadError");
        assert!(
            matches!(err, LoadError::FileNotFound(_) | LoadError::Gguf(_)),
            "unexpected LoadError variant: {err:?}",
        );
    }

    /// Metadata-only GGUF: config parses, no real weights bound.
    /// `frame_times` still honors the frame-count contract (bare timestamps,
    /// nothing readable as pitch), and `extract` refuses LOUDLY instead of
    /// handing back an all-zero track (RMVPE parity).
    #[test]
    fn fcpe_extract_frame_count_matches_hop_metadata_only() {
        let tmp =
            std::env::temp_dir().join(format!("vokra-fcpe-skeleton-{}.gguf", std::process::id()));
        let bytes = GgufBuilder::new().to_bytes().unwrap();
        std::fs::write(&tmp, &bytes).unwrap();
        let fcpe = FCPE::from_gguf(&tmp).expect("load metadata-only GGUF");
        assert!(!fcpe.has_real_weights(), "no tensors → no bound weights");
        let hop = fcpe.hop() as usize;
        assert_eq!(hop, 160);
        let pcm = vec![0.0f32; hop * 10];

        let times = fcpe.frame_times(pcm.len(), 16_000);
        assert_eq!(times.len(), pcm.len() / hop);
        for (i, t) in times.iter().enumerate() {
            let expected = (i * hop) as f32 / 16_000.0;
            assert!(
                (t - expected).abs() < 1e-9,
                "frame {i}: timestamp {t} != {expected}",
            );
        }

        let Err(err) = fcpe.extract(&pcm, 16_000) else {
            panic!("expected an error when no weight tensors were bound, got a track");
        };
        let msg = err.to_string();
        assert!(
            matches!(err, vokra_core::VokraError::ModelLoad(_)),
            "an unbound weight set is a model-load failure, got: {msg}",
        );
        assert!(
            msg.contains("output_proj.weight"),
            "the error must name a tensor whose absence it detected: {msg}",
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// A partial weight set (stem but no head) must fail loudly — never a
    /// silent fallback to skeleton (FR-EX-08).
    #[test]
    fn fcpe_partial_weight_set_is_loud() {
        let mut b = GgufBuilder::new();
        let d_model = 4u32;
        let n_mels = 4u32;
        let stem_k = 3u32;
        b.add_u32("vokra.f0.fcpe.d_model", d_model);
        b.add_u32("vokra.f0.fcpe.n_mels", n_mels);
        b.add_u32("vokra.f0.fcpe.ffn_dim", 8);
        b.add_u32("vokra.f0.fcpe.n_layers", 1);
        b.add_u32("vokra.f0.fcpe.conv_kernel", 3);
        b.add_u32("vokra.f0.fcpe.stem_kernel", stem_k);
        b.add_u32("vokra.f0.fcpe.stem_groups", 2);
        // Emit `input_stack.0.weight` only — output_proj.weight
        // deliberately missing.
        let n = (d_model * n_mels * stem_k) as usize;
        let payload: Vec<u8> = vec![0u8; n * 4];
        b.add_tensor(
            "input_stack.0.weight",
            GgmlType::F32,
            vec![d_model as u64, n_mels as u64, stem_k as u64],
            payload,
        )
        .unwrap();
        let tmp =
            std::env::temp_dir().join(format!("vokra-fcpe-partial-{}.gguf", std::process::id()));
        std::fs::write(&tmp, b.to_bytes().unwrap()).unwrap();
        let err = FCPE::from_gguf(&tmp).expect_err("partial weight set must be loud");
        assert!(matches!(err, LoadError::Gguf(_)));
        let _ = std::fs::remove_file(&tmp);
    }

    /// Writes a tiny but structurally complete synthetic FCPE checkpoint to
    /// `path` (no `vokra.f0.fcpe.sample_rate` key, so the config keeps the
    /// 16 kHz default) and returns the hop it encodes.
    ///
    /// Shared by every test that needs `has_real_weights() == true`. The
    /// weights come from a fixed-seed SplitMix64 stream, so the forward's
    /// output is reproducible.
    fn write_synthetic_fcpe_gguf(path: &Path) -> u32 {
        write_synthetic_fcpe_gguf_with_n_fft(path, 512)
    }

    /// [`write_synthetic_fcpe_gguf`] with the STFT size under the caller's
    /// control, so a test can hand the front-end a config it must refuse.
    /// No weight shape depends on `n_fft`, so the bind succeeds regardless.
    fn write_synthetic_fcpe_gguf_with_n_fft(path: &Path, n_fft: u32) -> u32 {
        // Deterministic synthetic weights via SplitMix64.
        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_uniform(state: &mut u64, scale: f32) -> f32 {
            let bits = splitmix64(state) >> 40;
            (bits as f32) * (1.0 / (1u32 << 24) as f32) * 2.0 * scale - scale
        }
        fn synth_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| next_uniform(state, scale)).collect()
        }
        fn f32_bytes(v: &[f32]) -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        }

        // Tiny topology so the fixture fits comfortably in a unit test.
        let n_mels = 8u32;
        let d_model = 8u32;
        let ffn_dim = 8u32; // GLU halves to inner_dim = 4
        let n_layers = 1u32;
        let conv_k = 3u32;
        let stem_k = 3u32;
        let stem_groups = 2u32;
        let n_bins = 16u32;
        let hop = 160u32;

        let mut b = GgufBuilder::new();
        b.add_u32("vokra.f0.fcpe.hop", hop);
        b.add_u32("vokra.f0.fcpe.n_mels", n_mels);
        b.add_u32("vokra.f0.fcpe.n_fft", n_fft);
        b.add_u32("vokra.f0.fcpe.n_pitch_bins", n_bins);
        b.add_u32("vokra.f0.fcpe.d_model", d_model);
        b.add_u32("vokra.f0.fcpe.ffn_dim", ffn_dim);
        b.add_u32("vokra.f0.fcpe.n_layers", n_layers);
        b.add_u32("vokra.f0.fcpe.conv_kernel", conv_k);
        b.add_u32("vokra.f0.fcpe.stem_kernel", stem_k);
        b.add_u32("vokra.f0.fcpe.stem_groups", stem_groups);

        let n_mels_u = n_mels as usize;
        let d_model_u = d_model as usize;
        let ffn_dim_u = ffn_dim as usize;
        let inner_dim_u = ffn_dim_u / 2;
        let n_bins_u = n_bins as usize;
        let conv_k_u = conv_k as usize;
        let stem_k_u = stem_k as usize;

        let mut rng = 0xCAFE_BABE_DEAD_BEEFu64;
        let scale = 0.03;

        // Stem
        let stem_w1 = synth_vec(&mut rng, d_model_u * n_mels_u * stem_k_u, scale);
        b.add_tensor(
            "input_stack.0.weight",
            GgmlType::F32,
            vec![d_model as u64, n_mels as u64, stem_k as u64],
            f32_bytes(&stem_w1),
        )
        .unwrap();
        let stem_b1 = synth_vec(&mut rng, d_model_u, scale);
        b.add_tensor(
            "input_stack.0.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&stem_b1),
        )
        .unwrap();
        // GroupNorm gamma/beta initialized to 1/0 for stability.
        b.add_tensor(
            "input_stack.1.weight",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model_u]),
        )
        .unwrap();
        b.add_tensor(
            "input_stack.1.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model_u]),
        )
        .unwrap();
        let stem_w2 = synth_vec(&mut rng, d_model_u * d_model_u * stem_k_u, scale);
        b.add_tensor(
            "input_stack.3.weight",
            GgmlType::F32,
            vec![d_model as u64, d_model as u64, stem_k as u64],
            f32_bytes(&stem_w2),
        )
        .unwrap();
        let stem_b2 = synth_vec(&mut rng, d_model_u, scale);
        b.add_tensor(
            "input_stack.3.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&stem_b2),
        )
        .unwrap();

        // Layer 0
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.0.weight",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model_u]),
        )
        .unwrap();
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.0.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model_u]),
        )
        .unwrap();
        let pw1_w = synth_vec(&mut rng, ffn_dim_u * d_model_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.2.weight",
            GgmlType::F32,
            vec![ffn_dim as u64, d_model as u64, 1u64],
            f32_bytes(&pw1_w),
        )
        .unwrap();
        let pw1_b = synth_vec(&mut rng, ffn_dim_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.2.bias",
            GgmlType::F32,
            vec![ffn_dim as u64],
            f32_bytes(&pw1_b),
        )
        .unwrap();
        let dw_w = synth_vec(&mut rng, inner_dim_u * conv_k_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.4.conv.weight",
            GgmlType::F32,
            vec![inner_dim_u as u64, 1u64, conv_k as u64],
            f32_bytes(&dw_w),
        )
        .unwrap();
        let dw_b = synth_vec(&mut rng, inner_dim_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.4.conv.bias",
            GgmlType::F32,
            vec![inner_dim_u as u64],
            f32_bytes(&dw_b),
        )
        .unwrap();
        let pw2_w = synth_vec(&mut rng, d_model_u * inner_dim_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.6.weight",
            GgmlType::F32,
            vec![d_model as u64, inner_dim_u as u64, 1u64],
            f32_bytes(&pw2_w),
        )
        .unwrap();
        let pw2_b = synth_vec(&mut rng, d_model_u, scale);
        b.add_tensor(
            "net.encoder_layers.0.conformer.net.6.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&pw2_b),
        )
        .unwrap();

        // Head
        b.add_tensor(
            "norm.weight",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![1.0f32; d_model_u]),
        )
        .unwrap();
        b.add_tensor(
            "norm.bias",
            GgmlType::F32,
            vec![d_model as u64],
            f32_bytes(&vec![0.0f32; d_model_u]),
        )
        .unwrap();
        let head_w = synth_vec(&mut rng, n_bins_u * d_model_u, scale);
        b.add_tensor(
            "output_proj.weight",
            GgmlType::F32,
            vec![n_bins as u64, d_model as u64],
            f32_bytes(&head_w),
        )
        .unwrap();
        let head_b = synth_vec(&mut rng, n_bins_u, scale);
        b.add_tensor(
            "output_proj.bias",
            GgmlType::F32,
            vec![n_bins as u64],
            f32_bytes(&head_b),
        )
        .unwrap();

        std::fs::write(path, b.to_bytes().unwrap()).unwrap();
        hop
    }

    /// A deterministic 220 Hz sine, `n_hops` hops long at 16 kHz.
    fn synthetic_sine_pcm(hop: u32, n_hops: usize) -> Vec<f32> {
        let n_samp = hop as usize * n_hops;
        (0..n_samp)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    /// Real-forward smoke test with a tiny synthetic FCPE checkpoint —
    /// exercises the mel front-end + input stack + encoder + head + local-
    /// argmax path end-to-end and asserts finite output + one frame per
    /// hop.
    #[test]
    fn fcpe_forward_end_to_end_smoke() {
        let tmp =
            std::env::temp_dir().join(format!("vokra-fcpe-forward-{}.gguf", std::process::id()));
        let hop = write_synthetic_fcpe_gguf(&tmp);
        let fcpe = FCPE::from_gguf(&tmp).expect("load real weights");
        assert!(fcpe.has_real_weights(), "real weights should bind");
        // Sine input, 20 hops.
        let hop_u = hop as usize;
        let pcm = synthetic_sine_pcm(hop, 20);
        let frames = fcpe
            .extract(&pcm, 16_000)
            .expect("bound weights at the declared rate must produce a real track");
        assert_eq!(frames.len(), pcm.len() / hop_u);
        for f in &frames {
            assert!(f.hz.is_finite(), "hz must be finite (got {})", f.hz);
            assert!(f.hz >= 0.0);
            assert!(f.confidence >= 0.0 && f.confidence <= 1.0);
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// Bound weights + a rate this checkpoint does not declare must be a
    /// LOUD error naming both rates — never a frame-count-correct all-zero
    /// track, and never a silent resample (FR-EX-08).
    #[test]
    fn fcpe_extract_refuses_rate_mismatch_with_bound_weights() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-fcpe-rate-mismatch-{}.gguf",
            std::process::id()
        ));
        let hop = write_synthetic_fcpe_gguf(&tmp);
        let fcpe = FCPE::from_gguf(&tmp).expect("load real weights");
        assert!(
            fcpe.has_real_weights(),
            "the synthetic fixture must bind weights for this test to mean anything",
        );
        assert_eq!(
            fcpe.config().sample_rate,
            16_000,
            "fixture omits the sample_rate key, so the 16 kHz default must apply",
        );

        let pcm = synthetic_sine_pcm(hop, 20);
        let Err(err) = fcpe.extract(&pcm, 44_100) else {
            panic!("expected an error at 44.1 kHz with weights bound, got a track");
        };
        let msg = err.to_string();
        assert!(
            matches!(err, vokra_core::VokraError::InvalidArgument(_)),
            "a rate mismatch is a caller-argument failure, got: {msg}",
        );
        assert!(
            msg.contains("44100"),
            "the error must name the rate it received: {msg}",
        );
        assert!(
            msg.contains("16000"),
            "the error must name the rate it needs: {msg}",
        );

        // `extract_real` is the same refusal, verbatim — a lenient `extract`
        // next to a strict `extract_real` would re-open the hole.
        let Err(direct) = fcpe.extract_real(&pcm, 44_100) else {
            panic!("`extract_real` must refuse what `extract` refuses");
        };
        assert_eq!(direct.to_string(), msg);
        let _ = std::fs::remove_file(&tmp);
    }

    /// A front-end that cannot run must surface its OWN error, not a zero
    /// track.
    ///
    /// Regression pin: the pre-2026-08-15 `extract` matched `compute_mel`
    /// with `Err(_) =>` and returned `n_frames` all-zero rows — directly
    /// under a comment claiming "no silent success on garbage weights". The
    /// weights here are perfectly good; it is the STFT config that is not,
    /// and the caller has to be told which.
    ///
    /// `n_fft = 0` is the deterministic trigger: `vokra_ops::stft::stft`
    /// rejects it outright, no weight shape depends on it (so the bind still
    /// succeeds), and `FcpeConfig::from_gguf` does not screen it — exactly
    /// the "hand-forged GGUF with an unusable config" the old comment said
    /// it was guarding against while quietly answering with zeros.
    #[test]
    fn fcpe_extract_propagates_front_end_error_instead_of_zeros() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-fcpe-unusable-stft-{}.gguf",
            std::process::id()
        ));
        let hop = write_synthetic_fcpe_gguf_with_n_fft(&tmp, 0);
        let fcpe = FCPE::from_gguf(&tmp).expect("load real weights");
        assert!(
            fcpe.has_real_weights(),
            "the weights are fine here — only the STFT config is broken",
        );
        assert_eq!(fcpe.config().n_fft, 0, "the fixture must carry n_fft = 0");

        // 20 hops: the old code would have had 20 rows to fabricate.
        let pcm = synthetic_sine_pcm(hop, 20);
        let Err(err) = fcpe.extract(&pcm, 16_000) else {
            panic!(
                "expected the STFT's own error; an all-zero track here is the bug \
                 (`Err(_) =>` swallowing the front-end failure)"
            );
        };
        let msg = err.to_string();
        assert!(
            msg.contains("stft"),
            "the front-end's own error must reach the caller verbatim, naming the op \
             that refused: {msg}",
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn cent_hz_roundtrip_is_lossless_up_to_fp32() {
        for hz in [55.0f32, 110.0, 220.0, 440.0, 880.0, 1760.0] {
            let cent = hz_to_cent(hz);
            let back = cent_to_hz(cent);
            let rel = (back - hz).abs() / hz;
            assert!(
                rel < 1e-6,
                "cent↔hz roundtrip lost precision at {hz}: back={back}, rel={rel}"
            );
        }
    }

    #[test]
    fn cent_grid_is_monotonic_and_bounded() {
        let grid = build_cent_grid(32.7, 1975.5, 360);
        assert_eq!(grid.len(), 360);
        for w in grid.windows(2) {
            assert!(w[1] > w[0], "cent grid must be strictly increasing");
        }
        assert!((grid[0] - hz_to_cent(32.7)).abs() < 1e-3);
        assert!((grid[359] - hz_to_cent(1975.5)).abs() < 1e-3);
    }

    /// Local-argmax centroid over a peaked distribution — verifies the
    /// 9-element window semantics match upstream `latent2cents_local_decoder`.
    #[test]
    fn local_argmax_centroid_matches_9_element_window() {
        let cent_grid: Vec<f32> = (0..20).map(|i| i as f32 * 10.0).collect();
        let mut probs = vec![0.0f32; 20];
        probs[10] = 1.0; // peak
        probs[9] = 0.5;
        probs[11] = 0.5;
        // Window is [6..14] inclusive (9 elements). Only bins 9, 10, 11
        // are non-zero.
        let expected = (0.5 * 90.0 + 1.0 * 100.0 + 0.5 * 110.0) / (0.5 + 1.0 + 0.5);
        let got = local_argmax_centroid(&probs, &cent_grid, 10);
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }

    /// A near-edge peak clamps the window inside `[0, out_dims-1]` —
    /// matches `torch.gather` with clamped indices.
    #[test]
    fn local_argmax_centroid_clamps_at_edge() {
        let cent_grid: Vec<f32> = (0..20).map(|i| i as f32 * 10.0).collect();
        let mut probs = vec![0.0f32; 20];
        probs[1] = 1.0; // peak near left edge
        // Window offsets [-4..+4] around peak_idx=1 → raw indices
        // [-3..5]; clamped to [0..5]. Only bin 1 is non-zero, so centroid
        // = cent_grid[1] = 10.
        let got = local_argmax_centroid(&probs, &cent_grid, 1);
        assert!((got - 10.0).abs() < 1e-5, "got {got}, want 10.0");
    }

    /// GLU splits channels in half and gates: verify the shape contract
    /// + a hand-computed value.
    #[test]
    fn glu_channel_matches_hand_computed_value() {
        // full_ch=4 → half=2, t=3
        // src = [ch0=A, ch1=B, ch2=G0, ch3=G1] where each ch has 3 elements.
        let src: Vec<f32> = vec![
            1.0, 2.0, 3.0, // ch0
            4.0, 5.0, 6.0, // ch1
            0.0, 1.0, -1.0, // ch2 (gate for ch0)
            2.0, -2.0, 0.5, // ch3 (gate for ch1)
        ];
        let mut dst = vec![0.0f32; 2 * 3];
        glu_channel(&src, 4, 3, &mut dst);
        // Expected: ch0 * sigmoid(ch2), ch1 * sigmoid(ch3)
        let expected: Vec<f32> = vec![
            1.0 * sigmoid(0.0),
            2.0 * sigmoid(1.0),
            3.0 * sigmoid(-1.0),
            4.0 * sigmoid(2.0),
            5.0 * sigmoid(-2.0),
            6.0 * sigmoid(0.5),
        ];
        for (g, e) in dst.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6);
        }
    }

    /// Real-weight forward end-to-end — opt-in via `VOKRA_FCPE_REAL_GGUF`
    /// and a matching WAV via `VOKRA_FCPE_REAL_WAV` (16 kHz mono PCM16)
    /// and assert every frame's Hz is finite. Full upstream parity (vs
    /// `torchfcpe.predict`) is deferred to the parity harness — this is
    /// the "wired end-to-end" smoke test.
    #[test]
    #[ignore = "real-weight leg: requires VOKRA_FCPE_REAL_GGUF + VOKRA_FCPE_REAL_WAV"]
    fn fcpe_real_gguf_forward_end_to_end() {
        let gguf = std::env::var("VOKRA_FCPE_REAL_GGUF")
            .expect("set VOKRA_FCPE_REAL_GGUF to point at a real FCPE Vokra GGUF");
        let wav = std::env::var("VOKRA_FCPE_REAL_WAV")
            .expect("set VOKRA_FCPE_REAL_WAV to point at a 16 kHz mono PCM16 WAV");
        let fcpe = FCPE::from_gguf(Path::new(&gguf)).expect("real GGUF must bind");
        assert!(
            fcpe.has_real_weights(),
            "real weights expected in real GGUF"
        );
        let bytes = std::fs::read(&wav).expect("read WAV");
        let samples = parse_pcm16_wav_16k_mono(&bytes).expect("PCM16 mono @ 16 kHz");
        let frames = fcpe
            .extract(&samples, 16_000)
            .expect("a real GGUF at its declared rate must produce a real track");
        assert!(!frames.is_empty(), "extract must emit frames");
        for f in frames {
            assert!(f.hz.is_finite() && f.hz >= 0.0);
        }
    }

    /// Best-effort 16 kHz mono PCM16 WAV parse — used exclusively by the
    /// env-gated real-weight harness above.
    #[allow(dead_code)]
    fn parse_pcm16_wav_16k_mono(bytes: &[u8]) -> Option<Vec<f32>> {
        if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return None;
        }
        let mut cursor = 12usize;
        let mut sr = 0u32;
        let mut channels = 0u16;
        let mut bits_per_sample = 0u16;
        let mut data_off = 0usize;
        let mut data_len = 0usize;
        while cursor + 8 <= bytes.len() {
            let id = &bytes[cursor..cursor + 4];
            let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().ok()?) as usize;
            let body = cursor + 8;
            if id == b"fmt " && size >= 16 {
                channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().ok()?);
                sr = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().ok()?);
                bits_per_sample = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().ok()?);
            } else if id == b"data" {
                data_off = body;
                data_len = size;
                break;
            }
            cursor = body + size + (size & 1);
        }
        if sr != 16_000 || channels != 1 || bits_per_sample != 16 || data_off == 0 {
            return None;
        }
        let end = data_off + data_len;
        if end > bytes.len() {
            return None;
        }
        let mut out = Vec::with_capacity(data_len / 2);
        for pair in bytes[data_off..end].chunks_exact(2) {
            let s = i16::from_le_bytes([pair[0], pair[1]]);
            out.push(s as f32 / 32768.0);
        }
        Some(out)
    }
}
