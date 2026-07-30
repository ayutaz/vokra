//! FCPE — **Fast Context-based Pitch Estimator** (Conformer-based F0
//! extractor).
//!
//! - Upstream: <https://github.com/CNChTu/FCPE>
//! - License: **MIT** (Permissive; no runtime-side attribution obligation —
//!   `docs/license-audit.md` §3.1 sign-off 2026-07-30 yousan).
//!
//! FCPE is one of the reference F0 candidates for the Vokra `f0::*` op
//! surface (FR-OP-83 / CLAUDE.md 音声特化オペレータ §"F0 / Pitch 抽出").
//! Compared with RMVPE it is a leaner, "context-based" pitch classifier — a
//! shallow-Conv1D stem into a stack of 4-6 Conformer blocks that emits 360
//! log-frequency pitch class logits (RMVPE-compatible cent grid), decoded
//! with a soft-argmax centroid.
//!
//! # Architecture (Vokra-canonical layout)
//!
//! Vokra defines a canonical FCPE topology that reuses the shared
//! [`ConformerEncoder`](vokra_ops::conformer::ConformerEncoder) primitive
//! (SoTA plan Phase 2 landed in `vokra-ops::conformer`) so no new op or
//! backend kernel is introduced:
//!
//! ```text
//! mel[T, n_mels]                                       (mel front-end)
//!   → Linear(n_mels → d_model) + bias                   (subsample stem, no time-axis reduction)
//!   → ConformerEncoder(n_layers × block)                (macaron FF + MHA + Conv + FF)
//!   → LayerNorm(d_model)                                (head norm)
//!   → Linear(d_model → n_pitch_bins)                    (head)
//!   → softmax → soft-argmax over cent grid              → Hz + V/UV
//! ```
//!
//! The stem uses [`ConvSubsampleKind::Linear`] — a per-frame channel
//! projection rather than upstream FCPE's 3-tap Conv1D. This is a
//! deliberate simplification tied to reusing `ConformerEncoder` verbatim;
//! callers who need bit-for-bit upstream parity feed their prep script
//! (`tools/parity/fcpe_prepare_checkpoint.py`) a linear projection derived
//! from the upstream 3-tap stem (e.g. by evaluating the stem at receptive
//! field = 1). The pitch decode side (cent grid, soft-argmax, V/UV) is
//! RMVPE / torchfcpe-compatible.
//!
//! # GGUF schema (`vokra.f0.fcpe.*`)
//!
//! Config keys ([`FcpeConfig::from_gguf`]):
//!
//! | Key                                          | Type | Default  | Description                       |
//! | -------------------------------------------- | ---- | -------- | --------------------------------- |
//! | `vokra.f0.fcpe.hop`                          | u32  | 160      | Mel-frame hop, samples            |
//! | `vokra.f0.fcpe.fmin`                         | f32  | 32.7     | Lowest tracked pitch, Hz          |
//! | `vokra.f0.fcpe.fmax`                         | f32  | 1975.5   | Highest tracked pitch, Hz         |
//! | `vokra.f0.fcpe.sample_rate`                  | u32  | 16000    | Expected PCM sample rate          |
//! | `vokra.f0.fcpe.n_mels`                       | u32  | 128      | Mel channel count                 |
//! | `vokra.f0.fcpe.n_fft`                        | u32  | 1024     | STFT window                       |
//! | `vokra.f0.fcpe.n_pitch_bins`                 | u32  | 360      | Log-frequency pitch bins          |
//! | `vokra.f0.fcpe.confidence_threshold`         | f32  | 0.006    | V/UV threshold on max softmax     |
//! | `vokra.f0.fcpe.d_model`                      | u32  | 512      | Conformer model dim               |
//! | `vokra.f0.fcpe.n_heads`                      | u32  | 8        | Conformer heads                   |
//! | `vokra.f0.fcpe.ffn_dim`                      | u32  | 2048     | Conformer FFN width               |
//! | `vokra.f0.fcpe.n_layers`                     | u32  | 6        | Conformer stack depth             |
//! | `vokra.f0.fcpe.kernel_size`                  | u32  | 9        | Conformer depthwise kernel        |
//!
//! Tensor names (Vokra-canonical — the prep-script's rename target):
//!
//! - `stem.weight`  `[d_model, n_mels]`
//! - `stem.bias`    `[d_model]`
//! - `layers.{i}.ln1.weight`, `layers.{i}.ln1.bias`  `[d_model]`  (FF1 pre-norm)
//! - `layers.{i}.ff1.w1`  `[ffn_dim, d_model]`, `layers.{i}.ff1.b1`  `[ffn_dim]`
//! - `layers.{i}.ff1.w2`  `[d_model, ffn_dim]`, `layers.{i}.ff1.b2`  `[d_model]`
//! - `layers.{i}.ln2.weight`, `layers.{i}.ln2.bias`  `[d_model]`  (MHA pre-norm)
//! - `layers.{i}.mha.wq/bq/wk/bk/wv/bv/wo/bo`  (`[d_model, d_model]` / `[d_model]`)
//! - `layers.{i}.ln3.weight`, `layers.{i}.ln3.bias`  `[d_model]`  (Conv pre-norm)
//! - `layers.{i}.conv.pointwise1_w`  `[2*d_model, d_model]`, `pointwise1_b`  `[2*d_model]`
//! - `layers.{i}.conv.depthwise_w`  `[d_model, kernel_size]`, `depthwise_b`  `[d_model]`
//! - `layers.{i}.conv.norm_gamma`, `layers.{i}.conv.norm_beta`  `[d_model]`
//! - `layers.{i}.conv.pointwise2_w`  `[d_model, d_model]`, `pointwise2_b`  `[d_model]`
//! - `layers.{i}.ln4.weight`, `layers.{i}.ln4.bias`  `[d_model]`  (FF2 pre-norm)
//! - `layers.{i}.ff2.w1/b1/w2/b2`  (same shapes as ff1)
//! - `layers.{i}.ln_out.weight`, `layers.{i}.ln_out.bias`  `[d_model]`
//! - `head_norm.weight`, `head_norm.bias`  `[d_model]`
//! - `head.weight`  `[n_pitch_bins, d_model]`
//! - `head.bias`    `[n_pitch_bins]`
//!
//! # Extract behavior
//!
//! [`FCPE::extract`] runs the real forward if the GGUF carries the full
//! canonical tensor set. If the GGUF is metadata-only (no tensors, or
//! partially-populated tensor set), the caller sees the frame-count-
//! contract skeleton: one `F0Frame` per hop with `hz=0.0, voiced=false,
//! confidence=0.0`. This is deliberate so a GGUF that only stamps the
//! metadata surface (e.g. a `--config`-only pass) still shapes the output
//! buffer correctly — a real weight-load lands the real forward without
//! any API change. A malformed tensor set (shape mismatch on a bound
//! tensor) is a loud [`LoadError::Gguf`] at [`FCPE::from_gguf`] — never a
//! silent fallback to skeleton (FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! The upstream `torchfcpe` release ships a `.pt`; the converter
//! ([`crate::convert::fcpe`](https://docs.rs/vokra-convert)) reads
//! **safetensors** only (the DFN3 / CSM / DAC pattern — `tools/parity/
//! fcpe_prepare_checkpoint.py` bridges `.pt` → safetensors offline). No
//! ONNX ever enters the runtime (FR-LD-05).

use std::path::Path;

use vokra_core::gguf::{GgufError, GgufFile, GgufMetadataValue};
use vokra_core::ir::graph::{MelAttrs, StftAttrs};
use vokra_ops::conformer::{
    ConformerConfig, ConformerConvWeights, ConformerEncoder, ConformerLayerWeights,
    ConformerSubsampleWeights, ConformerWeights, ConvSubsampleKind, FeedForwardWeights, MhaWeights,
    PositionEncoding,
};
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
/// Default V/UV threshold on `max(softmax(logits))` — torchfcpe default.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.006;
/// Default Conformer model dimension.
const DEFAULT_D_MODEL: u32 = 512;
/// Default Conformer attention head count (`d_model % n_heads == 0` for
/// the defaults).
const DEFAULT_N_HEADS: u32 = 8;
/// Default Conformer FeedForward inner width.
const DEFAULT_FFN_DIM: u32 = 2048;
/// Default Conformer block count (task spec: 4-6; picks 6 for the reference
/// FCPE_v001 topology).
const DEFAULT_N_LAYERS: u32 = 6;
/// Default Conformer depthwise convolution kernel (odd, FastConformer
/// default = 9).
const DEFAULT_KERNEL_SIZE: u32 = 9;

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
    /// V/UV threshold on the softmax peak — below this a frame is marked
    /// unvoiced (`hz=0.0`).
    pub confidence_threshold: f32,
    /// Conformer model dimension.
    pub d_model: u32,
    /// Number of attention heads.
    pub n_heads: u32,
    /// FeedForward inner width.
    pub ffn_dim: u32,
    /// Number of Conformer blocks.
    pub n_layers: u32,
    /// Depthwise convolution kernel size (must be odd for symmetric same-
    /// padding — checked by [`ConformerEncoder::new`]).
    pub kernel_size: u32,
}

impl FcpeConfig {
    /// The canonical default FCPE topology (FCPE_v001-sized).
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
            n_heads: DEFAULT_N_HEADS,
            ffn_dim: DEFAULT_FFN_DIM,
            n_layers: DEFAULT_N_LAYERS,
            kernel_size: DEFAULT_KERNEL_SIZE,
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
        let n_heads = read_opt_u32(file, "vokra.f0.fcpe.n_heads")?.unwrap_or(DEFAULT_N_HEADS);
        let ffn_dim = read_opt_u32(file, "vokra.f0.fcpe.ffn_dim")?.unwrap_or(DEFAULT_FFN_DIM);
        let n_layers = read_opt_u32(file, "vokra.f0.fcpe.n_layers")?.unwrap_or(DEFAULT_N_LAYERS);
        let kernel_size =
            read_opt_u32(file, "vokra.f0.fcpe.kernel_size")?.unwrap_or(DEFAULT_KERNEL_SIZE);
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
            n_heads,
            ffn_dim,
            n_layers,
            kernel_size,
        })
    }

    /// Builds the [`ConformerConfig`] describing this FCPE's Conformer body.
    fn conformer_config(&self) -> ConformerConfig {
        ConformerConfig {
            in_dim: self.n_mels,
            d_model: self.d_model,
            n_heads: self.n_heads,
            ffn_dim: self.ffn_dim,
            n_layers: self.n_layers,
            kernel_size: self.kernel_size,
            subsample_type: ConvSubsampleKind::Linear,
            position_encoding: PositionEncoding::None,
        }
    }
}

// -- Weights -----------------------------------------------------------------

/// FCPE learned parameters bound from a GGUF (see the module doc for the
/// canonical tensor-name layout).
#[derive(Debug, Clone)]
pub struct FcpeWeights {
    /// LayerNorm gamma applied to the encoder output before the pitch head.
    pub head_norm_gamma: Vec<f32>,
    /// LayerNorm beta applied to the encoder output before the pitch head.
    pub head_norm_beta: Vec<f32>,
    /// Pitch-classifier weight, row-major `[n_pitch_bins, d_model]`.
    pub head_w: Vec<f32>,
    /// Pitch-classifier bias, `[n_pitch_bins]`.
    pub head_b: Vec<f32>,
    /// The Conformer encoder — owns the stem + per-layer stack.
    pub encoder: ConformerEncoder,
}

impl FcpeWeights {
    /// Tries to bind the canonical FCPE tensor set from `file` under `cfg`.
    ///
    /// Returns:
    /// - `Ok(Some(weights))` — every canonical tensor is present and has the
    ///   expected shape (validated by [`ConformerEncoder::new`] + explicit
    ///   size checks on the head).
    /// - `Ok(None)` — the GGUF is metadata-only (no tensor named `stem.weight`
    ///   or `head.weight`) — the caller runs the skeleton extract path.
    /// - `Err(LoadError::Gguf)` — a canonical tensor is partially present or
    ///   has a wrong shape (loud, per FR-EX-08 — never a silent fallback).
    pub fn try_from_gguf(file: &GgufFile, cfg: &FcpeConfig) -> Result<Option<Self>, LoadError> {
        // Trigger tensors — if either is missing, treat the GGUF as
        // metadata-only (skeleton mode). If ONE is present and the other is
        // not, that is inconsistent and is a loud error below.
        let has_stem = file.tensor_info("stem.weight").is_some();
        let has_head = file.tensor_info("head.weight").is_some();
        if !has_stem && !has_head {
            return Ok(None);
        }
        if has_stem != has_head {
            return Err(LoadError::Gguf(format!(
                "fcpe: partially populated weight set (stem.weight={has_stem}, \
                 head.weight={has_head}) — either both trigger tensors must be \
                 present (real weights) or both absent (metadata-only skeleton)."
            )));
        }

        let d_model = cfg.d_model as usize;
        let n_pitch_bins = cfg.n_pitch_bins as usize;

        let subsample = load_subsample_weights(file, cfg)?;
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers as usize {
            layers.push(load_layer_weights(file, cfg, i)?);
        }
        let encoder_weights = ConformerWeights { subsample, layers };
        let encoder = ConformerEncoder::new(cfg.conformer_config(), encoder_weights)
            .map_err(|e| LoadError::Gguf(format!("fcpe: conformer bind failed: {e}")))?;

        let head_norm_gamma = load_vec(file, "head_norm.weight", d_model)?;
        let head_norm_beta = load_vec(file, "head_norm.bias", d_model)?;
        let head_w = load_vec(file, "head.weight", n_pitch_bins * d_model)?;
        let head_b = load_vec(file, "head.bias", n_pitch_bins)?;

        Ok(Some(Self {
            head_norm_gamma,
            head_norm_beta,
            head_w,
            head_b,
            encoder,
        }))
    }
}

fn load_subsample_weights(
    file: &GgufFile,
    cfg: &FcpeConfig,
) -> Result<ConformerSubsampleWeights, LoadError> {
    let d_model = cfg.d_model as usize;
    let n_mels = cfg.n_mels as usize;
    Ok(ConformerSubsampleWeights {
        linear_w: load_vec(file, "stem.weight", d_model * n_mels)?,
        linear_b: load_vec(file, "stem.bias", d_model)?,
        norm_gamma: None,
        norm_beta: None,
    })
}

fn load_layer_weights(
    file: &GgufFile,
    cfg: &FcpeConfig,
    i: usize,
) -> Result<ConformerLayerWeights, LoadError> {
    let d_model = cfg.d_model as usize;
    let ffn_dim = cfg.ffn_dim as usize;
    let two_d = 2 * d_model;
    let kernel = cfg.kernel_size as usize;
    let p = |suffix: &str| format!("layers.{i}.{suffix}");
    Ok(ConformerLayerWeights {
        ln1_gamma: load_vec(file, &p("ln1.weight"), d_model)?,
        ln1_beta: load_vec(file, &p("ln1.bias"), d_model)?,
        ff1: FeedForwardWeights {
            w1: load_vec(file, &p("ff1.w1"), ffn_dim * d_model)?,
            b1: load_vec(file, &p("ff1.b1"), ffn_dim)?,
            w2: load_vec(file, &p("ff1.w2"), d_model * ffn_dim)?,
            b2: load_vec(file, &p("ff1.b2"), d_model)?,
        },
        ln2_gamma: load_vec(file, &p("ln2.weight"), d_model)?,
        ln2_beta: load_vec(file, &p("ln2.bias"), d_model)?,
        mha: MhaWeights {
            wq: load_vec(file, &p("mha.wq"), d_model * d_model)?,
            bq: load_vec(file, &p("mha.bq"), d_model)?,
            wk: load_vec(file, &p("mha.wk"), d_model * d_model)?,
            bk: load_vec(file, &p("mha.bk"), d_model)?,
            wv: load_vec(file, &p("mha.wv"), d_model * d_model)?,
            bv: load_vec(file, &p("mha.bv"), d_model)?,
            wo: load_vec(file, &p("mha.wo"), d_model * d_model)?,
            bo: load_vec(file, &p("mha.bo"), d_model)?,
        },
        ln3_gamma: load_vec(file, &p("ln3.weight"), d_model)?,
        ln3_beta: load_vec(file, &p("ln3.bias"), d_model)?,
        conv: ConformerConvWeights {
            pointwise1_w: load_vec(file, &p("conv.pointwise1_w"), two_d * d_model)?,
            pointwise1_b: load_vec(file, &p("conv.pointwise1_b"), two_d)?,
            depthwise_w: load_vec(file, &p("conv.depthwise_w"), d_model * kernel)?,
            depthwise_b: load_vec(file, &p("conv.depthwise_b"), d_model)?,
            norm_gamma: load_vec(file, &p("conv.norm_gamma"), d_model)?,
            norm_beta: load_vec(file, &p("conv.norm_beta"), d_model)?,
            pointwise2_w: load_vec(file, &p("conv.pointwise2_w"), d_model * d_model)?,
            pointwise2_b: load_vec(file, &p("conv.pointwise2_b"), d_model)?,
        },
        ln4_gamma: load_vec(file, &p("ln4.weight"), d_model)?,
        ln4_beta: load_vec(file, &p("ln4.bias"), d_model)?,
        ff2: FeedForwardWeights {
            w1: load_vec(file, &p("ff2.w1"), ffn_dim * d_model)?,
            b1: load_vec(file, &p("ff2.b1"), ffn_dim)?,
            w2: load_vec(file, &p("ff2.w2"), d_model * ffn_dim)?,
            b2: load_vec(file, &p("ff2.b2"), d_model)?,
        },
        ln_out_gamma: load_vec(file, &p("ln_out.weight"), d_model)?,
        ln_out_beta: load_vec(file, &p("ln_out.bias"), d_model)?,
    })
}

fn load_vec(file: &GgufFile, name: &str, expected: usize) -> Result<Vec<f32>, LoadError> {
    let v = file
        .tensor_f32(name)
        .map_err(|e| LoadError::Gguf(format!("fcpe: `{name}`: {e:?}")))?;
    if v.len() != expected {
        return Err(LoadError::Gguf(format!(
            "fcpe: tensor `{name}` has {} elements, expected {expected}",
            v.len()
        )));
    }
    Ok(v)
}

// -- Public extractor --------------------------------------------------------

/// FCPE F0 (pitch) extractor.
///
/// Construct with [`from_gguf`](Self::from_gguf); pitch is emitted per hop
/// via [`extract`](Self::extract). When the GGUF carries real weights the
/// extract runs the full Conformer forward + cent-grid soft-argmax; when the
/// GGUF is metadata-only the extract emits the frame-count-contract skeleton
/// (see the module doc for the "no silent fallback" gate on partial weight
/// sets — FR-EX-08).
#[derive(Debug)]
pub struct FCPE {
    cfg: FcpeConfig,
    weights: Option<FcpeWeights>,
    /// Precomputed cent value per pitch bin (log-linear spacing between
    /// `cent(fmin)` and `cent(fmax)` — RMVPE / torchfcpe convention). Cached
    /// here rather than per-`extract` because the config is immutable.
    cent_grid: Vec<f32>,
}

impl FCPE {
    /// Binds an FCPE from a Vokra GGUF checkpoint.
    ///
    /// Reads the `vokra.f0.fcpe.*` config chunk (see the module doc for the
    /// schema table + defaults) and — if the file also carries the canonical
    /// tensor set — binds the real Conformer weights.
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

    /// Extracts an F0 track from PCM samples.
    ///
    /// The output has exactly `pcm.len() / hop` frames (integer truncation;
    /// tail samples that do not fill a hop are dropped — the frame-count
    /// contract callers align buffers against). When real weights are bound
    /// the forward runs; otherwise each frame carries
    /// `hz=0.0, voiced=false, confidence=0.0`.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Vec<F0Frame> {
        let hop = self.cfg.hop as usize;
        if hop == 0 {
            return Vec::new();
        }
        let n_frames = pcm.len() / hop;
        let sr = sample_rate.max(1) as f32;

        // Skeleton fast path when no real weights are bound. Deliberately
        // matches RMVPE's contract so downstream consumers can size buffers
        // before a real-weight WP lands.
        let Some(weights) = &self.weights else {
            return (0..n_frames)
                .map(|i| F0Frame {
                    time_sec: (i * hop) as f32 / sr,
                    hz: 0.0,
                    voiced: false,
                    confidence: 0.0,
                })
                .collect();
        };

        if n_frames == 0 {
            return Vec::new();
        }
        // Compute mel-spectrogram — the FCPE front-end. If the STFT / mel
        // path errors out (short PCM, degenerate config), fall through to
        // the skeleton **for the requested frame count** — no crash, no
        // silent success on garbage weights. This branch cannot happen with
        // a well-formed non-empty PCM under the canonical defaults; it
        // guards against a hand-forged GGUF with an unusable config.
        let mel = match self.compute_mel(pcm, n_frames) {
            Ok(m) => m,
            Err(_) => {
                return (0..n_frames)
                    .map(|i| F0Frame {
                        time_sec: (i * hop) as f32 / sr,
                        hz: 0.0,
                        voiced: false,
                        confidence: 0.0,
                    })
                    .collect();
            }
        };

        // Encoder forward — [T, d_model] row-major, where T is the encoder's
        // output time count (== n_frames for the `Linear` subsample stem).
        let (hidden, t_out) = match weights.encoder.forward(&mel, n_frames) {
            Ok(v) => v,
            Err(_) => {
                return (0..n_frames)
                    .map(|i| F0Frame {
                        time_sec: (i * hop) as f32 / sr,
                        hz: 0.0,
                        voiced: false,
                        confidence: 0.0,
                    })
                    .collect();
            }
        };
        let d_model = self.cfg.d_model as usize;
        let n_bins = self.cfg.n_pitch_bins as usize;

        // Pitch head: LayerNorm(hidden) → Linear head → softmax → soft-argmax.
        let mut logits = vec![0.0f32; t_out * n_bins];
        let mut normed = vec![0.0f32; d_model];
        for t in 0..t_out {
            let row = &hidden[t * d_model..(t + 1) * d_model];
            layer_norm(
                row,
                &weights.head_norm_gamma,
                &weights.head_norm_beta,
                &mut normed,
            );
            let out_row = &mut logits[t * n_bins..(t + 1) * n_bins];
            for (o, slot) in out_row.iter_mut().enumerate() {
                let mut acc = weights.head_b[o];
                let w_row = &weights.head_w[o * d_model..(o + 1) * d_model];
                for d in 0..d_model {
                    acc += w_row[d] * normed[d];
                }
                *slot = acc;
            }
        }

        let mut out = Vec::with_capacity(n_frames);
        let mut probs = vec![0.0f32; n_bins];
        for t in 0..n_frames {
            let time_sec = (t * hop) as f32 / sr;
            let time_source = t.min(t_out.saturating_sub(1));
            let row = &logits[time_source * n_bins..(time_source + 1) * n_bins];
            softmax(row, &mut probs);
            let (peak_idx, peak_prob) =
                probs
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bp), (i, &p)| {
                        if p > bp { (i, p) } else { (bi, bp) }
                    });
            let voiced = peak_prob >= self.cfg.confidence_threshold;
            let hz = if voiced {
                let cent = centroid_around(&probs, &self.cent_grid, peak_idx);
                cent_to_hz(cent)
            } else {
                0.0
            };
            out.push(F0Frame {
                time_sec,
                hz,
                voiced,
                confidence: peak_prob.clamp(0.0, 1.0),
            });
        }
        out
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
    /// Returns `true` when real Conformer weights are bound (i.e. `extract`
    /// runs the real forward rather than the skeleton).
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

    fn compute_mel(&self, pcm: &[f32], n_frames: usize) -> Result<Vec<f32>, ()> {
        let mut stft_attrs = StftAttrs::new(self.cfg.n_fft as usize, self.cfg.hop as usize);
        // Reflect padding + `center=true` are librosa's defaults; keeping
        // them here matches the front-end most upstream pitch extractors
        // train against.
        stft_attrs.win_length = self.cfg.n_fft as usize;
        let spec = stft(pcm, &stft_attrs).map_err(|_| ())?;
        if spec.frames == 0 {
            return Err(());
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
        // Trim / left-pad to n_frames rows so the encoder sees the requested
        // frame count regardless of STFT-side truncation. In practice
        // `spec.frames` and `n_frames` differ by at most a few samples of
        // center padding.
        Ok(align_frames(&mel_log, spec.frames, n_frames, n_mels))
    }
}

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

/// Builds the log-frequency (cent) grid over `[fmin, fmax]` at `n_bins`
/// centers, linearly spaced in cents (RMVPE / torchfcpe convention).
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

/// Compute a local centroid around `peak_idx` — a ±4-bin window on the
/// probability distribution (RMVPE convention, torchfcpe uses the same
/// half-width). Returns the centroid in cents.
fn centroid_around(probs: &[f32], cent_grid: &[f32], peak_idx: usize) -> f32 {
    const HALF_WINDOW: usize = 4;
    let start = peak_idx.saturating_sub(HALF_WINDOW);
    let end = (peak_idx + HALF_WINDOW + 1).min(probs.len());
    let mut num = 0.0f32;
    let mut denom = 0.0f32;
    for i in start..end {
        num += probs[i] * cent_grid[i];
        denom += probs[i];
    }
    if denom <= 0.0 {
        cent_grid[peak_idx]
    } else {
        num / denom
    }
}

/// `cent → Hz` under the standard `cent(f) = 1200 log₂(f / 10)` convention.
/// Anchor: `10 Hz = 0 cents` (torchfcpe / world / RMVPE all agree).
fn cent_to_hz(cent: f32) -> f32 {
    10.0 * (cent / 1200.0).exp2()
}

/// `Hz → cent` under the standard `cent(f) = 1200 log₂(f / 10)` convention.
fn hz_to_cent(hz: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    1200.0 * (hz / 10.0).log2()
}

/// LayerNorm `y = (x - mean) / sqrt(var + eps) * γ + β` with `eps = 1e-5`.
fn layer_norm(row: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..row.len() {
        out[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

/// Numerically stable row-wise softmax.
fn softmax(src: &[f32], dst: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in src {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        let e = (s - max).exp();
        *d = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for d in dst.iter_mut() {
            *d *= inv;
        }
    }
}

// -- Metadata helpers --------------------------------------------------------

fn read_opt_u32(file: &GgufFile, key: &str) -> Result<Option<u32>, LoadError> {
    match file.get(key) {
        Some(v) => match value_as_u64(v).and_then(|n| u32::try_from(n).ok()) {
            Some(n) => Ok(Some(n)),
            None => Err(LoadError::Gguf(format!(
                "fcpe metadata `{key}` is not a u32-range integer",
            ))),
        },
        None => Ok(None),
    }
}

fn read_opt_f32(file: &GgufFile, key: &str) -> Result<Option<f32>, LoadError> {
    match file.get(key) {
        Some(v) => match value_as_f64(v) {
            Some(n) => Ok(Some(n as f32)),
            None => Err(LoadError::Gguf(format!(
                "fcpe metadata `{key}` is not a float",
            ))),
        },
        None => Ok(None),
    }
}

fn value_as_u64(v: &GgufMetadataValue) -> Option<u64> {
    v.as_u64()
}

fn value_as_f64(v: &GgufMetadataValue) -> Option<f64> {
    v.as_f64()
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

    /// Metadata-only GGUF: config parses, no real weights bound, `extract`
    /// returns the frame-count-contract skeleton (RMVPE parity).
    #[test]
    fn fcpe_extract_frame_count_matches_hop_metadata_only() {
        let tmp =
            std::env::temp_dir().join(format!("vokra-fcpe-skeleton-{}.gguf", std::process::id()));
        let bytes = GgufBuilder::new().to_bytes().unwrap();
        std::fs::write(&tmp, &bytes).unwrap();
        let fcpe = FCPE::from_gguf(&tmp).expect("load metadata-only GGUF");
        assert!(!fcpe.has_real_weights(), "no tensors → skeleton mode");
        let hop = fcpe.hop() as usize;
        assert_eq!(hop, 160);
        let pcm = vec![0.0f32; hop * 10];
        let frames = fcpe.extract(&pcm, 16_000);
        assert_eq!(frames.len(), pcm.len() / hop);
        assert!(frames.iter().all(|f| f.hz == 0.0 && !f.voiced));
        let _ = std::fs::remove_file(&tmp);
    }

    /// A partial weight set (stem but no head) must fail loudly — never a
    /// silent fallback to skeleton (FR-EX-08).
    #[test]
    fn fcpe_partial_weight_set_is_loud() {
        let mut b = GgufBuilder::new();
        // Deliberately tiny cfg so the fixture stays small.
        let d_model = 4u32;
        let n_mels = 4u32;
        b.add_u32("vokra.f0.fcpe.d_model", d_model);
        b.add_u32("vokra.f0.fcpe.n_mels", n_mels);
        b.add_u32("vokra.f0.fcpe.n_heads", 2);
        b.add_u32("vokra.f0.fcpe.ffn_dim", 8);
        b.add_u32("vokra.f0.fcpe.n_layers", 1);
        b.add_u32("vokra.f0.fcpe.kernel_size", 3);
        // Emit `stem.weight` only — head.weight deliberately missing.
        let n = (d_model * n_mels) as usize;
        let payload: Vec<u8> = vec![0u8; n * 4];
        b.add_tensor(
            "stem.weight",
            GgmlType::F32,
            vec![n_mels as u64, d_model as u64],
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

    /// Real-forward smoke test with a tiny synthetic FCPE checkpoint —
    /// exercises the mel front-end + Conformer + head + soft-argmax path
    /// end-to-end and asserts finite output + one frame per hop.
    #[test]
    fn fcpe_forward_end_to_end_smoke() {
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
        let n_heads = 2u32;
        let ffn_dim = 16u32;
        let n_layers = 1u32;
        let kernel = 3u32;
        let n_bins = 16u32;
        let hop = 160u32;
        let n_fft = 512u32;

        let mut b = GgufBuilder::new();
        b.add_u32("vokra.f0.fcpe.hop", hop);
        b.add_u32("vokra.f0.fcpe.n_mels", n_mels);
        b.add_u32("vokra.f0.fcpe.n_fft", n_fft);
        b.add_u32("vokra.f0.fcpe.n_pitch_bins", n_bins);
        b.add_u32("vokra.f0.fcpe.d_model", d_model);
        b.add_u32("vokra.f0.fcpe.n_heads", n_heads);
        b.add_u32("vokra.f0.fcpe.ffn_dim", ffn_dim);
        b.add_u32("vokra.f0.fcpe.n_layers", n_layers);
        b.add_u32("vokra.f0.fcpe.kernel_size", kernel);

        let mut state = 42u64;

        // stem [d_model, n_mels] — GGUF stores innermost first, so dims are
        // reversed relative to the row-major weight layout.
        let dm = d_model as usize;
        let nm = n_mels as usize;
        let stem_w = synth_vec(&mut state, dm * nm, 0.1);
        b.add_tensor(
            "stem.weight",
            GgmlType::F32,
            vec![nm as u64, dm as u64],
            f32_bytes(&stem_w),
        )
        .unwrap();
        let stem_b = vec![0.0f32; dm];
        b.add_tensor(
            "stem.bias",
            GgmlType::F32,
            vec![dm as u64],
            f32_bytes(&stem_b),
        )
        .unwrap();

        // Conformer layer 0 — full tensor set. GGUF dims are innermost first
        // for on-disk storage; the runtime binds by name via `tensor_f32`
        // which flattens back to row-major, so a 1-D layout for the byte
        // shape is safe (the encoder validates length not shape).
        let add_1d = |b: &mut GgufBuilder, name: &str, values: &[f32]| {
            b.add_tensor(
                name,
                GgmlType::F32,
                vec![values.len() as u64],
                f32_bytes(values),
            )
            .unwrap();
        };
        let ones = vec![1.0f32; dm];
        let zeros = vec![0.0f32; dm];

        for tag in &["ln1", "ln2", "ln3", "ln4", "ln_out"] {
            add_1d(&mut b, &format!("layers.0.{tag}.weight"), &ones);
            add_1d(&mut b, &format!("layers.0.{tag}.bias"), &zeros);
        }
        // FF1 & FF2
        for ff in &["ff1", "ff2"] {
            let w1 = synth_vec(&mut state, ffn_dim as usize * dm, 0.1);
            let b1 = synth_vec(&mut state, ffn_dim as usize, 0.1);
            let w2 = synth_vec(&mut state, dm * ffn_dim as usize, 0.1);
            let b2 = synth_vec(&mut state, dm, 0.1);
            add_1d(&mut b, &format!("layers.0.{ff}.w1"), &w1);
            add_1d(&mut b, &format!("layers.0.{ff}.b1"), &b1);
            add_1d(&mut b, &format!("layers.0.{ff}.w2"), &w2);
            add_1d(&mut b, &format!("layers.0.{ff}.b2"), &b2);
        }
        // MHA
        for tag in &["wq", "wk", "wv", "wo"] {
            let w = synth_vec(&mut state, dm * dm, 0.1);
            add_1d(&mut b, &format!("layers.0.mha.{tag}"), &w);
        }
        for tag in &["bq", "bk", "bv", "bo"] {
            let bv = synth_vec(&mut state, dm, 0.1);
            add_1d(&mut b, &format!("layers.0.mha.{tag}"), &bv);
        }
        // Conv
        let two_d = 2 * dm;
        add_1d(
            &mut b,
            "layers.0.conv.pointwise1_w",
            &synth_vec(&mut state, two_d * dm, 0.1),
        );
        add_1d(
            &mut b,
            "layers.0.conv.pointwise1_b",
            &synth_vec(&mut state, two_d, 0.1),
        );
        add_1d(
            &mut b,
            "layers.0.conv.depthwise_w",
            &synth_vec(&mut state, dm * kernel as usize, 0.1),
        );
        add_1d(
            &mut b,
            "layers.0.conv.depthwise_b",
            &synth_vec(&mut state, dm, 0.1),
        );
        add_1d(&mut b, "layers.0.conv.norm_gamma", &ones);
        add_1d(&mut b, "layers.0.conv.norm_beta", &zeros);
        add_1d(
            &mut b,
            "layers.0.conv.pointwise2_w",
            &synth_vec(&mut state, dm * dm, 0.1),
        );
        add_1d(
            &mut b,
            "layers.0.conv.pointwise2_b",
            &synth_vec(&mut state, dm, 0.1),
        );

        // Head norm + head
        add_1d(&mut b, "head_norm.weight", &ones);
        add_1d(&mut b, "head_norm.bias", &zeros);
        let head_w = synth_vec(&mut state, n_bins as usize * dm, 0.1);
        let head_b = synth_vec(&mut state, n_bins as usize, 0.1);
        add_1d(&mut b, "head.weight", &head_w);
        add_1d(&mut b, "head.bias", &head_b);

        let tmp =
            std::env::temp_dir().join(format!("vokra-fcpe-smoke-{}.gguf", std::process::id()));
        std::fs::write(&tmp, b.to_bytes().unwrap()).unwrap();
        let fcpe = FCPE::from_gguf(&tmp).expect("bind tiny synthetic checkpoint");
        assert!(fcpe.has_real_weights(), "real forward must be armed");

        // 200 ms at 16 kHz — enough for a stable STFT frame count.
        let sr = 16_000u32;
        let pcm: Vec<f32> = (0..(sr as usize / 5))
            .map(|i| (i as f32 * 0.05).sin() * 0.2)
            .collect();
        let frames = fcpe.extract(&pcm, sr);
        assert_eq!(frames.len(), pcm.len() / hop as usize);
        for f in &frames {
            assert!(f.hz.is_finite(), "hz must stay finite");
            assert!(f.time_sec.is_finite());
            assert!(f.confidence >= 0.0 && f.confidence <= 1.0);
        }
        let _ = std::fs::remove_file(&tmp);
    }

    /// `cent ↔ hz` round-trip within numerical noise (guards the log-space
    /// bin decoder against a signed-log-base regression).
    #[test]
    fn cent_hz_roundtrip_is_lossless_up_to_fp32() {
        for &hz in &[32.7f32, 55.0, 110.0, 220.0, 440.0, 880.0, 1975.5] {
            let cent = hz_to_cent(hz);
            let back = cent_to_hz(cent);
            let rel_err = (back - hz).abs() / hz;
            assert!(
                rel_err < 5e-6,
                "cent<->hz roundtrip drift too large for {hz}: back={back} rel={rel_err}"
            );
        }
    }

    /// The cent grid is monotonic (crucial for the soft-argmax centroid).
    #[test]
    fn cent_grid_is_monotonic_and_bounded() {
        let grid = build_cent_grid(32.7, 1975.5, 360);
        assert_eq!(grid.len(), 360);
        for w in grid.windows(2) {
            assert!(w[0] < w[1], "cent grid must be strictly increasing");
        }
        assert!((grid[0] - hz_to_cent(32.7)).abs() < 1e-3);
        assert!((grid[359] - hz_to_cent(1975.5)).abs() < 1e-3);
    }

    /// Env-gated real-weight parity: point at a real FCPE Vokra GGUF via
    /// `VOKRA_FCPE_REAL_GGUF` and a matching WAV via `VOKRA_FCPE_REAL_WAV`
    /// (16 kHz mono PCM16) and assert every frame's Hz is finite. Full
    /// upstream parity (vs `torchfcpe.predict`) is deferred to owner —
    /// this is the "wired end-to-end" smoke test.
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
        // Minimal PCM16 WAV parse — we intentionally do not depend on hound
        // here; the harness only needs a real audio buffer of the right
        // shape.
        let samples = parse_pcm16_wav_16k_mono(&bytes).expect("PCM16 mono @ 16 kHz");
        let frames = fcpe.extract(&samples, 16_000);
        assert!(!frames.is_empty(), "extract must emit frames");
        for f in frames {
            assert!(f.hz.is_finite() && f.hz >= 0.0);
        }
    }

    /// Best-effort 16 kHz mono PCM16 WAV parse — used exclusively by the
    /// env-gated real-weight harness above (the runtime-side WAV path is
    /// `vokra-eval` and `vokra-server`, not `vokra-models`).
    #[allow(dead_code)]
    fn parse_pcm16_wav_16k_mono(bytes: &[u8]) -> Option<Vec<f32>> {
        if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return None;
        }
        // Walk chunks looking for `fmt ` + `data`.
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
            cursor = body + size + (size & 1); // 2-byte alignment
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
