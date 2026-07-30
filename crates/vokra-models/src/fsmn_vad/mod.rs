//! FSMN-VAD — Feed-forward Sequential Memory Network for voice activity
//! detection (SoTA plan Phase 5 VAD-2; FunASR family,
//! `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, MIT).
//!
//! # Distinct posture from Silero VAD v5
//!
//! Where Silero VAD v5 is a 1:1-preserved dedicated subgraph
//! (FR-LD-06 — its ONNX `If`-branch topology cannot be lowered cleanly),
//! FSMN-VAD is a first-class audio-dialect op stack. Its architecture is
//! a stack of stateless feed-forward + memory blocks over Kaldi fbank +
//! LFR (Low Frame Rate) + CMVN — a natural fit for graph-level ops.
//! The numeric forward lives in `vokra-ops::fsmn_vad`; this module is
//! the model-level veneer (GGUF binding + streaming handle + `VadEngine`
//! trait impl).
//!
//! # Architecture (source of truth: `SPEC.md` in this directory)
//!
//! ```text
//! PCM (16 kHz mono f32)
//!  -> Kaldi fbank (80-d, 25 ms / 10 ms, Povey window, per-frame DC
//!                  removal + pre-emphasis 0.97)                 [t_fbank, 80]
//!  -> LFR frame stacking (lfr_m=5, lfr_n=1)                     [t_lfr, 400]
//!  -> CMVN (global mean / variance normalisation)               [t_lfr, 400]
//!  -> FSMN encoder stack (4 blocks × [ffn + memory + residual]) [t_lfr, 128]
//!  -> output head (Linear(proj_dim -> n_class))                 [t_lfr, 2]
//!  -> softmax over last axis                                    [t_lfr, 2]
//! ```
//!
//! # Real-weight parity posture
//!
//! Real-weight parity against the upstream FunASR Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off recorded
//! 2026-07-30 yousan). This module ships:
//!
//! - the exact tensor / hparam contract the future
//!   `FsmnVadV1::from_gguf` binds against (see [`Self::from_gguf`]);
//! - synthetic-weight structural tests pinning FR-EX-08 (loud errors on
//!   every shape / config-mismatch violation);
//! - a `VadEngine` implementation that carries the FSMN rolling
//!   histories across chunks (mirror of the Silero VAD v5 API surface
//!   so the `stream::open_stream` glue in `vokra-core` sees no
//!   FSMN-vs-Silero asymmetry).
//!
//! Kaldi fbank + LFR + CMVN wiring uses the shared
//! `vokra_ops::kaldi_fbank` op and is scaffolded but does NOT ship a
//! full streaming pipeline yet (the model wrapper accepts pre-computed
//! features via [`FsmnVadV1::forward_features`] for consistency across
//! synthetic + real-weight tests; a full PCM entry-point lands with the
//! real-weight parity CI once the checkpoint is fetched).

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{Result, VokraError};

use vokra_ops::{
    FsmnBlockWeights, FsmnEncoderConfig, FsmnStreamState, FsmnVadWeights, KaldiFbankOpts,
    fsmn_vad_forward, kaldi_fbank, softmax_last_axis,
};

#[cfg(test)]
mod tests;

/// `vokra.model.arch` value for FSMN-VAD GGUFs — distinct from the
/// `"silero-vad"` sibling so the model dispatcher picks the right load
/// path. Silently sharing would misroute the loader.
pub const ARCH: &str = "fsmn-vad";

/// `vokra.model.name` value for the canonical release
/// (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`). Callers who ship a
/// distinct checkpoint override via the converter CLI.
pub const DEFAULT_NAME: &str = "fsmn-vad-zh-cn-16k-common";

/// `vokra.model.category` — VAD family. Same value the `silero_vad`
/// module uses (VAD load-path selector reads this rather than `arch`).
pub const CATEGORY: &str = "vad";

/// Upstream HF repository slug (recorded under
/// `vokra.provenance.upstream_hf`).
pub const UPSTREAM_HF: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";

// ---- metadata keys (`vokra.fsmn_vad.*`) ----------------------------------
//
// Kept as module-level `pub const` (mirror of `wespeaker::KEY_MODEL_CATEGORY`)
// so the converter can reference the exact strings without a chunks::KEY_*
// namespace expansion for an arch-scoped field group.

/// Model-category metadata key (`vokra.model.category`). Written by the
/// converter, read here (round-trip test in the tests submodule).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// Upstream HF slug metadata key (`vokra.provenance.upstream_hf`).
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Number of FSMN encoder blocks.
pub const KEY_N_BLOCKS: &str = "vokra.fsmn_vad.n_blocks";
/// LFR-stacked input width (== `lfr_m * n_mels`).
pub const KEY_INPUT_DIM: &str = "vokra.fsmn_vad.input_dim";
/// FSMN block projection width.
pub const KEY_PROJ_DIM: &str = "vokra.fsmn_vad.proj_dim";
/// FSMN block hidden (ReLU) width.
pub const KEY_HIDDEN_DIM: &str = "vokra.fsmn_vad.hidden_dim";
/// Memory-block left context (past frames).
pub const KEY_LORDER: &str = "vokra.fsmn_vad.lorder";
/// Memory-block right context (future frames; 0 for streaming).
pub const KEY_RORDER: &str = "vokra.fsmn_vad.rorder";
/// Output class count (2 = [silence, speech]).
pub const KEY_N_CLASS: &str = "vokra.fsmn_vad.n_class";
/// Kaldi fbank mel-bin count (per raw frame, pre-LFR).
pub const KEY_N_MELS: &str = "vokra.fsmn_vad.n_mels";
/// LFR stacking window (frames).
pub const KEY_LFR_M: &str = "vokra.fsmn_vad.lfr_m";
/// LFR stride (frames).
pub const KEY_LFR_N: &str = "vokra.fsmn_vad.lfr_n";
/// Sample rate the checkpoint expects (Hz).
pub const KEY_SAMPLE_RATE: &str = "vokra.fsmn_vad.sample_rate";
/// CMVN global mean stats (`Array<F32>` of `input_dim` elements — the
/// FunASR `am.mvn.mean_stats`). Applied per-column as
/// `(x - mean) / sqrt(var + eps)` after LFR stacking.
///
/// Stored as a `vokra.*` **metadata chunk** (not a tensor) because the
/// FunASR release ships the CMVN inside a small `am.mvn` transform file,
/// distinct from the model's `.pt` state-dict; keeping it in metadata
/// preserves that separation and makes the front-end config
/// self-describing at file open time (no tensor lookup needed).
pub const KEY_CMVN_MEAN: &str = "vokra.fsmn_vad.cmvn_mean";
/// CMVN global variance stats (`Array<F32>` of `input_dim` elements —
/// the FunASR `am.mvn.var_stats`). See [`KEY_CMVN_MEAN`] for the
/// storage rationale.
pub const KEY_CMVN_VAR: &str = "vokra.fsmn_vad.cmvn_var";
/// Numerical floor added to variance before the reciprocal-square-root
/// (guard against a zero-variance dim collapsing the front-end).
pub(crate) const CMVN_EPSILON: f32 = 1e-6;

// ---- tensor-name convention -----------------------------------------------
//
// Rows are `[out, in]` (PyTorch convention); the converter emits under the
// same names. Deliberate mirror of the upstream FunASR PyTorch parameter
// naming so `from_gguf` binds against the same identifiers the state-dict
// exposes.

/// Input projection weight `[proj_dim, input_dim]`.
pub const TENSOR_IN_PROJ_WEIGHT: &str = "encoder.in_linear.weight";
/// Input projection bias `[proj_dim]`.
pub const TENSOR_IN_PROJ_BIAS: &str = "encoder.in_linear.bias";
/// Output head weight `[n_class, proj_dim]`.
pub const TENSOR_OUT_WEIGHT: &str = "decoder.dec_dense3.weight";
/// Output head bias `[n_class]`.
pub const TENSOR_OUT_BIAS: &str = "decoder.dec_dense3.bias";

/// Formats the per-block ffn1 weight tensor name (`encoder.<i>.ffn.linear1.weight`).
pub fn tensor_ffn1_weight(block_idx: usize) -> String {
    format!("encoder.{block_idx}.ffn.linear1.weight")
}
/// Formats the per-block ffn1 bias tensor name.
pub fn tensor_ffn1_bias(block_idx: usize) -> String {
    format!("encoder.{block_idx}.ffn.linear1.bias")
}
/// Formats the per-block ffn2 weight tensor name.
pub fn tensor_ffn2_weight(block_idx: usize) -> String {
    format!("encoder.{block_idx}.ffn.linear2.weight")
}
/// Formats the per-block ffn2 bias tensor name.
pub fn tensor_ffn2_bias(block_idx: usize) -> String {
    format!("encoder.{block_idx}.ffn.linear2.bias")
}
/// Formats the per-block depthwise memory-conv weight tensor name.
pub fn tensor_memory_weight(block_idx: usize) -> String {
    format!("encoder.{block_idx}.memory.conv1.weight")
}
/// Formats the per-block depthwise memory-conv bias tensor name.
pub fn tensor_memory_bias(block_idx: usize) -> String {
    format!("encoder.{block_idx}.memory.conv1.bias")
}

/// Full FSMN-VAD checkpoint configuration.
///
/// Every field is transcribed from `vokra.fsmn_vad.*` GGUF metadata by
/// [`FsmnVadV1::from_gguf`]; `0`-sentinels are rejected loudly so a
/// half-populated GGUF cannot silently load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsmnVadConfig {
    /// Encoder-stack geometry (consumed by `vokra_ops::fsmn_vad`).
    pub encoder: FsmnEncoderConfig,
    /// Kaldi fbank mel-bin count (per raw frame, pre-LFR).
    pub n_mels: u32,
    /// LFR stacking window (frames).
    pub lfr_m: u32,
    /// LFR stride (frames).
    pub lfr_n: u32,
    /// Sample rate the checkpoint expects (Hz — upstream: 16 000).
    pub sample_rate: u32,
}

impl FsmnVadConfig {
    /// The upstream default (mirror of
    /// `FsmnEncoderConfig::upstream_default` extended with the fbank +
    /// LFR axes the model-level wrapper cares about).
    pub fn upstream_default() -> Self {
        Self {
            encoder: FsmnEncoderConfig::upstream_default(),
            n_mels: 80,
            lfr_m: 5,
            lfr_n: 1,
            sample_rate: 16_000,
        }
    }

    /// Validates the config loudly (FR-EX-08): `0`-sentinels are
    /// rejected on every hparam, and the LFR-stacked width must match
    /// the encoder's declared `input_dim`.
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        for (label, v) in [
            ("n_mels", self.n_mels),
            ("lfr_m", self.lfr_m),
            ("lfr_n", self.lfr_n),
            ("sample_rate", self.sample_rate),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "fsmn-vad config: {label} must be > 0 (got 0 — the GGUF's \
                     vokra.fsmn_vad.* chunk is missing or malformed)",
                )));
            }
        }
        let expected_input = (self.lfr_m as usize) * (self.n_mels as usize);
        if self.encoder.input_dim != expected_input {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad config: encoder.input_dim ({}) != lfr_m ({}) * n_mels ({}) = {}",
                self.encoder.input_dim, self.lfr_m, self.n_mels, expected_input,
            )));
        }
        Ok(())
    }

    /// Reads config from `vokra.fsmn_vad.*` metadata in a parsed GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| -> Result<u32> {
            let v = gguf.get(k).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!("fsmn-vad GGUF missing required u32 metadata `{k}`"))
            })?;
            u32::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "fsmn-vad GGUF metadata `{k}` = {v} does not fit in u32"
                ))
            })
        };
        let n_blocks = get_u32(KEY_N_BLOCKS)? as usize;
        let input_dim = get_u32(KEY_INPUT_DIM)? as usize;
        let proj_dim = get_u32(KEY_PROJ_DIM)? as usize;
        let hidden_dim = get_u32(KEY_HIDDEN_DIM)? as usize;
        let lorder = get_u32(KEY_LORDER)? as usize;
        let rorder = get_u32(KEY_RORDER)? as usize;
        let n_class = get_u32(KEY_N_CLASS)? as usize;
        let cfg = Self {
            encoder: FsmnEncoderConfig {
                n_blocks,
                input_dim,
                proj_dim,
                hidden_dim,
                lorder,
                rorder,
                n_class,
            },
            n_mels: get_u32(KEY_N_MELS)?,
            lfr_m: get_u32(KEY_LFR_M)?,
            lfr_n: get_u32(KEY_LFR_N)?,
            sample_rate: get_u32(KEY_SAMPLE_RATE)?,
        };
        cfg.validate()
            .map_err(|e| VokraError::ModelLoad(e.to_string()))?;
        Ok(cfg)
    }
}

/// Loads a required `Array<F32>` metadata chunk into an owned `Vec<f32>`,
/// enforcing element count + element-type (FR-EX-08 — refuse the load
/// rather than silently coerce or default). Used by the CMVN
/// `vokra.fsmn_vad.cmvn_{mean,var}` chunks; kept local to this module
/// because no other model needs the shape today.
fn read_f32_array(gguf: &GgufFile, key: &str, expect_len: usize) -> Result<Vec<f32>> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "fsmn-vad GGUF missing required Array<F32> metadata `{key}`",
        ))
    })?;
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "fsmn-vad GGUF metadata `{key}` is not an array (expected Array<F32>)",
        ))
    })?;
    if arr.element_type != GgufValueType::F32 {
        return Err(VokraError::ModelLoad(format!(
            "fsmn-vad GGUF metadata `{key}` has element_type {:?}, expected F32",
            arr.element_type
        )));
    }
    if arr.values.len() != expect_len {
        return Err(VokraError::ModelLoad(format!(
            "fsmn-vad GGUF metadata `{key}` has {} elements, expected {expect_len}",
            arr.values.len()
        )));
    }
    let mut out = Vec::with_capacity(expect_len);
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::F32(x) => out.push(*x),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad GGUF metadata `{key}[{i}]` is not F32 (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    Ok(out)
}

/// Kaldi fbank options for the FunASR FSMN-VAD front-end.
///
/// Mirror of the upstream `WavFrontend` config: 25 ms window, 10 ms hop,
/// per-frame DC removal, pre-emphasis 0.97, Povey window (via `povey()`
/// inside `kaldi_fbank`), snip-edges framing, power spectrum, Kaldi HTK
/// mel over the `20`–`Nyquist` Hz band, log magnitude. **CMN is off**
/// here because the FSMN-VAD front-end normalises with the checkpoint's
/// global CMVN (mean / var stats stored in the `vokra.fsmn_vad.cmvn_*`
/// chunks), not per-utterance mean subtraction — leaving both on would
/// double-subtract and change the distribution the encoder was trained
/// on.
fn fsmn_vad_fbank_opts(sample_rate: u32, num_mel_bins: usize) -> KaldiFbankOpts {
    KaldiFbankOpts {
        sample_rate,
        num_mel_bins,
        // 25 ms / 10 ms at any sample rate: `(sr * ms) / 1000`, matches
        // Kaldi `frame_shift_ms=10`, `frame_length_ms=25`.
        frame_length: ((sample_rate as usize) * 25) / 1000,
        frame_shift: ((sample_rate as usize) * 10) / 1000,
        remove_dc_offset: true,
        preemph_coeff: 0.97,
        low_freq: 20.0,
        high_freq: 0.0,
        use_power: true,
        use_log: true,
        // CMVN handles global normalisation; do NOT do per-utterance CMN
        // here (would double-subtract).
        subtract_mean: false,
        round_to_power_of_two: true,
    }
}

/// FSMN-VAD model — an immutable shareable weight bundle plus the
/// config it was bound against.
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open), then
/// obtain a stateful stream through the [`VadEngine`] trait
/// ([`open_stream`]). All mutable recurrent state lives in the stream
/// handle (mirror of Silero VAD v5, FR-LD-06).
///
/// [`VadEngine`]: vokra_core::engines::VadEngine
/// [`open_stream`]: vokra_core::engines::VadEngine::open_stream
#[derive(Debug)]
pub struct FsmnVadV1 {
    cfg: FsmnVadConfig,
    weights: Arc<FsmnVadWeights>,
    /// CMVN global mean vector (`[input_dim]`, transcribed from the
    /// `vokra.fsmn_vad.cmvn_mean` metadata chunk). Applied per-column
    /// after LFR stacking as `(x - mean) / sqrt(var + eps)`.
    cmvn_mean: Arc<Vec<f32>>,
    /// CMVN global variance vector (`[input_dim]`, transcribed from the
    /// `vokra.fsmn_vad.cmvn_var` metadata chunk).
    cmvn_var: Arc<Vec<f32>>,
}

impl FsmnVadV1 {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Returns [`VokraError::ModelLoad`] if any required
    /// `vokra.fsmn_vad.*` chunk is missing, any documented tensor is
    /// absent, or any tensor has the wrong shape / dtype (FR-EX-08 —
    /// no silent reshape).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Verify the arch tag first so the caller does not see a
        // downstream "missing tensor" when they handed us a silero-vad
        // GGUF by mistake.
        match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "fsmn-vad: GGUF is missing `vokra.model.arch` (converter did not stamp it)"
                        .to_owned(),
                ));
            }
        }

        let cfg = FsmnVadConfig::from_gguf(gguf)?;

        let load_f32 = |name: &str, expect: usize| -> Result<Vec<f32>> {
            let v = gguf.tensor_f32(name).map_err(|e| {
                VokraError::ModelLoad(format!("fsmn-vad: tensor `{name}` load failed: {e}"))
            })?;
            if v.len() != expect {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad: tensor `{name}` has {} elements, expected {expect}",
                    v.len()
                )));
            }
            Ok(v)
        };

        let in_proj_weight = load_f32(
            TENSOR_IN_PROJ_WEIGHT,
            cfg.encoder.proj_dim * cfg.encoder.input_dim,
        )?;
        let in_proj_bias = load_f32(TENSOR_IN_PROJ_BIAS, cfg.encoder.proj_dim)?;
        let out_weight = load_f32(
            TENSOR_OUT_WEIGHT,
            cfg.encoder.n_class * cfg.encoder.proj_dim,
        )?;
        let out_bias = load_f32(TENSOR_OUT_BIAS, cfg.encoder.n_class)?;

        let mut blocks = Vec::with_capacity(cfg.encoder.n_blocks);
        for b in 0..cfg.encoder.n_blocks {
            let ffn1_weight = load_f32(
                &tensor_ffn1_weight(b),
                cfg.encoder.hidden_dim * cfg.encoder.proj_dim,
            )?;
            let ffn1_bias = load_f32(&tensor_ffn1_bias(b), cfg.encoder.hidden_dim)?;
            let ffn2_weight = load_f32(
                &tensor_ffn2_weight(b),
                cfg.encoder.proj_dim * cfg.encoder.hidden_dim,
            )?;
            let ffn2_bias = load_f32(&tensor_ffn2_bias(b), cfg.encoder.proj_dim)?;
            let memory_weight = load_f32(
                &tensor_memory_weight(b),
                cfg.encoder.proj_dim * cfg.encoder.memory_kernel(),
            )?;
            let memory_bias = load_f32(&tensor_memory_bias(b), cfg.encoder.proj_dim)?;
            blocks.push(FsmnBlockWeights {
                ffn1_weight,
                ffn1_bias,
                ffn2_weight,
                ffn2_bias,
                memory_weight,
                memory_bias,
            });
        }

        let weights = FsmnVadWeights {
            in_proj_weight,
            in_proj_bias,
            blocks,
            out_weight,
            out_bias,
        };
        weights
            .validate(&cfg.encoder)
            .map_err(|e| VokraError::ModelLoad(e.to_string()))?;

        // CMVN mean / var are FunASR-standard front-end config; we require
        // them (FR-EX-08 — no silent identity fallback for a load-bearing
        // normalisation) but the converter always emits them (identity for
        // synthetic conversions, per-checkpoint `am.mvn.mean_stats` /
        // `var_stats` for real releases).
        let cmvn_mean = read_f32_array(gguf, KEY_CMVN_MEAN, cfg.encoder.input_dim)?;
        let cmvn_var = read_f32_array(gguf, KEY_CMVN_VAR, cfg.encoder.input_dim)?;
        for (i, &v) in cmvn_var.iter().enumerate() {
            if v < 0.0 || !v.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "fsmn-vad: {KEY_CMVN_VAR}[{i}] = {v} is not a non-negative finite \
                     variance (checkpoint's am.mvn.var_stats malformed?)"
                )));
            }
        }

        Ok(Self {
            cfg,
            weights: Arc::new(weights),
            cmvn_mean: Arc::new(cmvn_mean),
            cmvn_var: Arc::new(cmvn_var),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns the checkpoint's config.
    pub fn config(&self) -> &FsmnVadConfig {
        &self.cfg
    }

    /// Runs the encoder + softmax on `features` (`[t_lfr, input_dim]`,
    /// row-major, already LFR-stacked + CMVN-normalised) starting from
    /// a **fresh zero state** and returns the per-frame probabilities
    /// (`[t_lfr, n_class]`, row-major).
    ///
    /// This is the "one-shot" analogue of Silero VAD v5's
    /// `forward_chunk`; it does NOT carry any prior context. For
    /// streaming (state carried across chunks) use
    /// [`open_stream`](vokra_core::engines::VadEngine::open_stream).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape violation
    /// (`features.len()` not a multiple of `input_dim`, empty input).
    pub fn forward_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        let mut state = FsmnStreamState::zeros(&self.cfg.encoder)?;
        let logits = fsmn_vad_forward(&self.cfg.encoder, &self.weights, features, &mut state)?;
        Ok(softmax_last_axis(&logits, self.cfg.encoder.n_class))
    }
}

impl vokra_core::engines::VadEngine for FsmnVadV1 {
    fn open_stream(&self) -> Box<dyn vokra_core::engines::VadStreamHandle + Send> {
        Box::new(FsmnVadStream::new(
            self.cfg.clone(),
            Arc::clone(&self.weights),
            Arc::clone(&self.cmvn_mean),
            Arc::clone(&self.cmvn_var),
        ))
    }
}

/// Stateful FSMN-VAD stream: the per-block rolling histories cross
/// chunks, and the front-end (Kaldi fbank + LFR + CMVN) holds its own
/// three-layer streaming buffer (raw PCM → fbank frames → LFR frames)
/// so successive `push_pcm` calls produce the same speech-column probs
/// as one whole-utterance call would (feature-level chunk invariance;
/// pinned by `push_pcm_matches_whole_utterance` in the tests).
///
/// The features-in entry-point ([`Self::push_features`]) is still
/// available for parity harnesses that want to inject pre-computed
/// LFR-stacked + CMVN-normalised features, bypassing the front-end.
pub struct FsmnVadStream {
    cfg: FsmnVadConfig,
    weights: Arc<FsmnVadWeights>,
    state: FsmnStreamState,
    /// CMVN mean vector (shared with the parent model — never mutated).
    cmvn_mean: Arc<Vec<f32>>,
    /// CMVN variance vector (shared — never mutated).
    cmvn_var: Arc<Vec<f32>>,
    /// Kaldi fbank options built once from `cfg.sample_rate` /
    /// `cfg.n_mels` so `push_pcm` does not re-derive them per call.
    fbank_opts: KaldiFbankOpts,
    /// Rolling raw-PCM tail: samples the front-end has not yet consumed
    /// into a complete fbank frame. Snip-edges framing means the last
    /// `frame_length - frame_shift` samples of any pending window stay
    /// here until the next call closes a frame.
    pending_pcm: Vec<f32>,
    /// Rolling fbank-frame tail: `n_mels`-wide rows the LFR stack has
    /// not yet consumed into a complete LFR feature. Length is always
    /// a multiple of `n_mels`.
    pending_frames: Vec<f32>,
}

impl FsmnVadStream {
    fn new(
        cfg: FsmnVadConfig,
        weights: Arc<FsmnVadWeights>,
        cmvn_mean: Arc<Vec<f32>>,
        cmvn_var: Arc<Vec<f32>>,
    ) -> Self {
        let state = FsmnStreamState::zeros(&cfg.encoder)
            .expect("cfg was validated at FsmnVadV1::from_gguf time");
        let fbank_opts = fsmn_vad_fbank_opts(cfg.sample_rate, cfg.n_mels as usize);
        Self {
            cfg,
            weights,
            state,
            cmvn_mean,
            cmvn_var,
            fbank_opts,
            pending_pcm: Vec::new(),
            pending_frames: Vec::new(),
        }
    }

    /// Pushes LFR-stacked + CMVN-normalised features (`[n_frames,
    /// input_dim]`, row-major) and returns the per-frame speech-class
    /// probabilities (index 1 of the softmax output — the "speech"
    /// column by upstream convention).
    ///
    /// State (per-block rolling histories) is carried across calls, so
    /// splitting the same features across multiple `push_features`
    /// calls is bit-identical to one whole-utterance call — the
    /// FSMN-streaming contract (verified by
    /// `vokra_ops::fsmn_vad::tests::state_carry_matches_single_chunk`).
    ///
    /// The [`VadStreamHandle::push_pcm`] implementation goes through
    /// this method after the front-end (kaldi fbank + LFR + CMVN)
    /// produces LFR features; parity harnesses that pre-compute the
    /// features can bypass the front-end by calling here directly.
    pub fn push_features(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        let logits = fsmn_vad_forward(&self.cfg.encoder, &self.weights, features, &mut self.state)?;
        let probs = softmax_last_axis(&logits, self.cfg.encoder.n_class);
        // The "speech" class is index 1 by upstream convention
        // (n_class=2: [silence, speech]).
        let n_frames = probs.len() / self.cfg.encoder.n_class;
        let speech_col = 1usize.min(self.cfg.encoder.n_class - 1);
        let mut out = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            out.push(probs[f * self.cfg.encoder.n_class + speech_col]);
        }
        Ok(out)
    }

    /// Consumes as many complete fbank frames as `pending_pcm` allows
    /// (snip-edges framing: a frame must fit entirely) and appends them
    /// to `pending_frames`. Returns the number of new fbank frames
    /// appended (0 if the buffer is still too short).
    fn drain_pcm_into_frames(&mut self) -> Result<usize> {
        let flen = self.fbank_opts.frame_length;
        let fshift = self.fbank_opts.frame_shift;
        if self.pending_pcm.len() < flen || fshift == 0 {
            return Ok(0);
        }
        let (feats, n_new) = kaldi_fbank(&self.pending_pcm, &self.fbank_opts)?;
        if n_new == 0 {
            return Ok(0);
        }
        self.pending_frames.extend_from_slice(&feats);
        // Consume `n_new * fshift` samples from the front; the last
        // `frame_overlap` samples of the final frame stay for the next
        // frame that will start `fshift` samples ahead of it. This is
        // the standing snip-edges streaming contract.
        let consumed = n_new
            .checked_mul(fshift)
            .ok_or_else(|| {
                VokraError::InvalidArgument("fsmn-vad: pcm consumption overflow".into())
            })?
            .min(self.pending_pcm.len());
        self.pending_pcm.drain(..consumed);
        Ok(n_new)
    }

    /// Consumes as many complete LFR features as `pending_frames`
    /// allows (`lfr_m` frames × `n_mels` cols, stride `lfr_n`). Returns
    /// the row-major flat buffer of new LFR features — length is
    /// `n_new * input_dim`.
    fn drain_frames_into_lfr(&mut self) -> Vec<f32> {
        let n_mels = self.cfg.n_mels as usize;
        let lfr_m = self.cfg.lfr_m as usize;
        let lfr_n = self.cfg.lfr_n as usize;
        let input_dim = self.cfg.encoder.input_dim;
        let mut out: Vec<f32> = Vec::new();
        loop {
            let have_frames = self.pending_frames.len() / n_mels;
            if have_frames < lfr_m {
                break;
            }
            // Stack the leading `lfr_m` frames row-by-row: identical
            // memory layout as concatenating the `lfr_m` `n_mels`-wide
            // rows, because `pending_frames` is already row-major.
            let take = lfr_m * n_mels;
            debug_assert_eq!(take, input_dim);
            out.extend_from_slice(&self.pending_frames[..take]);
            // Stride forward by `lfr_n` frames (== `lfr_n * n_mels`
            // elements). If `lfr_n == 0` we would loop forever; the
            // config validator refuses that at load time.
            let drop = lfr_n
                .checked_mul(n_mels)
                .expect("lfr_n * n_mels fits usize (validated at load)")
                .min(self.pending_frames.len());
            self.pending_frames.drain(..drop);
        }
        out
    }

    /// Applies CMVN in place: `(x - mean) / sqrt(var + eps)` per column,
    /// row-major over `[n_new, input_dim]`. `mean` / `var` are the
    /// `[input_dim]`-wide CMVN vectors loaded from the GGUF chunks.
    fn apply_cmvn_in_place(&self, features: &mut [f32]) {
        let input_dim = self.cfg.encoder.input_dim;
        if features.is_empty() || input_dim == 0 {
            return;
        }
        debug_assert_eq!(self.cmvn_mean.len(), input_dim);
        debug_assert_eq!(self.cmvn_var.len(), input_dim);
        // Precompute inv_std once per column (constant across all
        // frames — pulling it out of the inner loop is both faster and
        // clearer than recomputing per element).
        let inv_std: Vec<f32> = self
            .cmvn_var
            .iter()
            .map(|v| 1.0 / (v + CMVN_EPSILON).sqrt())
            .collect();
        for row in features.chunks_exact_mut(input_dim) {
            for ((r, &m), &s) in row
                .iter_mut()
                .zip(self.cmvn_mean.iter())
                .zip(inv_std.iter())
            {
                *r = (*r - m) * s;
            }
        }
    }
}

impl vokra_core::engines::VadStreamHandle for FsmnVadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        // Sample rate is a load-bearing invariant of the CMVN stats
        // (they were fit against a specific rate); refuse mismatched
        // rates loudly rather than resampling silently (FR-EX-08).
        if sample_rate != self.cfg.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "fsmn-vad: sample rate mismatch — pushed {sample_rate} Hz but the model / \
                 CMVN stats were fit for {} Hz (resample upstream, or open a stream on the \
                 matching rate)",
                self.cfg.sample_rate
            )));
        }

        // Front-end pipeline: PCM → fbank (snip-edges streaming) → LFR
        // stacking (streaming) → CMVN → FSMN forward. Every stage
        // buffers its remainder so successive calls produce the same
        // probabilities a single whole-utterance call would.
        self.pending_pcm.extend_from_slice(pcm);
        self.drain_pcm_into_frames()?;
        let mut lfr = self.drain_frames_into_lfr();
        if lfr.is_empty() {
            return Ok(Vec::new());
        }
        self.apply_cmvn_in_place(&mut lfr);
        self.push_features(&lfr)
    }

    fn reset(&mut self) {
        self.state.reset();
        self.pending_pcm.clear();
        self.pending_frames.clear();
    }
}
