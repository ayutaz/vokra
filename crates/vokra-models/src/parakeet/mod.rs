//! Parakeet **TDT-0.6B-v3** — NVIDIA's FastConformer + Time-Duration
//! Transducer ASR (SoTA plan Phase 2, 2026-07-24).
//!
//! # What Parakeet-TDT-0.6B-v3 is (primary source)
//!
//! Parakeet-TDT-0.6B-v3 is NVIDIA NeMo's 0.6B FastConformer encoder + TDT
//! (Time-Duration Transducer) decoder for streaming ASR. The upstream
//! release ships two identical model files (the raw NeMo `.nemo`
//! checkpoint and the HuggingFace-transformers safetensors); every hparam
//! below is transcribed **verbatim** from
//! `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Model type** (`model_type`): `"parakeet_tdt"`
//!   (`architectures = ["ParakeetForTDT"]`).
//! - **Encoder** (`encoder_config`, `model_type = "parakeet_encoder"`):
//!   FastConformer.
//!   - `hidden_size` = 1024 (aka `d_model`),
//!   - `num_hidden_layers` = 24,
//!   - `num_attention_heads` = 8, `num_key_value_heads` = 8 (**MHA**,
//!     `num_heads_kv == num_heads` → no GQA broadcast),
//!   - `intermediate_size` = 4096 (FFN inner width; the ~4× expansion
//!     factor upstream `ff_expansion_factor=4` implies for `d_model=1024`),
//!   - `conv_kernel_size` = 9 (FastConformer convolution kernel),
//!   - `hidden_act` = `"silu"`,
//!   - `max_position_embeddings` = 5000,
//!   - `num_mel_bins` = 128 (input log-mel channels),
//!   - `subsampling_factor` = 8 (**FastConformer 8× downsampling**),
//!     `subsampling_conv_kernel_size` = 3, `subsampling_conv_stride` = 2,
//!     `subsampling_conv_channels` = 256,
//!   - `attention_bias` = false, `convolution_bias` = false,
//!   - `scale_input` = false, `initializer_range` = 0.02,
//!   - `dropout` / `attention_dropout` / `activation_dropout` / `layerdrop`
//!     / `dropout_positions` (train-time only — inference is dropout-free).
//! - **Decoder** (RNN-T prediction network):
//!   - `decoder_hidden_size` = 640,
//!   - `num_decoder_layers` = 2.
//! - **TDT / joint / vocab**:
//!   - `vocab_size` = 8193 (**8192 pieces + 1 blank**),
//!   - `blank_token_id` = 8192 (blank at the tail of the head — the
//!     NeMo-canonical convention that matches [`vokra_ops::rnnt_decode()`]'s
//!     `blank_id = vocab_size` default),
//!   - `pad_token_id` = 2,
//!   - `durations` = `[0, 1, 2, 3, 4]` (5 TDT duration bins),
//!   - `max_symbols_per_step` = 10 (zero-duration emission cap — NeMo
//!     greedy default; the same value drives
//!     [`vokra_ops::rnnt_decode::RnntAttrs::max_symbols_per_step`]).
//!   - `hidden_act` (top-level, joint post-activation) = `"relu"`.
//! - **Audio boundary**: `sample_rate` = 16 000 (16 kHz mono `.wav` /
//!   `.flac` per the model card — **not** written in `config.json`; the
//!   preprocessor side-car names it).
//! - **Weight license**: **CC-BY 4.0** (`AttributionRequired`) — the
//!   converter stamps the FR-MD-09 attribution text; the compliance
//!   registry maps `parakeet-tdt-0.6b-v3` / `parakeet-tdt` /
//!   `parakeet-tdt-0.6b` to
//!   [`vokra_core::LicenseClass::AttributionRequired`] so the M2-13 gate
//!   passes commercially *and* the FR-MD-09 attribution surface activates.
//!
//! # Runtime boundary
//!
//! [`ParakeetAsr::from_gguf`] strictly validates the official 699 inference
//! tensors (the 24 training-only BatchNorm counters are intentionally absent)
//! and binds the real decoder-side graph. [`ParakeetAsr::tdt_head_step`] runs
//! the encoder projector, embedding, two-layer LSTM prediction network,
//! decoder projector, ReLU join, and combined token/duration head through the
//! shared CPU GEMV kernel. The deterministic [`ParakeetWeights::synthesized`]
//! store remains only for shape and negative-path tests.
//!
//! Full PCM transcription remains loud-partial. The released encoder uses a
//! three-stage depthwise-separable Conv2D subsampler, relative-position
//! attention with separate Q/K/V projections and learned biases, and eval
//! BatchNorm in each convolution module. Those contracts are not equivalent
//! to the older generic stacking/RoPE Conformer scaffold, so this module does
//! not silently substitute it. Once the exact front end and FastConformer
//! encoder land, TDT sequence decoding will reuse [`vokra_ops::rnnt_decode()`]
//! with the released duration bins and blank id.
//!
//! # No ONNX (permanent)
//!
//! Parakeet ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet/`. This module never
//! touches ONNX.

use std::collections::BTreeSet;

use vokra_backend_cpu::kernels;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{LicenseClass, Result, VokraError};

/// `vokra.model.arch` a Parakeet GGUF must carry. Written by
/// `vokra-convert::models::parakeet::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `parakeet-tdt` /
/// `parakeet-tdt-0.6b-v3` / `parakeet-tdt-0.6b` as
/// [`vokra_core::LicenseClass::AttributionRequired`] (CC-BY 4.0 — the
/// M2-13 gate passes commercially *and* the FR-MD-09 attribution surface
/// activates).
pub const EXPECTED_ARCH: &str = "parakeet-tdt";

/// PCM sample rate Parakeet expects. Not written in the upstream
/// `config.json`; taken from the model card (16 kHz mono `.wav` / `.flac`).
pub const PARAKEET_SAMPLE_RATE: u32 = 16_000;

const KEY_SAMPLE_RATE: &str = "vokra.parakeet.sample_rate";
const KEY_ENC_N_LAYER: &str = "vokra.parakeet.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.parakeet.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.parakeet.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.parakeet.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.parakeet.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.parakeet.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.parakeet.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.parakeet.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str = "vokra.parakeet.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.parakeet.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.parakeet.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.parakeet.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.parakeet.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.parakeet.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.parakeet.arch.encoder.scale_input";
const KEY_DEC_N_LAYER: &str = "vokra.parakeet.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.parakeet.arch.decoder.d_model";
const KEY_JOINT_VOCAB_SIZE: &str = "vokra.parakeet.joint.vocab_size";
const KEY_JOINT_BLANK_ID: &str = "vokra.parakeet.joint.blank_token_id";
const KEY_JOINT_PAD_ID: &str = "vokra.parakeet.joint.pad_token_id";
const KEY_JOINT_MAX_SYMBOLS_PER_STEP: &str = "vokra.parakeet.joint.max_symbols_per_step";
const KEY_JOINT_ACT: &str = "vokra.parakeet.joint.hidden_act";
const KEY_N_DURATIONS: &str = "vokra.parakeet.joint.n_durations";
const PREFIX_DURATION: &str = "vokra.parakeet.joint.duration.";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// FastConformer encoder hparams (primary source: `encoder_config` — every
/// field is a verbatim transcription).
///
/// The encoder is a stack of pre-norm Conformer blocks with 8× subsampling
/// on the input (the "Fast" in FastConformer). `d_model` is the residual
/// width; the per-head width is `d_model / num_attention_heads`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetEncoderConfig {
    /// `num_hidden_layers` — 24 for TDT-0.6B-v3.
    pub n_layer: usize,
    /// `hidden_size` — hidden width, 1024.
    pub d_model: usize,
    /// `num_attention_heads` — Q-heads, 8.
    pub n_head: usize,
    /// `num_key_value_heads` — KV-heads; 8 for 0.6B (MHA, no GQA
    /// broadcast). Kept as a field so a hypothetical future GQA flavor is
    /// representable without a new type.
    pub n_head_kv: usize,
    /// `intermediate_size` — FFN inner width, 4096 (the ~4× expansion of
    /// `d_model=1024`).
    pub ffn_dim: usize,
    /// `conv_kernel_size` — FastConformer depthwise convolution kernel
    /// size, 9. Must be odd for symmetric same-padding.
    pub conv_kernel_size: usize,
    /// `num_mel_bins` — log-mel channels on the input, 128.
    pub in_dim: usize,
    /// `subsampling_factor` — 8 (FastConformer 8× downsampling).
    pub subsampling_factor: usize,
    /// `subsampling_conv_kernel_size` — 3.
    pub subsampling_conv_kernel_size: usize,
    /// `subsampling_conv_stride` — 2.
    pub subsampling_conv_stride: usize,
    /// `subsampling_conv_channels` — 256.
    pub subsampling_conv_channels: usize,
    /// `max_position_embeddings` — 5000 (upper bound on the RoPE / relpos
    /// index; a real forward asserts the incoming subsampled sequence
    /// length does not exceed it).
    pub max_position_embeddings: usize,
    /// `attention_bias` — false for 0.6B-v3 (Q/K/V/out projections are
    /// bias-free).
    pub attention_bias: bool,
    /// `convolution_bias` — false for 0.6B-v3 (depthwise + point-wise
    /// convolutions are bias-free).
    pub convolution_bias: bool,
    /// `scale_input` — false (upstream `ParakeetEncoder` skips the input
    /// scale when this is off).
    pub scale_input: bool,
}

impl ParakeetEncoderConfig {
    /// Per-head width (`d_model / n_head`); `0` when `n_head == 0`
    /// (shape-only converter sentinel) so shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_head).unwrap_or(0)
    }

    /// MHA / GQA algebraic constraint: Q-heads divide the width, and
    /// KV-heads divide Q-heads (broadcast). All non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_head != 0
            && self.n_head_kv != 0
            && self.d_model != 0
            && self.d_model % self.n_head == 0
            && self.n_head % self.n_head_kv == 0
    }

    /// KV hidden width, `n_head_kv * head_dim`. Equals `d_model` when
    /// `n_head_kv == n_head` (MHA — the Parakeet 0.6B-v3 case).
    #[must_use]
    pub fn kv_hidden(&self) -> usize {
        self.n_head_kv * self.head_dim()
    }
}

/// RNN-T prediction-network hparams (primary source: `decoder_hidden_size`
/// + `num_decoder_layers`).
///
/// The prediction network is a small LSTM that consumes the previously
/// emitted non-blank tokens and produces the decoder-side hidden state fed
/// into the joint. Upstream Parakeet uses a 2-layer LSTM at 640-d.
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetDecoderConfig {
    /// `num_decoder_layers` — 2.
    pub n_layer: usize,
    /// `decoder_hidden_size` — 640.
    pub d_model: usize,
}

/// Joint / TDT head hparams (primary source: `durations`, `vocab_size`,
/// `blank_token_id`, `max_symbols_per_step`, top-level `hidden_act`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetJointConfig {
    /// `vocab_size` — 8193 (8192 SentencePiece pieces + 1 blank at
    /// index 8192). The vocabulary head therefore has width
    /// `vocab_size` (blank inclusive) — this matches
    /// [`vokra_ops::rnnt_decode::RnntAttrs::vocab_size`]'s "excluding
    /// blank" convention **minus one**: the ops-side field is
    /// `vocab_size - 1 = 8192` when the head width is 8193.
    pub vocab_size: usize,
    /// `blank_token_id` — 8192. Matches `RnntAttrs::greedy(..).blank_id`
    /// = `vocab_size` (NeMo default) when the ops-side `vocab_size` is
    /// `8192`.
    pub blank_token_id: u32,
    /// `pad_token_id` — 2 (SentencePiece pad; never a decoder emission —
    /// tokens are consumed at the prediction-network input).
    pub pad_token_id: u32,
    /// `durations` — TDT duration bins in head-output order,
    /// `[0, 1, 2, 3, 4]`. Zero-duration is a legal emission but repeated
    /// zero-only emissions are capped by [`Self::max_symbols_per_step`].
    pub durations: Vec<u32>,
    /// `max_symbols_per_step` — 10 (NeMo greedy default). Passed straight
    /// into [`vokra_ops::rnnt_decode::RnntAttrs::max_symbols_per_step`].
    pub max_symbols_per_step: usize,
    /// Top-level `hidden_act` — `"relu"` (post-joint activation).
    /// Recorded verbatim; the ops-side [`vokra_ops::rnnt_decode()`] takes
    /// materialised joint log-probs so this is descriptive metadata.
    pub joint_act: String,
}

/// Resolved Parakeet hparam snapshot — every field is transcribed from
/// the upstream `config.json` (module docstring) or from the
/// preprocessor / model card (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetConfig {
    /// FastConformer encoder hparams.
    pub encoder: ParakeetEncoderConfig,
    /// RNN-T prediction-network hparams.
    pub decoder: ParakeetDecoderConfig,
    /// Joint / TDT-head hparams.
    pub joint: ParakeetJointConfig,
    /// PCM sample rate Parakeet expects — 16 000 (from the model card;
    /// **not** written in the upstream `config.json`).
    pub sample_rate: u32,
}

impl ParakeetConfig {
    /// Primary-source Parakeet-TDT-0.6B-v3 config (every value transcribed
    /// from `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`).
    #[must_use]
    pub fn parakeet_tdt_0_6b_v3() -> Self {
        Self {
            encoder: ParakeetEncoderConfig {
                n_layer: 24,
                d_model: 1024,
                n_head: 8,
                n_head_kv: 8,
                ffn_dim: 4096,
                conv_kernel_size: 9,
                in_dim: 128,
                subsampling_factor: 8,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 256,
                max_position_embeddings: 5000,
                attention_bias: false,
                convolution_bias: false,
                scale_input: false,
            },
            decoder: ParakeetDecoderConfig {
                n_layer: 2,
                d_model: 640,
            },
            joint: ParakeetJointConfig {
                vocab_size: 8193,
                blank_token_id: 8192,
                pad_token_id: 2,
                durations: vec![0, 1, 2, 3, 4],
                max_symbols_per_step: 10,
                joint_act: "relu".to_owned(),
            },
            sample_rate: PARAKEET_SAMPLE_RATE,
        }
    }

    /// Reads the complete converter-written metadata contract and rejects any
    /// missing, mistyped, or non-canonical axis. This binder targets the one
    /// audited `nvidia/parakeet-tdt-0.6b-v3` release; accepting a shape-like
    /// sibling here would route different FastConformer weights through the
    /// wrong decoder.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_durations = required_u32(file, KEY_N_DURATIONS)? as usize;
        let mut durations = Vec::with_capacity(n_durations);
        for index in 0..n_durations {
            durations.push(required_u32(file, &format!("{PREFIX_DURATION}{index}"))?);
        }
        let config = Self {
            encoder: ParakeetEncoderConfig {
                n_layer: required_u32(file, KEY_ENC_N_LAYER)? as usize,
                d_model: required_u32(file, KEY_ENC_D_MODEL)? as usize,
                n_head: required_u32(file, KEY_ENC_N_HEAD)? as usize,
                n_head_kv: required_u32(file, KEY_ENC_N_HEAD_KV)? as usize,
                ffn_dim: required_u32(file, KEY_ENC_FFN_DIM)? as usize,
                conv_kernel_size: required_u32(file, KEY_ENC_CONV_KERNEL)? as usize,
                in_dim: required_u32(file, KEY_ENC_IN_DIM)? as usize,
                subsampling_factor: required_u32(file, KEY_ENC_SUBSAMPLING_FACTOR)? as usize,
                subsampling_conv_kernel_size: required_u32(file, KEY_ENC_SUB_CONV_KERNEL)? as usize,
                subsampling_conv_stride: required_u32(file, KEY_ENC_SUB_CONV_STRIDE)? as usize,
                subsampling_conv_channels: required_u32(file, KEY_ENC_SUB_CONV_CHANNELS)? as usize,
                max_position_embeddings: required_u32(file, KEY_ENC_MAX_POS)? as usize,
                attention_bias: required_bool_u32(file, KEY_ENC_ATTN_BIAS)?,
                convolution_bias: required_bool_u32(file, KEY_ENC_CONV_BIAS)?,
                scale_input: required_bool_u32(file, KEY_ENC_SCALE_INPUT)?,
            },
            decoder: ParakeetDecoderConfig {
                n_layer: required_u32(file, KEY_DEC_N_LAYER)? as usize,
                d_model: required_u32(file, KEY_DEC_D_MODEL)? as usize,
            },
            joint: ParakeetJointConfig {
                vocab_size: required_u32(file, KEY_JOINT_VOCAB_SIZE)? as usize,
                blank_token_id: required_u32(file, KEY_JOINT_BLANK_ID)?,
                pad_token_id: required_u32(file, KEY_JOINT_PAD_ID)?,
                durations,
                max_symbols_per_step: required_u32(file, KEY_JOINT_MAX_SYMBOLS_PER_STEP)? as usize,
                joint_act: required_string(file, KEY_JOINT_ACT)?.to_owned(),
            },
            sample_rate: required_u32(file, KEY_SAMPLE_RATE)?,
        };
        config.validate_for_forward().map_err(|error| {
            VokraError::ModelLoad(format!("ParakeetConfig::from_gguf: {error}"))
        })?;
        let canonical = Self::parakeet_tdt_0_6b_v3();
        if config != canonical {
            return Err(VokraError::ModelLoad(format!(
                "ParakeetConfig::from_gguf: metadata axes do not match the audited Parakeet-TDT-0.6B-v3 contract; found {config:?}, expected {canonical:?}"
            )));
        }
        Ok(config)
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA head split, even head_dim, non-empty
    /// duration bins with at least one non-zero, blank at head tail)
    /// mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: ParakeetEncoderConfig {
                n_layer: 2,
                d_model: 16,
                n_head: 4,
                n_head_kv: 4,
                ffn_dim: 32,
                conv_kernel_size: 3,
                in_dim: 8,
                subsampling_factor: 2,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 16,
                max_position_embeddings: 128,
                attention_bias: false,
                convolution_bias: false,
                scale_input: false,
            },
            decoder: ParakeetDecoderConfig {
                n_layer: 1,
                d_model: 8,
            },
            joint: ParakeetJointConfig {
                vocab_size: 5,
                blank_token_id: 4,
                pad_token_id: 0,
                durations: vec![0, 1, 2],
                max_symbols_per_step: 4,
                joint_act: "relu".to_owned(),
            },
            sample_rate: PARAKEET_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        // ---- Encoder ------------------------------------------------------
        if !self.encoder.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: encoder ill-formed \
                 (n_layer={}, d_model={}, n_head={}, n_head_kv={}) — \
                 expected d_model % n_head == 0, n_head % n_head_kv == 0, \
                 all fields > 0",
                self.encoder.n_layer,
                self.encoder.d_model,
                self.encoder.n_head,
                self.encoder.n_head_kv,
            )));
        }
        if self.encoder.n_layer == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: encoder.n_layer must be > 0".to_owned(),
            ));
        }
        if self.encoder.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: encoder head_dim {} must be even \
                 (RoPE / rel-pos pairs)",
                self.encoder.head_dim(),
            )));
        }
        if self.encoder.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: encoder.ffn_dim must be > 0".to_owned(),
            ));
        }
        if self.encoder.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: encoder.in_dim (num_mel_bins) must be > 0".to_owned(),
            ));
        }
        if self.encoder.conv_kernel_size == 0 || self.encoder.conv_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: encoder.conv_kernel_size {} must be odd and > 0 \
                 (Conformer symmetric same-padding)",
                self.encoder.conv_kernel_size,
            )));
        }
        if self.encoder.subsampling_factor == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: encoder.subsampling_factor must be > 0 \
                 (FastConformer subsampling)"
                    .to_owned(),
            ));
        }
        if self.encoder.max_position_embeddings == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: encoder.max_position_embeddings must be > 0".to_owned(),
            ));
        }

        // ---- Decoder ------------------------------------------------------
        if self.decoder.n_layer == 0 || self.decoder.d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: decoder ill-formed \
                 (n_layer={}, d_model={}) — all fields > 0",
                self.decoder.n_layer, self.decoder.d_model,
            )));
        }

        // ---- Joint / TDT --------------------------------------------------
        if self.joint.vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: joint.vocab_size must be > 0".to_owned(),
            ));
        }
        // `blank_token_id` lives inside the vocab head width `[0,
        // vocab_size)`; the NeMo convention puts blank at the tail
        // (`blank = vocab_size - 1` with the head-width form the ops-side
        // `RnntAttrs` uses `vocab_size` for the non-blank width).
        if (self.joint.blank_token_id as usize) >= self.joint.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: blank_token_id={} must be < vocab_size={}",
                self.joint.blank_token_id, self.joint.vocab_size,
            )));
        }
        if (self.joint.pad_token_id as usize) >= self.joint.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-tdt config: pad_token_id={} must be < vocab_size={}",
                self.joint.pad_token_id, self.joint.vocab_size,
            )));
        }
        if self.joint.durations.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: joint.durations must be non-empty \
                 (TDT needs at least one duration bin)"
                    .to_owned(),
            ));
        }
        if self.joint.durations.iter().all(|d| *d == 0) {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: joint.durations must contain at least one \
                 non-zero bin — an all-zero set would deadlock the TDT decoder"
                    .to_owned(),
            ));
        }
        if self.joint.max_symbols_per_step == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: joint.max_symbols_per_step must be > 0 \
                 (zero-duration emission cap — NeMo default 10)"
                    .to_owned(),
            ));
        }
        if self.joint.joint_act.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parakeet-tdt config: joint.joint_act must be non-empty \
                 (upstream `hidden_act` — e.g. \"relu\")"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Vocabulary size **excluding** the blank symbol — the value
    /// [`vokra_ops::rnnt_decode::RnntAttrs::vocab_size`] takes. NeMo puts
    /// the blank at `vocab_size - 1` in the head-width form, so this is
    /// `head_width - 1 = 8192` for TDT-0.6B-v3.
    #[must_use]
    pub fn ops_vocab_size(&self) -> usize {
        self.joint.vocab_size.saturating_sub(1)
    }
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(vokra_core::gguf::GgufMetadataValue::U32(value)) => Ok(*value),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "ParakeetConfig::from_gguf: `{key}` must be u32, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "ParakeetConfig::from_gguf: missing required metadata `{key}`"
        ))),
    }
}

fn required_bool_u32(file: &GgufFile, key: &str) -> Result<bool> {
    match required_u32(file, key)? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(VokraError::ModelLoad(format!(
            "ParakeetConfig::from_gguf: `{key}` must be boolean u32 0/1, found {value}"
        ))),
    }
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "ParakeetConfig::from_gguf: missing or non-string metadata `{key}`"
            ))
        })
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-encoder-block scaffold weights (pre-norm Conformer FF1 / MHA / Conv
/// / FF2 branches).
///
/// Field names mirror the upstream NeMo `ConformerLayer` module names; the
/// real converter will bind them from the upstream tensor manifest under
/// the same names (the CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai
/// STT contract). Shape sizes below are the flat element counts.
#[derive(Debug, Clone)]
pub struct ParakeetEncoderBlockWeights {
    /// FF1 pre-norm γ, shape `[d_model]`.
    pub ff1_norm: Vec<f32>,
    /// FF1 hidden projection, shape `[d_model, ffn_dim]`.
    pub ff1_fc1: Vec<f32>,
    /// FF1 output projection, shape `[ffn_dim, d_model]`.
    pub ff1_fc2: Vec<f32>,
    /// Attention pre-norm γ, shape `[d_model]`.
    pub attn_norm: Vec<f32>,
    /// Fused Q/K/V projection, shape `[d_model, 3*d_model]` (MHA — for a
    /// future GQA flavor the shape would be `[d_model, d_model + 2 *
    /// kv_hidden]`).
    pub qkv_proj: Vec<f32>,
    /// Attention output projection, shape `[d_model, d_model]`.
    pub attn_out: Vec<f32>,
    /// Conv module pre-norm γ, shape `[d_model]`.
    pub conv_norm: Vec<f32>,
    /// Conv module point-wise 1: `[d_model, 2*d_model]` (GLU pre-split).
    pub conv_pw1: Vec<f32>,
    /// Depthwise conv kernel, shape `[d_model, 1, conv_kernel_size]`.
    pub conv_dw: Vec<f32>,
    /// Depthwise LayerNorm γ, shape `[d_model]` (upstream `norm_type =
    /// 'layer_norm'`).
    pub conv_dw_norm: Vec<f32>,
    /// Conv module point-wise 2: `[d_model, d_model]`.
    pub conv_pw2: Vec<f32>,
    /// FF2 pre-norm γ, shape `[d_model]`.
    pub ff2_norm: Vec<f32>,
    /// FF2 hidden projection, shape `[d_model, ffn_dim]`.
    pub ff2_fc1: Vec<f32>,
    /// FF2 output projection, shape `[ffn_dim, d_model]`.
    pub ff2_fc2: Vec<f32>,
    /// Final block LayerNorm γ, shape `[d_model]`.
    pub final_norm: Vec<f32>,
}

/// Subsample stem scaffold weights (a single Linear + optional LayerNorm
/// with `factor = subsampling_factor` — the [`vokra_ops::conformer`]
/// `Stacking` variant). Kept flat so the sizes are trivially checkable.
#[derive(Debug, Clone)]
pub struct ParakeetSubsampleWeights {
    /// `[d_model, factor * in_dim]`.
    pub linear_w: Vec<f32>,
    /// `[d_model]`.
    pub linear_b: Vec<f32>,
}

/// RNN-T prediction network (2-layer LSTM at `d_model=640` for Parakeet
/// TDT 0.6B v3). Scaffold: one flat weight vector per layer with the
/// stacked LSTM shape (`4 * d_dec`, `d_dec + d_dec`) — the runtime
/// forward binds the real gates by walking the same slots.
#[derive(Debug, Clone)]
pub struct ParakeetPredictionNetLayerWeights {
    /// `[4 * d_dec, d_dec + d_dec]` (input + hidden concat → gates).
    pub lstm_w: Vec<f32>,
    /// `[4 * d_dec]`.
    pub lstm_b: Vec<f32>,
}

/// Joint network scaffold (encoder proj + decoder proj → sum → activation
/// → vocab head, plus the TDT duration head).
#[derive(Debug, Clone)]
pub struct ParakeetJointWeights {
    /// `[d_enc, d_joint]`.
    pub enc_proj: Vec<f32>,
    /// `[d_dec, d_joint]`.
    pub dec_proj: Vec<f32>,
    /// `[d_joint, vocab_size]` — vocabulary head (blank inclusive at
    /// index `blank_token_id`).
    pub vocab_head: Vec<f32>,
    /// `[d_joint, durations.len()]` — TDT duration head.
    pub duration_head: Vec<f32>,
}

/// Parakeet weight store: subsample + encoder blocks + prediction net
/// (LSTM) + joint + final norm.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a follow-up
/// (T29-equivalent — tensor-name manifest fetch from the upstream release).
///
/// The joint width defaults to `encoder.d_model` (a common upstream
/// choice); a real config would carry it explicitly.
#[derive(Debug, Clone)]
pub struct ParakeetWeights {
    /// Subsample stem.
    pub subsample: ParakeetSubsampleWeights,
    /// Encoder blocks in order.
    pub encoder_blocks: Vec<ParakeetEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[d_model]`.
    pub encoder_final_norm: Vec<f32>,
    /// Prediction-network embedding, shape `[vocab_size, d_dec]` (blank
    /// row is included even though never a valid input — mirrors NeMo).
    pub pred_embedding: Vec<f32>,
    /// Prediction-network LSTM layers.
    pub pred_lstm_layers: Vec<ParakeetPredictionNetLayerWeights>,
    /// Joint / TDT head.
    pub joint: ParakeetJointWeights,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl ParakeetWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ParakeetConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let enc = &config.encoder;
        let dec = &config.decoder;
        let joint = &config.joint;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let d_dec = dec.d_model;
        let d_joint = d_enc; // conservative default — matches upstream shape
        let vocab = joint.vocab_size;
        let n_dur = joint.durations.len();
        let k = enc.conv_kernel_size;

        // Subsample stem — flat Linear + optional trailing norm (this
        // scaffold uses the plain Stacking projection, matching
        // `ConvSubsampleKind::Stacking`).
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let subsample = ParakeetSubsampleWeights {
            linear_w: xavier(&mut rng, d_enc * projection_in, projection_in, d_enc),
            linear_b: vec![0.0; d_enc],
        };

        // Encoder blocks.
        let mut encoder_blocks = Vec::with_capacity(enc.n_layer);
        for _ in 0..enc.n_layer {
            encoder_blocks.push(ParakeetEncoderBlockWeights {
                ff1_norm: vec![1.0; d_enc],
                ff1_fc1: xavier(&mut rng, d_enc * ffn, d_enc, ffn),
                ff1_fc2: xavier(&mut rng, ffn * d_enc, ffn, d_enc),
                attn_norm: vec![1.0; d_enc],
                qkv_proj: xavier(&mut rng, d_enc * 3 * d_enc, d_enc, 3 * d_enc),
                attn_out: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                conv_norm: vec![1.0; d_enc],
                conv_pw1: xavier(&mut rng, d_enc * 2 * d_enc, d_enc, 2 * d_enc),
                conv_dw: xavier(&mut rng, d_enc * k, k, 1),
                conv_dw_norm: vec![1.0; d_enc],
                conv_pw2: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                ff2_norm: vec![1.0; d_enc],
                ff2_fc1: xavier(&mut rng, d_enc * ffn, d_enc, ffn),
                ff2_fc2: xavier(&mut rng, ffn * d_enc, ffn, d_enc),
                final_norm: vec![1.0; d_enc],
            });
        }
        let encoder_final_norm = vec![1.0; d_enc];

        // Prediction network embedding + LSTM stack.
        let pred_embedding = xavier(&mut rng, vocab * d_dec, vocab, d_dec);
        let mut pred_lstm_layers = Vec::with_capacity(dec.n_layer);
        for _ in 0..dec.n_layer {
            pred_lstm_layers.push(ParakeetPredictionNetLayerWeights {
                lstm_w: xavier(&mut rng, 4 * d_dec * (2 * d_dec), d_dec, 4 * d_dec),
                lstm_b: vec![0.0; 4 * d_dec],
            });
        }

        // Joint / TDT head.
        let joint_w = ParakeetJointWeights {
            enc_proj: xavier(&mut rng, d_enc * d_joint, d_enc, d_joint),
            dec_proj: xavier(&mut rng, d_dec * d_joint, d_dec, d_joint),
            vocab_head: xavier(&mut rng, d_joint * vocab, d_joint, vocab),
            duration_head: xavier(&mut rng, d_joint * n_dur, d_joint, n_dur),
        };

        Ok(Self {
            subsample,
            encoder_blocks,
            encoder_final_norm,
            pred_embedding,
            pred_lstm_layers,
            joint: joint_w,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Map the top 24 bits of the u64 stream to a f32 in [0, 1).
        let raw = (rng.next_u64() >> 40) as u32;
        let u01 = (raw as f32) / ((1u32 << 24) as f32);
        out.push((u01 * 2.0 - 1.0) * a);
    }
    out
}

// ---------------------------------------------------------------------------
// Official real-weight binder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ParakeetBoundLstmLayer {
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
}

/// The decoder-side tensors needed for one official TDT joint step. The full
/// 699-tensor checkpoint is shape-validated before these tensors are decoded;
/// the FastConformer encoder remains a loud partial until its released Conv2D
/// subsampling, relative attention, and BatchNorm path is transcribed.
#[derive(Debug, Clone)]
struct ParakeetBoundHeadWeights {
    tensor_count: usize,
    encoder_projector_w: Vec<f32>,
    encoder_projector_b: Vec<f32>,
    embedding: Vec<f32>,
    lstm: Vec<ParakeetBoundLstmLayer>,
    decoder_projector_w: Vec<f32>,
    decoder_projector_b: Vec<f32>,
    joint_head_w: Vec<f32>,
    joint_head_b: Vec<f32>,
}

#[derive(Debug, Clone)]
enum ParakeetWeightStore {
    Synthesized(ParakeetWeights),
    Bound(ParakeetBoundHeadWeights),
}

fn expected_real_manifest(config: &ParakeetConfig) -> Vec<(String, Vec<usize>)> {
    let enc = &config.encoder;
    let dec = &config.decoder;
    let channels = enc.subsampling_conv_channels;
    let kernel = enc.subsampling_conv_kernel_size;
    let stride = enc.subsampling_conv_stride;
    let mut num_subsample_layers = 0usize;
    let mut factor = enc.subsampling_factor;
    while factor > 1 {
        num_subsample_layers += 1;
        factor /= 2;
    }
    let mut manifest = Vec::with_capacity(699);
    manifest.push((
        "encoder.subsampling.layers.0.weight".to_owned(),
        vec![channels, 1, kernel, kernel],
    ));
    manifest.push((
        "encoder.subsampling.layers.0.bias".to_owned(),
        vec![channels],
    ));
    for stage in 1..num_subsample_layers {
        let depthwise_index = 2 + (stage - 1) * 3;
        let pointwise_index = depthwise_index + 1;
        manifest.push((
            format!("encoder.subsampling.layers.{depthwise_index}.weight"),
            vec![channels, 1, kernel, kernel],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{depthwise_index}.bias"),
            vec![channels],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{pointwise_index}.weight"),
            vec![channels, channels, 1, 1],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{pointwise_index}.bias"),
            vec![channels],
        ));
    }
    let padding = (kernel - 1) / 2;
    let mut out_frequency = enc.in_dim;
    for _ in 0..num_subsample_layers {
        out_frequency = (out_frequency + 2 * padding - kernel) / stride + 1;
    }
    manifest.push((
        "encoder.subsampling.linear.weight".to_owned(),
        vec![enc.d_model, channels * out_frequency],
    ));
    manifest.push((
        "encoder.subsampling.linear.bias".to_owned(),
        vec![enc.d_model],
    ));

    for layer in 0..enc.n_layer {
        let prefix = format!("encoder.layers.{layer}");
        for branch in ["feed_forward1", "feed_forward2"] {
            manifest.push((
                format!("{prefix}.{branch}.linear1.weight"),
                vec![enc.ffn_dim, enc.d_model],
            ));
            manifest.push((
                format!("{prefix}.{branch}.linear2.weight"),
                vec![enc.d_model, enc.ffn_dim],
            ));
        }
        for norm in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ] {
            manifest.push((format!("{prefix}.{norm}.weight"), vec![enc.d_model]));
            manifest.push((format!("{prefix}.{norm}.bias"), vec![enc.d_model]));
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
            manifest.push((
                format!("{prefix}.self_attn.{projection}.weight"),
                vec![enc.d_model, enc.d_model],
            ));
        }
        manifest.push((
            format!("{prefix}.self_attn.bias_u"),
            vec![enc.n_head, enc.head_dim()],
        ));
        manifest.push((
            format!("{prefix}.self_attn.bias_v"),
            vec![enc.n_head, enc.head_dim()],
        ));
        manifest.push((
            format!("{prefix}.conv.pointwise_conv1.weight"),
            vec![2 * enc.d_model, enc.d_model, 1],
        ));
        manifest.push((
            format!("{prefix}.conv.depthwise_conv.weight"),
            vec![enc.d_model, 1, enc.conv_kernel_size],
        ));
        for stat in ["weight", "bias", "running_mean", "running_var"] {
            manifest.push((format!("{prefix}.conv.norm.{stat}"), vec![enc.d_model]));
        }
        manifest.push((
            format!("{prefix}.conv.pointwise_conv2.weight"),
            vec![enc.d_model, enc.d_model, 1],
        ));
    }

    manifest.push((
        "encoder_projector.weight".to_owned(),
        vec![dec.d_model, enc.d_model],
    ));
    manifest.push(("encoder_projector.bias".to_owned(), vec![dec.d_model]));
    manifest.push((
        "decoder.embedding.weight".to_owned(),
        vec![config.joint.vocab_size, dec.d_model],
    ));
    for layer in 0..dec.n_layer {
        manifest.push((
            format!("decoder.lstm.weight_ih_l{layer}"),
            vec![4 * dec.d_model, dec.d_model],
        ));
        manifest.push((
            format!("decoder.lstm.weight_hh_l{layer}"),
            vec![4 * dec.d_model, dec.d_model],
        ));
        manifest.push((
            format!("decoder.lstm.bias_ih_l{layer}"),
            vec![4 * dec.d_model],
        ));
        manifest.push((
            format!("decoder.lstm.bias_hh_l{layer}"),
            vec![4 * dec.d_model],
        ));
    }
    manifest.push((
        "decoder.decoder_projector.weight".to_owned(),
        vec![dec.d_model, dec.d_model],
    ));
    manifest.push((
        "decoder.decoder_projector.bias".to_owned(),
        vec![dec.d_model],
    ));
    let joint_width = config.joint.vocab_size + config.joint.durations.len();
    manifest.push((
        "joint.head.weight".to_owned(),
        vec![joint_width, dec.d_model],
    ));
    manifest.push(("joint.head.bias".to_owned(), vec![joint_width]));
    manifest
}

fn load_bound_head(file: &GgufFile, config: &ParakeetConfig) -> Result<ParakeetBoundHeadWeights> {
    let manifest = expected_real_manifest(config);
    let expected_names: BTreeSet<String> = manifest.iter().map(|(name, _)| name.clone()).collect();
    for (name, expected_shape) in &manifest {
        let info = file.tensor_info(name).ok_or_else(|| {
            VokraError::ModelLoad(format!("Parakeet-TDT: required tensor `{name}` is missing"))
        })?;
        let actual_shape: Vec<usize> = info.dimensions.iter().map(|&dim| dim as usize).collect();
        if &actual_shape != expected_shape {
            return Err(VokraError::ModelLoad(format!(
                "Parakeet-TDT: tensor `{name}` shape {actual_shape:?}, expected {expected_shape:?}"
            )));
        }
    }
    let actual_names: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(4).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(4).collect();
        return Err(VokraError::ModelLoad(format!(
            "Parakeet-TDT: tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len(),
        )));
    }

    let tensor = |name: &str| -> Result<Vec<f32>> {
        file.tensor_f32(name).map_err(|error| {
            VokraError::ModelLoad(format!(
                "Parakeet-TDT: tensor `{name}` decode failed: {error}"
            ))
        })
    };
    let mut lstm = Vec::with_capacity(config.decoder.n_layer);
    for layer in 0..config.decoder.n_layer {
        lstm.push(ParakeetBoundLstmLayer {
            w_ih: tensor(&format!("decoder.lstm.weight_ih_l{layer}"))?,
            w_hh: tensor(&format!("decoder.lstm.weight_hh_l{layer}"))?,
            b_ih: tensor(&format!("decoder.lstm.bias_ih_l{layer}"))?,
            b_hh: tensor(&format!("decoder.lstm.bias_hh_l{layer}"))?,
        });
    }
    Ok(ParakeetBoundHeadWeights {
        tensor_count: manifest.len(),
        encoder_projector_w: tensor("encoder_projector.weight")?,
        encoder_projector_b: tensor("encoder_projector.bias")?,
        embedding: tensor("decoder.embedding.weight")?,
        lstm,
        decoder_projector_w: tensor("decoder.decoder_projector.weight")?,
        decoder_projector_b: tensor("decoder.decoder_projector.bias")?,
        joint_head_w: tensor("joint.head.weight")?,
        joint_head_b: tensor("joint.head.bias")?,
    })
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Parakeet ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`] is
/// the primary PCM → text entry point; until real weights are bound (see
/// the module docstring) it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty transcript).
#[derive(Debug, Clone)]
pub struct ParakeetAsr {
    cfg: ParakeetConfig,
    weights: ParakeetWeightStore,
    weight_license: LicenseClass,
}

impl ParakeetAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (encoder block count, prediction
    /// net LSTM layer count, per-tensor sizes) so a mismatched pair fails
    /// loudly here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: ParakeetConfig, weights: ParakeetWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc = &cfg.encoder;
        let dec = &cfg.decoder;
        let joint = &cfg.joint;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let d_dec = dec.d_model;
        let vocab = joint.vocab_size;
        let n_dur = joint.durations.len();
        let k = enc.conv_kernel_size;
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let d_joint = d_enc;

        // Subsample stem.
        if weights.subsample.linear_w.len() != d_enc * projection_in {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: subsample.linear_w.len()={} != d_model * \
                 (subsampling_factor * in_dim) = {} * {} = {}",
                weights.subsample.linear_w.len(),
                d_enc,
                projection_in,
                d_enc * projection_in,
            )));
        }
        if weights.subsample.linear_b.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: subsample.linear_b.len()={} != d_model={}",
                weights.subsample.linear_b.len(),
                d_enc,
            )));
        }

        // Encoder blocks.
        if weights.encoder_blocks.len() != enc.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: encoder_blocks.len()={} != encoder.n_layer={}",
                weights.encoder_blocks.len(),
                enc.n_layer,
            )));
        }
        for (i, blk) in weights.encoder_blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("ff1_norm", blk.ff1_norm.len(), d_enc),
                ("ff1_fc1", blk.ff1_fc1.len(), d_enc * ffn),
                ("ff1_fc2", blk.ff1_fc2.len(), ffn * d_enc),
                ("attn_norm", blk.attn_norm.len(), d_enc),
                ("qkv_proj", blk.qkv_proj.len(), d_enc * 3 * d_enc),
                ("attn_out", blk.attn_out.len(), d_enc * d_enc),
                ("conv_norm", blk.conv_norm.len(), d_enc),
                ("conv_pw1", blk.conv_pw1.len(), d_enc * 2 * d_enc),
                ("conv_dw", blk.conv_dw.len(), d_enc * k),
                ("conv_dw_norm", blk.conv_dw_norm.len(), d_enc),
                ("conv_pw2", blk.conv_pw2.len(), d_enc * d_enc),
                ("ff2_norm", blk.ff2_norm.len(), d_enc),
                ("ff2_fc1", blk.ff2_fc1.len(), d_enc * ffn),
                ("ff2_fc2", blk.ff2_fc2.len(), ffn * d_enc),
                ("final_norm", blk.final_norm.len(), d_enc),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet weights: encoder block {i} `{name}` \
                         len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.encoder_final_norm.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: encoder_final_norm.len()={} != d_model={}",
                weights.encoder_final_norm.len(),
                d_enc,
            )));
        }

        // Prediction network.
        if weights.pred_embedding.len() != vocab * d_dec {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: pred_embedding.len()={} != vocab_size * d_dec = {} * {} = {}",
                weights.pred_embedding.len(),
                vocab,
                d_dec,
                vocab * d_dec,
            )));
        }
        if weights.pred_lstm_layers.len() != dec.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet weights: pred_lstm_layers.len()={} != decoder.n_layer={}",
                weights.pred_lstm_layers.len(),
                dec.n_layer,
            )));
        }
        for (i, layer) in weights.pred_lstm_layers.iter().enumerate() {
            let expected_w = 4 * d_dec * (2 * d_dec);
            if layer.lstm_w.len() != expected_w {
                return Err(VokraError::InvalidArgument(format!(
                    "parakeet weights: pred_lstm_layers[{i}].lstm_w.len()={} != \
                     4*d_dec*(2*d_dec) = {expected_w}",
                    layer.lstm_w.len(),
                )));
            }
            if layer.lstm_b.len() != 4 * d_dec {
                return Err(VokraError::InvalidArgument(format!(
                    "parakeet weights: pred_lstm_layers[{i}].lstm_b.len()={} != 4*d_dec={}",
                    layer.lstm_b.len(),
                    4 * d_dec,
                )));
            }
        }

        // Joint.
        for (name, len, expected) in [
            (
                "joint.enc_proj",
                weights.joint.enc_proj.len(),
                d_enc * d_joint,
            ),
            (
                "joint.dec_proj",
                weights.joint.dec_proj.len(),
                d_dec * d_joint,
            ),
            (
                "joint.vocab_head",
                weights.joint.vocab_head.len(),
                d_joint * vocab,
            ),
            (
                "joint.duration_head",
                weights.joint.duration_head.len(),
                d_joint * n_dur,
            ),
        ] {
            if len != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "parakeet weights: `{name}` len={len} != {expected}",
                )));
            }
        }

        Ok(Self {
            cfg,
            weights: ParakeetWeightStore::Synthesized(weights),
            weight_license: LicenseClass::Unknown,
        })
    }

    /// Strictly binds the audited official 699-tensor GGUF. The converter's
    /// metadata axes and every tensor name/shape must match; the 24 training-
    /// only BatchNorm counters stripped before conversion are intentionally
    /// absent. Decoder/LSTM/projector/head tensors are decoded for the real
    /// [`Self::tdt_head_step`] numerical consumer.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "ParakeetAsr::from_gguf: missing or non-string `{}`",
                    chunks::KEY_MODEL_ARCH
                ))
            })?;
        if arch != EXPECTED_ARCH {
            return Err(VokraError::ModelLoad(format!(
                "ParakeetAsr::from_gguf: arch {arch:?}, expected {EXPECTED_ARCH:?}; Parakeet-CTC and Parakeet-TDT-1.1B use different heads/axes"
            )));
        }
        let cfg = ParakeetConfig::from_gguf(file)?;
        let weights = load_bound_head(file, &cfg)?;
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            cfg,
            weights: ParakeetWeightStore::Bound(weights),
            weight_license,
        })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ParakeetConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`ParakeetWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        match &self.weights {
            ParakeetWeightStore::Synthesized(weights) => weights.is_synthesized,
            ParakeetWeightStore::Bound(_) => false,
        }
    }

    /// Weight-license class surfaced from GGUF provenance. Missing metadata
    /// remains `Unknown` so the outer compliance gate fails closed.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Exact number of official inference tensors validated by the binder.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        match &self.weights {
            ParakeetWeightStore::Synthesized(_) => 0,
            ParakeetWeightStore::Bound(weights) => weights.tensor_count,
        }
    }

    /// Runs one real zero-state prediction-network + combined TDT-head step.
    ///
    /// `encoder_hidden` is one FastConformer output row before the official
    /// `encoder_projector` (`[1024]` for 0.6B-v3). This independently
    /// executable subgraph proves that the decoder-side real weights are not
    /// merely name-scanned while the PCM/encoder path remains loud-partial.
    pub fn tdt_head_step(&self, encoder_hidden: &[f32], token_id: u32) -> Result<Vec<f32>> {
        let ParakeetWeightStore::Bound(weights) = &self.weights else {
            return Err(VokraError::NotImplemented(
                "ParakeetAsr::tdt_head_step requires a real GGUF-bound checkpoint",
            ));
        };
        if encoder_hidden.len() != self.cfg.encoder.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "ParakeetAsr::tdt_head_step: encoder_hidden len {}, expected {}",
                encoder_hidden.len(),
                self.cfg.encoder.d_model
            )));
        }
        if token_id as usize >= self.cfg.joint.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "ParakeetAsr::tdt_head_step: token_id {token_id} outside 0..{}",
                self.cfg.joint.vocab_size
            )));
        }
        let hidden = self.cfg.decoder.d_model;
        let mut encoder_projected = vec![0.0; hidden];
        linear_into(
            encoder_hidden,
            &weights.encoder_projector_w,
            &weights.encoder_projector_b,
            hidden,
            &mut encoder_projected,
        )?;
        let embed_offset = token_id as usize * hidden;
        let mut decoder = weights.embedding[embed_offset..embed_offset + hidden].to_vec();
        for layer in &weights.lstm {
            decoder = lstm_zero_state_step(&decoder, layer, hidden)?;
        }
        let mut decoder_projected = vec![0.0; hidden];
        linear_into(
            &decoder,
            &weights.decoder_projector_w,
            &weights.decoder_projector_b,
            hidden,
            &mut decoder_projected,
        )?;
        for index in 0..hidden {
            decoder_projected[index] =
                (decoder_projected[index] + encoder_projected[index]).max(0.0);
        }
        let output_dim = weights.joint_head_b.len();
        let mut logits = vec![0.0; output_dim];
        linear_into(
            &decoder_projected,
            &weights.joint_head_w,
            &weights.joint_head_b,
            output_dim,
            &mut logits,
        )?;
        Ok(logits)
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// This is the primary PCM → text entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// text (they would be noise or a hallucinated fixed sequence), so
    /// this returns [`VokraError::NotImplemented`] naming the blocker.
    /// Callers may verify the shape flow through [`ParakeetAsr::new`] +
    /// [`ParakeetWeights::synthesized`]. [`ParakeetAsr::from_gguf`] binds the
    /// real checkpoint and exposes the independently executable decoder-side
    /// subgraph through [`ParakeetAsr::tdt_head_step`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parakeet transcribe: pcm slice is empty".to_owned(),
            ));
        }
        if matches!(&self.weights, ParakeetWeightStore::Synthesized(_)) {
            return Err(VokraError::NotImplemented(
                "parakeet transcribe: this engine holds synthesized weights \
                 (deterministic fixture from ParakeetWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, \
                 not a real transcript. Bind real Parakeet-TDT-0.6B-v3 \
                 weights (CC-BY 4.0, nvidia/parakeet-tdt-0.6b-v3) before \
                 invoking transcribe. The shape flow (config validation, \
                 weight-store construction, PCM boundary check) remains \
                 available through ParakeetAsr::new; the real checkpoint path \
                 is ParakeetAsr::from_gguf.",
            ));
        }
        Err(VokraError::NotImplemented(
            "parakeet transcribe: the strict 699-tensor real-weight binder and \
             decoder/LSTM/combined-TDT-head step are implemented, but the released \
             log-mel front-end → depthwise-separable Conv2D subsampling → \
             relative-position FastConformer encoder (with eval BatchNorm) → \
             RNN-T prediction net → joint → rnnt_decode(Tdt { \
             duration_bins: joint.durations }) → SentencePiece detokenize \
             full path has not landed yet. The generic vokra_ops::conformer \
             stacking/RoPE path is not numerically equivalent and is deliberately \
             not substituted. Follow-up: add the exact official encoder and then \
             drive the already-bound TDT head with blank_id and duration bins.",
        ))
    }
}

fn linear_into(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    kernels::gemv_f32(output_dim, input.len(), weight, input, Some(bias), output)
}

fn lstm_zero_state_step(
    input: &[f32],
    weights: &ParakeetBoundLstmLayer,
    hidden: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(weights.w_hh.len(), 4 * hidden * hidden);
    let bias = weights
        .b_ih
        .iter()
        .zip(&weights.b_hh)
        .map(|(input, recurrent)| input + recurrent)
        .collect::<Vec<_>>();
    let mut gates = vec![0.0f32; 4 * hidden];
    // The recurrent term is exactly zero for this parity consumer's initial
    // state, but the strict binder still validates and loads `w_hh` so it
    // cannot disappear from the checkpoint contract.
    kernels::gemv_f32(
        4 * hidden,
        hidden,
        &weights.w_ih,
        input,
        Some(&bias),
        &mut gates,
    )?;
    let mut output = vec![0.0f32; hidden];
    for index in 0..hidden {
        let input_gate = sigmoid_f32(gates[index]);
        let candidate = gates[2 * hidden + index].tanh();
        let output_gate = sigmoid_f32(gates[3 * hidden + index]);
        let cell = input_gate * candidate;
        output[index] = output_gate * cell.tanh();
    }
    Ok(output)
}

#[inline]
fn sigmoid_f32(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn canonical_metadata_file() -> GgufFile {
        let config = ParakeetConfig::parakeet_tdt_0_6b_v3();
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        builder.add_u32(KEY_SAMPLE_RATE, config.sample_rate);
        builder.add_u32(KEY_ENC_N_LAYER, config.encoder.n_layer as u32);
        builder.add_u32(KEY_ENC_D_MODEL, config.encoder.d_model as u32);
        builder.add_u32(KEY_ENC_N_HEAD, config.encoder.n_head as u32);
        builder.add_u32(KEY_ENC_N_HEAD_KV, config.encoder.n_head_kv as u32);
        builder.add_u32(KEY_ENC_FFN_DIM, config.encoder.ffn_dim as u32);
        builder.add_u32(KEY_ENC_CONV_KERNEL, config.encoder.conv_kernel_size as u32);
        builder.add_u32(KEY_ENC_IN_DIM, config.encoder.in_dim as u32);
        builder.add_u32(
            KEY_ENC_SUBSAMPLING_FACTOR,
            config.encoder.subsampling_factor as u32,
        );
        builder.add_u32(
            KEY_ENC_SUB_CONV_KERNEL,
            config.encoder.subsampling_conv_kernel_size as u32,
        );
        builder.add_u32(
            KEY_ENC_SUB_CONV_STRIDE,
            config.encoder.subsampling_conv_stride as u32,
        );
        builder.add_u32(
            KEY_ENC_SUB_CONV_CHANNELS,
            config.encoder.subsampling_conv_channels as u32,
        );
        builder.add_u32(
            KEY_ENC_MAX_POS,
            config.encoder.max_position_embeddings as u32,
        );
        builder.add_u32(KEY_ENC_ATTN_BIAS, u32::from(config.encoder.attention_bias));
        builder.add_u32(
            KEY_ENC_CONV_BIAS,
            u32::from(config.encoder.convolution_bias),
        );
        builder.add_u32(KEY_ENC_SCALE_INPUT, u32::from(config.encoder.scale_input));
        builder.add_u32(KEY_DEC_N_LAYER, config.decoder.n_layer as u32);
        builder.add_u32(KEY_DEC_D_MODEL, config.decoder.d_model as u32);
        builder.add_u32(KEY_JOINT_VOCAB_SIZE, config.joint.vocab_size as u32);
        builder.add_u32(KEY_JOINT_BLANK_ID, config.joint.blank_token_id);
        builder.add_u32(KEY_JOINT_PAD_ID, config.joint.pad_token_id);
        builder.add_u32(
            KEY_JOINT_MAX_SYMBOLS_PER_STEP,
            config.joint.max_symbols_per_step as u32,
        );
        builder.add_string(KEY_JOINT_ACT, &config.joint.joint_act);
        builder.add_u32(KEY_N_DURATIONS, config.joint.durations.len() as u32);
        for (index, duration) in config.joint.durations.iter().enumerate() {
            builder.add_u32(&format!("{PREFIX_DURATION}{index}"), *duration);
        }
        GgufFile::parse(builder.to_bytes().expect("serialize metadata")).expect("parse metadata")
    }

    /// Every hparam matches the primary source
    /// (`huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`)
    /// verbatim.
    #[test]
    fn parakeet_tdt_0_6b_v3_matches_primary_source_config_json() {
        let c = ParakeetConfig::parakeet_tdt_0_6b_v3();
        // Encoder.
        assert_eq!(c.encoder.n_layer, 24);
        assert_eq!(c.encoder.d_model, 1024);
        assert_eq!(c.encoder.n_head, 8);
        assert_eq!(c.encoder.n_head_kv, 8);
        assert_eq!(c.encoder.ffn_dim, 4096);
        assert_eq!(c.encoder.conv_kernel_size, 9);
        assert_eq!(c.encoder.in_dim, 128);
        assert_eq!(c.encoder.subsampling_factor, 8);
        assert_eq!(c.encoder.subsampling_conv_kernel_size, 3);
        assert_eq!(c.encoder.subsampling_conv_stride, 2);
        assert_eq!(c.encoder.subsampling_conv_channels, 256);
        assert_eq!(c.encoder.max_position_embeddings, 5000);
        assert!(!c.encoder.attention_bias);
        assert!(!c.encoder.convolution_bias);
        assert!(!c.encoder.scale_input);
        // Decoder.
        assert_eq!(c.decoder.n_layer, 2);
        assert_eq!(c.decoder.d_model, 640);
        // Joint / TDT.
        assert_eq!(c.joint.vocab_size, 8193);
        assert_eq!(c.joint.blank_token_id, 8192);
        assert_eq!(c.joint.pad_token_id, 2);
        assert_eq!(c.joint.durations, vec![0, 1, 2, 3, 4]);
        assert_eq!(c.joint.max_symbols_per_step, 10);
        assert_eq!(c.joint.joint_act, "relu");
        // Audio boundary (model card).
        assert_eq!(c.sample_rate, 16_000);
        // Derived.
        assert_eq!(c.encoder.head_dim(), 128);
        assert_eq!(c.encoder.kv_hidden(), 1024); // MHA
        assert_eq!(c.ops_vocab_size(), 8192);
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("parakeet-tdt-0.6b-v3 is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        ParakeetConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_head_split_ill_formed_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.encoder.d_model = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_gqa_broadcast_not_dividing_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.n_head = 6;
        c.encoder.d_model = 24;
        c.encoder.n_head_kv = 4; // 6 % 4 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_layer_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_ffn_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.ffn_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_in_dim_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.in_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_even_conv_kernel_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.conv_kernel_size = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_subsampling_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.subsampling_factor = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_max_positions_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.max_position_embeddings = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_decoder_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.decoder.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut c = ParakeetConfig::tiny_for_tests();
        c.decoder.d_model = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_blank_out_of_range_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.blank_token_id = c.joint.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_pad_out_of_range_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.pad_token_id = c.joint.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_empty_durations_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.durations.clear();
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_all_zero_durations_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.durations = vec![0, 0, 0];
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_max_symbols_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.max_symbols_per_step = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_empty_joint_act_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.joint_act.clear();
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = ParakeetConfig::tiny_for_tests();
        let w1 = ParakeetWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = ParakeetWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.subsample.linear_w, w2.subsample.linear_w);
        assert_eq!(w1.encoder_blocks[0].qkv_proj, w2.encoder_blocks[0].qkv_proj);
        assert_eq!(w1.pred_embedding, w2.pred_embedding);
        assert_eq!(w1.joint.vocab_head, w2.joint.vocab_head);
        assert!(w1.is_synthesized);

        // Shape flow.
        let enc = &c.encoder;
        let dec = &c.decoder;
        let joint = &c.joint;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let d_dec = dec.d_model;
        let d_joint = d_enc;
        let vocab = joint.vocab_size;
        let n_dur = joint.durations.len();
        let k = enc.conv_kernel_size;
        let projection_in = enc.subsampling_factor * enc.in_dim;
        // Subsample.
        assert_eq!(w1.subsample.linear_w.len(), d_enc * projection_in);
        assert_eq!(w1.subsample.linear_b.len(), d_enc);
        // Encoder.
        assert_eq!(w1.encoder_blocks.len(), enc.n_layer);
        for blk in &w1.encoder_blocks {
            assert_eq!(blk.ff1_norm.len(), d_enc);
            assert_eq!(blk.ff1_fc1.len(), d_enc * ffn);
            assert_eq!(blk.ff1_fc2.len(), ffn * d_enc);
            assert_eq!(blk.attn_norm.len(), d_enc);
            assert_eq!(blk.qkv_proj.len(), d_enc * 3 * d_enc);
            assert_eq!(blk.attn_out.len(), d_enc * d_enc);
            assert_eq!(blk.conv_norm.len(), d_enc);
            assert_eq!(blk.conv_pw1.len(), d_enc * 2 * d_enc);
            assert_eq!(blk.conv_dw.len(), d_enc * k);
            assert_eq!(blk.conv_dw_norm.len(), d_enc);
            assert_eq!(blk.conv_pw2.len(), d_enc * d_enc);
            assert_eq!(blk.ff2_norm.len(), d_enc);
            assert_eq!(blk.ff2_fc1.len(), d_enc * ffn);
            assert_eq!(blk.ff2_fc2.len(), ffn * d_enc);
            assert_eq!(blk.final_norm.len(), d_enc);
        }
        assert_eq!(w1.encoder_final_norm.len(), d_enc);
        // Prediction net.
        assert_eq!(w1.pred_embedding.len(), vocab * d_dec);
        assert_eq!(w1.pred_lstm_layers.len(), dec.n_layer);
        for layer in &w1.pred_lstm_layers {
            assert_eq!(layer.lstm_w.len(), 4 * d_dec * (2 * d_dec));
            assert_eq!(layer.lstm_b.len(), 4 * d_dec);
        }
        // Joint.
        assert_eq!(w1.joint.enc_proj.len(), d_enc * d_joint);
        assert_eq!(w1.joint.dec_proj.len(), d_dec * d_joint);
        assert_eq!(w1.joint.vocab_head.len(), d_joint * vocab);
        assert_eq!(w1.joint.duration_head.len(), d_joint * n_dur);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = ParakeetConfig::tiny_for_tests();
        let w_a = ParakeetWeights::synthesized(&c, 1).expect("build a");
        let w_b = ParakeetWeights::synthesized(&c, 2).expect("build b");
        assert_ne!(w_a.subsample.linear_w, w_b.subsample.linear_w);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            ParakeetWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = ParakeetConfig::tiny_for_tests();
        let w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetAsr::new(c.clone(), w).expect("parakeet asr");
        assert_eq!(asr.config().encoder.d_model, c.encoder.d_model);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_tensor_size_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_proj.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_subsample_size_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.subsample.linear_w.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_final_norm_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.encoder_final_norm.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_prediction_embedding_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.pred_embedding.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_prediction_lstm_layer_count_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.pred_lstm_layers.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_prediction_lstm_tensor_size_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.pred_lstm_layers[0].lstm_w.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_joint_vocab_head_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.joint.vocab_head.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_joint_duration_head_mismatch() {
        let c = ParakeetConfig::tiny_for_tests();
        let mut w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        w.joint.duration_head.pop();
        assert!(matches!(
            ParakeetAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = ParakeetConfig::tiny_for_tests();
        let w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetAsr::new(c, w).expect("parakeet asr");
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the synthesized-weight
    /// blocker (FR-EX-08 — never a silent zero-fill / hallucinated
    /// transcript).
    #[test]
    fn transcribe_on_synthesized_weights_is_loud_not_implemented() {
        let c = ParakeetConfig::tiny_for_tests();
        let w = ParakeetWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetAsr::new(c, w).expect("parakeet asr");
        let pcm = vec![0.0f32; 1024];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized-weight blocker: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_parakeet_tdt() {
        assert_eq!(EXPECTED_ARCH, "parakeet-tdt");
    }

    #[test]
    fn sample_rate_matches_model_card_boundary() {
        // 16 kHz — per the model card (`.wav` / `.flac` mono @ 16 kHz).
        assert_eq!(PARAKEET_SAMPLE_RATE, 16_000);
    }

    #[test]
    fn canonical_metadata_round_trips_through_strict_reader() {
        let file = canonical_metadata_file();
        assert_eq!(
            ParakeetConfig::from_gguf(&file).expect("canonical config"),
            ParakeetConfig::parakeet_tdt_0_6b_v3()
        );
    }

    #[test]
    fn official_inference_manifest_is_exactly_699_unique_float_tensors() {
        let manifest = expected_real_manifest(&ParakeetConfig::parakeet_tdt_0_6b_v3());
        let names: BTreeSet<&str> = manifest.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(manifest.len(), 699);
        assert_eq!(names.len(), manifest.len());
        assert!(names.contains("encoder.layers.0.self_attn.bias_u"));
        assert!(names.contains("encoder.layers.23.conv.norm.running_var"));
        assert!(names.contains("joint.head.weight"));
        assert!(
            names
                .iter()
                .all(|name| !name.ends_with("num_batches_tracked")),
            "24 training-only BatchNorm counters are intentionally stripped"
        );
    }

    #[test]
    fn strict_bind_rejects_metadata_only_artifact_at_first_missing_tensor() {
        let file = canonical_metadata_file();
        let error = ParakeetAsr::from_gguf(&file).expect_err("weights are required");
        assert!(
            error
                .to_string()
                .contains("encoder.subsampling.layers.0.weight"),
            "error names first required tensor: {error}"
        );
    }
}
