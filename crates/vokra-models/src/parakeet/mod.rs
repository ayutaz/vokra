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
//!   - `vocab_size` = 8193 (**8192 BPE pieces + 1 blank**),
//!   - `blank_token_id` = 8192 (blank at the tail of the head — the
//!     NeMo-canonical convention that matches [`vokra_ops::rnnt_decode()`]'s
//!     `blank_id = vocab_size` default),
//!   - `pad_token_id` = 2,
//!   - `eos_token_id` = 3 and `decoder_start_token_id` = 8192 from the
//!     released `generation_config.json`,
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
//! and binds the complete inference graph. The native path implements the
//! released 128-bin log-mel frontend, three-stage depthwise-separable Conv2D
//! subsampler, 24 relative-position FastConformer blocks with eval BatchNorm,
//! recurrent two-layer LSTM prediction state, duration-aware greedy TDT
//! decoding, EOS termination, and the embedded official BPE + Metaspace
//! tokenizer. [`ParakeetAsr::tdt_head_step`] remains an independently testable
//! decoder/head parity seam. The deterministic
//! [`ParakeetWeights::synthesized`] store remains only for shape and
//! negative-path tests.
//!
//! # No ONNX (permanent)
//!
//! Parakeet ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet/`. This module never
//! touches ONNX.

pub(crate) mod tokenizer;

pub use tokenizer::ParakeetTokenizer;

use std::collections::BTreeSet;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::ir::graph::{MelAttrs, Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::rng::SplitMix64;
use vokra_core::{AsrEngine, BackendKind, LicenseClass, Result, Transcription, VokraError};
use vokra_ops::mel::MelFilterbank;
use vokra_ops::stft::stft;

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor, sha256_bytes};

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

/// Complete learned-op registry for the shared FastConformer + TDT route.
/// Two-dimensional subsampling convolutions are lowered to GEMM; the
/// time-domain depthwise Conformer convolution uses grouped Conv1d.
pub const PARAKEET_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupedConv1d,
];

const TDT_1_1B_LABEL: &str = "Parakeet-TDT-1.1B";
const TDT_1_1B_ARCH: &str = "parakeet-tdt-1_1b";
const TDT_1_1B_MODEL_NAME: &str = "parakeet-tdt-1.1b";
const TDT_1_1B_TENSOR_COUNT: usize = 1667;
const TDT_1_1B_MANIFEST_SHA256: [u8; 32] = [
    0x98, 0x80, 0x16, 0xb3, 0xf7, 0xf7, 0x56, 0x2d, 0x9f, 0xd1, 0xf1, 0x79, 0xb6, 0x78, 0x4c, 0x6f,
    0xe6, 0xd2, 0xfd, 0xf0, 0xac, 0xdb, 0xf3, 0x18, 0x4e, 0x44, 0x28, 0x68, 0x7c, 0xa1, 0x39, 0xf5,
];
const TDT_1_1B_TOKENIZER_VOCAB_SHA256: [u8; 32] = [
    0xdc, 0x8f, 0x48, 0x90, 0x9c, 0x2d, 0x3a, 0x03, 0x74, 0xf4, 0x5b, 0x74, 0x78, 0x22, 0x6d, 0x26,
    0xa7, 0xde, 0x16, 0xbb, 0xc5, 0x33, 0x44, 0x48, 0xa8, 0xe9, 0x89, 0xf4, 0x53, 0x83, 0x84, 0xd1,
];
const TDT_1_1B_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: TDT_1_1B_LABEL,
    arch: TDT_1_1B_ARCH,
    model_name: TDT_1_1B_MODEL_NAME,
    model_name_alias: None,
    tensor_count: TDT_1_1B_TENSOR_COUNT,
    manifest_sha256: TDT_1_1B_MANIFEST_SHA256,
};

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
const KEY_JOINT_EOS_ID: &str = "vokra.parakeet.joint.eos_token_id";
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
    /// `vocab_size` — 8193 (8192 BPE pieces + 1 blank at
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
    /// `pad_token_id` — 2 (tokenizer pad; never a decoder emission —
    /// tokens are consumed at the prediction-network input).
    pub pad_token_id: u32,
    /// Optional end-of-sequence token. The 0.6B-v3 Hugging Face release
    /// declares id 3; the original 1.1B NeMo release has no EOS token and
    /// therefore uses `None` instead of inventing a sentinel id.
    pub eos_token_id: Option<u32>,
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
                eos_token_id: Some(3),
                durations: vec![0, 1, 2, 3, 4],
                max_symbols_per_step: 10,
                joint_act: "relu".to_owned(),
            },
            sample_rate: PARAKEET_SAMPLE_RATE,
        }
    }

    /// Primary-source Parakeet-TDT-1.1B config from the immutable
    /// `nvidia/parakeet-tdt-1.1b` NeMo archive. Unlike the newer 0.6B-v3
    /// release this checkpoint uses 80 mel bins, 42 biased encoder blocks,
    /// a 1,024-piece SentencePiece vocabulary plus a tail blank, and no EOS
    /// token.
    #[must_use]
    pub fn parakeet_tdt_1_1b() -> Self {
        Self {
            encoder: ParakeetEncoderConfig {
                n_layer: 42,
                d_model: 1024,
                n_head: 8,
                n_head_kv: 8,
                ffn_dim: 4096,
                conv_kernel_size: 9,
                in_dim: 80,
                subsampling_factor: 8,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 256,
                max_position_embeddings: 5000,
                attention_bias: true,
                convolution_bias: true,
                scale_input: false,
            },
            decoder: ParakeetDecoderConfig {
                n_layer: 2,
                d_model: 640,
            },
            joint: ParakeetJointConfig {
                vocab_size: 1025,
                blank_token_id: 1024,
                pad_token_id: 1024,
                eos_token_id: None,
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
                // Older converter output predates the explicit generation
                // metadata but targets this same audited checkpoint.
                eos_token_id: Some(optional_u32(file, KEY_JOINT_EOS_ID)?.unwrap_or(3)),
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
                eos_token_id: Some(3),
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
        if let Some(eos_token_id) = self.joint.eos_token_id {
            if (eos_token_id as usize) >= self.joint.vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "parakeet-tdt config: eos_token_id={eos_token_id} must be < vocab_size={}",
                    self.joint.vocab_size,
                )));
            }
            if eos_token_id == self.joint.blank_token_id {
                return Err(VokraError::InvalidArgument(
                    "parakeet-tdt config: eos_token_id must differ from blank_token_id".to_owned(),
                ));
            }
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

fn optional_u32(file: &GgufFile, key: &str) -> Result<Option<u32>> {
    match file.get(key) {
        Some(vokra_core::gguf::GgufMetadataValue::U32(value)) => Ok(Some(*value)),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "ParakeetConfig::from_gguf: `{key}` must be u32, found {other:?}"
        ))),
        None => Ok(None),
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
pub(crate) struct ParakeetBoundLstmLayer {
    pub(crate) w_ih: Vec<f32>,
    pub(crate) w_hh: Vec<f32>,
    pub(crate) b_ih: Vec<f32>,
    pub(crate) b_hh: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetBoundSubsampling {
    pub(crate) conv0_w: Vec<f32>,
    pub(crate) conv0_b: Vec<f32>,
    pub(crate) depthwise_w: [Vec<f32>; 2],
    pub(crate) depthwise_b: [Vec<f32>; 2],
    /// `[in_channels, out_channels]`, transposed once at bind time for GEMM.
    pub(crate) pointwise_w_t: [Vec<f32>; 2],
    pub(crate) pointwise_b: [Vec<f32>; 2],
    /// `[channels * frequency, d_model]`, transposed once at bind time.
    pub(crate) linear_w_t: Vec<f32>,
    pub(crate) linear_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetBoundNorm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

/// Normalization contract inside the FastConformer convolution module.
///
/// Parakeet-TDT/CTC use inference BatchNorm after a symmetric depthwise
/// convolution. Nemotron 3.5 ASR uses per-frame LayerNorm after a causal
/// depthwise convolution. Keeping the variants in the bound weight type makes
/// it impossible to accidentally run one checkpoint through the other's
/// statistics contract.
#[derive(Debug, Clone)]
pub(crate) enum FastConformerConvNorm {
    BatchNorm {
        weight: Vec<f32>,
        bias: Vec<f32>,
        running_mean: Vec<f32>,
        running_var: Vec<f32>,
    },
    LayerNorm(ParakeetBoundNorm),
}

#[derive(Debug, Clone)]
pub(crate) struct ParakeetBoundEncoderBlock {
    pub(crate) ff1_w1_t: Vec<f32>,
    pub(crate) ff1_b1: Option<Vec<f32>>,
    pub(crate) ff1_w2_t: Vec<f32>,
    pub(crate) ff1_b2: Option<Vec<f32>>,
    pub(crate) ff2_w1_t: Vec<f32>,
    pub(crate) ff2_b1: Option<Vec<f32>>,
    pub(crate) ff2_w2_t: Vec<f32>,
    pub(crate) ff2_b2: Option<Vec<f32>>,
    pub(crate) norm_ff1: ParakeetBoundNorm,
    pub(crate) norm_attn: ParakeetBoundNorm,
    pub(crate) norm_conv: ParakeetBoundNorm,
    pub(crate) norm_ff2: ParakeetBoundNorm,
    pub(crate) norm_out: ParakeetBoundNorm,
    pub(crate) q_w_t: Vec<f32>,
    pub(crate) q_b: Option<Vec<f32>>,
    pub(crate) k_w_t: Vec<f32>,
    pub(crate) k_b: Option<Vec<f32>>,
    pub(crate) v_w_t: Vec<f32>,
    pub(crate) v_b: Option<Vec<f32>>,
    pub(crate) o_w_t: Vec<f32>,
    pub(crate) o_b: Option<Vec<f32>>,
    pub(crate) relative_k_w_t: Vec<f32>,
    pub(crate) bias_u: Vec<f32>,
    pub(crate) bias_v: Vec<f32>,
    pub(crate) conv_pw1_w_t: Vec<f32>,
    pub(crate) conv_pw1_b: Option<Vec<f32>>,
    pub(crate) conv_dw_w: Vec<f32>,
    pub(crate) conv_dw_b: Option<Vec<f32>>,
    pub(crate) conv_inner_norm: FastConformerConvNorm,
    pub(crate) conv_pw2_w_t: Vec<f32>,
    pub(crate) conv_pw2_b: Option<Vec<f32>>,
}

/// Bound tensors for the released Conv2D-subsampler + relative-position
/// FastConformer + TDT topology. `tensor_count` records the complete
/// authenticated checkpoint manifest; a release may also carry immutable
/// frontend buffers whose values are reproduced by the shared frontend.
#[derive(Debug, Clone)]
struct ParakeetBoundWeights {
    tensor_count: usize,
    subsampling: ParakeetBoundSubsampling,
    encoder: Vec<ParakeetBoundEncoderBlock>,
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
    Synthesized(Box<ParakeetWeights>),
    Bound(Box<ParakeetBoundWeights>),
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

pub(crate) fn transpose_out_in(weight: Vec<f32>, output: usize, input: usize) -> Vec<f32> {
    debug_assert_eq!(weight.len(), output * input);
    let mut transposed = vec![0.0; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}

fn load_bound_weights(file: &GgufFile, config: &ParakeetConfig) -> Result<ParakeetBoundWeights> {
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
    let enc = &config.encoder;
    let channels = enc.subsampling_conv_channels;
    let kernel = enc.subsampling_conv_kernel_size;
    let mut out_frequency = enc.in_dim;
    for _ in 0..3 {
        out_frequency =
            (out_frequency + 2 * ((kernel - 1) / 2) - kernel) / enc.subsampling_conv_stride + 1;
    }
    let subsampling = ParakeetBoundSubsampling {
        conv0_w: tensor("encoder.subsampling.layers.0.weight")?,
        conv0_b: tensor("encoder.subsampling.layers.0.bias")?,
        depthwise_w: [
            tensor("encoder.subsampling.layers.2.weight")?,
            tensor("encoder.subsampling.layers.5.weight")?,
        ],
        depthwise_b: [
            tensor("encoder.subsampling.layers.2.bias")?,
            tensor("encoder.subsampling.layers.5.bias")?,
        ],
        pointwise_w_t: [
            transpose_out_in(
                tensor("encoder.subsampling.layers.3.weight")?,
                channels,
                channels,
            ),
            transpose_out_in(
                tensor("encoder.subsampling.layers.6.weight")?,
                channels,
                channels,
            ),
        ],
        pointwise_b: [
            tensor("encoder.subsampling.layers.3.bias")?,
            tensor("encoder.subsampling.layers.6.bias")?,
        ],
        linear_w_t: transpose_out_in(
            tensor("encoder.subsampling.linear.weight")?,
            enc.d_model,
            channels * out_frequency,
        ),
        linear_b: tensor("encoder.subsampling.linear.bias")?,
    };

    let norm = |prefix: &str, name: &str| -> Result<ParakeetBoundNorm> {
        Ok(ParakeetBoundNorm {
            weight: tensor(&format!("{prefix}.{name}.weight"))?,
            bias: tensor(&format!("{prefix}.{name}.bias"))?,
        })
    };
    let mut encoder = Vec::with_capacity(enc.n_layer);
    for layer in 0..enc.n_layer {
        let prefix = format!("encoder.layers.{layer}");
        let ff = |branch: &str, linear: usize, output: usize, input: usize| {
            tensor(&format!("{prefix}.{branch}.linear{linear}.weight"))
                .map(|weight| transpose_out_in(weight, output, input))
        };
        let projection = |name: &str| {
            tensor(&format!("{prefix}.self_attn.{name}.weight"))
                .map(|weight| transpose_out_in(weight, enc.d_model, enc.d_model))
        };
        encoder.push(ParakeetBoundEncoderBlock {
            ff1_w1_t: ff("feed_forward1", 1, enc.ffn_dim, enc.d_model)?,
            ff1_b1: None,
            ff1_w2_t: ff("feed_forward1", 2, enc.d_model, enc.ffn_dim)?,
            ff1_b2: None,
            ff2_w1_t: ff("feed_forward2", 1, enc.ffn_dim, enc.d_model)?,
            ff2_b1: None,
            ff2_w2_t: ff("feed_forward2", 2, enc.d_model, enc.ffn_dim)?,
            ff2_b2: None,
            norm_ff1: norm(&prefix, "norm_feed_forward1")?,
            norm_attn: norm(&prefix, "norm_self_att")?,
            norm_conv: norm(&prefix, "norm_conv")?,
            norm_ff2: norm(&prefix, "norm_feed_forward2")?,
            norm_out: norm(&prefix, "norm_out")?,
            q_w_t: projection("q_proj")?,
            q_b: None,
            k_w_t: projection("k_proj")?,
            k_b: None,
            v_w_t: projection("v_proj")?,
            v_b: None,
            o_w_t: projection("o_proj")?,
            o_b: None,
            relative_k_w_t: projection("relative_k_proj")?,
            bias_u: tensor(&format!("{prefix}.self_attn.bias_u"))?,
            bias_v: tensor(&format!("{prefix}.self_attn.bias_v"))?,
            conv_pw1_w_t: transpose_out_in(
                tensor(&format!("{prefix}.conv.pointwise_conv1.weight"))?,
                2 * enc.d_model,
                enc.d_model,
            ),
            conv_pw1_b: None,
            conv_dw_w: tensor(&format!("{prefix}.conv.depthwise_conv.weight"))?,
            conv_dw_b: None,
            conv_inner_norm: FastConformerConvNorm::BatchNorm {
                weight: tensor(&format!("{prefix}.conv.norm.weight"))?,
                bias: tensor(&format!("{prefix}.conv.norm.bias"))?,
                running_mean: tensor(&format!("{prefix}.conv.norm.running_mean"))?,
                running_var: tensor(&format!("{prefix}.conv.norm.running_var"))?,
            },
            conv_pw2_w_t: transpose_out_in(
                tensor(&format!("{prefix}.conv.pointwise_conv2.weight"))?,
                enc.d_model,
                enc.d_model,
            ),
            conv_pw2_b: None,
        });
    }

    let mut lstm = Vec::with_capacity(config.decoder.n_layer);
    for layer in 0..config.decoder.n_layer {
        lstm.push(ParakeetBoundLstmLayer {
            w_ih: tensor(&format!("decoder.lstm.weight_ih_l{layer}"))?,
            w_hh: tensor(&format!("decoder.lstm.weight_hh_l{layer}"))?,
            b_ih: tensor(&format!("decoder.lstm.bias_ih_l{layer}"))?,
            b_hh: tensor(&format!("decoder.lstm.bias_hh_l{layer}"))?,
        });
    }
    Ok(ParakeetBoundWeights {
        tensor_count: manifest.len(),
        subsampling,
        encoder,
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

fn load_tdt_1_1b_bound_weights(
    file: &GgufFile,
    config: &ParakeetConfig,
) -> Result<ParakeetBoundWeights> {
    let enc = &config.encoder;
    let dec = &config.decoder;
    let tensor = |name: &str, shape: &[usize]| load_tensor(file, TDT_1_1B_LABEL, name, shape);
    let channels = enc.subsampling_conv_channels;
    let kernel = enc.subsampling_conv_kernel_size;
    let out_frequency = enc.in_dim / enc.subsampling_factor;
    let subsampling = ParakeetBoundSubsampling {
        conv0_w: tensor(
            "encoder.pre_encode.conv.0.weight",
            &[channels, 1, kernel, kernel],
        )?,
        conv0_b: tensor("encoder.pre_encode.conv.0.bias", &[channels])?,
        depthwise_w: [
            tensor(
                "encoder.pre_encode.conv.2.weight",
                &[channels, 1, kernel, kernel],
            )?,
            tensor(
                "encoder.pre_encode.conv.5.weight",
                &[channels, 1, kernel, kernel],
            )?,
        ],
        depthwise_b: [
            tensor("encoder.pre_encode.conv.2.bias", &[channels])?,
            tensor("encoder.pre_encode.conv.5.bias", &[channels])?,
        ],
        pointwise_w_t: [
            transpose_out_in(
                tensor(
                    "encoder.pre_encode.conv.3.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
            transpose_out_in(
                tensor(
                    "encoder.pre_encode.conv.6.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
        ],
        pointwise_b: [
            tensor("encoder.pre_encode.conv.3.bias", &[channels])?,
            tensor("encoder.pre_encode.conv.6.bias", &[channels])?,
        ],
        linear_w_t: transpose_out_in(
            tensor(
                "encoder.pre_encode.out.weight",
                &[enc.d_model, channels * out_frequency],
            )?,
            enc.d_model,
            channels * out_frequency,
        ),
        linear_b: tensor("encoder.pre_encode.out.bias", &[enc.d_model])?,
    };

    let mut encoder = Vec::with_capacity(enc.n_layer);
    for layer in 0..enc.n_layer {
        let prefix = format!("encoder.layers.{layer}");
        let norm = |name: &str| -> Result<ParakeetBoundNorm> {
            Ok(ParakeetBoundNorm {
                weight: tensor(&format!("{prefix}.{name}.weight"), &[enc.d_model])?,
                bias: tensor(&format!("{prefix}.{name}.bias"), &[enc.d_model])?,
            })
        };
        let ff_weight = |branch: &str, linear: usize, output: usize, input: usize| {
            tensor(
                &format!("{prefix}.{branch}.linear{linear}.weight"),
                &[output, input],
            )
            .map(|weight| transpose_out_in(weight, output, input))
        };
        let ff_bias = |branch: &str, linear: usize, output: usize| {
            tensor(&format!("{prefix}.{branch}.linear{linear}.bias"), &[output]).map(Some)
        };
        let attention_weight = |name: &str| {
            tensor(
                &format!("{prefix}.self_attn.{name}.weight"),
                &[enc.d_model, enc.d_model],
            )
            .map(|weight| transpose_out_in(weight, enc.d_model, enc.d_model))
        };
        let attention_bias = |name: &str| {
            tensor(&format!("{prefix}.self_attn.{name}.bias"), &[enc.d_model]).map(Some)
        };
        encoder.push(ParakeetBoundEncoderBlock {
            ff1_w1_t: ff_weight("feed_forward1", 1, enc.ffn_dim, enc.d_model)?,
            ff1_b1: ff_bias("feed_forward1", 1, enc.ffn_dim)?,
            ff1_w2_t: ff_weight("feed_forward1", 2, enc.d_model, enc.ffn_dim)?,
            ff1_b2: ff_bias("feed_forward1", 2, enc.d_model)?,
            ff2_w1_t: ff_weight("feed_forward2", 1, enc.ffn_dim, enc.d_model)?,
            ff2_b1: ff_bias("feed_forward2", 1, enc.ffn_dim)?,
            ff2_w2_t: ff_weight("feed_forward2", 2, enc.d_model, enc.ffn_dim)?,
            ff2_b2: ff_bias("feed_forward2", 2, enc.d_model)?,
            norm_ff1: norm("norm_feed_forward1")?,
            norm_attn: norm("norm_self_att")?,
            norm_conv: norm("norm_conv")?,
            norm_ff2: norm("norm_feed_forward2")?,
            norm_out: norm("norm_out")?,
            q_w_t: attention_weight("linear_q")?,
            q_b: attention_bias("linear_q")?,
            k_w_t: attention_weight("linear_k")?,
            k_b: attention_bias("linear_k")?,
            v_w_t: attention_weight("linear_v")?,
            v_b: attention_bias("linear_v")?,
            o_w_t: attention_weight("linear_out")?,
            o_b: attention_bias("linear_out")?,
            relative_k_w_t: attention_weight("linear_pos")?,
            bias_u: tensor(
                &format!("{prefix}.self_attn.pos_bias_u"),
                &[enc.n_head, enc.head_dim()],
            )?,
            bias_v: tensor(
                &format!("{prefix}.self_attn.pos_bias_v"),
                &[enc.n_head, enc.head_dim()],
            )?,
            conv_pw1_w_t: transpose_out_in(
                tensor(
                    &format!("{prefix}.conv.pointwise_conv1.weight"),
                    &[2 * enc.d_model, enc.d_model, 1],
                )?,
                2 * enc.d_model,
                enc.d_model,
            ),
            conv_pw1_b: Some(tensor(
                &format!("{prefix}.conv.pointwise_conv1.bias"),
                &[2 * enc.d_model],
            )?),
            conv_dw_w: tensor(
                &format!("{prefix}.conv.depthwise_conv.weight"),
                &[enc.d_model, 1, enc.conv_kernel_size],
            )?,
            conv_dw_b: Some(tensor(
                &format!("{prefix}.conv.depthwise_conv.bias"),
                &[enc.d_model],
            )?),
            conv_inner_norm: FastConformerConvNorm::BatchNorm {
                weight: tensor(&format!("{prefix}.conv.batch_norm.weight"), &[enc.d_model])?,
                bias: tensor(&format!("{prefix}.conv.batch_norm.bias"), &[enc.d_model])?,
                running_mean: tensor(
                    &format!("{prefix}.conv.batch_norm.running_mean"),
                    &[enc.d_model],
                )?,
                running_var: tensor(
                    &format!("{prefix}.conv.batch_norm.running_var"),
                    &[enc.d_model],
                )?,
            },
            conv_pw2_w_t: transpose_out_in(
                tensor(
                    &format!("{prefix}.conv.pointwise_conv2.weight"),
                    &[enc.d_model, enc.d_model, 1],
                )?,
                enc.d_model,
                enc.d_model,
            ),
            conv_pw2_b: Some(tensor(
                &format!("{prefix}.conv.pointwise_conv2.bias"),
                &[enc.d_model],
            )?),
        });
    }

    let mut lstm = Vec::with_capacity(dec.n_layer);
    for layer in 0..dec.n_layer {
        let prefix = format!("decoder.prediction.dec_rnn.lstm");
        lstm.push(ParakeetBoundLstmLayer {
            w_ih: tensor(
                &format!("{prefix}.weight_ih_l{layer}"),
                &[4 * dec.d_model, dec.d_model],
            )?,
            w_hh: tensor(
                &format!("{prefix}.weight_hh_l{layer}"),
                &[4 * dec.d_model, dec.d_model],
            )?,
            b_ih: tensor(&format!("{prefix}.bias_ih_l{layer}"), &[4 * dec.d_model])?,
            b_hh: tensor(&format!("{prefix}.bias_hh_l{layer}"), &[4 * dec.d_model])?,
        });
    }

    Ok(ParakeetBoundWeights {
        tensor_count: TDT_1_1B_TENSOR_COUNT,
        subsampling,
        encoder,
        encoder_projector_w: tensor("joint.enc.weight", &[dec.d_model, enc.d_model])?,
        encoder_projector_b: tensor("joint.enc.bias", &[dec.d_model])?,
        embedding: tensor(
            "decoder.prediction.embed.weight",
            &[config.joint.vocab_size, dec.d_model],
        )?,
        lstm,
        decoder_projector_w: tensor("joint.pred.weight", &[dec.d_model, dec.d_model])?,
        decoder_projector_b: tensor("joint.pred.bias", &[dec.d_model])?,
        joint_head_w: tensor(
            "joint.joint_net.2.weight",
            &[
                config.joint.vocab_size + config.joint.durations.len(),
                dec.d_model,
            ],
        )?,
        joint_head_b: tensor(
            "joint.joint_net.2.bias",
            &[config.joint.vocab_size + config.joint.durations.len()],
        )?,
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
    tokenizer: Option<ParakeetTokenizer>,
    weight_license: LicenseClass,
    backend: BackendKind,
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
            weights: ParakeetWeightStore::Synthesized(Box::new(weights)),
            tokenizer: None,
            weight_license: LicenseClass::Unknown,
            backend: BackendKind::Cpu,
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
        let weights = load_bound_weights(file, &cfg)?;
        let tokenizer = if file.get(tokenizer::KEY_TOKENIZER_JSON).is_some() {
            Some(ParakeetTokenizer::from_gguf(file, cfg.joint.vocab_size)?)
        } else {
            None
        };
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|value| value.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        Ok(Self {
            cfg,
            weights: ParakeetWeightStore::Bound(Box::new(weights)),
            tokenizer,
            weight_license,
            backend: BackendKind::Cpu,
        })
    }

    /// Strictly binds the immutable public Parakeet-TDT-1.1B GGUF and an
    /// optional official plaintext SentencePiece vocabulary. The complete
    /// 1,667-tensor manifest and sidecar SHA-256 are authenticated before
    /// any payload is decoded.
    pub(crate) fn from_tdt_1_1b_gguf(
        file: &GgufFile,
        tokenizer_vocab: Option<&[u8]>,
    ) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, TDT_1_1B_SPEC)?;
        let cfg = ParakeetConfig::parakeet_tdt_1_1b();
        cfg.validate_for_forward()?;
        let weights = load_tdt_1_1b_bound_weights(file, &cfg)?;
        let embedded_vocab = if file.get(tokenizer::KEY_SENTENCEPIECE_VOCAB).is_some() {
            Some(tokenizer::required_u8_array(
                file,
                tokenizer::KEY_SENTENCEPIECE_VOCAB,
            )?)
        } else {
            None
        };
        let tokenizer_bytes = tokenizer_vocab.or(embedded_vocab.as_deref());
        let tokenizer = if let Some(bytes) = tokenizer_bytes {
            if sha256_bytes(bytes) != TDT_1_1B_TOKENIZER_VOCAB_SHA256 {
                return Err(VokraError::ModelLoad(format!(
                    "{TDT_1_1B_LABEL}: tokenizer.vocab SHA-256 does not match the immutable nvidia/parakeet-tdt-1.1b sidecar"
                )));
            }
            Some(ParakeetTokenizer::from_sentencepiece_vocab_bytes(
                bytes,
                cfg.joint.vocab_size,
                cfg.joint.blank_token_id,
            )?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            weights: ParakeetWeightStore::Bound(Box::new(weights)),
            tokenizer,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Selects the backend used by subsequent encoder and decoder calls.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
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

    /// Whether the engine holds a verified decode-only tokenizer (the 0.6B-v3
    /// BPE/Metaspace JSON or the original 1.1B SentencePiece vocabulary).
    #[must_use]
    pub const fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Runs one real zero-state prediction-network + combined TDT-head step.
    ///
    /// `encoder_hidden` is one FastConformer output row before the official
    /// `encoder_projector` (`[1024]` for both audited releases). This
    /// independently executable subgraph provides a focused decoder-side
    /// parity boundary in addition to the complete PCM transcription path.
    pub fn tdt_head_step(&self, encoder_hidden: &[f32], token_id: u32) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, PARAKEET_HOT_OPS)?;
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
            &compute,
            encoder_hidden,
            &weights.encoder_projector_w,
            &weights.encoder_projector_b,
            hidden,
            &mut encoder_projected,
        )?;
        let embed_offset = token_id as usize * hidden;
        let mut decoder = weights.embedding[embed_offset..embed_offset + hidden].to_vec();
        for layer in &weights.lstm {
            decoder = lstm_zero_state_step(&compute, &decoder, layer, hidden)?;
        }
        let mut decoder_projected = vec![0.0; hidden];
        linear_into(
            &compute,
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
            &compute,
            &decoder_projected,
            &weights.joint_head_w,
            &weights.joint_head_b,
            output_dim,
            &mut logits,
        )?;
        Ok(logits)
    }

    /// Runs the configured log-mel frontend, depthwise-separable Conv2D
    /// subsampler and relative-position FastConformer encoder. The returned
    /// buffer is row-major `[encoder_frames, d_model]` before the TDT encoder
    /// projector. The 0.6B-v3 release uses 128 mel bins / 24 blocks; the
    /// original 1.1B release uses 80 mel bins / 42 blocks.
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let compute = Compute::for_backend(self.backend, PARAKEET_HOT_OPS)?;
        self.encode_pcm_with_compute(pcm, &compute)
    }

    fn encode_pcm_with_compute(&self, pcm: &[f32], compute: &Compute) -> Result<(Vec<f32>, usize)> {
        let ParakeetWeightStore::Bound(weights) = &self.weights else {
            return Err(VokraError::NotImplemented(
                "ParakeetAsr::encode_pcm requires a real GGUF-bound checkpoint",
            ));
        };
        let (features, frames) =
            parakeet_logmel(pcm, self.cfg.sample_rate, self.cfg.encoder.in_dim)?;
        let (mut hidden, encoded_frames) = subsampling_forward(
            compute,
            &features,
            frames,
            self.cfg.encoder.in_dim,
            &weights.subsampling,
            &self.cfg.encoder,
        )?;
        if self.cfg.encoder.scale_input {
            let scale = (self.cfg.encoder.d_model as f32).sqrt();
            for value in &mut hidden {
                *value *= scale;
            }
        }
        let positions = relative_positions(encoded_frames, self.cfg.encoder.d_model);
        for block in &weights.encoder {
            conformer_block_forward(
                compute,
                &mut hidden,
                encoded_frames,
                block,
                &positions,
                &self.cfg.encoder,
            )?;
        }
        Ok((hidden, encoded_frames))
    }

    /// Transcribes 16 kHz mono `f32` PCM into emitted non-blank TDT token ids.
    /// Repeated token ids are retained; TDT is not CTC and must not collapse
    /// adjacent equal emissions.
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
                 not a real transcript. Bind an audited real Parakeet-TDT \
                 checkpoint before invoking transcribe. The shape flow (config validation, \
                 weight-store construction, PCM boundary check) remains \
                 available through ParakeetAsr::new; the real checkpoint path \
                 is ParakeetAsr::from_gguf for 0.6B-v3 or the strict \
                 parakeet_tdt_1_1b wrapper for the original 1.1B release.",
            ));
        }
        let ParakeetWeightStore::Bound(weights) = &self.weights else {
            unreachable!("synthesized branch returned above")
        };
        let compute = Compute::for_backend(self.backend, PARAKEET_HOT_OPS)?;
        let (encoder, frames) = self.encode_pcm_with_compute(pcm, &compute)?;
        let hidden = self.cfg.decoder.d_model;
        let mut projected = vec![0.0; frames * hidden];
        for frame in 0..frames {
            linear_into(
                &compute,
                &encoder[frame * self.cfg.encoder.d_model..(frame + 1) * self.cfg.encoder.d_model],
                &weights.encoder_projector_w,
                &weights.encoder_projector_b,
                hidden,
                &mut projected[frame * hidden..(frame + 1) * hidden],
            )?;
        }

        let mut state = ParakeetDecoderState::new(self.cfg.decoder.n_layer, hidden);
        decoder_step(
            &compute,
            self.cfg.joint.blank_token_id,
            weights,
            hidden,
            &mut state,
        )?;
        let mut tokens = Vec::new();
        let mut frame = 0usize;
        let max_steps = frames
            .checked_mul(self.cfg.joint.max_symbols_per_step)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "parakeet transcribe: decode step bound overflow".to_owned(),
                )
            })?;
        let mut steps = 0usize;
        let vocab = self.cfg.joint.vocab_size;
        while frame < frames && steps < max_steps {
            let mut joint = vec![0.0; hidden];
            for index in 0..hidden {
                joint[index] =
                    (projected[frame * hidden + index] + state.projected[index]).max(0.0);
            }
            let mut logits = vec![0.0; weights.joint_head_b.len()];
            linear_into(
                &compute,
                &joint,
                &weights.joint_head_w,
                &weights.joint_head_b,
                logits.len(),
                &mut logits,
            )?;
            let token = argmax_finite(&logits[..vocab], "Parakeet TDT token logits")? as u32;
            let duration_index = argmax_finite(&logits[vocab..], "Parakeet TDT duration logits")?;
            let mut duration = self.cfg.joint.durations[duration_index] as usize;
            if self.cfg.joint.eos_token_id == Some(token) {
                break;
            } else if token == self.cfg.joint.blank_token_id {
                if duration == 0 {
                    duration = 1;
                }
            } else {
                tokens.push(token);
                decoder_step(&compute, token, weights, hidden, &mut state)?;
            }
            frame = frame.saturating_add(duration);
            steps += 1;
        }
        Ok(tokens)
    }
}

impl AsrEngine for ParakeetAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let ids = ParakeetAsr::transcribe(self, pcm)?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "Parakeet ASR: `{}` is absent; reconvert with `--tokenizer tokenizer.json`",
                tokenizer::KEY_TOKENIZER_JSON
            ))
        })?;
        let text = tokenizer.decode(
            &ids,
            self.cfg.joint.blank_token_id,
            self.cfg.joint.pad_token_id,
            self.cfg.joint.eos_token_id,
        )?;
        Ok(Transcription::new(text))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

#[derive(Debug)]
struct ParakeetDecoderState {
    hidden: Vec<Vec<f32>>,
    cell: Vec<Vec<f32>>,
    projected: Vec<f32>,
}

impl ParakeetDecoderState {
    fn new(layers: usize, hidden: usize) -> Self {
        Self {
            hidden: vec![vec![0.0; hidden]; layers],
            cell: vec![vec![0.0; hidden]; layers],
            projected: vec![0.0; hidden],
        }
    }
}

fn decoder_step(
    compute: &Compute,
    token: u32,
    weights: &ParakeetBoundWeights,
    hidden: usize,
    state: &mut ParakeetDecoderState,
) -> Result<()> {
    let offset = token as usize * hidden;
    let mut input = weights.embedding[offset..offset + hidden].to_vec();
    for (layer_index, layer) in weights.lstm.iter().enumerate() {
        let mut gates = vec![0.0; 4 * hidden];
        compute.gemv_f32(
            4 * hidden,
            hidden,
            &layer.w_ih,
            &input,
            Some(&layer.b_ih),
            &mut gates,
        )?;
        let mut recurrent = vec![0.0; 4 * hidden];
        compute.gemv_f32(
            4 * hidden,
            hidden,
            &layer.w_hh,
            &state.hidden[layer_index],
            Some(&layer.b_hh),
            &mut recurrent,
        )?;
        let mut next = vec![0.0; hidden];
        for index in 0..hidden {
            let input_gate = sigmoid_f32(gates[index] + recurrent[index]);
            let forget_gate = sigmoid_f32(gates[hidden + index] + recurrent[hidden + index]);
            let candidate = (gates[2 * hidden + index] + recurrent[2 * hidden + index]).tanh();
            let output_gate =
                sigmoid_f32(gates[3 * hidden + index] + recurrent[3 * hidden + index]);
            let cell = forget_gate * state.cell[layer_index][index] + input_gate * candidate;
            state.cell[layer_index][index] = cell;
            next[index] = output_gate * cell.tanh();
        }
        state.hidden[layer_index].clone_from(&next);
        input = next;
    }
    linear_into(
        compute,
        &input,
        &weights.decoder_projector_w,
        &weights.decoder_projector_b,
        hidden,
        &mut state.projected,
    )
}

pub(crate) fn parakeet_logmel(
    pcm: &[f32],
    sample_rate: u32,
    n_mels: usize,
) -> Result<(Vec<f32>, usize)> {
    const N_FFT: usize = 512;
    const HOP: usize = 160;
    const WIN: usize = 400;
    const PREEMPHASIS: f32 = 0.97;
    const LOG_GUARD: f32 = 1.0 / 16_777_216.0;
    const EPSILON: f32 = 1e-5;

    let frames = pcm.len() / HOP;
    if frames < 2 {
        return Err(VokraError::InvalidArgument(format!(
            "parakeet transcribe: PCM has {} samples; at least {} are required for two normalized feature frames",
            pcm.len(),
            2 * HOP
        )));
    }
    let mut emphasized = vec![0.0; pcm.len()];
    emphasized[0] = pcm[0];
    for index in 1..pcm.len() {
        emphasized[index] = pcm[index] - PREEMPHASIS * pcm[index - 1];
    }
    let attrs = StftAttrs {
        n_fft: N_FFT,
        hop_length: HOP,
        win_length: WIN,
        window: Window::Hann,
        window_symmetry: WindowSymmetry::Symmetric,
        center: true,
        pad_mode: PadMode::Constant,
        normalization: Normalization::Backward,
        causal: false,
        real_input: true,
    };
    let spectrum = stft(&emphasized, &attrs)?;
    if spectrum.frames < frames {
        return Err(VokraError::InvalidArgument(
            "parakeet frontend: STFT returned fewer frames than the valid attention mask"
                .to_owned(),
        ));
    }
    let bins = N_FFT / 2 + 1;
    let mut power = vec![0.0; frames * bins];
    for (index, value) in power.iter_mut().enumerate() {
        *value = spectrum.re[index] * spectrum.re[index] + spectrum.im[index] * spectrum.im[index];
    }
    let mel = MelFilterbank::new(&MelAttrs::new(sample_rate, N_FFT, n_mels));
    let mut features = mel.apply(&power, frames);
    for value in &mut features {
        *value = (*value + LOG_GUARD).ln();
    }
    for channel in 0..n_mels {
        let mut mean = 0.0f32;
        for frame in 0..frames {
            mean += features[frame * n_mels + channel];
        }
        mean /= frames as f32;
        let mut variance = 0.0f32;
        for frame in 0..frames {
            let delta = features[frame * n_mels + channel] - mean;
            variance += delta * delta;
        }
        variance /= (frames - 1) as f32;
        let std = variance.sqrt();
        for frame in 0..frames {
            let index = frame * n_mels + channel;
            features[index] = (features[index] - mean) / (std + EPSILON);
        }
    }
    Ok((features, frames))
}

/// Padding convention for the shared FastConformer Conv2D subsampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastConformerSubsamplingPadding {
    /// Parakeet: ordinary symmetric `same` padding on time and frequency.
    Symmetric,
    /// Nemotron streaming encoder in offline mode: each causal Conv2D pads
    /// `(kernel - 1, stride - 1)` on both axes before applying an unpadded
    /// convolution. This produces the released 17-bin projection width from
    /// 128 mel bins after three stride-2 stages.
    CausalOffline,
}

#[derive(Debug, Clone, Copy)]
struct Conv2dPadding {
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
}

impl FastConformerSubsamplingPadding {
    fn resolve(self, kernel: usize, stride: usize) -> Conv2dPadding {
        match self {
            Self::Symmetric => {
                let padding = (kernel - 1) / 2;
                Conv2dPadding {
                    top: padding,
                    bottom: padding,
                    left: padding,
                    right: padding,
                }
            }
            Self::CausalOffline => Conv2dPadding {
                top: kernel - 1,
                bottom: stride - 1,
                left: kernel - 1,
                right: stride - 1,
            },
        }
    }
}

fn conv_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    before: usize,
    after: usize,
) -> Result<usize> {
    let padded = input
        .checked_add(before)
        .and_then(|value| value.checked_add(after))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "FastConformer Conv2D padded input length overflow".to_owned(),
            )
        })?;
    if stride == 0 || kernel == 0 || padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "FastConformer Conv2D invalid geometry: input={input}, kernel={kernel}, stride={stride}, padding=({before},{after})"
        )));
    }
    Ok((padded - kernel) / stride + 1)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_single_input_with_compute(
    compute: &Compute,
    input: &[f32],
    height: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: Conv2dPadding,
) -> Result<(Vec<f32>, usize, usize)> {
    let out_h = conv_output_size(height, kernel, stride, padding.top, padding.bottom)?;
    let out_w = conv_output_size(width, kernel, stride, padding.left, padding.right)?;
    let positions = out_h * out_w;
    let kernel_elems = kernel * kernel;
    let mut patches = vec![0.0f32; positions * kernel_elems];
    for out_y in 0..out_h {
        for out_x in 0..out_w {
            let row = (out_y * out_w + out_x) * kernel_elems;
            for kernel_y in 0..kernel {
                let source_y = out_y * stride + kernel_y;
                if source_y < padding.top || source_y - padding.top >= height {
                    continue;
                }
                for kernel_x in 0..kernel {
                    let source_x = out_x * stride + kernel_x;
                    if source_x < padding.left || source_x - padding.left >= width {
                        continue;
                    }
                    patches[row + kernel_y * kernel + kernel_x] =
                        input[(source_y - padding.top) * width + source_x - padding.left];
                }
            }
        }
    }
    let mut weight_t = vec![0.0f32; kernel_elems * out_channels];
    for channel in 0..out_channels {
        for k in 0..kernel_elems {
            weight_t[k * out_channels + channel] = weight[channel * kernel_elems + k];
        }
    }
    let mut spatial = vec![0.0f32; positions * out_channels];
    compute.gemm_f32(
        positions,
        out_channels,
        kernel_elems,
        &patches,
        &weight_t,
        Some(bias),
        &mut spatial,
    )?;
    let mut output = vec![0.0f32; out_channels * positions];
    for position in 0..positions {
        for channel in 0..out_channels {
            output[channel * positions + position] = spatial[position * out_channels + channel];
        }
    }
    Ok((output, out_h, out_w))
}

#[allow(clippy::too_many_arguments)]
fn depthwise_conv2d_with_compute(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    kernel: usize,
    stride: usize,
    padding: Conv2dPadding,
) -> Result<(Vec<f32>, usize, usize)> {
    let out_h = conv_output_size(height, kernel, stride, padding.top, padding.bottom)?;
    let out_w = conv_output_size(width, kernel, stride, padding.left, padding.right)?;
    let positions = out_h * out_w;
    let kernel_elems = kernel * kernel;
    let mut output = vec![0.0f32; channels * positions];
    // Compute has no native Conv2d primitive. Each channel is therefore one
    // small im2col GEMM; this is complete backend execution, not CPU fallback.
    for channel in 0..channels {
        let mut patches = vec![0.0f32; positions * kernel_elems];
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                let row = (out_y * out_w + out_x) * kernel_elems;
                for kernel_y in 0..kernel {
                    let source_y = out_y * stride + kernel_y;
                    if source_y < padding.top || source_y - padding.top >= height {
                        continue;
                    }
                    for kernel_x in 0..kernel {
                        let source_x = out_x * stride + kernel_x;
                        if source_x < padding.left || source_x - padding.left >= width {
                            continue;
                        }
                        patches[row + kernel_y * kernel + kernel_x] =
                            input[(channel * height + source_y - padding.top) * width + source_x
                                - padding.left];
                    }
                }
            }
        }
        compute.gemm_f32(
            positions,
            1,
            kernel_elems,
            &patches,
            &weight[channel * kernel_elems..(channel + 1) * kernel_elems],
            Some(&bias[channel..channel + 1]),
            &mut output[channel * positions..(channel + 1) * positions],
        )?;
    }
    Ok((output, out_h, out_w))
}

pub(crate) fn subsampling_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    frequency: usize,
    weights: &ParakeetBoundSubsampling,
    config: &ParakeetEncoderConfig,
) -> Result<(Vec<f32>, usize)> {
    subsampling_forward_with_padding(
        compute,
        input,
        frames,
        frequency,
        weights,
        config,
        FastConformerSubsamplingPadding::Symmetric,
    )
}

pub(crate) fn subsampling_forward_with_padding(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    frequency: usize,
    weights: &ParakeetBoundSubsampling,
    config: &ParakeetEncoderConfig,
    padding_mode: FastConformerSubsamplingPadding,
) -> Result<(Vec<f32>, usize)> {
    let channels = config.subsampling_conv_channels;
    let kernel = config.subsampling_conv_kernel_size;
    let stride = config.subsampling_conv_stride;
    if kernel == 0 || stride == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "FastConformer subsampling requires non-zero kernel/stride, found kernel={kernel}, stride={stride}"
        )));
    }
    let padding = padding_mode.resolve(kernel, stride);
    let (mut value, mut time, mut freq) = conv2d_single_input_with_compute(
        compute,
        input,
        frames,
        frequency,
        &weights.conv0_w,
        &weights.conv0_b,
        channels,
        kernel,
        stride,
        padding,
    )?;
    for entry in &mut value {
        *entry = entry.max(0.0);
    }

    for stage in 0..2 {
        let (depthwise, next_time, next_freq) = depthwise_conv2d_with_compute(
            compute,
            &value,
            channels,
            time,
            freq,
            &weights.depthwise_w[stage],
            &weights.depthwise_b[stage],
            kernel,
            stride,
            padding,
        )?;
        let positions = next_time * next_freq;
        let mut spatial = vec![0.0; positions * channels];
        for channel in 0..channels {
            for position in 0..positions {
                spatial[position * channels + channel] = depthwise[channel * positions + position];
            }
        }
        let mut projected = vec![0.0; positions * channels];
        compute.gemm_f32(
            positions,
            channels,
            channels,
            &spatial,
            &weights.pointwise_w_t[stage],
            Some(&weights.pointwise_b[stage]),
            &mut projected,
        )?;
        for entry in &mut projected {
            *entry = entry.max(0.0);
        }
        value = vec![0.0; channels * positions];
        for channel in 0..channels {
            for position in 0..positions {
                value[channel * positions + position] = projected[position * channels + channel];
            }
        }
        time = next_time;
        freq = next_freq;
    }

    let projection_in = channels * freq;
    let mut flattened = vec![0.0; time * projection_in];
    for out_t in 0..time {
        for channel in 0..channels {
            for out_f in 0..freq {
                flattened[out_t * projection_in + channel * freq + out_f] =
                    value[(channel * time + out_t) * freq + out_f];
            }
        }
    }
    let mut output = vec![0.0; time * config.d_model];
    compute.gemm_f32(
        time,
        config.d_model,
        projection_in,
        &flattened,
        &weights.linear_w_t,
        Some(&weights.linear_b),
        &mut output,
    )?;
    Ok((output, time))
}

pub(crate) fn relative_positions(frames: usize, width: usize) -> Vec<f32> {
    let count = 2 * frames - 1;
    let mut output = vec![0.0; count * width];
    for position_index in 0..count {
        let position = (frames - 1) as isize - position_index as isize;
        for pair in 0..width / 2 {
            let exponent = (2 * pair) as f32 / width as f32;
            let frequency = 1.0f32 / 10_000.0f32.powf(exponent);
            let angle = position as f32 * frequency;
            output[position_index * width + 2 * pair] = angle.sin();
            output[position_index * width + 2 * pair + 1] = angle.cos();
        }
    }
    output
}

/// Fixed-width relative positions used by NeMo's
/// `RelPositionMultiHeadAttentionLongformer`. Unlike full Transformer-XL
/// attention, the positional table is independent of utterance length and
/// covers exactly `[left, ..., 0, ..., -right]`.
pub(crate) fn local_relative_positions(
    left_context: usize,
    right_context: usize,
    width: usize,
) -> Vec<f32> {
    let count = left_context + right_context + 1;
    let mut output = vec![0.0; count * width];
    for position_index in 0..count {
        let position = left_context as isize - position_index as isize;
        for pair in 0..width / 2 {
            let exponent = (2 * pair) as f32 / width as f32;
            let frequency = 1.0f32 / 10_000.0f32.powf(exponent);
            let angle = position as f32 * frequency;
            output[position_index * width + 2 * pair] = angle.sin();
            output[position_index * width + 2 * pair + 1] = angle.cos();
        }
    }
    output
}

fn layer_norm(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    norm: &ParakeetBoundNorm,
) -> Result<Vec<f32>> {
    let width = norm.weight.len();
    let mut output = vec![0.0; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        rows,
        width,
        &norm.weight,
        &norm.bias,
        1e-5,
    )?;
    Ok(output)
}

struct FeedForwardWeights<'a> {
    w1_t: &'a [f32],
    b1: Option<&'a [f32]>,
    w2_t: &'a [f32],
    b2: Option<&'a [f32]>,
}

fn feed_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    width: usize,
    inner: usize,
    weights: FeedForwardWeights<'_>,
) -> Result<Vec<f32>> {
    let mut expanded = vec![0.0; frames * inner];
    compute.gemm_f32(
        frames,
        inner,
        width,
        input,
        weights.w1_t,
        weights.b1,
        &mut expanded,
    )?;
    for value in &mut expanded {
        *value *= sigmoid_f32(*value);
    }
    let mut output = vec![0.0; frames * width];
    compute.gemm_f32(
        frames,
        width,
        inner,
        &expanded,
        weights.w2_t,
        weights.b2,
        &mut output,
    )?;
    Ok(output)
}

/// Attention visibility used by the shared relative-position FastConformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastConformerAttentionContext {
    /// Full bidirectional attention (Parakeet offline inference).
    Full,
    /// Cache-aware Nemotron offline mask. Queries see their current
    /// `right_context + 1` frame chunk and at most
    /// `left_context / chunk_size` earlier chunks.
    ChunkedLimited {
        left_context: usize,
        right_context: usize,
    },
    /// NeMo 1.21 Longformer-style local relative attention. Ordinary
    /// queries attend to their bounded left/right window plus every global
    /// key. Global queries attend to the complete sequence. Global keys are
    /// intentionally represented twice when they are also inside the local
    /// window, matching NeMo's concatenated global-key + sliding-window
    /// probability axis.
    LongformerLocal {
        left_context: usize,
        right_context: usize,
        global_tokens: usize,
        global_tokens_spacing: usize,
    },
}

impl FastConformerAttentionContext {
    fn allows(self, query: usize, key: usize) -> bool {
        match self {
            Self::Full => true,
            Self::ChunkedLimited {
                left_context,
                right_context,
            } => {
                let chunk_size = right_context + 1;
                let left_context_chunks = left_context / chunk_size;
                let query_chunk = query / chunk_size;
                let key_chunk = key / chunk_size;
                key_chunk <= query_chunk && query_chunk - key_chunk <= left_context_chunks
            }
            Self::LongformerLocal {
                left_context,
                right_context,
                global_tokens,
                global_tokens_spacing,
            } => {
                let global_limit = global_tokens.saturating_mul(global_tokens_spacing);
                let is_global = |position: usize| {
                    global_tokens_spacing != 0
                        && position < global_limit
                        && position % global_tokens_spacing == 0
                };
                is_global(query)
                    || is_global(key)
                    || (key.saturating_add(left_context) >= query
                        && key <= query.saturating_add(right_context))
            }
        }
    }
}

fn attention_forward_longformer(
    compute: &Compute,
    input: &[f32],
    positions: &[f32],
    frames: usize,
    block: &ParakeetBoundEncoderBlock,
    config: &ParakeetEncoderConfig,
    left_context: usize,
    right_context: usize,
    global_tokens: usize,
    global_tokens_spacing: usize,
) -> Result<Vec<f32>> {
    if frames == 0 || left_context == 0 || right_context == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "FastConformer Longformer attention requires non-zero frames and left/right context; found frames={frames}, left={left_context}, right={right_context}"
        )));
    }
    if global_tokens > 0 && global_tokens_spacing == 0 {
        return Err(VokraError::InvalidArgument(
            "FastConformer Longformer global_tokens_spacing must be > 0 when global tokens are enabled"
                .to_owned(),
        ));
    }
    let width = config.d_model;
    let heads = config.n_head;
    let head_dim = config.head_dim();
    let position_count = left_context
        .checked_add(right_context)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "FastConformer Longformer positional width overflow".to_owned(),
            )
        })?;
    if positions.len() != position_count * width {
        return Err(VokraError::InvalidArgument(format!(
            "FastConformer Longformer positions len {}, expected ({left_context} + {right_context} + 1) * {width} = {}",
            positions.len(),
            position_count * width
        )));
    }

    let project = |weight: &[f32], bias: Option<&[f32]>| -> Result<Vec<f32>> {
        let mut output = vec![0.0; frames * width];
        compute.gemm_f32(frames, width, width, input, weight, bias, &mut output)?;
        Ok(output)
    };
    let q = project(&block.q_w_t, block.q_b.as_deref())?;
    let k = project(&block.k_w_t, block.k_b.as_deref())?;
    let v = project(&block.v_w_t, block.v_b.as_deref())?;
    let mut relative_k = vec![0.0; position_count * width];
    compute.gemm_f32(
        position_count,
        width,
        width,
        positions,
        &block.relative_k_w_t,
        None,
        &mut relative_k,
    )?;

    let global_indices = (0..global_tokens)
        .filter_map(|index| index.checked_mul(global_tokens_spacing))
        .take_while(|&index| index < frames)
        .collect::<Vec<_>>();
    let is_global_query = |query: usize| global_indices.binary_search(&query).is_ok();
    let local_queries = (0..frames)
        .filter(|&query| !is_global_query(query))
        .collect::<Vec<_>>();
    let local_columns = global_indices.len() + position_count;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut context = vec![0.0f32; frames * width];

    if !local_queries.is_empty() {
        let rows = heads * local_queries.len();
        let mut scores = vec![f32::NEG_INFINITY; rows * local_columns];
        for head in 0..heads {
            let head_offset = head * head_dim;
            for (query_row, &query) in local_queries.iter().enumerate() {
                let row = head * local_queries.len() + query_row;
                let q_offset = query * width + head_offset;

                // NeMo concatenates content-only global-key logits in front
                // of the local relative-attention logits. A global key that
                // falls in the local window therefore appears twice by
                // design; do not deduplicate it here.
                for (slot, &key) in global_indices.iter().enumerate() {
                    let k_offset = key * width + head_offset;
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        score += q[q_offset + dim] * k[k_offset + dim];
                    }
                    scores[row * local_columns + slot] = score * scale;
                }

                for position_index in 0..position_count {
                    let delta = position_index as isize - left_context as isize;
                    let key_signed = query as isize + delta;
                    if key_signed < 0 || key_signed >= frames as isize {
                        continue;
                    }
                    let key = key_signed as usize;
                    let k_offset = key * width + head_offset;
                    let p_offset = position_index * width + head_offset;
                    let mut content = 0.0f32;
                    let mut relative = 0.0f32;
                    for dim in 0..head_dim {
                        let hidden = head_offset + dim;
                        content += (q[q_offset + dim] + block.bias_u[hidden]) * k[k_offset + dim];
                        relative +=
                            (q[q_offset + dim] + block.bias_v[hidden]) * relative_k[p_offset + dim];
                    }
                    scores[row * local_columns + global_indices.len() + position_index] =
                        (content + relative) * scale;
                }
            }
        }
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, rows, local_columns)?;
        for head in 0..heads {
            let head_offset = head * head_dim;
            for (query_row, &query) in local_queries.iter().enumerate() {
                let row = head * local_queries.len() + query_row;
                for (slot, &key) in global_indices.iter().enumerate() {
                    let probability = probabilities[row * local_columns + slot];
                    let v_offset = key * width + head_offset;
                    for dim in 0..head_dim {
                        context[query * width + head_offset + dim] +=
                            probability * v[v_offset + dim];
                    }
                }
                for position_index in 0..position_count {
                    let delta = position_index as isize - left_context as isize;
                    let key_signed = query as isize + delta;
                    if key_signed < 0 || key_signed >= frames as isize {
                        continue;
                    }
                    let key = key_signed as usize;
                    let probability =
                        probabilities[row * local_columns + global_indices.len() + position_index];
                    let v_offset = key * width + head_offset;
                    for dim in 0..head_dim {
                        context[query * width + head_offset + dim] +=
                            probability * v[v_offset + dim];
                    }
                }
            }
        }
    }

    // Global queries use content-only attention over every key and overwrite
    // the local result, exactly as NeMo 1.21's `_compute_out_global_to_all`.
    if !global_indices.is_empty() {
        let rows = heads * global_indices.len();
        let mut scores = vec![0.0f32; rows * frames];
        for head in 0..heads {
            let head_offset = head * head_dim;
            for (global_row, &query) in global_indices.iter().enumerate() {
                let row = head * global_indices.len() + global_row;
                let q_offset = query * width + head_offset;
                for key in 0..frames {
                    let k_offset = key * width + head_offset;
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        score += q[q_offset + dim] * k[k_offset + dim];
                    }
                    scores[row * frames + key] = score * scale;
                }
            }
        }
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, rows, frames)?;
        for head in 0..heads {
            let head_offset = head * head_dim;
            for (global_row, &query) in global_indices.iter().enumerate() {
                let row = head * global_indices.len() + global_row;
                for key in 0..frames {
                    let probability = probabilities[row * frames + key];
                    let v_offset = key * width + head_offset;
                    for dim in 0..head_dim {
                        context[query * width + head_offset + dim] +=
                            probability * v[v_offset + dim];
                    }
                }
            }
        }
    }

    let mut output = vec![0.0; frames * width];
    compute.gemm_f32(
        frames,
        width,
        width,
        &context,
        &block.o_w_t,
        block.o_b.as_deref(),
        &mut output,
    )?;
    Ok(output)
}

fn attention_forward(
    compute: &Compute,
    input: &[f32],
    positions: &[f32],
    frames: usize,
    block: &ParakeetBoundEncoderBlock,
    config: &ParakeetEncoderConfig,
    attention_context: FastConformerAttentionContext,
) -> Result<Vec<f32>> {
    if let FastConformerAttentionContext::LongformerLocal {
        left_context,
        right_context,
        global_tokens,
        global_tokens_spacing,
    } = attention_context
    {
        return attention_forward_longformer(
            compute,
            input,
            positions,
            frames,
            block,
            config,
            left_context,
            right_context,
            global_tokens,
            global_tokens_spacing,
        );
    }
    let width = config.d_model;
    let heads = config.n_head;
    let head_dim = config.head_dim();
    let project = |weight: &[f32], bias: Option<&[f32]>| -> Result<Vec<f32>> {
        let mut output = vec![0.0; frames * width];
        compute.gemm_f32(frames, width, width, input, weight, bias, &mut output)?;
        Ok(output)
    };
    let q = project(&block.q_w_t, block.q_b.as_deref())?;
    let k = project(&block.k_w_t, block.k_b.as_deref())?;
    let v = project(&block.v_w_t, block.v_b.as_deref())?;
    let position_count = 2 * frames - 1;
    let mut relative_k = vec![0.0; position_count * width];
    compute.gemm_f32(
        position_count,
        width,
        width,
        positions,
        &block.relative_k_w_t,
        None,
        &mut relative_k,
    )?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0; heads * frames * frames];
    for head in 0..heads {
        let mut q_content = vec![0.0f32; frames * head_dim];
        let mut q_position = vec![0.0f32; frames * head_dim];
        let mut k_t = vec![0.0f32; head_dim * frames];
        let mut relative_t = vec![0.0f32; head_dim * position_count];
        for frame in 0..frames {
            for dim in 0..head_dim {
                let hidden_index = head * head_dim + dim;
                let query_value = q[frame * width + hidden_index];
                q_content[frame * head_dim + dim] = query_value + block.bias_u[hidden_index];
                q_position[frame * head_dim + dim] = query_value + block.bias_v[hidden_index];
                k_t[dim * frames + frame] = k[frame * width + hidden_index];
            }
        }
        for position in 0..position_count {
            for dim in 0..head_dim {
                relative_t[dim * position_count + position] =
                    relative_k[position * width + head * head_dim + dim];
            }
        }
        let mut content_scores = vec![0.0f32; frames * frames];
        compute.gemm_f32(
            frames,
            frames,
            head_dim,
            &q_content,
            &k_t,
            None,
            &mut content_scores,
        )?;
        let mut position_scores = vec![0.0f32; frames * position_count];
        compute.gemm_f32(
            frames,
            position_count,
            head_dim,
            &q_position,
            &relative_t,
            None,
            &mut position_scores,
        )?;
        for query in 0..frames {
            for key in 0..frames {
                let relative = frames - 1 - query + key;
                scores[(head * frames + query) * frames + key] =
                    if attention_context.allows(query, key) {
                        (content_scores[query * frames + key]
                            + position_scores[query * position_count + relative])
                            * scale
                    } else {
                        f32::NEG_INFINITY
                    };
            }
        }
    }
    let mut probabilities = vec![0.0; scores.len()];
    compute.softmax_f32(&scores, &mut probabilities, heads * frames, frames)?;
    let mut context = vec![0.0; frames * width];
    for head in 0..heads {
        let mut v_head = vec![0.0f32; frames * head_dim];
        for key in 0..frames {
            for dim in 0..head_dim {
                v_head[key * head_dim + dim] = v[key * width + head * head_dim + dim];
            }
        }
        let mut context_head = vec![0.0f32; frames * head_dim];
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities[head * frames * frames..(head + 1) * frames * frames],
            &v_head,
            None,
            &mut context_head,
        )?;
        for query in 0..frames {
            for dim in 0..head_dim {
                context[query * width + head * head_dim + dim] =
                    context_head[query * head_dim + dim];
            }
        }
    }
    let mut output = vec![0.0; frames * width];
    compute.gemm_f32(
        frames,
        width,
        width,
        &context,
        &block.o_w_t,
        block.o_b.as_deref(),
        &mut output,
    )?;
    Ok(output)
}

fn convolution_forward(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    block: &ParakeetBoundEncoderBlock,
    config: &ParakeetEncoderConfig,
) -> Result<Vec<f32>> {
    let width = config.d_model;
    let mut doubled = vec![0.0; frames * 2 * width];
    compute.gemm_f32(
        frames,
        2 * width,
        width,
        input,
        &block.conv_pw1_w_t,
        block.conv_pw1_b.as_deref(),
        &mut doubled,
    )?;
    let mut gated = vec![0.0; frames * width];
    for frame in 0..frames {
        for channel in 0..width {
            gated[frame * width + channel] = doubled[frame * 2 * width + channel]
                * sigmoid_f32(doubled[frame * 2 * width + width + channel]);
        }
    }
    let kernel = config.conv_kernel_size;
    let mut gated_channels = vec![0.0f32; gated.len()];
    for frame in 0..frames {
        for channel in 0..width {
            gated_channels[channel * frames + frame] = gated[frame * width + channel];
        }
    }
    let (conv_input, conv_frames, padding) = match &block.conv_inner_norm {
        FastConformerConvNorm::BatchNorm { .. } => (gated_channels, frames, (kernel - 1) / 2),
        FastConformerConvNorm::LayerNorm(_) => {
            let left_padding = kernel - 1;
            let padded_frames = frames.checked_add(left_padding).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "FastConformer causal Conv1D frame count overflow".to_owned(),
                )
            })?;
            let mut padded = vec![0.0f32; width * padded_frames];
            for channel in 0..width {
                let source = &gated_channels[channel * frames..(channel + 1) * frames];
                let target = &mut padded[channel * padded_frames + left_padding
                    ..channel * padded_frames + left_padding + frames];
                target.copy_from_slice(source);
            }
            (padded, padded_frames, 0)
        }
    };
    let mut convolved_channels = vec![0.0f32; gated.len()];
    compute.grouped_conv1d_f32(
        &conv_input,
        width,
        conv_frames,
        &block.conv_dw_w,
        width,
        kernel,
        block.conv_dw_b.as_deref(),
        1,
        padding,
        width,
        &mut convolved_channels,
    )?;
    let mut convolved = vec![0.0; gated.len()];
    for frame in 0..frames {
        for channel in 0..width {
            convolved[frame * width + channel] = convolved_channels[channel * frames + frame];
        }
    }
    match &block.conv_inner_norm {
        FastConformerConvNorm::BatchNorm {
            weight,
            bias,
            running_mean,
            running_var,
        } => {
            for frame in 0..frames {
                for channel in 0..width {
                    let index = frame * width + channel;
                    let normalized = (convolved[index] - running_mean[channel])
                        / (running_var[channel] + 1e-5).sqrt()
                        * weight[channel]
                        + bias[channel];
                    convolved[index] = normalized * sigmoid_f32(normalized);
                }
            }
        }
        FastConformerConvNorm::LayerNorm(norm) => {
            convolved = layer_norm(compute, &convolved, frames, norm)?;
            for value in &mut convolved {
                *value *= sigmoid_f32(*value);
            }
        }
    }
    let mut output = vec![0.0; convolved.len()];
    compute.gemm_f32(
        frames,
        width,
        width,
        &convolved,
        &block.conv_pw2_w_t,
        block.conv_pw2_b.as_deref(),
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn conformer_block_forward(
    compute: &Compute,
    hidden: &mut [f32],
    frames: usize,
    block: &ParakeetBoundEncoderBlock,
    positions: &[f32],
    config: &ParakeetEncoderConfig,
) -> Result<()> {
    conformer_block_forward_with_context(
        compute,
        hidden,
        frames,
        block,
        positions,
        config,
        FastConformerAttentionContext::Full,
    )
}

pub(crate) fn conformer_block_forward_with_context(
    compute: &Compute,
    hidden: &mut [f32],
    frames: usize,
    block: &ParakeetBoundEncoderBlock,
    positions: &[f32],
    config: &ParakeetEncoderConfig,
    attention_context: FastConformerAttentionContext,
) -> Result<()> {
    let width = config.d_model;
    let normalized = layer_norm(compute, hidden, frames, &block.norm_ff1)?;
    let ff1 = feed_forward(
        compute,
        &normalized,
        frames,
        width,
        config.ffn_dim,
        FeedForwardWeights {
            w1_t: &block.ff1_w1_t,
            b1: block.ff1_b1.as_deref(),
            w2_t: &block.ff1_w2_t,
            b2: block.ff1_b2.as_deref(),
        },
    )?;
    for (value, branch) in hidden.iter_mut().zip(ff1) {
        *value += 0.5 * branch;
    }
    let normalized = layer_norm(compute, hidden, frames, &block.norm_attn)?;
    let attention = attention_forward(
        compute,
        &normalized,
        positions,
        frames,
        block,
        config,
        attention_context,
    )?;
    for (value, branch) in hidden.iter_mut().zip(attention) {
        *value += branch;
    }
    let normalized = layer_norm(compute, hidden, frames, &block.norm_conv)?;
    let convolution = convolution_forward(compute, &normalized, frames, block, config)?;
    for (value, branch) in hidden.iter_mut().zip(convolution) {
        *value += branch;
    }
    let normalized = layer_norm(compute, hidden, frames, &block.norm_ff2)?;
    let ff2 = feed_forward(
        compute,
        &normalized,
        frames,
        width,
        config.ffn_dim,
        FeedForwardWeights {
            w1_t: &block.ff2_w1_t,
            b1: block.ff2_b1.as_deref(),
            w2_t: &block.ff2_w2_t,
            b2: block.ff2_b2.as_deref(),
        },
    )?;
    for (value, branch) in hidden.iter_mut().zip(ff2) {
        *value += 0.5 * branch;
    }
    let normalized = layer_norm(compute, hidden, frames, &block.norm_out)?;
    hidden.copy_from_slice(&normalized);
    Ok(())
}

fn argmax_finite(values: &[f32], label: &str) -> Result<usize> {
    let mut best = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: non-finite value at index {index}: {value}"
            )));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: empty axis")))
}

fn linear_into(
    compute: &Compute,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    compute.gemv_f32(output_dim, input.len(), weight, input, Some(bias), output)
}

fn lstm_zero_state_step(
    compute: &Compute,
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
    compute.gemv_f32(
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

    #[test]
    fn causal_subsampling_reproduces_nemotron_projection_frequency() {
        let padding = FastConformerSubsamplingPadding::CausalOffline.resolve(3, 2);
        let mut frequency = 128;
        for _ in 0..3 {
            frequency = conv_output_size(frequency, 3, 2, padding.left, padding.right)
                .expect("canonical causal subsampling geometry");
        }
        assert_eq!(frequency, 17);
        assert_eq!(frequency * 256, 4_352);
    }

    #[test]
    fn chunked_limited_attention_matches_upstream_chunk_boundaries() {
        let context = FastConformerAttentionContext::ChunkedLimited {
            left_context: 56,
            right_context: 3,
        };
        // Queries 0..=3 share chunk zero and may see the whole current chunk.
        assert!(context.allows(0, 3));
        // A key in a future chunk is hidden.
        assert!(!context.allows(3, 4));
        // 56 frames / 4 = fourteen prior chunks are retained.
        assert!(context.allows(60, 4));
        assert!(!context.allows(60, 3));
    }

    #[test]
    fn longformer_visibility_matches_local_plus_global_contract() {
        let context = FastConformerAttentionContext::LongformerLocal {
            left_context: 2,
            right_context: 1,
            global_tokens: 1,
            global_tokens_spacing: 1,
        };
        // Token zero is global: it sees all keys and every query sees it.
        assert!(context.allows(0, 9));
        assert!(context.allows(9, 0));
        // Ordinary queries retain their bounded asymmetric window only.
        assert!(context.allows(5, 3));
        assert!(context.allows(5, 6));
        assert!(!context.allows(5, 2));
        assert!(!context.allows(5, 7));
    }

    #[test]
    fn local_relative_positions_are_the_matching_full_table_slice() {
        assert_eq!(local_relative_positions(2, 2, 8), relative_positions(3, 8));
    }

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
        if let Some(eos_token_id) = config.joint.eos_token_id {
            builder.add_u32(KEY_JOINT_EOS_ID, eos_token_id);
        }
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
    fn config_eos_out_of_range_or_blank_is_rejected() {
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.eos_token_id = Some(c.joint.vocab_size as u32);
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut c = ParakeetConfig::tiny_for_tests();
        c.joint.eos_token_id = Some(c.joint.blank_token_id);
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
