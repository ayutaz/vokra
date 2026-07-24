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
//!     NeMo-canonical convention that matches [`vokra_ops::rnnt_decode`]'s
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
//! # Boundary — Conformer / RNN-T decoder ops consumed, never re-implemented
//!
//! Parakeet reuses two shared Vokra primitives instead of duplicating them:
//!
//! - **Encoder body**: [`vokra_ops::conformer`] — the Conformer /
//!   FastConformer encoder covers both variants via
//!   `ConvSubsampleKind::Stacking { factor: 8 }` (matches
//!   `subsampling_factor=8`). The primitive was authored for exactly this
//!   family (its module docs list `parakeet` as the first consumer).
//! - **TDT decoding**: [`vokra_ops::rnnt_decode`] — the primitive covers
//!   greedy / beam / TDT with the exact NeMo semantics
//!   (`durations = [0..=4]`, `blank_id = vocab_size`, `max_symbols_per_step
//!   = 10`).
//!
//! # What lands in this Phase 2 slice
//!
//! - [`ParakeetConfig`] — every hparam transcribed from the primary
//!   source (no hardcoded fabrication; sample-rate is inherited from the
//!   preprocessor, documented on the field).
//! - [`ParakeetWeights`] — a scaffold weight store with a deterministic
//!   [`ParakeetWeights::synthesized`] fixture (SplitMix64 + Xavier) so
//!   shape / dtype / size flow can be exercised without the real HF
//!   checkpoint. Real-checkpoint parity is a follow-up wave gated on the
//!   real-checkpoint tensor-name manifest (T29-equivalent — the Moshi /
//!   CSM / Zonos / Kyutai STT pattern).
//! - [`ParakeetAsr`] — engine handle carrying config + weights.
//!   [`ParakeetAsr::transcribe`] returns [`VokraError::NotImplemented`]
//!   until real weights are bound (the real forward — 128-bin log-mel →
//!   FastConformer encoder → 640-dim RNN-T prediction net → joint →
//!   `rnnt_decode(Tdt { duration_bins: [0..=4] })` → SentencePiece
//!   detokenize — is a follow-up wave gated on the real-checkpoint tensor
//!   manifest).
//!
//! # No ONNX (permanent)
//!
//! Parakeet ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet/` (whisper.cpp
//! 型, CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

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
    /// Recorded verbatim; the ops-side [`vokra_ops::rnnt_decode`] takes
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
    weights: ParakeetWeights,
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

        Ok(Self { cfg, weights })
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
        self.weights.is_synthesized
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate.
    ///
    /// This is the primary PCM → text entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// text (they would be noise or a hallucinated fixed sequence), so
    /// this returns [`VokraError::NotImplemented`] naming the blocker.
    /// Callers verify the shape flow through [`ParakeetAsr::new`] +
    /// [`ParakeetWeights::synthesized`] today; a follow-up wave binds the
    /// real HF checkpoint tensor names and wires the forward.
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
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "parakeet transcribe: this engine holds synthesized weights \
                 (deterministic fixture from ParakeetWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, \
                 not a real transcript. Bind real Parakeet-TDT-0.6B-v3 \
                 weights (CC-BY 4.0, nvidia/parakeet-tdt-0.6b-v3) before \
                 invoking transcribe. The shape flow (config validation, \
                 weight-store construction, PCM boundary check) is exercised \
                 through ParakeetAsr::new; the real-checkpoint tensor-name \
                 manifest lands in a follow-up wave (T29-equivalent — the \
                 Moshi / CSM / Zonos / Kyutai STT pattern).",
            ));
        }
        Err(VokraError::NotImplemented(
            "parakeet transcribe: real weights are bound but the log-mel \
             front-end → FastConformer encoder (vokra_ops::conformer) → \
             RNN-T prediction net → joint → rnnt_decode(Tdt { \
             duration_bins: joint.durations }) → SentencePiece detokenize \
             forward path has not landed yet. Follow-up wave: wire \
             ParakeetWeights to vokra_ops::conformer::ConformerEncoder + a \
             per-frame prediction-net step + the rnnt_decode TDT path with \
             blank_id = joint.blank_token_id and max_symbols_per_step = \
             joint.max_symbols_per_step.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
}
