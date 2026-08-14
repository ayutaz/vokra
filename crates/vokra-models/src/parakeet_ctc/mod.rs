//! Parakeet **CTC-1.1B** — NVIDIA's FastConformer + CTC ASR (SoTA plan
//! Phase 2, 2026-07-24).
//!
//! # What Parakeet-CTC-1.1B is (primary source)
//!
//! Parakeet-CTC-1.1B is NVIDIA NeMo's 1.1B FastConformer encoder + CTC head
//! for English ASR. Unlike its Parakeet-TDT / Parakeet-RNN-T siblings, the
//! CTC variant has **no RNN-T prediction network and no joint / duration
//! head** — the model is encoder + a single CTC vocab head, and decoding is
//! a host-side greedy blank-fold or prefix beam search
//! ([`vokra_ops::ctc_decode`]).
//!
//! Every hparam below is transcribed **verbatim** from
//! `huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Model type** (`model_type`): `"parakeet_ctc"`
//!   (`architectures = ["ParakeetForCTC"]`).
//! - **Encoder** (`encoder_config`, `model_type = "parakeet_encoder"`):
//!   FastConformer.
//!   - `hidden_size` = 1024 (aka `d_model`),
//!   - `num_hidden_layers` = 42,
//!   - `num_attention_heads` = 8, `num_key_value_heads` = 8 (**MHA**,
//!     `num_heads_kv == num_heads` → no GQA broadcast),
//!   - `intermediate_size` = 4096 (FFN inner width; the ~4× expansion
//!     factor upstream `ff_expansion_factor=4` implies for `d_model=1024`),
//!   - `conv_kernel_size` = 9 (FastConformer convolution kernel),
//!   - `hidden_act` = `"silu"`,
//!   - `max_position_embeddings` = 5000,
//!   - `num_mel_bins` = 80 (**input log-mel channels — 80, not 128** —
//!     the 1.1B CTC checkpoint uses the classic 80-bin front-end, whereas
//!     the 0.6B TDT-v3 checkpoint uses 128 bins),
//!   - `subsampling_factor` = 8 (**FastConformer 8× downsampling**),
//!     `subsampling_conv_kernel_size` = 3, `subsampling_conv_stride` = 2,
//!     `subsampling_conv_channels` = 256,
//!   - `attention_bias` = **true** (**Q/K/V/out projections carry biases**
//!     — differs from Parakeet-TDT-0.6B-v3 where this is false),
//!   - `scale_input` = **true** (**subsample stem scales the input** —
//!     differs from Parakeet-TDT-0.6B-v3 where this is false),
//!   - `initializer_range` = 0.02,
//!   - `dropout` / `attention_dropout` / `activation_dropout` / `layerdrop`
//!     / `dropout_positions` (train-time only — inference is dropout-free).
//! - **CTC head + vocab**:
//!   - `vocab_size` = 1025 (**1024 SentencePiece pieces + 1 blank**),
//!   - `pad_token_id` = 1024 (**doubles as the CTC blank** — the
//!     NeMo-canonical convention that puts the blank at the tail of the
//!     head, matching [`vokra_ops::ctc_decode_greedy`]'s `blank_id`
//!     parameter),
//!   - `ctc_loss_reduction` = `"mean"` (train-time only — inference does
//!     not compute the loss),
//!   - `ctc_zero_infinity` = `true` (train-time only).
//! - **Audio boundary**: `sample_rate` = 16 000 (16 kHz mono `.wav` /
//!   `.flac` per the model card — **not** written in `config.json`; the
//!   preprocessor side-car names it).
//! - **Weight license**: **CC-BY 4.0** (`AttributionRequired`) — the
//!   converter stamps the FR-MD-09 attribution text; the compliance
//!   registry maps `parakeet-ctc-1.1b` / `parakeet-ctc-1.1B` /
//!   `parakeet-ctc` to
//!   [`vokra_core::LicenseClass::AttributionRequired`] via the family
//!   prefix walk (`parakeet-`) so the M2-13 gate passes commercially *and*
//!   the FR-MD-09 attribution surface activates.
//!
//! # Boundary — Conformer / CTC ops consumed, never re-implemented
//!
//! Parakeet-CTC reuses two shared Vokra primitives instead of duplicating them:
//!
//! - **Encoder body**: [`vokra_ops::conformer`] — the Conformer /
//!   FastConformer encoder covers both variants via
//!   `ConvSubsampleKind::Stacking { factor: 8 }` (matches
//!   `subsampling_factor=8`). This is the same primitive Parakeet-TDT uses.
//! - **CTC decoding**: [`vokra_ops::ctc_decode`] — greedy blank-fold or
//!   prefix beam search with LM shallow fusion + hotword boost, matching
//!   NeMo's `BeamCTCInfer` signature. Uses the `pad_token_id = 1024` as
//!   `blank_id`.
//!
//! # What lands in this Phase 2 slice
//!
//! - [`ParakeetCtcConfig`] — every hparam transcribed from the primary
//!   source (no hardcoded fabrication; sample-rate is inherited from the
//!   preprocessor, documented on the field).
//! - [`ParakeetCtcWeights`] — a scaffold weight store with a deterministic
//!   [`ParakeetCtcWeights::synthesized`] fixture (SplitMix64 + Xavier) so
//!   shape / dtype / size flow can be exercised without the real HF
//!   checkpoint. Real-checkpoint parity is a follow-up wave gated on the
//!   real-checkpoint tensor-name manifest (T29-equivalent — the Moshi /
//!   CSM / Zonos / Kyutai STT / Parakeet-TDT pattern).
//! - [`ParakeetCtcAsr`] — engine handle carrying config + weights.
//!   [`ParakeetCtcAsr::transcribe`] returns [`VokraError::NotImplemented`]
//!   until real weights are bound (the real forward — 80-bin log-mel →
//!   FastConformer encoder → CTC vocab head → `ctc_decode_greedy(blank_id
//!   = pad_token_id)` → SentencePiece detokenize — is a follow-up wave
//!   gated on the real-checkpoint tensor manifest).
//!
//! # No ONNX (permanent)
//!
//! Parakeet-CTC ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet_ctc/`
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{LicenseClass, Result, VokraError};

/// `vokra.model.arch` a Parakeet-CTC GGUF must carry. Written by
/// `vokra-convert::models::parakeet_ctc::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `parakeet-ctc-1.1b` / `parakeet-ctc` /
/// `parakeet-ctc-1_1b` as
/// [`vokra_core::LicenseClass::AttributionRequired`] (CC-BY 4.0 — the
/// M2-13 gate passes commercially *and* the FR-MD-09 attribution surface
/// activates).
pub const EXPECTED_ARCH: &str = "parakeet-ctc";

/// PCM sample rate Parakeet-CTC expects. Not written in the upstream
/// `config.json`; taken from the model card (16 kHz mono `.wav` / `.flac`).
pub const PARAKEET_CTC_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// `vokra.parakeet_ctc.*` chunk-key mirrors — duplicated verbatim from the
// converter (`crates/vokra-convert/src/models/parakeet_ctc.rs`) so
// `vokra-models` does not gain a dependency edge onto `vokra-convert`.
// This is the same layered-convention rule sibling BF16 pass-through
// binders (`pyannote` / `snac` / `hifigan` / `beat_this` / `mt3`) use.
//
// Booleans (`attention_bias`, `convolution_bias`, `scale_input`) are
// stamped by the converter as `u32` via `u32::from(bool)` (0 / 1); the
// read side inverts with `!= 0`.
// ---------------------------------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.parakeet_ctc.sample_rate";

// Encoder (FastConformer)
const KEY_ENC_N_LAYER: &str = "vokra.parakeet_ctc.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.parakeet_ctc.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.parakeet_ctc.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.parakeet_ctc.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.parakeet_ctc.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.parakeet_ctc.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.parakeet_ctc.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str =
    "vokra.parakeet_ctc.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.parakeet_ctc.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.parakeet_ctc.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.parakeet_ctc.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.parakeet_ctc.arch.encoder.scale_input";

// CTC head + vocab
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.parakeet_ctc.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.parakeet_ctc.head.pad_token_id";

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
pub struct ParakeetCtcEncoderConfig {
    /// `num_hidden_layers` — 42 for CTC-1.1B.
    pub n_layer: usize,
    /// `hidden_size` — hidden width, 1024.
    pub d_model: usize,
    /// `num_attention_heads` — Q-heads, 8.
    pub n_head: usize,
    /// `num_key_value_heads` — KV-heads; 8 for 1.1B (MHA, no GQA
    /// broadcast). Kept as a field so a hypothetical future GQA flavor is
    /// representable without a new type.
    pub n_head_kv: usize,
    /// `intermediate_size` — FFN inner width, 4096 (the ~4× expansion of
    /// `d_model=1024`).
    pub ffn_dim: usize,
    /// `conv_kernel_size` — FastConformer depthwise convolution kernel
    /// size, 9. Must be odd for symmetric same-padding.
    pub conv_kernel_size: usize,
    /// `num_mel_bins` — log-mel channels on the input, **80 for CTC-1.1B**
    /// (distinct from the 128 the 0.6B TDT-v3 checkpoint uses).
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
    /// `attention_bias` — **true** for CTC-1.1B (Q/K/V/out projections
    /// carry biases — differs from Parakeet-TDT-0.6B-v3).
    pub attention_bias: bool,
    /// `convolution_bias` — false for CTC-1.1B (depthwise + point-wise
    /// convolutions are bias-free; upstream config does not list
    /// `convolution_bias` for CTC, so it inherits the encoder default of
    /// `false` — consistent with the NeMo `ConformerLayer` inference
    /// path).
    pub convolution_bias: bool,
    /// `scale_input` — **true** for CTC-1.1B (upstream `ParakeetEncoder`
    /// scales the subsample stem's input when this is on — differs from
    /// Parakeet-TDT-0.6B-v3).
    pub scale_input: bool,
}

impl ParakeetCtcEncoderConfig {
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
    /// `n_head_kv == n_head` (MHA — the Parakeet-CTC-1.1B case).
    #[must_use]
    pub fn kv_hidden(&self) -> usize {
        self.n_head_kv * self.head_dim()
    }
}

/// CTC head hparams (primary source: `vocab_size`, `pad_token_id`).
///
/// Unlike the Parakeet-TDT joint head, CTC has **no duration bins, no
/// joint projection, no RNN-T prediction network**. The head is a single
/// linear from encoder `d_model` to `vocab_size`, and the blank id doubles
/// as `pad_token_id` (the NeMo-canonical convention places the blank at
/// the tail of the head).
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetCtcHeadConfig {
    /// `vocab_size` — 1025 (1024 SentencePiece pieces + 1 blank at
    /// index 1024). The head therefore has output width `vocab_size`
    /// (blank inclusive).
    pub vocab_size: usize,
    /// `pad_token_id` — 1024. This is also the **CTC blank id**
    /// (NeMo convention). Passed straight into
    /// [`vokra_ops::ctc_decode_greedy`]'s `blank_id` parameter.
    pub pad_token_id: u32,
}

impl ParakeetCtcHeadConfig {
    /// The CTC blank id — an alias for [`Self::pad_token_id`] that reads
    /// as the term of art in the decoding call site.
    #[must_use]
    pub fn blank_id(&self) -> u32 {
        self.pad_token_id
    }
}

/// Resolved Parakeet-CTC hparam snapshot — every field is transcribed
/// from the upstream `config.json` (module docstring) or from the
/// preprocessor / model card (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParakeetCtcConfig {
    /// FastConformer encoder hparams.
    pub encoder: ParakeetCtcEncoderConfig,
    /// CTC head hparams (vocab + blank id).
    pub head: ParakeetCtcHeadConfig,
    /// PCM sample rate Parakeet-CTC expects — 16 000 (from the model card;
    /// **not** written in the upstream `config.json`).
    pub sample_rate: u32,
}

impl ParakeetCtcConfig {
    /// Primary-source Parakeet-CTC-1.1B config (every value transcribed
    /// from `huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`).
    #[must_use]
    pub fn parakeet_ctc_1_1b() -> Self {
        Self {
            encoder: ParakeetCtcEncoderConfig {
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
                convolution_bias: false,
                scale_input: true,
            },
            head: ParakeetCtcHeadConfig {
                vocab_size: 1025,
                pad_token_id: 1024,
            },
            sample_rate: PARAKEET_CTC_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA head split, even head_dim, blank at head tail,
    /// attention_bias + scale_input flags mirroring the real model) mirror
    /// the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            encoder: ParakeetCtcEncoderConfig {
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
                attention_bias: true,
                convolution_bias: false,
                scale_input: true,
            },
            head: ParakeetCtcHeadConfig {
                vocab_size: 5,
                pad_token_id: 4,
            },
            sample_rate: PARAKEET_CTC_SAMPLE_RATE,
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
                "parakeet-ctc config: encoder ill-formed \
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
                "parakeet-ctc config: encoder.n_layer must be > 0".to_owned(),
            ));
        }
        if self.encoder.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc config: encoder head_dim {} must be even \
                 (RoPE / rel-pos pairs)",
                self.encoder.head_dim(),
            )));
        }
        if self.encoder.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc config: encoder.ffn_dim must be > 0".to_owned(),
            ));
        }
        if self.encoder.in_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc config: encoder.in_dim (num_mel_bins) must be > 0".to_owned(),
            ));
        }
        if self.encoder.conv_kernel_size == 0 || self.encoder.conv_kernel_size % 2 == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc config: encoder.conv_kernel_size {} must be odd and > 0 \
                 (Conformer symmetric same-padding)",
                self.encoder.conv_kernel_size,
            )));
        }
        if self.encoder.subsampling_factor == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc config: encoder.subsampling_factor must be > 0 \
                 (FastConformer subsampling)"
                    .to_owned(),
            ));
        }
        if self.encoder.max_position_embeddings == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc config: encoder.max_position_embeddings must be > 0".to_owned(),
            ));
        }

        // ---- CTC head -----------------------------------------------------
        if self.head.vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc config: head.vocab_size must be > 0".to_owned(),
            ));
        }
        // The CTC blank id (= `pad_token_id`) lives inside the vocab head
        // width `[0, vocab_size)`; the NeMo convention puts the blank at
        // the tail (`blank = vocab_size - 1`).
        if (self.head.pad_token_id as usize) >= self.head.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc config: pad_token_id={} (= CTC blank) must be < vocab_size={}",
                self.head.pad_token_id, self.head.vocab_size,
            )));
        }
        Ok(())
    }

    /// Reads every `vokra.parakeet_ctc.*` chunk from `gguf`. Missing
    /// axis = loud [`VokraError::ModelLoad`] naming the absent key
    /// (FR-EX-08 — no primary-source constant fallback because a
    /// converter that fails to stamp an axis is a converter bug, not a
    /// runtime silent-default).
    ///
    /// Primary source for the axis table:
    /// `huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`
    /// (fetched 2026-07-24 by the converter, transcribed verbatim into
    /// [`Self::parakeet_ctc_1_1b`]).
    ///
    /// Booleans (`attention_bias`, `convolution_bias`, `scale_input`)
    /// are stamped by the converter as u32 (0 / 1); this reader
    /// inverts back to `bool` with `!= 0`, mirroring the Zonos / CSM /
    /// Kyutai STT / Parakeet-TDT convention.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any mandatory
    ///   `vokra.parakeet_ctc.*` u32 chunk is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(vokra_core::gguf::GgufMetadataValue::as_u64)
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "parakeet-ctc: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `nvidia/parakeet-ctc-1.1b` `config.json` (fetched \
                         2026-07-24) has every FastConformer axis; a converter that \
                         fails to stamp one is a converter bug, not a runtime silent-default \
                         (FR-EX-08). Re-run `vokra-cli convert --model parakeet-ctc` \
                         against `nvidia/parakeet-ctc-1.1b` safetensors."
                    ))
                })
        }
        Ok(Self {
            encoder: ParakeetCtcEncoderConfig {
                n_layer: req_u32(gguf, KEY_ENC_N_LAYER)? as usize,
                d_model: req_u32(gguf, KEY_ENC_D_MODEL)? as usize,
                n_head: req_u32(gguf, KEY_ENC_N_HEAD)? as usize,
                n_head_kv: req_u32(gguf, KEY_ENC_N_HEAD_KV)? as usize,
                ffn_dim: req_u32(gguf, KEY_ENC_FFN_DIM)? as usize,
                conv_kernel_size: req_u32(gguf, KEY_ENC_CONV_KERNEL)? as usize,
                in_dim: req_u32(gguf, KEY_ENC_IN_DIM)? as usize,
                subsampling_factor: req_u32(gguf, KEY_ENC_SUBSAMPLING_FACTOR)? as usize,
                subsampling_conv_kernel_size: req_u32(gguf, KEY_ENC_SUB_CONV_KERNEL)? as usize,
                subsampling_conv_stride: req_u32(gguf, KEY_ENC_SUB_CONV_STRIDE)? as usize,
                subsampling_conv_channels: req_u32(gguf, KEY_ENC_SUB_CONV_CHANNELS)? as usize,
                max_position_embeddings: req_u32(gguf, KEY_ENC_MAX_POS)? as usize,
                attention_bias: req_u32(gguf, KEY_ENC_ATTN_BIAS)? != 0,
                convolution_bias: req_u32(gguf, KEY_ENC_CONV_BIAS)? != 0,
                scale_input: req_u32(gguf, KEY_ENC_SCALE_INPUT)? != 0,
            },
            head: ParakeetCtcHeadConfig {
                vocab_size: req_u32(gguf, KEY_HEAD_VOCAB_SIZE)? as usize,
                pad_token_id: req_u32(gguf, KEY_HEAD_PAD_ID)?,
            },
            sample_rate: req_u32(gguf, KEY_SAMPLE_RATE)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-encoder-block scaffold weights (pre-norm Conformer FF1 / MHA / Conv
/// / FF2 branches). Same shape as the Parakeet-TDT encoder block; the CTC
/// checkpoint reuses the identical FastConformer body.
///
/// Field names mirror the upstream NeMo `ConformerLayer` module names.
///
/// # Attention biases (CTC-1.1B specific)
///
/// CTC-1.1B has `attention_bias = true` (unlike TDT-0.6B-v3), so this
/// scaffold carries the four projection biases as separate optional
/// vectors. `None` on any of them means the config disabled bias for that
/// projection (a future 0.6B-style variant); present + right-sized means
/// the bias participates. All four are `Some` in a real CTC-1.1B build.
#[derive(Debug, Clone)]
pub struct ParakeetCtcEncoderBlockWeights {
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
    /// Optional fused Q/K/V bias, shape `[3*d_model]`. Present iff
    /// `encoder.attention_bias == true` (the CTC-1.1B case).
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection, shape `[d_model, d_model]`.
    pub attn_out: Vec<f32>,
    /// Optional attention output bias, shape `[d_model]`. Present iff
    /// `encoder.attention_bias == true`.
    pub attn_out_bias: Option<Vec<f32>>,
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
///
/// When `scale_input = true` (CTC-1.1B), the runtime applies a fixed
/// `sqrt(d_model)` scale to the projected input before the first block
/// (mirrors the upstream `ParakeetEncoder.scale_input` branch). No extra
/// tensor is bound for the scale — it is a config-derived scalar.
#[derive(Debug, Clone)]
pub struct ParakeetCtcSubsampleWeights {
    /// `[d_model, factor * in_dim]`.
    pub linear_w: Vec<f32>,
    /// `[d_model]`.
    pub linear_b: Vec<f32>,
}

/// CTC head scaffold: a single Linear from encoder `d_model` to
/// `vocab_size`, plus a bias (the NeMo CTC decoder has `bias=True`).
#[derive(Debug, Clone)]
pub struct ParakeetCtcHeadWeights {
    /// `[d_model, vocab_size]` — CTC vocab projection (blank inclusive at
    /// index `pad_token_id`).
    pub vocab_head: Vec<f32>,
    /// `[vocab_size]` — CTC vocab bias.
    pub vocab_bias: Vec<f32>,
}

/// Parakeet-CTC weight store: subsample + encoder blocks + final norm +
/// CTC head.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a follow-up
/// (T29-equivalent — tensor-name manifest fetch from the upstream release).
#[derive(Debug, Clone)]
pub struct ParakeetCtcWeights {
    /// Subsample stem.
    pub subsample: ParakeetCtcSubsampleWeights,
    /// Encoder blocks in order.
    pub encoder_blocks: Vec<ParakeetCtcEncoderBlockWeights>,
    /// Encoder-out LayerNorm γ, shape `[d_model]`.
    pub encoder_final_norm: Vec<f32>,
    /// CTC head.
    pub head: ParakeetCtcHeadWeights,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint. Real-checkpoint bindings set this to `false`.
    pub is_synthesized: bool,
}

impl ParakeetCtcWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every LayerNorm γ starts at `1.0`; every bias starts at `0.0`.
    ///
    /// Attention biases (`qkv_bias`, `attn_out_bias`) are `Some` iff
    /// `config.encoder.attention_bias == true` (the CTC-1.1B case) — the
    /// runtime shape-check reads these back and refuses a mismatch
    /// (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ParakeetCtcConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let enc = &config.encoder;
        let head = &config.head;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let vocab = head.vocab_size;
        let k = enc.conv_kernel_size;
        let bias_on = enc.attention_bias;

        // Subsample stem — flat Linear + optional trailing norm (this
        // scaffold uses the plain Stacking projection, matching
        // `ConvSubsampleKind::Stacking`).
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let subsample = ParakeetCtcSubsampleWeights {
            linear_w: xavier(&mut rng, d_enc * projection_in, projection_in, d_enc),
            linear_b: vec![0.0; d_enc],
        };

        // Encoder blocks.
        let mut encoder_blocks = Vec::with_capacity(enc.n_layer);
        for _ in 0..enc.n_layer {
            encoder_blocks.push(ParakeetCtcEncoderBlockWeights {
                ff1_norm: vec![1.0; d_enc],
                ff1_fc1: xavier(&mut rng, d_enc * ffn, d_enc, ffn),
                ff1_fc2: xavier(&mut rng, ffn * d_enc, ffn, d_enc),
                attn_norm: vec![1.0; d_enc],
                qkv_proj: xavier(&mut rng, d_enc * 3 * d_enc, d_enc, 3 * d_enc),
                qkv_bias: bias_on.then(|| vec![0.0; 3 * d_enc]),
                attn_out: xavier(&mut rng, d_enc * d_enc, d_enc, d_enc),
                attn_out_bias: bias_on.then(|| vec![0.0; d_enc]),
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

        // CTC head — single Linear from d_enc to vocab_size with bias.
        let head_w = ParakeetCtcHeadWeights {
            vocab_head: xavier(&mut rng, d_enc * vocab, d_enc, vocab),
            vocab_bias: vec![0.0; vocab],
        };

        Ok(Self {
            subsample,
            encoder_blocks,
            encoder_final_norm,
            head: head_w,
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

/// Parakeet-CTC ASR engine handle.
///
/// Carries the resolved config and weight store. [`Self::transcribe`] is
/// the primary PCM → text entry point; until real weights are bound (see
/// the module docstring) it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty transcript).
///
/// # Weight license surfacing
///
/// The `weight_license` field carries the compliance class surfaced from
/// the GGUF's `vokra.provenance.weight_license` chunk (populated by
/// [`Self::from_gguf`]) or defaults to [`LicenseClass::AttributionRequired`]
/// under [`Self::new`] (the CC-BY 4.0 class that is the only legitimate
/// class for real Parakeet-CTC weights per the compliance registry —
/// `vokra_core::compliance::license_class` maps `parakeet-ctc` /
/// `parakeet-ctc-1.1b` to [`LicenseClass::AttributionRequired`]). The
/// M2-13 outer compliance gate does the strict enforcement; this handle
/// simply surfaces the class so callers can cross-check.
#[derive(Debug, Clone)]
pub struct ParakeetCtcAsr {
    cfg: ParakeetCtcConfig,
    weights: ParakeetCtcWeights,
    weight_license: LicenseClass,
}

impl ParakeetCtcAsr {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (encoder block count, per-tensor
    /// sizes, attention-bias presence) so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: ParakeetCtcConfig, weights: ParakeetCtcWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let enc = &cfg.encoder;
        let head = &cfg.head;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let vocab = head.vocab_size;
        let k = enc.conv_kernel_size;
        let projection_in = enc.subsampling_factor * enc.in_dim;
        let bias_on = enc.attention_bias;

        // Subsample stem.
        if weights.subsample.linear_w.len() != d_enc * projection_in {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: subsample.linear_w.len()={} != d_model * \
                 (subsampling_factor * in_dim) = {} * {} = {}",
                weights.subsample.linear_w.len(),
                d_enc,
                projection_in,
                d_enc * projection_in,
            )));
        }
        if weights.subsample.linear_b.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: subsample.linear_b.len()={} != d_model={}",
                weights.subsample.linear_b.len(),
                d_enc,
            )));
        }

        // Encoder blocks.
        if weights.encoder_blocks.len() != enc.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: encoder_blocks.len()={} != encoder.n_layer={}",
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
                        "parakeet-ctc weights: encoder block {i} `{name}` \
                         len={len} != {expected}",
                    )));
                }
            }
            // Attention bias presence + shape cross-check (CTC-1.1B has
            // attention_bias = true, so both must be Some with the exact
            // expected shape; a hypothetical bias-free variant sets both
            // to None). A mismatch is a loud error — no silent zero-fill,
            // no silent drop (FR-EX-08).
            match (bias_on, &blk.qkv_bias) {
                (true, Some(v)) if v.len() == 3 * d_enc => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} qkv_bias.len()={} \
                         != 3*d_model={}",
                        v.len(),
                        3 * d_enc,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} qkv_bias is None but \
                         attention_bias=true — a bias-free variant must set attention_bias=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} qkv_bias is Some but \
                         attention_bias=false — a bias-carrying variant must set attention_bias=true",
                    )));
                }
                (false, None) => {}
            }
            match (bias_on, &blk.attn_out_bias) {
                (true, Some(v)) if v.len() == d_enc => {}
                (true, Some(v)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} attn_out_bias.len()={} \
                         != d_model={}",
                        v.len(),
                        d_enc,
                    )));
                }
                (true, None) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} attn_out_bias is None but \
                         attention_bias=true — a bias-free variant must set attention_bias=false",
                    )));
                }
                (false, Some(_)) => {
                    return Err(VokraError::InvalidArgument(format!(
                        "parakeet-ctc weights: encoder block {i} attn_out_bias is Some but \
                         attention_bias=false — a bias-carrying variant must set attention_bias=true",
                    )));
                }
                (false, None) => {}
            }
        }
        if weights.encoder_final_norm.len() != d_enc {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: encoder_final_norm.len()={} != d_model={}",
                weights.encoder_final_norm.len(),
                d_enc,
            )));
        }

        // CTC head.
        if weights.head.vocab_head.len() != d_enc * vocab {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: head.vocab_head.len()={} != d_enc * vocab = {} * {} = {}",
                weights.head.vocab_head.len(),
                d_enc,
                vocab,
                d_enc * vocab,
            )));
        }
        if weights.head.vocab_bias.len() != vocab {
            return Err(VokraError::InvalidArgument(format!(
                "parakeet-ctc weights: head.vocab_bias.len()={} != vocab_size={}",
                weights.head.vocab_bias.len(),
                vocab,
            )));
        }

        Ok(Self {
            cfg,
            weights,
            // Default weight-license class under `new()` mirrors the
            // compliance registry (`vokra_core::compliance::license_class`
            // maps `parakeet-ctc` / `parakeet-ctc-1.1b` to CC-BY 4.0 =
            // AttributionRequired). `from_gguf` overrides with whatever
            // the provenance chunk carries (or `Unknown` if absent).
            weight_license: LicenseClass::AttributionRequired,
        })
    }

    /// Binds a Parakeet-CTC GGUF: validates arch, reads the strict
    /// `vokra.parakeet_ctc.*` topology chunk group, builds a
    /// deterministic synthesized weight fixture matching the resolved
    /// config, and surfaces the stamped weight-license class for
    /// compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Loud-partial contract
    ///
    /// After this returns `Ok(_)`, the resulting engine is a
    /// **synthesized-weight** handle — the shape / dtype / size flow is
    /// exercised end-to-end (config chunk validation, weight-store
    /// construction, PCM boundary check), but calling
    /// [`Self::transcribe`] still returns [`VokraError::NotImplemented`]
    /// naming the real-checkpoint tensor-name manifest binding
    /// (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT /
    /// Parakeet-TDT pattern) as the follow-up wave's anchor. The
    /// primitives named in that message ([`vokra_ops::conformer`] +
    /// [`vokra_ops::ctc_decode`]) already exist; the missing piece is
    /// the HF safetensors tensor-name → [`ParakeetCtcWeights`] mapping
    /// plus SentencePiece detokenize (model-specific, not a shared op).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"parakeet-ctc"` (a `parakeet-tdt` GGUF handed to us by
    ///   mistake fails with a hint pointing at the TDT binder, matching
    ///   the sibling-arch disambiguation pattern used by
    ///   `Mt3::from_gguf` and `Snac::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.parakeet_ctc.*` chunk
    ///   is absent ([`ParakeetCtcConfig::from_gguf`] is strict).
    /// - [`VokraError::InvalidArgument`] from
    ///   [`ParakeetCtcConfig::validate_for_forward`] +
    ///   [`ParakeetCtcAsr::new`] shape gates.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    "vokra.parakeet_ctc.arch.encoder.n_layer missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            // Wave C1 (2026-08-15): `parakeet-tdt-1_1b` (UNDERSCORE) added —
            // that is the spelling `vokra-convert::models::parakeet_tdt_1_1b`
            // actually stamps, so before this the dedicated TDT hint never
            // fired for a real 1.1B TDT artifact (it fell through to the
            // generic `Some(other)` arm). The dotted `parakeet-tdt-1.1b` is
            // retained: it is the model NAME spelling and a plausible
            // hand-authored value.
            Some("parakeet-tdt")
            | Some("parakeet-tdt-0.6b-v3")
            | Some("parakeet-tdt-1.1b")
            | Some("parakeet-tdt-1_1b") => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-ctc: GGUF arch is a Parakeet-TDT variant (RNN-T + TDT \
                     joint / duration head), expected `{EXPECTED_ARCH}` (FastConformer \
                     + single CTC vocab head). These are different topologies — TDT \
                     has a prediction network + joint projection + duration bins that \
                     the CTC binder cannot dispatch. Route the GGUF through the \
                     sibling `parakeet::ParakeetAsr::from_gguf` TDT binder \
                     (`crates/vokra-models/src/parakeet/mod.rs`) instead — or, for the \
                     1.1B TDT SKU (arch `parakeet-tdt-1_1b`), through \
                     `parakeet_tdt_1_1b::ParakeetTdt11b::from_gguf` \
                     (`crates/vokra-models/src/parakeet_tdt_1_1b/mod.rs`)."
                )));
            }
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "parakeet-ctc: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model parakeet-ctc`? \
                     Sibling ASR arches — `whisper`, `voxtral`, `canary`, \
                     `parakeet-tdt` — are completely different topologies)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "parakeet-ctc: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native parakeet-ctc GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.parakeet_ctc.*` chunk
        //    group.
        let cfg = ParakeetCtcConfig::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license class
        //    for compliance gate cross-checks (defaults to `Unknown` if
        //    absent, which is fail-closed at the outer M2-13 gate).
        //    Matches the MT3 / SNAC precedent — surface the class here,
        //    let the outer gate do the strict enforcement.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        // 4. Build synthesized weights against the freshly-read config
        //    so the engine is constructible. `transcribe` still loud-
        //    partials with the synthesized-weight blocker message —
        //    binding real HF checkpoint tensor names is the follow-up
        //    wave (T29-equivalent).
        let weights = ParakeetCtcWeights::synthesized(&cfg, /* seed */ 0)?;
        let mut asr = Self::new(cfg, weights)?;
        asr.weight_license = weight_license;
        Ok(asr)
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. For real Parakeet-CTC
    /// checkpoints the compliance registry
    /// (`vokra_core::compliance::license_class`) maps `parakeet-ctc` /
    /// `parakeet-ctc-1.1b` to [`LicenseClass::AttributionRequired`]
    /// (CC-BY 4.0). A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed at the outer M2-13 gate);
    /// [`Self::new`] defaults to [`LicenseClass::AttributionRequired`]
    /// (the only legitimate class for real weights).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ParakeetCtcConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`ParakeetCtcWeights::synthesized`] (never a real upstream
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
    /// Callers verify the shape flow through [`ParakeetCtcAsr::new`] +
    /// [`ParakeetCtcWeights::synthesized`] today; a follow-up wave binds
    /// the real HF checkpoint tensor names and wires the forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "parakeet-ctc transcribe: pcm slice is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "parakeet-ctc transcribe: this engine holds synthesized weights \
                 (deterministic fixture from ParakeetCtcWeights::synthesized) — \
                 synthesized-weight text would be a hallucinated sequence, \
                 not a real transcript. Bind real Parakeet-CTC-1.1B \
                 weights (CC-BY 4.0, nvidia/parakeet-ctc-1.1b) before \
                 invoking transcribe. The shape flow (config validation, \
                 weight-store construction, PCM boundary check) is exercised \
                 through ParakeetCtcAsr::new; the real-checkpoint tensor-name \
                 manifest lands in a follow-up wave (T29-equivalent — the \
                 Moshi / CSM / Zonos / Kyutai STT / Parakeet-TDT pattern).",
            ));
        }
        Err(VokraError::NotImplemented(
            "parakeet-ctc transcribe: real weights are bound but the log-mel \
             front-end → FastConformer encoder (vokra_ops::conformer) → \
             CTC vocab head → ctc_decode_greedy(blank_id = head.pad_token_id) \
             → SentencePiece detokenize forward path has not landed yet. \
             Follow-up wave: wire ParakeetCtcWeights to \
             vokra_ops::conformer::ConformerEncoder + the CTC vocab head + \
             vokra_ops::ctc_decode_greedy with blank_id = head.blank_id() \
             (= head.pad_token_id).",
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
    /// (`huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`)
    /// verbatim.
    #[test]
    fn parakeet_ctc_1_1b_matches_primary_source_config_json() {
        let c = ParakeetCtcConfig::parakeet_ctc_1_1b();
        // Encoder.
        assert_eq!(c.encoder.n_layer, 42);
        assert_eq!(c.encoder.d_model, 1024);
        assert_eq!(c.encoder.n_head, 8);
        assert_eq!(c.encoder.n_head_kv, 8);
        assert_eq!(c.encoder.ffn_dim, 4096);
        assert_eq!(c.encoder.conv_kernel_size, 9);
        // NOTE: 80 (not 128 — the CTC-1.1B checkpoint uses the classic
        // 80-bin front-end, whereas TDT-0.6B-v3 uses 128).
        assert_eq!(c.encoder.in_dim, 80);
        assert_eq!(c.encoder.subsampling_factor, 8);
        assert_eq!(c.encoder.subsampling_conv_kernel_size, 3);
        assert_eq!(c.encoder.subsampling_conv_stride, 2);
        assert_eq!(c.encoder.subsampling_conv_channels, 256);
        assert_eq!(c.encoder.max_position_embeddings, 5000);
        // NOTE: attention_bias = true (differs from TDT-0.6B-v3 = false).
        assert!(c.encoder.attention_bias);
        assert!(!c.encoder.convolution_bias);
        // NOTE: scale_input = true (differs from TDT-0.6B-v3 = false).
        assert!(c.encoder.scale_input);
        // CTC head.
        assert_eq!(c.head.vocab_size, 1025);
        assert_eq!(c.head.pad_token_id, 1024);
        assert_eq!(c.head.blank_id(), 1024);
        // Audio boundary (model card).
        assert_eq!(c.sample_rate, 16_000);
        // Derived.
        assert_eq!(c.encoder.head_dim(), 128);
        assert_eq!(c.encoder.kv_hidden(), 1024); // MHA
        // Everything above adds up to a well-formed config.
        c.validate_for_forward()
            .expect("parakeet-ctc-1.1b is well-formed");
    }

    /// Guards the axis that distinguishes CTC-1.1B from TDT-0.6B-v3:
    /// `attention_bias = true` and `scale_input = true`. Getting these
    /// wrong at conversion time would silently drop the bias vectors or
    /// misread the runtime scale — a class of regression the CI should
    /// catch on sight.
    #[test]
    fn ctc_1_1b_differs_from_tdt_0_6b_v3_on_bias_and_scale_and_mel_bins() {
        let c = ParakeetCtcConfig::parakeet_ctc_1_1b();
        assert!(
            c.encoder.attention_bias,
            "CTC-1.1B: attention_bias must be true"
        );
        assert!(c.encoder.scale_input, "CTC-1.1B: scale_input must be true");
        assert_eq!(
            c.encoder.in_dim, 80,
            "CTC-1.1B: num_mel_bins must be 80 (TDT-0.6B-v3 uses 128)"
        );
        assert_eq!(
            c.encoder.n_layer, 42,
            "CTC-1.1B: num_hidden_layers must be 42 (TDT-0.6B-v3 uses 24)"
        );
    }

    #[test]
    fn tiny_config_is_well_formed() {
        ParakeetCtcConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_head_split_ill_formed_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.encoder.d_model = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_gqa_broadcast_not_dividing_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
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
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.n_layer = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_ffn_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.ffn_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_in_dim_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.in_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_even_conv_kernel_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.conv_kernel_size = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_subsampling_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.subsampling_factor = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_max_positions_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.max_position_embeddings = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_vocab_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.head.vocab_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_blank_out_of_range_is_rejected() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.head.pad_token_id = c.head.vocab_size as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let w1 = ParakeetCtcWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = ParakeetCtcWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.subsample.linear_w, w2.subsample.linear_w);
        assert_eq!(w1.encoder_blocks[0].qkv_proj, w2.encoder_blocks[0].qkv_proj);
        assert_eq!(w1.head.vocab_head, w2.head.vocab_head);
        assert!(w1.is_synthesized);

        // Shape flow.
        let enc = &c.encoder;
        let head = &c.head;
        let d_enc = enc.d_model;
        let ffn = enc.ffn_dim;
        let vocab = head.vocab_size;
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
            // Attention biases: Some with the right shape iff
            // attention_bias == true (the CTC-1.1B case; the tiny config
            // also has attention_bias == true).
            assert!(blk.qkv_bias.is_some(), "qkv_bias present when bias=true");
            assert_eq!(blk.qkv_bias.as_ref().unwrap().len(), 3 * d_enc);
            assert!(
                blk.attn_out_bias.is_some(),
                "attn_out_bias present when bias=true"
            );
            assert_eq!(blk.attn_out_bias.as_ref().unwrap().len(), d_enc);
        }
        assert_eq!(w1.encoder_final_norm.len(), d_enc);
        // CTC head.
        assert_eq!(w1.head.vocab_head.len(), d_enc * vocab);
        assert_eq!(w1.head.vocab_bias.len(), vocab);
    }

    /// A bias-free variant (a hypothetical future CTC config with
    /// `attention_bias=false`) drops both biases to None — the
    /// synthesized builder must respect the flag, and the runtime must
    /// accept the resulting None pair.
    #[test]
    fn synthesized_weights_respect_attention_bias_off() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.attention_bias = false;
        let w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        for blk in &w.encoder_blocks {
            assert!(
                blk.qkv_bias.is_none(),
                "qkv_bias must be None when bias=false"
            );
            assert!(
                blk.attn_out_bias.is_none(),
                "attn_out_bias must be None when bias=false"
            );
        }
        // And the runtime accepts the bias-free pair.
        ParakeetCtcAsr::new(c, w).expect("bias-free variant is loadable");
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let w_a = ParakeetCtcWeights::synthesized(&c, 1).expect("build a");
        let w_b = ParakeetCtcWeights::synthesized(&c, 2).expect("build b");
        assert_ne!(w_a.subsample.linear_w, w_b.subsample.linear_w);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            ParakeetCtcWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_matching_config_and_weights() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetCtcAsr::new(c.clone(), w).expect("parakeet-ctc asr");
        assert_eq!(asr.config().encoder.d_model, c.encoder.d_model);
        assert!(asr.is_synthesized());
    }

    #[test]
    fn asr_new_rejects_encoder_layer_count_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_tensor_size_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_proj.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_subsample_size_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.subsample.linear_w.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_encoder_final_norm_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_final_norm.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_head_vocab_head_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.head.vocab_head.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_head_vocab_bias_mismatch() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.head.vocab_bias.pop();
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// `attention_bias=true` (CTC-1.1B) requires the bias vectors to be
    /// present and correctly shaped — dropping either raises a loud
    /// `InvalidArgument`, not a silent zero-fill (FR-EX-08).
    #[test]
    fn asr_new_rejects_missing_qkv_bias_when_attention_bias_on() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].qkv_bias = None;
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_wrong_size_qkv_bias() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        if let Some(v) = w.encoder_blocks[0].qkv_bias.as_mut() {
            v.pop();
        }
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_missing_attn_out_bias_when_attention_bias_on() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        w.encoder_blocks[0].attn_out_bias = None;
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_rejects_present_qkv_bias_when_attention_bias_off() {
        // Build a bias-free variant, then splice in a stray bias vector —
        // the runtime must refuse it (a bias-carrying variant must set
        // attention_bias=true so the runtime knows to use them).
        let mut c = ParakeetCtcConfig::tiny_for_tests();
        c.encoder.attention_bias = false;
        let mut w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        // Inject an unexpected bias.
        w.encoder_blocks[0].qkv_bias = Some(vec![0.0; 3 * c.encoder.d_model]);
        assert!(matches!(
            ParakeetCtcAsr::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = ParakeetCtcConfig::tiny_for_tests();
        let w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetCtcAsr::new(c, w).expect("parakeet-ctc asr");
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
        let c = ParakeetCtcConfig::tiny_for_tests();
        let w = ParakeetCtcWeights::synthesized(&c, 7).expect("weights");
        let asr = ParakeetCtcAsr::new(c, w).expect("parakeet-ctc asr");
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
    fn expected_arch_is_parakeet_ctc() {
        assert_eq!(EXPECTED_ARCH, "parakeet-ctc");
    }

    #[test]
    fn sample_rate_matches_model_card_boundary() {
        // 16 kHz — per the model card (`.wav` / `.flac` mono @ 16 kHz).
        assert_eq!(PARAKEET_CTC_SAMPLE_RATE, 16_000);
    }

    // -----------------------------------------------------------------------
    // Wave 4: `from_gguf` loud-partial contract (real config validation,
    // arch + provenance surface, license class round-trip, engine
    // constructibility from GGUF, `transcribe` still loud-partials on the
    // synthesized-weight blocker so a follow-up wave has exactly one place
    // to walk — mirror of MT3 / SNAC / vocos / bigvgan wave-3 precedent).
    // -----------------------------------------------------------------------

    /// Builds a minimal Parakeet-CTC GGUF carrying the arch tag + full
    /// `vokra.parakeet_ctc.*` chunk group matching the primary-source
    /// CTC-1.1B config. `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn parakeet_ctc_gguf(
        cfg: &ParakeetCtcConfig,
        weight_license_class: Option<LicenseClass>,
    ) -> vokra_core::gguf::GgufFile {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "parakeet-ctc-1.1b");
        // Chunk group — mirrors the converter (`write_hparams`).
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Encoder
        b.add_u32(KEY_ENC_N_LAYER, cfg.encoder.n_layer as u32);
        b.add_u32(KEY_ENC_D_MODEL, cfg.encoder.d_model as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.n_head as u32);
        b.add_u32(KEY_ENC_N_HEAD_KV, cfg.encoder.n_head_kv as u32);
        b.add_u32(KEY_ENC_FFN_DIM, cfg.encoder.ffn_dim as u32);
        b.add_u32(KEY_ENC_CONV_KERNEL, cfg.encoder.conv_kernel_size as u32);
        b.add_u32(KEY_ENC_IN_DIM, cfg.encoder.in_dim as u32);
        b.add_u32(
            KEY_ENC_SUBSAMPLING_FACTOR,
            cfg.encoder.subsampling_factor as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_KERNEL,
            cfg.encoder.subsampling_conv_kernel_size as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_STRIDE,
            cfg.encoder.subsampling_conv_stride as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_CHANNELS,
            cfg.encoder.subsampling_conv_channels as u32,
        );
        b.add_u32(KEY_ENC_MAX_POS, cfg.encoder.max_position_embeddings as u32);
        b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(cfg.encoder.attention_bias));
        b.add_u32(KEY_ENC_CONV_BIAS, u32::from(cfg.encoder.convolution_bias));
        b.add_u32(KEY_ENC_SCALE_INPUT, u32::from(cfg.encoder.scale_input));
        // Head
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.vocab_size as u32);
        b.add_u32(KEY_HEAD_PAD_ID, cfg.head.pad_token_id);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A `whisper` / `voxtral` / `parakeet-tdt` GGUF handed to the
    /// Parakeet-CTC binder by mistake must fail loud with a specific
    /// message rather than silently mis-binding (FR-EX-08). The TDT case
    /// gets a dedicated hint pointing at the sibling `ParakeetAsr`
    /// binder.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        // Generic wrong arch — names both got + expected + sibling
        // ASR arches.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ParakeetCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`whisper`") && m.contains("`parakeet-ctc`"),
                    "message must name both got + expected arch tags, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // Parakeet-TDT sibling — dedicated hint pointing at the TDT
        // binder so a reader diagnosing this mis-route has exactly one
        // place to walk.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "parakeet-tdt");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ParakeetCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on parakeet-tdt");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("Parakeet-TDT") && m.contains("parakeet::ParakeetAsr"),
                    "message must name the TDT sibling + point at the TDT binder, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// A GGUF that omits `vokra.model.arch` entirely fails loud
    /// (converter did not stamp it — the GGUF is not Vokra-native).
    #[test]
    fn from_gguf_rejects_missing_arch() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        // No arch chunk at all — but we need at least one metadata key
        // to build a valid GGUF, so include a benign name.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "unknown");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ParakeetCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`") && m.contains("did not stamp"),
                    "message must name the missing arch key + fingerprint the converter, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// Every mandatory `vokra.parakeet_ctc.*` chunk is required — a
    /// converter that fails to stamp any one is a converter bug, not a
    /// runtime silent-default (FR-EX-08). The loud error names the
    /// exact absent chunk key.
    #[test]
    fn from_gguf_rejects_missing_encoder_axis() {
        use vokra_core::gguf::{GgufBuilder, GgufFile};

        let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        // Deliberately omit KEY_ENC_N_LAYER — every other encoder axis
        // is stamped so the loud error must fire on `n_layer`.
        b.add_u32(KEY_ENC_D_MODEL, cfg.encoder.d_model as u32);
        b.add_u32(KEY_ENC_N_HEAD, cfg.encoder.n_head as u32);
        b.add_u32(KEY_ENC_N_HEAD_KV, cfg.encoder.n_head_kv as u32);
        b.add_u32(KEY_ENC_FFN_DIM, cfg.encoder.ffn_dim as u32);
        b.add_u32(KEY_ENC_CONV_KERNEL, cfg.encoder.conv_kernel_size as u32);
        b.add_u32(KEY_ENC_IN_DIM, cfg.encoder.in_dim as u32);
        b.add_u32(
            KEY_ENC_SUBSAMPLING_FACTOR,
            cfg.encoder.subsampling_factor as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_KERNEL,
            cfg.encoder.subsampling_conv_kernel_size as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_STRIDE,
            cfg.encoder.subsampling_conv_stride as u32,
        );
        b.add_u32(
            KEY_ENC_SUB_CONV_CHANNELS,
            cfg.encoder.subsampling_conv_channels as u32,
        );
        b.add_u32(KEY_ENC_MAX_POS, cfg.encoder.max_position_embeddings as u32);
        b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(cfg.encoder.attention_bias));
        b.add_u32(KEY_ENC_CONV_BIAS, u32::from(cfg.encoder.convolution_bias));
        b.add_u32(KEY_ENC_SCALE_INPUT, u32::from(cfg.encoder.scale_input));
        b.add_u32(KEY_HEAD_VOCAB_SIZE, cfg.head.vocab_size as u32);
        b.add_u32(KEY_HEAD_PAD_ID, cfg.head.pad_token_id);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        let Err(err) = ParakeetCtcAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing encoder axis");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_ENC_N_LAYER),
                    "message must name the exact missing chunk key `{KEY_ENC_N_LAYER}`, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    /// The full CTC-1.1B primary-source config round-trips: stamp every
    /// chunk with the transcribed values, read them back with
    /// `from_gguf`, assert every field of the resulting
    /// `ParakeetCtcConfig` equals `parakeet_ctc_1_1b()`.
    #[test]
    fn from_gguf_reads_full_ctc_1_1b_config() {
        let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
        let file = parakeet_ctc_gguf(&cfg, None);
        let round_trip = ParakeetCtcConfig::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            round_trip, cfg,
            "every field of the resolved config must round-trip verbatim"
        );
    }

    /// A GGUF carrying `vokra.provenance.weight_license = "attribution-required"`
    /// (the CC-BY 4.0 class the Parakeet-CTC converter stamps) surfaces
    /// back through `Self::weight_license()` — the outer M2-13 gate can
    /// then enforce.
    #[test]
    fn from_gguf_surfaces_stamped_attribution_required() {
        let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
        let file = parakeet_ctc_gguf(&cfg, Some(LicenseClass::AttributionRequired));
        let asr = ParakeetCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::AttributionRequired,
            "CC-BY 4.0 = AttributionRequired must surface"
        );
    }

    /// A GGUF that omits `vokra.provenance.weight_license` reads back
    /// as `LicenseClass::Unknown` (fail-closed at the outer M2-13 gate,
    /// matching MT3 / SNAC precedent).
    #[test]
    fn from_gguf_defaults_missing_provenance_to_unknown() {
        let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
        let file = parakeet_ctc_gguf(&cfg, None);
        let asr = ParakeetCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Unknown,
            "missing provenance must default to Unknown (fail-closed at outer gate)"
        );
    }

    /// After a full CTC-1.1B GGUF round-trip, `transcribe` still
    /// returns `NotImplemented` naming the synthesized-weight blocker
    /// (loud-partial contract preserved — the follow-up wave binds real
    /// HF checkpoint tensor names via `tools/parity/
    /// parakeet_ctc_prepare_checkpoint.py`, T29-equivalent).
    #[test]
    fn from_gguf_engine_transcribe_is_loud_not_implemented() {
        let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
        let file = parakeet_ctc_gguf(&cfg, Some(LicenseClass::AttributionRequired));
        let asr = ParakeetCtcAsr::from_gguf(&file).expect("valid GGUF must bind");
        // Round-tripped engine is still synthesized-weight; the primary
        // NotImplemented path fires (not the "real weights bound but
        // forward path not landed" path — because the follow-up wave
        // will replace the synthesized weights with real ones and flip
        // the message).
        assert!(asr.is_synthesized(), "from_gguf builds synthesized weights");

        // 1 second of 16 kHz mono silence — legitimate input shape, so
        // the loud-partial gate fires (not the empty-pcm gate).
        let pcm = vec![0.0f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized")
                        && (msg.contains("nvidia/parakeet-ctc-1.1b")
                            || msg.contains("real Parakeet-CTC-1.1B")),
                    "message must name the synthesized-weight blocker + primary-source anchor: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }
}
