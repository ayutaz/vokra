//! **ICTNLP LLaMA-Omni2** — streaming speech-to-speech (Qwen2.5 backbone +
//! Whisper-family speech encoder + AR speech decoder). Coverage-audit
//! Wave B fast-track post-audit CC-gap 2026-08-14.
//!
//! # What LLaMA-Omni2 is (primary source)
//!
//! LLaMA-Omni2 is a **streaming S2S** family from ICTNLP (ACL 2025). Each
//! variant pairs three stages:
//!
//! 1. **Speech encoder** — Whisper-family (log-mel → contextual audio
//!    embeddings) that projects into the LM residual width.
//! 2. **Text backbone** — Qwen2.5-family decoder-only transformer (RoPE
//!    + SwiGLU + RMSNorm), the same family Voxtral / Canary-Qwen /
//!      Kyutai STT / FireRedASR-LLM-L would share as `vokra_ops::qwen2`,
//!      which is a PROPOSED op name rather than a landed one. No such
//!      module exists today; every Qwen2 sibling re-implements MHA +
//!      GEMM + LayerNorm inline, and a follow-up wave consolidates them.
//! 3. **Speech decoder** — streaming AR head that emits audio tokens /
//!    frames back to the caller (the streaming session infrastructure
//!    the Moshi / CSM full-duplex sibling family exercises).
//!
//! The 195 ms latency target puts LLaMA-Omni2 on the streaming-first
//! side of the S2S family neighbourhood (Moshi / CSM). Four sibling
//! HF repos ship as distinct scales — see [`LlamaOmni2Variant`].
//!
//! # Real-forward posture — loud-partial
//!
//! This scaffold lands the shape / provenance / license bindings so a
//! future wave (T29-equivalent — the CosyVoice2 T02 / CSM T29 / Moshi
//! T29 / Kyutai STT precedent) can flip the switch on the real
//! streaming AR forward without changing the surface. Until then,
//! [`LlamaOmni2::converse`] returns [`VokraError::UnsupportedOp`] with a
//! message that names the missing primitives (shared Qwen2.5 forward,
//! Whisper-style speech encoder forward, streaming AR speech decoder,
//! streaming session infrastructure) and the primary source URLs
//! (`huggingface.co/ICTNLP/LLaMA-Omni2-7B` + siblings +
//! `github.com/ictnlp/LLaMA-Omni2`). Never a silent zero-fill / noise
//! stream (FR-EX-08).
//!
//! # No per-variant primary-source hparam constants (yet)
//!
//! CLAUDE.md「ハルシネーション厳禁」: the scaffold does **not** hardcode
//! per-variant `n_layer` / `d_model` / `n_head` / `vocab` / adapter dims.
//! Every hparam is transcribed from the upstream `config.json` at
//! **convert time** and stamped into the GGUF; the runtime reads them
//! back verbatim through [`LlamaOmni2Config::from_gguf`]. Per-variant
//! primary-source constants (e.g. `llama_omni2_7b()`) land in a future
//! wave once the owner fetches each variant's `config.json` on vast.ai
//! and records the JSON in the audit ticket — this mirrors the exact
//! posture Kyutai STT / Voxtral / CanaryQwen kept before their real
//! per-variant transcribed constants shipped.
//!
//! # ELVIS Act posture
//!
//! LLaMA-Omni2 emits speech with a **fixed decoder voice** (task-
//! oriented S2S: the caller does not supply a target speaker prompt at
//! inference time — the same posture the sibling `voxtral` /
//! `canary_qwen` / `kyutai_stt` speech models keep). This is **not** a
//! voice-cloning trigger model in the sense of CLAUDE.md 設計判断 8 /
//! [ELVIS Act], so it stays in the main `vokra` repo rather than the
//! `vokra-voiceclone-experimental` fork. The `docs/license-audit.md`
//! §3.1 row still routes through owner sign-off (BLANK, fail-closed)
//! and the owner independently ratifies the ELVIS Act posture before
//! ticking Commercial (memory
//! `[[feedback-license-signoff-primary-source]]`).
//!
//! # Shared primitives (deferred)
//!
//! Real-weight binding depends on primitives that unlock voxtral /
//! canary_qwen / kyutai_stt / firered_asr_llm_l simultaneously:
//!
//! - `vokra_ops::qwen2` — shared RoPE / SwiGLU / RMSNorm forward (each
//!   Qwen2 sibling currently re-implements attention + FFN inline; the
//!   follow-up wave consolidates them into a single module).
//! - Whisper-family speech encoder forward — shared with voxtral /
//!   canary_qwen / kyutai_stt.
//! - Streaming session infrastructure — shared with Moshi / CSM full-
//!   duplex (`crates/vokra-models/src/moshi/` + `csm/` session state
//!   management for per-step KV cache / streaming send / receive).

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::rng::SplitMix64;
use vokra_core::{CompliancePolicy, Result, VokraError, check_weight_license};

/// `vokra.model.arch` value a LLaMA-Omni2 GGUF must carry. Written by
/// `vokra-convert::models::llama_omni2::ARCH`. Intentionally distinct
/// from every sibling Qwen2-family arch (`voxtral` / `canary_qwen` /
/// `kyutai_stt` / `firered_asr_llm_l`) — silently sharing an arch tag
/// would misroute the runtime dispatch (FR-EX-08).
pub const EXPECTED_ARCH: &str = "llama_omni2";

/// PCM sample rate LLaMA-Omni2 expects at the Whisper-family speech
/// encoder boundary. Whisper-family encoders universally consume
/// 16 kHz log-mel input (the CLAUDE.md §M0/M1 Whisper family + the
/// sibling Voxtral / Canary / Canary-Qwen / distil-Whisper / Kotoba-
/// Whisper / Moonshine speech encoders all key on 16 kHz). Documented
/// on the field so a future owner-verified per-variant primary-source
/// constant can override if any variant's `config.json` diverges.
pub const LLAMA_OMNI2_SAMPLE_RATE: u32 = 16_000;

/// Deterministic seed [`LlamaOmni2::from_gguf_with_policy`] threads into
/// [`LlamaOmni2Weights::synthesized`] until the real-checkpoint tensor-
/// name manifest lands (T29-equivalent — the Kyutai STT
/// [`KYUTAI_STT_FROM_GGUF_DEFAULT_SEED`](super::kyutai_stt::KYUTAI_STT_FROM_GGUF_DEFAULT_SEED)
/// pattern). Fixed so every `from_gguf` build against the same shape
/// config produces bit-identical weight bytes → reproducible bug
/// reports.
pub const LLAMA_OMNI2_FROM_GGUF_DEFAULT_SEED: u64 = 0x_10AD_11AD_10AD_11AD;

// ---------------------------------------------------------------------------
// `vokra.llama_omni2.*` metadata keys
// ---------------------------------------------------------------------------
//
// These strings mirror the offline converter
// (`vokra-convert::models::llama_omni2`) verbatim; the two crates only
// share `vokra-core`, so the string constants are the sole handshake
// (the cross-crate pattern established by CSM / CosyVoice2 / Kokoro /
// Dia / Zonos / KyutaiSTT — see this module docstring and the sibling
// `kyutai_stt/mod.rs` for the same layout).

const KEY_VARIANT: &str = "vokra.llama_omni2.variant";
const KEY_SAMPLE_RATE: &str = "vokra.llama_omni2.sample_rate";

// Backbone (Qwen2.5-family decoder-only transformer)
const KEY_BB_N_LAYER: &str = "vokra.llama_omni2.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.llama_omni2.arch.backbone.d_model";
const KEY_BB_N_HEAD: &str = "vokra.llama_omni2.arch.backbone.n_head";
const KEY_BB_VOCAB: &str = "vokra.llama_omni2.arch.backbone.vocab";
const KEY_BB_INTERMEDIATE_SIZE: &str = "vokra.llama_omni2.arch.backbone.intermediate_size";
const KEY_BB_ROPE_MAX_PERIOD: &str = "vokra.llama_omni2.arch.backbone.rope_max_period";
const KEY_BB_RMS_NORM_EPS: &str = "vokra.llama_omni2.arch.backbone.rms_norm_eps";

// Speech encoder / speech decoder dim
const KEY_ENC_DIM: &str = "vokra.llama_omni2.arch.speech_encoder.dim";
const KEY_DEC_DIM: &str = "vokra.llama_omni2.arch.speech_decoder.dim";

// ---------------------------------------------------------------------------
// Variant tag (runtime side — mirrors converter enum)
// ---------------------------------------------------------------------------

/// Runtime tag for the four LLaMA-Omni2 sibling releases. Kept parallel
/// to the converter-side [`LlamaOmni2Variant`](super::super::super::vokra_convert::LlamaOmni2Variant)
/// enum discriminator — the two crates only share `vokra-core`, so the
/// wire tag string is the handshake.
///
/// Silently sharing dispatch across variants is safe for the tokenizer
/// axis (all four inherit the Qwen2.5 tokenizer family) but not for the
/// LM axis (each variant retunes `n_layer` / `d_model` / `intermediate_size`)
/// nor the speech decoder axis (each variant retunes the decoder width
/// to match the LM width per the audit ticket). Route on this tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlamaOmni2Variant {
    /// `ICTNLP/LLaMA-Omni2-7B` — 7B, ~14 GB BF16.
    #[default]
    _7B,
    /// `ICTNLP/LLaMA-Omni2-3B-Bilingual` — 3B, ~6 GB BF16, EN + ZH.
    _3BBilingual,
    /// `ICTNLP/LLaMA-Omni2-1.5B` — 1.5B, ~3 GB BF16, edge-fit.
    _1_5B,
    /// `ICTNLP/LLaMA-Omni2-32B` — 32B, ~64 GB BF16.
    _32B,
}

impl LlamaOmni2Variant {
    /// Wire tag string written by the converter into
    /// `vokra.llama_omni2.variant`.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::_7B => "7b",
            Self::_3BBilingual => "3b-bilingual",
            Self::_1_5B => "1.5b",
            Self::_32B => "32b",
        }
    }

    /// Parses the wire tag back to the runtime discriminator. Returns
    /// [`None`] for an unrecognised tag (FR-EX-08 — the caller raises a
    /// loud `VokraError::ModelLoad` naming every accepted tag rather
    /// than defaulting to `_7B`).
    #[must_use]
    pub fn from_tag(s: &str) -> Option<Self> {
        Some(match s {
            "7b" => Self::_7B,
            "3b-bilingual" => Self::_3BBilingual,
            "1.5b" => Self::_1_5B,
            "32b" => Self::_32B,
            _ => return None,
        })
    }

    /// Canonical HF repo id for this variant (kept in sync with the
    /// converter-side `LlamaOmni2Variant::as_repo_id`).
    #[must_use]
    pub fn as_repo_id(self) -> &'static str {
        match self {
            Self::_7B => "ICTNLP/LLaMA-Omni2-7B",
            Self::_3BBilingual => "ICTNLP/LLaMA-Omni2-3B-Bilingual",
            Self::_1_5B => "ICTNLP/LLaMA-Omni2-1.5B",
            Self::_32B => "ICTNLP/LLaMA-Omni2-32B",
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Backbone hparams (Qwen2.5-family decoder-only transformer). Every
/// field is transcribed from the upstream `config.json` at convert time
/// — see the module docstring for the "no hardcoded per-variant
/// constants" rule.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaOmni2BackboneConfig {
    /// `num_hidden_layers` — Qwen2.5 backbone depth.
    pub n_layer: usize,
    /// `hidden_size` — Qwen2.5 residual width.
    pub d_model: usize,
    /// `num_attention_heads` — MHA head count (Q heads; GQA would need
    /// a separate `num_key_value_heads` field on this scaffold, which
    /// is deferred to the per-variant primary-source constant wave).
    pub n_head: usize,
    /// `vocab_size` — Qwen2.5 tokenizer vocab.
    pub vocab: usize,
    /// `intermediate_size` — SwiGLU FFN inner width.
    pub intermediate_size: usize,
    /// `rope_theta` (max period) — RoPE base.
    pub rope_max_period: f32,
    /// `rms_norm_eps` — RMSNorm ε.
    pub rms_norm_eps: f32,
}

impl LlamaOmni2BackboneConfig {
    /// Per-head width (`d_model / n_head`); `0` when `n_head == 0`
    /// (shape-only converter sentinel) so shape checks never panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_head).unwrap_or(0)
    }

    /// MHA algebraic constraint: heads divide the width, all non-zero.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.n_head != 0 && self.d_model != 0 && self.d_model % self.n_head == 0
    }
}

/// Resolved LLaMA-Omni2 hparam snapshot — every field is transcribed
/// from the upstream `config.json` at convert time (or Whisper-family
/// convention for `sample_rate`; documented on the field).
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaOmni2Config {
    /// Backbone (Qwen2.5-family) hparams.
    pub backbone: LlamaOmni2BackboneConfig,
    /// Whisper-family speech encoder projection width (the width the
    /// encoder emits into the LM residual stream). Owner primary-source
    /// verify per variant.
    pub speech_encoder_dim: usize,
    /// Streaming AR speech decoder width (typically retuned to match
    /// `backbone.d_model` per the audit ticket; documented as an
    /// independent axis so future non-square variants surface loudly
    /// through the shape gate).
    pub speech_decoder_dim: usize,
    /// PCM sample rate the Whisper-family speech encoder expects at the
    /// front-end — 16 kHz per Whisper-family convention.
    pub sample_rate: u32,
    /// Which of the four sibling releases this config represents. Used
    /// for dispatch + provenance messages.
    pub variant: LlamaOmni2Variant,
}

impl LlamaOmni2Config {
    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA well-formed head split, even head_dim for
    /// RoPE pairs) mirror the real model.
    ///
    /// Deliberately **not** a per-variant primary-source constant (see
    /// module docstring): the audit ticket lists `~14 GB` for the 7B but
    /// does not disclose the layer / head / hidden / vocab numbers from
    /// `config.json`, so a per-variant `llama_omni2_7b()` constant
    /// would be fabricated. That constant lands in the follow-up wave
    /// once the owner records `config.json` on vast.ai (the same
    /// posture Kyutai STT / Voxtral / Canary-Qwen kept pre-transcription).
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            backbone: LlamaOmni2BackboneConfig {
                n_layer: 2,
                d_model: 16,
                n_head: 4,
                vocab: 32,
                intermediate_size: 32,
                rope_max_period: 500_000.0,
                rms_norm_eps: 1e-6,
            },
            speech_encoder_dim: 16,
            speech_decoder_dim: 16,
            sample_rate: LLAMA_OMNI2_SAMPLE_RATE,
            variant: LlamaOmni2Variant::_7B,
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
        if !self.backbone.is_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 config: backbone ill-formed (n_layer={}, d_model={}, \
                 n_head={}) — expected d_model % n_head == 0, all fields > 0",
                self.backbone.n_layer, self.backbone.d_model, self.backbone.n_head,
            )));
        }
        if self.backbone.n_layer == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: backbone.n_layer must be > 0".to_owned(),
            ));
        }
        if self.backbone.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 config: backbone head_dim {} must be even (RoPE pairs)",
                self.backbone.head_dim(),
            )));
        }
        if self.backbone.vocab == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: backbone.vocab must be > 0".to_owned(),
            ));
        }
        if self.backbone.intermediate_size == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: backbone.intermediate_size must be > 0 (SwiGLU FFN)"
                    .to_owned(),
            ));
        }
        if self.speech_encoder_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: speech_encoder_dim must be > 0 (Whisper-family projection \
                 width — the encoder emits into the LM residual stream)"
                    .to_owned(),
            ));
        }
        if self.speech_decoder_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: speech_decoder_dim must be > 0 (streaming AR head width)"
                    .to_owned(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 config: sample_rate must be > 0".to_owned(),
            ));
        }
        Ok(())
    }

    /// Reads the LLaMA-Omni2 hparams from a LLaMA-Omni2 GGUF.
    ///
    /// Missing numeric keys read as `0` placeholders (the CSM /
    /// kyutai_stt `read_u32_or_zero` / `read_f32_or` pattern) so a
    /// shape-only converter path decays gracefully to
    /// [`Self::validate_for_forward`]'s loud gate; wrong-typed keys are
    /// loud [`VokraError::InvalidArgument`] here (FR-EX-08 — never a
    /// silent type coercion). The variant tag defaults to
    /// [`LlamaOmni2Variant::_7B`] when absent so a metadata-only
    /// fixture parses; an unrecognised tag is a loud
    /// [`VokraError::InvalidArgument`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any present key has the wrong
    /// metadata type, or if the variant tag is unrecognised.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let backbone = LlamaOmni2BackboneConfig {
            n_layer: read_u32_or_zero(file, KEY_BB_N_LAYER)? as usize,
            d_model: read_u32_or_zero(file, KEY_BB_D_MODEL)? as usize,
            n_head: read_u32_or_zero(file, KEY_BB_N_HEAD)? as usize,
            vocab: read_u32_or_zero(file, KEY_BB_VOCAB)? as usize,
            intermediate_size: read_u32_or_zero(file, KEY_BB_INTERMEDIATE_SIZE)? as usize,
            rope_max_period: read_f32_or(file, KEY_BB_ROPE_MAX_PERIOD, 0.0)?,
            rms_norm_eps: read_f32_or(file, KEY_BB_RMS_NORM_EPS, 1e-6)?,
        };
        let variant_str = read_string_or_empty(file, KEY_VARIANT)?;
        let variant = if variant_str.is_empty() {
            LlamaOmni2Variant::default()
        } else {
            LlamaOmni2Variant::from_tag(&variant_str).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "llama-omni2 config: unrecognised variant tag `{variant_str}` \
                     (expected one of: `7b`, `3b-bilingual`, `1.5b`, `32b`). \
                     Primary source: https://huggingface.co/ICTNLP/LLaMA-Omni2-7B \
                     + sibling repos"
                ))
            })?
        };
        Ok(Self {
            backbone,
            speech_encoder_dim: read_u32_or_zero(file, KEY_ENC_DIM)? as usize,
            speech_decoder_dim: read_u32_or_zero(file, KEY_DEC_DIM)? as usize,
            sample_rate: read_u32_or_zero(file, KEY_SAMPLE_RATE)?,
            variant,
        })
    }
}

// Missing numeric keys read as `0` placeholders (a shape-only converter
// path decays gracefully to `validate_for_forward`'s loud gate); wrong-
// typed keys are loud `VokraError::InvalidArgument` (FR-EX-08 — never a
// silent type coercion). Mirrors the CSM / kyutai_stt helper of the
// same name.
fn read_u32_or_zero(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => Ok(*v),
        None => Ok(0),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "llama-omni2 config: `{key}` is not a UINT32 (got {:?})",
            other.value_type()
        ))),
    }
}

fn read_f32_or(file: &GgufFile, key: &str, default: f32) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => Ok(*v),
        None => Ok(default),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "llama-omni2 config: `{key}` is not a FLOAT32 (got {:?})",
            other.value_type()
        ))),
    }
}

fn read_string_or_empty(file: &GgufFile, key: &str) -> Result<String> {
    match file.get(key) {
        Some(GgufMetadataValue::String(s)) => Ok(s.clone()),
        None => Ok(String::new()),
        Some(other) => Err(VokraError::InvalidArgument(format!(
            "llama-omni2 config: `{key}` is not a STRING (got {:?})",
            other.value_type()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Per-block backbone weights (Qwen2.5-family: pre-norm MHA + SwiGLU
/// FFN). Field names mirror the upstream Qwen2 convention, which is
/// also what a future shared `vokra_ops::qwen2` op would bind against
/// (that module is proposed, not landed — voxtral / kyutai_stt /
/// canary_qwen each still re-implement the forward inline).
#[derive(Debug, Clone)]
pub struct LlamaOmni2BlockWeights {
    /// Pre-attention RMSNorm γ, shape `[d_model]`.
    pub attn_norm: Vec<f32>,
    /// Fused Q/K/V projection (transposed), shape `[d_model, 3*d_model]`.
    /// (GQA-aware fused shape is a follow-up on the per-variant primary-
    /// source constant wave — this scaffold assumes vanilla MHA.)
    pub qkv_proj: Vec<f32>,
    /// Output projection (transposed), shape `[d_model, d_model]`.
    pub out_proj: Vec<f32>,
    /// Pre-FFN RMSNorm γ, shape `[d_model]`.
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate+up fused linear, shape
    /// `[d_model, 2 * intermediate_size]`.
    pub linear_in: Vec<f32>,
    /// SwiGLU down linear, shape `[intermediate_size, d_model]`.
    pub linear_out: Vec<f32>,
}

/// LLaMA-Omni2 weight store: text embedding + backbone blocks + final
/// norm + LM head + speech encoder projection + speech decoder head.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a
/// follow-up (T29-equivalent — tensor-name manifest fetch from the
/// upstream release).
#[derive(Debug, Clone)]
pub struct LlamaOmni2Weights {
    /// Text-token input embedding, shape `[vocab, d_model]`.
    pub text_embedding: Vec<f32>,
    /// Backbone blocks in order.
    pub blocks: Vec<LlamaOmni2BlockWeights>,
    /// Final backbone RMSNorm γ, shape `[d_model]`.
    pub final_norm: Vec<f32>,
    /// LM head (transposed), shape `[d_model, vocab]`. Independent from
    /// `text_embedding` so `tie_word_embeddings=false` and =true both
    /// bind cleanly (a tied checkpoint just copies the embedding into
    /// this slot at bind time — the loader is authoritative).
    pub lm_head: Vec<f32>,
    /// Whisper-family speech encoder projection into the LM residual
    /// stream. Shape `[speech_encoder_dim, d_model]`.
    pub speech_encoder_proj: Vec<f32>,
    /// Streaming AR speech decoder head projecting the LM residual
    /// back into the speech decoder stream. Shape `[d_model,
    /// speech_decoder_dim]`.
    pub speech_decoder_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint. Real-checkpoint bindings set this to
    /// `false`.
    pub is_synthesized: bool,
}

impl LlamaOmni2Weights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every RMSNorm γ starts at `1.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &LlamaOmni2Config, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let bb = &config.backbone;
        let d = bb.d_model;
        let ffn = bb.intermediate_size;
        let vocab = bb.vocab;
        let enc = config.speech_encoder_dim;
        let dec = config.speech_decoder_dim;

        let text_embedding = xavier(&mut rng, vocab * d, vocab, d);
        let mut blocks = Vec::with_capacity(bb.n_layer);
        for _ in 0..bb.n_layer {
            blocks.push(LlamaOmni2BlockWeights {
                attn_norm: vec![1.0; d],
                qkv_proj: xavier(&mut rng, d * 3 * d, d, 3 * d),
                out_proj: xavier(&mut rng, d * d, d, d),
                ffn_norm: vec![1.0; d],
                linear_in: xavier(&mut rng, d * 2 * ffn, d, 2 * ffn),
                linear_out: xavier(&mut rng, ffn * d, ffn, d),
            });
        }
        let final_norm = vec![1.0; d];
        let lm_head = xavier(&mut rng, d * vocab, d, vocab);
        let speech_encoder_proj = xavier(&mut rng, enc * d, enc, d);
        let speech_decoder_head = xavier(&mut rng, d * dec, d, dec);

        Ok(Self {
            text_embedding,
            blocks,
            final_norm,
            lm_head,
            speech_encoder_proj,
            speech_decoder_head,
            is_synthesized: true,
        })
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed `rng`.
fn xavier(rng: &mut SplitMix64, count: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let a = (6.0 / (fan_in + fan_out) as f32).sqrt();
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

/// LLaMA-Omni2 streaming S2S engine handle.
///
/// Carries the resolved config + weight store. [`Self::converse`] is
/// the primary PCM-in → PCM-out entry point; until real weights are
/// bound (see the module docstring) it returns
/// [`VokraError::UnsupportedOp`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill / noise stream).
#[derive(Debug, Clone)]
pub struct LlamaOmni2 {
    cfg: LlamaOmni2Config,
    weights: LlamaOmni2Weights,
}

/// Alias for [`LlamaOmni2`] — keeps naming parallel with sibling
/// runtime engines that spell out the `Engine` suffix (`MoshiEngine`,
/// `CsmEngine`).
pub type LlamaOmni2Engine = LlamaOmni2;

impl LlamaOmni2 {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, per-tensor
    /// sizes, embedding table shapes) so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: LlamaOmni2Config, weights: LlamaOmni2Weights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let bb = &cfg.backbone;
        let d = bb.d_model;
        let ffn = bb.intermediate_size;
        let vocab = bb.vocab;
        let enc = cfg.speech_encoder_dim;
        let dec = cfg.speech_decoder_dim;

        if weights.text_embedding.len() != vocab * d {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: text_embedding.len()={} != vocab*d_model={}",
                weights.text_embedding.len(),
                vocab * d,
            )));
        }
        if weights.blocks.len() != bb.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: blocks.len()={} != backbone.n_layer={}",
                weights.blocks.len(),
                bb.n_layer,
            )));
        }
        for (i, blk) in weights.blocks.iter().enumerate() {
            for (name, len, expected) in [
                ("attn_norm", blk.attn_norm.len(), d),
                ("qkv_proj", blk.qkv_proj.len(), d * 3 * d),
                ("out_proj", blk.out_proj.len(), d * d),
                ("ffn_norm", blk.ffn_norm.len(), d),
                ("linear_in", blk.linear_in.len(), d * 2 * ffn),
                ("linear_out", blk.linear_out.len(), ffn * d),
            ] {
                if len != expected {
                    return Err(VokraError::InvalidArgument(format!(
                        "llama-omni2 weights: block {i} `{name}` len={len} != {expected}",
                    )));
                }
            }
        }
        if weights.final_norm.len() != d {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: final_norm.len()={} != d_model={}",
                weights.final_norm.len(),
                d,
            )));
        }
        if weights.lm_head.len() != d * vocab {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: lm_head.len()={} != d_model*vocab={}",
                weights.lm_head.len(),
                d * vocab,
            )));
        }
        if weights.speech_encoder_proj.len() != enc * d {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: speech_encoder_proj.len()={} != speech_encoder_dim*d_model={}",
                weights.speech_encoder_proj.len(),
                enc * d,
            )));
        }
        if weights.speech_decoder_head.len() != d * dec {
            return Err(VokraError::InvalidArgument(format!(
                "llama-omni2 weights: speech_decoder_head.len()={} != d_model*speech_decoder_dim={}",
                weights.speech_decoder_head.len(),
                d * dec,
            )));
        }
        Ok(Self { cfg, weights })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &LlamaOmni2Config {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`LlamaOmni2Weights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Streaming speech-to-speech converse: input PCM at
    /// `config().sample_rate` (16 kHz per Whisper-family convention),
    /// output PCM at the same rate.
    ///
    /// This is the primary streaming S2S entry point. **Real weights
    /// required** and **real forward not yet bound**: synthesized-weight
    /// builds cannot produce meaningful audio (they would be noise or
    /// a hallucinated fixed sequence), and the shared Qwen2.5 forward
    /// primitives / Whisper-family speech encoder forward / streaming
    /// AR speech decoder / streaming session infrastructure are pending.
    /// This returns [`VokraError::UnsupportedOp`] naming the blockers
    /// (FR-EX-08 — never a silent zero-fill / noise stream). Callers
    /// verify the shape flow through [`LlamaOmni2::new`] +
    /// [`LlamaOmni2Weights::synthesized`] today.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm_in` is empty or
    ///   contains non-finite samples.
    /// - [`VokraError::UnsupportedOp`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn converse(&self, pcm_in: &[f32]) -> Result<Vec<f32>> {
        if pcm_in.is_empty() {
            return Err(VokraError::InvalidArgument(
                "llama-omni2 converse: pcm_in is empty".to_owned(),
            ));
        }
        for (i, sample) in pcm_in.iter().enumerate() {
            if !sample.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "llama-omni2 converse: pcm_in[{i}]={sample} is not finite \
                     (NaN / +Inf / -Inf) — reject at the boundary (FR-EX-08)"
                )));
            }
        }
        let bb = &self.cfg.backbone;
        // Built before the outer `format!` so the message does not nest one
        // `format!` inside another's arguments (`clippy::format_in_format_args`).
        let repo_url = format!("https://huggingface.co/{}", self.cfg.variant.as_repo_id());
        Err(VokraError::UnsupportedOp(format!(
            "llama-omni2 converse: streaming S2S forward not yet bound. \
             This scaffold binds the shape / provenance / license contract; \
             follow-up wave requires (1) Qwen2.5 backbone forward with RoPE + \
             SwiGLU + RMSNorm (n_layer={n_layer}, d_model={d_model}, \
             n_head={n_head}, vocab={vocab}) — landing this as a NEW shared \
             `vokra_ops::qwen2` op (no such module exists today) would unlock \
             voxtral / canary_qwen / kyutai_stt / \
             firered_asr_llm_l together, (2) Whisper-style speech encoder forward \
             (projection dim={enc}) share primitives with voxtral / \
             canary_qwen speech encoders, (3) streaming AR speech decoder \
             (head dim={dec}) with streaming session infrastructure — \
             mirror moshi / csm full-duplex session code, (4) real \
             checkpoint tensor-name manifest binding (currently \
             {source_note}). Primary source: {repo_url} \
             / https://github.com/ictnlp/LLaMA-Omni2 (ACL 2025).",
            n_layer = bb.n_layer,
            d_model = bb.d_model,
            n_head = bb.n_head,
            vocab = bb.vocab,
            enc = self.cfg.speech_encoder_dim,
            dec = self.cfg.speech_decoder_dim,
            source_note = if self.weights.is_synthesized {
                "synthesized weights (SplitMix64 + Xavier) — bind real ICTNLP checkpoint before invoking converse"
            } else {
                "real weights bound but forward path not yet lit"
            },
        )))
    }

    /// Loads a LLaMA-Omni2 GGUF from raw bytes under `policy` (M2-13
    /// gate — a non-commercial provenance without a research flag is
    /// refused).
    ///
    /// Weight posture: **synthesized bridge** until the real-checkpoint
    /// tensor-name manifest lands (T29-equivalent — the Kyutai STT
    /// [`from_gguf_with_policy`](super::kyutai_stt::KyutaiSttAsr::from_gguf_with_policy)
    /// precedent). The engine binds
    /// [`LlamaOmni2Weights::synthesized`] against the GGUF's shape
    /// config using [`LLAMA_OMNI2_FROM_GGUF_DEFAULT_SEED`] so shape /
    /// dtype / size flow can be exercised without the real HF checkpoint;
    /// a `converse` call fires the synthesized-weight loud-partial arm
    /// and names the primary source URL.
    ///
    /// The LLaMA-Omni2 weight license is **Apache-2.0** (Qwen2.5 派生
    /// chain, `Permissive`) — the converter's registry mapping and
    /// provenance stamps make the M2-13 gate pass commercially without
    /// a research opt-in. `docs/license-audit.md` §3.1 row records the
    /// fail-closed BLANK sign-off (CC never pre-fills Approval column).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on parse failure / wrong or missing
    ///   `vokra.model.arch` — the message names the expected arch tag
    ///   (`llama_omni2`), sibling arch tags (`voxtral` / `canary_qwen`
    ///   / `kyutai_stt` / `firered_asr_llm_l`) so a mis-routed GGUF
    ///   fails specifically here, and the primary source URL.
    /// - [`VokraError::ResearchLicenseRequired`] (from the M2-13 gate)
    ///   when the weight class is gated and `policy` grants no research
    ///   opt-in (never a silent skip / substitution).
    /// - [`VokraError::InvalidArgument`] on a `0`-placeholder shape
    ///   config (a scaffold converter path that never wrote the real
    ///   hparams) from the downstream
    ///   [`LlamaOmni2Config::validate_for_forward`] gate.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("llama-omni2 GGUF: {e}")))?;
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == EXPECTED_ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "llama-omni2: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model llama-omni2`? \
                     Sibling Qwen2-family arches — `voxtral` (Mistral-family ASR / S2S), \
                     `canary_qwen` (FastConformer + Qwen decoder ASR), \
                     `kyutai_stt` (Helium-style decoder-only ASR), \
                     `firered_asr_llm_l` (Conformer + Qwen2 LM ASR) — are different \
                     topologies with distinct tensor manifests). \
                     Primary source: https://huggingface.co/ICTNLP/LLaMA-Omni2-7B \
                     / https://github.com/ictnlp/LLaMA-Omni2"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "llama-omni2: GGUF is missing `vokra.model.arch` (converter did \
                     not stamp it — this is not a Vokra-native `{EXPECTED_ARCH}` \
                     GGUF). Primary source: \
                     https://huggingface.co/ICTNLP/LLaMA-Omni2-7B / \
                     https://github.com/ictnlp/LLaMA-Omni2"
                )));
            }
        }
        check_weight_license(&file, policy)?;
        let cfg = LlamaOmni2Config::from_gguf(&file)?;
        // `synthesized` runs `validate_for_forward` internally; keep the
        // explicit call here so a validate failure surfaces with the config
        // context intact (same posture as Kyutai STT from_gguf_with_policy).
        cfg.validate_for_forward()?;
        let weights = LlamaOmni2Weights::synthesized(&cfg, LLAMA_OMNI2_FROM_GGUF_DEFAULT_SEED)?;
        Self::new(cfg, weights)
    }

    /// Loads a LLaMA-Omni2 GGUF from a file path with the fail-closed
    /// strict policy ([`CompliancePolicy::strict`]).
    ///
    /// The LLaMA-Omni2 weight license is **Apache-2.0** (`Permissive`),
    /// which is commercially permitted — the M2-13 gate passes under
    /// `strict` without a research opt-in.
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::LicenseClass;
    use vokra_core::gguf::GgufBuilder;

    /// Builds a metadata-only GGUF whose `vokra.model.arch` is `arch`
    /// (unless `set_arch` is false). Adds every well-formed
    /// `vokra.llama_omni2.*` chunk group `LlamaOmni2Config::from_gguf`
    /// reads, mirroring the offline converter's `write_hparams` so a
    /// round-trip yields the same tiny config snapshot.
    fn build_gguf_with_hparams(arch: Option<&str>, cfg: &LlamaOmni2Config) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        if let Some(a) = arch {
            b.add_string(chunks::KEY_MODEL_ARCH, a);
        }
        b.add_string(KEY_VARIANT, cfg.variant.tag());
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(KEY_BB_N_LAYER, cfg.backbone.n_layer as u32);
        b.add_u32(KEY_BB_D_MODEL, cfg.backbone.d_model as u32);
        b.add_u32(KEY_BB_N_HEAD, cfg.backbone.n_head as u32);
        b.add_u32(KEY_BB_VOCAB, cfg.backbone.vocab as u32);
        b.add_u32(
            KEY_BB_INTERMEDIATE_SIZE,
            cfg.backbone.intermediate_size as u32,
        );
        b.add_f32(KEY_BB_ROPE_MAX_PERIOD, cfg.backbone.rope_max_period);
        b.add_f32(KEY_BB_RMS_NORM_EPS, cfg.backbone.rms_norm_eps);
        b.add_u32(KEY_ENC_DIM, cfg.speech_encoder_dim as u32);
        b.add_u32(KEY_DEC_DIM, cfg.speech_decoder_dim as u32);
        // Provenance — Permissive (Apache-2.0) so the M2-13 gate passes
        // under `CompliancePolicy::strict()`.
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, "apache-2.0");
        b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "llama-omni2-7b");
        b.to_bytes().expect("serialize llama-omni2 fixture GGUF")
    }

    #[test]
    fn expected_arch_is_llama_omni2() {
        assert_eq!(EXPECTED_ARCH, "llama_omni2");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        LlamaOmni2Config::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_head_split_ill_formed_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        c.backbone.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_odd_head_dim_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        // Make head_dim odd: d_model=12, n_head=4 → head_dim=3, odd.
        c.backbone.d_model = 12;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_vocab_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        c.backbone.vocab = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_intermediate_size_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        c.backbone.intermediate_size = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_encoder_dim_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        c.speech_encoder_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_zero_decoder_dim_is_rejected() {
        let mut c = LlamaOmni2Config::tiny_for_tests();
        c.speech_decoder_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn variant_tags_round_trip() {
        for v in [
            LlamaOmni2Variant::_7B,
            LlamaOmni2Variant::_3BBilingual,
            LlamaOmni2Variant::_1_5B,
            LlamaOmni2Variant::_32B,
        ] {
            let tag = v.tag();
            assert_eq!(LlamaOmni2Variant::from_tag(tag), Some(v));
            assert!(v.as_repo_id().starts_with("ICTNLP/LLaMA-Omni2"));
        }
        // Unknown tags return None (never a silent default to _7B).
        assert_eq!(LlamaOmni2Variant::from_tag("999b"), None);
        assert_eq!(LlamaOmni2Variant::from_tag(""), None);
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w1 = LlamaOmni2Weights::synthesized(&c, 0x42).expect("build 1");
        let w2 = LlamaOmni2Weights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.text_embedding, w2.text_embedding);
        assert_eq!(
            w1.blocks[0].qkv_proj, w2.blocks[0].qkv_proj,
            "same seed → same weights"
        );
        assert!(w1.is_synthesized);
        // Shape flow.
        let d = c.backbone.d_model;
        let ffn = c.backbone.intermediate_size;
        assert_eq!(w1.text_embedding.len(), c.backbone.vocab * d);
        assert_eq!(w1.blocks.len(), c.backbone.n_layer);
        for blk in &w1.blocks {
            assert_eq!(blk.attn_norm.len(), d);
            assert_eq!(blk.qkv_proj.len(), d * 3 * d);
            assert_eq!(blk.out_proj.len(), d * d);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.linear_in.len(), d * 2 * ffn);
            assert_eq!(blk.linear_out.len(), ffn * d);
        }
        assert_eq!(w1.final_norm.len(), d);
        assert_eq!(w1.lm_head.len(), d * c.backbone.vocab);
        assert_eq!(w1.speech_encoder_proj.len(), c.speech_encoder_dim * d);
        assert_eq!(w1.speech_decoder_head.len(), d * c.speech_decoder_dim);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w_a = LlamaOmni2Weights::synthesized(&c, 1).expect("build a");
        let w_b = LlamaOmni2Weights::synthesized(&c, 2).expect("build b");
        assert_ne!(w_a.text_embedding, w_b.text_embedding);
    }

    #[test]
    fn engine_new_accepts_matching_config_and_weights() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        let engine = LlamaOmni2::new(c.clone(), w).expect("llama-omni2 engine");
        assert_eq!(engine.config().backbone.d_model, c.backbone.d_model);
        assert!(engine.is_synthesized());
    }

    #[test]
    fn engine_new_rejects_layer_count_mismatch() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let mut w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            LlamaOmni2::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn engine_new_rejects_tensor_size_mismatch() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let mut w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        w.blocks[0].qkv_proj.pop();
        assert!(matches!(
            LlamaOmni2::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn engine_new_rejects_speech_encoder_proj_mismatch() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let mut w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        w.speech_encoder_proj.pop();
        assert!(matches!(
            LlamaOmni2::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn engine_new_rejects_speech_decoder_head_mismatch() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let mut w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        w.speech_decoder_head.pop();
        assert!(matches!(
            LlamaOmni2::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn converse_rejects_empty_input() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        let engine = LlamaOmni2::new(c, w).expect("llama-omni2 engine");
        assert!(matches!(
            engine.converse(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn converse_rejects_non_finite_samples() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        let engine = LlamaOmni2::new(c, w).expect("llama-omni2 engine");
        let with_nan = [0.5, f32::NAN, 0.25];
        assert!(matches!(
            engine.converse(&with_nan),
            Err(VokraError::InvalidArgument(_))
        ));
        let with_inf = [0.5, f32::INFINITY, 0.25];
        assert!(matches!(
            engine.converse(&with_inf),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The loud-partial converse gate names the primary source URLs +
    /// every missing primitive so a downstream caller / user can look up
    /// the real forward's status (Wave 4 loud-partial contract — never
    /// a silent noise stream).
    #[test]
    fn converse_returns_unsupported_op_naming_primary_source() {
        let c = LlamaOmni2Config::tiny_for_tests();
        let w = LlamaOmni2Weights::synthesized(&c, 7).expect("weights");
        let engine = LlamaOmni2::new(c, w).expect("llama-omni2 engine");
        let err = engine.converse(&[0.1, 0.2, 0.3]).unwrap_err();
        let VokraError::UnsupportedOp(msg) = err else {
            panic!("expected UnsupportedOp, got {err:?}");
        };
        assert!(
            msg.contains("https://huggingface.co/ICTNLP/LLaMA-Omni2-7B"),
            "message must name the HF primary source URL: {msg}"
        );
        assert!(
            msg.contains("github.com/ictnlp/LLaMA-Omni2"),
            "message must name the GitHub primary source URL: {msg}"
        );
        assert!(
            msg.contains("Qwen2.5"),
            "message must name the Qwen2.5 backbone: {msg}"
        );
        assert!(
            msg.contains("speech encoder"),
            "message must name the speech encoder blocker: {msg}"
        );
        assert!(
            msg.contains("streaming"),
            "message must name the streaming session blocker: {msg}"
        );
        assert!(
            msg.contains("synthesized"),
            "message must name the synthesized-weight source note: {msg}"
        );
    }

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let cfg = LlamaOmni2Config::tiny_for_tests();
        let bytes = build_gguf_with_hparams(None, &cfg);
        let err = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("missing arch must be rejected");
        let VokraError::ModelLoad(msg) = err else {
            panic!("expected ModelLoad, got {err:?}");
        };
        assert!(
            msg.contains(EXPECTED_ARCH),
            "message must name expected arch `{EXPECTED_ARCH}`: {msg}"
        );
        assert!(
            msg.contains("huggingface.co/ICTNLP/LLaMA-Omni2-7B"),
            "message must name the primary source URL: {msg}"
        );
    }

    /// A GGUF whose arch is a sibling Qwen2 family arch fails with a
    /// message that names both `llama_omni2` and the offending sibling
    /// tag so the caller can diagnose the mis-routed conversion.
    #[test]
    fn from_gguf_rejects_wrong_arch_naming_siblings() {
        let cfg = LlamaOmni2Config::tiny_for_tests();
        let bytes = build_gguf_with_hparams(Some("voxtral"), &cfg);
        let err = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong arch must be rejected");
        let VokraError::ModelLoad(msg) = err else {
            panic!("expected ModelLoad, got {err:?}");
        };
        assert!(
            msg.contains(EXPECTED_ARCH),
            "message must name expected arch `{EXPECTED_ARCH}`: {msg}"
        );
        assert!(
            msg.contains("voxtral"),
            "message must name the offending arch tag `voxtral`: {msg}"
        );
        // Sibling arches enumerated so a caller sees the diagnostic map.
        for sibling in ["canary_qwen", "kyutai_stt", "firered_asr_llm_l"] {
            assert!(
                msg.contains(sibling),
                "message must name sibling arch `{sibling}`: {msg}"
            );
        }
    }

    /// The `vokra.llama_omni2.*` chunk group round-trips through the
    /// offline-converter format: every field of a tiny config survives
    /// write → parse → read. Pins the cross-crate handshake with
    /// `vokra-convert` — the two crates only share `vokra-core`, so a
    /// converter-side key-string change surfaces as a runtime
    /// `from_gguf` regression here.
    #[test]
    fn config_round_trips_from_converter_written_gguf() {
        let cfg = LlamaOmni2Config::tiny_for_tests();
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH), &cfg);
        let file = GgufFile::parse(bytes).expect("parse fixture");
        let round = LlamaOmni2Config::from_gguf(&file).expect("from_gguf");
        assert_eq!(round, cfg);
    }

    /// A GGUF whose provenance advertises `Permissive` (Apache-2.0)
    /// passes the M2-13 gate under [`CompliancePolicy::strict`] (no
    /// research opt-in needed — the license is commercially permitted)
    /// and the resulting engine binds the synthesized-weight bridge.
    #[test]
    fn from_gguf_reads_permissive_license_and_binds() {
        let cfg = LlamaOmni2Config::tiny_for_tests();
        let bytes = build_gguf_with_hparams(Some(EXPECTED_ARCH), &cfg);
        let file = GgufFile::parse(bytes.clone()).expect("parse fixture");
        let resolution =
            check_weight_license(&file, &CompliancePolicy::strict()).expect("strict must pass");
        assert_eq!(resolution.class, LicenseClass::Permissive);
        assert!(
            !resolution.is_research_only(),
            "Apache-2.0 is commercial-permitted; must NOT be marked research-only"
        );
        let engine = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("from_gguf under strict policy");
        assert!(
            engine.is_synthesized(),
            "from_gguf binds synthesized bridge"
        );
        assert_eq!(engine.config(), &cfg);
    }

    /// A GGUF with `n_layer = 0` (a scaffold converter path that never
    /// wrote the real hparams) fails at the downstream
    /// [`LlamaOmni2Config::validate_for_forward`] gate — the loud
    /// FR-EX-08 surface, not deep inside a GEMM.
    #[test]
    fn from_gguf_rejects_zero_placeholder_config() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        // Deliberately omit every `vokra.llama_omni2.*` chunk — every
        // read decays to the `0` placeholder branch.
        let bytes = b.to_bytes().expect("serialize");
        let err = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("0-placeholder config must be rejected");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    /// A GGUF that mis-types `sample_rate` (F32 instead of U32 — a
    /// hypothetical bad converter path) fails with a loud
    /// [`VokraError::InvalidArgument`] naming the offending key
    /// (FR-EX-08 — never a silent type coercion). Pins the
    /// [`read_u32_or_zero`] helper's type check.
    #[test]
    fn from_gguf_rejects_wrong_typed_key() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        // sample_rate riding as F32 instead of U32.
        b.add_f32(KEY_SAMPLE_RATE, 16_000.0);
        let bytes = b.to_bytes().expect("serialize");
        let err = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("wrong-typed key must be rejected");
        let VokraError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            msg.contains(KEY_SAMPLE_RATE),
            "message must name the offending key `{KEY_SAMPLE_RATE}`: {msg}"
        );
        assert!(
            msg.contains("UINT32"),
            "message must name the expected type UINT32: {msg}"
        );
    }

    /// An unrecognised variant tag fails loudly (FR-EX-08 — never a
    /// silent fallthrough to `_7B` which would misroute the
    /// tokenizer / dispatch).
    #[test]
    fn from_gguf_rejects_unknown_variant_tag() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        );
        b.add_string(KEY_VARIANT, "999b-nonexistent");
        // Give the config valid dims so the failure is specifically the
        // variant tag mismatch, not a downstream shape reject.
        let cfg = LlamaOmni2Config::tiny_for_tests();
        b.add_u32(KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(KEY_BB_N_LAYER, cfg.backbone.n_layer as u32);
        b.add_u32(KEY_BB_D_MODEL, cfg.backbone.d_model as u32);
        b.add_u32(KEY_BB_N_HEAD, cfg.backbone.n_head as u32);
        b.add_u32(KEY_BB_VOCAB, cfg.backbone.vocab as u32);
        b.add_u32(
            KEY_BB_INTERMEDIATE_SIZE,
            cfg.backbone.intermediate_size as u32,
        );
        b.add_u32(KEY_ENC_DIM, cfg.speech_encoder_dim as u32);
        b.add_u32(KEY_DEC_DIM, cfg.speech_decoder_dim as u32);
        let bytes = b.to_bytes().expect("serialize");
        let err = LlamaOmni2::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect_err("unknown variant tag must be rejected");
        let VokraError::InvalidArgument(msg) = err else {
            panic!("expected InvalidArgument, got {err:?}");
        };
        assert!(
            msg.contains("999b-nonexistent"),
            "message must name the offending variant: {msg}"
        );
        assert!(
            msg.contains("7b") && msg.contains("3b-bilingual"),
            "message must enumerate accepted variants: {msg}"
        );
    }

    #[test]
    fn sample_rate_matches_whisper_family_convention() {
        // 16 kHz — Whisper-family convention (the speech encoder axis
        // shared with voxtral / canary / canary-qwen / distil-whisper /
        // kotoba-whisper / moonshine). Documented as a runtime constant
        // that a future per-variant primary-source override can shadow.
        assert_eq!(LLAMA_OMNI2_SAMPLE_RATE, 16_000);
    }
}
