//! **Qwen3-TTS-12Hz released family** — Alibaba's Qwen3-TTS codec-LM
//! speech synthesizer (SoTA plan Phase 3, 2026-07-24; released-variant
//! contract corrected 2026-08-27). Apache-2.0
//! **end-to-end** — the LM + the codec + the tokenizer + the speaker
//! encoder all ship under a single `apache-2.0` grant.
//!
//! # What Qwen3-TTS-0.6B is (primary source)
//!
//! `Qwen/Qwen3-TTS-12Hz-0.6B-Base` is Alibaba's discrete multi-codebook
//! LM TTS system. Every field below is transcribed **verbatim** from
//! the upstream `config.json` at
//! `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
//!
//! ```text
//!   model_type          = "qwen3_tts"
//!   architectures       = ["Qwen3TTSForConditionalGeneration"]
//!   tts_model_size      = "0b6"
//!   tts_model_type      = "base"
//!   transformers_version = "4.57.3"
//!
//!   # Talker (the main autoregressive LM)
//!   hidden_size              = 1024
//!   num_hidden_layers        = 28
//!   num_attention_heads      = 16
//!   num_key_value_heads      = 8       # GQA (16 Q ÷ 8 KV)
//!   head_dim                 = 128
//!   intermediate_size        = 3072    # SwiGLU inner dim
//!   vocab_size               = 3072    # (per-codebook speech-token vocab — see below)
//!   text_vocab_size          = 151936  # Qwen3 shared text vocabulary
//!   max_position_embeddings  = 32768
//!   rope_theta               = 1000000
//!   rms_norm_eps             = 1e-06
//!   position_id_per_seconds  = 13
//!   num_code_groups          = 16      # matches Qwen3-TTS-Codec num_quantizers
//!   text_hidden_size         = 2048    # projection from a separate text encoder
//!
//!   # Code predictor (the per-step multi-codebook parallel head)
//!   code_predictor.hidden_size          = 1024
//!   code_predictor.num_hidden_layers    = 5
//!   code_predictor.num_attention_heads  = 16
//!   code_predictor.num_key_value_heads  = 8      # GQA
//!   code_predictor.head_dim             = 128
//!   code_predictor.intermediate_size    = 3072
//!   code_predictor.vocab_size           = 2048   # acoustic per-codebook vocab
//!   code_predictor.rope_theta           = 1000000
//!   code_predictor.rms_norm_eps         = 1e-06
//!   code_predictor.num_code_groups      = 16
//!
//!   # Speaker encoder — 24 kHz sample rate, 1024-dim voice embedding.
//! ```
//!
//! The **talker** is a decoder-only Qwen3-flavour transformer (GQA
//! 16 Q ÷ 8 KV, RoPE θ = 1 000 000, RMSNorm ε = 1e-6, SwiGLU FFN — the
//! same op inventory as Qwen2, only widened head split + rope base). The
//! **code predictor** is a small (5-layer) parallel head that emits 16
//! codebook rows per talker step. [`vokra_ops::qwen3_tts_codec`] validates
//! and folds that 16-row layout into a 512-wide feature stream at 12.5 Hz;
//! it does **not** produce PCM. Terminal waveform synthesis is owned by the
//! separately authenticated [`Qwen3TtsTokenizer12HzDecoder`] companion.
//!
//! The **tokenizer** is `Qwen2Tokenizer` (a byte-level BPE with
//! `merges.txt` + `vocab.json` — same tokenizer class CosyVoice2 /
//! CosyVoice3 use, just with a 151 936-token Qwen3 vocabulary).
//! `tokenizer_config.json` sets `bos = null`, `eos = "<|im_end|>"`,
//! `pad = "<|endoftext|>"`, `unk = null`; the trained
//! `model_max_length = 131_072` is the tokenizer's ceiling and is
//! **not** the same as the transformer's `max_position_embeddings`
//! (32 768) — the runtime honours the transformer axis, not the
//! tokenizer one.
//!
//! Distinct architectural axes vs. the closest siblings
//!
//! - vs. **CosyVoice2/3** (also Qwen family + HiFT-GAN vocoder):
//!   Qwen3-TTS-0.6B is **codec-LM, not vocoder-LM**: its output is a
//!   sixteen-row code matrix, not a mel. The separate 12 Hz tokenizer
//!   decoder maps those codes to PCM; `HiFTChain` is not compatible.
//! - vs. **Chatterbox / Chatterbox-Nano** (T3 / GPT-2 LM + speech
//!   tokens): the LM emits **16 codebook rows per step**, not a single
//!   speech-token stream. The per-step multi-codebook head is the
//!   [`Qwen3TtsCodePredictorConfig`] axis this module surfaces.
//!
//! # License and files
//!
//! - **License**: `apache-2.0` — end-to-end. `README.md` YAML front
//!   matter carries `license: apache-2.0`, the HF API card lists
//!   `license: Apache 2.0`, and the codec submodule
//!   (`speech_tokenizer/`) ships under the same repo-wide grant. No
//!   attribution obligation on the runtime side (M2-13 gate passes
//!   commercially — same posture as CosyVoice2/3 and the Chatterbox
//!   family).
//! - **Weights & code**:
//!   `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base` — carries
//!   `config.json`, `generation_config.json`, `merges.txt`,
//!   `model.safetensors`, `preprocessor_config.json`, `tokenizer_config.json`,
//!   `vocab.json`, and the `speech_tokenizer/` submodule
//!   (`config.json` + `configuration.json` + `model.safetensors` +
//!   `preprocessor_config.json`). At pinned Hub revision
//!   `5d83992436eae1d760afd27aff78a71d676296fc`, the main BF16
//!   `model.safetensors` is 1,829,344,272 bytes; the separately loaded
//!   speech-tokenizer directory is another ~682 MB.
//!
//! # What lands in this Phase 3 slice
//!
//! - [`Qwen3TtsTalkerConfig`] + [`Qwen3TtsCodePredictorConfig`] +
//!   [`Qwen3TtsConfig`] — every architectural hparam **transcribed
//!   verbatim** from the primary source `config.json` plus the codec
//!   handshake with [`vokra_ops::qwen3_tts_codec::Qwen3TtsCodecConfig`]
//!   (the 16 codebook rows the code predictor emits per step must
//!   match the codec's `num_quantizers`). `validate_for_forward` fails
//!   loudly (FR-EX-08) on zeroed axes / broken GQA algebra / broken
//!   codec handshake.
//! - [`Qwen3TtsWeights`] — deterministic
//!   [`Qwen3TtsWeights::synthesized`] fixture (SplitMix64 + Xavier)
//!   against `config` so shape / dtype / size flow can be exercised
//!   without the real HF checkpoint.
//! - [`Qwen3TtsTts`] — legacy deterministic-fixture handle carrying config +
//!   synthesized weights. Its compatibility [`Qwen3TtsTts::synthesize`]
//!   entry point intentionally remains unsupported for non-real weights;
//!   real-weight callers use [`Qwen3TtsMain::synthesize_with_decoder`], which
//!   executes the mapped talker, code predictor, and authenticated companion
//!   decoder on one explicit CPU or Metal backend.
//!
//! # Code-layout seam and waveform companion
//!
//! [`vokra_ops::qwen3_tts_codec`] remains the shared 16-quantizer layout and
//! feature-fold contract. The official waveform checkpoint has its own
//! learned Euclidean codebooks, sliding Transformer, ConvNeXt upsamplers and
//! causal transposed-convolution decoder, bound separately by
//! [`Qwen3TtsTokenizer12HzDecoder`]. Its mapping-owning constructors execute
//! that complete code-to-PCM graph on CPU or Metal. Neither surface silently
//! substitutes for the other.
//!
//! # No ONNX (permanent)
//!
//! Qwen3-TTS-0.6B is distributed as safetensors + a Python pipeline;
//! the runtime **never** loads an ONNX graph (FR-LD-05, permanent
//! constraint); the pipeline is re-implemented natively from the
//! safetensors checkpoint (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

mod bound;
mod generation;
mod tokenizer;
mod tokenizer_12hz;
mod tokenizer_12hz_forward;
mod weights;

pub use bound::{
    Qwen3TtsBoundBlockWeights, Qwen3TtsCheckpoint, Qwen3TtsCheckpointVariant,
    qwen3_tts_code_predictor_block_forward, qwen3_tts_talker_block_forward,
};
pub use generation::{
    QWEN3_TTS_MAIN_HOT_OPS, Qwen3TtsGeneratedCodes, Qwen3TtsGenerationOptions, Qwen3TtsMain,
    Qwen3TtsSynthesis, Qwen3TtsTalkerOutput, Qwen3TtsTalkerSession,
};
pub use tokenizer::{
    CODEC_BOS_TOKEN_ID, CODEC_EOS_TOKEN_ID, CODEC_NOTHINK_TOKEN_ID, CODEC_PAD_TOKEN_ID,
    CODEC_THINK_BOS_TOKEN_ID, CODEC_THINK_EOS_TOKEN_ID, CODEC_THINK_TOKEN_ID, Qwen3TtsTokenizer,
    SUPPORTED_LANGUAGES as QWEN3_TTS_SUPPORTED_LANGUAGES, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID,
    TTS_PAD_TOKEN_ID,
};
pub use tokenizer_12hz::{Qwen3TtsTokenizer12HzConfig, Qwen3TtsTokenizer12HzDecoder};

// ---------------------------------------------------------------------------
// Public seam re-exports — shared with the codec primitive
// ---------------------------------------------------------------------------
//
// `vokra_ops::qwen3_tts_codec` describes the main model's 16-row code-layout
// seam; it is not the terminal waveform decoder. Re-export the config alias
// here so a caller wiring Qwen3-TTS sees that handshake under this module's path
// without a shape-drift wrapper (mirrors `crate::chatterbox_nano` /
// `crate::cosyvoice3` re-exporting `HiFTChain` from `cosyvoice2`).

pub use vokra_ops::qwen3_tts_codec::Qwen3TtsCodecConfig;

/// `vokra.model.arch` a Qwen3-TTS GGUF must carry. Written by
/// `vokra-convert::models::qwen3_tts::ARCH`. Intentionally **distinct**
/// from CosyVoice2/3's `"cosyvoice2"` / `"cosyvoice3"` and from the
/// Chatterbox family's `"chatterbox*"` so the runtime can label the
/// loaded model correctly in telemetry / logs / model cards. The
/// compliance registry (`vokra_core::compliance`) knows every
/// `qwen3-tts*` spelling as
/// [`vokra_core::LicenseClass::Permissive`] (apache-2.0 —
/// end-to-end, no runtime-side attribution obligation).
pub const EXPECTED_ARCH: &str = "qwen3_tts";

/// PCM sample rate the Qwen3-TTS speaker encoder consumes (Hz). Fixed
/// by the upstream release (`README.md` — "Speaker Encoder: 24kHz
/// sample rate, 1024-dim encoding"). Not the codec's *output* sample
/// rate — that is 24 kHz too and is carried by
/// [`Qwen3TtsCodecConfig::sample_rate`] on the shared codec seam.
pub const QWEN3_TTS_SAMPLE_RATE: u32 = 24_000;

/// Speaker-embedding width the speaker encoder emits (per the model
/// card). The talker projects this into `talker.hidden_size` at the
/// prompt boundary; the projection weight is a real
/// `[speaker_embed_dim, hidden_size]` tensor in the upstream
/// checkpoint (`speaker_proj.weight`).
pub const QWEN3_TTS_SPEAKER_EMBED_DIM: u32 = 1024;
/// Speaker-embedding width of the 1.7B-Base release. The official
/// `speaker_encoder_config.enc_dim` is 2048; CustomVoice/VoiceDesign do not
/// instantiate a speaker encoder at all.
pub const QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM: u32 = 2048;

/// Number of parallel codebook streams the code predictor emits per
/// talker step — matches [`Qwen3TtsCodecConfig::num_quantizers`] on
/// the shared codec seam (16 for every released Qwen3-TTS variant).
/// A future release that widens the codec must bump both this and
/// the codec's `num_quantizers` together.
pub const QWEN3_TTS_NUM_CODE_GROUPS: u32 = 16;

// ---------------------------------------------------------------------------
// Variant enum — LM hidden-size fork
// ---------------------------------------------------------------------------

/// The Qwen3-TTS size variant the loader targets.
///
/// Alibaba's Qwen3-TTS family scales the talker LM's hidden dim +
/// intermediate size (SwiGLU FFN) around a fixed head axis (16 Q ÷ 8
/// KV × 128 head_dim). The codec + code-predictor contracts are
/// size-invariant (all variants emit the same 16 codebook rows per step to
/// the shared Qwen3-TTS-Codec at 12.5 Hz). Speaker encoders are Base-only
/// and widen from 1024 at 0.6B to 2048 at 1.7B.
///
/// SoTA plan reuse bundle (2026-07-30): hidden-size branch of the
/// existing 0.6B loader — new variants are additive against the same
/// shared codec seam (`vokra_ops::qwen3_tts_codec`) so a downstream
/// picks a variant without duplicating arch code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen3TtsVariant {
    /// `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — the anchor 0.6B release
    /// (1,829,344,272-byte BF16 safetensors at the pinned revision). Talker:
    /// `hidden_dim=1024`,
    /// `n_layer=28`, GQA `16 Q ÷ 8 KV × head_dim=128`,
    /// SwiGLU `ffn_dim=3072`, `text_vocab_size=151936`,
    /// `max_position_embeddings=32768`. Primary source =
    /// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`
    /// (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    H0_6B,
    /// `Qwen/Qwen3-TTS-12Hz-1.7B-Base` family — widened talker
    /// `hidden_dim=2048`, `ffn_dim=6144`, 28 layers, with the same 1024-wide
    /// code predictor. Base additionally carries a 2048-wide speaker
    /// encoder and a learned 2048→1024 small-to-MTP projection.
    H1_7B,
}

impl Qwen3TtsVariant {
    /// Canonical model-card slug for this variant.
    #[must_use]
    pub fn model_id(self) -> &'static str {
        match self {
            Self::H0_6B => "qwen3-tts-0.6b",
            Self::H1_7B => "qwen3-tts-1.7b",
        }
    }
}

// ---------------------------------------------------------------------------
// Talker config — the main autoregressive LM
// ---------------------------------------------------------------------------

/// Qwen3-TTS talker (main AR LM) architectural hparams.
///
/// Every field is transcribed **verbatim** from the primary source
/// (`config.json.talker.*` at
/// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`, fetched 2026-07-24
/// — CLAUDE.md「ハルシネーション厳禁」). The talker is a Qwen3-flavour
/// decoder-only transformer with GQA (16 Q ÷ 8 KV) and a widened rope
/// base (1 000 000).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsTalkerConfig {
    /// Backbone hidden dimension (`config.json.talker.hidden_size`).
    /// `1024` for the 0.6B release.
    pub hidden_dim: u32,
    /// Backbone transformer block count
    /// (`config.json.talker.num_hidden_layers`). `28` for the 0.6B
    /// release.
    pub n_layer: u32,
    /// Backbone attention head count (`num_attention_heads`). `16`.
    pub n_head: u32,
    /// Backbone key/value head count for GQA (`num_key_value_heads`).
    /// `8` — the group ratio is `n_head / n_head_kv = 2` (each K/V
    /// head fans out to 2 Q heads).
    pub n_head_kv: u32,
    /// Backbone attention head dimension (`head_dim`). `128` — note
    /// this is **decoupled** from `hidden_dim / n_head`
    /// (1024 / 16 = 64 ≠ 128); Qwen3 uses `head_dim = 128` with a
    /// wider-than-hidden Q projection.
    pub head_dim: u32,
    /// SwiGLU FFN inner dimension (`intermediate_size`). `3072`.
    pub ffn_dim: u32,
    /// Per-codebook speech-token vocabulary the talker emits at each
    /// group slot (`vocab_size` in the talker sub-config). `3072` —
    /// distinct from the code predictor's `2048` (the talker keeps a
    /// wider vocab to absorb the semantic quantizer's larger alphabet;
    /// see the [`Qwen3TtsCodePredictorConfig::vocab_size`] docstring
    /// for the semantic vs acoustic split).
    pub vocab_size: u32,
    /// Shared text-token vocabulary size (`text_vocab_size`). `151 936`
    /// — the Qwen3 base tokenizer.
    pub text_vocab_size: u32,
    /// Max positions the talker can attend over
    /// (`max_position_embeddings`). `32 768` — distinct from the
    /// tokenizer's `model_max_length = 131 072`; the transformer axis
    /// is the runtime-authoritative one.
    pub max_position_embeddings: u32,
    /// RoPE base θ (`rope_theta`). `1_000_000` — widened from the
    /// Qwen2 default (500 000) to cover the longer TTS context.
    pub rope_base: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-6`.
    pub rms_norm_eps: f32,
    /// Position ids per second of audio (`position_id_per_seconds`).
    /// `13` — controls how the talker's position ids advance with the
    /// codec frame rate (12.5 Hz output; the extra half-second of
    /// slack is a training-side choice).
    pub position_id_per_seconds: u32,
    /// Number of parallel codebook rows the talker slots per step
    /// (`num_code_groups`). `16` — must equal
    /// [`QWEN3_TTS_NUM_CODE_GROUPS`] and, at codec time, the shared
    /// [`Qwen3TtsCodecConfig::num_quantizers`].
    pub num_code_groups: u32,
    /// Text encoder hidden dimension (`text_hidden_size`). `2048` —
    /// the width of the (separate, upstream) text encoder that feeds
    /// the talker; the talker projects it back to `hidden_dim` at the
    /// prompt boundary.
    pub text_hidden_size: u32,
}

impl Qwen3TtsTalkerConfig {
    /// Canonical Qwen3-TTS-0.6B-Base talker config (primary source:
    /// `config.json.talker.*`, fetched 2026-07-24).
    #[must_use]
    pub fn qwen3_tts_0_6b_base() -> Self {
        Self {
            hidden_dim: 1024,
            n_layer: 28,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            ffn_dim: 3072,
            vocab_size: 3072,
            text_vocab_size: 151_936,
            max_position_embeddings: 32_768,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            position_id_per_seconds: 13,
            num_code_groups: QWEN3_TTS_NUM_CODE_GROUPS,
            text_hidden_size: 2048,
        }
    }

    /// Qwen3-TTS-1.7B talker config transcribed from the official pinned
    /// `config.json`. Family-invariant axes remain:
    ///
    /// - `n_head = 16` (Q head count, family-fixed).
    /// - `n_head_kv = 8` (KV head count for GQA, family-fixed).
    /// - `head_dim = 128` (family-fixed).
    /// - `rope_base = 1_000_000` (family-fixed).
    /// - `rms_norm_eps = 1e-6` (family-fixed).
    /// - `position_id_per_seconds = 13` (family-fixed — controls how
    ///   position ids advance with codec frame rate).
    /// - `num_code_groups = 16` (family-fixed — cross-check with the
    ///   shared `Qwen3TtsCodecConfig::num_quantizers` at
    ///   validate-time).
    ///
    #[must_use]
    pub fn qwen3_tts_1_7b_base() -> Self {
        Self {
            hidden_dim: 2048,
            n_layer: 28,
            ffn_dim: 6144,
            text_hidden_size: 2048,
            vocab_size: 3072,
            text_vocab_size: 151_936,
            max_position_embeddings: 32_768,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            position_id_per_seconds: 13,
            num_code_groups: QWEN3_TTS_NUM_CODE_GROUPS,
        }
    }

    /// Miniature well-formed talker config for shape / stability tests.
    /// Dims are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (GQA well-formedness, positive FFN dim, non-zero
    /// vocab, RoPE even head_dim, `num_code_groups` matching the codec)
    /// mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_dim: 16,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 32,
            text_vocab_size: 64,
            max_position_embeddings: 128,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            position_id_per_seconds: 13,
            num_code_groups: 3,
            text_hidden_size: 24,
        }
    }
}

// ---------------------------------------------------------------------------
// Code predictor config — the per-step multi-codebook parallel head
// ---------------------------------------------------------------------------

/// Qwen3-TTS **code predictor** hparams — the small (5-layer) parallel
/// head that emits 16 codebook rows per talker step.
///
/// Every field transcribed verbatim from the primary source
/// (`config.json.code_predictor.*` at
/// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`, fetched 2026-07-24).
/// The code predictor is a shallow Qwen3-flavour transformer that
/// shares the talker's head axes (16 Q ÷ 8 KV, head_dim 128, RoPE θ =
/// 1 000 000, RMSNorm ε = 1e-6) but keeps its own smaller depth
/// (`num_hidden_layers = 5`) and a distinct per-codebook acoustic
/// vocabulary (`vocab_size = 2048`).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsCodePredictorConfig {
    /// Code-predictor hidden dimension
    /// (`config.json.code_predictor.hidden_size`). `1024`.
    pub hidden_dim: u32,
    /// Code-predictor transformer block count
    /// (`num_hidden_layers`). `5` — one-fifth the talker depth.
    pub n_layer: u32,
    /// Attention head count (`num_attention_heads`). `16`.
    pub n_head: u32,
    /// KV head count for GQA (`num_key_value_heads`). `8`.
    pub n_head_kv: u32,
    /// Attention head dimension (`head_dim`). `128`.
    pub head_dim: u32,
    /// SwiGLU FFN inner dimension (`intermediate_size`). `3072`.
    pub ffn_dim: u32,
    /// Per-codebook acoustic vocabulary size (`vocab_size`). `2048` —
    /// distinct from the codec's semantic vocabulary of `4096`; the
    /// difference is why [`Qwen3TtsCodecConfig`] carries both
    /// `codebook_size` and `semantic_codebook_size` (see the primary
    /// source docstring on the codec seam).
    pub vocab_size: u32,
    /// Maximum code-predictor positions. Every pinned release carries
    /// `max_position_embeddings = 65_536`.
    pub max_position_embeddings: u32,
    /// RoPE base θ (`rope_theta`). `1_000_000` — same as the talker.
    pub rope_base: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-6`.
    pub rms_norm_eps: f32,
    /// Number of parallel codebook rows emitted per step
    /// (`num_code_groups`). `16` — must equal the talker's and the
    /// codec's.
    pub num_code_groups: u32,
}

impl Qwen3TtsCodePredictorConfig {
    /// Canonical Qwen3-TTS-0.6B-Base code-predictor config (primary
    /// source: `config.json.code_predictor.*`, fetched 2026-07-24).
    #[must_use]
    pub fn qwen3_tts_0_6b_base() -> Self {
        Self {
            hidden_dim: 1024,
            n_layer: 5,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            ffn_dim: 3072,
            vocab_size: 2048,
            max_position_embeddings: 65_536,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            num_code_groups: QWEN3_TTS_NUM_CODE_GROUPS,
        }
    }

    /// Miniature code-predictor config for tests — mirrors the tiny
    /// talker's shape relationships.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_dim: 16,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 24,
            max_position_embeddings: 128,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            num_code_groups: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config — talker + code predictor + codec handshake
// ---------------------------------------------------------------------------

/// Full Qwen3-TTS config: [`Qwen3TtsTalkerConfig`] +
/// [`Qwen3TtsCodePredictorConfig`] plus the shared codec handshake
/// (`num_code_groups` must match the codec's `num_quantizers`, else the
/// parallel-head → codec bridge silently drops or duplicates codebook
/// rows — FR-EX-08).
///
/// The codec attributes are not carried here directly (they live on
/// the shared [`Qwen3TtsCodecConfig`] seam re-exported from
/// [`vokra_ops::qwen3_tts_codec`]); [`Self::codec_config`] returns the
/// canonical released-variant codec config, and the handshake gate
/// [`Self::validate_for_forward`] cross-checks the group counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsConfig {
    /// PCM sample rate the speaker encoder consumes (Hz). Fixed at
    /// 24 kHz by the released model card ("Speaker Encoder: 24kHz
    /// sample rate").
    pub sample_rate: u32,
    /// Speaker-embedding width the speaker encoder emits (per the
    /// model card).
    pub speaker_embed_dim: u32,
    /// Whether the main checkpoint contains the official ECAPA-style speaker
    /// encoder. Only Base releases do; CustomVoice and VoiceDesign select
    /// fixed speaker ids and omit all `speaker_encoder.*` tensors.
    pub has_speaker_encoder: bool,
    /// Talker sub-config.
    pub talker: Qwen3TtsTalkerConfig,
    /// Code-predictor sub-config.
    pub code_predictor: Qwen3TtsCodePredictorConfig,
}

impl Qwen3TtsConfig {
    /// Canonical Qwen3-TTS-0.6B-Base config
    /// (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`, primary source:
    /// `config.json`, `README.md`, fetched 2026-07-24).
    #[must_use]
    pub fn qwen3_tts_0_6b_base() -> Self {
        Self {
            sample_rate: QWEN3_TTS_SAMPLE_RATE,
            speaker_embed_dim: QWEN3_TTS_SPEAKER_EMBED_DIM,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig::qwen3_tts_0_6b_base(),
            code_predictor: Qwen3TtsCodePredictorConfig::qwen3_tts_0_6b_base(),
        }
    }

    /// Qwen3-TTS-1.7B Base config. The talker widens to 2048/6144, the
    /// speaker encoder widens to 2048, and the code predictor remains
    /// 1024/3072. The binder separately distinguishes Base from the
    /// speaker-less CustomVoice/VoiceDesign manifests.
    #[must_use]
    pub fn qwen3_tts_1_7b_base() -> Self {
        Self {
            sample_rate: QWEN3_TTS_SAMPLE_RATE,
            speaker_embed_dim: QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig::qwen3_tts_1_7b_base(),
            code_predictor: Qwen3TtsCodePredictorConfig::qwen3_tts_0_6b_base(),
        }
    }

    /// Variant-aware constructor — dispatches to
    /// `qwen3_tts_0_6b_base()` / `qwen3_tts_1_7b_base()` based on the
    /// passed [`Qwen3TtsVariant`]. Convenience for callers already
    /// carrying the variant tag (a converter side-car, a CLI arg).
    #[must_use]
    pub fn for_variant(variant: Qwen3TtsVariant) -> Self {
        match variant {
            Qwen3TtsVariant::H0_6B => Self::qwen3_tts_0_6b_base(),
            Qwen3TtsVariant::H1_7B => Self::qwen3_tts_1_7b_base(),
        }
    }

    /// Miniature well-formed config for shape / stability tests.
    /// `sample_rate` / `speaker_embed_dim` stay at their canonical
    /// values (the primary-source invariants), only the transformer
    /// dims shrink to KB-fit for synthesized-weight builds.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: QWEN3_TTS_SAMPLE_RATE,
            speaker_embed_dim: 8,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig::tiny_for_tests(),
            code_predictor: Qwen3TtsCodePredictorConfig::tiny_for_tests(),
        }
    }

    /// Returns the canonical released-variant codec config the parallel
    /// head is designed to feed (`Qwen3TtsCodecConfig::qwen3_tts_12hz`).
    /// Callers with a hypothetical future variant supply their own via
    /// [`Self::validate_for_forward_with_codec`].
    #[must_use]
    pub fn codec_config(&self) -> Qwen3TtsCodecConfig {
        Qwen3TtsCodecConfig::qwen3_tts_12hz()
    }

    /// True iff every architectural axis is at its `0` sentinel — the
    /// shape-only conversion path the runtime tolerates as
    /// inspectable-but-not-forward-ready.
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.talker.hidden_dim == 0
            && self.talker.n_layer == 0
            && self.talker.n_head == 0
            && self.talker.ffn_dim == 0
            && self.talker.vocab_size == 0
            && self.code_predictor.hidden_dim == 0
            && self.code_predictor.n_layer == 0
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs against the canonical
    /// [`Qwen3TtsCodecConfig::qwen3_tts_12hz`] codec (the released
    /// variant). See [`Self::validate_for_forward_with_codec`] for the
    /// hypothetical-future-variant path.
    ///
    /// Enforces the Qwen3 cross-checks (positive axes, GQA
    /// `n_head % n_head_kv == 0`, even RoPE `head_dim`, positive finite
    /// RoPE base / RMSNorm eps, talker `vocab_size >=` code-predictor
    /// `vocab_size` because the talker absorbs the wider semantic
    /// alphabet) plus the codec handshake
    /// (`num_code_groups == codec.num_quantizers`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        self.validate_for_forward_with_codec(&self.codec_config())
    }

    /// Variant-aware form of [`Self::validate_for_forward`] — cross-
    /// checks the `num_code_groups` handshake against the caller-
    /// supplied `codec` instead of the canonical released one. Useful
    /// for a hypothetical future variant that widens the codec.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward_with_codec(&self, codec: &Qwen3TtsCodecConfig) -> Result<()> {
        // Top-level axes.
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts config: sample_rate must be > 0 (bind a real \
                 checkpoint or use Qwen3TtsConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        if self.has_speaker_encoder != (self.speaker_embed_dim > 0) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts config: has_speaker_encoder={} is inconsistent with speaker_embed_dim={} (Base requires > 0; CustomVoice/VoiceDesign require 0)",
                self.has_speaker_encoder, self.speaker_embed_dim
            )));
        }
        // Talker axes.
        let t = &self.talker;
        if t.hidden_dim == 0
            || t.n_layer == 0
            || t.n_head == 0
            || t.n_head_kv == 0
            || t.head_dim == 0
            || t.ffn_dim == 0
            || t.vocab_size == 0
            || t.text_vocab_size == 0
            || t.max_position_embeddings == 0
            || t.position_id_per_seconds == 0
            || t.num_code_groups == 0
            || t.text_hidden_size == 0
        {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts talker config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use Qwen3TtsTalkerConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        if t.n_head % t.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts talker config: n_head_kv ({}) must divide n_head ({}) — Qwen3 GQA \
                 requires an integer group ratio",
                t.n_head_kv, t.n_head,
            )));
        }
        if t.head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts talker config: RoPE requires even head_dim (got {})",
                t.head_dim,
            )));
        }
        if !(t.rope_base.is_finite() && t.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts talker config: rope_base must be a positive finite f32 (got {})",
                t.rope_base,
            )));
        }
        if !(t.rms_norm_eps.is_finite() && t.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts talker config: rms_norm_eps must be a positive finite f32 (got {})",
                t.rms_norm_eps,
            )));
        }

        // Code-predictor axes.
        let cp = &self.code_predictor;
        if cp.hidden_dim == 0
            || cp.n_layer == 0
            || cp.n_head == 0
            || cp.n_head_kv == 0
            || cp.head_dim == 0
            || cp.ffn_dim == 0
            || cp.vocab_size == 0
            || cp.max_position_embeddings == 0
            || cp.num_code_groups == 0
        {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts code_predictor config: every architectural axis must be > 0 (bind a \
                 real checkpoint or use Qwen3TtsCodePredictorConfig::tiny_for_tests for shape \
                 tests)"
                    .to_owned(),
            ));
        }
        if cp.n_head % cp.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts code_predictor config: n_head_kv ({}) must divide n_head ({})",
                cp.n_head_kv, cp.n_head,
            )));
        }
        if cp.head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts code_predictor config: RoPE requires even head_dim (got {})",
                cp.head_dim,
            )));
        }
        if !(cp.rope_base.is_finite() && cp.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts code_predictor config: rope_base must be a positive finite f32 \
                 (got {})",
                cp.rope_base,
            )));
        }
        if !(cp.rms_norm_eps.is_finite() && cp.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts code_predictor config: rms_norm_eps must be a positive finite f32 \
                 (got {})",
                cp.rms_norm_eps,
            )));
        }

        // Cross-sub-config: `num_code_groups` must match between the talker,
        // the code predictor, and the codec — the talker slots N codebook
        // rows per step, the code predictor emits N rows per step, and the
        // codec expects N per-quantizer streams. A silent mismatch would
        // drop or duplicate codebook rows.
        if t.num_code_groups != cp.num_code_groups {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts config: talker.num_code_groups ({}) != code_predictor.num_code_groups \
                 ({}) — the parallel head must emit exactly as many rows per step as the talker \
                 slots",
                t.num_code_groups, cp.num_code_groups,
            )));
        }
        if t.num_code_groups as usize != codec.num_quantizers {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts config: talker.num_code_groups ({}) != codec.num_quantizers ({}) — the \
                 parallel head must feed the codec exactly one row per quantizer per step",
                t.num_code_groups, codec.num_quantizers,
            )));
        }

        // Sanity: the talker's per-codebook vocab must be >= the code
        // predictor's, because the talker absorbs the codec's semantic
        // alphabet (see the code-predictor docstring). A silent inversion
        // here would drop semantic tokens off the top of the acoustic
        // vocab.
        if t.vocab_size < cp.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts config: talker.vocab_size ({}) must be >= \
                 code_predictor.vocab_size ({}) — the talker absorbs the codec's semantic \
                 alphabet, so its per-codebook vocab is at least as wide",
                t.vocab_size, cp.vocab_size,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Legacy synthetic weights (shape tests only; not the real checkpoint)
// ---------------------------------------------------------------------------

/// Historical Qwen3-TTS synthetic talker weight store.
///
/// This type is retained for deterministic allocation/shape tests and public
/// API compatibility. It does **not** describe the inspected upstream
/// checkpoint: the real model has bias-free attention plus per-head `q_norm`
/// / `k_norm`, a `[151936, 2048]` text embedding followed by a two-layer
/// biased text projection, one codec embedding/head, and a separate ECAPA
/// speaker encoder. Production artifacts bind through
/// [`Qwen3TtsCheckpoint`]; do not construct this type from safetensors names.
#[derive(Debug, Clone)]
pub struct Qwen3TtsTalkerWeights {
    /// Text-token embedding: `[text_vocab_size, hidden_dim]` (Qwen3
    /// shared text vocab).
    pub text_embed: Vec<f32>,
    /// Per-codebook speech-token embedding:
    /// `[num_code_groups * vocab_size, hidden_dim]`. Stored flat; the
    /// per-group offset is `group_idx * vocab_size * hidden_dim`.
    pub speech_embed: Vec<f32>,
    /// Speaker embedding projection to backbone hidden width:
    /// `[speaker_embed_dim, hidden_dim]`.
    pub speaker_proj: Vec<f32>,
    /// Text encoder projection to backbone hidden width:
    /// `[text_hidden_size, hidden_dim]`.
    pub text_proj: Vec<f32>,
    /// Per-layer transformer block weights. Length = `n_layer`.
    pub blocks: Vec<Qwen3TtsBlockWeights>,
    /// Final RMSNorm γ, shape `[hidden_dim]`.
    pub final_norm: Vec<f32>,
}

/// Historical synthetic per-block weights.
///
/// The bias vectors here are fixture-only. Real Qwen3-TTS blocks use
/// [`Qwen3TtsBoundBlockWeights`] with bias-free projections and per-head Q/K
/// normalization.
#[derive(Debug, Clone)]
pub struct Qwen3TtsBlockWeights {
    /// Self-attention pre-norm γ, shape `[hidden_dim]`.
    pub self_attn_norm: Vec<f32>,
    /// Q projection, shape `[n_head * head_dim, hidden_dim]` — note
    /// this is **wider** than `hidden_dim * hidden_dim` when
    /// `n_head * head_dim > hidden_dim` (Qwen3 talker: 16 × 128 =
    /// 2048 out, 1024 in).
    pub q_proj: Vec<f32>,
    /// Q bias, shape `[n_head * head_dim]`.
    pub q_bias: Vec<f32>,
    /// K projection, shape `[n_head_kv * head_dim, hidden_dim]` (GQA
    /// narrower than Q).
    pub k_proj: Vec<f32>,
    /// K bias, shape `[n_head_kv * head_dim]`.
    pub k_bias: Vec<f32>,
    /// V projection, shape `[n_head_kv * head_dim, hidden_dim]` (GQA).
    pub v_proj: Vec<f32>,
    /// V bias, shape `[n_head_kv * head_dim]`.
    pub v_bias: Vec<f32>,
    /// O projection, shape `[hidden_dim, n_head * head_dim]` (mirror
    /// of Q — projects the wider Q axis back to `hidden_dim`).
    pub o_proj: Vec<f32>,
    /// FFN pre-norm γ, shape `[hidden_dim]`.
    pub ffn_norm: Vec<f32>,
    /// SwiGLU gate projection, shape `[ffn_dim, hidden_dim]`.
    pub ffn_gate: Vec<f32>,
    /// SwiGLU up projection, shape `[ffn_dim, hidden_dim]`.
    pub ffn_up: Vec<f32>,
    /// SwiGLU down projection, shape `[hidden_dim, ffn_dim]`.
    pub ffn_down: Vec<f32>,
}

/// Historical synthetic code-predictor weight store.
///
/// The real checkpoint has fifteen distinct residual codec embeddings and
/// fifteen LM heads (`num_code_groups - 1`), represented by the strict
/// checkpoint manifest rather than this flattened fixture field.
#[derive(Debug, Clone)]
pub struct Qwen3TtsCodePredictorWeights {
    /// Per-layer transformer block weights. Length = `n_layer`.
    pub blocks: Vec<Qwen3TtsBlockWeights>,
    /// Final RMSNorm γ, shape `[hidden_dim]`.
    pub final_norm: Vec<f32>,
    /// Per-codebook output head, shape
    /// `[num_code_groups * vocab_size, hidden_dim]`. Stored flat; the
    /// per-group offset is `group_idx * vocab_size * hidden_dim`.
    pub code_head: Vec<f32>,
}

/// Full Qwen3-TTS weight store — talker + code predictor.
#[derive(Debug, Clone)]
pub struct Qwen3TtsWeights {
    /// Talker sub-weights.
    pub talker: Qwen3TtsTalkerWeights,
    /// Code-predictor sub-weights.
    pub code_predictor: Qwen3TtsCodePredictorWeights,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

impl Qwen3TtsWeights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via
    /// a [`SplitMix64`] stream — reproducible, allocation-only,
    /// zero-dep. Every RMSNorm γ starts at `1.0`; every bias starts at
    /// `0.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if
    /// `config.validate_for_forward` fails.
    pub fn synthesized(config: &Qwen3TtsConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        Ok(Self {
            talker: synthesize_talker(&config.talker, config.speaker_embed_dim, &mut rng),
            code_predictor: synthesize_code_predictor(&config.code_predictor, &mut rng),
            is_synthesized: true,
        })
    }
}

fn synthesize_talker(
    cfg: &Qwen3TtsTalkerConfig,
    speaker_embed_dim: u32,
    rng: &mut SplitMix64,
) -> Qwen3TtsTalkerWeights {
    let d = cfg.hidden_dim as usize;
    let ffn = cfg.ffn_dim as usize;
    let text_vocab = cfg.text_vocab_size as usize;
    let speech_vocab = cfg.vocab_size as usize;
    let groups = cfg.num_code_groups as usize;
    let spk = speaker_embed_dim as usize;
    let text_h = cfg.text_hidden_size as usize;

    let q_out = (cfg.n_head * cfg.head_dim) as usize;
    let kv_out = (cfg.n_head_kv * cfg.head_dim) as usize;

    let text_embed = xavier(rng, text_vocab * d, d, d);
    let speech_embed = xavier(rng, groups * speech_vocab * d, d, d);
    let speaker_proj = xavier(rng, spk * d, spk, d);
    let text_proj = xavier(rng, text_h * d, text_h, d);

    let mut blocks = Vec::with_capacity(cfg.n_layer as usize);
    for _ in 0..cfg.n_layer {
        blocks.push(Qwen3TtsBlockWeights {
            self_attn_norm: vec![1.0; d],
            q_proj: xavier(rng, q_out * d, d, q_out),
            q_bias: vec![0.0; q_out],
            k_proj: xavier(rng, kv_out * d, d, kv_out),
            k_bias: vec![0.0; kv_out],
            v_proj: xavier(rng, kv_out * d, d, kv_out),
            v_bias: vec![0.0; kv_out],
            o_proj: xavier(rng, d * q_out, q_out, d),
            ffn_norm: vec![1.0; d],
            ffn_gate: xavier(rng, ffn * d, d, ffn),
            ffn_up: xavier(rng, ffn * d, d, ffn),
            ffn_down: xavier(rng, d * ffn, ffn, d),
        });
    }
    let final_norm = vec![1.0; d];

    Qwen3TtsTalkerWeights {
        text_embed,
        speech_embed,
        speaker_proj,
        text_proj,
        blocks,
        final_norm,
    }
}

fn synthesize_code_predictor(
    cfg: &Qwen3TtsCodePredictorConfig,
    rng: &mut SplitMix64,
) -> Qwen3TtsCodePredictorWeights {
    let d = cfg.hidden_dim as usize;
    let ffn = cfg.ffn_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let groups = cfg.num_code_groups as usize;

    let q_out = (cfg.n_head * cfg.head_dim) as usize;
    let kv_out = (cfg.n_head_kv * cfg.head_dim) as usize;

    let mut blocks = Vec::with_capacity(cfg.n_layer as usize);
    for _ in 0..cfg.n_layer {
        blocks.push(Qwen3TtsBlockWeights {
            self_attn_norm: vec![1.0; d],
            q_proj: xavier(rng, q_out * d, d, q_out),
            q_bias: vec![0.0; q_out],
            k_proj: xavier(rng, kv_out * d, d, kv_out),
            k_bias: vec![0.0; kv_out],
            v_proj: xavier(rng, kv_out * d, d, kv_out),
            v_bias: vec![0.0; kv_out],
            o_proj: xavier(rng, d * q_out, q_out, d),
            ffn_norm: vec![1.0; d],
            ffn_gate: xavier(rng, ffn * d, d, ffn),
            ffn_up: xavier(rng, ffn * d, d, ffn),
            ffn_down: xavier(rng, d * ffn, ffn, d),
        });
    }
    let final_norm = vec![1.0; d];
    let code_head = xavier(rng, groups * vocab * d, d, vocab);

    Qwen3TtsCodePredictorWeights {
        blocks,
        final_norm,
        code_head,
    }
}

/// Xavier-uniform draw of `count` `f32`s in `[-a, +a]` where
/// `a = sqrt(6 / (fan_in + fan_out))`. Deterministic under a fixed
/// `rng`.
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

/// Qwen3-TTS TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional
/// [`Qwen3TtsCodecConfig`] override for the shared
/// [`vokra_ops::qwen3_tts_codec`] seam (default = the canonical
/// released variant). [`Self::synthesize`] is a legacy fixture-only
/// compatibility entry point and intentionally returns
/// [`VokraError::NotImplemented`] for synthesized weights. Real checkpoints
/// use [`Qwen3TtsMain::synthesize_with_decoder`], which executes the complete
/// mapped talker → code-predictor → authenticated
/// [`Qwen3TtsTokenizer12HzDecoder`] chain on an explicit backend (FR-EX-08 —
/// never a silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct Qwen3TtsTts {
    cfg: Qwen3TtsConfig,
    weights: Qwen3TtsWeights,
    codec: Qwen3TtsCodecConfig,
}

impl Qwen3TtsTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, per-tensor
    /// sizes) so a mismatched pair fails loudly here rather than deep
    /// inside a forward.
    ///
    /// The codec seam defaults to the canonical released
    /// [`Qwen3TtsCodecConfig::qwen3_tts_12hz`]; callers with a
    /// hypothetical future variant chain through
    /// [`Self::with_codec_config`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: Qwen3TtsConfig, weights: Qwen3TtsWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        check_talker_shapes(&cfg, &weights.talker)?;
        check_code_predictor_shapes(&cfg, &weights.code_predictor)?;
        let codec = cfg.codec_config();
        Ok(Self {
            cfg,
            weights,
            codec,
        })
    }

    /// Injects a caller-supplied codec config — cross-validated against
    /// the config's talker `num_code_groups` at assembly time.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the codec handshake fails
    /// (`talker.num_code_groups != codec.num_quantizers`).
    pub fn with_codec_config(mut self, codec: Qwen3TtsCodecConfig) -> Result<Self> {
        self.cfg.validate_for_forward_with_codec(&codec)?;
        self.codec = codec;
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.cfg
    }

    /// The active codec config (default: `qwen3_tts_12hz`; overridden
    /// via [`Self::with_codec_config`]).
    #[must_use]
    pub fn codec_config(&self) -> &Qwen3TtsCodecConfig {
        &self.codec
    }

    /// True iff the weight store was built by
    /// [`Qwen3TtsWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Compatibility synthesis entry point for the deterministic fixture
    /// handle. **Real-weight synthesis is provided by
    /// [`Qwen3TtsMain::synthesize_with_decoder`];** this legacy API does not
    /// fabricate PCM from synthesized weights and therefore refuses them.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] for the legacy synthesized-weight
    ///   path (FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts synthesize: text is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "qwen3_tts synthesize: this engine holds synthesized weights \
                 (deterministic fixture from Qwen3TtsWeights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real \
                 speech. Bind real Qwen3-TTS weights before invoking this legacy \
                 compatibility API. Use Qwen3TtsMain::synthesize_with_decoder for \
                 real-weight speech. The shape flow (config validation, weight-store \
                 construction, text-empty check) is exercised through \
                 Qwen3TtsTts::new.",
            ));
        }
        Err(VokraError::NotImplemented(
            "qwen3_tts synthesize: this legacy weight-store handle cannot execute \
             real-weight speech. Use Qwen3TtsMain::synthesize_with_decoder with a \
             strictly mapped main checkpoint and authenticated 12-Hz decoder. The \
             smaller vokra_ops::qwen3_tts_codec feature-fold helper is not a waveform \
             fallback.",
        ))
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts weights: {name}.len()={got} != {expected}"
        )));
    }
    Ok(())
}

fn check_talker_shapes(cfg: &Qwen3TtsConfig, w: &Qwen3TtsTalkerWeights) -> Result<()> {
    let t = &cfg.talker;
    let d = t.hidden_dim as usize;
    let text_vocab = t.text_vocab_size as usize;
    let speech_vocab = t.vocab_size as usize;
    let groups = t.num_code_groups as usize;
    let spk = cfg.speaker_embed_dim as usize;
    let text_h = t.text_hidden_size as usize;
    let ffn = t.ffn_dim as usize;

    check_len("talker.text_embed", w.text_embed.len(), text_vocab * d)?;
    check_len(
        "talker.speech_embed",
        w.speech_embed.len(),
        groups * speech_vocab * d,
    )?;
    check_len("talker.speaker_proj", w.speaker_proj.len(), spk * d)?;
    check_len("talker.text_proj", w.text_proj.len(), text_h * d)?;
    check_len("talker.final_norm", w.final_norm.len(), d)?;

    if w.blocks.len() != t.n_layer as usize {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts weights: talker.blocks.len()={} != n_layer={}",
            w.blocks.len(),
            t.n_layer,
        )));
    }
    let q_out = (t.n_head * t.head_dim) as usize;
    let kv_out = (t.n_head_kv * t.head_dim) as usize;
    for (i, blk) in w.blocks.iter().enumerate() {
        check_block_shapes("talker", i, blk, d, ffn, q_out, kv_out)?;
    }
    Ok(())
}

fn check_code_predictor_shapes(
    cfg: &Qwen3TtsConfig,
    w: &Qwen3TtsCodePredictorWeights,
) -> Result<()> {
    let cp = &cfg.code_predictor;
    let d = cp.hidden_dim as usize;
    let ffn = cp.ffn_dim as usize;
    let vocab = cp.vocab_size as usize;
    let groups = cp.num_code_groups as usize;

    check_len("code_predictor.final_norm", w.final_norm.len(), d)?;
    check_len(
        "code_predictor.code_head",
        w.code_head.len(),
        groups * vocab * d,
    )?;

    if w.blocks.len() != cp.n_layer as usize {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts weights: code_predictor.blocks.len()={} != n_layer={}",
            w.blocks.len(),
            cp.n_layer,
        )));
    }
    let q_out = (cp.n_head * cp.head_dim) as usize;
    let kv_out = (cp.n_head_kv * cp.head_dim) as usize;
    for (i, blk) in w.blocks.iter().enumerate() {
        check_block_shapes("code_predictor", i, blk, d, ffn, q_out, kv_out)?;
    }
    Ok(())
}

fn check_block_shapes(
    stack: &str,
    i: usize,
    blk: &Qwen3TtsBlockWeights,
    d: usize,
    ffn: usize,
    q_out: usize,
    kv_out: usize,
) -> Result<()> {
    check_len(
        &format!("{stack}.block[{i}].self_attn_norm"),
        blk.self_attn_norm.len(),
        d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].q_proj"),
        blk.q_proj.len(),
        q_out * d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].q_bias"),
        blk.q_bias.len(),
        q_out,
    )?;
    check_len(
        &format!("{stack}.block[{i}].k_proj"),
        blk.k_proj.len(),
        kv_out * d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].k_bias"),
        blk.k_bias.len(),
        kv_out,
    )?;
    check_len(
        &format!("{stack}.block[{i}].v_proj"),
        blk.v_proj.len(),
        kv_out * d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].v_bias"),
        blk.v_bias.len(),
        kv_out,
    )?;
    check_len(
        &format!("{stack}.block[{i}].o_proj"),
        blk.o_proj.len(),
        d * q_out,
    )?;
    check_len(
        &format!("{stack}.block[{i}].ffn_norm"),
        blk.ffn_norm.len(),
        d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].ffn_gate"),
        blk.ffn_gate.len(),
        ffn * d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].ffn_up"),
        blk.ffn_up.len(),
        ffn * d,
    )?;
    check_len(
        &format!("{stack}.block[{i}].ffn_down"),
        blk.ffn_down.len(),
        d * ffn,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_arch_is_qwen3_tts() {
        assert_eq!(EXPECTED_ARCH, "qwen3_tts");
    }

    #[test]
    fn arch_is_distinct_from_neighbouring_families() {
        // Qwen3-TTS shares the Qwen family with CosyVoice2/3 but uses a
        // codec-LM topology (16-row codes plus a separate tokenizer decoder)
        // instead of a vocoder-LM topology (HiFTChain terminal). Silently sharing the
        // arch tag with CosyVoice2/3 would mis-route the runtime.
        // (The neighbouring `EXPECTED_ARCH` constants are private to
        // their modules; use the same string literals the converter
        // stamps into `vokra.model.arch` — the sole cross-crate
        // handshake — to keep the pin honest without widening
        // visibility just for this check.)
        assert_ne!(EXPECTED_ARCH, "cosyvoice2");
        assert_ne!(EXPECTED_ARCH, crate::cosyvoice3::EXPECTED_ARCH);
        // Chatterbox family uses speech-token AR + HiFTChain vocoder;
        // distinct topology at every axis.
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_nano::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_turbo::EXPECTED_ARCH);
    }

    #[test]
    fn sample_rate_matches_upstream_speaker_encoder() {
        // README.md: "Speaker Encoder: 24kHz sample rate".
        assert_eq!(QWEN3_TTS_SAMPLE_RATE, 24_000);
    }

    #[test]
    fn speaker_embed_dim_matches_upstream_readme() {
        // README.md: "Speaker Encoder: 24kHz sample rate, 1024-dim encoding".
        assert_eq!(QWEN3_TTS_SPEAKER_EMBED_DIM, 1024);
    }

    #[test]
    fn num_code_groups_matches_shared_codec_seam() {
        // The shared codec primitive (vokra_ops::qwen3_tts_codec) exposes
        // num_quantizers=16 on its canonical qwen3_tts_12hz config;
        // if either side changes without the other the parallel head <->
        // codec bridge silently drops or duplicates rows (FR-EX-08).
        let codec = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        assert_eq!(QWEN3_TTS_NUM_CODE_GROUPS as usize, codec.num_quantizers);
    }

    /// Every architectural axis carries its primary-source value
    /// verbatim (config.json.talker.*), and the config passes
    /// validate_for_forward under the canonical codec.
    #[test]
    fn canonical_config_matches_primary_source() {
        let c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.speaker_embed_dim, 1024);

        let t = &c.talker;
        assert_eq!(t.hidden_dim, 1024);
        assert_eq!(t.n_layer, 28);
        assert_eq!(t.n_head, 16);
        assert_eq!(t.n_head_kv, 8);
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.ffn_dim, 3072);
        assert_eq!(t.vocab_size, 3072);
        assert_eq!(t.text_vocab_size, 151_936);
        assert_eq!(t.max_position_embeddings, 32_768);
        assert!((t.rope_base - 1_000_000.0).abs() < 1e-3);
        assert!((t.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(t.position_id_per_seconds, 13);
        assert_eq!(t.num_code_groups, 16);
        assert_eq!(t.text_hidden_size, 2048);

        let cp = &c.code_predictor;
        assert_eq!(cp.hidden_dim, 1024);
        assert_eq!(cp.n_layer, 5);
        assert_eq!(cp.n_head, 16);
        assert_eq!(cp.n_head_kv, 8);
        assert_eq!(cp.head_dim, 128);
        assert_eq!(cp.ffn_dim, 3072);
        assert_eq!(cp.vocab_size, 2048);
        assert!((cp.rope_base - 1_000_000.0).abs() < 1e-3);
        assert!((cp.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(cp.num_code_groups, 16);

        // GQA algebra
        assert_eq!(t.n_head % t.n_head_kv, 0);
        assert_eq!(cp.n_head % cp.n_head_kv, 0);
        // RoPE evenness
        assert_eq!(t.head_dim % 2, 0);
        assert_eq!(cp.head_dim % 2, 0);
        // Semantic-vs-acoustic vocab ordering
        assert!(t.vocab_size >= cp.vocab_size);
        c.validate_for_forward()
            .expect("real Qwen3-TTS-0.6B config is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        let c = Qwen3TtsConfig::tiny_for_tests();
        // The tiny fixture uses 3 code groups (not 16) so the canonical
        // codec would reject the handshake. Cross-check against a
        // matching hypothetical codec instead.
        let mut codec = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        codec.num_quantizers = 3;
        codec.num_semantic_quantizers = 1;
        c.validate_for_forward_with_codec(&codec)
            .expect("tiny config well-formed against a matching codec");
    }

    #[test]
    fn tiny_config_rejects_canonical_codec_because_group_count_differs() {
        // The tiny fixture uses 3 groups; the canonical codec expects 16.
        // A silent match here would defeat the handshake gate.
        let c = Qwen3TtsConfig::tiny_for_tests();
        let err = c
            .validate_for_forward()
            .expect_err("tiny config vs canonical codec must fail");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("num_code_groups"), "message: {msg}");
                assert!(msg.contains("codec"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn placeholder_config_is_placeholder_shape() {
        let c = Qwen3TtsConfig {
            sample_rate: QWEN3_TTS_SAMPLE_RATE,
            speaker_embed_dim: QWEN3_TTS_SPEAKER_EMBED_DIM,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig {
                hidden_dim: 0,
                n_layer: 0,
                n_head: 0,
                n_head_kv: 0,
                head_dim: 0,
                ffn_dim: 0,
                vocab_size: 0,
                text_vocab_size: 0,
                max_position_embeddings: 0,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                position_id_per_seconds: 0,
                num_code_groups: 0,
                text_hidden_size: 0,
            },
            code_predictor: Qwen3TtsCodePredictorConfig {
                hidden_dim: 0,
                n_layer: 0,
                n_head: 0,
                n_head_kv: 0,
                head_dim: 0,
                ffn_dim: 0,
                vocab_size: 0,
                max_position_embeddings: 0,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                num_code_groups: 0,
            },
        };
        assert!(c.is_placeholder_shape());
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_talker_zero_axis() {
        let mutators: &[fn(&mut Qwen3TtsConfig)] = &[
            |c| c.talker.hidden_dim = 0,
            |c| c.talker.n_layer = 0,
            |c| c.talker.n_head = 0,
            |c| c.talker.n_head_kv = 0,
            |c| c.talker.head_dim = 0,
            |c| c.talker.ffn_dim = 0,
            |c| c.talker.vocab_size = 0,
            |c| c.talker.text_vocab_size = 0,
            |c| c.talker.max_position_embeddings = 0,
            |c| c.talker.position_id_per_seconds = 0,
            |c| c.talker.num_code_groups = 0,
            |c| c.talker.text_hidden_size = 0,
        ];
        let base = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        for mutate in mutators {
            let mut c = base.clone();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_code_predictor_zero_axis() {
        let mutators: &[fn(&mut Qwen3TtsConfig)] = &[
            |c| c.code_predictor.hidden_dim = 0,
            |c| c.code_predictor.n_layer = 0,
            |c| c.code_predictor.n_head = 0,
            |c| c.code_predictor.n_head_kv = 0,
            |c| c.code_predictor.head_dim = 0,
            |c| c.code_predictor.ffn_dim = 0,
            |c| c.code_predictor.vocab_size = 0,
        ];
        let base = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        for mutate in mutators {
            let mut c = base.clone();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_gqa_non_divisor_talker() {
        let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        c.talker.n_head_kv = 7; // 16 % 7 != 0
        let err = c.validate_for_forward().expect_err("bad GQA divisor fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("n_head_kv"), "message: {msg}");
                assert!(msg.contains("talker"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_odd_head_dim_talker() {
        let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        c.talker.head_dim = 65; // odd — RoPE needs pairs
        let err = c.validate_for_forward().expect_err("odd head_dim fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("head_dim"), "message: {msg}");
                assert!(msg.contains("RoPE"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_non_finite_rope_base() {
        for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
            c.talker.rope_base = bad;
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_group_count_mismatch_between_talker_and_predictor() {
        let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        c.code_predictor.num_code_groups = 15; // != talker's 16
        let err = c
            .validate_for_forward()
            .expect_err("group-count mismatch fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("num_code_groups"), "message: {msg}");
                assert!(msg.contains("code_predictor"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_talker_vocab_smaller_than_predictor_vocab() {
        // If the talker's vocab is narrower than the code predictor's,
        // semantic tokens spill off the top of the acoustic vocab silently.
        let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        c.talker.vocab_size = 1024; // < 2048 (code_predictor)
        let err = c
            .validate_for_forward()
            .expect_err("narrower talker vocab fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("talker.vocab_size"), "message: {msg}");
                assert!(msg.contains("code_predictor.vocab_size"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        // Use the tiny fixture with a matching codec so the group-count gate
        // passes.
        let c = Qwen3TtsConfig::tiny_for_tests();
        let mut codec = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        codec.num_quantizers = c.talker.num_code_groups as usize;
        codec.num_semantic_quantizers = 1;
        c.validate_for_forward_with_codec(&codec).expect("tiny ok");

        // synthesized calls validate_for_forward() against the canonical
        // codec, which the tiny fixture fails. Bypass by building the
        // synthesized weights against a tiny wrapper that swaps the codec.
        // We keep the interface consistent by making a canonical config
        // whose num_code_groups matches the tiny codec.
        let c_syn = {
            let mut c = c.clone();
            // Match against a synthetic canonical codec that carries this
            // group count.
            c.talker.num_code_groups = codec.num_quantizers as u32;
            c.code_predictor.num_code_groups = codec.num_quantizers as u32;
            c
        };

        // The tiny fixture already has num_code_groups=3, so trying
        // Qwen3TtsWeights::synthesized(&c_syn, seed) hits the group-count
        // gate against the canonical (16-quantizer) codec. Use the tiny
        // config directly with a matching hypothetical codec instead:
        // shape-only tests can still exercise the weight builder by going
        // through the synthesize functions via the same seed and comparing
        // manually — however for the `synthesized(&Qwen3TtsConfig, seed)`
        // signature the group-count gate is unavoidable.
        //
        // Solution: exercise Qwen3TtsWeights::synthesized on a canonical
        // config (which does validate) but shrink the block dims to 0? No —
        // that would trip the zero-axis gate. The realistic path is to
        // validate that the tiny fixture, when its group count is aligned
        // with a synthetic (still-16-quantizer) codec, works end-to-end.
        //
        // Simplest fix: use the canonical qwen3_tts_0_6b_base config for
        // this determinism / shape-correct test; the primary check we want
        // (bit-identical draws under the same seed) is dim-agnostic.
        let _ = c_syn;

        let real = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        // Real Qwen3-TTS-0.6B is 1.83 GB in BF16; the synthesized fixture
        // would allocate ~3.5 GB in F32. Instead of running that here, we
        // build the fixture against the tiny group-aligned codec via a
        // config whose block dims are tiny but whose num_code_groups
        // matches. Explicit hand-built config:
        let tiny_aligned = Qwen3TtsConfig {
            sample_rate: real.sample_rate,
            speaker_embed_dim: 8,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig {
                hidden_dim: 16,
                n_layer: 2,
                n_head: 4,
                n_head_kv: 2,
                head_dim: 8,
                ffn_dim: 32,
                vocab_size: 32,
                text_vocab_size: 64,
                max_position_embeddings: 128,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                position_id_per_seconds: 13,
                num_code_groups: 16, // match canonical codec
                text_hidden_size: 24,
            },
            code_predictor: Qwen3TtsCodePredictorConfig {
                hidden_dim: 16,
                n_layer: 2,
                n_head: 4,
                n_head_kv: 2,
                head_dim: 8,
                ffn_dim: 32,
                vocab_size: 24,
                max_position_embeddings: 128,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                num_code_groups: 16,
            },
        };
        tiny_aligned
            .validate_for_forward()
            .expect("tiny_aligned validates against canonical codec");

        let w1 = Qwen3TtsWeights::synthesized(&tiny_aligned, 0x42).expect("build 1");
        let w2 = Qwen3TtsWeights::synthesized(&tiny_aligned, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.talker.text_embed, w2.talker.text_embed);
        assert_eq!(w1.talker.blocks[0].q_proj, w2.talker.blocks[0].q_proj);
        assert_eq!(w1.code_predictor.code_head, w2.code_predictor.code_head);
        assert!(w1.is_synthesized);

        // Shape flow.
        let t = &tiny_aligned.talker;
        let d = t.hidden_dim as usize;
        let ffn = t.ffn_dim as usize;
        let text_vocab = t.text_vocab_size as usize;
        let speech_vocab = t.vocab_size as usize;
        let groups = t.num_code_groups as usize;
        let spk = tiny_aligned.speaker_embed_dim as usize;
        let text_h = t.text_hidden_size as usize;
        let q_out = (t.n_head * t.head_dim) as usize;
        let kv_out = (t.n_head_kv * t.head_dim) as usize;

        assert_eq!(w1.talker.text_embed.len(), text_vocab * d);
        assert_eq!(w1.talker.speech_embed.len(), groups * speech_vocab * d);
        assert_eq!(w1.talker.speaker_proj.len(), spk * d);
        assert_eq!(w1.talker.text_proj.len(), text_h * d);
        assert_eq!(w1.talker.final_norm.len(), d);
        assert_eq!(w1.talker.blocks.len(), t.n_layer as usize);
        for blk in &w1.talker.blocks {
            assert_eq!(blk.self_attn_norm.len(), d);
            assert_eq!(blk.q_proj.len(), q_out * d);
            assert_eq!(blk.q_bias.len(), q_out);
            assert_eq!(blk.k_proj.len(), kv_out * d);
            assert_eq!(blk.k_bias.len(), kv_out);
            assert_eq!(blk.v_proj.len(), kv_out * d);
            assert_eq!(blk.v_bias.len(), kv_out);
            assert_eq!(blk.o_proj.len(), d * q_out);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.ffn_gate.len(), ffn * d);
            assert_eq!(blk.ffn_up.len(), ffn * d);
            assert_eq!(blk.ffn_down.len(), d * ffn);
        }
        // Code predictor
        let cp = &tiny_aligned.code_predictor;
        let cp_d = cp.hidden_dim as usize;
        let cp_ffn = cp.ffn_dim as usize;
        let cp_vocab = cp.vocab_size as usize;
        let cp_groups = cp.num_code_groups as usize;
        assert_eq!(w1.code_predictor.final_norm.len(), cp_d);
        assert_eq!(
            w1.code_predictor.code_head.len(),
            cp_groups * cp_vocab * cp_d
        );
        assert_eq!(w1.code_predictor.blocks.len(), cp.n_layer as usize);
        for blk in &w1.code_predictor.blocks {
            assert_eq!(blk.ffn_gate.len(), cp_ffn * cp_d);
        }
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let mut c = Qwen3TtsConfig::tiny_for_tests();
        c.talker.num_code_groups = 16;
        c.code_predictor.num_code_groups = 16;
        let a = Qwen3TtsWeights::synthesized(&c, 1).expect("a");
        let b = Qwen3TtsWeights::synthesized(&c, 2).expect("b");
        assert_ne!(a.talker.text_embed, b.talker.text_embed);
        assert_ne!(a.code_predictor.code_head, b.code_predictor.code_head);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        c.talker.hidden_dim = 0;
        assert!(matches!(
            Qwen3TtsWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// Builds a small canonical-aligned config for engine tests. Group
    /// count matches the canonical codec so `Qwen3TtsTts::new` passes
    /// the handshake without extra plumbing.
    fn small_aligned_config() -> Qwen3TtsConfig {
        Qwen3TtsConfig {
            sample_rate: QWEN3_TTS_SAMPLE_RATE,
            speaker_embed_dim: 8,
            has_speaker_encoder: true,
            talker: Qwen3TtsTalkerConfig {
                hidden_dim: 16,
                n_layer: 2,
                n_head: 4,
                n_head_kv: 2,
                head_dim: 8,
                ffn_dim: 32,
                vocab_size: 32,
                text_vocab_size: 64,
                max_position_embeddings: 128,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                position_id_per_seconds: 13,
                num_code_groups: 16,
                text_hidden_size: 24,
            },
            code_predictor: Qwen3TtsCodePredictorConfig {
                hidden_dim: 16,
                n_layer: 2,
                n_head: 4,
                n_head_kv: 2,
                head_dim: 8,
                ffn_dim: 32,
                vocab_size: 24,
                max_position_embeddings: 128,
                rope_base: 1_000_000.0,
                rms_norm_eps: 1e-6,
                num_code_groups: 16,
            },
        }
    }

    #[test]
    fn tts_new_accepts_matching_config_and_weights() {
        let c = small_aligned_config();
        let w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        let tts = Qwen3TtsTts::new(c.clone(), w).expect("qwen3_tts tts");
        assert_eq!(tts.config().talker.hidden_dim, c.talker.hidden_dim);
        assert_eq!(tts.config().sample_rate, 24_000);
        assert!(tts.is_synthesized());
        // Default codec is the canonical 12hz one.
        assert_eq!(tts.codec_config().num_quantizers, 16);
    }

    #[test]
    fn tts_new_rejects_talker_block_count_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.talker.blocks.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_code_predictor_block_count_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.code_predictor.blocks.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_text_embed_shape_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.talker.text_embed.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_speech_embed_shape_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.talker.speech_embed.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_code_head_shape_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.code_predictor.code_head.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_block_qkv_size_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.talker.blocks[0].q_proj.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_ffn_gate_size_mismatch() {
        let c = small_aligned_config();
        let mut w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        w.code_predictor.blocks[1].ffn_gate.pop();
        assert!(matches!(
            Qwen3TtsTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_with_codec_config_accepts_matching_variant() {
        let c = small_aligned_config();
        let w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        let tts = Qwen3TtsTts::new(c, w).expect("qwen3_tts tts");
        // The default codec is already qwen3_tts_12hz — swap in an identical
        // hypothetical future variant.
        let mut variant = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        variant.num_semantic_quantizers = 2;
        let tts2 = tts
            .with_codec_config(variant)
            .expect("matching group count accepted");
        assert_eq!(tts2.codec_config().num_semantic_quantizers, 2);
    }

    #[test]
    fn tts_with_codec_config_rejects_mismatched_group_count() {
        let c = small_aligned_config();
        let w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        let tts = Qwen3TtsTts::new(c, w).expect("qwen3_tts tts");
        let mut variant = Qwen3TtsCodecConfig::qwen3_tts_12hz();
        variant.num_quantizers = 8; // != talker's 16
        assert!(matches!(
            tts.with_codec_config(variant),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_text() {
        let c = small_aligned_config();
        let w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        let tts = Qwen3TtsTts::new(c, w).expect("qwen3_tts tts");
        assert!(matches!(
            tts.synthesize(""),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path (synthesized weights) names the
    /// synthesized blocker (FR-EX-08 — never a silent zero-fill /
    /// hallucinated waveform). The synthesized branch fires **before**
    /// the fallthrough real-weight branch, so the message must name the
    /// synthesized-weight blocker and must not name the real-weight
    /// fallthrough branch (which would mean the wrong branch fired).
    #[test]
    fn synthesize_with_synthesized_weights_is_loud_not_implemented() {
        let c = small_aligned_config();
        let w = Qwen3TtsWeights::synthesized(&c, 7).expect("weights");
        let tts = Qwen3TtsTts::new(c, w).expect("qwen3_tts tts");
        let err = tts.synthesize("hello").unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized blocker: {msg}"
                );
                // The fallthrough real-weight branch would name
                // "real weights are bound" — confirm we did NOT reach
                // that arm (the branch ordering is synthesized first,
                // fallthrough second).
                assert!(
                    !msg.contains("real weights are bound"),
                    "must not reach the real-weight fallthrough branch: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// Qwen3-TTS id to Permissive (apache-2.0 — end-to-end). Cross-
    /// crate test to keep this module's registry-side contract honest.
    #[test]
    fn registry_lookup_maps_qwen3_tts_to_permissive_apache() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "qwen3-tts",
            "qwen3_tts",
            "qwen3-tts-0.6b",
            "qwen3-tts-0_6b",
            "qwen3-tts-12hz-0.6b-base",
            "qwen3-tts-12hz-0_6b-base",
            "qwen3-tts-12hz-0.6b",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (apache-2.0)"
            );
        }
    }

    // ---- SoTA reuse bundle (2026-07-30): variant enum + 1.7B fork ----

    #[test]
    fn variant_model_id_slugs_are_stable() {
        assert_eq!(Qwen3TtsVariant::H0_6B.model_id(), "qwen3-tts-0.6b");
        assert_eq!(Qwen3TtsVariant::H1_7B.model_id(), "qwen3-tts-1.7b");
    }

    /// The 1.7B talker config is pinned to the official released axes.
    #[test]
    fn qwen3_tts_1_7b_talker_matches_official_config() {
        let t = Qwen3TtsTalkerConfig::qwen3_tts_1_7b_base();
        assert_eq!(t.hidden_dim, 2048);
        assert_eq!(t.n_layer, 28);
        assert_eq!(t.ffn_dim, 6144);
        assert_eq!(t.text_hidden_size, 2048);
        assert_eq!(t.vocab_size, 3072);
        assert_eq!(t.text_vocab_size, 151_936);
        assert_eq!(t.max_position_embeddings, 32_768);
        assert_eq!(t.n_head, 16, "family-fixed Q head count");
        assert_eq!(t.n_head_kv, 8, "family-fixed KV head count (GQA)");
        assert_eq!(t.head_dim, 128, "family-fixed head_dim");
        assert!((t.rope_base - 1_000_000.0).abs() < 1.0);
        assert!((t.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(t.position_id_per_seconds, 13);
        assert_eq!(
            t.num_code_groups, QWEN3_TTS_NUM_CODE_GROUPS,
            "codec handshake must match the 16 groups the shared codec expects"
        );
    }

    /// The 1.7B full config uses the widened talker and 2048-wide Base
    /// speaker encoder while retaining the 1024-wide code predictor.
    #[test]
    fn qwen3_tts_1_7b_config_pins_widened_talker_and_speaker() {
        let c = Qwen3TtsConfig::qwen3_tts_1_7b_base();
        assert_eq!(c.sample_rate, QWEN3_TTS_SAMPLE_RATE);
        assert_eq!(c.speaker_embed_dim, QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM);
        assert!(c.has_speaker_encoder);
        assert_eq!(
            c.code_predictor,
            Qwen3TtsCodePredictorConfig::qwen3_tts_0_6b_base(),
            "1.7B code predictor must reuse 0.6B constants verbatim"
        );
        assert_eq!(c.talker.hidden_dim, 2048);
        assert_eq!(c.talker.ffn_dim, 6144);
    }

    /// The official 1.7B config is now forward-safe at the shape gate.
    #[test]
    fn qwen3_tts_1_7b_config_passes_shape_validation() {
        let c = Qwen3TtsConfig::qwen3_tts_1_7b_base();
        c.validate_for_forward().expect("official 1.7B config");
    }

    /// `for_variant()` dispatches correctly to the two config methods.
    #[test]
    fn for_variant_dispatches_to_matching_config() {
        assert_eq!(
            Qwen3TtsConfig::for_variant(Qwen3TtsVariant::H0_6B),
            Qwen3TtsConfig::qwen3_tts_0_6b_base()
        );
        assert_eq!(
            Qwen3TtsConfig::for_variant(Qwen3TtsVariant::H1_7B),
            Qwen3TtsConfig::qwen3_tts_1_7b_base()
        );
    }

    /// Variants have distinct talker widths / FFN widths — a converter that
    /// silently picks the wrong variant would mis-slot the talker weights.
    #[test]
    fn variants_have_distinct_talker_shapes() {
        let h06 = Qwen3TtsConfig::qwen3_tts_0_6b_base();
        let h17 = Qwen3TtsConfig::qwen3_tts_1_7b_base();
        assert_ne!(h06.talker.hidden_dim, h17.talker.hidden_dim);
        assert_ne!(h06.talker.ffn_dim, h17.talker.ffn_dim);
        assert_eq!(h06.talker.n_layer, h17.talker.n_layer);
        // Family-invariant axes are shared (not part of the distinction).
        assert_eq!(h06.talker.n_head, h17.talker.n_head);
        assert_eq!(h06.talker.n_head_kv, h17.talker.n_head_kv);
        assert_eq!(h06.talker.head_dim, h17.talker.head_dim);
    }

    /// Never materialize canonical 1.7B synthesized F32 weights in a unit
    /// test; the shape gate alone proves the released config is admitted.
    #[test]
    fn released_1_7b_shape_gate_does_not_allocate_weights() {
        Qwen3TtsConfig::qwen3_tts_1_7b_base()
            .validate_for_forward()
            .expect("official 1.7B shape contract");
    }
}
