//! **Chatterbox-Turbo** — Resemble AI's 350M-parameter distilled Turbo
//! variant of Chatterbox (SoTA plan Phase 3, 2026-07-24). MIT.
//!
//! # What Chatterbox-Turbo is (primary source)
//!
//! `ResembleAI/chatterbox-turbo` is Resemble AI's production-optimised
//! Turbo release of the Chatterbox TTS family. Compared to the base
//! [`crate::chatterbox`] release (500M-parameter T3 on a **Llama_520M**
//! backbone at 24 kHz with a 2454-token multilingual text tokenizer), the
//! Turbo variant swaps the backbone family and reshapes the token axes
//! for low-latency serving:
//!
//! - **Backbone family**: effective inference code uses GPT-2-medium
//!   (24 layers × 16 heads × 1024 hidden, context/positions 8196). The
//!   historical training YAML's 30-layer values are not authoritative.
//! - **Sample rate**: `32_000` Hz — the Turbo config sets
//!   `sample_rate: 32000`, distinct from the base's 24 kHz
//!   (`s3gen/const.py::S3GEN_SR`).
//! - **Text-token vocabulary**: 50 276 — the GPT-2 vocabulary
//!   (50 257 base tokens) plus 19 native paralinguistic tags shipped
//!   in `added_tokens.json` (`[angry]`, `[fear]`, `[surprised]`,
//!   `[whispering]`, `[cough]`, `[laugh]`, `[chuckle]`, and more). The
//!   base Chatterbox multilingual vocab is 2454 and English-only 704 —
//!   Turbo's token space is orders of magnitude larger to support
//!   inline paralinguistic annotation.
//! - **Speech-token vocabulary**: 6563 — the base ships 8194; Turbo
//!   shrinks the codebook via distillation.
//! - **Max text / speech tokens**: effective inference limits are 2048 /
//!   4096; stale YAML short-context values are not runtime facts.
//! - **Distilled decoder**: the speech-token-to-mel decoder has been
//!   distilled from 10 sampling steps to a single step (per the
//!   release notes on the model card). The vocoder terminal is still
//!   the S3Gen HiFT-GAN family — the same `HiFTGenerator` topology
//!   CosyVoice2 / CosyVoice3 / base Chatterbox use, wired through the
//!   shared [`HiFTChain`] seam.
//! - **Paralinguistic tags** (`added_tokens.json`): `[angry]` /
//!   `[fear]` / `[surprised]` / `[whispering]` / `[advertisement]` /
//!   `[dramatic]` / `[narration]` / `[crying]` / `[happy]` /
//!   `[sarcastic]` / `[clear throat]` / `[sigh]` / `[shush]` /
//!   `[cough]` / `[groan]` / `[sniff]` / `[gasp]` / `[chuckle]` /
//!   `[laugh]` — 19 total.
//!
//! ## License and files
//!
//! - **License**: MIT (`github.com/resemble-ai/chatterbox/LICENSE` —
//!   Copyright (c) 2025 Resemble AI, fetched 2026-07-24 — CLAUDE.md
//!   「ハルシネーション厳禁」). The whole Chatterbox family (base +
//!   Turbo + `-multilingual-*` variants) ships under a single MIT
//!   LICENSE.
//! - **Weights & code**: `huggingface.co/ResembleAI/chatterbox-turbo`.
//!   Backbone weights: `t3_turbo_v1.safetensors` (~1.92 GB); vocoder:
//!   `s3gen.safetensors` (~1.06 GB) + `s3gen_meanflow.safetensors`
//!   (~1.06 GB); voice encoder: `ve.safetensors` (~5.7 MB);
//!   configuration: `t3_turbo_v1.yaml` (~8.46 kB); tokenizer:
//!   `vocab.json` + `merges.txt` + `tokenizer_config.json` +
//!   `added_tokens.json` + `special_tokens_map.json` (GPT-2 BPE +
//!   paralinguistic tags).
//!
//! # What lands in this Phase 3 slice
//!
//! - [`ChatterboxTurboConfig`] — every architectural hparam
//!   recorded from effective upstream inference code (the YAML is a stale
//!   training record) plus the
//!   fixed vocoder sample rate and predicates that pin the axes
//!   distinguishing Turbo from base ([`ChatterboxTurboConfig::is_turbo`],
//!   [`ChatterboxTurboConfig::has_paralinguistic_tags`]).
//! - [`ChatterboxTurboWeights`] — deterministic
//!   [`ChatterboxTurboWeights::synthesized`] fixture (SplitMix64 seed
//!   plus Xavier initialisation) so shape / dtype / size flow can be
//!   exercised without the real HF checkpoint.
//! - [`ChatterboxTurboTts`] — engine handle carrying config plus weights
//!   plus an optional [`HiFTChain`]. [`ChatterboxTurboTts::synthesize`]
//!   returns [`VokraError::NotImplemented`] until real weights are
//!   bound and the GPT-2-medium ⇒ distilled 1-step mel decoder ⇒
//!   S3Gen HiFT-GAN chain is wired end-to-end (T29-equivalent
//!   follow-up wave).
//!
//! # No ONNX (permanent)
//!
//! Chatterbox-Turbo also ships a separate `ResembleAI/chatterbox-turbo-ONNX`
//! release. The runtime **never** loads an ONNX graph (FR-LD-05,
//! permanent constraint); the pipeline is re-implemented natively from
//! the safetensors checkpoint (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

mod bound;
pub use crate::chatterbox_family::{
    ChatterboxVariant, CompositeBinderEvidence, GenerationTopology, Gpt2T3Contract, Gpt2Tokenizer,
    RandomDraws, SOURCE_REVISION, SOURCE_URL, SamplingConfig, T3Architecture, apply_processors,
    generation_topology, processor_order, punc_norm_gpt2, remove_terminal_eos, sample_with_draw,
};
pub use bound::{ChatterboxTurboCheckpoint, ChatterboxTurboSpeakerProjection};

// ---------------------------------------------------------------------------
// Public seam re-exports (SoTA plan §1(a) 訂正 shared with CosyVoice2/3)
// ---------------------------------------------------------------------------
//
// Chatterbox-Turbo's terminal vocoder is HiFT-GAN — the exact same
// `HiFTGenerator` topology CosyVoice2 / CosyVoice3 / base Chatterbox
// wire through `crate::cosyvoice2::hift_chain::HiFTChain`. Re-export the
// aliases here so a caller wiring Chatterbox-Turbo sees the seam under
// its own module path without a shape-drift wrapper (mirrors
// `crate::chatterbox` / `crate::cosyvoice3`).

pub use crate::cosyvoice2::{HiFTChain, HiFTChainConfig, HiFTChainWeights};

/// `vokra.model.arch` a Chatterbox-Turbo GGUF must carry. Written by
/// `vokra-convert::models::chatterbox_turbo::ARCH`. Intentionally
/// **distinct** from base Chatterbox's `"chatterbox"` so the runtime
/// can label the loaded model correctly in telemetry / logs / model
/// cards (Turbo swaps backbone family + sample rate + token axes, so
/// silently sharing the base arch tag would misrepresent the loaded
/// model). The compliance registry (`vokra_core::compliance`) knows
/// every `chatterbox-turbo*` spelling as
/// [`vokra_core::LicenseClass::Permissive`] (MIT — no runtime-side
/// attribution obligation).
pub const EXPECTED_ARCH: &str = "chatterbox_turbo";

/// PCM sample rate Chatterbox-Turbo emits (Hz). Fixed by the S3Gen
/// Turbo vocoder's output rate (`t3_turbo_v1.yaml::sample_rate = 32000`)
/// — **distinct** from base Chatterbox's 24 kHz.
pub const CHATTERBOX_TURBO_SAMPLE_RATE: u32 = 32_000;

/// Text-token vocabulary size fixed by the Turbo release: GPT-2 base
/// vocabulary (50 257) + 19 paralinguistic tags = 50 276.
/// (`t3_turbo_v1.yaml::text_tokens_dict_size = 50276`).
pub const TEXT_VOCAB_TURBO: u32 = 50_276;

/// Number of paralinguistic tags natively supported by the Turbo model
/// (`added_tokens.json` — `[angry]`, `[fear]`, `[surprised]`,
/// `[whispering]`, `[cough]`, `[laugh]`, `[chuckle]`, etc.).
pub const PARALINGUISTIC_TAG_COUNT: u32 = 19;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Chatterbox-Turbo T3 architectural hparams.
///
/// Effective inference values are recorded from upstream source code;
/// the YAML sidecar is a stale training record and is not runtime authority.
/// [`ChatterboxTurboConfig::chatterbox_turbo_v1`] constructor is the
/// canonical Turbo config.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatterboxTurboConfig {
    /// Output PCM sample rate, Hz. Fixed at 32 kHz by the Turbo S3Gen
    /// vocoder — distinct from base Chatterbox's 24 kHz.
    pub sample_rate: u32,
    /// GPT-2-medium text-token vocabulary size (`t3_turbo_v1.yaml::text_tokens_dict_size`).
    /// `50_276` = GPT-2 base vocab (50 257) + 19 paralinguistic tags.
    pub text_vocab_size: u32,
    /// T3 speech-token vocabulary size (`t3_turbo_v1.yaml::speech_tokens_dict_size`).
    /// `6563` — smaller than base Chatterbox's 8194 (distilled).
    pub speech_vocab_size: u32,
    /// Max text-token positions from effective Turbo inference. `2048`.
    pub max_text_tokens: u32,
    /// Max speech-token positions from effective Turbo inference. `4096`.
    pub max_speech_tokens: u32,
    /// Speaker-embedding dimension (`t3_turbo_v1.yaml::speaker_embed_size`).
    /// `256`.
    pub speaker_embed_size: u32,
    /// Voice-encoder hidden dimension (`t3_turbo_v1.yaml::ve_hidden_size`).
    /// `768`. Used by the voice-encoder branch feeding the speaker
    /// embedding.
    pub ve_hidden_size: u32,
    /// GPT-2-medium backbone hidden dimension
    /// (`t3_turbo_v1.yaml::legacy_gpt_hidden_size` == `n_gpt_channels`).
    /// `1024`.
    pub hidden_dim: u32,
    /// GPT-2-medium effective inference transformer block count. `24`.
    pub n_layer: u32,
    /// GPT-2-medium backbone attention head count
    /// (`t3_turbo_v1.yaml::n_transformer_heads`). `16` — GPT-2 uses
    /// MHA, so `n_head_kv == n_head`.
    pub n_head: u32,
    /// GPT-2-medium backbone attention head dimension.
    /// Derived: `hidden_dim / n_head = 1024 / 16 = 64`.
    pub head_dim: u32,
    /// STFT hop size (`t3_turbo_v1.yaml::hop_size`). `320` samples
    /// (10 ms at 32 kHz).
    pub hop_size: u32,
    /// STFT window size (`t3_turbo_v1.yaml::win_size`). `2048` samples.
    pub win_size: u32,
    /// Number of mel bins (`t3_turbo_v1.yaml::num_mels`). `256`.
    pub num_mels: u32,
    /// Effective inference speech-conditioning prompt length. `375`.
    pub speech_cond_prompt_len: u32,
    /// Number of native paralinguistic tags in `added_tokens.json`
    /// (`[angry]`, `[fear]`, `[surprised]`, `[whispering]`, `[cough]`,
    /// `[laugh]`, `[chuckle]`, etc.). `19`.
    pub paralinguistic_tag_count: u32,
    /// Start-of-text token id (`t3_turbo_v1.yaml::start_text_token`). `255`.
    pub start_text_token: u32,
    /// End-of-text token id (`t3_turbo_v1.yaml::stop_text_token`). `0`.
    pub stop_text_token: u32,
    /// Start-of-speech token id (`t3_turbo_v1.yaml::start_speech_token`). `6561`.
    pub start_speech_token: u32,
    /// End-of-speech token id (`t3_turbo_v1.yaml::stop_speech_token`). `6562`.
    pub stop_speech_token: u32,
}

impl ChatterboxTurboConfig {
    /// Canonical Chatterbox-Turbo T3 config
    /// (`ResembleAI/chatterbox-turbo`, `t3_turbo_v1.safetensors`).
    ///
    /// Effective inference values are recorded from upstream source code;
    /// the YAML sidecar is not runtime authority.
    #[must_use]
    pub fn chatterbox_turbo_v1() -> Self {
        Self {
            sample_rate: CHATTERBOX_TURBO_SAMPLE_RATE,
            text_vocab_size: TEXT_VOCAB_TURBO,
            speech_vocab_size: 6563,
            max_text_tokens: 2048,
            max_speech_tokens: 4096,
            speaker_embed_size: 256,
            ve_hidden_size: 768,
            hidden_dim: 1024,
            n_layer: 24,
            n_head: 16,
            head_dim: 64,
            hop_size: 320,
            win_size: 2048,
            num_mels: 256,
            speech_cond_prompt_len: 375,
            paralinguistic_tag_count: PARALINGUISTIC_TAG_COUNT,
            start_text_token: 255,
            stop_text_token: 0,
            start_speech_token: 6561,
            stop_speech_token: 6562,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA well-formedness, positive FFN dim,
    /// non-zero vocab) mirror the real model. `text_vocab_size` is
    /// set to `32` (not the 50 276 sentinel), so
    /// [`Self::is_turbo`] returns `false` for the tiny fixture —
    /// callers relying on that predicate should use
    /// [`Self::chatterbox_turbo_v1`] instead.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: CHATTERBOX_TURBO_SAMPLE_RATE,
            text_vocab_size: 32,
            speech_vocab_size: 64,
            max_text_tokens: 16,
            max_speech_tokens: 16,
            speaker_embed_size: 8,
            ve_hidden_size: 8,
            hidden_dim: 16,
            n_layer: 2,
            n_head: 2,
            head_dim: 8,
            hop_size: 4,
            win_size: 8,
            num_mels: 4,
            speech_cond_prompt_len: 4,
            paralinguistic_tag_count: PARALINGUISTIC_TAG_COUNT,
            start_text_token: 30,
            stop_text_token: 0,
            start_speech_token: 60,
            stop_speech_token: 61,
        }
    }

    /// True iff `text_vocab_size == 50_276` — the primary-source flag
    /// distinguishing the Turbo variant (GPT-2 vocab + 19 tags) from
    /// the base Chatterbox multilingual (2454) / English-only (704)
    /// text-token spaces.
    #[must_use]
    pub fn is_turbo(&self) -> bool {
        self.text_vocab_size == TEXT_VOCAB_TURBO
    }

    /// True iff the config declares native paralinguistic tag support
    /// (`paralinguistic_tag_count > 0`) — every real Turbo checkpoint
    /// ships 19 tags in `added_tokens.json`.
    #[must_use]
    pub fn has_paralinguistic_tags(&self) -> bool {
        self.paralinguistic_tag_count > 0
    }

    /// True iff every architectural axis is at its `0` sentinel — the
    /// shape-only conversion path the runtime tolerates as
    /// inspectable-but-not-forward-ready.
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.text_vocab_size == 0
            && self.speech_vocab_size == 0
            && self.hidden_dim == 0
            && self.n_layer == 0
            && self.n_head == 0
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the GPT-2 backbone cross-checks
    /// (`hidden_dim == n_head * head_dim`, MHA) plus positivity on
    /// every axis.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.sample_rate == 0
            || self.text_vocab_size == 0
            || self.speech_vocab_size == 0
            || self.hidden_dim == 0
            || self.n_layer == 0
            || self.n_head == 0
            || self.head_dim == 0
            || self.max_text_tokens == 0
            || self.max_speech_tokens == 0
            || self.speaker_embed_size == 0
            || self.ve_hidden_size == 0
            || self.hop_size == 0
            || self.win_size == 0
            || self.num_mels == 0
            || self.speech_cond_prompt_len == 0
        {
            return Err(VokraError::InvalidArgument(
                "chatterbox_turbo config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use ChatterboxTurboConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        // GPT-2 backbone algebra: hidden_dim = n_head * head_dim (MHA).
        if self.hidden_dim != self.n_head * self.head_dim {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox_turbo config: hidden_dim ({}) must equal n_head ({}) * head_dim \
                 ({}) — got {} vs expected {}",
                self.hidden_dim,
                self.n_head,
                self.head_dim,
                self.hidden_dim,
                self.n_head * self.head_dim,
            )));
        }
        // Speech / text stop tokens must be inside the corresponding vocabulary.
        if self.stop_text_token >= self.text_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox_turbo config: stop_text_token ({}) must be < text_vocab_size ({})",
                self.stop_text_token, self.text_vocab_size,
            )));
        }
        if self.stop_speech_token >= self.speech_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox_turbo config: stop_speech_token ({}) must be < speech_vocab_size \
                 ({})",
                self.stop_speech_token, self.speech_vocab_size,
            )));
        }
        // STFT window must be at least the hop size (framing well-formedness).
        if self.win_size < self.hop_size {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox_turbo config: win_size ({}) must be >= hop_size ({})",
                self.win_size, self.hop_size,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights (scaffold — real binding delegates to a follow-up wave)
// ---------------------------------------------------------------------------

/// Chatterbox-Turbo GPT-2-medium weight store scaffold.
///
/// Carries the text-embedding + speech-embedding + speaker-embedding
/// projection + GPT-2 backbone stack. [`Self::synthesized`] builds a
/// deterministic fixture (SplitMix64 + Xavier) against `config` so
/// shape / dtype / size can be exercised without the real HF
/// checkpoint. Real-checkpoint binding is a follow-up (T29-equivalent,
/// the CosyVoice2 / CSM / base Chatterbox pattern).
#[derive(Debug, Clone)]
pub struct ChatterboxTurboWeights {
    /// Text-token embedding: `[text_vocab_size, hidden_dim]`.
    pub text_embed: Vec<f32>,
    /// Speech-token embedding: `[speech_vocab_size, hidden_dim]`.
    pub speech_embed: Vec<f32>,
    /// Speaker-embedding projection to backbone hidden width:
    /// `[speaker_embed_size, hidden_dim]`. Turbo skips the Perceiver
    /// resampler the base Chatterbox uses
    /// (`t3_turbo_v1.yaml::use_perceiver_resampler = false`).
    pub speaker_proj: Vec<f32>,
    /// Voice-encoder projection producing the speaker embedding:
    /// `[ve_hidden_size, speaker_embed_size]`.
    pub voice_encoder_proj: Vec<f32>,
    /// Per-layer transformer block scaffolds. Length = `n_layer`.
    pub blocks: Vec<ChatterboxTurboBlockWeights>,
    /// Final layer-norm γ, shape `[hidden_dim]` (GPT-2 uses LayerNorm,
    /// not RMSNorm).
    pub final_norm: Vec<f32>,
    /// Final layer-norm β, shape `[hidden_dim]` (GPT-2 LayerNorm has
    /// both scale and shift).
    pub final_norm_bias: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

/// Per-transformer-block weights (MHA self-attention + FFN, the GPT-2
/// block topology).
///
/// GPT-2 differs from Llama-style transformers in three material ways:
/// * LayerNorm (with bias) instead of RMSNorm;
/// * Fused Q/K/V projection with bias (GPT-2 packs the three matrices
///   into a single `[3 * hidden_dim, hidden_dim]` `c_attn` weight —
///   this scaffold splits them for clarity but the shapes stay
///   equivalent);
/// * GELU activation on a 2-linear FFN (no SwiGLU gate).
#[derive(Debug, Clone)]
pub struct ChatterboxTurboBlockWeights {
    /// Pre-self-attention LayerNorm γ, shape `[hidden_dim]`.
    pub self_attn_norm: Vec<f32>,
    /// Pre-self-attention LayerNorm β, shape `[hidden_dim]`.
    pub self_attn_norm_bias: Vec<f32>,
    /// Q projection, shape `[hidden_dim, hidden_dim]` (MHA — GPT-2
    /// packs QKV, we split for clarity).
    pub q_proj: Vec<f32>,
    /// Q bias, shape `[hidden_dim]` (GPT-2 has attention biases).
    pub q_bias: Vec<f32>,
    /// K projection, shape `[hidden_dim, hidden_dim]`.
    pub k_proj: Vec<f32>,
    /// K bias, shape `[hidden_dim]`.
    pub k_bias: Vec<f32>,
    /// V projection, shape `[hidden_dim, hidden_dim]`.
    pub v_proj: Vec<f32>,
    /// V bias, shape `[hidden_dim]`.
    pub v_bias: Vec<f32>,
    /// O projection, shape `[hidden_dim, hidden_dim]`.
    pub o_proj: Vec<f32>,
    /// O bias, shape `[hidden_dim]`.
    pub o_bias: Vec<f32>,
    /// FFN pre-LayerNorm γ, shape `[hidden_dim]`.
    pub ffn_norm: Vec<f32>,
    /// FFN pre-LayerNorm β, shape `[hidden_dim]`.
    pub ffn_norm_bias: Vec<f32>,
    /// FFN up projection, shape `[ffn_dim, hidden_dim]` where
    /// `ffn_dim = 4 * hidden_dim` for GPT-2 (transcribed from GPT-2
    /// medium: 4 × 1024 = 4096).
    pub ffn_up: Vec<f32>,
    /// FFN up bias, shape `[ffn_dim]`.
    pub ffn_up_bias: Vec<f32>,
    /// FFN down projection, shape `[hidden_dim, ffn_dim]`.
    pub ffn_down: Vec<f32>,
    /// FFN down bias, shape `[hidden_dim]`.
    pub ffn_down_bias: Vec<f32>,
}

impl ChatterboxTurboWeights {
    /// GPT-2 medium FFN inner dimension is fixed at `4 * hidden_dim`
    /// per the GPT-2 spec — a private helper so callers do not have
    /// to know the ratio.
    fn ffn_dim(config: &ChatterboxTurboConfig) -> usize {
        (config.hidden_dim as usize) * 4
    }

    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via
    /// a [`SplitMix64`] stream — reproducible, allocation-only,
    /// zero-dep. Every LayerNorm γ starts at `1.0`; every bias / β
    /// starts at `0.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if
    /// `config.validate_for_forward` fails.
    pub fn synthesized(config: &ChatterboxTurboConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let d = config.hidden_dim as usize;
        let ffn = Self::ffn_dim(config);
        let text_vocab = config.text_vocab_size as usize;
        let speech_vocab = config.speech_vocab_size as usize;
        let spk = config.speaker_embed_size as usize;
        let ve = config.ve_hidden_size as usize;

        let text_embed = xavier(&mut rng, text_vocab * d, d, d);
        let speech_embed = xavier(&mut rng, speech_vocab * d, d, d);
        let speaker_proj = xavier(&mut rng, spk * d, spk, d);
        let voice_encoder_proj = xavier(&mut rng, ve * spk, ve, spk);

        let mut blocks = Vec::with_capacity(config.n_layer as usize);
        for _ in 0..config.n_layer {
            blocks.push(ChatterboxTurboBlockWeights {
                self_attn_norm: vec![1.0; d],
                self_attn_norm_bias: vec![0.0; d],
                q_proj: xavier(&mut rng, d * d, d, d),
                q_bias: vec![0.0; d],
                k_proj: xavier(&mut rng, d * d, d, d),
                k_bias: vec![0.0; d],
                v_proj: xavier(&mut rng, d * d, d, d),
                v_bias: vec![0.0; d],
                o_proj: xavier(&mut rng, d * d, d, d),
                o_bias: vec![0.0; d],
                ffn_norm: vec![1.0; d],
                ffn_norm_bias: vec![0.0; d],
                ffn_up: xavier(&mut rng, ffn * d, d, ffn),
                ffn_up_bias: vec![0.0; ffn],
                ffn_down: xavier(&mut rng, d * ffn, ffn, d),
                ffn_down_bias: vec![0.0; d],
            });
        }
        let final_norm = vec![1.0; d];
        let final_norm_bias = vec![0.0; d];

        Ok(Self {
            text_embed,
            speech_embed,
            speaker_proj,
            voice_encoder_proj,
            blocks,
            final_norm,
            final_norm_bias,
            is_synthesized: true,
        })
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

/// Chatterbox-Turbo TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional
/// [`HiFTChain`] terminal vocoder (SoTA plan §1(a) 訂正 seam shared
/// with CosyVoice2 / CosyVoice3 / base Chatterbox).
/// [`Self::synthesize`] is the primary text → PCM entry point; until
/// real weights are bound and the GPT-2-medium ⇒ distilled 1-step mel
/// decoder ⇒ S3Gen HiFT-GAN chain is wired end-to-end (T29-equivalent
/// follow-up wave), it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty audio buffer).
#[derive(Debug, Clone)]
pub struct ChatterboxTurboTts {
    cfg: ChatterboxTurboConfig,
    weights: ChatterboxTurboWeights,
    hift_chain: Option<HiFTChain>,
}

impl ChatterboxTurboTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, per-tensor
    /// sizes) so a mismatched pair fails loudly here rather than deep
    /// inside a forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape
    ///   mismatch.
    pub fn new(cfg: ChatterboxTurboConfig, weights: ChatterboxTurboWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let d = cfg.hidden_dim as usize;
        let ffn = ChatterboxTurboWeights::ffn_dim(&cfg);
        let text_vocab = cfg.text_vocab_size as usize;
        let speech_vocab = cfg.speech_vocab_size as usize;
        let spk = cfg.speaker_embed_size as usize;
        let ve = cfg.ve_hidden_size as usize;

        check_len("text_embed", weights.text_embed.len(), text_vocab * d)?;
        check_len("speech_embed", weights.speech_embed.len(), speech_vocab * d)?;
        check_len("speaker_proj", weights.speaker_proj.len(), spk * d)?;
        check_len(
            "voice_encoder_proj",
            weights.voice_encoder_proj.len(),
            ve * spk,
        )?;
        check_len("final_norm", weights.final_norm.len(), d)?;
        check_len("final_norm_bias", weights.final_norm_bias.len(), d)?;

        if weights.blocks.len() != cfg.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox_turbo weights: blocks.len()={} != n_layer={}",
                weights.blocks.len(),
                cfg.n_layer,
            )));
        }
        for (i, blk) in weights.blocks.iter().enumerate() {
            check_block_shapes(i, blk, d, ffn)?;
        }

        Ok(Self {
            cfg,
            weights,
            hift_chain: None,
        })
    }

    /// Injects a [`HiFTChain`] — the terminal mel → PCM vocoder.
    ///
    /// SoTA plan §1(a) 訂正 seam (shared with CosyVoice2 / CosyVoice3
    /// / base Chatterbox). Until a caller provides a [`HiFTChain`],
    /// [`Self::synthesize`] returns [`VokraError::NotImplemented`]
    /// naming the missing vocoder as the blocker (FR-EX-08).
    #[must_use]
    pub fn with_hift_chain(mut self, chain: HiFTChain) -> Self {
        self.hift_chain = Some(chain);
        self
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ChatterboxTurboConfig {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`ChatterboxTurboWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// True iff a [`HiFTChain`] has been injected (the SoTA plan §1(a)
    /// 訂正 seam is present).
    #[must_use]
    pub fn has_hift_chain(&self) -> bool {
        self.hift_chain.is_some()
    }

    /// True iff the underlying config identifies as the Turbo variant
    /// (`text_vocab_size == 50_276`).
    #[must_use]
    pub fn is_turbo(&self) -> bool {
        self.cfg.is_turbo()
    }

    /// Synthesizes PCM for `text` at [`Self::config`]'s sample rate.
    ///
    /// This is the primary text → PCM entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// audio, so this returns [`VokraError::NotImplemented`] naming
    /// the blocker. Callers verify the shape flow through
    /// [`ChatterboxTurboTts::new`] +
    /// [`ChatterboxTurboWeights::synthesized`] today; a follow-up
    /// wave binds real Chatterbox-Turbo weights and wires the forward
    /// (GPT-2-medium → distilled 1-step mel decoder → S3Gen HiFT-GAN
    /// → PCM).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not
    ///   yet bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "chatterbox_turbo synthesize: text is empty".to_owned(),
            ));
        }
        if self.hift_chain.is_none() {
            return Err(VokraError::NotImplemented(
                "chatterbox_turbo synthesize: no HiFTChain has been injected. Call \
                 `.with_hift_chain(HiFTChain::new(cfg, weights)?)` first — Chatterbox-Turbo \
                 uses S3Gen HiFT-GAN as the terminal vocoder (SoTA plan §1(a) 訂正, \
                 2026-07-22), same seam as CosyVoice2 / CosyVoice3 / base Chatterbox. The \
                 vocoder module is shared via `crate::cosyvoice2::hift_chain`.",
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "chatterbox_turbo synthesize: this engine holds synthesized weights \
                 (deterministic fixture from ChatterboxTurboWeights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real \
                 speech. Bind real Chatterbox-Turbo weights (MIT, \
                 huggingface.co/ResembleAI/chatterbox-turbo) before invoking synthesize. \
                 The shape flow (config validation, weight-store construction, text-empty \
                 check) is exercised through ChatterboxTurboTts::new; real-checkpoint \
                 binding lands in a follow-up wave (T29-equivalent) that wires the \
                 GPT-2-medium backbone → distilled 1-step mel decoder → S3Gen HiFT-GAN \
                 chain.",
            ));
        }
        Err(VokraError::NotImplemented(
            "chatterbox_turbo synthesize: real weights are bound but the GPT-2-medium \
             backbone → speech-token AR sampling → distilled 1-step mel decoder → S3Gen \
             HiFT-GAN vocoder forward path has not landed yet. Follow-up wave: wire the \
             GPT-2 primitives (LayerNorm with bias / MHA / GELU FFN — different from base \
             Chatterbox's Llama-style RMSNorm + SwiGLU) and feed the sampled speech \
             tokens through the distilled 1-step S3Gen chain to the HiFTChain seam. The \
             LayerNorm-with-bias / QKV-with-bias / GELU inventory extends base \
             Chatterbox's Llama op set; the S3Gen HiFT-GAN vocoder body is shared with \
             CosyVoice2 / CosyVoice3 / base Chatterbox.",
        ))
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox_turbo weights: {name}.len()={got} != {expected}"
        )));
    }
    Ok(())
}

fn check_block_shapes(
    i: usize,
    blk: &ChatterboxTurboBlockWeights,
    d: usize,
    ffn: usize,
) -> Result<()> {
    check_len(
        &format!("block[{i}].self_attn_norm"),
        blk.self_attn_norm.len(),
        d,
    )?;
    check_len(
        &format!("block[{i}].self_attn_norm_bias"),
        blk.self_attn_norm_bias.len(),
        d,
    )?;
    check_len(&format!("block[{i}].q_proj"), blk.q_proj.len(), d * d)?;
    check_len(&format!("block[{i}].q_bias"), blk.q_bias.len(), d)?;
    check_len(&format!("block[{i}].k_proj"), blk.k_proj.len(), d * d)?;
    check_len(&format!("block[{i}].k_bias"), blk.k_bias.len(), d)?;
    check_len(&format!("block[{i}].v_proj"), blk.v_proj.len(), d * d)?;
    check_len(&format!("block[{i}].v_bias"), blk.v_bias.len(), d)?;
    check_len(&format!("block[{i}].o_proj"), blk.o_proj.len(), d * d)?;
    check_len(&format!("block[{i}].o_bias"), blk.o_bias.len(), d)?;
    check_len(&format!("block[{i}].ffn_norm"), blk.ffn_norm.len(), d)?;
    check_len(
        &format!("block[{i}].ffn_norm_bias"),
        blk.ffn_norm_bias.len(),
        d,
    )?;
    check_len(&format!("block[{i}].ffn_up"), blk.ffn_up.len(), ffn * d)?;
    check_len(
        &format!("block[{i}].ffn_up_bias"),
        blk.ffn_up_bias.len(),
        ffn,
    )?;
    check_len(&format!("block[{i}].ffn_down"), blk.ffn_down.len(), d * ffn)?;
    check_len(
        &format!("block[{i}].ffn_down_bias"),
        blk.ffn_down_bias.len(),
        d,
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
    fn expected_arch_is_chatterbox_turbo() {
        assert_eq!(EXPECTED_ARCH, "chatterbox_turbo");
    }

    #[test]
    fn arch_is_distinct_from_base_chatterbox() {
        // The Turbo variant swaps backbone family (GPT-2 vs Llama_520M) +
        // sample rate (32 kHz vs 24 kHz) + text vocab (50 276 vs 2454/704),
        // so silently sharing the base arch tag would misrepresent the
        // loaded model in telemetry / logs / model cards.
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
    }

    #[test]
    fn sample_rate_matches_s3gen_turbo_output() {
        // `t3_turbo_v1.yaml::sample_rate = 32000`
        assert_eq!(CHATTERBOX_TURBO_SAMPLE_RATE, 32_000);
    }

    #[test]
    fn sample_rate_is_distinct_from_base_chatterbox() {
        // Base Chatterbox = 24 kHz, Turbo = 32 kHz — a real distinction
        // the runtime must honour when routing to the vocoder.
        assert_ne!(
            CHATTERBOX_TURBO_SAMPLE_RATE,
            crate::chatterbox::CHATTERBOX_SAMPLE_RATE
        );
    }

    /// Text-vocab sentinel matches the primary-source Turbo constant —
    /// flipping it would silently change what `is_turbo()` reports for
    /// a real checkpoint.
    #[test]
    fn text_vocab_sentinel_matches_upstream_turbo_config() {
        assert_eq!(TEXT_VOCAB_TURBO, 50_276);
    }

    #[test]
    fn paralinguistic_tag_count_matches_added_tokens_json() {
        // `added_tokens.json` — [angry] / [fear] / [surprised] / [whispering] /
        // [advertisement] / [dramatic] / [narration] / [crying] / [happy] /
        // [sarcastic] / [clear throat] / [sigh] / [shush] / [cough] /
        // [groan] / [sniff] / [gasp] / [chuckle] / [laugh] = 19.
        assert_eq!(PARALINGUISTIC_TAG_COUNT, 19);
    }

    /// Effective architectural axes are stable and the Turbo predicate fires.
    #[test]
    fn turbo_v1_config_matches_primary_source() {
        let c = ChatterboxTurboConfig::chatterbox_turbo_v1();
        assert_eq!(c.sample_rate, 32_000);
        assert_eq!(c.text_vocab_size, 50_276);
        assert_eq!(c.speech_vocab_size, 6_563);
        assert_eq!(c.max_text_tokens, 2048);
        assert_eq!(c.max_speech_tokens, 4096);
        assert_eq!(c.speaker_embed_size, 256);
        assert_eq!(c.ve_hidden_size, 768);
        assert_eq!(c.hidden_dim, 1024);
        assert_eq!(c.n_layer, 24);
        assert_eq!(c.n_head, 16);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.hop_size, 320);
        assert_eq!(c.win_size, 2048);
        assert_eq!(c.num_mels, 256);
        assert_eq!(c.speech_cond_prompt_len, 375);
        assert_eq!(c.paralinguistic_tag_count, 19);
        assert_eq!(c.start_text_token, 255);
        assert_eq!(c.stop_text_token, 0);
        assert_eq!(c.start_speech_token, 6561);
        assert_eq!(c.stop_speech_token, 6562);
        assert!(c.is_turbo());
        assert!(c.has_paralinguistic_tags());
        assert!(!c.is_placeholder_shape());
        c.validate_for_forward()
            .expect("real Turbo config is well-formed");
        // hidden_dim = n_head * head_dim (MHA)
        assert_eq!(c.hidden_dim, c.n_head * c.head_dim);
    }

    #[test]
    fn tiny_config_is_well_formed_and_turbo_predicates_default_false() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        c.validate_for_forward().expect("tiny config well-formed");
        // Tiny fixture uses vocab=32, not 50 276 — so is_turbo() is
        // false. Real callers relying on the predicate must build the
        // canonical config, not the tiny fixture.
        assert!(!c.is_turbo());
        // But the paralinguistic tag count is preserved (a Turbo checkpoint
        // always ships the 19 tags).
        assert!(c.has_paralinguistic_tags());
        assert_eq!(c.hidden_dim, c.n_head * c.head_dim);
    }

    #[test]
    fn placeholder_config_is_placeholder_shape() {
        let c = ChatterboxTurboConfig {
            sample_rate: CHATTERBOX_TURBO_SAMPLE_RATE,
            text_vocab_size: 0,
            speech_vocab_size: 0,
            max_text_tokens: 0,
            max_speech_tokens: 0,
            speaker_embed_size: 0,
            ve_hidden_size: 0,
            hidden_dim: 0,
            n_layer: 0,
            n_head: 0,
            head_dim: 0,
            hop_size: 0,
            win_size: 0,
            num_mels: 0,
            speech_cond_prompt_len: 0,
            paralinguistic_tag_count: 0,
            start_text_token: 0,
            stop_text_token: 0,
            start_speech_token: 0,
            stop_speech_token: 0,
        };
        assert!(c.is_placeholder_shape());
        assert!(!c.has_paralinguistic_tags());
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_zero_axis() {
        // Each zeroing mutator must trip validate_for_forward.
        let mutators: &[fn(&mut ChatterboxTurboConfig)] = &[
            |c| c.text_vocab_size = 0,
            |c| c.speech_vocab_size = 0,
            |c| c.hidden_dim = 0,
            |c| c.n_layer = 0,
            |c| c.n_head = 0,
            |c| c.head_dim = 0,
            |c| c.sample_rate = 0,
            |c| c.max_text_tokens = 0,
            |c| c.max_speech_tokens = 0,
            |c| c.speaker_embed_size = 0,
            |c| c.ve_hidden_size = 0,
            |c| c.hop_size = 0,
            |c| c.win_size = 0,
            |c| c.num_mels = 0,
            |c| c.speech_cond_prompt_len = 0,
        ];
        for mutate in mutators {
            let mut c = ChatterboxTurboConfig::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_hidden_dim_not_matching_head_split() {
        let mut c = ChatterboxTurboConfig::tiny_for_tests();
        // Original: hidden_dim=16 = n_head 2 * head_dim 8. Break the algebra.
        c.hidden_dim = 24;
        let err = c
            .validate_for_forward()
            .expect_err("mismatched shape fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("hidden_dim"), "message: {msg}");
                assert!(msg.contains("head_dim"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_stop_token_outside_vocabulary() {
        let mut c = ChatterboxTurboConfig::tiny_for_tests();
        c.stop_text_token = c.text_vocab_size; // one past the end
        let err = c
            .validate_for_forward()
            .expect_err("out-of-range stop text token fails");
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let mut c = ChatterboxTurboConfig::tiny_for_tests();
        c.stop_speech_token = c.speech_vocab_size; // one past the end
        let err = c
            .validate_for_forward()
            .expect_err("out-of-range stop speech token fails");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn config_rejects_win_size_smaller_than_hop() {
        let mut c = ChatterboxTurboConfig::tiny_for_tests();
        c.hop_size = 8;
        c.win_size = 4;
        let err = c
            .validate_for_forward()
            .expect_err("win_size < hop_size fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("win_size"), "message: {msg}");
                assert!(msg.contains("hop_size"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let w1 = ChatterboxTurboWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = ChatterboxTurboWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.text_embed, w2.text_embed);
        assert_eq!(w1.speech_embed, w2.speech_embed);
        assert_eq!(w1.speaker_proj, w2.speaker_proj);
        assert_eq!(w1.voice_encoder_proj, w2.voice_encoder_proj);
        assert_eq!(w1.blocks[0].q_proj, w2.blocks[0].q_proj);
        assert_eq!(w1.blocks[1].ffn_up, w2.blocks[1].ffn_up);
        assert!(w1.is_synthesized);

        // Shape flow.
        let d = c.hidden_dim as usize;
        let ffn = ChatterboxTurboWeights::ffn_dim(&c);
        let text_vocab = c.text_vocab_size as usize;
        let speech_vocab = c.speech_vocab_size as usize;
        let spk = c.speaker_embed_size as usize;
        let ve = c.ve_hidden_size as usize;
        assert_eq!(w1.text_embed.len(), text_vocab * d);
        assert_eq!(w1.speech_embed.len(), speech_vocab * d);
        assert_eq!(w1.speaker_proj.len(), spk * d);
        assert_eq!(w1.voice_encoder_proj.len(), ve * spk);
        assert_eq!(w1.final_norm.len(), d);
        assert_eq!(w1.final_norm_bias.len(), d);
        assert_eq!(w1.blocks.len(), c.n_layer as usize);
        for blk in &w1.blocks {
            assert_eq!(blk.self_attn_norm.len(), d);
            assert_eq!(blk.self_attn_norm_bias.len(), d);
            assert_eq!(blk.q_proj.len(), d * d);
            assert_eq!(blk.q_bias.len(), d);
            assert_eq!(blk.k_proj.len(), d * d);
            assert_eq!(blk.k_bias.len(), d);
            assert_eq!(blk.v_proj.len(), d * d);
            assert_eq!(blk.v_bias.len(), d);
            assert_eq!(blk.o_proj.len(), d * d);
            assert_eq!(blk.o_bias.len(), d);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.ffn_norm_bias.len(), d);
            assert_eq!(blk.ffn_up.len(), ffn * d);
            assert_eq!(blk.ffn_up_bias.len(), ffn);
            assert_eq!(blk.ffn_down.len(), d * ffn);
            assert_eq!(blk.ffn_down_bias.len(), d);
        }
    }

    #[test]
    fn ffn_dim_helper_is_four_times_hidden_dim() {
        // GPT-2 medium spec: FFN inner = 4 * hidden_dim.
        let c = ChatterboxTurboConfig::chatterbox_turbo_v1();
        assert_eq!(
            ChatterboxTurboWeights::ffn_dim(&c),
            4 * c.hidden_dim as usize
        );
        // Concretely for the canonical Turbo config: 4096.
        assert_eq!(ChatterboxTurboWeights::ffn_dim(&c), 4096);
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let a = ChatterboxTurboWeights::synthesized(&c, 1).expect("a");
        let b = ChatterboxTurboWeights::synthesized(&c, 2).expect("b");
        assert_ne!(a.text_embed, b.text_embed);
        assert_ne!(a.speech_embed, b.speech_embed);
        assert_ne!(a.blocks[0].q_proj, b.blocks[0].q_proj);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = ChatterboxTurboConfig::tiny_for_tests();
        c.hidden_dim = 0;
        assert!(matches!(
            ChatterboxTurboWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_accepts_matching_config_and_weights() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTurboTts::new(c.clone(), w).expect("chatterbox_turbo tts");
        assert_eq!(tts.config().hidden_dim, c.hidden_dim);
        assert_eq!(tts.config().n_layer, c.n_layer);
        assert_eq!(tts.config().sample_rate, 32_000);
        assert!(tts.is_synthesized());
        assert!(!tts.has_hift_chain(), "fresh load has no HiFTChain");
        assert!(
            !tts.is_turbo(),
            "tiny fixture is not the 50 276-vocab variant"
        );
    }

    #[test]
    fn tts_new_rejects_block_count_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_text_embed_shape_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.text_embed.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_speech_embed_shape_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.speech_embed.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_speaker_proj_shape_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.speaker_proj.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_voice_encoder_proj_shape_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.voice_encoder_proj.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_block_qkv_size_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].q_proj.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_ffn_size_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.blocks[1].ffn_down.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_ffn_bias_size_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].ffn_up_bias.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_final_norm_size_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.final_norm.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_final_norm_bias_size_mismatch() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let mut w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        w.final_norm_bias.pop();
        assert!(matches!(
            ChatterboxTurboTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_text() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTurboTts::new(c, w).expect("chatterbox_turbo tts");
        assert!(matches!(
            tts.synthesize(""),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path (no HiFTChain injected) names the
    /// vocoder blocker (FR-EX-08 — never a silent zero-fill / hallucinated
    /// waveform).
    #[test]
    fn synthesize_without_hift_chain_is_loud_not_implemented() {
        let c = ChatterboxTurboConfig::tiny_for_tests();
        let w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTurboTts::new(c, w).expect("chatterbox_turbo tts");
        let err = tts.synthesize("hello").unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("HiFTChain"),
                    "message must name the vocoder blocker: {msg}"
                );
                assert!(
                    msg.contains("cosyvoice2::hift_chain"),
                    "message must name the shared seam: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// With a HiFTChain injected but synthesized weights, the message
    /// pivots to the synthesized-weight blocker (never a fallthrough to
    /// hallucinated audio).
    #[test]
    fn synthesize_with_chain_and_synthesized_weights_names_the_synthesized_blocker() {
        use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

        // Build a tiny well-formed HiFTChain (shape lifted from
        // `cosyvoice2::small_hift_chain_for_wiring` — identical pattern
        // to the base Chatterbox test).
        let cfg = HiFTChainConfig {
            in_channels: 4,
            base_channels: 8,
            nb_harmonics: 2,
            sampling_rate: 16_000,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            istft_n_fft: 8,
            istft_hop_len: 2,
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1]],
            source_resblock_kernel_sizes: vec![3, 3],
            source_resblock_dilation_sizes: vec![vec![1], vec![1]],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        };
        let mut f0_conv_weights: Vec<Vec<f32>> = vec![vec![0.0; 8 * 4 * 3]];
        for _ in 1..5 {
            f0_conv_weights.push(vec![0.0; 8 * 8 * 3]);
        }
        let f0_weights = F0PredictorWeights {
            conv_weights: f0_conv_weights,
            conv_biases: vec![vec![0.0; 8]; 5],
            linear_w: vec![0.0; 8],
            linear_b: vec![0.0; 1],
        };
        let ups_w = vec![vec![0.0; 8 * 4 * 4], vec![0.0; 4 * 2 * 4]];
        let ups_b = vec![vec![0.0; 4], vec![0.0; 2]];
        let n_fft_plus_2 = 10;
        let source_downs_w = vec![vec![0.0; 4 * n_fft_plus_2 * 4], vec![0.0; 2 * n_fft_plus_2]];
        let source_downs_b = vec![vec![0.0; 4], vec![0.0; 2]];
        let make_res_zero = |ch: usize, k: usize, n_branches: usize| ResBlockWeights {
            convs1_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs1_b: vec![vec![0.0; ch]; n_branches],
            convs2_w: vec![vec![0.0; ch * ch * k]; n_branches],
            convs2_b: vec![vec![0.0; ch]; n_branches],
            activations1_alpha: vec![vec![0.0; ch]; n_branches],
            activations2_alpha: vec![vec![0.0; ch]; n_branches],
        };
        let weights = HiFTChainWeights {
            conv_pre_w: vec![0.0; 8 * 4 * 7],
            conv_pre_b: vec![0.0; 8],
            ups_w,
            ups_b,
            source_downs_w,
            source_downs_b,
            source_resblock_weights: vec![make_res_zero(4, 3, 1), make_res_zero(2, 3, 1)],
            resblock_weights: vec![make_res_zero(4, 3, 1), make_res_zero(2, 3, 1)],
            conv_post_w: vec![0.0; n_fft_plus_2 * 2 * 7],
            conv_post_b: vec![0.0; n_fft_plus_2],
            m_source_linear_w: vec![0.0; 3],
            m_source_linear_b: 0.0,
            f0_predictor_weights: f0_weights,
        };
        let chain = HiFTChain::new(cfg, weights).expect("small HiFTChain builds");

        let c = ChatterboxTurboConfig::tiny_for_tests();
        let w = ChatterboxTurboWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTurboTts::new(c, w)
            .expect("chatterbox_turbo tts")
            .with_hift_chain(chain);
        assert!(tts.has_hift_chain());
        let err = tts.synthesize("hello").unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("synthesized"),
                    "message must name synthesized-weight blocker: {msg}"
                );
                // The vocoder blocker must NOT still be named — the pivot
                // proves the ordering (vocoder first, then synthesized).
                assert!(
                    !msg.contains("HiFTChain"),
                    "after chain injection the vocoder blocker must be resolved: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// Chatterbox-Turbo id to Permissive (MIT — the whole Chatterbox
    /// family ships under `github.com/resemble-ai/chatterbox/LICENSE`).
    /// Cross-crate test to keep this module's registry-side contract
    /// honest.
    #[test]
    fn registry_lookup_maps_chatterbox_turbo_to_permissive_mit() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "chatterbox-turbo",
            "chatterbox_turbo",
            "chatterbox-turbo-v1",
            "chatterbox-turbo-onnx",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (MIT)"
            );
        }
    }
}
