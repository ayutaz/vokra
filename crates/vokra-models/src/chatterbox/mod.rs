//! **Chatterbox-Multilingual** — Resemble AI's 23-language zero-shot TTS
//! (SoTA plan Phase 3, 2026-07-24). MIT.
//!
//! # What Chatterbox-Multilingual is (primary source)
//!
//! `ResembleAI/chatterbox` is Resemble AI's open TTS release. The multilingual
//! variant ships the **T3** stack (Token-To-Token TTS) on a **Llama 520M**
//! backbone with a **HiFT-GAN** (HiFTNet) vocoder — the same terminal-vocoder
//! seam CosyVoice2 / CosyVoice3 use (SoTA plan §1(a) 訂正 2026-07-22, shared
//! `HiFTChain` under `crate::cosyvoice2::hift_chain`). The multilingual variant
//! differs from `chatterbox` (English-only) only in its **text-token vocabulary
//! size** (2454 vs 704) — the LLM backbone shape, speech-token vocabulary, and
//! vocoder are byte-identical between the two.
//!
//! - **License:** MIT (`github.com/resemble-ai/chatterbox/LICENSE`,
//!   fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
//! - **Weights & code:** `ResembleAI/chatterbox`
//!   (`huggingface.co/ResembleAI/chatterbox`); multilingual T3 weights ship
//!   as `t3_mtl23ls_v{2,3}.safetensors` inside the same repo
//!   (`src/chatterbox/mtl_tts.py::MULTILINGUAL_T3_MODELS`).
//! - **Architecture:** T3 = Llama-flavor transformer + learned position
//!   embeddings + text embedding + speech embedding + conditioning encoder;
//!   the acoustic decoder is S3Gen (Token→Wav) whose terminal component is a
//!   HiFT-GAN vocoder — modelled here through the shared
//!   [`HiFTChain`] seam (SoTA plan §1(a) 訂正 seam shared with CosyVoice2 /
//!   CosyVoice3).
//! - **Backbone (primary source: `src/chatterbox/models/t3/llama_configs.py`
//!   :: `LLAMA_520M_CONFIG_DICT`):**
//!   `hidden_size = 1024`, `intermediate_size = 4096`, `num_hidden_layers = 30`,
//!   `num_attention_heads = 16`, `num_key_value_heads = 16` (MHA, **not** GQA),
//!   `head_dim = 64`, `rope_theta = 500000.0`, `rms_norm_eps = 1e-05`,
//!   `hidden_act = "silu"`, `attention_bias = false`, `mlp_bias = false`,
//!   `tie_word_embeddings = false`, `max_position_embeddings = 131072` with
//!   `rope_scaling.rope_type = "llama3"`.
//! - **T3 hparams (primary source: `src/chatterbox/models/t3/modules/t3_config.py`
//!   :: `T3Config`):** `text_tokens_dict_size = 2454` (English-only variant =
//!   704), `speech_tokens_dict_size = 8194`, `max_text_tokens = 2048`,
//!   `max_speech_tokens = 4096`, `start_text_token = 255`,
//!   `stop_text_token = 0`, `start_speech_token = 6561`,
//!   `stop_speech_token = 6562`, `speech_cond_prompt_len = 150`,
//!   `speaker_embed_size = 256`, `use_perceiver_resampler = true`,
//!   `emotion_adv = true`, `encoder_type = "voice_encoder"`.
//! - **Sample rate:** `24_000` Hz — the S3Gen vocoder's fixed output rate
//!   (`src/chatterbox/models/s3gen/const.py::S3GEN_SR = 24000`).
//! - **Multilingual coverage** (`src/chatterbox/mtl_tts.py::SUPPORTED_LANGUAGES`):
//!   23 languages — Arabic, Danish, German, Greek, English, Spanish, Finnish,
//!   French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch,
//!   Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish,
//!   Chinese.
//!
//! # What lands in this Phase 3 slice
//!
//! - [`ChatterboxConfig`] — every architectural hparam **transcribed
//!   verbatim** from the primary source, plus a [`ChatterboxConfig::is_multilingual`]
//!   predicate that pins the 2454-vs-704 axis distinguishing the multilingual
//!   variant from the English-only baseline.
//! - [`ChatterboxWeights`] — deterministic [`ChatterboxWeights::synthesized`]
//!   fixture (SplitMix64 seed + Xavier initialisation) so shape / dtype /
//!   size flow can be exercised without the real HF checkpoint.
//! - [`ChatterboxTts`] — engine handle carrying config + weights + an optional
//!   [`HiFTChain`]. [`ChatterboxTts::synthesize`] returns
//!   [`VokraError::NotImplemented`] until real weights are bound and the
//!   T3 (Llama) ⇒ speech-token sampling ⇒ S3Gen ⇒ HiFTNet chain is wired
//!   end-to-end (T29-equivalent follow-up wave).
//!
//! # No ONNX (permanent)
//!
//! Chatterbox is distributed as safetensors + a Python pipeline; the pipeline
//! is re-implemented natively (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4). The runtime never loads an ONNX graph
//! (FR-LD-05, permanent constraint).

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Public seam re-exports (SoTA plan §1(a) 訂正 shared with CosyVoice2/3)
// ---------------------------------------------------------------------------
//
// Chatterbox's terminal vocoder is HiFT-GAN — the exact same
// `HiFTGenerator` topology CosyVoice2 / CosyVoice3 wire through
// `crate::cosyvoice2::hift_chain::HiFTChain`. Re-export the aliases here so a
// caller wiring Chatterbox sees the seam under its own module path without a
// shape-drift wrapper (mirrors `crate::cosyvoice3`).

pub use crate::cosyvoice2::{HiFTChain, HiFTChainConfig, HiFTChainWeights};

/// `vokra.model.arch` a Chatterbox GGUF must carry. Written by
/// `vokra-convert::models::chatterbox::ARCH`. The compliance registry
/// (`vokra_core::compliance`) knows every canonical / prefix spelling
/// as [`vokra_core::LicenseClass::Permissive`] (MIT — no runtime-side
/// attribution obligation).
pub const EXPECTED_ARCH: &str = "chatterbox";

/// PCM sample rate Chatterbox emits (Hz). Fixed by the S3Gen vocoder's
/// output rate (`src/chatterbox/models/s3gen/const.py::S3GEN_SR = 24000`).
pub const CHATTERBOX_SAMPLE_RATE: u32 = 24_000;

/// Text-token vocabulary size that identifies the **multilingual** T3
/// variant (`src/chatterbox/models/t3/modules/t3_config.py::T3Config.multilingual`).
/// The English-only variant carries `704`.
pub const TEXT_VOCAB_MULTILINGUAL: u32 = 2454;

/// Text-token vocabulary size for the English-only T3 baseline
/// (`src/chatterbox/models/t3/modules/t3_config.py::T3Config.english_only`).
pub const TEXT_VOCAB_ENGLISH: u32 = 704;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Chatterbox T3 architectural hparams.
///
/// Every field is transcribed **verbatim** from the primary source (the
/// module docstring). Numeric axes stay `0` placeholders only on the
/// shape-only conversion path (which the runtime rejects at forward time,
/// FR-EX-08) — the [`ChatterboxConfig::chatterbox_multilingual_v3`] and
/// [`ChatterboxConfig::chatterbox_english`] constructors carry the real
/// upstream values.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatterboxConfig {
    /// Output PCM sample rate, Hz. Fixed at 24 kHz by the S3Gen vocoder.
    pub sample_rate: u32,
    /// T3 text-token vocabulary size. `2454` for the multilingual variant,
    /// `704` for the English-only variant.
    pub text_vocab_size: u32,
    /// T3 speech-token vocabulary size. `8194` for both variants.
    pub speech_vocab_size: u32,
    /// Max text-token positions the T3 backbone can attend over.
    /// `2048` for both variants (`T3Config.max_text_tokens`).
    pub max_text_tokens: u32,
    /// Max speech-token positions the T3 backbone can attend over.
    /// `4096` for both variants (`T3Config.max_speech_tokens`).
    pub max_speech_tokens: u32,
    /// Speaker-embedding dimension (`T3Config.speaker_embed_size`). `256`.
    pub speaker_embed_size: u32,
    /// LLM backbone hidden dimension (`LLAMA_520M_CONFIG_DICT.hidden_size`).
    /// `1024`.
    pub hidden_dim: u32,
    /// LLM backbone transformer block count
    /// (`LLAMA_520M_CONFIG_DICT.num_hidden_layers`). `30`.
    pub n_layer: u32,
    /// LLM backbone attention head count
    /// (`LLAMA_520M_CONFIG_DICT.num_attention_heads`). `16` (MHA — the
    /// Llama_520M config also sets `num_key_value_heads = 16`, so KV-heads
    /// equal query-heads and no GQA broadcast is performed).
    pub n_head: u32,
    /// LLM backbone KV-heads (`LLAMA_520M_CONFIG_DICT.num_key_value_heads`).
    /// `16` — equal to `n_head` (MHA, no GQA).
    pub n_head_kv: u32,
    /// LLM backbone attention head dimension (`LLAMA_520M_CONFIG_DICT.head_dim`).
    /// `64`.
    pub head_dim: u32,
    /// LLM backbone FFN inner dimension
    /// (`LLAMA_520M_CONFIG_DICT.intermediate_size`). `4096`.
    pub ffn_dim: u32,
    /// RoPE base θ (`LLAMA_520M_CONFIG_DICT.rope_theta`). `500_000.0`.
    pub rope_base: f32,
    /// RMSNorm epsilon (`LLAMA_520M_CONFIG_DICT.rms_norm_eps`). `1e-5`.
    pub rms_norm_eps: f32,
}

impl ChatterboxConfig {
    /// Canonical **multilingual** T3 config
    /// (`ResembleAI/chatterbox`, `t3_mtl23ls_v{2,3}.safetensors`).
    ///
    /// Every value is transcribed verbatim from the primary source (see
    /// module docstring).
    #[must_use]
    pub fn chatterbox_multilingual_v3() -> Self {
        Self {
            sample_rate: CHATTERBOX_SAMPLE_RATE,
            text_vocab_size: TEXT_VOCAB_MULTILINGUAL,
            speech_vocab_size: 8194,
            max_text_tokens: 2048,
            max_speech_tokens: 4096,
            speaker_embed_size: 256,
            hidden_dim: 1024,
            n_layer: 30,
            n_head: 16,
            n_head_kv: 16,
            head_dim: 64,
            ffn_dim: 4096,
            rope_base: 500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Canonical **English-only** T3 config
    /// (`ResembleAI/chatterbox`, default checkpoint) — same Llama_520M
    /// backbone, only `text_vocab_size` differs.
    #[must_use]
    pub fn chatterbox_english() -> Self {
        Self {
            text_vocab_size: TEXT_VOCAB_ENGLISH,
            ..Self::chatterbox_multilingual_v3()
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims are
    /// tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (MHA well-formedness, positive FFN dim, non-zero
    /// vocab) mirror the real model. `text_vocab_size` is set to the
    /// multilingual sentinel (2454 is over-large for tests, so we use 32
    /// but expose the flag).
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: CHATTERBOX_SAMPLE_RATE,
            text_vocab_size: 32,
            speech_vocab_size: 64,
            max_text_tokens: 16,
            max_speech_tokens: 16,
            speaker_embed_size: 8,
            hidden_dim: 16,
            n_layer: 2,
            n_head: 2,
            n_head_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            rope_base: 500_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// True iff `text_vocab_size == 2454` — the primary-source flag
    /// distinguishing the multilingual variant from the English-only
    /// baseline (`T3Config.is_multilingual`).
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.text_vocab_size == TEXT_VOCAB_MULTILINGUAL
    }

    /// True iff every architectural axis is at its `0` sentinel — the
    /// shape-only conversion path the runtime tolerates as inspectable-
    /// but-not-forward-ready.
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.text_vocab_size == 0
            && self.speech_vocab_size == 0
            && self.hidden_dim == 0
            && self.n_layer == 0
            && self.n_head == 0
            && self.ffn_dim == 0
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the Llama cross-checks (`hidden_dim == n_head * head_dim`,
    /// `n_head % n_head_kv == 0`) plus positivity on every axis.
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
            || self.n_head_kv == 0
            || self.head_dim == 0
            || self.ffn_dim == 0
            || self.max_text_tokens == 0
            || self.max_speech_tokens == 0
            || self.speaker_embed_size == 0
        {
            return Err(VokraError::InvalidArgument(
                "chatterbox config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use ChatterboxConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        // Llama backbone algebra: hidden_dim = n_head * head_dim (MHA)
        // and n_head must be a whole multiple of n_head_kv (GQA-compatible,
        // even though Llama_520M sets them equal).
        if self.hidden_dim != self.n_head * self.head_dim {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox config: hidden_dim ({}) must equal n_head ({}) * head_dim ({}) — \
                 got {} vs expected {}",
                self.hidden_dim,
                self.n_head,
                self.head_dim,
                self.hidden_dim,
                self.n_head * self.head_dim,
            )));
        }
        if self.n_head % self.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox config: n_head_kv ({}) must divide n_head ({})",
                self.n_head_kv, self.n_head,
            )));
        }
        // RoPE requires even head_dim (pairs).
        if self.head_dim % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox config: RoPE requires even head_dim (got {})",
                self.head_dim,
            )));
        }
        if !(self.rope_base.is_finite() && self.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox config: rope_base must be a positive finite f32 (got {})",
                self.rope_base,
            )));
        }
        if !(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox config: rms_norm_eps must be a positive finite f32 (got {})",
                self.rms_norm_eps,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights (scaffold — real binding delegates to a follow-up wave)
// ---------------------------------------------------------------------------

/// Chatterbox T3 weight store scaffold.
///
/// Carries the text-embedding + speech-embedding + speaker-embedding
/// projection + LLM backbone stack. [`Self::synthesized`] builds a
/// deterministic fixture (SplitMix64 + Xavier) against `config` so
/// shape / dtype / size can be exercised without the real HF checkpoint.
/// Real-checkpoint binding is a follow-up (T29-equivalent, the CosyVoice2
/// / CSM pattern).
#[derive(Debug, Clone)]
pub struct ChatterboxWeights {
    /// Text-token embedding: `[text_vocab_size, hidden_dim]`.
    pub text_embed: Vec<f32>,
    /// Speech-token embedding: `[speech_vocab_size, hidden_dim]`.
    pub speech_embed: Vec<f32>,
    /// Speaker-embedding projection to LLM hidden width:
    /// `[speaker_embed_size, hidden_dim]`. Chatterbox routes the
    /// `voice_encoder` output through a Perceiver resampler + this
    /// projection so the LLM sees a fixed-width prefix (see
    /// `src/chatterbox/models/t3/modules/cond_enc.py`).
    pub speaker_proj: Vec<f32>,
    /// Per-layer transformer block scaffolds. Length = `n_layer`.
    pub blocks: Vec<ChatterboxBlockWeights>,
    /// Final RMSNorm γ, shape `[hidden_dim]`.
    pub final_norm: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

/// Per-transformer-block weights (MHA self-attention + SwiGLU FFN, the
/// Llama block topology).
///
/// The Llama_520M config sets `attention_bias = false` and `mlp_bias =
/// false`, so no bias tensors are carried.
#[derive(Debug, Clone)]
pub struct ChatterboxBlockWeights {
    /// Pre-self-attention RMSNorm γ, shape `[hidden_dim]`.
    pub self_attn_norm: Vec<f32>,
    /// Q projection, shape `[hidden_dim, hidden_dim]` (MHA — n_head == n_head_kv).
    pub q_proj: Vec<f32>,
    /// K projection, shape `[hidden_dim, hidden_dim]` (MHA).
    pub k_proj: Vec<f32>,
    /// V projection, shape `[hidden_dim, hidden_dim]` (MHA).
    pub v_proj: Vec<f32>,
    /// O projection, shape `[hidden_dim, hidden_dim]`.
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

impl ChatterboxWeights {
    /// Builds a deterministic synthesized fixture from `config` and `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every RMSNorm γ starts at `1.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &ChatterboxConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let d = config.hidden_dim as usize;
        let ffn = config.ffn_dim as usize;
        let text_vocab = config.text_vocab_size as usize;
        let speech_vocab = config.speech_vocab_size as usize;
        let spk = config.speaker_embed_size as usize;

        let text_embed = xavier(&mut rng, text_vocab * d, d, d);
        let speech_embed = xavier(&mut rng, speech_vocab * d, d, d);
        let speaker_proj = xavier(&mut rng, spk * d, spk, d);

        let mut blocks = Vec::with_capacity(config.n_layer as usize);
        for _ in 0..config.n_layer {
            blocks.push(ChatterboxBlockWeights {
                self_attn_norm: vec![1.0; d],
                q_proj: xavier(&mut rng, d * d, d, d),
                k_proj: xavier(&mut rng, d * d, d, d),
                v_proj: xavier(&mut rng, d * d, d, d),
                o_proj: xavier(&mut rng, d * d, d, d),
                ffn_norm: vec![1.0; d],
                ffn_gate: xavier(&mut rng, ffn * d, d, ffn),
                ffn_up: xavier(&mut rng, ffn * d, d, ffn),
                ffn_down: xavier(&mut rng, d * ffn, ffn, d),
            });
        }
        let final_norm = vec![1.0; d];

        Ok(Self {
            text_embed,
            speech_embed,
            speaker_proj,
            blocks,
            final_norm,
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

/// Chatterbox-Multilingual TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional [`HiFTChain`]
/// terminal vocoder (SoTA plan §1(a) 訂正 seam shared with CosyVoice2 /
/// CosyVoice3). [`Self::synthesize`] is the primary text → PCM entry point;
/// until real weights are bound and the T3 (Llama) → speech-token sampling →
/// S3Gen → HiFTNet chain is wired end-to-end (T29-equivalent follow-up
/// wave), it returns [`VokraError::NotImplemented`] with a message naming
/// the blocker (FR-EX-08 — never a silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct ChatterboxTts {
    cfg: ChatterboxConfig,
    weights: ChatterboxWeights,
    hift_chain: Option<HiFTChain>,
}

impl ChatterboxTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// weight-store shapes against `cfg` (block count, per-tensor sizes)
    /// so a mismatched pair fails loudly here rather than deep inside a
    /// forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    /// - [`VokraError::InvalidArgument`] naming the first shape mismatch.
    pub fn new(cfg: ChatterboxConfig, weights: ChatterboxWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let d = cfg.hidden_dim as usize;
        let ffn = cfg.ffn_dim as usize;
        let text_vocab = cfg.text_vocab_size as usize;
        let speech_vocab = cfg.speech_vocab_size as usize;
        let spk = cfg.speaker_embed_size as usize;

        check_len("text_embed", weights.text_embed.len(), text_vocab * d)?;
        check_len("speech_embed", weights.speech_embed.len(), speech_vocab * d)?;
        check_len("speaker_proj", weights.speaker_proj.len(), spk * d)?;
        check_len("final_norm", weights.final_norm.len(), d)?;

        if weights.blocks.len() != cfg.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "chatterbox weights: blocks.len()={} != n_layer={}",
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
    /// SoTA plan §1(a) 訂正 seam (shared with CosyVoice2 / CosyVoice3).
    /// Until a caller provides a [`HiFTChain`], [`Self::synthesize`]
    /// returns [`VokraError::NotImplemented`] naming the missing vocoder
    /// as the blocker (FR-EX-08).
    #[must_use]
    pub fn with_hift_chain(mut self, chain: HiFTChain) -> Self {
        self.hift_chain = Some(chain);
        self
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ChatterboxConfig {
        &self.cfg
    }

    /// True iff the weight store was built by [`ChatterboxWeights::synthesized`]
    /// (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// True iff a [`HiFTChain`] has been injected (the SoTA plan §1(a) 訂正
    /// seam is present).
    #[must_use]
    pub fn has_hift_chain(&self) -> bool {
        self.hift_chain.is_some()
    }

    /// True iff the underlying config identifies as the multilingual T3
    /// variant (`text_vocab_size == 2454`).
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.cfg.is_multilingual()
    }

    /// Synthesizes PCM for `text` at [`Self::config`]'s sample rate.
    ///
    /// This is the primary text → PCM entry point. **Real weights required**:
    /// synthesized-weight builds cannot produce meaningful audio, so this
    /// returns [`VokraError::NotImplemented`] naming the blocker. Callers
    /// verify the shape flow through [`ChatterboxTts::new`] +
    /// [`ChatterboxWeights::synthesized`] today; a follow-up wave binds
    /// real Chatterbox weights and wires the forward (T3 (Llama) → speech
    /// token sampling → S3Gen → HiFTNet → PCM).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "chatterbox synthesize: text is empty".to_owned(),
            ));
        }
        if self.hift_chain.is_none() {
            return Err(VokraError::NotImplemented(
                "chatterbox synthesize: no HiFTChain has been injected. Call \
                 `.with_hift_chain(HiFTChain::new(cfg, weights)?)` first — Chatterbox uses \
                 HiFT-GAN as the terminal S3Gen vocoder (SoTA plan §1(a) 訂正, 2026-07-22), \
                 same seam as CosyVoice2 / CosyVoice3. The vocoder module is shared via \
                 `crate::cosyvoice2::hift_chain`.",
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "chatterbox synthesize: this engine holds synthesized weights \
                 (deterministic fixture from ChatterboxWeights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real speech. \
                 Bind real Chatterbox weights (MIT, huggingface.co/ResembleAI/chatterbox) \
                 before invoking synthesize. The shape flow (config validation, weight-store \
                 construction, text-empty check) is exercised through ChatterboxTts::new; \
                 real-checkpoint binding lands in a follow-up wave (T29-equivalent) that \
                 wires the T3 (Llama_520M) → speech-token sampling → S3Gen → HiFTNet chain.",
            ));
        }
        Err(VokraError::NotImplemented(
            "chatterbox synthesize: real weights are bound but the T3 (Llama_520M) → \
             speech-token AR sampling → S3Gen (Token→Wav) → HiFTNet vocoder forward path \
             has not landed yet. Follow-up wave: wire the shared Llama primitives \
             (RoPE θ=500000 / RMSNorm ε=1e-5 / SwiGLU / MHA — n_head == n_head_kv) and \
             feed the sampled speech tokens through the S3Gen chain to the HiFTChain \
             seam. The op set (RoPE / RMSNorm / SwiGLU / GEMM / GEMV / softmax / STFT / \
             iSTFT / snake activation) is already shared with CosyVoice2 / CosyVoice3; no \
             new op or backend kernel is added by Chatterbox.",
        ))
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "chatterbox weights: {name}.len()={got} != {expected}"
        )));
    }
    Ok(())
}

fn check_block_shapes(i: usize, blk: &ChatterboxBlockWeights, d: usize, ffn: usize) -> Result<()> {
    check_len(
        &format!("block[{i}].self_attn_norm"),
        blk.self_attn_norm.len(),
        d,
    )?;
    check_len(&format!("block[{i}].q_proj"), blk.q_proj.len(), d * d)?;
    check_len(&format!("block[{i}].k_proj"), blk.k_proj.len(), d * d)?;
    check_len(&format!("block[{i}].v_proj"), blk.v_proj.len(), d * d)?;
    check_len(&format!("block[{i}].o_proj"), blk.o_proj.len(), d * d)?;
    check_len(&format!("block[{i}].ffn_norm"), blk.ffn_norm.len(), d)?;
    check_len(&format!("block[{i}].ffn_gate"), blk.ffn_gate.len(), ffn * d)?;
    check_len(&format!("block[{i}].ffn_up"), blk.ffn_up.len(), ffn * d)?;
    check_len(&format!("block[{i}].ffn_down"), blk.ffn_down.len(), d * ffn)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_arch_is_chatterbox() {
        assert_eq!(EXPECTED_ARCH, "chatterbox");
    }

    #[test]
    fn sample_rate_matches_s3gen_output() {
        // src/chatterbox/models/s3gen/const.py::S3GEN_SR = 24000
        assert_eq!(CHATTERBOX_SAMPLE_RATE, 24_000);
    }

    /// Multilingual/English text-vocab sentinels are the primary-source
    /// constants — flipping the multilingual sentinel would silently
    /// change what `is_multilingual()` reports for a real checkpoint.
    #[test]
    fn text_vocab_sentinels_match_upstream_t3_config() {
        assert_eq!(TEXT_VOCAB_MULTILINGUAL, 2454);
        assert_eq!(TEXT_VOCAB_ENGLISH, 704);
    }

    /// Every architectural axis carries its primary-source Llama_520M
    /// value verbatim, and the multilingual predicate fires.
    #[test]
    fn multilingual_v3_config_matches_primary_source() {
        let c = ChatterboxConfig::chatterbox_multilingual_v3();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.text_vocab_size, 2454);
        assert_eq!(c.speech_vocab_size, 8194);
        assert_eq!(c.max_text_tokens, 2048);
        assert_eq!(c.max_speech_tokens, 4096);
        assert_eq!(c.speaker_embed_size, 256);
        assert_eq!(c.hidden_dim, 1024);
        assert_eq!(c.n_layer, 30);
        assert_eq!(c.n_head, 16);
        assert_eq!(c.n_head_kv, 16);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.ffn_dim, 4096);
        assert!((c.rope_base - 500_000.0).abs() < 1e-3);
        assert!((c.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert!(c.is_multilingual());
        assert!(!c.is_placeholder_shape());
        c.validate_for_forward()
            .expect("real config is well-formed");
        // hidden_dim = n_head * head_dim (MHA)
        assert_eq!(c.hidden_dim, c.n_head * c.head_dim);
    }

    /// English variant differs only in text-vocab size — a regression that
    /// silently kept the multilingual sentinel would break the honest
    /// multilingual/English fork.
    #[test]
    fn english_config_matches_primary_source() {
        let c = ChatterboxConfig::chatterbox_english();
        assert_eq!(c.text_vocab_size, 704);
        assert!(!c.is_multilingual());
        // Every other axis matches the multilingual config.
        let m = ChatterboxConfig::chatterbox_multilingual_v3();
        assert_eq!(c.speech_vocab_size, m.speech_vocab_size);
        assert_eq!(c.hidden_dim, m.hidden_dim);
        assert_eq!(c.n_layer, m.n_layer);
        assert_eq!(c.n_head, m.n_head);
        assert_eq!(c.head_dim, m.head_dim);
        assert_eq!(c.ffn_dim, m.ffn_dim);
        c.validate_for_forward()
            .expect("english config is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed_and_multilingual_flag_defaults_false() {
        let c = ChatterboxConfig::tiny_for_tests();
        c.validate_for_forward().expect("tiny config well-formed");
        // Tiny fixture uses vocab=32, not 2454 — so is_multilingual() is
        // false. Real callers of `is_multilingual()` must not rely on the
        // tiny fixture flipping the flag.
        assert!(!c.is_multilingual());
        assert_eq!(c.hidden_dim, c.n_head * c.head_dim);
    }

    #[test]
    fn placeholder_config_is_placeholder_shape() {
        let c = ChatterboxConfig {
            sample_rate: CHATTERBOX_SAMPLE_RATE,
            text_vocab_size: 0,
            speech_vocab_size: 0,
            max_text_tokens: 0,
            max_speech_tokens: 0,
            speaker_embed_size: 0,
            hidden_dim: 0,
            n_layer: 0,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            ffn_dim: 0,
            rope_base: 500_000.0,
            rms_norm_eps: 1e-5,
        };
        assert!(c.is_placeholder_shape());
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_zero_axis() {
        // Each zeroing mutator must trip validate_for_forward.
        let mutators: &[fn(&mut ChatterboxConfig)] = &[
            |c| c.text_vocab_size = 0,
            |c| c.speech_vocab_size = 0,
            |c| c.hidden_dim = 0,
            |c| c.n_layer = 0,
            |c| c.n_head = 0,
            |c| c.n_head_kv = 0,
            |c| c.head_dim = 0,
            |c| c.ffn_dim = 0,
            |c| c.sample_rate = 0,
            |c| c.max_text_tokens = 0,
            |c| c.max_speech_tokens = 0,
            |c| c.speaker_embed_size = 0,
        ];
        for mutate in mutators {
            let mut c = ChatterboxConfig::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_hidden_dim_not_matching_head_split() {
        let mut c = ChatterboxConfig::tiny_for_tests();
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
    fn config_rejects_kv_head_not_dividing_query_head() {
        let mut c = ChatterboxConfig::tiny_for_tests();
        c.n_head_kv = 3; // 2 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_odd_head_dim_rope() {
        let mut c = ChatterboxConfig::tiny_for_tests();
        c.head_dim = 7;
        c.hidden_dim = c.n_head * c.head_dim; // keep the MHA algebra
        let err = c.validate_for_forward().expect_err("odd head_dim fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("RoPE"), "message names RoPE: {msg}");
                assert!(msg.contains("head_dim"), "message: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_nonfinite_rope_base_or_norm_eps() {
        let mut c = ChatterboxConfig::tiny_for_tests();
        c.rope_base = 0.0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = ChatterboxConfig::tiny_for_tests();
        c.rope_base = f32::NAN;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = ChatterboxConfig::tiny_for_tests();
        c.rms_norm_eps = -1.0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = ChatterboxConfig::tiny_for_tests();
        let w1 = ChatterboxWeights::synthesized(&c, 0x42).expect("build 1");
        let w2 = ChatterboxWeights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.text_embed, w2.text_embed);
        assert_eq!(w1.speech_embed, w2.speech_embed);
        assert_eq!(w1.speaker_proj, w2.speaker_proj);
        assert_eq!(w1.blocks[0].q_proj, w2.blocks[0].q_proj);
        assert_eq!(w1.blocks[1].ffn_gate, w2.blocks[1].ffn_gate);
        assert!(w1.is_synthesized);

        // Shape flow.
        let d = c.hidden_dim as usize;
        let ffn = c.ffn_dim as usize;
        let text_vocab = c.text_vocab_size as usize;
        let speech_vocab = c.speech_vocab_size as usize;
        let spk = c.speaker_embed_size as usize;
        assert_eq!(w1.text_embed.len(), text_vocab * d);
        assert_eq!(w1.speech_embed.len(), speech_vocab * d);
        assert_eq!(w1.speaker_proj.len(), spk * d);
        assert_eq!(w1.final_norm.len(), d);
        assert_eq!(w1.blocks.len(), c.n_layer as usize);
        for blk in &w1.blocks {
            assert_eq!(blk.self_attn_norm.len(), d);
            assert_eq!(blk.q_proj.len(), d * d);
            assert_eq!(blk.k_proj.len(), d * d);
            assert_eq!(blk.v_proj.len(), d * d);
            assert_eq!(blk.o_proj.len(), d * d);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.ffn_gate.len(), ffn * d);
            assert_eq!(blk.ffn_up.len(), ffn * d);
            assert_eq!(blk.ffn_down.len(), d * ffn);
        }
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = ChatterboxConfig::tiny_for_tests();
        let a = ChatterboxWeights::synthesized(&c, 1).expect("a");
        let b = ChatterboxWeights::synthesized(&c, 2).expect("b");
        assert_ne!(a.text_embed, b.text_embed);
        assert_ne!(a.speech_embed, b.speech_embed);
        assert_ne!(a.blocks[0].q_proj, b.blocks[0].q_proj);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = ChatterboxConfig::tiny_for_tests();
        c.hidden_dim = 0;
        assert!(matches!(
            ChatterboxWeights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_accepts_matching_config_and_weights() {
        let c = ChatterboxConfig::tiny_for_tests();
        let w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTts::new(c.clone(), w).expect("chatterbox tts");
        assert_eq!(tts.config().hidden_dim, c.hidden_dim);
        assert_eq!(tts.config().n_layer, c.n_layer);
        assert_eq!(tts.config().sample_rate, 24_000);
        assert!(tts.is_synthesized());
        assert!(!tts.has_hift_chain(), "fresh load has no HiFTChain");
        assert!(
            !tts.is_multilingual(),
            "tiny fixture is not the 2454-vocab variant"
        );
    }

    #[test]
    fn tts_new_rejects_block_count_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_text_embed_shape_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.text_embed.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_speech_embed_shape_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.speech_embed.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_speaker_proj_shape_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.speaker_proj.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_block_qkv_size_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.blocks[0].q_proj.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_ffn_size_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.blocks[1].ffn_down.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_final_norm_size_mismatch() {
        let c = ChatterboxConfig::tiny_for_tests();
        let mut w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        w.final_norm.pop();
        assert!(matches!(
            ChatterboxTts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_text() {
        let c = ChatterboxConfig::tiny_for_tests();
        let w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTts::new(c, w).expect("chatterbox tts");
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
        let c = ChatterboxConfig::tiny_for_tests();
        let w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTts::new(c, w).expect("chatterbox tts");
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

    /// With a HiFTChain injected but synthesized weights, the message pivots
    /// to the synthesized-weight blocker.
    #[test]
    fn synthesize_with_chain_and_synthesized_weights_names_the_synthesized_blocker() {
        use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

        // Build a tiny well-formed HiFTChain (shape lifted from
        // `cosyvoice2::small_hift_chain_for_wiring` — identical pattern).
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

        let c = ChatterboxConfig::tiny_for_tests();
        let w = ChatterboxWeights::synthesized(&c, 7).expect("weights");
        let tts = ChatterboxTts::new(c, w)
            .expect("chatterbox tts")
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

    /// The M2-13 compliance registry must resolve every canonical Chatterbox
    /// id to Permissive (MIT). Cross-crate test to keep this module's
    /// registry-side contract honest.
    #[test]
    fn registry_lookup_maps_chatterbox_to_permissive_mit() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "chatterbox",
            "chatterbox-multilingual",
            "chatterbox-multilingual-v3",
            "chatterbox-multilingual-v2",
            "chatterbox-mtl23ls-v3",
            "chatterbox-english",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (MIT)"
            );
        }
    }
}
