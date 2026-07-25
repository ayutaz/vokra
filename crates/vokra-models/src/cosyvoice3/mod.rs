//! **Fun-CosyVoice3-0.5B** — Qwen2 LLM backbone + chunk-aware Flow Matching
//! CFM + **HiFTNet** vocoder (SoTA plan Phase 3, 2026-07-24).
//!
//! # What Fun-CosyVoice3-0.5B is (primary source)
//!
//! `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (HuggingFace / ModelScope) is the
//! third-generation FunAudio TTS model. The model card describes it as
//! *"an advanced text-to-speech (TTS) system based on large language models
//! (LLM), surpassing its predecessor (CosyVoice 2.0) in content consistency,
//! speaker similarity, and prosody naturalness"* — the topology is the same
//! CosyVoice2 chain (**FSQ tokens → Qwen2 AR decoder → chunk-aware CFM →
//! mel → HiFTNet → PCM**, arXiv:2505.17589 + `cosyvoice/hifigan/generator.py`
//! `HiFTGenerator`), with quality-driving refinements that leave the op
//! inventory **byte-identical** to CosyVoice2:
//!
//! - **Dual-Resolution Speech Representations (DRSR)** — a training-side
//!   representation scheme that lifts speaker similarity and prosody
//!   without changing the runtime forward operators.
//! - **Core-Cocktail Training** — a data-mixture strategy, again
//!   training-side.
//!
//! Neither addition requires a new runtime op or backend kernel: DRSR
//! reshapes what the LLM head + Flow Matching CFM consumes at train time
//! (the network topology is unchanged); Core-Cocktail is a data recipe
//! (the runtime never sees it). See the "Very-cheap follow-on" section
//! below.
//!
//! # Primary source
//!
//! - **License:** `apache-2.0`
//!   (`huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` model-card
//!   header, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
//! - **Weights & code:** `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
//!   (`huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` +
//!   `modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512`,
//!   9.75 GB total incl. `flow.decoder.estimator.fp32.onnx`,
//!   `speech_tokenizer_*`, `llm.pt`, `flow.pt`, `hift.pt`).
//! - **Architecture:** Qwen2 LLM backbone (same family as CosyVoice2's
//!   Qwen2.5-0.5B) — the model card explicitly names Qwen2 as the LLM
//!   backbone. Vocoder = **HiFTNet** (`cosyvoice/hifigan/generator.py`
//!   `HiFTGenerator`) — the SoTA-plan §1(a) 訂正 (2026-07-22) fixed the
//!   wrong-premise "Mimi" wiring that would have applied here otherwise;
//!   `crate::cosyvoice2::hift_chain::HiFTChain` is the shared seam and
//!   this module re-exports its aliases directly.
//! - **Paper:** arXiv:2505.17589 (Fun-CosyVoice technical report).
//!
//! # Numeric hparams — deferred to real-checkpoint bind
//!
//! The upstream `config.json` (fetched by URL) was not accessible via
//! anonymous WebFetch in the Phase 3 timebox (empty JSON body returned).
//! Rather than fabricate `hidden_size` / `num_hidden_layers` /
//! `num_attention_heads` etc. — which would violate the CLAUDE.md
//! hallucination ban and silently mis-shape the LLM backbone — the
//! [`CosyVoice3Config`] surface is:
//!
//! - **Auto-detected** from the checkpoint tensor shapes at
//!   convert-time (the CosyVoice2 shape-derivation path applies verbatim:
//!   `llm.model.model.embed_tokens.weight` → `vocab_size` / `hidden_dim`;
//!   `llm.model.model.layers.*` contiguous count → `n_layer`;
//!   layer-0 `mlp.gate_proj.weight` → `ffn_dim`; layer-0
//!   `self_attn.q_proj.weight` / `k_proj.weight` → GQA algebra
//!   cross-checks). The GQA head split (`num_attention_heads` /
//!   `num_key_value_heads`) plus `rope_theta` / `rms_norm_eps` /
//!   `max_position_embeddings` are **not** shape-derivable — a
//!   `--config <config.json>` side-car supplies them and is
//!   cross-checked against the tensor shapes, exactly like CosyVoice2.
//! - **Loud fail on absence** — a caller who binds an engine before the
//!   config lands hits [`VokraError::InvalidArgument`] naming the
//!   missing key, not a silent zero-shape forward (FR-EX-08).
//! - **Sample rate** — `24_000` (the CosyVoice family sample rate; the
//!   HiFTNet vocoder produces 24 kHz PCM, matching CosyVoice2).
//!
//! When the real checkpoint lands (owner T29-equivalent hand-off), the
//! [`CosyVoice3Config::fun_cosyvoice3_0_5b`] constructor can be filled
//! in with the transcribed values without changing any downstream code
//! (the [`CosyVoice3Tts`] engine and the converter both read the
//! shape-derived path today).
//!
//! # Very-cheap follow-on — reuses CosyVoice2 verbatim
//!
//! Because the topology is a CosyVoice2 chain with training-side
//! refinements (DRSR + Core-Cocktail), Fun-CosyVoice3-0.5B **does not
//! add any new op** (`vokra-ops`) or backend kernel. The forward path
//! is:
//!
//! - Text tokenizer — Qwen2 byte-level BPE
//!   ([`crate::cosyvoice2::text_encoder::CosyVoice2Tokenizer`]).
//! - LLM backbone — Qwen2 decoder-only transformer
//!   ([`crate::cosyvoice2::llm::LlmBackbone`], GQA / RoPE / SwiGLU /
//!   RMSNorm).
//! - Flow Matching CFM — chunk-aware
//!   ([`crate::cosyvoice2::flow_matching::ChunkAwareCfm`]).
//! - Terminal vocoder — HiFTNet
//!   ([`crate::cosyvoice2::hift_chain::HiFTChain`]).
//!
//! This module re-exports the CosyVoice2 [`HiFTChain`] / config / weights
//! aliases so the runtime seam stays identical: an operator supplying
//! HiFTNet weights for a CosyVoice3 checkpoint uses the same
//! `.with_hift_chain(HiFTChain::new(cfg, weights)?)` pattern.
//!
//! # What lands in this Phase 3 slice
//!
//! - [`CosyVoice3Config`] — shape-derived hparams surface with a
//!   `distil-whisper`-style `validate_for_forward` gate on `0`
//!   placeholders (LLM axes come from the GGUF; a shape-only conversion
//!   makes the engine's LLM handle honestly `None`, per the CosyVoice2
//!   contract).
//! - [`CosyVoice3Weights`] — deterministic
//!   [`CosyVoice3Weights::synthesized`] fixture (SplitMix64 seed) so
//!   shape / dtype / size flow can be exercised without the real HF
//!   checkpoint.
//! - [`CosyVoice3Tts`] — engine handle carrying config + weights + an
//!   optional [`HiFTChain`]. [`CosyVoice3Tts::synthesize`] returns
//!   [`VokraError::NotImplemented`] until real weights are bound and
//!   the LLM ⇒ CFM ⇒ HiFTNet chain is wired (T29-equivalent follow-up
//!   wave that delegates to [`crate::cosyvoice2`]).
//!
//! # No ONNX (permanent)
//!
//! Fun-CosyVoice3-0.5B ships `flow.decoder.estimator.fp32.onnx` alongside
//! PyTorch pickles. The runtime never loads the ONNX graph
//! (FR-LD-05, permanent constraint) — the pipeline is re-implemented
//! natively via [`crate::cosyvoice2`] (whisper.cpp 型, CLAUDE.md 設計判断
//! 4). The Flow Matching estimator is bound off the PyTorch checkpoint
//! (`flow.pt`) at convert-time, not from the ONNX file.

use vokra_core::rng::SplitMix64;
use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Public seam re-exports (SoTA plan §1(a) 訂正 shared with CosyVoice2)
// ---------------------------------------------------------------------------
//
// The HiFTNet vocoder chain is architecturally identical between CosyVoice2
// and CosyVoice3 (same `HiFTGenerator` in `cosyvoice/hifigan/generator.py`,
// same NSF + ISTFTNet composition). Re-export the CosyVoice2 aliases here
// so a caller wiring CosyVoice3 sees the seam under its own module path
// without a shape-drift wrapper (the same pattern
// `cosyvoice2::hift_chain::HiFTChainConfig` uses over
// `vokra_ops::hiftnet::HiFTGeneratorConfig`).

pub use crate::cosyvoice2::{HiFTChain, HiFTChainConfig, HiFTChainWeights};

/// `vokra.model.arch` a Fun-CosyVoice3 GGUF must carry. Written by
/// `vokra-convert::models::cosyvoice3::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `cosyvoice3` /
/// `fun-cosyvoice3-0.5b-2512` (and every family variant that lands later)
/// as [`vokra_core::LicenseClass::Permissive`] via the `cosyvoice-` /
/// `cosyvoice3-` family prefix walks (apache-2.0 — the M2-13 gate passes
/// commercially without any attribution obligation on the runtime side).
///
/// This arch string is intentionally **distinct** from CosyVoice2's
/// (`"cosyvoice2"`) so the runtime can label the loaded model correctly
/// in telemetry / logs / model cards while still delegating the numeric
/// forward through [`crate::cosyvoice2`] (the "very cheap follow-on"
/// contract in the task).
pub const EXPECTED_ARCH: &str = "cosyvoice3";

/// PCM sample rate Fun-CosyVoice3 emits (Hz). Same as CosyVoice2 (the
/// HiFTNet vocoder produces 24 kHz PCM by architecture).
pub const COSYVOICE3_SAMPLE_RATE: u32 = 24_000;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Fun-CosyVoice3-0.5B architectural hyperparameters.
///
/// A deliberate subset of the CosyVoice2 hparam schema — every field maps
/// 1-to-1 to the corresponding CosyVoice2 axis (see
/// [`crate::cosyvoice2::CosyVoice2Config`]). Numeric axes stay `0`
/// placeholders until a real GGUF is converted with a `--config`
/// side-car; the fixed axes (`sample_rate`, `flow_schedule_tag`,
/// canonical Mimi shape retained for compliance-gate compatibility with
/// pre-migration test GGUFs — see CosyVoice2 T13 codec-migration note)
/// carry their model-card invariants.
///
/// The `0`-placeholder posture is deliberate: without primary-source
/// hparams (see the module docstring), inventing them would silently
/// mis-shape the LLM backbone. A shape-only GGUF loads (the container
/// is inspectable), but the engine's LLM handle is honestly `None` and
/// [`CosyVoice3Tts::synthesize`] fails loudly naming re-conversion as
/// the fix (FR-EX-08).
#[derive(Debug, Clone, PartialEq)]
pub struct CosyVoice3Config {
    /// Output PCM sample rate, Hz. Fixed at 24 kHz by the HiFTNet
    /// vocoder (identical to CosyVoice2).
    pub sample_rate: u32,
    /// Text tokenizer vocabulary size. `0` = shape-only conversion
    /// (runtime rejects LLM bind).
    pub vocab_size: u32,
    /// LLM backbone hidden dimension. `0` = shape-only.
    pub hidden_dim: u32,
    /// LLM backbone transformer block count. `0` = shape-only.
    pub n_layer: u32,
    /// LLM backbone attention head count. `0` = shape-only.
    pub n_head: u32,
    /// LLM backbone FFN inner dimension. `0` = shape-only.
    pub ffn_dim: u32,
    /// Flow Matching sampler default NFE (number of function
    /// evaluations per chunk). `0` = runtime-overridable per invocation
    /// (FR-EX-10) with no baked default.
    pub flow_nfe: u32,
    /// Flow Matching schedule tag (`"linear"` / `"sway"` / `"epss"`).
    /// Same variant set as `vokra_ops::Schedule`.
    pub flow_schedule_tag: String,
    /// Chunk-aware streaming chunk size (frames per chunk boundary).
    /// `0` = not yet transcribed.
    pub streaming_chunk_size: u32,
    /// Chunk-aware streaming chunk hop (frames between chunk starts).
    /// `0` = not yet transcribed.
    pub streaming_chunk_hop: u32,
}

impl CosyVoice3Config {
    /// Placeholder-only config for the canonical
    /// `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` release. Numeric axes are
    /// `0` sentinels (see the module docstring on why the primary-source
    /// hparams are deferred — the config.json was not accessible via
    /// anonymous WebFetch in the Phase 3 timebox). Sample rate and
    /// schedule tag carry their model-card / op-crate invariants.
    ///
    /// The T29-equivalent follow-up wave fills in the numeric axes from
    /// the real config.json + tensor shape verification (the CosyVoice2
    /// path handles this via `--config` cross-check).
    #[must_use]
    pub fn fun_cosyvoice3_0_5b_placeholder() -> Self {
        Self {
            sample_rate: COSYVOICE3_SAMPLE_RATE,
            vocab_size: 0,
            hidden_dim: 0,
            n_layer: 0,
            n_head: 0,
            ffn_dim: 0,
            flow_nfe: 0,
            flow_schedule_tag: "linear".to_owned(),
            streaming_chunk_size: 0,
            streaming_chunk_hop: 0,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (GQA well-formedness, positive FFN dim, non-zero
    /// vocab) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            sample_rate: COSYVOICE3_SAMPLE_RATE,
            vocab_size: 32,
            hidden_dim: 16,
            n_layer: 2,
            n_head: 2,
            ffn_dim: 32,
            flow_nfe: 2,
            flow_schedule_tag: "linear".to_owned(),
            streaming_chunk_size: 4,
            streaming_chunk_hop: 4,
        }
    }

    /// True iff every numeric axis is at its `0` sentinel — the shape-
    /// only conversion path the runtime tolerates as inspectable-but-
    /// not-forward-ready (mirrors CosyVoice2's placeholder-shape
    /// tolerance in `CosyVoice2Tts::from_gguf_with_policy`).
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.vocab_size == 0
            && self.hidden_dim == 0
            && self.n_layer == 0
            && self.n_head == 0
            && self.ffn_dim == 0
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the Qwen2 cross-checks (`hidden_dim % n_head == 0`,
    /// non-zero axes) plus the flow / streaming positivity constraints.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.sample_rate == 0
            || self.vocab_size == 0
            || self.hidden_dim == 0
            || self.n_layer == 0
            || self.n_head == 0
            || self.ffn_dim == 0
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use CosyVoice3Config::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        if self.hidden_dim % self.n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice3 config: n_head ({}) must divide hidden_dim ({})",
                self.n_head, self.hidden_dim,
            )));
        }
        // Streaming positivity — if either boundary is set, both must be
        // set (a chunk hop with no size, or vice versa, is a broken
        // configuration). Both-zero (chunk-aware disabled) and
        // both-positive (chunk-aware enabled) are the two legal states.
        let sz = self.streaming_chunk_size;
        let hop = self.streaming_chunk_hop;
        let both_zero = sz == 0 && hop == 0;
        let both_positive = sz > 0 && hop > 0;
        if !(both_zero || both_positive) {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice3 config: streaming_chunk_size ({sz}) and streaming_chunk_hop \
                 ({hop}) must both be zero or both positive",
            )));
        }
        // Schedule tag must be one of the vokra_ops::Schedule variants.
        // Reject the empty string and any unrecognised tag loudly rather
        // than silently defaulting to "linear".
        match self.flow_schedule_tag.as_str() {
            "linear" | "sway" | "epss" => {}
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice3 config: flow_schedule_tag = {other:?} — expected one of \
                     `linear` / `sway` / `epss`",
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weights (scaffold — real binding delegates to CosyVoice2)
// ---------------------------------------------------------------------------

/// Fun-CosyVoice3-0.5B weight store scaffold.
///
/// Carries the token embedding + LLM backbone stack (per-layer
/// projection tensors — the same layout CosyVoice2's
/// [`crate::cosyvoice2::llm::LlmWeights`] consumes) so shape-driven
/// tests can exercise the loader boundary without inventing upstream
/// tensor names.
///
/// [`Self::synthesized`] builds a deterministic fixture (SplitMix64 +
/// Xavier) against `config` so shape / dtype / size can be exercised
/// without the real HF checkpoint. Real-checkpoint binding is a
/// follow-up (T29-equivalent — the CosyVoice2 pattern) that delegates
/// the loader to
/// [`crate::cosyvoice2::llm::LlmBackbone::from_gguf`].
#[derive(Debug, Clone)]
pub struct CosyVoice3Weights {
    /// Token embedding: `[vocab_size, hidden_dim]`.
    pub token_embed: Vec<f32>,
    /// Per-layer transformer block scaffolds. Length = `n_layer`.
    pub blocks: Vec<CosyVoice3BlockWeights>,
    /// Final RMSNorm γ, shape `[hidden_dim]`.
    pub final_norm: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

/// Per-transformer-block weights (GQA self-attention + SwiGLU FFN, the
/// Qwen2 block topology). Same convention as CosyVoice2's LLM path.
#[derive(Debug, Clone)]
pub struct CosyVoice3BlockWeights {
    /// Self-attention pre-norm γ, shape `[hidden_dim]`.
    pub self_attn_norm: Vec<f32>,
    /// Q projection, shape `[hidden_dim, hidden_dim]`.
    pub q_proj: Vec<f32>,
    /// Q bias, shape `[hidden_dim]` (Qwen2 has attention biases).
    pub q_bias: Vec<f32>,
    /// K projection, shape `[kv_out, hidden_dim]` where `kv_out =
    /// n_head_kv * head_dim`. Stored as `[hidden_dim, hidden_dim]` for
    /// the scaffold — real GQA-aware binding lands in the follow-up
    /// wave.
    pub k_proj: Vec<f32>,
    /// K bias, shape `[hidden_dim]` (scaffold).
    pub k_bias: Vec<f32>,
    /// V projection, shape `[kv_out, hidden_dim]` (scaffold same as K).
    pub v_proj: Vec<f32>,
    /// V bias, shape `[hidden_dim]` (scaffold).
    pub v_bias: Vec<f32>,
    /// O projection, shape `[hidden_dim, hidden_dim]` (no bias in
    /// Qwen2's output projection).
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

impl CosyVoice3Weights {
    /// Builds a deterministic synthesized fixture from `config` and
    /// `seed`.
    ///
    /// Draws are Xavier-uniform ± `sqrt(6 / (fan_in + fan_out))` via a
    /// [`SplitMix64`] stream — reproducible, allocation-only, zero-dep.
    /// Every RMSNorm γ starts at `1.0`; every bias starts at `0.0`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &CosyVoice3Config, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let mut rng = SplitMix64::new(seed);
        let d = config.hidden_dim as usize;
        let ffn = config.ffn_dim as usize;
        let vocab = config.vocab_size as usize;

        let token_embed = xavier(&mut rng, vocab * d, d, d);

        let mut blocks = Vec::with_capacity(config.n_layer as usize);
        for _ in 0..config.n_layer {
            blocks.push(CosyVoice3BlockWeights {
                self_attn_norm: vec![1.0; d],
                q_proj: xavier(&mut rng, d * d, d, d),
                q_bias: vec![0.0; d],
                k_proj: xavier(&mut rng, d * d, d, d),
                k_bias: vec![0.0; d],
                v_proj: xavier(&mut rng, d * d, d, d),
                v_bias: vec![0.0; d],
                o_proj: xavier(&mut rng, d * d, d, d),
                ffn_norm: vec![1.0; d],
                ffn_gate: xavier(&mut rng, ffn * d, d, ffn),
                ffn_up: xavier(&mut rng, ffn * d, d, ffn),
                ffn_down: xavier(&mut rng, d * ffn, ffn, d),
            });
        }
        let final_norm = vec![1.0; d];

        Ok(Self {
            token_embed,
            blocks,
            final_norm,
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

/// Fun-CosyVoice3-0.5B TTS engine handle.
///
/// Carries the resolved config, weight store, and an optional
/// [`HiFTChain`] terminal vocoder (SoTA plan §1(a) 訂正 seam shared
/// with CosyVoice2). [`Self::synthesize`] is the primary text → PCM
/// entry point; until real weights are bound and the LLM ⇒ CFM ⇒
/// HiFTNet chain is wired end-to-end (T29-equivalent follow-up wave
/// that delegates to [`crate::cosyvoice2`]), it returns
/// [`VokraError::NotImplemented`] with a message naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct CosyVoice3Tts {
    cfg: CosyVoice3Config,
    weights: CosyVoice3Weights,
    hift_chain: Option<HiFTChain>,
}

impl CosyVoice3Tts {
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
    pub fn new(cfg: CosyVoice3Config, weights: CosyVoice3Weights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let d = cfg.hidden_dim as usize;
        let ffn = cfg.ffn_dim as usize;
        let vocab = cfg.vocab_size as usize;

        check_len("token_embed", weights.token_embed.len(), vocab * d)?;
        check_len("final_norm", weights.final_norm.len(), d)?;

        if weights.blocks.len() != cfg.n_layer as usize {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice3 weights: blocks.len()={} != n_layer={}",
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
    /// SoTA plan §1(a) 訂正 seam (shared with CosyVoice2). Until a
    /// caller provides a [`HiFTChain`], [`Self::synthesize`] returns
    /// [`VokraError::NotImplemented`] naming the missing vocoder as
    /// the blocker (FR-EX-08).
    #[must_use]
    pub fn with_hift_chain(mut self, chain: HiFTChain) -> Self {
        self.hift_chain = Some(chain);
        self
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &CosyVoice3Config {
        &self.cfg
    }

    /// True iff the weight store was built by
    /// [`CosyVoice3Weights::synthesized`] (never a real upstream
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

    /// Synthesizes PCM for `text` at [`Self::config`]'s sample rate.
    ///
    /// This is the primary text → PCM entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// audio, so this returns [`VokraError::NotImplemented`] naming
    /// the blocker. Callers verify the shape flow through
    /// [`CosyVoice3Tts::new`] + [`CosyVoice3Weights::synthesized`]
    /// today; a follow-up wave binds real Fun-CosyVoice3 weights and
    /// wires the forward through [`crate::cosyvoice2`] with the
    /// Fun-CosyVoice3 config surface.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not
    ///   yet bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 synthesize: text is empty".to_owned(),
            ));
        }
        if self.hift_chain.is_none() {
            return Err(VokraError::NotImplemented(
                "cosyvoice3 synthesize: no HiFTChain has been injected. Call \
                 `.with_hift_chain(HiFTChain::new(cfg, weights)?)` first — Fun-CosyVoice3 \
                 uses HiFTNet (Neural Source Filter + ISTFTNet) as the terminal mel → \
                 PCM vocoder (SoTA plan §1(a) 訂正, 2026-07-22), same as CosyVoice2. \
                 The vocoder module is shared via `crate::cosyvoice2::hift_chain`.",
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "cosyvoice3 synthesize: this engine holds synthesized weights \
                 (deterministic fixture from CosyVoice3Weights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real \
                 speech. Bind real Fun-CosyVoice3 weights (apache-2.0, \
                 huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) before invoking \
                 synthesize. The shape flow (config validation, weight-store \
                 construction, text-empty check) is exercised through \
                 CosyVoice3Tts::new; real-checkpoint binding lands in a follow-up wave \
                 (T29-equivalent) that delegates the forward to the CosyVoice2 chain \
                 (Qwen2 LLM → chunk-aware CFM → HiFTNet).",
            ));
        }
        Err(VokraError::NotImplemented(
            "cosyvoice3 synthesize: real weights are bound but the Qwen2 LLM backbone → \
             chunk-aware Flow Matching CFM → HiFTNet vocoder forward path has not \
             landed yet. Follow-up wave: delegate to crate::cosyvoice2 with the \
             Fun-CosyVoice3 config surface — the op set (RoPE / RMSNorm / SwiGLU / \
             GEMM / GEMV / softmax / STFT / iSTFT / snake activation) and every \
             kernel are already shared with CosyVoice2 (arXiv:2505.17589 confirms the \
             topology is identical; DRSR + Core-Cocktail are training-side additions \
             that leave the runtime forward operators byte-identical).",
        ))
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice3 weights: {name}.len()={got} != {expected}"
        )));
    }
    Ok(())
}

fn check_block_shapes(i: usize, blk: &CosyVoice3BlockWeights, d: usize, ffn: usize) -> Result<()> {
    check_len(
        &format!("block[{i}].self_attn_norm"),
        blk.self_attn_norm.len(),
        d,
    )?;
    check_len(&format!("block[{i}].q_proj"), blk.q_proj.len(), d * d)?;
    check_len(&format!("block[{i}].q_bias"), blk.q_bias.len(), d)?;
    check_len(&format!("block[{i}].k_proj"), blk.k_proj.len(), d * d)?;
    check_len(&format!("block[{i}].k_bias"), blk.k_bias.len(), d)?;
    check_len(&format!("block[{i}].v_proj"), blk.v_proj.len(), d * d)?;
    check_len(&format!("block[{i}].v_bias"), blk.v_bias.len(), d)?;
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
    fn expected_arch_is_cosyvoice3() {
        assert_eq!(EXPECTED_ARCH, "cosyvoice3");
    }

    #[test]
    fn sample_rate_matches_cosyvoice_family() {
        // HiFTNet produces 24 kHz PCM by architecture — same as CosyVoice2.
        assert_eq!(COSYVOICE3_SAMPLE_RATE, 24_000);
    }

    /// The primary-source placeholder config carries the model-card
    /// invariants (sample rate + schedule tag) and `0` sentinels for
    /// the numeric axes that were not yet transcribed (see the module
    /// docstring).
    #[test]
    fn fun_cosyvoice3_placeholder_carries_invariants_only() {
        let c = CosyVoice3Config::fun_cosyvoice3_0_5b_placeholder();
        assert_eq!(c.sample_rate, 24_000);
        assert_eq!(c.flow_schedule_tag, "linear");
        // Every numeric axis is at its 0 sentinel — a shape-only conversion.
        assert!(c.is_placeholder_shape());
        assert_eq!(c.vocab_size, 0);
        assert_eq!(c.hidden_dim, 0);
        assert_eq!(c.n_layer, 0);
        assert_eq!(c.n_head, 0);
        assert_eq!(c.ffn_dim, 0);
        assert_eq!(c.flow_nfe, 0);
        assert_eq!(c.streaming_chunk_size, 0);
        assert_eq!(c.streaming_chunk_hop, 0);
        // Validation must refuse the placeholder — a shape-only config
        // is inspectable but not forward-ready (FR-EX-08).
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tiny_config_is_well_formed() {
        CosyVoice3Config::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    #[test]
    fn config_rejects_zero_axis() {
        for mutate in [
            |c: &mut CosyVoice3Config| c.vocab_size = 0,
            |c: &mut CosyVoice3Config| c.hidden_dim = 0,
            |c: &mut CosyVoice3Config| c.n_layer = 0,
            |c: &mut CosyVoice3Config| c.n_head = 0,
            |c: &mut CosyVoice3Config| c.ffn_dim = 0,
            |c: &mut CosyVoice3Config| c.sample_rate = 0,
        ] {
            let mut c = CosyVoice3Config::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_head_not_dividing_hidden_dim() {
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.n_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_unknown_schedule_tag() {
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.flow_schedule_tag = "bogus".to_owned();
        let err = c.validate_for_forward().expect_err("bogus schedule fails");
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("bogus"), "message: {msg}");
                assert!(msg.contains("linear"), "message names alternatives: {msg}");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn config_rejects_partial_streaming_pair() {
        // hop set, size zero — invalid.
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.streaming_chunk_size = 0;
        c.streaming_chunk_hop = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
        // size set, hop zero — invalid.
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.streaming_chunk_size = 4;
        c.streaming_chunk_hop = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_accepts_both_streaming_zero() {
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.streaming_chunk_size = 0;
        c.streaming_chunk_hop = 0;
        c.validate_for_forward()
            .expect("both-zero streaming is legal (chunk-aware disabled)");
    }

    #[test]
    fn synthesized_weights_are_deterministic_and_shape_correct() {
        let c = CosyVoice3Config::tiny_for_tests();
        let w1 = CosyVoice3Weights::synthesized(&c, 0x42).expect("build 1");
        let w2 = CosyVoice3Weights::synthesized(&c, 0x42).expect("build 2");
        // Determinism.
        assert_eq!(w1.token_embed, w2.token_embed);
        assert_eq!(w1.blocks[0].q_proj, w2.blocks[0].q_proj);
        assert_eq!(w1.blocks[1].ffn_gate, w2.blocks[1].ffn_gate);
        assert!(w1.is_synthesized);

        // Shape flow.
        let d = c.hidden_dim as usize;
        let ffn = c.ffn_dim as usize;
        let vocab = c.vocab_size as usize;
        assert_eq!(w1.token_embed.len(), vocab * d);
        assert_eq!(w1.final_norm.len(), d);
        assert_eq!(w1.blocks.len(), c.n_layer as usize);
        for blk in &w1.blocks {
            assert_eq!(blk.self_attn_norm.len(), d);
            assert_eq!(blk.q_proj.len(), d * d);
            assert_eq!(blk.q_bias.len(), d);
            assert_eq!(blk.k_proj.len(), d * d);
            assert_eq!(blk.k_bias.len(), d);
            assert_eq!(blk.v_proj.len(), d * d);
            assert_eq!(blk.v_bias.len(), d);
            assert_eq!(blk.o_proj.len(), d * d);
            assert_eq!(blk.ffn_norm.len(), d);
            assert_eq!(blk.ffn_gate.len(), ffn * d);
            assert_eq!(blk.ffn_up.len(), ffn * d);
            assert_eq!(blk.ffn_down.len(), d * ffn);
        }
    }

    #[test]
    fn synthesized_weights_different_seeds_diverge() {
        let c = CosyVoice3Config::tiny_for_tests();
        let a = CosyVoice3Weights::synthesized(&c, 1).expect("a");
        let b = CosyVoice3Weights::synthesized(&c, 2).expect("b");
        assert_ne!(a.token_embed, b.token_embed);
        assert_ne!(a.blocks[0].q_proj, b.blocks[0].q_proj);
    }

    #[test]
    fn synthesized_rejects_ill_formed_config() {
        let mut c = CosyVoice3Config::tiny_for_tests();
        c.hidden_dim = 0;
        assert!(matches!(
            CosyVoice3Weights::synthesized(&c, 7),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_accepts_matching_config_and_weights() {
        let c = CosyVoice3Config::tiny_for_tests();
        let w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        let tts = CosyVoice3Tts::new(c.clone(), w).expect("cosyvoice3 tts");
        assert_eq!(tts.config().hidden_dim, c.hidden_dim);
        assert_eq!(tts.config().n_layer, c.n_layer);
        assert_eq!(tts.config().sample_rate, 24_000);
        assert!(tts.is_synthesized());
        assert!(!tts.has_hift_chain(), "fresh load has no HiFTChain");
    }

    #[test]
    fn tts_new_rejects_block_count_mismatch() {
        let c = CosyVoice3Config::tiny_for_tests();
        let mut w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        w.blocks.pop();
        assert!(matches!(
            CosyVoice3Tts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_token_embed_shape_mismatch() {
        let c = CosyVoice3Config::tiny_for_tests();
        let mut w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        w.token_embed.pop();
        assert!(matches!(
            CosyVoice3Tts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_block_qkv_size_mismatch() {
        let c = CosyVoice3Config::tiny_for_tests();
        let mut w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        w.blocks[0].q_proj.pop();
        assert!(matches!(
            CosyVoice3Tts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_ffn_size_mismatch() {
        let c = CosyVoice3Config::tiny_for_tests();
        let mut w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        w.blocks[1].ffn_down.pop();
        assert!(matches!(
            CosyVoice3Tts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tts_new_rejects_final_norm_size_mismatch() {
        let c = CosyVoice3Config::tiny_for_tests();
        let mut w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        w.final_norm.pop();
        assert!(matches!(
            CosyVoice3Tts::new(c, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn synthesize_rejects_empty_text() {
        let c = CosyVoice3Config::tiny_for_tests();
        let w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        let tts = CosyVoice3Tts::new(c, w).expect("cosyvoice3 tts");
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
        let c = CosyVoice3Config::tiny_for_tests();
        let w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        let tts = CosyVoice3Tts::new(c, w).expect("cosyvoice3 tts");
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
    /// pivots to the synthesized-weight blocker.
    #[test]
    fn synthesize_with_chain_and_synthesized_weights_names_the_synthesized_blocker() {
        use vokra_ops::hiftnet::{F0PredictorWeights, ResBlockWeights};

        // Build a small HiFTChain identical in shape to
        // `cosyvoice2::mod::small_hift_chain_for_wiring`.
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

        let c = CosyVoice3Config::tiny_for_tests();
        let w = CosyVoice3Weights::synthesized(&c, 7).expect("weights");
        let tts = CosyVoice3Tts::new(c, w)
            .expect("cosyvoice3 tts")
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
    /// Fun-CosyVoice3 id to Permissive (apache-2.0). Cross-crate test
    /// to keep this module's registry-side contract honest.
    #[test]
    fn registry_lookup_maps_cosyvoice3_to_permissive_apache() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "cosyvoice3",
            "fun-cosyvoice3",
            "fun-cosyvoice3-0.5b",
            "fun-cosyvoice3-0.5b-2512",
            "cosyvoice3-0.5b",
            "cosyvoice3-0.5b-2512",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (apache-2.0)"
            );
        }
    }
}
