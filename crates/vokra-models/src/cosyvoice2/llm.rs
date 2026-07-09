//! CosyVoice2 LLM backbone — decoder-only Mistral-style transformer
//! (M3-09-T07 / T08 body).
//!
//! CosyVoice2 wraps a **decoder-only text-to-token LLM** (the upstream ships a
//! `Qwen2-0.5B` backbone in the `iic/CosyVoice2-0.5B` release) whose output
//! token stream drives the Flow Matching CFM (T10). This file lands the
//! **module + primitive surface + full Mistral-style forward body**
//! (T07 embedding + T08 transformer blocks) driven from **synthesized,
//! seed-deterministic weights**. Real HF-checkpoint parity (T02 tensor
//! manifest) is wired but the harness only runs when the checkpoint arrives
//! — no fabricated pass.
//!
//! # What lands in this session (M3-09 Wave 8 follow-on)
//!
//! - [`LlmBackboneConfig`] — snapshot of the LLM-side hparams the runtime
//!   reads from the `vokra.cosyvoice2.arch.*` chunk group (T04 sub-slice).
//!   Every field is **read from the GGUF** — nothing is hard-coded (CLAUDE.md
//!   "ハルシネーション厳禁": tensor names and hparams live in the metadata,
//!   never in Rust literals).
//! - [`LlmWeights`] — Mistral-style transformer weight store:
//!   token embedding + per-block (RMSNorm γ ×2, GQA Q/K/V/O, SwiGLU
//!   gate/up/down) + final RMSNorm γ, plus a synthesized-fixture builder
//!   ([`LlmWeights::synthesized`]) driven by [`vokra_core::rng::SplitMix64`]
//!   with Xavier-like initialisation so shape + numerical stability can be
//!   verified deterministically **without** the real HF checkpoint.
//! - [`LlmBackbone`] — the top-level type; wraps config + weights + Compute
//!   seam ([`crate::compute::Compute`]) so GPU seams (T19 CUDA / T20 Metal)
//!   inherit the same GEMM path when they are wired.
//! - **Forward paths** — full Mistral pre-norm block sequence:
//!   1. token embedding lookup → `[t, hidden]`,
//!   2. per-block: pre-norm RMSNorm → GQA attention (Q/K/V projections,
//!      RoPE apply, causal-masked softmax, O projection) → residual add
//!      → pre-norm RMSNorm → SwiGLU FFN (gate/up/down) → residual add,
//!   3. final RMSNorm,
//!   4. tied-logits head (`logits = h @ token_emb^T`).
//!   - [`LlmBackbone::forward`] — bulk forward over `[t]` tokens, returns
//!     `[t, vocab]` logits.
//!   - [`LlmBackbone::step`] — one autoregressive step with KV cache
//!     append, returns `[vocab]` logits for the new position.
//!   - [`LlmBackbone::greedy_decode`] — argmax loop with early stop on EOS
//!     and a configurable `max_new_tokens` cap.
//! - **Shared primitives** ([`voxtral::text_decoder::rms_norm`],
//!   [`voxtral::text_decoder::silu_inplace`],
//!   [`voxtral::text_decoder::hadamard_inplace`],
//!   [`voxtral::text_decoder::rope_apply`]) — imported at module level so
//!   the Mistral GQA/RoPE/SwiGLU/RMSNorm primitives serve both models. A
//!   future GQA bug fix lands once.
//!
//! # Real-checkpoint parity contract (T02 owner runs)
//!
//! - The upstream `iic/CosyVoice2-0.5B` HuggingFace release ships a
//!   Qwen2-0.5B backbone; the T02 owner ticket records
//!   `tensor_name → shape/dtype` in `docs/adr/M3-09-cosyvoice2.md` §T02
//!   and feeds `vokra-convert::models::cosyvoice2` (T03).
//! - Once the GGUF fixture arrives, [`LlmBackbone::from_gguf_with_weights`]
//!   swaps the synthesized store for a real `LlmWeights::from_gguf` binding
//!   (that method is a `NotImplemented` stub today so the owner-side path
//!   fails loudly — never a silent zero-fill).
//! - The parity harness lives beside [`crate::cosyvoice2::llm::parity`] —
//!   `forward_matches_step_by_step` is a deterministic property test that
//!   runs today; `parity_vs_hf_reference` is the flip-the-switch owner
//!   test that returns [`VokraError::NotImplemented`] until the fixture
//!   arrives.
//!
//! # No silent fallback (FR-EX-08)
//!
//! - Weights not bound: forward returns
//!   [`VokraError::ModelLoad`] with the missing tensor named — never a
//!   zero-fill.
//! - Config not GQA-well-formed: [`VokraError::InvalidArgument`] with the
//!   offending dims listed.
//! - Position past `n_ctx`: [`VokraError::InvalidArgument`] — no silent
//!   wrap-around.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, KvCache, Result, VokraError};

use super::config::CosyVoice2Config;
use crate::compute::{Compute, HotOp};
use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, rope_apply, silu_inplace};

/// Compute-seam hot ops the CosyVoice2 LLM backbone dispatches (Mistral
/// pre-norm block: GEMM for Q/K/V/O + FFN gate/up/down, GEMV for the tied
/// logits head, softmax for causal attention). RMSNorm / SwiGLU are pure
/// scalar loops and do not go through the seam. Kept module-local so a
/// GPU-backed session (T19 CUDA / T20 Metal) advertises the same coverage
/// gate.
const LLM_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv, HotOp::Softmax];

// --- `vokra.cosyvoice2.arch.*` LLM-side metadata keys ----------------------
//
// The five keys in [`CosyVoice2Config`] (vocab_size / hidden_dim / n_layer /
// n_head / ffn_dim) are already read; here we add the LLM-specific keys the
// backbone needs on top. All are optional today (converter shape-only path,
// T02 upstream inspection still open) — a `0` / sentinel means "not yet
// populated". The forward path enforces `!= 0` at first use so a `0`-
// placeholder GGUF fails loudly at the earliest wrong shape rather than
// silently deep inside a GEMM.

pub(crate) const KEY_LLM_N_HEAD_KV: &str = "vokra.cosyvoice2.arch.n_head_kv";
pub(crate) const KEY_LLM_ROPE_BASE: &str = "vokra.cosyvoice2.arch.rope_base";
pub(crate) const KEY_LLM_RMS_NORM_EPS: &str = "vokra.cosyvoice2.arch.rms_norm_eps";
pub(crate) const KEY_LLM_N_CTX: &str = "vokra.cosyvoice2.arch.n_ctx";

/// Safety-net RoPE base used **only** when the GGUF omits the key.
/// Matches the Qwen2 family default (Mistral / Qwen2 modern releases ship
/// `1_000_000.0`), so a well-formed GGUF trivially agrees. See
/// <https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json> field
/// `rope_theta` — the T02 owner ticket records the actual value the
/// CosyVoice2-0.5B ships.
pub const DEFAULT_ROPE_BASE_QWEN2: f32 = 1_000_000.0;

/// Same posture for RMSNorm ε. Mistral / Qwen2 ship `1e-5` per HF config
/// (`rms_norm_eps`).
pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

/// Seed for the synthesized weight fixture built by
/// [`LlmBackbone::from_gguf`] on the shape-only converter path (T02
/// tensor-store binding is still open). Arbitrary but stable so callers
/// can reproduce byte-for-byte; the constant reads as ASCII
/// `"cosyv0.9\0"` mixed with `0xC0DE_C0DE` to make it distinct from the
/// Voxtral / Kokoro fixtures.
pub const FROM_GGUF_DEFAULT_SEED: u64 = 0xC0DE_C0DE_C0DE_C0DE;

/// LLM-side hparam snapshot resolved from the CosyVoice2 GGUF metadata.
///
/// Kept separate from [`CosyVoice2Config`] so the LLM backbone surface can
/// evolve (add GQA head split / RoPE base / n_ctx / rms-norm ε) without
/// churning the top-level config. All fields are read at load time from the
/// GGUF; nothing here is hard-coded (CLAUDE.md hallucination ban).
#[derive(Debug, Clone)]
pub struct LlmBackboneConfig {
    /// Vocab table size (LLM input embedding rows) — mirrors
    /// [`CosyVoice2Config::vocab_size`].
    pub vocab_size: usize,
    /// Hidden width (`d`) — mirrors [`CosyVoice2Config::hidden_dim`].
    pub hidden_dim: usize,
    /// Transformer block count — mirrors [`CosyVoice2Config::n_layer`].
    pub n_layer: usize,
    /// Query attention heads (mirrors [`CosyVoice2Config::n_head`]).
    pub n_head_q: usize,
    /// Key/value attention heads (GQA — `n_head_q % n_head_kv == 0`). Read
    /// from `vokra.cosyvoice2.arch.n_head_kv`; falls back to `n_head_q`
    /// (i.e. MHA) when the key is absent, matching the classical
    /// Transformer-XL / Vaswani-2017 default.
    pub n_head_kv: usize,
    /// SwiGLU FFN inner width — mirrors [`CosyVoice2Config::ffn_dim`].
    pub ffn_dim: usize,
    /// RoPE base θ. `vokra.cosyvoice2.arch.rope_base` if present, else
    /// [`DEFAULT_ROPE_BASE_QWEN2`].
    pub rope_base: f32,
    /// RMSNorm ε. `vokra.cosyvoice2.arch.rms_norm_eps` if present, else
    /// [`DEFAULT_RMS_NORM_EPS`].
    pub rms_norm_eps: f32,
    /// Max sequence length the LLM backbone supports (positional table +
    /// paged KV cache reserve). `vokra.cosyvoice2.arch.n_ctx` if present,
    /// else `0` — the forward path rejects `0` before running.
    pub n_ctx: usize,
}

impl LlmBackboneConfig {
    /// Reads the LLM backbone hparams from a CosyVoice2 GGUF file. Never
    /// invents a value — a key that is present but of the wrong type is a
    /// loud [`VokraError::InvalidArgument`] (FR-EX-08).
    ///
    /// Pre-condition: `cfg` was read from the same file via
    /// [`CosyVoice2Config::from_gguf`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any present LLM-side key has the
    /// wrong metadata type.
    pub fn from_gguf(file: &GgufFile, cfg: &CosyVoice2Config) -> Result<Self> {
        let n_head_q = cfg.n_head as usize;
        let n_head_kv = match file.get(KEY_LLM_N_HEAD_KV) {
            Some(GgufMetadataValue::U32(v)) => *v as usize,
            None => n_head_q, // MHA fallback (documented above).
            Some(_) => {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LLM: `{KEY_LLM_N_HEAD_KV}` is not a UINT32"
                )));
            }
        };
        let rope_base = match file.get(KEY_LLM_ROPE_BASE) {
            Some(GgufMetadataValue::F32(v)) => *v,
            None => DEFAULT_ROPE_BASE_QWEN2,
            Some(_) => {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LLM: `{KEY_LLM_ROPE_BASE}` is not a FLOAT32"
                )));
            }
        };
        let rms_norm_eps = match file.get(KEY_LLM_RMS_NORM_EPS) {
            Some(GgufMetadataValue::F32(v)) => *v,
            None => DEFAULT_RMS_NORM_EPS,
            Some(_) => {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LLM: `{KEY_LLM_RMS_NORM_EPS}` is not a FLOAT32"
                )));
            }
        };
        let n_ctx = match file.get(KEY_LLM_N_CTX) {
            Some(GgufMetadataValue::U32(v)) => *v as usize,
            None => 0, // Placeholder — the forward path refuses to run at n_ctx = 0.
            Some(_) => {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LLM: `{KEY_LLM_N_CTX}` is not a UINT32"
                )));
            }
        };
        Ok(Self {
            vocab_size: cfg.vocab_size as usize,
            hidden_dim: cfg.hidden_dim as usize,
            n_layer: cfg.n_layer as usize,
            n_head_q,
            n_head_kv,
            ffn_dim: cfg.ffn_dim as usize,
            rope_base,
            rms_norm_eps,
            n_ctx,
        })
    }

    /// Per-query-head width (`hidden_dim / n_head_q`). Returns `0` when
    /// `n_head_q == 0` (shape-only converter sentinel) so callers can pass
    /// this to a shape check without panicking (FR-EX-08).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim.checked_div(self.n_head_q).unwrap_or(0)
    }

    /// KV-head width. `n_head_kv * head_dim` (GQA broadcasts K/V heads to
    /// `n_head_q / n_head_kv` query heads each). Returns `0` on any zero
    /// component (shape-only converter sentinel).
    #[must_use]
    pub fn kv_hidden_dim(&self) -> usize {
        self.n_head_kv.saturating_mul(self.head_dim())
    }

    /// True when this config satisfies the GQA algebraic constraint
    /// `n_head_q % n_head_kv == 0` and `hidden_dim % n_head_q == 0` — the
    /// forward path requires both. When either fails, the caller has
    /// misconfigured metadata and the forward will refuse to run.
    #[must_use]
    pub fn is_gqa_well_formed(&self) -> bool {
        self.n_head_q != 0
            && self.n_head_kv != 0
            && self.n_head_q % self.n_head_kv == 0
            && self.hidden_dim % self.n_head_q == 0
    }
}

/// Per-block Mistral-style weight bundle.
///
/// - `attn_norm_gamma` / `ffn_norm_gamma` — RMSNorm scales (`[hidden_dim]`,
///   no bias — RMSNorm is scale-only).
/// - `q_w_t` / `k_w_t` / `v_w_t` / `o_w_t` — attention projection weights
///   stored **already transposed** for row-major GEMM
///   (`out[i,j] = Σ_l x[i,l] * w_t[l,j]`); the incoming safetensors dumps
///   are `[out, in]` and get transposed once at load time (T07).
///   Shapes: `q_w_t = [d, d]`, `k_w_t = v_w_t = [d, kv_hidden]`,
///   `o_w_t = [d, d]`.
/// - `ffn_gate_w_t` / `ffn_up_w_t` — SwiGLU projections
///   (`[d, ffn_dim]` each), `ffn_down_w_t` — down projection
///   (`[ffn_dim, d]`).
///
/// The transformer projections are **bias-less** (Mistral / Qwen2
/// convention).
#[derive(Debug, Clone)]
pub struct LlmBlockWeights {
    /// Pre-attention RMSNorm γ (`[hidden_dim]`, no bias — RMSNorm is
    /// scale-only).
    pub attn_norm_gamma: Vec<f32>,
    /// Q projection weight, row-major `[hidden_dim, hidden_dim]` already
    /// transposed for `Compute::gemm_f32`.
    pub q_w_t: Vec<f32>,
    /// K projection weight, row-major `[hidden_dim, kv_hidden_dim]`
    /// already transposed for row-major GEMM. `kv_hidden_dim = n_head_kv
    /// * head_dim`.
    pub k_w_t: Vec<f32>,
    /// V projection weight, row-major `[hidden_dim, kv_hidden_dim]`
    /// already transposed.
    pub v_w_t: Vec<f32>,
    /// O (output) projection weight, row-major `[hidden_dim, hidden_dim]`
    /// already transposed.
    pub o_w_t: Vec<f32>,
    /// Pre-FFN RMSNorm γ (`[hidden_dim]`).
    pub ffn_norm_gamma: Vec<f32>,
    /// SwiGLU gate projection weight, row-major `[hidden_dim, ffn_dim]`.
    pub ffn_gate_w_t: Vec<f32>,
    /// SwiGLU up projection weight, row-major `[hidden_dim, ffn_dim]`.
    pub ffn_up_w_t: Vec<f32>,
    /// SwiGLU down projection weight, row-major `[ffn_dim, hidden_dim]`.
    pub ffn_down_w_t: Vec<f32>,
}

/// All LLM backbone weights.
///
/// - `token_emb` — `[vocab_size, hidden_dim]` row-major.
///   Also the **tied LM head** (`logits = h @ token_emb^T`), matching
///   Mistral / Qwen2 convention.
/// - `blocks` — one [`LlmBlockWeights`] per transformer layer.
/// - `final_norm_gamma` — post-block RMSNorm scale (`[hidden_dim]`).
#[derive(Debug, Clone)]
pub struct LlmWeights {
    /// Token embedding table (`[vocab_size, hidden_dim]` row-major).
    /// Also the tied LM head (`logits = h @ token_emb^T`).
    pub token_emb: Vec<f32>,
    /// Per-layer transformer block weight bundles.
    pub blocks: Vec<LlmBlockWeights>,
    /// Post-block final RMSNorm γ (`[hidden_dim]`, no bias).
    pub final_norm_gamma: Vec<f32>,
    /// Marker: `true` when the weights come from
    /// [`LlmWeights::synthesized`], `false` when they came from
    /// [`LlmWeights::from_gguf`]. Callers use this to gate real-checkpoint
    /// parity assertions without accidentally running them against the
    /// synthetic fixture.
    pub is_synthesized: bool,
}

impl LlmWeights {
    /// Builds a **synthesized** weight store from `config` and `seed`.
    ///
    /// # Init strategy
    ///
    /// The weights are drawn from [`SplitMix64`] mapped to a
    /// **uniform in `(-bound, +bound)`** where
    /// `bound = sqrt(6 / (fan_in + fan_out))` — the standard Xavier / Glorot
    /// bound for `nn.Linear` in PyTorch. RMSNorm γ vectors are initialised
    /// to `1.0` (the PyTorch default) so a fresh Mistral block is close to
    /// an identity residual on the first step, ensuring `NaN`-free
    /// end-to-end runs on any config that satisfies the GQA constraint.
    ///
    /// This is **not** a bit-for-bit reproduction of PyTorch's initialisation
    /// (PyTorch's `torch.nn.init.uniform_` uses a different PRNG order); it
    /// exists purely for numerical-stability / shape verification without
    /// the real HF checkpoint.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `config` is not GQA-well-formed
    ///   (dims cannot host a Mistral block).
    pub fn synthesized(config: &LlmBackboneConfig, seed: u64) -> Result<Self> {
        if !config.is_gqa_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM synthesized weights: config not GQA well-formed \
                 (n_head_q={}, n_head_kv={}, hidden_dim={}, vocab={}, n_layer={}, ffn_dim={})",
                config.n_head_q,
                config.n_head_kv,
                config.hidden_dim,
                config.vocab_size,
                config.n_layer,
                config.ffn_dim,
            )));
        }
        if config.vocab_size == 0 || config.n_layer == 0 || config.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM synthesized weights: zero-size hparam \
                 (vocab={}, n_layer={}, ffn_dim={})",
                config.vocab_size, config.n_layer, config.ffn_dim
            )));
        }
        let d = config.hidden_dim;
        let kv_hidden = config.kv_hidden_dim();
        let ffn = config.ffn_dim;
        let vocab = config.vocab_size;

        let mut rng = SplitMix64::new(seed);
        // Token embedding: fan_in = fan_out = hidden_dim.
        let token_emb = xavier_uniform(&mut rng, vocab * d, d, d);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            let attn_norm_gamma = vec![1.0f32; d];
            let ffn_norm_gamma = vec![1.0f32; d];
            // Attention projections.
            let q_w_t = xavier_uniform(&mut rng, d * d, d, d);
            let k_w_t = xavier_uniform(&mut rng, d * kv_hidden, d, kv_hidden);
            let v_w_t = xavier_uniform(&mut rng, d * kv_hidden, d, kv_hidden);
            let o_w_t = xavier_uniform(&mut rng, d * d, d, d);
            // SwiGLU FFN.
            let ffn_gate_w_t = xavier_uniform(&mut rng, d * ffn, d, ffn);
            let ffn_up_w_t = xavier_uniform(&mut rng, d * ffn, d, ffn);
            let ffn_down_w_t = xavier_uniform(&mut rng, ffn * d, ffn, d);
            blocks.push(LlmBlockWeights {
                attn_norm_gamma,
                q_w_t,
                k_w_t,
                v_w_t,
                o_w_t,
                ffn_norm_gamma,
                ffn_gate_w_t,
                ffn_up_w_t,
                ffn_down_w_t,
            });
        }
        let final_norm_gamma = vec![1.0f32; d];
        Ok(Self {
            token_emb,
            blocks,
            final_norm_gamma,
            is_synthesized: true,
        })
    }

    /// Loads real weights from the CosyVoice2 GGUF via T07 tensor-store
    /// binding.
    ///
    /// **Not implemented today** — the T02 upstream inspection is still
    /// open, so the tensor-name manifest does not exist. The runtime does
    /// not fabricate names (CLAUDE.md hallucination ban). When the T02
    /// owner ticket delivers `docs/adr/M3-09-cosyvoice2.md` §T02 with the
    /// `tensor_name → shape/dtype` table, this method binds every tensor
    /// verbatim through [`GgufFile::tensor_f32`] and returns
    /// `is_synthesized = false`.
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] — the honest signal until T02 lands.
    /// After T02 lands, propagates any tensor-missing / mis-shaped error
    /// verbatim with the offending name.
    pub fn from_gguf(_file: &GgufFile, _config: &LlmBackboneConfig) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "CosyVoice2 LLM real-weight binding is deferred to T02 upstream \
             inspection + T07 tensor-store scan; today the runtime refuses to \
             invent tensor names (FR-EX-08 — never a silent zero-fill fallback). \
             Use LlmWeights::synthesized(config, seed) for the numerical \
             stability / shape verification path against the deterministic \
             fixture.",
        ))
    }
}

/// Draws `n` f32 values uniformly in `(-bound, +bound)` where
/// `bound = sqrt(6 / (fan_in + fan_out))` (Xavier / Glorot).
fn xavier_uniform(rng: &mut SplitMix64, n: usize, fan_in: usize, fan_out: usize) -> Vec<f32> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // next_unit_f32() ∈ (0, 1); map to (-bound, +bound).
        let u = rng.next_unit_f32();
        v.push((u * 2.0 - 1.0) * bound);
    }
    v
}

/// CosyVoice2 LLM backbone — top-level type (M3-09-T08 body).
///
/// Owns the resolved [`LlmBackboneConfig`], the loaded [`LlmWeights`], and
/// a selected [`BackendKind`] (the [`Compute`] dispatcher is built on
/// demand per forward call, so the CUDA `!Sync` context never leaks into
/// the engine — the piper-plus pattern). The forward paths
/// ([`Self::forward`], [`Self::step`], [`Self::greedy_decode`]) run the
/// full Mistral pre-norm block sequence.
pub struct LlmBackbone {
    config: LlmBackboneConfig,
    weights: LlmWeights,
    backend: BackendKind,
}

impl std::fmt::Debug for LlmBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Weights are large; log only the shape summary + backend so a
        // debug print does not flood.
        f.debug_struct("LlmBackbone")
            .field("config", &self.config)
            .field("weights.is_synthesized", &self.weights.is_synthesized)
            .field("weights.n_blocks", &self.weights.blocks.len())
            .field("backend", &self.backend)
            .finish()
    }
}

impl LlmBackbone {
    /// Builds a backbone with an explicit weight store on the CPU backend.
    ///
    /// The default path is [`Self::synthesized`] for the fixture-driven
    /// integration tests; a follow-on ticket (T19/T20) will add a
    /// `with_backend` builder that routes GEMM through the Metal / CUDA
    /// seam once those arms compile the CosyVoice2 hot ops.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `config` is not GQA-well-formed
    ///   or the weight shapes disagree with `config`.
    pub fn new(config: LlmBackboneConfig, weights: LlmWeights) -> Result<Self> {
        validate_shapes(&config, &weights)?;
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
        })
    }

    /// Selects the [`BackendKind`] the forward path dispatches through.
    /// The [`Compute`] dispatcher is built on-demand per forward call
    /// (the piper-plus pattern) so the CUDA `!Sync` context does not
    /// leak into the engine.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The currently selected backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Builds a [`Compute`] dispatcher for the selected backend + LLM hot
    /// ops. Called once per forward invocation (piper-plus pattern) so
    /// non-`Sync` GPU contexts stay on the stack.
    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, LLM_HOT_OPS)
    }

    /// Convenience: builds a backbone with synthesized (seed-deterministic)
    /// weights for the numerical-stability / shape verification path.
    ///
    /// # Errors
    ///
    /// Propagates [`LlmWeights::synthesized`] errors.
    pub fn synthesized(config: LlmBackboneConfig, seed: u64) -> Result<Self> {
        let weights = LlmWeights::synthesized(&config, seed)?;
        Self::new(config, weights)
    }

    /// Loads the LLM backbone from a CosyVoice2 GGUF file with **synthesized**
    /// weights — the honest bridge for the shape-only converter path today.
    ///
    /// Reads the shape config verbatim from the GGUF metadata, then builds
    /// synthesized weights against that shape. This lets the caller
    /// (`CosyVoice2Tts::from_gguf_with_policy`) hold a config-only handle
    /// today; once T02/T07 land the real tensor-store binding path,
    /// [`Self::from_gguf_with_weights`] swaps the synthesized store for the
    /// real one.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on any GGUF metadata key with a
    ///   wrong type.
    /// - [`VokraError::InvalidArgument`] if the config carries a
    ///   0-placeholder sentinel (converter shape-only path — no synthesized
    ///   fixture is meaningful at zero dims).
    pub fn from_gguf(file: &GgufFile, cfg: &CosyVoice2Config) -> Result<Self> {
        let llm_cfg = LlmBackboneConfig::from_gguf(file, cfg)?;
        // Reject the 0-placeholder path — a converter without dims cannot
        // host a fixture (FR-EX-08, no silent zero-fill fallback).
        if !llm_cfg.is_gqa_well_formed()
            || llm_cfg.vocab_size == 0
            || llm_cfg.n_layer == 0
            || llm_cfg.ffn_dim == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone: GGUF carries a 0-placeholder shape config \
                 (vocab={}, n_layer={}, n_head_q={}, n_head_kv={}, hidden={}, ffn={}) — \
                 the shape-only converter path cannot host a synthesized fixture. \
                 Re-convert with real hparams (T04) or bind against a fixture-shaped \
                 config via LlmBackbone::synthesized directly.",
                llm_cfg.vocab_size,
                llm_cfg.n_layer,
                llm_cfg.n_head_q,
                llm_cfg.n_head_kv,
                llm_cfg.hidden_dim,
                llm_cfg.ffn_dim,
            )));
        }
        // Default seed for the from-GGUF path: arbitrary but stable
        // 64-bit constant, documented so callers can reproduce the
        // synthesized fixture bit-for-bit. Once T02/T07 land, this path
        // swaps for the real weight binding via
        // [`Self::from_gguf_with_weights`].
        Self::synthesized(llm_cfg, FROM_GGUF_DEFAULT_SEED)
    }

    /// Loads the LLM backbone from a CosyVoice2 GGUF **with real weights**
    /// via the T07 tensor-store scan.
    ///
    /// Today this delegates to [`LlmWeights::from_gguf`] which returns
    /// [`VokraError::NotImplemented`] until T02 lands.
    ///
    /// # Errors
    ///
    /// See [`LlmWeights::from_gguf`].
    pub fn from_gguf_with_weights(file: &GgufFile, cfg: &CosyVoice2Config) -> Result<Self> {
        let llm_cfg = LlmBackboneConfig::from_gguf(file, cfg)?;
        let weights = LlmWeights::from_gguf(file, &llm_cfg)?;
        Self::new(llm_cfg, weights)
    }

    /// The resolved LLM hparams.
    #[must_use]
    pub fn config(&self) -> &LlmBackboneConfig {
        &self.config
    }

    /// The weight store (useful for parity tests + shape sanity checks).
    #[must_use]
    pub fn weights(&self) -> &LlmWeights {
        &self.weights
    }

    /// Compute seam name (`"cpu"` / `"metal"` / `"cuda"`). Resolves the
    /// on-demand [`Compute`] dispatcher just to read its label; the same
    /// error surface as [`Self::forward`] applies (if the backend is
    /// unavailable in this build, returns "cpu" — the coverage check runs
    /// at forward-time, not on `backend_name`).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        // The name is available from the compute dispatcher when it
        // builds cleanly. If the backend is not available in this
        // build, we still return a stable label from the BackendKind.
        match self.compute() {
            Ok(c) => c.backend_name(),
            Err(_) => match self.backend {
                BackendKind::Cpu => "cpu",
                BackendKind::Metal => "metal",
                BackendKind::Cuda => "cuda",
                BackendKind::Vulkan => "vulkan",
                _ => "unknown",
            },
        }
    }

    /// Runs the LLM backbone forward once over `token_ids` and produces the
    /// per-token logits (`[t, vocab_size]` row-major).
    ///
    /// This is the **bulk forward** used by the parity harness and the
    /// initial prefix pass of a greedy decode. Every step recomputes from
    /// scratch (no KV cache is carried across invocations); use
    /// [`Self::step`] for the autoregressive path that appends to a KV
    /// cache.
    ///
    /// # Arguments
    ///
    /// - `token_ids` — the input token ids (`t` positions).
    /// - `position_offset` — absolute position of `token_ids[0]` in the
    ///   full decode. Used by RoPE and the causal mask. Callers building a
    ///   bulk forward from scratch pass `0`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if the config is not
    ///   GQA-well-formed, if any token id is out of range, or if
    ///   `position_offset + t > config.n_ctx` (when `n_ctx != 0`).
    pub fn forward(&self, token_ids: &[u32], position_offset: usize) -> Result<Vec<f32>> {
        if !self.config.is_gqa_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone: config not GQA well-formed \
                 (n_head_q={}, n_head_kv={}, hidden_dim={}) — need \
                 n_head_q > 0, n_head_kv > 0, n_head_q % n_head_kv == 0, \
                 hidden_dim % n_head_q == 0",
                self.config.n_head_q, self.config.n_head_kv, self.config.hidden_dim,
            )));
        }
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let t = token_ids.len();
        if self.config.n_ctx != 0 && position_offset + t > self.config.n_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone forward: position_offset + t = {} > n_ctx {}",
                position_offset + t,
                self.config.n_ctx
            )));
        }
        // Bulk forward: build a fresh KV cache sized to `t` and run one step
        // over the whole prefix. The result is the `[t, vocab]` logits row.
        let compute = self.compute()?;
        let mut kv = KvCache::with_reserve(self.config.n_layer, self.config.kv_hidden_dim(), t);
        forward_impl(
            &compute,
            &self.config,
            &self.weights,
            &mut kv,
            token_ids,
            position_offset,
        )
    }

    /// Runs a single **decode step** over one new token, appending its K/V
    /// rows to the caller-supplied [`LlmBackboneStep`] state and returning
    /// the `[vocab]` logits row for the new position.
    ///
    /// The step is the autoregressive workhorse: the KV cache grows one
    /// position per call, and `state.seq_len` advances by 1.
    ///
    /// # Arguments
    ///
    /// - `state` — the running decode state (KV cache + position counter).
    ///   Must have been constructed via [`LlmBackboneStep::new`] on the
    ///   same config.
    /// - `token_id` — the new token id to decode.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `state.seq_len >= config.n_ctx`
    ///   (`n_ctx != 0` case) or if `token_id >= vocab_size`.
    pub fn step(&self, state: &mut LlmBackboneStep, token_id: u32) -> Result<Vec<f32>> {
        if self.config.n_ctx != 0 && state.seq_len >= self.config.n_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone: seq_len {} would exceed n_ctx {} \
                 (FR-EX-08 — no silent wrap-around)",
                state.seq_len, self.config.n_ctx
            )));
        }
        if !self.config.is_gqa_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone step: config not GQA well-formed \
                 (n_head_q={}, n_head_kv={}, hidden_dim={})",
                self.config.n_head_q, self.config.n_head_kv, self.config.hidden_dim,
            )));
        }
        if state.kv_cache.is_none() {
            state.kv_cache = Some(KvCache::with_reserve(
                self.config.n_layer,
                self.config.kv_hidden_dim(),
                self.config.n_ctx.max(64),
            ));
        }
        let compute = self.compute()?;
        let kv = state
            .kv_cache
            .as_mut()
            .expect("KvCache just allocated above");
        // Run a single-token forward with the current position offset.
        let logits = forward_impl(
            &compute,
            &self.config,
            &self.weights,
            kv,
            &[token_id],
            state.seq_len,
        )?;
        // The returned logits are `[1, vocab]`; the last (only) row is the
        // new position's logits. Advance the state clock.
        state.seq_len += 1;
        // Trim to just the new position's logits row.
        Ok(logits)
    }

    /// Greedy autoregressive decode.
    ///
    /// Runs [`Self::step`] in a loop from `initial_tokens`, argmax-sampling
    /// each new token from the returned logits. Stops on `eos` (`eos` IS
    /// included in the returned sequence) or after `max_new_tokens`
    /// new tokens.
    ///
    /// # Arguments
    ///
    /// - `initial_tokens` — the prefix to prime the decode with. At least
    ///   one token is required (the first step needs somewhere to start).
    /// - `eos` — the end-of-sequence token id. Pass a value outside the
    ///   vocab range (e.g. `u32::MAX`) to disable early stopping.
    /// - `max_new_tokens` — max newly generated tokens (does not include
    ///   `initial_tokens`).
    ///
    /// Returns the generated token ids **excluding** the prefix (the same
    /// convention `voxtral::greedy_decode` uses).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `initial_tokens` is empty or
    ///   any id (including `eos` when < vocab_size) is out of range.
    pub fn greedy_decode(
        &self,
        initial_tokens: &[u32],
        eos: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        if initial_tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 LLM greedy_decode: initial_tokens must be non-empty".into(),
            ));
        }
        let vocab = self.config.vocab_size as u32;
        for &tok in initial_tokens {
            if tok >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LLM greedy_decode: initial_tokens contains {tok} >= vocab {vocab}"
                )));
            }
        }
        let mut state = LlmBackboneStep::new();
        // Consume the prefix into the KV cache; the final step's logits
        // seed the first sampled token.
        let mut last_logits: Option<Vec<f32>> = None;
        for &tok in initial_tokens {
            last_logits = Some(self.step(&mut state, tok)?);
        }
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let logits = last_logits
                .as_ref()
                .expect("prefix step populates last_logits");
            let next = argmax_u32(logits);
            generated.push(next);
            if next == eos {
                break;
            }
            // Guard: refuse to run past n_ctx (the step call itself would
            // enforce this, but a clean break avoids the error path).
            if self.config.n_ctx != 0 && state.seq_len >= self.config.n_ctx {
                break;
            }
            last_logits = Some(self.step(&mut state, next)?);
        }
        Ok(generated)
    }
}

/// Autoregressive decode state — CosyVoice2's analog of Whisper's
/// `DecoderState` and Voxtral's `TextDecoderStep`.
///
/// Carries the number of tokens processed and an owned KV cache. The cache
/// is lazy-allocated on the first [`LlmBackbone::step`] call, so callers can
/// construct a state without knowing the config's `kv_hidden_dim` in
/// advance.
///
/// Not `Clone` because a `KvCache` owns growable buffers — cloning a
/// running decode session is a specific ticket (M3-14 barge-in already
/// covers the interrupt path), not a byproduct of the type surface.
///
/// `Debug` is implemented by hand because [`KvCache`] does not derive
/// `Debug` — its `Vec<f32>` buffers would flood any log. We surface the
/// scalar counters + a `positions` summary instead.
#[derive(Default)]
pub struct LlmBackboneStep {
    /// Number of tokens processed so far.
    pub seq_len: usize,
    /// Owned per-layer KV cache. `None` before the first step; allocated
    /// on the first [`LlmBackbone::step`] call against the config's dims.
    pub kv_cache: Option<KvCache>,
}

impl std::fmt::Debug for LlmBackboneStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmBackboneStep")
            .field("seq_len", &self.seq_len)
            .field(
                "kv_cache_positions",
                &self.kv_cache.as_ref().map(|c| c.positions()),
            )
            .finish()
    }
}

impl LlmBackboneStep {
    /// Fresh state (nothing decoded).
    #[must_use]
    pub fn new() -> Self {
        Self {
            seq_len: 0,
            kv_cache: None,
        }
    }

    /// Advance one token (increment `seq_len`). Does **not** touch the KV
    /// cache — the runtime uses this on the scaffolding tests
    /// (`voxtral::TextDecoderStep::advance` pattern parity) and on
    /// non-forward passes (e.g. counter-based structural tests).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Rewinds the state for a fresh decode of the same model. The KV
    /// cache's reserved capacity is retained (fast re-use).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        if let Some(kv) = self.kv_cache.as_mut() {
            kv.reset();
        }
    }
}

// -----------------------------------------------------------------------------
// Forward implementation (internal)
// -----------------------------------------------------------------------------

fn validate_shapes(config: &LlmBackboneConfig, weights: &LlmWeights) -> Result<()> {
    if !config.is_gqa_well_formed() {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 LlmBackbone::new: config not GQA well-formed \
             (n_head_q={}, n_head_kv={}, hidden_dim={})",
            config.n_head_q, config.n_head_kv, config.hidden_dim,
        )));
    }
    let d = config.hidden_dim;
    let kv_hidden = config.kv_hidden_dim();
    let ffn = config.ffn_dim;
    let vocab = config.vocab_size;
    let n_layer = config.n_layer;

    if weights.token_emb.len() != vocab * d {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 LlmBackbone::new: token_emb len {} != vocab*hidden {}",
            weights.token_emb.len(),
            vocab * d,
        )));
    }
    if weights.blocks.len() != n_layer {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 LlmBackbone::new: blocks {} != config n_layer {n_layer}",
            weights.blocks.len(),
        )));
    }
    if weights.final_norm_gamma.len() != d {
        return Err(VokraError::InvalidArgument(format!(
            "cosyvoice2 LlmBackbone::new: final_norm_gamma len {} != hidden {}",
            weights.final_norm_gamma.len(),
            d,
        )));
    }
    for (i, b) in weights.blocks.iter().enumerate() {
        let checks = [
            ("attn_norm_gamma", b.attn_norm_gamma.len(), d),
            ("q_w_t", b.q_w_t.len(), d * d),
            ("k_w_t", b.k_w_t.len(), d * kv_hidden),
            ("v_w_t", b.v_w_t.len(), d * kv_hidden),
            ("o_w_t", b.o_w_t.len(), d * d),
            ("ffn_norm_gamma", b.ffn_norm_gamma.len(), d),
            ("ffn_gate_w_t", b.ffn_gate_w_t.len(), d * ffn),
            ("ffn_up_w_t", b.ffn_up_w_t.len(), d * ffn),
            ("ffn_down_w_t", b.ffn_down_w_t.len(), ffn * d),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice2 LlmBackbone::new: block[{i}].{name} len {got} != expected {want}",
                )));
            }
        }
    }
    Ok(())
}

/// Runs a Mistral pre-norm forward pass over `tokens` and returns
/// `[t, vocab]` logits. `kv_cache` is appended in place with the K/V rows
/// for every layer.
fn forward_impl(
    compute: &Compute,
    config: &LlmBackboneConfig,
    weights: &LlmWeights,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    position_offset: usize,
) -> Result<Vec<f32>> {
    let t = tokens.len();
    if t == 0 {
        return Ok(Vec::new());
    }
    let d = config.hidden_dim;
    let n_head_q = config.n_head_q;
    let n_head_kv = config.n_head_kv;
    let head_dim = config.head_dim();
    let kv_hidden = config.kv_hidden_dim();
    let ffn = config.ffn_dim;
    let vocab = config.vocab_size;
    let n_kv_groups = n_head_q / n_head_kv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let eps = config.rms_norm_eps;
    let rope_base = config.rope_base;

    // Token embedding lookup → h `[t, d]`.
    let mut h = vec![0.0f32; t * d];
    for (i, &tok) in tokens.iter().enumerate() {
        let tok = tok as usize;
        if tok >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM forward: token id {tok} >= vocab {vocab}"
            )));
        }
        let src = &weights.token_emb[tok * d..(tok + 1) * d];
        let dst = &mut h[i * d..(i + 1) * d];
        dst.copy_from_slice(src);
    }

    // Per-block scratch (reused across layers; sized once).
    let mut norm = vec![0.0f32; t * d];
    let mut q_proj = vec![0.0f32; t * d];
    let mut k_proj = vec![0.0f32; t * kv_hidden];
    let mut v_proj = vec![0.0f32; t * kv_hidden];
    let mut rope_scratch = vec![0.0f32; t * head_dim];
    let mut attn_out = vec![0.0f32; t * d];
    let mut attn_o = vec![0.0f32; t * d];
    let mut ffn_gate = vec![0.0f32; t * ffn];
    let mut ffn_up = vec![0.0f32; t * ffn];
    let mut ffn_down = vec![0.0f32; t * d];

    for (layer_idx, block) in weights.blocks.iter().enumerate() {
        // ---------- Pre-norm self-attention ----------
        rms_norm(&h, &block.attn_norm_gamma, eps, t, &mut norm)?;

        // Q = norm @ q_w_t: [t, d] × [d, d] → [t, d]
        compute.gemm_f32(t, d, d, &norm, &block.q_w_t, None, &mut q_proj)?;
        // K = norm @ k_w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(t, kv_hidden, d, &norm, &block.k_w_t, None, &mut k_proj)?;
        // V = norm @ v_w_t: [t, d] × [d, kv_hidden] → [t, kv_hidden]
        compute.gemm_f32(t, kv_hidden, d, &norm, &block.v_w_t, None, &mut v_proj)?;

        // Apply RoPE per-head to Q and K.
        for h_q in 0..n_head_q {
            for i in 0..t {
                let src = &q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                dst.copy_from_slice(&rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }
        for h_kv in 0..n_head_kv {
            for i in 0..t {
                let src =
                    &k_proj[i * kv_hidden + h_kv * head_dim..i * kv_hidden + (h_kv + 1) * head_dim];
                rope_scratch[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
            }
            rope_apply(
                &mut rope_scratch[..t * head_dim],
                t,
                head_dim,
                rope_base,
                position_offset,
            )?;
            for i in 0..t {
                let dst = &mut k_proj
                    [i * kv_hidden + h_kv * head_dim..i * kv_hidden + (h_kv + 1) * head_dim];
                dst.copy_from_slice(&rope_scratch[i * head_dim..(i + 1) * head_dim]);
            }
        }

        // Append K/V to cache.
        kv_cache.append(
            layer_idx,
            &k_proj[..t * kv_hidden],
            &v_proj[..t * kv_hidden],
        );
        let t_kv = position_offset + t;
        let k_cache = kv_cache.k(layer_idx);
        let v_cache = kv_cache.v(layer_idx);

        // Attention: for each Q head h_q, use K/V head h_kv = h_q / n_kv_groups.
        let mut scores = vec![0.0f32; t * t_kv];
        let mut probs = vec![0.0f32; t * t_kv];
        for h_q in 0..n_head_q {
            let h_kv = h_q / n_kv_groups;
            for i in 0..t {
                let q_row = &q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                let row_start = i * t_kv;
                for j in 0..t_kv {
                    let k_row = &k_cache
                        [j * kv_hidden + h_kv * head_dim..j * kv_hidden + (h_kv + 1) * head_dim];
                    let mut s = 0.0f32;
                    for c in 0..head_dim {
                        s += q_row[c] * k_row[c];
                    }
                    scores[row_start + j] = s * scale;
                }
                // Causal mask: row i's absolute position is position_offset + i,
                // so keys at j > position_offset + i are masked out.
                let cur_pos = position_offset + i;
                for j in (cur_pos + 1)..t_kv {
                    scores[row_start + j] = f32::NEG_INFINITY;
                }
            }
            // Row-wise softmax.
            compute.softmax_f32(&scores, &mut probs, t, t_kv)?;
            // head_out[i, c] = Σ_j probs[i, j] * V[j, h_kv*head_dim + c]
            for i in 0..t {
                let out_dst = &mut attn_out[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                for c in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..t_kv {
                        let v_row = &v_cache[j * kv_hidden + h_kv * head_dim
                            ..j * kv_hidden + (h_kv + 1) * head_dim];
                        sum += probs[i * t_kv + j] * v_row[c];
                    }
                    out_dst[c] = sum;
                }
            }
        }

        // O projection: attn_out @ o_w_t: [t, d] × [d, d] → [t, d]
        compute.gemm_f32(t, d, d, &attn_out, &block.o_w_t, None, &mut attn_o)?;
        // Residual add.
        for i in 0..t * d {
            h[i] += attn_o[i];
        }

        // ---------- Pre-norm SwiGLU FFN ----------
        rms_norm(&h, &block.ffn_norm_gamma, eps, t, &mut norm)?;
        // gate = norm @ gate_w_t → [t, ffn]
        compute.gemm_f32(t, ffn, d, &norm, &block.ffn_gate_w_t, None, &mut ffn_gate)?;
        // up = norm @ up_w_t → [t, ffn]
        compute.gemm_f32(t, ffn, d, &norm, &block.ffn_up_w_t, None, &mut ffn_up)?;
        // silu(gate) * up
        silu_inplace(&mut ffn_gate);
        hadamard_inplace(&mut ffn_gate, &ffn_up)?;
        // down = (silu(gate) * up) @ down_w_t → [t, d]
        compute.gemm_f32(
            t,
            d,
            ffn,
            &ffn_gate,
            &block.ffn_down_w_t,
            None,
            &mut ffn_down,
        )?;
        // Residual add.
        for i in 0..t * d {
            h[i] += ffn_down[i];
        }
    }
    // Advance the KV cache position clock (all layers appended `t` rows).
    kv_cache.advance(t);

    // Final RMSNorm.
    rms_norm(&h, &weights.final_norm_gamma, eps, t, &mut norm)?;

    // Tied logits head: logits[t, vocab] = norm[t, d] × token_emb.T[d, vocab].
    // token_emb is stored as [vocab, d] row-major; use gemv per row.
    let mut logits = vec![0.0f32; t * vocab];
    for i in 0..t {
        let x = &norm[i * d..(i + 1) * d];
        let out = &mut logits[i * vocab..(i + 1) * vocab];
        compute.gemv_f32(vocab, d, &weights.token_emb, x, None, out)?;
    }
    Ok(logits)
}

/// Argmax of a f32 slice. Ties resolved by lowest index.
fn argmax_u32(row: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

// -----------------------------------------------------------------------------
// Parity harness (T09 synthesized-fixture + T21+ real HF wire)
// -----------------------------------------------------------------------------

/// The synthesized-fixture + real-checkpoint parity harness for the LLM
/// backbone.
///
/// # Two levels
///
/// - **[`parity::forward_matches_step_by_step`]** — the deterministic
///   property test that runs today: builds a synthesized backbone, runs
///   `forward` over a prefix, then reruns the same prefix through `step`
///   one token at a time, and checks that the last-position logits
///   agree up to a tight tolerance (KV-cache consistency across bulk vs
///   incremental).
/// - **[`parity::assert_vs_hf_reference`]** — the "flip-the-switch" real
///   checkpoint harness owners run when the T02 tensor manifest arrives.
///   Returns [`VokraError::NotImplemented`] today.
pub mod parity {
    use super::{LlmBackbone, LlmBackboneConfig, LlmBackboneStep};
    use vokra_core::{Result, VokraError};

    /// Runs `forward([tokens[..n]])` and `step`-per-token over the same
    /// prefix, then checks the last-row logits agree to `atol`.
    ///
    /// This is the **numerical consistency** check between the bulk
    /// `forward` and the incremental `step` — the KV cache must be
    /// bit-comparable across the two paths (a tiny f32 drift is
    /// expected from GEMM associativity, hence `atol > 0`; 1e-3 is a
    /// tight-but-realistic bound).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `tokens` is empty.
    /// - Propagates forward / step errors verbatim.
    pub fn forward_matches_step_by_step(
        backbone: &LlmBackbone,
        tokens: &[u32],
        atol: f32,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 LLM parity: tokens must be non-empty".into(),
            ));
        }
        let cfg = backbone.config();
        let vocab = cfg.vocab_size;

        // Path A: bulk forward.
        let bulk = backbone.forward(tokens, 0)?;
        let bulk_last = &bulk[bulk.len() - vocab..];

        // Path B: per-token step.
        let mut state = LlmBackboneStep::new();
        let mut step_last: Vec<f32> = Vec::new();
        for &tok in tokens {
            step_last = backbone.step(&mut state, tok)?;
        }

        // Compare last-row logits.
        if step_last.len() != bulk_last.len() {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM parity: step last-row len {} != bulk last-row len {}",
                step_last.len(),
                bulk_last.len(),
            )));
        }
        let mut max_delta = 0.0f32;
        for (b, s) in bulk_last.iter().zip(step_last.iter()) {
            let delta = (b - s).abs();
            if delta > max_delta {
                max_delta = delta;
            }
        }
        if max_delta > atol {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM parity: forward vs step-by-step last-row delta {max_delta} > \
                 atol {atol}",
            )));
        }
        Ok(())
    }

    /// **Real-checkpoint parity vs the HuggingFace reference.**
    ///
    /// Owners run this against `iic/CosyVoice2-0.5B` once the T02 tensor
    /// manifest lands. Today it returns [`VokraError::NotImplemented`] —
    /// the honest signal that the real fixture is not wired.
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] until T02 lands.
    pub fn assert_vs_hf_reference(
        _config: &LlmBackboneConfig,
        _reference_logits: &[f32],
    ) -> Result<()> {
        Err(VokraError::NotImplemented(
            "CosyVoice2 LLM real-checkpoint parity is deferred to T02 upstream \
             inspection + T07 tensor-store scan; today the real HF weights are \
             not committed to the workspace and the runtime refuses to fabricate \
             a pass (FR-EX-08).",
        ))
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;

    /// Deterministic non-zero config so shape validators exercise real math
    /// (n_head_q=4, n_head_kv=2, hidden_dim=16 → head_dim=4). Non-zero
    /// n_ctx is documented so a step past it fails loudly.
    fn seed_config(b: &mut GgufBuilder) {
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 32);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 16);
        b.add_u32(super::super::config::KEY_N_LAYER, 2);
        b.add_u32(super::super::config::KEY_N_HEAD, 4);
        b.add_u32(super::super::config::KEY_FFN_DIM, 32);
        b.add_u32(super::super::config::KEY_FLOW_NFE, 4);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, 4);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, 16);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, 8);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 4);
    }

    fn parse_config(bytes: Vec<u8>) -> (GgufFile, CosyVoice2Config) {
        let file = GgufFile::parse(bytes).expect("parse");
        let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
        (file, cfg)
    }

    /// A canonical LlmBackboneConfig for the tests — well-formed GQA + non-zero
    /// n_ctx so `step` and `forward` can exercise the causal mask and the
    /// KV cache append path.
    fn test_config() -> LlmBackboneConfig {
        LlmBackboneConfig {
            vocab_size: 16,
            hidden_dim: 8,
            n_layer: 2,
            n_head_q: 2,
            n_head_kv: 1,
            ffn_dim: 16,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
            n_ctx: 8,
        }
    }

    // ---- GGUF metadata read tests --------------------------------------

    #[test]
    fn llm_config_defaults_populate_when_keys_absent() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).expect("read");
        assert_eq!(llm.n_head_kv, llm.n_head_q);
        assert!((llm.rope_base - DEFAULT_ROPE_BASE_QWEN2).abs() < 1e-3);
        assert!((llm.rms_norm_eps - DEFAULT_RMS_NORM_EPS).abs() < 1e-9);
        assert_eq!(llm.n_ctx, 0);
    }

    #[test]
    fn llm_config_reads_present_keys_verbatim() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        b.add_f32(KEY_LLM_ROPE_BASE, 500_000.0);
        b.add_f32(KEY_LLM_RMS_NORM_EPS, 1e-6);
        b.add_u32(KEY_LLM_N_CTX, 2048);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).expect("read");
        assert_eq!(llm.n_head_kv, 2);
        assert!((llm.rope_base - 500_000.0).abs() < 1e-1);
        assert!((llm.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert_eq!(llm.n_ctx, 2048);
    }

    #[test]
    fn llm_config_wrong_type_fails_loudly() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_metadata(
            KEY_LLM_N_HEAD_KV,
            GgufMetadataValue::String("nope".to_owned()),
        );
        let file = GgufFile::parse(b.to_bytes().unwrap()).expect("parse");
        let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
        let err = LlmBackboneConfig::from_gguf(&file, &cfg).expect_err("wrong type must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn head_dim_and_kv_hidden_dim_derive_from_shape() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).unwrap();
        assert_eq!(llm.head_dim(), 4);
        assert_eq!(llm.kv_hidden_dim(), 8);
    }

    #[test]
    fn head_dim_returns_zero_on_zero_n_head_q() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_metadata(super::super::config::KEY_N_HEAD, GgufMetadataValue::U32(0));
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).unwrap();
        assert_eq!(llm.head_dim(), 0);
        assert_eq!(llm.kv_hidden_dim(), 0);
        assert!(!llm.is_gqa_well_formed(), "zero heads → not well-formed");
    }

    #[test]
    fn gqa_well_formed_requires_head_split_and_hidden_divisibility() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        assert!(
            LlmBackboneConfig::from_gguf(&file, &cfg)
                .unwrap()
                .is_gqa_well_formed()
        );
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 3);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        assert!(
            !LlmBackboneConfig::from_gguf(&file, &cfg)
                .unwrap()
                .is_gqa_well_formed()
        );
    }

    // ---- from_gguf (top-level constructor) tests ------------------------

    #[test]
    fn from_gguf_rejects_zero_placeholder_shape_config() {
        // The scaffold seed_config leaves the LLM-side keys (n_head_kv,
        // rope_base, ...) as defaults but the CosyVoice2Config n_layer=2,
        // n_head=4, vocab=32, etc. from seed_config produce a well-formed
        // config, so the from_gguf path succeeds. Explicitly zero it to
        // exercise the reject path.
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 0);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 0);
        b.add_u32(super::super::config::KEY_N_LAYER, 0);
        b.add_u32(super::super::config::KEY_N_HEAD, 0);
        b.add_u32(super::super::config::KEY_FFN_DIM, 0);
        b.add_u32(super::super::config::KEY_FLOW_NFE, 0);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, 0);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, 0);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, 0);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 0);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 0);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let err = LlmBackbone::from_gguf(&file, &cfg).expect_err("0-placeholder must be rejected");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn from_gguf_with_weights_returns_not_implemented_until_t02_lands() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let err = LlmBackbone::from_gguf_with_weights(&file, &cfg)
            .expect_err("real weight binding is T02");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn from_gguf_produces_working_synthesized_backbone() {
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        b.add_u32(KEY_LLM_N_CTX, 8);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let backbone = LlmBackbone::from_gguf(&file, &cfg).expect("synthesized build");
        assert!(backbone.weights().is_synthesized);
        // A trivial forward runs.
        let logits = backbone.forward(&[0, 1, 2], 0).expect("forward runs");
        assert_eq!(logits.len(), 3 * 32);
        for &l in &logits {
            assert!(l.is_finite(), "logit must be finite");
        }
    }

    // ---- synthesized weight tests --------------------------------------

    #[test]
    fn synthesized_weights_have_correct_shapes() {
        let cfg = test_config();
        let w = LlmWeights::synthesized(&cfg, 42).unwrap();
        let d = cfg.hidden_dim;
        let kv_hidden = cfg.kv_hidden_dim();
        let ffn = cfg.ffn_dim;
        let vocab = cfg.vocab_size;
        assert_eq!(w.token_emb.len(), vocab * d);
        assert_eq!(w.blocks.len(), cfg.n_layer);
        assert_eq!(w.final_norm_gamma.len(), d);
        for b in &w.blocks {
            assert_eq!(b.attn_norm_gamma.len(), d);
            assert_eq!(b.q_w_t.len(), d * d);
            assert_eq!(b.k_w_t.len(), d * kv_hidden);
            assert_eq!(b.v_w_t.len(), d * kv_hidden);
            assert_eq!(b.o_w_t.len(), d * d);
            assert_eq!(b.ffn_norm_gamma.len(), d);
            assert_eq!(b.ffn_gate_w_t.len(), d * ffn);
            assert_eq!(b.ffn_up_w_t.len(), d * ffn);
            assert_eq!(b.ffn_down_w_t.len(), ffn * d);
        }
        assert!(w.is_synthesized);
    }

    #[test]
    fn synthesized_weights_are_deterministic_across_seeds() {
        let cfg = test_config();
        let a = LlmWeights::synthesized(&cfg, 42).unwrap();
        let b = LlmWeights::synthesized(&cfg, 42).unwrap();
        let c = LlmWeights::synthesized(&cfg, 43).unwrap();
        assert_eq!(a.token_emb, b.token_emb, "same seed → identical weights");
        assert_ne!(
            a.token_emb, c.token_emb,
            "different seeds → different weights (probabilistic)"
        );
    }

    #[test]
    fn synthesized_weights_reject_zero_placeholder_config() {
        let mut cfg = test_config();
        cfg.vocab_size = 0;
        assert!(matches!(
            LlmWeights::synthesized(&cfg, 42),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn from_gguf_weights_returns_not_implemented() {
        let cfg = test_config();
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        let bytes = b.to_bytes().unwrap();
        let file = GgufFile::parse(bytes).unwrap();
        let err = LlmWeights::from_gguf(&file, &cfg).expect_err("T02 not landed");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    // ---- primitive re-export tests --------------------------------------

    #[test]
    fn rms_norm_reexport_operates_on_row() {
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 0.0, 1, &mut out).expect("re-exported primitive works");
        let mean_sq: f32 = out.iter().map(|v| v * v).sum::<f32>() / d as f32;
        assert!(
            (mean_sq - 1.0).abs() < 1e-5,
            "rms_norm re-export must normalise to unit RMS"
        );
    }

    #[test]
    fn rms_norm_post_norm_rms_is_unit_within_atol() {
        // Property test: for any row with gamma=1, post-norm RMS ≈ 1 (up to
        // f32 precision + epsilon).
        let d = 8;
        let x: Vec<f32> = (0..d).map(|i| (i as f32) * 0.5 + 0.1).collect();
        let gamma = vec![1.0f32; d];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 1e-5, 1, &mut out).unwrap();
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-4,
            "post-norm RMS should be 1.0, got {rms}"
        );
    }

    #[test]
    fn rms_norm_gamma_scales_output() {
        // Property test: gamma=2 → post-norm RMS ≈ 2 (RMSNorm is scale-only,
        // and gamma scales the output linearly).
        let d = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![2.0f32; d];
        let mut out = vec![0.0f32; d];
        rms_norm(&x, &gamma, 1e-5, 1, &mut out).unwrap();
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
        assert!(
            (rms - 2.0).abs() < 1e-4,
            "gamma=2 → post-norm RMS should be 2.0, got {rms}"
        );
    }

    #[test]
    fn silu_reexport_saturates_positive_asymptote() {
        // silu(∞) → x (asymptotic): for x=50, silu(50) ≈ 50 (sigmoid(50) ≈ 1).
        let mut x = vec![50.0f32];
        silu_inplace(&mut x);
        assert!(
            (x[0] - 50.0).abs() < 1e-3,
            "silu re-export must saturate for large positive x"
        );
    }

    #[test]
    fn silu_reexport_is_zero_at_zero() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.
        let mut x = vec![0.0f32];
        silu_inplace(&mut x);
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn rope_reexport_position_zero_is_identity() {
        // apply_rotary(q, 0) == q.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_apply(&mut x, 1, 4, DEFAULT_ROPE_BASE_QWEN2, 0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_reexport_preserves_vector_norm() {
        // RoPE is a rotation → ||x||₂ preserved (cos^2 + sin^2 = 1 in the
        // per-pair rotation).
        let d = 8;
        let mut x: Vec<f32> = (0..d).map(|i| (i as f32) + 0.5).collect();
        let orig_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope_apply(&mut x, 1, d, 10_000.0, 3).unwrap();
        let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((orig_norm - new_norm).abs() < 1e-4);
    }

    #[test]
    fn rope_reexport_is_invertible() {
        // Apply RoPE at position m, then apply the inverse rotation
        // (flip the sign of sin) — should return to identity.
        let d = 4;
        let head_dim = d;
        let m = 5;
        let base = 10_000.0f32;
        let mut x = vec![0.3f32, -0.7, 1.1, 0.5];
        let orig = x.clone();
        rope_apply(&mut x, 1, head_dim, base, m).unwrap();
        // Undo: apply the transpose rotation manually — negate sin.
        let half = head_dim / 2;
        for j in 0..half {
            let theta = base.powf(-2.0 * (j as f32) / (head_dim as f32));
            let angle = m as f32 * theta;
            let (s, c) = angle.sin_cos();
            let a = x[j];
            let b = x[j + half];
            // Inverse rotation: [cos, sin; -sin, cos].
            x[j] = a * c + b * s;
            x[j + half] = -a * s + b * c;
        }
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "RoPE not invertible: got {a}, want {b}"
            );
        }
    }

    #[test]
    fn hadamard_reexport_multiplies_elementwise_for_swiglu_body() {
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        hadamard_inplace(&mut a, &b).expect("re-exported primitive works");
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }

    // ---- GQA head split test --------------------------------------------

    #[test]
    fn gqa_head_repeat_pattern_is_correct() {
        // Property: for n_head_q=4, n_head_kv=2 → n_kv_groups=2. Head 0 and
        // head 1 (Q) map to kv head 0; head 2, head 3 → kv head 1.
        let cfg = LlmBackboneConfig {
            vocab_size: 8,
            hidden_dim: 8,
            n_layer: 1,
            n_head_q: 4,
            n_head_kv: 2,
            ffn_dim: 8,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
            n_ctx: 4,
        };
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        // Two-token forward exercises the causal mask + GQA broadcast.
        let logits = backbone.forward(&[0, 1], 0).unwrap();
        assert_eq!(logits.len(), 2 * 8);
        for &l in &logits {
            assert!(l.is_finite());
        }
    }

    // ---- LlmBackbone: forward tests -------------------------------------

    #[test]
    fn forward_shape_is_t_by_vocab() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg.clone(), 7).unwrap();
        let logits = backbone.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
    }

    #[test]
    fn forward_is_finite_across_layer_stack() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 3).unwrap();
        let logits = backbone.forward(&[0, 1, 2, 3, 4], 0).unwrap();
        for &l in &logits {
            assert!(
                l.is_finite(),
                "logit must be finite (no NaN / Inf from layer stack)"
            );
        }
    }

    #[test]
    fn forward_empty_tokens_yields_empty_logits() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 1).unwrap();
        let logits = backbone.forward(&[], 0).unwrap();
        assert!(logits.is_empty());
    }

    #[test]
    fn forward_rejects_token_id_out_of_range() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 1).unwrap();
        let err = backbone.forward(&[16, 0, 1], 0).expect_err("oor token");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_rejects_position_past_n_ctx() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 1).unwrap();
        // n_ctx=8; positions 5..12 = 7 tokens starting at offset 5 → past
        // n_ctx.
        let err = backbone.forward(&[0; 7], 5).expect_err("past n_ctx");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn forward_is_deterministic() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        let a = backbone.forward(&[1, 2, 3], 0).unwrap();
        let b = backbone.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(a, b, "same input + same weights → identical logits");
    }

    #[test]
    fn forward_output_range_is_reasonable() {
        // Property: with synthesized weights the logits should not explode.
        // Xavier init keeps activations bounded; a well-formed forward's
        // logits stay within roughly [-1e3, +1e3] for small models.
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        let logits = backbone.forward(&[1, 2, 3], 0).unwrap();
        for &l in &logits {
            assert!(l.abs() < 1e3, "logit magnitude {l} too large (overflow?)");
        }
    }

    // ---- LlmBackbone: step tests ----------------------------------------

    #[test]
    fn step_shape_is_vocab() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg.clone(), 7).unwrap();
        let mut state = LlmBackboneStep::new();
        let logits = backbone.step(&mut state, 0).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    #[test]
    fn step_advances_seq_len() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 7).unwrap();
        let mut state = LlmBackboneStep::new();
        assert_eq!(state.seq_len, 0);
        let _ = backbone.step(&mut state, 0).unwrap();
        assert_eq!(state.seq_len, 1);
        let _ = backbone.step(&mut state, 1).unwrap();
        assert_eq!(state.seq_len, 2);
    }

    #[test]
    fn step_kv_cache_lazy_allocated_and_grows() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 7).unwrap();
        let mut state = LlmBackboneStep::new();
        assert!(
            state.kv_cache.is_none(),
            "cache lazy-allocated on first step"
        );
        let _ = backbone.step(&mut state, 0).unwrap();
        assert!(state.kv_cache.is_some());
        let positions_after_one = state.kv_cache.as_ref().unwrap().positions();
        assert_eq!(positions_after_one, 1);
        let _ = backbone.step(&mut state, 1).unwrap();
        assert_eq!(state.kv_cache.as_ref().unwrap().positions(), 2);
    }

    #[test]
    fn step_rejects_position_past_n_ctx() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 7).unwrap();
        let mut state = LlmBackboneStep::new();
        // Fill up to n_ctx=8.
        for i in 0..8 {
            let _ = backbone.step(&mut state, (i as u32) % 16).unwrap();
        }
        // The 9th step is past n_ctx.
        let err = backbone.step(&mut state, 0).expect_err("past n_ctx");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn step_reset_clears_seq_len_and_cache_positions() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 7).unwrap();
        let mut state = LlmBackboneStep::new();
        let _ = backbone.step(&mut state, 0).unwrap();
        let _ = backbone.step(&mut state, 1).unwrap();
        state.reset();
        assert_eq!(state.seq_len, 0);
        assert_eq!(state.kv_cache.as_ref().unwrap().positions(), 0);
    }

    // ---- LlmBackbone: greedy_decode tests --------------------------------

    #[test]
    fn greedy_decode_stops_at_max_new_tokens() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        // Use a token id that likely never argmaxes (u32::MAX far outside
        // vocab_size=16). We want the loop to run max_new times.
        let out = backbone.greedy_decode(&[0, 1], u32::MAX, 3).unwrap();
        assert_eq!(out.len(), 3);
        for &tok in &out {
            assert!(tok < 16, "generated token must be in vocab range");
        }
    }

    #[test]
    fn greedy_decode_stops_at_eos() {
        // Construct a synthetic backbone whose first step's argmax we can
        // pin: use the deterministic init and read the first argmax off a
        // dry run, then set that as eos.
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg.clone(), 42).unwrap();
        // Dry run to learn the first sampled token.
        let dry = backbone.greedy_decode(&[0, 1], u32::MAX, 1).unwrap();
        assert_eq!(dry.len(), 1);
        let expected_first_tok = dry[0];
        // Now set eos = expected_first_tok. The loop should sample it and
        // stop.
        let out = backbone
            .greedy_decode(&[0, 1], expected_first_tok, 5)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], expected_first_tok);
    }

    #[test]
    fn greedy_decode_stops_before_n_ctx() {
        // n_ctx=8; the loop breaks when `state.seq_len >= n_ctx` (guard
        // consulted *after* pushing the newly sampled token but *before*
        // the next `step()`). With prefix `[0, 1]` (2 tokens) the state
        // seq_len is 2 after the prefix consumption; each iteration then
        // pushes one new token and advances seq_len by 1 through the
        // next `step()`. Concretely: pushes happen at seq_len =
        // 2, 3, 4, 5, 6, 7, 8; the last push (seq_len == 8) satisfies
        // the guard and breaks — so the loop commits **7** new tokens
        // total, bounded by `n_ctx - prefix_len + 1`.
        //
        // Assert that (a) at least 1 token is generated and (b) at most
        // `n_ctx - prefix_len + 1 = 7` — the n_ctx guard fires.
        let cfg = test_config(); // n_ctx=8, vocab=16
        let n_ctx = cfg.n_ctx;
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        let prefix = [0u32, 1];
        let out = backbone.greedy_decode(&prefix, u32::MAX, 20).unwrap();
        let upper = n_ctx - prefix.len() + 1;
        assert!(
            !out.is_empty() && out.len() <= upper,
            "n_ctx-bounded decode produced {} tokens (bounds: [1, {}])",
            out.len(),
            upper,
        );
    }

    #[test]
    fn greedy_decode_rejects_empty_initial() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        let err = backbone
            .greedy_decode(&[], 0, 10)
            .expect_err("empty prefix");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn greedy_decode_rejects_out_of_range_prefix() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        let err = backbone
            .greedy_decode(&[999], 0, 10)
            .expect_err("oor prefix");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- LlmBackbone: parity harness tests -------------------------------

    #[test]
    fn parity_forward_matches_step_by_step_within_atol() {
        let cfg = test_config();
        let backbone = LlmBackbone::synthesized(cfg, 42).unwrap();
        parity::forward_matches_step_by_step(&backbone, &[0, 1, 2, 3], 1e-3)
            .expect("forward vs step-by-step consistency");
    }

    #[test]
    fn parity_forward_matches_step_by_step_across_seeds() {
        let cfg = test_config();
        for seed in [1u64, 42, 100, 12345] {
            let backbone = LlmBackbone::synthesized(cfg.clone(), seed).unwrap();
            parity::forward_matches_step_by_step(&backbone, &[0, 1, 2], 1e-3)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        }
    }

    #[test]
    fn parity_hf_reference_returns_not_implemented_today() {
        let cfg = test_config();
        let err = parity::assert_vs_hf_reference(&cfg, &[]).expect_err("HF fixture not wired");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    // ---- LlmBackboneStep API tests --------------------------------------

    #[test]
    fn llm_backbone_step_advance_increments_counter() {
        let mut s = LlmBackboneStep::new();
        assert_eq!(s.seq_len, 0);
        s.advance();
        s.advance();
        assert_eq!(s.seq_len, 2);
    }
}
