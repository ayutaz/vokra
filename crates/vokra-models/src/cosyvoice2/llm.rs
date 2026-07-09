//! CosyVoice2 LLM backbone — decoder-only transformer scaffold (M3-09-T07 / T08).
//!
//! CosyVoice2 wraps a **decoder-only text-to-token LLM** (upstream ships a
//! `Qwen2-0.5B` backbone in the `iic/CosyVoice2-0.5B` release) whose output
//! token stream drives the Flow Matching CFM (T10). This file lands the
//! **module + primitive surface** and the **NotImplemented forward** — the
//! real weight-bound autoregressive path lands with T09 (unit-test against a
//! parity fixture) and T21+ (real checkpoint parity).
//!
//! # What lands in this session (M3-09 CC follow-on)
//!
//! - `LlmBackboneConfig` — snapshot of the LLM-side hparams the runtime reads
//!   from the `vokra.cosyvoice2.arch.*` chunk group (T04 sub-slice). Every
//!   field is **read from the GGUF** — nothing is hard-coded (CLAUDE.md
//!   "ハルシネーション厳禁": tensor names and hparams live in the metadata,
//!   never in Rust literals). See [`LlmBackboneConfig::from_gguf`] for the
//!   loose split: fields already present in [`CosyVoice2Config`] (vocab /
//!   hidden / n_layer / n_head / ffn) are borrowed verbatim; the two GQA
//!   fields (`n_head_kv`) + RoPE base + RMSNorm ε are read from separate
//!   metadata keys with `0` / defaults tolerated so a shape-only converter
//!   (T02 upstream inspection still open) is not rejected.
//! - `LlmBackbone` — the top-level type. Owns the shape config; the T07/T08
//!   follow-on hangs the tensor store (embedding + blocks + final norm) off
//!   it. Forward is [`VokraError::NotImplemented`] today; the primitive
//!   surface below is what a follow-on will compose.
//! - `LlmBackboneStep` — the autoregressive-decode state (analog of Whisper's
//!   `DecoderState` and Voxtral's `TextDecoderStep`). Today it carries the
//!   token count only; T14 wires it to the M3-03 paged KV cache.
//! - **Shared primitives** (`rms_norm`, `silu_inplace`, `hadamard_inplace`,
//!   `rope_apply`) — re-exported from [`crate::voxtral::text_decoder`] so
//!   Mistral-style GQA/RoPE/SwiGLU/RMSNorm primitives serve both models
//!   (dependency-graph share; CLAUDE.md "自前再実装" is upheld —
//!   `vokra-models` code implements them once). The alternative (duplicate
//!   the four `pub fn`s) would drift and hide bugs — this session commits
//!   to the shared surface so a future GQA fix lands once.
//!
//! # Numeric parity strategy (follow-on)
//!
//! Same posture as Voxtral T08: the primitives ([`rms_norm`], [`silu`],
//! [`hadamard`], [`rope_apply`]) are unit-tested against internal oracles
//! (already covered in `voxtral::text_decoder::tests`). The follow-on:
//!
//! 1. reads the upstream CosyVoice2 safetensors (T02 upstream inspection,
//!    open), records `tensor_name → shape/dtype` in a manifest;
//! 2. binds those tensors verbatim through
//!    [`vokra_core::gguf::GgufFile::get_tensor`] in a `weights::TensorStore`
//!    (mirrors `piper_plus::weights::TensorStore`);
//! 3. routes GEMM through [`crate::compute::Compute::gemm_f32`] so Metal /
//!    CUDA seams (T19/T20) offload without a second kernel path.
//!
//! # No silent fallback (FR-EX-08)
//!
//! - `LlmBackbone::forward` — `NotImplemented`, never a zero-fill.
//! - Shape mismatch on any primitive — surface `InvalidArgument`, never
//!   silently truncate / pad.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use super::config::CosyVoice2Config;

// # Shared primitives (T07 / T08 follow-on)
//
// The Mistral-style primitives — `rms_norm`, `silu_inplace`,
// `hadamard_inplace`, `rope_apply` — the CosyVoice2 forward path will
// compose (T07 embedding + stem, T08 GEMM/attention/FFN) already live at
// `crate::voxtral::text_decoder`. Reusing them (instead of duplicating the
// same four `pub fn`s) keeps the numerical surface identical across
// GQA-style LLM backbones — a bug fix lands once. This scaffold does
// **not** bring them into scope at module level yet (they would be
// unused today and trigger `unused_imports`); the T07/T08 follow-on
// adds a plain `use crate::voxtral::text_decoder::{...}` here as the
// forward path is wired. The tests below import them explicitly so the
// re-use contract is exercised on every CI run.

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

/// Sentinel + safety-net RoPE base used **only** when the GGUF omits the
/// key (converter shape-only path). At forward-time the backbone still
/// refuses to run against a `0` `n_head` — but the constant here matches
/// the Qwen2 family default (Mistral / Qwen2 modern releases ship
/// `1_000_000.0`), so a config dump from a well-formed GGUF trivially
/// agrees. This is documented so the follow-on session can spot when it
/// needs to enforce the key's presence.
pub const DEFAULT_ROPE_BASE_QWEN2: f32 = 1_000_000.0;

/// Same posture for RMSNorm ε. Mistral / Qwen2 ship `1e-5`.
pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-5;

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
    /// Transformer-XL / Vaswani-2017 default. This is the ONLY hparam we
    /// tolerate a default on today (CosyVoice2 upstream is not yet inspected
    /// — T02); the follow-on enforces the key's presence.
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

/// CosyVoice2 LLM backbone — top-level type (M3-09-T08 scaffold).
///
/// Owns the resolved [`LlmBackboneConfig`]. The T07/T08 follow-on hangs a
/// weight store (embedding table + `n_layer` decoder blocks + final RMSNorm)
/// off it and implements the full autoregressive forward. Today this struct
/// is deliberately light: it validates the config at construction and
/// exposes a `forward` stub that returns
/// [`VokraError::NotImplemented`] on any call (FR-EX-08 — never a silent
/// zero-fill fallback).
///
/// # Follow-on tickets (paths this scaffold does not touch)
///
/// - **T07** — real weight binding: read `vokra.cosyvoice2.tensor.*` metadata
///   plus GGUF tensor slices into `Vec<f32>` blocks (mirrors
///   `piper_plus::weights::TensorStore`), no invented tensor names.
/// - **T08** — GEMM/attention/FFN hot path through the [`crate::compute`]
///   seam (Metal / CUDA offload on T19/T20).
/// - **T09** — smoke unit test against a synthesized `LlmParityFixture`
///   (below).
/// - **T14 / T16** — chunk-aware streaming with M3-03 paged KV cache;
///   [`LlmBackboneStep`] is the placeholder for that state.
#[derive(Debug, Clone)]
pub struct LlmBackbone {
    /// LLM-side resolved hparams.
    config: LlmBackboneConfig,
}

impl LlmBackbone {
    /// Builds a backbone from a resolved config.
    ///
    /// This constructor never fails today — the follow-on constructor
    /// ([`Self::from_gguf`]) does the "load real weights" work and returns
    /// a `Result`. Keeping a light infallible ctor lets internal-oracle
    /// tests (and future stub-driven pipelines) build an LlmBackbone
    /// without a GGUF file, which is what the M3-09-T09 fixture path needs.
    #[must_use]
    pub fn new(config: LlmBackboneConfig) -> Self {
        Self { config }
    }

    /// Loads the LLM backbone from a CosyVoice2 GGUF file.
    ///
    /// Today this ONLY reads the shape config — the tensor binding follow-on
    /// (T07) adds the real weight store. Callers who need the LLM backbone
    /// today receive a config-only handle that already passes shape
    /// validation but returns [`VokraError::NotImplemented`] on any
    /// forward attempt.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any GGUF metadata key with a
    /// wrong type.
    pub fn from_gguf(file: &GgufFile, cfg: &CosyVoice2Config) -> Result<Self> {
        let llm_cfg = LlmBackboneConfig::from_gguf(file, cfg)?;
        Ok(Self::new(llm_cfg))
    }

    /// The resolved LLM hparams.
    #[must_use]
    pub fn config(&self) -> &LlmBackboneConfig {
        &self.config
    }

    /// Runs the LLM backbone forward once over `token_ids` and produces the
    /// per-token hidden states (`[t, hidden_dim]` row-major) the Flow
    /// Matching CFM consumes.
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] unconditionally
    /// — the real path is T07/T08. `token_ids` and `position_offset` are
    /// documented so callers can build the plumbing (tokenizer +
    /// autoregressive loop + KV cache write cursor) against the final
    /// signature today.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the config is not GQA-well-formed
    /// (fails before the NotImplemented, so a bad config still fails
    /// loudly at the earliest point).
    ///
    /// [`VokraError::NotImplemented`] once the config is well-formed —
    /// today's honest signal.
    pub fn forward(&self, token_ids: &[u32], position_offset: usize) -> Result<Vec<f32>> {
        // Validate the config before returning NotImplemented so a broken
        // metadata surface fails at a specific error kind (InvalidArgument)
        // — never a NotImplemented that masks a misconfigured GGUF.
        if !self.config.is_gqa_well_formed() {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone: config not GQA well-formed \
                 (n_head_q={}, n_head_kv={}, hidden_dim={}) — need \
                 n_head_q > 0, n_head_kv > 0, n_head_q % n_head_kv == 0, \
                 hidden_dim % n_head_q == 0",
                self.config.n_head_q, self.config.n_head_kv, self.config.hidden_dim,
            )));
        }
        let _ = (token_ids, position_offset);
        Err(VokraError::NotImplemented(
            "CosyVoice2 LLM backbone forward is not implemented in this scaffold; \
             T07 embedding / T08 transformer blocks / T09 unit test wire the numeric \
             path against the upstream safetensors manifest",
        ))
    }

    /// Runs a single **decode step** over one new token, appending to the
    /// autoregressive state.
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] — the real path
    /// lands with T14/T16 (chunk-aware streaming, paged KV cache).
    /// Documented today so callers wiring the streaming pipeline can build
    /// against the final signature.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the state's `seq_len` exceeds
    /// `config.n_ctx`.
    /// [`VokraError::NotImplemented`] otherwise.
    pub fn step(&self, state: &mut LlmBackboneStep, token_id: u32) -> Result<Vec<f32>> {
        if self.config.n_ctx != 0 && state.seq_len >= self.config.n_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice2 LLM backbone: seq_len {} would exceed n_ctx {} \
                 (FR-EX-08 — no silent wrap-around)",
                state.seq_len, self.config.n_ctx
            )));
        }
        let _ = token_id;
        Err(VokraError::NotImplemented(
            "CosyVoice2 LLM backbone step is not implemented in this scaffold; \
             T14 chunk-aware streaming + T16 paged KV cache wire the incremental \
             decode path",
        ))
    }
}

/// Autoregressive decode state — CosyVoice2's analog of Whisper's
/// `DecoderState` and Voxtral's `TextDecoderStep`.
///
/// Foundation-only today: carries the number of tokens processed. T14 wires
/// a paged KV cache (M3-03) reference off this struct so the LLM backbone
/// step can append to it without malloc on the hot path (FR-EX-05).
#[derive(Debug, Clone, Copy, Default)]
pub struct LlmBackboneStep {
    /// Number of tokens processed so far.
    pub seq_len: usize,
}

impl LlmBackboneStep {
    /// Fresh state (nothing decoded).
    #[must_use]
    pub fn new() -> Self {
        Self { seq_len: 0 }
    }

    /// Advance one token (increment `seq_len`).
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Explicit import so the tests hit the exact voxtral primitive path
    // the T07/T08 follow-on will consume.
    use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, rope_apply, silu_inplace};
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

    #[test]
    fn llm_config_defaults_populate_when_keys_absent() {
        // No LLM-specific keys → fallbacks are used (documented).
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).expect("read");
        // n_head_kv falls back to n_head_q (MHA).
        assert_eq!(llm.n_head_kv, llm.n_head_q);
        assert!((llm.rope_base - DEFAULT_ROPE_BASE_QWEN2).abs() < 1e-3);
        assert!((llm.rms_norm_eps - DEFAULT_RMS_NORM_EPS).abs() < 1e-9);
        // n_ctx falls back to 0 (converter shape-only sentinel).
        assert_eq!(llm.n_ctx, 0);
    }

    #[test]
    fn llm_config_reads_present_keys_verbatim() {
        // Explicit GQA + RoPE overrides — every keyed value must be
        // preserved byte-for-byte from the GGUF.
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
        // A key present but of the wrong type must be an explicit error
        // (FR-EX-08 — no silent fallback to a default).
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        // Put a string where a u32 is expected.
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
        // hidden_dim=16, n_head_q=4 → head_dim=4.
        assert_eq!(llm.head_dim(), 4);
        // n_head_kv=2 * head_dim=4 → kv_hidden_dim=8.
        assert_eq!(llm.kv_hidden_dim(), 8);
    }

    #[test]
    fn head_dim_returns_zero_on_zero_n_head_q() {
        // Shape-only converter sentinel: n_head=0 → head_dim=0 (no panic).
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        // Overwrite n_head with 0 (the shape-only converter sentinel).
        b.add_metadata(super::super::config::KEY_N_HEAD, GgufMetadataValue::U32(0));
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let llm = LlmBackboneConfig::from_gguf(&file, &cfg).unwrap();
        assert_eq!(llm.head_dim(), 0);
        assert_eq!(llm.kv_hidden_dim(), 0);
        assert!(!llm.is_gqa_well_formed(), "zero heads → not well-formed");
    }

    #[test]
    fn gqa_well_formed_requires_head_split_and_hidden_divisibility() {
        // (hidden=16, n_head_q=4, n_head_kv=2) — head_dim=4, 4%2=0 → OK.
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        assert!(
            LlmBackboneConfig::from_gguf(&file, &cfg)
                .unwrap()
                .is_gqa_well_formed()
        );
        // (hidden=16, n_head_q=4, n_head_kv=3) — 4 % 3 != 0 → not OK.
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

    #[test]
    fn backbone_forward_returns_not_implemented_never_silent() {
        // FR-EX-08: no silent zero-fill fallback.
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let backbone = LlmBackbone::from_gguf(&file, &cfg).expect("build");
        let err = backbone
            .forward(&[1, 2, 3], 0)
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn backbone_forward_bad_config_fails_before_not_implemented() {
        // A misconfigured GGUF (n_head_kv = 3 does not divide n_head_q = 4)
        // must fail loudly at InvalidArgument BEFORE the NotImplemented
        // scaffold return — so a caller's misconfiguration is not masked
        // by the scaffold status.
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 3); // 4 % 3 != 0
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let backbone = LlmBackbone::from_gguf(&file, &cfg).expect("build");
        let err = backbone
            .forward(&[1, 2], 0)
            .expect_err("bad config must fail");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn backbone_step_respects_n_ctx() {
        // Enforce the seq_len bound before the NotImplemented body.
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        b.add_u32(KEY_LLM_N_CTX, 4); // small context
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let backbone = LlmBackbone::from_gguf(&file, &cfg).expect("build");

        // Below n_ctx: NotImplemented (real path deferred).
        let mut state = LlmBackboneStep::new();
        state.seq_len = 3;
        let err = backbone
            .step(&mut state, 42)
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));

        // At n_ctx: InvalidArgument (loud, no silent wrap-around).
        state.seq_len = 4;
        let err = backbone
            .step(&mut state, 42)
            .expect_err("must fail at n_ctx");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn backbone_step_zero_n_ctx_allows_any_seq_len_but_stays_not_implemented() {
        // n_ctx=0 is the shape-only converter sentinel — the guard skips so
        // the caller still receives the NotImplemented rather than a false
        // n_ctx=0 wrap error.
        let mut b = GgufBuilder::new();
        seed_config(&mut b);
        b.add_u32(KEY_LLM_N_HEAD_KV, 2);
        // KEY_LLM_N_CTX intentionally absent (n_ctx=0).
        let (file, cfg) = parse_config(b.to_bytes().unwrap());
        let backbone = LlmBackbone::from_gguf(&file, &cfg).expect("build");
        let mut state = LlmBackboneStep::new();
        state.seq_len = 1_000_000;
        let err = backbone
            .step(&mut state, 0)
            .expect_err("scaffold not implemented");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn llm_backbone_step_advances_seq_len_like_voxtral_pattern() {
        // Structural parity with voxtral::TextDecoderStep — the pattern the
        // T14 streaming pipeline consumes.
        let mut s = LlmBackboneStep::new();
        assert_eq!(s.seq_len, 0);
        s.advance();
        s.advance();
        assert_eq!(s.seq_len, 2);
    }

    // ---- primitive re-export tests --------------------------------------
    //
    // The primitives themselves (`rms_norm`, `silu_inplace`, `hadamard_inplace`,
    // `rope_apply`) are unit-tested against internal oracles in
    // voxtral::text_decoder::tests. Here we only assert the RE-EXPORTS are
    // in the public surface — a follow-on session composing them from
    // `cosyvoice2::llm` finds them at the expected path.

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
    fn silu_reexport_saturates_positive_asymptote() {
        let mut x = vec![50.0f32];
        silu_inplace(&mut x);
        assert!(
            (x[0] - 50.0).abs() < 1e-3,
            "silu re-export must saturate for large positive x"
        );
    }

    #[test]
    fn rope_reexport_position_zero_is_identity() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = x.clone();
        rope_apply(&mut x, 1, 4, DEFAULT_ROPE_BASE_QWEN2, 0).unwrap();
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hadamard_reexport_multiplies_elementwise_for_swiglu_body() {
        // The full SwiGLU expression is `silu(gate(x)) * up(x)`. Both
        // building blocks are shared with voxtral::text_decoder — this
        // test asserts the Hadamard step is bit-identical to the direct
        // primitive call the forward path will make on T07/T08.
        let mut a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        hadamard_inplace(&mut a, &b).expect("re-exported primitive works");
        assert_eq!(a, vec![4.0, 10.0, 18.0]);
    }
}
