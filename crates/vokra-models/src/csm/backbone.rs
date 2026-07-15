//! CSM backbone — Llama-3.2-flavor decoder-only transformer over 33-slot
//! frames (M4-05-T06 config/weights + T07 forward + T08 paged-KV step).
//!
//! # Architecture (ADR M4-05 §D2 — transcribed, never invented)
//!
//! The backbone is the `llama3_2_1B` flavor of `SesameAILabs/csm`
//! `models.py`: pre-norm RMSNorm blocks, GQA attention with Llama-3
//! **scaled** RoPE ([`super::rope`]), SwiGLU FFN, no biases. One *position*
//! of the sequence is one **frame** — `n_codebooks` audio slots plus one
//! text slot, each slot embedded separately and **summed** over the valid
//! slots (`models.py` `_embed_tokens` + `masked_embeds.sum(dim=2)`), see
//! [`CsmFrame`].
//!
//! Two heads hang off the final RMSNorm hidden state:
//!
//! - `codebook0_head` (`[audio_vocab, d]`) — the zeroth-codebook logits the
//!   backbone samples directly ([`CsmBackbone::c0_logits_into`]);
//! - the depth transformer ([`super::depth`]) — codebook 1.. autoregression
//!   conditioned on the same hidden state.
//!
//! # Multi-stream paged KV (FR-MD-08 / FR-EX-03 — ADR §D4)
//!
//! The backbone KV lives in the M3-03 [`PagedKvCache`] with its
//! `[time, stream, codebook]` 3D addressing and [`BlockSize::Two`]
//! (12.5 Hz audio-native, 1 block = 160 ms). "multi-stream KV" is
//! operationally two things:
//!
//! 1. **This cache**: time-axis paging plus the stream axis
//!    ([`CsmBackboneState::new_multi_stream`]) that a multi-session server
//!    (FR-SV-06, future wave) can drive without a layout change;
//! 2. the per-codebook RVQ feature streams, which are the
//!    `vokra_ops::mimi_rvq::mimi_rvq_decode_paged` codebook-axis contract
//!    (M3-06) — *not* this cache.
//!
//! The depth transformer's per-frame KV is deliberately **not** paged: it
//! resets every frame and holds at most `n_codebooks` positions, so page
//! management would cost more than it saves (ADR §D4).
//!
//! # Hot path (FR-EX-05)
//!
//! [`CsmBackboneState`] pre-allocates the paged arena
//! ([`PagedKvCache::pre_allocate`]) and a step-sized scratch at
//! construction; [`CsmBackbone::step_into`] runs the whole block stack with
//! **zero heap allocation** (pages come off the pre-allocated free list —
//! covered by `tests/csm_hot_path_alloc.rs`).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Weights not bound → [`VokraError::ModelLoad`] naming the tensor (never a
//! zero-fill); ill-formed config / out-of-range token / position past
//! `n_ctx` → [`VokraError::InvalidArgument`].

use vokra_core::cache::paged::{BlockSize, KvDims, PagedKvCache};
use vokra_core::rng::SplitMix64;
use vokra_core::{BackendKind, Result, VokraError};

use super::config::CsmConfig;
use super::rope::{llama3_inv_freqs, rope_apply_adjacent};
use crate::compute::{Compute, HotOp};
use crate::cosyvoice2::llm::LlmBlockWeights;
use crate::voxtral::text_decoder::{hadamard_inplace, rms_norm, silu_inplace};

/// Compute-seam hot ops the CSM backbone + depth transformer dispatch
/// (GEMM for projections / FFN, GEMV for the codebook heads, softmax for
/// attention). RMSNorm / SwiGLU / RoPE are scalar glue on the host. The
/// GPU sessions (T21 Metal / T22 CUDA) advertise the same set so the
/// coverage gate stays lock-step.
pub(crate) const CSM_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Gemv, HotOp::Softmax];

/// Seed for the synthesized fixture the loaders fall back to on the
/// shape-only converter path (T29 real binding pending). ASCII-ish stable
/// constant, distinct from the CosyVoice2 / Voxtral fixtures.
pub const CSM_FROM_GGUF_DEFAULT_SEED: u64 = 0x0C5A_0C5A_0C5A_0C5A;

/// One backbone sequence position: `n_codebooks` audio slots + 1 text slot
/// (`models.py` frame width 33 for the real checkpoint). A frame embeds as
/// the **sum of its valid slots**; upstream frames are either pure-text or
/// pure-audio, but the sum contract also accepts both-present (the masked
/// sum is closed under it). An empty frame is an explicit error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsmFrame {
    /// Audio slots — one RVQ code per codebook (`len == n_codebooks` when
    /// present).
    pub audio: Option<Vec<u32>>,
    /// Text slot (Llama-3.2 tokenizer id).
    pub text: Option<u32>,
}

impl CsmFrame {
    /// A pure-text frame (text slot valid, audio slots masked).
    #[must_use]
    pub fn text(token: u32) -> Self {
        Self {
            audio: None,
            text: Some(token),
        }
    }

    /// A pure-audio frame (all `n_codebooks` slots valid, text masked).
    #[must_use]
    pub fn audio(codes: Vec<u32>) -> Self {
        Self {
            audio: Some(codes),
            text: None,
        }
    }

    /// True when no slot is valid (embedding such a frame is an error —
    /// the upstream mask never produces one).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.audio.is_none() && self.text.is_none()
    }
}

/// All CSM backbone weights (ADR §D2 モデル head / embedding 表).
///
/// Layouts:
/// - `text_emb` — `[text_vocab, d]` row-major;
/// - `audio_emb` — `[audio_vocab * n_codebooks, d]` row-major, indexed
///   `token + codebook * audio_vocab` (`models.py` `_embed_audio`);
/// - `blocks` — per-layer Llama/Mistral block bundle. The bundle type is
///   **shared with CosyVoice2** ([`LlmBlockWeights`]) on purpose: the
///   block shape is identical (RMSNorm γ ×2 + GQA Q/K/V/O + SwiGLU
///   gate/up/down, all bias-less) so a GQA fix lands once for Voxtral /
///   CosyVoice2 / CSM;
/// - `final_norm_gamma` — `[d]`;
/// - `codebook0_head` — `[audio_vocab, d]` row-major (GEMV layout,
///   *not* transposed — the head is a matrix-vector product per step).
#[derive(Debug, Clone)]
pub struct CsmBackboneWeights {
    /// Text-slot embedding table `[text_vocab, d]`.
    pub text_emb: Vec<f32>,
    /// Audio-slot embedding table `[audio_vocab * n_codebooks, d]`.
    pub audio_emb: Vec<f32>,
    /// Per-layer transformer blocks (shared bundle type — see struct docs).
    pub blocks: Vec<LlmBlockWeights>,
    /// Final RMSNorm γ `[d]`.
    pub final_norm_gamma: Vec<f32>,
    /// Zeroth-codebook head `[audio_vocab, d]` row-major.
    pub codebook0_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`]; real-checkpoint parity
    /// assertions gate on `false`.
    pub is_synthesized: bool,
}

impl CsmBackboneWeights {
    /// Builds a synthesized (seed-deterministic) weight store: Xavier-like
    /// uniform for every projection/embedding, `1.0` for every RMSNorm γ
    /// (the M3-09 `LlmWeights::synthesized` recipe — numerical-stability /
    /// shape verification without the real checkpoint).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config` fails
    /// [`CsmConfig::validate_for_forward`].
    pub fn synthesized(config: &CsmConfig, seed: u64) -> Result<Self> {
        config.validate_for_forward()?;
        let d = config.backbone.d_model;
        let kv_hidden = config.backbone.kv_hidden_dim();
        let ffn = config.backbone.ffn_dim;
        let mut rng = SplitMix64::new(seed);
        let text_emb = xavier_uniform(&mut rng, config.text_vocab_size * d, d, d);
        let audio_emb = xavier_uniform(
            &mut rng,
            config.audio_vocab_size * config.n_codebooks * d,
            d,
            d,
        );
        let mut blocks = Vec::with_capacity(config.backbone.n_layer);
        for _ in 0..config.backbone.n_layer {
            blocks.push(synthesized_block(&mut rng, d, kv_hidden, ffn));
        }
        let final_norm_gamma = vec![1.0f32; d];
        let codebook0_head = xavier_uniform(
            &mut rng,
            config.audio_vocab_size * d,
            d,
            config.audio_vocab_size,
        );
        Ok(Self {
            text_emb,
            audio_emb,
            blocks,
            final_norm_gamma,
            codebook0_head,
            is_synthesized: true,
        })
    }

    /// Binds real weights from a CSM GGUF tensor store.
    ///
    /// **Honest stub** — the tensor-name manifest is T29 (owner checkpoint
    /// hand-off); the runtime never invents tensor names (CLAUDE.md
    /// hallucination ban, FR-EX-08 — never a silent zero-fill).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] until the T29 manifest lands.
    pub fn from_gguf(_file: &vokra_core::gguf::GgufFile, _config: &CsmConfig) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "CSM backbone real-weight binding is deferred to the T29 checkpoint \
             hand-off (ADR M4-05 §D2 tensor manifest). Use \
             CsmBackboneWeights::synthesized for the deterministic fixture path.",
        ))
    }
}

fn synthesized_block(
    rng: &mut SplitMix64,
    d: usize,
    kv_hidden: usize,
    ffn: usize,
) -> LlmBlockWeights {
    LlmBlockWeights {
        attn_norm_gamma: vec![1.0f32; d],
        q_w_t: xavier_uniform(rng, d * d, d, d),
        k_w_t: xavier_uniform(rng, d * kv_hidden, d, kv_hidden),
        v_w_t: xavier_uniform(rng, d * kv_hidden, d, kv_hidden),
        o_w_t: xavier_uniform(rng, d * d, d, d),
        ffn_norm_gamma: vec![1.0f32; d],
        ffn_gate_w_t: xavier_uniform(rng, d * ffn, d, ffn),
        ffn_up_w_t: xavier_uniform(rng, d * ffn, d, ffn),
        ffn_down_w_t: xavier_uniform(rng, ffn * d, ffn, d),
    }
}

/// Draws `n` f32 uniformly in `(-bound, +bound)`, `bound =
/// sqrt(6/(fan_in+fan_out))` (Xavier/Glorot — the M3-09 recipe; duplicated
/// locally because the cosyvoice2 helper is module-private).
pub(crate) fn xavier_uniform(
    rng: &mut SplitMix64,
    n: usize,
    fan_in: usize,
    fan_out: usize,
) -> Vec<f32> {
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_unit_f32();
        v.push((u * 2.0 - 1.0) * bound);
    }
    v
}

/// Pre-allocated per-state scratch. Sized once for `t_cap` positions; the
/// step path (`t = 1`) reuses it with zero allocation (FR-EX-05).
#[derive(Debug)]
pub(crate) struct BackboneScratch {
    t_cap: usize,
    embed: Vec<f32>,
    norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    rope_buf: Vec<f32>,
    k_hist: Vec<f32>,
    v_hist: Vec<f32>,
    scores: Vec<f32>,
    probs: Vec<f32>,
    attn_out: Vec<f32>,
    attn_o: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
    h: Vec<f32>,
}

impl BackboneScratch {
    fn new(config: &CsmConfig, t_cap: usize) -> Self {
        let d = config.backbone.d_model;
        let kv_hidden = config.backbone.kv_hidden_dim();
        let head_dim = config.backbone.head_dim();
        let ffn = config.backbone.ffn_dim;
        let n_ctx = config.n_ctx;
        Self {
            t_cap,
            embed: vec![0.0; t_cap * d],
            norm: vec![0.0; t_cap * d],
            q_proj: vec![0.0; t_cap * d],
            k_proj: vec![0.0; t_cap * kv_hidden],
            v_proj: vec![0.0; t_cap * kv_hidden],
            rope_buf: vec![0.0; t_cap * head_dim],
            k_hist: vec![0.0; n_ctx * kv_hidden],
            v_hist: vec![0.0; n_ctx * kv_hidden],
            scores: vec![0.0; t_cap * n_ctx],
            probs: vec![0.0; t_cap * n_ctx],
            attn_out: vec![0.0; t_cap * d],
            attn_o: vec![0.0; t_cap * d],
            ffn_gate: vec![0.0; t_cap * ffn],
            ffn_up: vec![0.0; t_cap * ffn],
            ffn_down: vec![0.0; t_cap * d],
            h: vec![0.0; t_cap * d],
        }
    }
}

/// Autoregressive backbone state: position clock + multi-stream paged KV +
/// step scratch. The paged arena is fully pre-allocated at construction —
/// the decode loop only pops pages off the free list (FR-EX-05 / ADR §D4).
pub struct CsmBackboneState {
    /// Per-stream position clocks (`seq_lens[stream]`).
    seq_lens: Vec<usize>,
    /// The stream this state currently decodes into.
    stream: usize,
    kv: PagedKvCache<f32>,
    scratch: BackboneScratch,
}

impl std::fmt::Debug for CsmBackboneState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmBackboneState")
            .field("seq_lens", &self.seq_lens)
            .field("stream", &self.stream)
            .field("pages_in_use", &self.kv.pages_in_use())
            .finish()
    }
}

impl CsmBackboneState {
    /// Single-stream state (the common dialog session shape).
    ///
    /// # Errors
    ///
    /// Propagates config validation and paged-arena allocation errors.
    pub fn new(config: &CsmConfig) -> Result<Self> {
        Self::new_multi_stream(config, 1, 0)
    }

    /// Multi-stream state: one paged arena hosting `n_stream` interleaved
    /// decode streams (FR-MD-08 / future FR-SV-06 server sessions), starting
    /// on `stream`. Streams share pages (the `[time, stream, codebook]` row
    /// layout) but keep independent position clocks.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `n_stream == 0` /
    /// `stream >= n_stream` or an ill-formed config; propagates arena
    /// allocation failure.
    pub fn new_multi_stream(config: &CsmConfig, n_stream: usize, stream: usize) -> Result<Self> {
        config.validate_for_forward()?;
        if n_stream == 0 || stream >= n_stream {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone state: stream {stream} out of range (n_stream {n_stream})"
            )));
        }
        let dims = KvDims {
            n_layer: config.backbone.n_layer,
            n_head: config.backbone.n_head_kv,
            d_head: config.backbone.head_dim(),
            n_stream,
            n_codebook: 1,
            max_time: config.n_ctx,
        };
        let kv = PagedKvCache::pre_allocate(dims, BlockSize::Two)?;
        Ok(Self {
            seq_lens: vec![0; n_stream],
            stream,
            kv,
            scratch: BackboneScratch::new(config, 1),
        })
    }

    /// The active stream's position clock.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_lens[self.stream]
    }

    /// The active stream index.
    #[must_use]
    pub fn stream(&self) -> usize {
        self.stream
    }

    /// Switches the active stream (multi-stream interleaving).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `stream` is out of range.
    pub fn set_stream(&mut self, stream: usize) -> Result<()> {
        if stream >= self.seq_lens.len() {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone state: stream {stream} out of range (n_stream {})",
                self.seq_lens.len()
            )));
        }
        self.stream = stream;
        Ok(())
    }

    /// Rewinds **every** stream and releases all pages back to the
    /// pre-allocated free list (arena capacity is retained — fast turn
    /// re-use, no realloc).
    pub fn reset(&mut self) {
        self.seq_lens.iter_mut().for_each(|l| *l = 0);
        self.kv.reset();
    }

    /// Paged-cache observability (tests + FR-EX-03 assertions).
    #[must_use]
    pub fn pages_in_use(&self) -> usize {
        self.kv.pages_in_use()
    }
}

/// The CSM backbone (config + weights + backend selection). The [`Compute`]
/// dispatcher is built per forward call (piper-plus pattern) so non-`Sync`
/// GPU contexts stay on the stack.
pub struct CsmBackbone {
    config: CsmConfig,
    weights: CsmBackboneWeights,
    backend: BackendKind,
    /// Precomputed Llama-3-scaled per-pair RoPE frequencies
    /// (`[head_dim/2]` — ADR §D3).
    inv_freqs: Vec<f32>,
}

impl std::fmt::Debug for CsmBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsmBackbone")
            .field("config", &self.config)
            .field("weights.is_synthesized", &self.weights.is_synthesized)
            .field("backend", &self.backend)
            .finish()
    }
}

impl CsmBackbone {
    /// Builds a backbone from an explicit weight store (CPU backend).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the config is ill-formed or any
    /// weight shape disagrees with it.
    pub fn new(config: CsmConfig, weights: CsmBackboneWeights) -> Result<Self> {
        config.validate_for_forward()?;
        validate_backbone_shapes(&config, &weights)?;
        let inv_freqs = llama3_inv_freqs(
            config.backbone.head_dim(),
            config.rope_base,
            config.rope_scaling.as_ref(),
        )?;
        Ok(Self {
            config,
            weights,
            backend: BackendKind::Cpu,
            inv_freqs,
        })
    }

    /// Synthesized-fixture constructor (shape / numerical-stability path).
    ///
    /// # Errors
    ///
    /// Propagates [`CsmBackboneWeights::synthesized`].
    pub fn synthesized(config: CsmConfig, seed: u64) -> Result<Self> {
        let weights = CsmBackboneWeights::synthesized(&config, seed)?;
        Self::new(config, weights)
    }

    /// Selects the backend the hot ops dispatch through (GPU sessions
    /// T21/T22 route here).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The selected backend.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The resolved config.
    #[must_use]
    pub fn config(&self) -> &CsmConfig {
        &self.config
    }

    /// The weight store (parity / shape assertions).
    #[must_use]
    pub fn weights(&self) -> &CsmBackboneWeights {
        &self.weights
    }

    fn compute(&self) -> Result<Compute> {
        Compute::for_backend(self.backend, CSM_HOT_OPS)
    }

    /// Embeds one frame into `out = [d]` — the masked sum of its valid
    /// slots (`models.py` `_embed_tokens`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an empty frame, a wrong-length
    /// audio slot vector, an out-of-range token id, or a wrong-sized `out`.
    pub fn embed_frame(&self, frame: &CsmFrame, out: &mut [f32]) -> Result<()> {
        let d = self.config.backbone.d_model;
        if out.len() != d {
            return Err(VokraError::InvalidArgument(format!(
                "csm embed_frame: out len {} != d_model {d}",
                out.len()
            )));
        }
        if frame.is_empty() {
            return Err(VokraError::InvalidArgument(
                "csm embed_frame: frame has no valid slot (upstream masks never \
                 produce an all-masked frame)"
                    .into(),
            ));
        }
        out.iter_mut().for_each(|v| *v = 0.0);
        if let Some(codes) = &frame.audio {
            if codes.len() != self.config.n_codebooks {
                return Err(VokraError::InvalidArgument(format!(
                    "csm embed_frame: audio slots {} != n_codebooks {}",
                    codes.len(),
                    self.config.n_codebooks
                )));
            }
            for (cb, &tok) in codes.iter().enumerate() {
                let row = self.audio_embedding(cb, tok)?;
                for (dst, src) in out.iter_mut().zip(row.iter()) {
                    *dst += *src;
                }
            }
        }
        if let Some(tok) = frame.text {
            let tok = tok as usize;
            if tok >= self.config.text_vocab_size {
                return Err(VokraError::InvalidArgument(format!(
                    "csm embed_frame: text token {tok} >= text_vocab {}",
                    self.config.text_vocab_size
                )));
            }
            let row = &self.weights.text_emb[tok * d..(tok + 1) * d];
            for (dst, src) in out.iter_mut().zip(row.iter()) {
                *dst += *src;
            }
        }
        Ok(())
    }

    /// The `d`-long audio embedding row for `(codebook, token)` — indexed
    /// `token + codebook * audio_vocab` (`models.py` `_embed_audio`). The
    /// depth transformer conditions on these rows.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on out-of-range codebook / token.
    pub fn audio_embedding(&self, codebook: usize, token: u32) -> Result<&[f32]> {
        let d = self.config.backbone.d_model;
        if codebook >= self.config.n_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio_embedding: codebook {codebook} >= n_codebooks {}",
                self.config.n_codebooks
            )));
        }
        let tok = token as usize;
        if tok >= self.config.audio_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "csm audio_embedding: token {tok} >= audio_vocab {}",
                self.config.audio_vocab_size
            )));
        }
        let row = codebook * self.config.audio_vocab_size + tok;
        Ok(&self.weights.audio_emb[row * d..(row + 1) * d])
    }

    /// Bulk forward over `frames`, appending their K/V rows to `state` and
    /// returning the final-RMSNorm hidden states `[t, d]` row-major.
    ///
    /// Allocates a `t`-sized scratch (prefill is not the hot loop); the
    /// per-frame decode path is [`Self::step_into`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on empty `frames`, a position past
    /// `n_ctx`, or any embed error; propagates Compute-seam errors.
    pub fn forward(&self, frames: &[CsmFrame], state: &mut CsmBackboneState) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Err(VokraError::InvalidArgument(
                "csm backbone forward: frames must be non-empty".into(),
            ));
        }
        let t = frames.len();
        let mut scratch = BackboneScratch::new(&self.config, t);
        let mut hidden = vec![0.0f32; t * self.config.backbone.d_model];
        self.forward_impl(frames, state, &mut scratch, &mut hidden)?;
        Ok(hidden)
    }

    /// One autoregressive step over a single frame with **zero heap
    /// allocation** (state scratch + pre-allocated pages only — FR-EX-05).
    /// Writes the final hidden state of the new position into
    /// `hidden_out = [d]`.
    ///
    /// # Errors
    ///
    /// Same surface as [`Self::forward`].
    pub fn step_into(
        &self,
        state: &mut CsmBackboneState,
        frame: &CsmFrame,
        hidden_out: &mut [f32],
    ) -> Result<()> {
        if hidden_out.len() != self.config.backbone.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone step: hidden_out len {} != d_model {}",
                hidden_out.len(),
                self.config.backbone.d_model
            )));
        }
        let frames = std::slice::from_ref(frame);
        // Move the scratch out to satisfy the borrow checker (forward_impl
        // needs &mut scratch alongside &mut state.kv). Vec moves are
        // pointer swaps — no allocation.
        let mut scratch = std::mem::replace(
            &mut state.scratch,
            BackboneScratch {
                t_cap: 0,
                embed: Vec::new(),
                norm: Vec::new(),
                q_proj: Vec::new(),
                k_proj: Vec::new(),
                v_proj: Vec::new(),
                rope_buf: Vec::new(),
                k_hist: Vec::new(),
                v_hist: Vec::new(),
                scores: Vec::new(),
                probs: Vec::new(),
                attn_out: Vec::new(),
                attn_o: Vec::new(),
                ffn_gate: Vec::new(),
                ffn_up: Vec::new(),
                ffn_down: Vec::new(),
                h: Vec::new(),
            },
        );
        let result = self.forward_impl(frames, state, &mut scratch, hidden_out);
        state.scratch = scratch;
        result
    }

    /// Allocating convenience wrapper over [`Self::step_into`].
    ///
    /// # Errors
    ///
    /// See [`Self::step_into`].
    pub fn step(&self, state: &mut CsmBackboneState, frame: &CsmFrame) -> Result<Vec<f32>> {
        let mut hidden = vec![0.0f32; self.config.backbone.d_model];
        self.step_into(state, frame, &mut hidden)?;
        Ok(hidden)
    }

    /// Zeroth-codebook logits from a final hidden state (`codebook0_head`
    /// GEMV): `out = [audio_vocab]`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch; propagates
    /// Compute-seam errors.
    pub fn c0_logits_into(&self, hidden: &[f32], out: &mut [f32]) -> Result<()> {
        let d = self.config.backbone.d_model;
        let vocab = self.config.audio_vocab_size;
        if hidden.len() != d || out.len() != vocab {
            return Err(VokraError::InvalidArgument(format!(
                "csm c0_logits: hidden len {} (want {d}) / out len {} (want {vocab})",
                hidden.len(),
                out.len()
            )));
        }
        let compute = self.compute()?;
        compute.gemv_f32(vocab, d, &self.weights.codebook0_head, hidden, None, out)
    }

    /// Allocating convenience wrapper over [`Self::c0_logits_into`].
    ///
    /// # Errors
    ///
    /// See [`Self::c0_logits_into`].
    pub fn c0_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.config.audio_vocab_size];
        self.c0_logits_into(hidden, &mut out)?;
        Ok(out)
    }

    /// The single forward body shared by bulk `forward` (t = frames.len())
    /// and `step` (t = 1): pre-norm Llama block stack with GQA attention
    /// over the paged KV history. `hidden_out` receives the final-RMSNorm
    /// hidden states `[t, d]`.
    fn forward_impl(
        &self,
        frames: &[CsmFrame],
        state: &mut CsmBackboneState,
        scratch: &mut BackboneScratch,
        hidden_out: &mut [f32],
    ) -> Result<()> {
        let t = frames.len();
        let cfg = &self.config.backbone;
        let d = cfg.d_model;
        let n_head_q = cfg.n_head_q;
        let n_head_kv = cfg.n_head_kv;
        let head_dim = cfg.head_dim();
        let kv_hidden = cfg.kv_hidden_dim();
        let ffn = cfg.ffn_dim;
        let n_kv_groups = n_head_q / n_head_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;
        let stream = state.stream;
        let position_offset = state.seq_lens[stream];

        if position_offset + t > self.config.n_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone: position {} + t {} > n_ctx {} (FR-EX-08 — no silent \
                 wrap-around)",
                position_offset, t, self.config.n_ctx
            )));
        }
        if scratch.t_cap < t {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone: scratch capacity {} < t {t} (internal sizing bug)",
                scratch.t_cap
            )));
        }
        if hidden_out.len() != t * d {
            return Err(VokraError::InvalidArgument(format!(
                "csm backbone: hidden_out len {} != t*d {}",
                hidden_out.len(),
                t * d
            )));
        }
        let compute = self.compute()?;

        // Frame embeddings → h [t, d].
        let h = &mut scratch.h[..t * d];
        for (i, frame) in frames.iter().enumerate() {
            // Split-borrow: embed into the dedicated embed row, then copy.
            self.embed_frame(frame, &mut scratch.embed[..d])?;
            h[i * d..(i + 1) * d].copy_from_slice(&scratch.embed[..d]);
        }

        let t_kv = position_offset + t;
        for (layer_idx, block) in self.weights.blocks.iter().enumerate() {
            // ---------- Pre-norm GQA attention ----------
            rms_norm(
                &scratch.h[..t * d],
                &block.attn_norm_gamma,
                eps,
                t,
                &mut scratch.norm[..t * d],
            )?;
            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.norm[..t * d],
                &block.q_w_t,
                None,
                &mut scratch.q_proj[..t * d],
            )?;
            compute.gemm_f32(
                t,
                kv_hidden,
                d,
                &scratch.norm[..t * d],
                &block.k_w_t,
                None,
                &mut scratch.k_proj[..t * kv_hidden],
            )?;
            compute.gemm_f32(
                t,
                kv_hidden,
                d,
                &scratch.norm[..t * d],
                &block.v_w_t,
                None,
                &mut scratch.v_proj[..t * kv_hidden],
            )?;

            // Llama-3 scaled RoPE per head (adjacent-pair — ADR §D3).
            for h_q in 0..n_head_q {
                for i in 0..t {
                    let src = &scratch.q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                    scratch.rope_buf[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
                }
                rope_apply_adjacent(
                    &mut scratch.rope_buf[..t * head_dim],
                    t,
                    head_dim,
                    &self.inv_freqs,
                    position_offset,
                )?;
                for i in 0..t {
                    let dst =
                        &mut scratch.q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                    dst.copy_from_slice(&scratch.rope_buf[i * head_dim..(i + 1) * head_dim]);
                }
            }
            for h_kv in 0..n_head_kv {
                for i in 0..t {
                    let src = &scratch.k_proj
                        [i * kv_hidden + h_kv * head_dim..i * kv_hidden + (h_kv + 1) * head_dim];
                    scratch.rope_buf[i * head_dim..(i + 1) * head_dim].copy_from_slice(src);
                }
                rope_apply_adjacent(
                    &mut scratch.rope_buf[..t * head_dim],
                    t,
                    head_dim,
                    &self.inv_freqs,
                    position_offset,
                )?;
                for i in 0..t {
                    let dst = &mut scratch.k_proj
                        [i * kv_hidden + h_kv * head_dim..i * kv_hidden + (h_kv + 1) * head_dim];
                    dst.copy_from_slice(&scratch.rope_buf[i * head_dim..(i + 1) * head_dim]);
                }
            }

            // Append the new K/V rows to the paged cache ([time, stream,
            // codebook=0] addressing — ADR §D4), then snapshot the full
            // history into the contiguous scratch for the attention loop.
            for i in 0..t {
                state.kv.append_step(
                    layer_idx,
                    position_offset + i,
                    stream,
                    0,
                    &scratch.k_proj[i * kv_hidden..(i + 1) * kv_hidden],
                    &scratch.v_proj[i * kv_hidden..(i + 1) * kv_hidden],
                )?;
            }
            for j in 0..t_kv {
                let (k_row, v_row) =
                    state.kv.read_step(layer_idx, j, stream, 0).ok_or_else(|| {
                        VokraError::InvalidArgument(format!(
                            "csm backbone: KV history hole at layer {layer_idx} t {j} \
                             (stream {stream}) — state was reset mid-decode?"
                        ))
                    })?;
                scratch.k_hist[j * kv_hidden..(j + 1) * kv_hidden].copy_from_slice(k_row);
                scratch.v_hist[j * kv_hidden..(j + 1) * kv_hidden].copy_from_slice(v_row);
            }

            // GQA attention (h_kv = h_q / n_kv_groups).
            let scores = &mut scratch.scores[..t * t_kv];
            let probs = &mut scratch.probs[..t * t_kv];
            for h_q in 0..n_head_q {
                let h_kv = h_q / n_kv_groups;
                for i in 0..t {
                    let q_row =
                        &scratch.q_proj[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                    let row_start = i * t_kv;
                    for j in 0..t_kv {
                        let k_row = &scratch.k_hist[j * kv_hidden + h_kv * head_dim
                            ..j * kv_hidden + (h_kv + 1) * head_dim];
                        let mut s = 0.0f32;
                        for c in 0..head_dim {
                            s += q_row[c] * k_row[c];
                        }
                        scores[row_start + j] = s * scale;
                    }
                    let cur_pos = position_offset + i;
                    for j in (cur_pos + 1)..t_kv {
                        scores[row_start + j] = f32::NEG_INFINITY;
                    }
                }
                compute.softmax_f32(scores, probs, t, t_kv)?;
                for i in 0..t {
                    let out_dst =
                        &mut scratch.attn_out[i * d + h_q * head_dim..i * d + (h_q + 1) * head_dim];
                    for (c, out) in out_dst.iter_mut().enumerate() {
                        let mut sum = 0.0f32;
                        for j in 0..t_kv {
                            sum += probs[i * t_kv + j]
                                * scratch.v_hist[j * kv_hidden + h_kv * head_dim + c];
                        }
                        *out = sum;
                    }
                }
            }

            compute.gemm_f32(
                t,
                d,
                d,
                &scratch.attn_out[..t * d],
                &block.o_w_t,
                None,
                &mut scratch.attn_o[..t * d],
            )?;
            for i in 0..t * d {
                scratch.h[i] += scratch.attn_o[i];
            }

            // ---------- Pre-norm SwiGLU FFN ----------
            rms_norm(
                &scratch.h[..t * d],
                &block.ffn_norm_gamma,
                eps,
                t,
                &mut scratch.norm[..t * d],
            )?;
            compute.gemm_f32(
                t,
                ffn,
                d,
                &scratch.norm[..t * d],
                &block.ffn_gate_w_t,
                None,
                &mut scratch.ffn_gate[..t * ffn],
            )?;
            compute.gemm_f32(
                t,
                ffn,
                d,
                &scratch.norm[..t * d],
                &block.ffn_up_w_t,
                None,
                &mut scratch.ffn_up[..t * ffn],
            )?;
            silu_inplace(&mut scratch.ffn_gate[..t * ffn]);
            hadamard_inplace(&mut scratch.ffn_gate[..t * ffn], &scratch.ffn_up[..t * ffn])?;
            compute.gemm_f32(
                t,
                d,
                ffn,
                &scratch.ffn_gate[..t * ffn],
                &block.ffn_down_w_t,
                None,
                &mut scratch.ffn_down[..t * d],
            )?;
            for i in 0..t * d {
                scratch.h[i] += scratch.ffn_down[i];
            }
        }
        state.kv.advance(t);
        state.seq_lens[stream] += t;

        // Final RMSNorm into the caller's buffer.
        rms_norm(
            &scratch.h[..t * d],
            &self.weights.final_norm_gamma,
            eps,
            t,
            hidden_out,
        )?;
        Ok(())
    }
}

fn validate_backbone_shapes(config: &CsmConfig, weights: &CsmBackboneWeights) -> Result<()> {
    let d = config.backbone.d_model;
    let kv_hidden = config.backbone.kv_hidden_dim();
    let ffn = config.backbone.ffn_dim;
    let checks = [
        (
            "text_emb",
            weights.text_emb.len(),
            config.text_vocab_size * d,
        ),
        (
            "audio_emb",
            weights.audio_emb.len(),
            config.audio_vocab_size * config.n_codebooks * d,
        ),
        ("final_norm_gamma", weights.final_norm_gamma.len(), d),
        (
            "codebook0_head",
            weights.codebook0_head.len(),
            config.audio_vocab_size * d,
        ),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(VokraError::InvalidArgument(format!(
                "csm CsmBackbone::new: {name} len {got} != expected {want}"
            )));
        }
    }
    if weights.blocks.len() != config.backbone.n_layer {
        return Err(VokraError::InvalidArgument(format!(
            "csm CsmBackbone::new: blocks {} != n_layer {}",
            weights.blocks.len(),
            config.backbone.n_layer
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
                    "csm CsmBackbone::new: block[{i}].{name} len {got} != expected {want}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backbone() -> CsmBackbone {
        CsmBackbone::synthesized(CsmConfig::tiny_for_tests(), 7).expect("synthesized backbone")
    }

    fn audio_frame(seed: u32, n_codebooks: usize, vocab: usize) -> CsmFrame {
        CsmFrame::audio(
            (0..n_codebooks)
                .map(|cb| ((seed as usize + cb * 3) % vocab) as u32)
                .collect(),
        )
    }

    #[test]
    fn synthesized_weights_have_config_shapes() {
        let b = backbone();
        let cfg = b.config().clone();
        let w = b.weights();
        assert_eq!(w.text_emb.len(), cfg.text_vocab_size * cfg.backbone.d_model);
        assert_eq!(
            w.audio_emb.len(),
            cfg.audio_vocab_size * cfg.n_codebooks * cfg.backbone.d_model
        );
        assert_eq!(w.blocks.len(), cfg.backbone.n_layer);
        assert!(w.is_synthesized);
    }

    #[test]
    fn wrong_shape_weights_are_rejected() {
        let cfg = CsmConfig::tiny_for_tests();
        let mut w = CsmBackboneWeights::synthesized(&cfg, 7).expect("weights");
        w.codebook0_head.pop();
        assert!(matches!(
            CsmBackbone::new(cfg, w),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn from_gguf_weights_are_an_honest_not_implemented_stub() {
        // T29 flip-the-switch: never a silent zero-fill (FR-EX-08).
        let cfg = CsmConfig::tiny_for_tests();
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        let file = vokra_core::gguf::GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            CsmBackboneWeights::from_gguf(&file, &cfg),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn embed_frame_sums_valid_slots_only() {
        let b = backbone();
        let d = b.config().backbone.d_model;
        let mut text_only = vec![0.0f32; d];
        b.embed_frame(&CsmFrame::text(3), &mut text_only).unwrap();
        let mut audio_only = vec![0.0f32; d];
        let af = audio_frame(1, b.config().n_codebooks, b.config().audio_vocab_size);
        b.embed_frame(&af, &mut audio_only).unwrap();
        let mut both = vec![0.0f32; d];
        let bf = CsmFrame {
            audio: af.audio.clone(),
            text: Some(3),
        };
        b.embed_frame(&bf, &mut both).unwrap();
        for i in 0..d {
            assert!(
                (both[i] - (text_only[i] + audio_only[i])).abs() < 1e-6,
                "sum contract at {i}"
            );
        }
    }

    #[test]
    fn embed_frame_rejects_empty_and_out_of_range() {
        let b = backbone();
        let d = b.config().backbone.d_model;
        let mut out = vec![0.0f32; d];
        let empty = CsmFrame {
            audio: None,
            text: None,
        };
        assert!(b.embed_frame(&empty, &mut out).is_err());
        let bad_text = CsmFrame::text(b.config().text_vocab_size as u32);
        assert!(b.embed_frame(&bad_text, &mut out).is_err());
        let bad_audio = CsmFrame::audio(vec![
            b.config().audio_vocab_size as u32;
            b.config().n_codebooks
        ]);
        assert!(b.embed_frame(&bad_audio, &mut out).is_err());
        let wrong_len = CsmFrame::audio(vec![0; b.config().n_codebooks + 1]);
        assert!(b.embed_frame(&wrong_len, &mut out).is_err());
    }

    #[test]
    fn forward_is_finite_and_deterministic() {
        let b = backbone();
        let frames = vec![
            CsmFrame::text(1),
            CsmFrame::text(5),
            audio_frame(2, b.config().n_codebooks, b.config().audio_vocab_size),
        ];
        let mut s1 = CsmBackboneState::new(b.config()).unwrap();
        let h1 = b.forward(&frames, &mut s1).unwrap();
        assert_eq!(h1.len(), frames.len() * b.config().backbone.d_model);
        assert!(h1.iter().all(|v| v.is_finite()), "hidden must be finite");
        let mut s2 = CsmBackboneState::new(b.config()).unwrap();
        let h2 = b.forward(&frames, &mut s2).unwrap();
        assert_eq!(h1, h2, "same seed + input → bit-identical");
    }

    #[test]
    fn forward_matches_step_by_step() {
        // Bulk forward (t = N, one causal-masked pass) vs N incremental
        // steps through the paged cache: the M3-09
        // `forward_matches_step_by_step` property (KV consistency).
        let b = backbone();
        let frames = vec![
            CsmFrame::text(1),
            audio_frame(0, b.config().n_codebooks, b.config().audio_vocab_size),
            audio_frame(4, b.config().n_codebooks, b.config().audio_vocab_size),
            CsmFrame::text(9),
        ];
        let d = b.config().backbone.d_model;
        let mut bulk_state = CsmBackboneState::new(b.config()).unwrap();
        let bulk = b.forward(&frames, &mut bulk_state).unwrap();
        let mut step_state = CsmBackboneState::new(b.config()).unwrap();
        let mut last = vec![0.0f32; d];
        for f in &frames {
            b.step_into(&mut step_state, f, &mut last).unwrap();
        }
        let bulk_last = &bulk[(frames.len() - 1) * d..];
        let max_delta = bulk_last
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta <= 1e-4,
            "bulk vs step last-hidden max |Δ| = {max_delta} > 1e-4"
        );
        assert_eq!(bulk_state.seq_len(), step_state.seq_len());
    }

    #[test]
    fn c0_logits_shape_and_error_paths() {
        let b = backbone();
        let mut state = CsmBackboneState::new(b.config()).unwrap();
        let hidden = b.step(&mut state, &CsmFrame::text(0)).unwrap();
        let logits = b.c0_logits(&hidden).unwrap();
        assert_eq!(logits.len(), b.config().audio_vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert!(b.c0_logits(&hidden[1..]).is_err(), "wrong hidden len");
    }

    #[test]
    fn position_past_n_ctx_is_a_loud_error() {
        let mut cfg = CsmConfig::tiny_for_tests();
        cfg.n_ctx = 2;
        let b = CsmBackbone::synthesized(cfg, 3).unwrap();
        let mut state = CsmBackboneState::new(b.config()).unwrap();
        b.step(&mut state, &CsmFrame::text(0)).unwrap();
        b.step(&mut state, &CsmFrame::text(1)).unwrap();
        assert!(matches!(
            b.step(&mut state, &CsmFrame::text(2)),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn reset_reuses_the_preallocated_arena() {
        let b = backbone();
        let mut state = CsmBackboneState::new(b.config()).unwrap();
        let cap_before = state.kv.arena_capacity_pages();
        b.step(&mut state, &CsmFrame::text(0)).unwrap();
        assert!(state.pages_in_use() > 0);
        state.reset();
        assert_eq!(state.pages_in_use(), 0, "reset releases pages");
        assert_eq!(
            state.kv.arena_capacity_pages(),
            cap_before,
            "arena capacity retained (free-list reuse, no realloc)"
        );
        assert_eq!(state.seq_len(), 0);
        // The state decodes again after reset.
        let h = b.step(&mut state, &CsmFrame::text(1)).unwrap();
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn multi_stream_states_are_independent_and_match_single_stream() {
        let b = backbone();
        let frames_a = vec![CsmFrame::text(1), CsmFrame::text(2)];
        let frames_b = vec![CsmFrame::text(3)];

        // Interleaved two-stream state.
        let mut multi = CsmBackboneState::new_multi_stream(b.config(), 2, 0).unwrap();
        let mut last_a = vec![0.0f32; b.config().backbone.d_model];
        let mut last_b = vec![0.0f32; b.config().backbone.d_model];
        b.step_into(&mut multi, &frames_a[0], &mut last_a).unwrap();
        multi.set_stream(1).unwrap();
        b.step_into(&mut multi, &frames_b[0], &mut last_b).unwrap();
        multi.set_stream(0).unwrap();
        b.step_into(&mut multi, &frames_a[1], &mut last_a).unwrap();
        assert_eq!(multi.seq_len(), 2, "stream 0 clock");
        multi.set_stream(1).unwrap();
        assert_eq!(multi.seq_len(), 1, "stream 1 clock");

        // Reference: two independent single-stream states.
        let mut ref_a = CsmBackboneState::new(b.config()).unwrap();
        let ha = b.forward(&frames_a, &mut ref_a).unwrap();
        let d = b.config().backbone.d_model;
        let ref_last_a = &ha[d..];
        let mut ref_b = CsmBackboneState::new(b.config()).unwrap();
        let hb = b.forward(&frames_b, &mut ref_b).unwrap();

        let da = ref_last_a
            .iter()
            .zip(last_a.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let db = hb
            .iter()
            .zip(last_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(da <= 1e-4, "stream 0 isolation max |Δ| = {da}");
        assert!(db <= 1e-4, "stream 1 isolation max |Δ| = {db}");
    }

    #[test]
    fn stream_selector_bounds_are_checked() {
        let b = backbone();
        assert!(CsmBackboneState::new_multi_stream(b.config(), 2, 2).is_err());
        assert!(CsmBackboneState::new_multi_stream(b.config(), 0, 0).is_err());
        let mut s = CsmBackboneState::new(b.config()).unwrap();
        assert!(s.set_stream(1).is_err());
    }
}
