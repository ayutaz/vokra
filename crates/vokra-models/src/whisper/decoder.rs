//! Whisper text decoder forward pass with a KV cache.
//!
//! Structure (openai/whisper `TextDecoder`, HF `WhisperDecoder`):
//!
//! - token embedding + learned positional embedding (`scale_embedding = false`,
//!   so no `√d` scaling);
//! - `n_text_layer` blocks of: pre-norm **causal** self-attention → pre-norm
//!   **cross**-attention over the encoder output → pre-norm MLP;
//! - a final LayerNorm and a logits projection **tied** to the token embedding
//!   (`logits = h · embed_tokensᵀ`, no bias).
//!
//! # KV cache (model-internal, M0)
//!
//! [`DecoderState`] keeps two kinds of cache:
//!
//! - **cross-attention** K/V, computed **once** from the encoder output and
//!   reused for every decode step;
//! - **self-attention** K/V, appended each step so past tokens are not
//!   recomputed.
//!
//! Promoting this to a first-class session state (FR-EX-02) is M1-04; here it
//! is a private implementation detail that never appears on the graph I/O. A
//! full-sequence call ([`DecoderState::step`] on the whole prefix) and a
//! token-by-token cached call produce identical logits (verified by the parity
//! tests), which is what the search integration (M0-06-T23) relies on.

use std::sync::Arc;

use vokra_backend_cpu::kernels::LAYER_NORM_DEFAULT_EPS;
use vokra_core::{BackendKind, DecoderLayerView, KvCache, Result, VokraError};

use super::WhisperModel;
use super::config::WhisperConfig;
use super::encoder::EncoderOutput;
use super::nn::{
    add_assign, attention_from_kv_into, layer_norm_into, mlp_into, project_kv, project_kv_into,
};
use super::scratch::{BlockScratch, LogitsScratch, resize_zeroed};
use super::weights::{DecoderLayer, DecoderWeights};
use crate::compute::{Compute, DecoderStepDims, DecoderStepSession};

/// Initial reservation *hint* for the self-attention KV cache, in positions
/// (tokens).
///
/// Deliberately **not** the static `n_text_ctx` maximum (448 for Whisper base):
/// a variable-length decode is usually far shorter, so reserving the worst-case
/// window upfront wastes memory on every short utterance (M1-04 sub-part 2).
/// The cache is seeded to this hint (capped at `n_text_ctx`, the hard upper
/// bound `step_into` enforces) and grows amortically if a longer decode needs
/// it. The per-step *compute* scratch is still bounded to `n_text_ctx` in
/// [`DecoderState::new`], so the arithmetic hot path stays allocation-free;
/// only the cache's own key/value buffers grow.
const SELF_KV_RESERVE_HINT: usize = 64;

/// A decoder run bound to one encoder output, holding the KV caches.
///
/// Owns the model through an [`Arc`] rather than borrowing it, so the state has
/// no lifetime and is [`Send`]: a decode can be moved across threads (the
/// M1-08 streaming foundation). The growable self-attention cache is the
/// first-class [`KvCache`]; the cross-attention K/V are computed once from the
/// encoder output and kept alongside.
///
/// # Reusable scratch (M1-04, FR-EX-05)
///
/// The residual buffer [`h`](Self::h), the per-block [`BlockScratch`] and the
/// [`LogitsScratch`] are owned here and **reused for every step and every
/// layer**: each is reserved once (to the text-context / prefix bounds) in
/// [`new`](Self::new) and thereafter only `clear()`/`resize()`-d, so the
/// autoregressive decode loop performs no heap allocation at steady state. This
/// is the whisper.cpp reused-buffer pattern in safe Rust; the capacity-stability
/// test below is its oracle.
pub struct DecoderState {
    /// The loaded model (config + weights), shared and kept alive by this run.
    model: Arc<WhisperModel>,
    /// Per-layer cross-attention `(k, v)`, each `[n_ctx, d]` (computed once).
    cross_kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Number of encoder context positions.
    n_ctx: usize,
    /// Growable per-layer self-attention key/value cache (`positions` tracks the
    /// committed token count).
    self_kv: KvCache,
    /// Residual hidden-state stream `[t, d]` for the current step (reused).
    h: Vec<f32>,
    /// Per-transformer-block scratch, reused across all layers of a step.
    block: BlockScratch,
    /// Tied-logits-head scratch; its `out` holds the last step's logits.
    logits: LogitsScratch,
    /// Backend selector for the step forward (`Copy`, so [`DecoderState`] stays
    /// `Send` — it never holds a live `!Send` backend). A [`Compute`] is built
    /// from it at each step entry (M2-01 Phase 3). Metal does not yet cover the
    /// Whisper op set, so a Metal state is an explicit error at construction /
    /// step (never a silent CPU fall back, FR-EX-08).
    backend_kind: BackendKind,
    /// Device-resident decoder-step session (Phase 3a, Metal-only in this slice).
    ///
    /// `Some(_)` when `Compute::for_backend(backend_kind, …)` reports
    /// [`Compute::decoder_step_is_session_backed`] `= true` (Metal); `None` for
    /// CPU and CUDA, which keep the per-op step loop untouched — so the CPU
    /// path is byte-for-byte the pre-Phase-3 code (FR-EX-08). See
    /// [`DecoderStepSession`] for the SAFETY note on why holding it here does
    /// not violate `DecoderState: Send`.
    device_session: Option<DecoderStepSession>,
}

impl DecoderState {
    /// Binds to `encoder` on the CPU backend and precomputes the
    /// cross-attention K/V for every layer. Takes ownership of a cloned [`Arc`]
    /// to the model (see [`WhisperModel::decoder`]).
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        Self::new_with_backend(model, encoder, BackendKind::Cpu)
    }

    /// Like [`new`](Self::new) but on an explicit backend (M2-01 Phase 3).
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`](vokra_core::VokraError) if `backend_kind`
    /// does not cover the Whisper hot-op set (e.g. Metal, whose softmax /
    /// layer-norm / … kernels are not yet landed) — an explicit error at
    /// construction, never a silent CPU fall back (FR-EX-08).
    pub(crate) fn new_with_backend(
        model: Arc<WhisperModel>,
        encoder: &EncoderOutput,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        // The cross-K/V precompute is a GEMM; build the backend dispatcher and
        // reject up front any backend that does not cover the full step op set.
        let compute = Compute::for_backend(backend_kind, super::WHISPER_HOT_OPS)?;
        let (cfg, w) = model.decoder_state();
        if encoder.d_model != cfg.d_model {
            return Err(VokraError::InvalidArgument(format!(
                "whisper decoder: encoder d_model {} != config {}",
                encoder.d_model, cfg.d_model
            )));
        }
        let n_layer = w.layers.len();
        // Seed the self-attention cache to a typical-decode hint rather than the
        // static `n_text_ctx` max (M1-04 sub-part 2): short utterances no longer
        // pay for the worst-case window, and a longer decode grows the cache
        // amortically. The hint is capped at `n_text_ctx` (the hard upper bound
        // `step_into` enforces), so a tiny window never over-reserves.
        let self_kv = KvCache::with_reserve(
            n_layer,
            cfg.d_model,
            SELF_KV_RESERVE_HINT.min(cfg.n_text_ctx),
        );
        let mut cross_kv = Vec::with_capacity(n_layer);
        for layer in &w.layers {
            cross_kv.push(project_kv(
                &compute,
                &encoder.hidden,
                encoder.n_ctx,
                &layer.cross_attn,
            )?);
        }
        let n_ctx = encoder.n_ctx;

        // Reusable scratch (sub-part 3). `t_q_max` is the prefix width — the
        // largest single greedy step (post-prefix steps decode one token). The
        // attention scratch must cover the largest `t_kv` seen: the self-
        // attention window `n_text_ctx` *or* the cross-attention key count
        // `n_ctx`, whichever is larger. Reserving to these bounds makes the
        // greedy step loop allocation-free; a rare larger one-shot call (e.g. a
        // full-window beam recompute) resizes up once, outside that loop.
        let d = cfg.d_model;
        let ff = cfg.ffn_dim;
        let n_head = cfg.n_text_head;
        let n_vocab = cfg.n_vocab;
        let t_q_max = cfg.decoder_start_ids.len().max(1);
        let attn_t_kv_max = cfg.n_text_ctx.max(n_ctx);
        let h = Vec::with_capacity(t_q_max * d);
        let block = BlockScratch::with_reserve(t_q_max, attn_t_kv_max, d, ff, n_head);
        let logits = LogitsScratch::with_reserve(t_q_max, d, n_vocab);

        // Phase-3a Metal wiring: on a session-backed backend (Metal), build the
        // device-resident decoder-step driver ONCE here — every weight uploaded,
        // the pre-projected cross-K/V pinned, the self-KV cache reserved to
        // `n_text_ctx`. Every `step_into` then advances the whole decode step
        // device-resident in one command-buffer submission (see [`Self::step_into`]).
        // On CPU (and, for now, CUDA — Phase 3b) this stays `None` and the per-op
        // step loop below runs unchanged — the CPU path is byte-for-byte pre-Phase-3
        // code (FR-EX-08, no silent fall back).
        let device_session = if compute.decoder_step_is_session_backed() {
            // Borrow every layer's slices as a plain-slice `DecoderLayerView`
            // (row-major `[in, out]`, matching the CPU `Linear` layout) plus the
            // pre-projected cross-K/V we just computed. This vector lives only in
            // the constructor; `new_decoder_step_session` copies every slice into
            // owned device buffers before returning, and the borrows end here.
            let views: Vec<DecoderLayerView<'_>> = w
                .layers
                .iter()
                .enumerate()
                .map(|(li, l)| decoder_layer_view(l, &cross_kv[li].0, &cross_kv[li].1))
                .collect();
            let dims = DecoderStepDims {
                d,
                n_head,
                ff,
                n_text_ctx: cfg.n_text_ctx,
                n_vocab,
                n_ctx,
                max_t_q: t_q_max,
                eps: LAYER_NORM_DEFAULT_EPS,
            };
            Some(compute.new_decoder_step_session(
                dims,
                &views,
                &w.token_emb,
                &w.ln_post.gamma,
                &w.ln_post.beta,
            )?)
        } else {
            None
        };

        // The `cfg` / `w` borrows of `model` end here, so `model` can be moved in.
        Ok(Self {
            model,
            cross_kv,
            n_ctx,
            self_kv,
            h,
            block,
            logits,
            backend_kind,
            device_session,
        })
    }

    /// Clears the self-attention cache (the cross K/V stay valid) so a fresh
    /// decode of the same audio reproduces the first run. The reserved capacity
    /// is kept.
    pub fn reset(&mut self) {
        self.self_kv.reset();
        // Mirror the host cache clear on the device: the resident weights + the
        // pre-projected cross-K/V stay valid, the position clock rewinds to 0,
        // and the next `step_into` starts writing self-KV rows from row 0 again.
        if let Some(session) = self.device_session.as_mut() {
            session.reset();
        }
    }

    /// Number of tokens currently in the self-attention cache.
    pub fn position(&self) -> usize {
        self.self_kv.positions()
    }

    /// Advances the decoder by `tokens`, appending their K/V to the cache, and
    /// returns the logits for **every** new token, row-major `[tokens, n_vocab]`.
    ///
    /// The caller reads the last row for greedy / beam expansion; the parity
    /// tests use all rows. Internally forwards to the allocation-free
    /// [`step_into`](Self::step_into) and clones its logits scratch out.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if a token id is out of range or the
    /// decode would exceed `n_text_ctx`.
    pub fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        self.step_into(tokens)?;
        Ok(self.logits.out.clone())
    }

    /// Logits for the last token after advancing by `tokens` (greedy / beam).
    /// `tokens` must be non-empty. Forwards to [`step_into`](Self::step_into)
    /// and clones only the final row.
    pub fn step_last(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.step_into(tokens)?;
        Ok(self.last_logits_row().to_vec())
    }

    /// The allocation-free core of a decode step: runs the forward over
    /// `tokens`, appends their self-attention K/V to the cache and leaves the
    /// `[tokens, n_vocab]` logits in the reused logits scratch. Read the final
    /// row back with [`last_logits_row`](Self::last_logits_row). Every transient
    /// lives in the reused scratch, so after warm-up this allocates nothing (the
    /// capacity-stability oracle proves it).
    ///
    /// A zero-length `tokens` is a no-op (does not touch the logits scratch or
    /// advance the cache); the public [`step`](Self::step) guards it.
    // ZERO-ALLOC-BEGIN — the decode step body must not allocate on the hot path
    // (guarded by scripts/check-hot-path-allocs.sh). Every transient is reused
    // scratch; only error paths (rare) build a `format!` string.
    pub(crate) fn step_into(&mut self, tokens: &[u32]) -> Result<()> {
        // `cfg` / `w` borrow `self.model`; every buffer mutated below lives in a
        // *disjoint* field (`self.h`, `self.block`, `self.self_kv`,
        // `self.logits`), so the shared model borrow coexists with them — and
        // the distinct scratch fields let one attention call hold `&ln`,
        // `&mut attn` and `&mut block_out` at once without aliasing.
        let (cfg, w) = self.model.decoder_state();
        let d = cfg.d_model;
        let ff = cfg.ffn_dim;
        let n_head = cfg.n_text_head;
        let t = tokens.len();
        if t == 0 {
            return Ok(());
        }
        // The query offset for this step: the position count *before* it, held
        // constant across all layers and committed once at the end.
        let start = self.self_kv.positions();
        if start + t > cfg.n_text_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "whisper decoder: position {} exceeds n_text_ctx {}",
                start + t,
                cfg.n_text_ctx
            )));
        }

        // Token + positional embedding into the reused residual buffer. This is
        // shared between the CPU per-op path (which consumes `self.h` block by
        // block) and the Metal device-session path (which writes it into the
        // session's resident `h` buffer via `session.step`).
        resize_zeroed(&mut self.h, t * d);
        for (i, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            if tok >= cfg.n_vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "whisper decoder: token id {tok} >= n_vocab {}",
                    cfg.n_vocab
                )));
            }
            let posidx = start + i;
            let emb = &w.token_emb[tok * d..tok * d + d];
            let pe = &w.pos_emb[posidx * d..posidx * d + d];
            for c in 0..d {
                self.h[i * d + c] = emb[c] + pe[c];
            }
        }

        // Phase-3a Metal device path: advance the whole step device-resident in
        // ONE command-buffer submission through the pre-built session, then read
        // back the full `[t, n_vocab]` logits into the reused scratch. The
        // session tracks its own position clock; we mirror the advance on the
        // host `self_kv` (never appending to it — the K/V rows live on the GPU)
        // so `position()` / the CPU-side `start` invariant stay coherent for
        // the next step. On the CPU (and, for now, CUDA) `device_session` is
        // `None` and the per-op path below runs unchanged.
        if let Some(session) = self.device_session.as_mut() {
            let v = cfg.n_vocab;
            session.step(&self.h, t, start)?;
            resize_zeroed(&mut self.logits.out, t * v);
            let all = session.all_logits();
            debug_assert_eq!(all.len(), t * v);
            self.logits.out.copy_from_slice(all);
            self.self_kv.advance(t);
            return Ok(());
        }

        // CPU / CUDA per-op path (unchanged from before Phase 3a): build the
        // backend dispatcher for this step (Copy `backend_kind`, so the state
        // stays `Send`) and drive the per-block kernels through `Compute`.
        let compute = Compute::for_backend(self.backend_kind, super::WHISPER_HOT_OPS)?;
        let t_kv = start + t;
        for (li, layer) in w.layers.iter().enumerate() {
            self.block.ensure_residual(t, d, ff);

            // Causal self-attention over the growing cache.
            layer_norm_into(&compute, &mut self.block.ln, &self.h, t, &layer.self_ln)?;
            project_kv_into(
                &compute,
                &mut self.block.k,
                &mut self.block.v,
                &self.block.ln,
                t,
                &layer.self_attn,
            )?;
            self.self_kv.append(li, &self.block.k, &self.block.v);
            attention_from_kv_into(
                &compute,
                &mut self.block.attn,
                &self.block.ln,
                t,
                self.self_kv.k(li),
                self.self_kv.v(li),
                t_kv,
                &layer.self_attn.q,
                &layer.self_attn.out,
                n_head,
                true,
                start,
                &mut self.block.block_out,
            )?;
            add_assign(&mut self.h, &self.block.block_out)?;

            // Cross-attention over the (fixed) encoder output.
            layer_norm_into(&compute, &mut self.block.ln, &self.h, t, &layer.cross_ln)?;
            let (ck, cv) = &self.cross_kv[li];
            attention_from_kv_into(
                &compute,
                &mut self.block.attn,
                &self.block.ln,
                t,
                ck,
                cv,
                self.n_ctx,
                &layer.cross_attn.q,
                &layer.cross_attn.out,
                n_head,
                false,
                0,
                &mut self.block.block_out,
            )?;
            add_assign(&mut self.h, &self.block.block_out)?;

            // MLP.
            layer_norm_into(&compute, &mut self.block.ln, &self.h, t, &layer.mlp_ln)?;
            mlp_into(
                &compute,
                &mut self.block.mlp_h,
                &mut self.block.mlp_a,
                &mut self.block.block_out,
                &self.block.ln,
                t,
                &layer.fc1,
                &layer.fc2,
            )?;
            add_assign(&mut self.h, &self.block.block_out)?;
        }

        // Final LayerNorm into the (now-free) block `ln` buffer, then the tied
        // logits head into the logits scratch.
        layer_norm_into(&compute, &mut self.block.ln, &self.h, t, &w.ln_post)?;
        // Commit this step's positions once, after every layer was appended.
        self.self_kv.advance(t);
        project_logits_into(&compute, &mut self.logits, &self.block.ln, t, cfg, w)
    }
    // ZERO-ALLOC-END

    /// The final token's logits `[n_vocab]` from the last
    /// [`step_into`](Self::step_into) — the greedy / beam read. Must not be
    /// called before a non-empty step (the logits scratch would be empty).
    pub(crate) fn last_logits_row(&self) -> &[f32] {
        let v = self.model.config().n_vocab;
        let out = &self.logits.out;
        &out[out.len() - v..]
    }
}

/// Borrows one Whisper [`DecoderLayer`]'s weights (+ its pre-projected cross
/// K/V) as a backend-agnostic [`DecoderLayerView`] for the device-resident
/// decoder-step session ([`Compute::new_decoder_step_session`]). Whisper's
/// self-attention `k_proj` has no bias (`self_k_bias: None`); every other
/// projection carries one; cross-attention `k`/`v` are supplied as the
/// pre-projected `[n_ctx, d]` slices (`ck`/`cv`) rather than as `k`/`v` weight
/// matrices, because they are identical for every step. Called once per block
/// in `new_with_backend`, off the ZERO-ALLOC hot region.
fn decoder_layer_view<'a>(
    l: &'a DecoderLayer,
    ck: &'a [f32],
    cv: &'a [f32],
) -> DecoderLayerView<'a> {
    DecoderLayerView {
        self_ln_gamma: &l.self_ln.gamma,
        self_ln_beta: &l.self_ln.beta,
        self_q_w: &l.self_attn.q.w_t,
        self_q_bias: l.self_attn.q.bias.as_deref(),
        self_k_w: &l.self_attn.k.w_t,
        self_k_bias: l.self_attn.k.bias.as_deref(),
        self_v_w: &l.self_attn.v.w_t,
        self_v_bias: l.self_attn.v.bias.as_deref(),
        self_out_w: &l.self_attn.out.w_t,
        self_out_bias: l.self_attn.out.bias.as_deref(),
        cross_ln_gamma: &l.cross_ln.gamma,
        cross_ln_beta: &l.cross_ln.beta,
        cross_q_w: &l.cross_attn.q.w_t,
        cross_q_bias: l.cross_attn.q.bias.as_deref(),
        cross_out_w: &l.cross_attn.out.w_t,
        cross_out_bias: l.cross_attn.out.bias.as_deref(),
        cross_k: ck,
        cross_v: cv,
        mlp_ln_gamma: &l.mlp_ln.gamma,
        mlp_ln_beta: &l.mlp_ln.beta,
        fc1_w: &l.fc1.w_t,
        fc1_bias: l.fc1.bias.as_deref(),
        fc2_w: &l.fc2.w_t,
        fc2_bias: l.fc2.bias.as_deref(),
    }
}

/// `logits[T, n_vocab] = h[T, d] · token_embᵀ` (tied weights, no bias) into the
/// reused [`LogitsScratch`].
///
/// Computed as `token_emb[n_vocab, d] · hᵀ[d, T] → [n_vocab, T]` (so the huge
/// `token_emb` is never transposed), then transposed to `[T, n_vocab]`. Same
/// arithmetic and order as the former allocating `project_logits`.
// ZERO-ALLOC-BEGIN — tied-head projection into reused scratch, no allocation.
fn project_logits_into(
    compute: &Compute,
    scratch: &mut LogitsScratch,
    h: &[f32],
    t: usize,
    cfg: &WhisperConfig,
    w: &DecoderWeights,
) -> Result<()> {
    let d = cfg.d_model;
    let v = cfg.n_vocab;
    scratch.ensure(t, d, v);
    // Steady-state greedy hot path: a single query position (`t == 1`) is a
    // pure matrix-vector product `token_emb[v, d] @ h[d]` — the largest single
    // per-token decode matmul. The general `gemm` below would run it as `n = 1`
    // and fall entirely through the kernel's scalar column tail; route it
    // through the vectorized `gemv` instead. With `t == 1` the transpose `hᵀ`
    // equals `h[..d]` and the final `[T, v]` transpose is the identity, so the
    // logits land directly in `out` with zero extra memory and zero copies.
    if t == 1 {
        return compute.gemv_f32(v, d, &w.token_emb, &h[..d], None, &mut scratch.out[..v]);
    }
    // hᵀ [d, T].
    for i in 0..t {
        for c in 0..d {
            scratch.h_t[c * t + i] = h[i * d + c];
        }
    }
    // logits_t [v, T] = token_emb [v, d] @ hᵀ [d, T].
    compute.gemm_f32(
        v,
        t,
        d,
        &w.token_emb,
        &scratch.h_t,
        None,
        &mut scratch.logits_t,
    )?;
    // Transpose to [T, v].
    for row in 0..v {
        for col in 0..t {
            scratch.out[col * v + row] = scratch.logits_t[row * t + col];
        }
    }
    Ok(())
}
// ZERO-ALLOC-END

/// Synthetic tiny-decoder builders, shared with the [`super::greedy`] tests so
/// the KV-cache / greedy loops run in CI without a GGUF fixture. Everything is
/// deterministic and small (`d_model = 2`, `n_vocab = 3`); the assertions are
/// internal oracles (error variants, full-vs-cached agreement, determinism),
/// never a reference number.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use crate::whisper::WhisperModel;
    use crate::whisper::config::WhisperConfig;
    use crate::whisper::encoder::EncoderOutput;
    use crate::whisper::weights::{
        Attention, DecoderLayer, DecoderWeights, EncoderWeights, LayerNorm, Linear, WhisperWeights,
    };

    /// A tiny valid config with `n_layer` decoder blocks (`d_model = 2`,
    /// `n_vocab = 3`, `n_text_ctx = 8`, single head).
    pub(crate) fn tiny_cfg(n_layer: usize) -> WhisperConfig {
        WhisperConfig {
            n_mels: 80,
            d_model: 2,
            n_audio_ctx: 4,
            n_audio_head: 1,
            n_audio_layer: 0,
            n_text_ctx: 8,
            n_text_head: 1,
            n_text_layer: n_layer,
            n_vocab: 3,
            ffn_dim: 2,
            eot: 0,
            decoder_start_ids: vec![1],
        }
    }

    /// Deterministic encoder hidden states `[n_ctx, d_model]`.
    pub(crate) fn tiny_encoder(d_model: usize, n_ctx: usize) -> EncoderOutput {
        let hidden = (0..n_ctx * d_model)
            .map(|i| 0.05 * i as f32 - 0.1)
            .collect();
        EncoderOutput {
            hidden,
            n_ctx,
            d_model,
        }
    }

    /// Decoder weights matching `cfg` (token / pos embeddings, `n_text_layer`
    /// blocks, unit final LayerNorm), all deterministic small values.
    pub(crate) fn tiny_weights(cfg: &WhisperConfig) -> DecoderWeights {
        let d = cfg.d_model;
        let token_emb = (0..cfg.n_vocab * d).map(|i| 0.1 * i as f32 - 0.2).collect();
        let pos_emb = (0..cfg.n_text_ctx * d)
            .map(|i| 0.05 - 0.02 * i as f32)
            .collect();
        let layers = (0..cfg.n_text_layer)
            .map(|_| tiny_layer(d, cfg.ffn_dim))
            .collect();
        DecoderWeights {
            token_emb,
            pos_emb,
            layers,
            ln_post: unit_ln(d),
        }
    }

    /// Minimal encoder weights matching `cfg`. The decoder tests never run the
    /// encoder, so every tensor is zero-filled at the correct shape (and there
    /// are no encoder layers — `tiny_cfg` sets `n_audio_layer = 0`).
    fn tiny_encoder_weights(cfg: &WhisperConfig) -> EncoderWeights {
        let d = cfg.d_model;
        EncoderWeights {
            conv1_w: vec![0.0; d * cfg.n_mels * 3],
            conv1_b: vec![0.0; d],
            conv2_w: vec![0.0; d * d * 3],
            conv2_b: vec![0.0; d],
            pos_emb: vec![0.0; cfg.n_audio_ctx * d],
            layers: Vec::new(),
            ln_post: unit_ln(d),
        }
    }

    /// A tiny loaded [`WhisperModel`] (config + weights) wrapped in an [`Arc`],
    /// ready for [`WhisperModel::decoder`]. This is the construction the decoder
    /// tests use now that [`super::DecoderState`] owns its model.
    pub(crate) fn tiny_model(n_layer: usize) -> Arc<WhisperModel> {
        let config = tiny_cfg(n_layer);
        let weights = WhisperWeights {
            encoder: tiny_encoder_weights(&config),
            decoder: tiny_weights(&config),
        };
        Arc::new(WhisperModel::new_for_test(config, weights))
    }

    /// Like [`tiny_model`] but with an explicit `n_text_ctx` (decoder positional
    /// length), so the variable-length regression test can use a text window
    /// larger than the KV reserve hint. Positional embeddings are sized to the
    /// chosen `n_text_ctx` by [`tiny_weights`].
    pub(crate) fn tiny_model_ctx(n_layer: usize, n_text_ctx: usize) -> Arc<WhisperModel> {
        let mut config = tiny_cfg(n_layer);
        config.n_text_ctx = n_text_ctx;
        let weights = WhisperWeights {
            encoder: tiny_encoder_weights(&config),
            decoder: tiny_weights(&config),
        };
        Arc::new(WhisperModel::new_for_test(config, weights))
    }

    /// Unit-scale / zero-shift LayerNorm of width `d`.
    fn unit_ln(d: usize) -> LayerNorm {
        LayerNorm {
            gamma: vec![1.0; d],
            beta: vec![0.0; d],
        }
    }

    /// A `Linear [in, out]` from an explicit row-major `w_t` and optional bias.
    fn lin(
        w_t: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Linear {
        Linear {
            w_t,
            in_features,
            out_features,
            bias,
        }
    }

    /// A deterministic `rows * cols` weight buffer with small distinct values.
    fn rect(rows: usize, cols: usize, base: f32) -> Vec<f32> {
        (0..rows * cols).map(|i| base + 0.03 * i as f32).collect()
    }

    /// A `[d, d]` attention block with deterministic projections (`k` has no
    /// bias, matching Whisper).
    fn tiny_attn(d: usize) -> Attention {
        Attention {
            q: lin(rect(d, d, 0.10), d, d, Some(vec![0.01; d])),
            k: lin(rect(d, d, -0.07), d, d, None),
            v: lin(rect(d, d, 0.05), d, d, Some(vec![0.02; d])),
            out: lin(rect(d, d, -0.04), d, d, Some(vec![0.0; d])),
        }
    }

    /// One decoder block (self-attn → cross-attn → MLP) with deterministic
    /// weights; `ff` is the MLP inner width.
    fn tiny_layer(d: usize, ff: usize) -> DecoderLayer {
        DecoderLayer {
            self_ln: unit_ln(d),
            self_attn: tiny_attn(d),
            cross_ln: unit_ln(d),
            cross_attn: tiny_attn(d),
            mlp_ln: unit_ln(d),
            fc1: lin(rect(d, ff, 0.06), d, ff, Some(vec![0.0; ff])),
            fc2: lin(rect(ff, d, -0.05), ff, d, Some(vec![0.0; d])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{tiny_encoder, tiny_model, tiny_model_ctx};
    use super::*;

    #[test]
    fn new_rejects_encoder_dim_mismatch() {
        let model = tiny_model(0);
        // Encoder hidden width differs from the config d_model. (DecoderState is
        // not Debug, so match instead of unwrap_err.)
        let enc = tiny_encoder(model.config().d_model + 1, 4);
        assert!(matches!(
            model.decoder(&enc),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn step_rejects_out_of_range_token() {
        let model = tiny_model(0);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();
        // 99 >= n_vocab (3): guarded before the embedding slice would panic.
        let err = st.step(&[99]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn step_rejects_exceeding_n_text_ctx() {
        let model = tiny_model(0);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();
        // n_text_ctx = 8; an n_text_ctx+1 step overflows (ids stay in vocab).
        let toks = vec![1u32; model.config().n_text_ctx + 1];
        let err = st.step(&toks).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn empty_step_returns_empty_and_does_not_advance() {
        let model = tiny_model(0);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();
        assert!(st.step(&[]).unwrap().is_empty());
        assert_eq!(st.position(), 0);
    }

    /// Full-sequence `step` must reproduce the token-by-token cached path for
    /// the last position — the KV-cache invariant the parity test owns at real
    /// scale (verified here at synthetic scale, both with and without a layer).
    fn assert_full_matches_cached(n_layer: usize) {
        let model = tiny_model(n_layer);
        let v = model.config().n_vocab;
        let enc = tiny_encoder(model.config().d_model, 4);
        let (a, b) = (1u32, 2u32);

        let mut st = model.decoder(&enc).unwrap();
        let full = st.step(&[a, b]).unwrap();
        assert_eq!(full.len(), 2 * v);
        let full_last = &full[v..2 * v];

        // Token-by-token after reset must land on the same last-position logits.
        st.reset();
        assert_eq!(st.position(), 0);
        let _ = st.step_last(&[a]).unwrap();
        let cached_last = st.step_last(&[b]).unwrap();
        assert_eq!(cached_last.len(), v);
        for (i, (&f, &c)) in full_last.iter().zip(&cached_last).enumerate() {
            assert!(
                (f - c).abs() < 1e-4,
                "layer {n_layer} idx {i}: full {f} vs cached {c}"
            );
        }
    }

    #[test]
    fn full_forward_matches_cached_stepping() {
        assert_full_matches_cached(0);
        assert_full_matches_cached(1);
    }

    #[test]
    fn reset_and_replay_is_bit_identical() {
        let model = tiny_model(1);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();

        let run1 = st.step(&[1, 2, 1]).unwrap();
        st.reset();
        let run2 = st.step(&[1, 2, 1]).unwrap();
        // Same code path, same inputs, same accumulation order → bit-identical.
        assert_eq!(run1, run2);
    }

    #[test]
    fn step_last_is_final_row_of_step() {
        let model = tiny_model(1);
        let v = model.config().n_vocab;
        let enc = tiny_encoder(model.config().d_model, 4);

        let mut st = model.decoder(&enc).unwrap();
        let all = st.step(&[1, 2]).unwrap();
        let last_slice = all[v..2 * v].to_vec();

        st.reset();
        let last = st.step_last(&[1, 2]).unwrap();
        assert_eq!(last, last_slice);
    }

    /// Compile-time proof that the promoted cache and the whole decode state are
    /// both thread-transferable — the point of dropping the lifetime and owning
    /// the model via `Arc` (M1-08 streaming foundation).
    fn assert_send<T: Send>() {}

    #[test]
    fn kv_cache_and_decoder_state_are_send() {
        assert_send::<vokra_core::KvCache>();
        assert_send::<DecoderState>();
    }

    #[test]
    fn decoder_state_moves_across_threads_bit_identically() {
        use std::thread;

        // A fixed prefix decoded on the main thread and on an independent,
        // identically-constructed state moved into a worker thread must agree
        // bit-for-bit — the cross-thread oracle for the ownable `Send` state.
        let prefix = [1u32, 2, 1];

        let model = tiny_model(1);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut main_state = model.decoder(&enc).unwrap();
        let main_logits = main_state.step(&prefix).unwrap();

        let worker_model = tiny_model(1);
        let worker_enc = tiny_encoder(worker_model.config().d_model, 4);
        let mut worker_state = worker_model.decoder(&worker_enc).unwrap();
        // Moving `worker_state` into the closure only compiles if it is `Send`.
        let worker_logits = thread::spawn(move || worker_state.step(&prefix).unwrap())
            .join()
            .expect("worker thread panicked");

        assert_eq!(
            main_logits, worker_logits,
            "cross-thread decode of the same prefix diverged"
        );
    }

    /// Gathers every reusable-scratch capacity plus the KV cache's reserved
    /// position count. A [`Vec`] reallocates *iff* a push/resize exceeds its
    /// capacity, so a capacity that never changes across steps is a direct proof
    /// that no reallocation (no `malloc`) happened.
    fn all_capacities(st: &DecoderState) -> Vec<usize> {
        let mut caps = st.block.capacities();
        caps.extend_from_slice(&st.logits.capacities());
        caps.push(st.h.capacity());
        caps.push(st.self_kv.capacity_positions());
        caps
    }

    /// Zero-malloc oracle (sub-part 3): after a warm-up, every scratch buffer's
    /// capacity — and the KV cache's reserved length — must stay **constant**
    /// across a run of further single-token decode steps, even as the cache (and
    /// thus each attention's `t_kv`) grows. This is the capacity-stability proof
    /// that the reused buffers eliminate the autoregressive hot path's `malloc`.
    #[test]
    fn scratch_capacity_is_stable_across_decode_steps() {
        // Two layers so the block scratch is provably reused *across layers*
        // within a step, not just across steps.
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();

        // Warm up to steady state (single-token steps, as greedy decode does
        // after the forced prefix).
        st.step_into(&[1]).unwrap();
        st.step_into(&[2]).unwrap();
        let before = all_capacities(&st);

        // The hot loop: more single-token steps. `t_kv = positions + 1` grows
        // every step, so the scores / probs / key-transpose / value buffers are
        // resized upward each time — yet must stay within the reserve.
        for tok in [1u32, 2, 1, 2, 1] {
            st.step_into(&[tok]).unwrap();
        }
        let after = all_capacities(&st);

        assert_eq!(
            before, after,
            "a reusable scratch buffer reallocated during the decode loop \
             (before {before:?}, after {after:?}) — the hot path is not malloc-free"
        );
        // Guard against a vacuous pass: the cache really did grow (so the
        // t_kv-dependent buffers were exercised at increasing sizes).
        assert_eq!(st.position(), 7, "decode loop did not advance as expected");
    }

    /// `step_into` + `last_logits_row` (the greedy read) must equal the last row
    /// of the allocating `step` — same forward, same scratch, so bit-identical.
    #[test]
    fn step_into_last_row_matches_step() {
        let model = tiny_model(1);
        let v = model.config().n_vocab;
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();

        let all = st.step(&[1, 2]).unwrap();
        let step_last_row = all[v..2 * v].to_vec();

        st.reset();
        st.step_into(&[1, 2]).unwrap();
        assert_eq!(st.last_logits_row(), step_last_row.as_slice());
    }

    /// Variable-length I/O regression (M1-04 sub-part 2): two DIFFERENT-length
    /// prefixes decode through the SAME reset state, and the self-attention KV
    /// cache is seeded to a hint *below* the static `n_text_ctx` max (proving we
    /// no longer pre-allocate the worst case) yet still grows to serve a decode
    /// longer than that hint.
    #[test]
    fn variable_length_prefixes_reuse_state_and_cache_grows() {
        // A text window larger than the reserve hint, so the seeded capacity is
        // strictly below `n_text_ctx`.
        let n_text_ctx = SELF_KV_RESERVE_HINT * 2;
        let model = tiny_model_ctx(1, n_text_ctx);
        let v = model.config().n_vocab;
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut st = model.decoder(&enc).unwrap();

        // (1) A SHORT prefix decodes fine and leaves the cache at the hint.
        let short = [1u32, 2, 1];
        let short_logits = st.step(&short).unwrap();
        assert_eq!(short_logits.len(), short.len() * v);
        assert_eq!(st.position(), short.len());
        // No static-max pre-alloc: the reserved capacity is strictly below the
        // full text window (it is the hint, which the short decode did not
        // exceed).
        assert!(
            st.self_kv.capacity_positions() < n_text_ctx,
            "self-attention KV cache pre-allocated the static n_text_ctx max \
             (capacity {} vs n_text_ctx {n_text_ctx})",
            st.self_kv.capacity_positions()
        );

        // (2) Reset and decode a DIFFERENT, LONGER prefix through the same state.
        st.reset();
        assert_eq!(st.position(), 0);
        let long_len = SELF_KV_RESERVE_HINT + 4; // > hint, <= n_text_ctx
        assert!(long_len > SELF_KV_RESERVE_HINT && long_len <= n_text_ctx);
        let long: Vec<u32> = (0..long_len).map(|i| (i % v) as u32).collect();
        let long_logits = st.step(&long).unwrap();
        assert_eq!(long_logits.len(), long_len * v);
        assert_eq!(st.position(), long_len);
        // The cache grew past the initial hint to serve the longer decode.
        assert!(
            st.self_kv.capacity_positions() >= long_len,
            "KV cache did not grow to hold the longer decode (capacity {}, need {long_len})",
            st.self_kv.capacity_positions()
        );
    }
}
