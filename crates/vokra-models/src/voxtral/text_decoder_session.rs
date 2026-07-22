//! Voxtral text-decoder session — Mistral decoder with per-block KV cache.
//!
//! # Purpose (M3-10-T13 autoregressive greedy decode)
//!
//! [`TextDecoderSession`] owns:
//! - a reference to a loaded [`TextDecoder`] (weights);
//! - a reference to the parent [`VoxtralConfig`] (hparams);
//! - a self-attention [`KvCache`] (one per-layer buffer of width
//!   `n_head_kv * head_dim`);
//! - a reusable [`StepScratch`] the block forward writes into;
//! - a [`Compute`] dispatcher (CPU foundation; a GPU
//!   `VoxtralDecodeSession` is reserved as a follow-up seam — see the
//!   streaming module's `allow_device_session` slot).
//!
//! # Zero-alloc hot path (FR-EX-05)
//!
//! Every scratch buffer is reserved once at [`TextDecoderSession::new`] and
//! reused across steps. The KV cache is seeded to a small hint
//! (`SELF_KV_RESERVE_HINT`) and grows amortically for longer decodes; a
//! decode past the hint reallocates once (Vec doubling) but the per-step
//! compute scratch stays constant-size.
//!
//! # No silent fallback (FR-EX-08)
//!
//! - Every out-of-range token / position surfaces as
//!   [`VokraError::InvalidArgument`] with the offending value named.
//! - A `0`-sentinel config (from the shape-only converter path) is rejected
//!   at [`TextDecoderSession::new`] — never a silent default.

use vokra_core::{BackendKind, KvCache, Result, VokraError};

use crate::compute::Compute;

use super::VOXTRAL_HOT_OPS;
use super::text_decoder::{self, StepScratch};
use super::{TextDecoder, VoxtralConfig};

/// Initial reservation hint for the self-attention KV cache, in positions
/// (mirrors `whisper::decoder::SELF_KV_RESERVE_HINT` — typical decodes are
/// short, so we don't pre-allocate the worst-case window).
const SELF_KV_RESERVE_HINT: usize = 64;

/// A single-decode session: greedy loop calls `step_into` then reads the
/// last-row logits back through `last_logits_row` (allocation-free hot path).
///
/// # Lifetime
///
/// Borrows `&'m TextDecoder` and `&'m VoxtralConfig` — the caller keeps the
/// loaded model alive for the session's lifetime. This matches the
/// scaffold-only borrow pattern used elsewhere in the Voxtral module (see
/// [`super::AsrHead`]). A follow-up ticket can promote the session to owning
/// an `Arc<VoxtralModel>` (mirroring Whisper's `DecoderState`) when the
/// streaming layer needs to move it across threads.
pub struct TextDecoderSession<'m> {
    config: &'m VoxtralConfig,
    decoder: &'m TextDecoder,
    compute: Compute,
    kv_cache: KvCache,
    scratch: StepScratch,
    /// Number of committed decode positions (same as `kv_cache.positions()`
    /// but cached to avoid a per-call kv_cache method invocation).
    position: usize,
    /// Cached hyperparameter dims for `last_logits_row` / bounds checks.
    n_layer: usize,
    d: usize,
    kv_hidden: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_t_kv: usize,
}

impl<'m> TextDecoderSession<'m> {
    /// Constructs a new decoder session on `backend`.
    ///
    /// The scratch buffers are sized so a decode step of up to 512 tokens
    /// (a typical prefix + one greedy step) does not reallocate. Longer
    /// steps grow the scratch amortically (single-token greedy stays
    /// within the reserve).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] if the config is the shape-only
    ///   `0`-sentinel path (missing GQA head split, RoPE base, RMSNorm eps,
    ///   or vocab size).
    /// - [`VokraError::UnsupportedOp`] if `backend` does not cover the
    ///   Voxtral hot ops (FR-EX-08).
    /// - [`VokraError::BackendUnavailable`] if `backend` is not built into
    ///   this binary.
    pub fn new(
        config: &'m VoxtralConfig,
        decoder: &'m TextDecoder,
        backend: BackendKind,
    ) -> Result<Self> {
        // Reject 0-sentinel config — every downstream method depends on
        // these values (FR-EX-08, no silent default).
        let n_layer = config.text.n_layer;
        let d = config.text.hidden_dim;
        let vocab_size = config.text.vocab_size;
        let n_head_q = config.text.n_head_q;
        let n_head_kv = config.text.n_head_kv;
        let ffn_dim = config.text.ffn_dim;
        if n_layer == 0 || d == 0 || vocab_size == 0 || n_head_q == 0 || n_head_kv == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::TextDecoderSession: config carries 0-sentinel — re-convert with a full \
                 VoxtralConfig (FR-EX-08 — no silent default)."
                    .into(),
            ));
        }
        if n_head_q % n_head_kv != 0 {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::TextDecoderSession: n_head_q ({n_head_q}) must be divisible by n_head_kv ({n_head_kv}) — GQA"
            )));
        }
        // Explicit-or-derived per-head width (see
        // `TextDecoderConfig::head_dim`) — NOT `d / n_head_q`: the real mini
        // decouples the two (q_hidden 4096 vs d 3072).
        let head_dim = config.text.head_dim();
        if head_dim == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::TextDecoderSession: head_dim resolves to 0 — re-convert with a \
                 converter that writes vokra.voxtral.text_decoder.head_dim (FR-EX-08 — no \
                 silent default)."
                    .into(),
            ));
        }
        let q_hidden = n_head_q * head_dim;
        let kv_hidden = n_head_kv * head_dim;
        // Residency-agnostic: a mapped decoder keeps `blocks` empty on
        // purpose (`MappedTextBlocks` is the source), so counting `blocks`
        // directly would reject it as "unloaded".
        if decoder.n_layer() != n_layer {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::TextDecoderSession: loaded blocks {} != config n_layer {n_layer}",
                decoder.n_layer()
            )));
        }

        let compute = Compute::for_backend(backend, VOXTRAL_HOT_OPS)?;

        // Reserve for a typical prefix step (up to a few tokens) plus the
        // greedy loop (single-token steps). Reserve for a larger t_kv so
        // the attention scratch stays constant for a moderate decode.
        // (The scratch will grow if a decode exceeds these bounds.)
        let reserve_t_q = 8usize;
        let reserve_t_kv = config.text.n_ctx.min(256).max(reserve_t_q);
        let kv_cache = KvCache::with_reserve(
            n_layer,
            kv_hidden,
            SELF_KV_RESERVE_HINT.min(config.text.n_ctx.max(1)),
        );
        let scratch = StepScratch::with_reserve(
            reserve_t_q,
            d,
            q_hidden,
            kv_hidden,
            head_dim,
            ffn_dim,
            vocab_size,
            reserve_t_kv,
        );

        Ok(Self {
            config,
            decoder,
            compute,
            kv_cache,
            scratch,
            position: 0,
            n_layer,
            d,
            kv_hidden,
            head_dim,
            ffn_dim,
            vocab_size,
            max_t_kv: reserve_t_kv,
        })
    }

    /// Convenience constructor on the CPU backend (zero-dep default).
    pub fn cpu(config: &'m VoxtralConfig, decoder: &'m TextDecoder) -> Result<Self> {
        Self::new(config, decoder, BackendKind::Cpu)
    }

    /// Number of committed positions so far.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Rewinds the session for a fresh decode of the same model.
    pub fn reset(&mut self) {
        self.kv_cache.reset();
        self.position = 0;
    }

    /// Runs one decode step: appends every layer's K/V rows for `tokens`
    /// (using RoPE at position `self.position + i`), then leaves the
    /// `[t, vocab_size]` logits in the reused scratch. Read the final row
    /// with [`last_logits_row`](Self::last_logits_row).
    ///
    /// A zero-length `tokens` is a no-op.
    ///
    /// # Errors
    ///
    /// See [`text_decoder::forward_step`] for the full error taxonomy.
    pub fn step_into(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let position_offset = self.position;
        text_decoder::forward_step(
            &self.compute,
            self.config,
            self.decoder,
            &mut self.scratch,
            &mut self.kv_cache,
            tokens,
            position_offset,
        )?;
        self.position += tokens.len();
        // Update the tracked t_kv max so the caller can size follow-up
        // scratch (informational; the scratch itself grew inside forward_step
        // if needed).
        if self.position > self.max_t_kv {
            self.max_t_kv = self.position;
        }
        Ok(())
    }

    /// Runs one decode step where the hidden state is a caller-supplied raw
    /// **embedding** rather than a token id sequence. This is the entry point
    /// the audio-conditioned ASR path (M3-10 Wave 8) uses to feed the audio
    /// adapter's soft-prefix output straight into the decoder residual stream.
    ///
    /// `prefix_embed` is `[t_prefix, hidden_dim]` row-major.
    /// A zero-length `prefix_embed` is a no-op.
    ///
    /// # Errors
    ///
    /// See [`text_decoder::forward_step_with_embed_prefix`] for the full error
    /// taxonomy — the same shape / config / n_ctx checks apply.
    pub fn step_into_with_embed_prefix(
        &mut self,
        prefix_embed: &[f32],
        t_prefix: usize,
    ) -> Result<()> {
        if t_prefix == 0 {
            return Ok(());
        }
        let position_offset = self.position;
        text_decoder::forward_step_with_embed_prefix(
            &self.compute,
            self.config,
            self.decoder,
            &mut self.scratch,
            &mut self.kv_cache,
            prefix_embed,
            t_prefix,
            position_offset,
        )?;
        self.position += t_prefix;
        if self.position > self.max_t_kv {
            self.max_t_kv = self.position;
        }
        Ok(())
    }

    /// Returns the logits for the last position (`[vocab_size]`) — the
    /// greedy / beam read. Must not be called before a non-empty
    /// [`step_into`](Self::step_into) (the logits scratch would be empty).
    #[must_use]
    pub fn last_logits_row(&self) -> &[f32] {
        let v = self.vocab_size;
        let out = &self.scratch.logits;
        // If step_into wrote at least one row, its length is a multiple of v.
        &out[out.len() - v..]
    }

    /// Returns the full `[t, vocab_size]` logits from the last step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        &self.scratch.logits
    }

    /// Backend name (`"cpu"` / `"metal"` / `"cuda"`).
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.compute.backend_name()
    }

    /// Number of decoder layers (informational — matches the config).
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    /// Config hidden width — for callers that want to size external
    /// projections (e.g. audio adapter follow-up).
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.d
    }

    /// Vocabulary size — for callers checking argmax bounds.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Maximum sequence length the underlying config allows. Callers use
    /// this to bound their own decode budget (e.g. beam search truncating
    /// `max_new_tokens` to `n_ctx - initial_tokens.len()`).
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.config.text.n_ctx
    }

    // Silence unused-field lints for the audio-adapter follow-up. These
    // dimensions are read by the future GPU session ctor + external
    // conditioning path; no runtime cost until then.
    #[doc(hidden)]
    pub fn kv_hidden(&self) -> usize {
        self.kv_hidden
    }
    #[doc(hidden)]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    #[doc(hidden)]
    pub fn ffn_dim(&self) -> usize {
        self.ffn_dim
    }

    // ---------------------------------------------------------------------
    // KV cache snapshot / restore (beam search + n-best decode, M3-10)
    // ---------------------------------------------------------------------

    /// Snapshots the current session's KV cache + position clock into a
    /// [`TextDecoderKvSnapshot`] the caller can [`Clone`] freely and later
    /// hand back to [`kv_restore`](Self::kv_restore) to rewind this session
    /// to the exact state at the moment of the call.
    ///
    /// Used by the Voxtral beam search + n-best decoder to branch a single
    /// session across candidate hypotheses without recomputing the shared
    /// prefix (M3-10).
    ///
    /// # Cost
    ///
    /// Deep-copies every layer's `k` / `v` buffer through
    /// [`KvCache::clone`] — `O(n_layer * position * kv_hidden)` — plus the
    /// (small) `position` usize. Intended for the beam-search hot path
    /// (per-step branching at beam widths 1..~16), not for the general
    /// per-token hot path (which the plain [`step_into`](Self::step_into)
    /// serves).
    ///
    /// The transient step-scratch buffers (holding the per-step logits
    /// row etc.) are **not** captured — a snapshot only records the
    /// persistent decoder state (KV cache, position clock). A
    /// [`kv_restore`](Self::kv_restore) followed by a fresh
    /// [`step_into`](Self::step_into) recomputes the scratch, so callers
    /// must not rely on [`last_logits_row`](Self::last_logits_row)
    /// immediately after a restore — call `step_into` first, exactly as
    /// one would after [`reset`](Self::reset).
    #[must_use]
    pub fn kv_snapshot(&self) -> TextDecoderKvSnapshot {
        TextDecoderKvSnapshot {
            kv_cache: self.kv_cache.clone(),
            position: self.position,
            max_t_kv: self.max_t_kv,
        }
    }

    /// Restores the session's KV cache + position clock from a previous
    /// [`kv_snapshot`](Self::kv_snapshot).
    ///
    /// After this call, a fresh [`step_into`](Self::step_into) resumes the
    /// decode from the snapshotted state — bit-identical to what would happen
    /// if the caller had never diverged. This is the "branch" primitive the
    /// beam-search inner loop uses to explore several candidate continuations
    /// from a shared prefix.
    ///
    /// The transient step-scratch buffers are **not** part of the snapshot
    /// (see the [`kv_snapshot`](Self::kv_snapshot) docstring): callers must
    /// call `step_into` before reading [`last_logits_row`](Self::last_logits_row).
    ///
    /// The snapshot is consumed by value (moved) so the caller cannot
    /// accidentally re-use it on a different session — pair with `Clone`
    /// if two restores from the same origin are needed.
    pub fn kv_restore(&mut self, snapshot: TextDecoderKvSnapshot) {
        self.kv_cache = snapshot.kv_cache;
        self.position = snapshot.position;
        // The scratch's `max_t_kv` bookkeeping tracks the largest KV window
        // the caller has ever asked for; keeping the pre-snapshot value is
        // strictly conservative (we never shrink it), and matches what the
        // step_into path would have done at the moment the snapshot was
        // taken.
        if snapshot.max_t_kv > self.max_t_kv {
            self.max_t_kv = snapshot.max_t_kv;
        }
    }
}

// -----------------------------------------------------------------------------
// TextDecoderKvSnapshot — opaque handle
// -----------------------------------------------------------------------------

/// Opaque snapshot of a [`TextDecoderSession`]'s persistent state — the
/// per-layer KV cache and the position clock.
///
/// Constructed by [`TextDecoderSession::kv_snapshot`] and consumed by
/// [`TextDecoderSession::kv_restore`]. The type is [`Clone`] (deep-copies the
/// KV cache) so a beam-search caller can branch a single "prefix" state
/// across multiple candidate hypotheses.
///
/// # Layout
///
/// The struct is a value type — no interior mutability, no reference to the
/// originating session. Two snapshots are independent objects: cloning one
/// and evolving each half separately does not alias.
///
/// # Compatibility
///
/// A snapshot is only meaningful when restored into a session that shares the
/// same model / config. This is enforced *at run time* by shape checks
/// inside the subsequent [`step_into`](TextDecoderSession::step_into) call
/// (the KV cache carries its own `width` and layer count); a mismatched
/// restore leaves the session in a legal but semantically inconsistent state
/// and the very next step will surface the mismatch as an
/// [`InvalidArgument`](vokra_core::VokraError::InvalidArgument) or panic in
/// debug builds. Callers should treat snapshots as **model-tied handles**.
#[derive(Clone)]
pub struct TextDecoderKvSnapshot {
    /// Deep-cloned KV cache (per-layer `Vec<f32>` k/v pairs + committed
    /// positions + hidden width).
    kv_cache: vokra_core::KvCache,
    /// Position clock (cached mirror of `kv_cache.positions()`, kept for the
    /// same reason [`TextDecoderSession`] mirrors it — avoiding a per-call
    /// method invocation on the hot path).
    position: usize,
    /// Largest KV window bookkeeping — same purpose as
    /// [`TextDecoderSession::max_t_kv`].
    max_t_kv: usize,
}

impl TextDecoderKvSnapshot {
    /// The committed position count at the moment the snapshot was taken.
    /// Useful for external bookkeeping (e.g. asserting a beam-search branch
    /// point matches the expected prefix length).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }
}

// -----------------------------------------------------------------------------
// Greedy decode driver
// -----------------------------------------------------------------------------

/// Default cap on generated tokens when the caller does not pass one.
/// Chosen to match Whisper's `n_text_ctx / 2 = 224` for a comparable
/// per-utterance budget; still bounded by `config.text.n_ctx`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 224;

/// Greedy argmax over a logits row.
fn argmax(row: &[f32]) -> u32 {
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

/// Greedy-decode starting from `start_ids`, stopping on `eos` (which IS
/// included in the result) or after `max_new` new tokens. Returns the
/// generated token ids (start_ids are NOT included).
///
/// The session is `reset()` at the top so a second call reproduces the
/// first.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `start_ids` is empty or any id is
///   out of range (surfaced from `step_into`).
pub fn greedy_decode(
    session: &mut TextDecoderSession<'_>,
    start_ids: &[u32],
    eos: u32,
    max_new: usize,
) -> Result<Vec<u32>> {
    if start_ids.is_empty() {
        return Err(VokraError::InvalidArgument(
            "voxtral::greedy_decode: start_ids must not be empty".into(),
        ));
    }
    session.reset();
    session.step_into(start_ids)?;
    let mut generated = Vec::with_capacity(max_new.min(64));
    let cap = max_new.max(1);
    let n_ctx_cap = session.config.text.n_ctx.saturating_sub(start_ids.len());
    let cap = cap.min(n_ctx_cap);
    for _ in 0..cap {
        let next = argmax(session.last_logits_row());
        generated.push(next);
        if next == eos {
            break;
        }
        session.step_into(&[next])?;
    }
    Ok(generated)
}

/// Audio-conditioned greedy decode (M3-10 Wave 8): prefill the decoder with a
/// caller-supplied soft-prefix embedding sequence (from the audio adapter),
/// then the standard `[bos_id]` prefix, then greedy-loop until `eos_id` or
/// `max_new` tokens. Returns only the generated tokens (prefix embed + BOS
/// are NOT included).
///
/// This is the counterpart of [`greedy_decode`] for the audio-conditioned
/// path: the tokens the caller sees are conditioned on the audio via the
/// adapter's projected representation, rather than the LM-only prior from
/// BOS. The session is `reset()` at the top so a second call reproduces the
/// first.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `t_prefix == 0` (use plain
///   [`greedy_decode`] instead) or `prefix_embed.len() != t_prefix *
///   hidden_dim` (surfaced from
///   [`step_into_with_embed_prefix`](TextDecoderSession::step_into_with_embed_prefix)).
pub fn greedy_decode_with_prefix(
    session: &mut TextDecoderSession<'_>,
    prefix_embed: &[f32],
    t_prefix: usize,
    bos_id: u32,
    eos: u32,
    max_new: usize,
) -> Result<Vec<u32>> {
    if t_prefix == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::greedy_decode_with_prefix: t_prefix must be > 0 (use greedy_decode instead)"
                .into(),
        ));
    }
    session.reset();
    session.step_into_with_embed_prefix(prefix_embed, t_prefix)?;
    session.step_into(&[bos_id])?;
    let mut generated = Vec::with_capacity(max_new.min(64));
    let cap = max_new.max(1);
    // n_ctx budget accounts for the prefix + the bos token already consumed.
    let consumed = t_prefix.saturating_add(1);
    let n_ctx_cap = session.config.text.n_ctx.saturating_sub(consumed);
    let cap = cap.min(n_ctx_cap);
    for _ in 0..cap {
        let next = argmax(session.last_logits_row());
        generated.push(next);
        if next == eos {
            break;
        }
        session.step_into(&[next])?;
    }
    Ok(generated)
}

/// Segmented greedy decode for the **trained transcription-prompt layout**
/// (P2 cc-05/07 follow-up): seed the session with
///
/// ```text
/// step_into(pre_tokens) → step_into_with_embed_prefix(audio rows) → step_into(post_tokens)
/// ```
///
/// then greedy-loop from the post-segment's last logits row until `eos` or
/// `max_new` tokens. This is the runtime replay of upstream
/// `VoxtralForConditionalGeneration.forward`'s `masked_scatter` semantics:
/// the audio soft-prefix rows occupy the `[AUDIO]` placeholder positions
/// between `pre_tokens` (`[<s>, [INST], [BEGIN_AUDIO]]`) and `post_tokens`
/// (`[[/INST], "lang:xx"…, [TRANSCRIBE]]`).
///
/// Returns only the generated tokens (no prompt segment is included); a
/// decode that stopped on `eos` has it as the last element. The session is
/// [`TextDecoderSession::reset`] at the top so a repeat call reproduces the
/// first.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `t_prefix == 0` (the trained layout
///   is only defined over an audio run — use [`greedy_decode`] /
///   [`greedy_decode_with_prefix`] otherwise) or `post_tokens` is empty
///   (the first generated token is sampled from the logits AFTER the
///   post-audio segment; an empty segment has no defined sampling point in
///   this layout);
/// - any error the underlying [`TextDecoderSession::step_into`] /
///   [`TextDecoderSession::step_into_with_embed_prefix`] surfaces
///   (out-of-vocab id, shape mismatch, exceeded `n_ctx`, backend error).
pub fn greedy_decode_with_segments(
    session: &mut TextDecoderSession<'_>,
    pre_tokens: &[u32],
    prefix_embed: &[f32],
    t_prefix: usize,
    post_tokens: &[u32],
    eos: u32,
    max_new: usize,
) -> Result<Vec<u32>> {
    if t_prefix == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::greedy_decode_with_segments: t_prefix must be > 0 — the transcription \
             layout is only defined over an audio soft-prefix run (use greedy_decode / \
             greedy_decode_with_prefix instead)"
                .into(),
        ));
    }
    if post_tokens.is_empty() {
        return Err(VokraError::InvalidArgument(
            "voxtral::greedy_decode_with_segments: post_tokens must not be empty — the first \
             generated token is sampled from the logits after the post-audio segment \
             ([/INST]…[TRANSCRIBE])"
                .into(),
        ));
    }
    session.reset();
    session.step_into(pre_tokens)?; // empty pre is a documented no-op
    session.step_into_with_embed_prefix(prefix_embed, t_prefix)?;
    session.step_into(post_tokens)?;
    let mut generated = Vec::with_capacity(max_new.min(64));
    let cap = max_new.max(1);
    let consumed = pre_tokens
        .len()
        .saturating_add(t_prefix)
        .saturating_add(post_tokens.len());
    let n_ctx_cap = session.config.text.n_ctx.saturating_sub(consumed);
    let cap = cap.min(n_ctx_cap);
    for _ in 0..cap {
        let next = argmax(session.last_logits_row());
        generated.push(next);
        if next == eos {
            break;
        }
        session.step_into(&[next])?;
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};

    fn tiny_cfg() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 4,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                head_dim: 0,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 4,
                n_ctx: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    /// Build a tiny hand-crafted TextDecoder with deterministic non-zero weights.
    /// The values are small enough to keep softmax stable and produce a
    /// finite, deterministic argmax without exploding logits.
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let n_head_q = cfg.text.n_head_q;
        let n_head_kv = cfg.text.n_head_kv;
        let head_dim = d / n_head_q;
        let kv_hidden = n_head_kv * head_dim;

        // Deterministic weight initialisation: small distinct values so
        // GEMMs actually mix rows (identity-scaled weights would collapse
        // the token embed unchanged through the whole stack — a degenerate
        // oracle).
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        let final_norm_gamma = vec![1.0f32; d];

        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
            // `w_t` is [in, out] row-major; rows = in, cols = out.
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }

        let blocks = (0..cfg.text.n_layer)
            .map(|_| DecoderBlock {
                attn_norm_gamma: vec![1.0f32; d],
                attn: GqaAttention {
                    q: linear(d, d, 0.10),
                    k: linear(d, kv_hidden, -0.07),
                    v: linear(d, kv_hidden, 0.05),
                    o: linear(d, d, -0.04),
                },
                ffn_norm_gamma: vec![1.0f32; d],
                ffn: SwiGluFfn {
                    gate: linear(d, ffn, 0.06),
                    up: linear(d, ffn, -0.02),
                    down: linear(ffn, d, 0.03),
                },
            })
            .collect();

        TextDecoder {
            token_emb,
            lm_head: None,
            blocks,
            final_norm_gamma,
            prefix: "",
            mapped: None,
            mapped_heads: None,
        }
    }

    #[test]
    fn new_rejects_zero_sentinel_config() {
        let mut cfg = tiny_cfg();
        cfg.text.n_layer = 0;
        let td = TextDecoder {
            token_emb: Vec::new(),
            lm_head: None,
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
            mapped: None,
            mapped_heads: None,
        };
        let err = match TextDecoderSession::new(&cfg, &td, BackendKind::Cpu) {
            Ok(_) => panic!("must fail — TextDecoderSession is not Debug"),
            Err(e) => e,
        };
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn new_rejects_gqa_head_split_mismatch() {
        let mut cfg = tiny_cfg();
        cfg.text.n_head_q = 3;
        cfg.text.n_head_kv = 2;
        // Since we construct decoder manually, blocks.len() also matters.
        let td = tiny_decoder(&tiny_cfg()); // note: uses original 2/1 split
        // With n_head_q=3 not divisible by n_head_kv=2, new must reject.
        let err = match TextDecoderSession::new(&cfg, &td, BackendKind::Cpu) {
            Ok(_) => panic!("must fail — TextDecoderSession is not Debug"),
            Err(e) => e,
        };
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }

    #[test]
    fn step_into_advances_position_and_produces_logits() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        assert_eq!(sess.position(), 0);
        sess.step_into(&[1u32, 0]).unwrap();
        assert_eq!(sess.position(), 2);
        let logits = sess.last_logits_row();
        assert_eq!(logits.len(), cfg.text.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn empty_step_is_noop() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        sess.step_into(&[]).unwrap();
        assert_eq!(sess.position(), 0);
    }

    #[test]
    fn out_of_range_token_is_error_not_panic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // vocab_size = 4, so id 99 is out of range.
        let err = sess.step_into(&[99u32]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn exceeding_n_ctx_is_error_not_panic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // n_ctx = 16; pushing 17 tokens overflows.
        let toks: Vec<u32> = (0..17).map(|i| i % cfg.text.vocab_size as u32).collect();
        let err = sess.step_into(&toks).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn full_prefix_matches_token_by_token() {
        // Full-sequence step must produce the same last-row logits as a
        // token-by-token decode after reset. Bit-for-bit is not guaranteed
        // because attention accumulator order changes; we assert atol=1e-4.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        let prefix = [1u32, 2, 0];
        sess.step_into(&prefix).unwrap();
        let full_last: Vec<f32> = sess.last_logits_row().to_vec();

        sess.reset();
        assert_eq!(sess.position(), 0);
        sess.step_into(&[prefix[0]]).unwrap();
        sess.step_into(&[prefix[1]]).unwrap();
        sess.step_into(&[prefix[2]]).unwrap();
        let cached_last = sess.last_logits_row();
        assert_eq!(cached_last.len(), full_last.len());
        for (i, (&f, &c)) in full_last.iter().zip(cached_last).enumerate() {
            assert!(
                (f - c).abs() < 5e-4,
                "idx {i}: full {f} vs cached {c} diff {}",
                (f - c).abs()
            );
        }
    }

    #[test]
    fn reset_and_replay_is_bit_identical() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let prefix = [1u32, 2];
        sess.step_into(&prefix).unwrap();
        let logits_1: Vec<f32> = sess.last_logits_row().to_vec();
        sess.reset();
        sess.step_into(&prefix).unwrap();
        let logits_2 = sess.last_logits_row();
        assert_eq!(logits_1.as_slice(), logits_2);
    }

    #[test]
    fn argmax_picks_first_max_on_ties() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn greedy_decode_respects_max_new() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // eos outside vocab => never terminates on EOS => cap at max_new
        // (bounded further by n_ctx - start_ids.len()).
        let eos = cfg.text.vocab_size as u32 + 100;
        let ids = greedy_decode(&mut sess, &[1u32], eos, 3).unwrap();
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|&t| (t as usize) < cfg.text.vocab_size));
    }

    #[test]
    fn greedy_decode_empty_start_is_rejected() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        assert!(matches!(
            greedy_decode(&mut sess, &[], 999, 3),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn greedy_decode_is_deterministic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let a = greedy_decode(&mut sess, &[1u32], 9999, 3).unwrap();
        let b = greedy_decode(&mut sess, &[1u32], 9999, 3).unwrap();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // KV snapshot / restore (beam search)
    // -----------------------------------------------------------------------

    /// Round-trip identity — the state after snapshot → step → restore →
    /// step must be identical to a plain step (from the snapshot point).
    /// This proves `restore` truly rewinds to the original state.
    #[test]
    fn kv_snapshot_round_trip_restores_original_state() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        // Take a snapshot at position 2.
        sess.step_into(&[1u32, 0]).unwrap();
        let snap = sess.kv_snapshot();
        assert_eq!(snap.position(), 2);

        // Reference: from the snapshot point, take one more step and read the
        // logits.
        sess.step_into(&[2u32]).unwrap();
        let reference: Vec<f32> = sess.last_logits_row().to_vec();

        // Divergence: from the snapshot point, evolve the session with a
        // different token sequence to prove that the snapshot really does
        // capture the entire persistent state (position + KV).
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        sess.step_into(&[1u32, 0]).unwrap();
        let snap = sess.kv_snapshot();
        sess.step_into(&[3u32]).unwrap();
        sess.step_into(&[0u32]).unwrap();

        // Restore + step: must match the reference bit-for-bit.
        sess.kv_restore(snap);
        assert_eq!(sess.position(), 2);
        sess.step_into(&[2u32]).unwrap();
        assert_eq!(sess.last_logits_row(), reference.as_slice());
    }

    /// Two clones of a single snapshot can evolve independently — extending
    /// one branch must not perturb the other.
    #[test]
    fn kv_snapshot_branches_do_not_interfere() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        // Common prefix.
        sess.step_into(&[1u32, 0]).unwrap();
        let base_snap = sess.kv_snapshot();

        // Branch A: continue with token 2.
        sess.kv_restore(base_snap.clone());
        sess.step_into(&[2u32]).unwrap();
        let branch_a: Vec<f32> = sess.last_logits_row().to_vec();
        let branch_a_pos = sess.position();

        // Branch B: continue with token 3 — must land on a different logits
        // row because the KV cache append content differs.
        sess.kv_restore(base_snap.clone());
        sess.step_into(&[3u32]).unwrap();
        let branch_b: Vec<f32> = sess.last_logits_row().to_vec();
        let branch_b_pos = sess.position();

        assert_eq!(branch_a_pos, branch_b_pos, "both branches step by 1");
        // The two logits vectors must not be exactly equal — a bit-identical
        // match would mean the KV cache append had no effect, i.e. the
        // snapshot/restore did not deep-copy. Use total-abs-difference for a
        // finite-precision check.
        let diff: f32 = branch_a
            .iter()
            .zip(&branch_b)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "branches must diverge — got zero total diff (branches likely aliased)"
        );

        // Re-branching to A from the same base must reproduce branch A
        // bit-for-bit (the snapshot is a true value snapshot, not a
        // one-shot).
        sess.kv_restore(base_snap.clone());
        sess.step_into(&[2u32]).unwrap();
        assert_eq!(
            sess.last_logits_row(),
            branch_a.as_slice(),
            "re-branch to A must reproduce A bit-for-bit"
        );
    }

    /// A snapshot taken at position 0 must be equivalent to a `reset()` —
    /// the restore path must handle the empty cache correctly.
    #[test]
    fn kv_snapshot_at_position_zero_is_a_reset() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let snap = sess.kv_snapshot();
        assert_eq!(snap.position(), 0);

        sess.step_into(&[1u32, 2, 0]).unwrap();
        sess.kv_restore(snap);
        assert_eq!(sess.position(), 0);

        // Fresh reference decode from scratch must match a decode after
        // restore.
        let mut ref_sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        ref_sess.step_into(&[1u32]).unwrap();
        sess.step_into(&[1u32]).unwrap();
        assert_eq!(sess.last_logits_row(), ref_sess.last_logits_row());
    }

    // -----------------------------------------------------------------
    // greedy_decode_with_segments (trained transcription-prompt layout)
    // -----------------------------------------------------------------

    #[test]
    fn segments_decode_matches_hand_driven_replay_bit_identically() {
        // The driver must be exactly reset → step(pre) → embed → step(post)
        // → argmax loop. Replay the same steps by hand and compare tokens.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let pre = [1u32, 3, 2];
        let post = [0u32, 2];
        let t_prefix = 2usize;
        let embed: Vec<f32> = (0..t_prefix * d).map(|i| 0.05 * (i as f32 - 3.0)).collect();
        let eos = 999u32; // unreachable
        let max_new = 4usize;

        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let got =
            greedy_decode_with_segments(&mut sess, &pre, &embed, t_prefix, &post, eos, max_new)
                .unwrap();

        let mut manual = TextDecoderSession::cpu(&cfg, &td).unwrap();
        manual.step_into(&pre).unwrap();
        manual
            .step_into_with_embed_prefix(&embed, t_prefix)
            .unwrap();
        manual.step_into(&post).unwrap();
        let mut want = Vec::new();
        for _ in 0..max_new {
            let next = argmax(manual.last_logits_row());
            want.push(next);
            manual.step_into(&[next]).unwrap();
        }
        assert_eq!(got, want, "driver must replay the manual step sequence");
        assert_eq!(got.len(), max_new);

        // Deterministic across calls (session reset at the top).
        let again =
            greedy_decode_with_segments(&mut sess, &pre, &embed, t_prefix, &post, eos, max_new)
                .unwrap();
        assert_eq!(got, again);
    }

    #[test]
    fn segments_decode_differs_from_bare_prefix_layout_inputs() {
        // Not an oracle on the values — a structural check that the post
        // segment participates: decoding with post = [0] vs post = [3]
        // starts the greedy loop from different logits, so the drivers see
        // different KV content. Both must complete and respect max_new.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let t_prefix = 2usize;
        let embed: Vec<f32> = (0..t_prefix * d).map(|i| 0.1 * (i as f32)).collect();
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let a =
            greedy_decode_with_segments(&mut sess, &[1], &embed, t_prefix, &[0], 999, 3).unwrap();
        let b =
            greedy_decode_with_segments(&mut sess, &[1], &embed, t_prefix, &[3], 999, 3).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn segments_decode_stops_on_eos_and_includes_it() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let embed: Vec<f32> = vec![0.05; d];
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // Find what the first greedy token would be, then use it as EOS —
        // the decode must stop immediately with exactly [eos].
        let probe = greedy_decode_with_segments(&mut sess, &[1], &embed, 1, &[0], 9999, 1).unwrap();
        let eos = probe[0];
        let got = greedy_decode_with_segments(&mut sess, &[1], &embed, 1, &[0], eos, 8).unwrap();
        assert_eq!(got, vec![eos]);
    }

    #[test]
    fn segments_decode_empty_pre_is_allowed_empty_post_is_not() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let embed: Vec<f32> = vec![0.05; d];
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        // Empty pre: documented no-op segment — decode proceeds.
        assert!(greedy_decode_with_segments(&mut sess, &[], &embed, 1, &[0], 999, 2).is_ok());
        // Empty post: explicit error (no defined sampling point).
        assert!(matches!(
            greedy_decode_with_segments(&mut sess, &[1], &embed, 1, &[], 999, 2),
            Err(VokraError::InvalidArgument(_))
        ));
        // t_prefix == 0: explicit error (layout needs an audio run).
        assert!(matches!(
            greedy_decode_with_segments(&mut sess, &[1], &[], 0, &[0], 999, 2),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn segments_decode_respects_n_ctx_budget() {
        // tiny_cfg has n_ctx = 16. pre(2) + prefix(4) + post(2) = 8 consumed
        // → at most 8 generated even when max_new asks for more.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let embed: Vec<f32> = vec![0.02; 4 * d];
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let got =
            greedy_decode_with_segments(&mut sess, &[1, 2], &embed, 4, &[0, 3], 999, 100).unwrap();
        assert!(got.len() <= 8, "n_ctx budget violated: {}", got.len());
    }
}
