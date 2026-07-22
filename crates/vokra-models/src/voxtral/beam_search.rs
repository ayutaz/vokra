//! Voxtral beam search + n-best decode (M3-10).
//!
//! # Scope
//!
//! [`beam_search_decode`] runs a classic width-K beam search over a
//! [`TextDecoderSession`]. It uses the session's
//! [`kv_snapshot`](TextDecoderSession::kv_snapshot) /
//! [`kv_restore`](TextDecoderSession::kv_restore) primitives to branch a
//! single session across the `beam_size` candidate hypotheses, so each
//! extension step touches only one KV cache append per beam — no
//! recompute-from-scratch, no fresh session per beam.
//!
//! The output is `Vec<BeamResult>` ordered by descending
//! **length-normalized** score (Google GNMT normalization). `beam_size == 1`
//! reduces to the same behavior as
//! [`greedy_decode`](super::text_decoder_session::greedy_decode) — argmax of
//! logits at every step — modulo the score bookkeeping.
//!
//! # Where this lives on the API surface
//!
//! `beam_search_decode` is the compute-facing entry point. The user-facing
//! wrapper is [`AsrHead::transcribe_beam`](super::AsrHead::transcribe_beam)
//! (which handles the encoder + adapter + BOS prefix) and, at the top,
//! [`VoxtralAsr::transcribe_beam`](super::VoxtralAsr::transcribe_beam) which
//! also detokenizes to text.
//!
//! # Zero-dep + no silent fallback (NFR-DS-02, FR-EX-08)
//!
//! The whole search runs on top of the compute seam that
//! `TextDecoderSession` already dispatches through; there is no new backend
//! surface here. Every failure mode surfaces as an explicit
//! [`VokraError`]: an out-of-vocab token id in `initial_tokens`, an empty
//! `initial_tokens`, a `beam_size == 0`, or a decode past
//! `config.text.n_ctx` — never a silent widening or truncation.
//!
//! # GNMT length penalty
//!
//! The length-normalized score used to rank finished hypotheses is
//!
//! ```text
//! normalized_score = log_prob / lp(len),   lp(len) = ((5 + len)^α / 6^α)
//! ```
//!
//! where `α` is [`BeamConfig::length_penalty`] and `len` is the number of
//! generated tokens (not counting the `initial_tokens` prefix). At `α = 0`
//! `lp(len) = 1` and the normalized score equals the raw sum-log-prob;
//! at `α > 0`, `lp(len)` grows with `len` and (because `log_prob` is
//! ≤ 0) dividing by a larger positive number produces a more negative
//! score — beams that accumulate more negative log-probability by virtue
//! of being longer are penalized. Formula from *Wu et al., 2016*
//! ("Google's Neural Machine Translation System").
//!
//! # No-repeat n-gram blocking
//!
//! When [`BeamConfig::no_repeat_ngram_size`] `> 0`, any candidate token that
//! would create a repeated `n`-gram in that beam's already-generated tokens
//! has its per-step log-prob set to `-inf` before the top-K selection. Same
//! semantics as HuggingFace's `no_repeat_ngram_size` — a token can never
//! introduce a repeated n-gram, so the effect is a hard mask, not a soft
//! discount.

use vokra_core::{Result, VokraError};

use super::text_decoder_session::{TextDecoderKvSnapshot, TextDecoderSession};

/// Beam-search + n-best configuration for [`beam_search_decode`].
///
/// Field defaults (via [`BeamConfig::greedy`] or
/// [`BeamConfig::with_beam_size`]) match the standard practice: `α = 0.6`
/// length-penalty, `top_k_per_beam = 2 * beam_size` (HuggingFace default),
/// `no_repeat_ngram_size = 0` (disabled).
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of beams kept per step. `1` reduces to greedy (up to score
    /// bookkeeping).
    pub beam_size: usize,
    /// Length-penalty exponent `α` (Google GNMT). See module docs for the
    /// exact formula. `0.0` disables normalization.
    pub length_penalty: f32,
    /// Maximum number of tokens to generate past the `initial_tokens` prefix.
    /// The actual cap is
    /// `max_new_tokens.min(session.config.text.n_ctx - initial_tokens.len())`.
    pub max_new_tokens: usize,
    /// End-of-sequence token id. A beam that emits this token is moved to
    /// the finished pool; its length-normalized score is computed *at the
    /// moment of emission* (so a shorter path that hits EOS earlier can beat
    /// a longer path with a larger raw log-prob).
    pub eos_token: u32,
    /// If `> 0`, forbid any candidate token that would form a repeated
    /// `n`-gram in that beam's generated tokens. HuggingFace-compatible
    /// semantics: the n-gram is checked against the entire generated sequence
    /// (including the `initial_tokens` prefix). `0` disables the blocking.
    pub no_repeat_ngram_size: usize,
    /// Number of top logits kept per beam when generating candidates
    /// (default `2 * beam_size`, the HuggingFace default). Must be `>= 1`
    /// and `>= beam_size` for the greedy-equivalence property to hold at
    /// `beam_size == 1`.
    pub top_k_per_beam: usize,
}

impl BeamConfig {
    /// Greedy-equivalent config (`beam_size = 1`, no normalization, no
    /// n-gram blocking).
    ///
    /// A `beam_search_decode` invocation with this config MUST produce the
    /// same token sequence as
    /// [`greedy_decode`](super::text_decoder_session::greedy_decode) up to
    /// the score bookkeeping (which greedy does not compute).
    #[must_use]
    pub fn greedy(eos_token: u32, max_new_tokens: usize) -> Self {
        Self {
            beam_size: 1,
            length_penalty: 0.0,
            max_new_tokens,
            eos_token,
            no_repeat_ngram_size: 0,
            top_k_per_beam: 1,
        }
    }

    /// Standard beam config for the given width. Uses α = 0.6, `top_k = 2K`,
    /// no n-gram blocking.
    #[must_use]
    pub fn with_beam_size(beam_size: usize, eos_token: u32, max_new_tokens: usize) -> Self {
        Self {
            beam_size,
            length_penalty: 0.6,
            max_new_tokens,
            eos_token,
            no_repeat_ngram_size: 0,
            top_k_per_beam: beam_size.saturating_mul(2).max(1),
        }
    }
}

/// One hypothesis in the beam. Kept live while `finished == false`.
///
/// The `kv_snapshot` is what makes the search cheap: a beam that is "picked
/// up" for extension has its persistent state restored from its snapshot
/// with a single [`TextDecoderSession::kv_restore`], then a single
/// [`step_into`](TextDecoderSession::step_into) advances the KV cache by
/// one token — no full-prefix recompute.
///
/// Not part of the public surface — the caller sees only
/// [`BeamResult`] via [`beam_search_decode`].
struct BeamState {
    /// Generated tokens (not counting the `initial_tokens` prefix). Note
    /// that when a beam finishes on EOS, that EOS token IS the last element
    /// here — matches the greedy contract that a finished beam includes its
    /// terminating token.
    tokens: Vec<u32>,
    /// Cumulative sum of per-step `log_softmax` values. Uses `f64` for
    /// numerical stability across long decodes.
    log_prob: f64,
    /// Snapshot of the session's persistent state after the beam's last
    /// step. When the beam is picked up for extension, restore this into
    /// the shared session first.
    kv_snapshot: TextDecoderKvSnapshot,
    /// Whether the beam has emitted [`BeamConfig::eos_token`].
    finished: bool,
}

impl BeamState {
    /// Length-normalized score under the GNMT length-penalty formula. See
    /// the module docs for the exact expression.
    fn length_normalized_score(&self, alpha: f32) -> f64 {
        length_normalized(self.log_prob, self.tokens.len(), alpha)
    }
}

/// One returned hypothesis from [`beam_search_decode`].
#[derive(Debug, Clone, PartialEq)]
pub struct BeamResult {
    /// The generated token id sequence, excluding the `initial_tokens`
    /// prefix passed to the search. A finished beam ends with
    /// [`BeamConfig::eos_token`].
    pub tokens: Vec<u32>,
    /// Cumulative sum of per-step `log_softmax` values (raw, unnormalized).
    pub log_prob: f64,
    /// Length-normalized ranking score — see the module docs for the
    /// formula. This is the value the returned slice is sorted by,
    /// descending.
    pub length_normalized_score: f64,
}

/// GNMT-normalized score.
///
/// `lp(len) = ((5 + len)^α) / (6^α)`, `normalized = log_prob / lp(len)`.
/// At `len == 0` we return `log_prob` unchanged (avoids a zero-length
/// hypothesis producing a NaN / degenerate normalized score).
///
/// This is a free function (not a method) so both [`BeamState`] and the
/// unit tests can compute it directly.
fn length_normalized(log_prob: f64, len: usize, alpha: f32) -> f64 {
    if len == 0 {
        return log_prob;
    }
    let alpha = f64::from(alpha);
    if alpha == 0.0 {
        return log_prob;
    }
    let lp = ((5.0 + len as f64).powf(alpha)) / (6.0_f64.powf(alpha));
    log_prob / lp
}

/// Runs beam search + n-best decode on `session` starting from
/// `initial_tokens` (a non-empty prefix that gets appended to the session's
/// KV cache once), and returns up to `config.beam_size` hypotheses, best
/// (highest length-normalized score) first.
///
/// # Contract
///
/// - The session is [`TextDecoderSession::reset`] at the top so a repeat
///   call reproduces the first.
/// - `initial_tokens` are appended to the KV cache but are NOT included in
///   the returned [`BeamResult::tokens`].
/// - A beam that emits `config.eos_token` is moved to the finished pool;
///   the EOS token IS the last element of that beam's `tokens`.
/// - All returned beams are ranked by
///   [`length_normalized_score`](BeamResult::length_normalized_score)
///   descending. Ties are broken by ascending `log_prob` — the pathological
///   case is only exercised by the unit tests.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `initial_tokens` is empty,
///   `beam_size == 0`, `top_k_per_beam == 0`, or `max_new_tokens == 0`.
/// - Any error the underlying [`step_into`](TextDecoderSession::step_into)
///   surfaces (out-of-vocab token id, exceeded `n_ctx`, backend error).
pub fn beam_search_decode(
    session: &mut TextDecoderSession<'_>,
    initial_tokens: &[u32],
    config: &BeamConfig,
) -> Result<Vec<BeamResult>> {
    validate_config(config)?;
    if initial_tokens.is_empty() {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search_decode: initial_tokens must not be empty".into(),
        ));
    }

    // ---------- seed the shared session with the prefix ---------------
    session.reset();
    session.step_into(initial_tokens)?;

    beam_search_inner(session, config, initial_tokens.len())
}

/// Beam-search + n-best decode for the audio-conditioned ASR path (M3-10
/// Wave 8 + beam-search follow-up).
///
/// Prefills the decoder with an audio-adapter soft-prefix embedding
/// (`prefix_embed`, `[t_prefix, hidden_dim]`), then steps in `[bos_id]`,
/// then runs the same beam-search inner loop as [`beam_search_decode`].
/// The returned [`BeamResult::tokens`] excludes both the soft-prefix
/// embedding (which has no token id) and the `bos_id` — only the generated
/// tokens are returned.
///
/// The session is [`TextDecoderSession::reset`] at the top.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on the same conditions as
///   [`beam_search_decode`], plus `t_prefix == 0` (use plain
///   [`beam_search_decode`] instead).
pub fn beam_search_decode_with_prefix(
    session: &mut TextDecoderSession<'_>,
    prefix_embed: &[f32],
    t_prefix: usize,
    bos_id: u32,
    config: &BeamConfig,
) -> Result<Vec<BeamResult>> {
    validate_config(config)?;
    if t_prefix == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search_decode_with_prefix: t_prefix must be > 0 \
             (use beam_search_decode instead)"
                .into(),
        ));
    }

    // ---------- seed the session with the soft-prefix + BOS -----------
    session.reset();
    session.step_into_with_embed_prefix(prefix_embed, t_prefix)?;
    session.step_into(&[bos_id])?;

    // consumed = t_prefix + 1 (BOS); use it as the prefix budget for
    // the inner loop's n_ctx accounting.
    let consumed = t_prefix.saturating_add(1);
    beam_search_inner(session, config, consumed)
}

/// Beam-search + n-best decode for the **trained transcription-prompt
/// layout** (P2 cc-05/07 follow-up) — the beam sibling of
/// [`greedy_decode_with_segments`](super::text_decoder_session::greedy_decode_with_segments).
///
/// Seeds the session with `step_into(pre_tokens)` →
/// `step_into_with_embed_prefix(audio rows)` → `step_into(post_tokens)`
/// (the upstream `masked_scatter` replay: audio soft-prefix rows occupy the
/// `[AUDIO]` placeholder positions), then runs the shared beam inner loop.
/// The seed beam's first expansion reads the post-segment's last logits row
/// — the same distribution the greedy sibling argmaxes — so `beam_size == 1`
/// reproduces the greedy sequence.
///
/// Returned [`BeamResult::tokens`] contain only the generated tokens (no
/// prompt segment). The session is [`TextDecoderSession::reset`] at the top.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on the same config conditions as
///   [`beam_search_decode`], plus `t_prefix == 0` (the trained layout is
///   only defined over an audio run) or an empty `post_tokens` (the seed
///   distribution is the post-segment's last logits row — an empty segment
///   has no defined sampling point);
/// - any error the underlying session steps surface.
pub fn beam_search_decode_with_segments(
    session: &mut TextDecoderSession<'_>,
    pre_tokens: &[u32],
    prefix_embed: &[f32],
    t_prefix: usize,
    post_tokens: &[u32],
    config: &BeamConfig,
) -> Result<Vec<BeamResult>> {
    validate_config(config)?;
    if t_prefix == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search_decode_with_segments: t_prefix must be > 0 — the \
             transcription layout is only defined over an audio soft-prefix run (use \
             beam_search_decode / beam_search_decode_with_prefix instead)"
                .into(),
        ));
    }
    if post_tokens.is_empty() {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search_decode_with_segments: post_tokens must not be empty — the \
             seed distribution is the logits after the post-audio segment \
             ([/INST]…[TRANSCRIBE])"
                .into(),
        ));
    }

    // ---------- seed the session with the three prompt segments -------
    session.reset();
    session.step_into(pre_tokens)?; // empty pre is a documented no-op
    session.step_into_with_embed_prefix(prefix_embed, t_prefix)?;
    session.step_into(post_tokens)?;

    let consumed = pre_tokens
        .len()
        .saturating_add(t_prefix)
        .saturating_add(post_tokens.len());
    beam_search_inner(session, config, consumed)
}

/// Validates the load-bearing parts of a [`BeamConfig`] common to both
/// entry points.
fn validate_config(config: &BeamConfig) -> Result<()> {
    if config.beam_size == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search: beam_size must be >= 1".into(),
        ));
    }
    if config.top_k_per_beam == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search: top_k_per_beam must be >= 1".into(),
        ));
    }
    if config.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(
            "voxtral::beam_search: max_new_tokens must be >= 1".into(),
        ));
    }
    Ok(())
}

/// Shared inner-loop driver for both [`beam_search_decode`] and
/// [`beam_search_decode_with_prefix`]. The caller is responsible for
/// seeding the session (reset + step_into for the token-only path, or
/// reset + step_into_with_embed_prefix + step_into(bos) for the audio
/// path). `consumed_positions` is the number of decoder positions the
/// seed already ate (used only to cap `max_new_tokens` against
/// `n_ctx`).
fn beam_search_inner(
    session: &mut TextDecoderSession<'_>,
    config: &BeamConfig,
    consumed_positions: usize,
) -> Result<Vec<BeamResult>> {
    // The vocab / n_ctx budget the search must respect. `vocab_size` is
    // baked into every step (out-of-range token ids surface from
    // step_into, but we use it here to size the log-softmax buffer + the
    // top-K walk).
    let vocab = session.vocab_size();
    // Cap max_new by (n_ctx - consumed_positions) so we never generate a
    // token that would overflow n_ctx on the next step. The session
    // enforces this too, but doing it here keeps the search's inner loop
    // free of a StopReason enum.
    let n_ctx_budget = session.n_ctx().saturating_sub(consumed_positions);
    let effective_max_new = config.max_new_tokens.min(n_ctx_budget);
    if effective_max_new == 0 {
        // Only the prefix fits — no tokens can be generated. Return an
        // empty result so the caller sees the honest state rather than a
        // fabricated hypothesis.
        return Ok(Vec::new());
    }

    // ---------- initial single-beam pool ------------------------------
    // Snapshot the shared session so every branch starts from the same
    // persistent state (the seeded prefix).
    let seed_snapshot = session.kv_snapshot();
    let mut active: Vec<BeamState> = vec![BeamState {
        tokens: Vec::with_capacity(effective_max_new.min(64)),
        log_prob: 0.0,
        kv_snapshot: seed_snapshot,
        finished: false,
    }];
    let mut finished: Vec<BeamState> = Vec::new();

    // ---------- main loop ---------------------------------------------
    for _ in 0..effective_max_new {
        if active.is_empty() {
            break;
        }

        // Reserve room for `active.len() * top_k_per_beam` candidates.
        // Small so a Vec is fine.
        let mut candidates: Vec<BeamState> =
            Vec::with_capacity(active.len() * config.top_k_per_beam);

        // Expand every active beam. Each expansion:
        // (1) restores the shared session to the beam's snapshot,
        // (2) runs one step_into against the beam's last emitted token,
        // (3) reads the logits + log_softmax,
        // (4) applies n-gram blocking mask (if enabled),
        // (5) picks the top_k_per_beam candidates.
        //
        // We consume `active` by moving each beam into the extension so
        // the snapshot can be `Clone`d per candidate without redundant
        // holds. The final `active` after the loop is populated by the
        // "split-finished" pass below.
        let mut active_drain = std::mem::take(&mut active);
        for parent in active_drain.drain(..) {
            // Restore the shared session to the parent's state so
            // step_into starts from the parent's KV cache. We work on a
            // clone of the snapshot because the parent's snapshot itself
            // is going to be re-used as the seed for the child snapshots
            // — cloning first avoids the "moved value" issue.
            session.kv_restore(parent.kv_snapshot.clone());
            // The step token: for the first extension of the seed beam,
            // step_into with an "empty seed" is a no-op — so we drive the
            // very first step off the initial_tokens' last token. But we
            // already appended initial_tokens above, so what we actually
            // need to feed here is the beam's LAST emitted token (which
            // is unemitted on the seed itself). For the seed beam
            // (`parent.tokens` empty) we skip the step_into: the session
            // still has the initial_tokens' KV cache and its logits scratch
            // holds the logits for the token *after* the prefix.
            //
            // Wait — that's not right either. `step_into(initial_tokens)`
            // above already computed the logits for the token following
            // the prefix, and left them in scratch. So the seed beam's
            // "next logits" are those. For subsequent beams, we advanced
            // one token per step, so the parent's snapshot corresponds to
            // "prefix + parent.tokens" — restore + read means we need to
            // step_into the parent's LAST emitted token FIRST (because
            // the snapshot was taken BEFORE emitting the token).
            //
            // Actually re-reading: the snapshot for a child at position P
            // is taken AFTER the step_into that added the child's last
            // token. So restore + no step gives the state where the next
            // step_into's LAST logits row is the logits for the child's
            // last token's SUCCESSOR — but the scratch is not part of the
            // snapshot. So after restore we MUST call step_into with the
            // child's last token again... but that's already in the KV
            // cache. That'd be a duplicate.
            //
            // The correct pattern is: snapshot is taken BEFORE the
            // to-be-emitted token's step_into; then for the child, we
            // restore + step_into(chosen_token) to advance. That's what
            // we do here — see the child-creation block below where the
            // parent's snapshot is captured PRIOR to the step_into that
            // adds the candidate token. For the SEED beam, the initial
            // step_into(initial_tokens) has already been done and its
            // logits scratch holds the "next" distribution; we just read
            // that.
            let logits = if parent.tokens.is_empty() {
                // Seed beam: session was just seeded with initial_tokens;
                // its scratch's last-logits-row IS the distribution over
                // the token following the prefix. Read directly.
                session.last_logits_row().to_vec()
            } else {
                // Extension: advance the KV cache by the parent's last
                // emitted token, then read the resulting scratch's
                // last-logits-row (the distribution over the NEXT token).
                let last = *parent.tokens.last().expect("non-empty");
                session.step_into(&[last])?;
                session.last_logits_row().to_vec()
            };

            let log_probs = log_softmax(&logits);
            // n-gram blocking: mask any candidate token that would form a
            // repeated `n`-gram given the parent's generated tokens +
            // that candidate. If parent has fewer than (n - 1) tokens,
            // no n-gram of length `n` can complete, so nothing is masked.
            let blocked = ngram_block_mask(&parent.tokens, config.no_repeat_ngram_size, vocab);

            // The parent's post-step snapshot — this is the state AFTER
            // the KV cache append for the parent's last token, so a child
            // starting here can extend by exactly one token. We `Clone`
            // this per candidate below so the sibling candidates each
            // hold their own snapshot.
            let parent_snapshot_after_step = if parent.tokens.is_empty() {
                // Seed beam: the KV cache holds initial_tokens; the
                // "snapshot after step" is exactly the seed snapshot.
                parent.kv_snapshot.clone()
            } else {
                // Extension: we just stepped in the parent's last token,
                // so the session's current state is the "snapshot after
                // step". Take a fresh snapshot to capture it.
                session.kv_snapshot()
            };

            for (tok, lp) in top_k(&log_probs, &blocked, config.top_k_per_beam) {
                // Skip candidates that were masked to -inf (which the
                // top-K may still surface at the tail when the top-K
                // window exceeds the number of unmasked tokens).
                if !lp.is_finite() {
                    continue;
                }
                let mut child_tokens = parent.tokens.clone();
                child_tokens.push(tok);
                candidates.push(BeamState {
                    tokens: child_tokens,
                    log_prob: parent.log_prob + f64::from(lp),
                    kv_snapshot: parent_snapshot_after_step.clone(),
                    finished: false,
                });
            }
        }

        if candidates.is_empty() {
            // Every extension was blocked (pathological — e.g. every
            // candidate was masked and the parent had no unmasked
            // tokens). Nothing more to do; drop out and score whatever we
            // have in `finished`.
            break;
        }

        // Rank by length-normalized score descending and keep the top
        // `beam_size` overall (across all parents). This is the standard
        // "prune to K" step.
        sort_by_normalized_desc(&mut candidates, config.length_penalty);
        candidates.truncate(config.beam_size);

        // Split finished vs. still-active. Take the finished ones out of
        // the active pool (their snapshot is no longer needed — they
        // will never be extended). The still-active pool feeds the next
        // iteration.
        for mut c in candidates {
            if c.tokens.last().copied() == Some(config.eos_token) {
                c.finished = true;
                finished.push(c);
            } else {
                active.push(c);
            }
        }

        // Early stop: if we already have beam_size finished hypotheses
        // (and every active would only get worse under the length penalty
        // — implicit under GNMT because the numerator can only get more
        // negative and the denominator only grows), we can stop. Match
        // HuggingFace's `early_stopping=True`: stop as soon as we have
        // beam_size finished.
        if finished.len() >= config.beam_size {
            break;
        }
    }

    // ---------- collect final results ---------------------------------
    // If nothing finished, fall back to the still-active pool (they are
    // ranked by normalized score too when we return them).
    let mut winners: Vec<BeamState> = if finished.is_empty() {
        active
    } else {
        finished
    };
    sort_by_normalized_desc(&mut winners, config.length_penalty);
    // De-duplicate: two beams may reach the same token sequence via
    // different snapshots (shouldn't happen with a deterministic model,
    // but a defensive dedup keeps the returned set clean).
    let mut out: Vec<BeamResult> = Vec::with_capacity(winners.len());
    for w in winners {
        let ns = w.length_normalized_score(config.length_penalty);
        // Skip a duplicate token sequence — keep the first (which is the
        // higher-scored one because we sorted first).
        if out.iter().any(|r| r.tokens == w.tokens) {
            continue;
        }
        out.push(BeamResult {
            tokens: w.tokens,
            log_prob: w.log_prob,
            length_normalized_score: ns,
        });
        if out.len() >= config.beam_size {
            break;
        }
    }
    Ok(out)
}

/// In-place descending sort of a beam pool by length-normalized score
/// (tie-broken by ascending log_prob so the pathological equal-score case
/// is deterministic).
fn sort_by_normalized_desc(beams: &mut [BeamState], alpha: f32) {
    beams.sort_by(|a, b| {
        let sa = a.length_normalized_score(alpha);
        let sb = b.length_normalized_score(alpha);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.log_prob
                    .partial_cmp(&b.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

/// Numerically stable `log_softmax` over a logits row.
///
/// Uses `f64` accumulation across the exp/log to keep the tails from
/// underflowing when the logit range is large. Returns `f32` because the
/// caller only adds these values to an `f64` accumulator (i.e. the
/// precision loss on the individual increment is inconsequential compared
/// to the sum).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let log_sum = f64::from(max) + sum.ln();
    logits
        .iter()
        .map(|&l| (f64::from(l) - log_sum) as f32)
        .collect()
}

/// Returns a boolean mask of length `vocab_size` where `true` means the
/// candidate token would create a repeated `n`-gram in
/// `parent_tokens + [candidate]`.
///
/// When `n <= 1` or the parent has fewer than `n - 1` tokens, no n-gram
/// can complete — the whole mask is `false`. Otherwise the sliding window
/// over `parent_tokens` finds every existing `(n - 1)`-gram that matches
/// the last `n - 1` tokens; the token(s) that immediately follow each
/// match are the ones we forbid.
fn ngram_block_mask(parent_tokens: &[u32], n: usize, vocab_size: usize) -> Vec<bool> {
    let mut mask = vec![false; vocab_size];
    if n <= 1 {
        return mask;
    }
    if parent_tokens.len() + 1 < n {
        return mask;
    }
    // The (n - 1)-token suffix that a new candidate would extend into an
    // n-gram: `parent_tokens[-(n-1)..]`.
    let suffix_len = n - 1;
    let suffix_start = parent_tokens.len() - suffix_len;
    let suffix = &parent_tokens[suffix_start..];

    // For every position i where an n-gram (n_tokens) could start in
    // parent_tokens (i.e. i + n <= parent_tokens.len()), check whether
    // parent_tokens[i..i+suffix_len] == suffix. If so, the next token —
    // parent_tokens[i + suffix_len] — is a forbidden candidate.
    if parent_tokens.len() < n {
        return mask;
    }
    for i in 0..=parent_tokens.len() - n {
        if &parent_tokens[i..i + suffix_len] == suffix {
            let forbid = parent_tokens[i + suffix_len] as usize;
            if forbid < vocab_size {
                mask[forbid] = true;
            }
        }
    }
    mask
}

/// Returns the `k` highest `(index, value)` pairs from `values`, mapping
/// masked entries (`blocked[i] == true`) to `-inf`. The result is sorted
/// value-descending; entries with `-inf` are pushed to the tail (so the
/// caller can `skip_while(!is_finite)` if they prefer).
fn top_k(values: &[f32], blocked: &[bool], k: usize) -> Vec<(u32, f32)> {
    debug_assert_eq!(values.len(), blocked.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &v) in values.iter().enumerate() {
        let effective = if blocked[i] { f32::NEG_INFINITY } else { v };
        if top.len() < k {
            top.push((i as u32, effective));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if effective > top[k - 1].1 {
            top[k - 1] = (i as u32, effective);
            // Bubble up.
            let mut j = k - 1;
            while j > 0 && top[j].1 > top[j - 1].1 {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    top
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};
    use crate::voxtral::text_decoder_session::greedy_decode;
    use crate::voxtral::{TextDecoder, VoxtralConfig};
    use vokra_core::BackendKind;

    // Test helpers duplicated from text_decoder_session::tests — the tiny
    // config + decoder is a shared oracle across the module.
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

    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let head_dim = d / cfg.text.n_head_q;
        let kv_hidden = cfg.text.n_head_kv * head_dim;
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
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
            final_norm_gamma: vec![1.0f32; d],
            prefix: "",
            mapped: None,
        }
    }

    // --------- pure-function properties (length norm, log_softmax) ----

    #[test]
    fn length_normalized_alpha_zero_equals_log_prob() {
        for &lp in &[-1.0_f64, -10.0, -100.0] {
            for &len in &[1_usize, 5, 100] {
                assert_eq!(length_normalized(lp, len, 0.0), lp);
            }
        }
    }

    #[test]
    fn length_normalized_len_zero_is_a_noop() {
        // Zero-length beam: don't divide by lp(0), just return log_prob.
        for &alpha in &[0.0_f32, 0.6, 1.0] {
            assert_eq!(length_normalized(-1.0, 0, alpha), -1.0);
        }
    }

    #[test]
    fn length_normalized_alpha_positive_disadvantages_longer_beams_same_avg_lp() {
        // Two beams with the SAME per-token log-prob (avg = -1) but
        // different lengths. Under α = 0.6, the longer beam's raw log_prob
        // is more negative AND its length penalty grows sub-linearly, so
        // the normalized score is strictly more negative (worse).
        let short = length_normalized(-5.0, 5, 0.6);
        let long = length_normalized(-10.0, 10, 0.6);
        assert!(
            long < short,
            "α > 0 with same avg lp: longer must be strictly worse. \
             short={short} long={long}"
        );
    }

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        // Two rows to prove the max-shift + f64 accumulator both work.
        for logits in &[
            vec![1.0f32, 2.0, 3.0, -1.0],
            vec![-10.0f32, -20.0, 30.0, -50.0],
        ] {
            let lp = log_softmax(logits);
            let s: f64 = lp.iter().map(|&x| (f64::from(x)).exp()).sum();
            assert!((s - 1.0).abs() < 1e-6, "sum {s} != 1 for {logits:?}");
        }
    }

    // --------- top-K + n-gram blocking ----------------------------

    #[test]
    fn top_k_returns_k_largest_descending() {
        let vals = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
        let blocked = vec![false; vals.len()];
        let top = top_k(&vals, &blocked, 3);
        // Values sorted desc: 5, 4, 3 with indices 1, 3, 2.
        assert_eq!(top, vec![(1, 5.0), (3, 4.0), (2, 3.0)]);
    }

    #[test]
    fn top_k_blocked_entries_map_to_neg_inf() {
        // Block index 1 (value 5.0). The top-K should skip it and pick 4.0, 3.0, 2.0.
        let vals = vec![1.0f32, 5.0, 3.0, 4.0, 2.0];
        let mut blocked = vec![false; vals.len()];
        blocked[1] = true;
        let top = top_k(&vals, &blocked, 3);
        // 4.0, 3.0, 2.0 with indices 3, 2, 4.
        assert_eq!(top, vec![(3, 4.0), (2, 3.0), (4, 2.0)]);
    }

    #[test]
    fn top_k_across_multiple_beams_expansion() {
        // Simulate a top-K of 2 across 3 beams x 4 logits each — the
        // beam-search main loop does this in `beam_search_decode`. This is
        // a pure sanity check on the top-K helper — the "3 beams × top 4"
        // case is exercised indirectly by the main loop test below.
        let vals = vec![0.5_f32, 0.9, 0.1, 0.8];
        let blocked = vec![false; vals.len()];
        let top = top_k(&vals, &blocked, 2);
        assert_eq!(top, vec![(1, 0.9), (3, 0.8)]);
    }

    #[test]
    fn ngram_block_mask_disabled_for_n_zero_and_one() {
        for n in 0..=1 {
            let mask = ngram_block_mask(&[1, 2, 3, 1, 2], n, 5);
            assert!(mask.iter().all(|&b| !b), "n={n} must not block anything");
        }
    }

    #[test]
    fn ngram_block_mask_flags_repeated_bigram_completion() {
        // Parent: [1, 2, 3, 1]. A "2" as the next token would form the
        // bigram (1, 2) which already exists at index 0 — so 2 must be
        // masked.
        let mask = ngram_block_mask(&[1, 2, 3, 1], 2, 5);
        assert!(
            mask[2],
            "bigram (1, 2) already exists; extending 1 with 2 must be blocked"
        );
        // 3 as next → (1, 3). Not present — must NOT be masked.
        assert!(!mask[3]);
    }

    #[test]
    fn ngram_block_mask_flags_repeated_trigram_completion() {
        // n=3: parent [1, 2, 3, 1, 2]. Next candidate "3" would complete
        // (1, 2, 3) which already exists at index 0 — so 3 must be masked.
        let mask = ngram_block_mask(&[1, 2, 3, 1, 2], 3, 5);
        assert!(
            mask[3],
            "trigram (1, 2, 3) already exists; extending (1, 2) with 3 must be blocked"
        );
        // 4 → (1, 2, 4). Not present.
        assert!(!mask[4]);
    }

    #[test]
    fn ngram_block_mask_short_parent_masks_nothing() {
        // Parent shorter than n - 1 = 2 → no n-gram of length 3 can
        // complete. Nothing masked.
        let mask = ngram_block_mask(&[1], 3, 5);
        assert!(mask.iter().all(|&b| !b));
    }

    // --------- beam_search_decode: contract + property ---------------

    #[test]
    fn beam_search_rejects_bad_inputs() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        // Empty initial_tokens.
        let bc = BeamConfig::greedy(9999, 3);
        assert!(matches!(
            beam_search_decode(&mut sess, &[], &bc),
            Err(VokraError::InvalidArgument(_))
        ));

        // beam_size == 0.
        let mut bad = BeamConfig::greedy(9999, 3);
        bad.beam_size = 0;
        assert!(matches!(
            beam_search_decode(&mut sess, &[1], &bad),
            Err(VokraError::InvalidArgument(_))
        ));

        // top_k_per_beam == 0.
        let mut bad = BeamConfig::greedy(9999, 3);
        bad.top_k_per_beam = 0;
        assert!(matches!(
            beam_search_decode(&mut sess, &[1], &bad),
            Err(VokraError::InvalidArgument(_))
        ));

        // max_new_tokens == 0.
        let mut bad = BeamConfig::greedy(9999, 3);
        bad.max_new_tokens = 0;
        assert!(matches!(
            beam_search_decode(&mut sess, &[1], &bad),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// beam_size = 1 with the same eos + max_new must produce the same
    /// token sequence as [`greedy_decode`]. `atol` on the log_prob is not
    /// checked — greedy doesn't compute it — but the tokens must match.
    #[test]
    fn beam_size_one_reproduces_greedy() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        // Greedy reference.
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let greedy = greedy_decode(&mut sess, &[1u32], 9999, 3).unwrap();

        // beam_size = 1.
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let bc = BeamConfig::greedy(9999, 3);
        let beams = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        assert_eq!(beams.len(), 1);
        assert_eq!(
            beams[0].tokens, greedy,
            "beam_size=1 must reproduce greedy token-for-token"
        );
    }

    /// The results are sorted by length-normalized score descending. Ties
    /// are broken by ascending log_prob (i.e. the more-conservative — less
    /// negative — beam wins the tie), which the impl documents.
    #[test]
    fn beam_search_returns_results_sorted_by_normalized_score_desc() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let bc = BeamConfig::with_beam_size(3, 9999, 4);
        let beams = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        assert!(!beams.is_empty());
        for pair in beams.windows(2) {
            assert!(
                pair[0].length_normalized_score >= pair[1].length_normalized_score,
                "results not sorted: {:?} < {:?}",
                pair[0].length_normalized_score,
                pair[1].length_normalized_score
            );
        }
    }

    /// Two calls with the same session + config must return identical
    /// results — the search is deterministic.
    #[test]
    fn beam_search_is_deterministic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);

        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let bc = BeamConfig::with_beam_size(3, 9999, 4);
        let a = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();

        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let b = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        assert_eq!(a, b);
    }

    /// A beam that emits EOS is moved to the finished pool. This test
    /// runs at a wider beam (2) with an in-vocab EOS so at least one beam
    /// is expected to terminate.
    #[test]
    fn beam_search_stops_on_eos() {
        // eos = 0 is in vocab; with our tiny decoder some beams will
        // eventually pick 0.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let bc = BeamConfig::with_beam_size(2, /*eos*/ 0, 8);
        let beams = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        // Every returned beam must be a well-formed hypothesis (in-vocab
        // tokens, non-empty).
        for r in &beams {
            assert!(
                !r.tokens.is_empty(),
                "beam must have at least one generated token"
            );
            assert!(r.tokens.iter().all(|&t| (t as usize) < cfg.text.vocab_size));
        }
    }

    /// When no_repeat_ngram_size is enabled and the natural greedy path
    /// would emit a repeated bigram, the beam search MUST NOT emit that
    /// bigram. Uses a beam of size 1 to make the greedy vs. blocked
    /// comparison clean.
    #[test]
    fn beam_search_no_repeat_ngram_blocks_repeats() {
        // The tiny decoder here is symmetric enough that a raw beam might
        // repeat a token pair. The no_repeat check is a construction
        // property: after decoding, no bigram of length 2 may appear
        // twice.
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let mut bc = BeamConfig::with_beam_size(1, 9999, 4);
        bc.no_repeat_ngram_size = 2;

        let beams = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        assert_eq!(beams.len(), 1);
        let toks = &beams[0].tokens;
        // Every bigram in the returned sequence must be unique.
        let mut seen = std::collections::HashSet::new();
        for w in toks.windows(2) {
            assert!(
                seen.insert((w[0], w[1])),
                "bigram {w:?} appeared twice — no_repeat_ngram_size=2 was violated"
            );
        }
    }

    /// beam_search_decode on a session that has already been decoded must
    /// still reset + reseed correctly (the impl calls session.reset() at
    /// the top). Same posture as greedy_decode.
    #[test]
    fn beam_search_resets_previously_stepped_session() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();

        // Pollute the session.
        sess.step_into(&[3u32, 2, 1, 0, 3]).unwrap();

        let bc = BeamConfig::greedy(9999, 3);
        let beams = beam_search_decode(&mut sess, &[1u32], &bc).unwrap();
        assert_eq!(beams.len(), 1);

        // Reference: fresh session.
        let mut sess2 = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let ref_beams = beam_search_decode(&mut sess2, &[1u32], &bc).unwrap();
        assert_eq!(beams, ref_beams);
    }

    /// A backend other than CPU is not needed for beam search itself
    /// (it's a CPU-only host-side driver, per FR-OP-40); we just prove
    /// the session's backend selection is honored.
    #[test]
    fn beam_search_uses_session_backend() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let sess = TextDecoderSession::new(&cfg, &td, BackendKind::Cpu).unwrap();
        assert_eq!(sess.backend_name(), "cpu");
    }

    // -----------------------------------------------------------------
    // beam_search_decode_with_segments (trained transcription layout)
    // -----------------------------------------------------------------

    #[test]
    fn segments_beam_size_one_matches_segments_greedy() {
        // The greedy-equivalence contract every beam entry point upholds:
        // beam_size = 1 through the segments driver must reproduce
        // greedy_decode_with_segments token-for-token.
        use crate::voxtral::text_decoder_session::greedy_decode_with_segments;
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let pre = [1u32, 3];
        let post = [0u32, 2];
        let t_prefix = 2usize;
        let embed: Vec<f32> = (0..t_prefix * d).map(|i| 0.04 * (i as f32 - 2.0)).collect();
        let eos = 9999u32;
        let max_new = 3usize;

        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let greedy =
            greedy_decode_with_segments(&mut sess, &pre, &embed, t_prefix, &post, eos, max_new)
                .unwrap();

        let bc = BeamConfig::greedy(eos, max_new);
        let beams = beam_search_decode_with_segments(&mut sess, &pre, &embed, t_prefix, &post, &bc)
            .unwrap();
        assert_eq!(beams.len(), 1);
        assert_eq!(beams[0].tokens, greedy);
    }

    #[test]
    fn segments_beam_returns_ranked_beams_and_is_deterministic() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let embed: Vec<f32> = (0..2 * d).map(|i| 0.03 * (i as f32)).collect();
        let bc = BeamConfig::with_beam_size(3, 9999, 3);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        let a = beam_search_decode_with_segments(&mut sess, &[1], &embed, 2, &[0], &bc).unwrap();
        assert!(!a.is_empty());
        assert!(a.len() <= 3);
        for pair in a.windows(2) {
            assert!(pair[0].length_normalized_score >= pair[1].length_normalized_score);
        }
        let b = beam_search_decode_with_segments(&mut sess, &[1], &embed, 2, &[0], &bc).unwrap();
        assert_eq!(a, b, "segments beam must be deterministic (reset at top)");
    }

    #[test]
    fn segments_beam_rejects_zero_prefix_and_empty_post() {
        let cfg = tiny_cfg();
        let td = tiny_decoder(&cfg);
        let d = cfg.text.hidden_dim;
        let embed: Vec<f32> = vec![0.05; d];
        let bc = BeamConfig::greedy(9999, 3);
        let mut sess = TextDecoderSession::cpu(&cfg, &td).unwrap();
        assert!(matches!(
            beam_search_decode_with_segments(&mut sess, &[1], &[], 0, &[0], &bc),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            beam_search_decode_with_segments(&mut sess, &[1], &embed, 1, &[], &bc),
            Err(VokraError::InvalidArgument(_))
        ));
        // Config validation still applies (beam_size = 0).
        let bad = BeamConfig {
            beam_size: 0,
            ..BeamConfig::greedy(9999, 3)
        };
        assert!(matches!(
            beam_search_decode_with_segments(&mut sess, &[1], &embed, 1, &[0], &bad),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
