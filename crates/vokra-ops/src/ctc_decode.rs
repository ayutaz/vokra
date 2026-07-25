//! CTC decoding — greedy blank-fold + prefix beam search with n-gram LM
//! shallow fusion and hotword boost (SoTA plan Phase 2 ASR primitive;
//! FR-OP-41).
//!
//! # Runtime function — NOT a graph node (FR-EX-10 / FR-OP-40)
//!
//! [`ctc_decode_greedy`] and [`ctc_decode_beam`] are **host-side runtime
//! functions**, not `OpKind` variants. Encoding a CTC decoder as a graph op
//! would break execution-provider compatibility (the "contrib op"
//! anti-pattern, FR-OP-40) and would freeze `beam_width`, `lm_alpha`,
//! `beam_beta` and the hotword list into the model at conversion time —
//! precisely the axes callers change most often. This mirrors the
//! [`beam_search`](vokra_core::decode::beam_search) posture (FR-OP-40) and
//! the [`flow_sample`](crate::flow_sampler) posture (FR-EX-10). The reserved
//! `OpKind` identifier for a future graph-side embedding lives in
//! [`vokra_core::m5_residual_ops::CTC_DECODE_OP`] and is deliberately
//! **unregistered** in the min-dtype registry — the runtime function this
//! module exposes and the reserved graph-op string coexist.
//!
//! # Upstream reference
//!
//! Primary source (as directed):
//! [`NeMo/nemo/collections/asr/parts/submodules/ctc_decoding.py`][nemo-ctc]
//! (`decode_hypothesis`, `fold_consecutive` block, upstream lines ~550-600):
//! the observable core is that greedy CTC decoding
//!
//! 1. takes the per-timestep argmax over the vocabulary, then
//! 2. **collapses consecutive duplicates** and **removes blanks** in one
//!    pass — for every position `p` in the argmax sequence, emit `p` iff
//!    `(p != previous OR previous == blank) AND p != blank`.
//!
//! The greedy submodule
//! [`ctc_greedy_decoding.py::GreedyCTCInfer._greedy_decode_logprobs`][nemo-greedy]
//! confirms the signature: input `x: [T, D]` log-probs, output token list.
//!
//! The beam-search side of NeMo's `ctc_decoding.py`
//! ([`ctc_beam_decoding.py::BeamCTCInfer`][nemo-beam]) delegates the core
//! recursion to external libraries
//! (`BeamSearchDecoderWithLM` / `pyctcdecode` / Flashlight), so the
//! algorithm implemented here follows the textbook **CTC prefix beam
//! search** in log space — Graves et al. 2006
//! (["Connectionist Temporal Classification"][graves-2006]) and Hannun
//! et al. 2014 (["First-Pass Large Vocabulary Continuous Speech Recognition
//! using Bi-Directional Recurrent DNNs", arXiv:1408.2873][hannun-2014]).
//! Its LM-fusion and hotword-boost fields are parametrised the same way
//! NeMo's `BeamCTCInfer` exposes them
//! (`ngram_lm_alpha`, `beam_beta`, `hotwords`, `hotword_weight`;
//! see the primary source), so a caller wiring shallow fusion + hotword
//! boost against this primitive lines up with the NeMo defaults.
//!
//! ## LM shallow fusion (NeMo defaults)
//!
//! ```text
//! score_final(prefix) = log_add(pb, pnb)                        // acoustic
//!                     + lm_alpha  · sum lm_logprob(prefix, tok)  // shallow fusion
//!                     + beam_beta · len(prefix)                  // insertion penalty
//!                     + sum hotword_boost(tok)                   // hotword bonus
//! ```
//!
//! - `lm_alpha`  ~= NeMo `ngram_lm_alpha`
//! - `beam_beta` ~= NeMo `beam_beta` (per-emission length bonus)
//! - `hotwords`  ~= NeMo `pyctcdecode_cfg.hotwords` + `hotword_weight`
//!   folded into per-token boost pairs (this primitive stays token-level;
//!   word-boundary tokenisation is a caller concern)
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Invalid inputs — shape mismatches, `blank_id >= vocab`, `time == 0`,
//! `vocab == 0`, `beam_width == 0`, `n_best == 0`, a non-finite `lm_alpha`
//! or `beam_beta`, a non-finite hotword boost, or a duplicate hotword
//! token — all raise [`VokraError::InvalidArgument`] rather than silently
//! clamping or dropping (mirrors the [`length_conditioning`] and
//! [`beam_search`] contracts).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! No third-party crate, no BLAS, no `serde`. `log_softmax` /
//! `log_add` / top-K live in this file (each a handful of lines) — the
//! root `Cargo.lock` continues to list only `vokra-*` packages.
//!
//! [nemo-ctc]: https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/submodules/ctc_decoding.py
//! [nemo-greedy]: https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/submodules/ctc_greedy_decoding.py
//! [nemo-beam]: https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/submodules/ctc_beam_decoding.py
//! [graves-2006]: https://www.cs.toronto.edu/~graves/icml_2006.pdf
//! [hannun-2014]: https://arxiv.org/abs/1408.2873
//! [`length_conditioning`]: crate::length_conditioning
//! [`beam_search`]: vokra_core::decode::beam_search

use std::collections::HashMap;

use vokra_core::{Result, VokraError};

/// Signature of the optional n-gram / shallow-fusion LM callback used by
/// [`ctc_decode_beam`]: given the emitted prefix so far and a candidate
/// token id, returns an `f32` **log-probability contribution** the beam
/// search adds (weighted by [`CtcBeamAttrs::lm_alpha`]) to that prefix's
/// running LM score. Kept as a top-level alias to satisfy
/// `clippy::type_complexity` on the borrow inside [`CtcBeamAttrs`].
pub type CtcLmScoreFn = dyn Fn(&[u32], u32) -> f32;

// ---------------------------------------------------------------------------
// Public greedy API (NeMo GreedyCTCInfer._greedy_decode_logprobs shape)
// ---------------------------------------------------------------------------

/// Greedy CTC decode: per-timestep argmax over the vocabulary, then
/// **blank fold + consecutive-repeat collapse** (FR-OP-41).
///
/// # Shape
///
/// `logits` is treated as a row-major `[time, vocab]` slice of length
/// `time * vocab`; entry `logits[t * vocab + v]` is the raw model score for
/// vocabulary token `v` at timestep `t`. Softmax is not applied — argmax is
/// invariant under `softmax`, so raw logits and log-probs give bit-identical
/// output.
///
/// # Algorithm (NeMo `decode_hypothesis` `fold_consecutive` block)
///
/// ```text
/// previous = blank
/// for p in argmax_per_timestep:
///     if (p != previous or previous == blank) and p != blank:
///         emit p
///     previous = p
/// ```
///
/// This is the observable core of NeMo's greedy decoder — a single pass over
/// the argmax sequence that emits a token iff it differs from the previous
/// non-blank emission and is not itself blank.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - `time == 0` or `vocab == 0`;
/// - `blank_id >= vocab`;
/// - `logits.len() != time * vocab` (FR-EX-08 — no silent truncation).
///
/// # Ties in argmax
///
/// `f32::partial_cmp` is used (NaN → `None` → equal); on a tie the
/// lowest-index token wins (matches `torch.argmax` for finite input on ties
/// — the first `max` seen in a left-to-right scan). A row containing a NaN
/// still returns a well-defined index (the first `max` seen in scan order);
/// no NaN detection is performed — CTC-time NaNs are a model-side bug the
/// caller must catch (FR-EX-08 posture).
pub fn ctc_decode_greedy(
    logits: &[f32],
    time: usize,
    vocab: usize,
    blank_id: usize,
) -> Result<Vec<u32>> {
    validate_logits_shape(logits, time, vocab, blank_id, "ctc_decode_greedy")?;

    // vocab is validated non-zero above; `blank_id < vocab` guarantees a valid
    // sentinel that no argmax will collide with the "no previous" state.
    let blank = blank_id as u32;

    let mut out: Vec<u32> = Vec::new();
    let mut previous: u32 = blank; // start-of-sequence sentinel: same fold as an actual leading blank
    for t in 0..time {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let p = argmax_row(row);
        // NeMo fold_consecutive: emit only on token change relative to
        // previous, unless previous was blank (post-blank re-emission is a
        // new label), and never emit blank itself.
        if (p != previous || previous == blank) && p != blank {
            out.push(p);
        }
        previous = p;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public beam API (NeMo BeamCTCInfer signature + textbook prefix beam search)
// ---------------------------------------------------------------------------

/// Beam-search attributes for [`ctc_decode_beam`] — n-gram LM shallow fusion
/// and hotword-boost parameters modelled on NeMo's `BeamCTCInfer` signature
/// (see the module-level docstring for the primary source).
///
/// # Lifetime
///
/// `'a` bounds every borrowed field — the hotword table, the optional LM
/// callback. The struct is inert data; the beam search takes it by `&`.
pub struct CtcBeamAttrs<'a> {
    /// Number of timesteps `T` in the `logits` slice.
    pub time: usize,
    /// Vocabulary size `V` in the `logits` slice.
    pub vocab: usize,
    /// Blank token id (`0 <= blank_id < vocab`).
    pub blank_id: usize,
    /// Beam width: number of prefixes kept after each timestep. `1` is
    /// equivalent to greedy up to LM / hotword ranking (the acoustic path
    /// still uses the prefix recursion, not `argmax`, so the two are not
    /// bit-identical).
    pub beam_width: usize,
    /// Number of top hypotheses to return, best-first (`1 <= n_best`).
    pub n_best: usize,
    /// Length insertion penalty (NeMo `beam_beta`). Added to the ranking
    /// score as `beam_beta * n_emitted_labels`. `0.0` disables it.
    pub beam_beta: f32,
    /// LM shallow-fusion weight (NeMo `ngram_lm_alpha`). Added to the
    /// ranking score as `lm_alpha * cumulative_lm_logprob`. Ignored when
    /// [`Self::lm_score_fn`] is `None` (still validated finite).
    pub lm_alpha: f32,
    /// Optional LM callback: given the emitted prefix so far and a
    /// candidate token, returns an `f32` **log-probability contribution**
    /// (log-space) for extending the prefix with that token. Called once
    /// per (prefix, candidate) pair the beam expansion evaluates.
    ///
    /// The default (`None`) skips shallow fusion; [`Self::lm_alpha`] is
    /// then a documented no-op. A stateful n-gram LM is implemented as a
    /// closure over interior-mutable state
    /// (e.g. `RefCell` / `Mutex`), matching the `Fn` bound — the beam
    /// search never asks for mutable access itself.
    pub lm_score_fn: Option<&'a CtcLmScoreFn>,
    /// Per-token hotword boost table: pairs of `(token_id, boost_logprob)`.
    /// Whenever the beam emits a token id present in this table, its boost
    /// is added to that prefix's cumulative hotword bonus. A token must
    /// appear at most once (duplicates raise
    /// [`VokraError::InvalidArgument`]).
    ///
    /// Word-level hotwords (NeMo `pyctcdecode_cfg.hotwords`) are folded to
    /// per-token boosts by the caller — this primitive stays token-level so
    /// it does not need a tokeniser.
    pub hotwords: &'a [(u32, f32)],
}

impl<'a> CtcBeamAttrs<'a> {
    /// A minimal attribute set: no LM, no hotwords, no length penalty.
    /// Useful as a base for tests and callers that want to opt in feature
    /// by feature.
    pub fn plain(time: usize, vocab: usize, blank_id: usize, beam_width: usize) -> Self {
        Self {
            time,
            vocab,
            blank_id,
            beam_width,
            n_best: 1,
            beam_beta: 0.0,
            lm_alpha: 0.0,
            lm_score_fn: None,
            hotwords: &[],
        }
    }
}

/// One CTC beam-search result.
#[derive(Debug, Clone, PartialEq)]
pub struct CtcHypothesis {
    /// Emitted token sequence — already blank-collapsed and
    /// consecutive-duplicate-folded (labels only, no blanks).
    pub tokens: Vec<u32>,
    /// Acoustic log-probability of the emitted labels
    /// (`log_add(pb, pnb)` at the final timestep).
    pub score: f32,
    /// Ranking score after LM shallow fusion, length insertion penalty and
    /// hotword boost:
    /// `score + lm_alpha * lm_score + beam_beta * len + sum(hotword_boosts)`.
    /// Hypotheses are returned sorted by this field, descending.
    pub normalized_score: f32,
}

/// Runs textbook CTC prefix beam search over `logits` with n-gram LM
/// shallow fusion and hotword boost (FR-OP-41).
///
/// # Shape
///
/// `logits` is a row-major `[time, vocab]` slice. Each row is log-softmax'd
/// internally before the recursion, so the caller may pass either raw
/// logits or already-normalised log-probabilities — the primitive treats
/// both identically (`log_softmax(log_softmax(x)) == log_softmax(x)` up to
/// float error, and a normalised input keeps the beam recursion numerically
/// equivalent).
///
/// # Algorithm (Hannun 2014 log-space recursion)
///
/// For each timestep `t`, for each active prefix `l` with `(pb_l, pnb_l)`:
///
/// ```text
/// // 1. Extend l with blank — labels unchanged.
/// pb_new(l) = log_add(pb_new(l), log_add(pb_l, pnb_l) + lp[t, blank])
///
/// // 2. Extend l with each non-blank candidate c:
/// //    a) c equals the last emitted label of l (repeat):
/// pnb_new(l) = log_add(pnb_new(l), pnb_l + lp[t, c])            // stays same label
/// pnb_new(l + [c]) = log_add(pnb_new(l + [c]), pb_l + lp[t, c])  // new label via a blank
/// //    b) c differs from the last emitted label (or l is empty):
/// pnb_new(l + [c]) = log_add(pnb_new(l + [c]),
///                            log_add(pb_l, pnb_l) + lp[t, c])
/// ```
///
/// After every step the prefixes are ranked by
/// `log_add(pb, pnb) + lm_alpha * lm_score + beam_beta * len + hotword_boost`
/// and the top [`CtcBeamAttrs::beam_width`] are kept.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - `time == 0` or `vocab == 0`;
/// - `blank_id >= vocab`;
/// - `logits.len() != time * vocab` (FR-EX-08 — no silent truncation);
/// - `beam_width == 0` or `n_best == 0`;
/// - non-finite `lm_alpha` or `beam_beta`;
/// - a hotword token id `>= vocab`, or a non-finite / duplicate hotword
///   boost (FR-EX-08 — no silent dedup / clamp).
pub fn ctc_decode_beam(logits: &[f32], attrs: &CtcBeamAttrs) -> Result<Vec<CtcHypothesis>> {
    validate_beam_attrs(logits, attrs)?;

    let CtcBeamAttrs {
        time,
        vocab,
        blank_id,
        beam_width,
        n_best,
        beam_beta,
        lm_alpha,
        lm_score_fn,
        hotwords,
    } = *attrs;
    let blank = blank_id as u32;

    // Fold the hotword slice into a lookup table. Duplicate tokens have
    // already been rejected by validate_beam_attrs (FR-EX-08 posture: a
    // duplicate is ambiguous — surface it, do not silently overwrite).
    let hotword_lookup: HashMap<u32, f32> =
        hotwords.iter().map(|&(tok, boost)| (tok, boost)).collect();

    // Beam state, keyed on the emitted label sequence (blank-folded).
    // The empty prefix starts at pb = 0.0 (log 1.0), pnb = -inf: probability
    // 1.0 of being "in the blank state" with no labels emitted, and 0.0
    // probability of having emitted anything yet.
    let mut beams: HashMap<Vec<u32>, BeamState> = HashMap::new();
    beams.insert(
        Vec::new(),
        BeamState {
            pb: 0.0,
            pnb: f32::NEG_INFINITY,
            lm_score: 0.0,
            hotword_boost: 0.0,
        },
    );

    for t in 0..time {
        let raw_row = &logits[t * vocab..(t + 1) * vocab];
        let lp = log_softmax(raw_row);

        // Every prefix from step `t` contributes to a fresh map at step
        // `t + 1`; the recursion accumulates via `log_add` over multiple
        // paths reaching the same prefix.
        let mut next: HashMap<Vec<u32>, BeamState> = HashMap::new();

        for (labels, state) in &beams {
            // (1) Blank extension — labels unchanged, updates pb only.
            let lp_blank = lp[blank as usize];
            let entry = next.entry(labels.clone()).or_insert(BeamState {
                pb: f32::NEG_INFINITY,
                pnb: f32::NEG_INFINITY,
                lm_score: state.lm_score,
                hotword_boost: state.hotword_boost,
            });
            entry.pb = log_add(entry.pb, log_add(state.pb, state.pnb) + lp_blank);

            // (2) Non-blank extensions. We iterate the full vocabulary
            // rather than a per-timestep top-K because the recursion has
            // path-dependent contributions (a "repeat" that stays same-label
            // vs a "same token after blank" that yields a new label): a
            // token that is not top-K acoustic can still be the highest
            // final-rank candidate under LM fusion + hotword boost. Beam
            // pruning happens after the step on the total ranking score.
            for (c, &lp_c) in lp.iter().enumerate() {
                if c == blank_id {
                    continue;
                }
                let c_u32 = c as u32;
                let last = labels.last().copied();

                if last == Some(c_u32) {
                    // (2a-repeat) Same label as prefix's tail: emit c
                    // *without* a separating blank → collapses onto the
                    // existing last label. `pnb` accumulates, labels
                    // unchanged.
                    let entry_same = next.entry(labels.clone()).or_insert(BeamState {
                        pb: f32::NEG_INFINITY,
                        pnb: f32::NEG_INFINITY,
                        lm_score: state.lm_score,
                        hotword_boost: state.hotword_boost,
                    });
                    entry_same.pnb = log_add(entry_same.pnb, state.pnb + lp_c);

                    // (2a-new) Same label as prefix's tail but through a
                    // blank: extends labels with a NEW c (post-blank
                    // re-emission counts as a new label).
                    let mut new_labels = labels.clone();
                    new_labels.push(c_u32);
                    let (delta_lm, delta_hot) =
                        new_label_deltas(labels, c_u32, lm_score_fn, lm_alpha, &hotword_lookup);
                    let entry_new = next.entry(new_labels).or_insert(BeamState {
                        pb: f32::NEG_INFINITY,
                        pnb: f32::NEG_INFINITY,
                        lm_score: state.lm_score + delta_lm,
                        hotword_boost: state.hotword_boost + delta_hot,
                    });
                    entry_new.pnb = log_add(entry_new.pnb, state.pb + lp_c);
                } else {
                    // (2b) Different label (or empty prefix): extend labels
                    // with c. Both blank-suffix and label-suffix paths
                    // contribute.
                    let mut new_labels = labels.clone();
                    new_labels.push(c_u32);
                    let (delta_lm, delta_hot) =
                        new_label_deltas(labels, c_u32, lm_score_fn, lm_alpha, &hotword_lookup);
                    let entry_new = next.entry(new_labels).or_insert(BeamState {
                        pb: f32::NEG_INFINITY,
                        pnb: f32::NEG_INFINITY,
                        lm_score: state.lm_score + delta_lm,
                        hotword_boost: state.hotword_boost + delta_hot,
                    });
                    entry_new.pnb = log_add(entry_new.pnb, log_add(state.pb, state.pnb) + lp_c);
                }
            }
        }

        // Prune to top `beam_width` by total ranking score.
        beams = prune_beams(next, beam_width, lm_alpha, beam_beta);
        if beams.is_empty() {
            // Every path went to -inf. This is a degenerate condition (a
            // -inf logit column, e.g. from a masked timestep); return no
            // hypotheses rather than fabricating a random beam.
            return Ok(Vec::new());
        }
    }

    // Emit hypotheses ranked by ranking score, descending.
    let mut hyps: Vec<CtcHypothesis> = beams
        .into_iter()
        .map(|(tokens, state)| {
            let acoustic = log_add(state.pb, state.pnb);
            let normalized = acoustic
                + lm_alpha * state.lm_score
                + beam_beta * tokens.len() as f32
                + state.hotword_boost;
            CtcHypothesis {
                tokens,
                score: acoustic,
                normalized_score: normalized,
            }
        })
        .collect();
    hyps.sort_by(|a, b| {
        b.normalized_score
            .partial_cmp(&a.normalized_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hyps.truncate(n_best);
    Ok(hyps)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Cumulative state carried by one prefix through the beam recursion.
#[derive(Debug, Clone)]
struct BeamState {
    /// `log P(prefix, last emission was a blank)` up to the current time.
    pb: f32,
    /// `log P(prefix, last emission was `prefix.last()`)` up to the
    /// current time. `NEG_INFINITY` for the empty prefix.
    pnb: f32,
    /// Cumulative LM shallow-fusion score (sum of per-label
    /// `lm_score_fn` returns), **without** the `lm_alpha` multiplier —
    /// that multiplier is applied at ranking time so a caller can retrieve
    /// the raw LM contribution from `normalized_score - beta*len - score`.
    lm_score: f32,
    /// Cumulative hotword boost (sum of per-emitted-token boosts from the
    /// hotword table).
    hotword_boost: f32,
}

/// Validates the `[time, vocab]` shape of a logits slice + the blank id
/// for both the greedy and beam entry points (FR-EX-08).
fn validate_logits_shape(
    logits: &[f32],
    time: usize,
    vocab: usize,
    blank_id: usize,
    ctx: &str,
) -> Result<()> {
    if time == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{ctx}: time must be >= 1 (got 0)"
        )));
    }
    if vocab == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{ctx}: vocab must be >= 1 (got 0)"
        )));
    }
    if blank_id >= vocab {
        return Err(VokraError::InvalidArgument(format!(
            "{ctx}: blank_id {blank_id} >= vocab {vocab} (no silent clamp — FR-EX-08)"
        )));
    }
    let expected = time.checked_mul(vocab).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{ctx}: time {time} * vocab {vocab} overflows usize"
        ))
    })?;
    if logits.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{ctx}: logits.len() {} != time*vocab {expected} (time={time}, vocab={vocab})",
            logits.len()
        )));
    }
    Ok(())
}

/// Validates the full [`CtcBeamAttrs`] payload up-front (FR-EX-08 — every
/// bad input surfaces before any recursion runs).
fn validate_beam_attrs(logits: &[f32], attrs: &CtcBeamAttrs) -> Result<()> {
    validate_logits_shape(
        logits,
        attrs.time,
        attrs.vocab,
        attrs.blank_id,
        "ctc_decode_beam",
    )?;
    if attrs.beam_width == 0 {
        return Err(VokraError::InvalidArgument(
            "ctc_decode_beam: beam_width must be >= 1".to_owned(),
        ));
    }
    if attrs.n_best == 0 {
        return Err(VokraError::InvalidArgument(
            "ctc_decode_beam: n_best must be >= 1".to_owned(),
        ));
    }
    if !attrs.lm_alpha.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "ctc_decode_beam: lm_alpha must be finite (got {})",
            attrs.lm_alpha
        )));
    }
    if !attrs.beam_beta.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "ctc_decode_beam: beam_beta must be finite (got {})",
            attrs.beam_beta
        )));
    }
    // Hotword table: bound checks + finite boost + no duplicates
    // (FR-EX-08 — duplicates are ambiguous, do not silently overwrite).
    let mut seen: Vec<u32> = Vec::with_capacity(attrs.hotwords.len());
    for &(tok, boost) in attrs.hotwords {
        if (tok as usize) >= attrs.vocab {
            return Err(VokraError::InvalidArgument(format!(
                "ctc_decode_beam: hotword token {tok} >= vocab {} (no silent drop — FR-EX-08)",
                attrs.vocab
            )));
        }
        if !boost.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "ctc_decode_beam: hotword boost for token {tok} must be finite (got {boost})"
            )));
        }
        if seen.contains(&tok) {
            return Err(VokraError::InvalidArgument(format!(
                "ctc_decode_beam: hotword token {tok} appears more than once (no silent \
                 dedup — FR-EX-08)"
            )));
        }
        seen.push(tok);
    }
    Ok(())
}

/// Returns the index of the maximum entry of `row`.
///
/// Uses `partial_cmp` (NaN → `None` → equal), so the lowest index of the
/// max value wins on ties. NaN handling: a row containing a NaN still
/// returns a well-defined index (the first `max` seen in scan order);
/// callers pushing NaN into a CTC decode have a model-side bug that the
/// caller must catch (FR-EX-08 — no silent NaN suppression here).
fn argmax_row(row: &[f32]) -> u32 {
    debug_assert!(!row.is_empty(), "argmax_row: row must be non-empty");
    let mut best_idx: u32 = 0;
    let mut best_val: f32 = row[0];
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v.partial_cmp(&best_val) == Some(std::cmp::Ordering::Greater) {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Numerically stable `log(exp(a) + exp(b))`. Standard identity:
/// `max + log1p(exp(min - max))`.
fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    max + (min - max).exp().ln_1p()
}

/// Row-level `log_softmax` — subtract the max, then `x - (max + log(sum
/// exp(x - max)))`. Preserves numerical stability without allocating a
/// scratch `exp` table.
fn log_softmax(row: &[f32]) -> Vec<f32> {
    debug_assert!(!row.is_empty(), "log_softmax: row must be non-empty");
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        // All -inf (or NaN — see `argmax_row` docstring): every entry maps
        // to -inf. This surfaces further down as an empty beam.
        return vec![f32::NEG_INFINITY; row.len()];
    }
    let sum_exp: f32 = row.iter().map(|&x| (x - m).exp()).sum();
    let log_z = m + sum_exp.ln();
    row.iter().map(|&x| x - log_z).collect()
}

/// Per-new-emission LM + hotword delta the beam recursion needs to fold
/// into a prefix that just gained a new label `c`.
///
/// Kept as a helper (rather than inlined twice at (2a-new) and (2b)) so
/// the two label-emitting branches share the exact same delta arithmetic
/// — a divergence between them would silently misprice the LM path.
fn new_label_deltas(
    labels_before: &[u32],
    c: u32,
    lm_score_fn: Option<&CtcLmScoreFn>,
    lm_alpha: f32,
    hotwords: &HashMap<u32, f32>,
) -> (f32, f32) {
    let delta_lm = match lm_score_fn {
        // `lm_alpha` is applied at ranking time (see `BeamState.lm_score`
        // doc), not folded here — that keeps `state.lm_score` interpretable
        // as a raw log-prob independent of the caller's `alpha`. The
        // `lm_alpha == 0` short-circuit avoids the closure call when the
        // caller has explicitly disabled shallow fusion.
        Some(f) if lm_alpha != 0.0 => f(labels_before, c),
        _ => 0.0,
    };
    let delta_hot = hotwords.get(&c).copied().unwrap_or(0.0);
    (delta_lm, delta_hot)
}

/// Prunes `beams` to the top `beam_width` by total ranking score.
fn prune_beams(
    beams: HashMap<Vec<u32>, BeamState>,
    beam_width: usize,
    lm_alpha: f32,
    beam_beta: f32,
) -> HashMap<Vec<u32>, BeamState> {
    // Drop dead beams (both pb and pnb == -inf: no acoustic mass at all).
    let mut ranked: Vec<(Vec<u32>, BeamState, f32)> = beams
        .into_iter()
        .filter(|(_, s)| s.pb != f32::NEG_INFINITY || s.pnb != f32::NEG_INFINITY)
        .map(|(labels, state)| {
            let rank = log_add(state.pb, state.pnb)
                + lm_alpha * state.lm_score
                + beam_beta * labels.len() as f32
                + state.hotword_boost;
            (labels, state, rank)
        })
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(beam_width);
    ranked
        .into_iter()
        .map(|(labels, state, _)| (labels, state))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- happy path: greedy ----------------------------------------------

    /// Blank id = 0, vocab = 3 (labels A=1, B=2). A stream of
    /// `[A, A, blank, A, B, B]` should collapse to `[A, A, B]`:
    /// - first A emitted;
    /// - second A folded (consecutive repeat);
    /// - blank clears the "previous label" latch;
    /// - the third A is a new emission (post-blank re-emission);
    /// - first B emitted (different from A);
    /// - second B folded (consecutive repeat).
    #[test]
    fn greedy_folds_consecutive_and_removes_blanks() {
        let vocab = 3;
        let blank = 0;
        // Build logits by placing a large positive value in the target
        // column at each timestep and zeros elsewhere — argmax then picks
        // the target column deterministically.
        let time = 6;
        let mut logits = vec![0.0f32; time * vocab];
        let targets = [1u32, 1, 0, 1, 2, 2];
        for (t, &tok) in targets.iter().enumerate() {
            logits[t * vocab + tok as usize] = 10.0;
        }
        let out = ctc_decode_greedy(&logits, time, vocab, blank).unwrap();
        assert_eq!(out, vec![1u32, 1, 2]);
    }

    /// A blank-only sequence collapses to the empty label list — the
    /// "silent audio" degenerate case, which must be legal (not an error).
    #[test]
    fn greedy_blank_only_yields_empty_sequence() {
        let vocab = 4;
        let blank = 3;
        let time = 5;
        let mut logits = vec![0.0f32; time * vocab];
        for t in 0..time {
            logits[t * vocab + blank] = 5.0;
        }
        let out = ctc_decode_greedy(&logits, time, vocab, blank).unwrap();
        assert!(
            out.is_empty(),
            "blank-only greedy sequence must fold to empty"
        );
    }

    /// Argmax invariance: raw logits and their log-softmax must produce
    /// the exact same greedy output (softmax preserves argmax on finite
    /// input, so no bit change).
    #[test]
    fn greedy_argmax_is_invariant_under_softmax() {
        let vocab = 4;
        let blank = 0;
        let time = 3;
        // Deterministic, non-trivial logits.
        let logits = vec![
            0.1, 2.0, 0.5, -1.0, // t=0 argmax = 1
            0.0, 0.0, 3.0, 0.0, // t=1 argmax = 2
            0.0, 0.0, 0.0, 4.5, // t=2 argmax = 3
        ];
        let out_raw = ctc_decode_greedy(&logits, time, vocab, blank).unwrap();
        // Manually log-softmax each row and compare.
        let mut lp = Vec::with_capacity(logits.len());
        for t in 0..time {
            lp.extend_from_slice(&log_softmax(&logits[t * vocab..(t + 1) * vocab]));
        }
        let out_lp = ctc_decode_greedy(&lp, time, vocab, blank).unwrap();
        assert_eq!(out_raw, out_lp);
    }

    /// Ties in argmax resolve to the lowest index (torch.argmax-compatible
    /// on ties for a left-to-right scan).
    #[test]
    fn greedy_argmax_ties_pick_lowest_index() {
        let vocab = 3;
        let blank = 2;
        let time = 1;
        // All zeros: argmax = 0. blank is 2, so 0 gets emitted.
        let logits = vec![0.0f32; vocab * time];
        let out = ctc_decode_greedy(&logits, time, vocab, blank).unwrap();
        assert_eq!(out, vec![0u32]);
    }

    // ---- shape validation: greedy ----------------------------------------

    #[test]
    fn greedy_rejects_time_zero() {
        let err = ctc_decode_greedy(&[], 0, 4, 0).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn greedy_rejects_vocab_zero() {
        let err = ctc_decode_greedy(&[], 3, 0, 0).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn greedy_rejects_blank_id_out_of_range() {
        let logits = vec![0.0f32; 3 * 4];
        let err = ctc_decode_greedy(&logits, 3, 4, 4).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn greedy_rejects_shape_mismatch() {
        // 3*4 = 12 expected, 11 provided.
        let logits = vec![0.0f32; 11];
        let err = ctc_decode_greedy(&logits, 3, 4, 0).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("logits.len()"), "message: {msg}");
                assert!(msg.contains("11"), "message: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    // ---- happy path: beam -------------------------------------------------

    /// Beam with `beam_width = 1` on a peaky distribution returns the same
    /// labels the greedy path would (up to the ranking-score fields — the
    /// tokens must be identical).
    #[test]
    fn beam_width_one_matches_greedy_labels_on_peaky_input() {
        let vocab = 3;
        let blank = 0;
        let time = 6;
        let mut logits = vec![-10.0f32; time * vocab]; // background floor
        let targets = [1u32, 1, 0, 1, 2, 2];
        for (t, &tok) in targets.iter().enumerate() {
            logits[t * vocab + tok as usize] = 10.0;
        }
        let attrs = CtcBeamAttrs::plain(time, vocab, blank, 1);
        let hyps = ctc_decode_beam(&logits, &attrs).unwrap();
        assert_eq!(hyps.len(), 1);
        let greedy = ctc_decode_greedy(&logits, time, vocab, blank).unwrap();
        assert_eq!(hyps[0].tokens, greedy);
        // Acoustic score is a marginal log-probability
        // (`log_add(pb, pnb)`). It must be finite and near-zero for a
        // peaky prefix that captures almost all mass — a small positive
        // drift is allowed because `log_add(0.0, -20.0)` computes
        // `log1p(exp(-20)) ≈ 4.5e-9 > 0` in f32 arithmetic even though
        // the underlying mass identity is `<= 1`. `score >> 0` would
        // still catch an accounting bug.
        assert!(hyps[0].score.is_finite());
        assert!(
            hyps[0].score < 0.5,
            "acoustic score must be near or below 0: {}",
            hyps[0].score
        );
    }

    /// N-best sorting: with `beam_width = 4` and `n_best = 3` we get up to
    /// three hypotheses, sorted by `normalized_score` descending.
    #[test]
    fn beam_returns_nbest_sorted_desc() {
        let vocab = 4;
        let blank = 0;
        let time = 4;
        // Ambiguous per-timestep distribution: two plausible tokens per
        // frame. This produces multiple competing prefixes.
        let mut logits = vec![-2.0f32; time * vocab];
        for t in 0..time {
            logits[t * vocab + 1] = 0.5;
            logits[t * vocab + 2] = 0.4;
        }
        let mut attrs = CtcBeamAttrs::plain(time, vocab, blank, 4);
        attrs.n_best = 3;
        let hyps = ctc_decode_beam(&logits, &attrs).unwrap();
        assert!(hyps.len() <= 3);
        for w in hyps.windows(2) {
            assert!(
                w[0].normalized_score >= w[1].normalized_score,
                "n-best must be sorted descending: {w:?}"
            );
        }
    }

    /// LM shallow fusion + `lm_alpha > 0` reranks the top hypothesis
    /// toward the LM's preferred sequence — a change vs the acoustic-only
    /// ranking, so the top-1 label must flip when the LM strongly prefers
    /// a different token.
    #[test]
    fn beam_lm_shallow_fusion_reranks_toward_lm() {
        let vocab = 4;
        let blank = 0;
        let time = 4;
        // Acoustic slightly prefers token 1 over token 2 every frame.
        let mut logits = vec![-4.0f32; time * vocab];
        for t in 0..time {
            logits[t * vocab + 1] = 0.5; // slightly favored acoustic
            logits[t * vocab + 2] = 0.3; // slightly weaker acoustic
        }
        // Acoustic-only baseline: expect a single "1" (the collapsed
        // sequence — every timestep argmax is 1 and consecutive repeats
        // fold).
        let baseline =
            ctc_decode_beam(&logits, &CtcBeamAttrs::plain(time, vocab, blank, 4)).unwrap();
        assert!(baseline[0].tokens.contains(&1u32));

        // LM strongly boosts token 2 whenever it is a candidate; with
        // `lm_alpha = 2.0`, the ranking must flip to a prefix containing 2.
        let lm_boost = |_prefix: &[u32], c: u32| -> f32 { if c == 2 { 4.0 } else { 0.0 } };
        let boxed: Box<CtcLmScoreFn> = Box::new(lm_boost);
        let mut attrs = CtcBeamAttrs::plain(time, vocab, blank, 4);
        attrs.lm_alpha = 2.0;
        attrs.lm_score_fn = Some(&*boxed);
        let biased = ctc_decode_beam(&logits, &attrs).unwrap();
        assert!(
            biased[0].tokens.contains(&2u32),
            "LM boost must rerank toward token 2, got {:?}",
            biased[0].tokens
        );
    }

    /// Hotword boost adds a bonus per emission — a strong boost on a
    /// weakly acoustic token surfaces it at rank 1.
    #[test]
    fn beam_hotword_boost_lifts_hotword_token() {
        let vocab = 4;
        let blank = 0;
        let time = 3;
        let mut logits = vec![-4.0f32; time * vocab];
        for t in 0..time {
            logits[t * vocab + 1] = 0.6; // acoustic winner
            logits[t * vocab + 3] = 0.4; // slightly weaker
        }
        // Baseline: token 1 wins.
        let baseline =
            ctc_decode_beam(&logits, &CtcBeamAttrs::plain(time, vocab, blank, 4)).unwrap();
        assert!(baseline[0].tokens.contains(&1u32));

        // Big boost on token 3; ranking flips.
        let mut attrs = CtcBeamAttrs::plain(time, vocab, blank, 4);
        let hotwords: [(u32, f32); 1] = [(3u32, 5.0)];
        attrs.hotwords = &hotwords;
        let boosted = ctc_decode_beam(&logits, &attrs).unwrap();
        assert!(
            boosted[0].tokens.contains(&3u32),
            "hotword boost must lift token 3 to rank 1, got {:?}",
            boosted[0].tokens
        );
    }

    /// A beam expansion whose every next-step candidate is `-inf` (a fully
    /// masked timestep) returns no hypotheses rather than fabricating a
    /// random beam — the degenerate degenerate-input case.
    #[test]
    fn beam_all_neg_inf_timestep_returns_empty() {
        let vocab = 3;
        let blank = 0;
        let time = 2;
        let mut logits = vec![f32::NEG_INFINITY; time * vocab];
        // Timestep 0 has a valid distribution so a beam gets seeded.
        logits[0] = 0.0;
        logits[1] = 0.0;
        logits[2] = 0.0;
        // Timestep 1 is fully -inf: no path can extend.
        let attrs = CtcBeamAttrs::plain(time, vocab, blank, 2);
        let hyps = ctc_decode_beam(&logits, &attrs).unwrap();
        assert!(
            hyps.is_empty(),
            "fully -inf timestep must return no hypotheses (no fabrication)"
        );
    }

    // ---- shape / arg validation: beam ------------------------------------

    #[test]
    fn beam_rejects_time_zero() {
        let attrs = CtcBeamAttrs::plain(0, 4, 0, 2);
        let err = ctc_decode_beam(&[], &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn beam_rejects_blank_id_out_of_range() {
        let logits = vec![0.0f32; 3 * 4];
        let attrs = CtcBeamAttrs::plain(3, 4, 4, 2);
        let err = ctc_decode_beam(&logits, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn beam_rejects_shape_mismatch() {
        let logits = vec![0.0f32; 10]; // 3*4=12 expected
        let attrs = CtcBeamAttrs::plain(3, 4, 0, 2);
        let err = ctc_decode_beam(&logits, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn beam_rejects_zero_beam_width() {
        let logits = vec![0.0f32; 3 * 4];
        let attrs = CtcBeamAttrs::plain(3, 4, 0, 0);
        let err = ctc_decode_beam(&logits, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn beam_rejects_zero_n_best() {
        let logits = vec![0.0f32; 3 * 4];
        let mut attrs = CtcBeamAttrs::plain(3, 4, 0, 2);
        attrs.n_best = 0;
        let err = ctc_decode_beam(&logits, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn beam_rejects_non_finite_lm_alpha_or_beta() {
        let logits = vec![0.0f32; 3 * 4];
        let mut a = CtcBeamAttrs::plain(3, 4, 0, 2);
        a.lm_alpha = f32::NAN;
        assert!(matches!(
            ctc_decode_beam(&logits, &a).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));

        let mut b = CtcBeamAttrs::plain(3, 4, 0, 2);
        b.beam_beta = f32::INFINITY;
        assert!(matches!(
            ctc_decode_beam(&logits, &b).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn beam_rejects_hotword_out_of_range() {
        let logits = vec![0.0f32; 3 * 4];
        let mut a = CtcBeamAttrs::plain(3, 4, 0, 2);
        let hot: [(u32, f32); 1] = [(4u32, 1.0)];
        a.hotwords = &hot;
        assert!(matches!(
            ctc_decode_beam(&logits, &a).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn beam_rejects_non_finite_hotword_boost() {
        let logits = vec![0.0f32; 3 * 4];
        let mut a = CtcBeamAttrs::plain(3, 4, 0, 2);
        let hot: [(u32, f32); 1] = [(1u32, f32::NAN)];
        a.hotwords = &hot;
        assert!(matches!(
            ctc_decode_beam(&logits, &a).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    #[test]
    fn beam_rejects_duplicate_hotword_token() {
        let logits = vec![0.0f32; 3 * 4];
        let mut a = CtcBeamAttrs::plain(3, 4, 0, 2);
        let hot: [(u32, f32); 2] = [(1u32, 0.5), (1u32, 1.0)];
        a.hotwords = &hot;
        assert!(matches!(
            ctc_decode_beam(&logits, &a).unwrap_err(),
            VokraError::InvalidArgument(_)
        ));
    }

    // ---- helper unit tests -----------------------------------------------

    #[test]
    fn log_add_handles_neg_inf() {
        // -inf + x = x
        assert_eq!(log_add(f32::NEG_INFINITY, -1.5), -1.5);
        assert_eq!(log_add(-1.5, f32::NEG_INFINITY), -1.5);
        // Two -inf stay -inf.
        assert_eq!(
            log_add(f32::NEG_INFINITY, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn log_add_matches_stable_identity() {
        // log_add(a, b) ≈ log(exp(a) + exp(b)); check within f32 tolerance.
        let a = -0.4f32;
        let b = -1.2f32;
        let got = log_add(a, b);
        let want = (a.exp() + b.exp()).ln();
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    #[test]
    fn log_softmax_sums_to_zero_in_prob_space() {
        // Σ exp(log_softmax(x)) == 1 (within f32 tolerance).
        let x = [0.1f32, 2.0, -3.0, 0.5];
        let lp = log_softmax(&x);
        let sum: f32 = lp.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn log_softmax_all_neg_inf_returns_neg_inf() {
        // A fully-masked row must not divide by zero; it maps to -inf.
        let lp = log_softmax(&[f32::NEG_INFINITY; 3]);
        assert!(lp.iter().all(|v| *v == f32::NEG_INFINITY));
    }

    #[test]
    fn argmax_row_picks_first_max_on_ties() {
        assert_eq!(argmax_row(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax_row(&[-1.0, 3.0, 3.0]), 1);
        assert_eq!(argmax_row(&[-1.0, -1.0, 3.0]), 2);
    }
}
