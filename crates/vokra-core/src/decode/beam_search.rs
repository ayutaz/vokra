//! Model-independent beam search (FR-OP-40).
//!
//! `beam_search` is a **host-side runtime function**, never a graph op (which
//! would break execution-provider compatibility — the "contrib op"
//! anti-pattern, FR-OP-40). Models plug in through the [`BeamScorer`] trait:
//! the search knows nothing about Whisper, attention or KV caches, only how to
//! ask "given this token prefix, what are the next-token log-probabilities?".
//!
//! # FR-OP-40 attributes
//!
//! [`BeamSearchConfig`] exposes all five: `beam_width`, `length_normalization`,
//! `early_stopping`, `n_best` and `word_timestamps`. `word_timestamps`
//! (implemented in **M4-20**) attaches word-level timings to the best
//! hypothesis via the scorer's [`BeamScorer::align_words`]; when the scorer
//! supplies no alignment it is an explicit [`VokraError::UnsupportedOp`], never
//! a silent no-op (FR-EX-08, ADR M4-20 §D-3 — this replaced the M0
//! `NotImplemented` gate). `max_new_tokens` is an operational termination bound
//! beyond the five attributes.
//!
//! # M0 simplicity
//!
//! The scorer is queried with the **full token prefix** each expansion, so it
//! stays model-independent and the search needs no per-beam state plumbing.
//! A model may recompute or use a cache internally; efficient per-beam cache
//! reuse/reordering (FR-EX-02 / M1-04) is a model-side optimization that does
//! not change this interface. Data structures avoid interior sharing so a
//! future static-arena pass (FR-EX-05) is not precluded.

use crate::decode::word_timing::WordTiming;
use crate::error::{Result, VokraError};

/// Optional shallow-fusion LM (FR-OP-41 / FR-OP-42): given the current token
/// prefix and a candidate next token, returns an `f32` **log-probability
/// contribution** the beam search folds into that candidate's per-step score
/// (multiplied by [`LmFusionConfig::weight`]) before the beam-level top-K
/// truncation. Same posture as the per-op LM callbacks in
/// [`vokra_ops::ctc_decode::CtcLmScoreFn`] and
/// [`vokra_ops::hybrid_ctc_attention::LmScoreFn`] — a plain trait so the
/// concrete LM (n-gram back-off, LSTM, transformer-LM, ...) is a call-site
/// choice, not a search-side dependency.
///
/// `score` takes `&self`, mirroring the read-only nature of an LM lookup;
/// stateful LMs place their mutable cache behind interior mutability
/// (`RefCell` / `Mutex`) and stay `LmScorer + ?Sync`.
pub trait LmScorer {
    /// Log-probability contribution for extending `context` (the FULL token
    /// prefix so far — search prefix + generated tokens) with `next`.
    fn score(&self, context: &[u32], next: u32) -> f32;
}

/// Shallow-fusion binding: an LM plus its scalar weight `α`. The search adds
/// `α · lm.score(prefix, tok)` to every per-candidate acoustic score BEFORE
/// the beam-level `truncate(beam_width)`, so an LM-boosted hypothesis may
/// survive a step it would otherwise be pruned in.
///
/// `weight == 0.0` short-circuits the LM call (mirrors
/// [`vokra_ops::hybrid_ctc_attention`] line 231) — so a caller who wires a
/// scorer whose `score` returns `-inf` for OOV entries never trips
/// `0.0 * -inf = NaN`. This is what makes the "weight = 0.0 is bit-identical
/// to lm_fusion = None" invariant hold.
pub struct LmFusionConfig<'a> {
    /// The scoring LM.
    pub lm: &'a dyn LmScorer,
    /// Shallow-fusion weight (`α`). `0.0` disables the LM without a call.
    pub weight: f32,
}

impl<'a> Clone for LmFusionConfig<'a> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> Copy for LmFusionConfig<'a> {}

impl<'a> std::fmt::Debug for LmFusionConfig<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmFusionConfig")
            .field("lm", &"<dyn LmScorer>")
            .field("weight", &self.weight)
            .finish()
    }
}

/// Scores next tokens for the beam search (model-independent interface).
pub trait BeamScorer {
    /// Returns the **log-probabilities** over the whole vocabulary for the
    /// token following `tokens` (the full sequence so far, including any forced
    /// prefix). Implementations return normalized log-probs (e.g. a
    /// `log_softmax` of the model logits), length `vocab_size()`.
    fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Vocabulary size (the length of every [`logprobs`](Self::logprobs) result).
    fn vocab_size(&self) -> usize;

    /// Batched [`logprobs`](Self::logprobs): one log-prob vector per prefix, in
    /// the **same order** as `prefixes` (M5-14-BACKLOG-T07). `beam_search`
    /// expands every active beam through this in one call, so a scorer with a
    /// batched decoder step folds `beam_width` per-beam forwards into one
    /// batched forward.
    ///
    /// The default loops [`logprobs`](Self::logprobs) in order — byte-for-byte
    /// identical to per-beam scoring, so every existing scorer is unaffected. An
    /// override must return the same bits as the loop (its parity oracle pins
    /// this); `beam_search` applies the per-beam no-repeat-n-gram mask and the
    /// top-K selection to the returned vectors unchanged, so batching cannot
    /// alter which hypotheses are kept.
    fn logprobs_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        prefixes.iter().map(|p| self.logprobs(p)).collect()
    }

    /// Word-level timestamps for a completed hypothesis (M4-20, FR-OP-40
    /// `word_timestamps`). Called **only** when
    /// [`BeamSearchConfig::word_timestamps`] is set, on the best hypothesis's
    /// full token sequence, after the search finished.
    ///
    /// The default returns `Ok(None)` — "this scorer supplies no alignment".
    /// A model that can align (e.g. Whisper via cross-attention DTW, ADR
    /// M4-20 §D-3) overrides this to return `Ok(Some(timings))`. Per FR-EX-08
    /// [`beam_search`] turns a `None` under `word_timestamps` into an explicit
    /// [`VokraError::UnsupportedOp`], never a silent no-op — so the default
    /// keeps every existing scorer backward-compatible while making the
    /// "requested but unavailable" case loud.
    fn align_words(&mut self, _tokens: &[u32]) -> Result<Option<Vec<WordTiming>>> {
        Ok(None)
    }
}

/// Beam-search attributes (FR-OP-40) plus an operational length cap.
///
/// The `'a` lifetime is threaded through [`Self::lm_fusion`] so the shallow-
/// fusion LM can be a borrowed reference (typically to a `NgramLm` or an
/// LSTM cell owned by the caller). Callers that do not enable LM fusion
/// ignore the lifetime — `BeamSearchConfig::greedy(N)` / `::new(w, N)` seed
/// `lm_fusion: None`, elision handles the `'a` at every existing call site.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig<'a> {
    /// Number of hypotheses kept per step (`beam_width = 1` == greedy).
    pub beam_width: usize,
    /// Length-penalty exponent `α` (HF `length_penalty`): a completed
    /// hypothesis is ranked by `score / len^α`, where `len` is the number of
    /// generated tokens. `0.0` disables normalization; `1.0` divides by length.
    pub length_normalization: f32,
    /// When `true`, stop as soon as `beam_width` completed hypotheses exist.
    /// When `false`, keep expanding until every beam has finished or the length
    /// cap is hit.
    pub early_stopping: bool,
    /// Number of hypotheses to return, best-first.
    pub n_best: usize,
    /// Emit word-level timestamps (M4-20). When set, the best hypothesis's
    /// [`BeamHypothesis::word_timestamps`] is filled from
    /// [`BeamScorer::align_words`]; a scorer with no alignment support is an
    /// explicit [`VokraError::UnsupportedOp`] (FR-EX-08).
    pub word_timestamps: bool,
    /// Maximum number of tokens to generate past the prefix (operational bound,
    /// not an FR-OP-40 attribute).
    pub max_new_tokens: usize,
    /// If `> 0`, forbid any candidate token that would form a repeated
    /// `n`-gram in that beam's already-generated tokens (Wave 12, M3-15
    /// follow-up). HuggingFace-compatible semantics: the n-gram is checked
    /// against the entire prefix + generated sequence, and blocked candidates
    /// have their per-step log-prob set to `-inf` before the top-K selection
    /// — a hard mask, not a soft discount. `0` disables blocking (default).
    ///
    /// Ported verbatim from
    /// `vokra_models::voxtral::beam_search::BeamConfig::no_repeat_ngram_size`
    /// so both search primitives share the same semantics.
    pub no_repeat_ngram_size: usize,
    /// Optional shallow-fusion LM (FR-OP-41 / FR-OP-42). When `Some`, the
    /// search adds `lm_fusion.weight · lm_fusion.lm.score(prefix, tok)` to
    /// every per-candidate acoustic score BEFORE the beam-level
    /// `truncate(beam_width)`, so an LM-boosted hypothesis may survive a
    /// step it would otherwise be pruned in.
    ///
    /// `None` (default) is bit-identical to omitting the field: the search
    /// takes the pre-LM code path. `Some { weight: 0.0, .. }` is also
    /// bit-identical to `None` — the scorer is never consulted, so a scorer
    /// that returns `-inf` for OOV entries does not trip `0.0 * -inf = NaN`
    /// (mirrors the [`vokra_ops::hybrid_ctc_attention`] `lm_weight == 0.0`
    /// short-circuit).
    pub lm_fusion: Option<LmFusionConfig<'a>>,
}

impl<'a> BeamSearchConfig<'a> {
    /// A greedy-equivalent config: `beam_width = 1`, no normalization, one
    /// result, no timestamps, no n-gram blocking, no LM fusion.
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            beam_width: 1,
            length_normalization: 0.0,
            early_stopping: true,
            n_best: 1,
            word_timestamps: false,
            max_new_tokens,
            no_repeat_ngram_size: 0,
            lm_fusion: None,
        }
    }

    /// A standard beam config of the given width (one best result), no n-gram
    /// blocking, no LM fusion.
    pub fn new(beam_width: usize, max_new_tokens: usize) -> Self {
        Self {
            beam_width,
            length_normalization: 1.0,
            early_stopping: true,
            n_best: 1,
            word_timestamps: false,
            max_new_tokens,
            no_repeat_ngram_size: 0,
            lm_fusion: None,
        }
    }
}

/// One beam-search result.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamHypothesis {
    /// Full token sequence, including the forced prefix and the terminal
    /// end-of-sequence token when the hypothesis finished.
    pub tokens: Vec<u32>,
    /// Cumulative log-probability of the generated tokens.
    pub score: f32,
    /// Length-normalized ranking score (`score / len^α`).
    pub normalized_score: f32,
    /// Word-level timings (M4-20, FR-OP-40). `None` unless
    /// [`BeamSearchConfig::word_timestamps`] was set **and** the scorer
    /// supplied an alignment; only the best (returned-first) hypothesis is
    /// aligned. Additive field — existing callers ignore it.
    pub word_timestamps: Option<Vec<WordTiming>>,
}

/// Runs beam search from `prefix`, expanding until `eot`, all beams finish, or
/// [`BeamSearchConfig::max_new_tokens`] is reached.
///
/// Returns up to [`BeamSearchConfig::n_best`] hypotheses, best (highest
/// normalized score) first, with duplicate token sequences removed.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] for an empty `prefix`, `beam_width == 0`
///   or `n_best == 0`;
/// - [`VokraError::UnsupportedOp`] if `word_timestamps` is set but the
///   [`BeamScorer`] supplies no alignment (FR-EX-08, M4-20);
/// - any error surfaced by the [`BeamScorer`].
pub fn beam_search<'a>(
    scorer: &mut dyn BeamScorer,
    prefix: &[u32],
    eot: u32,
    config: &BeamSearchConfig<'a>,
) -> Result<Vec<BeamHypothesis>> {
    if prefix.is_empty() {
        return Err(VokraError::InvalidArgument(
            "beam_search: prefix must not be empty".into(),
        ));
    }
    if config.beam_width == 0 || config.n_best == 0 {
        return Err(VokraError::InvalidArgument(
            "beam_search: beam_width and n_best must be >= 1".into(),
        ));
    }

    let prefix_len = prefix.len();
    let mut active: Vec<Hyp> = vec![Hyp {
        tokens: prefix.to_vec(),
        score: 0.0,
    }];
    let mut completed: Vec<Hyp> = Vec::new();

    for _ in 0..config.max_new_tokens {
        if active.is_empty() {
            break;
        }
        if config.early_stopping && completed.len() >= config.beam_width {
            break;
        }

        // Expand every active beam and gather all candidate continuations.
        // One batched scoring call over all active beams (M5-14-BACKLOG-T07):
        // a scorer with a batched decoder step folds the `beam_width` per-beam
        // forwards into one forward. The default `logprobs_batch` loops
        // `logprobs` in order, so for every existing scorer this is byte-for-byte
        // the prior per-beam behaviour; an override is pinned bit-identical to
        // the loop, so the masking + top-K below (and thus which hypotheses are
        // kept) are unchanged either way.
        let all_lp = {
            let batch: Vec<&[u32]> = active.iter().map(|h| h.tokens.as_slice()).collect();
            scorer.logprobs_batch(&batch)?
        };
        let vocab = scorer.vocab_size();
        if all_lp.len() != active.len() {
            return Err(VokraError::InvalidArgument(format!(
                "beam_search: logprobs_batch returned {} vectors, expected {} (one per active beam)",
                all_lp.len(),
                active.len()
            )));
        }
        let mut candidates: Vec<Hyp> = Vec::new();
        for (hyp, mut lp) in active.iter().zip(all_lp) {
            if lp.len() != vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "beam_search: scorer returned {} logprobs, expected vocab_size {vocab}",
                    lp.len(),
                )));
            }
            // No-repeat n-gram blocking: mask any candidate that would
            // complete a repeated `n`-gram in the beam's history (prefix +
            // generated tokens). Blocked entries become `-inf` so they still
            // may surface at the tail of `top_k` if fewer than `beam_width`
            // finite candidates exist — the loop below explicitly skips them
            // (mirrors the Voxtral pattern).
            if config.no_repeat_ngram_size > 0 {
                let mask = ngram_block_mask(&hyp.tokens, config.no_repeat_ngram_size, vocab);
                for (i, &blocked) in mask.iter().enumerate() {
                    if blocked {
                        lp[i] = f32::NEG_INFINITY;
                    }
                }
            }
            for (tok, delta) in top_k(&lp, config.beam_width) {
                // Skip masked entries `top_k` surfaced at the tail when the
                // beam width exceeds the number of unmasked candidates.
                if !delta.is_finite() {
                    continue;
                }
                let mut tokens = hyp.tokens.clone();
                tokens.push(tok);
                // Fold LM shallow fusion into the per-candidate score BEFORE
                // the beam-level `truncate(beam_width)` below, so an LM-
                // boosted candidate may survive a step it would otherwise be
                // pruned in. `weight == 0.0` (or `lm_fusion == None`) is a
                // short-circuit — the LM is never called and the arithmetic
                // is bit-identical to `hyp.score + delta` (guards a scorer
                // whose `score` returns `-inf` from tripping
                // `0.0 * -inf = NaN`; mirrors the
                // `vokra_ops::hybrid_ctc_attention` line 231 pattern).
                let mut score = hyp.score + delta;
                if let Some(lf) = &config.lm_fusion
                    && lf.weight != 0.0
                {
                    score += lf.lm.score(&hyp.tokens, tok) * lf.weight;
                }
                candidates.push(Hyp { tokens, score });
            }
        }

        // Keep the best `beam_width` candidates overall.
        sort_by_score_desc(&mut candidates);
        candidates.truncate(config.beam_width);

        // Split finished vs. still-active.
        active.clear();
        for c in candidates {
            if c.tokens.last() == Some(&eot) {
                completed.push(c);
            } else {
                active.push(c);
            }
        }
    }

    // If nothing finished, fall back to the best unfinished beams.
    let mut finals = if completed.is_empty() {
        active
    } else {
        completed
    };

    // Normalize, sort, de-duplicate, take n_best.
    let alpha = config.length_normalization;
    let mut out: Vec<BeamHypothesis> = finals
        .drain(..)
        .map(|h| {
            let gen_len = (h.tokens.len().saturating_sub(prefix_len)).max(1) as f32;
            let denom = gen_len.powf(alpha);
            BeamHypothesis {
                normalized_score: h.score / denom,
                score: h.score,
                tokens: h.tokens,
                word_timestamps: None,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.normalized_score
            .partial_cmp(&a.normalized_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dedup_by_tokens(&mut out);
    out.truncate(config.n_best);

    // M4-20 (a): fill word-level timestamps on the best hypothesis. The scorer
    // must supply an alignment (Whisper cross-attention DTW, ADR M4-20 §D-3);
    // a scorer that cannot is an explicit error under `word_timestamps`
    // (FR-EX-08) — never a silent no-op. Only the best (returned-first)
    // hypothesis is aligned; the alignment pass is decode-independent
    // (openai-whisper `find_alignment` posture).
    if config.word_timestamps
        && let Some(best) = out.first_mut()
    {
        match scorer.align_words(&best.tokens)? {
            Some(timings) => best.word_timestamps = Some(timings),
            None => {
                return Err(VokraError::UnsupportedOp(
                    "beam_search: word_timestamps requested but the scorer supplies no \
                     alignment (cross-attention) — FR-EX-08"
                        .into(),
                ));
            }
        }
    }
    Ok(out)
}

/// A working hypothesis (cumulative score, no normalization yet).
struct Hyp {
    tokens: Vec<u32>,
    score: f32,
}

fn sort_by_score_desc(hyps: &mut [Hyp]) {
    hyps.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Returns a boolean mask of length `vocab_size` where `true` means the
/// candidate token would create a repeated `n`-gram in
/// `tokens_so_far + [candidate]` (Wave 12, M3-15 follow-up).
///
/// When `n <= 1` or `tokens_so_far` has fewer than `n - 1` tokens, no
/// `n`-gram can complete — the whole mask is `false`. Otherwise the sliding
/// window over `tokens_so_far` finds every existing `(n - 1)`-gram that
/// matches the last `n - 1` tokens; the token(s) that immediately follow
/// each match are the ones we forbid.
///
/// Ported verbatim from
/// [`vokra_models::voxtral::beam_search::ngram_block_mask`] — the two
/// decoders share the same HuggingFace-compatible semantics.
fn ngram_block_mask(tokens_so_far: &[u32], n: usize, vocab_size: usize) -> Vec<bool> {
    let mut mask = vec![false; vocab_size];
    if n <= 1 {
        return mask;
    }
    if tokens_so_far.len() + 1 < n {
        return mask;
    }
    // The (n - 1)-token suffix that a new candidate would extend into an
    // n-gram: `tokens_so_far[-(n-1)..]`.
    let suffix_len = n - 1;
    let suffix_start = tokens_so_far.len() - suffix_len;
    let suffix = &tokens_so_far[suffix_start..];

    // For every position `i` where an n-gram could start in `tokens_so_far`
    // (`i + n <= tokens_so_far.len()`), check whether
    // `tokens_so_far[i..i+suffix_len] == suffix`. If so, the next token —
    // `tokens_so_far[i + suffix_len]` — is a forbidden candidate.
    if tokens_so_far.len() < n {
        return mask;
    }
    for i in 0..=tokens_so_far.len() - n {
        if &tokens_so_far[i..i + suffix_len] == suffix {
            let forbid = tokens_so_far[i + suffix_len] as usize;
            if forbid < vocab_size {
                mask[forbid] = true;
            }
        }
    }
    mask
}

/// Returns the `k` highest `(index, value)` pairs of `values`, value-descending.
///
/// Single pass keeping a small sorted top list (`k` is the beam width, tiny
/// relative to the vocabulary).
fn top_k(values: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    for (i, &v) in values.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, v));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if v > top[k - 1].1 {
            top[k - 1] = (i as u32, v);
            // Bubble the new element up into place.
            let mut j = k - 1;
            while j > 0 && top[j].1 > top[j - 1].1 {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    if top.len() < k {
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }
    top
}

/// Removes hypotheses whose token sequence already appeared (keeps the first,
/// i.e. highest-ranked after the sort).
fn dedup_by_tokens(hyps: &mut Vec<BeamHypothesis>) {
    let mut seen: Vec<Vec<u32>> = Vec::new();
    hyps.retain(|h| {
        if seen.iter().any(|t| t == &h.tokens) {
            false
        } else {
            seen.push(h.tokens.clone());
            true
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic mock scorer over a tiny vocabulary driven by a fixed
    /// transition table: `logprob[tokens_last][next]`. Unknown contexts fall
    /// back to a uniform distribution, and `eot` is an absorbing high-prob
    /// choice from a designated "sink" token so sequences terminate.
    struct MockScorer {
        vocab: usize,
        // table[last_token] = log-probs over next token.
        table: Vec<Vec<f32>>,
    }

    impl MockScorer {
        fn logprob_row(&self, last: u32) -> &[f32] {
            &self.table[last as usize]
        }
    }

    impl BeamScorer for MockScorer {
        fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            let last = *tokens.last().unwrap();
            Ok(self.logprob_row(last).to_vec())
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
    }

    /// vocab = {0=start, 1='a', 2='b', 3=eot}. From start, 'a' is likelier than
    /// 'b'; from any letter, eot is near-certain.
    fn scorer() -> MockScorer {
        let ln = |p: f32| p.ln();
        MockScorer {
            vocab: 4,
            table: vec![
                // from 0 (start): a=0.6, b=0.4 (eot/start negligible)
                vec![ln(0.001), ln(0.6), ln(0.399), ln(0.001)],
                // from 1 (a): eot=0.9, b=0.1
                vec![ln(0.001), ln(0.001), ln(0.098), ln(0.9)],
                // from 2 (b): eot=0.95
                vec![ln(0.001), ln(0.02), ln(0.001), ln(0.95)],
                // from 3 (eot): absorbing
                vec![ln(0.001), ln(0.001), ln(0.001), ln(0.997)],
            ],
        }
    }

    const EOT: u32 = 3;

    #[test]
    fn width_one_is_greedy() {
        let mut s = scorer();
        let cfg = BeamSearchConfig::greedy(8);
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        assert_eq!(hyps.len(), 1);
        // Greedy: start→a (0.6 > 0.399), a→eot (0.9). Sequence [0,1,3].
        assert_eq!(hyps[0].tokens, vec![0, 1, 3]);
    }

    #[test]
    fn wider_beam_does_not_reduce_best_score() {
        let mut s1 = scorer();
        let mut s2 = scorer();
        let g = beam_search(&mut s1, &[0], EOT, &BeamSearchConfig::greedy(8)).unwrap();
        let b = beam_search(&mut s2, &[0], EOT, &BeamSearchConfig::new(3, 8)).unwrap();
        // The wider search's best (unnormalized) score is at least the greedy
        // one's — beam search cannot do worse at finding a high-prob path.
        assert!(
            b[0].score >= g[0].score - 1e-5,
            "{} vs {}",
            b[0].score,
            g[0].score
        );
    }

    #[test]
    fn n_best_is_sorted_unique_and_capped() {
        let mut s = scorer();
        let mut cfg = BeamSearchConfig::new(3, 8);
        cfg.n_best = 3;
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        assert!(hyps.len() <= 3);
        // Sorted by normalized score, descending.
        for w in hyps.windows(2) {
            assert!(w[0].normalized_score >= w[1].normalized_score);
        }
        // Unique token sequences.
        for i in 0..hyps.len() {
            for j in (i + 1)..hyps.len() {
                assert_ne!(hyps[i].tokens, hyps[j].tokens);
            }
        }
    }

    #[test]
    fn early_stopping_keeps_the_same_best() {
        let mut s1 = scorer();
        let mut s2 = scorer();
        let mut on = BeamSearchConfig::new(3, 8);
        on.early_stopping = true;
        let mut off = on.clone();
        off.early_stopping = false;
        let a = beam_search(&mut s1, &[0], EOT, &on).unwrap();
        let b = beam_search(&mut s2, &[0], EOT, &off).unwrap();
        assert_eq!(a[0].tokens, b[0].tokens);
    }

    #[test]
    fn length_normalization_changes_ranking_monotonically() {
        // With alpha=0 the raw score wins; with a large alpha, longer sequences
        // are penalized less per token. Assert normalized == score when alpha=0.
        let mut s = scorer();
        let mut cfg = BeamSearchConfig::new(3, 8);
        cfg.length_normalization = 0.0;
        cfg.n_best = 3;
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        for h in &hyps {
            assert!((h.normalized_score - h.score).abs() < 1e-6);
        }
    }

    #[test]
    fn word_timestamps_without_alignment_is_explicit_unsupported() {
        // M4-20 (a) + FR-EX-08: a scorer that does not override `align_words`
        // (the default `Ok(None)`) requested with `word_timestamps = true`
        // must yield an EXPLICIT error, never a silent no-op / empty field.
        let mut s = scorer();
        let mut cfg = BeamSearchConfig::greedy(4);
        cfg.word_timestamps = true;
        match beam_search(&mut s, &[0], EOT, &cfg) {
            Err(VokraError::UnsupportedOp(msg)) => {
                assert!(msg.contains("word_timestamps"), "message: {msg}");
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite FR-EX-08: {msg}"
                );
            }
            other => panic!("expected UnsupportedOp for missing alignment, got {other:?}"),
        }
    }

    #[test]
    fn word_timestamps_off_never_calls_align() {
        // Regression: with the flag off, `align_words` must not be consulted —
        // a scorer whose `align_words` panics still succeeds when the flag is
        // clear (guards against always-calling the aligner).
        struct PanicAligner(MockScorer);
        impl BeamScorer for PanicAligner {
            fn logprobs(&mut self, t: &[u32]) -> Result<Vec<f32>> {
                self.0.logprobs(t)
            }
            fn vocab_size(&self) -> usize {
                self.0.vocab_size()
            }
            fn align_words(&mut self, _t: &[u32]) -> Result<Option<Vec<WordTiming>>> {
                panic!("align_words must not be called when word_timestamps is off");
            }
        }
        let mut s = PanicAligner(scorer());
        let cfg = BeamSearchConfig::greedy(8); // word_timestamps defaults false
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        assert!(hyps[0].word_timestamps.is_none());
    }

    #[test]
    fn word_timestamps_attaches_to_best_when_scorer_aligns() {
        // A scorer that DOES supply an alignment fills the best hypothesis's
        // `word_timestamps` (and only the best one).
        struct AligningScorer(MockScorer);
        impl BeamScorer for AligningScorer {
            fn logprobs(&mut self, t: &[u32]) -> Result<Vec<f32>> {
                self.0.logprobs(t)
            }
            fn vocab_size(&self) -> usize {
                self.0.vocab_size()
            }
            fn align_words(&mut self, tokens: &[u32]) -> Result<Option<Vec<WordTiming>>> {
                // A trivial one-word alignment spanning the whole sequence.
                Ok(Some(vec![WordTiming {
                    token_start: 0,
                    token_end: tokens.len(),
                    start: 0.0,
                    end: 0.5,
                }]))
            }
        }
        let mut s = AligningScorer(scorer());
        let mut cfg = BeamSearchConfig::new(3, 8);
        cfg.n_best = 3;
        cfg.word_timestamps = true;
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        let best = &hyps[0];
        let wt = best.word_timestamps.as_ref().expect("best is aligned");
        assert_eq!(wt.len(), 1);
        assert_eq!(wt[0].token_end, best.tokens.len());
        // Only the best hypothesis is aligned.
        for h in &hyps[1..] {
            assert!(h.word_timestamps.is_none(), "only the best is aligned");
        }
    }

    #[test]
    fn empty_prefix_and_zero_width_are_rejected() {
        let mut s = scorer();
        assert!(matches!(
            beam_search(&mut s, &[], EOT, &BeamSearchConfig::greedy(4)),
            Err(VokraError::InvalidArgument(_))
        ));
        let mut cfg = BeamSearchConfig::greedy(4);
        cfg.beam_width = 0;
        assert!(matches!(
            beam_search(&mut s, &[0], EOT, &cfg),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn beam_wider_than_vocab_does_not_panic() {
        let mut s = scorer();
        let cfg = BeamSearchConfig::new(10, 6); // width > vocab (4)
        let hyps = beam_search(&mut s, &[0], EOT, &cfg).unwrap();
        assert!(!hyps.is_empty());
    }

    #[test]
    fn immediate_eot_terminates() {
        // A scorer that always jumps to eot.
        struct AllEot;
        impl BeamScorer for AllEot {
            fn logprobs(&mut self, _t: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![f32::NEG_INFINITY, 0.0]) // token 1 = eot certain
            }
            fn vocab_size(&self) -> usize {
                2
            }
        }
        let mut s = AllEot;
        let hyps = beam_search(&mut s, &[0], 1, &BeamSearchConfig::greedy(5)).unwrap();
        assert_eq!(hyps[0].tokens, vec![0, 1]);
    }

    #[test]
    fn returns_best_unfinished_when_budget_exhausted_before_eot() {
        // A model that never emits eot: token 1 is certain, eot (token 0) is
        // -inf. When max_new_tokens runs out before any hypothesis completes,
        // the search must fall back to the best UNFINISHED beam rather than
        // returning nothing (the `completed.is_empty() -> active` branch).
        struct NeverEot;
        impl BeamScorer for NeverEot {
            fn logprobs(&mut self, _t: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![f32::NEG_INFINITY, 0.0]) // token 1 certain; token 0 (=eot) never
            }
            fn vocab_size(&self) -> usize {
                2
            }
        }
        let mut s = NeverEot;
        let hyps = beam_search(
            &mut s,
            &[7],
            /* eot = */ 0,
            &BeamSearchConfig::greedy(3),
        )
        .unwrap();
        assert!(!hyps.is_empty());
        // Prefix [7] plus exactly max_new_tokens (3) generated tokens, all of
        // which are token 1 (never the eot token 0).
        assert_eq!(hyps[0].tokens, vec![7, 1, 1, 1]);
        assert_ne!(*hyps[0].tokens.last().unwrap(), 0);
    }

    #[test]
    fn scorer_wrong_logprobs_length_is_rejected() {
        // The defensive length check between scorer.vocab_size() and the
        // returned logprobs vector (the model<->search contract, FR-OP-40).
        struct BadLenScorer;
        impl BeamScorer for BadLenScorer {
            fn logprobs(&mut self, _t: &[u32]) -> Result<Vec<f32>> {
                Ok(vec![0.0, 0.0, 0.0]) // length 3 ...
            }
            fn vocab_size(&self) -> usize {
                4 // ... but claims a vocab of 4.
            }
        }
        let mut s = BadLenScorer;
        match beam_search(&mut s, &[0], EOT, &BeamSearchConfig::greedy(4)) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("logprobs"), "unexpected message: {msg}");
            }
            other => panic!("expected InvalidArgument for the length mismatch, got {other:?}"),
        }
    }

    #[test]
    fn scorer_error_is_propagated() {
        // An Err from the scorer surfaces verbatim through beam_search (the
        // `scorer.logprobs(...)?` propagation).
        struct ErrScorer;
        impl BeamScorer for ErrScorer {
            fn logprobs(&mut self, _t: &[u32]) -> Result<Vec<f32>> {
                Err(VokraError::InvalidArgument("boom".into()))
            }
            fn vocab_size(&self) -> usize {
                2
            }
        }
        let mut s = ErrScorer;
        match beam_search(&mut s, &[0], EOT, &BeamSearchConfig::greedy(4)) {
            Err(VokraError::InvalidArgument(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected the scorer's own InvalidArgument, got {other:?}"),
        }
    }

    // ---------- no_repeat_ngram_size (Wave 12, M3-15 follow-up) --------
    //
    // Semantics mirror `vokra_models::voxtral::beam_search::ngram_block_mask`
    // one-to-one so Whisper's model-independent beam and Voxtral's
    // model-specific beam share the same HuggingFace-compatible behaviour.

    #[test]
    fn ngram_block_mask_n_zero_is_no_op() {
        // n == 0 must not mask anything, regardless of the token history.
        // (n == 1 is likewise a no-op by the semantics — a unigram is always
        //  a "repeat" so blocking would make every token forbidden.)
        for n in 0..=1 {
            let mask = ngram_block_mask(&[1, 2, 3, 1, 2], n, 5);
            assert!(mask.iter().all(|&b| !b), "n={n} must not block anything");
        }
    }

    #[test]
    fn ngram_block_mask_n_2_blocks_bigram_repetition() {
        // History: [1, 2, 3, 1]. A "2" as the next token would form the
        // bigram (1, 2) which already exists at index 0 — so 2 must be
        // masked. A "3" would form (1, 3), which is not in the history —
        // it must NOT be masked.
        let mask = ngram_block_mask(&[1, 2, 3, 1], 2, 5);
        assert!(
            mask[2],
            "bigram (1, 2) already exists in history; extending 1 with 2 must be blocked"
        );
        assert!(
            !mask[3],
            "bigram (1, 3) not in history; must not be blocked"
        );
    }

    #[test]
    fn ngram_block_mask_n_3_blocks_trigram_repetition() {
        // n = 3: history [1, 2, 3, 1, 2]. Next candidate "3" would complete
        // the trigram (1, 2, 3) which already exists at index 0 — so 3 must
        // be masked. "4" → (1, 2, 4), not in history → must not be masked.
        let mask = ngram_block_mask(&[1, 2, 3, 1, 2], 3, 5);
        assert!(
            mask[3],
            "trigram (1, 2, 3) already exists; extending (1, 2) with 3 must be blocked"
        );
        assert!(
            !mask[4],
            "trigram (1, 2, 4) not in history; must not be blocked"
        );
    }

    #[test]
    fn ngram_block_mask_short_history_no_op() {
        // History shorter than n - 1 → no n-gram of length n can complete.
        // Nothing must be masked. Two shapes:
        //   * history of length 1, n = 3 (len < n - 1 = 2)
        //   * empty history, n = 2 (len + 1 < n)
        for (history, n) in [(&[1u32][..], 3usize), (&[][..], 2usize)] {
            let mask = ngram_block_mask(history, n, 5);
            assert!(
                mask.iter().all(|&b| !b),
                "history {history:?} with n={n} must not block anything"
            );
        }
    }

    #[test]
    fn beam_search_no_repeat_ngram_prevents_repetition() {
        // A scorer that STRONGLY prefers to repeat a bigram: after <sos>
        // the greedy chain is [0, 1, 2, 1, 2, 1, 2, …]. eot is priced at
        // ~zero on every non-eot state so the search cannot terminate on
        // eot; it exhausts `max_new_tokens` and falls back to the best
        // unfinished beam — the natural repeating chain.
        //
        // vocab = {0=<sos>, 1=a, 2=b, 3=<eot>}. Table: from a → prefer b;
        // from b → prefer a; eot fires only when the previous token was eot
        // (absorbing), so the search never terminates naturally over the
        // 6-token budget.
        struct BigramRepeatScorer;
        impl BeamScorer for BigramRepeatScorer {
            fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
                let ln = |p: f32| p.ln();
                let last = *tokens.last().unwrap();
                Ok(match last {
                    // From <sos>: prefer a heavily.
                    0 => vec![ln(0.001), ln(0.997), ln(0.001), ln(0.001)],
                    // From a: prefer b (completes the (1, 2) bigram).
                    1 => vec![ln(0.001), ln(0.001), ln(0.997), ln(0.001)],
                    // From b: prefer a (drives back into another (1, 2)).
                    2 => vec![ln(0.001), ln(0.997), ln(0.001), ln(0.001)],
                    // From eot: absorbing.
                    _ => vec![ln(0.001), ln(0.001), ln(0.001), ln(0.997)],
                })
            }
            fn vocab_size(&self) -> usize {
                4
            }
        }

        // Baseline: no mask → the beam exhausts max_new_tokens along the
        // repeating chain, so (1, 2) appears multiple times.
        //
        // Length normalization is disabled (α = 0) so shorter completed
        // hypotheses can never dominate a longer unfinished one. That is
        // safe here because eot is essentially unreachable, but keeping α
        // at the default 1.0 makes the ranking dependent on the length
        // normalization arithmetic — flip it off to isolate the mask
        // behaviour.
        let mut cfg = BeamSearchConfig::new(2, 6);
        cfg.length_normalization = 0.0;
        cfg.n_best = 1;
        let baseline = beam_search(&mut BigramRepeatScorer, &[0], 3, &cfg).unwrap();
        let baseline_tokens = &baseline[0].tokens;
        let baseline_bigram_count = baseline_tokens
            .windows(2)
            .filter(|w| w == &[1u32, 2u32])
            .count();
        assert!(
            baseline_bigram_count >= 2,
            "baseline (no mask) must exhibit the (1, 2) bigram repetition to make this test \
             meaningful: got tokens {baseline_tokens:?} with (1, 2) count = {baseline_bigram_count}"
        );

        // With n = 2 blocking: once (1, 2) exists, the second "2" after a
        // "1" is masked to -inf and the top-K picks something else. The
        // (1, 2) bigram must appear at most once anywhere in the result.
        let mut cfg = BeamSearchConfig::new(2, 6);
        cfg.length_normalization = 0.0;
        cfg.n_best = 1;
        cfg.no_repeat_ngram_size = 2;
        let blocked = beam_search(&mut BigramRepeatScorer, &[0], 3, &cfg).unwrap();
        let blocked_tokens = &blocked[0].tokens;
        let blocked_bigram_count = blocked_tokens
            .windows(2)
            .filter(|w| w == &[1u32, 2u32])
            .count();
        assert!(
            blocked_bigram_count <= 1,
            "with no_repeat_ngram_size = 2 the (1, 2) bigram must appear at most once, got \
             tokens {blocked_tokens:?} with count = {blocked_bigram_count}"
        );
    }

    // ---------- logprobs_batch (M5-14-BACKLOG-T07) ----------------------

    #[test]
    fn default_logprobs_batch_matches_per_beam_loop() {
        // The default `logprobs_batch` must be byte-for-byte the per-prefix
        // loop (order preserved), so a scorer without a batched forward is
        // unaffected by the beam search calling the batch entry.
        let mut s = scorer();
        let prefixes: &[&[u32]] = &[&[0], &[0, 1], &[0, 2], &[0, 1, 3]];
        let batched = s.logprobs_batch(prefixes).unwrap();
        assert_eq!(batched.len(), prefixes.len());
        let mut looped = Vec::new();
        for p in prefixes {
            looped.push(s.logprobs(p).unwrap());
        }
        for (b, l) in batched.iter().zip(&looped) {
            for (x, y) in b.iter().zip(l) {
                assert_eq!(x.to_bits(), y.to_bits(), "batch != loop bitwise");
            }
        }
    }

    #[test]
    fn batched_scorer_sees_all_active_beams_and_result_is_unchanged() {
        // A scorer that overrides `logprobs_batch` (a) records the batch sizes
        // it is asked for — proving beam_search folds every active beam into a
        // single call — and (b) returns exactly what the per-beam loop would,
        // so the beam result is identical to the non-overriding scorer.
        struct BatchSpyScorer {
            inner: MockScorer,
            max_batch: std::cell::Cell<usize>,
        }
        impl BeamScorer for BatchSpyScorer {
            fn logprobs(&mut self, t: &[u32]) -> Result<Vec<f32>> {
                self.inner.logprobs(t)
            }
            fn vocab_size(&self) -> usize {
                self.inner.vocab_size()
            }
            fn logprobs_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
                self.max_batch.set(self.max_batch.get().max(prefixes.len()));
                // Bit-identical to the default loop (an optimized override would
                // fold the forwards but return the same bits — pinned elsewhere).
                prefixes.iter().map(|p| self.inner.logprobs(p)).collect()
            }
        }
        let mut cfg = BeamSearchConfig::new(3, 8);
        cfg.n_best = 3;

        let mut plain = scorer();
        let want = beam_search(&mut plain, &[0], EOT, &cfg).unwrap();

        let mut spy = BatchSpyScorer {
            inner: scorer(),
            max_batch: std::cell::Cell::new(0),
        };
        let got = beam_search(&mut spy, &[0], EOT, &cfg).unwrap();

        assert_eq!(got, want, "overriding logprobs_batch changed the result");
        assert!(
            spy.max_batch.get() >= 2,
            "beam_search never batched more than one beam (max seen {})",
            spy.max_batch.get()
        );
    }

    #[test]
    fn logprobs_batch_length_mismatch_is_rejected() {
        // A broken override that returns the wrong number of vectors must be an
        // explicit error, never a silent truncation (FR-EX-08).
        struct WrongCountScorer(MockScorer);
        impl BeamScorer for WrongCountScorer {
            fn logprobs(&mut self, t: &[u32]) -> Result<Vec<f32>> {
                self.0.logprobs(t)
            }
            fn vocab_size(&self) -> usize {
                self.0.vocab_size()
            }
            fn logprobs_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
                // Drop the last vector — one too few.
                let mut out: Vec<Vec<f32>> = prefixes
                    .iter()
                    .map(|p| self.0.logprobs(p))
                    .collect::<Result<_>>()?;
                out.pop();
                Ok(out)
            }
        }
        // Width 3 so a step has multiple active beams to mis-count.
        let mut s = WrongCountScorer(scorer());
        let cfg = BeamSearchConfig::new(3, 8);
        // The first step has a single active beam (the prefix), so pop → 0
        // vectors; either way the count check must fire on some step.
        match beam_search(&mut s, &[0], EOT, &cfg) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("logprobs_batch"), "message: {msg}");
            }
            other => panic!("expected InvalidArgument for wrong batch count, got {other:?}"),
        }
    }

    #[test]
    fn no_repeat_ngram_zero_is_bit_identical_to_omitted() {
        // Regression: adding the field with a default of 0 MUST NOT change
        // any pre-Wave-12 output. Two configs — one omits `no_repeat_ngram_size`
        // (goes through the field default), one sets it to 0 explicitly —
        // must produce byte-for-byte the same n-best list on the same scorer.
        // Guards against a stray "if config.no_repeat_ngram_size >= 0" that
        // would flip the mask branch on for the zero case.
        let mut s1 = scorer();
        let mut s2 = scorer();
        let mut cfg_omitted = BeamSearchConfig::new(3, 8);
        cfg_omitted.n_best = 3;
        let mut cfg_explicit_zero = cfg_omitted.clone();
        cfg_explicit_zero.no_repeat_ngram_size = 0;
        let a = beam_search(&mut s1, &[0], EOT, &cfg_omitted).unwrap();
        let b = beam_search(&mut s2, &[0], EOT, &cfg_explicit_zero).unwrap();
        assert_eq!(a, b, "n=0 must be bit-identical to omitting the field");
    }

    // ---------- LM shallow fusion (FR-OP-41 / FR-OP-42) -------------------
    //
    // Mirrors the semantics wired through
    // `vokra_ops::hybrid_ctc_attention::HybridCtcAttentionAttrs::lm_score_fn`
    // + `lm_weight` — the LM contribution is added to a candidate's score
    // BEFORE the beam-level `truncate(beam_width)`, and `weight == 0.0`
    // short-circuits the LM call so a scorer returning `-inf` never trips
    // `0.0 * -inf = NaN` (the invariant that makes the "weight = 0.0 is
    // bit-identical to lm_fusion = None" test hold).

    /// LM scorer that returns `-inf` for every query — used to prove
    /// the short-circuit fires when `weight = 0.0` (a would-be
    /// `0.0 * -inf = NaN` corruption is prevented).
    struct NegInfLm;
    impl LmScorer for NegInfLm {
        fn score(&self, _context: &[u32], _next: u32) -> f32 {
            f32::NEG_INFINITY
        }
    }

    /// LM scorer that returns `boost` for `target` and `0.0` for everything
    /// else — the "small stateless model that likes exactly one token" used
    /// to prove positive-weight fusion changes ranking.
    struct SingleTokenBoostLm {
        target: u32,
        boost: f32,
    }
    impl LmScorer for SingleTokenBoostLm {
        fn score(&self, _context: &[u32], next: u32) -> f32 {
            if next == self.target { self.boost } else { 0.0 }
        }
    }

    #[test]
    fn lm_fusion_none_is_bit_identical_to_omitted() {
        // Regression: adding the field with a default of `None` MUST NOT
        // change any pre-LM-fusion output. Two configs — one goes through
        // the `::new` field default, one sets `lm_fusion = None` explicitly
        // — must produce byte-for-byte the same n-best list. Guards against
        // a stray `if config.lm_fusion.is_some()` mis-branch that would
        // take the LM path for a `None` too.
        let mut s1 = scorer();
        let mut s2 = scorer();
        let mut cfg_omitted = BeamSearchConfig::new(3, 8);
        cfg_omitted.n_best = 3;
        let mut cfg_explicit_none = cfg_omitted.clone();
        cfg_explicit_none.lm_fusion = None;
        let a = beam_search(&mut s1, &[0], EOT, &cfg_omitted).unwrap();
        let b = beam_search(&mut s2, &[0], EOT, &cfg_explicit_none).unwrap();
        assert_eq!(
            a, b,
            "lm_fusion = None must be bit-identical to omitting the field"
        );
    }

    #[test]
    fn lm_fusion_weight_zero_is_bit_identical_to_none() {
        // Regression: `Some { lm, weight: 0.0 }` must be bit-identical to
        // `None` — the LM is never consulted, so a scorer whose `score`
        // returns `-inf` (which `0.0 * -inf = NaN` would corrupt) does not
        // affect the result. This is what makes the `lm_weight == 0.0`
        // short-circuit load-bearing rather than cosmetic.
        //
        // Comparison uses `to_bits()` equality per field — a `+0.0` vs
        // `-0.0` drift would surface here too.
        let neg_inf = NegInfLm;
        let mut cfg_none = BeamSearchConfig::new(3, 8);
        cfg_none.n_best = 3;
        let mut cfg_weight_zero = cfg_none.clone();
        cfg_weight_zero.lm_fusion = Some(LmFusionConfig {
            lm: &neg_inf,
            weight: 0.0,
        });

        let mut s1 = scorer();
        let mut s2 = scorer();
        let a = beam_search(&mut s1, &[0], EOT, &cfg_none).unwrap();
        let b = beam_search(&mut s2, &[0], EOT, &cfg_weight_zero).unwrap();
        assert_eq!(a.len(), b.len(), "n-best length mismatch");
        for (i, (ha, hb)) in a.iter().zip(&b).enumerate() {
            assert_eq!(ha.tokens, hb.tokens, "hyp {i} token drift");
            assert_eq!(
                ha.score.to_bits(),
                hb.score.to_bits(),
                "hyp {i} score bit drift (weight=0 must short-circuit LM call)"
            );
            assert_eq!(
                ha.normalized_score.to_bits(),
                hb.normalized_score.to_bits(),
                "hyp {i} normalized_score bit drift"
            );
        }
    }

    #[test]
    fn lm_fusion_positive_weight_reranks_toward_lm() {
        // Baseline beam (width = 2, so per-beam `top_k(2)` puts BOTH token 1
        // and token 2 in the candidate pool — LM can promote either) from
        // prefix [0] with the default scorer picks the acoustic winner
        // `[0, 1, 3]` at rank 1 (top-2 keeps [0,1,3] score -0.617 and
        // [0,2,3] score -0.970; length-normalized top-1 = [0,1,3]).
        //
        // With a heavy LM boost on token 2 (weight = 1.0, boost = 10.0),
        // the fused score at step 1 for extending [0] → 2 is
        // `ln(0.399) + 10 = 9.081`, way above extending → 1 (`ln(0.6) =
        // -0.511`). [0, 2, 3] rises to rank 1.
        //
        // NOTE: beam_width = 1 (`::greedy`) does NOT exercise this — per-beam
        // `top_k(1)` picks only the acoustic winner, so LM never sees the
        // second-place token as a candidate. width >= 2 is required to
        // demonstrate reranking, matching how a real caller wires the LM.
        let baseline = beam_search(&mut scorer(), &[0], EOT, &BeamSearchConfig::new(2, 8)).unwrap();
        assert_eq!(
            baseline[0].tokens,
            vec![0u32, 1, 3],
            "acoustic-only baseline picks token 1 at rank 1"
        );

        let boost = SingleTokenBoostLm {
            target: 2,
            boost: 10.0,
        };
        let mut cfg = BeamSearchConfig::new(2, 8);
        cfg.lm_fusion = Some(LmFusionConfig {
            lm: &boost,
            weight: 1.0,
        });
        let biased = beam_search(&mut scorer(), &[0], EOT, &cfg).unwrap();
        assert!(
            biased[0].tokens.contains(&2u32),
            "LM boost on token 2 must rerank the top-1 hypothesis to contain it, got {:?}",
            biased[0].tokens
        );
    }

    #[test]
    fn lm_added_before_top_k_can_promote_survivor() {
        // Proves LM fusion applies BEFORE the beam-level `truncate(beam_width)`,
        // not after — an LM-boosted candidate survives a pruning decision it
        // would otherwise lose.
        //
        // Setup — with `beam_width = 2` and `max_new_tokens = 2` on the
        // module scorer:
        //   step 1: prefix [0] → top-K=2 gives candidates [0, 1] (score
        //           ln(0.6) = -0.511) and [0, 2] (score ln(0.399) = -0.919).
        //           No pruning (2 candidates <= beam_width). Both active.
        //   step 2: expands each → 4 candidates.
        //     baseline (no LM):
        //       [0, 1, 3] = ln(0.6)+ln(0.9)   = -0.617   (EOT — completed)
        //       [0, 2, 3] = ln(0.399)+ln(0.95) = -0.970  (EOT — completed)
        //       [0, 1, 2] = ln(0.6)+ln(0.098) = -2.834
        //       [0, 2, 1] = ln(0.399)+ln(0.02) = -4.831
        //       top-2 → [[0,1,3], [0,2,3]]. Both complete.
        //     with LM boosting token 1 by +10, weight = 1.0:
        //       Step-1 LM contribution: +10 for extending [0]→1, +0 for →2.
        //         [0, 1] carries score ln(0.6)+10 = 9.489 into step 2.
        //         [0, 2] carries score ln(0.399)+0 = -0.919 into step 2.
        //       Step-2 candidates (LM adds boost when extending →1):
        //         [0, 1, 3] = 9.489 + ln(0.9)  + 0  = 9.384   (EOT)
        //         [0, 1, 2] = 9.489 + ln(0.098)+ 0  = 7.166
        //         [0, 2, 1] = -0.919 + ln(0.02)+ 10 = 5.169
        //         [0, 2, 3] = -0.919 + ln(0.95)+ 0  = -0.970  (EOT)
        //       top-2 → [[0,1,3], [0,1,2]]. [0,2,3] PRUNED even though
        //       under the acoustic-only ranking it would survive.
        //
        // The observable proof: baseline contains [0,2,3] as a top-2
        // result; the LM-fused search does NOT — [0,2,3] was pruned by the
        // step-2 beam truncate under the LM-inflated ranking. If LM were
        // applied AFTER the truncate, [0,2,3] would still be a candidate
        // (it survives the acoustic ranking) and would still appear in
        // the final n-best.

        let mut cfg = BeamSearchConfig::new(2, 2);
        cfg.n_best = 2;
        // α = 0 so ranking uses raw score (isolate the LM effect from the
        // length-normalization arithmetic — [0,1,3] and [0,2,3] have the
        // same length so it would not flip the ranking, but keeping α at
        // the default 1.0 needlessly muddies the trace).
        cfg.length_normalization = 0.0;

        let baseline = beam_search(&mut scorer(), &[0], EOT, &cfg).unwrap();
        let baseline_tokens: Vec<Vec<u32>> = baseline.iter().map(|h| h.tokens.clone()).collect();
        assert!(
            baseline_tokens.contains(&vec![0u32, 2, 3]),
            "acoustic-only baseline MUST include [0,2,3] for this test to be meaningful \
             (proves the pruning boundary sits where LM would move it), got {baseline_tokens:?}",
        );

        let boost = SingleTokenBoostLm {
            target: 1,
            boost: 10.0,
        };
        let mut cfg_lm = cfg.clone();
        cfg_lm.lm_fusion = Some(LmFusionConfig {
            lm: &boost,
            weight: 1.0,
        });
        let with_lm = beam_search(&mut scorer(), &[0], EOT, &cfg_lm).unwrap();
        let with_lm_tokens: Vec<Vec<u32>> = with_lm.iter().map(|h| h.tokens.clone()).collect();
        assert!(
            !with_lm_tokens.contains(&vec![0u32, 2, 3]),
            "with LM boost on token 1, [0,2,3] MUST be pruned by the step-2 beam \
             truncate — the LM-inflated ranking demotes it out of the top-{}. If \
             this fires, LM is being applied AFTER truncate, not before. Got {with_lm_tokens:?}",
            cfg_lm.beam_width,
        );
    }
}
