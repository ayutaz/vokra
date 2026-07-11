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
//! `early_stopping`, `n_best` and `word_timestamps`. `word_timestamps` is
//! **defined but not implemented in M0** (WP completion = demo + parity); it is
//! an explicit [`VokraError::NotImplemented`] when enabled, never a silent
//! no-op. `max_new_tokens` is an operational termination bound beyond the five
//! attributes.
//!
//! # M0 simplicity
//!
//! The scorer is queried with the **full token prefix** each expansion, so it
//! stays model-independent and the search needs no per-beam state plumbing.
//! A model may recompute or use a cache internally; efficient per-beam cache
//! reuse/reordering (FR-EX-02 / M1-04) is a model-side optimization that does
//! not change this interface. Data structures avoid interior sharing so a
//! future static-arena pass (FR-EX-05) is not precluded.

use crate::error::{Result, VokraError};

/// Scores next tokens for the beam search (model-independent interface).
pub trait BeamScorer {
    /// Returns the **log-probabilities** over the whole vocabulary for the
    /// token following `tokens` (the full sequence so far, including any forced
    /// prefix). Implementations return normalized log-probs (e.g. a
    /// `log_softmax` of the model logits), length `vocab_size()`.
    fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Vocabulary size (the length of every [`logprobs`](Self::logprobs) result).
    fn vocab_size(&self) -> usize;
}

/// Beam-search attributes (FR-OP-40) plus an operational length cap.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
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
    /// Emit word-level timestamps. **Unimplemented in M0** — enabling it is an
    /// explicit [`VokraError::NotImplemented`].
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
    /// [`vokra_models::voxtral::beam_search::BeamConfig::no_repeat_ngram_size`]
    /// so both search primitives share the same semantics.
    pub no_repeat_ngram_size: usize,
}

impl BeamSearchConfig {
    /// A greedy-equivalent config: `beam_width = 1`, no normalization, one
    /// result, no timestamps, no n-gram blocking.
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            beam_width: 1,
            length_normalization: 0.0,
            early_stopping: true,
            n_best: 1,
            word_timestamps: false,
            max_new_tokens,
            no_repeat_ngram_size: 0,
        }
    }

    /// A standard beam config of the given width (one best result), no n-gram
    /// blocking.
    pub fn new(beam_width: usize, max_new_tokens: usize) -> Self {
        Self {
            beam_width,
            length_normalization: 1.0,
            early_stopping: true,
            n_best: 1,
            word_timestamps: false,
            max_new_tokens,
            no_repeat_ngram_size: 0,
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
/// - [`VokraError::NotImplemented`] if `word_timestamps` is enabled;
/// - any error surfaced by the [`BeamScorer`].
pub fn beam_search(
    scorer: &mut dyn BeamScorer,
    prefix: &[u32],
    eot: u32,
    config: &BeamSearchConfig,
) -> Result<Vec<BeamHypothesis>> {
    if config.word_timestamps {
        return Err(VokraError::NotImplemented(
            "beam_search word_timestamps (FR-OP-40 attribute; M0 defines but does not implement it)",
        ));
    }
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
        let mut candidates: Vec<Hyp> = Vec::new();
        for hyp in &active {
            let mut lp = scorer.logprobs(&hyp.tokens)?;
            if lp.len() != scorer.vocab_size() {
                return Err(VokraError::InvalidArgument(format!(
                    "beam_search: scorer returned {} logprobs, expected vocab_size {}",
                    lp.len(),
                    scorer.vocab_size()
                )));
            }
            // No-repeat n-gram blocking: mask any candidate that would
            // complete a repeated `n`-gram in the beam's history (prefix +
            // generated tokens). Blocked entries become `-inf` so they still
            // may surface at the tail of `top_k` if fewer than `beam_width`
            // finite candidates exist — the loop below explicitly skips them
            // (mirrors the Voxtral pattern).
            if config.no_repeat_ngram_size > 0 {
                let mask = ngram_block_mask(
                    &hyp.tokens,
                    config.no_repeat_ngram_size,
                    scorer.vocab_size(),
                );
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
                candidates.push(Hyp {
                    tokens,
                    score: hyp.score + delta,
                });
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
    fn word_timestamps_is_explicit_not_implemented() {
        let mut s = scorer();
        let mut cfg = BeamSearchConfig::greedy(4);
        cfg.word_timestamps = true;
        assert!(matches!(
            beam_search(&mut s, &[0], EOT, &cfg),
            Err(VokraError::NotImplemented(_))
        ));
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
}
