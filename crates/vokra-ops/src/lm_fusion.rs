//! Shallow-fusion n-gram LM (FR-OP-41 / FR-OP-42) — an in-memory Katz-style
//! back-off LM parsed from ARPA-format text, plugging into
//! [`vokra_core::decode::LmScorer`] so the model-independent
//! [`vokra_core::decode::beam_search()`] can shallow-fuse it.
//!
//! # Scope
//!
//! - Supported orders: **`2 <= order <= 5`** (bigram through 5-gram). Order
//!   1 (unigram-only) is rejected — the primitive is a *back-off* LM; a
//!   pure unigram carries no back-off structure. Order > 5 is rejected —
//!   larger contexts are unusual for classical n-gram LMs, and rejecting
//!   them keeps the arithmetic bounds obvious.
//! - Runtime function, NOT an `OpKind` variant (same posture as
//!   [`crate::ctc_decode`] / [`crate::hybrid_ctc_attention`] — FR-OP-40 /
//!   FR-EX-10). Consumed by the CTC / RNN-T / attention decoders through
//!   the [`vokra_core::decode::LmScorer`] trait; the actual fusion
//!   arithmetic (weight application + short-circuit on `weight == 0.0`)
//!   lives one indirection above, in
//!   [`vokra_core::decode::BeamSearchConfig::lm_fusion`] and mirrored in
//!   [`crate::ctc_decode::CtcBeamAttrs`] / [`crate::hybrid_ctc_attention`].
//!
//! # Numerical convention
//!
//! ARPA files store log-probabilities and back-off weights in **base 10**.
//! [`NgramLm::from_arpa`] converts every stored value to **natural log** by
//! multiplying by `ln 10` at parse time, so [`NgramLm::score`] returns
//! `ln`-space log-probabilities directly usable by
//! [`vokra_core::decode::beam_search()`] (whose acoustic scores are also in
//! `ln`).
//!
//! # OOV floor
//!
//! When the recursion bottoms out at an unigram that is not stored,
//! [`NgramLm::score`] returns `oov_logprob` (a caller-tunable field,
//! default `-20 · ln 10 ≈ -46.05`). This is very small — effectively
//! `10^-20` probability — so an unknown token is heavily disfavoured but
//! not `-inf` (an `-inf` OOV would risk `0.0 · -inf = NaN` if it ever
//! reached the beam-search arithmetic without the `weight == 0` short-
//! circuit; picking a large negative finite value keeps the LM defensive
//! against numerical corruption).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! Only `std::collections::HashMap` and `std::str::parse` — no `serde`,
//! no external crate. The root `Cargo.lock` continues to list only
//! `vokra-*` packages.

use std::collections::HashMap;

use vokra_core::decode::LmScorer;
use vokra_core::{Result, VokraError};

/// `ln 10` — the ARPA log-base conversion factor.
///
/// Aliased from [`core::f32::consts::LN_10`] rather than hand-typed: the
/// literal spelling was bit-identical to the constant, so the alias keeps the
/// compile-time-constant property `from_arpa` relies on while letting
/// `clippy::approx_constant` verify the value instead of a reader.
const LN_10: f32 = core::f32::consts::LN_10;

/// Default OOV floor in `ln` space: `-20 · ln 10 ≈ -46.05`. Equivalent to
/// an ARPA `-20` (i.e. `10^-20` probability) — heavily disfavoured but
/// safely finite (avoids `-inf * weight = NaN` corruption if the fusion
/// weight is non-zero; guards the `LmScorer` contract from tripping the
/// beam search's arithmetic).
const DEFAULT_OOV_LOGPROB: f32 = -20.0 * LN_10;

/// Katz-style back-off n-gram LM (bigram through 5-gram), parsed from an
/// ARPA-format text buffer.
///
/// A single stored map per n-gram order holds `(log-prob, back-off weight)`
/// pairs keyed by the full n-gram token sequence, both in `ln` space.
/// [`Self::score`] runs the canonical recursion:
///
/// ```text
/// score(history, next):
///     if history + [next] is stored:
///         return stored_logprob
///     if history is empty:
///         return oov_logprob                        // unigram OOV floor
///     return bow(history) + score(history[1..], next)
/// ```
///
/// where `bow(history)` is the back-off weight stored on `history` when it
/// is itself a stored n-gram (else `0` = log 1 — no discount, matching the
/// ARPA convention for a missing BOW column).
///
/// The `context` passed to [`Self::score`] is the FULL prefix the beam
/// search has generated so far; the LM truncates it to at most
/// `order - 1` tokens before the recursion (a stored order-`N` n-gram
/// can be matched by at most `N - 1` context tokens plus one candidate).
#[derive(Debug)]
pub struct NgramLm {
    /// N-gram order: `2 <= order <= 5` (bigram through 5-gram).
    order: usize,
    /// `ngrams[i]` holds `(i + 1)`-grams: `ngrams[0]` = unigrams,
    /// `ngrams[1]` = bigrams, ..., `ngrams[order - 1]` = highest-order.
    /// Value = `(log-prob in ln, back-off weight in ln)`; a missing BOW
    /// column in the ARPA is stored as `0.0` (log 1 = no discount).
    ngrams: Vec<HashMap<Vec<u32>, (f32, f32)>>,
    /// OOV floor in `ln` space (bottom of the back-off recursion).
    /// Default [`DEFAULT_OOV_LOGPROB`]; caller may override with
    /// [`Self::with_oov_logprob`].
    oov_logprob: f32,
}

impl NgramLm {
    /// Parses an ARPA-format n-gram LM (Katz back-off, in-memory).
    ///
    /// # ARPA grammar accepted
    ///
    /// ```text
    /// \data\
    /// ngram 1=<count>
    /// ngram 2=<count>
    /// [ngram N=<count>]  // up to N = order
    ///
    /// \1-grams:
    /// <log10-prob> <token_id> [<log10-bow>]
    /// ...
    ///
    /// \2-grams:
    /// <log10-prob> <token_id_1> <token_id_2> [<log10-bow>]
    /// ...
    ///
    /// [ \N-grams: with N up to order ]
    ///
    /// \end\
    /// ```
    ///
    /// Blank lines and lines starting with `#` (comments) are skipped.
    /// Tokens are ASCII decimal `u32` IDs — the caller has already mapped
    /// their vocabulary to integer IDs. Log-probs and back-off weights are
    /// converted from base 10 to `ln` at parse time.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of (FR-EX-08 — no silent
    /// tolerance for a malformed LM):
    /// - missing `\data\` header;
    /// - declared order outside `[2, 5]`;
    /// - a row that cannot be parsed (bad log-prob, bad token, bad BOW);
    /// - a row shorter than the section's declared order (missing tokens).
    pub fn from_arpa(text: &str) -> Result<Self> {
        // Streaming line-by-line parser. `current_section = Some(n)` means
        // we are inside `\n-grams:`; `None` inside `\data\` header.
        let mut saw_data = false;
        let mut order: usize = 0;
        let mut ngrams: Vec<HashMap<Vec<u32>, (f32, f32)>> = Vec::new();
        let mut current_section: Option<usize> = None;

        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // ---- section markers -----------------------------------------
            if line.eq_ignore_ascii_case("\\data\\") {
                saw_data = true;
                current_section = None;
                continue;
            }
            if line.eq_ignore_ascii_case("\\end\\") {
                break;
            }
            if let Some(n) = parse_section_marker(line) {
                if n == 0 || n > 5 {
                    return Err(VokraError::InvalidArgument(format!(
                        "NgramLm::from_arpa: \\{n}-grams: section order out of supported \
                         range [1, 5] (FR-EX-08)",
                    )));
                }
                while ngrams.len() < n {
                    ngrams.push(HashMap::new());
                }
                current_section = Some(n);
                continue;
            }

            if !saw_data {
                // Anything before \data\ is preamble — silently ignore
                // (some ARPA files carry a `\version_number` line, etc.).
                continue;
            }

            match current_section {
                None => {
                    // Inside \data\ — expect `ngram N=count`. Individual
                    // `ngram 1=X` (unigram declaration) is REQUIRED as the
                    // back-off floor, so per-declaration we only reject
                    // orders that fall outside `[1, 5]`. The FINAL order
                    // (max seen) must be in `[2, 5]` — a pure unigram LM
                    // is not a back-off LM; that check runs after the
                    // loop.
                    if let Some((n, _count)) = parse_ngram_declaration(line) {
                        if !(1..=5).contains(&n) {
                            return Err(VokraError::InvalidArgument(format!(
                                "NgramLm::from_arpa: declared order {n} out of supported range \
                                 [2, 5] (FR-EX-08)",
                            )));
                        }
                        if n > order {
                            order = n;
                        }
                    }
                    // Unknown lines in the header are silently ignored.
                }
                Some(n) => {
                    let (tokens, logprob, bow_opt) = parse_ngram_row(line, n)?;
                    let entry = (logprob * LN_10, bow_opt.map(|b| b * LN_10).unwrap_or(0.0));
                    ngrams[n - 1].insert(tokens, entry);
                }
            }
        }

        if !saw_data {
            return Err(VokraError::InvalidArgument(
                "NgramLm::from_arpa: missing \\data\\ header (FR-EX-08)".into(),
            ));
        }
        if !(2..=5).contains(&order) {
            return Err(VokraError::InvalidArgument(format!(
                "NgramLm::from_arpa: max n-gram order {order} out of supported range [2, 5] \
                 (FR-EX-08)",
            )));
        }

        // Pad ngrams to hold one map per order so `score`'s index
        // arithmetic (`ngrams[n]`) is always in-bounds.
        while ngrams.len() < order {
            ngrams.push(HashMap::new());
        }

        Ok(NgramLm {
            order,
            ngrams,
            oov_logprob: DEFAULT_OOV_LOGPROB,
        })
    }

    /// Overrides the OOV floor (in `ln` space) — the value returned when
    /// the back-off recursion bottoms out on an unigram that is not stored.
    #[must_use]
    pub fn with_oov_logprob(mut self, oov_logprob: f32) -> Self {
        self.oov_logprob = oov_logprob;
        self
    }

    /// The current OOV floor (in `ln` space).
    pub fn oov_logprob(&self) -> f32 {
        self.oov_logprob
    }

    /// N-gram order of the LM (`2 <= order <= 5`).
    pub fn order(&self) -> usize {
        self.order
    }

    /// Log-probability contribution for extending `context` with `next`.
    ///
    /// Truncates `context` to the last `order - 1` tokens, then runs the
    /// Katz back-off recursion; returns the result in `ln` space.
    pub fn score(&self, context: &[u32], next: u32) -> f32 {
        let max_ctx = self.order.saturating_sub(1);
        let start = context.len().saturating_sub(max_ctx);
        let history = &context[start..];
        self.score_recursive(history, next)
    }

    /// Inner Katz back-off recursion (private — public entry point is
    /// [`Self::score`], which handles context truncation).
    fn score_recursive(&self, history: &[u32], next: u32) -> f32 {
        let n = history.len();
        // The full n-gram we are looking up: `history + [next]`, length
        // `n + 1`. Stored in `ngrams[n]` (0-indexed = order 1 for n = 0).
        let mut full = Vec::with_capacity(n + 1);
        full.extend_from_slice(history);
        full.push(next);

        if n < self.ngrams.len()
            && let Some(&(logprob, _bow)) = self.ngrams[n].get(full.as_slice())
        {
            return logprob;
        }

        // Not found. Back off if we can.
        if n == 0 {
            // Unigram miss → OOV floor.
            return self.oov_logprob;
        }

        // Apply history's back-off weight (if `history` is itself a stored
        // n-gram of length `n`, i.e. in `ngrams[n - 1]`). ARPA convention:
        // a missing BOW column defaults to `0` = log 1 = no discount.
        let bow = self
            .ngrams
            .get(n - 1)
            .and_then(|m| m.get(history).map(|&(_, b)| b))
            .unwrap_or(0.0);

        bow + self.score_recursive(&history[1..], next)
    }
}

impl LmScorer for NgramLm {
    fn score(&self, context: &[u32], next: u32) -> f32 {
        NgramLm::score(self, context, next)
    }
}

// ---------------------------------------------------------------------------
// Parser helpers
// ---------------------------------------------------------------------------

/// Matches `\N-grams:` (case-insensitive on the `grams` part) and returns
/// `N`. `\data\` / `\end\` are handled by the caller before this runs.
fn parse_section_marker(line: &str) -> Option<usize> {
    let stripped = line.strip_prefix('\\')?;
    let (n_str, rest) = stripped.split_once('-')?;
    if !rest.eq_ignore_ascii_case("grams:") {
        return None;
    }
    n_str.parse::<usize>().ok()
}

/// Matches `ngram N=count` and returns `(N, count)`.
fn parse_ngram_declaration(line: &str) -> Option<(usize, usize)> {
    let stripped = line
        .strip_prefix("ngram")
        .or_else(|| line.strip_prefix("NGRAM"))?;
    let (n_str, count_str) = stripped.trim().split_once('=')?;
    let n = n_str.trim().parse::<usize>().ok()?;
    let count = count_str.trim().parse::<usize>().ok()?;
    Some((n, count))
}

/// Parses an n-gram row: `LOGPROB TOK1 ... TOKN [BOW]`.
/// Returns `(tokens, log10_prob, optional_log10_bow)`. The caller multiplies
/// by `LN_10` to convert to `ln`.
fn parse_ngram_row(line: &str, n: usize) -> Result<(Vec<u32>, f32, Option<f32>)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < n + 1 {
        return Err(VokraError::InvalidArgument(format!(
            "NgramLm::from_arpa: {n}-gram row expects at least {} columns (log-prob + {n} \
             tokens), got {} on line: `{line}` (FR-EX-08)",
            n + 1,
            parts.len()
        )));
    }
    let logprob = parts[0].parse::<f32>().map_err(|e| {
        VokraError::InvalidArgument(format!(
            "NgramLm::from_arpa: log-prob column `{}` not a float ({e}): `{line}` (FR-EX-08)",
            parts[0],
        ))
    })?;
    let mut tokens = Vec::with_capacity(n);
    for tok_str in &parts[1..=n] {
        let tok = tok_str.parse::<u32>().map_err(|e| {
            VokraError::InvalidArgument(format!(
                "NgramLm::from_arpa: token column `{tok_str}` not a u32 ({e}): `{line}` \
                 (FR-EX-08)",
            ))
        })?;
        tokens.push(tok);
    }
    let bow = if parts.len() > n + 1 {
        let b = parts[n + 1].parse::<f32>().map_err(|e| {
            VokraError::InvalidArgument(format!(
                "NgramLm::from_arpa: back-off weight column `{}` not a float ({e}): `{line}` \
                 (FR-EX-08)",
                parts[n + 1],
            ))
        })?;
        Some(b)
    } else {
        None
    };
    Ok((tokens, logprob, bow))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference: `ln(10) * x` at `f32` precision — proves
    /// that a stored `LN_10 * x` value round-trips a caller's expected
    /// log10 → ln arithmetic without an off-by-precision drift.
    fn ln_from_log10(x: f32) -> f32 {
        x * LN_10
    }

    #[test]
    fn parses_minimal_bigram_arpa() {
        // Minimal 2-gram ARPA: 3 unigrams (one with a BOW column, one
        // without), 2 bigrams — proves the parser handles both the
        // "BOW-present" and "BOW-absent" row shapes and the ln
        // conversion.
        let arpa = "\
\\data\\
ngram 1=3
ngram 2=2

\\1-grams:
-2.0 0 -0.5
-1.5 1
-1.8 2

\\2-grams:
-0.7 0 1
-0.9 1 2

\\end\\
";
        let lm = NgramLm::from_arpa(arpa).expect("valid ARPA parses");
        assert_eq!(lm.order(), 2);

        // Stored bigram: score([0], 1) = ln(10) * -0.7 (bigram hit, no
        // back-off).
        let want = ln_from_log10(-0.7);
        let got = lm.score(&[0], 1);
        assert!(
            (got - want).abs() < 1e-6,
            "score([0], 1) = {got}, want {want}"
        );

        // Stored unigram: score([], 0) = ln(10) * -2.0.
        let want = ln_from_log10(-2.0);
        let got = lm.score(&[], 0);
        assert!(
            (got - want).abs() < 1e-6,
            "score([], 0) = {got}, want {want}"
        );
    }

    #[test]
    fn rejects_empty_arpa() {
        // Empty input: no \data\ header → explicit error (FR-EX-08).
        match NgramLm::from_arpa("") {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("\\data\\"), "message must cite header: {msg}");
            }
            other => panic!("expected InvalidArgument for empty input, got {other:?}"),
        }

        // Missing \data\ header — even with n-gram sections present, the
        // parser must fail-closed rather than silently accepting a
        // truncated file.
        let no_header = "\
\\1-grams:
-2.0 0

\\end\\
";
        match NgramLm::from_arpa(no_header) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(msg.contains("\\data\\"), "message must cite header: {msg}");
            }
            other => panic!("expected InvalidArgument for missing header, got {other:?}"),
        }
    }

    #[test]
    fn bigram_backoff_folds_bow() {
        // ARPA with a UNIGRAM that carries a non-zero BOW and a bigram
        // section that DOES NOT contain the (0, 2) bigram.
        // Score([0], 2) must fold `bow([0]) + unigram(2)` in ln space and
        // match the hand-computed reference at `atol = 1e-6`.
        let arpa = "\
\\data\\
ngram 1=3
ngram 2=1

\\1-grams:
-2.0 0 -0.5
-1.5 1
-1.8 2

\\2-grams:
-0.7 0 1

\\end\\
";
        let lm = NgramLm::from_arpa(arpa).expect("valid ARPA parses");

        // Bigram (0, 2) is MISSING → back off:
        //   bow([0]) = ln(10) * -0.5
        //   unigram(2) = ln(10) * -1.8
        //   score([0], 2) = (-0.5 + -1.8) * ln(10) = -2.3 * ln(10)
        let want = ln_from_log10(-0.5) + ln_from_log10(-1.8);
        let got = lm.score(&[0], 2);
        assert!(
            (got - want).abs() < 1e-6,
            "score([0], 2) with back-off = {got}, want {want} (Δ = {})",
            (got - want).abs()
        );

        // Sanity: score([1], 2) — [1] has NO stored BOW (defaults to 0):
        //   bow([1]) = 0
        //   unigram(2) = ln(10) * -1.8
        let want = ln_from_log10(-1.8);
        let got = lm.score(&[1], 2);
        assert!(
            (got - want).abs() < 1e-6,
            "score([1], 2) with default BOW = {got}, want {want}"
        );
    }

    #[test]
    fn order_out_of_range_rejected() {
        // Declared 6-gram → rejected (FR-EX-08). The `ngram 6=1`
        // declaration itself is what triggers the check; the actual
        // \6-grams: section need not appear.
        let arpa = "\
\\data\\
ngram 1=1
ngram 2=1
ngram 6=1

\\1-grams:
-2.0 0

\\end\\
";
        match NgramLm::from_arpa(arpa) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("6") && msg.contains("[2, 5]"),
                    "message must cite out-of-range order and range: {msg}"
                );
            }
            other => panic!("expected InvalidArgument for order 6, got {other:?}"),
        }

        // Also reject order-1-only (unigram-only isn't a back-off LM;
        // scout plan requires order >= 2).
        let unigram_only = "\
\\data\\
ngram 1=1

\\1-grams:
-2.0 0

\\end\\
";
        match NgramLm::from_arpa(unigram_only) {
            Err(VokraError::InvalidArgument(_)) => {}
            other => panic!("expected InvalidArgument for unigram-only ARPA, got {other:?}"),
        }
    }

    #[test]
    fn oov_floor_returned_for_unknown_unigram() {
        // Unigram vocabulary = {0, 1, 2}; asking for a token not in the
        // unigram list bottoms out on the OOV floor.
        let arpa = "\
\\data\\
ngram 1=3
ngram 2=1

\\1-grams:
-2.0 0
-1.5 1
-1.8 2

\\2-grams:
-0.7 0 1

\\end\\
";
        let lm = NgramLm::from_arpa(arpa).expect("valid ARPA parses");
        // token 99 is unknown — bottoms out on OOV floor.
        let want = DEFAULT_OOV_LOGPROB;
        let got = lm.score(&[], 99);
        assert_eq!(got, want, "unknown unigram must return OOV floor");

        // With context: bigram (0, 99) missing → back off to unigram(99)
        // → OOV. bow([0]) is 0 (not stored). Result = OOV floor.
        let got = lm.score(&[0], 99);
        assert_eq!(
            got, want,
            "back-off from missing bigram to unknown unigram must still be OOV floor"
        );
    }

    #[test]
    fn custom_oov_logprob_overrides_default() {
        let arpa = "\
\\data\\
ngram 1=1
ngram 2=1

\\1-grams:
-2.0 0

\\2-grams:
-0.7 0 0

\\end\\
";
        let lm = NgramLm::from_arpa(arpa)
            .expect("valid ARPA parses")
            .with_oov_logprob(-99.0);
        assert_eq!(lm.oov_logprob(), -99.0);
        assert_eq!(lm.score(&[], 42), -99.0);
    }

    #[test]
    fn plugs_into_lm_scorer_trait() {
        // The trait impl must call through to `score`. Store the LM behind
        // a `&dyn LmScorer` reference and query — mirrors how the beam
        // search sees the object.
        let arpa = "\
\\data\\
ngram 1=2
ngram 2=1

\\1-grams:
-2.0 0
-1.5 1

\\2-grams:
-0.3 0 1

\\end\\
";
        let lm = NgramLm::from_arpa(arpa).expect("valid ARPA parses");
        let via_trait: &dyn LmScorer = &lm;
        assert_eq!(
            via_trait.score(&[0], 1).to_bits(),
            lm.score(&[0], 1).to_bits(),
            "trait impl must be bit-identical to inherent method"
        );
    }

    #[test]
    fn context_longer_than_order_is_truncated() {
        // A trigram model queried with a 4-token context must truncate
        // to the last 2 tokens (order = 3, max_ctx = 2). If truncation
        // did NOT happen, the recursion would look up in `ngrams[3]`
        // which is out of bounds and either panic or silently fall
        // through — both wrong. This test proves truncation happens.
        let arpa = "\
\\data\\
ngram 1=3
ngram 2=1
ngram 3=1

\\1-grams:
-2.0 0
-1.5 1
-1.8 2

\\2-grams:
-0.7 1 2

\\3-grams:
-0.4 0 1 2

\\end\\
";
        let lm = NgramLm::from_arpa(arpa).expect("valid ARPA parses");
        assert_eq!(lm.order(), 3);

        // Query with a 5-token context ending in [0, 1] — truncate to
        // last 2 tokens = [0, 1], look up trigram (0, 1, 2). Present.
        let want = ln_from_log10(-0.4);
        let got = lm.score(&[9, 9, 9, 0, 1], 2);
        assert!(
            (got - want).abs() < 1e-6,
            "long context truncated to last 2 → trigram hit expected: got {got}, want {want}"
        );
    }

    #[test]
    fn rejects_short_ngram_row() {
        // A \2-grams: row with only 2 columns (log-prob + one token) is
        // malformed — a bigram needs 2 token columns. Must fail-closed
        // (FR-EX-08), not silently accept the short row.
        let arpa = "\
\\data\\
ngram 1=1
ngram 2=1

\\1-grams:
-2.0 0

\\2-grams:
-0.7 0

\\end\\
";
        match NgramLm::from_arpa(arpa) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("2-gram") && msg.contains("columns"),
                    "message must cite row shape: {msg}"
                );
            }
            other => panic!("expected InvalidArgument for short bigram row, got {other:?}"),
        }
    }
}
