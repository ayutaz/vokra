//! Word/character error rate via Levenshtein edit distance (std-only).
//!
//! Both metrics are the classic edit rate `(S + D + I) / N`: the minimum number
//! of single-token substitutions, deletions and insertions to turn the
//! hypothesis into the reference, divided by the reference length `N`. [`Wer`]
//! tokenises on Unicode whitespace; [`Cer`] tokenises on Unicode scalar values
//! (`char`s, spaces included). Original code — no `jiwer` / `torchmetrics`.

use super::{Direction, Metric, TextMetric};

/// Levenshtein edit distance between two sequences: the minimum number of
/// single-element substitutions, insertions and deletions to turn `a` into `b`.
///
/// Two-row dynamic program — `O(|a|·|b|)` time, `O(|b|)` extra space. The
/// distance is symmetric, so only the reference length matters for the
/// normalisation done by [`Wer`] / [`Cer`].
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let n = b.len();
    // `prev[j]` = distance from the empty prefix of `a` to `b[..j]`.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for (i, ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, bj) in b.iter().enumerate() {
            let cost = usize::from(ai != bj);
            // substitution / deletion / insertion.
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Normalised edit rate: `edit_distance / max(ref_len, 1)`.
///
/// The `max(·, 1)` guards the empty-reference case so the result is finite: an
/// empty reference against an empty hypothesis scores `0.0`, and against a
/// non-empty hypothesis scores that hypothesis's length (one error per inserted
/// token). Like real WER this can exceed `1.0` when insertions dominate.
fn edit_rate<T: PartialEq>(hyp: &[T], reference: &[T]) -> f64 {
    let d = edit_distance(hyp, reference);
    d as f64 / reference.len().max(1) as f64
}

/// Word Error Rate — edit rate over Unicode-whitespace-delimited tokens
/// (`str::split_whitespace`, so leading/trailing/repeated whitespace does not
/// create empty tokens).
#[derive(Debug, Default, Clone, Copy)]
pub struct Wer;

impl Wer {
    /// WER of `hyp` against `reference` (see [`edit_rate`] for the
    /// empty-reference convention).
    pub fn rate(hyp: &str, reference: &str) -> f64 {
        let h: Vec<&str> = hyp.split_whitespace().collect();
        let r: Vec<&str> = reference.split_whitespace().collect();
        edit_rate(&h, &r)
    }
}

impl Metric for Wer {
    fn name(&self) -> &str {
        "wer"
    }
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }
}

impl TextMetric for Wer {
    fn eval_text(&self, hyp: &str, reference: &str) -> f64 {
        Self::rate(hyp, reference)
    }
}

/// Character Error Rate — edit rate over Unicode scalar values (`char`s,
/// whitespace included).
#[derive(Debug, Default, Clone, Copy)]
pub struct Cer;

impl Cer {
    /// CER of `hyp` against `reference` (see [`edit_rate`] for the
    /// empty-reference convention).
    pub fn rate(hyp: &str, reference: &str) -> f64 {
        let h: Vec<char> = hyp.chars().collect();
        let r: Vec<char> = reference.chars().collect();
        edit_rate(&h, &r)
    }
}

impl Metric for Cer {
    fn name(&self) -> &str {
        "cer"
    }
    fn direction(&self) -> Direction {
        Direction::LowerIsBetter
    }
}

impl TextMetric for Cer {
    fn eval_text(&self, hyp: &str, reference: &str) -> f64 {
        Self::rate(hyp, reference)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compares two rates that should be mathematically identical; both sides are
    // computed the same way, so the tolerance only guards against surprise.
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn edit_distance_classic_cases() {
        // The textbook kitten→sitting example is 3 (k→s, e→i, +g).
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(edit_distance(&a, &b), 3);
        // Distance is symmetric.
        assert_eq!(edit_distance(&b, &a), 3);
        // Identity is 0; disjoint single tokens is 1 (a substitution).
        assert_eq!(edit_distance(&["a", "b"], &["a", "b"]), 0);
        assert_eq!(edit_distance(&["a"], &["b"]), 1);
        // Empty vs empty, and empty vs n (n pure inserts/deletes).
        let empty: [char; 0] = [];
        assert_eq!(edit_distance(&empty, &empty), 0);
        assert_eq!(edit_distance(&empty, &b), 7);
    }

    #[test]
    fn wer_hand_computed() {
        // 1 substitution over a 3-word reference → 1/3.
        close(Wer::rate("a x c", "a b c"), 1.0 / 3.0);
        // hyp drops a word: 1 deletion over a 2-word reference → 1/2.
        close(Wer::rate("the", "the cat"), 0.5);
        // hyp adds a word: 1 insertion over a 2-word reference → 1/2.
        close(Wer::rate("the cat sat", "the cat"), 0.5);
        // Perfect transcription → 0. Whitespace runs do not add empty tokens.
        close(Wer::rate("the cat", "the   cat"), 0.0);
    }

    #[test]
    fn cer_hand_computed() {
        // kitten vs sitting: 3 edits over a 7-char reference.
        close(Cer::rate("kitten", "sitting"), 3.0 / 7.0);
        // Identity → 0.
        close(Cer::rate("abc", "abc"), 0.0);
        // One character substitution over a 3-char reference → 1/3.
        close(Cer::rate("cat", "car"), 1.0 / 3.0);
    }

    #[test]
    fn empty_reference_is_finite() {
        // Empty vs empty is 0; empty reference vs a hypothesis counts each
        // hypothesis token as one error (finite, may exceed 1.0).
        close(Wer::rate("", ""), 0.0);
        close(Wer::rate("a b", ""), 2.0);
        close(Cer::rate("", ""), 0.0);
        close(Cer::rate("ab", ""), 2.0);
    }

    #[test]
    fn deterministic() {
        // Same inputs → bit-identical score across calls.
        assert_eq!(Wer::rate("a x c", "a b c"), Wer::rate("a x c", "a b c"));
        assert_eq!(
            Cer::rate("kitten", "sitting"),
            Cer::rate("kitten", "sitting")
        );
    }

    #[test]
    fn trait_surface() {
        // The metric is reachable through the pluggable Metric/TextMetric API.
        let m = Wer;
        assert_eq!(m.name(), "wer");
        assert_eq!(m.direction(), Direction::LowerIsBetter);
        close(m.eval_text("a x c", "a b c"), 1.0 / 3.0);
        assert_eq!(Cer.name(), "cer");
    }
}
