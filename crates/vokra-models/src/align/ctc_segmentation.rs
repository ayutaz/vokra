//! `ctc_segmentation` — CTC-based forced alignment (Kürzinger et al. 2020).
//!
//! # Reference
//!
//! - L. Kürzinger et al., "CTC-Segmentation of Large Corpora for German
//!   End-to-end Speech Recognition", Interspeech 2020, arXiv:2007.09127.
//! - `github.com/lumaku/ctc-segmentation` (Apache-2.0) — the canonical
//!   Python reference implementation.
//!
//! # Algorithm summary
//!
//! Given per-frame log-probabilities `L[t, v]` for `t = 0..T` and `v =
//! 0..V` and a token sequence `tokens[0..N]`, we recover monotone
//! non-overlapping time boundaries by running a Viterbi walk on the
//! standard CTC **extended sequence**
//!
//! ```text
//!     ext = [BLANK, tokens[0], BLANK, tokens[1], BLANK, ..., tokens[N-1], BLANK]
//! ```
//!
//! of length `2N + 1`. The Viterbi variable
//!
//! ```text
//!     alpha[t][s] = max log P for aligning frames 0..=t to ext[0..=s]
//! ```
//!
//! follows the standard CTC transition rules restricted to the max operator
//! (Viterbi / segmentation) instead of the sum operator (training):
//!
//! * stay:  `alpha[t][s] = alpha[t-1][s]     + L[t, ext[s]]`
//! * step:  `alpha[t][s] = alpha[t-1][s-1]   + L[t, ext[s]]`
//! * skip:  `alpha[t][s] = alpha[t-1][s-2]   + L[t, ext[s]]`
//!   (only when `ext[s]` is a real token and `ext[s-2]` is a real
//!   token distinct from `ext[s]` — the skip of the intermediate
//!   blank is forbidden for identical adjacent tokens by the
//!   standard CTC collapse rule.)
//!
//! The best terminal is `max(alpha[T-1][2N], alpha[T-1][2N-1])` and we
//! backtrack argmax pointers to recover the per-token frame ranges.
//!
//! # Token → vocab-id convention
//!
//! The public API accepts token *labels* (`&[String]`) so the caller can
//! echo them back into the returned records. The vocab id used inside the
//! DP is derived deterministically by *skipping* the blank slot:
//!
//! ```text
//!     vocab_id(i) = i        if i <  blank_id
//!     vocab_id(i) = i + 1    if i >= blank_id
//! ```
//!
//! For the common `blank_id = 0` case this yields `vocab_id(0..N) =
//! 1..=N`, matching the smoke test's synthetic layout. Callers with a
//! non-trivial vocabulary should map their own tokens into this canonical
//! order before calling; the algorithm itself is oblivious to token
//! labels.
//!
//! # Errors
//!
//! Returns an empty `Vec` when either `tokens.is_empty()` or
//! `num_frames < tokens.len()` (an alignment requires at least one frame
//! per token). A `num_frames * vocab_size` mismatch against
//! `log_probs.len()` panics — this is a caller-provided-shape bug, not a
//! recoverable runtime condition.

use super::AlignedToken;

/// Force-align a token sequence to a per-frame CTC log-probability matrix.
///
/// See the module docstring for the algorithm and the token→vocab id
/// convention.
///
/// # Arguments
///
/// * `log_probs` — flattened `[num_frames, vocab_size]` row-major matrix of
///   **log-probabilities** (natural log). No re-normalisation is performed;
///   the values are consumed as-is.
/// * `num_frames` — the number of time frames `T`.
/// * `vocab_size` — the vocabulary width `V`; must be `>= tokens.len() + 1`
///   (the `+1` accounts for the blank slot).
/// * `blank_id` — the vocab id of the CTC blank; must be `< vocab_size`.
/// * `frame_shift_sec` — the hop length in seconds used to convert
///   frame indices to `start_sec` / `end_sec`.
/// * `tokens` — the ground-truth token labels in emission order.
///
/// # Returns
///
/// A `Vec<AlignedToken>` of the same length as `tokens`, with monotone
/// non-overlapping time boundaries.
pub fn ctc_segmentation(
    log_probs: &[f32],
    num_frames: usize,
    vocab_size: usize,
    blank_id: usize,
    frame_shift_sec: f32,
    tokens: &[String],
) -> Vec<AlignedToken> {
    // ---- fast paths ---------------------------------------------------
    if tokens.is_empty() {
        return Vec::new();
    }
    if num_frames == 0 || num_frames < tokens.len() {
        // Not enough frames to place even one frame per token — the DP is
        // ill-posed. Return empty per the module-level "Errors" contract
        // instead of raising a panic (this is a legitimate runtime
        // condition on very short audio clips).
        return Vec::new();
    }
    assert!(
        blank_id < vocab_size,
        "blank_id ({}) must be < vocab_size ({})",
        blank_id,
        vocab_size,
    );
    assert_eq!(
        log_probs.len(),
        num_frames * vocab_size,
        "log_probs length must equal num_frames * vocab_size ({} vs {} * {} = {})",
        log_probs.len(),
        num_frames,
        vocab_size,
        num_frames * vocab_size,
    );

    // ---- token → vocab-id table --------------------------------------
    // See the module docstring for the "skip the blank slot" convention.
    let token_vocab_ids: Vec<usize> = (0..tokens.len())
        .map(|i| if i < blank_id { i } else { i + 1 })
        .collect();
    // Bounds check so downstream indexing is safe. The +1 in the skip
    // rule can push past `vocab_size` when the caller did not size their
    // vocabulary appropriately.
    for (i, &vid) in token_vocab_ids.iter().enumerate() {
        assert!(
            vid < vocab_size,
            "token {} maps to vocab id {} which is out of range (vocab_size={})",
            i,
            vid,
            vocab_size,
        );
    }

    // ---- extended sequence (CTC blank-interleaved) -------------------
    // ext[0] = BLANK, ext[1] = tokens[0], ext[2] = BLANK, ext[3] = tokens[1],
    // ..., ext[2N] = BLANK. Length = 2N + 1.
    let n = tokens.len();
    let s_len = 2 * n + 1;
    // For every extended-sequence position, remember which vocab id it
    // references — this saves a per-inner-loop branch.
    let mut ext_vocab: Vec<usize> = Vec::with_capacity(s_len);
    for s in 0..s_len {
        if s % 2 == 0 {
            ext_vocab.push(blank_id);
        } else {
            ext_vocab.push(token_vocab_ids[s / 2]);
        }
    }

    // ---- Viterbi DP ---------------------------------------------------
    // alpha[t][s] = max log P for aligning frames 0..=t to ext[0..=s].
    // We only keep the previous frame's row (rolling), but still keep the
    // full backpointer matrix because we need it for backtracking.
    const NEG_INF: f32 = f32::NEG_INFINITY;
    let t_frames = num_frames;
    let mut prev = vec![NEG_INF; s_len];
    let mut curr = vec![NEG_INF; s_len];
    // Backpointer: for each (t, s) store the *previous* s value we came
    // from (usize::MAX = no predecessor / initial state).
    let mut bp: Vec<usize> = vec![usize::MAX; t_frames * s_len];

    // Row t = 0: initialise only the two leftmost positions — CTC always
    // allows either the leading blank or the first real token to consume
    // frame 0.
    // Row-0 observations live in `log_probs[0..vocab_size]`; no explicit
    // frame-index multiplication (dropping the `0 * vocab_size` that
    // clippy's `erasing_op` lint would otherwise flag).
    prev[0] = log_probs[ext_vocab[0]]; // leading blank
    if s_len > 1 {
        prev[1] = log_probs[ext_vocab[1]]; // tokens[0]
    }

    for t in 1..t_frames {
        // Reset current row so any position we do not visit stays NEG_INF.
        for slot in curr.iter_mut() {
            *slot = NEG_INF;
        }
        let row_offset = t * vocab_size;
        for s in 0..s_len {
            // Gather up to three predecessors.
            let stay = prev[s];
            let step = if s >= 1 { prev[s - 1] } else { NEG_INF };
            // Skip transition: only allowed when the current extended
            // position is a real token (odd `s`) *and* the two-back
            // extended position is a real token distinct from us. The
            // even/odd separation guarantees `s - 2` is also odd and thus
            // also a real token, so the "distinct" check reduces to a
            // vocab-id compare.
            let skip = if s >= 2 && s % 2 == 1 && ext_vocab[s - 2] != ext_vocab[s] {
                prev[s - 2]
            } else {
                NEG_INF
            };

            // argmax {stay, step, skip} → best predecessor.
            let (best_val, best_prev_s) = argmax3(
                stay,
                s,
                step,
                s.saturating_sub(1),
                skip,
                s.saturating_sub(2),
            );
            if best_val == NEG_INF {
                continue;
            }
            let obs = log_probs[row_offset + ext_vocab[s]];
            curr[s] = best_val + obs;
            bp[t * s_len + s] = best_prev_s;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    // ---- pick best terminal & backtrack ------------------------------
    // Standard CTC termination allows the path to end in either the
    // trailing blank (ext[2N] = ext[s_len-1]) or the last real token
    // (ext[2N-1] = ext[s_len-2]).
    let last_row = &prev;
    let (mut s, _best_score) = {
        let a = last_row[s_len - 1];
        let b = if s_len >= 2 {
            last_row[s_len - 2]
        } else {
            NEG_INF
        };
        if a >= b {
            (s_len - 1, a)
        } else {
            (s_len - 2, b)
        }
    };
    // Degenerate: no finite path (should not happen for well-formed
    // inputs but be honest about it).
    if last_row[s] == NEG_INF {
        return Vec::new();
    }

    // Walk backwards to produce, for each frame t = 0..T, the extended
    // position s_at_frame[t].
    let mut s_at_frame = vec![0usize; t_frames];
    s_at_frame[t_frames - 1] = s;
    for t in (1..t_frames).rev() {
        let prev_s = bp[t * s_len + s];
        // If we hit the sentinel we had no predecessor — this means the
        // DP started from the row-0 initialisation for this s. Anchor at
        // the initial position (0 or 1) that matches.
        if prev_s == usize::MAX {
            // We reached the initial row; treat every remaining earlier
            // frame as sitting at s = 0 or s = 1 (whichever this s could
            // have come from). Pin at min(s, 1) which is valid because
            // only positions 0 and 1 are initialised.
            let anchor = s.min(1);
            for tt in (0..t).rev() {
                s_at_frame[tt] = anchor;
            }
            // The remaining backward walk terminates here — no further
            // predecessor assignment to `s` is required.
            break;
        }
        s = prev_s;
        s_at_frame[t - 1] = s;
    }

    // ---- convert to per-token frame ranges ---------------------------
    // For every real-token position ext[s = 2k+1] find the first and last
    // frame index t whose backtracked position equals s. Frames on the
    // interleaving blanks (even s) are excluded from that token but
    // still consumed by the total span.
    let mut token_first: Vec<Option<usize>> = vec![None; n];
    let mut token_last: Vec<Option<usize>> = vec![None; n];
    for (t, &s_here) in s_at_frame.iter().enumerate() {
        if s_here % 2 == 1 {
            let k = s_here / 2;
            if token_first[k].is_none() {
                token_first[k] = Some(t);
            }
            token_last[k] = Some(t);
        }
    }

    // ---- assemble result --------------------------------------------
    // If a token was somehow never visited (should not happen when the
    // backtrack reached a valid terminal) fall back to a one-frame span
    // wedged between neighbours to keep the "one record per input token"
    // contract without violating monotonicity.
    let mut out: Vec<AlignedToken> = Vec::with_capacity(n);
    for k in 0..n {
        let (first, last) = match (token_first[k], token_last[k]) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                // Wedge: last-known end (or 0) to last-known end + 1.
                let prev_end = out
                    .last()
                    .map(|r| r.end_sec / frame_shift_sec)
                    .unwrap_or(0.0);
                let a = prev_end as usize;
                (
                    a.min(t_frames.saturating_sub(1)),
                    a.min(t_frames.saturating_sub(1)),
                )
            }
        };

        let start_sec = first as f32 * frame_shift_sec;
        // `end_sec` is the exclusive upper bound — the boundary between
        // this token's last frame and the next frame — so add one frame
        // shift to the last-frame index.
        let end_sec = (last as f32 + 1.0) * frame_shift_sec;

        // Confidence heuristic: mean softmax-normalised posterior over
        // the aligned frames for this token's vocab id. Softmax is
        // computed per frame across the full vocabulary from the
        // log-prob row (numerically-stable log-sum-exp).
        let vid = token_vocab_ids[k];
        let mut acc = 0.0f32;
        let span = last - first + 1;
        for t in first..=last {
            let row = &log_probs[t * vocab_size..(t + 1) * vocab_size];
            acc += softmax_prob(row, vid);
        }
        let confidence = if span > 0 { acc / span as f32 } else { 0.0 };
        // Clamp to (epsilon, 1.0] so the "> 0" contract holds even for
        // very low-probability paths.
        let confidence = confidence.clamp(f32::MIN_POSITIVE, 1.0);

        out.push(AlignedToken {
            text: tokens[k].clone(),
            start_sec,
            end_sec,
            confidence,
        });
    }

    // Post-process: enforce strict monotonicity of start times and
    // strictly-positive spans. The backtracked ranges are already
    // ordered by construction, but blank-only frames between two tokens
    // can leave `token_last[k-1] + 1 == token_first[k]` which is fine
    // (spans do not overlap because `end_sec` is exclusive). We only
    // need to guard against equal or reversed starts, which the DP
    // never produces on a valid path but which we still defensively
    // fix by nudging forward one frame.
    for i in 1..out.len() {
        if out[i].start_sec <= out[i - 1].start_sec {
            let nudged = out[i - 1].start_sec + frame_shift_sec;
            out[i].start_sec = nudged;
            if out[i].end_sec <= out[i].start_sec {
                out[i].end_sec = out[i].start_sec + frame_shift_sec;
            }
        }
    }

    out
}

/// Return the greatest of three values along with the caller-supplied
/// tag associated with the winner. Ties go to the leftmost argument to
/// keep the algorithm deterministic across platforms (no `f32::max`
/// tie-breaking dependency).
#[inline]
fn argmax3(a: f32, ta: usize, b: f32, tb: usize, c: f32, tc: usize) -> (f32, usize) {
    let mut best = a;
    let mut tag = ta;
    if b > best {
        best = b;
        tag = tb;
    }
    if c > best {
        best = c;
        tag = tc;
    }
    (best, tag)
}

/// Numerically-stable softmax of a log-prob row evaluated at `vid`.
///
/// Given `row[v] = log p_v` (up to a shared additive constant), return
/// `exp(row[vid] - lse(row))` where `lse` is `log-sum-exp`. Guaranteed
/// in `[0, 1]`.
#[inline]
fn softmax_prob(row: &[f32], vid: usize) -> f32 {
    // Two-pass log-sum-exp with the shift by the row max for stability.
    let mut m = f32::NEG_INFINITY;
    for &x in row {
        if x > m {
            m = x;
        }
    }
    if m == f32::NEG_INFINITY {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &x in row {
        s += (x - m).exp();
    }
    // s is > 0 because at least one exponent is 1.0.
    ((row[vid] - m).exp() / s).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic 20-frame log-prob matrix where token `a` (vocab
    /// id 1) peaks at frames 0..=5, `b` (vocab id 2) at 6..=12 and `c`
    /// (vocab id 3) at 13..=19; the blank (vocab id 0) is a weak
    /// background everywhere. Returns the row-major `[T, V]` matrix and
    /// the natural token labels.
    fn synthetic_abc() -> (Vec<f32>, usize, usize, usize, Vec<String>) {
        const T: usize = 20;
        const V: usize = 4;
        const BLANK_ID: usize = 0;
        // Base log-prob (very low but finite) for non-peaking entries.
        // Using a moderate negative keeps every DP branch finite so the
        // Viterbi max is well-defined.
        let base: f32 = -10.0;
        let peak: f32 = 0.0;
        let mut lp = vec![base; T * V];
        // Blank stays a bit above `base` so the DP has a valid initial
        // path through the leading blank without swamping the token peaks.
        for t in 0..T {
            lp[t * V + BLANK_ID] = -5.0;
        }
        // Token peaks.
        for t in 0..=5 {
            lp[t * V + 1] = peak; // a
        }
        for t in 6..=12 {
            lp[t * V + 2] = peak; // b
        }
        for t in 13..=19 {
            lp[t * V + 3] = peak; // c
        }
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        (lp, T, V, BLANK_ID, tokens)
    }

    #[test]
    fn ctc_segmentation_produces_monotone_boundaries() {
        let (lp, t, v, blank_id, tokens) = synthetic_abc();
        let frame_shift = 0.02_f32;
        let aligned = ctc_segmentation(&lp, t, v, blank_id, frame_shift, &tokens);

        assert_eq!(aligned.len(), 3, "one record per input token");

        // Monotone strictly-increasing starts and strictly positive spans.
        for i in 0..aligned.len() {
            assert!(
                aligned[i].end_sec > aligned[i].start_sec,
                "end must strictly follow start for token {}",
                i
            );
            if i > 0 {
                assert!(
                    aligned[i].start_sec >= aligned[i - 1].start_sec,
                    "starts must be monotone non-decreasing (i={})",
                    i
                );
                // With three well-separated peaks the boundaries must strictly
                // advance.
                assert!(
                    aligned[i].start_sec > aligned[i - 1].start_sec,
                    "starts must strictly increase across distinct peaks (i={})",
                    i
                );
            }
            assert!(
                (0.0..=1.0).contains(&aligned[i].confidence),
                "confidence in (0, 1] for token {}",
                i
            );
            assert!(aligned[i].confidence > 0.0);
        }

        // Coverage of the 0.4 s clip (20 frames × 20 ms). The first token
        // must start at or before ~1 frame in, and the last must end at
        // or near the end of the clip.
        let total_sec = t as f32 * frame_shift;
        assert!(aligned[0].start_sec <= 2.0 * frame_shift);
        assert!(aligned.last().unwrap().end_sec >= total_sec - 2.0 * frame_shift);

        // Labels are echoed verbatim.
        assert_eq!(aligned[0].text, "a");
        assert_eq!(aligned[1].text, "b");
        assert_eq!(aligned[2].text, "c");
    }

    #[test]
    fn ctc_segmentation_empty_tokens_returns_empty() {
        // A caller with no ground-truth tokens must get an empty answer
        // regardless of what the log-prob matrix looks like.
        let lp = vec![0.0_f32; 4 * 3];
        let out = ctc_segmentation(&lp, 4, 3, 0, 0.02, &[]);
        assert!(out.is_empty());
    }
}
