//! Word-level timestamp alignment (M4-20, FR-OP-40 `word_timestamps`).
//!
//! Model-independent **host-side** alignment (never a graph op — the `beam_search`
//! "contrib op" anti-pattern rationale, FR-OP-40). This is the Whisper
//! **cross-attention DTW** alignment of openai-whisper `whisper/timing.py`
//! (`find_alignment` + `dtw`), transcribed from that upstream source (CLAUDE.md
//! ハルシネーション厳禁, ADR M4-20 §D-2): the procedure — per-token
//! normalization over the text axis, a `median_filter` over the audio axis,
//! mean over the selected alignment heads, `dtw(-matrix)`, and the jump
//! extraction — is upstream, not invented.
//!
//! # Not `force_align`
//!
//! Whisper word timestamps (this module) come from **cross-attention DTW**.
//! `force_align` (CLAUDE.md「Alignment」) is the **Montreal Forced Aligner**
//! (MFA) concept — a different algorithm at a different layer. They are not
//! interchangeable; this module is only the Whisper cross-attention path.
//!
//! # Boundary
//!
//! The caller (a model, e.g. the Whisper decoder) supplies the **selected
//! alignment heads'** cross-attention weights over the valid audio frames
//! ([`CrossAttention`]); how those weights are produced (raw QK logits +
//! softmax temperature, or the model's native attention softmax) is the model's
//! responsibility per upstream reference. This core is agnostic to it and
//! validated with synthetic attention (a diagonal ridge → a monotone
//! alignment). The token→word grouping ([`words_from_alignment`]) is likewise
//! tokenizer-specific and supplied by the caller.

use crate::error::{Result, VokraError};

/// One aligned word: a token span (indices into the hypothesis token vector)
/// plus its start / end time in seconds.
///
/// `token_end` is exclusive: the word covers `tokens[token_start..token_end]`.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// First token index (inclusive) of the word in the aligned token slice.
    pub token_start: usize,
    /// One-past-last token index (exclusive) of the word.
    pub token_end: usize,
    /// Word start time in seconds.
    pub start: f32,
    /// Word end time in seconds.
    pub end: f32,
}

/// Selected-alignment-head cross-attention weights, `[n_head, n_text, n_audio]`
/// row-major, restricted to the **valid audio frames** (openai-whisper
/// `weights[:, :, : num_frames // 2]`, ADR M4-20 §D-2). These are the weights
/// the caller supplies to [`token_alignment`].
#[derive(Debug, Clone)]
pub struct CrossAttention {
    /// Flat `[n_head, n_text, n_audio]` row-major attention weights.
    pub weights: Vec<f32>,
    /// Number of selected alignment heads.
    pub n_head: usize,
    /// Number of text tokens being aligned (the alignment matrix rows).
    pub n_text: usize,
    /// Number of audio tokens / frames (the alignment matrix columns).
    pub n_audio: usize,
}

impl CrossAttention {
    /// Validates the shape (`weights.len() == n_head * n_text * n_audio`,
    /// all-finite, non-empty axes) — FR-EX-08: a malformed alignment input is
    /// an explicit error, never a silent truncation.
    pub fn validate(&self) -> Result<()> {
        let expect = self
            .n_head
            .checked_mul(self.n_text)
            .and_then(|x| x.checked_mul(self.n_audio))
            .ok_or_else(|| {
                VokraError::InvalidArgument("word_timing: attention shape overflow".into())
            })?;
        if self.n_head == 0 || self.n_text == 0 || self.n_audio == 0 {
            return Err(VokraError::InvalidArgument(
                "word_timing: attention axes must all be >= 1".into(),
            ));
        }
        if self.weights.len() != expect {
            return Err(VokraError::InvalidArgument(format!(
                "word_timing: attention len {} != n_head({}) * n_text({}) * n_audio({})",
                self.weights.len(),
                self.n_head,
                self.n_text,
                self.n_audio
            )));
        }
        if self.weights.iter().any(|w| !w.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "word_timing: attention weights must be finite".into(),
            ));
        }
        Ok(())
    }
}

/// Alignment tunables (openai-whisper `timing.py` defaults, ADR M4-20 §D-2).
#[derive(Debug, Clone, Copy)]
pub struct AlignmentParams {
    /// Median-filter width over the audio axis (openai-whisper `medfilt_width`
    /// = 7). Must be odd and >= 1.
    pub median_filter_width: usize,
    /// Seconds per audio token. Whisper: `1 / TOKENS_PER_SECOND` where
    /// `TOKENS_PER_SECOND = SAMPLE_RATE / (HOP_LENGTH * 2) = 16000 / 320 = 50`,
    /// i.e. **0.02 s**. Supplied by the caller (model-config-driven) so this
    /// core stays model-independent.
    pub audio_time_per_token: f32,
}

impl Default for AlignmentParams {
    fn default() -> Self {
        Self {
            median_filter_width: 7,
            // Whisper audio-token rate (documented anchor, ADR M4-20 §D-2).
            audio_time_per_token: 0.02,
        }
    }
}

/// Per-token start times (seconds), length `n_text`, from the cross-attention
/// DTW alignment (openai-whisper `find_alignment`, ADR M4-20 §D-2).
///
/// Pipeline (all upstream, transcribed not invented):
/// 1. normalize each `(head, audio)` column over the text axis
///    (`std_mean(dim=text, unbiased=False)`, then `(w - mean) / std`);
/// 2. median-filter over the audio axis (reflect-padded, width
///    [`AlignmentParams::median_filter_width`]);
/// 3. mean over the alignment heads → cost matrix `[n_text, n_audio]`;
/// 4. `dtw(-matrix)` → `(text_indices, time_indices)`;
/// 5. jump extraction: the audio-frame index where each text token first
///    appears × `audio_time_per_token`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] for a malformed [`CrossAttention`]
/// ([`CrossAttention::validate`]) or an even / zero median width.
pub fn token_alignment(attn: &CrossAttention, params: &AlignmentParams) -> Result<Vec<f32>> {
    attn.validate()?;
    if params.median_filter_width == 0 || params.median_filter_width % 2 == 0 {
        return Err(VokraError::InvalidArgument(
            "word_timing: median_filter_width must be odd and >= 1".into(),
        ));
    }
    let (nh, nt, na) = (attn.n_head, attn.n_text, attn.n_audio);

    // (1) Per-(head, audio) normalization over the text axis. `weights[h][i][j]`
    //     at flat `h*nt*na + i*na + j`. For a fixed (h, j) column, subtract the
    //     mean over i and divide by the population std (unbiased=False).
    let mut w = attn.weights.clone();
    for h in 0..nh {
        for j in 0..na {
            let mut mean = 0.0f64;
            for i in 0..nt {
                mean += w[h * nt * na + i * na + j] as f64;
            }
            mean /= nt as f64;
            let mut var = 0.0f64;
            for i in 0..nt {
                let d = w[h * nt * na + i * na + j] as f64 - mean;
                var += d * d;
            }
            var /= nt as f64; // population variance (unbiased=False)
            let std = var.sqrt();
            // Guard std==0 (a flat column, e.g. n_text==1): leave it zero-mean,
            // no division blow-up (upstream divides by std with a nonzero var in
            // practice; the guard is a numerical safety net, ADR M4-20 §D-2).
            let inv = if std > 1e-12 { 1.0 / std } else { 0.0 };
            for i in 0..nt {
                let idx = h * nt * na + i * na + j;
                w[idx] = ((w[idx] as f64 - mean) * inv) as f32;
            }
        }
    }

    // (2) Median filter over the audio axis, per (head, text) row.
    let mut filt = vec![0.0f32; nh * nt * na];
    for h in 0..nh {
        for i in 0..nt {
            let base = h * nt * na + i * na;
            median_filter_row(
                &w[base..base + na],
                params.median_filter_width,
                &mut filt[base..base + na],
            );
        }
    }

    // (3) Mean over heads → cost matrix [n_text, n_audio].
    let mut matrix = vec![0.0f32; nt * na];
    let inv_h = 1.0 / nh as f32;
    for i in 0..nt {
        for j in 0..na {
            let mut s = 0.0f32;
            for h in 0..nh {
                s += filt[h * nt * na + i * na + j];
            }
            matrix[i * na + j] = s * inv_h;
        }
    }

    // (4) DTW on the NEGATED matrix (high attention = low cost).
    let mut neg = matrix;
    for v in &mut neg {
        *v = -*v;
    }
    let (text_indices, time_indices) = dtw(&neg, nt, na);

    // (5) Jump extraction: `time_indices[jumps]` where `jumps` marks where
    //     `text_indices` advances (diff != 0, with jumps[0] = true). Each text
    //     token's start = the audio-frame index at its first appearance.
    let mut token_frame = vec![0usize; nt];
    let mut seen = vec![false; nt];
    for k in 0..text_indices.len() {
        let ti = text_indices[k];
        if ti < nt && !seen[ti] {
            seen[ti] = true;
            token_frame[ti] = time_indices[k];
        }
    }
    Ok(token_frame
        .into_iter()
        .map(|f| f as f32 * params.audio_time_per_token)
        .collect())
}

/// Groups per-token start times into [`WordTiming`]s given each word's token
/// count (`word_token_lens`, tokenizer-specific — supplied by the caller). The
/// word `w` covers `tokens[b_w .. b_{w+1})`; its start is that token's start
/// time and its end is the next word's start (the last word ends at
/// `final_time`, the total audio duration in seconds). ADR M4-20 §D-3.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `sum(word_token_lens) != token_times.len()`.
pub fn words_from_alignment(
    token_times: &[f32],
    word_token_lens: &[usize],
    final_time: f32,
) -> Result<Vec<WordTiming>> {
    let total: usize = word_token_lens.iter().sum();
    if total != token_times.len() {
        return Err(VokraError::InvalidArgument(format!(
            "word_timing: word token count {total} != aligned token count {}",
            token_times.len()
        )));
    }
    let mut out = Vec::with_capacity(word_token_lens.len());
    let mut cursor = 0usize;
    for &len in word_token_lens {
        if len == 0 {
            continue; // an empty word contributes no span
        }
        let token_start = cursor;
        let token_end = cursor + len;
        let start = token_times[token_start];
        // End = the next word's first token time, or the total duration for the
        // last word. Clamp so end >= start (monotone) against tiny DTW jitter.
        let end = if token_end < token_times.len() {
            token_times[token_end]
        } else {
            final_time
        }
        .max(start);
        out.push(WordTiming {
            token_start,
            token_end,
            start,
            end,
        });
        cursor = token_end;
    }
    Ok(out)
}

/// Dynamic time warping (openai-whisper `timing.py::dtw`, ADR M4-20 §D-2).
///
/// `cost` is `[n, m]` row-major. Returns the aligned path as parallel
/// `(text_indices, time_indices)` arrays (both non-decreasing, length = path
/// length). The recurrence keeps a `(n+1) x (m+1)` accumulated-cost grid and a
/// trace grid (`0` = diagonal, `1` = up/advance-text, `2` = left/advance-time),
/// with the upstream tie-break (ties resolve to `2`, advance-time).
fn dtw(cost: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    debug_assert_eq!(cost.len(), n * m);
    let inf = f64::INFINITY;
    // Accumulated cost `(n+1) x (m+1)`, row-major with stride (m+1).
    let stride = m + 1;
    let mut acc = vec![inf; (n + 1) * stride];
    let mut trace = vec![-1i8; (n + 1) * stride];
    acc[0] = 0.0;

    for j in 1..=m {
        for i in 1..=n {
            let c0 = acc[(i - 1) * stride + (j - 1)]; // diagonal
            let c1 = acc[(i - 1) * stride + j]; // up (advance text)
            let c2 = acc[i * stride + (j - 1)]; // left (advance time)
            // Upstream tie-break (timing.py): strict-less cascade, default → 2.
            let (c, t): (f64, i8) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            acc[i * stride + j] = cost[(i - 1) * m + (j - 1)] as f64 + c;
            trace[i * stride + j] = t;
        }
    }

    // Backtrace from (n, m). Boundary rules (timing.py): `trace[0, :] = 2`
    // (at text=0 always advance time), `trace[:, 0] = 1` (at time=0 always
    // advance text).
    for j in 0..stride {
        trace[j] = 2;
    }
    for i in 0..=n {
        trace[i * stride] = 1;
    }
    let mut ti = Vec::new();
    let mut tj = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        ti.push(i - 1);
        tj.push(j - 1);
        match trace[i * stride + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => {
                i -= 1;
            }
            _ => {
                j -= 1;
            }
        }
    }
    ti.reverse();
    tj.reverse();
    (ti, tj)
}

/// PyTorch-`reflect`-padded sliding-window median over one row (openai-whisper
/// `timing.py::median_filter`, ADR M4-20 §D-2). `width` is odd; the output has
/// the same length as `row`. Reflect mode mirrors without repeating the edge
/// sample (`[a,b,c]` pad 2 → `[c,b,a,b,c,b,a]`).
fn median_filter_row(row: &[f32], width: usize, out: &mut [f32]) {
    let n = row.len();
    debug_assert_eq!(out.len(), n);
    if width <= 1 || n == 0 {
        out.copy_from_slice(row);
        return;
    }
    let pad = width / 2;
    let mut window: Vec<f32> = Vec::with_capacity(width);
    for center in 0..n {
        window.clear();
        for k in 0..width {
            let idx = center as isize + k as isize - pad as isize;
            window.push(row[reflect_index(idx, n)]);
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out[center] = window[width / 2];
    }
}

/// Maps a possibly-out-of-range index into `[0, len)` with PyTorch `reflect`
/// semantics (edge sample not duplicated). For `len == 1` every index maps to
/// `0`.
fn reflect_index(idx: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let l = len as isize;
    let period = 2 * (l - 1);
    // Fold into [0, period) then reflect the second half.
    let mut i = idx % period;
    if i < 0 {
        i += period;
    }
    if i >= l {
        i = period - i;
    }
    i as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_index_mirrors_without_duplicating_edge() {
        // len 3: [a(0), b(1), c(2)]. reflect at left: -1→1, -2→2; right: 3→1, 4→0.
        assert_eq!(reflect_index(-1, 3), 1);
        assert_eq!(reflect_index(-2, 3), 2);
        assert_eq!(reflect_index(0, 3), 0);
        assert_eq!(reflect_index(2, 3), 2);
        assert_eq!(reflect_index(3, 3), 1);
        assert_eq!(reflect_index(4, 3), 0);
        // len 1 collapses.
        assert_eq!(reflect_index(-5, 1), 0);
        assert_eq!(reflect_index(9, 1), 0);
    }

    #[test]
    fn median_filter_removes_a_spike() {
        // A single tall spike in an otherwise flat row is killed by a width-3
        // median; the flat neighbours are unchanged.
        let row = [0.0f32, 0.0, 9.0, 0.0, 0.0];
        let mut out = [0.0f32; 5];
        median_filter_row(&row, 3, &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn median_filter_width_one_is_identity() {
        let row = [3.0f32, 1.0, 2.0];
        let mut out = [0.0f32; 3];
        median_filter_row(&row, 1, &mut out);
        assert_eq!(out, row);
    }

    #[test]
    fn dtw_on_diagonal_cost_takes_the_diagonal() {
        // A square cost matrix that is 0 on the diagonal and 1 elsewhere: the
        // min-cost path is the exact diagonal (identity alignment).
        let n = 4;
        let mut cost = vec![1.0f32; n * n];
        for d in 0..n {
            cost[d * n + d] = 0.0;
        }
        let (ti, tj) = dtw(&cost, n, n);
        assert_eq!(ti, vec![0, 1, 2, 3]);
        assert_eq!(tj, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dtw_path_is_monotone_and_spans_corners() {
        // For any non-square cost the path must start at (0,0), end at
        // (n-1, m-1), and never decrease in either axis.
        let (n, m) = (3usize, 5usize);
        // A diagonal-ish ridge: low cost near j ≈ i * (m-1)/(n-1).
        let mut cost = vec![1.0f32; n * m];
        for i in 0..n {
            let j = (i * (m - 1)) / (n - 1);
            cost[i * m + j] = 0.0;
        }
        let (ti, tj) = dtw(&cost, n, m);
        assert_eq!((ti[0], tj[0]), (0, 0));
        assert_eq!((*ti.last().unwrap(), *tj.last().unwrap()), (n - 1, m - 1));
        for k in 1..ti.len() {
            assert!(
                ti[k] >= ti[k - 1] && tj[k] >= tj[k - 1],
                "path must be monotone"
            );
        }
    }

    /// A synthetic diagonal attention ridge (each text token attends to a
    /// distinct, increasing audio frame) must yield **strictly increasing**
    /// per-token start times — the core end-to-end property (ADR M4-20 §D-2).
    #[test]
    fn token_alignment_diagonal_attention_is_monotone() {
        let (nh, nt, na) = (2usize, 4usize, 12usize);
        // Each text token i attends to audio frame `i * (na-1)/(nt-1)` with a
        // clear peak; identical across heads.
        let mut weights = vec![0.0f32; nh * nt * na];
        for h in 0..nh {
            for i in 0..nt {
                let peak = (i * (na - 1)) / (nt - 1);
                for j in 0..na {
                    let dist = (j as isize - peak as isize).unsigned_abs();
                    weights[h * nt * na + i * na + j] = 1.0 / (1.0 + dist as f32);
                }
            }
        }
        let attn = CrossAttention {
            weights,
            n_head: nh,
            n_text: nt,
            n_audio: na,
        };
        let times = token_alignment(&attn, &AlignmentParams::default()).unwrap();
        assert_eq!(times.len(), nt);
        for k in 1..nt {
            assert!(
                times[k] >= times[k - 1],
                "token times must be non-decreasing: {times:?}"
            );
        }
        // The last token must land later than the first (the ridge spans the
        // whole audio axis).
        assert!(
            times[nt - 1] > times[0],
            "alignment must span the audio: {times:?}"
        );
        // All times are within [0, na * dt].
        let dt = AlignmentParams::default().audio_time_per_token;
        for &t in &times {
            assert!(
                (0.0..=na as f32 * dt).contains(&t),
                "time out of range: {t}"
            );
        }
    }

    #[test]
    fn words_from_alignment_groups_and_orders() {
        // token times: 0.0, 0.02, 0.06, 0.10 (4 tokens). Words: [2, 2] tokens.
        let token_times = [0.0f32, 0.02, 0.06, 0.10];
        let words = words_from_alignment(&token_times, &[2, 2], 0.20).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(
            words[0],
            WordTiming {
                token_start: 0,
                token_end: 2,
                start: 0.0,
                end: 0.06
            }
        );
        // last word ends at final_time.
        assert_eq!(
            words[1],
            WordTiming {
                token_start: 2,
                token_end: 4,
                start: 0.06,
                end: 0.20
            }
        );
    }

    #[test]
    fn words_from_alignment_rejects_count_mismatch() {
        let token_times = [0.0f32, 0.02];
        // 3 tokens requested but only 2 aligned.
        assert!(matches!(
            words_from_alignment(&token_times, &[3], 0.10),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn cross_attention_validate_rejects_bad_shape_and_nonfinite() {
        // len mismatch.
        let bad = CrossAttention {
            weights: vec![0.0; 5],
            n_head: 1,
            n_text: 2,
            n_audio: 3,
        };
        assert!(matches!(
            bad.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
        // NaN.
        let nan = CrossAttention {
            weights: vec![f32::NAN, 0.0],
            n_head: 1,
            n_text: 1,
            n_audio: 2,
        };
        assert!(matches!(
            nan.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
        // zero axis.
        let zero = CrossAttention {
            weights: vec![],
            n_head: 0,
            n_text: 1,
            n_audio: 1,
        };
        assert!(matches!(
            zero.validate(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn token_alignment_rejects_even_median_width() {
        let attn = CrossAttention {
            weights: vec![0.0; 1 * 2 * 2],
            n_head: 1,
            n_text: 2,
            n_audio: 2,
        };
        let params = AlignmentParams {
            median_filter_width: 4,
            audio_time_per_token: 0.02,
        };
        assert!(matches!(
            token_alignment(&attn, &params),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
