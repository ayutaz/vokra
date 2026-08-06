//! `rttm` — Rich Transcription Time Marked (RTTM) output writer.
//!
//! # Primary source
//!
//! The line format is the one **pyannote-core** emits from
//! `Annotation.write_rttm()` — the de-facto standard for pyannote /
//! kaldi / dscore-consumable diarization output:
//!
//! ```text
//! SPEAKER {uri} 1 {start} {duration} <NA> <NA> {label} <NA> <NA>
//! ```
//!
//! - `pyannote-core` source (MIT, Copyright (c) 2020 CNRS):
//!   <https://github.com/pyannote/pyannote-core/blob/develop/pyannote/core/annotation.py>
//! - NIST RT-09 evaluation plan defines the underlying `SPEAKER` record
//!   type with the ten `Type / File-id / Channel / Turn-onset /
//!   Turn-duration / Orthography / Speaker-type / Speaker-id / Confidence /
//!   Signal-lookahead-time` columns; the last four are `<NA>` for
//!   diarization output (no orthographic transcription, no priori speaker
//!   type, no per-turn confidence, no signal lookahead).
//!
//! Every line is exactly ten space-separated tokens (`SPEAKER` + nine
//! payload fields) and is terminated by a single `\n`. Consumers that
//! parse whitespace-separated columns (`str.split()` in Python, `%s` in
//! kaldi rttm loaders) recover the same field order.
//!
//! # Float rendering
//!
//! Times are rendered with two decimals (`{:.2}`) — a hop of 10 ms
//! (the pyannote default) is representable exactly at that precision and
//! `dscore` / `pyannote.metrics` treat two-decimal RTTM as first-class.
//! Callers requiring millisecond precision (`{:.3}`) can post-process the
//! output; the two-decimal rule here matches the task specification.
//!
//! # Speaker id rendering
//!
//! Cluster index `k: usize` renders as `SPEAKER_{k:02}` — at least two
//! digits, never truncated. `k = 0..100` render as `SPEAKER_00 ..
//! SPEAKER_99` (fixed-width, sorts lexicographically the same way as
//! numerically); `k >= 100` widens automatically (`SPEAKER_100`,
//! `SPEAKER_1000`, ...) and stops sorting lexicographically-numerically —
//! this is the standard pyannote / dscore behaviour, mirrored here.
//!
//! # Zero-dependency invariant (NFR-DS-02)
//!
//! Uses only [`std::fmt`] — no external crate. The output buffer is a
//! single `String`; no other allocations occur (see [`write_rttm`]).
//!
//! # No silent fallback (FR-EX-08)
//!
//! Neither entry point returns a `Result` — both are total functions on
//! their inputs (an empty segment list produces an empty string; a merge
//! with a zero / negative `merge_gap_s` is well-defined: no adjacent pair
//! can satisfy `gap < merge_gap_s <= 0` for a non-overlapping strictly
//! monotonic input, so the segments pass through unchanged, and an
//! overlapping input still merges because a negative gap is `< 0.0` — no
//! silent surprise).

use std::fmt::Write as _;

/// One speaker turn in a diarization output.
///
/// Times are in seconds and always non-negative; `speaker_id` is the
/// 0-indexed cluster label produced by the diarization pipeline (or by
/// hand, for tests). No unit conversion is performed by the writer — the
/// caller is responsible for converting frame indices to seconds using the
/// correct hop / sample-rate combination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarizationSegment {
    /// Segment start time in seconds.
    pub start_s: f32,
    /// Segment duration in seconds.
    pub duration_s: f32,
    /// 0-indexed speaker cluster (renders as `SPEAKER_{id:02}`).
    pub speaker_id: usize,
}

/// Writes a full RTTM file body for one `file_id`.
///
/// The input is expected to already be sorted by `start_s` and (typically)
/// pre-merged with [`merge_segments`] — the writer performs no re-ordering
/// or extra merging, so callers can retain full control over the output
/// segmentation.
///
/// An empty `segments` slice produces an empty string (not a lone
/// newline).
///
/// # Line format
///
/// See the module docstring — each segment renders as one line of ten
/// space-separated tokens, terminated by `\n`.
///
/// # Panics
///
/// Never: [`write!`] on a `String` cannot fail, and the `expect` covering
/// that impossibility is a documentation aid rather than a runtime
/// guard.
pub fn write_rttm(file_id: &str, segments: &[DiarizationSegment]) -> String {
    // The only allocation performed by this function. `writeln!` on a
    // `String` (via `std::fmt::Write`) writes in place without reallocation
    // past the amortised growth the `String` schedules itself, so a
    // sensible up-front reservation keeps the total allocation count at
    // one for a typical short segment list.
    //
    // Each line is roughly `len("SPEAKER  1 000.00 000.00 <NA> <NA> SPEAKER_00 <NA> <NA>\n")`
    // (~56 bytes) plus `file_id.len()`; we use 56 as the fixed per-line
    // over-estimate — 2-decimal times up to 999.99 s (~ 16 minutes) fit,
    // and for longer sessions the `String` will grow as needed.
    let mut out = String::with_capacity(segments.len() * (56 + file_id.len()));
    for seg in segments {
        // `writeln!` on a `String` uses `std::fmt::Write` and unconditionally
        // emits `\n` (Unix line terminator) — even on Windows it does NOT
        // emit `\r\n`, so the RTTM byte stream is portable across hosts.
        //
        // Line format is transcribed verbatim from the module docstring.
        writeln!(
            &mut out,
            "SPEAKER {} 1 {:.2} {:.2} <NA> <NA> SPEAKER_{:02} <NA> <NA>",
            file_id, seg.start_s, seg.duration_s, seg.speaker_id,
        )
        .expect("writing to a String cannot fail");
    }
    out
}

/// Merges adjacent same-speaker segments whose end-to-start gap is
/// strictly less than `merge_gap_s`.
///
/// # Semantics
///
/// The input is walked once in order — no re-sorting. For each incoming
/// segment `seg`:
///
/// 1. If the previously kept segment `last` has `last.speaker_id ==
///    seg.speaker_id` **and** `seg.start_s - (last.start_s + last.duration_s)
///    < merge_gap_s`, `seg` is folded into `last` by extending its
///    duration to cover `max(last_end, seg_end) - last.start_s`. Overlapping
///    input (gap < 0) always satisfies the second condition and therefore
///    merges — `merge_gap_s` behaves as a *maximum* allowed gap, not as an
///    absolute-value filter.
/// 2. Otherwise `seg` is appended as a new kept segment.
///
/// Different speaker ids never merge, even when the gap is arbitrarily
/// small (test `merge_segments_different_speakers_never_merge`).
///
/// # Complexity
///
/// `O(n)` time, one allocation for the output `Vec` (pre-reserved for
/// `segments.len()`; the merged output can only be shorter).
pub fn merge_segments(
    segments: &[DiarizationSegment],
    merge_gap_s: f32,
) -> Vec<DiarizationSegment> {
    let mut out: Vec<DiarizationSegment> = Vec::with_capacity(segments.len());
    for seg in segments {
        if let Some(last) = out.last_mut() {
            let last_end = last.start_s + last.duration_s;
            let gap = seg.start_s - last_end;
            if last.speaker_id == seg.speaker_id && gap < merge_gap_s {
                // Extend the last kept segment. Use `max` so a fully-inside
                // `seg` (i.e. seg_end <= last_end) does not shorten `last`;
                // this handles overlapping / duplicate inputs cleanly.
                let seg_end = seg.start_s + seg.duration_s;
                let new_end = last_end.max(seg_end);
                last.duration_s = new_end - last.start_s;
                continue;
            }
        }
        out.push(*seg);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty input → empty output (no lone newline, no whitespace).
    ///
    /// Consumers scripting `wc -l` over the RTTM would otherwise
    /// double-count an empty diarization as "one turn". The empty-string
    /// invariant keeps that count honest.
    #[test]
    fn write_rttm_empty_returns_empty_string() {
        let out = write_rttm("any-file-id", &[]);
        assert_eq!(out, "", "empty segments must render to an empty string");
    }

    /// Single segment: check the exact line — ten tokens, single-space
    /// separated, two-decimal floats, `SPEAKER_00` two-digit cluster,
    /// trailing `\n`.
    #[test]
    fn write_rttm_single_segment_renders_correctly() {
        let seg = DiarizationSegment {
            start_s: 1.5,
            duration_s: 2.3,
            speaker_id: 0,
        };
        let out = write_rttm("audio", &[seg]);
        assert_eq!(
            out, "SPEAKER audio 1 1.50 2.30 <NA> <NA> SPEAKER_00 <NA> <NA>\n",
            "single-segment RTTM line diverged from the pyannote-core format",
        );
    }

    /// Speaker ids 0, 1, 10 → `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_10` —
    /// two-digit padding is always applied (never truncated for `k >= 10`).
    #[test]
    fn write_rttm_multiple_speakers_render_with_padded_ids() {
        let segments = [
            DiarizationSegment {
                start_s: 0.0,
                duration_s: 1.0,
                speaker_id: 0,
            },
            DiarizationSegment {
                start_s: 1.0,
                duration_s: 1.0,
                speaker_id: 1,
            },
            DiarizationSegment {
                start_s: 2.0,
                duration_s: 1.0,
                speaker_id: 10,
            },
        ];
        let out = write_rttm("meeting", &segments);
        let expected = "\
SPEAKER meeting 1 0.00 1.00 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER meeting 1 1.00 1.00 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER meeting 1 2.00 1.00 <NA> <NA> SPEAKER_10 <NA> <NA>
";
        assert_eq!(
            out, expected,
            "multi-speaker RTTM diverged from the pyannote-core format",
        );
        // Redundant per-line invariants — if `expected` above ever drifts
        // by whitespace, these anchor the *intent* (padded ids) even when
        // the diff is noisy:
        assert!(out.contains(" SPEAKER_00 "), "SPEAKER_00 must be padded");
        assert!(out.contains(" SPEAKER_01 "), "SPEAKER_01 must be padded");
        assert!(out.contains(" SPEAKER_10 "), "SPEAKER_10 must be intact");
    }

    /// Two segments 0.3 s apart (same speaker) merge under the typical
    /// 0.5 s gap.
    #[test]
    fn merge_segments_within_gap_merges() {
        // Seg 1: [0.0, 1.0) — ends at 1.0.
        // Seg 2: [1.3, 2.3) — starts at 1.3, gap = 0.3 < 0.5.
        let input = [
            DiarizationSegment {
                start_s: 0.0,
                duration_s: 1.0,
                speaker_id: 0,
            },
            DiarizationSegment {
                start_s: 1.3,
                duration_s: 1.0,
                speaker_id: 0,
            },
        ];
        let out = merge_segments(&input, 0.5);
        assert_eq!(out.len(), 1, "0.3 s gap < 0.5 s must merge");
        assert_eq!(out[0].start_s, 0.0, "merged start preserves first start");
        // Merged duration = new_end (2.3) - start (0.0) = 2.3.
        assert!(
            (out[0].duration_s - 2.3).abs() < 1e-6,
            "merged duration expected 2.3 s, got {}",
            out[0].duration_s,
        );
        assert_eq!(out[0].speaker_id, 0, "speaker id preserved on merge");
    }

    /// Two segments 1.0 s apart (same speaker) do NOT merge under the
    /// typical 0.5 s gap.
    #[test]
    fn merge_segments_outside_gap_stay_separate() {
        // Seg 1: [0.0, 1.0) — ends at 1.0.
        // Seg 2: [2.0, 3.0) — starts at 2.0, gap = 1.0 > 0.5.
        let input = [
            DiarizationSegment {
                start_s: 0.0,
                duration_s: 1.0,
                speaker_id: 0,
            },
            DiarizationSegment {
                start_s: 2.0,
                duration_s: 1.0,
                speaker_id: 0,
            },
        ];
        let out = merge_segments(&input, 0.5);
        assert_eq!(out.len(), 2, "1.0 s gap >= 0.5 s must stay separate");
        assert_eq!(&out, &input, "non-merging pass-through must be identity");
    }

    /// Different speaker ids never merge, even with a tiny gap.
    #[test]
    fn merge_segments_different_speakers_never_merge() {
        // Seg 1: [0.0, 1.0) speaker 0.
        // Seg 2: [1.1, 2.1) speaker 1 — gap = 0.1 < 0.5 but speaker differs.
        let input = [
            DiarizationSegment {
                start_s: 0.0,
                duration_s: 1.0,
                speaker_id: 0,
            },
            DiarizationSegment {
                start_s: 1.1,
                duration_s: 1.0,
                speaker_id: 1,
            },
        ];
        let out = merge_segments(&input, 0.5);
        assert_eq!(out.len(), 2, "different speakers must not merge");
        assert_eq!(&out, &input, "cross-speaker pass-through must be identity",);
    }

    /// Overlapping (negative-gap) same-speaker segments merge and the
    /// output covers `max(seg1_end, seg2_end)` — a fully-contained second
    /// segment does not shorten the first. Not one of the required tests
    /// but pins the "extend by max" branch of the merge semantics
    /// (documented on `merge_segments`) so future refactors do not
    /// silently switch to seg2_end.
    #[test]
    fn merge_segments_overlapping_extends_by_max_end() {
        let input = [
            // Seg 1: [0.0, 2.0)
            DiarizationSegment {
                start_s: 0.0,
                duration_s: 2.0,
                speaker_id: 0,
            },
            // Seg 2: [1.0, 1.5) — fully inside seg 1, seg_end = 1.5 < 2.0.
            DiarizationSegment {
                start_s: 1.0,
                duration_s: 0.5,
                speaker_id: 0,
            },
        ];
        let out = merge_segments(&input, 0.5);
        assert_eq!(out.len(), 1, "overlap must merge");
        assert!(
            (out[0].duration_s - 2.0).abs() < 1e-6,
            "contained seg must not shrink last, got duration {}",
            out[0].duration_s,
        );
    }
}
