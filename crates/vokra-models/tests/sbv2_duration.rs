//! `SbV2SDP` (stochastic duration predictor) and `length_regulate`
//! external-API tests — the shape-and-invariants coverage that lives
//! outside the primitive-focused `tests/sbv2_sdp.rs` (Blocker 2c). Every
//! test here uses [`SbV2SDP::empty`] — the documented, exercised zero-
//! weight identity path (see `sbv2::duration`'s module doc's "Empty-flows
//! / all-zero-weight identity path (backward compat)" section and its own
//! `#[cfg(test)]` module for coverage of a non-empty flow stack).
//!
//! # No `tones` argument
//!
//! Post-Blocker-2c `SbV2SDP::sample` does not take a `tones` slice — real
//! VITS SDP has no tone-level conditioning at the SDP; per-phoneme tones
//! are baked into `hidden` by `SbV2TextEncoder::forward`'s `tone_embed`
//! table before `hidden` reaches this function. The tone-related tests
//! that lived here pre-Blocker-2c (`sample_returns_non_negative_durations`,
//! `sample_is_deterministic_for_fixed_seed`) were rewritten to exercise
//! the `g`-driven conditioning path instead.

use vokra_core::rng::GaussianSplitMix64;
use vokra_models::sbv2::{SbV2SDP, length_regulate};

/// Builds a minimal empty-SDP with the given hidden width and equal `gin`
/// (matches the synthetic-test factories' `d_speaker == d_model`
/// convention — real SBV2 v2 base uses `d_hidden = 192, gin = 512`).
fn make_sdp(d_hidden: usize, gin: usize) -> SbV2SDP {
    SbV2SDP::empty(d_hidden, gin)
}

/// `sample`'s output length always equals `text_seq_len` (one duration per
/// phoneme position), invariant across noise scales.
#[test]
fn sample_returns_text_seq_len_durations() {
    let d_hidden = 4;
    let gin = 4;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, gin);
    let hidden = vec![0.2_f32; text_seq_len * d_hidden];
    let g = vec![0.1_f32; gin];
    let mut rng = GaussianSplitMix64::new(7);
    let out = sdp.sample(&hidden, text_seq_len, &g, &mut rng, 0.8);
    assert_eq!(out.len(), text_seq_len, "one duration per phoneme position");
}

/// Every sampled duration is `>= 1` (the VITS-family convention for a
/// valid frame count — `SbV2SDP::sample`'s `.max(1.0)` clamp). Exercised
/// across a noise scale wide enough to draw both positive and negative
/// Gaussian latents from the RNG.
#[test]
fn sample_returns_non_negative_durations() {
    let d_hidden = 4;
    let gin = 4;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, gin);
    let hidden = vec![-0.4_f32; text_seq_len * d_hidden];
    let g = vec![0.3_f32; gin];
    let mut rng = GaussianSplitMix64::new(123);
    let out = sdp.sample(&hidden, text_seq_len, &g, &mut rng, 1.5);
    assert!(
        out.iter().all(|&d| d >= 1),
        "all durations must be >= 1 (SbV2SDP::sample .max(1.0) clamp): {out:?}"
    );
}

/// Same seed (a fresh `GaussianSplitMix64` from the same value) produces
/// the exact same duration sequence — `sample` carries no hidden
/// non-deterministic state beyond the RNG it's handed.
#[test]
fn sample_is_deterministic_for_fixed_seed() {
    let d_hidden = 4;
    let gin = 4;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, gin);
    let hidden = vec![0.3_f32; text_seq_len * d_hidden];
    let g = vec![0.5_f32; gin];
    let seed = 42;

    let out_1 = {
        let mut rng = GaussianSplitMix64::new(seed);
        sdp.sample(&hidden, text_seq_len, &g, &mut rng, 0.6)
    };
    let out_2 = {
        let mut rng = GaussianSplitMix64::new(seed);
        sdp.sample(&hidden, text_seq_len, &g, &mut rng, 0.6)
    };
    assert_eq!(out_1, out_2, "same seed must produce same durations");
}

/// Zero-noise (`noise_scale_w == 0.0`) with an empty-flows SDP is fully
/// deterministic — every duration is exactly `1` regardless of `hidden`,
/// `g`, or seed — per `SbV2SDP::empty`'s doc.
#[test]
fn sample_zero_noise_returns_ones_deterministic() {
    let sdp = SbV2SDP::empty(4, 4);
    let hidden = vec![0.7_f32; 3 * 4];
    let g = vec![0.4_f32; 4];
    let mut rng = GaussianSplitMix64::new(99);
    let out = sdp.sample(&hidden, 3, &g, &mut rng, 0.0);
    assert_eq!(out, vec![1_i32; 3]);
}

/// Brief's exact concrete example: 2 phonemes with durations `[2, 3]`,
/// `d_model = 2` — `[[1,2],[3,4]]` expands to `[[1,2],[1,2],[3,4],[3,4],[3,4]]`.
#[test]
fn length_regulate_simple_example() {
    let hidden = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] flat
    let durations = vec![2, 3];
    let out = length_regulate(&hidden, &durations, 2);
    assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]);
}

/// Non-positive durations (`0` and negative) contribute zero rows to the
/// output. This is the `SbV2SDP`-independent public-API contract for
/// defensive callers — `length_regulate` is a standalone `pub fn`, not
/// exclusively fed by `SbV2SDP::sample`'s own `.max(1)`-guaranteed output.
#[test]
fn length_regulate_skips_non_positive_durations() {
    let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 phonemes, d_model=2
    let durations = vec![2, 0, -1]; // second skipped (0), third skipped (negative)
    let out = length_regulate(&hidden, &durations, 2);
    assert_eq!(
        out,
        vec![1.0, 2.0, 1.0, 2.0],
        "only first phoneme (dur=2) emitted"
    );
}
