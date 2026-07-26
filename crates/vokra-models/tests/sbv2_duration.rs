//! `SbV2SDP` (stochastic duration predictor) tests (Task 19).
//!
//! All three tests use an empty `flow_layers` stack — a documented,
//! exercised no-op configuration (see `sbv2::duration`'s module doc and
//! its own internal `#[cfg(test)]` module for coverage of a non-empty
//! stack) — so the sampled Gaussian latent `z = rng.next_gaussian() *
//! noise_scale_w` passes straight through to `duration =
//! ceil(exp(z)).max(1)`.

use vokra_core::rng::GaussianSplitMix64;
use vokra_models::sbv2::SbV2SDP;

/// Builds a minimal `SbV2SDP` with an empty flow stack and small,
/// arbitrary (nonzero) tone conditioning tables.
fn make_sdp(d_hidden: usize, n_tones: usize) -> SbV2SDP {
    SbV2SDP::from_weights(
        Vec::new(), // empty flow_layers: documented no-op (see module doc)
        vec![0.1; n_tones * d_hidden],
        vec![0.05; d_hidden],
        d_hidden,
        n_tones,
    )
}

/// `sample`'s output length always equals `text_seq_len` (one duration per
/// phoneme position).
#[test]
fn sample_returns_text_seq_len_durations() {
    let d_hidden = 4;
    let n_tones = 3;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, n_tones);
    let hidden = vec![0.2_f32; text_seq_len * d_hidden];
    let tones = vec![0u8, 1, 2, 1, 0];
    let mut rng = GaussianSplitMix64::new(7);

    let out = sdp.sample(&hidden, &tones, text_seq_len, &mut rng, 0.8);

    assert_eq!(out.len(), text_seq_len, "one duration per phoneme position");
}

/// Every sampled duration is non-negative (in fact `>= 1`, the VITS-family
/// convention for a valid frame count — see `sbv2::duration`'s module doc),
/// exercised across a noise scale wide enough to draw both positive and
/// negative Gaussian latents.
#[test]
fn sample_returns_non_negative_durations() {
    let d_hidden = 4;
    let n_tones = 3;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, n_tones);
    let hidden = vec![-0.4_f32; text_seq_len * d_hidden];
    let tones = vec![2u8, 0, 1, 2, 0];
    let mut rng = GaussianSplitMix64::new(123);

    let out = sdp.sample(&hidden, &tones, text_seq_len, &mut rng, 1.5);

    assert!(
        out.iter().all(|&d| d >= 0),
        "all durations must be non-negative: {out:?}"
    );
}

/// Same seed (a fresh `GaussianSplitMix64` from the same value) produces
/// the exact same duration sequence — `sample` carries no hidden
/// non-deterministic state beyond the RNG it's handed.
#[test]
fn sample_is_deterministic_for_fixed_seed() {
    let d_hidden = 4;
    let n_tones = 3;
    let text_seq_len = 5;
    let sdp = make_sdp(d_hidden, n_tones);
    let hidden = vec![0.3_f32; text_seq_len * d_hidden];
    let tones = vec![1u8, 1, 0, 2, 1];
    let seed = 42;

    let out_1 = {
        let mut rng = GaussianSplitMix64::new(seed);
        sdp.sample(&hidden, &tones, text_seq_len, &mut rng, 0.6)
    };
    let out_2 = {
        let mut rng = GaussianSplitMix64::new(seed);
        sdp.sample(&hidden, &tones, text_seq_len, &mut rng, 0.6)
    };

    assert_eq!(out_1, out_2, "same seed must produce same durations");
}
