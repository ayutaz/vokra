//! `SbV2Flow` (VITS2 normalizing flow, Task 21) tests.
//!
//! All three tests here use an empty `coupling_layers` stack — the only
//! stack this external integration-test crate *can* build, since
//! `SbV2AffineCouplingLayer::new` is `pub(crate)` (see `sbv2::flow`'s
//! module doc for the reuse-decision rationale). Coverage of a non-empty
//! stack (the affine-coupling math, and the halves-swap between layers)
//! lives in `sbv2::flow`'s own internal `#[cfg(test)]` module, matching
//! `tests/sbv2_duration.rs`'s identical split for `SbV2SDP`.

use vokra_models::sbv2::SbV2Flow;

/// Empty `coupling_layers` + any `z` → `inverse` returns `z` unchanged
/// (the documented no-op scaffold path; real weight-load parity lands in
/// Task 24-27).
#[test]
fn inverse_empty_flow_returns_z_unchanged() {
    let d_z = 4;
    let mel_seq_len = 3;
    let flow = SbV2Flow::from_layers(Vec::new(), d_z);
    let z: Vec<f32> = (0..mel_seq_len * d_z).map(|i| i as f32 * 0.1).collect();
    let style_vec = vec![0.5, -0.2];
    let speaker_embed = vec![1.0, 2.0, 3.0];

    let out = flow.inverse(&z, mel_seq_len, &style_vec, &speaker_embed);

    assert_eq!(out, z, "empty coupling_layers must be an identity pass");
}

/// Output length always equals `mel_seq_len * d_z` (the `[mel_seq_len,
/// d_z]` row-major shape `z` itself uses).
#[test]
fn inverse_returns_expected_shape() {
    let d_z = 6;
    let mel_seq_len = 5;
    let flow = SbV2Flow::from_layers(Vec::new(), d_z);
    let z = vec![0.0_f32; mel_seq_len * d_z];

    // style_vec/speaker_embed are never read with an empty coupling stack
    // (see SbV2Flow::inverse's panic docs), so any length is valid here.
    let out = flow.inverse(&z, mel_seq_len, &[], &[]);

    assert_eq!(
        out.len(),
        mel_seq_len * d_z,
        "output shape must be [mel_seq_len, d_z]"
    );
}

/// Same input (`z`, `style_vec`, `speaker_embed`) always produces the same
/// output — `inverse` is a pure function with no internal RNG or hidden
/// state.
#[test]
fn inverse_is_deterministic() {
    let d_z = 4;
    let mel_seq_len = 2;
    let flow = SbV2Flow::from_layers(Vec::new(), d_z);
    let z = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let style_vec = vec![0.1, 0.2];
    let speaker_embed = vec![0.3];

    let out_1 = flow.inverse(&z, mel_seq_len, &style_vec, &speaker_embed);
    let out_2 = flow.inverse(&z, mel_seq_len, &style_vec, &speaker_embed);

    assert_eq!(out_1, out_2, "same input must produce same output");
}
