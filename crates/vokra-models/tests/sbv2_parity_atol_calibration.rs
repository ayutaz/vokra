//! Wave-4 PER-TENSOR-ATOL-CALIB (audit rank 18): pins the
//! [`AtolCalibration`] status of every [`PER_TENSOR_ATOL`] entry so
//! nobody can silently loosen a bound without touching the ADR
//! (memory `feedback-honest-parity-atol` redundant-recording rule).
//!
//! Every entry starts at [`AtolCalibration::EstimatedPreFixture`] and
//! flips to [`AtolCalibration::Measured`] only after the owner runs
//! the parity CI workflow_dispatch on a real SBV2 v2 fine-tune
//! checkpoint and records the measured max|Δ| in
//! `docs/adr/sbv2-parity-atol.md`. See [`PER_TENSOR_ATOL`]'s doc
//! "Wave-4 PER-TENSOR-ATOL-CALIB (2026-08-09) — status per entry"
//! section for the exact owner-side calibration steps.

use vokra_models::sbv2::{AtolCalibration, PER_TENSOR_ATOL, atol_calibration_for, tolerance_for};

/// Every [`PER_TENSOR_ATOL`] key MUST have a match arm in
/// [`atol_calibration_for`] — a silent add without corresponding
/// status entry is disallowed. Failing here means someone added a new
/// override without deciding whether it's a pre-fixture estimate or
/// a measured bound (the audit's "no silent loosening" contract).
#[test]
fn every_atol_entry_has_a_calibration_status() {
    for (name, _atol) in PER_TENSOR_ATOL {
        assert!(
            atol_calibration_for(name).is_some(),
            "PER_TENSOR_ATOL entry `{name}` has no `atol_calibration_for` \
             match arm — add one so nobody can silently loosen the bound \
             without touching docs/adr/sbv2-parity-atol.md"
        );
    }
}

/// The **snapshot** of every entry's current calibration status. If
/// the owner runs the parity CI + updates the ADR + flips an arm to
/// `Measured`, this test forces THEM to update this snapshot too —
/// no drift between the code's status and the test's expectation.
/// Kokoro-precedent: this same style of pin caught silent-loosening
/// attempts on the `PROSODY_F0_ATOL = 0.05` entry.
#[test]
fn atol_calibration_status_is_pinned() {
    // Every entry currently EstimatedPreFixture — owner action to flip
    // any of these to `Measured`: see `PER_TENSOR_ATOL`'s doc
    // "Wave-4 PER-TENSOR-ATOL-CALIB" section.
    let expected: &[(&str, AtolCalibration)] = &[
        ("bert_hidden_ja", AtolCalibration::EstimatedPreFixture),
        ("bert_hidden_en", AtolCalibration::EstimatedPreFixture),
        ("sdp_sample", AtolCalibration::EstimatedPreFixture),
        ("z_latent", AtolCalibration::EstimatedPreFixture),
        // Wave-9 (2026-08-09): `waveform` = 1.5 is `Measured` from CI
        // run 31303426623 max |Δ| = 0.9248 × ~1.6× margin. See
        // `PER_TENSOR_ATOL`'s `"waveform"` block-doc for derivation +
        // `docs/adr/sbv2-libm-strategy.md` §2.2 for why the bit-exact
        // libm follow-up is a documented deferral (not a fabricated pass).
        ("waveform", AtolCalibration::Measured),
    ];
    assert_eq!(
        expected.len(),
        PER_TENSOR_ATOL.len(),
        "PER_TENSOR_ATOL count changed ({} entries in code vs {} pinned here) — \
         update this snapshot table alongside the atol table",
        PER_TENSOR_ATOL.len(),
        expected.len(),
    );
    for (name, expected_status) in expected {
        let actual = atol_calibration_for(name).unwrap_or_else(|| {
            panic!("PER_TENSOR_ATOL entry `{name}` lost its calibration status arm")
        });
        assert_eq!(
            actual, *expected_status,
            "PER_TENSOR_ATOL entry `{name}` calibration status changed to {actual:?} \
             — if this is intentional, update BOTH this snapshot AND \
             `docs/adr/sbv2-parity-atol.md` with the measured max|Δ| + derivation \
             (memory `feedback-honest-parity-atol` redundant-recording rule)"
        );
    }
}

/// Every atol value is a POSITIVE float — the CI's per-tensor pass/fail
/// check would silently succeed on a zero or negative tolerance
/// (`|delta| <= 0` is only satisfied by delta==0, which never happens
/// on non-bit-exact reference forward passes). A pre-Wave-4 audit
/// found no ordering enforced here; this test locks it in.
#[test]
fn every_atol_value_is_positive() {
    for (name, atol) in PER_TENSOR_ATOL {
        assert!(
            *atol > 0.0 && atol.is_finite(),
            "PER_TENSOR_ATOL entry `{name}` = {atol} is non-positive or non-finite — \
             an atol of 0 makes the per-tensor pass gate silently unmeetable"
        );
    }
}

/// `tolerance_for` on any manifest tensor name returns a finite
/// positive f32. Consumers rely on this to divide by the atol or feed
/// it into the "abs(delta) <= atol" pass gate — a NaN or negative
/// would be a silent-wrong outcome the audit called out.
#[test]
fn tolerance_for_returns_finite_positive_on_every_manifest_tensor() {
    // Every tensor the Python dumper emits — see the design doc §10
    // "The 11 dumped tensors" table (dumper's TENSOR_SCHEMA).
    let manifest_tensors: &[&str] = &[
        "phoneme_embed",
        "text_hidden",
        "bert_hidden_ja",
        "bert_hidden_en",
        "bert_bridge_out",
        "speaker_embed",
        "style_projected",
        "sdp_sample",
        "mel_hidden",
        "z_latent",
        "waveform",
    ];
    for &name in manifest_tensors {
        let atol = tolerance_for(name);
        assert!(
            atol > 0.0 && atol.is_finite(),
            "tolerance_for({name}) = {atol} is non-positive or non-finite"
        );
    }
}
