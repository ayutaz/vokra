//! `sbv2::parity` (per-tensor atol calibration table) tests (Task 26 +
//! WP-24).
//!
//! The `tolerance_for_*` tests exercise `tolerance_for`'s two branches: a
//! tensor name listed in `PER_TENSOR_ATOL` returns its override, and any
//! other name (including the empty string) falls back to `ATOL_DEFAULT`.
//! `utmos_atol_is_pinned_at_0_05` pins the WP-24 UTMOS quality-gate bound
//! (see `sbv2::parity` module docs for the honest-atol derivation).
//!
//! These tests only pin lookup *behavior* and the constant *value*, not
//! the derivation — widening any of them requires updating the module doc's
//! rationale first (memory `feedback-honest-parity-atol`).

use vokra_models::sbv2::{ATOL_DEFAULT, UTMOS_ATOL, tolerance_for};

/// Every `PER_TENSOR_ATOL` entry must be reachable through `tolerance_for`
/// with its exact override value.
///
/// 2026-08-11: bounds updated to match the measured floors landed with
/// the encoder.conv order fix (crates/vokra-bert/src/deberta_v2.rs
/// `EncoderConv::forward`). See `PER_TENSOR_ATOL`'s rustdoc and
/// `tests/fixtures/sbv2/atol-measurements.json` for the redundant
/// derivation records.
#[test]
fn tolerance_for_known_tensor_returns_per_tensor_override() {
    assert_eq!(tolerance_for("bert_hidden_ja"), 0.05);
    assert_eq!(tolerance_for("bert_hidden_en"), 0.02);
    assert_eq!(tolerance_for("bert_bridge_out"), 0.07);
    assert_eq!(tolerance_for("mel_hidden"), 0.07);
    assert_eq!(tolerance_for("sdp_sample"), 0.05);
    assert_eq!(tolerance_for("z_latent"), 0.08);
}

/// A tensor name with no `PER_TENSOR_ATOL` entry — including the empty
/// string — falls back to `ATOL_DEFAULT` rather than erroring.
#[test]
fn tolerance_for_unknown_tensor_returns_default() {
    assert_eq!(tolerance_for("unknown_tensor_name"), ATOL_DEFAULT);
    assert_eq!(tolerance_for(""), ATOL_DEFAULT);
}

/// WP-24: pins the SBV2 UTMOS-delta gate at exactly `0.05` MOS points, so
/// widening it (to hide a real perceptual regression) is a code change
/// with a paper trail. The derivation lives in `sbv2::parity`'s module
/// doc; any change here must update the doc's rationale *and*
/// `parity_sbv2_real.rs`'s tail-position UTMOS assertion together, then be
/// recorded in an ADR — same discipline as Kokoro's `PROSODY_F0_ATOL`.
#[test]
fn utmos_atol_is_pinned_at_0_05() {
    assert_eq!(UTMOS_ATOL, 0.05_f64);
}
