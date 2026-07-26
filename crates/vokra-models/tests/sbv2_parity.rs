//! `sbv2::parity` (per-tensor atol calibration table) tests (Task 26).
//!
//! Both tests exercise `tolerance_for`'s two branches: a tensor name listed
//! in `PER_TENSOR_ATOL` returns its override, and any other name (including
//! the empty string) falls back to `ATOL_DEFAULT`. See `sbv2::parity`'s
//! module docs for the honest-atol derivation behind each override value —
//! these tests only pin the lookup *behavior*, not the values' rationale.

use vokra_models::sbv2::{ATOL_DEFAULT, tolerance_for};

/// Every `PER_TENSOR_ATOL` entry must be reachable through `tolerance_for`
/// with its exact override value.
#[test]
fn tolerance_for_known_tensor_returns_per_tensor_override() {
    assert_eq!(tolerance_for("bert_hidden_ja"), 0.02);
    assert_eq!(tolerance_for("bert_hidden_en"), 0.02);
    assert_eq!(tolerance_for("sdp_sample"), 0.05);
    assert_eq!(tolerance_for("z_latent"), 0.03);
}

/// A tensor name with no `PER_TENSOR_ATOL` entry — including the empty
/// string — falls back to `ATOL_DEFAULT` rather than erroring.
#[test]
fn tolerance_for_unknown_tensor_returns_default() {
    assert_eq!(tolerance_for("unknown_tensor_name"), ATOL_DEFAULT);
    assert_eq!(tolerance_for(""), ATOL_DEFAULT);
}
