//! Smoke test verifying that `ModelKind::DebertaV2` and `ModelKind::DebertaV3`
//! enum variants exist and are publicly accessible via the crate root — the
//! bare-minimum contract Task 12 (SBV2 v2 plan) wires up. Deeper dispatch
//! coverage (from_arg / as_arg / convert_file_licensed) lives in the
//! `deberta_convert.rs` integration test alongside the converter smoke tests.

use vokra_convert::ModelKind;

/// Verifies both DebertaV2 and DebertaV3 variants can be named at their
/// canonical path — construction alone is enough; the dispatch body is
/// exercised elsewhere.
#[test]
fn deberta_variants_exist() {
    let _ = ModelKind::DebertaV2;
    let _ = ModelKind::DebertaV3;
}
