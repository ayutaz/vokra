//! SBV2 v2 converter — external integration test (SBV2 v2 plan Task 25,
//! 2026-07-26).
//!
//! Mirrors `deberta_convert.rs` (Task 11)'s structure: an unconditional
//! smoke test pinning the externally-reachable `ModelKind` + `ConvertReport`
//! surface, plus a real-checkpoint round-trip `#[ignore]`d pending a real
//! SBV2 v2 safetensors fixture at `tests/fixtures/sbv2/*.safetensors`
//! (Task 28, `tools/parity/sbv2_prepare_checkpoint.py` — design doc §10).
//! Synthetic-fixture behavior (BF16 pass-through, config side-car parsing,
//! internal-consistency checks, provenance stamping) is covered by the
//! inline `#[cfg(test)]` module in
//! `crates/vokra-convert/src/models/sbv2.rs` — this file only pins the
//! externally-reachable surface.

use std::path::{Path, PathBuf};

use vokra_convert::{ModelKind, SbV2ConvertReport, convert_sbv2_file};

/// Repo-root-relative real-fixture directory for the SBV2 v2 safetensors
/// fixtures shared with the SBV2 v2 loader/parity tests
/// (`tests/fixtures/sbv2/`, gated by the committed `*.gguf.sha256` sidecars
/// for their converted GGUF siblings). `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-convert` — `cargo test` sets a test binary's working
/// directory to the crate root, not the invocation directory, so every
/// repo-root fixture path in this workspace is built this way
/// (`parity_sbv2_real.rs`, `parity_whisper.rs`, `parity_kokoro.rs`,
/// `parity_voxtral.rs`, `parity_csm.rs`, `parity_moshi.rs`) rather than as
/// a bare relative literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// `ModelKind::SbV2` exists and is reachable at its canonical path —
/// construction alone is enough; the dispatch body (`convert_file_licensed`)
/// is exercised via `convert_sbv2_real_checkpoint` below once a real
/// checkpoint fixture lands.
#[test]
fn sbv2_variant_exists() {
    let _ = ModelKind::SbV2;
    assert_eq!(ModelKind::from_arg("sbv2"), Some(ModelKind::SbV2));
    assert_eq!(ModelKind::SbV2.as_arg(), "sbv2");
}

/// Unconditional pin on `SbV2ConvertReport`'s externally-reachable shape
/// (mirrors `deberta_convert.rs`'s `convert_report_fields_exist`) — a
/// default-constructed report must read as "nothing converted yet, no
/// config side-car supplied".
#[test]
fn convert_report_fields_exist() {
    let r = SbV2ConvertReport::default();
    assert_eq!(r.read, 0);
    assert_eq!(r.written, 0);
    assert_eq!(r.skipped_non_float, 0);
    assert_eq!(r.bf16_passthrough, 0);
    assert!(!r.hparams_written);
}

/// Real-fixture gated: requires a real SBV2 v2 base safetensors checkpoint
/// plus its JSON config side-car (Task 28's `tools/parity/
/// sbv2_prepare_checkpoint.py` output, design doc §10). Never runs in CI
/// until that fixture is committed under `tests/fixtures/sbv2/` — see
/// `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10 "fixture
/// management" for the sidecar-hash-gated commit convention the parity CI
/// workflow (`parity-sbv2-real.yml`) uses for the same fixture family.
#[test]
#[ignore = "requires real SBV2 v2 safetensors fixture (Task 28)"]
fn convert_sbv2_real_checkpoint() {
    let dir = fixtures_dir();
    let input = dir.join("sbv2-v2-multilingual-base.safetensors");
    let config = dir.join("sbv2-v2-multilingual-base.config.json");
    let output = std::env::temp_dir().join("vokra-sbv2-real-checkpoint-smoke.gguf");

    let report = convert_sbv2_file(&input, &output, Some(&config), None)
        .unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    assert!(report.written > 0);
    assert_eq!(report.read, report.written + report.skipped_non_float);
    assert!(
        report.hparams_written,
        "a real config side-car must produce a hparam-complete GGUF"
    );
}
