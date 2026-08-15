//! DeBERTa v2 / v3 converter — external integration test (SBV2 v2 plan
//! Task 11, 2026-07-26).
//!
//! The real-checkpoint round-trips are `#[ignore]`d pending a real HF
//! safetensors fixture at `tests/fixtures/sbv2/*.safetensors` (Task 30;
//! a future `tools/parity/deberta_v2_prepare_checkpoint.py` is not yet
//! written — mirrors the gating convention
//! `crates/vokra-bert/tests/deberta_v2_loader.rs` already uses for the
//! loader side of this same fixture family).
//! `convert_report_fields_exist` runs unconditionally as the CI-visible
//! pin on `ConvertReport`'s shape. Synthetic-fixture behavior (BF16
//! pass-through, hparam shape-inference, provenance stamping) is covered
//! by the inline `#[cfg(test)]` modules in
//! `crates/vokra-convert/src/models/deberta_v2.rs` /
//! `deberta_v3.rs` — this file only pins the externally-reachable surface.

use std::path::{Path, PathBuf};

use vokra_convert::{ConvertReport, convert_deberta_v2_file, convert_deberta_v3_file};

/// Repo-root-relative real-fixture directory for the DeBERTa v2/v3
/// safetensors fixtures shared with the SBV2 v2 loader/parity/converter
/// tests (`tests/fixtures/sbv2/`, gated by the committed `*.gguf.sha256`
/// sidecars for their converted GGUF siblings). `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-convert` — `cargo test` sets a test binary's
/// working directory to the crate root, not the invocation directory, so
/// every repo-root fixture path in this workspace is built this way
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

#[test]
#[ignore = "requires real HF safetensors fixture (Task 30)"]
fn deberta_v2_convert_smoke() {
    let input = fixtures_dir().join("deberta-v2-large-japanese-char-wwm.safetensors");
    let output = std::env::temp_dir().join("vokra-deberta-v2-smoke.gguf");
    let report = convert_deberta_v2_file(&input, &output, None, None)
        .unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    // Post-Blocker-4 (Task 30 rename table, 2026-08-06): `report.written`
    // includes the Q/K content-position duplications (`query_proj.weight`
    // → wq.weight + wq_pos.weight, +1 per layer) and rel_embeddings →
    // per-layer pos_embed duplications, so `written` no longer equals
    // `read - skipped_non_float`. Just assert the invariant that survives
    // the rename: `written` is > 0 and >= `read`'s renamed-consumable
    // subset (i.e. `written >= read - skipped_non_float`, and can exceed
    // when duplications fire).
    assert!(report.written > 0);
    assert!(report.written >= report.read - report.skipped_non_float);
}

#[test]
#[ignore = "requires real HF safetensors fixture (Task 30)"]
fn deberta_v3_convert_smoke() {
    let input = fixtures_dir().join("deberta-v3-large.safetensors");
    let output = std::env::temp_dir().join("vokra-deberta-v3-smoke.gguf");
    let report = convert_deberta_v3_file(&input, &output, None, None)
        .unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    // Same invariant relaxation as `deberta_v2_convert_smoke` (see there).
    assert!(report.written > 0);
    assert!(report.written >= report.read - report.skipped_non_float);
}

#[test]
fn convert_report_fields_exist() {
    let r = ConvertReport::default();
    assert_eq!(r.read, 0);
    assert_eq!(r.written, 0);
    assert_eq!(r.skipped_non_float, 0);
    assert_eq!(r.bf16_passthrough, 0);
}
