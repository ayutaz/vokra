//! DeBERTa v2 / v3 converter — external integration test (SBV2 v2 plan
//! Task 11, 2026-07-26).
//!
//! The real-checkpoint round-trips are `#[ignore]`d pending a real HF
//! safetensors fixture at `tests/fixtures/sbv2/*.safetensors` (Task 30,
//! `tools/parity/deberta_v2_prepare_checkpoint.py` — mirrors the gating
//! convention `crates/vokra-bert/tests/deberta_v2_loader.rs` already uses
//! for the loader side of this same fixture family).
//! `convert_report_fields_exist` runs unconditionally as the CI-visible
//! pin on `ConvertReport`'s shape. Synthetic-fixture behavior (BF16
//! pass-through, hparam shape-inference, provenance stamping) is covered
//! by the inline `#[cfg(test)]` modules in
//! `crates/vokra-convert/src/models/deberta_v2.rs` /
//! `deberta_v3.rs` — this file only pins the externally-reachable surface.

use std::path::Path;

use vokra_convert::{ConvertReport, convert_deberta_v2_file, convert_deberta_v3_file};

#[test]
#[ignore = "requires real HF safetensors fixture (Task 30)"]
fn deberta_v2_convert_smoke() {
    let input = Path::new("tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.safetensors");
    let output = std::env::temp_dir().join("vokra-deberta-v2-smoke.gguf");
    let report = convert_deberta_v2_file(input, &output, None).expect("convert");
    assert!(report.written > 0);
    assert_eq!(report.read, report.written + report.skipped_non_float);
}

#[test]
#[ignore = "requires real HF safetensors fixture (Task 30)"]
fn deberta_v3_convert_smoke() {
    let input = Path::new("tests/fixtures/sbv2/deberta-v3-large.safetensors");
    let output = std::env::temp_dir().join("vokra-deberta-v3-smoke.gguf");
    let report = convert_deberta_v3_file(input, &output, None).expect("convert");
    assert!(report.written > 0);
    assert_eq!(report.read, report.written + report.skipped_non_float);
}

#[test]
fn convert_report_fields_exist() {
    let r = ConvertReport::default();
    assert_eq!(r.read, 0);
    assert_eq!(r.written, 0);
    assert_eq!(r.skipped_non_float, 0);
    assert_eq!(r.bf16_passthrough, 0);
}
