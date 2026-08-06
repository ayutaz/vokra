//! DNSMOS converter — external integration test (coverage-audit Wave A
//! ticket `dnsmos-p808-p835`, 2026-08-03).
//!
//! This file pins the externally-reachable surface of the DNSMOS
//! converter: the crate-root re-exports (`convert_dnsmos_file` +
//! `DnsmosReport`) plus the `ModelKind::Dnsmos` dispatch path through
//! `convert_file`. Synthetic fixture behaviour (BF16 pass-through, hard-
//! error on non-DNSMOS input, license override) is covered by the
//! inline `#[cfg(test)]` module in
//! `crates/vokra-convert/src/models/dnsmos.rs`; this file adds the
//! cross-crate contract fences on top:
//!
//! * `convert_dnsmos_file` is reachable via the crate root (a private
//!   `models::` module would be dead code otherwise).
//! * `convert_file(ModelKind::Dnsmos, …)` and the direct
//!   `convert_dnsmos_file` entry land the **same GGUF bytes** for the
//!   same input — a caller who prefers `--model dnsmos-p808-p835` via
//!   the CLI and a caller who calls the file-based converter directly
//!   must not observe any drift.
//! * The ticket's canonical CLI aliases (`dnsmos`, `dnsmos-p808-p835`,
//!   `microsoft/dnsmos`) all resolve to `ModelKind::Dnsmos`.
//!
//! Real-checkpoint parity (against the upstream Microsoft ONNX
//! ``dnsmos_local.py`` reference scorer) is a deferred owner follow-up
//! per the ticket §Implementation effort estimate — no real ONNX
//! fixture is committed here.

use std::path::PathBuf;

use vokra_convert::{DnsmosReport, ModelKind, convert_dnsmos_file, convert_file};

/// Per-test unique scratch path (PID + tag + nanosecond suffix). Same
/// discipline as `crates/vokra-convert/src/models/dnsmos.rs`'s inline
/// tests — two parallel `cargo test` binaries never collide.
fn scratch_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-dnsmos-rt-{}-{}-{}.{}",
        std::process::id(),
        tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        ext,
    ));
    p
}

/// Builds a minimal DNSMOS-like safetensors buffer with one tensor per
/// bundle variant (both `p808.` and `p835.` prefixed). Mirrors the
/// inline test fixture in `models/dnsmos.rs` but stays local to this
/// integration test so the two do not share private state.
fn safetensors_full_bundle() -> Vec<u8> {
    // Non-trivial payloads so a silent byte-level corruption cannot
    // trivially round-trip.
    let p808: Vec<u8> = [1.0f32, -2.5, 3.5, -0.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let p835: Vec<u8> = [0.5f32, 1.5, 2.5, -1.0, -3.0, 42.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let header = r#"{"p808.model_v8.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"p835.sig_bak_ovr.conv1.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[16,40]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&p808);
    buf.extend_from_slice(&p835);
    buf
}

#[test]
fn crate_root_reexports_reach_convert_dnsmos_file() {
    // Compilation itself pins the re-export: if `convert_dnsmos_file`
    // and `DnsmosReport` are not accessible from the crate root, this
    // test does not build. The runtime assertion is a defensive
    // check that the file-based entry runs a full conversion without
    // panicking on the smallest legal bundle.
    let input = scratch_path("reexport", "safetensors");
    let output = scratch_path("reexport", "gguf");
    std::fs::write(&input, safetensors_full_bundle()).expect("write input");

    let report: DnsmosReport =
        convert_dnsmos_file(&input, &output, None).expect("convert reached crate root");

    assert_eq!(report.bundle_variants, 2, "both variants detected");
    assert_eq!(report.written, 2, "both tensors passed through");
    assert_eq!(report.skipped_non_float, 0);

    let out_bytes = std::fs::read(&output).expect("read output");
    assert!(!out_bytes.is_empty(), "GGUF must be a non-empty file");

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}

#[test]
fn convert_file_dispatch_matches_direct_convert_dnsmos_file() {
    // Two callers, one bundle: `convert_file(ModelKind::Dnsmos, …)`
    // and the direct `convert_dnsmos_file` must land byte-identical
    // GGUFs. Any drift would indicate a metadata key being written by
    // only one of the two arms.
    let input = scratch_path("parity", "safetensors");
    let out_dispatch = scratch_path("parity-dispatch", "gguf");
    let out_direct = scratch_path("parity-direct", "gguf");
    std::fs::write(&input, safetensors_full_bundle()).expect("write input");

    let summary =
        convert_file(ModelKind::Dnsmos, &input, &out_dispatch).expect("convert_file dispatch");
    assert_eq!(summary.model, ModelKind::Dnsmos);
    assert!(
        summary.tensor_count > 0,
        "dispatch summary must report the emitted tensor count"
    );

    let report =
        convert_dnsmos_file(&input, &out_direct, None).expect("direct convert_dnsmos_file");
    assert_eq!(report.written, summary.tensor_count);

    let a = std::fs::read(&out_dispatch).expect("read dispatch output");
    let b = std::fs::read(&out_direct).expect("read direct output");
    assert_eq!(
        a, b,
        "dispatch and direct callers must produce byte-identical GGUFs — a \
         drift here indicates one arm stamps metadata the other does not"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&out_dispatch).ok();
    std::fs::remove_file(&out_direct).ok();
}

#[test]
fn convert_file_summary_notes_carry_bundle_variant_count() {
    // The dispatch arm's `notes` string must surface the bundle
    // variant count — the CLI prints it and an operator relies on it
    // to notice a partial bundle without opening the GGUF.
    let input = scratch_path("notes", "safetensors");
    let output = scratch_path("notes", "gguf");
    std::fs::write(&input, safetensors_full_bundle()).expect("write input");

    let summary = convert_file(ModelKind::Dnsmos, &input, &output).expect("convert_file");
    let joined = summary.notes.join(" | ");
    assert!(
        joined.contains("dnsmos:"),
        "notes must carry the `dnsmos:` prefix, got: {joined}"
    );
    assert!(
        joined.contains("2 bundle variant"),
        "notes must report the detected bundle count, got: {joined}"
    );

    std::fs::remove_file(&input).ok();
    std::fs::remove_file(&output).ok();
}
