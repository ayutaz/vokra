//! sber-gigaam-multilingual converter — external integration test
//! (coverage-audit-2026-08-03 Wave B).
//!
//! Mirrors `sbv2_convert.rs` (SBV2 v2 plan Task 25)'s structure: two
//! unconditional smoke tests pinning the externally-reachable
//! `ModelKind` + `SberGigaamMultilingualReport` surface a downstream
//! caller depends on, plus a real-checkpoint round-trip `#[ignore]`d
//! pending a real sber-gigaam-multilingual safetensors fixture at
//! `tests/fixtures/sber-gigaam-multilingual/*.safetensors` (from
//! `tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py` —
//! the upstream `.pt` is torch pickle so a Python-side bridge is
//! required, mirroring the DFN3 / DAC / Kokoro / nkf_aec pattern).
//! Synthetic-fixture behaviour (BF16 pass-through byte-identity,
//! mixed F32 / F16 conversion, license override, provenance
//! stamping) is covered by the inline `#[cfg(test)]` module in
//! `crates/vokra-convert/src/models/sber_gigaam_multilingual.rs` —
//! this file only pins the externally-reachable surface and gates
//! the real-checkpoint round-trip until a fixture lands.

use std::path::{Path, PathBuf};

use vokra_convert::{
    ModelKind, SberGigaamMultilingualReport, convert_file_licensed,
    convert_sber_gigaam_multilingual_file,
};

/// Repo-root-relative real-fixture directory for the
/// sber-gigaam-multilingual safetensors fixture the `#[ignore]`d
/// round-trip below expects. `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-convert` — `cargo test` sets a test binary's
/// working directory to the crate root, not the invocation directory,
/// so every repo-root fixture path in this workspace is built this
/// way (`parity_sbv2_real.rs`, `parity_whisper.rs`,
/// `parity_kokoro.rs`, `parity_voxtral.rs`, `parity_csm.rs`,
/// `parity_moshi.rs`) rather than as a bare relative literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sber-gigaam-multilingual")
}

/// `ModelKind::SberGigaamMultilingual` exists and every documented
/// `--model` argument spelling parses back to it. Construction alone
/// is enough for the `ModelKind` reachability half; the dispatch body
/// (`convert_file_licensed`) is exercised via
/// `convert_sber_gigaam_multilingual_real_checkpoint` below once a
/// real checkpoint fixture lands.
#[test]
fn sber_gigaam_multilingual_variant_exists() {
    let _ = ModelKind::SberGigaamMultilingual;
    assert_eq!(
        ModelKind::SberGigaamMultilingual.as_arg(),
        "sber-gigaam-multilingual",
    );
    // Every spelling `from_arg` accepts must land on the same variant
    // — silently accepting one and dropping another would be a
    // dispatcher regression that a downstream `--model` script could
    // trip in a way that only manifests as "silent unknown model" in
    // the CLI (per FR-EX-08, unknown model IDs are hard errors, so
    // the round-trip must stay total).
    for spelling in [
        "sber-gigaam-multilingual",
        "sber_gigaam_multilingual",
        "gigaam-multilingual",
        "gigaam_multilingual",
        "ai-sage/gigaam-multilingual",
        "ai-sage/GigaAM-Multilingual",
        "salute-developers/gigaam-multilingual",
        "salute-developers/GigaAM-Multilingual",
    ] {
        assert_eq!(
            ModelKind::from_arg(spelling),
            Some(ModelKind::SberGigaamMultilingual),
            "spelling {spelling:?} must route to SberGigaamMultilingual",
        );
    }
}

/// Unconditional pin on `SberGigaamMultilingualReport`'s
/// externally-reachable shape (mirrors `deberta_convert.rs`'s
/// `convert_report_fields_exist` / `sbv2_convert.rs`'s
/// `convert_report_fields_exist`) — a default-constructed report must
/// read as "nothing converted yet". Additive-only guard on the four
/// counters the CLI verify output prints and the model-card generator
/// summarises; a new counter would fail this test loudly (rather
/// than the caller silently reading the wrong field).
#[test]
fn convert_report_fields_exist() {
    let r = SberGigaamMultilingualReport::default();
    assert_eq!(r.read, 0);
    assert_eq!(r.written, 0);
    assert_eq!(r.skipped_non_float, 0);
    assert_eq!(r.bf16_passthrough, 0);
}

/// Real-fixture gated: requires a real sber-gigaam-multilingual
/// safetensors checkpoint (the flattened output of
/// `tools/parity/sber_gigaam_multilingual_prepare_checkpoint.py` fed
/// the upstream `.pt`) under
/// `tests/fixtures/sber-gigaam-multilingual/`. Never runs in CI until
/// that fixture is committed — the runtime binding + real-weight
/// parity is deferred to owner sign-off in
/// `docs/license-audit.md §3.1` per the audit ticket
/// (`docs/tickets/coverage-audit-2026-08-03/wave-b/
/// sber-gigaam-multilingual.md`), so this test's job is to pin the
/// **externally-reachable** conversion surface end-to-end once the
/// fixture is ready. Landing the fixture is the flip-the-switch
/// moment: this test moves from `#[ignore]` to unconditional (and the
/// audit ticket's Owner critical path row for §3.1 sign-off closes).
#[test]
#[ignore = "requires real sber-gigaam-multilingual safetensors fixture"]
fn convert_sber_gigaam_multilingual_real_checkpoint() {
    let dir = fixtures_dir();
    let input = dir.join("sber-gigaam-multilingual.safetensors");
    let output =
        std::env::temp_dir().join("vokra-sber-gigaam-multilingual-real-checkpoint-smoke.gguf");

    let report = convert_sber_gigaam_multilingual_file(&input, &output, None)
        .unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    assert!(report.written > 0);
    assert_eq!(report.read, report.written + report.skipped_non_float);
    assert!(
        report.bf16_passthrough <= report.written,
        "BF16 counter is a subset of written",
    );
}

/// Real-fixture gated: rerun through the dispatch entrypoint
/// (`convert_file_licensed`) rather than the direct
/// `convert_sber_gigaam_multilingual_file` — pins that the
/// `ModelKind::SberGigaamMultilingual` arm inside
/// `convert_file_licensed` produces the same bytes a direct caller
/// gets (no dispatch-only divergence — the two paths share the same
/// module-level implementation). Ignored on the same fixture gate as
/// the direct test above.
#[test]
#[ignore = "requires real sber-gigaam-multilingual safetensors fixture"]
fn convert_sber_gigaam_multilingual_via_dispatch() {
    let dir = fixtures_dir();
    let input = dir.join("sber-gigaam-multilingual.safetensors");
    let out_direct =
        std::env::temp_dir().join("vokra-sber-gigaam-multilingual-real-checkpoint-direct.gguf");
    let out_dispatch =
        std::env::temp_dir().join("vokra-sber-gigaam-multilingual-real-checkpoint-dispatch.gguf");

    let _direct = convert_sber_gigaam_multilingual_file(&input, &out_direct, None)
        .unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    let _via = convert_file_licensed(
        ModelKind::SberGigaamMultilingual,
        &input,
        &out_dispatch,
        None,
    )
    .unwrap_or_else(|e| panic!("{}: {e}", input.display()));

    let a = std::fs::read(&out_direct).expect("read direct GGUF");
    let b = std::fs::read(&out_dispatch).expect("read dispatch GGUF");
    assert_eq!(
        a, b,
        "direct convert_sber_gigaam_multilingual_file and \
         convert_file_licensed dispatch must produce byte-identical GGUFs",
    );
}
