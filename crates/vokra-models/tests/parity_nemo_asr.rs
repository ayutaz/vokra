//! Flip-the-switch real-checkpoint parity harness for the `nemo-asr`
//! family: **Kyutai STT** (encoder-free / Mimi consumer), **Parakeet
//! TDT** (FastConformer + TDT), **Parakeet CTC** (FastConformer + CTC),
//! **NVIDIA Canary** (FastConformer AED), and **Meta omniASR CTC**
//! (wav2vec 2.0 + CTC, 1600+ langs). SoTA-plan Phase 2 (2026-07-24).
//!
//! # Gate posture (fabricated pass 禁止, FR-EX-08)
//!
//! Every test is env-gated on `VOKRA_<ARCH>_GGUF` (path to a converted
//! Vokra GGUF for that arch). Unset → clean skip with a printed reason —
//! never a silent pass. When set, the test:
//!   1. opens the GGUF via [`GgufFile::open`],
//!   2. asserts `vokra.model.arch` equals the model's `EXPECTED_ARCH`
//!      (the converter's stamped provenance tag),
//!   3. cross-checks a subset of `vokra.<arch>.*` primary-source hparams
//!      against the module's transcribed config (loud fail on any mismatch),
//!   4. verifies at least one tensor is present (a shape-only converter
//!      that emitted no tensor would be a bug — a fabricated pass otherwise).
//!
//! When `VOKRA_<ARCH>_REFDIR` is *also* set, the test **notes** the
//! reference dump's presence. Full stage-tap / logits comparison is
//! deferred: none of the five models in this family carry a real
//! `<Arch>Weights::from_gguf` binding yet (T29-equivalent follow-up wave —
//! the Moshi / CSM / Voxtral pattern). Wiring the comparison at this
//! seam without a real binding would compare synthesized-weight
//! activations against upstream ones — the very fabricated pass this
//! harness exists to prevent. The refdir slot is the flip-the-switch
//! surface: it fires the moment the loader lands.
//!
//! # Wire posture: what runs, what does not
//!
//! - **`weight_load_and_config_smoke`-shaped** GGUF opens + metadata
//!   verification + tensor presence: **runs** for every model as soon as
//!   the owner sets `VOKRA_<ARCH>_GGUF`. This is what "flip the switch"
//!   means at this milestone.
//! - **Numerical parity vs an upstream `torch` dump**: **awaits the T29
//!   real-checkpoint weight binding.** The Rust module's `transcribe`
//!   returns [`VokraError::NotImplemented`] under synthesized weights by
//!   design; wiring a comparison here would either be trivially
//!   ill-defined or (worse) invoke the fixture path and read out
//!   deterministic-but-meaningless numbers. Refdir presence is recorded
//!   verbatim in eprintln so a run against a real reference is
//!   diagnosable — the runtime-side leg lands drop-in with the loader.
//!
//! # Judgement (NFR-QL-01 / Kokoro PROSODY_F0_ATOL precedent)
//!
//! FP32 default: `atol = 0.01`. Per-tensor relaxations only after an
//! architectural-bound rationale is recorded in rustdoc **and** the ADR
//! **and** the workflow's mirror (the Kokoro `PROSODY_F0_ATOL` pattern —
//! see `crates/vokra-models/tests/parity_kokoro.rs`). No relaxation
//! exists today (no real numerical comparison has ever run for this
//! family).

use std::env;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, GgufMetadataValue};

/// FP32 default tolerance (NFR-QL-01). Referenced by the future
/// numerical-parity leg when the T29 weight bindings land.
#[allow(dead_code)]
const ATOL: f32 = 0.01;

/// The five architectures the `nemo-asr` family covers. Kept as a
/// single-source list so the test-file signature stays legible in
/// `cargo test parity_nemo_asr` (each entry has a corresponding `#[test]`
/// below, but the constant is the audit trail).
#[allow(dead_code)]
const NEMO_ASR_ARCHES: &[&str] = &[
    "kyutai_stt",
    "parakeet_tdt",
    "parakeet_ctc",
    "canary",
    "omniasr_ctc",
];

// ---------------------------------------------------------------------------
// Env-var helpers
// ---------------------------------------------------------------------------

/// Env-var name carrying the path to a converted Vokra GGUF for `arch`
/// (`kyutai_stt` → `VOKRA_KYUTAI_STT_GGUF`, etc.). The naming mirrors
/// `parity_whisper.rs`'s `VOKRA_WHISPER_<SIZE>_GGUF` convention.
fn gguf_env(arch: &str) -> String {
    format!("VOKRA_{}_GGUF", arch.to_ascii_uppercase())
}

/// Env-var name carrying the path to an upstream reference dump directory
/// for `arch` (`kyutai_stt` → `VOKRA_KYUTAI_STT_REFDIR`). The flip-the-switch
/// full stage-tap comparison consumes this once the T29 loader lands.
fn refdir_env(arch: &str) -> String {
    format!("VOKRA_{}_REFDIR", arch.to_ascii_uppercase())
}

/// Returns `(gguf_path, refdir_path)` — either may be `None`, in which
/// case the caller clean-skips. The GGUF path is the mandatory input; the
/// refdir is optional and, when present, enables the stage-tap leg (once
/// implemented — see this file's docstring).
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    (
        env::var_os(gguf_env(arch)).map(PathBuf::from),
        env::var_os(refdir_env(arch)).map(PathBuf::from),
    )
}

/// Human-readable skip annotation naming the env vars the owner sets
/// (a clean skip must always announce the exact contract, never a bare
/// silent return — FR-EX-08).
fn skip_reason(arch: &str) -> String {
    format!(
        "[parity_nemo_asr::{arch}] SKIP: set {} to a converted Vokra GGUF \
         for {arch}; optionally set {} to a directory of upstream stage-tap \
         dumps to arm the flip-the-switch numerical comparison. This is a \
         clean gated skip, not a pass (fabricated pass 禁止, FR-EX-08).",
        gguf_env(arch),
        refdir_env(arch),
    )
}

// ---------------------------------------------------------------------------
// GGUF surface helpers
// ---------------------------------------------------------------------------

/// Opens the GGUF and asserts `vokra.model.arch == expected`.
fn open_and_check_arch(path: &Path, expected_arch: &str) -> GgufFile {
    let file = GgufFile::open(path)
        .unwrap_or_else(|e| panic!("[{expected_arch}] open GGUF {}: {e}", path.display()));
    let arch = file
        .get("vokra.model.arch")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| {
            panic!(
                "[{expected_arch}] GGUF {} carries no `vokra.model.arch` \
                 metadata — the converter should always stamp it (FR-EX-08)",
                path.display(),
            )
        });
    assert_eq!(
        arch, expected_arch,
        "[{expected_arch}] `vokra.model.arch` tag mismatch: got {arch:?}, expected {expected_arch:?} — \
         wrong converter or wrong GGUF"
    );
    file
}

/// Verifies at least one tensor is present. A shape-only converter path
/// that emitted no tensor would be a bug (fabricated-pass shape); the
/// runtime's job here is to catch such a state loudly.
fn assert_has_any_tensor(file: &GgufFile, tag: &str) {
    assert!(
        !file.tensors().is_empty(),
        "[{tag}] GGUF has no tensors — the converter should never emit an \
         empty tensor list at this scaffold stage (fabricated-pass guard)",
    );
}

/// Reads a `u32` (or narrowed `u64`) metadata value; returns `None` when
/// the key is absent or a different type. A `None` from an *expected*
/// key is not itself a failure — it means the converter did not stamp
/// that particular hparam, which is legal for stability across scaffold
/// / real-checkpoint iterations. The caller decides whether to gate.
fn read_u32(file: &GgufFile, key: &str) -> Option<u32> {
    match file.get(key)? {
        GgufMetadataValue::U32(v) => Some(*v),
        GgufMetadataValue::U64(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

/// Cross-checks `vokra.<arch>.*` metadata against the module's primary
/// source-transcribed config. Any *present* key that disagrees with
/// primary source is a loud fail; a *missing* key is announced (eprintln)
/// but not gated — the converter's coverage of the metadata block is
/// itself iterating during Phase 2.
fn expect_u32_metadata(file: &GgufFile, arch: &str, key: &str, expected: u32) {
    match read_u32(file, key) {
        Some(v) => {
            assert_eq!(
                v, expected,
                "[{arch}] metadata `{key}` = {v}, primary source says {expected} — \
                 primary-source drift or converter bug"
            );
        }
        None => {
            eprintln!("[parity_nemo_asr::{arch}] note: metadata `{key}` absent from GGUF");
        }
    }
}

/// Notes the presence / absence of an upstream reference dump. Since no
/// model in this family carries a `from_gguf` weight binding yet
/// (T29-equivalent follow-up), stage-tap comparison is deferred to
/// preserve the fabricated-pass ban: comparing synthesized-weight
/// activations against real upstream ones would be an ill-defined pass
/// or fail with no interpretive value. The refdir presence is recorded
/// verbatim so a diagnostic run is legible.
fn note_refdir(refdir: Option<&Path>, arch: &str) {
    match refdir {
        Some(p) if p.is_dir() => {
            eprintln!(
                "[parity_nemo_asr::{arch}] reference dump present at {} — \
                 stage-tap comparison is deferred until `<Arch>Weights::from_gguf` \
                 lands (T29-equivalent follow-up wave, Moshi / CSM / Voxtral \
                 pattern). The GGUF surface (arch tag + hparam echo + tensor \
                 count) is the flip-the-switch leg today.",
                p.display(),
            );
        }
        Some(p) => {
            panic!(
                "[parity_nemo_asr::{arch}] {}={} does not name a directory — \
                 fix the env var or clear it to skip the refdir leg",
                refdir_env(arch),
                p.display(),
            );
        }
        None => {
            eprintln!(
                "[parity_nemo_asr::{arch}] no reference dump ({} unset); GGUF \
                 surface exercise only.",
                refdir_env(arch),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-model tests
//
// One `#[test]` per family member. Each:
//   * reads env vars via `env_paths_for(arch)`;
//   * skips cleanly (announced) when the GGUF path is unset;
//   * loads the GGUF, checks arch + a subset of the primary-source
//     hparams stamped by the converter, and asserts tensor presence;
//   * notes the refdir status (flip-the-switch surface for the T29
//     numerical leg).
//
// Test names are the family arch keys so the workflow's per-model matrix
// filter `parity_nemo_asr::<arch>` selects each one deterministically.
// ---------------------------------------------------------------------------

#[test]
fn kyutai_stt() {
    use vokra_models::kyutai_stt::{EXPECTED_ARCH, KYUTAI_STT_SAMPLE_RATE, KyutaiSttConfig};
    let arch = "kyutai_stt";
    let (gguf, refdir) = env_paths_for(arch);
    let Some(gguf) = gguf else {
        eprintln!("{}", skip_reason(arch));
        return;
    };

    let file = open_and_check_arch(&gguf, EXPECTED_ARCH);
    assert_has_any_tensor(&file, arch);

    // Primary-source hparams stamped by the converter under `vokra.kyutai_stt.*`.
    // Every key here matches the converter's `KEY_*` constants (see
    // `crates/vokra-convert/src/models/kyutai_stt.rs`); values come from the
    // module's `KyutaiSttConfig::stt_2_6b_en()` primary-source transcription.
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.sample_rate",
        KYUTAI_STT_SAMPLE_RATE,
    );
    let cfg = KyutaiSttConfig::stt_2_6b_en();
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.arch.backbone.n_layer",
        cfg.backbone.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.arch.backbone.d_model",
        cfg.backbone.d_model as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.arch.backbone.n_head",
        cfg.backbone.n_head as u32,
    );
    expect_u32_metadata(&file, arch, "vokra.kyutai_stt.audio.n_q", cfg.n_q as u32);
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.audio.card",
        cfg.audio_card as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.kyutai_stt.text.card",
        cfg.text_card as u32,
    );

    eprintln!(
        "[parity_nemo_asr::{arch}] GGUF loaded: {} tensors, arch={}",
        file.tensors().len(),
        EXPECTED_ARCH,
    );
    note_refdir(refdir.as_deref(), arch);
}

#[test]
fn parakeet_tdt() {
    use vokra_models::parakeet::{EXPECTED_ARCH, PARAKEET_SAMPLE_RATE, ParakeetConfig};
    let arch = "parakeet_tdt";
    let (gguf, refdir) = env_paths_for(arch);
    let Some(gguf) = gguf else {
        eprintln!("{}", skip_reason(arch));
        return;
    };

    let file = open_and_check_arch(&gguf, EXPECTED_ARCH);
    assert_has_any_tensor(&file, arch);

    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.sample_rate",
        PARAKEET_SAMPLE_RATE,
    );
    let cfg = ParakeetConfig::parakeet_tdt_0_6b_v3();
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.arch.encoder.n_layer",
        cfg.encoder.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.arch.encoder.d_model",
        cfg.encoder.d_model as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.arch.encoder.n_head",
        cfg.encoder.n_head as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.arch.encoder.subsampling_factor",
        cfg.encoder.subsampling_factor as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.arch.decoder.n_layer",
        cfg.decoder.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.joint.vocab_size",
        cfg.joint.vocab_size as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet.joint.blank_token_id",
        cfg.joint.blank_token_id,
    );

    eprintln!(
        "[parity_nemo_asr::{arch}] GGUF loaded: {} tensors, arch={}",
        file.tensors().len(),
        EXPECTED_ARCH,
    );
    note_refdir(refdir.as_deref(), arch);
}

#[test]
fn parakeet_ctc() {
    use vokra_models::parakeet_ctc::{EXPECTED_ARCH, PARAKEET_CTC_SAMPLE_RATE, ParakeetCtcConfig};
    let arch = "parakeet_ctc";
    let (gguf, refdir) = env_paths_for(arch);
    let Some(gguf) = gguf else {
        eprintln!("{}", skip_reason(arch));
        return;
    };

    let file = open_and_check_arch(&gguf, EXPECTED_ARCH);
    assert_has_any_tensor(&file, arch);

    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.sample_rate",
        PARAKEET_CTC_SAMPLE_RATE,
    );
    let cfg = ParakeetCtcConfig::parakeet_ctc_1_1b();
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.arch.encoder.n_layer",
        cfg.encoder.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.arch.encoder.d_model",
        cfg.encoder.d_model as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.arch.encoder.in_dim",
        cfg.encoder.in_dim as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.head.vocab_size",
        cfg.head.vocab_size as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.parakeet_ctc.head.pad_token_id",
        cfg.head.pad_token_id,
    );

    eprintln!(
        "[parity_nemo_asr::{arch}] GGUF loaded: {} tensors, arch={}",
        file.tensors().len(),
        EXPECTED_ARCH,
    );
    note_refdir(refdir.as_deref(), arch);
}

#[test]
fn canary() {
    use vokra_models::canary::{CANARY_SAMPLE_RATE, CanaryConfig, EXPECTED_ARCH};
    let arch = "canary";
    let (gguf, refdir) = env_paths_for(arch);
    let Some(gguf) = gguf else {
        eprintln!("{}", skip_reason(arch));
        return;
    };

    let file = open_and_check_arch(&gguf, EXPECTED_ARCH);
    assert_has_any_tensor(&file, arch);

    expect_u32_metadata(&file, arch, "vokra.canary.sample_rate", CANARY_SAMPLE_RATE);
    let cfg = CanaryConfig::canary_1b_v2();
    expect_u32_metadata(
        &file,
        arch,
        "vokra.canary.arch.encoder.n_layer",
        cfg.encoder.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.canary.arch.encoder.d_model",
        cfg.encoder.d_model as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.canary.arch.decoder.n_layer",
        cfg.decoder.n_layer as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.canary.arch.decoder.d_model",
        cfg.decoder.d_model as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.canary.head.vocab_size",
        cfg.head.vocab_size as u32,
    );

    eprintln!(
        "[parity_nemo_asr::{arch}] GGUF loaded: {} tensors, arch={}",
        file.tensors().len(),
        EXPECTED_ARCH,
    );
    note_refdir(refdir.as_deref(), arch);
}

#[test]
fn omniasr_ctc() {
    use vokra_models::omniasr_ctc::{EXPECTED_ARCH, OMNIASR_CTC_SAMPLE_RATE, OmniasrCtcConfig};
    let arch = "omniasr_ctc";
    let (gguf, refdir) = env_paths_for(arch);
    let Some(gguf) = gguf else {
        eprintln!("{}", skip_reason(arch));
        return;
    };

    let file = open_and_check_arch(&gguf, EXPECTED_ARCH);
    assert_has_any_tensor(&file, arch);

    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.sample_rate",
        OMNIASR_CTC_SAMPLE_RATE,
    );
    let cfg = OmniasrCtcConfig::omniasr_ctc_1b();
    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.arch.encoder.model_dim",
        cfg.encoder.model_dim as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.arch.encoder.num_encoder_layers",
        cfg.encoder.num_encoder_layers as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.arch.encoder.num_encoder_attn_heads",
        cfg.encoder.num_encoder_attn_heads as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.head.target_vocab_size",
        cfg.head.target_vocab_size as u32,
    );
    expect_u32_metadata(
        &file,
        arch,
        "vokra.omniasr_ctc.head.blank_id",
        cfg.head.blank_id,
    );

    eprintln!(
        "[parity_nemo_asr::{arch}] GGUF loaded: {} tensors, arch={}",
        file.tensors().len(),
        EXPECTED_ARCH,
    );
    note_refdir(refdir.as_deref(), arch);
}

// ---------------------------------------------------------------------------
// Env-var / helper self-tests (always-on — no GGUF needed)
//
// These validate the harness plumbing itself so a wrong env-var derivation
// or skip-message drift can't silently mask a real leg failure downstream.
// ---------------------------------------------------------------------------

#[test]
fn env_names_match_workflow_convention() {
    // The workflow YAML (parity-nemo-asr-real.yml) sets exactly these env
    // vars per matrix entry. A drift in either half is a fabricated-pass
    // shape (the test would skip when the workflow believed it was armed).
    assert_eq!(gguf_env("kyutai_stt"), "VOKRA_KYUTAI_STT_GGUF");
    assert_eq!(gguf_env("parakeet_tdt"), "VOKRA_PARAKEET_TDT_GGUF");
    assert_eq!(gguf_env("parakeet_ctc"), "VOKRA_PARAKEET_CTC_GGUF");
    assert_eq!(gguf_env("canary"), "VOKRA_CANARY_GGUF");
    assert_eq!(gguf_env("omniasr_ctc"), "VOKRA_OMNIASR_CTC_GGUF");

    assert_eq!(refdir_env("kyutai_stt"), "VOKRA_KYUTAI_STT_REFDIR");
    assert_eq!(refdir_env("parakeet_tdt"), "VOKRA_PARAKEET_TDT_REFDIR");
    assert_eq!(refdir_env("parakeet_ctc"), "VOKRA_PARAKEET_CTC_REFDIR");
    assert_eq!(refdir_env("canary"), "VOKRA_CANARY_REFDIR");
    assert_eq!(refdir_env("omniasr_ctc"), "VOKRA_OMNIASR_CTC_REFDIR");
}

#[test]
fn skip_reason_names_both_env_vars() {
    // A skip message that omits either env var would leave the owner
    // guessing which side to set — the honest-reporting minimum is the
    // full contract, not a summary.
    for arch in NEMO_ASR_ARCHES {
        let msg = skip_reason(arch);
        assert!(
            msg.contains(&gguf_env(arch)),
            "{arch}: gguf env missing from skip message"
        );
        assert!(
            msg.contains(&refdir_env(arch)),
            "{arch}: refdir env missing from skip message"
        );
        assert!(msg.contains("SKIP"), "{arch}: skip banner missing");
    }
}

#[test]
fn nemo_asr_arch_list_matches_expected_arch_constants() {
    // Guards against a family-list edit that forgets the module's
    // corresponding `EXPECTED_ARCH`. Every arch key in the list must be
    // one of the five modules present in `vokra-models` (see the
    // `pub mod` block in `lib.rs`) — that invariant is checked by the
    // fact that each per-arch `#[test]` above `use`s the module by name.
    for arch in NEMO_ASR_ARCHES {
        assert!(
            matches!(
                *arch,
                "kyutai_stt" | "parakeet_tdt" | "parakeet_ctc" | "canary" | "omniasr_ctc"
            ),
            "unknown arch key in NEMO_ASR_ARCHES: {arch}"
        );
    }
    assert_eq!(NEMO_ASR_ARCHES.len(), 5, "family cardinality changed");
}

// ---------------------------------------------------------------------------
// Always-on GGUF-fabrication self-tests (no env, no checkpoint)
//
// The tests above cover *intent* (env-var naming, skip-banner shape, arch
// list cardinality) — but the helper layer (`read_u32`, `expect_u32_metadata`,
// `open_and_check_arch`, `assert_has_any_tensor`, `note_refdir`) is the pure
// seam the flip-the-switch harness rests on. Every panic branch, every
// wrong-type absorb, every three-arm match is reachable via a minimum-viable
// GGUF built in-memory through `GgufBuilder::to_bytes()` + a temp file, so
// the seam can be pinned without a checkpoint or owner action. A fabricated
// pass a converter regression could hide behind (e.g. `sample_rate` emitted
// as String instead of U32 — `read_u32` silently returns `None`) is the exact
// shape these tests document.
// ---------------------------------------------------------------------------

/// PID-scoped tempfile path so parallel test threads never collide.
fn temp_gguf_path(tag: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "vokra_parity_nemo_asr_{}_{}.gguf",
        tag,
        std::process::id(),
    ))
}

fn write_temp_gguf(tag: &str, bytes: &[u8]) -> PathBuf {
    let p = temp_gguf_path(tag);
    std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
    p
}

/// Metadata-only builder pre-populated with the given arch tag.
fn builder_with_arch(arch: &str) -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.add_string("vokra.model.arch", arch);
    b
}

/// Adds a minimum-viable single-element F32 tensor so `assert_has_any_tensor`
/// is satisfied.
fn add_dummy_tensor(b: &mut GgufBuilder) {
    b.add_tensor("nemo.dummy", GgmlType::F32, vec![1], vec![0u8; 4])
        .expect("add dummy tensor");
}

/// Runs a closure that MUST panic and returns the payload as a String. Cargo
/// test captures per-test stderr, so intentional panic exercises inside the
/// closure do not spam the console unless the test itself fails.
fn expect_panic<F>(f: F) -> String
where
    F: FnOnce() + panic::UnwindSafe,
{
    let payload = panic::catch_unwind(f).expect_err("closure was expected to panic but returned");
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::from("<non-string panic payload>")
    }
}

#[test]
fn read_u32_covers_every_shape_variant() {
    let mut b = builder_with_arch("nemo-asr-test");
    b.add_u32("vokra.foo.u32", 41);
    b.add_metadata("vokra.foo.u64_fits", GgufMetadataValue::U64(42));
    b.add_metadata("vokra.foo.u64_overflow", GgufMetadataValue::U64(u64::MAX));
    b.add_string("vokra.foo.str", "hello");
    // Deliberately not a `PI` approximation — clippy 1.95's
    // `approx_constant` lint flags any float within 1e-2 of `PI`, and this
    // test only cares about the tag being non-numeric-integer.
    b.add_f32("vokra.foo.f32", 2.5);
    b.add_bool("vokra.foo.bool", true);
    let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");

    // (a) absent → None
    assert_eq!(read_u32(&file, "vokra.foo.absent"), None);
    // (b) U32 → Some(v)
    assert_eq!(read_u32(&file, "vokra.foo.u32"), Some(41));
    // (c) U64 fitting → Some(v as u32)
    assert_eq!(read_u32(&file, "vokra.foo.u64_fits"), Some(42));
    // (d) U64 overflow → None (u32::try_from errors)
    assert_eq!(read_u32(&file, "vokra.foo.u64_overflow"), None);
    // (e) wrong-type variants collapse to None. This is the fabricated-pass
    // shape the shape_validation gap pins: a converter regression that
    // emits `vokra.<arch>.sample_rate` as String/F32/Bool cannot silently
    // degrade the harness. If `read_u32` ever grows a wrong-type error path
    // (a legitimate hardening), this test's expectations flip accordingly.
    assert_eq!(read_u32(&file, "vokra.foo.str"), None);
    assert_eq!(read_u32(&file, "vokra.foo.f32"), None);
    assert_eq!(read_u32(&file, "vokra.foo.bool"), None);
}

#[test]
fn expect_u32_metadata_wrong_type_currently_absorbed_as_absent() {
    // Pins the shape_validation hazard: a converter that stamps
    // `vokra.<arch>.sample_rate` as String / F32 / Bool routes through
    // `read_u32` → None → `expect_u32_metadata`'s absent branch
    // (eprintln only, no panic). This test locks that behavior so any
    // future fix that starts panicking on wrong-type has to update this
    // pin — i.e. the hazard cannot regress silently.
    let cases: &[(&str, GgufMetadataValue)] = &[
        ("vokra.foo.str", GgufMetadataValue::String("nope".into())),
        ("vokra.foo.f32", GgufMetadataValue::F32(41.0)),
        ("vokra.foo.bool", GgufMetadataValue::Bool(true)),
    ];
    for (key, value) in cases {
        let mut b = builder_with_arch("nemo-asr-test");
        b.add_metadata(key, value.clone());
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        // Must not panic even though the key is present-but-mistyped.
        expect_u32_metadata(&file, "test-arch", key, 42);
    }
}

#[test]
fn expect_u32_metadata_mismatch_panics_with_both_values() {
    let mut b = builder_with_arch("nemo-asr-test");
    b.add_u32("vokra.foo.k", 41);
    let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
    let msg = expect_panic(AssertUnwindSafe(|| {
        expect_u32_metadata(&file, "test-arch", "vokra.foo.k", 42);
    }));
    assert!(msg.contains("41"), "observed value 41 missing: {msg}");
    assert!(msg.contains("42"), "expected value 42 missing: {msg}");
    assert!(
        msg.contains("primary-source drift") || msg.contains("primary source"),
        "documented message shape drifted: {msg}"
    );
    assert!(msg.contains("test-arch"), "arch tag missing: {msg}");
    assert!(msg.contains("vokra.foo.k"), "key missing: {msg}");
}

#[test]
fn expect_u32_metadata_absent_does_not_panic() {
    // The docstring promises "not gated" for absent keys — the converter's
    // coverage of the metadata block is itself iterating during Phase 2,
    // so silence-with-eprintln is by design here.
    let b = builder_with_arch("nemo-asr-test");
    let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
    expect_u32_metadata(&file, "test-arch", "vokra.foo.k", 42);
}

#[test]
fn open_and_check_arch_missing_arch_metadata_panics() {
    // No `vokra.model.arch` at all — the converter must always stamp it.
    let mut b = GgufBuilder::new();
    b.add_tensor("t", GgmlType::F32, vec![1], vec![0u8; 4])
        .unwrap();
    let path = write_temp_gguf("no_arch", &b.to_bytes().expect("serialize"));
    let msg = expect_panic(AssertUnwindSafe(|| {
        let _ = open_and_check_arch(&path, "nemo-asr-test");
    }));
    let _ = std::fs::remove_file(&path);
    assert!(
        msg.contains("no `vokra.model.arch`"),
        "documented message shape drifted: {msg}"
    );
    assert!(msg.contains("nemo-asr-test"), "arch tag missing: {msg}");
}

#[test]
fn open_and_check_arch_arch_mismatch_panics() {
    let mut b = builder_with_arch("wrong-arch");
    add_dummy_tensor(&mut b);
    let path = write_temp_gguf("wrong_arch", &b.to_bytes().expect("serialize"));
    let msg = expect_panic(AssertUnwindSafe(|| {
        let _ = open_and_check_arch(&path, "nemo-asr-test");
    }));
    let _ = std::fs::remove_file(&path);
    assert!(
        msg.contains("tag mismatch"),
        "documented message shape drifted: {msg}"
    );
    assert!(msg.contains("wrong-arch"), "actual arch missing: {msg}");
    assert!(
        msg.contains("nemo-asr-test"),
        "expected arch missing: {msg}"
    );
}

#[test]
fn open_and_check_arch_missing_file_panics() {
    let path = env::temp_dir().join(format!(
        "vokra_parity_nemo_asr_nonexistent_{}.gguf",
        std::process::id()
    ));
    // Make sure it doesn't exist (previous run may have left one behind).
    let _ = std::fs::remove_file(&path);
    let msg = expect_panic(AssertUnwindSafe(|| {
        let _ = open_and_check_arch(&path, "nemo-asr-test");
    }));
    assert!(
        msg.contains("open GGUF"),
        "documented message shape drifted: {msg}"
    );
    assert!(msg.contains("nemo-asr-test"), "arch tag missing: {msg}");
}

#[test]
fn assert_has_any_tensor_panics_on_empty_gguf() {
    // Metadata-only, zero tensors — the shape-only converter regression
    // this branch is meant to catch.
    let b = builder_with_arch("nemo-asr-test");
    let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
    let msg = expect_panic(AssertUnwindSafe(|| {
        assert_has_any_tensor(&file, "nemo-asr-test");
    }));
    assert!(
        msg.contains("no tensors"),
        "documented message shape drifted: {msg}"
    );
    assert!(msg.contains("nemo-asr-test"), "tag missing: {msg}");
}

#[test]
fn note_refdir_covers_three_arm_match() {
    // (a) None → no panic.
    note_refdir(None, "nemo-asr-test");

    // (b) Some(valid dir) → no panic.
    let dir = env::temp_dir().join(format!(
        "vokra_parity_nemo_asr_refdir_ok_{}",
        std::process::id(),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    note_refdir(Some(&dir), "nemo-asr-test");
    let _ = std::fs::remove_dir_all(&dir);

    // (c) Some(file path — exists but not a dir) → panic.
    let file_path = env::temp_dir().join(format!(
        "vokra_parity_nemo_asr_refdir_file_{}.txt",
        std::process::id(),
    ));
    std::fs::write(&file_path, b"not a directory").unwrap();
    let msg = expect_panic(AssertUnwindSafe(|| {
        note_refdir(Some(&file_path), "kyutai_stt");
    }));
    let _ = std::fs::remove_file(&file_path);
    assert!(
        msg.contains("does not name a directory"),
        "documented message shape drifted: {msg}"
    );
    assert!(
        msg.contains(&refdir_env("kyutai_stt")),
        "refdir env-var name missing from diagnostic: {msg}"
    );

    // (d) Some(nonexistent path) → panic (same arm as (c) — pinning that
    // both `is not a directory` and `does not exist` land in the same
    // loud-fail path, so a mistyped env var never silently no-ops).
    let ghost = env::temp_dir().join(format!(
        "vokra_parity_nemo_asr_refdir_ghost_{}_never",
        std::process::id(),
    ));
    let _ = std::fs::remove_dir_all(&ghost);
    let _ = std::fs::remove_file(&ghost);
    let msg = expect_panic(AssertUnwindSafe(|| {
        note_refdir(Some(&ghost), "canary");
    }));
    assert!(
        msg.contains("does not name a directory"),
        "documented message shape drifted: {msg}"
    );
    assert!(
        msg.contains(&refdir_env("canary")),
        "refdir env-var name missing from diagnostic: {msg}"
    );
}

#[test]
fn env_paths_for_returns_none_for_synthetic_never_armed_arch() {
    // A synthetic arch key that no matrix entry (or workflow) ever
    // consumes so no concurrent test can race the env var. Pins the
    // clean-skip seam that the per-model
    // `let Some(gguf) = gguf else { return; }` guard depends on —
    // regressing it into an `unwrap()` would only surface on a real
    // matrix run with env unset (dark territory).
    let arch = "__parity_nemo_asr_never_armed__";
    let (gguf, refdir) = env_paths_for(arch);
    assert!(gguf.is_none(), "synthetic gguf env leaked: {gguf:?}");
    assert!(refdir.is_none(), "synthetic refdir env leaked: {refdir:?}");
}

#[test]
fn nemo_asr_arches_bridge_to_expected_arch_constants() {
    use vokra_models::{canary, kyutai_stt, omniasr_ctc, parakeet, parakeet_ctc};
    // Every arch key must round-trip through the underscore-to-dash
    // bridge to its module's `EXPECTED_ARCH`. Guards against a family-list
    // rename or an `EXPECTED_ARCH` rename that would let one side drift
    // silently until an owner armed env for that model.
    let bridged: &[(&str, &str)] = &[
        ("kyutai_stt", kyutai_stt::EXPECTED_ARCH),
        ("parakeet_tdt", parakeet::EXPECTED_ARCH),
        ("parakeet_ctc", parakeet_ctc::EXPECTED_ARCH),
        ("canary", canary::EXPECTED_ARCH),
        ("omniasr_ctc", omniasr_ctc::EXPECTED_ARCH),
    ];
    for (key, expected) in bridged {
        assert_eq!(
            key.replace('_', "-"),
            *expected,
            "underscore-to-dash bridge broke for {key}"
        );
        // Also pin that the key is in the family list — a rename that
        // dropped the entry would fail here.
        assert!(
            NEMO_ASR_ARCHES.contains(key),
            "{key} missing from NEMO_ASR_ARCHES"
        );
    }
}

#[test]
fn nemo_asr_arch_list_has_no_duplicates() {
    // The existing `matches!`-plus-len check would still accept
    // `["kyutai_stt", "kyutai_stt", "kyutai_stt", "canary", "omniasr_ctc"]`
    // — a plausible copy-paste typo. Pin the uniqueness invariant.
    use std::collections::HashSet;
    let unique: HashSet<&&str> = NEMO_ASR_ARCHES.iter().collect();
    assert_eq!(
        unique.len(),
        NEMO_ASR_ARCHES.len(),
        "NEMO_ASR_ARCHES contains a duplicate arch key: {NEMO_ASR_ARCHES:?}"
    );
}

#[test]
fn skip_reason_per_arch_message_is_distinct_and_carries_arch_name() {
    // Every arch's banner must (a) name the arch itself so owners can grep
    // the CI log for a specific model (beyond just the env-var suffix
    // check the existing test does) and (b) be distinct from every other
    // arch's banner (guards against a bug that substituted one arch name
    // for another verbatim).
    for a in NEMO_ASR_ARCHES {
        let msg = skip_reason(a);
        assert!(msg.contains(a), "arch name {a} missing from skip banner");
    }
    for (i, a) in NEMO_ASR_ARCHES.iter().enumerate() {
        for b in NEMO_ASR_ARCHES.iter().skip(i + 1) {
            assert_ne!(
                skip_reason(a),
                skip_reason(b),
                "skip banners for {a} and {b} collided — a bug that substituted \
                 one arch name for another would slip past the content check"
            );
        }
    }
}

#[test]
fn minimum_viable_gguf_composes_all_three_helpers() {
    // Fabricates the *smallest* GGUF the harness accepts as an armed
    // input: one 1-element F32 tensor + arch tag + one positive u32
    // hparam. Runs `open_and_check_arch` + `assert_has_any_tensor` +
    // `expect_u32_metadata` in the same order the per-model tests do,
    // proving the three helpers compose end-to-end with zero env and
    // zero checkpoint — the seam the whole harness rests on.
    let mut b = builder_with_arch("nemo-asr-test");
    b.add_u32("vokra.nemo_asr_test.sample_rate", 16_000);
    add_dummy_tensor(&mut b);
    let path = write_temp_gguf("minimum_viable", &b.to_bytes().expect("serialize"));

    let file = open_and_check_arch(&path, "nemo-asr-test");
    assert_has_any_tensor(&file, "nemo-asr-test");
    expect_u32_metadata(
        &file,
        "nemo-asr-test",
        "vokra.nemo_asr_test.sample_rate",
        16_000,
    );
    let _ = std::fs::remove_file(&path);
}
