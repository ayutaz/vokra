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
use std::path::{Path, PathBuf};

use vokra_core::gguf::{GgufFile, GgufMetadataValue};

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
