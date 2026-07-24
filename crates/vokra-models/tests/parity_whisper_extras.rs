//! `whisper-extras` family — flip-the-switch real-checkpoint parity harness
//! (SoTA Phase 1, 2026-07-25).
//!
//! Covered models (both = Whisper large-v3 encoder + 2-layer decoder, same
//! shape quintuple `(d_model=1280, n_audio_layer=32, n_text_layer=2,
//! n_mels=128, vocab=51866)`; distinct upstream + license + arch tag):
//!
//! - **distil-whisper / distil-large-v3.5** (MIT / MIT) —
//!   [`vokra_models::distil_whisper::EXPECTED_ARCH`] = `"distil-whisper"`.
//! - **kotoba-tech / kotoba-whisper-v2.2** (Apache-2.0 / Apache-2.0) —
//!   [`vokra_models::kotoba_whisper::EXPECTED_ARCH`] = `"kotoba-whisper"`;
//!   architecturally identical to `-v2.0`, only distilled on a newer
//!   ReazonSpeech snapshot.
//!
//! # Gating (fabricated pass 禁止 — FR-EX-08)
//!
//! Every test is env-gated on `VOKRA_<ARCH>_GGUF` pointing at a
//! pre-converted GGUF (produced by `vokra-cli convert --model
//! {distil-whisper,kotoba-whisper}` against a real HF snapshot at a
//! pinned revision — see `.github/workflows/parity-whisper-extras-real.yml`).
//! When the env var is unset, the test prints a clear skip reason and
//! returns — **never** a silent pass, **never** a synthesized substitute.
//!
//! An optional `VOKRA_<ARCH>_REFDIR` env var opts into the flip-the-switch
//! branch: when both env vars are present, the test additionally verifies
//! that the ref directory exists and is readable, and reports honestly that
//! per-tensor / per-logit comparison against upstream python dumps is
//! **pending the native forward binding** (both modules today are
//! primary-source-transcribed scaffolds whose [`.transcribe`] returns
//! [`vokra_core::VokraError::NotImplemented`] — see the module docstrings
//! of [`vokra_models::distil_whisper`] and [`vokra_models::kotoba_whisper`]
//! for the "very cheap follow-on" path). This keeps the CI wiring in place
//! so a future `T29`-shaped weight-binding wave can flip the branch to a
//! real numerical comparison by editing this file alone.
//!
//! # What the test verifies today (GGUF only)
//!
//! 1. `vokra.model.arch` string matches the expected canonical tag
//!    (distinct from vanilla Whisper's `"whisper"` — provenance / telemetry /
//!    model-card correctness).
//! 2. Every `vokra.whisper.*` hparam the converter emits equals the
//!    primary-source config transcribed in the runtime module (a converter
//!    that started writing a different `n_text_layer` value than the runtime
//!    honors would silently break the distil axis — this test catches it).
//! 3. The distil invariant `n_text_layer < n_audio_layer` holds on both
//!    sides (converter output *and* the Rust config factory).
//! 4. At least one representative encoder + decoder tensor exists with the
//!    expected element count (byte-level sanity on the shape flow — a
//!    zero-tensor GGUF would still pass steps 1-3 but is caught here).
//! 5. `<Model>Asr::transcribe(&[0.0; ...])` returns
//!    [`vokra_core::VokraError::NotImplemented`] — proof that the harness
//!    is **not** claiming a real transcription pass while the forward is
//!    still a scaffold (fabricated-pass ban self-audit).
//!
//! # What the test explicitly does NOT do
//!
//! - It does **not** invoke `Whisper*Asr::transcribe` on real PCM and
//!   compare a transcript. The forward is not bound yet; a "pass" there
//!   today would be a hallucinated one (see the modules' docstrings for
//!   the `T29`-shaped follow-up wave that does bind real weights and
//!   delegate to [`vokra_models::whisper::WhisperModel`]).
//! - It does **not** compare per-tensor floats against upstream python
//!   dumps. There is no vokra-side forward to dump activations from yet.
//!   The `VOKRA_<ARCH>_REFDIR` env var is honored (existence check +
//!   report) so the switch can be flipped in a single edit later.
//!
//! # How to flip the switch (future wave)
//!
//! When the `distil_whisper` / `kotoba_whisper` forward binding lands
//! (mirror of the Moshi / CSM / Kyutai STT / Parakeet-CTC `T29` pattern —
//! see [`vokra_models::distil_whisper`] docstring), replace the
//! `refdir_check_pending_forward_binding` block below with a real
//! comparison (per-tensor max |Δ| against files in
//! `${VOKRA_<ARCH>_REFDIR}/…`, verdict table like `parity_kokoro`).
//! Everything else in this harness — env plumbing, GGUF sanity, invariant
//! checks, honest skip messages — stays as-is.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::GgufFile;
use vokra_core::gguf::chunks;

// ---------------------------------------------------------------------------
// Family manifest — kept close to the top so a future model landing in the
// whisper-extras family (e.g. a distil-medium.en, or an upcoming kotoba-
// whisper-v3) has one obvious place to register.
// ---------------------------------------------------------------------------

/// Canonical member of the `whisper-extras` parity family.
///
/// `arch_slug` doubles as the human name (`distil_whisper` / `kotoba_whisper`)
/// and the env-var infix (`VOKRA_DISTIL_WHISPER_GGUF` /
/// `VOKRA_KOTOBA_WHISPER_GGUF` — see [`gguf_env_var`] / [`refdir_env_var`]).
struct FamilyMember {
    /// Slug matching the module name (`distil_whisper` / `kotoba_whisper`).
    arch_slug: &'static str,
    /// The `vokra.model.arch` string the converter writes and the runtime
    /// module publishes as `EXPECTED_ARCH`.
    expected_arch: &'static str,
    /// The upstream HF repo id (record-only; the workflow YAML pins the
    /// revision — this is the label that appears in skip messages so an
    /// owner reading a green PR knows which upstream slot is being probed).
    upstream_repo: &'static str,
    /// SPDX identifier of the weight license (matches
    /// `vokra.provenance.weight_license` written by the converter).
    weight_license_spdx: &'static str,
}

/// Every member of the whisper-extras parity family.
///
/// The order is stable so `[parity] whisper-extras: running …` step-summary
/// tables (added later) can rely on it.
const FAMILY: &[FamilyMember] = &[
    FamilyMember {
        arch_slug: "distil_whisper",
        expected_arch: "distil-whisper",
        upstream_repo: "distil-whisper/distil-large-v3.5",
        weight_license_spdx: "MIT",
    },
    FamilyMember {
        arch_slug: "kotoba_whisper",
        expected_arch: "kotoba-whisper",
        upstream_repo: "kotoba-tech/kotoba-whisper-v2.2",
        weight_license_spdx: "Apache-2.0",
    },
];

// ---------------------------------------------------------------------------
// Env plumbing (task contract: env_paths_for + skip_reason helpers).
// ---------------------------------------------------------------------------

/// Env-var name for a member's converted GGUF
/// (`distil_whisper` → `VOKRA_DISTIL_WHISPER_GGUF`).
fn gguf_env_var(arch: &str) -> String {
    format!("VOKRA_{}_GGUF", arch.to_ascii_uppercase())
}

/// Env-var name for a member's optional reference-dump directory
/// (`distil_whisper` → `VOKRA_DISTIL_WHISPER_REFDIR`).
fn refdir_env_var(arch: &str) -> String {
    format!("VOKRA_{}_REFDIR", arch.to_ascii_uppercase())
}

/// Returns `(gguf_path, refdir)` for `arch`, each `None` when the
/// corresponding env var is unset. This is the task-contracted helper —
/// the tests call it once and branch on the tuple.
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let gguf = std::env::var_os(gguf_env_var(arch)).map(PathBuf::from);
    let refdir = std::env::var_os(refdir_env_var(arch)).map(PathBuf::from);
    (gguf, refdir)
}

/// Human-readable skip annotation printed to stderr (visible with
/// `--nocapture` and in CI logs) when the required env var is absent. Names
/// the env var and the upstream repo so an owner reading a green run
/// immediately knows which slot to populate to flip the switch.
fn skip_reason(arch: &str, upstream_repo: &str) -> String {
    format!(
        "[parity_whisper_extras/{arch}] SKIP: set {env} to a converted GGUF \
         (vokra-cli convert --model {kebab} --input <hf snapshot>) to run this leg. \
         Upstream: {upstream_repo}. This is a clean gated skip, not a pass \
         (fabricated pass 禁止 — FR-EX-08).",
        env = gguf_env_var(arch),
        // The CLI flag uses the kebab-case model tag from vokra-cli/src/convert.rs:
        // "distil-whisper" / "kotoba-whisper" (never underscored).
        kebab = arch.replace('_', "-"),
    )
}

// ---------------------------------------------------------------------------
// GGUF invariant helpers (shared between the two per-model tests).
// ---------------------------------------------------------------------------

/// One `vokra.whisper.*` hparam pair (key + expected `u32` value) the
/// converter writes deterministically from the checkpoint shapes.
struct HparamExpectation {
    key: &'static str,
    expected: u32,
    human_name: &'static str,
}

/// Reads a `u32`-valued metadata entry, panicking loudly (with the key name)
/// if it is missing or of the wrong type. Loud panics here are the whole
/// point: a converter that started writing `n_text_layer` as `I32` instead
/// of `U32` would silently break the runtime, so we refuse to guess.
fn expect_u32(gguf: &GgufFile, key: &str) -> u32 {
    let raw = gguf
        .get(key)
        .unwrap_or_else(|| panic!("GGUF is missing required metadata key `{key}`"));
    let widened = raw.as_u64().unwrap_or_else(|| {
        panic!(
            "GGUF metadata key `{key}` is not an unsigned int (got {:?})",
            raw.value_type()
        )
    });
    u32::try_from(widened)
        .unwrap_or_else(|_| panic!("GGUF metadata key `{key}` = {widened} does not fit into u32"))
}

/// Reads a `String`-valued metadata entry, panicking with the key name on
/// absence or wrong type.
fn expect_string<'a>(gguf: &'a GgufFile, key: &str) -> &'a str {
    let raw = gguf
        .get(key)
        .unwrap_or_else(|| panic!("GGUF is missing required metadata key `{key}`"));
    raw.as_str().unwrap_or_else(|| {
        panic!(
            "GGUF metadata key `{key}` is not a String (got {:?})",
            raw.value_type()
        )
    })
}

/// Verifies each hparam pair. On mismatch, panics with a message that names
/// the human-readable axis, the observed value, and the expected value.
fn verify_hparams(gguf: &GgufFile, pairs: &[HparamExpectation]) {
    for h in pairs {
        let got = expect_u32(gguf, h.key);
        assert_eq!(
            got, h.expected,
            "GGUF `{}` ({}) = {got}, expected {} — a converter that emits a \
             different value than the runtime module transcribes silently breaks \
             the distil axis; refusing to compare",
            h.key, h.human_name, h.expected,
        );
    }
}

/// Confirms a tensor exists at `name` and its element count equals the
/// expected product. Dtype is intentionally not pinned (F16 / F32 / BF16
/// pass-through are all valid per the converter's rules; the runtime binds
/// through the same dequant path regardless).
fn expect_tensor_elements(gguf: &GgufFile, name: &str, expected_elements: u64) {
    let info = gguf.tensor_info(name).unwrap_or_else(|| {
        panic!(
            "GGUF is missing required tensor `{name}` (a zero-tensor / \
             metadata-only GGUF would still pass metadata sanity but fail here)"
        )
    });
    let got = info
        .element_count()
        .unwrap_or_else(|e| panic!("tensor `{name}`: element_count overflow: {e:?}"));
    assert_eq!(
        got, expected_elements,
        "tensor `{name}` element_count = {got}, expected {expected_elements} \
         (dims stored innermost-first: {:?})",
        info.dimensions,
    );
}

/// Refdir plumbing: when `VOKRA_<ARCH>_REFDIR` is set, verify the directory
/// is readable and honestly report that per-tensor comparison is pending
/// the native forward binding (fabricated-pass ban — we never claim a real
/// numerical comparison happened until it actually does).
fn refdir_check_pending_forward_binding(arch: &str, refdir: &Path) {
    assert!(
        refdir.is_dir(),
        "VOKRA_{}_REFDIR = {refdir:?} does not exist or is not a directory — \
         either provision the reference dump (see the workflow YAML) or unset \
         the env var; refusing to silently skip an opt-in comparison",
        arch.to_ascii_uppercase(),
    );
    // Enumerate the directory so a fs::read_dir permission error surfaces
    // here rather than deep in a later "flip-the-switch" branch.
    let entries: Vec<_> = std::fs::read_dir(refdir)
        .unwrap_or_else(|e| panic!("read {refdir:?}: {e}"))
        .filter_map(std::result::Result::ok)
        .collect();
    eprintln!(
        "[parity_whisper_extras/{arch}] REFDIR = {refdir:?} ({} entries). \
         Per-tensor / per-logit numerical comparison is PENDING the native \
         forward binding for `{arch}` — today the runtime module is a \
         primary-source-transcribed scaffold whose `.transcribe` returns \
         VokraError::NotImplemented (see the module docstring for the T29 \
         follow-up wave). This is a WIRING-ONLY smoke that keeps the CI \
         path warm; the comparison actually fires once the forward lands.",
        entries.len(),
    );
}

// ---------------------------------------------------------------------------
// distil_whisper
// ---------------------------------------------------------------------------

/// distil-whisper flip-the-switch test. Skips cleanly when
/// `VOKRA_DISTIL_WHISPER_GGUF` is unset.
#[test]
fn parity_whisper_extras_distil_whisper() {
    let member = FAMILY
        .iter()
        .find(|m| m.arch_slug == "distil_whisper")
        .expect("family entry present");
    let (gguf_path, refdir) = env_paths_for(member.arch_slug);
    let Some(gguf_path) = gguf_path else {
        eprintln!("{}", skip_reason(member.arch_slug, member.upstream_repo));
        return;
    };

    // Primary-source config transcribed in the runtime module (each field
    // was fetched from HF and pinned in Rust — see
    // `distil_whisper::DistilWhisperConfig::distil_large_v3_5`).
    let cfg = vokra_models::distil_whisper::DistilWhisperConfig::distil_large_v3_5();
    cfg.validate_for_forward()
        .expect("primary-source distil-large-v3.5 config must be well-formed");
    assert!(
        cfg.n_text_layer < cfg.n_audio_layer,
        "distil invariant must hold on the Rust config side too \
         (n_text_layer={} vs n_audio_layer={})",
        cfg.n_text_layer,
        cfg.n_audio_layer,
    );

    // GGUF load + arch stamp.
    let file = GgufFile::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open {}: {e:?}", gguf_path.display()));
    let arch = expect_string(&file, chunks::KEY_MODEL_ARCH);
    assert_eq!(
        arch,
        member.expected_arch,
        "GGUF `{}` = {arch:?}, expected {:?} — distil-whisper GGUFs must \
         carry the `distil-whisper` arch tag (distinct from vanilla \
         Whisper's `whisper` and from kotoba-whisper's `kotoba-whisper`) \
         so runtime telemetry / logs / model cards label the model correctly",
        chunks::KEY_MODEL_ARCH,
        member.expected_arch,
    );

    // Every `vokra.whisper.*` axis the converter emits.
    let pairs = [
        HparamExpectation {
            key: "vokra.whisper.n_mels",
            expected: cfg.n_mels as u32,
            human_name: "n_mels",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_ctx",
            expected: cfg.n_audio_ctx as u32,
            human_name: "n_audio_ctx",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_state",
            expected: cfg.d_model as u32,
            human_name: "n_audio_state (= d_model)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_head",
            expected: cfg.n_audio_head as u32,
            human_name: "n_audio_head",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_layer",
            expected: cfg.n_audio_layer as u32,
            human_name: "n_audio_layer",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_ctx",
            expected: cfg.n_text_ctx as u32,
            human_name: "n_text_ctx",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_state",
            expected: cfg.d_model as u32,
            human_name: "n_text_state (= d_model)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_head",
            expected: cfg.n_text_head as u32,
            human_name: "n_text_head",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_layer",
            expected: cfg.n_text_layer as u32,
            human_name: "n_text_layer (the distil axis)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_vocab",
            expected: cfg.n_vocab as u32,
            human_name: "n_vocab",
        },
        HparamExpectation {
            key: "vokra.whisper.ffn_dim",
            expected: cfg.ffn_dim as u32,
            human_name: "ffn_dim",
        },
        HparamExpectation {
            key: "vokra.whisper.eot",
            expected: cfg.eot,
            human_name: "eot",
        },
    ];
    verify_hparams(&file, &pairs);

    // Distil invariant on the GGUF side too — a converter-side regression
    // that swapped n_text_layer and n_audio_layer would break every
    // downstream binding (KV cache, decoder loop, greedy).
    let gguf_n_audio_layer = expect_u32(&file, "vokra.whisper.n_audio_layer");
    let gguf_n_text_layer = expect_u32(&file, "vokra.whisper.n_text_layer");
    assert!(
        gguf_n_text_layer < gguf_n_audio_layer,
        "GGUF distil invariant violated: n_text_layer={gguf_n_text_layer} \
         must be < n_audio_layer={gguf_n_audio_layer}",
    );

    // Provenance stamp: license must round-trip exactly (SPDX string, not
    // a normalized alias). A converter that started writing `mit` (lower)
    // vs `MIT` (upper) would silently break license-audit tooling that
    // pattern-matches on the exact SPDX literal.
    let license = expect_string(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE);
    assert_eq!(
        license,
        member.weight_license_spdx,
        "GGUF `{}` = {license:?}, expected {:?} (distil-whisper is MIT — see \
         module docstring `# Weight license`)",
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        member.weight_license_spdx,
    );

    // Representative tensor sanity — encoder block 0 Q-projection weight
    // and decoder block 0 Q-projection weight. Element counts (both = d²)
    // catch a "zero-tensor GGUF" that would otherwise clear every metadata
    // check. Layer-0 is chosen because it exists in every distil / kotoba
    // variant (n_audio_layer >= 1, n_text_layer >= 1 is enforced by the
    // config validator).
    let d = cfg.d_model as u64;
    expect_tensor_elements(
        &file,
        "model.encoder.layers.0.self_attn.q_proj.weight",
        d * d,
    );
    expect_tensor_elements(
        &file,
        "model.decoder.layers.0.self_attn.q_proj.weight",
        d * d,
    );

    // Fabricated-pass ban self-audit: the runtime side is a scaffold whose
    // `.transcribe` must loudly refuse. If a future wave binds real weights
    // and this expectation trips, DELETE this block and replace it with the
    // real transcription comparison (the whole point of "flip-the-switch").
    // Never delete it silently — a green pass here today would mean the
    // harness fabricated a transcription result.
    let asr = build_distil_whisper_asr_from_synthesized(&cfg);
    let err = asr
        .transcribe(&[0.0f32; 16_000])
        .expect_err("scaffold `.transcribe` must refuse loudly");
    match err {
        VokraError::NotImplemented(msg) => assert!(
            msg.contains("synthesized")
                || msg.contains("real weights")
                || msg.contains("has not landed"),
            "scaffold NotImplemented message must name the blocker \
             (fabricated-pass audit): {msg}",
        ),
        other => panic!("expected NotImplemented, got {other:?}"),
    }

    // Optional flip-the-switch branch — refdir wiring only, no comparison
    // yet (see rustdoc).
    if let Some(refdir) = refdir {
        refdir_check_pending_forward_binding(member.arch_slug, &refdir);
    }

    eprintln!(
        "[parity_whisper_extras/{}] GGUF sanity + hparam parity PASS \
         (distil axis: n_text_layer={} < n_audio_layer={}). \
         Transcription-level parity is PENDING the native forward binding.",
        member.arch_slug, cfg.n_text_layer, cfg.n_audio_layer,
    );
    let _ = gguf_path; // moved into `open`; retained for a clearer test-scope trace.
}

/// Builds a `DistilWhisperAsr` on synthesized weights of `cfg` shape — the
/// only path today that lets the test invoke `.transcribe` and prove the
/// NotImplemented refusal path is live.
fn build_distil_whisper_asr_from_synthesized(
    cfg: &vokra_models::distil_whisper::DistilWhisperConfig,
) -> vokra_models::distil_whisper::DistilWhisperAsr {
    // `cfg` is the real primary-source shape (large-v3.5); synthesizing
    // weights against it is fine — we never invoke the forward, only the
    // guarded `.transcribe` refusal path. Seed is arbitrary but fixed for
    // determinism.
    let w = vokra_models::distil_whisper::DistilWhisperWeights::synthesized(cfg, 0x5B_C0_D3_00)
        .expect("synthesized distil-whisper weights");
    vokra_models::distil_whisper::DistilWhisperAsr::new(cfg.clone(), w)
        .expect("build distil-whisper asr on primary-source shape")
}

// ---------------------------------------------------------------------------
// kotoba_whisper
// ---------------------------------------------------------------------------

/// kotoba-whisper flip-the-switch test. Skips cleanly when
/// `VOKRA_KOTOBA_WHISPER_GGUF` is unset.
///
/// The task pins `kotoba-tech/kotoba-whisper-v2.2`, whose architectural
/// quintuple is identical to `-v2.0` (both distilled from Whisper large-v3
/// with a 2-layer decoder). The runtime module factory
/// `KotobaWhisperConfig::kotoba_whisper_v2_0` is therefore the correct
/// ground truth for v2.2 as well (config values verified against
/// `huggingface.co/kotoba-tech/kotoba-whisper-v2.2/raw/main/config.json`,
/// fetched 2026-07-25).
#[test]
fn parity_whisper_extras_kotoba_whisper() {
    let member = FAMILY
        .iter()
        .find(|m| m.arch_slug == "kotoba_whisper")
        .expect("family entry present");
    let (gguf_path, refdir) = env_paths_for(member.arch_slug);
    let Some(gguf_path) = gguf_path else {
        eprintln!("{}", skip_reason(member.arch_slug, member.upstream_repo));
        return;
    };

    let cfg = vokra_models::kotoba_whisper::KotobaWhisperConfig::kotoba_whisper_v2_0();
    cfg.validate_for_forward()
        .expect("primary-source kotoba-whisper v2.x config must be well-formed");
    assert!(
        cfg.n_text_layer < cfg.n_audio_layer,
        "distil invariant must hold on the Rust config side too \
         (n_text_layer={} vs n_audio_layer={})",
        cfg.n_text_layer,
        cfg.n_audio_layer,
    );

    let file = GgufFile::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open {}: {e:?}", gguf_path.display()));

    let arch = expect_string(&file, chunks::KEY_MODEL_ARCH);
    assert_eq!(
        arch,
        member.expected_arch,
        "GGUF `{}` = {arch:?}, expected {:?} — kotoba-whisper GGUFs must \
         carry the `kotoba-whisper` arch tag (distinct from vanilla \
         Whisper's `whisper` and from distil-whisper's `distil-whisper`) \
         so runtime telemetry / logs / model cards label the model correctly",
        chunks::KEY_MODEL_ARCH,
        member.expected_arch,
    );

    let pairs = [
        HparamExpectation {
            key: "vokra.whisper.n_mels",
            expected: cfg.n_mels as u32,
            human_name: "n_mels",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_ctx",
            expected: cfg.n_audio_ctx as u32,
            human_name: "n_audio_ctx",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_state",
            expected: cfg.d_model as u32,
            human_name: "n_audio_state (= d_model)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_head",
            expected: cfg.n_audio_head as u32,
            human_name: "n_audio_head",
        },
        HparamExpectation {
            key: "vokra.whisper.n_audio_layer",
            expected: cfg.n_audio_layer as u32,
            human_name: "n_audio_layer",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_ctx",
            expected: cfg.n_text_ctx as u32,
            human_name: "n_text_ctx",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_state",
            expected: cfg.d_model as u32,
            human_name: "n_text_state (= d_model)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_head",
            expected: cfg.n_text_head as u32,
            human_name: "n_text_head",
        },
        HparamExpectation {
            key: "vokra.whisper.n_text_layer",
            expected: cfg.n_text_layer as u32,
            human_name: "n_text_layer (JA-ASR-2 axis)",
        },
        HparamExpectation {
            key: "vokra.whisper.n_vocab",
            expected: cfg.n_vocab as u32,
            human_name: "n_vocab",
        },
        HparamExpectation {
            key: "vokra.whisper.ffn_dim",
            expected: cfg.ffn_dim as u32,
            human_name: "ffn_dim",
        },
        HparamExpectation {
            key: "vokra.whisper.eot",
            expected: cfg.eot,
            human_name: "eot",
        },
    ];
    verify_hparams(&file, &pairs);

    let gguf_n_audio_layer = expect_u32(&file, "vokra.whisper.n_audio_layer");
    let gguf_n_text_layer = expect_u32(&file, "vokra.whisper.n_text_layer");
    assert!(
        gguf_n_text_layer < gguf_n_audio_layer,
        "GGUF distil invariant violated: n_text_layer={gguf_n_text_layer} \
         must be < n_audio_layer={gguf_n_audio_layer}",
    );

    let license = expect_string(&file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE);
    assert_eq!(
        license,
        member.weight_license_spdx,
        "GGUF `{}` = {license:?}, expected {:?} (kotoba-whisper is \
         Apache-2.0 — see module docstring `# Weight license`)",
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        member.weight_license_spdx,
    );

    let d = cfg.d_model as u64;
    expect_tensor_elements(
        &file,
        "model.encoder.layers.0.self_attn.q_proj.weight",
        d * d,
    );
    expect_tensor_elements(
        &file,
        "model.decoder.layers.0.self_attn.q_proj.weight",
        d * d,
    );

    // Fabricated-pass ban self-audit (same posture as distil_whisper —
    // see that test for the deletion contract when the forward lands).
    let asr = vokra_models::kotoba_whisper::KotobaWhisperAsr::new(cfg.clone())
        .expect("build kotoba-whisper asr on primary-source config");
    let err = asr
        .transcribe(&[0.0f32; 16_000])
        .expect_err("scaffold `.transcribe` must refuse loudly");
    match err {
        VokraError::NotImplemented(msg) => assert!(
            msg.contains("kotoba-whisper") || msg.contains("has not landed"),
            "scaffold NotImplemented message must name the blocker \
             (fabricated-pass audit): {msg}",
        ),
        other => panic!("expected NotImplemented, got {other:?}"),
    }

    if let Some(refdir) = refdir {
        refdir_check_pending_forward_binding(member.arch_slug, &refdir);
    }

    eprintln!(
        "[parity_whisper_extras/{}] GGUF sanity + hparam parity PASS \
         (distil axis: n_text_layer={} < n_audio_layer={}). \
         Transcription-level parity is PENDING the native forward binding.",
        member.arch_slug, cfg.n_text_layer, cfg.n_audio_layer,
    );
    let _ = gguf_path;
}

// ---------------------------------------------------------------------------
// Family invariants (fixture-free — always runs, catches manifest drift).
// ---------------------------------------------------------------------------

/// Family manifest smoke: the two arch slugs remain distinct, both from
/// each other and from vanilla Whisper's `"whisper"`. Drift here would mean
/// two different upstream models landing under the same `vokra.model.arch`
/// tag, silently breaking provenance / license attribution.
#[test]
fn family_arch_slugs_are_distinct() {
    let mut slugs: Vec<&'static str> = FAMILY.iter().map(|m| m.expected_arch).collect();
    slugs.sort_unstable();
    let mut deduped = slugs.clone();
    deduped.dedup();
    assert_eq!(
        slugs, deduped,
        "expected_arch tags must be pairwise distinct within the family"
    );
    for m in FAMILY {
        assert_ne!(
            m.expected_arch, "whisper",
            "`{}` must not collide with vanilla Whisper's `whisper` arch tag",
            m.arch_slug,
        );
    }
}

/// Family manifest ↔ runtime module cross-check: each expected arch string
/// must equal the module's own `EXPECTED_ARCH` constant. A rename in one
/// place without the other would silently break provenance stamping.
#[test]
fn family_expected_arch_matches_runtime_modules() {
    let distil = FAMILY
        .iter()
        .find(|m| m.arch_slug == "distil_whisper")
        .expect("distil_whisper in family");
    assert_eq!(
        distil.expected_arch,
        vokra_models::distil_whisper::EXPECTED_ARCH,
        "distil_whisper family manifest disagrees with runtime EXPECTED_ARCH",
    );

    let kotoba = FAMILY
        .iter()
        .find(|m| m.arch_slug == "kotoba_whisper")
        .expect("kotoba_whisper in family");
    assert_eq!(
        kotoba.expected_arch,
        vokra_models::kotoba_whisper::EXPECTED_ARCH,
        "kotoba_whisper family manifest disagrees with runtime EXPECTED_ARCH",
    );
}

/// The env-var naming helper must produce the exact strings the workflow
/// YAML sets (`VOKRA_DISTIL_WHISPER_GGUF` / `VOKRA_KOTOBA_WHISPER_GGUF`
/// and the corresponding `_REFDIR` pair). Drift between this file and the
/// YAML would mean the CI sets an env var that no test reads.
#[test]
fn env_var_naming_matches_workflow() {
    assert_eq!(gguf_env_var("distil_whisper"), "VOKRA_DISTIL_WHISPER_GGUF");
    assert_eq!(gguf_env_var("kotoba_whisper"), "VOKRA_KOTOBA_WHISPER_GGUF");
    assert_eq!(
        refdir_env_var("distil_whisper"),
        "VOKRA_DISTIL_WHISPER_REFDIR"
    );
    assert_eq!(
        refdir_env_var("kotoba_whisper"),
        "VOKRA_KOTOBA_WHISPER_REFDIR"
    );
}

/// `skip_reason` includes the env var name and upstream repo — the two
/// pieces of information an owner needs to flip a green skip into a real
/// run.
#[test]
fn skip_reason_includes_env_var_and_upstream_repo() {
    for m in FAMILY {
        let msg = skip_reason(m.arch_slug, m.upstream_repo);
        assert!(
            msg.contains(&gguf_env_var(m.arch_slug)),
            "skip_reason must name the env var: {msg}",
        );
        assert!(
            msg.contains(m.upstream_repo),
            "skip_reason must name the upstream repo: {msg}",
        );
        // The kebab-case CLI flag must appear so an owner can copy-paste
        // the reproduction command.
        assert!(
            msg.contains(&m.arch_slug.replace('_', "-")),
            "skip_reason must name the kebab-case model tag: {msg}",
        );
    }
}
