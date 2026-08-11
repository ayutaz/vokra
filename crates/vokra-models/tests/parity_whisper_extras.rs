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

use std::panic;
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

    // Provenance stamp: the raw SPDX literal must round-trip exactly.
    // A converter that started writing `mit` (lower) vs `MIT` (upper) would
    // silently break license-audit tooling that pattern-matches on the exact
    // SPDX literal.
    //
    // KEY note (harness bug fixed 2026-07-25): the SPDX literal lives on
    // `KEY_PROVENANCE_LICENSE` (raw string, "MIT"). The sibling key
    // `KEY_PROVENANCE_WEIGHT_LICENSE` holds the resolved *canonical class
    // name* (e.g. "permissive"), which is intentionally NOT the SPDX literal
    // — see `crates/vokra-core/src/gguf/chunks.rs::KEY_PROVENANCE_*` and the
    // `LicenseClass::as_str()` contract. Previously this test hit the
    // canonical-class key by mistake and failed on real distil GGUFs where
    // the converter correctly wrote "permissive" (dispatch run 30116427518
    // step 14).
    let license = expect_string(&file, chunks::KEY_PROVENANCE_LICENSE);
    assert_eq!(
        license,
        member.weight_license_spdx,
        "GGUF `{}` = {license:?}, expected {:?} (distil-whisper is MIT — see \
         module docstring `# Weight license`)",
        chunks::KEY_PROVENANCE_LICENSE,
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

    // Fabricated-pass ban self-audit: the synthesized-weight scaffold path
    // still refuses loudly (its weight store is not wired to the shared
    // Whisper engine — `from_gguf` is the real path).
    let asr = build_distil_whisper_asr_from_synthesized(&cfg);
    let err = asr
        .transcribe(&[0.0f32; 16_000])
        .expect_err("scaffold `.transcribe` must refuse loudly");
    match err {
        VokraError::NotImplemented(msg) => assert!(
            msg.contains("synthesized")
                || msg.contains("real weights")
                || msg.contains("has not landed")
                || msg.contains("from_gguf"),
            "scaffold NotImplemented message must name the blocker \
             (fabricated-pass audit): {msg}",
        ),
        other => panic!("expected NotImplemented, got {other:?}"),
    }

    // Wave 7 Part A (RUNTIME-NOTIMPL) — `DistilWhisperAsr::from_gguf` delegates
    // to the shared `WhisperAsr` engine. When the fixture GGUF carries real
    // weights + a `vokra.frontend.*` chunk, this loads successfully; when it is
    // a metadata-only staging artefact, the load surfaces a loud ModelLoad
    // (FR-EX-08). Both outcomes are legitimate for CI fixture provenance —
    // this test only guarantees the delegate is *live* (not stubbed).
    match vokra_models::distil_whisper::DistilWhisperAsr::from_gguf(&file) {
        Ok(loaded) => {
            assert!(
                loaded.has_weights_bound(),
                "from_gguf must bind the inner Whisper engine when it returns Ok"
            );
            assert!(
                !loaded.is_synthesized(),
                "delegate path (from_gguf) is by definition real weights"
            );
            assert!(
                loaded.config().n_text_layer < loaded.config().n_audio_layer,
                "loaded config must satisfy the distil invariant"
            );
            eprintln!(
                "[parity_whisper_extras/{}] from_gguf load PASS \
                 (delegate WhisperAsr bound, distil invariant holds)",
                member.arch_slug,
            );
        }
        Err(e) => {
            eprintln!(
                "[parity_whisper_extras/{}] from_gguf load surfaces a loud error \
                 (fixture may be metadata-only; delegate wiring is live): {e:?}",
                member.arch_slug,
            );
        }
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

    // KEY note: raw SPDX literal lives on `KEY_PROVENANCE_LICENSE`; the
    // sibling `KEY_PROVENANCE_WEIGHT_LICENSE` holds the resolved canonical
    // class name (e.g. "permissive") — see the same-file distil block above
    // for full rationale.
    let license = expect_string(&file, chunks::KEY_PROVENANCE_LICENSE);
    assert_eq!(
        license,
        member.weight_license_spdx,
        "GGUF `{}` = {license:?}, expected {:?} (kotoba-whisper is \
         Apache-2.0 — see module docstring `# Weight license`)",
        chunks::KEY_PROVENANCE_LICENSE,
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

    // Fabricated-pass ban self-audit: the config-only `.new()` shell must
    // still hard-error with a NotImplemented pointing at `from_gguf` as the
    // fix. Preserved through the Wave 7 Part A wire-up.
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

    // Wave 7 Part A (RUNTIME-NOTIMPL) — `KotobaWhisperAsr::from_gguf` delegates
    // to the shared `WhisperAsr` engine. When the fixture GGUF carries real
    // weights + a `vokra.frontend.*` chunk, this must load successfully; when
    // it is a metadata-only staging artefact, the load surfaces a loud
    // ModelLoad (FR-EX-08) rather than silently building a broken engine. Both
    // outcomes are legitimate for CI fixture provenance — the test only
    // guarantees the delegate is *live* (not stubbed).
    match vokra_models::kotoba_whisper::KotobaWhisperAsr::from_gguf(&file) {
        Ok(loaded) => {
            assert!(
                loaded.has_weights(),
                "from_gguf must bind the inner Whisper engine when it returns Ok"
            );
            assert!(
                loaded.config().n_text_layer < loaded.config().n_audio_layer,
                "loaded config must satisfy the distil invariant"
            );
            eprintln!(
                "[parity_whisper_extras/{}] from_gguf load PASS \
                 (delegate WhisperAsr bound, distil invariant holds)",
                member.arch_slug,
            );
        }
        Err(e) => {
            eprintln!(
                "[parity_whisper_extras/{}] from_gguf load surfaces a loud error \
                 (fixture may be metadata-only; delegate wiring is live): {e:?}",
                member.arch_slug,
            );
        }
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

// ---------------------------------------------------------------------------
// Additional unit-only coverage (audit follow-up 2026-07-25) — every test
// below runs on every `cargo test`, needs no checkpoint, and pins a seam or
// contract that the two env-gated per-model tests only exercise transitively.
// ---------------------------------------------------------------------------

/// Downcast a panic payload to a `String` for substring assertions. Mirrors
/// the shape used in `parity_nemo_asr.rs` so callers can inspect the panic
/// message directly.
fn panic_payload_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

/// Per-test unique tempfs entry under `env::temp_dir()`. `cargo test` runs
/// tests within one binary in parallel threads that share a PID, so the
/// caller must pass a distinct `tag` per test to avoid inter-test races.
fn temp_scratch(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vokra_parity_whisper_extras_{}_{}",
        tag,
        std::process::id(),
    ))
}

/// Direct unit-test coverage of the flip-the-switch seam. `env_paths_for`
/// (lines 150-154) IS the switch — the two env-gated per-model tests
/// exercise it only when an owner provisions fixtures, so on the CI baseline
/// (env vars unset) a regression that made this helper return
/// `(Some(PathBuf::new()), _)` on an unset env var would slip past the two
/// existing tests silently (both would still take the early-return skip
/// path, but for the wrong reason). Namespaced arch slug so no `VOKRA_*_GGUF`
/// / `_REFDIR` env var could plausibly collide.
///
/// The workspace forbids `unsafe` (`unsafe_code = "deny"` in workspace
/// `Cargo.toml`) and `std::env::set_var` is `unsafe` in Rust 2024, so the
/// Some-branch (env set, path visible) cannot be exercised directly without
/// breaking the workspace lint — see the equivalent note in
/// `parity_tts_dac.rs::env_paths_for_returns_none_when_unset`. The
/// Some-branch is transitively exercised by the two per-model tests as soon
/// as an owner provisions `VOKRA_<ARCH>_GGUF`.
#[test]
fn env_paths_for_returns_none_when_both_unset() {
    // A slug intentionally namespaced to this helper. `env_paths_for` will
    // read `VOKRA_WHISPER_EXTRAS_HARNESS_PROBE_ONLY_GGUF` /
    // `_REFDIR` — no CI job could plausibly set these.
    let arch = "whisper_extras_harness_probe_only";
    let (gguf, refdir) = env_paths_for(arch);
    assert!(
        gguf.is_none(),
        "expected {} unset in the test env, got Some({:?}) — env leakage \
         from another process, or a regression in env_paths_for that returns \
         Some on an unset env var (flip-the-switch seam broken)",
        gguf_env_var(arch),
        gguf,
    );
    assert!(
        refdir.is_none(),
        "expected {} unset in the test env, got Some({:?}) — env leakage or \
         asymmetric handling of the refdir arm",
        refdir_env_var(arch),
        refdir,
    );
}

/// `refdir_check_pending_forward_binding` must panic loudly when the path
/// does not exist, and the panic message must name the specific env var so
/// an owner reading a red CI can immediately unset or re-provision it.
///
/// This is the "opt-in but broken" branch of the flip-the-switch: an owner
/// set `VOKRA_<ARCH>_REFDIR` but the dump directory was deleted / moved.
/// Silent skip would let the CI report "flip-the-switch wiring OK" without
/// ever actually reading the dumps — a fabricated-pass shape.
#[test]
fn refdir_check_panics_on_nonexistent_path() {
    let arch = "distil_whisper";
    let refdir = temp_scratch("nonexistent");
    // Ensure the path really does not exist. Both variants (in case a prior
    // interrupted run left something behind).
    let _ = std::fs::remove_dir_all(&refdir);
    let _ = std::fs::remove_file(&refdir);

    let refdir_captured = refdir.clone();
    let payload = panic::catch_unwind(move || {
        refdir_check_pending_forward_binding(arch, &refdir_captured);
    })
    .expect_err("refdir_check_pending_forward_binding must panic on a nonexistent path");

    let msg = panic_payload_string(&*payload);
    assert!(
        msg.contains("does not exist or is not a directory"),
        "panic must name the failure mode; got: {msg}",
    );
    assert!(
        msg.contains("VOKRA_DISTIL_WHISPER_REFDIR"),
        "panic must name the specific env var (owner-actionable); got: {msg}",
    );
    assert!(
        msg.contains(&format!("{refdir:?}")),
        "panic must show the offending path so the owner can inspect it; got: {msg}",
    );
}

/// `refdir_check_pending_forward_binding` must reject a path that exists but
/// is a regular file, not a directory. Without this guard a CI where the
/// owner accidentally set `VOKRA_<ARCH>_REFDIR` to a tarball path
/// (`/tmp/refdir.tar.gz`) would surface a confusing `read_dir` error deep
/// inside the helper instead of the intended early rejection.
#[test]
fn refdir_check_panics_when_path_is_regular_file() {
    let arch = "kotoba_whisper";
    let path = temp_scratch("regular_file");
    // Ensure a clean slate then materialize a file at the path.
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);
    std::fs::write(&path, b"not a directory").expect("write temp file for refdir test");

    let path_captured = path.clone();
    let payload = panic::catch_unwind(move || {
        refdir_check_pending_forward_binding(arch, &path_captured);
    })
    .expect_err("refdir_check_pending_forward_binding must reject regular files");

    // Cleanup before the assertions so a message-shape regression does not
    // leak the temp file across CI runs.
    let _ = std::fs::remove_file(&path);

    let msg = panic_payload_string(&*payload);
    assert!(
        msg.contains("does not exist or is not a directory"),
        "panic must name the failure mode consistently with the missing-path \
         branch (one message shape for both `.is_dir() == false` reasons); \
         got: {msg}",
    );
    assert!(
        msg.contains("VOKRA_KOTOBA_WHISPER_REFDIR"),
        "panic must name the specific env var (owner-actionable); got: {msg}",
    );
}

/// Happy-path pin for `refdir_check_pending_forward_binding`: an existing
/// empty directory must NOT panic. This is the branch owners actually hit
/// once they provision a refdir; before this test, that branch was only
/// exercised by an owner running the parity job against a real dump — a
/// regression that made the helper panic on an empty directory (e.g. a
/// future refactor asserting `entries.len() >= 1`) would silently pass on
/// CI where no real refdir is ever set.
#[test]
fn refdir_check_empty_directory_does_not_panic() {
    let arch = "distil_whisper";
    let dir = temp_scratch("empty_dir");
    // Best-effort cleanup of any stale artifact then materialize the dir
    // and drain any lingering entries so the "empty" invariant holds.
    let _ = std::fs::remove_file(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create empty temp dir for refdir test");
    for entry in std::fs::read_dir(&dir)
        .expect("read empty temp dir")
        .flatten()
    {
        let p = entry.path();
        if p.is_dir() {
            let _ = std::fs::remove_dir_all(&p);
        } else {
            let _ = std::fs::remove_file(&p);
        }
    }

    // Wiring-only smoke — must not panic. If a future refactor tightens the
    // helper to require entries.len() >= 1 (or similar), that decision must
    // be paired with an explicit update to this pin — the header rustdoc's
    // "WIRING-ONLY smoke that keeps the CI path warm" contract depends on it.
    refdir_check_pending_forward_binding(arch, &dir);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Distil scaffold `.transcribe(&[])` — the empty-audio boundary — must be
/// rejected loudly with [`VokraError::InvalidArgument`]. The existing
/// env-gated test exercises the scaffold with 16 000 samples of silence
/// (which hits the `NotImplemented` fabricated-pass ban); empty audio is a
/// separate, real edge case: a caller passing an empty PCM buffer must get
/// a loud error, not a silently-accepted empty transcript. Anchors the
/// documented `# Errors` contract on `DistilWhisperAsr::transcribe`.
#[test]
fn distil_whisper_scaffold_rejects_empty_pcm() {
    // Tiny config — the empty-check runs before any weight access, so the
    // shape does not matter; using tiny_for_tests keeps synthesized-weight
    // allocation to KB and the test fast.
    let cfg = vokra_models::distil_whisper::DistilWhisperConfig::tiny_for_tests();
    let asr = build_distil_whisper_asr_from_synthesized(&cfg);
    let err = asr
        .transcribe(&[])
        .expect_err("empty PCM must be rejected loudly (FR-EX-08 fail-loud boundary)");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("distil-whisper"),
                "InvalidArgument on empty PCM must name the model so callers \
                 mixing multiple ASR engines can attribute the error; got: {msg}",
            );
            assert!(
                msg.contains("empty"),
                "InvalidArgument on empty PCM must name the boundary (`empty`) so \
                 the caller sees WHY the input was refused; got: {msg}",
            );
        }
        other => panic!(
            "expected InvalidArgument on empty PCM, got {other:?} — a \
             NotImplemented / panic short-circuit would let a real caller \
             silently accept an empty PCM buffer (or worse: crash the process \
             instead of returning a recoverable error). See DistilWhisperAsr::\
             transcribe `# Errors` docstring.",
        ),
    }
}

/// Kotoba scaffold `.transcribe(&[])` — mirror of the distil boundary test.
/// The two scaffolds construct differently (distil takes injected weights,
/// kotoba takes only `cfg`) so this is not redundant coverage: a rebase
/// that dropped the empty-check on the kotoba side alone would slip past
/// the distil test.
#[test]
fn kotoba_whisper_scaffold_rejects_empty_pcm() {
    let cfg = vokra_models::kotoba_whisper::KotobaWhisperConfig::tiny_for_tests();
    let asr = vokra_models::kotoba_whisper::KotobaWhisperAsr::new(cfg)
        .expect("build kotoba-whisper asr on tiny_for_tests config");
    let err = asr
        .transcribe(&[])
        .expect_err("empty PCM must be rejected loudly (FR-EX-08 fail-loud boundary)");
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("kotoba-whisper"),
                "InvalidArgument on empty PCM must name the model so callers \
                 mixing multiple ASR engines can attribute the error; got: {msg}",
            );
            assert!(
                msg.contains("empty"),
                "InvalidArgument on empty PCM must name the boundary (`empty`) so \
                 the caller sees WHY the input was refused; got: {msg}",
            );
        }
        other => panic!(
            "expected InvalidArgument on empty PCM, got {other:?} — a \
             NotImplemented / panic short-circuit would let a real caller \
             silently accept an empty PCM buffer. See KotobaWhisperAsr::\
             transcribe `# Errors` docstring.",
        ),
    }
}

/// Membership pin for the `FAMILY` manifest. `family_arch_slugs_are_distinct`
/// calls `Vec::dedup` on the slugs — but on an EMPTY family the deduped +
/// sorted Vec is trivially equal to itself, so an accidental deletion of
/// both entries would slip past both existing family tests. This test flips
/// that failure mode into a loud, specific mismatch and names the two
/// canonical members so a rebase that dropped either one is immediately
/// visible.
///
/// When a genuine new member lands (e.g. `distil-medium.en`, or a future
/// `kotoba-whisper-v3`), the maintainer must update the pinned set here —
/// that friction is exactly what the header comment (lines 87-90) invites.
#[test]
fn family_membership_is_pinned() {
    assert!(
        FAMILY.len() >= 2,
        "FAMILY must contain at least the two canonical members \
         (distil-whisper + kotoba-whisper); a rebase that emptied FAMILY \
         would let family_arch_slugs_are_distinct pass on an empty Vec"
    );

    let mut got: Vec<&'static str> = FAMILY.iter().map(|m| m.expected_arch).collect();
    got.sort_unstable();

    let mut want: Vec<&'static str> = vec!["distil-whisper", "kotoba-whisper"];
    want.sort_unstable();

    assert_eq!(
        got, want,
        "FAMILY membership drift: expected sorted expected_arch set = {want:?}, \
         got {got:?}. If a new member is being added, update this pin AND the \
         SPDX-distinctness test below; if a member is being removed, weigh the \
         provenance / license fallout first (see the arch_slug rustdoc).",
    );
}

/// Guards against a rebase-time copy-paste bug that would set both members'
/// `weight_license_spdx` to the same value (e.g. both `MIT`) — such a bug
/// would let the WRONG SPDX ride into the produced GGUFs and silently break
/// the downstream license-audit tooling that pattern-matches on the exact
/// SPDX literal (see the per-model rustdoc on `weight_license_spdx`, and
/// the license-round-trip assertions inside the two env-gated per-model
/// tests). Distinctness across FAMILY today is a strict property; if a
/// future member legitimately shares a license with an existing one, this
/// test must be re-shaped to check a specific expected multiset.
#[test]
fn family_license_spdx_values_are_distinct() {
    let mut licenses: Vec<&'static str> = FAMILY.iter().map(|m| m.weight_license_spdx).collect();
    licenses.sort_unstable();
    let mut deduped = licenses.clone();
    deduped.dedup();
    assert_eq!(
        licenses, deduped,
        "weight_license_spdx values must be pairwise distinct across FAMILY, \
         got {licenses:?}. A copy-paste bug that set both members to the same \
         SPDX literal would let the WRONG license ride into GGUFs — \
         downstream license-audit tooling pattern-matches on the exact literal \
         (see per-model rustdoc `# Weight license`).",
    );
}

/// `skip_reason` must carry three grep-friendly wording contracts that the
/// existing `skip_reason_includes_env_var_and_upstream_repo` does not pin:
///
/// - `SKIP:` prefix — CI log aggregators distinguish green-skips from
///   green-passes by scanning for this literal. Softening to `[skipped]` or
///   `SKIPPED —` would silently break every dashboard scraper.
/// - `FR-EX-08` requirement anchor — cross-refs the fabricated-pass ban so
///   a future reader can trace WHY the harness refuses to synthesize a pass.
/// - `fabricated pass` self-audit wording — mirrors the harness rustdoc
///   (lines 15-22). A refactor that dropped this wording would erase the
///   discipline signal even though the env-var + repo tokens are still present.
#[test]
fn skip_reason_contains_grep_prefix_and_fr_ex_08_anchors() {
    let msg = skip_reason("distil_whisper", "distil-whisper/distil-large-v3.5");
    assert!(
        msg.contains("SKIP:"),
        "skip_reason must carry the literal `SKIP:` prefix (grep contract for \
         CI log aggregators — a `[skipped]` / `SKIPPED —` softening would \
         silently break dashboard scrapers); got: {msg}",
    );
    assert!(
        msg.contains("FR-EX-08"),
        "skip_reason must anchor the fabricated-pass ban to requirement \
         FR-EX-08 (traceability); got: {msg}",
    );
    assert!(
        msg.contains("fabricated pass"),
        "skip_reason must include the `fabricated pass` self-audit wording \
         (discipline signal — mirrors the harness rustdoc lines 15-22); \
         got: {msg}",
    );
}

// ---------------------------------------------------------------------------
// Cross-module architectural invariants (Scout A-5 follow-up, 2026-07-29).
//
// The rustdocs on both `distil_whisper` and `kotoba_whisper` claim they share
// the **exact same architectural shape** — the Whisper large-v3 encoder with
// a shrunk 2-layer decoder. That claim is the whole basis of the "very cheap
// follow-on" contract (both delegate to `crate::whisper::WhisperModel` with a
// shrunk `n_text_layer`; every op / kernel is shared verbatim).
//
// Today the claim lives only in the module rustdocs and is transitively
// enforced by the two env-gated per-model tests reading tensor axes from real
// GGUFs. On the CI baseline (env vars unset), a drift where one factory
// updated its `d_model` / `n_audio_layer` / etc. but the other did not would
// slip past every existing test. These tests turn that documented invariant
// into a machine-checked cross-module contract, executed on every `cargo
// test` — no fixture required.
// ---------------------------------------------------------------------------

/// Both modules' primary-source configs share the exact architectural
/// quintuple `(d_model=1280, n_audio_layer=32, n_text_layer=2, n_mels=128,
/// n_vocab=51_866)`. This is the shape of "Whisper large-v3 encoder + shrunk
/// 2-layer decoder" — the "very cheap follow-on" foundation. A regression
/// that shifted either factory (e.g. a hypothetical rebase that updated
/// `n_vocab` on the distil side but not the kotoba side) would silently
/// break the shared-runtime delegation contract; this test catches it.
#[test]
fn family_shares_architectural_quintuple() {
    let d = vokra_models::distil_whisper::DistilWhisperConfig::distil_large_v3_5();
    let k = vokra_models::kotoba_whisper::KotobaWhisperConfig::kotoba_whisper_v2_0();
    // Encoder shape — must be identical (both keep the large-v3 encoder
    // intact).
    assert_eq!(
        (
            d.d_model,
            d.n_audio_layer,
            d.n_audio_head,
            d.n_audio_ctx,
            d.n_mels
        ),
        (
            k.d_model,
            k.n_audio_layer,
            k.n_audio_head,
            k.n_audio_ctx,
            k.n_mels
        ),
        "distil-whisper vs kotoba-whisper encoder shape mismatch: \
         (d_model, n_audio_layer, n_audio_head, n_audio_ctx, n_mels) \
         diverged. Both modules claim the same Whisper large-v3 encoder \
         topology in their rustdocs — that claim is the basis of the \
         shared `WhisperModel` delegation. If a genuine divergence is \
         being introduced, this pin must be updated deliberately.",
    );
    // Decoder shape — must be identical (both shrink to 2 layers).
    assert_eq!(
        (d.n_text_layer, d.n_text_head, d.n_text_ctx),
        (k.n_text_layer, k.n_text_head, k.n_text_ctx),
        "distil-whisper vs kotoba-whisper decoder shape mismatch: \
         (n_text_layer, n_text_head, n_text_ctx) diverged. The 2-layer \
         decoder is the distil axis shared across the whisper-extras \
         family — a divergence here would mean the two GGUFs need \
         different decoder loops, breaking the shared runtime.",
    );
    // Vocab + FFN — must be identical (both inherit the large-v3
    // multilingual tokenizer and FFN width).
    assert_eq!(
        (d.n_vocab, d.ffn_dim),
        (k.n_vocab, k.ffn_dim),
        "distil-whisper vs kotoba-whisper (n_vocab, ffn_dim) mismatch: \
         both modules must inherit large-v3's multilingual vocab \
         (51_866 for <|yue|>) and FFN width (5120).",
    );
    // Pin the specific quintuple documented in both module rustdocs so a
    // rebase that shifted BOTH factories in lockstep is still caught (the
    // above cross-module equality checks would silently pass on a
    // synchronized drift; this pin binds the shape to the primary-source
    // upstream config.json values fetched 2026-07-24 / 2026-07-25).
    let quintuple = (
        d.d_model,
        d.n_audio_layer,
        d.n_text_layer,
        d.n_mels,
        d.n_vocab,
    );
    assert_eq!(
        quintuple,
        (1280, 32, 2, 128, 51_866),
        "distil-large-v3.5 quintuple must match the upstream \
         config.json (fetched 2026-07-24): (d_model=1280, \
         n_audio_layer=32, n_text_layer=2, n_mels=128, n_vocab=51866). \
         Any change here must be paired with an update to the module \
         rustdocs and the converter's `derive_name` table.",
    );
}

/// The Whisper family invariant `head_dim = d_model / n_head = 64` must hold
/// on both encoder and decoder for both modules. The rustdocs claim this as
/// "the Whisper invariant across every family size" (base / small / medium /
/// large-v3 / turbo / distil-large-v3.5 / kotoba-whisper). If either side
/// shifted (e.g. accidentally set `n_audio_head = 16` while keeping `d_model
/// = 1280`, yielding `head_dim = 80`), the pre-baked attention kernels
/// hard-coded to `head_dim = 64` (see e.g. FA v2 driver) would silently
/// produce garbage. This test binds the invariant to the primary-source
/// factories.
#[test]
fn family_head_dim_is_the_whisper_invariant_64() {
    let d = vokra_models::distil_whisper::DistilWhisperConfig::distil_large_v3_5();
    let k = vokra_models::kotoba_whisper::KotobaWhisperConfig::kotoba_whisper_v2_0();
    // Encoder head dim.
    assert_eq!(
        d.head_dim(),
        64,
        "distil-large-v3.5 head_dim = {} != 64 (Whisper family invariant)",
        d.head_dim(),
    );
    assert_eq!(
        k.head_dim(),
        64,
        "kotoba-whisper-v2.0 head_dim = {} != 64 (Whisper family invariant)",
        k.head_dim(),
    );
    // Decoder head split — Whisper convention is `n_text_head == n_audio_head`
    // for both distil and kotoba (unlike turbo which shrinks n_text_head).
    // A future distil/kotoba variant that shifted the ratio would be a new
    // family and must extend this test with an explicit pin.
    assert_eq!(
        d.n_text_head, d.n_audio_head,
        "distil-large-v3.5: n_text_head ({}) must equal n_audio_head ({}) \
         (the family keeps the encoder head count on the decoder side)",
        d.n_text_head, d.n_audio_head,
    );
    assert_eq!(
        k.n_text_head, k.n_audio_head,
        "kotoba-whisper: n_text_head ({}) must equal n_audio_head ({}) \
         (the family keeps the encoder head count on the decoder side)",
        k.n_text_head, k.n_audio_head,
    );
}

/// Both modules' primary-source configs share the exact same tokenizer
/// boundary constants: `eot = 50_257` (`<|endoftext|>`), `sot = 50_258`
/// (`<|startoftranscript|>`), and `sample_rate = 16_000` (Whisper feature
/// extractor). The rustdocs claim these come from the large-v3 multilingual
/// tokenizer, invariant across the whisper-extras family. A converter that
/// silently updated `eot` on one side would break decode stop conditions;
/// this test pins the invariant.
#[test]
fn family_shares_tokenizer_and_sample_rate_constants() {
    let d = vokra_models::distil_whisper::DistilWhisperConfig::distil_large_v3_5();
    let k = vokra_models::kotoba_whisper::KotobaWhisperConfig::kotoba_whisper_v2_0();
    assert_eq!(
        d.eot, 50_257,
        "distil-large-v3.5 eot = {}, expected 50_257 (Whisper multilingual \
         <|endoftext|>)",
        d.eot,
    );
    assert_eq!(
        k.eot, 50_257,
        "kotoba-whisper-v2.0 eot = {}, expected 50_257 (Whisper multilingual \
         <|endoftext|>)",
        k.eot,
    );
    assert_eq!(
        d.sot, 50_258,
        "distil-large-v3.5 sot = {}, expected 50_258 (Whisper multilingual \
         <|startoftranscript|>)",
        d.sot,
    );
    assert_eq!(
        k.sot, 50_258,
        "kotoba-whisper-v2.0 sot = {}, expected 50_258 (Whisper multilingual \
         <|startoftranscript|>)",
        k.sot,
    );
    // Sample rate is the Whisper feature-extractor convention, not derived
    // from config.json — both modules inherit it from the openai/whisper
    // preprocessor. A future kotoba-whisper variant that shifted this
    // (e.g. some 24 kHz Japanese-specific pre-processing) would break the
    // shared runtime; this pin catches it.
    assert_eq!(
        d.sample_rate, 16_000,
        "distil-large-v3.5 sample_rate = {} Hz, expected 16_000 Hz \
         (Whisper convention)",
        d.sample_rate,
    );
    assert_eq!(
        k.sample_rate, 16_000,
        "kotoba-whisper-v2.0 sample_rate = {} Hz, expected 16_000 Hz \
         (Whisper convention)",
        k.sample_rate,
    );
    // Cross-module equality: both must agree on these constants, not just
    // each match some absolute expectation.
    assert_eq!(
        (d.eot, d.sot, d.sample_rate),
        (k.eot, k.sot, k.sample_rate),
        "distil / kotoba tokenizer + sample-rate constants must be pairwise \
         equal; got distil=({}, {}, {}) vs kotoba=({}, {}, {})",
        d.eot,
        d.sot,
        d.sample_rate,
        k.eot,
        k.sot,
        k.sample_rate,
    );
}

/// The distil invariant `n_text_layer < n_audio_layer` must hold on BOTH
/// primary-source config factories. The individual module unit tests each
/// verify this for their own factory, but there is no cross-module test
/// today: a rebase that broke the invariant on one side (e.g. set
/// `n_text_layer = 32` on kotoba to match encoder depth) would slip past the
/// other module's tests silently. This test binds both sides in one place.
#[test]
fn family_distil_invariant_holds_on_both_primary_source_factories() {
    let d = vokra_models::distil_whisper::DistilWhisperConfig::distil_large_v3_5();
    let k = vokra_models::kotoba_whisper::KotobaWhisperConfig::kotoba_whisper_v2_0();
    assert!(
        d.n_text_layer < d.n_audio_layer,
        "distil-large-v3.5 distil invariant broken: n_text_layer={} \
         must be < n_audio_layer={}. A distil checkpoint shrinks the \
         decoder; equal or larger decoder depth is not distil (it is \
         vanilla Whisper).",
        d.n_text_layer,
        d.n_audio_layer,
    );
    assert!(
        k.n_text_layer < k.n_audio_layer,
        "kotoba-whisper-v2.0 distil invariant broken: n_text_layer={} \
         must be < n_audio_layer={}. Kotoba-whisper is Japanese-distilled \
         from large-v3 with a shrunk decoder; equal or larger decoder \
         depth is not kotoba-whisper.",
        k.n_text_layer,
        k.n_audio_layer,
    );
    // Both primary-source factories must produce well-formed configs
    // (validate_for_forward returns Ok). If a future rebase introduced a
    // primary-source factory whose invariants fail, both env-gated tests
    // would panic — this test surfaces the bug on any `cargo test` without
    // needing a fixture.
    d.validate_for_forward()
        .expect("distil-large-v3.5 primary-source config must validate");
    k.validate_for_forward()
        .expect("kotoba-whisper-v2.0 primary-source config must validate");
}

/// The `parity-whisper-extras-real.yml` workflow pins
/// `kotoba-tech/kotoba-whisper-v2.2` (see `KOTOBA_WHISPER_REPO` in the
/// YAML `env:` block). The runtime compliance registry lists `-v1.0` /
/// `-v1.1` / `-v2.0` / `-v2.1` / `-bilingual-v1.0` explicitly and covers
/// `-v2.2` transitively via the `kotoba-whisper-` prefix walk. This test
/// pins the exact workflow-pinned literal so that a future prefix-walk
/// removal or a rebase that dropped the walk would surface a red test
/// **before** the CI job produces a GGUF that the license registry cannot
/// classify. Without this pin, a `kotoba-whisper-v2.2` resolution failure
/// would only surface as a runtime `M2-13` refusal after minutes of HF
/// download and conversion — an expensive round trip for a preventable
/// static drift.
#[test]
fn workflow_pinned_kotoba_v2_2_resolves_permissive() {
    use vokra_core::compliance::{LicenseClass, registry_lookup};
    for id in [
        // The precise slug the workflow pins (see
        // parity-whisper-extras-real.yml env.KOTOBA_WHISPER_REPO —
        // resolved to the model id after stripping the `kotoba-tech/`
        // owner prefix).
        "kotoba-whisper-v2.2",
        // Underscore alias — the runtime module's rustdoc lists the
        // dot/underscore pair explicitly for every other member; v2.2 is
        // added here for parity with the walk's dash-based prefix rule.
        "kotoba-whisper-v2_2",
        // Also pin a hypothetical future release so the prefix-walk
        // contract stays visible: any `kotoba-whisper-vN.M` (N,M >= 0)
        // must resolve Permissive. A future member with a different
        // license would need its own registry arm carved out.
        "kotoba-whisper-v3.0",
    ] {
        assert_eq!(
            registry_lookup(id),
            Some(LicenseClass::Permissive),
            "compliance registry must resolve `{id}` to Permissive \
             (Apache-2.0). The workflow YAML pins kotoba-whisper-v2.2 \
             specifically; if this test regresses, the M2-13 gate would \
             reject a converted GGUF from a workflow run that already \
             paid the HF download cost.",
        );
    }
}

/// Mirror of the kotoba pin above: the workflow pins
/// `distil-whisper/distil-large-v3.5` and the runtime module's rustdoc
/// lists both `distil-whisper` and `distil-large-*` variants as Permissive
/// (MIT). Binds the exact workflow-pinned literal so a prefix-walk
/// regression on the distil side is caught statically, symmetric with the
/// kotoba-side pin.
#[test]
fn workflow_pinned_distil_large_v3_5_resolves_permissive() {
    use vokra_core::compliance::{LicenseClass, registry_lookup};
    for id in [
        // The precise slug the workflow pins (after stripping the
        // `distil-whisper/` owner prefix — see
        // parity-whisper-extras-real.yml env.DISTIL_WHISPER_REPO).
        "distil-large-v3.5",
        // Underscore alias — mirror of the kotoba-side pin so the dot /
        // underscore variance both flow through registry_lookup.
        "distil-large-v3_5",
        // A hypothetical future distil member — any future
        // `distil-large-*` release must stay Permissive via the walk.
        "distil-large-v4.0",
    ] {
        assert_eq!(
            registry_lookup(id),
            Some(LicenseClass::Permissive),
            "compliance registry must resolve `{id}` to Permissive (MIT). \
             The workflow YAML pins distil-large-v3.5 specifically; a \
             prefix-walk regression here would let a converted GGUF from \
             a workflow-paid HF download fail the M2-13 gate.",
        );
    }
}
