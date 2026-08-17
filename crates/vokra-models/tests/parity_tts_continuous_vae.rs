//! **tts-continuous-vae** family — flip-the-switch real-checkpoint parity harness
//! (SoTA Phase 1 wiring, 2026-07-25).
//!
//! # What this family is
//!
//! The `tts-continuous-vae` family (see `docs/tickets/sota-coverage-plan-…`)
//! groups every native Vokra target whose terminal decoding hop is a
//! **continuous VAE decoder** driven by a diffusion/flow-matching sampler
//! predicting velocity in the VAE latent space (as opposed to the vocoder-LM
//! HiFT-Chain family or the codec-LM RVQ/FSQ family). Membership at Phase 1:
//!
//! | model            | HF repo                       | license    | sampler       |
//! |------------------|-------------------------------|------------|---------------|
//! | `voxcpm2`        | `openbmb/VoxCPM2`             | Apache 2.0 | flow-matching |
//! | `vibevoice`      | `microsoft/VibeVoice-1.5B`    | MIT        | DDPM          |
//!
//! Both native modules land the config-level scaffold in this repo
//! ([`vokra_models::voxcpm2`], [`vokra_models::vibevoice`]) but the actual
//! weight-binding walk (`VoxCpm2Weights::from_gguf` / `VibeVoiceWeights::from_gguf`)
//! and the LM → sampler → VAE-decode forward chain are a follow-up wave (the
//! T29-equivalent in each module's rustdoc). This harness exists so that the
//! moment those wave-lands happen — or the moment the owner drops a converted
//! GGUF into `VOKRA_<ARCH>_GGUF` — the parity leg *fires automatically*
//! (fabricated-pass 禁止, FR-EX-08: the leg does the work it can do today and
//! flips into byte-level comparison the instant the reference dump is present).
//!
//! # Two env vars per model — the "flip-the-switch" contract
//!
//! For arch `A ∈ {voxcpm2, vibevoice}`:
//!
//! - `VOKRA_<A_UPPER>_GGUF` = path to a converted Vokra GGUF for that model,
//!   produced by `vokra-cli convert --model <a>`. Absent → the test **skips
//!   cleanly** (never a fabricated pass): the skip message names the exact
//!   env vars and the convert step so the operator can reproduce.
//! - `VOKRA_<A_UPPER>_REFDIR` = optional path to a directory of upstream
//!   Python reference dumps (stage taps / logits as raw `.f32` blobs plus a
//!   `manifest.txt`). Absent → only the shape/metadata leg runs against the
//!   GGUF (loud). Present → the byte-level comparison also fires (`atol =
//!   0.01`, NFR-QL-01). Present-but-empty is treated as "no comparable
//!   stages" and reported as such.
//!
//! # What each test proves TODAY
//!
//! The `VoxCpm2Weights::from_gguf` / `VibeVoiceWeights::from_gguf` walks land
//! in a future wave (each `mod.rs` rustdoc's T29-equivalent). Until then, on a
//! **real converted GGUF** we can — and here do — assert:
//!
//! - the file opens (GGUF v3, well-formed);
//! - `vokra.model.arch` equals the runtime constant
//!   ([`voxcpm2::EXPECTED_ARCH`] / [`vibevoice::EXPECTED_ARCH`]) — a silent
//!   mis-routing across arch families is caught here, not "on the next
//!   forward";
//! - the family-specific hparam chunk is present and every documented axis
//!   agrees with the runtime canonical constants
//!   ([`VoxCpm2Config::voxcpm_0_5b`] / [`VibeVoiceConfig::vibevoice_1_5b`]);
//! - the converter passed at least one float tensor through (a metadata-only
//!   GGUF from the "no float tensors" note in each converter's rustdoc would
//!   here surface as a loud FAIL rather than a silent "shape OK");
//! - `synthesize` refuses loudly (`VokraError::NotImplemented`) so nobody
//!   confuses a scaffold engine for a real one (FR-EX-08 pin — this is the
//!   assertion that goes AWAY the moment real weights bind).
//!
//! Once `VOKRA_<A>_REFDIR` is populated, an additional byte-level comparison
//! step fires (`assert_close` at `atol = 0.01` against the reference `.f32`
//! blobs — see [`compare_against_refdir`]). Which tensors are compared is
//! reference-dir driven: a manifest of `sha256 <name> <hex>` lines names each
//! comparable stage, and `<name>.f32` alongside is the raw blob. The harness
//! reads the manifest, resolves each named blob to its shape via the manifest
//! `<name>.shape = d0 d1 …` sidecar entry, and compares against the Vokra
//! GGUF tensor of the same name.
//!
//! Reference-dir provenance is DELIBERATELY the operator's problem: Vokra
//! does not fetch upstream weights in this harness (that is the workflow YAML
//! `.github/workflows/parity-tts-continuous-vae-real.yml`'s job). The
//! harness's contract is "GGUF here, reference here → verdict", not "download
//! anything".
//!
//! # No third-party deps
//!
//! Standard library + `vokra-core::gguf` + `vokra-models::{voxcpm2,vibevoice}`
//! only. Root `Cargo.lock` is unaffected (NFR-DS-02). The workflow guard is
//! the final `git diff --exit-code Cargo.lock` step in the YAML.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};
use vokra_models::vibevoice::{
    VIBEVOICE_ENCODER_SAMPLE_RATE, VibeVoiceConfig, VibeVoiceTts, VibeVoiceWeights,
};
use vokra_models::voxcpm2::{
    VOXCPM_ENCODER_SAMPLE_RATE, VoxCpm2Config, VoxCpm2Tts, VoxCpm2Weights,
};

/// Global FP32 parity tolerance (NFR-QL-01). Applied to the reference-dir
/// byte-level leg. A per-tensor override lands here as an explicit `match` arm
/// (the [`atol_for`] helper) with a rustdoc + ADR rationale — never a bulk
/// widening. Today no tensor requires an override.
const ATOL: f32 = 0.01;

/// The membership of the `tts-continuous-vae` family at Phase 1. Every entry
/// gets one `#[test]` below; adding an entry here means adding a matching
/// test function (no dynamic discovery so `cargo test <name>` still surfaces
/// each row).
const FAMILY: &[&str] = &["voxcpm2", "vibevoice"];

/// Env-var suffix for a model's converted GGUF path: `voxcpm2` →
/// `VOKRA_VOXCPM2_GGUF`.
fn gguf_env_var(arch: &str) -> String {
    format!("VOKRA_{}_GGUF", arch.to_ascii_uppercase().replace('-', "_"))
}

/// Env-var suffix for a model's optional reference-dump directory: `voxcpm2`
/// → `VOKRA_VOXCPM2_REFDIR`. Absent → shape/metadata leg only.
fn refdir_env_var(arch: &str) -> String {
    format!(
        "VOKRA_{}_REFDIR",
        arch.to_ascii_uppercase().replace('-', "_")
    )
}

/// Reads both env-var paths for an arch. Either or both may be `None`.
///
/// The reference-dir path is *not* filesystem-validated here (the byte-level
/// leg does that): a set-but-nonexistent `REFDIR` value is passed through so
/// the byte-level leg can surface a loud FAIL rather than a silent "no
/// comparison found".
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let gguf = std::env::var_os(gguf_env_var(arch)).map(PathBuf::from);
    let refdir = std::env::var_os(refdir_env_var(arch)).map(PathBuf::from);
    (gguf, refdir)
}

/// Builds a stderr-facing skip message that names both env vars and the
/// convert command that would flip the leg green.
///
/// The message is deliberately verbose so an operator scanning `cargo test
/// -- --nocapture` output can reproduce the flip without reading this file.
/// A "fabricated pass" would be silent skipping; every skip here is loud.
fn skip_reason(arch: &str, hf_repo: &str) -> String {
    let gguf = gguf_env_var(arch);
    let refdir = refdir_env_var(arch);
    format!(
        "[parity/{arch}] SKIP: {gguf} is unset.\n  \
         To flip this leg green:\n    \
         (1) fetch the upstream checkpoint from huggingface.co/{hf_repo},\n    \
         (2) `vokra-cli convert --model {arch} --input <safetensors> --output <out.gguf>`,\n    \
         (3) re-run with `{gguf}=<out.gguf>` set.\n  \
         Add `{refdir}=<dir>` to additionally run the byte-level reference \
         comparison (atol = {ATOL})."
    )
}

// ---------------------------------------------------------------------------
// Sanity: the family list, env var derivations, and expected-arch pins
// ---------------------------------------------------------------------------
//
// These run everywhere (CI + local, no env vars, no GGUF), and pin the
// contract that keeps the two members' env vars derivable from the family
// list. A silent typo here (adding "voxcpm-2" instead of "voxcpm2") would
// silently produce `VOKRA_VOXCPM-2_GGUF` which no operator would guess.

#[test]
fn family_membership_is_stable() {
    // Adding / renaming an arch requires a conscious edit here — a silent
    // rename would drop the corresponding `#[test]` from the roster.
    assert_eq!(FAMILY, &["voxcpm2", "vibevoice"]);
}

#[test]
fn family_arches_agree_with_runtime_expected_constants() {
    // If a runtime EXPECTED_ARCH ever drifts, this test flips FAIL before any
    // GGUF is opened — cheaper than debugging a metadata mismatch during
    // parity.
    assert_eq!(FAMILY[0], vokra_models::voxcpm2::EXPECTED_ARCH);
    assert_eq!(FAMILY[1], vokra_models::vibevoice::EXPECTED_ARCH);
}

#[test]
fn env_var_derivation_is_stable_across_family() {
    // `voxcpm2` → `VOKRA_VOXCPM2_GGUF` / `VOKRA_VOXCPM2_REFDIR`
    assert_eq!(gguf_env_var("voxcpm2"), "VOKRA_VOXCPM2_GGUF");
    assert_eq!(refdir_env_var("voxcpm2"), "VOKRA_VOXCPM2_REFDIR");
    assert_eq!(gguf_env_var("vibevoice"), "VOKRA_VIBEVOICE_GGUF");
    assert_eq!(refdir_env_var("vibevoice"), "VOKRA_VIBEVOICE_REFDIR");
}

#[test]
fn skip_reason_names_both_env_vars_and_the_convert_recipe() {
    // Belt-and-braces on the "loud skip" invariant: an operator reading the
    // stderr on a skipped CI leg must see (a) which env var is missing,
    // (b) which HF repo to fetch, (c) the convert command to run.
    for &arch in FAMILY {
        let msg = skip_reason(arch, "example/repo");
        assert!(
            msg.contains(&gguf_env_var(arch)),
            "{arch}: skip message omits GGUF env var: {msg:?}"
        );
        assert!(
            msg.contains(&refdir_env_var(arch)),
            "{arch}: skip message omits REFDIR env var: {msg:?}"
        );
        assert!(
            msg.contains("vokra-cli convert"),
            "{arch}: skip message omits convert recipe: {msg:?}"
        );
        assert!(
            msg.contains("huggingface.co/example/repo"),
            "{arch}: skip message omits HF repo: {msg:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared GGUF-side assertions (arch tag, hparam chunk, tensor count > 0)
// ---------------------------------------------------------------------------

/// Panics loudly if `key` is absent or not a string (FR-EX-08: a missing arch
/// tag is not a "maybe compatible" state — it is a hard refusal).
fn expect_string(file: &GgufFile, key: &str) -> String {
    file.get(key)
        .unwrap_or_else(|| panic!("GGUF metadata key {key} missing"))
        .as_str()
        .unwrap_or_else(|| panic!("GGUF metadata key {key} is not a string"))
        .to_owned()
}

fn expect_u64(file: &GgufFile, key: &str) -> u64 {
    file.get(key)
        .unwrap_or_else(|| panic!("GGUF metadata key {key} missing"))
        .as_u64()
        .unwrap_or_else(|| panic!("GGUF metadata key {key} is not an integer"))
}

fn expect_f64(file: &GgufFile, key: &str) -> f64 {
    file.get(key)
        .unwrap_or_else(|| panic!("GGUF metadata key {key} missing"))
        .as_f64()
        .unwrap_or_else(|| panic!("GGUF metadata key {key} is not a float"))
}

fn expect_bool(file: &GgufFile, key: &str) -> bool {
    file.get(key)
        .unwrap_or_else(|| panic!("GGUF metadata key {key} missing"))
        .as_bool()
        .unwrap_or_else(|| panic!("GGUF metadata key {key} is not a bool"))
}

/// Asserts the file's `vokra.model.arch` equals `expected_arch`. A mismatch
/// here catches a mis-routed converter (a `voxcpm2` GGUF handed to the
/// vibevoice test would silently look "close enough" without this pin).
fn assert_arch(file: &GgufFile, expected_arch: &str, ctx: &str) {
    let arch = expect_string(file, chunks::KEY_MODEL_ARCH);
    assert_eq!(
        arch, expected_arch,
        "{ctx}: vokra.model.arch = {arch:?} but expected {expected_arch:?} \
         — is this the right converted GGUF for this test?"
    );
}

/// Asserts the converter shipped at least one float tensor through (a
/// metadata-only GGUF — the "no float tensors passed through" note in each
/// converter's rustdoc — is a loud FAIL, not a silent pass).
fn assert_tensor_count_positive(file: &GgufFile, ctx: &str) {
    let n = file.tensors().len();
    assert!(
        n > 0,
        "{ctx}: converted GGUF has zero tensors — the converter probably hit \
         the 'no float tensors passed through' path (BF16 unwidened source \
         with the F32/F16 pass-through arm only). Pre-widen offline or wait \
         for the streaming BF16 pass-through path."
    );
    eprintln!("[parity/{ctx}] GGUF tensor count = {n}");
}

// ---------------------------------------------------------------------------
// Reference-dir comparator (byte-level, atol = 0.01)
// ---------------------------------------------------------------------------

/// A reference-dir manifest row: `sha256 <name> <hex>` and (optionally,
/// alongside) `<name>.shape = d0 d1 …`. Missing shape is tolerated (the
/// element count from the GGUF drives the comparison length).
struct RefStage {
    name: String,
}

/// Reads `manifest.txt` in `dir` and returns each `sha256 <name> <hex>` name
/// exactly once, in file order. Non-matching lines are ignored.
///
/// Returns `Err` if the manifest is missing or unreadable — a set-but-empty
/// `REFDIR` is not silently downgraded to "no comparison" (fabricated pass).
fn read_ref_manifest(dir: &Path) -> Result<Vec<RefStage>, String> {
    let manifest = dir.join("manifest.txt");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let mut stages = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("sha256 ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let name = parts
            .next()
            .ok_or_else(|| format!("manifest line missing name: {line:?}"))?;
        // The hex is present-but-unused here (the fixture-bytes SHA guard
        // is a workflow step; this test consumes the raw bytes directly).
        let _hex = parts
            .next()
            .ok_or_else(|| format!("manifest line missing hex: {line:?}"))?;
        stages.push(RefStage {
            name: name.to_owned(),
        });
    }
    Ok(stages)
}

/// Compares every reference `<name>.f32` blob against the GGUF tensor of the
/// same name at `atol = 0.01`. Returns the (compared, skipped) counts. A
/// blob whose corresponding tensor is absent from the GGUF is reported as
/// SKIP (with a loud eprintln) — this is deliberately non-fatal because a
/// reference dump may cover stages the current wave has not yet bound (the
/// harness is meant to flip green stage-by-stage, not all-or-nothing).
///
/// A blob whose element count disagrees with the GGUF tensor's is a hard
/// FAIL (shape mismatch is the exact "fabricated pass" the honest-parity
/// rule bans).
fn compare_against_refdir(file: &GgufFile, refdir: &Path, ctx: &str) -> (usize, usize) {
    let stages = match read_ref_manifest(refdir) {
        Ok(s) => s,
        Err(e) => panic!(
            "{ctx}: refdir {} manifest.txt unreadable: {e}. \
             Set VOKRA_<A>_REFDIR only when the reference dump is materialized \
             (fabricated pass 禁止).",
            refdir.display()
        ),
    };
    if stages.is_empty() {
        eprintln!(
            "[parity/{ctx}] refdir {} manifest.txt has no `sha256 <name> <hex>` rows \
             — no reference stages to compare against yet. Populate the reference \
             dump to enable byte-level parity.",
            refdir.display()
        );
        return (0, 0);
    }

    let mut compared = 0usize;
    let mut skipped = 0usize;
    for stage in &stages {
        let blob_path = refdir.join(format!("{}.f32", stage.name));
        let Ok(blob) = std::fs::read(&blob_path) else {
            eprintln!(
                "[parity/{ctx}] SKIP {}: {} not readable",
                stage.name,
                blob_path.display()
            );
            skipped += 1;
            continue;
        };
        if blob.len() % 4 != 0 {
            panic!(
                "{ctx}: {} is not f32-aligned ({} bytes)",
                blob_path.display(),
                blob.len()
            );
        }
        let ref_vals: Vec<f32> = blob
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let Some(_info) = file.tensor_info(&stage.name) else {
            eprintln!(
                "[parity/{ctx}] SKIP {}: not present in GGUF (weight-binding wave \
                 may not have landed this stage yet)",
                stage.name
            );
            skipped += 1;
            continue;
        };
        let got = file.tensor_f32(&stage.name).unwrap_or_else(|e| {
            panic!(
                "{ctx}: tensor {:?} present in GGUF but tensor_f32 failed: {e}",
                stage.name
            )
        });
        assert_eq!(
            got.len(),
            ref_vals.len(),
            "{ctx}: tensor {:?}: GGUF has {} elements, reference has {} — shape \
             mismatch is a hard FAIL (fabricated pass 禁止, FR-EX-08)",
            stage.name,
            got.len(),
            ref_vals.len()
        );
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (g, r)) in got.iter().zip(ref_vals.iter()).enumerate() {
            let d = (g - r).abs();
            if d > worst {
                worst = d;
                worst_i = i;
            }
        }
        eprintln!(
            "[parity/{ctx}] tensor {}: max |Δ| = {worst:.3e} over {} elems (atol {ATOL})",
            stage.name,
            got.len()
        );
        assert!(
            worst <= ATOL,
            "{ctx}: tensor {}: max |Δ| = {worst} at index {worst_i} \
             (got {} vs reference {}) exceeds atol {ATOL}",
            stage.name,
            got[worst_i],
            ref_vals[worst_i]
        );
        compared += 1;
    }
    (compared, skipped)
}

// ---------------------------------------------------------------------------
// Per-model tests — one per FAMILY entry.
// ---------------------------------------------------------------------------

/// `voxcpm2` — `openbmb/VoxCPM-0.5B` (Apache 2.0) or `openbmb/VoxCPM2` (2B,
/// Apache 2.0).
///
/// Continuous VAE (`vokra_ops::vae_continuous`) + flow-matching sampler
/// (`vokra_ops::flow_sampler`); LM is MiniCPM-4 family. **Variant-aware
/// since 2026-07-30** (spec `docs/superpowers/specs/2026-07-28-voxcpm2-2b-
/// design.md` Option C hybrid): the converter side detects the variant
/// from the safetensors payload and stamps `vokra.model.name = voxcpm2-0.5b`
/// or `voxcpm2-2b`; this test dispatches on the name string and pins
/// every documented hparam against the matching runtime canonical
/// (`VoxCpm2Config::voxcpm_0_5b` / `VoxCpm2Config::voxcpm2_2b`). A GGUF
/// carrying the *legacy* `voxcpm-0.5b` name (any pre-rename artefact) is
/// still accepted and routed to the 0.5B canonical.
///
/// Real-weight binding is a follow-up wave — until then this test proves:
///
/// * GGUF opens + `vokra.model.arch = "voxcpm2"`;
/// * every documented hparam in the `vokra.voxcpm2.*` chunk group agrees
///   with the appropriate `VoxCpm2Config` variant;
/// * for the 2B variant, `vokra.vae_continuous.sr_bin_boundaries` matches
///   `ContinuousVaeConfig::voxcpm2_2b().sr_bin_boundaries` element-wise;
/// * the converter shipped ≥1 float tensor through (no metadata-only GGUF);
/// * `VoxCpm2Tts::synthesize` refuses loudly (FR-EX-08 pin — synthesize
///   MUST NOT return a hallucinated waveform from scaffold weights).
///
/// If `VOKRA_VOXCPM2_REFDIR` is also set, the byte-level leg (`atol = 0.01`)
/// fires as a bonus.
#[test]
fn parity_voxcpm2() {
    let (Some(gguf), refdir) = env_paths_for("voxcpm2") else {
        eprintln!("{}", skip_reason("voxcpm2", "openbmb/VoxCPM2"));
        return;
    };
    let file = GgufFile::open(&gguf)
        .unwrap_or_else(|e| panic!("open VOKRA_VOXCPM2_GGUF = {}: {e}", gguf.display()));
    assert_arch(&file, vokra_models::voxcpm2::EXPECTED_ARCH, "voxcpm2");
    assert_tensor_count_positive(&file, "voxcpm2");

    // Variant dispatch from `vokra.model.name`. The converter stamps
    // `voxcpm2-0.5b` (renamed 2026-07-30 from the earlier `voxcpm-0.5b`
    // to align both variants under the arch-family prefix) or
    // `voxcpm2-2b`. Any pre-rename artefact carrying `voxcpm-0.5b` is
    // still accepted and routed to the 0.5B canonical. An unknown name
    // is a loud FAIL — never a silent guess (FR-EX-08).
    let name = expect_string(&file, chunks::KEY_MODEL_NAME);
    let (canonical, ctx_label): (VoxCpm2Config, &str) = match name.as_str() {
        "voxcpm2-0.5b" | "voxcpm-0.5b" => (VoxCpm2Config::voxcpm_0_5b(), "voxcpm2/0.5b"),
        "voxcpm2-2b" => (VoxCpm2Config::voxcpm2_2b(), "voxcpm2/2b"),
        other => panic!(
            "voxcpm2: unrecognised vokra.model.name {other:?} — expected one of \
             voxcpm2-0.5b / voxcpm2-2b / voxcpm-0.5b (pre-rename backward compat). \
             This would silently mis-shape parity — refusing (FR-EX-08)."
        ),
    };
    eprintln!("[parity/{ctx_label}] variant dispatched from vokra.model.name = {name:?}");
    // Hparam chunk agreement with the dispatched canonical release.
    // Every value below is transcribed verbatim from
    // `huggingface.co/openbmb/VoxCPM-0.5B/config.json` (0.5B canonical) or
    // `huggingface.co/openbmb/VoxCPM2/config.json` (2B canonical) via the
    // corresponding `VoxCpm2Config` factory; a converter that stamped
    // different constants would silently mis-route (FR-EX-08 catch here).
    // Top-level
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.feat_dim"),
        u64::from(canonical.feat_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.patch_size"),
        u64::from(canonical.patch_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.residual_lm_n_layer"),
        u64::from(canonical.residual_lm_n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.scalar_quantization.latent_dim"),
        u64::from(canonical.scalar_quantization_latent_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.scalar_quantization.scale"),
        u64::from(canonical.scalar_quantization_scale)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.max_length"),
        u64::from(canonical.max_length)
    );
    // LM backbone (MiniCPM-4)
    let lm = &canonical.lm;
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.hidden_dim"),
        u64::from(lm.hidden_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.n_layer"),
        u64::from(lm.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.n_head"),
        u64::from(lm.n_head)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.n_head_kv"),
        u64::from(lm.n_head_kv)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.ffn_dim"),
        u64::from(lm.ffn_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.vocab_size"),
        u64::from(lm.vocab_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.max_position_embeddings"),
        u64::from(lm.max_position_embeddings)
    );
    // f32 fields (rope_base, rms_norm_eps, scale_depth) are stored via
    // `add_f32` → checked with a strict-equality f64 view (writer widens
    // f32 → f64 losslessly).
    assert!(
        (expect_f64(&file, "vokra.voxcpm2.lm.rope_base") as f32 - lm.rope_base).abs() < 1e-3,
        "rope_base drift"
    );
    assert!(
        (expect_f64(&file, "vokra.voxcpm2.lm.rms_norm_eps") as f32 - lm.rms_norm_eps).abs() < 1e-9,
        "rms_norm_eps drift"
    );
    assert!(
        (expect_f64(&file, "vokra.voxcpm2.lm.scale_depth") as f32 - lm.scale_depth).abs() < 1e-5,
        "scale_depth drift"
    );
    assert_eq!(
        expect_bool(&file, "vokra.voxcpm2.lm.rope_scaling.longrope"),
        lm.rope_scaling_longrope
    );
    // 2B-added key: LM per-head channel width (0.5B: 64 derived, 2B: 128
    // explicit in config.json.lm_config.kv_channels).
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.lm.kv_channels"),
        u64::from(lm.kv_channels)
    );
    // Encoder
    let enc = &canonical.encoder;
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.encoder.hidden_dim"),
        u64::from(enc.hidden_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.encoder.n_layer"),
        u64::from(enc.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.encoder.kv_channels"),
        u64::from(enc.kv_channels)
    );
    // DiT + CFM sampler
    let dit = &canonical.dit;
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.dit.hidden_dim"),
        u64::from(dit.hidden_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.dit.n_layer"),
        u64::from(dit.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.voxcpm2.dit.kv_channels"),
        u64::from(dit.kv_channels)
    );
    assert_eq!(
        expect_bool(&file, "vokra.voxcpm2.dit.mean_mode"),
        dit.mean_mode
    );
    // Residual acoustic LM: depth + 2B RoPE-skipped flag.
    assert_eq!(
        expect_bool(&file, "vokra.voxcpm2.residual_lm.no_rope"),
        canonical.residual_lm_no_rope
    );

    // 2B-only: bandwidth-adaptive VAE decoder-head boundaries pin. Absent
    // for the 0.5B variant (single decoder head, key omitted).
    if matches!(ctx_label, "voxcpm2/2b") {
        // Compare element-wise against the shared VAE seam
        // (`ContinuousVaeConfig::voxcpm2_2b`).
        let vae = vokra_models::voxcpm2::ContinuousVaeConfig::voxcpm2_2b();
        let expected = vae
            .sr_bin_boundaries
            .clone()
            .expect("2B VAE seam must carry sr_bin_boundaries");
        let arr = file
            .get("vokra.vae_continuous.sr_bin_boundaries")
            .and_then(|v| match v {
                vokra_core::gguf::GgufMetadataValue::Array(a) => Some(a),
                _ => None,
            })
            .expect("2B GGUF must carry vokra.vae_continuous.sr_bin_boundaries");
        let got: Vec<u32> = arr
            .values
            .iter()
            .map(|v| match v {
                vokra_core::gguf::GgufMetadataValue::U32(x) => *x,
                other => panic!("sr_bin_boundaries: unexpected element {other:?}"),
            })
            .collect();
        assert_eq!(
            got, expected,
            "2B: sr_bin_boundaries element-wise agreement"
        );
    } else {
        // 0.5B: the key MUST be absent (single-head decoder).
        assert!(
            file.get("vokra.vae_continuous.sr_bin_boundaries").is_none(),
            "0.5B: sr_bin_boundaries key must be absent (single decoder head)"
        );
    }

    // Sample rate anchor — the VAE consumer's encoder input rate is 16 kHz
    // (upstream `audio_vae_v2.py`); a converter that shipped a different
    // rate would silently drop / duplicate frames.
    assert_eq!(VOXCPM_ENCODER_SAMPLE_RATE, 16_000);

    // FR-EX-08 pin — the scaffold engine MUST refuse `synthesize` loudly. This
    // assertion goes AWAY the moment a real-weight `from_gguf` walk lands and
    // the engine can bind real weights (at which point this test's synthesize
    // call would become a real forward + audio-bound sanity check). Cloned
    // so we do not consume the dispatched `canonical` (still needed for
    // downstream matches / prints).
    let cfg = canonical.clone();
    let weights = VoxCpm2Weights::synthesized(&cfg).expect("build voxcpm2 scaffold weights");
    let tts = VoxCpm2Tts::new(cfg, weights).expect("build voxcpm2 scaffold engine");
    let err = tts
        .synthesize("hello world")
        .expect_err("scaffold synthesize must refuse loudly");
    assert!(
        matches!(err, VokraError::NotImplemented(_)),
        "voxcpm2: scaffold synthesize returned unexpected variant {err:?}"
    );

    // Optional flip-the-switch leg — byte-level parity against the reference
    // dump if the operator wired one up.
    if let Some(refdir) = refdir {
        let (compared, skipped) = compare_against_refdir(&file, &refdir, "voxcpm2");
        eprintln!(
            "[parity/voxcpm2] reference comparison: {compared} tensors compared, {skipped} skipped"
        );
    } else {
        eprintln!(
            "[parity/voxcpm2] byte-level reference skipped: VOKRA_VOXCPM2_REFDIR unset. \
             (Shape / metadata leg passed above.)"
        );
    }
}

/// `vibevoice` — `microsoft/VibeVoice-1.5B` (MIT).
///
/// Continuous VAE + **DDPM** sampler (v-prediction, cosine β schedule);
/// LM is Qwen2 family. Real-weight binding is a follow-up wave — until then
/// this test proves:
///
/// * GGUF opens + `vokra.model.arch = "vibevoice"`;
/// * every documented hparam in the `vokra.vibevoice.*` chunk group agrees
///   with [`VibeVoiceConfig::vibevoice_1_5b`];
/// * the converter shipped ≥1 float tensor through;
/// * `VibeVoiceTts::synthesize` refuses loudly.
///
/// If `VOKRA_VIBEVOICE_REFDIR` is also set, the byte-level leg fires.
#[test]
fn parity_vibevoice() {
    let (Some(gguf), refdir) = env_paths_for("vibevoice") else {
        eprintln!("{}", skip_reason("vibevoice", "microsoft/VibeVoice-1.5B"));
        return;
    };
    let file = GgufFile::open(&gguf)
        .unwrap_or_else(|e| panic!("open VOKRA_VIBEVOICE_GGUF = {}: {e}", gguf.display()));
    assert_arch(&file, vokra_models::vibevoice::EXPECTED_ARCH, "vibevoice");
    assert_tensor_count_positive(&file, "vibevoice");

    let canonical = VibeVoiceConfig::vibevoice_1_5b();
    // Top-level shortcuts
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.acoustic_vae_dim"),
        u64::from(canonical.acoustic_vae_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.semantic_vae_dim"),
        u64::from(canonical.semantic_vae_dim)
    );
    // Decoder LM (Qwen2)
    let dec = &canonical.decoder;
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.hidden_dim"),
        u64::from(dec.hidden_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.n_layer"),
        u64::from(dec.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.n_head"),
        u64::from(dec.n_head)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.n_head_kv"),
        u64::from(dec.n_head_kv)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.ffn_dim"),
        u64::from(dec.ffn_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.vocab_size"),
        u64::from(dec.vocab_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vibevoice.decoder.max_position_embeddings"),
        u64::from(dec.max_position_embeddings)
    );
    assert!(
        (expect_f64(&file, "vokra.vibevoice.decoder.rope_base") as f32 - dec.rope_base).abs()
            < 1e-3,
        "vibevoice decoder rope_base drift"
    );
    assert!(
        (expect_f64(&file, "vokra.vibevoice.decoder.rms_norm_eps") as f32 - dec.rms_norm_eps).abs()
            < 1e-9,
        "vibevoice decoder rms_norm_eps drift"
    );

    // Frame-rate anchor: 24 kHz PCM → 7.5 Hz LM step (product of encoder
    // ratios = 3200); a drift here would silently mis-frame every synth.
    assert_eq!(VIBEVOICE_ENCODER_SAMPLE_RATE, 24_000);

    // FR-EX-08 pin — scaffold synthesize must refuse loudly.
    let cfg = VibeVoiceConfig::vibevoice_1_5b();
    let weights = VibeVoiceWeights::synthesized(&cfg).expect("build vibevoice scaffold weights");
    let tts = VibeVoiceTts::new(cfg, weights).expect("build vibevoice scaffold engine");
    let err = tts
        .synthesize("hello world")
        .expect_err("scaffold synthesize must refuse loudly");
    assert!(
        matches!(err, VokraError::NotImplemented(_)),
        "vibevoice: scaffold synthesize returned unexpected variant {err:?}"
    );

    if let Some(refdir) = refdir {
        let (compared, skipped) = compare_against_refdir(&file, &refdir, "vibevoice");
        eprintln!(
            "[parity/vibevoice] reference comparison: {compared} tensors compared, {skipped} skipped"
        );
    } else {
        eprintln!(
            "[parity/vibevoice] byte-level reference skipped: VOKRA_VIBEVOICE_REFDIR unset. \
             (Shape / metadata leg passed above.)"
        );
    }
}

// ---------------------------------------------------------------------------
// Helper coverage extensions (Scout audit 2026-07-25 — parity_tts_continuous_vae
// coverage gap fill).
//
// The 4 sanity tests above pin the always-on contract (family list, arch
// pins, env-var derivation, skip_reason substrings) and the 2 parity tests
// gate the flip-the-switch path behind a real GGUF. Neither exercises the
// two non-trivial helpers that carry `Err` and `panic!` branches designed
// to prevent fabricated passes:
//
//   * `env_paths_for` — the seam every gated test depends on to produce a
//     CLEAN skip. A regression that returned `Some(PathBuf::new())` on an
//     unset env var (or dropped the OsString → PathBuf conversion) would
//     never surface without direct coverage.
//   * `read_ref_manifest` — 2 explicit `Err` arms (missing-name /
//     missing-hex) plus a "missing manifest.txt → Err" arm that the
//     rustdoc explicitly promises "is not silently downgraded to no
//     comparison (fabricated pass)". Zero direct tests today.
//   * `compare_against_refdir` — the load-bearing `panic!(...manifest.txt
//     unreadable ... fabricated pass 禁止)` message and the empty-manifest
//     `(0, 0)` short-circuit that keeps the "flip green stage-by-stage"
//     contract honest.
//
// Every test below uses std::fs + std::env::temp_dir + `GgufBuilder`, no
// real GGUF or third-party dep (NFR-DS-02 preserved). Temp dirs are stemmed
// with PID + nanoseconds so parallel `cargo test` never collides. No env
// var is mutated at runtime — the workspace commits to `-D unsafe-code`
// and `std::env::set_var` is `unsafe` in Rust 2024, so the negative case
// for `env_paths_for` uses a namespaced arch guaranteed unset in CI.
// ---------------------------------------------------------------------------

/// Returns a fresh, empty scratch directory under `std::env::temp_dir()`.
/// Stemmed with the test label + PID + nanoseconds so parallel `cargo
/// test` (and cross-file collisions with other parity harnesses in this
/// crate) never trip over each other.
fn make_scratch_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "vokra_parity_tts_continuous_vae_{}_{}_{}",
        label,
        std::process::id(),
        nanos,
    ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    dir
}

/// Build a minimal, well-formed GGUF byte image with the requested arch
/// and one 1-element F32 tensor named `probe.f32`. Enough to satisfy
/// `GgufFile::parse` + `tensor_info("probe.f32")` in the compare-against
/// tests below. The bytes here are the same shape a real converter would
/// emit for a scaffold weight; nothing model-specific.
fn build_minimal_gguf(arch: &str, tensor_name: &str, value: f32) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, arch);
    b.add_tensor(
        tensor_name,
        GgmlType::F32,
        vec![1],
        value.to_le_bytes().to_vec(),
    )
    .expect("writer accepts a well-formed 1-elem F32 tensor");
    b.to_bytes().expect("serialize synthetic GGUF")
}

/// Extract the `&str` panic message from a `catch_unwind` payload. `panic!`
/// with a formatted `String` yields a `String` payload; the plain `&str`
/// variant is also fielded so the helper is drop-in for both.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

// --- env_paths_for -----------------------------------------------------

/// `env_paths_for` MUST return `(None, None)` for an arch whose env vars
/// are guaranteed unset. Uses a namespaced arch slug (`ttscvae_env_probe`
/// → `VOKRA_TTSCVAE_ENV_PROBE_GGUF` / `_REFDIR`) so no CI env var of that
/// name could plausibly exist. This pins the "unset → clean skip"
/// contract at the seam: a regression that returned `Some(PathBuf::new())`
/// (or swapped the tuple order) would let both `parity_voxcpm2` and
/// `parity_vibevoice` take the WRONG code path — they would try to open
/// `""` and produce a synthetic-looking failure instead of an honest
/// skip. FR-EX-08 (silent fallback banned): a fabricated skip is exactly
/// the failure mode the seam-level test rules out.
#[test]
fn env_paths_for_returns_none_when_env_unset() {
    let arch = "ttscvae_env_probe";
    // Belt-and-braces: derive the env var names the harness would query
    // and assert they are indeed unset in the current process before we
    // trust the (None, None) return as evidence of correctness.
    let gguf_key = gguf_env_var(arch);
    let refdir_key = refdir_env_var(arch);
    assert!(
        std::env::var_os(&gguf_key).is_none(),
        "test precondition failed: {gguf_key} is set in the environment — either \
         env leakage or the arch namespace clashes with an operator-set var; \
         rename the probe arch"
    );
    assert!(
        std::env::var_os(&refdir_key).is_none(),
        "test precondition failed: {refdir_key} is set in the environment"
    );

    let (gguf, refdir) = env_paths_for(arch);
    assert!(
        gguf.is_none(),
        "expected {gguf_key} unset → env_paths_for gguf slot None; got Some({gguf:?}) \
         — regression: seam returns Some on an unset env var (flip-the-switch broken)"
    );
    assert!(
        refdir.is_none(),
        "expected {refdir_key} unset → env_paths_for refdir slot None; got Some({refdir:?}) \
         — regression: asymmetric handling of the refdir arm"
    );
}

// --- read_ref_manifest ---------------------------------------------------

/// `read_ref_manifest` MUST return `Err(...)` when `manifest.txt` is
/// absent from the refdir — the rustdoc explicitly promises "a set-but-
/// empty REFDIR is not silently downgraded to no comparison (fabricated
/// pass)". Without this test, a regression that swallowed the I/O error
/// and returned `Ok(vec![])` would silently downgrade the byte-level leg
/// to a no-op for every gated model.
#[test]
fn read_ref_manifest_missing_file_returns_err() {
    let dir = make_scratch_dir("no_manifest");
    let err = match read_ref_manifest(&dir) {
        Ok(stages) => panic!(
            "read_ref_manifest MUST surface Err when manifest.txt is absent \
             (set-but-empty REFDIR is not a fabricated pass); got Ok with {} \
             stage(s)",
            stages.len()
        ),
        Err(e) => e,
    };
    // The message must at least name the manifest path so the operator
    // knows *which* refdir is malformed. Anchor the substring loosely
    // (path components + "manifest.txt") to survive path-formatting drift
    // across platforms.
    assert!(
        err.contains("manifest.txt"),
        "err message must name the manifest.txt filename; got: {err}"
    );
    // The `read <path>: <io error>` prefix is the format string the
    // implementation currently ships — a change here should be a
    // conscious edit, not a silent regression.
    assert!(
        err.starts_with("read "),
        "err message must start with the 'read <path>:' prefix so an \
         operator can grep the stderr log; got: {err}"
    );
}

/// `read_ref_manifest` MUST reject a `sha256 ` line with no name token
/// (line is `sha256 ` followed by nothing / only whitespace) with an Err
/// naming the "missing name" arm. Distinct from the missing-hex case
/// because the two `ok_or_else` arms are dead code from a coverage
/// perspective — a regression that swapped the error messages or
/// reordered the two `parts.next()` calls would silently invert operator
/// diagnostics.
#[test]
fn read_ref_manifest_line_missing_name_returns_err() {
    let dir = make_scratch_dir("missing_name");
    // "sha256 " (prefix + trailing space only) → strip_prefix returns
    // Some(""), split_whitespace yields 0 tokens, first parts.next() =
    // None → "missing name" arm fires.
    std::fs::write(dir.join("manifest.txt"), "sha256 \n").expect("write malformed manifest");
    let err = match read_ref_manifest(&dir) {
        Ok(stages) => panic!(
            "sha256-prefixed line with no name token must be a hard Err, \
             not a silent skip; got Ok with {} stage(s)",
            stages.len()
        ),
        Err(e) => e,
    };
    assert!(
        err.contains("missing name"),
        "err message must name the specific arm ('missing name') so an \
         operator can distinguish it from the missing-hex arm; got: {err}"
    );
}

/// `read_ref_manifest` MUST reject a `sha256 <name>` line with no hex
/// token (line has exactly one name token, no hex) with an Err naming
/// the "missing hex" arm. Symmetric to the missing-name test — pins the
/// second `ok_or_else` arm and asserts the message is DIFFERENT from the
/// missing-name case (operator debuggability).
#[test]
fn read_ref_manifest_line_missing_hex_returns_err() {
    let dir = make_scratch_dir("missing_hex");
    // "sha256 name_only" → strip_prefix returns Some("name_only"),
    // first parts.next() = Some("name_only"), second parts.next() =
    // None → "missing hex" arm fires.
    std::fs::write(dir.join("manifest.txt"), "sha256 name_only\n")
        .expect("write malformed manifest");
    let err = match read_ref_manifest(&dir) {
        Ok(stages) => panic!(
            "sha256-prefixed line with a name but no hex must be a hard Err, \
             not a silent skip; got Ok with {} stage(s)",
            stages.len()
        ),
        Err(e) => e,
    };
    assert!(
        err.contains("missing hex"),
        "err message must name the specific arm ('missing hex') so an \
         operator can distinguish it from the missing-name arm; got: {err}"
    );
    // Cross-check: the missing-hex arm must NOT mis-report itself as
    // missing-name (that would defeat the whole point of two arms).
    assert!(
        !err.contains("missing name"),
        "err message must not conflate the two arms; got: {err}"
    );
}

/// `read_ref_manifest` happy path: a manifest with three valid
/// `sha256 <name> <hex>` lines yields three `RefStage` entries in file
/// order with correct `.name` values. The parser has zero direct tests
/// today — a regression that swapped `parts.next()` for `parts.next_back()`
/// or accidentally called `.dedup()` would silently break byte-level
/// parity across every reference dump.
#[test]
fn read_ref_manifest_happy_path_returns_stages_in_order() {
    let dir = make_scratch_dir("happy_path");
    // Three well-formed rows, ordered so a regression that reversed
    // iteration would surface in the assertions.
    let body = "\
sha256 encoder.h0 deadbeef00000001
sha256 dit.block.0.attn.wq cafefade00000002
sha256 vae.decoder.final feedbead00000003
";
    std::fs::write(dir.join("manifest.txt"), body).expect("write happy-path manifest");
    let stages = read_ref_manifest(&dir).expect("happy-path manifest must parse");
    assert_eq!(
        stages.len(),
        3,
        "expected exactly 3 RefStage entries; got {} — parser dropped or \
         duplicated rows",
        stages.len()
    );
    // Order is load-bearing (the compare loop iterates in file order and
    // an operator eyeballing the stderr log expects the same order).
    assert_eq!(
        stages[0].name, "encoder.h0",
        "row 0 name mismatch; parser walked out of order?"
    );
    assert_eq!(
        stages[1].name, "dit.block.0.attn.wq",
        "row 1 name mismatch; parser walked out of order?"
    );
    assert_eq!(
        stages[2].name, "vae.decoder.final",
        "row 2 name mismatch; parser walked out of order?"
    );
}

/// `read_ref_manifest` MUST silently ignore lines that do NOT begin with
/// `sha256 ` — blank lines, `# comment` lines, and any other prefix. This
/// undocumented behaviour lets an operator drop a comment header (e.g.
/// `# generated by dump.py at <sha>`) into a manifest without breaking
/// the parse. If a future refactor tightened this to `return Err(...)`
/// on unknown prefixes, real reference dumps with headers would break
/// silently — this test pins the current behaviour so the refactor
/// becomes visible.
#[test]
fn read_ref_manifest_ignores_non_sha256_lines() {
    let dir = make_scratch_dir("mixed_lines");
    // Mix: comment / blank / valid / unknown-prefix / blank / valid.
    // Only the two sha256 rows should surface in the output.
    //
    // The `tools/parity/dump_voxcpm2.py` string below is SYNTHETIC
    // FIXTURE TEXT — an arbitrary comment line the parser must ignore,
    // not a citation of a real tool. No such file exists and none is
    // owed; do not "fix" this by writing one.
    let body = "\
# generated by tools/parity/dump_voxcpm2.py at commit abc123

sha256 stage.one 1111111100000001
md5    stage.other 2222222200000002
some free-form footer text

sha256 stage.two 3333333300000003
";
    std::fs::write(dir.join("manifest.txt"), body).expect("write mixed manifest");
    let stages = read_ref_manifest(&dir).expect("mixed manifest must parse (non-sha256 ignored)");
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["stage.one", "stage.two"],
        "parser must silently drop non-`sha256 ` lines and preserve the \
         file order of the surviving rows; got {names:?}"
    );
}

// --- compare_against_refdir --------------------------------------------

/// `compare_against_refdir` MUST panic — not return `(0, 0)` — when the
/// refdir has no `manifest.txt`. The panic message is load-bearing: it
/// carries BOTH "manifest.txt unreadable" and "fabricated pass 禁止"
/// substrings so an operator eyeballing the CI log is taught the rule.
/// A refactor that shortened the banner to `panic!("manifest missing")`
/// would drop the self-documenting halves and silently degrade the
/// error surface.
#[test]
fn compare_against_refdir_panics_on_missing_manifest() {
    // A minimal but well-formed GGUF so the panic fires on the manifest
    // read (the very first thing compare_against_refdir does), not on
    // some earlier tensor-access path we would have to work around.
    let gguf_bytes = build_minimal_gguf("voxcpm2", "probe.f32", 0.0);
    let file = GgufFile::parse(gguf_bytes).expect("parse synthetic gguf");
    let bad_refdir = make_scratch_dir("panic_no_manifest");
    // Deliberately do NOT write manifest.txt into `bad_refdir`.

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compare_against_refdir(&file, &bad_refdir, "voxcpm2");
    }))
    .expect_err(
        "compare_against_refdir must panic (not return (0, 0)) when the \
         refdir has no manifest.txt — the operator explicitly opted into \
         the byte-level leg by setting REFDIR",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("manifest.txt unreadable"),
        "panic message must carry the 'manifest.txt unreadable' phrase \
         so an operator knows which arm fired; got: {msg}"
    );
    assert!(
        msg.contains("fabricated pass 禁止"),
        "panic message must carry the 'fabricated pass 禁止' rule anchor \
         so the panic itself self-documents the honest-parity contract; \
         got: {msg}"
    );
    // Also verify the ctx label is included so an operator with multiple
    // parity legs can identify the failing one.
    assert!(
        msg.contains("voxcpm2"),
        "panic message must carry the ctx label so an operator with \
         multiple gated legs can identify the failing one; got: {msg}"
    );
}

/// `compare_against_refdir` MUST return `(0, 0)` (not panic) when the
/// manifest exists but contains no `sha256 ...` rows — only comments /
/// blank lines. This is the legitimate operator state where a reference
/// dump is being staged incrementally: the manifest exists but no stage
/// has been added yet. The docstring at lines 322–324 promises "the
/// harness is meant to flip green stage-by-stage, not all-or-nothing" —
/// a regression that panicked here would break that contract.
#[test]
fn compare_against_refdir_returns_zero_zero_on_empty_manifest() {
    let gguf_bytes = build_minimal_gguf("vibevoice", "probe.f32", 0.0);
    let file = GgufFile::parse(gguf_bytes).expect("parse synthetic gguf");
    let refdir = make_scratch_dir("empty_manifest");
    // Manifest exists (satisfies the Err → panic arm) but has no
    // `sha256 ` lines. Comments + blank lines only.
    let body = "\
# reference dump staging in progress
# no stages materialized yet

";
    std::fs::write(refdir.join("manifest.txt"), body).expect("write empty-stages manifest");

    let (compared, skipped) = compare_against_refdir(&file, &refdir, "vibevoice");
    assert_eq!(
        (compared, skipped),
        (0, 0),
        "empty-stages manifest must yield (0, 0) — the flip-green-stage-\
         by-stage contract requires the harness to no-op cleanly while a \
         reference dump is being materialized, not panic",
    );
}
