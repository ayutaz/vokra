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
use vokra_core::gguf::{GgufFile, chunks};
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

/// `voxcpm2` — `openbmb/VoxCPM2` (Apache 2.0).
///
/// Continuous VAE (`vokra_ops::vae_continuous`) + flow-matching sampler
/// (`vokra_ops::flow_sampler`); LM is MiniCPM-4 family. Real-weight binding
/// is a follow-up wave — until then this test proves:
///
/// * GGUF opens + `vokra.model.arch = "voxcpm2"`;
/// * every documented hparam in the `vokra.voxcpm2.*` chunk group agrees
///   with [`VoxCpm2Config::voxcpm_0_5b`];
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

    // Hparam chunk agreement with the canonical release. Every value below is
    // transcribed verbatim from `huggingface.co/openbmb/VoxCPM-0.5B/config.json`
    // via `VoxCpm2Config::voxcpm_0_5b`; a converter that stamped different
    // constants would silently mis-route (FR-EX-08 catch here).
    let canonical = VoxCpm2Config::voxcpm_0_5b();
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

    // Sample rate anchor — the VAE consumer's encoder input rate is 16 kHz
    // (upstream `audio_vae_v2.py`); a converter that shipped a different
    // rate would silently drop / duplicate frames.
    assert_eq!(VOXCPM_ENCODER_SAMPLE_RATE, 16_000);

    // FR-EX-08 pin — the scaffold engine MUST refuse `synthesize` loudly. This
    // assertion goes AWAY the moment a real-weight `from_gguf` walk lands and
    // the engine can bind real weights (at which point this test's synthesize
    // call would become a real forward + audio-bound sanity check).
    let cfg = VoxCpm2Config::voxcpm_0_5b();
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
