//! **tts-japanese** family — flip-the-switch real-checkpoint parity harness
//! (SoTA Phase 1 wiring, 2026-07-25).
//!
//! # What this family is
//!
//! The `tts-japanese` family (see `docs/tickets/sota-coverage-plan-…`)
//! groups every native Vokra target whose primary use case is
//! **Japanese-locale TTS** and whose runtime module lives outside the
//! shared piper-plus / VITS-family plumbing. Membership at Phase 1:
//!
//! | model      | HF repo                         | license (code / weight) | note                                            |
//! |------------|---------------------------------|-------------------------|-------------------------------------------------|
//! | `irodori`  | `Aratako/Irodori-TTS-500M-v3`   | MIT / MIT               | Rectified-Flow DiT + Semantic-DACVAE 32-d, 48 k |
//! | `vits_ja`  | `espnet/kan-bayashi_jsut_vits`  | Apache-2.0 / restricted | arch-only parity; JSUT terms forbid weight r-d  |
//!
//! **HF slug correction — irodori.** The SoTA plan text originally named
//! `Irodori-tech/Irodori-TTS-500M-v3`; that slug returns HTTP 401 on
//! `huggingface.co/api/models/Irodori-tech/Irodori-TTS-500M-v3`. The
//! actual public HF repo is `Aratako/Irodori-TTS-500M-v3` — confirmed
//! 2026-07-25 with `curl huggingface.co/api/models/Aratako/Irodori-TTS-500M-v3`
//! returning `{ "license": "mit", "pipeline_tag": "text-to-speech", … }`
//! and matching the runtime module's docstring
//! (`crates/vokra-models/src/irodori/mod.rs` references
//! `huggingface.co/Aratako/Irodori-TTS-500M-v3` verbatim). The workflow
//! YAML pins the corrected slug (CLAUDE.md「ハルシネーション厳禁」).
//!
//! **Arch-only posture — vits_ja.** The `espnet/kan-bayashi_jsut_vits`
//! HF mirror returns HTTP 401 (gated / removed from public listing;
//! `www-authenticate: Bearer` on `huggingface.co/espnet/kan-bayashi_jsut_vits`,
//! confirmed 2026-07-25) and, independently, **the JSUT corpus terms
//! forbid re-distribution of the trained weight**
//! (`sites.google.com/site/shinnosuketakamichi/publication/jsut`,
//! Redistribution is not permitted). The workflow therefore never
//! attempts an HF snapshot download for this row — the `vits_ja` leg is
//! designed as **operator-provisioned only**: an operator with a
//! locally-produced GGUF (e.g. trained on a permissive corpus, or
//! obtained under the JSUT terms and used in place) sets
//! `VOKRA_VITS_JA_GGUF=/path/to/vits-ja.gguf` and the harness fires.
//! Absent that env var the leg **honest-skips loudly** (never a
//! fabricated pass, FR-EX-08 — the skip message names the reason and
//! the env var).
//!
//! # Two env vars per model — the "flip-the-switch" contract
//!
//! For arch `A ∈ {irodori, vits_ja}`:
//!
//! - `VOKRA_<A_UPPER>_GGUF` = path to a converted Vokra GGUF for that
//!   model, produced by `vokra-cli convert --model <slug>` (slugs are
//!   `irodori` / `vits-ja`, per the CLI's `ModelKind` map in
//!   `crates/vokra-cli/src/convert.rs`). Absent → the test **skips
//!   cleanly** (never a fabricated pass): the skip message names the
//!   exact env vars and the convert step so the operator can reproduce.
//! - `VOKRA_<A_UPPER>_REFDIR` = optional path to a directory of upstream
//!   Python reference dumps (stage taps / logits as raw `.f32` blobs
//!   plus a `manifest.txt`). Absent → only the shape / metadata leg
//!   runs against the GGUF (loud). Present → the byte-level comparison
//!   also fires (`atol = 0.01`, NFR-QL-01). Present-but-empty is treated
//!   as "no comparable stages" and reported as such.
//!
//! # What each test proves TODAY
//!
//! Both runtime modules (`crate::irodori`, `crate::vits_ja`) are
//! primary-source-transcribed SCAFFOLDS: their `synthesize` returns
//! [`VokraError::NotImplemented`] until a T29-shaped follow-up wave
//! binds real weights and wires the forward. On a **real converted
//! GGUF** we can — and here do — assert:
//!
//! - the file opens (GGUF v3, well-formed);
//! - `vokra.model.arch` equals the runtime constant
//!   ([`irodori::EXPECTED_ARCH`] / [`vits_ja::EXPECTED_ARCH`]) — a
//!   silent mis-routing across arch families is caught here, not "on
//!   the next forward";
//! - every documented axis in the arch's `vokra.<arch>.*` chunk group
//!   agrees with the runtime canonical constants
//!   ([`IrodoriConfig::irodori_500m_v3`] /
//!   [`VitsJaConfig::espnet_ja_jsut_22khz`]);
//! - the converter passed at least one float tensor through (a
//!   metadata-only GGUF from the "no float tensors" note in each
//!   converter's rustdoc would here surface as a loud FAIL rather than
//!   a silent "shape OK");
//! - the scaffold `synthesize` refuses loudly (FR-EX-08 pin — this is
//!   the assertion that goes AWAY the moment real weights bind).
//!
//! Once `VOKRA_<A>_REFDIR` is populated, an additional byte-level
//! comparison step fires (`assert_close` at `atol = 0.01` against the
//! reference `.f32` blobs — see [`compare_against_refdir`]).
//!
//! Reference-dir provenance is DELIBERATELY the operator's problem:
//! Vokra does not fetch upstream weights in this harness (that is the
//! workflow YAML `.github/workflows/parity-tts-japanese-real.yml`'s
//! job). The harness's contract is "GGUF here, reference here →
//! verdict", not "download anything".
//!
//! # No third-party deps
//!
//! Standard library + `vokra-core::gguf` + `vokra-models::{irodori, vits_ja}`
//! only. Root `Cargo.lock` is unaffected (NFR-DS-02). The workflow
//! guard is the final `git diff --exit-code Cargo.lock` step in the
//! YAML.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_models::irodori::{
    IRODORI_SAMPLE_RATE, IRODORI_TEXT_TOKENIZER_REPO, IrodoriConfig, IrodoriTts, IrodoriWeights,
};
use vokra_models::vits_ja::{
    VITS_JA_LEAKY_RELU_SLOPE, VITS_JA_SAMPLE_RATE, VitsJaConfig, VitsJaTts, VitsJaWeights,
};

/// Global FP32 parity tolerance (NFR-QL-01). Applied to the reference-dir
/// byte-level leg. A per-tensor override lands here as an explicit `match`
/// arm (the [`atol_for`] helper) with a rustdoc + ADR rationale — never a
/// bulk widening. Today no tensor requires an override.
const ATOL: f32 = 0.01;

/// The membership of the `tts-japanese` family at Phase 1. Every entry
/// gets one `#[test]` below; adding an entry here means adding a matching
/// test function (no dynamic discovery so `cargo test <name>` still
/// surfaces each row).
///
/// Slug convention: **snake_case** (matches the workflow matrix column
/// `arch_slug` and the CLI kebab-case `--model <slug>` after the sole
/// hyphen swap for `vits-ja`).
const FAMILY: &[&str] = &["irodori", "vits_ja"];

/// Per-arch upstream identifiers — used solely to build informative
/// skip messages. **Not** the source of truth for the workflow-side HF
/// pin (that lives in the YAML `env:` block); a drift between the two
/// is caught by the `hf_repo_for_arch_matches_module_rustdoc` unit test.
fn hf_repo_for(arch: &str) -> &'static str {
    match arch {
        "irodori" => "Aratako/Irodori-TTS-500M-v3",
        "vits_ja" => "espnet/kan-bayashi_jsut_vits",
        _ => panic!("hf_repo_for: unknown arch {arch:?}"),
    }
}

/// Env-var suffix for a model's converted GGUF path: `irodori` →
/// `VOKRA_IRODORI_GGUF`, `vits_ja` → `VOKRA_VITS_JA_GGUF`.
fn gguf_env_var(arch: &str) -> String {
    format!("VOKRA_{}_GGUF", arch.to_ascii_uppercase().replace('-', "_"))
}

/// Env-var suffix for a model's optional reference-dump directory:
/// `irodori` → `VOKRA_IRODORI_REFDIR`, `vits_ja` → `VOKRA_VITS_JA_REFDIR`.
/// Absent → shape / metadata leg only.
fn refdir_env_var(arch: &str) -> String {
    format!(
        "VOKRA_{}_REFDIR",
        arch.to_ascii_uppercase().replace('-', "_")
    )
}

/// Reads both env-var paths for an arch. Either or both may be `None`.
///
/// The reference-dir path is *not* filesystem-validated here (the
/// byte-level leg does that): a set-but-nonexistent `REFDIR` value is
/// passed through so the byte-level leg can surface a loud FAIL rather
/// than a silent "no comparison found".
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let gguf = std::env::var_os(gguf_env_var(arch)).map(PathBuf::from);
    let refdir = std::env::var_os(refdir_env_var(arch)).map(PathBuf::from);
    (gguf, refdir)
}

/// Builds a stderr-facing skip message that names both env vars and the
/// convert command that would flip the leg green.
///
/// The message is deliberately verbose so an operator scanning
/// `cargo test -- --nocapture` output can reproduce the flip without
/// reading this file. A "fabricated pass" would be silent skipping;
/// every skip here is loud.
///
/// For `vits_ja`, the message additionally spells out the corpus-terms
/// blocker so the operator understands why the CI leg never fetches
/// weights automatically (JSUT / JVS terms forbid re-distribution;
/// `espnet/kan-bayashi_jsut_vits` also happens to 401 on HF hub).
fn skip_reason(arch: &str, hf_repo: &str) -> String {
    let gguf = gguf_env_var(arch);
    let refdir = refdir_env_var(arch);
    let cli_model = match arch {
        "irodori" => "irodori",
        "vits_ja" => "vits-ja",
        _ => arch,
    };
    let corpus_note = if arch == "vits_ja" {
        "\n  \
         NOTE: `espnet/kan-bayashi_jsut_vits` currently returns HTTP 401 on \
         `huggingface.co/api/models/…` (gated / removed from public listing), \
         AND the JSUT corpus terms explicitly forbid re-distribution of the \
         trained weight (`sites.google.com/site/shinnosuketakamichi/publication/jsut`, \
         'Re-distribution is not permitted'). The workflow therefore never \
         auto-downloads this checkpoint — the vits_ja leg is operator-provisioned. \
         Architecture rides Apache-2.0 (ESPnet) + MIT (jaywalnut310/vits) code and \
         is always independently implementable (whisper.cpp 型 clean-room \
         re-imp, CLAUDE.md 設計判断 4)."
    } else {
        ""
    };
    format!(
        "[parity/{arch}] SKIP: {gguf} is unset.\n  \
         To flip this leg green:\n    \
         (1) fetch the upstream checkpoint from huggingface.co/{hf_repo},\n    \
         (2) `vokra-cli convert --model {cli_model} --input <safetensors> --output <out.gguf>`,\n    \
         (3) re-run with `{gguf}=<out.gguf>` set.\n  \
         Add `{refdir}=<dir>` to additionally run the byte-level reference \
         comparison (atol = {ATOL}).{corpus_note}"
    )
}

// ---------------------------------------------------------------------------
// Sanity: the family list, env var derivations, and expected-arch pins
// ---------------------------------------------------------------------------
//
// These run everywhere (CI + local, no env vars, no GGUF), and pin the
// contract that keeps the two members' env vars derivable from the family
// list. A silent typo here (adding "irodori-tts" instead of "irodori")
// would silently produce `VOKRA_IRODORI-TTS_GGUF` which no operator would
// guess.

#[test]
fn family_membership_is_stable() {
    // Adding / renaming an arch requires a conscious edit here — a
    // silent rename would drop the corresponding `#[test]` from the
    // roster.
    assert_eq!(FAMILY, &["irodori", "vits_ja"]);
}

#[test]
fn family_arches_agree_with_runtime_expected_constants() {
    // If a runtime EXPECTED_ARCH ever drifts, this test flips FAIL
    // before any GGUF is opened — cheaper than debugging a metadata
    // mismatch during parity.
    //
    // Note the intentional slug-vs-arch-tag asymmetry: the harness's
    // family SLUG is `irodori` (snake_case, matches the workflow
    // matrix column and CLI `--model`), while the arch TAG the GGUF
    // carries is `irodori-tts` (kebab-case, matches upstream module
    // naming). Likewise `vits_ja` (slug) vs `vits-ja` (tag).
    assert_eq!(vokra_models::irodori::EXPECTED_ARCH, "irodori-tts");
    assert_eq!(vokra_models::vits_ja::EXPECTED_ARCH, "vits-ja");
}

#[test]
fn env_var_derivation_is_stable_across_family() {
    // `irodori`  → `VOKRA_IRODORI_GGUF` / `VOKRA_IRODORI_REFDIR`
    // `vits_ja`  → `VOKRA_VITS_JA_GGUF` / `VOKRA_VITS_JA_REFDIR`
    assert_eq!(gguf_env_var("irodori"), "VOKRA_IRODORI_GGUF");
    assert_eq!(refdir_env_var("irodori"), "VOKRA_IRODORI_REFDIR");
    assert_eq!(gguf_env_var("vits_ja"), "VOKRA_VITS_JA_GGUF");
    assert_eq!(refdir_env_var("vits_ja"), "VOKRA_VITS_JA_REFDIR");
}

#[test]
fn hf_repo_for_arch_matches_module_rustdoc() {
    // The `hf_repo_for` map is the sole source of the skip-message HF
    // slug. It has to agree with (a) the runtime module rustdoc so an
    // operator following the skip message lands on the same page, and
    // (b) the workflow YAML `env:` pin block so a workflow_dispatch run
    // fetches the same repo the harness invited.
    assert_eq!(hf_repo_for("irodori"), "Aratako/Irodori-TTS-500M-v3");
    assert_eq!(hf_repo_for("vits_ja"), "espnet/kan-bayashi_jsut_vits");
}

#[test]
fn skip_reason_names_both_env_vars_and_the_convert_recipe() {
    // Belt-and-braces on the "loud skip" invariant: an operator reading
    // the stderr on a skipped CI leg must see (a) which env var is
    // missing, (b) which HF repo to fetch, (c) the convert command to
    // run, (d) for vits_ja specifically, the corpus-terms note.
    for &arch in FAMILY {
        let hf = hf_repo_for(arch);
        let msg = skip_reason(arch, hf);
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
            msg.contains(&format!("huggingface.co/{hf}")),
            "{arch}: skip message omits HF repo: {msg:?}"
        );
    }
    // vits_ja must additionally spell out the corpus-terms blocker so
    // an operator understands why CI never auto-fetches this weight.
    let vits_msg = skip_reason("vits_ja", hf_repo_for("vits_ja"));
    assert!(
        vits_msg.contains("JSUT"),
        "vits_ja skip message must name the JSUT corpus terms: {vits_msg:?}"
    );
    assert!(
        vits_msg.contains("Re-distribution is not permitted"),
        "vits_ja skip message must quote the JSUT redistribution ban: {vits_msg:?}"
    );
}

// ---------------------------------------------------------------------------
// Shared GGUF-side assertions (arch tag, hparam chunk, tensor count > 0)
// ---------------------------------------------------------------------------

/// Panics loudly if `key` is absent or not a string (FR-EX-08: a
/// missing arch tag is not a "maybe compatible" state — it is a hard
/// refusal).
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

/// Reads a `u32` array — used by the vits_ja decoder's `upsample_scales`
/// / `upsample_kernel_sizes` / `resblock_kernel_sizes` /
/// `resblock_dilations_flat_u32` chunks. Panics loudly on any type
/// mismatch (a `String` disguised as an array etc.).
fn expect_u32_array(file: &GgufFile, key: &str) -> Vec<u32> {
    let arr = file
        .get(key)
        .unwrap_or_else(|| panic!("GGUF metadata key {key} missing"))
        .as_array()
        .unwrap_or_else(|| panic!("GGUF metadata key {key} is not an array"));
    arr.values
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|u| u32::try_from(u).ok())
                .unwrap_or_else(|| {
                    panic!(
                        "GGUF metadata key {key}: element {v:?} does not fit in u32 (element_type = {:?})",
                        arr.element_type
                    )
                })
        })
        .collect()
}

/// Asserts the file's `vokra.model.arch` equals `expected_arch`. A
/// mismatch here catches a mis-routed converter (an `irodori-tts` GGUF
/// handed to the vits_ja test would silently look "close enough"
/// without this pin).
fn assert_arch(file: &GgufFile, expected_arch: &str, ctx: &str) {
    let arch = expect_string(file, chunks::KEY_MODEL_ARCH);
    assert_eq!(
        arch, expected_arch,
        "{ctx}: vokra.model.arch = {arch:?} but expected {expected_arch:?} \
         — is this the right converted GGUF for this test?"
    );
}

/// Asserts the converter shipped at least one float tensor through (a
/// metadata-only GGUF — the "no float tensors passed through" note in
/// each converter's rustdoc — is a loud FAIL, not a silent pass).
fn assert_tensor_count_positive(file: &GgufFile, ctx: &str) {
    let n = file.tensors().len();
    assert!(
        n > 0,
        "{ctx}: converted GGUF has zero tensors — the converter probably \
         hit the 'no float tensors passed through' path (BF16 unwidened \
         source with the F32/F16 pass-through arm only). Pre-widen \
         offline or wait for the streaming BF16 pass-through path."
    );
    eprintln!("[parity/{ctx}] GGUF tensor count = {n}");
}

// ---------------------------------------------------------------------------
// Reference-dir comparator (byte-level, atol = 0.01)
// ---------------------------------------------------------------------------

/// A reference-dir manifest row: `sha256 <name> <hex>`. Missing shape is
/// tolerated (the element count from the GGUF drives the comparison
/// length).
struct RefStage {
    name: String,
}

/// Reads `manifest.txt` in `dir` and returns each `sha256 <name> <hex>`
/// name exactly once, in file order. Non-matching lines are ignored.
///
/// Returns `Err` if the manifest is missing or unreadable — a set-but-
/// empty `REFDIR` is not silently downgraded to "no comparison"
/// (fabricated pass).
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
        // The hex is present-but-unused here (the fixture-bytes SHA
        // guard is a workflow step; this test consumes the raw bytes
        // directly).
        let _hex = parts
            .next()
            .ok_or_else(|| format!("manifest line missing hex: {line:?}"))?;
        stages.push(RefStage {
            name: name.to_owned(),
        });
    }
    Ok(stages)
}

/// Compares every reference `<name>.f32` blob against the GGUF tensor
/// of the same name at `atol = 0.01`. Returns the (compared, skipped)
/// counts. A blob whose corresponding tensor is absent from the GGUF is
/// reported as SKIP (with a loud eprintln) — this is deliberately
/// non-fatal because a reference dump may cover stages the current wave
/// has not yet bound (the harness is meant to flip green stage-by-stage,
/// not all-or-nothing).
///
/// A blob whose element count disagrees with the GGUF tensor's is a
/// hard FAIL (shape mismatch is the exact "fabricated pass" the
/// honest-parity rule bans).
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
            "[parity/{ctx}] refdir {} manifest.txt has no `sha256 <name> <hex>` \
             rows — no reference stages to compare against yet. Populate the \
             reference dump to enable byte-level parity.",
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
                "[parity/{ctx}] SKIP {}: not present in GGUF (weight-binding \
                 wave may not have landed this stage yet)",
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
            "{ctx}: tensor {:?}: GGUF has {} elements, reference has {} — \
             shape mismatch is a hard FAIL (fabricated pass 禁止, FR-EX-08)",
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

/// `irodori` — `Aratako/Irodori-TTS-500M-v3` (MIT).
///
/// Rectified-Flow DiT over a 32-dim continuous DACVAE latent driven by
/// a Semantic-DACVAE-Japanese-32dim decoder (48 kHz mono PCM). Real-
/// weight binding is a follow-up wave — until then this test proves:
///
/// * GGUF opens + `vokra.model.arch = "irodori-tts"`;
/// * every documented hparam in the `vokra.irodori.*` chunk group agrees
///   with [`IrodoriConfig::irodori_500m_v3`];
/// * the converter shipped ≥1 float tensor through (no metadata-only
///   GGUF);
/// * `IrodoriTts::synthesize` refuses loudly (FR-EX-08 pin — synthesize
///   MUST NOT return a hallucinated waveform from scaffold weights).
///
/// If `VOKRA_IRODORI_REFDIR` is also set, the byte-level leg
/// (`atol = 0.01`) fires as a bonus.
#[test]
fn parity_tts_japanese_irodori() {
    let (Some(gguf), refdir) = env_paths_for("irodori") else {
        eprintln!("{}", skip_reason("irodori", hf_repo_for("irodori")));
        return;
    };
    let file = GgufFile::open(&gguf)
        .unwrap_or_else(|e| panic!("open VOKRA_IRODORI_GGUF = {}: {e}", gguf.display()));
    assert_arch(&file, vokra_models::irodori::EXPECTED_ARCH, "irodori");
    assert_tensor_count_positive(&file, "irodori");

    // Hparam chunk agreement with the canonical release. Every value
    // below is transcribed verbatim from `IrodoriConfig::irodori_500m_v3`
    // (which itself is transcribed from
    // `configs/train_500m_v3_phase1_body.yaml` +
    // `configs/train_500m_v3_phase2_duration.yaml` at
    // `github.com/Aratako/Irodori-TTS`). A converter that stamped
    // different constants would silently mis-route (FR-EX-08 catch here).
    let canonical = IrodoriConfig::irodori_500m_v3();

    // Top-level
    assert_eq!(
        expect_string(&file, "vokra.irodori.model_family"),
        "irodori-tts"
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.sample_rate_hz"),
        u64::from(canonical.sample_rate)
    );
    assert_eq!(
        expect_string(&file, "vokra.irodori.text_tokenizer_repo"),
        canonical.text_tokenizer_repo,
    );

    // DiT
    let d = &canonical.dit;
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.latent_dim"),
        u64::from(d.latent_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.latent_patch_size"),
        u64::from(d.latent_patch_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.model_dim"),
        u64::from(d.model_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.num_layers"),
        u64::from(d.num_layers)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.num_heads"),
        u64::from(d.num_heads)
    );
    // f32 fields are stored via `add_f32` and read back via
    // `expect_f64` (writer widens f32 → f64 losslessly through the
    // as_f64 view).
    assert!(
        (expect_f64(&file, "vokra.irodori.dit.mlp_ratio") as f32 - d.mlp_ratio).abs() < 1e-6,
        "irodori dit.mlp_ratio drift"
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.timestep_embed_dim"),
        u64::from(d.timestep_embed_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.dit.adaln_rank"),
        u64::from(d.adaln_rank)
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.dit.norm_eps") as f32 - d.norm_eps).abs() < 1e-9,
        "irodori dit.norm_eps drift"
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.dit.dropout") as f32 - d.dropout).abs() < 1e-9,
        "irodori dit.dropout drift"
    );

    // Text encoder
    let t = &canonical.text;
    assert_eq!(
        expect_u64(&file, "vokra.irodori.text.vocab_size"),
        u64::from(t.vocab_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.text.dim"),
        u64::from(t.dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.text.n_layer"),
        u64::from(t.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.text.n_head"),
        u64::from(t.n_head)
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.text.mlp_ratio") as f32 - t.mlp_ratio).abs() < 1e-6,
        "irodori text.mlp_ratio drift"
    );
    assert_eq!(expect_bool(&file, "vokra.irodori.text.add_bos"), t.add_bos);

    // Speaker (reference-latent) encoder
    let s = &canonical.speaker;
    assert_eq!(
        expect_u64(&file, "vokra.irodori.speaker.dim"),
        u64::from(s.dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.speaker.n_layer"),
        u64::from(s.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.speaker.n_head"),
        u64::from(s.n_head)
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.speaker.mlp_ratio") as f32 - s.mlp_ratio).abs() < 1e-6,
        "irodori speaker.mlp_ratio drift"
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.speaker.patch_size"),
        u64::from(s.patch_size)
    );

    // Duration predictor (v3 phase-2)
    let dur = &canonical.duration;
    assert_eq!(
        expect_bool(&file, "vokra.irodori.duration.enabled"),
        dur.enabled
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.duration.aux_dim"),
        u64::from(dur.aux_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.duration.hidden_dim"),
        u64::from(dur.hidden_dim)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.duration.n_layer"),
        u64::from(dur.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.irodori.duration.n_head"),
        u64::from(dur.n_head)
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.duration.dropout") as f32 - dur.dropout).abs() < 1e-6,
        "irodori duration.dropout drift"
    );
    assert_eq!(
        expect_string(&file, "vokra.irodori.duration.architecture"),
        dur.architecture,
    );
    assert!(
        (expect_f64(&file, "vokra.irodori.duration.token_init_frames") as f32
            - dur.token_init_frames)
            .abs()
            < 1e-6,
        "irodori duration.token_init_frames drift"
    );
    assert_eq!(
        expect_string(&file, "vokra.irodori.duration.speaker_fusion"),
        dur.speaker_fusion,
    );

    // Sample-rate + tokenizer anchors (guard against a converter that
    // silently swapped the paired DACVAE codec's 48 kHz output rate or
    // the LLM-JP-3 tokenizer id — both would break e2e forwards).
    assert_eq!(IRODORI_SAMPLE_RATE, 48_000);
    assert_eq!(IRODORI_TEXT_TOKENIZER_REPO, "llm-jp/llm-jp-3-150m");

    // FR-EX-08 pin — the scaffold engine MUST refuse `synthesize`
    // loudly. This assertion goes AWAY the moment a real-weight
    // `from_gguf` walk lands and the engine can bind real weights (at
    // which point this test's synthesize call would become a real
    // forward + audio-bound sanity check).
    let cfg = IrodoriConfig::irodori_500m_v3();
    let weights = IrodoriWeights::synthesized(&cfg).expect("build irodori scaffold weights");
    let tts = IrodoriTts::new(cfg, weights).expect("build irodori scaffold engine");
    let err = tts
        .synthesize("こんにちは")
        .expect_err("scaffold synthesize must refuse loudly");
    assert!(
        matches!(err, VokraError::NotImplemented(_)),
        "irodori: scaffold synthesize returned unexpected variant {err:?}"
    );

    // Optional flip-the-switch leg — byte-level parity against the
    // reference dump if the operator wired one up.
    if let Some(refdir) = refdir {
        let (compared, skipped) = compare_against_refdir(&file, &refdir, "irodori");
        eprintln!(
            "[parity/irodori] reference comparison: {compared} tensors compared, {skipped} skipped"
        );
    } else {
        eprintln!(
            "[parity/irodori] byte-level reference skipped: VOKRA_IRODORI_REFDIR \
             unset. (Shape / metadata leg passed above.)"
        );
    }
}

/// `vits_ja` — `espnet/kan-bayashi_jsut_vits` (Apache-2.0 code /
/// **restricted** weight per JSUT terms).
///
/// Plain VITS (Kim et al. 2021) with a HiFi-GAN decoder — the ESPnet
/// JA VITS recipe. The publicly distributed ESPnet-JSUT / ESPnet-JVS /
/// COEIROINK checkpoints ride on corpus terms that forbid re-distribution
/// of the trained weight, so the workflow deliberately **does not
/// auto-fetch** this checkpoint. This test therefore expects to run in
/// one of two modes:
///
///   * **honest-skip** (default on hosted CI) — `VOKRA_VITS_JA_GGUF`
///     unset → the test prints a loud skip message naming the JSUT
///     redistribution ban and returns cleanly (never a fabricated pass);
///   * **operator-provisioned** — the operator has produced a GGUF
///     locally (either from a permissive-corpus re-training, or under a
///     JSUT-compliant local use) and sets `VOKRA_VITS_JA_GGUF=<path>`.
///
/// When a GGUF is provided, this test proves:
///
/// * GGUF opens + `vokra.model.arch = "vits-ja"`;
/// * every documented hparam in the `vokra.vits_ja.*` chunk group agrees
///   with [`VitsJaConfig::espnet_ja_jsut_22khz`];
/// * the decoder u32-array chunks (`upsample_scales`,
///   `upsample_kernel_sizes`, `resblock_kernel_sizes`,
///   `resblock_dilations_flat_u32`) round-trip verbatim;
/// * the converter shipped ≥1 float tensor through;
/// * `VitsJaTts::synthesize` refuses loudly.
///
/// If `VOKRA_VITS_JA_REFDIR` is also set, the byte-level leg fires.
#[test]
fn parity_tts_japanese_vits_ja() {
    let (Some(gguf), refdir) = env_paths_for("vits_ja") else {
        eprintln!("{}", skip_reason("vits_ja", hf_repo_for("vits_ja")));
        return;
    };
    let file = GgufFile::open(&gguf)
        .unwrap_or_else(|e| panic!("open VOKRA_VITS_JA_GGUF = {}: {e}", gguf.display()));
    assert_arch(&file, vokra_models::vits_ja::EXPECTED_ARCH, "vits_ja");
    assert_tensor_count_positive(&file, "vits_ja");

    let canonical = VitsJaConfig::espnet_ja_jsut_22khz();

    // Top-level shortcuts
    assert_eq!(
        expect_string(&file, "vokra.vits_ja.model_family"),
        "vits-ja"
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.sample_rate_hz"),
        u64::from(canonical.sample_rate)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.vocab_size"),
        u64::from(canonical.vocab_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.n_mels"),
        u64::from(canonical.n_mels)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.aux_channels"),
        u64::from(canonical.aux_channels)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.hidden_channels"),
        u64::from(canonical.hidden_channels)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.segment_size"),
        u64::from(canonical.segment_size)
    );
    // spks sentinel: `0` in the GGUF for the single-speaker JSUT default
    // (canonical.spks is `None`); the JVS variant would ship `100`.
    let spks = expect_u64(&file, "vokra.vits_ja.spks");
    assert!(
        spks == 0 || spks == u64::from(canonical.spks.unwrap_or(0)),
        "vits_ja spks: got {spks}, canonical single-speaker expects 0 (JVS: 100)"
    );

    // Text encoder
    let t = &canonical.text_encoder;
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.text.n_layer"),
        u64::from(t.n_layer)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.text.n_head"),
        u64::from(t.n_head)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.text.ffn_expand"),
        u64::from(t.ffn_expand)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.text.positionwise_conv_kernel"),
        u64::from(t.positionwise_conv_kernel_size)
    );
    assert!(
        (expect_f64(&file, "vokra.vits_ja.text.dropout_rate") as f32 - t.dropout_rate).abs() < 1e-6,
        "vits_ja text.dropout_rate drift"
    );
    assert!(
        (expect_f64(&file, "vokra.vits_ja.text.positional_dropout_rate") as f32
            - t.positional_dropout_rate)
            .abs()
            < 1e-9,
        "vits_ja text.positional_dropout_rate drift"
    );
    assert!(
        (expect_f64(&file, "vokra.vits_ja.text.attention_dropout_rate") as f32
            - t.attention_dropout_rate)
            .abs()
            < 1e-6,
        "vits_ja text.attention_dropout_rate drift"
    );
    assert_eq!(
        expect_bool(&file, "vokra.vits_ja.text.use_macaron_style"),
        t.use_macaron_style,
    );
    assert_eq!(
        expect_bool(&file, "vokra.vits_ja.text.use_conformer_conv"),
        t.use_conformer_conv,
    );

    // Flow
    let f = &canonical.flow;
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.flow.n_flow"),
        u64::from(f.n_flow)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.flow.kernel_size"),
        u64::from(f.kernel_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.flow.base_dilation"),
        u64::from(f.base_dilation)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.flow.n_layer"),
        u64::from(f.n_layer)
    );
    assert!(
        (expect_f64(&file, "vokra.vits_ja.flow.dropout_rate") as f32 - f.dropout_rate).abs() < 1e-9,
        "vits_ja flow.dropout_rate drift"
    );
    assert_eq!(
        expect_bool(&file, "vokra.vits_ja.flow.use_only_mean"),
        f.use_only_mean,
    );

    // SDP
    let sdp = &canonical.sdp;
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.sdp.kernel_size"),
        u64::from(sdp.kernel_size)
    );
    assert!(
        (expect_f64(&file, "vokra.vits_ja.sdp.dropout_rate") as f32 - sdp.dropout_rate).abs()
            < 1e-6,
        "vits_ja sdp.dropout_rate drift"
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.sdp.n_flow"),
        u64::from(sdp.n_flow)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.sdp.dds_conv_layers"),
        u64::from(sdp.dds_conv_layers)
    );

    // HiFi-GAN decoder — scalar axes
    let dec = &canonical.decoder;
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.decoder.kernel_size"),
        u64::from(dec.kernel_size)
    );
    assert_eq!(
        expect_u64(&file, "vokra.vits_ja.decoder.initial_channel"),
        u64::from(dec.initial_channel)
    );
    assert_eq!(
        expect_bool(&file, "vokra.vits_ja.decoder.use_weight_norm"),
        dec.use_weight_norm,
    );

    // HiFi-GAN decoder — u32 array chunks (verbatim slice equality is
    // the point: any transposition of upsample stages or resblock
    // branches would silently mis-shape the decoder).
    assert_eq!(
        expect_u32_array(&file, "vokra.vits_ja.decoder.upsample_scales"),
        dec.upsample_scales,
    );
    assert_eq!(
        expect_u32_array(&file, "vokra.vits_ja.decoder.upsample_kernel_sizes"),
        dec.upsample_kernel_sizes,
    );
    assert_eq!(
        expect_u32_array(&file, "vokra.vits_ja.decoder.resblock_kernel_sizes"),
        dec.resblock_kernel_sizes,
    );
    // `resblock_dilations_flat_u32` is a flattened 2-D matrix of shape
    // `(n_branches, stride)`; the stride is stamped alongside so the
    // reader can re-shape without depending on a fixed stride constant.
    let flat = expect_u32_array(&file, "vokra.vits_ja.decoder.resblock_dilations_flat_u32");
    let stride = expect_u64(&file, "vokra.vits_ja.decoder.resblock_dilations_stride") as usize;
    assert!(stride > 0, "vits_ja resblock_dilations stride must be > 0");
    assert_eq!(
        flat.len() % stride,
        0,
        "vits_ja resblock_dilations_flat_u32 length {} is not a multiple of stride {stride}",
        flat.len()
    );
    let mut got: Vec<Vec<u32>> = Vec::new();
    for chunk in flat.chunks_exact(stride) {
        got.push(chunk.to_vec());
    }
    assert_eq!(
        got, dec.resblock_dilations,
        "vits_ja resblock_dilations do not round-trip"
    );

    // Leaky-ReLU slope anchor (guard against a converter that silently
    // swapped 0.1 for a different slope — HiFi-GAN parity across the
    // ecosystem is 0.1).
    assert!((VITS_JA_LEAKY_RELU_SLOPE - 0.1).abs() < 1e-9);
    // Sample-rate anchor.
    assert_eq!(VITS_JA_SAMPLE_RATE, 22_050);

    // FR-EX-08 pin — scaffold synthesize must refuse loudly.
    let cfg = VitsJaConfig::espnet_ja_jsut_22khz();
    let weights = VitsJaWeights::synthesized(&cfg).expect("build vits_ja scaffold weights");
    let tts = VitsJaTts::new(cfg, weights).expect("build vits_ja scaffold engine");
    let err = tts
        .synthesize("こんにちは")
        .expect_err("scaffold synthesize must refuse loudly");
    assert!(
        matches!(err, VokraError::NotImplemented(_)),
        "vits_ja: scaffold synthesize returned unexpected variant {err:?}"
    );

    if let Some(refdir) = refdir {
        let (compared, skipped) = compare_against_refdir(&file, &refdir, "vits_ja");
        eprintln!(
            "[parity/vits_ja] reference comparison: {compared} tensors compared, {skipped} skipped"
        );
    } else {
        eprintln!(
            "[parity/vits_ja] byte-level reference skipped: VOKRA_VITS_JA_REFDIR \
             unset. (Shape / metadata leg passed above.)"
        );
    }
}

// ---------------------------------------------------------------------------
// Helper-level unit tests — the pure-function surface that the flip-the-switch
// contract rests on (env seam, manifest parser, shape validators, skip
// diagnostics). Every test below is deterministic (no wall-clock, no external
// state), fast (< 100 ms), and feasible without a real GGUF / checkpoint —
// scratch dirs are stemmed with PID + nanoseconds so parallel `cargo test`
// never collides, and synthetic GGUFs are built via `GgufBuilder::to_bytes()`
// + `GgufFile::parse()` so no filesystem GGUF is ever required.
//
// The workspace commits to `-D unsafe_code` and Rust 2024 marks
// `std::env::set_var` as `unsafe`, so the negative env case uses a namespaced
// arch guaranteed unset in CI (rather than mutating the process environment).
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
        "vokra_parity_tts_japanese_{}_{}_{}",
        label,
        std::process::id(),
        nanos,
    ));
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    dir
}

/// Build a minimal, well-formed GGUF byte image with the requested arch
/// tag and one 1-element F32 tensor named `probe.f32`. Enough to satisfy
/// `GgufFile::parse` + `tensor_info("probe.f32")` in the compare-against
/// tests below. Nothing model-specific.
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

/// Build a metadata-only GGUF (zero tensors). Used by the
/// `assert_tensor_count_positive` panic test.
fn build_metadata_only_gguf(arch: &str) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, arch);
    b.to_bytes().expect("serialize metadata-only GGUF")
}

/// Build a GGUF that carries one metadata key with an arbitrary typed
/// value. Used by the `expect_string` / `expect_u32_array` shape-validation
/// tests to plant a `wrong-type` value at a key that the harness reads.
fn build_gguf_with_metadata(arch: &str, key: &str, value: GgufMetadataValue) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, arch);
    b.add_metadata(key, value);
    // A single 1-elem F32 probe tensor so parses/tests that check tensor
    // presence don't trip on an entirely empty file.
    b.add_tensor(
        "probe.f32",
        GgmlType::F32,
        vec![1],
        0.0f32.to_le_bytes().to_vec(),
    )
    .expect("writer accepts a well-formed 1-elem F32 tensor");
    b.to_bytes().expect("serialize typed-metadata GGUF")
}

/// Extract the `&str` panic message from a `catch_unwind` payload.
/// `panic!` with a formatted `String` yields a `String` payload; the
/// plain `&str` variant is also fielded so the helper is drop-in for
/// both.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

// --- env_paths_for ------------------------------------------------------

/// `env_paths_for` MUST return `(None, None)` for an arch whose env vars
/// are guaranteed unset (gap #1). Uses a namespaced arch slug
/// (`ttsja_env_probe` → `VOKRA_TTSJA_ENV_PROBE_GGUF` / `_REFDIR`) so no
/// CI env var of that name could plausibly exist. This pins the "unset →
/// clean skip" contract at the seam a hosted-CI or operator-desktop run
/// traverses BEFORE ever touching a GGUF — a regression that returned
/// `Some(PathBuf::new())` (or swapped the tuple order) would let both
/// `parity_tts_japanese_irodori` and `parity_tts_japanese_vits_ja` take
/// the WRONG code path and produce a synthetic-looking failure instead of
/// an honest skip. FR-EX-08 (silent fallback banned).
#[test]
fn env_paths_for_returns_none_when_env_unset() {
    let arch = "ttsja_env_probe";
    // Precondition: derive the env var names the harness would query and
    // confirm they are indeed unset in the current process before we
    // trust the (None, None) return as evidence of correctness.
    let gguf_key = gguf_env_var(arch);
    let refdir_key = refdir_env_var(arch);
    assert!(
        std::env::var_os(&gguf_key).is_none(),
        "test precondition failed: {gguf_key} is set in the environment — either \
         env leakage or the probe arch namespace clashes with an operator-set \
         var; rename the probe arch"
    );
    assert!(
        std::env::var_os(&refdir_key).is_none(),
        "test precondition failed: {refdir_key} is set in the environment"
    );

    let (gguf, refdir) = env_paths_for(arch);
    assert!(
        gguf.is_none(),
        "expected {gguf_key} unset → env_paths_for gguf slot None; got \
         Some({gguf:?}) — regression: seam returns Some on an unset env var \
         (flip-the-switch broken)"
    );
    assert!(
        refdir.is_none(),
        "expected {refdir_key} unset → env_paths_for refdir slot None; got \
         Some({refdir:?}) — regression: asymmetric handling of the refdir arm"
    );
}

// --- read_ref_manifest --------------------------------------------------

/// `read_ref_manifest` MUST return `Err(...)` when `manifest.txt` is
/// absent from the refdir (gap #2). The rustdoc explicitly promises "a
/// set-but-empty REFDIR is not silently downgraded to no comparison
/// (fabricated pass)". A regression that swallowed the I/O error and
/// returned `Ok(vec![])` would silently downgrade the byte-level leg to
/// a no-op for every gated model.
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
    // The message must at least name the manifest filename so the
    // operator knows *which* refdir is malformed.
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

/// `read_ref_manifest` boundary: a zero-byte `manifest.txt` MUST return
/// `Ok(vec![])` — distinct from the missing-file case (gap #3). Together
/// with `compare_against_refdir_returns_zero_zero_on_empty_manifest`
/// below this pins that "materialised-but-empty manifest" is a valid,
/// loud "no stages yet" state — a legitimate operator misconfiguration
/// (touching the file to check permissions is common) must NOT collapse
/// to the missing-file Err arm.
#[test]
fn read_ref_manifest_empty_file_returns_empty_vec() {
    let dir = make_scratch_dir("empty_manifest");
    // Zero-byte file: the read succeeds, the split yields no lines, the
    // for loop body never executes → Ok(vec![]).
    std::fs::write(dir.join("manifest.txt"), b"").expect("write empty manifest");
    let stages = read_ref_manifest(&dir)
        .expect("zero-byte manifest.txt must return Ok(vec![]), not Err (missing-file arm)");
    assert!(
        stages.is_empty(),
        "zero-byte manifest must yield 0 RefStage entries; got {} — parser \
         hallucinated a row from empty input",
        stages.len()
    );
}

/// `read_ref_manifest` MUST silently ignore lines that do NOT begin with
/// `sha256 ` — blank lines, `# comment` lines, and any other prefix
/// (gap #4). This undocumented behaviour lets an operator drop a comment
/// header (e.g. `# generated by dump.py at <sha>`) into a manifest
/// without breaking the parse. If a future refactor tightened this to
/// `return Err(...)` on unknown prefixes, real reference dumps with
/// headers would break silently — this test pins the current behaviour
/// so any such refactor becomes visible.
#[test]
fn read_ref_manifest_ignores_non_sha256_lines() {
    let dir = make_scratch_dir("mixed_lines");
    // Mix: comment / blank / valid / unknown-prefix / blank / valid.
    // Only the two sha256 rows should surface in the output.
    let body = "\
# generated by tools/parity/dump_irodori.py at commit abc123

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

/// `read_ref_manifest` happy path: a manifest with three valid
/// `sha256 <name> <hex>` lines yields three `RefStage` entries in FILE
/// ORDER with correct `.name` values (gap #5). The rustdoc pins
/// "exactly once, in file order" — that invariant has zero direct tests
/// today. A regression that swapped `parts.next()` for `parts.next_back()`
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
sha256 vits.decoder.final feedbead00000003
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
        stages[2].name, "vits.decoder.final",
        "row 2 name mismatch; parser walked out of order?"
    );
}

/// `read_ref_manifest` MUST reject a `sha256 ` line with no name token
/// (line is `sha256 ` followed by nothing / only whitespace) with an Err
/// naming the "missing name" arm (gap #6). Distinct from the missing-hex
/// case because the two `ok_or_else` arms are dead code from a coverage
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
/// the "missing hex" arm (gap #7). Symmetric to the missing-name test —
/// pins the second `ok_or_else` arm and asserts the message is DIFFERENT
/// from the missing-name case (operator debuggability).
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

// --- hf_repo_for --------------------------------------------------------

/// `hf_repo_for("<unknown>")` MUST panic loudly (gap #8). The panic
/// branch at the `_ =>` arm is defensive code that would only fire on a
/// mis-typed `FAMILY` entry — which is exactly the failure mode we want
/// caught pre-commit (a silent typo would silently produce a broken
/// skip message pointing at a non-existent HF repo). `#[should_panic]`
/// pins that the "unknown arch" substring appears in the message so an
/// operator sees which arm fired.
#[test]
#[should_panic(expected = "unknown arch")]
fn hf_repo_for_panics_on_unknown_arch() {
    let _ = hf_repo_for("nemo");
}

// --- skip_reason --------------------------------------------------------

/// The `irodori` skip message MUST NOT leak the JSUT corpus-terms note
/// (gap #9). The corpus block is `vits_ja`-exclusive because JSUT is the
/// specific corpus that forbids re-distribution of the trained weight —
/// irodori rides Aratako's permissive MIT release. A regression that
/// widened the corpus note to every family would legally mis-attribute
/// the block and confuse an operator following the skip message. This
/// belt-and-braces test complements
/// `skip_reason_names_both_env_vars_and_the_convert_recipe` (which only
/// asserts vits_ja HAS these strings).
#[test]
fn skip_reason_irodori_omits_jsut_corpus_note() {
    let msg = skip_reason("irodori", hf_repo_for("irodori"));
    assert!(
        !msg.contains("JSUT"),
        "irodori skip message must NOT mention JSUT (that block is \
         vits_ja-exclusive); got: {msg:?}"
    );
    assert!(
        !msg.contains("Re-distribution is not permitted"),
        "irodori skip message must NOT quote the JSUT redistribution ban \
         (that block is vits_ja-exclusive); got: {msg:?}"
    );
    // Sanity: the vits_ja counterpart still carries both, so the family
    // asymmetry is what we're pinning (not a whole-family drop).
    let vits_msg = skip_reason("vits_ja", hf_repo_for("vits_ja"));
    assert!(
        vits_msg.contains("JSUT"),
        "vits_ja skip message MUST still name JSUT (the asymmetry is the \
         point of this test); got: {vits_msg:?}"
    );
}

/// The skip message MUST spell out the exact ATOL numeric value (gap #10)
/// so a future ATOL bump forces the operator-facing recipe text to update
/// in lock-step. The format string stringifies `{ATOL}` with Rust's
/// default float Display; a widening or format change (e.g. renaming
/// `ATOL` to a `Cow<'static, str>` recipe blob) would silently drop that
/// datum from operator diagnostics.
#[test]
fn skip_reason_includes_atol_value() {
    // Both members must carry the value in the "byte-level reference
    // comparison (atol = 0.01)" recipe blurb.
    for &arch in FAMILY {
        let msg = skip_reason(arch, hf_repo_for(arch));
        assert!(
            msg.contains("0.01"),
            "{arch}: skip message must contain the ATOL numeric value \
             (currently 0.01) so a future ATOL bump forces the recipe text \
             to update in lock-step; got: {msg:?}"
        );
        // The word "atol" itself must appear so the number isn't dangling.
        assert!(
            msg.contains("atol"),
            "{arch}: skip message must contain the word 'atol' next to the \
             numeric value; got: {msg:?}"
        );
    }
}

// --- env-var derivation kebab-case normalisation ------------------------

/// `gguf_env_var` / `refdir_env_var` MUST normalise `-` to `_` before
/// upper-casing (gap #11). Today only snake_case arches (`irodori`,
/// `vits_ja`) are covered by `env_var_derivation_is_stable_across_family`
/// — so the `.replace('-', '_')` call is dead-tested and could be
/// deleted with zero test failure. The `FAMILY` slug convention could
/// shift to kebab-case at any point (many upstream repos use it), and
/// the harness would still be expected to derive the same env var name
/// the operator has already exported.
#[test]
fn env_var_derivation_normalizes_kebab_case() {
    // The kebab form of `vits_ja` is what the CLI's `--model` flag
    // consumes (see `cli_model` mapping in `skip_reason`), and an
    // operator who accidentally passed the kebab form as the arch slug
    // must still see the canonical `_`-separated env var.
    assert_eq!(
        gguf_env_var("vits-ja"),
        "VOKRA_VITS_JA_GGUF",
        "gguf_env_var must normalise '-' to '_' before upper-casing so \
         kebab-case arch slugs map to the same env var as snake_case"
    );
    assert_eq!(
        refdir_env_var("vits-ja"),
        "VOKRA_VITS_JA_REFDIR",
        "refdir_env_var must normalise '-' to '_' before upper-casing so \
         kebab-case arch slugs map to the same env var as snake_case"
    );
    // A double-hyphen probe: multi-hyphen arch names (e.g. an
    // upstream that shipped as `foo-bar-baz`) must fully normalise.
    assert_eq!(
        gguf_env_var("foo-bar-baz"),
        "VOKRA_FOO_BAR_BAZ_GGUF",
        "gguf_env_var must normalise EVERY '-' occurrence, not just the first"
    );
}

// --- compare_against_refdir --------------------------------------------

/// `compare_against_refdir` MUST panic — not return `(0, 0)` — when the
/// refdir has no `manifest.txt` (gap #12). The panic message is
/// load-bearing: it carries BOTH "manifest.txt unreadable" and
/// "fabricated pass 禁止" substrings, and it names the ctx label, so an
/// operator eyeballing the CI log is taught the rule AND can identify
/// which parity leg fired. A refactor that shortened the banner or
/// quietly downgraded to `Ok((0, 0))` would defeat the honest-parity
/// contract the rustdoc promises.
#[test]
fn compare_against_refdir_panics_on_missing_manifest() {
    // A minimal but well-formed GGUF so the panic fires on the manifest
    // read (the very first thing compare_against_refdir does), not on
    // some earlier tensor-access path.
    let gguf_bytes = build_minimal_gguf("irodori-tts", "probe.f32", 0.0);
    let file = GgufFile::parse(gguf_bytes).expect("parse synthetic gguf");
    let bad_refdir = make_scratch_dir("compare_no_manifest");
    // Deliberately do NOT write manifest.txt into `bad_refdir`.

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compare_against_refdir(&file, &bad_refdir, "irodori");
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
    // The ctx label must be present so an operator with multiple parity
    // legs (irodori + vits_ja + …) can identify the failing one.
    assert!(
        msg.contains("irodori"),
        "panic message must carry the ctx label so an operator with \
         multiple gated legs can identify the failing one; got: {msg}"
    );
}

// --- expect_string ------------------------------------------------------

/// `expect_string` MUST panic with a message that names the missing key
/// when the key is entirely absent from the GGUF (gap #13). The panic
/// message shape is the FAIL surface an operator sees; a message
/// regression (e.g. loss of the key name) would surface as an
/// inscrutable "panicked at unwrap_or_else". Feasible without a real
/// weight — `GgufBuilder::to_bytes()` + `GgufFile::parse()` is enough.
#[test]
fn expect_string_panics_when_key_missing() {
    // GGUF with the arch tag but no `vokra.irodori.model_family` key.
    let bytes = build_minimal_gguf("irodori-tts", "probe.f32", 0.0);
    let file = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = expect_string(&file, "vokra.irodori.model_family");
    }))
    .expect_err(
        "expect_string must panic when the key is absent (a missing arch \
         tag is not a 'maybe compatible' state, FR-EX-08)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("vokra.irodori.model_family"),
        "panic message must name the missing key so an operator can find \
         it in the converter; got: {msg}"
    );
    assert!(
        msg.contains("missing"),
        "panic message must name the 'missing' arm so an operator can \
         distinguish it from the 'not a string' arm; got: {msg}"
    );
}

/// `expect_string` MUST panic with a message that names the wrong-type
/// arm when the key exists but is stored as a non-string type (gap #14).
/// This is exactly the branch a bad converter would trip (e.g. stamping
/// `model_family` as a `u32` opset instead of a `String` slug).
#[test]
fn expect_string_panics_when_wrong_type() {
    // GGUF where the target key is stored as U32 (writer offers no
    // native `add_u32_at_string_key`; `add_u32` is used for the wrong
    // type on purpose so `expect_string.as_str()` returns None).
    let key = "vokra.irodori.model_family";
    let bytes = {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "irodori-tts");
        b.add_u32(key, 42);
        b.add_tensor(
            "probe.f32",
            GgmlType::F32,
            vec![1],
            0.0f32.to_le_bytes().to_vec(),
        )
        .expect("writer accepts a well-formed 1-elem F32 tensor");
        b.to_bytes().expect("serialize wrong-type GGUF")
    };
    let file = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = expect_string(&file, key);
    }))
    .expect_err(
        "expect_string must panic when the key exists but is the wrong \
         type (FR-EX-08 — no silent type coercion)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains(key),
        "panic message must name the offending key; got: {msg}"
    );
    assert!(
        msg.contains("not a string"),
        "panic message must name the 'not a string' arm so an operator \
         knows it's a type mismatch (not an absence); got: {msg}"
    );
}

// --- expect_u32_array ---------------------------------------------------

/// `expect_u32_array` MUST panic with 'is not an array' when the key
/// exists but is a scalar (gap #15). This is a real converter-mistake
/// surface (a stringified list stored as one string instead of an
/// array); the panic message must name the offending key.
#[test]
fn expect_u32_array_panics_on_non_array() {
    let key = "vokra.vits_ja.decoder.upsample_scales";
    let bytes = build_gguf_with_metadata(
        "vits-ja",
        key,
        // Deliberately a String where the harness expects an Array.
        GgufMetadataValue::String("[8, 8, 2, 2]".to_owned()),
    );
    let file = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = expect_u32_array(&file, key);
    }))
    .expect_err(
        "expect_u32_array must panic when the key exists but is not an \
         array (a stringified list is exactly the converter mistake this \
         guard catches)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains(key),
        "panic message must name the offending key; got: {msg}"
    );
    assert!(
        msg.contains("is not an array"),
        "panic message must name the 'is not an array' arm; got: {msg}"
    );
}

/// `expect_u32_array` MUST panic with the 'does not fit in u32' message
/// when an element exceeds `u32::MAX` (gap #16). The suffix that
/// surfaces `element_type` is the single hardest datum for an operator
/// to reconstruct after the fact; a regression that dropped it would be
/// silently costly.
#[test]
fn expect_u32_array_panics_on_element_too_large() {
    let key = "vokra.vits_ja.decoder.upsample_scales";
    // Build a U64 array whose sole element is `u32::MAX + 1` — passes
    // `as_u64()` (widens U64 → u64 directly) but fails `u32::try_from`.
    let too_big: u64 = u64::from(u32::MAX) + 1;
    let bytes = build_gguf_with_metadata(
        "vits-ja",
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U64,
            values: vec![GgufMetadataValue::U64(too_big)],
        }),
    );
    let file = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = expect_u32_array(&file, key);
    }))
    .expect_err(
        "expect_u32_array must panic when an element exceeds u32::MAX \
         (no silent truncation, FR-EX-08)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains(key),
        "panic message must name the offending key; got: {msg}"
    );
    assert!(
        msg.contains("does not fit in u32"),
        "panic message must name the 'does not fit in u32' arm; got: {msg}"
    );
    // The element_type suffix is the operator's only clue to which
    // array-of-what tripped the guard.
    assert!(
        msg.contains("element_type"),
        "panic message must carry the 'element_type = ...' suffix so an \
         operator can see the offending array's element type; got: {msg}"
    );
    assert!(
        msg.contains("U64"),
        "panic message must render the offending element_type ({:?}); got: {msg}",
        GgufValueType::U64
    );
}

// --- assert_arch --------------------------------------------------------

/// `assert_arch` MUST panic with a message that names BOTH the actual
/// arch string in the GGUF and the expected arch string when they
/// differ (gap #17). This is the sole guard against a mis-routed
/// converter (an irodori GGUF handed to the vits_ja test); the error
/// message quality is load-bearing for triage.
#[test]
fn assert_arch_panics_on_mismatch() {
    // GGUF stamped with the WRONG arch tag on purpose.
    let bytes = build_minimal_gguf("wrong-arch", "probe.f32", 0.0);
    let file = GgufFile::parse(bytes).expect("parse synthetic gguf");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_arch(&file, "irodori-tts", "irodori");
    }))
    .expect_err(
        "assert_arch must panic when the GGUF's arch tag differs from \
         the expected string (no silent mis-routing, FR-EX-08)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("wrong-arch"),
        "panic message must name the ACTUAL arch found in the GGUF so an \
         operator sees what was stamped; got: {msg}"
    );
    assert!(
        msg.contains("irodori-tts"),
        "panic message must name the EXPECTED arch so an operator sees \
         what the harness demanded; got: {msg}"
    );
    // The ctx label ("irodori" here) must appear so an operator with
    // multiple parity legs can identify the failing one.
    assert!(
        msg.contains("irodori"),
        "panic message must carry the ctx label so an operator with \
         multiple gated legs can identify the failing one; got: {msg}"
    );
    // The guidance line ("right converted GGUF for this test") must
    // survive so the panic self-documents the fix.
    assert!(
        msg.contains("right converted GGUF"),
        "panic message must carry the 'right converted GGUF' guidance \
         so an operator knows the fix is 'point at the correct GGUF'; \
         got: {msg}"
    );
}

// --- assert_tensor_count_positive --------------------------------------

/// `assert_tensor_count_positive` MUST panic with the 'no float tensors
/// passed through' guidance when the GGUF has metadata only (zero
/// tensors) (gap #18). The guidance block is the difference between a
/// triageable failure and an inscrutable one: it points the operator at
/// the exact converter arm (the BF16 pass-through path) that would have
/// silently emitted a metadata-only GGUF.
#[test]
fn assert_tensor_count_positive_panics_on_zero_tensors() {
    let bytes = build_metadata_only_gguf("irodori-tts");
    let file = GgufFile::parse(bytes).expect("parse metadata-only gguf");
    // Precondition — the file really has 0 tensors (guards against a
    // future regression that auto-injects a placeholder tensor).
    assert_eq!(
        file.tensors().len(),
        0,
        "test precondition: metadata-only GGUF must actually have 0 tensors"
    );
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_tensor_count_positive(&file, "irodori");
    }))
    .expect_err(
        "assert_tensor_count_positive must panic when the GGUF has zero \
         tensors (metadata-only GGUFs are a converter-arm failure, not a \
         'shape OK' state, FR-EX-08)",
    );
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("irodori"),
        "panic message must carry the ctx label so an operator with \
         multiple gated legs can identify the failing one; got: {msg}"
    );
    assert!(
        msg.contains("zero tensors"),
        "panic message must name the actual failure mode ('zero tensors') \
         so an operator immediately understands the shape; got: {msg}"
    );
    assert!(
        msg.contains("no float tensors passed through"),
        "panic message must quote the converter rustdoc arm ('no float \
         tensors passed through') so an operator can grep the converter \
         source for the fix; got: {msg}"
    );
    assert!(
        msg.contains("streaming BF16 pass-through path"),
        "panic message must point at the 'streaming BF16 pass-through \
         path' so an operator sees the actionable next step; got: {msg}"
    );
}
