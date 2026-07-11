// package_licenses.rs — structural sanity checks for the M3-11 AssetLib
// package LICENSE + NOTICE files that this crate now ships from its own
// root (integrations/vokra-godot/LICENSE + NOTICE).
//
// Why these files exist as crate-local files:
//   scripts/build-godot-gdextension.sh:253-261 walks
//       for f in LICENSE NOTICE README.md; do
//           if [ -f "$GODOT_CRATE/$f" ]; then
//               cp -f "$GODOT_CRATE/$f" "$ADDONS_DIR/$f"
//           elif [ -f "$ROOT/$f" ]; then
//               cp -f "$ROOT/$f" "$ADDONS_DIR/$f"
//           else
//               echo "build-godot-gdextension: WARN $f not found …" >&2
//           fi
//       done
//   — crate-local wins over the repo-root fallback. Landing explicit
//   crate-local LICENSE + NOTICE gives the AssetLib package a
//   self-contained set of legal files (dep NOTICE roll-up + Godot MIT
//   credit) instead of the generic repo-root NOTICE that documents the
//   broader Vokra project's cross-cutting decisions.
//
//   The M3-11-T18 compliance scanner
//   (scripts/compliance/check-godot-package-no-nvidia.sh, tier 4) fails
//   any release build where addons/vokra/LICENSE or addons/vokra/NOTICE
//   is missing or zero-byte, so these files are a hard release
//   requirement, not a nice-to-have.
//
// Zero-dep unchanged (NFR-DS-02): pure `std::fs` reads under
// `CARGO_MANIFEST_DIR`; no new crate added.

use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn repo_root() -> PathBuf {
    // integrations/vokra-godot -> integrations -> <repo root>
    crate_root().join("..").join("..")
}

fn read_to_string(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()))
}

// ---- LICENSE ----------------------------------------------------------------

#[test]
fn license_is_present_and_non_empty() {
    let license = crate_root().join("LICENSE");
    assert!(
        license.exists(),
        "integrations/vokra-godot/LICENSE missing — the M3-11-T18 scanner \
         (scripts/compliance/check-godot-package-no-nvidia.sh, tier 4) \
         requires this file, and build-godot-gdextension.sh will fall back \
         to the repo-root generic LICENSE with only a WARN — that fallback \
         no longer applies once this file lands",
    );
    let src = read_to_string(&license);
    assert!(
        !src.trim().is_empty(),
        "integrations/vokra-godot/LICENSE is empty — a zero-byte LICENSE \
         fails the M3-11-T18 scanner's `-s` non-empty check and would make \
         the AssetLib package license-less",
    );
}

#[test]
fn license_is_apache_2_0_verbatim() {
    let src = read_to_string(&crate_root().join("LICENSE"));
    // Canonical Apache-2.0 header lines. If any one is missing, the file
    // is not the standard Apache-2.0 body and would fail an SPDX
    // audit.
    assert!(
        src.contains("Apache License"),
        "LICENSE missing `Apache License` header — must be a verbatim \
         Apache-2.0 body (workspace-wide policy, see ../../LICENSE)",
    );
    assert!(
        src.contains("Version 2.0, January 2004"),
        "LICENSE missing `Version 2.0, January 2004` — must be verbatim \
         Apache-2.0",
    );
    assert!(
        src.contains("http://www.apache.org/licenses/"),
        "LICENSE missing the canonical Apache-2.0 URL",
    );
    assert!(
        src.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"),
        "LICENSE missing the Apache-2.0 terms header — the file was \
         truncated or replaced with a non-standard body",
    );
    assert!(
        src.contains("END OF TERMS AND CONDITIONS"),
        "LICENSE missing the Apache-2.0 terms footer — the file was \
         truncated mid-body",
    );
}

#[test]
fn license_matches_repo_root_verbatim() {
    // The crate-local LICENSE MUST be a byte-for-byte copy of the repo-root
    // Apache-2.0 LICENSE. Divergence would mean the AssetLib package ships
    // a modified license text, which is a red flag for the M2-13
    // compliance gate. This is stronger than the "Apache License" string
    // check above and catches subtle edits (whitespace, typos).
    let crate_license = read_to_string(&crate_root().join("LICENSE"));
    let root_license = read_to_string(&repo_root().join("LICENSE"));
    assert_eq!(
        crate_license, root_license,
        "integrations/vokra-godot/LICENSE has diverged from the repo-root \
         Apache-2.0 LICENSE — the crate-local file must be a verbatim \
         copy",
    );
}

// ---- NOTICE -----------------------------------------------------------------

#[test]
fn notice_is_present_and_non_empty() {
    let notice = crate_root().join("NOTICE");
    assert!(
        notice.exists(),
        "integrations/vokra-godot/NOTICE missing — the M3-11-T18 scanner \
         (scripts/compliance/check-godot-package-no-nvidia.sh, tier 4) \
         requires this file per ADR-0011 §D9 (`依存 crate NOTICE 集約 + \
         Godot MIT`)",
    );
    let src = read_to_string(&notice);
    assert!(
        !src.trim().is_empty(),
        "integrations/vokra-godot/NOTICE is empty — a zero-byte NOTICE \
         fails the M3-11-T18 scanner's `-s` non-empty check",
    );
}

#[test]
fn notice_declares_the_package_and_apache_pointer() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // Header must identify the package by name and point at the sibling
    // LICENSE file so a consumer inspecting only NOTICE knows where the
    // full license text lives.
    assert!(
        src.contains("vokra-godot"),
        "NOTICE must identify itself as the vokra-godot package",
    );
    assert!(
        src.contains("Apache License, Version 2.0"),
        "NOTICE must point at the Apache-2.0 license so consumers can \
         locate the LICENSE body",
    );
}

#[test]
fn notice_credits_godot_engine_mit() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // ADR-0011 §D9 pins Godot MIT credit as a required NOTICE
    // component. The binding integrates with Godot but does NOT bundle
    // Godot itself; the credit must still be present because consumers'
    // Godot-loaded environment is the host runtime.
    assert!(
        src.contains("Godot Engine"),
        "NOTICE must reference `Godot Engine` (ADR-0011 §D9 pins Godot \
         MIT credit)",
    );
    assert!(
        src.contains("MIT"),
        "NOTICE must call out Godot's MIT license (ADR-0011 §D9)",
    );
    // MIT canonical clauses — a stub credit that just says "Godot is
    // MIT" would fail an SPDX audit; the license text must be present
    // in the NOTICE.
    assert!(
        src.contains("Permission is hereby granted"),
        "NOTICE Godot MIT block missing the canonical `Permission is \
         hereby granted` opening clause",
    );
    assert!(
        src.contains("THE SOFTWARE IS PROVIDED \"AS IS\""),
        "NOTICE Godot MIT block missing the canonical `AS IS` disclaimer",
    );
}

#[test]
fn notice_covers_nvidia_non_bundle_posture() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // The Godot AssetLib addons/ tree is a shared plugin location,
    // parallel to the Unity Plugins/ directory. The NVIDIA CUDA EULA
    // "private (non-shared) directory location" constraint applies
    // identically. The NOTICE must document the runtime-detection
    // strategy so downstream distributors know why cudart is absent.
    assert!(
        src.contains("NVIDIA CUDA runtime"),
        "NOTICE must document NVIDIA CUDA runtime non-bundle posture — \
         Godot addons/ is a shared plugin directory, same as Unity Plugins/",
    );
    assert!(
        src.contains("dlopen(\"libcuda.so\")"),
        "NOTICE must reference `dlopen(\"libcuda.so\")` runtime detection",
    );
    assert!(
        src.contains("LoadLibrary(\"nvcuda.dll\")"),
        "NOTICE must reference `LoadLibrary(\"nvcuda.dll\")` runtime \
         detection",
    );
    assert!(
        src.contains("EULA"),
        "NOTICE must invoke the NVIDIA CUDA EULA justification for the \
         non-bundle posture",
    );
    assert!(
        src.contains("private (non-shared) directory"),
        "NOTICE must quote the EULA `private (non-shared) directory` \
         constraint verbatim",
    );
    // FR-EX-08: no silent CPU fallback. The NOTICE must document this
    // so downstream consumers can't claim they weren't warned about the
    // hard-error path.
    assert!(
        src.contains("FR-EX-08"),
        "NOTICE must reference FR-EX-08 (no silent CPU fallback) so the \
         explicit-error contract is discoverable",
    );
}

#[test]
fn notice_references_the_compliance_scanner() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // The M3-11-T18 scanner is the release-time gate that validates this
    // very NOTICE. Cross-referencing it here forms a
    // documentation-level closed loop: consumers can trace from NOTICE
    // → scanner → CI job → release artifact.
    assert!(
        src.contains("check-godot-package-no-nvidia.sh"),
        "NOTICE must reference scripts/compliance/check-godot-package-no-nvidia.sh \
         (M3-11-T18 compliance scanner)",
    );
}

#[test]
fn notice_documents_mit_only_demo_policy() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // BR-10 / M2-13: the AssetLib demo channel is MIT-only. The NOTICE
    // must call this out because the fetch-demo-models.sh script is
    // shipped inside the package and its behavior is bounded by this
    // policy.
    assert!(
        src.contains("MIT-only"),
        "NOTICE must document the MIT-only demo weight policy (BR-10 / \
         M2-13)",
    );
    assert!(
        src.contains("CC-BY-NC"),
        "NOTICE must call out the CC-BY-NC exclusion — F5-TTS / \
         Fish-Speech / Bark / EnCodec weights are excluded from the \
         AssetLib channel",
    );
    assert!(
        src.contains("BR-08"),
        "NOTICE must reference BR-08 (voice-clone core-exclusion) — RVC / \
         GPT-SoVITS are categorically banned from Vokra core",
    );
}

#[test]
fn notice_points_at_repo_wide_provenance_docs() {
    let src = read_to_string(&crate_root().join("NOTICE"));
    // Consumers auditing the package need to be able to walk from this
    // crate-local NOTICE to the repo-wide provenance docs. The scanner
    // doesn't validate these references, but a broken cross-link would
    // mean the audit trail terminates at the AssetLib package.
    for pointer in [
        "docs/license-audit.md",
        "docs/legal-compliance.md",
        "third_party/NVIDIA-EULA.md",
    ] {
        assert!(
            src.contains(pointer),
            "NOTICE must point at `{pointer}` so consumers can walk to the \
             repo-wide provenance docs",
        );
    }
}

// ---- build-godot-gdextension.sh integration --------------------------------

#[test]
fn build_script_prefers_crate_local_license_notice_over_repo_root() {
    // The M3-11-T18 scanner assumes the LICENSE + NOTICE files copied
    // into addons/vokra/ come from THIS crate, not the repo root. The
    // build script's loop
    //   for f in LICENSE NOTICE README.md; do
    //       if [ -f "$GODOT_CRATE/$f" ]; then …
    //       elif [ -f "$ROOT/$f" ]; then …
    //   done
    // implements that precedence. If someone refactored the loop and
    // dropped the `$GODOT_CRATE/` branch, our crate-local files would be
    // silently ignored and the packaged NOTICE would revert to the
    // generic repo-root one — a change in behavior that the M3-11-T18
    // scanner would NOT catch (it only checks presence + non-empty,
    // not content lineage). Guard the precedence explicitly.
    let build_script = repo_root()
        .join("scripts")
        .join("build-godot-gdextension.sh");
    assert!(
        build_script.exists(),
        "scripts/build-godot-gdextension.sh missing — cannot verify \
         LICENSE / NOTICE precedence",
    );
    let src = read_to_string(&build_script);

    // The loop iterates LICENSE + NOTICE (+ README.md). Assert both are
    // covered.
    assert!(
        src.contains("for f in LICENSE NOTICE README.md"),
        "build-godot-gdextension.sh must iterate LICENSE + NOTICE + \
         README.md in a shared copy loop",
    );

    // Precedence: `$GODOT_CRATE/$f` MUST be tested first, `$ROOT/$f`
    // second. If the ordering flipped, the crate-local files would
    // never win. We search for the crate-local check position and the
    // repo-root check position and enforce the ordering.
    let crate_check = "if [ -f \"$GODOT_CRATE/$f\" ]; then";
    let root_check = "elif [ -f \"$ROOT/$f\" ]; then";
    let crate_pos = src.find(crate_check).expect(
        "build-godot-gdextension.sh must probe $GODOT_CRATE/$f first — \
         crate-local LICENSE / NOTICE precedence lost",
    );
    let root_pos = src.find(root_check).expect(
        "build-godot-gdextension.sh must fall back to $ROOT/$f second — \
         repo-root fallback branch missing",
    );
    assert!(
        crate_pos < root_pos,
        "build-godot-gdextension.sh must check $GODOT_CRATE/$f BEFORE \
         $ROOT/$f — crate-local precedence lost",
    );
}
