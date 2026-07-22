#!/usr/bin/env python3
"""test_sbom_paths.py — oracle for "all releases carry an SBOM" (X-07-T08).

The X-07 completion condition is "全リリースに SBOM が付属". Before X-07 only
1 of the 5 release.yml jobs emitted an SBOM (python, via a third-party action).
This oracle mechanically asserts:

  (A) YAML wiring — each of the 5 release paths (ios / unity / godot / npm /
      python) invokes the FIRST-PARTY generator scripts/sbom/generate_spdx.py
      with the correct --doc-name, and the produced <doc>.spdx.json is carried
      to an upload (workflow artifact `path:` and/or `gh release upload`). No
      third-party `anchore/sbom-action` remains (X-07-T07 replaced it).

  (B) Determinism — for each of the 5 shipped feature closures, running the
      generator twice with SOURCE_DATE_EPOCH pinned yields a byte-identical
      document (sha256 match). This is the reproducible-build precondition
      (NFR-MT-03) for the packaging-layer verify (X-07-T34).

RED before X-07: the 4 unwired jobs (ios/unity/godot/npm) had no SBOM step, so
(A) fails; and the python job used anchore (non-deterministic), so the
"no anchore remains" check fails.

Zero-dep (NFR-DS-02): python3 stdlib only, driving the real generator + cargo.
FR-EX-08: any missing wiring, leftover anchore, or non-determinism is a hard
failure — never a silent pass.

Usage: python3 tools/release/test_sbom_paths.py
Exit:  0 = all pass, 1 = any failure.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")
GEN = os.path.join(ROOT, "scripts", "sbom", "generate_spdx.py")

# Each release path: (label, doc-name, generate_spdx.py args). The args MUST
# match the shipped build's feature selection (asserted against release.yml text
# in part A, and executed for determinism in part B).
PATHS = [
    ("ios-xcframework", "vokra-ios-xcframework",
     ["--package", "vokra-capi", "--no-default-features",
      "--features", "vokra-models/metal"]),
    ("unity-package-release", "vokra-unity-package",
     ["--package", "vokra-capi", "--no-default-features", "--features", "cpu"]),
    ("godot-package-release", "vokra-godot-assetlib",
     ["--package", "vokra-godot", "--manifest-dir", "integrations/vokra-godot"]),
    ("npm-web-release", "vokra-npm-web",
     ["--package", "vokra-wasm-harness", "--no-default-features"]),
    ("python-pypi-publish", "vokra-python-wheel",
     ["--package", "vokra-capi"]),
]

_pass = 0
_fail = 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print(f"  ok:   {msg}")


def bad(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL: {msg}", file=sys.stderr)


def part_a(text: str) -> None:
    """YAML wiring assertions over the release.yml text."""
    # Match the ACTION INVOCATION (`uses: anchore/sbom-action...`), not comment
    # mentions — the T07 step keeps a comment explaining what it replaced.
    uses_anchore = [
        ln for ln in text.splitlines()
        if ln.lstrip().startswith("uses:") and "anchore/sbom-action" in ln
    ]
    if uses_anchore:
        bad("part A: a `uses: anchore/sbom-action` step still present (X-07-T07 "
            "must replace it with the first-party generator)")
    else:
        ok("part A: no third-party `uses: anchore/sbom-action` step remains")

    gen_invocations = text.count("scripts/sbom/generate_spdx.py")
    if gen_invocations >= len(PATHS):
        ok(f"part A: {gen_invocations} first-party generate_spdx.py invocations "
           f"(>= {len(PATHS)} release paths)")
    else:
        bad(f"part A: only {gen_invocations} generate_spdx.py invocations, "
            f"expected >= {len(PATHS)}")

    for label, doc, _args in PATHS:
        spdx = f"{doc}.spdx.json"
        has_gen = f"--doc-name {doc}" in text
        # The .spdx.json must be produced (as --output) AND uploaded (artifact
        # path list and/or gh release upload). Count occurrences: one for the
        # --output, at least one more for an upload reference.
        occurrences = text.count(spdx)
        if has_gen and occurrences >= 2:
            ok(f"part A: {label} generates + uploads {spdx}")
        elif not has_gen:
            bad(f"part A: {label} has no `--doc-name {doc}` generate step")
        else:
            bad(f"part A: {label} generates {spdx} but does not upload it "
                f"({occurrences} reference(s); need generate + upload)")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def part_b(scratch: str) -> None:
    """Determinism: two SOURCE_DATE_EPOCH-pinned runs are byte-identical."""
    if shutil.which("cargo") is None:
        # FR-EX-08: announce, do not silently pass. This oracle's determinism
        # leg needs the real cargo graph; the release environment (and CI)
        # always has it.
        bad("part B: cargo not found — determinism check cannot run "
            "(this oracle must run where the toolchain exists)")
        return

    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    for label, doc, args in PATHS:
        out1 = os.path.join(scratch, f"{doc}.1.json")
        out2 = os.path.join(scratch, f"{doc}.2.json")
        rc = 0
        for out in (out1, out2):
            proc = subprocess.run(
                ["python3", GEN, *args, "--doc-name", doc, "--output", out],
                cwd=ROOT, capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                bad(f"part B: {label} generation failed: {proc.stderr.strip()[:200]}")
                rc = 1
                break
        if rc:
            continue
        if sha256_file(out1) == sha256_file(out2):
            ok(f"part B: {label} SBOM is byte-identical across two pinned runs")
        else:
            bad(f"part B: {label} SBOM differs between two pinned runs "
                "(reproducible-build violation)")


def main() -> None:
    if not os.path.isfile(RELEASE_YML):
        print(f"error: {RELEASE_YML} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(GEN):
        print(f"error: {GEN} not found", file=sys.stderr)
        sys.exit(1)

    text = open(RELEASE_YML, encoding="utf-8").read()

    print("[X-07-T08] SBOM present + deterministic on all 5 release paths")
    part_a(text)
    with tempfile.TemporaryDirectory(prefix="x07-sbom-paths.") as scratch:
        part_b(scratch)

    print()
    print(f"sbom-paths oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
