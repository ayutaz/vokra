#!/usr/bin/env python3
"""test_package_managers.py — oracle for the 4 package-manager generators
(X-07-T28).

Two-stage design (fabricated-pass forbidden):

  PART (a) — CC-green NOW:
    * all 4 manager manifest sets render from templates;
    * one shared @@VERSION@@ is consistent across every manifest;
    * the generator HASHES the referenced assets (drift-detectable): the
      embedded checksum equals sha256/sha512 of a scratch FIXTURE asset;
      mutating the fixture WITHOUT re-rendering makes the embedded value stop
      matching (drift caught); re-rendering restores the match;
    * every one of the 4 managers references the SBOM (@@SBOM_URL@@).
    * absent assets yield an explicit PENDING placeholder, never a fake hash.

  PART (b) — HONEST DEFER (T32-gated):
    * checksum match against the REAL Release assets is only possible after the
      M4-11-T13 gap-flow decision (T32) enables T29/T30 and the owner tags. The
      vcpkg + Homebrew paths are source-tarball based, so their real-asset check
      is reachable from the tag alone; winget + Debian reference the T32-gated
      desktop binaries. This part is announced, NOT run — pinning a fixture as
      "the real asset" would be the exact fabricated pass this spec forbids.

Zero-dep (NFR-DS-02): python3 stdlib only + the generator.
Usage: python3 tools/release/test_package_managers.py
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GEN = os.path.join(ROOT, "tools", "release", "gen_package_managers.py")

VERSION = "1.0.0-rc.1"
# logical asset -> (release filename, fixture content)
FIXTURES = {
    "src": (f"vokra-{VERSION}-src.tar.gz", b"fixture source tarball content"),
    "cli-windows": (f"vokra-cli-{VERSION}-x86_64-windows.zip", b"fixture windows cli"),
    "lib-linux": (f"libvokra-{VERSION}-x86_64-linux.so", b"fixture linux .so"),
    "cli-linux": (f"vokra-cli-{VERSION}-x86_64-linux", b"fixture linux cli"),
}

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


def digest(data: bytes, algo: str) -> str:
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def write_fixtures(assets_dir: str) -> None:
    os.makedirs(assets_dir, exist_ok=True)
    for _logical, (fname, content) in FIXTURES.items():
        with open(os.path.join(assets_dir, fname), "wb") as fh:
            fh.write(content)


def run_gen(out_dir: str, assets_dir: str | None) -> int:
    cmd = ["python3", GEN, "--version", VERSION, "--out-dir", out_dir]
    if assets_dir:
        cmd += ["--assets-dir", assets_dir]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        bad(f"generator failed: {p.stderr.strip()[:200]}")
    return p.returncode


def read(out_dir: str, rel: str) -> str:
    return open(os.path.join(out_dir, rel), encoding="utf-8").read()


def part_a(scratch: str) -> None:
    assets = os.path.join(scratch, "assets")
    out = os.path.join(scratch, "out")
    write_fixtures(assets)
    if run_gen(out, assets) != 0:
        return

    expected = {
        "homebrew/vokra.rb",
        "winget/Vokra.Vokra.yaml",
        "winget/Vokra.Vokra.installer.yaml",
        "winget/Vokra.Vokra.locale.en-US.yaml",
        "vcpkg/ports/vokra/portfile.cmake",
        "vcpkg/ports/vokra/vcpkg.json",
        "debian/control",
        "debian/rules",
        "debian/changelog",
        "debian/copyright",
    }
    missing = [f for f in expected if not os.path.isfile(os.path.join(out, f))]
    if not missing:
        ok(f"(a) all {len(expected)} manager manifests rendered")
    else:
        bad(f"(a) missing rendered manifests: {missing}")

    # version consistency across the 4 managers.
    manager_files = {
        "homebrew": "homebrew/vokra.rb",
        "winget": "winget/Vokra.Vokra.installer.yaml",
        "vcpkg": "vcpkg/ports/vokra/vcpkg.json",
        "debian": "debian/changelog",
    }
    if all(VERSION in read(out, f) for f in manager_files.values()):
        ok(f"(a) version {VERSION} consistent across all 4 managers")
    else:
        bad("(a) version not consistent across managers")

    # checksum == fixture hash (generator really hashes the asset).
    src_sha256 = digest(FIXTURES["src"][1], "sha256")
    src_sha512 = digest(FIXTURES["src"][1], "sha512")
    win_sha256 = digest(FIXTURES["cli-windows"][1], "sha256")
    lib_sha256 = digest(FIXTURES["lib-linux"][1], "sha256")
    cli_sha256 = digest(FIXTURES["cli-linux"][1], "sha256")

    checks = [
        ("homebrew src sha256", src_sha256, read(out, "homebrew/vokra.rb")),
        ("vcpkg src sha512", src_sha512, read(out, "vcpkg/ports/vokra/portfile.cmake")),
        ("winget cli sha256", win_sha256, read(out, "winget/Vokra.Vokra.installer.yaml")),
        ("debian lib sha256", lib_sha256, read(out, "debian/rules")),
        ("debian cli sha256", cli_sha256, read(out, "debian/rules")),
    ]
    all_embedded = True
    for label, expected_hash, text in checks:
        if expected_hash not in text:
            all_embedded = False
            bad(f"(a) {label} not embedded (generator did not hash the asset)")
    if all_embedded:
        ok("(a) every checksum equals the fixture asset's real hash (generator hashes assets)")

    # SBOM reference in all 4 managers.
    sbom_ok = all(
        "spdx.json" in read(out, f)
        for f in ["homebrew/vokra.rb", "winget/Vokra.Vokra.installer.yaml",
                  "vcpkg/ports/vokra/portfile.cmake", "debian/control"]
    )
    if sbom_ok:
        ok("(a) SBOM reference present in all 4 managers")
    else:
        bad("(a) a manager manifest is missing its SBOM reference")

    # DRIFT detection: mutate a fixture, the OLD embedded hash must stop
    # matching the new fixture; re-render restores it.
    old_hb = read(out, "homebrew/vokra.rb")
    with open(os.path.join(assets, FIXTURES["src"][0]), "wb") as fh:
        fh.write(b"MUTATED source tarball content")
    mutated_sha = digest(b"MUTATED source tarball content", "sha256")
    if mutated_sha not in old_hb:
        ok("(a) drift detectable: old manifest hash != mutated fixture hash")
    else:
        bad("(a) drift NOT detectable (hash coincidence?)")
    out2 = os.path.join(scratch, "out2")
    run_gen(out2, assets)
    if mutated_sha in read(out2, "homebrew/vokra.rb"):
        ok("(a) re-render picks up the mutated asset's new hash")
    else:
        bad("(a) re-render did not update the checksum")

    # Absent asset -> explicit PENDING placeholder (never a fabricated hash).
    out3 = os.path.join(scratch, "out3")
    run_gen(out3, None)  # no assets-dir at all
    if "PENDING-RELEASE" in read(out3, "homebrew/vokra.rb"):
        ok("(a) absent asset -> explicit PENDING placeholder (no fabricated hash)")
    else:
        bad("(a) absent asset did not yield a PENDING placeholder")


def part_b_defer() -> None:
    print("  ----")
    print("  DEFER (b): checksum match against the REAL Release assets is "
          "T32-gated (M4-11-T13 gap-flow + T29/T30 enable + owner tag). "
          "Pinning a fixture as 'the real asset' would be a fabricated pass — "
          "not run here (honest defer, see spec §T28(b)).")


def main() -> None:
    print("[X-07-T28] package-manager generators (structure + drift; real-asset match deferred)")
    with tempfile.TemporaryDirectory(prefix="x07-pm.") as scratch:
        part_a(scratch)
    part_b_defer()
    print()
    print(f"package-managers oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
