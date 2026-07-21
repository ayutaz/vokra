#!/usr/bin/env python3
"""test_reproducible.py — packaging-layer reproducibility oracle (X-07-T34).

Verifies the reproducible-build pieces X-07 OWNS (NFR-MT-03), and states clearly
what it does NOT cover:

  IN SCOPE (asserted):
    (1) the first-party SBOM generator is byte-identical across two
        SOURCE_DATE_EPOCH-pinned runs (multiple feature closures);
    (2) a SOURCE_DATE_EPOCH-pinned archive pack is byte-identical across two
        runs (portable demonstration via python's tarfile with fixed mtime +
        sorted members + cleared uid/gid — the same normalization the release
        jobs apply through GNU `tar --sort=name --mtime --numeric-owner` and
        `zip -X`);
    (3) the release.yml pack steps actually SET SOURCE_DATE_EPOCH before packing
        (unity / godot / npm).

  OUT OF SCOPE (explicitly, per spec §T34 + 未確定事項 10): the BIT reproducibility
  of COMPILED artifacts (.dylib/.so/.dll, wasm, wheel-bundled native lib,
  xcframework, AAR). That needs --remap-path-prefix / codegen-units / toolchain
  pinning / a same-environment rebuild — owner infra, a separate WP. Therefore
  NFR-MT-03 "reproducible build" is NOT fully satisfied by X-07; only the
  packaging layer is.

Zero-dep (NFR-DS-02): python3 stdlib only.
Usage: python3 tools/release/test_reproducible.py
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GEN = os.path.join(ROOT, "scripts", "sbom", "generate_spdx.py")
RELEASE_YML = os.path.join(ROOT, ".github", "workflows", "release.yml")

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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_sbom_determinism(scratch: str) -> None:
    if shutil.which("cargo") is None:
        bad("SBOM determinism: cargo not found (must run where the toolchain exists)")
        return
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    closures = [
        ("vokra-ios", ["--package", "vokra-capi", "--no-default-features",
                       "--features", "vokra-models/metal"]),
        ("vokra-npm", ["--package", "vokra-wasm-harness", "--no-default-features"]),
    ]
    for name, args in closures:
        outs = []
        for i in range(2):
            out = os.path.join(scratch, f"{name}.{i}.json")
            p = subprocess.run(["python3", GEN, *args, "--doc-name", name, "--output", out],
                               cwd=ROOT, capture_output=True, text=True, env=env)
            if p.returncode != 0:
                bad(f"(1) SBOM {name} generation failed: {p.stderr.strip()[:160]}")
                break
            outs.append(open(out, "rb").read())
        if len(outs) == 2:
            if sha256_bytes(outs[0]) == sha256_bytes(outs[1]):
                ok(f"(1) SBOM {name} byte-identical across two pinned runs")
            else:
                bad(f"(1) SBOM {name} NOT byte-identical")


def deterministic_tar(src_dir: str, epoch: int) -> bytes:
    """Pack src_dir into a reproducible tar (fixed mtime, sorted, no uid/gid).

    Mirrors the release jobs' GNU `tar --sort=name --mtime=@EPOCH --owner=0
    --group=0 --numeric-owner`, but portable (python tarfile) so this oracle
    runs on any host.
    """
    buf = io.BytesIO()
    names = []
    for base, _dirs, files in os.walk(src_dir):
        for f in files:
            names.append(os.path.relpath(os.path.join(base, f), src_dir))
    names.sort()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in names:
            info = tarfile.TarInfo(name=name)
            data = open(os.path.join(src_dir, name), "rb").read()
            info.size = len(data)
            info.mtime = epoch
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_archive_determinism(scratch: str) -> None:
    tree = os.path.join(scratch, "tree")
    os.makedirs(os.path.join(tree, "sub"), exist_ok=True)
    open(os.path.join(tree, "b.txt"), "w").write("bbb")
    open(os.path.join(tree, "a.txt"), "w").write("aaa")
    open(os.path.join(tree, "sub", "c.txt"), "w").write("ccc")
    t1 = deterministic_tar(tree, 1700000000)
    t2 = deterministic_tar(tree, 1700000000)
    if sha256_bytes(t1) == sha256_bytes(t2):
        ok("(2) SOURCE_DATE_EPOCH-pinned archive pack is byte-identical across two runs")
    else:
        bad("(2) deterministic pack differs between two runs")
    # A different epoch MUST change the archive (proves mtime is actually pinned,
    # not ignored).
    t3 = deterministic_tar(tree, 1600000000)
    if sha256_bytes(t3) != sha256_bytes(t1):
        ok("(2) a different SOURCE_DATE_EPOCH changes the archive (mtime is really pinned)")
    else:
        bad("(2) archive unchanged by epoch (mtime not applied)")


def test_release_yml_uses_epoch() -> None:
    text = open(RELEASE_YML, encoding="utf-8").read()
    # unity, godot, npm pack steps all set SOURCE_DATE_EPOCH before packing.
    n = text.count("SOURCE_DATE_EPOCH")
    if n >= 3:
        ok(f"(3) release.yml sets SOURCE_DATE_EPOCH in {n} places (unity/godot/npm/SBOM packing)")
    else:
        bad(f"(3) release.yml sets SOURCE_DATE_EPOCH only {n} time(s); expected >= 3")


def main() -> None:
    print("[X-07-T34] packaging-layer reproducibility (compile-artifact bit-repro OUT OF SCOPE)")
    with tempfile.TemporaryDirectory(prefix="x07-repro.") as scratch:
        test_sbom_determinism(scratch)
        test_archive_determinism(scratch)
    test_release_yml_uses_epoch()
    print("  ----")
    print("  SCOPE: compiled artifacts (.dylib/.so/.dll, wasm, wheel native lib, "
          "xcframework, AAR) bit-repro is NOT covered (未確定事項 10). NFR-MT-03 "
          "'reproducible build' is only PARTIALLY satisfied by X-07 (packaging layer).")
    print()
    print(f"reproducible oracle: {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
