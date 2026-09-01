"""Fail-closed dependency/license preflight for the LiteRT oracle.

This is deliberately reference-only. It audits the locked dependency graph and
wheel identities but does not import LiteRT or inspect model bytes. The current
closure remains BLOCKED_UNREVIEWED_TRANSITIVE until an owner records bounded
primary-source license evidence for every transitive row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

EXPECTED_DEPENDENCIES = {"ai-edge-litert": "==2.2.0", "numpy": "==2.5.2"}
EXPECTED_VERSIONS = {
    "ai-edge-litert": "2.2.0",
    "backports-strenum": "1.3.1",
    "flatbuffers": "25.12.19",
    "ml-dtypes": "0.6.0",
    "numpy": "2.5.2",
    "protobuf": "7.36.1",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
    "vokra-microwakeword-reference": "0.1.0",
}
EXPECTED_AI_EDGE_WHEEL_SHA256 = "sha256:4e151f07229b2f714f9b34ea42a9cacffc98953b1f6e832adee4479f6b81f50a"
# This is the selected CPython 3.12 manylinux x86_64 wheel digest in uv.lock.
EXPECTED_NUMPY_WHEEL_SHA256 = "3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8"
EXPECTED_ROWS = frozenset(EXPECTED_VERSIONS)
EXPECTED_PROJECT_SHA256 = "2b114885d54470c8397528b37572e3632202ca0b9d65ac349ec7e7da4e331f03"
EXPECTED_LOCK_SHA256 = "da75839f6195c27c32a15f097a40450c18b317ad78e9036ec2a1618472b85555"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _wheel_hashes(row: dict[str, Any]) -> list[str]:
    return [str(wheel.get("hash", "")) for wheel in row.get("wheels", [])]


def inspect_documents(project_bytes: bytes, lock_bytes: bytes, *, enforce_file_digests: bool = True) -> dict[str, Any]:
    project = tomllib.loads(project_bytes.decode("utf-8"))
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    project_data = project.get("project", {})
    dependencies = {}
    for item in project_data.get("dependencies", []):
        name, specifier = item.split("==", 1)
        dependencies[name] = f"=={specifier}"
    failures: list[str] = []
    if dependencies != EXPECTED_DEPENDENCIES:
        failures.append("direct dependency set/version drift")
    if project_data.get("requires-python") != "==3.12.*":
        failures.append("Python 3.12 contract drift")
    packages = lock.get("package", [])
    names = [row.get("name") for row in packages if isinstance(row, dict)]
    duplicate_rows = sorted({name for name in names if names.count(name) > 1 and isinstance(name, str)})
    if duplicate_rows:
        failures.append(f"duplicate lock package rows: {duplicate_rows}")
    rows = {
        row.get("name"): row
        for row in packages
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    if set(rows) != EXPECTED_ROWS:
        failures.append(f"unknown/missing lock rows: {sorted(set(rows) ^ EXPECTED_ROWS)}")
    for name, version in EXPECTED_VERSIONS.items():
        if rows.get(name, {}).get("version") != version:
            failures.append(f"{name} version drift")
    if enforce_file_digests:
        # These bind the audited project/lock bytes, preventing a caller from
        # self-stamping a changed dependency graph. Update only with owner
        # review after a new lock is independently audited.
        if sha256_bytes(project_bytes) != EXPECTED_PROJECT_SHA256:
            failures.append("pyproject digest drift")
        if sha256_bytes(lock_bytes) != EXPECTED_LOCK_SHA256:
            failures.append("uv.lock digest drift")
    ai_edge = rows.get("ai-edge-litert", {})
    ai_wheels = ai_edge.get("wheels", [])
    if len(ai_wheels) != 1 or _wheel_hashes(ai_edge) != [EXPECTED_AI_EDGE_WHEEL_SHA256]:
        failures.append("ai-edge-litert wheel identity drift")
    numpy_wheels = rows.get("numpy", {}).get("wheels", [])
    if not any(wheel.get("hash") == f"sha256:{EXPECTED_NUMPY_WHEEL_SHA256}" for wheel in numpy_wheels):
        failures.append("numpy selected wheel identity drift")
    # This is intentionally not a guessed license classification. Every row
    # needs bounded primary evidence, including native/bundled payloads.
    unreviewed = sorted(EXPECTED_ROWS - {"vokra-microwakeword-reference"})
    if unreviewed:
        failures.append("BLOCKED_UNREVIEWED_TRANSITIVE: " + ",".join(unreviewed))
    if "tqdm" in unreviewed:
        failures.append("tqdm bounded primary-source license evidence not recorded")
    return {
        "schema": "microwakeword-reference-dependency-audit-v1",
        "status": "BLOCKED_UNREVIEWED_TRANSITIVE" if unreviewed or failures else "PASS",
        "publication_permitted": False,
        "fixture_generation_permitted": not unreviewed and not failures,
        "project_sha256": sha256_bytes(project_bytes),
        "uv_lock_sha256": sha256_bytes(lock_bytes),
        "locked_rows": sorted(name for name in names if isinstance(name, str)),
        "expected_rows": sorted(EXPECTED_ROWS),
        "locked_row_count": len(packages),
        "duplicate_rows": duplicate_rows,
        "unreviewed_rows": unreviewed,
        "license_review": {
            "status": "BLOCKED_UNREVIEWED_TRANSITIVE" if unreviewed else "PASS",
            "bounded_primary_source_evidence_recorded": not unreviewed,
            "rows_requiring_owner_evidence": unreviewed,
        },
        "failures": failures,
    }


def self_test() -> int:
    root = Path(__file__).parent
    project_bytes = (root / "pyproject.toml").read_bytes()
    lock_bytes = (root / "uv.lock").read_bytes()
    report = inspect_documents(project_bytes, lock_bytes)
    assert report["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert not report["fixture_generation_permitted"]
    tampered = lock_bytes.replace(b'version = "2.2.0"', b'version = "9.9.9"', 1)
    tampered_report = inspect_documents(project_bytes, tampered, enforce_file_digests=False)
    assert tampered_report["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert "ai-edge-litert version drift" in tampered_report["failures"]
    unknown = lock_bytes.replace(b'[[package]]\nname = "tqdm"', b'[[package]]\nname = "unknown-row"', 1)
    assert any("unknown/missing lock rows" in failure for failure in inspect_documents(project_bytes, unknown, enforce_file_digests=False)["failures"])
    duplicate = lock_bytes.replace(b'[[package]]\nname = "tqdm"', b'[[package]]\nname = "numpy"', 1)
    duplicate_report = inspect_documents(project_bytes, duplicate, enforce_file_digests=False)
    assert any("duplicate lock package rows" in failure for failure in duplicate_report["failures"])
    with tempfile.TemporaryDirectory(prefix="microwakeword-inspector-") as temporary:
        output = Path(temporary) / "audit.json"
        write_exclusive(output, "{}\n")
        try:
            write_exclusive(output, "{}\n")
        except SystemExit:
            pass
        else:
            raise AssertionError("inspector output clobber was accepted")
        target = Path(temporary) / "target.json"
        target.write_text("existing\n", encoding="utf-8")
        link = Path(temporary) / "output-link.json"
        try:
            link.symlink_to(target)
        except OSError:
            pass
        else:
            try:
                write_exclusive(link, "{}\n")
            except SystemExit:
                pass
            else:
                raise AssertionError("inspector symlink output was accepted")
    print("microWakeWord dependency inspector self-test: PASS (blocked fail-closed)", file=sys.stderr)
    return 0


def require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"input must be an existing regular non-symlink file: {path}")


def write_exclusive(path: Path, payload: str) -> None:
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SystemExit(f"output parent must be an existing real directory: {parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {path}") from error
    created = os.fstat(fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            fd = -1
            output.write(payload)
    except BaseException:
        # Never unlink a path that was replaced after our exclusive create.
        # lstat identity, not pathname equality, is the ownership proof.
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
            path.unlink()
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path, default=Path(__file__).parent / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=Path(__file__).parent / "uv.lock")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.project != Path(__file__).parent / "pyproject.toml" or args.lock != Path(__file__).parent / "uv.lock" or args.output is not None:
            parser.error("--self-test cannot be combined with paths/output")
        return self_test()
    require_regular_file(args.project)
    require_regular_file(args.lock)
    if args.output is not None and args.output.is_symlink():
        raise SystemExit(f"refusing symlink output: {args.output}")
    report = inspect_documents(args.project.read_bytes(), args.lock.read_bytes())
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        write_exclusive(args.output, payload)
    print(payload, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
