#!/usr/bin/env python3
"""Stdlib-only dependency boundary for the AudioGen model-free audit.

AudioGen's official AudioCraft reference is not executable until a dedicated
Python lock and a primary-source license review exist.  This module records
that absence without resolving, installing, or importing any dependency.  A
future VAST run may consume the same report after an owner-approved lock is
added; this file deliberately cannot turn a missing lock into an allow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any


SCHEMA = "vokra-audiogen-medium-dependency-audit-v1"
PROJECT_NAME = "vokra-audiogen-medium-reference"
BLOCKED_STATUS = "BLOCKED_MISSING_LOCK_INPUTS"
REVIEW_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
REQUIRED_EVIDENCE = (
    "dedicated uv.lock with resolved Python 3.12 package/source identities",
    "primary-source license evidence for every resolved transitive package",
    "native-library and wheel-bundle license evidence",
    "owner approval of the complete dependency closure before model access",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_report(path: Path) -> dict[str, Any]:
    """Load a report through the same strict JSON consumer used by gates."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid AudioGen dependency report: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("AudioGen dependency report is not an object")
    return value


def load_project(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("AudioGen dependency project pyproject.toml is missing/non-regular")
    raw = path.read_bytes()
    try:
        value = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"AudioGen dependency project TOML is invalid: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("AudioGen dependency project TOML is not a mapping")
    return value, raw


def audit(project: Path) -> dict[str, Any]:
    metadata, pyproject_bytes = load_project(project / "pyproject.toml")
    project_meta = metadata.get("project")
    if not isinstance(project_meta, dict):
        raise RuntimeError("AudioGen dependency project metadata is missing")
    if project_meta.get("name") != PROJECT_NAME or project_meta.get("requires-python") != ">=3.12,<3.13":
        raise RuntimeError("AudioGen dependency project identity drift")
    dependencies = project_meta.get("dependencies", [])
    if dependencies != []:
        raise RuntimeError("model-free dependency project must not gain executable dependencies")
    lock = project / "uv.lock"
    lock_present = lock.is_file() and not lock.is_symlink()
    lock_sha256 = sha256_bytes(lock.read_bytes()) if lock_present else None
    status = REVIEW_STATUS if lock_present else BLOCKED_STATUS
    blockers = [] if lock_present else ["dedicated uv.lock is absent; no resolved dependency closure can be authenticated"]
    scope = {
        "schema": SCHEMA,
        "project": PROJECT_NAME,
        "python": "3.12",
        "pyproject_sha256": sha256_bytes(pyproject_bytes),
        "uv_lock_sha256": lock_sha256,
        "lock_status": "PRESENT_UNREVIEWED" if lock_present else "MISSING",
        "direct_dependencies": [],
        "required_evidence": list(REQUIRED_EVIDENCE),
        "execution": "NO_LOCAL_EXECUTION",
        "publication": "NO_UPLOAD",
    }
    return {
        "schema": SCHEMA,
        "status": status,
        "project": PROJECT_NAME,
        "python": "3.12",
        "pyproject_sha256": scope["pyproject_sha256"],
        "uv_lock_sha256": lock_sha256,
        "lock_present": lock_present,
        "direct_dependencies": [],
        "required_evidence": list(REQUIRED_EVIDENCE),
        "blockers": blockers,
        "scope": scope,
        "scope_sha256": sha256_bytes(canonical(scope)),
        "execution": "NO_LOCAL_EXECUTION",
        "publication": "NO_UPLOAD",
    }


def validate_report(report: dict[str, Any], project: Path) -> None:
    expected = audit(project)
    if report != expected:
        raise RuntimeError("AudioGen dependency audit report drift")
    if report["status"] != BLOCKED_STATUS or report["lock_present"] is not False:
        raise RuntimeError("AudioGen model-free dependency audit crossed the missing-lock boundary")
    if report["execution"] != "NO_LOCAL_EXECUTION" or report["publication"] != "NO_UPLOAD":
        raise RuntimeError("AudioGen dependency audit execution/publication boundary drift")


def write_report(output: Path, report: dict[str, Any]) -> None:
    if not output.is_absolute() or "\x00" in str(output) or any(part in {".", ".."} for part in output.parts):
        raise RuntimeError("AudioGen dependency report output must be absolute and have no dot components")
    if output.exists() or output.is_symlink():
        raise RuntimeError("refusing to clobber AudioGen dependency report")
    ancestor = output.parent
    while ancestor != ancestor.parent:
        if ancestor.is_symlink() and not (ancestor.parent == Path("/") and ancestor.name in {"tmp", "var"}):
            raise RuntimeError("AudioGen dependency report output has a symlink ancestor")
        ancestor = ancestor.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    try:
        os.link(temporary, output, follow_symlinks=False)
    except FileExistsError as error:
        raise RuntimeError("refusing to clobber AudioGen dependency report") from error
    finally:
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="audiogen-medium-dependency-") as directory:
        project = Path(directory)
        (project / "pyproject.toml").write_text(
            '[project]\nname = "vokra-audiogen-medium-reference"\nversion = "0.1.0"\nrequires-python = ">=3.12,<3.13"\ndependencies = []\n',
            encoding="utf-8",
        )
        report = audit(project)
        assert report["status"] == BLOCKED_STATUS
        assert report["lock_present"] is False
        assert report["scope"]["uv_lock_sha256"] is None
        validate_report(report, project)
        report_path = project / "report.json"
        write_report(report_path, report)
        assert load_report(report_path) == report
        try:
            write_report(report_path, report)
        except RuntimeError:
            pass
        else:
            raise AssertionError("existing dependency report was clobbered")
        tampered = json.loads(json.dumps(report))
        tampered["publication"] = "UPLOAD"
        try:
            validate_report(tampered, project)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dependency publication boundary drift was accepted")
        target = project / "target"
        target.mkdir()
        linked = project / "linked"
        linked.symlink_to(target, target_is_directory=True)
        try:
            write_report(linked / "report.json", report)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink-parent dependency report was accepted")
        try:
            write_report(project / "nested" / ".." / "dot.json", report)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dot-component dependency report path was accepted")
        duplicate = project / "duplicate.json"
        duplicate.write_text('{"scope_sha256":"a","scope_sha256":"b"}\n', encoding="utf-8")
        try:
            load_report(duplicate)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate dependency JSON key was accepted")
    print("audiogen_medium_dependency_audit --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None:
            parser.error("--self-test accepts no project/output")
        self_test()
        return 0
    if args.project is None or args.output is None:
        parser.error("project and --output are required unless --self-test is used")
    try:
        report = audit(args.project)
        write_report(args.output, report)
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(f"AudioGen dependency audit BLOCKED: {error}")
        return 2
    print(json.dumps({"status": report["status"], "scope_sha256": report["scope_sha256"], "publication": "NO_UPLOAD"}, sort_keys=True))
    return 2 if report["status"] != "AUDITED_ALLOW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
