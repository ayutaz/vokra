#!/usr/bin/env -S uv run --script
"""Stdlib-only regression for the production MOSS API-smoke bridge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def copy_project_fixture(source: Path, destination: Path) -> None:
    """Copy only the authenticated three-file project closure into a fixture."""
    destination.mkdir()
    for name in ("pyproject.toml", "uv.lock", "api_smoke.py"):
        shutil.copy2(source / name, destination / name, follow_symlinks=False)


def synthetic_evidence(gate: Any, rows: list[dict[str, Any]], project_sha: str, lock_sha: str) -> dict[str, Any]:
    selected = ["4b"]
    source = {
        "repo": gate.API_SMOKE_SOURCE_REPO,
        "revision": gate.SOURCE_IDENTITY["revision"],
        "files": {
            path: {"sha256": digest, "bytes": 1}
            for path, digest in gate.API_SMOKE_SOURCE_FILES.items()
        },
    }
    variants = {}
    for variant in selected:
        identity = gate.VARIANTS[variant]
        variants[variant] = {
            "repo": identity["repo"],
            "revision": identity["revision"],
            "model_type": "moss_audio",
            "files": {
                name: {"sha256": expected[variant], "bytes": 1}
                for name, expected in gate.API_SMOKE_METADATA_FILES.items()
            },
        }
    package_reviews = [
        {
            "name": row["name"],
            "version": row["version"],
            "source": copy.deepcopy(row["source"]),
            "status": "REVIEWED",
            "license": "SELF_TEST",
            "native_review": "SELF_TEST",
            "bundled_review": "SELF_TEST",
        }
        for row in rows
    ]
    license_reviews = {
        label: {"status": "REVIEWED", "spdx": "SELF_TEST", "evidence_sha256": "0" * 64}
        for label in ("source", "4b", "8b")
    }
    scope = {
        "expected_head": "0" * 40,
        "variants": selected,
        "source_repo": gate.API_SMOKE_SOURCE_REPO,
        "source_revision": gate.SOURCE_IDENTITY["revision"],
        "project_sha256": project_sha,
        "lock_sha256": lock_sha,
        "package_rows_sha256": canonical(rows),
        "package_reviews": package_reviews,
        "license_reviews": license_reviews,
    }
    api_record = {
        "transformers": "5.10.4",
        "config_class": "src.configuration_moss_audio.MossAudioConfig",
        "model_class": "src.modeling_moss_audio.MossAudioModel",
        "processor_class": "src.processing_moss_audio.MossAudioProcessor",
        "config_signature": "synthetic",
        "model_signature": "synthetic",
        "processor_signature": "synthetic",
        "processor_from_pretrained_signature": "synthetic",
        "config_construction": "PASS",
        "processor_construction": "PASS",
        "checkpoint_load": "NOT_PERFORMED",
    }
    return {
        "format": gate.API_SMOKE_FORMAT,
        "status": "PASS",
        "publication": "NO_UPLOAD",
        "expected_head": "0" * 40,
        "approval_signer": "self-test",
        "approval_scope_sha256": canonical(scope),
        "approval_scope": scope,
        "source": source,
        "variants": variants,
        "project": {
            "sha256": project_sha,
            "lock_sha256": lock_sha,
            "package_rows_sha256": canonical(rows),
            "packages": rows,
        },
        "api": {"4b": api_record},
        "checkpoint_load": "NOT_PERFORMED",
        "environment": {"python": "3.12", "platform": "synthetic-fixture"},
    }


def refresh_scope(evidence: dict[str, Any]) -> None:
    rows = evidence["project"]["packages"]
    rows_sha = canonical(rows)
    evidence["project"]["package_rows_sha256"] = rows_sha
    evidence["approval_scope"]["package_rows_sha256"] = rows_sha
    evidence["approval_scope_sha256"] = canonical(evidence["approval_scope"])


def validate_fixture(gate: Any, evidence: dict[str, Any], directory: Path, project: Path) -> tuple[bool, str]:
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return gate.validate_api_smoke_evidence(
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "0" * 40,
        ["4b"],
        authoritative_project=project,
    )


def self_test() -> int:
    gate = load_module("moss_audio_preflight_gate", ROOT / "preflight_gate.py")
    api = load_module("moss_audio_api_smoke", ROOT / "api_smoke" / "api_smoke.py")
    project = ROOT / "api_smoke"
    project_ok, project_reason = gate.validate_api_smoke_project(project)
    if not project_ok:
        raise AssertionError(f"current API-smoke project rejected: {project_reason}")
    rows, project_sha, lock_sha = api.verify_project(project)
    if len(rows) != 44 or project_sha != gate.API_SMOKE_PROJECT_SHA256 or lock_sha != gate.API_SMOKE_LOCK_SHA256:
        raise AssertionError("current API-smoke identity/row baseline drifted")
    with tempfile.TemporaryDirectory(prefix="moss-api-preflight-binding-") as temporary:
        directory = Path(temporary)
        stale_lock_project = directory / "stale-lock-project"
        copy_project_fixture(project, stale_lock_project)
        (stale_lock_project / "uv.lock").write_text((stale_lock_project / "uv.lock").read_text(encoding="utf-8") + "\n", encoding="utf-8")
        if gate.validate_api_smoke_project(stale_lock_project)[0]:
            raise AssertionError("stale API-smoke lock was accepted")
        stale_project_project = directory / "stale-project-project"
        copy_project_fixture(project, stale_project_project)
        (stale_project_project / "pyproject.toml").write_text((stale_project_project / "pyproject.toml").read_text(encoding="utf-8") + "\n", encoding="utf-8")
        if gate.validate_api_smoke_project(stale_project_project)[0]:
            raise AssertionError("stale API-smoke project was accepted")
        forged_project = directory / "forged-verifier-project"
        copy_project_fixture(project, forged_project)
        sentinel = directory / "forged-verifier-executed"
        (forged_project / "api_smoke.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n"
            "def verify_project(project):\n"
            "    return [], '0' * 64, '0' * 64\n",
            encoding="utf-8",
        )
        forged_ok, forged_reason = gate.validate_api_smoke_project(forged_project)
        if forged_ok or "trusted parser" not in forged_reason or sentinel.exists():
            raise AssertionError("forged API-smoke verifier was executed or accepted")
        baseline = synthetic_evidence(gate, rows, project_sha, lock_sha)
        ok, reason = validate_fixture(gate, baseline, directory, project)
        if not ok or reason != "PASS":
            raise AssertionError(f"synthetic current evidence was rejected: {reason}")

        stale_bridge = copy.deepcopy(baseline)
        stale_bridge["approval_scope"]["lock_sha256"] = "fd8c1f2342da9512d1fb1e97ed0e7640ffa4a83db279aeaba655ce6cf89c1085"
        stale_bridge["project"]["lock_sha256"] = stale_bridge["approval_scope"]["lock_sha256"]
        refresh_scope(stale_bridge)
        if validate_fixture(gate, stale_bridge, directory, project)[0]:
            raise AssertionError("old bridge lock scope was accepted")

        stale_project = copy.deepcopy(baseline)
        stale_project["approval_scope"]["project_sha256"] = "af7486c52182a23cc64185e6ad95c42b8863673dc4e032fda79637a9ca0d427e"
        stale_project["project"]["sha256"] = stale_project["approval_scope"]["project_sha256"]
        refresh_scope(stale_project)
        if validate_fixture(gate, stale_project, directory, project)[0]:
            raise AssertionError("old bridge project scope was accepted")

        stale_rows_digest = copy.deepcopy(baseline)
        stale_rows_digest["project"]["package_rows_sha256"] = "0" * 64
        if validate_fixture(gate, stale_rows_digest, directory, project)[0]:
            raise AssertionError("stale project package-row digest was accepted")

        stale_scope_digest = copy.deepcopy(baseline)
        stale_scope_digest["approval_scope_sha256"] = "0" * 64
        if validate_fixture(gate, stale_scope_digest, directory, project)[0]:
            raise AssertionError("stale approval-scope digest was accepted")

        row_tamper = copy.deepcopy(baseline)
        row_tamper["project"]["packages"][0]["version"] = "9.9.9"
        row_tamper["approval_scope"]["package_reviews"][0]["version"] = "9.9.9"
        refresh_scope(row_tamper)
        ok, reason = validate_fixture(gate, row_tamper, directory, project)
        if ok or "authenticated lock" not in reason:
            raise AssertionError("self-consistent but lock-divergent row was accepted")

        omitted = copy.deepcopy(baseline)
        omitted["project"]["packages"].pop()
        omitted["approval_scope"]["package_reviews"].pop()
        refresh_scope(omitted)
        if validate_fixture(gate, omitted, directory, project)[0]:
            raise AssertionError("omitted authoritative row was accepted")

        duplicate = copy.deepcopy(baseline)
        duplicate["project"]["packages"].append(copy.deepcopy(duplicate["project"]["packages"][0]))
        duplicate["approval_scope"]["package_reviews"].append(copy.deepcopy(duplicate["approval_scope"]["package_reviews"][0]))
        refresh_scope(duplicate)
        if validate_fixture(gate, duplicate, directory, project)[0]:
            raise AssertionError("duplicate package row was accepted")
    print("moss_audio API-smoke preflight binding self-test PASS (stdlib-only, fixture evidence)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test is required")
    raise SystemExit(self_test())
