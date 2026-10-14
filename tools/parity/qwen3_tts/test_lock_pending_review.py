#!/usr/bin/env python3
"""Controlled stdlib-only regression for the current lock refresh.

This test calls ``license_gate.run`` against a temporary copy of the
non-model gate inputs.  It never imports torch/Qwen code, resolves packages,
downloads files, or touches the production evidence JSON.  The fixture is
intentionally expected to fail: the current lock must pass the first digest
check, then the historical dependency audit must refuse supersession.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parent
GATE_PATH = ROOT / "license_gate.py"


def load_gate():
    spec = importlib.util.spec_from_file_location("qwen3_tts_license_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load license_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"{path.name}:{name} must have one top-level assignment")
    value = ast.literal_eval(matches[0].value)
    if not isinstance(value, str):
        raise AssertionError(f"{path.name}:{name} is not a string")
    return value


def test_current_lock_then_historical_audit_rejects() -> None:
    gate = load_gate()
    lock = ROOT / "uv.lock"
    project = ROOT / "pyproject.toml"
    manifest = ROOT / "license_gate_manifest.json"
    audit = ROOT / "dependency_audit_evidence.json"
    operator_evidence = ROOT / "operator_approval_evidence.json"

    assert sha256(lock) == gate.LOCK_SHA256
    current_lock_sha256 = sha256(lock)
    for probe in (ROOT / "api_smoke.py", ROOT / "model_free_api_smoke.py"):
        assert literal_constant(probe, "LOCK_SHA256") == current_lock_sha256
    assert literal_constant(ROOT / "model_free_api_smoke.py", "PROJECT_SHA256") == sha256(project)
    assert literal_constant(ROOT / "api_smoke.py", "LOCK_SHA256") != "98cef03a391c9a31116b0b63c1dca4bb456c1a58ec407a4567e78e648eaf8857"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["lock_sha256"] == gate.LOCK_SHA256
    assert manifest_data["dependency_audit_evidence"]["status"] == "PENDING_OWNER_APPROVAL"

    lock_data = tomllib.loads(lock.read_text(encoding="utf-8"))
    reviews = gate.review_rows(gate.package_rows(lock_data), manifest_data)
    unresolved = {
        (row["name"], row["version"])
        for row in reviews
        if row["status"] == "BLOCKED_UNRESOLVED_REVIEW"
    }
    assert ("urllib3", "2.8.0") in unresolved
    assert {
        (row["name"], row["version"])
        for row in reviews
        if row["name"] in {"torch", "torchaudio"}
    } == {
        ("torch", "2.13.0"),
        ("torch", "2.13.0+cpu"),
        ("torchaudio", "2.11.0"),
        ("torchaudio", "2.11.0+cpu"),
    }
    assert all(
        row["status"] == "BLOCKED_UNRESOLVED_REVIEW"
        for row in reviews
        if row["name"] in {"torch", "torchaudio"}
    )

    with tempfile.TemporaryDirectory(prefix="qwen3-tts-current-lock-gate-") as directory:
        fixture = Path(directory)
        for source in (lock, project, manifest, audit, operator_evidence, ROOT / "dependency_audit.py"):
            shutil.copy2(source, fixture / source.name)

        stderr = io.StringIO()
        try:
            with redirect_stderr(stderr):
                gate.run(
                    fixture / lock.name,
                    fixture / manifest.name,
                    fixture / project.name,
                    fixture / operator_evidence.name,
                    gate.FIXED_IDENTITIES["official_source_revision"],
                    gate.FIXED_IDENTITIES["decoder_revision"],
                    gate.FIXED_IDENTITIES["decoder_checkpoint_sha256"],
                    {
                        slug: gate.FIXED_IDENTITIES["variants"][slug]["revision"]
                        for slug in gate.VARIANTS
                    },
                )
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("historical dependency audit unexpectedly passed")

        message = stderr.getvalue()
        assert "dependency audit" in message
        assert "uv.lock bytes drifted" not in message
        assert "real-weight" not in message.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stdlib-only Qwen3-TTS lock gate self-test")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("--self-test is required")
    test_current_lock_then_historical_audit_rejects()
    print("test_lock_pending_review: PASS (static current-lock ordering and fail-closed rejection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
