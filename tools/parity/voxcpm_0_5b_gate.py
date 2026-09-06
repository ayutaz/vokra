"""Stdlib-only fail-closed gate shared by the VoxCPM-0.5B tools.

The current composite is inspection-only.  This module authenticates the
caller-bound blocker record and checkout before any tool reads model/source
inputs; it deliberately has no torch, transformers, or safetensors imports.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "openbmb/VoxCPM-0.5B"
MODEL_REVISION = "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REPOSITORY = "https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION = "38a76704ee67935ccbafbe5b6725e83dbb1e9305"
PUBLIC_REPOSITORY = "vokra/voxcpm-0.5b"
PUBLIC_REVISION = "ee0ca6d5b9fab27bbb626b5cb3f01236e582d004"
AUDIOVAE_SOURCE = "src/voxcpm/modules/audiovae/audio_vae.py"
TOKENIZER_FILES = ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise RuntimeError(f"duplicate approval key: {key}")
        result[key] = value
    return result


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _safe_existing_file(value: str) -> Path:
    parts = value.split("/")
    if (
        not value
        or not value.startswith("/")
        or "\\" in value
        or any(not part or part in {".", ".."} for part in parts[1:])
    ):
        raise RuntimeError("approval path must be absolute, lexical-clean, and non-symlinked")
    path = Path(value)
    current = Path("/")
    for part in value.split("/")[1:]:
        if not part:
            continue
        current /= part
        if current.is_symlink():
            raise RuntimeError("approval path has symlink ancestry")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("approval evidence must be a regular non-symlink file")
    return path


def _expected_scope(data: dict[str, Any]) -> str:
    scope = {key: value for key, value in data.items() if key != "scope_sha256"}
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_approval(raw: bytes, expected_head: str, supplied_sha: str) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != supplied_sha:
        raise RuntimeError("approval bytes changed or SHA-256 is wrong")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    keys = {
        "schema", "status", "disposition", "expected_head", "model_repository", "model_revision",
        "source_repository", "source_revision", "public_repository", "public_revision", "audio_vae_source",
        "tokenizer_files", "license_status", "dependency_status", "audio_vae_status", "tokenizer_status",
        "native_status", "no_upload", "scope_sha256",
    }
    if not isinstance(data, dict) or set(data) != keys:
        raise RuntimeError("approval schema is not exact")
    expected = {
        "schema": "vokra-voxcpm-0.5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY",
        "expected_head": expected_head, "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION,
        "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
        "public_repository": PUBLIC_REPOSITORY, "public_revision": PUBLIC_REVISION,
        "audio_vae_source": AUDIOVAE_SOURCE, "tokenizer_files": TOKENIZER_FILES,
        "license_status": "DOCS_SIGNED_APACHE_2_0", "dependency_status": "BLOCKED_UNREVIEWED",
        "audio_vae_status": "UNRESOLVED", "tokenizer_status": "UNRESOLVED",
        "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "no_upload": True,
    }
    if any(data[key] != value for key, value in expected.items()):
        raise RuntimeError("approval identity or unresolved gate mismatch")
    if data["no_upload"] is not True:
        raise RuntimeError("approval no_upload must be the JSON boolean true")
    if not isinstance(data["scope_sha256"], str) or data["scope_sha256"] != _expected_scope(data):
        raise RuntimeError("approval scope mismatch")
    return data


def require_blocked_gate(expected_head: str, approval: str, approval_sha256: str, root: Path) -> None:
    if not isinstance(expected_head, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise RuntimeError("--expected-head must be lowercase 40-hex")
    if not isinstance(approval_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", approval_sha256):
        raise RuntimeError("--approval-sha256 must be lowercase 64-hex")
    path = _safe_existing_file(approval)
    validate_approval(path.read_bytes(), expected_head, approval_sha256)
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    if status:
        raise RuntimeError("checkout must be clean")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_head:
        raise RuntimeError(f"checkout HEAD {actual} differs from --expected-head {expected_head}")
    raise RuntimeError("BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE: no model, source, checkpoint, packet, or output reads are permitted")


def self_test(source: Path, gate_marker: str, input_marker: str, input_args: list[str]) -> None:
    assert validate_approval
    try:
        _pairs([("x", 1), ("x", 2)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate approval keys must fail")
    for bad in ("", "relative.json", "/tmp/./approval.json", "/tmp/../approval.json", "//tmp/approval.json", "/tmp/approval.json/"):
        try:
            _safe_existing_file(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe approval path accepted")
    text = source.read_text(encoding="utf-8")
    gate = text.index(gate_marker)
    inputs = text.index(input_marker)
    if gate >= inputs:
        raise AssertionError("blocked gate occurs after input handling")
    result = subprocess.run(
        [sys.executable, str(source), "--expected-head", "not-a-head", *input_args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 2 or "--expected-head must be lowercase 40-hex" not in result.stderr:
        raise AssertionError("normal CLI did not block before reading missing inputs")
