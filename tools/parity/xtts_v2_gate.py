#!/usr/bin/env -S uv run --no-cache --no-project --offline --python 3.12 python
"""Stdlib-only terminal gate for the XTTS-v2 inspection/preparation tools.

The current XTTS-v2 route is inspection-only.  This module authenticates the
caller-bound blocked disposition and checkout before any model/source input,
work/output path, cache, network, or torch import is touched.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "coqui/XTTS-v2"
MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
SOURCE_REPOSITORY = "https://github.com/coqui-ai/TTS.git"
SOURCE_REVISION = "480a6cdf7dab508063c5d2e1b92fb7cd9f4f63c1"
APPROVAL_SCHEMA = "vokra-xtts-v2-blocked-approval-v1"
APPROVAL_SCOPE = "XTTS_V2_INSPECTION_ONLY"
BLOCKED_MARKER = "XTTS_V2_BLOCKED_APPROVAL: status=BLOCKED decision=INSPECTION_ONLY NO_UPLOAD"


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise RuntimeError(f"duplicate approval key: {key}")
        result[key] = value
    return result


def _exact(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _scope(data: dict[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "scope_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def approval_fixture(expected_head: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": APPROVAL_SCHEMA,
        "status": "BLOCKED",
        "decision": "INSPECTION_ONLY",
        "expected_head": expected_head,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "model_license_status": "CPML_RESEARCH_ONLY_T4_OWNER_SIGNED",
        "source_license_status": "PRIMARY_REVIEW_REQUIRED",
        "dependency_status": "BLOCKED_UNREVIEWED",
        "voice_cloning_consent_status": "BLOCKED_UNASSESSED",
        "voice_cloning_policy_status": "BLOCKED_UNASSESSED",
        "dataset_status": "BLOCKED_UNAUTHENTICATED",
        "gpt_status": "BLOCKED_UNIMPLEMENTED",
        "dvae_status": "BLOCKED_UNIMPLEMENTED",
        "hifigan_status": "BLOCKED_UNIMPLEMENTED",
        "parity_status": "NOT_RUN",
        "no_upload": True,
        "scope": APPROVAL_SCOPE,
    }
    data["scope_sha256"] = _scope(data)
    return data


def approval_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _external_file(raw: str, label: str, root: Path) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/") or "\\" in raw or "//" in raw:
        raise RuntimeError(f"{label} must be an absolute lexical-clean path")
    parts = raw.split("/")
    if any(not part or part in {".", ".."} for part in parts[1:]):
        raise RuntimeError(f"{label} must be an absolute lexical-clean path")
    path = Path(raw)
    current = Path("/")
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"{label} has symlink ancestry")
    try:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} must be a regular non-symlink file")
        resolved = path.resolve(strict=True)
        checkout = root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} canonicalization failed: {error}") from error
    if resolved == checkout or checkout in resolved.parents:
        raise RuntimeError(f"{label} must be outside the checkout")
    return path


def validate_approval(raw: bytes, expected_head: str, supplied_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head or ""):
        raise RuntimeError("--expected-head must be lowercase HEX40")
    if not re.fullmatch(r"[0-9a-f]{64}", supplied_sha256 or ""):
        raise RuntimeError("--approval-sha256 must be lowercase HEX64")
    if hashlib.sha256(raw).hexdigest() != supplied_sha256:
        raise RuntimeError("approval bytes changed or SHA-256 is wrong")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"approval is not strict UTF-8 JSON: {error}") from error
    expected = approval_fixture(expected_head)
    if not isinstance(data, dict) or set(data) != set(expected) or any(not _exact(data[key], value) for key, value in expected.items()):
        raise RuntimeError("approval schema/identity/disposition mismatch")
    if data["no_upload"] is not True or data["scope_sha256"] != _scope(data):
        raise RuntimeError("approval NO_UPLOAD or scope binding mismatch")
    return data


def require_blocked_gate(expected_head: str, approval_path: str, approval_sha256: str, root: Path) -> None:
    path = _external_file(approval_path, "approval evidence", root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"approval evidence read failed: {error}") from error
    validate_approval(raw, expected_head, approval_sha256)
    try:
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"checkout identity verification failed: {error}") from error
    if status:
        raise RuntimeError("checkout must be clean")
    if actual != expected_head:
        raise RuntimeError(f"checkout HEAD {actual} differs from --expected-head {expected_head}")
    raise RuntimeError(BLOCKED_MARKER)


def self_test() -> None:
    for value in ("", "relative.json", "/tmp/./approval.json", "/tmp/../approval.json", "//tmp/approval.json"):
        with tempfile.TemporaryDirectory(prefix="xtts-gate-path-") as temporary:
            try:
                _external_file(value, "approval", Path(temporary))
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe approval path accepted")
    with tempfile.TemporaryDirectory(prefix="xtts-gate-repo-") as repo_tmp, tempfile.TemporaryDirectory(prefix="xtts-gate-approval-") as approval_tmp:
        repo = Path(repo_tmp).resolve()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "XTTS gate self-test"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "xtts-gate@example.invalid"], check=True)
        (repo / "seed").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "seed"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        approval = Path(approval_tmp).resolve() / "approval.json"
        raw = approval_bytes(approval_fixture(head)); approval.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        try:
            require_blocked_gate(head, str(approval), digest, repo)
        except RuntimeError as error:
            if BLOCKED_MARKER not in str(error):
                raise AssertionError(f"valid approval did not reach terminal blocker: {error}")
        else:
            raise AssertionError("valid blocked approval passed")
        try:
            validate_approval(raw, head, "0" * 64)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong approval SHA accepted")
        for bad in ("0" * 40, "0" * 64):
            try:
                validate_approval(raw, bad, digest)
            except RuntimeError:
                pass
            else:
                raise AssertionError("wrong head accepted")
        for value in (1, 0, "true"):
            changed = approval_fixture(head); changed["no_upload"] = value; changed["scope_sha256"] = _scope(changed); changed_raw = approval_bytes(changed)
            try:
                validate_approval(changed_raw, head, hashlib.sha256(changed_raw).hexdigest())
            except RuntimeError:
                pass
            else:
                raise AssertionError("non-boolean no_upload accepted")
        for malformed in (b"\xff", b"{not-json", b'{"schema":"one","schema":"two"}'):
            try:
                validate_approval(malformed, head, hashlib.sha256(malformed).hexdigest())
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("malformed/duplicate approval accepted")
        ignored = repo / "ignored-approval.json"
        (repo / ".gitignore").write_text("ignored-approval.json\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "ignore"], check=True)
        ignored.write_bytes(raw)
        try:
            _external_file(str(ignored), "approval", repo)
        except RuntimeError as error:
            if "outside the checkout" not in str(error):
                raise AssertionError(f"checkout-local approval rejected for wrong reason: {error}")
        else:
            raise AssertionError("checkout-local approval accepted")
    print("xtts_v2_gate self-test: OK")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    if len(sys.argv) == 9 and sys.argv[1] == "--verify":
        values = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        expected = {"--expected-head", "--approval-evidence", "--approval-sha256", "--root"}
        if set(values) != expected:
            print("usage: xtts_v2_gate.py --verify --expected-head HEX40 --approval-evidence FILE --approval-sha256 HEX64 --root CHECKOUT", file=sys.stderr)
            return 2
        try:
            require_blocked_gate(values["--expected-head"], values["--approval-evidence"], values["--approval-sha256"], Path(values["--root"]))
        except RuntimeError as error:
            if BLOCKED_MARKER in str(error):
                print(str(error), file=sys.stderr)
                return 2
            print(f"XTTS_V2_BLOCKED_APPROVAL_INVALID: {error}", file=sys.stderr)
            return 1
    print("usage: xtts_v2_gate.py --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
