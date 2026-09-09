#!/usr/bin/env -S uv run --no-cache --no-project --offline --python 3.12 python
"""Stdlib-only terminal gate for the unresolved Hibiki 2B composite."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "kyutai/hibiki-2b-pytorch-bf16"
MODEL_REVISION = "bd71144c96f26040612f6414716f5f48ee4fce69"
HIBIKI_REPOSITORY = "https://github.com/kyutai-labs/hibiki.git"
HIBIKI_REVISION = "f1cf9293e35c1dceffbe60dd325bdd702bc8305e"
MOSHI_REPOSITORY = "https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
BLOCKED_MARKER = "BLOCKED_UNRESOLVED_HIBIKI_2B_COMPOSITE"


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result: raise RuntimeError(f"duplicate approval key: {key}")
        result[key] = value
    return result


def _scope(data: dict[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "scope_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fixture(head: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": "vokra-hibiki-2b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY", "expected_head": head,
        "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION, "hibiki_repository": HIBIKI_REPOSITORY, "hibiki_revision": HIBIKI_REVISION,
        "moshi_repository": MOSHI_REPOSITORY, "moshi_revision": MOSHI_REVISION,
        "source_role_status": "BLOCKED_UNRESOLVED", "dependency_status": "BLOCKED_UNREVIEWED", "license_status": "BLOCKED_PRIMARY_REVIEW",
        "dataset_status": "UNAUTHENTICATED", "tokenizer_codec_status": "UNRESOLVED", "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "no_upload": True,
    }
    data["scope_sha256"] = _scope(data)
    return data


def _raw(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _exact_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _safe_file(value: str) -> Path:
    parts = value.split("/")
    if not value or not value.startswith("/") or "\\" in value or any(not part or part in {".", ".."} for part in parts[1:]): raise RuntimeError("approval path must be absolute, lexical-clean, and non-symlinked")
    current = Path("/")
    for part in parts[1:]:
        current /= part
        if current.is_symlink(): raise RuntimeError("approval path has symlink ancestry")
    path = Path(value)
    if not path.is_file() or path.is_symlink(): raise RuntimeError("approval evidence must be a regular non-symlink file")
    return path


def _require_external(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"approval/checkout canonicalization failed: {error}") from error
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        raise RuntimeError("approval evidence must be outside the checkout")


def validate_approval(raw: bytes, head: str, supplied_sha: str) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != supplied_sha: raise RuntimeError("approval bytes changed or SHA-256 is wrong")
    try: data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error: raise RuntimeError(f"approval JSON is malformed UTF-8/JSON: {error}") from error
    expected = _fixture(head)
    if not isinstance(data, dict) or set(data) != set(expected) or any(not _exact_equal(data[key], value) for key, value in expected.items()): raise RuntimeError("approval identity/schema mismatch")
    if data["no_upload"] is not True or data["scope_sha256"] != _scope(data): raise RuntimeError("approval scope or NO_UPLOAD binding mismatch")
    return data


def require_blocked_gate(head: str, approval: str, supplied_sha: str, root: Path) -> None:
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head): raise RuntimeError("--expected-head must be lowercase 40-hex")
    if not isinstance(supplied_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied_sha): raise RuntimeError("--approval-sha256 must be lowercase 64-hex")
    path = _safe_file(approval); _require_external(path, root); validate_approval(path.read_bytes(), head, supplied_sha)
    try:
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError) as error: raise RuntimeError(f"checkout identity verification failed: {error}") from error
    if status: raise RuntimeError("checkout must be clean")
    if actual != head: raise RuntimeError(f"checkout HEAD {actual} differs from --expected-head {head}")
    raise RuntimeError(f"{BLOCKED_MARKER}: native streaming translation, dependencies, dataset, and composite runtime remain unresolved")


def self_test(source: Path, input_args: list[str], marker: str) -> None:
    result = subprocess.run([sys.executable, str(source), "--verify", "--expected-head", "bad", "--approval-evidence", "/missing/approval", "--approval-sha256", "0" * 64, "--root", "/missing/root"], capture_output=True, text=True, check=False)
    if result.returncode != 1 or "lowercase 40-hex" not in result.stderr: raise AssertionError("input path was reached before gate")
    mixed = subprocess.run([sys.executable, str(source), "--self-test", "--expected-head", "bad"], capture_output=True, text=True, check=False)
    if mixed.returncode != 2 or "cannot be combined" not in mixed.stderr: raise AssertionError("mixed self-test arguments accepted")
    for value in ("", "relative.json", "/tmp/./approval.json", "/tmp/../approval.json", "//tmp/approval.json"):
        try: _safe_file(value)
        except RuntimeError: pass
        else: raise AssertionError("unsafe approval path accepted")
    with tempfile.TemporaryDirectory(prefix="hibiki-gate-") as repo_tmp, tempfile.TemporaryDirectory(prefix="hibiki-approval-") as approval_tmp:
        repo = Path(repo_tmp).resolve(); subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Hibiki self-test"], check=True); subprocess.run(["git", "-C", str(repo), "config", "user.email", "hibiki-self-test@example.invalid"], check=True)
        (repo / "seed").write_text("seed\n", encoding="utf-8"); subprocess.run(["git", "-C", str(repo), "add", "seed"], check=True, capture_output=True); subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True, capture_output=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(); approval = Path(approval_tmp).resolve() / "approval.json"; raw = _raw(_fixture(head)); approval.write_bytes(raw); sha = hashlib.sha256(raw).hexdigest()
        if validate_approval(raw, head, sha)["expected_head"] != head: raise AssertionError("valid approval rejected")
        try: require_blocked_gate(head, str(approval), sha, repo)
        except RuntimeError as error:
            if BLOCKED_MARKER not in str(error): raise AssertionError(f"valid approval did not reach blocker: {error}")
        else: raise AssertionError("valid blocked approval passed")
        (repo / ".gitignore").write_text("ignored-approval.json\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True, capture_output=True); subprocess.run(["git", "-C", str(repo), "commit", "-qm", "ignore"], check=True, capture_output=True)
        ignored = repo / "ignored-approval.json"; ignored.write_bytes(raw)
        try: require_blocked_gate(head, str(ignored), sha, repo)
        except RuntimeError as error:
            if "outside the checkout" not in str(error): raise AssertionError(f"checkout-local approval rejected for wrong reason: {error}")
        else: raise AssertionError("ignored checkout-local approval accepted")
        for bad_sha in ("0" * 64,):
            try: validate_approval(raw, head, bad_sha)
            except RuntimeError: pass
            else: raise AssertionError("wrong SHA accepted")
        try: require_blocked_gate("0" * 40, str(approval), sha, repo)
        except RuntimeError: pass
        else: raise AssertionError("wrong HEAD accepted")
        wrong = _fixture(head); wrong["model_revision"] = "0" * 40; wrong["scope_sha256"] = _scope(wrong); wrong_raw = _raw(wrong)
        try: validate_approval(wrong_raw, head, hashlib.sha256(wrong_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("wrong identity accepted")
        wrong_scope = _fixture(head); wrong_scope["scope_sha256"] = "0" * 64; wrong_raw = _raw(wrong_scope)
        try: validate_approval(wrong_raw, head, hashlib.sha256(wrong_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("wrong scope accepted")
        boolean = _fixture(head); boolean["no_upload"] = 1; boolean["scope_sha256"] = _scope(boolean); boolean_raw = _raw(boolean)
        try: validate_approval(boolean_raw, head, hashlib.sha256(boolean_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("non-boolean NO_UPLOAD accepted")
        for malformed in (b"{not-json", b"\xff", b'{"schema":"one","schema":"two"}'):
            try: validate_approval(malformed, head, hashlib.sha256(malformed).hexdigest())
            except (RuntimeError, ValueError): pass
            else: raise AssertionError("malformed/duplicate approval accepted")
        link = Path(approval_tmp).resolve() / "link.json"; link.symlink_to(approval)
        try: _safe_file(str(link))
        except RuntimeError: pass
        else: raise AssertionError("symlink approval path accepted")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--verify", action="store_true"); parser.add_argument("--expected-head"); parser.add_argument("--approval-evidence"); parser.add_argument("--approval-sha256"); parser.add_argument("--root", type=Path); args = parser.parse_args()
    if args.self_test:
        if args.verify or any(value is not None for value in (args.expected_head, args.approval_evidence, args.approval_sha256, args.root)):
            parser.error("--self-test cannot be combined with verification arguments")
        self_test(Path(__file__), ["--snapshot", "/missing/snapshot", "--hibiki-source", "/missing/hibiki", "--moshi-source", "/missing/moshi", "--server-tree", "/missing/tree", "--output", "/missing/output"], BLOCKED_MARKER)
        print("hibiki_2b_gate self-test: OK")
        return 0
    if not args.verify or any(value is None for value in (args.expected_head, args.approval_evidence, args.approval_sha256, args.root)): parser.error("--verify requires --expected-head, --approval-evidence, --approval-sha256 and --root")
    try: require_blocked_gate(args.expected_head, args.approval_evidence, args.approval_sha256, args.root)
    except RuntimeError as error:
        if BLOCKED_MARKER in str(error): print(str(error), file=sys.stderr); return 2
        print(f"gate rejected: {error}", file=sys.stderr); return 1
    return 1


if __name__ == "__main__": raise SystemExit(main())
