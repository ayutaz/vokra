"""Stdlib-only terminal gate for the unresolved Baichuan-Audio composite."""
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

MODEL_REPOSITORY = "baichuan-inc/Baichuan-Audio-Instruct"
MODEL_REVISION = "1c86512d863376f9ea0c32bb77451b9f428283c8"
SOURCE_REPOSITORY = "https://github.com/baichuan-inc/Baichuan-Audio.git"
SOURCE_REVISION = "805d456433dbf3e0edb2bdd302f733a4bd38ea84"
SOURCE_ROLE_STATUS = "AUTHENTICATED_PUBLIC_GITHUB_SOURCE_ROLES_INCOMPLETE_HF_CUSTOM_CODE"
# These are scope facts, not approvals.  They bind the unresolved component
# identities before a worker may acquire any HF/source payload.
MATCHA_STATUS = "ABSENT_NOT_A_SUBMODULE"
MATCHA_REVISION = None
HIFT_STATUS = "AUTHENTICATED_GIT_BLOB_ONLY_RAW_SHA256_PENDING"
HF_CUSTOM_CODE_STATUS = "UNRESOLVED_SEPARATE_HF_REPOSITORY"
HF_CUSTOM_CODE_ROLE_NAMES = [
    "audio_modeling_omni.py",
    "configuration_omni.py",
    "flow_matching.py",
    "generation_utils.py",
    "matcha_components.py",
    "matcha_feat.py",
    "matcha_transformer.py",
    "modeling_omni.py",
    "processor_omni.py",
    "vector_quantize.py",
]
BLOCKED_MARKER = "BLOCKED_UNRESOLVED_BAICHUAN_AUDIO_COMPOSITE"


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise RuntimeError(f"duplicate approval key: {key}")
        result[key] = value
    return result


def _scope(data: dict[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "scope_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _approval_fixture(expected_head: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema": "vokra-baichuan-audio-instruct-approval-v1",
        "status": "BLOCKED",
        "disposition": "INSPECTION_ONLY",
        "expected_head": expected_head,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_role_status": SOURCE_ROLE_STATUS,
        "matcha_status": MATCHA_STATUS,
        "matcha_revision": MATCHA_REVISION,
        "hift_status": HIFT_STATUS,
        "hf_custom_code_status": HF_CUSTOM_CODE_STATUS,
        "hf_custom_code_role_names": HF_CUSTOM_CODE_ROLE_NAMES,
        "dependency_status": "BLOCKED_UNREVIEWED_TRANSITIVE",
        "component_status": "UNRESOLVED",
        "dataset_status": "UNAUTHENTICATED",
        "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "no_upload": True,
    }
    data["scope_sha256"] = _scope(data)
    return data


def _approval_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_file(value: str) -> Path:
    parts = value.split("/")
    if not value or not value.startswith("/") or "\\" in value or any(not part or part in {".", ".."} for part in parts[1:]):
        raise RuntimeError("approval path must be absolute, lexical-clean, and non-symlinked")
    current = Path("/")
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            raise RuntimeError("approval path has symlink ancestry")
    path = Path(value)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("approval evidence must be a regular non-symlink file")
    return path


def _require_external(path: Path, root: Path) -> None:
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"approval/checkout canonicalization failed: {error}") from error
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        raise RuntimeError("approval evidence must be outside the checkout")


def validate_approval(raw: bytes, expected_head: str, supplied_sha: str) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != supplied_sha:
        raise RuntimeError("approval bytes changed or SHA-256 is wrong")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError(f"approval JSON is malformed UTF-8/JSON: {error}") from error
    keys = {
        "schema", "status", "disposition", "expected_head", "model_repository", "model_revision",
        "source_repository", "source_revision", "source_role_status", "matcha_status", "matcha_revision",
        "hift_status", "hf_custom_code_status", "hf_custom_code_role_names", "dependency_status", "component_status",
        "dataset_status", "native_status", "no_upload", "scope_sha256",
    }
    if not isinstance(data, dict) or set(data) != keys:
        raise RuntimeError("approval schema is not exact")
    expected = _approval_fixture(expected_head)
    if any(data[key] != value for key, value in expected.items()):
        raise RuntimeError("approval identity or unresolved gate mismatch")
    if data["no_upload"] is not True or data["scope_sha256"] != _scope(data):
        raise RuntimeError("approval scope or NO_UPLOAD binding mismatch")
    return data


def require_blocked_gate(expected_head: str, approval: str, approval_sha256: str, root: Path) -> None:
    if not isinstance(expected_head, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise RuntimeError("--expected-head must be lowercase 40-hex")
    if not isinstance(approval_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", approval_sha256):
        raise RuntimeError("--approval-sha256 must be lowercase 64-hex")
    path = _safe_file(approval)
    _require_external(path, root)
    validate_approval(path.read_bytes(), expected_head, approval_sha256)
    try:
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"checkout status verification failed: {error}") from error
    if status:
        raise RuntimeError("checkout must be clean")
    try:
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"checkout HEAD verification failed: {error}") from error
    if actual != expected_head:
        raise RuntimeError(f"checkout HEAD {actual} differs from --expected-head {expected_head}")
    raise RuntimeError(f"{BLOCKED_MARKER}: dependency/HF custom-code roles/dataset/native composite remain unresolved; public GitHub source roles are authenticated")


def self_test(source: Path, input_args: list[str], input_marker: str) -> None:
    text = source.read_text(encoding="utf-8")
    if text.index("require_blocked_gate(args.expected_head") >= text.index(input_marker):
        raise AssertionError("blocked gate occurs after input handling")
    blocked = subprocess.run([sys.executable, str(source), "--expected-head", "bad", *input_args], capture_output=True, text=True, check=False)
    if blocked.returncode != 2 or "--expected-head must be lowercase 40-hex" not in blocked.stderr:
        raise AssertionError("CLI did not block before input reads")
    mixed = subprocess.run([sys.executable, str(source), "--self-test", "--expected-head", "bad"], capture_output=True, text=True, check=False)
    if mixed.returncode != 2 or "cannot be combined" not in mixed.stderr:
        raise AssertionError("self-test and normal arguments were accepted together")
    for value in ("", "relative.json", "/tmp/./approval.json", "/tmp/../approval.json", "//tmp/approval.json"):
        try: _safe_file(value)
        except RuntimeError: pass
        else: raise AssertionError("unsafe approval path accepted")
    with tempfile.TemporaryDirectory(prefix="baichuan-gate-") as repo_tmp, tempfile.TemporaryDirectory(prefix="baichuan-approval-") as approval_tmp:
        repo = Path(repo_tmp).resolve()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Baichuan self-test"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "baichuan-self-test@example.invalid"], check=True)
        (repo / "seed").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "seed"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True, capture_output=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        approval = Path(approval_tmp).resolve() / "approval.json"; raw = _approval_bytes(_approval_fixture(head)); approval.write_bytes(raw); sha = hashlib.sha256(raw).hexdigest()
        if validate_approval(raw, head, sha)["expected_head"] != head: raise AssertionError("valid approval rejected")
        try: require_blocked_gate(head, str(approval), sha, repo)
        except RuntimeError as error:
            if BLOCKED_MARKER not in str(error): raise AssertionError(f"valid approval did not reach blocker: {error}")
        else: raise AssertionError("valid blocked approval passed")
        (repo / ".gitignore").write_text("ignored-approval.json\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "ignore"], check=True, capture_output=True)
        ignored = repo / "ignored-approval.json"; ignored.write_bytes(raw)
        try: require_blocked_gate(head, str(ignored), sha, repo)
        except RuntimeError as error:
            if "outside the checkout" not in str(error): raise AssertionError(f"checkout-local approval rejected for wrong reason: {error}")
        else: raise AssertionError("ignored checkout-local approval accepted")
        try: validate_approval(raw, head, "0" * 64)
        except RuntimeError: pass
        else: raise AssertionError("wrong SHA accepted")
        try: require_blocked_gate("0" * 40, str(approval), sha, repo)
        except RuntimeError: pass
        else: raise AssertionError("wrong HEAD accepted")
        wrong = _approval_fixture(head); wrong["model_revision"] = "0" * 40; wrong["scope_sha256"] = _scope(wrong); wrong_raw = _approval_bytes(wrong)
        try: validate_approval(wrong_raw, head, hashlib.sha256(wrong_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("wrong identity accepted")
        missing_scope = _approval_fixture(head); missing_scope.pop("hift_status"); missing_scope["scope_sha256"] = _scope(missing_scope); missing_raw = _approval_bytes(missing_scope)
        try: validate_approval(missing_raw, head, hashlib.sha256(missing_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("approval without HiFT scope was accepted")
        wrong_roles = _approval_fixture(head); wrong_roles["hf_custom_code_role_names"] = [*HF_CUSTOM_CODE_ROLE_NAMES, "spoof.py"]; wrong_roles["scope_sha256"] = _scope(wrong_roles); wrong_roles_raw = _approval_bytes(wrong_roles)
        try: validate_approval(wrong_roles_raw, head, hashlib.sha256(wrong_roles_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("approval with altered custom-code role scope was accepted")
        wrong_scope = _approval_fixture(head); wrong_scope["scope_sha256"] = "0" * 64; wrong_raw = _approval_bytes(wrong_scope)
        try: validate_approval(wrong_raw, head, hashlib.sha256(wrong_raw).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("wrong scope accepted")
        malformed = b"{not-json"
        try: validate_approval(malformed, head, hashlib.sha256(malformed).hexdigest())
        except (RuntimeError, ValueError): pass
        else: raise AssertionError("malformed approval accepted")
        duplicate = b'{"schema":"one","schema":"two"}'
        try: validate_approval(duplicate, head, hashlib.sha256(duplicate).hexdigest())
        except RuntimeError: pass
        else: raise AssertionError("duplicate key accepted")
        link = Path(approval_tmp).resolve() / "link.json"; link.symlink_to(approval)
        try: _safe_file(str(link))
        except RuntimeError: pass
        else: raise AssertionError("symlink approval path accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true"); parser.add_argument("--expected-head"); parser.add_argument("--approval-evidence"); parser.add_argument("--approval-sha256"); parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if not args.verify or any(value is None for value in (args.expected_head, args.approval_evidence, args.approval_sha256, args.root)):
        parser.error("--verify requires --expected-head, --approval-evidence, --approval-sha256 and --root")
    try: require_blocked_gate(args.expected_head, args.approval_evidence, args.approval_sha256, args.root)
    except RuntimeError as error:
        if BLOCKED_MARKER in str(error): print(str(error), file=sys.stderr); return 2
        print(f"gate rejected: {error}", file=sys.stderr); return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
