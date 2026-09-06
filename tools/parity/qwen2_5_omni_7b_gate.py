"""Stdlib-only blocked gate for Qwen2.5-Omni-7B inspection and preparation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

APPROVAL_SCHEMA = "vokra-qwen2-5-omni-7b-blocked-approval-v1"
SCOPE: dict[str, str] = {
    "dataset_status": "BLOCKED_TRAINING_PROVENANCE_UNAUTHENTICATED",
    "dependency_status": "BLOCKED_UNREVIEWED_TRANSFORMERS_AND_AUDIO_DEPENDENCIES",
    "component_license_status": "BLOCKED_COMPONENT_LICENSE_REVIEW",
    "model_repository": "Qwen/Qwen2.5-Omni-7B",
    "model_revision": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
    "native_status": "BLOCKED_NATIVE_MULTIMODAL_STREAMING_NOT_IMPLEMENTED",
    "runtime_status": "BLOCKED_THINKER_TALKER_AUDIO_VAE_NOT_IMPLEMENTED",
    "model_license_status": "RECORDED_APACHE_2_0_MODEL_LICENSE",
    "source_repository": "https://github.com/QwenLM/Qwen2.5-Omni.git",
    "source_revision": "d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42",
    "source_status": "BLOCKED_SOURCE_ROLE_AND_LICENSE_REVIEW",
    "source_license_status": "BLOCKED_SOURCE_LICENSE_UNKNOWN",
    "transformers_repository": "https://github.com/huggingface/transformers.git",
    "transformers_revision": "f4fc42216cd56ab6b68270bf80d811614d8d59e4",
    "transformers_tag": "v4.52.3",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """Malformed, stale, or unauthorized gate input."""


class GateBlocked(RuntimeError):
    """The exact blocked approval was validated; execution remains disabled."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def scope_sha256() -> str:
    return hashlib.sha256(canonical_json(SCOPE)).hexdigest()


def _safe_external_file(path: Path | str, root: Path) -> Path:
    raw = os.fspath(path)
    if not raw.startswith("/") or "//" in raw or "/./" in raw or "/../" in raw or raw.endswith(("/", "/.", "/..")):
        raise GateError("approval path must be absolute and free of dot components")
    path = Path(raw)
    if not path.is_absolute() or any(part in ("", ".", "..") for part in path.parts[1:]):
        raise GateError("approval path must be absolute and free of dot components")
    root = root.resolve()
    if path == root or root in path.parents:
        raise GateError("approval path must be outside the checkout")
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        if current.is_symlink():
            raise GateError("approval path has a symlinked ancestor")
    if path.is_symlink() or not path.is_file():
        raise GateError("approval must be a regular non-symlink file")
    return path


def _git_identity(root: Path, expected_head: str) -> None:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise GateError(f"git checkout identity check failed: {error}") from error
    if status:
        raise GateError("checkout must be clean before Qwen2.5-Omni helper execution")
    if head != expected_head:
        raise GateError("expected HEAD does not match checkout")


def validate_blocked_approval(path: Path | str, expected_head: str, root: Path, approval_bytes: bytes | None = None) -> None:
    if not HEX40.fullmatch(expected_head):
        raise GateError("expected HEAD must be lowercase 40-hex")
    path = _safe_external_file(path, root)
    _git_identity(root, expected_head)
    if approval_bytes is None:
        try:
            approval_bytes = path.read_bytes()
        except OSError as error:
            raise GateError(f"approval read failed: {error}") from error
    try:
        value = json.loads(approval_bytes.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise GateError(f"approval JSON/UTF-8 is malformed: {error}") from error
    if not isinstance(value, dict):
        raise GateError("approval must be a JSON object")
    required = {"schema", "status", "evidence_stage", "publication", "no_upload", "expected_head", "scope_sha256", *SCOPE}
    if set(value) != required:
        raise GateError("approval key set mismatch")
    if value.get("no_upload") is not True:
        raise GateError("approval no_upload must be the JSON boolean true")
    expected = {"schema": APPROVAL_SCHEMA, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "publication": "NO_UPLOAD", "no_upload": True, "expected_head": expected_head, "scope_sha256": scope_sha256(), **SCOPE}
    if value != expected:
        raise GateError("approval identity, scope, or blocked disposition mismatch")


def validate_bound_blocked_approval(path: Path | str, approval_sha256: str, expected_head: str, root: Path) -> None:
    if not HEX64.fullmatch(approval_sha256):
        raise GateError("approval SHA-256 must be lowercase 64-hex")
    path = _safe_external_file(path, root)
    try:
        approval_bytes = path.read_bytes()
    except OSError as error:
        raise GateError(f"approval read failed: {error}") from error
    if hashlib.sha256(approval_bytes).hexdigest() != approval_sha256:
        raise GateError("approval SHA-256 does not match caller binding")
    validate_blocked_approval(path, expected_head, root, approval_bytes)


def enforce_blocked_approval(path: Path | str, approval_sha256: str, expected_head: str, root: Path) -> None:
    validate_bound_blocked_approval(path, approval_sha256, expected_head, root)
    raise GateBlocked("BLOCKED_APPROVAL/INSPECTION_ONLY/NO_UPLOAD: Qwen2.5-Omni facts remain unresolved")


def _fixture(head: str) -> dict[str, Any]:
    return {"schema": APPROVAL_SCHEMA, "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "publication": "NO_UPLOAD", "no_upload": True, "expected_head": head, "scope_sha256": scope_sha256(), **SCOPE}


def self_test(root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="qwen2-omni-gate-") as directory:
        directory_path = Path(directory).resolve()
        path = directory_path / "approval.json"
        head = "0" * 40
        path.write_bytes(canonical_json(_fixture(head)))
        original_run = subprocess.run

        def fake_run(command: list[str], **kwargs: Any) -> Any:
            if command[-2:] == ["--verify", "HEAD"]:
                return type("Result", (), {"stdout": head})()
            if command[-1:] == ["--untracked-files=all"]:
                return type("Result", (), {"stdout": ""})()
            return original_run(command, **kwargs)

        globals()["subprocess"].run = fake_run  # type: ignore[method-assign]
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_bound_blocked_approval(path, digest, head, root)
            try:
                enforce_blocked_approval(path, digest, head, root)
            except GateBlocked:
                pass
            else:
                raise AssertionError("valid blocked approval authorized execution")
            for bad_sha, bad_head in (("0" * 64, head), (digest, "1" * 40)):
                try:
                    validate_bound_blocked_approval(path, bad_sha, bad_head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("wrong caller binding accepted")
            for mutation in ({"status": "APPROVED"}, {"scope_sha256": "0" * 64}, {"publication": "UPLOAD"}, {"no_upload": 1}, {"no_upload": 0}, {"no_upload": "true"}):
                value = _fixture(head)
                value.update(mutation)
                path.write_bytes(canonical_json(value))
                try:
                    validate_bound_blocked_approval(path, hashlib.sha256(path.read_bytes()).hexdigest(), head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("invalid blocked approval accepted")
            path.write_bytes(b'{"schema":"x","schema":"y"}\n')
            try:
                validate_blocked_approval(path, head, root)
            except GateError:
                pass
            else:
                raise AssertionError("duplicate approval key accepted")
            path.write_bytes(b"\xff\xfe\n")
            try:
                validate_blocked_approval(path, head, root)
            except GateError:
                pass
            else:
                raise AssertionError("invalid UTF-8 approval accepted")
            target = directory_path / "target"
            target.mkdir()
            (target / "approval.json").write_bytes(canonical_json(_fixture(head)))
            link = directory_path / "link"
            link.symlink_to(target, target_is_directory=True)
            try:
                validate_blocked_approval(link / "approval.json", head, root)
            except GateError:
                pass
            else:
                raise AssertionError("symlinked approval ancestry accepted")
            for raw in (str(path.parent) + "/./" + path.name, str(path.parent) + "/../" + path.name):
                try:
                    validate_blocked_approval(raw, head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("dot-component approval path accepted")
            try:
                _safe_external_file(root / ".gitignore-approval.json", root)
            except GateError:
                pass
            else:
                raise AssertionError("checkout-internal ignored approval path accepted")
        finally:
            globals()["subprocess"].run = original_run  # type: ignore[method-assign]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--approval-evidence")
    parser.add_argument("--approval-sha256")
    parser.add_argument("--expected-head")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        if any(value is not None for value in (args.approval_evidence, args.approval_sha256, args.expected_head, args.root)) or args.validate or args.enforce:
            parser.error("--self-test accepts no validation arguments")
        self_test(Path.cwd())
        print("qwen2_5_omni_7b_gate self-test: OK")
        return 0
    if args.validate == args.enforce or not all((args.approval_evidence, args.approval_sha256, args.expected_head, args.root)):
        parser.error("exactly one of --validate/--enforce plus all validation arguments is required")
    try:
        if args.enforce:
            enforce_blocked_approval(args.approval_evidence, args.approval_sha256, args.expected_head, args.root)
        else:
            validate_bound_blocked_approval(args.approval_evidence, args.approval_sha256, args.expected_head, args.root)
    except GateBlocked as error:
        print(str(error), file=sys.stderr)
        return 2
    except GateError as error:
        print(f"Qwen2.5-Omni gate rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
