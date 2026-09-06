"""Stdlib-only gate for the unresolved ACE-Step 1.5 inspection path.

The only accepted record is an authenticated *blocked* inspection approval.
Validating that record is deliberately followed by :class:`GateBlocked`; it
can never authorize downloading or importing the model.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from typing import Any

APPROVAL_SCHEMA = "vokra-ace-step-v15-blocked-approval-v1"
SCOPE: dict[str, str] = {
    "dataset_status": "BLOCKED_UNAUTHENTICATED_TRAINING_PROVENANCE",
    "dependency_status": "BLOCKED_UNREVIEWED_DEPENDENCY_LICENSES",
    "model_repository": "ACE-Step/Ace-Step1.5",
    "model_revision": "19671f406d603126926c1b7e2adc169acbcade22",
    "native_status": "BLOCKED_NATIVE_COMPOSITE_NOT_IMPLEMENTED",
    "source_repository": "https://github.com/ace-step/ACE-Step-1.5",
    "source_revision": "7202bc354d7fc31d1c0e5a90b0b49fb610e52362",
    "runtime_status": "BLOCKED_RUNTIME_NOT_IMPLEMENTED",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """Malformed, stale, or unauthorized gate input."""


class GateBlocked(RuntimeError):
    """The exact blocked record was validated; execution remains disabled."""


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
    if (
        not raw.startswith("/")
        or "//" in raw
        or "/./" in raw
        or "/../" in raw
        or raw.endswith(("/", "/.", "/.."))
    ):
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
        actual_head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise GateError(f"git checkout identity check failed: {error}") from error
    if status:
        raise GateError("checkout must be clean before ACE-Step helper execution")
    if actual_head != expected_head:
        raise GateError("expected HEAD does not match checkout")


def validate_blocked_approval(
    path: Path | str,
    expected_head: str,
    root: Path,
    approval_bytes: bytes | None = None,
) -> None:
    """Validate one exact blocked record, using one byte snapshot when given."""
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
    required = {
        "schema", "status", "disposition", "expected_head", "no_upload", "scope_sha256", *SCOPE
    }
    if set(value) != required:
        raise GateError("approval key set mismatch")
    expected = {
        "schema": APPROVAL_SCHEMA,
        "status": "BLOCKED",
        "disposition": "INSPECTION_ONLY",
        "expected_head": expected_head,
        "no_upload": True,
        "scope_sha256": scope_sha256(),
        **SCOPE,
    }
    if value != expected:
        raise GateError("approval identity, scope, or blocked disposition mismatch")


def validate_bound_blocked_approval(
    path: Path | str, approval_sha256: str, expected_head: str, root: Path
) -> None:
    """Validate caller SHA and the exact blocked record without authorizing work."""
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
    raise GateBlocked("BLOCKED_APPROVAL/INSPECTION_ONLY/NO_UPLOAD: ACE-Step facts remain unresolved")


def _fixture(head: str) -> dict[str, Any]:
    return {
        "schema": APPROVAL_SCHEMA,
        "status": "BLOCKED",
        "disposition": "INSPECTION_ONLY",
        "expected_head": head,
        "no_upload": True,
        "scope_sha256": scope_sha256(),
        **SCOPE,
    }


def self_test(root: Path) -> None:
    """Exercise only stdlib validation; no model, cache, or network is touched."""
    with tempfile.TemporaryDirectory(prefix="ace-step-v15-gate-") as directory:
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
            for mutation in (
                {"status": "APPROVED"},
                {"scope_sha256": "0" * 64},
                {"disposition": "EXECUTE"},
            ):
                value = _fixture(head)
                value.update(mutation)
                path.write_bytes(canonical_json(value))
                try:
                    validate_bound_blocked_approval(path, hashlib.sha256(path.read_bytes()).hexdigest(), head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("non-blocked approval accepted")
            path.write_bytes(b'{"schema":"x","schema":"y"}\n')
            try:
                validate_blocked_approval(path, head, root)
            except GateError:
                pass
            else:
                raise AssertionError("duplicate key accepted")
            path.write_bytes(b"\xff\xfe\n")
            try:
                validate_blocked_approval(path, head, root)
            except GateError:
                pass
            else:
                raise AssertionError("invalid UTF-8 accepted")
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
                raise AssertionError("symlinked ancestor accepted")
            for raw in (str(path.parent) + "/./" + path.name, str(path.parent) + "/../" + path.name):
                try:
                    validate_blocked_approval(raw, head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("dot-component path accepted")
            path.write_bytes(canonical_json(_fixture(head)))
            output = io.StringIO()
            with redirect_stderr(output):
                cli_status = main(
                    [
                        "--enforce",
                        "--approval-evidence", str(path),
                        "--approval-sha256", hashlib.sha256(path.read_bytes()).hexdigest(),
                        "--expected-head", head,
                        "--root", str(root),
                    ]
                )
            if cli_status != 2 or "BLOCKED_APPROVAL/INSPECTION_ONLY/NO_UPLOAD" not in output.getvalue():
                raise AssertionError("enforce CLI did not emit the exact blocked marker")
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
        print("ace_step_v15_gate self-test: OK")
        return 0
    if args.validate == args.enforce or not all((args.approval_evidence, args.approval_sha256, args.expected_head, args.root)):
        parser.error("exactly one of --validate/--enforce plus approval evidence, SHA, expected HEAD, and root is required")
    try:
        if args.enforce:
            enforce_blocked_approval(args.approval_evidence, args.approval_sha256, args.expected_head, args.root)
        else:
            validate_bound_blocked_approval(args.approval_evidence, args.approval_sha256, args.expected_head, args.root)
    except GateBlocked as error:
        print(str(error), file=sys.stderr)
        return 2
    except GateError as error:
        print(f"ACE-Step gate rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
