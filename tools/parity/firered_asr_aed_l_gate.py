"""Stdlib-only fail-closed gate shared by FireRed inspection helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

APPROVAL_SCHEMA = "vokra-firered-asr-aed-l-blocked-approval-v1"
SCOPE: dict[str, str] = {
    "cmvn_status": "BLOCKED_STRUCTURAL_REVIEW_REQUIRED",
    "config_status": "BLOCKED_EMPTY_CONFIG",
    "dependency_status": "BLOCKED_UNREVIEWED_TRANSITIVE",
    "kaldi_native_fbank_revision": "f68c6b43f739697d7ab02ff6debacee130e1d541",
    "kaldi_native_fbank_url": "https://github.com/csukuangfj/kaldi-native-fbank.git",
    "license_status": "BLOCKED_TRAINING_AND_DEPENDENCY_PROVENANCE",
    "model_repository": "FireRedTeam/FireRedASR-AED-L",
    "model_revision": "e57f5960d03cff1071ff7acbb409314d1e70ed3d",
    "native_status": "BLOCKED_NATIVE_BINDING",
    "source_revision": "834635e4cf277ed8ca92049fc375b17c3dc20748",
    "source_status": "AUTHENTICATED_SOURCE_CONTRACT",
    "source_url": "https://github.com/FireRedTeam/FireRedASR.git",
    "tokenizer_status": "BLOCKED_TOKENIZER_BINDING",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """Malformed, stale, or unauthorized gate input."""


class GateBlocked(RuntimeError):
    """Exact blocked approval was validated; execution remains disabled."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _safe_external_file(path: Path | str, root: Path) -> Path:
    raw = os.fspath(path)
    if not raw.startswith("/") or "//" in raw or "/./" in raw or "/../" in raw or raw.endswith("/") or raw.endswith("/.") or raw.endswith("/.."):
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


def validate_blocked_approval(path: Path | str, expected_head: str, root: Path) -> None:
    if not HEX40.fullmatch(expected_head):
        raise GateError("expected HEAD must be lowercase 40-hex")
    path = _safe_external_file(path, root)
    actual_head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout:
        raise GateError("checkout must be clean before FireRed helper execution")
    if actual_head != expected_head:
        raise GateError("expected HEAD does not match checkout")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise GateError("approval must be a JSON object")
    keys = {"schema", "decision", "status", "evidence_stage", "no_upload", "expected_head", *SCOPE, "scope_sha256"}
    if set(value) != keys:
        raise GateError("approval key set mismatch")
    expected = {
        "schema": APPROVAL_SCHEMA,
        "decision": "BLOCKED",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "no_upload": True,
        "expected_head": expected_head,
        **SCOPE,
        "scope_sha256": hashlib.sha256(_canonical(SCOPE).encode()).hexdigest(),
    }
    if value != expected:
        raise GateError("approval identity, scope, or blocked disposition mismatch")


def enforce_blocked_approval(path: Path | str, approval_sha256: str, expected_head: str, root: Path) -> None:
    if not HEX64.fullmatch(approval_sha256):
        raise GateError("approval SHA-256 must be lowercase 64-hex")
    path = _safe_external_file(path, root)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != approval_sha256:
        raise GateError("approval SHA-256 does not match caller binding")
    validate_blocked_approval(path, expected_head, root)
    raise GateBlocked("BLOCKED_APPROVAL/INSPECTION_ONLY/NO_UPLOAD: FireRed facts remain unresolved")


def self_test(root: Path) -> None:
    """Exercise the gate without touching model inputs or importing torch."""
    with tempfile.TemporaryDirectory(prefix="firered-gate-") as directory:
        path = Path(directory).resolve() / "approval.json"
        head = "0" * 40
        value = {
            "schema": APPROVAL_SCHEMA,
            "decision": "BLOCKED",
            "status": "BLOCKED",
            "evidence_stage": "INSPECTION_ONLY",
            "no_upload": True,
            "expected_head": head,
            **SCOPE,
            "scope_sha256": hashlib.sha256(_canonical(SCOPE).encode()).hexdigest(),
        }
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
        original = subprocess.run
        def fake_run(command: list[str], **kwargs: Any) -> Any:
            if command[-2:] == ["--verify", "HEAD"]:
                return type("Result", (), {"stdout": head})()
            if command[-1:] == ["--untracked-files=all"]:
                return type("Result", (), {"stdout": ""})()
            return original(command, **kwargs)
        globals()["subprocess"].run = fake_run  # type: ignore[method-assign]
        try:
            validate_blocked_approval(path, head, root)
            try:
                enforce_blocked_approval(path, hashlib.sha256(path.read_bytes()).hexdigest(), head, root)
            except GateBlocked:
                pass
            else:
                raise AssertionError("valid blocked approval did not stop execution")
            for bad_sha, bad_head in [("0" * 64, head), (hashlib.sha256(path.read_bytes()).hexdigest(), "1" * 40)]:
                try:
                    enforce_blocked_approval(path, bad_sha, bad_head, root)
                except GateError:
                    pass
                else:
                    raise AssertionError("wrong caller binding was accepted")
            malformed = dict(value); malformed["status"] = "OWNER_APPROVED"; path.write_text(json.dumps(malformed), encoding="utf-8")
            try: validate_blocked_approval(path, head, root)
            except GateError: pass
            else: raise AssertionError("non-blocked approval accepted")
            malformed = dict(value); malformed["scope_sha256"] = "0" * 64; path.write_text(json.dumps(malformed), encoding="utf-8")
            try: validate_blocked_approval(path, head, root)
            except GateError: pass
            else: raise AssertionError("wrong scope was accepted")
            path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
            try: validate_blocked_approval(path, head, root)
            except GateError: pass
            else: raise AssertionError("duplicate approval accepted")
            target = Path(directory) / "target"; target.mkdir(); target_file = target / "approval.json"; target_file.write_text("{}", encoding="utf-8")
            link = Path(directory) / "link"; link.symlink_to(target, target_is_directory=True)
            try: validate_blocked_approval(link / "approval.json", head, root)
            except GateError: pass
            else: raise AssertionError("symlinked approval ancestry accepted")
            try: validate_blocked_approval(str(path.parent) + "/./" + path.name, head, root)
            except GateError: pass
            else: raise AssertionError("literal dot approval path accepted")
            try: validate_blocked_approval(str(path.parent) + "/../" + path.name, head, root)
            except GateError: pass
            else: raise AssertionError("dot-component approval path accepted")
        finally:
            globals()["subprocess"].run = original  # type: ignore[method-assign]


if __name__ == "__main__":
    self_test(Path.cwd())
    print("firered_asr_aed_l_gate self-test PASS")
