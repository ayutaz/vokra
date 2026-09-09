"""Audited source-only compatibility adapter for the Qwen3-TTS API probes.

The pinned Qwen source uses ``@check_model_inputs()`` while the reviewed
Transformers release exposes ``check_model_inputs(func)``.  This module owns
the one bounded source transformation used by both API workers.  It never
downloads source or models and only mutates a disposable staged checkout.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

PATCH_TARGET = "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
PATCH_ORIGINAL_BYTES = 40519
PATCH_ORIGINAL_SHA256 = "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628"
PATCHED_BYTES = 40517
PATCHED_SHA256 = "a9da44f2f6b7ff0beb4dd43e8c4c48138e51423e9bcc515a253ea088381d3b9c"
PATCH_FROM = b"@check_model_inputs()"
PATCH_TO = b"@check_model_inputs"
PATCH_STATUS = "COMPATIBILITY_PATCH_APPLIED"
PATCH_OPERATION = "replace_exactly_one_decorator"
TRANSFORMERS_API = "check_model_inputs(func)"


class CompatibilityPatchError(RuntimeError):
    """The staged source did not satisfy the fixed patch contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def patch_source_bytes(original: bytes) -> bytes:
    """Apply the exact, hash-bound decorator adaptation in memory."""
    if len(original) != PATCH_ORIGINAL_BYTES or sha256_bytes(original) != PATCH_ORIGINAL_SHA256:
        raise CompatibilityPatchError("compatibility patch original source identity drifted")
    count = original.count(PATCH_FROM)
    if count != 1:
        raise CompatibilityPatchError(f"compatibility patch expected exactly one decorator, found {count}")
    patched = original.replace(PATCH_FROM, PATCH_TO)
    if len(patched) != PATCHED_BYTES or sha256_bytes(patched) != PATCHED_SHA256:
        raise CompatibilityPatchError("compatibility patch output identity drifted")
    if patched.count(PATCH_FROM) != 0 or patched.count(PATCH_TO) != 1:
        raise CompatibilityPatchError("compatibility patch result is not exact")
    return patched


def patch_source_checkout(source: Path) -> dict[str, Any]:
    """Patch exactly one file in a clean disposable source checkout."""
    source = source.resolve(strict=False)
    target = source / PATCH_TARGET
    try:
        target.resolve().relative_to(source)
    except ValueError as error:
        raise CompatibilityPatchError("compatibility patch target escapes source checkout") from error
    if target.is_symlink() or not target.is_file():
        raise CompatibilityPatchError("compatibility patch target is missing or symlinked")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise CompatibilityPatchError("official source must be clean before compatibility patch")
    original = target.read_bytes()
    patched = patch_source_bytes(original)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.compat.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise CompatibilityPatchError("compatibility patch temporary path already exists")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(patched)
        os.chmod(temporary, target.stat().st_mode & 0o777)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    expected_status = f" M {PATCH_TARGET}"
    after_status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout.rstrip("\r\n")
    if after_status != expected_status:
        raise CompatibilityPatchError(f"compatibility patch changed unexpected source paths: {after_status!r}")
    return {
        "status": PATCH_STATUS,
        "target": PATCH_TARGET,
        "operation": PATCH_OPERATION,
        "original_bytes": PATCH_ORIGINAL_BYTES,
        "original_sha256": PATCH_ORIGINAL_SHA256,
        "patched_bytes": PATCHED_BYTES,
        "patched_sha256": PATCHED_SHA256,
        "replacement_count": 1,
        "transformers_api": TRANSFORMERS_API,
    }


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clean_repo(root: Path, content: bytes) -> Path:
    target = root / PATCH_TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    _git("init", "--quiet", cwd=root)
    _git("config", "user.email", "self-test@example.invalid", cwd=root)
    _git("config", "user.name", "Qwen self-test", cwd=root)
    _git("add", PATCH_TARGET, cwd=root)
    _git("commit", "--quiet", "-m", "fixture", cwd=root)
    return target


def self_test_filesystem() -> None:
    """Exercise the real atomic checkout seam without official source data."""
    global PATCH_TARGET, PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256
    original_contract = (PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256)
    sample = b"prefix\n@check_model_inputs()\nsuffix\n"
    patched = b"prefix\n@check_model_inputs\nsuffix\n"
    PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256 = len(sample), sha256_bytes(sample)
    PATCHED_BYTES, PATCHED_SHA256 = len(patched), sha256_bytes(patched)
    try:
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-git-") as directory:
            root = Path(directory)
            target = _clean_repo(root, sample)
            record = patch_source_checkout(root)
            assert record["patched_bytes"] == len(patched) and record["patched_sha256"] == sha256_bytes(patched)
            assert target.read_bytes() == patched
            _git("add", PATCH_TARGET, cwd=root)
            _git("commit", "--quiet", "-m", "patched", cwd=root)
            try:
                patch_source_checkout(root)
            except CompatibilityPatchError:
                pass
            else:
                raise AssertionError("already patched source was accepted")

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-dirty-") as directory:
            root = Path(directory)
            target = _clean_repo(root, sample)
            (root / "unrelated.txt").write_text("dirty", encoding="utf-8")
            before = target.read_bytes()
            try:
                patch_source_checkout(root)
            except CompatibilityPatchError:
                pass
            else:
                raise AssertionError("pre-dirty checkout was accepted")
            assert target.read_bytes() == before

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-temp-") as directory:
            root = Path(directory)
            target = _clean_repo(root, sample)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.compat.tmp")
            temporary.write_bytes(b"must-not-clobber")
            _git("add", "-f", str(temporary.relative_to(root)), cwd=root)
            _git("commit", "--quiet", "-m", "temporary", cwd=root)
            try:
                patch_source_checkout(root)
            except CompatibilityPatchError:
                pass
            else:
                raise AssertionError("existing temporary path was clobbered")
            assert target.read_bytes() == sample and temporary.read_bytes() == b"must-not-clobber"

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-symlink-") as directory:
            root = Path(directory)
            target = _clean_repo(root, sample)
            outside = root / "outside.py"
            outside.write_bytes(sample)
            target.unlink()
            target.symlink_to(outside)
            try:
                patch_source_checkout(root)
            except CompatibilityPatchError:
                pass
            else:
                raise AssertionError("symlink target was accepted")

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-escape-") as directory:
            root = Path(directory)
            _clean_repo(root, sample)
            original_target = PATCH_TARGET
            PATCH_TARGET = "../outside.py"
            try:
                try:
                    patch_source_checkout(root)
                except CompatibilityPatchError:
                    pass
                else:
                    raise AssertionError("path escape was accepted")
            finally:
                PATCH_TARGET = original_target
    finally:
        PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256 = original_contract
