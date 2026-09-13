"""Audited source-only compatibility patches for the Qwen3-TTS API probes."""

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

PATCH_25HZ_TARGET = "qwen_tts/__init__.py"
PATCH_25HZ_ORIGINAL_BYTES = 839
PATCH_25HZ_ORIGINAL_SHA256 = "ea52de59d070fde366467a6902d0edcfc1b0575b8c570a0c71020c41d6a593ed"
PATCH_25HZ_PATCHED_BYTES = 778
PATCH_25HZ_PATCHED_SHA256 = "82aa6d0f83b36bc1447f067b37e9a85578fc32a27abc98b6a747ad3741c126c4"
PATCH_25HZ_FROM = b"from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer\n"
PATCH_25HZ_TO = b""
PATCH_25HZ_OPERATION = "remove_exactly_one_25hz_tokenizer_import"

PATCH_CORE_25HZ_TARGET = "qwen_tts/core/__init__.py"
PATCH_CORE_25HZ_ORIGINAL_BYTES = 990
PATCH_CORE_25HZ_ORIGINAL_SHA256 = "1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e"
PATCH_CORE_25HZ_PATCHED_BYTES = 814
PATCH_CORE_25HZ_PATCHED_SHA256 = "c3d2f2f28cae7a0ec4d8dd8251470c8871acd2bf143d2fc239fcfbe8f2938497"
PATCH_CORE_25HZ_FROM = (
    b"from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config\n"
    b"from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\n"
)
PATCH_CORE_25HZ_TO = b""
PATCH_CORE_25HZ_OPERATION = "remove_exactly_two_core_25hz_imports"

COMPATIBILITY_PATCH_TARGETS = (PATCH_TARGET, PATCH_25HZ_TARGET, PATCH_CORE_25HZ_TARGET)
FORBIDDEN_IMPORT_MODULES = ("onnxruntime", "sox")
FORBIDDEN_IMPORT_PREFIXES = ("qwen_tts.core.tokenizer_25hz",)


class CompatibilityPatchError(RuntimeError):
    """The staged source did not satisfy the fixed patch contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def loaded_forbidden_imports() -> list[str]:
    import sys

    return sorted(
        name for name in sys.modules
        if name in FORBIDDEN_IMPORT_MODULES
        or any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    )


def patch_source_bytes(original: bytes) -> bytes:
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


def patch_25hz_source_bytes(original: bytes) -> bytes:
    if len(original) != PATCH_25HZ_ORIGINAL_BYTES or sha256_bytes(original) != PATCH_25HZ_ORIGINAL_SHA256:
        raise CompatibilityPatchError("25Hz compatibility patch original source identity drifted")
    count = original.count(PATCH_25HZ_FROM)
    if count != 1:
        raise CompatibilityPatchError(f"25Hz compatibility patch expected exactly one import, found {count}")
    patched = original.replace(PATCH_25HZ_FROM, PATCH_25HZ_TO)
    if len(patched) != PATCH_25HZ_PATCHED_BYTES or sha256_bytes(patched) != PATCH_25HZ_PATCHED_SHA256:
        raise CompatibilityPatchError("25Hz compatibility patch output identity drifted")
    if patched.count(PATCH_25HZ_FROM) != 0:
        raise CompatibilityPatchError("25Hz compatibility patch result is not exact")
    return patched


def patch_core_25hz_source_bytes(original: bytes) -> bytes:
    if len(original) != PATCH_CORE_25HZ_ORIGINAL_BYTES or sha256_bytes(original) != PATCH_CORE_25HZ_ORIGINAL_SHA256:
        raise CompatibilityPatchError("core 25Hz compatibility patch original source identity drifted")
    count = original.count(PATCH_CORE_25HZ_FROM)
    if count != 1:
        raise CompatibilityPatchError(f"core 25Hz compatibility patch expected exactly two imports, found {count}")
    patched = original.replace(PATCH_CORE_25HZ_FROM, PATCH_CORE_25HZ_TO)
    if len(patched) != PATCH_CORE_25HZ_PATCHED_BYTES or sha256_bytes(patched) != PATCH_CORE_25HZ_PATCHED_SHA256:
        raise CompatibilityPatchError("core 25Hz compatibility patch output identity drifted")
    if patched.count(PATCH_CORE_25HZ_FROM) != 0:
        raise CompatibilityPatchError("core 25Hz compatibility patch result is not exact")
    return patched


def _patch_records(originals: dict[str, bytes], patched: dict[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "status": PATCH_STATUS, "target": PATCH_TARGET, "operation": PATCH_OPERATION,
            "original_bytes": len(originals[PATCH_TARGET]), "original_sha256": sha256_bytes(originals[PATCH_TARGET]),
            "patched_bytes": len(patched[PATCH_TARGET]), "patched_sha256": sha256_bytes(patched[PATCH_TARGET]),
            "replacement_count": 1, "transformers_api": TRANSFORMERS_API,
        },
        {
            "status": PATCH_STATUS, "target": PATCH_25HZ_TARGET, "operation": PATCH_25HZ_OPERATION,
            "original_bytes": len(originals[PATCH_25HZ_TARGET]), "original_sha256": sha256_bytes(originals[PATCH_25HZ_TARGET]),
            "patched_bytes": len(patched[PATCH_25HZ_TARGET]), "patched_sha256": sha256_bytes(patched[PATCH_25HZ_TARGET]),
            "replacement_count": 1,
        },
        {
            "status": PATCH_STATUS, "target": PATCH_CORE_25HZ_TARGET, "operation": PATCH_CORE_25HZ_OPERATION,
            "original_bytes": len(originals[PATCH_CORE_25HZ_TARGET]), "original_sha256": sha256_bytes(originals[PATCH_CORE_25HZ_TARGET]),
            "patched_bytes": len(patched[PATCH_CORE_25HZ_TARGET]), "patched_sha256": sha256_bytes(patched[PATCH_CORE_25HZ_TARGET]),
            "replacement_count": 2,
        },
    ]


def _expected_patch_records() -> list[dict[str, Any]]:
    return [
        {"status": PATCH_STATUS, "target": PATCH_TARGET, "operation": PATCH_OPERATION, "original_bytes": PATCH_ORIGINAL_BYTES, "original_sha256": PATCH_ORIGINAL_SHA256, "patched_bytes": PATCHED_BYTES, "patched_sha256": PATCHED_SHA256, "replacement_count": 1, "transformers_api": TRANSFORMERS_API},
        {"status": PATCH_STATUS, "target": PATCH_25HZ_TARGET, "operation": PATCH_25HZ_OPERATION, "original_bytes": PATCH_25HZ_ORIGINAL_BYTES, "original_sha256": PATCH_25HZ_ORIGINAL_SHA256, "patched_bytes": PATCH_25HZ_PATCHED_BYTES, "patched_sha256": PATCH_25HZ_PATCHED_SHA256, "replacement_count": 1},
        {"status": PATCH_STATUS, "target": PATCH_CORE_25HZ_TARGET, "operation": PATCH_CORE_25HZ_OPERATION, "original_bytes": PATCH_CORE_25HZ_ORIGINAL_BYTES, "original_sha256": PATCH_CORE_25HZ_ORIGINAL_SHA256, "patched_bytes": PATCH_CORE_25HZ_PATCHED_BYTES, "patched_sha256": PATCH_CORE_25HZ_PATCHED_SHA256, "replacement_count": 2},
    ]


def compatibility_patch_record() -> dict[str, Any]:
    """Return the canonical record for the fixed three-patch contract."""
    return {"status": PATCH_STATUS, "operation": "apply_exactly_three_source_patches", "patch_count": 3, "patches": _expected_patch_records()}


def validate_patch_record(record: Any) -> None:
    if record != compatibility_patch_record():
        raise CompatibilityPatchError("compatibility patch evidence is not the fixed three-patch contract")


def verify_patched_source_checkout(source: Path, record: Any = None) -> dict[str, Any]:
    """Verify an already patched checkout without changing it."""
    if source.is_symlink() or not source.is_dir():
        raise CompatibilityPatchError("patched source checkout is missing or symlinked")
    source = source.resolve(strict=False)
    targets = {relative: source / relative for relative in COMPATIBILITY_PATCH_TARGETS}
    for relative, target in targets.items():
        if target.is_symlink() or not target.is_file():
            raise CompatibilityPatchError(f"patched compatibility target is missing or symlinked: {relative}")
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout.rstrip("\r\n").splitlines()
    expected_status = [f" M {relative}" for relative in sorted(COMPATIBILITY_PATCH_TARGETS)]
    if status != expected_status:
        raise CompatibilityPatchError(f"patched source checkout has unexpected status: {status!r}")
    expected = _expected_patch_records()
    for row in expected:
        target = targets[row["target"]]
        content = target.read_bytes()
        if len(content) != row["patched_bytes"] or sha256_bytes(content) != row["patched_sha256"]:
            raise CompatibilityPatchError(f"patched source identity drifted: {row['target']}")
        if row["target"] == PATCH_TARGET and PATCH_FROM in content:
            raise CompatibilityPatchError("decorator compatibility patch was reverted")
        if row["target"] == PATCH_25HZ_TARGET and PATCH_25HZ_FROM in content:
            raise CompatibilityPatchError("top-level 25Hz import compatibility patch was reverted")
        if row["target"] == PATCH_CORE_25HZ_TARGET and PATCH_CORE_25HZ_FROM in content:
            raise CompatibilityPatchError("core 25Hz import compatibility patch was reverted")
    canonical = compatibility_patch_record()
    if record is not None:
        validate_patch_record(record)
    return canonical


def patch_source_checkout(source: Path) -> dict[str, Any]:
    """Apply all three patches, or leave the clean checkout completely untouched."""
    if source.is_symlink():
        raise CompatibilityPatchError("official source checkout path must not be a symlink")
    source = source.resolve(strict=False)
    targets = {relative: source / relative for relative in COMPATIBILITY_PATCH_TARGETS}
    for relative, target in targets.items():
        try:
            target.resolve().relative_to(source)
        except ValueError as error:
            raise CompatibilityPatchError(f"compatibility patch target escapes source checkout: {relative}") from error
        if target.is_symlink() or not target.is_file():
            raise CompatibilityPatchError(f"compatibility patch target is missing or symlinked: {relative}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise CompatibilityPatchError("official source must be clean before compatibility patch")
    originals = {relative: target.read_bytes() for relative, target in targets.items()}
    patched = {
        PATCH_TARGET: patch_source_bytes(originals[PATCH_TARGET]),
        PATCH_25HZ_TARGET: patch_25hz_source_bytes(originals[PATCH_25HZ_TARGET]),
        PATCH_CORE_25HZ_TARGET: patch_core_25hz_source_bytes(originals[PATCH_CORE_25HZ_TARGET]),
    }
    temporary: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for index, (relative, target) in enumerate(targets.items()):
            path = target.with_name(f".{target.name}.{os.getpid()}.{index}.compat.tmp")
            if path.exists() or path.is_symlink():
                raise CompatibilityPatchError(f"compatibility patch temporary path already exists: {path.name}")
            temporary[relative] = path
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(patched[relative])
            os.chmod(path, target.stat().st_mode & 0o777)
        for relative, target in targets.items():
            os.replace(temporary[relative], target)
            replaced.append(relative)
    except BaseException as error:
        if replaced:
            try:
                for relative in replaced:
                    target = targets[relative]
                    if target.is_symlink():
                        raise CompatibilityPatchError(f"cannot rollback symlinked target: {relative}")
                    rollback = target.with_name(f".{target.name}.{os.getpid()}.{relative.count('/')}.rollback.tmp")
                    fd = os.open(rollback, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(originals[relative])
                    os.replace(rollback, target)
            except BaseException as rollback_error:
                raise CompatibilityPatchError(f"compatibility patch application failed and rollback failed: {rollback_error}") from error
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    try:
        after_status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.rstrip("\r\n").splitlines()
        expected_status = [f" M {relative}" for relative in sorted(COMPATIBILITY_PATCH_TARGETS)]
        if after_status != expected_status:
            raise CompatibilityPatchError(f"compatibility patch changed unexpected source paths: {after_status!r}")
        for relative, target in targets.items():
            if target.read_bytes() != patched[relative]:
                raise CompatibilityPatchError(f"compatibility patch output changed unexpectedly: {relative}")
    except BaseException as error:
        if replaced:
            try:
                for relative in replaced:
                    target = targets[relative]
                    if target.is_symlink():
                        raise CompatibilityPatchError(f"cannot rollback symlinked target: {relative}")
                    rollback = target.with_name(f".{target.name}.{os.getpid()}.{relative.count('/')}.rollback.tmp")
                    fd = os.open(rollback, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(originals[relative])
                    os.replace(rollback, target)
                restored_status = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
                if restored_status:
                    raise CompatibilityPatchError(f"rollback left source dirty: {restored_status!r}")
                for relative, target in targets.items():
                    if target.read_bytes() != originals[relative]:
                        raise CompatibilityPatchError(f"rollback identity drifted: {relative}")
            except BaseException as rollback_error:
                raise CompatibilityPatchError(f"compatibility patch validation failed and rollback failed: {rollback_error}") from error
        raise
    record = compatibility_patch_record()
    record["patches"] = _patch_records(originals, patched)
    validate_patch_record(record)
    return record


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clean_repo(root: Path, decorator: bytes, init: bytes, core: bytes) -> None:
    (root / PATCH_TARGET).parent.mkdir(parents=True)
    (root / PATCH_TARGET).write_bytes(decorator)
    (root / PATCH_25HZ_TARGET).write_bytes(init)
    (root / PATCH_CORE_25HZ_TARGET).write_bytes(core)
    _git("init", "--quiet", cwd=root)
    _git("config", "user.email", "self-test@example.invalid", cwd=root)
    _git("config", "user.name", "Qwen self-test", cwd=root)
    _git("add", PATCH_TARGET, PATCH_25HZ_TARGET, PATCH_CORE_25HZ_TARGET, cwd=root)
    _git("commit", "--quiet", "-m", "fixture", cwd=root)


def self_test_filesystem() -> None:
    """Exercise atomicity, identity, ordering, symlink and dirty-tree failures."""
    global PATCH_TARGET, COMPATIBILITY_PATCH_TARGETS, PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256
    global PATCH_25HZ_ORIGINAL_BYTES, PATCH_25HZ_ORIGINAL_SHA256, PATCH_25HZ_PATCHED_BYTES, PATCH_25HZ_PATCHED_SHA256
    global PATCH_CORE_25HZ_ORIGINAL_BYTES, PATCH_CORE_25HZ_ORIGINAL_SHA256, PATCH_CORE_25HZ_PATCHED_BYTES, PATCH_CORE_25HZ_PATCHED_SHA256
    contract = (PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256, PATCH_25HZ_ORIGINAL_BYTES, PATCH_25HZ_ORIGINAL_SHA256, PATCH_25HZ_PATCHED_BYTES, PATCH_25HZ_PATCHED_SHA256, PATCH_CORE_25HZ_ORIGINAL_BYTES, PATCH_CORE_25HZ_ORIGINAL_SHA256, PATCH_CORE_25HZ_PATCHED_BYTES, PATCH_CORE_25HZ_PATCHED_SHA256)
    decorator = b"prefix\n@check_model_inputs()\nsuffix\n"
    decorator_patched = b"prefix\n@check_model_inputs\nsuffix\n"
    init = b"prefix\nfrom .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer\nsuffix\n"
    init_patched = b"prefix\nsuffix\n"
    core = b"from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config\nfrom .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model\nfrom .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config\nfrom .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model"
    core_patched = b"from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config\nfrom .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model"
    PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256 = len(decorator), sha256_bytes(decorator)
    PATCHED_BYTES, PATCHED_SHA256 = len(decorator_patched), sha256_bytes(decorator_patched)
    PATCH_25HZ_ORIGINAL_BYTES, PATCH_25HZ_ORIGINAL_SHA256 = len(init), sha256_bytes(init)
    PATCH_25HZ_PATCHED_BYTES, PATCH_25HZ_PATCHED_SHA256 = len(init_patched), sha256_bytes(init_patched)
    PATCH_CORE_25HZ_ORIGINAL_BYTES, PATCH_CORE_25HZ_ORIGINAL_SHA256 = len(core), sha256_bytes(core)
    PATCH_CORE_25HZ_PATCHED_BYTES, PATCH_CORE_25HZ_PATCHED_SHA256 = len(core_patched), sha256_bytes(core_patched)
    try:
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-git-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core)
            record = patch_source_checkout(root)
            assert record["patch_count"] == 3
            assert (root / PATCH_TARGET).read_bytes() == decorator_patched
            assert (root / PATCH_25HZ_TARGET).read_bytes() == init_patched
            assert (root / PATCH_CORE_25HZ_TARGET).read_bytes() == core_patched
            assert verify_patched_source_checkout(root, record) == record
            reordered = {**record, "patches": list(reversed(record["patches"]))}
            try: verify_patched_source_checkout(root, reordered)
            except CompatibilityPatchError: pass
            else: raise AssertionError("reordered patch evidence was accepted")
            tampered = {**record, "patches": [dict(row) for row in record["patches"]]}
            tampered["patches"][1]["patched_bytes"] += 1
            try: verify_patched_source_checkout(root, tampered)
            except CompatibilityPatchError: pass
            else: raise AssertionError("tampered patch evidence was accepted")
            _git("add", PATCH_TARGET, PATCH_25HZ_TARGET, PATCH_CORE_25HZ_TARGET, cwd=root); _git("commit", "--quiet", "-m", "patched", cwd=root)
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("already patched source was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-dirty-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); (root / "unrelated.txt").write_text("dirty", encoding="utf-8")
            before = {(root / relative).read_bytes() for relative in COMPATIBILITY_PATCH_TARGETS}
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("pre-dirty checkout was accepted")
            assert before == {(root / relative).read_bytes() for relative in COMPATIBILITY_PATCH_TARGETS}

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-clean-unpatched-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core)
            try: verify_patched_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("clean unpatched source was accepted")

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-extra-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); record = patch_source_checkout(root)
            (root / "extra.txt").write_text("unexpected", encoding="utf-8")
            try: verify_patched_source_checkout(root, record)
            except CompatibilityPatchError: pass
            else: raise AssertionError("extra dirty source was accepted")

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-tampered-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); record = patch_source_checkout(root)
            (root / PATCH_25HZ_TARGET).write_bytes(b"tampered")
            try: verify_patched_source_checkout(root, record)
            except CompatibilityPatchError: pass
            else: raise AssertionError("tampered source was accepted")

        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-partial-state-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); patch_source_checkout(root)
            (root / PATCH_CORE_25HZ_TARGET).write_bytes(core)
            try: verify_patched_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("partial patched source was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-partial-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); (root / PATCH_25HZ_TARGET).write_bytes(b"drift")
            before = (root / PATCH_TARGET).read_bytes()
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("partial/hash-drift source was accepted")
            assert (root / PATCH_TARGET).read_bytes() == before
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-temp-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core)
            temporary = (root / PATCH_TARGET).with_name(f".{Path(PATCH_TARGET).name}.{os.getpid()}.0.compat.tmp"); temporary.write_bytes(b"must-not-clobber")
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("existing temporary path was clobbered")
            assert temporary.read_bytes() == b"must-not-clobber"
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-symlink-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); outside = root / "outside.py"; outside.write_bytes(decorator)
            target = root / PATCH_TARGET; target.unlink(); target.symlink_to(outside)
            try: patch_source_checkout(root)
            except CompatibilityPatchError: pass
            else: raise AssertionError("symlink target was accepted")
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-compat-escape-") as directory:
            root = Path(directory); _clean_repo(root, decorator, init, core); original_target = PATCH_TARGET; original_targets = COMPATIBILITY_PATCH_TARGETS; PATCH_TARGET = "../outside.py"; COMPATIBILITY_PATCH_TARGETS = (PATCH_TARGET, PATCH_25HZ_TARGET, PATCH_CORE_25HZ_TARGET)
            try:
                try: patch_source_checkout(root)
                except CompatibilityPatchError: pass
                else: raise AssertionError("path escape was accepted")
            finally: PATCH_TARGET = original_target; COMPATIBILITY_PATCH_TARGETS = original_targets
    finally:
        (PATCH_ORIGINAL_BYTES, PATCH_ORIGINAL_SHA256, PATCHED_BYTES, PATCHED_SHA256, PATCH_25HZ_ORIGINAL_BYTES, PATCH_25HZ_ORIGINAL_SHA256, PATCH_25HZ_PATCHED_BYTES, PATCH_25HZ_PATCHED_SHA256, PATCH_CORE_25HZ_ORIGINAL_BYTES, PATCH_CORE_25HZ_ORIGINAL_SHA256, PATCH_CORE_25HZ_PATCHED_BYTES, PATCH_CORE_25HZ_PATCHED_SHA256) = contract
