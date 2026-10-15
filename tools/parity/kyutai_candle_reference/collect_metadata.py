#!/usr/bin/env -S uv run --no-project --python 3.12
"""Capture a bounded, no-upload Candle CPU dependency metadata packet.

The production path is deliberately narrow: it authenticates only the pinned
Moshi source bytes, builds a moshi-core-only projection, and invokes Cargo's
metadata/tree/precise-update commands under a fresh Cargo home.  It never
builds, checks, tests, runs, downloads models, or claims license/advisory
approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


SOURCE_COMMIT = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
TARGET = "x86_64-unknown-linux-gnu"
PATCH_PACKAGE = "memmap2"
BASE_MEMMAP_VERSION = "0.9.5"
PATCH_MEMMAP_VERSION = "0.9.11"
ADVISORY_DB = "6aed84668c23d7f535077a9fe918bf1fa0f6a5bc"
MAX_FILE_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_CAPTURE_BYTES = 512 * 1024
MAX_WHOLE_LOCK_CAPTURE_BYTES = 2 * 1024 * 1024
MAX_DEADLINE_SECONDS = 600.0
MAX_LOCK_DIAGNOSTIC_BYTES = 16 * 1024
MAX_LOCK_DIAGNOSTIC_ITEMS = 16
MAX_LOCK_DIAGNOSTIC_FIELDS = 16
RUST_VERSION = "1.99.0"
RAW_BASE = f"https://raw.githubusercontent.com/kyutai-labs/moshi/{SOURCE_COMMIT}/rust/"
WHOLE_LOCK_METADATA_ARGS = ("metadata", "--format-version", "1", "--all-features")


class CaptureError(Exception):
    """Fail-closed capture error."""


# Git blob SHA-1 and exact byte length are authentication material, not a
# self-declared source receipt.  Keep this inventory immutable in production.
SOURCE_FILES: dict[str, tuple[str, int]] = {
    "Cargo.lock": ("70182edd0eb78002fac85d824c6c35b558328b0d", 150274),
    "Cargo.toml": ("2d9f68820bc169c1d68e9c001743977da412dd7a", 2309),
    "LICENSE": ("261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", 11357),
    "README.md": ("48f17b43c5244725241231853566e7e31e7fb2fe", 2813),
    "moshi-core/Cargo.toml": ("13bae99a2b2c8adeb54e75c14e8054b585625187", 662),
    "moshi-core/src/asr.rs": ("0f3ad70082f152e3250aa81ab4e0b48ffef550db", 9125),
    "moshi-core/src/batched_transformer.rs": ("10433968d5d74677cf1517f4789a78091011078b", 20399),
    "moshi-core/src/conditioner.rs": ("42e1a382ee424df9f3978f834f87c89e6e74e8a7", 6305),
    "moshi-core/src/conv.rs": ("f5c7d4d9bb5060b0a59b7fd8a21f369769c51bae", 24344),
    "moshi-core/src/kv_cache.rs": ("e909b4ee7f5f3f10a067a1dc11fa2d4d31a86aba", 11190),
    "moshi-core/src/lib.rs": ("c33b8208e9c6dfbda12ae2cae8785bff346c4c57", 718),
    "moshi-core/src/lm.rs": ("6775ec8c5453c40a4f777303bd583db76e089dd8", 37671),
    "moshi-core/src/lm_generate.rs": ("7fdcfd173a6bb8fe75f442a8969cc677ad78abba", 7319),
    "moshi-core/src/lm_generate_multistream.rs": ("5e395c22d73bc3463cb6bddd957576a02cb6f48c", 12144),
    "moshi-core/src/mimi.rs": ("603ecae8b89f9a718bdace7375b84984c2636847", 8940),
    "moshi-core/src/nn.rs": ("69eac5b4179046a181f400502d76b3444e0067ef", 6198),
    "moshi-core/src/quantization.rs": ("c8b58a4ec183f20f3a54967a4e9c9cffc5fcef8e", 13100),
    "moshi-core/src/seanet.rs": ("fe6ed49ea95b22fa9655fc32a4d367fb0f9faee2", 15785),
    "moshi-core/src/streaming.rs": ("be22d304d5738261b28d448e8d28238c1ca67404", 8156),
    "moshi-core/src/transformer.rs": ("4382d71044bd723ebbedade5043ff883683a70b0", 40480),
    "moshi-core/src/tts.rs": ("9022ccc89a11b757400a20e0c5d0102c219464cf", 8422),
    "moshi-core/src/tts_streaming.rs": ("66157784bbda4c3d500331e7d672f66783a77b54", 14398),
    "moshi-core/src/wav.rs": ("48942a3ff7b7c3c84443c00dc47f76db72394b04", 2052),
}
SOURCE_URLS = {path: RAW_BASE + path for path in SOURCE_FILES}
GPU_FEATURE_WORDS = {"cuda", "metal", "flash-attn", "mkl", "accelerate"}
FORBIDDEN_CARGO_ACTIONS = {"build", "check", "test", "run", "fetch"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not value.is_integer():
        raise CaptureError("non-integral float in JSON packet")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(_json_safe(value), sort_keys=True, indent=2) + "\n").encode()
    if len(encoded) > MAX_JSON_BYTES:
        raise CaptureError(f"JSON output too large: {path.name}")
    path.write_bytes(encoded)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_receipt(path: Path) -> dict[str, Any]:
    data = read_bytes(path, max_bytes=MAX_FILE_BYTES)
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _safe_relative(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CaptureError(f"unsafe relative path: {path}")
    return relative


def _assert_fresh_directory(path: Path) -> None:
    _assert_no_symlink_ancestors(path)
    if path.exists() or path.is_symlink():
        raise CaptureError(f"destination must not exist: {path}")
    path.mkdir(parents=True)


def _assert_no_symlink_ancestors(path: Path) -> None:
    """Reject symlinks anywhere in a destination path before writing."""
    absolute = path.absolute()
    ancestors = list(absolute.parents)
    for candidate in reversed(ancestors):
        if candidate.is_symlink():
            raise CaptureError(f"symlink ancestor is not allowed: {candidate}")
    if absolute.is_symlink():
        raise CaptureError(f"symlink destination is not allowed: {absolute}")


class _SourceIODeadline:
    """Interrupt blocking source I/O at the shared absolute deadline."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.previous_handler: Any = None
        self.previous_timer: tuple[float, float] = (0.0, 0.0)

    def __enter__(self) -> "_SourceIODeadline":
        if threading.current_thread() is not threading.main_thread():
            raise CaptureError("source I/O must run on the main thread")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureError("deadline expired before source I/O")
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        armed_timer = signal.getitimer(signal.ITIMER_REAL)
        if armed_timer != (0.0, 0.0):
            raise CaptureError("preexisting ITIMER_REAL alarm is not supported")
        self.previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

        def interrupt(_signum: int, _frame: Any) -> None:
            raise CaptureError("source I/O exceeded shared deadline")

        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        if self.previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *self.previous_timer)


def _bounded_url_fetch(url: str, deadline: float) -> bytes:
    if url not in SOURCE_URLS.values():
        raise CaptureError("source URL is outside the fixed official allowlist")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CaptureError("deadline expired before source fetch")
    request = urllib.request.Request(url, headers={"User-Agent": "vokra-candle-metadata/1"})
    try:
        with _SourceIODeadline(deadline):
            with urllib.request.urlopen(request, timeout=min(30.0, remaining)) as response:
                if response.geturl() != url:
                    raise CaptureError("source redirect changed the pinned URL")
                header = response.headers.get("Content-Length")
                if header is not None and (not header.isdigit() or int(header) > MAX_FILE_BYTES):
                    raise CaptureError("source Content-Length exceeds per-file bound")
                chunks: list[bytes] = []
                total = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise CaptureError("deadline expired during source fetch")
                    chunk = response.read(min(64 * 1024, MAX_FILE_BYTES + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise CaptureError("source exceeds per-file bound")
                    chunks.append(chunk)
                data = b"".join(chunks)
    except CaptureError:
        raise
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise CaptureError(f"source fetch failed: {url}") from exc
    if len(data) > MAX_FILE_BYTES:
        raise CaptureError(f"source exceeds per-file bound: {url}")
    return data


def fetch_and_verify_sources(
    source_root: Path,
    *,
    deadline: float,
    fetcher: Callable[[str, float], bytes] = _bounded_url_fetch,
) -> dict[str, Any]:
    total = 0
    receipt: dict[str, Any] = {
        "source_commit": SOURCE_COMMIT,
        "base_url": RAW_BASE,
        "provenance": "official-raw-fixed-git-blob-authenticated",
        "files": {},
    }
    if sum(size for _, size in SOURCE_FILES.values()) > MAX_SOURCE_BYTES:
        raise CaptureError("fixed source inventory exceeds total bound")
    for relative_name, (expected_blob, expected_size) in SOURCE_FILES.items():
        if time.monotonic() >= deadline:
            raise CaptureError("deadline expired before source fetch")
        data = fetcher(SOURCE_URLS[relative_name], deadline)
        if time.monotonic() >= deadline:
            raise CaptureError("deadline expired after source fetch")
        if len(data) != expected_size:
            raise CaptureError(f"source byte length mismatch: {relative_name}")
        if git_blob_sha1(data) != expected_blob:
            raise CaptureError(f"source Git blob mismatch: {relative_name}")
        total += len(data)
        if total > MAX_SOURCE_BYTES:
            raise CaptureError("source total exceeds bound")
        destination = source_root / _safe_relative(relative_name)
        _assert_no_symlink_ancestors(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise CaptureError(f"source destination already exists: {relative_name}")
        destination.write_bytes(data)
        receipt["files"][relative_name] = {
            "url": SOURCE_URLS[relative_name],
            "bytes": len(data),
            "git_blob_sha1": expected_blob,
            "sha256": sha256_bytes(data),
        }
    receipt["total_bytes"] = total
    return receipt


def read_bytes(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CaptureError(f"regular file required: {path}")
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise CaptureError(f"input exceeds bound: {path}")
    return data


def read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    try:
        value = tomllib.loads(read_bytes(path).decode("utf-8"))
    except Exception as exc:
        raise CaptureError(f"invalid TOML: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"TOML object required: {path}")
    return value


def replace_workspace_members(manifest: str) -> str:
    workspace = re.search(r"(?ms)^\[workspace\]\s*(.*?)(?=^\[|\Z)", manifest)
    if workspace is None:
        raise CaptureError("workspace table missing")
    body = workspace.group(1)
    replaced, count = re.subn(r"(?ms)^members\s*=\s*\[.*?\]\s*", 'members = ["moshi-core"]\n', body, count=1)
    if count != 1:
        raise CaptureError("workspace members declaration missing")
    return manifest[:workspace.start(1)] + replaced + manifest[workspace.end(1):]


def _copy_source_to_projection(source_root: Path, projection: Path) -> None:
    for relative_name in SOURCE_FILES:
        source = source_root / _safe_relative(relative_name)
        destination = projection / _safe_relative(relative_name)
        _assert_no_symlink_ancestors(destination)
        if not source.is_file() or source.is_symlink():
            raise CaptureError(f"source file missing: {relative_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise CaptureError(f"projection destination reused: {relative_name}")
        data = read_bytes(source)
        if relative_name == "Cargo.toml":
            data = replace_workspace_members(data.decode("utf-8")).encode("utf-8")
        destination.write_bytes(data)
    workspace = read_toml(projection / "Cargo.toml")
    members = workspace.get("workspace", {}).get("members")
    if members != ["moshi-core"]:
        raise CaptureError("generated projection workspace members are not exactly moshi-core")
    if (projection / "Cargo.lock").samefile(source_root / "Cargo.lock"):
        raise CaptureError("projection lock must be an owned copy")
    if (projection / "Cargo.lock").is_symlink():
        raise CaptureError("projection lock must be a regular copy")


def _package_key(package: dict[str, Any]) -> tuple[str, str, str | None]:
    name, version, source = package.get("name"), package.get("version"), package.get("source")
    if not isinstance(name, str) or not isinstance(version, str):
        raise CaptureError("malformed Cargo.lock package identity")
    if source is not None and not isinstance(source, str):
        raise CaptureError("malformed Cargo.lock source")
    return name, version, source


def lock_packages(lock: dict[str, Any]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise CaptureError("Cargo.lock package table missing")
    result: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise CaptureError("malformed Cargo.lock package")
        key = _package_key(package)
        if key in result:
            raise CaptureError(f"duplicate Cargo.lock package: {key}")
        result[key] = package
    return result


def _lock_source_kind(source: str | None) -> str:
    if source is None:
        return "none"
    if source.startswith("registry+"):
        return "registry"
    if source.startswith("git+"):
        return "git"
    if source.startswith("path+"):
        return "path"
    return "other"


def _plain_package_token(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
        return value
    return f"<sha256:{sha256_bytes(value.encode())[:12]}>"


def _plain_field_token(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.+-]+", value):
        return value
    return f"<sha256:{sha256_bytes(value.encode())[:12]}>"


def _lock_package_label(key: tuple[str, str, str | None]) -> str:
    name, version, source = key
    kind = _lock_source_kind(source)
    source_digest = "" if source is None else f":sha256:{sha256_bytes(source.encode())[:12]}"
    return f"{_plain_package_token(name)}@{_plain_package_token(version)}[{kind}{source_digest}]"


def _dependency_label(value: Any) -> str:
    """Return a plain package label without exposing Cargo source URLs."""
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    prefix = value.split("(", 1)[0].strip()
    parts = prefix.split()
    if not parts or re.fullmatch(r"[A-Za-z0-9_.+-]+", parts[0]) is None:
        return "<dependency>"
    name = parts[0]
    if len(parts) > 1 and re.fullmatch(r"[0-9][A-Za-z0-9_.+-]*", parts[1]):
        return f"{name}@{parts[1]}"
    return name


def _canonical_value(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _dependency_display(value: Any) -> str:
    return f"{_dependency_label(value)}[sha256:{sha256_bytes(_canonical_value(value))[:12]}]"


def _value_summary(value: Any) -> dict[str, Any]:
    if value is _MISSING:
        encoded = b"<missing>"
        value_type = "missing"
    else:
        encoded = _canonical_value(value)
        value_type = type(value).__name__
    return {"type": value_type, "size": len(encoded), "sha256": sha256_bytes(encoded)}


def _dependency_list_diff(old: list[Any], new: list[Any]) -> dict[str, Any]:
    old_keys = [_canonical_value(item) for item in old]
    new_keys = [_canonical_value(item) for item in new]
    old_remaining: dict[bytes, int] = {}
    for key in old_keys:
        old_remaining[key] = old_remaining.get(key, 0) + 1
    added: list[Any] = []
    for item, key in zip(new, new_keys):
        count = old_remaining.get(key, 0)
        if count:
            old_remaining[key] = count - 1
        else:
            added.append(item)
    new_remaining: dict[bytes, int] = {}
    for key in new_keys:
        new_remaining[key] = new_remaining.get(key, 0) + 1
    removed: list[Any] = []
    for item, key in zip(old, old_keys):
        count = new_remaining.get(key, 0)
        if count:
            new_remaining[key] = count - 1
        else:
            removed.append(item)
    same_membership = not added and not removed
    return {
        "kind": "dependency-list",
        "old": _value_summary(old),
        "new": _value_summary(new),
        "old_total": len(old),
        "new_total": len(new),
        "added_total": len(added),
        "removed_total": len(removed),
        "added": [_dependency_display(item) for item in added[:MAX_LOCK_DIAGNOSTIC_ITEMS]],
        "removed": [_dependency_display(item) for item in removed[:MAX_LOCK_DIAGNOSTIC_ITEMS]],
        "order_changed": same_membership and old_keys != new_keys,
        "truncated": len(added) > MAX_LOCK_DIAGNOSTIC_ITEMS or len(removed) > MAX_LOCK_DIAGNOSTIC_ITEMS,
    }


_MISSING = object()


def _field_diff(old: Any, new: Any, field: str) -> dict[str, Any]:
    if field == "dependencies" and isinstance(old, list) and isinstance(new, list):
        return _dependency_list_diff(old, new)
    return {"kind": "value", "old": _value_summary(old), "new": _value_summary(new)}


def _retained_mismatch_rows(
    original: dict[tuple[str, str, str | None], dict[str, Any]],
    projected: dict[tuple[str, str, str | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(original) & set(projected), key=_lock_package_label):
        old, new = original[key], projected[key]
        if old == new:
            continue
        fields = sorted(name for name in set(old) | set(new) if old.get(name, _MISSING) != new.get(name, _MISSING))
        rows.append({
            "package": _lock_package_label(key),
            "field_total": len(fields),
            "fields": [_plain_field_token(field) for field in fields[:MAX_LOCK_DIAGNOSTIC_FIELDS]],
            "fields_truncated": len(fields) > MAX_LOCK_DIAGNOSTIC_FIELDS,
            "field_diffs": {
                _plain_field_token(field): _field_diff(old.get(field, _MISSING), new.get(field, _MISSING), field)
                for field in fields[:MAX_LOCK_DIAGNOSTIC_FIELDS]
            },
        })
    return rows


def _compact_lock_diagnostic(payload: dict[str, Any]) -> bytes:
    prefix = b"LOCK_DIAGNOSTIC: "
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if len(prefix) + len(encoded) + 1 <= MAX_LOCK_DIAGNOSTIC_BYTES:
        return prefix + encoded + b"\n"
    keys = payload["package_keys"]
    mismatches = payload["retained_mismatches"]
    compact: dict[str, Any] = {
        "kind": payload["kind"],
        "truncated": True,
        "package_keys": {
            "original_total": keys["original_total"],
            "projected_total": keys["projected_total"],
            "added_total": keys["added_total"],
            "removed_total": keys["removed_total"],
            "added": keys["added"][:4],
            "removed": keys["removed"][:4],
            "truncated": True,
        },
        "retained_mismatches": {
            "total": mismatches["total"],
            "items": [
                {"package": item["package"], "field_total": item["field_total"], "fields": item["fields"][:8]}
                for item in mismatches["items"][:4]
            ],
            "truncated": True,
        },
    }
    encoded = json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if len(prefix) + len(encoded) + 1 <= MAX_LOCK_DIAGNOSTIC_BYTES:
        return prefix + encoded + b"\n"
    fallback = {
        "kind": payload["kind"],
        "truncated": True,
        "package_keys": {
            "original_total": keys["original_total"],
            "projected_total": keys["projected_total"],
            "added_total": keys["added_total"],
            "removed_total": keys["removed_total"],
        },
        "retained_mismatches_total": mismatches["total"],
    }
    encoded = json.dumps(fallback, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return prefix + encoded + b"\n"


def _emit_lock_diagnostic(
    original: dict[tuple[str, str, str | None], dict[str, Any]],
    projected: dict[tuple[str, str, str | None], dict[str, Any]],
) -> None:
    projected_only = sorted(set(projected) - set(original), key=_lock_package_label)
    original_only = sorted(set(original) - set(projected), key=_lock_package_label)
    mismatches = _retained_mismatch_rows(original, projected)
    payload = {
        "kind": "cargo-lock-base-validation",
        "truncated": False,
        "package_keys": {
            "original_total": len(original),
            "projected_total": len(projected),
            "added_total": len(projected_only),
            "removed_total": len(original_only),
            "added": [_lock_package_label(key) for key in projected_only[:MAX_LOCK_DIAGNOSTIC_ITEMS]],
            "removed": [_lock_package_label(key) for key in original_only[:MAX_LOCK_DIAGNOSTIC_ITEMS]],
            "truncated": len(projected_only) > MAX_LOCK_DIAGNOSTIC_ITEMS or len(original_only) > MAX_LOCK_DIAGNOSTIC_ITEMS,
        },
        "retained_mismatches": {
            "total": len(mismatches),
            "items": mismatches[:MAX_LOCK_DIAGNOSTIC_ITEMS],
            "truncated": len(mismatches) > MAX_LOCK_DIAGNOSTIC_ITEMS,
        },
    }
    sys.stderr.buffer.write(_compact_lock_diagnostic(payload))
    sys.stderr.flush()


def _metadata_package_key(package: dict[str, Any]) -> tuple[str, str, str | None]:
    name, version, source = package.get("name"), package.get("version"), package.get("source")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise CaptureError("malformed Cargo metadata package identity")
    if source is not None and not isinstance(source, str):
        raise CaptureError("malformed Cargo metadata package source")
    return name, version, source


def _metadata_name(value: str) -> str:
    # Cargo exposes dependency aliases with underscores in resolve.nodes while
    # manifests commonly spell the same crate with hyphens.
    return value.replace("-", "_")


def _lock_dependency_key(
    value: Any,
    complete: dict[tuple[str, str, str | None], dict[str, Any]],
) -> tuple[str, str, str | None]:
    if not isinstance(value, str):
        raise CaptureError("Cargo.lock dependency label is malformed")
    parts = value.split()
    if len(parts) not in (1, 2) or re.fullmatch(r"[A-Za-z0-9_.+-]+", parts[0]) is None:
        raise CaptureError("Cargo.lock dependency label is malformed")
    if len(parts) == 2 and re.fullmatch(r"[0-9][A-Za-z0-9_.+-]*", parts[1]) is None:
        raise CaptureError("Cargo.lock dependency label is malformed")
    candidates = [
        key for key in complete
        if key[0] == parts[0] and (len(parts) == 1 or key[1] == parts[1])
    ]
    if len(candidates) != 1:
        raise CaptureError("Cargo.lock dependency label is unresolved or ambiguous")
    return candidates[0]


def _lock_package_without_dependencies(package: dict[str, Any]) -> dict[str, Any]:
    result = dict(package)
    result.pop("dependencies", None)
    return result


def _metadata_declarations(package: dict[str, Any]) -> list[dict[str, Any]]:
    declarations = package.get("dependencies")
    if declarations is None:
        return []
    if not isinstance(declarations, list):
        raise CaptureError("Cargo metadata package dependencies are malformed")
    result: list[dict[str, Any]] = []
    for declaration in declarations:
        if not isinstance(declaration, dict) or not isinstance(declaration.get("name"), str):
            raise CaptureError("Cargo metadata dependency declaration is malformed")
        for field in ("rename", "source", "target", "registry"):
            value = declaration.get(field)
            if value is not None and not isinstance(value, str):
                raise CaptureError("Cargo metadata dependency declaration is malformed")
        kind = declaration.get("kind")
        if kind not in (None, "normal", "build", "dev"):
            raise CaptureError("Cargo metadata dependency kind is unsupported")
        if not isinstance(declaration.get("optional", False), bool):
            raise CaptureError("Cargo metadata dependency optional flag is malformed")
        if "uses_default_features" in declaration and not isinstance(declaration["uses_default_features"], bool):
            raise CaptureError("Cargo metadata dependency default-feature flag is malformed")
        features = declaration.get("features", [])
        if not isinstance(features, list) or not all(isinstance(feature, str) for feature in features):
            raise CaptureError("Cargo metadata dependency features are malformed")
        req = declaration.get("req")
        if not isinstance(req, str):
            raise CaptureError("Cargo metadata dependency requirement is malformed")
        result.append(declaration)
    return result


def _metadata_graph(metadata: dict[str, Any], *, require_complete_nodes: bool) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise CaptureError("Cargo metadata object required")
    packages = metadata.get("packages")
    if not isinstance(packages, list) or not packages:
        raise CaptureError("Cargo metadata packages are required and nonempty")
    packages_by_id: dict[str, dict[str, Any]] = {}
    keys_by_id: dict[str, tuple[str, str, str | None]] = {}
    ids_by_key: dict[tuple[str, str, str | None], str] = {}
    declarations_by_id: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise CaptureError("Cargo metadata package is malformed")
        package_id = package["id"]
        if package_id in packages_by_id:
            raise CaptureError("Cargo metadata package id is duplicated")
        key = _metadata_package_key(package)
        if key in ids_by_key:
            raise CaptureError("Cargo metadata package identity is duplicated")
        features = package.get("features", {})
        if not isinstance(features, dict) or not all(
            isinstance(name, str) and isinstance(values, list) and all(isinstance(value, str) for value in values)
            for name, values in features.items()
        ):
            raise CaptureError("whole Cargo package features are malformed")
        packages_by_id[package_id] = package
        keys_by_id[package_id] = key
        ids_by_key[key] = package_id
        declarations_by_id[package_id] = _metadata_declarations(package)

    members = metadata.get("workspace_members")
    if not isinstance(members, list) or len(members) != 1 or not isinstance(members[0], str):
        raise CaptureError("whole Cargo metadata must expose exactly one workspace member")
    member_id = members[0]
    if member_id not in packages_by_id or packages_by_id[member_id].get("name") != "moshi":
        raise CaptureError("whole Cargo metadata workspace member is not exactly moshi")
    resolve = metadata.get("resolve")
    if not isinstance(resolve, dict) or "root" not in resolve:
        raise CaptureError("whole Cargo metadata resolve root is required")
    root = resolve.get("root")
    if root is None:
        root = member_id
    if root != member_id:
        raise CaptureError("whole Cargo metadata root is not the moshi workspace member")
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise CaptureError("whole Cargo metadata resolve nodes are required and nonempty")
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise CaptureError("whole Cargo resolve node is malformed")
        node_id = node["id"]
        if node_id in nodes_by_id or node_id not in packages_by_id:
            raise CaptureError("whole Cargo resolve node is unknown or duplicated")
        nodes_by_id[node_id] = node
    if require_complete_nodes and set(nodes_by_id) != set(packages_by_id):
        raise CaptureError("whole Cargo metadata package and node identities differ")
    nondev_edges: dict[str, set[str]] = {}
    for node_id, node in nodes_by_id.items():
        deps = node.get("deps")
        if not isinstance(deps, list):
            raise CaptureError("whole Cargo resolve node deps are required")
        plain_dependencies = node.get("dependencies")
        if plain_dependencies is not None:
            if not isinstance(plain_dependencies, list) or not all(isinstance(value, str) for value in plain_dependencies):
                raise CaptureError("whole Cargo resolve node dependencies are malformed")
            if len(set(plain_dependencies)) != len(plain_dependencies) or set(plain_dependencies) != {edge.get("pkg") for edge in deps if isinstance(edge, dict)}:
                raise CaptureError("whole Cargo resolve node dependencies disagree with typed deps")
        package_declarations = declarations_by_id[node_id]
        features = node.get("features", [])
        if not isinstance(features, list) or not all(isinstance(value, str) for value in features):
            raise CaptureError("whole Cargo activated features are malformed")
        edge_ids: set[str] = set()
        for edge in deps:
            if not isinstance(edge, dict) or not isinstance(edge.get("pkg"), str) or not isinstance(edge.get("name"), str):
                raise CaptureError("whole Cargo dependency edge is malformed")
            target_id = edge["pkg"]
            if target_id not in packages_by_id:
                raise CaptureError("whole Cargo dependency edge references an unknown package")
            kinds = edge.get("dep_kinds")
            if not isinstance(kinds, list) or not kinds:
                raise CaptureError("whole Cargo dependency edge kinds are required")
            aliases = [
                declaration for declaration in package_declarations
                if _metadata_name(str(declaration.get("rename") or declaration["name"])) == _metadata_name(edge["name"])
            ]
            if not aliases:
                raise CaptureError("whole Cargo dependency edge has no manifest declaration")
            for dep_kind in kinds:
                if not isinstance(dep_kind, dict):
                    raise CaptureError("whole Cargo dependency kind is malformed")
                kind, target = dep_kind.get("kind"), dep_kind.get("target")
                if kind not in (None, "normal", "build", "dev"):
                    raise CaptureError("whole Cargo dependency kind is unsupported")
                if target is not None and not isinstance(target, str):
                    raise CaptureError("whole Cargo dependency target is malformed")
                matching = [
                    declaration for declaration in aliases
                    if (kind in (None, "normal") and declaration.get("kind") in (None, "normal"))
                    or (kind == "build" and declaration.get("kind") == "build")
                    or (kind == "dev" and declaration.get("kind") == "dev")
                ]
                target_package = packages_by_id[target_id]
                matching = [
                    declaration for declaration in matching
                    if declaration["name"] == target_package["name"]
                    and declaration.get("target") == target
                ]
                if not matching:
                    raise CaptureError("whole Cargo dependency kind has no manifest declaration")
                target_source = keys_by_id[target_id][2]
                if not any(declaration.get("source") == target_source for declaration in matching):
                    raise CaptureError("whole Cargo dependency source disagrees with manifest")
                if kind != "dev":
                    edge_ids.add(target_id)
        nondev_edges[node_id] = edge_ids
    return {
        "packages_by_id": packages_by_id,
        "keys_by_id": keys_by_id,
        "ids_by_key": ids_by_key,
        "nodes_by_id": nodes_by_id,
        "declarations_by_id": declarations_by_id,
        "nondev_edges": nondev_edges,
        "root": root,
    }


def _activated_dependency_aliases(package: dict[str, Any], node: dict[str, Any], declarations: list[dict[str, Any]]) -> set[str]:
    features = package.get("features", {})
    if not isinstance(features, dict) or not all(isinstance(name, str) and isinstance(values, list) and all(isinstance(value, str) for value in values) for name, values in features.items()):
        raise CaptureError("whole Cargo package features are malformed")
    activated = node.get("features", [])
    if not isinstance(activated, list) or not all(isinstance(value, str) for value in activated):
        raise CaptureError("whole Cargo activated features are malformed")
    all_aliases = {
        _metadata_name(str(declaration.get("rename") or declaration["name"])): declaration
        for declaration in declarations
    }
    aliases = {name: declaration for name, declaration in all_aliases.items() if declaration.get("optional") is True}
    active: set[str] = set()
    queue = list(activated)
    seen_features: set[str] = set()
    while queue:
        feature = queue.pop()
        if feature in seen_features:
            continue
        seen_features.add(feature)
        if feature in aliases and feature not in features:
            active.add(feature)
            continue
        if feature not in features:
            raise CaptureError("whole Cargo activated feature is undeclared")
        for expression in features[feature]:
            if expression.startswith("dep:"):
                alias = _metadata_name(expression[4:])
                if alias not in aliases:
                    raise CaptureError("whole Cargo feature references an unknown dependency alias")
                active.add(alias)
                continue
            if "/" in expression:
                alias, _child = expression.split("/", 1)
                weak = alias.endswith("?")
                alias = _metadata_name(alias[:-1] if weak else alias)
                if alias in all_aliases:
                    if not weak and alias in aliases:
                        active.add(alias)
                    continue
                if expression in features:
                    queue.append(expression)
                    continue
                raise CaptureError("whole Cargo feature expression is unsupported")
            if expression in features:
                queue.append(expression)
            elif _metadata_name(expression) in all_aliases and _metadata_name(expression) not in features:
                if _metadata_name(expression) in aliases:
                    active.add(_metadata_name(expression))
            else:
                raise CaptureError("whole Cargo feature expression is undeclared")
    return active


def _validate_cpu_metadata_against_whole(cpu_metadata: dict[str, Any], whole: dict[str, Any], base_keys: set[tuple[str, str, str | None]]) -> None:
    # Keep the existing CPU inspection as the authoritative no-GPU execution
    # guard.  The whole-feature graph is never passed to inspect_metadata.
    cpu_graph = inspect_metadata(json.dumps(cpu_metadata, separators=(",", ":")))
    cpu = _metadata_graph(cpu_metadata, require_complete_nodes=False)
    whole_keys = whole["ids_by_key"]
    cpu_key_by_id = cpu["keys_by_id"]
    for package_id in cpu_graph["active_root_closure"]:
        key = cpu_key_by_id.get(package_id)
        if key is None or key not in base_keys or key not in whole_keys:
            raise CaptureError("CPU metadata closure is outside the whole/base package graph")
    for source_id in cpu_graph["active_root_closure"]:
        source_key = cpu_key_by_id[source_id]
        whole_source_id = whole_keys.get(source_key)
        if whole_source_id is None:
            raise CaptureError("CPU metadata source package is absent from whole graph")
        if cpu["declarations_by_id"].get(source_id, []) != whole["declarations_by_id"].get(whole_source_id, []):
            raise CaptureError("CPU metadata declarations differ from whole metadata for an active package")
        for target_id in cpu["nondev_edges"].get(source_id, set()):
            target_key = cpu_key_by_id[target_id]
            if target_key not in base_keys or whole["ids_by_key"].get(target_key) not in whole["nondev_edges"].get(whole_source_id, set()):
                raise CaptureError("CPU metadata edge is outside the whole/base graph")


def _validate_semantic_base_projection(
    original: dict[tuple[str, str, str | None], dict[str, Any]],
    projected: dict[tuple[str, str, str | None], dict[str, Any]],
    *,
    cpu_metadata: dict[str, Any],
    whole_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not set(projected).issubset(original):
        _emit_lock_diagnostic(original, projected)
        raise CaptureError("projected base lock introduced an unknown package")
    whole = _metadata_graph(whole_metadata, require_complete_nodes=True)
    if set(whole["ids_by_key"]) != set(projected):
        raise CaptureError("whole Cargo metadata package identities differ from projected base lock")
    _validate_cpu_metadata_against_whole(cpu_metadata, whole, set(projected))
    removed_by_parent: dict[tuple[str, str, str | None], list[tuple[str, str, str | None]]] = {}
    for key, package in projected.items():
        if _lock_package_without_dependencies(package) != _lock_package_without_dependencies(original[key]):
            raise CaptureError("projected base lock mutated retained non-dependency package fields")
        old_dependencies = original[key].get("dependencies", [])
        new_dependencies = package.get("dependencies", [])
        if not isinstance(old_dependencies, list) or not isinstance(new_dependencies, list):
            raise CaptureError("Cargo.lock dependency list is malformed")
        old_edges = [_lock_dependency_key(value, original) for value in old_dependencies]
        new_edges = [_lock_dependency_key(value, projected) for value in new_dependencies]
        if len(set(old_edges)) != len(old_edges) or len(set(new_edges)) != len(new_edges):
            raise CaptureError("Cargo.lock dependency labels are duplicated")
        retained = [edge for edge in old_edges if edge in new_edges]
        if new_edges != retained:
            raise CaptureError("projected base lock changed canonical dependency additions, replacements, or order")
        removed_by_parent[key] = [edge for edge in old_edges if edge not in new_edges]
        whole_id = whole["ids_by_key"].get(key)
        if whole_id is None:
            raise CaptureError("whole Cargo metadata is missing a projected package")
        whole_edges = {whole["keys_by_id"][target] for target in whole["nondev_edges"].get(whole_id, set())}
        if set(new_edges) != whole_edges:
            raise CaptureError("projected base lock canonical edges differ from whole Cargo metadata")

    for parent_key, removed in removed_by_parent.items():
        if not removed:
            continue
        parent_id = whole["ids_by_key"][parent_key]
        package = whole["packages_by_id"][parent_id]
        declarations = whole["declarations_by_id"][parent_id]
        active_aliases = _activated_dependency_aliases(package, whole["nodes_by_id"][parent_id], declarations)
        for target_key in removed:
            candidates = [
                declaration for declaration in declarations
                if declaration.get("kind") in (None, "normal", "build")
                and _metadata_name(str(declaration["name"])) == _metadata_name(target_key[0])
                and declaration.get("source") == target_key[2]
            ]
            if not candidates:
                raise CaptureError("removed Cargo.lock edge lacks a non-dev manifest declaration")
            signatures = {
                (declaration.get("name"), declaration.get("rename"), declaration.get("source"), declaration.get("req"), declaration.get("kind"), declaration.get("target"))
                for declaration in candidates
            }
            if len(signatures) != 1:
                raise CaptureError("removed Cargo.lock edge has ambiguous manifest declarations")
            if any(declaration.get("optional") is not True for declaration in candidates):
                raise CaptureError("removed Cargo.lock edge has a mandatory non-dev manifest declaration")
            aliases = {_metadata_name(str(declaration.get("rename") or declaration["name"])) for declaration in candidates}
            if aliases & active_aliases:
                raise CaptureError("removed optional Cargo.lock edge is activated by whole Cargo features")
    return {
        "status": "SOURCE_ONLY_METADATA_REVIEW_PENDING",
        "projection_status": "SUBSET_UNCHANGED",
        "original_packages": len(original),
        "projected_packages": len(projected),
        "whole_metadata_packages": len(whole["packages_by_id"]),
        "whole_metadata_review": "SOURCE_ONLY_METADATA_REVIEW_PENDING",
    }


def validate_base_lock(
    original_path: Path,
    projected_path: Path,
    *,
    cpu_metadata: dict[str, Any] | None = None,
    whole_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original = lock_packages(read_toml(original_path))
    projected = lock_packages(read_toml(projected_path))
    if (cpu_metadata is None) != (whole_metadata is None):
        raise CaptureError("CPU and whole Cargo metadata evidence must be supplied together")
    if cpu_metadata is not None and whole_metadata is not None:
        return _validate_semantic_base_projection(
            original,
            projected,
            cpu_metadata=cpu_metadata,
            whole_metadata=whole_metadata,
        )
    projected_only = set(projected) - set(original)
    mismatches = _retained_mismatch_rows(original, projected)
    if projected_only or mismatches:
        _emit_lock_diagnostic(original, projected)
    if not set(projected).issubset(original):
        raise CaptureError("projected base lock introduced an unknown package")
    for key, package in projected.items():
        if package != original[key]:
            raise CaptureError(f"projected base lock mutated retained package: {key[0]}")
    return {
        "status": "SUBSET_UNCHANGED" if len(projected) < len(original) else "UNCHANGED",
        "original_packages": len(original),
        "projected_packages": len(projected),
    }


def _normalize_memmap_dependency(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value in {"memmap2", "memmap2 0.9.5", "memmap2 0.9.11"}:
        return "memmap2 @ patched-version"
    return value


def _normalized_package(package: dict[str, Any], *, memmap_record: bool = False) -> dict[str, Any]:
    result = dict(package)
    if result.get("name") == PATCH_PACKAGE and not memmap_record:
        result.pop("checksum", None)
    if isinstance(result.get("dependencies"), list):
        result["dependencies"] = sorted(_normalize_memmap_dependency(v) for v in result["dependencies"])
    return result


def validate_patch_lock(base_path: Path, patched_path: Path) -> dict[str, Any]:
    base = lock_packages(read_toml(base_path))
    patched = lock_packages(read_toml(patched_path))
    base_memmap = [p for p in base.values() if p.get("name") == PATCH_PACKAGE]
    patched_memmap = [p for p in patched.values() if p.get("name") == PATCH_PACKAGE]
    if len(base_memmap) != 1 or len(patched_memmap) != 1:
        raise CaptureError("memmap2 must occur exactly once in each lock")
    old, new = base_memmap[0], patched_memmap[0]
    if old.get("version") != BASE_MEMMAP_VERSION or new.get("version") != PATCH_MEMMAP_VERSION:
        raise CaptureError("unexpected memmap2 version transition")
    source = old.get("source")
    if not isinstance(source, str) or source != new.get("source") or not source.startswith("registry+"):
        raise CaptureError("memmap2 registry source changed or is not a registry source")
    for item, label in ((old, "base"), (new, "patched")):
        checksum = item.get("checksum")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise CaptureError(f"memmap2 {label} checksum is malformed")
    if old.get("checksum") == new.get("checksum"):
        raise CaptureError("memmap2 patched checksum did not change")
    expected_delta = {
        (PATCH_PACKAGE, BASE_MEMMAP_VERSION, source),
        (PATCH_PACKAGE, PATCH_MEMMAP_VERSION, source),
    }
    if set(base) ^ set(patched) != expected_delta:
        raise CaptureError("patch lock changed packages beyond memmap2")
    old_memmap = dict(old)
    new_memmap = dict(new)
    old_memmap.pop("version", None)
    old_memmap.pop("checksum", None)
    new_memmap.pop("version", None)
    new_memmap.pop("checksum", None)
    if _normalized_package(old_memmap, memmap_record=True) != _normalized_package(new_memmap, memmap_record=True):
        raise CaptureError("memmap2 record changed beyond version/checksum")
    for key in set(base) & set(patched):
        left, right = base[key], patched[key]
        if key[0] == PATCH_PACKAGE:
            left = dict(left); right = dict(right)
            left.pop("version", None); left.pop("checksum", None)
            right.pop("version", None); right.pop("checksum", None)
        if _normalized_package(left, memmap_record=True) != _normalized_package(right, memmap_record=True):
            raise CaptureError(f"unexpected patch mutation: {key[0]}")
    return {
        "status": "PATCH_DELTA_ONLY",
        "package": PATCH_PACKAGE,
        "from": BASE_MEMMAP_VERSION,
        "to": PATCH_MEMMAP_VERSION,
        "source": source,
        "base_checksum": old["checksum"],
        "patched_checksum": new["checksum"],
    }


def validate_cargo_argv(argv: list[str]) -> None:
    if not argv or Path(argv[0]).name != "cargo":
        raise CaptureError("only a cargo executable is allowed")
    if "--config" in argv or any(arg.startswith("--config=") for arg in argv):
        raise CaptureError("Cargo --config is forbidden")
    if len(argv) > 1 and argv[1] in FORBIDDEN_CARGO_ACTIONS:
        raise CaptureError("Cargo build/check/test/run/fetch is forbidden")
    if argv[1:] in (
        ["metadata", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features"],
        list(WHOLE_LOCK_METADATA_ARGS),
    ):
        return
    if argv[1:] in (
        ["tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "normal"],
        ["tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "build"],
        ["tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "features"],
        ["update", "-p", PATCH_PACKAGE, "--precise", PATCH_MEMMAP_VERSION],
    ):
        return
    raise CaptureError("Cargo argv is outside the fixed allowlist")


def clean_cargo_environment(cargo_home: Path) -> dict[str, str]:
    allowed = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "RUSTUP_HOME", "TMPDIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["CARGO_HOME"] = str(cargo_home)
    env["RUSTUP_TOOLCHAIN"] = RUST_VERSION
    env["CARGO_TERM_COLOR"] = "never"
    env["CARGO_NET_GIT_FETCH_WITH_CLI"] = "false"
    env.pop("HF_TOKEN", None)
    env.pop("HF_HOME", None)
    for secret_key in list(env):
        if any(word in secret_key.upper() for word in ("TOKEN", "SECRET", "API_KEY", "VAST")):
            env.pop(secret_key, None)
    return env


def _live_process_group_members(pgid: int) -> list[tuple[int, str]] | None:
    """Return non-zombie members using portable POSIX `ps` output."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,pgid=,stat="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    members: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            pid, group = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        state = fields[2]
        if group == pgid and not state.startswith("Z"):
            members.append((pid, state))
    return members


def _kill_process_group(process: subprocess.Popen[bytes], grace: float = 1.0) -> None:
    # The leader may have exited while a child remains in its start_new_session
    # group.  Never use leader liveness as proof that the group is gone.
    members = _live_process_group_members(process.pid)
    if members == []:
        return
    if members is None:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    if _live_process_group_members(process.pid) == []:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if _live_process_group_members(process.pid) == []:
            return


def run_bounded_subprocess(
    argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, validate: bool = True
) -> dict[str, Any]:
    if validate:
        validate_cargo_argv(argv)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CaptureError("deadline expired before Cargo command")
    try:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise CaptureError(f"Cargo command could not start: {argv[0]}") from exc
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    capture_bound = (
        MAX_WHOLE_LOCK_CAPTURE_BYTES
        if Path(argv[0]).name == "cargo" and tuple(argv[1:]) == WHOLE_LOCK_METADATA_ARGS
        else MAX_CAPTURE_BYTES
    )
    timed_out = False
    try:
        while selector.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                _kill_process_group(process)
                break
            for key, _ in selector.select(timeout=min(0.2, left)):
                chunk = key.fileobj.read1(65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer = captured[key.data]
                if len(buffer) + len(chunk) > capture_bound:
                    _kill_process_group(process)
                    raise CaptureError(f"Cargo {key.data} exceeded capture bound")
                buffer.extend(chunk)
        if not timed_out:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
    except BaseException:
        _kill_process_group(process)
        raise
    finally:
        selector.close()
        _kill_process_group(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    if timed_out:
        raise CaptureError("Cargo command exceeded shared deadline")
    return {
        "argv": argv,
        "returncode": process.returncode,
        "stdout": bytes(captured["stdout"]).decode("utf-8", errors="replace"),
        "stderr": bytes(captured["stderr"]).decode("utf-8", errors="replace"),
    }


def _cargo_package(metadata: dict[str, Any]) -> dict[str, Any]:
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise CaptureError("Cargo metadata packages missing")
    members = metadata.get("workspace_members")
    if not isinstance(members, list) or len(members) != 1:
        raise CaptureError("workspace must expose exactly one member")
    matched = [p for p in packages if isinstance(p, dict) and p.get("id") in members]
    if len(matched) != 1 or matched[0].get("name") != "moshi":
        raise CaptureError("workspace member is not exactly package moshi")
    return matched[0]


def inspect_metadata(metadata_text: str) -> dict[str, Any]:
    if len(metadata_text.encode()) > MAX_JSON_BYTES:
        raise CaptureError("Cargo metadata JSON exceeds bound")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise CaptureError("Cargo metadata output is not JSON") from exc
    if not isinstance(metadata, dict):
        raise CaptureError("Cargo metadata object required")
    member = _cargo_package(metadata)
    resolve = metadata.get("resolve")
    if not isinstance(resolve, dict):
        raise CaptureError("Cargo resolve is required")
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise CaptureError("Cargo resolve nodes are required and nonempty")
    if "root" not in resolve:
        raise CaptureError("Cargo resolve root is required")
    root = resolve.get("root")
    if root is None:
        # The generated projection is a virtual workspace.  Its sole member
        # is the effective root; preserve that fact instead of calling all
        # resolved/dev nodes an active closure.
        root = member["id"]
        virtual_root = True
    elif root != member["id"]:
        raise CaptureError("Cargo resolve root is not the moshi workspace member")
    else:
        virtual_root = False
    package_ids = {
        package.get("id") for package in metadata.get("packages", [])
        if isinstance(package, dict) and isinstance(package.get("id"), str)
    }
    if root not in package_ids:
        raise CaptureError("Cargo resolve root does not reference a package")
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise CaptureError("Cargo resolve node malformed")
        if node["id"] not in package_ids or node["id"] in nodes_by_id:
            raise CaptureError("Cargo resolve node references an unknown or duplicate package")
        nodes_by_id[node["id"]] = node
    if root not in nodes_by_id:
        raise CaptureError("Cargo resolve root node is missing")

    edges: dict[str, list[tuple[str, str | None, str | None]]] = {}
    for node in nodes:
        assert isinstance(node, dict)
        dependencies = node.get("deps")
        if not isinstance(dependencies, list):
            raise CaptureError("Cargo resolve node deps are required")
        typed_edges: list[tuple[str, str | None, str | None]] = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("pkg"), str):
                raise CaptureError("Cargo dependency edge malformed")
            if dependency["pkg"] not in package_ids:
                raise CaptureError("Cargo dependency references an unknown package")
            dep_kinds = dependency.get("dep_kinds")
            if not isinstance(dep_kinds, list) or not dep_kinds:
                raise CaptureError("Cargo dependency edge kinds are required")
            for dep_kind in dep_kinds:
                if not isinstance(dep_kind, dict):
                    raise CaptureError("Cargo dependency kind malformed")
                kind = dep_kind.get("kind")
                target = dep_kind.get("target")
                if kind not in (None, "normal", "build", "dev"):
                    raise CaptureError(f"unknown Cargo dependency kind: {kind}")
                if target is not None and not isinstance(target, str):
                    raise CaptureError("Cargo dependency target must be string or null")
                if target is not None and re.fullmatch(r"[A-Za-z0-9_]+-[A-Za-z0-9_]+-[A-Za-z0-9_]+", target) and target != TARGET:
                    raise CaptureError(f"Cargo dependency target is outside {TARGET}: {target}")
                typed_edges.append((dependency["pkg"], kind, target))
        edges[node["id"]] = typed_edges
    for source_id, typed_edges in edges.items():
        for dependency, _kind, _target in typed_edges:
            if dependency not in nodes_by_id:
                raise CaptureError(f"Cargo dependency node is absent from resolve.nodes: {source_id} -> {dependency}")

    active_ids: set[str] = set()
    normal_ids: set[str] = set()
    build_ids: set[str] = set()
    queue = [root]
    while queue:
        current = queue.pop()
        if current in active_ids:
            continue
        active_ids.add(current)
        for dependency, kind, _target in edges[current]:
            if kind == "dev":
                continue
            if kind in (None, "normal"):
                normal_ids.add(dependency)
            elif kind == "build":
                build_ids.add(dependency)
            if dependency not in active_ids:
                queue.append(dependency)
    if not active_ids:
        raise CaptureError("Cargo active root closure is empty")

    active_features: dict[str, list[str]] = {}
    gpu_active: list[str] = []
    for node_id in sorted(active_ids):
        node = nodes_by_id[node_id]
        features = node.get("features", [])
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise CaptureError("Cargo activated features malformed")
        active_features[node["id"]] = sorted(features)
        for feature in features:
            low = feature.lower()
            if low in GPU_FEATURE_WORDS or any(low.endswith("/" + word) for word in GPU_FEATURE_WORDS):
                gpu_active.append(f"{node['id']}:{feature}")
    if gpu_active:
        raise CaptureError("GPU feature became active: " + ",".join(sorted(gpu_active)))
    return {
        "workspace_member": {"id": member["id"], "name": member["name"], "manifest_path": member.get("manifest_path")},
        "workspace_members": metadata["workspace_members"],
        "resolved_root": root,
        "virtual_workspace_root": virtual_root,
        "resolved_node_ids": sorted(nodes_by_id),
        "active_root_closure": sorted(active_ids),
        "target": TARGET,
        "normal_package_ids": sorted(normal_ids),
        "build_package_ids": sorted(build_ids),
        "actual_activated_features": active_features,
        "gpu_features_active": [],
        "whole_lock_is_active_closure": False,
    }


def _toolchain_probe(cargo_path: str, *, cwd: Path, env: dict[str, str], deadline: float) -> dict[str, Any]:
    rustc = shutil.which("rustc")
    if rustc is None or Path(rustc).name != "rustc":
        raise CaptureError("fixed rustc executable not found")
    result = run_bounded_subprocess(
        [rustc, "--version", "--verbose"], cwd=cwd, env=env, deadline=min(deadline, time.monotonic() + 10), validate=False
    )
    if result["returncode"] != 0:
        raise CaptureError("rustc toolchain probe failed")
    verbose = result["stdout"]
    first_line = verbose.splitlines()[0] if verbose.splitlines() else ""
    version_match = re.fullmatch(r"rustc ([0-9]+\.[0-9]+\.[0-9]+)(?: \([^)]*\))?", first_line.strip())
    host_match = re.search(r"(?m)^host:\s*(\S+)\s*$", verbose)
    if version_match is None or version_match.group(1) != RUST_VERSION:
        raise CaptureError(f"unsupported rustc version; expected {RUST_VERSION}")
    if host_match is None or host_match.group(1) != TARGET:
        raise CaptureError(f"rustc host does not match {TARGET}")
    return {
        "cargo_executable": str(Path(cargo_path).resolve()),
        "rustc_executable": str(Path(rustc).resolve()),
        "rustc_version": version_match.group(1),
        "host": host_match.group(1),
        "rustc_version_verbose": verbose,
    }


def capture(
    output_root: Path,
    *,
    deadline_seconds: float = 600.0,
    source_fetcher: Callable[[str, float], bytes] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    if (platform_name or platform.system()) != "Linux":
        raise CaptureError("production capture is Linux-only")
    if os.environ.get("VOKRA_CANDLE_REMOTE_LINUX") != "1":
        raise CaptureError("explicit VOKRA_CANDLE_REMOTE_LINUX=1 marker is required")
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0 or deadline_seconds > MAX_DEADLINE_SECONDS:
        raise CaptureError("deadline must be >0 and <=600 seconds")
    deadline = time.monotonic() + deadline_seconds
    _assert_fresh_directory(output_root)
    source_root = output_root / "source" / "rust"
    projection = output_root / "projection"
    base = output_root / "base"
    patched = output_root / "patched"
    for path in (source_root, projection, base, patched):
        path.mkdir(parents=True)
    source_receipt = fetch_and_verify_sources(
        source_root,
        deadline=deadline,
        fetcher=source_fetcher or _bounded_url_fetch,
    )
    write_json(output_root / "source_receipt.json", source_receipt)
    _copy_source_to_projection(source_root, projection)
    original_lock = source_root / "Cargo.lock"
    base_lock = base / "Cargo.lock"
    patched_lock = patched / "Cargo.lock"
    cargo = shutil.which("cargo")
    if cargo is None or Path(cargo).name != "cargo":
        raise CaptureError("fixed cargo executable not found")
    toolchain: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="vokra-cargo-home-", dir=str(output_root)) as cargo_home:
        env = clean_cargo_environment(Path(cargo_home))
        commands = [
            [cargo, "metadata", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features"],
            [cargo, "tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "normal"],
            [cargo, "tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "build"],
            [cargo, "tree", "--package", "moshi", "--target", TARGET, "--no-default-features", "--edges", "features"],
        ]
        # Verify the exact toolchain before permitting any Cargo invocation.
        toolchain = _toolchain_probe(cargo, cwd=projection, env=env, deadline=deadline)
        initial = [run_bounded_subprocess(cmd, cwd=projection, env=env, deadline=deadline) for cmd in commands]
        for result in initial:
            if result["returncode"] != 0:
                raise CaptureError(f"Cargo command failed: {result['argv']}")
        graph = inspect_metadata(initial[0]["stdout"])
        try:
            initial_metadata = json.loads(initial[0]["stdout"])
        except json.JSONDecodeError as exc:
            raise CaptureError("Cargo metadata output is not JSON") from exc
        if not isinstance(initial_metadata, dict):
            raise CaptureError("Cargo metadata object required")
        write_json(
            output_root / "initial_metadata.json",
            {
                "status": "INITIAL_METADATA_UNREVIEWED",
                "source_commit": SOURCE_COMMIT,
                "target": TARGET,
                "toolchain": toolchain,
                "metadata": initial_metadata,
                "graph": graph,
                "approval": None,
                "upload": "NO_UPLOAD",
                "model_execution": "NOT_AUTHORIZED",
            },
        )
        base_lock.write_bytes(read_bytes(projection / "Cargo.lock", max_bytes=MAX_FILE_BYTES))
        if base_lock.is_symlink():
            raise CaptureError("base lock must be regular")
        whole_lock_command = [cargo, *WHOLE_LOCK_METADATA_ARGS]
        projection_lock_before = read_bytes(projection / "Cargo.lock", max_bytes=MAX_FILE_BYTES)
        whole_lock_result = run_bounded_subprocess(
            whole_lock_command,
            cwd=projection,
            env=env,
            deadline=deadline,
        )
        if whole_lock_result["returncode"] != 0:
            raise CaptureError(f"Cargo command failed: {whole_lock_result['argv']}")
        try:
            whole_lock_metadata = json.loads(whole_lock_result["stdout"])
        except json.JSONDecodeError as exc:
            raise CaptureError("Cargo all-features metadata output is not JSON") from exc
        if not isinstance(whole_lock_metadata, dict):
            raise CaptureError("Cargo all-features metadata object required")
        projection_lock_after = read_bytes(projection / "Cargo.lock", max_bytes=MAX_FILE_BYTES)
        whole_lock_evidence = {
            "status": "WHOLE_LOCK_METADATA_UNREVIEWED",
            "source_commit": SOURCE_COMMIT,
            "target": None,
            "platform_filter": None,
            "cpu_target": TARGET,
            "toolchain": toolchain,
            "command": whole_lock_command,
            "metadata": whole_lock_metadata,
            "cpu_execution_profile": "NOT_CPU_EXECUTION_PROFILE",
            "gpu_execution": "NOT_AUTHORIZED",
            "model_execution": "NOT_AUTHORIZED",
            "approval": None,
            "upload": "NO_UPLOAD",
            "projection_lock_before": {
                "bytes": len(projection_lock_before),
                "sha256": sha256_bytes(projection_lock_before),
            },
            "projection_lock_after": {
                "bytes": len(projection_lock_after),
                "sha256": sha256_bytes(projection_lock_after),
            },
            "postcommand_projection_lock": projection_lock_after.decode("utf-8"),
        }
        write_json(output_root / "whole_lock_metadata.json", whole_lock_evidence)
        if projection_lock_after != projection_lock_before:
            raise CaptureError("Cargo all-features metadata mutated projection Cargo.lock")
        base_result = validate_base_lock(
            original_lock,
            base_lock,
            cpu_metadata=initial_metadata,
            whole_metadata=whole_lock_metadata,
        )
        update = [cargo, "update", "-p", PATCH_PACKAGE, "--precise", PATCH_MEMMAP_VERSION]
        update_result = run_bounded_subprocess(update, cwd=projection, env=env, deadline=deadline)
        if update_result["returncode"] != 0:
            raise CaptureError("Cargo precise memmap2 update failed")
        patched_lock.write_bytes(read_bytes(projection / "Cargo.lock", max_bytes=MAX_FILE_BYTES))
        patch_result = validate_patch_lock(base_lock, patched_lock)
        post_metadata = run_bounded_subprocess(commands[0], cwd=projection, env=env, deadline=deadline)
        if post_metadata["returncode"] != 0:
            raise CaptureError("post-update Cargo metadata failed")
        post_graph = inspect_metadata(post_metadata["stdout"])
        tree_results = [run_bounded_subprocess(cmd, cwd=projection, env=env, deadline=deadline) for cmd in commands[1:]]
        if any(result["returncode"] != 0 for result in tree_results):
            raise CaptureError("post-update Cargo tree failed")
        if read_bytes(projection / "Cargo.lock", max_bytes=MAX_FILE_BYTES) != read_bytes(patched_lock, max_bytes=MAX_FILE_BYTES):
            raise CaptureError("projection Cargo.lock changed after patched lock receipt")
    if toolchain is None:
        raise CaptureError("toolchain probe did not run")
    projection_core = read_toml(projection / "moshi-core/Cargo.toml")
    core_license = projection_core.get("package", {}).get("license")
    packet = {
        "status": "METADATA_CAPTURED_REVIEW_PENDING",
        "source_commit": SOURCE_COMMIT,
        "target": TARGET,
        "current_features": [],
        "workspace_members": ["moshi-core"],
        "workspace_member_packages": ["moshi"],
        "source_receipt": source_receipt,
        "license_declarations": {
            "workspace": read_toml(projection / "Cargo.toml").get("workspace", {}).get("package", {}).get("license"),
            "moshi": {
                "license": core_license if isinstance(core_license, str) else None,
                "license_workspace": core_license.get("workspace") if isinstance(core_license, dict) else None,
            },
        },
        "source_license_receipt": {
            "LICENSE": source_receipt["files"]["LICENSE"],
            "signoff": None,
        },
        "registry_checksums": {
            "base": sorted(_registry_checksums(read_toml(base_lock))),
            "patched": sorted(_registry_checksums(read_toml(patched_lock))),
        },
        "lock_receipts": {
            "original": file_receipt(original_lock),
            "projected_base": file_receipt(base_lock),
            "patched": file_receipt(patched_lock),
            "final_projection": file_receipt(projection / "Cargo.lock"),
        },
        "toolchain": toolchain,
        "initial": {"graph": graph, "trees": {edge: result["stdout"] for edge, result in zip(("normal", "build", "features"), initial[1:])}},
        "post_update": {"graph": post_graph, "trees": {edge: result["stdout"] for edge, result in zip(("normal", "build", "features"), tree_results)}},
        "base_lock": base_result,
        "patch_lock": patch_result,
        "known_rustsec_candidate": {"package": PATCH_PACKAGE, "patched_version": PATCH_MEMMAP_VERSION, "database_revision": ADVISORY_DB, "status": "CANDIDATE_ONLY"},
        "license_audit": "NOT_RUN",
        "advisory_audit": "NOT_RUN",
        "model_execution": "NOT_AUTHORIZED",
        "upload": "NO_UPLOAD",
        "approval": None,
        "remote_job_contract": {"input": "fresh owned output root + explicit Linux marker", "output": "small metadata JSON and lock receipts", "lifecycle": "allocation/destroy handled by separate review"},
    }
    write_json(output_root / "metadata.json", packet)
    return packet


def _registry_checksums(lock: dict[str, Any]) -> Iterable[str]:
    for package in lock_packages(lock).values():
        source = package.get("source")
        checksum = package.get("checksum")
        if source is not None:
            if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise CaptureError(f"registry checksum missing: {package.get('name')}")
            yield f"{package['name']}@{package['version']}={checksum}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    try:
        capture(args.output_root, deadline_seconds=args.deadline_seconds)
    except (CaptureError, OSError, ValueError) as exc:
        print(f"CAPTURE_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("METADATA_CAPTURED_REVIEW_PENDING: no upload/model execution/license or advisory PASS claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
