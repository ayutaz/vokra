#!/usr/bin/env python3
"""Static, fail-closed inventory for the pinned Moshi e6 Rust Mimi core.

This module never imports, builds, resolves, or executes the audited Rust
source.  It creates only a bounded derived workspace and a fact inventory.
The result is STATIC_SOURCE_INVENTORY / OPEN, never an owner approval, native
closure proof, model compatibility result, or parity result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import selectors
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

PINNED_REV = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
OFFICIAL_ORIGINS = {
    "https://github.com/kyutai-labs/moshi.git",
    "https://github.com/kyutai-labs/moshi",
}
EXPECTED_LOCK_SHA256 = "bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab"
COLLECTOR_STATUS = "STATIC_SOURCE_INVENTORY"
MAX_READ = 16 * 1024 * 1024
MAX_GIT_OUTPUT = 2 * 1024 * 1024
MAX_FILES = 512
MAX_TOTAL_BYTES = 32 * 1024 * 1024
FORBIDDEN_MARKERS = (
    "cuda", "metal", "flash-attn", "bindgen_cuda", "candle-metal-kernels",
    "cudarc", "ug-cuda", "ug-metal",
)
REQUIRED_LICENSES = ("LICENSE-APACHE", "LICENSE-MIT")
DIRECT_CORE_DEPS = ("candle", "candle-nn", "candle-transformers", "rayon", "serde", "tracing", "candle-flash-attn")
CORE_PACKAGE_NAME = "moshi"
SOURCE_DECLARATIONS = (
    "rust/Cargo.toml", "rust/moshi-core/Cargo.toml", "rust/Cargo.lock",
    "rust/README.md", "LICENSE-APACHE", "LICENSE-MIT",
)


class AuditError(RuntimeError):
    pass


def _bounded_bytes(path: Path, limit: int = MAX_READ) -> bytes:
    """Read one regular file without following symlinks or accepting swaps."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AuditError(f"cannot open regular file: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise AuditError(f"non-regular or oversized file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise AuditError(f"file grew beyond bound: {path}")
        after = os.fstat(fd)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity:
            raise AuditError(f"file changed during read: {path}")
        try:
            named = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise AuditError(f"path vanished during read: {path}") from exc
        named_identity = (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns)
        if named_identity != before_identity:
            raise AuditError(f"path identity changed during read: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _safe_rel(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw or len(raw.encode()) > 512:
        raise AuditError(f"unsafe relative path: {raw!r}")
    if raw.startswith("/") or raw.startswith("./") or "//" in raw:
        raise AuditError(f"non-canonical relative path: {raw!r}")
    parts = raw.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise AuditError(f"unsafe path component: {raw!r}")
    if str(PurePosixPath(raw)) != raw:
        raise AuditError(f"non-canonical path: {raw!r}")
    return raw


def _ensure_parent_chain(root: Path, rel: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise AuditError(f"symlink/non-directory root: {root}")
    current = root
    for part in rel.split("/")[:-1]:
        current = current / part
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise AuditError(f"symlink/non-directory parent: {current}")
    return root / rel


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _command_bytes(command: list[str], limit: int, timeout: float = 10.0) -> tuple[bytes, bytes]:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        selector = selectors.DefaultSelector()
    except BaseException:
        proc.kill()
        proc.wait()
        proc.stdout.close()
        proc.stderr.close()
        raise
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                raise AuditError(f"command timed out: {' '.join(command)}")
            for key, _ in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffer = buffers[key.data]
                buffer.extend(chunk)
                if len(buffer) > limit:
                    proc.kill()
                    raise AuditError(f"command output exceeds bound: {' '.join(command)}")
        returncode = proc.wait(timeout=max(0.1, deadline - time.monotonic()))
    except AuditError:
        proc.kill()
        proc.wait()
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        proc.kill()
        proc.wait()
        raise AuditError(f"command failed to complete: {' '.join(command)}") from exc
    finally:
        selector.close()
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if returncode != 0:
        raise AuditError(f"command failed ({returncode}): {' '.join(command)}")
    return bytes(buffers["stdout"]), bytes(buffers["stderr"])


def _git_bytes(root: Path, args: list[str], limit: int = MAX_GIT_OUTPUT) -> bytes:
    stdout, _ = _command_bytes(["git", "-C", str(root), *args], limit)
    return stdout


def _git(root: Path, args: list[str]) -> str:
    return _git_bytes(root, args).decode("utf-8")


def _git_facts(root: Path) -> dict[str, Any]:
    head = _git(root, ["rev-parse", "HEAD"]).strip()
    origin = _git(root, ["remote", "get-url", "origin"]).strip()
    status = _git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    return {"revision": head, "origin": origin, "clean": status == "", "status_bytes": len(status.encode())}


def _tracked_source_paths(root: Path) -> list[str]:
    output = _git(root, ["ls-files", "-z", "rust/moshi-core/src"])
    raw_paths = output.split("\0")
    paths: list[str] = []
    for raw in raw_paths:
        if not raw:
            continue
        rel = _safe_rel(raw)
        if not rel.startswith("rust/moshi-core/src/") or not rel.endswith(".rs"):
            raise AuditError(f"unexpected tracked core path: {rel}")
        paths.append(rel)
    if not paths or len(paths) > MAX_FILES:
        raise AuditError("unexpected core source file count")
    return sorted(paths)


def _file_row(root: Path, rel: str, data: bytes | None = None) -> dict[str, Any]:
    rel = _safe_rel(rel)
    path = _ensure_parent_chain(root, rel)
    if data is None:
        data = _bounded_bytes(path)
    return {"path": rel, "bytes": len(data), "sha256": _sha(data)}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items()) + " }"
    if isinstance(value, int):
        return str(value)
    raise AuditError(f"unsupported TOML value: {type(value).__name__}")


def _minimal_workspace(root: Path, destination: Path, cargo_root: dict[str, Any], cargo_core: dict[str, Any], source_paths: list[str], snapshots: dict[str, bytes]) -> dict[str, Any]:
    workspace_config = cargo_root.get("workspace", {})
    package = workspace_config.get("package", {})
    dependencies = workspace_config.get("dependencies", {})
    core_deps = cargo_core.get("dependencies", {})
    inherited = [name for name in DIRECT_CORE_DEPS if name in core_deps and name in dependencies]
    if set(inherited) != set(DIRECT_CORE_DEPS):
        raise AuditError(f"moshi-core inherited dependency set mismatch: {inherited}")
    lines = ["[workspace]", 'members = ["moshi-core"]', 'resolver = "2"', "", "[workspace.package]"]
    for key in ("version", "edition", "license", "description", "repository", "keywords", "categories"):
        if key not in package:
            raise AuditError(f"missing workspace package field: {key}")
        lines.append(f"{key} = {_toml_value(package[key])}")
    lines.extend(["", "[workspace.dependencies]"])
    for name in DIRECT_CORE_DEPS:
        lines.append(f"{name} = {_toml_value(dependencies[name])}")
    lines.extend(["", "[profile.release]", "debug = true", "", "[profile.release-no-debug]", 'inherits = "release"', "debug = false", ""])
    workspace = destination
    workspace.mkdir()
    member = workspace / "moshi-core"
    member.mkdir()
    (member / "src").mkdir()
    _write_noclobber(workspace / "Cargo.toml", "\n".join(lines).encode())
    for rel in source_paths:
        if not rel.startswith("rust/moshi-core/src/"):
            continue
        target = workspace / rel.removeprefix("rust/")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_noclobber(target, snapshots[rel])
    for rel in ("rust/moshi-core/Cargo.toml", "rust/README.md"):
        target = workspace / rel.removeprefix("rust/")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_noclobber(target, snapshots[rel])
    source_manifest = snapshots["rust/moshi-core/Cargo.toml"]
    derived_manifest = _bounded_bytes(workspace / "Cargo.toml")
    files = sorted(str(path.relative_to(workspace)) for path in workspace.rglob("*") if path.is_file())
    copied_paths = ["rust/moshi-core/Cargo.toml", "rust/README.md"] + [rel for rel in source_paths if rel.startswith("rust/moshi-core/src/")]
    return {
        "path": str(workspace),
        "files": files,
        "sha256": _sha(derived_manifest),
        "source_core_manifest_sha256": _sha(source_manifest),
        "derived_manifest_sha256": _sha(derived_manifest),
        "manifest_delta": "DERIVED_MINIMAL_WORKSPACE_MANIFEST",
        "copied_source_paths": sorted(copied_paths),
        "copied_source_sha256": {rel: _sha(snapshots[rel]) for rel in sorted(copied_paths)},
    }


def _write_noclobber(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise AuditError(f"refusing to overwrite output: {path}")
    _validate_output_parent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_parent(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise AuditError(f"short write: {path}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_output_parent(path: Path) -> None:
    chain = list(path.parent.parents)[::-1] + [path.parent]
    for parent in chain:
        if parent.exists() and parent.is_symlink():
            raise AuditError(f"output parent is symlink: {parent}")


def _validate_path_ancestors(path: Path) -> None:
    for candidate in [path, *path.parents]:
        if candidate.is_symlink():
            raise AuditError(f"path ancestor is symlink: {candidate}")


def _registry_inventory(metadata: dict[str, Any], registry_root: Path | None) -> dict[str, Any]:
    if registry_root is None:
        return {"status": "OPEN_REGISTRY_SOURCE_NOT_PROVIDED", "packages": []}
    if registry_root.is_symlink() or not registry_root.is_dir():
        raise AuditError("registry source root must be a regular directory")
    rows: list[dict[str, Any]] = []
    missing = 0
    for package in metadata.get("active_packages", []):
        name = package["name"]
        version = package["version"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+-]*", version):
            raise AuditError(f"unsafe registry package identity: {name}@{version}")
        expected_prefix = f"{name}-{version}/"
        archive = registry_root / f"{name}-{version}.crate"
        if archive.is_file() and not archive.is_symlink():
            archive_bytes = _bounded_bytes(archive, 64 * 1024 * 1024)
            archive_sha = _sha(archive_bytes)
            lock_checksum = package.get("lock_checksum")
            if not isinstance(lock_checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", lock_checksum):
                rows.append({"name": name, "version": version, "status": "BLOCKED_LOCK_CHECKSUM_MISSING", "archive_sha256": archive_sha, "lock_checksum": lock_checksum})
                missing += 1
                continue
            if archive_sha != lock_checksum:
                rows.append({"name": name, "version": version, "status": "BLOCKED_ARCHIVE_SHA_MISMATCH", "archive_sha256": archive_sha, "lock_checksum": lock_checksum})
                missing += 1
                continue
            files: list[dict[str, Any]] = []
            seen: set[str] = set()
            seen_normalized: set[str] = set()
            member_count = 0
            total_member_bytes = 0
            try:
                expanded = bytearray()
                with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes), mode="rb") as gz:
                    while True:
                        chunk = gz.read(1024 * 1024)
                        if not chunk:
                            break
                        expanded.extend(chunk)
                        if len(expanded) > MAX_TOTAL_BYTES:
                            raise AuditError(f"registry archive compressed payload expands beyond bound: {archive}")
                with tarfile.open(fileobj=io.BytesIO(bytes(expanded)), mode="r:") as tar:
                    for member in tar:
                        member_count += 1
                        if member_count > MAX_FILES * 4:
                            raise AuditError(f"registry archive member count exceeds bound: {archive}")
                        raw_name = member.name
                        if raw_name in seen:
                            raise AuditError(f"duplicate registry archive member: {raw_name}")
                        seen.add(raw_name)
                        if raw_name.startswith("/") or "\\" in raw_name or ".." in PurePosixPath(raw_name).parts or not raw_name.startswith(expected_prefix):
                            raise AuditError(f"unsafe registry archive path: {raw_name}")
                        inner = raw_name[len(expected_prefix):].rstrip("/")
                        if inner:
                            _safe_rel(inner)
                        normalized_name = raw_name.rstrip("/")
                        if normalized_name in seen_normalized:
                            raise AuditError(f"duplicate normalized registry archive member: {raw_name}")
                        seen_normalized.add(normalized_name)
                        if member.issym() or member.islnk():
                            raise AuditError(f"registry archive link member: {raw_name}")
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise AuditError(f"registry archive non-regular member: {raw_name}")
                        total_member_bytes += member.size
                        if total_member_bytes > MAX_TOTAL_BYTES:
                            raise AuditError(f"registry archive expanded size exceeds bound: {archive}")
                        basename = Path(raw_name).name.lower()
                        if basename == "cargo.toml" or basename in {"license", "license-apache", "license-mit", "copying", "notice", "build.rs"}:
                            if member.size > MAX_READ:
                                raise AuditError(f"oversized registry archive member: {raw_name}")
                            extracted = tar.extractfile(member)
                            if extracted is None:
                                raise AuditError(f"unreadable registry archive member: {raw_name}")
                            data = extracted.read(member.size + 1)
                            if len(data) > member.size or len(data) > MAX_READ:
                                raise AuditError(f"oversized registry archive member: {raw_name}")
                            files.append({"path": raw_name, "bytes": len(data), "sha256": _sha(data)})
            except (OSError, tarfile.TarError) as exc:
                raise AuditError(f"invalid registry archive: {archive}") from exc
            rows.append({"name": name, "version": version, "status": "AUTHENTICATED_ARCHIVE_INSPECTED_OPEN", "archive_sha256": archive_sha, "lock_checksum": lock_checksum, "files": files, "license_notice_open": True, "member_count": member_count, "expanded_bytes": total_member_bytes})
            continue
        candidate = registry_root / f"{name}-{version}"
        if candidate.is_symlink() or not candidate.is_dir():
            missing += 1
            rows.append({"name": name, "version": version, "status": "OPEN_SOURCE_DIRECTORY_MISSING", "lock_checksum": package.get("lock_checksum")})
            continue
        files: list[dict[str, Any]] = []
        for child in sorted(candidate.iterdir()):
            if child.name == "Cargo.toml" or child.name.lower() in {"license", "license-apache", "license-mit", "copying", "notice", "build.rs"}:
                if child.is_symlink() or not child.is_file():
                    raise AuditError(f"registry audit path is not regular: {child}")
                data = _bounded_bytes(child)
                files.append({"path": child.name, "bytes": len(data), "sha256": _sha(data)})
        manifest = next((row for row in files if row["path"] == "Cargo.toml"), None)
        rows.append({"name": name, "version": version, "status": "UNAUTHENTICATED_OPEN_DIRECTORY", "lock_checksum": package.get("lock_checksum"), "files": files, "manifest_present": manifest is not None, "build_rs_present": any(row["path"] == "build.rs" for row in files)})
        missing += 1
    return {"status": "OPEN_MISSING_SOURCE_ROWS" if missing else "SOURCE_BYTES_INSPECTED_OPEN", "packages": rows, "missing_count": missing}


def _read_json(path: Path) -> dict[str, Any]:
    data = _bounded_bytes(path, 4 * 1024 * 1024)
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid metadata JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError("metadata root must be an object")
    return value


def _lock_rows(lock_data: dict[str, Any]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    rows: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in lock_data.get("package", []):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("name", "version")):
            raise AuditError("malformed Cargo.lock package row")
        if row.get("source") is not None and not isinstance(row.get("source"), str):
            raise AuditError("malformed Cargo.lock source")
        if row.get("checksum") is not None and (not isinstance(row.get("checksum"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["checksum"])):
            raise AuditError("malformed Cargo.lock checksum")
        key = (row["name"], row["version"], row.get("source"))
        if key in rows:
            raise AuditError(f"duplicate Cargo.lock package: {key}")
        rows[key] = row
    return rows


def _metadata_inventory(metadata: dict[str, Any], lock_data: dict[str, Any], expected_core_version: str | None = None) -> dict[str, Any]:
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(packages, list) or not isinstance(resolve, dict):
        raise AuditError("metadata must contain packages and resolve")
    workspace_members = metadata.get("workspace_members")
    if not isinstance(workspace_members, list) or len(workspace_members) != 1 or not isinstance(workspace_members[0], str):
        raise AuditError("metadata must identify exactly one workspace member")
    if "workspace_default_members" in metadata:
        defaults = metadata["workspace_default_members"]
        if not isinstance(defaults, list) or defaults != workspace_members:
            raise AuditError("workspace default members do not match sole member")
    lock = _lock_rows(lock_data)
    package_by_id: dict[str, dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise AuditError("malformed metadata package")
        if package["id"] in package_by_id:
            raise AuditError(f"duplicate metadata package: {package['id']}")
        if not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise AuditError("metadata package name/version missing")
        if not isinstance(package.get("manifest_path"), str):
            raise AuditError("metadata package manifest path missing")
        package_by_id[package["id"]] = package
    root_id = workspace_members[0]
    root_package = package_by_id.get(root_id)
    if root_package is None or root_package["name"] != CORE_PACKAGE_NAME or (expected_core_version is not None and root_package["version"] != expected_core_version) or not root_package["manifest_path"].replace("\\", "/").endswith("/moshi-core/Cargo.toml"):
        raise AuditError("metadata root package is not the sole moshi-core member")
    root_source = root_package.get("source")
    if root_source is not None:
        raise AuditError("malformed package source: moshi root source must be null")
    root_lock = lock.get((root_package["name"], root_package["version"], None))
    if root_lock is None:
        raise AuditError("moshi root package is absent from Cargo.lock")
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list):
        raise AuditError("metadata resolve.nodes missing")
    resolve_root = resolve.get("root")
    if not isinstance(resolve_root, str) or resolve_root != root_id:
        raise AuditError("resolve.root does not identify the sole workspace member")
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] in node_by_id:
            raise AuditError("duplicate or malformed resolve node")
        node_by_id[node["id"]] = node
    if root_id not in node_by_id:
        raise AuditError("workspace root absent from resolve graph")
    edges: dict[str, set[str]] = {}
    for node in nodes:
        deps = node.get("deps", node.get("dependencies", []))
        if not isinstance(deps, list):
            raise AuditError(f"malformed dependencies: {node['id']}")
        targets: set[str] = set()
        for dep in deps:
            if not isinstance(dep, dict) or not isinstance(dep.get("pkg"), str):
                raise AuditError(f"malformed dependency edge: {node['id']}")
            if dep["pkg"] not in node_by_id:
                raise AuditError(f"dangling dependency edge: {dep['pkg']}")
            dep_kinds = dep.get("dep_kinds", [])
            if not isinstance(dep_kinds, list):
                raise AuditError(f"malformed dependency kinds: {node['id']}")
            targets.add(dep["pkg"])
        edges[node["id"]] = targets
    reachable: set[str] = set()
    pending = [root_id]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(edges[current] - reachable)
    if not reachable:
        raise AuditError("empty resolve graph")
    active: list[dict[str, Any]] = []
    all_packages: list[dict[str, Any]] = []
    for package in packages:
        name = package["name"]
        version = package["version"]
        source = package.get("source")
        if source is None and package["id"] != root_id:
            raise AuditError(f"unbound non-root source-less package: {package['id']}")
        if source is not None and not isinstance(source, str):
            raise AuditError(f"malformed package source: {package['id']}")
        if isinstance(source, str) and not (source.startswith("registry+") or source.startswith("git+")):
            raise AuditError(f"unknown package source: {source}")
        lock_row = lock.get((name, version, source))
        if source is not None and lock_row is None:
            raise AuditError(f"sourced package absent from lock: {(name, version, source)}")
        if source is not None and source.startswith("registry+") and not isinstance((lock_row or {}).get("checksum"), str):
            raise AuditError(f"registry package checksum missing from lock: {(name, version, source)}")
        all_packages.append({
            "id": package["id"], "name": name, "version": version, "source": source,
            "declared_features": package.get("features", []),
            "lock_present": lock_row is not None,
            "lock_checksum": (lock_row or {}).get("checksum"),
        })
    blocked: list[str] = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise AuditError("malformed resolve node")
        package = package_by_id.get(node["id"])
        if package is None:
            raise AuditError(f"resolve node absent from packages: {node['id']}")
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if source is None and node["id"] != root_id:
            raise AuditError(f"unbound non-root source-less package: {node['id']}")
        key = (name, version, source)
        lock_row = lock.get(key)
        if source is not None and lock_row is None:
            raise AuditError(f"active package absent from lock: {key}")
        if source is not None and source.startswith("registry+") and not isinstance((lock_row or {}).get("checksum"), str):
            raise AuditError(f"registry package checksum missing from lock: {key}")
        if node["id"] not in reachable:
            continue
        features = node.get("features", [])
        if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
            raise AuditError(f"malformed feature list: {name}")
        markers = [marker for marker in FORBIDDEN_MARKERS if marker in name.lower() or any(marker in feature.lower() for feature in features)]
        if markers:
            blocked.extend(f"{name}:{marker}" for marker in markers)
        active.append({"id": node["id"], "name": name, "version": version, "source": source, "features": features, "lock_checksum": (lock_row or {}).get("checksum"), "dependencies": sorted(edges[node["id"]])})
    return {
        "status": "BLOCKED_FEATURE_OR_NATIVE_MARKER" if blocked else "VALID_METADATA_OPEN_NATIVE_LICENSE",
        "active_packages": active,
        "all_packages": all_packages,
        "active_package_count": len(active),
        "all_package_count": len(packages),
        "blocked_markers": sorted(set(blocked)),
        "target_filter": metadata.get("target_filter") if isinstance(metadata.get("target_filter"), str) else "OPEN_EXTERNAL_TARGET_FILTER_NOT_BOUND",
        "workspace_root": root_id,
        "reachable_package_count": len(reachable),
        "activation_status": "OPEN_REACHABILITY_ONLY_TARGET_DEP_KINDS_NOT_BOUND",
        "external_binding": "OPEN_METADATA_DERIVED_WORKSPACE_LOCK_TARGET_BINDING_NOT_PROVIDED",
        "sourced_packages_lock_bound": True,
        "native_elf_and_license_approval": "OPEN",
    }


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    _write_noclobber(path, encoded)


def audit(source_root: Path, output: Path, metadata_path: Path | None = None, registry_root: Path | None = None) -> int:
    source_root = source_root.absolute()
    _validate_path_ancestors(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise AuditError("source root must be a regular directory")
    facts = _git_facts(source_root)
    if facts["revision"] != PINNED_REV:
        raise AuditError(f"revision mismatch: {facts['revision']}")
    if facts["origin"] not in OFFICIAL_ORIGINS:
        raise AuditError(f"origin is not official: {facts['origin']}")
    if not facts["clean"]:
        raise AuditError("source checkout is dirty")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise AuditError(f"output exists; no-clobber: {output}")
    _validate_output_parent(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output, 0o700)
    try:
        source_paths = list(SOURCE_DECLARATIONS) + _tracked_source_paths(source_root)
        if len(source_paths) > MAX_FILES or len(set(source_paths)) != len(source_paths):
            raise AuditError("source selection has duplicate/excess paths")
        source_rows = []
        snapshots: dict[str, bytes] = {}
        total = 0
        for rel in sorted(source_paths):
            data = _bounded_bytes(_ensure_parent_chain(source_root, rel))
            total += len(data)
            if total > MAX_TOTAL_BYTES:
                raise AuditError("selected source exceeds aggregate bound")
            snapshots[rel] = data
            row = _file_row(source_root, rel, data)
            blob_id = _git(source_root, ["rev-parse", f"{facts['revision']}:{rel}"]).strip()
            if not re.fullmatch(r"[0-9a-f]{40}", blob_id):
                raise AuditError(f"unexpected git blob id: {rel}")
            blob_data = _git_bytes(source_root, ["cat-file", "blob", f"{facts['revision']}:{rel}"], MAX_READ)
            if blob_data != data:
                raise AuditError(f"git blob bytes changed or do not match: {rel}")
            if blob_id != _git_blob_id(blob_data):
                raise AuditError(f"git blob object id mismatch: {rel}")
            row["git_blob_id"] = blob_id
            source_rows.append(row)
        root_cargo = tomllib.loads(snapshots["rust/Cargo.toml"].decode())
        core_cargo = tomllib.loads(snapshots["rust/moshi-core/Cargo.toml"].decode())
        lock_bytes = snapshots["rust/Cargo.lock"]
        lock_sha = _sha(lock_bytes)
        if lock_sha != EXPECTED_LOCK_SHA256:
            raise AuditError(f"pinned lock hash mismatch: {lock_sha}")
        workspace = _minimal_workspace(source_root, output / "derived-workspace", root_cargo, core_cargo, source_paths, snapshots)
        copied_lock = output / "derived-workspace" / "Cargo.lock"
        _write_noclobber(copied_lock, lock_bytes)
        workspace["files"].append("Cargo.lock")
        workspace["files"].sort()
        workspace["copied_lock"] = {"path": "Cargo.lock", "bytes": len(lock_bytes), "sha256": lock_sha}
        workspace["derived_lock"] = {"status": "NOT_GENERATED_SOURCE_ONLY", "sha256": None}
        metadata: dict[str, Any]
        if metadata_path is None:
            metadata = {"status": "OPEN_METADATA_NOT_PROVIDED", "activated_graph": "OPEN", "registry_source": "OPEN"}
        else:
            expected_version = root_cargo.get("workspace", {}).get("package", {}).get("version")
            if not isinstance(expected_version, str):
                raise AuditError("source workspace package version is not a literal")
            metadata = _metadata_inventory(_read_json(metadata_path), tomllib.loads(lock_bytes.decode()), expected_version)
            metadata["registry_source"] = _registry_inventory(metadata, registry_root)
        final_facts = _git_facts(source_root)
        if final_facts != facts:
            raise AuditError("source checkout changed during collection")
        manifest = {
            "status": COLLECTOR_STATUS,
            "overall": "OPEN",
            "execution": {"source_import": False, "cargo": False, "build": False, "download": False, "model": False},
            "source": {"before": facts, "after": final_facts, "revision": facts["revision"], "origin": facts["origin"], "clean": facts["clean"], "rows": source_rows},
            "lock": {"path": "rust/Cargo.lock", "bytes": len(lock_bytes), "sha256": lock_sha, "unchanged": True},
            "workspace": workspace,
            "core_dependencies": list(DIRECT_CORE_DEPS),
            "metadata": metadata,
            "license_owner_signoff": "OPEN_NOT_OWNER_SIGNABLE",
            "native_payload_and_elf": "OPEN",
            "next_gate": "authorized VAST metadata/source archive audit; no local Cargo or model execution",
        }
        manifest["inventory_payload_scope"] = "manifest_without_inventory_payload_sha256"
        manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        manifest["inventory_payload_sha256"] = _sha(manifest_payload)
        _write_manifest(output / "source-inventory.json", manifest)
        return 0
    except Exception:
        # The output path is intentionally retained as a bounded failure artifact;
        # callers can inspect it without risking deletion of an unrelated path.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--registry-root", type=Path)
    args = parser.parse_args(argv)
    try:
        return audit(args.source_root, args.output, args.metadata_json, args.registry_root)
    except (AuditError, OSError, subprocess.SubprocessError, tomllib.TOMLDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
