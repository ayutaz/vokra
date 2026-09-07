#!/usr/bin/env python3
"""Model-free factual audit for the frozen YuE xcodec-mini Python closure.

This auditor deliberately imports no YuE, xcodec, Vocos, Torch, or reference
code.  It records only the authenticated lock rows and installed
``importlib.metadata`` files.  The resulting report is evidence, not a legal
approval: the checked-in manifest keeps every dependency and component in a
fail-closed pending state until the owner reviews this evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import posixpath
import platform
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


SCHEMA = "vokra-yue-xcodec-mini-dependency-audit-v1"
REGISTRIES = {
    "https://pypi.org/simple": "files.pythonhosted.org",
    "https://download.pytorch.org/whl/cpu": "download-r2.pytorch.org",
}
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
ELF_MAGIC = b"\x7fELF"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
READELF_TIMEOUT_SECONDS = 30


class AuditError(ValueError):
    """A malformed contract or unsafe installed payload."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def has_symlink_ancestry(path: Path) -> bool:
    """Return true when an existing component of *path* is a symlink."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def require_expected_head(value: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise AuditError("expected HEAD must be exactly 40 lowercase hexadecimal characters")
    return value


def verify_repository(repo_root: Path, expected_head: str, audit_script: Path) -> dict[str, Any]:
    expected_head = require_expected_head(expected_head)
    if not repo_root.is_absolute() or repo_root.is_symlink() or has_symlink_ancestry(repo_root):
        raise AuditError("repository root is symlinked or not absolute")
    if not repo_root.is_dir() or not (repo_root / ".git").exists():
        raise AuditError("repository root is not a checkout")
    if audit_script.is_symlink() or not audit_script.is_file() or has_symlink_ancestry(audit_script):
        raise AuditError("audit script is missing or symlinked")
    try:
        top = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"cannot verify repository state: {type(exc).__name__}") from exc
    try:
        same_root = Path(top).resolve(strict=True) == repo_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditError(f"repository root cannot be resolved: {exc}") from exc
    clean = status == ""
    if not same_root or head != expected_head or not clean:
        raise AuditError(f"repository binding failed: expected_head={expected_head} head={head!r} clean={clean}")
    return {"expected_head": expected_head, "head": head, "clean": True, "audit_script_sha256": sha256_file(audit_script)}


def require_absent_output(path: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink() or has_symlink_ancestry(path.parent):
        raise AuditError(f"output must be an absent absolute path without symlink ancestry: {path}")


def atomic_write_no_replace(path: Path, payload: bytes) -> None:
    """Create one regular file atomically without replacing a raced path."""
    require_absent_output(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise AuditError(f"output appeared during no-replace write: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def safe_relative(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise AuditError(f"unsafe installed path: {value!r}")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AuditError(f"unsafe installed path: {value!r}")
    return path


def inside(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def is_license_name(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    name = path.name.casefold()
    return (
        name in LICENSE_NAMES
        or name in {"licenses", "licences"}
        or any(name.startswith(f"{candidate}{sep}") for candidate in LICENSE_NAMES for sep in (".", "-", "_"))
        or any(part.casefold() in {"licenses", "licences"} for part in path.parts)
    )


def resolve_target(path: Path, prefix: Path) -> Path:
    """Strictly resolve a license/native candidate inside the Python prefix."""
    if has_symlink_ancestry(path):
        raise AuditError("symlink ancestry")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditError(f"candidate cannot be strictly resolved: {exc}") from exc
    try:
        resolved.relative_to(prefix)
    except ValueError as exc:
        raise AuditError("candidate escapes sys.prefix") from exc
    if has_symlink_ancestry(resolved):
        raise AuditError("resolved symlink ancestry")
    return resolved


def file_fact(path: Path, relative: PurePosixPath, *, limit: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"installed payload is not a regular file: {relative}")
    size = path.stat().st_size
    fact: dict[str, Any] = {"path": relative.as_posix(), "bytes": size, "sha256": sha256_file(path)}
    if limit is not None and size > limit:
        fact["too_large"] = True
    return fact


def elf_needed(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if handle.read(4) != ELF_MAGIC:
            return {"format": "non-elf", "needed": []}
    try:
        result = subprocess.run(
            ["readelf", "-d", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=READELF_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "error": type(exc).__name__}
    return {"format": "elf", "needed": sorted(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout))}


def native_candidate(path: Path) -> bool:
    lower = path.name.casefold()
    if path.suffix.casefold() in NATIVE_SUFFIXES or ".so." in lower or ".dylib." in lower or lower.endswith(".dll"):
        return True
    try:
        mode = path.stat().st_mode
        if not mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return False
        with path.open("rb") as handle:
            return handle.read(4) == ELF_MAGIC
    except OSError as exc:
        raise AuditError(f"cannot inspect native candidate {path}: {exc}") from exc


def artifact_facts(row: dict[str, Any]) -> list[dict[str, Any]]:
    source = row.get("source")
    if source == {"virtual": "."}:
        return []
    if not isinstance(source, dict) or source.get("registry") not in REGISTRIES:
        raise AuditError(f"unapproved registry for {row.get('name')}")
    expected_host = REGISTRIES[source["registry"]]
    artifacts: list[tuple[str, dict[str, Any]]] = []
    if row.get("sdist") is not None:
        artifacts.append(("sdist", row["sdist"]))
    artifacts.extend(("wheel", item) for item in row.get("wheels", []))
    if not artifacts:
        raise AuditError(f"locked artifact table is empty: {row['name']}")
    result: list[dict[str, Any]] = []
    for kind, artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) - {"url", "hash", "size", "upload-time"} or not {"url", "hash", "size", "upload-time"}.issubset(artifact):
            raise AuditError(f"locked artifact schema drifted: {row['name']}")
        url = artifact.get("url")
        parsed = urlsplit(url) if isinstance(url, str) else None
        try:
            port = parsed.port if parsed is not None else None
        except ValueError as exc:
            raise AuditError(f"locked artifact URL has an invalid port: {row['name']}") from exc
        if parsed is None or parsed.scheme != "https" or parsed.hostname != expected_host or parsed.username or parsed.password or parsed.query or parsed.fragment or port not in (None, 443):
            raise AuditError(f"locked artifact URL is not bound to registry: {row['name']}")
        if not isinstance(artifact.get("hash"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
            raise AuditError(f"locked artifact hash is malformed: {row['name']}")
        size = artifact.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise AuditError(f"locked artifact size is missing: {row['name']}")
        if not isinstance(artifact.get("upload-time"), str) or not artifact["upload-time"].strip():
            raise AuditError(f"locked artifact upload time is missing: {row['name']}")
        result.append({"kind": kind, "url": url, "hash": artifact["hash"], "size": size, "upload-time": artifact["upload-time"]})
    return result


def distribution_fact(row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    name = str(row["name"])
    version = str(row["version"])
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError as exc:
        raise AuditError(f"locked package is not installed: {name}=={version}") from exc
    if dist.version != version:
        raise AuditError(f"installed version drift: {name} {dist.version!r} != {version!r}")
    root = Path(dist.locate_file(""))
    if root.is_symlink() or has_symlink_ancestry(root) or not root.is_dir():
        raise AuditError(f"distribution root is not regular: {name}")
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditError(f"distribution root cannot be resolved: {name}") from exc
    prefix = Path(sys.prefix)
    if prefix.is_symlink() or has_symlink_ancestry(prefix):
        raise AuditError(f"Python prefix is symlinked: {prefix}")
    try:
        prefix = prefix.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AuditError(f"Python prefix cannot be resolved: {exc}") from exc
    licenses: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    ignored: list[str] = []
    path_blockers: list[dict[str, Any]] = []
    native_errors: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=str):
        raw = str(entry)
        normalized = posixpath.normpath(raw.replace("\\", "/"))
        parts = raw.replace("\\", "/").split("/")
        traversal = normalized.startswith("../") or normalized == ".." or any(part in {"", ".", ".."} for part in parts)
        target_license = is_license_name(normalized)
        target_native = any(normalized.casefold().endswith(suffix) or f"{suffix}." in normalized.casefold() for suffix in NATIVE_SUFFIXES)
        # RECORD entries commonly contain harmless ../../../bin and ../../../man
        # entries.  Ignore those non-target paths, but never ignore a path
        # that could carry license or native evidence.
        if traversal and not (target_license or target_native):
            ignored.append(raw)
            continue
        path = Path(dist.locate_file(entry))
        if traversal or target_license or target_native:
            try:
                resolved = resolve_target(path, prefix)
            except (OSError, RuntimeError, ValueError, AuditError) as exc:
                path_blockers.append({"path": raw, "kind": "license" if target_license else "native" if target_native else "target", "reason": str(exc)})
                continue
            path = resolved
        elif not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            path_blockers.append({"path": raw, "kind": "license" if target_license else "native", "reason": "not a regular file or symlink"})
            continue
        relative = PurePosixPath(posixpath.relpath(str(path), str(root.resolve(strict=True))))
        try:
            if is_license_name(normalized):
                licenses.append(file_fact(path, relative, limit=MAX_LICENSE_BYTES))
            if native_candidate(path):
                fact = file_fact(path, relative)
                fact["elf"] = elf_needed(path)
                if "error" in fact["elf"]:
                    native_errors.append({"path": relative.as_posix(), "error": fact["elf"]["error"]})
                native.append(fact)
        except (OSError, AuditError) as exc:
            path_blockers.append({"path": raw, "kind": "license" if is_license_name(normalized) else "native", "reason": str(exc)})
    artifacts = artifact_facts(row)
    result: dict[str, Any] = {
        "name": name,
        "version": version,
        "source": row["source"],
        "artifacts": artifacts,
        "artifact_identity_sha256": digest(artifacts),
        "publisher_license": dist.metadata.get("License") or "",
        "license_classifiers": sorted(dist.metadata.get_all("Classifier") or []),
        "license_files": licenses,
        "license_files_sha256": digest(licenses),
        "native_files": native,
        "native_files_sha256": digest(native),
        "ignored_paths": sorted(ignored),
        "path_blockers": sorted(path_blockers, key=lambda item: (item["path"], item["kind"], item["reason"])),
        "native_errors": sorted(native_errors, key=lambda item: (item["path"], item["error"])),
        "native_inventory_status": "BLOCKED" if path_blockers or native_errors else "OK",
        "review": {
            "status": review["status"],
            "license": review["license"],
            "native_review": review["native_review"],
            "bundled_review": review["bundled_review"],
            "manifest_payload_sha256": review["payload_sha256"],
        },
    }
    result["fact_sha256"] = digest(result)
    return result


def load_contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = [project / name for name in ("pyproject.toml", "uv.lock", "license_gate_manifest.json", "preflight_gate.py")]
    if not project.is_dir() or project.is_symlink() or any(path.is_symlink() or not path.is_file() for path in paths):
        raise AuditError("YuE contract files are missing or symlinked")
    try:
        project_bytes = (project / "pyproject.toml").read_bytes()
        lock_bytes = (project / "uv.lock").read_bytes()
        project_doc = tomllib.loads(project_bytes.decode("utf-8"))
        lock_doc = tomllib.loads(lock_bytes.decode("utf-8"))
        manifest = strict_json(project / "license_gate_manifest.json")
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"YuE contract is unreadable: {exc}") from exc
    if not isinstance(project_doc.get("project"), dict) or project_doc["project"].get("name") != "vokra-yue-xcodec-mini-parity":
        raise AuditError("project identity drifted")
    if not isinstance(manifest, dict):
        raise AuditError("license gate manifest is not an object")
    expected_keys = {"approval_scope_sha256", "component_reviews", "dependency_reviews", "dependency_reviews_sha256", "fixed_identities", "gate_version", "lock_sha256", "no_upload", "operator_approval", "package_rows_sha256", "pyproject_sha256"}
    if set(manifest) != expected_keys:
        raise AuditError("license gate manifest schema drifted")
    if sha256_bytes(project_bytes) != manifest["pyproject_sha256"] or sha256_bytes(lock_bytes) != manifest["lock_sha256"]:
        raise AuditError("pyproject.toml/uv.lock bytes differ from manifest")
    rows = lock_doc.get("package")
    if not isinstance(rows, list) or not rows:
        raise AuditError("uv.lock package rows are missing")
    seen: set[tuple[str, str, str]] = set()
    virtual = 0
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not isinstance(row.get("source"), dict):
            raise AuditError("uv.lock package row identity is malformed")
        key = (row["name"], row["version"], canonical(row["source"]))
        if key in seen:
            raise AuditError(f"duplicate locked package identity: {row['name']}")
        seen.add(key)
        if row["source"] == {"virtual": "."}:
            virtual += 1
        elif row["source"].get("registry") not in REGISTRIES:
            raise AuditError(f"unapproved lock registry: {row['name']}")
        normalized.append(row)
    normalized.sort(key=lambda row: (normalized_name(row["name"]), row["version"], canonical(row["source"])))
    if virtual != 1:
        raise AuditError("uv.lock must contain exactly one virtual project row")
    # Match the canonical row projection used by the existing stdlib gate.
    projected = []
    for row in normalized:
        projected.append({"name": row["name"], "version": row["version"], "source": row["source"], "resolution-markers": row.get("resolution-markers", []), "dependencies": row.get("dependencies", []), "sdist": row.get("sdist"), "wheels": row.get("wheels", [])})
    if digest(projected) != manifest["package_rows_sha256"]:
        raise AuditError("canonical lock rows differ from manifest")
    reviews = manifest.get("dependency_reviews")
    if not isinstance(reviews, list) or manifest.get("dependency_reviews_sha256") != digest(reviews):
        raise AuditError("dependency review table is not bound")
    if len(reviews) != len(projected):
        raise AuditError("dependency review count does not match lock rows")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {"bundled_review", "id", "license", "name", "native_review", "payload_sha256", "source", "status", "version"}:
            raise AuditError("dependency review row schema drifted")
        key = (review["name"], review["version"], canonical(review["source"]))
        if key in by_key:
            raise AuditError("duplicate dependency review row")
        by_key[key] = review
    package_rows = [row for row in projected if row["source"] != {"virtual": "."}]
    for row in package_rows:
        key = (row["name"], row["version"], canonical(row["source"]))
        if key not in by_key or by_key[key]["id"] != f"{row['name']}@{row['version']}":
            raise AuditError(f"missing dependency review: {row['name']}")
    virtual = next(row for row in projected if row["source"] == {"virtual": "."})
    virtual_key = (virtual["name"], virtual["version"], canonical(virtual["source"]))
    if virtual_key not in by_key or by_key[virtual_key]["id"] != f"{virtual['name']}@{virtual['version']}":
        raise AuditError("missing dependency review for virtual project row")
    if manifest.get("no_upload") != "NO_UPLOAD":
        raise AuditError("manifest publication boundary drifted")
    components = manifest.get("component_reviews")
    if not isinstance(components, list) or not components or any(
        not isinstance(item, dict)
        or set(item) != {"approval_digest", "id", "identity", "license", "payload_sha256", "role", "signer", "status"}
        or not isinstance(item.get("id"), str)
        for item in components
    ):
        raise AuditError("component review table schema drifted")
    return project_doc, lock_doc, manifest, package_rows, by_key, projected


def validate_report(report: dict[str, Any]) -> None:
    required = {"schema", "status", "publication", "repository", "project", "locked_rows", "packages", "components", "environment", "failures"}
    if set(report) != required or report["schema"] != SCHEMA or report["publication"] != "NO_UPLOAD":
        raise AuditError("audit report schema drifted")
    if report["status"] not in {"BLOCKED_OWNER_REVIEW", "BLOCKED"}:
        raise AuditError("audit report status must remain blocked")
    repository = report["repository"]
    if not isinstance(repository, dict) or set(repository) != {"expected_head", "head", "clean", "audit_script_sha256"} or HEX40.fullmatch(str(repository.get("expected_head"))) is None or repository.get("head") != repository.get("expected_head") or repository.get("clean") is not True or HEX64.fullmatch(str(repository.get("audit_script_sha256"))) is None:
        raise AuditError("repository binding is invalid")
    project = report["project"]
    if not isinstance(project, dict) or set(project) != {"name", "version", "pyproject_sha256", "uv_lock_sha256", "manifest_sha256"} or project.get("name") != "vokra-yue-xcodec-mini-parity" or not all(HEX64.fullmatch(str(project.get(key))) for key in ("pyproject_sha256", "uv_lock_sha256", "manifest_sha256")):
        raise AuditError("project binding is invalid")
    locked = report["locked_rows"]
    if not isinstance(locked, dict) or set(locked) != {"rows", "package_rows_sha256", "count", "registry_count", "lock_sha256"} or not isinstance(locked.get("rows"), list) or locked.get("count") != len(locked["rows"]) or locked.get("registry_count") != len(locked["rows"]) - 1 or not HEX64.fullmatch(str(locked.get("package_rows_sha256"))) or not HEX64.fullmatch(str(locked.get("lock_sha256"))):
        raise AuditError("locked row coverage is invalid")
    if digest(locked["rows"]) != locked["package_rows_sha256"]:
        raise AuditError("locked row digest is invalid")
    lock_row_fields = {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels"}
    if any(not isinstance(row, dict) or set(row) != lock_row_fields or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not isinstance(row.get("source"), dict) for row in locked["rows"]):
        raise AuditError("locked row schema is invalid")
    package_fields = {"name", "version", "source", "artifacts", "artifact_identity_sha256", "publisher_license", "license_classifiers", "license_files", "license_files_sha256", "native_files", "native_files_sha256", "ignored_paths", "path_blockers", "native_errors", "native_inventory_status", "review", "fact_sha256"}
    packages = report["packages"]
    if not isinstance(packages, list) or len(packages) != locked["registry_count"]:
        raise AuditError("package count does not cover all registry rows")
    lock_identities = {
        (row.get("name"), row.get("version"), canonical(row.get("source")))
        for row in locked["rows"]
        if row.get("source") != {"virtual": "."}
    }
    identities: set[tuple[str, str, str]] = set()
    for item in packages:
        key = (item.get("name"), item.get("version"), canonical(item.get("source"))) if isinstance(item, dict) else None
        if not isinstance(item, dict) or set(item) != package_fields or key in identities:
            raise AuditError("package fact schema or identity is invalid")
        identities.add(key)
        if digest({k: v for k, v in item.items() if k != "fact_sha256"}) != item.get("fact_sha256"):
            raise AuditError("package fact digest is invalid")
        if not isinstance(item["artifacts"], list) or digest(item["artifacts"]) != item["artifact_identity_sha256"]:
            raise AuditError("artifact fact digest is invalid")
        for artifact in item["artifacts"]:
            if not isinstance(artifact, dict) or set(artifact) != {"kind", "url", "hash", "size", "upload-time"} or artifact["kind"] not in {"sdist", "wheel"} or not isinstance(artifact["url"], str) or not artifact["url"].startswith("https://") or not isinstance(artifact["hash"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]) is None or isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0 or not isinstance(artifact["upload-time"], str):
                raise AuditError("artifact fact schema is invalid")
        if not isinstance(item["publisher_license"], str) or not isinstance(item["license_classifiers"], list) or any(not isinstance(value, str) for value in item["license_classifiers"]):
            raise AuditError("publisher license fact schema is invalid")
        license_fields = {"path", "bytes", "sha256"}
        if not isinstance(item["license_files"], list) or digest(item["license_files"]) != item["license_files_sha256"] or any(not isinstance(value, dict) or not set(value).issubset(license_fields | {"too_large"}) or set(value) < license_fields or not isinstance(value["path"], str) or isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] < 0 or HEX64.fullmatch(str(value["sha256"])) is None or ("too_large" in value and not isinstance(value["too_large"], bool)) for value in item["license_files"]):
            raise AuditError("license file fact schema or digest is invalid")
        native_fields = {"path", "bytes", "sha256", "elf"}
        if not isinstance(item["native_files"], list) or digest(item["native_files"]) != item["native_files_sha256"] or any(not isinstance(value, dict) or set(value) != native_fields or not isinstance(value["path"], str) or isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] < 0 or HEX64.fullmatch(str(value["sha256"])) is None or not isinstance(value["elf"], dict) or set(value["elf"]) - {"format", "needed", "error"} or "format" not in value["elf"] or "needed" not in value["elf"] or not isinstance(value["elf"]["format"], str) or not isinstance(value["elf"]["needed"], list) or any(not isinstance(needed, str) for needed in value["elf"]["needed"]) or ("error" in value["elf"] and not isinstance(value["elf"]["error"], str)) for value in item["native_files"]):
            raise AuditError("native file fact schema or digest is invalid")
        if not isinstance(item["ignored_paths"], list) or any(not isinstance(value, str) for value in item["ignored_paths"]):
            raise AuditError("ignored RECORD path schema is invalid")
        blocker_fields = {"path", "kind", "reason"}
        if not isinstance(item["path_blockers"], list) or any(not isinstance(value, dict) or set(value) != blocker_fields or not all(isinstance(value[key], str) and value[key] for key in blocker_fields) for value in item["path_blockers"]):
            raise AuditError("target path blocker schema is invalid")
        error_fields = {"path", "error"}
        if not isinstance(item["native_errors"], list) or any(not isinstance(value, dict) or set(value) != error_fields or not isinstance(value["path"], str) or not isinstance(value["error"], str) or not value["error"] for value in item["native_errors"]):
            raise AuditError("native error schema is invalid")
        review_fields = {"status", "license", "native_review", "bundled_review", "manifest_payload_sha256"}
        review = item["review"]
        if not isinstance(review, dict) or set(review) != review_fields or not isinstance(review["status"], str) or not isinstance(review["license"], (str, type(None))) or not isinstance(review["native_review"], (str, type(None))) or not isinstance(review["bundled_review"], (str, type(None))) or (review["manifest_payload_sha256"] is not None and HEX64.fullmatch(str(review["manifest_payload_sha256"])) is None):
            raise AuditError("dependency review fact schema is invalid")
        if item.get("native_inventory_status") not in {"OK", "BLOCKED"} or item["native_inventory_status"] != ("BLOCKED" if item["path_blockers"] or item["native_errors"] else "OK"):
            raise AuditError("native inventory coverage is invalid")
    if identities != lock_identities:
        raise AuditError("package facts do not cover exactly the locked registry rows")
    components = report["components"]
    if not isinstance(components, dict) or set(components) != {"reviews", "pending_ids", "count"} or not isinstance(components.get("reviews"), list) or components.get("count") != len(components["reviews"]) or not isinstance(components.get("pending_ids"), list):
        raise AuditError("component coverage is invalid")
    component_fields = {"approval_digest", "id", "identity", "license", "payload_sha256", "role", "signer", "status"}
    if any(not isinstance(item, dict) or set(item) != component_fields for item in components["reviews"]):
        raise AuditError("component review schema is invalid")
    expected_pending = sorted(item["id"] for item in components["reviews"] if item["status"] != "REVIEWED")
    if components["pending_ids"] != expected_pending:
        raise AuditError("component pending coverage is invalid")
    environment = report["environment"]
    if not isinstance(environment, dict) or not {"platform", "machine", "python", "weights_acquired", "model_imported", "model_executed", "upload"}.issubset(environment) or environment.get("weights_acquired") is not False or environment.get("model_imported") is not False or environment.get("model_executed") is not False or environment.get("upload") != "NO_UPLOAD" or not isinstance(report["failures"], list) or not report["failures"]:
        raise AuditError("model-free boundary drifted")


def audit(project: Path, output: Path, repo_root: Path, expected_head: str) -> int:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise AuditError("YuE dependency audit requires Linux x86_64 VAST")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise AuditError("VOKRA_PUBLISH_ON_VAST=1 is required")
    repository = verify_repository(repo_root, expected_head, project / "dependency_audit.py")
    require_absent_output(output)
    sidecar = output.with_name(output.name + ".sha256")
    require_absent_output(sidecar)
    _, lock, manifest, package_rows, reviews, all_rows = load_contract(project)
    packages = [distribution_fact(row, reviews[(row["name"], row["version"], canonical(row["source"]))]) for row in package_rows]
    components = manifest["component_reviews"]
    pending_packages = [item["name"] for item in packages if item["review"]["status"] != "REVIEWED"]
    pending_components = sorted(item["id"] for item in components if item.get("status") != "REVIEWED")
    path_blockers = sum(len(item["path_blockers"]) for item in packages)
    native_errors = sum(len(item["native_errors"]) for item in packages)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED_OWNER_REVIEW",
        "publication": "NO_UPLOAD",
        "repository": repository,
        "project": {"name": "vokra-yue-xcodec-mini-parity", "version": "0.1.0", "pyproject_sha256": manifest["pyproject_sha256"], "uv_lock_sha256": manifest["lock_sha256"], "manifest_sha256": sha256_file(project / "license_gate_manifest.json")},
        "locked_rows": {"rows": all_rows, "package_rows_sha256": manifest["package_rows_sha256"], "count": len(all_rows), "registry_count": len(package_rows), "lock_sha256": manifest["lock_sha256"]},
        "packages": packages,
        "components": {"reviews": components, "pending_ids": pending_components, "count": len(components)},
        "environment": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version, "weights_acquired": False, "model_imported": False, "model_executed": False, "upload": "NO_UPLOAD"},
        "failures": [f"OWNER_REVIEW_REQUIRED: {len(pending_packages)} dependency rows remain pending", f"COMPONENT_REVIEW_REQUIRED: {len(pending_components)} component rows remain pending", *([f"INSTALLED_PATH_BLOCKED: {path_blockers} targeted paths were blocked"] if path_blockers else []), *([f"READELF_BLOCKED: {native_errors} native inventories failed"] if native_errors else []), "MODEL_FREE_ONLY: no weight acquisition/import/execution was performed"],
    }
    validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_bytes = (canonical(report) + "\n").encode("utf-8")
    atomic_write_no_replace(output, report_bytes)
    atomic_write_no_replace(sidecar, f"{sha256_bytes(report_bytes)}  {output.name}\n".encode("ascii"))
    print(f"YuE xcodec-mini dependency audit: BLOCKED_OWNER_REVIEW ({output})", file=sys.stderr)
    return 2


def self_test() -> int:
    project = Path(__file__).resolve().parent
    # Production contract must remain blocked and must never be promoted by a
    # self-test that uses synthetic approvals.
    try:
        load_contract(project)
    except AuditError as exc:
        if "dependency review" not in str(exc) and "canonical lock" not in str(exc):
            print(f"unexpected production contract result: {exc}", file=sys.stderr)
            return 1
    for candidate in ("../escape", "/absolute", "a//b", "a/../b", ""):
        try:
            safe_relative(candidate)
        except AuditError:
            continue
        print(f"unsafe path accepted: {candidate!r}", file=sys.stderr)
        return 1
    try:
        require_expected_head("A" * 40)
    except AuditError:
        pass
    else:
        print("uppercase expected-head self-test failed", file=sys.stderr)
        return 1
    try:
        require_expected_head("0" * 39)
    except AuditError:
        pass
    else:
        print("short expected-head self-test failed", file=sys.stderr)
        return 1
    markers: set[str] = set()
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() and not Path("/private/tmp").is_symlink() else None
    with tempfile.TemporaryDirectory(prefix="yue-audit-self-test-", dir=temp_parent) as directory:
        missing = Path(directory) / "missing.json"
        try:
            strict_json(missing)
        except AuditError:
            pass
        else:
            print("missing JSON self-test failed", file=sys.stderr)
            return 1
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
        try:
            strict_json(duplicate)
        except AuditError:
            pass
        else:
            print("duplicate JSON self-test failed", file=sys.stderr)
            return 1
        root = Path(directory) / "root"
        root.mkdir()
        payload = root / "payload.so"
        payload.write_bytes(ELF_MAGIC + b"test")
        if not native_candidate(payload) or sha256_file(payload) != sha256_bytes(payload.read_bytes()):
            print("native payload self-test failed", file=sys.stderr)
            return 1
        markers.add("native")
        prefix = root / "prefix"
        (prefix / "share" / "licenses").mkdir(parents=True)
        license_path = prefix / "share" / "licenses" / "LICENSE"
        license_path.write_bytes(b"license\n")
        if not is_license_name("../../../share/licenses/LICENSE") or is_license_name("../../../bin/pygrun"):
            print("RECORD target classification self-test failed", file=sys.stderr)
            return 1
        if resolve_target(license_path, prefix) != license_path:
            print("strict RECORD resolve self-test failed", file=sys.stderr)
            return 1
        try:
            resolve_target(root / "outside" / "LICENSE", prefix)
        except AuditError:
            pass
        else:
            print("prefix-external RECORD target self-test failed", file=sys.stderr)
            return 1
        link = prefix / "link"
        try:
            link.symlink_to(root / "outside", target_is_directory=True)
        except OSError as exc:
            print(f"symlink ancestry self-test setup failed: {exc}", file=sys.stderr)
            return 1
        try:
            resolve_target(link / "LICENSE", prefix)
        except AuditError:
            pass
        else:
            print("symlink ancestry self-test failed", file=sys.stderr)
            return 1
        markers.add("record")
        atomic = Path(directory) / "atomic.json"
        atomic_write_no_replace(atomic, b"first\n")
        if atomic.read_bytes() != b"first\n":
            print("atomic write self-test failed", file=sys.stderr)
            return 1
        try:
            atomic_write_no_replace(atomic, b"second\n")
        except AuditError:
            pass
        else:
            print("no-replace race self-test failed", file=sys.stderr)
            return 1
        if atomic.read_bytes() != b"first\n":
            print("no-replace clobber self-test failed", file=sys.stderr)
            return 1
        markers.add("atomic")
    if markers != {"native", "record", "atomic"}:
        print(f"self-test marker coverage failed: {sorted(markers)}", file=sys.stderr)
        return 1
    print("yue_xcodec_mini dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None or args.repo_root is not None or args.expected_head is not None:
            parser.error("--self-test accepts no paths")
        return self_test()
    if args.project is None or args.output is None or args.repo_root is None or args.expected_head is None:
        parser.error("--project, --output, --repo-root and --expected-head are required")
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise SystemExit("--output must be an absent absolute path")
    try:
        return audit(args.project, args.output, args.repo_root, args.expected_head)
    except AuditError as exc:
        print(f"YuE xcodec-mini dependency audit: BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
