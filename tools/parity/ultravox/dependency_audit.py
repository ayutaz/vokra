#!/usr/bin/env python3
"""Model-free factual audit for the reviewed Ultravox Python closure.

The audit reads only the checked-in contract and ``importlib.metadata``.  It
never imports Ultravox, Transformers, Torch, or model code.  Network access is
limited to exact locked PyPI sdists needed for missing publisher evidence and
the exact fixed LICENSE paths named by the manifest.
"""

from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
from email.message import Message
import hashlib
import importlib.metadata as metadata
import io
import json
import platform
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # pragma: no cover - direct package invocation
    from tools.parity.ultravox import license_gate


SCHEMA = "vokra-ultravox-dependency-audit-v1"
PYPI_HOST = "files.pythonhosted.org"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_SDIST_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000
MAX_SDIST_REDIRECTS = 4
MAX_LICENSE_REDIRECTS = 4
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
LICENSE_NAMES = {"license", "copying", "notice", "copyright"}
HF_CDN_LICENSE_HOSTS = {"cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co"}
HF_API_HOSTS = {"huggingface.co", "hf.co"}
MAX_MODEL_INFO_BYTES = 2 * 1024 * 1024
MAX_MODEL_INFO_REDIRECTS = 4
MAX_MODEL_INFO_MEMBERS = 10000
MODEL_INFO_SCHEMA = "vokra-ultravox-hf-model-metadata-v1"
MODEL_INFO_EXPECTED = {
    "fixie-ai/ultravox-v0_5-llama-3_2-1b": {
        "revision": "b95bec8ab291eeb04b5cd600dd473377f6b79026",
        "license": "mit",
        "gated": False,
        "license_files": [],
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "revision": "9213176726f574b556790deb65791e0c5aa438b6",
        "license": "llama3.2",
        "gated": "manual",
        "license_files": ["LICENSE.txt"],
    },
}


class AuditError(ValueError):
    """A factual or structural blocker."""


class LicensePathError(AuditError):
    """The exact LICENSE path failed with an HTTP status."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity(name: str, version: str) -> str:
    return f"{re.sub(r'[-_.]+', '-', name.strip()).casefold()}=={re.sub(r'\s+', '', version.strip()).casefold()}"


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


def regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _validate_lock_shape(lock: dict[str, Any], project: dict[str, Any]) -> None:
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "package"}:
        raise AuditError("uv.lock top-level schema drifted")
    if lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*" or not isinstance(lock.get("resolution-markers"), list) or not isinstance(lock.get("package"), list):
        raise AuditError("uv.lock resolver schema drifted")
    if set(project) != {"project", "tool"} or not isinstance(project.get("project"), dict) or not isinstance(project.get("tool"), dict):
        raise AuditError("pyproject root schema drifted")
    if set(project["project"]) != {"name", "version", "description", "requires-python", "dependencies"} or not isinstance(project["project"].get("dependencies"), list):
        raise AuditError("pyproject project schema drifted")
    uv = project["tool"].get("uv")
    if set(project["tool"]) != {"uv"} or not isinstance(uv, dict) or set(uv) != {"package", "index", "sources"}:
        raise AuditError("pyproject uv schema drifted")
    seen: set[tuple[str, str, str]] = set(); virtual = 0
    for row in lock["package"]:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not row["name"].strip() or not row["version"].strip():
            raise AuditError("uv.lock package identity is malformed")
        source = row.get("source")
        if not isinstance(source, dict) or set(source) not in ({"virtual"}, {"registry"}):
            raise AuditError("uv.lock package source schema drifted")
        source_key = json.dumps(source, sort_keys=True)
        key = (row["name"], row["version"], source_key)
        if key in seen:
            raise AuditError(f"uv.lock duplicate package identity: {key!r}")
        seen.add(key)
        if "virtual" in source:
            virtual += 1
            if source != {"virtual": "."} or row["name"] != project["project"]["name"] or row["version"] != project["project"]["version"]:
                raise AuditError("uv.lock virtual root is not bound to pyproject")
        else:
            if source["registry"] not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
                raise AuditError("uv.lock registry is not approved")
            if source["registry"] == "https://download.pytorch.org/whl/cpu" and row["name"] != "torch":
                raise AuditError("PyTorch CPU registry is bound to an unexpected package")
            if "wheels" in row and (not isinstance(row["wheels"], list) or not row["wheels"]):
                raise AuditError("uv.lock wheels are malformed")
            if "sdist" not in row and "wheels" not in row:
                raise AuditError("uv.lock registry row has no artifacts")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list) or any(not isinstance(dep, dict) or not isinstance(dep.get("name"), str) or not dep["name"].strip() for dep in dependencies):
            raise AuditError("uv.lock dependencies are malformed")
    if virtual != 1:
        raise AuditError("uv.lock must contain exactly one virtual root")


def _contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, bytes]:
    project_path, lock_path, manifest_path = (project / name for name in ("pyproject.toml", "uv.lock", "license_gate_manifest.json"))
    if not all(regular_file(path) for path in (project_path, lock_path, manifest_path)):
        raise AuditError("Ultravox pyproject.toml, uv.lock, or manifest is missing/symlinked")
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project_data = tomllib.loads(project_bytes.decode("utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"Ultravox contract is unreadable: {exc}") from exc
    manifest = strict_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != license_gate.GATE_VERSION:
        raise AuditError("Ultravox manifest version is unsupported")
    expected_keys = {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "required_package_rows", "package_review_rows", "package_review_rows_sha256", "forbidden_dependencies", "identities", "license_rows", "license_rows_sha256", "approval", "publication"}
    if set(manifest) != expected_keys:
        raise AuditError("Ultravox manifest schema drifted")
    if sha256_bytes(project_bytes) != manifest["project_sha256"] or sha256_bytes(lock_bytes) != manifest["lock_sha256"]:
        raise AuditError("pyproject.toml/uv.lock bytes differ from the reviewed contract")
    try:
        _validate_lock_shape(lock_data, project_data)
    except (KeyError, TypeError, ValueError, AuditError) as exc:
        raise AuditError(f"uv.lock/pyproject schema is invalid: {exc}") from exc
    rows = license_gate.package_rows(lock_data)
    if license_gate.canonical_digest(rows) != manifest["package_rows_sha256"]:
        raise AuditError("canonical lock rows differ from the manifest")
    _validate_manifest_reviews(manifest, rows)
    _validate_artifacts(lock_data)
    return project_data, lock_data, manifest, project_bytes, lock_bytes


def _validate_artifacts(lock: dict[str, Any]) -> None:
    for row in lock.get("package", []):
        source = row.get("source", {})
        if "registry" not in source:
            continue
        expected_host = "download-r2.pytorch.org" if source["registry"] == "https://download.pytorch.org/whl/cpu" else PYPI_HOST
        for artifact_kind, artifact in ([["sdist", row["sdist"]]] if "sdist" in row else []) + [["wheel", wheel] for wheel in row.get("wheels", [])]:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("url"), str) or not isinstance(artifact.get("hash"), str) or not isinstance(artifact.get("upload-time"), str) or not artifact["upload-time"].strip():
                raise AuditError(f"artifact schema drifted: {row.get('name')}")
            required_keys = {"url", "hash", "upload-time"} | ({"size"} if artifact_kind == "sdist" else set())
            if set(artifact) not in (required_keys, required_keys | {"size"}):
                raise AuditError(f"artifact schema drifted: {row.get('name')}")
            parsed = urlsplit(artifact["url"])
            if parsed.scheme != "https" or parsed.hostname != expected_host or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in (None, 443):
                raise AuditError(f"artifact URL is not bound to its reviewed registry: {row.get('name')}")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                raise AuditError(f"artifact hash is malformed: {row.get('name')}")
            if "size" in artifact and (isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0):
                raise AuditError(f"artifact size is malformed: {row.get('name')}")


def _package_review_key(row: dict[str, Any]) -> tuple[str, str, str]:
    name, version, source = row.get("name"), row.get("version"), row.get("source")
    if not isinstance(name, str) or not isinstance(version, str) or not isinstance(source, dict):
        raise AuditError("package review row identity is malformed")
    return name, version, canonical_json(source)


def _validate_manifest_reviews(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        raise AuditError("package review rows do not cover every lock row")
    expected = {_package_review_key(row) for row in rows}
    actual: set[tuple[str, str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict):
            raise AuditError("package review row is malformed")
        required = {"name", "version", "source", "license", "status", "native_bundled_review"}
        if set(review) != required:
            raise AuditError("package review row schema drifted")
        key = _package_review_key(review)
        if key in actual:
            raise AuditError(f"duplicate package review row: {key!r}")
        actual.add(key)
        if key not in expected:
            raise AuditError(f"package review row is not an exact lock row: {key!r}")
    if actual != expected:
        raise AuditError("package review rows do not cover the exact lock closure")
    if license_gate.canonical_digest(reviews) != manifest.get("package_review_rows_sha256"):
        raise AuditError("package review row bytes differ from the manifest digest")

    license_rows = manifest.get("license_rows")
    if not isinstance(license_rows, list) or not license_rows:
        raise AuditError("license review rows are missing")
    seen_ids: set[str] = set()
    for row in license_rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise AuditError("license review row identity is malformed")
        if row["id"] in seen_ids:
            raise AuditError(f"duplicate license review row: {row['id']}")
        seen_ids.add(row["id"])
    required_ids = {"ultravox-audio-weight", "llama-companion-meta-conditional", "python-closure"}
    if seen_ids != required_ids:
        raise AuditError(f"license review rows do not cover the fixed contract: {sorted(seen_ids ^ required_ids)}")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != {"status", "signer", "digest"}:
        raise AuditError("approval record schema drifted")
    if manifest.get("publication") != "NO_UPLOAD":
        raise AuditError("publication policy drifted from NO_UPLOAD")


def _approval_blockers(manifest: dict[str, Any]) -> list[str]:
    """Report unresolved owner decisions verbatim; do not classify licenses."""
    blockers: list[str] = []
    unresolved = {"", "UNRESOLVED", "OWNER_REVIEW_REQUIRED", "PENDING_REVIEW", "REVIEW_REQUIRED"}
    def unresolved_value(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value in unresolved)
    for row in manifest["package_review_rows"]:
        identity_text = f"{row['name']}=={row['version']}"
        if row.get("status") != "REVIEWED":
            blockers.append(f"package_review:{identity_text}:status={row.get('status')!r}")
        if unresolved_value(row.get("license")):
            blockers.append(f"package_review:{identity_text}:license={row.get('license')!r}")
        if unresolved_value(row.get("native_bundled_review")):
            blockers.append(f"package_review:{identity_text}:native_bundled_review={row.get('native_bundled_review')!r}")
    for row in manifest["license_rows"]:
        if row.get("status") != "REVIEWED":
            blockers.append(f"license_review:{row['id']}:status={row.get('status')!r}")
        if unresolved_value(row.get("license")):
            blockers.append(f"license_review:{row['id']}:license={row.get('license')!r}")
        if unresolved_value(row.get("conclusion")):
            blockers.append(f"license_review:{row['id']}:conclusion={row.get('conclusion')!r}")
    computed_license_digest = license_gate.canonical_digest(manifest["license_rows"])
    if computed_license_digest != manifest.get("license_rows_sha256"):
        blockers.append("manifest:license_rows_sha256 does not match the recorded license_rows bytes")
    approval = manifest["approval"]
    if approval.get("status") != "OWNER_SIGNOFF_APPROVED":
        blockers.append(f"approval:status={approval.get('status')!r}")
    if not approval.get("signer"):
        blockers.append("approval:signer=null")
    if not approval.get("digest"):
        blockers.append("approval:digest=null")
    return sorted(set(blockers))


def _marker_value(name: str) -> str:
    values = {"sys_platform": "linux", "platform_machine": "x86_64", "platform_python_implementation": "CPython"}
    if name not in values:
        raise AuditError(f"unsupported lock marker variable: {name}")
    return values[name]


def _eval_marker(node: ast.AST) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_eval_marker(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1 and isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Constant) and isinstance(node.comparators[0].value, str):
        actual, expected = _marker_value(node.left.id), node.comparators[0].value
        if isinstance(node.ops[0], ast.Eq):
            return actual == expected
        if isinstance(node.ops[0], ast.NotEq):
            return actual != expected
    raise AuditError("unsupported lock marker expression")


def marker_active(markers: list[str]) -> bool:
    if not markers:
        return True
    try:
        return any(_eval_marker(ast.parse(marker, mode="eval").body) for marker in markers)
    except (SyntaxError, AuditError) as exc:
        raise AuditError(f"lock marker cannot be evaluated: {exc}") from exc


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return _normalized_name(row["name"]), row["version"], canonical_json(row["source"])


def _dependency_candidates(dep: dict[str, Any], by_name: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    name = dep.get("name")
    if not isinstance(name, str) or not name.strip():
        raise AuditError("dependency edge has no package name")
    candidates = list(by_name.get(_normalized_name(name), []))
    if isinstance(dep.get("version"), str):
        candidates = [row for row in candidates if row["version"] == dep["version"]]
    elif "version" in dep:
        raise AuditError(f"dependency selector version is malformed: {name}")
    if "source" in dep:
        source = dep["source"]
        if not isinstance(source, dict) or set(source) != {"registry"}:
            raise AuditError(f"dependency selector source is malformed: {name}")
        candidates = [row for row in candidates if row["source"] == source]
    if not candidates:
        raise AuditError(f"dependency selector does not resolve: {name}")
    if len(candidates) > 1:
        raise AuditError(f"dependency selector is ambiguous: {name}")
    return candidates


def classify_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(lock["package"])
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(_normalized_name(row["name"]), []).append(row)
    virtual_rows = [row for row in rows if row["source"] == {"virtual": "."}]
    if len(virtual_rows) != 1:
        raise AuditError("uv.lock dependency graph requires exactly one virtual root")
    reachable: set[tuple[str, str, str]] = set()
    inactive_reasons: dict[tuple[str, str, str], list[str]] = {}

    def walk(row: dict[str, Any], chain: str) -> None:
        key = _row_key(row)
        if key in reachable:
            return
        if row["source"] != {"virtual": "."} and not marker_active(row.get("resolution-markers", [])):
            inactive_reasons.setdefault(key, []).append(f"package resolution marker is inactive on Linux x86_64 CPython 3.12: {row.get('resolution-markers', [])}")
            return
        reachable.add(key)
        for dep in row.get("dependencies", []):
            if not isinstance(dep, dict):
                raise AuditError(f"dependency edge is malformed from {row['name']}=={row['version']}")
            marker = dep.get("marker")
            if marker is not None and not isinstance(marker, str):
                raise AuditError(f"dependency marker is malformed from {row['name']}=={row['version']}")
            edge_active = marker_active([marker]) if marker is not None else True
            candidate = _dependency_candidates(dep, by_name)[0]
            candidate_key = _row_key(candidate)
            if not edge_active:
                inactive_reasons.setdefault(candidate_key, []).append(
                    f"unreachable from {row['name']}=={row['version']}: dependency marker false on Linux x86_64 CPython 3.12: {marker}"
                )
                continue
            if candidate["source"] != {"virtual": "."} and not marker_active(candidate.get("resolution-markers", [])):
                raise AuditError(f"active dependency selects an inactive package resolution row: {dep.get('name')}")
            walk(candidate, f"{chain}->{candidate['name']}")

    walk(virtual_rows[0], virtual_rows[0]["name"])
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["name"], item["version"], json.dumps(item["source"], sort_keys=True))):
        item = dict(row)
        key = _row_key(row)
        if row["source"] == {"virtual": "."}:
            item.update(status="INACTIVE_VIRTUAL_PROJECT", reason="virtual root is not an installed distribution")
            inactive.append(item)
        elif key in reachable:
            item.update(status="ACTIVE_LINUX_INSTALLED", reason="reachable from the virtual root through Linux x86_64 CPython 3.12 marker-aware dependency edges")
            active.append(item)
        else:
            reason = "; ".join(sorted(set(inactive_reasons.get(key, [])))) or "unreachable from the virtual root on Linux x86_64 CPython 3.12"
            if marker_active(row.get("resolution-markers", [])) and key not in inactive_reasons:
                reason = "unreachable from the virtual root on Linux x86_64 CPython 3.12 (no active marker-aware dependency edge)"
            item.update(status="INACTIVE_MARKER_ALTERNATIVE" if not marker_active(row.get("resolution-markers", [])) else "INACTIVE_UNREACHABLE_DEPENDENCY", reason=reason)
            inactive.append(item)
    return active, inactive


def _distribution_records() -> list[dict[str, Any]]:
    result = []
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if name and version:
            result.append({"distribution": dist, "name": name, "version": version, "identity": identity(name, version), "location": str(Path(dist.locate_file("")))})
    return sorted(result, key=lambda item: (item["identity"], item["location"]))


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    return {"expected": sorted(expected), "installed": sorted(actual), "missing": missing, "unexpected": unexpected, "duplicate_identities": sorted(item for item, count in actual_counts.items() if count > 1), "exact": not missing and not unexpected}


def _entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    path, root = Path(dist.locate_file(entry)), Path(dist.locate_file(""))
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return None
    return path


def _is_license_path(relative: str) -> bool:
    basename = Path(relative).name.casefold()
    return bool(re.fullmatch(r"(?:licen[cs]e|copying|notice|copyright)(?:[._-].*)?", basename))


def _publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result, unsafe = [], []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not _is_license_path(relative):
            continue
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
        else:
            result.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return result, unsafe


def _first_four(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


def _elf_needed(path: Path) -> dict[str, Any]:
    magic = _first_four(path)
    if magic != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        completed = subprocess.run(["readelf", "-d", str(path)], check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": str(exc)}
    needed = sorted(match.group(1) for line in completed.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
    output = f"{completed.stdout}\n{completed.stderr}"
    inspection = "ok" if completed.returncode == 0 else ("no-dynamic-section" if "no dynamic section" in output.casefold() else "error")
    result = {"format": "elf", "needed": needed, "inspection": inspection, "readelf_returncode": completed.returncode}
    if inspection == "error":
        result["error"] = output.strip()[-2000:]
    return result


def _native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result, unsafe = [], []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        basename = Path(relative).name.casefold()
        candidate = Path(basename).suffix in NATIVE_SUFFIXES or ".so." in basename or basename.endswith(".dll")
        try:
            path = _entry_path(dist, entry)
        except (OSError, ValueError) as exc:
            if candidate:
                unsafe.append(f"{relative}:locate-failed:{exc}")
            continue
        if path is None or path.is_symlink() or not path.is_file():
            if candidate:
                unsafe.append(f"{relative}:unsafe-path")
            continue
        try:
            magic = _first_four(path)
        except OSError as exc:
            if candidate:
                unsafe.append(f"{relative}:magic-read-failed:{exc}")
            continue
        if not candidate and magic != ELF_MAGIC:
            continue
        try:
            inspected, size, digest = _elf_needed(path), path.stat().st_size, sha256_file(path)
        except OSError as exc:
            unsafe.append(f"{relative}:native-read-failed:{exc}")
            continue
        result.append({"distribution_shipped": True, "bundled": True, "origin": "installed-distribution", "path": relative, "size": size, "sha256": digest, "candidate": "elf-magic" if magic == ELF_MAGIC and not candidate else "native-suffix", "needed": inspected})
    return result, unsafe


def _validate_sdist_url(artifact: dict[str, Any], url: str, *, initial: bool = False) -> None:
    expected = artifact.get("url")
    if not isinstance(expected, str) or not isinstance(url, str):
        raise AuditError("locked sdist URL is not a string")
    parsed, expected_parts = urlsplit(url), urlsplit(expected)
    try:
        port, expected_port = parsed.port, expected_parts.port
    except ValueError as exc:
        raise AuditError("locked sdist URL has an invalid port") from exc
    if parsed.scheme != "https" or parsed.hostname != PYPI_HOST or port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.path:
        raise AuditError(f"locked sdist URL is not the official exact host: {url}")
    if expected_parts.scheme != "https" or expected_parts.hostname != PYPI_HOST or expected_port not in (None, 443) or expected_parts.username or expected_parts.password or expected_parts.query or expected_parts.fragment or not expected_parts.path:
        raise AuditError("uv.lock sdist URL is not an exact official PyPI URL")
    if initial and url != expected:
        raise AuditError("initial sdist URL differs from uv.lock")
    if parsed.path != expected_parts.path:
        raise AuditError("sdist redirect changed the exact path")


class _SdistRedirects(HTTPRedirectHandler):
    def __init__(self, artifact: dict[str, Any], trace: list[str]) -> None:
        super().__init__(); self.artifact = artifact; self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("locked sdist redirect chain exceeds the bounded limit")
        if not isinstance(newurl, str):
            raise AuditError("locked sdist redirect is not a URL")
        resolved = urljoin(request.full_url, newurl)
        _validate_sdist_url(self.artifact, resolved)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _archive_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise AuditError("sdist archive contains an invalid/backslash member path")
    if name.startswith("/") or re.match(r"^[A-Za-z]:/", name):
        raise AuditError(f"sdist archive contains an absolute member path: {name}")
    parts = name.split("/")
    if any(part in ("..", ".") for part in parts) or any(part == "" for part in parts[:-1]):
        raise AuditError(f"sdist archive contains an unsafe member path: {name}")
    normalized = posixpath.normpath(name)
    if normalized in ("", ".") or normalized.startswith("../"):
        raise AuditError(f"sdist archive contains an unsafe member path: {name}")
    return normalized


def _archive_format(url: str) -> str:
    path = urlsplit(url).path.casefold()
    for suffix, result in ((".tar.gz", "tar.gz"), (".tgz", "tar.gz"), (".tar.bz2", "tar.bz2"), (".tbz2", "tar.bz2"), (".tar.xz", "tar.xz"), (".txz", "tar.xz"), (".zip", "zip")):
        if path.endswith(suffix):
            return result
    raise AuditError(f"unsupported sdist archive format: {path}")


def _archive_license_files(body: bytes, archive_format: str, archive_identity: dict[str, Any]) -> list[dict[str, Any]]:
    candidates, seen, aggregate = [], set(), 0

    def add(path: str, payload: bytes) -> None:
        if not _is_license_path(path):
            return
        if len(payload) > MAX_LICENSE_BYTES or sum(item["size"] for item in candidates) + len(payload) > MAX_LICENSE_BYTES:
            raise AuditError(f"sdist license candidate aggregate is oversized: {path}")
        candidates.append({"path": path, "size": len(payload), "sha256": sha256_bytes(payload), "content_base64": base64.b64encode(payload).decode("ascii"), "archive_identity": archive_identity})

    try:
        if archive_format == "zip":
            archive = zipfile.ZipFile(io.BytesIO(body))
            with archive:
                for number, info in enumerate(archive.infolist(), start=1):
                    if number > MAX_ARCHIVE_MEMBERS:
                        raise AuditError("sdist archive contains too many members")
                    path = _archive_path(info.filename)
                    if path in seen:
                        raise AuditError(f"sdist archive contains duplicate member: {path}")
                    seen.add(path)
                    file_type = (info.external_attr >> 16) & 0o170000
                    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise AuditError(f"sdist archive contains a special member: {path}")
                    if file_type == stat.S_IFREG and info.is_dir():
                        raise AuditError(f"sdist archive has a regular-file/directory contradiction: {path}")
                    if file_type == stat.S_IFDIR and not info.is_dir():
                        raise AuditError(f"sdist archive contains a malformed directory: {path}")
                    if info.is_dir():
                        continue
                    if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise AuditError(f"sdist archive member is oversized: {path}")
                    aggregate += info.file_size
                    if aggregate > MAX_ARCHIVE_TOTAL_BYTES:
                        raise AuditError("sdist archive aggregate is oversized")
                    if _is_license_path(path):
                        payload = archive.read(info)
                        if len(payload) != info.file_size:
                            raise AuditError(f"sdist license member size changed: {path}")
                        add(path, payload)
        else:
            modes = {"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}
            archive = tarfile.open(fileobj=io.BytesIO(body), mode=modes[archive_format])
            with archive:
                for number, member in enumerate(archive, start=1):
                    if number > MAX_ARCHIVE_MEMBERS:
                        raise AuditError("sdist archive contains too many members")
                    path = _archive_path(member.name)
                    if path in seen:
                        raise AuditError(f"sdist archive contains duplicate member: {path}")
                    seen.add(path)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise AuditError(f"sdist archive contains a non-file/non-directory member: {path}")
                    if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                        raise AuditError(f"sdist archive member is oversized: {path}")
                    aggregate += member.size
                    if aggregate > MAX_ARCHIVE_TOTAL_BYTES:
                        raise AuditError("sdist archive aggregate is oversized")
                    if _is_license_path(path):
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise AuditError(f"sdist license member is unreadable: {path}")
                        payload = stream.read(MAX_LICENSE_BYTES + 1)
                        if len(payload) != member.size:
                            raise AuditError(f"sdist license member size changed: {path}")
                        add(path, payload)
    except AuditError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed archive is a blocker
        raise AuditError(f"sdist archive inspection failed: {exc}") from exc
    if not candidates:
        raise AuditError("locked sdist contains no LICENSE/LICENCE/COPYING/NOTICE/COPYRIGHT candidate")
    return candidates


def _fetch_sdist(row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    artifact = row.get("sdist")
    if not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"} or not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip():
        raise AuditError(f"locked sdist is missing for {row.get('name')}=={row.get('version')}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact["hash"])) or isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0 or artifact["size"] > MAX_SDIST_BYTES:
        raise AuditError("locked sdist hash/size is invalid or above the audit bound")
    _validate_sdist_url(artifact, artifact["url"], initial=True)
    trace = [artifact["url"]]
    if fetcher is None:
        opener = build_opener(_SdistRedirects(artifact, trace))
        request = Request(artifact["url"], headers={"Accept": "application/octet-stream", "User-Agent": "vokra-ultravox-audit/1"})
        try:
            with opener.open(request, timeout=30) as response:  # noqa: S310 - exact URL validated
                final_url = urljoin(artifact["url"], response.geturl()); _validate_sdist_url(artifact, final_url)
                if final_url != trace[-1]:
                    trace.append(final_url)
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and int(length) > MAX_SDIST_BYTES:
                    raise AuditError("locked sdist Content-Length exceeds the audit bound")
                body = response.read(MAX_SDIST_BYTES + 1)
        except AuditError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport failure is factual blocker
            raise AuditError(f"locked sdist is not retrievable: {exc}") from exc
    else:
        try:
            final_url, body = fetcher(artifact["url"])
        except Exception as exc:  # noqa: BLE001 - injected transport failure is blocker
            raise AuditError(f"locked sdist is not retrievable: {exc}") from exc
        if not isinstance(final_url, str):
            raise AuditError("locked sdist fetcher returned a non-URL final value")
        final_url = urljoin(artifact["url"], final_url)
        _validate_sdist_url(artifact, final_url)
        if final_url != trace[-1]:
            trace.append(final_url)
    if not isinstance(body, bytes) or len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist body exceeds the bounded audit size")
    actual_hash = "sha256:" + sha256_bytes(body)
    if len(body) != artifact["size"] or actual_hash != artifact["hash"]:
        raise AuditError(f"locked sdist size/hash mismatch for {row.get('name')}=={row.get('version')}")
    archive_format = _archive_format(artifact["url"])
    archive_identity = {"requested_url": artifact["url"], "final_url": final_url, "url_trace": trace, "size": len(body), "sha256": actual_hash, "format": archive_format}
    return {"status": "PASS", "archive_identity": archive_identity, "publisher_files": _archive_license_files(body, archive_format, archive_identity)}


def _metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: "))
    return {"license": dist.metadata.get("License") or None, "license_expression": dist.metadata.get("License-Expression") or None, "license_classifiers": classifiers}


def _inspect_package(row: dict[str, Any], record: dict[str, Any] | None, duplicate: bool, sdist_fetcher: Callable[[str], tuple[str, bytes]] | None) -> tuple[dict[str, Any], list[str]]:
    lock_data = {"name": row["name"], "version": row["version"], "source": row["source"], "resolution_markers": row.get("resolution-markers", []), "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])}}
    if record is None:
        return {"lock": lock_data, "installed": None}, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe_publisher = _publisher_files(dist)
    native, unsafe_native = _native_files(dist)
    locked_sdist = None
    if not publisher:
        try:
            locked_sdist = _fetch_sdist(row, sdist_fetcher)
        except AuditError as exc:
            locked_sdist = {"status": "BLOCKED", "archive_identity": {"requested_url": row.get("sdist", {}).get("url") if isinstance(row.get("sdist"), dict) else None}, "publisher_files": [], "error": str(exc)}
    installed = {"name": dist.metadata.get("Name"), "version": dist.version, "normalized_identity": record["identity"], "location": record["location"], **_metadata_fields(dist), "publisher_files": publisher, "locked_sdist_license_audit": locked_sdist, "native_files": native, "bundled_libraries": [{"distribution": dist.metadata.get("Name"), "path": item["path"], "size": item["size"], "sha256": item["sha256"], "needed": item["needed"]} for item in native]}
    valid_sdist_license = bool(locked_sdist and locked_sdist.get("status") == "PASS" and locked_sdist.get("publisher_files"))
    failures = []
    if not installed["license"] and not installed["license_expression"] and not installed["license_classifiers"] and not valid_sdist_license:
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher and not valid_sdist_license:
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
    if locked_sdist and locked_sdist.get("status") == "BLOCKED":
        failures.append(f"locked sdist publisher evidence blocked: {record['identity']}: {locked_sdist['error']}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{item}" for item in unsafe_publisher)
    failures.extend(f"unsafe native path: {record['identity']}:{item}" for item in unsafe_native)
    if any(item["needed"]["inspection"] == "error" for item in native):
        failures.append(f"ELF NEEDED inspection failed: {record['identity']}")
    if duplicate:
        failures.append(f"duplicate installed distribution: {record['identity']}")
    return {"lock": lock_data, "installed": installed}, failures


def _dependency_acquisition(packages: list[dict[str, Any]]) -> dict[str, Any]:
    requests = []
    for package in packages:
        installed = package.get("installed")
        audit = installed.get("locked_sdist_license_audit") if isinstance(installed, dict) else None
        if not isinstance(audit, dict):
            continue
        lock = package["lock"]
        archive = audit.get("archive_identity", {})
        requests.append({"identity": identity(lock["name"], lock["version"]), "package": lock["name"], "requested_url": archive.get("requested_url"), "status": audit.get("status"), "purpose": "publisher-license-evidence-only"})
    return {"policy": "exact locked PyPI sdist only when installed publisher evidence is missing", "in_memory_archive_inspection": True, "requests": sorted(requests, key=lambda item: (item["identity"], item["requested_url"] or "")), "out_of_scope_requests": [], "model_files": []}


def _manifest_reviews(manifest: dict[str, Any]) -> dict[str, Any]:
    """Expose the recorded and independently recomputed review digests."""
    return {
        "package_review_rows": manifest["package_review_rows"],
        "package_review_rows_sha256": manifest["package_review_rows_sha256"],
        "license_rows": manifest["license_rows"],
        "license_rows_sha256": manifest["license_rows_sha256"],
        "license_rows_computed_sha256": license_gate.canonical_digest(manifest["license_rows"]),
        "approval": manifest["approval"],
        "publication": manifest["publication"],
    }


def _fixed_license_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    identities = manifest["identities"]
    return [{"id": "ultravox-public", "repo": identities["public_repo"], "revision": identities["public_revision"]}, {"id": "ultravox-upstream", "repo": identities["upstream_repo"], "revision": identities["upstream_revision"]}, {"id": "llama-companion", "repo": identities["companion_repo"], "revision": identities["companion_revision"]}]


def fixed_model_info_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the two exact upstream/companion model-info identities."""
    items = []
    for item in _fixed_license_items(manifest):
        expected = MODEL_INFO_EXPECTED.get(item["repo"])
        if expected is None or expected["revision"] != item["revision"]:
            continue
        items.append({
            **item,
            "requested_url": f"https://huggingface.co/api/models/{item['repo']}/revision/{item['revision']}",
            "expected_license": expected["license"],
            "expected_gated": expected["gated"],
            "expected_license_files": expected["license_files"],
        })
    if {item["repo"] for item in items} != set(MODEL_INFO_EXPECTED):
        raise AuditError("Ultravox exact HF model-info identity set is incomplete")
    return items


def _license_url(item: dict[str, Any]) -> str:
    return f"https://huggingface.co/{item['repo']}/raw/{item['revision']}/LICENSE"


def _validate_license_url(item: dict[str, Any], url: str, *, initial: bool = False) -> None:
    parsed = urlsplit(url); expected = urlsplit(_license_url(item))
    try:
        port, expected_port = parsed.port, expected.port
    except ValueError as exc:
        raise AuditError("LICENSE URL has an invalid port") from exc
    if parsed.scheme != "https" or parsed.hostname not in {"huggingface.co", *HF_CDN_LICENSE_HOSTS} or port not in (None, 443) or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AuditError("LICENSE URL host/scheme/query is not approved")
    raw_path = expected.path
    cdn_path = f"/{item['repo']}/resolve/{item['revision']}/LICENSE"
    if not ((parsed.hostname == "huggingface.co" and parsed.path == raw_path) or (parsed.hostname in HF_CDN_LICENSE_HOSTS and parsed.path == cdn_path)):
        raise AuditError("LICENSE URL is not the exact fixed identity path")
    if initial and url != _license_url(item):
        raise AuditError("initial LICENSE URL is not the generated fixed URL")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, Any], trace: list[str]) -> None:
        super().__init__(); self.item = item; self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_LICENSE_REDIRECTS:
            raise AuditError("LICENSE redirect chain exceeds the bounded limit")
        resolved = urljoin(request.full_url, newurl)
        _validate_license_url(self.item, resolved)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _fetch_license(item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    requested, trace = _license_url(item), []
    _validate_license_url(item, requested, initial=True); trace.append(requested)
    if fetcher is None:
        try:
            with build_opener(_LicenseRedirects(item, trace)).open(Request(requested, headers={"Accept": "text/plain", "User-Agent": "vokra-ultravox-audit/1"}), timeout=30) as response:
                final_url = response.geturl(); _validate_license_url(item, final_url)
                if final_url != trace[-1]: trace.append(final_url)
                body = response.read(MAX_LICENSE_BYTES + 1)
        except AuditError:
            raise
        except HTTPError as exc:
            raise LicensePathError(f"exact LICENSE path is not retrievable: HTTP {exc.code}", status_code=exc.code) from exc
        except Exception as exc:  # noqa: BLE001
            raise LicensePathError(f"exact LICENSE path is not retrievable: {exc}") from exc
    else:
        try:
            final_url, body = fetcher(requested)
        except HTTPError as exc:
            raise LicensePathError(f"exact LICENSE path is not retrievable: HTTP {exc.code}", status_code=exc.code) from exc
        _validate_license_url(item, final_url)
        if final_url != trace[-1]: trace.append(final_url)
    if not isinstance(body, bytes) or len(body) > MAX_LICENSE_BYTES:
        raise AuditError("LICENSE response exceeds the bounded audit size")
    return {"id": item["id"], "repo": item["repo"], "revision": item["revision"], "requested_url": requested, "final_url": final_url, "url_trace": trace, "acquired_file": "LICENSE", "size": len(body), "sha256": sha256_bytes(body), "content_base64": base64.b64encode(body).decode("ascii"), "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"}


def _model_info_url(item: dict[str, Any]) -> str:
    return item["requested_url"]


def _validate_model_info_url(item: dict[str, Any], url: str, *, initial: bool = False) -> None:
    expected = _model_info_url(item)
    if initial and url != expected:
        raise AuditError("initial HF model-info URL is not the exact revision endpoint")
    try:
        parsed = urlsplit(url)
    except TypeError as exc:
        raise AuditError("HF model-info URL is not a string") from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError("HF model-info URL has an invalid port") from exc
    if (
        parsed.scheme != "https" or parsed.hostname not in HF_API_HOSTS or parsed.username or parsed.password
        or parsed.query or parsed.fragment or port not in (None, 443)
        or parsed.path != f"/api/models/{item['repo']}/revision/{item['revision']}"
    ):
        raise AuditError("HF model-info URL is not the exact revision endpoint")


class _ModelInfoRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, Any], trace: list[str]) -> None:
        super().__init__()
        self.item = item
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_MODEL_INFO_REDIRECTS:
            raise AuditError("HF model-info redirect chain exceeds the bounded limit")
        resolved = urljoin(request.full_url, newurl)
        _validate_model_info_url(self.item, resolved)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _model_info_projection(item: dict[str, Any], body: bytes) -> dict[str, Any]:
    if not isinstance(body, bytes) or len(body) > MAX_MODEL_INFO_BYTES:
        raise AuditError("HF model-info response exceeds the bounded audit size")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate HF model-info JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError("HF model-info response is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AuditError("HF model-info response is not an object")
    required = {"id", "sha", "private", "gated", "disabled", "cardData", "siblings"}
    if not required.issubset(payload):
        raise AuditError("HF model-info response is missing required fields")
    if payload["id"] != item["repo"]:
        raise AuditError("HF model-info returned repository identity drifted")
    if not isinstance(payload["sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", payload["sha"]):
        raise AuditError("HF model-info returned sha is malformed")
    if payload["sha"] != item["revision"]:
        raise AuditError("HF model-info returned sha does not match the pinned revision")
    if payload["private"] is not False or payload["disabled"] is not False:
        raise AuditError("HF model-info repository is private or disabled")
    if payload["gated"] != item["expected_gated"]:
        raise AuditError("HF model-info gated state does not match the reviewed identity")
    card_data = payload["cardData"]
    if not isinstance(card_data, dict) or card_data.get("license") != item["expected_license"]:
        raise AuditError("HF model-info cardData.license does not match the reviewed tag")
    siblings = payload["siblings"]
    if not isinstance(siblings, list) or not siblings or len(siblings) > MAX_MODEL_INFO_MEMBERS:
        raise AuditError("HF model-info sibling tree is missing, empty, or oversized")
    tree_files: list[str] = []
    for entry in siblings:
        if not isinstance(entry, dict) or set(entry) != {"rfilename"} or not isinstance(entry["rfilename"], str):
            raise AuditError("HF model-info sibling schema drifted")
        path = entry["rfilename"]
        if (
            not path or "\x00" in path or "\\" in path or path.startswith("/") or path.endswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/")) or str(PurePosixPath(path)) != path
        ):
            raise AuditError("HF model-info sibling path is unsafe")
        if path in tree_files:
            raise AuditError("HF model-info sibling tree contains duplicate paths")
        tree_files.append(path)
    tree_files.sort()
    license_files = sorted(path for path in tree_files if _is_license_path(path))
    if license_files != sorted(item["expected_license_files"]):
        raise AuditError("HF model-info sibling license-file set does not match the reviewed identity")
    return {
        "schema": MODEL_INFO_SCHEMA,
        "id": item["id"],
        "repo": item["repo"],
        "requested_revision": item["revision"],
        "returned_repo": payload["id"],
        "returned_sha": payload["sha"],
        "private": payload["private"],
        "gated": payload["gated"],
        "disabled": payload["disabled"],
        "license": card_data["license"],
        "license_source": "HF_API_CARD_DATA_LICENSE",
        "license_files": license_files,
        "tree_file_count": len(tree_files),
        "tree_files_sha256": sha256_bytes(canonical_json(tree_files).encode("utf-8")),
        "payload_sha256": sha256_bytes(body),
        "payload_size": len(body),
    }


def _fetch_model_info(item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    requested = _model_info_url(item)
    _validate_model_info_url(item, requested, initial=True)
    trace = [requested]
    try:
        if fetcher is not None:
            final_url, body = fetcher(requested)
        else:
            opener = build_opener(_ModelInfoRedirects(item, trace))
            request = Request(requested, headers={"Accept": "application/json", "User-Agent": "vokra-ultravox-audit/1"})
            with opener.open(request, timeout=30) as response:
                length = response.headers.get("Content-Length")
                if length is not None and (not length.isdigit() or int(length) > MAX_MODEL_INFO_BYTES):
                    raise AuditError("HF model-info Content-Length exceeds the bounded audit size")
                final_url, body = response.geturl(), response.read(MAX_MODEL_INFO_BYTES + 1)
    except HTTPError:
        raise
    except AuditError:
        raise
    except Exception as exc:  # noqa: BLE001 - network/decoder failures are factual blockers
        raise AuditError(f"HF model-info request failed: {type(exc).__name__}") from exc
    if final_url != trace[-1]:
        trace.append(final_url)
    _validate_model_info_url(item, final_url)
    return {
        "id": item["id"],
        "repo": item["repo"],
        "revision": item["revision"],
        "requested_url": requested,
        "final_url": final_url,
        "redirect_trace": trace,
        "resolved_host": urlsplit(final_url).hostname,
        "resolved_path": urlsplit(final_url).path,
        **_model_info_projection(item, body),
    }


def fetch_model_license_metadata(item: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    """Public model-info fallback entrypoint for focused offline tests."""
    return _fetch_model_info(item, fetcher)


def requested_model_metadata_urls(records: list[dict[str, Any]]) -> list[str]:
    """Return only exact model-info URLs that were actually requested."""
    return [record["requested_url"] for record in records]


def _http_status(exc: BaseException) -> int | None:
    if isinstance(exc, HTTPError):
        return exc.code
    return getattr(exc, "status_code", None)


def audit_model_licenses(
    manifest: dict[str, Any],
    license_fetcher: Callable[[dict[str, Any]], dict[str, Any]] = _fetch_license,
    metadata_fetcher: Callable[[dict[str, Any]], dict[str, Any]] = fetch_model_license_metadata,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    files: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    failures: list[str] = []
    model_items = {item["id"]: item for item in fixed_model_info_items(manifest)}
    for item in _fixed_license_items(manifest):
        try:
            files.append(license_fetcher(item))
            continue
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            status = _http_status(exc)
            error = {"kind": "HTTP_ERROR", "status": status} if status is not None else {"kind": type(exc).__name__}
            record = {
                **item,
                "requested_url": _license_url(item),
                "final_url": None,
                "status": "BLOCKED_FACTUAL_LICENSE_PATH",
                "error": error,
            }
            model_item = model_items.get(item["id"])
            if status not in {401, 404} or model_item is None:
                files.append(record)
                failures.append(f"BLOCKED_FACTUAL_LICENSE_PATH: {item['id']}: {error}")
                continue
            try:
                metadata_record = metadata_fetcher(model_item)
            except (OSError, UnicodeError, ValueError, TypeError) as metadata_error:
                metadata_status = _http_status(metadata_error)
                metadata_error_value = {"kind": "HTTP_ERROR", "status": metadata_status} if metadata_status is not None else {"kind": type(metadata_error).__name__}
                record["error"] = error
                record["metadata_fallback"] = {"status": "BLOCKED", "error": metadata_error_value}
                files.append(record)
                failures.append(f"BLOCKED_FACTUAL_MODEL_METADATA_LICENSE: {item['id']}: {metadata_error_value}")
            else:
                record["status"] = "PASS_AUTHENTICATED_HF_METADATA_LICENSE"
                record["error"] = error
                record["metadata_fallback"] = {"status": "PASS", "requested_url": metadata_record["requested_url"]}
                files.append(record)
                metadata.append(metadata_record)
    return files, metadata, failures


def audit_environment(project: Path, fetch_model_licenses: bool = True, sdist_fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    project_data, lock, manifest, project_bytes, lock_bytes = _contract(project)
    active_rows, inactive_rows = classify_rows(lock)
    records = _distribution_records(); expected_ids = [identity(row["name"], row["version"]) for row in active_rows]
    # The virtual and inactive alternatives are intentionally excluded from installed closure expectations.
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for record in records: by_identity.setdefault(record["identity"], []).append(record)
    closure = compare_multiset(expected_ids, [record["identity"] for record in records])
    packages, failures = [], _approval_blockers(manifest)
    for row in active_rows:
        key = identity(row["name"], row["version"]); candidates = by_identity.get(key, [])
        package, package_failures = _inspect_package(row, candidates[0] if len(candidates) == 1 else None, len(candidates) > 1, sdist_fetcher)
        packages.append(package); failures.extend(package_failures)
    if not closure["exact"]: failures.append("installed normalized name+version multiset does not exactly match active Linux lock rows")
    if sys.version_info[:2] != (3, 12): failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}: failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")
    model_files, model_metadata, model_failures = [], [], []
    if fetch_model_licenses:
        model_files, model_metadata, model_failures = audit_model_licenses(manifest)
    failures.extend(model_failures)
    return {"schema": SCHEMA, "status": "BLOCKED" if failures else "PASS", "publication_permitted": False, "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "model_code_imported": False, "cargo_invoked": False, "readelf_required": True}, "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"], "pyproject_bytes": len(project_bytes), "pyproject_sha256": sha256_bytes(project_bytes), "uv_lock_bytes": len(lock_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes)}, "manifest_reviews": _manifest_reviews(manifest), "approval_blockers": sorted(set(_approval_blockers(manifest))), "lock_rows": {"accounted_rows": len(lock["package"]), "active_linux_installed": active_rows, "inactive_or_virtual": inactive_rows, "all_rows_accounted": len(active_rows) + len(inactive_rows) == len(lock["package"])}, "closure": closure, "packages": packages, "dependency_acquisition": _dependency_acquisition(packages), "fixed_source_model_companion_identities": _fixed_license_items(manifest), "model_license_files": model_files, "model_license_metadata": model_metadata, "model_acquisition": {"scope": "fixed source/model/Meta companion LICENSE paths plus exact HF model-info metadata fallback", "policy": "allow-listed exact primary-source LICENSE-only fetch; 401/404 model LICENSE failures may use exact-revision HF cardData.license and sibling-name metadata", "requested_files": [item["requested_url"] for item in model_files], "requested_metadata": [item["requested_url"] for item in model_metadata], "non_license_requests": [], "non_license_files": [], "proof": "bounded metadata only; no README/card text retention and no model-weight acquisition"}, "failures": sorted(set(failures))}


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try: report = audit_environment(project, fetch_model_licenses)
    except (AuditError, OSError, UnicodeError, ValueError) as exc: report = {"schema": SCHEMA, "status": "BLOCKED", "publication_permitted": False, "environment": {"model_code_imported": False, "cargo_invoked": False}, "dependency_acquisition": {"requests": [], "out_of_scope_requests": [], "model_files": []}, "model_acquisition": {"requested_files": [], "non_license_requests": [], "non_license_files": []}, "failures": [str(exc)]}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    if report["failures"]: print("ultravox dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr); return 2
    print(f"ultravox dependency audit: PASS ({output})"); return 0


def self_test() -> int:
    assert compare_multiset(["torch==2.13.0+cpu"], ["torch==2.13.0+cpu"])["exact"]
    assert identity("foo_bar", "1.0") == "foo-bar==1.0"
    assert marker_active(["sys_platform == 'linux' and platform_machine == 'x86_64'"])
    assert not marker_active(["sys_platform == 'darwin'"])
    manifest = strict_json(Path(__file__).resolve().parent / "license_gate_manifest.json")
    project_root = Path(__file__).resolve().parent
    _, checked_lock, checked_manifest, _, _ = _contract(project_root)
    active_rows, inactive_rows = classify_rows(checked_lock)
    assert len(checked_lock["package"]) == 40 and len(active_rows) == 37 and len(inactive_rows) == 3
    colorama_rows = [row for row in inactive_rows if row["name"] == "colorama"]
    assert len(colorama_rows) == 1 and colorama_rows[0]["status"] == "INACTIVE_UNREACHABLE_DEPENDENCY" and "win32" in colorama_rows[0]["reason"]
    assert any(row["name"] == "torch" and row["version"] == "2.13.0" and row["status"] == "INACTIVE_MARKER_ALTERNATIVE" for row in inactive_rows)
    assert len(checked_manifest["package_review_rows"]) == len(checked_lock["package"])
    assert len({(row["name"], row["version"], canonical_json(row["source"])) for row in checked_manifest["package_review_rows"]}) == len(checked_manifest["package_review_rows"])
    assert {row["id"] for row in checked_manifest["license_rows"]} == {"ultravox-audio-weight", "llama-companion-meta-conditional", "python-closure"}
    assert checked_manifest["approval"] == {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None}
    assert checked_manifest["publication"] == "NO_UPLOAD"
    manifest_reviews = _manifest_reviews(checked_manifest)
    assert set(manifest_reviews) == {"package_review_rows", "package_review_rows_sha256", "license_rows", "license_rows_sha256", "license_rows_computed_sha256", "approval", "publication"}
    assert manifest_reviews["license_rows_sha256"] == manifest_reviews["license_rows_computed_sha256"]
    assert "manifest:license_rows_sha256 does not match the recorded license_rows bytes" not in _approval_blockers(checked_manifest)
    stale_manifest = {**checked_manifest, "license_rows_sha256": "0" * 64}
    stale_reviews = _manifest_reviews(stale_manifest)
    assert stale_reviews["license_rows_sha256"] != stale_reviews["license_rows_computed_sha256"]
    assert "manifest:license_rows_sha256 does not match the recorded license_rows bytes" in _approval_blockers(stale_manifest)
    assert _approval_blockers(checked_manifest)
    item = _fixed_license_items(manifest)[0]; body = b"self-test LICENSE\n"; url = _license_url(item)
    assert _is_license_path("LICENSE.txt") and _is_license_path("LICENCE") and _is_license_path("licence.md") and _is_license_path("NOTICE-3")
    assert not _is_license_path("unlicensed-file") and not _is_license_path("noticeable")
    assert _fetch_license(item, lambda requested: (requested, body))["url_trace"] == [url]
    for bad_url in (
        url + "?x=1",
        url.replace("huggingface.co", "hf.co"),
        url.replace(item["repo"], "nearby/repo"),
        url.replace(item["revision"], "0" * 40),
        url.replace("/LICENSE", "/model.safetensors"),
        url.replace("https://", "https://user:pass@"),
        url.replace("https://huggingface.co", "https://huggingface.co:444"),
    ):
        try: _fetch_license(item, lambda requested, final=bad_url: (final, body))
        except AuditError: pass
        else: raise AssertionError(f"accepted unsafe LICENSE URL: {bad_url}")
    cdn_url = f"https://cdn-lfs.huggingface.co/{item['repo']}/resolve/{item['revision']}/LICENSE"
    trace = [url]
    _LicenseRedirects(item, trace).redirect_request(Request(url), None, 302, "", {}, cdn_url)
    assert trace == [url, cdn_url]
    boundary = [url] * MAX_LICENSE_REDIRECTS
    _LicenseRedirects(item, boundary).redirect_request(Request(url), None, 302, "", {}, cdn_url)
    assert len(boundary) == MAX_LICENSE_REDIRECTS + 1
    try: _LicenseRedirects(item, [url] * (MAX_LICENSE_REDIRECTS + 1)).redirect_request(Request(url), None, 302, "", {}, cdn_url)
    except AuditError: pass
    else: raise AssertionError("accepted overlong LICENSE redirect chain")

    model_items = fixed_model_info_items(manifest)
    assert {item["repo"] for item in model_items} == set(MODEL_INFO_EXPECTED)

    def model_payload(item: dict[str, Any]) -> dict[str, Any]:
        siblings = [{"rfilename": "README.md"}, {"rfilename": "config.json"}, {"rfilename": "model.safetensors"}]
        siblings.extend({"rfilename": name} for name in item["expected_license_files"])
        return {
            "id": item["repo"], "sha": item["revision"], "private": False,
            "gated": item["expected_gated"], "disabled": False,
            "cardData": {"license": item["expected_license"], "README": "ignored card text"},
            "siblings": siblings,
        }

    for model_item in model_items:
        model_result = _fetch_model_info(
            model_item,
            lambda requested, payload=model_payload(model_item): (requested, canonical_json(payload).encode("utf-8")),
        )
        assert model_result["requested_url"] == model_item["requested_url"]
        assert model_result["returned_sha"] == model_item["revision"]
        assert model_result["license"] == model_item["expected_license"]
        assert model_result["license_files"] == model_item["expected_license_files"]
        assert "README" not in model_result and "content_base64" not in model_result
    fixie_item = next(item for item in model_items if item["repo"].startswith("fixie-ai/"))
    expected_model_info_url = f"https://huggingface.co/api/models/{fixie_item['repo']}/revision/{fixie_item['revision']}"
    assert fixie_item["requested_url"] == expected_model_info_url
    for bad_url in (
        f"https://huggingface.co/api/models/{fixie_item['repo']}?revision={fixie_item['revision']}",
        fixie_item["requested_url"] + "?download=1",
        fixie_item["requested_url"].replace(f"/revision/{fixie_item['revision']}", "/revision/main"),
        fixie_item["requested_url"].replace("https://huggingface.co", "https://evil.example"),
        fixie_item["requested_url"].replace("https://", "https://audit-user@"),
        fixie_item["requested_url"] + "#fragment",
        fixie_item["requested_url"].replace("/api/models/", "/README.md/"),
    ):
        try:
            _fetch_model_info({**fixie_item, "requested_url": bad_url}, lambda _: (_ for _ in ()).throw(AssertionError("unsafe model-info URL reached fetcher")))
        except AuditError: pass
        else: raise AssertionError(f"accepted unsafe HF model-info URL: {bad_url}")
    model_trace = [fixie_item["requested_url"]]
    _ModelInfoRedirects(fixie_item, model_trace).redirect_request(Request(model_trace[0]), None, 302, "", {}, model_trace[0])
    assert len(model_trace) == 2
    try:
        _ModelInfoRedirects(fixie_item, [fixie_item["requested_url"]] * (MAX_MODEL_INFO_REDIRECTS + 1)).redirect_request(Request(fixie_item["requested_url"]), None, 302, "", {}, fixie_item["requested_url"])
    except AuditError: pass
    else: raise AssertionError("accepted overlong HF model-info redirect chain")
    base_payload = model_payload(fixie_item)
    invalid_payloads = (
        {**base_payload, "id": "other/model"}, {**base_payload, "sha": "0" * 40},
        {**base_payload, "private": True}, {**base_payload, "gated": "manual"},
        {**base_payload, "disabled": True}, {**base_payload, "cardData": {"license": "apache-2.0"}},
        {**base_payload, "siblings": []},
        {**base_payload, "siblings": [{"rfilename": "../config.json"}]},
        {**base_payload, "siblings": [{"rfilename": "config\\\\json"}]},
        {**base_payload, "siblings": [{"rfilename": "config.json"}, {"rfilename": "config.json"}]},
        {**base_payload, "siblings": [{"rfilename": "LICENSE"}]},
        {**base_payload, "siblings": [{"rfilename": "README.md", "size": 1}]},
    )
    for invalid in invalid_payloads:
        try:
            _fetch_model_info(fixie_item, lambda requested, invalid=invalid: (requested, canonical_json(invalid).encode("utf-8")))
        except AuditError: pass
        else: raise AssertionError("accepted invalid HF model-info projection")
    try:
        _fetch_model_info(fixie_item, lambda requested: (requested, b'{"id":"x","id":"x"}'))
    except AuditError: pass
    else: raise AssertionError("accepted duplicate HF model-info JSON keys")
    try:
        _fetch_model_info(fixie_item, lambda requested: (requested, b"x" * (MAX_MODEL_INFO_BYTES + 1)))
    except AuditError: pass
    else: raise AssertionError("accepted oversized HF model-info response")

    def synthetic_license_fetcher(license_item: dict[str, Any]) -> dict[str, Any]:
        if license_item["id"] == "ultravox-upstream":
            raise HTTPError(_license_url(license_item), 404, "missing", {}, None)
        if license_item["id"] == "llama-companion":
            raise HTTPError(_license_url(license_item), 401, "gated", {}, None)
        return {**license_item, "status": "PASS", "requested_url": _license_url(license_item)}

    def synthetic_metadata_fetcher(metadata_item: dict[str, Any]) -> dict[str, Any]:
        return _fetch_model_info(metadata_item, lambda requested: (requested, canonical_json(model_payload(metadata_item)).encode("utf-8")))

    fallback_files, fallback_metadata, fallback_failures = audit_model_licenses(
        manifest, synthetic_license_fetcher, synthetic_metadata_fetcher
    )
    assert not fallback_failures and len(fallback_metadata) == 2
    assert [record["status"] for record in fallback_files] == [
        "PASS", "PASS_AUTHENTICATED_HF_METADATA_LICENSE", "PASS_AUTHENTICATED_HF_METADATA_LICENSE"
    ]
    meta_record = next(record for record in fallback_metadata if record["repo"].startswith("meta-llama/"))
    assert meta_record["license_files"] == ["LICENSE.txt"] and "content_base64" not in meta_record

    def tar_bytes(entries: list[tuple[str, bytes, str]]) -> bytes:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, payload, kind in entries:
                member = tarfile.TarInfo(name)
                if kind == "dir": member.type = tarfile.DIRTYPE; archive.addfile(member)
                elif kind == "link": member.type = tarfile.SYMTYPE; member.linkname = "target"; archive.addfile(member)
                elif kind == "fifo": member.type = tarfile.FIFOTYPE; archive.addfile(member)
                else: member.size = len(payload); archive.addfile(member, io.BytesIO(payload))
        return output.getvalue()

    def row(body: bytes, suffix: str = ".tar.gz") -> dict[str, Any]:
        url = f"https://files.pythonhosted.org/packages/self-test/demo-1{suffix}"
        return {"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "sdist": {"url": url, "hash": "sha256:" + sha256_bytes(body), "size": len(body), "upload-time": "2026-01-01"}}

    good = tar_bytes([("demo-1/", b"", "dir"), ("demo-1/LICENSE", b"license", "file")]); good_row = row(good)
    assert _fetch_sdist(good_row, lambda url: (url, good))["publisher_files"][0]["size"] == 7
    assert _fetch_sdist(good_row, lambda url: ("./demo-1.tar.gz", good))["archive_identity"]["final_url"] == good_row["sdist"]["url"]
    british = tar_bytes([("demo-1/", b"", "dir"), ("demo-1/LICENCE", b"licence", "file")]); british_row = row(british)
    assert _fetch_sdist(british_row, lambda url: (url, british))["publisher_files"][0]["path"] == "demo-1/LICENCE"
    zip_body = io.BytesIO()
    with zipfile.ZipFile(zip_body, "w") as archive:
        archive.writestr("demo-1/LICENCE", b"licence")
    british_zip = zip_body.getvalue(); british_zip_row = row(british_zip, ".zip")
    assert _fetch_sdist(british_zip_row, lambda url: (url, british_zip))["publisher_files"][0]["path"] == "demo-1/LICENCE"
    redirect_trace = [good_row["sdist"]["url"]]
    _SdistRedirects(good_row["sdist"], redirect_trace).redirect_request(Request(redirect_trace[0]), None, 302, "", {}, "demo-1.tar.gz")
    assert redirect_trace == [good_row["sdist"]["url"]] * 2
    for bad in (tar_bytes([("../LICENSE", b"x", "file")]), tar_bytes([("demo-1\\LICENSE", b"x", "file")]), tar_bytes([("demo-1/LICENSE", b"", "link")]), tar_bytes([("demo-1/device", b"", "fifo")]), tar_bytes([("demo-1/unlicensed-file", b"x", "file")]), tar_bytes([("demo-1/README", b"x", "file")])):
        try: _fetch_sdist(row(bad), lambda url, payload=bad: (url, payload))
        except AuditError: pass
        else: raise AssertionError("accepted unsafe/no-license archive")
    zip_traversal = io.BytesIO()
    with zipfile.ZipFile(zip_traversal, "w") as archive:
        archive.writestr("../LICENCE", b"x")
    try: _archive_license_files(zip_traversal.getvalue(), "zip", {"requested_url": "self-test"})
    except AuditError: pass
    else: raise AssertionError("accepted ZIP traversal archive")
    bounded_members = MAX_ARCHIVE_MEMBERS
    try:
        globals()["MAX_ARCHIVE_MEMBERS"] = 1
        try: _fetch_sdist(good_row, lambda url: (url, good))
        except AuditError: pass
        else: raise AssertionError("accepted archive beyond member bound")
    finally:
        globals()["MAX_ARCHIVE_MEMBERS"] = bounded_members
    bounded_member_bytes = MAX_ARCHIVE_MEMBER_BYTES
    try:
        globals()["MAX_ARCHIVE_MEMBER_BYTES"] = 1
        try: _fetch_sdist(good_row, lambda url: (url, good))
        except AuditError: pass
        else: raise AssertionError("accepted oversized archive member")
    finally:
        globals()["MAX_ARCHIVE_MEMBER_BYTES"] = bounded_member_bytes
    bounded_total_bytes = MAX_ARCHIVE_TOTAL_BYTES
    try:
        globals()["MAX_ARCHIVE_TOTAL_BYTES"] = 1
        try: _fetch_sdist(good_row, lambda url: (url, good))
        except AuditError: pass
        else: raise AssertionError("accepted oversized archive aggregate")
    finally:
        globals()["MAX_ARCHIVE_TOTAL_BYTES"] = bounded_total_bytes
    special_zip = io.BytesIO()
    with zipfile.ZipFile(special_zip, "w") as archive:
        special = zipfile.ZipInfo("demo-1/LICENSE/")
        special.external_attr = stat.S_IFIFO << 16
        archive.writestr(special, b"x")
    special_body = special_zip.getvalue()
    try: _archive_license_files(special_body, "zip", {"requested_url": "self-test"})
    except AuditError: pass
    else: raise AssertionError("accepted ZIP special directory entry")
    for filename, mode in (("demo-1/LICENSE/", stat.S_IFREG), ("demo-1/LICENSE", stat.S_IFDIR)):
        contradiction = io.BytesIO()
        with zipfile.ZipFile(contradiction, "w") as archive:
            info = zipfile.ZipInfo(filename); info.external_attr = mode << 16; archive.writestr(info, b"x")
        try: _archive_license_files(contradiction.getvalue(), "zip", {"requested_url": "self-test"})
        except AuditError: pass
        else: raise AssertionError("accepted ZIP type/name contradiction")
    try: _fetch_sdist({**good_row, "sdist": {**good_row["sdist"], "hash": "sha256:" + "0" * 64}}, lambda url: (url, good))
    except AuditError: pass
    else: raise AssertionError("accepted sdist hash tamper")
    try: _fetch_sdist({**good_row, "sdist": {**good_row["sdist"], "url": good_row["sdist"]["url"].replace("files.pythonhosted.org", "example.invalid")}}, lambda url: (url, good))
    except AuditError: pass
    else: raise AssertionError("accepted sdist host tamper")
    for malformed_sdist in (
        {key: value for key, value in good_row["sdist"].items() if key != "upload-time"},
        {**good_row["sdist"], "extra": True},
        {**good_row["sdist"], "size": True},
        {**good_row["sdist"], "upload-time": ""},
    ):
        malformed_row = {**good_row, "sdist": malformed_sdist}
        try: _fetch_sdist(malformed_row, lambda url: (url, good))
        except AuditError: pass
        else: raise AssertionError("accepted malformed locked sdist schema")
    class FakeDist:
        files: list[str] = []; metadata = Message(); version = "1"
        def locate_file(self, entry: Any) -> Path: return Path(entry)
    FakeDist.metadata["Name"] = "demo"
    package, failures = _inspect_package(good_row, {"distribution": FakeDist(), "identity": "demo==1", "location": "self-test"}, False, lambda url: (url, good))
    assert package["installed"]["locked_sdist_license_audit"]["status"] == "PASS" and not any("missing package license metadata" in item for item in failures)
    blocked_row = row(good); blocked_row["sdist"]["hash"] = "sha256:" + "0" * 64
    blocked, failures = _inspect_package(blocked_row, {"distribution": FakeDist(), "identity": "demo==1", "location": "self-test"}, False, lambda url: (url, good))
    assert blocked["installed"]["locked_sdist_license_audit"]["status"] == "BLOCKED" and any("locked sdist publisher evidence blocked" in item for item in failures)
    no_sdist_row = row(good); no_sdist_row.pop("sdist")
    no_sdist, no_sdist_failures = _inspect_package(no_sdist_row, {"distribution": FakeDist(), "identity": "demo==1", "location": "self-test"}, False, lambda url: (url, good))
    assert no_sdist["installed"]["locked_sdist_license_audit"]["status"] == "BLOCKED" and any("locked sdist publisher evidence blocked" in item for item in no_sdist_failures)
    acquisition = _dependency_acquisition([package, blocked]); assert acquisition["out_of_scope_requests"] == [] and acquisition["model_files"] == [] and len(acquisition["requests"]) == 2
    with tempfile.TemporaryDirectory(prefix="ultravox-audit-self-test-") as directory:
        output = Path(directory) / "blocked.json"; assert run(Path(directory) / "missing", output, False) == 2; assert strict_json(output)["status"] == "BLOCKED"
    print("ultravox dependency audit: self-test PASS"); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--fetch-model-licenses", action="store_true"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None or args.fetch_model_licenses: parser.error("--self-test accepts no project/output/fetch arguments")
        return self_test()
    if args.project is None or args.output is None: parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__":
    raise SystemExit(main())
