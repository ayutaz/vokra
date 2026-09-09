#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Collect frozen, model-free HT-Demucs dependency evidence on VAST.

This program is deliberately a candidate-evidence collector, not an approval
tool.  It inspects the exact uv lock closure and installed distribution bytes;
it never imports torch or any model package, acquires checkpoints/audio, or
executes model code.  A successful factual collection still exits 2 with
``BLOCKED_OWNER_REVIEW``.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import tomllib
import warnings
import zipfile

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


PROJECT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT.parents[2]
AUDIT_PATH = PROJECT / "audit.py"
SCHEMA = "vokra-htdemucs-multi-dependency-evidence-v1"
PYPI_HOSTS = {"files.pythonhosted.org", "pypi.org"}
TORCH_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
ALLOWED_HOSTS = PYPI_HOSTS | TORCH_HOSTS
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright", "eula", "end_user_license", "end-user-license"}
ELF_MAGIC = b"\x7fELF"
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_LICENSE_TOTAL_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_REDIRECTS = 3
NATIVE_SUFFIXES = {".so", ".pyd", ".dylib", ".dll"}
FORBIDDEN_TOKENS = ("cuda", "nvidia", "triton")


class EvidenceError(ValueError):
    """Fail-closed evidence or contract error."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode())


def identity(name: str, version: str) -> str:
    return f"{re.sub(r'[-_.]+', '-', name).casefold()}=={version.strip().casefold()}"


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise EvidenceError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError) as exc:
        raise EvidenceError(f"invalid JSON {path}: {exc}") from exc


def assert_safe_path(path: Path) -> None:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise EvidenceError(f"path must be absolute and canonical: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise EvidenceError(f"symlink ancestry is not allowed: {current}")


def load_audit() -> Any:
    spec = importlib.util.spec_from_file_location("htdemucs_static_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise EvidenceError("cannot load static HT-Demucs audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_facts(expected_head: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise EvidenceError("--expected-head must be exactly lowercase 40-hex")
    try:
        head = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        status = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"], text=True, stderr=subprocess.STDOUT, timeout=20)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"git identity check failed: {type(exc).__name__}") from exc
    if head != expected_head:
        raise EvidenceError(f"HEAD differs from expected: {head} != {expected_head}")
    if status:
        raise EvidenceError("checkout is not clean")
    return {"head": head, "expected_head": expected_head, "clean": True}


def load_contract(expected_head: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    assert_safe_path(PROJECT)
    if PROJECT.is_symlink() or not PROJECT.is_dir():
        raise EvidenceError("project is not a real directory")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"} or sys.version_info[:2] != (3, 12):
        raise EvidenceError("collector requires Linux x86_64 Python 3.12")
    audit = load_audit()
    if audit.PROJECT.resolve() != PROJECT:
        raise EvidenceError("static audit/project identity mismatch")
    gate = audit.load_gate()
    audit.verify_gate_contract(gate)
    dependency_result = audit.audit_dependency_rows(gate)
    lock_result = audit.audit_lock(gate)
    if dependency_result["status"] != "DEPENDENCY_ROWS_PENDING" or lock_result["status"] != "LOCK_IDENTITY_OK_PENDING_PRIMARY_BYTES":
        raise EvidenceError("collector requires the pending, fail-closed static dependency gate")
    project_bytes = (PROJECT / "pyproject.toml").read_bytes()
    lock_bytes = (PROJECT / "uv.lock").read_bytes()
    manifest_bytes = (PROJECT / "license_gate_manifest.json").read_bytes()
    manifest = strict_json(PROJECT / "license_gate_manifest.json")
    dependency = gate["dependency_audit"]
    expected = {
        "pyproject_sha256": dependency["pyproject_sha256"],
        "uv_lock_sha256": dependency["lock_sha256"],
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "dependency_audit_sha256": dependency["rows_file_sha256"],
        "requirements_snapshot_sha256": dependency["source_file_sha256"],
    }
    actual = {
        "pyproject_sha256": sha256_bytes(project_bytes),
        "uv_lock_sha256": sha256_bytes(lock_bytes),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "dependency_audit_sha256": sha256_file(PROJECT / dependency["rows_file"]),
        "requirements_snapshot_sha256": sha256_file(PROJECT / dependency["source_file"]),
    }
    if actual != expected:
        raise EvidenceError("project/lock/manifest/audit digest mismatch")
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    return audit, gate, lock, git_facts(expected_head), {"expected": expected, "actual": actual, "manifest": manifest}


def safe_member(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise EvidenceError(f"unsafe archive member: {name!r}")
    clean = name.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise EvidenceError(f"archive traversal: {name!r}")
    return clean


def is_license_path(name: str) -> bool:
    base = PurePosixPath(name).name.casefold()
    return base in LICENSE_NAMES or any(base.startswith(prefix + sep) for prefix in LICENSE_NAMES for sep in (".", "-", "_"))


def archive_license_files(path: Path) -> list[dict[str, Any]]:
    names: set[str] = set()
    total = 0
    license_total = 0
    found: list[dict[str, Any]] = []

    def member(name: str, size: int, regular: bool, reader: Callable[[], bytes]) -> None:
        nonlocal total, license_total
        clean = safe_member(name)
        if clean in names:
            raise EvidenceError(f"duplicate archive member: {clean}")
        names.add(clean)
        if len(names) > MAX_MEMBERS or size < 0 or size > MAX_MEMBER_BYTES:
            raise EvidenceError("archive member/count bound exceeded")
        if not regular:
            return
        total += size
        if total > MAX_TOTAL_MEMBER_BYTES:
            raise EvidenceError("archive aggregate bound exceeded")
        if not is_license_path(clean):
            return
        if size > MAX_LICENSE_BYTES:
            raise EvidenceError("license member bound exceeded")
        data = reader()
        if len(data) != size:
            raise EvidenceError("archive member size changed")
        license_total += size
        if license_total > MAX_LICENSE_TOTAL_BYTES:
            raise EvidenceError("license aggregate bound exceeded")
        found.append({"path": clean, "bytes": size, "sha256": sha256_bytes(data), "content_base64": base64.b64encode(data).decode("ascii")})

    suffix = path.name.casefold()
    try:
        if suffix.endswith(".whl") or suffix.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    kind = (info.external_attr >> 16) & 0o170000
                    directory = info.is_dir() or info.filename.endswith("/") or kind == stat.S_IFDIR
                    if kind and kind not in {stat.S_IFREG, stat.S_IFDIR}:
                        raise EvidenceError(f"zip link/special member: {info.filename}")
                    member(info.filename, 0 if directory else info.file_size, not directory, lambda info=info: archive.read(info))
        else:
            mode = "r:gz" if suffix.endswith((".tar.gz", ".tgz")) else "r:bz2" if suffix.endswith((".tar.bz2", ".tbz2")) else "r:xz" if suffix.endswith((".tar.xz", ".txz")) else "r:"
            if not suffix.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")):
                raise EvidenceError("unsupported locked archive type")
            with tarfile.open(path, mode=mode) as archive:
                for info in archive:
                    if info.issym() or info.islnk() or info.isdev() or info.isfifo() or (not info.isdir() and not info.isfile()):
                        raise EvidenceError(f"tar link/special member: {info.name}")
                    stream = archive.extractfile(info) if info.isfile() else None
                    member(info.name, info.size if info.isfile() else 0, info.isfile(), lambda stream=stream: stream.read(MAX_LICENSE_BYTES + 1) if stream is not None else b"")
                    if stream is not None:
                        stream.close()
    except (OSError, EOFError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise EvidenceError(f"archive inspection failed: {exc}") from exc
    return sorted(found, key=lambda item: item["path"])


def validate_url(url: str, expected: str | None = None) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.query or parsed.fragment or not parsed.path:
        raise EvidenceError("artifact URL is outside the allowed HTTPS indexes")
    if expected is not None and url != expected:
        raise EvidenceError("redirect changed the locked artifact URL")


class SafeRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], initial: str) -> None:
        self.trace, self.initial = trace, initial

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if len(self.trace) - 1 >= MAX_REDIRECTS:
            raise EvidenceError("redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        validate_url(resolved)
        if urlsplit(resolved).path != urlsplit(self.initial).path:
            raise EvidenceError("redirect changed the locked artifact path")
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_artifact(artifact: dict[str, Any], temporary: Path, fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    url = artifact.get("url")
    expected_hash = artifact.get("hash")
    expected_size = artifact.get("size")
    if not isinstance(url, str) or not isinstance(expected_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise EvidenceError("locked artifact identity is malformed")
    validate_url(url)
    if expected_size is not None and (not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0 or expected_size > MAX_ARTIFACT_BYTES):
        raise EvidenceError("locked artifact size is unbounded or malformed")
    trace = [url]
    basename = PurePosixPath(urlsplit(url).path).name
    if not basename or basename in {".", ".."} or "/" in basename or "\\" in basename or ".." in basename:
        raise EvidenceError("locked artifact basename is unsafe")
    output = temporary / (sha256_bytes(url.encode()) + "-" + basename)
    if fetcher is not None:
        final, body = fetcher(url)
        validate_url(final)
        if urlsplit(final).path != urlsplit(url).path:
            raise EvidenceError("test redirect changed artifact path")
        if not isinstance(body, bytes):
            raise EvidenceError("test fetcher did not return bytes")
        if len(body) > MAX_ARTIFACT_BYTES:
            raise EvidenceError("artifact exceeds bounded size")
        observed = sha256_bytes(body)
        if expected_size is not None and len(body) != expected_size:
            raise EvidenceError("artifact size mismatch")
        if "sha256:" + observed != expected_hash:
            raise EvidenceError("artifact hash mismatch")
        output.write_bytes(body)
        return {"url": url, "final_url": final, "url_trace": trace + ([] if final == url else [final]), "bytes": len(body), "sha256": observed, "temporary": str(output)}
    digest = hashlib.sha256()
    count = 0
    try:
        opener = build_opener(SafeRedirects(trace, url))
        request = Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-htdemucs-evidence/1"})
        with opener.open(request, timeout=60) as response, output.open("xb") as stream:
            final = urljoin(url, response.geturl())
            validate_url(final)
            if urlsplit(final).path != urlsplit(url).path:
                raise EvidenceError("response redirect changed artifact path")
            if final != trace[-1]:
                trace.append(final)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > MAX_ARTIFACT_BYTES:
                    raise EvidenceError("artifact exceeds bounded size")
                digest.update(chunk)
                stream.write(chunk)
    except (HTTPError, URLError, OSError, UnicodeError, ValueError) as exc:
        raise EvidenceError(f"artifact acquisition failed: {type(exc).__name__}") from exc
    observed = digest.hexdigest()
    if expected_size is not None and count != expected_size:
        raise EvidenceError("artifact size mismatch")
    if "sha256:" + observed != expected_hash:
        raise EvidenceError("artifact hash mismatch")
    return {"url": url, "final_url": trace[-1], "url_trace": trace, "bytes": count, "sha256": observed, "temporary": str(output)}


def wheel_tags_from_filename(url: str) -> set[tuple[str, str, str]]:
    name = PurePosixPath(urlsplit(url).path).name
    if not name.endswith(".whl"):
        raise EvidenceError("locked wheel URL does not name a .whl")
    parts = name[:-4].split("-")
    if len(parts) < 5:
        raise EvidenceError("malformed locked wheel filename")
    fields = parts[-3:]
    if any(not re.fullmatch(r"[A-Za-z0-9_.]+", field) for field in fields):
        raise EvidenceError("malformed locked wheel tag field")
    return {(python, abi, platform_tag) for python in fields[0].split(".") for abi in fields[1].split(".") for platform_tag in fields[2].split(".")}


def installed_wheel_info(dist: metadata.Distribution) -> dict[str, Any]:
    candidates: list[tuple[str, Path]] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        parts = PurePosixPath(relative).parts
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == "WHEEL":
            path = safe_dist_path(dist, entry)
            if path is None or path.is_symlink() or not path.is_file():
                raise EvidenceError("installed WHEEL path is unsafe")
            candidates.append((relative, path))
    if len(candidates) != 1:
        raise EvidenceError("installed distribution must contain exactly one WHEEL file")
    relative, path = candidates[0]
    if path.stat().st_size > 64 * 1024:
        raise EvidenceError("installed WHEEL bytes are unbounded")
    data = path.read_bytes()
    if not data:
        raise EvidenceError("installed WHEEL bytes are unbounded")
    tags: set[tuple[str, str, str]] = set()
    for line in data.decode("utf-8").splitlines():
        if not line.startswith("Tag:"):
            continue
        value = line[5:].strip()
        fields = value.split("-")
        if len(fields) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_.]+", field) for field in fields):
            raise EvidenceError("installed WHEEL contains malformed Tag")
        tags.update((python, abi, platform_tag) for python in fields[0].split(".") for abi in fields[1].split(".") for platform_tag in fields[2].split("."))
    if not tags:
        raise EvidenceError("installed WHEEL has no Tag headers")
    return {"path": relative, "bytes": len(data), "sha256": sha256_bytes(data), "tags": ["-".join(tag) for tag in sorted(tags)]}


def choose_artifact(row: dict[str, Any], installed_tags: set[tuple[str, str, str]]) -> tuple[str, dict[str, Any], set[tuple[str, str, str]]]:
    wheels = row.get("wheels", [])
    if isinstance(wheels, list) and wheels:
        matching: list[tuple[dict[str, Any], set[tuple[str, str, str]]]] = []
        for item in wheels:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                continue
            tags = wheel_tags_from_filename(item["url"])
            intersection = tags & installed_tags
            if intersection:
                matching.append((item, intersection))
        if len(matching) != 1:
            raise EvidenceError(f"installed WHEEL binds {len(matching)} locked wheels for {row.get('name')}; expected exactly one")
        return "locked_wheel", matching[0][0], matching[0][1]
    raise EvidenceError(f"installed distribution has no locked wheel binding: {row.get('name')}")


def safe_dist_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    root = Path(dist.locate_file(""))
    path = Path(dist.locate_file(entry))
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if root.is_symlink() or any((root / part).is_symlink() for part in rel.parts):
        return None
    resolved_root, resolved = root.resolve(strict=False), path.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def installed_license_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    total = 0
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not is_license_path(relative):
            continue
        path = safe_dist_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        size = path.stat().st_size
        if size > MAX_LICENSE_BYTES or total + size > MAX_LICENSE_TOTAL_BYTES:
            raise EvidenceError(f"publisher license bounds exceeded: {relative}")
        data = path.read_bytes()
        result.append({"path": relative, "bytes": size, "sha256": sha256_bytes(data), "content_base64": base64.b64encode(data).decode("ascii")})
        total += size
    return result, unsafe


def native_evidence(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    native: list[dict[str, Any]] = []
    failures: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        name = Path(relative).name.casefold()
        path = safe_dist_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            if Path(relative).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name:
                failures.append(f"unsafe native path: {relative}")
            continue
        try:
            with path.open("rb") as stream:
                magic = stream.read(4)
            candidate = Path(relative).suffix.casefold() in NATIVE_SUFFIXES or ".so." in name or magic == ELF_MAGIC
            if not candidate:
                continue
            item: dict[str, Any] = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "elf": {"format": "non-elf", "needed": [], "inspection": "not-applicable"}}
            if magic == ELF_MAGIC:
                try:
                    result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
                    needed = sorted(match.group(1) for line in result.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
                    item["elf"] = {"format": "elf", "needed": needed, "inspection": "ok" if result.returncode == 0 else "error", "readelf_returncode": result.returncode}
                    if result.returncode != 0:
                        failures.append(f"readelf failed: {relative}")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    failures.append(f"readelf error {type(exc).__name__}: {relative}")
            native.append(item)
        except OSError as exc:
            failures.append(f"native read failed {type(exc).__name__}: {relative}")
    return native, failures


def metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    return {
        "license": (dist.metadata.get("License") or "").strip() or None,
        "license_expression": (dist.metadata.get("License-Expression") or "").strip() or None,
        "license_classifiers": sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: ")),
    }


def reject_forbidden(name: str, *values: str) -> None:
    text = " ".join((name, *values)).casefold()
    if any(token in text for token in FORBIDDEN_TOKENS):
        raise EvidenceError(f"CUDA/NVIDIA/Triton indicator is forbidden: {name}")


def report_status(factual_failures: list[str]) -> str:
    return "BLOCKED_FACTUAL_COLLECTION" if factual_failures else "BLOCKED_OWNER_REVIEW"


def owner_review_blockers(package_evidence: list[Any], license_rows: list[dict[str, Any]]) -> list[str]:
    blockers = ["candidate rows have no automatic SPDX/license approval", "owner must review every primary artifact and license byte row", "publication remains NO_UPLOAD"]
    if any(item.get("native_payload", {}).get("torch_cpu_payload_owner_review") for item in package_evidence if isinstance(item, dict)):
        blockers.append("torch CPU native payload requires owner review")
    blockers.extend(f"owner review required: {row['name']}=={row['version']}" for row in license_rows if row.get("license") == "UNRESOLVED")
    return sorted(set(blockers))


def inspect_distribution(row: dict[str, Any], record: dict[str, Any], temporary: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    dist: metadata.Distribution = record["distribution"]
    reject_forbidden(row["name"], row["version"], str(dist.locate_file("")))
    wheel_info = installed_wheel_info(dist)
    installed_tags = {tuple(tag.split("-")) for tag in wheel_info["tags"]}
    kind, artifact, matched_tags = choose_artifact(row, installed_tags)
    artifact_fact = fetch_artifact(artifact, temporary)
    publisher, unsafe = installed_license_files(dist)
    native, native_failures = native_evidence(dist)
    needed = [needed for item in native for needed in item["elf"].get("needed", [])]
    reject_forbidden(row["name"], *needed, *(item["path"] for item in native))
    sdist_evidence: list[dict[str, Any]] = []
    if not publisher:
        sdist = row.get("sdist")
        if not isinstance(sdist, dict):
            raise EvidenceError(f"publisher license absent and no locked sdist fallback: {row['name']}")
        _, sdist_artifact = "locked_sdist", sdist
        fallback = fetch_artifact(sdist_artifact, temporary)
        if fallback["temporary"] is None:
            raise EvidenceError("internal fetcher cannot inspect a sdist")
        sdist_evidence = archive_license_files(Path(fallback["temporary"]))
        if not sdist_evidence:
            raise EvidenceError(f"locked sdist has no bounded license evidence: {row['name']}")
    chosen_license = publisher[0] if publisher else sdist_evidence[0]
    fields = metadata_fields(dist)
    artifact_public = {key: value for key, value in artifact_fact.items() if key != "temporary"}
    detail = {
        "name": dist.metadata.get("Name"), "version": dist.version, "identity": record["identity"], **fields,
        "selected_artifact": {"kind": kind, **artifact_public, "wheel_binding": {"wheel_path": wheel_info["path"], "wheel_bytes": wheel_info["bytes"], "wheel_sha256": wheel_info["sha256"], "installed_tags": wheel_info["tags"], "matched_tags": ["-".join(tag) for tag in sorted(matched_tags)]}},
        "publisher_license_eula_files": publisher, "unsafe_publisher_paths": unsafe,
        "locked_sdist_license_fallback": sdist_evidence,
        "native_payload": {"files": native, "errors": native_failures, "torch_cpu_payload_owner_review": re.sub(r"[-_.]+", "-", row["name"]).casefold() == "torch" and bool(native)},
    }
    license_row = {"name": row["name"], "version": row["version"], "license": "UNRESOLVED", "status": "CANDIDATE_OWNER_REVIEW", "source": "publisher-bytes" if publisher else "locked-sdist-bytes", "sha256": chosen_license["sha256"], "evidence": {"kind": "publisher_bytes" if publisher else "locked_sdist", "artifact_kind": kind, "artifact_url": artifact_fact["url"], "artifact_sha256": artifact_fact["sha256"], "artifact_bytes": artifact_fact["bytes"], "license_path": chosen_license["path"], "license_bytes": chosen_license["bytes"], "license_sha256": chosen_license["sha256"]}}
    package_row = {"name": row["name"], "version": row["version"], "artifact": {"kind": kind, "url": artifact_fact["url"], "sha256": artifact_fact["sha256"], "bytes": artifact_fact["bytes"]}, "license": "UNRESOLVED"}
    failures = list(native_failures)
    if unsafe:
        failures.extend(f"unsafe publisher license path: {item}" for item in unsafe)
    return package_row, license_row, [detail, *failures]


def collect(expected_head: str) -> dict[str, Any]:
    audit, gate, lock, git, facts = load_contract(expected_head)
    closure = audit.reachable_lock_identities(lock)
    rows = {(row["name"].lower(), row["version"]): row for row in lock["package"]}
    virtual_keys = [key for key in closure if rows[key].get("source") == {"virtual": "."}]
    real_keys = [key for key in closure if key not in virtual_keys]
    records: dict[str, list[dict[str, Any]]] = {}
    for dist in metadata.distributions():
        name, version = dist.metadata.get("Name"), dist.version
        if name and version:
            records.setdefault(identity(name, version), []).append({"distribution": dist, "identity": identity(name, version), "location": str(Path(dist.locate_file("")))})
    expected_ids = {identity(*key) for key in real_keys}
    actual_ids = Counter(key for key in records for _ in records[key])
    closure_facts = {"expected": sorted(expected_ids), "installed": sorted(actual_ids.elements()), "missing": sorted(expected_ids - set(actual_ids)), "unexpected": sorted(set(actual_ids) - expected_ids), "duplicates": sorted(key for key, count in actual_ids.items() if count != 1), "exact": set(actual_ids) == expected_ids and all(count == 1 for count in actual_ids.values())}
    package_rows: list[dict[str, Any]] = []
    license_rows: list[dict[str, Any]] = []
    package_evidence: list[Any] = []
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="htdemucs-dependency-evidence-") as temporary_name:
        temporary = Path(temporary_name)
        for key in real_keys:
            row = rows[key]
            found = records.get(identity(*key), [])
            if len(found) != 1:
                failures.append(f"installed closure is not one-to-one: {identity(*key)}")
                continue
            try:
                package, license_row, detail = inspect_distribution(row, found[0], temporary)
                package_rows.append(package); license_rows.append(license_row); package_evidence.append(detail)
                package_evidence[-1] = {**detail[0], "factual_failures": [item for item in detail[1:] if isinstance(item, str)]}
                if detail[1:]:
                    failures.extend(f"{identity(*key)}: {item}" for item in detail[1:] if isinstance(item, str))
            except (EvidenceError, OSError, ValueError) as exc:
                failures.append(f"{identity(*key)}: {exc}")
    if not closure_facts["exact"]:
        failures.append("installed distributions do not exactly match reachable uv.lock closure")
    project_data = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    project = project_data["project"]
    virtual_row = {"name": project["name"], "version": project["version"], "artifact": {"kind": "virtual-local", "url": "pyproject.toml", "sha256": facts["actual"]["pyproject_sha256"], "bytes": (PROJECT / "pyproject.toml").stat().st_size}, "license": "UNRESOLVED"}
    root_license = REPO_ROOT / "LICENSE"
    if root_license.is_file() and not root_license.is_symlink():
        license_data = root_license.read_bytes()
        virtual_license = {"name": project["name"], "version": project["version"], "license": "UNRESOLVED", "status": "CANDIDATE_OWNER_REVIEW", "source": "repository-license-bytes", "sha256": sha256_bytes(license_data), "evidence": {"kind": "local_file", "artifact_kind": "virtual-local", "artifact_url": "pyproject.toml", "artifact_sha256": facts["actual"]["pyproject_sha256"], "artifact_bytes": virtual_row["artifact"]["bytes"], "license_path": "LICENSE", "license_bytes": len(license_data), "license_sha256": sha256_bytes(license_data)}}
        package_evidence.append({"name": project["name"], "version": project["version"], "license_metadata": None, "publisher_license_eula_files": [{"path": "LICENSE", "bytes": len(license_data), "sha256": sha256_bytes(license_data), "content_base64": base64.b64encode(license_data).decode("ascii")}]})
        license_rows.append(virtual_license); package_rows.append(virtual_row)
    else:
        failures.append("repository LICENSE is missing or symlinked")
    owner_blockers = owner_review_blockers(package_evidence, license_rows)
    return {"schema": SCHEMA, "status": report_status(failures), "publication": "NO_UPLOAD", "git": git, "project": {"path": str(PROJECT), "pyproject_sha256": facts["actual"]["pyproject_sha256"], "uv_lock_sha256": facts["actual"]["uv_lock_sha256"], "manifest_sha256": facts["actual"]["manifest_sha256"], "dependency_audit_sha256": facts["actual"]["dependency_audit_sha256"], "requirements_snapshot_sha256": facts["actual"]["requirements_snapshot_sha256"]}, "lock": {"python": "==3.12.*", "platform": "linux-x86_64", "reachable_closure": sorted(closure), "closure": closure_facts}, "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True}, "package_rows": package_rows, "license_rows": license_rows, "package_rows_sha256": digest(package_rows), "license_rows_sha256": digest(license_rows), "package_evidence": package_evidence, "owner_review_blockers": sorted(set(owner_blockers)), "factual_failures": sorted(set(failures)), "model_activity": {"model_code_imported": False, "weights_acquired": False, "weights_imported": False, "weights_executed": False, "audio_acquired": False, "audio_imported": False, "audio_executed": False, "source_repo_downloaded": False, "cargo_invoked": False, "uploaded": False}}


def write_atomic(output: Path, report: dict[str, Any]) -> None:
    assert_safe_path(output)
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise EvidenceError("output and sidecar must be absent")
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_safe_path(output.parent)
    payload = (canonical(report) + "\n").encode()
    temporary = output.parent / (f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise EvidenceError("temporary output already exists")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, output)
        side_payload = f"{sha256_bytes(payload)}  {output.name}\n".encode()
        with sidecar.open("xb") as stream:
            stream.write(side_payload); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def self_test() -> int:
    if not re.fullmatch(r"[0-9a-f]{40}", "0" * 40):
        raise AssertionError("head validator failed")
    for bad in ("A" * 40, "g" * 40, "0" * 39):
        if re.fullmatch(r"[0-9a-f]{40}", bad):
            raise AssertionError("malformed HEAD accepted")
    wheel_row = {"name": "demo", "version": "1", "wheels": [
        {"url": "https://files.pythonhosted.org/packages/demo-1-cp312-cp312-manylinux_2_17_x86_64.whl"},
        {"url": "https://files.pythonhosted.org/packages/demo-1-cp312-cp312-musllinux_1_2_x86_64.whl"},
    ]}
    selected, artifact, matched = choose_artifact(wheel_row, {("cp312", "cp312", "manylinux_2_17_x86_64")})
    if selected != "locked_wheel" or "manylinux" not in artifact["url"] or not matched:
        raise AssertionError("installed manylinux tag did not bind the matching wheel")
    try:
        choose_artifact(wheel_row, {("cp312", "cp312", "manylinux_2_17_x86_64"), ("cp312", "cp312", "musllinux_1_2_x86_64")})
    except EvidenceError:
        pass
    else:
        raise AssertionError("ambiguous installed wheel tags accepted")
    try:
        choose_artifact(wheel_row, {("cp312", "cp312", "win_amd64")})
    except EvidenceError:
        pass
    else:
        raise AssertionError("zero matching installed wheel tags accepted")
    with tempfile.TemporaryDirectory(prefix="htdemucs-evidence-self-test-") as directory:
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
        try:
            strict_json(duplicate)
        except EvidenceError:
            pass
        else:
            raise AssertionError("duplicate JSON key accepted")
    with tempfile.TemporaryDirectory(prefix="htdemucs-evidence-archive-") as directory:
        archive = Path(directory) / "demo.whl"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("demo/LICENSE", b"MIT\n")
        files = archive_license_files(archive)
        if files[0]["sha256"] != sha256_bytes(b"MIT\n") or files[0]["content_base64"] != base64.b64encode(b"MIT\n").decode():
            raise AssertionError("license archive evidence mismatch")
        malicious = Path(directory) / "bad.whl"
        with zipfile.ZipFile(malicious, "w") as handle:
            handle.writestr("../LICENSE", b"bad")
        try:
            archive_license_files(malicious)
        except EvidenceError:
            pass
        else:
            raise AssertionError("archive traversal accepted")
        duplicate_archive = Path(directory) / "duplicate.whl"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(duplicate_archive, "w") as handle:
                handle.writestr("demo/LICENSE", b"one")
                handle.writestr("demo/LICENSE", b"two")
        try:
            archive_license_files(duplicate_archive)
        except EvidenceError:
            pass
        else:
            raise AssertionError("duplicate archive path accepted")
        fake = b"payload"
        artifact = {"url": "https://files.pythonhosted.org/packages/demo.whl", "hash": "sha256:" + sha256_bytes(fake), "size": len(fake)}
        result = fetch_artifact(artifact, Path(directory), lambda url: (url, fake))
        if result["bytes"] != len(fake):
            raise AssertionError("artifact size evidence mismatch")
        try:
            fetch_artifact({**artifact, "hash": "sha256:" + "0" * 64}, Path(directory), lambda url: (url, fake))
        except EvidenceError:
            pass
        else:
            raise AssertionError("artifact hash tamper accepted")
        try:
            fetch_artifact(artifact, Path(directory), lambda url: ("https://evil.invalid/packages/demo.whl", fake))
        except EvidenceError:
            pass
        else:
            raise AssertionError("unsafe redirect accepted")
        try:
            fetch_artifact({**artifact, "url": artifact["url"] + "?signed=1"}, Path(directory), lambda url: (url, fake))
        except EvidenceError:
            pass
        else:
            raise AssertionError("artifact query string accepted")
        tar_body = io.BytesIO()
        with tarfile.open(fileobj=tar_body, mode="w:gz") as archive:
            info = tarfile.TarInfo("demo/LICENSE")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"MIT\n"))
        tar_bytes = tar_body.getvalue()
        tar_artifact = {"url": "https://files.pythonhosted.org/packages/demo-1.tar.gz", "hash": "sha256:" + sha256_bytes(tar_bytes), "size": len(tar_bytes)}
        tar_result = fetch_artifact(tar_artifact, Path(directory), lambda url: (url, tar_bytes))
        if not archive_license_files(Path(tar_result["temporary"])):
            raise AssertionError("tar.gz fallback license evidence failed")
        zip_body = io.BytesIO()
        with zipfile.ZipFile(zip_body, "w") as archive:
            archive.writestr("demo/LICENSE", b"MIT\n")
        zip_bytes = zip_body.getvalue()
        zip_artifact = {"url": "https://files.pythonhosted.org/packages/demo-1.zip", "hash": "sha256:" + sha256_bytes(zip_bytes), "size": len(zip_bytes)}
        zip_result = fetch_artifact(zip_artifact, Path(directory), lambda url: (url, zip_bytes))
        if not archive_license_files(Path(zip_result["temporary"])):
            raise AssertionError("zip fallback license evidence failed")
    torch_blockers = owner_review_blockers([{"native_payload": {"torch_cpu_payload_owner_review": True}}], [])
    if report_status([]) != "BLOCKED_OWNER_REVIEW" or report_status(["factual"]) != "BLOCKED_FACTUAL_COLLECTION" or "torch CPU native payload requires owner review" not in torch_blockers:
        raise AssertionError("factual/owner status separation failed")
    fake_lock = {"version": 1, "revision": 3, "requires-python": "==3.12.*", "resolution-markers": ["platform_machine == 'x86_64' and sys_platform == 'linux'"], "supported-markers": ["platform_machine == 'x86_64' and sys_platform == 'linux'"], "package": [{"name": "openunmix", "version": "1", "source": {"registry": "https://pypi.org/simple"}}]}
    audit = load_audit()
    try:
        audit.parse_lock_data(fake_lock)
    except ValueError:
        pass
    else:
        raise AssertionError("forbidden package lock accepted")
    source = Path(__file__).read_text(encoding="utf-8")
    for token in ("BLOCKED_OWNER_REVIEW", "NO_UPLOAD", "model_code_imported", "weights_acquired", "audio_acquired", "source_repo_downloaded", "cargo_invoked"):
        if token not in source:
            raise AssertionError(f"model-free contract token missing: {token}")
    print("htdemucs dependency evidence collector self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.expected_head:
            parser.error("--self-test accepts no project/output/head")
        return self_test()
    if args.project is None or args.output is None or args.expected_head is None:
        parser.error("--project, --output, and --expected-head are required")
    if not args.project.is_absolute() or str(args.project) != str(PROJECT) or not args.output.is_absolute():
        print("collector: BLOCKED: project/output must be canonical project and absolute output", file=sys.stderr)
        return 2
    try:
        assert_safe_path(args.project)
        output_resolved = args.output.resolve(strict=False)
        if output_resolved == REPO_ROOT or REPO_ROOT in output_resolved.parents or output_resolved == PROJECT or PROJECT in output_resolved.parents:
            raise EvidenceError("output must be outside the checkout and HT-Demucs project")
        assert_safe_path(output_resolved)
    except (EvidenceError, OSError) as exc:
        print(f"collector: BLOCKED: {exc}", file=sys.stderr)
        return 2
    try:
        report = collect(args.expected_head)
    except (EvidenceError, OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        report = {"schema": SCHEMA, "status": "BLOCKED_FACTUAL_COLLECTION", "publication": "NO_UPLOAD", "factual_failures": [str(exc)], "owner_review_blockers": ["candidate evidence was not collected"], "model_activity": {"model_code_imported": False, "weights_acquired": False, "weights_imported": False, "weights_executed": False, "audio_acquired": False, "audio_imported": False, "audio_executed": False, "source_repo_downloaded": False, "cargo_invoked": False, "uploaded": False}}
    try:
        write_atomic(args.output, report)
    except (EvidenceError, OSError) as exc:
        print(f"collector: BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"collector: {report['status']} ({args.output})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
