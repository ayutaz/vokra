#!/usr/bin/env -S uv run --frozen --project tools/parity/owsm_v4_medium_1b_reference --python 3.12 python
"""Model-free audit of the frozen OWSM reference dependency closure.

This worker is deliberately separate from the OWSM frontend reference.  It
only examines the checked-in lock and the installed distributions on the
disposable worker.  It does not import ESPnet, torch, model code, or weights.
Package license files are read from the installed distribution; when a wheel
does not carry one, the exact locked PyPI sdist is fetched and its license
members are hashed after verifying the lock URL, size, and SHA-256.  The
report is factual evidence, not an owner approval, and therefore remains
blocked until the owner reviews every row (including native payloads).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
from urllib.parse import urlparse
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "vokra-owsm-v4-medium-1b-dependency-audit-v1"
PYPI_HOST = "files.pythonhosted.org"
PYPI_REGISTRY = "https://pypi.org/simple"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CPU_VERSION = "2.6.0+cpu"
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_LICENSE_TOTAL = 4 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000
ELF_MAGIC = b"\x7fELF"
LICENSE_BASENAMES = {
    "license", "licence", "copying", "notice", "copyright", "eula",
    "nvidia_sla", "nvidia-sla", "end_user_license", "end-user-license",
}
DIRECT = {
    "humanfriendly": "10.0",
    "librosa": "0.10.2.post1",
    "numpy": "2.2.5",
    "packaging": "24.2",
    "pyyaml": "6.0.3",
    "torch": "2.6.0",
    "torch-complex": "0.4.4",
    "typeguard": "4.6.0",
}


class AuditError(ValueError):
    """A structural, provenance, or safety failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def norm_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{norm_name(name)}=={re.sub(r'\\s+', '', version).casefold()}"


def license_candidate(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return name in LICENSE_BASENAMES or any(
        name.startswith(prefix + separator)
        for prefix in LICENSE_BASENAMES
        for separator in (".", "-", "_")
    )


def safe_relative(path: str) -> str:
    if not path or "\x00" in path or "\\" in path or path.startswith("/"):
        raise AuditError(f"unsafe archive path: {path!r}")
    clean = path.rstrip("/")
    if not clean or any(part in {"", ".", ".."} for part in clean.split("/")):
        raise AuditError(f"archive path traversal: {path!r}")
    return clean


def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = lock.get("package")
    if not isinstance(rows, list) or not rows:
        raise AuditError("uv.lock package table is missing")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for raw in rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str) or not isinstance(raw.get("version"), str):
            raise AuditError("uv.lock contains a malformed package row")
        source = raw.get("source")
        if not isinstance(source, dict) or len(source) != 1:
            raise AuditError(f"invalid package source: {raw['name']}")
        row = {
            "name": raw["name"], "version": raw["version"], "source": source,
            "resolution-markers": raw.get("resolution-markers", []),
            "dependencies": raw.get("dependencies", []),
        }
        key = (row["name"], row["version"], canonical(source))
        if key in identities:
            raise AuditError(f"duplicate lock identity: {key[:2]}")
        identities.add(key)
        if source == {"virtual": "."}:
            if set(raw) - {"name", "version", "source", "dependencies", "metadata"}:
                raise AuditError("virtual project row has unexpected fields")
            result.append(row)
            continue
        if source not in ({"registry": PYPI_REGISTRY}, {"registry": TORCH_CPU_INDEX}):
            raise AuditError(f"unapproved package source for {row['name']}: {source!r}")
        artifact = raw.get("sdist")
        # The official PyTorch CPU index publishes wheels but no sdist row.
        # Every PyPI row must retain the exact locked sdist for the bounded
        # primary-license fallback; torch is audited from its wheel metadata
        # and publisher files instead.
        if source == {"registry": PYPI_REGISTRY}:
            if not isinstance(artifact, dict) or set(artifact) != {"url", "hash", "size", "upload-time"}:
                raise AuditError(f"locked sdist identity missing for {row['name']}=={row['version']}")
            if not isinstance(artifact["url"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact["hash"])):
                raise AuditError(f"malformed locked sdist identity for {row['name']}")
            if not isinstance(artifact["size"], int) or artifact["size"] <= 0 or artifact["size"] > MAX_SDIST_BYTES:
                raise AuditError(f"unbounded locked sdist for {row['name']}")
            row["sdist"] = artifact
        elif artifact is not None:
            raise AuditError(f"non-PyPI row unexpectedly carries sdist: {row['name']}")
        row["wheels"] = raw.get("wheels", [])
        if not isinstance(row["wheels"], list):
            raise AuditError(f"locked wheels table is malformed for {row['name']}")
        result.append(row)
    return result


def active_linux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = []
    for row in rows:
        if row["source"] == {"virtual": "."}:
            continue
        markers = row["resolution-markers"]
        if not isinstance(markers, list) or any(not isinstance(item, str) for item in markers):
            raise AuditError(f"invalid resolution marker for {row['name']}")
        if not markers or "sys_platform != 'darwin'" in markers:
            active.append(row)
        elif markers != ["sys_platform == 'darwin'"]:
            raise AuditError(f"unsupported Linux resolution marker for {row['name']}: {markers!r}")
    return sorted(active, key=lambda row: (norm_name(row["name"]), row["version"]))


def validate_contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    project_path, lock_path = project / "pyproject.toml", project / "uv.lock"
    if not project.is_dir() or project.is_symlink() or not project_path.is_file() or not lock_path.is_file():
        raise AuditError("OWSM reference project/lock is missing or symlinked")
    project_bytes, lock_bytes = project_path.read_bytes(), lock_path.read_bytes()
    project_data = tomllib.loads(project_bytes.decode("utf-8"))
    lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    gate = project_data.get("tool", {}).get("vokra", {}).get("owsm_v4_medium_1b_reference", {})
    if gate.get("dependency_license_audit") != "PENDING_PRIMARY_SOURCE_AUDIT_FAIL_CLOSED":
        raise AuditError("OWSM gate must remain pending until owner review")
    dependencies = project_data.get("project", {}).get("dependencies", [])
    direct: dict[str, str] = {}
    for spec in dependencies:
        value = str(spec)
        if "==" not in value:
            raise AuditError(f"direct dependency is not exact-pinned: {value!r}")
        name, version = value.split("==", 1)
        direct[norm_name(name)] = version
    if direct != DIRECT:
        raise AuditError(f"direct dependency contract mismatch: {direct!r}")
    rows = package_rows(lock_data)
    if len(rows) != 43 or sum(row["source"] == {"virtual": "."} for row in rows) != 1:
        raise AuditError(f"expected 42 distributions plus one virtual row, found {len(rows)}")
    by_name = {norm_name(name): version for name, version in ((row["name"], row["version"]) for row in rows) if norm_name(name) in DIRECT}
    for name, version in DIRECT.items():
        if name != "torch" and by_name.get(name) != version:
            raise AuditError(f"direct lock pin mismatch: {name}=={by_name.get(name)!r}")
    torch_rows = [row for row in rows if norm_name(row["name"]) == "torch"]
    if {(row["version"], canonical(row["source"])) for row in torch_rows} != {
        ("2.6.0", canonical({"registry": TORCH_CPU_INDEX})),
        (TORCH_CPU_VERSION, canonical({"registry": TORCH_CPU_INDEX})),
    }:
        raise AuditError("torch lock does not contain the exact Darwin/Linux CPU pair")
    if any(norm_name(row["name"]) in {"triton", "nvidia-cuda", "cuda"} or norm_name(row["name"]).startswith("nvidia-") for row in rows):
        raise AuditError("CUDA/NVIDIA/Triton row is present in the OWSM lock")
    return project_data, lock_data, rows, project_bytes, lock_bytes


def installed_distributions() -> dict[str, list[metadata.Distribution]]:
    result: dict[str, list[metadata.Distribution]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(identity(name, dist.version), []).append(dist)
    return result


def safe_dist_file(dist: metadata.Distribution, entry: Any) -> Path | None:
    root = Path(dist.locate_file(""))
    path = Path(dist.locate_file(entry))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if root.is_symlink() or any((root / part).is_symlink() for part in relative.parts):
        return None
    resolved_root, resolved = root.resolve(), path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    files, unsafe = [], []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not license_candidate(relative):
            continue
        path = safe_dist_file(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        size = path.stat().st_size
        if size > MAX_LICENSE_BYTES:
            unsafe.append(f"{relative}:oversized")
            continue
        files.append({"path": relative, "size": size, "sha256": sha256_file(path)})
    return files, unsafe


def native_payloads(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    files, errors = [], []
    for entry in sorted(dist.files or [], key=str):
        path = safe_dist_file(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
            if magic != ELF_MAGIC:
                continue
            try:
                result = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True, timeout=60, check=False)
                needed = sorted(match.group(1) for line in result.stdout.splitlines() if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line)))
                inspection = "ok" if result.returncode == 0 else "error"
            except (OSError, subprocess.TimeoutExpired) as exc:
                needed, inspection = [], type(exc).__name__
            files.append({"path": str(entry), "size": path.stat().st_size, "sha256": sha256_file(path), "needed": needed, "inspection": inspection})
        except OSError as exc:
            errors.append(f"{entry}: {exc}")
    return files, errors


def native_inspection_failures(package: str, payloads: list[dict[str, Any]]) -> list[str]:
    """Turn every non-successful ELF inspection into a factual blocker."""

    return [
        f"native payload inspection incomplete: {package}: {item.get('path', '<unknown>')}"
        for item in payloads
        if item.get("inspection") != "ok"
    ]


def archive_license_files(url: str, body: bytes) -> list[dict[str, Any]]:
    if len(body) > MAX_SDIST_BYTES:
        raise AuditError("locked sdist exceeds size bound")
    if url.endswith((".tar.gz", ".tgz")):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
    elif url.endswith(".tar.bz2"):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:bz2")
    elif url.endswith(".tar.xz"):
        opener = lambda: tarfile.open(fileobj=io.BytesIO(body), mode="r:xz")
    elif url.endswith(".zip"):
        opener = lambda: zipfile.ZipFile(io.BytesIO(body))
    else:
        raise AuditError("unsupported locked sdist format")
    result: list[dict[str, Any]] = []
    total = 0
    with opener() as archive:
        members = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise AuditError("locked sdist has too many members")
        for info in members:
            name = info.filename if isinstance(archive, zipfile.ZipFile) else info.name
            clean = safe_relative(name)
            is_file = not (info.is_dir() if isinstance(archive, zipfile.ZipFile) else not info.isfile())
            if not is_file or not license_candidate(clean):
                continue
            size = info.file_size if isinstance(archive, zipfile.ZipFile) else info.size
            if size > MAX_LICENSE_BYTES or total + size > MAX_LICENSE_TOTAL:
                raise AuditError("locked sdist license bytes exceed bound")
            data = archive.read(info) if isinstance(archive, zipfile.ZipFile) else archive.extractfile(info).read()  # type: ignore[union-attr]
            if len(data) != size:
                raise AuditError("locked sdist license member size changed")
            total += size
            result.append({"path": clean, "size": size, "sha256": sha256_bytes(data)})
    if not result:
        raise AuditError("locked sdist has no license candidate")
    return sorted(result, key=lambda item: item["path"])


def locked_sdist_license(row: dict[str, Any]) -> dict[str, Any]:
    if row["source"] != {"registry": PYPI_REGISTRY}:
        return {"status": "NOT_APPLICABLE_NON_PYPI_SOURCE", "license_files": []}
    artifact = row["sdist"]
    url = artifact["url"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != PYPI_HOST or parsed.query or parsed.fragment:
        raise AuditError(f"locked sdist URL escaped PyPI host: {url}")
    request = urllib.request.Request(url, headers={"Accept": "application/octet-stream", "User-Agent": "vokra-owsm-dependency-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            final = response.geturl()
            body = response.read(MAX_SDIST_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        return {"status": "BLOCKED_LOCKED_SDIST_FETCH", "error": type(exc).__name__, "license_files": []}
    final_parts = urlparse(final)
    if final_parts.scheme != "https" or final_parts.hostname != PYPI_HOST or final_parts.path != parsed.path:
        raise AuditError(f"locked sdist redirect escaped or changed path: {final}")
    observed = sha256_bytes(body)
    if len(body) != artifact["size"] or "sha256:" + observed != artifact["hash"]:
        return {"status": "BLOCKED_LOCKED_SDIST_IDENTITY", "observed_size": len(body), "observed_sha256": "sha256:" + observed, "license_files": []}
    try:
        files = archive_license_files(final, body)
    except AuditError as exc:
        return {"status": "BLOCKED_LOCKED_SDIST_LICENSE", "error": str(exc), "license_files": []}
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "url": url, "final_url": final, "size": len(body), "sha256": "sha256:" + observed, "license_files": files}


def audit(project: Path, expected_head: str | None) -> dict[str, Any]:
    project_data, lock_data, rows, project_bytes, lock_bytes = validate_contract(project)
    active = active_linux(rows)
    installed = installed_distributions()
    expected_ids = [identity(row["name"], row["version"]) for row in active]
    observed_ids = sorted(installed)
    failures: list[str] = []
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append("audit host must be Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"audit Python must be 3.12, got {platform.python_version()}")
    if expected_ids != observed_ids:
        missing = sorted(set(expected_ids) - set(observed_ids)); extra = sorted(set(observed_ids) - set(expected_ids))
        failures.append(f"installed closure differs from Linux lock: missing={missing!r} extra={extra!r}")
    package_reports = []
    for row in active:
        key = identity(row["name"], row["version"]); candidates = installed.get(key, [])
        if len(candidates) != 1:
            package_reports.append({"lock": row, "installed": None, "status": "BLOCKED_INSTALLED_IDENTITY"})
            failures.append(f"installed distribution count is not one: {key}")
            continue
        dist = candidates[0]
        publisher, unsafe = publisher_files(dist)
        native, native_errors = native_payloads(dist)
        metadata_fields = {
            "name": dist.metadata.get("Name"), "version": dist.version,
            "license": dist.metadata.get("License"),
            "license_expression": dist.metadata.get("License-Expression"),
            "license_classifiers": sorted(value.removeprefix("License :: ") for value in (dist.metadata.get_all("Classifier") or []) if value.startswith("License :: ")),
        }
        sdist = None if publisher else locked_sdist_license(row)
        if not publisher and not (sdist and sdist.get("license_files")):
            failures.append(f"primary license bytes unavailable: {key}")
        if unsafe:
            failures.append(f"unsafe/oversized publisher license path: {key}")
        if native_errors:
            failures.extend(f"native payload inspection failed: {key}: {item}" for item in native_errors)
        failures.extend(native_inspection_failures(key, native))
        package_reports.append({"lock": row, "installed": {**metadata_fields, "publisher_license_files": publisher, "unsafe_license_paths": unsafe, "locked_sdist_license": sdist, "native_payloads": native}})
    status = "BLOCKED_FACTUAL_AUDIT" if failures else "BLOCKED_OWNER_REVIEW"
    return {
        "schema": SCHEMA, "status": status, "review": "PENDING_OWNER_APPROVAL",
        "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"], "pyproject_sha256": sha256_bytes(project_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes), "lock_sha256_scope": sha256_bytes(canonical(rows).encode())},
        "closure": {"lock_rows": len(rows), "active_linux_rows": len(active), "expected": sorted(expected_ids), "observed": observed_ids, "exact": expected_ids == observed_ids},
        "packages": package_reports, "failures": sorted(set(failures)),
        "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "readelf_required": True, "model_code_imported": False, "weights_acquired": False, "weights_imported": False, "weights_executed": False, "cargo_invoked": False},
        "git": {"expected_head": expected_head, "head_unverified": expected_head is None},
        "model_acquisition": {"requested_files": [], "policy": "NO_MODEL_OR_CHECKPOINT_REQUESTS"},
        "publication": "NO_UPLOAD",
    }


def write_no_replace(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise AuditError("output must be an absent absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())


def self_test() -> int:
    project = Path(__file__).resolve().parent
    _, lock, rows, _, _ = validate_contract(project)
    if len(active_linux(rows)) != 41:
        raise SystemExit("self-test expected 41 Linux active distributions")
    if any(norm_name(row["name"]).startswith("nvidia-") or norm_name(row["name"]) == "triton" for row in rows):
        raise SystemExit("self-test found accelerator row")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w:gz") as archive:
        value = b"Apache-2.0\n"; info = tarfile.TarInfo("pkg-1.0/LICENSE"); info.size = len(value); archive.addfile(info, io.BytesIO(value))
    files = archive_license_files("https://files.pythonhosted.org/pkg-1.0.tar.gz", body.getvalue())
    if files[0]["sha256"] != sha256_bytes(b"Apache-2.0\n"):
        raise SystemExit("self-test lost exact license bytes")
    if native_inspection_failures("demo==1", [{"path": "demo.so", "inspection": "error"}]) != [
        "native payload inspection incomplete: demo==1: demo.so"
    ]:
        raise SystemExit("self-test did not fail closed on an incomplete ELF inspection")
    if native_inspection_failures("demo==1", [{"path": "demo.so", "inspection": "ok"}]):
        raise SystemExit("self-test rejected a complete ELF inspection")
    for bad in ("../LICENSE", "/LICENSE", "pkg\\LICENSE", "pkg/./LICENSE"):
        try: safe_relative(bad)
        except AuditError: pass
        else: raise SystemExit(f"self-test accepted unsafe archive path: {bad}")
    with tempfile.TemporaryDirectory(prefix="owsm-dependency-audit-test-", dir="/private/tmp") as directory:
        output = Path(directory) / "report.json"; write_no_replace(output, b"owner")
        try: write_no_replace(output, b"clobber")
        except AuditError: pass
        else: raise SystemExit("self-test allowed report clobber")
    print("owsm_v4_medium_1b dependency_audit self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.expected_head: parser.error("--self-test accepts no other arguments")
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    try:
        report = audit(args.project, args.expected_head)
        write_no_replace(args.output, (canonical(report) + "\n").encode())
    except (AuditError, OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        report = {"schema": SCHEMA, "status": "BLOCKED_FACTUAL_AUDIT", "review": "PENDING_OWNER_APPROVAL", "failures": [str(exc)], "model_acquisition": {"requested_files": []}, "publication": "NO_UPLOAD"}
        try: write_no_replace(args.output, (canonical(report) + "\n").encode())
        except (AuditError, OSError) as write_error:
            print(f"OWSM dependency audit blocked: {write_error}", file=sys.stderr); return 2
    print(f"OWSM dependency audit: {report['status']} ({args.output})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
