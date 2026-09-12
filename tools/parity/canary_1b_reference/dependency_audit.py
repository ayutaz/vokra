#!/usr/bin/env -S uv run --project tools/parity/canary_1b_reference --frozen --python 3.12 python
"""Model-free factual audit for the dedicated Canary NeMo closure.

This program intentionally imports only Python's standard library and
``importlib.metadata``.  It never imports NeMo, accesses a model/source
repository, reads a checkpoint, invokes Cargo, or uploads an artifact.  The
result is evidence for a later owner/legal review and is permanently
fail-closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
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


SCHEMA = "vokra-canary-1b-reference-dependency-audit-v1"
STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
OWNER_REVIEW = "PENDING_OWNER_APPROVAL"
PYPI = "https://pypi.org/simple"
PYTORCH_CPU = "https://download.pytorch.org/whl/cpu"
REGISTRIES = {PYPI, PYTORCH_CPU}
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd", ".a"}
ELF_MAGIC = b"\x7fELF"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MIN_MEMORY_BYTES = 64 * 1024**3
MODEL_FREE_FIELDS = {
    "weights_acquired": False,
    "source_acquired": False,
    "model_imported": False,
    "model_executed": False,
    "cargo_invoked": False,
    "upload": PUBLICATION,
}


class AuditError(ValueError):
    """Malformed or unsafe audit input."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def reject_symlink_ancestry(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or "/")
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise AuditError(f"symlinked path ancestry: {path}")


def regular_file(path: Path, label: str) -> Path:
    reject_symlink_ancestry(path)
    if not path.is_file() or path.is_symlink():
        raise AuditError(f"{label} is missing or not a regular file: {path}")
    return path


def regular_directory(path: Path, label: str) -> Path:
    reject_symlink_ancestry(path)
    if not path.is_dir() or path.is_symlink():
        raise AuditError(f"{label} is missing or symlinked: {path}")
    return path


def absent_output(path: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise AuditError(f"output must be an absent absolute path: {path}")
    reject_symlink_ancestry(path.parent)
    regular_directory(path.parent, "output parent")


def safe_relative(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise AuditError(f"unsafe distribution path: {value!r}")
    return PurePosixPath(normalized)


def write_no_replace(path: Path, payload: bytes) -> None:
    absent_output(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def load_toml(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    source = regular_file(path, label)
    try:
        payload = source.read_bytes()
        return payload, tomllib.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"cannot parse {label}: {error}") from error


def lock_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*":
        raise AuditError("uv lock version or Python contract is not exact")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise AuditError("uv lock package table is empty")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, str]] = set()
    for row in packages:
        if not isinstance(row, dict):
            raise AuditError("uv lock contains a non-object package row")
        name, version, source = row.get("name"), row.get("version"), row.get("source")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise AuditError("uv lock package identity is malformed")
        identity = (
            name.casefold(), version, canonical(source),
            canonical(row.get("resolution-markers", [])),
        )
        if identity in identities:
            raise AuditError(f"duplicate lock package identity: {identity}")
        identities.add(identity)
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise AuditError(f"lock source is malformed for {name}")
        if "registry" in source and source["registry"] not in REGISTRIES:
            raise AuditError(f"unapproved lock registry for {name}: {source}")
        rows.append(row)
    roots = [row for row in rows if row.get("source") == {"virtual": "."}]
    if len(roots) != 1:
        raise AuditError(f"expected one virtual project row, found {len(roots)}")
    return rows


def marker_context() -> dict[str, str]:
    """Return the actual host/interpreter values used for lock markers."""
    return {
        "sys_platform": sys.platform,
        "platform_machine": platform.machine().lower(),
        "platform_system": platform.system(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full_version": platform.python_version(),
        "implementation_name": sys.implementation.name,
    }


def marker_active(marker: str | None, *, extra: str | None = None, context: dict[str, str] | None = None) -> bool:
    """Evaluate the small PEP 508 marker subset emitted by uv for this host."""
    if marker is None or not marker.strip():
        return True
    values = marker_context() if context is None else dict(context)
    values["extra"] = extra or ""
    for disjunction in re.split(r"\s+or\s+", marker.strip()):
        terms = re.split(r"\s+and\s+", disjunction)
        if all(_marker_term(term, values) for term in terms):
            return True
    return False


def numeric_python_version(value: str, term: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in value.split("."))
    except ValueError as error:
        raise AuditError(f"invalid Python version marker: {term!r}") from error
    return parts + (0,) * max(0, 3 - len(parts))


def _marker_term(term: str, context: dict[str, str]) -> bool:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(not in|in|==|!=|<=|>=|<|>)\s*(['\"])(.*?)\3\s*", term)
    if match is None:
        raise AuditError(f"unsupported uv marker expression: {term!r}")
    variable, operator, _, expected = match.groups()
    actual = context.get(variable)
    if actual is None:
        raise AuditError(f"unknown uv marker variable: {variable}")
    is_python_version = variable in {"python_version", "python_full_version"}
    if is_python_version and operator not in {"in", "not in"}:
        actual_value = numeric_python_version(actual, term)
        expected_value = numeric_python_version(expected, term)
    else:
        actual_value, expected_value = actual, expected
    if operator == "==":
        return actual_value == expected_value
    if operator == "!=":
        return actual_value != expected_value
    if operator == "in":
        return actual_value in expected_value
    if operator == "not in":
        return actual_value not in expected_value
    if operator == "<":
        return actual_value < expected_value
    if operator == ">":
        return actual_value > expected_value
    if operator == "<=":
        return actual_value <= expected_value
    return actual_value >= expected_value


def active_lock_rows(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Resolve the root project's Linux x86_64 dependency/extras closure."""
    rows = lock_rows(lock)
    root = next(row for row in rows if row.get("source") == {"virtual": "."})
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source") != {"virtual": "."}:
            by_name.setdefault(str(row["name"]).casefold(), []).append(row)
    active: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    processed_extras: dict[tuple[str, str, str, str], set[str]] = {}
    paths: dict[tuple[str, str, str, str], set[tuple[str, ...]]] = {}
    failures: list[str] = []
    queue: list[tuple[str, list[str], str | None, tuple[str, ...]]] = []
    for requirement in root.get("metadata", {}).get("requires-dist", []):
        if not isinstance(requirement, dict):
            failures.append("root requires-dist row is malformed")
            continue
        if marker_active(requirement.get("marker")):
            extras = list(requirement.get("extras", []))
            suffix = f"[{','.join(extras)}]" if extras else ""
            queue.append((str(requirement["name"]), extras, requirement.get("marker"), ("project", f"{requirement['name']}{suffix}")))
    while queue:
        name, extras, _marker, dependency_path = queue.pop(0)
        candidates = [
            row for row in by_name.get(name.casefold(), [])
            if all(marker_active(marker) for marker in row.get("resolution-markers", []))
        ]
        if len(candidates) != 1:
            failures.append(f"active package resolution is not unique for {name}: {len(candidates)} candidates")
            continue
        row = candidates[0]
        key = (str(row["name"]).casefold(), str(row["version"]), canonical(row["source"]), canonical(row.get("resolution-markers", [])))
        paths.setdefault(key, set()).add(dependency_path)
        first_visit = key not in active
        if first_visit:
            active[key] = row
        known_extras = processed_extras.setdefault(key, set())
        new_extras = [selected for selected in extras if selected not in known_extras]
        known_extras.update(new_extras)
        if not first_visit and not new_extras:
            continue
        if first_visit:
            for dependency in row.get("dependencies", []):
                if marker_active(dependency.get("marker")):
                    target_extras = dependency.get("extra", [])
                    if isinstance(target_extras, str):
                        target_extras = [target_extras]
                    if not isinstance(target_extras, list) or not all(isinstance(item, str) for item in target_extras):
                        failures.append(f"dependency extra metadata is malformed for {dependency.get('name')}")
                        continue
                    suffix = f"[{','.join(target_extras)}]" if target_extras else ""
                    queue.append((str(dependency["name"]), target_extras, dependency.get("marker"), (*dependency_path, f"{dependency['name']}{suffix}")))
        optional = row.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for selected_extra in new_extras:
                for dependency in optional.get(selected_extra, []):
                    if isinstance(dependency, dict) and marker_active(dependency.get("marker"), extra=selected_extra):
                        queue.append((str(dependency["name"]), [], dependency.get("marker"), (*dependency_path, f"{dependency['name']}[extra={selected_extra}]")))
    inactive = [row for row in rows if row.get("source") != {"virtual": "."} and (
        str(row["name"]).casefold(), str(row["version"]), canonical(row["source"]), canonical(row.get("resolution-markers", []))
    ) not in active]
    rows = [root, *active.values()]
    dependency_paths = []
    for row in rows:
        if row.get("source") == {"virtual": "."}:
            row_paths = [["project"]]
        else:
            key = (str(row["name"]).casefold(), str(row["version"]), canonical(row["source"]), canonical(row.get("resolution-markers", [])))
            row_paths = [list(path) for path in sorted(paths.get(key, set()))]
        dependency_paths.append({
            "name": row["name"],
            "version": row["version"],
            "source": row["source"],
            "resolution_markers": row.get("resolution-markers", []),
            "paths": row_paths,
        })
    return rows, inactive, sorted(set(failures)), dependency_paths


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def verify_canonical_sources(repo_root: Path, project_path: Path, lock_path: Path, wrapper_path: Path) -> None:
    """Ensure direct auditor callers cannot substitute external source files."""
    root = regular_directory(repo_root, "repository root").resolve(strict=True)
    expected = {
        "project": root / "tools/parity/canary_1b_reference/pyproject.toml",
        "lock": root / "tools/parity/canary_1b_reference/uv.lock",
        "auditor": root / "tools/parity/canary_1b_reference/dependency_audit.py",
        "wrapper": root / "scripts/publish/vast-ai/audit-canary-1b-dependencies.sh",
    }
    actual = {
        "project": regular_file(project_path, "dedicated pyproject").resolve(strict=True),
        "lock": regular_file(lock_path, "dedicated uv.lock").resolve(strict=True),
        "auditor": regular_file(Path(__file__), "dependency auditor").resolve(strict=True),
        "wrapper": regular_file(wrapper_path, "executed audit wrapper").resolve(strict=True),
    }
    for label in expected:
        if actual[label] != expected[label]:
            raise AuditError(f"{label} is outside the canonical repository path")


def installed_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = sorted(
        f"{normalized_name(str(row['name']))}=={row['version']}"
        for row in rows
        if row.get("source") != {"virtual": "."}
    )
    installed: list[str] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name") or getattr(distribution, "name", None)
        version = distribution.metadata.get("Version") or getattr(distribution, "version", None)
        if isinstance(name, str) and isinstance(version, str):
            installed.append(f"{normalized_name(name)}=={version}")
    expected_counts, installed_counts = Counter(expected), Counter(installed)
    missing = sorted((expected_counts - installed_counts).elements())
    unexpected = sorted((installed_counts - expected_counts).elements())
    duplicates = sorted(name for name, count in installed_counts.items() if count > 1)
    return {
        "expected": expected,
        "installed": sorted(installed),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "exact": not missing and not unexpected and not duplicates,
    }


def license_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    base = path.name.casefold()
    return (
        base in LICENSE_NAMES
        or base in {"licenses", "licences"}
        or any(base.startswith(f"{item}.") for item in LICENSE_NAMES)
        or any(part.casefold() in {"licenses", "licences"} for part in path.parts)
    )


def native_name(name: str) -> bool:
    lower = name.casefold()
    return lower.endswith(tuple(NATIVE_SUFFIXES)) or ".so." in lower


def is_elf(path: Path) -> bool:
    try:
        if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            with path.open("rb") as handle:
                return handle.read(4) == ELF_MAGIC
    except OSError as error:
        raise AuditError(f"cannot inspect native candidate {path}: {error}") from error
    return False


def native_fact(path: Path, relative: PurePosixPath) -> dict[str, Any]:
    fact = {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    with path.open("rb") as handle:
        elf = handle.read(4) == ELF_MAGIC
    fact["format"] = "elf" if elf else "archive-or-other"
    fact["needed"] = []
    if elf:
        try:
            result = subprocess.run(
                ["readelf", "-d", str(path)], capture_output=True, text=True,
                check=False, timeout=30,
            )
            if result.returncode == 0:
                fact["needed"] = sorted(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout))
            else:
                fact["readelf_error"] = f"exit-{result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as error:
            fact["readelf_error"] = type(error).__name__
    return fact


def license_flags(publisher_license: str, classifiers: list[str], files: list[dict[str, Any]]) -> list[str]:
    """Record license signals without turning metadata into an approval."""
    text = " ".join([publisher_license, *classifiers, *(str(item["path"]) for item in files)]).casefold()
    flags = [name for name in ("AGPL", "GPL", "LGPL") if re.search(rf"(?<![a-z]){name.casefold()}", text)]
    if flags:
        return flags
    return ["UNKNOWN"] if not text.strip() else ["UNCLASSIFIED_DECLARED"]


def distribution_fact(row: dict[str, Any], archive_dir: Path) -> dict[str, Any]:
    name, version = str(row["name"]), str(row["version"])
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise AuditError(f"locked package is not installed: {name}=={version}") from error
    if distribution.version != version:
        raise AuditError(f"installed version drift: {name} {distribution.version} != {version}")
    root = Path(distribution.locate_file(""))
    prefix = Path(sys.prefix).resolve(strict=True)
    licenses: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    inventory_errors: list[str] = []
    for entry in sorted(distribution.files or [], key=str):
        raw = str(entry).replace("\\", "/")
        candidate = license_name(raw) or native_name(raw)
        if raw.startswith("/") or any(part in {"", ".", ".."} for part in raw.split("/")):
            if not candidate:
                continue
            inventory_errors.append(f"unsafe candidate path: {raw}")
            continue
        try:
            relative = safe_relative(raw)
            path = Path(distribution.locate_file(entry))
            if path.is_symlink() or not path.is_file():
                raise AuditError(f"candidate is not a regular file: {raw}")
            resolved = path.resolve(strict=True)
            resolved.relative_to(prefix)
            if license_name(raw):
                payload = resolved.read_bytes()
                if len(payload) > 2 * 1024 * 1024:
                    raise AuditError(f"publisher license file is too large: {raw}")
                archive_name = f"{normalized_name(name)}-{version}-{len(licenses)}-{relative.name}"
                archive_path = archive_dir / archive_name
                write_no_replace(archive_path, payload)
                licenses.append({
                    "path": relative.as_posix(),
                    "archive_path": archive_path.name,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                })
            if native_name(raw) or is_elf(resolved):
                native.append(native_fact(resolved, relative))
        except (AuditError, OSError, RuntimeError, ValueError) as error:
            if candidate:
                inventory_errors.append(f"{raw}: {error}")
    publisher_license = distribution.metadata.get("License") or ""
    classifiers = sorted(distribution.metadata.get_all("Classifier") or [])
    fact: dict[str, Any] = {
        "name": name,
        "version": version,
        "source": row["source"],
        "lock_row_sha256": digest(row),
        "publisher_license": publisher_license,
        "license_classifiers": classifiers,
        "license_flags": license_flags(publisher_license, classifiers, licenses),
        "license_files": licenses,
        "license_files_sha256": digest(licenses),
        "native_files": native,
        "native_files_sha256": digest(native),
        "file_inventory_errors": sorted(inventory_errors),
        "license_review": OWNER_REVIEW,
        "native_bundled_review": OWNER_REVIEW,
    }
    fact["fact_sha256"] = digest(fact)
    return fact


def git_identity(repo_root: Path, expected_head: str) -> dict[str, Any]:
    if not HEX40.fullmatch(expected_head):
        raise AuditError("expected HEAD must be 40 lowercase hexadecimal characters")
    regular_directory(repo_root, "repository root")
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False, timeout=30)
        if result.returncode != 0:
            raise AuditError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()
    top = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if top != repo_root.resolve(strict=True) or head != expected_head or dirty:
        raise AuditError(f"clean exact HEAD required: expected={expected_head} actual={head} dirty={bool(dirty)}")
    return {"expected_head": expected_head, "head": head, "clean": True}


def memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def verify_project(project_path: Path, lock_path: Path) -> tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    project_bytes, project = load_toml(project_path, "dedicated pyproject")
    lock_bytes, lock = load_toml(lock_path, "dedicated uv.lock")
    if project.get("project", {}).get("name") != "vokra-canary-1b-reference":
        raise AuditError("wrong dedicated project name")
    if project.get("project", {}).get("requires-python") != "==3.12.*":
        raise AuditError("dedicated project is not Python 3.12-only")
    tool_uv = project.get("tool", {}).get("uv", {})
    if tool_uv.get("environments") != ["sys_platform == 'linux' and platform_machine == 'x86_64'"]:
        raise AuditError("dedicated project environment is not Linux x86_64-only")
    sources = tool_uv.get("sources", {})
    if sources != {"torch": {"index": "pytorch-cpu"}}:
        raise AuditError("Torch CPU source binding is missing")
    indexes = tool_uv.get("index", [])
    if indexes != [{"name": "pytorch-cpu", "url": PYTORCH_CPU, "explicit": True}]:
        raise AuditError("PyTorch CPU index is not explicit and pinned")
    dependencies = project.get("project", {}).get("dependencies", [])
    if sorted(dependencies) != ["nemo-toolkit[asr]==3.0.0", "torch==2.7.1"]:
        raise AuditError("dedicated dependency contract drifted")
    rows, inactive, failures, dependency_paths = active_lock_rows(lock)
    if sha256_bytes(lock_bytes) == sha256_bytes(project_bytes):
        raise AuditError("project and lock unexpectedly share digest")
    return project_bytes, lock_bytes, project, rows, inactive, failures, dependency_paths


def audit(project_path: Path, lock_path: Path, repo_root: Path, expected_head: str, output: Path, archive_dir: Path, wrapper_path: Path, expected_project_sha256: str | None, expected_lock_sha256: str | None, expected_audit_sha256: str | None, expected_wrapper_sha256: str | None, write_sums: bool = True) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise AuditError("Canary dependency audit requires Linux x86_64 VAST")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise AuditError("VOKRA_PUBLISH_ON_VAST=1 is required")
    available = memory_bytes()
    if available is None or available < MIN_MEMORY_BYTES:
        raise AuditError(f"VAST host RAM is below {MIN_MEMORY_BYTES} bytes")
    absent_output(output)
    regular_directory(archive_dir, "license archive directory")
    if any(archive_dir.iterdir()):
        raise AuditError("license archive directory must be empty")
    verify_canonical_sources(repo_root, project_path, lock_path, wrapper_path)
    project_bytes, lock_bytes, _project, rows, inactive, collector_failures, dependency_paths = verify_project(project_path, lock_path)
    project_sha256 = sha256_bytes(project_bytes)
    lock_sha256 = sha256_bytes(lock_bytes)
    audit_sha256 = sha256_file(Path(__file__).resolve())
    wrapper_source = regular_file(wrapper_path, "executed audit wrapper")
    wrapper_sha256 = sha256_file(wrapper_source)
    for label, actual, expected in (("project", project_sha256, expected_project_sha256), ("lock", lock_sha256, expected_lock_sha256), ("audit", audit_sha256, expected_audit_sha256), ("wrapper", wrapper_sha256, expected_wrapper_sha256)):
        if expected is not None and (not HEX64.fullmatch(expected) or expected != actual):
            raise AuditError(f"{label} SHA-256 binding mismatch")
    git = git_identity(repo_root, expected_head)
    inventory = installed_inventory(rows)
    facts = [distribution_fact(row, archive_dir) for row in rows if row.get("source") != {"virtual": "."}]
    archive_files = sorted(
        {path.name: {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in archive_dir.iterdir() if path.is_file()}.values(),
        key=lambda item: item["path"],
    )
    package_scope = {"active_rows": rows, "active_facts": facts, "dependency_paths": dependency_paths, "inactive_rows": inactive, "collector_failures": collector_failures, "project_sha256": project_sha256, "lock_sha256": lock_sha256, "audit_script_sha256": audit_sha256, "wrapper_sha256": wrapper_sha256, "model_free": MODEL_FREE_FIELDS}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "publication": PUBLICATION,
        "owner_review": OWNER_REVIEW,
        "project_sha256": project_sha256,
        "lock_sha256": lock_sha256,
        "audit_script_sha256": audit_sha256,
        "wrapper_sha256": wrapper_sha256,
        **git,
        "project_environment": "linux-x86_64-python3.12",
        "memory_bytes": available,
        "distribution_inventory": inventory,
        "package_rows": rows,
        "package_rows_sha256": digest(rows),
        "dependency_paths": dependency_paths,
        "dependency_paths_sha256": digest(dependency_paths),
        "inactive_lock_rows": inactive,
        "collector_failures": collector_failures,
        "package_facts": facts,
        "package_facts_sha256": digest(facts),
        "license_archive": archive_files,
        "license_archive_sha256": digest(archive_files),
        "candidate_owner_scope_sha256": digest(package_scope),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            **MODEL_FREE_FIELDS,
        },
        "blockers": [
            "all package publisher license metadata and license-file bytes require owner review",
            "all bundled/native payloads and ELF NEEDED entries require owner review",
            "any GPL/LGPL/AGPL/unknown/native rows remain fail-closed",
            "no model/source/checkpoint was acquired; runtime/parity remain locked",
            *collector_failures,
        ],
    }
    validate_report_semantics(report)
    snapshots = {
        "pyproject.toml": project_bytes,
        "uv.lock": lock_bytes,
        "dependency_audit.py": Path(__file__).read_bytes(),
        "audit-canary-1b-dependencies.sh": wrapper_source.read_bytes(),
    }
    for name, payload in snapshots.items():
        write_no_replace(output.parent / name, payload)
    write_no_replace(output, (canonical(report) + "\n").encode("utf-8"))
    if write_sums:
        sums = output.parent / "SHA256SUMS"
        sum_lines = [
            f"{sha256_file(output.parent / 'pyproject.toml')}  pyproject.toml\n",
            f"{sha256_file(output.parent / 'uv.lock')}  uv.lock\n",
            f"{sha256_file(output.parent / 'dependency_audit.py')}  dependency_audit.py\n",
            f"{sha256_file(output.parent / 'audit-canary-1b-dependencies.sh')}  audit-canary-1b-dependencies.sh\n",
            f"{sha256_file(output)}  {output.name}\n",
        ]
        archive_root = archive_dir.relative_to(output.parent)
        for path in sorted(archive_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.is_symlink():
                raise AuditError(f"license archive contains an unexpected entry: {path}")
            sum_lines.append(f"{sha256_file(path)}  {(archive_root / path.name).as_posix()}\n")
        write_no_replace(sums, "".join(sum_lines).encode("utf-8"))


def validate_report_semantics(report: dict[str, Any]) -> None:
    """Reject a plausible-looking report that is not fail-closed evidence."""
    if report.get("schema") != SCHEMA or report.get("status") != STATUS or report.get("publication") != PUBLICATION:
        raise AuditError("report is not permanently blocked/no-upload")
    if report.get("clean") is not True or report.get("expected_head") != report.get("head"):
        raise AuditError("report does not bind a clean exact HEAD")
    for field in ("project_sha256", "lock_sha256", "audit_script_sha256", "wrapper_sha256", "package_rows_sha256", "dependency_paths_sha256", "package_facts_sha256", "license_archive_sha256", "candidate_owner_scope_sha256"):
        if not isinstance(report.get(field), str) or not HEX64.fullmatch(report[field]):
            raise AuditError(f"report has missing/null digest: {field}")
    environment = report.get("environment")
    if not isinstance(environment, dict) or any(environment.get(key) is not False for key in ("weights_acquired", "source_acquired", "model_imported", "model_executed", "cargo_invoked")) or environment.get("upload") != PUBLICATION:
        raise AuditError("report model-free/no-upload contract drifted")
    inventory = report.get("distribution_inventory")
    if not isinstance(inventory, dict) or inventory.get("exact") is not True:
        raise AuditError("installed active closure is not exact")
    if not isinstance(report.get("collector_failures"), list) or not isinstance(report.get("inactive_lock_rows"), list):
        raise AuditError("active/inactive closure evidence is incomplete")
    paths = report.get("dependency_paths")
    if not isinstance(paths, list) or any(not isinstance(row, dict) or not row.get("paths") for row in paths):
        raise AuditError("dependency closure paths are incomplete")


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "BLOCKED_UNREVIEWED_TRANSITIVE" in source
    assert '"NO_UPLOAD"' in source
    assert "import " + "nemo" not in source
    assert "huggingface_" + "hub" not in source
    assert "cargo" in source
    assert native_name("x.so") and native_name("x.so.1") and native_name("x.a")
    assert license_name("LICENSE") and license_name("NOTICE.txt")
    assert not license_name("not-a-license.txt")
    assert license_flags("LGPL-3.0", [], []) == ["LGPL"]
    assert license_flags("", [], []) == ["UNKNOWN"]
    assert normalized_name("Demo_pkg-1") == "demo-pkg-1"
    assert marker_active(f"sys_platform == '{sys.platform}'")
    linux_context = {
        "sys_platform": "linux", "platform_machine": "x86_64", "platform_system": "Linux",
        "python_version": "3.12", "python_full_version": "3.12.10", "implementation_name": "cpython",
    }
    assert marker_active("sys_platform == 'linux'", context=linux_context)
    assert marker_active("platform_machine == 'x86_64'", context=linux_context)
    assert not marker_active("sys_platform == 'darwin'", context=linux_context)
    assert not marker_active("platform_machine == 'aarch64'", context=linux_context)
    assert marker_active("python_full_version > '3.12.9'", context=linux_context)
    assert not marker_active("python_full_version < '3.12.9'", context=linux_context)
    assert marker_active("python_version == '3.12.0'", context=linux_context)
    assert marker_active("python_full_version in '3.12.10'", context=linux_context)
    assert marker_active("python_full_version not in '3.12.9'", context=linux_context)
    try:
        marker_active("unsupported_marker == 'value'", context=linux_context)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted an unsupported marker")
    canonical_root = Path(__file__).resolve().parents[3]
    verify_canonical_sources(
        canonical_root,
        canonical_root / "tools/parity/canary_1b_reference/pyproject.toml",
        canonical_root / "tools/parity/canary_1b_reference/uv.lock",
        canonical_root / "scripts/publish/vast-ai/audit-canary-1b-dependencies.sh",
    )
    try:
        verify_canonical_sources(
            canonical_root,
            canonical_root / "tools/parity/canary_1b_reference/uv.lock",
            canonical_root / "tools/parity/canary_1b_reference/pyproject.toml",
            canonical_root / "scripts/publish/vast-ai/audit-canary-1b-dependencies.sh",
        )
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted swapped external audit inputs")
    synthetic_lock = {
        "version": 1,
        "revision": 3,
        "requires-python": "==3.12.*",
        "package": [
            {
                "name": "root", "version": "0", "source": {"virtual": "."},
                "metadata": {"requires-dist": [{"name": "demo", "specifier": "==1"}]},
            },
            {"name": "demo", "version": "1", "source": {"registry": PYPI}, "dependencies": [{"name": "extra-target", "extra": ["http"]}]},
            {"name": "extra-target", "version": "1", "source": {"registry": PYPI}, "optional-dependencies": {"http": [{"name": "http-child"}]}},
            {"name": "http-child", "version": "1", "source": {"registry": PYPI}},
            {"name": "darwin-only", "version": "1", "source": {"registry": PYPI}, "resolution-markers": ["sys_platform == 'darwin'"]},
        ],
    }
    active, inactive, failures, paths = active_lock_rows(synthetic_lock)
    assert [row["name"] for row in active] == ["root", "demo", "extra-target", "http-child"]
    assert [row["name"] for row in inactive] == ["darwin-only"]
    assert failures == []
    assert paths[1]["paths"] == [["project", "demo"]]
    try:
        validate_report_semantics({"schema": SCHEMA, "status": STATUS, "publication": PUBLICATION})
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted a report with null/missing digests")
    for bad in ("", "/absolute", "../escape", "a/../b"):
        try:
            safe_relative(bad)
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted unsafe path: {bad}")
    valid = "a" * 40
    assert HEX40.fullmatch(valid)
    assert digest({"x": 1}) == digest({"x": 1})
    temp_root = Path("/private/tmp")
    if not temp_root.is_dir() or temp_root.is_symlink():
        temp_root = Path("/tmp")
    with tempfile.TemporaryDirectory(prefix="vokra-canary-audit-", dir=temp_root) as directory:
        root = Path(directory)
        target = root / "evidence.json"
        write_no_replace(target, b"first\n")
        try:
            write_no_replace(target, b"second\n")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test replaced an existing output")
        linked = root / "linked"
        linked.symlink_to(root, target_is_directory=True)
        try:
            regular_directory(linked, "linked")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted symlinked directory")
    print("canary_1b_reference dependency audit self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--wrapper", type=Path)
    parser.add_argument("--project-sha256")
    parser.add_argument("--lock-sha256")
    parser.add_argument("--audit-sha256")
    parser.add_argument("--wrapper-sha256")
    parser.add_argument("--defer-sums", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.defer_sums or any(value is not None for value in (args.project, args.lock, args.repo_root, args.expected_head, args.output, args.archive_dir, args.wrapper, args.project_sha256, args.lock_sha256, args.audit_sha256, args.wrapper_sha256)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.project, args.lock, args.repo_root, args.expected_head, args.output, args.archive_dir, args.wrapper, args.project_sha256, args.lock_sha256, args.audit_sha256, args.wrapper_sha256)):
        parser.error("normal runs require project, lock, repo-root, expected-head, output, archive-dir, wrapper, and four SHA-256 bindings")
    try:
        audit(args.project, args.lock, args.repo_root, args.expected_head, args.output, args.archive_dir, args.wrapper, args.project_sha256, args.lock_sha256, args.audit_sha256, args.wrapper_sha256, write_sums=not args.defer_sums)
    except (AuditError, OSError, ValueError) as error:
        print(f"canary_1b_reference dependency audit: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"canary_1b_reference dependency audit: BLOCKED pending owner review; evidence={args.output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
