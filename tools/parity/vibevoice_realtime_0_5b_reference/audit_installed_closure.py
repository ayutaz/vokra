#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python
"""Audit the locked, installed official VibeVoice reference closure.

This is evidence collection only.  It never imports the reference model,
downloads a checkpoint, or infers a license from package metadata.  The
result is deliberately left at OWNER_REVIEW_REQUIRED/NO_UPLOAD until a
primary-source review and owner sign-off are recorded elsewhere.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


FORMAT = "vokra-vibevoice-realtime-installed-closure-evidence-v1"
STATUS = "OWNER_REVIEW_REQUIRED"
EXECUTION = {
    "model_download": "NO_MODEL_DOWNLOAD",
    "model_execution": "NO_MODEL_EXECUTION",
    "publication": "NO_UPLOAD",
}
PROJECT_NAME = "vokra-vibevoice-realtime-0-5b-reference"
PYTHON_REQUIREMENT = "==3.12.*"
OFFICIAL_SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice.git"
OFFICIAL_SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
REGISTRY_HOSTS = ("https://pypi.org/simple", "https://download.pytorch.org/whl")
LICENSE_BASENAMES = {
    "license",
    "license.txt",
    "license.md",
    "copying",
    "copying.txt",
    "notice",
    "notice.txt",
}
METADATA_BASENAMES = {
    "metadata",
    "wheel",
    "record",
    "direct_url.json",
    "entry_points.txt",
}
NATIVE_SUFFIXES = (".so", ".dylib", ".pyd")
DEFAULT_MAX_NATIVE_HASH_BYTES = 512 * 1024 * 1024


class AuditError(ValueError):
    """The installed closure cannot be authenticated."""


def fail(message: str) -> None:
    raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_name(name: str) -> str:
    """Apply PEP 503 distribution-name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def require_real_directory_chain(path: Path, label: str) -> None:
    if not path.is_absolute():
        fail(f"{label} must be absolute")
    current = path
    while True:
        if current.is_symlink():
            fail(f"{label} traverses a symlink: {current}")
        if not current.is_dir():
            fail(f"{label} must be an existing directory: {current}")
        if current == current.parent:
            break
        current = current.parent


def require_parent(path: Path) -> None:
    if not path.is_absolute() or path.is_symlink():
        fail("evidence output must have an absolute non-symlink path")
    require_real_directory_chain(path.parent, "evidence output parent")


def absolute_without_resolving(path: Path) -> Path:
    """Make CLI paths absolute while preserving symlinks for fail-closed checks."""
    return path if path.is_absolute() else Path.cwd() / path


def read_toml(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked TOML: {path}")
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        fail(f"invalid TOML {path}: {error}")


def file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or symlinked identity file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_project(document: dict[str, Any]) -> None:
    project = document.get("project")
    if not isinstance(project, dict) or project.get("name") != PROJECT_NAME:
        fail("reference project name mismatch")
    if project.get("requires-python") != PYTHON_REQUIREMENT:
        fail("reference project is not pinned to Python 3.12")
    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies or any(not isinstance(dep, str) for dep in dependencies):
        fail("reference project dependencies are malformed")
    tool_uv = document.get("tool", {}).get("uv")
    if not isinstance(tool_uv, dict) or tool_uv.get("package") is not False:
        fail("reference project must remain non-package")
    environments = tool_uv.get("environments")
    if not isinstance(environments, list) or "sys_platform == 'linux' and platform_machine == 'x86_64'" not in environments:
        fail("reference project must retain a Linux x86_64 environment")


def validate_lock(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if document.get("version") != 1 or document.get("revision") != 3:
        fail("uv.lock schema/version drifted")
    if document.get("requires-python") != PYTHON_REQUIREMENT:
        fail("uv.lock Python requirement drifted")
    rows = document.get("package")
    if not isinstance(rows, list) or not rows:
        fail("uv.lock has no package rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            fail("malformed uv.lock package row")
        name = row["name"]
        normalized = normalize_name(name)
        if normalized in result:
            fail(f"duplicate uv.lock package row: {name}")
        source = row.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            fail(f"malformed uv.lock source for {name}")
        if "registry" in source:
            registry = source["registry"]
            if not isinstance(registry, str) or not registry.startswith(REGISTRY_HOSTS):
                fail(f"unapproved registry source for {name}: {registry!r}")
            for key in ("sdist", "wheels"):
                artifacts = row.get(key, [])
                if isinstance(artifacts, dict):
                    artifacts = [artifacts]
                if not isinstance(artifacts, list):
                    fail(f"malformed {key} artifacts for {name}")
                for artifact in artifacts:
                    if not isinstance(artifact, dict) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("hash", ""))):
                        fail(f"artifact hash missing for {name}")
                    if not isinstance(artifact.get("url"), str) or not artifact["url"].startswith(("https://files.pythonhosted.org/", "https://download-r2.pytorch.org/")):
                        fail(f"artifact URL host is not approved for {name}")
        result[normalized] = row
    virtual_rows = [row for row in result.values() if "virtual" in row["source"]]
    if len(virtual_rows) != 1 or virtual_rows[0]["source"].get("virtual") != ".":
        fail("uv.lock must contain exactly one repository-root virtual project row")
    return result


def write_all(stream: Any, body: bytes) -> None:
    offset = 0
    while offset < len(body):
        written = stream.write(body[offset:])
        remaining = len(body) - offset
        if type(written) is not int or written <= 0 or written > remaining:
            fail("short or invalid evidence write")
        offset += written


def write_no_replace(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    require_parent(path)
    if path.exists() or path.is_symlink():
        fail("evidence output must be absent")
    body = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        for attempt in range(100):
            candidate = path.parent / f".{path.name}.vibevoice-audit-{os.getpid()}-{attempt}"
            try:
                with candidate.open("xb") as stream:
                    write_all(stream, body)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary = candidate
                break
            except FileExistsError:
                continue
        if temporary is None:
            fail("could not allocate evidence temporary")
        os.link(temporary, path)
        temporary.unlink(missing_ok=True)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return {"path": str(path), "bytes": len(body), "sha256": sha256_bytes(body)}


def _safe_installed_path(dist: metadata.Distribution, relative: Any, root: Path, label: str) -> Path:
    relative_text = Path(relative).as_posix()
    raw = Path(dist.locate_file(relative))
    if not raw.is_absolute():
        fail(f"{label} path is not absolute: {relative_text}")
    current = raw
    while True:
        if current.is_symlink():
            fail(f"{label} path traverses a symlink: {relative_text}")
        if current == current.parent:
            break
        current = current.parent
    absolute = raw.resolve(strict=False)
    root_resolved = root.resolve()
    if not absolute.is_relative_to(root_resolved) or not absolute.is_file() or absolute.is_symlink():
        fail(f"{label} path escapes the environment or is not a regular file: {relative_text}")
    return absolute


def _file_record(relative: Any, absolute: Path, *, hash_limit: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": Path(relative).as_posix(), "bytes": absolute.stat().st_size}
    if hash_limit is not None and record["bytes"] > hash_limit:
        record["hash_status"] = "SKIPPED_TOO_LARGE"
    else:
        record["sha256"] = sha256_file(absolute)
        record["hash_status"] = "SHA256"
    return record


def _is_license(relative: Any, license_headers: set[str]) -> bool:
    path = Path(relative)
    text = path.as_posix().casefold()
    return text in {header.casefold() for header in license_headers} or path.name.casefold() in LICENSE_BASENAMES or "/licenses/" in f"/{text}/"


def _is_native(relative: Any) -> bool:
    name = Path(relative).name.casefold()
    return name.endswith(NATIVE_SUFFIXES) or ".so." in name


def distribution_metadata(dist: metadata.Distribution, root: Path, max_native_hash_bytes: int) -> dict[str, Any]:
    name = dist.metadata.get("Name")
    version = dist.version
    if not name or not version:
        fail("installed distribution has no Name/Version")
    files = list(dist.files or [])
    license_headers = {str(value).replace("\\", "/") for value in dist.metadata.get_all("License-File", failobj=[])}
    license_files: list[dict[str, Any]] = []
    metadata_files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    for relative in files:
        path = Path(relative)
        lower = path.name.casefold()
        is_metadata = path.parent.name.casefold().endswith(".dist-info") and (lower in METADATA_BASENAMES or lower == "top_level.txt")
        is_license = _is_license(relative, license_headers)
        is_native = _is_native(relative)
        if not (is_metadata or is_license or is_native):
            continue
        absolute = _safe_installed_path(dist, relative, root, name)
        if is_metadata:
            metadata_files.append(_file_record(relative, absolute))
        if is_license:
            license_files.append(_file_record(relative, absolute))
        if is_native:
            native_files.append(_file_record(relative, absolute, hash_limit=max_native_hash_bytes))
    return {
        "name": name,
        "version": version,
        "metadata": {
            "license_expression": dist.metadata.get("License-Expression"),
            "license": dist.metadata.get("License"),
            "classifiers": sorted(dist.metadata.get_all("Classifier", failobj=[])),
            "license_file_headers": sorted(license_headers),
        },
        "metadata_files": sorted(metadata_files, key=lambda row: row["path"]),
        "license_files": sorted(license_files, key=lambda row: row["path"]),
        "license_file_status": "PRESENT" if license_files else "MISSING_IN_DISTRIBUTION",
        "native_payloads": sorted(native_files, key=lambda row: row["path"]),
        "native_payload_status": "HASHED_OR_SIZE_ONLY" if native_files else "NONE_OBSERVED",
    }


def _run_git(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", "-C", str(source_root), *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"source git command failed ({' '.join(args)}): {error}")
    return result.stdout.strip()


def source_identity(source_root: Path) -> dict[str, Any]:
    require_real_directory_chain(source_root, "official source checkout")
    revision = _run_git(source_root, "rev-parse", "--verify", "HEAD")
    if revision != OFFICIAL_SOURCE_REVISION:
        fail(f"official source revision mismatch: expected {OFFICIAL_SOURCE_REVISION}, got {revision}")
    status = _run_git(source_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        fail("official source checkout is dirty")
    remote = _run_git(source_root, "remote", "get-url", "origin")
    accepted = {OFFICIAL_SOURCE_REPOSITORY, OFFICIAL_SOURCE_REPOSITORY.removesuffix(".git")}
    if remote not in accepted:
        fail(f"official source origin mismatch: expected {sorted(accepted)}, got {remote}")
    return {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "origin": remote,
        "revision": revision,
        "working_tree": "CLEAN",
        "root": str(source_root),
    }


def _uv_version() -> str:
    try:
        return subprocess.check_output(["uv", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"uv version could not be recorded: {error}")


def versions_match(normalized_name: str, installed_version: str, locked_version: str) -> bool:
    """Require the installed distribution version to equal the lock exactly."""
    return installed_version == locked_version


def audit(project: Path, lock: Path, source_root: Path, output: Path, max_native_hash_bytes: int) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        fail("installed closure audit requires Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        fail(f"installed closure audit requires Python 3.12, got {platform.python_version()}")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        fail("VOKRA_PUBLISH_ON_VAST=1 is required")
    require_real_directory_chain(Path(sys.prefix), "Python environment")
    project_document = read_toml(project)
    validate_project(project_document)
    locked = validate_lock(read_toml(lock))
    source = source_identity(source_root)
    registry_rows = {key: row for key, row in locked.items() if "registry" in row["source"]}
    virtual_rows = [row for row in locked.values() if "virtual" in row["source"]]
    distributions: dict[str, metadata.Distribution] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            normalized = normalize_name(name)
            if normalized in distributions:
                fail(f"duplicate installed distribution: {name}")
            distributions[normalized] = dist
    expected_names = set(registry_rows)
    if set(distributions) != expected_names:
        fail(f"installed closure differs from uv.lock: extra={sorted(set(distributions) - expected_names)}, missing={sorted(expected_names - set(distributions))}")
    packages: list[dict[str, Any]] = []
    for normalized in sorted(expected_names):
        dist = distributions[normalized]
        lock_row = registry_rows[normalized]
        if not versions_match(normalized, dist.version, lock_row["version"]):
            fail(f"installed {dist.metadata['Name']} version does not match uv.lock")
        packages.append({
            "lock": {"name": lock_row["name"], "version": lock_row["version"], "source": lock_row["source"]},
            "installed": distribution_metadata(dist, Path(sys.prefix), max_native_hash_bytes),
            "license_status": "PENDING_PRIMARY_SOURCE_REVIEW",
        })
    missing = [row["installed"]["name"] for row in packages if row["installed"]["license_file_status"] == "MISSING_IN_DISTRIBUTION"]
    document = {
        "format": FORMAT,
        "status": STATUS,
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "uv": _uv_version(), "sys_prefix": sys.prefix},
        "project": file_identity(project),
        "uv_lock": file_identity(lock),
        "official_source": source,
        "lock_rows": {
            "registry": sorted(({"name": row["name"], "version": row["version"], "source": row["source"]} for row in registry_rows.values()), key=lambda row: normalize_name(row["name"])),
            "virtual_project": {"name": virtual_rows[0]["name"], "version": virtual_rows[0]["version"], "source": virtual_rows[0]["source"], "status": "LOCAL_PROJECT_NO_INSTALLED_DISTRIBUTION"},
        },
        "packages": packages,
        "native_hash_policy": {"max_bytes": max_native_hash_bytes, "oversize": "SIZE_ONLY_NO_SHA256"},
        "execution": EXECUTION,
        "license_evidence_summary": {"missing_license_distribution_count": len(missing), "missing_license_distributions": sorted(missing)},
        "blockers": [
            "Primary-source package license, native-payload, and owner review is unresolved; this evidence does not authorize model execution or publication.",
            *([f"Installed distributions without discoverable license files require upstream review: {', '.join(sorted(missing))}."] if missing else []),
        ],
    }
    write_no_replace(output, document)
    return document


def self_test() -> None:
    assert STATUS == "OWNER_REVIEW_REQUIRED"
    assert EXECUTION["publication"] == "NO_UPLOAD"
    assert normalize_name("Example_Package.Name") == "example-package-name"
    assert versions_match("torch", "2.7.1", "2.7.1")
    assert not versions_match("torch", "2.7.1+cpu", "2.7.1")
    assert _is_license("foo.dist-info/licenses/LICENSE.txt", set())
    assert _is_native("torch/lib/libtorch_cpu.so.1")
    class EmptyHeaders(dict[str, str]):
        def get_all(self, key: str, failobj: Any = None) -> list[str]:
            return []

    class EmptyDistribution:
        metadata = EmptyHeaders(Name="example-package")
        version = "1.0.0"
        files: list[Any] = []

    row = distribution_metadata(EmptyDistribution(), Path(tempfile.gettempdir()).resolve(), DEFAULT_MAX_NATIVE_HASH_BYTES)
    assert row["license_file_status"] == "MISSING_IN_DISTRIBUTION"
    assert row["native_payload_status"] == "NONE_OBSERVED"
    registry = validate_lock({
        "version": 1,
        "revision": 3,
        "requires-python": PYTHON_REQUIREMENT,
        "package": [
            {"name": PROJECT_NAME, "version": "0.1.0", "source": {"virtual": "."}},
            {"name": "demo", "version": "1.0.0", "source": {"registry": "https://pypi.org/simple"}, "wheels": [{"url": "https://files.pythonhosted.org/packages/demo.whl", "hash": "sha256:" + "0" * 64}]},
        ],
    })
    assert set(registry) == {PROJECT_NAME, "demo"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--source-root", type=Path, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-native-hash-bytes", type=int, default=DEFAULT_MAX_NATIVE_HASH_BYTES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test: PASS")
        return 0
    if args.source_root is None or args.output is None:
        parser.error("--source-root and --output are required unless --self-test is used")
    if args.max_native_hash_bytes <= 0:
        parser.error("--max-native-hash-bytes must be positive")
    try:
        document = audit(
            absolute_without_resolving(args.project),
            absolute_without_resolving(args.lock),
            absolute_without_resolving(args.source_root),
            absolute_without_resolving(args.output),
            args.max_native_hash_bytes,
        )
    except AuditError as error:
        print(f"audit: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": document["status"], "publication": document["execution"]["publication"], "output": str(absolute_without_resolving(args.output))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
