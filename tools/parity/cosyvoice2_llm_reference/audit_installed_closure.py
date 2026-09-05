#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_llm_reference python
"""Audit the installed, locked LLM reference closure without importing it."""

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preflight_gate as pinned  # noqa: E402

FORMAT = "vokra-cosyvoice2-llm-installed-closure-evidence-v1"
STATUS = "OWNER_REVIEW_REQUIRED"
EXECUTION = {"model_download": "NO_MODEL_DOWNLOAD", "model_execution": "NO_MODEL_EXECUTION", "publication": "NO_UPLOAD"}
LICENSE_BASENAMES = {"license", "license.txt", "copying", "copying.txt", "notice", "notice.txt"}
NATIVE_SUFFIXES = (".pyd", ".dylib")


class AuditError(ValueError):
    """Installed closure evidence cannot be authenticated."""


def fail(message: str) -> None:
    raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    """Apply the PEP 503 distribution-name normalization."""
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
            candidate = path.parent / f".{path.name}.cosyvoice2-llm-audit-{os.getpid()}-{attempt}"
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
        try:
            temporary.unlink()
        except OSError:
            pass
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
            try:
                temporary.unlink()
            except OSError:
                pass
        raise
    return {"path": str(path), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def classify_lock_rows(locked: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    registry = {normalize_name(name): row for name, row in locked.items() if "registry" in row["source"]}
    virtual = [row for row in locked.values() if "virtual" in row["source"]]
    if len(virtual) != 1 or virtual[0]["source"].get("virtual") != ".":
        fail("uv.lock must contain exactly one repository-root virtual project row")
    if len(registry) != len([row for row in locked.values() if "registry" in row["source"]]):
        fail("uv.lock registry names are ambiguous after PEP 503 normalization")
    return registry, virtual[0]


def distribution_metadata(dist: metadata.Distribution, root: Path) -> dict[str, Any]:
    name = dist.metadata.get("Name")
    version = dist.version
    if not name or not version:
        fail("installed distribution has no Name/Version")
    files = list(dist.files or [])
    license_files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    license_headers = dist.metadata.get_all("License-File", failobj=[])
    license_names = {str(value).replace("\\", "/") for value in license_headers}
    for relative in files:
        relative_text = relative.as_posix()
        base = relative.name.lower()
        is_license = relative_text in license_names or base in LICENSE_BASENAMES or "/licenses/" in f"/{relative_text.lower()}/"
        lower_name = relative.name.casefold()
        is_native = lower_name.endswith(".so") or lower_name.endswith(NATIVE_SUFFIXES) or ".so." in lower_name
        if not (is_license or is_native):
            continue
        try:
            raw_path = Path(dist.locate_file(relative))
            if not raw_path.is_absolute():
                fail(f"installed file path is not absolute: {name}/{relative_text}")
            current = raw_path
            while True:
                if current.is_symlink():
                    fail(f"installed file path contains a symlink: {name}/{relative_text}")
                if current == current.parent:
                    break
                current = current.parent
            absolute = raw_path.resolve(strict=False)
        except OSError as error:
            fail(f"cannot resolve installed file {name}/{relative_text}: {error}")
        if not absolute.is_relative_to(root.resolve()):
            fail(f"installed file escapes the environment: {name}/{relative_text}")
        if not absolute.is_file() or absolute.is_symlink():
            fail(f"installed file is not a regular in-environment file: {name}/{relative_text}")
        record = {"path": relative_text, "bytes": absolute.stat().st_size, "sha256": sha256_file(absolute)}
        if is_license:
            license_files.append(record)
        if is_native:
            native_files.append(record)
    if not license_files:
        fail(f"installed distribution has no discoverable license file: {name}")
    return {
        "name": name,
        "version": version,
        "metadata": {
            "license_expression": dist.metadata.get("License-Expression"),
            "license": dist.metadata.get("License"),
            "classifiers": sorted(dist.metadata.get_all("Classifier", failobj=[])),
            "license_files": sorted(license_names),
        },
        "license_files": sorted(license_files, key=lambda row: row["path"]),
        "native_payloads": sorted(native_files, key=lambda row: row["path"]),
    }


def audit(project: Path, lock: Path, license_manifest: Path, output: Path) -> dict[str, Any]:
    project_document = pinned.read_toml(project)
    pinned.validate_project(project_document)
    lock_document = pinned.read_toml(lock)
    locked = pinned.validate_lock(lock_document)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        fail("installed closure audit requires Linux x86_64")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        fail("VOKRA_PUBLISH_ON_VAST=1 is required")
    require_real_directory_chain(Path(sys.prefix), "Python environment")
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"uv version could not be recorded: {error}")
    registry_rows, virtual_row = classify_lock_rows(locked)
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
    package_rows = []
    for normalized in sorted(expected_names):
        dist = distributions[normalized]
        lock_row = registry_rows[normalized]
        if dist.version != lock_row["version"] and not (normalized == "torch" and dist.version == "2.7.1+cpu" and lock_row["version"] == "2.7.1"):
            fail(f"installed {dist.metadata['Name']} version does not match lock")
        package_rows.append({"lock": {"name": lock_row["name"], "version": lock_row["version"]}, "installed": distribution_metadata(dist, Path(sys.prefix)), "license_status": "PENDING_PRIMARY_SOURCE_REVIEW"})
    manifest = pinned.validate_license_manifest(license_manifest)
    document = {
        "format": FORMAT,
        "status": STATUS,
        "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "uv": uv_version},
        "project": {"path": str(project), "sha256": sha256_file(project)},
        "uv_lock": {"path": str(lock), "sha256": sha256_file(lock)},
        "license_gate": {"path": str(license_manifest), "sha256": sha256_file(license_manifest), "status": manifest["status"], "owner_signoff": manifest["owner_signoff"]},
        "lock_rows": {
            "registry": sorted(
                ({"name": row["name"], "version": row["version"], "source": row["source"]} for row in registry_rows.values()),
                key=lambda row: normalize_name(row["name"]),
            ),
            "virtual_project": {
                "name": virtual_row["name"],
                "version": virtual_row["version"],
                "source": virtual_row["source"],
                "status": "LOCAL_PROJECT_NO_INSTALLED_DISTRIBUTION",
            },
        },
        "license_gate_coverage": {
            "registry_rows_require_package_review": sorted(expected_names),
            "virtual_project_excluded_from_installed_distribution_match": True,
        },
        "packages": package_rows,
        "execution": EXECUTION,
        "blockers": ["Primary-source license and native-payload review is unresolved; this evidence does not authorize execution or publication."],
    }
    write_no_replace(output, document)
    return document


def self_test() -> None:
    assert STATUS == "OWNER_REVIEW_REQUIRED" and EXECUTION["publication"] == "NO_UPLOAD"
    assert normalize_name("Example_Package.Name") == "example-package-name"
    registry, virtual = classify_lock_rows(
        {
            "demo_project": {"name": "demo_project", "version": "0", "source": {"virtual": "."}},
            "Example_Package": {"name": "Example_Package", "version": "1", "source": {"registry": "https://pypi.org/simple"}},
        }
    )
    assert set(registry) == {"example-package"} and virtual["source"] == {"virtual": "."}
    try:
        classify_lock_rows(
            {
                "demo": {"name": "demo", "version": "0", "source": {"virtual": "."}},
                "example_package": {"name": "example_package", "version": "1", "source": {"registry": "https://pypi.org/simple"}},
                "example-package": {"name": "example-package", "version": "1", "source": {"registry": "https://pypi.org/simple"}},
            }
        )
    except AuditError:
        pass
    else:
        raise AssertionError("ambiguous PEP 503 lock names accepted")
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-llm-audit-", dir=Path(tempfile.gettempdir()).resolve()) as temp:
        root = Path(temp)
        output = root / "evidence.json"
        result = write_no_replace(output, {"format": FORMAT, "status": STATUS, "execution": EXECUTION})
        assert result["bytes"] == output.stat().st_size and len(result["sha256"]) == 64
        try:
            write_no_replace(output, {})
        except AuditError:
            pass
        else:
            raise AssertionError("existing evidence was overwritten")
        malformed = root / "malformed.toml"
        malformed.write_text("[project]\nname='wrong'\n", encoding="utf-8")
        try:
            pinned.validate_project(tomllib.loads(malformed.read_text(encoding="utf-8")))
        except pinned.GateError:
            pass
        else:
            raise AssertionError("malformed project was accepted")
    print("cosyvoice2_llm closure audit self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path, default=HERE / "pyproject.toml")
    parser.add_argument("--lock", type=Path, default=HERE / "uv.lock")
    parser.add_argument("--license-manifest", type=Path, default=HERE / "license_gate_manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.output is None:
            parser.error("--output is required")
        result = audit(args.project, args.lock, args.license_manifest, args.output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 2
    except (AuditError, pinned.GateError, OSError, UnicodeError) as error:
        print(f"cosyvoice2_llm closure audit: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
