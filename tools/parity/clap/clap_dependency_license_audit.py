#!/usr/bin/env python3
"""Inventory the locked CLAP Python environment without loading a model.

This is an owner-independent evidence collector.  It reads the frozen uv lock,
inspects installed distribution metadata, and hashes bundled license and native
payload files.  It never imports Transformers/Torch and never acquires or
opens a checkpoint.  License disposition remains pending even when the
inventory is structurally complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "vokra-clap-htsat-fused-dependency-license-inventory-v1"
PENDING_STATUS = "PENDING_OWNER_REVIEW"
PENDING_DEPENDENCY_STATUS = "PENDING_VAST_AUDIT"
NO_WEIGHTS = "NOT_ACQUIRED"
NO_UPLOAD = "NO_UPLOAD"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
NATIVE_SUFFIXES = (".dylib", ".dll", ".pyd", ".so")
LICENSE_PREFIXES = ("license", "copying", "notice")


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_no_replace(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"output already exists: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise RuntimeError(f"{label} is missing or is not a sha256 hash: {value!r}")
    return value


def artifact_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a table")
    url = value.get("url")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"{label} has no URL")
    record: dict[str, Any] = {
        "url": url,
        "hash": parse_hash(value.get("hash"), label=f"{label}.hash"),
    }
    if "size" in value:
        size = value["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"{label}.size is invalid: {size!r}")
        record["size"] = size
    return record


def parse_lock(lock_path: Path, project_path: Path) -> dict[str, Any]:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"uv.lock is missing or symlinked: {lock_path}")
    if project_path.is_symlink() or not project_path.is_file():
        raise RuntimeError(f"pyproject.toml is missing or symlinked: {project_path}")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("uv.lock has no package entries")
    project_name = normalize_name(project["project"]["name"])
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    virtual_packages: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            raise RuntimeError(f"uv.lock package {index} is not a table")
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RuntimeError(f"uv.lock package {index} has incomplete name/version")
        normalized = normalize_name(name)
        if normalized in seen:
            raise RuntimeError(f"uv.lock has duplicate package name: {name}")
        seen.add(normalized)
        if not isinstance(source, dict) or not source:
            raise RuntimeError(f"uv.lock {name} has no source table")
        if source.get("virtual") == "." or normalized == project_name:
            virtual_packages.append({"name": name, "version": version})
            continue
        artifacts: dict[str, Any] = {}
        if "sdist" in package:
            artifacts["sdist"] = artifact_record(package["sdist"], label=f"{name}.sdist")
        wheels = package.get("wheels", [])
        if not isinstance(wheels, list):
            raise RuntimeError(f"uv.lock {name}.wheels is not a list")
        artifacts["wheels"] = [
            artifact_record(item, label=f"{name}.wheels[{wheel_index}]")
            for wheel_index, item in enumerate(wheels)
        ]
        if not artifacts.get("sdist") and not artifacts["wheels"]:
            raise RuntimeError(f"uv.lock {name} has no hashed sdist or wheel")
        inventory.append(
            {
                "name": name,
                "normalized_name": normalized,
                "version": version,
                "source": source,
                "artifacts": artifacts,
            }
        )
    if not inventory:
        raise RuntimeError("uv.lock has no non-virtual active packages")
    return {
        "project_name": project["project"]["name"],
        "project_dependencies": sorted(project["project"].get("dependencies", [])),
        "selection": "all non-virtual package entries under the single Linux x86_64 locked resolution",
        "virtual_packages": virtual_packages,
        "requires_python": lock.get("requires-python"),
        "resolution_markers": lock.get("resolution-markers", []),
        "packages": sorted(inventory, key=lambda item: item["normalized_name"]),
    }


def metadata_values(metadata: Any, key: str) -> list[str]:
    values = metadata.get_all(key) if hasattr(metadata, "get_all") else None
    if not values:
        return []
    return [str(value) for value in values]


def regular_file_record(path: Path, *, relative: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"bundled file is missing or symlinked: {relative}")
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def safe_dist_path(dist: Any, file: Any, *, prefix: Path | None = None) -> Path:
    raw = Path(dist.locate_file(file))
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    prefix_path = Path(sys.prefix) if prefix is None else prefix
    prefix_absolute = prefix_path.absolute()
    prefix_real = prefix_path.resolve()
    raw_absolute = raw.absolute()
    try:
        raw_absolute.relative_to(prefix_absolute)
    except ValueError as exc:
        raise RuntimeError(f"distribution file escapes sys.prefix: {raw}") from exc
    current = raw_absolute
    while current != prefix_absolute:
        if current.is_symlink():
            raise RuntimeError(f"distribution file has symlinked ancestry: {raw}")
        if current == current.parent:
            raise RuntimeError(f"distribution file has unsafe ancestry: {raw}")
        current = current.parent
    resolved = raw_absolute.resolve(strict=False)
    try:
        resolved.relative_to(prefix_real)
    except ValueError as exc:
        raise RuntimeError(f"distribution file resolves outside sys.prefix: {raw}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"distribution file is missing or not a regular file: {raw}")
    return resolved


def is_license_file(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(LICENSE_PREFIXES)


def is_native_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(NATIVE_SUFFIXES) or ".so." in name


def distribution_files(dist: Any) -> tuple[list[Any] | None, list[str]]:
    files = dist.files
    if files is None:
        return None, []
    result: list[Any] = []
    unknown: list[str] = []
    for file in files:
        path = Path(str(file))
        if path.is_absolute() or ".." in path.parts:
            unknown.append(str(file))
            continue
        result.append(file)
    return result, unknown


def collect_distribution(
    dist: Any, expected: dict[str, Any], *, prefix: Path | None = None
) -> dict[str, Any]:
    metadata = dist.metadata
    observed_name = metadata.get("Name")
    observed_version = metadata.get("Version")
    license_expression = metadata.get("License-Expression")
    legacy_license = metadata.get("License")
    license_values = metadata_values(metadata, "License")
    classifiers = [value for value in metadata_values(metadata, "Classifier") if value.startswith("License ::")]
    files, unknown_files = distribution_files(dist)
    license_files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    file_errors: list[str] = list(unknown_files)
    if files is not None:
        for file in files:
            relative = str(file)
            try:
                path = safe_dist_path(dist, file, prefix=prefix)
                if is_license_file(path):
                    license_files.append(regular_file_record(path, relative=relative))
                if is_native_file(path):
                    native_files.append(regular_file_record(path, relative=relative))
            except RuntimeError as exc:
                file_errors.append(str(exc))
    license_status = "UNKNOWN" if file_errors and not license_files else "MISSING" if not license_files else "PRESENT" if len(license_files) == 1 else "MULTIPLE"
    native_status = "UNKNOWN" if file_errors else "NONE" if not native_files else "PRESENT" if len(native_files) == 1 else "MULTIPLE"
    findings: list[str] = []
    if observed_name is None or observed_version is None:
        findings.append("distribution metadata name/version missing")
    if normalize_name(str(observed_name)) != expected["normalized_name"]:
        findings.append("distribution name does not match uv.lock")
    if str(observed_version) != expected["version"]:
        findings.append("distribution version does not match uv.lock")
    if license_expression is None:
        findings.append("SPDX License-Expression missing")
    if license_status in {"MISSING", "UNKNOWN"}:
        findings.append(f"bundled license files {license_status.lower()}")
    if file_errors:
        findings.append("distribution file inventory contains unknown entries")
    return {
        "name": observed_name,
        "version": observed_version,
        "expected": {"name": expected["name"], "version": expected["version"]},
        "license": {
            "spdx_expression": license_expression,
            "legacy_license": legacy_license,
            "metadata_license_values": license_values,
            "classifiers": classifiers,
            "bundled_files_status": license_status,
            "bundled_files": license_files,
        },
        "native_payload": {
            "status": native_status,
            "files": native_files,
        },
        "file_inventory_errors": file_errors,
        "findings": findings,
    }


def collect_inventory(
    lock_data: dict[str, Any], distributions: Iterable[Any], *, prefix: Path | None = None
) -> dict[str, Any]:
    expected = {item["normalized_name"]: item for item in lock_data["packages"]}
    installed: dict[str, list[Any]] = {}
    for dist in distributions:
        name = dist.metadata.get("Name")
        if name is not None and normalize_name(str(name)) in expected:
            installed.setdefault(normalize_name(str(name)), []).append(dist)
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    review_flags: list[str] = []
    for normalized_name, package in sorted(expected.items()):
        candidates = installed.get(normalized_name, [])
        if not candidates:
            rows.append({
                "name": package["name"],
                "version": None,
                "expected": {"name": package["name"], "version": package["version"]},
                "distribution_status": "MISSING",
                "findings": ["installed distribution missing"],
            })
            findings.append(f"missing distribution: {package['name']}")
            continue
        if len(candidates) > 1:
            rows.append({
                "name": package["name"],
                "version": None,
                "expected": {"name": package["name"], "version": package["version"]},
                "distribution_status": "MULTIPLE",
                "candidate_count": len(candidates),
                "findings": ["multiple installed distributions match normalized name"],
            })
            findings.append(f"multiple distributions: {package['name']}")
            continue
        row = collect_distribution(candidates[0], package, prefix=prefix)
        row["distribution_status"] = "PRESENT"
        rows.append(row)
        findings.extend(f"{package['name']}: {finding}" for finding in row["findings"])
        if row["license"]["bundled_files_status"] == "MULTIPLE":
            review_flags.append(f"multiple bundled license files: {package['name']}")
        if row["native_payload"]["status"] == "MULTIPLE":
            review_flags.append(f"multiple native payload files: {package['name']}")
        if row["native_payload"]["status"] == "UNKNOWN":
            findings.append(f"native payload inventory unknown: {package['name']}")
    return {
        "distributions": rows,
        "findings": sorted(set(findings)),
        "review_flags": sorted(set(review_flags)),
    }


def audit(
    project_path: Path,
    lock_path: Path,
    distributions: Iterable[Any] | None = None,
    *,
    prefix: Path | None = None,
) -> dict[str, Any]:
    lock_data = parse_lock(lock_path, project_path)
    installed = collect_inventory(
        lock_data,
        importlib.metadata.distributions() if distributions is None else distributions,
        prefix=prefix,
    )
    findings = installed["findings"]
    return {
        "schema": SCHEMA,
        "status": "BLOCKED" if findings else PENDING_STATUS,
        "dependency_audit_status": PENDING_DEPENDENCY_STATUS,
        "owner_approval": "PENDING_OWNER_APPROVAL",
        "weights": NO_WEIGHTS,
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "publication": NO_UPLOAD,
        "inputs": {
            "project": {"path": project_path.name, "sha256": sha256_file(project_path)},
            "lock": {"path": lock_path.name, "sha256": sha256_file(lock_path)},
            "python": platform.python_version(),
            "prefix": sys.prefix,
        },
        "locked_environment": lock_data,
        "installed_environment": installed,
        "findings": findings,
    }


class _SyntheticDistribution:
    def __init__(
        self,
        root: Path,
        name: str,
        version: str,
        *,
        license_text: str | None,
        native: bool = False,
        license_expression: str | None = "MIT",
    ):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.metadata = {
            "Name": name,
            "Version": version,
            "License-Expression": license_expression,
            "License": "MIT" if license_text is not None else None,
            "Classifier": "License :: OSI Approved :: MIT License",
        }
        files: list[str] = []
        if license_text is not None:
            license_path = root / "LICENSE.txt"
            license_path.write_text(license_text, encoding="utf-8")
            files.append("LICENSE.txt")
        if native:
            native_path = root / "module.so"
            native_path.write_bytes(b"native")
            files.append("module.so")
        self._files = files

    @property
    def files(self) -> list[str]:  # type: ignore[override]
        return self._files

    def locate_file(self, file: str) -> Path:
        return self.root / str(file)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="vokra-clap-license-") as temporary:
        root = Path(temporary)
        project = root / "pyproject.toml"
        lock = root / "uv.lock"
        project.write_text('[project]\nname = "synthetic-clap"\nversion = "0.0.0"\n', encoding="utf-8")
        lock.write_text(
            """version = 1\nrequires-python = \"==3.12.*\"\n[[package]]\nname = \"synthetic-clap\"\nversion = \"0.0.0\"\nsource = { virtual = \".\" }\n[[package]]\nname = \"demo-package\"\nversion = \"1.0.0\"\nsource = { registry = \"https://pypi.org/simple\" }\nsdist = { url = \"https://example.invalid/demo.tar.gz\", hash = \"sha256:0000000000000000000000000000000000000000000000000000000000000000\" }\n""",
            encoding="utf-8",
        )
        malformed_lock = root / "malformed.lock"
        malformed_lock.write_text(
            lock.read_text(encoding="utf-8").replace("sha256:" + "0" * 64, "sha256:not-a-hash"),
            encoding="utf-8",
        )
        try:
            parse_lock(malformed_lock, project)
        except RuntimeError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("malformed lock artifact hash was accepted")
        dist = _SyntheticDistribution(root / "dist", "demo-package", "1.0.0", license_text="MIT\n", native=True)
        result = audit(project, lock, [dist], prefix=root)
        assert result["status"] == PENDING_STATUS
        assert result["dependency_audit_status"] != "COMPLETE"
        assert result["locked_environment"]["virtual_packages"][0]["name"] == "synthetic-clap"
        license_record = result["installed_environment"]["distributions"][0]["license"]
        assert license_record["spdx_expression"] == "MIT"
        assert license_record["legacy_license"] == "MIT"
        assert result["installed_environment"]["distributions"][0]["native_payload"]["status"] == "PRESENT"
        duplicate = audit(project, lock, [dist, dist], prefix=root)
        assert duplicate["status"] == "BLOCKED"
        assert duplicate["installed_environment"]["distributions"][0]["distribution_status"] == "MULTIPLE"
        missing_license = _SyntheticDistribution(
            root / "missing",
            "demo-package",
            "1.0.0",
            license_text=None,
            license_expression=None,
        )
        missing = audit(project, lock, [missing_license], prefix=root)
        assert missing["status"] == "BLOCKED"
        assert "SPDX License-Expression missing" in " ".join(missing["findings"])
        assert missing["installed_environment"]["distributions"][0]["license"]["legacy_license"] is None
        with tempfile.TemporaryDirectory(
            prefix="vokra-clap-license-escape-", dir=root.parent
        ) as escaped_root:
            escaped = _SyntheticDistribution(
                Path(escaped_root),
                "demo-package",
                "1.0.0",
                license_text="MIT\n",
            )
            escaped_result = audit(project, lock, [escaped], prefix=root)
            assert escaped_result["status"] == "BLOCKED"
            escaped_row = escaped_result["installed_environment"]["distributions"][0]
            assert "escapes sys.prefix" in " ".join(escaped_row["file_inventory_errors"])
        output = root / "inventory.json"
        write_atomic_no_replace(output, "first\n")
        try:
            write_atomic_no_replace(output, "second\n")
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("inventory output replacement was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.lock, args.output)):
            parser.error("--self-test accepts no audit paths")
        self_test()
        print("clap dependency/license audit self-test: OK")
        return 0
    if args.project is None or args.lock is None or args.output is None:
        parser.error("normal runs require --project, --lock, and --output")
    evidence = audit(args.project, args.lock)
    try:
        write_atomic_no_replace(args.output, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"CLAP_DEPENDENCY_LICENSE_AUDIT {evidence['status']}: {args.output}")
    return 0 if evidence["status"] != "BLOCKED" else 2


if __name__ == "__main__":
    sys.exit(main())
