#!/usr/bin/env -S uv run --frozen --project tools/parity/firered_asr_aed_l --python 3.12 python
"""Audit the dedicated FireRed Python closure without acquiring a model.

The lock is the authority for the active Linux/x86_64 package graph.  This
tool records installed distribution metadata, publisher/license candidates,
and native ELF payload hashes.  It deliberately remains BLOCKED until every
transitive row has an explicit owner review; an audit is not a license grant.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
MODEL_REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
AUDIT_FORMAT = "vokra-firered-asr-aed-l-dependency-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_json_no_clobber(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise ValueError(f"refusing to overwrite dependency audit: {path}")
    if path.parent.is_symlink():
        raise ValueError(f"dependency audit path ancestor is a symlink: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matches_linux_x86_64(row: dict[str, Any]) -> bool:
    marker = str(row.get("resolution-markers", row.get("marker", ""))).lower()
    return not marker or not any(token in marker for token in ("sys_platform == 'win32'", "sys_platform == 'darwin'", "platform_machine == 'aarch64'"))


def active_rows(lock_path: Path) -> list[dict[str, Any]]:
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("uv lock has no package rows")
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("uv lock package row is malformed")
        if _matches_linux_x86_64(row):
            by_name.setdefault(row["name"], []).append(row)
    roots = [row for row in packages if isinstance(row.get("name"), str) and row["name"].startswith("vokra-firered-asr-aed-l")]
    if not roots:
        # uv may name the local project after its directory; retaining all
        # rows is safer than silently omitting a dependency from an audit.
        selected = list(by_name.values())
    else:
        selected_names = {row["name"] for row in roots}
        pending = list(selected_names)
        while pending:
            name = pending.pop()
            for row in by_name.get(name, []):
                for dependency in row.get("dependencies", []):
                    dep_name = dependency.get("name") if isinstance(dependency, dict) else None
                    if isinstance(dep_name, str) and dep_name not in selected_names:
                        selected_names.add(dep_name)
                        pending.append(dep_name)
        selected = [by_name[name] for name in sorted(selected_names) if name in by_name]
    rows: list[dict[str, Any]] = []
    for variants in selected:
        for row in variants:
            rows.append({
                "name": row["name"],
                "version": row.get("version"),
                "source": row.get("source"),
                "dependencies": row.get("dependencies", []),
                "resolution_markers": row.get("resolution-markers", row.get("marker")),
            })
    if not rows:
        raise ValueError("active Linux/x86_64 closure is empty")
    return sorted(rows, key=lambda row: (row["name"], str(row.get("version")), json.dumps(row.get("source"), sort_keys=True)))


def distribution_record(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False, "license_candidates": [], "native_payloads": []}
    metadata = distribution.metadata
    licenses = []
    for key in ("License", "License-Expression", "License-File"):
        values = metadata.get_all(key) or []
        licenses.extend({"field": key, "value": value} for value in values if value)
    licenses.extend({"field": "Classifier", "value": value} for value in (metadata.get_all("Classifier") or []) if "License" in value)
    native: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = list(licenses)
    for file in distribution.files or ():
        path = distribution.locate_file(file)
        lower = str(file).lower()
        if any(token in Path(lower).name for token in ("license", "copying", "notice")):
            candidate = {"path": str(file), "exists": path.is_file()}
            if path.is_file():
                candidate.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
            candidates.append(candidate)
        if lower.endswith((".so", ".so.1", ".pyd", ".dylib")) or (path.is_file() and path.read_bytes()[:4] == b"\x7fELF"):
            item = {"path": str(file), "exists": path.is_file()}
            if path.is_file():
                item.update({"bytes": path.stat().st_size, "sha256": sha256_file(path), "elf": path.read_bytes()[:4] == b"\x7fELF"})
            native.append(item)
    return {"name": name, "installed": True, "version": distribution.version, "license_candidates": candidates, "native_payloads": native}


def build_manifest(lock_path: Path, project: Path | None) -> dict[str, Any]:
    rows = active_rows(lock_path)
    row_records = []
    for row in rows:
        row_records.append({**row, "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    packages = [distribution_record(row["name"]) for row in rows if not row["name"].startswith("vokra-firered-asr-aed-l")]
    native_source = None
    if project is not None:
        source = project / "LICENSE"
        if source.is_file() and not source.is_symlink():
            native_source = {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
    return {
        "format": AUDIT_FORMAT,
        "status": "BLOCKED_UNREVIEWED_TRANSITIVE",
        "publication": "NO_UPLOAD",
        "platform": {"system": platform.system(), "machine": platform.machine(), "required": "Linux/x86_64"},
        "lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "model": {"repository": REPOSITORY, "revision": MODEL_REVISION},
        "active_closure": {"rows": row_records, "row_count": len(row_records), "row_digest": hashlib.sha256(json.dumps(row_records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
        "installed_distributions": packages,
        "native_source": native_source,
        "owner_approval": {"status": "MISSING", "required": "per-row license, publisher, and native payload review"},
        "gate": {"status": "BLOCKED_UNREVIEWED_TRANSITIVE", "reason": "all transitive closure rows require explicit owner approval before model snapshot"},
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="firered-audit-") as directory:
        lock = Path(directory) / "uv.lock"
        lock.write_text('''version = 1\n[[package]]\nname = "firered-asr-aed-l"\nversion = "0.0.0"\ndependencies = [{ name = "synthetic" }]\n[[package]]\nname = "synthetic"\nversion = "1.2.3"\nsource = { registry = "https://example.invalid" }\n''', encoding="utf-8")
        rows = active_rows(lock)
        assert [row["name"] for row in rows] == ["firered-asr-aed-l", "synthetic"]
        manifest = build_manifest(lock, None)
        assert manifest["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        assert manifest["publication"] == "NO_UPLOAD"
        assert len(manifest["lock"]["sha256"]) == 64
        assert len(manifest["active_closure"]["row_digest"]) == 64
        assert all(len(row["row_sha256"]) == 64 for row in manifest["active_closure"]["rows"])
        output = Path(directory) / "audit.json"
        publish_json_no_clobber(output, manifest)
        try:
            publish_json_no_clobber(output, manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("dependency audit clobber accepted")
        assert not list(Path(directory).glob("*.tmp"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("firered dependency audit self-test PASS")
        return 0
    if not args.lock or not args.output:
        parser.error("--lock and --output are required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Linux/x86_64 audit is required")
    manifest = build_manifest(args.lock, args.project)
    publish_json_no_clobber(args.output, manifest)
    print("firered dependency audit: BLOCKED_UNREVIEWED_TRANSITIVE")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
