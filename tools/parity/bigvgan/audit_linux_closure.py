#!/usr/bin/env python3
"""Collect hash-bound owner-review evidence for the active Linux closure.

This offline collector never downloads packages and never creates approval
signatures. The operator supplies exact artifacts selected from uv.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import tomllib


LINUX_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
NATIVE_SUFFIXES = (".so", ".dylib", ".dll", ".a")
LICENSE_NAMES = {
    "license", "license.txt", "license.md", "copying", "copying.txt",
    "notice", "notice.txt", "notice.md",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"bigvgan closure audit: BLOCKED: {message}")


def active_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        fail("uv.lock package table is missing")
    rows = []
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            fail("uv.lock package row is malformed")
        source = row.get("source")
        if not isinstance(source, dict) or "registry" not in source:
            continue
        markers = row.get("resolution-markers", [])
        if markers and LINUX_MARKER not in markers:
            continue
        rows.append(row)
    if not rows:
        fail("no active Linux registry packages found")
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def artifact_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    wheels = row.get("wheels", [])
    if not isinstance(wheels, list):
        fail(f"{row['name']} wheels table is malformed")
    return [item for item in wheels if isinstance(item, dict) and isinstance(item.get("url"), str)]


def wheel_compatibility(filename: str) -> str | None:
    """Return the only two accepted CPython 3.12 x86_64 glibc classes."""
    if not filename.endswith(".whl"):
        return None
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4:
        return None
    _name_version, python_tag, abi_tag, platform_tag = parts
    if python_tag == "py3" and abi_tag == "none" and platform_tag == "any":
        return "py3-none-any-universal"
    platforms = platform_tag.split(".")
    if python_tag == "cp312" and abi_tag == "cp312" and platforms and all(
        tag.startswith("manylinux") and tag.endswith("_x86_64") for tag in platforms
    ):
        return "cp312-cp312-manylinux-x86_64-glibc"
    return None


def select_artifact(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    compatible: list[tuple[dict[str, Any], str]] = []
    for candidate in artifact_candidates(row):
        filename = Path(unquote(urlparse(candidate["url"]).path)).name
        basis = wheel_compatibility(filename)
        if basis is not None:
            compatible.append((candidate, basis))
    compatible.sort(key=lambda item: (0 if item[1].startswith("cp312") else 1, item[0]["url"]))
    if not compatible:
        fail(
            f"{row['name']} has no CPython 3.12 x86_64 glibc wheel; "
            "musllinux/aarch64/macOS/cp311 and sdist fallback are refused"
        )
    return compatible[0]


def artifact_path(row: dict[str, Any], artifacts_dir: Path) -> tuple[Path, dict[str, Any], str]:
    candidate, basis = select_artifact(row)
    filename = Path(unquote(urlparse(candidate["url"]).path)).name
    path = artifacts_dir / filename
    if path.is_file() and not path.is_symlink():
        return path, candidate, basis
    fail(f"missing locked payload for {row['name']} (stage: {filename})")


def member_payloads(path: Path) -> dict[str, bytes]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
    try:
        with tarfile.open(path, mode="r:*") as archive:
            payloads: dict[str, bytes] = {}
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is not None:
                    payloads[member.name] = stream.read()
            return payloads
    except (tarfile.TarError, OSError) as exc:
        fail(f"unsupported package payload {path.name}: {exc}")


def metadata_value(metadata: bytes, key: str) -> list[str]:
    prefix = f"{key.casefold()}:"
    return [
        line.split(":", 1)[1].strip()
        for line in metadata.decode("utf-8", errors="replace").splitlines()
        if line.casefold().startswith(prefix)
    ]


def is_license_payload(name: str) -> bool:
    basename = Path(name).name.casefold()
    return basename in LICENSE_NAMES or basename.startswith("license-") or basename.startswith("notice-")


def payload_record(name: str, payload: bytes) -> dict[str, Any]:
    return {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def ensure_safe_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        fail(f"output already exists or is symlinked: {path}")
    parent = path.parent if path.parent.is_absolute() else Path.cwd() / path.parent
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current /= component
        if current.is_symlink():
            resolved = current.resolve()
            if (current, resolved) not in ((Path("/var"), Path("/private/var")), (Path("/tmp"), Path("/private/tmp"))):
                fail(f"output parent contains a symlink: {current}")
        if current.exists() and not current.is_dir():
            fail(f"output parent is not a directory: {current}")


def inspect_package(row: dict[str, Any], path: Path, locked_artifact: dict[str, Any], selection_basis: str) -> dict[str, Any]:
    payloads = member_payloads(path)
    metadata_names = sorted(name for name in payloads if name.endswith(".dist-info/METADATA"))
    if len(metadata_names) != 1:
        fail(f"{row['name']} must contain exactly one dist-info/METADATA")
    metadata = payloads[metadata_names[0]]
    native = sorted(
        (payload_record(name, payload) for name, payload in payloads.items() if name.casefold().endswith(NATIVE_SUFFIXES)),
        key=lambda item: item["path"],
    )
    license_payloads = sorted(
        (payload_record(name, payload) for name, payload in payloads.items() if is_license_payload(name)),
        key=lambda item: item["path"],
    )
    return {
        "id": f"{row['name']}@{row['version']}",
        "name": row["name"],
        "version": row["version"],
        "registry": row["source"]["registry"],
        "artifact_url": locked_artifact["url"],
        "selected_filename": path.name,
        "selection_basis": selection_basis,
        "artifact_sha256": sha256_file(path),
        "artifact_bytes": path.stat().st_size,
        "lock_sha256": locked_artifact["hash"].removeprefix("sha256:"),
        "metadata": {
            "path": metadata_names[0],
            "sha256": sha256_bytes(metadata),
            "license": metadata_value(metadata, "License"),
            "license_classifiers": metadata_value(metadata, "Classifier"),
            "project_urls": metadata_value(metadata, "Project-URL"),
            "home_page": metadata_value(metadata, "Home-page"),
        },
        "license_payloads": license_payloads,
        "notice_payloads": [item for item in license_payloads if Path(item["path"]).name.casefold().startswith("notice")],
        "native_bundled_payloads": native,
        "native_bundled_review": "OWNER_REVIEW_REQUIRED",
        "status": "CANDIDATE_PENDING_OWNER_SIGNOFF",
    }


def audit(lock_path: Path, artifacts_dir: Path, output: Path) -> None:
    if not lock_path.is_file() or lock_path.is_symlink():
        fail("lock is missing or symlinked")
    if not artifacts_dir.is_dir() or artifacts_dir.is_symlink():
        fail("artifacts directory is missing or symlinked")
    ensure_safe_output(output)
    lock_bytes = lock_path.read_bytes()
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        fail(f"uv.lock is not valid TOML: {exc}")
    packages = []
    for row in active_rows(lock):
        path, locked_artifact, selection_basis = artifact_path(row, artifacts_dir)
        actual = sha256_file(path)
        expected = locked_artifact.get("hash", "").removeprefix("sha256:")
        if actual != expected:
            fail(f"{row['name']} payload SHA-256 {actual} != locked {expected}")
        if isinstance(locked_artifact.get("size"), int) and path.stat().st_size != locked_artifact["size"]:
            fail(f"{row['name']} payload size does not match uv.lock")
        packages.append(inspect_package(row, path, locked_artifact, selection_basis))
    candidate = {
        "schema": "bigvgan-linux-closure-candidate-v1",
        "decision": "OWNER_REVIEW_REQUIRED",
        "platform": "x86_64-linux",
        "lock_sha256": sha256_bytes(lock_bytes),
        "active_package_count": len(packages),
        "packages": packages,
        "approval": {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None},
        "review_scope": {
            "execution_closure": "active x86_64-linux packages only",
            "supported_platform_license_review": "license_gate_manifest still covers all 12 lock rows, including inactive Darwin torch",
        },
        "publication": "NO_UPLOAD",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"bigvgan Linux closure candidate: {len(packages)} packages, owner approval remains required")


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bigvgan-closure-") as directory:
        root = Path(directory)
        artifacts = root / "artifacts"
        artifacts.mkdir()
        import io

        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(
                "demo-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: demo\nLicense: MIT\n",
            )
            archive.writestr("LICENSE", "MIT License\n")
            archive.writestr("NOTICE", "demo notice\n")
            archive.writestr("demo/libdemo.so", b"native")
        wheel = payload.getvalue()
        wheel_path = artifacts / "demo-1.0-py3-none-any.whl"
        wheel_path.write_bytes(wheel)
        lock = root / "uv.lock"
        lock.write_text(
            """version = 1
revision = 3
requires-python = '==3.12.*'
resolution-markers = []
supported-markers = []

[[package]]
name = 'demo'
version = '1.0'
source = { registry = 'https://pypi.org/simple' }
wheels = [{ url = 'https://files.pythonhosted.org/packages/demo-1.0-py3-none-any.whl', hash = 'sha256:PLACEHOLDER', size = 0 }]

[[package]]
name = 'inactive'
version = '1.0'
source = { registry = 'https://pypi.org/simple' }
resolution-markers = ["platform_machine == 'arm64' and sys_platform == 'darwin'"]
wheels = [{ url = 'https://files.pythonhosted.org/packages/inactive.whl', hash = 'sha256:00', size = 1 }]
""".replace("PLACEHOLDER", sha256_bytes(wheel)).replace("size = 0", f"size = {len(wheel)}"),
            encoding="utf-8",
        )
        candidate = root / "candidate.json"
        audit(lock, artifacts, candidate)
        value = json.loads(candidate.read_text(encoding="utf-8"))
        assert value["active_package_count"] == 1
        assert value["packages"][0]["selected_filename"] == wheel_path.name
        assert value["packages"][0]["selection_basis"] == "py3-none-any-universal"
        assert value["packages"][0]["native_bundled_payloads"][0]["sha256"] == sha256_bytes(b"native")
        assert value["packages"][0]["notice_payloads"][0]["path"] == "NOTICE"
        assert value["approval"] == {
            "status": "OWNER_SIGNOFF_REQUIRED",
            "signer": None,
            "digest": None,
        }
        try:
            audit(lock, artifacts, candidate)
        except SystemExit as exc:
            assert "output already exists" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test overwrote an existing output")
        wheel_path.write_bytes(wheel + b"tampered")
        tampered_output = root / "tampered.json"
        try:
            audit(lock, artifacts, tampered_output)
        except SystemExit as exc:
            assert "payload SHA-256" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted a tampered payload")
        wrong_rows = {
            "musllinux": "demo-1.0-cp312-cp312-musllinux_1_2_x86_64.whl",
            "aarch64": "demo-1.0-cp312-cp312-manylinux_2_28_aarch64.whl",
            "cp311": "demo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
            "macos": "demo-1.0-cp312-cp312-macosx_11_0_arm64.whl",
        }
        for label, filename in wrong_rows.items():
            row = {"name": "demo", "wheels": [{"url": f"https://files.pythonhosted.org/{filename}"}]}
            try:
                select_artifact(row)
            except SystemExit as exc:
                assert "no CPython 3.12 x86_64 glibc wheel" in str(exc), label
            else:
                raise SystemExit(f"bigvgan closure self-test accepted wrong-only {label} wheel")
        linked_parent = root / "linked-parent"
        linked_parent.symlink_to(artifacts, target_is_directory=True)
        try:
            audit(lock, artifacts, linked_parent / "candidate.json")
        except SystemExit as exc:
            assert "output parent contains a symlink" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted a symlinked output parent")
    print("audit_linux_closure.py self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.artifacts_dir, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return
    if args.lock is None or args.artifacts_dir is None or args.output is None:
        parser.error("--lock, --artifacts-dir, and --output are required")
    audit(args.lock, args.artifacts_dir, args.output)


if __name__ == "__main__":
    main()
