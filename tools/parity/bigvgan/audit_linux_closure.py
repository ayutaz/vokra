#!/usr/bin/env python3
"""Collect hash-bound owner-review evidence for the active Linux closure.

This offline collector never downloads packages and never creates approval
signatures. The operator supplies exact artifacts selected from uv.lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import tarfile
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import tomllib


LINUX_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
LINUX_ENV = {"platform_machine": "x86_64", "sys_platform": "linux"}
REGISTRY_HOSTS = {"pypi.org", "download.pytorch.org"}
DOWNLOAD_HOSTS = {"files.pythonhosted.org", "download-r2.pytorch.org"}
NATIVE_SUFFIXES = (".so", ".dylib", ".dll", ".a")
NATIVE_MAGICS = (
    b"\x7fELF",
    b"!<arch>\n",
    b"MZ",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
)
MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXPANSION_RATIO = 1000
MAX_METADATA_BYTES = 1024 * 1024
MAX_LICENSE_BYTES = 16 * 1024 * 1024
MAX_IO_CHUNK = 1 << 20
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


def marker_applies(marker: str) -> bool:
    """Evaluate the deliberately small marker grammar used by this lock."""
    if not isinstance(marker, str) or not marker.strip():
        fail("empty or non-string resolution marker")
    token_re = re.compile(r"\s*(?:(==|!=)|([()])|([A-Za-z_][A-Za-z0-9_-]*)|('(?:[^']*)'|\"(?:[^\"]*)\"))")
    tokens: list[str] = []
    offset = 0
    while offset < len(marker):
        match = token_re.match(marker, offset)
        if match is None:
            fail(f"unsupported resolution marker grammar: {marker!r}")
        tokens.append(next(group for group in match.groups() if group is not None))
        offset = match.end()
    cursor = 0

    def peek() -> str | None:
        return tokens[cursor] if cursor < len(tokens) else None

    def consume(expected: str | None = None) -> str:
        nonlocal cursor
        token = peek()
        if token is None or (expected is not None and token != expected):
            fail(f"unsupported resolution marker grammar: {marker!r}")
        cursor += 1
        return token

    def parse_atom() -> bool:
        if peek() == "(":
            consume("(")
            value = parse_or()
            consume(")")
            return value
        variable = consume()
        if variable not in LINUX_ENV:
            fail(f"unsupported resolution marker variable {variable!r}: {marker!r}")
        operator = consume()
        if operator not in {"==", "!="}:
            fail(f"unsupported resolution marker operator {operator!r}: {marker!r}")
        literal = consume()
        if len(literal) < 2 or literal[0] not in {"'", '"'} or literal[-1] != literal[0]:
            fail(f"resolution marker comparison must use a quoted value: {marker!r}")
        result = LINUX_ENV[variable] == literal[1:-1]
        return result if operator == "==" else not result

    def parse_and() -> bool:
        value = parse_atom()
        while peek() == "and":
            consume("and")
            value = parse_atom() and value
        return value

    def parse_or() -> bool:
        value = parse_and()
        while peek() == "or":
            consume("or")
            value = parse_and() or value
        return value

    value = parse_or()
    if peek() is not None:
        fail(f"unsupported resolution marker grammar: {marker!r}")
    return value


def marker_list(row: dict[str, Any], field: str) -> list[str]:
    markers = row.get(field, [])
    if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
        fail(f"{row.get('name', '<root>')} {field} is malformed")
    for marker in markers:
        marker_applies(marker)
    return markers


def row_applies(row: dict[str, Any]) -> bool:
    markers = marker_list(row, "resolution-markers")
    return not markers or any(marker_applies(marker) for marker in markers)


def validate_registry_url(url: Any) -> str:
    if not isinstance(url, str):
        fail("registry source URL is missing")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in REGISTRY_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        fail(f"registry host is not allowlisted: {url!r}")
    return parsed.hostname


def canonical_wheel_name(url: Any) -> str:
    if not isinstance(url, str):
        fail("wheel URL is missing")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in DOWNLOAD_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        fail(f"wheel download host is not allowlisted: {url!r}")
    raw_segment = parsed.path.rsplit("/", 1)[-1]
    filename = unquote(raw_segment)
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(unicodedata.category(char) == "Cc" for char in filename)
        or not filename.endswith(".whl")
        or filename != Path(filename).name
    ):
        fail(f"wheel URL does not contain a safe filename: {url!r}")
    return filename


def locked_artifact(row: dict[str, Any]) -> tuple[dict[str, Any], str, str, str, int]:
    candidate, basis = select_artifact(row)
    filename = canonical_wheel_name(candidate.get("url"))
    digest = candidate.get("hash")
    size = candidate.get("size")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest.removeprefix("sha256:")) != 64
        or any(char not in "0123456789abcdef" for char in digest.removeprefix("sha256:"))
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        fail(f"{row.get('name', '<unknown>')} selected wheel lock row is malformed")
    return candidate, basis, filename, digest.removeprefix("sha256:"), size


def active_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        fail("uv.lock package table is missing")
    for field in ("resolution-markers", "supported-markers"):
        markers = lock.get(field, [])
        if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
            fail(f"uv.lock {field} is malformed")
        for marker in markers:
            marker_applies(marker)
    virtual_rows: list[dict[str, Any]] = []
    registry_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            fail("uv.lock package row is malformed")
        source = row.get("source")
        if not isinstance(source, dict):
            fail(f"{row['name']} package source is malformed")
        marker_list(row, "resolution-markers")
        dependencies = row.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(dep, dict) for dep in dependencies):
            fail(f"{row['name']} dependencies table is malformed")
        for dep in dependencies:
            if "marker" in dep:
                if not isinstance(dep["marker"], str):
                    fail(f"{row['name']} dependency marker is malformed")
                marker_applies(dep["marker"])
        if "virtual" in source:
            if source.get("virtual") != ".":
                fail(f"{row['name']} virtual source is not the repository root")
            virtual_rows.append(row)
            continue
        if set(source) != {"registry"}:
            fail(f"{row['name']} package source is malformed")
        registry = validate_registry_url(source["registry"])
        key = (row["name"], row.get("version"), registry)
        if not isinstance(row.get("version"), str) or key in registry_rows:
            fail(f"ambiguous duplicate registry row: {key!r}")
        registry_rows[key] = row
    if len(virtual_rows) != 1:
        fail("uv.lock must contain exactly one virtual repository root")
    root = virtual_rows[0]
    reached: dict[tuple[str, str, str], dict[str, Any]] = {}
    pending = list(root.get("dependencies", []))

    def resolve_dependency(dependency: dict[str, Any]) -> dict[str, Any]:
        name = dependency.get("name")
        version = dependency.get("version")
        source = dependency.get("source")
        if not isinstance(name, str) or (version is not None and not isinstance(version, str)):
            fail("reachable dependency identity is incomplete")
        candidates = [row for row in registry_rows.values() if row["name"] == name]
        if version is not None:
            candidates = [row for row in candidates if row["version"] == version]
        if source is not None:
            if not isinstance(source, dict) or set(source) != {"registry"}:
                fail(f"{name} dependency source is malformed")
            registry_url = source.get("registry")
            validate_registry_url(registry_url)
            candidates = [
                row for row in candidates if row["source"]["registry"] == registry_url
            ]
        candidates = [row for row in candidates if row_applies(row)]
        if len(candidates) != 1:
            fail(f"ambiguous or missing reachable dependency row: {name!r}")
        return candidates[0]

    while pending:
        dependency = pending.pop(0)
        if not isinstance(dependency, dict):
            fail("repository root dependency is malformed")
        marker = dependency.get("marker")
        if marker is not None and (not isinstance(marker, str) or not marker_applies(marker)):
            continue
        row = resolve_dependency(dependency)
        registry = urlparse(row["source"]["registry"]).hostname
        key = (row["name"], row["version"], registry)
        if not row_applies(row):
            fail(f"reachable dependency row is inactive for Linux: {key!r}")
        if key in reached:
            continue
        reached[key] = row
        dependencies = row.get("dependencies", [])
        for child in dependencies:
            child_marker = child.get("marker")
            if child_marker is None or marker_applies(child_marker):
                pending.append(child)
    rows = list(reached.values())
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
    candidate, basis, filename, _digest, _size = locked_artifact(row)
    path = artifacts_dir / filename
    if path.is_file() and not path.is_symlink():
        return path, candidate, basis
    fail(f"missing locked payload for {row['name']} (stage: {filename})")


def is_license_payload(name: str) -> bool:
    basename = Path(name).name.casefold()
    return (
        basename in LICENSE_NAMES
        or basename.startswith((
            "license",
            "notice",
            "copying",
            "copyright",
            "third_party_license",
            "third_party_notice",
        ))
    )


def normalized_member_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
    ):
        fail(f"archive member path is unsafe: {name!r}")
    if any(part == ".." for part in name.split("/")):
        fail(f"archive member path traverses a parent: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        fail(f"archive member path is unsafe: {name!r}")
    return normalized


def stream_record(
    name: str,
    declared_size: int,
    stream: Any,
    capture_limit: int = 0,
) -> tuple[dict[str, Any], bytes]:
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
        fail(f"archive member has an invalid declared size: {name}")
    if declared_size > MAX_MEMBER_BYTES:
        fail(f"archive member exceeds per-member limit: {name}")
    if capture_limit and declared_size > capture_limit:
        fail(f"archive metadata/license member exceeds bounded read limit: {name}")
    digest = hashlib.sha256()
    captured = bytearray()
    prefix = bytearray()
    total = 0
    while True:
        chunk = stream.read(min(MAX_IO_CHUNK, declared_size - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > declared_size:
            fail(f"archive member extracted size exceeds declaration: {name}")
        digest.update(chunk)
        if len(prefix) < 4:
            prefix.extend(chunk[: 4 - len(prefix)])
        if capture_limit:
            captured.extend(chunk)
    if total != declared_size:
        fail(f"archive member extracted size differs from declaration: {name}")
    preview = bytes(captured) if capture_limit else bytes(prefix)
    return {"path": name, "bytes": total, "sha256": digest.hexdigest()}, preview


def archive_member_type_is_safe(info: Any, is_zip: bool) -> bool:
    if is_zip:
        mode = (info.external_attr >> 16) & 0o170000
        if mode in {stat.S_IFLNK, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFIFO, stat.S_IFSOCK}:
            fail(f"archive member is a symlink or special file: {info.filename}")
        if mode == stat.S_IFDIR or info.is_dir():
            if info.file_size != 0:
                fail(f"archive directory has a non-zero declared size: {info.filename}")
            return False
        if mode not in {0, stat.S_IFREG}:
            fail(f"archive member has an unsupported file type: {info.filename}")
        if info.flag_bits & 0x1:
            fail(f"encrypted archive member is not accepted: {info.filename}")
        return True
    if info.isdir():
        if info.size != 0:
            fail(f"archive directory has a non-zero declared size: {info.name}")
        return False
    if not info.isfile() or info.issym() or info.islnk() or info.isdev():
        fail(f"archive member is not a regular file: {info.name}")
    return True


def inspect_archive(path: Path) -> dict[str, Any]:
    """Inspect payloads with bounded streaming reads and no archive-wide bytes map."""
    archive_size = path.stat().st_size
    declared_total = 0
    seen: set[str] = set()
    metadata: list[tuple[str, bytes]] = []
    licenses: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []

    def begin_member(raw_name: str, declared_size: int) -> str:
        nonlocal declared_total
        name = normalized_member_name(raw_name)
        if name in seen:
            fail(f"archive contains duplicate normalized member path: {name}")
        seen.add(name)
        if declared_size > MAX_MEMBER_BYTES:
            fail(f"archive member exceeds per-member limit: {name}")
        declared_total += declared_size
        if declared_total > MAX_TOTAL_BYTES:
            fail("archive declared expansion exceeds total limit")
        if archive_size and declared_total > archive_size * MAX_EXPANSION_RATIO:
            fail("archive declared expansion ratio exceeds limit")
        return name

    def consume(raw_name: str, declared_size: int, stream: Any) -> None:
        name = begin_member(raw_name, declared_size)
        lower = name.casefold()
        capture_limit = MAX_METADATA_BYTES if lower.endswith(".dist-info/metadata") else 0
        if is_license_payload(name):
            capture_limit = MAX_LICENSE_BYTES
        record, preview = stream_record(name, declared_size, stream, capture_limit)
        if lower.endswith(".dist-info/metadata"):
            metadata.append((name, preview))
        if is_license_payload(name):
            licenses.append(record)
        if (
            lower.endswith(NATIVE_SUFFIXES)
            or lower.endswith((".pyd", ".lib"))
            or ".so." in lower
            or ".dylib." in lower
            or any(preview.startswith(magic) for magic in NATIVE_MAGICS)
        ):
            native.append(record)

    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    name = normalized_member_name(info.filename)
                    if name in seen:
                        fail(f"archive contains duplicate normalized member path: {name}")
                    is_file = archive_member_type_is_safe(info, True)
                    if not is_file:
                        seen.add(name)
                        continue
                    with archive.open(info, "r") as stream:
                        consume(info.filename, info.file_size, stream)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            fail(f"unsupported or unreadable ZIP payload {path.name}: {exc}")
    else:
        try:
            with tarfile.open(path, mode="r:*") as archive:
                for info in archive:
                    is_file = archive_member_type_is_safe(info, False)
                    name = normalized_member_name(info.name)
                    if name in seen:
                        fail(f"archive contains duplicate normalized member path: {name}")
                    if not is_file:
                        seen.add(name)
                        continue
                    stream = archive.extractfile(info)
                    if stream is None:
                        fail(f"archive member has no readable payload: {name}")
                    with stream:
                        consume(info.name, info.size, stream)
        except (tarfile.TarError, OSError) as exc:
            fail(f"unsupported or unreadable TAR payload {path.name}: {exc}")
    return {"metadata": metadata, "licenses": licenses, "native": native}


def pep503_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def parse_metadata(metadata: bytes, row: dict[str, Any], path: str) -> dict[str, list[str]]:
    try:
        text = metadata.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{row['name']} METADATA is not UTF-8: {exc}")
    fields: dict[str, list[str]] = {}
    current: str | None = None
    multi = {"classifier", "project-url", "provides-extra", "requires-dist"}
    for line in text.splitlines():
        if not line:
            break
        if line[0].isspace():
            if current is None:
                fail(f"{row['name']} METADATA has an orphan continuation: {path}")
            fields[current][-1] += " " + line.strip()
            continue
        if ":" not in line:
            fail(f"{row['name']} METADATA has a malformed header: {path}")
        key, value = line.split(":", 1)
        key = key.casefold()
        if not key or key in fields and key not in multi:
            fail(f"{row['name']} METADATA has an ambiguous duplicate header: {key}")
        fields.setdefault(key, []).append(value.strip())
        current = key
    names = fields.get("name", [])
    versions = fields.get("version", [])
    if len(names) != 1 or pep503_name(names[0]) != pep503_name(row["name"]):
        fail(f"{row['name']} METADATA Name does not match the lock row")
    if len(versions) != 1 or versions[0] != row["version"]:
        fail(f"{row['name']} METADATA Version does not match the lock row")
    return fields


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


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON beside the destination and publish it without replacement."""
    ensure_safe_output(path)
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            fail(f"output was created concurrently; refusing to overwrite: {path}")
            raise AssertionError("unreachable") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def inspect_package(row: dict[str, Any], path: Path, locked_artifact: dict[str, Any], selection_basis: str) -> dict[str, Any]:
    inventory = inspect_archive(path)
    metadata = inventory["metadata"]
    if len(metadata) != 1:
        fail(f"{row['name']} must contain exactly one dist-info/METADATA")
    metadata_path, metadata_bytes = metadata[0]
    metadata_fields = parse_metadata(metadata_bytes, row, metadata_path)
    native = sorted(inventory["native"], key=lambda item: item["path"])
    license_payloads = sorted(inventory["licenses"], key=lambda item: item["path"])
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
            "path": metadata_path,
            "sha256": sha256_bytes(metadata_bytes),
            "license": metadata_fields.get("license", []),
            "license_classifiers": metadata_fields.get("classifier", []),
            "project_urls": metadata_fields.get("project-url", []),
            "home_page": metadata_fields.get("home-page", []),
        },
        "license_payloads": license_payloads,
        "notice_payloads": [item for item in license_payloads if Path(item["path"]).name.casefold().startswith("notice")],
        "native_bundled_payloads": native,
        "native_bundled_review": "OWNER_REVIEW_REQUIRED",
        "dependency_review": "BLOCKED_UNREVIEWED_TRANSITIVE",
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
        expected = locked_artifact["hash"].removeprefix("sha256:")
        if actual != expected:
            fail(f"{row['name']} payload SHA-256 {actual} != locked {expected}")
        if path.stat().st_size != locked_artifact["size"]:
            fail(f"{row['name']} payload size does not match uv.lock")
        packages.append(inspect_package(row, path, locked_artifact, selection_basis))
    candidate = {
        "schema": "bigvgan-linux-closure-candidate-v1",
        "decision": "OWNER_REVIEW_REQUIRED",
        "platform": "x86_64-linux",
        "lock_sha256": sha256_bytes(lock_bytes),
        "active_package_count": len(packages),
        "packages": packages,
        "dependency_review": "BLOCKED_UNREVIEWED_TRANSITIVE",
        "approval": {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None},
        "review_scope": {
            "execution_closure": "active x86_64-linux packages only",
            "supported_platform_license_review": "license_gate_manifest still covers all 12 lock rows, including inactive Darwin torch",
        },
        "publication": "NO_UPLOAD",
    }
    atomic_write_json(output, candidate)
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
                "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nLicense: MIT\n",
            )
            archive.writestr("LICENSE", "MIT License\n")
            archive.writestr("NOTICE", "demo notice\n")
            archive.writestr("COPYRIGHT.txt", "copyright\n")
            archive.writestr("THIRD_PARTY_LICENSES.txt", "third party\n")
            archive.writestr("demo/libdemo.so", b"native")
        wheel = payload.getvalue()
        wheel_path = artifacts / "demo-1.0+cpu-py3-none-any.whl"
        wheel_path.write_bytes(wheel)

        def expect_archive_blocked(label: str, builder: Any, needle: str) -> None:
            malformed = io.BytesIO()
            with zipfile.ZipFile(malformed, "w") as archive:
                builder(archive)
            malformed_path = root / f"{label}.whl"
            malformed_path.write_bytes(malformed.getvalue())
            try:
                inspect_archive(malformed_path)
            except SystemExit as exc:
                assert needle in str(exc), (label, exc)
            else:
                raise SystemExit(f"bigvgan closure self-test accepted malformed {label} archive")

        expect_archive_blocked(
            "duplicate",
            lambda archive: (archive.writestr("member", b"one"), archive.writestr("./member", b"two")),
            "duplicate normalized",
        )
        expect_archive_blocked("traversal", lambda archive: archive.writestr("../escape", b"x"), "traverses")

        def write_symlink(archive: zipfile.ZipFile) -> None:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, b"target")

        expect_archive_blocked("symlink", write_symlink, "symlink or special")

        encrypted_buffer = io.BytesIO()
        with zipfile.ZipFile(encrypted_buffer, "w") as archive:
            archive.writestr("secret", b"secret")
        encrypted_bytes = bytearray(encrypted_buffer.getvalue())
        for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            cursor = encrypted_bytes.find(signature)
            assert cursor >= 0
            flags = int.from_bytes(encrypted_bytes[cursor + flag_offset : cursor + flag_offset + 2], "little")
            encrypted_bytes[cursor + flag_offset : cursor + flag_offset + 2] = (flags | 0x1).to_bytes(2, "little")
        encrypted_path = root / "encrypted.whl"
        encrypted_path.write_bytes(encrypted_bytes)
        try:
            inspect_archive(encrypted_path)
        except SystemExit as exc:
            assert "encrypted" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted encrypted archive")
        native_payload = io.BytesIO()
        with zipfile.ZipFile(native_payload, "w") as archive:
            archive.writestr("extensionless", b"\x7fELF" + b"payload")
        native_path = root / "extensionless.whl"
        native_path.write_bytes(native_payload.getvalue())
        assert inspect_archive(native_path)["native"][0]["path"] == "extensionless"
        native_magic_payload = io.BytesIO()
        with zipfile.ZipFile(native_magic_payload, "w") as archive:
            archive.writestr("extensionless-pe", b"MZ" + b"payload")
            archive.writestr("extensionless-mach-o", b"\xfe\xed\xfa\xcf" + b"payload")
            archive.writestr("extension.pyd", b"payload")
            archive.writestr("library.dylib.1", b"payload")
        native_magic_path = root / "native-magic.whl"
        native_magic_path.write_bytes(native_magic_payload.getvalue())
        native_magic_names = {record["path"] for record in inspect_archive(native_magic_path)["native"]}
        assert native_magic_names == {
            "extensionless-mach-o",
            "extensionless-pe",
            "extension.pyd",
            "library.dylib.1",
        }
        lock_row = {"name": "demo", "version": "1.0"}
        try:
            parse_metadata(b"Metadata-Version: 2.1\nName: other\nVersion: 1.0\n", lock_row, "METADATA")
        except SystemExit as exc:
            assert "Name does not match" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted a metadata Name mismatch")
        try:
            parse_metadata(
                b"Metadata-Version: 2.1\nName: demo\nName: demo\nVersion: 1.0\n",
                lock_row,
                "METADATA",
            )
        except SystemExit as exc:
            assert "duplicate header" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted duplicate metadata headers")
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
wheels = [{ url = 'https://files.pythonhosted.org/packages/demo-1.0%2Bcpu-py3-none-any.whl', hash = 'sha256:PLACEHOLDER', size = 0 }]

[[package]]
name = 'vokra-bigvgan-parity'
version = '0.1.0'
source = { virtual = '.' }
dependencies = [{ name = 'demo', version = '1.0', source = { registry = 'https://pypi.org/simple' } }]

[[package]]
name = 'unreachable-orphan'
version = '1.0'
source = { registry = 'https://pypi.org/simple' }
wheels = [{ url = 'https://files.pythonhosted.org/packages/demo-1.0%2Bcpu-py3-none-any.whl' }]

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
        assert value["dependency_review"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        assert value["packages"][0]["selected_filename"] == wheel_path.name
        assert value["packages"][0]["selection_basis"] == "py3-none-any-universal"
        assert value["packages"][0]["native_bundled_payloads"][0]["sha256"] == sha256_bytes(b"native")
        assert value["packages"][0]["notice_payloads"][0]["path"] == "NOTICE"
        assert {item["path"] for item in value["packages"][0]["license_payloads"]} == {
            "COPYRIGHT.txt",
            "LICENSE",
            "NOTICE",
            "THIRD_PARTY_LICENSES.txt",
        }
        assert value["packages"][0]["dependency_review"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
        assert value["approval"] == {
            "status": "OWNER_SIGNOFF_REQUIRED",
            "signer": None,
            "digest": None,
        }
        try:
            marker_applies("python_version > '3.12'")
        except SystemExit as exc:
            assert "unsupported resolution marker" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted unknown marker grammar")
        duplicate_lock = root / "duplicate.lock"
        duplicate_lock.write_text(
            lock.read_text(encoding="utf-8")
            + "\n[[package]]\nname = 'demo'\nversion = '1.0'\nsource = { registry = 'https://pypi.org/simple' }\n",
            encoding="utf-8",
        )
        try:
            active_rows(tomllib.loads(duplicate_lock.read_text(encoding="utf-8")))
        except SystemExit as exc:
            assert "duplicate" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted an ambiguous duplicate")
        missing_size_lock = root / "missing-size.lock"
        missing_size_lock.write_text(
            lock.read_text(encoding="utf-8").replace(f", size = {len(wheel)}", ""),
            encoding="utf-8",
        )
        missing_size_output = root / "missing-size.json"
        try:
            audit(missing_size_lock, artifacts, missing_size_output)
        except SystemExit as exc:
            assert "malformed" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test accepted a missing wheel size")
        assert not missing_size_output.exists()
        try:
            audit(lock, artifacts, candidate)
        except SystemExit as exc:
            assert "output already exists" in str(exc)
        else:
            raise SystemExit("bigvgan closure self-test overwrote an existing output")
        race_output = root / "race.json"
        real_link = os.link

        def create_sentinel(source: str | bytes, destination: str | bytes, *args: Any, **kwargs: Any) -> None:
            Path(destination).write_text("sentinel\n", encoding="utf-8")
            real_link(source, destination, *args, **kwargs)

        os.link = create_sentinel
        try:
            atomic_write_json(race_output, {"unexpected": True})
        except SystemExit as exc:
            assert "concurrently" in str(exc)
        finally:
            os.link = real_link
        assert race_output.read_text(encoding="utf-8") == "sentinel\n"
        assert not list(root.glob(".race.json.*.tmp"))
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
