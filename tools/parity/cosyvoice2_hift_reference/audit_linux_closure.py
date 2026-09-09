#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Offline, fail-closed audit of the authenticated Linux wheel closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import tomllib

LINUX_MARKER = "platform_machine == 'x86_64' and sys_platform == 'linux'"
REGISTRIES = {"https://pypi.org/simple": "files.pythonhosted.org", "https://download.pytorch.org/whl/cpu": "download-r2.pytorch.org"}
EXPECTED_NAMES = {"cosyvoice2-hift-reference", "filelock", "fsspec", "jinja2", "markupsafe", "mpmath", "networkx", "numpy", "scipy", "setuptools", "sympy", "torch", "typing-extensions"}
EXPECTED_LOCK_MANIFEST = {
    "constraints": [{"name": "setuptools", "specifier": ">=83.0.0"}],
}
TORCH_LINUX_URL = "https://download-r2.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
TORCH_LINUX_HASH = "sha256:8f8b3cfc53010a4b4a3c7ecb88c212e9decc4f5eeb6af75c3c803937d2d60947"
TORCH_LINUX_SIZE = 175_833_687
TORCH_LINUX_UPLOAD_TIME = "2025-06-03T18:27:57Z"
NATIVE_SUFFIXES = (".so", ".dylib", ".dll", ".a")
NATIVE_MAGICS = (b"\x7fELF", b"!<arch>\n", b"MZ")
FORBIDDEN = ("nvidia", "triton", "librosa", "soxr", "soundfile")
FORBIDDEN_EXACT_NAMES = {"triton", "librosa", "soxr", "soundfile"}
LICENSE_NAMES = ("license", "copying", "notice", "copyright")
MAX_MEMBER = 512 * 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024 * 1024
MAX_RATIO = 1000
MAX_READ = 16 * 1024 * 1024
MULTI_VALUE_METADATA_HEADERS = {
    "classifier",
    "dynamic",
    "license-file",
    "obsoletes-dist",
    "platform",
    "project-url",
    "provides-dist",
    "provides-extra",
    "requires-dist",
    "requires-external",
}


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"cosyvoice2 HiFT Linux closure audit: BLOCKED: {message}")


def require_vast_linux() -> None:
    if (
        os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1"
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        fail("closure audit requires VOKRA_PUBLISH_ON_VAST=1 on Linux x86_64")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def forbidden_dependency(name: str) -> bool:
    normalized = normalize_package_name(name)
    return normalized in FORBIDDEN_EXACT_NAMES or normalized.startswith("nvidia-")


def load_lock(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail("uv.lock missing or symlinked")
    try:
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        fail(f"uv.lock unreadable: {error}")
    if (
        set(lock)
        != {
            "version",
            "revision",
            "requires-python",
            "resolution-markers",
            "supported-markers",
            "manifest",
            "package",
        }
        or lock["version"] != 1
        or lock["revision"] != 3
        or lock["requires-python"] != "==3.12.*"
    ):
        fail("uv.lock top-level schema drifted")
    if lock["manifest"] != EXPECTED_LOCK_MANIFEST:
        fail("uv.lock manifest constraints drifted")
    if lock["resolution-markers"] != [LINUX_MARKER] or lock["supported-markers"] != [LINUX_MARKER]:
        fail("uv.lock Linux marker drifted")
    return lock


def active_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list):
        fail("uv.lock package table missing")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in packages:
        if not isinstance(row, dict) or set(row) - {"name", "version", "source", "dependencies", "resolution-markers", "sdist", "wheels", "metadata"}:
            fail("unknown package row field")
        if not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            fail("package identity malformed")
        source = row.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"virtual"}, {"registry"}):
            fail("package source malformed")
        key = (row["name"], row["version"])
        if key in rows:
            fail(f"duplicate package identity: {key}")
        rows[key] = row
        if "registry" in source and source["registry"] not in REGISTRIES:
            fail(f"unapproved registry: {source['registry']}")
    if {row["name"] for row in rows.values()} != EXPECTED_NAMES or len(rows) != 13:
        fail("active closure is not the exact 13-row HiFT closure")
    virtual = [row for row in rows.values() if "virtual" in row["source"]]
    if len(virtual) != 1 or virtual[0]["source"]["virtual"] != ".":
        fail("missing repository-root virtual row")
    for row in rows.values():
        if forbidden_dependency(row["name"]):
            fail(f"forbidden package identity: {row['name']}")
        deps = row.get("dependencies", [])
        if not isinstance(deps, list):
            fail(f"{row['name']} dependencies malformed")
        for dep in deps:
            if (
                not isinstance(dep, dict)
                or set(dep) != {"name", "marker"}
                or not isinstance(dep["name"], str)
                or not isinstance(dep["marker"], str)
            ):
                fail(f"{row['name']} dependency schema drifted")
            if dep["marker"] != LINUX_MARKER:
                fail(f"{row['name']} dependency marker is not the pinned Linux marker")
            if normalize_package_name(dep["name"]) not in {normalize_package_name(name) for name in EXPECTED_NAMES}:
                fail(f"{row['name']} dependency does not resolve to the exact closure")
            if forbidden_dependency(dep["name"]):
                fail(f"forbidden dependency: {dep['name']}")
    return sorted(rows.values(), key=lambda row: (row["name"], row["version"]))


def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {host for host in REGISTRIES.values()}
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
    ):
        fail(f"wheel URL host is not allowlisted: {url!r}")
    name = unquote(parsed.path.rsplit("/", 1)[-1])
    if (
        not name.endswith(".whl")
        or not name
        or name != Path(name).name
        or any(c in name for c in "\\\x00")
        or any(unicodedata.category(c) == "Cc" for c in name)
        or name in {".", ".."}
    ):
        fail(f"unsafe wheel filename: {url!r}")
    return name


def wheel_compatible(name: str) -> bool:
    parts = name[:-4].rsplit("-", 3) if name.endswith(".whl") else []
    if len(parts) != 4:
        return False
    _, py, abi, platform = parts
    if (py, abi, platform) == ("py3", "none", "any"):
        return True
    return py == abi == "cp312" and all(tag.startswith("manylinux") and tag.endswith("_x86_64") for tag in platform.split("."))


def locked_artifact(row: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
    wheels = row.get("wheels")
    if not isinstance(wheels, list):
        fail(f"{row['name']} has no wheels")
    candidates = []
    for artifact in wheels:
        authenticated_torch = (
            row.get("name") == "torch"
            and row.get("version") == "2.7.1+cpu"
            and row.get("source", {}).get("registry") == "https://download.pytorch.org/whl/cpu"
        )
        if not isinstance(artifact, dict) or set(artifact) not in (
            {"url", "hash", "size", "upload-time"},
            {"url", "hash", "upload-time"},
        ) or ("size" not in artifact and not authenticated_torch):
            fail(f"{row['name']} artifact schema drifted")
        url = artifact["url"]
        name = safe_filename(url)
        if urlparse(url).hostname != REGISTRIES[row["source"]["registry"]]:
            fail(f"{row['name']} artifact host does not match its registry")
        if authenticated_torch and (
            artifact.get("url") != TORCH_LINUX_URL
            or artifact.get("hash") != TORCH_LINUX_HASH
            or artifact.get("upload-time") != TORCH_LINUX_UPLOAD_TIME
            or ("size" in artifact and artifact.get("size") != TORCH_LINUX_SIZE)
        ):
            fail("authenticated Linux torch artifact drifted")
        if wheel_compatible(name):
            candidates.append((name, artifact))
    if not candidates:
        fail(f"{row['name']} has no Linux cp312-compatible wheel")
    candidates.sort(key=lambda item: (0 if "cp312" in item[0] else 1, item[0]))
    name, artifact = candidates[0]
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", artifact["hash"])
    if match is None:
        fail(f"{row['name']} artifact hash/size malformed")
    if "size" in artifact:
        if isinstance(artifact["size"], bool) or not isinstance(artifact["size"], int) or artifact["size"] <= 0:
            fail(f"{row['name']} artifact hash/size malformed")
        size = artifact["size"]
    elif row.get("name") == "torch" and row.get("version") == "2.7.1+cpu":
        size = TORCH_LINUX_SIZE
    else:
        fail(f"{row['name']} artifact hash/size malformed")
    return artifact, name, size


def normalize_member(name: str) -> str:
    if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(part == ".." for part in name.split("/")):
        fail(f"unsafe archive member path: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        fail(f"unsafe archive member path: {name!r}")
    return normalized


def member_identity(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def stream_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int, str, bytes]:
    name = normalize_member(info.filename)
    mode = (info.external_attr >> 16) & 0o170000
    if info.is_dir():
        if mode not in (0, stat.S_IFDIR):
            fail(f"directory has a non-directory mode: {name}")
        return name, 0, digest_bytes(b""), b""
    if mode == stat.S_IFLNK or mode not in (0, stat.S_IFREG):
        fail(f"non-regular archive member is not allowed: {name}")
    if info.file_size > MAX_MEMBER or (
        info.compress_size and info.file_size > info.compress_size * MAX_RATIO
    ):
        fail(f"archive expansion limit exceeded: {name}")
    digest = hashlib.sha256()
    captured = bytearray()
    total = 0
    with archive.open(info, "r") as stream:
        while True:
            chunk = stream.read(min(1 << 20, MAX_MEMBER - total + 1))
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
            if len(captured) < MAX_READ:
                captured.extend(chunk[: MAX_READ - len(captured)])
            if total > MAX_MEMBER:
                fail(f"member exceeded limit while reading: {name}")
    if total != info.file_size:
        fail(f"member size changed while reading: {name}")
    return name, total, digest.hexdigest(), bytes(captured)


def parse_metadata_headers(data: bytes) -> tuple[dict[str, Any], list[str]]:
    """Parse only the RFC 822-style Core Metadata header block.

    A blank line terminates the header block; everything after it is the
    Description body and must not be interpreted as headers. Continuation
    lines are unfolded onto the preceding field before field multiplicity is
    checked.
    """

    metadata: dict[str, Any] = {}
    requires_dist: list[str] = []
    current_key: str | None = None
    current_value: str | None = None

    def commit() -> None:
        if current_key is None or current_value is None:
            return
        if current_key in MULTI_VALUE_METADATA_HEADERS:
            metadata.setdefault(current_key, []).append(current_value)
        elif current_key in metadata:
            fail(f"duplicate METADATA {current_key} header")
        else:
            metadata[current_key] = current_value
        if current_key == "requires-dist":
            requires_dist.append(current_value)

    for line in data.decode("utf-8", "replace").splitlines():
        if not line:
            break
        if line.startswith((" ", "\t")):
            if current_key is None or current_value is None:
                fail("METADATA continuation has no preceding header")
            current_value += " " + line.strip()
            continue
        if ":" not in line:
            fail(f"malformed METADATA header: {line!r}")
        key, value = line.split(":", 1)
        key = key.casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key):
            fail(f"malformed METADATA field name: {key!r}")
        commit()
        current_key = key
        current_value = value.strip()
    commit()
    return metadata, requires_dist


def inspect_wheel(path: Path, expected_name: str, expected_version: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"wheel missing or symlinked: {path}")
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        fail(f"wheel is not a valid zip: {error}")
    names: set[str] = set()
    canonical_names: set[str] = set()
    total = 0
    licenses = []
    native = []
    suspicious = []
    metadata_members = 0
    metadata_candidate: tuple[int, bytes] | None = None
    with archive:
        for info in archive.infolist():
            name = normalize_member(info.filename)
            canonical_name = member_identity(name)
            if name in names or canonical_name in canonical_names:
                fail(f"duplicate archive member: {name}")
            names.add(name)
            canonical_names.add(canonical_name)
            name, size, digest, captured = stream_member(archive, info)
            total += size
            if total > MAX_TOTAL:
                fail("wheel total expansion limit exceeded")
            lower = name.casefold()
            metadata_parts = name.split("/")
            is_metadata = (
                len(metadata_parts) >= 2
                and metadata_parts[-1].casefold() == "metadata"
                and metadata_parts[-2].casefold().endswith(".dist-info")
            )
            is_top_level_metadata = (
                is_metadata
                and len(metadata_parts) == 2
                and metadata_parts[0].casefold().endswith(".dist-info")
            )
            if is_top_level_metadata:
                metadata_members += 1
                if size > MAX_READ:
                    fail("METADATA exceeds bounded read limit")
                if metadata_candidate is None:
                    metadata_candidate = (size, captured)
            if not info.is_dir():
                text = captured[:MAX_READ].decode("utf-8", "ignore").casefold()
                for marker in ("cuda", "nvidia", "triton", "librosa", "soxr", "soundfile"):
                    if marker in lower or marker in text:
                        suspicious.append({"path": name, "marker": marker, "kind": "owner_review_suspicion"})
                if Path(name).name.casefold().startswith(LICENSE_NAMES) or any(Path(name).name.casefold().startswith(prefix) for prefix in LICENSE_NAMES):
                    licenses.append({"path": name, "bytes": size, "sha256": digest})
                if lower.endswith(NATIVE_SUFFIXES) or any(captured.startswith(magic) for magic in NATIVE_MAGICS):
                    native.append({"path": name, "bytes": size, "sha256": digest})
    if metadata_members != 1 or metadata_candidate is None:
        fail(f"wheel must contain exactly one .dist-info/METADATA (found {metadata_members})")
    _, metadata_bytes = metadata_candidate
    metadata, requires_dist = parse_metadata_headers(metadata_bytes)
    if (
        normalize_package_name(metadata.get("name", ""))
        != normalize_package_name(expected_name)
        or metadata.get("version") != expected_version
    ):
        fail(f"METADATA identity mismatch for {path.name}")
    for requirement in requires_dist:
        match = re.match(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)", requirement)
        if match is None:
            fail(f"malformed Requires-Dist entry: {requirement!r}")
        if forbidden_dependency(match.group(1)):
            fail(f"forbidden Requires-Dist dependency: {match.group(1)}")
    return {"name": expected_name, "version": expected_version, "wheel": path.name, "members": len(names), "expanded_bytes": total, "metadata": metadata, "requires_dist": sorted(requires_dist), "license_notice": sorted(licenses, key=lambda x: x["path"]), "native": sorted(native, key=lambda x: x["path"]), "suspicious_markers": sorted(suspicious, key=lambda x: (x["path"], x["marker"]))}


def audit(lock_path: Path, project_path: Path, artifacts: Path, output: Path) -> None:
    lock = load_lock(lock_path)
    rows = active_rows(lock)
    if project_path.is_symlink() or not project_path.is_file():
        fail("pyproject missing or symlinked")
    if artifacts.is_symlink() or not artifacts.is_dir():
        fail("wheel staging directory is missing or symlinked")
    if output.parent.is_symlink() or not output.parent.is_dir():
        fail("candidate parent is missing or symlinked")
    project_sha = sha256_file(project_path)
    lock_sha = sha256_file(lock_path)
    if output.exists() or output.is_symlink():
        fail(f"candidate output already exists: {output}")
    inventory = []
    aggregate = []
    expected_artifacts = {}
    for row in rows:
        if "registry" not in row["source"]:
            continue
        artifact, filename, expected_size = locked_artifact(row)
        expected_artifacts[filename] = (row, artifact, expected_size)
    actual_names = {entry.name for entry in artifacts.iterdir()}
    if actual_names != set(expected_artifacts):
        fail("staging directory contains missing or unexpected wheel names")
    for filename, (row, artifact, expected_size) in expected_artifacts.items():
        wheel = artifacts / filename
        if wheel.is_symlink() or not wheel.is_file():
            fail(f"staged wheel is not a regular file: {filename}")
        if wheel.stat().st_size != expected_size or sha256_file(wheel) != artifact["hash"].removeprefix("sha256:"):
            fail(f"staged wheel does not match lock: {filename}")
        record = inspect_wheel(wheel, row["name"], row["version"])
        record.update(
            {
                "url": artifact["url"],
                "sha256": artifact["hash"].removeprefix("sha256:"),
                "bytes": expected_size,
            }
        )
        inventory.append(record)
        aggregate.append(record)
    payload = {
        "format": "vokra-cosyvoice2-hift-linux-closure-candidate-v1",
        "status": "OWNER_REVIEW_REQUIRED",
        "license_status": "PENDING_PACKAGE_AND_NATIVE_PAYLOAD_REVIEW",
        "publication": "NO_UPLOAD",
        "project_sha256": project_sha,
        "uv_lock_sha256": lock_sha,
        "package_rows": rows,
        "wheels": inventory,
        "archive_aggregate_sha256": canonical(aggregate),
        "blockers": ["Owner must review every package license and native/bundled payload; no approval is inferred."],
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    except FileExistsError:
        fail("candidate appeared concurrently; refusing overwrite")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    assert Path(__file__).with_name("uv.lock").is_file()
    assert Path(__file__).with_name("pyproject.toml").is_file()
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-hift-audit-") as temp:
        root = Path(temp)
        tampered_lock = root / "tampered-uv.lock"
        tampered_lock.write_text(
            Path(__file__).with_name("uv.lock")
            .read_text(encoding="utf-8")
            .replace('specifier = ">=83.0.0"', 'specifier = ">=84.0.0"', 1),
            encoding="utf-8",
        )
        try:
            load_lock(tampered_lock)
        except SystemExit as error:
            assert "manifest constraints" in str(error)
        else:
            raise AssertionError("uv.lock manifest constraint tamper was accepted")
        wheel = root / "demo-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("demo-1.0.dist-info/", b"")
            archive.writestr(
                "demo-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\n"
                "Name: Demo_Pkg\n"
                "Version: 1.0\n"
                "Classifier: License :: OSI Approved :: MIT License\n"
                "Classifier: Programming Language :: Python :: 3\n"
                "Requires-Dist: packaging>=20\n"
                "Requires-Dist: wheel\n"
                "License: MIT\n"
                "Description: first line\n"
                " second line\n"
                "\n"
                "body: this colon is not a header\n",
            )
            archive.writestr("LICENSE", "MIT\n")
        record = inspect_wheel(wheel, "demo-pkg", "1.0")
        assert record["license_notice"][0]["sha256"] == digest_bytes(b"MIT\n")
        assert record["metadata"]["classifier"] == [
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
        ]
        assert record["requires_dist"] == ["packaging>=20", "wheel"]
        assert record["metadata"]["description"] == "first line second line"
        nested_metadata = root / "nested-metadata-1.0-py3-none-any.whl"
        with zipfile.ZipFile(nested_metadata, "w") as archive:
            archive.writestr(
                "setuptools-78.1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: setuptools\nVersion: 78.1.0\n",
            )
            archive.writestr(
                "setuptools/_vendor/more_itertools-10.8.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: more-itertools\nVersion: 10.8.0\n",
            )
        nested_record = inspect_wheel(nested_metadata, "setuptools", "78.1.0")
        assert nested_record["metadata"]["name"] == "setuptools"
        license_directory = root / "license-directory-1.0-py3-none-any.whl"
        with zipfile.ZipFile(license_directory, "w") as archive:
            archive.writestr(
                "markupsafe-3.0.3.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: markupsafe\nVersion: 3.0.3\n",
            )
            archive.writestr("markupsafe-3.0.3.dist-info/licenses/", b"")
        directory_record = inspect_wheel(license_directory, "markupsafe", "3.0.3")
        assert directory_record["license_notice"] == []
        duplicate_top_level = root / "duplicate-top-level-1.0-py3-none-any.whl"
        with zipfile.ZipFile(duplicate_top_level, "w") as archive:
            for directory in ("first-1.0.dist-info", "second-1.0.dist-info"):
                archive.writestr(
                    f"{directory}/METADATA",
                    "Metadata-Version: 2.1\nName: duplicate\nVersion: 1.0\n",
                )
        try:
            inspect_wheel(duplicate_top_level, "duplicate", "1.0")
        except SystemExit as error:
            assert "exactly one" in str(error)
        else:
            raise AssertionError("multiple top-level METADATA files accepted")
        duplicate_header = root / "duplicate-header-1.0-py3-none-any.whl"
        with zipfile.ZipFile(duplicate_header, "w") as archive:
            archive.writestr(
                "duplicate-header-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\n"
                "Name: duplicate-header\n"
                "Name: duplicate-header\n"
                "Version: 1.0\n",
            )
        try:
            inspect_wheel(duplicate_header, "duplicate-header", "1.0")
        except SystemExit as error:
            assert "duplicate METADATA name header" in str(error)
        else:
            raise AssertionError("duplicate singleton METADATA header accepted")
        for bad in ("../x", "/x", "a\\x"):
            try:
                normalize_member(bad)
            except SystemExit:
                pass
            else:
                raise AssertionError("unsafe archive path accepted")
        assert member_identity("Demo/data") == member_identity("demo/data")
        assert member_identity("A\u030a/data") == member_identity("\u00c5/data")
        collision = root / "collision-1.0-py3-none-any.whl"
        with zipfile.ZipFile(collision, "w") as archive:
            archive.writestr("collision-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: collision\nVersion: 1.0\n")
            archive.writestr("A\u030a.txt", b"a")
            archive.writestr("\u00c5.txt", b"b")
        try:
            inspect_wheel(collision, "collision", "1.0")
        except SystemExit as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("NFC/casefold collision accepted")
        for mode, label in ((stat.S_IFLNK, "symlink"), (stat.S_IFIFO, "special")):
            unsafe = root / f"{label}-1.0-py3-none-any.whl"
            member = zipfile.ZipInfo(f"{label}-1.0.dist-info/METADATA")
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr(member, f"Metadata-Version: 2.1\nName: {label}\nVersion: 1.0\n")
                special = zipfile.ZipInfo("payload")
                special.external_attr = (mode | 0o644) << 16
                archive.writestr(special, b"x")
            try:
                inspect_wheel(unsafe, label, "1.0")
            except SystemExit as error:
                assert label in str(error) or "non-regular" in str(error)
            else:
                raise AssertionError(f"{label} archive member accepted")
        native = root / "native-1.0-py3-none-any.whl"
        with zipfile.ZipFile(native, "w") as archive:
            archive.writestr("native-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: native\nVersion: 1.0\n")
            archive.writestr("native.so", b"\x7fELF")
        native_record = inspect_wheel(native, "native", "1.0")
        assert native_record["native"][0]["path"] == "native.so"
        forbidden = root / "forbidden-1.0-py3-none-any.whl"
        with zipfile.ZipFile(forbidden, "w") as archive:
            archive.writestr("forbidden-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: forbidden\nVersion: 1.0\nRequires-Dist: nvidia-cuda\n")
        try:
            inspect_wheel(forbidden, "forbidden", "1.0")
        except SystemExit as error:
            assert "forbidden" in str(error)
        else:
            raise AssertionError("forbidden dependency marker accepted")
    print("cosyvoice2_hift Linux closure audit self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("uv.lock"))
    parser.add_argument("--project", type=Path, default=Path(__file__).with_name("pyproject.toml"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.artifacts or not args.output:
            parser.error("--artifacts and --output are required")
        require_vast_linux()
        audit(args.lock, args.project, args.artifacts, args.output)
        return 0
    except (OSError, UnicodeError, zipfile.BadZipFile, SystemExit) as error:
        if isinstance(error, SystemExit):
            raise
        fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
