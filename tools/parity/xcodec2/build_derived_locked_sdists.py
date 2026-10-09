#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Build bounded, first-party derived wheels from two locked pure-Python sdists.

This is deliberately not a setuptools replacement and never executes setup.py,
a PEP 517 backend, package code, or third-party code.  It accepts only the two
locked XCodec2 parity sdists whose source layouts are explicitly mapped below.
The resulting wheel is a derived artifact, not an official upstream wheel and
does not grant license, publication, or model-execution approval.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import email.parser
import email.policy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import zipfile
from typing import Any

import inspect_locked_sdist_sources as inspector


SUPPORTED = {
    "antlr4-python3-runtime==4.9.3",
    "xcodec2==0.1.5",
}
PINNED_ARTIFACTS = {
    "antlr4-python3-runtime==4.9.3": {
        "url": "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz",
        "bytes": 117034,
        "sha256": "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
    },
    "xcodec2==0.1.5": {
        "url": "https://files.pythonhosted.org/packages/80/69/5b99cc4de97f861d6b0a9acd20b4a22f44eabcc20051c0991b1cca479138/xcodec2-0.1.5.tar.gz",
        "bytes": 22329,
        "sha256": "dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b",
    },
}
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
SCRIPT_NAME = "vokra-derived-locked-sdists/1"
WHEEL_TAG = "py3-none-any"


class BuildError(ValueError):
    """A fail-closed derived-wheel construction error."""


def _canonical(path: Path) -> None:
    try:
        inspector.canonical_path(path)
    except inspector.InspectionError as exc:
        raise BuildError(str(exc)) from exc


def _regular_file(path: Path) -> None:
    _canonical(path)
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"not a regular file: {path}")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    _regular_file(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            if size > MAX_OUTPUT_BYTES:
                raise BuildError("file exceeds bounded size")
            digest.update(chunk)
    return size, digest.hexdigest()


def _bounded_read(path: Path, limit: int) -> bytes:
    _regular_file(path)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise BuildError(f"file exceeds bounded size: {path}")
    return data


def _safe_output_dir(path: Path) -> None:
    _canonical(path)
    if path.is_symlink() or not path.is_dir():
        raise BuildError(f"output directory is not an existing real directory: {path}")


def _read_archive(path: Path, expected: dict[str, Any]) -> tuple[str, dict[str, bytes], set[str]]:
    """Read and authenticate a complete bounded tar member map."""

    try:
        inspector.regular_archive(path)
        size, digest = inspector.sha256_file(path)
    except inspector.InspectionError as exc:
        raise BuildError(str(exc)) from exc
    if size != expected["bytes"] or digest != expected["sha256"]:
        raise BuildError(f"archive does not match lock: {path}")

    files: dict[str, bytes] = {}
    directories: set[str] = set()
    seen: set[str] = set()
    root: str | None = None
    total = 0
    with tarfile.open(path, mode="r:*") as archive:
        for index, info in enumerate(archive, start=1):
            if index > inspector.MAX_ARCHIVE_MEMBERS:
                raise BuildError("archive member count exceeds bound")
            try:
                name = inspector.safe_member(info.name)
            except inspector.InspectionError as exc:
                raise BuildError(str(exc)) from exc
            if name in seen:
                raise BuildError(f"duplicate archive member: {name}")
            seen.add(name)
            parts = PurePosixPath(name).parts
            if not parts:
                raise BuildError("empty archive member")
            if root is None:
                root = parts[0]
            elif parts[0] != root:
                raise BuildError("archive has multiple top-level roots")
            if info.isdir():
                directories.add("/".join(parts[1:]))
                continue
            if info.issym() or info.islnk() or info.isdev() or info.isfifo() or not info.isfile():
                raise BuildError(f"archive link or special member: {name}")
            if info.size < 0 or info.size > inspector.MAX_MEMBER_BYTES:
                raise BuildError(f"archive member exceeds bound: {name}")
            relative = "/".join(parts[1:])
            if not relative:
                raise BuildError("archive root is a file")
            parent_parts = PurePosixPath(relative).parts[:-1]
            for end in range(1, len(parent_parts) + 1):
                parent = "/".join(parent_parts[:end])
                if parent in files:
                    raise BuildError(f"file-directory collision: {parent}")
            total += info.size
            if total > inspector.MAX_TOTAL_MEMBER_BYTES:
                raise BuildError("archive uncompressed member bound exceeded")
            stream = archive.extractfile(info)
            if stream is None:
                raise BuildError(f"archive member cannot be read: {name}")
            body = stream.read(info.size + 1)
            stream.close()
            if len(body) != info.size:
                raise BuildError(f"archive member size changed: {name}")
            files[relative] = body
    if root is None:
        raise BuildError("archive is empty")
    collisions = set(files) & directories
    if collisions:
        raise BuildError(f"file-directory collision: {sorted(collisions)}")
    for relative in files:
        parent_parts = PurePosixPath(relative).parts[:-1]
        for end in range(1, len(parent_parts) + 1):
            parent = "/".join(parent_parts[:end])
            if parent not in directories:
                raise BuildError(f"missing explicit directory member: {parent}")
    return root, files, directories


def _single_header(message: email.message.Message, name: str) -> str:
    values = message.get_all(name, [])
    if len(values) != 1 or not isinstance(values[0], str) or not values[0].strip():
        raise BuildError(f"PKG-INFO requires exactly one {name} header")
    return values[0].strip()


def _normal_req(value: str) -> str:
    return re.sub(r"[\s'\"]+", "", value)


def _active_requirement(value: str) -> str | None:
    requirement, separator, marker = value.partition(";")
    if not separator:
        return _normal_req(requirement)
    match = re.fullmatch(
        r"\s*(python_version|python_full_version)\s*(==|!=|<=|>=|<|>)\s*['\"]?(\d+(?:\.\d+)*)['\"]?\s*",
        marker,
    )
    if match is None:
        raise BuildError(f"unsupported environment marker in setup.py: {marker.strip()}")
    field, operator, raw_version = match.groups()
    del field  # Both supported fields are intentionally bound to target CPython 3.12.
    target = (3, 12)
    observed = tuple(int(part) for part in raw_version.split("."))
    width = max(len(target), len(observed))
    left = target + (0,) * (width - len(target))
    right = observed + (0,) * (width - len(observed))
    active = {
        "==": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }[operator]
    return _normal_req(requirement) if active else None


def _setup_call(body: bytes) -> dict[str, Any]:
    try:
        text = body.decode("utf-8")
        tree = ast.parse(text, filename="setup.py")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise BuildError("setup.py is not valid UTF-8 Python") from exc
    findings = inspector._ast_findings("setup.py", body)
    if findings:
        raise BuildError("; ".join(sorted(set(findings))))
    calls = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup"
    ]
    if len(calls) != 1:
        raise BuildError("setup.py must contain exactly one setup call")
    call = calls[0]
    values: dict[str, Any] = {}
    seen_keywords: set[str] = set()
    for keyword in call.keywords:
        if keyword.arg is None:
            raise BuildError("setup.py **kwargs are not supported")
        if keyword.arg in seen_keywords:
            raise BuildError(f"duplicate setup keyword: {keyword.arg}")
        seen_keywords.add(keyword.arg)
        if keyword.arg == "packages" and isinstance(keyword.value, ast.Call):
            if not isinstance(keyword.value.func, ast.Name) or keyword.value.func.id != "find_packages":
                raise BuildError("unsupported package discovery call")
            if keyword.value.args or len(keyword.value.keywords) != 1 or keyword.value.keywords[0].arg != "exclude":
                raise BuildError("find_packages requires only a literal exclude")
            try:
                excludes = ast.literal_eval(keyword.value.keywords[0].value)
            except (ValueError, SyntaxError) as exc:
                raise BuildError("find_packages exclude is not literal") from exc
            if not isinstance(excludes, (list, tuple)) or any(not isinstance(item, str) for item in excludes):
                raise BuildError("find_packages exclude is malformed")
            values[keyword.arg] = {"kind": "find_packages", "exclude": list(excludes)}
            continue
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError) as exc:
            raise BuildError(f"setup keyword is not literal: {keyword.arg}") from exc
    for required in ("name", "version"):
        if not isinstance(values.get(required), str) or not values[required]:
            raise BuildError(f"setup.py {required} is missing or non-literal")
    return values


def _metadata_identity(pkg_info: bytes, setup: dict[str, Any], identity: str) -> dict[str, Any]:
    message = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(pkg_info)
    name = _single_header(message, "Name")
    version = _single_header(message, "Version")
    expected_name, expected_version = identity.split("==", 1)
    if name != expected_name or version != expected_version:
        raise BuildError(f"PKG-INFO identity mismatch for {identity}")
    if setup["name"] != name or setup["version"] != version:
        raise BuildError(f"setup.py identity mismatch for {identity}")
    fields = {
        "description": "Summary",
        "url": "Home-page",
        "author": "Author",
        "author_email": "Author-email",
        "python_requires": "Requires-Python",
        "license": "License",
    }
    for setup_key, metadata_key in fields.items():
        if setup_key in setup and isinstance(setup[setup_key], str):
            observed = message.get(metadata_key)
            if observed is not None and observed.strip() != setup[setup_key].strip():
                raise BuildError(f"setup.py/{metadata_key} mismatch for {identity}")
    classifiers = setup.get("classifiers")
    if classifiers is not None:
        if not isinstance(classifiers, (list, tuple)) or any(not isinstance(item, str) for item in classifiers):
            raise BuildError("setup classifiers are malformed")
        observed = message.get_all("Classifier", [])
        if list(classifiers) != observed:
            raise BuildError(f"classifier metadata mismatch for {identity}")
    install_requires = setup.get("install_requires")
    if install_requires is not None:
        if not isinstance(install_requires, (list, tuple)) or any(not isinstance(item, str) for item in install_requires):
            raise BuildError("setup install_requires are malformed")
        observed = message.get_all("Requires-Dist", [])
        expected_requirements = sorted(
            requirement
            for item in install_requires
            if (requirement := _active_requirement(item)) is not None
        )
        if expected_requirements != sorted(map(_normal_req, observed)):
            raise BuildError(f"Requires-Dist metadata mismatch for {identity}")
    return {
        "name": name,
        "version": version,
        "metadata_sha256": _hash_bytes(pkg_info),
        "metadata_bytes": len(pkg_info),
    }


def _normalized_name(name: str) -> str:
    normalized = re.sub(r"[-_.]+", "_", name)
    if not re.fullmatch(r"[A-Za-z0-9_]+", normalized):
        raise BuildError(f"distribution name is unsafe: {name}")
    return normalized


def _license_sources(files: dict[str, bytes]) -> list[str]:
    rows = []
    for path in sorted(files):
        if ".egg-info/" in path:
            continue
        basename = PurePosixPath(path).name
        if re.fullmatch(r"(?i)(license|copying|notice)(?:[._-].*)?", basename):
            rows.append(path)
    return rows


def _package_mapping(identity: str, root: str, files: dict[str, bytes], setup: dict[str, Any], dist_info: str) -> tuple[dict[str, bytes], list[dict[str, Any]], list[str]]:
    mapping: dict[str, bytes] = {}
    source_rows: list[dict[str, Any]] = []
    data_rows: list[str] = []
    consumed: set[str] = set()

    def add(source: str, target: str, *, data: bool = False) -> None:
        if target in mapping:
            raise BuildError(f"wheel path collision: {target}")
        if target.startswith("/") or ".." in PurePosixPath(target).parts or not target:
            raise BuildError(f"unsafe wheel path: {target}")
        body = files[source]
        mapping[target] = body
        consumed.add(source)
        source_rows.append({"source": source, "target": target, "bytes": len(body), "sha256": _hash_bytes(body)})
        if data:
            data_rows.append(target)

    if identity.startswith("antlr4-python3-runtime=="):
        package_value = setup.get("packages")
        package_names = package_value if isinstance(package_value, list) else None
        if package_names is None or any(not isinstance(item, str) for item in package_names):
            raise BuildError("ANTLR packages must be an explicit string list")
        if setup.get("package_dir") != {"": "src"}:
            raise BuildError("ANTLR package_dir mapping is not the authenticated expected mapping")
        package_prefixes = {item.replace(".", "/") for item in package_names}
        discovered: set[str] = set()
        for path, body in files.items():
            if path.startswith("src/") and path.endswith(".py"):
                relative = path[4:]
                package_path = relative.rsplit("/", 1)[0] if "/" in relative else ""
                if package_path not in package_prefixes:
                    raise BuildError(f"ANTLR source is outside declared package: {path}")
                discovered.add(package_path.replace("/", "."))
                add(path, relative)
        if discovered != set(package_names):
            raise BuildError("ANTLR package list does not match source package directories")
        scripts = setup.get("scripts")
        if scripts != ["bin/pygrun"]:
            raise BuildError("ANTLR script mapping is not the authenticated expected mapping")
        if "bin/pygrun" not in files:
            raise BuildError("ANTLR script is missing")
        add("bin/pygrun", f"{_normalized_name(setup['name'])}-{setup['version']}.data/scripts/pygrun", data=True)
    elif identity == "xcodec2==0.1.5":
        package_value = setup.get("packages")
        if not isinstance(package_value, dict) or package_value.get("kind") != "find_packages" or package_value.get("exclude") != ["tests*", "docs*"]:
            raise BuildError("XCodec2 package discovery is not the authenticated expected mapping")
        if "package_dir" in setup:
            raise BuildError("XCodec2 package_dir must be absent for the authenticated mapping")
        if "xcodec2/__init__.py" not in files:
            raise BuildError("XCodec2 root package lacks __init__.py")
        package_dirs = {"xcodec2"}
        for path in files:
            if path.startswith("xcodec2/") and path.endswith("/__init__.py"):
                package_dirs.add(path[:-12].rstrip("/"))
        for package_dir in package_dirs:
            parents = PurePosixPath(package_dir).parts
            for index in range(1, len(parents)):
                if "/".join(parents[:index]) not in package_dirs:
                    raise BuildError(f"XCodec2 package parent lacks __init__.py: {package_dir}")
        for path in files:
            if path.startswith("xcodec2/") and path.endswith(".py"):
                package_dir = path.rsplit("/", 1)[0]
                if package_dir not in package_dirs:
                    raise BuildError(f"XCodec2 source is outside discovered package: {path}")
                add(path, path)
        if not any(path.startswith("xcodec2/") for path in mapping):
            raise BuildError("XCodec2 package source is missing")
    else:
        raise BuildError(f"unsupported locked sdist: {identity}")

    for source in _license_sources(files):
        add(source, f"{dist_info}/licenses/{source}")
    ignored = {"PKG-INFO", "setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
    ignored.update(path for path in files if ".egg-info/" in path)
    ignored.update(
        path
        for path in files
        if "/" not in path and re.fullmatch(r"(?i)(readme|release)(?:[._-].*)?", path)
    )
    ignored.update(_license_sources(files))
    unaccounted = sorted(set(files) - consumed - ignored)
    if unaccounted:
        raise BuildError(f"unmapped source payload is not allowed: {unaccounted[:4]}")
    return mapping, source_rows, data_rows


def _zip_bytes(mapping: dict[str, bytes], metadata: bytes, dist_info: str) -> tuple[bytes, list[dict[str, Any]]]:
    if not metadata.endswith(b"\n"):
        metadata += b"\n"
    wheel_body = (
        "Wheel-Version: 1.0\n"
        f"Generator: {SCRIPT_NAME}\n"
        "Root-Is-Purelib: true\n"
        f"Tag: {WHEEL_TAG}\n\n"
    ).encode("utf-8")
    payload = dict(mapping)
    metadata_path = f"{dist_info}/METADATA"
    wheel_path = f"{dist_info}/WHEEL"
    if metadata_path in payload or wheel_path in payload:
        raise BuildError("metadata path collision")
    payload[metadata_path] = metadata
    payload[wheel_path] = wheel_body
    rows: list[dict[str, Any]] = []
    record_lines: list[str] = []
    def record_line(fields: list[str]) -> str:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(fields)
        return buffer.getvalue()

    for path in sorted(payload):
        body = payload[path]
        encoded = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode("ascii")
        record_lines.append(record_line([path, f"sha256={encoded}", str(len(body))]).rstrip("\n"))
        rows.append({"path": path, "bytes": len(body), "sha256": _hash_bytes(body)})
    record_path = f"{dist_info}/RECORD"
    record_body = ("\n".join(record_lines) + "\n" + record_line([record_path, "", ""])).encode("utf-8")
    payload[record_path] = record_body
    rows.append({"path": record_path, "bytes": len(record_body), "sha256": _hash_bytes(record_body), "record_self_hash": False})
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(payload):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = ((0o755 if "/scripts/" in path else 0o644) << 16)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, payload[path])
    result = output.getvalue()
    if len(result) > MAX_OUTPUT_BYTES:
        raise BuildError("derived wheel exceeds bounded size")
    return result, rows


def _atomic_new(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BuildError(f"refusing to overwrite output: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    except (OSError, ValueError) as exc:
        raise BuildError(f"atomic output failed: {path}") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _parse_archive_arg(value: str) -> tuple[str, Path]:
    match = re.fullmatch(r"(.+==[^=]+)=(/.*)", value)
    if match is None:
        raise BuildError("--archive must be IDENTITY=/absolute/archive.tar.gz")
    identity, raw_path = match.groups()
    path = Path(raw_path)
    _canonical(path)
    return identity, path


def _build_one(
    identity: str,
    row: dict[str, Any],
    archive_path: Path,
    output_dir: Path,
    generator_size: int,
    generator_sha: str,
    inspector_size: int,
    inspector_sha: str,
    lock_sha: str,
) -> tuple[Path, Path]:
    root, files, directories = _read_archive(archive_path, row["artifact"])
    if not root or not directories:
        raise BuildError(f"archive directory structure is missing for {identity}")
    setup_body = files.get("setup.py")
    pkg_info = files.get("PKG-INFO")
    if setup_body is None or pkg_info is None:
        raise BuildError(f"{identity} requires root setup.py and PKG-INFO")
    if "pyproject.toml" in files:
        raise BuildError("pyproject.toml is not part of the authenticated two-sdist mapping")
    if "setup.cfg" in files:
        findings = inspector._cfg_findings("setup.cfg", files["setup.cfg"].decode("utf-8"))
        if findings:
            raise BuildError("; ".join(sorted(set(findings))))
    if "MANIFEST.in" in files:
        findings = inspector._manifest_findings("MANIFEST.in", files["MANIFEST.in"].decode("utf-8"))
        if findings:
            raise BuildError("; ".join(sorted(set(findings))))
    setup = _setup_call(setup_body)
    metadata = _metadata_identity(pkg_info, setup, identity)
    dist_info = f"{_normalized_name(metadata['name'])}-{metadata['version']}.dist-info"
    mapping, source_rows, data_rows = _package_mapping(identity, root, files, setup, dist_info)
    build_sources = [
        {"path": source, "bytes": len(files[source]), "sha256": _hash_bytes(files[source])}
        for source in sorted(files)
        if source in {"PKG-INFO", "setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
    ]
    wheel_body, wheel_rows = _zip_bytes(mapping, pkg_info, dist_info)
    wheel_name = f"{_normalized_name(metadata['name'])}-{metadata['version']}-{WHEEL_TAG}.whl"
    wheel_path = output_dir / wheel_name
    provenance_path = output_dir / f"{wheel_name}.provenance.json"
    provenance = {
        "schema": "vokra-derived-locked-sdist-wheel-v1",
        "status": "DERIVED_NOT_OFFICIAL",
        "execution_policy": "UNAPPROVED_NO_EXECUTION",
        "publication": "NO_UPLOAD",
        "build_executed": False,
        "model_activity": {"imports": False, "weights_acquired": False, "weights_executed": False, "network": False},
        "identity": identity,
        "generator": {"path": str(Path(__file__).resolve()), "bytes": generator_size, "sha256": generator_sha},
        "generator_dependencies": [
            {
                "path": str(Path(inspector.__file__).absolute()),
                "bytes": inspector_size,
                "sha256": inspector_sha,
            }
        ],
        "lock": {"path": str(row.get("lock_path", "")), "sha256": lock_sha},
        "input_archive": row["artifact"],
        "source_root": root,
        "metadata": metadata,
        "build_sources": build_sources,
        "source_files": source_rows,
        "data_mappings": sorted(data_rows),
        "wheel": {"filename": wheel_name, "bytes": len(wheel_body), "sha256": _hash_bytes(wheel_body), "payload": wheel_rows},
        "license_policy": "UNRESOLVED_OWNER_REVIEW",
        "official_wheel": False,
    }
    provenance_body = (json.dumps(provenance, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_new(wheel_path, wheel_body)
    try:
        _atomic_new(provenance_path, provenance_body)
    except BuildError:
        wheel_path.unlink(missing_ok=True)
        raise
    return wheel_path, provenance_path


def build(lock_path: Path, archives: dict[str, Path], output_dir: Path) -> list[tuple[Path, Path]]:
    _regular_file(lock_path)
    lock_bytes = _bounded_read(lock_path, inspector.MAX_LOCK_BYTES)
    try:
        lock = inspector.tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, inspector.tomllib.TOMLDecodeError) as exc:
        raise BuildError("uv.lock is malformed") from exc
    active, _optional = inspector.active_sdists(lock)
    rows = {row["identity"]: {**row, "lock_path": str(lock_path)} for row in active}
    if set(rows) != SUPPORTED or set(archives) != SUPPORTED:
        raise BuildError(f"only the two supported active sdists are allowed: {sorted(SUPPORTED)}")
    for identity in sorted(SUPPORTED):
        if rows[identity]["artifact"] != PINNED_ARTIFACTS[identity]:
            raise BuildError(f"lock artifact is not the authenticated fixed pin: {identity}")
    _safe_output_dir(output_dir)
    generator_path = Path(__file__).absolute()
    generator_size, generator_sha = _hash_file(generator_path)
    if generator_size <= 0:
        raise BuildError("generator source is empty")
    inspector_path = Path(inspector.__file__).absolute()
    inspector_size, inspector_sha = _hash_file(inspector_path)
    if inspector_size <= 0:
        raise BuildError("inspector source is empty")
    lock_sha = _hash_bytes(lock_bytes)
    outputs: list[tuple[Path, Path]] = []
    try:
        for identity in sorted(SUPPORTED):
            outputs.append(_build_one(identity, rows[identity], archives[identity], output_dir, generator_size, generator_sha, inspector_size, inspector_sha, lock_sha))
    except Exception:
        for wheel_path, provenance_path in outputs:
            wheel_path.unlink(missing_ok=True)
            provenance_path.unlink(missing_ok=True)
        raise
    return outputs


def _build_rows_for_test(rows: dict[str, dict[str, Any]], archives: dict[str, Path], output_dir: Path) -> list[tuple[Path, Path]]:
    """Private synthetic-fixture seam; never reachable from the CLI."""

    if set(rows) != SUPPORTED or set(archives) != SUPPORTED:
        raise BuildError("synthetic fixture identities do not match supported identities")
    _safe_output_dir(output_dir)
    generator_path = Path(__file__).absolute()
    generator_size, generator_sha = _hash_file(generator_path)
    if generator_size <= 0:
        raise BuildError("generator source is empty")
    inspector_path = Path(inspector.__file__).absolute()
    inspector_size, inspector_sha = _hash_file(inspector_path)
    if inspector_size <= 0:
        raise BuildError("inspector source is empty")
    outputs: list[tuple[Path, Path]] = []
    try:
        for identity in sorted(SUPPORTED):
            outputs.append(_build_one(identity, rows[identity], archives[identity], output_dir, generator_size, generator_sha, inspector_size, inspector_sha, "synthetic-test-lock"))
    except Exception:
        for wheel_path, provenance_path in outputs:
            wheel_path.unlink(missing_ok=True)
            provenance_path.unlink(missing_ok=True)
        raise
    return outputs


def self_test() -> int:
    # The external test module contains the full synthetic archive matrix.  This
    # smoke test is intentionally model/package/backend-free and exercises the
    # two supported setup identities using tiny in-memory source archives.
    from test_build_derived_locked_sdists import run_smoke_test

    run_smoke_test()
    print("derived locked sdist builder: PASS (stdlib static self-test)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=False)
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.lock or args.archive or args.output_dir:
            parser.error("--self-test accepts no lock/archive/output-dir")
        return self_test()
    try:
        if args.lock is None or args.output_dir is None:
            raise BuildError("--lock and --output-dir are required")
        archives: dict[str, Path] = {}
        for raw in args.archive:
            identity, path = _parse_archive_arg(raw)
            if identity in archives:
                raise BuildError(f"duplicate --archive identity: {identity}")
            archives[identity] = path
        outputs = build(args.lock, archives, args.output_dir)
        for wheel_path, provenance_path in outputs:
            print(f"derived_wheel={wheel_path}")
            print(f"provenance={provenance_path}")
        return 0
    except (BuildError, OSError, UnicodeError, ValueError) as exc:
        print(f"derived locked sdist builder: BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
