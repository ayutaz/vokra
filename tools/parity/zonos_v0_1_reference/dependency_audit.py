"""Model-free dependency, native-file, and publisher-license audit for Zonos.

The audit intentionally uses only Python's standard library.  It can therefore
run before a source checkout, checkpoint, or Zonos/Torch import exists.  A
green structural report is not an execution approval: the owner/legal gate
remains ``BLOCKED_UNREVIEWED_TRANSITIVE`` and every publication disposition is
``NO_UPLOAD``.
"""
from __future__ import annotations

import hashlib
import binascii
import io
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import urllib.parse
import urllib.request
import base64
import csv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_PATH = PROJECT_DIR / "pyproject.toml"
LOCK_PATH = PROJECT_DIR / "uv.lock"
CONSTRAINTS_PATH = PROJECT_DIR / "numpy-build-constraints.txt"
REPOSITORY_ROOT = PROJECT_DIR.parents[2]
AUDITOR_PATH = Path(__file__).resolve()
WRAPPER_PATH = REPOSITORY_ROOT / "scripts/publish/vast-ai/audit-zonos-v0-1-dependencies.sh"
PREPARER_PATH = PROJECT_DIR / "prepare_numpy_no_blas.sh"
EXPECTED_PROJECT_SHA256 = "5cb58da85195f8f0812aa18bedd6a320226c7a3ef94e64c33e5782414c115b29"
EXPECTED_LOCK_SHA256 = "40fa51a7cffcfed126e073ecf0813fcbdb0935ea1bef05f51be1e75585fbcf76"
EXPECTED_CONSTRAINTS_SHA256 = "812ab3e215d7756738ab9a9aca7b8c94b53a3b10e8f1e43228e1b74965be020a"
NUMPY_SDIST_URL = "https://files.pythonhosted.org/packages/ec/d0/c12ddfd3a02274be06ffc71f3efc6d0e457b0409c4481596881e748cb264/numpy-2.2.2.tar.gz"
NUMPY_SDIST_SHA256 = "ed6906f61834d687738d25988ae117683705636936cc605be0bb208b23df4d8f"
NUMPY_SDIST_BYTES = 20233295
AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
PUBLICATION = "NO_UPLOAD"
DIRECT_DEPENDENCIES = {
    "huggingface-hub": "0.28.1",
    "numpy": "2.2.2",
    "safetensors": "0.5.3",
    "torch": "2.6.0+cpu",
    "torchaudio": "2.6.0+cpu",
    "tqdm": "4.67.1",
    "transformers": "4.48.1",
}
FORBIDDEN_PACKAGES = frozenset(
    {
        "cffi",
        "espeak",
        "espeak-ng",
        "librosa",
        "libsndfile",
        "onnx",
        "onnxruntime",
        "phonemizer",
        "soundfile",
        "soxr",
    }
)
LICENSE_CONCLUSIONS = {
    "certifi": "MPL-2.0_POLICY_REVIEW_REQUIRED",
    "charset-normalizer": "MIT_REVIEWED",
    "colorama": "BSD-3-Clause_REVIEWED",
    "filelock": "UNLICENSE_POLICY_REVIEW_REQUIRED",
    "fsspec": "BSD-3-Clause_REVIEWED",
    "huggingface-hub": "Apache-2.0_REVIEWED",
    "idna": "BSD-3-Clause_REVIEWED",
    "jinja2": "BSD-3-Clause_REVIEWED",
    "markupsafe": "BSD-3-Clause_REVIEWED",
    "mpmath": "BSD_STYLE_PRIMARY_REVIEW_REQUIRED",
    "networkx": "BSD-3-Clause_REVIEWED",
    "numpy": "BSD-3-Clause_NATIVE_BUNDLE_REVIEW_REQUIRED",
    "packaging": "Apache-2.0_REVIEWED",
    "pyyaml": "MIT_REVIEWED",
    "regex": "Apache-2.0_REVIEWED",
    "requests": "Apache-2.0_REVIEWED",
    "safetensors": "Apache-2.0_REVIEWED",
    "setuptools": "MIT_REVIEWED",
    "sympy": "BSD-3-Clause_REVIEW_REQUIRED",
    "tokenizers": "Apache-2.0_REVIEWED",
    "torch": "BSD-3-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "torchaudio": "BSD-2-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "tqdm": "MPL-2.0_OR_MIT_POLICY_REVIEW_REQUIRED",
    "transformers": "Apache-2.0_REVIEWED",
    "typing-extensions": "PSF-2.0_POLICY_REVIEW_REQUIRED",
    "urllib3": "MIT_REVIEWED",
    "vokra-zonos-v0-1-reference": "FIRST_PARTY_NOT_INDEPENDENT_SCOPE",
}
OWNER_CANDIDATE_SCOPE = (
    "owner/legal: review every resolved Linux x86_64 package, Torch/torchaudio "
    "bundled/native files, and publisher LICENSE/NOTICE bytes before execution"
)
# The locked NumPy 2.2.2 sdist contains 7,758 unique regular files. This
# publisher-evidence reader therefore shares the preparer's bounded 8,192 cap;
# it still rejects larger or unsafe archives before extracting any bytes.
MAX_SDIST_TAR_MEMBERS = 8192
MAX_PUBLISHER_MEMBER_BYTES = 1 << 20
MAX_PUBLISHER_TOTAL_BYTES = 8 << 20
GENERATED_NUMPY_METADATA = frozenset(
    {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pep503_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _streaming_binary_facts(path: Path) -> tuple[int, str, bool]:
    """Hash native payloads without retaining a potentially huge file."""
    digest = hashlib.sha256()
    size = 0
    header = bytearray()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
            if len(header) < 4:
                header.extend(block[: 4 - len(header)])
    return size, digest.hexdigest(), bytes(header) == b"\x7fELF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_identity() -> dict[str, Any]:
    if not PROJECT_PATH.is_file() or PROJECT_PATH.is_symlink():
        raise RuntimeError("dedicated Zonos pyproject.toml is missing or symlinked")
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise RuntimeError("dedicated Zonos uv.lock is missing or symlinked")
    if not CONSTRAINTS_PATH.is_file() or CONSTRAINTS_PATH.is_symlink():
        raise RuntimeError("dedicated NumPy builder constraints are missing or symlinked")
    project_sha = sha256(PROJECT_PATH)
    lock_sha = sha256(LOCK_PATH)
    constraints_sha = sha256(CONSTRAINTS_PATH)
    if (
        project_sha != EXPECTED_PROJECT_SHA256
        or lock_sha != EXPECTED_LOCK_SHA256
        or constraints_sha != EXPECTED_CONSTRAINTS_SHA256
    ):
        raise RuntimeError("dedicated Zonos project identity drifted")
    return {
        "project": PROJECT_DIR.name,
        "python": "3.12",
        "pyproject_sha256": project_sha,
        "uv_lock_sha256": lock_sha,
        "constraints_sha256": constraints_sha,
        "expected_direct_versions": dict(DIRECT_DEPENDENCIES),
    }


def execution_identity(expected_head: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise RuntimeError("expected clean HEAD must be 40 lowercase hexadecimal characters")
    try:
        actual_head = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot bind repository HEAD: {error}") from error
    if actual_head != expected_head:
        raise RuntimeError(f"repository HEAD {actual_head} differs from expected {expected_head}")
    for path in (AUDITOR_PATH, WRAPPER_PATH, PREPARER_PATH):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"audit identity file is missing or symlinked: {path}")
    identity = {
        "expected_head": expected_head,
        "actual_head": actual_head,
        "dependency_audit_sha256": sha256(AUDITOR_PATH),
        "wrapper_sha256": sha256(WRAPPER_PATH),
        "preparer_sha256": sha256(PREPARER_PATH),
        "pyproject_sha256": sha256(PROJECT_PATH),
        "uv_lock_sha256": sha256(LOCK_PATH),
        "constraints_sha256": sha256(CONSTRAINTS_PATH),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "sys_platform": sys.platform,
            "python": platform.python_version(),
        },
    }
    if identity["platform"]["system"] != "Linux" or identity["platform"]["machine"] != "x86_64":
        raise RuntimeError("dependency evidence requires Linux x86_64")
    return identity


def _lock_rows() -> list[dict[str, Any]]:
    document = tomllib.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if document.get("requires-python") != "==3.12.*":
        raise RuntimeError("dedicated Zonos lock Python contract drifted")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for package in document.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("uv.lock has an invalid package identity")
        normalized_name = pep503_name(name)
        if normalized_name in FORBIDDEN_PACKAGES:
            raise RuntimeError(f"forbidden Zonos dependency in lock: {name}")
        if normalized_name not in LICENSE_CONCLUSIONS:
            raise RuntimeError(f"unclassified Zonos dependency in lock: {name}")
        source = package.get("source", {})
        source_key = json.dumps(source, sort_keys=True, separators=(",", ":"))
        identity = (name, version, source_key)
        if identity in seen:
            raise RuntimeError(f"duplicate lock identity: {name} {version}")
        seen.add(identity)
        rows.append(
            {
                "name": name,
                "version": version,
                "source": source,
                "markers": sorted(
                    dependency.get("marker")
                    for dependency in package.get("dependencies", [])
                    if isinstance(dependency, dict) and isinstance(dependency.get("marker"), str)
                ),
                "dependencies": package.get("dependencies", []),
                "sdist": package.get("sdist"),
                "license_conclusion": LICENSE_CONCLUSIONS[normalized_name],
            }
        )
    names = {pep503_name(row["name"]) for row in rows}
    if not set(DIRECT_DEPENDENCIES).issubset(names):
        raise RuntimeError("a direct Zonos dependency is absent from uv.lock")
    torch_rows = [row for row in rows if pep503_name(row["name"]) == "torch"]
    audio_rows = [row for row in rows if pep503_name(row["name"]) == "torchaudio"]
    if not any(row["version"] == "2.6.0+cpu" and row["source"].get("registry") == "https://download.pytorch.org/whl/cpu" for row in torch_rows):
        raise RuntimeError("Linux x86_64 CPU torch resolution is not pinned")
    if not any(row["version"] == "2.6.0+cpu" and row["source"].get("registry") == "https://download.pytorch.org/whl/cpu" for row in audio_rows):
        raise RuntimeError("Linux x86_64 CPU torchaudio resolution is not pinned")
    rows.sort(key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))
    return rows


def lock_audit() -> dict[str, Any]:
    rows = _lock_rows()
    return {
        "schema": "vokra-zonos-uv-lock-license-audit-v1",
        "status": AUDIT_STATUS,
        "publication": PUBLICATION,
        "owner_candidate_scope": OWNER_CANDIDATE_SCOPE,
        "package_count": len(rows),
        "rows_sha256": _digest(rows),
        "rows": rows,
    }


def _active_lock_rows(
    rows: list[dict[str, Any]], environment: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Select the exact Linux x86_64 lock closure, not every platform fork."""
    active: list[dict[str, Any]] = []
    virtual = [row for row in rows if row["source"].get("virtual") is not None]
    if len(virtual) != 1:
        raise RuntimeError("dedicated lock must contain exactly one virtual root")
    selected: dict[str, dict[str, Any]] = {}
    pending = list(virtual[0].get("dependencies", []))
    environment = environment or {
        "sys_platform": "linux",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "python_version": "3.12",
        "python_full_version": "3.12.0",
    }
    while pending:
        dependency = pending.pop()
        if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
            raise RuntimeError("virtual root has an invalid dependency row")
        marker = dependency.get("marker")
        if marker is not None and not _marker_matches(marker, environment):
            continue
        name = pep503_name(dependency["name"])
        candidates = [
            row for row in rows
            if pep503_name(row["name"]) == name and row["source"].get("virtual") is None
        ]
        version = dependency.get("version")
        source = dependency.get("source")
        if version is not None:
            candidates = [row for row in candidates if row["version"] == version]
        if source is not None:
            candidates = [row for row in candidates if row["source"] == source]
        if name in selected:
            if selected[name] not in candidates:
                raise RuntimeError(f"active Linux lock resolution conflicts: {name}")
            continue
        if len(candidates) != 1:
            raise RuntimeError(f"active Linux lock resolution is not unique: {name}")
        selected[name] = candidates[0]
        for child in selected[name].get("dependencies", []):
            if isinstance(child, dict):
                pending.append(child)
    active = [selected[name] for name in sorted(selected)]
    expected_torch_version = "2.6.0+cpu" if environment["sys_platform"] == "linux" else "2.6.0"
    for name in ("torch", "torchaudio"):
        candidates = [row for row in active if pep503_name(row["name"]) == name]
        if len(candidates) != 1 or candidates[0]["version"] != expected_torch_version:
            raise RuntimeError(f"Linux CPU {name} resolution is not pinned")
    return active


def _marker_matches(marker: str, environment: dict[str, str]) -> bool:
    """Evaluate uv lock markers with packaging, with a strict stdlib fallback."""
    try:
        from packaging.markers import Marker
    except ImportError:
        # Self-tests intentionally run without syncing the project.  This
        # parser accepts only the lock grammar (quoted ==/!= leaves joined by
        # and/or); unknown marker syntax remains fail-closed.
        def evaluate(expression: str) -> bool:
            expression = expression.strip()
            while expression.startswith("(") and expression.endswith(")"):
                depth = 0
                closes_at = None
                for index, character in enumerate(expression):
                    if character == "(":
                        depth += 1
                    elif character == ")":
                        depth -= 1
                        if depth == 0:
                            closes_at = index
                            break
                if closes_at != len(expression) - 1:
                    break
                expression = expression[1:-1].strip()
            for operator in (" or ", " and "):
                parts: list[str] = []
                start = 0
                depth = 0
                quote = False
                index = 0
                while index < len(expression):
                    character = expression[index]
                    if character == "'":
                        quote = not quote
                    elif not quote and character == "(":
                        depth += 1
                    elif not quote and character == ")":
                        depth -= 1
                    if not quote and depth == 0 and expression.startswith(operator, index):
                        parts.append(expression[start:index])
                        start = index + len(operator)
                        index = start
                        continue
                    index += 1
                if parts:
                    parts.append(expression[start:])
                if len(parts) > 1:
                    values = [evaluate(part) for part in parts]
                    return any(values) if operator.strip() == "or" else all(values)
            match = re.fullmatch(r"([a-z_]+)\s*(==|!=)\s*'([^']*)'", expression)
            if match is None or match.group(1) not in environment:
                raise RuntimeError(f"unsupported lock dependency marker: {marker}")
            actual = environment[match.group(1)]
            return actual == match.group(3) if match.group(2) == "==" else actual != match.group(3)

        return evaluate(marker)
    try:
        return bool(Marker(marker).evaluate(environment))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid lock dependency marker: {marker}") from error


def _elf_needed(path: Path) -> list[str]:
    try:
        output = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot collect ELF NEEDED for {path}: {error}") from error
    return sorted(set(re.findall(r"Shared library: \[(.+?)\]", output)))


def _prepare_archive(path: Path | None) -> Path | None:
    if path is None:
        return None
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RuntimeError("publisher archive must be an absent absolute path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("publisher archive parent must be an existing non-symlink directory")
    repository_root = REPOSITORY_ROOT.resolve()
    if path == repository_root or repository_root in path.parents:
        raise RuntimeError("publisher archive must be outside the checkout")
    cursor = path.parent
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise RuntimeError("publisher archive path has a symlinked ancestor")
        cursor = cursor.parent
    path.mkdir()
    return path


def _native_record(path: Path, root: Path, failures: list[str]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        failures.append(f"native file is unsafe: {path}")
        raise RuntimeError(f"native file is unsafe: {path}")
    byte_count, file_sha256, is_elf = _streaming_binary_facts(path)
    needed: list[str] = []
    if is_elf:
        try:
            needed = _elf_needed(path)
        except RuntimeError as error:
            failures.append(str(error))
    policy_review_tokens = (
        "gpl", "lgpl", "soxr", "espeak", "sndfile",
        "libgfortran", "libquadmath", "openblas",
    )
    policy_review_items = [path.name, *needed]
    policy_review_hits = [
        item
        for item in policy_review_items
        if any(token in item.lower() for token in policy_review_tokens)
    ]
    if policy_review_hits:
        failures.append(
            f"native policy-review boundary for {path}: {','.join(sorted(set(policy_review_hits)))}"
        )
    return {
        "root": str(root),
        "path": str(path.relative_to(root)),
        "kind": ".so.*" if re.search(r"\.so\.", path.name, re.IGNORECASE) else path.suffix.lower(),
        "bytes": byte_count,
        "sha256": file_sha256,
        "format": "ELF" if is_elf else "non-ELF",
        "needed": needed,
    }


def _sdist_license_fallback(
    name: str,
    row: dict[str, Any],
    archive_root: Path | None,
    failures: list[str],
) -> list[dict[str, Any]]:
    """Recover publisher bytes from the lock-pinned sdist when wheel metadata omits them."""
    sdist = row.get("sdist")
    if not isinstance(sdist, dict):
        failures.append(f"publisher LICENSE/NOTICE evidence missing and no locked sdist: {name}")
        return []
    url = sdist.get("url")
    locked_hash = sdist.get("hash")
    locked_size = sdist.get("size")
    if (
        not isinstance(url, str)
        or not isinstance(locked_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", locked_hash)
        or not isinstance(locked_size, int)
        or locked_size < 0
    ):
        failures.append(f"locked sdist identity is malformed: {name}")
        return []
    expected_sha = locked_hash.removeprefix("sha256:")
    expected_version = row.get("version", "")
    parsed_url = urllib.parse.urlsplit(url)
    path_parts = parsed_url.path.split("/")
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "files.pythonhosted.org"
        or parsed_url.query
        or parsed_url.fragment
        or len(path_parts) != 6
        or path_parts[1] != "packages"
        or not re.fullmatch(r"[0-9a-f]{2}", path_parts[2] or "")
        or not re.fullmatch(r"[0-9a-f]{2}", path_parts[3] or "")
        or not re.fullmatch(r"[0-9a-f]{32,64}", path_parts[4] or "")
        or not path_parts[5].endswith(".tar.gz")
    ):
        failures.append(f"locked sdist URL is not an exact files.pythonhosted.org path: {name}")
        return []
    filename_stem = path_parts[5][:-len(".tar.gz")]
    package_prefix, separator, filename_version = filename_stem.rpartition("-")
    if (
        not separator
        or filename_version != expected_version
        or pep503_name(package_prefix) != pep503_name(name)
    ):
        failures.append(f"locked sdist filename is not bound to the package row: {name}")
        return []
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f"zonos-{name}-", suffix=".tar.gz")
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                response_url = response.geturl()
                if response_url != url:
                    failures.append(f"locked sdist redirect is forbidden: {name}")
                    return []
                content_length = response.headers.get("Content-Length")
                if content_length is not None and (
                    not content_length.isdigit() or int(content_length) != locked_size
                ):
                    failures.append(f"locked sdist Content-Length mismatch: {name}")
                    return []
                for block in iter(lambda: response.read(1 << 20), b""):
                    if size + len(block) > locked_size:
                        failures.append(f"locked sdist stream exceeds locked size: {name}")
                        return []
                    digest.update(block)
                    size += len(block)
                    output.write(block)
        except (OSError, ValueError) as error:
            failures.append(f"locked sdist download failed for {name}: {error}")
            return []
        if size != locked_size or digest.hexdigest() != expected_sha:
            failures.append(f"locked sdist bytes/hash mismatch: {name}")
            return []
        recovered: list[dict[str, Any]] = []
        try:
            with tarfile.open(temporary, "r:*") as package:
                members = package.getmembers()
                if len(members) > MAX_SDIST_TAR_MEMBERS:
                    failures.append(f"locked sdist has too many tar members: {name}")
                    return []
                seen_members: set[str] = set()
                extracted_total = 0
                for member in sorted(members, key=lambda item: item.name):
                    member_path = PurePosixPath(member.name)
                    basename = member_path.name.lower()
                    if not basename.startswith(("license", "licence", "notice", "copying")):
                        continue
                    if member.name in seen_members:
                        failures.append(f"duplicate publisher member in locked sdist: {name}:{member.name}")
                        continue
                    seen_members.add(member.name)
                    if member.issym() or member.islnk() or not member.isfile() or member_path.is_absolute() or ".." in member_path.parts:
                        failures.append(f"unsafe publisher member in locked sdist: {name}:{member.name}")
                        continue
                    if member.size > MAX_PUBLISHER_MEMBER_BYTES:
                        failures.append(f"publisher member exceeds byte bound: {name}:{member.name}")
                        continue
                    if extracted_total + member.size > MAX_PUBLISHER_TOTAL_BYTES:
                        failures.append(f"publisher members exceed cumulative byte bound: {name}")
                        return []
                    source = package.extractfile(member)
                    if source is None:
                        failures.append(f"publisher member cannot be read: {name}:{member.name}")
                        continue
                    archive_relative = Path(name) / "sdist" / Path(*member_path.parts)
                    payload_size = 0
                    payload_digest = hashlib.sha256()
                    destination = archive_root / archive_relative if archive_root is not None else None
                    if destination is not None:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists() or destination.is_symlink():
                            failures.append(f"publisher archive destination already exists: {destination}")
                            continue
                        target = destination.open("xb")
                    else:
                        target = None
                    try:
                        with source:
                            for block in iter(lambda: source.read(1 << 20), b""):
                                if payload_size + len(block) > MAX_PUBLISHER_MEMBER_BYTES:
                                    failures.append(f"publisher member stream exceeds byte bound: {name}:{member.name}")
                                    break
                                payload_digest.update(block)
                                payload_size += len(block)
                                if target is not None:
                                    target.write(block)
                    finally:
                        if target is not None:
                            target.close()
                    if payload_size > MAX_PUBLISHER_MEMBER_BYTES or payload_size != member.size:
                        if destination is not None:
                            destination.unlink(missing_ok=True)
                        failures.append(f"publisher member size is not exact: {name}:{member.name}")
                        continue
                    extracted_total += payload_size
                    row_value: dict[str, Any] = {
                        "distribution": name,
                        "path": member.name,
                        "bytes": payload_size,
                        "sha256": payload_digest.hexdigest(),
                        "source": {
                            "kind": "locked-sdist",
                            "url": url,
                            "sha256": expected_sha,
                            "bytes": locked_size,
                        },
                    }
                    if destination is not None:
                        row_value["archive_path"] = str(archive_relative)
                    recovered.append(row_value)
        except (OSError, tarfile.TarError) as error:
            failures.append(f"locked sdist publisher archive failed for {name}: {error}")
            return []
        if not recovered:
            failures.append(f"locked sdist has no publisher LICENSE/NOTICE member: {name}")
        return recovered
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _numpy_runtime_config(failures: list[str]) -> dict[str, Any]:
    try:
        import numpy

        config = _json_safe(numpy.__config__.show(mode="dicts"))
    except Exception as error:  # pragma: no cover - depends on installed VAST wheel
        failures.append(f"NumPy runtime build config collection failed: {error}")
        return {
            "schema": "vokra-zonos-numpy-runtime-config-v1",
            "status": "FAIL_CONFIG_UNAVAILABLE",
            "config": None,
            "forbidden_boundaries": ["config-unavailable"],
        }
    forbidden = _numpy_config_forbidden(config)
    if forbidden:
        failures.append("NumPy runtime config policy boundary: " + ",".join(forbidden))
    return {
        "schema": "vokra-zonos-numpy-runtime-config-v1",
        "status": "PASS_NO_FORBIDDEN_BLAS" if not forbidden else "FAIL_FORBIDDEN_BLAS",
        "config": config,
        "forbidden_boundaries": forbidden,
    }


def _numpy_config_forbidden(config: Any) -> list[str]:
    """Return active/native BLAS boundaries named by a NumPy config object."""
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":")).casefold()
    forbidden = [
        token for token in ("openblas", "gfortran", "quadmath") if token in serialized
    ]

    def active_backend_values(value: Any, under_backend: bool = False) -> list[str]:
        if isinstance(value, dict):
            values: list[str] = []
            for key, item in value.items():
                key_backend = under_backend or any(
                    token in str(key).casefold() for token in ("blas", "lapack")
                )
                values.extend(active_backend_values(item, key_backend))
            return values
        if under_backend and isinstance(value, bool) and value:
            return ["active-blas-lapack-flag"]
        if under_backend and isinstance(value, str) and value.casefold() not in {
            "none", "unknown", "false", "not found", "not-found",
        }:
            return [value]
        return []

    forbidden.extend(active_backend_values(config))
    return sorted(set(forbidden))


def _require_no_bytecode_writes() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError(
            "Zonos audit requires PYTHONDONTWRITEBYTECODE=1 before runtime inspection"
        )


def installed_audit(
    preparation_path: Path, publisher_archive: Path | None = None
) -> dict[str, Any]:
    """Collect exact installed distribution/native/license evidence on VAST."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("installed closure audit requires Linux x86_64")
    _require_no_bytecode_writes()
    preparation = _validate_preparation(preparation_path)
    prepared_venv = Path(preparation["venv"]["path"])
    lock_rows = _lock_rows()
    active_rows = _active_lock_rows(lock_rows)
    expected_versions = {
        pep503_name(row["name"]): row["version"] for row in active_rows
    }
    failures: list[str] = []
    if Path(sys.prefix).resolve() != prepared_venv.resolve():
        failures.append("installed audit sys.prefix is outside the prepared venv")
    executable = Path(sys.executable).absolute()
    if not executable.is_relative_to(prepared_venv):
        failures.append("installed audit sys.executable is outside the prepared venv")
    distributions: dict[str, importlib.metadata.Distribution] = {}
    duplicate_distributions: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str):
            failures.append("installed distribution has no canonical Name metadata")
            continue
        normalized = pep503_name(name)
        if normalized in distributions:
            duplicate_distributions.add(normalized)
        distributions.setdefault(normalized, distribution)
    if duplicate_distributions:
        failures.append("duplicate installed distributions: " + ",".join(sorted(duplicate_distributions)))
    actual_names = set(distributions)
    expected_names = set(expected_versions)
    for name in sorted(expected_names - actual_names):
        failures.append(f"missing active locked distribution: {name}")
    for name in sorted(actual_names - expected_names):
        failures.append(f"unexpected installed distribution: {name}")
    installed_rows: list[dict[str, Any]] = []
    numpy_distribution: importlib.metadata.Distribution | None = None
    for name, distribution in sorted(distributions.items()):
        version = distribution.version
        if name in {"torch", "torchaudio"} and version == "2.6.0":
            version = "2.6.0+cpu"
        expected = expected_versions.get(name)
        if expected is not None and version != expected:
            failures.append(f"installed version mismatch: {name}={version!r}, expected {expected!r}")
        if name in FORBIDDEN_PACKAGES:
            failures.append(f"forbidden installed distribution: {name}")
        if name == "numpy":
            numpy_distribution = distribution
        installed_rows.append(
            {
                "name": name,
                "version": version,
                "metadata_license": distribution.metadata.get("License", ""),
            }
        )
    roots = {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib") if sysconfig.get_paths().get(key)}
    native: list[dict[str, Any]] = []
    native_extensions = re.compile(r"(?:\.so(?:\..*)?|\.dylib|\.dll|\.pyd|\.a)$", re.IGNORECASE)
    for root in sorted(roots):
        if not root.is_dir() or root.is_symlink():
            failures.append(f"site-packages root is unsafe: {root}")
            continue
        for path in sorted(root.rglob("*")):
            if not native_extensions.search(path.name):
                continue
            if path.is_symlink() or not path.is_file():
                failures.append(f"native file is unsafe: {path}")
                continue
            native.append(_native_record(path, root, failures))
    numpy_native = [
        row for row in native
        if str(row.get("path", "")).casefold().startswith(("numpy/", "numpy.libs/"))
    ]
    numpy_forbidden = sorted(
        {
            item
            for row in numpy_native
            for item in [row.get("path", ""), *row.get("needed", [])]
            if any(
                token in str(item).casefold()
                for token in ("openblas", "libgfortran", "libquadmath")
            )
        }
    )
    if not numpy_native:
        failures.append("NumPy native payload evidence is missing")
    if numpy_forbidden:
        failures.append(
            "NumPy no-BLAS policy boundary detected: " + ",".join(numpy_forbidden)
        )
    numpy_installation: dict[str, Any] = {
        "version": numpy_distribution.version if numpy_distribution is not None else None,
        "expected_version": "2.2.2",
        "wheel": preparation["wheel"],
        "direct_url": None,
    }
    numpy_record: dict[str, Any] = {
        "schema": "vokra-zonos-numpy-record-v2",
        "status": "FAIL_DISTRIBUTION_MISSING",
        "rows": [],
        "generated": [],
    }
    if numpy_distribution is None:
        failures.append("NumPy distribution is missing from the prepared environment")
    else:
        try:
            direct_url_text = numpy_distribution.read_text("direct_url.json")
            if direct_url_text is None:
                failures.append("NumPy installed wheel direct_url.json evidence is missing")
            else:
                direct_url = _strict_json_text(direct_url_text)
                numpy_installation["direct_url"] = direct_url
                wheel_path = preparation_path.parent / "wheelhouse" / preparation["wheel"]["basename"]
                _validate_numpy_direct_url(direct_url, wheel_path, preparation["wheel"])
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(f"NumPy installed wheel direct_url.json is invalid: {error}")
        numpy_record = _numpy_record_evidence(
            numpy_distribution,
            preparation_path.parent / "wheelhouse" / preparation["wheel"]["basename"],
            prepared_venv,
            failures,
        )
    numpy_config = _numpy_runtime_config(failures)
    numpy_native_policy = {
        "schema": "vokra-zonos-numpy-native-policy-v1",
        "status": "PASS_NO_FORBIDDEN_BLAS" if numpy_native and not numpy_forbidden else "FAIL_FORBIDDEN_BLAS",
        "native_file_count": len(numpy_native),
        "forbidden_boundaries": numpy_forbidden,
    }
    archive_root = _prepare_archive(publisher_archive)
    publisher_files: list[dict[str, Any]] = []
    archived_files: list[dict[str, Any]] = []
    active_rows_by_name = {pep503_name(row["name"]): row for row in active_rows}
    for name, distribution in sorted(distributions.items()):
        matched = 0
        for item in distribution.files or ():
            base = Path(item).name.lower()
            path = Path(distribution.locate_file(item))
            if not base.startswith(("license", "licence", "notice", "copying")):
                continue
            if path.is_symlink() or not path.is_file():
                failures.append(f"publisher file is unsafe: {path}")
                continue
            matched += 1
            if archive_root is not None:
                relative = Path(item)
                if relative.is_absolute() or "\\" in str(relative) or any(
                    component in {"", ".", ".."} for component in relative.parts
                ):
                    failures.append(f"publisher file path is unsafe: {item}")
                    continue
                archive_relative = Path(name) / relative
                destination = archive_root / archive_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink() or destination.exists():
                    failures.append(f"publisher archive destination already exists: {destination}")
                    continue
                with path.open("rb") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                archived_files.append(
                    {
                        "distribution": name,
                        "path": str(relative),
                        "archive_path": str(archive_relative),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            publisher_files.append(
                {
                    "distribution": name,
                    "path": str(item),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        if matched == 0:
            recovered = _sdist_license_fallback(name, active_rows_by_name.get(name, {}), archive_root, failures)
            if recovered:
                publisher_files.extend(recovered)
                archived_files.extend(row for row in recovered if "archive_path" in row)
            else:
                publisher_files.append({"distribution": name, "status": "MISSING"})
                failures.append(f"missing publisher LICENSE/LICENCE/NOTICE/COPYING evidence: {name}")
    publisher_files.sort(key=lambda row: (row["distribution"], row.get("path", "")))
    archive_manifest: dict[str, Any] | None = None
    if archive_root is not None:
        archived_files.sort(key=lambda row: (row["distribution"], row["path"]))
        manifest_path = archive_root / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(archived_files, indent=2, sort_keys=True) + "\n")
        archive_manifest = {
            "directory": str(archive_root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "files": archived_files,
        }
    return {
        "status": "COLLECTED",
        "platform": {"system": "Linux", "machine": "x86_64"},
        "active_lock_packages": active_rows,
        "installed_distributions": installed_rows,
        "failures": sorted(set(failures)),
        "native_files": native,
        "publisher_license_notice_files": publisher_files,
        "publisher_archive": archive_manifest,
        "preparation": preparation,
        "numpy_installation": numpy_installation,
        "numpy_record": numpy_record,
        "numpy_runtime_config": numpy_config,
        "numpy_native_policy": numpy_native_policy,
        "digests": {
            "installed_closure_sha256": _digest(installed_rows),
            "native_files_sha256": _digest(native),
            "publisher_files_sha256": _digest(publisher_files),
            "numpy_record_sha256": _digest(numpy_record),
            "numpy_runtime_config_sha256": _digest(numpy_config),
        },
    }


def audit(
    output: Path | None = None,
    installed: bool = False,
    expected_head: str | None = None,
    publisher_archive: Path | None = None,
    preparation: Path | None = None,
) -> dict[str, Any]:
    identity = project_identity()
    if installed and expected_head is None:
        raise RuntimeError("installed audit requires expected clean HEAD")
    if installed and preparation is None:
        raise RuntimeError("installed audit requires the prepared NumPy environment identity")
    if not installed and preparation is not None:
        raise RuntimeError("preparation identity is only valid for installed audit")
    execution = execution_identity(expected_head) if expected_head is not None else None
    report: dict[str, Any] = {
        "schema": "vokra-zonos-dependency-audit-v1",
        "status": AUDIT_STATUS,
        "publication": PUBLICATION,
        "project": identity,
        "lock": lock_audit(),
        "execution_identity": execution,
        "preparation_path": str(preparation) if preparation is not None else None,
    }
    report["installed"] = (
        installed_audit(preparation, publisher_archive)
        if installed
        else {"status": "NOT_COLLECTED_PRE_ACQUISITION"}
    )
    installed_report = report["installed"]
    report["candidate_scope"] = {
        "schema": "vokra-zonos-dependency-approval-scope-v1",
        "lock_rows_sha256": report["lock"]["rows_sha256"],
        "preparation_path": str(preparation) if preparation is not None else None,
        "preparation_sha256": sha256(preparation) if preparation is not None else None,
        "constraints_sha256": identity["constraints_sha256"],
        "sdist_identity": (
            installed_report.get("preparation", {}).get("sdist")
            if installed
            else None
        ),
        "wheel_identity": (
            installed_report.get("preparation", {}).get("wheel")
            if installed
            else None
        ),
        "installed_closure_sha256": installed_report.get("digests", {}).get("installed_closure_sha256"),
        "native_files_sha256": installed_report.get("digests", {}).get("native_files_sha256"),
        "publisher_files_sha256": installed_report.get("digests", {}).get("publisher_files_sha256"),
        "numpy_record_sha256": installed_report.get("digests", {}).get("numpy_record_sha256"),
        "numpy_native_policy_sha256": _digest(installed_report.get("numpy_native_policy")),
        "numpy_runtime_config_sha256": _digest(installed_report.get("numpy_runtime_config")),
        "publisher_archive_manifest_sha256": (
            installed_report.get("publisher_archive") or {}
        ).get("manifest_sha256"),
        "failures": installed_report.get("failures", []),
        "model_access": False,
        "source_access": False,
        "checkpoint_access": False,
        "publication": PUBLICATION,
        "execution_identity": execution,
    }
    report["candidate_scope_sha256"] = _digest(report["candidate_scope"])
    report["failures"] = installed_report.get("failures", [])
    if output is not None:
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"dependency audit output already exists or is symlinked: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _strict_json(path: Path) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs
    )
    return value


def _strict_json_text(value: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=unique_pairs)


def _validate_numpy_direct_url(
    direct_url: Any, wheel_path: Path, wheel: dict[str, Any]
) -> None:
    if not isinstance(direct_url, dict) or direct_url.get("url") != wheel_path.as_uri():
        raise RuntimeError("NumPy direct_url does not match the prepared wheel URL")
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise RuntimeError("NumPy direct_url archive_info is malformed")
    if archive_info and archive_info.get("hash") != "sha256=" + str(wheel["sha256"]):
        raise RuntimeError("NumPy direct_url wheel hash does not match preparation")


def _numpy_record_evidence(
    distribution: importlib.metadata.Distribution,
    wheel_path: Path,
    venv_root: Path,
    failures: list[str],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "schema": "vokra-zonos-numpy-record-v2",
        "status": "FAIL_RECORD_MISSING",
        "wheel": None,
        "installed": None,
        "rows": [],
        "generated": [],
    }
    rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    try:
        if not wheel_path.is_file() or wheel_path.is_symlink():
            raise ValueError("prepared NumPy wheel is missing or symlinked")
        wheel_record_path: str | None = None
        wheel_record_bytes: bytes | None = None
        wheel_files: set[str] = set()
        with zipfile.ZipFile(wheel_path, "r") as archive:
            names: set[str] = set()
            for info in archive.infolist():
                name = info.filename
                if not name or "\\" in name:
                    raise ValueError(f"wheel member path is unsafe: {name!r}")
                member_path = PurePosixPath(name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"wheel member path is unsafe: {name!r}")
                if name in names:
                    raise ValueError(f"wheel member is duplicated: {name}")
                names.add(name)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if stat.S_ISLNK(mode):
                    raise ValueError(f"wheel member is symlinked: {name}")
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError(f"wheel member has unsafe file type: {name}")
                if not name.endswith("/"):
                    wheel_files.add(name)
                if name.endswith(".dist-info/RECORD"):
                    if wheel_record_path is not None:
                        raise ValueError("wheel contains multiple dist-info RECORD files")
                    wheel_record_path = name
                    wheel_record_bytes = archive.read(info)
            if wheel_record_path is None or wheel_record_bytes is None:
                raise ValueError("wheel dist-info RECORD is missing")
        if not wheel_record_bytes.endswith(b"\n"):
            raise ValueError("wheel RECORD does not end with a newline")
        wheel_record_sha256 = hashlib.sha256(wheel_record_bytes).hexdigest()
        installed_record_path = Path(distribution.locate_file(wheel_record_path))
        if installed_record_path.is_symlink() or not installed_record_path.is_file():
            raise ValueError("installed NumPy RECORD is missing or symlinked")
        installed_record_bytes = installed_record_path.read_bytes()
        evidence["wheel"] = {
            "path": wheel_record_path,
            "bytes": len(wheel_record_bytes),
            "sha256": wheel_record_sha256,
        }
        evidence["installed"] = {
            "path": wheel_record_path,
            "bytes": len(installed_record_bytes),
            "sha256": hashlib.sha256(installed_record_bytes).hexdigest(),
        }
        def parse_record(
            raw: bytes, label: str, allow_generated: bool
        ) -> dict[str, tuple[str, str, str]]:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{label} RECORD is not UTF-8: {error}") from error
            parsed = csv.reader(io.StringIO(text))
            values: dict[str, tuple[str, str, str]] = {}
            for fields in parsed:
                if len(fields) != 3:
                    raise ValueError(f"{label} RECORD row must have path, hash, and size")
                relative, encoded_hash, size_text = fields
                if not relative or relative in values:
                    raise ValueError(f"{label} RECORD path is empty or duplicated: {relative!r}")
                relative_path = PurePosixPath(relative)
                generated_path = relative in generated_paths
                if (
                    relative_path.is_absolute()
                    or "\\" in relative
                    or ".." in relative_path.parts
                ) and not (allow_generated and generated_path):
                    raise ValueError(f"{label} RECORD path is unsafe: {relative!r}")
                if encoded_hash:
                    if not encoded_hash.startswith("sha256=") or not size_text.isdigit():
                        raise ValueError(f"{label} RECORD row is not hash/size authenticated: {relative}")
                    try:
                        decoded = base64.urlsafe_b64decode(
                            encoded_hash.removeprefix("sha256=") + "=="
                        )
                    except (ValueError, binascii.Error) as error:
                        raise ValueError(f"{label} RECORD hash is malformed: {relative}") from error
                    if len(decoded) != 32:
                        raise ValueError(f"{label} RECORD hash length is invalid: {relative}")
                elif size_text:
                    raise ValueError(f"{label} RECORD hash is missing: {relative}")
                elif relative != wheel_record_path and not generated_path:
                    raise ValueError(f"{label} RECORD unhashed row is not generated metadata: {relative}")
                values[relative] = (relative, encoded_hash, size_text)
            return values

        generated_paths = {
            "../../../bin/f2py",
            "../../../bin/numpy-config",
            *(
                f"{PurePosixPath(wheel_record_path).parent}/{name}"
                for name in GENERATED_NUMPY_METADATA
            ),
        }
        wheel_rows = parse_record(wheel_record_bytes, "wheel", False)
        installed_rows = parse_record(installed_record_bytes, "installed", True)
        wheel_paths = set(wheel_rows)
        installed_paths = set(installed_rows)
        if wheel_record_path not in wheel_rows:
            raise ValueError("wheel RECORD does not contain its own RECORD row")
        if wheel_files != wheel_paths:
            unexpected = sorted(wheel_files - wheel_paths)
            missing = sorted(wheel_paths - wheel_files)
            raise ValueError(
                f"wheel RECORD member set mismatch: unexpected={unexpected}, missing={missing}"
            )
        for path, fields in wheel_rows.items():
            if installed_rows.get(path) != fields:
                raise ValueError(f"installed NumPy RECORD row differs from wheel: {path}")
        installed_only = installed_paths - wheel_paths
        if installed_only != generated_paths:
            unexpected_generated = installed_only - generated_paths
            missing_generated = generated_paths - installed_only
            raise ValueError(
                "installed NumPy RECORD has unexpected generated rows: "
                + "unexpected="
                + ",".join(sorted(unexpected_generated))
                + "; missing="
                + ",".join(sorted(missing_generated))
            )

        def verify_file(
            relative: str, encoded_hash: str, size_text: str, *, generated_row: bool
        ) -> dict[str, Any]:
            relative_path = Path(relative)
            target = Path(distribution.locate_file(relative_path))
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"RECORD installed file is missing or symlinked: {relative}")
            if generated_row:
                site_packages = Path(distribution.locate_file("")).resolve()
                target_resolved = target.resolve()
                if relative.startswith("../../../bin/"):
                    expected = venv_root.resolve() / "bin" / PurePosixPath(relative).name
                else:
                    expected = site_packages / PurePosixPath(relative)
                if target_resolved != expected or not target_resolved.is_relative_to(
                    venv_root.resolve()
                ):
                    raise ValueError(f"generated RECORD path escapes the prepared venv: {relative}")
            if not encoded_hash or not size_text:
                raise ValueError(f"generated RECORD row is not hash/size authenticated: {relative}")
            expected_hash = base64.urlsafe_b64decode(
                encoded_hash.removeprefix("sha256=") + "=="
            )
            actual_digest = sha256(target)
            actual_size = target.stat().st_size
            if bytes.fromhex(actual_digest) != expected_hash or actual_size != int(size_text):
                raise ValueError(f"RECORD hash/size mismatch: {relative}")
            return {
                "path": relative,
                "bytes": actual_size,
                "sha256": actual_digest,
                "generated": generated_row,
            }

        for relative, (_path, encoded_hash, size_text) in wheel_rows.items():
            target = Path(distribution.locate_file(relative))
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"RECORD installed file is missing or symlinked: {relative}")
            if not encoded_hash and not size_text:
                if relative != wheel_record_path:
                    raise ValueError(f"wheel RECORD unhashed row is not its own RECORD: {relative}")
                rows.append(
                    {
                        "path": relative,
                        "bytes": target.stat().st_size,
                        "sha256": None,
                        "generated": True,
                    }
                )
            else:
                rows.append(verify_file(relative, encoded_hash, size_text, generated_row=False))
        for relative in sorted(installed_only):
            _path, encoded_hash, size_text = installed_rows[relative]
            generated.append(verify_file(relative, encoded_hash, size_text, generated_row=True))

        # package roots come exclusively from the wheel RECORD.  This scan is
        # deliberately independent of installed-only RECORD additions.
        record_paths = set(wheel_rows)
        generated_seen = {row["path"] for row in generated}
        package_roots = sorted({PurePosixPath(path).parts[0] for path in record_paths})
        site_packages = Path(distribution.locate_file(""))
        for root_name in package_roots:
            root = site_packages / root_name
            if not root.exists() or root.is_symlink():
                continue
            paths = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in paths:
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(site_packages).as_posix()
                if relative in record_paths or relative in generated_seen:
                    continue
                if path.suffix == ".pyc":
                    raise ValueError(f"unrecorded generated pyc file: {relative}")
                raise ValueError(f"unrecorded NumPy installed file: {relative}")
    except (OSError, ValueError) as error:
        failures.append(f"NumPy installed RECORD audit failed: {error}")
        evidence["status"] = "FAIL_RECORD_INVALID"
        evidence["rows"] = rows
        evidence["generated"] = generated
        return evidence
    evidence["status"] = "PASS"
    evidence["rows"] = rows
    evidence["generated"] = sorted(generated, key=lambda row: row["path"])
    return evidence


def _validate_preparation(path: Path) -> dict[str, Any]:
    """Validate the exact no-BLAS build artifact handed to the installed audit."""
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("NumPy preparation JSON must be an absolute regular non-symlink file")
    document = _strict_json(path)
    if not isinstance(document, dict):
        raise RuntimeError("NumPy preparation JSON must be an object")
    if (
        document.get("schema") != "vokra-zonos-numpy-no-blas-preparation-v2"
        or document.get("status") != "PREPARED_NO_BLAS"
        or document.get("publication") != PUBLICATION
    ):
        raise RuntimeError("NumPy preparation status/publication is not exact")
    project = document.get("project")
    if not isinstance(project, dict) or project != {
        "name": PROJECT_DIR.name,
        "pyproject_sha256": EXPECTED_PROJECT_SHA256,
        "uv_lock_sha256": EXPECTED_LOCK_SHA256,
        "constraints_sha256": EXPECTED_CONSTRAINTS_SHA256,
    }:
        raise RuntimeError("NumPy preparation project identity drifted")
    if document.get("preparer_sha256") != sha256(PREPARER_PATH):
        raise RuntimeError("NumPy preparation helper identity drifted")
    sdist = document.get("sdist")
    if sdist != {
        "url": NUMPY_SDIST_URL,
        "sha256": NUMPY_SDIST_SHA256,
        "bytes": NUMPY_SDIST_BYTES,
    }:
        raise RuntimeError("NumPy preparation sdist identity drifted")
    wheel = document.get("wheel")
    if not isinstance(wheel, dict):
        raise RuntimeError("NumPy preparation wheel identity is missing")
    basename = wheel.get("basename")
    if not isinstance(basename, str) or not re.fullmatch(r"numpy-2\.2\.2-[^/]+\.whl", basename):
        raise RuntimeError("NumPy preparation wheel basename is not exact")
    preparation_root = path.parent.resolve()
    wheel_path = preparation_root / "wheelhouse" / basename
    sdist_path = preparation_root / "numpy-2.2.2.tar.gz"
    for artifact in (sdist_path, wheel_path):
        if not artifact.is_file() or artifact.is_symlink():
            raise RuntimeError(f"NumPy preparation artifact is missing or symlinked: {artifact}")
    _validate_prepared_wheel(wheel_path, wheel)
    if sdist_path.stat().st_size != NUMPY_SDIST_BYTES or sha256(sdist_path) != NUMPY_SDIST_SHA256:
        raise RuntimeError("NumPy preparation artifact bytes/hash drifted")
    sums_path = preparation_root / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise RuntimeError("NumPy preparation SHA256SUMS is missing or symlinked")
    expected_sum_lines = {
        f"{sha256(path)}  preparation.json",
        f"{sha256(sdist_path)}  numpy-2.2.2.tar.gz",
        f"{sha256(wheel_path)}  wheelhouse/{wheel_path.name}",
    }
    actual_sum_lines = {
        line.strip() for line in sums_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    if actual_sum_lines != expected_sum_lines:
        raise RuntimeError("NumPy preparation SHA256SUMS does not bind the artifacts")
    build = document.get("build")
    if build != {
        "no_build_isolation": True,
        "arguments": ["-Dblas=none", "-Dlapack=none", "-Dallow-noblas=true"],
    }:
        raise RuntimeError("NumPy preparation build arguments drifted")
    venv = document.get("venv")
    if not isinstance(venv, dict) or not isinstance(venv.get("path"), str):
        raise RuntimeError("NumPy preparation venv identity is missing")
    venv_path = Path(venv["path"])
    if not venv_path.is_absolute() or venv_path.resolve() != preparation_root / "venv":
        raise RuntimeError("NumPy preparation venv path is not canonical")
    interpreter = venv.get("interpreter")
    if not isinstance(interpreter, dict):
        raise RuntimeError("NumPy preparation interpreter identity is missing")
    executable = Path(str(interpreter.get("executable", "")))
    if (
        interpreter.get("prefix") != str(venv_path)
        or executable != venv_path / "bin" / "python"
        or interpreter.get("implementation") != "CPython"
        or not re.fullmatch(r"3\.12\.\d+", str(interpreter.get("version", "")))
    ):
        raise RuntimeError("NumPy preparation interpreter identity is not venv-bound")
    return document


def _validate_prepared_wheel(path: Path, wheel: dict[str, Any]) -> None:
    if (
        not isinstance(wheel.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"])
        or not isinstance(wheel.get("bytes"), int)
        or wheel["bytes"] < 1
        or wheel.get("sha256") != sha256(path)
        or wheel.get("bytes") != path.stat().st_size
    ):
        raise RuntimeError("NumPy prepared wheel bytes/hash drifted")


def _validate_numpy_native_policy(native: list[Any], policy: Any) -> None:
    numpy_rows = [
        row for row in native
        if isinstance(row, dict)
        and str(row.get("path", "")).casefold().startswith(("numpy/", "numpy.libs/"))
    ]
    if (
        not isinstance(policy, dict)
        or policy.get("schema") != "vokra-zonos-numpy-native-policy-v1"
        or policy.get("status") != "PASS_NO_FORBIDDEN_BLAS"
        or policy.get("forbidden_boundaries") != []
        or policy.get("native_file_count") != len(numpy_rows)
        or not numpy_rows
    ):
        raise RuntimeError("NumPy native policy does not prove a no-BLAS closure")


def validate_report(path: Path) -> None:
    """Reject incomplete/fallback evidence before it can be owner-reviewed."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("audit evidence must be a regular non-symlink file")
    report = _strict_json(path)
    if report.get("schema") != "vokra-zonos-dependency-audit-v1":
        raise RuntimeError("audit evidence schema is not exact")
    if report.get("status") != AUDIT_STATUS or report.get("publication") != PUBLICATION:
        raise RuntimeError("audit evidence status/publication is not fail-closed")
    installed = report.get("installed")
    scope = report.get("candidate_scope")
    if not isinstance(installed, dict) or installed.get("status") != "COLLECTED":
        raise RuntimeError("installed closure was not completely collected")
    if not isinstance(scope, dict):
        raise RuntimeError("candidate scope is missing")
    hex64 = re.compile(r"[0-9a-f]{64}")
    required_scope = {
        "schema", "lock_rows_sha256", "preparation_path", "preparation_sha256", "constraints_sha256",
        "sdist_identity", "wheel_identity", "installed_closure_sha256", "native_files_sha256",
        "publisher_files_sha256", "numpy_record_sha256", "numpy_native_policy_sha256", "numpy_runtime_config_sha256",
        "publisher_archive_manifest_sha256", "failures",
        "model_access", "source_access", "checkpoint_access", "publication",
        "execution_identity",
    }
    if not required_scope.issubset(scope):
        raise RuntimeError("candidate scope is incomplete")
    if scope["schema"] != "vokra-zonos-dependency-approval-scope-v1":
        raise RuntimeError("candidate scope schema is not exact")
    if any(scope[key] is not False for key in ("model_access", "source_access", "checkpoint_access")):
        raise RuntimeError("audit evidence crosses the acquisition boundary")
    if scope["publication"] != PUBLICATION:
        raise RuntimeError("candidate scope publication is not NO_UPLOAD")
    if not isinstance(scope["preparation_sha256"], str) or not hex64.fullmatch(scope["preparation_sha256"]):
        raise RuntimeError("prepared NumPy identity is missing from candidate scope")
    preparation_path_value = report.get("preparation_path")
    if (
        not isinstance(preparation_path_value, str)
        or not Path(preparation_path_value).is_absolute()
        or scope["preparation_path"] != preparation_path_value
    ):
        raise RuntimeError("prepared NumPy path is missing from candidate scope")
    preparation_path = Path(preparation_path_value)
    if sha256(preparation_path) != scope["preparation_sha256"]:
        raise RuntimeError("candidate scope preparation identity does not match evidence directory")
    _require_no_bytecode_writes()
    preparation = _validate_preparation(preparation_path)
    prepared_venv = Path(preparation["venv"]["path"]).resolve()
    expected_executable = Path(preparation["venv"]["interpreter"]["executable"]).resolve()
    if Path(sys.prefix).resolve() != prepared_venv:
        raise RuntimeError("audit validator sys.prefix is outside the prepared venv")
    if Path(sys.executable).resolve() != expected_executable:
        raise RuntimeError("audit validator sys.executable is not the prepared interpreter")
    if installed.get("preparation") != preparation:
        raise RuntimeError("installed report preparation identity is not file-bound")
    if scope["constraints_sha256"] != preparation["project"]["constraints_sha256"]:
        raise RuntimeError("candidate scope constraints identity is not preparation-bound")
    if scope["sdist_identity"] != preparation["sdist"] or scope["wheel_identity"] != preparation["wheel"]:
        raise RuntimeError("candidate scope build artifacts are not preparation-bound")
    execution = report.get("execution_identity")
    if not isinstance(execution, dict) or scope["execution_identity"] != execution:
        raise RuntimeError("execution identity is missing or not scope-bound")
    expected_head = execution.get("expected_head")
    if not isinstance(expected_head, str):
        raise RuntimeError("execution identity HEAD is missing")
    if execution_identity(expected_head) != execution:
        raise RuntimeError("audit code, repository, or platform identity drifted")
    for key in (
        "lock_rows_sha256", "installed_closure_sha256", "native_files_sha256",
        "publisher_files_sha256", "numpy_record_sha256", "numpy_native_policy_sha256", "publisher_archive_manifest_sha256",
    ):
        if not isinstance(scope[key], str) or not hex64.fullmatch(scope[key]):
            raise RuntimeError(f"candidate scope digest is missing or malformed: {key}")
    if not isinstance(report.get("candidate_scope_sha256"), str) or not hex64.fullmatch(report["candidate_scope_sha256"]):
        raise RuntimeError("candidate scope canonical digest is missing or malformed")
    if report["candidate_scope_sha256"] != _digest(scope):
        raise RuntimeError("candidate scope canonical digest mismatch")
    lock = report.get("lock")
    if not isinstance(lock, dict) or not isinstance(lock.get("rows"), list):
        raise RuntimeError("lock rows are missing")
    if report.get("project") != project_identity():
        raise RuntimeError("dedicated Zonos project identity is not report-bound")
    if lock["rows"] != _lock_rows():
        raise RuntimeError("audit lock rows do not match the reviewed dedicated lock")
    if scope["lock_rows_sha256"] != _digest(lock["rows"]):
        raise RuntimeError("lock row digest mismatch")
    failures = installed.get("failures")
    if not isinstance(failures, list) or report.get("failures") != failures or scope["failures"] != failures:
        raise RuntimeError("audit failures are not consistently bound")
    installed_rows = installed.get("installed_distributions")
    native = installed.get("native_files")
    publisher = installed.get("publisher_license_notice_files")
    digests = installed.get("digests")
    archive = installed.get("publisher_archive")
    numpy_native_policy = installed.get("numpy_native_policy")
    if (
        not all(isinstance(value, list) for value in (installed_rows, native, publisher))
        or not isinstance(digests, dict)
        or not isinstance(numpy_native_policy, dict)
    ):
        raise RuntimeError("installed closure/native/publisher facts are incomplete")
    _validate_numpy_native_policy(native, numpy_native_policy)
    numpy_config = installed.get("numpy_runtime_config")
    if (
        not isinstance(numpy_config, dict)
        or numpy_config.get("schema") != "vokra-zonos-numpy-runtime-config-v1"
        or numpy_config.get("status") != "PASS_NO_FORBIDDEN_BLAS"
        or numpy_config.get("forbidden_boundaries") != []
        or scope["numpy_runtime_config_sha256"] != _digest(numpy_config)
    ):
        raise RuntimeError("NumPy runtime build config does not prove a no-BLAS closure")
    if digests.get("numpy_runtime_config_sha256") != _digest(numpy_config):
        raise RuntimeError("NumPy runtime config digest is not hash-bound")
    numpy_installation = installed.get("numpy_installation")
    if (
        not isinstance(numpy_installation, dict)
        or numpy_installation.get("version") != "2.2.2"
        or numpy_installation.get("expected_version") != "2.2.2"
        or numpy_installation.get("wheel") != preparation["wheel"]
    ):
        raise RuntimeError("installed NumPy wheel identity is incomplete")
    direct_url = numpy_installation.get("direct_url")
    wheel_path = preparation_path.parent / "wheelhouse" / preparation["wheel"]["basename"]
    try:
        _validate_numpy_direct_url(direct_url, wheel_path, preparation["wheel"])
    except RuntimeError as error:
        raise RuntimeError(f"installed NumPy direct_url is not wheel-bound: {error}") from error
    numpy_record = installed.get("numpy_record")
    if (
        not isinstance(numpy_record, dict)
        or numpy_record.get("schema") != "vokra-zonos-numpy-record-v2"
        or numpy_record.get("status") != "PASS"
        or not isinstance(numpy_record.get("rows"), list)
        or not isinstance(numpy_record.get("generated"), list)
        or scope["numpy_record_sha256"] != _digest(numpy_record)
        or digests.get("numpy_record_sha256") != _digest(numpy_record)
    ):
        raise RuntimeError("installed NumPy RECORD evidence is incomplete or not hash-bound")
    current_numpy = importlib.metadata.distribution("numpy")
    record_failures: list[str] = []
    current_numpy_record = _numpy_record_evidence(
        current_numpy, wheel_path, prepared_venv, record_failures
    )
    if record_failures or current_numpy_record != numpy_record:
        raise RuntimeError("installed NumPy RECORD/file identity drifted")
    if not isinstance(archive, dict):
        raise RuntimeError("publisher archive evidence is missing")
    archive_root = Path(archive.get("directory", ""))
    manifest_path = Path(archive.get("manifest", ""))
    if (
        not archive_root.is_absolute()
        or archive_root.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise RuntimeError("publisher archive path is unsafe")
    repository_root = REPOSITORY_ROOT.resolve()
    if archive_root == repository_root or repository_root in archive_root.parents:
        raise RuntimeError("publisher archive must be outside the checkout")
    if manifest_path.parent != archive_root or archive.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("publisher archive manifest identity mismatch")
    manifest_rows = _strict_json(manifest_path)
    if not isinstance(manifest_rows, list) or manifest_rows != archive.get("files"):
        raise RuntimeError("publisher archive manifest contents are not bound")
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise RuntimeError("publisher archive manifest row is malformed")
        relative = Path(row.get("archive_path", ""))
        payload = archive_root / relative
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("publisher archive path traversal")
        if (
            not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or not payload.is_file()
            or payload.is_symlink()
            or payload.stat().st_size != row["bytes"]
            or sha256(payload) != row.get("sha256")
        ):
            raise RuntimeError("publisher archive payload identity mismatch")
    expected_archive_paths = {Path(row["archive_path"]) for row in manifest_rows}
    actual_archive_paths = {
        path.relative_to(archive_root)
        for path in archive_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_archive_paths != expected_archive_paths:
        raise RuntimeError("publisher archive contains unexpected or missing files")
    archived_identities = {
        (row.get("distribution"), row.get("path"), row.get("bytes"), row.get("sha256"))
        for row in manifest_rows
    }
    lock_rows_by_name = {
        pep503_name(row["name"]): row for row in lock.get("rows", []) if isinstance(row, dict)
    }
    expected_digests = {
        "installed_closure_sha256": _digest(installed_rows),
        "native_files_sha256": _digest(native),
        "publisher_files_sha256": _digest(publisher),
        "numpy_record_sha256": _digest(numpy_record),
    }
    for key, digest in expected_digests.items():
        if digests.get(key) != digest or scope[key] != digest:
            raise RuntimeError(f"{key} is not hash-bound")
    if scope["publisher_archive_manifest_sha256"] != archive["manifest_sha256"]:
        raise RuntimeError("publisher archive manifest is not scope-bound")
    if scope["numpy_native_policy_sha256"] != _digest(numpy_native_policy):
        raise RuntimeError("NumPy native policy is not scope-bound")
    for row in native:
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not hex64.fullmatch(row["sha256"]):
            raise RuntimeError("native file hash evidence is malformed")
    for row in publisher:
        if not isinstance(row, dict) or row.get("status") == "MISSING":
            raise RuntimeError("publisher LICENSE/NOTICE evidence is missing")
        if not isinstance(row.get("sha256"), str) or not hex64.fullmatch(row["sha256"]):
            raise RuntimeError("publisher LICENSE/NOTICE hash evidence is malformed")
        if (row.get("distribution"), row.get("path"), row.get("bytes"), row.get("sha256")) not in archived_identities:
            raise RuntimeError("publisher evidence is absent from the legal archive")
        source = row.get("source")
        if source is not None:
            if not isinstance(source, dict) or source.get("kind") != "locked-sdist":
                raise RuntimeError("publisher fallback source identity is malformed")
            locked = lock_rows_by_name.get(pep503_name(str(row.get("distribution", ""))))
            locked_sdist = locked.get("sdist") if locked is not None else None
            expected_source = {
                "kind": "locked-sdist",
                "url": locked_sdist.get("url") if isinstance(locked_sdist, dict) else None,
                "sha256": (
                    locked_sdist.get("hash", "").removeprefix("sha256:")
                    if isinstance(locked_sdist, dict) and isinstance(locked_sdist.get("hash"), str)
                    else None
                ),
                "bytes": locked_sdist.get("size") if isinstance(locked_sdist, dict) else None,
            }
            if source != expected_source or "archive_path" not in row:
                raise RuntimeError("publisher fallback is not bound to the locked sdist")


def self_test() -> None:
    assert project_identity()["python"] == "3.12"
    assert pep503_name("Foo_bar.baz") == "foo-bar-baz"
    original_bytecode_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    original_bytecode_flag = sys.dont_write_bytecode
    try:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        _require_no_bytecode_writes()
        os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        sys.dont_write_bytecode = False
        try:
            _require_no_bytecode_writes()
        except RuntimeError:
            pass
        else:
            raise AssertionError("absent bytecode guard must fail closed")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "0"
        sys.dont_write_bytecode = False
        try:
            _require_no_bytecode_writes()
        except RuntimeError:
            pass
        else:
            raise AssertionError("false bytecode guard must fail closed")
    finally:
        if original_bytecode_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = original_bytecode_env
        sys.dont_write_bytecode = original_bytecode_flag
    report = audit()
    assert report["status"] == AUDIT_STATUS
    assert report["publication"] == PUBLICATION
    assert report["lock"]["package_count"] == 29
    assert not set(row["name"].lower() for row in report["lock"]["rows"]) & FORBIDDEN_PACKAGES
    active = _active_lock_rows(report["lock"]["rows"])
    assert len(active) == 25
    assert all(row["source"].get("virtual") is None for row in active)
    assert {row["version"] for row in active if row["name"] in {"torch", "torchaudio"}} == {"2.6.0+cpu"}
    darwin = _active_lock_rows(
        report["lock"]["rows"],
        {
            "sys_platform": "darwin",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "python_version": "3.12",
            "python_full_version": "3.12.0",
        },
    )
    darwin_names = {row["name"] for row in darwin}
    assert "colorama" not in darwin_names
    assert {row["version"] for row in darwin if row["name"] == "torch"} == {"2.6.0"}
    assert {row["version"] for row in darwin if row["name"] == "torchaudio"} == {"2.6.0"}
    ambiguous = list(report["lock"]["rows"])
    ambiguous.append(dict(next(row for row in ambiguous if row["name"] == "numpy"), version="2.2.3"))
    try:
        _active_lock_rows(ambiguous)
    except RuntimeError:
        pass
    else:
        raise AssertionError("ambiguous active lock closure must fail closed")
    assert report["candidate_scope"]["model_access"] is False
    assert report["candidate_scope"]["publication"] == PUBLICATION
    assert report["candidate_scope"]["installed_closure_sha256"] is None
    assert re.fullmatch(r"[0-9a-f]{64}", report["candidate_scope_sha256"])
    assert report["candidate_scope_sha256"] == _digest(report["candidate_scope"])
    config_forbidden = _numpy_config_forbidden(
        {"Build Dependencies": {"blas": {"name": "openblas", "found": True}}}
    )
    assert "openblas" in config_forbidden
    try:
        _validate_numpy_native_policy(
            [],
            {
                "schema": "vokra-zonos-numpy-native-policy-v1",
                "status": "PASS_NO_FORBIDDEN_BLAS",
                "native_file_count": 0,
                "forbidden_boundaries": [],
            },
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero NumPy native files must never validate as PASS")
    synthetic_native = Path(tempfile.mkdtemp(prefix="vokra-zonos-native-")) / "synthetic.so"
    synthetic_native.write_bytes(b"\x7fELFsynthetic")
    size, digest, is_elf = _streaming_binary_facts(synthetic_native)
    assert (size, digest, is_elf) == (synthetic_native.stat().st_size, sha256(synthetic_native), True)
    calls: list[Path] = []
    original_elf_needed = globals()["_elf_needed"]
    globals()["_elf_needed"] = lambda path: calls.append(path) or sorted(
        ["libgfortran.so.5", "libquadmath.so.0", "libopenblas.so.0"]
    )
    native_failures: list[str] = []
    try:
        native_record = _native_record(synthetic_native, synthetic_native.parent, native_failures)
        assert native_record["format"] == "ELF"
        assert native_record["needed"] == [
            "libgfortran.so.5", "libopenblas.so.0", "libquadmath.so.0"
        ]
    finally:
        globals()["_elf_needed"] = original_elf_needed
        synthetic_native.unlink(missing_ok=True)
        synthetic_native.parent.rmdir()
    assert calls == [synthetic_native]
    assert any("policy-review" in failure for failure in native_failures)
    assert all(
        token in " ".join(native_failures)
        for token in ("libgfortran", "libquadmath", "libopenblas")
    )
    direct_url_root = Path(tempfile.mkdtemp(prefix="vokra-zonos-direct-url-"))
    direct_url_wheel = direct_url_root / "numpy-2.2.2-cp312-cp312-linux_x86_64.whl"
    direct_url_wheel.write_bytes(b"wheel")
    wheel_identity = {"sha256": sha256(direct_url_wheel), "bytes": direct_url_wheel.stat().st_size}
    _validate_numpy_direct_url(
        {"url": direct_url_wheel.as_uri(), "archive_info": {}},
        direct_url_wheel,
        wheel_identity,
    )
    direct_url_wheel.write_bytes(b"tampered wheel")
    try:
        _validate_prepared_wheel(direct_url_wheel, wheel_identity)
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered prepared wheel must fail closed")
    direct_url_wheel.write_bytes(b"wheel")
    try:
        _validate_numpy_direct_url(
            {"url": direct_url_wheel.as_uri() + "?tampered=1", "archive_info": {}},
            direct_url_wheel,
            wheel_identity,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered direct_url URL must fail closed")
    try:
        _validate_numpy_direct_url(
            {
                "url": direct_url_wheel.as_uri(),
                "archive_info": {"hash": "sha256:" + wheel_identity["sha256"]},
            },
            direct_url_wheel,
            wheel_identity,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("tampered direct_url wheel hash must fail closed")
    synthetic_venv = Path(tempfile.mkdtemp(prefix="vokra-zonos-record-venv-"))
    record_root = synthetic_venv / "lib" / "python3.12" / "site-packages"
    record_root.mkdir(parents=True)
    dist_info = record_root / "numpy-2.2.2.dist-info"
    package_file = record_root / "numpy" / "core.py"
    dist_info.mkdir(parents=True)
    package_file.parent.mkdir()
    package_file.write_bytes(b"numpy payload")
    for generated_name in sorted(GENERATED_NUMPY_METADATA):
        (dist_info / generated_name).write_bytes(generated_name.encode())
    record_path = dist_info / "RECORD"
    package_hash = base64.urlsafe_b64encode(bytes.fromhex(sha256(package_file))).decode().rstrip("=")
    record_path.write_text(
        "numpy/core.py,sha256=" + package_hash + "," + str(package_file.stat().st_size) + "\n"
        "numpy-2.2.2.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    original_record_bytes = record_path.read_bytes()
    synthetic_wheel = direct_url_root / "numpy-2.2.2-cp312-cp312-linux_x86_64.whl"

    def write_synthetic_wheel(
        destination: Path,
        record_bytes: bytes | None = original_record_bytes,
        duplicate: bool = False,
        include_record: bool = True,
        symlink: bool = False,
    ) -> None:
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("numpy/core.py", package_file.read_bytes())
            if duplicate:
                archive.writestr("numpy/core.py", package_file.read_bytes())
            if symlink:
                link = zipfile.ZipInfo("numpy/link.so")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, b"link")
            if include_record and record_bytes is not None:
                archive.writestr("numpy-2.2.2.dist-info/RECORD", record_bytes)

    write_synthetic_wheel(synthetic_wheel)
    class SyntheticDistribution:
        def read_text(self, name: str) -> str | None:
            if name == "RECORD":
                return record_path.read_text(encoding="utf-8")
            return None

        def locate_file(self, relative: str | Path) -> Path:
            if str(relative).startswith("../../../bin/"):
                return synthetic_venv / "bin" / PurePosixPath(str(relative)).name
            return record_root / relative

    record_failures: list[str] = []
    record_evidence = _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )
    assert record_evidence["status"] == "FAIL_RECORD_INVALID"
    assert any("missing=../../../bin/f2py" in failure for failure in record_failures)
    assert record_evidence["wheel"]["bytes"] == len(original_record_bytes)
    assert record_evidence["wheel"]["sha256"] == hashlib.sha256(original_record_bytes).hexdigest()
    script_root = synthetic_venv / "bin"
    script_root.mkdir()
    script_paths = {"../../../bin/f2py": script_root / "f2py", "../../../bin/numpy-config": script_root / "numpy-config"}
    for script_path in script_paths.values():
        script_path.write_bytes(b"#!/usr/bin/env python3\n")

    def authenticated_record_line(relative: str, target: Path) -> bytes:
        encoded = base64.urlsafe_b64encode(bytes.fromhex(sha256(target))).decode().rstrip("=")
        return f"{relative},sha256={encoded},{target.stat().st_size}\n".encode()

    script_record_bytes = b"".join(
        authenticated_record_line(relative, target) for relative, target in script_paths.items()
    )
    metadata_record_bytes = b"".join(
        authenticated_record_line(
            "numpy-2.2.2.dist-info/" + name,
            dist_info / name,
        )
        for name in sorted(GENERATED_NUMPY_METADATA)
    )
    extra_record_bytes = original_record_bytes + script_record_bytes + metadata_record_bytes
    record_path.write_bytes(extra_record_bytes)
    record_failures = []
    extra_evidence = _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )
    assert not record_failures and extra_evidence["status"] == "PASS", record_failures
    assert {row["path"] for row in extra_evidence["generated"]} == {
        *script_paths,
        *("numpy-2.2.2.dist-info/" + name for name in GENERATED_NUMPY_METADATA),
    }
    original_bytecode_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    original_bytecode_flag = sys.dont_write_bytecode
    try:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        second_evidence = _numpy_record_evidence(
            SyntheticDistribution(), synthetic_wheel, synthetic_venv, []
        )
        assert second_evidence == extra_evidence
        assert not any(
            row.get("reason") == "pyc-not-wheel-recorded"
            for row in second_evidence["generated"]
        )
    finally:
        if original_bytecode_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = original_bytecode_env
        sys.dont_write_bytecode = original_bytecode_flag
    pycache = record_root / "numpy" / "__pycache__"
    pycache.mkdir()
    (pycache / "generated.cpython-312.pyc").write_bytes(b"pyc")
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("unrecorded generated pyc" in failure for failure in record_failures)
    shutil.rmtree(pycache)
    for missing_row in (
        b"numpy-2.2.2.dist-info/REQUESTED",
        b"../../../bin/f2py",
    ):
        missing_record = extra_record_bytes.replace(
            next(
                line
                for line in extra_record_bytes.splitlines(keepends=True)
                if line.startswith(missing_row + b",")
            ),
            b"",
        )
        record_path.write_bytes(missing_record)
        record_failures = []
        assert _numpy_record_evidence(
            SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
        )["status"] == "FAIL_RECORD_INVALID"
        assert any("missing=" in failure for failure in record_failures)
    record_path.write_bytes(
        extra_record_bytes.replace(
            next(
                line
                for line in extra_record_bytes.splitlines(keepends=True)
                if line.startswith(b"../../../bin/f2py,")
            ),
            b"../../../bin/f2py,,\n",
        )
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("hash/size authenticated" in failure for failure in record_failures)
    record_path.write_bytes(
        extra_record_bytes.replace(
            next(
                line
                for line in extra_record_bytes.splitlines(keepends=True)
                if line.startswith(b"../../../bin/f2py,")
            ),
            b"../../../bin/f2py,sha256=bad,22\n",
        )
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("hash" in failure for failure in record_failures)
    record_path.write_bytes(extra_record_bytes)
    f2py_path = script_paths["../../../bin/f2py"]
    outside_target = synthetic_venv / "outside-f2py"
    outside_target.write_bytes(f2py_path.read_bytes())
    f2py_path.unlink()
    f2py_path.symlink_to(outside_target)
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("missing or symlinked" in failure for failure in record_failures)
    f2py_path.unlink()
    f2py_path.mkdir()
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("missing or symlinked" in failure for failure in record_failures)
    f2py_path.rmdir()
    f2py_path.write_bytes(b"#!/usr/bin/env python3\n")
    record_path.write_bytes(
        extra_record_bytes
        + authenticated_record_line("../../bin/f2py", f2py_path)
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("unsafe" in failure for failure in record_failures)
    record_path.write_bytes(original_record_bytes)
    altered_wheel = direct_url_root / "altered-record.whl"
    altered_hash = base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")
    write_synthetic_wheel(
        altered_wheel,
        b"numpy/core.py,sha256=" + altered_hash.encode() + b",12\n"
        b"numpy-2.2.2.dist-info/RECORD,,\n",
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), altered_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("differs from wheel" in failure for failure in record_failures)
    missing_row_wheel = direct_url_root / "missing-row.whl"
    write_synthetic_wheel(
        missing_row_wheel,
        b"numpy-2.2.2.dist-info/RECORD,,\n",
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), missing_row_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("member set mismatch" in failure for failure in record_failures)
    unknown_file = record_root / "numpy" / "unknown.py"
    unknown_file.write_bytes(b"unknown")
    record_path.write_bytes(
        original_record_bytes
        + authenticated_record_line("numpy/unknown.py", unknown_file)
    )
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("unexpected generated rows" in failure for failure in record_failures)
    unknown_file.unlink(missing_ok=True)
    record_path.write_bytes(extra_record_bytes)
    package_file.write_bytes(b"tampered numpy payload")
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("hash/size mismatch" in failure for failure in record_failures)
    package_file.write_bytes(b"numpy payload")
    record_path.write_text("../escape,sha256=bad,1\n", encoding="utf-8")
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("unsafe" in failure for failure in record_failures)
    record_path.write_bytes(original_record_bytes)
    package_file.write_bytes(b"tampered numpy payload")
    tampered_installed_record = extra_record_bytes.replace(
        next(
            line
            for line in extra_record_bytes.splitlines(keepends=True)
            if line.startswith(b"numpy/core.py,")
        ),
        authenticated_record_line("numpy/core.py", package_file),
    )
    record_path.write_bytes(tampered_installed_record)
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), synthetic_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("differs from wheel" in failure for failure in record_failures)
    package_file.write_bytes(b"numpy payload")
    record_path.write_bytes(original_record_bytes)
    tampered_wheel = direct_url_root / "tampered-record.whl"
    write_synthetic_wheel(tampered_wheel, b"../escape,sha256=bad,1\n")
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), tampered_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("wheel RECORD" in failure for failure in record_failures)
    duplicate_wheel = direct_url_root / "duplicate-record.whl"
    write_synthetic_wheel(duplicate_wheel, duplicate=True)
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), duplicate_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("duplicated" in failure for failure in record_failures)
    missing_wheel = direct_url_root / "missing-record.whl"
    write_synthetic_wheel(missing_wheel, include_record=False)
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), missing_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("RECORD is missing" in failure for failure in record_failures)
    symlink_wheel = direct_url_root / "symlink-member.whl"
    write_synthetic_wheel(symlink_wheel, symlink=True)
    record_failures = []
    assert _numpy_record_evidence(
        SyntheticDistribution(), symlink_wheel, synthetic_venv, record_failures
    )["status"] == "FAIL_RECORD_INVALID"
    assert any("symlink" in failure for failure in record_failures)
    direct_url_wheel.unlink(missing_ok=True)
    shutil.rmtree(direct_url_root, ignore_errors=True)
    shutil.rmtree(script_root, ignore_errors=True)
    shutil.rmtree(synthetic_venv, ignore_errors=True)
    shutil.rmtree(record_root, ignore_errors=True)
    synthetic_sdist_fd, synthetic_sdist_name = tempfile.mkstemp(
        prefix="vokra-zonos-sdist-", suffix=".tar.gz"
    )
    os.close(synthetic_sdist_fd)
    synthetic_sdist = Path(synthetic_sdist_name)
    archive_root = Path(tempfile.mkdtemp(prefix="vokra-zonos-sdist-archive-"))
    try:
        with tarfile.open(synthetic_sdist, "w:gz") as package:
            payload = b"synthetic Apache license bytes\n"
            member = tarfile.TarInfo("safetensors-0.5.3/LICENSE")
            member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
        sdist_bytes = synthetic_sdist.stat().st_size
        sdist_sha = sha256(synthetic_sdist)
        safetensors_row = next(row for row in report["lock"]["rows"] if row["name"] == "safetensors")
        synthetic_row = dict(safetensors_row)
        synthetic_row["sdist"] = {
            "url": safetensors_row["sdist"]["url"],
            "hash": f"sha256:{sdist_sha}",
            "size": sdist_bytes,
        }

        class SyntheticResponse:
            def __init__(self, value: bytes):
                self.value = value
                self.headers = {"Content-Length": str(len(value))}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return safetensors_row["sdist"]["url"]

            def read(self, size: int = -1) -> bytes:
                value, self.value = self.value[:size], self.value[size:]
                return value

        original_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda _url, timeout=60: SyntheticResponse(synthetic_sdist.read_bytes())
        fallback_failures: list[str] = []
        recovered = _sdist_license_fallback(
            "safetensors", synthetic_row, archive_root, fallback_failures
        )
        urllib.request.urlopen = original_urlopen
        assert not fallback_failures
        assert len(recovered) == 1
        assert recovered[0]["source"] == {
            "kind": "locked-sdist",
            "url": safetensors_row["sdist"]["url"],
            "sha256": sdist_sha,
            "bytes": sdist_bytes,
        }
        assert (archive_root / recovered[0]["archive_path"]).is_file()
        def synthetic_row_for(path: Path) -> dict[str, Any]:
            value = dict(safetensors_row)
            value["sdist"] = {
                "url": safetensors_row["sdist"]["url"],
                "hash": f"sha256:{sha256(path)}",
                "size": path.stat().st_size,
            }
            return value

        malformed_failures: list[str] = []
        malformed_row = dict(synthetic_row)
        malformed_row["sdist"] = dict(synthetic_row["sdist"], url="http://files.pythonhosted.org/bad.tar.gz")
        assert not _sdist_license_fallback(
            "safetensors", malformed_row, archive_root / "bad-url", malformed_failures
        )
        assert any("exact files.pythonhosted.org path" in failure for failure in malformed_failures)

        oversized_failures: list[str] = []
        oversized_data = b"x" * (sdist_bytes + 1)

        class OversizedResponse(SyntheticResponse):
            def __init__(self, value: bytes):
                super().__init__(value)
                self.headers = {}

        original_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda _url, timeout=60: OversizedResponse(oversized_data)
        try:
            assert not _sdist_license_fallback(
                "safetensors", synthetic_row, archive_root / "oversized", oversized_failures
            )
        finally:
            urllib.request.urlopen = original_urlopen
        assert any("stream exceeds locked size" in failure for failure in oversized_failures)

        redirect_failures: list[str] = []

        class RedirectResponse(SyntheticResponse):
            def geturl(self):
                return "https://files.pythonhosted.org/redirected.tar.gz"

        urllib.request.urlopen = lambda _url, timeout=60: RedirectResponse(synthetic_sdist.read_bytes())
        try:
            assert not _sdist_license_fallback(
                "safetensors", synthetic_row, archive_root / "redirect", redirect_failures
            )
        finally:
            urllib.request.urlopen = original_urlopen
        assert any("redirect is forbidden" in failure for failure in redirect_failures)

        unsafe_sdist = synthetic_sdist.with_name("unsafe.tar.gz")
        with tarfile.open(unsafe_sdist, "w:gz") as package:
            safe_payload = b"safe\n"
            safe_member = tarfile.TarInfo("safetensors/LICENSE")
            safe_member.size = len(safe_payload)
            package.addfile(safe_member, io.BytesIO(safe_payload))
            duplicate = tarfile.TarInfo("safetensors/LICENSE")
            duplicate.size = len(safe_payload)
            package.addfile(duplicate, io.BytesIO(safe_payload))
            traversal = tarfile.TarInfo("../LICENSE")
            traversal.size = len(safe_payload)
            package.addfile(traversal, io.BytesIO(safe_payload))
            link = tarfile.TarInfo("LICENSE-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "safetensors/LICENSE"
            package.addfile(link)
            oversized_member = tarfile.TarInfo("LICENSE-oversized")
            oversized_member.size = MAX_PUBLISHER_MEMBER_BYTES + 1
            package.addfile(oversized_member, io.BytesIO(b"x" * (MAX_PUBLISHER_MEMBER_BYTES + 1)))
        unsafe_row = synthetic_row_for(unsafe_sdist)
        unsafe_failures: list[str] = []
        urllib.request.urlopen = lambda _url, timeout=60: SyntheticResponse(unsafe_sdist.read_bytes())
        unsafe_archive = archive_root / "unsafe"
        unsafe_archive.mkdir()
        try:
            unsafe_recovered = _sdist_license_fallback(
                "safetensors", unsafe_row, unsafe_archive, unsafe_failures
            )
        finally:
            urllib.request.urlopen = original_urlopen
        assert len(unsafe_recovered) == 1
        assert any("duplicate publisher member" in failure for failure in unsafe_failures)
        assert any("unsafe publisher member" in failure for failure in unsafe_failures)
        assert any("exceeds byte bound" in failure for failure in unsafe_failures)
        unsafe_sdist.unlink(missing_ok=True)
    finally:
        urllib.request.urlopen = original_urlopen
        synthetic_sdist.unlink(missing_ok=True)
        shutil.rmtree(archive_root, ignore_errors=True)
    complete = json.loads(json.dumps(report))
    archive_dir = Path(tempfile.mkdtemp(prefix="vokra-zonos-publisher-"))
    archive_manifest = archive_dir / "manifest.json"
    archive_manifest.write_text("[]\n", encoding="utf-8")
    preparation_path = archive_dir / "preparation.json"
    preparation_path.write_text("{}\n", encoding="utf-8")
    (archive_dir / "wheelhouse").mkdir()
    (archive_dir / "wheelhouse" / "numpy-2.2.2-cp312-cp312-linux_x86_64.whl").write_bytes(b"x")
    current_head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    original_execution_identity = globals()["execution_identity"]
    fake_execution_identity = {
        "expected_head": current_head,
        "actual_head": current_head,
        "dependency_audit_sha256": sha256(AUDITOR_PATH),
        "wrapper_sha256": sha256(WRAPPER_PATH),
        "preparer_sha256": sha256(PREPARER_PATH),
        "pyproject_sha256": sha256(PROJECT_PATH),
        "uv_lock_sha256": sha256(LOCK_PATH),
        "constraints_sha256": sha256(CONSTRAINTS_PATH),
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "sys_platform": "linux",
            "python": "3.12.0",
        },
    }
    globals()["execution_identity"] = lambda _expected: fake_execution_identity
    fake_preparation = {
        "project": {
            "name": PROJECT_DIR.name,
            "pyproject_sha256": EXPECTED_PROJECT_SHA256,
            "uv_lock_sha256": EXPECTED_LOCK_SHA256,
            "constraints_sha256": EXPECTED_CONSTRAINTS_SHA256,
        },
        "sdist": {"url": NUMPY_SDIST_URL, "sha256": NUMPY_SDIST_SHA256, "bytes": NUMPY_SDIST_BYTES},
        "wheel": {"basename": "numpy-2.2.2-cp312-cp312-linux_x86_64.whl", "sha256": "0" * 64, "bytes": 1},
        "venv": {
            "path": str(Path(sys.prefix).resolve()),
            "interpreter": {
                "prefix": str(Path(sys.prefix).resolve()),
                "executable": str(Path(sys.executable).resolve()),
                "implementation": "CPython",
                "version": platform.python_version(),
            },
        },
    }
    original_validate_preparation = globals()["_validate_preparation"]
    globals()["_validate_preparation"] = lambda _path: fake_preparation
    fake_numpy_config = {
        "schema": "vokra-zonos-numpy-runtime-config-v1",
        "status": "PASS_NO_FORBIDDEN_BLAS",
        "config": {},
        "forbidden_boundaries": [],
    }
    fake_numpy_record = {
        "schema": "vokra-zonos-numpy-record-v2",
        "status": "PASS",
        "rows": [],
        "generated": [],
    }
    complete["preparation_path"] = str(preparation_path)
    complete["execution_identity"] = execution_identity(current_head)
    complete["installed"]["publisher_archive"] = {
        "directory": str(archive_dir),
        "manifest": str(archive_manifest),
        "manifest_sha256": sha256(archive_manifest),
        "files": [],
    }
    complete["candidate_scope"]["execution_identity"] = complete["execution_identity"]
    complete["candidate_scope"]["publisher_archive_manifest_sha256"] = sha256(archive_manifest)
    complete["candidate_scope_sha256"] = _digest(complete["candidate_scope"])
    complete["installed"] = {
        "status": "COLLECTED",
        "active_lock_packages": active,
        "installed_distributions": [],
        "failures": [],
        "native_files": [],
        "numpy_native_policy": {
            "schema": "vokra-zonos-numpy-native-policy-v1",
            "status": "PASS_NO_FORBIDDEN_BLAS",
            "native_file_count": 0,
            "forbidden_boundaries": [],
        },
        "publisher_license_notice_files": [],
        "preparation": fake_preparation,
        "numpy_installation": {
            "version": "2.2.2",
            "expected_version": "2.2.2",
            "wheel": fake_preparation["wheel"],
            "direct_url": {
                "url": (archive_dir / "wheelhouse" / fake_preparation["wheel"]["basename"]).as_uri(),
                "archive_info": {},
            },
        },
        "numpy_record": fake_numpy_record,
        "numpy_runtime_config": fake_numpy_config,
        "publisher_archive": {
            "directory": str(archive_dir),
            "manifest": str(archive_manifest),
            "manifest_sha256": sha256(archive_manifest),
            "files": [],
        },
        "digests": {
            "installed_closure_sha256": _digest([]),
            "native_files_sha256": _digest([]),
            "publisher_files_sha256": _digest([]),
            "numpy_record_sha256": _digest(fake_numpy_record),
            "numpy_runtime_config_sha256": _digest(fake_numpy_config),
        },
    }
    complete["failures"] = []
    complete["candidate_scope"].update(
        {
            "installed_closure_sha256": complete["installed"]["digests"]["installed_closure_sha256"],
            "native_files_sha256": complete["installed"]["digests"]["native_files_sha256"],
            "publisher_files_sha256": complete["installed"]["digests"]["publisher_files_sha256"],
            "preparation_sha256": sha256(preparation_path),
            "preparation_path": str(preparation_path),
            "constraints_sha256": EXPECTED_CONSTRAINTS_SHA256,
            "sdist_identity": fake_preparation["sdist"],
            "wheel_identity": fake_preparation["wheel"],
            "numpy_native_policy_sha256": _digest(complete["installed"]["numpy_native_policy"]),
            "numpy_record_sha256": _digest(fake_numpy_record),
            "numpy_runtime_config_sha256": _digest(complete["installed"]["numpy_runtime_config"]),
            "publisher_archive_manifest_sha256": complete["installed"]["publisher_archive"]["manifest_sha256"],
            "failures": [],
        }
    )
    complete["candidate_scope_sha256"] = _digest(complete["candidate_scope"])
    broken_path = archive_dir / ".audit-self-test.json"
    environment_broken_path = archive_dir / ".audit-environment-self-test.json"
    validation_bytecode_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    validation_bytecode_flag = sys.dont_write_bytecode
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    bad_preparation = json.loads(json.dumps(fake_preparation))
    bad_preparation["venv"] = {
        "path": str(archive_dir / "different-venv"),
        "interpreter": {
            "prefix": str(archive_dir / "different-venv"),
            "executable": str(archive_dir / "different-venv" / "bin" / "python"),
            "implementation": "CPython",
            "version": platform.python_version(),
        },
    }
    environment_broken = json.loads(json.dumps(complete))
    environment_broken["installed"]["preparation"] = bad_preparation
    try:
        globals()["_validate_preparation"] = lambda _path: bad_preparation
        environment_broken_path.write_text(json.dumps(environment_broken), encoding="utf-8")
        try:
            validate_report(environment_broken_path)
        except RuntimeError as error:
            assert "sys.prefix" in str(error) or "sys.executable" in str(error)
        else:
            raise AssertionError("validator must reject a different prepared venv")
    finally:
        globals()["_validate_preparation"] = lambda _path: fake_preparation
        environment_broken_path.unlink(missing_ok=True)
    try:
        broken_path.write_text(json.dumps(complete), encoding="utf-8")
        try:
            validate_report(broken_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("zero NumPy native files must fail report validation")
        complete["candidate_scope"]["installed_closure_sha256"] = None
        broken_path.write_text(json.dumps(complete), encoding="utf-8")
        try:
            validate_report(broken_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete candidate scope must fail closed")
    finally:
        globals()["execution_identity"] = original_execution_identity
        globals()["_validate_preparation"] = original_validate_preparation
        if validation_bytecode_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = validation_bytecode_env
        sys.dont_write_bytecode = validation_bytecode_flag
        broken_path.unlink(missing_ok=True)
        preparation_path.unlink(missing_ok=True)
        archive_manifest.unlink(missing_ok=True)
        shutil.rmtree(archive_dir, ignore_errors=True)
    try:
        original = set(FORBIDDEN_PACKAGES)
        globals()["FORBIDDEN_PACKAGES"] = frozenset((*original, "torch"))
        _lock_rows()
    except RuntimeError:
        pass
    else:
        raise AssertionError("forbidden dependency drift must fail closed")
    finally:
        globals()["FORBIDDEN_PACKAGES"] = frozenset({
            "cffi", "espeak", "espeak-ng", "librosa", "libsndfile", "onnx", "onnxruntime", "phonemizer", "soundfile", "soxr",
        })
    print("zonos dependency audit self-test: OK")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--publisher-archive", type=Path)
    parser.add_argument("--preparation", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if (
            args.preflight_only or args.installed or args.output is not None
            or args.validate_output is not None or args.expected_head is not None
            or args.publisher_archive is not None or args.preparation is not None
        ):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.preflight_only and (args.installed or args.output is not None or args.preparation is not None):
        parser.error("--preflight-only accepts no installed scan or output")
    if args.validate_output is not None and (
        args.preflight_only or args.installed or args.output is not None
        or args.expected_head is not None or args.publisher_archive is not None
        or args.preparation is not None
    ):
        parser.error("--validate-output accepts no audit collection arguments")
    if args.validate_output is not None:
        try:
            validate_report(args.validate_output)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"zonos dependency audit evidence BLOCKED: {error}", file=sys.stderr)
            return 2
        print("zonos dependency audit evidence: complete and hash-bound")
        return 0
    try:
        report = audit(
            args.output,
            installed=args.installed,
            expected_head=args.expected_head,
            publisher_archive=args.publisher_archive,
            preparation=args.preparation,
        )
    except (OSError, RuntimeError, ValueError) as error:
        if args.output is not None and not args.output.exists() and not args.output.is_symlink():
            failure = {
                "schema": "vokra-zonos-dependency-audit-v1",
                "status": AUDIT_STATUS,
                "publication": PUBLICATION,
                "execution_identity": None,
                "failures": [str(error)],
                    "candidate_scope": {
                        "schema": "vokra-zonos-dependency-approval-scope-v1",
                        "preparation_path": None,
                        "lock_rows_sha256": None,
                        "preparation_sha256": None,
                        "constraints_sha256": None,
                        "sdist_identity": None,
                        "wheel_identity": None,
                        "installed_closure_sha256": None,
                        "native_files_sha256": None,
                        "publisher_files_sha256": None,
                        "numpy_record_sha256": None,
                        "numpy_native_policy_sha256": None,
                        "numpy_runtime_config_sha256": None,
                    "publisher_archive_manifest_sha256": None,
                    "failures": [str(error)],
                    "model_access": False,
                    "source_access": False,
                    "checkpoint_access": False,
                    "publication": PUBLICATION,
                    "execution_identity": None,
                },
            }
            failure["candidate_scope_sha256"] = _digest(failure["candidate_scope"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"zonos dependency audit BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"zonos dependency audit: {AUDIT_STATUS}; publication={PUBLICATION}")
    return 2 if args.installed and report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
