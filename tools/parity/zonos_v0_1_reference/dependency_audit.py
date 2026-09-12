"""Model-free dependency, native-file, and publisher-license audit for Zonos.

The audit intentionally uses only Python's standard library.  It can therefore
run before a source checkout, checkpoint, or Zonos/Torch import exists.  A
green structural report is not an execution approval: the owner/legal gate
remains ``BLOCKED_UNREVIEWED_TRANSITIVE`` and every publication disposition is
``NO_UPLOAD``.
"""
from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_PATH = PROJECT_DIR / "pyproject.toml"
LOCK_PATH = PROJECT_DIR / "uv.lock"
REPOSITORY_ROOT = PROJECT_DIR.parents[2]
AUDITOR_PATH = Path(__file__).resolve()
WRAPPER_PATH = REPOSITORY_ROOT / "scripts/publish/vast-ai/audit-zonos-v0-1-dependencies.sh"
EXPECTED_PROJECT_SHA256 = "5cb58da85195f8f0812aa18bedd6a320226c7a3ef94e64c33e5782414c115b29"
EXPECTED_LOCK_SHA256 = "40fa51a7cffcfed126e073ecf0813fcbdb0935ea1bef05f51be1e75585fbcf76"
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
    project_sha = sha256(PROJECT_PATH)
    lock_sha = sha256(LOCK_PATH)
    if project_sha != EXPECTED_PROJECT_SHA256 or lock_sha != EXPECTED_LOCK_SHA256:
        raise RuntimeError("dedicated Zonos project identity drifted")
    return {
        "project": PROJECT_DIR.name,
        "python": "3.12",
        "pyproject_sha256": project_sha,
        "uv_lock_sha256": lock_sha,
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
    for path in (AUDITOR_PATH, WRAPPER_PATH):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"audit identity file is missing or symlinked: {path}")
    identity = {
        "expected_head": expected_head,
        "actual_head": actual_head,
        "dependency_audit_sha256": sha256(AUDITOR_PATH),
        "wrapper_sha256": sha256(WRAPPER_PATH),
        "pyproject_sha256": sha256(PROJECT_PATH),
        "uv_lock_sha256": sha256(LOCK_PATH),
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
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=f"zonos-{name}-", suffix=".tar.gz")
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1 << 20), b""):
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
                for member in sorted(package.getmembers(), key=lambda item: item.name):
                    member_path = PurePosixPath(member.name)
                    basename = member_path.name.lower()
                    if not basename.startswith(("license", "licence", "notice", "copying")):
                        continue
                    if member.issym() or member.islnk() or not member.isfile() or member_path.is_absolute() or ".." in member_path.parts:
                        failures.append(f"unsafe publisher member in locked sdist: {name}:{member.name}")
                        continue
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
                                payload_digest.update(block)
                                payload_size += len(block)
                                if target is not None:
                                    target.write(block)
                    finally:
                        if target is not None:
                            target.close()
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


def installed_audit(publisher_archive: Path | None = None) -> dict[str, Any]:
    """Collect exact installed distribution/native/license evidence on VAST."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("installed closure audit requires Linux x86_64")
    lock_rows = _lock_rows()
    active_rows = _active_lock_rows(lock_rows)
    expected_versions = {
        pep503_name(row["name"]): row["version"] for row in active_rows
    }
    failures: list[str] = []
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
    for name, distribution in sorted(distributions.items()):
        version = distribution.version
        if name in {"torch", "torchaudio"} and version == "2.6.0":
            version = "2.6.0+cpu"
        expected = expected_versions.get(name)
        if expected is not None and version != expected:
            failures.append(f"installed version mismatch: {name}={version!r}, expected {expected!r}")
        if name in FORBIDDEN_PACKAGES:
            failures.append(f"forbidden installed distribution: {name}")
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
        "digests": {
            "installed_closure_sha256": _digest(installed_rows),
            "native_files_sha256": _digest(native),
            "publisher_files_sha256": _digest(publisher_files),
        },
    }


def audit(
    output: Path | None = None,
    installed: bool = False,
    expected_head: str | None = None,
    publisher_archive: Path | None = None,
) -> dict[str, Any]:
    identity = project_identity()
    if installed and expected_head is None:
        raise RuntimeError("installed audit requires expected clean HEAD")
    execution = execution_identity(expected_head) if expected_head is not None else None
    report: dict[str, Any] = {
        "schema": "vokra-zonos-dependency-audit-v1",
        "status": AUDIT_STATUS,
        "publication": PUBLICATION,
        "project": identity,
        "lock": lock_audit(),
        "execution_identity": execution,
    }
    report["installed"] = (
        installed_audit(publisher_archive) if installed else {"status": "NOT_COLLECTED_PRE_ACQUISITION"}
    )
    installed_report = report["installed"]
    report["candidate_scope"] = {
        "schema": "vokra-zonos-dependency-approval-scope-v1",
        "lock_rows_sha256": report["lock"]["rows_sha256"],
        "installed_closure_sha256": installed_report.get("digests", {}).get("installed_closure_sha256"),
        "native_files_sha256": installed_report.get("digests", {}).get("native_files_sha256"),
        "publisher_files_sha256": installed_report.get("digests", {}).get("publisher_files_sha256"),
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

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    return value


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
    required_scope = {
        "schema", "lock_rows_sha256", "installed_closure_sha256", "native_files_sha256",
        "publisher_files_sha256", "publisher_archive_manifest_sha256", "failures",
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
    execution = report.get("execution_identity")
    if not isinstance(execution, dict) or scope["execution_identity"] != execution:
        raise RuntimeError("execution identity is missing or not scope-bound")
    expected_head = execution.get("expected_head")
    if not isinstance(expected_head, str):
        raise RuntimeError("execution identity HEAD is missing")
    if execution_identity(expected_head) != execution:
        raise RuntimeError("audit code, repository, or platform identity drifted")
    hex64 = re.compile(r"[0-9a-f]{64}")
    for key in (
        "lock_rows_sha256", "installed_closure_sha256", "native_files_sha256",
        "publisher_files_sha256", "publisher_archive_manifest_sha256",
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
    if not all(isinstance(value, list) for value in (installed_rows, native, publisher)) or not isinstance(digests, dict):
        raise RuntimeError("installed closure/native/publisher facts are incomplete")
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
    }
    for key, digest in expected_digests.items():
        if digests.get(key) != digest or scope[key] != digest:
            raise RuntimeError(f"{key} is not hash-bound")
    if scope["publisher_archive_manifest_sha256"] != archive["manifest_sha256"]:
        raise RuntimeError("publisher archive manifest is not scope-bound")
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

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

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
    finally:
        urllib.request.urlopen = original_urlopen
        synthetic_sdist.unlink(missing_ok=True)
        shutil.rmtree(archive_root, ignore_errors=True)
    complete = json.loads(json.dumps(report))
    archive_dir = Path(tempfile.mkdtemp(prefix="vokra-zonos-publisher-"))
    archive_manifest = archive_dir / "manifest.json"
    archive_manifest.write_text("[]\n", encoding="utf-8")
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
        "pyproject_sha256": sha256(PROJECT_PATH),
        "uv_lock_sha256": sha256(LOCK_PATH),
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "sys_platform": "linux",
            "python": "3.12.0",
        },
    }
    globals()["execution_identity"] = lambda _expected: fake_execution_identity
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
        "publisher_license_notice_files": [],
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
        },
    }
    complete["failures"] = []
    complete["candidate_scope"].update(
        {
            "installed_closure_sha256": complete["installed"]["digests"]["installed_closure_sha256"],
            "native_files_sha256": complete["installed"]["digests"]["native_files_sha256"],
            "publisher_files_sha256": complete["installed"]["digests"]["publisher_files_sha256"],
            "publisher_archive_manifest_sha256": complete["installed"]["publisher_archive"]["manifest_sha256"],
            "failures": [],
        }
    )
    complete["candidate_scope_sha256"] = _digest(complete["candidate_scope"])
    broken_path = PROJECT_DIR / ".audit-self-test.json"
    try:
        broken_path.write_text(json.dumps(complete), encoding="utf-8")
        validate_report(broken_path)
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
        broken_path.unlink(missing_ok=True)
        archive_manifest.unlink(missing_ok=True)
        archive_dir.rmdir()
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
    args = parser.parse_args()
    if args.self_test:
        if (
            args.preflight_only or args.installed or args.output is not None
            or args.validate_output is not None or args.expected_head is not None
            or args.publisher_archive is not None
        ):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.preflight_only and (args.installed or args.output is not None):
        parser.error("--preflight-only accepts no installed scan or output")
    if args.validate_output is not None and (
        args.preflight_only or args.installed or args.output is not None
        or args.expected_head is not None or args.publisher_archive is not None
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
                    "lock_rows_sha256": None,
                    "installed_closure_sha256": None,
                    "native_files_sha256": None,
                    "publisher_files_sha256": None,
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
