"""Model-free dependency, native-file, and publisher-license audit for Zonos.

The audit intentionally uses only Python's standard library.  It can therefore
run before a source checkout, checkpoint, or Zonos/Torch import exists.  A
green structural report is not an execution approval: the owner/legal gate
remains ``BLOCKED_UNREVIEWED_TRANSITIVE`` and every publication disposition is
``NO_UPLOAD``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_PATH = PROJECT_DIR / "pyproject.toml"
LOCK_PATH = PROJECT_DIR / "uv.lock"
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
    return size, digest.hexdigest(), bytes(header) == b"\\x7fELF"


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
        if name.lower() in FORBIDDEN_PACKAGES:
            raise RuntimeError(f"forbidden Zonos dependency in lock: {name}")
        if name not in LICENSE_CONCLUSIONS:
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
                "license_conclusion": LICENSE_CONCLUSIONS[name],
            }
        )
    names = {row["name"] for row in rows}
    if not set(DIRECT_DEPENDENCIES).issubset(names):
        raise RuntimeError("a direct Zonos dependency is absent from uv.lock")
    torch_rows = [row for row in rows if row["name"] == "torch"]
    audio_rows = [row for row in rows if row["name"] == "torchaudio"]
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
        name = dependency["name"]
        candidates = [row for row in rows if row["name"] == name and row["source"].get("virtual") is None]
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
        candidates = [row for row in active if row["name"] == name]
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


def installed_audit() -> dict[str, Any]:
    """Collect exact installed distribution/native/license evidence on VAST."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("installed closure audit requires Linux x86_64")
    lock_rows = _lock_rows()
    active_rows = _active_lock_rows(lock_rows)
    expected_versions = {
        row["name"].lower().replace("_", "-"): row["version"] for row in active_rows
    }
    failures: list[str] = []
    distributions: dict[str, importlib.metadata.Distribution] = {}
    duplicate_distributions: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str):
            failures.append("installed distribution has no canonical Name metadata")
            continue
        normalized = name.lower().replace("_", "-")
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
            byte_count, file_sha256, is_elf = _streaming_binary_facts(path)
            needed: list[str] = []
            if is_elf:
                try:
                    needed = _elf_needed(path)
                except RuntimeError as error:
                    failures.append(str(error))
            forbidden_needed = [
                item
                for item in needed
                if any(token in item.lower() for token in ("gpl", "lgpl", "soxr", "espeak", "sndfile"))
            ]
            if forbidden_needed:
                failures.append(
                    f"forbidden native NEEDED for {path}: {','.join(sorted(set(forbidden_needed)))}"
                )
            native.append(
                {
                    "root": str(root),
                    "path": str(path.relative_to(root)),
                    "kind": (
                        ".so.*" if re.search(r"\.so\.", path.name, re.IGNORECASE)
                        else path.suffix.lower()
                    ),
                    "bytes": byte_count,
                    "sha256": file_sha256,
                    "format": "ELF" if is_elf else "non-ELF",
                    "needed": needed,
                }
            )
    publisher_files: list[dict[str, Any]] = []
    for name, distribution in sorted(distributions.items()):
        matched = 0
        for item in distribution.files or ():
            base = Path(item).name.lower()
            path = Path(distribution.locate_file(item))
            if not base.startswith(("license", "licence", "notice")):
                continue
            if path.is_symlink() or not path.is_file():
                failures.append(f"publisher file is unsafe: {path}")
                continue
            matched += 1
            publisher_files.append(
                {
                    "distribution": name,
                    "path": str(item),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        if matched == 0:
            publisher_files.append({"distribution": name, "status": "MISSING"})
            failures.append(f"missing publisher LICENSE/NOTICE evidence: {name}")
    publisher_files.sort(key=lambda row: (row["distribution"], row.get("path", "")))
    return {
        "status": "COLLECTED",
        "platform": {"system": "Linux", "machine": "x86_64"},
        "active_lock_packages": active_rows,
        "installed_distributions": installed_rows,
        "failures": sorted(set(failures)),
        "native_files": native,
        "publisher_license_notice_files": publisher_files,
        "digests": {
            "installed_closure_sha256": _digest(installed_rows),
            "native_files_sha256": _digest(native),
            "publisher_files_sha256": _digest(publisher_files),
        },
    }


def audit(output: Path | None = None, installed: bool = False) -> dict[str, Any]:
    identity = project_identity()
    report: dict[str, Any] = {"schema": "vokra-zonos-dependency-audit-v1", "status": AUDIT_STATUS, "publication": PUBLICATION, "project": identity, "lock": lock_audit()}
    report["installed"] = installed_audit() if installed else {"status": "NOT_COLLECTED_PRE_ACQUISITION"}
    installed_report = report["installed"]
    report["candidate_scope"] = {
        "schema": "vokra-zonos-dependency-approval-scope-v1",
        "lock_rows_sha256": report["lock"]["rows_sha256"],
        "installed_closure_sha256": installed_report.get("digests", {}).get("installed_closure_sha256"),
        "native_files_sha256": installed_report.get("digests", {}).get("native_files_sha256"),
        "publisher_files_sha256": installed_report.get("digests", {}).get("publisher_files_sha256"),
        "failures": installed_report.get("failures", []),
        "model_access": False,
        "source_access": False,
        "checkpoint_access": False,
        "publication": PUBLICATION,
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


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("audit evidence must be a JSON object")
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
        "publisher_files_sha256", "failures", "model_access", "source_access",
        "checkpoint_access", "publication",
    }
    if not required_scope.issubset(scope):
        raise RuntimeError("candidate scope is incomplete")
    if scope["schema"] != "vokra-zonos-dependency-approval-scope-v1":
        raise RuntimeError("candidate scope schema is not exact")
    if any(scope[key] is not False for key in ("model_access", "source_access", "checkpoint_access")):
        raise RuntimeError("audit evidence crosses the acquisition boundary")
    if scope["publication"] != PUBLICATION:
        raise RuntimeError("candidate scope publication is not NO_UPLOAD")
    hex64 = re.compile(r"[0-9a-f]{64}")
    for key in ("lock_rows_sha256", "installed_closure_sha256", "native_files_sha256", "publisher_files_sha256"):
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
    if not all(isinstance(value, list) for value in (installed_rows, native, publisher)) or not isinstance(digests, dict):
        raise RuntimeError("installed closure/native/publisher facts are incomplete")
    expected_digests = {
        "installed_closure_sha256": _digest(installed_rows),
        "native_files_sha256": _digest(native),
        "publisher_files_sha256": _digest(publisher),
    }
    for key, digest in expected_digests.items():
        if digests.get(key) != digest or scope[key] != digest:
            raise RuntimeError(f"{key} is not hash-bound")
    for row in native:
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not hex64.fullmatch(row["sha256"]):
            raise RuntimeError("native file hash evidence is malformed")
    for row in publisher:
        if not isinstance(row, dict) or row.get("status") == "MISSING":
            if not isinstance(row, dict) or row.get("status") != "MISSING":
                raise RuntimeError("publisher evidence row is malformed")
            continue
        if not isinstance(row.get("sha256"), str) or not hex64.fullmatch(row["sha256"]):
            raise RuntimeError("publisher LICENSE/NOTICE hash evidence is malformed")


def self_test() -> None:
    assert project_identity()["python"] == "3.12"
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
    complete = json.loads(json.dumps(report))
    complete["installed"] = {
        "status": "COLLECTED",
        "active_lock_packages": active,
        "installed_distributions": [],
        "failures": [],
        "native_files": [],
        "publisher_license_notice_files": [],
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
        broken_path.unlink(missing_ok=True)
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
    args = parser.parse_args()
    if args.self_test:
        if args.preflight_only or args.installed or args.output is not None or args.validate_output is not None:
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.preflight_only and (args.installed or args.output is not None):
        parser.error("--preflight-only accepts no installed scan or output")
    if args.validate_output is not None and (args.preflight_only or args.installed or args.output is not None):
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
        report = audit(args.output, installed=args.installed)
    except (OSError, RuntimeError, ValueError) as error:
        if args.output is not None and not args.output.exists() and not args.output.is_symlink():
            failure = {
                "schema": "vokra-zonos-dependency-audit-v1",
                "status": AUDIT_STATUS,
                "publication": PUBLICATION,
                "failures": [str(error)],
                "candidate_scope": {
                    "schema": "vokra-zonos-dependency-approval-scope-v1",
                    "lock_rows_sha256": None,
                    "installed_closure_sha256": None,
                    "native_files_sha256": None,
                    "publisher_files_sha256": None,
                    "failures": [str(error)],
                    "model_access": False,
                    "source_access": False,
                    "checkpoint_access": False,
                    "publication": PUBLICATION,
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
