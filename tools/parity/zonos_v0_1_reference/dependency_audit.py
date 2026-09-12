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
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


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
    lock_versions: dict[str, set[str]] = {}
    for row in lock_rows:
        lock_versions.setdefault(row["name"], set()).add(row["version"])
    actual_versions: dict[str, str] = {}
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for package, expected in DIRECT_DEPENDENCIES.items():
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required distribution is not installed: {package}") from error
        actual = distribution.version
        if package in {"torch", "torchaudio"} and actual == "2.6.0":
            actual = "2.6.0+cpu"
        if actual != expected:
            raise RuntimeError(f"installed version mismatch: {package}={actual!r}")
        actual_versions[package] = actual
        distributions[package] = distribution
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str):
            continue
        normalized = name.lower().replace("_", "-")
        if normalized in FORBIDDEN_PACKAGES:
            raise RuntimeError(f"forbidden installed distribution: {name}")
        if normalized not in lock_versions:
            raise RuntimeError(f"installed distribution is outside the locked closure: {name}")
        if distribution.version not in lock_versions[normalized]:
            raise RuntimeError(f"installed transitive version is not locked: {name}={distribution.version!r}")
        distributions.setdefault(normalized, distribution)
    roots = {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib") if sysconfig.get_paths().get(key)}
    native: list[dict[str, Any]] = []
    for root in sorted(roots):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"site-packages root is unsafe: {root}")
        for path in sorted(root.rglob("*.so")):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"native file is unsafe: {path}")
            native.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path), "needed": _elf_needed(path)})
    publisher_files: list[dict[str, Any]] = []
    for name, distribution in sorted(distributions.items()):
        for item in distribution.files or ():
            base = Path(item).name.lower()
            path = Path(distribution.locate_file(item))
            if base.startswith(("license", "licence", "notice")) and path.is_file():
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"publisher file is unsafe: {path}")
                publisher_files.append({"distribution": name, "path": str(item), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"platform": {"system": "Linux", "machine": "x86_64"}, "versions": actual_versions, "native_elf": native, "publisher_license_notice_files": publisher_files}


def audit(output: Path | None = None, installed: bool = False) -> dict[str, Any]:
    identity = project_identity()
    report: dict[str, Any] = {"schema": "vokra-zonos-dependency-audit-v1", "status": AUDIT_STATUS, "publication": PUBLICATION, "project": identity, "lock": lock_audit()}
    report["installed"] = installed_audit() if installed else {"status": "NOT_COLLECTED_PRE_ACQUISITION"}
    if output is not None:
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"dependency audit output already exists or is symlinked: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def self_test() -> None:
    assert project_identity()["python"] == "3.12"
    report = audit()
    assert report["status"] == AUDIT_STATUS
    assert report["publication"] == PUBLICATION
    assert report["lock"]["package_count"] == 29
    assert not set(row["name"].lower() for row in report["lock"]["rows"]) & FORBIDDEN_PACKAGES
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
    args = parser.parse_args()
    if args.self_test:
        if args.preflight_only or args.installed or args.output is not None:
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.preflight_only and (args.installed or args.output is not None):
        parser.error("--preflight-only accepts no installed scan or output")
    try:
        audit(args.output, installed=args.installed)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"zonos dependency audit BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"zonos dependency audit: {AUDIT_STATUS}; publication={PUBLICATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
