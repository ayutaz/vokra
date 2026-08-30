#!/usr/bin/env python3
"""Model-free factual audit for the pinned Bark reference environment.

This module is deliberately independent of Bark, Transformers and Vokra
runtime imports.  It is run only after an authorised VAST frozen sync and
inspects ``importlib.metadata`` records, not model code or model weights.
The only network operation is an allow-listed request for the two upstream
``LICENSE`` files.  Their bytes are measured and retained in the report as a
primary-source hash; no license class is inferred from a package expectation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # pragma: no cover - package invocation fallback
    from tools.parity.bark import license_gate


SCHEMA = "vokra-bark-dependency-audit-v1"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
ELF_MAGIC = b"\x7fELF"
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
LICENSE_FILE_NAMES = {"license", "copying", "notice", "copyright"}
LICENSE_CDN_HOSTS = {
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
}

# These are the fixed upstream identities already present in the Bark worker
# and gate manifest.  No source-code revision is invented here: the source
# contract is inspected separately and remains blocked when absent.
MODEL_LICENSES = (
    {
        "id": "bark-small-weight-mit",
        "repo": "suno/bark-small",
        "revision": "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
    },
    {
        "id": "bark-full-weight-mit",
        "repo": "suno/bark",
        "revision": "70a8a7d34168586dc5d028fa9666aceade177992",
    },
)
MODEL_REPO_FIELDS = {
    "small_upstream_repo": "suno/bark-small",
    "small_upstream_revision": MODEL_LICENSES[0]["revision"],
    "full_upstream_repo": "suno/bark",
    "full_upstream_revision": MODEL_LICENSES[1]["revision"],
}
REQUIRED_IDENTITY_FIELDS = (
    "small_public_repo",
    "small_public_revision",
    "small_public_bytes",
    "small_public_sha256",
    "small_public_file",
    "small_upstream_repo",
    "small_upstream_revision",
    "small_checkpoint_bytes",
    "small_checkpoint_sha256",
    "small_config_bytes",
    "small_config_sha256",
    "full_public_repo",
    "full_public_revision",
    "full_public_bytes",
    "full_public_sha256",
    "full_public_file",
    "full_upstream_repo",
    "full_upstream_revision",
    "full_checkpoint_bytes",
    "full_checkpoint_sha256",
    "full_config_bytes",
    "full_config_sha256",
    "generation_config_bytes",
    "generation_config_sha256",
    "transformers_version",
    "transformers_source_revision",
    "transformers_sdist_sha256",
    "transformers_wheel_sha256",
)


class AuditError(ValueError):
    """A fail-closed factual audit input or environment error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalized_name(name: str) -> str:
    """Return the PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name.strip()).casefold()


def normalized_version(version: str) -> str:
    # The lock stores the exact resolved PEP 440 spelling.  Case-folding is
    # safe for the local-version labels and catches presentation-only drift.
    return re.sub(r"\s+", "", version.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={normalized_version(version)}"


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _contract(project: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    project_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    manifest_path = project / "license_gate_manifest.json"
    if not all(_regular(path) for path in (project_path, lock_path, manifest_path)):
        raise AuditError("Bark pyproject.toml, uv.lock, or gate manifest is missing/symlinked")
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project_data = tomllib.loads(project_bytes.decode("utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"Bark closure bytes are unreadable: {exc}") from exc
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != 1:
        raise AuditError("Bark gate manifest version is unsupported")
    if sha256_bytes(project_bytes) != manifest.get("project_sha256"):
        raise AuditError("pyproject.toml bytes differ from the reviewed Bark contract")
    if sha256_bytes(lock_bytes) != manifest.get("lock_sha256"):
        raise AuditError("uv.lock bytes differ from the reviewed Bark contract")
    try:
        locked_rows = license_gate.rows(lock_data)
        license_gate.validate_project_schema(project_data)
    except SystemExit as exc:
        raise AuditError("Bark pyproject/uv.lock structural schema is invalid") from exc
    identities = manifest.get("identities")
    if not isinstance(identities, dict):
        raise AuditError("Bark fixed identity table is missing")
    for field in REQUIRED_IDENTITY_FIELDS:
        if field not in identities:
            raise AuditError(f"Bark fixed identity field is missing: {field}")
    for field, expected in MODEL_REPO_FIELDS.items():
        if identities.get(field) != expected:
            raise AuditError(f"Bark model identity drifted: {field}")
    return project_data, lock_data, manifest, locked_rows, project_bytes, lock_bytes


def _expected_packages(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    locked_rows = sorted(
        lock.get("package", []),
        key=lambda row: (normalized_name(row["name"]), normalized_version(row["version"])),
    )
    virtual_rows = [row for row in locked_rows if row.get("source") == {"virtual": "."}]
    packages = [row for row in locked_rows if row.get("source") != {"virtual": "."}]
    if not packages:
        raise AuditError("Bark lock has no registry packages")
    if len(virtual_rows) != 1:
        raise AuditError("Bark lock must contain exactly one virtual project row")
    if any(row.get("source", {}).get("registry") not in {
        "https://pypi.org/simple",
        "https://download.pytorch.org/whl/cpu",
    } for row in packages):
        raise AuditError("Bark lock contains an unapproved package index")
    return locked_rows, packages, virtual_rows[0]


def _distribution_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        version = dist.version
        if not name or not version:
            continue
        location = Path(dist.locate_file(""))
        records.append(
            {
                "distribution": dist,
                "identity": identity(name, version),
                "name": name,
                "version": version,
                "location": str(location),
            }
        )
    return sorted(records, key=lambda item: (item["identity"], item["location"]))


def _git_identity(project: Path) -> dict[str, Any]:
    """Bind evidence to the checkout commit and this audit script's bytes."""
    root = project.resolve()
    for _ in range(3):
        root = root.parent
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"cannot bind audit to a git commit: {exc}") from exc
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AuditError("git HEAD is not a full commit identity")
    script = Path(__file__).resolve()
    return {"commit": commit, "audit_script_sha256": sha256_file(script)}


def compare_multiset(expected: list[str], actual: list[str]) -> dict[str, Any]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    duplicates = sorted(
        item for item, count in actual_counts.items() if count > 1
    )
    return {
        "expected": sorted(expected),
        "installed": sorted(actual),
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_identities": duplicates,
        "exact": not missing and not unexpected,
    }


def _metadata_fields(dist: metadata.Distribution) -> dict[str, Any]:
    classifiers = sorted(
        value.removeprefix("License :: ")
        for value in (dist.metadata.get_all("Classifier") or [])
        if value.startswith("License :: ")
    )
    expression = dist.metadata.get("License-Expression")
    declared = dist.metadata.get("License")
    return {
        "license": declared.strip() if isinstance(declared, str) and declared.strip() else None,
        "license_expression": expression.strip() if isinstance(expression, str) and expression.strip() else None,
        "license_classifiers": classifiers,
    }


def _entry_path(dist: metadata.Distribution, entry: Any) -> Path | None:
    path = Path(dist.locate_file(entry))
    try:
        path.relative_to(Path(dist.locate_file("")))
    except ValueError:
        return None
    return path


def _is_license_path(relative: str) -> bool:
    basename = Path(relative).name.casefold()
    stem = Path(basename).stem
    return stem in LICENSE_FILE_NAMES or any(
        token in basename for token in ("license", "copying", "notice", "copyright")
    )


def _publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not _is_license_path(relative):
            continue
        path = _entry_path(dist, entry)
        if path is None or path.is_symlink() or not path.is_file():
            unsafe.append(relative)
            continue
        result.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return result, unsafe


def _first_four(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4)


def _elf_needed(path: Path, magic: bytes | None = None) -> dict[str, Any]:
    if (magic if magic is not None else _first_four(path)) != ELF_MAGIC:
        return {"format": "non-elf", "needed": [], "inspection": "not-applicable"}
    try:
        completed = subprocess.run(
            ["readelf", "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"format": "elf", "needed": [], "inspection": "error", "error": str(exc)}
    output = f"{completed.stdout}\n{completed.stderr}"
    needed = sorted(
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line))
    )
    if completed.returncode == 0:
        status = "ok"
    elif "no dynamic section" in output.casefold():
        status = "no-dynamic-section"
    else:
        status = "error"
    result: dict[str, Any] = {
        "format": "elf",
        "needed": needed,
        "inspection": status,
        "readelf_returncode": completed.returncode,
    }
    if status == "error":
        result["error"] = output.strip()[-2000:]
    return result


def _native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        path = _entry_path(dist, entry)
        suffix_candidate = (
            Path(relative).suffix.casefold() in NATIVE_SUFFIXES
            or ".so." in Path(relative).name.casefold()
            or Path(relative).name.casefold().endswith(".dll")
        )
        if path is None:
            if suffix_candidate:
                errors.append({"path": relative, "stage": "locate", "error": "path escapes distribution root"})
            continue
        if path.is_symlink():
            if suffix_candidate:
                errors.append({"path": relative, "stage": "symlink", "error": "native candidate is symlinked"})
            continue
        if not path.is_file():
            if suffix_candidate:
                errors.append({"path": relative, "stage": "regular-file", "error": "native candidate is not a regular file"})
            continue
        try:
            magic = _first_four(path)
        except OSError as exc:
            errors.append({"path": relative, "stage": "magic", "error": str(exc)})
            continue
        if not suffix_candidate and magic != ELF_MAGIC:
            continue
        try:
            size = path.stat().st_size
            file_hash = sha256_file(path)
            inspection = _elf_needed(path, magic)
        except OSError as exc:
            errors.append({"path": relative, "stage": "hash-or-inspect", "error": str(exc)})
            continue
        result.append({
            "path": relative,
            "size": size,
            "sha256": file_hash,
            "bundled": True,
            "origin": "installed-distribution",
            "candidate": "elf-magic" if magic == ELF_MAGIC and not suffix_candidate else "native-suffix",
            "needed": inspection,
        })
    return result, errors


def _inspect_package(
    row: dict[str, Any], record: dict[str, Any] | None, review: dict[str, Any] | None
) -> tuple[dict[str, Any], list[str]]:
    if record is None:
        return {
            "lock": {"name": row["name"], "version": row["version"], "source": row["source"]},
            "installed": None,
            "review": review,
        }, [f"installed closure missing: {identity(row['name'], row['version'])}"]
    dist = record["distribution"]
    publisher, unsafe_publisher = _publisher_files(dist)
    native, native_errors = _native_files(dist)
    installed = {
        "name": dist.metadata.get("Name"),
        "version": dist.version,
        "normalized_identity": record["identity"],
        **_metadata_fields(dist),
        "publisher_files": publisher,
        "native_files": native,
        "native_errors": native_errors,
        "bundled_libraries": [
            {
                "distribution": dist.metadata.get("Name"),
                "path": item["path"],
                "sha256": item["sha256"],
                "needed": item["needed"],
            }
            for item in native
        ],
    }
    failures: list[str] = []
    if not installed["license"] and not installed["license_expression"] and not installed["license_classifiers"]:
        failures.append(f"missing package license metadata: {record['identity']}")
    if not publisher:
        failures.append(f"missing publisher LICENSE/NOTICE evidence: {record['identity']}")
    failures.extend(f"unsafe publisher path: {record['identity']}:{path}" for path in unsafe_publisher)
    failures.extend(
        f"native candidate inspection failed: {record['identity']}:{item['path']}:{item['stage']}"
        for item in native_errors
    )
    if any(item["needed"]["inspection"] == "error" for item in native):
        failures.append(f"ELF NEEDED inspection failed: {record['identity']}")
    return {
        "lock": {
            "name": row["name"],
            "version": row["version"],
            "source": row["source"],
            "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
        },
        "installed": installed,
        "review": review,
    }, failures


def _license_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/raw/{revision}/LICENSE"


def _validate_license_url(url: str, item: dict[str, str]) -> None:
    if not any(item == allowed for allowed in MODEL_LICENSES):
        raise AuditError("LICENSE item is not one of the fixed Bark model identities")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError(f"invalid LICENSE URL port: {url}") from exc
    raw_url = _license_url(item["repo"], item["revision"])
    cdn_path = f"/{item['repo']}/resolve/{item['revision']}/LICENSE"
    raw_match = parsed.hostname == "huggingface.co" and url == raw_url
    cdn_match = parsed.hostname in LICENSE_CDN_HOSTS and parsed.path == cdn_path
    if (
        parsed.scheme != "https"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not (raw_match or cdn_match)
    ):
        raise AuditError(f"non-allow-listed LICENSE host or scheme: {url}")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, trace: list[str], item: dict[str, str]) -> None:
        super().__init__()
        self.trace = trace
        self.item = item

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        _validate_license_url(newurl, self.item)
        self.trace.append(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _fetch_license(
    item: dict[str, str],
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
    claimed_license: str | None = None,
) -> dict[str, Any]:
    requested_url = _license_url(item["repo"], item["revision"])
    _validate_license_url(requested_url, item)
    trace = [requested_url]
    if fetcher is None:
        opener = build_opener(_LicenseRedirects(trace, item))
        request = Request(requested_url, headers={"Accept": "text/plain", "User-Agent": "vokra-bark-audit/1"})
        with opener.open(request, timeout=30) as response:  # noqa: S310 - host/path validated above
            final_url = response.geturl()
            _validate_license_url(final_url, item)
            if trace[-1] != final_url:
                trace.append(final_url)
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_LICENSE_BYTES:
                raise AuditError("upstream LICENSE Content-Length exceeds the bounded audit size")
            body = response.read(MAX_LICENSE_BYTES + 1)
    else:
        final_url, body = fetcher(requested_url)
        _validate_license_url(final_url, item)
        if trace[-1] != final_url:
            trace.append(final_url)
    if len(body) > MAX_LICENSE_BYTES:
        raise AuditError("upstream LICENSE response exceeds the bounded audit size")
    body_hash = sha256_bytes(body)
    return {
        "id": item["id"],
        "repo": item["repo"],
        "revision": item["revision"],
        "requested_url": requested_url,
        "final_url": final_url,
        "url_trace": trace,
        "size": len(body),
        "sha256": body_hash,
        "primary_source_bytes": len(body),
        "primary_source_sha256": body_hash,
        "claimed_license": claimed_license,
        "claimed_license_source": "existing manifest only; unresolved claims are recorded as null",
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
        "acquired_file": "LICENSE",
    }


def audit_model_licenses(
    manifest: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None
) -> list[dict[str, Any]]:
    claims = {
        row.get("id"): row.get("license")
        for row in manifest.get("license_rows", [])
        if isinstance(row, dict) and row.get("id") in {item["id"] for item in MODEL_LICENSES}
    }
    return [_fetch_license(item, fetcher, claims.get(item["id"]) if claims.get(item["id"]) not in {None, "UNRESOLVED"} else None) for item in MODEL_LICENSES]


def audit_environment(project: Path, fetch_model_licenses: bool = True) -> dict[str, Any]:
    project_data, lock, manifest, locked_rows_from_contract, project_bytes, lock_bytes = _contract(project)
    locked_rows, expected_rows, virtual_row = _expected_packages(lock)
    if locked_rows != locked_rows_from_contract:
        raise AuditError("Bark lock row ordering or content is not canonical")
    records = _distribution_records()
    expected_ids = [identity(row["name"], row["version"]) for row in expected_rows]
    actual_ids = [record["identity"] for record in records]
    closure = compare_multiset(expected_ids, actual_ids)
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_identity.setdefault(record["identity"], []).append(record)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    review_map = {
        (row.get("name"), row.get("version")): row
        for row in manifest.get("package_review_rows", [])
        if isinstance(row, dict)
    }
    for row in expected_rows:
        key = identity(row["name"], row["version"])
        candidates = by_identity.get(key, [])
        review = review_map.get((row["name"], row["version"]))
        package, package_failures = _inspect_package(
            row, candidates[0] if len(candidates) == 1 else None, review
        )
        if review is None:
            package_failures.append(f"package review evidence missing: {key}")
        if len(candidates) > 1:
            package_failures.append(f"duplicate installed distribution: {key}")
        packages.append(package)
        failures.extend(package_failures)
    virtual_review = next(
        (
            row for row in manifest.get("package_review_rows", [])
            if isinstance(row, dict)
            and (row.get("name"), row.get("version")) == (virtual_row["name"], virtual_row["version"])
        ),
        None,
    )
    packages.append({
        "lock": virtual_row,
        "installed": None,
        "review": virtual_review,
        "activity": {
            "status": "INACTIVE_VIRTUAL_PROJECT",
            "reason": "uv.lock virtual project row (package=false) is not an installed distribution",
            "evidence": "source={virtual='.'}; installed distribution multiset intentionally excludes this row",
        },
    })
    if virtual_review is None:
        failures.append("virtual project row has no matching package review evidence")
    if not closure["exact"]:
        failures.append("installed distribution multiset does not exactly match uv.lock")
    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        failures.append(f"audit host is not Linux x86_64: {sys.platform}/{platform.machine()}")

    identities = manifest["identities"]
    source_fields = sorted(key for key in identities if "bark_source" in key or key in {"source_repo", "source_revision"})
    source_contract = {
        "status": "BLOCKED_MISSING_PINNED_SOURCE_REVISION",
        "required_fields": ["bark_source_repo", "bark_source_revision"],
        "present_related_fields": source_fields,
        "license_file": "LICENSE",
    }
    failures.append("Bark source-code LICENSE revision is absent from the existing identity contract")
    model_license_files = audit_model_licenses(manifest) if fetch_model_licenses else []
    git = _git_identity(project)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED" if failures else "PASS",
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "readelf_required": True,
            "model_code_imported": False,
            "cargo_invoked": False,
        },
        "project": {
            "name": project_data["project"]["name"],
            "version": project_data["project"]["version"],
            "pyproject_bytes": len(project_bytes),
            "pyproject_sha256": sha256_bytes(project_bytes),
            "uv_lock_bytes": len(lock_bytes),
            "uv_lock_sha256": sha256_bytes(lock_bytes),
        },
        "git": git,
        "fixed_bark_identities": {key: identities[key] for key in REQUIRED_IDENTITY_FIELDS},
        "locked_rows": locked_rows,
        "virtual_project_row": {
            "row": virtual_row,
            "status": "INACTIVE_VIRTUAL_PROJECT",
            "reason": "uv.lock virtual project row (package=false) is not an installed distribution",
        },
        "active_installed_rows": [
            {
                "name": record["name"],
                "version": record["version"],
                "normalized_identity": record["identity"],
                "location": record["location"],
            }
            for record in records
        ],
        "closure": closure,
        "packages": packages,
        "source_license_contract": source_contract,
        "model_license_files": model_license_files,
        "model_acquisition": {
            "policy": "allow-listed LICENSE-only in-memory fetch",
            "requested_files": [item["requested_url"] for item in model_license_files],
            "non_license_files": [],
            "non_license_requests": [],
            "proof": "no model-weight request path exists in this audit; only exact LICENSE URLs are fetched",
        },
        "failures": sorted(set(failures)),
    }
    return report


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try:
        report = audit_environment(project, fetch_model_licenses)
    except (AuditError, OSError, UnicodeError, ValueError) as exc:
        report = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "environment": {"model_code_imported": False, "cargo_invoked": False},
            "model_acquisition": {"requested_files": [], "non_license_files": [], "non_license_requests": []},
            "failures": [str(exc)],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    if report["failures"]:
        print("bark dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"bark dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    expected = [identity("foo-bar", "1.0"), identity("torch", "2.13.0+cpu")]
    exact = compare_multiset(expected, [identity("torch", "2.13.0+cpu"), identity("foo_bar", "1.0")])
    assert exact["exact"] and not exact["missing"] and not exact["unexpected"]
    mismatch = compare_multiset(expected, [identity("foo-bar", "1.0"), identity("torch", "2.12.0+cpu")])
    assert mismatch["missing"] == [identity("torch", "2.13.0+cpu")]
    assert mismatch["unexpected"] == [identity("torch", "2.12.0+cpu")]
    duplicate = compare_multiset([identity("foo-bar", "1.0")], [identity("foo_bar", "1.0"), identity("foo-bar", "1.0")])
    assert duplicate["duplicate_identities"] == [identity("foo-bar", "1.0")]
    assert duplicate["unexpected"] == [identity("foo-bar", "1.0")]

    good = _license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"])
    cdn_good = (
        "https://cdn-lfs.huggingface.co/"
        f"{MODEL_LICENSES[0]['repo']}/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE"
    )
    body = b"primary source bytes\n"
    result = _fetch_license(MODEL_LICENSES[0], lambda url: (url, body))
    assert result["requested_url"] == good
    assert result["final_url"] == good
    assert result["size"] == len(body) and result["sha256"] == sha256_bytes(body)
    assert result["license_classification"] == "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"
    cdn_result = _fetch_license(MODEL_LICENSES[0], lambda _url: (cdn_good, body))
    assert cdn_result["final_url"] == cdn_good and cdn_result["url_trace"] == [good, cdn_good]
    for bad in (
        good.replace("/LICENSE", "/model.safetensors"),
        "https://example.invalid/LICENSE",
        "http://huggingface.co/suno/bark/LICENSE",
        "https://huggingface.co:444/suno/bark/LICENSE",
        "https://user:hunter2@huggingface.co/suno/bark/LICENSE",
        f"{good}?download=true",
        f"{good}#fragment",
        f"https://huggingface.co/suno/bark/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{'0' * 40}/LICENSE",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE.txt",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/model.safetensors",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE?download=true",
        f"https://cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE#fragment",
        f"https://cdn-lfs.huggingface.co:444/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
        f"https://user:hunter2@cdn-lfs.huggingface.co/suno/bark-small/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE",
    ):
        try:
            _fetch_license(MODEL_LICENSES[0], lambda _url, bad=bad: (bad, body))
        except AuditError:
            pass
        else:
            raise AssertionError(f"accepted unsafe LICENSE redirect: {bad}")
    try:
        _fetch_license(MODEL_LICENSES[0], lambda url: (url, b"x" * (MAX_LICENSE_BYTES + 1)))
    except AuditError:
        pass
    else:
        raise AssertionError("accepted an over-sized LICENSE response")

    with tempfile.TemporaryDirectory(prefix="bark-audit-self-test-") as directory:
        path = Path(directory) / "not-native"
        path.write_bytes(b"not an ELF file")
        assert _elf_needed(path)["format"] == "non-elf"
        first = canonical_json({"b": 2, "a": 1})
        second = canonical_json(json.loads(first))
        assert first == second == '{"a":1,"b":2}'
        output = Path(directory) / "report.json"
        output.write_text(first + "\n", encoding="utf-8")
        assert output.read_text(encoding="utf-8") == first + "\n"
    print("bark dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-model-licenses", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None or args.fetch_model_licenses:
            parser.error("--self-test accepts no project/output/fetch arguments")
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__":
    raise SystemExit(main())
