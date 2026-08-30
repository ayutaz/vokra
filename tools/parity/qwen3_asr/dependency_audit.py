#!/usr/bin/env python3
"""Model-free audit of the pinned Qwen3-ASR reference environment.

The audit is intentionally separate from :mod:`preflight_gate`: the gate
authenticates the inputs before synchronization, while this script inspects
the already synchronized, exact environment.  It never imports qwen-asr,
torch, or any model code.  The only upstream model requests it can make are
the two allow-listed LICENSE URLs below.
"""

from __future__ import annotations

import argparse
import base64
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
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tomllib

try:
    import preflight_gate
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses the first branch
    from tools.parity.qwen3_asr import preflight_gate


SCHEMA = "vokra-qwen3-asr-dependency-audit-v1"
MODEL_LICENSES = (
    {
        "repo": "Qwen/Qwen3-ASR-0.6B",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    },
    {
        "repo": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    },
)
LICENSE_FILE_NAMES = {"license", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
ELF_MAGIC = b"\x7fELF"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
HF_LICENSE_REDIRECT_HOSTS = {
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
}


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


def model_license_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/raw/{revision}/LICENSE"


def _validate_license_url(url: str, repo: str, revision: str, *, initial: bool) -> None:
    """Accept only the fixed raw URL or its bounded CDN redirect shape."""
    expected = model_license_url(repo, revision)
    if initial and url != expected:
        raise ValueError(f"unexpected initial model LICENSE URL: {url}")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid model LICENSE URL: {url}") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unsafe model LICENSE URL: {url}")
    raw_path = urlparse(expected).path
    redirect_path = f"/{repo}/resolve/{revision}/LICENSE"
    if not initial and not (
        url == expected
        or (parsed.hostname in HF_LICENSE_REDIRECT_HOSTS and parsed.path == redirect_path)
    ):
        raise ValueError(f"non-license model URL was returned: {url}")
    if initial and (parsed.hostname != "huggingface.co" or parsed.path != raw_path):
        raise ValueError(f"unexpected initial model LICENSE URL: {url}")


def _license_file(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _publisher_files(dist: metadata.Distribution) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=lambda value: str(value)):
        relative = str(entry)
        basename = Path(relative).name.casefold()
        stem = Path(basename).stem
        if stem not in LICENSE_FILE_NAMES and not any(
            marker in basename for marker in ("license", "copying", "notice", "copyright")
        ):
            continue
        path = dist.locate_file(entry)
        if path.is_file() and not path.is_symlink():
            result.append(_license_file(path, relative))
    return result


def _needed_libraries(path: Path) -> dict[str, Any]:
    if not path.is_file() or not _has_elf_magic(path):
        return {"format": "non-elf", "needed": []}
    try:
        completed = subprocess.run(
            ["readelf", "-d", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"format": "elf", "needed": [], "error": str(exc)}
    needed = sorted(
        match.group(1)
        for line in completed.stdout.splitlines()
        if (match := re.search(r"\(NEEDED\).*\[([^]]+)\]", line))
    )
    return {"format": "elf", "needed": needed}


def _has_elf_magic(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == ELF_MAGIC


def _native_files(dist: metadata.Distribution) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in sorted(dist.files or [], key=lambda value: str(value)):
        relative = str(entry)
        path = dist.locate_file(entry)
        lower_name = path.name.casefold()
        if not (
            path.suffix.casefold() in NATIVE_SUFFIXES
            or any(token in lower_name for token in (".so.", ".dylib."))
            or lower_name.endswith(".dll")
        ) or not path.is_file() or path.is_symlink():
            continue
        libraries = _needed_libraries(path)
        result.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "bundled": True,
                "needed": libraries,
            }
        )
    return result


def _license_fields(dist: metadata.Distribution) -> dict[str, Any]:
    value = dist.metadata.get("License-Expression") or dist.metadata.get("License")
    classifiers = sorted(
        entry.removeprefix("License :: ")
        for entry in dist.metadata.get_all("Classifier", [])
        if entry.startswith("License :: ")
    )
    return {
        "license": value.strip() if isinstance(value, str) and value.strip() else None,
        "license_expression": dist.metadata.get("License-Expression"),
        "license_classifiers": classifiers,
    }


def _active_lock_packages(lock: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in lock["package"] if row["source"] != {"virtual": "."}]
    if sys.platform == "darwin":
        wanted_torch = "2.13.0"
    else:
        wanted_torch = "2.13.0+cpu"
    active: list[dict[str, Any]] = []
    for row in rows:
        if row["name"] == "torch" and row["version"] != wanted_torch:
            continue
        active.append(row)
    return sorted(active, key=lambda row: (row["name"], row["version"]))


def _distribution_map() -> dict[str, list[metadata.Distribution]]:
    result: dict[str, list[metadata.Distribution]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            result.setdefault(name.casefold().replace("-", "_"), []).append(dist)
    return result


def _closure_differences(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = sorted((expected_counts - actual_counts).elements())
    unexpected = sorted((actual_counts - expected_counts).elements())
    return missing, unexpected


def _lock_identity(row: dict[str, Any]) -> str:
    return f"{row['name']}=={row['version']} ({canonical_json(row['source'])})"


def _repository_identity(project: Path) -> dict[str, Any]:
    repository = project.parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"git commit identity unavailable: {exc}") from exc
    return {
        "commit": commit,
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def audit_environment(project: Path) -> dict[str, Any]:
    lock_path = project / "uv.lock"
    pyproject_path = project / "pyproject.toml"
    lock_bytes = lock_path.read_bytes()
    pyproject_bytes = pyproject_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    preflight_gate._validate_lock_shape(lock, pyproject)
    if sha256_file(lock_path) != preflight_gate.LOCK_SHA256:
        raise ValueError("uv.lock bytes do not match the reviewed preflight digest")
    if sha256_file(pyproject_path) != preflight_gate.PYPROJECT_SHA256:
        raise ValueError("pyproject.toml bytes do not match the reviewed preflight digest")

    all_rows = sorted(lock["package"], key=lambda row: (row["name"], row["version"], canonical_json(row["source"])))
    expected = _active_lock_packages(lock)
    active_keys = {(row["name"], row["version"], canonical_json(row["source"])) for row in expected}
    distributions = _distribution_map()
    expected_identities = [
        f"{row['name'].casefold().replace('-', '_')}=={row['version']}" for row in expected
    ]
    actual_identities = sorted(
        f"{name}=={dist.version}"
        for name, values in distributions.items()
        for dist in values
    )
    missing, unexpected = _closure_differences(expected_identities, actual_identities)
    packages: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in expected:
        key = row["name"].casefold().replace("-", "_")
        candidates = [dist for dist in distributions.get(key, []) if dist.version == row["version"]]
        if len(candidates) != 1:
            failures.append(f"installed closure mismatch: {row['name']}=={row['version']}")
            packages.append({"lock": row, "installed": None})
            continue
        dist = candidates[0]
        license_data = _license_fields(dist)
        publisher_files = _publisher_files(dist)
        native_files = _native_files(dist)
        if not license_data["license"] and not license_data["license_classifiers"]:
            failures.append(f"missing publisher license metadata: {row['name']}=={row['version']}")
        if not publisher_files:
            failures.append(f"missing publisher license/notice files: {row['name']}=={row['version']}")
        if any("error" in item["needed"] for item in native_files):
            failures.append(f"ELF NEEDED inspection failed: {row['name']}=={row['version']}")
        packages.append(
            {
                "lock": {
                    "name": row["name"],
                    "version": row["version"],
                    "source": row["source"],
                    "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
                },
                "installed": {
                    "name": dist.metadata.get("Name"),
                    "version": dist.version,
                    **license_data,
                    "publisher_files": publisher_files,
                    "native_files": native_files,
                    "bundled_libraries": [item for item in native_files if item["bundled"]],
                },
            }
        )
    inactive_rows: list[dict[str, str]] = []
    active_packages = packages
    packages = []
    active_by_key = {
        (item["lock"]["name"], item["lock"]["version"], canonical_json(item["lock"]["source"])): item
        for item in active_packages
    }
    for row in all_rows:
        key = (row["name"], row["version"], canonical_json(row["source"]))
        if key in active_keys:
            packages.append(active_by_key[key])
            continue
        reason = (
            "virtual project row; no installed distribution is expected"
            if row["source"] == {"virtual": "."}
            else f"platform-inactive lock alternative on {sys.platform}"
        )
        inactive_rows.append({"identity": _lock_identity(row), "reason": reason})
        packages.append(
            {
                "lock": {
                    "name": row["name"],
                    "version": row["version"],
                    "source": row["source"],
                    "artifacts": {"sdist": row.get("sdist"), "wheels": row.get("wheels", [])},
                },
                "audit_status": "INACTIVE_LOCK_ROW",
                "inactive_reason": reason,
                "installed": None,
            }
        )
    if missing or unexpected:
        failures.append(
            "installed closure mismatch: "
            + canonical_json({"missing": missing, "unexpected": unexpected})
        )
    return {
        "schema": SCHEMA,
        "repository": _repository_identity(project),
        "project": {
            "name": pyproject["project"]["name"],
            "version": pyproject["project"]["version"],
            "pyproject_sha256": sha256_bytes(pyproject_bytes),
            "lock_sha256": sha256_bytes(lock_bytes),
        },
        "closure": {
            "platform": sys.platform,
            "python": platform.python_version(),
            "expected": expected_identities,
            "installed": actual_identities,
            "missing": missing,
            "unexpected": unexpected,
        },
        "locked_rows": [_lock_identity(row) for row in all_rows],
        "active_installed_rows": expected_identities,
        "inactive_rows": inactive_rows,
        "packages": packages,
        "failures": sorted(failures),
    }


def _controlled_license_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, HTTPError):
        return {"kind": "HTTP_ERROR", "status": exc.code}
    if isinstance(exc, URLError):
        return {"kind": "URL_ERROR"}
    if isinstance(exc, ValueError):
        return {"kind": "VALIDATION_ERROR"}
    if isinstance(exc, OSError):
        return {"kind": "OS_ERROR"}
    return {"kind": "UNEXPECTED_ERROR"}


def _blocked_license_record(repo: str, revision: str, exc: Exception) -> dict[str, Any]:
    url = model_license_url(repo, revision)
    return {
        "status": "BLOCKED_FACTUAL_LICENSE_PATH",
        "repo": repo,
        "revision": revision,
        "requested_url": url,
        "url": url,
        "resolved_host": None,
        "resolved_path": None,
        "acquired_bytes": False,
        "size": None,
        "sha256": None,
        "content_base64": None,
        "error": _controlled_license_error(exc),
    }


def _fetch_license(
    repo: str,
    revision: str,
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    if not any(item["repo"] == repo and item["revision"] == revision for item in MODEL_LICENSES):
        raise ValueError(f"model LICENSE identity is not fixed in the audit allowlist: {repo}@{revision}")
    url = model_license_url(repo, revision)
    _validate_license_url(url, repo, revision, initial=True)
    if fetcher is None:
        request = Request(url, headers={"Accept": "text/plain", "User-Agent": "vokra-qwen3-asr-audit/1"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - exact URL is allow-listed below
            final_url = response.geturl()
            _validate_license_url(final_url, repo, revision, initial=False)
            parsed = urlparse(final_url)
            body = response.read(MAX_LICENSE_BYTES + 1)
    else:
        final_url, body = fetcher(url)
        _validate_license_url(final_url, repo, revision, initial=False)
        parsed = urlparse(final_url)
    if len(body) > MAX_LICENSE_BYTES:
        raise ValueError("model LICENSE response exceeds the bounded audit size")
    return {
        "repo": repo,
        "revision": revision,
        "url": url,
        "resolved_host": parsed.hostname,
        "resolved_path": parsed.path,
        "size": len(body),
        "sha256": sha256_bytes(body),
        "content_base64": base64.b64encode(body).decode("ascii"),
        "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY",
    }


def audit_model_licenses(
    fetcher: Callable[[str], tuple[str, bytes]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in MODEL_LICENSES:
        try:
            records.append(_fetch_license(item["repo"], item["revision"], fetcher))
        except (OSError, UnicodeError, ValueError) as exc:
            records.append(_blocked_license_record(item["repo"], item["revision"], exc))
            error = _controlled_license_error(exc)
            failures.append(
                "BLOCKED_FACTUAL_LICENSE_PATH: "
                f"{item['repo']}@{item['revision']}: {canonical_json(error)}"
            )
    return records, failures


def _minimal_blocked_report(project: Path, exc: Exception) -> dict[str, Any]:
    project_data: dict[str, Any] = {
        "name": project.name,
        "version": None,
        "pyproject_sha256": None,
        "lock_sha256": None,
    }
    for filename, field in (("pyproject.toml", "pyproject_sha256"), ("uv.lock", "lock_sha256")):
        path = project / filename
        try:
            project_data[field] = sha256_file(path)
        except OSError:
            pass
    try:
        project_data["name"] = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        pass
    try:
        repository = _repository_identity(project)
    except Exception:
        repository = {"commit": None, "audit_script_sha256": sha256_file(Path(__file__).resolve())}
    failure = f"ENVIRONMENT_AUDIT_BLOCKED: {_controlled_license_error(exc)['kind']}"
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "repository": repository,
        "project": project_data,
        "closure": {
            "platform": sys.platform,
            "python": platform.python_version(),
            "expected": [],
            "installed": [],
            "missing": [],
            "unexpected": [],
        },
        "locked_rows": [],
        "active_installed_rows": [],
        "inactive_rows": [],
        "packages": [],
        "failures": [failure],
    }


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    environment_blocked = False
    try:
        report = audit_environment(project)
    except Exception as exc:  # fail closed while preserving a reviewable blocked report
        report = _minimal_blocked_report(project, exc)
        environment_blocked = True
    if fetch_model_licenses and not environment_blocked:
        records, license_failures = audit_model_licenses()
    else:
        records, license_failures = [], []
    report["model_license_files"] = records
    report["model_acquisition"] = {
        "policy": "allowlist-only LICENSE URLs",
        "requested_files": [item["requested_url"] if "requested_url" in item else item["url"] for item in records],
        "non_license_files": [],
    }
    report["failures"] = sorted([*report.get("failures", []), *license_failures])
    report["status"] = "BLOCKED" if report["failures"] else "PASS"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"qwen3-asr dependency audit: BLOCKED: report write failed ({type(exc).__name__})", file=sys.stderr)
        return 2
    if report["status"] == "BLOCKED":
        print("qwen3-asr dependency audit: BLOCKED: " + "; ".join(report["failures"]), file=sys.stderr)
        return 2
    print(f"qwen3-asr dependency audit: PASS ({output})")
    return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    assert len(_active_lock_packages(lock)) == 93
    expected_torch = {"2.13.0"} if sys.platform == "darwin" else {"2.13.0+cpu"}
    assert {row["version"] for row in _active_lock_packages(lock) if row["name"] == "torch"} == expected_torch
    assert _closure_differences(["torch==2.13.0+cpu"], ["torch==2.13.0"])[0] == ["torch==2.13.0+cpu"]
    assert _closure_differences(["anyio==4.14.2"], ["anyio==4.14.2", "anyio==4.14.2"])[1] == ["anyio==4.14.2"]
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert model_license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"]).endswith("/LICENSE")
    good_url = model_license_url(MODEL_LICENSES[0]["repo"], MODEL_LICENSES[0]["revision"])
    good_body = b"Apache License\n"
    requested_urls: list[str] = []

    def fetch_good(url: str) -> tuple[str, bytes]:
        requested_urls.append(url)
        return url, good_body

    good_records, good_failures = audit_model_licenses(fetch_good)
    assert good_records[0]["sha256"] == sha256_bytes(good_body)
    assert not good_failures
    assert requested_urls == [
        model_license_url(item["repo"], item["revision"]) for item in MODEL_LICENSES
    ]
    valid_cdn = (
        "https://cdn-lfs.huggingface.co/"
        f"{MODEL_LICENSES[0]['repo']}/resolve/{MODEL_LICENSES[0]['revision']}/LICENSE"
    )
    assert _fetch_license(
        MODEL_LICENSES[0]["repo"],
        MODEL_LICENSES[0]["revision"],
        lambda url: (valid_cdn, good_body),
    )["size"] == len(good_body)
    second_url = model_license_url(MODEL_LICENSES[1]["repo"], MODEL_LICENSES[1]["revision"])

    def blocked_record_test(
        fetcher: Callable[[str], tuple[str, bytes]],
        blocked_count: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records, failures = audit_model_licenses(fetcher)
        assert len(records) == len(MODEL_LICENSES)
        assert len(failures) == blocked_count
        assert sum(item.get("status") == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records) == blocked_count
        for item in records:
            if item.get("status") == "BLOCKED_FACTUAL_LICENSE_PATH":
                assert item["acquired_bytes"] is False
                assert item["content_base64"] is None
        return records, failures

    def fail_first(url: str) -> tuple[str, bytes]:
        if url == good_url:
            raise HTTPError(url, 404, "not found", {}, None)
        return url, good_body

    first_failed, first_failures = blocked_record_test(fail_first, 1)
    assert first_failed[0]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert first_failed[1]["sha256"] == sha256_bytes(good_body)
    assert first_failures[0].startswith("BLOCKED_FACTUAL_LICENSE_PATH:")

    def fail_second(url: str) -> tuple[str, bytes]:
        if url == second_url:
            raise URLError("offline")
        return url, good_body

    second_failed, second_failures = blocked_record_test(fail_second, 1)
    assert second_failed[0]["sha256"] == sha256_bytes(good_body)
    assert second_failed[1]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert second_failures[0].startswith("BLOCKED_FACTUAL_LICENSE_PATH:")

    both_failed, both_failures = blocked_record_test(
        lambda url: (_ for _ in ()).throw(ValueError("blocked path")), 2
    )
    assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in both_failed)
    assert len(both_failures) == 2

    partial_records, partial_failures = blocked_record_test(
        lambda url: (valid_cdn, good_body) if url == good_url else (_ for _ in ()).throw(URLError("offline")),
        1,
    )
    assert partial_records[0]["resolved_host"] == "cdn-lfs.huggingface.co"
    assert partial_records[1]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"
    assert len(partial_failures) == 1

    blocked_responses = (
        lambda url: (url.replace("LICENSE", "model.safetensors"), b"weights"),
        lambda url: ("https://example.invalid/LICENSE", good_body),
    )
    for blocked_response in blocked_responses:
        records, failures = blocked_record_test(blocked_response, 2)
        assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
        assert len(failures) == 2
    for redirected in (
        "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE.txt",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/model.safetensors",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE?download=1",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE#fragment",
        "https://user:password@cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE",
        "https://cdn-lfs.huggingface.co:8443/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/LICENSE.txt",
        "https://cdn-lfs.huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/model.safetensors",
    ):
        records, failures = blocked_record_test(lambda url, redirected=redirected: (redirected, good_body), 2)
        assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
        assert len(failures) == 2
    records, failures = blocked_record_test(lambda url: (url, b"x" * (MAX_LICENSE_BYTES + 1)), 2)
    assert all(item["status"] == "BLOCKED_FACTUAL_LICENSE_PATH" for item in records)
    assert len(failures) == 2

    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-blocked-") as directory:
        output = Path(directory) / "blocked.json"
        original_audit_environment = globals()["audit_environment"]
        original_audit_model_licenses = globals()["audit_model_licenses"]
        globals()["audit_environment"] = lambda project: {
            "schema": SCHEMA,
            "closure": {"expected": ["closure==facts"], "installed": ["closure==facts"]},
            "failures": [],
        }
        blocked = {
            "status": "BLOCKED_FACTUAL_LICENSE_PATH",
            "repo": MODEL_LICENSES[0]["repo"],
            "revision": MODEL_LICENSES[0]["revision"],
            "requested_url": good_url,
            "acquired_bytes": False,
            "content_base64": None,
        }
        globals()["audit_model_licenses"] = lambda: ([blocked], ["BLOCKED_FACTUAL_LICENSE_PATH: test"])
        try:
            assert run(Path(directory) / "project", output, True) == 2
        finally:
            globals()["audit_environment"] = original_audit_environment
            globals()["audit_model_licenses"] = original_audit_model_licenses
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["status"] == "BLOCKED"
        assert written["closure"]["expected"] == ["closure==facts"]
        assert written["model_license_files"][0]["status"] == "BLOCKED_FACTUAL_LICENSE_PATH"

    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-invalid-") as directory:
        output = Path(directory) / "blocked.json"
        assert run(Path(directory) / "missing-project", output, False) == 2
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["status"] == "BLOCKED"
        assert written["failures"]
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-audit-") as directory:
        output = Path(directory) / "audit.json"
        short = output.with_name("short")
        short.write_bytes(b"not an ELF payload")
        assert _needed_libraries(short) == {"format": "non-elf", "needed": []}
        report = {
            "schema": SCHEMA,
            "model_license_files": [],
            "model_acquisition": {"policy": "allowlist-only LICENSE URLs", "requested_files": [], "non_license_files": []},
        }
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
        assert json.loads(output.read_text(encoding="utf-8"))["schema"] == SCHEMA
    print("qwen3-asr dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch-model-licenses", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__":
    raise SystemExit(main())
