#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Fail-closed, model-free XCodec2 dependency and closure audit.

This module only validates the pinned documents and derives the Linux
CPython 3.12 x86_64 lock closure.  It never imports Torch, XCodec2, or any
model package.  Primary license bytes, native payloads, and owner decisions
are separate evidence gates collected by ``collect_dependency_evidence.py``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


PROJECT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT.parents[2]
PYPROJECT = PROJECT / "pyproject.toml"
LOCK = PROJECT / "uv.lock"
MANIFEST = PROJECT / "license_gate_manifest.json"
ROWS = PROJECT / "dependency_audit.json"
SCHEMA = "vokra-xcodec2-dependency-audit-v1"
MANIFEST_SCHEMA = "vokra-xcodec2-license-gate-v1"
SOURCE_DIGEST_FILES = (
    "dependency_audit.py",
    "dependency_guard.py",
    "collect_dependency_evidence.py",
    "inspect_locked_sdist_sources.py",
    "build_derived_locked_sdists.py",
    "dump_reference.py",
)
MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
TARGET_ENV = {
    "implementation_name": "cpython",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
}
PYPI_HOSTS = {"files.pythonhosted.org", "pypi.org"}
TORCH_HOSTS = {"download.pytorch.org", "download-r2.pytorch.org"}
TORCH_REGISTRIES = {"https://download.pytorch.org/whl/cpu"}
SETUPTOOLS_WHEEL = {
    "url": "https://files.pythonhosted.org/packages/95/9c/c510029fc6ef33a6275cd2c5d3cecd6613dfd6aa401d57c54f1c18852ccf/setuptools-84.0.0-py3-none-any.whl",
    "sha256": "51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670",
    "bytes": 818216,
}
SETUPTOOLS_DEVENDORING = {
    "distribution": "setuptools",
    "version": "84.0.0",
    "locked_wheel_sha256": SETUPTOOLS_WHEEL["sha256"],
    "locked_wheel_bytes": SETUPTOOLS_WHEEL["bytes"],
    "upstream_contract_url": "https://setuptools.pypa.io/en/latest/history.html",
    "upstream_contract_release": "71.0.0",
    "vendor_root": "setuptools/_vendor",
    "status": "CANDIDATE_NOT_BUILT",
}
FORBIDDEN_ROWS = {"transformers", "tokenizers", "typer", "shellingham"}
EXPECTED_NON_LINUX_ROWS_EXCLUDED_FROM_PRIOR_TALLY = {("torch", "2.13.0"), ("torchaudio", "2.11.0")}
# Independent `uv tree --frozen --offline --python-version 3.12
# --python-platform x86_64-unknown-linux-gnu` output at the bound lock SHA.
EXPECTED_LINUX_CLOSURE = frozenset({
    ("aiohappyeyeballs", "2.7.1"), ("aiohttp", "3.14.3"), ("aiosignal", "1.4.0"), ("anyio", "4.14.2"),
    ("antlr4-python3-runtime", "4.9.3"), ("attrs", "26.1.0"), ("blobfile", "3.3.0"), ("certifi", "2026.7.22"),
    ("charset-normalizer", "3.5.1"), ("click", "8.5.0"), ("datasets", "5.0.1"), ("dill", "0.4.1"),
    ("einops", "0.8.0"), ("einx", "0.4.3"), ("filelock", "3.32.4"), ("frozenlist", "1.8.0"),
    ("frozendict", "2.4.7"), ("fsspec", "2026.6.0"), ("gguf", "0.19.0"), ("h11", "0.16.0"),
    ("hf-xet", "1.6.0"), ("httpcore", "1.0.9"), ("httpx", "0.28.1"), ("huggingface-hub", "1.33.0"),
    ("idna", "3.19"), ("jinja2", "3.1.6"), ("lxml", "6.1.2"), ("markupsafe", "3.0.3"),
    ("mpmath", "1.3.0"), ("multidict", "6.9.1"), ("multiprocess", "0.70.19"), ("networkx", "3.6.1"),
    ("numpy", "2.0.2"), ("omegaconf", "2.3.1"), ("packaging", "26.3"), ("pandas", "3.0.5"),
    ("pillow", "12.3.0"), ("propcache", "0.5.2"), ("psutil", "7.2.2"), ("pyarrow", "25.0.1"),
    ("pycryptodomex", "3.23.0"), ("python-dateutil", "2.9.0.post0"), ("pyyaml", "6.0.3"), ("regex", "2026.7.19"),
    ("requests", "2.34.2"), ("safetensors", "0.8.0"), ("sentencepiece", "0.2.2"), ("setuptools", "84.0.0"),
    ("six", "1.17.0"), ("sympy", "1.14.0"), ("tiktoken", "0.14.0"), ("torch", "2.13.0+cpu"),
    ("torchao", "0.5.0"), ("torchaudio", "2.11.0+cpu"), ("torchtune", "0.3.1"), ("tqdm", "4.70.0"),
    ("typing-extensions", "4.16.0"), ("urllib3", "2.8.0"), ("vector-quantize-pytorch", "1.17.8"),
    ("vokra-xcodec2-parity", "0.1.0"), ("xcodec2", "0.1.5"), ("xxhash", "4.0.1"), ("yarl", "1.24.5"),
})
EXPECTED_PROJECT_NAME = "vokra-xcodec2-parity"
EXPECTED_LOCK_OVERRIDES = [
    {"name": "torch", "specifier": "==2.13.0", "index": "https://download.pytorch.org/whl/cpu"},
    {"name": "torchaudio", "specifier": "==2.11.0", "index": "https://download.pytorch.org/whl/cpu"},
    {"name": "setuptools", "specifier": "==84.0.0"},
    {"name": "transformers", "marker": "python_full_version < '0'"},
]
EXPECTED_PYPROJECT_OVERRIDES = [
    "torch==2.13.0",
    "torchaudio==2.11.0",
    "setuptools==84.0.0",
    "transformers ; python_version < '0'",
]


class AuditError(ValueError):
    """A fail-closed document or identity error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest_map() -> dict[str, str]:
    """Return the bounded, hash-bound production helper source set.

    The manifest deliberately excludes the manifest and row files themselves;
    those are already bound by the project/rows hashes and including either
    would create a self-referential hash.  Every helper used before model
    import is nevertheless covered by this explicit map.
    """
    result: dict[str, str] = {}
    for relative in SOURCE_DIGEST_FILES:
        path = PROJECT / relative
        try:
            stat = path.lstat()
        except OSError as exc:
            raise AuditError(f"source helper is unreadable: {relative}") from exc
        if not path.is_file() or path.is_symlink() or stat.st_size > MAX_SOURCE_FILE_BYTES:
            raise AuditError(f"source helper is not a bounded regular file: {relative}")
        result[relative] = sha256_file(path)
    return result


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical(value).encode("utf-8"))


def strict_json(path: Path) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"invalid JSON {path}: {exc}") from exc


def _marker_value(node: ast.AST, environment: dict[str, str]) -> tuple[str, bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, False
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id], node.id in {"python_version", "python_full_version"}
    raise AuditError("unsupported dependency marker value")


def _version_tuple(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise AuditError(f"unsupported Python version marker value: {value}")
    parts = [int(part) for part in value.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def marker_eval(node: ast.AST, environment: dict[str, str] = TARGET_ENV) -> bool:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [marker_eval(item, environment) for item in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left, left_version = _marker_value(node.left, environment)
        right, right_version = _marker_value(node.comparators[0], environment)
        if left_version or right_version:
            left, right = _version_tuple(left), _version_tuple(right)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
    raise AuditError("unsupported dependency marker expression")


def marker_reaches(marker: str, environment: dict[str, str] = TARGET_ENV) -> bool:
    if not isinstance(marker, str) or not marker.strip():
        raise AuditError("dependency marker must be non-empty text")
    try:
        return marker_eval(ast.parse(marker, mode="eval").body, environment)
    except (SyntaxError, AuditError) as exc:
        raise AuditError(f"dependency marker cannot be evaluated: {marker}") from exc


def marker_reaches_any(markers: Any) -> bool:
    if markers is None:
        return True
    if not isinstance(markers, list) or not all(isinstance(item, str) for item in markers):
        raise AuditError("resolution-markers must be a string array")
    return not markers or any(marker_reaches(item) for item in markers)


def _artifact_url_ok(url: Any, *, torch: bool) -> None:
    if not isinstance(url, str):
        raise AuditError("artifact URL must be an HTTPS URL without query/fragment")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise AuditError("artifact URL contains a control character")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError("artifact URL port is malformed") from exc
    allowed = TORCH_HOSTS if torch else PYPI_HOSTS
    if (
        parsed.scheme != "https"
        or "@" in parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port not in (None, 443)
        or parsed.hostname not in allowed
        or not parsed.path
    ):
        raise AuditError("artifact URL is not a strict HTTPS URL on the pinned host")


def _validate_artifact(artifact: Any, *, torch: bool) -> None:
    if not isinstance(artifact, dict):
        raise AuditError("lock artifact must be an object")
    required = {"url", "hash", "upload-time"}
    if not torch:
        required.add("size")
    if set(artifact) != required:
        raise AuditError(f"lock artifact fields drifted: {sorted(artifact)}")
    _artifact_url_ok(artifact["url"], torch=torch)
    if not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
        raise AuditError("lock artifact hash is not a lowercase SHA-256")
    if not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip():
        raise AuditError("lock artifact upload-time is missing")
    if not torch and (not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0):
        raise AuditError("lock artifact size is invalid")


def parse_lock_data(lock: dict[str, Any]) -> list[dict[str, Any]]:
    expected_top = {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}
    if set(lock) != expected_top or lock["version"] != 1 or lock["revision"] != 3 or lock["requires-python"] != "==3.12.*":
        raise AuditError("uv.lock top-level identity drifted")
    if not isinstance(lock["resolution-markers"], list) or not any("sys_platform != 'darwin'" in item for item in lock["resolution-markers"]):
        raise AuditError("Linux resolution marker is missing")
    if not isinstance(lock["manifest"], dict) or lock["manifest"].get("overrides") != EXPECTED_LOCK_OVERRIDES:
        raise AuditError("uv.lock override contract drifted")
    packages = lock["package"]
    if not isinstance(packages, list) or not packages:
        raise AuditError("uv.lock package rows are missing")
    identities: set[tuple[str, str]] = set()
    for row in packages:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
            raise AuditError("uv.lock package identity is malformed")
        identity = (row["name"].casefold(), row["version"])
        if identity in identities:
            raise AuditError(f"duplicate lock identity: {identity}")
        identities.add(identity)
        if row["name"].casefold() in FORBIDDEN_ROWS:
            raise AuditError(f"forbidden package row reintroduced: {row['name']}")
        source = row.get("source")
        if not isinstance(source, dict) or (source.get("virtual") != "." and source.get("registry") not in {"https://pypi.org/simple", *TORCH_REGISTRIES}):
            raise AuditError(f"unsupported package source: {row['name']}")
        if source.get("virtual") == ".":
            if "sdist" in row or "wheels" in row:
                raise AuditError("virtual project must not carry distribution artifacts")
            continue
        is_torch = source.get("registry") in TORCH_REGISTRIES
        if row.get("sdist") is not None:
            _validate_artifact(row["sdist"], torch=is_torch)
        wheels = row.get("wheels", [])
        if not isinstance(wheels, list):
            raise AuditError("lock wheels must be an array")
        if not wheels and row.get("sdist") is None:
            raise AuditError(f"package has no locked wheel or sdist: {row['name']}=={row['version']}")
        for wheel in wheels:
            _validate_artifact(wheel, torch=is_torch)
    return packages


def reachable_lock_identities(lock: dict[str, Any]) -> list[tuple[str, str]]:
    packages = parse_lock_data(lock)
    rows = {(row["name"].casefold(), row["version"]): row for row in packages}
    virtual = [key for key, row in rows.items() if row.get("source") == {"virtual": "."}]
    if len(virtual) != 1:
        raise AuditError("uv.lock must contain exactly one virtual project row")
    reachable: set[tuple[str, str]] = set(virtual)
    activated_extras: dict[tuple[str, str], set[str]] = {virtual[0]: set()}
    queue = [(virtual[0], set())]
    while queue:
        current, extras = queue.pop()
        row = rows[current]
        dependencies = list(row.get("dependencies", []))
        optional = row.get("optional-dependencies", {})
        if optional is not None:
            if not isinstance(optional, dict):
                raise AuditError("lock optional-dependencies must be an object")
            for extra in extras:
                if extra not in optional:
                    raise AuditError(f"lock extra activation is missing: {current[0]}[{extra!r}]")
                activated = optional[extra]
                if not isinstance(activated, list):
                    raise AuditError("lock activated extra dependencies must be an array")
                dependencies.extend(activated)
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise AuditError("lock dependency edge is malformed")
            edge_extras = dependency.get("extra", [])
            if not isinstance(edge_extras, list) or not all(isinstance(extra, str) for extra in edge_extras):
                raise AuditError("lock dependency extras are malformed")
            marker = dependency.get("marker")
            if marker is not None and not marker_reaches(marker):
                continue
            edge_version = dependency.get("version")
            if edge_version is not None and (not isinstance(edge_version, str) or not edge_version):
                raise AuditError("lock dependency version is malformed")
            edge_source = dependency.get("source")
            if edge_source is not None and not isinstance(edge_source, dict):
                raise AuditError("lock dependency source is malformed")
            candidates = [
                key
                for key, row in rows.items()
                if key[0] == dependency["name"].casefold()
                and (edge_version is None or key[1] == edge_version)
                and (edge_source is None or row.get("source") == edge_source)
                and marker_reaches_any(row.get("resolution-markers"))
            ]
            if len(candidates) != 1:
                raise AuditError(f"Linux closure edge is ambiguous or missing: {dependency['name']}")
            candidate = candidates[0]
            requested_extras = set(edge_extras)
            if candidate not in reachable:
                reachable.add(candidate)
                activated_extras[candidate] = requested_extras
                queue.append((candidate, requested_extras))
            elif requested_extras - activated_extras[candidate]:
                activated_extras[candidate].update(requested_extras)
                queue.append((candidate, activated_extras[candidate].copy()))
    return sorted(reachable)


def git_facts(expected_head: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_head):
        raise AuditError("--expected-head must be exactly lowercase 40-hex")
    try:
        head = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT, timeout=20).strip()
        status = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"], text=True, stderr=subprocess.STDOUT, timeout=20)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"git identity check failed: {type(exc).__name__}") from exc
    if head != expected_head:
        raise AuditError(f"HEAD differs from expected: {head} != {expected_head}")
    if status:
        raise AuditError("checkout is not clean")
    return {"head": head, "expected_head": expected_head, "clean": True}


def load_contract(expected_head: str | None = None) -> dict[str, Any]:
    if sys.platform != "linux" or __import__("platform").machine().casefold() not in {"x86_64", "amd64"} or sys.version_info[:2] != (3, 12):
        raise AuditError("XCodec2 collector requires Linux x86_64 CPython 3.12")
    pyproject_bytes = PYPROJECT.read_bytes()
    lock_bytes = LOCK.read_bytes()
    manifest = strict_json(MANIFEST)
    rows = strict_json(ROWS)
    pyproject = tomllib.loads(pyproject_bytes.decode("utf-8"))
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    if pyproject.get("project", {}).get("name") != EXPECTED_PROJECT_NAME:
        raise AuditError("project name drifted")
    tool_uv = pyproject.get("tool", {}).get("uv", {})
    if tool_uv.get("override-dependencies") != EXPECTED_PYPROJECT_OVERRIDES:
        raise AuditError("pyproject override contract drifted")
    if "setuptools==84.0.0" not in pyproject.get("project", {}).get("dependencies", []):
        raise AuditError("setuptools 84 must remain an explicit project dependency")
    parse_lock_data(lock)
    actual = {
        "pyproject_sha256": sha256_bytes(pyproject_bytes),
        "uv_lock_sha256": sha256_bytes(lock_bytes),
        "manifest_sha256": sha256_file(MANIFEST),
        "dependency_audit_sha256": sha256_file(ROWS),
        "collector_sha256": sha256_file(PROJECT / "collect_dependency_evidence.py"),
        "source_digests": source_digest_map(),
    }
    closure = reachable_lock_identities(lock)
    validate_gate_documents(manifest, rows, actual, closure)
    git = git_facts(expected_head) if expected_head is not None else None
    return {"pyproject": pyproject, "lock": lock, "manifest": manifest, "rows": rows, "hashes": actual, "closure": closure, "git": git}


def validate_gate_documents(manifest: dict[str, Any], rows: dict[str, Any], actual: dict[str, str], closure: list[tuple[str, str]]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("publication") != "NO_UPLOAD" or manifest.get("status") != "BLOCKED_PENDING_PRIMARY_BYTES":
        raise AuditError("license gate manifest is not fail-closed")
    project_contract = manifest.get("project")
    if not isinstance(project_contract, dict) or project_contract.get("path") != "tools/parity/xcodec2" or project_contract.get("python") != "==3.12.*" or project_contract.get("platform") != "linux-x86_64":
        raise AuditError("manifest target contract drifted")
    if project_contract.get("pyproject_sha256") != actual["pyproject_sha256"] or project_contract.get("uv_lock_sha256") != actual["uv_lock_sha256"] or project_contract.get("dependency_audit_sha256") != actual["dependency_audit_sha256"]:
        raise AuditError("manifest does not bind project/lock/rows hashes")
    expected_sources = actual.get("source_digests")
    if not isinstance(expected_sources, dict) or project_contract.get("source_digests") != expected_sources:
        raise AuditError("manifest does not bind audited helper source digests")
    dependency_contract = manifest.get("dependency_audit")
    if not isinstance(dependency_contract, dict) or dependency_contract.get("rows_file") != "dependency_audit.json" or dependency_contract.get("collector") != "collect_dependency_evidence.py" or dependency_contract.get("status") != "BLOCKED_PENDING_PRIMARY_BYTES" or type(dependency_contract.get("owner_review_required")) is not bool or dependency_contract["owner_review_required"] is not True:
        raise AuditError("manifest dependency audit contract drifted")
    if manifest.get("setuptools_devendoring") != {
        "distribution": "setuptools",
        "version": "84.0.0",
        "locked_wheel": SETUPTOOLS_WHEEL,
        "upstream_contract": {
            "source_url": SETUPTOOLS_DEVENDORING["upstream_contract_url"],
            "release": SETUPTOOLS_DEVENDORING["upstream_contract_release"],
            "vendor_root": SETUPTOOLS_DEVENDORING["vendor_root"],
            "operation": "downstream_devendor_by_removing_vendor_root",
        },
        "status": "CANDIDATE_NOT_BUILT",
        "owner_review_required": True,
    }:
        raise AuditError("setuptools de-vendoring contract drifted")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("license_classification") != "UNRESOLVED_PRIMARY_BYTES" or policy.get("native_payload") != "UNRESOLVED_PRIMARY_BYTES_AND_NEEDED" or policy.get("numpy_runtime") != "UNRESOLVED_GPL_GCC_LGPL_OR_OTHER" or type(policy.get("automatic_exceptions")) is not bool or policy["automatic_exceptions"] is not False or type(policy.get("no_upload")) is not bool or policy["no_upload"] is not True:
        raise AuditError("manifest policy is not fail-closed")
    if rows.get("schema") != SCHEMA or rows.get("status") != "BLOCKED_PENDING_PRIMARY_BYTES" or rows.get("publication") != "NO_UPLOAD":
        raise AuditError("dependency rows are not fail-closed")
    if rows.get("pyproject_sha256") != actual["pyproject_sha256"] or rows.get("uv_lock_sha256") != actual["uv_lock_sha256"]:
        raise AuditError("dependency rows do not bind project/lock hashes")
    if rows.get("source_digests") != expected_sources:
        raise AuditError("dependency rows do not bind audited helper source digests")
    if rows.get("setuptools_devendoring") != SETUPTOOLS_DEVENDORING:
        raise AuditError("dependency rows lost the setuptools de-vendoring contract")
    if set(closure) != EXPECTED_LINUX_CLOSURE:
        raise AuditError("Linux closure identity set differs from the audited UV target tree")
    if rows.get("expected_linux_closure_rows") != len(closure) or rows.get("expected_linux_external_rows") != len(closure) - 1:
        raise AuditError("dependency rows closure count drifted")


def self_test() -> int:
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    closure = reachable_lock_identities(lock)
    if set(closure) != EXPECTED_LINUX_CLOSURE or len(closure) != 63 or len(closure) - 1 != 62:
        raise AssertionError(f"unexpected Linux closure size: {len(closure)}")
    if any(name in FORBIDDEN_ROWS for name, _ in closure):
        raise AssertionError("forbidden row reached Linux closure")
    if EXPECTED_NON_LINUX_ROWS_EXCLUDED_FROM_PRIOR_TALLY & set(closure):
        raise AssertionError("macOS-only rows reached Linux closure")
    if not marker_reaches("sys_platform != 'darwin' and platform_machine == 'x86_64'"):
        raise AssertionError("Linux marker evaluator failed")
    if marker_reaches("sys_platform == 'darwin'"):
        raise AssertionError("macOS marker reached Linux target")
    for bad in ("A" * 40, "g" * 40, "0" * 39):
        try:
            git_facts(bad)
        except AuditError:
            pass
        else:
            raise AssertionError("malformed HEAD accepted")
    duplicate = '{"a": 1, "a": 2}'
    with tempfile.TemporaryDirectory(prefix="xcodec2-audit-self-test-") as directory:
        path = Path(directory) / "duplicate.json"
        path.write_text(duplicate, encoding="utf-8")
        try:
            strict_json(path)
        except AuditError:
            pass
        else:
            raise AssertionError("duplicate JSON key accepted")
    print("xcodec2 dependency audit: PASS (model-free static documents)")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is supported locally; collection is a separate VAST command")
    raise SystemExit(self_test())
