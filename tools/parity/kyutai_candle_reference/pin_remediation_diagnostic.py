#!/usr/bin/env -S uv run --no-project --python 3.12
"""Run a bounded, source-only Cargo pin experiment.

This diagnostic consumes one authenticated output from ``collect_metadata.py``
and operates on a fresh copy of its authenticated Moshi projection.  It is
deliberately not an advisory, license, or parity result: the output remains
review-pending and retains the unresolved ``paste`` finding.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_COLLECTOR_SHA256 = "75bdaa3da0ce7a3add28a1018e7d319b48b9f18448d7c3093c55fa94fdb2773f"
TARGET = "x86_64-unknown-linux-gnu"
DEFAULT_DEADLINE_SECONDS = 480.0
MAX_DEADLINE_SECONDS = 480.0
COLLECTOR_MARGIN_SECONDS = 2.0
MAX_FILE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_FAILURE_TEXT_BYTES = 16 * 1024
REGISTRY_SOURCE_PREFIX = "registry+"
PASTE_NAME = "paste"
PINS: tuple[tuple[str, str], ...] = (
    ("crossbeam-epoch", "0.9.20"),
    ("anyhow", "1.0.103"),
    ("rand", "0.9.3"),
)
CPU_METADATA_ARGS = ("metadata", "--locked", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features")
# Keep this exactly aligned with the frozen collector's 2 MiB whole-metadata
# capture branch. Cargo.lock is fingerprinted around every invocation below.
WHOLE_METADATA_ARGS = ("metadata", "--format-version", "1", "--all-features")


class DiagnosticError(Exception):
    """Fail-closed source-only diagnostic error."""


def _load_collector() -> Any:
    path = Path(__file__).with_name("collect_metadata.py")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_COLLECTOR_SHA256:
        raise DiagnosticError("collect_metadata.py does not match the reviewed frozen hash")
    spec = importlib.util.spec_from_file_location("kyutai_candle_collector_frozen", path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("frozen collector could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_bytes(collector: Any, path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    try:
        return collector.read_bytes(path, max_bytes=limit)
    except Exception as exc:
        if isinstance(exc, collector.CaptureError):
            raise DiagnosticError(str(exc)) from exc
        raise


def _read_json(collector: Any, path: Path, limit: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(collector, path, limit).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"invalid JSON evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON object required: {path.name}")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fresh(path: Path) -> None:
    collector = _load_collector()
    try:
        collector._assert_fresh_directory(path)
    except collector.CaptureError as exc:
        raise DiagnosticError(str(exc)) from exc


def _safe_rel(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DiagnosticError(f"unsafe evidence path: {value}")
    return relative


def _assert_file_receipt(collector: Any, path: Path, receipt: dict[str, Any], *, name: str) -> bytes:
    if not isinstance(receipt, dict):
        raise DiagnosticError(f"missing packet receipt: {name}")
    data = _read_bytes(collector, path)
    if receipt.get("bytes") != len(data) or receipt.get("sha256") != _sha256(data):
        raise DiagnosticError(f"packet receipt mismatch: {name}")
    return data


def _authenticate_packet(packet_root: Path, collector: Any) -> dict[str, Any]:
    if not packet_root.is_dir() or packet_root.is_symlink():
        raise DiagnosticError("packet root must be a regular directory")
    receipt = _read_json(collector, packet_root / "source_receipt.json")
    if receipt.get("source_commit") != collector.SOURCE_COMMIT:
        raise DiagnosticError("source receipt commit is not the fixed collector commit")
    if receipt.get("base_url") != collector.RAW_BASE:
        raise DiagnosticError("source receipt base URL is not the fixed official base")
    if receipt.get("provenance") != "official-raw-fixed-git-blob-authenticated":
        raise DiagnosticError("source receipt provenance is not authenticated")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(collector.SOURCE_FILES):
        raise DiagnosticError("source receipt file inventory is not exact")
    source_root = packet_root / "source" / "rust"
    total_bytes = 0
    for relative, (expected_blob, expected_size) in collector.SOURCE_FILES.items():
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise DiagnosticError(f"source receipt entry missing: {relative}")
        data = _read_bytes(collector, source_root / _safe_rel(relative))
        if len(data) != expected_size or entry.get("bytes") != expected_size:
            raise DiagnosticError(f"source byte bound mismatch: {relative}")
        if entry.get("url") != collector.SOURCE_URLS[relative]:
            raise DiagnosticError(f"source URL is not the fixed official URL: {relative}")
        if collector.git_blob_sha1(data) != expected_blob:
            raise DiagnosticError(f"source frozen Git blob mismatch: {relative}")
        if entry.get("git_blob_sha1") != expected_blob or entry.get("sha256") != _sha256(data):
            raise DiagnosticError(f"source receipt hash mismatch: {relative}")
        total_bytes += len(data)
    if receipt.get("total_bytes") != total_bytes:
        raise DiagnosticError("source receipt total byte count mismatch")

    packet = _read_json(collector, packet_root / "metadata.json")
    if packet.get("source_commit") != collector.SOURCE_COMMIT:
        raise DiagnosticError("metadata packet source commit mismatch")
    receipts = packet.get("lock_receipts")
    if not isinstance(receipts, dict):
        raise DiagnosticError("metadata packet lock receipts are missing")
    lock_paths = {
        "original": packet_root / "source" / "rust" / "Cargo.lock",
        "projected_base": packet_root / "base" / "Cargo.lock",
        "patched": packet_root / "patched" / "Cargo.lock",
        "final_projection": packet_root / "projection" / "Cargo.lock",
    }
    lock_bytes: dict[str, bytes] = {}
    for name, path in lock_paths.items():
        value = receipts.get(name)
        if not isinstance(value, dict):
            raise DiagnosticError(f"metadata packet lock receipt missing: {name}")
        lock_bytes[name] = _assert_file_receipt(collector, path, value, name=name)

    initial = _read_json(collector, packet_root / "initial_metadata.json")
    whole = _read_json(collector, packet_root / "whole_lock_metadata.json")
    if initial.get("source_commit") != collector.SOURCE_COMMIT or whole.get("source_commit") != collector.SOURCE_COMMIT:
        raise DiagnosticError("metadata evidence source commit mismatch")
    if initial.get("status") != "INITIAL_METADATA_UNREVIEWED":
        raise DiagnosticError("initial metadata evidence has an unexpected status")
    if whole.get("status") != "WHOLE_LOCK_METADATA_UNREVIEWED":
        raise DiagnosticError("whole metadata evidence has an unexpected status")
    if whole.get("target") is not None or whole.get("platform_filter") is not None:
        raise DiagnosticError("whole metadata evidence must be unfiltered")
    if not isinstance(initial.get("metadata"), dict) or not isinstance(whole.get("metadata"), dict):
        raise DiagnosticError("metadata evidence objects are missing")
    return {"receipt": receipt, "packet": packet, "initial": initial, "whole": whole, "lock_bytes": lock_bytes}


def _rebuild_projection(packet_root: Path, workspace: Path, collector: Any, final_lock: bytes) -> Path:
    source = packet_root / "source" / "rust"
    projection = workspace / "projection"
    try:
        collector._copy_source_to_projection(source, projection)
    except collector.CaptureError as exc:
        raise DiagnosticError(str(exc)) from exc
    lock = projection / "Cargo.lock"
    lock.write_bytes(final_lock)
    if lock.is_symlink() or _read_bytes(collector, lock) != final_lock:
        raise DiagnosticError("candidate projection lock was not replaced by an owned final copy")
    return projection


def _validate_argv(argv: list[str], cargo_name: str) -> None:
    if not argv or Path(argv[0]).name != cargo_name:
        raise DiagnosticError("candidate command executable is not the fixed cargo")
    tail = tuple(argv[1:])
    if tail in (CPU_METADATA_ARGS, WHOLE_METADATA_ARGS):
        return
    for package, version in PINS:
        if tail == ("update", "-p", package, "--precise", version):
            return
    raise DiagnosticError("candidate Cargo argv is outside the precise metadata allowlist")


def _stream_evidence(value: str) -> dict[str, Any]:
    data = value.encode("utf-8", errors="replace")
    return {
        "bytes": len(data),
        "sha256": _sha256(data),
        "text": data[:MAX_FAILURE_TEXT_BYTES].decode("utf-8", errors="replace"),
        "truncated": len(data) > MAX_FAILURE_TEXT_BYTES,
    }


def _command_evidence(argv: list[str], result: dict[str, Any], index: int) -> dict[str, Any]:
    if result.get("capture_error") is not None:
        return {
            "index": index,
            "argv": [Path(argv[0]).name, *argv[1:]],
            "returncode": None,
            "stdout": None,
            "stderr": None,
            "stream_status": "UNAVAILABLE_CAPTURE_ERROR",
            "capture_error": str(result["capture_error"]),
        }
    evidence = {
        "index": index,
        "argv": [Path(argv[0]).name, *argv[1:]],
        "returncode": result.get("returncode"),
        "stdout": _stream_evidence(str(result.get("stdout", ""))),
        "stderr": _stream_evidence(str(result.get("stderr", ""))),
    }
    if result.get("capture_error") is not None:
        evidence["capture_error"] = str(result["capture_error"])
    return evidence


def _lock_fingerprint(collector: Any, path: Path) -> dict[str, Any]:
    data = _read_bytes(collector, path)
    return {"bytes": len(data), "sha256": _sha256(data)}


def _assert_lock_unchanged(before: dict[str, Any], after: dict[str, Any], *, label: str) -> None:
    if before != after:
        raise DiagnosticError(f"Cargo.lock changed during {label} metadata capture")


def _assert_forbidden_packages_absent(candidate_lock: dict[tuple[str, str, str | None], dict[str, Any]]) -> None:
    forbidden_remaining = sorted({key[0] for key in candidate_lock if key[0] in {"backtrace", "adler"}})
    if forbidden_remaining:
        raise DiagnosticError(f"candidate full lock retains forbidden removed package(s): {', '.join(forbidden_remaining)}")


def _run_candidate(collector: Any, argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, cargo_name: str) -> dict[str, Any]:
    _validate_argv(argv, cargo_name)
    try:
        result = collector.run_bounded_subprocess(argv, cwd=cwd, env=env, deadline=deadline, validate=False)
    except collector.CaptureError as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": "", "capture_error": str(exc)}
    return result


def _lock_label(collector: Any, key: tuple[str, str, str | None]) -> str:
    return collector._lock_package_label(key)


def _safe_lock_record(collector: Any, record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, value in record.items():
        if field == "source" and isinstance(value, str):
            result[field] = {"kind": collector._lock_source_kind(value), "sha256": _sha256(value.encode())}
        elif field == "dependencies" and isinstance(value, list):
            result[field] = [collector._dependency_display(item) for item in value]
        else:
            result[field] = value
    return result


def _ordered_lock_delta(
    collector: Any,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    before_document: dict[str, Any] | None = None,
    after_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    old_keys = list(before)
    new_keys = list(after)
    removed = [key for key in old_keys if key not in after]
    added = [key for key in new_keys if key not in before]
    changed: list[dict[str, Any]] = []
    for key in sorted(set(before) & set(after), key=lambda item: _lock_label(collector, item)):
        left, right = before[key], after[key]
        if left == right:
            continue
        changed.append({"package": _lock_label(collector, key), "before": _safe_lock_record(collector, left), "after": _safe_lock_record(collector, right)})

    old_stable = [key for key in old_keys if key not in removed]
    new_stable = [key for key in new_keys if key not in added]
    before_root = {key: value for key, value in (before_document or {}).items() if key != "package"}
    after_root = {key: value for key, value in (after_document or {}).items() if key != "package"}
    root_changed = before_root != after_root
    return {
        "before_order": [_lock_label(collector, key) for key in old_keys],
        "after_order": [_lock_label(collector, key) for key in new_keys],
        "removed": [_lock_label(collector, key) for key in removed],
        "added": [_lock_label(collector, key) for key in added],
        "removed_records": [{"package": _lock_label(collector, key), "record": _safe_lock_record(collector, before[key])} for key in removed],
        "added_records": [{"package": _lock_label(collector, key), "record": _safe_lock_record(collector, after[key])} for key in added],
        "root_fields": {"before": before_root, "after": after_root, "changed": root_changed},
        "retained_record_changes": changed,
        "retained_order_changed": old_stable != new_stable,
        "pin_identity_changes": {
            name: {
                "requested_version": version,
                "before": [_lock_label(collector, key) for key in removed if key[0] == name],
                "after": [_lock_label(collector, key) for key in added if key[0] == name],
            }
            for name, version in PINS
        },
        "status": "COMPLETE_DIFF_REVIEW_PENDING",
    }


def _metadata_package_key(package: dict[str, Any]) -> tuple[str, str, str | None]:
    name, version, source = package.get("name"), package.get("version"), package.get("source")
    if not isinstance(name, str) or not isinstance(version, str) or (source is not None and not isinstance(source, str)):
        raise DiagnosticError("malformed candidate metadata package identity")
    return name, version, source


def _safe_metadata_value(value: Any, *, field: str | None = None) -> Any:
    if field == "source" and isinstance(value, str):
        return {"kind": value.split("+", 1)[0] if "+" in value else "other", "sha256": _sha256(value.encode())}
    if isinstance(value, dict):
        return {str(key): _safe_metadata_value(item, field=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_metadata_value(item) for item in value]
    return value


def _metadata_snapshot(collector: Any, metadata: dict[str, Any], *, whole: bool) -> dict[str, Any]:
    if whole:
        try:
            graph = collector._metadata_graph(metadata, require_complete_nodes=True)
        except collector.CaptureError as exc:
            raise DiagnosticError(str(exc)) from exc
    else:
        try:
            graph = collector.inspect_metadata(json.dumps(metadata, separators=(",", ":")))
        except collector.CaptureError as exc:
            raise DiagnosticError(str(exc)) from exc
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise DiagnosticError("candidate metadata packages are missing")
    package_by_id = {package["id"]: package for package in packages if isinstance(package, dict) and isinstance(package.get("id"), str)}
    labels = {package_id: _lock_label(collector, _metadata_package_key(package)) for package_id, package in package_by_id.items()}
    nodes = metadata.get("resolve", {}).get("nodes", [])
    edges: list[dict[str, Any]] = []
    features: dict[str, list[str]] = {}
    declarations: dict[str, list[dict[str, Any]]] = {}
    package_records: dict[str, dict[str, Any]] = {}
    for package_id, package in package_by_id.items():
        package_records[labels[package_id]] = _safe_metadata_value(package)
    for node in nodes:
        if not isinstance(node, dict) or node.get("id") not in labels:
            continue
        node_id = node["id"]
        features[labels[node_id]] = list(node.get("features", []))
        package = package_by_id[node_id]
        declarations[labels[node_id]] = [
            _safe_metadata_value(declaration)
            for declaration in package.get("dependencies", [])
            if isinstance(declaration, dict)
        ]
        for edge in node.get("deps", []):
            if isinstance(edge, dict) and edge.get("pkg") in labels:
                edges.append({"source": labels[node_id], "name": edge.get("name"), "target": labels[edge["pkg"]], "dep_kinds": edge.get("dep_kinds")})
    return {
        "package_order": [labels[package["id"]] for package in packages if isinstance(package, dict) and package.get("id") in labels],
        "features": features,
        "declarations": declarations,
        "package_records": package_records,
        "edges": edges,
        "active_graph": graph,
    }


def _graph_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def delta_map(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(set(left) | set(right)):
            if left.get(key, []) != right.get(key, []):
                rows.append({"package": key, "before": left.get(key, []), "after": right.get(key, [])})
        return rows

    before_edges = {_canonical_json(edge) for edge in before["edges"]}
    after_edges = {_canonical_json(edge) for edge in after["edges"]}
    return {
        "package_order_changed": before["package_order"] != after["package_order"],
        "feature_changes": delta_map(before["features"], after["features"]),
        "declaration_changes": delta_map(before["declarations"], after["declarations"]),
        "edges_added": [json.loads(edge) for edge in sorted(after_edges - before_edges)],
        "edges_removed": [json.loads(edge) for edge in sorted(before_edges - after_edges)],
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assert_success(result: dict[str, Any], *, label: str) -> None:
    if result.get("returncode") != 0:
        detail = result.get("capture_error") or f"returncode={result.get('returncode')}"
        raise DiagnosticError(f"{label} failed: {detail}")


def _graph_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Keep native Cargo metadata as the sole full graph serialization."""
    return {
        "package_count": len(snapshot["package_order"]),
        "package_order_sha256": _sha256(_canonical_json(snapshot["package_order"]).encode()),
        "features_sha256": _sha256(_canonical_json(snapshot["features"]).encode()),
        "declarations_sha256": _sha256(_canonical_json(snapshot["declarations"]).encode()),
        "edges_sha256": _sha256(_canonical_json(snapshot["edges"]).encode()),
        "edge_count": len(snapshot["edges"]),
    }


def _write_graph_evidence(collector: Any, path: Path, *, status: str, source_commit: str, target: str | None, metadata: dict[str, Any], snapshot: dict[str, Any]) -> None:
    collector.write_json(path, {
        "status": status,
        "source_commit": source_commit,
        "target": target,
        "platform_filter": target,
        "metadata": metadata,
        "graph": _graph_summary(snapshot),
        "cpu_execution_profile": "NOT_CPU_EXECUTION_PROFILE" if target is None else "CPU_PROFILE_OBSERVED_NO_GPU_APPROVAL",
        "gpu_execution": "NOT_AUTHORIZED",
        "approval": None,
        "upload": "NO_UPLOAD",
        "model_execution": "NOT_AUTHORIZED",
    })


def run(
    packet_root: Path,
    output_root: Path,
    workspace_root: Path,
    *,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    deadline_mono: float | None = None,
    collector: Any | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    if (platform_name or platform.system()) != "Linux":
        raise DiagnosticError("production diagnostic is Linux-only")
    if os.environ.get("VOKRA_CANDLE_REMOTE_LINUX") != "1":
        raise DiagnosticError("explicit VOKRA_CANDLE_REMOTE_LINUX=1 marker is required")
    if deadline_mono is None:
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0 or deadline_seconds > MAX_DEADLINE_SECONDS:
            raise DiagnosticError("deadline must be >0 and <=480 seconds")
        deadline_mono = time.monotonic() + deadline_seconds
    elif deadline_mono <= time.monotonic():
        raise DiagnosticError("shared deadline expired before diagnostic")
    if packet_root.absolute() == output_root.absolute() or packet_root.absolute() == workspace_root.absolute() or output_root.absolute() == workspace_root.absolute():
        raise DiagnosticError("packet, output, and workspace roots must be distinct")
    _fresh(output_root)
    _fresh(workspace_root)
    collector = collector or _load_collector()
    deadline = deadline_mono
    evidence: dict[str, Any] = {}
    commands: list[dict[str, Any]] = []
    lock_checkpoints: list[dict[str, Any]] = []
    diagnostic: dict[str, Any] = {
        "status": "PIN_REMEDIATION_BLOCKED",
        "source_only": True,
        "pins": [{"package": name, "version": version} for name, version in PINS],
        "unresolved": {"package": PASTE_NAME, "status": "EXPLICITLY_UNRESOLVED"},
        "approval": None,
        "upload": "NO_UPLOAD",
        "model_execution": "NOT_AUTHORIZED",
        "commands": commands,
        "lock_checkpoints": lock_checkpoints,
    }
    try:
        evidence = _authenticate_packet(packet_root, collector)
        base_lock_bytes = evidence["lock_bytes"]["projected_base"]
        patched_lock_bytes = evidence["lock_bytes"]["patched"]
        final_lock_bytes = evidence["lock_bytes"]["final_projection"]
        if final_lock_bytes != patched_lock_bytes:
            raise DiagnosticError("authenticated final projection lock differs from patched lock")
        try:
            base_validation = collector.validate_base_lock(
                packet_root / "source" / "rust" / "Cargo.lock",
                packet_root / "base" / "Cargo.lock",
                cpu_metadata=evidence["initial"]["metadata"],
                whole_metadata=evidence["whole"]["metadata"],
            )
            collector.validate_patch_lock(packet_root / "base" / "Cargo.lock", packet_root / "patched" / "Cargo.lock")
        except collector.CaptureError as exc:
            raise DiagnosticError(f"authenticated base/patch lock failed validation: {exc}") from exc
        final_lock_document = collector.read_toml(packet_root / "projection" / "Cargo.lock")
        final_lock = collector.lock_packages(final_lock_document)
        projection = _rebuild_projection(packet_root, workspace_root, collector, final_lock_bytes)
        cargo = shutil.which("cargo")
        if cargo is None or Path(cargo).name != "cargo":
            raise DiagnosticError("fixed cargo executable not found")
        cargo_home = workspace_root / "cargo-home"
        _fresh(cargo_home)
        env = collector.clean_cargo_environment(cargo_home)
        toolchain = collector._toolchain_probe(cargo, cwd=projection, env=env, deadline=min(deadline, time.monotonic() + 10))
        diagnostic["toolchain"] = toolchain
        diagnostic["historical_packet"] = {
            "source_receipt": evidence["receipt"],
            "base_lock": {"bytes": len(base_lock_bytes), "sha256": _sha256(base_lock_bytes)},
            "patched_lock": {"bytes": len(patched_lock_bytes), "sha256": _sha256(patched_lock_bytes)},
            "final_projection_lock": {"bytes": len(final_lock_bytes), "sha256": _sha256(final_lock_bytes)},
            "initial_status": evidence["initial"].get("status"),
            "whole_status": evidence["whole"].get("status"),
            "base_validation": base_validation,
        }
        baseline_cpu_argv = [cargo, *CPU_METADATA_ARGS]
        baseline_cpu_result = _run_candidate(collector, baseline_cpu_argv, cwd=projection, env=env, deadline=deadline, cargo_name=Path(cargo).name)
        commands.append(_command_evidence(baseline_cpu_argv, baseline_cpu_result, len(commands)))
        _assert_success(baseline_cpu_result, label="post-patch CPU metadata")
        baseline_whole_argv = [cargo, *WHOLE_METADATA_ARGS]
        baseline_whole_lock_before = _lock_fingerprint(collector, projection / "Cargo.lock")
        baseline_whole_result = _run_candidate(collector, baseline_whole_argv, cwd=projection, env=env, deadline=deadline, cargo_name=Path(cargo).name)
        commands.append(_command_evidence(baseline_whole_argv, baseline_whole_result, len(commands)))
        baseline_whole_lock_after = _lock_fingerprint(collector, projection / "Cargo.lock")
        _assert_lock_unchanged(baseline_whole_lock_before, baseline_whole_lock_after, label="post-patch whole")
        _assert_success(baseline_whole_result, label="post-patch whole metadata")
        try:
            baseline_cpu_metadata = json.loads(baseline_cpu_result["stdout"])
            baseline_whole_metadata = json.loads(baseline_whole_result["stdout"])
        except json.JSONDecodeError as exc:
            raise DiagnosticError("post-patch baseline metadata output is not JSON") from exc
        if not isinstance(baseline_cpu_metadata, dict) or not isinstance(baseline_whole_metadata, dict):
            raise DiagnosticError("post-patch baseline metadata objects are required")
        baseline_cpu = _metadata_snapshot(collector, baseline_cpu_metadata, whole=False)
        baseline_whole = _metadata_snapshot(collector, baseline_whole_metadata, whole=True)
        if set(baseline_whole["active_graph"]["ids_by_key"]) != set(final_lock):
            raise DiagnosticError("post-patch whole metadata identities do not match final lock")
        try:
            collector._validate_cpu_metadata_against_whole(baseline_cpu_metadata, baseline_whole["active_graph"], set(final_lock))
        except collector.CaptureError as exc:
            raise DiagnosticError(f"post-patch CPU/whole metadata mismatch: {exc}") from exc
        _write_graph_evidence(collector, output_root / "baseline_cpu_metadata.json", status="PIN_REMEDIATION_BASELINE_CPU_UNREVIEWED", source_commit=collector.SOURCE_COMMIT, target=TARGET, metadata=baseline_cpu_metadata, snapshot=baseline_cpu)
        _write_graph_evidence(collector, output_root / "baseline_whole_metadata.json", status="PIN_REMEDIATION_BASELINE_WHOLE_UNREVIEWED", source_commit=collector.SOURCE_COMMIT, target=None, metadata=baseline_whole_metadata, snapshot=baseline_whole)
        diagnostic["baseline"] = {
            "lock": {"bytes": len(final_lock_bytes), "sha256": _sha256(final_lock_bytes), "package_count": len(final_lock), "package_order": [_lock_label(collector, key) for key in final_lock]},
            "cpu": baseline_cpu["package_order"],
            "whole": baseline_whole["package_order"],
        }
        for package, version in PINS:
            argv = [cargo, "update", "-p", package, "--precise", version]
            result = _run_candidate(collector, argv, cwd=projection, env=env, deadline=deadline, cargo_name=Path(cargo).name)
            commands.append(_command_evidence(argv, result, len(commands)))
            _assert_success(result, label=f"precise update {package}")
            checkpoint = _read_bytes(collector, projection / "Cargo.lock")
            lock_checkpoints.append({"package": package, "version": version, "bytes": len(checkpoint), "sha256": _sha256(checkpoint)})
        candidate_lock_bytes = _read_bytes(collector, projection / "Cargo.lock")
        candidate_lock_document = collector.read_toml(projection / "Cargo.lock")
        candidate_lock = collector.lock_packages(candidate_lock_document)
        _assert_forbidden_packages_absent(candidate_lock)
        lock_delta = _ordered_lock_delta(
            collector,
            final_lock,
            candidate_lock,
            before_document=final_lock_document,
            after_document=candidate_lock_document,
        )
        for package, desired in PINS:
            before_matches = [key for key in final_lock if key[0] == package]
            matches = [key for key in candidate_lock if key[0] == package]
            if len(before_matches) != 1:
                raise DiagnosticError(f"baseline pin identity is ambiguous or missing: {package}")
            if len(matches) != 1 or matches[0][1] != desired or matches[0][2] != before_matches[0][2] or not isinstance(matches[0][2], str) or not matches[0][2].startswith(REGISTRY_SOURCE_PREFIX):
                raise DiagnosticError(f"candidate pin did not resolve exactly: {package} {desired}")
        cpu_argv = [cargo, *CPU_METADATA_ARGS]
        cpu_result = _run_candidate(collector, cpu_argv, cwd=projection, env=env, deadline=deadline, cargo_name=Path(cargo).name)
        commands.append(_command_evidence(cpu_argv, cpu_result, len(commands)))
        _assert_success(cpu_result, label="candidate CPU metadata")
        whole_argv = [cargo, *WHOLE_METADATA_ARGS]
        candidate_whole_lock_before = _lock_fingerprint(collector, projection / "Cargo.lock")
        whole_result = _run_candidate(collector, whole_argv, cwd=projection, env=env, deadline=deadline, cargo_name=Path(cargo).name)
        commands.append(_command_evidence(whole_argv, whole_result, len(commands)))
        candidate_whole_lock_after = _lock_fingerprint(collector, projection / "Cargo.lock")
        _assert_lock_unchanged(candidate_whole_lock_before, candidate_whole_lock_after, label="candidate whole")
        _assert_success(whole_result, label="candidate whole metadata")
        try:
            candidate_cpu_metadata = json.loads(cpu_result["stdout"])
            candidate_whole_metadata = json.loads(whole_result["stdout"])
        except json.JSONDecodeError as exc:
            raise DiagnosticError("candidate Cargo metadata output is not JSON") from exc
        if not isinstance(candidate_cpu_metadata, dict) or not isinstance(candidate_whole_metadata, dict):
            raise DiagnosticError("candidate Cargo metadata objects are required")
        candidate_cpu = _metadata_snapshot(collector, candidate_cpu_metadata, whole=False)
        candidate_whole = _metadata_snapshot(collector, candidate_whole_metadata, whole=True)
        candidate_keys = set(candidate_lock)
        if set(candidate_whole["active_graph"]["ids_by_key"]) != candidate_keys:
            raise DiagnosticError("candidate whole metadata identities do not match candidate lock")
        try:
            collector._validate_cpu_metadata_against_whole(candidate_cpu_metadata, candidate_whole["active_graph"], candidate_keys)
        except collector.CaptureError as exc:
            raise DiagnosticError(str(exc)) from exc
        cpu_keys = set(candidate_cpu["active_graph"]["resolved_node_ids"])
        whole_ids = set(candidate_whole["active_graph"]["ids_by_key"].values())
        if not cpu_keys.issubset(whole_ids):
            raise DiagnosticError("candidate CPU graph is outside candidate whole graph")
        final_lock = _read_bytes(collector, projection / "Cargo.lock")
        if final_lock != candidate_lock_bytes:
            raise DiagnosticError("candidate Cargo.lock changed after metadata capture")
        (output_root / "candidate.lock").write_bytes(candidate_lock_bytes)
        _write_graph_evidence(
            collector,
            output_root / "candidate_cpu_metadata.json",
            status="PIN_REMEDIATION_CPU_METADATA_UNREVIEWED",
            source_commit=collector.SOURCE_COMMIT,
            target=TARGET,
            metadata=candidate_cpu_metadata,
            snapshot=candidate_cpu,
        )
        _write_graph_evidence(
            collector,
            output_root / "candidate_whole_metadata.json",
            status="PIN_REMEDIATION_WHOLE_METADATA_UNREVIEWED",
            source_commit=collector.SOURCE_COMMIT,
            target=None,
            metadata=candidate_whole_metadata,
            snapshot=candidate_whole,
        )
        diagnostic.update({
            "status": "PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING",
            "source_commit": collector.SOURCE_COMMIT,
            "lock_delta": lock_delta,
            "candidate": {"bytes": len(candidate_lock_bytes), "sha256": _sha256(candidate_lock_bytes), "package_count": len(candidate_lock), "package_order": [_lock_label(collector, key) for key in candidate_lock]},
            "graph_differences": {"cpu": _graph_delta(baseline_cpu, candidate_cpu), "whole": _graph_delta(baseline_whole, candidate_whole)},
            "paste_status": "UNRESOLVED_AND_NOT_WAIVED",
        })
        collector.write_json(output_root / "pin_diagnostic.json", diagnostic)
        return diagnostic
    except (DiagnosticError, collector.CaptureError, OSError, ValueError) as exc:
        diagnostic["error"] = str(exc)
        diagnostic["completed_commands"] = len(commands)
        try:
            collector.write_json(output_root / "pin_diagnostic.json", diagnostic)
        except Exception:
            pass
        if isinstance(exc, DiagnosticError):
            raise
        raise DiagnosticError(str(exc)) from exc


def capture_and_run(
    packet_root: Path,
    output_root: Path,
    workspace_root: Path,
    *,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    platform_name: str | None = None,
) -> dict[str, Any]:
    if (platform_name or platform.system()) != "Linux":
        raise DiagnosticError("production capture is Linux-only")
    if os.environ.get("VOKRA_CANDLE_REMOTE_LINUX") != "1":
        raise DiagnosticError("explicit VOKRA_CANDLE_REMOTE_LINUX=1 marker is required")
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0 or deadline_seconds > MAX_DEADLINE_SECONDS:
        raise DiagnosticError("deadline must be >0 and <=480 seconds")
    collector = _load_collector()
    outer_deadline = time.monotonic() + deadline_seconds
    collector_seconds = outer_deadline - time.monotonic() - COLLECTOR_MARGIN_SECONDS
    if collector_seconds <= 0:
        raise DiagnosticError("shared deadline expired before source capture")
    try:
        collector.capture(packet_root, deadline_seconds=collector_seconds, platform_name="Linux")
    except collector.CaptureError as exc:
        raise DiagnosticError(f"source capture blocked: {exc}") from exc
    if time.monotonic() >= outer_deadline:
        raise DiagnosticError("shared deadline expired after source capture")
    return run(packet_root, output_root, workspace_root, deadline_mono=outer_deadline, collector=collector, platform_name="Linux")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("workspace_root", type=Path)
    parser.add_argument("--capture-source", action="store_true")
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.capture_source:
            capture_and_run(args.packet_root, args.output_root, args.workspace_root, deadline_seconds=args.deadline_seconds)
        else:
            run(args.packet_root, args.output_root, args.workspace_root, deadline_seconds=args.deadline_seconds)
    except (DiagnosticError, OSError, ValueError) as exc:
        print(f"PIN_REMEDIATION_BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING: no advisory/license/model/upload PASS claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
