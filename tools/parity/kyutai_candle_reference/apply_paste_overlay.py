#!/usr/bin/env -S uv run --no-project --python 3.12
"""Apply the authenticated paste overlay and collect bounded Cargo metadata.

This is a source-only diagnostic.  It never builds, checks, tests, runs, or
executes a downloaded crate.  The packet, candidate lock, and receipts are
authenticated before any overlay is applied; all commands share one absolute
deadline and every result is retained in a small failure receipt.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import tarfile
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
TARGET = "x86_64-unknown-linux-gnu"
COLLECTOR_SHA256 = "75bdaa3da0ce7a3add28a1018e7d319b48b9f18448d7c3093c55fa94fdb2773f"
PIN_SHA256 = "7bb61d1f2adaa58e64492584f67ecf7ca2b8d5f8f3943aa2a143b767e8661f6d"
PLANNER_SHA256 = "3e204bd5c0d47c097e5b5396f6c9787c39b49a296aedaf7c7ce528b0a4515dec"
PLAN_BYTES = 187212
PLAN_SHA256 = "677d8d840d43974c9f2ae1e567feb7df49169a3f06425e58f560e83b99b21874"
PARENT_RECEIPT_BYTES = 218910
PARENT_RECEIPT_SHA256 = "234a530ba183472daf0a34444703b934df741a50520f4c48cf421be25ec3ff36"
PASTEY_RECEIPT_BYTES = 176564
PASTEY_RECEIPT_SHA256 = "d4ca85e59f16d9ae6dc36f406473b6c534fe86895c353c3aca18930bce6edba2"
CANDIDATE_BYTES = 38252
CANDIDATE_SHA256 = "d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8"
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
MAX_DEADLINE = 480.0
MAX_JSON = 2 * 1024 * 1024
MAX_STREAM = 512 * 1024
MAX_WHOLE_STREAM = 2 * 1024 * 1024
MAX_ARCHIVE = 2 * 1024 * 1024
MAX_METAL_ARCHIVE = 3 * 1024 * 1024
MAX_ARCHIVE_TOTAL = 8 * 1024 * 1024
MAX_EXPANDED_TOTAL = 64 * 1024 * 1024
MAX_ARCHIVE_EXPANDED = 16 * 1024 * 1024
RECOVERY_RESERVE_SECONDS = 1.0
CPU_ARGS = ("metadata", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features")
CPU_LOCKED_ARGS = ("metadata", "--locked", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features")
WHOLE_ARGS = ("metadata", "--format-version", "1", "--all-features")

class OverlayError(RuntimeError):
    pass

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _read(path: Path, limit: int = MAX_JSON) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise OverlayError(f"missing regular evidence file: {path}")
    data = path.read_bytes()
    if len(data) > limit:
        raise OverlayError(f"evidence exceeds bound: {path}")
    return data

def _json(path: Path, limit: int = MAX_JSON) -> dict[str, Any]:
    value = json.loads(_read(path, limit))
    if not isinstance(value, dict):
        raise OverlayError(f"object JSON required: {path}")
    return value

def _write(path: Path, value: Any) -> None:
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_JSON:
        raise OverlayError(f"JSON exceeds bound: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise OverlayError(f"output must be fresh: {path}")
    path.write_bytes(raw)

def _rewrite(path: Path, value: Any) -> None:
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_JSON:
        raise OverlayError(f"JSON exceeds bound: {path}")
    path.write_bytes(raw)

def _rewrite_recovery(path: Path, value: Any) -> None:
    """Bounded receipt recovery path kept separate from normal finalization."""
    raw = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if len(raw) > MAX_JSON:
        raise OverlayError(f"JSON exceeds bound: {path}")
    path.write_bytes(raw)

def _fixed(path: Path, size: int, digest: str) -> bytes:
    data = _read(path, size + 1)
    if len(data) != size or _sha(data) != digest:
        raise OverlayError(f"authenticated input mismatch: {path}")
    return data

def _module(name: str, path: Path, digest: str) -> Any:
    data = _read(path, 8 * 1024 * 1024)
    if _sha(data) != digest:
        raise OverlayError(f"frozen helper hash mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OverlayError(f"cannot load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _remaining(end: float) -> float:
    left = end - time.monotonic()
    if left <= 0:
        raise OverlayError("shared deadline expired")
    return left

def _bounded_stage(collector: Any, end: float, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one post-capture blocking stage under the shared absolute deadline."""
    _remaining(end)
    with collector._SourceIODeadline(end):
        value = function(*args, **kwargs)
    _remaining(end)
    return value

def _receipt_file(path: Path, expected_size: int, expected_sha: str) -> dict[str, Any]:
    raw = _fixed(path, expected_size, expected_sha)
    return json.loads(raw)

def _safe_member(name: str, root: str) -> Path:
    if "\x00" in name or "\\" in name:
        raise OverlayError("unsafe archive member")
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts or p.parts[0] != root:
        raise OverlayError(f"unsafe archive member: {name}")
    return Path(*p.parts[1:])

def _fetch(url: str, size: int, digest: str, deadline: float, fetcher: Callable[..., bytes] | None = None) -> bytes:
    if not url.startswith("https://static.crates.io/crates/") or ".." in url:
        raise OverlayError("archive URL is not fixed static.crates.io")
    _remaining(deadline)
    if fetcher:
        data = fetcher(url, size, deadline)
    else:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise OverlayError("archive redirect rejected")
        opener = urllib.request.build_opener(NoRedirect)
        with opener.open(url, timeout=max(1, int(_remaining(deadline)))) as response:
            data = response.read(size + 1)
    if len(data) != size or _sha(data) != digest:
        raise OverlayError("archive size or checksum mismatch before extraction")
    return data

def _inventory(receipt: dict[str, Any], *, pastey: bool = False) -> dict[str, dict[str, Any]]:
    rows = receipt.get("members" if pastey else "regular_members")
    if not isinstance(rows, list):
        raise OverlayError("archive inventory missing")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise OverlayError("malformed archive inventory")
        path = row["path"]
        if pastey and "/" in path:
            path = path.split("/", 1)[1]
        if path in out:
            raise OverlayError("duplicate archive inventory member")
        out[path] = row
    return out

def _extract(data: bytes, receipt: dict[str, Any], dest: Path | None, *, pastey: bool = False, expected_root: str | None = None, deadline: float | None = None) -> list[dict[str, Any]]:
    expected = _inventory(receipt, pastey=pastey)
    if not isinstance(receipt.get("expanded_bytes"), int) or receipt["expanded_bytes"] > MAX_ARCHIVE_EXPANDED:
        raise OverlayError("archive expanded per-package bound exceeded")
    seen: set[str] = set()
    seen_dirs: set[str] = set()
    roots: set[str] = set()
    proof: list[dict[str, Any]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        expanded = gz.read(MAX_EXPANDED_TOTAL + 1)
    if len(expanded) != receipt.get("expanded_bytes"):
        raise OverlayError("archive expanded size mismatch")
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as tar:
        for member in tar:
            if deadline is not None:
                _remaining(deadline)
            if member.pax_headers or not member.name:
                raise OverlayError("PAX/invalid archive member rejected")
            parts = Path(member.name).parts
            if len(parts) < 1 or parts[0] in {".", ".."}:
                raise OverlayError(f"archive member has no package root: {member.name}")
            if expected_root is not None and parts[0] != expected_root:
                raise OverlayError(f"archive package root mismatch: {member.name}")
            roots.add(parts[0])
            if len(parts) == 1 and not member.isdir():
                raise OverlayError(f"archive root is not a directory: {member.name}")
            if len(parts) == 1 and member.isdir():
                if parts[0] in seen_dirs:
                    raise OverlayError(f"duplicate archive directory: {member.name}")
                seen_dirs.add(parts[0])
                continue
            rel = _safe_member(member.name, parts[0])
            key = rel.as_posix()
            if member.isdir():
                if key in expected or key in seen_dirs:
                    raise OverlayError(f"duplicate archive directory: {member.name}")
                seen_dirs.add(key)
                continue
            if key not in expected or key in seen:
                raise OverlayError(f"archive inventory mismatch: {member.name}")
            seen.add(key)
            row = expected[key]
            if not member.isfile():
                raise OverlayError(f"non-regular archive member: {member.name}")
            stream = tar.extractfile(member)
            body = stream.read() if stream is not None else b""
            if len(body) != row.get("bytes", row.get("size")) or _sha(body) != row.get("sha256"):
                raise OverlayError(f"archive member hash mismatch: {member.name}")
            if dest is not None:
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists() or out.is_symlink():
                    raise OverlayError(f"archive destination collision: {rel}")
                out.write_bytes(body)
            proof.append({"path": key, "archive_path": member.name, "bytes": len(body), "sha256": _sha(body)})
    if deadline is not None:
        _remaining(deadline)
    if set(seen) != set(expected):
        raise OverlayError("archive did not contain complete authenticated inventory")
    return proof

def _verify_tree(dest: Path, receipt: dict[str, Any], *, proposed_manifest: str) -> dict[str, Any]:
    expected = _inventory(receipt)
    expected_paths = set(expected)
    checked = 0
    for relative, row in expected.items():
        path = dest / relative
        if path.is_symlink() or not path.is_file():
            raise OverlayError(f"post-application member missing or non-regular: {relative}")
        body = path.read_bytes()
        if relative == "Cargo.toml":
            if body.decode() != proposed_manifest:
                raise OverlayError("parent manifest is not the exact planned overlay")
        elif len(body) != row["bytes"] or _sha(body) != row["sha256"]:
            raise OverlayError(f"post-application member changed: {relative}")
        checked += 1
    actual_paths = {path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise OverlayError("post-application tree contains missing or extra files")
    return {"members_checked": checked, "non_manifest_members": max(0, checked - 1)}


def _source_projection_proof(
    collector: Any,
    packet_root: Path,
    projection: Path,
    packet: dict[str, Any],
    *,
    expected_root_manifest: bytes,
    expected_lock: bytes,
    phase: str,
) -> dict[str, Any]:
    source_root = packet_root / "source" / "rust"
    receipt = packet.get("receipt", {})
    files = receipt.get("files") if isinstance(receipt, dict) else None
    if not isinstance(files, dict) or set(files) != set(collector.SOURCE_FILES):
        raise OverlayError(f"{phase} source receipt inventory is not exact")
    checked: list[dict[str, Any]] = []
    unchanged = 0
    for relative, (expected_blob, expected_size) in collector.SOURCE_FILES.items():
        source_body = _read(source_root / relative, expected_size + 1)
        entry = files.get(relative)
        if not isinstance(entry, dict) or len(source_body) != expected_size or collector.git_blob_sha1(source_body) != expected_blob:
            raise OverlayError(f"{phase} captured source file authentication failed: {relative}")
        source_sha = _sha(source_body)
        if entry.get("bytes") != expected_size or entry.get("git_blob_sha1") != expected_blob or entry.get("sha256") != source_sha:
            raise OverlayError(f"{phase} captured source receipt mismatch: {relative}")
        projection_body = _read(projection / relative, collector.MAX_FILE_BYTES)
        if relative == "Cargo.toml":
            if projection_body != expected_root_manifest:
                raise OverlayError(f"{phase} root Cargo.toml is not the exact expected overlay state")
        elif relative == "Cargo.lock":
            if projection_body != expected_lock:
                raise OverlayError(f"{phase} root Cargo.lock is not the exact expected lock state")
        else:
            if projection_body != source_body:
                raise OverlayError(f"{phase} unchanged Moshi source file drifted: {relative}")
            unchanged += 1
        checked.append({"path": relative, "bytes": len(source_body), "git_blob_sha1": expected_blob, "sha256": source_sha})
    if len(checked) != 23 or unchanged != 21:
        raise OverlayError(f"{phase} source proof is not exactly 23 files/21 unchanged files")
    return {"files_checked": len(checked), "unchanged_files": unchanged, "entries": checked, "phase": phase}

def _record_command(pin: Any, argv: list[str], result: dict[str, Any], index: int) -> dict[str, Any]:
    return pin._command_evidence(argv, result, index)

def _run(collector: Any, cargo: str, args: tuple[str, ...], cwd: Path, env: dict[str, str], end: float) -> dict[str, Any]:
    if tuple(args) not in (CPU_ARGS, CPU_LOCKED_ARGS, WHOLE_ARGS):
        raise OverlayError("metadata argv is outside fixed allowlist")
    try:
        return collector.run_bounded_subprocess([cargo, *args], cwd=cwd, env=env, deadline=end, validate=False)
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": "", "capture_error": str(exc)}

def _metadata(value: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value.get("packages"), list) or not isinstance(value.get("resolve"), dict):
        raise OverlayError(f"{label} metadata is incomplete")
    return {"label": label, "package_count": len(value["packages"]), "metadata": value}

def _lock_delta(collector: Any, before: dict[str, Any], after: dict[str, Any], before_doc: dict[str, Any], after_doc: dict[str, Any], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    """Return and validate the complete ordered lock delta, never counts only."""
    expected_parent = {(p["identity"].split("@", 1)[0], p["identity"].split("@", 1)[1], REGISTRY) for p in proposals}
    old_keys, new_keys = list(before), list(after)
    removed = set(before) - set(after)
    added = set(after) - set(before)
    paste_old = ("paste", "1.0.15", REGISTRY)
    paste_new = ("pastey", "0.2.3", REGISTRY)
    expected_removed = expected_parent | {paste_old}
    if removed != expected_removed:
        raise OverlayError(f"unexpected lock removals: {sorted(removed ^ expected_removed)}")
    for key in expected_parent | {paste_old}:
        record = before.get(key)
        if not isinstance(record, dict) or record.get("source") != REGISTRY or not isinstance(record.get("checksum"), str) or re.fullmatch(r"[0-9a-f]{64}", record["checksum"]) is None:
            raise OverlayError(f"registry parent lock record is missing its literal source/checksum: {key}")
    parent_new = {
        key for key in added
        if key[0] in {p["identity"].split("@", 1)[0] for p in proposals}
        and key[2] is None
    }
    if paste_new not in added or len(parent_new) != len(expected_parent) or added != parent_new | {paste_new}:
        raise OverlayError("candidate lock additions are not exactly source-free parent records plus registry pastey")
    if after[paste_new].get("checksum") != "2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4":
        raise OverlayError("pastey lock checksum is not the authenticated official archive")
    if after[paste_new].get("name") != "pastey" or after[paste_new].get("version") != "0.2.3" or after[paste_new].get("source") != REGISTRY or after[paste_new].get("dependencies", []) != [] or set(after[paste_new]) - {"name", "version", "source", "checksum"}:
        raise OverlayError("pastey lock record has unexpected fields or dependencies")
    def dependencies_allowed(before_deps: Any, after_deps: Any) -> bool:
        if not isinstance(before_deps, list) or not isinstance(after_deps, list) or len(before_deps) != len(after_deps):
            return False
        for old_dep, new_dep in zip(before_deps, after_deps):
            if old_dep == new_dep:
                continue
            if old_dep in {"paste", "paste 1.0.15"} and new_dep in {"pastey", "pastey 0.2.3"}:
                continue
            return False
        return True

    for old in expected_parent:
        candidates = [key for key in added if key[:2] == old[:2]]
        if len(candidates) != 1 or candidates[0][2] is not None:
            raise OverlayError(f"parent was not replaced by one source-free identity: {old}")
        new = candidates[0]
        old_record, new_record = before[old], after[new]
        if "source" in new_record or "checksum" in new_record:
            raise OverlayError(f"source-free parent lock record retained source/checksum: {new}")
        for field in set(old_record) | set(new_record):
            if field in {"source", "checksum", "dependencies"}:
                continue
            if old_record.get(field) != new_record.get(field):
                raise OverlayError(f"parent full record changed unexpectedly: {old} field {field}")
        if not dependencies_allowed(old_record.get("dependencies", []), new_record.get("dependencies", [])):
            raise OverlayError(f"parent dependency edges changed outside paste identity mapping: {old}")
    old_root = {k: v for k, v in before_doc.items() if k != "package"}
    new_root = {k: v for k, v in after_doc.items() if k != "package"}
    if old_root != new_root:
        raise OverlayError("Cargo.lock root fields changed during overlay")
    stable_old = [key for key in old_keys if key not in removed]
    stable_new = [key for key in new_keys if key not in added]
    if stable_old != stable_new:
        raise OverlayError("unrelated Cargo.lock package order changed")
    changed = []
    for key in stable_old:
        if before[key] != after[key]:
            left = dict(before[key]); right = dict(after[key])
            old_dependencies = left.pop("dependencies", [])
            new_dependencies = right.pop("dependencies", [])
            if old_dependencies != new_dependencies:
                if key not in expected_parent or not dependencies_allowed(old_dependencies, new_dependencies):
                    raise OverlayError(f"retained lock record changed outside paste identity mapping: {key}")
            if left != right:
                raise OverlayError(f"retained lock record changed outside paste identity mapping: {key}")
            changed.append({"package": str(key), "before": before[key], "after": after[key]})
    return {"status": "COMPLETE_ORDERED_DIFF_REVIEW_PENDING", "before_order": [str(k) for k in old_keys], "after_order": [str(k) for k in new_keys], "removed_records": [{"package": str(k), "record": before[k]} for k in old_keys if k in removed], "added_records": [{"package": str(k), "record": after[k]} for k in new_keys if k in added], "retained_record_changes": changed, "root_fields": {"before": old_root, "after": new_root, "changed": False}}

def _raw_graph_shape(metadata: dict[str, Any], label: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(packages, list) or not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list):
        raise OverlayError(f"{label} raw metadata graph is incomplete")
    package_by_id: dict[str, dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise OverlayError(f"{label} raw metadata package is malformed")
        if package["id"] in package_by_id:
            raise OverlayError(f"{label} raw metadata package id is duplicated")
        package_by_id[package["id"]] = package
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in resolve["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or node["id"] not in package_by_id or node["id"] in node_by_id:
            raise OverlayError(f"{label} raw metadata resolve node is malformed")
        node_by_id[node["id"]] = node
        if not isinstance(node.get("deps"), list) or not isinstance(node.get("dependencies", []), list):
            raise OverlayError(f"{label} raw metadata resolve edges are malformed")
        for edge in node["deps"]:
            if not isinstance(edge, dict) or not isinstance(edge.get("pkg"), str) or edge["pkg"] not in package_by_id:
                raise OverlayError(f"{label} raw metadata resolve edge is unknown")
    if set(node_by_id) != set(package_by_id):
        raise OverlayError(f"{label} raw metadata package/node identities differ")
    return package_by_id, node_by_id


def _canonicalize_overlay_metadata(
    metadata: dict[str, Any],
    before: dict[str, Any],
    proposals: list[dict[str, Any]],
    vendor_root: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package_by_id, node_by_id = _raw_graph_shape(metadata, label)
    old_by_identity = {
        (package.get("name"), package.get("version"), _package_source(package)): package
        for package in before.get("packages", [])
        if isinstance(package, dict)
    }
    parent_keys = {
        (proposal["identity"].split("@", 1)[0], proposal["identity"].split("@", 1)[1], None)
        for proposal in proposals
    }
    canonical = copy.deepcopy(metadata)
    canonical_packages, canonical_nodes = _raw_graph_shape(canonical, f"{label} canonical")
    path_ids: set[str] = set()
    proof_rows: list[dict[str, Any]] = []
    normalized = 0
    for package_id, package in package_by_id.items():
        key = (package.get("name"), package.get("version"), _package_source(package))
        if key not in parent_keys:
            continue
        path_ids.add(package_id)
        old = old_by_identity.get((package.get("name"), package.get("version"), REGISTRY))
        if old is None:
            raise OverlayError(f"{label} source-free parent has no registry identity proof")
        if not _owned_vendor_path(package.get("manifest_path"), package, vendor_root, "Cargo.toml"):
            raise OverlayError(f"{label} source-free parent manifest path is outside owned vendor root")
        canonical_packages[package_id]["source"] = REGISTRY
        proof_rows.append({"package_id": package_id, "field": "source", "from": None, "to": REGISTRY, "reason": "exact-owned-path-lock-identity"})
        normalized += 1
        old_targets = old.get("targets", [])
        new_targets = package.get("targets", [])
        if not isinstance(old_targets, list) or not isinstance(new_targets, list) or len(old_targets) != len(new_targets):
            raise OverlayError(f"{label} source-free parent target inventory changed")
        for old_target, new_target in zip(old_targets, new_targets):
            old_src = old_target.get("src_path") if isinstance(old_target, dict) else None
            new_src = new_target.get("src_path") if isinstance(new_target, dict) else None
            try:
                relative_src = Path(str(old_src)).relative_to(Path(str(old.get("manifest_path"))).parent)
            except ValueError:
                raise OverlayError(f"{label} source-free parent target source path is not under original manifest")
            if not _owned_vendor_path(new_src, package, vendor_root, relative_src):
                raise OverlayError(f"{label} source-free parent target source path is outside owned vendor root")
    expected_parent_count = len(parent_keys) if label == "overlay whole" else sum(1 for proposal in proposals if proposal.get("cpu"))
    if len(path_ids) != expected_parent_count:
        raise OverlayError(f"{label} source-free parent identity count is not exact")
    for node_id, node in node_by_id.items():
        declarations = canonical_packages[node_id].get("dependencies", [])
        if not isinstance(declarations, list):
            raise OverlayError(f"{label} package declarations are malformed")
        for edge in node.get("deps", []):
            target_id = edge["pkg"]
            if target_id not in path_ids:
                continue
            edge_name = str(edge.get("name"))
            matched = 0
            for declaration in declarations:
                if not isinstance(declaration, dict):
                    continue
                alias = str(declaration.get("rename") or declaration.get("name"))
                if alias.replace("-", "_") != edge_name.replace("-", "_"):
                    continue
                matched += 1
                source = declaration.get("source")
                if source is None:
                    declaration["source"] = REGISTRY
                    normalized += 1
                    proof_rows.append({"package_id": node_id, "edge_name": edge_name, "target_id": target_id, "field": "source", "from": None, "to": REGISTRY, "reason": "exact-owned-path-lock-identity"})
                elif source != REGISTRY:
                    raise OverlayError(f"{label} path-parent declaration has unexpected source")
            if matched != 1:
                raise OverlayError(f"{label} path-parent edge has no unique declaration correspondence")
    return canonical, {
        "mode": "EXACT_16_OWNED_PATH_CANONICAL_VIEW",
        "path_parent_ids": sorted(path_ids),
        "declaration_source_normalizations": proof_rows,
        "normalized_fields": normalized,
        "raw_preserved": True,
        "raw_sha256": _sha(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()),
        "canonical_sha256": _sha(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()),
    }


def _validate_graphs(
    collector: Any,
    cpu: dict[str, Any],
    whole: dict[str, Any],
    lock: dict[tuple[str, str, str | None], dict[str, Any]],
    label: str,
    *,
    before_cpu: dict[str, Any] | None = None,
    before_whole: dict[str, Any] | None = None,
    proposals: list[dict[str, Any]] | None = None,
    vendor_root: Path | None = None,
) -> dict[str, Any]:
    try:
        if len(whole.get("packages", [])) != 154:
            raise OverlayError(f"{label} metadata package/CPU closure counts are not 154/105")
        cpu_graph = None
        normalization: dict[str, Any] = {"mode": "RAW_GRAPH", "raw_preserved": True}
        graph_cpu, graph_whole = cpu, whole
        if before_cpu is not None or before_whole is not None or proposals is not None or vendor_root is not None:
            if not (before_cpu and before_whole and proposals and vendor_root):
                raise OverlayError(f"{label} canonical graph inputs are incomplete")
            graph_cpu, cpu_proof = _canonicalize_overlay_metadata(cpu, before_cpu, proposals, vendor_root, f"{label} CPU")
            graph_whole, whole_proof = _canonicalize_overlay_metadata(whole, before_whole, proposals, vendor_root, f"{label} whole")
            normalization = {"mode": "EXACT_16_OWNED_PATH_CANONICAL_VIEW", "cpu": cpu_proof, "whole": whole_proof, "raw_preserved": True}
        cpu_graph = collector.inspect_metadata(json.dumps(graph_cpu, separators=(",", ":")))
        if len(cpu_graph.get("resolved_node_ids", [])) != 105:
            raise OverlayError(f"{label} metadata package/CPU closure counts are not 154/105")
        whole_graph = collector._metadata_graph(graph_whole, require_complete_nodes=True)
        lock_keys = set(lock)
        canonical_lock_keys = lock_keys
        if proposals:
            path_lock_keys = {
                (proposal["identity"].split("@", 1)[0], proposal["identity"].split("@", 1)[1], None)
                for proposal in proposals
            }
            if not path_lock_keys.issubset(lock_keys):
                raise OverlayError(f"{label} final lock is missing exact source-free parent identities")
            canonical_lock_keys = {
                (name, version, REGISTRY) if (name, version, source) in path_lock_keys else (name, version, source)
                for name, version, source in lock_keys
            }
        ids_by_key = whole_graph.get("ids_by_key")
        if not isinstance(ids_by_key, dict) or set(ids_by_key) != canonical_lock_keys:
            raise OverlayError(f"{label} whole metadata identities do not equal complete lock")
        cpu_ids = set(cpu_graph.get("resolved_node_ids", []))
        whole_ids = set(ids_by_key.values())
        if not cpu_ids.issubset(whole_ids):
            raise OverlayError(f"{label} CPU graph is outside whole graph")
        collector._validate_cpu_metadata_against_whole(graph_cpu, whole_graph, canonical_lock_keys)
        return normalization
    except OverlayError:
        raise
    except Exception as exc:
        raise OverlayError(f"{label} metadata graph validation failed: {exc}") from exc

def _package_source(package: dict[str, Any]) -> str | None:
    source = package.get("source")
    return source if isinstance(source, str) else None

def _identity_map(before: dict[str, Any], after: dict[str, Any], proposals: list[dict[str, Any]]) -> dict[str, str]:
    old = {p.get("id"): p for p in before.get("packages", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}
    new = {p.get("id"): p for p in after.get("packages", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}
    parent_names = {p["identity"].split("@", 1)[0] for p in proposals}
    result: dict[str, str] = {}
    for old_id, package in old.items():
        name, version, source = package.get("name"), package.get("version"), _package_source(package)
        wanted_name, wanted_version, wanted_source = name, version, source
        if name == "paste" and version == "1.0.15" and source == REGISTRY:
            wanted_name, wanted_version, wanted_source = "pastey", "0.2.3", REGISTRY
        elif name in parent_names and source == REGISTRY:
            wanted_source = None
        matches = [new_id for new_id, candidate in new.items() if candidate.get("name") == wanted_name and candidate.get("version") == wanted_version and _package_source(candidate) == wanted_source]
        if len(matches) != 1:
            raise OverlayError(f"metadata identity mapping is ambiguous or missing: {name} {version}")
        result[old_id] = matches[0]
    if len(result) != len(old) or set(result.values()) != set(new):
        raise OverlayError("metadata identity mapping is not a complete bijection")
    return result

def _assert_graph_delta(before: dict[str, Any], after: dict[str, Any], proposals: list[dict[str, Any]], label: str, vendor_root: Path) -> None:
    old = {p.get("id"): p for p in before.get("packages", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}
    new = {p.get("id"): p for p in after.get("packages", []) if isinstance(p, dict) and isinstance(p.get("id"), str)}
    mapping = _identity_map(before, after, proposals)
    parent_names = {p["identity"].split("@", 1)[0] for p in proposals}
    for old_id, new_id in mapping.items():
        left, right = old[old_id], new[new_id]
        if left.get("name") == "paste" and left.get("version") == "1.0.15":
            continue
        allowed_path = left.get("name") in parent_names and _package_source(left) == REGISTRY
        for field in set(left) | set(right):
            if field in {"id", "source", "manifest_path", "targets", "dependencies"}:
                continue
            if left.get(field) != right.get(field):
                raise OverlayError(f"{label} package record changed unexpectedly: {left.get('name')} field {field}")
        if not allowed_path and left.get("manifest_path") != right.get("manifest_path"):
            raise OverlayError(f"{label} unrelated manifest path changed")
        if allowed_path:
            if not _owned_vendor_path(right.get("manifest_path"), left, vendor_root, "Cargo.toml"):
                raise OverlayError(f"{label} parent manifest path is not exact vendor path")
        if allowed_path:
            if not isinstance(left.get("targets"), list) or not isinstance(right.get("targets"), list) or len(left["targets"]) != len(right["targets"]):
                raise OverlayError(f"{label} target records changed unexpectedly")
            for old_target, new_target in zip(left["targets"], right["targets"]):
                old_target = dict(old_target); new_target = dict(new_target)
                old_src = old_target.pop("src_path", None); new_src = new_target.pop("src_path", None)
                try:
                    relative_src = Path(str(old_src)).relative_to(Path(str(left.get("manifest_path"))).parent)
                except ValueError:
                    relative_src = None
                if old_target != new_target or relative_src is None or not _owned_vendor_path(new_src, left, vendor_root, relative_src):
                    raise OverlayError(f"{label} parent target transformation is not exact")
        elif left.get("targets") != right.get("targets"):
            raise OverlayError(f"{label} target records changed unexpectedly")
        if left.get("dependencies") != right.get("dependencies"):
            if not allowed_path or not _parent_dependency_transform(left.get("dependencies"), right.get("dependencies")):
                raise OverlayError(f"{label} dependency declaration changed unexpectedly")
    old_nodes = {node.get("id"): node for node in before.get("resolve", {}).get("nodes", []) if isinstance(node, dict)}
    new_nodes = {node.get("id"): node for node in after.get("resolve", {}).get("nodes", []) if isinstance(node, dict)}
    for old_id, new_id in mapping.items():
        left, right = old_nodes.get(old_id), new_nodes.get(new_id)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise OverlayError(f"{label} resolve node mapping is incomplete")
        if old.get(old_id, {}).get("name") == "paste":
            if left.get("features", []) != right.get("features", []) or right.get("features", []) != []:
                raise OverlayError(f"{label} pastey node features changed unexpectedly")
            if left.get("dependencies", []) != [] or right.get("dependencies", []) != [] or left.get("deps", []) != [] or right.get("deps", []) != []:
                raise OverlayError(f"{label} pastey node has unexpected dependency edges")
            continue
        if left.get("features") != right.get("features"):
            raise OverlayError(f"{label} activated features changed")
        if left.get("dep_kinds") != right.get("dep_kinds"):
            raise OverlayError(f"{label} dependency kinds changed")
        left_deps = [mapping.get(value, value) for value in left.get("dependencies", [])]
        if left_deps != right.get("dependencies", []):
            raise OverlayError(f"{label} plain dependency IDs changed outside identity mapping")
        left_edges = [{**edge, "pkg": mapping.get(edge.get("pkg"), edge.get("pkg"))} for edge in left.get("deps", [])]
        if left_edges != right.get("deps", []):
            raise OverlayError(f"{label} typed dependency edges or aliases changed")
    if mapping.get(before.get("resolve", {}).get("root")) != after.get("resolve", {}).get("root"):
        raise OverlayError(f"{label} resolve root changed outside identity mapping")

def _parent_dependency_transform(before: Any, after: Any) -> bool:
    if not isinstance(before, list) or not isinstance(after, list) or len(after) != len(before):
        return False
    changed = 0
    for left, right in zip(before, after):
        if left == right:
            continue
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        l, r = dict(left), dict(right)
        if l.get("name") != "paste" or r.get("name") != "pastey":
            return False
        expected = copy.deepcopy(l)
        expected["name"] = "pastey"
        expected["req"] = "=0.2.3"
        expected["rename"] = "paste"
        if r != expected:
            return False
        changed += 1
    return changed == 1

def _validate_pastey_metadata(metadata: dict[str, Any], pastey_receipt: dict[str, Any]) -> None:
    packages = [p for p in metadata.get("packages", []) if isinstance(p, dict) and p.get("name") == "pastey" and p.get("version") == "0.2.3" and p.get("source") == REGISTRY]
    if len(packages) != 1:
        raise OverlayError("metadata does not contain one authenticated registry pastey package")
    package = packages[0]
    expected_manifest = pastey_receipt.get("manifest", {})
    expected_package = expected_manifest.get("package", {})
    expected_api = pastey_receipt.get("api", {}).get("version", {})
    if expected_package.get("build") is not False or expected_manifest.get("dependencies", {}) not in ({}, None) or expected_manifest.get("features", {}) not in ({}, None):
        raise OverlayError("authenticated pastey manifest has unexpected build/dependency/feature declarations")
    if package.get("license") != expected_package.get("license"):
        raise OverlayError("pastey metadata license differs from authenticated manifest")
    if package.get("edition") != expected_api.get("edition") or package.get("rust_version") != expected_api.get("rust_version"):
        raise OverlayError("pastey metadata edition/MSRV differs from authenticated registry record")
    if package.get("features") != expected_api.get("features") or package.get("dependencies") != []:
        raise OverlayError("pastey metadata has unexpected features or dependencies")
    targets = package.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        raise OverlayError("pastey metadata target inventory is not exactly one target")
    target = targets[0]
    if not isinstance(target, dict) or target.get("name") != expected_manifest.get("lib", {}).get("name") or "proc_macro" in target:
        raise OverlayError("pastey metadata target name differs from authenticated manifest")
    if target.get("kind") != ["proc-macro"] or target.get("crate_types") != ["proc-macro"]:
        raise OverlayError("pastey metadata target is not exactly the authenticated proc-macro")
    if target.get("src_path") is not None and not str(target["src_path"]).endswith("/pastey-0.2.3/src/lib.rs"):
        raise OverlayError("pastey metadata proc-macro source path is unexpected")


def _owned_vendor_path(value: Any, package: dict[str, Any], vendor_root: Path, relative: Path | str) -> bool:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        return False
    path = Path(value)
    name, version = package.get("name"), package.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        return False
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    expected = vendor_root / f"{name}-{version}" / relative_path
    if path != expected:
        return False
    if vendor_root.is_symlink() or path.is_symlink() or not path.is_file():
        return False
    try:
        relative_to_vendor = path.relative_to(vendor_root)
    except ValueError:
        return False
    current = vendor_root
    for component in relative_to_vendor.parts[:-1]:
        current = current / component
        if current.is_symlink():
            return False
    return True

def _materialize_patch_snippet(plan: dict[str, Any], projection: Path, vendor_root: Path) -> tuple[str, list[dict[str, str]]]:
    """Translate the accepted plan's vendor-relative paths into the actual sibling layout."""
    snippet = plan.get("isolated_patch_crates_io")
    proposals = plan.get("proposals")
    if not isinstance(snippet, str) or not isinstance(proposals, list) or len(proposals) != 16:
        raise OverlayError("authenticated patch plan is incomplete")
    if vendor_root != projection.parent / "vendor" or vendor_root.is_symlink():
        raise OverlayError("vendor materialization root is not the exact projection sibling")
    expected_lines = ["[patch.crates-io]\n"]
    mappings: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    for proposal in proposals:
        identity = proposal.get("identity") if isinstance(proposal, dict) else None
        if not isinstance(identity, str) or identity.count("@") != 1:
            raise OverlayError("patch plan identity is malformed")
        name, version = identity.split("@", 1)
        key = re.sub(r"[^A-Za-z0-9_]", "_", f"{name}_v{version.replace('.', '_')}")
        if key in seen_keys:
            raise OverlayError("patch plan key is duplicated")
        seen_keys.add(key)
        plan_path = f"vendor/{name}-{version}"
        expected_lines.append(f'"{key}" = {{ package = "{name}", path = "{plan_path}" }}\n')
        mappings.append({"identity": identity, "plan_path": plan_path, "materialized_path": f"../vendor/{name}-{version}"})
    expected = "".join(expected_lines)
    if snippet != expected:
        raise OverlayError("authenticated patch plan snippet differs from exact 16-parent bindings")
    materialized = "".join(line.replace(f'path = "{row["plan_path"]}"', f'path = "{row["materialized_path"]}"') for line, row in zip(expected_lines[1:], mappings))
    materialized = expected_lines[0] + materialized
    parsed = tomllib.loads(materialized)
    if set(parsed) != {"patch"} or set(parsed["patch"]) != {"crates-io"} or len(parsed["patch"]["crates-io"]) != 16:
        raise OverlayError("materialized patch snippet is not exactly 16 crates-io entries")
    return materialized, mappings

def _write_metadata(path: Path, status: str, value: dict[str, Any], target: str | None) -> None:
    _write(path, {"status": status, "source_only": True, "target": target, "platform_filter": target, "metadata": value, "approval": None, "upload": "NO_UPLOAD", "model_execution": "NOT_AUTHORIZED", "cpu_execution_profile": "NOT_APPROVED" if target is None else "OBSERVED_ONLY"})

def _load_inputs(planner: Any) -> tuple[dict[str, Any], dict[str, Any], bytes, dict[str, Any]]:
    parent_path = ROOT / "docs/handoff/kyutai-candle-paste-parents-source-2026-10-15.json"
    pastey_path = ROOT / "docs/handoff/kyutai-candle-pastey-source-2026-10-15.json"
    candidate_path = ROOT / "docs/handoff/kyutai-candle-candidate-2026-10-15.lock"
    parent = _receipt_file(parent_path, PARENT_RECEIPT_BYTES, PARENT_RECEIPT_SHA256)
    pastey = _receipt_file(pastey_path, PASTEY_RECEIPT_BYTES, PASTEY_RECEIPT_SHA256)
    candidate = _fixed(candidate_path, CANDIDATE_BYTES, CANDIDATE_SHA256)
    plan = planner.build_plan(parent, pastey)
    plan_bytes = json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False).encode()
    if len(plan_bytes) != PLAN_BYTES or _sha(plan_bytes) != PLAN_SHA256:
        raise OverlayError("authenticated planner output differs from accepted 16-parent plan")
    if len(plan.get("proposals", [])) != 16 or plan.get("parent_counts", {}).get("total") != 16:
        raise OverlayError("planner did not produce exactly 16 parent proposals")
    return parent, pastey, candidate, plan

def _fresh(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise OverlayError(f"output/workspace must be fresh: {path}")
    path.mkdir(parents=True)

def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved, right_resolved = left.resolve(), right.resolve()
    return left_resolved == right_resolved or left_resolved in right_resolved.parents or right_resolved in left_resolved.parents

def apply_overlay(packet_root: Path, output_root: Path, workspace_root: Path, *, deadline_seconds: float = MAX_DEADLINE, platform_name: str | None = None, archive_fetcher: Callable[..., bytes] | None = None) -> dict[str, Any]:
    if (platform_name or platform.system()) != "Linux" or os.environ.get("VOKRA_CANDLE_REMOTE_LINUX") != "1":
        raise OverlayError("Linux production marker is required")
    if not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= MAX_DEADLINE:
        raise OverlayError("deadline must be in (0,480]")
    outer_end = time.monotonic() + deadline_seconds
    recovery_reserve = min(RECOVERY_RESERVE_SECONDS, deadline_seconds / 4)
    end = outer_end - recovery_reserve
    if any(_paths_overlap(left, right) for left, right in ((packet_root, output_root), (packet_root, workspace_root), (output_root, workspace_root))):
        raise OverlayError("roots must be distinct; roots must not overlap; non-overlapping roots required")
    _remaining(end)
    _fresh(output_root)
    _remaining(end)
    _fresh(workspace_root)
    _remaining(end)
    receipt: dict[str, Any] = {"status": "SOURCE_APPLICATION_BLOCKED", "source_only": True, "approval": None, "upload": "NO_UPLOAD", "model_execution": "NOT_AUTHORIZED", "commands": [], "archives": [], "module_hashes": {"collector": COLLECTOR_SHA256, "pin": PIN_SHA256, "planner": PLANNER_SHA256}}
    receipt_path = output_root / "application_receipt.json"
    _remaining(end)
    _write(receipt_path, receipt)
    _remaining(end)
    collector: Any | None = None
    try:
        _remaining(end)
        collector = _module("candle_overlay_collector", ROOT / "tools/parity/kyutai_candle_reference/collect_metadata.py", COLLECTOR_SHA256)
        _remaining(end)
        pin = _bounded_stage(collector, end, _module, "candle_overlay_pin", ROOT / "tools/parity/kyutai_candle_reference/pin_remediation_diagnostic.py", PIN_SHA256)
        planner = _bounded_stage(collector, end, _module, "candle_overlay_planner", ROOT / "tools/parity/kyutai_candle_reference/prepare_paste_overlay.py", PLANNER_SHA256)
        parent, pastey, candidate_bytes, plan = _bounded_stage(collector, end, _load_inputs, planner)
        receipt["source_inputs"] = {"parent_receipt_sha256": PARENT_RECEIPT_SHA256, "pastey_receipt_sha256": PASTEY_RECEIPT_SHA256, "candidate_lock_sha256": CANDIDATE_SHA256}
        capture_budget = _remaining(end) - 0.25
        if capture_budget <= 0:
            raise OverlayError("shared deadline expired before source capture")
        collector.capture(packet_root, deadline_seconds=min(MAX_DEADLINE, capture_budget), platform_name="Linux")
        _remaining(end)
        with collector._SourceIODeadline(end):
            packet = pin._authenticate_packet(packet_root, collector)
            projection = pin._rebuild_projection(packet_root, workspace_root, collector, candidate_bytes)
        source_files = packet.get("receipt", {}).get("files")
        if not isinstance(source_files, dict) or set(source_files) != set(collector.SOURCE_FILES):
            raise OverlayError("authenticated Moshi source file inventory is not exact")
        source_manifest = _bounded_stage(collector, end, _read, packet_root / "source" / "rust" / "Cargo.toml", collector.MAX_FILE_BYTES)
        try:
            projected_manifest = collector.replace_workspace_members(source_manifest.decode("utf-8")).encode("utf-8")
        except Exception as exc:
            raise OverlayError("captured root Cargo.toml cannot form the exact moshi-core projection") from exc
        source_receipt_bytes = _bounded_stage(collector, end, _read, packet_root / "source_receipt.json", MAX_JSON)
        _bounded_stage(collector, end, (output_root / "source_receipt.json").write_bytes, source_receipt_bytes)
        _bounded_stage(collector, end, shutil.copy2, projection / "Cargo.lock", output_root / "candidate.lock")
        receipt["source_proof_pre"] = _bounded_stage(collector, end, _source_projection_proof,
            collector,
            packet_root,
            projection,
            packet,
            expected_root_manifest=projected_manifest,
            expected_lock=candidate_bytes,
            phase="PRE_OVERLAY",
        )
        cargo = shutil.which("cargo")
        if not cargo or Path(cargo).name != "cargo":
            raise OverlayError("cargo executable is unavailable")
        env = collector.clean_cargo_environment(workspace_root / "cargo-home")
        cargo_home = workspace_root / "cargo-home"
        _bounded_stage(collector, end, cargo_home.mkdir, parents=True, exist_ok=True)
        collector._toolchain_probe(cargo, cwd=projection, env=env, deadline=min(end, time.monotonic() + 10))
        base_lock = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
        baseline_values: dict[str, dict[str, Any]] = {}
        for args, out, status in ((CPU_LOCKED_ARGS, "baseline_cpu_metadata.json", "OVERLAY_BASELINE_CPU_UNREVIEWED"), (WHOLE_ARGS, "baseline_whole_metadata.json", "OVERLAY_BASELINE_WHOLE_UNREVIEWED")):
            before = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
            result = _run(collector, cargo, args, projection, env, end)
            receipt["commands"].append(_record_command(pin, [cargo, *args], result, len(receipt["commands"])))
            after = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
            pin._assert_lock_unchanged(before, after, label=f"baseline {'whole' if args == WHOLE_ARGS else 'cpu'}")
            if result.get("returncode") != 0:
                raise OverlayError(f"baseline {'whole' if args == WHOLE_ARGS else 'cpu'} metadata failed")
            value = json.loads(result.get("stdout", "")); _metadata(value, out); baseline_values[out] = value
            _bounded_stage(collector, end, _write_metadata, output_root / out, status, value, None if args == WHOLE_ARGS else TARGET)
        candidate_doc = _bounded_stage(collector, end, collector.read_toml, projection / "Cargo.lock")
        candidate_packages = collector.lock_packages(candidate_doc)
        receipt["graph_validation_baseline"] = _bounded_stage(collector, end, _validate_graphs, collector, baseline_values["baseline_cpu_metadata.json"], baseline_values["baseline_whole_metadata.json"], candidate_packages, "baseline")
        vendor = workspace_root / "vendor"
        _bounded_stage(collector, end, vendor.mkdir)
        parent_rows = parent.get("archives") or []
        if len(plan.get("proposals", [])) != 16:
            raise OverlayError("exact parent proposal count changed")
        total_c = total_e = 0
        archive_specs: list[dict[str, Any]] = []
        for proposal in plan["proposals"]:
            _remaining(end)
            name, version = proposal["identity"].split("@", 1)
            row = next((r for r in parent_rows if r.get("name") == name and r.get("version") == version), None)
            if not isinstance(row, dict): raise OverlayError(f"missing parent receipt row: {name} {version}")
            limit = MAX_METAL_ARCHIVE if name == "metal" else MAX_ARCHIVE
            compressed = row.get("archive_bytes")
            if not isinstance(compressed, int) or compressed < 0 or compressed > limit:
                raise OverlayError(f"archive compressed bound exceeded: {name} {version}")
            with collector._SourceIODeadline(end):
                raw = _fetch("https://static.crates.io/crates/{0}/{0}-{1}.crate".format(name, version), compressed, row["archive_sha256"], end, archive_fetcher)
            total_c += len(raw)
            expected = row.get("expanded_bytes", 0); total_e += expected
            if total_c > MAX_ARCHIVE_TOTAL or total_e > MAX_EXPANDED_TOTAL: raise OverlayError("archive total cap exceeded")
            dest = vendor / f"{name}-{version}"; _bounded_stage(collector, end, dest.mkdir, parents=True)
            with collector._SourceIODeadline(end):
                proof = _extract(raw, row, dest, expected_root=f"{name}-{version}", deadline=end)
            cargo_toml = dest / "Cargo.toml"
            if not cargo_toml.is_file():
                raise OverlayError(f"extracted parent manifest missing: {name} {version}")
            proposed = proposal.get("proposed_manifest", {}).get("text")
            if not isinstance(proposed, str) or _sha(proposed.encode()) != proposal["proposed_manifest"].get("sha256"):
                raise OverlayError(f"planner proposal manifest hash mismatch: {name} {version}")
            _bounded_stage(collector, end, cargo_toml.write_text, proposed)
            archive_specs.append({"dest": dest, "receipt": row, "proposed_manifest": proposed})
            receipt["archives"].append({"name": name, "version": version, "bytes": len(raw), "sha256": _sha(raw), "members": proof})
        paste_row = pastey.get("archive") or pastey
        expected_pastey_url = "https://static.crates.io/crates/pastey/pastey-0.2.3.crate"
        if paste_row.get("url") != expected_pastey_url:
            raise OverlayError("pastey archive URL is not the fixed official static URL")
        paste_compressed = paste_row.get("compressed_bytes")
        paste_expanded = paste_row.get("expanded_bytes")
        if not isinstance(paste_compressed, int) or not isinstance(paste_expanded, int) or paste_compressed < 0 or paste_compressed > MAX_ARCHIVE or paste_expanded < 0:
            raise OverlayError("pastey archive size evidence is malformed")
        with collector._SourceIODeadline(end):
            raw = _fetch(paste_row["url"], paste_compressed, paste_row["compressed_sha256"], end, archive_fetcher)
        total_c += len(raw)
        total_e += paste_expanded
        if total_c > MAX_ARCHIVE_TOTAL or total_e > MAX_EXPANDED_TOTAL:
            raise OverlayError("archive total cap exceeded including pastey")
        pastey_extract_receipt = dict(pastey)
        pastey_extract_receipt["expanded_bytes"] = paste_row["expanded_bytes"]
        with collector._SourceIODeadline(end):
            _extract(raw, pastey_extract_receipt, None, pastey=True, expected_root="pastey-0.2.3", deadline=end)
        receipt["archives"].append({"name": "pastey", "version": "0.2.3", "bytes": len(raw), "sha256": _sha(raw), "registry_identity": True})
        manifest = projection / "Cargo.toml"
        original = _bounded_stage(collector, end, manifest.read_text)
        snippet, patch_mapping = _materialize_patch_snippet(plan, manifest.parent, vendor)
        original_parsed = tomllib.loads(original)
        if "patch" in original_parsed:
            raise OverlayError("root manifest already contains a patch table")
        parsed = tomllib.loads(original + "\n" + snippet)
        patch = parsed.get("patch")
        parsed_without_patch = {key: value for key, value in parsed.items() if key != "patch"}
        if parsed_without_patch != original_parsed or not isinstance(patch, dict) or set(patch) != {"crates-io"} or not isinstance(patch["crates-io"], dict) or len(patch["crates-io"]) != 16:
            raise OverlayError("root manifest patch table is not exact")
        appended = original + ("" if original.endswith("\n") else "\n") + "\n" + snippet
        _bounded_stage(collector, end, manifest.write_text, appended)
        final_manifest_bytes = _bounded_stage(collector, end, manifest.read_bytes)
        receipt["manifest"] = {"before_sha256": _sha(original.encode()), "after_sha256": _sha(final_manifest_bytes), "patch_entries": 16, "plan_patch_snippet": plan["isolated_patch_crates_io"], "applied_patch_snippet": snippet, "path_mapping": patch_mapping}
        with collector._SourceIODeadline(end):
            receipt["member_premetadata_checks"] = [_verify_tree(spec["dest"], spec["receipt"], proposed_manifest=spec["proposed_manifest"]) for spec in archive_specs]
        pre_members = sum(row["members_checked"] for row in receipt["member_premetadata_checks"])
        if pre_members != 308 or sum(row["non_manifest_members"] for row in receipt["member_premetadata_checks"]) != 292:
            raise OverlayError("pre-metadata source proof is not exactly 308/292 members")
        result = _run(collector, cargo, CPU_ARGS, projection, env, end)
        receipt["commands"].append(_record_command(pin, [cargo, *CPU_ARGS], result, len(receipt["commands"])))
        if result.get("returncode") != 0: raise OverlayError("overlay unlocked CPU metadata failed")
        unlocked_value = json.loads(result.get("stdout", "")); _metadata(unlocked_value, "overlay_cpu_metadata.json")
        _bounded_stage(collector, end, _write_metadata, output_root / "overlay_cpu_metadata.json", "OVERLAY_CPU_METADATA_UNREVIEWED", unlocked_value, TARGET)
        after_unlocked = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
        receipt["overlay_unlocked_lock"] = after_unlocked
        overlay_values: dict[str, dict[str, Any]] = {"overlay_cpu_metadata.json": unlocked_value}
        for args, out, status in ((CPU_LOCKED_ARGS, None, None), (WHOLE_ARGS, "overlay_whole_metadata.json", "OVERLAY_WHOLE_METADATA_UNREVIEWED")):
            before = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
            result = _run(collector, cargo, args, projection, env, end)
            receipt["commands"].append(_record_command(pin, [cargo, *args], result, len(receipt["commands"])))
            if result.get("returncode") != 0: raise OverlayError(f"overlay metadata failed: {' '.join(args)}")
            after = _bounded_stage(collector, end, pin._lock_fingerprint, collector, projection / "Cargo.lock")
            pin._assert_lock_unchanged(before, after, label=f"overlay {'whole' if args == WHOLE_ARGS else 'locked CPU'}")
            if out:
                value = json.loads(result.get("stdout", "")); _metadata(value, out); overlay_values[out] = value; _bounded_stage(collector, end, _write_metadata, output_root / out, status, value, None if args == WHOLE_ARGS else TARGET)
        final = projection / "Cargo.lock"
        final_bytes = _bounded_stage(collector, end, _read, final, MAX_JSON)
        _bounded_stage(collector, end, (output_root / "final.lock").write_bytes, final_bytes)
        final_doc = _bounded_stage(collector, end, collector.read_toml, final)
        final_packages = collector.lock_packages(final_doc)
        delta = _bounded_stage(collector, end, _lock_delta, collector, candidate_packages, final_packages, candidate_doc, final_doc, plan["proposals"])
        _bounded_stage(collector, end, _assert_graph_delta, baseline_values["baseline_cpu_metadata.json"], overlay_values["overlay_cpu_metadata.json"], plan["proposals"], "CPU", vendor)
        _bounded_stage(collector, end, _assert_graph_delta, baseline_values["baseline_whole_metadata.json"], overlay_values["overlay_whole_metadata.json"], plan["proposals"], "whole", vendor)
        receipt["graph_validation_overlay"] = _bounded_stage(collector, end, _validate_graphs,
            collector,
            overlay_values["overlay_cpu_metadata.json"],
            overlay_values["overlay_whole_metadata.json"],
            final_packages,
            "overlay",
            before_cpu=baseline_values["baseline_cpu_metadata.json"],
            before_whole=baseline_values["baseline_whole_metadata.json"],
            proposals=plan["proposals"],
            vendor_root=vendor,
        )
        _bounded_stage(collector, end, _validate_pastey_metadata, overlay_values["overlay_whole_metadata.json"], pastey)
        root_manifest_final = _bounded_stage(collector, end, manifest.read_bytes)
        if root_manifest_final != appended.encode():
            raise OverlayError("root Moshi manifest changed after exact overlay append")
        receipt["root_manifest_final_sha256"] = _sha(root_manifest_final)
        with collector._SourceIODeadline(end):
            receipt["member_postchecks"] = [_verify_tree(spec["dest"], spec["receipt"], proposed_manifest=spec["proposed_manifest"]) for spec in archive_specs]
        post_members = sum(row["members_checked"] for row in receipt["member_postchecks"])
        if post_members != 308 or sum(row["non_manifest_members"] for row in receipt["member_postchecks"]) != 292:
            raise OverlayError("post-metadata source proof is not exactly 308/292 members")
        if post_members != sum(len(_inventory(spec["receipt"])) for spec in archive_specs):
            raise OverlayError("post-application member verification was incomplete")
        receipt["source_proof_post"] = _bounded_stage(collector, end, _source_projection_proof,
            collector,
            packet_root,
            projection,
            packet,
            expected_root_manifest=appended.encode(),
            expected_lock=final_bytes,
            phase="POST_OVERLAY",
        )
        _remaining(end)
        receipt.update({"status": "SOURCE_APPLICATION_REVIEW_PENDING", "candidate_lock": {"bytes": len(candidate_bytes), "sha256": _sha(candidate_bytes)}, "final_lock": {"bytes": len(final_bytes), "sha256": _sha(final_bytes)}, "lock_delta": delta, "archive_totals": {"compressed_bytes": total_c, "expanded_bytes": total_e}, "paste_status": "REGISTRY_PASTEY_IDENTITY_RETAINED_NO_SECURITY_OR_LEGAL_APPROVAL"})
        _remaining(end)
        _bounded_stage(collector, end, _rewrite, receipt_path, receipt)
        _remaining(end)
        return receipt
    except Exception as exc:
        receipt["status"] = "SOURCE_APPLICATION_BLOCKED"
        receipt["error"] = str(exc)
        try:
            if collector is None:
                _remaining(outer_end)
                _rewrite_recovery(receipt_path, receipt)
                _remaining(outer_end)
            else:
                _bounded_stage(collector, outer_end, _rewrite_recovery, receipt_path, receipt)
        except Exception:
            pass
        raise

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="source-only paste overlay metadata diagnostic")
    parser.add_argument("packet_root", type=Path); parser.add_argument("output_root", type=Path); parser.add_argument("workspace_root", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=MAX_DEADLINE)
    args = parser.parse_args(argv)
    try:
        result = apply_overlay(args.packet_root, args.output_root, args.workspace_root, deadline_seconds=args.deadline_seconds)
    except (OverlayError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"SOURCE_APPLICATION_BLOCKED: {exc}")
        return 2
    print(result["status"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
