#!/usr/bin/env -S uv run --no-project --no-sync --python 3.12 python
"""Build a source-only Python import graph for the Kyutai reference path.

This module deliberately does not import the authenticated source, resolve a
distribution name, or execute an import.  It authenticates source bytes from
the retained flat source receipt, parses them with :mod:`ast`, and emits an
evidence-only graph.  A missing root or source role is a blocker, not a
permission to guess a package or dependency version.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import types
from typing import Any, Iterable


MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024
MAX_FILES = 512
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_IMPORTS_PER_MODULE = 512
MAX_GRAPH_MODULES = 512
MAX_GRAPH_EDGES = 16 * 1024
EXPECTED_VOKRA_HEAD = "8bcf661f8209830fc110492dce02c10dbf3ec88d"
VOKRA_REPOSITORY = "https://github.com/ayutaz/vokra.git"
VOKRA_SOURCE_PATHS = (
    "tools/parity/kyutai_stt_streaming_reference/pcm_dump.py",
    "tools/parity/kyutai_stt_streaming_reference/dump.py",
    "tools/parity/kyutai_stt_streaming_reference/contract.py",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

SCHEMA = "vokra-kyutai-python-import-graph-v1"
GRAPH_STATUS = "SOURCE_DEPENDENCY_GRAPH_ONLY"
EXECUTION_STATUS = "NOT_EXECUTION_CLOSURE"
REVIEW_STATUS = "BLOCKED_REVIEW"
PUBLICATION_STATUS = "NO_UPLOAD"

# These roots describe the actual producer route.  DSM evaluator scripts are
# source-contract evidence only and are never treated as runtime roots.
DEFAULT_ROOTS = (
    ("pcm_dump", "tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"),
    ("dump", "tools/parity/kyutai_stt_streaming_reference/dump.py"),
    ("contract", "tools/parity/kyutai_stt_streaming_reference/contract.py"),
)

# These files are fixed upstream source roles from contract.py.  They are
# inspected as authenticated source roles, not silently promoted to an
# execution closure when the producer reaches Moshi through a dynamic loader.
UPSTREAM_ROLE_ROOTS = (
    ("moshi.models.lm", "moshi/moshi/models/lm.py"),
    ("moshi.models.compression", "moshi/moshi/models/compression.py"),
    ("moshi.modules.transformer", "moshi/moshi/modules/transformer.py"),
    ("moshi.modules.streaming", "moshi/moshi/modules/streaming.py"),
)

STDLIB_TOP_LEVEL = frozenset(
    {
        "__future__", "abc", "argparse", "array", "asyncio", "base64", "bisect",
        "calendar", "collections", "contextlib", "copy", "dataclasses", "datetime",
        "enum", "functools", "gc", "glob", "gzip", "hashlib", "heapq", "io", "itertools",
        "json", "logging", "math", "numbers", "operator", "os", "pathlib", "platform",
        "queue", "random", "re", "secrets", "shutil", "socket", "struct", "subprocess",
        "sys", "tempfile", "textwrap", "threading", "time", "traceback", "types", "typing",
        "importlib",
        "unicodedata", "urllib", "uuid", "warnings", "weakref", "zipfile",
    }
)


class GraphError(ValueError):
    """A malformed or unauthenticated graph input."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraphError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(body: bytes, label: str) -> Any:
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError, GraphError) as exc:
        raise GraphError(f"{label} is not unique-key UTF-8 JSON") from exc


def _safe_regular(path: Path, label: str, *, max_bytes: int) -> None:
    if not path.is_absolute() or path.is_symlink() or any(part in {".", ".."} for part in path.parts):
        raise GraphError(f"{label} must be an absolute non-symlink path")
    if not path.is_file():
        raise GraphError(f"{label} is not a regular file")
    for ancestor in path.parents:
        if ancestor.is_symlink() and ancestor not in {Path("/private"), Path("/var"), Path("/tmp")}:
            raise GraphError(f"{label} has symlink ancestry")
    if path.stat().st_size > max_bytes:
        raise GraphError(f"{label} exceeds the bounded size")


def _safe_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or any(part in {".", ".."} for part in path.parts):
        raise GraphError(f"{label} must be an absolute non-symlink directory")
    if not path.is_dir():
        raise GraphError(f"{label} is not a directory")
    for ancestor in path.parents:
        if ancestor.is_symlink() and ancestor not in {Path("/private"), Path("/var"), Path("/tmp")}:
            raise GraphError(f"{label} has symlink ancestry")


def _read_once(path: Path, label: str, limit: int) -> bytes:
    _safe_regular(path, label, max_bytes=limit)
    before = path.stat()
    body = bytearray()
    with path.open("rb") as stream:
        while len(body) <= limit:
            block = stream.read(min(1 << 20, limit + 1 - len(body)))
            if not block:
                break
            body.extend(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise GraphError(f"{label} changed while reading")
    if len(body) > limit:
        raise GraphError(f"{label} exceeds the bounded size")
    return bytes(body)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _git_blob_sha1(body: bytes) -> str:
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


def _contract_module(path: Path | None = None, body: bytes | None = None) -> Any:
    """Execute only the already-authenticated stdlib validator bytes."""
    if path is None:
        raise GraphError("source contract validator path is required")
    if body is None:
        body = _read_once(path, "source contract validator", MAX_FILE_BYTES)
    try:
        code = compile(body, str(path), "exec")
        module = types.ModuleType("_vokra_authenticated_contract")
        module.__file__ = str(path)
        module.__package__ = ""
        exec(code, module.__dict__)
        return module
    except GraphError:
        raise
    except (ImportError, OSError, AttributeError, SyntaxError, TypeError) as exc:
        raise GraphError("source contract validator could not be loaded") from exc


def _authenticate_upstream_packet(
    packet: Path,
    *,
    trusted_contract: dict[str, Any] | None = None,
    vokra_root: Path | None = None,
) -> dict[str, Any]:
    """Reuse the fixed PCM packet validator; never accept caller-supplied pins."""
    if trusted_contract is not None:
        expected_file = (vokra_root / VOKRA_SOURCE_PATHS[2]) if vokra_root is not None else None
        if expected_file is None:
            raise GraphError("source contract validator path is unavailable")
        current_body = _read_once(expected_file, "selected Vokra contract", MAX_FILE_BYTES)
        if current_body != trusted_contract.get("body"):
            raise GraphError("selected Vokra contract bytes differ from authenticated Git source")
    if trusted_contract is not None:
        expected_file = (vokra_root / VOKRA_SOURCE_PATHS[2]) if vokra_root is not None else None
        validator = _contract_module(expected_file, trusted_contract["body"])
        module_file = getattr(validator, "__file__", None)
        if not isinstance(module_file, str) or Path(module_file).resolve() != expected_file.resolve():
            raise GraphError("source contract validator is not the authenticated Vokra file")
    else:
        validator = _contract_module()
    try:
        receipt = validator.require_pcm_source_packet(packet)
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError) as exc:
        raise GraphError(f"PCM source packet authentication failed: {exc}") from exc
    if receipt.get("manifest_sha256") != validator.PCM_SOURCE_PACKET_MANIFEST_SHA256:
        raise GraphError("PCM source packet validator returned an unexpected manifest")
    return {
        "status": "AUTHENTICATED_FIXED_PCM_SOURCE_PACKET",
        "manifest_sha256": receipt["manifest_sha256"],
        "dsm_tree_sha": receipt["dsm_tree_sha"],
        "moshi_tree_sha": receipt["moshi_tree_sha"],
        "role_contract": "contract.require_pcm_source_packet",
    }


def _git_capture(root: Path, args: list[str], label: str, *, text: bool = True) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GraphError(f"{label} failed") from exc
    if completed.stderr:
        raise GraphError(f"{label} emitted stderr")
    return completed.stdout


def _authenticate_vokra_sources(root: Path, expected_head: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Bind first-party source bytes to one clean, explicitly selected Git commit."""
    _safe_directory(root, "Vokra source checkout")
    top = str(_git_capture(root, ["rev-parse", "--show-toplevel"], "Vokra repository root")).strip()
    if Path(top).resolve() != root.resolve():
        raise GraphError("Vokra source checkout root mismatch")
    head = str(_git_capture(root, ["rev-parse", "HEAD"], "Vokra HEAD")).strip()
    if head != expected_head or not HEX40.fullmatch(head):
        raise GraphError("Vokra source HEAD mismatch")
    origin = str(_git_capture(root, ["remote", "get-url", "origin"], "Vokra origin")).strip()
    if origin not in {VOKRA_REPOSITORY, VOKRA_REPOSITORY.removesuffix(".git")}:
        raise GraphError("Vokra source origin mismatch")
    changed = str(_git_capture(
        root,
        ["status", "--porcelain", "--untracked-files=all", "--", *VOKRA_SOURCE_PATHS],
        "Vokra selected-source status",
    ))
    if changed:
        raise GraphError("Vokra selected source paths are dirty")
    rows: dict[str, dict[str, Any]] = {}
    for source_path in VOKRA_SOURCE_PATHS:
        object_name = f"{head}:{source_path}"
        blob = str(_git_capture(root, ["rev-parse", object_name], f"Vokra blob {source_path}")).strip()
        kind = str(_git_capture(root, ["cat-file", "-t", object_name], f"Vokra object {source_path}")).strip()
        size_text = str(_git_capture(root, ["cat-file", "-s", object_name], f"Vokra size {source_path}")).strip()
        if not HEX40.fullmatch(blob) or kind != "blob":
            raise GraphError(f"Vokra source object is not a blob: {source_path}")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise GraphError(f"Vokra source size is malformed: {source_path}") from exc
        if not 0 <= size <= MAX_FILE_BYTES:
            raise GraphError(f"Vokra source exceeds bound: {source_path}")
        body = _git_capture(root, ["show", object_name], f"Vokra source {source_path}", text=False)
        if not isinstance(body, bytes) or len(body) != size:
            raise GraphError(f"Vokra source changed while reading: {source_path}")
        if _git_blob_sha1(body) != blob:
            raise GraphError(f"Vokra Git blob binding mismatch: {source_path}")
        module = _module_from_path(source_path)
        if module is None:
            raise GraphError(f"Vokra source is not a Python module: {source_path}")
        rows[source_path] = {
            "path": source_path,
            "packet_name": None,
            "bytes": size,
            "sha256": _sha256(body),
            "git_blob_sha1": blob,
            "revision": head,
            "repository": VOKRA_REPOSITORY,
            "body": body,
            "is_package": _is_package_path(source_path),
            "source_kind": "vokra-clean-git-source",
            "module": module,
        }
    return {
        "status": "AUTHENTICATED_VOKRA_GIT_SOURCE",
        "repository": VOKRA_REPOSITORY,
        "origin": origin,
        "revision": head,
        "paths": [
            {"path": path, "git_blob_sha1": rows[path]["git_blob_sha1"], "sha256": rows[path]["sha256"], "bytes": rows[path]["bytes"]}
            for path in VOKRA_SOURCE_PATHS
        ],
    }, rows


def _safe_rel(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise GraphError(f"{label} is not a safe relative POSIX path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise GraphError(f"{label} is not a safe relative POSIX path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in {"", ".", ".."} for part in p.parts):
        raise GraphError(f"{label} is not a safe relative POSIX path")
    return value


def _module_from_path(path: str) -> str | None:
    """Map authenticated source paths to import names without guessing dists."""
    local_prefix = "tools/parity/kyutai_stt_streaming_reference/"
    if path.startswith(local_prefix) and path.endswith(".py"):
        stem = path[len(local_prefix):-3]
        if stem.endswith("/__init__"):
            stem = stem[:-9]
        return stem.rsplit("/", 1)[-1]
    if path.endswith("/__init__.py"):
        stem = path[:-12]
    elif path.endswith(".py"):
        stem = path[:-3]
    else:
        return None
    parts = stem.split("/")
    if parts[:2] == ["moshi", "moshi"]:
        return ".".join(parts[1:])
    if parts[:1] == ["moshi_mlx"]:
        return ".".join(parts)
    return None


def _is_package_path(path: str) -> bool:
    return path.endswith("/__init__.py")


def _top_name(name: str) -> str:
    return name.split(".", 1)[0]


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.edges: list[dict[str, Any]] = []
        self._conditions: list[str] = []
        self._try_depth = 0

    def _add(self, name: str, node: ast.AST, kind: str, *, level: int = 0) -> None:
        if len(self.edges) >= MAX_IMPORTS_PER_MODULE:
            raise GraphError(f"{self.module} exceeds import edge bound")
        condition = "+".join(self._conditions) if self._conditions else "unconditional"
        self.edges.append({
            "name": name,
            "kind": kind,
            "line": int(getattr(node, "lineno", 0)),
            "column": int(getattr(node, "col_offset", 0)),
            "condition": condition,
            "relative_level": level,
        })

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name, node, "import")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        # Keep the relative level separate from the module name.  Encoding
        # dots into ``name`` would make ``from ..modules`` resolve to an
        # invalid literal instead of the authenticated parent package.
        if base or not node.level:
            self._add(base, node, "from-import", level=node.level)
        for alias in node.names:
            if alias.name != "*":
                self._add(
                    f"{base}.{alias.name}" if base else alias.name,
                    node,
                    "from-import-member",
                    level=node.level,
                )

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        dynamic = False
        if isinstance(target, ast.Name) and target.id in {"__import__", "import_module"}:
            dynamic = True
        if isinstance(target, ast.Attribute) and target.attr == "import_module":
            dynamic = True
        if dynamic:
            self._add("<dynamic-import>", node, "dynamic-import")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        marker = "TYPE_CHECKING" if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING" else "conditional"
        self._conditions.append(marker)
        self.visit(node.test)
        for child in node.body:
            self.visit(child)
        self._conditions.pop()
        self._conditions.append(f"{marker}-else")
        for child in node.orelse:
            self.visit(child)
        self._conditions.pop()

    def visit_Try(self, node: ast.Try) -> None:
        marker = "optional-import" if any(
            isinstance(handler.type, ast.Name) and handler.type.id in {"ImportError", "ModuleNotFoundError"}
            for handler in node.handlers
        ) else "try-block"
        self._conditions.append(marker)
        for child in node.body:
            self.visit(child)
        self._conditions.pop()
        self._conditions.append(f"{marker}-handler")
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            for child in handler.body:
                self.visit(child)
        self._conditions.pop()
        self._conditions.append("try-else")
        for child in node.orelse:
            self.visit(child)
        self._conditions.pop()
        self._conditions.append("try-finally")
        for child in node.finalbody:
            self.visit(child)
        self._conditions.pop()


def _resolve_relative(module: str, name: str, level: int, *, source_is_package: bool) -> str:
    package = module.split(".") if source_is_package else module.split(".")[:-1]
    if level <= 0 or level > len(package):
        return "<invalid-relative-import>"
    base = package[: len(package) - level + 1] if level else package
    if name:
        base.append(name)
    return ".".join(base)


def _source_receipt(packet: Path, manifest: dict[str, Any], manifest_bytes: bytes) -> dict[str, Any]:
    receipt_path = packet / "source-receipt.json"
    if not receipt_path.exists():
        return {"status": "MISSING_SOURCE_RECEIPT", "path_rows": [], "missing": ["source-receipt.json"]}
    receipt_bytes = _read_once(receipt_path, "source-receipt.json", MAX_RECEIPT_BYTES)
    receipt = _json(receipt_bytes, "source-receipt.json")
    if not isinstance(receipt, dict):
        raise GraphError("source-receipt.json must be an object")
    manifest_by_name = {entry["name"]: entry for entry in manifest["files"]}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    repos = receipt.get("repositories")
    if not isinstance(repos, list) or not repos:
        return {"status": "MISSING_REPOSITORY_ROWS", "path_rows": [], "missing": ["repositories"]}
    for repo in repos:
        if not isinstance(repo, dict):
            raise GraphError("source-receipt repository row is malformed")
        commit_name = repo.get("commit_receipt_name")
        commit_sha = repo.get("commit_sha")
        tree_name = repo.get("tree_receipt_name")
        tree_sha = repo.get("tree_sha")
        if not all(isinstance(x, str) for x in (commit_name, commit_sha, tree_name, tree_sha)):
            raise GraphError("source-receipt repository identity is malformed")
        if not HEX40.fullmatch(commit_sha) or not HEX40.fullmatch(tree_sha):
            raise GraphError("source-receipt repository digest is malformed")
        if commit_name not in manifest_by_name or tree_name not in manifest_by_name:
            missing.extend([x for x in (commit_name, tree_name) if x not in manifest_by_name])
            continue
        commit_body = _read_once(packet / commit_name, commit_name, MAX_FILE_BYTES)
        tree_body = _read_once(packet / tree_name, tree_name, MAX_FILE_BYTES)
        for name, body in ((commit_name, commit_body), (tree_name, tree_body)):
            entry = manifest_by_name[name]
            if entry["size"] != len(body) or entry["sha256"] != _sha256(body):
                raise GraphError(f"manifest binding mismatch: {name}")
        commit_doc = _json(commit_body, commit_name)
        tree_doc = _json(tree_body, tree_name)
        if commit_doc.get("sha") != commit_sha or commit_doc.get("commit", {}).get("tree", {}).get("sha") != tree_sha:
            raise GraphError(f"repository identity mismatch: {commit_name}")
        if tree_doc.get("sha") != tree_sha or tree_doc.get("truncated") is not False:
            raise GraphError(f"source tree is not complete: {tree_name}")
        tree_rows = tree_doc.get("tree")
        if not isinstance(tree_rows, list) or len(tree_rows) > MAX_FILES * 32:
            raise GraphError(f"source tree exceeds bound: {tree_name}")
        tree_by_path: dict[str, dict[str, Any]] = {}
        for row in tree_rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in tree_by_path:
                raise GraphError(f"source tree row is malformed: {tree_name}")
            path = _safe_rel(row["path"], f"{tree_name} path")
            if row.get("type") not in {"blob", "tree"} or not isinstance(row.get("sha"), str):
                raise GraphError(f"source tree row identity is malformed: {tree_name}")
            if row["type"] == "blob" and not HEX40.fullmatch(row["sha"]):
                raise GraphError(f"source tree blob digest is malformed: {tree_name}")
            tree_by_path[path] = row
        for file_row in repo.get("files", []):
            if not isinstance(file_row, dict):
                raise GraphError("source-receipt file row is malformed")
            path = _safe_rel(file_row.get("path"), "source-receipt path")
            packet_name = file_row.get("receipt_name")
            if not isinstance(packet_name, str) or packet_name not in manifest_by_name:
                missing.append(packet_name or path)
                continue
            tree_row = tree_by_path.get(path)
            if tree_row is None or tree_row.get("type") != "blob":
                missing.append(path)
                continue
            body = _read_once(packet / packet_name, packet_name, MAX_FILE_BYTES)
            entry = manifest_by_name[packet_name]
            if entry["size"] != len(body) or entry["sha256"] != _sha256(body):
                raise GraphError(f"manifest binding mismatch: {packet_name}")
            if file_row.get("bytes") != len(body) or file_row.get("sha256") != _sha256(body):
                raise GraphError(f"source-receipt digest mismatch: {path}")
            if file_row.get("git_blob_sha1") != tree_row.get("sha") or _git_blob_sha1(body) != tree_row.get("sha"):
                raise GraphError(f"Git blob binding mismatch: {path}")
            rows.append({
                "path": path,
                "packet_name": packet_name,
                "bytes": len(body),
                "sha256": _sha256(body),
                "git_blob_sha1": tree_row["sha"],
                "revision": commit_sha,
                "repository": repo.get("repository", "UNKNOWN"),
                "body": body,
                "is_package": _is_package_path(path),
                "source_kind": "authenticated-upstream-packet",
                "module": _module_from_path(path),
            })
    return {
        "status": "AUTHENTICATED_SOURCE_RECEIPT" if not missing else "AUTHENTICATED_WITH_MISSING_ROWS",
        "path_rows": rows,
        "missing": sorted(set(str(x) for x in missing)),
        "receipt_sha256": _sha256(receipt_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "legal_approval": receipt.get("legal_approval", "UNKNOWN"),
        "owner_review": receipt.get("owner_review", "UNKNOWN"),
        "publication": receipt.get("publication", "UNKNOWN"),
    }


def _load_packet(packet: Path) -> tuple[dict[str, Any], bytes, dict[str, Any], dict[str, dict[str, Any]]]:
    _safe_directory(packet, "source packet")
    manifest_path = packet / "manifest.json"
    manifest_bytes = _read_once(manifest_path, "manifest.json", MAX_MANIFEST_BYTES)
    manifest = _json(manifest_bytes, "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != "vokra-flat-source-receipt-v1" or manifest.get("status") != "SOURCE_ONLY_PREPARATION":
        raise GraphError("manifest is not source-only preparation")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) > MAX_FILES:
        raise GraphError("manifest file inventory exceeds bound")
    by_name: dict[str, dict[str, Any]] = {}
    total = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "role", "size", "sha256"}:
            raise GraphError("manifest file row is malformed")
        name = entry["name"]
        if not isinstance(name, str) or Path(name).name != name or name in by_name:
            raise GraphError("manifest names must be unique safe basenames")
        if entry["role"] != "authenticated-source" or not isinstance(entry["size"], int) or isinstance(entry["size"], bool) or not 0 <= entry["size"] <= MAX_FILE_BYTES:
            raise GraphError(f"manifest file row is invalid: {name}")
        if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
            raise GraphError(f"manifest digest is invalid: {name}")
        total += entry["size"]
        if total > MAX_TOTAL_BYTES:
            raise GraphError("source packet exceeds aggregate byte bound")
        by_name[name] = entry
    receipt = _source_receipt(packet, manifest, manifest_bytes)
    return manifest, manifest_bytes, receipt, by_name


def _edge_target(module: str, edge: dict[str, Any], modules: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    name = edge["name"]
    if name == "<dynamic-import>":
        return None, "dynamic-import-unknown"
    if edge["relative_level"]:
        source_row = modules.get(module, {})
        candidate = _resolve_relative(
            module,
            name,
            edge["relative_level"],
            source_is_package=bool(source_row.get("is_package")),
        )
        if candidate in modules:
            return candidate, "authenticated-source"
        if "." in candidate and candidate.rsplit(".", 1)[0] in modules:
            parent = modules[candidate.rsplit(".", 1)[0]]
            if not parent.get("is_package"):
                return candidate.rsplit(".", 1)[0], "authenticated-source-symbol"
            return None, "missing-authenticated-source-or-symbol"
        return candidate, "missing-authenticated-source"
    top = _top_name(name)
    if name in modules:
        return name, "authenticated-source"
    if edge["kind"] == "from-import" and top in modules:
        return name, "missing-authenticated-source"
    if edge["kind"] == "import":
        if top in modules:
            return name, "missing-authenticated-source"
        if top in STDLIB_TOP_LEVEL:
            return None, "stdlib"
        return None, "third-party-candidate"
    if edge["kind"] == "from-import-member" and "." in name:
        parent_name = name.rsplit(".", 1)[0]
        if parent_name in modules:
            parent = modules[parent_name]
            if not parent.get("is_package"):
                return parent_name, "authenticated-source-symbol"
            return None, "missing-authenticated-source-or-symbol"
    if top in modules:
        return top, "authenticated-source-symbol"
    if top in STDLIB_TOP_LEVEL:
        return None, "stdlib"
    return None, "third-party-candidate"


def _parse_roots(root_specs: Iterable[str]) -> list[tuple[str, str]]:
    roots: list[tuple[str, str]] = []
    for raw in root_specs:
        if "=" not in raw:
            raise GraphError("root must be MODULE=SOURCE_PATH")
        module, path = raw.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
            raise GraphError("root module name is malformed")
        roots.append((module, _safe_rel(path, "root source path")))
    if not roots:
        roots = list(DEFAULT_ROOTS)
    if len(roots) > 32:
        raise GraphError("too many roots")
    if len({module for module, _path in roots}) != len(roots):
        raise GraphError("duplicate root module")
    return roots


def _package_ancestors(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts))]


def _enqueue_module(
    module: str,
    scope: str,
    modules: dict[str, dict[str, Any]],
    queue: list[tuple[str, str]],
    scopes: dict[str, set[str]],
    unresolved: list[dict[str, Any]],
) -> None:
    """Queue parents before children while retaining missing-init evidence."""
    for ancestor in _package_ancestors(module):
        row = modules.get(ancestor)
        if row is None or not row.get("is_package"):
            unresolved.append({
                "status": "MISSING_AUTHENTICATED_PACKAGE_INIT",
                "module": ancestor,
                "requested_by": module,
                "scope": scope,
            })
            continue
        if scope not in scopes.setdefault(ancestor, set()):
            scopes[ancestor].add(scope)
            queue.append((ancestor, scope))
    if module not in modules:
        unresolved.append({
            "status": "MISSING_AUTHENTICATED_MODULE",
            "module": module,
            "scope": scope,
        })
        return
    if scope not in scopes.setdefault(module, set()):
        scopes[module].add(scope)
        queue.append((module, scope))


def build_graph(
    packet: Path,
    roots: Iterable[str] = (),
    *,
    vokra_root: Path | None = None,
    expected_head: str = EXPECTED_VOKRA_HEAD,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not synthetic and (not isinstance(expected_head, str) or HEX40.fullmatch(expected_head) is None):
        raise GraphError("expected Vokra HEAD is malformed")
    manifest, manifest_bytes, receipt, _manifest_by_name = _load_packet(packet)
    path_rows: dict[str, dict[str, Any]] = {}
    for row in receipt["path_rows"]:
        if row["path"] in path_rows:
            raise GraphError(f"duplicate authenticated source path: {row['path']}")
        path_rows[row["path"]] = dict(row)
        if synthetic:
            path_rows[row["path"]]["source_kind"] = "synthetic-test-input"
    vokra_auth: dict[str, Any]
    if synthetic:
        upstream_auth = {"status": "SYNTHETIC_TEST_INPUT_NOT_AUTHENTICATED"}
        vokra_auth = {"status": "SYNTHETIC_TEST_INPUT_NOT_AUTHENTICATED"}
    else:
        vokra_auth, local_rows = _authenticate_vokra_sources(vokra_root or Path.cwd(), expected_head)
        for path, row in local_rows.items():
            if path in path_rows:
                raise GraphError(f"Vokra path collides with upstream packet: {path}")
            path_rows[path] = row
        upstream_auth = _authenticate_upstream_packet(
            packet,
            trusted_contract=local_rows[VOKRA_SOURCE_PATHS[2]],
            vokra_root=vokra_root or Path.cwd(),
        )
    modules: dict[str, dict[str, Any]] = {}
    for path, row in path_rows.items():
        module = row.get("module") or _module_from_path(path)
        if module is not None:
            if module in modules:
                raise GraphError(f"duplicate authenticated module: {module}")
            modules[module] = row

    root_specs = _parse_roots(roots)
    missing_roots: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    queue: list[tuple[str, str]] = []
    scopes: dict[str, set[str]] = {}
    for module, path in root_specs:
        row = path_rows.get(path)
        if row is None:
            missing_roots.append({"module": module, "path": path, "status": "MISSING_AUTHENTICATED_SOURCE"})
            continue
        if module not in modules:
            modules[module] = row
        _enqueue_module(module, "producer", modules, queue, scopes, unresolved)
    role_status: list[dict[str, Any]] = []
    if not synthetic:
        for module, path in UPSTREAM_ROLE_ROOTS:
            row = path_rows.get(path)
            if row is None:
                role_status.append({"module": module, "path": path, "status": "MISSING_AUTHENTICATED_SOURCE_ROLE"})
                unresolved_role = {"status": "MISSING_AUTHENTICATED_SOURCE_ROLE", "module": module, "path": path}
                # Keep provenance missing separate from the producer's import
                # queue; it is not an excuse to classify an evaluator module
                # as a runtime dependency.
                missing_roots.append(unresolved_role)
                continue
            if module not in modules:
                modules[module] = row
            role_status.append({"module": module, "path": path, "status": "AUTHENTICATED_SOURCE_ROLE"})
            _enqueue_module(module, "source-contract-role", modules, queue, scopes, unresolved)
    else:
        role_status = [
            {"module": module, "path": path, "status": "SYNTHETIC_TEST_INPUT_ROLE"}
            for module, path in UPSTREAM_ROLE_ROOTS
        ]

    parsed_edges: dict[str, list[dict[str, Any]]] = {}
    module_records: dict[str, dict[str, Any]] = {}
    edge_records: dict[tuple[Any, ...], dict[str, Any]] = {}
    edge_scopes: dict[tuple[Any, ...], set[str]] = {}
    external: dict[str, dict[str, Any]] = {}
    external_scopes: dict[str, set[str]] = {}
    unresolved_keys: set[tuple[Any, ...]] = set()
    processed_pairs: set[tuple[str, str]] = set()
    unresolved.extend(missing_roots)
    unresolved.extend(
        {"status": "MISSING_SOURCE_ROW", "path": missing} for missing in receipt["missing"]
    )
    while queue:
        module, scope = queue.pop(0)
        pair = (module, scope)
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        row = modules.get(module)
        if row is None:
            unresolved_key = ("MISSING_AUTHENTICATED_MODULE", module, scope)
            if unresolved_key not in unresolved_keys:
                unresolved_keys.add(unresolved_key)
                unresolved.append({"status": "MISSING_AUTHENTICATED_MODULE", "module": module, "scope": scope})
            continue
        if module not in parsed_edges:
            if len(parsed_edges) >= MAX_GRAPH_MODULES:
                raise GraphError("graph module bound exceeded")
            body = row["body"]
            try:
                tree = ast.parse(body.decode("utf-8"), filename=row["path"], mode="exec")
            except (UnicodeDecodeError, SyntaxError) as exc:
                parsed_edges[module] = []
                unresolved_key = ("SOURCE_PARSE_FAILED", module, row["path"])
                if unresolved_key not in unresolved_keys:
                    unresolved_keys.add(unresolved_key)
                    unresolved.append({"status": "SOURCE_PARSE_FAILED", "module": module, "path": row["path"], "detail": type(exc).__name__})
                continue
            visitor = _ImportVisitor(module)
            visitor.visit(tree)
            parsed_edges[module] = visitor.edges
            module_records[module] = {
                "module": module,
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "git_blob_sha1": row["git_blob_sha1"],
                "source_kind": row.get("source_kind", "UNKNOWN"),
                "imports": len(visitor.edges),
            }
        for edge in parsed_edges[module]:
            target, resolution = _edge_target(module, edge, modules)
            record = {"source": module, "target": target, "name": edge["name"], "kind": edge["kind"], "line": edge["line"], "column": edge["column"], "condition": edge["condition"], "resolution": resolution}
            record_key = (record["source"], record["target"], record["name"], record["kind"], record["line"], record["column"], record["condition"], record["resolution"])
            edge_records.setdefault(record_key, record)
            edge_scopes.setdefault(record_key, set()).add(scope)
            if len(edge_records) > MAX_GRAPH_EDGES:
                raise GraphError("graph edge bound exceeded")
            if resolution.startswith("third-party"):
                candidate_name = _top_name(edge["name"])
                external.setdefault(candidate_name, {"name": candidate_name, "distribution": "UNKNOWN", "status": "CANDIDATE_UNKNOWN", "first_seen": record})
                external_scopes.setdefault(candidate_name, set()).add(scope)
            elif resolution.startswith("dynamic") or resolution.startswith("missing"):
                unresolved_key = (resolution, module, edge["name"], edge["line"])
                if unresolved_key not in unresolved_keys:
                    unresolved_keys.add(unresolved_key)
                    unresolved.append({"status": resolution.upper().replace("-", "_"), "source": module, "name": edge["name"], "line": edge["line"], "scope": scope})
            if target is not None and target in modules:
                _enqueue_module(target, scope, modules, queue, scopes, unresolved)

    for record in module_records.values():
        record["reachability_scopes"] = sorted(scopes.get(record["module"], set()))
    edges = list(edge_records.values())
    for key, record in edge_records.items():
        record["reachability_scopes"] = sorted(edge_scopes[key])
    for candidate in external.values():
        candidate["reachability_scopes"] = sorted(external_scopes[candidate["name"]])

    # Source-only status is intentionally blocked if any root/source edge is
    # missing.  This is not a closure approval and cannot become one by adding
    # a caller-supplied version or distribution name.
    output: dict[str, Any] = {
        "schema": SCHEMA,
        "status": GRAPH_STATUS,
        "execution_status": EXECUTION_STATUS,
        "review_status": REVIEW_STATUS,
        "publication": PUBLICATION_STATUS,
        "source_packet_manifest_sha256": _sha256(manifest_bytes),
        "source_receipt_sha256": receipt.get("receipt_sha256", "UNKNOWN"),
        "source_receipt_status": (
            "SYNTHETIC_TEST_INPUT_NOT_AUTHENTICATED" if synthetic else receipt["status"]
        ),
        "upstream_packet_authentication": upstream_auth,
        "vokra_source_authentication": vokra_auth,
        "roots": [{"module": module, "path": path} for module, path in root_specs],
        "source_contract_roles": role_status,
        "authenticated_modules": sorted(module_records.values(), key=lambda row: row["module"]),
        "edges": sorted(edges, key=lambda row: (row["source"], row["line"], row["name"], row["target"] or "")),
        "external_candidates": sorted(external.values(), key=lambda row: row["name"]),
        "unresolved": sorted(unresolved, key=lambda row: json.dumps(row, sort_keys=True)),
        "license_status": "UNKNOWN_DEPENDENCY_LICENSES",
        "native_status": "UNKNOWN_NATIVE_PAYLOADS",
        "closure_status": "BLOCKED_REVIEW",
        "claim_boundary": "AST source graph only; no distribution/version/license/native/API/runtime/parity approval",
    }
    canonical = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    output["graph_sha256"] = _sha256(canonical)
    return output


def _write_exclusive(path: Path, body: bytes) -> None:
    if not path.is_absolute() or path.is_symlink() or any(part in {".", ".."} for part in path.parts):
        raise GraphError("output must be an absolute non-symlink path")
    if len(body) > MAX_OUTPUT_BYTES:
        raise GraphError("graph output exceeds bounded size")
    if not path.parent.exists():
        raise GraphError("output parent must already exist")
    _safe_directory(path.parent, "output parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise GraphError("refusing to overwrite an existing graph output") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--root", action="append", default=[], metavar="MODULE=SOURCE_PATH")
    parser.add_argument("--vokra-root", type=Path, help="clean Vokra checkout for fixed first-party source authentication")
    parser.add_argument(
        "--expected-vokra-head",
        default=EXPECTED_VOKRA_HEAD,
        help="exact 40-hex Vokra commit to authenticate (default: historical source-review base)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_graph(
            args.packet,
            args.root,
            vokra_root=args.vokra_root,
            expected_head=args.expected_vokra_head,
        )
        body = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if args.output is None:
            sys.stdout.buffer.write(body)
        else:
            _write_exclusive(args.output, body)
            print(f"source dependency graph written: {args.output}")
        return 0
    except (GraphError, OSError) as exc:
        print(f"BLOCKED_REVIEW: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
