"""Authenticate a small, source-only package metadata receipt.

This module consumes explicitly captured GitHub commit, recursive-tree, and
blob JSON responses.  It never imports a package named by a capture, executes
captured text, resolves a dependency, or changes the Kyutai source packet.
The result is supplemental ``SOURCE_METADATA_ONLY`` evidence, not a runtime,
license, or dependency-closure approval.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import stat
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "vokra-kyutai-package-metadata-receipt-v1"
STATUS = "SOURCE_METADATA_ONLY"
MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_BLOB_BYTES = 2 * 1024 * 1024
MAX_TREE_ENTRIES = 250_000
MAX_PATH_BYTES = 4096

MOSHI = {
    "repository": "kyutai-labs/moshi",
    "revision": "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362",
    "tree": "2a6d6afe53d70bac490117651dfc478cf87a940e",
    "files": {
        "moshi/pyproject.toml": {
            "git_blob_sha1": "0a99f52ea834cdcfe1b07ec3cfcbb7e96083fb61",
            "bytes": 1305,
        },
        "moshi/requirements.txt": {
            "git_blob_sha1": "89cee794c64d70932c56be7f964f23aa921dd0fa",
            "bytes": 195,
        },
        "moshi/setup.cfg": {
            "git_blob_sha1": "4c7f6cd5b66032ce7b1d45a779b7692ad9c1d443",
            "bytes": 125,
        },
    },
}
DSM = {
    "repository": "kyutai-labs/delayed-streams-modeling",
    "revision": "4c4f65e147df056adf3346290d64c7b9649b18c9",
    "tree": "1ab73718d99c5bb6ff94c1bc84a783a4d2e3a7e0",
}

THIRD_PARTY_CANDIDATES = (
    "bitsandbytes",
    "einops",
    "huggingface_hub",
    "numpy",
    "safetensors",
    "sentencepiece",
    "torch",
)
DYNAMIC_IMPORT_UNRESOLVED = {
    "name": "kyutai_stt_decoder_dump_reference",
    "status": "DYNAMIC_IMPORT_LITERAL_AUTHENTICATED_SOURCE",
    "origin": "UNKNOWN_RUNTIME_ORIGIN",
}
PACKAGING_NAME = re.compile(
    r"(?:^|/)(?:pyproject\.toml|setup\.py|setup\.cfg|requirements[^/]*|uv\.lock)$"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CaptureError(ValueError):
    """Raised when an explicit API capture is malformed or stale."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_file(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CaptureError(f"cannot stat capture {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CaptureError(f"capture is not a regular file: {path}")
    if info.st_size > MAX_CAPTURE_BYTES:
        raise CaptureError(f"capture exceeds bounded size: {path}")
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_CAPTURE_BYTES:
            raise CaptureError(f"capture grew beyond bounded size: {path}")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"invalid JSON capture: {path}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"capture root is not an object: {path}")
    return value


def _string(value: Any, label: str, *, limit: int = MAX_PATH_BYTES) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise CaptureError(f"invalid {label}")
    return value


def _sha(value: Any, label: str, length: int) -> str:
    value = _string(value, label, limit=length)
    pattern = HEX40 if length == 40 else HEX64
    if not pattern.fullmatch(value):
        raise CaptureError(f"invalid {label}")
    return value


def git_blob_sha1(body: bytes) -> str:
    return hashlib.sha1(f"blob {len(body)}\0".encode("ascii") + body).hexdigest()


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _validate_commit(capture: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if _sha(capture.get("sha"), "commit sha", 40) != spec["revision"]:
        raise CaptureError("commit revision mismatch")
    tree_candidates = []
    commit = capture.get("commit")
    if commit is not None:
        if not isinstance(commit, dict) or not isinstance(commit.get("tree"), dict):
            raise CaptureError("commit response has malformed nested tree")
        tree_candidates.append(commit["tree"].get("sha"))
    if capture.get("tree") is not None:
        if not isinstance(capture["tree"], dict):
            raise CaptureError("commit response has malformed root tree")
        tree_candidates.append(capture["tree"].get("sha"))
    if not tree_candidates or any(_sha(value, "commit tree sha", 40) != spec["tree"] for value in tree_candidates):
        raise CaptureError("commit tree mismatch")


def _validate_path(path: Any) -> str:
    path = _string(path, "tree path")
    if (
        path.startswith("/")
        or path.endswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise CaptureError(f"unsafe tree path: {path!r}")
    return path


def _validate_tree(capture: Mapping[str, Any], expected_root: str) -> dict[str, dict[str, Any]]:
    if _sha(capture.get("sha"), "tree sha", 40) != expected_root:
        raise CaptureError("tree root mismatch")
    if capture.get("truncated") is not False:
        raise CaptureError("tree capture is truncated")
    rows = capture.get("tree")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_TREE_ENTRIES:
        raise CaptureError("invalid or empty recursive tree")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CaptureError("tree row is not an object")
        path = _validate_path(row.get("path"))
        if path in by_path:
            raise CaptureError(f"duplicate tree path: {path}")
        kind = row.get("type")
        mode = row.get("mode")
        if kind not in {"blob", "tree"} or mode not in {"040000", "100644", "100755"}:
            raise CaptureError(f"non-regular tree entry: {path}")
        if kind == "tree" and mode != "040000":
            raise CaptureError(f"tree has non-tree mode: {path}")
        if kind == "blob" and mode == "040000":
            raise CaptureError(f"blob has tree mode: {path}")
        row = dict(row)
        row["path"] = path
        row["sha"] = _sha(row.get("sha"), f"tree row sha {path}", 40)
        if kind == "blob":
            size = row.get("size")
            if type(size) is not int or size < 0:
                raise CaptureError(f"invalid blob size: {path}")
        by_path[path] = row

    children: dict[str, list[dict[str, Any]]] = {"": []}
    for path, row in by_path.items():
        parts = path.split("/")
        parent = "/".join(parts[:-1])
        if parent and (parent not in by_path or by_path[parent]["type"] != "tree"):
            raise CaptureError(f"tree parent missing: {path}")
        children.setdefault(parent, []).append(row)
        if row["type"] == "tree":
            children.setdefault(path, [])

    computed: dict[str, str] = {}
    tree_paths = {""} | {path for path, row in by_path.items() if row["type"] == "tree"}
    # Reconstruct descendants before parents, with the synthetic root last
    # when it has the same depth as a top-level directory.
    for tree_path in sorted(tree_paths, key=lambda item: (-item.count("/"), item == "")):
        entries = children.get(tree_path, [])
        encoded: list[tuple[bytes, bytes]] = []
        for row in entries:
            name = row["path"].rsplit("/", 1)[-1].encode("utf-8")
            if row["type"] == "tree":
                child_sha = computed.get(row["path"])
                if child_sha is None:
                    raise CaptureError(f"tree child was not reconstructed: {row['path']}")
                object_sha = child_sha
                sort_name = name + b"/"
            else:
                object_sha = row["sha"]
                sort_name = name
            # GitHub exposes directory metadata as 040000; Git's canonical
            # tree object serializes that mode as 40000 (no leading zero).
            git_mode = "40000" if row["type"] == "tree" else row["mode"]
            encoded.append((sort_name, f"{git_mode} ".encode("ascii") + name + b"\0" + bytes.fromhex(object_sha)))
        encoded.sort(key=lambda item: item[0])
        body = b"".join(item[1] for item in encoded)
        digest = hashlib.sha1(f"tree {len(body)}\0".encode("ascii") + body).hexdigest()
        if tree_path:
            if digest != by_path[tree_path]["sha"]:
                raise CaptureError(f"child tree hash mismatch: {tree_path}")
        computed[tree_path] = digest
    if computed.get("") != expected_root:
        raise CaptureError("recursive tree content does not match root hash")
    return by_path


def _blob_body(capture: Mapping[str, Any], path: str, expected: Mapping[str, Any], tree_row: Mapping[str, Any]) -> bytes:
    if capture.get("encoding") != "base64":
        raise CaptureError(f"blob encoding is not base64: {path}")
    if _sha(capture.get("sha"), f"blob response sha {path}", 40) != expected["git_blob_sha1"]:
        raise CaptureError(f"blob response sha mismatch: {path}")
    if type(capture.get("size")) is not int or type(tree_row.get("size")) is not int:
        raise CaptureError(f"blob size is not an integer: {path}")
    if capture.get("size") != expected["bytes"] or tree_row.get("size") != expected["bytes"]:
        raise CaptureError(f"blob size mismatch: {path}")
    content = capture.get("content")
    if not isinstance(content, str) or len(content) > MAX_BLOB_BYTES * 2:
        raise CaptureError(f"invalid blob content: {path}")
    try:
        compact = "".join(content.split())
        body = base64.b64decode(compact, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CaptureError(f"invalid blob base64: {path}") from exc
    if len(body) != expected["bytes"] or git_blob_sha1(body) != expected["git_blob_sha1"]:
        raise CaptureError(f"blob content hash mismatch: {path}")
    return body


def _compact(text: str) -> str:
    return re.sub(r"[\s\"']", "", text).lower()


def _constraint_lines(text: str) -> list[str]:
    package_names = "torch|bitsandbytes|numpy|safetensors|huggingface[-_]hub|einops|sentencepiece|sounddevice|sphn|aiohttp|tqdm"
    result = []
    for line in text.splitlines():
        if re.search(rf"(?i)(?:{package_names}|requires-python)\s*(?:[<>=!~]|\")", line):
            result.append(line.strip())
    return result


def _require_constraints(files: Mapping[str, bytes]) -> dict[str, Any]:
    try:
        pyproject = files["moshi/pyproject.toml"].decode("utf-8")
        requirements = files["moshi/requirements.txt"].decode("utf-8")
        setup_cfg = files["moshi/setup.cfg"].decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise CaptureError("packaging blobs must be UTF-8 text") from exc
    try:
        document = tomllib.loads(pyproject)
        project = document["project"]
        python_constraint = project["requires-python"]
        dependencies = project["dependencies"]
        if type(python_constraint) is not str or not isinstance(dependencies, list):
            raise ValueError("invalid project metadata")
        dependency_lines = [value for value in dependencies if isinstance(value, str)]
    except (KeyError, TypeError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise CaptureError("pyproject packaging metadata is not valid TOML") from exc
    compact_dependencies = {_compact(value) for value in dependency_lines}
    required = {
        "python": _compact(python_constraint) == ">=3.10,<3.15",
        "torch": "torch>=2.2.0,<2.10" in compact_dependencies,
        "bitsandbytes": "bitsandbytes>=0.45,<0.50.0;sys_platform==linux" in compact_dependencies,
        "sphn_pyproject": "sphn>=0.2.0,<0.3.0" in compact_dependencies,
        "sphn_requirements": "sphn==0.1.4" in {_compact(line) for line in requirements.splitlines()},
    }
    if not all(required.values()):
        raise CaptureError("required raw packaging constraint or conflict is absent")
    return {
        "raw_constraint_lines": {
            "moshi/pyproject.toml": _constraint_lines(pyproject),
            "moshi/requirements.txt": _constraint_lines(requirements),
            "moshi/setup.cfg": _constraint_lines(setup_cfg),
        },
        "required_facts": {
            "python": ">= 3.10,<3.15",
            "torch": ">= 2.2.0,< 2.10",
            "bitsandbytes": ">= 0.45,< 0.50.0",
            "bitsandbytes_linux_marker": "sys_platform == 'linux'",
            "sphn_pyproject": ">= 0.2.0,< 0.3.0",
            "sphn_requirements": "==0.1.4",
        },
        "conflicts": [
            {
                "package": "sphn",
                "left": "moshi/pyproject.toml: >=0.2,<0.3",
                "right": "moshi/requirements.txt: ==0.1.4",
                "status": "PACKAGING_CONSTRAINT_CONFLICT",
            }
        ],
    }


def _file_record(path: str, body: bytes, api: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(body),
        "git_blob_sha1": git_blob_sha1(body),
        "sha256": _sha256(body),
        "content_base64": base64.b64encode(body).decode("ascii"),
        "encoding": "base64",
        "api_sha": api["sha"],
    }


def _audit_captures_from_specs(
    moshi_commit: Mapping[str, Any],
    moshi_tree: Mapping[str, Any],
    moshi_blobs: Mapping[str, Mapping[str, Any]],
    dsm_commit: Mapping[str, Any],
    dsm_tree: Mapping[str, Any],
    *,
    moshi_spec: Mapping[str, Any],
    dsm_spec: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_commit(moshi_commit, moshi_spec)
    _validate_commit(dsm_commit, dsm_spec)
    moshi_rows = _validate_tree(moshi_tree, moshi_spec["tree"])
    dsm_rows = _validate_tree(dsm_tree, dsm_spec["tree"])
    expected_files = moshi_spec["files"]
    if set(moshi_blobs) != set(expected_files):
        raise CaptureError("packaging capture paths do not exactly match fixed paths")
    scoped_packaging_paths = {
        path for path in moshi_rows if path.startswith("moshi/") and PACKAGING_NAME.search(path)
    }
    if scoped_packaging_paths != set(expected_files):
        raise CaptureError("CPU moshi/ packaging paths differ from the fixed capture set")
    all_packaging_paths = {path for path in moshi_rows if PACKAGING_NAME.search(path)}
    unassessed_packaging_paths = sorted(all_packaging_paths - scoped_packaging_paths)
    bodies: dict[str, bytes] = {}
    file_records = []
    for path, expected in expected_files.items():
        row = moshi_rows.get(path)
        if row is None or row["type"] != "blob" or row["sha"] != expected["git_blob_sha1"]:
            raise CaptureError(f"packaging path is not the fixed tree blob: {path}")
        body = _blob_body(moshi_blobs[path], path, expected, row)
        bodies[path] = body
        file_records.append(_file_record(path, body, moshi_blobs[path]))
    packaging = _require_constraints(bodies)
    matched_dsm = sorted(path for path in dsm_rows if PACKAGING_NAME.search(path))
    if matched_dsm:
        raise CaptureError(f"DSM packaging metadata unexpectedly present: {matched_dsm}")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "scope": "fixed upstream packaging metadata only",
        "execution": "NOT_RUN",
        "execution_closure": False,
        "runtime_origin": "UNKNOWN",
        "runtime_execution": False,
        "dependency_resolution": False,
        "model_activity": False,
        "weight_activity": False,
        "audio_activity": False,
        "owner_review": False,
        "legal_approval": False,
        "license_approval": False,
        "native_approval": False,
        "license_status": "UNKNOWN_DEPENDENCY_LICENSES",
        "native_payload_status": "UNKNOWN_NATIVE_PAYLOADS",
        "reviewed_dependency_closure_sha256": "",
        "no_upload": True,
        "repositories": [
            {
                "repository": moshi_spec["repository"],
                "revision": moshi_spec["revision"],
                "tree_sha": moshi_spec["tree"],
                "complete_tree": True,
                "packaging_paths": sorted(expected_files),
                "unassessed_packaging_paths": unassessed_packaging_paths,
                "unassessed_packaging_scope": "outside CPU moshi/ subtree",
            },
            {
                "repository": dsm_spec["repository"],
                "revision": dsm_spec["revision"],
                "tree_sha": dsm_spec["tree"],
                "complete_tree": True,
                "matched_packaging_paths": matched_dsm,
                "absence_rule": PACKAGING_NAME.pattern,
            },
        ],
        "files": file_records,
        "packaging": packaging,
        "source_graph_preserved": {
            "third_party_candidates": list(THIRD_PARTY_CANDIDATES),
            "candidate_status": "CANDIDATE_UNKNOWN",
            "dynamic_import": dict(DYNAMIC_IMPORT_UNRESOLVED),
        },
    }


def audit_captures(
    moshi_commit: Mapping[str, Any],
    moshi_tree: Mapping[str, Any],
    moshi_blobs: Mapping[str, Mapping[str, Any]],
    dsm_commit: Mapping[str, Any],
    dsm_tree: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit only the fixed production identities; test specs stay private."""
    return _audit_captures_from_specs(
        moshi_commit,
        moshi_tree,
        moshi_blobs,
        dsm_commit,
        dsm_tree,
        moshi_spec=MOSHI,
        dsm_spec=DSM,
    )


def audit_paths(
    *,
    moshi_commit: Path,
    moshi_tree: Path,
    moshi_pyproject: Path,
    moshi_requirements: Path,
    moshi_setup_cfg: Path,
    dsm_commit: Path,
    dsm_tree: Path,
) -> dict[str, Any]:
    return audit_captures(
        _json_file(moshi_commit),
        _json_file(moshi_tree),
        {
            "moshi/pyproject.toml": _json_file(moshi_pyproject),
            "moshi/requirements.txt": _json_file(moshi_requirements),
            "moshi/setup.cfg": _json_file(moshi_setup_cfg),
        },
        _json_file(dsm_commit),
        _json_file(dsm_tree),
    )


def _write_new_file(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise CaptureError(f"refusing to overwrite existing output: {path}") from exc
    except OSError as exc:
        raise CaptureError(f"cannot create output: {path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moshi-commit", type=Path, required=True)
    parser.add_argument("--moshi-tree", type=Path, required=True)
    parser.add_argument("--moshi-pyproject", type=Path, required=True)
    parser.add_argument("--moshi-requirements", type=Path, required=True)
    parser.add_argument("--moshi-setup-cfg", type=Path, required=True)
    parser.add_argument("--dsm-commit", type=Path, required=True)
    parser.add_argument("--dsm-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = audit_paths(
        moshi_commit=args.moshi_commit,
        moshi_tree=args.moshi_tree,
        moshi_pyproject=args.moshi_pyproject,
        moshi_requirements=args.moshi_requirements,
        moshi_setup_cfg=args.moshi_setup_cfg,
        dsm_commit=args.dsm_commit,
        dsm_tree=args.dsm_tree,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write_new_file(args.output, rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
