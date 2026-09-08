#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Metadata-only identity audit for VibeVoice-ASR's Qwen companion.

The VibeVoice source names ``Qwen/Qwen2.5-7B`` as an external language-model
dependency.  This oracle authenticates the current Hugging Face metadata for
the fixed revision without resolving a file, downloading a shard, importing a
model library, or running inference.  It deliberately remains blocked for
owner/runtime/parity review.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


REPOSITORY = "Qwen/Qwen2.5-7B"
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice"
SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
SOURCE_COMMIT_DATE = "2026-07-24"
SOURCE_DECLARATION_PATH = "demo/vibevoice_asr_inference_from_file.py"
SOURCE_DECLARATION_BLOB = "bf4f75df74c7299c6596bb09d77e8c67400ff1ac"
SOURCE_DECLARATION_MARKER = 'language_model_pretrained_name="Qwen/Qwen2.5-7B"'
MODEL_INFO_URL = f"https://huggingface.co/api/models/{REPOSITORY}?revision={REVISION}&blobs=true"
CURRENT_MODEL_INFO_URL = f"https://huggingface.co/api/models/{REPOSITORY}"
HISTORY_URL = f"https://huggingface.co/api/models/{REPOSITORY}/commits/main"
LICENSE_URL = f"https://huggingface.co/{REPOSITORY}/raw/{REVISION}/LICENSE"
GITHUB_SOURCE_COMMIT_URL = f"https://api.github.com/repos/microsoft/VibeVoice/commits/{SOURCE_REVISION}"
GITHUB_SOURCE_TREE_URL = f"https://api.github.com/repos/microsoft/VibeVoice/git/trees/{SOURCE_REVISION}?recursive=1"
GITHUB_SOURCE_BLOB_URL = f"https://api.github.com/repos/microsoft/VibeVoice/git/blobs/{SOURCE_DECLARATION_BLOB}"
EXPECTED_FILES = frozenset(
    {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
SHARD_FILES = frozenset(path for path in EXPECTED_FILES if path.endswith(".safetensors"))
SCHEMA = "vokra-vibevoice-asr-qwen2-5-7b-metadata-v1"
BLOCKERS = [
    "PENDING_OWNER_REVIEW",
    "NO_CHECKPOINT_ACQUIRED",
    "NATIVE_ASR_CPU_METAL_AND_INDEPENDENT_PARITY_REMAIN_BLOCKED",
]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_MODEL_INFO_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BLOB_BYTES = 4 * 1024 * 1024
AUTH_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF", "GH_TOKEN", "GITHUB_TOKEN")
FIXED_ENDPOINTS = frozenset(
    {
        MODEL_INFO_URL,
        CURRENT_MODEL_INFO_URL,
        HISTORY_URL,
        GITHUB_SOURCE_COMMIT_URL,
        GITHUB_SOURCE_TREE_URL,
        GITHUB_SOURCE_BLOB_URL,
    }
)


class AuditError(ValueError):
    """A malformed, drifting, or unsafe metadata audit input."""


def strict_json(raw: bytes, source: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key from {source}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, AuditError) as error:
        raise AuditError(f"strict JSON decode failed for {source}: {error}") from error


def validate_head(value: Any, label: str = "expected_head") -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise AuditError(f"{label} must be exactly 40 lowercase hexadecimal characters")
    return value


def _real_directory(path: Path, label: str) -> Path:
    raw = str(path)
    if (
        not path.is_absolute()
        or "\x00" in raw
        or "\\" in raw
        or "//" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
    ):
        raise AuditError(f"{label} must be an absolute dot-free path")
    if path.is_symlink() or not path.is_dir():
        raise AuditError(f"{label} must be an existing real directory")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise AuditError(f"{label} has symlinked ancestry")
    return path


def verify_checkout(repo_root: Path, expected_head: str, *, runner: Any = subprocess.run) -> dict[str, Any]:
    """Verify Vokra identity before constructing any network request."""

    expected = validate_head(expected_head)
    root = _real_directory(repo_root, "repo-root")

    def git(*args: str) -> str:
        result = runner(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AuditError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    actual = validate_head(git("rev-parse", "--verify", "HEAD"), "actual_head")
    clean = git("status", "--porcelain", "--untracked-files=all") == ""
    if actual != expected:
        raise AuditError(f"Vokra checkout HEAD {actual} differs from {expected}")
    if not clean:
        raise AuditError("Vokra checkout is dirty; metadata audit requires a clean tree")
    return {"expected_head": expected, "actual_head": actual, "clean": True}


def safe_absent_output(repo_root: Path, path: Path) -> Path:
    root = _real_directory(repo_root, "repo-root").resolve()
    raw = str(path)
    if (
        not path.is_absolute()
        or "\x00" in raw
        or "\\" in raw
        or "//" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
        or path.name in {"", ".", ".."}
    ):
        raise AuditError("output must be an absolute dot-free path")
    if path.exists() or path.is_symlink():
        raise AuditError("output already exists; refusing replacement")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise AuditError("output parent must be an existing real directory")
    for ancestor in (parent, *parent.parents):
        if ancestor.is_symlink():
            raise AuditError("output has symlinked ancestry")
    resolved = path.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise AuditError("output must be outside the Vokra checkout")
    return path


def write_no_replace(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditError("output appeared during audit; refusing replacement") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def safe_existing_evidence(repo_root: Path, path: Path) -> Path:
    root = _real_directory(repo_root, "repo-root").resolve()
    raw = str(path)
    if not path.is_absolute() or "\x00" in raw or "\\" in raw or "//" in raw or any(part in {"", ".", ".."} for part in raw.split("/")[1:]):
        raise AuditError("evidence must be an absolute dot-free path")
    if path.is_symlink() or not path.is_file():
        raise AuditError("evidence must be an existing regular file")
    for ancestor in (path.parent, *path.parent.parents):
        if ancestor.is_symlink():
            raise AuditError("evidence has symlinked ancestry")
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        raise AuditError("evidence must be outside the Vokra checkout")
    return path


def _assert_no_ambient_auth() -> None:
    leaked = [name for name in AUTH_ENV_NAMES if os.environ.get(name)]
    if leaked:
        raise AuditError(f"ambient token/auth environment is forbidden: {','.join(leaked)}")


def _read_bounded(stream: BinaryIO, limit: int) -> bytes:
    data = bytearray()
    while len(data) <= limit:
        chunk = stream.read(min(64 * 1024, limit + 1 - len(data)))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > limit:
            raise AuditError(f"response exceeds hard bound of {limit} bytes")
    raise AuditError(f"response exceeds hard bound of {limit} bytes")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        raise AuditError(f"redirect is forbidden: {newurl}")


def _fixed_url(url: str, expected: str) -> None:
    try:
        parsed = urlsplit(url)
        wanted = urlsplit(expected)
        port = parsed.port
    except ValueError as error:
        raise AuditError(f"response URL is malformed: {url}") from error
    if (
        parsed.scheme != "https"
        or parsed.netloc != wanted.netloc
        or parsed.path != wanted.path
        or parsed.query != wanted.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise AuditError(f"response URL is not the fixed HTTPS endpoint: {url}")


def fetch_json(url: str, *, limit: int = MAX_RESPONSE_BYTES) -> tuple[Any, bytes]:
    """Fetch exactly one public endpoint with bounded raw strict JSON."""

    if url not in FIXED_ENDPOINTS:
        raise AuditError(f"URL is not in the fixed metadata endpoint allowlist: {url}")
    _fixed_url(url, url)
    _assert_no_ambient_auth()
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "vokra-vibevoice-asr-qwen-audit/1"})
    if any(key.casefold() == "authorization" for key, _ in request.header_items()):
        raise AuditError("metadata request unexpectedly carries Authorization")
    opener = build_opener(_NoRedirect(), ProxyHandler({}))
    try:
        with opener.open(request, timeout=30.0) as response:
            if response.geturl() != url:
                raise AuditError(f"response URL drifted: {response.geturl()}")
            if response.getcode() != 200:
                raise AuditError(f"endpoint returned HTTP {response.getcode()}: {url}")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
            if content_type != "application/json":
                raise AuditError(f"endpoint did not return JSON: {content_type!r}")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    if int(declared) < 0 or int(declared) > limit:
                        raise AuditError("Content-Length exceeds hard response bound")
                except ValueError as error:
                    raise AuditError("Content-Length is malformed") from error
            raw = _read_bounded(response, limit)
    except AuditError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        raise AuditError(f"metadata request failed: {error}") from error
    return strict_json(raw, url), raw


def fetch_bytes(url: str, *, limit: int, content_type: str | None = None) -> tuple[bytes, dict[str, str]]:
    """Fetch one fixed non-model payload (the small LICENSE only)."""

    if url != LICENSE_URL:
        raise AuditError(f"URL is not the fixed LICENSE endpoint: {url}")
    _fixed_url(url, url)
    _assert_no_ambient_auth()
    request = Request(url, headers={"Accept": "text/plain", "User-Agent": "vokra-vibevoice-asr-qwen-audit/1"})
    opener = build_opener(_NoRedirect(), ProxyHandler({}))
    try:
        with opener.open(request, timeout=30.0) as response:
            if response.geturl() != url or response.getcode() != 200:
                raise AuditError("fixed LICENSE response URL/status drifted")
            if content_type is not None and response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold() != content_type:
                raise AuditError("fixed LICENSE content type drifted")
            raw = _read_bounded(response, limit)
            headers = {key.casefold(): value for key, value in response.headers.items()}
    except AuditError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        raise AuditError(f"LICENSE request failed: {error}") from error
    return raw, headers


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _validate_path(path: Any) -> str:
    if not isinstance(path, str) or not path or "\\" in path or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        raise AuditError(f"unsafe HF path: {path!r}")
    return path


def validate_file(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise AuditError("HF siblings entry is not an object")
    path = _validate_path(item.get("rfilename"))
    size = item.get("size")
    blob = item.get("blobId")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise AuditError(f"invalid positive file size: {path}")
    if not isinstance(blob, str) or not HEX40.fullmatch(blob):
        raise AuditError(f"invalid regular Git blob identity: {path}")
    lfs = item.get("lfs")
    if lfs is None:
        return {"path": path, "size": size, "git_blob_sha1": blob, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None, "lfs_pointer_size": None}
    if not isinstance(lfs, dict):
        raise AuditError(f"LFS metadata is not an object: {path}")
    digest = lfs.get("sha256")
    payload_size = lfs.get("size")
    pointer_size = lfs.get("pointerSize")
    oid = lfs.get("oid")
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise AuditError(f"invalid LFS payload digest: {path}")
    if isinstance(payload_size, bool) or not isinstance(payload_size, int) or payload_size <= 0 or payload_size != size:
        raise AuditError(f"LFS payload size is not bound to file size: {path}")
    if isinstance(pointer_size, bool) or not isinstance(pointer_size, int) or pointer_size <= 0:
        raise AuditError(f"LFS pointerSize is malformed: {path}")
    if oid is not None and (not isinstance(oid, str) or not HEX64.fullmatch(oid) or oid != digest):
        raise AuditError(f"LFS oid is not bound to the payload digest: {path}")
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {payload_size}\n".encode()
    if pointer_size != len(pointer) or _git_blob_sha1(pointer) != blob:
        raise AuditError(f"LFS pointer identity/size mismatch: {path}")
    return {"path": path, "size": size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": blob, "lfs_payload_sha256": digest, "lfs_payload_size": payload_size, "lfs_pointer_size": pointer_size}


def validate_model_info(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("sha") != REVISION:
        raise AuditError("HF fixed metadata revision does not match the pinned head")
    card = payload.get("cardData")
    if not isinstance(card, dict) or card.get("license") != "apache-2.0":
        raise AuditError("HF model-card license is not the recorded Apache-2.0 identity")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or len(siblings) != len(EXPECTED_FILES):
        raise AuditError("HF model tree does not have the exact 14-file cardinality")
    rows: dict[str, dict[str, Any]] = {}
    for item in siblings:
        row = validate_file(item)
        if row["path"] in rows:
            raise AuditError(f"duplicate HF path: {row['path']}")
        rows[row["path"]] = row
    if set(rows) != EXPECTED_FILES:
        raise AuditError(f"HF model tree path set drifted: missing={sorted(EXPECTED_FILES - set(rows))}, extra={sorted(set(rows) - EXPECTED_FILES)}")
    license_row = rows["LICENSE"]
    if license_row["git_blob_sha1"] is None:
        raise AuditError("LICENSE must be a regular server blob, not an LFS payload")
    return rows, {"license": card["license"], "license_file": license_row}


def validate_current_head(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AuditError("current HF model-info response is not an object")
    if payload.get("sha") != REVISION:
        raise AuditError(f"HF current head drifted from fixed Qwen revision: {payload.get('sha')!r}")
    return {"current_head": payload["sha"], "matches_fixed_revision": True, "scope": "Hugging Face model-info current head only; other refs are not audited"}


def validate_history(payload: Any, source_date: str) -> dict[str, Any]:
    """Authenticate the bounded official HF main-branch commit history."""

    if not isinstance(payload, list) or len(payload) != 11:
        raise AuditError("HF main history must contain exactly 11 recorded commits")
    commits: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise AuditError("HF history entry is not an object")
        commit_id = item.get("id")
        created = item.get("createdAt")
        if not isinstance(commit_id, str) or not HEX40.fullmatch(commit_id) or commit_id in seen:
            raise AuditError("HF history commit identity is malformed or duplicated")
        created = _iso_date(created, "HF history createdAt")
        if created[:10] >= source_date[:10]:
            raise AuditError("HF history contains a commit not preceding the source commit")
        seen.add(commit_id)
        commits.append({"id": commit_id, "created_at": created})
    if commits[0]["id"] != REVISION:
        raise AuditError("HF history head does not equal the fixed Qwen revision")
    if any(commits[index]["created_at"] < commits[index + 1]["created_at"] for index in range(len(commits) - 1)):
        raise AuditError("HF history is not in newest-to-oldest order")
    return {"endpoint": HISTORY_URL, "count": len(commits), "head_matches_fixed": True, "all_precede_source": True, "commits": commits, "scope": "official Hugging Face main branch only; other refs are not inferred"}


def _iso_date(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T[^\s]+Z", value):
        raise AuditError(f"{label} is not an ISO-8601 UTC timestamp")
    return value


def validate_source_evidence(
    commit: Any,
    tree: Any,
    blob: Any,
    *,
    expected_blob: str = SOURCE_DECLARATION_BLOB,
) -> dict[str, Any]:
    if not isinstance(commit, dict) or commit.get("sha") != SOURCE_REVISION:
        raise AuditError("fixed Microsoft source commit identity drifted")
    commit_data = commit.get("commit")
    if not isinstance(commit_data, dict) or not isinstance(commit_data.get("committer"), dict):
        raise AuditError("source commit date evidence is missing")
    source_date = _iso_date(commit_data["committer"].get("date"), "source commit date")
    if not source_date.startswith(SOURCE_COMMIT_DATE):
        raise AuditError("source commit date drifted from the recorded source date")
    tree_sha = commit_data.get("tree", {}).get("sha") if isinstance(commit_data.get("tree"), dict) else None
    if not isinstance(tree_sha, str) or not HEX40.fullmatch(tree_sha):
        raise AuditError("source commit tree SHA is missing")
    if not isinstance(tree, dict) or tree.get("sha") != tree_sha or tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise AuditError("source tree evidence is malformed")
    paths: set[str] = set()
    for item in tree["tree"]:
        if not isinstance(item, dict) or _validate_path(item.get("path")) in paths:
            raise AuditError("source tree contains an unsafe or duplicate path")
        path = item["path"]
        if item.get("type") not in {"blob", "tree"} or not isinstance(item.get("mode"), str) or not HEX40.fullmatch(item.get("sha", "")):
            raise AuditError(f"source tree entry is malformed: {path}")
        paths.add(path)
    matches = [item for item in tree["tree"] if isinstance(item, dict) and item.get("path") == SOURCE_DECLARATION_PATH]
    if len(matches) != 1 or matches[0].get("type") != "blob" or matches[0].get("sha") != expected_blob or matches[0].get("size") is None:
        raise AuditError("source declaration role is not the already pinned Git blob")
    if not isinstance(blob, dict) or blob.get("sha") != expected_blob or blob.get("encoding") != "base64":
        raise AuditError("source declaration blob metadata is not fixed")
    encoded = blob.get("content")
    if not isinstance(encoded, str):
        raise AuditError("source declaration blob content is missing")
    try:
        content = base64.b64decode("".join(encoded.split()), validate=True)
    except (ValueError, binascii.Error) as error:
        raise AuditError("source declaration blob base64 is malformed") from error
    if len(content) > MAX_SOURCE_BLOB_BYTES or matches[0].get("size") != len(content) or blob.get("size") != len(content) or _git_blob_sha1(content) != expected_blob or SOURCE_DECLARATION_MARKER.encode() not in content:
        raise AuditError("source declaration role content/blob/marker authentication failed")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "path": SOURCE_DECLARATION_PATH, "git_blob_sha1": expected_blob, "marker": SOURCE_DECLARATION_MARKER, "status": "AUTHENTICATED", "source_commit_date": source_date}


def validate_license(raw: bytes, headers: dict[str, str], *, expected_blob: str = "6634c8cc3133b3848ec74b9f275acaaa1ea618ab", expected_bytes: int = 11_343, expected_revision: str = REVISION) -> dict[str, Any]:
    if len(raw) != expected_bytes or _git_blob_sha1(raw) != expected_blob:
        raise AuditError("fixed Qwen LICENSE bytes/blob identity drifted")
    text = raw.decode("utf-8", errors="strict")
    markers = ("Apache License", "Version 2.0", "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION")
    if any(marker not in text for marker in markers):
        raise AuditError("fixed Qwen LICENSE does not contain Apache-2.0 clauses")
    if headers.get("x-repo-commit") != expected_revision:
        raise AuditError("fixed Qwen LICENSE X-Repo-Commit header drifted")
    etag = headers.get("etag", "").strip().strip('"')
    if etag != expected_blob:
        raise AuditError("fixed Qwen LICENSE ETag/blob identity drifted")
    return {"url": LICENSE_URL, "path": "LICENSE", "bytes": len(raw), "git_blob_sha1": etag, "sha256": hashlib.sha256(raw).hexdigest(), "x_repo_commit": headers["x-repo-commit"], "apache_2_0": True, "status": "AUTHENTICATED"}


def build_report(fixed: Any, current: Any, history: Any, source_commit: Any, source_tree: Any, source_blob: Any, license_raw: bytes, license_headers: dict[str, str], checkout: dict[str, Any], raw_fixed: bytes, raw_current: bytes, raw_history: bytes, raw_commit: bytes, raw_tree: bytes, raw_blob: bytes, *, expected_source_blob: str = SOURCE_DECLARATION_BLOB, expected_license_blob: str = "6634c8cc3133b3848ec74b9f275acaaa1ea618ab", expected_license_bytes: int = 11_343) -> dict[str, Any]:
    rows, license_info = validate_model_info(fixed)
    current_info = validate_current_head(current)
    source_info = validate_source_evidence(source_commit, source_tree, source_blob, expected_blob=expected_source_blob)
    qwen_date = _iso_date(fixed.get("lastModified"), "Qwen fixed-head date")
    if qwen_date[:10] >= source_info["source_commit_date"][:10]:
        raise AuditError("Qwen fixed head does not precede the pinned VibeVoice source commit")
    history_info = validate_history(history, source_info["source_commit_date"])
    license_bytes = validate_license(license_raw, license_headers, expected_blob=expected_license_blob, expected_bytes=expected_license_bytes)
    license_report = {"card_license": license_info["license"], "license_file_metadata": license_info["license_file"], "license_bytes": license_bytes}
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "publication": "NO_UPLOAD",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "model_weights": "NOT_DOWNLOADED",
        "external_dependency": {"repository": REPOSITORY, "revision": REVISION, "selection_status": "AUTHENTICATED_METADATA_ONLY", "files": "NOT_DOWNLOADED", "model_weights": "NOT_DOWNLOADED"},
        "upstream": {"repository": REPOSITORY, "fixed_revision": REVISION, "fixed_head_date": qwen_date, "file_count": len(rows), "files": [rows[path] for path in sorted(rows)]},
        "license": license_report,
        "chronology": {"qwen_head_precedes_source_commit": True, "qwen_head_date": qwen_date, "source_commit_date": source_info["source_commit_date"], "current_head": current_info, "main_history": history_info},
        "source_declaration": source_info,
        "vokra_checkout": checkout,
        "responses": [{"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "strict_json": strict, "max_bytes": limit} for url, raw, limit, strict in ((MODEL_INFO_URL, raw_fixed, MAX_MODEL_INFO_BYTES, True), (CURRENT_MODEL_INFO_URL, raw_current, MAX_MODEL_INFO_BYTES, True), (HISTORY_URL, raw_history, MAX_MODEL_INFO_BYTES, True), (GITHUB_SOURCE_COMMIT_URL, raw_commit, MAX_RESPONSE_BYTES, True), (GITHUB_SOURCE_TREE_URL, raw_tree, MAX_RESPONSE_BYTES, True), (GITHUB_SOURCE_BLOB_URL, raw_blob, MAX_SOURCE_BLOB_BYTES, True), (LICENSE_URL, license_raw, 16 * 1024, False))],
        "acquisition": {"transport": "stdlib urllib HTTPS GET; redirects disabled; proxy disabled", "authorization_header": False, "ambient_token": False, "checkpoint_download": False, "file_resolve": False, "model_construction": False, "model_execution": False},
        "blockers": BLOCKERS.copy(),
    }


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuditError(f"{label} schema is not exact")
    return value


def _validate_file_row(row: Any, expected_path: str, expect_lfs: bool) -> None:
    value = _exact(row, {"path", "size", "git_blob_sha1", "lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size", "lfs_pointer_size"}, expected_path)
    if value["path"] != expected_path or isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        raise AuditError(f"upstream file row is malformed: {expected_path}")
    if expect_lfs:
        if value["git_blob_sha1"] is not None or not isinstance(value["lfs_pointer_git_blob_sha1"], str) or not HEX40.fullmatch(value["lfs_pointer_git_blob_sha1"]) or not isinstance(value["lfs_payload_sha256"], str) or not HEX64.fullmatch(value["lfs_payload_sha256"]) or value["lfs_payload_size"] != value["size"] or isinstance(value["lfs_pointer_size"], bool) or not isinstance(value["lfs_pointer_size"], int) or value["lfs_pointer_size"] <= 0:
            raise AuditError(f"LFS upstream file row is malformed: {expected_path}")
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{value['lfs_payload_sha256']}\nsize {value['lfs_payload_size']}\n".encode()
        if value["lfs_pointer_size"] != len(pointer) or value["lfs_pointer_git_blob_sha1"] != _git_blob_sha1(pointer):
            raise AuditError(f"LFS pointer identity/size is not bound: {expected_path}")
    elif not isinstance(value["git_blob_sha1"], str) or not HEX40.fullmatch(value["git_blob_sha1"]) or any(value[key] is not None for key in ("lfs_pointer_git_blob_sha1", "lfs_payload_sha256", "lfs_payload_size", "lfs_pointer_size")):
        raise AuditError(f"regular upstream file row is malformed: {expected_path}")


def validate_report(report: Any, expected_head: str, *, expected_source_blob: str = SOURCE_DECLARATION_BLOB, expected_license_blob: str = "6634c8cc3133b3848ec74b9f275acaaa1ea618ab", expected_license_bytes: int = 11_343) -> None:
    validate_head(expected_head)
    required = {"schema", "status", "evidence_stage", "publication", "runtime_status", "cpu_status", "metal_status", "parity_status", "model_weights", "external_dependency", "upstream", "license", "chronology", "source_declaration", "vokra_checkout", "responses", "acquisition", "blockers"}
    _exact(report, required, "metadata report")
    if report["schema"] != SCHEMA or report["status"] != "BLOCKED" or report["evidence_stage"] != "INSPECTION_ONLY" or report["publication"] != "NO_UPLOAD" or report["runtime_status"] != "NOT_IMPLEMENTED_FAIL_CLOSED" or report["cpu_status"] != "UNSUPPORTED" or report["metal_status"] != "BLOCKED_BY_CPU" or report["parity_status"] != "NOT_RUN" or report["model_weights"] != "NOT_DOWNLOADED":
        raise AuditError("metadata report status is not fail-closed")
    if report["external_dependency"] != {"repository": REPOSITORY, "revision": REVISION, "selection_status": "AUTHENTICATED_METADATA_ONLY", "files": "NOT_DOWNLOADED", "model_weights": "NOT_DOWNLOADED"}:
        raise AuditError("external dependency identity/status drifted")
    if report["vokra_checkout"] != {"expected_head": expected_head, "actual_head": expected_head, "clean": True}:
        raise AuditError("Vokra checkout binding drifted")
    upstream = _exact(report["upstream"], {"repository", "fixed_revision", "fixed_head_date", "file_count", "files"}, "upstream")
    if upstream["repository"] != REPOSITORY or upstream["fixed_revision"] != REVISION or upstream["file_count"] != 14 or not isinstance(upstream["files"], list) or len(upstream["files"]) != 14:
        raise AuditError("upstream 14-file closure is not exact")
    by_path: dict[str, Any] = {}
    for row in upstream["files"]:
        path = row.get("path") if isinstance(row, dict) else None
        if path in by_path:
            raise AuditError("upstream file path is duplicated")
        by_path[path] = row
    if set(by_path) != EXPECTED_FILES:
        raise AuditError("upstream path set drifted")
    for path in sorted(EXPECTED_FILES):
        _validate_file_row(by_path[path], path, path in SHARD_FILES)
    _iso_date(upstream["fixed_head_date"], "Qwen fixed-head date")
    license_value = _exact(report["license"], {"card_license", "license_file_metadata", "license_bytes"}, "license")
    if license_value["card_license"] != "apache-2.0":
        raise AuditError("card license drifted")
    if license_value["license_file_metadata"] != by_path["LICENSE"]:
        raise AuditError("LICENSE metadata is not bound to the upstream row")
    license_bytes = _exact(license_value["license_bytes"], {"url", "path", "bytes", "git_blob_sha1", "sha256", "x_repo_commit", "apache_2_0", "status"}, "license bytes")
    if license_bytes["url"] != LICENSE_URL or license_bytes["path"] != "LICENSE" or license_bytes["bytes"] != expected_license_bytes or license_bytes["git_blob_sha1"] != expected_license_blob or license_bytes["x_repo_commit"] != REVISION or license_bytes["apache_2_0"] is not True or license_bytes["status"] != "AUTHENTICATED" or not HEX64.fullmatch(license_bytes["sha256"]):
        raise AuditError("LICENSE byte identity/status drifted")
    if license_value["license_file_metadata"]["git_blob_sha1"] != license_bytes["git_blob_sha1"] or license_value["license_file_metadata"]["git_blob_sha1"] != expected_license_blob:
        raise AuditError("LICENSE row is not bound to authenticated LICENSE bytes")
    chronology = _exact(report["chronology"], {"qwen_head_precedes_source_commit", "qwen_head_date", "source_commit_date", "current_head", "main_history"}, "chronology")
    if chronology["qwen_head_precedes_source_commit"] is not True or chronology["qwen_head_date"] != upstream["fixed_head_date"] or chronology["qwen_head_date"][:10] >= chronology["source_commit_date"][:10]:
        raise AuditError("chronology relation drifted")
    _iso_date(chronology["source_commit_date"], "source commit date")
    current = _exact(chronology["current_head"], {"current_head", "matches_fixed_revision", "scope"}, "current head")
    if current != {"current_head": REVISION, "matches_fixed_revision": True, "scope": "Hugging Face model-info current head only; other refs are not audited"}:
        raise AuditError("current-head scope drifted")
    history = _exact(chronology["main_history"], {"endpoint", "count", "head_matches_fixed", "all_precede_source", "commits", "scope"}, "main history")
    if history["endpoint"] != HISTORY_URL or history["count"] != 11 or history["head_matches_fixed"] is not True or history["all_precede_source"] is not True or history["scope"] != "official Hugging Face main branch only; other refs are not inferred" or not isinstance(history["commits"], list) or len(history["commits"]) != 11:
        raise AuditError("HF history proof drifted")
    ids: set[str] = set()
    for item in history["commits"]:
        commit = _exact(item, {"id", "created_at"}, "history commit")
        if not HEX40.fullmatch(commit["id"]) or commit["id"] in ids or commit["created_at"][:10] >= chronology["source_commit_date"][:10]:
            raise AuditError("HF history commit row is malformed")
        _iso_date(commit["created_at"], "history created_at"); ids.add(commit["id"])
    if history["commits"][0]["id"] != REVISION:
        raise AuditError("HF history head drifted")
    if any(history["commits"][index]["created_at"] < history["commits"][index + 1]["created_at"] for index in range(10)):
        raise AuditError("HF history order drifted")
    source = _exact(report["source_declaration"], {"repository", "revision", "path", "git_blob_sha1", "marker", "status", "source_commit_date"}, "source declaration")
    if source != {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "path": SOURCE_DECLARATION_PATH, "git_blob_sha1": expected_source_blob, "marker": SOURCE_DECLARATION_MARKER, "status": "AUTHENTICATED", "source_commit_date": chronology["source_commit_date"]}:
        raise AuditError("source declaration evidence drifted")
    expected_responses = ((MODEL_INFO_URL, MAX_MODEL_INFO_BYTES, True), (CURRENT_MODEL_INFO_URL, MAX_MODEL_INFO_BYTES, True), (HISTORY_URL, MAX_MODEL_INFO_BYTES, True), (GITHUB_SOURCE_COMMIT_URL, MAX_RESPONSE_BYTES, True), (GITHUB_SOURCE_TREE_URL, MAX_RESPONSE_BYTES, True), (GITHUB_SOURCE_BLOB_URL, MAX_SOURCE_BLOB_BYTES, True), (LICENSE_URL, 16 * 1024, False))
    if not isinstance(report["responses"], list) or len(report["responses"]) != len(expected_responses):
        raise AuditError("response evidence count/order drifted")
    for item, (url, limit, strict) in zip(report["responses"], expected_responses):
        value = _exact(item, {"url", "bytes", "sha256", "strict_json", "max_bytes"}, "response")
        if value["url"] != url or isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] <= 0 or value["bytes"] > limit or value["max_bytes"] != limit or value["strict_json"] is not strict or not HEX64.fullmatch(value["sha256"]):
            raise AuditError("response evidence row drifted")
    if report["responses"][-1]["bytes"] != license_bytes["bytes"] or report["responses"][-1]["sha256"] != license_bytes["sha256"]:
        raise AuditError("LICENSE response is not bound to authenticated LICENSE bytes")
    if report["acquisition"] != {"transport": "stdlib urllib HTTPS GET; redirects disabled; proxy disabled", "authorization_header": False, "ambient_token": False, "checkpoint_download": False, "file_resolve": False, "model_construction": False, "model_execution": False} or report["blockers"] != BLOCKERS:
        raise AuditError("acquisition/blocker contract drifted")


def _fake_runner(root: Path, head: str, status: str = "") -> Any:
    def runner(command: list[str], **_: Any) -> Any:
        args = tuple(command[3:])
        if args == ("rev-parse", "--verify", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=head, stderr="")
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return SimpleNamespace(returncode=0, stdout=status, stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected git command")
    return runner


def _fixture() -> tuple[Any, Any, Any, Any, Any, bytes, bytes, bytes, bytes, bytes]:
    rows = []
    for index, path in enumerate(sorted(EXPECTED_FILES), 1):
        size = 100 + index
        payload = f"{index:064x}"
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload}\nsize {size}\n".encode()
        if path in SHARD_FILES:
            rows.append({"rfilename": path, "size": size, "blobId": _git_blob_sha1(pointer), "lfs": {"sha256": payload, "size": size, "pointerSize": len(pointer)}})
        else:
            rows.append({"rfilename": path, "size": size, "blobId": _git_blob_sha1(path.encode()), "lfs": None})
    fixed = {"sha": REVISION, "lastModified": "2024-09-25T00:00:00Z", "cardData": {"license": "apache-2.0"}, "siblings": rows}
    current = {"sha": REVISION}
    source_content = SOURCE_DECLARATION_MARKER.encode()
    source_blob_sha = _git_blob_sha1(source_content)
    tree_sha = _git_blob_sha1(b"tree fixture")
    commit = {"sha": SOURCE_REVISION, "commit": {"committer": {"date": "2026-07-24T00:00:00Z"}, "tree": {"sha": tree_sha}}}
    tree = {"sha": tree_sha, "truncated": False, "tree": [{"path": SOURCE_DECLARATION_PATH, "type": "blob", "mode": "100644", "sha": source_blob_sha, "size": len(source_content)}]}
    blob = {"sha": source_blob_sha, "encoding": "base64", "size": len(source_content), "content": base64.b64encode(source_content).decode()}
    history = [{"id": REVISION if index == 0 else f"{index:040x}", "createdAt": f"2024-09-{25-index:02d}T00:00:00Z"} for index in range(11)]
    license_raw = (b"Apache License\nVersion 2.0\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n" + b"x" * 100)
    license_blob_sha = _git_blob_sha1(license_raw)
    next(row for row in rows if row["rfilename"] == "LICENSE")["blobId"] = license_blob_sha
    license_headers = {"etag": f'"{license_blob_sha}"', "x-repo-commit": REVISION}
    # Supply the same fixed role identity to the synthetic contract through the
    # real validator's constants only after checking its negative paths.
    raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob = [json.dumps(value, separators=(",", ":")).encode() for value in (fixed, current, history, commit, tree, blob)]
    return fixed, current, history, commit, tree, blob, license_raw, license_headers, raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob


def self_test() -> None:
    root = Path.cwd()
    head = "a" * 40
    checkout = verify_checkout(root, head, runner=_fake_runner(root, head))
    fixed, current, history, commit, tree, blob, license_raw, license_headers, raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob = _fixture()
    source_blob_sha = _git_blob_sha1(SOURCE_DECLARATION_MARKER.encode())
    license_blob_sha = _git_blob_sha1(license_raw)
    for field, value in (("sha", "0" * 40), ("truncated", True)):
        broken_tree = json.loads(json.dumps(tree)); broken_tree[field] = value
        try:
            validate_source_evidence(commit, broken_tree, blob, expected_blob=source_blob_sha)
        except AuditError:
            pass
        else:
            raise AssertionError(f"source tree {field} tamper was accepted")
    broken_tree = json.loads(json.dumps(tree)); broken_tree["tree"][0]["size"] += 1
    try:
        validate_source_evidence(commit, broken_tree, blob, expected_blob=source_blob_sha)
    except AuditError:
        pass
    else:
        raise AssertionError("source declaration size tamper was accepted")
    report = build_report(fixed, current, history, commit, tree, blob, license_raw, license_headers, checkout, raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
    validate_report(report, head, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
    saved_evidence_tamper_cases = (
        lambda value: next(row for row in value["upstream"]["files"] if row["path"] in SHARD_FILES).__setitem__("lfs_pointer_size", next(row for row in value["upstream"]["files"] if row["path"] in SHARD_FILES)["lfs_pointer_size"] + 1),
        lambda value: next(row for row in value["upstream"]["files"] if row["path"] == "LICENSE").__setitem__("git_blob_sha1", "0" * 40),
        lambda value: value["responses"][-1].__setitem__("bytes", value["responses"][-1]["bytes"] + 1),
    )
    for mutate in saved_evidence_tamper_cases:
        broken = json.loads(json.dumps(report)); mutate(broken)
        try:
            validate_report(broken, head, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
        except AuditError:
            pass
        else:
            raise AssertionError("saved evidence binding tamper was accepted")
    for candidate, runner in (("", _fake_runner(root, head)), ("b" * 40, _fake_runner(root, head)), (head, _fake_runner(root, head, " M unrelated"))):
        try:
            verify_checkout(root, candidate, runner=runner)
        except AuditError:
            pass
        else:
            raise AssertionError("HEAD/dirty checkout was accepted")
    for raw in (b'{"x":1,"x":2}', b"{"):
        try:
            strict_json(raw, "fixture")
        except AuditError:
            pass
        else:
            raise AssertionError("duplicate/malformed JSON was accepted")
    try:
        _read_bounded(io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1)), MAX_RESPONSE_BYTES)
    except AuditError:
        pass
    else:
        raise AssertionError("oversized response was accepted")
    for unsafe_url in (
        MODEL_INFO_URL.replace("https://", "http://", 1),
        MODEL_INFO_URL + "&extra=1",
        "https://example.invalid/qwen",
    ):
        try:
            fetch_json(unsafe_url)
        except AuditError:
            pass
        else:
            raise AssertionError("unsafe/non-fixed URL was accepted")
    tampered = json.loads(json.dumps(fixed)); tampered["siblings"] = tampered["siblings"][:-1]
    try:
        validate_model_info(tampered)
    except AuditError:
        pass
    else:
        raise AssertionError("omitted HF path was accepted")
    tampered = json.loads(json.dumps(fixed)); next(item for item in tampered["siblings"] if item["rfilename"] in SHARD_FILES)["lfs"]["pointerSize"] += 1
    try:
        validate_model_info(tampered)
    except AuditError:
        pass
    else:
        raise AssertionError("LFS pointer tamper was accepted")
    tampered = json.loads(json.dumps(fixed)); tampered["cardData"]["license"] = "mit"
    try:
        validate_model_info(tampered)
    except AuditError:
        pass
    else:
        raise AssertionError("license drift was accepted")
    tampered = json.loads(json.dumps(fixed)); tampered["lastModified"] = "2027-01-01T00:00:00Z"
    try:
        build_report(tampered, current, history, commit, tree, blob, license_raw, license_headers, checkout, raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
    except AuditError:
        pass
    else:
        raise AssertionError("chronology drift was accepted")
    tampered_report = json.loads(json.dumps(report)); tampered_report["publication"] = "UPLOAD"
    try:
        validate_report(tampered_report, head, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
    except AuditError:
        pass
    else:
        raise AssertionError("unsafe publication status was accepted")
    for mutate in (
        lambda value: value["upstream"]["files"].__setitem__(0, {**value["upstream"]["files"][0], "git_blob_sha1": "not-a-blob"}),
        lambda value: value["responses"].__setitem__(0, {**value["responses"][0], "url": CURRENT_MODEL_INFO_URL}),
        lambda value: value["chronology"]["main_history"]["commits"].__setitem__(0, {**value["chronology"]["main_history"]["commits"][0], "id": "0" * 40}),
    ):
        broken = json.loads(json.dumps(report)); mutate(broken)
        try:
            validate_report(broken, head, expected_source_blob=source_blob_sha, expected_license_blob=license_blob_sha, expected_license_bytes=len(license_raw))
        except AuditError:
            pass
        else:
            raise AssertionError("deep report tamper was accepted")
    try:
        main(["--self-test", "--self-test"])
    except SystemExit as error:
        if error.code != 2:
            raise AssertionError("duplicate CLI option did not fail with parser status 2")
    else:
        raise AssertionError("duplicate CLI option was accepted")
    inside = root / "qwen-metadata-self-test.json"
    try:
        safe_absent_output(root, inside)
    except AuditError:
        pass
    else:
        raise AssertionError("checkout-internal output was accepted")
    existing = Path(tempfile.mkdtemp(dir="/private/tmp")) / "evidence.json"
    try:
        safe_absent_output(root, existing)
        write_no_replace(existing, report)
        try:
            write_no_replace(existing, report)
        except AuditError:
            pass
        else:
            raise AssertionError("output replacement was accepted")
    finally:
        existing.unlink(missing_ok=True); existing.parent.rmdir()
    print("qwen_metadata_audit.py self-test: PASS (14-file/LFS/source/chronology fail-closed contract)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-evidence", action="store_true")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = sys.argv[1:] if argv is None else argv
    for option in ("--expected-head", "--repo-root", "--output", "--validate-evidence", "--evidence", "--self-test"):
        if arguments.count(option) > 1:
            parser.error(f"duplicate {option}")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.validate_evidence or args.evidence is not None or any(value is not None for value in (args.expected_head, args.repo_root, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test(); return 0
    if args.validate_evidence:
        if args.evidence is None or args.expected_head is None or args.repo_root is None or args.output is not None:
            parser.error("--validate-evidence requires --expected-head, --repo-root, and --evidence only")
        try:
            verify_checkout(args.repo_root, args.expected_head)
            evidence_path = safe_existing_evidence(args.repo_root, args.evidence)
            value = strict_json(evidence_path.read_bytes(), str(evidence_path))
            validate_report(value, args.expected_head)
        except (AuditError, OSError, subprocess.SubprocessError) as error:
            print(f"qwen_metadata_audit.py: evidence rejected: {error}", file=os.sys.stderr)
            return 2
        print(f"qwen_metadata_audit.py: evidence validated: {args.evidence}")
        return 0
    if args.evidence is not None or any(value is None for value in (args.expected_head, args.repo_root, args.output)):
        parser.error("normal runs require --expected-head, --repo-root, and --output")
    try:
        checkout = verify_checkout(args.repo_root, args.expected_head)
        output = safe_absent_output(args.repo_root, args.output)
        fixed, raw_fixed = fetch_json(MODEL_INFO_URL, limit=MAX_MODEL_INFO_BYTES)
        current, raw_current = fetch_json(CURRENT_MODEL_INFO_URL, limit=MAX_MODEL_INFO_BYTES)
        history, raw_history = fetch_json(HISTORY_URL, limit=MAX_MODEL_INFO_BYTES)
        source_commit, raw_commit = fetch_json(GITHUB_SOURCE_COMMIT_URL)
        source_tree, raw_tree = fetch_json(GITHUB_SOURCE_TREE_URL)
        source_blob, raw_blob = fetch_json(GITHUB_SOURCE_BLOB_URL, limit=MAX_SOURCE_BLOB_BYTES)
        license_raw, license_headers = fetch_bytes(LICENSE_URL, limit=16 * 1024, content_type="text/plain")
        report = build_report(fixed, current, history, source_commit, source_tree, source_blob, license_raw, license_headers, checkout, raw_fixed, raw_current, raw_history, raw_commit, raw_tree, raw_blob)
        validate_report(report, args.expected_head)
        write_no_replace(output, report)
    except (AuditError, OSError, subprocess.SubprocessError) as error:
        print(f"qwen_metadata_audit.py: BLOCKED: {error}", file=os.sys.stderr)
        return 2
    print(f"qwen_metadata_audit.py: BLOCKED/INSPECTION_ONLY/NO_UPLOAD; evidence={args.output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
