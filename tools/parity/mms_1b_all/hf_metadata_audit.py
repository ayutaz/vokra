#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Authenticate the MMS-1B-All eight-file closure without model bytes.

The normal path performs one bounded HTTPS request to the fixed Hugging Face
model-info endpoint. It uses only the standard-library urllib transport and
strict JSON decoding. It never resolves or downloads a file, imports a model
library, constructs a model, or executes inference. The resulting evidence is
deliberately blocked for owner review: server metadata is not publication
permission.
"""

from __future__ import annotations

import argparse
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
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPOSITORY = "facebook/mms-1b-all"
REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
WEIGHT_LICENSE = "cc-by-nc-4.0"
SCHEMA = "vokra-mms-1b-all-hf-metadata-evidence-v2"
LANGUAGE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_SIBLINGS = 4096
HF_HOST = "huggingface.co"
MODEL_INFO_URL = (
    f"https://{HF_HOST}/api/models/{REPOSITORY}"
    f"?revision={REVISION}&blobs=true"
)
# Independent fixed-endpoint spot-check (2026-09-08): the pinned response
# returned sha exactly 3d33597edbdaaba14a8e858e2c8caa76e3cec0cd, license
# cc-by-nc-4.0, 3606 sibling rows, and approximately 650030 compact-JSON
# characters; all eight route files were present for one explicit eng
# selection. This auditor never selects that language (or any language)
# implicitly.

ROLE_PATHS = {
    "config": "config.json",
    "preprocessor_config": "preprocessor_config.json",
    "tokenizer_config": "tokenizer_config.json",
    "vocabulary": "vocab.json",
    "special_tokens_map": "special_tokens_map.json",
    "backbone": "model.safetensors",
    "adapter": "adapter.{language}.safetensors",
    "language_vocabulary": "vocabs/{language}.txt",
}

REPORT_KEYS = {
    "schema",
    "status",
    "publication",
    "owner_review",
    "license_spdx",
    "upstream",
    "vokra_checkout",
    "language",
    "roles",
    "response",
    "acquisition",
    "blockers",
}
ROLE_KEYS = {
    "path",
    "size",
    "git_blob_sha1",
    "lfs_pointer_git_blob_sha1",
    "lfs_payload_sha256",
    "lfs_payload_size",
}
UPSTREAM_KEYS = {"repository", "requested_revision", "resolved_revision"}
CHECKOUT_KEYS = {"expected_head", "actual_head", "clean"}
RESPONSE_KEYS = {"url", "bytes", "max_bytes", "sha256", "strict_json"}
ACQUISITION_KEYS = {
    "transport",
    "authorization_header",
    "ambient_token",
    "checkpoint_download",
    "file_resolve",
    "model_construction",
    "model_execution",
}
EXPECTED_BLOCKERS = [
    "PENDING_OWNER_REVIEW",
    "NO_CHECKPOINT_ACQUIRED",
    "NATIVE_CPU_METAL_AND_PARITY_NOT_RUN",
]


class AuditError(ValueError):
    """A fail-closed metadata audit error."""


def strict_json_load(data: bytes) -> Any:
    """Decode raw API bytes while rejecting duplicate object keys."""

    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError, AuditError) as error:
        raise AuditError(f"strict JSON decode failed: {error}") from error


def validate_language(language: Any) -> str:
    if not isinstance(language, str) or not LANGUAGE.fullmatch(language):
        raise AuditError(
            "--language must be one explicit lowercase adapter code "
            "using only letters, digits, '-' or '_'"
        )
    return language


def validate_head(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise AuditError(f"{label} must be exactly 40 lowercase hexadecimal characters")
    return value


def role_paths(language: str) -> dict[str, str]:
    language = validate_language(language)
    return {role: path.format(language=language) for role, path in ROLE_PATHS.items()}


def _safe_existing_directory(path: Path, label: str) -> Path:
    raw = str(path)
    if (
        not path.is_absolute()
        or "\x00" in raw
        or "\\" in raw
        or "//" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
    ):
        raise AuditError(f"{label} must be an absolute lexical path without traversal")
    if path.is_symlink() or not path.is_dir():
        raise AuditError(f"{label} is not a regular directory: {path}")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise AuditError(f"{label} has symlinked ancestry: {path}")
    return path


def _safe_absent_output(path: Path) -> Path:
    raw = str(path)
    if (
        not path.is_absolute()
        or "\x00" in raw
        or "\\" in raw
        or "//" in raw
        or any(part in {"", ".", ".."} for part in raw.split("/")[1:])
        or path.name in {"", ".", ".."}
    ):
        raise AuditError(f"output must be an absolute lexical path without traversal: {path}")
    if path.exists() or path.is_symlink():
        raise AuditError(f"output already exists; refusing replacement: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise AuditError(f"output parent is not a regular directory: {parent}")
    for ancestor in (parent, *parent.parents):
        if ancestor.is_symlink():
            raise AuditError(f"output has symlinked ancestry: {path}")
    return path


def ensure_output_disjoint(repo_root: Path, output: Path) -> None:
    """Reject an evidence file that would dirty the checkout it binds."""

    root = _safe_existing_directory(repo_root, "repository root").resolve()
    candidate = output.parent.resolve() / output.name
    if candidate == root or root in candidate.parents:
        raise AuditError("metadata output must be outside the bound Vokra checkout")


def write_no_replace(path: Path, report: dict[str, Any]) -> None:
    path = _safe_absent_output(path)
    encoded = (
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditError(
                f"output appeared during audit; refusing replacement: {path}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_checkout(
    repo_root: Path,
    expected_head: str,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Verify the local Vokra tree before opening any network connection."""

    expected = validate_head(expected_head, "expected_head")
    root = _safe_existing_directory(repo_root, "repository root")

    def git(*args: str) -> str:
        result = runner(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AuditError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    top = Path(git("rev-parse", "--show-toplevel"))
    if top.resolve() != root.resolve():
        raise AuditError(f"git root mismatch: {top} != {root}")
    actual = validate_head(git("rev-parse", "--verify", "HEAD"), "actual_head")
    clean = git("status", "--porcelain", "--untracked-files=all") == ""
    if actual != expected:
        raise AuditError(f"checkout HEAD {actual} differs from expected {expected}")
    if not clean:
        raise AuditError("Vokra checkout is dirty; metadata evidence requires a clean tree")
    return {"expected_head": expected, "actual_head": actual, "clean": True}


def _validate_hf_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        expected = urlsplit(MODEL_INFO_URL)
        port = parsed.port
    except ValueError as error:
        raise AuditError(f"HF response URL is malformed: {url}") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != HF_HOST
        or port is not None
        or parsed.path != expected.path
        or parsed.query != expected.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise AuditError(f"HF response URL is not the fixed HTTPS endpoint: {url}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
        raise AuditError(f"HF metadata endpoint redirected to an untrusted URL: {newurl}")


def _read_bounded(stream: BinaryIO) -> bytes:
    data = bytearray()
    while len(data) <= MAX_RESPONSE_BYTES:
        chunk = stream.read(min(64 * 1024, MAX_RESPONSE_BYTES + 1 - len(data)))
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > MAX_RESPONSE_BYTES:
            raise AuditError(f"HF metadata response exceeds {MAX_RESPONSE_BYTES} bytes")
    raise AuditError(f"HF metadata response exceeds {MAX_RESPONSE_BYTES} bytes")


def build_metadata_request() -> Request:
    request = Request(
        MODEL_INFO_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "vokra-mms-1b-all-metadata-audit/1",
        },
        method="GET",
    )
    if any(key.casefold() == "authorization" for key, _ in request.header_items()):
        raise AuditError("metadata request unexpectedly carries Authorization")
    return request


def fetch_model_info() -> tuple[dict[str, Any], bytes]:
    """Fetch and strictly decode one bounded raw model-info response."""

    request = build_metadata_request()
    try:
        opener = build_opener(_NoRedirect)
        with opener.open(request, timeout=30.0) as response:
            final_url = response.geturl()
            _validate_hf_url(final_url)
            if response.getcode() != 200:
                raise AuditError(f"HF metadata endpoint returned HTTP {response.getcode()}")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
            if content_type != "application/json":
                raise AuditError(f"HF metadata content type is not JSON: {content_type!r}")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0 or declared > MAX_RESPONSE_BYTES:
                        raise AuditError("HF metadata Content-Length exceeds hard response bound")
                except ValueError as error:
                    raise AuditError("HF metadata Content-Length is malformed") from error
            raw = _read_bounded(response)
    except AuditError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        raise AuditError(f"HF metadata request failed: {error}") from error
    value = strict_json_load(raw)
    if not isinstance(value, dict):
        raise AuditError("HF model-info response is not an object")
    return value, raw


def _sha1_git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _validate_file_metadata(item: Any, expected_path: str) -> dict[str, Any]:
    """Validate one raw siblings row and retain only its file identity."""

    if not isinstance(item, dict):
        raise AuditError(f"server file metadata is not an object: {expected_path}")
    path = item.get("rfilename")
    if (
        path != expected_path
        or not isinstance(path, str)
        or not path
        or "\\" in path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise AuditError(f"server path drifted for required role: {path!r}")
    size = item.get("size")
    blob_id = item.get("blobId")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise AuditError(f"server size is not a positive integer: {expected_path}")
    if not isinstance(blob_id, str) or not HEX40.fullmatch(blob_id):
        raise AuditError(f"server Git blob identity is malformed: {expected_path}")

    lfs = item.get("lfs")
    if lfs is None:
        return {
            "path": path,
            "size": size,
            "git_blob_sha1": blob_id,
            "lfs_pointer_git_blob_sha1": None,
            "lfs_payload_sha256": None,
            "lfs_payload_size": None,
        }
    if not isinstance(lfs, dict):
        raise AuditError(f"server LFS identity is not an object: {expected_path}")
    lfs_sha = lfs.get("sha256")
    lfs_size = lfs.get("size")
    if not isinstance(lfs_sha, str) or not HEX64.fullmatch(lfs_sha):
        raise AuditError(f"server LFS payload digest is malformed: {expected_path}")
    if isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size <= 0 or lfs_size != size:
        raise AuditError(f"server LFS payload size is not bound to file size: {expected_path}")
    pointer_size = lfs.get("pointerSize")
    if pointer_size is not None and (
        isinstance(pointer_size, bool) or not isinstance(pointer_size, int) or pointer_size <= 0
    ):
        raise AuditError(f"server LFS pointer size is malformed: {expected_path}")
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {lfs_size}\n".encode()
    if pointer_size is not None and pointer_size != len(pointer):
        raise AuditError(f"server LFS pointer size does not match reconstruction: {expected_path}")
    if _sha1_git_blob(pointer) != blob_id:
        raise AuditError(f"server LFS pointer Git blob mismatch: {expected_path}")
    return {
        "path": path,
        "size": size,
        "git_blob_sha1": None,
        "lfs_pointer_git_blob_sha1": blob_id,
        "lfs_payload_sha256": lfs_sha,
        "lfs_payload_size": lfs_size,
    }


def select_roles(payload: dict[str, Any], language: str) -> dict[str, dict[str, Any]]:
    language = validate_language(language)
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or len(siblings) > MAX_SIBLINGS:
        raise AuditError(f"HF siblings response is missing or exceeds {MAX_SIBLINGS} entries")
    paths = role_paths(language)
    by_path = {path: role for role, path in paths.items()}
    selected: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for sibling in siblings:
        if not isinstance(sibling, dict):
            raise AuditError("HF siblings contains a non-object entry")
        path = sibling.get("rfilename")
        if not isinstance(path, str) or path in seen_paths:
            raise AuditError(f"HF siblings contains an unsafe or duplicate path: {path!r}")
        seen_paths.add(path)
        role = by_path.get(path)
        if role is not None:
            if role in selected:
                raise AuditError(f"required role appears more than once: {role}")
            selected[role] = _validate_file_metadata(sibling, path)
    if set(selected) != set(paths):
        missing = sorted(set(paths) - set(selected))
        raise AuditError(f"eight-file MMS snapshot closure is incomplete: missing {missing}")
    return selected


def build_report(
    payload: dict[str, Any],
    raw: bytes,
    language: str,
    checkout: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("sha") != REVISION:
        raise AuditError(f"HF revision drift: {payload.get('sha')!r} != {REVISION}")
    card_data = payload.get("cardData")
    if not isinstance(card_data, dict) or card_data.get("license") != WEIGHT_LICENSE:
        raise AuditError("HF cardData license is not the fixed MMS noncommercial license")
    return {
        "schema": SCHEMA,
        "status": "BLOCKED_PENDING_OWNER_REVIEW",
        "publication": "NO_UPLOAD",
        "owner_review": "PENDING_OWNER_REVIEW",
        "license_spdx": WEIGHT_LICENSE,
        "upstream": {
            "repository": REPOSITORY,
            "requested_revision": REVISION,
            "resolved_revision": payload["sha"],
        },
        "vokra_checkout": checkout,
        "language": validate_language(language),
        "roles": select_roles(payload, language),
        "response": {
            "url": MODEL_INFO_URL,
            "bytes": len(raw),
            "max_bytes": MAX_RESPONSE_BYTES,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "strict_json": True,
        },
        "acquisition": {
            "transport": "stdlib urllib HTTPS GET model-info?revision=<pinned>&blobs=true",
            "authorization_header": False,
            "ambient_token": False,
            "checkpoint_download": False,
            "file_resolve": False,
            "model_construction": False,
            "model_execution": False,
        },
        "blockers": EXPECTED_BLOCKERS.copy(),
    }


def _validate_role_row(value: Any, expected_path: str) -> None:
    if not isinstance(value, dict) or set(value) != ROLE_KEYS:
        raise AuditError(f"role metadata schema is not exact: {expected_path}")
    if value["path"] != expected_path:
        raise AuditError(f"role path does not match selected language: {expected_path}")
    if isinstance(value["size"], bool) or not isinstance(value["size"], int) or value["size"] <= 0:
        raise AuditError(f"role size is malformed: {expected_path}")
    for key in ("git_blob_sha1", "lfs_pointer_git_blob_sha1"):
        if value[key] is not None and (not isinstance(value[key], str) or not HEX40.fullmatch(value[key])):
            raise AuditError(f"role Git blob identity is malformed: {expected_path}")
    if value["lfs_payload_sha256"] is not None and (
        not isinstance(value["lfs_payload_sha256"], str) or not HEX64.fullmatch(value["lfs_payload_sha256"])
    ):
        raise AuditError(f"role LFS digest is malformed: {expected_path}")
    if value["lfs_payload_size"] is not None and (
        isinstance(value["lfs_payload_size"], bool)
        or not isinstance(value["lfs_payload_size"], int)
        or value["lfs_payload_size"] != value["size"]
    ):
        raise AuditError(f"role LFS size is not bound: {expected_path}")
    regular = value["git_blob_sha1"] is not None
    lfs = value["lfs_pointer_git_blob_sha1"] is not None
    if regular == lfs:
        raise AuditError(f"role must have exactly one regular/LFS identity: {expected_path}")
    if regular and (value["lfs_payload_sha256"] is not None or value["lfs_payload_size"] is not None):
        raise AuditError(f"regular role unexpectedly carries LFS payload metadata: {expected_path}")
    if lfs and (value["lfs_payload_sha256"] is None or value["lfs_payload_size"] is None):
        raise AuditError(f"LFS role is missing payload metadata: {expected_path}")


def validate_report(value: Any, language: str, expected_head: str) -> None:
    language = validate_language(language)
    expected = validate_head(expected_head, "expected_head")
    if not isinstance(value, dict) or set(value) != REPORT_KEYS:
        raise AuditError("metadata report schema is not exact")
    if (
        value["schema"] != SCHEMA
        or value["status"] != "BLOCKED_PENDING_OWNER_REVIEW"
        or value["publication"] != "NO_UPLOAD"
        or value["owner_review"] != "PENDING_OWNER_REVIEW"
        or value["license_spdx"] != WEIGHT_LICENSE
    ):
        raise AuditError("metadata report status/license is not fail-closed")
    if value["language"] != language:
        raise AuditError("metadata report language differs from explicit language")
    upstream = value["upstream"]
    if not isinstance(upstream, dict) or set(upstream) != UPSTREAM_KEYS or upstream != {
        "repository": REPOSITORY,
        "requested_revision": REVISION,
        "resolved_revision": REVISION,
    }:
        raise AuditError("metadata report upstream identity is not exact")
    checkout = value["vokra_checkout"]
    if not isinstance(checkout, dict) or set(checkout) != CHECKOUT_KEYS or checkout != {
        "expected_head": expected,
        "actual_head": expected,
        "clean": True,
    }:
        raise AuditError("metadata report Vokra checkout binding is not exact")
    roles = value["roles"]
    paths = role_paths(language)
    if not isinstance(roles, dict) or set(roles) != set(paths):
        raise AuditError("metadata report eight-file role set is not exact")
    for role, path in paths.items():
        _validate_role_row(roles[role], path)
    response = value["response"]
    if not isinstance(response, dict) or set(response) != RESPONSE_KEYS:
        raise AuditError("metadata response schema is not exact")
    if (
        response["url"] != MODEL_INFO_URL
        or isinstance(response["bytes"], bool)
        or not isinstance(response["bytes"], int)
        or response["bytes"] <= 0
        or response["bytes"] > MAX_RESPONSE_BYTES
        or response["max_bytes"] != MAX_RESPONSE_BYTES
        or not isinstance(response["sha256"], str)
        or not HEX64.fullmatch(response["sha256"])
        or response["strict_json"] is not True
    ):
        raise AuditError("metadata response bound is not exact")
    acquisition = value["acquisition"]
    if not isinstance(acquisition, dict) or set(acquisition) != ACQUISITION_KEYS or acquisition != {
        "transport": "stdlib urllib HTTPS GET model-info?revision=<pinned>&blobs=true",
        "authorization_header": False,
        "ambient_token": False,
        "checkpoint_download": False,
        "file_resolve": False,
        "model_construction": False,
        "model_execution": False,
    }:
        raise AuditError("metadata acquisition boundary is not exact")
    if value["blockers"] != EXPECTED_BLOCKERS:
        raise AuditError("metadata report blockers are not exact")


def _synthetic_payload(language: str) -> tuple[dict[str, Any], bytes]:
    files: list[dict[str, Any]] = []
    for index, path in enumerate(role_paths(language).values(), start=1):
        size = 100 + index
        payload_sha = f"{index:064x}"
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha}\nsize {size}\n".encode()
        files.append(
            {
                "rfilename": path,
                "size": size,
                "blobId": _sha1_git_blob(pointer),
                "lfs": {"sha256": payload_sha, "size": size, "pointerSize": len(pointer)},
            }
        )
    payload = {"sha": REVISION, "cardData": {"license": WEIGHT_LICENSE}, "siblings": files}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, raw


def _fake_runner(root: Path, actual_head: str, status: str = "") -> Any:
    def runner(command: list[str], **_: Any) -> Any:
        args = tuple(command[3:])
        if args == ("rev-parse", "--show-toplevel"):
            return SimpleNamespace(returncode=0, stdout=str(root), stderr="")
        if args == ("rev-parse", "--verify", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=actual_head, stderr="")
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return SimpleNamespace(returncode=0, stdout=status, stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected git command")

    return runner


def self_test() -> None:
    language = "fra"
    expected = "a" * 40
    root = Path.cwd()
    checkout = verify_checkout(root, expected, runner=_fake_runner(root, expected))
    payload, raw = _synthetic_payload(language)
    report = build_report(payload, raw, language, checkout)
    validate_report(report, language, expected)
    request = build_metadata_request()
    if any(key.casefold() == "authorization" for key, _ in request.header_items()):
        raise SystemExit("self-test accepted ambient Authorization header")

    # Missing/drifted checkout HEAD and dirty state must fail before metadata.
    for candidate, runner in (
        ("", _fake_runner(root, expected)),
        ("b" * 40, _fake_runner(root, expected)),
        (expected, _fake_runner(root, expected, " M unrelated")),
    ):
        try:
            verify_checkout(root, candidate, runner=runner)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted missing/drifted or dirty checkout")

    # The full eight-role closure rejects omission and cross-role tampering.
    missing = json.loads(json.dumps(payload))
    missing["siblings"] = missing["siblings"][1:]
    try:
        select_roles(missing, language)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted sidecar omission")
    tampered = json.loads(json.dumps(report))
    tampered["roles"]["adapter"]["path"] = "model.safetensors"
    try:
        validate_report(tampered, language, expected)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted sidecar path tamper")
    pointer_tamper = json.loads(json.dumps(payload))
    pointer_tamper["siblings"][0]["lfs"]["pointerSize"] += 1
    try:
        select_roles(pointer_tamper, language)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted LFS pointer-size tamper")

    inside = root / "mms-metadata-self-test-inside.json"
    try:
        ensure_output_disjoint(root, inside)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted output inside bound checkout")

    # Strict parsing is applied to the actual raw response; the byte cap is
    # exercised independently with a stream larger than the hard bound.
    try:
        strict_json_load(b'{"path": 1, "path": 2}')
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted duplicate raw JSON key")
    try:
        _read_bounded(io.BytesIO(b"x" * (MAX_RESPONSE_BYTES + 1)))
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted oversized raw response")

    with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
        output = Path(temporary) / "metadata.json"
        write_no_replace(output, report)
        try:
            write_no_replace(output, report)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted output replacement")
    print("hf_metadata_audit.py self-test: PASS (8-role synthetic pass/tamper, NO_UPLOAD)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language")
    parser.add_argument("--expected-head")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    for option in ("--language", "--expected-head", "--repo-root", "--output", "--self-test"):
        if sys.argv[1:].count(option) > 1:
            parser.error(f"duplicate {option}")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.language, args.expected_head, args.repo_root, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.language, args.expected_head, args.repo_root, args.output)):
        parser.error("normal runs require --language, --expected-head, --repo-root, and --output")
    try:
        # This local git check intentionally precedes urllib setup/network.
        checkout = verify_checkout(args.repo_root, args.expected_head)
        output = _safe_absent_output(args.output)
        ensure_output_disjoint(args.repo_root, output)
        payload, raw = fetch_model_info()
        report = build_report(payload, raw, args.language, checkout)
        validate_report(report, args.language, args.expected_head)
        write_no_replace(output, report)
    except (AuditError, OSError, subprocess.SubprocessError) as error:
        print(f"hf_metadata_audit.py: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"hf_metadata_audit.py: BLOCKED_PENDING_OWNER_REVIEW; evidence={output}; NO_UPLOAD")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
