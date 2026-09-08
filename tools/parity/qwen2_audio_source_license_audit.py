#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Reproduce the Qwen2-Audio source-license absence decision.

This oracle uses only the public GitHub API.  It never clones the upstream
repository, downloads model files, resolves Python dependencies, constructs a
model, or executes inference.  The fixed execution source is the immutable
``SOURCE_REVISION`` and every reachable parent commit (including merge
parents) is inspected by its Git tree object.  A successful audit is still a
``SOURCE_LICENSE_UNKNOWN_BLOCKER``: README prose, a repository metadata field,
or a license found on an unrelated commit cannot authorize the source.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import contextlib
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SOURCE_REPOSITORY = "https://github.com/QwenLM/Qwen2-Audio"
SOURCE_REPOSITORY_PATH = "QwenLM/Qwen2-Audio"
SOURCE_REVISION = "595360e82b5839c1507492ec83cae5bda6d5c7d4"
API_ROOT = "https://api.github.com"
FORMAT = "vokra-qwen2-audio-source-license-history-v1"
BLOCKER = "SOURCE_LICENSE_UNKNOWN_BLOCKER"
INCOMPLETE_BLOCKER = "SOURCE_LICENSE_AUDIT_INCOMPLETE_BLOCKER"
DEFAULT_TIP_BLOCKER = "SOURCE_DEFAULT_BRANCH_TIP_DRIFT_BLOCKER"
BRANCH_SET_BLOCKER = "SOURCE_PUBLIC_BRANCH_SET_DRIFT_BLOCKER"
HEAD_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MAX_HISTORY_COMMITS = 256
MAX_TREE_ENTRIES = 100_000
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
FIXED_TREE_BLOB_COUNT = 19
REQUIRED_ROLE_PATHS = (
    "README.md",
    "README_CN.md",
    "demo/demo.sh",
    "demo/requirements_web_demo.txt",
    "demo/web_demo_audio.py",
    "eval_audio/EVALUATION.md",
    "eval_audio/cn_tn.py",
    "eval_audio/evaluate_asr.py",
    "eval_audio/evaluate_chat.py",
    "eval_audio/evaluate_emotion.py",
    "eval_audio/evaluate_st.py",
    "eval_audio/evaluate_tokenizer.py",
    "eval_audio/evaluate_vocal_sound.py",
    "eval_audio/whisper_normalizer/basic.py",
    "eval_audio/whisper_normalizer/english.json",
    "eval_audio/whisper_normalizer/english.py",
)
LICENSE_FILE_NAMES = frozenset(
    {
        "license",
        "license.md",
        "license.txt",
        "licence",
        "licence.md",
        "licence.txt",
        "copying",
        "copying.md",
        "copying.txt",
        "notice",
        "notice.md",
        "notice.txt",
        "eula",
        "eula.md",
        "eula.txt",
        "copyright",
        "copyright.md",
        "copyright.txt",
    }
)

REPORT_KEYS = frozenset(
    {
        "format",
        "status",
        "publication",
        "execution",
        "checkout",
        "source",
        "repository_metadata",
        "releases_and_tags",
        "readme",
        "license_decision",
    }
)
CHECKOUT_KEYS = frozenset({"expected_head", "actual_head", "clean"})
SOURCE_KEYS = frozenset(
    {
        "repository",
        "revision",
        "history_commit_count",
        "history_scope",
        "license_filename_policy",
        "history_commits",
        "fixed_tree_blob_count",
        "fixed_tree_license_like_files",
        "fixed_role_files",
    }
)
REPOSITORY_KEYS = frozenset({"full_name", "default_branch", "default_branch_tip", "default_branch_tip_matches_fixed", "branches", "license"})
RELEASE_KEYS = frozenset({"releases", "tags", "exact_revision_releases", "exact_revision_tags"})
README_KEYS = frozenset({"path", "bytes", "sha256", "license_related_lines"})
DECISION_KEYS = frozenset({"source_license", "status", "reason", "historical_license_candidates", "equivalent_license_bearing_revision"})


class CheckoutError(RuntimeError):
    def __init__(self, message: str, binding: dict[str, Any]):
        super().__init__(message)
        self.binding = binding


class _UniqueValue(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            raise argparse.ArgumentError(self, f"{option_string} may be specified only once")
        setattr(namespace, self.dest, values)


class _UniqueFlag(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, values: Any, option_string: str | None = None) -> None:
        if getattr(namespace, self.dest, False):
            raise argparse.ArgumentError(self, f"{option_string} may be specified only once")
        setattr(namespace, self.dest, True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()


def checkout_binding(expected_head: str) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "expected_head": expected_head if isinstance(expected_head, str) and HEAD_PATTERN.fullmatch(expected_head) else None,
        "actual_head": None,
        "clean": False,
    }
    if not isinstance(expected_head, str) or not HEAD_PATTERN.fullmatch(expected_head):
        raise CheckoutError("expected_head must be lowercase HEX40", binding)
    root = repo_root()
    try:
        actual = git_output(root, "rev-parse", "HEAD")
        dirty = git_output(root, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise CheckoutError(f"could not inspect Vokra checkout: {error}", binding) from error
    binding["actual_head"] = actual
    binding["clean"] = not bool(dirty)
    if not HEAD_PATTERN.fullmatch(actual) or actual != expected_head:
        raise CheckoutError("checkout HEAD does not match --expected-head", binding)
    if dirty:
        raise CheckoutError("checkout must be clean before API/network access", binding)
    return binding


def validate_output_path(path: Path) -> Path:
    """Return a safe absent output path; never create its parent."""
    if not isinstance(path, Path) or not path.is_absolute() or path.parts[0] != "/" or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeError("output must be an absolute dot-free path")
    if path.exists() or path.is_symlink():
        raise RuntimeError("output target must be absent")
    current = Path("/")
    for part in path.parts[1:-1]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError("output parent must be an existing real directory")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("output parent must be an existing real directory")
    root = repo_root()
    resolved = path.resolve(strict=False)
    if resolved == root or root in resolved.parents:
        raise RuntimeError("output must be outside the checkout")
    return path


class _RejectRedirect(HTTPRedirectHandler):
    """The public API contract permits no redirects at all."""

    def redirect_request(self, request: Request, *args: Any) -> Request | None:
        raise RuntimeError(f"GitHub API redirect is forbidden: {args[-1]}")


_API_OPENER = build_opener(_RejectRedirect())


def open_api(request: Request, timeout: int):
    return _API_OPENER.open(request, timeout=timeout)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json(raw: bytes, source: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid JSON from {source}: {error}") from error


def api_url(path: str, **params: str | int) -> str:
    if not path.startswith("/") or ".." in path.split("/"):
        raise RuntimeError(f"unsafe GitHub API path: {path}")
    query = urlencode(params)
    return f"{API_ROOT}{path}" + (f"?{query}" if query else "")


def fetch_json(url: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "vokra-qwen2-audio-source-license-audit/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with open_api(request, timeout=30) as response:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"GitHub API returned HTTP status {response.status}: {url}")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except Exception as error:
        raise RuntimeError(f"GitHub API request failed: {url}: {error}") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"GitHub API response exceeds {MAX_RESPONSE_BYTES} bytes: {url}")
    return strict_json(raw, url)


def paged(path: str) -> list[Any]:
    values: list[Any] = []
    for page in range(1, 101):
        payload = fetch_json(api_url(path, per_page=100, page=page))
        if not isinstance(payload, list):
            raise RuntimeError(f"GitHub API page is not a list: {path}")
        values.extend(payload)
        if len(payload) < 100:
            return values
    raise RuntimeError(f"GitHub API pagination exceeded 100 pages: {path}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def license_like(path: str) -> bool:
    basename = path.rsplit("/", 1)[-1].lower()
    return basename in LICENSE_FILE_NAMES or basename.startswith(
        (
            "license-",
            "license.",
            "licence-",
            "licence.",
            "copying-",
            "copying.",
            "notice-",
            "notice.",
            "eula-",
            "eula.",
            "copyright-",
            "copyright.",
        )
    )


def safe_relative_path(path: Any, label: str) -> str:
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\x00" in path or ".." in path.split("/"):
        raise RuntimeError(f"{label} must be a safe relative path")
    return path


def object_record(entry: Any, *, require_size: bool = False) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        raise RuntimeError("GitHub tree entry is not an object")
    path, kind, oid = entry.get("path"), entry.get("type"), entry.get("sha")
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path or path.startswith("/") or ".." in path.split("/"):
        raise RuntimeError(f"unsafe GitHub tree path: {path!r}")
    if kind == "tree":
        return None
    if kind != "blob" or not isinstance(oid, str) or len(oid) != 40 or any(character not in "0123456789abcdef" for character in oid):
        raise RuntimeError(f"malformed GitHub tree blob: {path!r}")
    size = entry.get("size")
    if require_size and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise RuntimeError(f"GitHub tree blob size is missing: {path!r}")
    return {"path": path, "sha": oid, "size": size}


def fetch_commit(sha: str) -> dict[str, Any]:
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError(f"invalid commit SHA: {sha!r}")
    payload = fetch_json(api_url(f"/repos/{SOURCE_REPOSITORY_PATH}/commits/{sha}"))
    if not isinstance(payload, dict) or payload.get("sha") != sha:
        raise RuntimeError(f"GitHub commit identity mismatch: {sha}")
    commit = payload.get("commit")
    tree = commit.get("tree") if isinstance(commit, dict) else None
    tree_sha = tree.get("sha") if isinstance(tree, dict) else None
    parents = payload.get("parents")
    if not isinstance(tree_sha, str) or len(tree_sha) != 40 or not isinstance(parents, list):
        raise RuntimeError(f"GitHub commit envelope malformed: {sha}")
    parent_shas: list[str] = []
    for parent in parents:
        if not isinstance(parent, dict) or not isinstance(parent.get("sha"), str):
            raise RuntimeError(f"GitHub commit parent malformed: {sha}")
        parent_shas.append(parent["sha"])
    return {
        "sha": sha,
        "tree": tree_sha,
        "parents": parent_shas,
        "date": commit.get("author", {}).get("date") if isinstance(commit.get("author"), dict) else None,
        "subject": commit.get("message", "").split("\n", 1)[0] if isinstance(commit.get("message"), str) else None,
    }


def fetch_tree(tree_sha: str) -> list[dict[str, Any]]:
    if len(tree_sha) != 40 or any(character not in "0123456789abcdef" for character in tree_sha):
        raise RuntimeError(f"invalid tree SHA: {tree_sha!r}")
    payload = fetch_json(api_url(f"/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{tree_sha}", recursive=1))
    if not isinstance(payload, dict) or payload.get("sha") != tree_sha or payload.get("truncated") is not False:
        raise RuntimeError(f"GitHub tree identity/truncation mismatch: {tree_sha}")
    entries = payload.get("tree")
    if not isinstance(entries, list) or len(entries) > MAX_TREE_ENTRIES:
        raise RuntimeError(f"GitHub tree entry count is invalid: {tree_sha}")
    records = [record for entry in entries if (record := object_record(entry, require_size=True)) is not None]
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"GitHub tree contains duplicate paths: {tree_sha}")
    return sorted(records, key=lambda record: record["path"])


def reachable_history() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    queue: deque[str] = deque([SOURCE_REVISION])
    commits: dict[str, dict[str, Any]] = {}
    trees: dict[str, list[dict[str, Any]]] = {}
    while queue:
        sha = queue.popleft()
        if sha in commits:
            continue
        if len(commits) >= MAX_HISTORY_COMMITS:
            raise RuntimeError("source commit history exceeded safety bound")
        commit = fetch_commit(sha)
        commits[sha] = commit
        trees[sha] = fetch_tree(commit["tree"])
        queue.extend(commit["parents"])
    ordered = sorted(commits.values(), key=lambda row: (row.get("date") or "", row["sha"]))
    return ordered, trees


def tree_roles(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_path = {record["path"]: record for record in records}
    missing = [path for path in REQUIRED_ROLE_PATHS if path not in by_path]
    if missing:
        raise RuntimeError(f"fixed source role files are missing: {missing}")
    # This is intentionally an exact, audited role set.  An arbitrary subset
    # of the demo/evaluation tree cannot establish equivalence to the fixed
    # execution source.
    return {path: by_path[path] for path in REQUIRED_ROLE_PATHS}


def read_blob(blob_sha: str) -> bytes:
    payload = fetch_json(api_url(f"/repos/{SOURCE_REPOSITORY_PATH}/git/blobs/{blob_sha}"))
    if not isinstance(payload, dict) or payload.get("sha") != blob_sha or payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise RuntimeError(f"GitHub blob envelope mismatch: {blob_sha}")
    try:
        # The GitHub API wraps base64 content at a fixed column width.  Strip
        # ASCII whitespace before strict alphabet/padding validation.
        encoded = "".join(payload["content"].split())
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError, TypeError) as error:
        raise RuntimeError(f"GitHub blob is not valid base64: {blob_sha}") from error
    if git_blob_sha1(content) != blob_sha:
        raise RuntimeError(f"GitHub blob SHA-1 mismatch: {blob_sha}")
    return content


def readme_evidence(readme: dict[str, Any]) -> dict[str, Any]:
    content = read_blob(readme["sha"])
    lines = content.decode("utf-8", errors="replace").splitlines()
    selected = [
        {"line": index, "text": line}
        for index, line in enumerate(lines, 1)
        if any(term in line.lower() for term in ("license", "apache", "mit", "bsd", "copyright", "commercial"))
    ]
    return {"path": "README.md", "bytes": len(content), "sha256": sha256_bytes(content), "license_related_lines": selected}


def repository_evidence() -> dict[str, Any]:
    payload = fetch_json(api_url(f"/repos/{SOURCE_REPOSITORY_PATH}"))
    if not isinstance(payload, dict) or payload.get("full_name") != SOURCE_REPOSITORY_PATH:
        raise RuntimeError("official repository identity mismatch")
    default_branch = payload.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch or "/" in default_branch or ".." in default_branch:
        raise RuntimeError(f"{DEFAULT_TIP_BLOCKER}: official default branch is malformed: {default_branch!r}")
    branches = paged(f"/repos/{SOURCE_REPOSITORY_PATH}/branches")
    branch_records = []
    for branch in branches:
        if not isinstance(branch, dict) or not isinstance(branch.get("name"), str) or not isinstance(branch.get("commit"), dict) or not isinstance(branch["commit"].get("sha"), str):
            raise RuntimeError(f"{BRANCH_SET_BLOCKER}: official branch entry is malformed")
        name = branch["name"]
        sha = branch["commit"]["sha"]
        if not name or "\x00" in name or "\\" in name or ".." in name.split("/"):
            raise RuntimeError(f"{BRANCH_SET_BLOCKER}: official branch name is unsafe: {name!r}")
        _sha(sha, 40, "official branch tip")
        branch_records.append({"name": name, "sha": sha})
    if len({row["name"] for row in branch_records}) != len(branch_records):
        raise RuntimeError(f"{BRANCH_SET_BLOCKER}: duplicate official branch name")
    tip = fetch_json(api_url(f"/repos/{SOURCE_REPOSITORY_PATH}/commits/{default_branch}"))
    tip_sha = tip.get("sha") if isinstance(tip, dict) else None
    if tip_sha != SOURCE_REVISION:
        raise RuntimeError(f"{DEFAULT_TIP_BLOCKER}: default branch {default_branch!r} tip is {tip_sha!r}, expected {SOURCE_REVISION}")
    expected_branches = [{"name": default_branch, "sha": SOURCE_REVISION}]
    if branch_records != expected_branches:
        raise RuntimeError(f"{BRANCH_SET_BLOCKER}: public branch tips drifted: {branch_records!r}")
    license_payload = payload.get("license")
    metadata = None
    if license_payload is not None:
        if not isinstance(license_payload, dict):
            raise RuntimeError("repository license metadata is malformed")
        metadata = {"key": license_payload.get("key"), "name": license_payload.get("name"), "spdx_id": license_payload.get("spdx_id")}
    return {
        "full_name": payload["full_name"],
        "default_branch": default_branch,
        "default_branch_tip": tip_sha,
        "default_branch_tip_matches_fixed": True,
        "branches": branch_records,
        "license": metadata,
    }


def release_evidence() -> dict[str, Any]:
    releases = paged(f"/repos/{SOURCE_REPOSITORY_PATH}/releases")
    tags = paged(f"/repos/{SOURCE_REPOSITORY_PATH}/tags")
    release_records = []
    for release in releases:
        if not isinstance(release, dict):
            raise RuntimeError("release entry is malformed")
        release_records.append({"id": release.get("id"), "tag_name": release.get("tag_name"), "target_commitish": release.get("target_commitish"), "draft": release.get("draft"), "prerelease": release.get("prerelease")})
    tag_records = []
    for tag in tags:
        if not isinstance(tag, dict) or not isinstance(tag.get("name"), str) or not isinstance(tag.get("commit"), dict) or not isinstance(tag["commit"].get("sha"), str):
            raise RuntimeError("tag entry is malformed")
        tag_records.append({"name": tag["name"], "commit": tag["commit"]["sha"]})
    return {"releases": release_records, "tags": tag_records, "exact_revision_releases": [row for row in release_records if row.get("target_commitish") == SOURCE_REVISION], "exact_revision_tags": [row for row in tag_records if row["commit"] == SOURCE_REVISION]}


def _require_keys(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"{label} schema drifted")


def _sha(value: Any, length: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} must be lowercase hexadecimal {length}")


def validate_report(report: dict[str, Any], binding: dict[str, Any]) -> None:
    """Validate both normal and error evidence before it reaches disk."""

    _require_keys(report, REPORT_KEYS, "report")
    if report["format"] != FORMAT or report["status"] not in {BLOCKER, INCOMPLETE_BLOCKER, DEFAULT_TIP_BLOCKER, BRANCH_SET_BLOCKER} or report["publication"] != "NO_UPLOAD" or report["execution"] != "NOT_AUTHORIZED":
        raise RuntimeError("report disposition/format drifted")
    _require_keys(report["checkout"], CHECKOUT_KEYS, "checkout")
    if report["checkout"] != binding:
        raise RuntimeError("report checkout binding drifted")
    expected_head = binding.get("expected_head")
    actual_head = binding.get("actual_head")
    if expected_head is not None:
        _sha(expected_head, 40, "checkout.expected_head")
    if actual_head is not None:
        _sha(actual_head, 40, "checkout.actual_head")
    if type(binding.get("clean")) is not bool:
        raise RuntimeError("checkout.clean must be a JSON boolean")

    source = report["source"]
    _require_keys(source, SOURCE_KEYS, "source")
    if source["repository"] != SOURCE_REPOSITORY or source["revision"] != SOURCE_REVISION or source["license_filename_policy"] != sorted(LICENSE_FILE_NAMES):
        raise RuntimeError("source identity/license filename policy drifted")
    count = source["history_commit_count"]
    fixed_count = source["fixed_tree_blob_count"]
    if count is not None and (type(count) is not int or count < 1) or fixed_count is not None and (type(fixed_count) is not int or fixed_count < 0):
        raise RuntimeError("source tree/history counts are malformed")
    if source["history_scope"] not in {None, "complete_public_branch_history_from_fixed_tip"} or not isinstance(source["history_commits"], list) or not isinstance(source["fixed_tree_license_like_files"], list) or not isinstance(source["fixed_role_files"], dict):
        raise RuntimeError("source evidence collections are malformed")
    if source["history_scope"] is not None and count != len(source["history_commits"]):
        raise RuntimeError("history commit count does not match history evidence")
    history_shas: list[str] = []
    for row in source["history_commits"]:
        _require_keys(row, frozenset({"commit", "tree_blob_count", "license_like_paths", "required_execution_roles_present"}), "history commit")
        _require_keys(row["commit"], frozenset({"sha", "tree", "parents", "date", "subject"}), "history commit identity")
        _sha(row["commit"]["sha"], 40, "history commit SHA")
        _sha(row["commit"]["tree"], 40, "history tree SHA")
        if not isinstance(row["commit"]["parents"], list) or any((not isinstance(parent, str) or not HEAD_PATTERN.fullmatch(parent)) for parent in row["commit"]["parents"]):
            raise RuntimeError("history parent list is malformed")
        if not isinstance(row["commit"]["date"], str) or not row["commit"]["date"] or not isinstance(row["commit"]["subject"], str) or not row["commit"]["subject"]:
            raise RuntimeError("history date/subject is malformed")
        if type(row["tree_blob_count"]) is not int or row["tree_blob_count"] < 0 or not isinstance(row["license_like_paths"], list) or not isinstance(row["required_execution_roles_present"], bool):
            raise RuntimeError("history tree evidence is malformed")
        history_shas.append(row["commit"]["sha"])
        if len(row["license_like_paths"]) != len(set(row["license_like_paths"])):
            raise RuntimeError("duplicate history license path evidence")
        if any(not license_like(safe_relative_path(path, "history license path")) for path in row["license_like_paths"]):
            raise RuntimeError("history license path is malformed")
    if len(history_shas) != len(set(history_shas)):
        raise RuntimeError("duplicate history commit evidence")
    for row in source["fixed_tree_license_like_files"]:
        _require_keys(row, frozenset({"path", "sha", "size"}), "fixed license-like file")
        if not license_like(safe_relative_path(row["path"], "fixed license path")) or type(row["size"]) is not int or row["size"] < 0:
            raise RuntimeError("fixed license-like file evidence is malformed")
        _sha(row["sha"], 40, "fixed license-like file SHA")
    if source["history_scope"] is None:
        if source["fixed_role_files"]:
            raise RuntimeError("error report contains fixed role evidence")
    elif set(source["fixed_role_files"]) != set(REQUIRED_ROLE_PATHS):
        raise RuntimeError("fixed role path set is incomplete or contains extras")
    if source["history_scope"] is not None and fixed_count != FIXED_TREE_BLOB_COUNT:
        raise RuntimeError("fixed tree blob count does not match the authenticated immutable tree")
    for path, row in source["fixed_role_files"].items():
        if path not in REQUIRED_ROLE_PATHS:
            raise RuntimeError("unexpected fixed role path")
        _require_keys(row, frozenset({"path", "sha", "size"}), "fixed role file")
        if row["path"] != path or type(row["size"]) is not int or row["size"] < 0:
            raise RuntimeError("fixed role file evidence is malformed")
        _sha(row["sha"], 40, "fixed role file SHA")

    repository = report["repository_metadata"]
    _require_keys(repository, REPOSITORY_KEYS, "repository metadata")
    if repository["full_name"] is not None and repository["full_name"] != SOURCE_REPOSITORY_PATH:
        raise RuntimeError("repository metadata identity drifted")
    if repository["default_branch"] is not None and (not isinstance(repository["default_branch"], str) or not repository["default_branch"] or "/" in repository["default_branch"] or ".." in repository["default_branch"]):
        raise RuntimeError("repository default branch drifted")
    if repository["default_branch_tip"] is not None:
        _sha(repository["default_branch_tip"], 40, "repository default branch tip")
    if type(repository["default_branch_tip_matches_fixed"]) is not bool:
        raise RuntimeError("repository tip match must be a JSON boolean")
    if not isinstance(repository["branches"], list):
        raise RuntimeError("repository branch evidence is malformed")
    branch_names = []
    for branch in repository["branches"]:
        _require_keys(branch, frozenset({"name", "sha"}), "branch evidence")
        name = branch["name"]
        if not isinstance(name, str) or not name or "\x00" in name or "\\" in name or ".." in name.split("/"):
            raise RuntimeError("branch name is malformed")
        _sha(branch["sha"], 40, "branch tip")
        branch_names.append(name)
    if len(branch_names) != len(set(branch_names)):
        raise RuntimeError("duplicate branch evidence")
    if repository["branches"] and repository["branches"] != [{"name": repository["default_branch"], "sha": SOURCE_REVISION}]:
        raise RuntimeError("public branch evidence does not prove the fixed default tip")
    metadata_license = repository["license"]
    if metadata_license is not None:
        _require_keys(metadata_license, frozenset({"key", "name", "spdx_id"}), "repository license metadata")
        if any(value is not None and not isinstance(value, str) for value in metadata_license.values()):
            raise RuntimeError("repository license metadata types are malformed")
    releases = report["releases_and_tags"]
    _require_keys(releases, RELEASE_KEYS, "release/tag evidence")
    if not all(isinstance(value, list) for value in releases.values()):
        raise RuntimeError("release/tag evidence collections are malformed")
    release_rows = {"releases", "exact_revision_releases"}
    for key in release_rows:
        for row in releases[key]:
            _require_keys(row, frozenset({"id", "tag_name", "target_commitish", "draft", "prerelease"}), "release evidence")
            if type(row["id"]) is not int or row["id"] < 0 or not isinstance(row["tag_name"], str) or not isinstance(row["target_commitish"], str) or type(row["draft"]) is not bool or type(row["prerelease"]) is not bool:
                raise RuntimeError("release row types are malformed")
    for key in {"tags", "exact_revision_tags"}:
        for row in releases[key]:
            _require_keys(row, frozenset({"name", "commit"}), "tag evidence")
            if not isinstance(row["name"], str) or not row["name"]:
                raise RuntimeError("tag row name is malformed")
            _sha(row["commit"], 40, "tag commit")
    expected_exact_releases = [row for row in releases["releases"] if row["target_commitish"] == SOURCE_REVISION]
    expected_exact_tags = [row for row in releases["tags"] if row["commit"] == SOURCE_REVISION]
    if releases["exact_revision_releases"] != expected_exact_releases:
        raise RuntimeError("exact release evidence is not the derived release subset")
    if releases["exact_revision_tags"] != expected_exact_tags:
        raise RuntimeError("exact tag evidence is not the derived tag subset")

    readme = report["readme"]
    _require_keys(readme, README_KEYS, "README evidence")
    if readme["path"] is not None and readme["path"] != "README.md":
        raise RuntimeError("README evidence path drifted")
    if readme["bytes"] is not None and (type(readme["bytes"]) is not int or readme["bytes"] < 0):
        raise RuntimeError("README evidence byte count is malformed")
    if readme["sha256"] is not None:
        _sha(readme["sha256"], 64, "README SHA-256")
    if not isinstance(readme["license_related_lines"], list):
        raise RuntimeError("README license lines are malformed")
    for line in readme["license_related_lines"]:
        _require_keys(line, frozenset({"line", "text"}), "README license line")
        if type(line["line"]) is not int or line["line"] < 1 or not isinstance(line["text"], str):
            raise RuntimeError("README license line types are malformed")
    line_numbers = [line["line"] for line in readme["license_related_lines"]]
    if line_numbers != sorted(set(line_numbers)):
        raise RuntimeError("README license lines are not unique and ordered")
    if source["history_scope"] is not None and readme["bytes"] != source["fixed_role_files"]["README.md"]["size"]:
        raise RuntimeError("README evidence size does not match the authenticated tree")
    decision = report["license_decision"]
    _require_keys(decision, DECISION_KEYS, "license decision")
    if decision["source_license"] != "UNKNOWN" or decision["status"] != report["status"] or not isinstance(decision["reason"], str) or not isinstance(decision["historical_license_candidates"], list) or decision["equivalent_license_bearing_revision"] is not None:
        raise RuntimeError("license decision drifted")
    for row in decision["historical_license_candidates"]:
        if row not in source["history_commits"] or not row["license_like_paths"] or row["required_execution_roles_present"] is not True:
            raise RuntimeError("non-equivalent historical license candidate accepted")

    if report["status"] == BLOCKER:
        if binding["actual_head"] != expected_head or binding["clean"] is not True or repository["default_branch_tip"] != SOURCE_REVISION or repository["default_branch_tip_matches_fixed"] is not True or repository["branches"] != [{"name": repository["default_branch"], "sha": SOURCE_REVISION}] or source["history_scope"] != "complete_public_branch_history_from_fixed_tip" or count is None or not source["history_commits"]:
            raise RuntimeError("normal source-license report lacks clean HEAD/default-tip proof")
    elif report["status"] == INCOMPLETE_BLOCKER:
        if source["history_scope"] is not None or any(value is not None for value in (repository["full_name"], repository["default_branch"], repository["default_branch_tip"], readme["path"], readme["bytes"], readme["sha256"])):
            raise RuntimeError("incomplete source-license report contains completed or partial network evidence")
    elif report["status"] == DEFAULT_TIP_BLOCKER:
        if repository["default_branch_tip_matches_fixed"] is True or source["history_scope"] is not None:
            raise RuntimeError("default-tip blocker claims a complete history")
    elif report["status"] == BRANCH_SET_BLOCKER:
        if repository["branches"] or source["history_scope"] is not None:
            raise RuntimeError("branch-set blocker claims complete branch evidence")


def validate_completed_report(report: dict[str, Any], expected_head: str) -> None:
    """Require the exact evidence state that can state the factual blocker."""
    _sha(expected_head, 40, "expected evidence HEAD")
    if report["status"] != BLOCKER or report["publication"] != "NO_UPLOAD" or report["execution"] != "NOT_AUTHORIZED":
        raise RuntimeError("evidence is not the completed factual license audit")
    checkout = report["checkout"]
    if checkout != {"expected_head": expected_head, "actual_head": expected_head, "clean": True}:
        raise RuntimeError("evidence checkout is not bound to the requested clean HEAD")
    source = report["source"]
    if source["history_scope"] != "complete_public_branch_history_from_fixed_tip" or source["history_commit_count"] != len(source["history_commits"]):
        raise RuntimeError("evidence does not contain complete public branch history")
    repository = report["repository_metadata"]
    if repository["full_name"] != SOURCE_REPOSITORY_PATH or repository["default_branch_tip"] != SOURCE_REVISION or repository["default_branch_tip_matches_fixed"] is not True or repository["branches"] != [{"name": repository["default_branch"], "sha": SOURCE_REVISION}]:
        raise RuntimeError("evidence does not bind the current default branch tip")
    decision = report["license_decision"]
    if decision["source_license"] != "UNKNOWN" or decision["status"] != BLOCKER or decision["equivalent_license_bearing_revision"] is not None:
        raise RuntimeError("evidence does not preserve the factual unknown-license blocker")


def audit(binding: dict[str, Any]) -> dict[str, Any]:
    # Prove that the pinned revision is the current default-branch tip before
    # treating its reachable parent graph as the complete public history.
    repository = repository_evidence()
    history, trees = reachable_history()
    fixed_records = trees[SOURCE_REVISION]
    roles = tree_roles(fixed_records)
    history_license_files = []
    for commit in history:
        records = trees[commit["sha"]]
        paths = [record["path"] for record in records if license_like(record["path"])]
        role_paths = {record["path"] for record in records}
        history_license_files.append(
            {
                "commit": commit,
                "tree_blob_count": len(records),
                "license_like_paths": paths,
                "required_execution_roles_present": all(path in role_paths for path in REQUIRED_ROLE_PATHS)
                and any(path.startswith("eval_audio/") and path.endswith(".py") for path in role_paths),
            }
        )
    fixed_license_files = [record for record in fixed_records if license_like(record["path"])]
    applicable_candidates = [
        row
        for row in history_license_files
        if row["license_like_paths"] and row["required_execution_roles_present"]
    ]
    report = {
        "format": FORMAT,
        "status": BLOCKER,
        "publication": "NO_UPLOAD",
        "execution": "NOT_AUTHORIZED",
        "checkout": binding,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "history_commit_count": len(history),
            "history_scope": "complete_public_branch_history_from_fixed_tip",
            "license_filename_policy": sorted(LICENSE_FILE_NAMES),
            "history_commits": history_license_files,
            "fixed_tree_blob_count": len(fixed_records),
            "fixed_tree_license_like_files": fixed_license_files,
            "fixed_role_files": roles,
        },
        "repository_metadata": repository,
        "releases_and_tags": release_evidence(),
        "readme": readme_evidence(roles["README.md"]),
        "license_decision": {
            "source_license": "UNKNOWN",
            "status": BLOCKER,
            "reason": (
                "The fixed execution source has no primary license file; no reachable immutable history commit with the required execution roles has a license-like file, and no exact release/tag exists. README model-license guidance is not a source-code license."
                if not applicable_candidates
                else "A reachable historical commit has a license-like file, but the fixed execution source remains unlicensed; no license is inherited across revisions."
            ),
            "historical_license_candidates": applicable_candidates,
            "equivalent_license_bearing_revision": None,
        },
    }
    validate_report(report, binding)
    return report


def write_exclusive(path: Path, payload: dict[str, Any], binding: dict[str, Any]) -> None:
    validate_output_path(path)
    validate_report(payload, binding)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def blocked_payload(error: Exception, binding: dict[str, Any]) -> dict[str, Any]:
    error_text = str(error)
    if error_text.startswith(f"{DEFAULT_TIP_BLOCKER}:"):
        status = DEFAULT_TIP_BLOCKER
    elif error_text.startswith(f"{BRANCH_SET_BLOCKER}:"):
        status = BRANCH_SET_BLOCKER
    else:
        status = INCOMPLETE_BLOCKER
    return {
        "format": FORMAT,
        "status": status,
        "publication": "NO_UPLOAD",
        "execution": "NOT_AUTHORIZED",
        "checkout": binding,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "history_commit_count": None,
            "history_scope": None,
            "license_filename_policy": sorted(LICENSE_FILE_NAMES),
            "history_commits": [],
            "fixed_tree_blob_count": None,
            "fixed_tree_license_like_files": [],
            "fixed_role_files": {},
        },
        "repository_metadata": {
            "full_name": None,
            "default_branch": None,
            "default_branch_tip": None,
            "default_branch_tip_matches_fixed": False,
            "branches": [],
            "license": None,
        },
        "releases_and_tags": {"releases": [], "tags": [], "exact_revision_releases": [], "exact_revision_tags": []},
        "readme": {"path": None, "bytes": None, "sha256": None, "license_related_lines": []},
        "license_decision": {"source_license": "UNKNOWN", "status": status, "reason": f"Primary source-history audit could not complete; fail closed: {type(error).__name__}: {error}", "historical_license_candidates": [], "equivalent_license_bearing_revision": None},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, action=_UniqueValue)
    parser.add_argument("--expected-head", action=_UniqueValue)
    parser.add_argument("--validate-evidence", type=Path, action=_UniqueValue)
    parser.add_argument("--self-test", action=_UniqueFlag, nargs=0)
    return parser


def self_test() -> None:
    original_fetch = globals()["fetch_json"]
    original_read_blob = globals()["read_blob"]
    original_repo_root = globals()["repo_root"]
    original_git_output = globals()["git_output"]
    root = "https://api.github.com"
    initial = SOURCE_REVISION
    parent = "a" * 40
    initial_tree = "b" * 40
    parent_tree = "c" * 40
    readme_content = b"## License Agreement\nCheck the license of each model inside its HF repo.\n"
    records = [
        {"path": path, "mode": "100644", "type": "blob", "sha": f"{index + 1:040x}", "size": len(readme_content) if path == "README.md" else 16 + index}
        for index, path in enumerate(REQUIRED_ROLE_PATHS)
    ]
    records.extend(
        {"path": f"assets/fixture-{index}.png", "mode": "100644", "type": "blob", "sha": f"{index + 100:040x}", "size": 8}
        for index in range(3)
    )
    fixture = {
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}": {"full_name": SOURCE_REPOSITORY_PATH, "default_branch": "main", "license": None},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/branches?per_page=100&page=1": [{"name": "main", "commit": {"sha": initial}}],
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/commits/main": {"sha": initial},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/commits/{initial}": {"sha": initial, "commit": {"tree": {"sha": initial_tree}, "author": {"date": "2025-04-21T08:50:49Z"}, "message": "fixed"}, "parents": [{"sha": parent}]},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/commits/{parent}": {"sha": parent, "commit": {"tree": {"sha": parent_tree}, "author": {"date": "2024-07-16T03:25:11Z"}, "message": "initial"}, "parents": []},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{initial_tree}?recursive=1": {"sha": initial_tree, "truncated": False, "tree": records},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{parent_tree}?recursive=1": {"sha": parent_tree, "truncated": False, "tree": records},
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/releases?per_page=100&page=1": [],
        f"{root}/repos/{SOURCE_REPOSITORY_PATH}/tags?per_page=100&page=1": [],
    }
    def fake_fetch(url: str) -> Any:
        parsed = urlsplit(url)
        key = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
        if key not in fixture:
            raise AssertionError(f"unexpected fixture URL: {url}")
        return fixture[key]
    globals()["fetch_json"] = fake_fetch
    readme_fixture = lambda _sha: readme_content
    globals()["read_blob"] = readme_fixture
    try:
        with tempfile.TemporaryDirectory(prefix="vokra-qwen2-audio-audit-") as directory:
            fixture_root = Path(directory)
            git_state = {"head": "a" * 40, "dirty": ""}
            globals()["repo_root"] = lambda: fixture_root

            def fake_git_output(_root: Path, *args: str) -> str:
                if args == ("rev-parse", "HEAD"):
                    return git_state["head"]
                if args == ("status", "--porcelain", "--untracked-files=all"):
                    return git_state["dirty"]
                raise AssertionError(f"unexpected fixture git call: {args}")

            globals()["git_output"] = fake_git_output
            binding = checkout_binding("a" * 40)
            assert binding == {"expected_head": "a" * 40, "actual_head": "a" * 40, "clean": True}
            git_state["head"] = "b" * 40
            try:
                checkout_binding("a" * 40)
            except CheckoutError:
                pass
            else:
                raise AssertionError("HEAD drift accepted")
            git_state["head"] = "a" * 40
            git_state["dirty"] = " M unrelated"
            try:
                checkout_binding("a" * 40)
            except CheckoutError:
                pass
            else:
                raise AssertionError("dirty checkout accepted")
            git_state["dirty"] = ""
            for unsafe in (Path("relative.json"), Path("/tmp/../unsafe.json"), Path("/tmp/./unsafe.json")):
                try:
                    validate_output_path(unsafe)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("unsafe output path accepted")
            existing = fixture_root / "existing.json"
            existing.write_text("x", encoding="utf-8")
            try:
                validate_output_path(existing)
            except RuntimeError:
                pass
            else:
                raise AssertionError("existing output accepted")
            linked_parent = fixture_root / "linked-parent"
            linked_parent.symlink_to(Path("/private/tmp"), target_is_directory=True)
            try:
                validate_output_path(linked_parent / "new.json")
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlink output parent accepted")
        blob = b"source-license-fixture"
        blob_sha = git_blob_sha1(blob)
        original_blob_fetch = globals()["fetch_json"]
        globals()["fetch_json"] = lambda _url: {"sha": blob_sha, "encoding": "base64", "content": base64.b64encode(blob).decode("ascii")}
        source_read_blob = original_read_blob
        globals()["read_blob"] = source_read_blob
        assert read_blob(blob_sha) == blob
        globals()["fetch_json"] = lambda _url: {"sha": blob_sha, "encoding": "base64", "content": base64.b64encode(b"tampered").decode("ascii")}
        try:
            read_blob(blob_sha)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered GitHub blob accepted")
        globals()["fetch_json"] = original_blob_fetch
        globals()["read_blob"] = readme_fixture
        binding = {"expected_head": "a" * 40, "actual_head": "a" * 40, "clean": True}
        result = audit(binding)
        assert result["status"] == BLOCKER
        assert result["source"]["history_commit_count"] == 2
        assert result["source"]["fixed_tree_blob_count"] == FIXED_TREE_BLOB_COUNT
        assert set(result["source"]["fixed_role_files"]) == set(REQUIRED_ROLE_PATHS)
        assert result["source"]["fixed_tree_license_like_files"] == []
        assert result["license_decision"]["equivalent_license_bearing_revision"] is None
        validate_completed_report(result, "a" * 40)
        with tempfile.TemporaryDirectory(prefix="vokra-qwen2-audio-evidence-", dir="/private/tmp") as evidence_directory:
            evidence_path = Path(evidence_directory) / "evidence.json"
            write_exclusive(evidence_path, result, binding)
            original_argv = list(sys.argv)
            try:
                sys.argv = ["qwen2_audio_source_license_audit.py", "--validate-evidence", str(evidence_path), "--expected-head", "a" * 40]
                assert main() == 0
            finally:
                sys.argv = original_argv
        incomplete = blocked_payload(RuntimeError("public API unavailable"), binding)
        assert incomplete["status"] == INCOMPLETE_BLOCKER
        validate_report(incomplete, binding)
        try:
            validate_completed_report(incomplete, "a" * 40)
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete audit accepted as factual license blocker")
        omitted = [record for record in records if record["path"] != REQUIRED_ROLE_PATHS[-1]]
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{initial_tree}?recursive=1"] = {"sha": initial_tree, "truncated": False, "tree": omitted}
        try:
            audit(binding)
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete fixed execution role set accepted")
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{initial_tree}?recursive=1"] = {"sha": initial_tree, "truncated": False, "tree": records}
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{parent_tree}?recursive=1"] = {"sha": parent_tree, "truncated": False, "tree": [{"path": "LICENSE", "mode": "100644", "type": "blob", "sha": "4" * 40, "size": 10}]}
        non_equivalent = audit(binding)
        assert non_equivalent["status"] == BLOCKER and non_equivalent["license_decision"]["historical_license_candidates"] == []
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{parent_tree}?recursive=1"] = {"sha": parent_tree, "truncated": False, "tree": records}
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{initial_tree}?recursive=1"] = {"sha": initial_tree, "truncated": False, "tree": [*records, {"path": "LICENSE", "mode": "100644", "type": "blob", "sha": "4" * 40, "size": 10}]}
        try:
            audit(binding)
        except RuntimeError:
            pass
        else:
            raise AssertionError("immutable fixed-tree blob-count tampering accepted")
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/git/trees/{initial_tree}?recursive=1"] = {"sha": initial_tree, "truncated": False, "tree": records}
        mutated = dict(result)
        mutated["unexpected"] = True
        try:
            validate_report(mutated, binding)
        except RuntimeError:
            pass
        else:
            raise AssertionError("extra report key accepted")
        missing = dict(result)
        del missing["readme"]
        try:
            validate_report(missing, binding)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing report key accepted")
        def expect_invalid(candidate: dict[str, Any], label: str) -> None:
            try:
                validate_report(candidate, binding)
            except RuntimeError:
                return
            raise AssertionError(f"{label} tampering accepted")

        parent_index = next(index for index, row in enumerate(result["source"]["history_commits"]) if row["commit"]["parents"])
        parent_sha_tamper = copy.deepcopy(result)
        parent_sha_tamper["source"]["history_commits"][parent_index]["commit"]["parents"][0] = "not-a-sha"
        expect_invalid(parent_sha_tamper, "parent SHA")
        date_tamper = copy.deepcopy(result)
        date_tamper["source"]["history_commits"][0]["commit"]["date"] = 123
        expect_invalid(date_tamper, "history date")
        subject_tamper = copy.deepcopy(result)
        subject_tamper["source"]["history_commits"][0]["commit"]["subject"] = None
        expect_invalid(subject_tamper, "history subject")
        path_tamper = copy.deepcopy(result)
        path_tamper["source"]["history_commits"][0]["license_like_paths"] = ["README.md"]
        expect_invalid(path_tamper, "license path")
        branch_tamper = copy.deepcopy(result)
        branch_tamper["repository_metadata"]["branches"][0]["sha"] = "not-a-sha"
        expect_invalid(branch_tamper, "branch SHA")
        release_tamper = copy.deepcopy(result)
        release_tamper["releases_and_tags"]["releases"] = [{"id": "1", "tag_name": "v1", "target_commitish": initial, "draft": False, "prerelease": False}]
        expect_invalid(release_tamper, "release row")
        release_subset_tamper = copy.deepcopy(result)
        release_row = {"id": 1, "tag_name": "v1", "target_commitish": initial, "draft": False, "prerelease": False}
        release_subset_tamper["releases_and_tags"]["releases"] = [release_row]
        expect_invalid(release_subset_tamper, "release subset")
        tag_tamper = copy.deepcopy(result)
        tag_tamper["releases_and_tags"]["tags"] = [{"name": "v1", "commit": "not-a-sha"}]
        expect_invalid(tag_tamper, "tag row")
        tag_subset_tamper = copy.deepcopy(result)
        tag_subset_tamper["releases_and_tags"]["tags"] = [{"name": "v1", "commit": initial}]
        expect_invalid(tag_subset_tamper, "tag subset")
        readme_tamper = copy.deepcopy(result)
        readme_tamper["readme"]["license_related_lines"] = [{"line": "1", "text": "License"}]
        expect_invalid(readme_tamper, "README line")
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/commits/main"] = {"sha": "d" * 40}
        try:
            audit(binding)
        except RuntimeError as error:
            assert str(error).startswith(f"{DEFAULT_TIP_BLOCKER}:")
            validate_report(blocked_payload(error, binding), binding)
        else:
            raise AssertionError("default branch tip drift accepted")
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/commits/main"] = {"sha": initial}
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/branches?per_page=100&page=1"] = [
            {"name": "main", "commit": {"sha": initial}},
            {"name": "feature", "commit": {"sha": "e" * 40}},
        ]
        try:
            audit(binding)
        except RuntimeError as error:
            assert str(error).startswith(f"{BRANCH_SET_BLOCKER}:")
            validate_report(blocked_payload(error, binding), binding)
        else:
            raise AssertionError("extra public branch accepted")
        fixture[f"{root}/repos/{SOURCE_REPOSITORY_PATH}/branches?per_page=100&page=1"] = [{"name": "main", "commit": {"sha": initial}}]
        duplicate_parser = build_parser()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                duplicate_parser.parse_args(["--output", "/private/tmp/a", "--output", "/private/tmp/b"])
        except SystemExit:
            pass
        else:
            raise AssertionError("duplicate CLI option accepted")
        duplicate = b'{"x":1,"x":2}'
        try:
            strict_json(duplicate, "fixture")
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate JSON key accepted")
    finally:
        globals()["fetch_json"] = original_fetch
        globals()["read_blob"] = original_read_blob
        globals()["repo_root"] = original_repo_root
        globals()["git_output"] = original_git_output
    print("qwen2_audio_source_license_audit.py self-test: OK (immutable history/source-license blocker)")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        if args.output is not None or args.expected_head is not None or args.validate_evidence is not None:
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if args.validate_evidence is not None:
        if args.output is not None or args.expected_head is None:
            parser.error("--validate-evidence requires --expected-head and accepts no --output")
        try:
            if not isinstance(args.expected_head, str) or not HEAD_PATTERN.fullmatch(args.expected_head):
                raise RuntimeError("--expected-head must be lowercase HEX40")
            evidence_path = args.validate_evidence
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise RuntimeError("evidence must be an existing regular file")
            if evidence_path.stat().st_size > MAX_RESPONSE_BYTES:
                raise RuntimeError(f"evidence exceeds {MAX_RESPONSE_BYTES} bytes")
            with evidence_path.open("rb") as stream:
                raw = stream.read(MAX_RESPONSE_BYTES + 1)
            report = strict_json(raw, str(evidence_path))
            if not isinstance(report, dict):
                raise RuntimeError("evidence root must be a JSON object")
            binding = report.get("checkout")
            validate_report(report, binding)
            validate_completed_report(report, args.expected_head)
        except Exception as error:
            print(f"QWEN2_AUDIO_SOURCE_LICENSE_EVIDENCE_INVALID: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print("QWEN2_AUDIO_SOURCE_LICENSE_EVIDENCE_VALID: complete SOURCE_LICENSE_UNKNOWN_BLOCKER NO_UPLOAD")
        return 0
    if args.output is None or args.expected_head is None:
        parser.error("--output and --expected-head are required")
    try:
        output = validate_output_path(args.output)
    except Exception as error:
        print(f"QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_OUTPUT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    try:
        binding = checkout_binding(args.expected_head)
    except CheckoutError as error:
        result = blocked_payload(error, error.binding)
        print(f"QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        try:
            write_exclusive(output, result, error.binding)
        except Exception as output_error:
            print(f"QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_OUTPUT_ERROR: {type(output_error).__name__}: {output_error}", file=sys.stderr)
        return 2
    try:
        result = audit(binding)
    except Exception as error:
        result = blocked_payload(error, binding)
        print(f"QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
    else:
        print("QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_BLOCKED: SOURCE_LICENSE_UNKNOWN_BLOCKER NO_UPLOAD", file=sys.stderr)
    try:
        write_exclusive(output, result, binding)
    except Exception as error:
        print(f"QWEN2_AUDIO_SOURCE_LICENSE_AUDIT_OUTPUT_ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
