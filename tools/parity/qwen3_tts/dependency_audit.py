#!/usr/bin/env python3
"""Model-free factual audit for the frozen Qwen3-TTS Linux closure.

This module deliberately imports no reference/model package.  It only reads
the lock, installed ``importlib.metadata`` records and installed files.  The
optional network requests are exact locked sdists, fixed source LICENSE paths,
and the bounded HF model-info license fallback for exact model paths returning
HTTP 404; weights, snapshots, Cargo and publication are outside this audit.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import tomllib

try:
    import license_gate
except ModuleNotFoundError:  # direct execution from repository root
    from tools.parity.qwen3_tts import license_gate


SCHEMA = "vokra-qwen3-tts-dependency-audit-v2"
REPORT_FIELDS = {
    "schema", "status", "environment", "repository", "project", "license_gate",
    "closure", "locked_rows", "inactive_rows", "packages",
    "fixed_source_model_decoder_license_paths", "model_license_files", "model_license_metadata",
    "model_acquisition", "failures",
}
PYPI_HOST = "files.pythonhosted.org"
HF_HOSTS = {"huggingface.co", "hf.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co"}
HF_API_HOSTS = {"huggingface.co", "hf.co"}
ELF_MAGIC = b"\x7fELF"
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
MAX_LICENSE_BYTES = 2 * 1024 * 1024
MAX_MODEL_INFO_BYTES = license_gate.MODEL_LICENSE_METADATA_POLICY["max_response_bytes"]
MAX_SDIST_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_SDIST_REDIRECTS = 3


class AuditError(ValueError):
    pass


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


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).casefold()


def identity(name: str, version: str) -> str:
    return f"{normalized_name(name)}=={version.strip().casefold()}"


def load_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise AuditError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    except (OSError, UnicodeError, json.JSONDecodeError, AuditError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return normalized_name(row["name"]), row["version"], canonical_json(row["source"])


_MARKER = re.compile(r"\s*(?:(and|or|==|!=|\(|\))|([A-Za-z_][A-Za-z0-9_]*)|('(?:[^'\\]|\\.)*'))")
_MARKER_VARS = {"implementation_name", "platform_machine", "sys_platform"}


def marker_matches(marker: Any, env: dict[str, str]) -> bool:
    if marker is None:
        return True
    if not isinstance(marker, str) or not marker.strip():
        raise AuditError("empty or malformed lock marker")
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(marker):
        match = _MARKER.match(marker, pos)
        if not match:
            if marker[pos:].strip():
                raise AuditError(f"unsupported lock marker: {marker}")
            break
        tokens.append(("op", match.group(1)) if match.group(1) else
                      ("id", match.group(2)) if match.group(2) else
                      ("str", match.group(3)[1:-1]))
        pos = match.end()
    cursor = 0
    def peek(value: str | None = None) -> tuple[str, str] | None:
        return tokens[cursor] if cursor < len(tokens) and (value is None or tokens[cursor][1] == value) else None
    def take(value: str | None = None) -> tuple[str, str]:
        nonlocal cursor
        item = peek(value)
        if item is None:
            raise AuditError(f"malformed lock marker: {marker}")
        cursor += 1
        return item
    def atom() -> bool:
        if peek("("):
            take("("); result = disjunction(); take(")"); return result
        variable = take()[1]
        if variable not in _MARKER_VARS:
            raise AuditError(f"unsupported lock marker variable: {variable}")
        op = take()[1]
        if op not in {"==", "!="} or take()[0] != "str":
            raise AuditError(f"unsupported lock marker comparison: {marker}")
        literal = tokens[cursor - 1][1]
        result = env[variable] == literal
        return result if op == "==" else not result
    def conjunction() -> bool:
        result = atom()
        while peek("and"):
            take("and"); result = atom() and result
        return result
    def disjunction() -> bool:
        result = conjunction()
        while peek("or"):
            take("or"); result = conjunction() or result
        return result
    result = disjunction()
    if cursor != len(tokens):
        raise AuditError(f"trailing lock marker tokens: {marker}")
    return result


def active_rows(lock: dict[str, Any], env: dict[str, str]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], str]]:
    rows = lock.get("package")
    if not isinstance(rows, list):
        raise AuditError("uv.lock package rows are malformed")
    by_name: dict[str, list[dict[str, Any]]] = {}
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or not isinstance(row.get("source"), dict):
            raise AuditError("uv.lock package identity is malformed")
        key = _row_key(row)
        if key in by_key:
            raise AuditError(f"duplicate lock package identity: {row['name']}")
        by_key[key] = row; by_name.setdefault(normalized_name(row["name"]), []).append(row)
        markers = row.get("resolution-markers")
        if markers is not None and (not isinstance(markers, list) or sum(marker_matches(x, env) for x in markers) > 1):
            raise AuditError(f"ambiguous package resolution marker: {row['name']}")
        for dep in row.get("dependencies", []):
            if not isinstance(dep, dict) or not isinstance(dep.get("name"), str):
                raise AuditError(f"malformed dependency row: {row['name']}")
            if dep["name"] in license_gate.FORBIDDEN_PACKAGES:
                raise AuditError(f"forbidden package dependency is locked: {dep['name']}")
            marker_matches(dep.get("marker"), env)
    roots = [r for r in rows if r.get("source") == {"virtual": "."}]
    if len(roots) != 1:
        raise AuditError("uv.lock must contain one virtual root")
    def resolved(row: dict[str, Any]) -> bool:
        markers = row.get("resolution-markers")
        return markers is None or any(marker_matches(x, env) for x in markers)
    seen: set[tuple[str, str, str]] = set(); visiting: set[tuple[str, str, str]] = set()
    def visit(row: dict[str, Any]) -> None:
        key = _row_key(row)
        if key in visiting: raise AuditError(f"dependency cycle at {row['name']}")
        if key in seen: return
        visiting.add(key)
        for dep in row.get("dependencies", []):
            if not marker_matches(dep.get("marker"), env): continue
            candidates = [r for r in by_name.get(normalized_name(dep["name"]), []) if resolved(r)]
            if "version" in dep: candidates = [r for r in candidates if r["version"] == dep["version"]]
            if "source" in dep: candidates = [r for r in candidates if canonical_json(r["source"]) == canonical_json(dep["source"])]
            if len(candidates) != 1: raise AuditError(f"missing or ambiguous dependency: {row['name']} -> {dep['name']}")
            visit(candidates[0])
        visiting.remove(key); seen.add(key)
    visit(roots[0])
    inactive: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = _row_key(row)
        if row.get("source") == {"virtual": "."}: inactive[key] = "virtual project row; no installed distribution is expected"
        elif key not in seen: inactive[key] = "resolution marker is false or row is unreachable from the virtual project"
    return sorted((by_key[k] for k in seen if by_key[k].get("source") != {"virtual": "."}), key=lambda r: (normalized_name(r["name"]), r["version"])), inactive


def _license_candidate(path: str) -> bool:
    name = PurePosixPath(path).name.casefold()
    return name in LICENSE_NAMES or any(name.startswith(x + sep) for x in LICENSE_NAMES for sep in (".", "-", "_"))


def _inside(root: Path, path: Path) -> bool:
    try: path.resolve(strict=False).relative_to(root.resolve(strict=False)); return True
    except (OSError, RuntimeError, ValueError): return False


def publisher_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []; unsafe: list[str] = []; root = Path(dist.locate_file(""))
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry)
        if not _license_candidate(relative): continue
        path = Path(dist.locate_file(entry))
        if not _inside(root, path) or path.is_symlink() or not path.is_file(): unsafe.append(relative); continue
        out.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return out, unsafe


def _elf_needed(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if handle.read(4) != ELF_MAGIC: return {"format": "non-elf", "needed": []}
    try:
        result = subprocess.run(["readelf", "-d", str(path)], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"format": "elf", "needed": [], "error": type(exc).__name__}
    return {"format": "elf", "needed": sorted(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout))}


def native_files(dist: metadata.Distribution) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(dist.locate_file("")); out: list[dict[str, Any]] = []; unsafe: list[str] = []
    for entry in sorted(dist.files or [], key=str):
        relative = str(entry); path = Path(dist.locate_file(entry)); lower = path.name.casefold()
        candidate = path.suffix.casefold() in NATIVE_SUFFIXES or ".so." in lower or ".dylib." in lower or lower.endswith(".dll")
        if not candidate: continue
        if not _inside(root, path) or path.is_symlink() or not path.is_file(): unsafe.append(relative); continue
        out.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path), "bundled": True, "elf": _elf_needed(path)})
    return out, unsafe


def _artifact(row: dict[str, Any]) -> dict[str, Any]:
    item = row.get("sdist")
    if not isinstance(item, dict): raise AuditError(f"{row['name']} has no locked sdist")
    required = {"url", "hash", "size", "upload-time"}
    if set(item) != required or not isinstance(item["url"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item["hash"])) or not isinstance(item["size"], int) or item["size"] <= 0:
        raise AuditError(f"{row['name']} locked sdist artifact is malformed")
    _validate_sdist_url(item["url"], item["url"], initial=True)
    return item


def _validate_sdist_url(value: str, expected: str, *, initial: bool) -> None:
    """Validate the exact PyPI URL before opening a connection or accepting bytes."""
    if initial and value != expected: raise AuditError("initial sdist URL drifted")
    parsed = urlparse(value); expected_parsed = urlparse(expected)
    try: port = parsed.port; expected_port = expected_parsed.port
    except ValueError as exc: raise AuditError("sdist URL has an invalid port") from exc
    if (parsed.scheme != "https" or parsed.hostname != PYPI_HOST or parsed.username or parsed.password
            or port not in (None, 443) or parsed.query or parsed.fragment or not parsed.path):
        raise AuditError("sdist URL is not exact files.pythonhosted.org")
    if (expected_parsed.scheme != "https" or expected_parsed.hostname != PYPI_HOST
            or expected_parsed.username or expected_parsed.password or expected_port not in (None, 443)
            or expected_parsed.query or expected_parsed.fragment or not expected_parsed.path):
        raise AuditError("locked sdist URL contract is malformed")
    if parsed.path != expected_parsed.path: raise AuditError("sdist redirect changed exact path")


def _archive_license(body: bytes, url: str) -> list[dict[str, Any]]:
    suffix = urlparse(url).path.casefold()
    if suffix.endswith((".zip",)): kind = "zip"
    elif suffix.endswith((".tar.gz", ".tgz")): kind = "tar.gz"
    elif suffix.endswith((".tar.bz2", ".tbz2")): kind = "tar.bz2"
    elif suffix.endswith((".tar.xz", ".txz")): kind = "tar.xz"
    else: raise AuditError("unsupported locked sdist format")
    if len(body) > MAX_SDIST_BYTES: raise AuditError("locked sdist exceeds bounded size")
    found: list[dict[str, Any]] = []; names: set[str] = set(); total = 0
    def add(name: str, payload: bytes) -> None:
        if not _license_candidate(name): return
        if len(payload) > MAX_MEMBER_BYTES: raise AuditError("license member is oversized")
        found.append({"path": name, "size": len(payload), "sha256": sha256_bytes(payload), "content_base64": base64.b64encode(payload).decode("ascii")})
    def safe(name: str) -> str:
        if name.endswith("/"): name = name[:-1]
        if not name or "\x00" in name or "\\" in name or name.startswith("/") or any(x in {"", ".", ".."} for x in name.split("/")): raise AuditError("unsafe archive member path")
        return str(PurePosixPath(name))
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            for index, info in enumerate(archive.infolist(), 1):
                if index > MAX_MEMBERS: raise AuditError("archive has too many members")
                name = safe(info.filename)
                if name in names: raise AuditError("archive contains duplicate member")
                names.add(name); ftype = (info.external_attr >> 16) & 0o170000
                if ftype not in (0, stat.S_IFREG, stat.S_IFDIR): raise AuditError("archive contains special member")
                if info.is_dir(): continue
                if info.file_size > MAX_MEMBER_BYTES: raise AuditError("archive member is oversized")
                total += info.file_size
                if total > MAX_ARCHIVE_BYTES: raise AuditError("archive is oversized")
                add(name, archive.read(info))
    else:
        mode = {"tar.gz": "r:gz", "tar.bz2": "r:bz2", "tar.xz": "r:xz"}[kind]
        with tarfile.open(fileobj=io.BytesIO(body), mode=mode) as archive:
            for index, member in enumerate(archive, 1):
                if index > MAX_MEMBERS: raise AuditError("archive has too many members")
                name = safe(member.name)
                if name in names: raise AuditError("archive contains duplicate member")
                names.add(name)
                if member.isdir(): continue
                if not member.isfile() or member.size < 0 or member.size > MAX_MEMBER_BYTES: raise AuditError("archive contains unsafe member")
                total += member.size
                if total > MAX_ARCHIVE_BYTES: raise AuditError("archive is oversized")
                stream = archive.extractfile(member)
                if stream is None: raise AuditError("archive member cannot be read")
                add(name, stream.read(MAX_MEMBER_BYTES + 1))
    if not found: raise AuditError("locked sdist has no LICENSE/COPYING/NOTICE/COPYRIGHT member")
    return found


def fetch_sdist(row: dict[str, Any], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    art = _artifact(row); url = art["url"]; final = url; trace = [url]
    if fetcher:
        final, body = fetcher(url)
    else:
        class Redirects(HTTPRedirectHandler):
            def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
                if len(trace) - 1 >= MAX_SDIST_REDIRECTS: raise AuditError("sdist redirect limit exceeded")
                resolved = urljoin(request.full_url, newurl)
                _validate_sdist_url(resolved, url, initial=False)
                trace.append(resolved); return super().redirect_request(request, fp, code, msg, headers, resolved)
        with build_opener(Redirects()).open(Request(url, headers={"User-Agent": "vokra-qwen3-tts-audit/1"}), timeout=30) as response:
            final = response.geturl(); body = response.read(MAX_SDIST_BYTES + 1)
    _validate_sdist_url(final, url, initial=False)
    if not isinstance(body, bytes) or len(body) != art["size"] or sha256_bytes(body) != art["hash"].removeprefix("sha256:"): raise AuditError("locked sdist size/hash mismatch")
    return {"status": "ACQUIRED_LOCKED_SDIST_LICENSE_BYTES", "requested_url": url, "final_url": final, "redirect_trace": trace, "size": len(body), "sha256": sha256_bytes(body), "license_files": _archive_license(body, url)}


def _controlled(exc: Exception) -> dict[str, Any]:
    return {"kind": "HTTP_ERROR", "status": exc.code} if isinstance(exc, HTTPError) else {"kind": "URL_ERROR"} if isinstance(exc, URLError) else {"kind": "VALIDATION_ERROR"} if isinstance(exc, ValueError) else {"kind": type(exc).__name__}


def fixed_license_items() -> list[dict[str, str]]:
    items = []
    for component in license_gate.fixed_component_identities():
        repo = component["repo"]; revision = component["revision"]
        if component["kind"] == "source": url = f"https://raw.githubusercontent.com/{repo}/{revision}/LICENSE"
        else: url = f"https://huggingface.co/{repo}/raw/{revision}/LICENSE"
        items.append({"component": component["component"], "kind": component["kind"], "repo": repo, "revision": revision, "requested_url": url})
    return items


def fixed_model_info_items() -> list[dict[str, str]]:
    """Return the five fixed model-info API identities, never README URLs."""
    return [
        {
            "component": component["component"],
            "kind": component["kind"],
            "repo": component["repo"],
            "revision": component["revision"],
            "requested_url": f"https://huggingface.co/api/models/{component['repo']}?revision={component['revision']}",
        }
        for component in license_gate.fixed_component_identities()
        if component["kind"] == "model"
    ]


def _validate_license_url(item: dict[str, str], value: str, initial: bool) -> None:
    expected = item["requested_url"]
    if initial and value != expected: raise AuditError("initial fixed LICENSE URL drifted")
    parsed = urlparse(value); base = urlparse(expected)
    try: port = parsed.port
    except ValueError as exc: raise AuditError("unsafe LICENSE URL port") from exc
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment or port not in (None, 443): raise AuditError("unsafe LICENSE URL")
    if initial and (parsed.hostname != base.hostname or parsed.path != base.path): raise AuditError("initial LICENSE host/path drifted")
    if not initial and not (value == expected or (item["kind"] == "model" and parsed.hostname in HF_HOSTS and parsed.path == f"/{item['repo']}/resolve/{item['revision']}/LICENSE")): raise AuditError("LICENSE response is not the fixed path")


class _LicenseRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, str], trace: list[str]) -> None:
        super().__init__(); self.item = item; self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS: raise AuditError("fixed LICENSE redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        _validate_license_url(self.item, resolved, False)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def fetch_license(item: dict[str, str], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    url = item["requested_url"]; _validate_license_url(item, url, True)
    trace = [url]
    if fetcher:
        final, body = fetcher(url)
        if final != url: trace.append(final)
    else:
        opener = build_opener(_LicenseRedirects(item, trace))
        with opener.open(Request(url, headers={"User-Agent": "vokra-qwen3-tts-audit/1"}), timeout=30) as response:
            final, body = response.geturl(), response.read(MAX_LICENSE_BYTES + 1)
    _validate_license_url(item, final, False)
    if not isinstance(body, bytes) or len(body) > MAX_LICENSE_BYTES: raise AuditError("fixed LICENSE response exceeds bound")
    return {**item, "final_url": final, "redirect_trace": trace, "resolved_host": urlparse(final).hostname, "resolved_path": urlparse(final).path, "acquired_bytes": True, "size": len(body), "sha256": sha256_bytes(body), "content_base64": base64.b64encode(body).decode("ascii"), "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"}


def _validate_model_info_url(item: dict[str, str], value: str, initial: bool) -> None:
    """Allow only the exact HF API path/query for one pinned repository."""
    expected = item["requested_url"]
    if initial and value != expected:
        raise AuditError("initial HF model-info URL drifted")
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise AuditError("unsafe HF model-info URL port") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in HF_API_HOSTS
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.fragment
        or parsed.path != f"/api/models/{item['repo']}"
    ):
        raise AuditError("unsafe HF model-info URL")
    try:
        query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise AuditError("malformed HF model-info query") from exc
    if query != [("revision", item["revision"])]:
        raise AuditError("HF model-info query is not the fixed revision")


class _ModelInfoRedirects(HTTPRedirectHandler):
    def __init__(self, item: dict[str, str], trace: list[str]) -> None:
        super().__init__()
        self.item = item
        self.trace = trace

    def redirect_request(self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request:
        if len(self.trace) - 1 >= MAX_SDIST_REDIRECTS:
            raise AuditError("HF model-info redirect limit exceeded")
        resolved = urljoin(request.full_url, newurl)
        _validate_model_info_url(self.item, resolved, False)
        self.trace.append(resolved)
        return super().redirect_request(request, fp, code, msg, headers, resolved)


def _model_info_projection(item: dict[str, str], body: bytes) -> dict[str, Any]:
    """Parse only the authenticated API fields needed for license evidence."""
    if not isinstance(body, bytes) or len(body) > MAX_MODEL_INFO_BYTES:
        raise AuditError("HF model-info response exceeds bound")
    try:
        payload = license_gate.strict_json_loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise AuditError("HF model-info response is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AuditError("HF model-info response is not an object")
    required = {"id", "sha", "private", "gated", "disabled", "cardData", "siblings"}
    if not required.issubset(payload):
        raise AuditError("HF model-info response is missing required fields")
    if payload["id"] != item["repo"]:
        raise AuditError("HF model-info repository identity drifted")
    if not isinstance(payload["sha"], str) or not license_gate.HEX40.fullmatch(payload["sha"]):
        raise AuditError("HF model-info returned sha is malformed")
    if payload["sha"] != item["revision"]:
        raise AuditError("HF model-info returned sha does not match pinned revision")
    if payload["private"] is not False or payload["gated"] is not False:
        raise AuditError("HF model-info repository is private or gated")
    if payload["disabled"] is not False:
        raise AuditError("HF model-info repository is disabled")
    siblings = payload["siblings"]
    if not isinstance(siblings, list) or not siblings or len(siblings) > MAX_MEMBERS:
        raise AuditError("HF model-info siblings tree is missing, empty, or oversized")
    tree_files: list[str] = []
    for entry in siblings:
        if not isinstance(entry, dict) or set(entry) != {"rfilename"} or not isinstance(entry["rfilename"], str):
            raise AuditError("HF model-info sibling schema drifted")
        path = entry["rfilename"]
        if (
            not path
            or "\x00" in path
            or "\\" in path
            or path.startswith("/")
            or path.endswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or str(PurePosixPath(path)) != path
        ):
            raise AuditError("HF model-info sibling path is unsafe")
        if path in tree_files:
            raise AuditError("HF model-info sibling tree contains duplicate paths")
        if _license_candidate(path):
            raise AuditError("HF model-info sibling tree contains a license file; API fallback is refused")
        tree_files.append(path)
    tree_files.sort()
    card_data = payload["cardData"]
    if not isinstance(card_data, dict) or card_data.get("license") != license_gate.MODEL_LICENSE_METADATA_POLICY["required_license"]:
        raise AuditError("HF model-info cardData.license is not the reviewed Apache tag")
    # Do not retain or inspect README text or arbitrary cardData fields.  The
    # returned projection is intentionally the only accepted metadata source.
    return {
        "schema": license_gate.MODEL_LICENSE_METADATA_POLICY["schema"],
        "component": item["component"],
        "repo": item["repo"],
        "returned_repo": payload["id"],
        "requested_revision": item["revision"],
        "returned_sha": payload["sha"],
        "license": card_data["license"],
        "license_source": "HF_API_CARD_DATA_LICENSE",
        "tree_file_count": len(tree_files),
        "tree_files": tree_files,
        "tree_files_sha256": sha256_bytes(canonical_json(tree_files).encode("utf-8")),
        "payload_sha256": sha256_bytes(body),
        "payload_size": len(body),
    }


def fetch_model_license_metadata(item: dict[str, str], fetcher: Callable[[str], tuple[str, bytes]] | None = None) -> dict[str, Any]:
    """Fetch and authenticate one exact HF model-info license projection."""
    url = item["requested_url"]
    _validate_model_info_url(item, url, True)
    trace = [url]
    if fetcher:
        final, body = fetcher(url)
        if final != url:
            trace.append(final)
    else:
        opener = build_opener(_ModelInfoRedirects(item, trace))
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "vokra-qwen3-tts-audit/1"})
        with opener.open(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > MAX_MODEL_INFO_BYTES):
                raise AuditError("HF model-info Content-Length exceeds bound")
            final, body = response.geturl(), response.read(MAX_MODEL_INFO_BYTES + 1)
    _validate_model_info_url(item, final, False)
    projection = _model_info_projection(item, body)
    return {
        **item,
        "final_url": final,
        "redirect_trace": trace,
        "resolved_host": urlparse(final).hostname,
        "resolved_path": urlparse(final).path,
        **projection,
    }


def audit_model_licenses(
    *,
    license_fetcher: Callable[[dict[str, str]], dict[str, Any]] = fetch_license,
    metadata_fetcher: Callable[[dict[str, str]], dict[str, Any]] = fetch_model_license_metadata,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []; metadata_records: list[dict[str, Any]] = []; failures: list[str] = []
    for item in fixed_license_items():
        try: records.append(license_fetcher(item))
        except (OSError, UnicodeError, ValueError) as exc:
            error = _controlled(exc)
            records.append({**item, "final_url": None, "resolved_host": None, "resolved_path": None, "acquired_bytes": False, "size": None, "sha256": None, "content_base64": None, "license_classification": "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY", "error": error})
            if item["kind"] == "source" or error != {"kind": "HTTP_ERROR", "status": 404}:
                failures.append(f"BLOCKED_FACTUAL_LICENSE_PATH: {item['component']}: {error}")
                continue
            metadata_item = next(candidate for candidate in fixed_model_info_items() if candidate["component"] == item["component"])
            try:
                metadata_records.append(metadata_fetcher(metadata_item))
            except (OSError, UnicodeError, ValueError) as metadata_error:
                metadata_error_value = _controlled(metadata_error)
                metadata_records.append({**metadata_item, "final_url": None, "redirect_trace": [metadata_item["requested_url"]], "resolved_host": None, "resolved_path": None, "schema": license_gate.MODEL_LICENSE_METADATA_POLICY["schema"], "returned_repo": None, "requested_revision": metadata_item["revision"], "returned_sha": None, "license": None, "license_source": None, "tree_file_count": None, "tree_files": None, "tree_files_sha256": None, "payload_sha256": None, "payload_size": None, "error": metadata_error_value})
                failures.append(f"BLOCKED_FACTUAL_MODEL_METADATA_LICENSE: {item['component']}: {metadata_error_value}")
    return records, metadata_records, failures


def requested_model_metadata_urls(records: list[dict[str, Any]]) -> list[str]:
    """List only model-info requests actually attempted during this audit."""
    return [record["requested_url"] for record in records]


def gate_snapshot(project: Path, lock_bytes: bytes, project_bytes: bytes) -> dict[str, Any]:
    manifest_path = project / "license_gate_manifest.json"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("gate_version") != license_gate.GATE_VERSION: raise AuditError("license gate manifest schema drifted")
    if sha256_bytes(lock_bytes) != license_gate.LOCK_SHA256 or sha256_bytes(project_bytes) != license_gate.PYPROJECT_SHA256: raise AuditError("reviewed lock/project digest drifted")
    lock = tomllib.loads(lock_bytes.decode()); project_data = tomllib.loads(project_bytes.decode())
    license_gate._validate_lock_shape(lock, project_data)
    rows = license_gate.package_rows(lock)
    if license_gate.canonical_digest(rows) != manifest.get("package_rows_sha256"): raise AuditError("manifest package scope drifted")
    components = license_gate.fixed_component_identities()
    expected = {x["component"]: x for x in components}
    if manifest.get("identities") != license_gate.FIXED_IDENTITIES: raise AuditError("fixed model identities drifted")
    if not isinstance(manifest.get("component_rows"), list) or {x.get("component") for x in manifest["component_rows"]} != set(expected): raise AuditError("component rows do not cover fixed identities")
    unresolved = [x["name"] + "==" + x["version"] for x in manifest.get("review_rows", []) if x.get("status") != "REVIEWED" or x.get("license") in license_gate.PLACEHOLDER_SENTINELS]
    unresolved += [x["component"] for x in manifest.get("component_rows", []) if x.get("status") != "REVIEWED" or x.get("license") in license_gate.PLACEHOLDER_SENTINELS]
    return {"manifest_sha256": sha256_file(manifest_path), "package_rows_sha256": manifest.get("package_rows_sha256"), "review_rows_sha256": manifest.get("review_rows_sha256"), "component_rows_sha256": manifest.get("component_rows_sha256"), "approval_scope_sha256": manifest.get("approval_scope_sha256"), "status": "BLOCKED_UNRESOLVED_REVIEW" if unresolved else "REVIEWED", "unresolved_rows": sorted(unresolved)}


def repository_identity(project: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(["git", "-C", str(project), "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditError(f"git checkout identity unavailable: {type(exc).__name__}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not root or not Path(root).is_dir():
        raise AuditError("git checkout root/HEAD identity is malformed")
    return {"root": str(Path(root).resolve()), "head": head, "clean": not bool(status), "audit_script_sha256": sha256_file(Path(__file__).resolve())}


def canonicalize_uncreated(path: Path) -> Path:
    """Canonicalize an absent path while rejecting every symlink in its ancestry."""
    if not path.is_absolute(): raise AuditError("output path must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink(): raise AuditError("output path has symlink ancestry")
    return path.resolve(strict=False)


def production_preflight(project: Path, output: Path) -> None:
    """Direct Python entrypoint gate; the shell wrapper is not a trust boundary."""
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1": raise AuditError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if platform.system() != "Linux" or platform.machine().casefold() != "x86_64": raise AuditError("Linux x86_64 VAST host is required")
    if sys.version_info[:2] != (3, 12): raise AuditError(f"Python runtime is not 3.12: {platform.python_version()}")
    validate_output_path(project, output)


def validate_output_path(project: Path, output: Path) -> None:
    """Validate output safety independently of the host/platform gate."""
    if not output.is_absolute() or output.exists() or output.is_symlink(): raise AuditError("output must be an absent absolute path")
    output_canonical = canonicalize_uncreated(output)
    project_canonical = canonicalize_uncreated(project.resolve())
    repository = repository_identity(project)
    checkout = canonicalize_uncreated(Path(repository["root"]))
    if paths_overlap(output_canonical, checkout) or paths_overlap(output_canonical, project_canonical):
        raise AuditError("output overlaps checkout or parity project")


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def audit_environment(project: Path, fetch_model_licenses: bool) -> dict[str, Any]:
    project_path = project / "pyproject.toml"; lock_path = project / "uv.lock"
    project_bytes = project_path.read_bytes(); lock_bytes = lock_path.read_bytes()
    project_data = tomllib.loads(project_bytes.decode()); lock = tomllib.loads(lock_bytes.decode())
    gate = gate_snapshot(project, lock_bytes, project_bytes)
    repository = repository_identity(project)
    env = {"implementation_name": sys.implementation.name, "platform_machine": platform.machine().casefold(), "sys_platform": sys.platform}
    expected_rows, inactive = active_rows(lock, env)
    expected = [identity(row["name"], row["version"]) for row in expected_rows]
    records: list[metadata.Distribution] = list(metadata.distributions())
    installed = [identity(d.metadata.get("Name", ""), d.version) for d in records if d.metadata.get("Name")]
    counts = Counter(expected); actual_counts = Counter(installed)
    missing = sorted((counts - actual_counts).elements()); unexpected = sorted((actual_counts - counts).elements())
    by_id: dict[str, list[metadata.Distribution]] = {}
    for dist in records:
        if dist.metadata.get("Name"): by_id.setdefault(identity(dist.metadata["Name"], dist.version), []).append(dist)
    packages: list[dict[str, Any]] = []; failures: list[str] = []
    locked_names = {row["name"] for row in lock.get("package", []) if isinstance(row, dict)}
    forbidden_locked = sorted(locked_names & set(license_gate.FORBIDDEN_PACKAGES))
    if forbidden_locked:
        failures.append(f"forbidden packages are locked: {forbidden_locked}")
    for row in expected_rows:
        key = identity(row["name"], row["version"]); candidates = by_id.get(key, [])
        if len(candidates) != 1:
            failures.append(f"installed closure mismatch: {key}"); packages.append({"lock": row, "installed": None}); continue
        dist = candidates[0]; files, unsafe = publisher_files(dist); native, native_unsafe = native_files(dist); sdist = None
        declared = dist.metadata.get("License-Expression") or dist.metadata.get("License")
        classifiers = sorted(x.removeprefix("License :: ") for x in dist.metadata.get_all("Classifier", []) if x.startswith("License :: "))
        if not files:
            try: sdist = fetch_sdist(row)
            except (OSError, UnicodeError, ValueError) as exc: sdist = {"status": "BLOCKED_FACTUAL_SDIST_LICENSE_PATH", "error": _controlled(exc)}; failures.append(f"BLOCKED_FACTUAL_SDIST_LICENSE_PATH: {key}")
        if not declared and not classifiers and not (sdist and sdist.get("license_files")): failures.append(f"missing factual license metadata: {key}")
        if unsafe or native_unsafe: failures.append(f"unsafe installed file path: {key}")
        packages.append({"lock": {"name": row["name"], "version": row["version"], "source": row["source"], "sdist": row.get("sdist"), "wheels": row.get("wheels", [])}, "installed": {"name": dist.metadata.get("Name"), "version": dist.version, "license": declared.strip() if isinstance(declared, str) and declared.strip() else None, "license_classifiers": classifiers, "publisher_files": files, "publisher_files_unsafe": unsafe, "native_files": native, "native_files_unsafe": native_unsafe, "sdist_license_evidence": sdist, "location": str(Path(dist.locate_file("")))}})
    if missing or unexpected: failures.append(f"installed closure mismatch: {canonical_json({'missing': missing, 'unexpected': unexpected})}")
    if sys.version_info[:2] != (3, 12): failures.append(f"Python runtime is not 3.12: {platform.python_version()}")
    if sys.platform != "linux" or platform.machine().casefold() not in {"x86_64", "amd64"}: failures.append("audit host is not Linux x86_64")
    if gate["status"] != "REVIEWED": failures.append("license gate remains factually BLOCKED")
    if not repository["clean"]: failures.append("git checkout is dirty")
    model_records, model_metadata, model_failures = audit_model_licenses() if fetch_model_licenses else ([], [], [])
    failures.extend(model_failures)
    return {"schema": SCHEMA, "status": "BLOCKED" if failures else "PASS", "environment": {"python": platform.python_version(), "platform": sys.platform, "machine": platform.machine(), "model_code_imported": False, "cargo_invoked": False, "upload_performed": False}, "repository": repository, "project": {"name": project_data["project"]["name"], "version": project_data["project"]["version"], "pyproject_sha256": sha256_bytes(project_bytes), "uv_lock_sha256": sha256_bytes(lock_bytes)}, "license_gate": gate, "closure": {"expected": sorted(expected), "installed": sorted(installed), "missing": missing, "unexpected": unexpected, "active_rows": len(expected_rows), "inactive_rows": len(inactive)}, "locked_rows": [{"name": x["name"], "version": x["version"], "source": x["source"]} for x in lock["package"]], "inactive_rows": [{"identity": identity(x["name"], x["version"]), "reason": reason} for x, reason in ((next(r for r in lock["package"] if _row_key(r) == key), reason) for key, reason in inactive.items())], "packages": packages, "fixed_source_model_decoder_license_paths": [{"component": x["component"], "kind": x["kind"], "repo": x["repo"], "revision": x["revision"], "requested_url": x["requested_url"]} for x in fixed_license_items()], "model_license_files": model_records, "model_license_metadata": model_metadata, "model_acquisition": {"policy": "fixed LICENSE paths plus exact HF model-info API cardData.license; no model weights", "requested_files": [x["requested_url"] for x in model_records], "requested_metadata": requested_model_metadata_urls(model_metadata), "non_license_files": [], "model_files": []}, "failures": sorted(set(failures))}


def validate_report(report: Any) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS or report.get("schema") != SCHEMA:
        raise AuditError("audit report schema is not exact")
    if report.get("status") not in {"PASS", "BLOCKED"} or not isinstance(report.get("failures"), list):
        raise AuditError("audit report status/failures are malformed")


def run(project: Path, output: Path, fetch_model_licenses: bool) -> int:
    try:
        production_preflight(project, output)
    except (AuditError, OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"qwen3-tts dependency audit: BLOCKED before evidence write: {_controlled(exc)}", file=sys.stderr)
        return 2
    try:
        report = audit_environment(project, fetch_model_licenses)
    except (AuditError, OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        try: repository = repository_identity(project)
        except (AuditError, OSError, ValueError, subprocess.SubprocessError): repository = {"root": None, "head": None, "clean": False, "audit_script_sha256": sha256_file(Path(__file__).resolve())}
        report = {"schema": SCHEMA, "status": "BLOCKED", "environment": {"python": None, "platform": None, "machine": None, "model_code_imported": False, "cargo_invoked": False, "upload_performed": False}, "repository": repository, "project": {"name": project.name, "version": None, "pyproject_sha256": None, "uv_lock_sha256": None}, "license_gate": {"status": "BLOCKED", "unresolved_rows": []}, "closure": {"expected": [], "installed": [], "missing": [], "unexpected": [], "active_rows": 0, "inactive_rows": 0}, "locked_rows": [], "inactive_rows": [], "packages": [], "fixed_source_model_decoder_license_paths": fixed_license_items(), "model_license_files": [], "model_license_metadata": [], "model_acquisition": {"policy": "fixed LICENSE paths plus exact HF model-info API cardData.license; no model weights", "requested_files": [], "requested_metadata": [], "non_license_files": [], "model_files": []}, "failures": [f"ENVIRONMENT_AUDIT_BLOCKED: {_controlled(exc)}"]}
    validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    if report.get("status") != "PASS": print("qwen3-tts dependency audit: BLOCKED: " + "; ".join(report.get("failures", [])), file=sys.stderr); return 2
    print(f"qwen3-tts dependency audit: PASS ({output})"); return 0


def self_test() -> int:
    project = Path(__file__).resolve().parent
    lock_path = project / "uv.lock"; project_path = project / "pyproject.toml"
    lock_bytes = lock_path.read_bytes(); project_bytes = project_path.read_bytes()
    lock = tomllib.loads(lock_bytes.decode("utf-8")); project_data = tomllib.loads(project_bytes.decode("utf-8"))
    gate = gate_snapshot(project, lock_bytes, project_bytes)
    linux_rows, linux_inactive = active_rows(lock, {"implementation_name": "cpython", "platform_machine": "x86_64", "sys_platform": "linux"})
    assert project_data["project"]["name"] == "vokra-qwen3-tts-parity"
    assert len(linux_rows) > 0 and len(linux_rows) + len(linux_inactive) == len(lock["package"])
    assert sum(row.get("source") == {"virtual": "."} for row in lock["package"]) == 1
    assert "setuptools" not in {row["name"] for row in lock["package"]}
    assert "setuptools" in license_gate.FORBIDDEN_PACKAGES
    tampered_lock = json.loads(json.dumps(lock))
    torch_row = next(row for row in tampered_lock["package"] if row["name"] == "torch")
    torch_row.setdefault("dependencies", []).append({"name": "setuptools"})
    try:
        active_rows(tampered_lock, {"implementation_name": "cpython", "platform_machine": "x86_64", "sys_platform": "linux"})
    except AuditError:
        pass
    else:
        raise AssertionError("setuptools dependency reintroduction was accepted")
    assert gate["status"] == "BLOCKED_UNRESOLVED_REVIEW"
    assert "accelerate==1.12.0" in gate["unresolved_rows"]
    assert identity("foo_bar", "1.0") == "foo-bar==1.0"
    assert marker_matches("sys_platform == 'linux' and platform_machine == 'x86_64'", {"sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython"})
    try: marker_matches("python_version == '3.12'", {"sys_platform": "linux", "platform_machine": "x86_64", "implementation_name": "cpython"})
    except AuditError: pass
    else: raise AssertionError("unsupported marker variable accepted")
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-audit-missing-") as directory:
        try: load_json(Path(directory) / "missing.json")
        except AuditError: pass
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-audit-self-test-") as directory:
        path = Path(directory) / "dup.json"; path.write_text('{"a":1,"a":2}', encoding="utf-8")
        try: load_json(path)
        except AuditError: pass
        else: raise AssertionError("duplicate JSON key accepted")
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w:gz") as archive:
            directory_entry = tarfile.TarInfo("demo/"); directory_entry.type = tarfile.DIRTYPE; archive.addfile(directory_entry)
            member = tarfile.TarInfo("demo/LICENCE"); payload = b"licence"; member.size = len(payload); archive.addfile(member, io.BytesIO(payload))
        url = "https://files.pythonhosted.org/packages/demo-1.tar.gz"; artifact = {"url": url, "hash": "sha256:" + sha256_bytes(body.getvalue()), "size": len(body.getvalue()), "upload-time": "2026-01-01T00:00:00Z"}
        result = fetch_sdist({"name": "demo", "version": "1", "sdist": artifact}, lambda value: (value, body.getvalue()))
        assert result["license_files"][0]["path"] == "demo/LICENCE"
        for bad in (url.replace("https://", "https://audit-user@", 1), url.replace("files.pythonhosted.org", "files.pythonhosted.org:8443", 1), url + "?download=1", url + "#fragment"):
            try: fetch_sdist({"name": "demo", "version": "1", "sdist": {**artifact, "url": bad}}, lambda value: (_ for _ in ()).throw(AssertionError("unsafe sdist URL reached fetcher")))
            except AuditError: pass
            else: raise AssertionError(f"unsafe initial sdist URL accepted: {bad}")
        item = fixed_license_items()[0]; assert fetch_license(item, lambda value: (value, b"source bytes"))["license_classification"] == "UNCLASSIFIED_PRIMARY_SOURCE_BYTES_ONLY"
        try: fetch_license(item, lambda value: (value.replace("raw.githubusercontent.com", "example.invalid"), b"x"))
        except AuditError: pass
        else: raise AssertionError("unsafe fixed LICENSE redirect accepted")
        redirects = _LicenseRedirects(item, [item["requested_url"]])
        safe_redirect = item["requested_url"]
        assert redirects.redirect_request(Request(safe_redirect), None, 302, "found", {}, safe_redirect)
        for bad in (
            "http://raw.githubusercontent.com/QwenLM/Qwen3-TTS/" + item["revision"] + "/LICENSE",
            item["requested_url"] + "?download=1",
            item["requested_url"].replace("raw.githubusercontent.com", "raw.githubusercontent.com:8443"),
            item["requested_url"].replace("/LICENSE", "/README.md"),
        ):
            try: _LicenseRedirects(item, [item["requested_url"]]).redirect_request(Request(item["requested_url"]), None, 302, "found", {}, bad)
            except AuditError: pass
            else: raise AssertionError(f"unsafe fixed LICENSE redirect accepted: {bad}")
        bounded = _LicenseRedirects(item, [item["requested_url"]] * (MAX_SDIST_REDIRECTS + 1))
        try: bounded.redirect_request(Request(item["requested_url"]), None, 302, "found", {}, item["requested_url"])
        except AuditError: pass
        else: raise AssertionError("overlong fixed LICENSE redirect accepted")
        model_item = fixed_model_info_items()[0]
        model_payload = json.dumps({
            "id": model_item["repo"],
            "sha": model_item["revision"],
            "private": False,
            "gated": False,
            "disabled": False,
            "siblings": [{"rfilename": "config.json"}],
            "cardData": {"license": "apache-2.0", "README": "not a license source"},
        }, separators=(",", ":")).encode("utf-8")
        model_result = fetch_model_license_metadata(model_item, lambda value: (value, model_payload))
        assert model_result["returned_sha"] == model_item["revision"]
        assert model_result["license"] == "apache-2.0"
        assert model_result["license_source"] == "HF_API_CARD_DATA_LICENSE"
        assert model_result["payload_size"] == len(model_payload)
        assert model_result["tree_file_count"] == 1
        assert model_result["tree_files"] == ["config.json"]
        assert "content_base64" not in model_result
        try:
            fetch_model_license_metadata(model_item, lambda value: (value, b'{"id":"x","id":"x"}'))
        except AuditError:
            pass
        else:
            raise AssertionError("duplicate HF model-info JSON keys were accepted")
        for bad in (
            model_item["requested_url"].replace("https://huggingface.co", "https://example.invalid"),
            model_item["requested_url"].replace("?revision=", "?other="),
            model_item["requested_url"] + "&revision=" + model_item["revision"],
            model_item["requested_url"].replace("/api/models/", "/README.md/"),
        ):
            try:
                fetch_model_license_metadata({**model_item, "requested_url": bad}, lambda value: (_ for _ in ()).throw(AssertionError("unsafe model-info URL reached fetcher")))
            except AuditError:
                pass
            else:
                raise AssertionError(f"unsafe initial model-info URL accepted: {bad}")
        for invalid in (
            {**json.loads(model_payload), "sha": "0" * 40},
            {**json.loads(model_payload), "id": "other/model"},
            {**json.loads(model_payload), "private": True},
            {key: value for key, value in json.loads(model_payload).items() if key != "disabled"},
            {**json.loads(model_payload), "disabled": True},
            {**json.loads(model_payload), "cardData": {"license": "mit"}},
            {**json.loads(model_payload), "cardData": {"README": "apache-2.0"}},
            {**json.loads(model_payload), "cardData": {"license": ["apache-2.0"]}},
            {**json.loads(model_payload), "siblings": []},
            {**json.loads(model_payload), "siblings": [{"rfilename": "../config.json"}]},
            {**json.loads(model_payload), "siblings": [{"rfilename": "config\\.json"}]},
            {**json.loads(model_payload), "siblings": [{"rfilename": "config.json"}, {"rfilename": "config.json"}]},
            {**json.loads(model_payload), "siblings": [{"rfilename": "LICENSE"}]},
            {**json.loads(model_payload), "siblings": [{"rfilename": "NOTICE.txt"}]},
            {**json.loads(model_payload), "siblings": [{"rfilename": "config.json", "size": 1}]},
        ):
            try:
                fetch_model_license_metadata(model_item, lambda value, invalid=invalid: (value, json.dumps(invalid).encode("utf-8")))
            except AuditError:
                pass
            else:
                raise AssertionError("invalid HF model-info projection was accepted")
        try:
            fetch_model_license_metadata(model_item, lambda value: (value, b"x" * (MAX_MODEL_INFO_BYTES + 1)))
        except AuditError:
            pass
        else:
            raise AssertionError("oversized HF model-info response was accepted")
        def synthetic_license_ok(value: dict[str, str]) -> dict[str, Any]:
            return {"component": value["component"], "status": "LICENSE_OK"}

        def synthetic_license_404(value: dict[str, str]) -> dict[str, Any]:
            if value["kind"] == "model":
                raise HTTPError(value["requested_url"], 404, "missing", {}, None)
            return synthetic_license_ok(value)

        def synthetic_metadata_ok(value: dict[str, str]) -> dict[str, Any]:
            return {"component": value["component"], "requested_url": value["requested_url"], "status": "METADATA_OK"}

        _, fallback_metadata, fallback_failures = audit_model_licenses(
            license_fetcher=synthetic_license_404,
            metadata_fetcher=synthetic_metadata_ok,
        )
        assert len(fallback_metadata) == len(fixed_model_info_items()) and not fallback_failures
        assert requested_model_metadata_urls(fallback_metadata) == [item["requested_url"] for item in fixed_model_info_items()]
        _, direct_metadata, direct_failures = audit_model_licenses(
            license_fetcher=synthetic_license_ok,
            metadata_fetcher=lambda value: (_ for _ in ()).throw(AssertionError("metadata fallback ran after LICENSE success")),
        )
        assert direct_metadata == [] and requested_model_metadata_urls(direct_metadata) == [] and not direct_failures
        def synthetic_metadata_500(value: dict[str, str]) -> dict[str, Any]:
            raise HTTPError(value["requested_url"], 500, "server error", {}, None)

        _, failed_metadata, failed_failures = audit_model_licenses(
            license_fetcher=synthetic_license_404,
            metadata_fetcher=synthetic_metadata_500,
        )
        assert len(failed_metadata) == len(fixed_model_info_items())
        assert len(failed_failures) == len(fixed_model_info_items())
        assert all("BLOCKED_FACTUAL_MODEL_METADATA_LICENSE" in failure for failure in failed_failures)
        installed_root = Path(directory) / "installed"; installed_root.mkdir()
        installed_license = installed_root / "LICENCE"; installed_license.write_bytes(b"installed licence")
        class InstalledDistribution:
            files = ["LICENCE"]
            def locate_file(self, entry: Any) -> Path: return installed_root / str(entry)
        installed_files, unsafe = publisher_files(InstalledDistribution())
        assert not unsafe and installed_files[0]["path"] == "LICENCE"
    try: validate_report({"schema": SCHEMA, "status": "BLOCKED", "failures": [], "unknown": True})
    except AuditError: pass
    else: raise AssertionError("unknown report field accepted")
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() and not Path("/private/tmp").is_symlink() else None
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-audit-paths-", dir=temp_parent) as directory:
        root = Path(directory); safe = root / "nested" / "report.json"
        validate_output_path(project, safe); assert canonicalize_uncreated(safe) == safe
        existing = root / "existing.json"; existing.write_text("x", encoding="utf-8")
        for bad in (Path("relative.json"), existing, project / "in-repo.json"):
            try:
                validate_output_path(project, bad)
            except AuditError: pass
            else: raise AssertionError(f"unsafe output accepted: {bad}")
        linked = root / "link"; linked.symlink_to(root, target_is_directory=True)
        try: validate_output_path(project, linked / "report.json")
        except AuditError: pass
        else: raise AssertionError("symlinked output ancestry accepted")
    print("qwen3-tts dependency audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--fetch-model-licenses", action="store_true"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test:
        if args.project or args.output or args.fetch_model_licenses: parser.error("--self-test accepts no audit arguments")
        return self_test()
    if args.project is None or args.output is None: parser.error("--project and --output are required")
    return run(args.project, args.output, args.fetch_model_licenses)


if __name__ == "__main__": raise SystemExit(main())
