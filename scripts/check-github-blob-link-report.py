#!/usr/bin/env -S uv run --script
"""Fail-closed final gate for the pinned lychee v0.24.2 JSON report.

The ordinary lychee scan remains authoritative for every link.  This gate has
one deliberately narrow recovery path: a GitHub ``/blob/<single-segment-ref>/
<path>`` response whose original status is exactly HTTP 503 may be rechecked
through the authenticated Contents API.  The API response must identify the
same repository, ref, path, bytes, Git blob SHA, and requested line anchor.
No URL is ignored and no other status or host is recovered.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_API_BYTES = 1 * 1024 * 1024
MAX_FALLBACK_URLS = 10
DEFAULT_DEADLINE_SECONDS = 60.0
GITHUB_API_ORIGIN = "https://api.github.com"
_OWNER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REF_RE = re.compile(r"^[^/\\?#]+$")
_ANCHOR_RE = re.compile(r"^L([1-9][0-9]*)(?:-L([1-9][0-9]*))?$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_REF_PAGE_SIZE = 100


class GateError(Exception):
    """An intentionally non-sensitive validation error."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise GateError("non-finite JSON number")


def _reject_nonfinite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise GateError("non-finite JSON number")
    return result


def load_json_bytes(raw: bytes) -> Any:
    if len(raw) > MAX_REPORT_BYTES:
        raise GateError("report exceeds size limit")
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
            parse_float=_reject_nonfinite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("malformed JSON report") from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(obj: Mapping[str, Any], key: str) -> int:
    value = obj.get(key)
    if not _is_int(value) or value < 0:
        raise GateError("invalid report counter")
    return value


def _require_map(obj: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = obj.get(key)
    if not isinstance(value, dict):
        raise GateError("invalid report map")
    return value


def _validate_duration(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"secs", "nanos"}:
        raise GateError("invalid response duration")
    if not _is_int(value["secs"]) or value["secs"] < 0:
        raise GateError("invalid response duration")
    if not _is_int(value["nanos"]) or not 0 <= value["nanos"] < 1_000_000_000:
        raise GateError("invalid response duration")


def _validate_response_body(value: Any) -> tuple[str, int | None]:
    if not isinstance(value, dict):
        raise GateError("invalid response body")
    required = {"url", "status"}
    if not required.issubset(value):
        raise GateError("incomplete response body")
    if set(value) - {"url", "status", "redirects", "remap", "span", "duration"}:
        raise GateError("unknown response body field")
    url = value["url"]
    status = value["status"]
    span = value.get("span")
    if not isinstance(url, str) or not url:
        raise GateError("invalid response URL")
    if not isinstance(status, dict) or not isinstance(status.get("text"), str):
        raise GateError("invalid response status")
    code = status.get("code")
    # http::StatusCode accepts the full 100..999 range; lychee preserves
    # unknown-but-well-formed codes in JSON (for example 999).
    if code is not None and (not _is_int(code) or not 100 <= code <= 999):
        raise GateError("invalid response status code")
    if "span" in value and span is not None:
        if not isinstance(span, dict) or not _is_int(span.get("line")) or span["line"] < 1:
            raise GateError("invalid response span")
        if "column" in span and span["column"] is not None and (
            not _is_int(span["column"]) or span["column"] < 1
        ):
            raise GateError("invalid response span")
    if "duration" in value and value["duration"] is not None:
        _validate_duration(value["duration"])
    if "remap" in value and value["remap"] is not None:
        remap = value["remap"]
        if not isinstance(remap, dict) or not isinstance(remap.get("original"), str) or not isinstance(remap.get("new"), str):
            raise GateError("invalid response remap")
    if "redirects" in value and value["redirects"] is not None:
        redirects = value["redirects"]
        if not isinstance(redirects, dict) or not isinstance(redirects.get("origin"), str) or not isinstance(redirects.get("redirects"), list):
            raise GateError("invalid response redirects")
        for redirect in redirects["redirects"]:
            if not isinstance(redirect, dict) or not isinstance(redirect.get("url"), str) or not _is_int(redirect.get("code")):
                raise GateError("invalid response redirects")
    return url, code


def validate_report(report: Any, exit_code: int) -> tuple[list[dict[str, Any]], int]:
    """Validate the exact v0.24.2 OutputStats shape and return error bodies."""
    if not isinstance(report, dict):
        raise GateError("report is not an object")
    counters = (
        "total", "unique", "successful", "unknown", "unsupported", "timeouts",
        "redirects", "remaps", "excludes", "errors", "cached",
    )
    values = {key: _require_int(report, key) for key in counters}
    for key in (
        "success_map", "error_map", "timeout_map", "suggestion_map",
        "redirect_map", "excluded_map",
    ):
        _require_map(report, key)
    duration = report.get("duration")
    _validate_duration(duration)
    if not isinstance(report.get("detailed_stats"), bool):
        raise GateError("invalid detailed_stats")
    if values["total"] <= 0:
        raise GateError("empty lychee report")
    if values["unique"] <= 0 or values["unique"] > values["total"]:
        raise GateError("inconsistent report totals")
    if values["successful"] + values["unknown"] + values["unsupported"] + values["timeouts"] + values["excludes"] + values["errors"] != values["total"]:
        raise GateError("inconsistent report status totals")
    if values["cached"] > values["total"]:
        raise GateError("inconsistent cached total")

    error_bodies: list[dict[str, Any]] = []
    for map_key, expected_count in (("error_map", values["errors"]), ("timeout_map", values["timeouts"])):
        mapping = report[map_key]
        body_count = 0
        for source, bodies in mapping.items():
            if not isinstance(source, str) or not source:
                raise GateError("invalid response map source")
            if not isinstance(bodies, list) or not bodies:
                raise GateError("invalid response map entry")
            for body in bodies:
                url, code = _validate_response_body(body)
                body_count += 1
                if map_key == "error_map":
                    error_bodies.append(body)
                if map_key == "timeout_map" and code is not None:
                    raise GateError("timeout has an HTTP status code")
        if expected_count == 0 and mapping:
            raise GateError("nonempty map with zero counter")
        # error_map/timeout_map are keyed by source and hold HashSet values in
        # v0.24.2, so identical responses can be deduplicated and produce
        # fewer bodies than the counter. More bodies than the counter cannot
        # be a valid OutputStats result.
        if body_count > expected_count:
            raise GateError("response map exceeds status counter")
        if expected_count > 0 and body_count == 0:
            raise GateError("missing failed response bodies")
    # The v0.24.2 formatter serializes all response maps, including successful
    # and excluded bodies when detailed_stats is enabled.  Validate their
    # response-body shape too, even though only error_map is recoverable.
    for map_key in ("success_map", "excluded_map"):
        for source, bodies in report[map_key].items():
            if not isinstance(source, str) or not source or not isinstance(bodies, list) or not bodies:
                raise GateError("invalid response map entry")
            for body in bodies:
                _validate_response_body(body)
    for map_key, keys in (("suggestion_map", {"original", "suggestion"}),):
        for source, entries in report[map_key].items():
            if not isinstance(source, str) or not source or not isinstance(entries, list) or not entries:
                raise GateError("invalid suggestion map entry")
            for entry in entries:
                if not isinstance(entry, dict) or not keys.issubset(entry) or any(
                    not isinstance(entry[key], str) for key in keys
                ):
                    raise GateError("invalid suggestion map entry")
    for source, entries in report["redirect_map"].items():
        if not isinstance(source, str) or not source or not isinstance(entries, list) or not entries:
            raise GateError("invalid redirect map entry")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("origin"), str) or not isinstance(entry.get("redirects"), list):
                raise GateError("invalid redirect map entry")
            for redirect in entry["redirects"]:
                if not isinstance(redirect, dict) or not isinstance(redirect.get("url"), str) or not _is_int(redirect.get("code")):
                    raise GateError("invalid redirect map entry")
    if exit_code == 0:
        if values["errors"] or values["timeouts"]:
            raise GateError("successful exit contradicts failed responses")
    elif exit_code == 2:
        if values["errors"] + values["timeouts"] == 0:
            raise GateError("failure exit has no failed responses")
    elif exit_code in (1, 3):
        raise GateError("lychee runtime or configuration failure")
    else:
        raise GateError("unknown lychee exit code")
    return error_bodies, values["timeouts"]


@dataclass(frozen=True)
class BlobTarget:
    url: str
    owner: str
    repo: str
    ref: str
    path: str
    fragment: str


def parse_blob_target(url: str) -> BlobTarget:
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in url):
        raise GateError("unsafe GitHub blob URL")
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "github.com" or parts.query:
        raise GateError("unsupported GitHub blob URL")
    if parts.username or parts.password or parts.port:
        raise GateError("unsupported GitHub blob URL")
    segments = parts.path.split("/")
    if len(segments) < 5 or segments[0] or segments[3] != "blob":
        raise GateError("unsupported GitHub blob URL")
    owner, repo, _, ref, *path_parts = segments[1:]
    if not _OWNER_RE.fullmatch(owner) or not _OWNER_RE.fullmatch(repo):
        raise GateError("invalid GitHub repository")
    if not _REF_RE.fullmatch(ref) or not path_parts or any(not item for item in path_parts):
        raise GateError("ambiguous GitHub ref or path")
    all_segments = (owner, repo, ref, *path_parts)
    if any(
        segment in {".", ".."}
        or any(ord(character) < 0x20 or ord(character) == 0x7F or character == "\\" for character in segment)
        for segment in all_segments
    ):
        raise GateError("unsafe GitHub ref or path")
    # A conventional fully-qualified ref such as refs/heads/main cannot be
    # split unambiguously from the file path in a /blob/ URL.  Do not guess a
    # split; the recovery contract accepts only a single-segment ref.
    if ref == "refs" and path_parts[0] in {"heads", "tags", "remotes"}:
        raise GateError("ambiguous GitHub ref or path")
    path = "/".join(path_parts)
    if unquote(path) != path or unquote(ref) != ref:
        raise GateError("encoded GitHub ref or path")
    if parts.fragment and not _ANCHOR_RE.fullmatch(parts.fragment):
        raise GateError("unsupported GitHub fragment")
    return BlobTarget(url, owner, repo, ref, path, parts.fragment)


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: HTTPResponse, code: int, msg: str, headers: Any, new_url: str) -> None:
        raise GateError("API redirect rejected")


def _read_limited(response: Any) -> bytes:
    header = response.headers.get("Content-Length")
    if header is not None:
        try:
            if int(header) > MAX_API_BYTES:
                raise GateError("API response exceeds size limit")
        except ValueError as exc:
            raise GateError("invalid API response length") from exc
    data = response.read(MAX_API_BYTES + 1)
    if len(data) > MAX_API_BYTES:
        raise GateError("API response exceeds size limit")
    return data


def _strict_b64(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise GateError("invalid API content")
    if any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n" for ch in value):
        raise GateError("invalid API content encoding")
    compact = value.replace("\r", "").replace("\n", "")
    if len(compact) % 4 or re.search(r"=[^=]", compact) or compact.count("=") > 2:
        raise GateError("invalid API content encoding")
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GateError("invalid API content encoding") from exc
    if base64.b64encode(decoded).decode("ascii") != compact:
        raise GateError("noncanonical API content encoding")
    return decoded


def _verify_content(target: BlobTarget, payload: Mapping[str, Any], requested_api_url: str) -> None:
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise GateError("API response is not a file")
    if "target" in payload or "submodule_git_url" in payload:
        raise GateError("API response contains non-file identity")
    expected_git_url = f"{GITHUB_API_ORIGIN}/repos/{target.owner}/{target.repo}/git/blobs/"
    expected_download_url = f"https://raw.githubusercontent.com/{target.owner}/{target.repo}/{target.ref}/{target.path}"
    if payload.get("name") != target.path.rsplit("/", 1)[-1]:
        raise GateError("API name mismatch")
    if payload.get("path") != target.path:
        raise GateError("API path mismatch")
    if payload.get("html_url") != target.url.split("#", 1)[0]:
        raise GateError("API ref identity mismatch")
    if payload.get("url") != requested_api_url:
        raise GateError("API request identity mismatch")
    content = _strict_b64(payload.get("content"))
    if len(content) != payload.get("size") or not _is_int(payload.get("size")):
        raise GateError("API size mismatch")
    expected_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if payload.get("sha") != expected_sha or not _SHA1_RE.fullmatch(payload["sha"]):
        raise GateError("API blob identity mismatch")
    if payload.get("git_url") != expected_git_url + expected_sha:
        raise GateError("API git identity mismatch")
    if payload.get("download_url") != expected_download_url:
        raise GateError("API download identity mismatch")
    links = payload.get("_links")
    if not isinstance(links, dict) or links.get("self") != requested_api_url or links.get("git") != expected_git_url + expected_sha or links.get("html") != target.url.split("#", 1)[0]:
        raise GateError("API links identity mismatch")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError("API content is not UTF-8") from exc
    if target.fragment:
        match = _ANCHOR_RE.fullmatch(target.fragment)
        assert match is not None
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        # GitHub line anchors count LF-delimited source lines. Do not let
        # splitlines() treat vertical-tab/form-feed/U+2028 as a line break.
        lines = text.split("\n")
        if text.endswith("\n"):
            lines.pop()
        if start > end or end > len(lines):
            raise GateError("GitHub line anchor is absent")


def recover_503_blob(
    target: BlobTarget,
    opener: Any,
    token: str | None,
    deadline: float,
) -> None:
    if not isinstance(token, str) or not token.strip():
        raise GateError("GITHUB_TOKEN is required for recovery")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in token):
        raise GateError("invalid GITHUB_TOKEN")
    if time.monotonic() >= deadline:
        raise GateError("fallback deadline exceeded")
    api_path = quote(target.path, safe="/")
    api_url = f"{GITHUB_API_ORIGIN}/repos/{quote(target.owner)}/{quote(target.repo)}/contents/{api_path}?ref={quote(target.ref, safe='')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vokra-ci-blob-gate/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _request_api_json(opener, api_url, headers, deadline)
    if not isinstance(payload, dict):
        raise GateError("API response is not an object")
    _verify_content(target, payload, api_url)
    _verify_ref_is_unambiguous(target, opener, headers, deadline)


def _request_api_json(
    opener: Any,
    api_url: str,
    headers: Mapping[str, str],
    deadline: float,
    *,
    reject_pagination: bool = False,
) -> Any:
    """Fetch one exact-origin API JSON response under the shared deadline."""
    if time.monotonic() >= deadline:
        raise GateError("fallback deadline exceeded")
    timeout = max(0.1, min(10.0, deadline - time.monotonic()))
    response = None
    try:
        # Keep construction inside the sanitizing try: urllib validates header
        # values here and must not expose a token-bearing ValueError.
        request = Request(api_url, headers=dict(headers), method="GET")
        response = opener.open(request, timeout=timeout)
        if getattr(response, "geturl", lambda: api_url)() != api_url:
            raise GateError("API redirect rejected")
        if getattr(response, "status", 200) != 200:
            raise GateError("API response was not successful")
        raw = _read_limited(response)
        if reject_pagination and getattr(response, "headers", {}).get("Link"):
            raise GateError("GitHub ref evidence is paginated")
        if time.monotonic() >= deadline:
            raise GateError("fallback deadline exceeded")
    except GateError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        raise GateError("API request failed") from exc
    except Exception as exc:
        # A test transport or a malformed runtime response must never expose
        # its exception representation, which could contain a token/header.
        raise GateError("API request failed") from exc
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    return load_json_bytes(raw)


def _verify_ref_is_unambiguous(
    target: BlobTarget,
    opener: Any,
    headers: Mapping[str, str],
    deadline: float,
) -> None:
    """Reject refs that blob URLs cannot prove without paginated evidence.

    The matching-refs endpoint is queried with a bounded page size.  A Link
    header or a full page is treated as incomplete rather than silently
    assuming that no longer slash-containing ref exists.
    """
    if _SHA1_RE.fullmatch(target.ref):
        return
    exact_count = 0
    for kind in ("heads", "tags"):
        refs_url = (
            f"{GITHUB_API_ORIGIN}/repos/{quote(target.owner)}/{quote(target.repo)}"
            f"/git/matching-refs/{kind}/{quote(target.ref, safe='')}"
            f"?per_page={_REF_PAGE_SIZE}"
        )
        payload = _request_api_json(opener, refs_url, headers, deadline, reject_pagination=True)
        if not isinstance(payload, list) or len(payload) >= _REF_PAGE_SIZE:
            raise GateError("GitHub ref evidence is incomplete")
        exact_ref = f"refs/{kind}/{target.ref}"
        for entry in payload:
            if not isinstance(entry, dict) or not isinstance(entry.get("ref"), str):
                raise GateError("invalid GitHub ref evidence")
            candidate = entry["ref"]
            if candidate == exact_ref:
                exact_count += 1
            elif candidate.startswith(exact_ref + "/"):
                raise GateError("ambiguous GitHub ref")
            elif not candidate.startswith(f"refs/{kind}/"):
                raise GateError("invalid GitHub ref evidence")
    if exact_count == 0:
        raise GateError("GitHub ref evidence missing exact ref")
    if exact_count != 1:
        raise GateError("ambiguous GitHub ref")


def check_report(
    report: Any,
    exit_code: int,
    *,
    token: str | None = None,
    opener: Any | None = None,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> bool:
    errors, timeout_count = validate_report(report, exit_code)
    if timeout_count:
        raise GateError("timeout responses remain unhandled")
    if exit_code == 0:
        return True
    opener = opener or build_opener(NoRedirectHandler())
    deadline = time.monotonic() + deadline_seconds
    candidates: dict[str, BlobTarget] = {}
    for body in errors:
        status = body["status"]
        if status.get("code") != 503:
            raise GateError("unhandled non-503 link failure")
        target = parse_blob_target(body["url"])
        candidates.setdefault(target.url, target)
    if not candidates:
        raise GateError("no recoverable 503 blob failure")
    if len(candidates) > MAX_FALLBACK_URLS:
        raise GateError("fallback URL limit exceeded")
    recovered: list[str] = []
    for target in candidates.values():
        recover_503_blob(target, opener, token, deadline)
        recovered.append(target.url)
    for url in recovered:
        print(f"Recovered original HTTP 503 via exact GitHub Contents file: {url}")
    return True


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--lychee-exit-code", required=True, type=int)
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    return parser


def main(argv: list[str] | None = None, *, opener: Any | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    try:
        if args.deadline_seconds <= 0 or not math.isfinite(args.deadline_seconds):
            raise GateError("invalid fallback deadline")
        with open(args.report, "rb") as handle:
            raw = handle.read(MAX_REPORT_BYTES + 1)
        report = load_json_bytes(raw)
        check_report(
            report,
            args.lychee_exit_code,
            token=os.environ.get("GITHUB_TOKEN"),
            opener=opener,
            deadline_seconds=args.deadline_seconds,
        )
        print("GitHub blob link gate passed")
        return 0
    except Exception:
        print("GitHub blob link gate failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
