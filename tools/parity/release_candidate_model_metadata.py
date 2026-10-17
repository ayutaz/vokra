#!/usr/bin/env python3
"""Fetch and validate only the two reviewed public HF model metadata records.

This helper never follows a model resolver URL and never downloads weights.  It
emits only the reviewed pin manifest plus a compact metadata identity receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


REVISIONS = {
    "whisper-base": ("vokra/whisper-base", "dddd93ff6301a82aa3750d398cb2e0a599cd9e83"),
    "silero-vad-v5": ("vokra/silero-vad-v5", "3670dfed054980307eeb1711c6329c29b80bdf14"),
}
MAX_API_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 32 * 1024
SOCKET_TIMEOUT = 30
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class MetadataError(RuntimeError):
    pass


def _load_smoke_module() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).with_name("release_candidate_cpu_smoke.py")
    spec = importlib.util.spec_from_file_location("release_candidate_cpu_smoke", path)
    if spec is None or spec.loader is None:
        raise MetadataError(f"cannot load smoke validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _api_url(repo: str, revision: str) -> str:
    return f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"


ALLOWED_API_URLS = frozenset(_api_url(repo, revision) for repo, revision in REVISIONS.values())
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        raise MetadataError("metadata API must not redirect")


def _require_hosted_metadata_environment() -> dict[str, str]:
    if sys.platform != "linux":
        raise MetadataError("metadata fetch requires a hosted Linux runner")
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise MetadataError("metadata fetch requires GITHUB_ACTIONS=true")
    github_sha = os.environ.get("GITHUB_SHA", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not HEX40.fullmatch(github_sha) or not run_id:
        raise MetadataError("metadata fetch requires GITHUB_SHA and GITHUB_RUN_ID")
    return {"github_sha": github_sha.lower(), "github_run_id": run_id}


def _fetch_json(
    url: str,
    *,
    opener: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], int, str]:
    if url not in ALLOWED_API_URLS:
        raise MetadataError("metadata URL is not one of the fixed reviewed API URLs")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise MetadataError("metadata URL is not the fixed HTTPS Hugging Face API")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    open_fn = opener or urllib.request.build_opener(_NoRedirectHandler()).open
    deadline = clock() + SOCKET_TIMEOUT
    try:
        response = open_fn(request, timeout=SOCKET_TIMEOUT)
        with response:
            if response.geturl() != url:
                raise MetadataError("metadata API response was redirected")
            content_type = response.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise MetadataError("metadata API response is not JSON")
            data = bytearray()
            while True:
                if clock() >= deadline:
                    raise MetadataError("metadata API read deadline exceeded")
                remaining = MAX_API_BYTES + 1 - len(data)
                chunk = response.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_API_BYTES:
                    raise MetadataError("metadata API response exceeds bounded size")
            raw = bytes(data)
    except MetadataError:
        raise
    except (OSError, urllib.error.URLError) as exc:
        raise MetadataError(f"metadata API request failed: {exc}") from exc
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, ValueError) as exc:
        raise MetadataError("metadata API response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise MetadataError("metadata API response must be an object")
    return payload, len(raw), hashlib.sha256(raw).hexdigest()


def _extract_pin(name: str, repo: str, revision: str, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload_sha = payload.get("sha")
    if payload.get("id") != repo or not isinstance(payload_sha, str) or payload_sha.lower() != revision:
        raise MetadataError(f"{name}: API identity/revision does not match reviewed pin")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list):
        raise MetadataError(f"{name}: API response has no sibling list")
    ggufs = [item for item in siblings if isinstance(item, dict) and isinstance(item.get("rfilename"), str) and item["rfilename"].endswith(".gguf")]
    if len(ggufs) != 1:
        raise MetadataError(f"{name}: expected exactly one GGUF sibling, found {len(ggufs)}")
    item = ggufs[0]
    filename = item["rfilename"]
    lfs = item.get("lfs")
    if not isinstance(lfs, dict) or not isinstance(lfs.get("sha256"), str) or not HEX64.fullmatch(lfs["sha256"]):
        raise MetadataError(f"{name}: GGUF sibling has no valid LFS SHA-256")
    size = lfs.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MetadataError(f"{name}: GGUF sibling has no positive LFS size")
    listed_size = item.get("size")
    if listed_size is not None and listed_size != size:
        raise MetadataError(f"{name}: sibling size disagrees with LFS size")
    pin = {
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "sha256": lfs["sha256"],
        "size_bytes": size,
    }
    identity = {
        "name": name,
        "repo": repo,
        "revision": revision,
        "filename": filename,
        "sha256": lfs["sha256"],
        "size_bytes": size,
    }
    return pin, identity


def collect_metadata(
    *,
    fetcher: Callable[[str], tuple[dict[str, Any], int, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if fetcher is None:
        _require_hosted_metadata_environment()
        fetch = _fetch_json
    else:
        fetch = fetcher
    pins: dict[str, dict[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    for name, (repo, revision) in REVISIONS.items():
        url = _api_url(repo, revision)
        payload, response_bytes, response_sha256 = fetch(url)
        pin, identity = _extract_pin(name, repo, revision, payload)
        identity.update({"api_url": url, "response_bytes": response_bytes, "response_sha256": response_sha256})
        pins[name] = pin
        identities.append(identity)
    smoke = _load_smoke_module()
    try:
        smoke.parse_pins(pins)
    except smoke.SmokeError as exc:
        raise MetadataError(f"metadata pins failed smoke manifest validation: {exc}") from exc
    return pins, identities


def write_outputs(
    pins_path: Path,
    receipt_path: Path,
    pins: dict[str, Any],
    identities: list[dict[str, Any]],
    source_identity: dict[str, str] | None = None,
) -> None:
    receipt = {"schema": 1, "status": "PASS", "models": identities}
    if source_identity is not None:
        receipt["source"] = source_identity
    pins_bytes = json.dumps(pins, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(pins_bytes) + len(receipt_bytes) > MAX_OUTPUT_BYTES:
        raise MetadataError("metadata outputs exceed compact receipt bound")
    pins_path.write_bytes(pins_bytes)
    receipt_path.write_bytes(receipt_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source_identity = _require_hosted_metadata_environment()
        pins, identities = collect_metadata()
        args.pins_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        write_outputs(args.pins_output, args.receipt, pins, identities, source_identity)
    except MetadataError as exc:
        print(f"FAIL release-candidate-metadata: {exc}", file=sys.stderr)
        return 1
    print("metadata identities valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
