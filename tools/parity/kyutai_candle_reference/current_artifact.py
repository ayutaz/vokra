#!/usr/bin/env -S uv run --no-project --python 3.12
"""Authenticate the current-run Candle application evidence envelope.

The primary application artifact intentionally remains the exact eight files
used by the historical profiles.  This helper emits a separate, bounded
envelope containing the producer's member map and the API facts captured after
upload.  Consumers still re-check every fact through the GitHub API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import urllib.error
import urllib.request
from typing import Any

MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_ENVELOPE_BYTES = 64 * 1024
CURRENT_SCHEMA = 1
PRODUCTION_REPOSITORY = "ayutaz/vokra"
PRODUCER_WORKFLOW_PATH = ".github/workflows/kyutai-candle-paste-application.yml"
PRODUCER_WORKFLOW_NAME = "Kyutai Candle paste overlay source-only application"
PRODUCER_JOB_NAME = "source-only paste overlay metadata application (not build/parity)"
SOURCE_COMMIT = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
FINAL_LOCK_SHA256 = "21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6"
MEMBER_NAMES = (
    "application_receipt.json",
    "baseline_cpu_metadata.json",
    "baseline_whole_metadata.json",
    "candidate.lock",
    "final.lock",
    "overlay_cpu_metadata.json",
    "overlay_whole_metadata.json",
    "source_receipt.json",
)
CALLERS = {
    "applied-advisory": "kyutai-candle-applied-advisory",
    "cpu-compatibility": "Kyutai Candle CPU compatibility (one-shot)",
    "standalone": PRODUCER_WORKFLOW_NAME,
}
ENVELOPE_ARTIFACT_PREFIX = "kyutai-candle-paste-current-envelope-"
CALLER_PATHS = {
    "applied-advisory": ".github/workflows/kyutai-candle-applied-advisory.yml",
    "cpu-compatibility": ".github/workflows/kyutai-candle-cpu-compatibility.yml",
    "standalone": PRODUCER_WORKFLOW_PATH,
}
PRODUCER_JOB_API_NAMES = {
    "applied-advisory": (
        "current source-only Candle producer / source-only paste overlay metadata application (not build/parity)",
    ),
    "cpu-compatibility": (
        "current source-only Candle producer / source-only paste overlay metadata application (not build/parity)",
    ),
    "standalone": (PRODUCER_JOB_NAME,),
}


def referenced_workflow_paths(workflow_sha: str) -> tuple[str, ...]:
    """Exact API path spellings accepted for a local reusable workflow."""
    return (f"{PRODUCTION_REPOSITORY}/{PRODUCER_WORKFLOW_PATH}@{workflow_sha}",)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REF = re.compile(r"^refs/(?:pull/[1-9][0-9]*/merge|heads/[A-Za-z0-9._/-]+)$")


class CurrentArtifactError(RuntimeError):
    pass


def _reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CurrentArtifactError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, limit: int = MAX_MEMBER_BYTES) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise CurrentArtifactError(f"missing or non-regular member: {path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CurrentArtifactError(f"missing or non-regular member: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise CurrentArtifactError(f"member exceeds bound: {path.name}")
    return data


def normalize_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise CurrentArtifactError("artifact digest must be a string")
    digest = value.removeprefix("sha256:")
    if not HEX64.fullmatch(digest.lower()) or digest != digest.lower():
        raise CurrentArtifactError("artifact digest must be lowercase SHA-256")
    return digest


def _exact_int(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise CurrentArtifactError(f"{name} is outside its integer bound")
    return value


def _head(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value) or value != value.lower():
        raise CurrentArtifactError(f"{name} must be a lowercase 40-hex SHA")
    return value


def member_map(root: Path) -> tuple[dict[str, Any], str]:
    try:
        root_metadata = root.lstat()
    except FileNotFoundError as exc:
        raise CurrentArtifactError("application artifact root is not a directory") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise CurrentArtifactError("application artifact root is not a directory")
    actual = sorted(path.name for path in root.iterdir())
    if actual != sorted(MEMBER_NAMES):
        raise CurrentArtifactError(f"application member set mismatch: {actual}")
    rows = []
    total = 0
    for name in MEMBER_NAMES:
        data = _read(root / name)
        total += len(data)
        rows.append({"path": name, "bytes": len(data), "sha256": _sha256(data)})
    if total > 16 * 1024 * 1024:
        raise CurrentArtifactError("application artifact exceeds aggregate bound")
    receipt_hash = next(row["sha256"] for row in rows if row["path"] == "application_receipt.json")
    return {"members": rows, "application_receipt_sha256": receipt_hash}, receipt_hash


def validate_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value.get("schema") != CURRENT_SCHEMA:
        raise CurrentArtifactError("current evidence envelope schema mismatch")
    expected_keys = {
        "schema", "repository", "caller", "workflow_name", "caller_workflow_path", "producer_workflow_path",
        "producer_workflow_name", "producer_job_name", "run_id", "run_attempt",
        "source_head", "workflow_sha", "run_ref", "artifact_id", "artifact_name", "artifact_digest",
        "artifact_size", "source_commit", "final_lock_sha256", "members",
        "application_receipt_sha256",
    }
    if set(value) != expected_keys:
        raise CurrentArtifactError("current evidence envelope fields changed")
    repository = value.get("repository")
    if repository != PRODUCTION_REPOSITORY:
        raise CurrentArtifactError("current evidence repository mismatch")
    caller = value.get("caller")
    if not isinstance(caller, str) or caller not in CALLERS:
        raise CurrentArtifactError("current evidence caller is not registered")
    if value.get("workflow_name") != CALLERS[caller]:
        raise CurrentArtifactError("current evidence caller workflow mismatch")
    if value.get("caller_workflow_path") != CALLER_PATHS[caller]:
        raise CurrentArtifactError("current evidence caller workflow path mismatch")
    if value.get("producer_workflow_path") != PRODUCER_WORKFLOW_PATH or value.get("producer_workflow_name") != PRODUCER_WORKFLOW_NAME or value.get("producer_job_name") != PRODUCER_JOB_NAME:
        raise CurrentArtifactError("current producer workflow/job identity changed")
    _head(value.get("source_head"), "source_head")
    _head(value.get("workflow_sha"), "workflow_sha")
    if not isinstance(value.get("run_ref"), str) or not REF.fullmatch(value["run_ref"]):
        raise CurrentArtifactError("run_ref is not a supported GitHub ref")
    _exact_int(value.get("run_id"), "run_id", 1)
    _exact_int(value.get("run_attempt"), "run_attempt", 1)
    _exact_int(value.get("artifact_id"), "artifact_id", 1)
    _exact_int(value.get("artifact_size"), "artifact_size", 1, MAX_ARTIFACT_BYTES)
    artifact_name = value.get("artifact_name")
    if not isinstance(artifact_name, str) or len(artifact_name) > 256 or artifact_name != f"kyutai-candle-paste-application-evidence-{value['run_id']}-{value['run_attempt']}":
        raise CurrentArtifactError("current evidence artifact name is malformed")
    normalize_digest(value.get("artifact_digest"))
    members = value.get("members")
    if not isinstance(members, list) or len(members) != len(MEMBER_NAMES):
        raise CurrentArtifactError("current evidence member map is incomplete")
    paths = [row.get("path") for row in members if isinstance(row, dict)]
    if paths != list(MEMBER_NAMES):
        raise CurrentArtifactError("current evidence member order or names changed")
    total = 0
    for row in members:
        if not isinstance(row, dict):
            raise CurrentArtifactError("current evidence member row is malformed")
        if set(row) != {"path", "bytes", "sha256"}:
            raise CurrentArtifactError("current evidence member fields changed")
        size = _exact_int(row.get("bytes"), "member bytes", 0, MAX_MEMBER_BYTES)
        digest = row.get("sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest) or digest != digest.lower():
            raise CurrentArtifactError("current evidence member digest is malformed")
        total += size
    if total > 16 * 1024 * 1024:
        raise CurrentArtifactError("current evidence members exceed aggregate bound")
    receipt_hash = value.get("application_receipt_sha256")
    if receipt_hash != next(row["sha256"] for row in members if row["path"] == "application_receipt.json"):
        raise CurrentArtifactError("application receipt self-hash mismatch")
    if value.get("source_commit") != SOURCE_COMMIT or value.get("final_lock_sha256") != FINAL_LOCK_SHA256:
        raise CurrentArtifactError("fixed source or final lock identity changed")
    return value


def build_envelope(root: Path, *, repository: str, run_id: int, run_attempt: int, source_head: str, workflow_sha: str, run_ref: str, caller: str, artifact_id: int, artifact_name: str, artifact_digest: str, artifact_size: int) -> dict[str, Any]:
    if not isinstance(caller, str) or caller not in CALLERS:
        raise CurrentArtifactError("current evidence caller is not registered")
    members, receipt_hash = member_map(root)
    envelope = {
        "schema": CURRENT_SCHEMA,
        "repository": repository,
        "caller": caller,
        "workflow_name": CALLERS.get(caller),
        "caller_workflow_path": CALLER_PATHS.get(caller),
        "producer_workflow_path": PRODUCER_WORKFLOW_PATH,
        "producer_workflow_name": PRODUCER_WORKFLOW_NAME,
        "producer_job_name": PRODUCER_JOB_NAME,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "source_head": source_head,
        "workflow_sha": workflow_sha,
        "run_ref": run_ref,
        "artifact_id": artifact_id,
        "artifact_name": artifact_name,
        "artifact_digest": normalize_digest(artifact_digest),
        "artifact_size": artifact_size,
        "source_commit": SOURCE_COMMIT,
        "final_lock_sha256": FINAL_LOCK_SHA256,
        **members,
    }
    validate_envelope(envelope)
    if receipt_hash != envelope["application_receipt_sha256"]:
        raise CurrentArtifactError("receipt hash changed during envelope construction")
    return envelope


def _validate_capture_inputs(*, repository: str, token: str, run_id: int, run_attempt: int, source_head: str, workflow_sha: str, run_ref: str, caller: str, artifact_id: int, artifact_name: str, upload_digest: str) -> None:
    if repository != PRODUCTION_REPOSITORY or not isinstance(token, str) or not token:
        raise CurrentArtifactError("current producer capture requires the exact production repository and token")
    if not isinstance(caller, str) or caller not in CALLERS:
        raise CurrentArtifactError("current producer capture caller is not registered")
    _head(source_head, "source_head")
    _head(workflow_sha, "workflow_sha")
    if not isinstance(run_ref, str) or not REF.fullmatch(run_ref):
        raise CurrentArtifactError("run_ref is not a supported GitHub ref")
    _exact_int(run_id, "run_id", 1)
    _exact_int(run_attempt, "run_attempt", 1)
    _exact_int(artifact_id, "artifact_id", 1)
    expected_name = f"kyutai-candle-paste-application-evidence-{run_id}-{run_attempt}"
    if artifact_name != expected_name:
        raise CurrentArtifactError("current producer capture artifact name is malformed")
    normalize_digest(upload_digest)


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise CurrentArtifactError("GitHub API redirects are rejected")


def _api_json(url: str, token: str) -> dict[str, Any]:
    if not url.startswith("https://api.github.com/repos/") or not token:
        raise CurrentArtifactError("unexpected GitHub API request")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urllib.request.build_opener(_RejectRedirect()).open(request, timeout=30) as response:
            raw = response.read(MAX_ENVELOPE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise CurrentArtifactError(f"GitHub API request failed: {type(exc).__name__}") from exc
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise CurrentArtifactError("GitHub API response exceeds bounded cap")
    value = json.loads(raw, object_pairs_hook=_reject_duplicate_json)
    if not isinstance(value, dict):
        raise CurrentArtifactError("GitHub API response must be an object")
    return value


def capture_primary_artifact(*, root: Path, repository: str, token: str, run_id: int, run_attempt: int, source_head: str, workflow_sha: str, run_ref: str, caller: str, artifact_id: int, artifact_name: str, upload_digest: str) -> dict[str, Any]:
    _validate_capture_inputs(repository=repository, token=token, run_id=run_id, run_attempt=run_attempt, source_head=source_head, workflow_sha=workflow_sha, run_ref=run_ref, caller=caller, artifact_id=artifact_id, artifact_name=artifact_name, upload_digest=upload_digest)
    base = f"https://api.github.com/repos/{repository}"
    listing = _api_json(f"{base}/actions/runs/{run_id}/artifacts?per_page=100", token)
    artifacts = listing.get("artifacts")
    total = listing.get("total_count")
    if not isinstance(artifacts, list) or type(total) is not int or total != len(artifacts) or total > 100:
        raise CurrentArtifactError("artifact API response is incomplete or malformed")
    if any(not isinstance(item, dict) for item in artifacts):
        raise CurrentArtifactError("artifact API contains a malformed row")
    ids = [item.get("id") for item in artifacts]
    names = [item.get("name") for item in artifacts]
    if any(type(value) is not int or value <= 0 for value in ids) or any(not isinstance(value, str) or not value for value in names) or len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise CurrentArtifactError("artifact API contains duplicate or malformed artifact identities")
    matches = [item for item in artifacts if isinstance(item, dict) and item.get("id") == artifact_id]
    if len(matches) != 1:
        raise CurrentArtifactError("uploaded artifact is missing or duplicated")
    artifact = matches[0]
    if artifact.get("name") != artifact_name or artifact.get("expired") is not False:
        raise CurrentArtifactError("uploaded artifact identity or expiry changed")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict) or workflow_run.get("id") != run_id or workflow_run.get("head_sha") != source_head or "run_attempt" in workflow_run and workflow_run.get("run_attempt") != run_attempt:
        raise CurrentArtifactError("uploaded artifact workflow binding mismatch")
    api_digest = normalize_digest(artifact.get("digest"))
    if api_digest != normalize_digest(upload_digest):
        raise CurrentArtifactError("upload output digest differs from API digest")
    size = _exact_int(artifact.get("size_in_bytes"), "artifact size", 1, MAX_ARTIFACT_BYTES)
    return build_envelope(root, repository=repository, run_id=run_id, run_attempt=run_attempt, source_head=source_head, workflow_sha=workflow_sha, run_ref=run_ref, caller=caller, artifact_id=artifact_id, artifact_name=artifact_name, artifact_digest=api_digest, artifact_size=size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--run-ref", required=True)
    parser.add_argument("--caller", choices=tuple(CALLERS), required=True)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--upload-digest", required=True)
    args = parser.parse_args(argv)
    envelope = capture_primary_artifact(root=args.artifact_root, repository=args.repository, token=os.environ.get("GITHUB_TOKEN", ""), run_id=args.run_id, run_attempt=args.run_attempt, source_head=args.source_head, workflow_sha=args.workflow_sha, run_ref=args.run_ref, caller=args.caller, artifact_id=args.artifact_id, artifact_name=args.artifact_name, upload_digest=args.upload_digest)
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("envelope output must be fresh")
    encoded = (json.dumps(envelope, sort_keys=True, indent=2) + "\n").encode()
    if len(encoded) > MAX_ENVELOPE_BYTES:
        raise SystemExit("envelope exceeds bound")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
