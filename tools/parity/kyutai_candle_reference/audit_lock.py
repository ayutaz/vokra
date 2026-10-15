#!/usr/bin/env -S uv run --no-project --python 3.12
"""Bounded, source-only RustSec audit of the recovered Candle lock."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
import tomllib
from typing import Any, Callable

PRODUCER_RUN_ID = 38690241603
PRODUCER_ATTEMPT = 1
PRODUCER_HEAD_SHA = "b111f377bfbfc2f8d079f10791d61befb2d3ff5d"
ARTIFACT_ID = 11951567227
ARTIFACT_NAME = "kyutai-candle-source-metadata-38690241603-1"
ARTIFACT_SIZE = 210734
PRODUCTION_REPOSITORY = "ayutaz/vokra"
PATCHED_LOCK_SHA256 = "303c49433187d5938d0f2a27d3f5ded9eb094388dd67e6702a0a8b7558eaec01"
ADVISORY_DB_COMMIT = "edb58f5974b83eb8302332821cb6d389777aa351"
ADVISORY_DB_TREE = "9cadd9fc13d75527fa5fd3f14376ed3a1a5e3b63"
CARGO_AUDIT_VERSION = "0.22.2"
CARGO_AUDIT_RELEASE_COMMIT = "281452c35cf0870969042374110f099a411bc185"
INSTALL_ACTION_COMMIT = "183e4297cca2404691e9380e1307288dced5c82a"
DOWNLOAD_ACTION_COMMIT = "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_STDOUT_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
MAX_AUDITOR_BYTES = 64 * 1024 * 1024
DEFAULT_DEADLINE_SECONDS = 480.0
FULL_WARNING_CLASSES = {"unmaintained", "unsound", "notice"}
KNOWN_WARNING_CLASSES = FULL_WARNING_CLASSES | {"yanked"}

EXPECTED_MEMBERS: dict[str, tuple[int, str]] = {
    "source_receipt.json": (8320, "157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec"),
    "source/rust/Cargo.lock": (150274, "bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab"),
    "base/Cargo.lock": (39842, "eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de"),
    "patched/Cargo.lock": (39843, PATCHED_LOCK_SHA256),
    "initial_metadata.json": (898699, "904ba6da53c222ad8feda4c5a82d90dac12b9c84defa6ef081a613e7c9b4b736"),
    "whole_lock_metadata.json": (1264493, "61982123e29c2d51f3147d1b2a4755252da4b040ccae741a42c42d594a220e00"),
    "metadata.json": (324957, "99a22ac5c6ab9b34e5b21c7ceee308011ae42bf880d3c711403e9aeec1847a02"),
}


@dataclass(frozen=True)
class AuditProfile:
    """One of the two authenticated, non-customizable audit contracts."""

    name: str
    producer_run_id: int
    producer_attempt: int
    producer_head_sha: str
    artifact_id: int
    artifact_name: str
    artifact_size: int
    member_prefixes: tuple[str, ...]
    expected_members: tuple[tuple[str, int, str], ...]
    extracted_bytes: int
    lock_relative_path: str
    lock_sha256: str
    lock_dependency_count: int
    evidence_scope: str
    cpu_active_packages: int | None


CANDIDATE_CAPTURE_PREFIX = "vokra-candle-pin-capture-38711006458-1/"
CANDIDATE_RESULT_PREFIX = "vokra-candle-pin-result-38711006458-1/"
CANDIDATE_MEMBERS: tuple[tuple[str, int, str], ...] = (
    (CANDIDATE_CAPTURE_PREFIX + "source_receipt.json", 8320, "157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec"),
    (CANDIDATE_CAPTURE_PREFIX + "source/rust/Cargo.lock", 150274, "bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab"),
    (CANDIDATE_CAPTURE_PREFIX + "base/Cargo.lock", 39842, "eafbfd9c58c0b8aa60c4298f4df7e0187c78161a8aa898d0490a98c52faf28de"),
    (CANDIDATE_CAPTURE_PREFIX + "patched/Cargo.lock", 39843, PATCHED_LOCK_SHA256),
    (CANDIDATE_CAPTURE_PREFIX + "initial_metadata.json", 896267, "c4712fec75c50bf1b0c2813bc0456994c7e383e145d3a5ec41c42dec9095bbc5"),
    (CANDIDATE_CAPTURE_PREFIX + "whole_lock_metadata.json", 1261025, "3ed8961b442e32541af940ab0b84b00f93de7942c8de34b852f4b59af68301a6"),
    (CANDIDATE_CAPTURE_PREFIX + "metadata.json", 324861, "3a988e3219f9b25d128ec52c27798aaf93f07909e6ad9ef8269d9faee8319fff"),
    (CANDIDATE_RESULT_PREFIX + "candidate.lock", 38252, "d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8"),
    (CANDIDATE_RESULT_PREFIX + "candidate_cpu_metadata.json", 849118, "9c351398903613d266f41d4d9769f488780aba5c4592312cee54b732c634f043"),
    (CANDIDATE_RESULT_PREFIX + "candidate_whole_metadata.json", 1148255, "959357f105d4118301fc0be3bf94bc6f49944e4df5122cb6a8124bfc2b6c2793"),
    (CANDIDATE_RESULT_PREFIX + "pin_diagnostic.json", 228834, "5ee83d9cf3e2976b6350a7cd45346f27bfb40e4194616dcfb52e89c7f9385e44"),
    (CANDIDATE_RESULT_PREFIX + "baseline_cpu_metadata.json", 849497, "6964ac9b62708caa3cd794cf26055628e455a5768f9312476049c468ca3c7e0d"),
    (CANDIDATE_RESULT_PREFIX + "baseline_whole_metadata.json", 1205830, "cc3728fc52f80ea0554ca3cb9916de0bd45ebe7fa8d5379f541873ef0d7bed99"),
)

BASELINE_PROFILE = AuditProfile(
    "baseline", PRODUCER_RUN_ID, PRODUCER_ATTEMPT, PRODUCER_HEAD_SHA, ARTIFACT_ID,
    ARTIFACT_NAME, ARTIFACT_SIZE, (), (), MAX_ARTIFACT_BYTES, "patched/Cargo.lock",
    PATCHED_LOCK_SHA256, 161, "candle_cpu_patched_lock", 105,
)
CANDIDATE_PROFILE = AuditProfile(
    "candidate", 38711006458, 1, "27a2841384220c078aa37ff93ea8f9fc8a18594f",
    11961100426, "kyutai-candle-pin-remediation-38711006458-1", 457074,
    (CANDIDATE_CAPTURE_PREFIX, CANDIDATE_RESULT_PREFIX), CANDIDATE_MEMBERS,
    7040218, CANDIDATE_RESULT_PREFIX + "candidate.lock",
    "d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8", 154,
    "candle_candidate_whole_lock", None,
)
_PROFILES = {BASELINE_PROFILE.name: BASELINE_PROFILE, CANDIDATE_PROFILE.name: CANDIDATE_PROFILE}


class AuditError(RuntimeError):
    pass


def resolve_profile(profile: str | AuditProfile | None = None) -> AuditProfile:
    if profile is None:
        return BASELINE_PROFILE
    if isinstance(profile, str):
        try:
            return _PROFILES[profile]
        except KeyError as exc:
            raise AuditError("profile is not registered") from exc
    if profile is BASELINE_PROFILE or profile is CANDIDATE_PROFILE:
        return profile
    raise AuditError("custom audit profiles are not allowed")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool = False


def _now() -> float:
    return time.monotonic()


def _remaining(deadline: float) -> float:
    value = deadline - _now()
    if value <= 0:
        raise AuditError("shared deadline expired")
    return value


@contextmanager
def _absolute_alarm(deadline: float):
    """Interrupt blocking POSIX network I/O at the shared absolute deadline."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise AuditError("POSIX absolute deadline support is unavailable")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise AuditError("inherited SIGALRM timer is not allowed")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def interrupt(_signum, _frame):
        raise AuditError("shared deadline expired")

    remaining = _remaining(deadline)
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bounded(path: Path, cap: int) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(cap + 1)
    if len(data) > cap:
        raise AuditError(f"file exceeds bounded cap: {path}")
    return data


def _write_json(path: Path, value: Any) -> None:
    if not path.is_absolute():
        raise AuditError("evidence output must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise AuditError("evidence parent must be an existing regular directory")
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise AuditError("evidence JSON exceeds bounded cap")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise AuditError(f"output must be fresh: {path}") from exc
    try:
        offset = 0
        while offset < len(encoded):
            offset += os.write(fd, encoded[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_clean_config(cwd: Path, cargo_home: Path) -> None:
    if not cwd.is_dir() or cwd.is_symlink() or any(cwd.iterdir()):
        raise AuditError("audit cwd must be an existing empty regular directory")
    if not cargo_home.is_dir() or cargo_home.is_symlink() or any(cargo_home.iterdir()):
        raise AuditError("CARGO_HOME must be an existing empty regular directory")


def _host_guard(env: dict[str, str]) -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise AuditError("host must be Linux x86_64")
    if env.get("GITHUB_ACTIONS") != "true" or env.get("RUNNER_OS") != "Linux" or env.get("RUNNER_ARCH") != "X64":
        raise AuditError("host must be GitHub-hosted Linux X64")


def _child_env(cargo_home: Path, source: dict[str, str] | None = None) -> dict[str, str]:
    source = source or dict(os.environ)
    if not source.get("HOME"):
        raise AuditError("HOME must remain the runner-provided value")
    return {"HOME": source["HOME"], "PATH": source.get("PATH", ""), "CARGO_HOME": str(cargo_home), "CARGO_TERM_COLOR": "never", "LANG": "C", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def build_audit_argv(auditor: str, lockfile: Path, db: Path) -> tuple[str, ...]:
    return (auditor, "audit", "--file", str(lockfile), "--db", str(db), "--no-fetch", "--no-yanked", "--quiet", "--json", "--deny", "warnings")


def _kill_owned_group(process: subprocess.Popen[bytes], grace: float = 1.0) -> None:
    try:
        group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        group_id = process.pid
    group_exists = True
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        group_exists = False
    except PermissionError:
        # Permission denied is not proof that the owned group is gone.
        group_exists = True
    end = _now() + grace
    alive = True
    if group_exists:
        while _now() < end:
            try:
                os.killpg(group_id, 0)
            except ProcessLookupError:
                alive = False
                break
            except PermissionError:
                # Permission denied is not proof that the owned group is gone.
                alive = True
                break
            time.sleep(0.05)
    if alive:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Fall through to direct-child termination; do not treat this as
            # successful group cleanup or mask the original failure.
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


def run_command(argv: tuple[str, ...], *, cwd: Path, env: dict[str, str], deadline: float, stdout_cap: int, stderr_cap: int) -> CommandResult:
    _remaining(deadline)
    process = subprocess.Popen(list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    selector: selectors.BaseSelector | None = None
    streams: dict[int, tuple[Any, str]] = {}
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    caps = {"stdout": stdout_cap, "stderr": stderr_cap}
    timed_out = False
    cleanup_needed = False
    try:
        selector = selectors.DefaultSelector()
        assert process.stdout is not None and process.stderr is not None
        streams = {process.stdout.fileno(): (process.stdout, "stdout"), process.stderr.fileno(): (process.stderr, "stderr")}
        for stream, name in streams.values():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            timeout = min(0.25, _remaining(deadline))
            for key, _ in selector.select(timeout):
                fd = key.fileobj.fileno()
                try:
                    chunk = os.read(fd, 8192)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                room = caps[name] - len(captured[name])
                if room > 0:
                    captured[name].extend(chunk[:room])
                if len(chunk) > max(room, 0):
                    truncated[name] = True
        process.wait(timeout=min(0.5, _remaining(deadline)))
    except (AuditError, subprocess.TimeoutExpired):
        timed_out = True
        cleanup_needed = True
        _kill_owned_group(process)
    except BaseException:
        # Preserve unexpected I/O errors/interruptions, but never leave the
        # owned process group running while unwinding.
        cleanup_needed = True
        try:
            _kill_owned_group(process)
        except BaseException:
            pass
        raise
    finally:
        if cleanup_needed:
            try:
                _kill_owned_group(process)
            except BaseException:
                pass
        if selector is not None:
            selector.close()
        for stream, _ in streams.values():
            if not stream.closed:
                stream.close()
    return CommandResult(tuple(argv), 124 if timed_out else process.returncode, bytes(captured["stdout"]), bytes(captured["stderr"]), truncated["stdout"], truncated["stderr"], timed_out)


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise AuditError("GitHub API redirects are rejected")


def _api_json(url: str, token: str, deadline: float) -> dict[str, Any]:
    if not url.startswith("https://api.github.com/repos/") or not token:
        raise AuditError("unexpected GitHub API request")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"})
    opener = urllib.request.build_opener(_RejectRedirect())
    try:
        with _absolute_alarm(deadline):
            with opener.open(request, timeout=min(30.0, _remaining(deadline))) as response:
                socket_obj = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if socket_obj is not None and hasattr(socket_obj, "settimeout"):
                    socket_obj.settimeout(min(30.0, _remaining(deadline)))
                raw = bytearray()
                while True:
                    _remaining(deadline)
                    try:
                        chunk = response.read(min(8192, MAX_JSON_BYTES + 1 - len(raw)))
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    raw.extend(chunk)
                    if len(raw) > MAX_JSON_BYTES:
                        raise AuditError("GitHub API response exceeds bounded cap")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise AuditError(f"GitHub API request failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(bytes(raw))
    except json.JSONDecodeError as exc:
        raise AuditError("GitHub API response is not JSON") from exc
    if not isinstance(value, dict):
        raise AuditError("GitHub API response must be an object")
    return value


def validate_producer(*, repository: str, token: str, run_id: int | None = None, artifact_id: int | None = None, expected_head: str | None = None, api: Callable[[str, str, float], dict[str, Any]] = _api_json, deadline: float, profile: str | AuditProfile | None = None) -> dict[str, Any]:
    selected = resolve_profile(profile)
    expected_run = selected.producer_run_id
    expected_artifact = selected.artifact_id
    expected_sha = selected.producer_head_sha
    if run_id is not None and run_id != expected_run:
        raise AuditError("producer run does not match registered profile")
    if artifact_id is not None and artifact_id != expected_artifact:
        raise AuditError("producer artifact does not match registered profile")
    if expected_head is not None and expected_head != expected_sha:
        raise AuditError("producer head does not match registered profile")
    run_id, artifact_id, expected_head = expected_run, expected_artifact, expected_sha
    if repository != PRODUCTION_REPOSITORY or not token:
        raise AuditError("producer binding requires the exact production repository and token")
    base = f"https://api.github.com/repos/{repository}"
    run = api(f"{base}/actions/runs/{run_id}", token, deadline)
    if run.get("id") != run_id or run.get("status") != "completed" or run.get("conclusion") != "success" or run.get("head_sha") != expected_head or run.get("run_attempt") != selected.producer_attempt:
        raise AuditError("producer run/head/attempt mismatch")
    listing = api(f"{base}/actions/runs/{run_id}/artifacts?per_page=100", token, deadline)
    artifacts = listing.get("artifacts")
    if not isinstance(artifacts, list):
        raise AuditError("artifact API response lacks artifacts list")
    matches = [item for item in artifacts if isinstance(item, dict) and item.get("id") == artifact_id]
    if len(matches) != 1:
        raise AuditError("expected artifact ID is missing or duplicated")
    artifact = matches[0]
    if artifact.get("name") != selected.artifact_name or artifact.get("size_in_bytes") != selected.artifact_size or artifact.get("expired") is True:
        raise AuditError("artifact identity or size mismatch")
    expires_at = artifact.get("expires_at")
    if not expires_at:
        raise AuditError("artifact has no expiry")
    try:
        expiry = datetime.datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError("artifact expiry is not an ISO timestamp") from exc
    if expiry <= datetime.datetime.now(datetime.timezone.utc):
        raise AuditError("artifact is expired")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict) or workflow_run.get("id") != run_id or workflow_run.get("head_sha") != expected_head:
        raise AuditError("artifact workflow binding mismatch")
    return {"run_id": run_id, "run_attempt": run.get("run_attempt"), "head_sha": expected_head, "artifact_id": artifact_id, "artifact_name": artifact.get("name"), "artifact_size": artifact.get("size_in_bytes"), "expires_at": artifact.get("expires_at")}


def _walk_regular_files(root: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES, exact_bytes: int | None = None) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise AuditError("artifact root must be a regular directory")
    files: dict[str, Path] = {}
    total = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise AuditError(f"artifact symlink rejected: {relative}")
        if path.is_dir():
            continue
        if not path.is_file() or path.stat().st_nlink != 1:
            raise AuditError(f"artifact member is not a regular single-link file: {relative}")
        if relative in files:
            raise AuditError(f"duplicate artifact member: {relative}")
        size = path.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise AuditError(f"artifact member exceeds cap: {relative}")
        total += size
        if total > max_bytes:
            raise AuditError("artifact extracted bytes exceed cap")
        files[relative] = path
    if exact_bytes is not None and total != exact_bytes:
        raise AuditError("artifact extracted bytes do not match registered profile")
    return files


def _validate_candidate_metadata(files: dict[str, Path]) -> None:
    def read_json(relative: str) -> dict[str, Any]:
        try:
            value = json.loads(_read_bounded(files[relative], MAX_MEMBER_BYTES))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditError(f"candidate evidence is not valid JSON: {relative}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"candidate evidence is not an object: {relative}")
        return value

    lock_relative = CANDIDATE_RESULT_PREFIX + "candidate.lock"
    try:
        lock = tomllib.loads(_read_bounded(files[lock_relative], MAX_MEMBER_BYTES).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError("candidate lock is not valid TOML") from exc
    lock_packages = lock.get("package")
    if not isinstance(lock_packages, list) or len(lock_packages) != CANDIDATE_PROFILE.lock_dependency_count:
        raise AuditError("candidate lock package count is not 154")
    identities = []
    for package in lock_packages:
        if not isinstance(package, dict) or not all(isinstance(package.get(key), str) for key in ("name", "version")):
            raise AuditError("candidate lock package identity is malformed")
        identities.append((package["name"], package["version"], package.get("source")))
    if len(set(identities)) != len(identities):
        raise AuditError("candidate lock contains duplicate package identities")

    whole = read_json(CANDIDATE_RESULT_PREFIX + "candidate_whole_metadata.json")
    if whole.get("status") != "PIN_REMEDIATION_WHOLE_METADATA_UNREVIEWED" or whole.get("approval") is not None or whole.get("platform_filter") is not None or whole.get("target") is not None or whole.get("gpu_execution") != "NOT_AUTHORIZED" or whole.get("model_execution") != "NOT_AUTHORIZED" or whole.get("upload") != "NO_UPLOAD":
        raise AuditError("candidate whole metadata is not an unfiltered review-pending packet")
    if whole.get("source_commit") != "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362":
        raise AuditError("candidate whole metadata source commit mismatch")
    metadata = whole.get("metadata")
    packages = metadata.get("packages") if isinstance(metadata, dict) else None
    resolve = metadata.get("resolve") if isinstance(metadata, dict) else None
    nodes = resolve.get("nodes") if isinstance(resolve, dict) else None
    if not isinstance(metadata, dict) or not isinstance(packages, list) or not isinstance(nodes, list) or len(packages) != 154 or len(nodes) != 154:
        raise AuditError("candidate whole metadata does not contain the complete 154-node graph")
    members = metadata.get("workspace_members")
    if not isinstance(members, list) or len(members) != 1:
        raise AuditError("candidate whole metadata workspace root is not unique")
    package_ids = {item.get("id") for item in packages if isinstance(item, dict)}
    node_ids = {item.get("id") for item in nodes if isinstance(item, dict)}
    if len(package_ids) != 154 or package_ids != node_ids or not all(isinstance(item, str) for item in package_ids):
        raise AuditError("candidate whole metadata package/node identities disagree")
    metadata_identities = set()
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str) or package.get("source") is not None and not isinstance(package.get("source"), str):
            raise AuditError("candidate whole metadata package identity is malformed")
        metadata_identities.add((package["name"], package["version"], package.get("source")))
    lock_identities = {(package["name"], package["version"], package.get("source")) for package in lock_packages}
    if len(metadata_identities) != 154 or metadata_identities != lock_identities:
        raise AuditError("candidate whole metadata package identities disagree with the candidate lock")
    workspace_package = [item for item in packages if isinstance(item, dict) and item.get("id") in members]
    if len(workspace_package) != 1 or workspace_package[0].get("name") != "moshi":
        raise AuditError("candidate whole metadata workspace member is not Moshi")
    root = resolve.get("root")
    if root is not None:
        root_packages = [item for item in packages if isinstance(item, dict) and item.get("id") == root and item.get("name") == "moshi"]
        if len(root_packages) != 1:
            raise AuditError("candidate whole metadata root is not the unique Moshi package")
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("deps"), list):
            raise AuditError("candidate whole metadata node dependencies are malformed")
        for dependency in node["deps"]:
            if not isinstance(dependency, dict) or dependency.get("pkg") not in node_ids or not isinstance(dependency.get("dep_kinds"), list) or not dependency["dep_kinds"]:
                raise AuditError("candidate whole metadata dependency references an unknown node")
            for dep_kind in dependency["dep_kinds"]:
                if not isinstance(dep_kind, dict) or dep_kind.get("kind") not in {None, "normal", "build", "dev"} or dep_kind.get("target") is not None and not isinstance(dep_kind.get("target"), str):
                    raise AuditError("candidate whole metadata dependency kind is malformed")
    cpu = read_json(CANDIDATE_RESULT_PREFIX + "candidate_cpu_metadata.json")
    if cpu.get("status") != "PIN_REMEDIATION_CPU_METADATA_UNREVIEWED" or cpu.get("target") != "x86_64-unknown-linux-gnu" or cpu.get("platform_filter") != cpu.get("target") or cpu.get("approval") is not None or cpu.get("upload") != "NO_UPLOAD" or cpu.get("model_execution") != "NOT_AUTHORIZED" or cpu.get("gpu_execution") != "NOT_AUTHORIZED":
        raise AuditError("candidate CPU metadata provenance mismatch")
    diagnostic = read_json(CANDIDATE_RESULT_PREFIX + "pin_diagnostic.json")
    if diagnostic.get("status") != "PIN_REMEDIATION_SOURCE_ONLY_REVIEW_PENDING" or diagnostic.get("approval") is not None or diagnostic.get("upload") != "NO_UPLOAD" or diagnostic.get("model_execution") != "NOT_AUTHORIZED":
        raise AuditError("candidate diagnostic is not review-pending source-only evidence")


def validate_artifact(root: Path, *, profile: str | AuditProfile | None = None) -> dict[str, Any]:
    selected = resolve_profile(profile)
    if selected is BASELINE_PROFILE:
        expected = EXPECTED_MEMBERS
        files = _walk_regular_files(root)
    else:
        expected = {relative: (size, digest) for relative, size, digest in selected.expected_members}
        files = _walk_regular_files(root, max_bytes=selected.extracted_bytes, exact_bytes=selected.extracted_bytes)
        if not all(any(relative.startswith(prefix) for prefix in selected.member_prefixes) for relative in files):
            raise AuditError("candidate artifact contains an unregistered member prefix")
    if set(files) != set(expected):
        raise AuditError("artifact member set mismatch")
    receipts: dict[str, Any] = {}
    for relative, (expected_size, expected_hash) in expected.items():
        data = _read_bounded(files[relative], MAX_MEMBER_BYTES)
        digest = _sha256(data)
        if len(data) != expected_size or digest != expected_hash:
            raise AuditError(f"artifact member hash/size mismatch: {relative}")
        receipts[relative] = {"bytes": len(data), "sha256": digest}
    if selected is CANDIDATE_PROFILE:
        _validate_candidate_metadata(files)
    return {"members": receipts, "total_bytes": sum(item["bytes"] for item in receipts.values())}


def git_provenance(db: Path, *, cwd: Path, env: dict[str, str], deadline: float, runner: Callable[..., CommandResult] = run_command) -> dict[str, Any]:
    if not db.is_dir() or db.is_symlink():
        raise AuditError("advisory DB must be a regular directory")
    values: dict[str, str] = {}
    for key, argv in {"commit": ("git", "-C", str(db), "rev-parse", "--verify", "HEAD"), "tree": ("git", "-C", str(db), "rev-parse", "--verify", "HEAD^{tree}"), "status": ("git", "-C", str(db), "status", "--porcelain", "--untracked-files=all")}.items():
        result = runner(argv, cwd=cwd, env=env, deadline=deadline, stdout_cap=65536, stderr_cap=65536)
        if result.returncode != 0 or result.timed_out or result.stdout_truncated or result.stderr_truncated:
            raise AuditError(f"advisory DB git provenance command failed: {key}")
        values[key] = result.stdout.decode("utf-8", "strict").strip()
    if values["commit"] != ADVISORY_DB_COMMIT or values["tree"] != ADVISORY_DB_TREE or values["status"]:
        raise AuditError("advisory DB external Git provenance mismatch or dirty tree")
    return {"commit": values["commit"], "tree": values["tree"], "tracked_untracked_status": "clean"}


def verify_auditor(auditor: str, *, cwd: Path, env: dict[str, str], deadline: float, runner: Callable[..., CommandResult] = run_command) -> dict[str, Any]:
    resolved = shutil.which(auditor, path=env.get("PATH"))
    if not resolved:
        raise AuditError("cargo-audit binary is not on PATH")
    binary = Path(resolved)
    if not binary.is_file() or binary.is_symlink():
        raise AuditError("cargo-audit path is not a regular file")
    version = runner((resolved, "audit", "--version"), cwd=cwd, env=env, deadline=deadline, stdout_cap=4096, stderr_cap=4096)
    def mismatch() -> AuditError:
        error = AuditError("cargo-audit version mismatch")
        setattr(error, "version_evidence", _command_evidence(version))
        return error
    if version.returncode != 0 or version.timed_out or version.stdout_truncated or version.stderr_truncated:
        raise mismatch()
    try:
        text = (version.stdout + version.stderr).decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise mismatch()
    expected_line = f"cargo-audit-audit {CARGO_AUDIT_VERSION}"
    if text.strip() != expected_line:
        raise mismatch()
    digest = _sha256(_read_bounded(binary, MAX_AUDITOR_BYTES))
    return {"cargo_audit_version": CARGO_AUDIT_VERSION, "version_string": text.strip(), "version_output": text, "binary_path": str(binary), "binary_sha256": digest, "release_commit": CARGO_AUDIT_RELEASE_COMMIT, "install_action_commit": INSTALL_ACTION_COMMIT}


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AuditError(f"{name} must be an integer >= {minimum}")
    return value


def validate_report(report: Any, *, profile: str | AuditProfile | None = None) -> dict[str, Any]:
    selected = resolve_profile(profile)
    if not isinstance(report, dict):
        raise AuditError("audit JSON must be an object")
    for field in ("database", "lockfile", "settings", "vulnerabilities", "warnings"):
        if field not in report:
            raise AuditError(f"audit JSON missing {field}")
    database, lockfile, settings = report["database"], report["lockfile"], report["settings"]
    if not isinstance(database, dict) or "last-commit" not in database or database["last-commit"] is not None:
        raise AuditError("pinned Database::open report must have explicit null database.last-commit")
    advisory_count = _exact_int(database.get("advisory-count"), "advisory count", minimum=1)
    if not isinstance(lockfile, dict) or _exact_int(lockfile.get("dependency-count"), "lock dependency count") != selected.lock_dependency_count:
        raise AuditError(f"whole-lock dependency count is not {selected.lock_dependency_count}")
    if not isinstance(settings, dict) or settings.get("ignore") != [] or settings.get("target_arch") != [] or settings.get("target_os") != [] or settings.get("severity") is not None:
        raise AuditError("audit report contains ignore, target, or severity filters")
    if set(settings.get("informational_warnings", [])) != FULL_WARNING_CLASSES:
        raise AuditError("audit report does not include all informational warning classes")
    vulnerabilities = report["vulnerabilities"]
    if not isinstance(vulnerabilities, dict) or type(vulnerabilities.get("found")) is not bool:
        raise AuditError("vulnerability found flag is missing or not boolean")
    vulnerability_count = _exact_int(vulnerabilities.get("count"), "vulnerability count")
    vulnerability_list = vulnerabilities.get("list")
    if not isinstance(vulnerability_list, list) or len(vulnerability_list) != vulnerability_count or vulnerabilities["found"] != (vulnerability_count > 0):
        raise AuditError("vulnerability counters disagree")
    warnings = report["warnings"]
    if not isinstance(warnings, dict) or set(warnings) - KNOWN_WARNING_CLASSES:
        raise AuditError("warning classes are not recognized")
    warning_counts: dict[str, int] = {}
    warning_entries = 0
    for key, value in warnings.items():
        if not isinstance(value, list):
            raise AuditError("warning values must be lists")
        warning_counts[key] = len(value)
        warning_entries += len(value)
    return {"database_advisory_count": advisory_count, "database_last_commit": None, "database_last_commit_fact": "PINNED_DATABASE_OPEN_DOES_NOT_POPULATE_COMMIT", "lock_dependency_count": selected.lock_dependency_count, "vulnerability_count": vulnerability_count, "vulnerability_found": vulnerabilities["found"], "warning_counts": warning_counts, "warning_entries": warning_entries}


def _command_evidence(result: CommandResult) -> dict[str, Any]:
    return {"argv": list(result.argv), "returncode": result.returncode, "stdout": result.stdout.decode("utf-8", "replace"), "stdout_bytes": len(result.stdout), "stdout_sha256": _sha256(result.stdout), "stdout_truncated": result.stdout_truncated, "stderr": result.stderr.decode("utf-8", "replace"), "stderr_bytes": len(result.stderr), "stderr_sha256": _sha256(result.stderr), "stderr_truncated": result.stderr_truncated, "timed_out": result.timed_out}


def run_audit(*, artifact_root: Path, db: Path, cwd: Path, cargo_home: Path, auditor: str, repository: str, output: Path, deadline_seconds: float = DEFAULT_DEADLINE_SECONDS, api: Callable[[str, str, float], dict[str, Any]] = _api_json, command_runner: Callable[..., CommandResult] = run_command, verifier: Callable[..., dict[str, Any]] = verify_auditor, producer_validator: Callable[..., dict[str, Any]] = validate_producer, artifact_validator: Callable[..., dict[str, Any]] = validate_artifact, provenance: Callable[..., dict[str, Any]] = git_provenance, host_guard: Callable[[dict[str, str]], None] = _host_guard, profile: str | AuditProfile | None = None) -> int:
    if type(deadline_seconds) not in (int, float) or not math.isfinite(float(deadline_seconds)) or deadline_seconds <= 0 or deadline_seconds > DEFAULT_DEADLINE_SECONDS:
        raise AuditError("deadline must be finite, positive, and no greater than 480 seconds")
    selected = resolve_profile(profile)
    deadline = _now() + deadline_seconds
    lock_hash_field = "patched_lock_sha256" if selected is BASELINE_PROFILE else "candidate_lock_sha256"
    evidence: dict[str, Any] = {"scope": selected.evidence_scope, "status": "ERROR", "advisory_audit": "ERROR", "approval": None, "license_review": "NOT_PERFORMED", "root_runtime_audit": "NOT_RUN", "model_execution": "NOT_AUTHORIZED", "upload": "NO_UPLOAD", "lock_records": selected.lock_dependency_count, "cpu_active_packages": selected.cpu_active_packages, lock_hash_field: selected.lock_sha256, "tool": {"expected_cargo_audit_version": CARGO_AUDIT_VERSION, "install_action_commit": INSTALL_ACTION_COMMIT, "release_commit": CARGO_AUDIT_RELEASE_COMMIT}, "advisory_database": {"commit": ADVISORY_DB_COMMIT, "tree": ADVISORY_DB_TREE, "no_fetch": True, "report_last_commit_fact": "PINNED_DATABASE_OPEN_DOES_NOT_POPULATE_COMMIT"}, "yanked_check": "NOT_CHECKED", "producer": None, "artifact": None, "audit": None}
    try:
        source_env = dict(os.environ)
        host_guard(source_env)
        _assert_clean_config(cwd, cargo_home)
        env = _child_env(cargo_home, source_env)
        token = source_env.get("GITHUB_TOKEN", "")
        evidence["tool"] = verifier(auditor, cwd=cwd, env=env, deadline=deadline, runner=command_runner)
        verified_auditor = evidence["tool"].get("binary_path")
        if not isinstance(verified_auditor, str) or not verified_auditor:
            raise AuditError("auditor verifier did not return a resolved binary path")
        if selected is BASELINE_PROFILE:
            evidence["producer"] = producer_validator(repository=repository, token=token, api=api, deadline=deadline)
            evidence["artifact"] = artifact_validator(artifact_root)
        else:
            evidence["producer"] = producer_validator(repository=repository, token=token, api=api, deadline=deadline, profile=selected)
            evidence["artifact"] = artifact_validator(artifact_root, profile=selected)
        before = provenance(db, cwd=cwd, env=env, deadline=deadline, runner=command_runner)
        lockfile = artifact_root / selected.lock_relative_path
        result = command_runner(build_audit_argv(verified_auditor, lockfile, db), cwd=cwd, env=env, deadline=deadline, stdout_cap=MAX_STDOUT_BYTES, stderr_cap=MAX_STDERR_BYTES)
        audit: dict[str, Any] = {"command": _command_evidence(result), "git_before": before}
        evidence["audit"] = audit
        if result.timed_out:
            audit["classification"] = "ERROR_TIMEOUT"
            _write_json(output, evidence)
            return 2
        if result.returncode not in (0, 1, 2):
            audit["classification"] = "ERROR_UNEXPECTED_RETURNCODE"
            _write_json(output, evidence)
            return 2
        if result.stdout_truncated or result.stderr_truncated:
            audit["classification"] = "ERROR_OUTPUT_LIMIT"
            _write_json(output, evidence)
            return 2
        try:
            parsed = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            audit.update({"classification": "ERROR_BAD_JSON", "parse_error": type(exc).__name__})
            _write_json(output, evidence)
            return 2
        audit["report"] = parsed
        try:
            facts = validate_report(parsed, profile=selected)
        except AuditError as exc:
            audit.update({"classification": "ERROR_REPORT_MISMATCH", "error": {"type": type(exc).__name__, "message": str(exc)}})
            _write_json(output, evidence)
            return 2
        after = provenance(db, cwd=cwd, env=env, deadline=deadline, runner=command_runner)
        if after != before:
            audit.update({"classification": "ERROR_DATABASE_CHANGED", "error": {"type": "AuditError", "message": "advisory DB changed during audit"}})
            _write_json(output, evidence)
            return 2
        audit["git_after"], audit["report_facts"] = after, facts
        if result.returncode == 0 and (facts["vulnerability_count"] or facts["warning_entries"]):
            classification, status, rc = "ERROR_RC0_FINDINGS", "ERROR", 2
        elif result.returncode == 0:
            classification, status, rc = "PASS", "PASS", 0
        elif result.returncode == 1:
            classification, status, rc = ("FAIL_FINDINGS", "FAIL", 1) if (facts["vulnerability_count"] or facts["warning_entries"]) else ("FAIL_NO_FINDINGS_RETURN1", "FAIL", 1)
        else:
            classification, status, rc = "ERROR_TOOL_RETURN", "ERROR", 2
        audit["classification"] = classification
        evidence["status"] = status
        evidence["advisory_audit"] = status
        _write_json(output, evidence)
        return rc
    except Exception as exc:
        version_evidence = getattr(exc, "version_evidence", None)
        if isinstance(version_evidence, dict):
            evidence["tool"]["version_check"] = version_evidence
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            _write_json(output, evidence)
        except Exception as write_exc:
            print(f"evidence write failed: {type(write_exc).__name__}: {write_exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--cargo-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--auditor", default="cargo-audit")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--profile", choices=tuple(_PROFILES), default=BASELINE_PROFILE.name)
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or args.deadline_seconds > DEFAULT_DEADLINE_SECONDS:
        parser.error("deadline must be positive and no greater than 480 seconds")
    return run_audit(artifact_root=args.artifact_root, db=args.db, cwd=args.cwd, cargo_home=args.cargo_home, auditor=args.auditor, repository=args.repository, output=args.output, deadline_seconds=args.deadline_seconds, profile=args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
