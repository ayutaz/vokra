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

import current_artifact

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
REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"

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
    """One of the three authenticated, non-customizable audit contracts."""

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

APPLIED_MEMBERS: tuple[tuple[str, int, str], ...] = (
    ("application_receipt.json", 258168, "767a29af6e6a72a052caf4a65867d0245315e64533c92550ef8a8d257cbe5a1b"),
    ("baseline_cpu_metadata.json", 849778, "40fc5791a5c5e81e847ca57e03328ec885075954b5d639a1596c699f00665c24"),
    ("baseline_whole_metadata.json", 1149393, "4c7102a2b1948980d56fe8eb42d84bcfe26413c92caf1e3ec00972ba13ace721"),
    ("candidate.lock", 38252, "d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8"),
    ("final.lock", 35980, "21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6"),
    ("overlay_cpu_metadata.json", 847544, "1c3bf26f65e4a13947ef5e5ed88d89feb6e9d4afd562761a5cd315b1c03a2b94"),
    ("overlay_whole_metadata.json", 1145613, "2bfa92f66975af7ca65c68c2d769f9224c319ef6fecd0e34441b0be8b653b6b2"),
    ("source_receipt.json", 8320, "157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec"),
)

APPLIED_PATH_PARENT_NAMES = (
    "gemm", "gemm-c32", "gemm-c64", "gemm-common", "gemm-f16", "gemm-f32", "gemm-f64", "metal",
)
APPLIED_PATH_PARENT_IDENTITIES = tuple(
    (name, version, None)
    for name in APPLIED_PATH_PARENT_NAMES
    for version in (("0.27.0", "0.29.0") if name == "metal" else ("0.17.1", "0.18.2"))
)
APPLIED_SOURCE_ABSENT_IDENTITIES = APPLIED_PATH_PARENT_IDENTITIES + (("moshi", "0.6.4", None),)

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
APPLIED_PROFILE = AuditProfile(
    "applied", 38758745134, 1, "89fbbc512beac5debbf4807a1dde3f2e8588096e",
    11985400172, "kyutai-candle-paste-application-evidence-38758745134-1", 259361,
    (), APPLIED_MEMBERS, 4333048, "final.lock",
    "21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6", 154,
    "candle_applied_final_lock", 105,
)
CURRENT_PROFILE = AuditProfile(
    "current", 0, 0, "", 0, "", 0, (), (), MAX_ARTIFACT_BYTES,
    "final.lock", APPLIED_PROFILE.lock_sha256, 154, "candle_current_final_lock", 105,
)
_PROFILES = {profile.name: profile for profile in (BASELINE_PROFILE, CANDIDATE_PROFILE, APPLIED_PROFILE, CURRENT_PROFILE)}


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
    if profile is BASELINE_PROFILE or profile is CANDIDATE_PROFILE or profile is APPLIED_PROFILE or profile is CURRENT_PROFILE:
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


def _reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


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
        value = json.loads(bytes(raw), object_pairs_hook=_reject_duplicate_json)
    except json.JSONDecodeError as exc:
        raise AuditError("GitHub API response is not JSON") from exc
    if not isinstance(value, dict):
        raise AuditError("GitHub API response must be an object")
    return value


def _current_env_int(name: str) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw)
    except ValueError as exc:
        raise AuditError(f"{name} must be an integer") from exc
    if value <= 0:
        raise AuditError(f"{name} must be positive")
    return value


def _current_env_hex(name: str, length: int) -> str:
    value = os.environ.get(name, "")
    if len(value) != length or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise AuditError(f"{name} must be lowercase hexadecimal")
    return value


def _current_artifacts(listing: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = listing.get("artifacts")
    total = listing.get("total_count")
    if not isinstance(artifacts, list) or type(total) is not int or total != len(artifacts) or total > 100:
        raise AuditError("artifact API response is incomplete or malformed")
    if any(not isinstance(item, dict) for item in artifacts):
        raise AuditError("artifact API contains a malformed row")
    ids = [item.get("id") for item in artifacts]
    names = [item.get("name") for item in artifacts]
    if any(type(value) is not int or value <= 0 for value in ids) or any(not isinstance(value, str) or not value for value in names):
        raise AuditError("artifact API identity fields are malformed")
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise AuditError("artifact API contains duplicate IDs or names")
    return artifacts


def _current_jobs(listing: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = listing.get("jobs")
    total = listing.get("total_count")
    if not isinstance(jobs, list) or type(total) is not int or total != len(jobs) or total > 100:
        raise AuditError("producer job API response is incomplete or malformed")
    if any(not isinstance(item, dict) for item in jobs):
        raise AuditError("producer job API contains a malformed row")
    ids = [item.get("id") for item in jobs]
    names = [item.get("name") for item in jobs]
    if any(type(value) is not int or value <= 0 for value in ids) or any(not isinstance(value, str) or not value for value in names):
        raise AuditError("producer job identity fields are malformed")
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise AuditError("producer job API contains duplicate IDs or names")
    return jobs


def validate_current_producer(*, repository: str, token: str, envelope: dict[str, Any], api: Callable[[str, str, float], dict[str, Any]], deadline: float) -> dict[str, Any]:
    current_artifact.validate_envelope(envelope)
    if repository != current_artifact.PRODUCTION_REPOSITORY or not token:
        raise AuditError("current producer binding requires the exact production repository and token")
    expected_run = _current_env_int("VOKRA_CANDLE_CURRENT_RUN_ID")
    expected_attempt = _current_env_int("VOKRA_CANDLE_CURRENT_RUN_ATTEMPT")
    expected_head = _current_env_hex("VOKRA_CANDLE_CURRENT_SOURCE_HEAD", 40)
    expected_workflow_sha = _current_env_hex("VOKRA_CANDLE_CURRENT_WORKFLOW_SHA", 40)
    expected_ref = os.environ.get("VOKRA_CANDLE_CURRENT_RUN_REF", "")
    if not isinstance(expected_ref, str) or not current_artifact.REF.fullmatch(expected_ref):
        raise AuditError("VOKRA_CANDLE_CURRENT_RUN_REF is not a supported GitHub ref")
    producer_head = _current_env_hex("VOKRA_CANDLE_PRODUCER_SOURCE_HEAD", 40)
    producer_workflow_sha = _current_env_hex("VOKRA_CANDLE_PRODUCER_WORKFLOW_SHA", 40)
    producer_ref = os.environ.get("VOKRA_CANDLE_PRODUCER_RUN_REF", "")
    producer_caller = os.environ.get("VOKRA_CANDLE_PRODUCER_CALLER", "")
    caller = os.environ.get("VOKRA_CANDLE_CURRENT_CALLER", "")
    if caller not in current_artifact.CALLERS or caller == "standalone":
        raise AuditError("current producer caller is not an active registered consumer")
    if producer_head != expected_head or producer_workflow_sha != expected_workflow_sha or producer_ref != expected_ref or producer_caller != caller:
        raise AuditError("producer outputs do not match independent caller context")
    expected_workflow_name = current_artifact.CALLERS[caller]
    primary_id = _current_env_int("VOKRA_CANDLE_CURRENT_ARTIFACT_ID")
    primary_name = os.environ.get("VOKRA_CANDLE_CURRENT_ARTIFACT_NAME", "")
    primary_upload_digest = current_artifact.normalize_digest(os.environ.get("VOKRA_CANDLE_CURRENT_ARTIFACT_DIGEST", ""))
    envelope_id = _current_env_int("VOKRA_CANDLE_CURRENT_ENVELOPE_ID")
    envelope_name = os.environ.get("VOKRA_CANDLE_CURRENT_ENVELOPE_NAME", "")
    envelope_upload_digest = current_artifact.normalize_digest(os.environ.get("VOKRA_CANDLE_CURRENT_ENVELOPE_DIGEST", ""))
    expected_envelope_name = f"{current_artifact.ENVELOPE_ARTIFACT_PREFIX}{expected_run}-{expected_attempt}-{caller}"
    if envelope_name != expected_envelope_name:
        raise AuditError("current envelope output name is not bound to the caller run")
    expected_caller_path = current_artifact.CALLER_PATHS[caller]
    if envelope.get("run_id") != expected_run or envelope.get("run_attempt") != expected_attempt or envelope.get("source_head") != expected_head or envelope.get("workflow_sha") != expected_workflow_sha or envelope.get("run_ref") != expected_ref or envelope.get("caller") != caller or envelope.get("caller_workflow_path") != expected_caller_path:
        raise AuditError("current evidence envelope does not bind to the caller run")
    if envelope.get("artifact_id") != primary_id or envelope.get("artifact_name") != primary_name or envelope.get("artifact_digest") != primary_upload_digest:
        raise AuditError("current evidence envelope does not bind to upload outputs")
    base = f"https://api.github.com/repos/{repository}"
    run = api(f"{base}/actions/runs/{expected_run}", token, deadline)
    repository_binding = run.get("repository")
    if run.get("id") != expected_run or run.get("run_attempt") != expected_attempt or run.get("head_sha") != expected_head or run.get("status") != "in_progress" or run.get("conclusion") is not None or run.get("name") != expected_workflow_name or run.get("path") != expected_caller_path or not isinstance(repository_binding, dict) or repository_binding.get("full_name") != repository:
        raise AuditError("current run identity/status/workflow mismatch")
    references = run.get("referenced_workflows")
    if not isinstance(references, list):
        raise AuditError("current run lacks reusable workflow binding")
    matching_references = [item for item in references if isinstance(item, dict) and item.get("path") in current_artifact.referenced_workflow_paths(expected_workflow_sha)]
    if len(matching_references) != 1 or matching_references[0].get("sha") != expected_workflow_sha or matching_references[0].get("ref") != expected_ref:
        raise AuditError("current reusable workflow path/SHA binding mismatch")
    jobs = _current_jobs(api(f"{base}/actions/runs/{expected_run}/attempts/{expected_attempt}/jobs?per_page=100", token, deadline))
    matching_jobs = [item for item in jobs if item.get("name") in current_artifact.PRODUCER_JOB_API_NAMES[caller]]
    if len(matching_jobs) != 1:
        raise AuditError("current producer job is missing or duplicated")
    job = matching_jobs[0]
    if type(job.get("id")) is not int or job.get("id") <= 0 or job.get("run_id") != expected_run or job.get("run_attempt") != expected_attempt or job.get("head_sha") != expected_head or job.get("workflow_name") != expected_workflow_name or job.get("status") != "completed" or job.get("conclusion") != "success":
        raise AuditError("current producer job identity/status mismatch")
    artifacts = _current_artifacts(api(f"{base}/actions/runs/{expected_run}/artifacts?per_page=100", token, deadline))
    primary_matches = [item for item in artifacts if item.get("id") == primary_id]
    envelope_matches = [item for item in artifacts if item.get("id") == envelope_id]
    if len(primary_matches) != 1 or len(envelope_matches) != 1:
        raise AuditError("current producer artifact ID is missing or duplicated")
    primary, envelope_artifact = primary_matches[0], envelope_matches[0]
    if primary.get("name") != primary_name or primary.get("expired") is not False or primary.get("digest") is None or current_artifact.normalize_digest(primary.get("digest")) != primary_upload_digest or primary.get("size_in_bytes") != envelope.get("artifact_size") or type(primary.get("size_in_bytes")) is not int or not 0 < primary["size_in_bytes"] <= MAX_ARTIFACT_BYTES:
        raise AuditError("current primary artifact identity/digest/size mismatch")
    if envelope_artifact.get("name") != envelope_name or envelope_artifact.get("expired") is not False or envelope_artifact.get("digest") is None or current_artifact.normalize_digest(envelope_artifact.get("digest")) != envelope_upload_digest or type(envelope_artifact.get("size_in_bytes")) is not int or not 0 < envelope_artifact["size_in_bytes"] <= current_artifact.MAX_ENVELOPE_BYTES:
        raise AuditError("current envelope artifact identity/digest/size mismatch")
    for artifact in (primary, envelope_artifact):
        workflow_run = artifact.get("workflow_run")
        if not isinstance(workflow_run, dict) or workflow_run.get("id") != expected_run or workflow_run.get("head_sha") != expected_head or "run_attempt" in workflow_run and workflow_run.get("run_attempt") != expected_attempt:
            raise AuditError("current artifact workflow binding mismatch")
    return {"run_id": expected_run, "run_attempt": expected_attempt, "source_head": expected_head, "workflow_sha": expected_workflow_sha, "run_ref": expected_ref, "caller": caller, "workflow_name": expected_workflow_name, "producer_job_id": job["id"], "producer_job_name": job["name"], "primary_artifact": {"id": primary_id, "name": primary_name, "size_in_bytes": primary["size_in_bytes"], "digest": primary_upload_digest}, "envelope_artifact": {"id": envelope_id, "name": envelope_name, "size_in_bytes": envelope_artifact["size_in_bytes"], "digest": envelope_upload_digest}, "referenced_workflow": {"path": matching_references[0]["path"], "ref": expected_ref, "sha": expected_workflow_sha}}


def validate_producer(*, repository: str, token: str, run_id: int | None = None, artifact_id: int | None = None, expected_head: str | None = None, api: Callable[[str, str, float], dict[str, Any]] = _api_json, deadline: float, profile: str | AuditProfile | None = None, current_envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = resolve_profile(profile)
    if selected is CURRENT_PROFILE:
        if current_envelope is None:
            raise AuditError("current producer envelope is required")
        return validate_current_producer(repository=repository, token=token, envelope=current_envelope, api=api, deadline=deadline)
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


def _artifact_json(files: dict[str, Path], relative: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_bounded(files[relative], MAX_JSON_BYTES))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"applied evidence is not valid JSON: {relative}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"applied evidence is not an object: {relative}")
    return value


def _lock_packages(files: dict[str, Path], relative: str) -> list[dict[str, Any]]:
    try:
        value = tomllib.loads(_read_bounded(files[relative], MAX_MEMBER_BYTES).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError(f"applied lock is not valid TOML: {relative}") from exc
    packages = value.get("package")
    if not isinstance(packages, list):
        raise AuditError(f"applied lock has no package list: {relative}")
    if any(not isinstance(item, dict) for item in packages):
        raise AuditError(f"applied lock package row is malformed: {relative}")
    return packages


def _identity(row: dict[str, Any]) -> tuple[str, str, str | None]:
    if not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str):
        raise AuditError("package identity is malformed")
    source = row.get("source")
    if source is not None and not isinstance(source, str):
        raise AuditError("package source is malformed")
    return row["name"], row["version"], source


def _metadata_graph(payload: dict[str, Any], *, expected_status: str, expected_count: int, expected_target: str | None) -> tuple[set[tuple[str, str, str | None]], set[str]]:
    if payload.get("status") != expected_status or payload.get("approval") is not None or payload.get("upload") != "NO_UPLOAD" or payload.get("model_execution") != "NOT_AUTHORIZED" or payload.get("source_only") is not True:
        raise AuditError("applied metadata provenance is not an unapproved source-only packet")
    if payload.get("target") != expected_target or payload.get("platform_filter") != expected_target:
        raise AuditError("applied metadata target boundary mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise AuditError("applied metadata payload is missing metadata")
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    nodes = resolve.get("nodes") if isinstance(resolve, dict) else None
    if not isinstance(packages, list) or len(packages) != expected_count or not isinstance(nodes, list) or len(nodes) != expected_count:
        raise AuditError("applied metadata graph count mismatch")
    package_ids: set[str] = set()
    identities: set[tuple[str, str, str | None]] = set()
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("id"), str):
            raise AuditError("applied metadata package identity is malformed")
        package_ids.add(package["id"])
        identities.add(_identity(package))
    if len(package_ids) != expected_count or len(identities) != expected_count:
        raise AuditError("applied metadata package identities are duplicated")
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not isinstance(node.get("deps"), list):
            raise AuditError("applied metadata node is malformed")
        node_ids.add(node["id"])
        for dependency in node["deps"]:
            if not isinstance(dependency, dict) or dependency.get("pkg") not in package_ids or not isinstance(dependency.get("dep_kinds"), list) or not dependency["dep_kinds"]:
                raise AuditError("applied metadata dependency references an unknown node")
            for dep_kind in dependency["dep_kinds"]:
                if not isinstance(dep_kind, dict) or dep_kind.get("kind") not in {None, "normal", "build", "dev"} or dep_kind.get("target") is not None and not isinstance(dep_kind.get("target"), str):
                    raise AuditError("applied metadata dependency kind is malformed")
    if node_ids != package_ids:
        raise AuditError("applied metadata package/node identities disagree")
    if resolve.get("root") is not None:
        raise AuditError("applied metadata must retain the virtual workspace root")
    members = metadata.get("workspace_members")
    if not isinstance(members, list) or len(members) != 1 or members[0] not in package_ids:
        raise AuditError("applied metadata workspace member is not unique")
    member = next(package for package in packages if package["id"] == members[0])
    if member.get("name") != "moshi" or member.get("version") != "0.6.4" or member.get("source") is not None:
        raise AuditError("applied metadata workspace member is not the source-absent Moshi root")
    return identities, package_ids


def _validate_applied_metadata(files: dict[str, Path]) -> None:
    application = _artifact_json(files, "application_receipt.json")
    if application.get("status") != "SOURCE_APPLICATION_REVIEW_PENDING" or application.get("approval") is not None or application.get("upload") != "NO_UPLOAD" or application.get("model_execution") != "NOT_AUTHORIZED" or application.get("source_only") is not True:
        raise AuditError("applied application receipt is not review-pending source-only evidence")
    source = _artifact_json(files, "source_receipt.json")
    if source.get("source_commit") != "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362" or not isinstance(source.get("files"), dict) or len(source["files"]) != 23:
        raise AuditError("applied source receipt binding mismatch")
    final_rows = _lock_packages(files, "final.lock")
    candidate_rows = _lock_packages(files, "candidate.lock")
    if len(final_rows) != APPLIED_PROFILE.lock_dependency_count or len(candidate_rows) != APPLIED_PROFILE.lock_dependency_count:
        raise AuditError("applied lock dependency count is not 154")
    final_ids = [_identity(row) for row in final_rows]
    candidate_ids = [_identity(row) for row in candidate_rows]
    if len(set(final_ids)) != 154 or len(set(candidate_ids)) != 154:
        raise AuditError("applied lock contains duplicate identities")
    expected_absent = set(APPLIED_SOURCE_ABSENT_IDENTITIES)
    actual_absent = {identity for identity in final_ids if identity[2] is None}
    if actual_absent != expected_absent or sum(identity[2] == REGISTRY_SOURCE for identity in final_ids) != 137:
        raise AuditError("applied final lock source identity set mismatch")
    for identity, row in zip(final_ids, final_rows):
        if identity[2] is None and ("source" in row or "checksum" in row):
            raise AuditError("applied source-absent package contains source/checksum fields")
    candidate_by_id = {identity: row for identity, row in zip(candidate_ids, candidate_rows)}
    final_by_id = {identity: row for identity, row in zip(final_ids, final_rows)}
    parent_old = {(name, version, REGISTRY_SOURCE) for name, version, _ in APPLIED_PATH_PARENT_IDENTITIES}
    expected_added = expected_absent - {("moshi", "0.6.4", None)}
    expected_removed = parent_old | {("paste", "1.0.15", REGISTRY_SOURCE)}
    expected_added = expected_added | {("pastey", "0.2.3", REGISTRY_SOURCE)}
    if set(candidate_by_id) - set(final_by_id) != expected_removed or set(final_by_id) - set(candidate_by_id) != expected_added:
        raise AuditError("applied candidate-to-final lock identity delta is not the fixed 17-transition map")
    for identity in set(candidate_by_id) & set(final_by_id):
        if candidate_by_id[identity] != final_by_id[identity]:
            raise AuditError("unrelated applied lock record changed")
    for old_id in parent_old:
        old = candidate_by_id[old_id]
        new = final_by_id[(old_id[0], old_id[1], None)]
        expected = {key: value for key, value in old.items() if key not in {"source", "checksum"}}
        expected["dependencies"] = ["pastey" if dep == "paste" else dep for dep in expected.get("dependencies", [])]
        if new != expected:
            raise AuditError("applied path-parent lock record changed outside source/checksum/paste mapping")
    pastey = final_by_id[("pastey", "0.2.3", REGISTRY_SOURCE)]
    if not isinstance(pastey.get("checksum"), str) or not pastey.get("checksum"):
        raise AuditError("applied pastey registry record lacks checksum")
    mapped_order = []
    for identity in candidate_ids:
        if identity in parent_old:
            mapped_order.append((identity[0], identity[1], None))
        elif identity == ("paste", "1.0.15", REGISTRY_SOURCE):
            mapped_order.append(("pastey", "0.2.3", REGISTRY_SOURCE))
        else:
            mapped_order.append(identity)
    if mapped_order != final_ids:
        raise AuditError("applied lock package order changed outside the fixed identity mapping")

    whole_ids, _whole_nodes = _metadata_graph(_artifact_json(files, "baseline_whole_metadata.json"), expected_status="OVERLAY_BASELINE_WHOLE_UNREVIEWED", expected_count=154, expected_target=None)
    cpu_ids, _ = _metadata_graph(_artifact_json(files, "baseline_cpu_metadata.json"), expected_status="OVERLAY_BASELINE_CPU_UNREVIEWED", expected_count=105, expected_target="x86_64-unknown-linux-gnu")
    overlay_whole_ids, _ = _metadata_graph(_artifact_json(files, "overlay_whole_metadata.json"), expected_status="OVERLAY_WHOLE_METADATA_UNREVIEWED", expected_count=154, expected_target=None)
    overlay_cpu_ids, _ = _metadata_graph(_artifact_json(files, "overlay_cpu_metadata.json"), expected_status="OVERLAY_CPU_METADATA_UNREVIEWED", expected_count=105, expected_target="x86_64-unknown-linux-gnu")
    if whole_ids != set(candidate_ids) or overlay_whole_ids != set(final_ids) or not cpu_ids <= whole_ids or not overlay_cpu_ids <= overlay_whole_ids:
        raise AuditError("applied CPU/whole metadata identity binding mismatch")


def _validate_applied_source_absence(db: Path, lockfile: Path, *, deadline: float | None = None) -> dict[str, Any]:
    if deadline is not None and _now() >= deadline:
        raise AuditError("applied supplemental coverage deadline exceeded")
    crates = db / "crates"
    if not crates.is_dir() or crates.is_symlink():
        raise AuditError("fixed RustSec database crates directory is missing or symlinked")
    try:
        rows = tomllib.loads(_read_bounded(lockfile, MAX_MEMBER_BYTES).decode("utf-8")).get("package")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise AuditError("applied audit lock cannot be parsed") from exc
    if not isinstance(rows, list) or len(rows) != APPLIED_PROFILE.lock_dependency_count:
        raise AuditError("applied audit lock package count is not 154")
    identities = [_identity(row) for row in rows if isinstance(row, dict)]
    if len(identities) != len(rows) or len(set(identities)) != len(identities):
        raise AuditError("applied audit lock identities are malformed")
    absent = {identity for identity in identities if identity[2] is None}
    if absent != set(APPLIED_SOURCE_ABSENT_IDENTITIES):
        raise AuditError("applied audit source-absent identity set mismatch")
    if sum(identity[2] == REGISTRY_SOURCE for identity in identities) != 137:
        raise AuditError("applied audit registry package count is not 137")
    checked = []
    for identity, row in zip(identities, rows):
        if identity[2] is None and ("source" in row or "checksum" in row):
            raise AuditError("applied source-absent package contains source/checksum fields")
    for name, version, _ in APPLIED_SOURCE_ABSENT_IDENTITIES:
        if deadline is not None and _now() >= deadline:
            raise AuditError("applied supplemental coverage deadline exceeded")
        entry = crates / name
        try:
            entry.lstat()
        except FileNotFoundError:
            pass
        else:
            raise AuditError(f"fixed RustSec database unexpectedly contains a crate path: {name}")
        checked.append({"name": name, "version": version, "source": None, "database_path": f"crates/{name}", "role": "workspace_root" if name == "moshi" else "path_parent", "status": "ABSENT"})
    if deadline is not None and _now() >= deadline:
        raise AuditError("applied supplemental coverage deadline exceeded")
    return {"database_commit": ADVISORY_DB_COMMIT, "database_tree": ADVISORY_DB_TREE, "registry_package_count": 137, "source_absent_package_count": 17, "source_absent_packages": checked, "coverage_scope": "FIXED_DB_ABSENCE_ONLY_NOT_A_GENERIC_SCANNER"}


def validate_artifact(root: Path, *, profile: str | AuditProfile | None = None, envelope: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = resolve_profile(profile)
    if selected is BASELINE_PROFILE:
        expected = EXPECTED_MEMBERS
        files = _walk_regular_files(root)
    elif selected is CURRENT_PROFILE:
        if envelope is None:
            raise AuditError("current artifact envelope is required")
        current_artifact.validate_envelope(envelope)
        expected = {row["path"]: (row["bytes"], row["sha256"]) for row in envelope["members"]}
        files = _walk_regular_files(root, max_bytes=MAX_ARTIFACT_BYTES, exact_bytes=sum(size for size, _digest in expected.values()))
    else:
        expected = {relative: (size, digest) for relative, size, digest in selected.expected_members}
        files = _walk_regular_files(root, max_bytes=selected.extracted_bytes, exact_bytes=selected.extracted_bytes)
        if selected.member_prefixes and not all(any(relative.startswith(prefix) for prefix in selected.member_prefixes) for relative in files):
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
    elif selected is APPLIED_PROFILE or selected is CURRENT_PROFILE:
        _validate_applied_metadata(files)
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


def _load_current_envelope(path: Path | None) -> dict[str, Any]:
    if path is None or path.is_symlink() or not path.is_file():
        raise AuditError("current producer envelope file is missing or unsafe")
    try:
        value = json.loads(_read_bounded(path, current_artifact.MAX_ENVELOPE_BYTES), object_pairs_hook=_reject_duplicate_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("current producer envelope is not valid JSON") from exc
    try:
        return current_artifact.validate_envelope(value)
    except current_artifact.CurrentArtifactError as exc:
        raise AuditError(str(exc)) from exc


def run_audit(*, artifact_root: Path, db: Path, cwd: Path, cargo_home: Path, auditor: str, repository: str, output: Path, deadline_seconds: float = DEFAULT_DEADLINE_SECONDS, api: Callable[[str, str, float], dict[str, Any]] = _api_json, command_runner: Callable[..., CommandResult] = run_command, verifier: Callable[..., dict[str, Any]] = verify_auditor, producer_validator: Callable[..., dict[str, Any]] = validate_producer, artifact_validator: Callable[..., dict[str, Any]] = validate_artifact, provenance: Callable[..., dict[str, Any]] = git_provenance, host_guard: Callable[[dict[str, str]], None] = _host_guard, profile: str | AuditProfile | None = None, current_envelope_path: Path | None = None) -> int:
    if type(deadline_seconds) not in (int, float) or not math.isfinite(float(deadline_seconds)) or deadline_seconds <= 0 or deadline_seconds > DEFAULT_DEADLINE_SECONDS:
        raise AuditError("deadline must be finite, positive, and no greater than 480 seconds")
    selected = resolve_profile(profile)
    deadline = _now() + deadline_seconds
    if selected is BASELINE_PROFILE:
        lock_hash_field = "patched_lock_sha256"
    elif selected is APPLIED_PROFILE or selected is CURRENT_PROFILE:
        lock_hash_field = "final_lock_sha256"
    else:
        lock_hash_field = "candidate_lock_sha256"
    evidence: dict[str, Any] = {"scope": selected.evidence_scope, "status": "ERROR", "advisory_audit": "ERROR", "approval": None, "license_review": "NOT_PERFORMED", "root_runtime_audit": "NOT_RUN", "model_execution": "NOT_AUTHORIZED", "upload": "NO_UPLOAD", "lock_records": selected.lock_dependency_count, "cpu_active_packages": selected.cpu_active_packages, lock_hash_field: selected.lock_sha256, "tool": {"expected_cargo_audit_version": CARGO_AUDIT_VERSION, "install_action_commit": INSTALL_ACTION_COMMIT, "release_commit": CARGO_AUDIT_RELEASE_COMMIT}, "advisory_database": {"commit": ADVISORY_DB_COMMIT, "tree": ADVISORY_DB_TREE, "no_fetch": True, "report_last_commit_fact": "PINNED_DATABASE_OPEN_DOES_NOT_POPULATE_COMMIT"}, "yanked_check": "NOT_CHECKED", "producer": None, "artifact": None, "audit": None}
    if selected is APPLIED_PROFILE or selected is CURRENT_PROFILE:
        evidence.update({"native_registry_package_count": 137, "parsed_lock_record_count": 154, "supplemental_path_parent_package_count": 16, "supplemental_workspace_package_count": 1, "supplemental_coverage": "FIXED_DB_ABSENCE_ONLY_NOT_A_GENERIC_SCANNER"})
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
        current_envelope = _load_current_envelope(current_envelope_path) if selected is CURRENT_PROFILE else None
        if selected is BASELINE_PROFILE:
            evidence["producer"] = producer_validator(repository=repository, token=token, api=api, deadline=deadline)
            evidence["artifact"] = artifact_validator(artifact_root)
        elif selected is CURRENT_PROFILE:
            evidence["producer"] = producer_validator(repository=repository, token=token, api=api, deadline=deadline, profile=selected, current_envelope=current_envelope)
            evidence["artifact"] = artifact_validator(artifact_root, profile=selected, envelope=current_envelope)
        else:
            evidence["producer"] = producer_validator(repository=repository, token=token, api=api, deadline=deadline, profile=selected)
            evidence["artifact"] = artifact_validator(artifact_root, profile=selected)
        before = provenance(db, cwd=cwd, env=env, deadline=deadline, runner=command_runner)
        lockfile = artifact_root / selected.lock_relative_path
        lock_before_hash = None
        if selected is APPLIED_PROFILE or selected is CURRENT_PROFILE:
            lock_before_hash = _sha256(_read_bounded(lockfile, MAX_MEMBER_BYTES))
            if lock_before_hash != selected.lock_sha256:
                raise AuditError("registered audit lock hash changed before cargo-audit")
            evidence["supplemental_coverage_before"] = _validate_applied_source_absence(db, lockfile, deadline=deadline)
        result = command_runner(build_audit_argv(verified_auditor, lockfile, db), cwd=cwd, env=env, deadline=deadline, stdout_cap=MAX_STDOUT_BYTES, stderr_cap=MAX_STDERR_BYTES)
        audit: dict[str, Any] = {"command": _command_evidence(result), "git_before": before}
        evidence["audit"] = audit
        if selected is APPLIED_PROFILE or selected is CURRENT_PROFILE:
            lock_after_command_hash = _sha256(_read_bounded(lockfile, MAX_MEMBER_BYTES))
            if lock_after_command_hash != lock_before_hash:
                audit["classification"] = "ERROR_LOCK_MUTATED"
                _write_json(output, evidence)
                return 2
            try:
                evidence["supplemental_coverage_after_command"] = _validate_applied_source_absence(db, lockfile, deadline=deadline)
            except AuditError as exc:
                classification = "ERROR_REQUIRES_PATH_ADVISORY_REVIEW" if "unexpectedly contains a crate path" in str(exc) else "ERROR_SUPPLEMENTAL_COVERAGE"
                audit.update({"classification": classification, "error": {"type": type(exc).__name__, "message": str(exc)}})
                _write_json(output, evidence)
                return 2
            if evidence["supplemental_coverage_after_command"] != evidence["supplemental_coverage_before"]:
                audit.update({"classification": "ERROR_SUPPLEMENTAL_COVERAGE_CHANGED", "error": {"type": "AuditError", "message": "fixed-DB source-absence evidence changed during audit"}})
                _write_json(output, evidence)
                return 2
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
        if (selected is APPLIED_PROFILE or selected is CURRENT_PROFILE) and _sha256(_read_bounded(lockfile, MAX_MEMBER_BYTES)) != selected.lock_sha256:
            audit.update({"classification": "ERROR_LOCK_MUTATED", "error": {"type": "AuditError", "message": "registered audit lock changed during final provenance check"}})
            _write_json(output, evidence)
            return 2
        if _now() >= deadline:
            audit.update({"classification": "ERROR_DEADLINE", "error": {"type": "AuditError", "message": "shared audit deadline exceeded before final classification"}})
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
    parser.add_argument("--current-envelope", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or args.deadline_seconds > DEFAULT_DEADLINE_SECONDS:
        parser.error("deadline must be positive and no greater than 480 seconds")
    return run_audit(artifact_root=args.artifact_root, db=args.db, cwd=args.cwd, cargo_home=args.cargo_home, auditor=args.auditor, repository=args.repository, output=args.output, deadline_seconds=args.deadline_seconds, profile=args.profile, current_envelope_path=args.current_envelope)


if __name__ == "__main__":
    raise SystemExit(main())
