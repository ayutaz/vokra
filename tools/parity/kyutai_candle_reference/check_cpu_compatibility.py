#!/usr/bin/env python3
"""One-shot Linux CPU compatibility check for the authenticated Candle overlay.

This is deliberately a diagnostic.  It authenticates the existing source-only
application packet, replays the frozen applier, then performs one offline
library build and a type-signature probe.  It never loads models or runs
inference and never turns a compiler result into a legal or parity approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


MAX_DEADLINE = 900
TARGET = "x86_64-unknown-linux-gnu"
APPLIED_RUN = "38758745134"
APPLIED_ATTEMPT = "1"
APPLIED_HEAD = "89fbbc512beac5debbf4807a1dde3f2e8588096e"
APPLIED_ARTIFACT = "kyutai-candle-paste-application-evidence-38758745134-1"
FINAL_LOCK_SHA256 = "21bb31459ac8c2f337cba3e9f79d0e64f48259dc4ed0df9c99ccb0480578a7d6"
SOURCE_RECEIPT_SHA256 = "157eae9e76a348ff25f9fcf10d3d5943b775563648a7505e2c3e7046670d82ec"
CPU_FINGERPRINT = "99364d685eb1898f2ccf442c68fb51d70f0b5e42f90c04dd0171cb07933be51b"
CPU_EVIDENCE_SHA256 = "58163fc14031daaea2181c14f19b8a309946fe680c535629772426399479c4fe"
ROOT_MANIFEST_SHA256 = "7f77735418b92d679e7eaa1eda3345165f6c5f19a238d5b55e5907374eec29e5"

FROZEN_FILES = {
    "tools/parity/kyutai_candle_reference/apply_paste_overlay.py":
        "63a010aabfaa2e0b974cce634befdf5be3cb6d4533dcc6404351292296319e76",
    "tools/parity/kyutai_candle_reference/collect_metadata.py":
        "75bdaa3da0ce7a3add28a1018e7d319b48b9f18448d7c3093c55fa94fdb2773f",
    "tools/parity/kyutai_candle_reference/pin_remediation_diagnostic.py":
        "7bb61d1f2adaa58e64492584f67ecf7ca2b8d5f8f3943aa2a143b767e8661f6d",
    "tools/parity/kyutai_candle_reference/prepare_paste_overlay.py":
        "3e204bd5c0d47c097e5b5396f6c9787c39b49a296aedaf7c7ce528b0a4515dec",
}

APPLICATION_FILES = {
    "application_receipt.json": "767a29af6e6a72a052caf4a65867d0245315e64533c92550ef8a8d257cbe5a1b",
    "baseline_cpu_metadata.json": "40fc5791a5c5e81e847ca57e03328ec885075954b5d639a1596c699f00665c24",
    "baseline_whole_metadata.json": "4c7102a2b1948980d56fe8eb42d84bcfe26413c92caf1e3ec00972ba13ace721",
    "candidate.lock": "d10a08e5b3c519c86ab04654eaf88e9d064904b50cdbf1752dd9d629cb725cc8",
    "final.lock": FINAL_LOCK_SHA256,
    "overlay_cpu_metadata.json": "1c3bf26f65e4a13947ef5e5ed88d89feb6e9d4afd562761a5cd315b1c03a2b94",
    "overlay_whole_metadata.json": "2bfa92f66975af7ca65c68c2d769f9224c319ef6fecd0e34441b0be8b653b6b2",
    "source_receipt.json": SOURCE_RECEIPT_SHA256,
}
GPU_TOKENS = ("cuda", "cudarc", "metal", "accelerate", "mkl", "flash-attn")
ROOT_FEATURES = {
    "candle-core": ["default"],
    "candle-nn": ["default"],
    "candle-transformers": ["default"],
    "moshi": [],
}
REQUIRED_BUILD_ARTIFACTS = (
    "gemm@0.17.1", "gemm@0.18.2",
    "gemm-c32@0.17.1", "gemm-c32@0.18.2",
    "gemm-c64@0.17.1", "gemm-c64@0.18.2",
    "gemm-common@0.17.1", "gemm-common@0.18.2",
    "gemm-f16@0.17.1", "gemm-f16@0.18.2",
    "gemm-f32@0.17.1", "gemm-f32@0.18.2",
    "gemm-f64@0.17.1", "gemm-f64@0.18.2",
    "pastey@0.2.3",
)
REQUIRED_BUILD_KINDS = {item: (["proc-macro"] if item.startswith("pastey@") else ["lib"]) for item in REQUIRED_BUILD_ARTIFACTS}
NAMESPACE_PREFIX = [
    "unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
    "--mount-proc", "--kill-child=KILL", "setpriv", "--no-new-privs", "--",
]


class CompatibilityError(RuntimeError):
    pass


class BoundedCommandError(CompatibilityError):
    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise CompatibilityError("shared compatibility deadline expired")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompatibilityError(f"JSON root is not an object: {path}")
    return value


def _bounded_read(path: Path, limit: int = 2 * 1024 * 1024) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CompatibilityError(f"missing or non-regular evidence file: {path}")
    if path.stat().st_size > limit:
        raise CompatibilityError(f"evidence exceeds bound: {path}")
    data = path.read_bytes()
    if len(data) > limit:
        raise CompatibilityError(f"evidence grew beyond bound: {path}")
    return data


def _tree_fingerprint(root: Path, *, deadline: float | None = None) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise CompatibilityError(f"required source tree missing: {root}")
    rows: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        if deadline is not None:
            _remaining(deadline)
        if path.is_symlink():
            raise CompatibilityError(f"source tree contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            rows.append((relative, path.stat().st_size, _sha256(path)))
    encoded = json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode()
    return {"files": len(rows), "bytes": sum(row[1] for row in rows), "sha256": hashlib.sha256(encoded).hexdigest()}


def _selected_source_scope(metadata_path: Path, cargo_home: Path, *, deadline: float) -> dict[str, Any]:
    """Fingerprint projection/vendor and only registry sources selected by metadata."""
    metadata = _cargo_metadata(_json(metadata_path))
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise CompatibilityError("source-scope metadata packages are malformed")
    registry_packages = [
        package for package in packages
        if isinstance(package, dict) and isinstance(package.get("source"), str)
        and package["source"].startswith("registry+")
    ]
    registry_root = cargo_home / "registry" / "src"
    if not registry_root.is_dir() or registry_root.is_symlink():
        raise CompatibilityError("authenticated Cargo registry source root is missing")
    selected: dict[str, dict[str, Any]] = {}
    for package in registry_packages:
        _remaining(deadline)
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise CompatibilityError("registry source identity is malformed")
        identity = f"{name}@{version}"
        matches: list[Path] = []
        for index_dir in sorted(registry_root.iterdir()):
            _remaining(deadline)
            if index_dir.is_symlink() or not index_dir.is_dir():
                raise CompatibilityError("registry source index contains an invalid entry")
            candidate = index_dir / f"{name}-{version}"
            if candidate.exists():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise CompatibilityError(f"registry source identity is not a regular directory: {identity}")
                matches.append(candidate)
        if len(matches) != 1 or identity in selected:
            raise CompatibilityError(f"registry source identity is ambiguous or missing: {identity}")
        selected[identity] = _tree_fingerprint(matches[0], deadline=deadline)
    if len(selected) != len(registry_packages):
        raise CompatibilityError("registry source scope is incomplete")
    return {"registry_sources": selected}


def _source_snapshot(*, metadata_path: Path, workspace: Path, cargo_home: Path, deadline: float) -> dict[str, Any]:
    snapshot = {
        "projection": _tree_fingerprint(workspace / "projection", deadline=deadline),
        "vendor": _tree_fingerprint(workspace / "vendor", deadline=deadline),
    }
    snapshot.update(_selected_source_scope(metadata_path, cargo_home, deadline=deadline))
    return snapshot


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode()
    if len(encoded) > 2 * 1024 * 1024:
        raise CompatibilityError("compatibility receipt exceeds 2 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded + b"\n")


def _verify_fixed_files(root: Path) -> None:
    for relative, expected in FROZEN_FILES.items():
        path = root / relative
        if _sha256(path) != expected:
            raise CompatibilityError(f"frozen helper hash mismatch: {relative}")
    factual = root / "docs/handoff/kyutai-candle-current-cpu-evidence-2026-10-16.json"
    if _sha256(factual) != CPU_EVIDENCE_SHA256:
        raise CompatibilityError("current CPU evidence binding changed")


def _verify_application_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("status") != "SOURCE_APPLICATION_REVIEW_PENDING":
        raise CompatibilityError("application receipt is not review-pending")
    if receipt.get("approval") is not None or receipt.get("upload") != "NO_UPLOAD":
        raise CompatibilityError("application receipt has an approval/upload state")
    if receipt.get("model_execution") != "NOT_AUTHORIZED":
        raise CompatibilityError("application receipt authorizes model execution")
    for phase, expected in (("source_proof_pre", "PRE_OVERLAY"), ("source_proof_post", "POST_OVERLAY")):
        proof = receipt.get(phase)
        if not isinstance(proof, dict) or proof.get("phase") != expected or proof.get("files_checked") != 23 or proof.get("unchanged_files") != 21:
            raise CompatibilityError(f"application {phase} proof is incomplete")
        entries = proof.get("entries")
        if not isinstance(entries, list) or len(entries) != 23 or len({entry.get("path") for entry in entries if isinstance(entry, dict)}) != 23:
            raise CompatibilityError(f"application {phase} source entries are incomplete")
    if receipt["source_proof_pre"]["entries"] != receipt["source_proof_post"]["entries"]:
        raise CompatibilityError("application source proof changed between phases")
    checks_pre = receipt.get("member_premetadata_checks")
    checks_post = receipt.get("member_postchecks")
    if not isinstance(checks_pre, list) or not isinstance(checks_post, list):
        raise CompatibilityError("application member proofs are missing")
    for checks, label in ((checks_pre, "pre"), (checks_post, "post")):
        if sum(item.get("members_checked", 0) for item in checks if isinstance(item, dict)) != 308 or sum(item.get("non_manifest_members", 0) for item in checks if isinstance(item, dict)) != 292:
            raise CompatibilityError(f"application {label} member proof counts changed")
    return receipt


def _verify_application(root: Path) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise CompatibilityError("application artifact root is not a directory")
    entries = list(root.iterdir())
    actual = {p.name for p in entries}
    if actual != set(APPLICATION_FILES) or any(not p.is_file() or p.is_symlink() for p in entries):
        raise CompatibilityError(f"application artifact file set/type mismatch: {sorted(actual)}")
    if any(p.stat().st_size > 2 * 1024 * 1024 for p in entries):
        raise CompatibilityError("application evidence member exceeds 2 MiB")
    if sum(p.stat().st_size for p in entries) > 16 * 1024 * 1024:
        raise CompatibilityError("application evidence exceeds aggregate bound")
    for name, expected in APPLICATION_FILES.items():
        path = root / name
        if _sha256(path) != expected:
            raise CompatibilityError(f"application artifact hash mismatch: {name}")
    return _verify_application_receipt(_json(root / "application_receipt.json"))


def _sanitized_env(*, cargo_home: Path, target_dir: Path, offline: bool, home: Path | None = None) -> dict[str, str]:
    keep = {"PATH", "LANG", "LC_ALL", "CARGO_TERM_COLOR"}
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update({
        "HOME": str(home or cargo_home.parent / "home"),
        "CARGO_HOME": str(cargo_home),
        "CARGO_TARGET_DIR": str(target_dir),
        "CARGO_NET_OFFLINE": "true" if offline else "false",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    })
    for forbidden in ("RUSTFLAGS", "RUSTC_WRAPPER", "CARGO_BUILD_RUSTFLAGS", "HF_TOKEN", "HUGGINGFACE_TOKEN", "VAST_API_KEY"):
        env.pop(forbidden, None)
    return env


def _bind_trusted_toolchain(env: dict[str, str], *, rustup_home: Path | None = None) -> None:
    """Bind rustup to the setup action's installed, exact toolchain.

    The build uses a fresh Cargo home, but rustup's installed toolchains live
    separately.  Preserve only that toolchain store and force the version;
    never inherit a caller-selected toolchain or wrapper/configuration.
    """
    configured = os.environ.get("RUSTUP_HOME") if rustup_home is None else str(rustup_home)
    if configured:
        rustup_home = Path(configured)
    else:
        host_home = os.environ.get("HOME")
        rustup_home = (Path(host_home) if host_home else Path.home()) / ".rustup"
    if not rustup_home.is_dir() or rustup_home.is_symlink():
        raise CompatibilityError("installed Rust toolchain store is unavailable")
    env["RUSTUP_HOME"] = str(rustup_home)
    env["RUSTUP_TOOLCHAIN"] = "1.99.0"


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        # Some non-Linux hosts reject killpg after the leader has exited.  The
        # exact owned leader is still safe to terminate; hosted Linux must use
        # the process-group path above and therefore cannot silently fall back.
        if sys.platform.startswith("linux"):
            raise CompatibilityError("cannot signal the owned Linux child process group")
        try:
            process.terminate()
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            if sys.platform.startswith("linux"):
                raise CompatibilityError("cannot inspect the owned Linux child process group") from exc
            if process.poll() is not None:
                return
            try:
                process.terminate()
            except ProcessLookupError:
                return
            return
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            continue
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reap_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired as final_exc:
            raise CompatibilityError("child process could not be reaped") from final_exc


def _run_bounded(argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, max_output: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if not argv or any(not isinstance(item, str) for item in argv):
        raise CompatibilityError("invalid subprocess argv")
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    timed_out = False
    interrupted = False
    previous_handlers = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))

    def interrupt(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        _kill_process_group(process)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while selector.get_map() or process.poll() is None:
            remaining = _remaining(deadline)
            for key, _ in selector.select(min(0.25, remaining)):
                data = key.fileobj.read1(65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[key.data]
                remaining_capacity = max_output - len(target)
                if len(data) > remaining_capacity:
                    target.extend(data[: max(0, remaining_capacity)])
                    truncated = True
                    _kill_process_group(process)
                    break
                target.extend(data)
            if process.poll() is not None and not selector.get_map():
                break
    except CompatibilityError:
        timed_out = True
        _kill_process_group(process)
    finally:
        selector.close()
        _kill_process_group(process)
        _reap_process(process)
        signal.signal(signal.SIGINT, previous_handlers[0])
        signal.signal(signal.SIGTERM, previous_handlers[1])
    timed_out = timed_out or interrupted
    process.stdout.close()
    process.stderr.close()
    result = {
        "argv": argv,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "timed_out": timed_out,
        "truncated": truncated,
        "stdout_sha256": hashlib.sha256(buffers["stdout"]).hexdigest(),
        "stderr_sha256": hashlib.sha256(buffers["stderr"]).hexdigest(),
        "stdout_bytes": bytes(buffers["stdout"]),
        "stderr_bytes": bytes(buffers["stderr"]),
        "stdout": bytes(buffers["stdout"]).decode("utf-8", "replace"),
        "stderr": bytes(buffers["stderr"]).decode("utf-8", "replace"),
    }
    if timed_out or truncated or process.returncode != 0:
        raise BoundedCommandError(f"bounded command failed: {argv[0]} rc={process.returncode}", result)
    return result


def _command_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in ("argv", "returncode", "elapsed_seconds", "timed_out", "truncated", "stdout_sha256", "stderr_sha256")
    }


def _write_stream_evidence(directory: Path, stage: str, result: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stream in ("stdout", "stderr"):
        data = result.get(f"{stream}_bytes", b"")
        if not isinstance(data, bytes) or len(data) > 2 * 1024 * 1024:
            raise CompatibilityError(f"{stage} {stream} evidence is unavailable or oversized")
        path = directory / f"{stage}.{stream}"
        path.write_bytes(data)
        if _sha256(path) != result[f"{stream}_sha256"]:
            raise CompatibilityError(f"{stage} {stream} evidence hash mismatch")


def _evidence_total(directory: Path) -> int:
    total = 0
    for path in directory.glob("*"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise CompatibilityError("compatibility evidence member is invalid or oversized")
        total += path.stat().st_size
    if total > 16 * 1024 * 1024:
        raise CompatibilityError("compatibility evidence exceeds aggregate bound")
    return total


def _run_stage(stage: str, *, receipt: dict[str, Any], evidence_dir: Path, argv: list[str], cwd: Path, env: dict[str, str], deadline: float, max_output: int = 2 * 1024 * 1024) -> dict[str, Any]:
    try:
        result = _run_bounded(argv, cwd=cwd, env=env, deadline=deadline, max_output=max_output)
    except BoundedCommandError as exc:
        exc.result["stage"] = stage
        _write_stream_evidence(evidence_dir, stage, exc.result)
        receipt["commands"].append({"stage": stage, **_command_summary(exc.result)})
        raise
    result["stage"] = stage
    _write_stream_evidence(evidence_dir, stage, result)
    receipt["commands"].append({"stage": stage, **_command_summary(result)})
    return result


def _cargo_metadata(value: dict[str, Any]) -> dict[str, Any]:
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise CompatibilityError("metadata evidence missing metadata object")
    return metadata


def _canonical_fingerprint(metadata: dict[str, Any]) -> str:
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(packages, list) or not isinstance(resolve, dict):
        raise CompatibilityError("Cargo metadata package/resolve shape is invalid")
    identities: dict[str, tuple[str, str]] = {}
    for package in packages:
        if not isinstance(package, dict) or not all(isinstance(package.get(key), str) for key in ("id", "name", "version")):
            raise CompatibilityError("Cargo metadata has an invalid package identity")
        if package["id"] in identities:
            raise CompatibilityError("Cargo metadata has duplicate package IDs")
        identities[package["id"]] = (package["name"], package["version"])
    rows = []
    for node in resolve["nodes"]:
        if not isinstance(node, dict) or node.get("id") not in identities:
            raise CompatibilityError("Cargo metadata resolve node is not a known package")
        name, version = identities[node["id"]]
        deps = []
        for dependency in node.get("deps", []):
            if not isinstance(dependency, dict) or dependency.get("pkg") not in identities:
                raise CompatibilityError("Cargo metadata has an unresolved typed dependency")
            package_name, package_version = identities[dependency["pkg"]]
            dep_kinds = dependency.get("dep_kinds")
            if not isinstance(dep_kinds, list):
                raise CompatibilityError("Cargo metadata dependency kinds are malformed")
            deps.append({
                "name": dependency.get("name"),
                "package_name": package_name,
                "package_version": package_version,
                "dep_kinds": dep_kinds,
            })
        deps.sort(key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
        rows.append({"name": name, "version": version, "features": sorted(node.get("features", [])), "deps": deps})
    rows.sort(key=lambda item: (item["name"], item["version"]))
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_cpu_metadata(path: Path, *, expected_fingerprint: str | None = CPU_FINGERPRINT, role_counts: tuple[int, int, int] = (90, 14, 1)) -> dict[str, Any]:
    evidence = _json(path)
    metadata = _cargo_metadata(evidence)
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(packages, list) or len(packages) != 105:
        raise CompatibilityError("CPU metadata must contain exactly 105 packages")
    if not isinstance(resolve, dict) or not isinstance(resolve.get("nodes"), list) or len(resolve["nodes"]) != 105:
        raise CompatibilityError("CPU metadata resolve graph must contain 105 nodes")
    package_ids = {
        item.get("id") for item in packages
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    node_ids = {
        node.get("id") for node in resolve["nodes"]
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    identities = {
        (item.get("name"), item.get("version"), item.get("source"))
        for item in packages if isinstance(item, dict)
    }
    if len(package_ids) != 105 or package_ids != node_ids or len(identities) != 105:
        raise CompatibilityError("CPU metadata package identities or node IDs are not unique")
    if any(any(token in str(item.get("name", "")).lower() for token in GPU_TOKENS) for item in packages if isinstance(item, dict)):
        raise CompatibilityError("CPU metadata contains a GPU package")
    registry_count = sum(
        1 for item in packages
        if isinstance(item, dict) and str(item.get("source", "")).startswith("registry+")
    )
    path_count = sum(1 for item in packages if isinstance(item, dict) and item.get("source") is None and item.get("name") != "moshi")
    moshi_count = sum(1 for item in packages if isinstance(item, dict) and item.get("source") is None and item.get("name") == "moshi")
    if (registry_count, path_count, moshi_count) != role_counts:
        raise CompatibilityError("CPU metadata source-role counts changed")
    if any(not isinstance(item.get("targets"), list) or not item["targets"] for item in packages if isinstance(item, dict)):
        raise CompatibilityError("CPU metadata has a package without targets")
    for item in packages:
        if not isinstance(item.get("manifest_path"), str) or not item["manifest_path"].endswith("Cargo.toml"):
            raise CompatibilityError("CPU metadata package source path is malformed")
        name_version = f"{item.get('name')}-{item.get('version')}"
        manifest_path = item["manifest_path"]
        source = item.get("source")
        if isinstance(source, str) and source.startswith("registry+"):
            if f"/registry/src/" not in manifest_path or not manifest_path.endswith(f"/{name_version}/Cargo.toml"):
                raise CompatibilityError("registry package source path is not bound to its identity")
        elif item.get("name") == "moshi" and source is None:
            if not manifest_path.endswith("/projection/moshi-core/Cargo.toml"):
                raise CompatibilityError("Moshi source path is not the authenticated projection")
        elif source is None:
            if not manifest_path.endswith(f"/vendor/{name_version}/Cargo.toml"):
                raise CompatibilityError("derived package source path is not the authenticated vendor identity")
        else:
            raise CompatibilityError("CPU package has an unsupported source role")
        for target in item["targets"]:
            if not isinstance(target, dict) or not isinstance(target.get("src_path"), str) or not target["src_path"].endswith(".rs"):
                raise CompatibilityError("CPU metadata target source path is malformed")
            if not target["src_path"].startswith(manifest_path.rsplit("/Cargo.toml", 1)[0] + "/"):
                raise CompatibilityError("CPU target source path escapes its package")
    for package_name, expected in ROOT_FEATURES.items():
        package_ids = {
            item.get("id") for item in packages
            if isinstance(item, dict) and item.get("name") == package_name
        }
        matching = [
            node for node in resolve["nodes"]
            if isinstance(node, dict) and node.get("id") in package_ids
        ]
        if len(matching) != 1 or matching[0].get("features") != expected:
            raise CompatibilityError(f"CPU root feature mismatch: {package_name}")
    fingerprint = _canonical_fingerprint(metadata)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise CompatibilityError(f"CPU canonical fingerprint mismatch: {fingerprint}")
    return {"packages": len(packages), "nodes": len(resolve["nodes"]), "fingerprint": fingerprint}


def _find_moshi_rlib(target_dir: Path) -> Path:
    candidates = sorted((target_dir / TARGET / "debug" / "deps").glob("libmoshi-*.rlib"))
    if len(candidates) != 1:
        raise CompatibilityError(f"cargo build produced ambiguous moshi rlibs: {len(candidates)}")
    return candidates[-1]


def _validate_host_dependency_dir(target_dir: Path) -> Path:
    host_deps = target_dir / "debug" / "deps"
    if not host_deps.is_dir() or host_deps.is_symlink():
        raise CompatibilityError("fresh host dependency directory is missing or non-regular")
    artifacts = [
        path for path in host_deps.iterdir()
        if path.name.endswith((".rlib", ".so", ".dylib"))
    ]
    if not artifacts or any(path.is_symlink() or not path.is_file() for path in artifacts):
        raise CompatibilityError("fresh host dependency artifacts are missing or unsafe")
    return host_deps


def _artifact_ids(result: dict[str, Any], *, package_identities: dict[str, str] | None = None) -> list[str]:
    ids: list[str] = []
    observed: dict[str, set[str]] = {}
    for line in result.get("stdout", "").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("reason") != "compiler-artifact" or not isinstance(record.get("package_id"), str):
            continue
        package_id = record["package_id"]
        if package_identities is not None:
            identity = package_identities.get(package_id)
            if identity is None:
                raise CompatibilityError(f"compiler artifact package ID is absent from authenticated metadata: {package_id}")
        else:
            package_tail = package_id.rsplit("#", 1)[-1]
            if "@" not in package_tail:
                continue
            name, version = package_tail.rsplit("@", 1)
            identity = f"{name}@{version}"
        target = record.get("target")
        kinds = target.get("kind") if isinstance(target, dict) else None
        if not isinstance(kinds, list):
            continue
        observed.setdefault(identity, set()).update(str(kind) for kind in kinds)
        ids.append(record["package_id"])
    missing = [needle for needle in REQUIRED_BUILD_ARTIFACTS if not (needle in observed and set(REQUIRED_BUILD_KINDS[needle]).issubset(observed[needle]))]
    if missing:
        raise CompatibilityError(f"required compiler artifacts missing: {missing}")
    return ids


def _metadata_package_identities(path: Path) -> dict[str, str]:
    metadata = _cargo_metadata(_json(path))
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise CompatibilityError("build metadata packages are malformed")
    identities: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict) or not all(isinstance(package.get(key), str) for key in ("id", "name", "version")):
            raise CompatibilityError("build metadata package identity is malformed")
        identity = f"{package['name']}@{package['version']}"
        if package["id"] in identities or identity in identities.values():
            raise CompatibilityError("build metadata package identity is ambiguous")
        identities[package["id"]] = identity
    return identities


def _run_api_probe(*, root: Path, target_dir: Path, deadline: float, env: dict[str, str], receipt: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    rlib = _find_moshi_rlib(target_dir)
    host_deps = _validate_host_dependency_dir(target_dir)
    probe = root / "compatibility-api-probe.rs"
    probe.write_text(
        """extern crate moshi;
use moshi::{candle, conditioner, StreamMask, StreamTensor};
use moshi::asr::{self, State};
use moshi::mimi::Mimi;

fn signatures() {
    let _encode: fn(&mut Mimi, &StreamTensor, &StreamMask) -> candle::Result<StreamTensor> = Mimi::encode_step;
    let _decode: fn(&mut Mimi, &StreamTensor, &StreamMask) -> candle::Result<StreamTensor> = Mimi::decode_step;
    let _idx: fn(&State) -> usize = State::model_step_idx;
    fn step(s: &mut State, pcm: candle::Tensor, c: Option<&conditioner::Condition>, mask: &StreamMask) -> candle::Result<Vec<asr::AsrMsg>> {
        s.step_pcm(pcm, c, mask, |_items, _tensor, _tensors| {})
    }
    let _ = step;
}
""",
        encoding="utf-8",
    )
    output = root / "compatibility-api-probe.rlib"
    command = NAMESPACE_PREFIX + ["rustc", "--crate-type", "lib", "--edition", "2021", "--target", TARGET, str(probe), "--extern", f"moshi={rlib}", "-L", f"dependency={rlib.parent}", "-L", f"dependency={host_deps}", "-o", str(output)]
    result = _run_stage("api_probe", receipt=receipt, evidence_dir=evidence_dir, argv=command, cwd=root, env=env, deadline=deadline, max_output=512 * 1024)
    if not output.is_file() or output.stat().st_size > 16 * 1024 * 1024:
        raise CompatibilityError("API probe output is missing or oversized")
    return result


def check(application_root: Path, output: Path, deadline_seconds: int) -> int:
    started = time.monotonic()
    deadline = started + deadline_seconds
    receipt: dict[str, Any] = {
        "status": "CPU_COMPATIBILITY_REVIEW_PENDING",
        "source_only": True,
        "approval": None,
        "model_execution": "NOT_AUTHORIZED",
        "upload": "NO_UPLOAD",
        "target": TARGET,
        "application": {"run": APPLIED_RUN, "attempt": APPLIED_ATTEMPT, "head": APPLIED_HEAD, "artifact": APPLIED_ARTIFACT},
        "commands": [],
    }
    evidence_dir = output.parent / "evidence"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not isinstance(deadline_seconds, int) or deadline_seconds <= 0 or deadline_seconds > MAX_DEADLINE:
            raise CompatibilityError(f"deadline_seconds must be in 1..{MAX_DEADLINE}")
        if not sys.platform.startswith("linux") or os.environ.get("VOKRA_CANDLE_REMOTE_LINUX") != "1":
            raise CompatibilityError("CPU compatibility requires the explicit Linux hosted marker")
        workflow_head = os.environ.get("VOKRA_CANDLE_WORKFLOW_HEAD")
        if not workflow_head or len(workflow_head) != 40 or any(char not in "0123456789abcdef" for char in workflow_head.lower()):
            raise CompatibilityError("workflow checkout head is malformed")
        receipt["workflow_checkout_head"] = workflow_head
        repo = Path(__file__).resolve().parents[3]
        application_receipt = _verify_application(application_root)
        receipt["application_proofs"] = {
            "source_files_pre": application_receipt["source_proof_pre"]["files_checked"],
            "source_files_post": application_receipt["source_proof_post"]["files_checked"],
            "unchanged_files_pre": application_receipt["source_proof_pre"]["unchanged_files"],
            "unchanged_files_post": application_receipt["source_proof_post"]["unchanged_files"],
            "member_checks_pre": sum(item.get("members_checked", 0) for item in application_receipt["member_premetadata_checks"]),
            "member_checks_post": sum(item.get("members_checked", 0) for item in application_receipt["member_postchecks"]),
            "non_manifest_pre": sum(item.get("non_manifest_members", 0) for item in application_receipt["member_premetadata_checks"]),
            "non_manifest_post": sum(item.get("non_manifest_members", 0) for item in application_receipt["member_postchecks"]),
        }
        _verify_fixed_files(repo)
        if _sha256(application_root / "final.lock") != FINAL_LOCK_SHA256:
            raise CompatibilityError("authenticated application final.lock mismatch")
        with tempfile.TemporaryDirectory(prefix="vokra-candle-cpu-") as temporary:
            root = Path(temporary)
            git_env = _sanitized_env(cargo_home=root / "git-cargo-home", target_dir=root / "git-target", offline=True, home=root / "git-home")
            checkout_result = _run_stage(
                "checkout_head",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=["git", "rev-parse", "HEAD"],
                cwd=repo,
                env=git_env,
                deadline=deadline,
                max_output=128,
            )
            actual_head = checkout_result["stdout"].strip()
            if actual_head != workflow_head:
                raise CompatibilityError("workflow checkout head does not match the checked-out HEAD")
            receipt["checked_out_head"] = actual_head
            packet = root / "packet"
            applied_output = root / "application-output"
            workspace = root / "workspace"
            applier_budget = min(480, max(1, int(_remaining(deadline)) - 2))
            applier = repo / "tools/parity/kyutai_candle_reference/apply_paste_overlay.py"
            command = [sys.executable, str(applier), str(packet), str(applied_output), str(workspace), "--deadline-seconds", str(applier_budget)]
            receipt["commands"].append({"stage": "source_application", "argv": command})
            application_home = root / "application-home"
            application_home.mkdir()
            application_env = _sanitized_env(cargo_home=root / "cargo-home", target_dir=root / "target", offline=False, home=application_home)
            _bind_trusted_toolchain(application_env)
            application_env["VOKRA_CANDLE_REMOTE_LINUX"] = "1"
            _run_stage("source_application", receipt=receipt, evidence_dir=evidence_dir, argv=command, cwd=repo, env=application_env, deadline=deadline)
            replay_source_receipt = applied_output / "source_receipt.json"
            replay_final_lock = applied_output / "final.lock"
            if hashlib.sha256(_bounded_read(replay_source_receipt)).hexdigest() != SOURCE_RECEIPT_SHA256:
                raise CompatibilityError("replayed source receipt changed")
            source_receipt = _json(replay_source_receipt)
            if not isinstance(source_receipt.get("files"), dict) or len(source_receipt["files"]) != 23:
                raise CompatibilityError("replayed source receipt file manifest is incomplete")
            replay_receipt_path = applied_output / "application_receipt.json"
            if not replay_receipt_path.is_file():
                raise CompatibilityError("replayed application receipt is missing")
            replay_receipt = _verify_application_receipt(_json(replay_receipt_path))
            receipt["replayed_application_proofs"] = {
                "source_pre": replay_receipt["source_proof_pre"]["files_checked"],
                "source_post": replay_receipt["source_proof_post"]["files_checked"],
                "members_pre": sum(item.get("members_checked", 0) for item in replay_receipt["member_premetadata_checks"]),
                "members_post": sum(item.get("members_checked", 0) for item in replay_receipt["member_postchecks"]),
            }
            if hashlib.sha256(_bounded_read(replay_final_lock)).hexdigest() != FINAL_LOCK_SHA256:
                raise CompatibilityError("replayed final lock changed")
            if _sha256(workspace / "projection" / "Cargo.lock") != FINAL_LOCK_SHA256:
                raise CompatibilityError("fresh applier final Cargo.lock mismatch")
            baseline_cpu_summary = _validate_cpu_metadata(applied_output / "baseline_cpu_metadata.json", expected_fingerprint=None, role_counts=(104, 0, 1))
            cpu_summary = _validate_cpu_metadata(applied_output / "overlay_cpu_metadata.json")
            receipt["cpu_metadata"] = {"baseline": baseline_cpu_summary, "overlay": cpu_summary}
            manifest = workspace / "projection" / "moshi-core" / "Cargo.toml"
            if _sha256(workspace / "projection" / "Cargo.toml") != ROOT_MANIFEST_SHA256:
                raise CompatibilityError("fresh projection root manifest mismatch")
            if not manifest.is_file():
                raise CompatibilityError("fresh moshi-core manifest missing")
            cargo_home = workspace / "cargo-home"
            source_snapshot = {
                "projection": _tree_fingerprint(workspace / "projection", deadline=deadline),
                "vendor": _tree_fingerprint(workspace / "vendor", deadline=deadline),
            }
            source_snapshot.update(_selected_source_scope(applied_output / "overlay_cpu_metadata.json", cargo_home, deadline=deadline))
            receipt["source_snapshot_before_build"] = source_snapshot
            target_dir = root / "target-build"
            build_home = root / "build-home"
            build_home.mkdir()
            build_env = _sanitized_env(cargo_home=cargo_home, target_dir=target_dir, offline=True, home=build_home)
            _bind_trusted_toolchain(build_env)
            build_env["HOME"] = str(build_home)
            parent_net_ns = os.readlink("/proc/self/ns/net")
            build_env["VOKRA_PARENT_NET_NS"] = parent_net_ns
            preflight_result = _run_stage("namespace_preflight", receipt=receipt, evidence_dir=evidence_dir, argv=NAMESPACE_PREFIX + ["/bin/sh", "-c", "test \"$(readlink /proc/self/ns/net)\" != \"$VOKRA_PARENT_NET_NS\" && test \"$(awk '/NoNewPrivs/ {print $2}' /proc/self/status)\" = 1 && rustc --version && cargo --version"], cwd=root, env=build_env, deadline=deadline, max_output=64 * 1024)
            if "rustc 1.99.0" not in preflight_result["stdout"] or "cargo 1.99.0" not in preflight_result["stdout"]:
                raise CompatibilityError("namespace preflight did not select Rust/Cargo 1.99.0")
            receipt["isolation"] = {"network_namespace": "unshare --net verified", "no_new_privs": True, "runner_uid_preserved": True}
            build_command = NAMESPACE_PREFIX + ["cargo", "build", "--lib", "--message-format=json-render-diagnostics", "--locked", "--offline", "--no-default-features", "--target", TARGET, "--manifest-path", str(manifest)]
            build_result = _run_stage("cargo_build", receipt=receipt, evidence_dir=evidence_dir, argv=build_command, cwd=workspace / "projection", env=build_env, deadline=deadline, max_output=2 * 1024 * 1024)
            receipt["compiler_artifacts"] = _artifact_ids(
                build_result,
                package_identities=_metadata_package_identities(applied_output / "overlay_cpu_metadata.json"),
            )
            built_lock_sha256 = _sha256(workspace / "projection" / "Cargo.lock")
            if built_lock_sha256 != FINAL_LOCK_SHA256:
                raise CompatibilityError("Cargo.lock changed during offline build")
            receipt["post_build_final_lock_sha256"] = built_lock_sha256
            source_after = {
                "projection": _tree_fingerprint(workspace / "projection", deadline=deadline),
                "vendor": _tree_fingerprint(workspace / "vendor", deadline=deadline),
            }
            source_after.update(_selected_source_scope(applied_output / "overlay_cpu_metadata.json", cargo_home, deadline=deadline))
            receipt["source_snapshot_after_build"] = source_after
            if source_after != source_snapshot:
                raise CompatibilityError("source or vendor tree changed during build")
            post_metadata_command = NAMESPACE_PREFIX + ["cargo", "metadata", "--locked", "--offline", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features", "--manifest-path", str(manifest)]
            post_metadata_result = _run_stage("post_build_cpu_metadata", receipt=receipt, evidence_dir=evidence_dir, argv=post_metadata_command, cwd=workspace / "projection", env=build_env, deadline=deadline, max_output=2 * 1024 * 1024)
            try:
                post_metadata = json.loads(post_metadata_result["stdout"])
            except json.JSONDecodeError as exc:
                raise CompatibilityError("post-build CPU metadata is not JSON") from exc
            post_metadata_path = root / "post-build-cpu-metadata.json"
            post_metadata_path.write_text(json.dumps({"metadata": post_metadata}), encoding="utf-8")
            receipt["post_build_cpu_metadata"] = _validate_cpu_metadata(post_metadata_path)
            probe_result = _run_api_probe(root=root, target_dir=target_dir, deadline=deadline, env=build_env, receipt=receipt, evidence_dir=evidence_dir)
            if _sha256(workspace / "projection" / "Cargo.lock") != FINAL_LOCK_SHA256:
                raise CompatibilityError("Cargo.lock changed during API probe")
            if {
                "projection": _tree_fingerprint(workspace / "projection", deadline=deadline),
                "vendor": _tree_fingerprint(workspace / "vendor", deadline=deadline),
                **_selected_source_scope(applied_output / "overlay_cpu_metadata.json", cargo_home, deadline=deadline),
            } != source_snapshot:
                raise CompatibilityError("source or vendor tree changed during API probe")
            receipt["build"] = {"target": TARGET, "offline": True, "lib": True, "no_default_features": True}
            receipt["status"] = "CPU_COMPATIBILITY_REVIEW_PENDING"
            receipt["evidence_bytes"] = _evidence_total(evidence_dir)
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _remaining(deadline)
            _write_json(output, receipt)
            return 0
    except Exception as exc:  # bounded diagnostic must preserve truthful failure evidence
        receipt["status"] = "CPU_COMPATIBILITY_FAILED"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if isinstance(exc, BoundedCommandError):
            failed = dict(exc.result)
            failed.pop("stdout_bytes", None)
            failed.pop("stderr_bytes", None)
            failed["stdout"] = failed.get("stdout", "")[-32768:]
            failed["stderr"] = failed.get("stderr", "")[-32768:]
            receipt["failed_command"] = failed
        if evidence_dir.is_dir():
            try:
                receipt["evidence_bytes"] = _evidence_total(evidence_dir)
            except CompatibilityError as evidence_error:
                receipt["evidence_bound_error"] = str(evidence_error)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            _write_json(output, receipt)
        except Exception:
            pass
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or args.deadline_seconds > MAX_DEADLINE:
        parser.error(f"--deadline-seconds must be in 1..{MAX_DEADLINE}")
    return check(args.application_root, args.output, args.deadline_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
