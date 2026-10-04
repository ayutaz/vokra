#!/usr/bin/env python3
"""VAST-only, fail-closed Cargo metadata binding for Moshi e6.

This is a source/dependency evidence collector. It never imports or loads
Moshi, resolves dependencies, fetches packages, accesses weights, or claims
native, licence, owner, model, or parity approval. Source identity and the
derived workspace are collected by the accepted
``mimi_rust_core_source_audit`` collector; this file only binds an externally
provided, exact ``cargo metadata --locked --offline`` result to that evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import selectors
import shutil
import stat
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, Callable

PINNED_REV = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
EXPECTED_LOCK_SHA256 = "bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab"
TARGET = "x86_64-unknown-linux-gnu"
CORE_PACKAGE = "moshi"
CORE_VERSION = "0.6.4"
MAX_READ = 64 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
MAX_PACKET = 64 * 1024 * 1024
METADATA_ARGS = ("metadata", "--locked", "--offline", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features")


class BindingError(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_rel(raw: str) -> str:
    """Expose the accepted auditor's canonical path validator to callers."""
    try:
        return SOURCE_AUDITOR._safe_rel(raw)
    except SOURCE_AUDITOR.AuditError as exc:
        raise BindingError(str(exc)) from exc


def validate_ancestors(path: Path) -> None:
    """Expose the accepted auditor's no-follow ancestor check for tests."""
    try:
        SOURCE_AUDITOR._validate_path_ancestors(path)
    except SOURCE_AUDITOR.AuditError as exc:
        raise BindingError(str(exc)) from exc


def _auditor() -> Any:
    path = Path(__file__).with_name("mimi_rust_core_source_audit.py")
    spec = importlib.util.spec_from_file_location("vokra_mimi_rust_core_source_audit", path)
    if spec is None or spec.loader is None:
        raise BindingError("accepted source auditor is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE_AUDITOR = _auditor()


def _safe_read(path: Path, limit: int = MAX_READ) -> bytes:
    return SOURCE_AUDITOR._bounded_bytes(path, limit)


def _stream_hash(path: Path, limit: int = 256 * 1024 * 1024) -> tuple[str, int]:
    """Hash an authenticated regular binary without allocating its contents."""
    SOURCE_AUDITOR._validate_path_ancestors(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise BindingError(f"tool is not a bounded regular file: {path}")
        digest = hashlib.sha256(); total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise BindingError(f"tool grew beyond hash bound: {path}")
            digest.update(chunk)
        after = os.fstat(fd)
        ident = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        named = os.stat(path, follow_symlinks=False)
        if ident(before) != ident(after) or ident(before) != ident(named):
            raise BindingError(f"tool changed while hashed: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(fd)


def command_bytes(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, limit: int = MAX_OUTPUT, timeout: float = 30.0) -> tuple[int, bytes, bytes]:
    """Run one bounded process and always close both pipes and the child."""
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector: selectors.BaseSelector | None = None
    deadline = time.monotonic() + timeout
    try:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill(); process.wait()
                raise BindingError(f"command timeout: {' '.join(argv)}")
            for key, _ in selector.select(min(0.1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj); key.fileobj.close(); continue
                buffers[key.data].extend(chunk)
                if len(buffers[key.data]) > limit:
                    process.kill(); process.wait()
                    raise BindingError(f"command output exceeds bound: {' '.join(argv)}")
        return process.wait(timeout=max(0.1, deadline - time.monotonic())), bytes(buffers["stdout"]), bytes(buffers["stderr"])
    finally:
        if selector is not None:
            selector.close()
        if process.poll() is None:
            process.kill(); process.wait()
        for stream in (process.stdout, process.stderr):
            try: stream.close()
            except OSError: pass


def _exclusive(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise BindingError(f"refusing to overwrite packet file: {path}")
    SOURCE_AUDITOR._validate_output_parent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_AUDITOR._validate_output_parent(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        done = 0
        while done < len(data):
            written = os.write(fd, data[done:])
            if written <= 0: raise BindingError(f"short packet write: {path}")
            done += written
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if len(payload) > MAX_PACKET: raise BindingError(f"packet member exceeds bound: {path}")
    _exclusive(path, payload)


def _tool_row(name: str) -> dict[str, Any]:
    launcher = shutil.which(name); rustup = shutil.which("rustup")
    if not launcher or not rustup: raise BindingError(f"unbound {name}/rustup toolchain")
    rustup = str(Path(rustup).resolve())
    code, out, err = command_bytes([rustup, "which", name], limit=64 * 1024, timeout=10)
    if code != 0 or not out.strip() or err: raise BindingError(f"rustup which {name} failed")
    resolved = Path(out.decode().strip())
    if not resolved.is_absolute() or resolved.is_symlink() or not resolved.is_file():
        raise BindingError(f"rustup did not bind an absolute regular {name}")
    launcher_resolved = Path(launcher).resolve()
    launcher_digest, launcher_size = _stream_hash(launcher_resolved)
    rustup_resolved = Path(rustup).resolve()
    rustup_digest, rustup_size = _stream_hash(rustup_resolved)
    digest, size = _stream_hash(resolved)
    code, version, version_err = command_bytes([str(resolved), "--version"], limit=64 * 1024, timeout=10)
    if code != 0 or not version.strip() or version_err: raise BindingError(f"authenticated {name} --version failed")
    return {"launcher": str(Path(launcher).absolute()), "launcher_resolved": str(launcher_resolved), "launcher_sha256": launcher_digest, "launcher_bytes": launcher_size, "rustup": rustup, "rustup_resolved": str(rustup_resolved), "rustup_sha256": rustup_digest, "rustup_bytes": rustup_size, "rustup_which": str(resolved), "resolved": str(resolved), "sha256": digest, "bytes": size, "version": version.decode(errors="replace").strip()}


def tool_identity() -> dict[str, Any]:
    return {name: _tool_row(name) for name in ("cargo", "rustc")}


def _source_readback(source_root: Path, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    facts = SOURCE_AUDITOR._git_facts(source_root)
    lock_bytes = _safe_read(source_root / "rust/Cargo.lock")
    selected: dict[str, str] = {}
    for row in rows or []:
        rel = row.get("path") if isinstance(row, dict) else None
        if isinstance(rel, str):
            selected[rel] = sha256(_safe_read(source_root / rel))
    return {"facts": facts, "lock_sha256": sha256(lock_bytes), "lock_bytes": len(lock_bytes), "selected_sha256": selected}


def _inventory_payload_sha(value: dict[str, Any]) -> str:
    without_digest = {key: item for key, item in value.items() if key != "inventory_payload_sha256"}
    encoded = (json.dumps(without_digest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    return sha256(encoded)


def _derived_readback(derived: Path, workspace: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = _safe_read(derived / "Cargo.toml")
    lock = _safe_read(derived / "Cargo.lock")
    copied: dict[str, str] = {}
    for source_rel in (workspace or {}).get("copied_source_paths", []):
        if source_rel == "rust/moshi-core/Cargo.toml":
            target_rel = "moshi-core/Cargo.toml"
        elif source_rel == "rust/README.md":
            target_rel = "README.md"
        elif source_rel.startswith("rust/moshi-core/src/"):
            target_rel = source_rel.removeprefix("rust/")
        else:
            raise BindingError(f"unexpected copied source path: {source_rel}")
        copied[source_rel] = sha256(_safe_read(derived / target_rel))
    return {"manifest_sha256": sha256(manifest), "lock_sha256": sha256(lock), "lock_bytes": len(lock), "copied_source_sha256": copied}


def _verify_source_snapshot(source: dict[str, Any], readback: dict[str, Any]) -> None:
    inventory_source = source.get("source")
    inventory_lock = source.get("lock")
    if not isinstance(inventory_source, dict) or not isinstance(inventory_lock, dict):
        raise BindingError("source auditor inventory lacks source/lock identity")
    expected_facts = inventory_source.get("before")
    if not isinstance(expected_facts, dict) or readback["facts"] != expected_facts:
        raise BindingError("source facts do not match authenticated inventory")
    if inventory_lock.get("sha256") != EXPECTED_LOCK_SHA256 or readback["lock_sha256"] != EXPECTED_LOCK_SHA256:
        raise BindingError("source lock does not match authenticated pinned lock")
    expected_rows = {row.get("path"): row.get("sha256") for row in inventory_source.get("rows", []) if isinstance(row, dict)}
    if readback["selected_sha256"] != expected_rows:
        raise BindingError("selected source file hashes do not match authenticated inventory")


def _verify_derived_snapshot(source: dict[str, Any], readback: dict[str, Any]) -> None:
    workspace = source.get("workspace")
    if not isinstance(workspace, dict): raise BindingError("source auditor inventory lacks derived workspace")
    if readback["manifest_sha256"] != workspace.get("derived_manifest_sha256"):
        raise BindingError("derived Cargo.toml differs from authenticated workspace")
    copied_lock = workspace.get("copied_lock")
    if not isinstance(copied_lock, dict) or readback["lock_sha256"] != copied_lock.get("sha256"):
        raise BindingError("derived Cargo.lock differs from authenticated source lock")
    if readback["copied_source_sha256"] != workspace.get("copied_source_sha256", {}):
        raise BindingError("derived copied source differs from authenticated inventory")


def _lock_rows(lock_data: dict[str, Any]) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    rows = lock_data.get("package")
    if not isinstance(rows, list): raise BindingError("Cargo.lock package table missing")
    result: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str): raise BindingError("malformed Cargo.lock package row")
        source = row.get("source"); checksum = row.get("checksum")
        if source is not None and not isinstance(source, str): raise BindingError("malformed Cargo.lock source")
        if checksum is not None and (not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)): raise BindingError("malformed Cargo.lock checksum")
        key = (row["name"], row["version"], source)
        if key in result: raise BindingError(f"duplicate Cargo.lock package: {key}")
        result[key] = row
    return result


def inspect_metadata(metadata: dict[str, Any], lock_data: dict[str, Any], *, authenticated_target: str | None = None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise BindingError("metadata root must be an object")
    if authenticated_target is not None and authenticated_target != TARGET:
        raise BindingError("authenticated target is not x86_64-unknown-linux-gnu")
    # The accepted auditor owns package/lock/root/node/source/checksum
    # validation. This binder adds only the raw dep_kinds evidence that the
    # source-only inventory intentionally does not retain.
    try:
        base = SOURCE_AUDITOR._metadata_inventory(metadata, lock_data, CORE_VERSION)
    except SOURCE_AUDITOR.AuditError as exc:
        raise BindingError(str(exc)) from exc
    nodes = metadata["resolve"].get("nodes")
    if not isinstance(nodes, list):
        raise BindingError("metadata resolve.nodes missing")
    detailed: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        node_id = node.get("id") if isinstance(node, dict) else None
        deps = node.get("deps", node.get("dependencies", [])) if isinstance(node, dict) else None
        if not isinstance(node_id, str) or not isinstance(deps, list):
            raise BindingError("dependency list malformed")
        rows: list[dict[str, Any]] = []
        for dep in deps:
            if not isinstance(dep, dict) or not isinstance(dep.get("pkg"), str):
                raise BindingError("dangling dependency edge")
            kinds = dep.get("dep_kinds", [])
            if not isinstance(kinds, list) or not all(isinstance(item, dict) for item in kinds):
                raise BindingError("dependency kinds malformed")
            rows.append({"pkg": dep["pkg"], "dep_kinds": kinds})
        detailed[node_id] = rows
    active = []
    for row in base["active_packages"]:
        enriched = dict(row)
        enriched["dependencies"] = detailed.get(row["id"], [])
        active.append(enriched)
    blocked = base.get("blocked_markers", [])
    return {"status": "BLOCKED_NATIVE_MARKER" if blocked else "OPEN_TARGET_DEP_KINDS_BINDING", "root": base["workspace_root"], "active_reachable": active, "active_count": base["active_package_count"], "all_count": base["all_package_count"], "blocked_markers": blocked, "authenticated_target": authenticated_target or "OPEN_TARGET_NOT_BOUND", "activation": base["activation_status"]}


def _load_source_inventory(audit_dir: Path) -> dict[str, Any]:
    data = json.loads(_safe_read(audit_dir / "source-inventory.json", MAX_PACKET))
    if not isinstance(data, dict) or data.get("status") != "STATIC_SOURCE_INVENTORY": raise BindingError("accepted source auditor did not produce STATIC_SOURCE_INVENTORY")
    if data.get("inventory_payload_sha256") != _inventory_payload_sha(data):
        raise BindingError("source inventory payload digest mismatch")
    return data


def _packet_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BindingError(f"unexpected packet symlink: {path}")
        if not path.is_file(): continue
        if len(rows) >= 512: raise BindingError("packet file count exceeds bound")
        data = _safe_read(path, MAX_PACKET)
        total += len(data)
        if total > MAX_PACKET: raise BindingError("packet aggregate exceeds bound")
        rows.append({"path": str(path.relative_to(root)), "bytes": len(data), "sha256": sha256(data)})
    return rows


def bind(source_root: Path, output: Path, *, source_audit: Callable[[Path, Path], int] | None = None, tools_fn: Callable[[], dict[str, Any]] | None = None, command_fn: Callable[..., tuple[int, bytes, bytes]] | None = None) -> int:
    source_root = source_root.absolute(); output = output.absolute()
    if output.exists() or output.is_symlink(): raise BindingError("output exists; no-clobber")
    try:
        SOURCE_AUDITOR._validate_output_parent(output)
    except SOURCE_AUDITOR.AuditError as exc:
        raise BindingError(str(exc)) from exc
    output.mkdir(parents=True)
    try:
        SOURCE_AUDITOR._validate_output_parent(output)
    except SOURCE_AUDITOR.AuditError as exc:
        raise BindingError(str(exc)) from exc
    audit_dir = output / "source-audit"
    packet: dict[str, Any] = {"status": "BLOCKED", "execution": {"source_import": False, "cargo": {"attempted": False, "completed": False, "exit": None}, "build": False, "download": False, "model": False}}
    source_audit = source_audit or (lambda root, dest: SOURCE_AUDITOR.audit(root, dest))
    tools_fn = tools_fn or tool_identity; command_fn = command_fn or command_bytes
    try:
        source_audit(source_root, audit_dir)
        source = _load_source_inventory(audit_dir); source_rows = source.get("source", {}).get("rows", [])
        derived = Path(source["workspace"]["path"]); before = _source_readback(source_root, source_rows); derived_before = _derived_readback(derived, source["workspace"])
        _verify_source_snapshot(source, before); _verify_derived_snapshot(source, derived_before)
        packet["source"] = {"inventory": source, "before": before, "derived_before": derived_before}
        tools_before = tools_fn(); cargo = tools_before["cargo"]["resolved"]
        env = {"CARGO_HOME": os.environ.get("CARGO_HOME", "/root/.cache/vokra-mimi-cargo"), "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", "/root/.rustup"), "CARGO_NET_OFFLINE": "true", "CARGO_TERM_COLOR": "never", "RUST_BACKTRACE": "0", "HOME": os.environ.get("HOME", "/root"), "PATH": str(Path(tools_before["rustc"]["resolved"]).parent) + ":/usr/bin:/bin"}
        argv = [cargo, *METADATA_ARGS, "--manifest-path", str(derived / "Cargo.toml")]
        packet["execution"]["cargo"]["attempted"] = True
        code, stdout, stderr = command_fn(argv, cwd=derived, env=env)
        packet["execution"]["cargo"].update({"completed": True, "exit": code})
        _exclusive(output / "metadata-command.stdout", stdout); _exclusive(output / "metadata-command.stderr", stderr); _exclusive(output / "metadata-command.exit", f"{code}\n".encode()); _write_json(output / "metadata-command.argv.json", argv); _write_json(output / "metadata-command.env.json", env)
        packet["command"] = {"argv": argv, "env": env, "exit": code, "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr)}
        if code != 0: packet["metadata"] = {"status": "BLOCKED_METADATA_COMMAND", "exit": code}
        else:
            _exclusive(output / "metadata.json", stdout); packet["metadata"] = inspect_metadata(json.loads(stdout), tomllib.loads(_safe_read(derived / "Cargo.lock").decode()), authenticated_target=TARGET)
        tools_after = tools_fn(); after = _source_readback(source_root, source_rows); derived_after = _derived_readback(derived, source["workspace"])
        if tools_before != tools_after: raise BindingError("authenticated toolchain changed during metadata command")
        _verify_source_snapshot(source, after); _verify_derived_snapshot(source, derived_after)
        if before != after or derived_before != derived_after: raise BindingError("source, copied lock, or derived manifest changed during metadata command")
        packet["source"].update({"after": after, "derived_after": derived_after}); packet["tools"] = tools_before
        packet["native"] = "BLOCKED_NATIVE_MARKER" if packet.get("metadata", {}).get("status") == "BLOCKED_NATIVE_MARKER" else "OPEN"; packet["license_owner_signoff"] = "OPEN_NOT_OWNER_SIGNABLE"; packet["registry"] = "OPEN_NOT_COLLECTED"; packet["checkpoint"] = False; packet["model_execution"] = False; packet["parity"] = "OPEN"
        packet["status"] = "BLOCKED" if packet.get("metadata", {}).get("status", "").startswith("BLOCKED") else "OPEN_METADATA_BOUND_NO_NATIVE_LICENSE_APPROVAL"
    except Exception as exc:
        packet["error"] = str(exc); packet["status"] = "BLOCKED"
        if not (output / "metadata-command.stdout").exists(): _exclusive(output / "metadata-command.stdout", b"")
        if not (output / "metadata-command.stderr").exists(): _exclusive(output / "metadata-command.stderr", str(exc).encode()[:MAX_OUTPUT])
        if not (output / "metadata-command.exit").exists(): _exclusive(output / "metadata-command.exit", b"NOT_RUN\n")
    packet["source_inventory_payload_sha256"] = packet.get("source", {}).get("inventory", {}).get("inventory_payload_sha256")
    packet["files"] = _packet_files(output); packet["packet_file_inventory_sha256"] = sha256(json.dumps(packet["files"], sort_keys=True, separators=(",", ":")).encode()); payload = (json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(); packet["packet_payload_sha256"] = sha256(payload); _write_json(output / "metadata-binding.json", packet)
    return 0 if packet["status"] != "BLOCKED" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(argv)
    try: return bind(args.source_root, args.output)
    except (BindingError, SOURCE_AUDITOR.AuditError, OSError, ValueError, TypeError) as exc:
        print(f"BLOCKED: {exc}", file=os.sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
