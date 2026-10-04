#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Create a bounded, static inventory of the pinned Moshi source checkout.

This collector never imports or executes Moshi and never touches a model or
checkpoint.  Its output is a fact inventory only: it is not an approval,
license classification, dependency lock, or owner-signable record.  The
runtime reference gate remains blocked until a separately reviewed canonical
owner/dependency/native-payload gate exists.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SOURCE_URL = "https://github.com/kyutai-labs/moshi.git"
SOURCE_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
MAX_FILES = 5_000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_PATH_BYTES = 4 * 1024
MAX_GIT_OUTPUT_BYTES = 32 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 20


def _reap_and_close_process(process: subprocess.Popen[bytes]) -> None:
    """Ensure a bounded git child and both Python pipe wrappers are closed."""
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait()
        except OSError:
            pass
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _safe_absolute(value: str | Path, label: str) -> Path:
    raw = os.fspath(value)
    if not raw.startswith("/") or "\x00" in raw or "//" in raw or "/./" in raw or "/../" in raw or raw.endswith(("/.", "/..")):
        raise ValueError(f"{label} must be an absolute path")
    components = raw.split("/")
    if components[0] != "" or any(not part or part in {".", ".."} for part in components[1:]):
        raise ValueError(f"{label} contains an unsafe path component")
    path = Path(raw)
    if path == Path("/"):
        raise ValueError(f"{label} must not be the filesystem root")
    current = Path("/")
    for component in components[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise ValueError(f"{label} contains a symlink component: {current}")
        except OSError as error:
            raise ValueError(f"{label} component cannot be inspected: {current}") from error
    return path


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size)


def _read_regular_bytes(path: Path, label: str, *, max_bytes: int | None = None) -> tuple[bytes, tuple[int, int, int]]:
    limit = MAX_FILE_BYTES if max_bytes is None else max_bytes
    try:
        path_before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or cannot be inspected") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if path_before.st_size > limit:
        raise ValueError(f"{label} exceeds the bounded file-size limit")
    path_identity = _identity(path_before)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing, symlinked, or cannot be opened") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        if _identity(before) != path_identity:
            raise ValueError(f"{label} changed before being opened")
        if before.st_size > limit:
            raise ValueError(f"{label} exceeds the bounded file-size limit")
        chunks = bytearray()
        while len(chunks) <= limit:
            chunk = os.read(fd, min(1 << 20, limit + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
            if len(chunks) > limit:
                raise ValueError(f"{label} grew beyond the bounded file-size limit")
        after = os.fstat(fd)
        try:
            path_after = path.lstat()
        except OSError as error:
            raise ValueError(f"{label} disappeared while being read") from error
        if _identity(before) != _identity(after) or path_identity != _identity(path_after) or len(chunks) != after.st_size:
            raise ValueError(f"{label} changed while being read")
        return bytes(chunks), _identity(after)
    except OSError as error:
        raise ValueError(f"{label} could not be read safely") from error
    finally:
        os.close(fd)


def _git(root: Path, *args: str) -> str:
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    try:
        process = subprocess.Popen(
            ["git", "-C", str(root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
        while selector.get_map():
            if time.monotonic() > deadline:
                process.kill()
                process.wait()
                raise ValueError(f"git {' '.join(args)} timed out")
            for key, _ in selector.select(timeout=0.25):
                chunk = os.read(key.fileobj.fileno(), 1 << 16)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[key.data].extend(chunk)
                if len(buffers[key.data]) > MAX_GIT_OUTPUT_BYTES:
                    process.kill()
                    process.wait()
                    raise ValueError(f"git {' '.join(args)} output exceeds the bounded limit")
        return_code = process.wait(timeout=2)
        stdout = bytes(buffers["stdout"])
        stderr = bytes(buffers["stderr"])
        if return_code != 0:
            detail = stderr.decode("utf-8", "replace")
            raise ValueError(f"git {' '.join(args)} failed: {detail.strip()}")
        return stdout.decode("utf-8")
    except (OSError, subprocess.TimeoutExpired) as error:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise ValueError(f"git {' '.join(args)} failed: {error}") from error
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            _reap_and_close_process(process)


def _validate_origin(origin: str) -> str:
    normalized = origin.strip().rstrip("/")
    accepted = {
        SOURCE_URL,
        SOURCE_URL.removesuffix(".git"),
        "git@github.com:kyutai-labs/moshi.git",
        "ssh://git@github.com/kyutai-labs/moshi.git",
    }
    if normalized not in accepted:
        raise ValueError(f"origin is not the official kyutai-labs/moshi repository: {origin!r}")
    return normalized


def _checkout_state(git: Callable[..., str]) -> tuple[str, str]:
    origin = _validate_origin(git("remote", "get-url", "origin"))
    revision = git("rev-parse", "HEAD").strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"source revision mismatch: {revision!r}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError("source checkout is dirty")
    return origin, revision


def _tracked_paths(root: Path) -> list[str]:
    raw = _git(root, "ls-files", "-z")
    paths = raw.split("\0")
    if paths and paths[-1] == "":
        paths.pop()
    if len(paths) > MAX_FILES:
        raise ValueError("tracked source file count exceeds the bounded limit")
    if len(set(paths)) != len(paths):
        raise ValueError("git reported duplicate tracked paths")
    for value in paths:
        _validate_relative(value)
    return sorted(paths)


def _selected_paths(tracked: list[str]) -> list[str]:
    selected = [
        path
        for path in tracked
        if (path.startswith("moshi/") and path.endswith(".py"))
        or path in {
            "moshi/pyproject.toml",
            "moshi/LICENSE",
            "moshi/LICENSE.audiocraft",
        }
    ]
    required = {
        "moshi/pyproject.toml",
        "moshi/LICENSE",
        "moshi/LICENSE.audiocraft",
    }
    missing = sorted(required.difference(selected))
    if missing:
        raise ValueError(f"required Moshi source/license files are missing: {missing}")
    if not selected:
        raise ValueError("no Moshi source files were selected")
    return sorted(selected)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_relative(value: str) -> None:
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError(f"tracked path exceeds the bounded path limit: {value!r}")
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe tracked path: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"unsafe tracked path: {value!r}")


def _assert_no_symlink_components(root: Path, relative: str) -> None:
    current = root
    for component in PurePosixPath(relative).parts:
        current /= component
        try:
            if current.is_symlink():
                raise ValueError(f"tracked source path has a symlink component: {relative}")
        except OSError as error:
            raise ValueError(f"tracked source path cannot be inspected: {relative}") from error


def _read_selected(root: Path, paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, bytes], int]:
    rows: list[dict[str, Any]] = []
    contents: dict[str, bytes] = {}
    total = 0
    for relative in paths:
        path = root / Path(*PurePosixPath(relative).parts)
        _assert_no_symlink_components(root, relative)
        if relative == "moshi/pyproject.toml" or relative.startswith("moshi/LICENSE"):
            limit = MAX_METADATA_BYTES
        else:
            limit = MAX_FILE_BYTES
        raw, _source_identity = _read_regular_bytes(path, f"tracked source {relative}", max_bytes=limit)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("selected source bytes exceed the bounded aggregate limit")
        contents[relative] = raw
        kind = "python" if relative.endswith(".py") else "metadata" if relative.endswith(".toml") else "license_bytes"
        rows.append({"path": relative, "kind": kind, "bytes": len(raw), "sha256": _sha256(raw)})
    return rows, contents, total


def _attribute_name(value: ast.AST) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        parent = _attribute_name(value.value)
        return f"{parent}.{value.attr}" if parent else value.attr
    return None


def _resolve_alias(name: str | None, aliases: dict[str, str]) -> str | None:
    if not name:
        return None
    head, separator, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if separator else resolved


def _ast_inventory(rows: list[dict[str, Any]], contents: dict[str, bytes]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    imports: list[dict[str, Any]] = []
    dynamic_literals: list[dict[str, Any]] = []
    dynamic_unknown: list[dict[str, Any]] = []
    for row in rows:
        if row["kind"] != "python":
            continue
        relative = str(row["path"])
        try:
            tree = ast.parse(contents[relative].decode("utf-8"), filename=relative)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise ValueError(f"cannot statically parse {relative}: {error}") from error
        aliases: dict[str, str] = {}
        nodes = sorted(ast.walk(tree), key=lambda value: (getattr(value, "lineno", 0), getattr(value, "col_offset", 0)))
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"path": relative, "line": node.lineno, "kind": "import", "module": alias.name})
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append({"path": relative, "line": node.lineno, "kind": "from", "module": module, "level": node.level})
                for alias in node.names:
                    if alias.name != "*":
                        prefix = "." * node.level + module
                        aliases[alias.asname or alias.name] = f"{prefix}.{alias.name}".lstrip(".")
            elif isinstance(node, ast.Call):
                name = _resolve_alias(_attribute_name(node.func), aliases)
                if name not in {"__import__", "importlib.import_module", "importlib.util.find_spec"}:
                    continue
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    dynamic_literals.append({"path": relative, "line": node.lineno, "call": name, "module": node.args[0].value})
                else:
                    dynamic_unknown.append({"path": relative, "line": node.lineno, "call": name, "reason": "non-literal import target"})
    return sorted(imports, key=lambda value: (value["path"], value["line"], value["kind"], value["module"])), sorted(dynamic_literals, key=lambda value: (value["path"], value["line"], value["call"])), sorted(dynamic_unknown, key=lambda value: (value["path"], value["line"], value["call"]))


def _declared_dependencies(raw: bytes) -> dict[str, Any]:
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Moshi pyproject.toml is not valid TOML: {error}") from error
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("Moshi pyproject.toml has no project table")
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict) or any(not isinstance(key, str) or not isinstance(value, list) or any(not isinstance(item, str) for item in value) for key, value in optional.items()):
        raise ValueError("Moshi optional-dependencies has an unsupported shape")
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
        raise ValueError("Moshi project dependencies has an unsupported shape")
    build = document.get("build-system", {})
    build_requires = build.get("requires", []) if isinstance(build, dict) else []
    if not isinstance(build_requires, list) or any(not isinstance(item, str) for item in build_requires):
        raise ValueError("Moshi build-system requires has an unsupported shape")
    return {
        "project_name": project.get("name"),
        "requires_python": project.get("requires-python"),
        "dependencies": list(dependencies),
        "optional_dependencies": {key: list(optional[key]) for key in sorted(optional)},
        "build_system_requires": list(build_requires),
    }


def collect_inventory(source_root: str | Path, *, git_runner: Callable[..., str] | None = None) -> dict[str, Any]:
    root = _safe_absolute(source_root, "source_root")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source_root must be a real directory")
    git = git_runner or (lambda *args: _git(root, *args))
    origin, revision = _checkout_state(git)
    tracked = _tracked_paths(root) if git_runner is None else _split_tracked(git("ls-files", "-z"))
    selected = _selected_paths(tracked)
    files, contents, total = _read_selected(root, selected)
    imports, dynamic_literals, dynamic_unknown = _ast_inventory(files, contents)
    dependencies = _declared_dependencies(contents["moshi/pyproject.toml"])
    final_origin, final_revision = _checkout_state(git)
    if (origin, revision) != (final_origin, final_revision):
        raise ValueError("source checkout identity changed while being inventoried")
    inventory: dict[str, Any] = {
        "kind": "STATIC_SOURCE_INVENTORY",
        "status": "PENDING_REVIEW_NOT_OWNER_SIGNABLE",
        "no_upload": True,
        "model_runtime_download_import_build": False,
        "source": {
            "repository": SOURCE_URL,
            "origin": origin,
            "revision": revision,
            "clean": True,
            "tracked_selected_files": len(files),
            "selected_bytes": total,
            "files": files,
        },
        "declared_dependencies": dependencies,
        "ast_imports": imports,
        "literal_dynamic_imports": dynamic_literals,
        "unresolved_dynamic_imports": dynamic_unknown,
        "closure_claim": "NOT_COMPLETE_RUNTIME_CLOSURE; STATIC AST INVENTORY ONLY",
    }
    digest_input = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    inventory["inventory_sha256"] = _sha256(digest_input)
    return inventory


def _split_tracked(raw: str) -> list[str]:
    values = raw.split("\0")
    if values and values[-1] == "":
        values.pop()
    if len(values) > MAX_FILES or len(set(values)) != len(values):
        raise ValueError("tracked path count or uniqueness check failed")
    for value in values:
        _validate_relative(value)
    return values


def publish_no_clobber(output: str | Path, value: dict[str, Any]) -> None:
    destination = _safe_absolute(output, "output")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    encoded = json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("inventory JSON exceeds the bounded output limit")
    temporary: Path | None = None
    owned_identity: tuple[int, int, int] | None = None

    def current_identity(path: Path) -> tuple[int, int, int] | None:
        try:
            info = path.stat(follow_symlinks=False)
        except OSError:
            return None
        return _identity(info) if stat.S_ISREG(info.st_mode) else None

    try:
        fd, raw_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=parent)
        temporary = Path(raw_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            owned_identity = _identity(os.fstat(stream.fileno()))
        if current_identity(temporary) != owned_identity:
            raise ValueError("temporary inventory file was replaced before publication")
        os.link(temporary, destination)
        if current_identity(destination) != owned_identity:
            raise ValueError("published inventory identity changed during publication")
        if current_identity(temporary) == owned_identity:
            temporary.unlink()
            temporary = None
    finally:
        if temporary is not None:
            if owned_identity is not None and current_identity(temporary) == owned_identity:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        inventory = collect_inventory(args.source_root)
        publish_no_clobber(args.output, inventory)
    except (OSError, ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "STATIC_SOURCE_INVENTORY", "inventory_sha256": inventory["inventory_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
