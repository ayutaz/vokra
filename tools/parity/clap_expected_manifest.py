#!/usr/bin/env python3
"""Derive a CLAP state-dict manifest from an official meta-device model.

This VAST-only route constructs ``ClapModel(config)`` on the PyTorch meta
device. It never calls ``from_pretrained``, reads checkpoint bytes, or runs a
forward. The output is deliberately named SOURCE_DERIVED and is not an
observed checkpoint manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "clap_dump_reference.py"
REPOSITORY = "laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
SCHEMA = "vokra-clap-htsat-fused-source-derived-expected-manifest-v1"


def cli_path(raw: str, label: str) -> Path:
    """Preserve and validate raw CLI spelling before Path normalization."""

    if not isinstance(raw, str) or not raw.startswith("/") or raw == "/" or "\x00" in raw:
        raise RuntimeError(f"{label} path must be absolute, non-root, and NUL-free")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        raise RuntimeError(f"{label} path contains unsafe lexical components")
    current = Path("/")
    for part in parts[1:]:
        current /= part
        if current.is_symlink() and current != Path("/var"):
            raise RuntimeError(f"{label} path has symlink ancestry: {current}")
    return Path(raw)


def write_atomic_no_replace(path: Path, text: str) -> None:
    require_output_parent(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            temporary_stat = os.stat(temporary, follow_symlinks=False)
            if not stat.S_ISREG(temporary_stat.st_mode):
                raise RuntimeError(f"temporary output is not regular: {temporary}")
            temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        verify_temporary_identity(temporary, temporary_identity)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"output already exists: {path}") from exc
        verify_file_identity(path, temporary_identity, "published output")
    finally:
        cleanup_temporary(temporary, temporary_identity)


def verify_temporary_identity(path: Path, expected: tuple[int, int] | None) -> None:
    verify_file_identity(path, expected, "temporary output")


def verify_file_identity(path: Path, expected: tuple[int, int] | None, label: str) -> None:
    if expected is None or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"{label} identity cannot be verified safely")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != expected:
            raise RuntimeError(f"{label} was replaced or is not regular: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def cleanup_temporary(path: Path | None, expected: tuple[int, int] | None) -> None:
    if path is None or expected is None:
        return
    try:
        current = os.stat(path, follow_symlinks=False)
        if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == expected:
            path.unlink()
    except OSError:
        pass


def require_output_parent(path: Path) -> None:
    """Require an existing, regular, symlink-free output ancestry."""

    if not path.is_absolute() or path.parent == Path("/") or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RuntimeError(f"output path must be absolute and dot-free: {path}")
    current = path.parent
    while True:
        if current.is_symlink() and current != Path("/var"):
            raise RuntimeError(f"output path has symlink ancestry: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError(f"output parent must be an existing regular directory: {path.parent}")


def load_manifest_helpers() -> tuple[Any, Any]:
    spec = importlib.util.spec_from_file_location("vokra_clap_dump_reference", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared CLAP reference: {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_state_dict_manifest, module.validate_tensor_manifest


def source_fact(cls: Any) -> dict[str, str]:
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError(f"cannot locate official source for {cls.__name__}")
    path = Path(source).resolve()
    return {
        "class": cls.__name__,
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "signature": str(inspect.signature(cls)),
    }


def run(config_path: Path, output_path: Path) -> None:
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError(f"config is missing or symlinked: {config_path}")
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError(f"output already exists: {output_path}")

    import torch
    import transformers
    from transformers import ClapConfig, ClapModel

    config = ClapConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    try:
        with torch.device("meta"):
            model = ClapModel(config)
        build_manifest, validate_manifest = load_manifest_helpers()
        manifest, roles = build_manifest(model.state_dict())
        validate_manifest(manifest, roles)
    except Exception as exc:
        blocked = {
            "schema": SCHEMA,
            "status": "BLOCKED_META_CONSTRUCTION",
            "repository": REPOSITORY,
            "revision": REVISION,
            "weights": "NOT_ACQUIRED",
            "model_load": "NOT_PERFORMED",
            "model_forward": "NOT_PERFORMED",
            "execution": "META_CONSTRUCTION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_atomic_no_replace(output_path, json.dumps(blocked, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"SOURCE_DERIVED_EXPECTED_MANIFEST blocked: {exc}") from exc

    evidence = {
        "schema": SCHEMA,
        "status": "SOURCE_DERIVED_EXPECTED_MANIFEST",
        "repository": REPOSITORY,
        "revision": REVISION,
        "transformers_version": transformers.__version__,
        "transformers_sources": {
            "config": source_fact(ClapConfig),
            "model": source_fact(ClapModel),
        },
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "execution": "META_CONSTRUCTION_ONLY",
        "manifest_kind": "SOURCE_DERIVED_EXPECTED_MANIFEST",
        "state_dict_roles": roles,
        "tensor_manifest": manifest,
    }
    write_atomic_no_replace(output_path, json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    assert SCHEMA.endswith("-v1")
    assert REPOSITORY == "laion/clap-htsat-fused"
    assert len(REVISION) == 40 and all(char in "0123456789abcdef" for char in REVISION)
    source = (ROOT / "clap_expected_manifest.py").read_text(encoding="utf-8")
    for token in ("torch.device(\"meta\")", "SOURCE_DERIVED_EXPECTED_MANIFEST", "from_pretrained"):
        assert token in source
    assert "model." + "forward(" not in source
    with tempfile.TemporaryDirectory(prefix="vokra-clap-manifest-") as temporary:
        for raw in ("relative", "/", "/tmp/./manifest.json", "/tmp/../manifest.json", "/tmp/manifest\x00.json"):
            try:
                cli_path(raw, "output")
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe raw output path was accepted")
        owned_temp = Path(temporary) / "owned.tmp"
        owned_temp.write_text("owner", encoding="utf-8")
        owned_stat = os.stat(owned_temp, follow_symlinks=False)
        owned_temp.unlink()
        owned_temp.write_text("replacement", encoding="utf-8")
        try:
            verify_temporary_identity(owned_temp, (owned_stat.st_dev, owned_stat.st_ino))
        except RuntimeError:
            pass
        else:
            raise AssertionError("replacement temporary was accepted")
        cleanup_temporary(owned_temp, (owned_stat.st_dev, owned_stat.st_ino))
        assert owned_temp.exists(), "cleanup removed a replacement temporary"
        output = Path(temporary) / "manifest.json"
        write_atomic_no_replace(output, "{}\n")
        try:
            write_atomic_no_replace(output, "{\"tampered\":true}\n")
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("expected manifest replacement was accepted")
        real_output_dir = Path(temporary) / "real-output"
        real_output_dir.mkdir()
        linked_output_dir = Path(temporary) / "linked-output"
        linked_output_dir.symlink_to(real_output_dir, target_is_directory=True)
        try:
            write_atomic_no_replace(linked_output_dir / "unsafe.json", "{}\n")
        except RuntimeError as exc:
            assert "symlink ancestry" in str(exc)
        else:
            raise AssertionError("symlinked output ancestry was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        if args.config is not None or args.output is not None:
            parser.error("--self-test accepts no model paths")
        self_test()
        print("clap expected manifest self-test: OK")
        return 0
    if args.config is None or args.output is None:
        parser.error("normal runs require --config and --output")
    config = cli_path(args.config, "config")
    output = cli_path(args.output, "output")
    run(config, output)
    print(f"CLAP_SOURCE_DERIVED_EXPECTED_MANIFEST: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
