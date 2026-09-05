#!/usr/bin/env python3
"""Safe, inspection-only evidence collector for Coqui XTTS-v2.

The XTTS release is a multi-file pickle checkpoint.  This tool inventories
archives and unsafe globals, then attempts only ``weights_only=True`` loading;
it never adds globals, executes a model, merges files, or writes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any, Iterable


MODEL_REPOSITORY = "coqui/XTTS-v2"
MODEL_REVISION = "6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
SOURCE_REPOSITORY = "coqui-ai/TTS"
SOURCE_URL = "https://github.com/coqui-ai/TTS.git"
SOURCE_REVISION = "480a6cdf7dab508063c5d2e1b92fb7cd9f4f63c1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"symlink is not an authenticated bundle member: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes bundle root: {path}") from error
    if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"unsafe bundle path: {path}")
    return relative.as_posix()


def files_under(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    blockers: list[str] = []
    if not root.is_dir():
        return [], [f"missing bundle directory: {root}"]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".cache", ".git"} for part in relative.parts) or path.is_dir():
            continue
        try:
            safe_relative(path, root)
        except ValueError as error:
            blockers.append(str(error))
            continue
        if not path.is_file():
            blockers.append(f"bundle member is not regular: {path}")
            continue
        files.append(path)
    if not files:
        blockers.append(f"bundle has no regular files: {root}")
    return files, blockers


def identity(path: Path, root: Path) -> dict[str, Any]:
    return {"path": safe_relative(path, root), "size": path.stat().st_size, "sha256": sha256(path)}


def license_records(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in files:
        if "license" not in path.name.lower() and path.name.lower() not in {"notice", "copying", "readme.md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            records.append({**identity(path, root), "status": "READ_ERROR", "error": str(error)})
            continue
        front = ""
        if text.startswith("---"):
            parts = text.split("---", 2)
            front = parts[1] if len(parts) == 3 else ""
        declaration_text = front if path.name.lower() == "readme.md" else text + "\n" + front
        declarations = re.findall(
            r"(?im)(?:spdx-license-identifier\s*:\s*|^\s*license\s*:\s*|^\s*license\s*=\s*)([^\n#]+)",
            declaration_text,
        )
        declarations.extend(
            match.strip()
            for match in re.findall(
                r"(?i)\b(?:Coqui Public Model License|coqui-public-model-license|Apache License(?:,? Version)?\s*[0-9.]*|MIT License|BSD(?:-\d-Clause)?)\b",
                text,
            )
        )
        declarations = sorted({item.strip().strip('"\'') for item in declarations if item.strip()})
        records.append(
            {
                **identity(path, root),
                "status": "DECLARED_UNVERIFIED" if declarations else "UNKNOWN",
                "spdx_identifiers": re.findall(r"(?i)spdx-license-identifier\s*:\s*([a-z0-9.+-]+)", text),
                "declared_license": declarations,
            }
        )
    return records


def server_tree(root: Path, tree: Path | None, repository: str, revision: str, blockers: list[str]) -> dict[str, Any]:
    if tree is None:
        blockers.append(f"missing HF server tree evidence: {repository}")
        return {"status": "BLOCKED_MISSING_SERVER_TREE"}
    try:
        remote = json.loads(tree.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        blockers.append(f"HF server tree parse failed: {error}")
        return {"status": "BLOCKED_SERVER_TREE_PARSE"}
    remote_files = {
        (str(item["path"]), int(item["size"]))
        for item in remote.get("files", [])
        if item.get("type") in (None, "file") and item.get("path") is not None and item.get("size") is not None
    }
    local, local_blockers = files_under(root)
    blockers.extend(local_blockers)
    local_files = {(safe_relative(path, root), path.stat().st_size) for path in local}
    missing, extra = sorted(remote_files - local_files), sorted(local_files - remote_files)
    if remote.get("repository") != repository or remote.get("revision") != revision:
        blockers.append(f"HF tree identity mismatch for {repository}")
    if missing or extra:
        blockers.append(f"HF server/local tree mismatch: missing={missing!r} extra={extra!r}")
    return {
        "status": "MATCHED" if not missing and not extra else "MISMATCH",
        "repository": repository,
        "revision": revision,
        "server_tree_sha256": sha256(tree),
        "missing_local": missing,
        "unexpected_local": extra,
    }


def archive_inventory(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path.name, "container": "unknown"}
    blockers: list[str] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                result["container"] = "zip"
                members = archive.infolist()
                seen: set[str] = set()
                result["members"] = []
                for item in members:
                    name = item.filename
                    if name in seen:
                        blockers.append(f"archive duplicate member: {path}:{name}")
                    seen.add(name)
                    member_path = Path(name)
                    if "\x00" in name or "\\" in name or member_path.is_absolute() or ".." in member_path.parts:
                        blockers.append(f"archive unsafe member path: {path}:{name}")
                    mode = (item.external_attr >> 16) & 0o170000
                    allowed_modes = {0, 0o100000, 0o040000}
                    is_directory = item.is_dir() or name.endswith("/")
                    if mode not in allowed_modes or (is_directory and mode == 0o100000):
                        blockers.append(f"archive non-regular member: {path}:{name}")
                    result["members"].append(
                        {"name": name, "size": item.file_size, "compressed_size": item.compress_size, "type": "directory" if is_directory else "file"}
                    )
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                result["container"] = "tar"
                members = archive.getmembers()
                seen = set()
                result["members"] = []
                for item in members:
                    name = item.name
                    if name in seen:
                        blockers.append(f"archive duplicate member: {path}:{name}")
                    seen.add(name)
                    member_path = Path(name)
                    if "\x00" in name or "\\" in name or member_path.is_absolute() or ".." in member_path.parts:
                        blockers.append(f"archive unsafe member path: {path}:{name}")
                    if not (item.isdir() or item.isreg()):
                        blockers.append(f"archive non-regular member: {path}:{name}")
                    result["members"].append(
                        {"name": name, "size": item.size, "type": item.type.decode(errors="replace")}
                    )
        else:
            result["archive_error"] = "unrecognized archive format"
            blockers.append(f"archive format is unrecognized: {path}")
    except Exception as error:
        result["archive_error"] = str(error)
        blockers.append(f"archive inventory failed: {path}: {error}")
    result["archive_blockers"] = blockers
    return result


def tensor_manifest(value: Any, prefix: str = "") -> tuple[dict[str, Any], list[str]]:
    tensors: dict[str, Any] = {}
    unsupported: list[str] = []
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all().item())
        tensors[prefix or "<root>"] = {
            "shape": [int(axis) for axis in value.shape],
            "dtype": str(value.dtype),
            "count": int(value.numel()),
            "finite": finite,
        }
        if not finite:
            unsupported.append(f"non-finite tensor: {prefix}")
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            child, bad = tensor_manifest(value[key], f"{prefix}.{key}" if prefix else str(key))
            tensors.update(child)
            unsupported.extend(bad)
    elif isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            child, bad = tensor_manifest(child_value, f"{prefix}[{index}]")
            tensors.update(child)
            unsupported.extend(bad)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        unsupported.append(f"{prefix}:{type(value).__name__}")
    return tensors, unsupported


def pickle_evidence(root: Path, files: list[Path], blockers: list[str]) -> dict[str, Any]:
    paths = [path for path in files if path.suffix.lower() in {".pth", ".pt", ".bin"}]
    if not paths:
        blockers.append("XTTS bundle has no pickle checkpoint assets")
        return {"files": []}
    try:
        import torch
    except ImportError as error:
        blockers.append(f"torch unavailable for safe checkpoint inventory: {error}")
        return {"files": []}
    evidence = []
    for path in paths:
        item = {**identity(path, root), "archive": archive_inventory(path)}
        blockers.extend(item["archive"]["archive_blockers"])
        try:
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
        except Exception as error:  # noqa: BLE001 - preserve blocker
            unsafe = []
            item["unsafe_globals_error"] = f"{type(error).__name__}: {error}"
            blockers.append(f"unsafe-global inventory failed for {path}: {error}")
        item["unsafe_globals"] = sorted(map(str, unsafe))
        if unsafe:
            blockers.append(f"checkpoint requires custom/unsafe globals: {path}")
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as error:  # noqa: BLE001 - no fallback is permitted
            item["safe_load_status"] = "BLOCKED_WEIGHTS_ONLY"
            item["safe_load_error"] = f"{type(error).__name__}: {error}"
            blockers.append(f"weights_only load failed for {path}: {error}")
            evidence.append(item)
            continue
        tensors, unsupported = tensor_manifest(payload)
        item.update(
            {
                "safe_load_status": "SAFE_LOADED",
                "resident_scope": "one loaded container; released before next checkpoint",
                "tensor_count": len(tensors),
                "parameter_count": sum(entry["count"] for entry in tensors.values()),
                "tensor_manifest": tensors,
                "unsupported_objects": unsupported,
            }
        )
        if not tensors or unsupported:
            blockers.append(f"checkpoint payload is not a pure tensor manifest: {path}")
        del payload
        evidence.append(item)
    return {"files": evidence}


def source_evidence(root: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository": SOURCE_REPOSITORY,
        "url": SOURCE_URL,
        "pinned_revision": SOURCE_REVISION,
        "license": "UNPINNED_UNKNOWN",
        "tracked_files": [],
        "role_files": {},
        "dependency_files": [],
    }
    try:
        resolved_origin = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        resolved_origin = ""
        blockers.append(f"source origin unavailable: {error}")
    result["resolved_origin"] = resolved_origin
    if resolved_origin != SOURCE_URL:
        blockers.append(f"source origin {resolved_origin!r} != pinned {SOURCE_URL!r}")
    try:
        output = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
        files = [root / os.fsdecode(item) for item in output.split(b"\0") if item]
        files = sorted(path for path in files if path.is_file() and not path.is_symlink())
    except (OSError, subprocess.CalledProcessError) as error:
        files = []
        blockers.append(f"source tracked inventory failed: {error}")
    result["tracked_files"] = [identity(path, root) for path in files]
    try:
        actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        actual = ""
        blockers.append(f"source revision unavailable: {error}")
    result["resolved_revision"] = actual
    if actual != SOURCE_REVISION:
        blockers.append(f"source revision {actual!r} != pinned {SOURCE_REVISION!r}")
    roles = {
        "config": ("config",),
        "model": ("model",),
        "gpt": ("gpt", "xtts"),
        "dvae": ("dvae",),
        "hifigan": ("hifigan",),
        "speaker_encoder": ("speaker",),
        "tokenizer": ("token",),
        "inference": ("inference",),
    }
    for role, markers in roles.items():
        matches = [path for path in files if all(marker in safe_relative(path, root).lower() for marker in markers)]
        result["role_files"][role] = [identity(path, root) for path in matches]
        if not matches:
            blockers.append(f"source has no tracked {role} path")
    for path in files:
        if path.suffix in {".py", ".toml", ".txt", ".cfg", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            mentions = [line.strip() for line in text.splitlines() if any(name in line for name in ("torch", "torchaudio", "coqui", "soundfile", "tokenizer"))]
            if mentions:
                result["dependency_files"].append({**identity(path, root), "dependency_mentions": mentions})
    result["code_license_status"] = "PRIMARY_REVIEW_REQUIRED"
    result["dependency_license_status"] = "UNREVIEWED_BLOCKER"
    result["license_records"] = license_records(root, files)
    if not result["license_records"]:
        blockers.append("XTTS source has no tracked LICENSE/NOTICE/README license record")
    blockers.append("XTTS source/dependency licenses remain unreviewed")
    return result


def _inspect(model_dir: Path, source_dir: Path, evidence_dir: Path, model_tree: Path | None, revision: str) -> int:
    blockers: list[str] = []
    files, file_blockers = files_under(model_dir)
    blockers.extend(file_blockers)
    model_license = license_records(model_dir, files)
    model = {
        "repository": MODEL_REPOSITORY,
        "revision": revision,
        "server_tree": server_tree(model_dir, model_tree, MODEL_REPOSITORY, revision, blockers),
        "files": [identity(path, model_dir) for path in files],
        "pickle_assets": pickle_evidence(model_dir, files, blockers),
        "license_records": model_license,
        "config_files": [],
        "vocab_files": [],
        "samples": [],
    }
    for path in files:
        rel = safe_relative(path, model_dir).lower()
        if path.name == "config.json" or path.name == "vocab.json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                blockers.append(f"JSON parse failed for {path}: {error}")
                continue
            record = {**identity(path, model_dir), "json": value}
            if path.name == "config.json":
                record["model_type"] = value.get("model_type") if isinstance(value, dict) else None
                record["architectures"] = value.get("architectures") if isinstance(value, dict) else None
                model["config_files"].append(record)
            else:
                model["vocab_files"].append(record)
        if "sample" in rel or path.suffix.lower() in {".wav", ".mp3", ".flac"}:
            model["samples"].append(identity(path, model_dir))
    if not model["config_files"]:
        blockers.append("XTTS bundle has no parseable config.json")
    if not model["vocab_files"]:
        blockers.append("XTTS bundle has no parseable vocab.json")
    cpml = {
        "coqui public model license",
        "coqui-public-model-license",
    }
    if not any(
        any(any(marker in declaration.lower() for marker in cpml) for declaration in item.get("declared_license", []))
        for item in model_license
    ):
        blockers.append("XTTS model has no authenticated Coqui Public Model License declaration")
    payload = {
        "format": "vokra-xtts-v2-inspection-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "model": model,
        "source": source_evidence(source_dir, blockers),
        "dataset_training_provenance": "RECORDED_ONLY_NOT_AUTHENTICATED",
        "speaker_preset_sample_rights": "NOT_AUTHORIZED",
        "voice_cloning_biometric_consent": "NOT_ASSESSED",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "blockers": sorted(set(blockers)),
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "xtts_v2_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blockers:
        print(f"XTTS-v2 inspection blocked; evidence preserved at {output}", file=sys.stderr)
        return 2
    print(f"XTTS-v2 inspection blocked; evidence preserved at {output}", file=sys.stderr)
    return 2


def _write_blocked_manifest(evidence_dir: Path, source_dir: Path, revision: str, error: Exception) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        resolved_origin = subprocess.run(
            ["git", "-C", str(source_dir), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (AttributeError, OSError, subprocess.CalledProcessError):
        resolved_origin = ""
    payload = {
        "format": "vokra-xtts-v2-inspection-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "model": {"repository": MODEL_REPOSITORY, "revision": revision},
        "source": {
            "repository": SOURCE_REPOSITORY,
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "resolved_origin": resolved_origin,
        },
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error": f"{type(error).__name__}: {error}",
        "blockers": ["inspection_error", f"{type(error).__name__}: {error}"],
    }
    output = evidence_dir / "xtts_v2_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"XTTS-v2 inspection blocked; evidence preserved at {output}", file=sys.stderr)


def inspect(model_dir: Path, source_dir: Path, evidence_dir: Path, model_tree: Path | None, revision: str) -> int:
    """Preserve a blocked manifest for every result, including unexpected errors."""
    try:
        return _inspect(model_dir, source_dir, evidence_dir, model_tree, revision)
    except Exception as error:  # noqa: BLE001 - the exception is itself evidence
        _write_blocked_manifest(evidence_dir, source_dir, revision, error)
        return 2


def self_test() -> None:
    root = Path.cwd().resolve()
    try:
        safe_relative(root.parent / "escape", root)
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal accepted")
    with tempfile.TemporaryDirectory(prefix="xtts-inspect-") as temporary:
        local = Path(temporary) / "local"
        local.mkdir()
        readme = local / "README.md"
        readme.write_text("description only\n", encoding="utf-8")
        assert license_records(local, [readme])[0]["status"] == "UNKNOWN"
        tree = Path(temporary) / "tree.json"
        tree.write_text(json.dumps({"repository": MODEL_REPOSITORY, "revision": "a" * 40, "files": [{"path": "other", "size": 1, "type": "file"}]}), encoding="utf-8")
        blockers: list[str] = []
        assert server_tree(local, tree, MODEL_REPOSITORY, "a" * 40, blockers)["status"] == "MISMATCH"
        assert any("tree mismatch" in blocker for blocker in blockers)
        archive = Path(temporary) / "unsafe.pth"
        archive_bytes = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive_bytes, "w") as unsafe_zip:
                unsafe_zip.writestr("../escape", b"x")
                unsafe_zip.writestr("../escape", b"y")
                unsafe_zip.writestr("safe\\escape", b"z")
                unsafe_zip.writestr("safe\x00nul", b"q")
        archive.write_bytes(archive_bytes.getvalue())
        archive_blockers: list[str] = []
        inventory = archive_inventory(archive)
        archive_blockers.extend(inventory["archive_blockers"])
        assert any("duplicate member" in blocker for blocker in archive_blockers)
        assert any("unsafe member path" in blocker for blocker in archive_blockers)
        assert sum("unsafe member path" in blocker for blocker in archive_blockers) >= 3
        safe_zip = Path(temporary) / "safe.zip"
        with zipfile.ZipFile(safe_zip, "w") as safe_archive:
            directory = zipfile.ZipInfo("safe/")
            directory.external_attr = 0o040755 << 16
            safe_archive.writestr(directory, b"")
            regular = zipfile.ZipInfo("safe/file.bin")
            regular.external_attr = 0o100644 << 16
            safe_archive.writestr(regular, b"x")
        assert archive_inventory(safe_zip)["archive_blockers"] == []
        symlink_zip = Path(temporary) / "symlink.zip"
        with zipfile.ZipFile(symlink_zip, "w") as symlink_archive:
            symlink = zipfile.ZipInfo("safe-link")
            symlink.external_attr = 0o120777 << 16
            symlink_archive.writestr(symlink, b"target")
        assert archive_inventory(symlink_zip)["archive_blockers"]
        hardlink_tar = Path(temporary) / "hardlink.tar"
        with tarfile.open(hardlink_tar, "w") as hardlink_archive:
            regular = tarfile.TarInfo("safe.bin")
            regular.size = 1
            hardlink_archive.addfile(regular, io.BytesIO(b"x"))
            hardlink = tarfile.TarInfo("alias.bin")
            hardlink.type = tarfile.LNKTYPE
            hardlink.linkname = "safe.bin"
            hardlink_archive.addfile(hardlink)
        assert archive_inventory(hardlink_tar)["archive_blockers"]
        broken = Path(temporary) / "broken.zip"
        broken.write_bytes(b"not a zip")
        broken_inventory = archive_inventory(broken)
        assert "archive_error" in broken_inventory
        assert broken_inventory["archive_blockers"]
        missing_evidence = Path(temporary) / "missing-evidence"
        assert inspect(local / "missing-model", local / "missing-source", missing_evidence, None, MODEL_REVISION) == 2
        missing_manifest = json.loads((missing_evidence / "xtts_v2_manifest.json").read_text(encoding="utf-8"))
        assert missing_manifest["status"] == "BLOCKED"
        assert missing_manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert missing_manifest["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
        assert missing_manifest["cpu_status"] == "UNSUPPORTED"
        assert missing_manifest["metal_status"] == "BLOCKED_BY_CPU"
        assert missing_manifest["parity_status"] == "NOT_RUN"
        original_files_under = globals()["files_under"]
        try:
            globals()["files_under"] = lambda _root: (_ for _ in ()).throw(RuntimeError("synthetic inspector failure"))
            exception_evidence = Path(temporary) / "exception-evidence"
            assert inspect(local, local, exception_evidence, None, MODEL_REVISION) == 2
        finally:
            globals()["files_under"] = original_files_under
        exception_manifest = json.loads((exception_evidence / "xtts_v2_manifest.json").read_text(encoding="utf-8"))
        assert exception_manifest["status"] == "BLOCKED"
        assert exception_manifest["evidence_stage"] == "INSPECTION_ONLY"
        assert exception_manifest["model"] == {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}
        assert exception_manifest["source"]["revision"] == SOURCE_REVISION
        assert exception_manifest["source"]["resolved_origin"] == ""
        assert exception_manifest["publication"] == "NO_UPLOAD"
        assert exception_manifest["blockers"]
    assert MODEL_REVISION.isascii() and len(MODEL_REVISION) == 40
    assert SOURCE_REVISION.isascii() and len(SOURCE_REVISION) == 40
    assert "get_unsafe_globals_in_checkpoint" in Path(__file__).read_text(encoding="utf-8")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        tensors, unsupported = tensor_manifest({"safe": torch.ones((2, 2))})
        assert tensors["safe"]["finite"] and not unsupported
    print("xtts_v2_inspect self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--model-tree", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.model_dir, args.source_dir, args.evidence_dir, args.model_tree, args.revision)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.model_dir, args.source_dir, args.evidence_dir, args.model_tree, args.revision)):
        parser.error("normal runs require model/source/evidence dirs, model tree, and immutable revision")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a complete 40-hex immutable HF revision")
    return inspect(args.model_dir, args.source_dir, args.evidence_dir, args.model_tree, args.revision)


if __name__ == "__main__":
    raise SystemExit(main())
