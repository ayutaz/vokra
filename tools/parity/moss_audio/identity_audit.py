#!/usr/bin/env -S uv run --frozen --project tools/parity/moss_audio/api_smoke --python 3.12 python
"""No-weight identity audit for the pinned MOSS-Audio releases.

This tool authenticates only source files, model metadata, checkpoint-index
references, and remote shard identities.  It never downloads or opens a
checkpoint shard.  License classification and owner approval stay pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/OpenMOSS/MOSS-Audio.git"
SOURCE_REVISION = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
SOURCE_FILES = {
    "src/configuration_moss_audio.py": "configuration",
    "src/modeling_moss_audio.py": "modeling",
    "src/processing_moss_audio.py": "processing",
}
LICENSE_FILENAME_PREFIXES = ("license", "licence", "copying", "notice")
VARIANTS = {
    "4b": {
        "repository": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
        "revision": "6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
    },
    "8b": {
        "repository": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
        "revision": "6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
    },
}
METADATA_FILES = {
    "config.json": "config",
    "tokenizer_config.json": "tokenizer",
    "processor_config.json": "processor",
    "vocab.json": "common_asset",
    "merges.txt": "common_asset",
    "chat_template.jinja": "common_asset",
    "generation_config.json": "common_asset",
    "model.safetensors.index.json": "checkpoint_index",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORMAT = "vokra-moss-audio-no-weight-identity-audit-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def normalize_card_data(value: Any) -> dict[str, Any]:
    """Return JSON-authenticated model_info cardData without inferring fields."""
    if isinstance(value, dict):
        result = value
    elif hasattr(value, "to_dict"):
        result = value.to_dict()
    else:
        raise ValueError("HF model_info cardData is not a JSON object")
    if not isinstance(result, dict) or not isinstance(result.get("license"), str) or not result["license"].strip():
        raise ValueError("HF model_info cardData license is missing")
    try:
        canonical(result)
    except (TypeError, ValueError) as exc:
        raise ValueError("HF model_info cardData is not JSON-serializable") from exc
    return result


def git_blob_sha1_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def safe_relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing, symlinked, or non-regular")


def license_filename(relative: str) -> bool:
    name = Path(relative).name.casefold()
    return name in LICENSE_FILENAME_PREFIXES or name.startswith(
        tuple(f"{prefix}." for prefix in LICENSE_FILENAME_PREFIXES)
    )


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def source_audit(root: Path, *, expected_revision: str = SOURCE_REVISION) -> dict[str, Any]:
    """Return authenticated source bytes and the complete tracked-tree identity."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("official source root is not a real directory")
    if git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official source checkout is dirty")
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(f"official source revision mismatch: {revision}")
    origin = git_output(root, "remote", "get-url", "origin")
    if origin != SOURCE_REPOSITORY:
        raise ValueError(f"official source origin mismatch: {origin}")
    tracked = sorted(path for path in git_output(root, "ls-files", "-z").split("\x00") if path)
    if not tracked:
        raise ValueError("official source tracked tree is empty")
    if any(license_filename(path) for path in tracked):
        raise ValueError("official source contains a tracked license filename")
    files: dict[str, Any] = {}
    for relative, role in SOURCE_FILES.items():
        safe_relative(relative)
        path = root / relative
        regular(path, f"official source {relative}")
        files[relative] = {
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_sha1": git_output(root, "hash-object", relative),
            "status": "AUTHENTICATED_BYTES_AND_GIT_BLOB",
        }
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": revision,
        "origin": origin,
        "tracked_tree": {
            "status": "AUTHENTICATED_CLEAN_EXACT_REVISION",
            "files": tracked,
            "files_sha256": digest(tracked),
        },
        "license": {
            "status": "PENDING_OWNER_APPROVAL",
            "spdx": None,
        },
        "license_file": {
            "path": None,
            "status": "ABSENT_AT_FIXED_REVISION",
        },
        "files": files,
    }


def load_tree(path: Path, variant: str) -> dict[str, Any]:
    regular(path, "HF tree evidence")
    tree = strict_json(path)
    identity = VARIANTS[variant]
    if not isinstance(tree, dict) or set(tree) != {
        "repository", "revision", "resolved_revision", "files", "card_data",
        "card_data_sha256", "license_file",
    }:
        raise ValueError("HF tree evidence schema is not exact")
    if tree["repository"] != identity["repository"] or tree["revision"] != identity["revision"] or tree["resolved_revision"] != identity["revision"]:
        raise ValueError("HF tree repository/revision identity mismatch")
    card_data = normalize_card_data(tree["card_data"])
    if tree["card_data_sha256"] != digest(card_data):
        raise ValueError("HF model_info cardData digest mismatch")
    license_file = tree["license_file"]
    if license_file != {"path": "LICENSE", "status": "ABSENT_AT_FIXED_REVISION"}:
        raise ValueError("HF model LICENSE absence evidence is not exact")
    rows = tree["files"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("HF tree file list is empty")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise ValueError("HF tree row schema is not exact")
        name = safe_relative(row["path"])
        if row["type"] != "file" or name in by_path or isinstance(row["size"], bool) or not isinstance(row["size"], int) or row["size"] < 0:
            raise ValueError(f"HF tree row is malformed: {name}")
        if not HEX40.fullmatch(str(row["git_blob_sha1"])):
            raise ValueError(f"HF tree Git blob is malformed: {name}")
        lfs = row["lfs_sha256"]
        if lfs is not None and not HEX64.fullmatch(str(lfs)):
            raise ValueError(f"HF tree LFS OID is malformed: {name}")
        by_path[name] = row
    if "LICENSE" in by_path:
        raise ValueError("HF model LICENSE is present despite absence evidence")
    return {
        "repository": tree["repository"],
        "revision": tree["revision"],
        "files": by_path,
        "card_data": card_data,
        "card_data_sha256": tree["card_data_sha256"],
        "license_file": license_file,
        "tree_sha256": sha256_file(path),
    }


def checkpoint_index(path: Path, tree: dict[str, Any]) -> dict[str, Any]:
    regular(path, "checkpoint index")
    index = strict_json(path)
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ValueError("checkpoint index schema is not exact")
    if not isinstance(index["metadata"], dict) or not isinstance(index["weight_map"], dict) or not index["weight_map"]:
        raise ValueError("checkpoint index metadata/weight_map is incomplete")
    values = list(index["weight_map"].values())
    if any(not isinstance(name, str) for name in values):
        raise ValueError("checkpoint index shard names are malformed")
    shards = sorted({safe_relative(name) for name in values})
    if len(shards) == 0 or any(name not in tree["files"] for name in shards):
        raise ValueError("checkpoint index references a shard absent from the authenticated HF tree")
    shard_records = []
    for name in shards:
        row = tree["files"][name]
        if row["lfs_sha256"] is None:
            raise ValueError(f"checkpoint shard has no authenticated LFS OID: {name}")
        shard_records.append({
            "path": name,
            "bytes": row["size"],
            "git_blob_sha1": row["git_blob_sha1"],
            "lfs_sha256": row["lfs_sha256"],
            "status": "IDENTITY_ONLY_NO_PAYLOAD",
        })
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "weight_map_entries": len(index["weight_map"]),
        "shards": shard_records,
        "payload_downloaded": False,
    }


def metadata_audit(root: Path, tree_path: Path, variant: str) -> dict[str, Any]:
    identity = VARIANTS[variant]
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"metadata root is not a real directory: {root}")
    cache = root / ".cache"
    if cache.exists() and (cache.is_symlink() or not cache.is_dir()):
        raise ValueError("metadata transport cache is not a real directory")
    visible = sorted(path.name for path in root.iterdir() if path.name != ".cache")
    if visible != sorted(METADATA_FILES):
        raise ValueError(f"metadata root contains non-allowlisted files: {visible}")
    tree = load_tree(tree_path, variant)
    files: dict[str, Any] = {}
    for relative, role in METADATA_FILES.items():
        safe_relative(relative)
        path = root / relative
        regular(path, f"model metadata {relative}")
        row = tree["files"].get(relative)
        if row is None or row["size"] != path.stat().st_size:
            raise ValueError(f"metadata tree identity mismatch: {relative}")
        payload_sha256 = sha256_file(path)
        if row["lfs_sha256"] is not None and row["lfs_sha256"] != payload_sha256:
            raise ValueError(f"metadata LFS identity mismatch: {relative}")
        blob = git_blob_sha1_file(path)
        files[relative] = {
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_sha1": row["git_blob_sha1"],
            "local_git_blob_sha1": blob,
            "lfs_sha256": row["lfs_sha256"],
            "status": (
                "AUTHENTICATED_BYTES_AND_LFS_OID"
                if row["lfs_sha256"] is not None
                else "AUTHENTICATED_BYTES_AND_REMOTE_GIT_BLOB"
            ),
        }
        if row["lfs_sha256"] is None and blob != row["git_blob_sha1"]:
            raise ValueError(f"metadata Git blob mismatch: {relative}")
    index = checkpoint_index(root / "model.safetensors.index.json", tree)
    return {
        "repository": identity["repository"],
        "revision": identity["revision"],
        "files": files,
        "checkpoint_index": index,
        "hf_tree": {
            "repository": tree["repository"],
            "revision": tree["revision"],
            "tree_sha256": tree["tree_sha256"],
            "card_data_sha256": tree["card_data_sha256"],
            "license_file": tree["license_file"],
        },
        "license_file": {
            "path": "LICENSE",
            "status": "ABSENT_AT_FIXED_REVISION",
            "spdx": None,
        },
        "license_card_data": {
            "source": "HF_MODEL_INFO_CARD_DATA",
            "field": "license",
            "value": tree["card_data"]["license"],
            "value_sha256": hashlib.sha256(tree["card_data"]["license"].encode()).hexdigest(),
            "spdx": None,
            "status": "PENDING_OWNER_APPROVAL",
        },
        "checkpoint_payload_downloaded": False,
    }


def collect_metadata_only(metadata_root: Path, tree_root: Path, variants: list[str]) -> None:
    """Fetch only metadata/index files and record the complete HF tree."""
    try:
        from huggingface_hub import HfApi, RepoFile, RepoFolder, hf_hub_download
    except ImportError as exc:
        raise ValueError(f"huggingface_hub is required for collection: {exc}") from exc
    metadata_root.mkdir(parents=True, exist_ok=True)
    tree_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for variant in variants:
        identity = VARIANTS[variant]
        info = api.model_info(identity["repository"], revision=identity["revision"])
        if info.sha != identity["revision"]:
            raise ValueError(f"HF resolved revision mismatch for {variant}: {info.sha}")
        model_card_data = normalize_card_data(info.card_data)
        rows = []
        for entry in api.list_repo_tree(identity["repository"], revision=identity["revision"], recursive=True):
            if isinstance(entry, RepoFolder):
                continue
            if not isinstance(entry, RepoFile):
                raise ValueError(f"unknown HF tree entry for {variant}: {entry!r}")
            relative = safe_relative(str(entry.path))
            size = getattr(entry, "size", None)
            blob = getattr(entry, "blob_id", None) or getattr(entry, "oid", None)
            lfs = getattr(entry, "lfs", None)
            lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
            if isinstance(lfs, dict):
                lfs_sha = lfs.get("sha256")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not HEX40.fullmatch(str(blob)):
                raise ValueError(f"HF tree identity is incomplete for {variant}: {relative}")
            if lfs_sha is not None and not HEX64.fullmatch(str(lfs_sha)):
                raise ValueError(f"HF LFS identity is malformed for {variant}: {relative}")
            rows.append({"path": relative, "type": "file", "size": size, "git_blob_sha1": blob, "lfs_sha256": lfs_sha})
        row_names = {row["path"] for row in rows}
        if len(row_names) != len(rows) or any(name not in row_names for name in METADATA_FILES):
            raise ValueError(f"HF metadata/index closure is incomplete for {variant}")
        if "LICENSE" in row_names:
            raise ValueError(f"HF model LICENSE is present despite absence evidence for {variant}")
        variant_root = metadata_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        for relative in METADATA_FILES:
            hf_hub_download(
                repo_id=identity["repository"],
                filename=relative,
                revision=identity["revision"],
                local_dir=str(variant_root),
            )
        tree_path = tree_root / f"{variant}.json"
        tree_path.write_bytes(canonical({
            "repository": identity["repository"],
            "revision": identity["revision"],
            "resolved_revision": info.sha,
            "files": rows,
            "card_data": model_card_data,
            "card_data_sha256": digest(model_card_data),
            "license_file": {"path": "LICENSE", "status": "ABSENT_AT_FIXED_REVISION"},
        }))


def build_manifest(source: dict[str, Any], variants: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "format": FORMAT,
        "status": "IDENTITY_AUDIT_COMPLETE",
        "no_model_weights": True,
        "approval": "PENDING_OWNER_APPROVAL",
        "publication": "NO_UPLOAD",
        "source": source,
        "variants": variants,
    }
    payload["manifest_sha256"] = digest(payload)
    return payload


def write_no_replace(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("output must be an absent absolute regular path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(canonical(value))
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="moss-audio-identity-") as temporary:
        root = Path(temporary)
        source = root / "source"
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "MOSS identity self-test"], check=True)
        for relative in SOURCE_FILES:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "identity"], check=True)
        subprocess.run(["git", "-C", str(source), "remote", "add", "origin", SOURCE_REPOSITORY], check=True)
        source_record = source_audit(source, expected_revision=git_output(source, "rev-parse", "HEAD"))
        assert source_record["license"]["status"] == "PENDING_OWNER_APPROVAL"
        assert source_record["license"]["spdx"] is None
        assert source_record["license_file"] == {
            "path": None,
            "status": "ABSENT_AT_FIXED_REVISION",
        }
        assert source_record["tracked_tree"]["files_sha256"] == digest(source_record["tracked_tree"]["files"])

        def assert_source_rejected(label: str, expected_revision: str) -> None:
            try:
                source_audit(source, expected_revision=expected_revision)
            except ValueError:
                return
            raise AssertionError(f"synthetic {label} source was accepted")

        assert_source_rejected("revision-drift", "4" * 40)
        (source / "LICENSE").write_text("synthetic license\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "add", "LICENSE"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "tracked license"], check=True)
        assert_source_rejected("tracked-license", git_output(source, "rev-parse", "HEAD"))
        snapshot = root / "snapshot"
        snapshot.mkdir()
        index = {"metadata": {}, "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}
        for relative in METADATA_FILES:
            path = snapshot / relative
            if relative == "model.safetensors.index.json":
                path.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")
            else:
                path.write_text(f"{relative}\n", encoding="utf-8")
        shard = "model-00001-of-00001.safetensors"
        tree_rows = []
        for relative in [*METADATA_FILES, shard]:
            path = snapshot / relative
            tree_rows.append({
                "path": relative,
                "type": "file",
                "size": path.stat().st_size if path.exists() else 4,
                "git_blob_sha1": "1" * 40,
                "lfs_sha256": "2" * 64 if relative == shard else None,
            })
        tree_path = root / "tree.json"
        model_card_data = {"license": "cc-by-4.0", "source": "synthetic"}
        tree_path.write_text(json.dumps({
            "repository": VARIANTS["4b"]["repository"],
            "revision": VARIANTS["4b"]["revision"],
            "resolved_revision": VARIANTS["4b"]["revision"],
            "files": tree_rows,
            "card_data": model_card_data,
            "card_data_sha256": digest(model_card_data),
            "license_file": {"path": "LICENSE", "status": "ABSENT_AT_FIXED_REVISION"},
        }), encoding="utf-8")
        # Replace metadata Git blobs with their actual local object IDs.
        tree = strict_json(tree_path)
        for row in tree["files"]:
            path = snapshot / row["path"]
            if path.exists():
                row["git_blob_sha1"] = git_blob_sha1_file(path)
        tree_path.write_text(json.dumps(tree), encoding="utf-8")
        record = metadata_audit(snapshot, tree_path, "4b")
        assert record["checkpoint_payload_downloaded"] is False
        assert record["checkpoint_index"]["shards"][0]["status"] == "IDENTITY_ONLY_NO_PAYLOAD"
        assert record["license_file"] == {
            "path": "LICENSE",
            "status": "ABSENT_AT_FIXED_REVISION",
            "spdx": None,
        }
        assert record["license_card_data"]["source"] == "HF_MODEL_INFO_CARD_DATA"
        assert record["license_card_data"]["status"] == "PENDING_OWNER_APPROVAL"

        def assert_rejected(value: dict[str, Any], label: str) -> None:
            candidate = root / f"{label}.json"
            candidate.write_text(json.dumps(value), encoding="utf-8")
            try:
                load_tree(candidate, "4b")
            except ValueError:
                return
            raise AssertionError(f"synthetic {label} tree was accepted")

        false_license = json.loads(tree_path.read_text(encoding="utf-8"))
        false_license["files"].append({
            "path": "LICENSE",
            "type": "file",
            "size": 1,
            "git_blob_sha1": "3" * 40,
            "lfs_sha256": None,
        })
        assert_rejected(false_license, "false-license")
        card_data_mismatch = json.loads(tree_path.read_text(encoding="utf-8"))
        card_data_mismatch["card_data"]["license"] = "Apache-2.0"
        assert_rejected(card_data_mismatch, "card-data-mismatch")
        revision_drift = json.loads(tree_path.read_text(encoding="utf-8"))
        revision_drift["revision"] = "4" * 40
        assert_rejected(revision_drift, "revision-drift")
        manifest = build_manifest(source_record, {"4b": record})
        assert manifest["approval"] == "PENDING_OWNER_APPROVAL"
    print("moss_audio identity audit self-test PASS (no model payload)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--tree-root", type=Path)
    parser.add_argument("--variant", choices=["4b", "8b", "all"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        if args.collect or any(value is not None for value in (args.source_dir, args.metadata_root, args.tree_root, args.variant, args.output)):
            raise SystemExit("--self-test accepts no audit inputs")
        self_test()
        return 0
    if any(value is None for value in (args.source_dir, args.metadata_root, args.tree_root, args.variant, args.output)):
        raise SystemExit("normal audit requires source, metadata, tree root, variant, and output")
    variants = ["4b", "8b"] if args.variant == "all" else [args.variant]
    try:
        if args.collect:
            if args.source_dir.exists():
                raise ValueError("source target must be absent before collection")
            if args.metadata_root.exists() or args.tree_root.exists():
                raise ValueError("metadata/tree targets must be absent before collection")
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPOSITORY, str(args.source_dir)], check=True)
            subprocess.run(["git", "-C", str(args.source_dir), "checkout", "--detach", SOURCE_REVISION], check=True)
            collect_metadata_only(args.metadata_root, args.tree_root, variants)
        source = source_audit(args.source_dir)
        records = {
            variant: metadata_audit(args.metadata_root / variant, args.tree_root / f"{variant}.json", variant)
            for variant in variants
        }
        write_no_replace(args.output, build_manifest(source, records))
    except Exception as exc:  # noqa: BLE001 - audit must fail closed
        print(f"MOSS_AUDIO_IDENTITY_AUDIT_BLOCKED: {exc}")
        return 2
    print(f"MOSS_AUDIO_IDENTITY_AUDIT_COMPLETE: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
