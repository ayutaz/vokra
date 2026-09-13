#!/usr/bin/env -S uv run --frozen --project tools/parity/zonos_v0_1_reference --python 3.12 python
"""VAST-only fixed revision staging for the Zonos inspection worker.

This helper downloads only immutable snapshots and emits typed server-tree
packets. It never converts or executes a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from zonos_v0_1_reference import dependency_audit

PUBLIC_REPOSITORY = "vokra/zonos-v0.1-transformer"
PUBLIC_REVISION = "b1bf5c56d470eb9097e9b04f9deca364576574ba"
SOURCE_REPOSITORY = "https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION = "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
UPSTREAM_REPOSITORY = "Zyphra/Zonos-v0.1-transformer"
UPSTREAM_REVISION = "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e"
MANIFEST_SHA256 = "6543af3747d3e85bde862c3337744eea31f0105f9df6d8617c1c9afdae805847"
DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
PROJECT_PATH = dependency_audit.PROJECT_PATH
LOCK_PATH = dependency_audit.LOCK_PATH
SOURCE_LICENSE_PATH = "LICENSE"
SOURCE_LICENSE_SPDX = "Apache-2.0"
SOURCE_LICENSE_BYTES = 11_357
SOURCE_LICENSE_GIT_BLOB_SHA1 = "7a4a3ea2424c09fbe48d455aed1eaa94d9124835"
SOURCE_LICENSE_SHA256 = "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd"
UPSTREAM_MODEL_ID = UPSTREAM_REPOSITORY
UPSTREAM_MODEL_PRIVATE = False
UPSTREAM_MODEL_GATED = False
UPSTREAM_MODEL_DISABLED = False
UPSTREAM_MODEL_CARD_DATA_LICENSE = "apache-2.0"


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_pointer(oid: str, size: int) -> str:
    payload = ("version https://git-lfs.github.com/spec/v1\n"
               f"oid sha256:{oid}\nsize {size}\n").encode()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"approval evidence is not valid duplicate-free JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("approval evidence must be a JSON object")
    return value


def info_field(info: Any, name: str) -> Any:
    """Read both huggingface_hub objects and the raw API-shaped test doubles."""
    if isinstance(info, dict):
        return info.get(name)
    return getattr(info, name, None)


def card_data_license(info: Any) -> Any:
    card_data = info_field(info, "card_data")
    if card_data is None:
        card_data = info_field(info, "cardData")
    if isinstance(card_data, dict):
        return card_data.get("license")
    return getattr(card_data, "license", None)


def license_identity_contract() -> dict[str, Any]:
    """Return the immutable, primary-source license facts used by every gate."""
    return {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_LICENSE_PATH,
            "spdx": SOURCE_LICENSE_SPDX,
            "bytes": SOURCE_LICENSE_BYTES,
            "git_blob_sha1": SOURCE_LICENSE_GIT_BLOB_SHA1,
            "sha256": SOURCE_LICENSE_SHA256,
        },
        "upstream_model": {
            "id": UPSTREAM_MODEL_ID,
            "revision": UPSTREAM_REVISION,
            "private": UPSTREAM_MODEL_PRIVATE,
            "gated": UPSTREAM_MODEL_GATED,
            "disabled": UPSTREAM_MODEL_DISABLED,
            "card_data_license": UPSTREAM_MODEL_CARD_DATA_LICENSE,
        },
    }


def validate_license_identity_contract() -> None:
    """Fail closed if the reviewed primary-source identity is edited or absent."""
    expected = {
        "source": {
            "repository": "https://github.com/Zyphra/Zonos.git",
            "revision": SOURCE_REVISION,
            "path": "LICENSE",
            "spdx": "Apache-2.0",
            "bytes": 11_357,
            "git_blob_sha1": "7a4a3ea2424c09fbe48d455aed1eaa94d9124835",
            "sha256": "58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd",
        },
        "upstream_model": {
            "id": "Zyphra/Zonos-v0.1-transformer",
            "revision": "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
            "private": False,
            "gated": False,
            "disabled": False,
            "card_data_license": "apache-2.0",
        },
    }
    if license_identity_contract() != expected:
        raise RuntimeError("Zonos primary-source license identity contract drifted")


def model_license_identity(info: Any) -> dict[str, Any]:
    """Authenticate the exact public HF model-info fields before acquisition."""
    actual = {
        "id": info_field(info, "id"),
        "revision": info_field(info, "sha"),
        "private": info_field(info, "private"),
        "gated": info_field(info, "gated"),
        "disabled": info_field(info, "disabled"),
        "card_data_license": card_data_license(info),
    }
    expected = license_identity_contract()["upstream_model"]
    if actual != expected:
        raise RuntimeError(f"upstream HF model license identity mismatch: {actual!r}")
    return actual


def source_license_identity(source: Path) -> dict[str, Any]:
    """Authenticate the exact LICENSE blob in a checked-out upstream source."""
    path = source / SOURCE_LICENSE_PATH
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Zyphra/Zonos source LICENSE is missing or symlinked")
    actual = {
        "path": SOURCE_LICENSE_PATH,
        "spdx": SOURCE_LICENSE_SPDX,
        "bytes": path.stat().st_size,
        "git_blob_sha1": git_blob(path),
        "sha256": sha256(path),
    }
    expected = license_identity_contract()["source"]
    if any(actual[key] != expected[key] for key in actual):
        raise RuntimeError(f"Zyphra/Zonos source LICENSE identity mismatch: {actual!r}")
    return actual


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"path contains a symlink ancestor: {path}")


def validate_root(root: Path) -> None:
    if not root.is_absolute():
        raise RuntimeError("--root must be an absolute path")
    reject_symlink_ancestors(root)
    if root.exists() or root.is_symlink():
        raise RuntimeError("--root must be an absent, non-symlink directory")
    parent = root.parent
    suffix: list[str] = []
    while not parent.exists():
        if parent == parent.parent:
            raise RuntimeError("--root has no existing parent")
        suffix.append(parent.name)
        parent = parent.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("--root nearest existing parent is unsafe")
    candidate = parent.resolve()
    for component in reversed(suffix):
        candidate /= component
    candidate /= root.name
    checkout = Path(__file__).resolve().parents[2]
    if candidate == checkout or checkout in candidate.parents or candidate in checkout.parents:
        raise RuntimeError("--root overlaps the repository checkout")


def claim_root(root: Path) -> None:
    """Atomically claim the validated staging leaf before any network work."""
    parent = root.parent
    if not parent.exists() and not parent.is_symlink():
        parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_ancestors(root)
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("staging root parent is not a real directory")
    try:
        root.mkdir()
    except FileExistsError as error:
        raise RuntimeError("staging root was concurrently claimed or already exists") from error
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("staging root claim did not produce a real directory")


def validate_staged_paths(root: Path, inputs: list[Path], outputs: list[Path]) -> None:
    """Bind optional paths to the newly staged root before reading/writing."""
    root_real = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("staged root is not a real directory")
    resolved: list[Path] = []
    for path in inputs + outputs:
        reject_symlink_ancestors(path)
        candidate = path.resolve(strict=False)
        if candidate != root_real and root_real not in candidate.parents:
            raise RuntimeError(f"optional path is outside staged root: {path}")
        if path in outputs:
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"output already exists or is symlinked: {path}")
        elif not path.is_file() or path.is_symlink():
            raise RuntimeError(f"input is not a regular non-symlink file: {path}")
        if candidate in resolved:
            raise RuntimeError(f"optional input/output paths overlap: {path}")
        resolved.append(candidate)


def validate_optional_paths_before_network(root: Path, inputs: list[Path], outputs: list[Path]) -> None:
    """Reject unsafe optional path names before either snapshot is fetched."""
    root_candidate = root.resolve(strict=False)
    for path in inputs + outputs:
        if not path.is_absolute():
            raise RuntimeError(f"optional path must be absolute: {path}")
        reject_symlink_ancestors(path)
        candidate = path.resolve(strict=False)
        if candidate != root_candidate and root_candidate not in candidate.parents:
            raise RuntimeError(f"optional path is outside staged root: {path}")
        if path in outputs and (path.exists() or path.is_symlink()):
            raise RuntimeError(f"output already exists or is symlinked: {path}")


def verify_local_snapshot(destination: Path, rows: list[dict[str, Any]]) -> None:
    """Verify the downloaded tree against the authenticated server rows."""
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeError(f"snapshot destination is not a real directory: {destination}")
    expected = {row["path"]: row for row in rows}
    actual: dict[str, Path] = {}
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"downloaded snapshot contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"downloaded snapshot contains a non-regular entry: {relative}")
        # Hugging Face may leave transport metadata under this one directory;
        # it is not part of the authenticated model tree, but it is still
        # required to be regular and non-symlinked by the checks above.
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        if relative in actual:
            raise RuntimeError(f"duplicate downloaded snapshot path: {relative}")
        actual[relative] = path
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise RuntimeError(f"downloaded snapshot tree mismatch: missing={missing!r} extra={extra!r}")
    for name, row in expected.items():
        path = actual[name]
        if path.stat().st_size != row["size"]:
            raise RuntimeError(f"downloaded snapshot size mismatch: {name}")
        if row.get("lfs_sha256") is not None:
            actual_digest = sha256_file(path)
            if actual_digest != row["lfs_sha256"]:
                raise RuntimeError(f"downloaded snapshot payload mismatch: {name}")
        elif git_blob(path) != row["git_blob_sha1"]:
            raise RuntimeError(f"downloaded snapshot Git blob mismatch: {name}")


def approval_scope(project_sha256: str, lock_sha256: str) -> dict[str, Any]:
    validate_license_identity_contract()
    return {
        "schema": "zonos-vast-approval-scope-v1",
        "project_sha256": project_sha256,
        "lock_sha256": lock_sha256,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "public_repository": PUBLIC_REPOSITORY,
        "public_revision": PUBLIC_REVISION,
        "no_upload": True,
        "license_review": "AUTHENTICATED_LICENSE_IDENTITY_REQUIRED",
        "license_identity": license_identity_contract(),
    }


def preflight_gate(approval: Path) -> None:
    # This is standard-library-only and must run before any HF/source access.
    # It rejects a widened or forbidden closure while leaving the owner/legal
    # decision in the external exact-hash approval record below.
    dependency_audit.audit()
    validate_license_identity_contract()
    if not approval.is_file() or approval.is_symlink():
        raise RuntimeError("--approval-evidence must be a regular non-symlink file")
    reject_symlink_ancestors(approval)
    if not PROJECT_PATH.is_file() or not LOCK_PATH.is_file():
        raise RuntimeError("dedicated project lock is missing")
    project_sha256 = sha256_file(PROJECT_PATH)
    lock_sha256 = sha256_file(LOCK_PATH)
    value = json_file(approval)
    expected = {
        "schema", "decision", "signer", "project_sha256", "lock_sha256",
        "scope_sha256", "no_upload", "source_repository", "source_revision",
        "upstream_repository", "upstream_revision", "public_repository",
        "public_revision", "license_review",
    }
    if set(value) != expected:
        raise RuntimeError("approval schema is not exact")
    if value["schema"] != "zonos-vast-approval-v1" or value["decision"] != "APPROVED":
        raise RuntimeError("approval decision is not APPROVED")
    signer = value["signer"]
    if not isinstance(signer, str) or not signer.strip() or signer.strip().upper() in {"TODO", "UNRESOLVED", "PENDING", "OWNER_REVIEW_REQUIRED"}:
        raise RuntimeError("approval signer is unresolved")
    if value["project_sha256"] != project_sha256 or value["lock_sha256"] != lock_sha256:
        raise RuntimeError("approval project/lock identity mismatch")
    scope = approval_scope(project_sha256, lock_sha256)
    scope_digest = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value["scope_sha256"] != scope_digest:
        raise RuntimeError("approval scope digest mismatch")
    expected_values = {
        "source_repository": scope["source_repository"], "source_revision": scope["source_revision"],
        "upstream_repository": UPSTREAM_REPOSITORY, "upstream_revision": UPSTREAM_REVISION,
        "public_repository": PUBLIC_REPOSITORY, "public_revision": PUBLIC_REVISION,
        "license_review": scope["license_review"],
    }
    if any(value[key] != expected for key, expected in expected_values.items()) or value["no_upload"] is not True:
        raise RuntimeError("approval fixed identity/publication fields mismatch")
    # The exact source LICENSE blob and HF cardData identity are included in
    # the scope digest above.  This keeps the external approval binding
    # fail-closed while avoiding a mutable/unconditional boolean gate.


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = bytearray()
    for item in sorted(rows, key=lambda value: value["name"]):
        shape = item["shape"]
        canonical.extend(item["name"].encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def expected_tensor_roles() -> list[dict[str, Any]]:
    """Return the fixed native Zonos transformer role/shape contract.

    This is deliberately duplicated from the converter-side contract: the
    VAST staging worker must reject a manifest before any artifact is handed
    to the runtime, and importing workspace Rust here would violate the
    model-free staging boundary.
    """
    d_model = 2048
    roles: list[dict[str, Any]] = []

    def add(name: str, shape: list[int]) -> None:
        roles.append({"name": name, "shape": shape})

    for codebook in range(9):
        add(f"embeddings.{codebook}.weight", [1026, d_model])
        add(f"heads.{codebook}.weight", [1025, d_model])
    add("backbone.norm_f.weight", [d_model])
    add("backbone.norm_f.bias", [d_model])
    for layer in range(26):
        prefix = f"backbone.layers.{layer}"
        add(f"{prefix}.norm.weight", [d_model])
        add(f"{prefix}.norm.bias", [d_model])
        add(f"{prefix}.mixer.in_proj.weight", [3072, d_model])
        add(f"{prefix}.mixer.out_proj.weight", [d_model, d_model])
        add(f"{prefix}.norm2.weight", [d_model])
        add(f"{prefix}.norm2.bias", [d_model])
        add(f"{prefix}.mlp.fc1.weight", [16384, d_model])
        add(f"{prefix}.mlp.fc2.weight", [d_model, 8192])
    add("prefix_conditioner.conditioners.0.phoneme_embedder.weight", [189, d_model])
    add("prefix_conditioner.conditioners.1.project.weight", [d_model, 128])
    add("prefix_conditioner.conditioners.1.project.bias", [d_model])
    add("prefix_conditioner.conditioners.1.uncond_vector", [d_model])
    for index, input_dim in ((2, 8), (3, 1), (4, 1), (5, 1)):
        add(f"prefix_conditioner.conditioners.{index}.weight", [1024, input_dim])
        add(f"prefix_conditioner.conditioners.{index}.uncond_vector", [d_model])
    add("prefix_conditioner.conditioners.6.int_embedder.weight", [128, d_model])
    add("prefix_conditioner.conditioners.6.uncond_vector", [d_model])
    add("prefix_conditioner.project.weight", [d_model, d_model])
    add("prefix_conditioner.project.bias", [d_model])
    add("prefix_conditioner.norm.weight", [d_model])
    add("prefix_conditioner.norm.bias", [d_model])
    if len(roles) != 246:
        raise AssertionError(f"native Zonos role contract has {len(roles)} entries")
    return roles


def validate_tensor_rows(rows: list[dict[str, Any]], label: str) -> None:
    expected = sorted(expected_tensor_roles(), key=lambda row: row["name"])
    actual = sorted(
        ({"name": row["name"], "shape": row["shape"]} for row in rows),
        key=lambda row: row["name"],
    )
    if actual != expected:
        raise RuntimeError(f"{label} manifest roles/shapes do not match the fixed native 246-tensor contract")
    if any(row.get("dtype") not in FLOAT_DTYPES for row in rows):
        raise RuntimeError(f"{label} manifest contains a non-dense-float tensor dtype")


def server_file_row(item: Any) -> dict[str, Any]:
    """Normalize an expanded huggingface_hub ``RepoFile`` row."""
    if getattr(item, "type", None) != "file":
        raise RuntimeError("server tree contains a non-file entry")
    path = getattr(item, "path", None)
    size = getattr(item, "size", None)
    blob_id = getattr(item, "blob_id", None)
    if (not isinstance(path, str) or not path or path.startswith("/")
            or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))):
        raise RuntimeError(f"invalid server file path: {path!r}")
    if (not isinstance(size, int) or isinstance(size, bool) or size < 0
            or not isinstance(blob_id, str) or not re.fullmatch(r"[0-9a-f]{40}", blob_id)):
        raise RuntimeError(f"invalid server file identity: {path}")
    row: dict[str, Any] = {"type": "file", "path": path,
                           "size": size, "git_blob_sha1": blob_id}
    lfs = getattr(item, "lfs", None)
    if lfs is not None:
        if isinstance(lfs, dict):
            lfs_oid = lfs.get("sha256") or lfs.get("oid")
            lfs_size = lfs.get("size", size)
        else:
            lfs_oid = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            lfs_size = getattr(lfs, "size", size)
        if (not isinstance(lfs_size, int) or isinstance(lfs_size, bool)
                or not isinstance(lfs_oid, str)
                or not re.fullmatch(r"[0-9a-f]{64}", lfs_oid)
                or lfs_size != size or lfs_pointer(lfs_oid, size) != blob_id):
            raise RuntimeError(f"invalid LFS identity: {path}")
        row.update({"lfs_sha256": lfs_oid, "lfs_size": lfs_size})
    return row


def server_tree_packet(
    repository: str,
    revision: str,
    resolved_revision: str,
    rows: list[dict[str, Any]],
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable public packet and the authenticated upstream variant."""
    if repository == UPSTREAM_REPOSITORY:
        if model_info is None:
            raise RuntimeError("upstream server-tree packet requires model_info")
    elif model_info is not None:
        raise RuntimeError("public server-tree packet must not contain model_info")
    packet: dict[str, Any] = {
        "repository": repository,
        "revision": revision,
        "resolved_revision": resolved_revision,
        "walk": "recursive_file_only",
        "complete_recursive": True,
        "files": sorted(rows, key=lambda row: row["path"]),
    }
    if repository == UPSTREAM_REPOSITORY:
        packet["model_info"] = model_info
    return packet


def stage_snapshot(label: str, repository: str, revision: str, root: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(repository, revision=revision)
    if info.sha != revision:
        raise RuntimeError(f"{repository}: resolved {info.sha} != {revision}")
    model_info: dict[str, Any] | None = None
    if repository == UPSTREAM_REPOSITORY:
        model_info = model_license_identity(info)
    rows: list[dict[str, Any]] = []
    for item in api.list_repo_tree(repository, revision=revision, recursive=True, expand=True):
        if getattr(item, "type", None) != "file":
            continue
        row = server_file_row(item)
        rows.append(row)
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("server tree is empty or duplicated")
    destination = root / label
    downloaded = Path(snapshot_download(repo_id=repository, revision=revision, local_dir=destination))
    if downloaded.resolve() != destination.resolve():
        raise RuntimeError(f"snapshot_download returned an unexpected local path: {downloaded}")
    verify_local_snapshot(destination, rows)
    packet = server_tree_packet(repository, revision, info.sha, rows, model_info)
    (root / f"{label}-server-tree.json").write_text(
        json.dumps(packet, sort_keys=True) + "\n", encoding="utf-8")


def safetensors_manifest(path: Path, output: Path, revision: str) -> None:
    with path.open("rb") as handle:
        size = path.stat().st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise RuntimeError("safetensors prefix is truncated")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 2 or header_size > 64 * 1024 * 1024 or header_size > size - 8:
            raise RuntimeError("unsafe safetensors header length")
        header = json.loads(handle.read(header_size), object_pairs_hook=unique)
    if not isinstance(header, dict):
        raise RuntimeError("safetensors header is not an object")
    rows = []
    occupied: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name or ".." in name.split("/"):
            raise RuntimeError(f"unsafe tensor name: {name!r}")
        if (not isinstance(descriptor, dict)
                or not isinstance(descriptor.get("shape"), list)
                or not isinstance(descriptor.get("dtype"), str)
                or not isinstance(descriptor.get("data_offsets"), list)
                or len(descriptor["data_offsets"]) != 2):
            raise RuntimeError(f"malformed descriptor: {name}")
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if (descriptor["dtype"] not in DTYPE_BYTES
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
                or any(isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 for offset in offsets)
                or offsets[1] < offsets[0]):
            raise RuntimeError(f"unsafe descriptor values: {name}")
        elements = 1
        for dim in shape:
            elements *= dim
        expected_bytes = elements * DTYPE_BYTES[descriptor["dtype"]]
        if expected_bytes != offsets[1] - offsets[0] or offsets[1] > size - 8 - header_size:
            raise RuntimeError(f"descriptor range mismatch: {name}")
        occupied.append((offsets[0], offsets[1], name))
        rows.append({"name": name, "shape": descriptor["shape"],
                     "dtype": descriptor["dtype"]})
    occupied.sort()
    body_size = size - 8 - header_size
    if occupied and (occupied[0][0] != 0 or occupied[-1][1] != body_size):
        raise RuntimeError("safetensors body has an unaccounted gap")
    for previous, current in zip(occupied, occupied[1:]):
        if current[0] != previous[1]:
            raise RuntimeError(f"non-contiguous descriptor ranges: {previous[2]}, {current[2]}")
    rows.sort(key=lambda row: row["name"])
    validate_tensor_rows(rows, "upstream safetensors")
    digest = manifest_sha256(rows)
    if len(rows) != 246 or digest != MANIFEST_SHA256:
        raise RuntimeError(f"upstream safetensors manifest mismatch: {digest}")
    payload = json.dumps({"revision": revision,
                          "manifest_sha256": digest,
                          "tensors": rows}, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def gguf_manifest(path: Path, output: Path, revision: str) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    rows = sorted(
        [{"name": tensor.name, "shape": [int(x) for x in tensor.shape],
          "dtype": getattr(tensor.tensor_type, "name", str(tensor.tensor_type).rsplit(".", 1)[-1])}
         for tensor in reader.tensors],
        key=lambda row: row["name"],
    )
    validate_tensor_rows(rows, "public GGUF")
    digest = manifest_sha256(rows)
    if len(rows) != 246 or digest != MANIFEST_SHA256:
        raise RuntimeError(f"public GGUF manifest mismatch: {digest}")
    payload = json.dumps({"revision": revision,
                          "manifest_sha256": digest,
                          "tensors": rows}, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--upstream-safetensors", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--public-gguf", type=Path)
    parser.add_argument("--public-manifest-output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for option in ("--root", "--approval-evidence", "--upstream-safetensors",
                   "--manifest-output", "--public-gguf", "--public-manifest-output",
                   "--preflight-only", "--self-test"):
        if sys.argv[1:].count(option) > 1:
            parser.error(f"duplicate {option}")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.root, args.approval_evidence,
                                                args.upstream_safetensors, args.manifest_output,
                                                args.public_gguf, args.public_manifest_output)) or args.preflight_only:
            parser.error("--self-test accepts no other arguments")
        assert len(PUBLIC_REVISION) == len(UPSTREAM_REVISION) == 40
        assert len(MANIFEST_SHA256) == 64
        role_rows = [dict(row, dtype="F32") for row in expected_tensor_roles()]
        assert len(role_rows) == 246
        assert manifest_sha256(role_rows) == MANIFEST_SHA256
        validate_tensor_rows(role_rows, "self-test")
        role_rows[0]["dtype"] = "I32"
        try:
            validate_tensor_rows(role_rows, "self-test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("integer tensor dtype must fail the native dense-float contract")
        role_rows[0]["dtype"] = "F32"
        role_rows[0]["shape"] = [1]
        try:
            validate_tensor_rows(role_rows, "self-test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("role/shape drift must fail closed")
        validate_license_identity_contract()
        from types import SimpleNamespace
        valid_info = SimpleNamespace(
            id=UPSTREAM_REPOSITORY,
            sha=UPSTREAM_REVISION,
            private=False,
            gated=False,
            disabled=False,
            card_data=SimpleNamespace(license=UPSTREAM_MODEL_CARD_DATA_LICENSE),
        )
        assert model_license_identity(valid_info)["card_data_license"] == "apache-2.0"
        valid_info.card_data.license = "mit"
        try:
            model_license_identity(valid_info)
        except RuntimeError:
            pass
        else:
            raise AssertionError("HF cardData license drift must fail closed")
        valid_info.card_data.license = UPSTREAM_MODEL_CARD_DATA_LICENSE
        packet_rows = [{"path": "config.json", "size": 1}]
        public_packet = server_tree_packet(
            PUBLIC_REPOSITORY, PUBLIC_REVISION, PUBLIC_REVISION, packet_rows,
        )
        assert set(public_packet) == {
            "repository", "revision", "resolved_revision", "walk",
            "complete_recursive", "files",
        }
        upstream_packet = server_tree_packet(
            UPSTREAM_REPOSITORY, UPSTREAM_REVISION, UPSTREAM_REVISION,
            packet_rows, model_license_identity(valid_info),
        )
        assert set(upstream_packet) == {
            "repository", "revision", "resolved_revision", "walk",
            "complete_recursive", "files", "model_info",
        }
        for repository, info in (
            (PUBLIC_REPOSITORY, model_license_identity(valid_info)),
            (UPSTREAM_REPOSITORY, None),
        ):
            try:
                server_tree_packet(repository, PUBLIC_REVISION, PUBLIC_REVISION,
                                   packet_rows, info)
            except RuntimeError:
                pass
            else:
                raise AssertionError("server-tree packet schema drift must fail closed")
        with TemporaryDirectory(prefix="zonos-license-selftest-") as license_directory:
            try:
                source_license_identity(Path(license_directory))
            except RuntimeError:
                pass
            else:
                raise AssertionError("missing source LICENSE must fail closed")
        original_license_sha = SOURCE_LICENSE_SHA256
        globals()["SOURCE_LICENSE_SHA256"] = "0" * 64
        try:
            validate_license_identity_contract()
        except RuntimeError:
            pass
        else:
            raise AssertionError("license contract drift must fail closed")
        finally:
            globals()["SOURCE_LICENSE_SHA256"] = original_license_sha
        oid = "0" * 64
        assert len(lfs_pointer(oid, 1)) == 40
        assert lfs_pointer(oid, 1) != hashlib.sha1(b"x").hexdigest()
        valid_row = SimpleNamespace(type="file", path="weights/model.safetensors",
                                    size=1, blob_id="0" * 40, lfs=None)
        assert server_file_row(valid_row)["git_blob_sha1"] == "0" * 40
        for bad_path in ("", "/absolute", "../escape", "a/../b", "a\\b"):
            try:
                server_file_row(SimpleNamespace(type="file", path=bad_path,
                                                size=1, blob_id="0" * 40, lfs=None))
            except RuntimeError:
                pass
            else:
                raise AssertionError("unsafe server paths must fail closed")
        for bad_blob in ("0" * 39, "A" * 40):
            try:
                server_file_row(SimpleNamespace(type="file", path="x", size=1,
                                                blob_id=bad_blob, lfs=None))
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid server blob IDs must fail closed")
        try:
            json.loads('{"x":1,"x":2}', object_pairs_hook=unique)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate approval JSON keys must fail")
        with TemporaryDirectory(prefix="zonos-stage-selftest-") as directory:
            snapshot = Path(directory).resolve()
            payload = snapshot / "payload.bin"
            payload.write_bytes(b"x")
            verify_local_snapshot(snapshot, [{
                "path": "payload.bin", "size": 1,
                "git_blob_sha1": git_blob(payload),
            }])
            (snapshot / "extra.bin").write_bytes(b"x")
            try:
                verify_local_snapshot(snapshot, [{
                    "path": "payload.bin", "size": 1,
                    "git_blob_sha1": git_blob(payload),
                }])
            except RuntimeError:
                pass
            else:
                raise AssertionError("extra downloaded files must fail closed")
            (snapshot / "extra.bin").unlink()
            (snapshot / "linked.bin").symlink_to(payload)
            try:
                verify_local_snapshot(snapshot, [{
                    "path": "payload.bin", "size": 1,
                    "git_blob_sha1": git_blob(payload),
                }])
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlinked downloaded files must fail closed")
            safe_root = snapshot / "nested" / "stage"
            validate_root(safe_root)
            claim_root(safe_root)
            assert safe_root.is_dir() and not safe_root.is_symlink()
            try:
                claim_root(safe_root)
            except RuntimeError:
                pass
            else:
                raise AssertionError("competing staging root claim must fail closed")
            (snapshot / "occupied").mkdir()
            try:
                validate_root(snapshot / "occupied")
            except RuntimeError:
                pass
            else:
                raise AssertionError("existing staging roots must fail closed")
            (snapshot / "real").mkdir()
            (snapshot / "link").symlink_to(snapshot / "real", target_is_directory=True)
            try:
                validate_root(snapshot / "link" / "nested")
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlinked staging ancestors must fail closed")
            staged = snapshot / "staged"
            staged.mkdir()
            staged_input = staged / "input.bin"
            staged_input.write_bytes(b"input")
            staged_output = staged / "manifest.json"
            validate_optional_paths_before_network(staged, [staged_input], [staged_output])
            try:
                validate_optional_paths_before_network(staged, [snapshot / "outside.bin"], [])
            except RuntimeError:
                pass
            else:
                raise AssertionError("optional paths outside the stage must fail closed")
            validate_staged_paths(staged, [staged_input], [staged_output])
            staged_output.write_bytes(b"old")
            try:
                validate_staged_paths(staged, [staged_input], [staged_output])
            except RuntimeError:
                pass
            else:
                raise AssertionError("existing staged outputs must fail closed")
            staged_output.unlink()
            (staged / "input-link").symlink_to(staged_input)
            try:
                validate_staged_paths(staged, [staged / "input-link"], [])
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlinked staged inputs must fail closed")
        approval_parent = "/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
        with TemporaryDirectory(prefix="zonos-approval-selftest-", dir=approval_parent) as approval_directory:
            approval = Path(approval_directory) / "approval.json"
            project_sha = sha256_file(PROJECT_PATH)
            lock_sha = sha256_file(LOCK_PATH)
            scope = approval_scope(project_sha, lock_sha)
            approval.write_text(json.dumps({
                "schema": "zonos-vast-approval-v1",
                "decision": "APPROVED",
                "signer": "self-test",
                "project_sha256": project_sha,
                "lock_sha256": lock_sha,
                "scope_sha256": hashlib.sha256(json.dumps(
                    scope, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest(),
                "no_upload": True,
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_revision": UPSTREAM_REVISION,
                "public_repository": PUBLIC_REPOSITORY,
                "public_revision": PUBLIC_REVISION,
                "license_review": "AUTHENTICATED_LICENSE_IDENTITY_REQUIRED",
            }), encoding="utf-8")
            preflight_gate(approval)
            approval_value = json.loads(approval.read_text(encoding="utf-8"))
            approval_value["scope_sha256"] = "0" * 64
            approval.write_text(json.dumps(approval_value), encoding="utf-8")
            try:
                preflight_gate(approval)
            except RuntimeError:
                pass
            else:
                raise AssertionError("approval scope drift must fail closed")
        print("zonos_vast_stage.py self-test: OK")
        return 0
    if args.preflight_only:
        if args.approval_evidence is None or any(value is not None for value in (
                args.root, args.upstream_safetensors, args.manifest_output,
                args.public_gguf, args.public_manifest_output)):
            parser.error("--preflight-only accepts only --approval-evidence")
        try:
            preflight_gate(args.approval_evidence)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"zonos VAST preflight BLOCKED: {error}", file=sys.stderr)
            return 2
        return 0
    if args.root is None or args.approval_evidence is None:
        parser.error("--root and --approval-evidence are required for staging")
    try:
        # This gate intentionally runs before importing huggingface_hub or
        # touching the requested output root.  The external approval scope is
        # bound to the authenticated source LICENSE and HF model identity.
        preflight_gate(args.approval_evidence)
        validate_root(args.root)
        claim_root(args.root)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"zonos VAST preflight BLOCKED: {error}", file=sys.stderr)
        return 2
    if (args.upstream_safetensors is None) != (args.manifest_output is None):
        parser.error("safetensors and manifest output must be supplied together")
    if (args.public_gguf is None) != (args.public_manifest_output is None):
        parser.error("public GGUF and manifest output must be supplied together")
    optional_inputs = [path for path in (args.upstream_safetensors, args.public_gguf) if path is not None]
    optional_outputs = [path for path in (args.manifest_output, args.public_manifest_output) if path is not None]
    try:
        validate_optional_paths_before_network(args.root, optional_inputs, optional_outputs)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"zonos VAST optional-path BLOCKED: {error}", file=sys.stderr)
        return 2
    for output in optional_outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    stage_snapshot("public", PUBLIC_REPOSITORY, PUBLIC_REVISION, args.root)
    stage_snapshot("upstream", UPSTREAM_REPOSITORY, UPSTREAM_REVISION, args.root)
    try:
        validate_staged_paths(args.root, optional_inputs, optional_outputs)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"zonos VAST staged-path BLOCKED: {error}", file=sys.stderr)
        return 2
    if args.upstream_safetensors is not None and args.manifest_output is not None:
        safetensors_manifest(args.upstream_safetensors, args.manifest_output,
                              UPSTREAM_REVISION)
    if args.public_gguf is not None and args.public_manifest_output is not None:
        gguf_manifest(args.public_gguf, args.public_manifest_output, PUBLIC_REVISION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
