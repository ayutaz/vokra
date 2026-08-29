#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect the official HT-Demucs ensemble without producing a GGUF.

The ``.th`` files are treated as untrusted inputs.  Only PyTorch's
``weights_only=True`` loader is attempted; there is deliberately no pickle
fallback.  This tool records the source configs, member identity, and any
safe-load tensor manifest, while keeping the overall result BLOCKED/INSPECTION_ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
import yaml


UPSTREAM_REPOSITORY = "facebookresearch/demucs"
UPSTREAM_URL = "https://github.com/facebookresearch/demucs"
UPSTREAM_REVISION = "e976d93ecc3865e5757426930257e200846a520a"
WEIGHT_ROOT = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
MEMBERS = {
    "f7e0c4bc": "f7e0c4bc-ba3fe64a.th",
    "d12395a8": "d12395a8-e57c48e6.th",
    "92cfc3b6": "92cfc3b6-ef3bcb9c.th",
    "04573f0d": "04573f0d-f3cf25b2.th",
    "5c90dfd2": "5c90dfd2-34c22ccb.th",
}
FT_MODELS = ["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"]
SIX_MODELS = ["5c90dfd2"]
SOURCE_ROLE_BLOBS = {
    "LICENSE": "a45a376fb0fcd4a3de06b6c096e620289a2dcb",
    "demucs/apply.py": "1540f3d44fc8bbca1ce377cc80af3cd9212278be",
    "demucs/hdemucs.py": "711d47157a975e04a0ffb044991d2dc3cfd54b66",
    "demucs/htdemucs.py": "5d2eaaa1eb2620a5d2147eb86361e9964fb94528",
    "demucs/pretrained.py": "80ae49cb1d3e1894f07eafbb49f95559a9f3de33",
    "demucs/repo.py": "5e20ff5199e5003cf3e2228d41998e53200691de",
    "demucs/states.py": "361bb4196569bffaf622e39dd067d802dd43b38b",
    "demucs/remote/htdemucs_ft.yaml": "ba5c69c272770f5e5db3dd5fcda75b94ba523250",
    "demucs/remote/htdemucs_6s.yaml": "651a0fa536038a3e6d650f7b2bcc0b50ff7a4be9",
}
MEMBER_ORDER = [MEMBERS[prefix] for prefix in ["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d", "5c90dfd2"]]
KNOWN_HEAD_BYTES = {"f7e0c4bc-ba3fe64a.th": 84_141_271, "5c90dfd2-34c22ccb.th": 54_996_327}
FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER = "FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER"
EVIDENCE_FILENAME = "htdemucs_multi_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT).strip()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load_unique(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


class StrictYamlLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: StrictYamlLoader, node: yaml.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in mapping:
            raise ValueError(f"YAML mapping key is invalid or duplicated: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictYamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_config(text: str, expected_models: list[str]) -> dict[str, Any]:
    try:
        value = yaml.load(text, Loader=StrictYamlLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"strict YAML parse failed: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    keys = set(value)
    if expected_models == SIX_MODELS:
        if keys != {"models"}:
            raise ValueError("the fixed 6s config must contain only models; weights are not declared")
        declared_weights = None
    else:
        if keys != {"models", "weights"}:
            raise ValueError("the fixed ft config must declare exactly models and weights")
        declared_weights = value["weights"]
    models = value["models"]
    if not isinstance(models, list) or models != expected_models or any(not isinstance(model, str) for model in models):
        raise ValueError(f"config models {models!r} do not match {expected_models!r}")
    if expected_models == SIX_MODELS:
        return {"models": models, "weights": [[1.0]], "declared_weights": None, "weight_semantics": "DERIVED_SINGLE_MEMBER_IDENTITY"}
    weights_value = declared_weights
    if not isinstance(weights_value, list) or len(weights_value) != len(models):
        raise ValueError("config weights must be a square matrix")
    matrix: list[list[float]] = []
    for row in weights_value:
        if not isinstance(row, list) or len(row) != len(models):
            raise ValueError("config weights must be a square matrix")
        converted: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ValueError("config weights contain a non-finite or non-numeric value")
            converted.append(float(item))
        matrix.append(converted)
    expected_matrix = [
        [1.0 if row == column else 0.0 for column in range(len(models))]
        for row in range(len(models))
    ]
    if matrix != expected_matrix:
        raise ValueError(f"config weights {matrix!r} do not match identity matrix")
    return {"models": models, "weights": matrix, "declared_weights": weights_value, "weight_semantics": "DECLARED_IDENTITY_MATRIX"}


def source_inventory(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise RuntimeError("source checkout lacks .git metadata")
    head = git(source, "rev-parse", "HEAD")
    origin = git(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    dirty = git(source, "status", "--porcelain", "--untracked-files=all")
    if head != UPSTREAM_REVISION:
        raise RuntimeError(f"source HEAD mismatch: {head}")
    if origin != UPSTREAM_URL.removesuffix("/").removesuffix(".git"):
        raise RuntimeError(f"source origin mismatch: {origin}")
    if dirty:
        raise RuntimeError(f"source checkout is dirty: {dirty!r}")
    license_path = source / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    license_lower = license_text.lower()
    has_grant = "permission is hereby granted, free of charge" in license_lower
    has_warranty = "the software is provided" in license_lower and "without warranty" in license_lower
    if not has_grant or not has_warranty:
        raise RuntimeError("source LICENSE lacks both MIT grant and warranty clauses")
    entries: dict[str, tuple[str, str]] = {}
    for record in git(source, "ls-files", "-s", "-z").split("\0"):
        if not record:
            continue
        metadata, role = record.split("\t", 1)
        mode, object_id, stage = metadata.split()
        if mode not in {"100644", "100755"} or stage != "0":
            raise RuntimeError(f"source has non-regular or staged tracked entry: {role}")
        path = source / role
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source tracked file is missing or non-regular: {role}")
        expected_mode = {"100644": 0o644, "100755": 0o755}[mode]
        if stat.S_IMODE(path.stat().st_mode) != expected_mode:
            raise RuntimeError(f"source filesystem mode mismatch: {role}")
        if role in entries:
            raise RuntimeError(f"source index has duplicate tracked path: {role}")
        entries[role] = (mode, object_id)
    roles = []
    for role, expected in SOURCE_ROLE_BLOBS.items():
        mode_object = entries.get(role)
        if mode_object is None:
            raise RuntimeError(f"source role is not tracked: {role}")
        mode, index_object = mode_object
        head_object = git(source, "rev-parse", f"HEAD:{role}")
        working_object = git_blob_sha1(source / role)
        if index_object != expected or head_object != expected or working_object != expected:
            raise RuntimeError(f"source role object mismatch: {role}")
        if mode != "100644":
            raise RuntimeError(f"fixed source role must be mode 100644: {role}")
        roles.append({"path": role, "mode": mode, "git_blob_sha1": expected})
    return {
        "repository": UPSTREAM_URL,
        "revision": UPSTREAM_REVISION,
        "worktree_status": "CLEAN",
        "license": {"spdx": "MIT", "path": "LICENSE", "bytes": license_path.stat().st_size, "sha256": sha256(license_path)},
        "roles": roles,
    }


def tensor_manifest(value: Any, path: str = "", depth: int = 0, seen: set[int] | None = None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if depth > 64:
        raise RuntimeError("safe container nesting exceeds 64 levels")
    seen = set() if seen is None else seen
    tensors: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"non-finite tensor at {path}")
        tensors[path or "<root>"] = {
            "shape": [int(axis) for axis in value.shape],
            "dtype": str(value.dtype),
            "count": int(value.numel()),
        }
    elif isinstance(value, dict):
        if id(value) in seen:
            raise RuntimeError(f"cyclic container at {path}")
        seen.add(id(value))
        for key in sorted(value, key=str):
            child, child_unsupported = tensor_manifest(value[key], f"{path}.{key}" if path else str(key), depth + 1, seen)
            tensors.update(child)
            unsupported.extend(child_unsupported)
        seen.remove(id(value))
    elif isinstance(value, (list, tuple)):
        if id(value) in seen:
            raise RuntimeError(f"cyclic container at {path}")
        seen.add(id(value))
        for index, child_value in enumerate(value):
            child, child_unsupported = tensor_manifest(child_value, f"{path}[{index}]", depth + 1, seen)
            tensors.update(child)
            unsupported.extend(child_unsupported)
        seen.remove(id(value))
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        unsupported.append(f"{path}:{type(value).__name__}")
    return tensors, unsupported


def expected_member_url(filename: str) -> str:
    return f"{WEIGHT_ROOT}/{filename}"


def inspect_member(path: Path, response: dict[str, Any] | None) -> dict[str, Any]:
    digest = sha256(path)
    stem = path.name.removesuffix(".th")
    parts = stem.split("-", 1)
    prefix_check = len(parts) == 2 and digest.startswith(parts[1][:8])
    result: dict[str, Any] = {
        "filename": path.name,
        "sha256": digest,
        "sha256_filename_prefix_match": prefix_check,
        "source_url": expected_member_url(path.name),
    }
    blockers: list[str] = []
    if not prefix_check:
        blockers.append("filename SHA-256 prefix mismatch")
    if not isinstance(response, dict):
        blockers.append("missing response identity packet")
    else:
        expected_url = expected_member_url(path.name)
        if response.get("filename") != path.name:
            blockers.append("response filename mismatch")
        for key in ("requested_url", "effective_url"):
            value = response.get(key)
            parsed = urlsplit(value) if isinstance(value, str) else None
            if value != expected_url or parsed is None or parsed.scheme != "https" or parsed.netloc != "dl.fbaipublicfiles.com" or parsed.query or parsed.fragment:
                blockers.append(f"response {key} mismatch")
        if response.get("status") != 200:
            blockers.append("response status is not 200")
        content_length = response.get("content_length")
        if content_length is not None and (isinstance(content_length, bool) or not isinstance(content_length, int) or content_length != path.stat().st_size):
            blockers.append("response content length mismatch")
        if response.get("bytes") != path.stat().st_size or response.get("sha256") != digest:
            blockers.append("response observed payload identity mismatch")
        if any(response.get(key) is not None and not isinstance(response.get(key), str) for key in ("etag", "last_modified")):
            blockers.append("response cache validator type mismatch")
        if any(response.get(key) is not None and not isinstance(response.get(key), str) for key in ("x_amz_version_id", "x_amz_meta_s3cmd_attrs")):
            blockers.append("response S3 evidence type mismatch")
        if set(response) != {"filename", "requested_url", "effective_url", "status", "content_length", "etag", "last_modified", "x_amz_version_id", "x_amz_meta_s3cmd_attrs", "bytes", "sha256"}:
            blockers.append("response identity packet schema mismatch")
    if blockers:
        result["safe_load_status"] = "BLOCKED_RESPONSE_IDENTITY"
        result["response_blockers"] = blockers
        return result
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - record the loud safe-load blocker
        result["safe_load_status"] = "BLOCKED_WEIGHTS_ONLY"
        result["safe_load_error"] = f"{type(error).__name__}: {error}"
        return result
    try:
        tensors, unsupported = tensor_manifest(payload)
    except RuntimeError as error:
        result["safe_load_status"] = "BLOCKED_CONTAINER"
        result["safe_load_error"] = str(error)
        return result
    result["safe_load_status"] = "SAFE_LOADED"
    result["tensor_count"] = len(tensors)
    result["parameter_count"] = sum(item["count"] for item in tensors.values())
    result["tensor_manifest"] = tensors
    result["unsupported_objects"] = unsupported
    if unsupported:
        result["safe_load_status"] = "BLOCKED_UNSUPPORTED_OBJECT"
    if not tensors:
        result["safe_load_status"] = "BLOCKED_EMPTY_TENSOR_MANIFEST"
    return result


def self_test() -> None:
    global sha256
    assert parse_config("models: ['f7e0c4bc']\nweights: [[1.]]\n", ["f7e0c4bc"])["weights"] == [[1.0]]
    assert parse_config(
        "models: ['f7e0c4bc', 'd12395a8']\nweights: [[1., 0.], [0., 1.]]\n",
        ["f7e0c4bc", "d12395a8"],
    )["models"] == ["f7e0c4bc", "d12395a8"]
    ft_text = (
        "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\n"
        "weights: [[1., 0., 0., 0.], [0., 1., 0., 0.], "
        "[0., 0., 1., 0.], [0., 0., 0., 1.]]\n"
    )
    assert parse_config(ft_text, FT_MODELS)["models"] == FT_MODELS
    six_config = parse_config("models: ['5c90dfd2']\n", SIX_MODELS)
    assert six_config["models"] == SIX_MODELS
    assert six_config["declared_weights"] is None and six_config["weight_semantics"] == "DERIVED_SINGLE_MEMBER_IDENTITY"
    assert MEMBERS["5c90dfd2"].endswith(".th")
    assert UPSTREAM_URL == "https://github.com/facebookresearch/demucs"
    assert UPSTREAM_REVISION.isascii() and len(UPSTREAM_REVISION) == 40
    for invalid in (
        "models: ['f7e0c4bc']\nmodels: ['f7e0c4bc']\nweights: [[1.]]\n",
        "models: ['f7e0c4bc']\nweights: [[1.]]\nextra: true\n",
        "models: ['f7e0c4bc']\nweights: [[true]]\n",
        "models: ['f7e0c4bc']\nweights: [[.nan]]\n",
        "models: ['f7e0c4bc']\nweights: [[0.]]\n",
        "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\n",
    ):
        try:
            parse_config(invalid, ["f7e0c4bc"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Demucs YAML config was accepted")
    for invalid_six in (
        "models: ['5c90dfd2']\nweights: [[1.]]\n",
        "models: ['5c90dfd2']\nextra: 1\n",
        "models: ['5c90dfd2']\nmodels: ['5c90dfd2']\n",
    ):
        try:
            parse_config(invalid_six, SIX_MODELS)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid 6s Demucs YAML config was accepted")
    cycle: list[Any] = []
    cycle.append(cycle)
    try:
        tensor_manifest(cycle)
    except RuntimeError as error:
        assert "cyclic" in str(error)
    else:
        raise AssertionError("cyclic safe-load object was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-htdemucs-inspect-") as directory:
        path = Path(directory) / MEMBERS["f7e0c4bc"]
        torch.save({"tensor": torch.ones((2, 2))}, path)
        real_sha256 = sha256
        sha256 = lambda _path: "0" * 64
        valid_response = {
            "filename": path.name,
            "requested_url": expected_member_url(path.name),
            "effective_url": expected_member_url(path.name),
            "status": 200,
            "content_length": path.stat().st_size,
            "etag": None,
            "last_modified": None,
            "x_amz_version_id": None,
            "x_amz_meta_s3cmd_attrs": None,
            "bytes": path.stat().st_size,
            "sha256": "0" * 64,
        }
        mismatch = inspect_member(path, valid_response)
        assert mismatch["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        assert "filename SHA-256 prefix mismatch" in mismatch["response_blockers"]
        sha256 = lambda _path: "ba3fe64a" + "0" * 56
        valid_response["sha256"] = "ba3fe64a" + "0" * 56
        result = inspect_member(path, valid_response)
        assert result["safe_load_status"] == "SAFE_LOADED"
        spoof = dict(valid_response)
        spoof["sha256"] = "0" * 64
        assert inspect_member(path, spoof)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        spoof = dict(valid_response)
        spoof["effective_url"] = expected_member_url(path.name) + "?spoof=1"
        assert inspect_member(path, spoof)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        spoof = dict(valid_response)
        spoof["status"] = 206
        assert inspect_member(path, spoof)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        spoof = dict(valid_response)
        spoof["content_length"] = path.stat().st_size + 1
        assert inspect_member(path, spoof)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        spoof = dict(valid_response)
        spoof["x_amz_version_id"] = 123
        assert inspect_member(path, spoof)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        assert inspect_member(path, None)["safe_load_status"] == "BLOCKED_RESPONSE_IDENTITY"
        unsupported = Path(directory) / MEMBERS["d12395a8"]
        torch.save({"unsupported": object()}, unsupported)
        unsupported_response = dict(valid_response)
        unsupported_response.update({"filename": unsupported.name, "requested_url": expected_member_url(unsupported.name), "effective_url": expected_member_url(unsupported.name), "content_length": unsupported.stat().st_size, "bytes": unsupported.stat().st_size, "sha256": "e57c48e6" + "0" * 56})
        sha256 = lambda _path: "e57c48e6" + "0" * 56
        assert inspect_member(unsupported, unsupported_response)["safe_load_status"] in {"BLOCKED_WEIGHTS_ONLY", "BLOCKED_UNSUPPORTED_OBJECT"}
        sha256 = real_sha256
    with tempfile.TemporaryDirectory(prefix="vokra-htdemucs-error-") as directory:
        error_evidence = Path(directory)
        write_error_manifest(error_evidence, RuntimeError("self-test error"))
        error_manifest = json_load_unique(error_evidence / EVIDENCE_FILENAME)
        assert error_manifest["status"] == "BLOCKED"
        assert error_manifest["inspection_status"] == "ERROR"
        assert error_manifest["collection_status"] == "FAILED"
        assert error_manifest["publication"] == "NO_UPLOAD"


def inspect(source_dir: Path, weights_dir: Path, response_packet: Path, evidence: Path) -> int:
    configs: dict[str, Any] = {}
    blockers: list[str] = []
    try:
        source = source_inventory(source_dir)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        source = {"status": "BLOCKED", "error": str(error)}
        blockers.append(f"source identity: {error}")
    try:
        response_document = json_load_unique(response_packet)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        response_document = {}
        blockers.append(f"response packet parse failed: {error}")
    if not isinstance(response_document, dict) or set(response_document) != {"members"}:
        blockers.append("response packet schema mismatch")
    responses = response_document.get("members") if isinstance(response_document, dict) else None
    if not isinstance(responses, dict) or list(responses) != MEMBER_ORDER:
        blockers.append("response packet member order/set mismatch")
        responses = {}
    for filename, expected in (("htdemucs_ft.yaml", FT_MODELS), ("htdemucs_6s.yaml", SIX_MODELS)):
        path = source_dir / "demucs" / "remote" / filename
        if not path.is_file():
            blockers.append(f"missing config {path}")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            parsed = parse_config(text, expected)
        except (OSError, SyntaxError, ValueError) as error:
            blockers.append(f"{filename}: {error}")
            continue
        configs[filename] = {
            "sha256": sha256(path),
            "models": parsed["models"],
            "weights": parsed["weights"],
            "declared_weights": parsed["declared_weights"],
            "weight_semantics": parsed["weight_semantics"],
            "raw": text,
        }

    members: dict[str, Any] = {}
    for signature, filename in MEMBERS.items():
        path = weights_dir / filename
        if not path.is_file():
            blockers.append(f"missing member {path}")
            continue
        result = inspect_member(path, responses.get(filename))
        result["model_id"] = signature
        if filename in KNOWN_HEAD_BYTES:
            result["known_head_bytes"] = KNOWN_HEAD_BYTES[filename]
            if path.stat().st_size != KNOWN_HEAD_BYTES[filename]:
                blockers.append(f"{filename}: observed bytes differ from authenticated HEAD size")
            response = responses.get(filename)
            if isinstance(response, dict) and response.get("content_length") != KNOWN_HEAD_BYTES[filename]:
                blockers.append(f"{filename}: response content length differs from authenticated HEAD size")
        result["declared_variants"] = [
            variant
            for variant, model_ids in (("htdemucs_ft", FT_MODELS), ("htdemucs_6s", SIX_MODELS))
            if signature in model_ids
        ]
        members[signature] = result
        if result.get("safe_load_status") != "SAFE_LOADED":
            blockers.append(f"{filename}: {result.get('safe_load_status')}")

    evidence.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "vokra-htdemucs-multi-inspection-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": "COMPLETE",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_url": UPSTREAM_URL,
        "upstream_revision": UPSTREAM_REVISION,
        "source": source,
        "weight_root": WEIGHT_ROOT,
        "configs": configs,
        "member_order": {"htdemucs_ft": FT_MODELS, "htdemucs_6s": SIX_MODELS},
        "members": members,
        "safe_load_status": "BLOCKED",
        "blockers": sorted(set(blockers + [
            FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER,
            "native/runtime implementation is not available",
            "dependency license audit is unreviewed",
            "weight license/provenance audit is unreviewed",
            "dataset provenance is unauthenticated",
        ])),
        "runtime_status": "NOT_IMPLEMENTED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "source_license": source.get("license", "BLOCKED"),
        "weight_provenance": "OBSERVED_DIGESTS_ONLY_UNREVIEWED",
    }
    (evidence / EVIDENCE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "HT-Demucs inspection blocked; see htdemucs manifest: " + "; ".join(payload["blockers"]),
        file=sys.stderr,
    )
    return 2


def write_error_manifest(evidence: Path, error: Exception) -> None:
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "vokra-htdemucs-multi-inspection-v1",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "inspection_status": "ERROR",
        "collection_status": "FAILED",
        "runtime_status": "NOT_IMPLEMENTED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "error": str(error),
        "blockers": ["inspection error; source/weights are not authenticated", FULL_WEIGHT_DIGESTS_UNREVIEWED_BLOCKER],
    }
    (evidence / EVIDENCE_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--response-packet", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source_dir, args.weights_dir, args.response_packet, args.evidence_dir)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("htdemucs_multi_inspect self-test: OK")
        return 0
    if None in (args.source_dir, args.weights_dir, args.response_packet, args.evidence_dir):
        parser.error("normal runs require --source-dir, --weights-dir, --response-packet, and --evidence-dir")
    try:
        return inspect(args.source_dir, args.weights_dir, args.response_packet, args.evidence_dir)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        write_error_manifest(args.evidence_dir, error)
        print(f"HT-Demucs inspection error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
