#!/usr/bin/env python3
"""Prove that the pinned Parler DAC HF artifact is the official DAC release.

This is a provenance-only, VAST/Linux/x86_64 tool.  It loads the official
``weights.pth`` with ``weights_only=True`` and reads the HF safetensors file,
but never runs model code or inference.  The output contains only identities,
metadata, and tensor byte digests; tensor values are never serialized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable


SCHEMA = "vokra-parler-tts-dac-provenance-v1"
OFFICIAL_REPOSITORY = "descriptinc/descript-audio-codec"
OFFICIAL_SOURCE_URL = "https://github.com/descriptinc/descript-audio-codec"
OFFICIAL_RELEASE_TAG = "0.0.1"
OFFICIAL_RELEASE_REVISION = "a436c61015ee2c8dd92a2ed121cf15e13fe38789"
OFFICIAL_ASSET_URL = (
    "https://github.com/descriptinc/descript-audio-codec/releases/download/"
    "0.0.1/weights.pth"
)
OFFICIAL_WEIGHTS_BYTES = 306_717_287
OFFICIAL_WEIGHTS_SHA256 = "a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa"
OFFICIAL_LICENSE_BYTES = 1_074
OFFICIAL_LICENSE_SHA256 = "5d7a5f644313f30aad5ef032669ada67efdacefecb2c7d7b875683698bd70e87"
HF_REPOSITORY = "parler-tts/dac_44khZ_8kbps"
HF_REVISION = "5cf6b8ad50fbb17e52c341410a1d00083201b6a9"
HF_MODEL_BYTES = 306_642_416
HF_MODEL_SHA256 = "f65197de6142f9e0d186f78fb3aa12d47fde62f4c650e7ee5a254157618230f7"
HF_CONFIG_BYTES = 227
HF_CONFIG_SHA256 = "b68d924f6a8dc14a549010809cdc76fa3466085f09237adb10e7817fee058d41"
EXPECTED_TENSOR_COUNT = 301
EXPECTED_KWARGS = {
    "n_codebooks": 9,
    "codebook_size": 1024,
    "codebook_dim": 8,
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 4, 2],
    "encoder_dim": 64,
    "encoder_rates": [2, 4, 8, 8],
    "sample_rate": 44100,
    "quantizer_dropout": False,
}
EXPECTED_CONFIG = {
    "architectures": ["DACModel"],
    "codebook_size": 1024,
    "latent_dim": 1024,
    "model_bitrate": 8,
    "model_type": "dac",
    "num_codebooks": 9,
    "torch_dtype": "float32",
    "transformers_version": "4.38.0.dev0",
}


class ProvenanceError(ValueError):
    """A fail-closed provenance or input-contract violation."""


@dataclass(frozen=True)
class Contract:
    official_weights_bytes: int
    official_weights_sha256: str
    hf_model_bytes: int
    hf_model_sha256: str
    hf_config_bytes: int
    hf_config_sha256: str
    license_bytes: int
    license_sha256: str
    tensor_count: int
    kwargs: dict[str, Any]
    config: dict[str, Any]
    license_markers: tuple[str, ...]


PRODUCTION_CONTRACT = Contract(
    official_weights_bytes=OFFICIAL_WEIGHTS_BYTES,
    official_weights_sha256=OFFICIAL_WEIGHTS_SHA256,
    hf_model_bytes=HF_MODEL_BYTES,
    hf_model_sha256=HF_MODEL_SHA256,
    hf_config_bytes=HF_CONFIG_BYTES,
    hf_config_sha256=HF_CONFIG_SHA256,
    license_bytes=OFFICIAL_LICENSE_BYTES,
    license_sha256=OFFICIAL_LICENSE_SHA256,
    tensor_count=EXPECTED_TENSOR_COUNT,
    kwargs=EXPECTED_KWARGS,
    config=EXPECTED_CONFIG,
    license_markers=(
        "MIT License",
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
    ),
)

# These are the immutable facts of the VAST-generated proof checked into this
# tree.  The tool hash is the generator used on VAST (not necessarily the hash
# of a later validator), so the proof remains independently reproducible while
# still binding the exact historical generator.
PROOF_BINDING = {
    "path": "dac_provenance_evidence.json",
    "bytes": 101_082,
    "sha256": "7eebc272fbd9451bd88b4b7d12dc14057e09d1991ea64e97689654f2917e81a1",
    "repository_head": "31e6a2fc04ec6b500fdf5121c58610a09af0462a",
    "tool_path": "tools/parity/parler_tts/dac_provenance.py",
    "tool_sha256": "0fa3326f2a813c324785b446a5fa098b477a34fd23944094e85acf8d63f87c68",
    "tensor_count": EXPECTED_TENSOR_COUNT,
    "tensor_manifest_sha256": "3f9f0e1e2e239bd35a64bf0603c15763851bfdf648fea6176befffb8fe85e92b",
    "source_repository": OFFICIAL_REPOSITORY,
    "source_revision": OFFICIAL_RELEASE_REVISION,
    "source_release_tag": OFFICIAL_RELEASE_TAG,
    "hf_repository": HF_REPOSITORY,
    "hf_revision": HF_REVISION,
}

PROOF_SCHEMA_KEYS = {
    "dac_derived", "dac_kwargs", "environment", "hf", "inference_parity",
    "inference_run", "key_mapping", "model_artifacts_read", "model_code_imported",
    "publication", "repository", "schema", "source", "status", "tensor_count",
    "tensor_manifest_sha256", "tensors", "tool",
}


def _exact_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProvenanceError(f"{label} schema is not exact")
    return value


def _exact_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ProvenanceError(f"{label} is not a lowercase SHA-256 digest")
    return value


def validate_proof(
    path: Path,
    *,
    contract: Contract = PRODUCTION_CONTRACT,
    expected_file_bytes: int | None = PROOF_BINDING["bytes"],
    expected_file_sha256: str | None = PROOF_BINDING["sha256"],
    expected_head: str | None = PROOF_BINDING["repository_head"],
    expected_tool_sha256: str | None = PROOF_BINDING["tool_sha256"],
) -> dict[str, Any]:
    """Validate a generated proof without importing torch or reading weights."""
    proof_path = require_input(path, label="DAC provenance proof")
    if expected_file_bytes is not None and proof_path.stat().st_size != expected_file_bytes:
        raise ProvenanceError("DAC provenance proof byte size mismatch")
    actual_file_sha = sha256_file(proof_path)
    if expected_file_sha256 is not None and actual_file_sha != expected_file_sha256:
        raise ProvenanceError("DAC provenance proof SHA-256 mismatch")
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ProvenanceError) as exc:
        raise ProvenanceError(f"DAC provenance proof is not strict JSON: {exc}") from exc
    _exact_dict(proof, PROOF_SCHEMA_KEYS, "DAC provenance proof")
    if proof["schema"] != SCHEMA or proof["status"] != "PASS" or proof["publication"] != "NO_UPLOAD":
        raise ProvenanceError("DAC provenance proof status/publication is not fail-closed")
    if proof["model_artifacts_read"] is not True or proof["model_code_imported"] is not False:
        raise ProvenanceError("DAC provenance artifact/code flags are invalid")
    if proof["inference_run"] is not False or proof["inference_parity"] != "NOT_CLAIMED":
        raise ProvenanceError("DAC provenance proof claims inference")

    repository = _exact_dict(proof["repository"], {"root", "head", "clean"}, "repository")
    if not isinstance(repository["root"], str) or repository["clean"] is not True:
        raise ProvenanceError("DAC provenance repository is not clean")
    if expected_head is not None and repository["head"] != expected_head:
        raise ProvenanceError("DAC provenance repository HEAD mismatch")
    tool = _exact_dict(proof["tool"], {"path", "sha256"}, "tool")
    if tool["path"] != PROOF_BINDING["tool_path"]:
        raise ProvenanceError("DAC provenance tool path mismatch")
    if expected_tool_sha256 is not None and tool["sha256"] != expected_tool_sha256:
        raise ProvenanceError("DAC provenance tool SHA-256 mismatch")

    source = _exact_dict(proof["source"], {"repository", "url", "release_tag", "revision", "asset_url", "weights", "license"}, "source")
    if source["repository"] != OFFICIAL_REPOSITORY or source["url"] != OFFICIAL_SOURCE_URL:
        raise ProvenanceError("DAC official source identity mismatch")
    if source["release_tag"] != OFFICIAL_RELEASE_TAG or source["revision"] != OFFICIAL_RELEASE_REVISION or source["asset_url"] != OFFICIAL_ASSET_URL:
        raise ProvenanceError("DAC release identity mismatch")
    weights = _exact_dict(source["weights"], {"bytes", "sha256"}, "official weights identity")
    if weights != {"bytes": contract.official_weights_bytes, "sha256": contract.official_weights_sha256}:
        raise ProvenanceError("official weights identity mismatch")
    license_identity = _exact_dict(source["license"], {"bytes", "sha256", "spdx"}, "release license identity")
    if license_identity != {"bytes": contract.license_bytes, "sha256": contract.license_sha256, "spdx": "MIT"}:
        raise ProvenanceError("release-tag MIT license identity mismatch")

    hf = _exact_dict(proof["hf"], {"repository", "revision", "model", "config", "config_semantics"}, "HF identity")
    if hf["repository"] != HF_REPOSITORY or hf["revision"] != HF_REVISION:
        raise ProvenanceError("HF repository or exact revision mismatch")
    model_identity = _exact_dict(hf["model"], {"bytes", "sha256"}, "HF model identity")
    if model_identity != {"bytes": contract.hf_model_bytes, "sha256": contract.hf_model_sha256}:
        raise ProvenanceError("HF model identity mismatch")
    config_identity = _exact_dict(hf["config"], {"bytes", "sha256"}, "HF config identity")
    if config_identity != {"bytes": contract.hf_config_bytes, "sha256": contract.hf_config_sha256}:
        raise ProvenanceError("HF config identity mismatch")
    if hf["config_semantics"] != contract.config:
        raise ProvenanceError("HF config semantics mismatch")
    if proof["dac_kwargs"] != contract.kwargs:
        raise ProvenanceError("DAC raw kwargs mismatch")
    if proof["dac_derived"] != {"d_model": 1024, "hop_length": 512}:
        raise ProvenanceError("DAC derived semantics mismatch")
    if proof["key_mapping"] != {"bijective": True, "official_to_hf_prefix": "model."}:
        raise ProvenanceError("DAC key mapping is not the exact model. bijection")

    if proof["tensor_count"] != contract.tensor_count or not isinstance(proof["tensors"], list):
        raise ProvenanceError("DAC tensor count/list mismatch")
    rows = proof["tensors"]
    if len(rows) != contract.tensor_count:
        raise ProvenanceError("DAC tensor proof row count mismatch")
    previous = None
    seen: set[str] = set()
    for row in rows:
        row = _exact_dict(row, {"official_key", "hf_key", "shape", "dtype", "numel", "official_sha256", "hf_sha256"}, "tensor row")
        key = row["official_key"]
        if not isinstance(key, str) or key in seen or (previous is not None and key <= previous):
            raise ProvenanceError("DAC tensor keys are not unique and sorted")
        seen.add(key)
        previous = key
        if row["hf_key"] != f"model.{key}" or not isinstance(row["dtype"], str) or not row["dtype"]:
            raise ProvenanceError("DAC tensor key mapping or dtype mismatch")
        shape = row["shape"]
        if not isinstance(shape, list) or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ProvenanceError("DAC tensor shape is invalid")
        numel = 1
        for dim in shape:
            numel *= dim
        if not isinstance(row["numel"], int) or isinstance(row["numel"], bool) or row["numel"] != numel:
            raise ProvenanceError("DAC tensor numel does not match shape")
        official_sha = _exact_digest(row["official_sha256"], "official tensor digest")
        hf_sha = _exact_digest(row["hf_sha256"], "HF tensor digest")
        if official_sha != hf_sha:
            raise ProvenanceError("official and HF tensor digests differ")
    manifest_sha = _exact_digest(proof["tensor_manifest_sha256"], "tensor manifest digest")
    if manifest_sha != sha256_json(rows):
        raise ProvenanceError("DAC tensor manifest hash drifted")
    environment = _exact_dict(
        proof["environment"],
        {"cargo_invoked", "inference_run", "machine", "model_code_imported", "platform", "python", "safetensors", "torch"},
        "environment",
    )
    if environment["cargo_invoked"] is not False or environment["inference_run"] is not False or environment["model_code_imported"] is not False:
        raise ProvenanceError("DAC provenance environment claims forbidden work")
    if any(not isinstance(environment[key], str) or not environment[key] for key in ("machine", "platform", "python", "safetensors", "torch")):
        raise ProvenanceError("DAC provenance environment versions/platform are malformed")
    if expected_head == PROOF_BINDING["repository_head"]:
        if len(rows) != PROOF_BINDING["tensor_count"] or manifest_sha != PROOF_BINDING["tensor_manifest_sha256"]:
            raise ProvenanceError("checked-in DAC tensor proof binding drifted")
        if environment != {
            "cargo_invoked": False, "inference_run": False, "machine": "x86_64", "model_code_imported": False,
            "platform": "Linux", "python": "3.12.14", "safetensors": "0.8.0", "torch": "2.11.0+cpu",
        }:
            raise ProvenanceError("checked-in DAC proof environment drifted")
    return proof


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError("config.json is not strict UTF-8 JSON") from exc


def _absolute_without_symlink(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ProvenanceError(f"{label} contains a symlink: {current}")
    return absolute


def require_input(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_symlink(path, label=label)
    if not absolute.is_file():
        raise ProvenanceError(f"{label} is not an existing regular file: {path}")
    return absolute


def require_absent_output(path: Path, *, repo_root: Path) -> Path:
    absolute = _absolute_without_symlink(path, label="output")
    parent = absolute.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ProvenanceError(f"output parent is not a non-symlink directory: {parent}")
    repo = repo_root.resolve()
    try:
        absolute.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ProvenanceError("output must not be inside the repository checkout")
    if absolute.exists() or absolute.is_symlink():
        raise ProvenanceError(f"output already exists; refusing overwrite: {path}")
    return absolute


def verify_file(path: Path, *, label: str, size: int, digest: str) -> dict[str, Any]:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ProvenanceError(f"{label} byte size mismatch: got {actual_size}, expected {size}")
    actual_digest = sha256_file(path)
    if actual_digest != digest:
        raise ProvenanceError(f"{label} SHA-256 mismatch: got {actual_digest}, expected {digest}")
    return {"bytes": actual_size, "sha256": actual_digest}


def load_torch_and_safetensors() -> tuple[Any, Callable[..., Any]]:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise ProvenanceError("torch and safetensors must be installed in the frozen UV project") from exc
    return torch, safe_open


def unique_string_keys(keys: list[Any], *, label: str) -> list[str]:
    if any(not isinstance(key, str) for key in keys):
        raise ProvenanceError(f"{label} contains a non-string key")
    strings = [str(key) for key in keys]
    if len(strings) != len(set(strings)):
        raise ProvenanceError(f"{label} contains duplicate keys")
    return strings


def tensor_bytes(tensor: Any, torch: Any, *, label: str) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise ProvenanceError(f"{label} is not a tensor")
    if tensor.layout != torch.strided:
        raise ProvenanceError(f"{label} is not a dense strided tensor")
    try:
        return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"{label} cannot be converted to canonical bytes") from exc


def load_official_checkpoint(path: Path, torch: Any, contract: Contract) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - checkpoint loader errors are factual blockers
        raise ProvenanceError(f"official checkpoint failed weights_only=True load: {type(exc).__name__}") from exc
    if not isinstance(payload, dict) or set(payload) != {"metadata", "state_dict"}:
        raise ProvenanceError("official checkpoint must contain exactly metadata and state_dict")
    state = payload["state_dict"]
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"kwargs"}:
        raise ProvenanceError("official checkpoint metadata must contain exactly kwargs")
    kwargs = metadata["kwargs"]
    if not isinstance(state, dict):
        raise ProvenanceError("official checkpoint state_dict is not a mapping")
    unique_string_keys(list(state), label="official state_dict")
    if len(state) != contract.tensor_count:
        raise ProvenanceError(f"official tensor count mismatch: got {len(state)}, expected {contract.tensor_count}")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ProvenanceError("official state_dict contains a non-tensor value")
    if not isinstance(kwargs, dict) or kwargs != contract.kwargs:
        raise ProvenanceError("official DAC kwargs semantics mismatch")
    return state, kwargs


def derive_dac_semantics(kwargs: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    rates = kwargs.get("encoder_rates")
    encoder_dim = kwargs.get("encoder_dim")
    if not isinstance(rates, list) or any(
        not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0 for rate in rates
    ):
        raise ProvenanceError("DAC encoder_rates are not a positive integer list")
    if not isinstance(encoder_dim, int) or isinstance(encoder_dim, bool) or encoder_dim <= 0:
        raise ProvenanceError("DAC encoder_dim is not a positive integer")
    hop_length = 1
    for rate in rates:
        hop_length *= rate
    d_model = encoder_dim * (2 ** len(rates))
    if hop_length != 512 or d_model != 1024:
        raise ProvenanceError("derived DAC hop_length/d_model semantics mismatch")
    if config.get("latent_dim") != d_model:
        raise ProvenanceError("derived DAC d_model does not match HF config latent_dim")
    return {"hop_length": hop_length, "d_model": d_model}


def parse_config(path: Path, contract: Contract) -> dict[str, Any]:
    config = strict_json_bytes(path.read_bytes())
    if config != contract.config:
        raise ProvenanceError("HF config.json semantics mismatch")
    return config


def verify_license(path: Path, contract: Contract) -> dict[str, Any]:
    body = path.read_bytes()
    for marker in contract.license_markers:
        if marker.encode("utf-8") not in body:
            raise ProvenanceError("release-tag LICENSE is not the expected MIT text")
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "spdx": "MIT"}


def compare_tensors(
    state: dict[str, Any], hf_path: Path, torch: Any, safe_open: Callable[..., Any], contract: Contract
) -> list[dict[str, Any]]:
    try:
        with safe_open(str(hf_path), framework="pt", device="cpu") as handle:
            hf_keys = unique_string_keys(list(handle.keys()), label="HF safetensors")
            if len(hf_keys) != contract.tensor_count:
                raise ProvenanceError(f"HF tensor count mismatch: got {len(hf_keys)}, expected {contract.tensor_count}")
            if any(not key.startswith("model.") or not key.removeprefix("model.") for key in hf_keys):
                raise ProvenanceError("every HF key must have exactly one nonempty model. prefix")
            normalized = {key.removeprefix("model.") for key in hf_keys}
            official_keys = set(unique_string_keys(list(state), label="official state_dict"))
            if normalized != official_keys:
                missing = sorted(official_keys - normalized)
                extra = sorted(normalized - official_keys)
                raise ProvenanceError(f"HF/official key bijection mismatch: missing={missing[:3]} extra={extra[:3]}")
            rows: list[dict[str, Any]] = []
            for official_key in sorted(official_keys):
                hf_key = f"model.{official_key}"
                official = state[official_key]
                candidate = handle.get_tensor(hf_key)
                if tuple(official.shape) != tuple(candidate.shape) or official.dtype != candidate.dtype:
                    raise ProvenanceError(f"shape/dtype mismatch for tensor {official_key}")
                if not torch.equal(official, candidate):
                    raise ProvenanceError(f"value mismatch for tensor {official_key}")
                official_raw = tensor_bytes(official, torch, label=official_key)
                hf_raw = tensor_bytes(candidate, torch, label=hf_key)
                official_digest = hashlib.sha256(official_raw).hexdigest()
                hf_digest = hashlib.sha256(hf_raw).hexdigest()
                if official_digest != hf_digest:
                    raise ProvenanceError(f"canonical byte digest mismatch for tensor {official_key}")
                rows.append(
                    {
                        "official_key": official_key,
                        "hf_key": hf_key,
                        "shape": list(official.shape),
                        "dtype": str(official.dtype),
                        "numel": official.numel(),
                        "official_sha256": official_digest,
                        "hf_sha256": hf_digest,
                    }
                )
            return rows
    except ProvenanceError:
        raise
    except Exception as exc:  # noqa: BLE001 - malformed safetensors are factual blockers
        raise ProvenanceError(f"HF safetensors failed strict read: {type(exc).__name__}") from exc


def repository_identity(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProvenanceError(f"git identity unavailable: {type(exc).__name__}") from exc

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if root != repo_root.resolve():
        raise ProvenanceError("tool is not running from its repository checkout")
    head = git("rev-parse", "HEAD")
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ProvenanceError("repository HEAD is not a full commit SHA")
    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProvenanceError("repository worktree is dirty")
    return {"root": str(root), "head": head, "clean": True}


def environment_identity(torch: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "safetensors": str(__import__("safetensors").__version__),
        "model_code_imported": False,
        "inference_run": False,
        "cargo_invoked": False,
    }


def generate(
    official_weights: Path,
    hf_model: Path,
    hf_config: Path,
    release_license: Path,
    output: Path,
    *,
    contract: Contract = PRODUCTION_CONTRACT,
    repo_root: Path | None = None,
    allow_non_vast: bool = False,
) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    if not allow_non_vast:
        if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
            raise ProvenanceError("VOKRA_PUBLISH_ON_VAST=1 is required")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ProvenanceError("DAC provenance requires Linux x86_64 VAST")
        if sys.version_info[:2] != (3, 12):
            raise ProvenanceError("DAC provenance requires Python 3.12")
        repo = repository_identity(root)
    else:
        repo = {"root": str(root), "head": "synthetic-self-test", "clean": True}
    official_path = require_input(official_weights, label="official weights")
    model_path = require_input(hf_model, label="HF safetensors")
    config_path = require_input(hf_config, label="HF config")
    license_path = require_input(release_license, label="release LICENSE")
    output_path = require_absent_output(output, repo_root=root)
    official_identity = verify_file(
        official_path,
        label="official weights",
        size=contract.official_weights_bytes,
        digest=contract.official_weights_sha256,
    )
    model_identity = verify_file(
        model_path, label="HF safetensors", size=contract.hf_model_bytes, digest=contract.hf_model_sha256
    )
    config_identity = verify_file(
        config_path, label="HF config", size=contract.hf_config_bytes, digest=contract.hf_config_sha256
    )
    license_identity = verify_file(
        license_path, label="release LICENSE", size=contract.license_bytes, digest=contract.license_sha256
    )
    config = parse_config(config_path, contract)
    license_semantics = verify_license(license_path, contract)
    torch, safe_open = load_torch_and_safetensors()
    state, kwargs = load_official_checkpoint(official_path, torch, contract)
    derived = derive_dac_semantics(kwargs, config)
    tensor_rows = compare_tensors(state, model_path, torch, safe_open, contract)
    if len(tensor_rows) != contract.tensor_count:
        raise ProvenanceError("tensor proof row count drifted after comparison")
    tensor_manifest_sha256 = sha256_json(tensor_rows)
    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "publication": "NO_UPLOAD",
        "model_artifacts_read": True,
        "model_code_imported": False,
        "inference_run": False,
        "inference_parity": "NOT_CLAIMED",
        "tool": {
            "path": "tools/parity/parler_tts/dac_provenance.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "repository": repo,
        "environment": environment_identity(torch),
        "source": {
            "repository": OFFICIAL_REPOSITORY,
            "url": OFFICIAL_SOURCE_URL,
            "release_tag": OFFICIAL_RELEASE_TAG,
            "revision": OFFICIAL_RELEASE_REVISION,
            "asset_url": OFFICIAL_ASSET_URL,
            "weights": official_identity,
            "license": {**license_identity, **license_semantics},
        },
        "hf": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "model": model_identity,
            "config": config_identity,
            "config_semantics": config,
        },
        "dac_kwargs": kwargs,
        "dac_derived": derived,
        "tensor_count": len(tensor_rows),
        "key_mapping": {"official_to_hf_prefix": "model.", "bijective": True},
        "tensor_manifest_sha256": tensor_manifest_sha256,
        "tensors": tensor_rows,
    }
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(proof) + "\n")
    except FileExistsError as exc:
        raise ProvenanceError("output appeared during proof generation; refusing overwrite") from exc
    return proof


def _expect_blocked(operation: Callable[[], Any], label: str) -> None:
    try:
        operation()
    except ProvenanceError:
        return
    raise AssertionError(f"self-test accepted {label}")


def self_test() -> int:
    torch, safe_open = load_torch_and_safetensors()
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory(prefix="parler-dac-provenance-", dir="/private/tmp") as directory:
        root = Path(directory)
        repo_root = root / "repo"
        repo_root.mkdir()
        official = root / "weights.pth"
        hf_model = root / "model.safetensors"
        hf_config = root / "config.json"
        release_license = root / "LICENSE"
        output = root / "proof.json"
        state = {
            "decoder.0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "quantizer.embed": torch.tensor([1, 2, 3], dtype=torch.int64),
        }
        kwargs = {
            "n_codebooks": 9,
            "codebook_size": 1024,
            "codebook_dim": 8,
            "decoder_dim": 1536,
            "decoder_rates": [8, 8, 4, 2],
            "encoder_dim": 64,
            "encoder_rates": [2, 4, 8, 8],
            "sample_rate": 44100,
            "quantizer_dropout": False,
        }
        config = {
            "architectures": ["DACModel"],
            "codebook_size": 1024,
            "latent_dim": 1024,
            "model_bitrate": 8,
            "model_type": "dac",
            "num_codebooks": 9,
            "torch_dtype": "float32",
            "transformers_version": "4.38.0.dev0",
        }
        license_body = b"MIT License\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
        torch.save({"state_dict": state, "metadata": {"kwargs": kwargs}}, official)
        save_file({f"model.{key}": value for key, value in state.items()}, str(hf_model))
        config_body = canonical_json(config).encode("utf-8")
        hf_config.write_bytes(config_body)
        release_license.write_bytes(license_body)
        contract = Contract(
            official_weights_bytes=official.stat().st_size,
            official_weights_sha256=sha256_file(official),
            hf_model_bytes=hf_model.stat().st_size,
            hf_model_sha256=sha256_file(hf_model),
            hf_config_bytes=len(config_body),
            hf_config_sha256=hashlib.sha256(config_body).hexdigest(),
            license_bytes=len(license_body),
            license_sha256=hashlib.sha256(license_body).hexdigest(),
            tensor_count=2,
            kwargs=kwargs,
            config=config,
            license_markers=("MIT License", "Permission is hereby granted, free of charge"),
        )
        proof = generate(
            official,
            hf_model,
            hf_config,
            release_license,
            output,
            contract=contract,
            repo_root=repo_root,
            allow_non_vast=True,
        )
        assert proof["status"] == "PASS" and proof["tensor_count"] == 2
        assert proof["model_artifacts_read"] is True
        assert proof["model_code_imported"] is False
        assert proof["inference_run"] is False
        assert proof["dac_derived"] == {"hop_length": 512, "d_model": 1024}
        serialized = output.read_text(encoding="utf-8")
        assert "tensor_values" not in serialized and "[[1.0,2.0]" not in serialized
        _expect_blocked(
            lambda: generate(
                official,
                hf_model,
                hf_config,
                release_license,
                output,
                contract=contract,
                repo_root=repo_root,
                allow_non_vast=True,
            ),
            "output overwrite",
        )
        _expect_blocked(
            lambda: require_absent_output(repo_root / "inside.json", repo_root=repo_root),
            "output inside repo",
        )
        for label, mutate in (
            (
                "value mismatch",
                lambda: save_file(
                    {
                        "model.decoder.0.weight": state["decoder.0.weight"] + 1,
                        "model.quantizer.embed": state["quantizer.embed"],
                    },
                    str(hf_model),
                ),
            ),
            (
                "prefix mismatch",
                lambda: save_file(
                    {
                        "decoder.0.weight": state["decoder.0.weight"],
                        "model.quantizer.embed": state["quantizer.embed"],
                    },
                    str(hf_model),
                ),
            ),
            (
                "key mismatch",
                lambda: save_file(
                    {
                        "model.decoder.0.weight": state["decoder.0.weight"],
                        "model.other": state["quantizer.embed"],
                    },
                    str(hf_model),
                ),
            ),
            (
                "shape mismatch",
                lambda: save_file(
                    {
                        "model.decoder.0.weight": state["decoder.0.weight"].reshape(4),
                        "model.quantizer.embed": state["quantizer.embed"],
                    },
                    str(hf_model),
                ),
            ),
            (
                "dtype mismatch",
                lambda: save_file(
                    {
                        "model.decoder.0.weight": state["decoder.0.weight"].to(torch.float64),
                        "model.quantizer.embed": state["quantizer.embed"],
                    },
                    str(hf_model),
                ),
            ),
        ):
            mutate()
            _expect_blocked(lambda: compare_tensors(state, hf_model, torch, safe_open, contract), label)
            save_file({f"model.{key}": value for key, value in state.items()}, str(hf_model))
        original_model_bytes = hf_model.read_bytes()
        hf_model.write_bytes(original_model_bytes + b"x")
        _expect_blocked(
            lambda: verify_file(
                hf_model,
                label="HF safetensors",
                size=contract.hf_model_bytes,
                digest=contract.hf_model_sha256,
            ),
            "input hash mismatch",
        )
        hf_model.write_bytes(original_model_bytes)
        hf_config.write_text(canonical_json({**config, "latent_dim": 1}), encoding="utf-8")
        _expect_blocked(lambda: parse_config(hf_config, contract), "config semantics mismatch")
        hf_config.write_bytes(config_body)
        release_license.write_bytes(b"not MIT")
        _expect_blocked(lambda: verify_license(release_license, contract), "license identity mismatch")
        release_license.write_bytes(license_body)
        bad_checkpoint = root / "bad.pth"
        torch.save({"state_dict": {"bad": "not tensor"}, "metadata": {"kwargs": kwargs}}, bad_checkpoint)
        _expect_blocked(lambda: load_official_checkpoint(bad_checkpoint, torch, contract), "non-tensor checkpoint")
        malformed_metadata = root / "malformed-metadata.pth"
        torch.save({"state_dict": state, "metadata": {"wrong": kwargs}}, malformed_metadata)
        _expect_blocked(lambda: load_official_checkpoint(malformed_metadata, torch, contract), "malformed metadata")
        _expect_blocked(
            lambda: derive_dac_semantics({**kwargs, "encoder_rates": [2, 4]}, config),
            "derived semantics mismatch",
        )
        _expect_blocked(lambda: unique_string_keys(["x", "x"], label="duplicate"), "duplicate keys")
        input_link = root / "weights-link.pth"
        input_link.symlink_to(official)
        _expect_blocked(lambda: require_input(input_link, label="symlinked weights"), "symlinked input")
        output_link = root / "proof-link.json"
        output_link.symlink_to(official)
        _expect_blocked(lambda: require_absent_output(output_link, repo_root=root), "symlinked output")
    print("parler DAC provenance: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-weights", type=Path)
    parser.add_argument("--hf-safetensors", type=Path)
    parser.add_argument("--hf-config", type=Path)
    parser.add_argument("--release-license", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.official_weights, args.hf_safetensors, args.hf_config, args.release_license, args.output)):
            parser.error("--self-test accepts no input or output paths")
        try:
            return self_test()
        except (AssertionError, ProvenanceError) as exc:
            print(f"parler DAC provenance: self-test FAIL: {exc}", file=sys.stderr)
            return 1
    values = (args.official_weights, args.hf_safetensors, args.hf_config, args.release_license, args.output)
    if any(value is None for value in values):
        parser.error("all four input paths and --output are required")
    try:
        generate(*values)  # type: ignore[arg-type]
    except ProvenanceError as exc:
        print(f"parler DAC provenance: BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"parler DAC provenance: PASS: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
