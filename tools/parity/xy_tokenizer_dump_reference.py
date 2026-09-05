#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Run the pinned XY-Tokenizer implementation as a VAST-only oracle.

This is deliberately an adapter, rather than a second implementation of the
codec.  It imports ``xy_tokenizer.model.XY_Tokenizer`` from the authenticated
upstream checkout, calls its public ``inference_tokenize`` and
``inference_detokenize`` methods, and records f32 taps from the official
modules.  The dependency gate is fail-closed: the repository currently has no
version-keyed primary-source license review for this source's transitive
closure, so no checkpoint or source import is permitted yet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/gyt1145028706/XY-Tokenizer"
SOURCE_REVISION = "5df5609c5883e555bd39a2d0b1005ca8f1a8f12e"
SOURCE_ROLE_BLOBS = {
    "config/xy_tokenizer_config.yaml": "83c50a60b3c0db62ce30b9cd65e0b0f5cd290f89",
    "inference.py": "9bb00a176f878d872f8eb7ed7a98501d3abb7e70",
    "inference_for_codec_evaluation.py": "4a98524ac90506a21b6155b31e945163c5d35d5b",
    "requirements.txt": "46b7b2d2aabb074ce87433eba2f55b31eee2363b",
    "utils/helpers.py": "9b144a4ce5ca6fd57b1a2903d940c4b4ffec4d97",
    "xy_tokenizer/model.py": "188f1b607d3e9a5953b3015ea9d262008ef535c0",
    "xy_tokenizer/nn/feature_extractor.py": "4d397b012ffe756fa9dfadc771f81e0afddd3963",
    "xy_tokenizer/nn/modules.py": "cc186d9dadd674172837d527fef0f0de183feb4c",
    "xy_tokenizer/nn/quantizer.py": "a7d28b963e98ea4f62f2a6e06b419cf0da0c2cc4",
}
CHECKPOINT_BYTES = 2_137_328_977
CHECKPOINT_SHA256 = "37c7ac18d0a48f5a1d0687e31af7c0264861232c500206718c98acd8e37d1671"
CONFIG_SHA256 = "e7d48677e34f77e5b9fd7dc7a3e0eef7f2d2dd9be9a245d5c1d56489dc748938"
UPSTREAM_REPOSITORY = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0"
UPSTREAM_REVISION = "c83433728e698ed0698e88cb5096bc221fb8f8c5"
FORMAT = "vokra-xy-tokenizer-official-reference-v1"
DEPENDENCY_LICENSE_AUDIT_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
DEPENDENCY_LICENSE_BLOCKER = "DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER"
TOPOLOGY_UNVERIFIED_BLOCKER = "TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER"
REFERENCE_PROJECT = Path(__file__).parent / "xy_tokenizer_reference"
FRONTEND_CLASS_DECLARATION = "class MelFeatureExtractor(SequenceFeatureExtractor):"
FRONTEND_CAPTURE_DESCRIPTION = "official feature extractor output observed as semantic/acoustic encoder pre-hook input"
TAP_MODULES = (
    "semantic_encoder",
    "semantic_encoder_adapter",
    "acoustic_encoder",
    "pre_rvq_adapter",
    "downsample",
    "quantizer",
    "post_rvq_adapter",
    "upsample",
    "acoustic_decoder",
    "vocos",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load_unique(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def authenticate_source(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError("official source must be a real directory")
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official source revision mismatch")
    origin = git(source, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    if origin != SOURCE_REPOSITORY:
        raise RuntimeError("official source origin mismatch")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("official source checkout is dirty")
    files: dict[str, dict[str, Any]] = {}
    for relative, expected_blob in SOURCE_ROLE_BLOBS.items():
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing official source role: {relative}")
        blob = git_blob_sha1(path)
        if blob != expected_blob:
            raise RuntimeError(f"official source role changed: {relative}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "git_blob_sha1": blob,
            "sha256": sha256_file(path),
        }
    extractor_text = (source / "xy_tokenizer/nn/feature_extractor.py").read_text(encoding="utf-8")
    model_text = (source / "xy_tokenizer/model.py").read_text(encoding="utf-8")
    if FRONTEND_CLASS_DECLARATION not in extractor_text:
        raise RuntimeError("official frontend class structure mismatch")
    if "self.feature_extractor = MelFeatureExtractor" not in model_text:
        raise RuntimeError("official model does not bind the authenticated frontend")
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "files": files,
    }


def authenticate_artifacts(checkpoint: Path, config: Path) -> dict[str, Any]:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RuntimeError("checkpoint must be a real regular file")
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise RuntimeError("checkpoint byte size mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if config.is_symlink() or not config.is_file():
        raise RuntimeError("config must be a real regular file")
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("config SHA-256 mismatch")
    return {
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "checkpoint": {"bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256},
        "config": {"bytes": config.stat().st_size, "sha256": CONFIG_SHA256},
    }


def canonical_input(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink")
    if not path.exists():
        raise RuntimeError(f"{label} does not exist")
    return path.resolve(strict=True)


def canonical_output(path: Path) -> Path:
    if not path.is_absolute():
        raise RuntimeError("output must be an absolute path")
    if path.is_symlink():
        raise RuntimeError("output must not be a symlink")
    if path.exists():
        if not path.is_dir():
            raise RuntimeError("output must be a directory")
        return path.resolve(strict=True)
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("output parent must be an existing non-symlink directory")
    return parent.resolve(strict=True) / path.name


def require_disjoint_output(output: Path, *inputs: Path) -> None:
    paths = (output, *inputs)
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                raise RuntimeError("source/checkpoint/config/output/dependency project must be disjoint")


def require_empty_output(output: Path) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise RuntimeError("output must be a real directory")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)


def require_dependency_audit(project: Path) -> dict[str, Any]:
    """Require a separately reviewed, exact project before importing torch.

    The current repository has only the broad parity lock and no source-specific
    package/license rows.  In particular, upstream leaves all requirements
    unpinned and the transitive native bundles (torchaudio/librosa/scipy) have
    not been reviewed here.  Keeping this check before any source import makes
    the VAST worker report the real blocker instead of silently accepting that
    broad environment.
    """
    project = canonical_input(project, "dependency project")
    if not project.is_dir():
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": dependency project is not a directory")
    audit = project / "dependency_audit.json"
    lock = project / "uv.lock"
    pyproject = project / "pyproject.toml"
    for path in (audit, lock, pyproject):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": exact project/license evidence is absent or symlinked")
    try:
        value = json_load_unique(audit)
    except (OSError, ValueError) as error:
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": malformed audit") from error
    if not isinstance(value, dict) or set(value) != {
        "schema", "status", "lock_sha256", "pyproject_sha256", "packages"
    } or value.get("status") != "AUDITED_ALLOW":
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": audit is not affirmatively approved")
    if value.get("schema") != "vokra-xy-tokenizer-dependency-audit-v1":
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": audit schema is not recognized")
    for key in ("lock_sha256", "pyproject_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": project digest is invalid")
    if value.get("lock_sha256") != sha256_file(lock) or value.get("pyproject_sha256") != sha256_file(pyproject):
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": audit is not bound to the exact project files")
    try:
        lock_data = tomllib.loads(lock.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": uv.lock is malformed") from error
    lock_rows = lock_data.get("package")
    audit_rows = value.get("packages")
    if not isinstance(lock_rows, list) or not isinstance(audit_rows, list):
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": complete package rows are absent")
    lock_identity: set[tuple[str, str]] = set()
    for row in lock_rows:
        if not isinstance(row, dict):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": lock package row is not an object")
        name, version = row.get("name"), row.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": lock package identity is invalid")
        identity = (name, version)
        if identity in lock_identity:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": duplicate lock package identity")
        lock_identity.add(identity)
    audited_identity: set[tuple[Any, Any]] = set()
    for row in audit_rows:
        if not isinstance(row, dict) or set(row) != {"name", "version", "license", "evidence"}:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": package license row schema mismatch")
        name, version, license_name, evidence = (
            row.get("name"), row.get("version"), row.get("license"), row.get("evidence")
        )
        if not all(isinstance(item, str) and item for item in (name, version, license_name)):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": package license row is incomplete")
        if not isinstance(evidence, dict) or set(evidence) != {"source", "revision", "sha256"}:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": primary-source evidence schema mismatch")
        evidence_source, evidence_revision, evidence_sha256 = (
            evidence.get("source"), evidence.get("revision"), evidence.get("sha256")
        )
        if not isinstance(evidence_source, str) or not evidence_source.startswith("https://"):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": primary-source URL is invalid")
        if not isinstance(evidence_revision, str) or not evidence_revision:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": primary-source revision is missing")
        if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in evidence_sha256
        ):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": primary-source digest is invalid")
        upper_license = license_name.upper()
        if any(token in upper_license for token in ("GPL", "LGPL", "UNKNOWN", "UNLICENSED")):
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": disallowed/unknown dependency license")
        identity = (name, version)
        if identity in audited_identity:
            raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": duplicate audited package identity")
        audited_identity.add(identity)
    if audited_identity != lock_identity:
        raise RuntimeError(DEPENDENCY_LICENSE_BLOCKER + ": audit does not cover every locked package")
    return value


def _tensor_items(value: Any, prefix: str = "0") -> list[tuple[str, Any]]:
    """Extract tensors from official module returns without reimplementing them."""
    # Importing torch here is intentional and only reached on VAST after gates.
    import torch

    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            items.extend(_tensor_items(child, f"{prefix}.{key}"))
        return items
    if isinstance(value, (tuple, list)):
        items = []
        for index, child in enumerate(value):
            items.extend(_tensor_items(child, f"{prefix}.{index}"))
        return items
    return []


def _write_f32(output: Path, label: str, tensor: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    data = np.asarray(values, dtype="<f4").tobytes(order="C")
    path = output / f"{label}.f32"
    path.write_bytes(data)
    return {
        "path": path.name,
        "shape": [int(value) for value in values.shape],
        "source_dtype": str(tensor.dtype),
        "tap_dtype": "F32",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_discrete(output: Path, label: str, tensor: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    values = tensor.detach().to(device="cpu")
    if values.dtype == torch.bool or not values.dtype.is_floating_point and not values.dtype.is_complex:
        values = values.to(dtype=torch.int64).contiguous()
    else:
        raise RuntimeError(f"discrete tap has non-integer dtype: {label}={values.dtype}")
    signed = values.numpy()
    if bool((values < 0).any()):
        data = np.asarray(signed, dtype="<i8").tobytes(order="C")
        encoding = "I64LE"
    else:
        if bool((values > 0xFFFFFFFF).any()):
            raise RuntimeError(f"unsigned discrete tap exceeds U32: {label}")
        data = np.asarray(signed, dtype="<u4").tobytes(order="C")
        encoding = "U32LE"
    path = output / f"{label}.{encoding.lower()}"
    path.write_bytes(data)
    return {
        "path": path.name,
        "shape": [int(value) for value in values.shape],
        "source_dtype": str(tensor.dtype),
        "dtype": encoding,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_value(output: Path, label: str, tensor: Any) -> dict[str, Any]:
    import torch

    if tensor.dtype == torch.bool or not tensor.dtype.is_floating_point and not tensor.dtype.is_complex:
        return _write_discrete(output, label, tensor)
    return _write_f32(output, label, tensor)


def _fixed_waveform(torch: Any, sample_rate: int = 16_000, seconds: int = 1) -> Any:
    samples = sample_rate * seconds
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    return (0.25 * torch.sin(2 * math.pi * 220 * time) + 0.1 * torch.sin(2 * math.pi * 440 * time)).reshape(1, 1, -1)


def run_official(
    source: Path,
    checkpoint: Path,
    config_path: Path,
    output: Path,
    dependency_audit: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    import yaml

    sys.path.insert(0, str(source))
    module = importlib.import_module("xy_tokenizer.model")
    model_class = module.XY_Tokenizer
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or not isinstance(config.get("generator_params"), Mapping):
        raise RuntimeError("official YAML generator_params mapping is missing")
    model = model_class(config["generator_params"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise RuntimeError("official checkpoint is not a mapping")
    state = state.get("generator", state)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("official checkpoint is not a string-keyed state dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    captures: dict[str, list[dict[str, Any]]] = {name: [] for name in TAP_MODULES}
    frontend_inputs: dict[str, list[Any]] = {"semantic_encoder": [], "acoustic_encoder": []}
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], result: Any) -> None:
            captures[name].extend({"name": key, "tensor": value} for key, value in _tensor_items(result))

        return hook

    def make_frontend_pre_hook(name: str):
        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"official {name} pre-hook did not receive frontend tensor")
            frontend_inputs[name].append(inputs[0].detach())

        return pre_hook

    for name in TAP_MODULES:
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
    handles.extend(
        getattr(model, name).register_forward_pre_hook(make_frontend_pre_hook(name))
        for name in frontend_inputs
    )
    try:
        waveform = _fixed_waveform(torch)
        input_lengths = torch.tensor([waveform.shape[-1]], dtype=torch.long)
        with torch.inference_mode():
            encoded = model.inference_tokenize(waveform, input_lengths)
            decoded = model.inference_detokenize(encoded["codes"], encoded["codes_lengths"])
    finally:
        for handle in handles:
            handle.remove()

    records: dict[str, Any] = {}
    input_data = waveform.detach().numpy().astype("<f4").tobytes(order="C")
    (output / "input_waveform.f32").write_bytes(input_data)
    records["input_waveform"] = {
        "path": "input_waveform.f32",
        "shape": [int(value) for value in waveform.shape],
        "dtype": "F32",
        "bytes": len(input_data),
        "sha256": hashlib.sha256(input_data).hexdigest(),
    }
    if any(len(values) != 1 for values in frontend_inputs.values()):
        raise RuntimeError("official frontend was not observed exactly once per encoder")
    semantic_frontend = frontend_inputs["semantic_encoder"][0]
    acoustic_frontend = frontend_inputs["acoustic_encoder"][0]
    if not torch.equal(semantic_frontend, acoustic_frontend):
        raise RuntimeError("semantic/acoustic encoder frontend inputs differ")
    records["frontend_input"] = _write_f32(output, "frontend_input", semantic_frontend)
    for name, items in captures.items():
        if not items:
            raise RuntimeError(f"official module produced no tap: {name}")
        for index, item in enumerate(items):
            safe_name = name + "__" + str(index)
            records[safe_name] = _write_value(output, safe_name, item["tensor"])
    for name, value in (("encoded_zq", encoded["zq"]), ("encoded_codes", encoded["codes"]), ("encoded_codes_lengths", encoded["codes_lengths"]), ("waveform", decoded["y"]), ("decoded_output_lengths", decoded["output_length"])):
        records[name] = _write_value(output, name, value)
    manifest = {
        "format": FORMAT,
        "status": "REFERENCE_COMPLETE",
        "source": authenticate_source(source),
        "upstream": authenticate_artifacts(checkpoint, config_path),
        "oracle": {
            "implementation": "official XY_Tokenizer",
            "methods": ["inference_tokenize", "inference_detokenize"],
            "module_taps": list(TAP_MODULES),
            "frontend_capture": FRONTEND_CAPTURE_DESCRIPTION,
            "input": "deterministic 1-second two-tone waveform; no random state or dataset",
            "fixture": {
                "sample_rate": 16_000,
                "seconds": 1,
                "formula": "0.25*sin(2*pi*220*t) + 0.1*sin(2*pi*440*t)",
            },
        },
        "dependency_license_audit": dependency_audit["status"],
        "dependency_audit": dependency_audit,
        "records": records,
        "blockers": [TOPOLOGY_UNVERIFIED_BLOCKER],
        "native_status": "BLOCKED_UNTIL_RUNTIME_AND_PARITY_REVIEW",
        "publication": "NO_UPLOAD",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def self_test() -> None:
    assert len(SOURCE_REVISION) == len(UPSTREAM_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == len(CONFIG_SHA256) == 64
    assert CHECKPOINT_BYTES > 2_000_000_000
    assert set(SOURCE_ROLE_BLOBS) == {
        "config/xy_tokenizer_config.yaml", "inference.py", "inference_for_codec_evaluation.py",
        "requirements.txt", "utils/helpers.py", "xy_tokenizer/model.py",
        "xy_tokenizer/nn/feature_extractor.py", "xy_tokenizer/nn/modules.py", "xy_tokenizer/nn/quantizer.py",
    }
    assert len(set(SOURCE_ROLE_BLOBS.values())) == len(SOURCE_ROLE_BLOBS)
    assert DEPENDENCY_LICENSE_AUDIT_STATUS == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert DEPENDENCY_LICENSE_BLOCKER == "DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER"
    assert TOPOLOGY_UNVERIFIED_BLOCKER == "TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER"
    assert "feature_extractor" not in TAP_MODULES
    assert TAP_MODULES[0] == "semantic_encoder" and TAP_MODULES[-1] == "vocos"
    assert FRONTEND_CLASS_DECLARATION == "class MelFeatureExtractor(SequenceFeatureExtractor):"
    source_text = Path(__file__).read_text(encoding="utf-8")
    assert "register_forward_pre_hook" in source_text
    assert "register_forward_hook" in source_text
    assert "feature_extractor" not in {name for name in TAP_MODULES}
    assert '"semantic_encoder"' in source_text
    assert '"acoustic_encoder"' in source_text
    assert "U32LE" in _write_discrete.__code__.co_consts
    assert "I64LE" in _write_discrete.__code__.co_consts
    assert "inference_tokenize" in run_official.__code__.co_names
    assert "inference_detokenize" in run_official.__code__.co_names
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        (output / "existing").write_text("x", encoding="utf-8")
        try:
            require_empty_output(output)
        except RuntimeError as error:
            assert "empty" in str(error)
        else:
            raise AssertionError("non-empty output accepted")
        duplicate = output / "duplicate.json"
        duplicate.write_text('{"status":"x","status":"y"}\n', encoding="utf-8")
        try:
            json_load_unique(duplicate)
        except ValueError as error:
            assert "duplicate JSON key" in str(error)
        else:
            raise AssertionError("duplicate JSON key accepted")
        symlink = output / "link"
        symlink.symlink_to(output / "existing")
        for check in (
            lambda: canonical_input(symlink, "input"),
            lambda: canonical_output(symlink),
        ):
            try:
                check()
            except RuntimeError as error:
                assert "symlink" in str(error)
            else:
                raise AssertionError("symlink input/output accepted")
        project = output / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (project / "dependency_audit.json").symlink_to(output / "duplicate.json")
        try:
            require_dependency_audit(project)
        except RuntimeError as error:
            assert "symlinked" in str(error)
        else:
            raise AssertionError("symlinked dependency audit accepted")
    print("xy_tokenizer_dump_reference.py self-test: OK (official-route/fail-closed contracts)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dependency-project", type=Path, default=REFERENCE_PROJECT)
    parser.add_argument("--dependency-audit", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source_dir, args.checkpoint, args.config, args.output)):
            parser.error("--self-test accepts no model/source arguments")
        self_test()
        return 0
    dependency_project = canonical_input(args.dependency_project, "dependency project")
    if args.dependency_audit:
        if any(value is not None for value in (args.source_dir, args.checkpoint, args.config, args.output)):
            parser.error("--dependency-audit accepts no source/model arguments")
        evidence = require_dependency_audit(dependency_project)
        print(json.dumps({"status": evidence["status"], "project": str(dependency_project)}, sort_keys=True))
        return 0
    if not all(value is not None for value in (args.source_dir, args.checkpoint, args.config, args.output)):
        parser.error("source, checkpoint, config and output are required")
    source = canonical_input(args.source_dir, "source")
    checkpoint = canonical_input(args.checkpoint, "checkpoint")
    config = canonical_input(args.config, "config")
    output = canonical_output(args.output)
    require_disjoint_output(output, source, checkpoint, config, dependency_project)
    # This must stay before source/model imports and before output creation.
    dependency_audit = require_dependency_audit(dependency_project)
    authenticate_source(source)
    authenticate_artifacts(checkpoint, config)
    require_empty_output(output)
    run_official(source, checkpoint, config, output, dependency_audit)
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
