#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Safely bridge the pinned FireRedASR-AED-L checkpoint to safetensors.

This sidecar is VAST-only for the real 4.7 GB checkpoint. It uses PyTorch's
``weights_only`` loader with the one approved ``argparse.Namespace`` global,
requires the exact checkpoint identity, inventories the archive and every
state-dict tensor, and writes a hash-bound audit sidecar next to the prepared
file. It never executes a model or imports the upstream inference package.

Only explicitly audited training counters (``.num_batches_tracked``,
``.total_ops`` and ``.total_params``) may be stripped. Unknown non-floating
dtypes, shared storage and duplicate tensor names are hard errors. This is a
preparation artifact, not a conversion, runtime, parity or publication claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

MODEL_REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
MODEL_REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
CHECKPOINT_BYTES = 4_678_597_714
CHECKPOINT_SHA256 = "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3"
EXPECTED_UNSAFE_GLOBALS = ["argparse.Namespace"]
ALLOWED_FLOAT_DTYPES = {"torch.float16", "torch.float32", "torch.bfloat16"}
DROP_SUFFIXES = (".num_batches_tracked", ".total_ops", ".total_params")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PREPARATION_FORMAT = "vokra-firered-asr-aed-l-preparation-v1"


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")


def inventory_archive(path: Path) -> list[dict[str, Any]]:
    """Record exact ZIP members without unpickling any data."""
    if not zipfile.is_zipfile(path):
        raise ValueError("checkpoint is not the authenticated torch ZIP container")
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 100_000:
            raise ValueError("checkpoint archive member count exceeds bound")
        for item in archive.infolist():
            safe_member_name(item.filename)
            if item.filename in seen or item.flag_bits & 1:
                raise ValueError(f"duplicate or encrypted archive member: {item.filename!r}")
            mode = (item.external_attr >> 16) & 0o170000
            if mode not in {0, 0o040000, 0o100000}:
                raise ValueError(f"unsafe archive member type: {item.filename!r}")
            seen.add(item.filename)
            members.append({"name": item.filename, "bytes": item.file_size, "crc32": item.CRC})
    return members


def require_checkpoint_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint must be a regular file")
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint size mismatch: {size} != {CHECKPOINT_BYTES}")
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    return {"bytes": size, "sha256": digest, "repository": MODEL_REPOSITORY, "revision": MODEL_REVISION}


def guard_paths(checkpoint: Path, output: Path, audit_output: Path, *, reject_existing: bool) -> None:
    """Reject aliases, symlink targets and unsafe output targets before writes."""
    raw_paths = (checkpoint, output, audit_output)
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("checkpoint, output and audit-output must not be symlinks")
    normalized = tuple(path.expanduser().resolve(strict=False) for path in raw_paths)
    if len(set(normalized)) != 3:
        raise ValueError("checkpoint, output and audit-output must be three distinct normalized paths")
    for path in (output, audit_output):
        if path.exists() and not path.is_file():
            raise ValueError(f"output target is not a regular file: {path}")
        if path.parent.exists() and path.parent.is_symlink():
            raise ValueError(f"output parent must not be a symlink: {path.parent}")
        if reject_existing and path.exists():
            raise ValueError(f"refusing to overwrite existing output target: {path}")
    if output.exists() and checkpoint.exists() and os.path.samefile(output, checkpoint):
        raise ValueError("output aliases checkpoint")
    if audit_output.exists() and checkpoint.exists() and os.path.samefile(audit_output, checkpoint):
        raise ValueError("audit-output aliases checkpoint")
    if output.exists() and audit_output.exists() and os.path.samefile(output, audit_output):
        raise ValueError("output aliases audit-output")


def tensor_bytes(tensor: Any) -> bytes:
    """Obtain deterministic raw CPU bytes for float tensors, including BF16."""
    import torch

    value = tensor.detach().to(device="cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(tensor_bytes(tensor)).hexdigest(),
    }


def audit_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and return ``(float tensors, audit details)``."""
    import torch

    prepared: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    stripped: list[dict[str, Any]] = []
    names: set[str] = set()
    storage_names: dict[int, str] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"duplicate or invalid tensor name: {name!r}")
        names.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state entry is not a tensor: {name!r}")
        storage = tensor.detach().untyped_storage().data_ptr()
        if storage in storage_names:
            raise ValueError(f"shared tensor storage: {name!r} aliases {storage_names[storage]!r}")
        storage_names[storage] = name
        dtype = str(tensor.dtype)
        if dtype not in ALLOWED_FLOAT_DTYPES:
            if any(name.endswith(suffix) for suffix in DROP_SUFFIXES) and tensor.dtype in {torch.int32, torch.int64}:
                stripped.append({"name": name, "dtype": dtype, "shape": list(tensor.shape), "reason": "audited_training_counter"})
                continue
            raise ValueError(f"unsupported non-float or unknown tensor dtype {dtype} for {name!r}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"non-finite floating tensor: {name!r}")
        prepared[name] = tensor.detach().contiguous()
        records.append(tensor_record(name, tensor))
    if not prepared:
        raise ValueError("state_dict contains no inference tensors after audit")
    return prepared, {"tensor_count": len(records), "tensors": records, "stripped": stripped, "shared_storage": False, "duplicate_names": False}


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only the approved checkpoint envelope and inventory its tensors."""
    import argparse as argparse_module
    import torch

    unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
    if unsafe != EXPECTED_UNSAFE_GLOBALS:
        raise ValueError(f"unsafe globals are not the exact approved set: {unsafe!r}")
    with torch.serialization.safe_globals([argparse_module.Namespace]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint envelope must be a dict, got {type(payload).__name__}")
    required = {"args", "model_state_dict"}
    if set(payload) != required:
        raise ValueError(f"checkpoint envelope keys mismatch: {sorted(payload)!r} != {sorted(required)!r}")
    if not isinstance(payload["args"], argparse_module.Namespace):
        raise ValueError("checkpoint args must be argparse.Namespace")
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")
    prepared, audit = audit_state_dict(state_dict)
    return prepared, {"envelope_keys": sorted(payload), "args_fields": sorted(vars(payload["args"])), "state_dict": audit}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_preparation_manifest(main_manifest: dict[str, Any], preparation: dict[str, Any], prepared_path: Path) -> None:
    """Validate the merged evidence contract without loading model weights."""
    if main_manifest.get("status") != "BLOCKED" or main_manifest.get("evidence_stage") != "INSPECTION_ONLY":
        raise ValueError("inspection status is not blocked/inspection-only")
    if main_manifest.get("publication") != "NO_UPLOAD" or main_manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or main_manifest.get("parity_status") != "NOT_RUN":
        raise ValueError("main manifest publication/runtime/parity status mismatch")
    model = main_manifest.get("model")
    if not isinstance(model, dict) or model.get("repository") != MODEL_REPOSITORY or model.get("revision") != MODEL_REVISION:
        raise ValueError("main manifest model identity mismatch")
    artifact = main_manifest.get("artifacts", {}).get("model.pth.tar")
    if not isinstance(artifact, dict) or artifact.get("bytes") != CHECKPOINT_BYTES or artifact.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("main manifest checkpoint artifact identity mismatch")
    if preparation.get("format") != PREPARATION_FORMAT or preparation.get("status") != "PREPARED" or preparation.get("publication") != "NO_UPLOAD":
        raise ValueError("preparation status/publication mismatch")
    if preparation.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or preparation.get("parity_status") != "NOT_RUN":
        raise ValueError("preparation runtime/parity status mismatch")
    preparation_model = preparation.get("model")
    if not isinstance(preparation_model, dict) or set(preparation_model) != {"repository", "revision"} or preparation_model.get("repository") != MODEL_REPOSITORY or preparation_model.get("revision") != MODEL_REVISION:
        raise ValueError("preparation model identity mismatch")
    checkpoint = preparation.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"bytes", "sha256", "repository", "revision", "archive_members"}:
        raise ValueError("preparation checkpoint identity mismatch")
    if checkpoint.get("repository") != MODEL_REPOSITORY or checkpoint.get("revision") != MODEL_REVISION or checkpoint.get("bytes") != CHECKPOINT_BYTES or checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("preparation checkpoint identity mismatch")
    archive_members = checkpoint.get("archive_members")
    if not isinstance(archive_members, list) or not archive_members:
        raise ValueError("archive member inventory is missing")
    archive_names: set[str] = set()
    for member in archive_members:
        if not isinstance(member, dict) or set(member) != {"name", "bytes", "crc32"}:
            raise ValueError("archive member fields mismatch")
        name, size, crc32 = (member[key] for key in ("name", "bytes", "crc32"))
        if not isinstance(name, str) or not name or name in archive_names:
            raise ValueError("archive member name is missing or duplicated")
        safe_member_name(name)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(crc32, int) or isinstance(crc32, bool) or not 0 <= crc32 <= 0xFFFFFFFF:
            raise ValueError(f"archive member identity is invalid: {name}")
        archive_names.add(name)
    output = preparation.get("output")
    if not isinstance(output, dict) or set(output) != {"path", "bytes", "sha256"} or output.get("path") != str(prepared_path):
        raise ValueError("preparation output path mismatch")
    if not prepared_path.is_file() or prepared_path.is_symlink():
        raise ValueError("prepared output is not a regular file")
    if output.get("bytes") != prepared_path.stat().st_size or not isinstance(output.get("sha256"), str) or not HEX64.fullmatch(output["sha256"]):
        raise ValueError("preparation output metadata mismatch")
    actual_output_sha = sha256_file(prepared_path)
    if actual_output_sha != output["sha256"]:
        raise ValueError("prepared output SHA-256 mismatch")
    audit = preparation.get("audit")
    if not isinstance(audit, dict) or set(audit) != {"envelope_keys", "args_fields", "state_dict"}:
        raise ValueError("audit envelope fields mismatch")
    if audit.get("envelope_keys") != ["args", "model_state_dict"]:
        raise ValueError("checkpoint envelope keys mismatch")
    args_fields = audit.get("args_fields")
    if not isinstance(args_fields, list) or not args_fields or len(args_fields) > 4096 or args_fields != sorted(args_fields) or len(set(args_fields)) != len(args_fields) or any(not isinstance(field, str) or not field or len(field) > 4096 for field in args_fields):
        raise ValueError("checkpoint args_fields are missing, duplicated or unbounded")
    state = audit.get("state_dict")
    if not isinstance(state, dict) or set(state) != {"tensor_count", "tensors", "stripped", "shared_storage", "duplicate_names"}:
        raise ValueError("state_dict audit is missing")
    records = state.get("tensors")
    count = state.get("tensor_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or not isinstance(records, list) or len(records) != count:
        raise ValueError("tensor audit count/list mismatch")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "shape", "dtype", "numel", "sha256"}:
            raise ValueError("tensor audit record fields mismatch")
        name, shape, dtype, numel, digest = (record[key] for key in ("name", "shape", "dtype", "numel", "sha256"))
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("tensor audit name is missing or duplicated")
        if not isinstance(shape, list) or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError(f"tensor audit shape is invalid: {name}")
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        if not isinstance(numel, int) or isinstance(numel, bool) or numel < 0 or numel != expected_numel:
            raise ValueError(f"tensor audit numel is invalid: {name}")
        if dtype not in ALLOWED_FLOAT_DTYPES or not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError(f"tensor audit dtype/hash is invalid: {name}")
        names.add(name)
    stripped = state.get("stripped")
    if not isinstance(stripped, list):
        raise ValueError("stripped audit is missing")
    for record in stripped:
        if not isinstance(record, dict) or set(record) != {"name", "dtype", "shape", "reason"}:
            raise ValueError("stripped audit record fields mismatch")
        name, dtype, shape, reason = (record[key] for key in ("name", "dtype", "shape", "reason"))
        if not isinstance(name, str) or not name or name in names or not any(name.endswith(suffix) for suffix in DROP_SUFFIXES):
            raise ValueError("stripped audit name is not an approved training counter")
        if dtype not in {"torch.int32", "torch.int64"} or reason != "audited_training_counter":
            raise ValueError("stripped audit dtype/reason is not approved")
        if not isinstance(shape, list) or any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise ValueError("stripped audit shape is invalid")
        names.add(name)
    if state.get("shared_storage") is not False or state.get("duplicate_names") is not False:
        raise ValueError("shared/duplicate audit must be explicitly false")
    future = preparation.get("future_gate")
    if not isinstance(future, dict) or set(future) != {"reference", "status", "blocker", "fp32_atol", "fp32_atol_status"} or future.get("status") != "BLOCKED_NOT_RUN" or future.get("fp32_atol") != 0.01 or future.get("fp32_atol_status") != "PREREGISTERED_NOT_RUN" or not isinstance(future.get("reference"), str) or not future["reference"] or not isinstance(future.get("blocker"), str) or not future["blocker"]:
        raise ValueError("future reference gate status mismatch")


def prepare(checkpoint: Path, output: Path, audit_output: Path) -> dict[str, Any]:
    guard_paths(checkpoint, output, audit_output, reject_existing=True)
    identity = require_checkpoint_identity(checkpoint)
    archive_members = inventory_archive(checkpoint)
    prepared, audit = load_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        from safetensors.torch import save_file

        save_file(prepared, str(temporary_path))
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    if not output.is_file() or output.is_symlink():
        raise ValueError("prepared output is not a regular file")
    result = {
        "format": PREPARATION_FORMAT,
        "status": "PREPARED",
        "publication": "NO_UPLOAD",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "parity_status": "NOT_RUN",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "checkpoint": {**identity, "archive_members": archive_members},
        "audit": audit,
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "future_gate": {
            "reference": "independent upstream importer required",
            "status": "BLOCKED_NOT_RUN",
            "blocker": "the locked tools/parity environment has no pinned FireRedASR upstream package/import project; no local mirror oracle is permitted",
            "fp32_atol": 0.01,
            "fp32_atol_status": "PREREGISTERED_NOT_RUN",
        },
    }
    write_json(audit_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.validate_manifest:
        if not args.inspection_manifest or not args.preparation_manifest or not args.prepared:
            parser.error("--validate-manifest requires --inspection-manifest, --preparation-manifest and --prepared")
        try:
            main_manifest = json.loads(args.inspection_manifest.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
            preparation = json.loads(args.preparation_manifest.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)
            validate_preparation_manifest(main_manifest, preparation, args.prepared)
        except Exception as error:
            print(f"FireRedASR-AED-L manifest validation failed closed: {error}", file=sys.stderr)
            return 2
        print("firered preparation manifest validation PASS")
        return 0
    if not args.ckpt or not args.output:
        parser.error("--ckpt and --output are required unless --validate-manifest is used")
    audit_output = args.audit_output or Path(f"{args.output}.manifest.json")
    try:
        result = prepare(args.ckpt, args.output, audit_output)
    except Exception as error:
        print(f"FireRedASR-AED-L preparation failed closed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def self_test() -> None:
    import torch

    assert MODEL_REPOSITORY == "FireRedTeam/FireRedASR-AED-L"
    assert HEX64.fullmatch(CHECKPOINT_SHA256)
    assert DROP_SUFFIXES == (".num_batches_tracked", ".total_ops", ".total_params")
    good = {"encoder.weight": torch.ones((2, 2), dtype=torch.float32), "encoder.bias": torch.zeros(2, dtype=torch.bfloat16), "bn.num_batches_tracked": torch.tensor(1, dtype=torch.int64)}
    prepared, audit = audit_state_dict(good)
    assert set(prepared) == {"encoder.weight", "encoder.bias"}
    assert len(audit["stripped"]) == 1
    for bad in ({"bad": torch.ones(1, dtype=torch.int64)}, {"bad": torch.ones(1, dtype=torch.bool)}, {"bad": torch.tensor([float("nan")])}):
        try:
            audit_state_dict(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe tensor state accepted")
    shared = torch.ones(2)
    try:
        audit_state_dict({"a": shared, "b": shared})
    except ValueError as error:
        assert "shared tensor storage" in str(error)
    else:
        raise AssertionError("shared tensor storage accepted")
    with tempfile.TemporaryDirectory(prefix="firered-preparer-") as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint.pth.tar"
        output = root / "prepared.safetensors"
        audit_output = root / "prepared.manifest.json"
        checkpoint.write_bytes(b"checkpoint")
        for aliases in ((checkpoint, checkpoint, audit_output), (checkpoint, output, checkpoint)):
            try:
                guard_paths(*aliases, reject_existing=True)
            except ValueError:
                pass
            else:
                raise AssertionError("path alias accepted")
        output.write_bytes(b"existing")
        try:
            guard_paths(checkpoint, output, audit_output, reject_existing=True)
        except ValueError:
            pass
        else:
            raise AssertionError("existing output target accepted")
        output.unlink()
        try:
            output.symlink_to(checkpoint)
            guard_paths(checkpoint, output, audit_output, reject_existing=True)
        except ValueError:
            pass
        else:
            raise AssertionError("symlink output target accepted")
        finally:
            output.unlink(missing_ok=True)
        try:
            audit_output.symlink_to(checkpoint)
            guard_paths(checkpoint, output, audit_output, reject_existing=True)
        except ValueError:
            pass
        else:
            raise AssertionError("symlink audit target accepted")
        finally:
            audit_output.unlink(missing_ok=True)
        prepared_path = root / "prepared.safetensors"
        prepared_path.write_bytes(b"prepared")
        prepared_digest = sha256_file(prepared_path)
        main_manifest = {
            "status": "BLOCKED",
            "evidence_stage": "INSPECTION_ONLY",
            "publication": "NO_UPLOAD",
            "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
            "parity_status": "NOT_RUN",
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "artifacts": {"model.pth.tar": {"bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256}},
        }
        preparation_manifest = {
            "format": PREPARATION_FORMAT,
            "status": "PREPARED",
            "publication": "NO_UPLOAD",
            "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
            "parity_status": "NOT_RUN",
            "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
            "checkpoint": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256, "archive_members": [{"name": "archive/data.pkl", "bytes": 1, "crc32": 0}]},
            "output": {"path": str(prepared_path), "bytes": prepared_path.stat().st_size, "sha256": prepared_digest},
            "audit": {"envelope_keys": ["args", "model_state_dict"], "args_fields": ["sos_id"], "state_dict": {"tensor_count": 1, "tensors": [{"name": "x", "shape": [1], "dtype": "torch.float32", "numel": 1, "sha256": "0" * 64}], "stripped": [], "shared_storage": False, "duplicate_names": False}},
            "future_gate": {"reference": "independent upstream importer required", "status": "BLOCKED_NOT_RUN", "blocker": "not available in synthetic self-test", "fp32_atol": 0.01, "fp32_atol_status": "PREREGISTERED_NOT_RUN"},
        }
        validate_preparation_manifest(main_manifest, preparation_manifest, prepared_path)
        inspection_path = root / "inspection.json"
        preparation_path = root / "preparation.json"
        write_json(inspection_path, main_manifest)
        write_json(preparation_path, preparation_manifest)
        validator_args = [sys.executable, str(Path(__file__)), "--validate-manifest", "--inspection-manifest", str(inspection_path), "--preparation-manifest", str(preparation_path), "--prepared", str(prepared_path)]
        result = subprocess.run(validator_args, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        swapped = json.loads(json.dumps(preparation_manifest))
        swapped["model"]["repository"], swapped["model"]["revision"] = swapped["model"]["revision"], swapped["model"]["repository"]
        swapped["checkpoint"]["repository"], swapped["checkpoint"]["revision"] = swapped["checkpoint"]["revision"], swapped["checkpoint"]["repository"]
        write_json(preparation_path, swapped)
        result = subprocess.run(validator_args, capture_output=True, text=True, check=False)
        assert result.returncode == 2, "swapped identity accepted by CLI validator"
        preparation_path.write_text('{"format":"x","format":"y"}\n', encoding="utf-8")
        result = subprocess.run(validator_args, capture_output=True, text=True, check=False)
        assert result.returncode == 2, "duplicate JSON key accepted by CLI validator"
        write_json(preparation_path, preparation_manifest)
        malformed = json.loads(json.dumps(preparation_manifest))
        del malformed["output"]["sha256"]
        try:
            validate_preparation_manifest(main_manifest, malformed, prepared_path)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed preparation manifest accepted")
        malformed = json.loads(json.dumps(preparation_manifest))
        malformed["audit"]["state_dict"]["tensor_count"] = 2
        try:
            validate_preparation_manifest(main_manifest, malformed, prepared_path)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched tensor count accepted")
        malformed = json.loads(json.dumps(preparation_manifest))
        duplicate = dict(malformed["audit"]["state_dict"]["tensors"][0])
        malformed["audit"]["state_dict"]["tensors"].append(duplicate)
        malformed["audit"]["state_dict"]["tensor_count"] = 2
        try:
            validate_preparation_manifest(main_manifest, malformed, prepared_path)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate tensor audit name accepted")
    print("firered preparer self-test PASS")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        raise SystemExit(main())
