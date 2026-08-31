#!/usr/bin/env python3
"""Remove only BatchNorm training counters from the pinned voice-gender file.

The official checkpoint contains 202 floating inference tensors and 31 scalar
``torch.int64`` ``*.num_batches_tracked`` tensors.  Vokra's safetensors reader
is inference-only and intentionally rejects the counters.  This sidecar keeps
the official raw file untouched for the independent reference dumper and emits
an auditable 202-tensor file for the converter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

UPSTREAM_HF_REPOSITORY = "JaesungHuh/voice-gender-classifier"
UPSTREAM_HF_REVISION = "db1222153bd60337e900be22add7af180452adc0"
UPSTREAM_SOURCE_REPOSITORY = "https://github.com/JaesungHuh/voice-gender-classifier.git"
UPSTREAM_SOURCE_REVISION = "49bcbecfd929ba5a043bde645fdff1a375eb79c7"
TRANSFORM = "remove_authenticated_batchnorm_num_batches_tracked_v1"
EXPECTED_INPUT_TENSOR_COUNT = 233
EXPECTED_FLOATING_TENSOR_COUNT = 202
EXPECTED_COUNTER_COUNT = 31
COUNTER_SUFFIX = ".num_batches_tracked"


def counter_names() -> set[str]:
    names = {"bn1.num_batches_tracked", "bn5.num_batches_tracked", "bn6.num_batches_tracked"}
    names.add("attention.2.num_batches_tracked")
    for layer in range(1, 4):
        names.add(f"layer{layer}.bn1.num_batches_tracked")
        names.update(f"layer{layer}.bns.{inner}.num_batches_tracked" for inner in range(7))
        names.add(f"layer{layer}.bn3.num_batches_tracked")
    if len(names) != EXPECTED_COUNTER_COUNT:
        raise AssertionError(f"internal counter manifest has {len(names)} names")
    return names


EXPECTED_COUNTER_NAMES = frozenset(counter_names())
ALLOWED_FLOAT_DTYPES: frozenset[Any]


def bind_tensor_dependencies() -> tuple[Any, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    return torch, (load_file, save_file)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_bytes(tensor: Any, torch: Any) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_manifest(tensors: dict[str, Any], torch: Any) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(tensors):
        tensor = tensors[name]
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "sha256": sha256_bytes(tensor_bytes(tensor, torch)),
            }
        )
    return rows


def validate_state(tensors: dict[str, Any], torch: Any) -> tuple[dict[str, Any], list[str]]:
    if len(tensors) != EXPECTED_INPUT_TENSOR_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_INPUT_TENSOR_COUNT} input tensors, got {len(tensors)}"
        )
    actual_counter_names = {name for name in tensors if name.endswith(COUNTER_SUFFIX)}
    if actual_counter_names != EXPECTED_COUNTER_NAMES:
        missing = sorted(EXPECTED_COUNTER_NAMES - actual_counter_names)
        unexpected = sorted(actual_counter_names - EXPECTED_COUNTER_NAMES)
        raise ValueError(f"counter name manifest mismatch: missing={missing}, unexpected={unexpected}")

    floating: dict[str, Any] = {}
    for name, tensor in tensors.items():
        if name in EXPECTED_COUNTER_NAMES:
            if tensor.dtype != torch.int64:
                raise ValueError(f"counter {name} must be torch.int64, got {tensor.dtype}")
            if tensor.ndim != 0:
                raise ValueError(f"counter {name} must be scalar, got shape {tuple(tensor.shape)}")
            continue
        if not torch.is_floating_point(tensor):
            raise ValueError(f"unexpected non-floating inference tensor {name}: {tensor.dtype}")
        if tensor.dtype not in ALLOWED_FLOAT_DTYPES:
            raise ValueError(f"unsupported inference dtype for {name}: {tensor.dtype}")
        floating[name] = tensor
    if len(floating) != EXPECTED_FLOATING_TENSOR_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_FLOATING_TENSOR_COUNT} floating tensors, got {len(floating)}"
        )
    return floating, sorted(actual_counter_names)


def write_prepared_checkpoint(
    input_path: Path, output_path: Path, audit_path: Path, torch: Any, load_file: Any, save_file: Any
) -> dict[str, Any]:
    if input_path.suffix != ".safetensors":
        raise ValueError(f"input must be a safetensors file: {input_path}")
    if output_path.suffix != ".safetensors":
        raise ValueError(f"output must be a safetensors file: {output_path}")
    if audit_path.suffix != ".json":
        raise ValueError(f"audit output must be JSON: {audit_path}")
    if not input_path.is_file() or input_path.is_symlink():
        raise ValueError(f"input is missing or symlinked: {input_path}")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"refusing to overwrite output: {output_path}")
    if audit_path.exists() or audit_path.is_symlink():
        raise ValueError(f"refusing to overwrite audit: {audit_path}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("input and output must be different files")
    if output_path.resolve() == audit_path.resolve():
        raise ValueError("output and audit paths must be different files")

    tensors = load_file(str(input_path), device="cpu")
    floating, removed = validate_state(tensors, torch)
    floating_manifest = tensor_manifest(floating, torch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        save_file(floating, str(temporary_path))
        prepared = load_file(str(temporary_path), device="cpu")
        if tensor_manifest(prepared, torch) != floating_manifest:
            raise ValueError("prepared checkpoint changed a floating tensor name, shape, dtype, or value")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    input_bytes = input_path.stat().st_size
    output_bytes = output_path.stat().st_size
    audit = {
        "schema": "vokra-voice-gender-checkpoint-normalization-v1",
        "status": "AUTHENTICATED_NORMALIZED",
        "input_file": input_path.name,
        "input_bytes": input_bytes,
        "input_sha256": sha256_bytes(input_path.read_bytes()),
        "output_file": output_path.name,
        "output_bytes": output_bytes,
        "output_sha256": sha256_bytes(output_path.read_bytes()),
        "source_repository": UPSTREAM_HF_REPOSITORY,
        "source_revision": UPSTREAM_HF_REVISION,
        "upstream_source_repository": UPSTREAM_SOURCE_REPOSITORY,
        "upstream_source_revision": UPSTREAM_SOURCE_REVISION,
        "transform": TRANSFORM,
        "input_tensor_count": len(tensors),
        "input_floating_tensor_count": len(floating),
        "input_counter_count": len(removed),
        "output_tensor_count": len(prepared),
        "removed_counter_names": removed,
        "floating_tensor_manifest": floating_manifest,
        "floating_tensor_manifest_sha256": sha256_bytes(
            json.dumps(floating_manifest, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def self_test() -> None:
    torch, (load_file, save_file) = bind_tensor_dependencies()
    global ALLOWED_FLOAT_DTYPES
    ALLOWED_FLOAT_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})
    with tempfile.TemporaryDirectory(prefix="vokra-voice-gender-prepare-") as directory:
        root = Path(directory)
        float_tensors = {f"float_{index:03d}": torch.arange(4, dtype=torch.float32) for index in range(202)}
        counters = {name: torch.tensor(0, dtype=torch.int64) for name in EXPECTED_COUNTER_NAMES}
        valid = {**float_tensors, **counters}
        input_path = root / "model.safetensors"
        save_file(valid, str(input_path))
        audit = write_prepared_checkpoint(
            input_path, root / "prepared.safetensors", root / "audit.json", torch, load_file, save_file
        )
        assert audit["input_tensor_count"] == EXPECTED_INPUT_TENSOR_COUNT
        assert audit["input_floating_tensor_count"] == EXPECTED_FLOATING_TENSOR_COUNT
        assert audit["input_counter_count"] == EXPECTED_COUNTER_COUNT
        assert audit["output_tensor_count"] == EXPECTED_FLOATING_TENSOR_COUNT
        assert audit["removed_counter_names"] == sorted(EXPECTED_COUNTER_NAMES)

        def expect_failure(label: str, state: dict[str, Any]) -> None:
            try:
                validate_state(state, torch)
            except ValueError:
                return
            raise AssertionError(f"accepted invalid synthetic checkpoint: {label}")

        expect_failure("unexpected int", {**valid, "unexpected.int": torch.tensor(0, dtype=torch.int64)})
        expect_failure("wrong count", {**float_tensors, **dict(list(counters.items())[:-1])})
        wrong_shape = {**float_tensors, **counters, "bn1.num_batches_tracked": torch.zeros(1, dtype=torch.int64)}
        expect_failure("wrong counter shape", wrong_shape)
        wrong_dtype = {**float_tensors, **counters, "bn1.num_batches_tracked": torch.tensor(0.0)}
        expect_failure("wrong counter dtype", wrong_dtype)
        expect_failure("unsupported dtype", {**float_tensors, **counters, "float_000": torch.arange(4, dtype=torch.float64)})
    print("voice_gender_classifier_prepare_checkpoint.py self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch, (load_file, save_file) = bind_tensor_dependencies()
    global ALLOWED_FLOAT_DTYPES
    ALLOWED_FLOAT_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})
    if args.self_test:
        if any(value is not None for value in (args.input, args.output, args.audit_json)):
            raise ValueError("--self-test accepts no fixture arguments")
        self_test()
        return 0
    if args.input is None or args.output is None or args.audit_json is None:
        raise ValueError("--input, --output, and --audit-json are required")
    audit = write_prepared_checkpoint(args.input.resolve(), args.output.resolve(), args.audit_json.resolve(), torch, load_file, save_file)
    print(
        f"normalized voice-gender checkpoint: input_tensors={audit['input_tensor_count']} "
        f"output_tensors={audit['output_tensor_count']} removed_counters={audit['input_counter_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
