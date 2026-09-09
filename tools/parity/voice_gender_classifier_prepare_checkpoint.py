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


def reject_symlink_ancestry(path: Path | str, label: str) -> None:
    raw = os.fspath(path)
    if not raw or any(component in {".", ".."} for component in raw.split("/")):
        raise ValueError(f"{label} must not contain lexical dot components")
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        # macOS exposes /var as the system /private/var alias; it is not
        # user-controlled output redirection and is safe to traverse.
        if ancestor.is_symlink() and ancestor != Path("/var"):
            raise ValueError(f"{label} has symlink ancestry: {ancestor}")


def cleanup_temp(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


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


def publish_pair(
    output_temp: Path, output: Path, audit_temp: Path, audit: Path
) -> None:
    """Claim output+audit as a pair, rolling back only our output claim."""
    output_claimed = False
    try:
        reject_symlink_ancestry(output, "output")
        if output.parent.is_symlink() or not output.parent.is_dir():
            raise ValueError("output parent must be an existing regular non-symlink directory")
        os.link(output_temp, output)
        output_claimed = True
        reject_symlink_ancestry(audit, "audit")
        if audit.parent.is_symlink() or not audit.parent.is_dir():
            raise ValueError("audit parent must be an existing regular non-symlink directory")
        os.link(audit_temp, audit)
    except BaseException:
        if output_claimed:
            try:
                if os.path.samestat(
                    os.stat(output_temp, follow_symlinks=False),
                    os.stat(output, follow_symlinks=False),
                ):
                    output.unlink()
            except OSError:
                pass
        raise
    finally:
        cleanup_temp(output_temp)
        cleanup_temp(audit_temp)


def write_prepared_checkpoint(
    input_path: Path | str,
    output_path: Path | str,
    audit_path: Path | str,
    torch: Any,
    load_file: Any,
    save_file: Any,
) -> dict[str, Any]:
    raw_input = os.fspath(input_path)
    raw_output = os.fspath(output_path)
    raw_audit = os.fspath(audit_path)
    reject_symlink_ancestry(raw_input, "input")
    reject_symlink_ancestry(raw_output, "output")
    reject_symlink_ancestry(raw_audit, "audit")
    input_path = Path(raw_input)
    output_path = Path(raw_output)
    audit_path = Path(raw_audit)
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
    reject_symlink_ancestry(output_path, "output")
    reject_symlink_ancestry(audit_path, "audit")
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise ValueError("output parent must be an existing regular non-symlink directory")
    if audit_path.parent.is_symlink() or not audit_path.parent.is_dir():
        raise ValueError("audit parent must be an existing regular non-symlink directory")
    temporary_path: Path | None = None
    temporary_audit: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        save_file(floating, str(temporary_path))
        prepared = load_file(str(temporary_path), device="cpu")
        if tensor_manifest(prepared, torch) != floating_manifest:
            raise ValueError("prepared checkpoint changed a floating tensor name, shape, dtype, or value")
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())

        input_bytes = input_path.stat().st_size
        output_bytes = temporary_path.stat().st_size
        audit = {
            "schema": "vokra-voice-gender-checkpoint-normalization-v1",
            "status": "AUTHENTICATED_NORMALIZED",
            "input_file": input_path.name,
            "input_bytes": input_bytes,
            "input_sha256": sha256_bytes(input_path.read_bytes()),
            "output_file": output_path.name,
            "output_bytes": output_bytes,
            "output_sha256": sha256_bytes(temporary_path.read_bytes()),
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
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=audit_path.parent,
            prefix=f".{audit_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_audit = Path(handle.name)
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_pair(temporary_path, output_path, temporary_audit, audit_path)
        return audit
    finally:
        if temporary_audit is not None:
            cleanup_temp(temporary_audit)
        if temporary_path is not None:
            cleanup_temp(temporary_path)


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
        assert not list(root.glob(".*.tmp")), "successful publish leaked a temporary link"
        existing_output = root / "existing.safetensors"
        existing_audit = root / "existing.json"
        existing_output.write_bytes(b"keep-output")
        existing_audit.write_bytes(b"keep-audit")
        try:
            write_prepared_checkpoint(
                input_path, existing_output, existing_audit, torch, load_file, save_file
            )
        except ValueError:
            pass
        else:
            raise AssertionError("existing output was accepted")
        assert existing_output.read_bytes() == b"keep-output"
        assert existing_audit.read_bytes() == b"keep-audit"
        first = root / "first.safetensors"
        second_audit = root / "second.json"
        first_temp = root / ".first.tmp"
        audit_temp = root / ".audit.tmp"
        first_temp.write_bytes(b"first")
        audit_temp.write_bytes(b"audit")
        second_audit.write_bytes(b"keep-audit")
        try:
            publish_pair(first_temp, first, audit_temp, second_audit)
        except FileExistsError:
            pass
        else:
            raise AssertionError("audit collision was accepted")
        assert not first.exists()
        assert not first_temp.exists() and not audit_temp.exists()
        assert second_audit.read_bytes() == b"keep-audit"
        from unittest.mock import patch

        cleanup_output = root / "cleanup.safetensors"
        cleanup_audit = root / "cleanup.json"
        cleanup_output_temp = root / ".cleanup-output.tmp"
        cleanup_audit_temp = root / ".cleanup-audit.tmp"
        cleanup_output_temp.write_bytes(b"complete-output")
        cleanup_audit_temp.write_bytes(b"complete-audit")
        with patch.object(Path, "unlink", side_effect=PermissionError("test cleanup failure")):
            publish_pair(
                cleanup_output_temp,
                cleanup_output,
                cleanup_audit_temp,
                cleanup_audit,
            )
        assert cleanup_output.read_bytes() == b"complete-output"
        assert cleanup_audit.read_bytes() == b"complete-audit"
        os.unlink(cleanup_output_temp)
        os.unlink(cleanup_audit_temp)
        for dotted in (str(root) + "/./prepared.safetensors", str(root) + "/../prepared.safetensors"):
            try:
                dotted_path = Path(dotted)
                write_prepared_checkpoint(
                    input_path, dotted, dotted_path.with_suffix(".json"), torch, load_file, save_file
                )
            except ValueError:
                pass
            else:
                raise AssertionError(f"lexical dot component accepted: {dotted}")
        real_parent = root / "real"
        real_parent.mkdir()
        symlink_parent = root / "link"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        try:
            write_prepared_checkpoint(
                input_path,
                symlink_parent / "prepared.safetensors",
                symlink_parent / "audit.json",
                torch,
                load_file,
                save_file,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("symlink ancestor accepted")

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


def self_test_contract() -> None:
    """Exercise preparation invariants without importing Torch or weights.

    The full self-test additionally round-trips synthetic safetensors tensors
    when the VAST dependency environment is available. This small contract
    pass lets CI and Apple-side gate checks verify constants and path fences
    without importing or downloading anything locally.
    """
    if len(EXPECTED_COUNTER_NAMES) != EXPECTED_COUNTER_COUNT:
        raise AssertionError("counter manifest cardinality drifted")
    if not all(name.endswith(COUNTER_SUFFIX) for name in EXPECTED_COUNTER_NAMES):
        raise AssertionError("counter manifest contains a non-counter name")
    source = Path(__file__).read_text(encoding="utf-8")
    for marker in (
        "AUTHENTICATED_NORMALIZED",
        "EXPECTED_FLOATING_TENSOR_COUNT = 202",
        "refusing to overwrite output",
        "publish_pair",
        "remove_authenticated_batchnorm_num_batches_tracked_v1",
    ):
        if marker not in source:
            raise AssertionError(f"preparation contract marker missing: {marker}")
    for bad in (
        "",
        ".",
        "..",
        "/tmp/./prepared.safetensors",
        "/tmp/../prepared.safetensors",
    ):
        try:
            reject_symlink_ancestry(bad, "self-test")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe self-test path accepted: {bad!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--audit-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        if any(value is not None for value in (args.input, args.output, args.audit_json)):
            raise ValueError("--self-test accepts no fixture arguments")
        try:
            bind_tensor_dependencies()
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "safetensors"}:
                raise
            self_test_contract()
            print("voice_gender_classifier_prepare_checkpoint.py contract self-test: PASS")
            return 0
        self_test()
        return 0
    torch, (load_file, save_file) = bind_tensor_dependencies()
    global ALLOWED_FLOAT_DTYPES
    ALLOWED_FLOAT_DTYPES = frozenset({torch.float16, torch.float32, torch.bfloat16})
    if args.input is None or args.output is None or args.audit_json is None:
        raise ValueError("--input, --output, and --audit-json are required")
    audit = write_prepared_checkpoint(args.input, args.output, args.audit_json, torch, load_file, save_file)
    print(
        f"normalized voice-gender checkpoint: input_tensors={audit['input_tensor_count']} "
        f"output_tensors={audit['output_tensor_count']} removed_counters={audit['input_counter_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
