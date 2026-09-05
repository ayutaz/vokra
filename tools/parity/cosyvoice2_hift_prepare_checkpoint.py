#!/usr/bin/env -S uv run --frozen --project tools/parity/cosyvoice2_hift_reference python
"""Validate a CosyVoice2 HiFT checkpoint and emit a plain F32 safetensors file.

This is deliberately a VAST-only preparation step.  It does not download a
checkpoint and it never replaces an existing output.  The tensor manifest is
an input so that this utility cannot silently redefine the converter contract.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_CHECKPOINT_BYTES = 83_390_254
EXPECTED_CHECKPOINT_SHA256 = "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879"
EXPECTED_TENSOR_COUNT = 328
EXPECTED_MANIFEST_SHA256 = "cecbb2d68f91337f263db0f0333c75573516e7087b6e75d6ea647b3f86afec7c"
MANIFEST_FORMAT = "vokra-pytorch-state-dict-manifest-v1"


class PrepareError(ValueError):
    """The checkpoint cannot be admitted to the exact HiFT contract."""


def authorize_license(path: Path) -> dict[str, Any]:
    gate_path = Path(__file__).with_name("cosyvoice2_hift_reference") / "preflight_gate.py"
    spec = importlib.util.spec_from_file_location("cosyvoice2_hift_preflight", gate_path)
    if spec is None or spec.loader is None:
        raise PrepareError("preflight gate module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.gate(gate_path.with_name("pyproject.toml"), gate_path.with_name("uv.lock"), path)
    except Exception as error:
        raise PrepareError(f"license preflight blocked: {error}") from error


def is_plain_state_dict(value: object) -> bool:
    return type(value) in (dict, collections.OrderedDict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PrepareError(f"{label} must be a regular non-symlink file: {path}")


def tensor_manifest_digest(tensors: dict[str, dict[str, Any]]) -> str:
    """Hash names/shapes using the repository's StrictCheckpoint contract."""

    encoded = bytearray()
    for name in sorted(tensors):
        shape = tensors[name]["shape"]
        encoded.extend(name.encode("utf-8"))
        encoded.append(0)
        encoded.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            encoded.extend(struct.pack("<q", dimension))
    return hashlib.sha256(encoded).hexdigest()


def read_manifest(path: Path) -> dict[str, list[int]]:
    require_regular_file(path, "tensor manifest")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PrepareError(f"cannot read tensor manifest: {error}") from error
    if not isinstance(document, dict) or document.get("format") != MANIFEST_FORMAT:
        raise PrepareError("tensor manifest format is not the exact Vokra v1 format")
    if document.get("tensor_count") != EXPECTED_TENSOR_COUNT:
        raise PrepareError("tensor manifest count is not 328")
    tensors = document.get("tensors")
    if not isinstance(tensors, dict) or len(tensors) != EXPECTED_TENSOR_COUNT:
        raise PrepareError("tensor manifest must contain exactly 328 tensors")
    normalized: dict[str, dict[str, Any]] = {}
    for name, entry in tensors.items():
        if not isinstance(name, str) or not name or any(char in name for char in "\x00/\\"):
            raise PrepareError(f"invalid tensor name in manifest: {name!r}")
        if not isinstance(entry, dict) or entry.get("dtype") != "F32":
            raise PrepareError(f"tensor {name!r} is not declared as F32")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise PrepareError(f"tensor {name!r} has an invalid shape")
        normalized[name] = {"shape": shape}
    digest = tensor_manifest_digest(normalized)
    if document.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256 or digest != EXPECTED_MANIFEST_SHA256:
        raise PrepareError("tensor manifest SHA256 does not match the registered contract")
    return {name: entry["shape"] for name, entry in normalized.items()}


def load_state_dict(checkpoint: Path, expected: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    require_regular_file(checkpoint, "checkpoint")
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise PrepareError("checkpoint byte size does not match the registered artifact")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise PrepareError("checkpoint SHA256 does not match the registered artifact")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:  # torch may expose several loader exception types.
        raise PrepareError(f"torch.load(weights_only=True) failed: {error}") from error
    if not is_plain_state_dict(state):
        raise PrepareError("checkpoint is not a plain state dict")
    if set(state) != set(expected):
        raise PrepareError("checkpoint tensor names differ from the registered manifest")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(expected):
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise PrepareError(f"checkpoint tensor {name!r} is not CPU F32")
        if value.device.type != "cpu" or tuple(value.shape) != tuple(expected[name]):
            raise PrepareError(f"checkpoint tensor {name!r} shape/device differs from manifest")
        if not torch.isfinite(value).all().item():
            raise PrepareError(f"checkpoint tensor {name!r} contains non-finite values")
        result[name] = value.detach().contiguous()
    return result


def write_safetensors(path: Path, state: dict[str, torch.Tensor]) -> None:
    """Write the safetensors v1 layout without adding a runtime dependency."""

    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(state):
        tensor = state[name]
        data = tensor.numpy().tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    header["__metadata__"] = {"format": MANIFEST_FORMAT}
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks))


def self_test() -> None:
    import torch

    assert is_plain_state_dict({})
    assert is_plain_state_dict(collections.OrderedDict())
    assert not is_plain_state_dict(collections.defaultdict(dict))
    state = {"a": torch.tensor([1.0, -2.0], dtype=torch.float32)}
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-hift-preparer-") as temp:
        output = Path(temp) / "tiny.safetensors"
        write_safetensors(output, state)
        raw = output.read_bytes()
        header_size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_size])
        assert header["a"] == {"data_offsets": [0, 8], "dtype": "F32", "shape": [2]}
        assert raw[8 + header_size :] == struct.pack("<2f", 1.0, -2.0)
        existing = Path(temp) / "existing.safetensors"
        existing.write_bytes(b"keep")
        try:
            os.link(output, existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("no-replace link accepted an existing output")
    print("cosyvoice2_hift preparer self-test: OK")


def require_vast_environment() -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise PrepareError("VAST guard requires VOKRA_PUBLISH_ON_VAST=1")
    if sys.platform != "linux" or platform.machine().lower() != "x86_64":
        raise PrepareError("preparer is restricted to Linux x86_64 VAST workers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--license-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.checkpoint or not args.manifest or not args.output or not args.license_manifest:
            parser.error("--checkpoint, --manifest, --output, and --license-manifest are required")
        require_vast_environment()
        authorization = authorize_license(args.license_manifest)
        global torch
        import torch

        if args.output.exists() or args.output.is_symlink():
            raise PrepareError(f"refusing to overwrite existing output: {args.output}")
        expected = read_manifest(args.manifest)
        state = load_state_dict(args.checkpoint, expected)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
            write_safetensors(temporary, state)
            with temporary.open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, args.output)
            except FileExistsError as error:
                raise PrepareError(f"refusing to replace output created concurrently: {args.output}") from error
            os.unlink(temporary)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        print(json.dumps({
            "status": "PASS",
            "output": args.output.as_posix(),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "tensor_count": len(state),
            "tensor_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "license_manifest_sha256": authorization["license_manifest_sha256"],
            "approval_scope_sha256": authorization["approval_scope_sha256"],
            "project_sha256": authorization["project_sha256"],
            "uv_lock_sha256": authorization["lock_sha256"],
        }))
        return 0
    except (PrepareError, OSError, RuntimeError) as error:
        print(f"cosyvoice2_hift preparer: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
