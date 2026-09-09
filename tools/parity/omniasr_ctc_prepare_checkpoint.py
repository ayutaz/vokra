#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Prepare the Meta omniASR-CTC-1B fairseq2 checkpoint on VAST.

The public checkpoint is a trusted-source PyTorch pickle (``.pt``), while
``vokra-convert`` intentionally accepts only safetensors.  This worker is the
audited bridge for this one checkpoint.  It deliberately does not download
anything and must be run on VAST for the 3.9 GiB source artifact.

The pickle is loaded with ``weights_only=True``.  The worker never falls back
to unrestricted pickle execution; if the pinned checkpoint is incompatible
with PyTorch's safe unpickler, the command fails and the VAST run must record
that evidence before any narrowly reviewed loader change.

Unlike the generic ``nemo_pt_to_safetensors.py`` helper, this worker has a
fail-closed model contract: it accepts exactly the canonical fairseq2 envelope
(``{model: non-empty dict, fs2: True}``) seen in the published extraction
record, keeps only floating tensors, rejects unknown dtypes and integer
tensors, and requires the recorded 807-tensor payload.  It writes a manifest
containing every name and shape; that manifest is the input to the later
native Rust binder and is not a parity result.

Usage (on a provisioned VAST instance)::

    uv run --frozen --project tools/parity --python 3.12 python \
      tools/parity/omniasr_ctc_prepare_checkpoint.py \
      --input omniASR-CTC-1B.pt --output merged.safetensors

No model artifact or generated fixture belongs in the repository.  The
``--self-test`` path is local and does not import torch or touch a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

EXPECTED_TENSOR_COUNT = 807
EXPECTED_MODEL_ID = "facebook/omniASR-CTC-1B"
HF_REVISION = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}
INT_DTYPES = {
    "torch.bool",
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.uint8",
    "torch.uint16",
    "torch.uint32",
    "torch.uint64",
}


def sha256_file(path: Path) -> str:
    """Hash a large artifact incrementally without loading it into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_canonical_model(payload: Any) -> dict[str, Any]:
    """Return the published fairseq2 state dict, refusing loose wrappers.

    The 2026-07-28 VAST extraction recorded the exact top-level envelope
    ``{model: non-empty dict, fs2: True}``.  A permissive recursive walk could
    accidentally select an optimizer, EMA, or auxiliary model and produce a
    valid-looking GGUF.  ``fs2`` is checked as an actual bool so values such
    as ``1`` cannot masquerade as the fairseq2 marker.
    """

    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint top level must be a dict, got {type(payload)!r}")
    if set(payload) != {"model", "fs2"}:
        raise ValueError(
            "omniASR-CTC-1B checkpoint must have exactly the top-level "
            "`model` and `fs2` keys; refusing ambiguous/auxiliary payload"
        )
    if not isinstance(payload["model"], dict):
        raise ValueError(
            "omniASR-CTC-1B checkpoint top-level `model` must be a state-dict "
            f"dict, got {type(payload['model'])!r}"
        )
    if not payload["model"]:
        raise ValueError("checkpoint model state dict is empty")
    if type(payload["fs2"]) is not bool or payload["fs2"] is not True:
        raise ValueError(
            "omniASR-CTC-1B checkpoint top-level `fs2` marker must be the "
            "boolean True"
        )
    state = payload["model"]
    return state


def classify_state(state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate tensor dtypes and return tensors plus a human-readable manifest."""

    kept: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise ValueError(f"state-dict tensor name is not a string: {name!r}")
        if not hasattr(tensor, "dtype") or not hasattr(tensor, "shape"):
            raise ValueError(f"state-dict entry `{name}` is not a tensor")
        dtype = str(tensor.dtype)
        shape = [int(axis) for axis in tensor.shape]
        if dtype in INT_DTYPES:
            raise ValueError(
                f"unexpected integer tensor `{name}` dtype={dtype} shape={shape}; "
                "the pinned extraction record contains zero integer tensors"
            )
        if dtype not in KEEP_DTYPES:
            raise ValueError(
                f"unexpected tensor `{name}` dtype={dtype} shape={shape}; "
                "only F32/F16/BF16 inference tensors are accepted"
            )
        if any(axis <= 0 for axis in shape):
            raise ValueError(f"tensor `{name}` has a non-positive shape {shape}")
        detached = tensor.detach().contiguous()
        kept[name] = detached
        entries.append({"name": name, "dtype": dtype, "shape": shape})
    entries.sort(key=lambda item: item["name"])
    if len(kept) != EXPECTED_TENSOR_COUNT:
        raise ValueError(
            f"omniASR-CTC-1B payload has {len(kept)} floating tensors; "
            f"the pinned VAST extraction record requires exactly "
            f"{EXPECTED_TENSOR_COUNT}. Refusing an unrecognised checkpoint."
        )
    return kept, entries


def deduplicate_shared_tensors(tensors: dict[str, Any]) -> list[dict[str, str]]:
    """Clone later aliases so safetensors can serialize the state dict.

    Tied parameters are valid PyTorch state-dict contents but
    ``safetensors.torch.save_file`` rejects shared storage.  Cloning only the
    later alias preserves values and makes the conversion deterministic; the
    manifest records the operation for the later native binder audit.
    """

    seen: dict[int, str] = {}
    shared_pairs: list[dict[str, str]] = []
    for name, tensor in list(tensors.items()):
        data_ptr = getattr(tensor, "data_ptr", None)
        if data_ptr is None:
            continue
        pointer = int(data_ptr())
        previous = seen.get(pointer)
        if previous is None:
            seen[pointer] = name
            continue
        tensors[name] = tensor.clone().contiguous()
        shared_pairs.append({"canonical": previous, "cloned": name})
    return shared_pairs


def self_test() -> None:
    class FakeTensor:
        dtype = "torch.float32"
        shape = (2, 3)

        def detach(self) -> "FakeTensor":
            return self

        def contiguous(self) -> "FakeTensor":
            return self

    class SharedFakeTensor(FakeTensor):
        def __init__(self, pointer: int) -> None:
            self.pointer = pointer

        def data_ptr(self) -> int:
            return self.pointer

        def clone(self) -> "SharedFakeTensor":
            return SharedFakeTensor(self.pointer + 1)

    valid_state = {"weight": FakeTensor()}
    assert unwrap_canonical_model({"model": valid_state, "fs2": True}) is valid_state

    def assert_rejected(payload: Any, expected: str) -> None:
        try:
            unwrap_canonical_model(payload)
        except ValueError as error:
            assert expected in str(error), (expected, str(error))
        else:
            raise AssertionError(f"payload must fail closed: {payload!r}")

    assert_rejected([], "top level must be a dict")
    assert_rejected({"state_dict": valid_state, "fs2": True}, "exactly the top-level")
    assert_rejected(
        {"model": valid_state, "fs2": True, "optimizer": {}},
        "exactly the top-level",
    )
    assert_rejected({"fs2": True}, "exactly the top-level")
    assert_rejected({"model": valid_state}, "exactly the top-level")
    assert_rejected({"model": [], "fs2": True}, "must be a state-dict dict")
    assert_rejected({"model": {}, "fs2": True}, "state dict is empty")
    assert_rejected({"model": valid_state, "fs2": False}, "boolean True")
    for invalid_marker in (1, 0, "true", None):
        assert_rejected({"model": valid_state, "fs2": invalid_marker}, "boolean True")
    try:
        classify_state({"weight": FakeTensor()})
    except ValueError as error:
        assert "807" in str(error)
    else:
        raise AssertionError("wrong tensor count must fail")
    class IntFakeTensor(FakeTensor):
        dtype = "torch.int64"

    try:
        classify_state({"counter": IntFakeTensor()})
    except ValueError as error:
        assert "integer tensor" in str(error)
    else:
        raise AssertionError("integer tensors must fail")
    tensors = {
        "encoder.weight": SharedFakeTensor(7),
        "head.weight": SharedFakeTensor(7),
    }
    pairs = deduplicate_shared_tensors(tensors)
    assert pairs == [{"canonical": "encoder.weight", "cloned": "head.weight"}]
    assert tensors["encoder.weight"].data_ptr() != tensors["head.weight"].data_ptr()
    with TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "source.pt"
        source.write_bytes(b"offline digest probe")
        assert sha256_file(source) == hashlib.sha256(b"offline digest probe").hexdigest()
    assert EXPECTED_MODEL_ID == "facebook/omniASR-CTC-1B"
    assert HF_REVISION == "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
    print("omniasr_ctc_prepare_checkpoint: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        source_sha256 = sha256_file(args.input)
    except OSError as error:
        print(f"could not hash input: {error}", file=sys.stderr)
        return 2

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        print(f"missing VAST-side dependency: {error}", file=sys.stderr)
        return 2

    try:
        payload = torch.load(str(args.input), map_location="cpu", weights_only=True)
        state = unwrap_canonical_model(payload)
        kept, entries = classify_state(state)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
        print(f"omniasr_ctc_prepare_checkpoint: refusing input: {error}", file=sys.stderr)
        return 3

    shared_pairs = deduplicate_shared_tensors(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))
    digest = hashlib.sha256()
    with args.output.open("rb") as output_file:
        for chunk in iter(lambda: output_file.read(1024 * 1024), b""):
            digest.update(chunk)
    output_sha256 = digest.hexdigest()
    manifest = {
        "model_id": EXPECTED_MODEL_ID,
        "hf_revision": HF_REVISION,
        "source": str(args.input),
        "source_sha256": source_sha256,
        "output": str(args.output),
        "expected_tensor_count": EXPECTED_TENSOR_COUNT,
        "tensor_count": len(entries),
        "integer_tensor_count": 0,
        "unknown_dtype_count": 0,
        "shared_tensor_pairs": shared_pairs,
        "output_sha256": output_sha256,
        "tensors": entries,
        "note": (
            "Manifest generated by the upstream fairseq2 checkpoint loader; "
            "it is an inspection/binding input, not a numerical parity fixture."
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"omniasr_ctc_prepare_checkpoint: kept {len(entries)} tensors; "
        f"safetensors={args.output}; manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
