#!/usr/bin/env python3
"""Dump an independent official MioCodec v2 decode reference.

The oracle imports ``MioCodecModel`` from an exact clean checkout of the
official Aratako/MioCodec source, verifies the pinned Hugging Face config and
safetensors digests, then calls the upstream public ``decode`` method. Forward
hooks capture official-module stage outputs; this file does not mirror the
decoder equations or import Vokra.

Run this only on VAST through the pinned upstream checkout's Python 3.12
``pyproject.toml`` and ``uv.lock``. The checkpoint and torchaudio WavLM cache
must not be materialized on the maintainer Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


UPSTREAM_REPOSITORY = "https://github.com/Aratako/MioCodec"
SOURCE_REVISION = "77473544375d57e96cbdfd5d7d257e8f280fa8e3"
HF_REPOSITORY = "Aratako/MioCodec-25Hz-44.1kHz-v2"
HF_REVISION = "67faba34153fe74e6665991c432a7327e23c5c1c"
MODEL_SHA256 = "8e319ef2231bad184f17cb73fd5a21b685c25c6c1622ef33ed9271187e81cd4a"
CONFIG_SHA256 = "bfabffffaaa5709b8dc69585111ee3d53c1b0609c23d293cd1b4903eafa5bec1"
MODEL_BYTES = 528_105_436
CONFIG_BYTES = 2_705
TENSOR_COUNT = 350
SAMPLE_RATE = 44_100
HOP_LENGTH = 98
UPSAMPLE_TOTAL = 9
GLOBAL_DIM = 128
CODEBOOK_SIZE = 12_800
DECODE_INPUT_MAGIC = b"VKRMIO01"
DEFAULT_TARGET_SAMPLES = 14_111
DEFAULT_CODES = (0, 1, 7, 8, 127, 128, 4_096, 12_799)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_source_checkout(checkout: Path) -> Path:
    checkout = checkout.resolve()
    entry = checkout / "src" / "miocodec" / "model.py"
    if not entry.is_file():
        raise ValueError(f"{checkout} is not an official MioCodec source checkout")
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"MioCodec source revision {revision!r} != pinned {SOURCE_REVISION!r}"
        )
    dirty = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(
            "MioCodec source checkout is dirty; reference provenance requires "
            f"an exact clean {SOURCE_REVISION} tree"
        )
    return checkout


def validate_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned input: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes != pinned {expected_bytes}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest} != pinned {expected_sha256}")


def deterministic_global_embedding() -> tuple[float, ...]:
    values = []
    for index in range(GLOBAL_DIM):
        value = 0.17 * math.sin((index + 1) * 0.23) + 0.09 * math.cos(
            (index + 1) * 0.071 + 0.2
        )
        values.append(value)
    return tuple(values)


def output_samples(target_samples: int) -> int:
    pre_frames = target_samples // HOP_LENGTH // UPSAMPLE_TOTAL
    if pre_frames == 0:
        raise ValueError(
            f"target_samples {target_samples} is shorter than "
            f"{HOP_LENGTH * UPSAMPLE_TOTAL}"
        )
    return pre_frames * UPSAMPLE_TOTAL * HOP_LENGTH


def encode_vkrmio01(
    target_samples: int, global_embedding: tuple[float, ...], codes: tuple[int, ...]
) -> bytes:
    output_samples(target_samples)
    if len(global_embedding) != GLOBAL_DIM:
        raise ValueError(f"global embedding length must be {GLOBAL_DIM}")
    if not codes or len(codes) > 0xFFFF_FFFF:
        raise ValueError("code count is outside VKRMIO01 u32")
    if any(code < 0 or code >= CODEBOOK_SIZE for code in codes):
        raise ValueError(f"codes must be in 0..{CODEBOOK_SIZE}")
    if any(not math.isfinite(value) for value in global_embedding):
        raise ValueError("global embedding contains a non-finite value")
    payload = bytearray(DECODE_INPUT_MAGIC)
    payload.extend(struct.pack("<QII", target_samples, len(codes), 0))
    payload.extend(struct.pack(f"<{GLOBAL_DIM}f", *global_embedding))
    payload.extend(struct.pack(f"<{len(codes)}I", *codes))
    return bytes(payload)


def as_f32(value: Any) -> Any:
    import numpy as np

    return value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def write_array(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value, dtype="<f4")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": "float32-le",
        "shape": list(array.shape),
        "sha256": sha256_file(path),
    }


def capture_hook(captured: dict[str, Any], name: str):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if isinstance(output, tuple):
            if len(output) != 1:
                raise RuntimeError(f"{name}: unexpected tuple output length {len(output)}")
            output = output[0]
        captured[name] = as_f32(output)

    return hook


def validate_checkpoint_state(model: Any, weights_path: Path) -> tuple[int, list[str]]:
    import torch
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device="cpu")
    if len(state) != TENSOR_COUNT:
        raise ValueError(f"checkpoint has {len(state)} tensors, expected {TENSOR_COUNT}")
    non_f32 = sorted(name for name, value in state.items() if value.dtype != torch.float32)
    if non_f32:
        raise ValueError(f"checkpoint contains non-F32 tensors: {non_f32[:8]!r}")

    model_state = model.state_dict()
    unexpected = sorted(set(state) - set(model_state))
    if unexpected:
        raise ValueError(f"checkpoint has unexpected model tensors: {unexpected[:8]!r}")
    for name, value in state.items():
        if tuple(value.shape) != tuple(model_state[name].shape):
            raise ValueError(
                f"{name}: checkpoint shape {tuple(value.shape)} != official module "
                f"shape {tuple(model_state[name].shape)}"
            )

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"official module rejected checkpoint keys: {incompatible.unexpected_keys!r}"
        )
    allowed_missing = [
        name
        for name in incompatible.missing_keys
        if name.startswith("ssl_feature_extractor.")
        or name == "istft_head.istft.window"
    ]
    if sorted(allowed_missing) != sorted(incompatible.missing_keys):
        disallowed = sorted(set(incompatible.missing_keys) - set(allowed_missing))
        raise ValueError(f"checkpoint is missing decoder/model tensors: {disallowed[:8]!r}")
    return len(state), sorted(incompatible.missing_keys)


def dump(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    source = validate_source_checkout(args.source)
    validate_file(args.weights, MODEL_BYTES, MODEL_SHA256)
    validate_file(args.config, CONFIG_BYTES, CONFIG_SHA256)
    if args.output.exists():
        raise ValueError(f"output directory already exists: {args.output}")

    sys.path.insert(0, str(source / "src"))
    try:
        import miocodec
        from miocodec import MioCodecModel
    finally:
        sys.path.pop(0)
    imported = Path(miocodec.__file__).resolve()
    if source not in imported.parents:
        raise ValueError(f"imported miocodec from {imported}, outside {source}")

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MioCodecModel.from_pretrained(
        config_path=str(args.config), weights_path=str(args.weights)
    )
    model.eval().cpu()
    loaded_count, missing_keys = validate_checkpoint_state(model, args.weights)

    captured: dict[str, Any] = {}
    modules = {
        "content_embedding": model.local_quantizer.proj_out,
        "wave_prenet": model.wave_prenet,
        "wave_conv_upsample": model.wave_conv_upsample,
        "wave_prior_net": model.wave_prior_net,
        "wave_decoder": model.wave_decoder,
        "wave_post_net": model.wave_post_net,
        "wave_upsampler": model.wave_upsampler,
        "istft_parameters": model.istft_head.out,
        "decoded_pcm": model.istft_head,
    }
    if any(module is None for module in modules.values()):
        missing = sorted(name for name, module in modules.items() if module is None)
        raise ValueError(f"pinned config lacks decoder modules: {missing!r}")
    handles = [
        module.register_forward_hook(capture_hook(captured, name))
        for name, module in modules.items()
    ]

    codes = tuple(args.codes or DEFAULT_CODES)
    global_embedding = deterministic_global_embedding()
    code_tensor = torch.tensor(codes, dtype=torch.long)
    global_tensor = torch.tensor(global_embedding, dtype=torch.float32)
    try:
        with torch.inference_mode():
            decoded = model.decode(
                global_embedding=global_tensor,
                content_token_indices=code_tensor,
                target_audio_length=args.target_samples,
            )
    finally:
        for handle in handles:
            handle.remove()

    expected_samples = output_samples(args.target_samples)
    if tuple(decoded.shape) != (expected_samples,):
        raise RuntimeError(
            f"official decode shape {tuple(decoded.shape)} != ({expected_samples},)"
        )
    if set(captured) != set(modules):
        raise RuntimeError(
            f"official hook set {sorted(captured)!r} != expected {sorted(modules)!r}"
        )
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official MioCodec emitted non-finite PCM")
    if not np.array_equal(captured["decoded_pcm"].reshape(-1), as_f32(decoded)):
        raise RuntimeError("official ISTFT hook output differs from public decode output")

    args.output.mkdir(parents=True)
    vmi_path = args.output / "decode_input.vmi"
    vmi_path.write_bytes(
        encode_vkrmio01(args.target_samples, global_embedding, codes)
    )
    tensors = {
        name: write_array(args.output / f"{name}.f32le", value)
        for name, value in captured.items()
    }
    manifest = {
        "format": "vokra-miocodec-reference-v1",
        "oracle": "official MioCodecModel.decode with official-module forward hooks",
        "source_repository": UPSTREAM_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "model_sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "miocodec_import": str(imported),
        "torch": str(torch.__version__),
        "official_checkpoint_tensors_loaded": loaded_count,
        "official_defaulted_state_keys": missing_keys,
        "sample_rate": SAMPLE_RATE,
        "target_samples": args.target_samples,
        "output_samples": expected_samples,
        "codes": list(codes),
        "decode_input": {
            "file": vmi_path.name,
            "bytes": vmi_path.stat().st_size,
            "sha256": sha256_file(vmi_path),
        },
        "tensors": tensors,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def self_test() -> None:
    embedding = deterministic_global_embedding()
    payload = encode_vkrmio01(DEFAULT_TARGET_SAMPLES, embedding, DEFAULT_CODES)
    expected_bytes = 8 + 8 + 4 + 4 + GLOBAL_DIM * 4 + len(DEFAULT_CODES) * 4
    assert len(payload) == expected_bytes
    assert payload[:8] == DECODE_INPUT_MAGIC
    target, count, reserved = struct.unpack("<QII", payload[8:24])
    assert target == DEFAULT_TARGET_SAMPLES
    assert count == len(DEFAULT_CODES)
    assert reserved == 0
    assert output_samples(DEFAULT_TARGET_SAMPLES) == 13_230
    assert len(embedding) == GLOBAL_DIM and all(math.isfinite(v) for v in embedding)
    assert min(DEFAULT_CODES) == 0 and max(DEFAULT_CODES) == CODEBOOK_SIZE - 1
    print("miocodec_dump_reference self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    parser.add_argument("--codes", nargs="+", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [
            name
            for name in ("source", "config", "weights", "output")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
        if args.target_samples < HOP_LENGTH * UPSAMPLE_TOTAL:
            parser.error(
                f"--target-samples must be >= {HOP_LENGTH * UPSAMPLE_TOTAL}"
            )
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
