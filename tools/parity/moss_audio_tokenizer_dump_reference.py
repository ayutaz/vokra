#!/usr/bin/env python3
"""Dump an independent MOSS Audio Tokenizer Full or Nano decode reference.

The oracle is the exact upstream custom-code module loaded by Hugging Face
``AutoModel.from_pretrained(..., trust_remote_code=True)`` at a pinned commit.
This script never mirrors the quantizer, patching, RoPE, attention, or channel
restore math in Python. It calls the official ``decode_codes`` and decoder
modules, then proves that path is bit-identical to the official public
``model.decode`` entry point.

Do not run this on the maintainer Mac. Run it on VAST through the repository's
Python 3.12 policy after provisioning the selected snapshot, for example::

    uv run --no-project --python 3.12 \
      --with 'torch>=2.4,<3' \
      --with 'transformers==5.15.0' \
      --with 'accelerate>=1,<2' \
      python tools/parity/moss_audio_tokenizer_dump_reference.py \
      --variant full --device cuda \
      --output /workspace/moss-audio-tokenizer-full-reference.csv

The output is intentionally not created or committed by this source-only
landing. A fixture becomes authoritative only after this script has run
against the pinned real upstream snapshot and the resulting provenance/source
hashes have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import platform
from pathlib import Path


CODEBOOK_SIZE = 1_024
VARIANTS = {
    "full": {
        "model_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "revision": "10cda397411ce6ddb802173f8d8a6c9fee3b845e",
        "sample_rate": 24_000,
        "channels": 1,
        "samples_per_channel_per_frame": 1_920,
        "max_quantizers": 32,
        "restore_channels": False,
        "model_source_sha256": (
            "65cae7744845f1b8ac65957e918cea508efe331a38e87b882b7530b6c8d7caa5"
        ),
        "config_source_sha256": (
            "349b7ff7e1b3f160f9c80df9a0311672b326b8b73e90459122fb39e6878962bf"
        ),
    },
    "nano": {
        "model_id": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        "revision": "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68",
        "sample_rate": 48_000,
        "channels": 2,
        "samples_per_channel_per_frame": 3_840,
        "max_quantizers": 16,
        "restore_channels": True,
        "model_source_sha256": None,
        "config_source_sha256": None,
    },
}


def deterministic_frame_major_codes(frames: int, num_quantizers: int) -> list[int]:
    """Input selection only; this is not a model/reference implementation."""

    return [
        (frame * 257 + quantizer * 503 + 17) % CODEBOOK_SIZE
        for frame in range(frames)
        for quantizer in range(num_quantizers)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(obj: object, label: str) -> Path:
    candidate = inspect.getsourcefile(obj)
    if candidate is None:
        raise RuntimeError(f"{label} has no inspectable source file")
    path = Path(candidate).resolve()
    # trust_remote_code copies the audited module into HF's
    # transformers_modules cache. Refuse an in-script/local mirror import.
    if "transformers_modules" not in path.parts:
        raise RuntimeError(
            f"{label} came from {path}, not Hugging Face transformers_modules; "
            "refusing a non-upstream oracle"
        )
    return path


def flat_values(tensor: object, torch_module: object, label: str) -> tuple[str, str]:
    torch = torch_module
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} is not a torch.Tensor: {type(tensor)!r}")
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{label} contains a non-finite value")
    shape = "x".join(str(axis) for axis in value.shape)
    values = ",".join(format(float(item), ".17g") for item in value.reshape(-1))
    return shape, values


def runtime_environment(torch_module: object, device: str) -> list[str]:
    """Record the execution environment before interpreting numeric output."""

    torch = torch_module
    capability = "unknown"
    get_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if get_capability is not None:
        capability = str(get_capability())
    lines = [
        (
            f"environment,cpu,{platform.processor() or 'unknown'},"
            f"machine-{platform.machine()},logical-{os.cpu_count()},"
            f"torch-capability-{capability}"
        ),
        f"environment,device,{device}",
    ]
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        lines.append(
            f"environment,cuda,{torch.cuda.get_device_name(0)},"
            f"capability-{torch.cuda.get_device_capability(0)},"
            f"runtime-{torch.version.cuda}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="nano")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--num-quantizers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    model_id = str(variant["model_id"])
    revision = str(variant["revision"])
    max_quantizers = int(variant["max_quantizers"])
    num_quantizers = (
        max_quantizers if args.num_quantizers is None else args.num_quantizers
    )
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if not 1 <= num_quantizers <= max_quantizers:
        raise ValueError(f"--num-quantizers must be in 1..={max_quantizers}")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {args.output.parent}")

    import torch
    import transformers
    from transformers import AutoModel

    torch.manual_seed(0)
    torch.set_num_threads(1)
    environment = runtime_environment(torch, args.device)
    load_kwargs = {}
    if args.device == "cuda":
        load_kwargs["device_map"] = {"": "cuda:0"}
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    model.eval()

    model_source = source_file(type(model), "model class")
    config_source = source_file(type(model.config), "config class")
    model_source_sha256 = sha256_file(model_source)
    config_source_sha256 = sha256_file(config_source)
    for label, actual, expected in (
        ("model", model_source_sha256, variant["model_source_sha256"]),
        ("config", config_source_sha256, variant["config_source_sha256"]),
    ):
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"upstream {label} source sha256 {actual} != pinned {expected}"
            )
    observed_commit = getattr(model.config, "_commit_hash", None)
    if observed_commit is not None and observed_commit != revision:
        raise RuntimeError(
            f"loaded commit {observed_commit!r}, expected {revision!r}"
        )
    expected_axes = {
        "sampling_rate": int(variant["sample_rate"]),
        "downsample_rate": int(variant["samples_per_channel_per_frame"]),
        "code_dim": 768,
    }
    if bool(variant["restore_channels"]):
        expected_axes["number_channels"] = int(variant["channels"])
    for key, expected in expected_axes.items():
        actual = getattr(model.config, key, None)
        if actual != expected:
            raise RuntimeError(f"unexpected config.{key}={actual!r}, expected {expected}")
    quantizer_config = model.config.quantizer_kwargs
    for key, expected in {
        "num_quantizers": max_quantizers,
        "codebook_size": CODEBOOK_SIZE,
        "codebook_dim": 8,
        "rvq_dim": 512,
        "output_dim": 768,
    }.items():
        actual = quantizer_config.get(key)
        if actual != expected:
            raise RuntimeError(
                f"unexpected config.quantizer_kwargs[{key!r}]={actual!r}, "
                f"expected {expected}"
            )

    frame_major = deterministic_frame_major_codes(args.frames, num_quantizers)
    input_device = next(model.parameters()).device
    codes = (
        torch.tensor(frame_major, dtype=torch.long, device=input_device)
        .reshape(args.frames, num_quantizers)
        .transpose(0, 1)
        .contiguous()
        .unsqueeze(1)
    )
    lengths = torch.tensor([args.frames], dtype=torch.long, device=input_device)
    snapshots: list[tuple[str, object]] = []
    with torch.inference_mode():
        hidden = model.quantizer.decode_codes(codes).float()
        snapshots.append(("quantizer", hidden))
        hidden_lengths = lengths
        for index, module in enumerate(model.decoder):
            hidden, hidden_lengths = module(hidden, hidden_lengths)
            snapshots.append((f"decoder_{index}", hidden))
        if bool(variant["restore_channels"]):
            direct_audio, direct_lengths = model._restore_channels_from_codec(
                hidden, hidden_lengths
            )
        else:
            direct_audio, direct_lengths = hidden, hidden_lengths
        public = model.decode(
            codes,
            num_quantizers=num_quantizers,
            return_dict=True,
        )

    public_audio = public.audio
    public_lengths = public.audio_lengths
    if public_audio is None or public_lengths is None:
        raise RuntimeError("official model.decode returned no audio/audio_lengths")
    if direct_audio.shape != public_audio.shape:
        raise RuntimeError(
            f"official direct/public shapes differ: {tuple(direct_audio.shape)} "
            f"vs {tuple(public_audio.shape)}"
        )
    if not torch.equal(direct_audio, public_audio):
        max_abs = float((direct_audio - public_audio).abs().max().item())
        raise RuntimeError(
            f"official direct/public decode mismatch; max_abs={max_abs:.17g}"
        )
    if not torch.equal(direct_lengths, public_lengths):
        raise RuntimeError(
            f"official direct/public lengths differ: {direct_lengths.tolist()} "
            f"vs {public_lengths.tolist()}"
        )
    expected_samples = args.frames * int(variant["samples_per_channel_per_frame"])
    expected_shape = (1, int(variant["channels"]), expected_samples)
    if tuple(public_audio.shape) != expected_shape:
        raise RuntimeError(
            f"audio shape {tuple(public_audio.shape)} != {expected_shape}"
        )
    if public_lengths.tolist() != [expected_samples]:
        raise RuntimeError(
            f"audio lengths {public_lengths.tolist()} != [{expected_samples}]"
        )

    lines = [
        f"source,{args.variant},{model_id},{revision}",
        f"runtime,torch-{torch.__version__},transformers-{transformers.__version__}",
        *environment,
        f"source_file,model,{model_source},{model_source_sha256}",
        f"source_file,config,{config_source},{config_source_sha256}",
        (
            f"contract,{args.frames},{num_quantizers},{CODEBOOK_SIZE},"
            f"{variant['sample_rate']},{variant['channels']},"
            f"{variant['samples_per_channel_per_frame']}"
        ),
        "codes," + ",".join(str(code) for code in frame_major),
    ]
    for label, tensor in snapshots:
        shape, values = flat_values(tensor, torch, label)
        lines.append(f"tensor,{label},{shape},{values}")
    audio_shape, audio_values = flat_values(public_audio, torch, "audio")
    lines.append(f"tensor,audio,{audio_shape},{audio_values}")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
