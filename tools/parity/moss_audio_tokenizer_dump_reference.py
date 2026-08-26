#!/usr/bin/env python3
"""Dump an independent MOSS Audio Tokenizer Nano decode reference.

The oracle is the exact upstream custom-code module loaded by Hugging Face
``AutoModel.from_pretrained(..., trust_remote_code=True)`` at a pinned commit.
This script never mirrors the quantizer, patching, RoPE, attention, or channel
restore math in Python. It calls the official ``decode_codes`` and decoder
modules, then proves that path is bit-identical to the official public
``model.decode`` entry point.

Do not run this on the maintainer Mac. Run it on VAST through the repository's
Python 3.12 policy after provisioning the Nano snapshot, for example::

    uv run --no-project --python 3.12 \
      --with 'torch>=2.4,<3' \
      --with 'transformers==5.15.0' \
      --with 'accelerate>=1,<2' \
      python tools/parity/moss_audio_tokenizer_dump_reference.py \
      --output /workspace/moss-audio-tokenizer-nano-reference.csv

The output is intentionally not created or committed by this source-only
landing. A fixture becomes authoritative only after this script has run
against the pinned real upstream snapshot and the resulting provenance/source
hashes have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path


MODEL_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
MODEL_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLES_PER_CHANNEL_PER_FRAME = 3_840
CODEBOOK_SIZE = 1_024
MAX_QUANTIZERS = 16


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--num-quantizers", type=int, default=MAX_QUANTIZERS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if not 1 <= args.num_quantizers <= MAX_QUANTIZERS:
        raise ValueError(f"--num-quantizers must be in 1..={MAX_QUANTIZERS}")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    if not args.output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {args.output.parent}")

    import torch
    import transformers
    from transformers import AutoModel

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = AutoModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    model_source = source_file(type(model), "model class")
    config_source = source_file(type(model.config), "config class")
    observed_commit = getattr(model.config, "_commit_hash", None)
    if observed_commit is not None and observed_commit != args.revision:
        raise RuntimeError(
            f"loaded commit {observed_commit!r}, expected {args.revision!r}"
        )
    expected_axes = {
        "sampling_rate": SAMPLE_RATE,
        "number_channels": CHANNELS,
        "downsample_rate": SAMPLES_PER_CHANNEL_PER_FRAME,
        "code_dim": 768,
    }
    for key, expected in expected_axes.items():
        actual = getattr(model.config, key, None)
        if actual != expected:
            raise RuntimeError(f"unexpected config.{key}={actual!r}, expected {expected}")
    quantizer_config = model.config.quantizer_kwargs
    for key, expected in {
        "num_quantizers": MAX_QUANTIZERS,
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

    frame_major = deterministic_frame_major_codes(args.frames, args.num_quantizers)
    codes = (
        torch.tensor(frame_major, dtype=torch.long)
        .reshape(args.frames, args.num_quantizers)
        .transpose(0, 1)
        .contiguous()
        .unsqueeze(1)
    )
    lengths = torch.tensor([args.frames], dtype=torch.long)
    snapshots: list[tuple[str, object]] = []
    with torch.inference_mode():
        hidden = model.quantizer.decode_codes(codes).float()
        snapshots.append(("quantizer", hidden))
        hidden_lengths = lengths
        for index, module in enumerate(model.decoder):
            hidden, hidden_lengths = module(hidden, hidden_lengths)
            snapshots.append((f"decoder_{index}", hidden))
        direct_audio, direct_lengths = model._restore_channels_from_codec(
            hidden, hidden_lengths
        )
        public = model.decode(
            codes,
            num_quantizers=args.num_quantizers,
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
    expected_samples = args.frames * SAMPLES_PER_CHANNEL_PER_FRAME
    expected_shape = (1, CHANNELS, expected_samples)
    if tuple(public_audio.shape) != expected_shape:
        raise RuntimeError(
            f"audio shape {tuple(public_audio.shape)} != {expected_shape}"
        )
    if public_lengths.tolist() != [expected_samples]:
        raise RuntimeError(
            f"audio lengths {public_lengths.tolist()} != [{expected_samples}]"
        )

    lines = [
        f"source,{args.model_id},{args.revision}",
        f"runtime,torch-{torch.__version__},transformers-{transformers.__version__}",
        f"source_file,model,{model_source},{sha256_file(model_source)}",
        f"source_file,config,{config_source},{sha256_file(config_source)}",
        (
            f"contract,{args.frames},{args.num_quantizers},{CODEBOOK_SIZE},"
            f"{SAMPLE_RATE},{CHANNELS},{SAMPLES_PER_CHANNEL_PER_FRAME}"
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
