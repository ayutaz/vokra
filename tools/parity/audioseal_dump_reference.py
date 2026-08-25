#!/usr/bin/env python3
"""Dump independent AudioSeal generator/detector reference tensors.

This oracle imports Meta's AudioSeal implementation from the exact source
revision below and loads the exact official checkpoint files verified by
``audioseal_prepare_checkpoint.py``. It does not mirror the Rust forward.

Run on VAST through the repository's Python 3.12 policy::

    uv run --no-project --python 3.12 --with torch --with numpy --with omegaconf \
      python tools/parity/audioseal_dump_reference.py \
      --audioseal-source /workspace/audioseal \
      --variant base \
      --generator /workspace/checkpoints/generator_base.pth \
      --detector /workspace/checkpoints/detector_base.pth \
      --output-dir /workspace/reference/audioseal-base

Repeat with ``--variant streaming`` and the two streaming checkpoints. The
streaming-trained checkpoint is evaluated as one complete buffer, matching the
initial Rust binder; this command does not enter AudioSeal's stateful
``model.streaming(...)`` context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from audioseal_prepare_checkpoint import (
    CHECKPOINT_REVISION,
    INPUTS,
    SOURCE_REVISION,
    require_source,
)


SAMPLE_RATE = 16_000
NBITS = 16
DEFAULT_SAMPLES = 1_283
MESSAGE = tuple((index * 5 + 1) % 2 for index in range(NBITS))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_source_checkout(path: Path) -> Path:
    source = path.resolve()
    loader = source / "src" / "audioseal" / "loader.py"
    if not loader.is_file():
        raise ValueError(f"AudioSeal checkout has no {loader.relative_to(source)}")
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"AudioSeal source revision {revision} != pinned {SOURCE_REVISION}"
        )
    return source


def deterministic_pcm(samples: int) -> Any:
    import numpy as np

    if samples < 321:
        raise ValueError("--samples must be >= 321 to exercise more than one hop")
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    envelope = np.linspace(0.35, 1.0, samples, dtype=np.float64)
    pcm = envelope * (
        0.17 * np.sin(2.0 * np.pi * 173.0 * time)
        + 0.09 * np.cos(2.0 * np.pi * 431.0 * time + 0.2)
        + 0.025 * np.sin(2.0 * np.pi * 997.0 * time + 0.7)
    )
    return pcm.astype(np.float32)


def as_f32(value: Any) -> Any:
    import numpy as np

    return value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)


def write_array(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256_file(path),
    }


def dump(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    source = require_source_checkout(args.audioseal_source)
    generator_prefix = f"generator_{args.variant}"
    detector_prefix = f"detector_{args.variant}"
    require_source(args.generator, generator_prefix)
    require_source(args.detector, detector_prefix)

    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    sys.path.insert(0, str(source / "src"))
    from audioseal.loader import AudioSeal
    import audioseal

    imported = Path(audioseal.__file__).resolve()
    if source not in imported.parents:
        raise ValueError(
            f"imported audioseal from {imported}, outside pinned checkout {source}"
        )

    torch.manual_seed(0)
    torch.set_num_threads(1)
    generator = AudioSeal.load_generator(str(args.generator), device=torch.device("cpu"))
    detector = AudioSeal.load_detector(str(args.detector), device=torch.device("cpu"))
    generator.eval()
    detector.eval()

    pcm_np = deterministic_pcm(args.samples)
    message_np = np.asarray(MESSAGE, dtype=np.uint8)
    pcm = torch.from_numpy(pcm_np).reshape(1, 1, -1)
    message = torch.from_numpy(message_np.astype(np.int64)).reshape(1, -1)

    with torch.no_grad():
        latent = generator.encoder(pcm)
        conditioned = generator.msg_processor(latent, message)
        decoded = generator.decoder(conditioned)[..., : args.samples]
        watermark = generator.get_watermark(
            pcm, sample_rate=SAMPLE_RATE, message=message
        )
        if not torch.equal(decoded, watermark):
            raise ValueError("official get_watermark differs from explicit encoder/decoder")
        embedded = generator(
            pcm, sample_rate=SAMPLE_RATE, message=message, alpha=1.0
        )
        if not torch.equal(embedded, pcm + watermark):
            raise ValueError("official generator forward differs from pcm + watermark")

        raw_detector_logits = detector.detector(embedded)
        detection_binary, message_probabilities = detector(
            embedded, sample_rate=SAMPLE_RATE
        )
        detection_probability, detected_message = detector.detect_watermark(
            embedded,
            sample_rate=SAMPLE_RATE,
            detection_threshold=0.5,
            message_threshold=0.5,
        )

    tensors: dict[str, dict[str, Any]] = {}
    values = {
        "input_pcm": pcm_np,
        "message": message_np,
        "generator_latent": as_f32(latent)[0],
        "generator_conditioned": as_f32(conditioned)[0],
        "watermark": as_f32(watermark)[0, 0],
        "embedded": as_f32(embedded)[0, 0],
        "raw_detector_logits": as_f32(raw_detector_logits)[0],
        "detector_positive": as_f32(detection_binary)[0, 1],
        "message_probabilities": as_f32(message_probabilities)[0],
        "detection_probability": as_f32(detection_probability).reshape(1),
        "detected_message": detected_message.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.uint8, copy=False)[0],
    }
    for name, value in values.items():
        suffix = "u8" if np.asarray(value).dtype == np.uint8 else "f32le"
        tensors[name] = write_array(args.output_dir / f"{name}.{suffix}", value)

    metadata = {
        "schema": "vokra-audioseal-reference-v1",
        "source": "facebookresearch/audioseal official Python forward",
        "source_revision": SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "variant": args.variant,
        "sample_rate": SAMPLE_RATE,
        "samples": args.samples,
        "generator": {
            "file": INPUTS[generator_prefix][0],
            "sha256": sha256_file(args.generator),
        },
        "detector": {
            "file": INPUTS[detector_prefix][0],
            "sha256": sha256_file(args.detector),
        },
        "audioseal_import": str(imported),
        "torch_version": torch.__version__,
        "tensors": tensors,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


def self_test() -> None:
    assert len(MESSAGE) == NBITS
    assert set(MESSAGE) == {0, 1}
    assert DEFAULT_SAMPLES % 320 != 0
    assert INPUTS["generator_base"][2] == 101
    assert INPUTS["generator_streaming"][2] == 101
    assert INPUTS["detector_base"][2] == 54
    assert INPUTS["detector_streaming"][2] == 54
    print("audioseal_dump_reference self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audioseal-source", type=Path)
    parser.add_argument("--variant", choices=("base", "streaming"))
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        missing = [
            name
            for name in (
                "audioseal_source",
                "variant",
                "generator",
                "detector",
                "output_dir",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("missing required arguments: " + ", ".join(missing))
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
