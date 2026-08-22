#!/usr/bin/env python3
"""Dump one real-weight NVIDIA BigVGAN mel-to-waveform reference.

The oracle imports upstream ``bigvgan.py`` from a caller-supplied checkout and
loads the official ``bigvgan_generator.pt``. A tiny stand-in for upstream's
plotting-heavy ``utils`` module supplies only the two functions imported by
``bigvgan.py``; no model math is mirrored here.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch  # type: ignore[import-not-found]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    upstream = args.upstream_dir.resolve()
    sys.path.insert(0, str(upstream))

    # bigvgan.py imports only these symbols from utils.py. Avoid importing the
    # unrelated matplotlib/librosa plotting and dataset stack into the oracle.
    upstream_utils = types.ModuleType("utils")
    upstream_utils.get_padding = lambda kernel_size, dilation=1: int(
        (kernel_size * dilation - dilation) / 2
    )
    upstream_utils.init_weights = lambda _module, mean=0.0, std=0.01: None
    sys.modules["utils"] = upstream_utils

    from bigvgan import BigVGAN  # type: ignore[import-not-found]
    from env import AttrDict  # type: ignore[import-not-found]

    config = AttrDict(json.loads(args.config.read_text(encoding="utf-8")))
    generator = BigVGAN(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    generator.load_state_dict(checkpoint["generator"])
    generator.remove_weight_norm()
    generator.eval()

    mel = torch.tensor(
        [((index * 17) % 31 - 15) / 20.0 for index in range(config.num_mels)],
        dtype=torch.float32,
    ).reshape(1, config.num_mels, 1)
    with torch.inference_mode():
        waveform = generator(mel).reshape(-1)

    lines = [
        "input," + ",".join(f"{value:.9g}" for value in mel.reshape(-1).tolist()),
        "output,"
        + ",".join(f"{value:.9g}" for value in waveform.tolist()),
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
