#!/usr/bin/env python3
"""Extract the canonical Xiph RNNoise v0.2 arrays into safetensors.

The v0.2 release contains the trained network in ``src/rnnoise_data.c``;
there is no standalone weight asset.  This offline tool accepts either the
official release tarball or that C file, validates the complete array
manifest, and preserves every value required by the default (non-SU) C
forward.  Quantized int8 weights and sparse int indices are represented by
exactly-valued F32 tensors because Vokra's runtime GGUF dialect deliberately
does not add a private integer tensor type.  All int8/int32 values used here
are exactly representable as F32 and the runtime binder validates integrality
and range before converting them back.

Run from ``tools/parity`` so the repository's Python 3.12 environment is used::

    uv run python rnnoise_prepare_checkpoint.py \
      --input /path/to/rnnoise-0.2.tar.gz \
      --output /path/to/rnnoise-v0.2.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

RELEASE_SHA256 = "90fce4b00b9ff24c08dbfe31b82ffd43bae383d85c5535676d28b0a2b11c0d37"
UPSTREAM_URL = "https://github.com/xiph/rnnoise/releases/tag/v0.2"

FLOAT_ARRAYS = {
    "conv1_weights_float": 24_960,
    "conv1_bias": 128,
    "conv2_scale": 384,
    "conv2_bias": 384,
    "dense_out_weights_float": 12_288,
    "dense_out_bias": 32,
    "vad_dense_weights_float": 384,
    "vad_dense_bias": 1,
}
I8_ARRAYS = {"conv2_weights_int8": 147_456}
INDEX_ARRAYS: dict[str, int] = {}
for _layer in ("gru1", "gru2", "gru3"):
    for _part in ("input", "recurrent"):
        _prefix = f"{_layer}_{_part}"
        I8_ARRAYS[f"{_prefix}_weights_int8"] = 147_456
        INDEX_ARRAYS[f"{_prefix}_weights_idx"] = 4_752
        FLOAT_ARRAYS[f"{_prefix}_scale"] = 1_152
        FLOAT_ARRAYS[f"{_prefix}_bias"] = 1_152
        if _part == "recurrent":
            FLOAT_ARRAYS[f"{_prefix}_weights_diag"] = 1_152

EXPECTED = {**FLOAT_ARRAYS, **I8_ARRAYS, **INDEX_ARRAYS}
ARRAY_RE = re.compile(
    r"static\s+const\s+(float|opus_int8|int)\s+([A-Za-z0-9_]+)"
    r"\[(\d+)\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?"
)


def _read_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if tarfile.is_tarfile(path):
        if digest != RELEASE_SHA256:
            raise ValueError(
                f"release tarball sha256 {digest} does not match canonical v0.2 "
                f"sha256 {RELEASE_SHA256}"
            )
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
            matches = [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith("/src/rnnoise_data.c")
            ]
            if len(matches) != 1:
                raise ValueError(
                    "release tarball must contain exactly one */src/rnnoise_data.c"
                )
            extracted = archive.extractfile(matches[0])
            if extracted is None:
                raise ValueError("could not read rnnoise_data.c from release tarball")
            return extracted.read().decode("utf-8"), digest
    if path.name != "rnnoise_data.c":
        raise ValueError("--input must be the v0.2 tarball or rnnoise_data.c")
    return raw.decode("utf-8"), digest


def _parse_arrays(source: str) -> dict[str, np.ndarray]:
    parsed: dict[str, np.ndarray] = {}
    for c_type, name, declared_text, body in ARRAY_RE.findall(source):
        if name not in EXPECTED:
            continue
        declared = int(declared_text)
        if declared != EXPECTED[name]:
            raise ValueError(
                f"{name}: declared {declared} elements, expected {EXPECTED[name]}"
            )
        tokens = NUMBER_RE.findall(body)
        if len(tokens) != declared:
            raise ValueError(
                f"{name}: parsed {len(tokens)} values, expected {declared}"
            )
        values = np.asarray(
            [float(token.rstrip("fF")) for token in tokens], dtype=np.float32
        )
        if c_type == "opus_int8":
            if np.any(values != np.trunc(values)) or np.any(np.abs(values) > 127):
                raise ValueError(f"{name}: invalid int8 value")
        elif c_type == "int":
            if np.any(values != np.trunc(values)):
                raise ValueError(f"{name}: non-integral sparse index")
            if np.any(np.abs(values) > 16_777_216):
                raise ValueError(f"{name}: index is not exactly representable as F32")
        parsed[name] = values

    missing = sorted(set(EXPECTED) - set(parsed))
    if missing:
        raise ValueError(f"rnnoise_data.c missing required arrays: {', '.join(missing)}")
    if len(parsed) != len(EXPECTED):
        raise AssertionError("internal manifest accounting error")
    return parsed


def prepare(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise ValueError(f"--input is not a regular file: {input_path}")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing --output: {output_path}")
    source, source_sha256 = _read_source(input_path)
    tensors = _parse_arrays(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_path),
        metadata={
            "format": "vokra-rnnoise-v0.2-canonical-arrays-v1",
            "upstream_url": UPSTREAM_URL,
            "source_sha256": source_sha256,
            "release_tarball_sha256": RELEASE_SHA256,
            "license_spdx": "bsd-3-clause",
            "quantized_container": "signed-i8-and-i32-as-exact-f32",
            "prep_tool": "rnnoise_prepare_checkpoint.py",
            "prep_tool_version": "1.0.0",
        },
    )
    print(
        f"rnnoise_prepare_checkpoint: wrote {len(tensors)} canonical arrays to "
        f"{output_path} (sha256 {hashlib.sha256(output_path.read_bytes()).hexdigest()})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.input, args.output)
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"rnnoise_prepare_checkpoint: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
