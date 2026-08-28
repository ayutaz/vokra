"""Generate the tiny independent CoreML execution fixture.

The fixture is intentionally not a mirror of Rust code: it is the elementary
function ``encoder_hidden = log_mel + 1`` built through Apple's MIL builder.
It exercises `.mlpackage` generation, `coremlcompiler`, raw Objective-C model
load, feature binding, prediction, strides, and F32 readback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types


def build(output: Path) -> None:
    if output.suffix != ".mlpackage":
        raise ValueError(f"output must end in .mlpackage: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2, 3), dtype=types.fp32)],
        opset_version=ct.target.macOS14,
    )
    def tiny_encoder(log_mel):
        one = np.ones((1, 2, 3), dtype=np.float32)
        return mb.add(x=log_mel, y=one, name="encoder_hidden")

    model = ct.convert(
        tiny_encoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )
    model.author = "Vokra test fixture"
    model.license = "Apache-2.0 (fixture code and generated constants)"
    model.short_description = "Independent add-one fixture for CoreML raw-FFI tests"
    model.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
