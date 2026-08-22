#!/usr/bin/env python3
"""Generate independent PyTorch fixtures for ``vokra_ops::flow_sample``.

The reference deliberately uses PyTorch tensors and direct integration
equations.  It does not call, bind, or translate the Rust implementation.
Two cases cover the runtime contracts that previously ended in a fixture-
presence panic:

* seven-step linear Euler without CFG;
* eight-step F5 sway Heun with dual-forward CFG.

Run from ``tools/parity`` with ``uv run flow_sampler_dump_reference.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/vokra-ops/tests/fixtures/flow_sampler"


def field(x: torch.Tensor, t: torch.Tensor, *, conditioned: bool) -> torch.Tensor:
    """Small deterministic velocity field with x and time dependence."""

    gain = torch.tensor(-0.35 if conditioned else -0.55, dtype=torch.float32)
    bias = torch.tensor(0.12 if conditioned else -0.08, dtype=torch.float32)
    return gain * x + bias * t


def euler_reference(initial: torch.Tensor) -> torch.Tensor:
    x = initial.clone()
    timesteps = torch.linspace(0.0, 1.0, 8, dtype=torch.float32)
    for left, right in zip(timesteps[:-1], timesteps[1:], strict=True):
        x = x + (right - left) * field(x, left, conditioned=False)
    return x


def sway_timesteps(nfe: int) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, nfe + 1, dtype=torch.float32)
    # F5-TTS sway schedule, s=-1.0: t + s*(cos(pi*t/2)-1+t).
    return base - (torch.cos(math.pi * base / 2.0) - 1.0 + base)


def guided_field(x: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    uncond = field(x, t, conditioned=False)
    cond = field(x, t, conditioned=True)
    return uncond + scale * (cond - uncond)


def heun_cfg_reference(initial: torch.Tensor) -> torch.Tensor:
    x = initial.clone()
    timesteps = sway_timesteps(8)
    scale = torch.tensor(1.75, dtype=torch.float32)
    for left, right in zip(timesteps[:-1], timesteps[1:], strict=True):
        dt = right - left
        k1 = guided_field(x, left, scale)
        predicted = x + dt * k1
        k2 = guided_field(predicted, right, scale)
        x = x + dt * torch.tensor(0.5, dtype=torch.float32) * (k1 + k2)
    return x


def write_f32(path: Path, values: torch.Tensor) -> None:
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    path.write_bytes(array.tobytes())


def main() -> None:
    torch.set_grad_enabled(False)
    initial = torch.tensor([0.75, -1.25, 0.125, 2.0, -0.5], dtype=torch.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    write_f32(OUT / "initial.f32.bin", initial)
    write_f32(OUT / "euler_linear_nfe7.f32.bin", euler_reference(initial))
    write_f32(OUT / "heun_sway_dual_cfg_nfe8.f32.bin", heun_cfg_reference(initial))

    # Fail if a host NumPy default ever changes the byte contract.
    for path in OUT.glob("*.f32.bin"):
        values = np.fromfile(path, dtype="<f4")
        if values.shape != (5,) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid generated fixture {path}: {values!r}")


if __name__ == "__main__":
    main()
