#!/usr/bin/env python3
"""Generate the committed DFN3 primitive parity fixtures (M4-20 T17).

Numerical-parity discipline: the reference implementations are the REAL
upstream ``df.modules`` classes (``Conv2dNormAct`` /
``ConvTranspose2dNormAct`` / ``GroupedLinearEinsum`` / ``SqueezedGRU_S``) —
never a re-derivation of the Rust port. Each case instantiates the upstream
module with SplitMix-free torch-seeded parameters, runs a seeded input
through it in eval mode, and dumps weights + input + output as raw
little-endian f32 files under ``tests/parity/dfn3/``.

The Rust side (``crates/vokra-ops/src/denoise.rs`` primitive parity tests)
reconstructs each case from the dumped weights and asserts the forward
output against the dumped reference.

Cases (all shapes chosen to exercise the exact structural variants the DFN3
graph uses):

* ``conv_full33``   — Conv2dNormAct(1→4, (3,3), separable) → plain conv +
                      BN + ReLU (groups = gcd(1,4) = 1: erb_conv0 shape).
* ``conv_sep13``    — Conv2dNormAct(4→4, (1,3), fstride=2, separable) →
                      depthwise + pointwise + BN + ReLU (erb_conv1 shape).
* ``conv_grp2``     — Conv2dNormAct(2→4, (3,3), separable) → groups=2 +
                      pointwise + BN + ReLU (df_conv0 shape).
* ``conv_path``     — Conv2dNormAct(4→4, kernel=1, separable) → depthwise
                      1×1 + BN + ReLU (conv3p shape: groups survive the
                      max(kernel)==1 downgrade).
* ``conv_kt51``     — Conv2dNormAct(4→2, (5,1), separable) → groups=2 +
                      pointwise + BN + ReLU (df_convp shape).
* ``convt_sep13``   — ConvTranspose2dNormAct(4→4, (1,3), fstride=2,
                      separable) → depthwise transposed + pointwise + BN +
                      ReLU (convt2 shape).
* ``glin``          — GroupedLinearEinsum(6→4, groups=2).
* ``sgru``          — SqueezedGRU_S(6, 8, output_size=6, 2 layers,
                      linear_groups=2, ReLU acts) (erb_dec.emb_gru shape).

# Usage

::

    ~/.cache/vokra-eval/venv-dfn3/bin/python tools/parity/dfn3_primitives_fixture.py \\
        --out tests/parity/dfn3
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
import torch
from torch import nn

from df.modules import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    SqueezedGRU_S,
)


def dump(out_dir: str, name: str, t: torch.Tensor) -> None:
    a = t.detach().cpu().numpy().astype("<f4")
    np.ascontiguousarray(a).tofile(os.path.join(out_dir, f"{name}.f32"))


def seed_module(m: nn.Module, gen: torch.Generator) -> None:
    """Reproducible non-degenerate parameters/buffers (incl. BN stats)."""
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=gen))
        for name, b in m.named_buffers():
            if name.endswith("running_var"):
                b.copy_(torch.empty_like(b).uniform_(0.5, 1.5, generator=gen))
            elif name.endswith("running_mean"):
                b.copy_(torch.empty_like(b).uniform_(-0.5, 0.5, generator=gen))


def dump_state(out_dir: str, case: str, m: nn.Module) -> None:
    for name, t in m.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        dump(out_dir, f"{case}.{name}", t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen = torch.Generator().manual_seed(20260717)

    t_len, f_erb = 6, 8

    def run_conv(case: str, m: nn.Module, in_ch: int, f_in: int) -> None:
        seed_module(m, gen)
        m.eval()
        x = torch.empty(1, in_ch, t_len, f_in).uniform_(-1.0, 1.0, generator=gen)
        with torch.no_grad():
            y = m(x)
        dump_state(args.out, case, m)
        dump(args.out, f"{case}.x", x.squeeze(0))
        dump(args.out, f"{case}.y", y.squeeze(0))
        print(case, tuple(x.shape), "->", tuple(y.shape))

    run_conv(
        "conv_full33",
        Conv2dNormAct(1, 4, kernel_size=(3, 3), bias=False, separable=True),
        1,
        f_erb,
    )
    run_conv(
        "conv_sep13",
        Conv2dNormAct(4, 4, kernel_size=(1, 3), fstride=2, bias=False, separable=True),
        4,
        f_erb,
    )
    run_conv(
        "conv_grp2",
        Conv2dNormAct(2, 4, kernel_size=(3, 3), bias=False, separable=True),
        2,
        f_erb,
    )
    run_conv(
        "conv_path",
        Conv2dNormAct(4, 4, kernel_size=1, bias=False, separable=True),
        4,
        f_erb,
    )
    run_conv(
        "conv_kt51",
        Conv2dNormAct(4, 2, kernel_size=(5, 1), bias=False, separable=True),
        4,
        f_erb,
    )
    run_conv(
        "convt_sep13",
        ConvTranspose2dNormAct(4, 4, kernel_size=(1, 3), fstride=2, bias=False, separable=True),
        4,
        f_erb,
    )

    # GroupedLinearEinsum.
    gl = GroupedLinearEinsum(6, 4, groups=2)
    seed_module(gl, gen)
    x = torch.empty(1, t_len, 6).uniform_(-1.0, 1.0, generator=gen)
    with torch.no_grad():
        y = gl(x)
    dump_state(args.out, "glin", gl)
    dump(args.out, "glin.x", x.squeeze(0))
    dump(args.out, "glin.y", y.squeeze(0))
    print("glin", tuple(x.shape), "->", tuple(y.shape))

    # SqueezedGRU_S (2-layer, grouped in/out, ReLU acts — no skip, like every
    # DFN3 instance).
    sg = SqueezedGRU_S(
        6,
        8,
        output_size=6,
        num_layers=2,
        linear_groups=2,
        batch_first=True,
        gru_skip_op=None,
        linear_act_layer=partial(nn.ReLU, inplace=True),
    )
    seed_module(sg, gen)
    sg.eval()
    x = torch.empty(1, t_len, 6).uniform_(-1.0, 1.0, generator=gen)
    with torch.no_grad():
        y, _h = sg(x)
    dump_state(args.out, "sgru", sg)
    dump(args.out, "sgru.x", x.squeeze(0))
    dump(args.out, "sgru.y", y.squeeze(0))
    print("sgru", tuple(x.shape), "->", tuple(y.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
