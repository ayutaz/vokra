#!/usr/bin/env python3
"""Emits the exact SBV2 SDP noise buffer that
`crates/vokra-models/src/sbv2/duration.rs::SbV2SDP::sample`'s inner fill
loop would produce given a torch-parity RNG (Vokra's `TorchRandnStream`
under the PhiloxRNGEngine.h path).

The buffer shape is `[2, T]` in C-contiguous (channel-major) layout — 2
latent channels × T text-sequence timesteps — flattened to `2 * T` f32
words emitted little-endian. `SbV2SDP::sample` fills the exact same
buffer via `for v in &mut z { *v = rng.next_normal() * noise_scale_w;
}` with `noise_scale_w = 1.0`; setting the scale to 1.0 means the
noise buffer IS the raw RNG output, so this dumper reduces to K = 2*T
calls to the Python port of torch_randn_f32 (from
torch_philox_dump.py) — no SBV2-specific math involved.

Usage:
    uv run sbv2_sdp_noise_dump.py --seed 0 --T 50 \\
        --out ../../crates/vokra-models/tests/fixtures/sbv2/sdp_noise_seed0_T50.f32.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# Reuse the audited PhiloxRNGEngine.h port from the sibling script — no
# reimplementation, so a bug fix or SCALE change there propagates
# automatically without a second edit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_philox_dump import TorchPhiloxState, philox_randn_sample  # noqa: E402


def torch_randn_f32(seed: int, k: int) -> list[float]:
    """Same as `TorchRandnStream::new(seed); next_f32() × k` in Rust —
    one Philox block per f32 sample under the "one block per one
    sample" convention documented in the Rust module."""
    state = TorchPhiloxState(seed)
    return [philox_randn_sample(state.next_block()) for _ in range(k)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True, help="u64 seed passed to TorchRandnStream::new")
    p.add_argument("--T", dest="t", type=int, required=True, help="text sequence length")
    p.add_argument("--out", type=Path, required=True, help="output .bin path")
    args = p.parse_args()

    if args.t <= 0:
        p.error("--T must be positive")

    # `SbV2SDP::sample` fills `z = vec![0.0_f32; 2 * text_seq_len]` and
    # advances the RNG left-to-right; matching that here means we emit
    # 2*T f32 samples in the same order.
    k = 2 * args.t
    samples = torch_randn_f32(args.seed, k)

    payload = b"".join(struct.pack("<f", s) for s in samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"wrote 2*T = {k} f32 samples ({len(payload)} bytes) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
