#!/usr/bin/env python3
"""Byte-exact `torch.manual_seed(N); torch.randn(K, device='cpu')` fixture
generator for `K < 16` (or non-contiguous tensors), implemented as a
pure-Python port of `at::mt19937_engine` +
`at::normal_distribution<double>` (torch source, BSD-3-Clause) so it can
regenerate fixtures on a torch-less host and stays independent of the
Rust `vokra_core::rng::TorchRandnStream` under test (a shared-bug
hazard between the Rust port and its own Python mirror would let both
sides drift together — see
`crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs` for the load-
bearing byte anchor against real `torch.randn` at seed=0).

# 2026-08-08 origin — replaces torch_philox_dump.py's --randn-samples

Prior to today, `torch_philox_dump.py --randn-samples` was believed to
reproduce `torch.randn(device='cpu')`. A byte-level bisect found NO
match at any sample: CPU torch uses MT19937, not Philox — see this
file's Rust counterpart doc for the full backstory. This script is the
honest replacement; `torch_philox_dump.py` is kept as a Philox
primitive (the block function + Random123 KATs still pass on their own
merits, useful for a future CUDA parity path).

# Scope caveat

`torch.randn(K)` on CPU for `K >= 16` and contiguous dispatches to a
SIMD `normal_fill` fast path (ATen/native/cpu/DistributionTemplates.h:
168-220) with a *different* Box-Muller formula and NO pair caching.
This script matches the small-K / non-contiguous path only; SBV2's SDP
noise buffer stays under 16 elements per timestep in practice.

# Usage
    uv run torch_randn_cpu_dump.py --self-test
    uv run torch_randn_cpu_dump.py --seed 0 --randn-samples 4 \\
        --out crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed0_k4.f32.bin

The `--out` file for `--randn-samples` mode is raw little-endian f32
words (4 bytes per f32); for `--json-debug` mode it is a JSON dict
of `{seed, samples, sample_bits, tempered_u32s}` for diff-friendly
debugging.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

# ---------- MT19937 (bit-exact port of ATen/core/MT19937RNGEngine.h) ----------

_N = 624
_MERSENNE_M = 397
_MATRIX_A = 0x9908_B0DF
_UMASK = 0x8000_0000
_LMASK = 0x7FFF_FFFF
_INIT_MULT = 1_812_433_253


def mt19937_seed(seed: int) -> list[int]:
    """Matsumoto & Nishimura 1998 init recurrence — MT19937RNGEngine.h:156-165.

    Only the low 32 bits of `seed` participate (torch's own convention —
    MT19937 is a 32-bit generator, and CPUGeneratorImpl.h truncates its
    u64 seed at this seam). `torch.manual_seed(0x1_0000_0000)` produces
    the same stream as `torch.manual_seed(0)`.
    """
    state = [seed & 0xFFFF_FFFF]
    for j in range(1, _N):
        prev = state[-1]
        mixed = prev ^ (prev >> 30)
        new = (_INIT_MULT * mixed + j) & 0xFFFF_FFFF
        state.append(new)
    return state


def mt19937_twist(state: list[int]) -> list[int]:
    """The MT19937 twist step — MT19937RNGEngine.h:175-189."""
    for i in range(_N - _MERSENNE_M):
        y = (state[i] & _UMASK) | (state[i + 1] & _LMASK)
        state[i] = state[i + _MERSENNE_M] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
    for i in range(_N - _MERSENNE_M, _N - 1):
        y = (state[i] & _UMASK) | (state[i + 1] & _LMASK)
        state[i] = state[i + (_MERSENNE_M - _N)] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
    y = (state[_N - 1] & _UMASK) | (state[0] & _LMASK)
    state[_N - 1] = state[_MERSENNE_M - 1] ^ (y >> 1) ^ ((y & 1) * _MATRIX_A)
    return state


class TorchMt19937:
    """Bit-exact port of `at::mt19937_engine` — state array + `left` counter
    + tempering exactly per MT19937RNGEngine.h."""

    __slots__ = ("state", "left", "next")

    def __init__(self, seed: int) -> None:
        self.state = mt19937_seed(seed)
        self.left = 1
        self.next = 0

    def next_u32(self) -> int:
        self.left -= 1
        if self.left <= 0:
            self.state = mt19937_twist(self.state)
            self.left = _N
            self.next = 0
        y = self.state[self.next]
        self.next += 1
        # Tempering — MT19937 paper §4 + torch header.
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        return y & 0xFFFF_FFFF

    def random64(self) -> int:
        """hi-first packing, matching ATen's random64() helper."""
        hi = self.next_u32()
        lo = self.next_u32()
        return (hi << 32) | lo


# ---------- normal_distribution<double> (DistributionsHelper.h:187-198) --------

_MASK_53 = (1 << 53) - 1
_DIVISOR_53 = float(1 << 53)


def _uniform_real_f64(v: int) -> float:
    """`uniform_real<double>` from TransformationHelper.h:84-90: mask to
    53 bits, cast to f64, divide by 2^53. The result is a pure f64 arithmetic
    (mask value fits exactly in the mantissa)."""
    return (v & _MASK_53) / _DIVISOR_53


class TorchRandn:
    """`at::normal_distribution<double>::operator()` +
    `static_cast<float>(result)`. Every two Box-Muller evaluations
    produce a (cos, sin) pair and the sine is cached."""

    __slots__ = ("mt", "cached_sin")

    def __init__(self, seed: int) -> None:
        self.mt = TorchMt19937(seed)
        self.cached_sin: float | None = None

    def next_f32(self) -> float:
        if self.cached_sin is not None:
            v = self.cached_sin
            self.cached_sin = None
            return struct.unpack("<f", struct.pack("<f", v))[0]
        # Order matters: u1 first, u2 second — u1 → theta, u2 → r.
        u1 = _uniform_real_f64(self.mt.random64())
        u2 = _uniform_real_f64(self.mt.random64())
        # log1p(-u2) = ln(1 - u2), numerically stable near u2 = 0.
        r = math.sqrt(-2.0 * math.log1p(-u2))
        theta = 2.0 * math.pi * u1
        # Cache the sine in f64 so the deferred f32 cast rounds once
        # (matching torch's `static_cast<float>` at return time).
        self.cached_sin = r * math.sin(theta)
        return struct.unpack("<f", struct.pack("<f", r * math.cos(theta)))[0]


# ---------- Self-test (anchors against real torch) ----------------------------


def run_self_test() -> int:
    """Anchor: torch.manual_seed(0); torch.randn(4) → these exact bit
    patterns on CPU (values from the bisect report wf_20fa0933-53d)."""
    ok = True

    # Anchor 1: first 8 raw MT19937 u32s at seed=0.
    mt = TorchMt19937(0)
    expected_mt = [
        0x8C7F_0AAC,
        0x97C4_AA2F,
        0xB716_A675,
        0xD821_CCC0,
        0x9A4E_B343,
        0xDBA2_52FB,
        0x8B7D_76C3,
        0xD8E5_7D67,
    ]
    for i, want in enumerate(expected_mt):
        got = mt.next_u32()
        if got != want:
            print(
                f"MT19937(0) u32 #{i}: got {got:#010x}, want {want:#010x}",
                file=sys.stderr,
            )
            ok = False

    # Anchor 2: first 4 f32 samples of TorchRandn(0).
    rng = TorchRandn(0)
    got_samples = [rng.next_f32() for _ in range(4)]
    got_bits = [struct.unpack("<I", struct.pack("<f", s))[0] for s in got_samples]
    expected_bits = [0x3FC5_3F5C, 0xBE96_3C50, 0xC00B_7149, 0x3F11_84B6]
    if got_bits != expected_bits:
        print(
            f"torch.randn(seed=0, k=4) bits: got {[hex(b) for b in got_bits]}, "
            f"want {[hex(b) for b in expected_bits]}",
            file=sys.stderr,
        )
        ok = False

    if ok:
        print("torch_randn_cpu_dump.py: all self-tests passed")
    return 0 if ok else 1


# ---------- CLI ---------------------------------------------------------------


def _write_f32_samples(path: Path, samples: list[float]) -> None:
    payload = b"".join(struct.pack("<f", s) for s in samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="run built-in anchors and exit")
    p.add_argument("--seed", type=int, default=0, help="u64 seed (default 0)")
    p.add_argument(
        "--randn-samples",
        type=int,
        default=0,
        help="number of f32 samples to emit",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output file. If ends in .json: JSON debug bundle "
            "{seed, samples, sample_bits}. Otherwise raw LE f32 bytes."
        ),
    )
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if args.randn_samples <= 0:
        p.error("pass --randn-samples <K> (or --self-test)")
    if args.out is None:
        p.error("--out required unless --self-test")

    rng = TorchRandn(args.seed)
    samples = [rng.next_f32() for _ in range(args.randn_samples)]
    if args.out.suffix == ".json":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "samples": samples,
                    "sample_bits": [
                        struct.unpack("<I", struct.pack("<f", s))[0] for s in samples
                    ],
                },
                indent=2,
            )
        )
    else:
        _write_f32_samples(args.out, samples)
    print(f"wrote {args.randn_samples} torch.randn samples (seed={args.seed}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
