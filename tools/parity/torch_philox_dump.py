#!/usr/bin/env python3
"""Pure-Python port of ATen's `PhiloxRNGEngine.h` block function + counter
increment, used to generate byte-exact reference fixtures for Vokra's Rust
`vokra_core::rng::philox4x32_10` and friends.

The port stays independent of PyTorch (no `import torch`) so it can (a)
regenerate fixtures on a torch-less host, and (b) cross-check the Rust
implementation against a second, algorithmically identical but source-
independent implementation. If both had the same bug the tests would
falsely pass, so this file's `--self-test` mode also verifies the block
function against Random123 v1.14's published KAT vectors before writing
any fixture — Random123 is an independent implementation of the same
algorithm (Salmon et al., SC'11), which forecloses that shared-bug hazard.

Usage:
    uv run torch_philox_dump.py --self-test
    uv run torch_philox_dump.py --seed 0 --n 8 \\
        --out crates/vokra-core/tests/fixtures/rng_torch/torch_philox_seed0_n8.u32.bin
    uv run torch_philox_dump.py --seed 0 --randn-samples 4 \\
        --out /tmp/ref.json

The `--out` file for `--n` mode is raw little-endian u32 words (4 bytes per
u32); for `--randn-samples` mode it is a JSON dict of `{blocks, samples}`.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import struct
import sys
from pathlib import Path

# We bind to libm's `logf` / `sqrtf` / `cosf` directly via ctypes because
# numpy's f32 transcendentals dispatch to Accelerate framework on Apple
# Silicon (and MKL / OpenLibm elsewhere) — those can differ from `logf` /
# `cosf` by up to 1 ULP, cascading to visible sample divergence from
# Rust's `f32::ln` / `f32::cos` (which link to libm directly).
_libm_path = ctypes.util.find_library("m")
if _libm_path is None:
    raise RuntimeError(
        "torch_philox_dump: could not locate libm — the byte-exact torch parity "
        "fixtures require single-precision transcendentals via ctypes, not "
        "numpy (see the top-of-file rationale)."
    )
_libm = ctypes.CDLL(_libm_path)
_libm.logf.argtypes = [ctypes.c_float]
_libm.logf.restype = ctypes.c_float
_libm.sqrtf.argtypes = [ctypes.c_float]
_libm.sqrtf.restype = ctypes.c_float
_libm.cosf.argtypes = [ctypes.c_float]
_libm.cosf.restype = ctypes.c_float

# ---------- Philox4x32-10 constants (Random123 v1.14 philox.h) ----------------

M0 = 0xD251_1F53
M1 = 0xCD9E_8D57
W0 = 0x9E37_79B9
W1 = 0xBB67_AE85
ROUNDS = 10


def _mulhilo32(a: int, b: int) -> tuple[int, int]:
    """Full-width 32×32 → 64 multiply, returning (lo, hi) as two u32s."""
    product = (a & 0xFFFF_FFFF) * (b & 0xFFFF_FFFF)
    return product & 0xFFFF_FFFF, (product >> 32) & 0xFFFF_FFFF


def _single_round(ctr: list[int], key: list[int]) -> list[int]:
    lo0, hi0 = _mulhilo32(M0, ctr[0])
    lo1, hi1 = _mulhilo32(M1, ctr[2])
    return [
        (hi1 ^ ctr[1] ^ key[0]) & 0xFFFF_FFFF,
        lo1,
        (hi0 ^ ctr[3] ^ key[1]) & 0xFFFF_FFFF,
        lo0,
    ]


def philox4x32_10(ctr: list[int], key: list[int]) -> list[int]:
    """Ten rounds: nine key-bumping iterations, then a tenth without a bump."""
    c = list(ctr)
    k = list(key)
    for _ in range(ROUNDS - 1):
        c = _single_round(c, k)
        k[0] = (k[0] + W0) & 0xFFFF_FFFF
        k[1] = (k[1] + W1) & 0xFFFF_FFFF
    return _single_round(c, k)


# ---------- TorchPhiloxState (seed init + counter advance) --------------------


class TorchPhiloxState:
    """Mirrors `PhiloxRNGEngine.h`'s (seed, subsequence=0, offset) constructor:
        key      = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF]
        counter  = [offset_lo, offset_hi, 0, 0]
    Each next_block advances offset by 1 (matches torch's incr())."""

    __slots__ = ("seed", "offset")

    def __init__(self, seed: int, offset: int = 0) -> None:
        self.seed = seed & 0xFFFF_FFFF_FFFF_FFFF
        self.offset = offset & 0xFFFF_FFFF_FFFF_FFFF

    def next_block(self) -> list[int]:
        key = [self.seed & 0xFFFF_FFFF, (self.seed >> 32) & 0xFFFF_FFFF]
        ctr = [self.offset & 0xFFFF_FFFF, (self.offset >> 32) & 0xFFFF_FFFF, 0, 0]
        out = philox4x32_10(ctr, key)
        self.offset = (self.offset + 1) & 0xFFFF_FFFF_FFFF_FFFF
        return out


# ---------- Box-Muller (mirrors PhiloxRNGEngine.h::randn) ---------------------

SCALE = struct.unpack("<f", struct.pack("<I", 0x2FFF_FFFF))[0]
# = 4.6566127342e-10 rounded to f32 = 0x2FFF_FFFF, verified in Rust
# scratchpad/probe_scale.rs.


def _to_f32(x: float) -> float:
    """Round a Python float (f64) to f32 and back, matching what C++ does at
    every constexpr float or `float x = …` assignment."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# f32-precision math primitives via libm (bound above via ctypes).
#
# Rust's `f32::ln` / `f32::sqrt` / `f32::cos` dispatch to libm's `logf` /
# `sqrtf` / `cosf` — SINGLE-precision throughout. We MUST use the same
# libm routines to get byte-exact parity: numpy's f32 math and Python's
# f64-then-round-to-f32 both differ from `logf` / `cosf` by up to 1 ULP
# for a non-trivial fraction of inputs, which cascades to visible sample
# divergence (empirically first hits at seed=12345 sample 4 with numpy).
_F32_PI = _to_f32(3.141592653589793)


def _logf(x: float) -> float:
    return _libm.logf(x)


def _sqrtf(x: float) -> float:
    return _libm.sqrtf(x)


def _cosf(x: float) -> float:
    return _libm.cosf(x)


def uint32_to_uniform_f32(v: int) -> float:
    """Bit-exact torch: mask to 31 bits, cast to f32, multiply by SCALE."""
    masked = (v & 0x7FFF_FFFF) & 0xFFFF_FFFF
    # Casting `int` to Python float goes through f64; then multiplying by an
    # f32-rounded SCALE and rounding the product back to f32 mirrors what a
    # C++ `float`-typed expression does. The intermediate `_to_f32(masked)`
    # is redundant for u32s <= 2**24 (the f32 mantissa fits them exactly),
    # but for larger u32s it enforces the same round-to-nearest-even the C++
    # compiler does when converting a `uint32_t` to `float`.
    masked_f = _to_f32(float(masked))
    return _to_f32(masked_f * SCALE)


def philox_randn_sample(block: list[int]) -> float:
    """One 128-bit Philox block → one f32 normal sample. The `[2]` and `[3]`
    words are IMPLICITLY DISCARDED — this is Vokra's "one block per one
    sample" convention (see `crates/vokra-core/src/rng/normal_kernel.rs`
    module doc for why we differ from torch's pipelined stream).

    All math primitives dispatch through numpy.float32 so `logf` / `sqrtf`
    / `cosf` (same libm routines Rust's `f32` methods call) run at f32
    precision throughout — a naïve `math.log(u1)` computes at f64 then
    rounds to f32, which differs from `logf(u1)` by up to 1 ULP for ~10%
    of inputs and cascades to visible sample divergence."""
    u1 = _to_f32(1.0 - uint32_to_uniform_f32(block[0]))
    u2 = _to_f32(1.0 - uint32_to_uniform_f32(block[1]))
    # r = sqrt(-2 * ln(u1)); theta = 2 * pi * u2
    r = _to_f32(_sqrtf(_to_f32(-2.0 * _logf(u1))))
    theta = _to_f32(2.0 * _F32_PI * u2)
    return _to_f32(r * _cosf(theta))


def torch_randn_f32(seed: int, k: int) -> list[float]:
    state = TorchPhiloxState(seed)
    return [philox_randn_sample(state.next_block()) for _ in range(k)]


# ---------- Self-test (KATs) --------------------------------------------------

RANDOM123_KATS: list[tuple[list[int], list[int], list[int]]] = [
    # (ctr, key, expected)  — from Random123 v1.14
    ([0, 0, 0, 0], [0, 0], [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]),
    (
        [0xFFFF_FFFF] * 4,
        [0xFFFF_FFFF, 0xFFFF_FFFF],
        [0x408F_276D, 0x41C8_3B0E, 0xA20B_C7C6, 0x6D54_51FD],
    ),
    (
        [0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E, 0x0370_7344],
        [0xA409_3822, 0x299F_31D0],
        [0xD16C_FE09, 0x94FD_CCEB, 0x5001_E420, 0x2412_6EA1],
    ),
]


def run_self_test() -> int:
    ok = True
    for i, (ctr, key, expected) in enumerate(RANDOM123_KATS, 1):
        got = philox4x32_10(ctr, key)
        if got != expected:
            print(f"KAT #{i} FAIL: got {got}, expected {expected}", file=sys.stderr)
            ok = False
    # Torch seed=0 first block equals KAT #1.
    state = TorchPhiloxState(0)
    first = state.next_block()
    if first != RANDOM123_KATS[0][2]:
        print(
            f"TorchPhiloxState(0).next_block() != KAT #1: got {first}", file=sys.stderr
        )
        ok = False
    # Uniform boundary.
    if uint32_to_uniform_f32(0) != 0.0:
        print("uint32_to_uniform_f32(0) != 0.0", file=sys.stderr)
        ok = False
    if uint32_to_uniform_f32(0xFFFF_FFFF) >= 1.0:
        print("uint32_to_uniform_f32(u32::MAX) >= 1.0", file=sys.stderr)
        ok = False
    # SCALE bit pattern.
    if struct.pack("<f", SCALE) != struct.pack("<I", 0x2FFF_FFFF):
        print(f"SCALE bits = {struct.unpack('<I', struct.pack('<f', SCALE))[0]:#010x}, expected 0x2FFFFFFF", file=sys.stderr)
        ok = False
    if ok:
        print("torch_philox_dump.py: all self-tests passed")
    return 0 if ok else 1


# ---------- CLI ---------------------------------------------------------------


def _write_u32_words(path: Path, words: list[int]) -> None:
    payload = b"".join(struct.pack("<I", w & 0xFFFF_FFFF) for w in words)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_f32_samples(path: Path, samples: list[float]) -> None:
    payload = b"".join(struct.pack("<f", s) for s in samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="run built-in KATs and exit")
    p.add_argument("--seed", type=int, default=0, help="u64 seed (default 0)")
    p.add_argument("--offset", type=int, default=0, help="initial block offset")
    p.add_argument("--n", type=int, default=0, help="number of raw u32 words to emit")
    p.add_argument(
        "--randn-samples", type=int, default=0, help="number of Box-Muller f32 samples to emit"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "output file. If --n > 0: raw LE u32 bytes. "
            "If --randn-samples > 0 and out ends in .json: JSON debug bundle. "
            "If --randn-samples > 0 and out ends in .bin: raw LE f32 bytes."
        ),
    )
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if args.n <= 0 and args.randn_samples <= 0:
        p.error("pass --n <N> or --randn-samples <K> (or --self-test)")
    if args.out is None:
        p.error("--out required unless --self-test")

    if args.n > 0:
        state = TorchPhiloxState(args.seed, args.offset)
        words: list[int] = []
        blocks_needed = (args.n + 3) // 4
        for _ in range(blocks_needed):
            words.extend(state.next_block())
        _write_u32_words(args.out, words[: args.n])
        print(f"wrote {args.n} u32 words to {args.out}")

    if args.randn_samples > 0:
        state = TorchPhiloxState(args.seed, args.offset)
        blocks: list[list[int]] = []
        samples: list[float] = []
        for _ in range(args.randn_samples):
            blk = state.next_block()
            blocks.append(blk)
            samples.append(philox_randn_sample(blk))
        if args.out.suffix == ".json":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "offset": args.offset,
                        "scale_bits": 0x2FFF_FFFF,
                        "blocks": blocks,
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
        print(f"wrote {args.randn_samples} randn samples to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
