#!/usr/bin/env python3
"""Dump SpeexDSP echo-canceller reference fixtures for M4-03 `aec` parity.

Offline, developer-side tool (NFR-DS-02: the Vokra runtime never gains a C
build step or Python dependency from this — CI consumes the *committed*
fixtures under ``tests/parity/aec/`` and needs neither this script nor a C
toolchain, the same operating model as ``mimi_dump.py`` / M3-06).

What it does
------------

1. Downloads the SpeexDSP sources (the float-build echo canceller only) at
   the pinned upstream commit — nothing is vendored into the repo; the BSD
   attribution for the *Rust port* lives in ``NOTICE`` §7 and
   ``crates/vokra-ops/THIRD_PARTY_LICENSES/speexdsp-LICENSE.txt``.
2. Builds a ~90-line dump harness with the system C compiler using
   ``-DFLOATING_POINT -DUSE_SMALLFT`` and — load-bearing —
   ``-ffp-contract=off``: the Rust port performs plain (non-FMA) f32
   arithmetic, so FMA contraction on the C side would introduce a spurious
   rounding divergence (ADR M4-03 §D-(g)).
3. Generates deterministic signals (SplitMix64, integer math — bit-stable
   across Python versions and platforms):

   - **single-talk** (``st_*``): far-end = uniform int16 noise in
     [-8000, 8000]; near-end = far ⊛ fixed FIR echo path (f64 conv, rounded
     to int16). Pure echo — the convergence / ERLE fixture.
   - **double-talk** (``dt_*``): same far-end process; near-end = echo +
     an independent near-end "speech" (gated two-tone bursts) that starts
     after a convergence preamble. The near-end-preservation fixture.

4. Runs the upstream canceller frame-by-frame
   (``speex_echo_state_init`` + ``SPEEX_ECHO_SET_SAMPLING_RATE`` +
   ``speex_echo_cancellation``) and writes its int16 output.
5. Writes ``manifest.txt`` with every parameter and the sha256 of every
   fixture file. Determinism contract: same script + same pinned commit +
   same compiler ⇒ same sha256 for the *input* files always; the upstream
   *output* is additionally libm/compiler-dependent, so the manifest records
   the generating toolchain (the committed fixture is the reference — the
   Rust parity test calibrates its tolerances against it, see
   ``crates/vokra-ops/tests/parity_aec.rs``).

Usage
-----

    python3 tools/parity/aec_dump.py            # writes tests/parity/aec/
    python3 tools/parity/aec_dump.py --outdir X # elsewhere

Requires: python3 (stdlib only), a C compiler (`cc`), network access to
raw.githubusercontent.com (or pre-seed --srcdir with the pinned sources).
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Pinned upstream: xiph/speexdsp master as of 2025-07-05 (the commit the
# Rust port in crates/vokra-ops/src/aec.rs cites — keep the two in lock-step).
SPEEXDSP_COMMIT = "7a158783df74efe7c2d1c6ee8363c1e695c71226"
RAW_BASE = f"https://raw.githubusercontent.com/xiph/speexdsp/{SPEEXDSP_COMMIT}"
SOURCES = [
    "libspeexdsp/mdf.c",
    "libspeexdsp/fftwrap.c",
    "libspeexdsp/fftwrap.h",
    "libspeexdsp/smallft.c",
    "libspeexdsp/smallft.h",
    "libspeexdsp/arch.h",
    "libspeexdsp/os_support.h",
    "libspeexdsp/pseudofloat.h",
    "libspeexdsp/math_approx.h",
    "include/speex/speex_echo.h",
    "include/speex/speexdsp_types.h",
]

# Fixture configuration (single source of truth — mirrored into manifest.txt
# and read back by parity_aec.rs).
SAMPLE_RATE = 16_000
FRAME_SIZE = 256          # 16 ms @ 16 kHz (speex_echo.h: 10-20 ms guidance)
FILTER_LENGTH = 2048      # 128 ms tail (speex_echo.h: 100-500 ms guidance)
ST_FRAMES = 250           # 4.0 s single-talk
DT_FRAMES = 150           # 2.4 s double-talk (preamble + bursts)
DT_SPEECH_START_FRAME = 100
FAR_AMP = 8000
SEED_FAR_ST = 0x5EED_AEC0_0001
SEED_FAR_DT = 0x5EED_AEC0_0002
# Echo path: sparse decaying FIR, support < FILTER_LENGTH (taps in the
# unit-gain domain; index = delay in samples).
ECHO_TAPS = [(2, 0.5), (11, -0.3), (25, 0.15), (300, 0.08), (700, -0.04), (1500, 0.02)]

HARNESS_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "speex/speex_echo.h"

/* aec_dump harness: raw int16 LE far/near in, canceller int16 LE out. */
int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s frame_size filter_length rate far.i16le near.i16le out.i16le\n",
                argv[0]);
        return 2;
    }
    int frame = atoi(argv[1]);
    int tail = atoi(argv[2]);
    int rate = atoi(argv[3]);
    FILE *ff = fopen(argv[4], "rb");
    FILE *nf = fopen(argv[5], "rb");
    FILE *of = fopen(argv[6], "wb");
    if (!ff || !nf || !of) { fprintf(stderr, "open failed\n"); return 2; }

    SpeexEchoState *st = speex_echo_state_init(frame, tail);
    speex_echo_ctl(st, SPEEX_ECHO_SET_SAMPLING_RATE, &rate);

    spx_int16_t *far = malloc(sizeof(spx_int16_t) * frame);
    spx_int16_t *near = malloc(sizeof(spx_int16_t) * frame);
    spx_int16_t *out = malloc(sizeof(spx_int16_t) * frame);
    for (;;) {
        size_t nfar = fread(far, sizeof(spx_int16_t), frame, ff);
        size_t nnear = fread(near, sizeof(spx_int16_t), frame, nf);
        if (nfar < (size_t)frame || nnear < (size_t)frame) break;
        speex_echo_cancellation(st, near, far, out);
        fwrite(out, sizeof(spx_int16_t), frame, of);
    }
    speex_echo_state_destroy(st);
    free(far); free(near); free(out);
    fclose(ff); fclose(nf); fclose(of);
    return 0;
}
"""

CONFIG_TYPES_H = """\
#ifndef SPEEXDSP_CONFIG_TYPES_H
#define SPEEXDSP_CONFIG_TYPES_H
#include <stdint.h>
typedef int16_t spx_int16_t;
typedef uint16_t spx_uint16_t;
typedef int32_t spx_int32_t;
typedef uint32_t spx_uint32_t;
#endif
"""


class SplitMix64:
    """The same SplitMix64 stream as vokra_core::SplitMix64 (integer math,
    bit-stable everywhere)."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)


def gen_far(num_samples: int, seed: int) -> list[int]:
    """Uniform int16 noise in [-FAR_AMP, FAR_AMP] (pure integer math)."""
    rng = SplitMix64(seed)
    span = 2 * FAR_AMP + 1
    return [int(rng.next_u64() % span) - FAR_AMP for _ in range(num_samples)]


def convolve_echo(far: list[int]) -> list[float]:
    """f64 sparse FIR convolution (the synthetic room)."""
    out = [0.0] * len(far)
    for i in range(len(far)):
        acc = 0.0
        for delay, coef in ECHO_TAPS:
            if i >= delay:
                acc += coef * far[i - delay]
        out[i] = acc
    return out


def clamp_i16(v: float) -> int:
    q = int(round(v))
    return max(-32768, min(32767, q))


def gen_dt_speech(num_samples: int) -> list[int]:
    """Gated two-tone near-end bursts (deterministic, no RNG): 0 during the
    convergence preamble, then 0.4 s on / 0.2 s off bursts."""
    import math

    start = DT_SPEECH_START_FRAME * FRAME_SIZE
    on = int(0.4 * SAMPLE_RATE)
    off = int(0.2 * SAMPLE_RATE)
    period = on + off
    out = [0] * num_samples
    for i in range(start, num_samples):
        t = (i - start) % period
        if t < on:
            x = 4000.0 * math.sin(2.0 * math.pi * 310.0 * i / SAMPLE_RATE)
            x += 2500.0 * math.sin(2.0 * math.pi * 1370.0 * i / SAMPLE_RATE + 0.7)
            # 8 Hz AM so the burst is speech-ish rather than a steady tone.
            x *= 0.6 + 0.4 * math.sin(2.0 * math.pi * 8.0 * i / SAMPLE_RATE)
            out[i] = clamp_i16(x)
    return out


def write_i16le(path: Path, samples: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(samples)}h", *samples))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_sources(srcdir: Path) -> None:
    speex = srcdir / "speex"
    speex.mkdir(parents=True, exist_ok=True)
    for rel in SOURCES:
        name = Path(rel).name
        dest = (speex / name) if "/speex/" in rel else (srcdir / name)
        if dest.exists():
            continue
        url = f"{RAW_BASE}/{rel}"
        print(f"  fetch {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    (speex / "speexdsp_config_types.h").write_text(CONFIG_TYPES_H)


def build_harness(srcdir: Path) -> Path:
    (srcdir / "harness.c").write_text(HARNESS_C)
    exe = srcdir / "aec_harness"
    cmd = [
        "cc",
        "-O2",
        # Load-bearing: no FMA contraction — keep plain f32 rounding so the
        # committed reference is comparable to the (non-FMA) Rust port.
        "-ffp-contract=off",
        "-DFLOATING_POINT",
        "-DUSE_SMALLFT",
        "-DEXPORT=",
        f"-I{srcdir}",
        str(srcdir / "harness.c"),
        str(srcdir / "mdf.c"),
        str(srcdir / "fftwrap.c"),
        str(srcdir / "smallft.c"),
        "-lm",
        "-o",
        str(exe),
    ]
    print("  cc:", " ".join(cmd[1:6]), "...")
    subprocess.run(cmd, check=True)
    return exe


def run_harness(exe: Path, far: Path, near: Path, out: Path) -> None:
    subprocess.run(
        [
            str(exe),
            str(FRAME_SIZE),
            str(FILTER_LENGTH),
            str(SAMPLE_RATE),
            str(far),
            str(near),
            str(out),
        ],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    ap.add_argument("--outdir", type=Path, default=repo_root / "tests" / "parity" / "aec")
    ap.add_argument(
        "--srcdir",
        type=Path,
        default=None,
        help="pre-seeded speexdsp source dir (skips the download)",
    )
    args = ap.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="speexdsp-aec-") as tmp:
        srcdir = args.srcdir or Path(tmp) / "src"
        print(f"[1/4] sources @ {SPEEXDSP_COMMIT[:12]}")
        fetch_sources(srcdir)
        print("[2/4] build harness")
        exe = build_harness(srcdir)

        print("[3/4] signals")
        # Single-talk.
        st_n = ST_FRAMES * FRAME_SIZE
        st_far = gen_far(st_n, SEED_FAR_ST)
        st_echo = convolve_echo(st_far)
        st_near = [clamp_i16(v) for v in st_echo]
        write_i16le(outdir / "st_far.i16le", st_far)
        write_i16le(outdir / "st_near.i16le", st_near)
        # Double-talk.
        dt_n = DT_FRAMES * FRAME_SIZE
        dt_far = gen_far(dt_n, SEED_FAR_DT)
        dt_echo = convolve_echo(dt_far)
        dt_speech = gen_dt_speech(dt_n)
        dt_near = [clamp_i16(e + s) for e, s in zip(dt_echo, dt_speech)]
        write_i16le(outdir / "dt_far.i16le", dt_far)
        write_i16le(outdir / "dt_near.i16le", dt_near)
        write_i16le(outdir / "dt_speech.i16le", dt_speech)

        print("[4/4] upstream reference run")
        run_harness(exe, outdir / "st_far.i16le", outdir / "st_near.i16le",
                    outdir / "st_out_speexdsp.i16le")
        run_harness(exe, outdir / "dt_far.i16le", outdir / "dt_near.i16le",
                    outdir / "dt_out_speexdsp.i16le")

    files = sorted(p.name for p in outdir.glob("*.i16le"))
    lines = [
        "M4-03 aec parity fixtures (SpeexDSP float-build reference)",
        f"speexdsp_commit = {SPEEXDSP_COMMIT}",
        f"sample_rate = {SAMPLE_RATE}",
        f"frame_size = {FRAME_SIZE}",
        f"filter_length = {FILTER_LENGTH}",
        f"st_frames = {ST_FRAMES}",
        f"dt_frames = {DT_FRAMES}",
        f"dt_speech_start_frame = {DT_SPEECH_START_FRAME}",
        f"far_amp = {FAR_AMP}",
        f"seed_far_st = {SEED_FAR_ST:#x}",
        f"seed_far_dt = {SEED_FAR_DT:#x}",
        "echo_taps = " + ", ".join(f"({d}, {c})" for d, c in ECHO_TAPS),
        "cflags = -O2 -ffp-contract=off -DFLOATING_POINT -DUSE_SMALLFT",
        f"generator_platform = {platform.platform()} / {platform.machine()}",
        "format = raw little-endian int16, mono",
        "",
    ]
    for name in files:
        lines.append(f"sha256 {name} = {sha256(outdir / name)}")
    (outdir / "manifest.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir}/manifest.txt")
    for line in lines[-len(files):]:
        print(" ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
