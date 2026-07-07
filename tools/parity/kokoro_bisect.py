#!/usr/bin/env python3
"""Bisect Kokoro decoder stage-by-stage vs the PyTorch reference (T17-fixup #1).

Reads paired stage dumps `<dir>/native_<stage>.f32` and `<dir>/ref_<stage>.f32`
written by the Rust parity harness (via `super::maybe_dump_stage`) and the
Python reference dumper (via `_maybe_dump_stage`). For each pair, prints:

    stage = <name>
      max |Δ|       = <value>
      worst @ idx   = <index>
      worst native  = <val>   ref = <val>
      length        = <N> floats
      tail-localized: <Y/N>  (worst index in last 1% of samples)
      histogram (|Δ|): [<1e-4, <1e-3, <1e-2, <1e-1, <1, ≥1]

The FIRST stage where max |Δ| exceeds `atol = 0.01` is highlighted as the
divergence site. Zero external deps (stdlib only).
"""

from __future__ import annotations

import os
import struct
import sys

ATOL = 0.01
BUCKETS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # <1e-4, <1e-3, <1e-2, <1e-1, <1, else ≥1


def read_f32(path: str) -> list[float]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise SystemExit(f"{path}: {len(data)} bytes, not divisible by 4")
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def worst_delta(got: list[float], ref: list[float]) -> tuple[float, int, list[int]]:
    """Returns (max |Δ|, worst_idx, histogram of 6 buckets).

    Skips positions where either side is non-finite. NaN / Inf pairs are counted
    only in the sample-count total, not in the histogram.
    """
    assert len(got) == len(ref), f"length mismatch {len(got)} vs {len(ref)}"
    worst = 0.0
    worst_i = 0
    hist = [0, 0, 0, 0, 0, 0]
    for i, (g, r) in enumerate(zip(got, ref)):
        # `!= g` is the classic NaN check; Inf compares equal to itself so we
        # allow Inf-Inf to pair (both models can hit exp() overflow at tails).
        if g != g or r != r:
            continue
        d = abs(g - r)
        if d > worst:
            worst = d
            worst_i = i
        placed = False
        for b, thresh in enumerate(BUCKETS):
            if d < thresh:
                hist[b] += 1
                placed = True
                break
        if not placed:
            hist[5] += 1
    return worst, worst_i, hist


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: kokoro_bisect.py <dump-dir>\n"
            "  where <dump-dir> contains native_<stage>.f32 and ref_<stage>.f32 files"
        )
    dump_dir = sys.argv[1]
    if not os.path.isdir(dump_dir):
        raise SystemExit(f"not a directory: {dump_dir}")

    # Discover paired stages (both native_* and ref_* must exist).
    files = os.listdir(dump_dir)
    native = {f[len("native_") : -len(".f32")] for f in files
              if f.startswith("native_") and f.endswith(".f32")}
    ref = {f[len("ref_") : -len(".f32")] for f in files
           if f.startswith("ref_") and f.endswith(".f32")}
    paired = sorted(native & ref)
    native_only = sorted(native - ref)
    ref_only = sorted(ref - native)

    if native_only:
        print(f"[bisect] native-only stages (no ref dump): {native_only}")
    if ref_only:
        print(f"[bisect] ref-only stages (no native dump): {ref_only}")
    if not paired:
        raise SystemExit(
            "no paired native_<stage>.f32/ref_<stage>.f32 files found. "
            "Run the parity test AND the reference dumper with "
            "VOKRA_KOKORO_PARITY_DUMP=<dir> set to the same directory."
        )

    print(f"[bisect] {len(paired)} paired stages under {dump_dir}\n")

    first_diverge: str | None = None
    ordered = _sort_by_pipeline_order(paired)

    for stage in ordered:
        native_path = os.path.join(dump_dir, f"native_{stage}.f32")
        ref_path = os.path.join(dump_dir, f"ref_{stage}.f32")
        try:
            g = read_f32(native_path)
            r = read_f32(ref_path)
        except SystemExit as e:
            print(f"stage = {stage}  [SKIP: {e}]")
            continue
        if len(g) != len(r):
            print(
                f"stage = {stage}\n"
                f"  LENGTH MISMATCH: native {len(g)} vs ref {len(r)}"
            )
            if first_diverge is None:
                first_diverge = stage + " (length mismatch)"
            continue
        max_d, idx, hist = worst_delta(g, r)
        n = len(g)
        tail_thresh = int(n * 0.99)
        tail_local = "Y" if idx >= tail_thresh else "N"
        exceeds = max_d > ATOL
        marker = " *** EXCEEDS ATOL ***" if exceeds else ""
        print(
            f"stage = {stage}{marker}\n"
            f"  length        = {n} floats\n"
            f"  max |Δ|       = {max_d:.3e}   (atol = {ATOL})\n"
            f"  worst @ idx   = {idx}\n"
            f"  worst native  = {g[idx]:.6f}   ref = {r[idx]:.6f}\n"
            f"  tail-localized (idx >= 99%): {tail_local}\n"
            f"  histogram (|Δ|): <1e-4:{hist[0]} <1e-3:{hist[1]} "
            f"<1e-2:{hist[2]} <1e-1:{hist[3]} <1:{hist[4]} >=1:{hist[5]}"
        )
        if exceeds and first_diverge is None:
            first_diverge = stage

    print()
    if first_diverge is None:
        print("[bisect] ALL STAGES PASS AT ATOL = 0.01. No divergence found.")
    else:
        print(f"[bisect] FIRST DIVERGENCE: stage = {first_diverge}")
        print(
            "[bisect] Fix the op that produces this stage. Later stages "
            "propagate the error and should be ignored until this one is fixed."
        )


def _sort_by_pipeline_order(stages: list[str]) -> list[str]:
    """Sort stages in Kokoro decoder pipeline order (encode → decode.0..3 →
    asr_res → pre_generator → gen_stage_0_* → gen_stage_1_* → gen_pre_conv_post
    → gen_conv_post). Unknown stages fall to the end alphabetically.
    """
    prefix_order = [
        "dec_encode",
        "dec_asr_res",
        "dec_decode_0",
        "dec_decode_1",
        "dec_decode_2",
        "dec_decode_3",
        "dec_pre_generator",
        "gen_stage_0_pre_ups",
        "gen_stage_0_ups",
        "gen_stage_0_noise_pre_res",
        "gen_stage_0_noise_post_res",
        "gen_stage_0_fused",
        "gen_stage_0_rb_0",
        "gen_stage_0_rb_1",
        "gen_stage_0_rb_2",
        "gen_stage_0_mrf",
        "gen_stage_1_pre_ups",
        "gen_stage_1_ups",
        "gen_stage_1_noise_pre_res",
        "gen_stage_1_noise_post_res",
        "gen_stage_1_fused",
        "gen_stage_1_rb_0",
        "gen_stage_1_rb_1",
        "gen_stage_1_rb_2",
        "gen_stage_1_mrf",
        "gen_pre_conv_post",
        "gen_conv_post",
        "decoder_pre_istft_mag",
        "decoder_pre_istft_phase",
        "decoder_pcm",
    ]
    order_map = {name: i for i, name in enumerate(prefix_order)}

    def key(s: str):
        # Known stage → deterministic priority; unknown → sort alphabetically
        # after all knowns.
        return (order_map.get(s, len(prefix_order) + 1), s)

    return sorted(stages, key=key)


if __name__ == "__main__":
    main()
