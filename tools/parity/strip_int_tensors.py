#!/usr/bin/env python3
# strip_int_tensors.py — copy a safetensors file while dropping tensors whose
# dtype is int8/int16/int32/int64/uint8/uint16/uint32/uint64/bool.
#
# Why this exists (do NOT delete this docstring):
#   Vokra's safetensors parser (crates/vokra-core/src/safetensors.rs) accepts
#   ONLY F32 / F16 / BF16 dtypes at header-parse time. Some upstream
#   checkpoints ship training-only counters like BatchNorm
#   `num_batches_tracked` (torch.int64, scalar shape []) alongside the float
#   weights. These counters are inference-inert (they are only touched during
#   training's running-stat update — the eval() path reads only
#   running_mean / running_var / weight / bias, all F32) so stripping them
#   preserves inference correctness bit-exactly.
#
#   This is NOT a "silent skip" — it is a deliberate front-end normalizer,
#   invoked explicitly by the caller, that produces a smaller intermediate
#   safetensors AND emits a manifest of what was dropped so the removal is
#   auditable. The converter itself remains strict (F32/F16/BF16 only),
#   which is the desired invariant.
#
# Usage:
#   uv run python strip_int_tensors.py --input <in.safetensors> --output <out.safetensors>
#     [--allow-strip-any]   # also strip float64/complex if present (default: refuse)

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

INT_DTYPES = {"torch.int8", "torch.int16", "torch.int32", "torch.int64",
              "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
              "torch.bool"}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-strip-any", action="store_true",
                   help="also strip float64/complex (default: refuse them loudly)")
    args = p.parse_args()

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch  # noqa: F401 — loaded so torch dtypes stringify
    except ImportError as e:
        print(f"strip_int_tensors: missing dep {e}", file=sys.stderr)
        return 2

    kept: dict[str, "torch.Tensor"] = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []

    with safe_open(str(args.input), framework="pt") as sf:
        keys = list(sf.keys())
        for name in keys:
            t = sf.get_tensor(name)
            dtype_s = str(t.dtype)
            if dtype_s in KEEP_DTYPES:
                kept[name] = t
            elif dtype_s in INT_DTYPES:
                dropped.append((name, dtype_s, list(t.shape)))
            else:
                unknown.append((name, dtype_s, list(t.shape)))

    if unknown and not args.allow_strip_any:
        print(f"strip_int_tensors: refusing to drop {len(unknown)} tensors of "
              f"unknown class (first 3: {unknown[:3]}). Re-run with "
              f"--allow-strip-any if you have verified these are training-only "
              f"and inference-inert.", file=sys.stderr)
        return 3

    # Save the kept float tensors + emit a manifest sidecar next to the output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "dropped_tensors": [{"name": n, "dtype": d, "shape": s} for n, d, s in dropped],
        "unknown_stripped": [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown] if args.allow_strip_any else [],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"strip_int_tensors: kept {len(kept)}, dropped {len(dropped)} "
          f"int tensors; manifest -> {manifest_path}")
    if dropped[:3]:
        print(f"  first dropped: {dropped[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
