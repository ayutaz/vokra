#!/usr/bin/env python3
# sber_gigaam_v3_prepare_checkpoint.py — bridge an upstream Sber GigaAM v3
# release (`ai-sage/GigaAM-v3`) into a single safetensors file the vokra
# `--model sber-gigaam-v3` converter can consume.
#
# Supported inputs (dispatch on file extension):
#   - .nemo  — NVIDIA NeMo bundle (tar/tar.gz/zip containing
#              model_weights.{ckpt,pt}); the Sber GigaAM release ships both
#              a NeMo bundle and flattened safetensors.
#   - .pt / .pth / .ckpt / .bin — raw torch pickle state_dict (or a
#              {`state_dict`|`model`|`module`} wrapper around one).
#   - .safetensors — already flattened; the script hard-errors and points
#              the caller at `vokra-convert --model sber-gigaam-v3` directly
#              (running this bridge is a no-op in that case).
#
# Output:
#   - <output>.safetensors — every F32 / F16 / BF16 tensor from the upstream
#     state_dict, deduped for shared storage (SBv2 / Bark-style tied
#     embeddings would collide in `safetensors.torch.save_file` otherwise —
#     see memory [[reference-safetensors-shared-tensor-dedup]]).
#   - <output>.shared_pairs.json — audit trail of every shared-storage
#     dedup that happened (empty list if there was no aliasing).
#   - <output>.stripped-manifest.json — training-only int / bool counters
#     that were dropped (`.num_batches_tracked`, etc), plus any unknown
#     dtypes the caller opted to strip.
#
# Runtime posture: uv-managed Python 3.12 (pyproject.toml in this tree
# pins `requires-python = ">=3.12"`). This script imports torch +
# safetensors + numpy at run time only; the vokra runtime NEVER imports
# them (FR-LD-05).
#
# Usage:
#   uv run python sber_gigaam_v3_prepare_checkpoint.py \
#       --input <in.nemo|in.pt|in.pth|in.ckpt|in.bin> \
#       --output <out.safetensors> \
#       [--tensor-prefix-strip <prefix>] \
#       [--allow-strip-any]

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

# Dtype tables shared with `nemo_pt_to_safetensors.py` — the vokra
# `--model sber-gigaam-v3` converter only accepts F32 / F16 / BF16 (the
# safetensors reader hard-errors on anything else at parse time,
# `crates/vokra-core/src/safetensors.rs map_dtype`), so keeping the two
# sets aligned is critical.
INT_DTYPES = {
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.uint8",
    "torch.uint16",
    "torch.uint32",
    "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def extract_state_dict_from_nemo(path: Path):
    """Extract the state_dict from a `.nemo` tarball / zip.

    NeMo bundles typically contain `model_weights.ckpt` at the top level
    (with optional `.yaml` config sidecars); older releases may name the
    weight file `weights.ckpt` or `model_weights.pt`. This helper walks
    every plausible candidate before giving up.
    """
    import torch

    # Try tar auto-detect (handles both plain tar and tar.gz).
    tar = None
    try:
        tar = tarfile.open(path, "r:*")
    except tarfile.ReadError:
        # Fall back to zip.
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                ckpt_name = None
                for cand in (
                    "model_weights.ckpt",
                    "weights.ckpt",
                    "model_weights.pt",
                    "weights.pt",
                ):
                    if cand in names:
                        ckpt_name = cand
                        break
                if not ckpt_name:
                    for m in names:
                        if m.endswith((".ckpt", ".pt", ".pth")):
                            ckpt_name = m
                            break
                if not ckpt_name:
                    raise SystemExit(
                        f"no .ckpt/.pt inside zip {path}. members={names[:20]}"
                    )
                print(f"  extracting {ckpt_name} from {path.name} zip")
                data = zf.read(ckpt_name)
                print(f"  torch.load({len(data):,} bytes)")
                return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        raise SystemExit(f"{path} is neither tar/tar.gz nor zip")

    with tar:
        members = tar.getnames()
        ckpt_name = None
        for cand in (
            "model_weights.ckpt",
            "weights.ckpt",
            "model_weights.pt",
            "weights.pt",
        ):
            if cand in members:
                ckpt_name = cand
                break
        if not ckpt_name:
            for m in members:
                if m.endswith((".ckpt", ".pt", ".pth")):
                    ckpt_name = m
                    break
        if not ckpt_name:
            raise SystemExit(f"no .ckpt/.pt inside {path}. members={members[:20]}")
        print(f"  extracting {ckpt_name} from {path.name} tar")
        f = tar.extractfile(ckpt_name)
        if f is None:
            raise SystemExit(f"could not open {ckpt_name} inside tar")
        data = f.read()
    print(f"  torch.load({len(data):,} bytes)")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def extract_state_dict_from_pt(path: Path):
    """Raw `torch.load`, possibly wrapped `{'state_dict': ...}` or
    `{'model': ...}` (Lightning + NeMo convention)."""
    import torch

    print(f"  torch.load({path.stat().st_size:,} bytes)")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def flatten_and_partition(sd, prefix_strip: str | None = None):
    """Walk any dict wrapper (`state_dict` / `model` / `module`) and
    partition the resulting tensor map into:

    - `kept`: float tensors (F32 / F16 / BF16) to write to safetensors,
    - `dropped`: int / bool counters explicitly recognised as
      training-only (BN `num_batches_tracked` etc),
    - `unknown`: tensors whose dtype the caller must opt into stripping
      with `--allow-strip-any`.

    Returns `(kept, dropped, unknown, shared_pairs)` — `shared_pairs`
    records every (canonical, aliased) name pair the dedup pass folded
    into a single storage slot (see memory
    [[reference-safetensors-shared-tensor-dedup]]).
    """
    import torch

    # Common wrapper patterns (Lightning: `state_dict`; some NeMo
    # bundles: `model_state_dict` / `model` / `module`).
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "module"):
            if k in sd and isinstance(sd[k], dict):
                inner = sd[k]
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    sd = inner
                    print(f"  unwrapped ['{k}']")
                    break

    if not isinstance(sd, dict):
        raise SystemExit(f"expected dict at top level, got {type(sd)}")

    kept: dict[str, torch.Tensor] = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    # Shared-storage dedup: safetensors.torch.save_file refuses two
    # entries with the same data_ptr (tied embeddings would trip this).
    # Track (data_ptr -> canonical name) so a second alias is folded
    # into a JSON audit trail instead of duplicating the payload or
    # crashing the save call.
    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        # Skip non-tensor entries (metadata dicts etc).
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)
        n = name
        if prefix_strip and n.startswith(prefix_strip):
            n = n[len(prefix_strip) :]

        if dtype_s in KEEP_DTYPES:
            # Detach + ensure contiguous storage so safetensors' invariants
            # hold (non-contiguous / gradient-tracking tensors would fail
            # `save_file`).
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "contiguous"):
                t = t.contiguous()

            # GGUF caps tensor rank at 8 dims (raised from 4 in
            # `58629ab feat(core/gguf): raise tensor rank cap to 8`).
            # Sber GigaAM's Conformer tensors are all ≤4D by
            # construction, but pin the invariant here so a future
            # weird release surfaces loudly.
            if len(t.shape) > 8:
                orig_shape = tuple(t.shape)
                t = t.squeeze()
                if hasattr(t, "contiguous"):
                    t = t.contiguous()
                if len(t.shape) == 0:
                    t = t.reshape(1)
                if len(t.shape) > 8:
                    raise SystemExit(
                        f"tensor {n!r} has {len(orig_shape)}D shape {orig_shape} "
                        f"and cannot be reduced to <=8D by squeezing singleton dims; "
                        f"post-squeeze shape = {tuple(t.shape)}. GGUF hard cap = 8D."
                    )
                print(
                    f"  squeezed {n}: {orig_shape} -> {tuple(t.shape)} (GGUF 8D cap)"
                )

            # Shared-storage dedup.
            ptr = t.data_ptr()
            if ptr in seen:
                canonical = seen[ptr]
                shared_pairs.append((canonical, n))
                continue
            seen[ptr] = n
            kept[n] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((n, dtype_s, list(t.shape)))
        else:
            unknown.append((n, dtype_s, list(t.shape)))
    return kept, dropped, unknown, shared_pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--tensor-prefix-strip",
        default=None,
        help="strip this prefix from tensor names (e.g. 'model.' or 'module.')",
    )
    p.add_argument(
        "--allow-strip-any",
        action="store_true",
        help="also strip fp64/complex tensors (default: refuse them loudly)",
    )
    args = p.parse_args()

    try:
        from safetensors.torch import save_file
        import torch  # noqa: F401
    except ImportError as e:
        print(f"missing dep {e}", file=sys.stderr)
        return 2

    inp = args.input
    if not inp.exists():
        print(f"input not found: {inp}", file=sys.stderr)
        return 2

    # Dispatch on extension.
    suffix = inp.suffix.lower()
    if suffix == ".safetensors":
        print(
            f"input {inp} is already safetensors — run "
            f"`vokra-convert --model sber-gigaam-v3 --input {inp} --output <out.gguf>` "
            f"directly instead of the prepare step.",
            file=sys.stderr,
        )
        return 2
    if suffix == ".nemo":
        sd = extract_state_dict_from_nemo(inp)
    elif suffix in (".pt", ".pth", ".ckpt", ".bin"):
        sd = extract_state_dict_from_pt(inp)
    else:
        print(f"unknown input extension {suffix}", file=sys.stderr)
        return 2

    kept, dropped, unknown, shared_pairs = flatten_and_partition(
        sd, args.tensor_prefix_strip
    )

    if unknown and not args.allow_strip_any:
        print(
            f"refusing to drop {len(unknown)} tensors of unknown class "
            f"(first 3: {unknown[:3]}); re-run with --allow-strip-any if verified inference-inert",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "prefix_strip": args.tensor_prefix_strip,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_stripped": (
            [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown]
            if args.allow_strip_any
            else []
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    shared_path = args.output.with_suffix(args.output.suffix + ".shared_pairs.json")
    shared_path.write_text(
        json.dumps(
            {
                "shared_pairs": [
                    {"canonical": c, "aliased": a} for c, a in shared_pairs
                ]
            },
            indent=2,
        )
    )

    print(
        f"sber_gigaam_v3_prepare_checkpoint: kept {len(kept)}, dropped "
        f"{len(dropped)} int, stripped {len(unknown) if args.allow_strip_any else 0} "
        f"unknown, deduped {len(shared_pairs)} shared-storage aliases; "
        f"manifest -> {manifest_path.name}, shared-pairs -> {shared_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
