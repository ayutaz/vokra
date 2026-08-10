#!/usr/bin/env python3
# nemo_pt_to_safetensors.py — extract weights from NVIDIA .nemo (tar.gz) or
# torch .pt / .pth pickle into a single .safetensors + a .stripped-manifest.json
# sidecar. Companion of strip_int_tensors.py (which strips int tensors from
# already-safetensors files).
#
# Why:
#   Vokra converters expect safetensors input. Some upstream ASR checkpoints
#   ship .nemo (NeMo Toolkit's tar-of-yaml+ckpt) or raw .pt (torch pickle).
#   This tool unwraps them into safetensors + drops training-only int
#   counters (BatchNorm num_batches_tracked etc), producing an artifact the
#   vokra-cli converter can consume.
#
# Usage:
#   uv run python nemo_pt_to_safetensors.py --input <in.nemo|in.pt> \
#                                            --output <out.safetensors> \
#                                            [--tensor-prefix-strip <prefix>] \
#                                            [--allow-strip-any]

from __future__ import annotations
import argparse
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

INT_DTYPES = {"torch.int8", "torch.int16", "torch.int32", "torch.int64",
              "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
              "torch.bool"}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def extract_state_dict_from_nemo(path: Path):
    """`.nemo` is either tar / tar.gz / zip containing model_weights.{ckpt,pt}."""
    import torch

    # Try tar auto-detect (handles both plain tar and tar.gz)
    tar = None
    try:
        tar = tarfile.open(path, "r:*")
    except tarfile.ReadError:
        # Try zip
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                ckpt_name = None
                for cand in ("model_weights.ckpt", "weights.ckpt", "model_weights.pt", "weights.pt"):
                    if cand in names:
                        ckpt_name = cand
                        break
                if not ckpt_name:
                    for m in names:
                        if m.endswith((".ckpt", ".pt", ".pth")):
                            ckpt_name = m
                            break
                if not ckpt_name:
                    raise SystemExit(f"no .ckpt/.pt inside zip {path}. members={names[:20]}")
                print(f"  extracting {ckpt_name} from {path.name} zip")
                data = zf.read(ckpt_name)
                print(f"  torch.load({len(data):,} bytes)")
                return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        raise SystemExit(f"{path} is neither tar/tar.gz nor zip")

    with tar:
        members = tar.getnames()
        # Prefer model_weights.ckpt or weights.ckpt
        ckpt_name = None
        for cand in ("model_weights.ckpt", "weights.ckpt", "model_weights.pt", "weights.pt"):
            if cand in members:
                ckpt_name = cand
                break
        if not ckpt_name:
            # Any .ckpt or .pt file
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
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return sd


def extract_state_dict_from_pt(path: Path):
    """Raw torch.load, possibly wrapped {'state_dict': ...} or {'model': ...}."""
    import torch
    print(f"  torch.load({path.stat().st_size:,} bytes)")
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    return sd


def flatten_and_partition(sd, prefix_strip: str | None = None):
    """Walk any dict wrapper (state_dict / model / module.), return (kept float dict, dropped list, unknown list)."""
    import torch

    # Common wrapper patterns
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "module"):
            if k in sd and isinstance(sd[k], dict):
                inner = sd[k]
                # Only unwrap if inner keys look tensor-ish
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    sd = inner
                    print(f"  unwrapped ['{k}']")
                    break

    if not isinstance(sd, dict):
        raise SystemExit(f"expected dict at top level, got {type(sd)}")

    kept: dict[str, "torch.Tensor"] = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []

    for name, t in sd.items():
        # Skip non-tensor entries (metadata dicts etc)
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)
        n = name
        if prefix_strip and n.startswith(prefix_strip):
            n = n[len(prefix_strip):]
        if dtype_s in KEEP_DTYPES:
            # Ensure contiguous
            if hasattr(t, "contiguous"):
                t = t.contiguous()
            # Detach to avoid gradient tracking
            if hasattr(t, "detach"):
                t = t.detach()
            # GGUF caps tensor rank at 4 dims. Squeeze trailing/interior singleton
            # dims to reach <=4D. Safe for broadcasting-shaped tensors (e.g.
            # ALiBi scale (1,1,16,1,1) → (1,1,16) after full squeeze then re-pad).
            if len(t.shape) > 4:
                orig_shape = tuple(t.shape)
                # Squeeze all size-1 dims, then keep result contiguous
                t = t.squeeze()
                if hasattr(t, "contiguous"):
                    t = t.contiguous()
                # Ensure we didn't over-squeeze scalars into 0D
                if len(t.shape) == 0:
                    t = t.reshape(1)
                if len(t.shape) > 4:
                    raise SystemExit(
                        f"tensor {n!r} has {len(orig_shape)}D shape {orig_shape} "
                        f"and cannot be reduced to <=4D by squeezing singleton dims; "
                        f"post-squeeze shape = {tuple(t.shape)}. GGUF hard cap = 4D."
                    )
                print(f"  squeezed {n}: {orig_shape} -> {tuple(t.shape)} (GGUF 4D cap)")
            kept[n] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((n, dtype_s, list(t.shape)))
        else:
            unknown.append((n, dtype_s, list(t.shape)))
    return kept, dropped, unknown


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tensor-prefix-strip", default=None,
                   help="strip this prefix from tensor names (e.g. 'model.' or 'module.')")
    p.add_argument("--allow-strip-any", action="store_true",
                   help="also strip fp64/complex (default: refuse them loudly)")
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

    # Dispatch on extension
    suffix = inp.suffix.lower()
    if suffix == ".nemo":
        sd = extract_state_dict_from_nemo(inp)
    elif suffix in (".pt", ".pth", ".ckpt", ".bin"):
        sd = extract_state_dict_from_pt(inp)
    else:
        print(f"unknown input extension {suffix}", file=sys.stderr)
        return 2

    kept, dropped, unknown = flatten_and_partition(sd, args.tensor_prefix_strip)

    if unknown and not args.allow_strip_any:
        print(f"refusing to drop {len(unknown)} tensors of unknown class "
              f"(first 3: {unknown[:3]}); re-run with --allow-strip-any if verified inference-inert",
              file=sys.stderr)
        return 3

    # Tied-embedding dedup — memory [[reference-safetensors-shared-tensor-dedup]]:
    # safetensors.torch.save_file refuses tensors that share the same data_ptr
    # (Bark / XTTS-v2 / MOSS variants / BERT MLM heads with tied
    # bert.embeddings.word_embeddings.weight <-> cls.predictions.decoder.weight
    # are the recurring cases). Clone the later occurrences so each name gets
    # a distinct storage — semantics are preserved (both names load identical
    # values), disk cost is minimal for a single tied pair, and downstream
    # converters that need only one of the pair simply pick the canonical one.
    # Audit trail lands in shared_pairs.json alongside the manifest.
    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []
    for n, t in list(kept.items()):
        try:
            ptr = t.data_ptr()
        except Exception:
            continue
        if ptr in seen:
            shared_pairs.append((seen[ptr], n))
            kept[n] = t.clone().contiguous()
        else:
            seen[ptr] = n

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "prefix_strip": args.tensor_prefix_strip,
        "dropped_tensors": [{"name": n, "dtype": d, "shape": s} for n, d, s in dropped],
        "unknown_stripped": [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown] if args.allow_strip_any else [],
        "shared_pairs": [{"canonical": a, "cloned": b} for a, b in shared_pairs],
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"nemo_pt_to_safetensors: kept {len(kept)}, dropped {len(dropped)} int, "
          f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
          f"manifest -> {manifest_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
