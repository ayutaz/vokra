#!/usr/bin/env python3
"""Merge SpeechBrain SepFormer 3-part checkpoint bundle → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``speechbrain/sepformer-*`` HF repos (``wsj02mix`` /
``wham16k-enhancement`` / ``whamr16k``) all share the same 4-file layout,
split by SpeechBrain's ``Checkpointer`` which writes one ``.ckpt`` per
recoverable Module via ``torch_recovery = torch.save(state_dict, path)``:

=================   ========   ==========================================
file                size       payload
=================   ========   ==========================================
``encoder.ckpt``    ~17 kB     ``nn.Conv1d`` — waveform → latent
``decoder.ckpt``    ~17 kB     ``ConvTranspose1d`` — latent → waveform
``masknet.ckpt``    ~113 MB    ``SepformerWrapper`` (dual-path Transformer)
``brain.ckpt``      ~30 bytes  epoch/step counter (opt-in via --include-brain)
=================   ========   ==========================================

The Rust converter (``crates/vokra-convert/src/models/sepformer.rs``) consumes
**safetensors only** (no pickle in the runtime tree). This script bridges the
two: it loads each ``.ckpt`` state dict, prefixes every key with its
file-stem (``encoder.`` / ``decoder.`` / ``masknet.``) so a future
``SepFormer::from_gguf`` can locate the sub-module a tensor belongs to, and
writes a single ``.safetensors`` the caller feeds to
``vokra-cli convert --model sepformer[-wham16k-enhancement|-whamr16k]``.

Precedent: ``nemo_pt_to_safetensors.py`` (single-file .pt/.nemo → safetensors)
+ ``kokoro_prepare_checkpoint.py`` (nested .pth + per-voice .pt merge →
safetensors). This script is the multi-file cousin — same INT-dtype filter,
same ``.stripped-manifest.json`` sidecar, same fail-loud posture.

# Usage

::

    uv run --project tools/parity python tools/parity/sepformer_prepare_checkpoint.py \\
        --ckpt-dir /path/to/sepformer-wham16k-enhancement \\
        --output /tmp/sepformer-wham16k-enh.safetensors \\
        [--include-brain] [--allow-strip-any]

Only the 3 usable ckpts (encoder/decoder/masknet) are merged by default;
``brain.ckpt`` is a training epoch counter and carries no weight the
inference forward path uses.

# Determinism

Keys are ordered by (stem_order, stem-local dict-iteration order — Python
dict preserves insertion order since 3.7, and ``torch.load(weights_only=True)``
is deterministic). Identical ``--ckpt-dir`` input produces byte-identical
output (safetensors serialization is deterministic for fixed key ordering).

# Redistribution

Upstream weight license is ``apache-2.0`` (SpeechBrain family) — see
``docs/license-audit.md`` §3.1 rows "SepFormer WSJ0-2mix" /
"SepFormer WHAM 16k enhancement" / "SepFormer WHAM-R 16k", all
☑ Commercial as of 2026-07-30.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The 3 checkpoint stems the runtime forward path needs. ``brain.ckpt`` is
# an epoch counter — inert at inference — and is opt-in via --include-brain
# so a caller who has a reason to keep the training marker in the merged
# artifact can, but the default artifact stays weight-only.
INFER_CKPTS = ("encoder", "decoder", "masknet")
INERT_CKPTS = ("brain",)

# Mirrors the classification in ``nemo_pt_to_safetensors.py``. INT dtypes
# come from BatchNorm ``num_batches_tracked`` counters and similar training
# artefacts — safe to strip. Any dtype outside both sets is refused unless
# --allow-strip-any is passed (fail-loud posture: the runtime forward path
# would refuse them anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into dotted-key ``{name: Tensor}``.

    SpeechBrain's ``torch_recovery`` normally pickles a flat state_dict, but
    some Lightning-style forks wrap it as ``{"state_dict": {...}}`` — this
    walk handles both without special-casing the wrapper name (the unwrap
    happens in ``_load_ckpt`` before this walk is invoked).
    """
    import torch

    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(key, v))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    # else: silently drop non-Tensor scalars/arrays — safetensors would
    # refuse them anyway; the runtime doesn't need them.
    return out


def _load_ckpt(path: Path, stem: str) -> dict:
    """Load one .ckpt state_dict and return a flat ``{stem.name: Tensor}``.

    Fail-loud on any load failure OR on a payload that yields zero tensors —
    better a hard exit than a silently-empty prefix (the downstream Rust
    converter would then emit a GGUF with a valid header but no weights and
    the runtime forward would only fail much later at first-forward, which
    is the classic "silent partial" trap this project bans).
    """
    import torch

    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"torch.load({path!s}, weights_only=True) failed: {exc}")

    # Common Lightning-style wrapper unwrap. SpeechBrain's default
    # ``torch_recovery`` writes the flat state_dict; forks that wrap it are
    # rare but do exist in the wild.
    if isinstance(raw, dict):
        for wrapper in ("state_dict", "model_state_dict", "model", "module"):
            inner = raw.get(wrapper)
            if isinstance(inner, dict) and inner:
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    print(f"  {stem}.ckpt: unwrapped ['{wrapper}']")
                    raw = inner
                    break

    flat_local = _flatten("", raw)
    if not flat_local:
        sys.exit(
            f"{path!s} yielded no tensors — expected a SpeechBrain state_dict "
            f"pickled by ``torch_recovery`` (see "
            f"github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/checkpoints.py)."
        )
    # Namespace under the ckpt stem so a future ``SepFormer::from_gguf`` can
    # locate the sub-module a tensor belongs to (``encoder.conv.weight`` vs.
    # ``masknet.mdl.0.norm.weight`` etc.). Without this prefix an encoder
    # ``weight`` key and a masknet inner ``weight`` key would collide in the
    # merged dict.
    prefixed = {f"{stem}.{k}": v for k, v in flat_local.items()}
    return prefixed


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy the
    ``nemo_pt_to_safetensors.py`` precedent uses."""
    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)
        if dtype_s in KEEP_DTYPES:
            if hasattr(t, "contiguous"):
                t = t.contiguous()
            if hasattr(t, "detach"):
                t = t.detach()
            kept[name] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((name, dtype_s, list(t.shape)))
        else:
            unknown.append((name, dtype_s, list(t.shape)))
    return kept, dropped, unknown


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge SpeechBrain SepFormer 3-part checkpoint bundle → single .safetensors",
    )
    ap.add_argument(
        "--ckpt-dir", required=True, type=Path,
        help=(
            "directory holding encoder.ckpt / decoder.ckpt / masknet.ckpt "
            "(and optionally brain.ckpt) — typically the output of "
            "`huggingface-cli download speechbrain/sepformer-* --local-dir <dir>`."
        ),
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--include-brain", action="store_true",
        help=(
            "also merge brain.ckpt (training epoch counter — inert at "
            "inference). Off by default: brain.ckpt is ~30 bytes and carries "
            "no weight the forward path uses."
        ),
    )
    ap.add_argument(
        "--allow-strip-any", action="store_true",
        help="also strip fp64 / complex tensors (default: refuse them loudly).",
    )
    args = ap.parse_args()

    try:
        from safetensors.torch import save_file
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"missing dep {exc}. run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    ckpt_dir: Path = args.ckpt_dir
    if not ckpt_dir.is_dir():
        print(f"--ckpt-dir must be an existing directory: {ckpt_dir}", file=sys.stderr)
        return 2

    stems: list[str] = list(INFER_CKPTS)
    if args.include_brain:
        stems.extend(INERT_CKPTS)

    merged: dict = {}
    per_ckpt_counts: dict[str, int] = {}
    for stem in stems:
        ckpt_path = ckpt_dir / f"{stem}.ckpt"
        if not ckpt_path.is_file():
            if stem in INERT_CKPTS:
                print(
                    f"  {stem}.ckpt not found (opt-in inert), skipping",
                    file=sys.stderr,
                )
                continue
            print(f"required checkpoint missing: {ckpt_path}", file=sys.stderr)
            return 2
        print(f"  loading {ckpt_path.name} ({ckpt_path.stat().st_size:,} bytes)")
        sub = _load_ckpt(ckpt_path, stem)
        # Duplicate-key guard: the stem prefix makes cross-stem collision
        # impossible in practice, but this redundant assert catches an
        # accidental within-stem duplicate the flatten walk cannot see (a
        # nested dict that re-uses a leaf name under two branches).
        overlap = set(merged) & set(sub)
        if overlap:
            print(
                f"  duplicate keys after prefix (first 5): {sorted(overlap)[:5]}",
                file=sys.stderr,
            )
            return 3
        merged.update(sub)
        per_ckpt_counts[stem] = len(sub)

    kept, dropped, unknown = _partition(merged, args.allow_strip_any)

    if unknown and not args.allow_strip_any:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"refusing to drop {len(unknown)} tensors of unknown dtype "
            f"(first 3: {first}); re-run with --allow-strip-any if verified "
            f"inference-inert.",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input_dir": str(ckpt_dir),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "per_ckpt": per_ckpt_counts,
        "include_brain": args.include_brain,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_stripped": (
            [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown]
            if args.allow_strip_any else []
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"sepformer_prepare_checkpoint: kept {len(kept)}, "
        f"dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"per-ckpt {per_ckpt_counts}; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
