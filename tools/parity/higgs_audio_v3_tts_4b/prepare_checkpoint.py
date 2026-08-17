#!/usr/bin/env python3
"""Bridge BosonAI Higgs-Audio v3 TTS 4B sharded safetensors → flat safetensors.

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream `bosonai/higgs-audio-v3-tts-4b` release ships
**sharded safetensors** (`model-00001-of-000NN.safetensors` +
`model.safetensors.index.json`, ~8 GB total in BF16 for the 4B
backbone). The Rust converter
(``crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs``) consumes
a **single** safetensors file so this script merges the shards, dedupes
tied tensors, strips non-float training-scaffold state, and writes
one flat output.

# What this script does

* Discovers the shard set from the sibling `model.safetensors.index.json`
  (upstream's canonical shard manifest) at `--input-dir`, or accepts an
  explicit shard glob via `--shards`. Refuses if no shards are found
  rather than silently emitting an empty output (FR-EX-08).
* Loads each shard on CPU via `safetensors.torch.load_file` (never
  runs any pickle — safetensors format is inert JSON header + raw
  bytes). Merges the shard state dicts in `index.json` weight-map
  order.
* Dedupes shared-storage tensors via a `data_ptr → canonical name`
  pass (memory `[[reference-safetensors-shared-tensor-dedup]]`):
  `safetensors.torch.save_file` refuses two names pointing at the
  same storage, so every duplicate is cloned + made contiguous into
  genuinely independent storage. The alias graph is preserved in
  `<output>.shared_pairs.json` so a downstream runtime binder can
  restore ties (e.g. tied text embedding + lm_head — a Qwen /
  MiniCPM family posture and a plausible Higgs-Audio topology).
* Drops non-float training-scaffold entries **explicitly** and reports
  each one:
   - `.num_batches_tracked` BatchNorm I64 counters (no inference
     role — eval-mode BatchNorm consumes only running_mean /
     running_var / weight / bias).
   - `.total_ops` / `.total_params` `torch.profiler` bookkeeping.
* Rejects unexpected non-float dtypes (I32 / I64 / F64 / Bool) that
  are NOT in the drop-list — FR-EX-08 no silent fallback. This
  catches upstream state that a future runtime binder needs to know
  about explicitly.
* F32 / F16 / BF16 pass through under their upstream dtype (the Rust
  converter's `GgmlType::F32 | GgmlType::F16 | GgmlType::BF16`
  pass-through arm handles all three; the runtime widens BF16 → f32
  losslessly at load via `crates/vokra-core/src/gguf/quant/mod.rs
  decode_bf16`).
* Emits a `<output>.sha256` line + parameter count + shard count to
  stdout for the fixture / workflow logs.

# Memory footprint

The 4B BF16 weights are ~8 GB on disk. This script loads shards
sequentially and merges into a single in-memory state dict — peak
resident memory is roughly the whole model (~8 GB) because
`safetensors.torch.save_file` needs the full dict for its serialise
call. This is the wave-b posture (magpietts_v2602 /
firered_asr_aed_l) and works on any ≥16 GB box. On the CC laptop
(M1 iMac 16 GB) this is at the edge of the safe zone; per memory
`[[feedback-large-models-on-vast-ai]]` the actual conversion runs
on vast.ai per `docs/handoff/vast-ai-large-model-publish.md`.

# Usage

Managed through `uv` per the tools/parity contract
(memory `[[feedback-python-uses-uv]] / [[feedback-python-3-12]]`).

::

    cd tools/parity/higgs_audio_v3_tts_4b
    uv sync

    # From a downloaded shard bundle (via `hf download`):
    uv run python prepare_checkpoint.py \\
        --input-dir /root/models/higgs-audio-v3-tts-4b \\
        --output    /root/models/higgs-audio-v3-tts-4b/merged.safetensors

Then:

::

    vokra-cli convert --model higgs-audio-v3-tts-4b \\
        --input  /root/models/higgs-audio-v3-tts-4b/merged.safetensors \\
        --output /root/gguf/higgs-audio-v3-tts-4b.gguf

# Determinism

Shard iteration order follows `model.safetensors.index.json`'s
weight-map order (upstream-produced); identical inputs produce
byte-identical safetensors output.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


# Non-float training-scaffold state that has no inference role. Any
# other non-float dtype is a hard error (FR-EX-08).
DROP_SUFFIXES = (
    ".num_batches_tracked",  # BatchNorm training counter.
    ".total_ops",  # torch.profiler bookkeeping.
    ".total_params",  # torch.profiler bookkeeping.
)

# Permitted float dtypes (mirror the Rust converter's pass-through arm).
ALLOWED_DTYPES = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
}


def discover_shards(input_dir: Path, explicit_shards: list[str] | None) -> list[Path]:
    """Enumerate the shard files to merge.

    Priority order:
    1. `--shards` glob if the caller supplied it (deterministic
       explicit control — used by tests / CI matrix runs).
    2. `model.safetensors.index.json` weight-map (the upstream
       canonical shard manifest).
    3. Glob `model-*-of-*.safetensors` in `input_dir` as a last
       resort — with a sanity check that the count matches the
       filename suffix `-of-000NN`.

    Refuses (hard exit) if no shards resolve, rather than emitting an
    empty output.
    """
    if explicit_shards:
        paths = sorted({Path(p) for pattern in explicit_shards for p in glob.glob(pattern)})
        if not paths:
            raise SystemExit(f"--shards patterns matched no files: {explicit_shards!r}")
        return paths

    index_path = input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"{index_path}: malformed JSON: {e}") from e
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SystemExit(
                f"{index_path}: `weight_map` is empty or not a dict — cannot enumerate shards"
            )
        # Preserve insertion order to walk shards in the same order the
        # upstream training loop wrote them (deterministic).
        shard_names: list[str] = []
        seen: set[str] = set()
        for shard in weight_map.values():
            if shard not in seen:
                seen.add(shard)
                shard_names.append(shard)
        paths = [input_dir / name for name in shard_names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(
                f"{index_path}: referenced shard files missing on disk: {missing[:5]}"
            )
        return paths

    # Last resort: glob the canonical shard filename shape. Refuses
    # if no matches, and warns if the count mismatches the `-of-000NN`
    # suffix.
    fallback = sorted(input_dir.glob("model-*-of-*.safetensors"))
    if not fallback:
        raise SystemExit(
            f"no shards found under {input_dir}: expected either "
            f"`model.safetensors.index.json` + `model-000NN-of-000MM.safetensors` shards, "
            f"or `--shards <glob>` explicit control"
        )
    # Sanity: last shard's `-of-000NN` suffix must equal `len(fallback)`.
    last = fallback[-1].name
    try:
        # e.g. "model-00003-of-00003.safetensors" → "00003"
        of_part = last.split("-of-")[1].split(".")[0]
        declared_count = int(of_part)
    except (IndexError, ValueError):
        # Not a fatal error — we still have shards, but log the drift.
        print(
            f"warning: cannot parse `-of-NNNNN` suffix from {last!r}; "
            f"proceeding with {len(fallback)} shards discovered by glob",
            file=sys.stderr,
        )
    else:
        if declared_count != len(fallback):
            raise SystemExit(
                f"shard count mismatch: filename declares {declared_count} shards "
                f"but glob found {len(fallback)} — refusing to guess"
            )
    return fallback


def dedupe_shared_storage(
    state: "OrderedDict[str, torch.Tensor]",
) -> tuple["OrderedDict[str, torch.Tensor]", list[tuple[str, str]]]:
    """Clone tensors sharing storage so `safetensors.torch.save_file` accepts them.

    Returns the deduped dict + a list of ``(alias_name, canonical_name)``
    pairs that were cloned; caller writes the pair list to a
    ``.shared_pairs.json`` audit trail so a downstream consumer knows
    which upstream names were tied in the original release (memory
    `[[reference-safetensors-shared-tensor-dedup]]`).

    In the Higgs-Audio family a plausible tie is the text embedding
    ↔ lm_head (Qwen / MiniCPM `tie_word_embeddings=true` posture). The
    audit trail preserves that fact even after we clone-and-detach
    the alias so the writer stops rejecting.
    """
    seen: dict[int, str] = {}
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    pairs: list[tuple[str, str]] = []
    for name, t in state.items():
        ptr = t.data_ptr()
        if ptr in seen:
            canonical = seen[ptr]
            # Clone + make contiguous — the destination is now genuinely
            # independent storage the safetensors writer will accept.
            out[name] = t.detach().clone().contiguous()
            pairs.append((name, canonical))
        else:
            seen[ptr] = name
            out[name] = t.detach().contiguous()
    return out, pairs


def load_and_merge(shards: list[Path]) -> "OrderedDict[str, torch.Tensor]":
    """Load every shard in order and merge into a single OrderedDict.

    Detects duplicate names across shards (a shard that repeats a name
    a previous shard already emitted) and refuses — the upstream shard
    convention guarantees each key lives in exactly one shard, and a
    duplicate would indicate a corrupted release.
    """
    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for shard in shards:
        print(f"loading shard: {shard}")
        # `safetensors.torch.load_file` returns a plain dict on CPU
        # with each tensor as a genuine `torch.Tensor` (F32 / F16 /
        # BF16 preserved). No pickle involved.
        chunk: dict[str, Any] = load_file(str(shard), device="cpu")
        for name, t in chunk.items():
            if name in merged:
                raise SystemExit(
                    f"duplicate tensor name across shards: {name!r} first seen earlier, "
                    f"repeated in {shard}"
                )
            merged[name] = t
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dir",
        type=Path,
        help="directory containing model.safetensors.index.json + shards "
        "(upstream `bosonai/higgs-audio-v3-tts-4b` layout after `hf download`)",
    )
    src.add_argument(
        "--shards",
        nargs="+",
        help="explicit glob(s) for the shard files (bypasses index.json — "
        "useful for tests / CI matrix runs); mutually exclusive with --input-dir",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .safetensors path (the Rust converter's input)",
    )
    args = ap.parse_args()

    input_dir = args.input_dir if args.input_dir else Path(args.shards[0]).parent
    shards = discover_shards(input_dir, args.shards)
    print(f"discovered {len(shards)} shard(s) under {input_dir}")

    merged = load_and_merge(shards)
    print(f"merged state dict: {len(merged)} entries across {len(shards)} shard(s)")

    # Drop non-float training-scaffold state + reject any other
    # non-float dtype (FR-EX-08 no silent fallback).
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped: list[str] = []
    for name, t in merged.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"non-tensor state entry {name!r}: {type(t)}")
        if any(name.endswith(suffix) for suffix in DROP_SUFFIXES):
            dropped.append(name)
            continue
        if t.dtype not in ALLOWED_DTYPES:
            raise SystemExit(
                f"unexpected dtype {t.dtype} for tensor {name!r}: "
                f"the safetensors pass-through path handles only "
                f"F32 / F16 / BF16 (FR-EX-08 no silent fallback). If this "
                f"upstream tensor is genuine inference state, extend "
                f"ALLOWED_DTYPES here + the Rust converter arm; if it is "
                f"training scaffold, extend DROP_SUFFIXES above."
            )
        filtered[name] = t

    # Dedupe shared storage (tied embeddings etc.) — safetensors.torch
    # refuses two names pointing at the same storage.
    deduped, shared_pairs = dedupe_shared_storage(filtered)

    # Write safetensors.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(deduped, str(args.output))

    # Emit a shared_pairs.json audit trail alongside the output.
    audit_path = Path(f"{args.output}.shared_pairs.json")
    audit_path.write_text(
        json.dumps(
            {"shared_pairs": [{"alias": a, "canonical": c} for (a, c) in shared_pairs]},
            indent=2,
        )
    )

    # Report.
    for name in dropped:
        print(f"dropped (training-scaffold state): {name}")
    for alias, canonical in shared_pairs:
        print(f"deduped shared storage: {alias} → {canonical} (cloned)")

    total_params = sum(t.numel() for t in deduped.values())
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    sha_path = Path(f"{args.output}.sha256")
    sha_path.write_text(f"{sha}  {args.output.name}\n")
    print(f"{sha}  {args.output}")
    print(
        f"tensors={len(deduped)} params={total_params} "
        f"dropped={len(dropped)} shared_pairs={len(shared_pairs)} shards={len(shards)}"
    )
    print(f"audit trail: {audit_path}")
    print(f"sha256 sidecar: {sha_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
