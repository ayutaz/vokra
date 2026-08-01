#!/usr/bin/env python3
"""Prepare a ``facebook/musicgen-medium`` checkpoint for the Vokra
``vokra-convert --model musicgen-medium`` converter.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``facebook/musicgen-medium`` release ships as a bundle:
LM decoder (~5-6 GB, potentially torch pickle ``pytorch_model.bin`` OR
sharded ``model.safetensors``) + frozen T5-base text encoder (~1 GB) +
paired EnCodec RVQ audio codec (~150 MB) + config side-cars. Total
~11.4 GB per HF cardData primary source (2026-08-01).

Vokra's Rust converter (``crates/vokra-convert/src/models/musicgen_medium.rs``)
consumes **single-file safetensors** by design — the runtime never grows
a shard-index reader or a pickle parser (NFR-DS-02 zero-dep + FR-LD-05
no pickle in runtime). This script bridges the two by handling both
distribution shapes upstream releases can take:

1. **safetensors-native (single or sharded)**: walk
   ``model.safetensors.index.json`` weight-map when present, load every
   shard, merge into a single in-memory state_dict, re-serialize as one
   safetensors — the ``moss_audio_tokenizer_prepare_checkpoint.py`` +
   ``granite_speech_prepare_checkpoint.py`` precedent.

2. **torch pickle (pytorch_model.bin)**: delegate to the shared
   ``bin_to_safetensors.py`` bridge — the SBV2 v2 / SpeechT5-HiFi-GAN /
   DeBERTa v3 large / VoxCPM-0.5B / Fun-CosyVoice3 precedent.

Uses only ``torch.load`` (BSD-3, ``weights_only=True`` — the safe path
that disallows arbitrary object construction) plus ``safetensors``
(Apache-2.0) plus ``huggingface_hub.snapshot_download`` (Apache-2.0).
No AudioCraft source is read or referenced (clean-room).

# Scale — vast.ai handoff

MusicGen-Medium is ~11.4 GB. The M1 iMac 16 GB machine cannot safely
convert this class of publish (memory ``[[feedback-large-models-on-vast-ai]]``:
Voxtral-Small-24B 48 GB confirmed swap-death at ~40 GB peak). Run this
script on vast.ai per ``docs/handoff/vast-ai-large-model-publish.md``.

# License — Meta AudioCraft weight policy

Weights ship **cc-by-nc-4.0** per HF cardData primary source; the code
layer at ``github.com/facebookresearch/audiocraft`` is MIT, but this
prep script + the downstream Vokra converter both treat the artifact
under the weight-distribution license (same posture X-Codec 2 landed
2026-07-28 = T4 tier Research-only). ``publish-one.sh
--allow-noncommercial`` gate must fire before any upload.

# NOT REFERENCED (clean-room)

- github.com/facebookresearch/audiocraft (MIT code, but AGPL-adjacent
  research-tool posture — treat as opaque for prep)

# FR-EX-08 loud-error posture

- Missing / malformed pickle → propagates torch.load's own exception.
- Missing safetensors index → fails loudly with the missing path.
- Any INT-dtype tensor (BatchNorm ``num_batches_tracked`` etc.) is
  dropped with a warn — the sibling BF16 pass-through converters all
  do this at the bridge layer since the Rust safetensors reader admits
  only F32 / F16 / BF16.

Usage
-----

::

    uv run tools/parity/musicgen_medium_prepare_checkpoint.py \\
        --hf-repo facebook/musicgen-medium \\
        --output ./musicgen-medium.safetensors

Or point at an already-downloaded checkpoint directory::

    uv run tools/parity/musicgen_medium_prepare_checkpoint.py \\
        --checkpoint-dir /path/to/musicgen-medium \\
        --output ./musicgen-medium.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO = "facebook/musicgen-medium"


def _log(msg: str) -> None:
    print(f"[musicgen-medium-prep] {msg}", file=sys.stderr, flush=True)


def _dtype_ok_for_vokra(dtype_str: str) -> bool:
    """The Vokra safetensors reader admits only F32 / F16 / BF16.

    Any INT dtype (BatchNorm ``num_batches_tracked``, position ids, etc.)
    must be filtered here — the Rust reader rejects unknown dtypes at
    parse time, so silently passing them through would fail the whole
    conversion. Mirrors the sibling
    ``naturalspeech3_facodec_prepare_checkpoint.py`` +
    ``yue_bundle_prepare_checkpoint.py`` INT-strip posture.
    """
    return dtype_str.upper() in {"F32", "F16", "BF16"}


def _resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    """Return a local checkpoint directory, downloading from HF if needed."""
    if args.checkpoint_dir is not None:
        d = Path(args.checkpoint_dir)
        if not d.is_dir():
            raise SystemExit(f"--checkpoint-dir does not exist or is not a directory: {d}")
        return d

    # Delayed import: huggingface_hub is a heavyweight dep only pulled
    # in on the download path.
    from huggingface_hub import snapshot_download  # type: ignore[import]

    _log(f"downloading {args.hf_repo} to local snapshot cache ...")
    local = snapshot_download(
        repo_id=args.hf_repo,
        # Skip .md / .png / .gitattributes etc. — pull only what the
        # converter walks (weight files + shard index + configs).
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.json",
            "*.yaml",
        ],
    )
    return Path(local)


def _load_from_safetensors_shards(ckpt_dir: Path) -> dict[str, Any]:
    """Walk model.safetensors.index.json (if present) or single-file
    model.safetensors, merge into an in-memory state_dict."""
    # Delayed import: safetensors is only needed on this path.
    from safetensors import safe_open  # type: ignore[import]

    index_path = ckpt_dir / "model.safetensors.index.json"
    single_path = ckpt_dir / "model.safetensors"

    state: dict[str, Any] = {}

    if index_path.is_file():
        _log(f"walking sharded safetensors index: {index_path.name}")
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise SystemExit(
                f"malformed weight-map at {index_path}: missing 'weight_map' dict"
            )
        # De-dup the shard set — usually 2-8 shards for models of this scale.
        shards = sorted({shard for shard in weight_map.values()})
        for shard_name in shards:
            shard_path = ckpt_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(f"shard listed in index but missing on disk: {shard_path}")
            _log(f"loading shard {shard_name} ({shard_path.stat().st_size:,} bytes)")
            with safe_open(shard_path, framework="numpy") as reader:
                for key in reader.keys():
                    state[key] = reader.get_tensor(key)
        return state

    if single_path.is_file():
        _log(f"loading single-file safetensors: {single_path.name}")
        with safe_open(single_path, framework="numpy") as reader:
            for key in reader.keys():
                state[key] = reader.get_tensor(key)
        return state

    return state  # empty = caller falls through to pickle path


def _load_from_pytorch_bin(ckpt_dir: Path) -> dict[str, Any]:
    """Fall back to the pytorch pickle bridge (``bin_to_safetensors.py``
    posture) — MusicGen releases may ship .bin instead of .safetensors."""
    # Delayed imports: torch is only needed on this path.
    import torch  # type: ignore[import]

    bin_path = ckpt_dir / "pytorch_model.bin"
    if not bin_path.is_file():
        return {}

    _log(f"loading torch pickle: {bin_path.name} ({bin_path.stat().st_size:,} bytes)")
    # weights_only=True is the safe path — refuses arbitrary object
    # construction (no code execution on load, no arbitrary imports).
    loaded = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise SystemExit(
            f"pytorch pickle at {bin_path} did not deserialize to a state_dict"
        )
    # Move to numpy so the downstream save_file path is torch-free.
    state: dict[str, Any] = {}
    for k, v in loaded.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        state[k] = v
    return state


def _save_state(state: dict[str, Any], out_path: Path) -> None:
    """Serialize the merged in-memory state to a single safetensors file
    with INT-dtype stripping."""
    # Delayed import: safetensors save is only needed here.
    from safetensors.numpy import save_file  # type: ignore[import]

    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for name, tensor in state.items():
        # numpy arrays have .dtype.name = "float32" / "float16" / "int64" etc.
        # We normalize to the safetensors dtype tags (F32 / F16 / BF16 / …).
        dtype_name = getattr(getattr(tensor, "dtype", None), "name", "unknown")
        norm = {
            "float32": "F32",
            "float16": "F16",
            "bfloat16": "BF16",
        }.get(dtype_name, dtype_name.upper())
        if _dtype_ok_for_vokra(norm):
            kept[name] = tensor
        else:
            dropped.append(f"{name} (dtype={dtype_name})")

    if dropped:
        _log(f"dropped {len(dropped)} non-float tensors (INT/etc):")
        for entry in dropped[:20]:
            _log(f"  - {entry}")
        if len(dropped) > 20:
            _log(f"  ... and {len(dropped) - 20} more")

    _log(f"writing {len(kept):,} float tensors to {out_path}")
    save_file(kept, str(out_path))
    _log(f"done: {out_path.stat().st_size:,} bytes on disk")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a facebook/musicgen-medium checkpoint for the Vokra "
            "vokra-convert --model musicgen-medium converter (safetensors-only)."
        )
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"HF repo id to snapshot_download from (default: {DEFAULT_HF_REPO})",
    )
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to an already-downloaded local checkpoint directory (overrides --hf-repo)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output safetensors path (e.g. ./musicgen-medium.safetensors)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing file: {out_path} (remove it first)"
        )

    ckpt_dir = _resolve_checkpoint_dir(args)
    _log(f"checkpoint dir: {ckpt_dir}")

    # Prefer safetensors when the release ships them. Fall back to
    # pytorch_model.bin only if safetensors are absent — never merge
    # both (a state key present in both would silently favour one).
    state = _load_from_safetensors_shards(ckpt_dir)
    if not state:
        _log("no safetensors found, falling back to pytorch_model.bin")
        state = _load_from_pytorch_bin(ckpt_dir)

    if not state:
        raise SystemExit(
            "no loadable checkpoint found: neither model.safetensors(.index.json) "
            "nor pytorch_model.bin exist in the checkpoint directory"
        )

    _save_state(state, out_path)


if __name__ == "__main__":
    main()
