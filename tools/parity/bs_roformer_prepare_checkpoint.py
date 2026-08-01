#!/usr/bin/env python3
"""Prepare a BS-Roformer / Mel-Band Roformer checkpoint for the Vokra
``vokra-convert --model bs-roformer`` converter.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

BS-Roformer trainers ship checkpoints in **PyTorch pickle** format
(``.ckpt`` — the Lightning training loop default, or ``.pth`` — the
raw ``torch.save`` posture some UVR-community forks use). The
Lucidrains reference (``github.com/lucidrains/BS-RoFormer``) exposes
the model as an ``nn.Module``; downstream trainers wrap it in
``pytorch_lightning.LightningModule`` and ship a `.ckpt` with the
wrapped ``state_dict``. Third-party mirror
``chenmozhijin/BSRoformer-GGUF`` aggregates converted GGUFs across
trainers (which we do not consume — Vokra converts to its own GGUF
namespace with ``vokra.*`` metadata chunks that the third-party GGUFs
lack); the underlying `.ckpt` sources are what this bridge accepts.

Vokra's Rust converter (``crates/vokra-convert/src/models/bs_roformer.rs``)
consumes **single-file safetensors** by design — the runtime never
grows a pickle parser (NFR-DS-02 zero-dep + FR-LD-05 no pickle in
runtime). This script bridges the two by handling both distribution
shapes upstream trainers can take:

1. **torch pickle (`.ckpt` or `.pth`)**: delegate to the shared
   ``bin_to_safetensors.py`` bridge — the SBV2 v2 / SpeechT5-HiFi-GAN /
   DeBERTa v3 large / VoxCPM-0.5B / Fun-CosyVoice3 / MusicGen-Medium /
   MusicGen-Large precedent. Handles the LightningModule wrapper
   unwrap (``["state_dict"]`` walk) same as
   ``sepformer_prepare_checkpoint.py``.

2. **safetensors-native (single or sharded)**: walk
   ``model.safetensors.index.json`` weight-map when present, load
   every shard, merge into a single in-memory state_dict, re-
   serialize as one safetensors — the ``moss_audio_tokenizer_prepare_
   checkpoint.py`` + ``musicgen_large_prepare_checkpoint.py``
   precedent.

Uses only ``torch.load`` (BSD-3, ``weights_only=True`` — the safe
path that disallows arbitrary object construction) plus
``safetensors`` (Apache-2.0). No AudioCraft / no HF snapshot download
by default (the third-party mirror is a heterogeneous aggregator, so
users must supply their own `.ckpt` path per their license posture —
the downloader path is not wired here for clarity of provenance:
whatever this script emits carries the caller's own choice of
checkpoint, not a silent snapshot of a random mirror file).

# License provenance — user-supplied only

**This prep script deliberately does NOT include an HF snapshot-
download path** (unlike the sibling ``musicgen_*_prepare_checkpoint.py``
scripts, which know the canonical Meta AudioCraft repo). BS-Roformer
weights come from many trainers under mixed licenses (some GPL-3.0
under Ultimate-Vocal-Remover / MDX-Net-community derivatives, some
CC-BY-NC-4.0 under MoisesDB / MusDB fine-tunes, most no explicit
license under hobbyist releases). The caller must supply the
`.ckpt` / `.pth` / `.safetensors` path themselves and pass the
corresponding SPDX id via ``vokra-convert --license <spdx>`` at
conversion time.

Refer to ``docs/license-audit.md`` §3.1 "BS-Roformer (upstream 未確定)"
sign-off row for the fail-closed publish gate: the row is blank
pending an owner ADR selecting a specific checkpoint + license, so
the publish path is **completely blocked** at
``scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS``
(``UNKNOWN_REPO`` fail-closed) regardless of what the caller supplies
here.

# NOT REFERENCED (clean-room)

- ``github.com/lucidrains/BS-RoFormer`` (MIT reference code — the
  architecture is independently re-implementable per whisper.cpp 型
  self re-implementation, CLAUDE.md 設計判断 4)
- ``github.com/Anjok07/ultimatevocalremovergui`` (GPL-3.0 community
  trainer — one of many; the license is trainer-specific, this
  script treats every input as opaque)

# FR-EX-08 loud-error posture

- Missing / malformed pickle → propagates torch.load's own exception.
- Missing safetensors → fails loudly with the missing path.
- Any INT-dtype tensor (BatchNorm ``num_batches_tracked``, position
  ids, etc.) is dropped with a warn — the sibling BF16 pass-through
  converters all do this at the bridge layer since the Rust
  safetensors reader admits only F32 / F16 / BF16.

# Scale — usually local-safe, occasionally vast.ai

BS-Roformer checkpoints range from ~150 MB (Mel-Band variants) to
~4-5 GB (full BS-Roformer with high band-count). The 4.68 GB
flagship class sits just under the M1 iMac 16 GB comfortable-local-
convert threshold — per memory ``[[feedback-large-models-on-vast-ai]]``,
Mel-Band ~150MB variants convert locally without concern; the
top-of-range 4-5 GB variants are safer on vast.ai for a 16 GB box.

Usage
-----

::

    uv run tools/parity/bs_roformer_prepare_checkpoint.py \\
        --checkpoint /path/to/your/bs_roformer_vocals.ckpt \\
        --output ./bs-roformer-vocals.safetensors

Or point at a directory holding a safetensors bundle::

    uv run tools/parity/bs_roformer_prepare_checkpoint.py \\
        --checkpoint-dir /path/to/bs-roformer-dir \\
        --output ./bs-roformer.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    print(f"[bs-roformer-prep] {msg}", file=sys.stderr, flush=True)


def _dtype_ok_for_vokra(dtype_str: str) -> bool:
    """The Vokra safetensors reader admits only F32 / F16 / BF16.

    Any INT dtype (BatchNorm ``num_batches_tracked``, position ids,
    etc.) must be filtered here — the Rust reader rejects unknown
    dtypes at parse time, so silently passing them through would fail
    the whole conversion. Mirrors the sibling
    ``musicgen_large_prepare_checkpoint.py`` +
    ``naturalspeech3_facodec_prepare_checkpoint.py`` +
    ``yue_bundle_prepare_checkpoint.py`` INT-strip posture.
    """
    return dtype_str.upper() in {"F32", "F16", "BF16"}


def _unwrap_lightning(loaded: Any) -> dict[str, Any]:
    """Unwrap a Lightning-style checkpoint if present.

    The Lucidrains reference ``BSRoformer`` is a plain ``nn.Module``;
    downstream trainers commonly wrap it in a
    ``pytorch_lightning.LightningModule`` and ship the state under a
    ``state_dict`` (or ``model_state_dict`` / ``module``) key. Unwrap
    silently if the wrapper is present — the sibling
    ``sepformer_prepare_checkpoint.py`` precedent applies.
    """
    if not isinstance(loaded, dict):
        raise SystemExit(
            f"pytorch pickle did not deserialize to a dict "
            f"(type = {type(loaded).__name__})"
        )
    for wrapper in ("state_dict", "model_state_dict", "model", "module"):
        inner = loaded.get(wrapper)
        if isinstance(inner, dict) and inner:
            sample = next(iter(inner.values()), None)
            if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                _log(f"unwrapped Lightning-style wrapper ['{wrapper}']")
                return inner
    return loaded


def _load_from_ckpt(ckpt_path: Path) -> dict[str, Any]:
    """Load a torch pickle checkpoint via the safe path.

    ``weights_only=True`` refuses arbitrary object construction — no
    code execution on load, no arbitrary imports (torch 2.6+ default,
    but we set it explicitly here so a caller on an older torch still
    gets the safe posture).
    """
    # Delayed import: torch is only needed on this path.
    import torch  # type: ignore[import]

    _log(f"loading torch pickle: {ckpt_path.name} ({ckpt_path.stat().st_size:,} bytes)")
    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"torch.load({ckpt_path!s}, weights_only=True) failed: {exc}"
        )
    unwrapped = _unwrap_lightning(raw)
    # Move to numpy so the downstream save_file path is torch-free.
    state: dict[str, Any] = {}
    for k, v in unwrapped.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        state[k] = v
    return state


def _load_from_safetensors_dir(ckpt_dir: Path) -> dict[str, Any]:
    """Walk ``model.safetensors.index.json`` (sharded) or a single-
    file ``model.safetensors`` in the directory."""
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
        shards = sorted({shard for shard in weight_map.values()})
        for shard_name in shards:
            shard_path = ckpt_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(
                    f"shard listed in index but missing on disk: {shard_path}"
                )
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

    return state  # empty = caller falls through


def _load_from_safetensors_file(st_path: Path) -> dict[str, Any]:
    """Load a single safetensors file directly."""
    from safetensors import safe_open  # type: ignore[import]

    _log(f"loading safetensors: {st_path.name} ({st_path.stat().st_size:,} bytes)")
    state: dict[str, Any] = {}
    with safe_open(str(st_path), framework="numpy") as reader:
        for key in reader.keys():
            state[key] = reader.get_tensor(key)
    return state


def _save_state(state: dict[str, Any], out_path: Path) -> None:
    """Serialize the merged in-memory state to a single safetensors
    file with INT-dtype stripping."""
    # Delayed import: safetensors save is only needed here.
    from safetensors.numpy import save_file  # type: ignore[import]

    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for name, tensor in state.items():
        # numpy arrays have .dtype.name = "float32" / "float16" /
        # "int64" etc. We normalize to the safetensors dtype tags.
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

    if not kept:
        raise SystemExit(
            "no F32 / F16 / BF16 tensors survived the dtype filter; "
            "either the checkpoint is malformed or every tensor is an "
            "integer artefact (BatchNorm counter etc.) — the Vokra "
            "safetensors reader admits only F32 / F16 / BF16 so we "
            "refuse to emit an empty safetensors."
        )

    _log(f"writing {len(kept):,} float tensors to {out_path}")
    save_file(kept, str(out_path))
    _log(f"done: {out_path.stat().st_size:,} bytes on disk")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a BS-Roformer / Mel-Band Roformer checkpoint for the "
            "Vokra vokra-convert --model bs-roformer converter (safetensors-only)."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a single .ckpt / .pth / .safetensors file. Torch pickle "
            "(.ckpt / .pth) is unwrapped through the Lightning-style "
            "state_dict walk; safetensors is loaded directly."
        ),
    )
    src.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Path to a directory holding either a sharded safetensors bundle "
            "(model.safetensors.index.json + shards) or a single-file "
            "model.safetensors."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output safetensors path (e.g. ./bs-roformer.safetensors)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing file: {out_path} (remove it first)"
        )

    state: dict[str, Any] = {}
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_file():
            raise SystemExit(f"--checkpoint does not exist or is not a file: {ckpt_path}")
        suffix = ckpt_path.suffix.lower()
        if suffix in (".ckpt", ".pth", ".bin"):
            state = _load_from_ckpt(ckpt_path)
        elif suffix == ".safetensors":
            state = _load_from_safetensors_file(ckpt_path)
        else:
            raise SystemExit(
                f"unrecognized checkpoint suffix {suffix!r} (expected .ckpt / "
                f".pth / .bin / .safetensors); use --checkpoint-dir for a "
                f"directory bundle"
            )
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_dir():
            raise SystemExit(
                f"--checkpoint-dir does not exist or is not a directory: {ckpt_dir}"
            )
        state = _load_from_safetensors_dir(ckpt_dir)

    if not state:
        raise SystemExit(
            "no loadable checkpoint found: neither a single .ckpt / .pth / "
            ".safetensors file nor a safetensors bundle directory produced a "
            "non-empty state_dict"
        )

    _save_state(state, out_path)

    # Provenance reminder — the whole reason this script exists is that
    # BS-Roformer weight provenance cannot be machine-checked, so we
    # remind the caller loudly at end-of-run to supply the correct
    # SPDX id at conversion time.
    _log("")
    _log("=" * 70)
    _log("REMINDER — supply --license <spdx> at vokra-convert time")
    _log("=" * 70)
    _log(
        "BS-Roformer weights ship under mixed licenses (some GPL-3.0 UVR "
        "derivatives, some CC-BY-NC-4.0 MoisesDB fine-tunes, most no "
        "explicit license). The Vokra converter defaults to "
        "LicenseClass::RedistributionForbidden (fail-closed publish gate). "
        "Supply --license <spdx> at conversion time to record the specific "
        "license for YOUR checkpoint:"
    )
    _log("")
    _log(f"    vokra-convert --model bs-roformer --license <spdx> \\")
    _log(f"        --input {out_path} --output <output.gguf>")
    _log("")
    _log(
        "See docs/license-audit.md §3.1 'BS-Roformer (upstream 未確定)' for "
        "the publish blocker and owner ADR requirements."
    )


if __name__ == "__main__":
    main()
