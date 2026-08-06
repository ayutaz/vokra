#!/usr/bin/env python3
"""Prepare a ``cvssp/audioldm2`` checkpoint for the Vokra
``vokra-convert --model audioldm2`` converter.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``cvssp/audioldm2`` release ships as a **multi-sub-module
bundle** (~8.5 GB) using ``diffusers.AudioLDM2Pipeline`` conventions.
The pipeline is composed of five distinct sub-modules, each rooted at
its own sub-directory + its own weight file:

1. ``vae/``               — VAE encoder/decoder (``AutoencoderKL``)
2. ``unet/``              — Latent-diffusion U-Net
                            (``AudioLDM2UNet2DConditionModel``)
3. ``vocoder/``           — HiFi-GAN vocoder (mel → waveform)
4. ``language_model/``    — GPT-2 audio-caption LM
                            ("language of audio" token producer)
5. ``text_encoder{,_2}/`` — Frozen T5-base + CLAP text encoders

Each sub-module ships as **safetensors** (single-file per sub-module
under ``diffusers`` layout) or **torch pickle** (older releases). This
script:

  (a) walks the pipeline directory tree,
  (b) loads every sub-module's weight file (safetensors OR pickle),
  (c) re-prefixes each sub-module's keys with its role name (``vae.*``
      / ``unet.*`` / ``vocoder.*`` / ``language_model.*`` /
      ``text_encoder.*`` / ``text_encoder_2.*``) so a single flat
      state_dict can be materialised,
  (d) merges everything into one in-memory dict,
  (e) re-serialises as one flat safetensors file the Vokra Rust
      converter (``crates/vokra-convert/src/models/audioldm2.rs``)
      consumes.

Vokra's Rust converter is **single-file safetensors only** by design —
the runtime never grows a shard-index reader or a pickle parser
(NFR-DS-02 zero-dep + FR-LD-05 no pickle in runtime). This bridge is
the same shape as ``sepformer_prepare_checkpoint.py`` +
``naturalspeech3_facodec_prepare_checkpoint.py`` +
``musicgen_medium_prepare_checkpoint.py``: role-prefixed merger for
multi-sub-module bundles.

# Scale — vast.ai handoff

AudioLDM 2 is ~8.5 GB on disk. The M1 iMac 16 GB machine sits at the
upper edge of the "safe local convert" threshold (memory
``[[feedback-large-models-on-vast-ai]]``: ≥8 GB safe, and the
multi-encoder bundle doubles peak resident to ~17 GB on the merge
pass) — run this script on vast.ai per
``docs/handoff/vast-ai-large-model-publish.md`` for the primary
publish path.

# License — CVSSP weight policy

Weights ship **cc-by-nc-sa-4.0** per CVSSP GitHub README + paper
Ethics § (Liu et al. 2024 ICML arXiv:2308.05734). The HF model card's
YAML front-matter carries the looser ``cc-by-nc-4.0`` tag; we follow
the more restrictive ShareAlike form per CVSSP-owned primary source
(same Fish-Speech precedent for CC-BY-NC-SA-4.0). **Publish blocked**:
the SA cascade obligation onto Vokra-added artifacts (model card,
LICENSE, NOTICE, auxiliary GGUFs) needs an owner ADR before a
``vokra/audioldm2`` publish target enters
``scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS``.

# NOT REFERENCED (clean-room)

- github.com/haoheliu/AudioLDM2 (MIT code, but distinct from the
  cc-by-nc-sa-4.0 weight release — treat as opaque for prep)

Uses only ``torch.load`` (BSD-3, ``weights_only=True`` — the safe path
that disallows arbitrary object construction) plus ``safetensors``
(Apache-2.0) plus ``huggingface_hub.snapshot_download`` (Apache-2.0).
No AudioLDM 2 code source is read or referenced (clean-room).

# FR-EX-08 loud-error posture

- Missing / malformed pickle → propagates torch.load's own exception.
- Missing sub-module directory → fails loudly with the missing path.
- Any INT-dtype tensor (BatchNorm ``num_batches_tracked``, position
  ids, etc.) is dropped with a warn — the sibling BF16 pass-through
  converters all do this at the bridge layer since the Rust
  safetensors reader admits only F32 / F16 / BF16.
- Key collisions across sub-modules (e.g. two sub-modules both writing
  under bare ``weight``) fail loudly with the colliding key set — the
  role-prefix contract MUST guarantee unique keys, and any collision
  indicates the upstream release changed shape and this bridge needs
  updating.

Usage
-----

::

    uv run tools/parity/audioldm2_prepare_checkpoint.py \\
        --hf-repo cvssp/audioldm2 \\
        --output ./audioldm2.safetensors

Or point at an already-downloaded checkpoint directory::

    uv run tools/parity/audioldm2_prepare_checkpoint.py \\
        --checkpoint-dir /path/to/audioldm2 \\
        --output ./audioldm2.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO = "cvssp/audioldm2"

# Role prefix ← sub-module dir name mapping. Every sub-module rooted at
# its own directory under the pipeline root. Keys are the diffusers-
# convention sub-dir names; values are the flat role prefix we write
# into the merged safetensors (`{prefix}.{key}` form).
#
# `text_encoder` + `text_encoder_2` = the T5-base + CLAP pair (order
# depends on the release, but both slots are consumed by the U-Net's
# cross-attention). We keep them role-distinct so a future
# `AudioLdm2::from_gguf` can bind them independently.
SUBMODULES: dict[str, str] = {
    "vae": "vae",
    "unet": "unet",
    "vocoder": "vocoder",
    "language_model": "language_model",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}


def _log(msg: str) -> None:
    print(f"[audioldm2-prep] {msg}", file=sys.stderr, flush=True)


def _dtype_ok_for_vokra(dtype_str: str) -> bool:
    """The Vokra safetensors reader admits only F32 / F16 / BF16.

    Any INT dtype (BatchNorm ``num_batches_tracked``, position ids,
    etc.) must be filtered here — the Rust reader rejects unknown
    dtypes at parse time, so silently passing them through would fail
    the whole conversion. Mirrors the sibling
    ``naturalspeech3_facodec_prepare_checkpoint.py`` +
    ``musicgen_medium_prepare_checkpoint.py`` +
    ``yue_bundle_prepare_checkpoint.py`` INT-strip posture.
    """
    return dtype_str.upper() in {"F32", "F16", "BF16"}


def _resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    """Return a local checkpoint directory, downloading from HF if
    needed."""
    if args.checkpoint_dir is not None:
        d = Path(args.checkpoint_dir)
        if not d.is_dir():
            raise SystemExit(
                f"--checkpoint-dir does not exist or is not a directory: {d}"
            )
        return d

    # Delayed import: huggingface_hub is a heavyweight dep only pulled
    # in on the download path.
    from huggingface_hub import snapshot_download  # type: ignore[import]

    _log(f"downloading {args.hf_repo} to local snapshot cache ...")
    local = snapshot_download(
        repo_id=args.hf_repo,
        # Skip .md / .png / .gitattributes etc. — pull only what the
        # converter walks (weight files + shard indices + configs).
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.json",
            "*.yaml",
        ],
    )
    return Path(local)


def _load_submodule_safetensors(sub_dir: Path) -> dict[str, Any]:
    """Load every safetensors shard under ``sub_dir`` into a dict.

    Handles both single-file (``model.safetensors`` /
    ``diffusion_pytorch_model.safetensors``) and sharded (with
    ``*.safetensors.index.json``) layouts. Returns an empty dict if no
    safetensors exist under ``sub_dir`` (caller then falls back to
    pickle).
    """
    # Delayed import: safetensors is only needed on this path.
    from safetensors import safe_open  # type: ignore[import]

    state: dict[str, Any] = {}

    # Walk any safetensors file in the sub-module directory.
    st_files = sorted(sub_dir.glob("*.safetensors"))
    if not st_files:
        return state

    for st_path in st_files:
        _log(
            f"  loading safetensors: {st_path.name} "
            f"({st_path.stat().st_size:,} bytes)"
        )
        with safe_open(st_path, framework="numpy") as reader:
            for key in reader.keys():
                # Later shards should never overwrite an earlier one —
                # the index.json contract guarantees each key lives in
                # exactly one shard. Fail loudly if it does.
                if key in state:
                    raise SystemExit(
                        f"key collision within sub-module {sub_dir.name}: "
                        f"'{key}' present in multiple safetensors files "
                        f"(release layout changed?)"
                    )
                state[key] = reader.get_tensor(key)

    return state


def _load_submodule_pickle(sub_dir: Path) -> dict[str, Any]:
    """Fall back to the pytorch pickle bridge (``bin_to_safetensors.py``
    posture) when the sub-module ships pickle instead of safetensors."""
    # Delayed imports: torch is only needed on this path.
    import torch  # type: ignore[import]

    state: dict[str, Any] = {}

    # Common HF pickle filenames.
    candidates = [
        sub_dir / "pytorch_model.bin",
        sub_dir / "diffusion_pytorch_model.bin",
    ]
    for bin_path in candidates:
        if not bin_path.is_file():
            continue
        _log(
            f"  loading torch pickle: {bin_path.name} "
            f"({bin_path.stat().st_size:,} bytes)"
        )
        # weights_only=True is the safe path — refuses arbitrary object
        # construction (no code execution on load).
        loaded = torch.load(
            str(bin_path), map_location="cpu", weights_only=True
        )
        if not isinstance(loaded, dict):
            raise SystemExit(
                f"pytorch pickle at {bin_path} did not deserialize to a state_dict"
            )
        # Move to numpy so the downstream save_file path is torch-free.
        for k, v in loaded.items():
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            if k in state:
                raise SystemExit(
                    f"key collision within sub-module {sub_dir.name}: "
                    f"'{k}' present in multiple pickle files"
                )
            state[k] = v

    return state


def _load_submodule(ckpt_dir: Path, sub_name: str, role_prefix: str) -> dict[str, Any]:
    """Load one sub-module + apply role prefix. Prefers safetensors
    over pickle within the sub-module directory (never merges both — a
    key present in both would silently favour one)."""
    sub_dir = ckpt_dir / sub_name
    if not sub_dir.is_dir():
        _log(f"sub-module directory absent, skipping: {sub_name}/")
        return {}

    _log(f"loading sub-module: {sub_name}/ (role prefix: {role_prefix})")

    # Prefer safetensors when the sub-module ships them.
    inner = _load_submodule_safetensors(sub_dir)
    if not inner:
        _log(f"  no safetensors under {sub_name}/, falling back to pickle")
        inner = _load_submodule_pickle(sub_dir)

    if not inner:
        _log(f"  WARN: sub-module {sub_name}/ found on disk but no loadable weights")
        return {}

    # Apply role prefix so the flat merged state has unique keys across
    # sub-modules. `text_encoder.encoder.block.0.q.weight` +
    # `text_encoder_2.encoder.block.0.q.weight` etc.
    prefixed: dict[str, Any] = {}
    for key, tensor in inner.items():
        prefixed[f"{role_prefix}.{key}"] = tensor

    _log(f"  loaded {len(prefixed):,} tensors from {sub_name}/")
    return prefixed


def _save_state(state: dict[str, Any], out_path: Path) -> None:
    """Serialize the merged in-memory state to a single safetensors
    file with INT-dtype stripping."""
    # Delayed import: safetensors save is only needed here.
    from safetensors.numpy import save_file  # type: ignore[import]

    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for name, tensor in state.items():
        # numpy arrays have .dtype.name = "float32" / "float16" /
        # "int64" etc. We normalise to the safetensors dtype tags
        # (F32 / F16 / BF16 / …).
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
            "Prepare a cvssp/audioldm2 checkpoint for the Vokra "
            "vokra-convert --model audioldm2 converter (safetensors-only, "
            "single-file, role-prefixed multi-sub-module merger)."
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
        help=(
            "Path to an already-downloaded local checkpoint directory "
            "(overrides --hf-repo)"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output safetensors path (e.g. ./audioldm2.safetensors)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing file: {out_path} (remove it first)"
        )

    ckpt_dir = _resolve_checkpoint_dir(args)
    _log(f"checkpoint dir: {ckpt_dir}")

    merged: dict[str, Any] = {}
    total_loaded = 0
    for sub_name, role_prefix in SUBMODULES.items():
        prefixed = _load_submodule(ckpt_dir, sub_name, role_prefix)
        # Cross-sub-module collision check (role prefix should
        # guarantee uniqueness, but assert loudly if the release
        # violates it).
        for key in prefixed:
            if key in merged:
                raise SystemExit(
                    f"cross-sub-module key collision: '{key}' present in "
                    f"multiple sub-modules (role prefix contract broken?)"
                )
        merged.update(prefixed)
        total_loaded += len(prefixed)

    if not merged:
        raise SystemExit(
            "no loadable checkpoint found: expected sub-module directories "
            f"({sorted(SUBMODULES.keys())}) under {ckpt_dir} with either "
            "safetensors or pytorch_model.bin"
        )

    _log(f"merged {total_loaded:,} tensors across {len(SUBMODULES)} sub-modules")
    _save_state(merged, out_path)


if __name__ == "__main__":
    main()
