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
      would consume only after its authenticated bundle binder is enabled;
      the current converter is intentionally BLOCKED.

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
- Any INT/bool tensor is rejected. Dropping it would make the fixed
  component and sidecar contract omission-prone, so this preparer fails
  closed instead of stripping inference metadata.
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
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO = "cvssp/audioldm2"
DEFAULT_REVISION = "c8e7e189d324425c05c4c2f81214041ef4107983"
REQUIRED_TREE = {
    ".gitattributes", "README.md", "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "language_model/config.json", "language_model/model.safetensors",
    "language_model/pytorch_model.bin",
    "projection_model/config.json", "projection_model/diffusion_pytorch_model.bin",
    "projection_model/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model.safetensors", "text_encoder/pytorch_model.bin",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    "text_encoder_2/pytorch_model.bin", "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json", "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json", "tokenizer_2/tokenizer_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.bin",
    "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json", "vocoder/model.safetensors",
    "vocoder/pytorch_model.bin",
}

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
    "projection_model": "projection_model",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}


def _log(msg: str) -> None:
    print(f"[audioldm2-prep] {msg}", file=sys.stderr, flush=True)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _lfs_pointer_sha1(sha256: str, size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()
    return _git_blob_sha1(pointer)


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
    if args.hf_repo != DEFAULT_HF_REPO:
        raise SystemExit(
            f"audioldm2-prep: BLOCKED: only fixed repository {DEFAULT_HF_REPO} is allowed"
        )
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
        revision=DEFAULT_REVISION,
        # Skip .md / .png / .gitattributes etc. — pull only what the
        # converter walks (weight files + shard indices + configs).
        allow_patterns=["*"],
    )
    return Path(local)


def _validate_fixed_bundle(root: Path) -> None:
    """Require the complete, exact official tree before reading weights."""
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise SystemExit(f"audioldm2-prep: BLOCKED: symlinks are not allowed: {symlinks[:3]}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.relative_to(root).as_posix()
        not in {".vokra-source-revision", ".vokra-server-tree.json"}
    }
    missing = sorted(REQUIRED_TREE - actual)
    extra = sorted(actual - REQUIRED_TREE)
    if missing or extra:
        raise SystemExit(
            "audioldm2-prep: BLOCKED: snapshot tree is not the fixed official "
            f"{DEFAULT_HF_REPO}@{DEFAULT_REVISION}; missing={missing[:4]} "
            f"extra={extra[:4]}"
        )
    packet_path = root / ".vokra-server-tree.json"
    if not packet_path.is_file():
        raise SystemExit(
            "audioldm2-prep: BLOCKED: authoritative server-tree packet is required; "
            "a self-created revision marker is not source/model authentication"
        )
    try:
        packet = json.loads(packet_path.read_text())
        rows = packet["files"]
        if (
            packet.get("repository") != DEFAULT_HF_REPO
            or packet.get("revision") != DEFAULT_REVISION
            or packet.get("resolved_revision") != DEFAULT_REVISION
            or not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)
            or {row["path"] for row in rows} != REQUIRED_TREE
            or any(
                not isinstance(row.get("git_blob_sha1"), str)
                or len(row["git_blob_sha1"]) != 40
                or any(char not in "0123456789abcdef" for char in row["git_blob_sha1"])
                for row in rows
            )
        ):
            raise ValueError
        expected = {row["path"]: row for row in rows}
        for relative in REQUIRED_TREE:
            data = (root / relative).read_bytes()
            row = expected[relative]
            if row.get("lfs_sha256") is not None:
                sha = row["lfs_sha256"]
                size = row.get("lfs_size")
                if not isinstance(sha, str) or not isinstance(size, int) or len(sha) != 64 or size != len(data) or hashlib.sha256(data).hexdigest() != sha or row.get("git_blob_sha1") != _lfs_pointer_sha1(sha, size):
                    raise ValueError(relative)
            elif row.get("git_blob_sha1") != _git_blob_sha1(data):
                raise ValueError(relative)
    except (OSError, KeyError, TypeError, AttributeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"audioldm2-prep: BLOCKED: invalid server-tree identity packet: {exc}")


def _authenticated_tree(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        }
        for relative in sorted(REQUIRED_TREE)
    ]


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
        raise SystemExit(f"audioldm2-prep: required sub-module is absent: {sub_name}/")

    _log(f"loading sub-module: {sub_name}/ (role prefix: {role_prefix})")

    # Prefer safetensors when the sub-module ships them.
    inner = _load_submodule_safetensors(sub_dir)
    if not inner:
        raise SystemExit(
            f"audioldm2-prep: required safetensors are absent under {sub_name}/; "
            "pickle fallback is disabled by the fixed bundle contract"
        )

    if not inner:
        raise SystemExit(f"audioldm2-prep: no loadable weights under {sub_name}/")

    # Apply role prefix so the flat merged state has unique keys across
    # sub-modules. `text_encoder.encoder.block.0.q.weight` +
    # `text_encoder_2.encoder.block.0.q.weight` etc.
    prefixed: dict[str, Any] = {}
    for key, tensor in inner.items():
        prefixed[f"{role_prefix}.{key}"] = tensor

    _log(f"  loaded {len(prefixed):,} tensors from {sub_name}/")
    return prefixed


def _save_state(state: dict[str, Any], out_path: Path) -> None:
    """Serialize only after every source tensor passes the fixed dtype contract."""
    # Delayed import: safetensors save is only needed here.
    from safetensors.numpy import save_file  # type: ignore[import]

    kept: dict[str, Any] = {}
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
            raise SystemExit(
                f"audioldm2-prep: BLOCKED: tensor {name} has non-floating dtype "
                f"{dtype_name}; omission/stripping is forbidden"
            )

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
        required=False,
        help="Output safetensors path (e.g. ./audioldm2.safetensors)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the local negative dtype-contract self-test",
    )
    args = parser.parse_args()

    if args.self_test:
        assert _dtype_ok_for_vokra("F32")
        assert not _dtype_ok_for_vokra("I64")
        print("audioldm2-prep --self-test: OK")
        return

    if args.output is None:
        parser.error("--output is required unless --self-test is selected")

    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(
            f"refusing to overwrite existing file: {out_path} (remove it first)"
        )

    ckpt_dir = _resolve_checkpoint_dir(args)
    _log(f"checkpoint dir: {ckpt_dir}")
    _validate_fixed_bundle(ckpt_dir)

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

    _log(f"merged {total_loaded:,} tensors across {len(SUBMODULES)} sub-modules")
    _save_state(merged, out_path)
    manifest = {
        "schema": "vokra.audioldm2.bundle.v1",
        "status": "SNAPSHOT_TREE_AUTHENTICATED_COMPONENTS_STAGED",
        "source_repo": DEFAULT_HF_REPO,
        "source_revision": DEFAULT_REVISION,
        "tree": sorted(REQUIRED_TREE),
        "tree_files": _authenticated_tree(ckpt_dir),
        "components": sorted(SUBMODULES),
        "output": out_path.name,
        "output_bytes": out_path.stat().st_size,
        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "tensor_count": total_loaded,
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
