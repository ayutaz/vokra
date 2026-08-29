#!/usr/bin/env python3
"""Merge a ``cvssp/audioldm2-large`` HF snapshot → single ``.safetensors``.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

Upstream ``cvssp/audioldm2-large`` (~14 GB on disk) ships as a
diffusers-layout **multi-sub-module bundle** — the same shape as
``cvssp/audioldm2`` (see sibling ``audioldm2_prepare_checkpoint.py``)
but larger and, per sub-module, more likely to be **sharded** across
``model-00001-of-000NN.safetensors`` + ``model.safetensors.index.json``.
The pipeline root looks roughly like::

    audioldm2-large/
      vae/            model.safetensors                  # AutoencoderKL
      unet/           model-00001-of-00002.safetensors   # AudioLDM2UNet2D
                      model-00002-of-00002.safetensors   # (sharded)
                      model.safetensors.index.json
      vocoder/        model.safetensors                  # HiFi-GAN
      language_model/ pytorch_model.bin OR .safetensors  # GPT-2 caption LM
      text_encoder/   model.safetensors                  # T5-base
      text_encoder_2/ model.safetensors                  # CLAP

The Vokra Rust converter
(``crates/vokra-convert/src/models/audioldm2.rs``) reads **single-file
safetensors only** by design — the runtime never grows a shard-index
reader nor a pickle parser (NFR-DS-02 zero-dep + FR-LD-05 no pickle in
runtime). This script bridges the two:

  (a) walks each declared sub-module directory,
  (b) if ``model.safetensors.index.json`` is present, dedup + sort the
      ``weight_map`` values → unique shard filenames, then load each
      shard in insertion order,
  (c) else load the single ``model.safetensors`` (or fall back to
      ``pytorch_model.bin`` via ``torch.load(weights_only=True)`` for the
      ``language_model/`` GPT-2 sub-module which older releases ship as
      pickle),
  (d) re-prefixes each sub-module's keys with its role name
      (``vae.*`` / ``unet.*`` / ``vocoder.*`` / ``language_model.*`` /
      ``text_encoder.*`` / ``text_encoder_2.*``) so the merged flat
      state_dict has globally-unique keys,
  (e) dedups shared-storage tensors via ``untyped_storage().data_ptr()``
      (safetensors refuses shared storage — must ``.clone().contiguous()``
      on aliases; first occurrence stays as-is),
  (f) rejects INT/bool counters instead of stripping them; omission would
      invalidate the fixed component and sidecar contract,
  (g) re-serialises as one flat safetensors + emits a
      ``.manifest.json`` side-car recording per-shard tensor counts and
      the shared-tied pair list for owner audit.

Precedent: ``audioldm2_prepare_checkpoint.py`` (multi-sub-module merge,
same SUBMODULES table), ``demucs_prepare_checkpoint.py`` (shared-storage
dedup via data_ptr + manifest side-car + ``--self-test``),
``moss_audio_tokenizer_prepare_checkpoint.py`` (shard-index walk).

# Scale — vast.ai handoff

AudioLDM 2 Large is ~14 GB on disk; the merge pass roughly doubles peak
resident RSS (all sub-modules held in memory before the flat safetensors
write). The M1 iMac 16 GB machine sits well past the safe local threshold
per memory ``[[feedback-large-models-on-vast-ai]]`` (≥8 GB → vast.ai) —
run this script on vast.ai per
``docs/handoff/vast-ai-large-model-publish.md``.

# License — CVSSP weight policy

Weights ship **cc-by-nc-sa-4.0** per CVSSP GitHub README + paper Ethics §
(Liu et al. 2024 ICML arXiv:2308.05734). The HF model card's YAML
front-matter carries the looser ``cc-by-nc-4.0`` tag; we follow the more
restrictive ShareAlike form per CVSSP-owned primary source (same
Fish-Speech precedent for CC-BY-NC-SA-4.0). Redistribution to
``huggingface.co/vokra/audioldm2-large`` is **blocked** on an owner ADR
covering the SA cascade obligation onto Vokra-added artifacts (model
card, LICENSE, NOTICE) — see
``scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`` +
``docs/license-audit.md`` §3.1.

# NOT REFERENCED (clean-room)

- github.com/haoheliu/AudioLDM2 (MIT code, but distinct from the
  cc-by-nc-sa-4.0 weight release — treat as opaque for prep)

Uses only ``torch`` (BSD-3) + ``safetensors`` (Apache-2.0). No AudioLDM 2
code source is read or referenced (clean-room).

# FR-EX-08 loud-error posture

- Missing sub-module directory → fail closed; every fixed component is
  mandatory and the aggregate merge is never partial.
- Malformed shard-index (missing declared shard file, empty weight_map,
  cross-shard key overlap) → ``sys.exit(3)``.
- Cross-sub-module key collision (role prefix should guarantee
  uniqueness) → fail loudly with the colliding key set.
- INT/bool or unknown dtype → always refused; there is no omission-prone
  non-strict bypass.

Usage
-----

::

    uv run --project tools/parity python \\
        tools/parity/audioldm2_large_prepare_checkpoint.py \\
        --input-dir /workspace/cvssp/audioldm2-large \\
        --output /workspace/audioldm2-large.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model audioldm2-large \\
        --input /workspace/audioldm2-large.safetensors \\
        --output /workspace/audioldm2-large.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


# Role prefix ← sub-module dir name mapping. Every sub-module rooted at
# its own directory under the pipeline root. Keys are the diffusers-
# convention sub-dir names; values are the flat role prefix we write
# into the merged safetensors (``{prefix}.{key}`` form). ``text_encoder``
# + ``text_encoder_2`` = the T5-base + CLAP pair (both slots consumed by
# the U-Net's cross-attention); kept role-distinct so a future
# ``AudioLdm2Large::from_gguf`` can bind them independently.
SUBMODULES: dict[str, str] = {
    "vae": "vae",
    "unet": "unet",
    "vocoder": "vocoder",
    "language_model": "language_model",
    "projection_model": "projection_model",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}

DEFAULT_HF_REPO = "cvssp/audioldm2-large"
DEFAULT_REVISION = "4b0b875a9e0c5305dfc917da808584e50e1c7ed4"
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

# Fixed dtype taxonomy. INT/bool and unknown dtypes are never stripped:
# omission would invalidate the authenticated component manifest.
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _log(msg: str) -> None:
    print(f"[audioldm2-large-prep] {msg}", file=sys.stderr, flush=True)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _lfs_pointer_sha1(sha256: str, size: int) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n".encode()
    return _git_blob_sha1(pointer)


def _validate_fixed_bundle(root: Path) -> None:
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise SystemExit(
            f"audioldm2-large-prep: BLOCKED: symlinks are not allowed: {symlinks[:3]}"
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix()
        not in {".vokra-source-revision", ".vokra-server-tree.json"}
    }
    missing = sorted(REQUIRED_TREE - actual)
    extra = sorted(actual - REQUIRED_TREE)
    if missing or extra:
        raise SystemExit(
            "audioldm2-large-prep: BLOCKED: snapshot tree is not the fixed "
            f"official {DEFAULT_HF_REPO}@{DEFAULT_REVISION}; "
            f"missing={missing[:4]} extra={extra[:4]}"
        )
    packet_path = root / ".vokra-server-tree.json"
    if not packet_path.is_file():
        raise SystemExit(
            "audioldm2-large-prep: BLOCKED: authoritative server-tree packet is "
            "required; a self-created revision marker is not source/model authentication"
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
        raise SystemExit(
            f"audioldm2-large-prep: BLOCKED: invalid server-tree identity packet: {exc}"
        )


def _authenticated_tree(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        }
        for relative in sorted(REQUIRED_TREE)
    ]


def _load_shards_from_index(sub_dir: Path, index_path: Path) -> dict[str, Any]:
    """Walk ``model.safetensors.index.json`` and load each unique shard.

    Contract mirrors ``moss_audio_tokenizer_prepare_checkpoint.py``:
    dedup + sort ``weight_map.values()`` preserving Python 3.7+ dict
    insertion order (Python 3.12 spec-guaranteed), load each shard, refuse
    cross-shard key overlap or a shard file that is declared but missing.
    """
    from safetensors.torch import load_file  # local: only needed here

    idx = json.loads(index_path.read_text())
    wm: dict[str, str] = idx.get("weight_map") or {}
    if not wm:
        sys.exit(
            f"audioldm2-large-prep: {sub_dir.name}/{index_path.name} has empty "
            "weight_map — release layout changed?"
        )

    # Dedup while preserving first-seen order so the manifest per-shard
    # entries come out in a deterministic sequence.
    seen: set[str] = set()
    shard_order: list[str] = []
    for fname in wm.values():
        if fname not in seen:
            seen.add(fname)
            shard_order.append(fname)

    merged: dict[str, Any] = {}
    for fname in shard_order:
        shard_path = sub_dir / fname
        if not shard_path.is_file():
            sys.exit(
                f"audioldm2-large-prep: index.json declared shard "
                f"{sub_dir.name}/{fname} but the file is missing"
            )
        _log(
            f"    shard: {fname} ({shard_path.stat().st_size:,} bytes)"
        )
        shard_state = load_file(str(shard_path))
        for k, v in shard_state.items():
            if k in merged:
                sys.exit(
                    f"audioldm2-large-prep: key '{k}' present in multiple "
                    f"shards under {sub_dir.name}/ (index.json contract "
                    "violation)"
                )
            merged[k] = v

    # Sanity: index.json declares every key; verify none went missing.
    missing = [k for k in wm.keys() if k not in merged]
    if missing:
        sys.exit(
            f"audioldm2-large-prep: {len(missing)} keys declared in "
            f"{sub_dir.name}/{index_path.name} but absent from loaded shards "
            f"(first 3: {missing[:3]})"
        )
    return merged


def _load_single_safetensors(sub_dir: Path) -> dict[str, Any]:
    """Load ``model.safetensors`` / ``diffusion_pytorch_model.safetensors``
    from ``sub_dir`` if either exists; else empty dict.

    Prefers the diffusers-convention filename when both are present (rare
    but legal in older mixed-layout releases).
    """
    from safetensors.torch import load_file  # local: only needed here

    for cand in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        p = sub_dir / cand
        if p.is_file():
            _log(f"    single: {cand} ({p.stat().st_size:,} bytes)")
            return load_file(str(p))
    return {}


def _load_pickle_fallback(sub_dir: Path) -> dict[str, Any]:
    """Fall back to ``pytorch_model.bin`` / ``diffusion_pytorch_model.bin``
    for older releases that ship pickle instead of safetensors.

    ``weights_only=True`` is the safe path — refuses arbitrary object
    construction (no code execution on load). This is the only pickle
    surface in the script and it is offline-only (never enters runtime).
    """
    import torch  # local: only needed here

    for cand in ("pytorch_model.bin", "diffusion_pytorch_model.bin"):
        p = sub_dir / cand
        if p.is_file():
            _log(f"    pickle: {cand} ({p.stat().st_size:,} bytes)")
            loaded = torch.load(
                str(p), map_location="cpu", weights_only=True
            )
            if not isinstance(loaded, dict):
                sys.exit(
                    f"audioldm2-large-prep: {sub_dir.name}/{cand} did not "
                    "deserialize to a state_dict"
                )
            return loaded
    return {}


def _load_submodule(
    root: Path, sub_name: str, role_prefix: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one sub-module + apply role prefix.

    Returns ``(prefixed_state, sub_manifest)`` where ``sub_manifest`` is
    a dict recording which loader path was taken + how many tensors came
    from each shard (for the aggregate ``.manifest.json``).
    """
    sub_dir = root / sub_name
    if not sub_dir.is_dir():
        raise SystemExit(f"audioldm2-large-prep: required sub-module is absent: {sub_name}/")

    _log(f"loading sub-module: {sub_name}/ (role prefix: {role_prefix})")

    index_path = sub_dir / "model.safetensors.index.json"
    inner: dict[str, Any] = {}
    loader: str = ""
    per_shard: list[dict[str, Any]] = []

    if index_path.is_file():
        loader = "sharded_safetensors"
        _log(f"  sharded via {index_path.name}")
        # For per-shard counts we replay the walk cheaply so the manifest
        # can report tensor counts without re-loading anything (uses
        # weight_map inversion).
        idx = json.loads(index_path.read_text())
        wm: dict[str, str] = idx.get("weight_map") or {}
        inner = _load_shards_from_index(sub_dir, index_path)
        counts: dict[str, int] = {}
        for k, fname in wm.items():
            counts[fname] = counts.get(fname, 0) + 1
        for fname in sorted(counts.keys()):
            per_shard.append({"shard": fname, "tensor_count": counts[fname]})
    else:
        inner = _load_single_safetensors(sub_dir)
        if inner:
            loader = "single_safetensors"
            per_shard.append({"shard": "model.safetensors", "tensor_count": len(inner)})
        else:
            raise SystemExit(
                f"audioldm2-large-prep: required safetensors are absent under {sub_name}/; "
                "pickle fallback is disabled by the fixed bundle contract"
            )

    if not inner:
        raise SystemExit(f"audioldm2-large-prep: no loadable weights under {sub_name}/")

    prefixed: dict[str, Any] = {}
    for key, tensor in inner.items():
        prefixed[f"{role_prefix}.{key}"] = tensor

    _log(f"  loaded {len(prefixed):,} tensors from {sub_name}/ via {loader}")
    sub_manifest: dict[str, Any] = {
        "present": True,
        "loader": loader,
        "tensor_count": len(prefixed),
        "shards": per_shard,
    }
    return prefixed, sub_manifest


def _partition_and_dedup(sd: dict[str, Any], strict: bool):
    """Split into ``(kept, dropped_int, unknown_other, shared_pairs)``.

    Shared-storage tensors are cloned (first occurrence kept verbatim,
    subsequent occurrences ``.clone().contiguous()`` into fresh storage
    so safetensors accepts them). ``shared_pairs`` records the
    (clone_name, original_name) tuples for the audit manifest.
    """
    import torch  # noqa: F401  # needed indirectly for tensor accessors

    kept: dict[str, Any] = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)

        if dtype_s in KEEP_DTYPES:
            # untyped_storage() is the torch>=2.0 forward-compatible
            # accessor (.storage() warns TypedStorage-deprecated).
            ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else 0
            if ptr and ptr in seen:
                shared_pairs.append((name, seen[ptr]))
                t = t.detach().clone().contiguous()
            else:
                if ptr:
                    seen[ptr] = name
                t = t.detach().contiguous()
            kept[name] = t
        elif dtype_s in INT_DTYPES:
            raise SystemExit(
                f"audioldm2-large-prep: BLOCKED: tensor {name} has dtype {dtype_s}; "
                "integer/bool omission is forbidden by the fixed bundle contract"
            )
        else:
            raise SystemExit(
                f"audioldm2-large-prep: BLOCKED: tensor {name} has unsupported dtype {dtype_s}"
            )

    return kept, dropped, unknown, shared_pairs


def _run_pipeline(
    root: Path,
    output: Path,
    strict: bool,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift."""
    from safetensors.torch import save_file

    merged: dict[str, Any] = {}
    sub_manifests: dict[str, dict[str, Any]] = {}

    for sub_name, role_prefix in SUBMODULES.items():
        prefixed, sub_manifest = _load_submodule(root, sub_name, role_prefix)
        sub_manifests[sub_name] = sub_manifest
        # Cross-sub-module collision check (role prefix contract).
        for key in prefixed:
            if key in merged:
                sys.exit(
                    f"audioldm2-large-prep: cross-sub-module key collision: "
                    f"'{key}' present in multiple sub-modules (role prefix "
                    "contract broken?)"
                )
        merged.update(prefixed)

    if not merged:
        sys.exit(
            "audioldm2-large-prep: no loadable sub-modules found under "
            f"{root} (expected one of {sorted(SUBMODULES.keys())} with either "
            "safetensors, sharded safetensors, or pytorch_model.bin)"
        )

    kept, dropped, unknown, shared_pairs = _partition_and_dedup(merged, strict)

    if unknown:
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(output))
    written_bytes = output.stat().st_size

    manifest = {
        "output": str(output),
        "root": str(root),
        "kept_count": len(kept),
        "skipped_count": len(dropped) + len(unknown),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "strict": strict,
        "schema": "vokra.audioldm2.bundle.v1",
        "status": "SNAPSHOT_TREE_AUTHENTICATED_COMPONENTS_STAGED",
        "source_repo": DEFAULT_HF_REPO,
        "source_revision": DEFAULT_REVISION,
        "tree": sorted(REQUIRED_TREE),
        "tree_files": _authenticated_tree(root),
        "components": sorted(SUBMODULES),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "submodules": sub_manifests,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in unknown
        ],
        "shared_tied": [
            {"clone": clone, "original": orig} for clone, orig in shared_pairs
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    skipped = len(dropped) + len(unknown)
    print(
        f"audioldm2-large-prep: kept={len(kept)} skipped={skipped} "
        f"shared_cloned={len(shared_pairs)} written_bytes={written_bytes:,} "
        f"manifest -> {manifest_path.name}"
    )
    return 0
def _self_test() -> int:
    """Run only the negative fixed-contract test; no synthetic bundle is promoted."""
    try:
        import torch
        _partition_and_dedup({"forbidden.int": torch.tensor(1, dtype=torch.int64)}, False)
    except SystemExit:
        print("audioldm2-large-prep --self-test: OK")
        return 0
    except ImportError as exc:
        print(f"audioldm2-large-prep --self-test: torch missing ({exc})", file=sys.stderr)
        return 2
    print("audioldm2-large-prep --self-test: integer rejection did not fire", file=sys.stderr)
    return 4


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge a cvssp/audioldm2-large HF snapshot (multi-sub-module "
            "bundle, some sub-modules sharded) → single .safetensors the "
            "Vokra converter consumes."
        ),
    )
    ap.add_argument(
        "--input-dir", type=Path,
        help=(
            "path to a downloaded cvssp/audioldm2-large snapshot directory "
            "(the diffusers pipeline root containing vae/ unet/ vocoder/ "
            "language_model/ text_encoder/ text_encoder_2/ sub-directories)."
        ),
    )
    ap.add_argument(
        "--output", type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "compatibility flag; the fixed contract always fails loudly on "
            "integer, bool, or unknown tensor dtypes."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a 3-sub-module audioldm2-large-shaped snapshot in a "
            "tempdir, round-trip through the pipeline, and assert "
            "kept/skipped/shared_cloned/loader counts. Does NOT touch any "
            "upstream weight file."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.input_dir is None or args.output is None:
        print(
            "audioldm2-large-prep: --input-dir and --output are required "
            "(unless --self-test).",
            file=sys.stderr,
        )
        return 2

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"audioldm2-large-prep: torch missing ({exc}). "
            "run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2
    try:
        import safetensors.torch  # noqa: F401
    except ImportError as exc:
        print(
            f"audioldm2-large-prep: safetensors missing ({exc}). "
            "run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    if not args.input_dir.is_dir():
        print(
            f"--input-dir must be an existing directory: {args.input_dir}",
            file=sys.stderr,
        )
        return 2

    try:
        _validate_fixed_bundle(args.input_dir)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 3
    _log(f"input snapshot: {args.input_dir}")
    return _run_pipeline(args.input_dir, args.output, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
