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
  (f) strips INT-dtype counters (BatchNorm ``num_batches_tracked``,
      position ids, etc.) which the Vokra safetensors reader admits only
      F32 / F16 / BF16,
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

- Missing sub-module directory → warn + skip (release layouts can drop
  optional sub-modules); the aggregate merge must still contain ≥1
  loadable sub-module or the script fails loudly at end.
- Malformed shard-index (missing declared shard file, empty weight_map,
  cross-shard key overlap) → ``sys.exit(3)``.
- Cross-sub-module key collision (role prefix should guarantee
  uniqueness) → fail loudly with the colliding key set.
- INT-dtype tensor → dropped with a manifest entry (default), or refused
  with ``--strict``.
- Unknown dtype (fp64 / complex / …) → refused with ``--strict``; without
  ``--strict`` silently dropped (mirrors demucs precedent).

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
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder_2",
}

# Same dtype taxonomy as the demucs / sepformer / musicgen precedents.
# INT dtypes are training artefacts (BatchNorm num_batches_tracked
# counters, position ids etc.) — safe to strip. Any dtype outside both
# sets is refused under --strict; without --strict the script silently
# skips them (they are inference-inert for the U-Net / VAE / vocoder).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _log(msg: str) -> None:
    print(f"[audioldm2-large-prep] {msg}", file=sys.stderr, flush=True)


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
        _log(f"sub-module absent, skipping: {sub_name}/")
        return {}, {"present": False}

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
            inner = _load_pickle_fallback(sub_dir)
            if inner:
                loader = "pickle_fallback"
                per_shard.append({"shard": "pytorch_model.bin", "tensor_count": len(inner)})

    if not inner:
        _log(f"  WARN: sub-module {sub_name}/ present on disk but no loadable weights")
        return {}, {"present": True, "loader": "none", "tensor_count": 0}

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
            dropped.append((name, dtype_s, list(t.shape)))
        else:
            unknown.append((name, dtype_s, list(t.shape)))

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

    if unknown and strict:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"audioldm2-large-prep: --strict refusing to drop "
            f"{len(unknown)} tensors of unknown dtype (first 3: {first}); "
            "re-run without --strict if verified inference-inert.",
            file=sys.stderr,
        )
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
    """Synthesize a 3-submodule audioldm2-large-shaped snapshot on disk,
    round-trip through the pipeline, and assert kept/skipped/shared/loader
    counts + safetensors reload.

    Exercises the four audioldm2-large-specific quirks:
      (a) SHARDED safetensors under ``unet/`` with ``model.safetensors.
          index.json`` (walks the shard-index)
      (b) SINGLE-file safetensors under ``vae/`` (single-file path)
      (c) SHARED-storage tensor across two keys under ``text_encoder/``
          (data_ptr dedup + clone)
      (d) INT-dtype counter (num_batches_tracked) → dropped

    No upstream weight file is touched; the synthetic snapshot is a
    tempdir tree the script tears down at exit.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        print(
            f"audioldm2-large-prep --self-test: torch/safetensors missing "
            f"({exc}). run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "audioldm2-large"

        # --- (a) unet/ sharded via model.safetensors.index.json --------
        unet_dir = root / "unet"
        unet_dir.mkdir(parents=True)
        # shard 1: 2 tensors
        s1 = {
            "conv_in.weight": torch.randn(4, 3, dtype=torch.float32),
            "conv_in.bias": torch.randn(4, dtype=torch.float32),
        }
        save_file(s1, str(unet_dir / "model-00001-of-00002.safetensors"))
        # shard 2: 1 tensor + 1 int counter (int64 → drop path)
        s2 = {
            "down_blocks.0.attn.weight": torch.randn(8, 4, dtype=torch.float16),
            "down_blocks.0.bn.num_batches_tracked": torch.tensor(
                0, dtype=torch.int64
            ),
        }
        save_file(s2, str(unet_dir / "model-00002-of-00002.safetensors"))
        wm = {
            "conv_in.weight": "model-00001-of-00002.safetensors",
            "conv_in.bias": "model-00001-of-00002.safetensors",
            "down_blocks.0.attn.weight": "model-00002-of-00002.safetensors",
            "down_blocks.0.bn.num_batches_tracked":
                "model-00002-of-00002.safetensors",
        }
        (unet_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 999}, "weight_map": wm})
        )

        # --- (b) vae/ single-file model.safetensors --------------------
        vae_dir = root / "vae"
        vae_dir.mkdir(parents=True)
        save_file(
            {"encoder.conv1.weight": torch.randn(2, 1, dtype=torch.float32)},
            str(vae_dir / "model.safetensors"),
        )

        # --- (c) text_encoder/ with SHARED-storage tensor pair ---------
        # We must materialize the shared-storage aliasing on the LOAD side
        # (safetensors serializer would refuse to write it, and any reload
        # would break sharing). So: write two independent tensors to disk,
        # then post-hoc monkey-patch the loaded dict to alias them.
        te_dir = root / "text_encoder"
        te_dir.mkdir(parents=True)
        base = torch.randn(6, 8, dtype=torch.bfloat16)
        save_file(
            {"shared.weight": base},
            str(te_dir / "model.safetensors"),
        )
        # We rely on the runtime path: replace safetensors.torch.load_file
        # inside _load_single_safetensors so that when it loads this
        # sub-module, we return TWO keys sharing storage via .view().
        import safetensors.torch as st_torch  # type: ignore[import]

        real_load = st_torch.load_file

        def load_file_with_alias(path):  # type: ignore[no-untyped-def]
            loaded = real_load(path)
            if Path(path).parent.name == "text_encoder":
                shared_view = loaded["shared.weight"].view(6, 8)
                loaded["shared.tied_alias.weight"] = shared_view
            return loaded

        st_torch.load_file = load_file_with_alias  # type: ignore[assignment]
        try:
            out = Path(td) / "self-test.safetensors"
            rc = _run_pipeline(root, out, strict=False)
        finally:
            st_torch.load_file = real_load  # restore, always

        if rc != 0:
            print(
                "audioldm2-large-prep --self-test: pipeline non-zero",
                file=sys.stderr,
            )
            return rc

        # --- Assertions ------------------------------------------------
        loaded = load_file(str(out))
        expected_keys = {
            "unet.conv_in.weight",
            "unet.conv_in.bias",
            "unet.down_blocks.0.attn.weight",
            "vae.encoder.conv1.weight",
            "text_encoder.shared.weight",
            "text_encoder.shared.tied_alias.weight",
        }
        if set(loaded.keys()) != expected_keys:
            print(
                f"self-test: kept keys {sorted(loaded.keys())} != expected "
                f"{sorted(expected_keys)}",
                file=sys.stderr,
            )
            return 4

        manifest_path = out.with_suffix(out.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["kept_count"] != 6:
            print(f"self-test: kept_count={manifest['kept_count']} != 6", file=sys.stderr)
            return 4
        if manifest["dropped_int_count"] != 1:
            print(
                f"self-test: dropped_int_count={manifest['dropped_int_count']} != 1",
                file=sys.stderr,
            )
            return 4
        if manifest["shared_cloned_count"] != 1:
            print(
                f"self-test: shared_cloned_count={manifest['shared_cloned_count']} "
                "!= 1 (text_encoder tied alias should have been cloned)",
                file=sys.stderr,
            )
            return 4
        subs = manifest["submodules"]
        if subs["unet"]["loader"] != "sharded_safetensors":
            print(
                f"self-test: unet loader={subs['unet']['loader']!r} != 'sharded_safetensors'",
                file=sys.stderr,
            )
            return 4
        if subs["vae"]["loader"] != "single_safetensors":
            print(
                f"self-test: vae loader={subs['vae']['loader']!r} != 'single_safetensors'",
                file=sys.stderr,
            )
            return 4
        if subs["vocoder"]["present"] is not False:
            print(
                f"self-test: vocoder present={subs['vocoder']['present']!r} != False "
                "(sub-module was not synthesized; should be reported absent)",
                file=sys.stderr,
            )
            return 4
        # unet manifest should record 2 shards.
        unet_shards = subs["unet"]["shards"]
        if len(unet_shards) != 2:
            print(
                f"self-test: unet shards={len(unet_shards)} != 2",
                file=sys.stderr,
            )
            return 4

    print("audioldm2-large-prep --self-test: OK")
    return 0


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
            "fail-loud on tensors of unknown dtype (fp64 / complex / etc.). "
            "Default: silently skip them (they are inference-inert for the "
            "AudioLDM 2 pipeline)."
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

    _log(f"input snapshot: {args.input_dir}")
    return _run_pipeline(args.input_dir, args.output, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
