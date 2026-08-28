#!/usr/bin/env python3
"""Merge OpenMOSS MOSS-Audio-8B-Instruct sharded safetensors → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``OpenMOSS-Team/MOSS-Audio-8B-Instruct`` release ships as a
multi-shard safetensors bundle (``model-00001-of-000NN.safetensors`` ...)
plus a ``model.safetensors.index.json`` weight-map (~15–17 GB total for
the 8B variant per HF cardData API 2026-08-01). This size class is
squarely in the ``[[feedback-large-models-on-vast-ai]]`` >8 GB bucket
which is why the Wave 10 attempt on the M1 iMac (16 GB RAM) was retired
and the Wave 11 retry runs on vast.ai.

Vokra's Rust converter (``crates/vokra-convert/src/models/moss_audio_*``)
consumes **single-file safetensors** by design — the runtime never grows
a shard-index reader (NFR-DS-02 zero-dep). This script bridges the two:
it walks the weight-map, loads every shard, dedups shared storage,
strips training-artefact int/bool counters, and re-serializes as one
safetensors the caller feeds to ``vokra-cli convert --model moss-audio``.

MOSS-Audio-8B-Instruct is a decoder / instruct LLM variant — architec-
turally distinct from the sibling ``MOSS-Audio-Tokenizer`` (RVQ codec)
even though both live under the OpenMOSS-Team org. The shard-flatten
plumbing is however identical, so this file mirrors the layout of
``moss_audio_tokenizer_prepare_checkpoint.py`` with two demucs-lineage
upgrades:

  1. shared-storage dedup via ``data_ptr()`` +
     ``.clone().contiguous()`` (LLM instruct fine-tunes routinely tie
     ``lm_head.weight`` to ``model.embed_tokens.weight``; safetensors
     hard-errors on ``data_ptr()`` collision).
  2. self-test path that synthesizes a 2-shard payload + one tied
     tensor so the pipeline can be verified end-to-end without touching
     ~17 GB of real weights.

# CLI contract

Wave 11 retry harness pins the surface to
``--input-dir <HF-snapshot> --output <path> [--strict] [--self-test]``
(the operator has already snapshotted the release onto the vast.ai
volume before invoking this script; there is no HF fetch here — the
retry orchestrator drives ``hf download`` separately). This contrasts
with ``moss_audio_tokenizer_prepare_checkpoint.py`` which still bundles
the HF fetch; both patterns are precedent in ``tools/parity/``.

# Determinism

Shards are loaded in the order declared by ``model.safetensors.index.json``'s
``weight_map`` (Python 3.12 dict insertion order is spec-guaranteed).
Identical input → byte-identical output (safetensors serialization is
deterministic for a fixed key ordering).

# Pickle-trust posture

Pure safetensors path — no ``torch.load`` on ``.bin`` files, no
``weights_only=False``. Even for MOSS-Audio's ``modeling_moss_audio_*.py``
+ ``configuration_moss_audio_*.py`` custom code (``trust_remote_code=True``),
this script reads tensor bytes verbatim without invoking the modeling
code. The runtime tree never touches pickle (FR-LD-05).

# Redistribution

Upstream weight license is ``apache-2.0`` — see ``docs/license-audit.md``
§3.1 row "MOSS-Audio-8B-Instruct" (owner sign-off queue as of Wave 11).

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/moss_audio_8b_instruct_prepare_checkpoint.py \\
        --input-dir /vast/snapshots/moss-audio-8b-instruct \\
        --output /vast/snapshots/moss-audio-8b-instruct/model.merged.safetensors

The merged input intentionally remains inside the fixed-revision snapshot so
the Rust converter can authenticate and embed the adjacent tokenizer, chat,
generation and processor sidecars before reading the multi-gigabyte weights.

Self-test (no HF snapshot required)::

    uv run --project tools/parity python \\
        tools/parity/moss_audio_8b_instruct_prepare_checkpoint.py \\
        --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

LOG_PREFIX = "moss_audio_8b_instruct_prepare_checkpoint:"

# Same dtype taxonomy as the sepformer / nemo / demucs / moss-tokenizer
# precedents. INT dtypes are training artefacts (BatchNorm counters,
# rotary_emb inv_freq caches at int stride, tokenizer id lookups etc.) —
# safe to strip. Any dtype outside both sets is refused under --strict.
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _partition_and_dedup(sd: dict, strict: bool):
    """Split into ``(kept, dropped_int, unknown_other, shared_pairs)``.

    Shared-storage tensors are cloned (first occurrence kept verbatim,
    subsequent occurrences ``.clone().contiguous()`` into fresh storage
    so safetensors accepts them). ``shared_pairs`` records the
    (clone_name, original_name) tuples for the audit manifest.

    Mirrors ``demucs_prepare_checkpoint._partition_and_dedup`` — see
    that module for the FR-EX-08 fail-loud rationale on unknown dtypes.
    """
    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)

        if dtype_s in KEEP_DTYPES:
            # data_ptr() dedup — safetensors refuses shared storage.
            # ``untyped_storage()`` is the torch >=2.0 forward-compatible
            # accessor (``.storage()`` warns TypedStorage-deprecated).
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


def _load_shard(path: Path) -> dict:
    """Load one .safetensors file into a flat state_dict.

    Uses ``safetensors.torch.load_file`` (pure safetensors path — no
    ``torch.load`` on ``.bin`` files, no ``weights_only=False``). Fails
    loud on load error OR on a payload that yields zero tensors — better
    a hard exit than a silently-empty prefix (FR-EX-08).
    """
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{LOG_PREFIX} safetensors.torch.load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{LOG_PREFIX} {path!s} yielded no tensors — expected a "
            f"MOSS-Audio-8B-Instruct shard. Corrupt download or wrong file?"
        )
    return sd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_from_input_dir(src_dir: Path) -> dict:
    """Walk ``model.safetensors.index.json`` (or fall back to a single
    ``model.safetensors``) and return the merged state_dict.

    Fail-loud on missing shards, empty weight_map, cross-shard key
    overlap, or missing declared weights (FR-EX-08).
    """
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        single = src_dir / "model.safetensors"
        if single.is_file():
            print(
                f"{LOG_PREFIX} no weight-map found; single-shard release "
                f"detected ({single.name}). Loading directly.",
                file=sys.stderr,
            )
            return _load_shard(single)
        sys.exit(
            f"{LOG_PREFIX} neither model.safetensors.index.json nor "
            f"model.safetensors found in {src_dir}"
        )

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    wm = index.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        sys.exit(
            f"{LOG_PREFIX} weight_map is missing or empty in {index_path}"
        )

    # Unique shards in first-seen order (Python 3.12 dict insertion
    # order is spec-guaranteed).
    seen: dict[str, None] = {}
    for shard_rel in wm.values():
        if not isinstance(shard_rel, str):
            continue
        seen.setdefault(shard_rel, None)

    merged: dict = {}
    for shard_rel in seen:
        shard_path = src_dir / shard_rel
        if not shard_path.is_file():
            sys.exit(
                f"{LOG_PREFIX} weight_map references missing shard: "
                f"{shard_path}"
            )
        print(
            f"{LOG_PREFIX}   loading {shard_rel} "
            f"({shard_path.stat().st_size:,} bytes)",
            file=sys.stderr,
        )
        sub = _load_shard(shard_path)
        overlap = set(merged) & set(sub)
        if overlap:
            sys.exit(
                f"{LOG_PREFIX} duplicate keys across shards "
                f"(first 5): {sorted(overlap)[:5]}"
            )
        merged.update(sub)

    missing = [k for k in wm if k not in merged]
    if missing:
        sys.exit(
            f"{LOG_PREFIX} weight_map declared {len(missing)} tensors "
            f"absent from merged state_dict (first 5: {missing[:5]})"
        )

    return merged


def _run_pipeline(
    merged: dict,
    output: Path,
    strict: bool,
    input_dir: str | None,
    shard_stats: list[dict] | None = None,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift."""
    from safetensors.torch import save_file

    kept, dropped, unknown, shared_pairs = _partition_and_dedup(merged, strict)

    if unknown and strict:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"{LOG_PREFIX} --strict refusing to drop {len(unknown)} "
            f"tensors of unknown dtype (first 3: {first}); re-run without "
            f"--strict if verified inference-inert.",
            file=sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(output))
    written_bytes = output.stat().st_size

    manifest = {
        "output": str(output),
        "input_dir": input_dir,
        "kept_count": len(kept),
        "skipped_count": len(dropped) + len(unknown),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "strict": strict,
        "sha256": _sha256_file(output),
        "shard_stats": shard_stats or [],
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in unknown
        ],
        "shared_pairs": [
            {"clone": clone, "original": orig} for clone, orig in shared_pairs
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    skipped = len(dropped) + len(unknown)
    print(
        f"{LOG_PREFIX} kept={len(kept)} skipped={skipped} "
        f"shared_cloned={len(shared_pairs)} written_bytes={written_bytes:,} "
        f"sha256={manifest['sha256'][:16]}... manifest -> {manifest_path.name}"
    )
    return 0


def _self_test() -> int:
    """Synthesize a 2-shard MOSS-Audio-shaped payload + index.json,
    round-trip through the pipeline, and assert kept/skipped/shared
    counts + safetensors reload.

    Exercises: (a) index-json shard walk, (b) shared-storage clone
    (LLM lm_head↔embed_tokens tied weight), (c) int64 counter strip.
    Uses only tempfile + torch + safetensors — no HF snapshot required.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} --self-test: torch/safetensors missing ({exc}). "
            f"run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()

        # Shard 1: embed_tokens (tied) + a decoder layer weight.
        embed = torch.randn(32, 8, dtype=torch.bfloat16)
        shard1 = {
            "model.embed_tokens.weight": embed,
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
        }
        save_file(shard1, str(src / "model-00001-of-00002.safetensors"))

        # Shard 2: lm_head tied to embed_tokens + another layer + int counter.
        # NB: safetensors.save_file dedups shared storage inside a single
        # shard by writing an "alias" entry; to force the *cross-shard*
        # tied-weight case (the one the Rust converter would see after
        # merge) we save lm_head with the same values but as an
        # independent tensor here, then have _partition_and_dedup treat
        # them as separate. To exercise the *actual* dedup path we
        # additionally place an in-shard-2 alias.
        lm_head_shared_source = torch.randn(16, 8, dtype=torch.bfloat16)
        lm_head_alias = lm_head_shared_source.view(16, 8)  # same storage
        shard2 = {
            "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "lm_head.weight": lm_head_shared_source,
            "lm_head.tied_alias": lm_head_alias,  # aliases lm_head.weight storage
            "model.layers.1.input_layernorm.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
        }
        # safetensors.save_file itself deduplicates shared tensors when
        # writing; to keep the alias distinguishable through the round
        # trip we bypass it by writing shard2 with cloned storage and
        # then reintroducing the alias in the in-memory merged dict
        # (the real-world MOSS-Audio bundle has already-materialised
        # per-shard files where cross-shard ties are re-encountered via
        # the merge — this mirrors that).
        shard2_serialisable = {
            k: (v.clone().contiguous() if k == "lm_head.tied_alias" else v)
            for k, v in shard2.items()
        }
        save_file(shard2_serialisable, str(src / "model-00002-of-00002.safetensors"))

        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.input_layernorm.num_batches_tracked": "model-00001-of-00002.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "lm_head.tied_alias": "model-00002-of-00002.safetensors",
                "model.layers.1.input_layernorm.num_batches_tracked": "model-00002-of-00002.safetensors",
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        # Merge from the synthetic snapshot.
        merged = _merge_from_input_dir(src)

        # Post-load: reintroduce a cross-tensor shared-storage alias so
        # the dedup path fires (safetensors.load_file returns each
        # tensor with its own storage). We rebind lm_head.tied_alias to
        # share storage with lm_head.weight to simulate the tied-weight
        # case the LLM converters hit after merge.
        merged["lm_head.tied_alias"] = merged["lm_head.weight"].view_as(
            merged["lm_head.tied_alias"]
        )

        out = Path(td) / "self-test.safetensors"
        rc = _run_pipeline(
            merged, out, strict=False, input_dir=str(src),
            shard_stats=[
                {"shard": "model-00001-of-00002.safetensors", "tensors": 3},
                {"shard": "model-00002-of-00002.safetensors", "tensors": 4},
            ],
        )
        if rc != 0:
            print(f"{LOG_PREFIX} --self-test: pipeline non-zero", file=sys.stderr)
            return rc

        loaded = load_file(str(out))
        expected_keys = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "lm_head.weight",
            "lm_head.tied_alias",
        }
        if set(loaded.keys()) != expected_keys:
            print(
                f"{LOG_PREFIX} --self-test: kept keys "
                f"{sorted(loaded.keys())} != expected {sorted(expected_keys)}",
                file=sys.stderr,
            )
            return 4

        manifest_path = out.with_suffix(out.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["kept_count"] != 5:
            print(
                f"{LOG_PREFIX} --self-test: kept_count="
                f"{manifest['kept_count']} != 5",
                file=sys.stderr,
            )
            return 4
        if manifest["dropped_int_count"] != 2:
            print(
                f"{LOG_PREFIX} --self-test: dropped_int_count="
                f"{manifest['dropped_int_count']} != 2",
                file=sys.stderr,
            )
            return 4
        if manifest["shared_cloned_count"] != 1:
            print(
                f"{LOG_PREFIX} --self-test: shared_cloned_count="
                f"{manifest['shared_cloned_count']} != 1 "
                "(lm_head↔tied_alias should have been cloned)",
                file=sys.stderr,
            )
            return 4
        if not manifest["sha256"]:
            print(f"{LOG_PREFIX} --self-test: sha256 missing", file=sys.stderr)
            return 4

    print(f"{LOG_PREFIX} self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge OpenMOSS MOSS-Audio-8B-Instruct sharded safetensors → "
            "single .safetensors for consumption by vokra-cli convert "
            "--model moss-audio."
        ),
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Pre-downloaded HF snapshot directory (must contain "
            "model.safetensors.index.json and every shard listed therein, "
            "or a single model.safetensors). The Wave 11 retry harness "
            "runs `hf download OpenMOSS-Team/MOSS-Audio-8B-Instruct` "
            "into this directory before invoking this script."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "fail-loud on tensors of unknown dtype (fp64 / complex / "
            "etc.). Default: silently skip them (they are inference-inert "
            "for the LLM decoder forward path)."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a 2-shard MOSS-Audio-shaped payload + index.json "
            "in a tempdir, round-trip through the pipeline, and assert "
            "kept/skipped/shared_cloned counts. Does NOT touch any "
            "upstream weight file."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.input_dir is None or args.output is None:
        print(
            f"{LOG_PREFIX} --input-dir and --output are required "
            "(unless --self-test).",
            file=sys.stderr,
        )
        return 2

    try:
        import torch  # noqa: F401
        import safetensors.torch  # noqa: F401
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} missing dep {exc}. run: uv sync (from "
            "tools/parity/)",
            file=sys.stderr,
        )
        return 2

    if not args.input_dir.is_dir():
        print(
            f"{LOG_PREFIX} --input-dir must be an existing directory: "
            f"{args.input_dir}",
            file=sys.stderr,
        )
        return 2

    print(f"{LOG_PREFIX} merging shards from {args.input_dir}", file=sys.stderr)
    merged = _merge_from_input_dir(args.input_dir)

    # Per-shard tensor counts for the manifest (recompute cheaply from
    # the index for the audit log).
    shard_stats: list[dict] = []
    index_path = args.input_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            wm = json.load(f).get("weight_map", {})
        by_shard: dict[str, int] = {}
        for shard_rel in wm.values():
            if isinstance(shard_rel, str):
                by_shard[shard_rel] = by_shard.get(shard_rel, 0) + 1
        shard_stats = [
            {"shard": s, "tensors": c} for s, c in by_shard.items()
        ]

    return _run_pipeline(
        merged,
        args.output,
        strict=args.strict,
        input_dir=str(args.input_dir),
        shard_stats=shard_stats,
    )


if __name__ == "__main__":
    sys.exit(main())
