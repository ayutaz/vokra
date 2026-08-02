#!/usr/bin/env python3
"""Merge Meta SeamlessM4T-v2-Large sharded safetensors → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``facebook/seamless-m4t-v2-large`` release ships as **sharded
safetensors** (``model-00001-of-000NN.safetensors`` … + a
``model.safetensors.index.json`` weight-map, ~9-10 GB total). The model is
a multi-modal speech translation transformer (speech encoder + text
encoder + text decoder + T2U + vocoder heads) — architecturally single-
directory, not the multi-sub-module diffusers layout that
``audioldm2_prepare_checkpoint.py`` handles.

Vokra's Rust converter (``crates/vokra-convert/src/models/seamless_m4t_v2.rs``
— pending, this script's landing unblocks the Wave 11 vast.ai retry)
consumes **single-file safetensors** by design — the runtime never grows
a shard-index reader (NFR-DS-02 zero-dep). This script bridges the two:
it walks the weight-map, loads every shard, dedups any shared-storage
tensors (fail-loud if safetensors serialization would collide), strips
INT / bool training-artifact counters, and re-serializes as one
safetensors the caller feeds to ``vokra-cli convert
--model seamless-m4t-v2``.

Precedent: this posture mirrors ``moss_audio_tokenizer_prepare_checkpoint.py``
(2-shard merge) for the weight-map traversal + fail-loud loading, and
``demucs_prepare_checkpoint.py`` for the shared-storage
``.clone().contiguous()`` dedup. SeamlessM4T-v2 may tie speech-encoder ↔
text-encoder positional embeddings across sub-modules — the dedup is
mandatory even in the single-directory case.

# Scale / vast.ai posture

At ~9-10 GB the merged output crosses the [[feedback-large-models-on-vast-ai]]
≥8 GB threshold. Do NOT run against real HF weights on the M1 iMac —
merged safetensors + mmap pages during ``save_file`` would push swap
past the 16 GB envelope (per Voxtral-Small-24B empirical). Retry on
vast.ai as part of Wave 11.

# Determinism

Shards are loaded in the order declared by ``model.safetensors.index.json``'s
``weight_map`` (Python 3.12 dict insertion order is spec-guaranteed).
Identical ``--input-dir`` snapshot produces byte-identical output
(safetensors serialization is deterministic for a fixed key ordering).

# Redistribution

Upstream weight license is ``cc-by-nc-4.0`` (SeamlessM4T-v2 release) —
see ``docs/license-audit.md`` §3.1 row "SeamlessM4T-v2-Large" (owner
sign-off queue). T4 tier (Research-only, non-commercial) precedent per
[[project-x-codec2-t4-precedent]]. This script does not itself gate the
license — that is enforced downstream by the Vokra converter +
``publish-one.sh --allow-noncommercial``.

# Pickle-trust posture

SeamlessM4T-v2-Large ships **safetensors**, not pickle — this script uses
``safetensors.safe_open`` / ``safetensors.torch.load_file`` exclusively
(no ``torch.load`` codepath). The runtime tree never touches pickle
(FR-LD-05).

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/seamless_m4t_v2_large_prepare_checkpoint.py \\
        --input-dir ~/hf-cache/seamless-m4t-v2-large \\
        --output /tmp/seamless-m4t-v2-large.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model seamless-m4t-v2 \\
        --input /tmp/seamless-m4t-v2-large.safetensors \\
        --output /tmp/seamless-m4t-v2-large.gguf

Self-test (no real weights)::

    uv run --project tools/parity python \\
        tools/parity/seamless_m4t_v2_large_prepare_checkpoint.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

LOG_PREFIX = "seamless_m4t_v2_large_prepare_checkpoint:"

# INT dtypes are training-artifact counters (BatchNorm num_batches_tracked,
# rotary-embedding step counters, etc.) — safe to strip. Any dtype outside
# both sets is refused under --strict; without --strict the script logs
# them into the manifest and drops them.
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _partition_and_dedup(sd: dict, strict: bool):
    """Split ``sd`` into ``(kept, dropped_int, unknown_other, shared_pairs)``.

    Shared-storage tensors are cloned (first occurrence stays as-is,
    subsequent occurrences ``.clone().contiguous()`` into fresh storage so
    safetensors accepts them). ``shared_pairs`` records the
    ``(clone_name, original_name)`` tuples for the audit manifest.

    Mirrors the demucs precedent: safetensors hard-errors on ``data_ptr()``
    collision, so we dedup once here rather than let the ``save_file`` call
    fail cryptically at the tail of a ~10 GB serialization.
    """
    import torch  # noqa: F401  (needed for isinstance guards below)

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

    Fail-loud on any load failure OR on a payload that yields zero
    tensors — better a hard exit than a silently-empty prefix (the
    downstream Rust converter would then emit a GGUF with a valid header
    but no weights and the runtime forward would only fail much later at
    first-forward, the classic "silent partial" trap this project bans —
    FR-EX-08).
    """
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{LOG_PREFIX} load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{LOG_PREFIX} {path!s} yielded no tensors — expected a "
            f"SeamlessM4T-v2 shard. Corrupt download or wrong file?"
        )
    return sd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_shards(src_dir: Path) -> tuple[list[Path], dict[str, str] | None]:
    """Discover shard files in ``src_dir``.

    Returns ``(shard_paths, weight_map)`` where ``weight_map`` is the
    parsed ``model.safetensors.index.json`` (or None if the release is a
    single-file un-sharded ``model.safetensors``).
    """
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        single = src_dir / "model.safetensors"
        if single.is_file():
            print(
                f"{LOG_PREFIX} no weight-map found; single-shard release detected "
                f"({single.name}). Loading directly.",
                file=sys.stderr,
            )
            return [single], None
        sys.exit(
            f"{LOG_PREFIX} neither model.safetensors.index.json nor "
            f"model.safetensors found in {src_dir}"
        )

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    wm = index.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        sys.exit(f"{LOG_PREFIX} weight_map is missing or empty in {index_path}")

    # Preserve first-seen order (Python 3.7+ dict insertion order).
    seen: dict[str, None] = {}
    for shard_rel in wm.values():
        if not isinstance(shard_rel, str):
            continue
        seen.setdefault(shard_rel, None)

    shard_paths: list[Path] = []
    for shard_rel in seen:
        shard_path = src_dir / shard_rel
        if not shard_path.is_file():
            sys.exit(
                f"{LOG_PREFIX} weight_map references missing shard: "
                f"{shard_path}"
            )
        shard_paths.append(shard_path)

    return shard_paths, wm


def _run_pipeline(
    src_dir: Path,
    output: Path,
    strict: bool,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift."""
    from safetensors.torch import save_file

    shard_paths, wm = _walk_shards(src_dir)

    merged: dict = {}
    per_shard_stats: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        size_bytes = shard_path.stat().st_size
        print(
            f"{LOG_PREFIX}   loading {shard_path.name} ({size_bytes:,} bytes)",
            file=sys.stderr,
        )
        sub = _load_shard(shard_path)
        overlap = set(merged) & set(sub)
        if overlap:
            print(
                f"{LOG_PREFIX}   duplicate keys across shards "
                f"(first 5): {sorted(overlap)[:5]}",
                file=sys.stderr,
            )
            return 3
        per_shard_stats.append({
            "shard": shard_path.name,
            "size_bytes": size_bytes,
            "tensor_count": len(sub),
        })
        merged.update(sub)

    # Sanity: every declared weight ought to be present after merge.
    if wm is not None:
        missing = [k for k in wm if k not in merged]
        if missing:
            print(
                f"{LOG_PREFIX} weight_map declared {len(missing)} tensors "
                f"absent from merged state_dict (first 5: {missing[:5]})",
                file=sys.stderr,
            )
            return 3

    kept, dropped, unknown, shared_pairs = _partition_and_dedup(merged, strict)

    if unknown and strict:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"{LOG_PREFIX} --strict refusing to drop {len(unknown)} tensors "
            f"of unknown dtype (first 3: {first}); re-run without --strict "
            "if verified inference-inert.",
            file=sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(output))
    written_bytes = output.stat().st_size

    manifest = {
        "input_dir": str(src_dir),
        "output": str(output),
        "kept_count": len(kept),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "sha256": _sha256_file(output),
        "strict": strict,
        "per_shard": per_shard_stats,
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

    print(
        f"{LOG_PREFIX} kept={len(kept)} dropped_int={len(dropped)} "
        f"unknown={len(unknown)} shared_cloned={len(shared_pairs)} "
        f"written_bytes={written_bytes:,} sha256={manifest['sha256'][:16]}... "
        f"manifest -> {manifest_path.name}"
    )
    return 0


def _self_test() -> int:
    """Two-part self-test.

    **Part A (filesystem shard-walk)**: synthesize a 2-shard
    SeamlessM4T-v2-shaped snapshot on disk with the ``.index.json``
    weight-map, round-trip through the full ``_run_pipeline``, and
    assert kept/dropped counts + safetensors reload. Real HF shards
    never contain intra-shard shared storage (safetensors ``save_file``
    refuses it upstream), so this part exercises the weight-map traversal
    + 2-shard merge + INT dtype strip paths with unique tensors — which
    is exactly the shape a real ``facebook/seamless-m4t-v2-large``
    snapshot presents.

    **Part B (in-memory dedup)**: directly call ``_partition_and_dedup``
    on a synthetic dict with tied (shared-storage) tensors, verify the
    dedup path clones the alias and records the pair. This is a
    defensive safety net — even though ``safetensors.torch.load_file``
    always allocates fresh tensors (so real shards can't produce shared
    storage after loading), the dedup guard defends against future
    refactors where the caller merges an in-memory state_dict directly.

    No real weight file is touched — validates the pipeline can be
    walked end-to-end even when the caller has no upstream
    SeamlessM4T-v2 snapshot.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} --self-test: torch/safetensors missing ({exc}). "
            "run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    # ------------------------------------------------------------------
    # Part A: filesystem shard-walk
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        src_dir = Path(td) / "snapshot"
        src_dir.mkdir()

        # Shard 1: speech encoder pieces (all unique tensors — real
        # HF shards look like this after upstream ``save_pretrained``
        # has already deduped tied embeddings).
        shard1 = {
            "speech_encoder.layer.0.attn.q_proj.weight": torch.randn(8, 8),
            "speech_encoder.embed_positions.weight": torch.randn(4, 8),
        }
        shard1_path = src_dir / "model-00001-of-00002.safetensors"
        save_file(shard1, str(shard1_path))

        # Shard 2: text decoder pieces + an INT counter that must drop.
        shard2 = {
            "text_decoder.layer.0.mlp.fc1.weight": torch.randn(16, 8),
            "text_decoder.layer.0.mlp.fc1.bias": torch.randn(16),
            "text_decoder.layer.0.mlp.num_batches_tracked": torch.tensor(
                0, dtype=torch.int64
            ),
        }
        shard2_path = src_dir / "model-00002-of-00002.safetensors"
        save_file(shard2, str(shard2_path))

        # Weight-map declaring all 5 tensors and mapping each to its shard.
        weight_map = {
            "speech_encoder.layer.0.attn.q_proj.weight": shard1_path.name,
            "speech_encoder.embed_positions.weight": shard1_path.name,
            "text_decoder.layer.0.mlp.fc1.weight": shard2_path.name,
            "text_decoder.layer.0.mlp.fc1.bias": shard2_path.name,
            "text_decoder.layer.0.mlp.num_batches_tracked": shard2_path.name,
        }
        (src_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )

        out = Path(td) / "merged.safetensors"
        rc = _run_pipeline(src_dir, out, strict=False)
        if rc != 0:
            print(f"{LOG_PREFIX} --self-test A: pipeline non-zero rc={rc}", file=sys.stderr)
            return rc

        # Assert: 4 float tensors kept (1 int dropped), no shared clones,
        # safetensors reload yields the expected key set.
        loaded = load_file(str(out))
        expected_keys = {
            "speech_encoder.layer.0.attn.q_proj.weight",
            "speech_encoder.embed_positions.weight",
            "text_decoder.layer.0.mlp.fc1.weight",
            "text_decoder.layer.0.mlp.fc1.bias",
        }
        if set(loaded.keys()) != expected_keys:
            print(
                f"{LOG_PREFIX} --self-test A: kept keys "
                f"{sorted(loaded.keys())} != expected {sorted(expected_keys)}",
                file=sys.stderr,
            )
            return 4

        manifest_path = out.with_suffix(out.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["kept_count"] != 4:
            print(f"self-test A: kept_count={manifest['kept_count']} != 4", file=sys.stderr)
            return 4
        if manifest["dropped_int_count"] != 1:
            print(
                f"self-test A: dropped_int_count={manifest['dropped_int_count']} != 1",
                file=sys.stderr,
            )
            return 4
        if len(manifest["per_shard"]) != 2:
            print(
                f"self-test A: per_shard entries={len(manifest['per_shard'])} != 2",
                file=sys.stderr,
            )
            return 4

    # ------------------------------------------------------------------
    # Part B: in-memory dedup (direct _partition_and_dedup unit exercise)
    # ------------------------------------------------------------------
    shared = torch.randn(4, 8, dtype=torch.float32)
    tied_alias = shared.view(4, 8)  # same storage as `shared`
    in_memory_sd = {
        "speech_encoder.embed_positions.weight": shared,
        "text_decoder.embed_tokens.weight_alias": tied_alias,
        "shared_bias.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    kept, dropped, unknown, shared_pairs = _partition_and_dedup(
        in_memory_sd, strict=False
    )
    if len(kept) != 2:
        print(
            f"self-test B: kept={len(kept)} != 2 (both float tensors should survive)",
            file=sys.stderr,
        )
        return 4
    if len(dropped) != 1:
        print(f"self-test B: dropped={len(dropped)} != 1", file=sys.stderr)
        return 4
    if len(shared_pairs) != 1:
        print(
            f"self-test B: shared_pairs={len(shared_pairs)} != 1 "
            "(tied embed alias should have been cloned)",
            file=sys.stderr,
        )
        return 4
    # After dedup, the cloned tensor must have its own storage (data_ptr
    # differs from the original).
    orig_ptr = kept["speech_encoder.embed_positions.weight"].untyped_storage().data_ptr()
    clone_ptr = kept["text_decoder.embed_tokens.weight_alias"].untyped_storage().data_ptr()
    if orig_ptr == clone_ptr:
        print(
            "self-test B: clone shares storage with original (dedup did not fire)",
            file=sys.stderr,
        )
        return 4

    print(f"{LOG_PREFIX} --self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge Meta SeamlessM4T-v2-Large sharded safetensors → single "
            ".safetensors for consumption by vokra-cli convert --model "
            "seamless-m4t-v2."
        ),
    )
    ap.add_argument(
        "--input-dir", type=Path,
        help=(
            "Pre-downloaded HF snapshot directory (must contain "
            "model.safetensors.index.json and every shard listed therein, "
            "or a single un-sharded model.safetensors as a fallback). "
            "Required unless --self-test."
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
            "Default: log them into the manifest and drop them."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a 2-shard SeamlessM4T-v2-shaped snapshot in a "
            "temporary directory, round-trip through the pipeline, and "
            "assert kept/dropped/shared counts. Does NOT touch any "
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
            f"{LOG_PREFIX} missing dep {exc}. run: uv sync (from tools/parity/)",
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

    return _run_pipeline(args.input_dir, args.output, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
