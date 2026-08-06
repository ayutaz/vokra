#!/usr/bin/env python3
"""Merge Qwen/Qwen2.5-Omni-7B sharded safetensors → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``Qwen/Qwen2.5-Omni-7B`` release ships as **multiple sharded
safetensors** files (``model-00001-of-000NN.safetensors`` …
``model-000NN-of-000NN.safetensors``, ~15 shards ~15 GB total) plus a
``model.safetensors.index.json`` weight-map (Qwen omni-modal architecture:
thinker + talker + audio encoder + vision encoder + LLM decoder all in
one flat state_dict). Because the released bundle is >8 GB, the operator
runs this script on **vast.ai** rather than on the M1 iMac
([[feedback-large-models-on-vast-ai]]).

Vokra's Rust converter (``crates/vokra-convert/src/models/qwen2_5_omni.rs``,
slug ``qwen2-5-omni``) consumes **single-file safetensors** by design —
the runtime never grows a shard-index reader (NFR-DS-02 zero-dep). This
script bridges the two: it walks the weight-map, loads every shard,
de-duplicates shared storage (Qwen's thinker/talker tie some embedding
weights), merges into a single in-memory state_dict, strips int-dtype
training counters, and re-serializes as one safetensors the caller feeds
to ``vokra-cli convert --model qwen2-5-omni``.

The bundle is **safetensors-native** end-to-end — no pickle bridge is
required. Pickled ``pytorch_model*.bin`` shards are NOT accepted (this
script exits 3 rather than fall back to ``torch.load(weights_only=True)``,
because the Rust converter's parity dumper is the pickle-trust boundary,
not this script — see the ``demucs_prepare_checkpoint.py`` precedent for
where pickle trust IS acknowledged, but only because that upstream ships
``.th`` archives with class references).

Precedent: this posture mirrors ``moss_audio_tokenizer_prepare_checkpoint.py``
(canonical 2-shard flat merger, HF-native safetensors bundle) and
``demucs_prepare_checkpoint.py`` (shared-storage dedup via ``data_ptr()``
map + ``.clone().contiguous()``). This script combines both into the
Qwen omni-modal N-shard flat merger.

# Determinism

Shards are loaded in the order declared by ``model.safetensors.index.json``'s
``weight_map`` (Python 3.12 dict insertion order preserves iteration
verbatim). Identical ``--input-dir`` input produces byte-identical
output (safetensors serialization is deterministic for a fixed key
ordering; the state_dict iteration order becomes the safetensors header
order becomes the on-disk tensor order).

# Redistribution

Upstream weight license (``Qwen/Qwen2.5-Omni-7B``) is per the Qwen
research license — see ``docs/license-audit.md`` §3.1 row "Qwen2.5-Omni-7B"
(owner sign-off queue; publish path is the T4 Research-only tier
precedented by X-Codec-2, requiring ``--allow-noncommercial`` explicit
on ``publish-one.sh``). This offline sidecar does not itself publish
anything.

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/qwen2_5_omni_7b_prepare_checkpoint.py \\
        --input-dir /workspace/qwen2-5-omni-7b-src \\
        --output /workspace/qwen2-5-omni-7b.safetensors

    # Then downstream:
    vokra-cli convert --model qwen2-5-omni \\
        --input  /workspace/qwen2-5-omni-7b.safetensors \\
        --output /workspace/qwen2-5-omni-7b.gguf

Fail-loud posture (FR-EX-08):
  - missing ``model.safetensors.index.json`` → exit 3
  - empty ``weight_map`` → exit 3
  - shard listed in weight_map but absent on disk → exit 3
  - key overlap across shards → exit 3
  - declared tensor absent after merge → exit 3
  - tensor of unknown dtype (fp64 / complex / …) + ``--strict`` → exit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

LOG_PREFIX = "qwen2_5_omni_7b_prepare_checkpoint:"

# INT dtypes come from training-artifact counters (BatchNorm
# num_batches_tracked, RoPE inv_freq caches persisted as int registers,
# etc.). Safe to strip. bool is treated as an int variant (attention
# masks etc. are inference-inert when persisted). Any dtype outside both
# KEEP + INT sets is refused under --strict (fail-loud posture).
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
    so safetensors accepts them — the serializer hard-errors on
    ``data_ptr()`` collision with ``RuntimeError: The weights trying to
    be saved contain shared tensors``). ``shared_pairs`` records the
    (clone_name, original_name) tuples for the audit manifest.

    Mirrors ``demucs_prepare_checkpoint.py::_partition_and_dedup``
    verbatim — Qwen2.5-Omni ties thinker/talker embedding weights so the
    dedup is mandatory, not a defensive nicety.
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
            # Empty tensors have no meaningful data_ptr; skip the dedup
            # for them (ptr=0 sentinel).
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
    first-forward, the classic "silent partial" trap this project bans
    per FR-EX-08).
    """
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{LOG_PREFIX} safetensors.torch.load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{LOG_PREFIX} {path!s} yielded no tensors — expected a "
            f"Qwen2.5-Omni-7B shard. Corrupt download or wrong file?"
        )
    return sd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_pipeline(
    src_dir: Path,
    output: Path,
    strict: bool,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift.

    Walks ``model.safetensors.index.json``'s ``weight_map`` in insertion
    order, loads every unique shard, checks for cross-shard key overlap,
    verifies every declared weight is present after merge, then
    partitions + dedups shared storage + saves.
    """
    from safetensors.torch import save_file

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # Some releases ship a single un-sharded model.safetensors and
        # omit the index. Fall back to that if present, mirroring the
        # moss_audio_tokenizer precedent.
        single = src_dir / "model.safetensors"
        if single.is_file():
            print(
                f"{LOG_PREFIX} no weight-map found; single-shard release "
                f"detected ({single}). Loading directly.",
                file=sys.stderr,
            )
            merged = _load_shard(single)
            per_shard_stats: list[dict] = [
                {"shard": single.name, "bytes": single.stat().st_size, "kept_count": len(merged)}
            ]
        else:
            print(
                f"{LOG_PREFIX} neither {index_path.name} nor "
                f"model.safetensors found in {src_dir}",
                file=sys.stderr,
            )
            return 3
    else:
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        wm = index.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            print(
                f"{LOG_PREFIX} weight_map is missing or empty in {index_path}",
                file=sys.stderr,
            )
            return 3

        # Preserve first-seen shard order (Python 3.7+ dict insertion
        # order guarantee — this is what makes the byte-identical output
        # claim in the docstring hold).
        seen_shards: dict[str, None] = {}
        for shard_rel in wm.values():
            if not isinstance(shard_rel, str):
                continue
            seen_shards.setdefault(shard_rel, None)

        merged: dict = {}
        per_shard_stats = []
        for shard_rel in seen_shards:
            shard_path = src_dir / shard_rel
            if not shard_path.is_file():
                print(
                    f"{LOG_PREFIX} weight_map references missing shard: "
                    f"{shard_path}",
                    file=sys.stderr,
                )
                return 3
            shard_bytes = shard_path.stat().st_size
            print(
                f"{LOG_PREFIX}   loading {shard_rel} ({shard_bytes:,} bytes)",
                file=sys.stderr,
            )
            sub = _load_shard(shard_path)
            overlap = set(merged) & set(sub)
            if overlap:
                # A well-formed index should never map a tensor name to
                # two shards; assert loudly if it does.
                print(
                    f"{LOG_PREFIX}   duplicate keys across shards "
                    f"(first 5): {sorted(overlap)[:5]}",
                    file=sys.stderr,
                )
                return 3
            per_shard_stats.append(
                {"shard": shard_rel, "bytes": shard_bytes, "kept_count": len(sub)}
            )
            merged.update(sub)

        # Sanity: every declared weight ought to be present after merge.
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
            f"if verified inference-inert.",
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
        "skipped_count": len(dropped) + len(unknown),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "strict": strict,
        "sha256": _sha256_file(output),
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

    skipped = len(dropped) + len(unknown)
    print(
        f"{LOG_PREFIX} kept={len(kept)} skipped={skipped} "
        f"shared_cloned={len(shared_pairs)} written_bytes={written_bytes:,} "
        f"sha256={manifest['sha256'][:16]}... "
        f"manifest -> {manifest_path.name}"
    )
    return 0


def _self_test() -> int:
    """Synthesize an on-disk 2-shard Qwen2.5-Omni-shaped bundle, round-trip
    through the pipeline, and assert kept/skipped/shared_cloned counts
    plus safetensors reload.

    Exercises the four end-to-end quirks: (a) ``model.safetensors.index.json``
    walk (b) 2-shard merge with cross-shard key uniqueness check
    (c) shared-storage tensor clone (Qwen thinker/talker tied embedding)
    (d) int-dtype strip. No real weight file is touched — this validates
    the code path can be walked end-to-end even on a fresh vast.ai box
    with no upstream weights downloaded yet.
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

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()

        # Shard 1: thinker embedding + a bfloat16 body weight.
        # Shard 2: talker embedding aliased onto shard-1 storage (tied)
        #          + int64 counter (training artefact, must be dropped).
        # We can't actually tie storage ACROSS shards in real
        # safetensors (each shard is its own file), so we simulate the
        # in-memory shared-storage state that arises after ALL shards
        # are load_file()'d into one Python dict and a tie hook re-links
        # them. For the self-test we do the tie in-memory in _partition:
        # write two distinct storages to disk, but in a third synthetic
        # step (after load) alias them via .view() before partition.
        #
        # Approach: (i) write two independent shards, (ii) simulate the
        # merged dict + tie by aliasing after load, (iii) manually run
        # the partition+dedup step and assert. This mirrors what happens
        # in production when a Qwen HF hook re-ties weights.

        thinker_embed = torch.randn(4, 3, dtype=torch.bfloat16)
        body_weight = torch.randn(8, 5, dtype=torch.float32)
        shard1 = {
            "thinker.embed_tokens.weight": thinker_embed,
            "thinker.layers.0.self_attn.q_proj.weight": body_weight,
        }
        save_file(shard1, str(src / "model-00001-of-00002.safetensors"))

        talker_embed_indep = torch.randn(4, 3, dtype=torch.bfloat16)
        int_counter = torch.tensor(0, dtype=torch.int64)
        shard2 = {
            "talker.embed_tokens.weight": talker_embed_indep,
            "talker.layers.0.self_attn.q_proj.num_batches_tracked": int_counter,
        }
        save_file(shard2, str(src / "model-00002-of-00002.safetensors"))

        # Weight map: every declared tensor points to its shard.
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "thinker.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "thinker.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "talker.embed_tokens.weight": "model-00002-of-00002.safetensors",
                "talker.layers.0.self_attn.q_proj.num_batches_tracked": "model-00002-of-00002.safetensors",
            },
        }
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        out = Path(td) / "out.safetensors"
        rc = _run_pipeline(src, out, strict=False)
        if rc != 0:
            print(f"{LOG_PREFIX} --self-test: pipeline non-zero: {rc}", file=sys.stderr)
            return rc

        # Assert: 3 float tensors kept (thinker.embed, body, talker.embed),
        # 1 int dropped (num_batches_tracked). No shared_cloned in this
        # basic 2-shard payload because save_file()+load_file() rebuilds
        # storage independently for each shard — so the natural
        # shared_pairs count is 0. That's fine: the code path for dedup
        # is exercised even when it yields 0 pairs (the ``seen`` dict is
        # populated), and the aliasing branch is covered by
        # ``demucs_prepare_checkpoint.py --self-test``.
        loaded = load_file(str(out))
        expected_keys = {
            "thinker.embed_tokens.weight",
            "thinker.layers.0.self_attn.q_proj.weight",
            "talker.embed_tokens.weight",
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
        if manifest["kept_count"] != 3:
            print(f"self-test: kept_count={manifest['kept_count']} != 3", file=sys.stderr)
            return 4
        if manifest["dropped_int_count"] != 1:
            print(
                f"self-test: dropped_int_count={manifest['dropped_int_count']} != 1",
                file=sys.stderr,
            )
            return 4
        if len(manifest["per_shard"]) != 2:
            print(
                f"self-test: per_shard len={len(manifest['per_shard'])} != 2",
                file=sys.stderr,
            )
            return 4
        # Now exercise the shared-storage dedup path with an in-memory
        # payload where two names alias the same storage (the actual
        # Qwen thinker/talker tie).
        shared_src = torch.randn(4, 3, dtype=torch.bfloat16)
        alias = shared_src.view(4, 3)
        aliased_sd = {
            "thinker.embed_tokens.weight": shared_src,
            "talker.embed_tokens.weight": alias,
        }
        kept, dropped, unknown, shared_pairs = _partition_and_dedup(
            aliased_sd, strict=False
        )
        if len(shared_pairs) != 1:
            print(
                f"self-test: aliased dedup produced {len(shared_pairs)} pairs != 1",
                file=sys.stderr,
            )
            return 4
        if len(kept) != 2:
            print(
                f"self-test: aliased dedup kept={len(kept)} != 2",
                file=sys.stderr,
            )
            return 4

    print(f"{LOG_PREFIX} self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge Qwen/Qwen2.5-Omni-7B sharded safetensors → single "
            ".safetensors for consumption by vokra-cli convert --model "
            "qwen2-5-omni."
        ),
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Path to a pre-downloaded Qwen/Qwen2.5-Omni-7B HF snapshot "
            "directory (must contain model.safetensors.index.json and every "
            "shard listed therein, or a single model.safetensors). Typically "
            "the output of `hf download Qwen/Qwen2.5-Omni-7B --local-dir <dir>` "
            "run on vast.ai (see [[feedback-large-models-on-vast-ai]] — the "
            "~15 GB bundle exceeds the M1 iMac 16 GB RAM ceiling)."
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
            "fail-loud on tensors of unknown dtype (fp64 / complex / etc.). "
            "Default: silently skip them (they are inference-inert for "
            "Qwen2.5-Omni's audio/vision/text encoders + LLM decoder)."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a 2-shard Qwen2.5-Omni-shaped bundle on-disk, round-trip "
            "through the pipeline, and assert kept/skipped/shared_pairs counts. "
            "Does NOT touch any upstream weight file."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.input_dir is None or args.output is None:
        print(
            f"{LOG_PREFIX} --input-dir and --output are required "
            f"(unless --self-test).",
            file=sys.stderr,
        )
        return 2

    try:
        from safetensors.torch import save_file  # noqa: F401
        import torch  # noqa: F401
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
