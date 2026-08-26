#!/usr/bin/env python3
"""Merge OpenMOSS MOSS-Audio-4B-Instruct sharded safetensors → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``OpenMOSS-Team/MOSS-Audio-4B-Instruct`` release ships as **N
sharded safetensors** (``model-000NN-of-000NN.safetensors``) plus a
``model.safetensors.index.json`` weight-map. MOSS-Audio-4B-Instruct is
the 4B-parameter decoder/instruct variant of the OpenMOSS audio-LLM
family — architecturally distinct from ``MOSS-Audio-Tokenizer`` (an
encoder-only codec) despite the shared org: it needs its own Rust
converter (``crates/vokra-convert/src/models/moss_audio.rs``) and its
own prep-script (this one). Slug-aliasing the tokenizer prep-script is
wrong — the shard shape overlaps but the downstream Rust converter and
tensor-name schema differ.

Vokra's Rust converter consumes **single-file safetensors** by design —
the runtime never grows a shard-index reader (NFR-DS-02 zero-dep). This
script bridges the two: it walks ``model.safetensors.index.json``, loads
every shard, merges them into a single in-memory state_dict, handles
any shared-storage tensors via the standard ``data_ptr`` dedup +
``.clone().contiguous()`` pattern (safetensors refuses shared storage),
strips int-dtype counters, and writes a single ``.safetensors`` the
caller feeds to ``vokra-cli convert --model moss-audio``.

# Scale / vast.ai posture

4B params × BF16 (2 B/param) ≈ 8 GB — right at the M1-iMac 16 GB RAM
ceiling per memory ``[[feedback-large-models-on-vast-ai]]``. Wave 11
retry is scheduled on vast.ai (Linux box with ≥16 GB RAM headroom)
because a single in-RAM merged state_dict at this scale approaches the
Mac's swap threshold. This script is process-model-neutral — same
byte-for-byte output on either host — but the operator should choose
the host per the memory rule.

# Licence

Upstream weight license is ``apache-2.0`` (per HF cardData primary
source, OpenMOSS-Team release). Wave 11 T4-tier publish path is
un-precedented for a 4B instruct variant; owner sign-off lives at
``docs/license-audit.md`` §3.1 (separate impl task per the wave 11
handoff — this script does NOT modify the license audit).

# Custom code / trust_remote_code

The 4B instruct variant likely ships ``modeling_moss_audio_*.py`` +
``configuration_moss_audio_*.py`` requiring ``trust_remote_code=True``
for the reference Python forward. Vokra never touches Python at runtime,
so this only affects the owner-side parity dumper. This script reads
tensor bytes verbatim without invoking the modeling code — safetensors
is a pure numeric format (no arbitrary pickled objects), so no
``torch.load`` is required in this pipeline.

# Determinism

Shards are loaded in the order declared by ``model.safetensors.index.json``'s
``weight_map`` (Python 3.7+ dict insertion order preserves iteration).
Within a shard, safetensors iterates keys in file-order. Identical
input → byte-identical output.

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/moss_audio_4b_instruct_prepare_checkpoint.py \\
        --input-dir ~/hf-snapshots/moss-audio-4b-instruct \\
        --output ~/hf-snapshots/moss-audio-4b-instruct/model.merged.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model moss-audio \\
        --input ~/hf-snapshots/moss-audio-4b-instruct/model.merged.safetensors \\
        --output /tmp/moss-audio-4b-instruct.gguf

The merged input intentionally remains inside the fixed-revision snapshot so
the Rust converter can authenticate and embed the adjacent tokenizer, chat,
generation and processor sidecars before reading the multi-gigabyte weights.

# Self-test

``--self-test`` fabricates a synthetic 2-shard payload in a tempdir
(``model.safetensors.index.json`` + two shards), one shard containing a
tensor whose storage is shared with a tensor in the same shard (to
exercise the clone path), one shard containing an int64 counter (to
exercise the strip path). Round-trips through the pipeline and asserts
the kept / dropped / shared_cloned counts. No real HF weights touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LOG_PREFIX = "moss_audio_4b_instruct_prepare_checkpoint:"

# Same dtype taxonomy as the sepformer / demucs / moss-audio-tokenizer
# precedents. INT dtypes are training-artefact counters (BatchNorm
# num_batches_tracked etc.) — safe to strip. Any dtype outside both
# sets is refused under --strict; without --strict the script silently
# skips them (they are inference-inert for the target Rust converter).
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

    Cross-shard shared storage is a theoretical possibility (though the
    HF sharding convention typically materialises each shard's tensors
    into distinct buffers on load). We dedup across the full merged
    state_dict regardless, so the manifest surfaces any surprises.
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
            # First tensor at a given ptr stays as-is; aliases clone.
            # ``untyped_storage()`` is the torch >=2.0 forward-compatible
            # accessor.
            try:
                ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else 0
            except Exception:  # noqa: BLE001 — non-tensor slipped through
                ptr = 0
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
    tensors — better a hard exit than a silently-empty prefix.
    """
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{LOG_PREFIX} safetensors.torch.load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{LOG_PREFIX} {path!s} yielded no tensors — expected a "
            f"MOSS-Audio-4B-Instruct shard. Corrupt download or wrong file?"
        )
    return sd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_pipeline(
    src_dir: Path, output: Path, strict: bool,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift.

    Reads ``model.safetensors.index.json`` from ``src_dir``, loads every
    unique shard, merges, partitions/dedups, and writes ``output`` +
    the manifest side-car.
    """
    from safetensors.torch import save_file

    # Locate the weight-map.
    index_path = src_dir / "model.safetensors.index.json"
    per_shard_stats: list[dict] = []

    if not index_path.is_file():
        # Some releases ship a single un-sharded model.safetensors and
        # omit the index. Fall back to that if present.
        single = src_dir / "model.safetensors"
        if single.is_file():
            print(
                f"{LOG_PREFIX} no weight-map found; single-shard release detected "
                f"({single.name}). Loading directly.",
                file=sys.stderr,
            )
            merged = _load_shard(single)
            per_shard_stats.append({
                "shard": single.name,
                "bytes": single.stat().st_size,
                "tensor_count": len(merged),
            })
        else:
            print(
                f"{LOG_PREFIX} neither {index_path.name} nor model.safetensors "
                f"found in {src_dir}",
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

        # Load every unique shard listed in the weight_map, preserving
        # first-seen order (Python 3.7+ dict insertion order).
        seen_shards: dict[str, None] = {}
        for shard_rel in wm.values():
            if not isinstance(shard_rel, str):
                continue
            seen_shards.setdefault(shard_rel, None)

        merged: dict = {}
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
            per_shard_stats.append({
                "shard": shard_rel,
                "bytes": shard_bytes,
                "tensor_count": len(sub),
            })
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
            f"{LOG_PREFIX} --strict refusing to drop {len(unknown)} tensors of "
            f"unknown dtype (first 3: {first}); re-run without --strict if "
            f"verified inference-inert.",
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
    """Synthesize a 2-shard MOSS-Audio-4B-Instruct-shaped payload in a
    tempdir, round-trip through the pipeline, and assert kept/dropped/
    shared_cloned counts + safetensors reload.

    The synthetic payload exercises three quirks:
      - 2 real shards + ``model.safetensors.index.json``
      - one shared-storage tensor pair inside shard 1 (must be cloned)
      - one int64 counter in shard 2 (must be dropped)

    No real HF snapshot is touched — this validates the pipeline can
    walk end-to-end even when the caller has no upstream weights.
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        print(
            f"{LOG_PREFIX} --self-test: torch/safetensors missing "
            f"({exc}). run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src_dir = Path(td) / "hf-snapshot"
        src_dir.mkdir()

        # Shard 1: two float tensors that share storage (safetensors
        # itself will materialize them into distinct buffers on load,
        # so to actually exercise the clone path we save the two names
        # in shard 1 pointing at the same storage BEFORE serializing —
        # but safetensors.torch.save_file will hard-error on shared
        # storage. So we must save them as two distinct copies then
        # arrange for the merged state_dict to alias post-load. We do
        # that by having shard 1 emit only tensor A, and shard 2 emit
        # tensor B and a bfloat16 that mirrors A's *bytes* but is
        # loaded separately (safetensors gives fresh storage per file).
        #
        # To honestly exercise cross-key data_ptr dedup we instead
        # construct the shared-storage aliasing in-memory AFTER load,
        # in the same process, by writing shard 1 with tensor A, then
        # in the pipeline we assert the dedup logic on a pair we
        # construct in-memory. Simpler: we exercise dedup by loading
        # the two shards then explicitly aliasing one key to another's
        # storage via a monkeypatch — but that changes the pipeline.
        #
        # Cleanest: write shard 1 with two DIFFERENT tensors, shard 2
        # with the int64 counter, run _run_pipeline, then separately
        # test _partition_and_dedup directly with a hand-crafted shared
        # pair (the code path is trivial and shared between real and
        # synthetic).
        tensor_a = torch.randn(4, 8, dtype=torch.float32)
        tensor_b = torch.randn(2, 4, dtype=torch.bfloat16)
        tensor_c = torch.randn(3, 3, dtype=torch.float16)
        counter = torch.tensor(0, dtype=torch.int64)

        shard1 = {
            "encoder.layer.0.weight": tensor_a,
            "encoder.layer.0.bias": tensor_b,
        }
        shard2 = {
            "decoder.head.weight": tensor_c,
            "decoder.head.num_batches_tracked": counter,
        }

        save_file(shard1, str(src_dir / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(src_dir / "model-00002-of-00002.safetensors"))

        weight_map = {
            "encoder.layer.0.weight": "model-00001-of-00002.safetensors",
            "encoder.layer.0.bias": "model-00001-of-00002.safetensors",
            "decoder.head.weight": "model-00002-of-00002.safetensors",
            "decoder.head.num_batches_tracked": "model-00002-of-00002.safetensors",
        }
        (src_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2)
        )

        out = Path(td) / "self-test.safetensors"
        rc = _run_pipeline(src_dir, out, strict=False)
        if rc != 0:
            print(f"{LOG_PREFIX} --self-test: pipeline non-zero", file=sys.stderr)
            return rc

        # Assert: 3 float tensors kept, 1 int dropped, 0 shared pairs
        # (no aliased storage in the 2-shard synthetic payload).
        loaded = load_file(str(out))
        expected_keys = {
            "encoder.layer.0.weight",
            "encoder.layer.0.bias",
            "decoder.head.weight",
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
            print(f"self-test: dropped_int_count={manifest['dropped_int_count']} != 1", file=sys.stderr)
            return 4
        if manifest["shared_cloned_count"] != 0:
            print(
                f"self-test: shared_cloned_count={manifest['shared_cloned_count']} != 0 "
                "(no aliased storage in synthetic payload)",
                file=sys.stderr,
            )
            return 4
        if len(manifest["per_shard"]) != 2:
            print(
                f"self-test: per_shard entries={len(manifest['per_shard'])} != 2",
                file=sys.stderr,
            )
            return 4

        # Directly exercise the shared-storage clone path (the shard
        # serializer refuses shared storage on write, so we test the
        # dedup logic directly on an in-memory aliased pair).
        shared_base = torch.randn(6, 8, dtype=torch.bfloat16)
        tied_alias = shared_base.view(6, 8)  # same underlying storage
        aliased_sd = {
            "primary.weight": shared_base,
            "tied.weight": tied_alias,
        }
        kept2, dropped2, unknown2, shared_pairs2 = _partition_and_dedup(
            aliased_sd, strict=False,
        )
        if len(kept2) != 2:
            print(f"self-test: aliased kept={len(kept2)} != 2", file=sys.stderr)
            return 4
        if len(shared_pairs2) != 1:
            print(
                f"self-test: aliased shared_cloned={len(shared_pairs2)} != 1 "
                "(tied tensor should have been cloned)",
                file=sys.stderr,
            )
            return 4
        # After clone, the two tensors must NOT share storage.
        ptr1 = kept2["primary.weight"].untyped_storage().data_ptr()
        ptr2 = kept2["tied.weight"].untyped_storage().data_ptr()
        if ptr1 == ptr2:
            print(
                "self-test: post-clone data_ptrs still equal — clone did not "
                "materialize independent storage",
                file=sys.stderr,
            )
            return 4

    print(f"{LOG_PREFIX} self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge OpenMOSS MOSS-Audio-4B-Instruct sharded safetensors → single "
            ".safetensors for consumption by vokra-cli convert --model moss-audio."
        ),
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Pre-downloaded HF snapshot directory (must contain "
            "model.safetensors.index.json and every shard listed therein, OR "
            "a single model.safetensors fallback). Required unless --self-test."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "destination .safetensors path (parent will be mkdir'd). "
            "Required unless --self-test."
        ),
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "fail-loud on tensors of unknown dtype (fp64 / complex / etc.). "
            "Default: silently skip them (inference-inert for the target "
            "Rust converter)."
        ),
    )
    ap.add_argument(
        "--self-test", action="store_true",
        help=(
            "synthesize a 2-shard MOSS-Audio-4B-Instruct-shaped payload in a "
            "tempdir, round-trip through the pipeline, and assert kept/"
            "skipped/shared_cloned counts. Does NOT touch any upstream weight "
            "file."
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
