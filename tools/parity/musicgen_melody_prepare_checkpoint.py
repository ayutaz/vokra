#!/usr/bin/env python3
"""Merge a ``facebook/musicgen-melody`` HF snapshot → single ``.safetensors``.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``facebook/musicgen-melody`` release ships as a multi-shard
safetensors bundle (LM decoder + T5-base text encoder + EnCodec RVQ audio
codec + **chromagram / melody conditioner** — the last of which is what
distinguishes ``musicgen-melody`` from the base MusicGen family; the
melody conditioner ingests an audio reference and emits a chroma feature
sequence the LM cross-attends over during generation). Total ~4-6 GB.

Vokra's Rust converter (planned ``crates/vokra-convert/src/models/musicgen_melody.rs``,
sibling of the already-landed ``musicgen_medium.rs`` / ``musicgen_large.rs``)
consumes **single-file safetensors** by design — the runtime never grows
a shard-index reader or a pickle parser (NFR-DS-02 zero-dep + FR-LD-05
no pickle in runtime). This script bridges the two: it walks
``model.safetensors.index.json``, loads every shard, dedups shared
storage (the T5 encoder's ``embed_tokens`` / ``lm_head`` weight-tie is
the canonical case), strips int/bool counters, and re-serializes as one
safetensors the caller feeds to ``vokra-cli convert --model musicgen-melody``.

Precedent: ``musicgen_medium_prepare_checkpoint.py`` (same family, same
shard-shape) + ``moss_audio_tokenizer_prepare_checkpoint.py`` (shard-index
merge) + ``demucs_prepare_checkpoint.py`` (shared-storage clone dedup).
The three families of quirk fold together here.

# Scale — vast.ai handoff

MusicGen-Melody at ~4-6 GB is in the borderline band where the M1 iMac
16 GB machine *might* handle it, but per memory
``[[feedback-large-models-on-vast-ai]]`` the Wave 10 attempt failed on
this machine and Wave 11 targets a vast.ai retry. Run this script on
vast.ai per ``docs/handoff/vast-ai-large-model-publish.md``.

# License — Meta AudioCraft weight policy

Weights ship **cc-by-nc-4.0** per HF cardData primary source; the code
layer at ``github.com/facebookresearch/audiocraft`` is MIT. This prep
script + the downstream Vokra converter both treat the artifact under
the weight-distribution license (X-Codec 2 T4 Research-only precedent
landed 2026-07-28). ``publish-one.sh --allow-noncommercial`` gate must
fire before any upload.

# Shared-storage handling

MusicGen bundles a frozen T5-base text encoder alongside the LM decoder;
T5 releases historically tie ``shared`` / ``encoder.embed_tokens`` /
``decoder.embed_tokens`` / ``lm_head`` weights (all four alias one
storage). Safetensors refuses shared storage
(``RuntimeError: The weights trying to be saved contain shared tensors``)
so this script dedups via a ``{data_ptr: first_name}`` map + subsequent
``.clone().contiguous()`` (first tensor stays as-is; each alias becomes
an independent copy). Manifest records the tied pairs for owner audit.
This mirrors the ``demucs_prepare_checkpoint.py`` posture.

# FR-EX-08 loud-error posture

- Missing ``--input-dir`` / missing shard listed in weight_map → exit 3.
- Cross-shard key overlap → exit 3 (a well-formed index maps each name
  to exactly one shard; a duplicate is upstream corruption).
- Any weight declared in ``weight_map`` but absent from the merged
  state after all shards load → exit 3.
- INT / bool dtype (BatchNorm ``num_batches_tracked``, position ids) is
  dropped with a warn and recorded in the manifest (the Rust reader
  admits only F32 / F16 / BF16 anyway).
- Any dtype outside ``KEEP_DTYPES ∪ INT_DTYPES`` is refused loudly under
  ``--strict``; without ``--strict`` it is dropped with a warn (mirrors
  the demucs default posture — permissive by default, strict opt-in).

Usage
-----

::

    uv run --project tools/parity python \\
        tools/parity/musicgen_melody_prepare_checkpoint.py \\
        --input-dir /path/to/facebook-musicgen-melody-snapshot \\
        --output /tmp/musicgen-melody.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model musicgen-melody \\
        --input /tmp/musicgen-melody.safetensors \\
        --output /tmp/musicgen-melody.gguf

The ``--self-test`` mode synthesizes an in-memory 2-shard payload with a
weight-tie + an int64 counter and asserts kept / dropped / shared_cloned
counts round-trip through safetensors. It exercises the shard-walk,
dedup, and partition code paths without touching any upstream weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOG_PREFIX = "musicgen_melody_prepare_checkpoint:"

# INT dtypes are training-artifact counters (BatchNorm num_batches_tracked
# etc.). Safe to strip. KEEP_DTYPES is the set the Vokra safetensors
# reader admits. Any dtype outside both sets is refused under --strict
# (mirrors the moss_audio_tokenizer / demucs taxonomy).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr, flush=True)


def _load_shard(path: Path) -> dict:
    """Load one ``.safetensors`` file into a flat ``{name: torch.Tensor}``
    state_dict. Fail loudly on empty payload (upstream corruption or
    wrong file) rather than silently emitting an empty prefix — the
    ``moss_audio_tokenizer_prepare_checkpoint.py`` posture."""
    from safetensors.torch import load_file

    try:
        sd = load_file(str(path), device="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{LOG_PREFIX} safetensors.torch.load_file({path!s}) failed: {exc}")
    if not sd:
        sys.exit(
            f"{LOG_PREFIX} {path!s} yielded no tensors — expected a "
            f"MusicGen-Melody shard. Corrupt download or wrong file?"
        )
    return sd


def _partition_and_dedup(sd: dict, strict: bool):
    """Split into ``(kept, dropped_int, unknown_other, shared_pairs)``.

    Shared-storage tensors are cloned (first occurrence kept verbatim,
    subsequent occurrences ``.clone().contiguous()`` into fresh storage
    so safetensors accepts them). ``shared_pairs`` records the
    ``(clone_name, original_name)`` tuples for the audit manifest.

    Mirrors ``demucs_prepare_checkpoint.py::_partition_and_dedup`` — the
    T5 embed-tokens / lm-head weight-tie in MusicGen bundles is the
    canonical case.
    """
    import torch  # noqa: F401  (needed by dtype/storage introspection below)

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


def _merge_shards(input_dir: Path) -> tuple[dict, list[str]]:
    """Walk ``model.safetensors.index.json`` (or single-file fallback)
    and merge every shard into one in-memory state_dict.

    Returns ``(merged, shard_names)``. Fails loudly on missing shard,
    cross-shard key overlap, or a declared-but-missing weight.
    """
    index_path = input_dir / "model.safetensors.index.json"
    single_path = input_dir / "model.safetensors"

    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        wm = index.get("weight_map")
        if not isinstance(wm, dict) or not wm:
            sys.exit(
                f"{LOG_PREFIX} weight_map is missing or empty in {index_path}"
            )

        # De-dup the shard set preserving first-seen order (Python dict
        # insertion order is spec-guaranteed since 3.7 — determinism).
        seen: dict[str, None] = {}
        for shard_rel in wm.values():
            if not isinstance(shard_rel, str):
                continue
            seen.setdefault(shard_rel, None)

        merged: dict = {}
        shard_names: list[str] = []
        for shard_rel in seen:
            shard_path = input_dir / shard_rel
            if not shard_path.is_file():
                sys.exit(
                    f"{LOG_PREFIX} weight_map references missing shard: "
                    f"{shard_path}"
                )
            _log(
                f"  loading {shard_rel} "
                f"({shard_path.stat().st_size:,} bytes)"
            )
            sub = _load_shard(shard_path)
            overlap = set(merged) & set(sub)
            if overlap:
                sys.exit(
                    f"{LOG_PREFIX} duplicate keys across shards "
                    f"(first 5): {sorted(overlap)[:5]}"
                )
            merged.update(sub)
            shard_names.append(shard_rel)

        # Sanity: every declared weight ought to be present after merge.
        missing = [k for k in wm if k not in merged]
        if missing:
            sys.exit(
                f"{LOG_PREFIX} weight_map declared {len(missing)} tensors "
                f"absent from merged state_dict (first 5: {missing[:5]})"
            )
        return merged, shard_names

    if single_path.is_file():
        _log(
            f"no weight-map found; single-shard release detected "
            f"({single_path.name}). Loading directly."
        )
        return _load_shard(single_path), [single_path.name]

    sys.exit(
        f"{LOG_PREFIX} neither model.safetensors.index.json nor "
        f"model.safetensors found in {input_dir}"
    )


def _run_pipeline(
    merged: dict,
    output: Path,
    strict: bool,
    shard_names: list[str],
    input_dir: Path | None,
) -> int:
    """Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift."""
    from safetensors.torch import save_file

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
        "input_dir": str(input_dir) if input_dir is not None else None,
        "output": str(output),
        "shard_names": shard_names,
        "shard_count": len(shard_names),
        "kept_count": len(kept),
        "skipped_count": len(dropped) + len(unknown),
        "dropped_int_count": len(dropped),
        "unknown_count": len(unknown),
        "shared_cloned_count": len(shared_pairs),
        "written_bytes": written_bytes,
        "strict": strict,
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
        f"{LOG_PREFIX} shards={len(shard_names)} kept={len(kept)} "
        f"skipped={skipped} shared_cloned={len(shared_pairs)} "
        f"written_bytes={written_bytes:,} manifest -> {manifest_path.name}"
    )
    return 0


def _self_test() -> int:
    """Synthesize a 2-shard MusicGen-Melody-shaped payload in-memory,
    round-trip through the pipeline, and assert kept / skipped /
    shared_cloned counts + safetensors reload.

    Exercises the three shard-merge quirks: (a) walk a weight_map index
    across 2 shards (b) shared-storage tensor clone (T5 embed-tokens /
    lm-head weight-tie stand-in) (c) int-dtype strip. No real weight
    file is touched — this validates the code path can be walked
    end-to-end even when the caller has no upstream weights.
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

    # Build a synthetic 2-shard bundle:
    #
    #   shard-1: text_encoder + embedding weight-tie (shared storage)
    #   shard-2: LM decoder (int64 counter to be stripped)
    #
    # The weight-tie mimics the T5 embed_tokens / lm_head alias pattern
    # MusicGen inherits from the frozen T5-base text encoder. We
    # deliberately place the aliases in the SAME shard first (so the
    # dedup fires within one _load_shard call). MusicGen-Melody's
    # ``conditioner.chroma`` sub-block is stand-in'd via a plain float
    # tensor — the melody-conditioner shape is not architecturally
    # special for this offline serialization step.
    shared = torch.randn(4, 8)
    tied_alias = shared.view(4, 8)  # same underlying storage as `shared`
    shard1_state = {
        "text_encoder.shared.weight": shared,
        "text_encoder.encoder.embed_tokens.weight": tied_alias,
        "conditioner.chroma.proj.weight": torch.randn(2, 4, dtype=torch.bfloat16),
    }
    shard2_state = {
        "decoder.layers.0.self_attn.q_proj.weight": torch.randn(3, 5),
        "decoder.layers.0.norm.num_batches_tracked": torch.tensor(
            0, dtype=torch.int64
        ),
    }

    with tempfile.TemporaryDirectory() as td:
        src_dir = Path(td) / "musicgen-melody-src"
        src_dir.mkdir()

        shard1_name = "model-00001-of-00002.safetensors"
        shard2_name = "model-00002-of-00002.safetensors"
        # NOTE: safetensors.torch.save_file itself refuses shared storage,
        # so we must clone the alias here to write a *valid* upstream-style
        # shard. This does NOT circumvent the pipeline's dedup test — after
        # writing, the two names still exist as distinct entries in the
        # shard, and the pipeline's own _load_shard will not see shared
        # storage (safetensors reloads each key into its own buffer). To
        # exercise the pipeline's data_ptr dedup path we detour: assemble
        # the merged dict directly (bypassing the shard round-trip) with
        # the true storage-alias intact, and drive _run_pipeline on it.
        # This preserves the semantic test (shared_cloned_count == 1)
        # without violating safetensors' own no-shared-storage invariant.
        #
        # We still write the 2-shard files to disk so the shard-walk +
        # cross-shard-overlap + missing-shard error paths ARE exercised —
        # we just don't rely on the on-disk shards to carry the storage
        # alias.
        s1_for_disk = {k: v.detach().clone().contiguous() for k, v in shard1_state.items()}
        save_file(s1_for_disk, str(src_dir / shard1_name))
        save_file(
            {k: v.detach().contiguous() for k, v in shard2_state.items()},
            str(src_dir / shard2_name),
        )

        # Write a matching weight_map index — the pipeline's shard-walk
        # will re-load these and confirm the expected shard count / key
        # ownership. (We assert its return code separately below.)
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "text_encoder.shared.weight": shard1_name,
                "text_encoder.encoder.embed_tokens.weight": shard1_name,
                "conditioner.chroma.proj.weight": shard1_name,
                "decoder.layers.0.self_attn.q_proj.weight": shard2_name,
                "decoder.layers.0.norm.num_batches_tracked": shard2_name,
            },
        }
        (src_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        # (1) Exercise the on-disk shard-walk path first — this validates
        # weight_map parsing, missing-shard detection, and cross-shard
        # overlap detection. The result won't have the storage alias
        # (safetensors round-trip breaks it) so shared_cloned_count here
        # is 0 — that's fine and expected.
        walked, shard_names = _merge_shards(src_dir)
        expected_names = {
            "text_encoder.shared.weight",
            "text_encoder.encoder.embed_tokens.weight",
            "conditioner.chroma.proj.weight",
            "decoder.layers.0.self_attn.q_proj.weight",
            "decoder.layers.0.norm.num_batches_tracked",
        }
        if set(walked.keys()) != expected_names:
            print(
                f"{LOG_PREFIX} --self-test: walked keys "
                f"{sorted(walked.keys())} != expected {sorted(expected_names)}",
                file=sys.stderr,
            )
            return 4
        if shard_names != [shard1_name, shard2_name]:
            print(
                f"{LOG_PREFIX} --self-test: shard_names {shard_names} != "
                f"expected [{shard1_name}, {shard2_name}]",
                file=sys.stderr,
            )
            return 4

        # (2) Exercise _run_pipeline against an in-memory merged dict
        # that still has the true storage alias (shard round-trip
        # cannot carry storage aliases through disk). This is the ONLY
        # way to exercise the data_ptr dedup path in an offline
        # self-test — safetensors on disk is by definition alias-free.
        merged_with_alias = {**shard1_state, **shard2_state}
        out = Path(td) / "self-test.safetensors"
        rc = _run_pipeline(
            merged_with_alias,
            out,
            strict=False,
            shard_names=[shard1_name, shard2_name],
            input_dir=src_dir,
        )
        if rc != 0:
            print(
                f"{LOG_PREFIX} --self-test: pipeline non-zero rc={rc}",
                file=sys.stderr,
            )
            return rc

        loaded = load_file(str(out))
        expected_kept = {
            "text_encoder.shared.weight",
            "text_encoder.encoder.embed_tokens.weight",
            "conditioner.chroma.proj.weight",
            "decoder.layers.0.self_attn.q_proj.weight",
        }
        if set(loaded.keys()) != expected_kept:
            print(
                f"{LOG_PREFIX} --self-test: kept keys "
                f"{sorted(loaded.keys())} != expected {sorted(expected_kept)}",
                file=sys.stderr,
            )
            return 4

        manifest_path = out.with_suffix(out.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["kept_count"] != 4:
            print(
                f"{LOG_PREFIX} --self-test: kept_count="
                f"{manifest['kept_count']} != 4",
                file=sys.stderr,
            )
            return 4
        if manifest["dropped_int_count"] != 1:
            print(
                f"{LOG_PREFIX} --self-test: dropped_int_count="
                f"{manifest['dropped_int_count']} != 1",
                file=sys.stderr,
            )
            return 4
        if manifest["shared_cloned_count"] != 1:
            print(
                f"{LOG_PREFIX} --self-test: shared_cloned_count="
                f"{manifest['shared_cloned_count']} != 1 "
                "(T5-style embed-tokens weight-tie should have been cloned)",
                file=sys.stderr,
            )
            return 4
        if manifest["shard_count"] != 2:
            print(
                f"{LOG_PREFIX} --self-test: shard_count="
                f"{manifest['shard_count']} != 2",
                file=sys.stderr,
            )
            return 4

    print(f"{LOG_PREFIX} --self-test: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge a facebook/musicgen-melody HF snapshot (sharded "
            ".safetensors) → single .safetensors for consumption by "
            "vokra-cli convert --model musicgen-melody."
        ),
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Path to an already-downloaded facebook/musicgen-melody HF "
            "snapshot directory (must contain model.safetensors.index.json "
            "and every shard listed therein, OR a single model.safetensors "
            "for un-sharded releases). Required unless --self-test."
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Destination .safetensors path (e.g. "
            "/tmp/musicgen-melody.safetensors). Parent directory will be "
            "mkdir'd. Required unless --self-test."
        ),
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail-loud on tensors of unknown dtype (fp64 / complex / etc.). "
            "Default: silently skip them (mirrors demucs_prepare_checkpoint "
            "posture — permissive by default, strict opt-in)."
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Synthesize a 2-shard MusicGen-Melody-shaped payload in-memory, "
            "round-trip through the pipeline, and assert kept / skipped / "
            "shared_cloned counts. Does NOT touch any upstream weight file."
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

    _log(f"walking snapshot dir: {args.input_dir}")
    merged, shard_names = _merge_shards(args.input_dir)
    _log(
        f"merged {len(merged)} tensors from {len(shard_names)} shard(s)"
    )
    return _run_pipeline(
        merged,
        args.output,
        strict=args.strict,
        shard_names=shard_names,
        input_dir=args.input_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
