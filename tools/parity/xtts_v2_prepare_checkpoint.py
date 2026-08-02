#!/usr/bin/env python3
"""Merge Coqui XTTS-v2 `.pth` release bundle → single `.safetensors`.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

Upstream ``coqui/XTTS-v2`` ships weights as a multi-file ``.pth`` bundle::

    model.pth          # GPT-style AR text-to-codebook + HiFi-GAN decoder
    dvae.pth           # Discrete VAE codebook encoder/decoder (~30 MB)
    speakers_xtts.pth  # Frozen reference-speaker embedding table
    config.json        # Model hyperparameters (JSON, not merged into weights)
    vocab.json         # BPE tokenizer vocab (JSON, tokenizer side-car)
    hash.md5           # Upstream integrity check

Weight-only files are torch.save() archives of a flat ``{str: Tensor}`` dict
(Coqui's ``TTS/tts/models/xtts.py::save_checkpoint`` layout — see
``github.com/idiap/coqui-ai-TTS/blob/main/TTS/tts/models/xtts.py``). The
Vokra Rust converter (``crates/vokra-convert/src/models/xtts_v2.rs``)
consumes **safetensors only** (no pickle in the runtime tree). This script
bridges the three .pth files by merging into a single namespaced
safetensors under distinct prefixes.

# Usage

::

    uv run --project tools/parity python tools/parity/xtts_v2_prepare_checkpoint.py \\
        --input-dir /tmp/staging/xtts-v2-src \\
        --output /tmp/xtts-v2.safetensors \\
        [--strict]

Then::

    vokra-cli convert --model xtts-v2 \\
        --input /tmp/xtts-v2.safetensors \\
        --license coqui-public-model-license \\
        --output /tmp/xtts-v2.gguf

# Prefix scheme

Coqui distributes the three sub-models as independent files with
non-overlapping tensor names; this script merges them under distinct
namespaces so ``vokra-cli convert --model xtts-v2`` can address each
compartment via prefix walk (matching the Rust converter's expected
scheme in ``models::xtts_v2``)::

    model.pth:          <keys>       →  model.<keys>
    dvae.pth:           <keys>       →  dvae.<keys>
    speakers_xtts.pth:  <keys>       →  speakers.<keys>

Precedent: ``sepformer_prepare_checkpoint.py`` (multi-file .ckpt bundle),
``dac_prepare_checkpoint.py`` (single .pth + config side-car),
``demucs_prepare_checkpoint.py`` (single .th shared-tensor). This script
is the multi-.pth cousin with a distinct XTTS-v2 twist: three sibling
.pth files that must be merged with prefixed namespacing.

# Pickle-trust posture

Coqui XTTS-v2 model.pth is a Coqui trainer artifact — the outermost
object is a ``{str: Tensor | non_tensor_metadata}`` dict; the
non-tensor metadata (training statistics, optimizer state) can carry
arbitrary pickled objects that ``torch.load(weights_only=True)`` refuses.

**Safety posture**: this script attempts ``weights_only=True`` first; if
the loader raises ``UnpicklingError`` (or any subclass of ``Exception``
tied to a class-blocklist), it falls back to ``weights_only=False`` with
a visible warning. The upstream primary source is Coqui's CPML-licensed
official XTTS-v2 release at ``huggingface.co/coqui/XTTS-v2``; per memory
``[[feedback-license-signoff-primary-source]]`` the pickle-trust
boundary is acknowledged at the point of running this offline sidecar
(the runtime tree never touches pickle — FR-LD-05). Do not run this
script against unverified .pth files from unknown sources.

# Redistribution

Upstream weight license is **Coqui Public Model License (CPML)** —
NonCommercial. See ``docs/license-audit.md`` §3.1 row "XTTS-v2"
(Coqui primary source, owner sign-off = 2026-08-01 yousan T4). Publishing
the produced GGUF requires ``publish-one.sh --allow-noncommercial``.

# Shared-storage handling

XTTS-v2's HiFi-GAN decoder can share the same tensor storage between
mel-spec conditioning and codebook input; the underlying safetensors
serializer hard-errors on ``data_ptr()`` collision. The script dedups
via a ``{data_ptr: first_name}`` map + ``.clone().contiguous()`` on
subsequent references (first tensor stays as-is; each alias becomes an
independent copy). Manifest records the tied pairs for owner audit.

# Fail-loud discipline

If any of the three required .pth files is missing, or if the merged
state_dict produces zero float tensors, the script exits with a
nonzero code and does NOT write an empty safetensors (FR-EX-08: no
silent partial artifacts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Same dtype taxonomy as sepformer / demucs / nemo precedents.
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}

# Sub-bundle layout: (source_filename, output_prefix, required).
# ``required=True`` triggers a fail-loud exit if the file is absent.
# ``speakers_xtts.pth`` is required by the XTTS-v2 Rust converter — a
# speaker-less XTTS-v2 is not a valid zero-shot TTS SKU.
SUB_BUNDLES = (
    ("model.pth", "model", True),
    ("dvae.pth", "dvae", True),
    ("speakers_xtts.pth", "speakers", True),
)

# Wrapper-key probe order — same as demucs. ``"model"`` is FIRST for
# Coqui trainer artifacts (Coqui's Trainer wraps state_dict in
# ``{"model": {...}, "optimizer": ..., "step": ...}``).
WRAPPER_KEYS = ("model", "state_dict", "model_state_dict", "state", "module")


def _looks_like_state_dict(obj: Any) -> bool:
    """Return True if ``obj`` is a ``{str: Tensor}`` dict with ≥1 leaf."""
    import torch
    if not isinstance(obj, dict) or not obj:
        return False
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        # Coqui speakers_xtts.pth may include nested per-speaker dicts —
        # tolerate one level of nesting by requiring at least one direct
        # tensor leaf here.
        if isinstance(v, torch.Tensor):
            return True
    return False


def _extract_state_dict(raw: Any, sub_name: str) -> dict:
    """Unwrap a Coqui .pth payload down to the flat state_dict.

    Probes ``WRAPPER_KEYS`` in order (``"model"`` first — Coqui-canonical).
    Fails loudly on a payload that yields no state_dict rather than
    silently emitting an empty safetensors (FR-EX-08).
    """
    if _looks_like_state_dict(raw):
        return raw

    if isinstance(raw, dict):
        for wrapper in WRAPPER_KEYS:
            inner = raw.get(wrapper)
            if _looks_like_state_dict(inner):
                print(f"  {sub_name}: unwrapped ['{wrapper}']")
                return inner  # type: ignore[return-value]

        # Speakers file may be a flat {speaker_name: embedding_tensor}
        # or {speaker_name: {"speaker_embedding": tensor, ...}} — handle both.
        if sub_name == "speakers":
            flat: dict = {}
            for k, v in raw.items():
                import torch
                if isinstance(v, torch.Tensor):
                    flat[k] = v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            flat[f"{k}.{kk}"] = vv
            if flat:
                print(f"  {sub_name}: flattened {len(flat)} speaker embeddings")
                return flat

    sys.exit(
        f"xtts_v2_prepare_checkpoint: could not locate a state_dict inside "
        f"{sub_name} — expected a Coqui .pth archive with a top-level 'model' "
        f"key (Coqui Trainer.save_checkpoint layout). Top-level keys observed: "
        f"{sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
    )


def _load_pth(path: Path, sub_name: str) -> Any:
    """Load a .pth with safe-first strategy.

    Attempts ``weights_only=True`` first (torch >=2.0 safe-unpickler).
    Coqui's model.pth is a trainer artifact that may embed non-tensor
    metadata (optimizer state, training step counters) — if the safe
    loader refuses, fall back to ``weights_only=False`` with a visible
    warning. The pickle-trust boundary is acknowledged in the module
    docstring; callers should have verified the upstream source
    (``coqui/XTTS-v2`` HF repo per §3.1 sign-off) before running.
    """
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as safe_err:  # noqa: BLE001 — any refusal triggers fallback
        print(
            f"  {sub_name}: weights_only=True refused ({type(safe_err).__name__}: "
            f"{str(safe_err)[:80]}); falling back to weights_only=False "
            f"(pickle-trust: upstream = coqui/XTTS-v2, §3.1 sign-off required)",
            file=sys.stderr,
        )
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _partition_and_dedup(sd: dict, seen: dict[int, str], strict: bool):
    """Split into (kept, dropped_int, unknown_other, shared_pairs).

    ``seen`` is the merged-across-bundles data_ptr dict so shared storage
    is deduped even across the three .pth sibling files (Coqui shares
    codebook embeddings between model.pth and dvae.pth in some releases).
    """
    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    shared_pairs: list[tuple[str, str]] = []

    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)

        if dtype_s in KEEP_DTYPES:
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


def _run_pipeline(bundles: dict[str, Any], output: Path, strict: bool) -> int:
    """Merge state_dicts from all sub-bundles under distinct prefixes.

    Shared body used by both ``main`` and ``--self-test`` so the two
    paths cannot drift.
    """
    from safetensors.torch import save_file

    merged: dict = {}
    all_dropped: list = []
    all_unknown: list = []
    all_shared_pairs: list = []
    seen_ptrs: dict[int, str] = {}
    per_bundle_stats: list[tuple[str, int, int, int, int]] = []

    for sub_name, prefix, _required in SUB_BUNDLES:
        if sub_name not in bundles:
            continue
        raw = bundles[sub_name]
        state = _extract_state_dict(raw, sub_name)
        kept, dropped, unknown, shared_pairs = _partition_and_dedup(
            state, seen_ptrs, strict
        )
        for k, v in kept.items():
            merged[f"{prefix}.{k}"] = v
        all_dropped.extend((sub_name, n, d, s) for n, d, s in dropped)
        all_unknown.extend((sub_name, n, d, s) for n, d, s in unknown)
        all_shared_pairs.extend(
            (f"{prefix}.{a}", f"{prefix}.{b}") for a, b in shared_pairs
        )
        per_bundle_stats.append(
            (sub_name, len(kept), len(dropped), len(unknown), len(shared_pairs))
        )

    if not merged:
        print(
            "xtts_v2_prepare_checkpoint: refusing to write empty safetensors — "
            "no float tensors survived merge (FR-EX-08).",
            file=sys.stderr,
        )
        return 4

    if all_unknown and strict:
        first = list(all_unknown[:3])
        print(
            f"xtts_v2_prepare_checkpoint: --strict refusing to drop "
            f"{len(all_unknown)} tensors of unknown dtype (first 3: {first}); "
            "re-run without --strict if verified inference-inert.",
            file=sys.stderr,
        )
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output))
    written_bytes = output.stat().st_size

    manifest = {
        "source_bundles": [b for b, *_ in per_bundle_stats],
        "per_bundle": [
            {
                "file": b,
                "kept": k,
                "dropped_int": d,
                "unknown_other": u,
                "shared_cloned": s,
            }
            for b, k, d, u, s in per_bundle_stats
        ],
        "total_kept_tensors": len(merged),
        "total_dropped_int": len(all_dropped),
        "total_unknown_other": len(all_unknown),
        "total_shared_cloned": len(all_shared_pairs),
        "output_bytes": written_bytes,
        "prefix_scheme": {sub: pre for sub, pre, _ in SUB_BUNDLES},
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print()
    print(f"kept={len(merged)} bundles={len(per_bundle_stats)}")
    for b, k, d, u, s in per_bundle_stats:
        print(f"  {b:24s} kept={k:5d} dropped_int={d:4d} unknown={u:3d} shared_cloned={s:3d}")
    print(f"wrote {written_bytes} bytes to {output}")
    print(f"manifest: {manifest_path}")
    return 0


def _self_test() -> int:
    """Argparse + torch/safetensors import smoke; synthetic three-bundle merge.

    Verifies the pipeline logic without requiring real Coqui .pth files:
    creates in-memory dicts that mimic the model/dvae/speakers layouts
    and confirms the merge + prefix + dedup + write path all succeed.
    """
    import tempfile

    import torch
    from safetensors.torch import safe_open

    # Prove argparse accepts the documented surface.
    p = _build_parser()
    for argv in (
        ["--input-dir", "/nonexistent", "--output", "/tmp/x.safetensors"],
        ["--input-dir", "/nonexistent", "--output", "/tmp/x.safetensors", "--strict"],
        ["--self-test"],
    ):
        p.parse_args(argv)

    # Synthetic 3-bundle payload with a deliberate cross-bundle shared
    # tensor to exercise the dedup path.
    shared = torch.randn(4, 4)
    model_sd = {
        "model": {
            "gpt.attn.weight": torch.randn(8, 8),
            "hifigan.conv1.weight": torch.randn(3, 3),
            "shared_codebook": shared,  # shared with dvae
            "num_batches_tracked": torch.tensor(42, dtype=torch.int64),  # drop
        }
    }
    dvae_sd = {
        "model": {
            "encoder.conv1.weight": torch.randn(2, 2),
            "codebook": shared,  # same storage — will clone
            "step": torch.tensor(100, dtype=torch.int64),  # drop
        }
    }
    speakers_sd = {
        "female_1": torch.randn(512),
        "male_1": torch.randn(512),
    }
    bundles = {
        "model.pth": model_sd,
        "dvae.pth": dvae_sd,
        "speakers_xtts.pth": speakers_sd,
    }

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.safetensors"
        rc = _run_pipeline(bundles, out, strict=False)
        if rc != 0:
            print("self-test: FAIL (_run_pipeline nonzero)", file=sys.stderr)
            return 1
        # Confirm the safetensors file is readable and contains the
        # expected prefixed keys.
        with safe_open(str(out), framework="pt") as f:
            keys = set(f.keys())

    expected_prefixes = {"model.", "dvae.", "speakers."}
    for pre in expected_prefixes:
        if not any(k.startswith(pre) for k in keys):
            print(f"self-test: FAIL (missing prefix '{pre}')", file=sys.stderr)
            return 1
    # int64 tensors must have been dropped.
    if any("num_batches_tracked" in k or k.endswith(".step") for k in keys):
        print("self-test: FAIL (int64 counter not stripped)", file=sys.stderr)
        return 1
    # Speaker embeddings under speakers. prefix.
    if not any(k.startswith("speakers.female_1") for k in keys):
        print("self-test: FAIL (speaker embedding missing)", file=sys.stderr)
        return 1

    print("xtts_v2_prepare_checkpoint self-test: OK")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge Coqui XTTS-v2 .pth bundle (model + dvae + speakers) into a "
            "single Vokra-consumable .safetensors. Sibling of "
            "demucs/sepformer/dac_prepare_checkpoint.py; see the module "
            "docstring for the CPML redistribution posture."
        ),
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing model.pth, dvae.pth, and speakers_xtts.pth "
            "(e.g. the local_dir passed to snapshot_download('coqui/XTTS-v2'))."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Path to the merged .safetensors file to write.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail loudly on any non-float, non-int dtype in the source "
            "(default: silently skip). Use when the source is a known-good "
            "release and you want assurance nothing surprising was dropped."
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run an internal self-test with synthetic three-bundle payloads "
            "and confirm the merge + prefix + dedup pipeline. Exits 0 on "
            "success, nonzero on failure."
        ),
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.self_test:
        return _self_test()

    if args.input_dir is None or args.output is None:
        print(
            "xtts_v2_prepare_checkpoint: --input-dir and --output are required "
            "(use --self-test for smoke).",
            file=sys.stderr,
        )
        return 2

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(
            f"xtts_v2_prepare_checkpoint: --input-dir '{input_dir}' is not a "
            "directory.",
            file=sys.stderr,
        )
        return 2

    bundles: dict[str, Any] = {}
    missing: list[str] = []
    for sub_name, _prefix, required in SUB_BUNDLES:
        p = input_dir / sub_name
        if p.exists():
            print(f"loading {sub_name} ...")
            bundles[sub_name] = _load_pth(p, sub_name)
        elif required:
            missing.append(sub_name)

    if missing:
        print(
            f"xtts_v2_prepare_checkpoint: --input-dir '{input_dir}' is missing "
            f"required file(s): {missing}. Expected the full Coqui XTTS-v2 "
            "release bundle from huggingface.co/coqui/XTTS-v2.",
            file=sys.stderr,
        )
        return 2

    return _run_pipeline(bundles, args.output, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
