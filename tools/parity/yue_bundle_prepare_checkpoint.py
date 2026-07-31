#!/usr/bin/env python3
"""Merge YuE bundle (upsampler / xcodec-mini) checkpoint → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The YuE full-song music-generation system (Yuan et al. 2025,
arXiv:2503.08638, ``github.com/multimodal-art-projection/YuE``) publishes
its **codec / vocoder half** across two sibling HF repos, each of which
ships **only torch pickle** (``.pth`` / ``.bin``) — there is no
``model.safetensors`` mirror on either side (verified 2026-08-01 via
``https://huggingface.co/api/models/m-a-p/YuE-upsampler`` and
``https://huggingface.co/api/models/m-a-p/xcodec_mini_infer``):

===================================================   ======   ============================================
Variant                                              Size     Payload
===================================================   ======   ============================================
``upsampler``   (`m-a-p/YuE-upsampler`)              145 MB   Vocos backbone + iSTFT head — 44.1 kHz
                                                              vocoder decoding YuE codec latents
``xcodec-mini`` (`m-a-p/xcodec_mini_infer`)          2.2 GB   SoundStream RVQ codec + HuBERT-base
                                                              semantic encoder + Vocos decoder head
===================================================   ======   ============================================

The Rust converter (``crates/vokra-convert/src/models/yue_bundle.rs``)
consumes **single-file safetensors only** (no pickle in the runtime
tree, no multi-file reader). This script bridges the two: for a
caller-selected ``--variant`` it loads the corresponding sub-parts,
prefixes every key with its role (``codec.`` / ``semantic.`` /
``decoder.`` for xcodec-mini; bare ``backbone.`` / ``head.`` for
upsampler because there is only one sub-module) so a future
``YueXcodecMini::from_gguf`` can locate sub-modules, and writes a
single ``.safetensors`` the caller feeds to
``vokra-cli convert --model {yue-upsampler,yue-xcodec-mini}``.

Precedent: ``naturalspeech3_facodec_prepare_checkpoint.py`` (multi-file
cousin — same shape, same INT-dtype filter, same
``.stripped-manifest.json`` sidecar, same fail-loud posture) +
``sepformer_prepare_checkpoint.py`` (SpeechBrain ``.ckpt`` bundle) +
``bin_to_safetensors.py`` (single-file pickle → safetensors).

# Usage

::

    # Vocoder half (145 MB)
    uv run --project tools/parity python \\
        tools/parity/yue_bundle_prepare_checkpoint.py \\
        --ckpt-dir /path/to/m-a-p--YuE-upsampler \\
        --variant upsampler \\
        --snapshot 151000 \\
        --output /tmp/yue-upsampler.safetensors \\
        [--allow-strip-any]

    # Codec bundle (~1.88 GB unique weights)
    uv run --project tools/parity python \\
        tools/parity/yue_bundle_prepare_checkpoint.py \\
        --ckpt-dir /path/to/m-a-p--xcodec_mini_infer \\
        --variant xcodec-mini \\
        --snapshot 151000 \\
        --output /tmp/yue-xcodec-mini.safetensors \\
        [--allow-strip-any]

Both variants comfortably fit on an M1 iMac 16 GB — vast.ai is NOT
required (per memory ``[[feedback-large-models-on-vast-ai]]`` the
≥8 GB threshold does not fire; total unique weights = 1.88 GB).

# Snapshot selection (131k vs 151k)

Both upstream repos ship two training snapshots of the Vocos decoder
head: ``decoder_131000.pth`` and ``decoder_151000.pth``. These files
are **byte-identical** between the two sibling repos (same xet
content-address hash: ``c030b262…`` for 131k / ``70e4fbd9…`` for
151k, verified 2026-08-01 via HF API). The 151k snapshot is the
later training step and is typically the "final" one — default here.

# Determinism

Keys are ordered by (variant subset order, per-file dict-iteration
order — Python dict preserves insertion order since 3.7, and
``torch.load(weights_only=True)`` is deterministic). Identical
``--ckpt-dir`` + ``--variant`` + ``--snapshot`` input produces
byte-identical output (safetensors serialization is deterministic
for fixed key ordering).

# Sub-tree skip list (xcodec-mini)

The upstream ``m-a-p/xcodec_mini_infer`` repo ships full source-tree
copies of RepCodec (ByteDance/Chutong Meng, MIT) and Descript-Audio-
Codec (MIT) at ``RepCodec/`` and ``descriptaudiocodec/dac/``. These
are inference-tree artefacts of the upstream release process — they
are **not** loaded weights and this bridge does NOT recurse into them.
NOTICE credit is preserved (their code informed the YuE codec
design), but no tensors are lifted from them. Same policy for the
``__pycache__/`` bytecode cache leaked into the release.

# Redistribution

Both variants ship apache-2.0 end-to-end — see
``docs/license-audit.md`` §3.1 rows "YuE-upsampler" and
"YuE xcodec-mini", both ☑ Commercial as of 2026-08-01. Upstream YuE
code at ``github.com/multimodal-art-projection/YuE`` LICENSE is also
apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Which upstream files each --variant reads, and under what role
# prefix they land in the merged safetensors. The upsampler variant
# has only one sub-module (Vocos decoder head), so its role prefix
# is empty ("" = bare `backbone.*` / `head.*` names, matching the
# sibling `charactr/vocos-mel-24khz` and `charactr/vocos-encodec-24khz`
# tensor-name contract). The xcodec-mini variant has three
# sub-modules (SoundStream codec, HuBERT semantic encoder, Vocos
# decoder head), so each gets a distinct role prefix.
#
# Each tuple = (relative_path_template, role_prefix).
# {snapshot} is substituted with the --snapshot argument (131000 or
# 151000) for the .pth files that come in two training snapshots.
VARIANT_SUBSETS: dict[str, list[tuple[str, str]]] = {
    "upsampler": [
        ("decoder_{snapshot}.pth", ""),  # bare backbone.* / head.* names
    ],
    "xcodec-mini": [
        # SoundStream RVQ codec generator (Lightning wrapper — the
        # loader unwraps `state_dict` / `generator` below if present).
        ("final_ckpt/ckpt_00360000.pth", "codec"),
        # HuBERT-base semantic encoder (HF-transformers pickle).
        ("semantic_ckpts/hf_1_325000/pytorch_model.bin", "semantic"),
        # Vocos decoder head (byte-identical to the sibling repo's
        # decoder_{snapshot}.pth — same xet content-address hash).
        ("decoders/decoder_{snapshot}.pth", "decoder"),
    ],
}

# Mirrors the classification in ``nemo_pt_to_safetensors.py`` /
# ``sepformer_prepare_checkpoint.py`` /
# ``naturalspeech3_facodec_prepare_checkpoint.py``. INT dtypes come
# from BatchNorm ``num_batches_tracked`` counters and similar training
# artefacts — safe to strip. Any dtype outside both sets is refused
# unless --allow-strip-any is passed (fail-loud posture: the runtime
# forward path would refuse them anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into dotted-key ``{name: Tensor}``.

    The upstream files carry mostly-flat state dicts, but the
    SoundStream ``ckpt_00360000.pth`` file is Lightning-wrapped (it
    contains ``generator`` / ``discriminator`` / ``optimizer_states``
    top-level keys); the caller-visible unwrap happens in ``_load_one``
    before this walk is invoked, but this recursive flatten still
    protects against any residual nesting.
    """
    import torch

    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(key, v))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    # else: silently drop non-Tensor scalars/arrays — safetensors would
    # refuse them anyway; the runtime doesn't need them.
    return out


def _load_one(path: Path, role: str) -> dict:
    """Load one .pth / .bin state_dict and return a flat ``{role.name: Tensor}``.

    Fail-loud on any load failure OR on a payload that yields zero tensors —
    better a hard exit than a silently-empty prefix (the downstream Rust
    converter would then emit a GGUF with a valid header but no weights and
    the runtime forward would only fail much later at first-forward, which
    is the classic "silent partial" trap this project bans, mirror of the
    sepformer / facodec / dfn3 posture).

    Handles Lightning wrappers by walking a fixed list of known outer
    keys (``state_dict`` / ``model_state_dict`` / ``model`` / ``module`` /
    ``generator``) — the SoundStream ``ckpt_00360000.pth`` in
    xcodec_mini_infer specifically wraps its generator under
    ``{"generator": {...}, "discriminator": {...}, "optimizer_states":
    [...], ...}`` and we want the generator half only (discriminator +
    optimizer state are training artefacts inference does not use).
    """
    import torch

    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"torch.load({path!s}, weights_only=True) failed: {exc}")

    # Common Lightning-style wrapper unwrap. We walk in priority order —
    # `generator` first because the SoundStream ckpt is our known
    # "please lift only the generator, ignore discriminator + optimizer"
    # case. The other wrapper names are defense against upstream forks.
    if isinstance(raw, dict):
        for wrapper in ("generator", "state_dict", "model_state_dict", "model", "module"):
            inner = raw.get(wrapper)
            if isinstance(inner, dict) and inner:
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    print(f"  {path.name}: unwrapped [{wrapper!r}]")
                    raw = inner
                    break

    flat_local = _flatten("", raw)
    if not flat_local:
        sys.exit(
            f"{path!s} yielded no tensors — expected an m-a-p YuE bundle "
            f"state_dict (see github.com/multimodal-art-projection/YuE)."
        )
    # Namespace under the role prefix so a future ``YueXcodecMini::from_gguf``
    # can locate the sub-module a tensor belongs to (``codec.encoder.conv.weight``
    # vs. ``semantic.encoder.layer_norm.weight`` vs. ``decoder.head.out.weight``).
    # For the upsampler variant the role is empty ("") = bare backbone.* /
    # head.* names (matches the sibling Charactr AI Vocos tensor-name
    # contract so a future runtime binder can share the loader path).
    if role:
        prefixed = {f"{role}.{k}": v for k, v in flat_local.items()}
    else:
        prefixed = dict(flat_local)
    return prefixed


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy the
    ``nemo_pt_to_safetensors.py`` / ``sepformer_prepare_checkpoint.py`` /
    ``naturalspeech3_facodec_prepare_checkpoint.py`` precedents use."""
    kept: dict = {}
    dropped: list[tuple[str, str, list[int]]] = []
    unknown: list[tuple[str, str, list[int]]] = []
    for name, t in sd.items():
        if not hasattr(t, "dtype") or not hasattr(t, "shape"):
            continue
        dtype_s = str(t.dtype)
        if dtype_s in KEEP_DTYPES:
            if hasattr(t, "contiguous"):
                t = t.contiguous()
            if hasattr(t, "detach"):
                t = t.detach()
            kept[name] = t
        elif dtype_s in INT_DTYPES:
            dropped.append((name, dtype_s, list(t.shape)))
        else:
            unknown.append((name, dtype_s, list(t.shape)))
    return kept, dropped, unknown


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Merge YuE bundle (upsampler or xcodec-mini) upstream pickle "
            "checkpoint(s) → single .safetensors for vokra-convert. "
            "Variant selects which HF repo layout to expect."
        ),
    )
    ap.add_argument(
        "--ckpt-dir", required=True, type=Path,
        help=(
            "directory holding the upstream files — typically the output of "
            "`huggingface-cli download m-a-p/YuE-upsampler --local-dir <dir>` "
            "or `huggingface-cli download m-a-p/xcodec_mini_infer --local-dir <dir>`."
        ),
    )
    ap.add_argument(
        "--variant", required=True,
        choices=sorted(VARIANT_SUBSETS.keys()),
        help=(
            "which of the two YuE bundle variants to merge: "
            "'upsampler' = m-a-p/YuE-upsampler (Vocos vocoder head, 145 MB); "
            "'xcodec-mini' = m-a-p/xcodec_mini_infer (SoundStream RVQ codec + "
            "HuBERT semantic encoder + Vocos decoder head bundle, ~1.88 GB)."
        ),
    )
    ap.add_argument(
        "--snapshot", required=False, default="151000",
        choices=("131000", "151000"),
        help=(
            "which Vocos decoder training snapshot to pick "
            "(default: 151000 = later training step, typically the 'final' one). "
            "The two snapshots are byte-identical between the two sibling repos."
        ),
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="destination .safetensors path (parent will be mkdir'd).",
    )
    ap.add_argument(
        "--allow-strip-any", action="store_true",
        help="also strip fp64 / complex tensors (default: refuse them loudly).",
    )
    args = ap.parse_args()

    try:
        from safetensors.torch import save_file
        import torch  # noqa: F401
    except ImportError as exc:
        print(
            f"missing dep {exc}. run: uv sync (from tools/parity/)",
            file=sys.stderr,
        )
        return 2

    ckpt_dir: Path = args.ckpt_dir
    if not ckpt_dir.is_dir():
        print(f"--ckpt-dir must be an existing directory: {ckpt_dir}", file=sys.stderr)
        return 2

    subset = VARIANT_SUBSETS[args.variant]

    merged: dict = {}
    per_file_counts: dict[str, int] = {}
    for rel_template, role in subset:
        rel = rel_template.format(snapshot=args.snapshot)
        f_path = ckpt_dir / rel
        if not f_path.is_file():
            print(
                f"required checkpoint missing: {f_path} "
                f"(variant={args.variant!r}, snapshot={args.snapshot!r})",
                file=sys.stderr,
            )
            return 2
        print(
            f"  loading {rel} ({f_path.stat().st_size:,} bytes)"
            f"{' as role ' + role!r if role else ' (no role prefix)'}"
        )
        sub = _load_one(f_path, role)
        # Duplicate-key guard: the role prefix makes cross-file collision
        # impossible in practice for xcodec-mini, but this redundant assert
        # catches an accidental within-file duplicate the flatten walk
        # cannot see (a nested dict that re-uses a leaf name under two
        # branches). For the upsampler variant which has only one file,
        # this is a no-op.
        overlap = set(merged) & set(sub)
        if overlap:
            print(
                f"  duplicate keys after prefix (first 5): {sorted(overlap)[:5]}",
                file=sys.stderr,
            )
            return 3
        merged.update(sub)
        per_file_counts[rel] = len(sub)

    kept, dropped, unknown = _partition(merged, args.allow_strip_any)

    if unknown and not args.allow_strip_any:
        first = [(n, d, s) for n, d, s in unknown[:3]]
        print(
            f"refusing to drop {len(unknown)} tensors of unknown dtype "
            f"(first 3: {first}); re-run with --allow-strip-any if verified "
            f"inference-inert.",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(args.output))

    manifest = {
        "input_dir": str(ckpt_dir),
        "variant": args.variant,
        "snapshot": args.snapshot,
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "per_file": per_file_counts,
        "dropped_tensors": [
            {"name": n, "dtype": d, "shape": s} for n, d, s in dropped
        ],
        "unknown_stripped": (
            [{"name": n, "dtype": d, "shape": s} for n, d, s in unknown]
            if args.allow_strip_any else []
        ),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".stripped-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"yue_bundle_prepare_checkpoint: variant={args.variant!r}, "
        f"snapshot={args.snapshot!r}, kept {len(kept)}, "
        f"dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"per-file {per_file_counts}; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
