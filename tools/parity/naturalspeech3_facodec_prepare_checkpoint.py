#!/usr/bin/env python3
"""Merge Amphion **NaturalSpeech 3 FACodec** 3-part checkpoint bundle → single .safetensors.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``amphion/naturalspeech3_facodec`` HF repo ships five separate
``torch.save()`` pickle ``.bin`` files at repo root (no ``model.safetensors``
mirror, no ``config.json`` — hparams live in the paper + Amphion source
tree, not in the HF release):

======================================   ========   ====================================
file                                     size       payload
======================================   ========   ====================================
``ns3_facodec_encoder.bin``              ~16.9 MB   encoder v1 (waveform → latent)
``ns3_facodec_encoder_v2.bin``           ~17.1 MB   encoder v2 (improved variant)
``ns3_facodec_decoder.bin``              ~398 MB    decoder v1 + 3 FVQ heads
                                                    (prosody 1cb + content 2cb + detail 3cb)
``ns3_facodec_decoder_v2.bin``           ~432 MB    decoder v2 (pair for encoder_v2)
``ns3_facodec_redecoder.bin``            ~151 MB    redecoder (zero-shot voice conversion —
                                                    swap timbre while preserving
                                                    prosody + content codes)
======================================   ========   ====================================

The Rust converter (``crates/vokra-convert/src/models/naturalspeech3_facodec.rs``)
consumes **single-file safetensors only** (no pickle in the runtime tree, no
multi-file reader). This script bridges the two: for a caller-selected
``--variant`` it loads the corresponding subset of ``.bin`` files via
``torch.load(..., weights_only=True)``, prefixes every key with its role
(``encoder.`` / ``decoder.`` / ``redecoder.``) so a future
``Facodec::from_gguf`` can locate sub-modules, and writes a single
``.safetensors`` the caller feeds to
``vokra-cli convert --model naturalspeech3-facodec``.

Precedent: ``sepformer_prepare_checkpoint.py`` (multi-file cousin — same
shape, same INT-dtype filter, same ``.stripped-manifest.json`` sidecar,
same fail-loud posture) + ``kokoro_prepare_checkpoint.py`` (nested .pth
+ per-voice .pt merge → safetensors) + ``dfn3_prepare_checkpoint.py``
(upstream .pt bridge).

# Usage

::

    uv run --project tools/parity python \\
        tools/parity/naturalspeech3_facodec_prepare_checkpoint.py \\
        --ckpt-dir /path/to/amphion/naturalspeech3_facodec \\
        --variant v2 \\
        --output /tmp/facodec-v2.safetensors \\
        [--allow-strip-any]

# Variant matrix

- ``v1``          = encoder    + decoder      (~415 MB peak resident)
- ``v2``          = encoder_v2 + decoder_v2   (~450 MB, DEFAULT — best-quality)
- ``redecoder-v1`` = encoder + decoder + redecoder    (~566 MB, zero-shot VC)
- ``redecoder-v2`` = encoder_v2 + decoder_v2 + redecoder (~601 MB, zero-shot VC)

All variants comfortably fit on an M1 iMac 16 GB — vast.ai is NOT required
even for the largest (``redecoder-v2`` = ~601 MB peak resident).

# Determinism

Keys are ordered by (variant subset order, per-file dict-iteration order —
Python dict preserves insertion order since 3.7, and
``torch.load(weights_only=True)`` is deterministic). Identical
``--ckpt-dir`` + ``--variant`` input produces byte-identical output
(safetensors serialization is deterministic for fixed key ordering).

# Redistribution

Upstream weight license is ``apache-2.0`` end-to-end — see
``docs/license-audit.md`` §3.1 row "NaturalSpeech 3 FACodec (Amphion)".
The Amphion GitHub LICENSE (``open-mmlab/Amphion/LICENSE``) and HF
``amphion/naturalspeech3_facodec`` cardData both declare apache-2.0.

**Note on redecoder variants**: FACodec itself is a codec (encoder +
decoder + FVQ), NOT a voice-clone trigger model like RVC v2 /
GPT-SoVITS which live in ``vokra-voiceclone-experimental``. However
the redecoder variants specifically enable zero-shot voice conversion
by swapping the timbre subspace while preserving prosody + content
codes. Whether the redecoder variants should be published in the main
``ayutaz/vokra`` org or gated to ``vokra-voiceclone-experimental`` per
the ELVIS Act / NO FAKES Act policy that pushed openvoice_v2 / knn_vc /
freevc / meanvc into the separate repo is an owner routing decision;
this script generates the artifact but does not decide where it lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Variant → ordered list of (stem, role_prefix) tuples the merger walks.
# The stems match the upstream .bin filenames minus the ``ns3_facodec_``
# prefix and ``.bin`` extension; the role prefix is what every state_dict
# key gets namespaced under so a future ``Facodec::from_gguf`` can locate
# the sub-module.
VARIANT_SUBSETS: dict[str, list[tuple[str, str]]] = {
    "v1": [("encoder", "encoder"), ("decoder", "decoder")],
    "v2": [("encoder_v2", "encoder"), ("decoder_v2", "decoder")],
    "redecoder-v1": [
        ("encoder", "encoder"),
        ("decoder", "decoder"),
        ("redecoder", "redecoder"),
    ],
    "redecoder-v2": [
        ("encoder_v2", "encoder"),
        ("decoder_v2", "decoder"),
        ("redecoder", "redecoder"),
    ],
}

# Mirrors the classification in ``nemo_pt_to_safetensors.py`` /
# ``sepformer_prepare_checkpoint.py``. INT dtypes come from BatchNorm
# ``num_batches_tracked`` counters and similar training artefacts —
# safe to strip. Any dtype outside both sets is refused unless
# --allow-strip-any is passed (fail-loud posture: the runtime forward
# path would refuse them anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into dotted-key ``{name: Tensor}``.

    The upstream ``.bin`` files carry flat state dicts, but Lightning-style
    forks may wrap them; this walk handles both without special-casing the
    wrapper name (the unwrap happens in ``_load_bin`` before this walk is
    invoked).
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


def _load_bin(path: Path, role: str) -> dict:
    """Load one .bin state_dict and return a flat ``{role.name: Tensor}``.

    Fail-loud on any load failure OR on a payload that yields zero tensors —
    better a hard exit than a silently-empty prefix (the downstream Rust
    converter would then emit a GGUF with a valid header but no weights and
    the runtime forward would only fail much later at first-forward, which
    is the classic "silent partial" trap this project bans, mirror of the
    sepformer / dfn3 posture).
    """
    import torch

    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"torch.load({path!s}, weights_only=True) failed: {exc}")

    # Common Lightning-style wrapper unwrap. The Amphion release ships flat
    # state dicts, but this defense against fork wrappers costs nothing.
    if isinstance(raw, dict):
        for wrapper in ("state_dict", "model_state_dict", "model", "module"):
            inner = raw.get(wrapper)
            if isinstance(inner, dict) and inner:
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    print(f"  {path.name}: unwrapped ['{wrapper}']")
                    raw = inner
                    break

    flat_local = _flatten("", raw)
    if not flat_local:
        sys.exit(
            f"{path!s} yielded no tensors — expected an Amphion NaturalSpeech3 "
            f"FACodec state_dict (see "
            f"github.com/open-mmlab/Amphion/tree/main/models/codec/ns3_codec)."
        )
    # Namespace under the role prefix so a future ``Facodec::from_gguf`` can
    # locate the sub-module a tensor belongs to (``encoder.conv.weight`` vs.
    # ``decoder.quantizer.codebook.weight`` etc.). Without this prefix an
    # encoder ``weight`` key and a decoder inner ``weight`` key would collide
    # in the merged dict.
    prefixed = {f"{role}.{k}": v for k, v in flat_local.items()}
    return prefixed


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy the
    ``nemo_pt_to_safetensors.py`` / ``sepformer_prepare_checkpoint.py``
    precedents use."""
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
            "Merge Amphion NaturalSpeech 3 FACodec .bin bundle → single "
            ".safetensors for vokra-convert. Variant selects which subset of "
            "the 5 upstream .bin files to merge."
        ),
    )
    ap.add_argument(
        "--ckpt-dir", required=True, type=Path,
        help=(
            "directory holding the 5 upstream .bin files "
            "(ns3_facodec_{encoder,encoder_v2,decoder,decoder_v2,redecoder}.bin) — "
            "typically the output of `huggingface-cli download "
            "amphion/naturalspeech3_facodec --local-dir <dir>`."
        ),
    )
    ap.add_argument(
        "--variant", required=False, default="v2",
        choices=sorted(VARIANT_SUBSETS.keys()),
        help=(
            "which subset of the 5 .bin files to merge (default: v2 = "
            "encoder_v2 + decoder_v2, the best-quality pair). See module "
            "docstring for the full variant matrix."
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
    for stem, role in subset:
        bin_path = ckpt_dir / f"ns3_facodec_{stem}.bin"
        if not bin_path.is_file():
            print(f"required checkpoint missing: {bin_path}", file=sys.stderr)
            return 2
        print(f"  loading {bin_path.name} ({bin_path.stat().st_size:,} bytes) as {role!r}")
        sub = _load_bin(bin_path, role)
        # Duplicate-key guard: the role prefix makes cross-file collision
        # impossible in practice, but this redundant assert catches an
        # accidental within-file duplicate the flatten walk cannot see (a
        # nested dict that re-uses a leaf name under two branches).
        overlap = set(merged) & set(sub)
        if overlap:
            print(
                f"  duplicate keys after prefix (first 5): {sorted(overlap)[:5]}",
                file=sys.stderr,
            )
            return 3
        merged.update(sub)
        per_file_counts[bin_path.name] = len(sub)

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
        f"naturalspeech3_facodec_prepare_checkpoint: variant={args.variant!r}, "
        f"kept {len(kept)}, dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"per-file {per_file_counts}; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
