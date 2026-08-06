#!/usr/bin/env python3
"""Bridge SpeechBrain SGMSE-VoiceBank ``score_model_ema.ckpt`` → single ``.safetensors``.

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

# Why this exists

The upstream ``speechbrain/sgmse-voicebank`` HF repo ships a
``hyperparams.yaml`` (SpeechBrain SGMSEEnhancement pretrainer config) + a
single ``score_model_ema.ckpt`` (~250 MiB torch pickle, EMA weights of the
internal NCSN++ v2 score network). The Rust converter
(``crates/vokra-convert/src/models/sgmse.rs``) consumes **safetensors only**
(no pickle in the runtime tree — FR-LD-05). This script bridges the two.

Unlike the SepFormer 3-part bundle (``encoder.ckpt`` / ``decoder.ckpt`` /
``masknet.ckpt``), SGMSE's checkpoint is a **single flat state_dict**: the
upstream pretrainer maps ``score_model_ema.ckpt`` → the ``score_model`` module
at load time, but the ``.ckpt`` file itself carries no ``score_model.``
prefix on its keys (the state_dict is the internal NCSN++ v2 network
directly). We preserve that flat layout so a future
``Sgmse::from_gguf`` walks the same NCSN++ v2 tensor names
(``input_layer.weight``, ``blocks.0.norm1.weight``, etc.) the upstream
``sgmse`` code reference uses.

# Layout

=========================   ==========  =========================================
file                        size        payload
=========================   ==========  =========================================
``score_model_ema.ckpt``    ~250 MiB    NCSN++ v2 EMA state_dict (torch pickle)
``hyperparams.yaml``        ~1 KiB      SpeechBrain pretrainer config (advisory)
=========================   ==========  =========================================

Only ``score_model_ema.ckpt`` is bridged; ``hyperparams.yaml`` is a
SpeechBrain-side pretrainer config the runtime binder consumes separately
(the numerical hparams — theta, sigma_min, sigma_max, N, corrector iterations,
SNR, STFT n_fft/hop — will land as ``vokra.sgmse.*`` GGUF chunks in the
follow-up runtime wave, not through this bridge).

Precedent: ``nemo_pt_to_safetensors.py`` (single-file .pt/.nemo → safetensors,
handles ``.ckpt`` uniformly) + ``sepformer_prepare_checkpoint.py``
(SpeechBrain-family torch_recovery unwrap). This script is the single-file
cousin — same INT-dtype filter, same ``.stripped-manifest.json`` sidecar,
same fail-loud posture.

# Usage

::

    uv run --project tools/parity python tools/parity/sgmse_prepare_checkpoint.py \\
        --ckpt /path/to/sgmse-voicebank/score_model_ema.ckpt \\
        --output /tmp/sgmse-voicebank.safetensors \\
        [--allow-strip-any]

# Determinism

Keys are ordered by ``torch.load`` state_dict iteration order (Python dict
preserves insertion order since 3.7, and ``torch.load(weights_only=True)``
is deterministic). Identical ``--ckpt`` input produces byte-identical
output (safetensors serialization is deterministic for fixed key ordering).

# Redistribution

Upstream weight license is ``apache-2.0`` (SpeechBrain family) — see
``docs/license-audit.md`` §3.1 row "SGMSE-VoiceBank
(``speechbrain/sgmse-voicebank``)", ☑ Commercial 2026-08-04 yousan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Mirrors the classification in ``nemo_pt_to_safetensors.py`` +
# ``sepformer_prepare_checkpoint.py``. INT dtypes come from BatchNorm
# ``num_batches_tracked`` counters and similar training artefacts — safe to
# strip. Any dtype outside both sets is refused unless --allow-strip-any is
# passed (fail-loud posture: the runtime forward path would refuse them
# anyway).
INT_DTYPES = {
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.uint8", "torch.uint16", "torch.uint32", "torch.uint64",
    "torch.bool",
}
KEEP_DTYPES = {"torch.float32", "torch.float16", "torch.bfloat16"}


def _flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into dotted-key ``{name: Tensor}``.

    SpeechBrain's ``torch_recovery`` normally pickles a flat state_dict, but
    some Lightning-style forks / EMA wrappers wrap it as
    ``{"state_dict": {...}}`` or ``{"ema_state_dict": {...}}`` — this walk
    handles both without special-casing the wrapper name (the unwrap
    happens in ``_load_ckpt`` before this walk is invoked).
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
    # refuse them anyway; the runtime doesn't need them (EMA decay float,
    # step counter, etc).
    return out


def _load_ckpt(path: Path) -> dict:
    """Load ``score_model_ema.ckpt`` and return a flat ``{name: Tensor}``.

    Fail-loud on any load failure OR on a payload that yields zero tensors —
    better a hard exit than a silently-empty GGUF with a valid header but no
    weights (the classic "silent partial" trap this project bans).
    """
    import torch

    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"torch.load({path!s}, weights_only=True) failed: {exc}")

    # Common Lightning-style / EMA-wrapper unwrap. SpeechBrain's default
    # ``torch_recovery`` writes the flat state_dict; forks that wrap it are
    # rare but do exist in the wild. We deliberately do NOT prepend a
    # `score_model.` prefix — the upstream pretrainer adds that at load
    # time, but the .ckpt file itself carries the internal NCSN++ v2
    # network's names verbatim, and the Rust converter preserves them.
    if isinstance(raw, dict):
        for wrapper in ("state_dict", "model_state_dict", "model", "module", "ema_state_dict"):
            inner = raw.get(wrapper)
            if isinstance(inner, dict) and inner:
                sample = next(iter(inner.values()), None)
                if hasattr(sample, "dtype") and hasattr(sample, "shape"):
                    print(f"  unwrapped ['{wrapper}']")
                    raw = inner
                    break

    flat = _flatten("", raw)
    if not flat:
        sys.exit(
            f"{path!s} yielded no tensors — expected an NCSN++ v2 state_dict "
            f"pickled by SpeechBrain's SGMSE pretrainer (see "
            f"upstream `hyperparams.yaml` `score_model:` block)."
        )
    return flat


def _partition(sd: dict, allow_strip_any: bool):
    """Split into ``(kept, dropped_int, unknown_other)`` — same taxonomy the
    ``nemo_pt_to_safetensors.py`` + ``sepformer_prepare_checkpoint.py``
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
            "Bridge SpeechBrain SGMSE-VoiceBank score_model_ema.ckpt → single .safetensors"
        ),
    )
    ap.add_argument(
        "--ckpt", required=True, type=Path,
        help=(
            "path to score_model_ema.ckpt — typically fetched via "
            "`huggingface-cli download speechbrain/sgmse-voicebank score_model_ema.ckpt`."
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

    ckpt: Path = args.ckpt
    if not ckpt.is_file():
        print(f"--ckpt must be an existing file: {ckpt}", file=sys.stderr)
        return 2

    print(f"  loading {ckpt.name} ({ckpt.stat().st_size:,} bytes)")
    sd = _load_ckpt(ckpt)

    kept, dropped, unknown = _partition(sd, args.allow_strip_any)

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
        "input": str(ckpt),
        "output": str(args.output),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
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
        f"sgmse_prepare_checkpoint: kept {len(kept)}, "
        f"dropped {len(dropped)} int, "
        f"stripped {len(unknown) if args.allow_strip_any else 0} unknown; "
        f"manifest -> {manifest_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
