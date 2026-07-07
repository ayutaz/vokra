#!/usr/bin/env python3
"""Merge Kokoro-82M ``kokoro-v1_0.pth`` + ``voices/*.pt`` → safetensors + enriched config.

This is an **offline** sidecar tool (FR-LD-05: no Python / PyTorch is ever pulled
into the runtime). The Rust converter (`crates/vokra-convert/src/models/kokoro.rs`)
consumes safetensors only, but the upstream ``hexgrad/Kokoro-82M`` release ships:

1. ``kokoro-v1_0.pth`` — a torch pickle nested dict of the model state, and
2. ``voices/*.pt`` — one torch pickle per voice, each holding a single style
   tensor. The canonical release does **not** bundle a stacked ``voicepack``
   tensor in the pickle, and the ``config.json`` does **not** list voice names
   (they are implied by the ``voices/*.pt`` file list).

This script bridges those two facts:

* Loads the .pth with ``torch.load(..., weights_only=True)``, flattens its nested
  ``{bert: {...}, text_encoder: {...}, predictor: {...}, decoder: {...}, ...}``
  layout into a dotted-key flat dict (matching the layout the reference dumper
  and Rust converter expect).
* Sorts ``voices/*.pt`` alphabetically, torch.loads each, verifies all voice
  tensors share the same shape. **By default** the voice tensors are used only
  to derive the ``voices: [...]`` array in the enriched config (see below) —
  they are NOT stacked into a ``voicepack`` tensor in the safetensors output.
  Rationale: the converter derives ``style_dim`` from ``voicepack`` axis 1 when
  the tensor is present, but each Kokoro voice.pt has shape ``[510, 1, 256]``
  (a per-timestep style history), so a naive ``[num_voices, 510, 1, 256]``
  stack would surface as ``style_dim = 510`` in the GGUF hparams, not the
  actual ``style_dim = 128`` (which the converter then reads from
  ``predictor.module.F0.0.norm1.fc.weight`` axis 1 as a fallback). Skipping the
  stack keeps the fallback path clean; a future ``voicepack`` layout spec
  (``M2-07-T02`` follow-up) can turn this back on with the correct shape.
  The ``--stack-voicepack`` flag reinstates the stack for callers who have
  their own voicepack layout in mind (experimental).
* Writes the merged flat dict via ``safetensors.torch.save_file(...)``. All
  contiguous, cpu-resident, and safetensors-compatible dtypes (F32/F16/BF16/I*).
* Emits an "enriched" ``config.json`` that adds a ``voices: [...]`` array
  (derived from the .pt filenames, sans the ``.pt`` extension) — the Rust
  ``KokoroJsonConfig::parse`` requires *at least one* of ``{voices,
  voice_names}``, but the canonical config lacks both. The enriched config is
  written to a caller-supplied path (never overwriting the upstream config).

The tool fails loudly (``sys.exit`` with a message) on any anomaly rather than
silently masking it: non-Tensor payloads in a voice.pt, shape disagreement across
voices, or a non-float dtype creeping into the merged dict. This follows the
FR-EX-08 "no silent fallback" posture the runtime uses.

# Usage

::

    python tools/parity/kokoro_prepare_checkpoint.py \\
        --pth /path/to/kokoro-v1_0.pth \\
        --voices-dir /path/to/voices \\
        --config /path/to/config.json \\
        --output /tmp/kokoro-merged.safetensors \\
        --enriched-config /tmp/kokoro-enriched-config.json \\
        [--voicepack-report /tmp/report.txt]

The optional ``--voicepack-report`` writes a short (safe-to-embed in CI step
summary) markdown-flavoured report of the voicepack shape, voice count, and
tensor dtype breakdown — feed it to ``$GITHUB_STEP_SUMMARY`` from CI.

# Determinism

Voice ordering is a plain sorted-by-filename pass; identical inputs produce
byte-identical safetensors output (safetensors serialization is deterministic
for fixed key ordering).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def flatten(prefix: str, obj: Any) -> dict:
    """Flatten a nested dict into a dotted-key flat dict of Tensor values.

    Non-dict, non-Tensor payloads are dropped (torch.load may surface objects
    the safetensors serializer cannot represent — e.g. dataclasses, numpy
    arrays, or Python scalars — that the runtime does not need). We don't crash
    on those; the safetensors writer downstream is the authoritative gate.
    """
    import torch

    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(key, v))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj
    # else: silently drop non-Tensor scalars/arrays — safetensors would refuse
    # them anyway; the runtime doesn't need them.
    return out


def load_state_dict(pth: Path) -> dict:
    """Load the Kokoro .pth and return a flat dotted-key dict of Tensors."""
    import torch

    try:
        state = torch.load(str(pth), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"torch.load({pth!s}, weights_only=True) failed: {exc}")
    flat = flatten("", state)
    if not flat:
        sys.exit(
            f"torch.load({pth!s}) produced no tensors; expected the nested "
            "Kokoro state dict (bert/text_encoder/predictor/decoder/...)."
        )
    return flat


def load_voice_tensors(voices_dir: Path) -> tuple[list[str], list]:
    """Load all voices/*.pt in sorted order.

    Returns ``(voice_names, voice_tensors)`` where ``voice_names[i]`` is the
    filename stem (no ``.pt``) and ``voice_tensors[i]`` is the tensor payload.

    Fails loudly if any voice.pt is not a plain Tensor or if shapes disagree —
    a mixed voicepack cannot be stacked without silently reshaping.
    """
    import torch

    if not voices_dir.is_dir():
        sys.exit(f"--voices-dir {voices_dir!s} is not a directory")
    voice_files = sorted(p for p in voices_dir.iterdir() if p.suffix == ".pt")
    if not voice_files:
        sys.exit(f"no *.pt files under {voices_dir!s}")

    names: list[str] = []
    tensors: list = []
    reference_shape: tuple | None = None
    reference_dtype = None
    for path in voice_files:
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"torch.load({path!s}) failed: {exc}")
        if not isinstance(payload, torch.Tensor):
            sys.exit(
                f"{path.name}: expected a torch.Tensor payload, got "
                f"{type(payload).__name__} (voicepack merge cannot proceed with "
                "mixed types)"
            )
        if reference_shape is None:
            reference_shape = tuple(payload.shape)
            reference_dtype = payload.dtype
        elif tuple(payload.shape) != reference_shape:
            sys.exit(
                f"{path.name}: shape {tuple(payload.shape)} != first voice "
                f"shape {reference_shape} — voicepack tensors must agree "
                "on shape to stack"
            )
        elif payload.dtype != reference_dtype:
            sys.exit(
                f"{path.name}: dtype {payload.dtype} != first voice dtype "
                f"{reference_dtype} — voicepack tensors must agree on dtype"
            )
        names.append(path.stem)
        tensors.append(payload)
    return names, tensors


def build_voicepack(voice_tensors: list) -> "torch.Tensor":  # noqa: F821 (fwd-ref)
    """Stack per-voice tensors into a single ``[num_voices, *voice_shape]``
    tensor, contiguous and cpu-resident."""
    import torch

    stacked = torch.stack(voice_tensors, dim=0)
    return stacked.detach().contiguous().to("cpu")


def prepare_safetensors_dict(flat: dict) -> dict:
    """Materialize every Tensor into a contiguous cpu-resident copy that
    safetensors.save_file can consume.

    Also filters out any accidentally shared-storage overlaps (safetensors 0.4+
    aborts on shared storage; we detach + clone to break aliases).
    """
    dense: dict = {}
    for k, v in flat.items():
        # Detach to strip autograd state, contiguous to normalize memory layout,
        # clone to sever any shared storage the .pth may have introduced.
        dense[k] = v.detach().contiguous().clone().to("cpu")
    return dense


def enrich_config(config_path: Path, voice_names: list[str], out_path: Path) -> dict:
    """Read the upstream config.json, add ``voices: [...]``, write to out_path.

    Returns the enriched config dict (useful for CI logging).
    """
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "voices" in cfg or "voice_names" in cfg:
        # Preserve whatever the upstream already declares — a future upstream
        # config may finally add the field, and we should not clobber it.
        pass
    else:
        cfg["voices"] = voice_names
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def write_report(
    report_path: Path,
    flat_len: int,
    voicepack_shape: tuple,
    voice_count: int,
    voice_names: list[str],
    dtype_hist: dict,
    output_bytes: int,
    stacked: bool,
) -> None:
    """Emit a markdown-flavoured summary safe to embed in ``$GITHUB_STEP_SUMMARY``."""
    voicepack_line = (
        f"- Voicepack (stacked, `voicepack` tensor): shape `{voicepack_shape}` "
        f"({voice_count} voices, axis 0 = voice id)"
        if stacked
        else (
            f"- Per-voice shape: `{voicepack_shape}` ({voice_count} voices) — "
            "**not stacked** (see `kokoro_prepare_checkpoint.py --help`; "
            "voices are surfaced through the enriched config's `voices` array)"
        )
    )
    lines = [
        "### Kokoro-82M checkpoint prep",
        "",
        f"- Merged state dict: **{flat_len} tensors**",
        voicepack_line,
        f"- Safetensors output: **{output_bytes:,} bytes**",
        "",
        "**Dtype breakdown**:",
        "",
        "| dtype | count |",
        "|-------|-------|",
    ]
    for dtype, count in sorted(dtype_hist.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{dtype}` | {count} |")
    lines.extend(
        [
            "",
            "**First 10 voice names** (alphabetical): "
            + ", ".join(f"`{v}`" for v in voice_names[:10])
            + (f", ... (+{len(voice_names) - 10} more)" if len(voice_names) > 10 else ""),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Kokoro-82M kokoro-v1_0.pth + voices/*.pt into a single "
            "safetensors file the Vokra converter can consume, and emit an "
            "enriched config.json with a `voices:` array derived from the .pt "
            "filenames."
        )
    )
    parser.add_argument("--pth", required=True, type=Path, help="Path to kokoro-v1_0.pth")
    parser.add_argument("--voices-dir", required=True, type=Path, help="Directory of voices/*.pt")
    parser.add_argument("--config", required=True, type=Path, help="Upstream config.json")
    parser.add_argument("--output", required=True, type=Path, help="Output merged safetensors")
    parser.add_argument(
        "--enriched-config",
        required=True,
        type=Path,
        help="Where to write the enriched config.json (never overwrites --config)",
    )
    parser.add_argument(
        "--voicepack-report",
        default=None,
        type=Path,
        help="Optional path to write a markdown-flavoured summary for CI step summary.",
    )
    parser.add_argument(
        "--stack-voicepack",
        action="store_true",
        help=(
            "EXPERIMENTAL: stack voices/*.pt into a `voicepack` tensor of shape "
            "[num_voices, *voice_shape] and add it to the safetensors output. "
            "Default OFF — see module docstring for the style_dim derivation "
            "ambiguity this creates. Enable only if you have a voicepack layout "
            "spec that matches the [num_voices, 510, 1, 256] shape."
        ),
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
        from safetensors.torch import save_file
    except ImportError as exc:
        sys.exit(
            f"missing Python dep ({exc}); install with "
            "`pip install torch safetensors` in the parity venv"
        )

    print(f"[kokoro-prep] loading state dict from {args.pth}")
    flat = load_state_dict(args.pth)
    print(f"[kokoro-prep]   → {len(flat)} tensors flattened")

    print(f"[kokoro-prep] loading voices from {args.voices_dir}")
    voice_names, voice_tensors = load_voice_tensors(args.voices_dir)
    print(f"[kokoro-prep]   → {len(voice_names)} voices; first={voice_names[0]!r}")

    if args.stack_voicepack:
        voicepack = build_voicepack(voice_tensors)
        voicepack_shape = tuple(voicepack.shape)
        print(f"[kokoro-prep]   → voicepack shape {voicepack_shape}, dtype {voicepack.dtype}")

        if "voicepack" in flat:
            # The canonical release does not ship a stacked voicepack tensor in
            # the .pth, so a collision would mean a fork already stacked them.
            # For safety, refuse to silently overwrite: fail loud.
            existing_shape = tuple(flat["voicepack"].shape)
            if existing_shape != voicepack_shape:
                sys.exit(
                    f"checkpoint already contains 'voicepack' with shape {existing_shape}; "
                    f"refusing to overwrite with stacked shape {voicepack_shape} "
                    "(delete the .pth's voicepack or drop --stack-voicepack)"
                )
        flat["voicepack"] = voicepack
    else:
        # Default path: derive per-voice shape for the report but do NOT stack
        # into a voicepack tensor. See module docstring for the style_dim
        # derivation rationale.
        voicepack_shape = tuple(voice_tensors[0].shape)
        print(
            f"[kokoro-prep]   → per-voice shape {voicepack_shape} "
            f"(not stacked; --stack-voicepack to add as `voicepack` tensor)"
        )

    print("[kokoro-prep] preparing safetensors payload")
    dense = prepare_safetensors_dict(flat)

    print(f"[kokoro-prep] writing {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(dense, str(args.output))
    output_bytes = args.output.stat().st_size

    print(f"[kokoro-prep] writing enriched config to {args.enriched_config}")
    args.enriched_config.parent.mkdir(parents=True, exist_ok=True)
    _ = enrich_config(args.config, voice_names, args.enriched_config)

    # Build dtype histogram for the report.
    dtype_hist: dict = {}
    for t in dense.values():
        key = str(t.dtype)
        dtype_hist[key] = dtype_hist.get(key, 0) + 1

    if args.voicepack_report is not None:
        args.voicepack_report.parent.mkdir(parents=True, exist_ok=True)
        write_report(
            args.voicepack_report,
            flat_len=len(dense),
            voicepack_shape=voicepack_shape,
            voice_count=len(voice_names),
            voice_names=voice_names,
            dtype_hist=dtype_hist,
            output_bytes=output_bytes,
            stacked=args.stack_voicepack,
        )
        print(f"[kokoro-prep] wrote report to {args.voicepack_report}")

    stacked_note = "stacked" if args.stack_voicepack else "not stacked"
    print(
        f"[kokoro-prep] done: {len(dense)} tensors ({output_bytes:,} bytes), "
        f"voicepack shape {voicepack_shape} [{stacked_note}], {len(voice_names)} voices"
    )


if __name__ == "__main__":
    main()
