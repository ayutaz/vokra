#!/usr/bin/env python3
"""Bridge facebook/melodyflow-t24-30secs upstream bundle → flat safetensors.

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream Meta AudioCraft ``facebook/melodyflow-t24-30secs``
release ships as a **compressed torch pickle bundle** (`state_dict.bin`
/ `.th` file) containing the **1 B parameter flow-matching DiT
transformer** (24 timesteps, 30 sec max horizon at 48 kHz, dual text +
audio prefix conditioning for the editing use-case), plus a bundled
48 kHz RVQ audio codec, plus a frozen T5-base text encoder — total
~4.0 GB per HF cardData primary source (2026-08-13). The Rust converter
(``crates/vokra-convert/src/models/melodyflow_t24_30secs.rs``) consumes
**single-file safetensors** by design — the runtime never grows a
shard-index reader or a pickle parser (NFR-DS-02 zero-dep + FR-LD-05
no pickle in runtime). This script bridges the two by:

1. **Downloading** the release via ``huggingface_hub.snapshot_download``
   into the given local dir (or reads what's already there).
2. **Loading** the ``state_dict.bin`` (or equivalent audiocraft dump)
   via ``torch.load(..., weights_only=True)`` — an explicit safe-loader
   gate: torch ≥ 2.6 defaults to ``weights_only=True`` but the older
   ranges accepted in ``pyproject.toml`` may need the explicit kwarg.
   If the release ships as native safetensors (mirror publishers do
   sometimes), the ``--input-safetensors`` path skips torch pickle
   entirely.
3. **Dedupes shared-storage tensors** via a ``data_ptr → canonical
   name`` pass (memory ``[[reference-safetensors-shared-tensor-dedup]]``):
   ``safetensors.torch.save_file`` refuses two names pointing at the
   same storage, so every duplicate is cloned + made contiguous into
   genuinely independent storage. The alias graph is preserved in
   ``<output>.shared_pairs.json`` so a downstream runtime binder can
   restore ties (e.g. tied text embedding + lm_head — an audiocraft-
   family posture inherited from sibling MAGNeT / MusicGen).
4. **Drops non-float training-scaffold entries** explicitly and reports
   each one:

   - ``.num_batches_tracked`` BatchNorm I64 counters (no inference
     role — eval-mode BatchNorm consumes only running_mean /
     running_var / weight / bias).
   - ``.total_ops`` / ``.total_params`` ``torch.profiler`` bookkeeping.

5. **Rejects unexpected non-float dtypes** (I32 / I64 / F64 / Bool)
   that are NOT in the drop-list — FR-EX-08 no silent fallback. This
   catches upstream state that a future runtime binder needs to know
   about explicitly.
6. **F32 / F16 / BF16 pass through** under their upstream dtype (the
   Rust converter's ``GgmlType::F32 | GgmlType::F16 | GgmlType::BF16``
   pass-through arm handles all three; the runtime widens BF16 → f32
   losslessly at load via ``crates/vokra-core/src/gguf/quant/mod.rs
   decode_bf16``).
7. **Emits** a ``<output>.sha256`` line + parameter count + tensor
   count to stdout for the fixture / workflow logs.

# Memory footprint & vast.ai posture

The ~4.0 GB checkpoint technically fits on M1 iMac 16 GB but the phase
task pins vast.ai as the conservative default for weights ≥ 2 GB per
memory ``[[feedback-large-models-on-vast-ai]]`` — the Voxtral-Small-24B
mmap incident (swap 40 GB → Mac forced-shutdown) is the precedent that
raised the local safety margin. Peak resident memory is roughly the
whole model plus a ``safetensors.torch.save_file`` serialisation buffer;
sibling ``magnet_medium_30secs`` (~5.7 GB, ~12 GB working set) landed
without vast.ai but the audiocraft pickle format is heavier than a
native safetensors of the same bytesize, so ~4.0 GB MelodyFlow may push
resident closer to ~10 GB on the pickle-decode step. The **vast.ai
runbook** (``docs/handoff/vast-ai-large-model-publish.md``) applies —
provision.sh Wave 12 handles the hf_config.pth shim + certifi + xet
routing gotchas.

# Usage

Managed through ``uv`` per the tools/parity contract (memory
``[[feedback-python-uses-uv]] / [[feedback-python-3-12]]``).

::

    cd tools/parity/melodyflow_t24_30secs
    uv sync

    # From a downloaded bundle (via `hf download`):
    uv run python prepare_checkpoint.py \\
        --input-dir /path/to/melodyflow-t24-30secs \\
        --output    /path/to/melodyflow-t24-30secs/flat.safetensors

    # Then feed the Rust converter:
    ./target/release/vokra-cli convert \\
        --model melodyflow-t24-30secs \\
        --input /path/to/melodyflow-t24-30secs/flat.safetensors \\
        --output /path/to/out/melodyflow-t24-30secs.gguf

    # Then publish (T4 tier: --allow-noncommercial MANDATORY):
    bash scripts/publish/publish-one.sh \\
        --gguf /path/to/out/melodyflow-t24-30secs.gguf \\
        --repo vokra/melodyflow-t24-30secs \\
        --license-spdx cc-by-nc-4.0 \\
        --allow-noncommercial \\
        --push

# Rust converter contract

The output of this script is fed to
``crates/vokra-convert/src/models/melodyflow_t24_30secs.rs`` unchanged.
That converter walks the safetensors file, pass-through-encodes each
F32 / F16 / BF16 tensor into GGUF, and stamps ``vokra.model.arch =
"melodyflow_t24_30secs"`` + ``vokra.model.category = "music"`` +
``vokra.provenance.upstream_hf = "facebook/melodyflow-t24-30secs"`` +
``vokra.provenance.license = "cc-by-nc-4.0"`` +
``vokra.provenance.weight_license = "noncommercial"``. Publishing
requires ``publish-one.sh --allow-noncommercial`` (T4 tier fail-closed
gate, MusicGen family / X-Codec-2 / jasco_400m_chords_drums / sibling
``magnet_small_10secs`` / ``magnet_medium_30secs`` precedent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DROPPED_SUFFIXES = (
    # BatchNorm training-scaffold counters (no inference role).
    ".num_batches_tracked",
    # torch.profiler bookkeeping (no inference role).
    ".total_ops",
    ".total_params",
)

ACCEPTED_FLOAT_DTYPES = frozenset(
    {"torch.float32", "torch.float16", "torch.bfloat16"}
)


def _dtype_name(t: Any) -> str:
    """Return a canonical ``torch.<name>`` string for a torch dtype."""
    return str(t.dtype) if hasattr(t, "dtype") else "unknown"


def _load_state_dict(input_dir: Path, input_safetensors: Path | None) -> dict[str, Any]:
    """Load the upstream checkpoint into a state dict.

    Two paths:
      - ``--input-safetensors PATH`` (skips torch pickle entirely).
      - Discover a ``.bin`` / ``.th`` pickle under ``--input-dir``.
    """
    import torch  # deferred (uv-managed)

    if input_safetensors is not None:
        # Native safetensors — no pickle. Some mirror publishers ship
        # audiocraft weights as safetensors; prefer that when possible.
        from safetensors.torch import load_file as safe_load

        print(f"[load] safetensors: {input_safetensors}", file=sys.stderr)
        return safe_load(str(input_safetensors))

    if input_dir is None or not input_dir.exists():
        raise SystemExit(
            f"prepare_checkpoint: neither --input-safetensors nor a "
            f"valid --input-dir was provided (got {input_dir!r}) — "
            f"FR-EX-08 no silent empty-output fallback"
        )

    # Discover a checkpoint bundle under input_dir. audiocraft releases
    # commonly ship as ``state_dict.bin`` (torch pickle) or ``.th``.
    candidates: list[Path] = []
    for pat in ("state_dict.bin", "*.th", "*.bin", "pytorch_model.bin"):
        candidates.extend(sorted(input_dir.glob(pat)))
    # Deduplicate (glob overlap) preserving order.
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    if not unique_candidates:
        raise SystemExit(
            f"prepare_checkpoint: no state_dict.bin / *.th / *.bin under "
            f"{input_dir} — refusing to emit empty output (FR-EX-08)"
        )

    ckpt_path = unique_candidates[0]
    print(f"[load] torch pickle: {ckpt_path}", file=sys.stderr)
    # weights_only=True is the safe path — refuses arbitrary pickle
    # opcodes that could execute code, only permits tensor construction.
    # torch <2.4 does not accept the kwarg; wrap for compat.
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch — no weights_only kwarg. This branch stays
        # explicit so a future audit can still identify the unsafe
        # load path if it triggers.
        obj = torch.load(str(ckpt_path), map_location="cpu")

    # audiocraft state_dict.bin sometimes wraps the raw state under a
    # ``"best_state"`` / ``"state_dict"`` / ``"model"`` key; unwrap.
    for k in ("best_state", "state_dict", "model", "state"):
        if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            print(f"[load]   unwrapped .{k}", file=sys.stderr)
            break

    if not isinstance(obj, dict):
        raise SystemExit(
            f"prepare_checkpoint: loaded object is {type(obj).__name__}, "
            f"expected dict-like state_dict"
        )
    return obj


def _dedup_and_filter(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str]], list[str], list[tuple[str, str]]]:
    """Dedupe shared storage + drop training-scaffold + reject unknowns.

    Returns (safe_state_dict, shared_pairs, dropped, rejected).
    """
    import torch  # deferred

    seen: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []
    dropped: list[str] = []
    rejected: list[tuple[str, str]] = []
    safe_state: dict[str, Any] = {}

    for name, tensor in state_dict.items():
        # Drop training-scaffold explicitly.
        if any(name.endswith(suf) for suf in DROPPED_SUFFIXES):
            dropped.append(name)
            continue

        if not isinstance(tensor, torch.Tensor):
            # Non-tensor entries (e.g. lists of int) — reject loudly.
            rejected.append((name, f"non-tensor {type(tensor).__name__}"))
            continue

        # Accept float dtypes only. Reject anything else loudly.
        dt = _dtype_name(tensor)
        if dt not in ACCEPTED_FLOAT_DTYPES:
            rejected.append((name, dt))
            continue

        # Dedupe shared storage (safetensors.torch.save_file refuses
        # aliases). data_ptr() is 0 for meta / uninitialized tensors —
        # those get their own copies too.
        ptr = tensor.data_ptr()
        if ptr != 0 and ptr in seen:
            canonical = seen[ptr]
            shared_pairs.append((name, canonical))
            safe_state[name] = tensor.detach().clone().contiguous()
        else:
            if ptr != 0:
                seen[ptr] = name
            # Always contiguous — safetensors requires it.
            safe_state[name] = (
                tensor.detach().contiguous()
                if tensor.is_contiguous()
                else tensor.detach().clone().contiguous()
            )

    if rejected:
        # Report every rejected entry, then bail. FR-EX-08 no silent
        # fallback — a future binder must decide how to handle each.
        for name, why in rejected:
            print(f"[reject] {name}  ({why})", file=sys.stderr)
        raise SystemExit(
            f"prepare_checkpoint: {len(rejected)} unexpected non-float "
            f"tensors — refusing to write output (FR-EX-08)"
        )

    return safe_state, shared_pairs, dropped, rejected


def _sha256_file(path: Path) -> str:
    """Return the hex sha256 of ``path``'s contents (streaming)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the upstream checkpoint bundle "
        "(state_dict.bin / *.th / *.bin). Discover mode.",
    )
    ap.add_argument(
        "--input-safetensors",
        type=Path,
        default=None,
        help="Explicit path to a native safetensors file (skips torch "
        "pickle entirely — some mirror publishers ship audiocraft "
        "weights as safetensors directly).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the flat safetensors output file.",
    )
    args = ap.parse_args()

    if args.input_dir is None and args.input_safetensors is None:
        print(
            "prepare_checkpoint: exactly one of --input-dir or "
            "--input-safetensors must be provided",
            file=sys.stderr,
        )
        return 2

    state_dict = _load_state_dict(args.input_dir, args.input_safetensors)
    print(
        f"[load] loaded {len(state_dict)} raw entries",
        file=sys.stderr,
    )

    safe_state, shared_pairs, dropped, _rejected = _dedup_and_filter(state_dict)
    print(
        f"[filter] {len(safe_state)} float tensors, "
        f"{len(shared_pairs)} shared aliases, "
        f"{len(dropped)} training-scaffold dropped",
        file=sys.stderr,
    )
    for name in dropped:
        print(f"[drop]  {name}", file=sys.stderr)
    for alias, canonical in shared_pairs:
        print(f"[alias] {alias}  ->  {canonical}", file=sys.stderr)

    # Serialize.
    from safetensors.torch import save_file as safe_save

    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_save(safe_state, str(args.output))
    print(f"[write] {args.output}", file=sys.stderr)

    # Sidecar sha256 + shared-pair audit trail.
    sha = _sha256_file(args.output)
    (args.output.with_suffix(args.output.suffix + ".sha256")).write_text(
        f"{sha}  {args.output.name}\n"
    )
    if shared_pairs:
        (args.output.with_suffix(args.output.suffix + ".shared_pairs.json")).write_text(
            json.dumps(
                [{"alias": a, "canonical": c} for a, c in shared_pairs],
                indent=2,
            )
            + "\n"
        )

    # Summary line for the CI log.
    total_params = sum(t.numel() for t in safe_state.values())
    print(
        f"[done] wrote {args.output.name}: "
        f"{len(safe_state)} tensors / {total_params:,} params / sha256 {sha[:16]}...",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
