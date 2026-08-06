#!/usr/bin/env python3
"""Flatten an FRCRN torch checkpoint → safetensors (coverage-audit wave-a, 2026-08-03).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the runtime).
The upstream FRCRN release ships torch pickle ``.pt`` / ``.pth`` files under
two distribution paths:

* ``github.com/alibabasglab/FRCRN`` — the original author's repository (a
  ``pretrained_model/`` folder with the DNS Challenge 2022 checkpoint), and
* ``github.com/modelscope/ClearerVoice-Studio`` — the ClearerVoice-Studio
  umbrella that pulls the same checkpoint via ModelScope hub.

The Rust converter (``crates/vokra-convert/src/models/frcrn.rs``) consumes
safetensors only, so this script bridges the two:

* Loads the ``.pt`` / ``.pth`` with ``torch.load(..., weights_only=True)``
  first (the release checkpoints are expected to load cleanly under the
  safe loader). A checkpoint that does not load under ``weights_only=True``
  is refused with a loud error rather than silently falling back to unsafe
  unpickling — same posture as ``dfn3_prepare_checkpoint.py`` /
  ``dac_prepare_checkpoint.py``.
* Unwraps the state dict from common upstream wrappers: (a) a top-level
  flat ``OrderedDict`` (used by the standalone FRCRN release), (b) a
  ClearerVoice-Studio ``{"state_dict": …}`` wrapper (used by the CVS
  pipeline), (c) a Lightning-style ``{"model": …}`` wrapper (some
  ModelScope mirrors). Any other top-level shape is a hard error.
* De-duplicates shared-storage tensors (mirror of the memory
  ``reference-safetensors-shared-tensor-dedup`` pattern used by
  ``bark`` / ``xtts-v2`` / ``moss`` variants). Two names pointing at
  the same underlying storage are recorded in the audit trail
  ``<output>.shared_pairs.json`` next to the safetensors so the runtime
  side can re-tie them on load without silent divergence.
* Writes the flat state dict (dotted upstream keys preserved:
  ``encoder.complex_conv0.weight`` / ``rnn.weight_ih_l0`` / … — the exact
  FRCRN topology the Rust converter round-trips via BF16 pass-through)
  using ``safetensors.torch.save_file`` when the ``safetensors`` package
  is available, else falling back to a hand-rolled writer (stdlib
  ``json`` + raw bytes — the same fallback DFN3 uses so the offline
  venv needs no extra package).
* F32 / F16 / BF16 tensors are written as-is (verbatim dtype
  preservation — the ``vokra-convert::models::frcrn`` pass-through
  contract). Non-float tensors are DROPPED with an explicit line per
  drop (BatchNorm ``*.num_batches_tracked`` I64 counters, integer
  buffer indices — none have inference roles and vokra-core's
  safetensors parser is float-only by design). Any *other* anomaly
  (non-tensor entry, unknown dtype) is a hard error, never a silent
  drop (FR-EX-08 posture).
* Prints a sha256 manifest line for the output for the fixture /
  workflow logs.

Fails loudly on any anomaly rather than masking it — FR-EX-08 posture.

# Environment (memory: [[feedback-python-uses-uv]] / [[feedback-python-3-12]])

This script is designed for the ``tools/parity`` uv-managed venv (Python
3.12). The parent ``pyproject.toml`` in ``tools/parity/`` pins the torch
dependency; run via:

::

    cd tools/parity
    uv run python frcrn_prepare_checkpoint.py \\
        --ckpt ~/checkpoints/frcrn/ckpt/model.pth \\
        --output ~/checkpoints/frcrn/model.safetensors

Then (from the repo root, after ``cargo build --release -p vokra-cli``):

::

    ./target/release/vokra-cli convert --model frcrn \\
        --input ~/checkpoints/frcrn/model.safetensors \\
        --output ~/gguf/frcrn.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import torch


# safetensors dtype tag map for the hand-rolled writer fallback (used only
# when the `safetensors` package is unavailable — the DFN3 posture).
DTYPE_TAG = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}

# torch dtype → element byte width for the hand-rolled writer's payload
# copies (must match the on-disk layout the vokra-core safetensors reader
# expects). F32 = 4, F16 / BF16 = 2 — a superset of DTYPE_TAG so the
# widths stay a single source of truth if a future dtype is added.
DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


def _unwrap_state_dict(obj: object, path: str) -> "OrderedDict[str, torch.Tensor]":
    """Extract the flat state dict from the loaded checkpoint object.

    Handles three known upstream wrapper shapes:

    * Flat ``OrderedDict[str, Tensor]`` (author-repo release).
    * ``{"state_dict": OrderedDict, ...}`` (ClearerVoice-Studio / most
      Lightning checkpoints).
    * ``{"model": OrderedDict, ...}`` (some ModelScope mirrors).

    Any other shape is a loud error (FR-EX-08 posture).
    """
    if isinstance(obj, (OrderedDict, dict)):
        # Flat state dict? Every value is a tensor.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj)
        # Wrapped state dict — walk two known keys.
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], (OrderedDict, dict)):
                inner = obj[key]
                if all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return OrderedDict(inner)
                raise SystemExit(
                    f"{path}: checkpoint['{key}'] contains non-tensor entries "
                    f"(first offender: "
                    f"{next(k for k, v in inner.items() if not isinstance(v, torch.Tensor))!r})"
                )
        raise SystemExit(
            f"{path}: checkpoint top-level dict has neither a 'state_dict' nor a "
            f"'model' key, and its own values are not all tensors "
            f"(keys={list(obj)[:8]}{'...' if len(obj) > 8 else ''})"
        )
    raise SystemExit(
        f"{path}: checkpoint top level is {type(obj).__name__}, expected a dict / OrderedDict"
    )


def _dedup_shared_storage(
    state: "OrderedDict[str, torch.Tensor]",
) -> tuple["OrderedDict[str, torch.Tensor]", list[tuple[str, str]]]:
    """Drop duplicate references to the same underlying storage.

    Two safetensors entries pointing at the same ``data_ptr`` fail the
    writer (mirror memory [[reference-safetensors-shared-tensor-dedup]] —
    ``safetensors.torch.save_file`` raises RuntimeError). Keep the FIRST
    seen name; record every subsequent alias as a ``(alias, primary)``
    pair the caller writes out as an audit trail so the runtime side
    can re-tie them on load. Contiguous clone is issued so a downstream
    ``.numpy()`` cannot depend on a shared-storage layout the alias
    map has already discarded.
    """
    seen: dict[int, str] = {}
    kept: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    aliases: list[tuple[str, str]] = []
    for name, t in state.items():
        ptr = t.data_ptr()
        if ptr in seen:
            aliases.append((name, seen[ptr]))
            continue
        seen[ptr] = name
        kept[name] = t.detach().contiguous()
    return kept, aliases


def _write_safetensors_stdlib(
    path: Path, tensors: "OrderedDict[str, torch.Tensor]"
) -> None:
    """Hand-rolled safetensors writer (stdlib only).

    Header layout: 8-byte little-endian header length + UTF-8 JSON header
    + contiguous little-endian tensor data. BF16 is emitted as raw
    little-endian bytes — same as ``F16``, just tagged ``BF16`` in the
    header so a vokra-core parser identifies it correctly. Runtime widens
    BF16 → f32 losslessly via ``crates/vokra-core/src/gguf/quant/mod.rs
    decode_bf16`` (``bits << 16``).
    """
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        if t.dtype not in DTYPE_TAG:
            raise SystemExit(f"unsupported dtype {t.dtype} for tensor {name!r}")
        # `.numpy()` refuses BF16 (numpy has no bf16 dtype); use a raw
        # `.view(torch.uint8)` cast for a byte-identical serialization
        # of the underlying storage. For F32 / F16 the `.numpy().tobytes()`
        # path is byte-identical to `.view(torch.uint8)` on a
        # little-endian host, so the raw path works everywhere.
        raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        expected = t.numel() * DTYPE_BYTES[t.dtype]
        if len(raw) != expected:
            raise SystemExit(
                f"tensor {name!r}: raw payload {len(raw)} B "
                f"disagrees with numel × width {expected} B"
            )
        header[name] = {
            "dtype": DTYPE_TAG[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _write_safetensors(
    path: Path, tensors: "OrderedDict[str, torch.Tensor]"
) -> str:
    """Save via ``safetensors.torch.save_file`` when available, else fall
    back to the hand-rolled stdlib writer. Returns the writer used."""
    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError:
        _write_safetensors_stdlib(path, tensors)
        return "stdlib"
    save_file(tensors, str(path))
    return "safetensors.torch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        "--input",
        required=True,
        dest="ckpt",
        help="FRCRN torch checkpoint (.pt or .pth)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output .safetensors path",
    )
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.output)
    if not ckpt.is_file():
        raise SystemExit(f"--ckpt does not exist or is not a file: {ckpt}")
    out.parent.mkdir(parents=True, exist_ok=True)

    obj = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(obj, str(ckpt))

    # Drop non-float tensors up front (BatchNorm counters, integer
    # buffer indices). Report every drop so a downstream owner can spot
    # a legitimate float tensor that was mis-classified as non-inference.
    dropped: list[str] = []
    kept_floats: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(
                f"state[{name!r}] is not a tensor: got {type(t).__name__}"
            )
        if t.dtype in DTYPE_TAG:
            kept_floats[name] = t
        else:
            dropped.append(f"{name}\t{t.dtype}")

    # De-duplicate shared-storage tensors before writing.
    unique, aliases = _dedup_shared_storage(kept_floats)

    # Contiguous cast is applied inside `_dedup_shared_storage`; the
    # writer sees uncorrupted tensors regardless of whether upstream
    # weight tying handed us a view.
    writer = _write_safetensors(out, unique)

    # Aliases audit trail — the runtime side re-ties them on load. Emit
    # even when empty so the file's presence proves the dedup pass ran.
    aliases_path = out.with_suffix(out.suffix + ".shared_pairs.json")
    aliases_path.write_text(
        json.dumps({"aliases": aliases}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    n_params = sum(int(t.numel()) for t in unique.values())
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    for name in dropped:
        print(f"dropped (non-float): {name}")
    for alias, primary in aliases:
        print(f"aliased: {alias} -> {primary} (dropped duplicate storage)")
    print(f"{sha}  {out}")
    print(f"tensors={len(unique)} params={n_params} writer={writer}")
    print(f"aliases_json={aliases_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
