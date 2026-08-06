#!/usr/bin/env python3
"""Flatten a salute-developers/GigaAM ``pretrained/*.pt`` → safetensors
(coverage-audit-2026-08-03 Wave B, sber-gigaam-multilingual).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream GigaAM release ships torch-pickle checkpoints
under ``github.com/salute-developers/GigaAM/tree/main/pretrained``
(the multilingual variant follows the same layout the sibling
``sber-gigaam-v3`` uses — either a flat state dict or a dict wrapping
one under a common key ``state_dict`` / ``model_state_dict`` /
``model``). The Rust converter
(``crates/vokra-convert/src/models/sber_gigaam_multilingual.rs``)
consumes safetensors only, so this script bridges the two:

* Loads the ``.pt`` with ``torch.load(..., weights_only=True)`` (the
  release checkpoint loads cleanly under the safe loader; a checkpoint
  that does not is refused rather than falling back to unsafe
  unpickling — the CVE-2020-27786-class ``pickle.loads`` supply-chain
  posture the sibling ``dfn3_prepare_checkpoint.py`` /
  ``nkf_aec_prepare_checkpoint.py`` / ``csm_dump.py`` scripts adopt).
* Unwraps a single-level ``state_dict`` container if present (the
  upstream release ships a flat OrderedDict but future releases may
  wrap it; the DFN3 / NKF-AEC pattern of "unwrap known container keys
  but never execute arbitrary unpickled callables" applies).
* Writes the flat state dict as safetensors. The writer is
  hand-rolled (stdlib ``json`` + raw bytes + a minimal safetensors
  ``shared_pairs.json`` audit trail on the side) so the eval venv
  needs no ``safetensors`` package; the format is the standard one
  vokra-core parses.
* F32 tensors are written as-is. F16 / BF16 pass through under their
  original dtype (the runtime widens BF16 → f32 losslessly at load
  via ``crates/vokra-core/src/gguf/quant/mod.rs decode_bf16``).
* I64 ``num_batches_tracked`` BatchNorm training counters and any
  other integer-typed state are DROPPED here (explicitly, each one
  reported) — they have no inference role and vokra-core's
  safetensors parser is float-only by design (FR-EX-08). Any
  ``I64`` / ``I32`` / other non-float tensor that is NOT a
  ``num_batches_tracked`` counter is a hard error rather than a
  silent drop.
* Handles ``share_memory_`` / tied-weight cases (``data_ptr``
  collisions) via a dedup pass that keeps the first-seen contiguous
  copy and records the alias graph in
  ``<output>.shared_pairs.json`` — the pattern
  ``[[reference-safetensors-shared-tensor-dedup]]`` describes for
  the general publish path (Bark / XTTS-v2 / MOSS variants).

Fails loudly on any anomaly (non-tensor entry, unexpected non-float
non-``num_batches_tracked`` dtype, empty state) rather than masking
it — FR-EX-08 posture.

# Usage

Managed through ``uv`` per the tools/parity contract
(``[[feedback-python-uses-uv]] / [[feedback-python-3-12]]``):

::

    cd tools/parity
    uv run python sber_gigaam_multilingual_prepare_checkpoint.py \\
        --input ~/checkpoints/sber-gigaam-multilingual/repo/pretrained/model.pt \\
        --output ~/checkpoints/sber-gigaam-multilingual/model.safetensors

Then:

::

    vokra-cli convert --model sber-gigaam-multilingual \\
        --input ~/checkpoints/sber-gigaam-multilingual/model.safetensors \\
        --output ~/gguf/sber-gigaam-multilingual.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

# Mapping torch dtypes → safetensors dtype tags the vokra-core parser
# accepts (F32 / F16 / BF16 today). I64 counters are handled via the
# ``num_batches_tracked`` skip below and are intentionally NOT in this
# table so any other integer tensor trips the "unexpected dtype" hard
# error.
DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}

# Container keys the wrapped-state-dict unwrap step recognises. Order
# matters only for reporting; every present key is checked and if more
# than one is populated we refuse rather than guess which one is the
# real state dict.
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


def _unwrap_state_dict(loaded: Any) -> "OrderedDict[str, torch.Tensor]":
    """Reduce ``loaded`` (whatever torch.load returned) to a flat
    ``OrderedDict[str, torch.Tensor]``.

    - If ``loaded`` is itself a dict-of-tensors, return it verbatim.
    - Else if it is a dict with exactly one known state-dict container
      key, unwrap that.
    - Else refuse (loudly), since silently choosing between multiple
      containers is exactly the kind of guess FR-EX-08 forbids.
    """
    if not isinstance(loaded, (dict, OrderedDict)):
        raise SystemExit(
            f"checkpoint top level is {type(loaded).__name__}, expected a flat state "
            "dict or a dict containing one under 'state_dict' / 'model_state_dict' / "
            "'model'"
        )
    # Direct state dict (values are tensors).
    if all(isinstance(v, torch.Tensor) for v in loaded.values()):
        return OrderedDict(loaded)
    # Wrapped state dict.
    present = [k for k in _STATE_DICT_KEYS if k in loaded]
    if len(present) == 0:
        raise SystemExit(
            "checkpoint dict contains no tensors at the top level and no "
            f"known state-dict container key {_STATE_DICT_KEYS!r}"
        )
    if len(present) > 1:
        raise SystemExit(
            f"checkpoint dict has multiple state-dict container keys {present!r}; "
            "refuse to guess which is the real state dict"
        )
    inner = loaded[present[0]]
    if not isinstance(inner, (dict, OrderedDict)) or not all(
        isinstance(v, torch.Tensor) for v in inner.values()
    ):
        raise SystemExit(
            f"container key {present[0]!r} does not hold a flat "
            "OrderedDict[str, Tensor]"
        )
    return OrderedDict(inner)


def _dedup_and_contiguousize(
    state: "OrderedDict[str, torch.Tensor]",
) -> tuple["OrderedDict[str, torch.Tensor]", list[list[str]]]:
    """Return (deduped_state, shared_pairs).

    Any two tensors that share underlying storage (``data_ptr``
    equality) would trip safetensors' ``RuntimeError: cannot serialise
    two tensors sharing memory``. We keep the first-seen name's
    contiguous copy and drop later aliases; every alias group is
    recorded in ``shared_pairs`` for a downstream audit trail (mirror
    of the general ``[[reference-safetensors-shared-tensor-dedup]]``
    pattern).
    """
    seen: dict[int, str] = {}
    shared: dict[str, list[str]] = {}
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, t in state.items():
        detached = t.detach().contiguous().cpu()
        ptr = detached.data_ptr()
        # `.contiguous()` may allocate a new buffer breaking the
        # original alias; use the ORIGINAL tensor's data_ptr to detect
        # sharing (mirroring what safetensors' own check sees).
        orig_ptr = t.detach().data_ptr()
        if orig_ptr in seen:
            keeper = seen[orig_ptr]
            shared.setdefault(keeper, [keeper]).append(name)
            continue
        seen[orig_ptr] = name
        out[name] = detached
        # Silence the unused-ptr warning without dropping the
        # after-contiguous ptr we might want in a future extension.
        _ = ptr
    return out, [group for group in shared.values() if len(group) > 1]


def write_safetensors(path: str, tensors: "OrderedDict[str, torch.Tensor]") -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header
    length + JSON header + contiguous little-endian tensor data.

    Refuses any dtype not in [`DTYPE_MAP`]. BF16 is emitted as GGUF
    type ``BF16`` verbatim (top 16 bits of an f32) — the runtime
    widens BF16 → f32 losslessly at load via
    ``crates/vokra-core/src/gguf/quant/mod.rs decode_bf16``.
    """
    header: dict[str, dict[str, Any]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        if t.dtype not in DTYPE_MAP:
            raise SystemExit(f"unsupported dtype {t.dtype} for tensor {name!r}")
        # PyTorch has no numpy view for BF16 (numpy has no bfloat16
        # dtype), so route BF16 explicitly through its raw byte
        # representation. For F32 / F16 use the standard numpy path.
        if t.dtype is torch.bfloat16:
            # BF16 is the top 16 bits of an f32; ``.view(torch.int16)``
            # then ``.numpy()`` gives us the raw 2-byte-per-element
            # payload the safetensors writer expects, in host byte
            # order (which is little-endian on every architecture
            # Vokra supports).
            data = t.contiguous().view(torch.int16).cpu().numpy().tobytes()
        else:
            data = t.detach().contiguous().cpu().numpy().tobytes()
        header[name] = {
            "dtype": DTYPE_MAP[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        help="path to salute-developers/GigaAM multilingual .pt (torch pickle)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output .safetensors path",
    )
    args = ap.parse_args()

    loaded = torch.load(args.input, map_location="cpu", weights_only=True)
    state = _unwrap_state_dict(loaded)
    if len(state) == 0:
        raise SystemExit("state dict is empty — refuse to emit a zero-tensor GGUF")

    # Drop known non-inference integer counters, keep every other float
    # tensor, hard-error on any unexpected non-float dtype.
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped: list[str] = []
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"non-tensor state entry {name!r}: {type(t).__name__}")
        if name.endswith(".num_batches_tracked"):
            # BatchNorm training counter (I64 scalar): no inference
            # role, and vokra-core's safetensors parser is float-only.
            dropped.append(name)
            continue
        if t.dtype not in DTYPE_MAP:
            raise SystemExit(
                f"unexpected dtype {t.dtype} for tensor {name!r} — the safetensors "
                "bridge is float-only (F32 / F16 / BF16), and this tensor is not a "
                "known integer counter that can be safely dropped"
            )
        out[name] = t

    deduped, shared = _dedup_and_contiguousize(out)

    write_safetensors(args.output, deduped)

    n_params = sum(t.numel() for t in deduped.values())

    # Write a `shared_pairs.json` audit trail only if any aliasing was
    # actually detected — an empty sidecar just adds noise to the eval
    # tree.
    if shared:
        sidecar = Path(args.output).with_suffix(Path(args.output).suffix + ".shared_pairs.json")
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(shared, f, indent=2)
        print(f"shared-tensor alias groups written to: {sidecar}")

    for name in dropped:
        print(f"dropped (non-inference counter): {name}")

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    print(f"{sha}  {args.output}")
    print(f"tensors={len(deduped)} params={n_params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
