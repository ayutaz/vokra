#!/usr/bin/env python3
"""Bridge FireRedTeam/FireRedASR-AED-L torch pickle → safetensors.

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream `FireRedTeam/FireRedASR-AED-L` release ships a
torch pickle checkpoint (typically ``model.pth.tar`` or ``model.pt``);
this converter's Rust side
(``crates/vokra-convert/src/models/firered_asr_aed_l.rs``) consumes
safetensors only, so this script bridges the two:

* Loads the pickle with ``torch.load(..., weights_only=True)`` (the
  release checkpoint loads cleanly under the safe loader — a checkpoint
  that does not is refused rather than falling back to unsafe
  unpickling).
* Unwraps the flat ``state_dict`` — the pickle top level may itself be
  the state dict, or a single-key container ``{"state_dict": {...}}`` /
  ``{"model": {...}}`` / ``{"model_state_dict": {...}}`` (all three are
  common variants across NeMo / Lightning / raw-Torch training loops).
* Dedupes shared-storage tensors (memory
  [[reference-safetensors-shared-tensor-dedup]]): ``safetensors.torch.
  save_file`` refuses two names that point at the same storage
  (`RuntimeError: The tensors X and Y are pointing at the same
  storage`), so a ``seen: dict[data_ptr → name]`` audit trail records
  the mapping to ``<output>.shared_pairs.json``; every duplicate is
  cloned + made contiguous so the destination is genuinely independent
  storage — but the audit trail preserves the shared-tie fact for
  downstream inspection (a tied embedding is a legitimate topology
  feature that a future runtime binding must honour).
* Drops BatchNorm ``.num_batches_tracked`` I64 counters explicitly —
  they have no inference role (eval-mode BatchNorm uses only
  running_mean / running_var / weight / bias); every dropped name is
  reported on stdout so a silent shape drift is impossible.
* Preserves F32 / F16 / BF16 dtypes verbatim (the Rust converter
  pass-through arm handles all three); any other dtype is a hard error
  (FR-EX-08 no silent fallback).
* Emits a sha256 manifest line + parameter count on stdout for the
  fixture / workflow logs.

# Usage

::

    cd tools/parity
    uv run python firered_asr_aed_l_prepare_checkpoint.py \\
        --ckpt ~/.cache/vokra/weights/firered-asr-aed-l/model.pth.tar \\
        --output ~/.cache/vokra/weights/firered-asr-aed-l/firered-asr-aed-l.safetensors

Then:

::

    vokra-cli convert --model firered-asr-aed-l \\
        --input firered-asr-aed-l.safetensors \\
        --output firered-asr-aed-l.gguf

# Determinism

Key ordering is the input state dict's iteration order (Python 3.7+
dicts preserve insertion order); identical inputs produce byte-identical
safetensors output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


# BatchNorm training counters — no inference role. Any other non-float
# dtype is a hard error (FR-EX-08).
DROP_SUFFIX = ".num_batches_tracked"

# Permitted float dtypes (mirror the Rust converter's pass-through arm).
ALLOWED_DTYPES = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
}


def unwrap_state_dict(obj: Any) -> "OrderedDict[str, torch.Tensor]":
    """Peel a single-key container off the pickle top level.

    Accepts the raw flat state dict, or one of the three common
    container spellings: ``{"state_dict": ...}``, ``{"model": ...}``,
    ``{"model_state_dict": ...}``. Refuses anything else so a caller
    knows to add the right key rather than debugging a phantom empty
    conversion.
    """
    if isinstance(obj, (dict, OrderedDict)):
        # If it's a flat tensor-valued dict, use it as-is.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return OrderedDict(obj)
        # Try known container keys in order of preference.
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], (dict, OrderedDict)):
                inner = obj[key]
                if inner and all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return OrderedDict(inner)
        raise SystemExit(
            f"checkpoint top level is a dict but no known container key "
            f"({{state_dict, model_state_dict, model}}) contains a flat tensor dict; "
            f"keys observed: {sorted(obj.keys())[:10]}..."
        )
    raise SystemExit(
        f"checkpoint top level is {type(obj)}, expected a flat state dict "
        f"or a single-key container ({{state_dict, model_state_dict, model}})"
    )


def dedupe_shared_storage(
    state: "OrderedDict[str, torch.Tensor]",
) -> tuple["OrderedDict[str, torch.Tensor]", list[tuple[str, str]]]:
    """Clone tensors that share storage so `safetensors.torch.save_file` accepts them.

    Returns the deduped dict + a list of ``(alias_name, canonical_name)``
    pairs that were cloned; caller writes the pair list to a
    ``.shared_pairs.json`` audit trail so a downstream consumer knows
    which upstream names were tied in the original pickle.
    """
    seen: dict[int, str] = {}
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    pairs: list[tuple[str, str]] = []
    for name, t in state.items():
        ptr = t.data_ptr()
        if ptr in seen:
            canonical = seen[ptr]
            # Clone + make contiguous — the destination is now genuinely
            # independent storage the safetensors writer will accept.
            out[name] = t.detach().clone().contiguous()
            pairs.append((name, canonical))
        else:
            seen[ptr] = name
            out[name] = t.detach().contiguous()
    return out, pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        required=True,
        help="upstream FireRedTeam/FireRedASR-AED-L torch pickle (model.pth.tar or model.pt)",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="output .safetensors path (the Rust converter's input)",
    )
    args = ap.parse_args()

    print(f"loading checkpoint: {args.ckpt}")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(raw)
    print(f"unwrapped state dict: {len(state)} entries")

    # Drop BatchNorm training counters explicitly + reject any other
    # non-float dtype (FR-EX-08 no silent fallback).
    filtered: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped: list[str] = []
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"non-tensor state entry {name!r}: {type(t)}")
        if name.endswith(DROP_SUFFIX):
            dropped.append(name)
            continue
        if t.dtype not in ALLOWED_DTYPES:
            raise SystemExit(
                f"unexpected dtype {t.dtype} for tensor {name!r}: "
                f"the safetensors pass-through path handles only "
                f"F32 / F16 / BF16 (FR-EX-08 no silent fallback)"
            )
        filtered[name] = t

    # Dedupe shared storage (tied embeddings etc.) — safetensors.torch
    # refuses two names pointing at the same storage.
    deduped, shared_pairs = dedupe_shared_storage(filtered)

    # Write safetensors.
    save_file(deduped, args.output)

    # Emit a shared_pairs.json audit trail alongside the output — a
    # future runtime binding needs to know which upstream names were
    # tied so it can restore the tie internally if that matters.
    audit_path = Path(f"{args.output}.shared_pairs.json")
    audit_path.write_text(
        json.dumps(
            {"shared_pairs": [{"alias": a, "canonical": c} for (a, c) in shared_pairs]},
            indent=2,
        )
    )

    # Report.
    for name in dropped:
        print(f"dropped (BatchNorm training counter): {name}")
    for alias, canonical in shared_pairs:
        print(f"deduped shared storage: {alias} → {canonical} (cloned)")

    total_params = sum(t.numel() for t in deduped.values())
    sha = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(f"{sha}  {args.output}")
    print(f"tensors={len(deduped)} params={total_params} dropped={len(dropped)} shared_pairs={len(shared_pairs)}")
    print(f"audit trail: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
