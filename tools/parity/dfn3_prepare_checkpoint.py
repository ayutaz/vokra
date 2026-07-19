#!/usr/bin/env python3
"""Flatten a DeepFilterNet3 ``model_*.ckpt.best`` → safetensors (M4-20 T17).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the runtime).
The upstream Rikorose/DeepFilterNet release ships ``models/DeepFilterNet3.zip``
→ ``DeepFilterNet3/checkpoints/model_120.ckpt.best`` — a torch-pickle file
whose top level is the flat ``OrderedDict`` state dict itself (133 entries,
2.168 M params incl. buffers). The Rust converter
(``crates/vokra-convert/src/models/denoise.rs``) consumes safetensors only,
so this script bridges the two:

* Loads the ``.ckpt.best`` with ``torch.load(..., weights_only=True)`` (the
  release checkpoint loads cleanly under the safe loader — verified on the
  sha256 ``49c52edc…`` file; a checkpoint that does not is refused rather
  than falling back to unsafe unpickling).
* Writes the flat state dict (dotted upstream keys preserved:
  ``enc.erb_conv0.1.weight`` / ``erb_dec.emb_gru.gru.weight_ih_l0`` /
  ``mask.erb_inv_fb`` / …) as safetensors. The writer is hand-rolled
  (stdlib ``json`` + raw bytes) so the eval venv needs no ``safetensors``
  package; the format is the standard one vokra-core parses.
* F32 tensors are written as-is. The 15 ``*.num_batches_tracked`` I64
  BatchNorm training counters are DROPPED here (explicitly, each one
  reported) — they have no inference role (eval-mode BatchNorm uses only
  running_mean/running_var/weight/bias) and vokra-core's safetensors
  parser is float-only by design. Any *other* non-F32 tensor is a hard
  error, never a silent drop (FR-EX-08 posture).
* Prints a sha256 manifest line per output for the fixture / workflow logs.

Fails loudly on any anomaly (non-tensor entry, unexpected dtype) rather than
masking it — FR-EX-08 posture.

# Usage

::

    ~/.cache/vokra-eval/venv-dfn3/bin/python tools/parity/dfn3_prepare_checkpoint.py \\
        --ckpt ~/.cache/vokra-eval/weights/dfn3/DeepFilterNet3/checkpoints/model_120.ckpt.best \\
        --output ~/.cache/vokra-eval/weights/dfn3/dfn3.safetensors

Then:

::

    vokra-cli convert --model denoise --input dfn3.safetensors --output dfn3.gguf
"""

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict

import torch

DTYPE_MAP = {
    torch.float32: "F32",
    torch.int64: "I64",
}


def write_safetensors(path: str, tensors: "OrderedDict[str, torch.Tensor]") -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length +
    JSON header + contiguous little-endian tensor data."""
    header = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        if t.dtype not in DTYPE_MAP:
            raise SystemExit(f"unsupported dtype {t.dtype} for tensor {name!r}")
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
    ap.add_argument("--ckpt", required=True, help="model_*.ckpt.best (torch pickle)")
    ap.add_argument("--output", required=True, help="output .safetensors path")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, (dict, OrderedDict)):
        raise SystemExit(f"checkpoint top level is {type(state)}, expected a flat state dict")

    out = OrderedDict()
    n_params = 0
    dropped = []
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(f"non-tensor state entry {name!r}: {type(t)}")
        if name.endswith(".num_batches_tracked"):
            # BatchNorm training counter (I64 scalar): no inference role.
            dropped.append(name)
            continue
        if t.dtype != torch.float32:
            raise SystemExit(f"unexpected dtype {t.dtype} for tensor {name!r}")
        out[name] = t
        n_params += t.numel()

    write_safetensors(args.output, out)
    for name in dropped:
        print(f"dropped (BatchNorm training counter): {name}")

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    print(f"{sha}  {args.output}")
    print(f"tensors={len(out)} params={n_params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
