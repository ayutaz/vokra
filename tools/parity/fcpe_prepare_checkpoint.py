#!/usr/bin/env python3
"""Flatten an upstream FCPE ``.pt`` → safetensors under verbatim upstream
state-dict names (M5-16 / FR-OP-83).

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

Upstream: ``CNChTu/FCPE`` (MIT). The reference release ships a
``torchfcpe.model.CFNaiveMelPE`` state dict serialized as a torch pickle
(``fcpe_c_v001.pt`` — bundled inside every
``torchfcpe-0.0.{1..4}-py3-none-any.whl`` unchanged). The Rust converter
(``crates/vokra-convert/src/models/fcpe.rs``) consumes safetensors only;
this script bridges the two.

# Rewritten 2026-07-30 — actual CFNaiveMelPEInfer topology

The prior version of this script targeted an attention-based Conformer
layout (``net.layer_stack.{i}.attn.wq/wk/wv/wo``) that the released
checkpoint does **not** contain. Inspection of the actual state dict
shows the released model is ``CFNaiveMelPEInfer`` with
``conv_only=True`` — a GLU-only Sequential encoder with no attention
weights (see upstream ``torchfcpe/model_conformer_naive.py::
CFNEncoderLayer`` + ``ConformerConvModule``, and
``torchfcpe/models.py::CFNaiveMelPE``). Attempting to run the old prep
script against the real checkpoint failed loudly on
``missing upstream tensor `net.layer_stack.0.norm1.weight```. This
rewrite matches the released state dict verbatim.

# Contract

The safetensors this script emits carries the upstream state-dict names
**verbatim** (matching the Silero / FSMN-VAD "upstream names verbatim"
posture), with two documented, per-tensor transformations:

1. ``output_proj.weight_g`` + ``output_proj.weight_v`` are folded into a
   plain ``output_proj.weight``. Upstream wraps the final Linear in
   ``torch.nn.utils.weight_norm(dim=0)`` which reparametrises the weight
   as ``weight = weight_g * (weight_v / ‖weight_v‖_dim=1)`` at every
   forward; the runtime binds the folded weight so the forward is a
   single Linear pass. ``output_proj.bias`` passes through unchanged.

2. The upstream state dict carries two non-trainable buffers
   (``cent_table`` and ``gaussian_blurred_cent_mask``) that the Vokra
   runtime re-computes from ``fmin``/``fmax``/``n_pitch_bins`` at load
   time — this script drops them.

3. ``net.encoder_layers.{i}.norm.weight/bias`` is a leftover LayerNorm
   declared by ``CFNEncoderLayer.__init__`` for the attention branch
   (``self.norm``) — never called at inference when ``conv_only=True``.
   The script keeps these tensors so the emitted safetensors is a
   superset of what the runtime consumes; the runtime binder ignores
   them (they are not part of the trigger set).

# Emitted tensor names (verbatim from ``CFNaiveMelPEInfer.state_dict()``)

Stem (``input_stack`` = ``nn.Sequential(Conv1d, GroupNorm, LeakyReLU, Conv1d)``):

::

    input_stack.0.weight   [d_model, n_mels, stem_kernel]
    input_stack.0.bias     [d_model]
    input_stack.1.weight   [d_model]   # GroupNorm gamma
    input_stack.1.bias     [d_model]   # GroupNorm beta
    input_stack.3.weight   [d_model, d_model, stem_kernel]
    input_stack.3.bias     [d_model]

Encoder (``net.encoder_layers[i].conformer.net`` = ``nn.Sequential(
LayerNorm, Transpose, Conv1d, GLU, DepthWiseConv1d, SiLU, Conv1d,
Transpose, Dropout)``, per layer ``i`` in ``0..n_layers``):

::

    net.encoder_layers.{i}.conformer.net.0.weight   [d_model]           # LayerNorm gamma
    net.encoder_layers.{i}.conformer.net.0.bias     [d_model]           # LayerNorm beta
    net.encoder_layers.{i}.conformer.net.2.weight   [ffn_dim, d_model, 1]   # Conv1d pointwise
    net.encoder_layers.{i}.conformer.net.2.bias     [ffn_dim]
    net.encoder_layers.{i}.conformer.net.4.conv.weight   [ffn_dim/2, 1, conv_kernel]  # DepthwiseConv1d wraps a Conv1d under `.conv`
    net.encoder_layers.{i}.conformer.net.4.conv.bias     [ffn_dim/2]
    net.encoder_layers.{i}.conformer.net.6.weight   [d_model, ffn_dim/2, 1]   # Conv1d pointwise back
    net.encoder_layers.{i}.conformer.net.6.bias     [d_model]

Plus the unused (retained for state-dict completeness — runtime ignores):

::

    net.encoder_layers.{i}.norm.weight   [d_model]   # unused when conv_only=True
    net.encoder_layers.{i}.norm.bias     [d_model]

Head (``CFNaiveMelPE.norm`` + ``CFNaiveMelPE.output_proj``):

::

    norm.weight              [d_model]
    norm.bias                [d_model]
    output_proj.weight       [n_pitch_bins, d_model]   # folded from weight_g/weight_v
    output_proj.bias         [n_pitch_bins]

Dropped (buffers, re-computed at load):

::

    cent_table                     [n_pitch_bins]
    gaussian_blurred_cent_mask     scalar

If any expected upstream name is missing, this script fails loudly rather
than silently substituting — FR-EX-08 posture inherited from
``dfn3_prepare_checkpoint.py``.

# Usage

::

    uv run python tools/parity/fcpe_prepare_checkpoint.py \\
        --ckpt ~/.cache/vokra-eval/weights/fcpe/fcpe_c_v001.pt \\
        --output ~/.cache/vokra-eval/weights/fcpe/fcpe.safetensors

Then:

::

    vokra-cli convert --model fcpe --input fcpe.safetensors --output fcpe.gguf
"""

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict

import torch  # type: ignore[import-not-found]

# Upstream torchfcpe FCPE_v001 reference layout. A checkpoint with a
# different layer count reveals itself here (loudly) rather than silently
# succeeding with truncated weights.
DEFAULT_N_LAYERS = 6

DTYPE_MAP = {
    torch.float32: "F32",
    torch.float16: "F16",
}


def write_safetensors(path: str, tensors: "OrderedDict[str, torch.Tensor]") -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length +
    JSON header + contiguous little-endian tensor data. Mirror of the
    DFN3 prep script's writer — a shared dependency-free path so the eval
    venv needs no ``safetensors`` package."""
    header: dict[str, dict[str, object]] = {}
    blobs: list[bytes] = []
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


def _pick(state: "OrderedDict[str, torch.Tensor]", key: str) -> torch.Tensor:
    """Look up ``key`` in ``state``, failing loudly on absence — FR-EX-08."""
    if key not in state:
        raise SystemExit(f"missing upstream tensor `{key}` — refusing to substitute")
    return state[key]


def _fold_weight_norm(state: "OrderedDict[str, torch.Tensor]", base: str) -> torch.Tensor:
    """Fold torch's ``weight_norm`` reparameterization
    (``weight = weight_g * weight_v / ‖weight_v‖_dim=0``) back to a single
    weight tensor.

    Upstream FCPE wraps ``output_proj`` in ``nn.utils.weight_norm`` with the
    default ``dim=0``, so the state dict carries ``output_proj.weight_g``
    (shape ``[out_dims, 1]``, per-output scalar) and ``output_proj.weight_v``
    (shape ``[out_dims, in_dims]``, the direction) rather than a plain
    ``weight``. Vokra's runtime binds the folded weight verbatim.
    """
    g_key = f"{base}.weight_g"
    v_key = f"{base}.weight_v"
    w_key = f"{base}.weight"
    if w_key in state:
        return state[w_key]
    if g_key in state and v_key in state:
        g = state[g_key]  # [out, 1] (dim=0 → norm over non-first axes)
        v = state[v_key]  # [out, in]
        # weight = g * (v / ||v||_2 over dim=1). For a plain Linear the
        # non-first axes are just dim=1 so this is v.norm(dim=1, keepdim=True).
        v_flat = v.reshape(v.size(0), -1)
        norm = v_flat.norm(dim=1, keepdim=True)  # [out, 1]
        # Reshape g and norm to broadcast back over v's shape.
        g_bcast = g.reshape(v.size(0), *([1] * (v.dim() - 1)))
        norm_bcast = norm.reshape(v.size(0), *([1] * (v.dim() - 1)))
        w = g_bcast * v / (norm_bcast + 1e-12)
        return w
    raise SystemExit(
        f"missing weight-normed tensor at `{base}` "
        "(neither `weight` nor `weight_g`+`weight_v` present)"
    )


def build_expected_names(n_layers: int) -> list[str]:
    """The complete verbatim tensor-name list this script writes."""
    names = [
        "input_stack.0.weight",
        "input_stack.0.bias",
        "input_stack.1.weight",
        "input_stack.1.bias",
        "input_stack.3.weight",
        "input_stack.3.bias",
    ]
    for i in range(n_layers):
        p = f"net.encoder_layers.{i}"
        names.extend(
            [
                f"{p}.conformer.net.0.weight",
                f"{p}.conformer.net.0.bias",
                f"{p}.conformer.net.2.weight",
                f"{p}.conformer.net.2.bias",
                f"{p}.conformer.net.4.conv.weight",
                f"{p}.conformer.net.4.conv.bias",
                f"{p}.conformer.net.6.weight",
                f"{p}.conformer.net.6.bias",
                # Retained but ignored by runtime — a leftover LayerNorm
                # from CFNEncoderLayer.__init__ that never runs when
                # conv_only=True.
                f"{p}.norm.weight",
                f"{p}.norm.bias",
            ]
        )
    names.extend(
        [
            "norm.weight",
            "norm.bias",
            # `output_proj.weight` is the folded result (written last).
            "output_proj.weight",
            "output_proj.bias",
        ]
    )
    return names


# Upstream buffers we deliberately drop (re-computed at runtime from
# `fmin` / `fmax` / `n_pitch_bins`).
UPSTREAM_BUFFERS_TO_DROP: set[str] = {
    "cent_table",
    "gaussian_blurred_cent_mask",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="upstream FCPE checkpoint (.pt / torch pickle)")
    ap.add_argument("--output", required=True, help="output .safetensors path")
    ap.add_argument(
        "--n-layers",
        type=int,
        default=DEFAULT_N_LAYERS,
        help=f"encoder layer count (default {DEFAULT_N_LAYERS} for FCPE_v001)",
    )
    args = ap.parse_args()

    # weights_only=False required because the released checkpoint uses the
    # torch pickle format (torchfcpe wraps the state dict in a plain dict).
    # The file is downloaded from a fixed HF/GitHub release, not user input.
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], (dict, OrderedDict)):
        state = state["model"]
    if not isinstance(state, (dict, OrderedDict)):
        raise SystemExit(f"checkpoint top level is {type(state)}, expected a state dict")

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    consumed: set[str] = set()

    # Stem — verbatim.
    for name in [
        "input_stack.0.weight",
        "input_stack.0.bias",
        "input_stack.1.weight",
        "input_stack.1.bias",
        "input_stack.3.weight",
        "input_stack.3.bias",
    ]:
        out[name] = _pick(state, name)
        consumed.add(name)

    # Encoder — verbatim per layer.
    for i in range(args.n_layers):
        p = f"net.encoder_layers.{i}"
        for tail in [
            "conformer.net.0.weight",
            "conformer.net.0.bias",
            "conformer.net.2.weight",
            "conformer.net.2.bias",
            "conformer.net.4.conv.weight",
            "conformer.net.4.conv.bias",
            "conformer.net.6.weight",
            "conformer.net.6.bias",
            "norm.weight",
            "norm.bias",
        ]:
            name = f"{p}.{tail}"
            out[name] = _pick(state, name)
            consumed.add(name)

    # Top-level LayerNorm + folded output projection.
    out["norm.weight"] = _pick(state, "norm.weight")
    out["norm.bias"] = _pick(state, "norm.bias")
    consumed.update({"norm.weight", "norm.bias"})
    out["output_proj.weight"] = _fold_weight_norm(state, "output_proj")
    out["output_proj.bias"] = _pick(state, "output_proj.bias")
    for tail in ("weight", "weight_g", "weight_v", "bias"):
        consumed.add(f"output_proj.{tail}")

    # Drop known buffers.
    for name in UPSTREAM_BUFFERS_TO_DROP:
        if name in state:
            consumed.add(name)

    # Loud sanity: any un-consumed upstream tensor is either a variant
    # this script does not yet handle or a hidden hparam. Reported, not
    # silently included.
    dropped: list[str] = []
    for name in state.keys():
        if name in consumed:
            continue
        dropped.append(name)

    write_safetensors(args.output, out)
    for name in dropped:
        print(f"dropped (unrecognised, not written): {name}")
    if dropped:
        print(
            "WARNING: unrecognised upstream tensors above may indicate a "
            "checkpoint variant this script does not yet map."
        )

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    print(f"{sha}  {args.output}")
    total = sum(t.numel() for t in out.values())
    print(f"tensors={len(out)} params={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
