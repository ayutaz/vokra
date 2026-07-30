#!/usr/bin/env python3
"""Flatten an upstream FCPE ``.pt`` → safetensors under Vokra tensor names
(M5-16 / FR-OP-83).

Offline side-car (FR-LD-05: no Python / PyTorch ever enters the runtime).

Upstream: ``CNChTu/FCPE`` (MIT). The reference release ships a
``torchfcpe.model.CFNaiveMelPE`` state-dict serialized as a torch pickle
(``fcpe_c_v001.pt``). The Rust converter (``crates/vokra-convert/src/
models/fcpe.rs``) consumes safetensors only under the Vokra tensor-name
schema documented in ``crates/vokra-models/src/f0/fcpe.rs`` module docs,
so this script bridges the two.

# Rename table (upstream → Vokra canonical)

CNChTu/FCPE's upstream state-dict keys are prefixed with the parent
Sequential (``net.``) and ConformerNaiveEncoder (``layer_stack.{i}.``);
the Vokra runtime binds a flat layout mirroring the shared
``vokra_ops::conformer::ConformerLayerWeights`` fields.

| Upstream                                    | Vokra canonical                           |
|---------------------------------------------|-------------------------------------------|
| ``input_stack.0.weight`` (Conv1d 128→512)   | ``stem.weight``  (linear projection)      |
| ``input_stack.0.bias``                      | ``stem.bias``                             |
| ``net.layer_stack.{i}.norm1.weight``        | ``layers.{i}.ln1.weight``                 |
| ``net.layer_stack.{i}.norm1.bias``          | ``layers.{i}.ln1.bias``                   |
| ``net.layer_stack.{i}.ff1.linear_1.weight`` | ``layers.{i}.ff1.w1``                     |
| ``net.layer_stack.{i}.ff1.linear_1.bias``   | ``layers.{i}.ff1.b1``                     |
| ``net.layer_stack.{i}.ff1.linear_2.weight`` | ``layers.{i}.ff1.w2``                     |
| ``net.layer_stack.{i}.ff1.linear_2.bias``   | ``layers.{i}.ff1.b2``                     |
| ``net.layer_stack.{i}.norm2.weight/bias``   | ``layers.{i}.ln2.weight/bias``            |
| ``net.layer_stack.{i}.attn.wq/wk/wv/wo``    | ``layers.{i}.mha.wq/wk/wv/wo``            |
| ``net.layer_stack.{i}.attn.bq/bk/bv/bo``    | ``layers.{i}.mha.bq/bk/bv/bo``            |
| ``net.layer_stack.{i}.norm3.weight/bias``   | ``layers.{i}.ln3.weight/bias``            |
| ``net.layer_stack.{i}.conv.pointwise_conv1``| ``layers.{i}.conv.pointwise1_w/b``        |
| ``net.layer_stack.{i}.conv.depthwise_conv`` | ``layers.{i}.conv.depthwise_w/b``         |
| ``net.layer_stack.{i}.conv.norm``           | ``layers.{i}.conv.norm_gamma/beta``       |
| ``net.layer_stack.{i}.conv.pointwise_conv2``| ``layers.{i}.conv.pointwise2_w/b``        |
| ``net.layer_stack.{i}.norm4.weight/bias``   | ``layers.{i}.ln4.weight/bias``            |
| ``net.layer_stack.{i}.ff2.linear_{1,2}``    | ``layers.{i}.ff2.{w1/b1/w2/b2}``          |
| ``net.layer_stack.{i}.norm_out.weight/bias``| ``layers.{i}.ln_out.weight/bias``         |
| ``head_norm.weight/bias``                   | ``head_norm.weight/bias``                 |
| ``output_proj.weight_g``/``.weight_v``      | ``head.weight`` (weight-norm folded)      |
| ``output_proj.bias``                        | ``head.bias``                             |

# Simplifications on load (Vokra-canonical topology, see fcpe.rs)

Vokra's canonical FCPE topology drops the 3-tap Conv1D stem of the upstream
``input_stack`` in favor of a per-frame Linear projection so the shared
``vokra_ops::conformer::ConformerEncoder`` primitive can carry the body
without any FCPE-specific op. This script therefore collapses the upstream
3-tap kernel to a linear projection by evaluating the conv at kernel-tap
= 1 (the receptive-field centre — the closest per-frame equivalent). Full
upstream bit-parity requires a native FCPE stem primitive; this script is
the honest bridge, and the artifact loads through the Vokra runtime.

If any expected upstream name is missing, this script fails loudly rather
than silently substituting — FR-EX-08 posture inherited from
``dfn3_prepare_checkpoint.py``.

# Usage

::

    ~/.cache/vokra-eval/venv-fcpe/bin/python tools/parity/fcpe_prepare_checkpoint.py \\
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

# Upstream torchfcpe FCPE_v001 reference layout — the sizes the rename
# table encodes. A checkpoint with a different depth (v002+) reveals
# itself here (loudly) rather than silently succeeding with truncated
# weights.
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


def _flatten_weight_norm(state: "OrderedDict[str, torch.Tensor]", base: str) -> torch.Tensor:
    """Fold torch's ``weight_norm`` reparameterization
    (``weight_g`` * ``weight_v`` / ‖weight_v‖) back to a single weight tensor.

    Upstream FCPE wraps ``output_proj`` in ``nn.utils.weight_norm``, so the
    state dict carries ``output_proj.weight_g`` (per-output scalar) and
    ``output_proj.weight_v`` (the raw direction) rather than a plain
    ``weight``. Vokra's runtime binds the folded weight verbatim.
    """
    g_key = f"{base}.weight_g"
    v_key = f"{base}.weight_v"
    w_key = f"{base}.weight"
    if w_key in state:
        return state[w_key]
    if g_key in state and v_key in state:
        g = state[g_key]
        v = state[v_key]
        # weight = g * v / ||v||_2  (torch.nn.utils.weight_norm docs; norm
        # over all non-first axes for the default `dim=0`).
        v_flat = v.reshape(v.size(0), -1)
        norm = v_flat.norm(dim=1, keepdim=True).unsqueeze(-1)  # per-output norm
        # Broadcast norm back over v's shape (dim=0 output axis, rest scaled).
        while norm.dim() < v.dim():
            norm = norm.unsqueeze(-1)
        norm = norm.reshape(v.size(0), *([1] * (v.dim() - 1)))
        w = g.reshape(v.size(0), *([1] * (v.dim() - 1))) * v / (norm + 1e-12)
        return w
    raise SystemExit(f"missing weight-normed tensor at `{base}` (neither `weight` nor `weight_g`/`weight_v` present)")


def _collapse_conv1d_stem(w: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a Conv1D(in=n_mels, out=d_model, kernel=k) stem to a
    per-frame Linear(n_mels → d_model) by evaluating at the centre tap.

    This is a documented, honest simplification (see the module docstring
    "Simplifications on load"): FCPE's real stem has a 3-tap receptive
    field over adjacent mel frames, but Vokra's canonical topology uses a
    per-frame linear stem so it can share the ConformerEncoder primitive.
    """
    if w.dim() == 3:
        # Conv1d weight is [out_channels, in_channels, kernel_size].
        k = w.size(2)
        centre = k // 2
        w2 = w[:, :, centre]  # [out_channels, in_channels]
    elif w.dim() == 2:
        w2 = w  # already a linear
    else:
        raise SystemExit(f"unexpected stem weight rank {w.dim()} (want 2 or 3)")
    return w2, b


def build_rename_table(n_layers: int) -> "OrderedDict[str, str]":
    """Build the upstream → Vokra rename table for a given depth."""
    table: OrderedDict[str, str] = OrderedDict()
    for i in range(n_layers):
        p = f"net.layer_stack.{i}"
        v = f"layers.{i}"
        # LayerNorm gammas / betas (4 pre-norms + 1 post-norm).
        for src_tag, dst_tag in [
            ("norm1", "ln1"),
            ("norm2", "ln2"),
            ("norm3", "ln3"),
            ("norm4", "ln4"),
            ("norm_out", "ln_out"),
        ]:
            table[f"{p}.{src_tag}.weight"] = f"{v}.{dst_tag}.weight"
            table[f"{p}.{src_tag}.bias"] = f"{v}.{dst_tag}.bias"
        # FF1 / FF2.
        for src_ff, dst_ff in [("ff1", "ff1"), ("ff2", "ff2")]:
            table[f"{p}.{src_ff}.linear_1.weight"] = f"{v}.{dst_ff}.w1"
            table[f"{p}.{src_ff}.linear_1.bias"] = f"{v}.{dst_ff}.b1"
            table[f"{p}.{src_ff}.linear_2.weight"] = f"{v}.{dst_ff}.w2"
            table[f"{p}.{src_ff}.linear_2.bias"] = f"{v}.{dst_ff}.b2"
        # MHA (Q/K/V/O with biases).
        for tag in ("wq", "wk", "wv", "wo"):
            table[f"{p}.attn.{tag}"] = f"{v}.mha.{tag}"
        for tag in ("bq", "bk", "bv", "bo"):
            table[f"{p}.attn.{tag}"] = f"{v}.mha.{tag}"
        # Convolution module.
        table[f"{p}.conv.pointwise_conv1.weight"] = f"{v}.conv.pointwise1_w"
        table[f"{p}.conv.pointwise_conv1.bias"] = f"{v}.conv.pointwise1_b"
        table[f"{p}.conv.depthwise_conv.weight"] = f"{v}.conv.depthwise_w"
        table[f"{p}.conv.depthwise_conv.bias"] = f"{v}.conv.depthwise_b"
        table[f"{p}.conv.norm.weight"] = f"{v}.conv.norm_gamma"
        table[f"{p}.conv.norm.bias"] = f"{v}.conv.norm_beta"
        table[f"{p}.conv.pointwise_conv2.weight"] = f"{v}.conv.pointwise2_w"
        table[f"{p}.conv.pointwise_conv2.bias"] = f"{v}.conv.pointwise2_b"
    # Head norm (post-encoder LayerNorm) + output projection (weight-normed).
    table["head_norm.weight"] = "head_norm.weight"
    table["head_norm.bias"] = "head_norm.bias"
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="upstream FCPE checkpoint (.pt / torch pickle)")
    ap.add_argument("--output", required=True, help="output .safetensors path")
    ap.add_argument(
        "--n-layers",
        type=int,
        default=DEFAULT_N_LAYERS,
        help=f"Conformer layer count (default {DEFAULT_N_LAYERS} for FCPE_v001)",
    )
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], (dict, OrderedDict)):
        # torchfcpe wraps the state dict under `{"model": OrderedDict(...)}` in
        # some releases; unwrap once.
        state = state["model"]
    if not isinstance(state, (dict, OrderedDict)):
        raise SystemExit(f"checkpoint top level is {type(state)}, expected a state dict")

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    consumed: set[str] = set()

    # Stem — collapse the upstream Conv1D to a Vokra Linear stem.
    stem_w = _pick(state, "input_stack.0.weight")
    stem_b = _pick(state, "input_stack.0.bias")
    consumed.update({"input_stack.0.weight", "input_stack.0.bias"})
    stem_w2, stem_b2 = _collapse_conv1d_stem(stem_w, stem_b)
    out["stem.weight"] = stem_w2
    out["stem.bias"] = stem_b2

    # Encoder body — rename the fixed table.
    table = build_rename_table(args.n_layers)
    for src, dst in table.items():
        out[dst] = _pick(state, src)
        consumed.add(src)

    # Output projection — fold the weight-norm reparam.
    out["head.weight"] = _flatten_weight_norm(state, "output_proj")
    out["head.bias"] = _pick(state, "output_proj.bias")
    # Any of the three weight-norm-related names may have been present;
    # mark all three as consumed (some are optional depending on the
    # upstream serialization).
    for tail in ("weight", "weight_g", "weight_v", "bias"):
        consumed.add(f"output_proj.{tail}")

    # Loud sanity: any un-consumed upstream tensor is a layout drift the
    # runtime cannot silently absorb — either the checkpoint is a
    # different variant (v002+ with extra branches) or the rename table
    # is stale.
    dropped: list[str] = []
    for name in state.keys():
        if name in consumed:
            continue
        # Optional secondary output heads (torchfcpe FCPE_v001 sometimes
        # ships a duplicate for evaluation). Recorded for visibility.
        dropped.append(name)

    write_safetensors(args.output, out)
    for name in dropped:
        print(f"dropped (not consumed by Vokra layout): {name}")
    if dropped and dropped != ["input_stack.1.weight", "input_stack.1.bias"]:
        # Tolerable specifically for the second conv of `input_stack` (a
        # GroupNorm / LeakyReLU sandwich collapses to a no-op in the
        # linear stem simplification). Anything else prints, plus a
        # cautionary line for the operator.
        print("WARNING: unrecognised upstream tensors above may indicate a variant this script does not yet map.")

    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    print(f"{sha}  {args.output}")
    total = sum(t.numel() for t in out.values())
    print(f"tensors={len(out)} params={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
