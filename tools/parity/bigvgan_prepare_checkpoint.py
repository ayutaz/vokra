#!/usr/bin/env python3
"""Flatten a NVIDIA BigVGAN ``bigvgan_generator.pt`` → safetensors (SoTA plan
Phase D2-D5, 2026-07-31 land).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the runtime).
The upstream NVIDIA/BigVGAN releases ship ``bigvgan_generator.pt`` — a torch
pickle whose top level is ``{"generator": OrderedDict[str, Tensor]}`` (see
upstream ``inference.py``
``generator.load_state_dict(state_dict_g["generator"])`` and
``utils.py::load_checkpoint`` which calls
``torch.load(filepath, map_location=device)`` and returns the wrapper dict
as-is). Vokra's Rust converter
(``crates/vokra-convert/src/models/bigvgan.rs``) consumes safetensors only;
this script bridges the two.

# Contract

Writes the flat generator state_dict as safetensors, preserving every
upstream tensor name verbatim (``conv_pre.weight``, ``ups.{i}.0.weight``,
``resblocks.{n}.convs1.{k}.weight``, ``resblocks.{n}.convs2.{k}.weight``,
``resblocks.{n}.activations.{k}.act.alpha`` (+ ``.beta`` for snakebeta),
``resblocks.{n}.activations.{k}.upsample.filter`` /
``.downsample.lowpass.filter`` for anti-aliased Activation1d Kaiser buffers,
``conv_post.weight``, all biases). BF16 stays BF16, F16 stays F16, F32
stays F32 — no convert-time widening (the Rust side owns dtype
normalization at load; sibling mirror of ``convert_bigvgan_file``'s BF16
pass-through arm).

One documented transformation: torch's ``nn.utils.weight_norm`` (default
``dim=0``) reparameterises every ``Conv1d`` / ``ConvTranspose1d`` in the
BigVGAN generator as ``weight = weight_g * (weight_v / ‖weight_v‖2_{axes≠0})``.
This script folds every ``<base>.weight_g`` + ``<base>.weight_v`` pair
into a plain ``<base>.weight`` at write time — exact match of upstream
``bigvgan.py``'s post-``remove_weight_norm()`` state so that the Rust
runt's ``bigvgan_generator`` op skeleton binds a single Linear/Conv
weight without re-implementing the reparametrisation. Mirror of the
``fcpe_prepare_checkpoint._fold_weight_norm`` pattern (which folds the
same ``output_proj`` weight_norm form).

Emits a side-car ``config.json`` verbatim from the upstream ``config.json``
next to ``bigvgan_generator.pt``, so the future native BigVGAN loader
can read hparams (num_mels, sampling_rate, upsample_rates,
upsample_kernel_sizes, upsample_initial_channel, resblock_kernel_sizes,
resblock_dilation_sizes, activation, snake_logscale, use_bias_at_final,
use_tanh_at_final) without re-downloading the release. The current Rust
``convert_bigvgan_file`` does not consume the config JSON — the variant is
picked by slug and every hparam is shape-derived at bind time (FR-EX-08
authoritative gate) — so the config side-car is a provenance artifact,
not a converter input.

Variants (``--variant`` selector, cross-checked against config.json):
  * ``v2_22khz_80band_256x``   – ``nvidia/bigvgan_v2_22khz_80band_256x``
  * ``v2_24khz_100band_256x``  – ``nvidia/bigvgan_v2_24khz_100band_256x``
  * ``v2_44khz_128band_512x``  – ``nvidia/bigvgan_v2_44khz_128band_512x``
  * ``base_v1_24khz_100band``  – ``nvidia/bigvgan_base_24khz_100band``

The triple ``(num_mels, sampling_rate, upsample_initial_channel)``
uniquely identifies each variant (v2_24khz and base_v1_24khz share the
(100, 24000) pair — disambiguated by upsample_initial_channel 1536 vs
512). A ``--variant`` that does not match the config triggers a loud
refuse rather than silently writing a mis-labeled safetensors, which
would then flow into ``convert_bigvgan_file``'s
``vokra.bigvgan.variant`` GGUF chunk and mis-route the future runtime
dispatch (FR-EX-08 posture inherited from ``dfn3_prepare_checkpoint`` /
``dac_prepare_checkpoint``).

# Usage

::

    uv --directory tools/parity run python bigvgan_prepare_checkpoint.py \\
        --pt   /tmp/bigvgan_v2_22khz/bigvgan_generator.pt \\
        --config /tmp/bigvgan_v2_22khz/config.json \\
        --variant v2_22khz_80band_256x \\
        --output /tmp/bigvgan_v2_22khz/bigvgan.safetensors \\
        --config-out /tmp/bigvgan_v2_22khz/config.publish.json

Then::

    vokra-cli convert --model bigvgan_v2_22khz_80band_256x \\
        --input  /tmp/bigvgan_v2_22khz/bigvgan.safetensors \\
        --output /tmp/bigvgan_v2_22khz/bigvgan.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import torch  # type: ignore[import-not-found]

# torch dtype -> safetensors dtype tag. BF16 goes out as safetensors
# ``BF16`` (numpy has no bfloat16, so we reinterpret via ``view(int16)``
# just to reach ``.numpy().tobytes()`` — a bit-preserving cast, not a
# widening).
DTYPE_MAP = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
}

# variant → (num_mels, sampling_rate, upsample_initial_channel).
# Verified 2026-07-31 by fetching each variant's config.json from HF:
#   nvidia/bigvgan_v2_22khz_80band_256x      → (80,  22050, 1536)
#   nvidia/bigvgan_v2_24khz_100band_256x     → (100, 24000, 1536)
#   nvidia/bigvgan_v2_44khz_128band_512x     → (128, 44100, 1536)
#   nvidia/bigvgan_base_24khz_100band        → (100, 24000, 512)
# v2_24khz and base_v1_24khz share (num_mels, sampling_rate) so
# upsample_initial_channel is the required tie-breaker.
VARIANTS: dict[str, tuple[int, int, int]] = {
    "v2_22khz_80band_256x":   (80,  22050, 1536),
    "v2_24khz_100band_256x":  (100, 24000, 1536),
    "v2_44khz_128band_512x":  (128, 44100, 1536),
    "base_v1_24khz_100band":  (100, 24000, 512),
}

VARIANT_UPSTREAM_HF: dict[str, str] = {
    "v2_22khz_80band_256x":  "nvidia/bigvgan_v2_22khz_80band_256x",
    "v2_24khz_100band_256x": "nvidia/bigvgan_v2_24khz_100band_256x",
    "v2_44khz_128band_512x": "nvidia/bigvgan_v2_44khz_128band_512x",
    "base_v1_24khz_100band": "nvidia/bigvgan_base_24khz_100band",
}


def write_safetensors(path: str, tensors: "OrderedDict[str, torch.Tensor]") -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header length +
    JSON header + contiguous little-endian tensor payload. Mirror of the
    dfn3 / fcpe prep-script writer — shared dependency-free path so the
    parity venv needs no ``safetensors`` package to run this script."""
    header: dict[str, dict[str, object]] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        dt = DTYPE_MAP.get(t.dtype)
        if dt is None:
            raise SystemExit(
                f"bigvgan_prepare: unsupported dtype {t.dtype} for tensor {name!r} — "
                "only F32 / F16 / BF16 pass through"
            )
        cpu = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            # numpy has no bfloat16; view() is a bit-preserving reinterpret
            # cast (same 2-byte little-endian layout the safetensors reader
            # expects for BF16) so downstream Rust's decode_bf16 sees the
            # exact upstream bytes.
            data = cpu.view(torch.int16).numpy().tobytes()
        else:
            data = cpu.numpy().tobytes()
        header[name] = {
            "dtype": dt,
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


def fold_weight_norm(state: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    """Fold every ``<base>.weight_g`` + ``<base>.weight_v`` pair (as produced
    by ``torch.nn.utils.weight_norm`` with default ``dim=0``) into a plain
    ``<base>.weight``. Returns a NEW OrderedDict with:

      * folded ``<base>.weight`` for every base that had (weight_g, weight_v);
      * all other tensors (biases, ``.alpha`` / ``.beta`` for Snake/SnakeBeta,
        anti-aliased upsample/downsample Kaiser filter buffers) passed
        through verbatim under their upstream names.

    Fails loudly on any dangling ``.weight_g`` without ``.weight_v`` (or vice
    versa) — FR-EX-08. A pre-existing plain ``.weight`` at a base that also
    carries ``.weight_g`` + ``.weight_v`` is dropped in favour of the folded
    weight (a pre-``remove_weight_norm()`` checkpoint that also carries the
    materialised weight; the fold is authoritative).

    Math (matches PyTorch's ``torch._weight_norm`` for ``dim=0``, so
    round-tripping through this fold matches ``model.remove_weight_norm()``
    byte-for-byte, up to fp rounding):

        v_flat = v.reshape(v.size(0), -1)          # [out, prod(rest)]
        norm   = v_flat.norm(dim=1, keepdim=True)  # [out, 1]
        weight = g.broadcast_as(v) * v / norm.broadcast_as(v)

    Adds a small (1e-12) floor to the divisor to match numerical practice
    (a zero-norm filter never occurs in a trained checkpoint; the floor is
    defensive against a corrupt input).
    """
    # Pass 1: find every base with weight_norm sides and detect dangling pairs.
    bases_with_norm: set[str] = set()
    for name in state.keys():
        if name.endswith(".weight_g"):
            base = name[: -len(".weight_g")]
            if base + ".weight_v" not in state:
                raise SystemExit(
                    f"bigvgan_prepare: dangling weight_norm at `{name}` — "
                    f"missing sibling `{base}.weight_v` (refusing to write a "
                    "partial fold)"
                )
            bases_with_norm.add(base)
        elif name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            if base + ".weight_g" not in state:
                raise SystemExit(
                    f"bigvgan_prepare: dangling weight_norm at `{name}` — "
                    f"missing sibling `{base}.weight_g` (refusing to write a "
                    "partial fold)"
                )

    # Pass 2: fold-or-passthrough in the state's insertion order, so the
    # folded ``.weight`` lands where the first side of the pair appeared.
    # This preserves upstream layout for readers that walk the safetensors
    # header sequentially (safetensors.parse is deterministic on JSON key
    # order).
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    folded_bases: set[str] = set()
    for name, tensor in state.items():
        if name.endswith(".weight_g") or name.endswith(".weight_v"):
            base = name[: -len(".weight_g")] if name.endswith(".weight_g") else name[: -len(".weight_v")]
            if base in folded_bases:
                continue  # already emitted the folded .weight when the other side was hit
            g = state[base + ".weight_g"]
            v = state[base + ".weight_v"]
            # ``g`` is stored with the same rank as ``v`` but with all
            # axes-except-dim set to 1 (shape ``[out, 1, 1, ...]`` for dim=0).
            # Reshape defensively in case an older torch stored it as ``[out]``.
            g_bcast = g.reshape(v.size(0), *([1] * (v.dim() - 1)))
            v_flat = v.reshape(v.size(0), -1)
            norm = v_flat.norm(dim=1, keepdim=True)  # [out, 1]
            norm_bcast = norm.reshape(v.size(0), *([1] * (v.dim() - 1)))
            weight = g_bcast * v / (norm_bcast + 1e-12)
            out[base + ".weight"] = weight
            folded_bases.add(base)
            continue
        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            if base in bases_with_norm:
                # A pre-materialised ``.weight`` alongside its weight_norm
                # sides — the fold is the authoritative source, drop the
                # duplicate.
                continue
        out[name] = tensor
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ or "bigvgan_prepare_checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pt", type=Path, required=True,
        help="upstream bigvgan_generator.pt (torch pickle from nvidia/bigvgan_*)",
    )
    ap.add_argument(
        "--config", type=Path, required=True,
        help="upstream config.json (bundled next to bigvgan_generator.pt)",
    )
    ap.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS.keys()),
        help="target BigVGAN variant — cross-checked against config.json",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="output safetensors path (fed to `vokra-cli convert --model bigvgan_*`)",
    )
    ap.add_argument(
        "--config-out", type=Path, required=True,
        help="output config.json path (verbatim copy for provenance)",
    )
    args = ap.parse_args()

    # -------- config sanity + variant cross-check --------
    if not args.config.is_file():
        print(f"bigvgan_prepare: --config not found: {args.config}", file=sys.stderr)
        return 2
    try:
        cfg = json.loads(args.config.read_text())
    except json.JSONDecodeError as e:
        print(f"bigvgan_prepare: --config is not valid JSON: {e}", file=sys.stderr)
        return 2

    required_keys = [
        "num_mels", "sampling_rate", "upsample_rates",
        "upsample_kernel_sizes", "upsample_initial_channel",
        "resblock_kernel_sizes", "resblock_dilation_sizes",
        "activation",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        print(
            f"bigvgan_prepare: config.json missing required keys: {missing}",
            file=sys.stderr,
        )
        return 2

    expected = VARIANTS[args.variant]
    actual = (
        int(cfg["num_mels"]),
        int(cfg["sampling_rate"]),
        int(cfg["upsample_initial_channel"]),
    )
    if actual != expected:
        print(
            f"bigvgan_prepare: --variant `{args.variant}` requires "
            f"(num_mels, sampling_rate, upsample_initial_channel)={expected} "
            f"but config.json reports {actual}. Refusing to write a mis-labeled "
            "safetensors — pass a --variant that matches, or verify you have "
            "the right upstream release.",
            file=sys.stderr,
        )
        return 2

    # -------- load .pt --------
    # ``weights_only=False`` is required (and safe) here: the file is
    # downloaded from a fixed NVIDIA HF release (nvidia/bigvgan_*), not user
    # input, and the pickle is the plain torch state-dict format upstream
    # loads with ``torch.load(filepath, map_location=device)`` (no
    # weights_only arg — verified 2026-07-31 by reading upstream utils.py::
    # load_checkpoint). Same posture as fcpe_prepare_checkpoint.
    state = torch.load(args.pt, map_location="cpu", weights_only=False)

    if not isinstance(state, dict) or "generator" not in state:
        keys_hint = (
            f" with top-level keys {sorted(state.keys())[:8]}"
            if isinstance(state, dict) else ""
        )
        print(
            f"bigvgan_prepare: unexpected checkpoint layout — expected top-level "
            f"dict with 'generator' key wrapping the state_dict (upstream "
            f"inference.py: `generator.load_state_dict(state_dict_g['generator'])`), "
            f"got {type(state).__name__}{keys_hint}",
            file=sys.stderr,
        )
        return 2

    generator_sd = state["generator"]
    if not isinstance(generator_sd, (dict, OrderedDict)):
        print(
            f"bigvgan_prepare: state['generator'] is {type(generator_sd).__name__}, "
            "expected OrderedDict",
            file=sys.stderr,
        )
        return 2

    # Preserve upstream insertion order (OrderedDict semantics).
    generator_od: OrderedDict[str, torch.Tensor] = OrderedDict(generator_sd)

    # Validate every entry is a tensor of a supported dtype BEFORE folding,
    # so a bad entry surfaces the offending key rather than blowing up inside
    # ``fold_weight_norm``.
    for k, v in generator_od.items():
        if not isinstance(v, torch.Tensor):
            print(
                f"bigvgan_prepare: non-tensor state entry `{k}` "
                f"({type(v).__name__})",
                file=sys.stderr,
            )
            return 2
        if v.dtype not in DTYPE_MAP:
            print(
                f"bigvgan_prepare: unsupported dtype {v.dtype} for tensor `{k}` — "
                "only F32/F16/BF16 pass through",
                file=sys.stderr,
            )
            return 2

    # -------- fold weight_norm --------
    n_pairs_in = sum(1 for k in generator_od if k.endswith(".weight_g"))
    folded = fold_weight_norm(generator_od)

    # -------- write outputs --------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(str(args.output), folded)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    # Verbatim copy of the upstream config — no field rename, no unit
    # coercion beyond ``json.dumps`` reformatting.
    args.config_out.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"variant: {args.variant} ({VARIANT_UPSTREAM_HF[args.variant]})")
    print(f"tensors in:  {len(generator_od)}")
    print(f"tensors out: {len(folded)} (weight_norm pairs folded: {n_pairs_in})")
    print(f"sha256 {args.output.name}     {sha256_of(args.output)}")
    print(f"sha256 {args.config_out.name} {sha256_of(args.config_out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
