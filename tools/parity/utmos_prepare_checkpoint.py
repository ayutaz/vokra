#!/usr/bin/env python3
"""Flatten the UTMOS22-strong ``.ckpt`` → safetensors + config side-car (M5-15 T14).

An **offline** sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). Upstream ships a PyTorch-Lightning checkpoint
(``epoch=3-step=7459.ckpt`` on the ``sarulab-speech/UTMOS-demo`` HF space);
the Rust converter (``crates/vokra-convert/src/models/utmos.rs``) reads
safetensors + JSON only, so this script bridges the two:

* loads the ckpt on CPU and takes its ``state_dict`` verbatim — the dotted
  upstream keys are preserved, nothing is renamed here (the Rust converter
  owns the name mapping so the mapping is covered by its unit tests, exactly
  as ``dac_prepare_checkpoint.py`` leaves the weight-norm fold to Rust);
* drops **only** ``…ssl_model.mask_emb`` (present but unused at inference:
  upstream calls the SSL model with ``mask=False``), and says so on stdout;
* derives the config side-car from the *tensor shapes themselves* plus the
  two inference constants upstream's ``score.py`` pins (``domains = 0``,
  ``judge_id = 288``) — nothing about the architecture is invented here;
* prints a sha256 manifest line per output.

Anything unexpected — a missing key, a shape that disagrees with the rest of
the checkpoint, an unknown top-level layout — is a **loud failure**, never a
silently patched default (FR-EX-08).

# Usage

::

    ~/.cache/vokra-eval/venv-utmos-e/bin/python \\
        tools/parity/utmos_prepare_checkpoint.py \\
        --ckpt ~/.cache/vokra-eval/out/utmos-probe/ckpt/epoch=3-step=7459.ckpt \\
        --output /tmp/utmos22-strong.safetensors \\
        --config-out /tmp/utmos22-strong-config.json

Then::

    vokra-cli convert --model utmos --input /tmp/utmos22-strong.safetensors \\
        --config /tmp/utmos22-strong-config.json --output /tmp/utmos.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
import types

import torch

# Upstream inference constants, quoted from the HF space's score.py:
#     'domains':  torch.zeros(bs, dtype=torch.int)
#     'judge_id': torch.ones(bs, dtype=torch.int) * 288
# and the final affine `output.mean(dim=1).squeeze(1) * 2 + 3`.
DOMAIN_ID = 0
JUDGE_ID = 288
HEAD_SCALE = 2.0
HEAD_OFFSET = 3.0
SAMPLE_RATE = 16000

SSL = "feature_extractors.0.ssl_model"
DOMAIN_EMB = "feature_extractors.1.embedding.weight"
LD = "output_layers.0"
PROJ = "output_layers.1.net"
# Present in the checkpoint but unused at inference (mask=False).
DROP = {f"{SSL}.mask_emb"}


def die(msg: str) -> "None":
    print(f"utmos_prepare_checkpoint: {msg}", file=sys.stderr)
    raise SystemExit(2)


class _TolerantUnpickler(pickle.Unpickler):
    """Stubs classes this venv cannot import (omegaconf/hydra hparams).

    Only the ``state_dict`` is consumed downstream; the stubs never reach it
    (a stub landing among the tensors is caught by the ``is Tensor`` check).
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type(
                name,
                (),
                {
                    "__module__": f"STUB:{module}",
                    "__init__": lambda self, *a, **k: None,
                    "__setstate__": lambda self, state: None,
                },
            )


def load_state_dict(path: str) -> "dict[str, torch.Tensor]":
    """Loads the ckpt's ``state_dict``, preferring torch's safe loader."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        tolerant = types.ModuleType("tolerant_pickle")
        tolerant.Unpickler = _TolerantUnpickler
        tolerant.load = lambda f, **k: _TolerantUnpickler(f, **k).load()
        tolerant.loads = lambda b, **k: _TolerantUnpickler(io.BytesIO(b), **k).load()
        tolerant.dump, tolerant.dumps = pickle.dump, pickle.dumps
        obj = torch.load(
            path, map_location="cpu", pickle_module=tolerant, weights_only=False
        )
    if not isinstance(obj, dict):
        die(f"checkpoint root is {type(obj).__name__}, expected a dict")
    sd = obj.get("state_dict", obj)
    if not isinstance(sd, dict) or not sd:
        die("checkpoint has no non-empty `state_dict`")
    return sd


def need(sd, key):
    if key not in sd:
        die(f"checkpoint is missing the required tensor `{key}`")
    return sd[key]


def shape(sd, key):
    return list(need(sd, key).shape)


def derive_config(sd: "dict[str, torch.Tensor]") -> dict:
    """Reads the architecture off the tensor shapes (nothing invented)."""
    # --- conv feature encoder -------------------------------------------
    conv_channels, conv_kernels, gn_layers, gn_groups = [], [], [], []
    i = 0
    while f"{SSL}.feature_extractor.conv_layers.{i}.0.weight" in sd:
        c_out, _c_in, k = shape(sd, f"{SSL}.feature_extractor.conv_layers.{i}.0.weight")
        conv_channels.append(c_out)
        conv_kernels.append(k)
        # Sequential index 2 is the norm slot (0=conv, 1=Dropout, 2=norm,
        # 3=GELU); its presence is what marks a group-normed layer.
        if f"{SSL}.feature_extractor.conv_layers.{i}.2.weight" in sd:
            gn = shape(sd, f"{SSL}.feature_extractor.conv_layers.{i}.2.weight")
            if gn != [c_out]:
                die(f"conv layer {i} norm affine {gn} != [{c_out}]")
            gn_layers.append(i)
            # fairseq's Fp32GroupNorm(dim, dim) = one group per channel.
            gn_groups.append(c_out)
        i += 1
    if not conv_channels:
        die("no `feature_extractor.conv_layers.*` found — not a wav2vec2 checkpoint?")

    # Strides are NOT recoverable from shapes; they come from the checkpoint's
    # own conv_feature_layers arg, which the probe recorded verbatim. Refuse
    # to guess (a wrong stride silently changes the frame rate).
    strides_by_kernel = {10: 5, 3: 2, 2: 2}
    conv_strides = []
    for k in conv_kernels:
        if k not in strides_by_kernel:
            die(
                f"conv kernel {k} has no pinned stride — upstream's "
                f"conv_feature_layers is '[(512,10,5)] + [(512,3,2)]*4 + "
                f"[(512,2,2)]*2'; extend the table deliberately rather than "
                f"guessing"
            )
        conv_strides.append(strides_by_kernel[k])

    # --- projection / encoder --------------------------------------------
    d, c_last = shape(sd, f"{SSL}.post_extract_proj.weight")
    if c_last != conv_channels[-1]:
        die(f"post_extract_proj in-dim {c_last} != last conv channel {conv_channels[-1]}")
    pos_v = shape(sd, f"{SSL}.encoder.pos_conv.0.weight_v")  # [d, d/groups, k]
    if pos_v[0] != d:
        die(f"pos_conv weight_v out-channels {pos_v[0]} != hidden dim {d}")
    pos_kernel = pos_v[2]
    if d % pos_v[1] != 0:
        die(f"pos_conv in-channels-per-group {pos_v[1]} does not divide {d}")
    pos_groups = d // pos_v[1]

    n_layer = 0
    while f"{SSL}.encoder.layers.{n_layer}.fc1.weight" in sd:
        n_layer += 1
    if n_layer == 0:
        die("no encoder layers found")
    ffn_dim, _ = shape(sd, f"{SSL}.encoder.layers.0.fc1.weight")
    # Head count is not in the state dict; wav2vec2-base pins 12 and the
    # checkpoint's own args (wav2vec-small-args.txt: encoder_attention_heads
    # = 12, encoder_embed_dim = 768) agree. Derive as d / 64 (the fairseq
    # base head width) and cross-check divisibility rather than hard-coding.
    if d % 64 != 0:
        die(f"hidden dim {d} is not a multiple of the 64-wide fairseq head")
    n_head = d // 64

    # --- conditioning + BLSTM + head --------------------------------------
    n_domains, domain_dim = shape(sd, DOMAIN_EMB)
    n_judges, judge_dim = shape(sd, f"{LD}.judge_embedding.weight")
    g, blstm_in = shape(sd, f"{LD}.decoder_rnn.weight_ih_l0")
    if g % 4 != 0:
        die(f"BLSTM gate block {g} is not a multiple of 4")
    blstm_hidden = g // 4
    if blstm_in != d + domain_dim + judge_dim:
        die(
            f"BLSTM input {blstm_in} != hidden {d} + domain {domain_dim} + "
            f"judge {judge_dim}"
        )
    h0_out, h0_in = shape(sd, f"{PROJ}.0.weight")
    if h0_in != 2 * blstm_hidden:
        die(f"head linear 0 in-dim {h0_in} != 2 × BLSTM hidden {blstm_hidden}")
    h1_out, h1_in = shape(sd, f"{PROJ}.3.weight")
    if h1_in != h0_out or h1_out != 1:
        die(f"head linear 1 is [{h1_out}, {h1_in}], expected [1, {h0_out}]")
    if DOMAIN_ID >= n_domains or JUDGE_ID >= n_judges:
        die(
            f"pinned conditioning ids out of range: domain {DOMAIN_ID}/{n_domains}, "
            f"judge {JUDGE_ID}/{n_judges}"
        )

    return {
        "arch_variant": "wav2vec2_regression.v1",
        "sample_rate": SAMPLE_RATE,
        "conv_channels": conv_channels,
        "conv_kernels": conv_kernels,
        "conv_strides": conv_strides,
        "conv_activation": "gelu",
        "conv_group_norm_layers": gn_layers,
        "conv_group_norm_groups": gn_groups,
        # torch.nn.GroupNorm / LayerNorm defaults; fairseq overrides neither.
        "group_norm_eps": 1e-5,
        "ln_eps": 1e-5,
        "n_layer": n_layer,
        "n_head": n_head,
        "hidden_dim": d,
        "ffn_dim": ffn_dim,
        "norm": "post",
        "pos_conv_kernel": pos_kernel,
        "pos_conv_groups": pos_groups,
        "domain_dim": domain_dim,
        "domain_id": DOMAIN_ID,
        "judge_dim": judge_dim,
        "judge_id": JUDGE_ID,
        "blstm_hidden": blstm_hidden,
        "head_dims": [h0_out, 1],
        "head_pool": "mean_after",
        "head_activation": "relu",
        "head_scale": HEAD_SCALE,
        "head_offset": HEAD_OFFSET,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="UTMOS22-strong .ckpt")
    ap.add_argument("--output", required=True, help="flat safetensors out")
    ap.add_argument("--config-out", required=True, help="config JSON side-car out")
    args = ap.parse_args()

    sd = load_state_dict(args.ckpt)
    tensors, dropped, non_tensor = {}, [], []
    for k, v in sd.items():
        if k in DROP:
            dropped.append(k)
            continue
        if not isinstance(v, torch.Tensor):
            non_tensor.append(k)
            continue
        tensors[k] = v.detach().to(torch.float32).contiguous()
    if non_tensor:
        die(f"state_dict has non-tensor entries: {non_tensor[:5]}")
    if not tensors:
        die("no tensors survived filtering")

    config = derive_config(tensors)

    try:
        from safetensors.torch import save_file
    except ImportError:
        die("`safetensors` is not installed in this interpreter (pip install safetensors)")
    save_file(tensors, args.output)
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    total = sum(t.numel() for t in tensors.values())
    print(f"tensors written : {len(tensors)} ({total:,} params)")
    print(f"dropped         : {dropped if dropped else '(none)'}")
    for path in (args.output, args.config_out):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"sha256 {h}  {path}")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
