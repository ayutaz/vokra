#!/usr/bin/env python3
"""Dump an independent Qwen3-TTS talker-layer reference fixture.

The oracle is the official ``Qwen3TTSTalkerDecoderLayer`` and
``Qwen3TTSTalkerRotaryEmbedding`` imported from a checkout of
QwenLM/Qwen3-TTS pinned to ``UPSTREAM_REVISION``.  No Vokra formula is
imported or mirrored here.  The resulting small fixture covers bias-free GQA,
per-head Q/K RMSNorm, causal attention, interleaved multimodal RoPE, SwiGLU,
and both residual additions.

Run with the repository's uv-managed Python 3.12 environment, for example:

    uv run --project tools/parity \
      tools/parity/qwen3_tts_block_dump_reference.py \
      --upstream /work/Qwen3-TTS \
      --output crates/vokra-models/tests/fixtures/qwen3_tts/talker_block.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

UPSTREAM_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"


def flat(tensor):
    return tensor.detach().cpu().float().contiguous().view(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise SystemExit(
            f"Qwen3-TTS checkout revision {revision} != pinned {UPSTREAM_REVISION}"
        )

    sys.path.insert(0, str(args.upstream))
    import torch
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerDecoderLayer,
        Qwen3TTSTalkerRotaryEmbedding,
    )

    torch.manual_seed(0x5157454E)
    cfg = Qwen3TTSTalkerConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1.0e-6,
        rope_theta=1_000_000.0,
        rope_scaling={
            "type": "default",
            "rope_type": "default",
            "interleaved": True,
            "mrope_section": [24, 20, 20],
        },
        attention_bias=False,
        attention_dropout=0.0,
        num_code_groups=16,
        text_hidden_size=24,
    )
    # PreTrainedModel normally resolves this private dispatch selector before
    # constructing its layers.  This standalone official-layer oracle must do
    # the same explicitly; eager is the reference implementation whose math is
    # exercised below (no flash/fused kernel tolerance is mixed into the dump).
    cfg._attn_implementation = "eager"
    layer = Qwen3TTSTalkerDecoderLayer(cfg, layer_idx=0).eval().float()
    rotary = Qwen3TTSTalkerRotaryEmbedding(cfg).eval().float()

    hidden = torch.linspace(-0.75, 0.85, steps=3 * cfg.hidden_size, dtype=torch.float32)
    hidden = hidden.reshape(1, 3, cfg.hidden_size)
    position_ids = torch.tensor(
        [
            [[0, 1, 2]],
            [[0, 2, 4]],
            [[0, 3, 6]],
        ],
        dtype=torch.long,
    )
    causal_mask = torch.full((1, 1, 3, 3), torch.finfo(torch.float32).min)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        position_embeddings = rotary(hidden, position_ids)
        output = layer(
            hidden,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            use_cache=False,
        )[0]

    state = layer.state_dict()
    wanted = {
        "input_layernorm": "input_layernorm.weight",
        "q_proj": "self_attn.q_proj.weight",
        "q_norm": "self_attn.q_norm.weight",
        "k_proj": "self_attn.k_proj.weight",
        "k_norm": "self_attn.k_norm.weight",
        "v_proj": "self_attn.v_proj.weight",
        "o_proj": "self_attn.o_proj.weight",
        "post_attention_layernorm": "post_attention_layernorm.weight",
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj": "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight",
    }
    payload = {
        "upstream_repo": "https://github.com/QwenLM/Qwen3-TTS",
        "upstream_revision": revision,
        "oracle": "Qwen3TTSTalkerDecoderLayer.forward",
        "torch_version": torch.__version__,
        "config": {
            "hidden_dim": cfg.hidden_size,
            "n_head": cfg.num_attention_heads,
            "n_head_kv": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "ffn_dim": cfg.intermediate_size,
            "rope_base": cfg.rope_theta,
            "rms_norm_eps": cfg.rms_norm_eps,
        },
        "positions": position_ids[:, 0, :].transpose(0, 1).tolist(),
        "hidden": flat(hidden),
        "weights": {key: flat(state[name]) for key, name in wanted.items()},
        "output": flat(output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
