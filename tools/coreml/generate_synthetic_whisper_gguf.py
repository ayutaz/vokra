"""Generate a tiny deterministic GGUF for CoreML structural integration.

This fixture is not a numerical-parity oracle and must never be described as
one. It carries only the metadata and encoder tensors consumed by
``generate_whisper_encoder.py``; real-model parity uses a converted upstream
Whisper GGUF and Vokra's Rust CPU encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def generate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    rng = np.random.default_rng(1234)
    n_mels, n_ctx, width, heads, layers, ffn = 2, 2, 4, 2, 1, 8
    writer = GGUFWriter(output, "whisper")
    writer.add_string("vokra.model.arch", "whisper")
    writer.add_uint32("vokra.whisper.n_mels", n_mels)
    writer.add_uint32("vokra.whisper.n_audio_ctx", n_ctx)
    writer.add_uint32("vokra.whisper.n_audio_state", width)
    writer.add_uint32("vokra.whisper.n_audio_head", heads)
    writer.add_uint32("vokra.whisper.n_audio_layer", layers)
    writer.add_uint32("vokra.whisper.ffn_dim", ffn)

    def tensor(name: str, shape: tuple[int, ...], *, scale: float = 0.05) -> None:
        value = rng.normal(0.0, scale, size=shape).astype(np.float32)
        # GGUFWriter follows ggml's reversed dimension serialization, whereas
        # Vokra's reader contract records the logical HF shape verbatim. Pass
        # a reversed raw_shape so the resulting header matches a Vokra-convert
        # GGUF while retaining the logical row-major payload bytes.
        writer.add_tensor(name, value, raw_shape=tuple(reversed(shape)))

    tensor("model.encoder.conv1.weight", (width, n_mels, 3))
    tensor("model.encoder.conv1.bias", (width,))
    tensor("model.encoder.conv2.weight", (width, width, 3))
    tensor("model.encoder.conv2.bias", (width,))
    tensor("model.encoder.embed_positions.weight", (n_ctx, width))
    prefix = "model.encoder.layers.0"
    tensor(f"{prefix}.self_attn_layer_norm.weight", (width,), scale=0.01)
    tensor(f"{prefix}.self_attn_layer_norm.bias", (width,))
    tensor(f"{prefix}.self_attn.q_proj.weight", (width, width))
    tensor(f"{prefix}.self_attn.q_proj.bias", (width,))
    tensor(f"{prefix}.self_attn.k_proj.weight", (width, width))
    tensor(f"{prefix}.self_attn.v_proj.weight", (width, width))
    tensor(f"{prefix}.self_attn.v_proj.bias", (width,))
    tensor(f"{prefix}.self_attn.out_proj.weight", (width, width))
    tensor(f"{prefix}.self_attn.out_proj.bias", (width,))
    tensor(f"{prefix}.final_layer_norm.weight", (width,), scale=0.01)
    tensor(f"{prefix}.final_layer_norm.bias", (width,))
    tensor(f"{prefix}.fc1.weight", (ffn, width))
    tensor(f"{prefix}.fc1.bias", (ffn,))
    tensor(f"{prefix}.fc2.weight", (width, ffn))
    tensor(f"{prefix}.fc2.bias", (width,))
    tensor("model.encoder.layer_norm.weight", (width,), scale=0.01)
    tensor("model.encoder.layer_norm.bias", (width,))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
