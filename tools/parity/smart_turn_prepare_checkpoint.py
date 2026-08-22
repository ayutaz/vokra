#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "safetensors>=0.4",
#     "soundfile>=0.12",
#     "torch==2.5.1",
#     "transformers==4.48.2",
# ]
# ///
"""Validate and run the pinned official SmartTurn v2 checkpoint.

This offline Python 3.12 sidecar is the independent PyTorch/Transformers
reference for Vokra's native Rust forward. It transcribes the endpoint head
from Pipecat's BSD-2-Clause ``local_smart_turn_v2.py`` at commit
``c560a748b4213ca8db6f43a5d165d91aaa124a52`` and uses the released
``Wav2Vec2Model`` implementation rather than mirroring the Rust kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


REVISION = "3267e96b50db03fe030b9869eb35f849a5eea1fa"
REFERENCE_REVISION = "c560a748b4213ca8db6f43a5d165d91aaa124a52"
EXPECTED_SHA256 = {
    "model.safetensors": "0c4429a3f55d42d055e08903eb961f6ec4021c9e35d489007f3dc4981b6b028b",
    "config.json": "31aa20aebdee3f961077a9482f909efce4d46199aabd848def1c4d9456e2c716",
    "preprocessor_config.json": "617bd0950f8cc9ac4062e8c73a7be60305ca5790a243df55fa6f44fb671b55b1",
}
SAMPLE_RATE = 16_000
MAX_INPUT_SAMPLES = SAMPLE_RATE * 16


class Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    """Pipecat's pinned SmartTurn v2 endpoint class, inference branch only."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_values, attention_mask=None):
        hidden_states = self.wav2vec2(
            input_values, attention_mask=attention_mask
        )[0]
        if attention_mask is None:
            raise ValueError("attention_mask must be provided")
        input_length = attention_mask.size(1)
        hidden_length = hidden_states.size(1)
        ratio = input_length / hidden_length
        indices = (
            torch.arange(hidden_length, device=attention_mask.device) * ratio
        ).long()
        pool_mask = attention_mask[:, indices].bool()
        attention_weights = self.pool_attention(hidden_states)
        attention_weights = attention_weights + (
            (1.0 - pool_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)
        logits = self.classifier(pooled)
        probabilities = torch.sigmoid(logits)
        return probabilities, hidden_states, pool_mask, pooled


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_pcm(seconds: float) -> np.ndarray:
    count = int(round(seconds * SAMPLE_RATE))
    t = np.arange(count, dtype=np.float32) / np.float32(SAMPLE_RATE)
    envelope = np.minimum(1.0, np.arange(count, dtype=np.float32) / 800.0)
    envelope *= np.minimum(1.0, np.arange(count, 0, -1, dtype=np.float32) / 800.0)
    signal = (
        0.31 * np.sin(2.0 * np.pi * 173.0 * t)
        + 0.17 * np.sin(2.0 * np.pi * 613.0 * t + 0.37)
        + 0.07 * np.sin(2.0 * np.pi * (97.0 + 23.0 * t) * t)
    )
    return np.ascontiguousarray(signal * envelope, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-wav", required=True, type=Path)
    parser.add_argument("--output-ref", required=True, type=Path)
    parser.add_argument("--output-pcm", required=True, type=Path)
    parser.add_argument("--output-probability", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not (0.025 <= args.seconds <= 16.0):
        parser.error("--seconds must be in [0.025, 16.0]")
    for output in (
        args.output_wav,
        args.output_ref,
        args.output_pcm,
        args.output_probability,
    ):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")
    for name, expected in EXPECTED_SHA256.items():
        path = args.checkpoint_dir / name
        if not path.is_file():
            parser.error(f"missing checkpoint file: {path}")
        actual = sha256(path)
        if actual != expected:
            parser.error(f"{name} SHA-256 is {actual}, expected {expected}")

    pcm = deterministic_pcm(args.seconds)
    import soundfile as sf

    sf.write(args.output_wav, pcm, SAMPLE_RATE, subtype="FLOAT")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.checkpoint_dir, local_files_only=True
    )
    inputs = extractor(
        pcm,
        sampling_rate=SAMPLE_RATE,
        padding="max_length",
        truncation=True,
        max_length=MAX_INPUT_SAMPLES,
        return_attention_mask=True,
        return_tensors="pt",
    )
    model = Wav2Vec2ForEndpointing.from_pretrained(
        args.checkpoint_dir, local_files_only=True
    ).eval()
    # CPU is intentional: the committed reference must not depend on a GPU
    # kernel family or reduction order. VAST is used for memory safety, not
    # to change the reference backend.
    device = torch.device("cpu")
    model = model.to(device)
    valid_feature_frames = int(
        model._get_feat_extract_output_lengths(torch.tensor([pcm.size])).item()
    )
    with torch.no_grad():
        probability, hidden, pool_mask, pooled = model(
            inputs.input_values.to(device), inputs.attention_mask.to(device)
        )
        # Validate the Rust optimization independently: retain exactly the
        # queries selected by Pipecat's ratio mask while preserving the
        # Wav2Vec2 feature-level key mask.
        extracted = model.wav2vec2.feature_extractor(inputs.input_values).transpose(1, 2)
        feature_mask = model.wav2vec2._get_feature_vector_attention_mask(
            extracted.shape[1], inputs.attention_mask
        )
        projected, _ = model.wav2vec2.feature_projection(extracted)
        trimmed_frames = int(pool_mask.sum().item())
        trimmed_key_mask = feature_mask[:, :trimmed_frames]
        trimmed_hidden = projected[:, :trimmed_frames, :].clone()
        trimmed_hidden[~trimmed_key_mask.unsqueeze(-1).expand_as(trimmed_hidden)] = 0
        trimmed_hidden = trimmed_hidden + model.wav2vec2.encoder.pos_conv_embed(
            trimmed_hidden
        )
        trimmed_hidden = model.wav2vec2.encoder.layer_norm(trimmed_hidden)
        additive_mask = 1.0 - trimmed_key_mask[:, None, None, :].to(
            trimmed_hidden.dtype
        )
        additive_mask = additive_mask * torch.finfo(trimmed_hidden.dtype).min
        additive_mask = additive_mask.expand(1, 1, trimmed_frames, trimmed_frames)
        for layer in model.wav2vec2.encoder.layers:
            trimmed_hidden = layer(trimmed_hidden, attention_mask=additive_mask)[0]
        trimmed_scores = model.pool_attention(trimmed_hidden)
        trimmed_scores = F.softmax(trimmed_scores, dim=1)
        trimmed_pooled = torch.sum(trimmed_hidden * trimmed_scores, dim=1)
        trimmed_probability = torch.sigmoid(model.classifier(trimmed_pooled))
    probability_value = float(probability[0, 0].cpu())
    trimmed_probability_value = float(trimmed_probability[0, 0].cpu())
    optimization_delta = abs(probability_value - trimmed_probability_value)
    if optimization_delta > 1e-6:
        raise RuntimeError(
            "trimmed-query optimization diverged from the official padded forward: "
            f"delta={optimization_delta:.9e}"
        )
    pcm_bytes = pcm.astype("<f4", copy=False).tobytes()
    probability_bytes = np.asarray([probability_value], dtype="<f4").tobytes()
    args.output_pcm.write_bytes(pcm_bytes)
    args.output_probability.write_bytes(probability_bytes)
    reference = {
        "revision": REVISION,
        "reference_revision": REFERENCE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "input_samples": int(pcm.size),
        "padded_input_samples": MAX_INPUT_SAMPLES,
        "hidden_frames": int(hidden.shape[1]),
        "valid_feature_frames": valid_feature_frames,
        "pooled_frames": int(pool_mask.sum().item()),
        "ratio_index_dtype": str(
            (
                torch.arange(hidden.shape[1])
                * (inputs.attention_mask.shape[1] / hidden.shape[1])
            ).dtype
        ),
        "completion_probability": probability_value,
        "trimmed_completion_probability": trimmed_probability_value,
        "trimmed_optimization_abs_delta": optimization_delta,
        "pcm_f32_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
        "probability_f32_sha256": hashlib.sha256(probability_bytes).hexdigest(),
        "pooled_prefix": pooled[0, :16].float().cpu().tolist(),
    }
    args.output_ref.write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(reference, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
