#!/usr/bin/env python3
"""Dump an independent official Audiobox Aesthetics reference.

The oracle imports ``AesMultiOutput`` and ``make_inference_batch`` directly
from a checkout of facebookresearch/audiobox-aesthetics at the pinned source
revision. It loads the pinned Hugging Face snapshot through the upstream
``PyTorchModelHubMixin`` implementation. There is deliberately no local WavLM
or projection-head mirror and no fallback if the official package cannot be
imported.

Forward hooks capture only values produced by official upstream modules:
waveform-stem output, feature LayerNorm, post projection, encoder input, all
12 block outputs, normalized per-axis embeddings and raw head outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

CHECKPOINT_REVISION = "9b1dd8e5df9af7216e836a98974fe3b82c56ded6"
SOURCE_REVISION = "2618e9d451b456e9328b39495b5e6234678aa550"
UPSTREAM_HF = "facebook/audiobox-aesthetics"
AXES = ["CE", "CU", "PC", "PQ"]
SAMPLE_RATE = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(source_tree: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot inspect official source checkout: {error}") from error
    return result.stdout.strip().lower()


def read_pcm16_mono(path: Path) -> "Any":
    import numpy as np

    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        frames = stream.getnframes()
        payload = stream.readframes(frames)
    if channels != 1 or width != 2 or rate != SAMPLE_RATE:
        raise SystemExit(
            "expected mono PCM16 16 kHz WAV, got "
            f"channels={channels}, width={width}, rate={rate}"
        )
    pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise SystemExit("input WAV must contain finite non-empty PCM")
    return pcm


def write_f32(path: Path, value: "Any") -> None:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    path.write_bytes(np.asarray(value, dtype="<f4").tobytes(order="C"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="local snapshot of facebook/audiobox-aesthetics at the pinned revision",
    )
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    revision = source_revision(args.source_tree)
    if revision != SOURCE_REVISION:
        raise SystemExit(
            f"official source checkout is {revision}, expected {SOURCE_REVISION}"
        )
    source_package = args.source_tree / "src"
    if not (source_package / "audiobox_aesthetics" / "model" / "aes.py").is_file():
        raise SystemExit(f"official package missing under {source_package}")
    for required in ["config.json", "model.safetensors"]:
        if not (args.checkpoint_dir / required).is_file():
            raise SystemExit(f"checkpoint snapshot is missing {required}")
    sys.path.insert(0, str(source_package))

    try:
        import numpy as np
        import torch
        from audiobox_aesthetics.infer import make_inference_batch
        from audiobox_aesthetics.model.aes import AesMultiOutput
    except Exception as error:  # noqa: BLE001 - loud independent-oracle failure
        raise SystemExit(
            "could not import the official Audiobox implementation "
            f"({type(error).__name__}: {error}); mirror fallback is forbidden"
        ) from error

    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        model = AesMultiOutput.from_pretrained(str(args.checkpoint_dir))
    except Exception as error:  # noqa: BLE001 - preserve official loader detail
        raise SystemExit(
            "official AesMultiOutput could not load the pinned local snapshot "
            f"({type(error).__name__}: {error})"
        ) from error
    model.cpu().eval()
    if list(model.axes_name) != AXES:
        raise SystemExit(f"official axes changed: {list(model.axes_name)!r}")
    if model.nth_layer != 13 or model.proj_num_layer != 5:
        raise SystemExit(
            f"official topology changed: nth_layer={model.nth_layer}, "
            f"proj_num_layer={model.proj_num_layer}"
        )

    captures: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def capture_tensor(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"official hook {name} returned {type(output).__name__}")
            captures[name] = output.detach().cpu().contiguous()

        return hook

    hooks.append(
        model.wavlm_model.feature_extractor.register_forward_hook(
            capture_tensor("stem_channel_major")
        )
    )
    hooks.append(
        model.wavlm_model.layer_norm.register_forward_hook(
            capture_tensor("feature_layer_norm")
        )
    )
    hooks.append(
        model.wavlm_model.post_extract_proj.register_forward_hook(
            capture_tensor("post_extract_projection")
        )
    )
    hooks.append(
        model.wavlm_model.encoder.layer_norm.register_forward_hook(
            capture_tensor("encoder_input")
        )
    )

    for index, layer in enumerate(model.wavlm_model.encoder.layers):
        def capture_layer(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            layer_index: int = index,
        ) -> None:
            if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
                raise RuntimeError(f"official layer {layer_index} returned an invalid value")
            captures[f"encoder_layer_{layer_index + 1:02d}"] = (
                output[0].transpose(0, 1).detach().cpu().contiguous()
            )

        hooks.append(layer.register_forward_hook(capture_layer))

    for axis in AXES:
        def capture_axis_embed(
            _module: Any,
            inputs: Any,
            *,
            axis_name: str = axis,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"official {axis_name} head received invalid input")
            captures[f"axis_embed_{axis_name}"] = (
                inputs[0].detach().cpu().contiguous()
            )

        hooks.append(model.proj_layer[axis].register_forward_pre_hook(capture_axis_embed))

    pcm = read_pcm16_mono(args.wav)
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)
    wavs, masks, weights, bids = make_inference_batch(
        [waveform], hop_size=10, window_size=10, sample_rate=SAMPLE_RATE
    )
    wav_batch = torch.stack(wavs)
    mask_batch = torch.stack(masks)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if set(bids) != {0}:
        raise SystemExit(f"unexpected official batch ids {bids!r}")
    with torch.inference_mode():
        predictions = model({"wav": wav_batch, "mask": mask_batch})
    for hook in hooks:
        hook.remove()

    raw_scores = torch.stack([predictions[axis] for axis in AXES], dim=-1).cpu()
    means = torch.tensor(
        [float(model.target_transform[axis]["mean"]) for axis in AXES]
    )
    stds = torch.tensor(
        [float(model.target_transform[axis]["std"]) for axis in AXES]
    )
    chunk_scores = raw_scores * stds + means
    final_scores = (chunk_scores * weight_tensor[:, None]).sum(0) / weight_tensor.sum()
    if not torch.isfinite(final_scores).all():
        raise SystemExit("official model produced non-finite scores")

    expected_capture_names = {
        "stem_channel_major",
        "feature_layer_norm",
        "post_extract_projection",
        "encoder_input",
        *(f"encoder_layer_{index:02d}" for index in range(1, 13)),
        *(f"axis_embed_{axis}" for axis in AXES),
    }
    missing = expected_capture_names - captures.keys()
    if missing:
        raise SystemExit(f"official hooks did not fire: {sorted(missing)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32le", pcm)
    write_f32(output / "chunk_weights.f32le", weight_tensor)
    write_f32(output / "raw_scores.f32le", raw_scores)
    write_f32(output / "chunk_scores.f32le", chunk_scores)
    write_f32(output / "final_scores.f32le", final_scores)
    for name, value in sorted(captures.items()):
        write_f32(output / f"{name}.f32le", value)

    checkpoint_hashes = {
        name: sha256(args.checkpoint_dir / name)
        for name in ["config.json", "model.safetensors"]
    }
    license_path = args.source_tree / "LICENSE"
    manifest = {
        "format": "vokra.audiobox-aesthetics.official-parity.v1",
        "upstream_hf": UPSTREAM_HF,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "source_revision": SOURCE_REVISION,
        "source_license_sha256": sha256(license_path),
        "checkpoint_sha256": checkpoint_hashes,
        "wav_sha256": sha256(args.wav),
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "axes": AXES,
        "target_means": means.tolist(),
        "target_stds": stds.tolist(),
        "chunk_count": len(wavs),
        "chunk_weights": [float(value) for value in weights],
        "captures": {
            name: list(value.shape) for name, value in sorted(captures.items())
        },
        "raw_score_shape": list(raw_scores.shape),
        "final_scores": [float(value) for value in final_scores],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
