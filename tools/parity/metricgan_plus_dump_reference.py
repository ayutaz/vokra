#!/usr/bin/env python3
"""Dump an independent SpeechBrain MetricGAN+ reference fixture.

The oracle is the pinned ``speechbrain==1.0.3`` package loading the official
``speechbrain/metricgan-plus-voicebank`` checkpoint at an immutable Hugging
Face revision.  This file deliberately defines no LSTM, dense layer, mask, or
resynthesis mirror.  Instead it runs
``SpectralMaskEnhancement.enhance_batch`` and hooks the upstream modules:

* the generator pre-hook captures its official log-magnitude input;
* ``enhance_model.blstm`` captures the official bidirectional LSTM output;
* the ``linear2`` pre-hook captures the post-LeakyReLU ``linear1`` values;
* the generator hook captures the learned-sigmoid mask; and
* ``enhance_batch`` supplies the official phase-resynthesized waveform.

If the real SpeechBrain model cannot be imported or loaded, the script aborts.
There is no local reimplementation fallback (NFR-QL-04).

Run only through the parity tree's uv environment, and generate the real
fixture on VAST as part of the real-checkpoint validation lifecycle::

    uv run --project tools/parity --python 3.12 python \
      tools/parity/metricgan_plus_dump_reference.py \
      --output-dir /root/metricgan-reference \
      --savedir /root/metricgan-upstream
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import huggingface_hub
import numpy as np
import torch
import torchaudio
from huggingface_hub.errors import RemoteEntryNotFoundError
from requests.exceptions import HTTPError

# SpeechBrain 1.0.3 probes APIs removed by newer transport dependencies.  The
# model path below consumes an in-memory tensor, so the audio-backend probe is
# irrelevant.  The HF shim only translates the retired keyword and optional
# custom.py 404 into the exception type SpeechBrain 1.0.3 already handles.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except RemoteEntryNotFoundError as error:
        raise HTTPError(f"404 Client Error: {error}") from error


huggingface_hub.hf_hub_download = _hf_hub_download_compat

try:
    import speechbrain  # noqa: E402
    from speechbrain.inference.enhancement import (  # noqa: E402
        SpectralMaskEnhancement,
    )
except Exception as error:  # noqa: BLE001 - loud independent-oracle failure
    raise SystemExit(
        "metricgan_plus_dump_reference: could not import the real SpeechBrain "
        f"implementation ({type(error).__name__}: {error}); a mirror fallback "
        "is forbidden"
    ) from error


DEFAULT_MODEL = "speechbrain/metricgan-plus-voicebank"
DEFAULT_REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def deterministic_pcm() -> np.ndarray:
    """A fixed finite multitone with an onset ramp and no random draw."""

    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.20 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.11 * np.sin(2.0 * math.pi * 421.0 * time + 0.3)
        + 0.04 * np.cos(2.0 * math.pi * 997.0 * time)
    )
    signal *= np.minimum(1.0, index / 160.0)
    return signal.astype(np.float32)


def tensor_output(name: str, value: object) -> torch.Tensor:
    """Extract a tensor from one official hook output, or abort loudly."""

    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"official MetricGAN+ stage {name} returned {type(value)!r}, not a tensor"
        )
    return value.detach().cpu().contiguous()


def write_f32(path: Path, tensor: torch.Tensor | np.ndarray) -> list[int]:
    values = np.asarray(
        tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor,
        dtype="<f4",
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path.name}: official output contains non-finite values")
    path.write_bytes(values.tobytes(order="C"))
    return list(values.shape)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def locate_checkpoint(savedir: Path) -> Path:
    checkpoints = sorted(savedir.rglob("enhance_model.ckpt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            "expected exactly one official enhance_model.ckpt below "
            f"{savedir}, found {[str(path) for path in checkpoints]}"
        )
    return checkpoints[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--savedir", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-id",
        help="pinned upstream org/repo identity when --source is a local directory",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="optional fail-closed pin for a repeat generation",
    )
    args = parser.parse_args()

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    source_path = Path(args.source)
    revision = None if source_path.exists() else args.revision
    model_id = args.model_id or (DEFAULT_MODEL if source_path.exists() else args.source)
    try:
        inference = SpectralMaskEnhancement.from_hparams(
            source=args.source,
            revision=revision,
            savedir=args.savedir,
            run_opts={"device": "cpu"},
        )
    except Exception as error:  # noqa: BLE001 - preserve upstream failure detail
        raise SystemExit(
            "metricgan_plus_dump_reference: the real pinned SpeechBrain model "
            f"could not be loaded ({type(error).__name__}: {error})"
        ) from error

    generator = inference.mods.enhance_model
    generator.eval()
    traced: dict[str, torch.Tensor] = {}

    def capture_output(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            traced[name] = tensor_output(name, output)

        return hook

    def capture_input(name: str):
        def hook(
            _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> None:
            if not inputs:
                raise RuntimeError(f"official MetricGAN+ stage {name} had no input")
            traced[name] = tensor_output(name, inputs[0])

        return hook

    handles = [
        generator.register_forward_pre_hook(capture_input("features")),
        generator.blstm.register_forward_hook(capture_output("bilstm")),
        generator.linear2.register_forward_pre_hook(capture_input("linear1")),
        generator.register_forward_hook(capture_output("mask")),
    ]

    pcm = deterministic_pcm()
    noisy = torch.from_numpy(pcm).unsqueeze(0)
    with torch.no_grad():
        waveform = inference.enhance_batch(noisy, lengths=torch.ones(1))
    for handle in handles:
        handle.remove()

    required = {"features", "bilstm", "linear1", "mask"}
    if set(traced) != required:
        raise RuntimeError(
            f"official hook set mismatch: got {sorted(traced)}, expected {sorted(required)}"
        )
    waveform = tensor_output("waveform", waveform)
    if tuple(waveform.shape) != (1, PCM_SAMPLES):
        raise RuntimeError(f"unexpected official waveform shape {tuple(waveform.shape)}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    shapes = {
        "pcm": write_f32(output / "pcm.f32.bin", pcm),
        "features": write_f32(output / "features.f32.bin", traced["features"]),
        "bilstm": write_f32(output / "bilstm.f32.bin", traced["bilstm"]),
        "linear1": write_f32(output / "linear1.f32.bin", traced["linear1"]),
        "mask": write_f32(output / "mask.f32.bin", traced["mask"]),
        "waveform": write_f32(output / "waveform.f32.bin", waveform),
    }

    checkpoint = locate_checkpoint(args.savedir)
    checkpoint_sha256 = sha256_file(checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha256 != args.expected_checkpoint_sha256
    ):
        raise RuntimeError(
            f"checkpoint sha256 {checkpoint_sha256} != expected "
            f"{args.expected_checkpoint_sha256}"
        )

    fixture_files = sorted(output.glob("*.f32.bin"))
    fixture_hashes = {path.name: sha256_file(path) for path in fixture_files}
    manifest = {
        "format": "vokra-metricgan-plus-reference-v1",
        "oracle": "speechbrain.inference.enhancement.SpectralMaskEnhancement",
        "hooks": {
            "features": "enhance_model forward pre-hook",
            "bilstm": "enhance_model.blstm forward hook",
            "linear1": "enhance_model.linear2 forward pre-hook (post-LeakyReLU)",
            "mask": "enhance_model forward hook",
            "waveform": "SpectralMaskEnhancement.enhance_batch return",
        },
        "model_id": model_id,
        "revision": args.revision,
        "source": args.source,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_rate": SAMPLE_RATE,
        "shapes": shapes,
        "fixture_sha256": fixture_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
        "cpu": platform.processor(),
        "machine": platform.machine(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{fixture_hashes[path.name]}  {path.name}" for path in fixture_files
    ]
    checksums.append(f"{sha256_file(output / 'manifest.json')}  manifest.json")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
