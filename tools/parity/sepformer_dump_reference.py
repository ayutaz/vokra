#!/usr/bin/env python3
"""Dump an independent SpeechBrain SepFormer waveform reference.

The oracle is the pinned ``speechbrain==1.0.3`` implementation and official
three-part checkpoint.  It never reads a Vokra GGUF and has no local layer
mirror fallback.  The output shape is read from the official model so the same
dumper covers the released one-, two-, and three-stream SepFormer family.
"""

from __future__ import annotations

import argparse
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

# SpeechBrain 1.0.3 probes an API removed by newer torchaudio.  The model path
# below consumes an in-memory tensor and never asks torchaudio to decode audio.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]

# SpeechBrain 1.0.3 still passes the retired ``use_auth_token`` keyword to
# huggingface_hub.  Keep the parity tree's newer client usable without changing
# model code or checkpoint resolution: translate only that transport keyword
# to its current spelling before SpeechBrain imports the module.
_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args: object, **kwargs: object) -> str:
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    try:
        return _hf_hub_download(*args, **kwargs)
    except RemoteEntryNotFoundError as error:
        # SpeechBrain 1.0.3 recognizes the requests-era 404 type and turns it
        # into ValueError so its optional custom.py probe remains optional.
        raise HTTPError(f"404 Client Error: {error}") from error


huggingface_hub.hf_hub_download = _hf_hub_download_compat

import speechbrain  # noqa: E402
from speechbrain.inference.separation import SepformerSeparation  # noqa: E402


DEFAULT_MODEL = "speechbrain/sepformer-wham16k-enhancement"
DEFAULT_REVISION = "90b3c5c3ffe3e04387b566715ab5fff36ec7b9d9"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 4_096


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    signal = (
        0.20 * np.sin(2.0 * math.pi * 173.0 * time)
        + 0.11 * np.sin(2.0 * math.pi * 421.0 * time + 0.3)
        + 0.04 * np.cos(2.0 * math.pi * 997.0 * time)
    )
    signal *= np.minimum(1.0, index / 160.0)
    return signal.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-id",
        help="pinned upstream org/repo identity when --source is a local directory",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--savedir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="official model compute dtype; float64 is the high-precision oracle",
    )
    parser.add_argument(
        "--trace-stages",
        action="store_true",
        help="dump official mask-network stage tensors captured by forward hooks",
    )
    args = parser.parse_args()

    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    source_path = Path(args.source)
    revision = None if source_path.exists() else args.revision
    model_id = args.model_id or (DEFAULT_MODEL if source_path.exists() else args.source)
    model = SepformerSeparation.from_hparams(
        source=args.source,
        revision=revision,
        savedir=args.savedir,
        run_opts={"device": "cpu"},
    )
    for module in model.mods.values():
        module.eval()
        if args.dtype == "float64":
            module.double()

    traced: dict[str, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if args.trace_stages:
        masknet = model.mods.masknet

        def capture(name: str):
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        f"official SepFormer stage {name} returned {type(output)!r}"
                    )
                traced[name] = output.detach().cpu().contiguous()

            return hook

        modules = {
            "mask_norm": masknet.norm,
            "mask_input": masknet.conv1d,
            "dual_block_0": masknet.dual_mdl[0],
            "dual_block_1": masknet.dual_mdl[1],
            "prelu": masknet.prelu,
            "speaker_projection": masknet.conv2d,
            "output": masknet.output,
            "output_gate": masknet.output_gate,
            "end": masknet.end_conv1x1,
            "masknet": masknet,
        }
        for block_index, block in enumerate(masknet.dual_mdl):
            modules.update(
                {
                    f"dual_{block_index}_intra_transformer": block.intra_mdl,
                    f"dual_{block_index}_intra_norm": block.intra_norm,
                    f"dual_{block_index}_inter_transformer": block.inter_mdl,
                    f"dual_{block_index}_inter_norm": block.inter_norm,
                }
            )
        hooks = [module.register_forward_hook(capture(name)) for name, module in modules.items()]

    pcm = deterministic_pcm()
    mixture = torch.from_numpy(pcm).unsqueeze(0)
    if args.dtype == "float64":
        mixture = mixture.double()
    encoder = model.mods.encoder(mixture)
    separated = model.separate_batch(mixture)
    for hook in hooks:
        hook.remove()
    if separated.ndim != 3 or tuple(separated.shape[:2]) != (1, PCM_SAMPLES):
        raise SystemExit(f"unexpected separated shape {tuple(separated.shape)}")
    output_streams = int(separated.shape[2])
    if output_streams < 1:
        raise SystemExit("official SepFormer emitted zero output streams")
    model_sample_rate = int(model.hparams.sample_rate)
    if model_sample_rate <= 0:
        raise SystemExit(f"invalid official model sample rate {model_sample_rate}")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "encoder.f32.bin", encoder[0].cpu().numpy())
    write_f32(output / "separated.f32.bin", separated[0].cpu().numpy())
    if args.trace_stages:
        traced["gated"] = traced["output"] * traced["output_gate"]
        for name, values in traced.items():
            write_f32(output / f"stage-{name}.f32.bin", values.numpy())

    manifest = {
        "format": "vokra-sepformer-reference-v1",
        "compute_dtype": args.dtype,
        "model_id": model_id,
        "revision": args.revision,
        "source": args.source,
        "sample_rate": model_sample_rate,
        "pcm_generation_sample_rate": SAMPLE_RATE,
        "pcm_samples": PCM_SAMPLES,
        "output_streams": output_streams,
        "encoder_shape": list(encoder.shape),
        "separated_shape": list(separated.shape),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
    }
    if args.trace_stages:
        manifest["trace_stages"] = {
            name: list(values.shape) for name, values in sorted(traced.items())
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
