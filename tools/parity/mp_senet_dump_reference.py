#!/usr/bin/env python3
"""Dump an independent official MP-SENet DNS reference.

The oracle imports ``MPSENet`` and ``mag_pha_stft`` from the exact clean
``JacobLinCool/MPSENet`` segment-wrapper commit, strictly loads the pinned
``JacobLinCool/MP-SENet-DNS`` safetensors/config pair, hooks the official
encoder/TS blocks/mask/phase modules, and calls the package's public waveform
entry point.  It never imports Vokra and defines no mirror model.

Run model execution only on VAST through ``tools/parity/pyproject.toml``.  The
``--self-test`` path is stdlib-only and checks only deterministic PCM setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/JacobLinCool/MPSENet"
SOURCE_REVISION = "958141ca51703c5b1e0c30362ab5b1c8b0e49957"
PUBLICATION_REVISION = "a65c76f340a0c8a885fbbf1893d5ec0ea009d718"
PUBLICATION_MODEL_SHA256 = (
    "63d0ddc067e87b5ebe556e60a89fa4384f5fba51fed37b6cb477abfaa19cb208"
)
SOURCE_MODEL_PATH = Path("MPSENet/model/mpsenet.py")
SOURCE_MODEL_BYTES = 11_002
SOURCE_MODEL_SHA256 = (
    "e629e2858836489a598f9b325aa3abfc2a2360c72fc676d45c458c17efcaa7e8"
)
SOURCE_TRANSFORMER_PATH = Path("MPSENet/model/transformer.py")
SOURCE_TRANSFORMER_BYTES = 1_612
SOURCE_TRANSFORMER_SHA256 = (
    "44fb17b9a604f861304fd72517bfea73508393ca0ef00b58aaab6083c012ef0b"
)
SOURCE_LICENSE_BYTES = 1_069
SOURCE_LICENSE_SHA256 = (
    "df6322ce3ca3c70a0845c4a384432a9af50e7d70886d316741e2f47b5ae01f34"
)

HF_REPOSITORY = "JacobLinCool/MP-SENet-DNS"
HF_REVISION = "8b78493f536df1aa53bd3bcbb2f620f705e8589c"
MODEL_BYTES = 9_081_872
MODEL_SHA256 = "74912046c8b352d78ca4056c9624d7256ac4d7eac45ce015822a7f2282749cdc"
CONFIG_BYTES = 248
CONFIG_SHA256 = "0c5973617000142390726f8dad98a5b6b1429b4ef1a94da25f3bc009f86a3365"
TENSOR_COUNT = 247
SAMPLE_RATE = 16_000
DEFAULT_SAMPLES = 4_096
EXPECTED_CONFIG = {
    "h": {
        "beta": 2.0,
        "compress_factor": 0.3,
        "dense_channel": 64,
        "hop_size": 100,
        "n_fft": 400,
        "num_tsconformers": 4,
        "sampling_rate": 16000,
        "segment_size": 32000,
        "win_size": 400,
    },
    "num_tsblocks": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned input: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes != pinned {expected_bytes}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest} != pinned {expected_sha256}")


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_source_checkout(checkout: Path) -> Path:
    checkout = checkout.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"MP-SENet source revision {revision!r} != pinned {SOURCE_REVISION!r}"
        )
    dirty = git_output(checkout, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("official MP-SENet source checkout must be exactly clean")
    validate_file(
        checkout / SOURCE_MODEL_PATH, SOURCE_MODEL_BYTES, SOURCE_MODEL_SHA256
    )
    validate_file(
        checkout / SOURCE_TRANSFORMER_PATH,
        SOURCE_TRANSFORMER_BYTES,
        SOURCE_TRANSFORMER_SHA256,
    )
    validate_file(
        checkout / "LICENSE", SOURCE_LICENSE_BYTES, SOURCE_LICENSE_SHA256
    )
    return checkout


def deterministic_pcm(samples: int) -> tuple[float, ...]:
    if samples < 400:
        raise ValueError("samples must be at least the 400-sample analysis window")
    values = []
    for index in range(samples):
        time = index / SAMPLE_RATE
        onset = min(1.0, index / 160.0)
        value = onset * (
            0.17 * math.sin(2.0 * math.pi * 173.0 * time + 0.1)
            + 0.09 * math.cos(2.0 * math.pi * 421.0 * time + 0.3)
            + 0.03 * math.sin(2.0 * math.pi * 997.0 * time)
        )
        values.append(value)
    return tuple(values)


def write_f32(path: Path, values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{path.name}: official output contains non-finite values")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "dtype": "float32-le",
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tensor_output(name: str, value: object) -> Any:
    import torch

    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"official MP-SENet stage {name} returned {type(value)!r}, not a tensor"
        )
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"official MP-SENet stage {name} is non-finite")
    return value.detach().cpu().contiguous()


def dump(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    source = validate_source_checkout(args.source)
    validate_file(args.weights, MODEL_BYTES, MODEL_SHA256)
    validate_file(args.config, CONFIG_BYTES, CONFIG_SHA256)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise ValueError(f"config content {config!r} != pinned {EXPECTED_CONFIG!r}")
    if args.output.exists():
        raise ValueError(f"output directory already exists: {args.output}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from MPSENet.model.mpsenet import MPSENet as OfficialMpSenet
        from MPSENet.model.mpsenet import mag_pha_stft

        imported_module = sys.modules[OfficialMpSenet.__module__]
    except Exception as error:  # noqa: BLE001 - loud independent-oracle failure
        raise RuntimeError(
            "could not import the real pinned MP-SENet package; a mirror fallback is forbidden"
        ) from error
    finally:
        sys.path.pop(0)
    imported = Path(imported_module.__file__).resolve()
    if imported != (source / SOURCE_MODEL_PATH).resolve():
        raise ValueError(f"imported MP-SENet from {imported}, expected source checkout")

    state = load_file(str(args.weights), device="cpu")
    if len(state) != TENSOR_COUNT:
        raise ValueError(f"checkpoint has {len(state)} tensors, expected {TENSOR_COUNT}")
    non_f32 = sorted(name for name, value in state.items() if value.dtype != torch.float32)
    if non_f32:
        raise ValueError(f"checkpoint contains non-F32 tensors: {non_f32[:8]!r}")

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = OfficialMpSenet(
        config["h"], num_tsblocks=config["num_tsblocks"]
    ).cpu().eval()
    model.load_state_dict(state, strict=True)

    traced: dict[str, Any] = {}

    def capture(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: object) -> None:
            traced[name] = tensor_output(name, output)

        return hook

    handles = [model.dense_encoder.register_forward_hook(capture("encoder"))]
    handles.extend(
        block.register_forward_hook(capture(f"ts_block_{index}"))
        for index, block in enumerate(model.TSTransformer)
    )
    handles.extend(
        [
            model.mask_decoder.register_forward_hook(capture("mask")),
            model.phase_decoder.register_forward_hook(capture("phase")),
        ]
    )

    pcm = np.asarray(deterministic_pcm(args.samples), dtype=np.float32)
    tensor = torch.from_numpy(pcm.copy())
    norm_factor = torch.sqrt(tensor.numel() / torch.sum(tensor**2.0))
    normalized = (tensor * norm_factor).unsqueeze(0)
    noisy_magnitude, noisy_phase, _ = mag_pha_stft(
        normalized,
        model.hann_window,
        config["h"]["n_fft"],
        config["h"]["hop_size"],
        config["h"]["win_size"],
        config["h"]["compress_factor"],
    )
    with torch.inference_mode():
        waveform, rate, labels = model(pcm)
    for handle in handles:
        handle.remove()

    expected_traces = {
        "encoder",
        "ts_block_0",
        "ts_block_1",
        "ts_block_2",
        "ts_block_3",
        "mask",
        "phase",
    }
    if set(traced) != expected_traces:
        raise RuntimeError(
            f"official hook set mismatch: got {sorted(traced)}, expected {sorted(expected_traces)}"
        )
    if rate != SAMPLE_RATE or labels != ["denoised_audio"]:
        raise RuntimeError(f"official public return metadata {(rate, labels)!r} is unexpected")
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    expected_samples = (args.samples // config["h"]["hop_size"]) * config["h"][
        "hop_size"
    ]
    if waveform.shape != (expected_samples,):
        raise RuntimeError(
            f"official waveform shape {waveform.shape} != {(expected_samples,)}"
        )

    args.output.mkdir(parents=True)
    files = {
        "pcm": write_f32(args.output / "pcm.f32le", pcm),
        "normalized_pcm": write_f32(
            args.output / "normalized_pcm.f32le", normalized.squeeze(0).numpy()
        ),
        # Official tensors are [B,F,T]; transpose to runtime [B,T,F].
        "noisy_magnitude": write_f32(
            args.output / "noisy_magnitude.f32le",
            noisy_magnitude.transpose(1, 2).contiguous().numpy(),
        ),
        "noisy_phase": write_f32(
            args.output / "noisy_phase.f32le",
            noisy_phase.transpose(1, 2).contiguous().numpy(),
        ),
        "encoder": write_f32(args.output / "encoder.f32le", traced["encoder"].numpy()),
        "mask": write_f32(
            args.output / "mask.f32le",
            traced["mask"].transpose(1, 2).contiguous().numpy(),
        ),
        "phase": write_f32(
            args.output / "phase.f32le",
            traced["phase"].transpose(1, 2).contiguous().numpy(),
        ),
        "waveform": write_f32(args.output / "waveform.f32le", waveform),
    }
    for index in range(4):
        key = f"ts_block_{index}"
        files[key] = write_f32(
            args.output / f"{key}.f32le", traced[key].numpy()
        )

    manifest = {
        "format": "vokra.mp-senet.official-parity.v1",
        "oracle": "MPSENet public waveform entry and hooked official submodules",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "publication_revision": PUBLICATION_REVISION,
        "publication_model_sha256": PUBLICATION_MODEL_SHA256,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_transformer_sha256": SOURCE_TRANSFORMER_SHA256,
        "source_license": "MIT",
        "source_license_sha256": SOURCE_LICENSE_SHA256,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "model_sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "official_import": str(imported),
        "attention_batch_first": False,
        "sample_rate": SAMPLE_RATE,
        "input_samples": args.samples,
        "output_samples": expected_samples,
        "norm_factor": float(norm_factor),
        "files": files,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "machine": platform.machine(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{details['sha256']}  {details['file']}"
        for details in files.values()
    ]
    checksums.append(f"{sha256_file(args.output / 'manifest.json')}  manifest.json")
    (args.output / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def self_test(source: Path | None) -> None:
    pcm = deterministic_pcm(DEFAULT_SAMPLES)
    if len(pcm) != DEFAULT_SAMPLES or not all(math.isfinite(value) for value in pcm):
        raise AssertionError("invalid deterministic PCM")
    payload = struct.pack(f"<{len(pcm)}f", *pcm)
    if len(payload) != DEFAULT_SAMPLES * 4:
        raise AssertionError("invalid PCM encoding")
    if source is not None:
        validate_source_checkout(source)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in ("source", "weights", "config", "output")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"model dump requires: {', '.join('--' + name for name in missing)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test(args.source)
    else:
        dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
