#!/usr/bin/env python3
"""Dump an independent official FRCRN-SE-16K waveform reference.

The oracle imports ``DCCRN`` from an exact clean ClearerVoice-Studio checkout,
verifies every relevant source file, safe-loads the pinned official checkpoint,
strict-loads the real module, hooks its actual STFT/U-Net/FSMN/decoder/iSTFT
stages, and calls the upstream ``DCCRN.forward``. It never imports Vokra and
defines no mirror network.

Model execution is VAST-only through ``tools/parity/pyproject.toml``. The
``--self-test`` path is stdlib-only and checks deterministic PCM generation.
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
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SOURCE_REPOSITORY = "https://github.com/modelscope/ClearerVoice-Studio"
SOURCE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
UPSTREAM_HF = "alibabasglab/FRCRN_SE_16K"
UPSTREAM_REVISION = "3766e6a64b0d8cb58f08d913d617bf129f11ed53"
CHECKPOINT_BYTES = 161_053_751
CHECKPOINT_SHA256 = "b22256adbb91b68cf5a3db8f6657a4fb17066eecd5f069803e59c186c1cf3ebb"
TENSOR_COUNT = 812
PARAMETER_COUNT = 14_387_164
TENSOR_MANIFEST_SHA256 = "ca71dad1ae5293d3d63628b71127c0efdf004cec684e5a341ab376ce3e2851b7"
SAMPLE_RATE = 16_000
DEFAULT_SAMPLES = 16_000
SOURCE_FILES = {
    Path("clearvoice/clearvoice/models/frcrn_se/frcrn.py"): (
        10_685,
        "17f83883f0a3ce2dc5498cbc58e65aedafeba37ec2da221fec01f773e67b4603",
    ),
    Path("clearvoice/clearvoice/models/frcrn_se/unet.py"): (
        18_933,
        "45bbfe65da07f49a529b4ca23b7e01bceb477efce7aa07c6fa8968ebccb9431e",
    ),
    Path("clearvoice/clearvoice/models/frcrn_se/complex_nn.py"): (
        17_953,
        "3dcda8502c6d588493a59dcb0910624a088be3e1c8b82b9d4b9408e1c5f3b5cb",
    ),
    Path("clearvoice/clearvoice/models/frcrn_se/conv_stft.py"): (
        12_734,
        "683ead65ab66688dfcd7a595d7235cda391a1428e7121d7ed06adca4b2494953",
    ),
    Path("clearvoice/clearvoice/models/frcrn_se/se_layer.py"): (
        3_315,
        "265384bdcf473b4f96e8fe166ff76b52a8fd12cc809da7da38c76a18862c68c0",
    ),
    Path("clearvoice/clearvoice/config/inference/FRCRN_SE_16K.yaml"): (
        614,
        "1aef17c64948a539bb7845f09f449744a79e1488be4d7dc7f616461bdabcfd3c",
    ),
    Path("LICENSE"): (
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def validate_source(checkout: Path) -> Path:
    checkout = checkout.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError(f"source revision {revision!r} != pinned {SOURCE_REVISION!r}")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official source checkout must be exactly clean")
    for relative, (expected_bytes, expected_sha256) in SOURCE_FILES.items():
        path = checkout / relative
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"{relative}: unexpected/missing byte length")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{relative}: unexpected SHA-256")
    return checkout


def validate_checkpoint(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint byte length != pinned {CHECKPOINT_BYTES}")
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 {digest} != pinned {CHECKPOINT_SHA256}")


def deterministic_pcm(samples: int) -> tuple[float, ...]:
    if samples < 640:
        raise ValueError("FRCRN reference requires at least one 640-sample frame")
    values = []
    for index in range(samples):
        time = index / SAMPLE_RATE
        onset = min(1.0, index / 320.0)
        chirp = math.sin(2.0 * math.pi * (91.0 * time + 23.0 * time * time))
        value = onset * (
            0.16 * math.sin(2.0 * math.pi * 173.0 * time + 0.1)
            + 0.08 * math.cos(2.0 * math.pi * 421.0 * time + 0.3)
            + 0.025 * chirp
        )
        values.append(value)
    return tuple(values)


def manifest_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack("<Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", int(dimension)))
    return digest.hexdigest()


def unwrap_checkpoint(obj: object, checkpoint: Path) -> OrderedDict[str, Any]:
    import torch

    if not isinstance(obj, dict):
        raise ValueError(f"{checkpoint}: checkpoint top level is not a dict")
    candidate: object = obj.get("model", obj)
    if not isinstance(candidate, dict) or not candidate:
        raise ValueError(f"{checkpoint}: official model state is absent or empty")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise ValueError(f"{checkpoint}: official model state contains non-tensors")
    state = OrderedDict(candidate)
    prefixed = [name.startswith("module.") for name in state]
    if any(prefixed):
        if not all(prefixed):
            raise ValueError(f"{checkpoint}: mixed module-prefix state is refused")
        state = OrderedDict((name[7:], value) for name, value in state.items())
    return state


def validate_inference_manifest(state: OrderedDict[str, Any]) -> None:
    import torch

    inference = {
        name: value
        for name, value in state.items()
        if not name.endswith(".num_batches_tracked")
    }
    counters = {
        name: value
        for name, value in state.items()
        if name.endswith(".num_batches_tracked")
    }
    if len(inference) != TENSOR_COUNT:
        raise ValueError(f"inference tensor count {len(inference)} != {TENSOR_COUNT}")
    if any(value.dtype != torch.float32 for value in inference.values()):
        raise ValueError("official FRCRN inference manifest contains a non-F32 tensor")
    if any(value.dtype not in (torch.int32, torch.int64) for value in counters.values()):
        raise ValueError("official FRCRN BatchNorm counter has non-integer dtype")
    parameters = sum(int(value.numel()) for value in inference.values())
    if parameters != PARAMETER_COUNT:
        raise ValueError(f"inference value count {parameters} != {PARAMETER_COUNT}")
    digest = manifest_sha256(inference)
    if digest != TENSOR_MANIFEST_SHA256:
        raise ValueError(
            f"inference manifest SHA-256 {digest} != {TENSOR_MANIFEST_SHA256}"
        )


def tensor_output(name: str, value: object) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"official stage {name} returned {type(value)!r}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"official stage {name} contains non-finite values")
    return value.detach().cpu().contiguous()


def write_f32(path: Path, values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{path.name}: official output contains non-finite values")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": str(path.name),
        "dtype": "float32-le",
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def register_stage_hooks(model: Any, captures: dict[str, Any]) -> list[Any]:
    handles = []

    def hook(name: str):
        def capture(_module: object, _inputs: object, value: object) -> None:
            captures[name] = tensor_output(name, value)

        return capture

    handles.append(model.stft.register_forward_hook(hook("stft")))
    handles.append(model.istft.register_forward_hook(hook("istft")))
    for root_name, unet in (("unet", model.unet), ("unet2", model.unet2)):
        handles.append(unet.register_forward_hook(hook(f"{root_name}.output")))
        handles.append(unet.fsmn.register_forward_hook(hook(f"{root_name}.fsmn")))
        handles.append(unet.linear.register_forward_hook(hook(f"{root_name}.linear")))
        for index, module in enumerate(unet.encoders):
            handles.append(
                module.register_forward_hook(hook(f"{root_name}.encoder{index}"))
            )
        for index, module in enumerate(unet.decoders):
            handles.append(
                module.register_forward_hook(hook(f"{root_name}.decoder{index}"))
            )
        for index in range(1, 7):
            handles.append(
                unet.fsmn_enc[index].register_forward_hook(
                    hook(f"{root_name}.fsmn_enc{index}")
                )
            )
        for index in range(6):
            handles.append(
                unet.fsmn_dec[index].register_forward_hook(
                    hook(f"{root_name}.fsmn_dec{index}")
                )
            )
    return handles


def expected_capture_names() -> set[str]:
    names = {"stft", "istft"}
    for root in ("unet", "unet2"):
        names.update({f"{root}.output", f"{root}.fsmn", f"{root}.linear"})
        names.update(f"{root}.encoder{index}" for index in range(7))
        names.update(f"{root}.decoder{index}" for index in range(7))
        names.update(f"{root}.fsmn_enc{index}" for index in range(1, 7))
        names.update(f"{root}.fsmn_dec{index}" for index in range(6))
    return names


def dump(source: Path, checkpoint: Path, output: Path, samples: int) -> None:
    import numpy as np
    import torch

    source = validate_source(source)
    validate_checkpoint(checkpoint)
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")

    sys.dont_write_bytecode = True
    package_root = source / "clearvoice"
    sys.path.insert(0, str(package_root))
    try:
        from clearvoice.models.frcrn_se.frcrn import FRCRN_SE_16K

        imported = Path(sys.modules[FRCRN_SE_16K.__module__].__file__).resolve()
    except Exception as error:  # noqa: BLE001 - loud independent-oracle boundary
        raise RuntimeError(
            "could not import pinned ClearerVoice FRCRN; mirror fallback forbidden"
        ) from error
    finally:
        sys.path.pop(0)
    expected_import = (
        source / "clearvoice/clearvoice/models/frcrn_se/frcrn.py"
    ).resolve()
    if imported != expected_import:
        raise ValueError(f"imported FRCRN from {imported}, expected {expected_import}")

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args = SimpleNamespace(win_len=640, win_inc=320, fft_len=640, win_type="hanning")
    model = FRCRN_SE_16K(args).model.cpu().eval()
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = unwrap_checkpoint(obj, checkpoint)
    validate_inference_manifest(state)
    model.load_state_dict(state, strict=True)

    captures: dict[str, Any] = {}
    handles = register_stage_hooks(model, captures)
    pcm = np.asarray(deterministic_pcm(samples), dtype=np.float32)
    with torch.inference_mode():
        result = model(torch.from_numpy(pcm.copy()).reshape(1, -1))
    for handle in handles:
        handle.remove()
    if not isinstance(result, list) or len(result) != 3:
        raise RuntimeError("official DCCRN.forward returned an unexpected result")
    waveform = tensor_output("waveform", result[1])
    if set(captures) != expected_capture_names():
        raise RuntimeError(
            f"official hooks yielded {sorted(captures)}, expected {sorted(expected_capture_names())}"
        )

    output.mkdir(parents=True)
    taps = output / "taps"
    taps.mkdir()
    files = {
        "pcm": write_f32(output / "pcm.f32le", pcm),
        "waveform": write_f32(output / "waveform.f32le", waveform.numpy()),
    }
    for name in sorted(captures):
        files[name] = write_f32(taps / f"{name}.f32le", captures[name].numpy())

    manifest = {
        "format": "vokra.frcrn.official-parity.v1",
        "oracle": "ClearerVoice-Studio DCCRN.forward from pinned clean source",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_files": {
            str(path): {"bytes": values[0], "sha256": values[1]}
            for path, values in SOURCE_FILES.items()
        },
        "source_license": "Apache-2.0",
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "tensor_count": TENSOR_COUNT,
        "parameter_count": PARAMETER_COUNT,
        "tensor_manifest_sha256": TENSOR_MANIFEST_SHA256,
        "sample_rate": SAMPLE_RATE,
        "samples": samples,
        "official_import": str(imported),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test and not all((args.source, args.checkpoint, args.output)):
        parser.error("--source, --checkpoint and --output are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        if args.source is not None:
            validate_source(args.source)
        pcm = deterministic_pcm(DEFAULT_SAMPLES)
        payload = struct.pack(f"<{len(pcm)}f", *pcm)
        if len(payload) != DEFAULT_SAMPLES * 4 or pcm != deterministic_pcm(DEFAULT_SAMPLES):
            raise AssertionError("deterministic PCM self-test failed")
        print(
            "frcrn_dump_reference self-test: "
            f"sha256={hashlib.sha256(payload).hexdigest()}"
        )
        return 0
    dump(args.source, args.checkpoint, args.output, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
