#!/usr/bin/env python3
"""Dump an independent official Facebook Denoiser DNS48 reference.

The oracle imports ``denoiser.pretrained.dns48`` from the exact clean upstream
revision, strict-loads the official checkpoint, hooks the real encoder/LSTM/
decoder modules, and calls the package's own ``Demucs.forward``. It never
imports Vokra and defines no mirror model.

Run model execution only on VAST through ``tools/parity/pyproject.toml``. The
``--self-test`` path is stdlib-only and checks deterministic input generation.
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


SOURCE_REVISION = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0"
CHECKPOINT_BYTES = 75_478_395
CHECKPOINT_SHA256_PREFIX = "11decc9d8e3f0998"
TENSOR_COUNT = 48
SAMPLE_RATE = 16_000
DEFAULT_SAMPLES = 4_096
SOURCE_FILES = {
    Path("denoiser/demucs.py"): (
        17_080,
        "8e9c21935c647e24f31cefcc63a298cb2a1c25bc99aab44bbe63a7b5570836be",
    ),
    Path("denoiser/resample.py"): (
        2_187,
        "3e8ea258036660b7d33415794fe09ee010510f4d760bdfc5d5de268d6efb40f5",
    ),
    Path("denoiser/pretrained.py"): (
        3_070,
        "885ad1ddd6cee5d4ecf5b4bc32784ceee97dc37ae19570b7ce0f9869b360d108",
    ),
    Path("LICENSE"): (
        19_333,
        "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf",
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
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{relative}: unexpected byte length")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{relative}: unexpected SHA-256")
    return checkout


def validate_checkpoint(path: Path) -> str:
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(f"checkpoint byte length != pinned {CHECKPOINT_BYTES}")
    digest = sha256_file(path)
    if not digest.startswith(CHECKPOINT_SHA256_PREFIX):
        raise ValueError(
            f"checkpoint SHA-256 {digest} does not start with {CHECKPOINT_SHA256_PREFIX}"
        )
    return digest


def deterministic_pcm(samples: int) -> tuple[float, ...]:
    if samples < 2:
        raise ValueError("samples must be at least two")
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
        raise RuntimeError(f"official stage {name} returned {type(value)!r}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"official stage {name} contains non-finite values")
    return value.detach().cpu().contiguous()


def dump(source: Path, checkpoint: Path, output: Path, samples: int) -> None:
    import numpy as np
    import torch

    source = validate_source(source)
    checkpoint_sha256 = validate_checkpoint(checkpoint)
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from denoiser.pretrained import dns48

        imported = Path(sys.modules[dns48.__module__].__file__).resolve()
    except Exception as error:  # noqa: BLE001 - loud independent-oracle boundary
        raise RuntimeError(
            "could not import the pinned official denoiser package; mirror fallback forbidden"
        ) from error
    finally:
        sys.path.pop(0)
    expected_import = (source / "denoiser/pretrained.py").resolve()
    if imported != expected_import:
        raise ValueError(f"imported denoiser from {imported}, expected {expected_import}")

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = dns48(pretrained=False).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if len(state) != TENSOR_COUNT:
        raise ValueError(f"checkpoint has {len(state)} tensors, expected {TENSOR_COUNT}")
    model.load_state_dict(state, strict=True)
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("official DNS48 checkpoint must contain only F32 tensors")

    captures: dict[str, Any] = {}
    handles = []

    def hook(name: str):
        def capture(_module: object, _inputs: object, value: object) -> None:
            captures[name] = tensor_output(name, value)

        return capture

    for index, module in enumerate(model.encoder):
        handles.append(module.register_forward_hook(hook(f"encoder_{index}")))
    handles.append(model.lstm.register_forward_hook(hook("lstm")))
    for index, module in enumerate(model.decoder):
        handles.append(module.register_forward_hook(hook(f"decoder_{index}")))

    pcm = np.asarray(deterministic_pcm(samples), dtype=np.float32)
    with torch.inference_mode():
        waveform = model(torch.from_numpy(pcm).unsqueeze(0))
    for handle in handles:
        handle.remove()
    waveform = tensor_output("waveform", waveform)
    expected_captures = {
        *(f"encoder_{index}" for index in range(5)),
        "lstm",
        *(f"decoder_{index}" for index in range(5)),
    }
    if set(captures) != expected_captures:
        raise RuntimeError(
            f"official hooks yielded {sorted(captures)}, expected {sorted(expected_captures)}"
        )

    output.mkdir(parents=True)
    files = {
        "pcm": write_f32(output / "pcm.f32le", pcm),
        "waveform": write_f32(output / "waveform.f32le", waveform.numpy()),
    }
    taps = output / "taps"
    taps.mkdir()
    for name in sorted(captures):
        files[name] = write_f32(taps / f"{name}.f32le", captures[name].numpy())

    manifest = {
        "oracle": "facebookresearch/denoiser::pretrained.dns48 Demucs.forward",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "sample_rate": SAMPLE_RATE,
        "samples": samples,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not all([args.source, args.checkpoint, args.output]):
        parser.error("--source, --checkpoint and --output are required")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        if args.source is not None:
            validate_source(args.source)
        values = deterministic_pcm(32)
        payload = b"".join(struct.pack("<f", value) for value in values)
        assert len(payload) == 128
        assert values == deterministic_pcm(32)
        print(
            "facebook_denoiser_dump_reference self-test: "
            f"sha256={hashlib.sha256(payload).hexdigest()}"
        )
        return 0
    dump(args.source, args.checkpoint, args.output, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
