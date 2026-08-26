#!/usr/bin/env python3
"""Dump an independent official pyannote.audio 3.0.0 PyanNet reference.

The oracle imports ``PyanNet`` from the exact upstream source checkout at
``795b92ab265888c58d160f90ae4d91b7bcc6aa2c``.  It restores the immutable
public Vokra GGUF into the official PyTorch modules and executes their forward
method.  No PyanNet layer or arithmetic is mirrored in this file.

The real run belongs on vast.ai through
``scripts/publish/vast-ai/run-pyannote-segmentation-parity.sh``.  ``--self-test``
uses only the Python standard library and is safe on the maintainer Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


PYANNOTE_AUDIO_REPO = "https://github.com/pyannote/pyannote-audio.git"
PYANNOTE_AUDIO_REVISION = "795b92ab265888c58d160f90ae4d91b7bcc6aa2c"
PYANNOTE_AUDIO_VERSION = "3.0.0"

PUBLIC_REPO = "vokra/pyannote-segmentation-3.0"
PUBLIC_REVISION = "50bf4e510e0c689668384aec0f866f02e0fcaea8"
PUBLIC_FILE = "pyannote-seg.gguf"
PUBLIC_BYTES = 5_898_272
PUBLIC_SHA256 = "22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1"
PUBLIC_TENSOR_COUNT = 54

SOURCE_FILES = {
    "pyannote/audio/models/segmentation/PyanNet.py": (
        6_650,
        "8d576a0992d56f23f1c065e1ee211747649e3fd1494eee50e994aec7856f09ff",
    ),
    "pyannote/audio/models/blocks/sincnet.py": (
        3_383,
        "cb54c32a5e7965b2c068dedf9314168cf79de7e1f45f92740d380dee8e56db03",
    ),
    "pyannote/audio/core/model.py": (
        25_328,
        "e2f019a1f083db8c5a4956238c6f4e05dcda5a9ccfcd2343a926df88a54b951d",
    ),
    "pyannote/audio/core/task.py": (
        17_526,
        "a5903cd9e1e16ec96267a4b5ebe6d8786fec8a1ebad246fb84eca5a8094c47e2",
    ),
    "pyannote/audio/utils/params.py": (
        227,
        "d00744902570de2de28d4872c809b67b951d3ca2a8ca7715f10c5e17aed013fa",
    ),
    "LICENSE": (
        1_061,
        "a3b53644a76e70e289b25271b119c0a1aadaaf0db7a16225fb494fdc0e36c32a",
    ),
}

SAMPLE_RATE = 16_000
SAMPLES = 1_600
SIGNAL_SEED = 0x6D2B79F5
SIGNAL_SHA256 = "25c545e325976d426a1414e0408bc54f7b14d6f1e3c317e965b479938f4f8c81"
EXPECTED_OUTPUT_FRAMES = 3
EXPECTED_CLASSES = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label}: missing {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"{label}: {actual_size} bytes != expected {size}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label}: SHA-256 {actual_hash} != expected {expected_hash}"
        )


def verify_source_checkout(source: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PYANNOTE_AUDIO_REVISION:
        raise RuntimeError(
            f"pyannote.audio checkout is {revision}, expected {PYANNOTE_AUDIO_REVISION}"
        )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"pyannote.audio checkout is dirty:\n{dirty}")
    for relative, (size, expected_hash) in SOURCE_FILES.items():
        verify_file(source / relative, size, expected_hash, f"official source {relative}")


def deterministic_pcm_bytes() -> bytes:
    """Return exact binary-rational f32 PCM without a platform RNG."""

    state = SIGNAL_SEED
    values: list[float] = []
    for _ in range(SAMPLES):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        signed_24 = (state >> 8) - 8_388_608
        values.append(signed_24 / 33_554_432.0)
    return struct.pack(f"<{SAMPLES}f", *values)


def tensor_value(value: Any):
    if isinstance(value, tuple):
        value = value[0]
    return value.detach().to(device="cpu", dtype=value.dtype).contiguous()


def gguf_tensor(item: Any, expected_shape: Any, np: Any, torch: Any):
    if int(item.tensor_type) != 0:
        raise RuntimeError(f"{item.name}: public PyanNet tensor is not F32")
    values = item.data.copy().reshape(-1).astype(np.float32, copy=False)
    expected_elements = 1
    for axis in expected_shape:
        expected_elements *= int(axis)
    if values.size != expected_elements:
        raise RuntimeError(
            f"{item.name}: {values.size} values != expected {expected_elements}"
        )
    return torch.from_numpy(values.reshape(tuple(expected_shape)).copy())


def build_official_model(source: Path, gguf: Path):
    try:
        import numpy as np
        import torch
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(f"missing locked parity dependency: {error}") from error

    sys.path.insert(0, str(source))
    try:
        import pyannote.audio
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from pyannote.audio.models.segmentation.PyanNet import PyanNet
    except ImportError as error:
        raise RuntimeError(f"cannot import official pyannote.audio source: {error}") from error

    imported_source = Path(inspect.getsourcefile(PyanNet) or "").resolve()
    expected_source = (source / "pyannote/audio/models/segmentation/PyanNet.py").resolve()
    if imported_source != expected_source:
        raise RuntimeError(
            f"PyanNet imported from {imported_source}, expected checkout file {expected_source}"
        )
    if pyannote.audio.__version__ != PYANNOTE_AUDIO_VERSION:
        raise RuntimeError(
            f"checkout pyannote.audio is {pyannote.audio.__version__}, "
            f"expected {PYANNOTE_AUDIO_VERSION}"
        )
    installed_version = importlib.metadata.version("pyannote.audio")
    if installed_version != PYANNOTE_AUDIO_VERSION:
        raise RuntimeError(
            f"pyannote.audio distribution is {installed_version}, expected {PYANNOTE_AUDIO_VERSION}"
        )

    model = PyanNet(
        sincnet={"stride": 10},
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.0,
        },
        linear={"hidden_size": 128, "num_layers": 2},
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )
    model.specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=SAMPLES / SAMPLE_RATE,
        classes=["speaker#1", "speaker#2", "speaker#3"],
        powerset_max_classes=2,
        permutation_invariant=True,
    )
    model.build()

    expected_state = model.state_dict()
    if len(expected_state) != PUBLIC_TENSOR_COUNT:
        raise RuntimeError(
            f"official PyanNet state has {len(expected_state)} tensors, expected {PUBLIC_TENSOR_COUNT}"
        )
    reader = GGUFReader(str(gguf))
    by_name = {item.name: item for item in reader.tensors}
    if len(by_name) != len(reader.tensors):
        raise RuntimeError("public GGUF contains duplicate tensor names")
    expected_names = set(expected_state)
    actual_names = set(by_name)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"official/GGUF tensor-name mismatch: missing={missing}, extra={extra}")

    restored = {
        name: gguf_tensor(by_name[name], target.shape, np, torch)
        for name, target in expected_state.items()
    }
    incompatible = model.load_state_dict(restored, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict official state load mismatch: {incompatible}")
    model.eval()
    return model, np, torch


def dump(args: argparse.Namespace) -> None:
    verify_source_checkout(args.pyannote_source)
    verify_file(args.gguf, PUBLIC_BYTES, PUBLIC_SHA256, "public PyanNet GGUF")
    model, np, torch = build_official_model(args.pyannote_source, args.gguf)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pcm_bytes = deterministic_pcm_bytes()
    actual_signal_hash = hashlib.sha256(pcm_bytes).hexdigest()
    if actual_signal_hash != SIGNAL_SHA256:
        raise RuntimeError(
            f"deterministic PCM SHA-256 {actual_signal_hash} != {SIGNAL_SHA256}"
        )
    pcm = np.frombuffer(pcm_bytes, dtype="<f4").copy()
    waveform = torch.from_numpy(pcm).reshape(1, 1, -1)

    captured: dict[str, Any] = {}

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[name] = tensor_value(output)

        return hook

    def capture_lstm(_module: Any, _inputs: Any, output: Any) -> None:
        sequence, (hidden, cell) = output
        captured["lstm_features"] = tensor_value(sequence)
        captured["lstm_hidden"] = tensor_value(hidden)
        captured["lstm_cell"] = tensor_value(cell)

    def capture_input(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            if len(inputs) != 1:
                raise RuntimeError(f"{name}: expected one positional input")
            captured[name] = tensor_value(inputs[0])

        return hook

    handles = [
        model.sincnet.register_forward_hook(capture("sincnet_features")),
        model.lstm.register_forward_hook(capture_lstm),
        model.linear[1].register_forward_pre_hook(capture_input("linear0_activated")),
        model.classifier.register_forward_pre_hook(capture_input("linear1_activated")),
        model.classifier.register_forward_hook(capture("logits")),
    ]
    with torch.inference_mode():
        log_probabilities = model(waveform)
        probabilities = log_probabilities.exp()
    for handle in handles:
        handle.remove()

    expected_shapes = {
        "sincnet_features": (1, 60, EXPECTED_OUTPUT_FRAMES),
        "lstm_features": (1, EXPECTED_OUTPUT_FRAMES, 256),
        "lstm_hidden": (8, 1, 128),
        "lstm_cell": (8, 1, 128),
        "linear0_activated": (1, EXPECTED_OUTPUT_FRAMES, 128),
        "linear1_activated": (1, EXPECTED_OUTPUT_FRAMES, 128),
        "logits": (1, EXPECTED_OUTPUT_FRAMES, EXPECTED_CLASSES),
    }
    actual_shapes = {name: tuple(value.shape) for name, value in captured.items()}
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"official hook shapes changed: {actual_shapes!r} != {expected_shapes!r}"
        )
    if tuple(log_probabilities.shape) != (1, EXPECTED_OUTPUT_FRAMES, EXPECTED_CLASSES):
        raise RuntimeError(f"official output shape changed: {tuple(log_probabilities.shape)}")
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("official PyanNet emitted non-finite probabilities")
    row_sums = probabilities.sum(dim=-1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=0.0)):
        raise RuntimeError(f"official PyanNet probability rows do not sum to one: {row_sums}")

    outputs = {
        "input_pcm": torch.from_numpy(pcm),
        **captured,
        "log_probabilities": log_probabilities,
        "probabilities": probabilities,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs: dict[str, dict[str, Any]] = {}
    for name, value in outputs.items():
        array = value.detach().cpu().numpy().astype("<f4", copy=False)
        destination = args.output_dir / f"{name}.f32"
        array.tofile(destination)
        manifest_outputs[name] = {
            "shape": list(array.shape),
            "elements": int(array.size),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    manifest = {
        "format": "vokra-pyannote-segmentation-reference-v1",
        "oracle": "official pyannote.audio 3.0.0 PyanNet.forward imported without reimplementation",
        "source_repo": PYANNOTE_AUDIO_REPO,
        "source_revision": PYANNOTE_AUDIO_REVISION,
        "source_files": {
            path: {"bytes": size, "sha256": digest}
            for path, (size, digest) in SOURCE_FILES.items()
        },
        "public_repo": PUBLIC_REPO,
        "public_revision": PUBLIC_REVISION,
        "public_file": PUBLIC_FILE,
        "public_bytes": PUBLIC_BYTES,
        "public_sha256": PUBLIC_SHA256,
        "official_state_tensors_loaded": PUBLIC_TENSOR_COUNT,
        "sample_rate": SAMPLE_RATE,
        "signal_sha256": SIGNAL_SHA256,
        "topology": {
            "sincnet_stride": 10,
            "lstm_hidden_size": 128,
            "lstm_num_layers": 4,
            "lstm_bidirectional": True,
            "lstm_monolithic": True,
            "linear_hidden_size": 128,
            "linear_num_layers": 2,
            "powerset_classes": EXPECTED_CLASSES,
        },
        "activation": "official LogSoftmax; probabilities.f32 is exp(forward output)",
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "pyannote_audio": importlib.metadata.version("pyannote.audio"),
            "gguf": importlib.metadata.version("gguf"),
        },
        "outputs": manifest_outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PYANNOTE_OFFICIAL_REFERENCE "
        f"frames={EXPECTED_OUTPUT_FRAMES} classes={EXPECTED_CLASSES} "
        f"tensors={PUBLIC_TENSOR_COUNT} verdict=PASS"
    )


def self_test() -> None:
    if hashlib.sha256(deterministic_pcm_bytes()).hexdigest() != SIGNAL_SHA256:
        raise RuntimeError("deterministic signal contract drifted")
    if len(PYANNOTE_AUDIO_REVISION) != 40 or len(PUBLIC_REVISION) != 40:
        raise RuntimeError("revision contract must use full 40-hex commits")
    if len(PUBLIC_SHA256) != 64 or len(SOURCE_FILES) != 6:
        raise RuntimeError("identity contract is incomplete")
    if PUBLIC_TENSOR_COUNT != 54 or EXPECTED_CLASSES != 7:
        raise RuntimeError("released topology contract drifted")
    print("pyannote_segmentation_dump_reference: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyannote-source", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and any(
        value is None for value in (args.pyannote_source, args.gguf, args.output_dir)
    ):
        parser.error("--pyannote-source, --gguf, and --output-dir are required")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
