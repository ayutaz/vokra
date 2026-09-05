#!/usr/bin/env python3
"""Dump an independent WeSpeaker ResNet34-LM reference fixture.

The oracle imports the pinned upstream WeSpeaker source tree, loads the
official ``avg_model`` checkpoint directly, and uses torchaudio's Kaldi fbank.
It never reads a Vokra GGUF and contains no local copy of the ResNet forward.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


np = None
torch = None
torchaudio = None


MODEL_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODEL_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
SOURCE_REVISION = "45941e7cba2c3ea99e232d02bedf617fc71b0dad"
CHECKPOINT_FILE = "avg_model.pt"
CHECKPOINT_BYTES = 45_053_131
CHECKPOINT_SHA256 = "9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 32_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runtime() -> None:
    """Load model dependencies into module globals only for production runs."""
    global np, torch, torchaudio
    import numpy as np_module
    import torch as torch_module
    import torchaudio as torchaudio_module

    np = np_module
    torch = torch_module
    torchaudio = torchaudio_module


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))


def deterministic_pcm() -> np.ndarray:
    index = np.arange(PCM_SAMPLES, dtype=np.float64)
    time = index / SAMPLE_RATE
    rng = np.random.default_rng(0x5753504B)
    envelope = np.minimum(1.0, index / 1200.0) * np.minimum(
        1.0, (PCM_SAMPLES - 1 - index) / 1600.0
    )
    chirp_phase = 2.0 * np.pi * (95.0 * time + 0.5 * 185.0 * time * time)
    pcm = envelope * (
        0.31 * np.sin(chirp_phase)
        + 0.17 * np.sin(2.0 * np.pi * 233.0 * time + 0.4)
        + 0.09 * np.sin(2.0 * np.pi * 711.0 * time + 1.1)
    )
    pcm += rng.normal(0.0, 0.003, PCM_SAMPLES)
    return np.asarray(np.clip(pcm, -0.95, 0.95), dtype=np.float32)


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"expected checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and nested and all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in nested.items()
        ):
            checkpoint = nested
            break
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in checkpoint.items()
    ):
        raise SystemExit("checkpoint is not a string-to-tensor state dict")
    return checkpoint  # type: ignore[return-value]


def stdlib_self_test() -> None:
    if len(MODEL_REVISION) != 40 or len(SOURCE_REVISION) != 40:
        raise SystemExit("WeSpeaker identity constants are malformed")
    source = Path(__file__).read_text(encoding="utf-8")
    unsafe = "weights_only=" + "False"
    if unsafe in source or "torch.load" not in source:
        raise SystemExit("unsafe or missing checkpoint loader contract")
    if "_load_runtime()" not in source or "global np, torch, torchaudio" not in source:
        raise SystemExit("model helpers are not bound to the lazy runtime loader")
    for guard in ("args.checkpoint.is_symlink()", "args.output_dir.is_symlink()", "source_input.is_symlink()"):
        if guard not in source:
            raise SystemExit(f"missing symlink guard: {guard}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wespeaker-source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        if args.output_dir or args.checkpoint or args.wespeaker_source:
            parser.error("--self-test accepts no model paths")
        stdlib_self_test()
        print("wespeaker_dump_reference self-test: OK")
        return 0
    if args.output_dir is None or args.checkpoint is None or args.wespeaker_source is None:
        parser.error("--output-dir, --checkpoint and --wespeaker-source are required")
    if args.output_dir.is_symlink() or args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("--output-dir must be absent or an empty regular directory")

    if args.checkpoint.is_symlink() or not args.checkpoint.is_file() or args.checkpoint.name != CHECKPOINT_FILE:
        raise SystemExit(f"checkpoint must be the exact regular {CHECKPOINT_FILE} file")
    if args.checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint identity mismatch")

    _load_runtime()

    source_input = args.wespeaker_source
    if source_input.is_symlink() or not source_input.is_dir():
        raise SystemExit(f"not a regular WeSpeaker source tree: {source_input}")
    source_root = source_input.resolve()
    if not (source_root / "wespeaker" / "models" / "resnet.py").is_file():
        raise SystemExit(f"not a WeSpeaker source tree: {source_root}")
    if subprocess.run(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout:
        raise SystemExit("source tree is dirty")
    if subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip() != SOURCE_REVISION:
        raise SystemExit("source revision mismatch")
    source_files = {
        "LICENSE": (11_357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        "wespeaker/models/resnet.py": (9_564, "6f3c8219be2c9a8b9eabed8169c1abaec3e48670be7aaf1e792138b2b20e68c4"),
        "wespeaker/models/pooling_layers.py": (10_255, "768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9"),
    }
    for relative, (size, expected) in source_files.items():
        path = source_root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source_root.resolve()) or path.stat().st_size != size or sha256(path) != expected:
            raise SystemExit(f"source identity mismatch: {relative}")
        blob = subprocess.run(["git", "-C", str(source_root), "hash-object", relative], check=True, capture_output=True, text=True).stdout.strip()
        expected_blob = {"LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "wespeaker/models/resnet.py": "17607e6d2c72627e15db4214cacfa9d7b89ca945", "wespeaker/models/pooling_layers.py": "47120eead47a511939267470496539804c17b7d3"}[relative]
        if blob != expected_blob:
            raise SystemExit(f"source Git blob mismatch: {relative}")
    sys.path.insert(0, str(source_root))
    resnet = importlib.import_module("wespeaker.models.resnet")

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    model = resnet.ResNet34(
        feat_dim=80,
        embed_dim=256,
        pooling_func="TSTP",
        two_emb_layer=False,
    ).eval()
    state = unwrap_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    expected_names = set(model.state_dict())
    available_names = set(state)
    missing = sorted(expected_names - available_names)
    extras = sorted(available_names - expected_names)
    if missing:
        raise SystemExit(f"checkpoint misses model tensors: {missing}")
    if extras != ["projection.weight"]:
        raise SystemExit(
            "expected only the unused LM classifier as an extra tensor, got "
            f"{extras}"
        )
    model.load_state_dict({name: state[name] for name in expected_names}, strict=True)

    pcm = deterministic_pcm()
    waveform = torch.from_numpy(pcm.copy()).unsqueeze(0)
    features = torchaudio.compliance.kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=80,
        frame_length=25.0,
        frame_shift=10.0,
        round_to_power_of_two=True,
        snip_edges=True,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
        use_energy=False,
    )
    features = features - features.mean(dim=0, keepdim=True)
    _, embedding = model(features.unsqueeze(0))

    if tuple(features.shape) != (198, 80):
        raise SystemExit(f"unexpected feature shape {tuple(features.shape)}")
    if tuple(embedding.shape) != (1, 256):
        raise SystemExit(f"unexpected embedding shape {tuple(embedding.shape)}")
    if not torch.isfinite(features).all() or not torch.isfinite(embedding).all():
        raise SystemExit("reference tensors contain non-finite values")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_f32(output / "pcm.f32.bin", pcm)
    write_f32(output / "features.f32.bin", features.cpu().numpy())
    write_f32(output / "embedding.f32.bin", embedding[0].cpu().numpy())
    manifest = {
        "format": "vokra-wespeaker-reference-v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_revision": SOURCE_REVISION,
        "sample_rate": SAMPLE_RATE,
        "pcm_samples": int(pcm.size),
        "feature_shape": list(features.shape),
        "embedding_shape": list(embedding.shape),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "runtime": "torch-cpu",
        "device": "cpu",
        "pcm_dtype": "float32-le",
        "features_dtype": "float32-le",
        "embedding_dtype": "float32-le",
    }
    for name in ("pcm.f32.bin", "features.f32.bin", "embedding.f32.bin"):
        manifest[f"sha256_{name.replace('.', '_')}"] = sha256(output / name)
        manifest[f"bytes_{name.replace('.', '_')}"] = (output / name).stat().st_size
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected_files = {"manifest.json", "pcm.f32.bin", "features.f32.bin", "embedding.f32.bin"}
    if {path.name for path in output.iterdir()} != expected_files:
        raise SystemExit("reference output file set is not exact")
    if manifest["device"] != "cpu" or manifest["runtime"] != "torch-cpu":
        raise SystemExit("reference runtime/device metadata drifted")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
