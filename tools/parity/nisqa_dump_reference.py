#!/usr/bin/env python3
"""Dump an independent official NISQA v2 multidimensional reference.

The oracle imports ``nisqa.NISQA_lib.NISQA_DIM`` from the exact clean upstream
revision, calls the official mel/segmentation functions, strict-loads
``weights/nisqa.tar``, hooks real official modules, and invokes the official
forward. It never imports Vokra and defines no mirror model. Execute the real
dump only on VAST; ``--self-test`` is stdlib-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REVISION = "fe84f0f252abec382b24367d5b22498a7ce34dbb"
CHECKPOINT_SHA256 = "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c"
SOURCE_FILES = {
    Path("nisqa/NISQA_lib.py"): (
        77_206,
        "f3ace1c00e21ae06e5d0fed9710f4e988c13685b2316a3b3ded46607fb25b71e",
    ),
    Path("LICENSE"): (
        1_098,
        "6c6c762447306a0fa89b130d5df177a5c2a79f39fc8cbf58aad68c5245da3f16",
    ),
    Path("weights/LICENSE_model_weights"): (
        20_843,
        "5b8e7938e1b5e0a675869ffe429cc8e7cc187d76a7c6ea1e0546c412782a43da",
    ),
}
MODEL_ARG_KEYS = (
    "ms_seg_length",
    "ms_n_mels",
    "cnn_model",
    "cnn_c_out_1",
    "cnn_c_out_2",
    "cnn_c_out_3",
    "cnn_kernel_size",
    "cnn_dropout",
    "cnn_pool_1",
    "cnn_pool_2",
    "cnn_pool_3",
    "cnn_fc_out_h",
    "td",
    "td_sa_d_model",
    "td_sa_nhead",
    "td_sa_pos_enc",
    "td_sa_num_layers",
    "td_sa_h",
    "td_sa_dropout",
    "td_lstm_h",
    "td_lstm_num_layers",
    "td_lstm_dropout",
    "td_lstm_bidirectional",
    "td_2",
    "td_2_sa_d_model",
    "td_2_sa_nhead",
    "td_2_sa_pos_enc",
    "td_2_sa_num_layers",
    "td_2_sa_h",
    "td_2_sa_dropout",
    "td_2_lstm_h",
    "td_2_lstm_num_layers",
    "td_2_lstm_dropout",
    "td_2_lstm_bidirectional",
    "pool",
    "pool_att_h",
    "pool_att_dropout",
)


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
    if git_output(checkout, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError("NISQA source revision differs from the pin")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("NISQA source checkout must be exactly clean")
    for relative, (expected_bytes, expected_sha256) in SOURCE_FILES.items():
        path = checkout / relative
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
            raise ValueError(f"{relative}: pinned source validation failed")
    return checkout


def deterministic_pcm(sample_rate: int, samples: int) -> tuple[float, ...]:
    if sample_rate <= 0 or samples < 2:
        raise ValueError("sample_rate must be positive and samples at least two")
    values = []
    for index in range(samples):
        time = index / sample_rate
        onset = min(1.0, index / max(1.0, sample_rate * 0.02))
        values.append(
            onset
            * (
                0.19 * math.sin(2.0 * math.pi * 173.0 * time + 0.1)
                + 0.07 * math.cos(2.0 * math.pi * 421.0 * time + 0.3)
                + 0.025 * math.sin(2.0 * math.pi * 997.0 * time)
            )
        )
    return tuple(values)


def tensor_output(name: str, value: object) -> Any:
    import torch

    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"official stage {name} returned {type(value)!r}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"official stage {name} contains non-finite values")
    return value.detach().cpu().contiguous()


def write_f32(path: Path, values: object) -> dict[str, object]:
    import numpy as np

    array = np.asarray(values, dtype="<f4")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{path.name}: output contains non-finite values")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "file": path.name,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dump(
    source: Path,
    checkpoint_path: Path,
    output: Path,
    sample_rate: int,
    samples: int,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch

    source = validate_source(source)
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("weights/nisqa.tar SHA-256 differs from the pin")
    if output.exists():
        raise ValueError(f"refusing to overwrite output directory: {output}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from nisqa import NISQA_lib as official

        imported = Path(official.__file__).resolve()
    except Exception as error:  # noqa: BLE001 - independent oracle boundary
        raise RuntimeError("could not import pinned official NISQA; mirror fallback forbidden") from error
    finally:
        sys.path.pop(0)
    expected_import = (source / "nisqa/NISQA_lib.py").resolve()
    if imported != expected_import:
        raise ValueError(f"imported NISQA_lib from {imported}, expected {expected_import}")

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model = official.NISQA_DIM(**{key: args[key] for key in MODEL_ARG_KEYS}).cpu().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    pcm = np.asarray(deterministic_pcm(sample_rate, samples), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav") as wave:
        sf.write(wave.name, pcm, sample_rate, subtype="FLOAT")
        mel = official.get_librosa_melspec(
            wave.name,
            sr=args["ms_sr"],
            n_fft=args["ms_n_fft"],
            hop_length=args["ms_hop_length"],
            win_length=args["ms_win_length"],
            n_mels=args["ms_n_mels"],
            fmax=args["ms_fmax"],
            ms_channel=None,
        )
    segments, n_wins_scalar = official.segment_specs(
        "deterministic.wav",
        mel,
        args["ms_seg_length"],
        seg_hop=args["ms_seg_hop_length"],
        max_length=args["ms_max_segments"],
    )
    n_wins_value = int(n_wins_scalar)
    n_wins = torch.tensor([n_wins_value], dtype=torch.long)

    captures: dict[str, Any] = {}
    handles = []

    def hook(name: str):
        def capture(_module: object, _inputs: object, value: object) -> None:
            captures[name] = tensor_output(name, value)

        return capture

    handles.append(model.cnn.model.register_forward_hook(hook("cnn")))
    handles.append(model.time_dependency.model.linear.register_forward_hook(hook("td_linear")))
    handles.append(model.time_dependency.model.norm1.register_forward_hook(hook("td_input_norm")))
    for index, layer in enumerate(model.time_dependency.model.layers):
        handles.append(layer.register_forward_hook(hook(f"attention_{index}")))
    for index, head in enumerate(model.pool_layers):
        handles.append(head.register_forward_hook(hook(f"pool_{index}")))

    with torch.inference_mode():
        score = model(segments.unsqueeze(0), n_wins)
    for handle in handles:
        handle.remove()
    score = tensor_output("score", score)
    expected_captures = {
        "cnn",
        "td_linear",
        "td_input_norm",
        "attention_0",
        "attention_1",
        "pool_0",
        "pool_1",
        "pool_2",
        "pool_3",
        "pool_4",
    }
    if set(captures) != expected_captures:
        raise RuntimeError(f"official hook set drift: {sorted(captures)}")

    output.mkdir(parents=True)
    taps = output / "taps"
    taps.mkdir()
    files = {
        "pcm": write_f32(output / "pcm.f32le", pcm),
        "mel": write_f32(output / "mel.f32le", mel),
        "segments": write_f32(output / "segments.f32le", segments[:n_wins_value].numpy()),
        "score": write_f32(output / "score.f32le", score.numpy()),
    }
    for name in sorted(captures):
        files[name] = write_f32(taps / f"{name}.f32le", captures[name].numpy())
    manifest = {
        "oracle": "official gabrielmittag/NISQA NISQA_lib.NISQA_DIM direct import",
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "sample_rate": sample_rate,
        "samples": samples,
        "n_wins": n_wins_value,
        "head_order": ["mos", "noi", "dis", "col", "loud"],
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"score": score.numpy().reshape(-1).tolist(), "n_wins": n_wins_value}))


def self_test() -> None:
    first = deterministic_pcm(48_000, 32)
    second = deterministic_pcm(48_000, 32)
    assert first == second and len(first) == 32
    assert max(abs(value) for value in first) < 1.0
    assert len(MODEL_ARG_KEYS) == 37
    print("nisqa_dump_reference: self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--samples", type=int, default=144_000)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if None in (args.source, args.checkpoint, args.output):
        raise SystemExit("--source, --checkpoint, and --output are required")
    dump(args.source, args.checkpoint, args.output, args.sample_rate, args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
