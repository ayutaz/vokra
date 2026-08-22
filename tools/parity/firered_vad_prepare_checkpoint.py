#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kaldi-native-fbank>=1.22",
#     "kaldiio>=2.18",
#     "numpy>=1.26",
#     "onnx>=1.17",
#     "onnxruntime>=1.19",
#     "safetensors>=0.4",
#     "soundfile>=0.12",
# ]
# ///
"""Prepare the official FireRedVAD streaming checkpoint for Vokra.

The runtime never loads ONNX or Kaldi archives.  This offline Python 3.12
bridge consumes the official ``fireredvad_stream_vad_with_cache.onnx`` and
``cmvn.ark`` files, writes a canonical 39-tensor safetensors bundle, and runs
the official ONNX graph directly to produce an independent feature/probability
reference JSON.

Pinned primary source: ``FireRedTeam/FireRedVAD`` commit
``c30ec49e8cc69642b0ee65362eba11b9d11c6e54`` (Apache-2.0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


UPSTREAM_REVISION = "c30ec49e8cc69642b0ee65362eba11b9d11c6e54"
SAMPLE_RATE = 16_000
N_MELS = 80
FRAME_LENGTH = 400
FRAME_SHIFT = 160
N_BLOCKS = 8
HIDDEN_DIM = 256
PROJECTION_DIM = 128
MEMORY_ORDER = 20
MEMORY_STRIDE = 1
CACHE_LENGTH = 19


def tensor_spec() -> dict[str, tuple[str, tuple[int, ...]]]:
    spec: dict[str, tuple[str, tuple[int, ...]]] = {
        "firered_vad.dfsmn.fc1.weight": ("onnx::MatMul_403", (80, 256)),
        "firered_vad.dfsmn.fc1.bias": ("model.dfsmn.fc1.0.bias", (256,)),
        "firered_vad.dfsmn.fc2.weight": ("onnx::MatMul_404", (256, 128)),
        "firered_vad.dfsmn.fc2.bias": ("model.dfsmn.fc2.0.bias", (128,)),
        "firered_vad.dfsmn.memory.0.weight": (
            "model.dfsmn.fsmn1.lookback_filter.weight",
            (128, 1, 20),
        ),
        "firered_vad.dfsmn.dnn.0.weight": ("onnx::MatMul_515", (128, 256)),
        "firered_vad.dfsmn.dnn.0.bias": ("model.dfsmn.dnns.0.bias", (256,)),
        "firered_vad.output.weight": ("onnx::MatMul_516", (256, 1)),
        "firered_vad.output.bias": ("model.out.bias", (1,)),
    }
    matmul_ids = (417, 431, 445, 459, 473, 487, 501)
    projection_ids = (418, 432, 446, 460, 474, 488, 502)
    for index, (fc1_id, fc2_id) in enumerate(zip(matmul_ids, projection_ids)):
        spec[f"firered_vad.dfsmn.block.{index}.fc1.weight"] = (
            f"onnx::MatMul_{fc1_id}",
            (128, 256),
        )
        spec[f"firered_vad.dfsmn.block.{index}.fc1.bias"] = (
            f"model.dfsmn.fsmns.{index}.fc1.0.bias",
            (256,),
        )
        spec[f"firered_vad.dfsmn.block.{index}.fc2.weight"] = (
            f"onnx::MatMul_{fc2_id}",
            (256, 128),
        )
        spec[f"firered_vad.dfsmn.memory.{index + 1}.weight"] = (
            f"model.dfsmn.fsmns.{index}.fsmn.lookback_filter.weight",
            (128, 1, 20),
        )
    return spec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cmvn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import kaldiio

    stats = np.asarray(kaldiio.load_mat(str(path)), dtype=np.float64)
    if stats.shape != (2, N_MELS + 1):
        raise SystemExit(
            f"FireRedVAD CMVN shape is {stats.shape}, expected (2, {N_MELS + 1})"
        )
    count = float(stats[0, N_MELS])
    if not np.isfinite(count) or count < 1.0:
        raise SystemExit(f"FireRedVAD CMVN count is invalid: {count}")
    mean = stats[0, :N_MELS] / count
    variance = stats[1, :N_MELS] / count - mean * mean
    inverse_std = 1.0 / np.sqrt(np.maximum(variance, 1.0e-20))
    return mean.astype(np.float32), inverse_std.astype(np.float32)


def extract_weights(onnx_path: Path, cmvn_path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(onnx_path).graph
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    output: dict[str, np.ndarray] = {}
    used: set[str] = set()
    for canonical, (source, shape) in tensor_spec().items():
        if source not in initializers:
            raise SystemExit(f"official ONNX is missing initializer {source!r}")
        value = np.asarray(initializers[source])
        if value.shape != shape:
            raise SystemExit(
                f"initializer {source!r} has shape {value.shape}, expected {shape}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise SystemExit(f"initializer {source!r} is not floating: {value.dtype}")
        output[canonical] = np.ascontiguousarray(value, dtype=np.float32)
        used.add(source)

    float_initializers = {
        name
        for name, value in initializers.items()
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    }
    if used != float_initializers:
        missing = sorted(float_initializers - used)
        stale = sorted(used - float_initializers)
        raise SystemExit(
            "official ONNX float-initializer manifest drifted: "
            f"unmapped={missing}, non_float_mapped={stale}"
        )

    mean, inverse_std = read_cmvn(cmvn_path)
    output["firered_vad.cmvn.mean"] = mean
    output["firered_vad.cmvn.inverse_std"] = inverse_std
    if len(output) != 39:
        raise SystemExit(f"canonical bundle has {len(output)} tensors, expected 39")
    return output


def read_pcm16(path: Path) -> np.ndarray:
    import soundfile as sf

    pcm, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"input WAV is {sample_rate} Hz, expected {SAMPLE_RATE} Hz")
    if pcm.ndim != 1:
        raise SystemExit(f"input WAV has shape {pcm.shape}, expected mono")
    return np.asarray(pcm, dtype=np.int16)


def official_features(pcm16: np.ndarray, mean: np.ndarray, inverse_std: np.ndarray) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = N_MELS
    opts.mel_opts.debug_mel = False
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(SAMPLE_RATE, pcm16.tolist())
    rows = [fbank.get_frame(index) for index in range(fbank.num_frames_ready)]
    if not rows:
        raise SystemExit("input WAV is too short to produce one FireRedVAD frame")
    features = np.asarray(rows, dtype=np.float32)
    return np.ascontiguousarray((features - mean) * inverse_std, dtype=np.float32)


def reference(
    onnx_path: Path,
    wav_path: Path,
    mean: np.ndarray,
    inverse_std: np.ndarray,
    max_frames: int,
) -> dict[str, object]:
    import onnxruntime as ort

    pcm16 = read_pcm16(wav_path)
    features = official_features(pcm16, mean, inverse_std)
    if max_frames > 0:
        features = features[:max_frames]
    caches = np.zeros(
        (N_BLOCKS, 1, PROJECTION_DIM, CACHE_LENGTH), dtype=np.float32
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probabilities, caches_out = session.run(
        None,
        {"feat": features[None, :, :], "caches_in": caches},
    )
    expected_cache_shape = (N_BLOCKS, 1, PROJECTION_DIM, CACHE_LENGTH)
    if caches_out.shape != expected_cache_shape:
        raise SystemExit(
            f"official ONNX cache output is {caches_out.shape}, expected {expected_cache_shape}"
        )
    return {
        "sample_rate": SAMPLE_RATE,
        "frame_length": FRAME_LENGTH,
        "frame_shift": FRAME_SHIFT,
        "n_frames": int(features.shape[0]),
        "features": features.tolist(),
        "probabilities": np.asarray(probabilities[0, :, 0], dtype=np.float32).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--cmvn", required=True, type=Path)
    parser.add_argument("--input-wav", required=True, type=Path)
    parser.add_argument("--output-st", required=True, type=Path)
    parser.add_argument("--output-ref", required=True, type=Path)
    parser.add_argument("--max-reference-frames", type=int, default=64)
    args = parser.parse_args()

    for label, path in (("--onnx", args.onnx), ("--cmvn", args.cmvn), ("--input-wav", args.input_wav)):
        if not path.is_file():
            parser.error(f"{label} is not a regular file: {path}")
    for path in (args.output_st, args.output_ref):
        if path.exists():
            parser.error(f"refusing to overwrite output: {path}")
    if args.max_reference_frames < 0:
        parser.error("--max-reference-frames must be >= 0")

    from safetensors.numpy import save_file

    tensors = extract_weights(args.onnx, args.cmvn)
    mean = tensors["firered_vad.cmvn.mean"]
    inverse_std = tensors["firered_vad.cmvn.inverse_std"]
    save_file(tensors, str(args.output_st))
    payload = reference(
        args.onnx,
        args.input_wav,
        mean,
        inverse_std,
        args.max_reference_frames,
    )
    payload.update(
        {
            "upstream_revision": UPSTREAM_REVISION,
            "onnx_sha256": sha256(args.onnx),
            "cmvn_sha256": sha256(args.cmvn),
            "input_wav_sha256": sha256(args.input_wav),
            "safetensors_sha256": sha256(args.output_st),
        }
    )
    args.output_ref.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"firered-vad prepared: tensors={len(tensors)}, frames={payload['n_frames']}, "
        f"safetensors_sha256={payload['safetensors_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
