#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "onnx>=1.17",
#     "onnxruntime>=1.19",
#     "safetensors>=0.4",
#     "soundfile>=0.12",
# ]
# ///
"""openWakeWord safetensors + reference JSON preparer (owner-side, uv-managed).

Bridges the upstream `dscripka/openWakeWord` ONNX release to the Vokra
runtime binder at `crates/vokra-models/src/kws/openwakeword` by:

  1. Flattening `melspectrogram.onnx`, the shared
     `embedding_model.onnx` extractor, and each `<wakeword>.onnx` DNN into a
     single safetensors buffer that `vokra-cli convert --model
     openwakeword-op` consumes.
  2. Writing the `--output-config` side-car that same convert step
     REQUIRES. It carries `wakeword_names`, the per-wake-word labels the
     runtime returns to callers. Those labels are the one axis that
     exists nowhere in the safetensors — tensors are indexed
     positionally (`openwakeword.classifier.{idx}.…`) — and the
     converter refuses to invent them, consistent with this script's own
     refusal to infer a name from a path basename (see
     `_parse_wakeword_spec`).
  3. Running the official ONNX graphs with `onnxruntime` over a
     16 kHz mono WAV to emit per-hop wake-word probabilities as a JSON
     reference the Vokra parity harness at
     `crates/vokra-models/tests/parity_openwakeword.rs` compares against
     (max |Δ| bound = 1e-4).

The runtime never sees the ONNX (FR-LD-05); this script runs OFFLINE in
its own uv-managed Python 3.12 environment (per
`[[feedback-python-uses-uv]]` / `[[feedback-python-3-12]]`) so the
zero-dep NFR-DS-02 root Cargo.lock stays vokra-* only.

Usage
-----

    uv run python tools/parity/openwakeword_prepare_checkpoint.py \\
        --melspectrogram ~/openwakeword/melspectrogram.onnx \\
        --embedding     ~/openwakeword/embedding_model.onnx \\
        --wakeword      alexa=~/openwakeword/alexa_v0.1.onnx \\
        --wakeword      hey_jarvis=~/openwakeword/hey_jarvis_v0.1.onnx \\
        --input-wav     ~/test-speech.wav \\
        --output-st     ~/openwakeword.safetensors \\
        --output-config ~/openwakeword_config.json \\
        --output-ref    ~/openwakeword_reference.json \\
        --output-wav    ~/openwakeword-16k.wav

Then convert (the `--config` side-car is required):

    vokra-cli convert --model openwakeword-op \\
        --input  ~/openwakeword.safetensors \\
        --config ~/openwakeword_config.json \\
        --output ~/openwakeword.gguf

Every `--wakeword` argument is `name=path`; ordering is preserved into
the merged safetensors, the config side-car's `wakeword_names`, and the
reference JSON's `hop_probs` keys. That single ordering is what ties a
positionally-indexed classifier tensor back to its label.

Front-end axes
--------------

The graph shapes are the primary source for `window_frames`, `mel_bins`,
`embedding_dim`, `classifier_input_frames`, and classifier depth. The
streaming sample-rate/chunk contract is pinned to upstream release v0.5.1.

License / distribution note
---------------------------

The reference release ships the ONNX under Apache-2.0 code + a
CC-BY-NC-SA-4.0 official-weight term (upstream README §License). Vokra
does NOT redistribute the upstream weights — this script is a
user-side offline bridge. Users who redistribute the CC-BY-NC-SA-4.0
official weights must convert with `vokra-cli convert
--model openwakeword-op --license cc-by-nc-sa-4.0` which flips the
publish gate to NonCommercialShareAlike (fail-closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare openWakeWord safetensors + reference JSON for Vokra parity."
    )
    p.add_argument(
        "--melspectrogram",
        type=Path,
        required=True,
        help="Path to the official v0.5.1 melspectrogram.onnx. Its learned "
        "DFT and mel projection are required by the native runtime.",
    )
    p.add_argument(
        "--embedding",
        type=Path,
        required=True,
        help="Path to the official v0.5.1 embedding_model.onnx. All 20 "
        "convolutions are normalized into execution-order tensor names.",
    )
    p.add_argument(
        "--wakeword",
        action="append",
        required=True,
        help="Per-wake-word ONNX model as `name=path`. May be repeated. "
        "Order is preserved into the merged safetensors and reference JSON.",
    )
    p.add_argument(
        "--input-wav",
        type=Path,
        required=True,
        help="16 kHz mono WAV the reference pipeline runs over.",
    )
    p.add_argument(
        "--output-st",
        type=Path,
        required=True,
        help="Output safetensors path (fed to `vokra-cli convert --model openwakeword-op`).",
    )
    p.add_argument(
        "--output-config",
        type=Path,
        required=True,
        help="Output config side-car path (fed to `vokra-cli convert --model "
        "openwakeword-op --config`). Carries `wakeword_names` — the per-wake-word "
        "labels the runtime returns to callers, which exist nowhere in the "
        "safetensors and which the converter refuses to invent. Required, so the "
        "artifact chain from ONNX to a loadable GGUF cannot be left incomplete.",
    )
    p.add_argument(
        "--output-ref",
        type=Path,
        required=True,
        help="Output reference JSON path (fed to the Rust parity harness).",
    )
    p.add_argument(
        "--output-wav",
        type=Path,
        required=False,
        help="Optional path to write the resampled 16 kHz mono WAV the "
        "Rust parity harness consumes (defaults to `--input-wav` when omitted).",
    )
    return p.parse_args()


def _parse_wakeword_spec(spec: str) -> tuple[str, Path]:
    """`name=path` → (name, path). Loud-fails on missing `=`."""
    if "=" not in spec:
        raise SystemExit(
            f"--wakeword must be `name=path` (got `{spec}`) — the runtime binder "
            "keys the per-wake-word classifier bundle on `name` and needs it "
            "explicit (no silent path-basename inference)."
        )
    name, path_s = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--wakeword name must be non-empty (got `{spec}`)")
    return name, Path(path_s).expanduser()


def _load_onnx(path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    """Loads an ONNX graph plus all float initializers."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    out: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype not in (np.float32, np.float16):
            # Reshape/axes constants remain graph structure and are not
            # runtime weights. Any non-float initializer referenced by a
            # required Conv/Gemm/MatMul weight input still fails through
            # `_initializer` below.
            continue
        out[init.name] = np.ascontiguousarray(arr).astype(np.float32)
    return model, out


def _initializer(
    path: Path, tensors: dict[str, np.ndarray], name: str, context: str
) -> np.ndarray:
    try:
        return tensors[name]
    except KeyError as error:
        raise SystemExit(
            f"{path}: {context} input `{name}` is not a float initializer"
        ) from error


def _extract_melspectrogram(path: Path) -> dict[str, np.ndarray]:
    model, initializers = _load_onnx(path)
    convs = [node for node in model.graph.node if node.op_type == "Conv"]
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    if len(convs) != 2 or len(matmuls) != 1:
        raise SystemExit(
            f"{path}: expected two learned-DFT Conv nodes and one mel MatMul, "
            f"got {len(convs)} Conv + {len(matmuls)} MatMul"
        )
    dft: list[np.ndarray] = []
    for index, node in enumerate(convs):
        if len(node.input) < 2:
            raise SystemExit(f"{path}: DFT Conv {index} has no weight input")
        weight = _initializer(path, initializers, node.input[1], f"DFT Conv {index}")
        if weight.shape != (257, 1, 512):
            raise SystemExit(
                f"{path}: DFT Conv {index} weight has shape {weight.shape}, "
                "expected (257, 1, 512)"
            )
        dft.append(np.ascontiguousarray(weight[:, 0, :]))
    mel_node = matmuls[0]
    mel_candidates = [name for name in mel_node.input if name in initializers]
    if len(mel_candidates) != 1:
        raise SystemExit(
            f"{path}: mel MatMul must have exactly one initializer input, got "
            f"{mel_candidates}"
        )
    mel = initializers[mel_candidates[0]]
    if mel.shape != (257, 32):
        raise SystemExit(
            f"{path}: mel projection has shape {mel.shape}, expected (257, 32)"
        )
    return {
        "openwakeword.melspec.dft_real": dft[0],
        "openwakeword.melspec.dft_imag": dft[1],
        "openwakeword.melspec.mel": np.ascontiguousarray(mel),
    }


def _extract_embedding(path: Path) -> dict[str, np.ndarray]:
    model, initializers = _load_onnx(path)
    if len(model.graph.input) != 1:
        raise SystemExit(f"{path}: embedding graph must have exactly one input")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    values = [int(dim.dim_value) for dim in dims]
    if len(values) != 4 or values[1:] != [76, 32, 1]:
        raise SystemExit(
            f"{path}: embedding input shape is {values}, expected [batch, 76, 32, 1]"
        )
    convs = [node for node in model.graph.node if node.op_type == "Conv"]
    if len(convs) != 20:
        raise SystemExit(
            f"{path}: expected 20 embedding Conv nodes, got {len(convs)}"
        )
    out: dict[str, np.ndarray] = {}
    for index, node in enumerate(convs):
        if len(node.input) < 2:
            raise SystemExit(f"{path}: embedding Conv {index} has no weight input")
        weight = _initializer(
            path, initializers, node.input[1], f"embedding Conv {index}"
        )
        if weight.ndim != 4:
            raise SystemExit(
                f"{path}: embedding Conv {index} weight has rank {weight.ndim}, expected 4"
            )
        out[f"openwakeword.embedding.conv.{index}.weight"] = weight
        if index == 19:
            if len(node.input) >= 3 and node.input[2] in initializers:
                raise SystemExit(
                    f"{path}: final embedding Conv unexpectedly has a bias initializer"
                )
        else:
            if len(node.input) < 3:
                raise SystemExit(f"{path}: embedding Conv {index} has no bias input")
            bias = _initializer(
                path, initializers, node.input[2], f"embedding Conv {index}"
            )
            if bias.shape != (weight.shape[0],):
                raise SystemExit(
                    f"{path}: embedding Conv {index} bias has shape {bias.shape}, "
                    f"expected ({weight.shape[0]},)"
                )
            out[f"openwakeword.embedding.conv.{index}.bias"] = bias
    return out


def _attribute_int(node: Any, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def _static_classifier_input_shape(model: Any, path: Path) -> tuple[int, int]:
    if len(model.graph.input) != 1:
        raise SystemExit(f"{path}: classifier must have exactly one graph input")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    values = [int(dim.dim_value) for dim in dims]
    if len(values) != 3 or values[0] != 1 or values[1] <= 0 or values[2] <= 0:
        raise SystemExit(
            f"{path}: classifier input shape is {values}, expected static "
            "[1, frames, embedding]"
        )
    return values[1], values[2]


def _extract_classifier(
    path: Path, index: int
) -> tuple[dict[str, np.ndarray], int, int, int]:
    model, initializers = _load_onnx(path)
    ops = [node.op_type for node in model.graph.node]
    expected = ["Flatten", "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Sigmoid"]
    if ops != expected:
        raise SystemExit(
            f"{path}: classifier ops are {ops}, expected exact v0.5.1 topology {expected}"
        )
    input_frames, embedding_dim = _static_classifier_input_shape(model, path)
    previous = input_frames * embedding_dim
    out: dict[str, np.ndarray] = {}
    gemms = [node for node in model.graph.node if node.op_type == "Gemm"]
    for layer_index, node in enumerate(gemms):
        if len(node.input) != 3:
            raise SystemExit(
                f"{path}: Gemm layer {layer_index} needs data, weight, and bias inputs"
            )
        weight = _initializer(
            path, initializers, node.input[1], f"Gemm layer {layer_index} weight"
        )
        bias = _initializer(
            path, initializers, node.input[2], f"Gemm layer {layer_index} bias"
        )
        if weight.ndim != 2 or bias.ndim != 1:
            raise SystemExit(
                f"{path}: Gemm layer {layer_index} has weight {weight.shape}, bias "
                f"{bias.shape}; expected rank 2 + rank 1"
            )
        trans_b = _attribute_int(node, "transB", 0)
        if trans_b == 1:
            normalized = weight
        elif trans_b == 0:
            normalized = weight.T
        else:
            raise SystemExit(
                f"{path}: Gemm layer {layer_index} transB={trans_b}, expected 0 or 1"
            )
        if normalized.shape[1] != previous or normalized.shape[0] != bias.shape[0]:
            raise SystemExit(
                f"{path}: Gemm layer {layer_index} normalizes to {normalized.shape} with "
                f"bias {bias.shape}, expected [out, {previous}] + [out]"
            )
        prefix = f"openwakeword.classifier.{index}.linear.{layer_index}"
        out[f"{prefix}.weight"] = np.ascontiguousarray(normalized)
        out[f"{prefix}.bias"] = np.ascontiguousarray(bias)
        previous = int(normalized.shape[0])
    if previous != 1:
        raise SystemExit(f"{path}: final classifier width is {previous}, expected 1")
    return out, input_frames, embedding_dim, len(gemms)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_wav_16k_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    """Reads a WAV file into a mono `float32` numpy array. Loud-fails on
    non-16 kHz, non-PCM or multi-channel inputs (owner responsibility to
    resample upstream — the bridge does not fabricate a resampler)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sr != 16_000:
        raise SystemExit(
            f"{path}: sample rate is {sr} Hz — openwakeword expects 16 kHz. "
            "Resample upstream (e.g. `sox in.wav -r 16000 -c 1 out.wav`)."
        )
    if n_channels != 1:
        raise SystemExit(
            f"{path}: has {n_channels} channels — openwakeword expects mono. "
            "Downmix upstream."
        )
    if sampwidth != 2:
        raise SystemExit(
            f"{path}: sample width {sampwidth} bytes — expected PCM16."
        )
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sr


def _write_wav_16k_mono_f32(path: Path, samples: np.ndarray) -> None:
    """Writes a mono `float32` numpy array back out as 16 kHz PCM16 WAV."""
    pcm = np.clip(samples * 32768.0, -32768.0, 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())


def _emit_reference_probs(
    melspectrogram_path: Path,
    embedding_path: Path,
    wakeword_paths: list[tuple[str, Path]],
    samples: np.ndarray,
) -> dict[str, list[float]]:
    """Runs the official graphs with ONNX Runtime and returns hop scores.

    This deliberately avoids importing the `openwakeword` wheel: v0.5.1
    depends on a `tflite-runtime` build that has no CPython 3.12 wheel.
    ONNX Runtime remains independent of the Rust implementation while
    satisfying this repository's Python 3.12 rule.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    melspec_session = ort.InferenceSession(
        str(melspectrogram_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    embedding_session = ort.InferenceSession(
        str(embedding_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    classifier_sessions = [
        (
            name,
            ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            ),
        )
        for name, path in wakeword_paths
    ]

    hop = 1_280  # openwakeword default: 80 ms @ 16 kHz
    hop_probs: dict[str, list[float]] = {name: [] for name, _ in wakeword_paths}
    mel_buffer = np.ones((76, 32), dtype=np.float32)
    embedding_buffer = np.zeros((16, 96), dtype=np.float32)
    raw_context = np.empty((0,), dtype=np.int16)
    prediction_index = 0
    for start in range(0, len(samples) - hop + 1, hop):
        chunk = np.clip(
            np.rint(samples[start : start + hop] * 32768.0), -32768, 32767
        ).astype(np.int16)
        model_input = np.concatenate((raw_context, chunk)).astype(np.float32)[None, :]
        mel = melspec_session.run(
            None, {melspec_session.get_inputs()[0].name: model_input}
        )[0]
        mel = np.asarray(mel, dtype=np.float32).reshape(-1, 32) / 10.0 + 2.0
        mel_buffer = np.concatenate((mel_buffer, mel), axis=0)[-76:, :]
        embedding_input = mel_buffer[None, :, :, None]
        embedding = embedding_session.run(
            None, {embedding_session.get_inputs()[0].name: embedding_input}
        )[0]
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, 96)
        embedding_buffer = np.concatenate((embedding_buffer, embedding), axis=0)[-16:, :]
        for name, session in classifier_sessions:
            probability = float(
                np.asarray(
                    session.run(
                        None,
                        {
                            session.get_inputs()[0].name: embedding_buffer[
                                None, :, :
                            ]
                        },
                    )[0]
                ).reshape(-1)[0]
            )
            if prediction_index < 5:
                probability = 0.0
            hop_probs[name].append(probability)
        raw_context = np.concatenate((raw_context, chunk))[-480:]
        prediction_index += 1
    return hop_probs


def main() -> int:
    args = parse_args()
    wakeword_paths = [_parse_wakeword_spec(s) for s in args.wakeword]
    if len({name for name, _ in wakeword_paths}) != len(wakeword_paths):
        raise SystemExit("--wakeword names must be unique")

    # ---- safetensors half ---------------------------------------------------

    tensors: dict[str, np.ndarray] = {}
    tensors.update(_extract_melspectrogram(args.melspectrogram))
    tensors.update(_extract_embedding(args.embedding))

    classifier_input_frames: int | None = None
    embedding_dim: int | None = None
    classifier_layer_counts: list[int] = []
    for idx, (name, path) in enumerate(wakeword_paths):
        cls_tensors, frames, width, layer_count = _extract_classifier(path, idx)
        if classifier_input_frames is None:
            classifier_input_frames = frames
            embedding_dim = width
        elif frames != classifier_input_frames or width != embedding_dim:
            raise SystemExit(
                f"{path}: classifier input [{frames}, {width}] disagrees with "
                f"earlier [{classifier_input_frames}, {embedding_dim}]"
            )
        tensors.update(cls_tensors)
        classifier_layer_counts.append(layer_count)
        print(
            f"[{name}] classifier bound: input=[{frames}, {width}], "
            f"dense_layers={layer_count}",
            file=sys.stderr,
        )

    from safetensors.numpy import save_file

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st))
    print(f"wrote {args.output_st} ({len(tensors)} tensors)", file=sys.stderr)

    # ---- config side-car half -----------------------------------------------
    #
    # The ordering below is the SAME ordering used for the positional
    # `openwakeword.classifier.{idx}.*` tensor names above, which is what
    # lets the converter pair label i with classifier group i.
    config: dict[str, Any] = {
        "wakeword_names": [name for name, _ in wakeword_paths],
        "window_frames": 76,
        "mel_bins": 32,
        "sample_rate": 16_000,
        "hop_samples": 160,
        "predict_chunk_samples": 1_280,
        "classifier_format": "dnn-relu-sigmoid-v1",
        "classifier_input_frames": classifier_input_frames,
        "classifier_layer_counts": classifier_layer_counts,
        "upstream_release": "v0.5.1",
        "upstream_revision": "1eec2158c5c54150ac5f4c15065adacb1003b1e7",
        "source_sha256": {
            "melspectrogram": _sha256(args.melspectrogram),
            "embedding": _sha256(args.embedding),
            "wakewords": {
                name: _sha256(path) for name, path in wakeword_paths
            },
        },
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(
        f"wrote {args.output_config} "
        f"({len(config['wakeword_names'])} wake-word name(s)); pass it as "
        f"`vokra-cli convert --model openwakeword-op --config {args.output_config}`",
        file=sys.stderr,
    )

    # ---- reference JSON half ------------------------------------------------

    samples, _sr = _read_wav_16k_mono_f32(args.input_wav)
    ref_wav = args.output_wav if args.output_wav is not None else args.input_wav
    if args.output_wav is not None:
        _write_wav_16k_mono_f32(args.output_wav, samples)

    hop_probs = _emit_reference_probs(
        args.melspectrogram, args.embedding, wakeword_paths, samples
    )

    # NOTE ON THE TWO DIFFERENT "HOPS" (2026-08-15)
    #
    # This key used to be called `hop_samples`, which collided with
    # `vokra.openwakeword.hop_samples` in the GGUF while meaning something
    # else entirely, and 1280 vs 160 is an eight-fold difference that
    # would look plausible in either slot:
    #
    #   - 1280 samples (80 ms) is the chunk `openwakeword.Model.predict`
    #     consumes per call, i.e. how often a probability is emitted.
    #     That is what this reference JSON steps by, so it belongs here.
    #   - 160 samples (10 ms) is the mel ANALYSIS hop between melspec
    #     frames, which is what the GGUF key means.
    #
    # Renamed so nobody can copy the wrong one into a converter side-car.
    reference: dict[str, Any] = {
        "sample_rate": 16_000,
        "predict_chunk_samples": 1_280,
        "upstream_release": "v0.5.1",
        "upstream_revision": "1eec2158c5c54150ac5f4c15065adacb1003b1e7",
        "wav_path": str(ref_wav),
        "wakeword_names": [name for name, _ in wakeword_paths],
        "hop_probs": hop_probs,
    }
    args.output_ref.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ref.open("w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)
    print(
        f"wrote {args.output_ref} "
        f"({sum(len(v) for v in hop_probs.values())} total per-hop probs)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
