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

  1. Flattening the per-wake-word `<wakeword>.onnx` classifier MLPs and
     (optionally) the shared `embedding_model.onnx` extractor into a
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
  3. Running the upstream `openwakeword` + `onnxruntime` pipeline over a
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

The config side-car deliberately does NOT write `window_frames`,
`mel_bins`, `sample_rate` or `hop_samples`. This script has no
primary source for them, and the converter already carries documented
mirrors of the runtime binder's constants (76 / 32 / 16000 / 160) that
apply when the key is absent. Writing them here would put the same
numbers in a third place to drift. If you are converting a self-trained
checkpoint whose front-end differs, add the keys to the emitted JSON by
hand — the converter honours any that are present.

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
        "--embedding",
        type=Path,
        required=False,
        help="Optional path to embedding_model.onnx (Google speech_embedding). "
        "When present, its weights are also flattened into the merged safetensors "
        "under the `openwakeword.embedding.*` prefix — the runtime binder will "
        "pick these up in the follow-up wave that lights the extractor.",
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


def _load_onnx_initializer(path: Path) -> dict[str, np.ndarray]:
    """Loads an ONNX model file and returns every graph initializer as a
    `{name: ndarray}` dict (float initializers only; loud-skip on int /
    string initializers that the openWakeWord classifier + embedding
    graphs never carry)."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    out: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype not in (np.float32, np.float16):
            # openwakeword's classifier + embedding graphs use float
            # weights only; catch any surprise dtypes with a loud error
            # (mirror of the Rust-side FR-EX-08 posture).
            raise SystemExit(
                f"{path}: initializer `{init.name}` has dtype {arr.dtype} — "
                "the openwakeword bridge only supports float weights"
            )
        out[init.name] = np.ascontiguousarray(arr).astype(np.float32)
    return out


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
    embedding_path: Path,
    wakeword_paths: list[tuple[str, Path]],
    samples: np.ndarray,
) -> dict[str, list[float]]:
    """Runs the upstream `openwakeword` pipeline over `samples` and
    returns `{wakeword_name: [per_hop_prob, ...]}`. Skips gracefully with
    a stub warning if the `openwakeword` package is unavailable (the
    prep script's `safetensors` half still works)."""
    try:
        from openwakeword.model import Model
    except ImportError:
        print(
            "warning: `openwakeword` Python package unavailable — the reference "
            "JSON will contain empty per-wakeword probability arrays. Install "
            "with `uv add openwakeword` and re-run to fill them.",
            file=sys.stderr,
        )
        return {name: [] for name, _ in wakeword_paths}

    # `openwakeword.Model` wants the wakeword model paths + a shared
    # embedding path. Pass them explicitly rather than relying on the
    # bundled defaults (the caller may have staged custom files).
    m = Model(
        wakeword_models=[str(p) for _, p in wakeword_paths],
        melspec_model_path=None,  # Use the packaged mel front-end (same as runtime default).
        embedding_model_path=str(embedding_path) if embedding_path else None,
    )

    # Iterate the clip in 80 ms hops and record per-hop scores from
    # `m.prediction_buffer`. `m.predict` is called once per chunk (no
    # prior whole-clip call — that would prime state, then get wiped by
    # `reset()` a line later, which is dead work).
    hop = 1_280  # openwakeword default: 80 ms @ 16 kHz
    hop_probs: dict[str, list[float]] = {name: [] for name, _ in wakeword_paths}
    for start in range(0, len(samples) - hop + 1, hop):
        chunk = samples[start : start + hop]
        m.predict(chunk)
        for name in list(hop_probs):
            # `m.prediction_buffer[name]` grows one entry per `predict`.
            buf = list(m.prediction_buffer.get(name, []))
            if buf:
                hop_probs[name].append(float(buf[-1]))
    return hop_probs


def main() -> int:
    args = parse_args()
    wakeword_paths = [_parse_wakeword_spec(s) for s in args.wakeword]
    if len({name for name, _ in wakeword_paths}) != len(wakeword_paths):
        raise SystemExit("--wakeword names must be unique")

    # ---- safetensors half ---------------------------------------------------

    tensors: dict[str, np.ndarray] = {}

    if args.embedding is not None:
        embed_tensors = _load_onnx_initializer(args.embedding)
        for k, v in embed_tensors.items():
            # Prefix so the runtime binder can tell embedding weights
            # apart from classifier weights.
            tensors[f"openwakeword.embedding.{k}"] = v

    for idx, (name, path) in enumerate(wakeword_paths):
        cls_tensors = _load_onnx_initializer(path)
        # openWakeWord classifier ONNX graphs typically carry two Gemm
        # layers whose weights show up as `.weight` / `.bias`
        # initializers (naming varies per release). To make the merged
        # safetensors self-descriptive for the runtime binder we look
        # for the two matmul-shaped weights and their bias siblings and
        # rename them to the runtime's expected key convention.
        weights: list[tuple[str, np.ndarray]] = sorted(
            (
                (k, v)
                for k, v in cls_tensors.items()
                if v.ndim == 2 and min(v.shape) > 0
            ),
            key=lambda kv: -kv[1].size,  # Widest matmul first (linear1).
        )
        biases: list[tuple[str, np.ndarray]] = sorted(
            ((k, v) for k, v in cls_tensors.items() if v.ndim == 1),
            key=lambda kv: -kv[1].size,  # Widest bias first (linear1_bias).
        )
        if len(weights) < 2 or len(biases) < 2:
            raise SystemExit(
                f"{path}: expected at least 2 Gemm weight tensors and 2 bias "
                f"tensors in the classifier ONNX (got {len(weights)} + "
                f"{len(biases)}). The bridge cannot infer the linear1/linear2 "
                "split — re-check the release."
            )
        # linear1_weight is the widest (hidden × embedding), linear2_weight
        # is (1 × hidden). Bias order matches.
        (l1_w_name, l1_w) = weights[0]
        (l2_w_name, l2_w) = weights[1]
        (l1_b_name, l1_b) = biases[0]
        (l2_b_name, l2_b) = biases[1]
        # openwakeword ONNX Gemm layers store weight as `[out, in]` which
        # matches the runtime's row-major `[hidden_dim, embedding_dim]`
        # convention directly — no transpose needed.
        tensors[f"openwakeword.classifier.{idx}.linear1.weight"] = l1_w
        tensors[f"openwakeword.classifier.{idx}.linear1.bias"] = l1_b
        tensors[f"openwakeword.classifier.{idx}.linear2.weight"] = l2_w
        tensors[f"openwakeword.classifier.{idx}.linear2.bias"] = l2_b
        print(
            f"[{name}] classifier bound: linear1={l1_w.shape} "
            f"(from `{l1_w_name}`), linear2={l2_w.shape} "
            f"(from `{l2_w_name}`)",
            file=sys.stderr,
        )

    from safetensors.numpy import save_file

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st))
    print(f"wrote {args.output_st} ({len(tensors)} tensors)", file=sys.stderr)

    # ---- config side-car half -----------------------------------------------
    #
    # `wakeword_names` only. The four front-end axes (window_frames /
    # mel_bins / sample_rate / hop_samples) are deliberately omitted: this
    # script has no primary source for them, and the converter carries
    # documented mirrors of the runtime binder's constants that apply when
    # a key is absent. Emitting them here would be a third copy of the
    # same numbers, free to drift from both.
    #
    # The ordering below is the SAME ordering used for the positional
    # `openwakeword.classifier.{idx}.*` tensor names above, which is what
    # lets the converter pair label i with classifier group i.
    config: dict[str, Any] = {
        "wakeword_names": [name for name, _ in wakeword_paths],
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

    hop_probs = _emit_reference_probs(args.embedding, wakeword_paths, samples)

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
