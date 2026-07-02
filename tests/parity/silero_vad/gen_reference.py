#!/usr/bin/env python3
"""Generate the Silero VAD v5 parity fixtures for M0-05 (NFR-QL-01 / NFR-QL-05).

Ground truth = onnxruntime running the upstream ``silero_vad.onnx``
(snakers4/silero-vad). Our GGUF weights are extracted from the *same* ONNX, so
ORT is a faithful oracle for the native Rust re-implementation.

This script writes, into this directory:

* ``test_16k.wav`` / ``test_8k.wav`` — deterministic float32 mono PCM (silence /
  noise bursts / tone), the shared input for the streaming parity test and the
  ``vad_demo`` example.
* ``probs_16k.txt`` / ``probs_8k.txt`` — the ORT speech probability of every
  fixed frame (512 samples @ 16 kHz, 256 @ 8 kHz), LSTM state carried across
  frames — the e2e streaming reference (the WP's mandatory deliverable).
* ``step_stftconv_<rate>.txt`` / ``step_mag_<rate>.txt`` /
  ``step_encoder_<rate>.txt`` — per-stage ground truth for the *first* frame
  (zero state), obtained by lifting the matching ``If`` branch into a standalone
  ONNX graph and exposing the internal tensors as outputs. These back the
  layer-by-layer parity tests (T04–T06).
* ``silero-vad-v5.gguf`` — the *corrected* GGUF the native model loads: it
  carries **both** sample-rate weight sets (``sr8k.*`` and ``sr16k.*``). The
  production ``vokra-convert`` currently emits only one rate (see README); this
  fixture is produced here directly from the ONNX until that is fixed.

Re-generate with::

    parity-venv/bin/python gen_reference.py path/to/silero_vad.onnx

Requires onnx / onnxruntime / numpy (see tests/parity/parity-requirements.txt).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent

# Fixed-window sizes: single Silero v5 model, 8 kHz -> 256, 16 kHz -> 512
# samples per frame (research 03 §3.1). Right-side reflection pad width per rate.
FRAME = {8000: 256, 16000: 512}
PAD = {8000: 32, 16000: 64}
PREFIX = {"then": "If_0_then_branch__Inline_0__",
          "else": "If_0_else_branch__Inline_0__"}
RATE_OF_BRANCH = {"then": 16000, "else": 8000}
PARAMS = [
    "stft.forward_basis_buffer",
    "encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias",
    "decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih", "decoder.rnn.bias_hh",
    "decoder.decoder.2.weight", "decoder.decoder.2.bias",
]


def synth(rate: int) -> np.ndarray:
    """Deterministic test signal: silence, band-ish noise bursts, a tone, silence.

    Content is irrelevant to parity (we compare to ORT on the same samples); it
    only needs to be fixed and to exercise a range of probabilities.
    """
    rng = np.random.default_rng(20260705)
    n = FRAME[rate] * 48  # 48 frames
    t = np.arange(n) / rate
    x = np.zeros(n, dtype=np.float64)
    seg = n // 4
    # 1: silence [0, seg)
    # 2: modulated noise (speech-like) [seg, 2seg)
    burst = rng.standard_normal(seg) * (0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t[seg:2 * seg])))
    x[seg:2 * seg] = burst
    # 3: 300 Hz + 900 Hz tones [2seg, 3seg)
    x[2 * seg:3 * seg] = 0.3 * np.sin(2 * np.pi * 300 * t[2 * seg:3 * seg]) \
        + 0.15 * np.sin(2 * np.pi * 900 * t[2 * seg:3 * seg])
    # 4: quieter noise [3seg, n)
    x[3 * seg:] = 0.05 * rng.standard_normal(n - 3 * seg)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def write_wav_f32(path: Path, rate: int, samples: np.ndarray) -> None:
    """Minimal mono float32 WAV (WAVE_FORMAT_IEEE_FLOAT = 3)."""
    data = np.ascontiguousarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    chunks = b"".join([
        b"fmt ", struct.pack("<I", len(fmt)), fmt,
        b"data", struct.pack("<I", len(data)), data,
    ])
    riff = b"WAVE" + chunks
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)


def write_floats(path: Path, arr: np.ndarray) -> None:
    """One float per line, full float32 precision (round-trippable via str::parse)."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    path.write_text("".join(f"{v:.9g}\n" for v in flat))


# ---------------------------------------------------------------------------
# Weight extraction (verbatim from the ONNX Constant nodes)
# ---------------------------------------------------------------------------
def extract_weights(model) -> dict[str, np.ndarray]:
    consts: dict[str, np.ndarray] = {}

    def walk(sg):
        for node in sg.node:
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value" and a.t.data_type in (1, 10):
                        consts[node.output[0]] = np.asarray(
                            numpy_helper.to_array(a.t), dtype=np.float32)
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
                for sub in a.graphs:
                    walk(sub)

    walk(model.graph)
    out: dict[str, np.ndarray] = {}
    for branch, pref in PREFIX.items():
        rate = RATE_OF_BRANCH[branch]
        tag = f"sr{rate // 1000}k"
        for p in PARAMS:
            key = pref + p
            if key not in consts:
                raise SystemExit(f"missing weight {key}")
            out[f"{tag}.{p}"] = consts[key]
    return out


# ---------------------------------------------------------------------------
# GGUF v3 writer (dependency-free; validated by the Rust GgufFile loader)
# ---------------------------------------------------------------------------
GGUF_ALIGN = 32
GGML_F32 = 0
GGUF_TYPE_STRING = 8


def _align(n: int) -> int:
    return (n + GGUF_ALIGN - 1) // GGUF_ALIGN * GGUF_ALIGN


def write_gguf(path: Path, tensors: dict[str, np.ndarray]) -> None:
    def gstr(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    meta = [("vokra.model.arch", "silero-vad"),
            ("vokra.model.name", "silero-vad-v5")]

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)                 # version
    header += struct.pack("<Q", len(tensors))      # tensor_count
    header += struct.pack("<Q", len(meta))         # metadata_kv_count
    for k, v in meta:
        header += gstr(k)
        header += struct.pack("<I", GGUF_TYPE_STRING)
        header += gstr(v)

    # Deterministic tensor order.
    names = sorted(tensors)
    payloads = {n: np.ascontiguousarray(tensors[n], dtype="<f4").tobytes() for n in names}

    # Assign 32-aligned offsets relative to the tensor-data region.
    offsets: dict[str, int] = {}
    cur = 0
    for n in names:
        offsets[n] = cur
        cur = _align(cur + len(payloads[n]))

    for n in names:
        arr = tensors[n]
        header += gstr(n)
        header += struct.pack("<I", arr.ndim)
        for d in arr.shape:                        # dims in numpy/ONNX order
            header += struct.pack("<Q", int(d))
        header += struct.pack("<I", GGML_F32)
        header += struct.pack("<Q", offsets[n])

    data_start = _align(len(header))
    buf = bytearray(header)
    buf += b"\x00" * (data_start - len(header))
    for n in names:
        want = data_start + offsets[n]
        buf += b"\x00" * (want - len(buf))
        buf += payloads[n]
    path.write_bytes(bytes(buf))


# ---------------------------------------------------------------------------
# Lift one If branch into a standalone model exposing internal tensors
# ---------------------------------------------------------------------------
def lift_branch(model, branch: str, want: dict[str, str]) -> ort.InferenceSession:
    sg = None
    for node in model.graph.node:
        if node.op_type == "If":
            for a in node.attribute:
                if a.name == f"{branch}_branch":
                    sg = a.g
    assert sg is not None
    produced = set()
    for node in sg.node:
        produced.update(node.output)
    consumed = set()
    for node in sg.node:
        consumed.update(i for i in node.input if i)
    external = sorted(consumed - produced)
    ins = []
    for name in external:
        if name == "input":
            ins.append(helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, None]))
        elif name == "state":
            ins.append(helper.make_tensor_value_info("state", TensorProto.FLOAT, [2, 1, 128]))
        elif name == "sr":
            ins.append(helper.make_tensor_value_info("sr", TensorProto.INT64, []))
        else:
            ins.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    outs = [helper.make_tensor_value_info(v, TensorProto.FLOAT, None) for v in want.values()]
    g = helper.make_graph(list(sg.node), f"{branch}_lifted", ins, outs,
                          initializer=list(sg.initializer))
    lifted = helper.make_model(g, opset_imports=[helper.make_opsetid("", 16)])
    lifted.ir_version = model.ir_version
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(lifted.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    return sess, external, list(want.values())


def main() -> None:
    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "silero_vad.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"upstream silero_vad.onnx not found: {onnx_path}\n"
                         f"Pass its path as argv[1] (see README).")
    model = onnx.load(str(onnx_path))

    # 1) corrected both-rate GGUF
    weights = extract_weights(model)
    write_gguf(HERE / "silero-vad-v5.gguf", weights)
    print(f"wrote silero-vad-v5.gguf ({len(weights)} tensors)")

    # 2) full-model streaming reference + WAV, per rate
    so = ort.SessionOptions()
    so.log_severity_level = 3
    full = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    for rate in (16000, 8000):
        pcm = synth(rate)
        write_wav_f32(HERE / f"test_{rate // 1000}k.wav", rate, pcm)
        fw = FRAME[rate]
        state = np.zeros((2, 1, 128), dtype=np.float32)
        probs = []
        for i in range(len(pcm) // fw):
            frame = pcm[i * fw:(i + 1) * fw][None, :]
            out, state = full.run(None, {"input": frame, "state": state,
                                         "sr": np.array(rate, np.int64)})
            probs.append(float(out.reshape(-1)[0]))
        write_floats(HERE / f"probs_{rate // 1000}k.txt", np.array(probs, np.float32))
        print(f"rate={rate}: {len(probs)} frames, prob range "
              f"[{min(probs):.4f}, {max(probs):.4f}]")

    # 3) per-stage ground truth for the first frame (zero state)
    want = {
        "stftconv": None,  # filled per branch below
        "magnitude": None,
        "encoder": None,
    }
    for branch, rate in RATE_OF_BRANCH.items():
        pref = PREFIX[branch]
        names = {
            "stftconv": pref + "/stft/Conv_output_0",
            "magnitude": pref + "/stft/Sqrt_output_0",
            "encoder": pref + "/encoder/3/activation/Relu_output_0",
        }
        sess, external, order = lift_branch(model, branch, names)
        pcm = synth(rate)
        frame = pcm[:FRAME[rate]][None, :]
        feeds = {"input": frame, "state": np.zeros((2, 1, 128), np.float32),
                 "sr": np.array(rate, np.int64)}
        feeds = {k: v for k, v in feeds.items() if k in external}
        res = sess.run(order, feeds)
        tag = f"{rate // 1000}k"
        for key, arr in zip(order, res):
            stage = [s for s, nm in names.items() if nm == key][0]
            write_floats(HERE / f"step_{stage}_{tag}.txt", arr)
        print(f"rate={rate}: wrote step_stftconv/step_magnitude/step_encoder_{tag}.txt")


if __name__ == "__main__":
    main()
