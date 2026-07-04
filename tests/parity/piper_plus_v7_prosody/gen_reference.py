#!/usr/bin/env python3
"""Generate the piper-plus v7 **non-zero prosody** reference fixtures.

Companion to `../piper_plus_v7/gen_reference.py`. That suite feeds
`prosody_features = zeros`, so the JA prosody projection contributes only its
**bias** to the decoder conditioning — `ProsodyProj.weight` is never exercised
(`weight @ 0 == 0`). This suite feeds a **non-zero** per-phoneme prosody buffer
with `lid = 0` (Japanese — the only language the v7 graph gates prosody on,
`Equal(lid, 0)`), so the reference `dec_input` / `pcm` now depend on
`weight @ features`. A native run that matches these therefore proves the full
`ProsodyProj` path (weight + gate + bias), not just the bias.

OFFLINE reference generator (onnxruntime) — NOT part of the runtime. Vokra stays
zero-external-dependency; onnx / onnxruntime / numpy are used here only to dump
committed reference tensors.

Determinism: identical to the sibling suite — scales = [0, 1, 0] makes the ONNX
fully deterministic. The prosody buffer is a fixed, checked-in integer pattern
(also written to `prosody.i64` so the Rust test feeds byte-identical values).

Usage:
    python3 gen_reference.py path/to/v7-epoch32-zs.onnx  [out_dir=.]
"""
import hashlib
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

# Same intermediates the sibling suite exposes; dec_input carries the prosody
# contribution into the decoder, pcm is the end-to-end audio.
INTERMEDIATES = {
    "g": "/Add_output_0",
    "m_p": "/enc_p/Split_output_0",
    "logs_p": "/enc_p/Split_output_1",
    "flow_z": "/flow/flows.0/Concat_output_0",
    "dec_input": "/Mul_9_output_0",
    "sdp_body": "/dp/proj/Conv_output_0",
}


def prosody_pattern(t: int) -> np.ndarray:
    """Fixed non-zero (A1, A2, A3) per phoneme. Deterministic, varied per slot
    and per channel so every column of `ProsodyProj.weight` is hit by a distinct
    integer — a zero column or a swapped channel would change `dec_input`."""
    rows = []
    for i in range(t):
        a1 = i - 5           # -5 .. t-6  (signed: accent-nucleus distance sign)
        a2 = (i % 5) + 1     # 1 .. 5
        a3 = (i * 2) % 7     # 0 .. 6
        rows.append([a1, a2, a3])
    return np.array(rows, dtype=np.int64)


def main() -> None:
    onnx_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    for name in INTERMEDIATES.values():
        if name not in existing:
            vi = onnx.helper.ValueInfoProto()
            vi.name = name
            model.graph.output.append(vi)
    tmp = os.path.join(out_dir, "_dbg.onnx")
    onnx.save(model, tmp)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])

    t = 12
    ids = np.array([[1] + list(range(2, 2 + t)) + [0]], dtype=np.int64)  # T = 14
    seq = ids.shape[1]
    prosody = prosody_pattern(seq)  # (14, 3), non-zero
    feed = {
        "input": ids,
        "input_lengths": np.array([seq], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # deterministic
        "speaker_embedding": np.zeros((1, 192), dtype=np.float32),
        "lid": np.array([0], dtype=np.int64),  # JA — prosody gate ON
        "prosody_features": prosody.reshape(1, seq, 3),
    }
    names = ["output", "durations"] + list(INTERMEDIATES.values())
    res = sess.run(names, feed)
    out = {"pcm": res[0], "durations": res[1]}
    for i, key in enumerate(INTERMEDIATES):
        out[key] = res[2 + i]
    os.remove(tmp)

    # The exact prosody buffer, little-endian i64, so the Rust test is byte-exact.
    prosody.astype("<i8").tofile(os.path.join(out_dir, "prosody.i64"))

    with open(os.path.join(out_dir, "manifest.txt"), "w") as man:
        man.write("repo=ayousanz/piper-plus-zero-shot-multi-6lang-v7 file=v7-epoch32-zs.onnx\n")
        man.write("phoneme_ids=%s lid=0 scales=[0,1,0] spk_emb=zeros192\n" % ids.tolist())
        man.write("prosody=NON-ZERO pattern a1=i-5 a2=i%%5+1 a3=2i%%7 (see prosody.i64)\n")
        man.write("prosody_rows=%s\n" % prosody.tolist())
        man.write("sr=22050 gin=512 t_phonemes=%d\n" % seq)
        for key in ["pcm", "durations", "m_p", "logs_p", "flow_z", "dec_input", "sdp_body", "g"]:
            arr = np.ascontiguousarray(out[key].ravel().astype("<f4"))
            arr.tofile(os.path.join(out_dir, key + ".f32"))
            man.write("%s shape=%s len=%d sha256=%s\n" % (
                key, list(out[key].shape), arr.size, hashlib.sha256(arr.tobytes()).hexdigest()[:16]))
    print("wrote non-zero-prosody fixtures to", out_dir)


if __name__ == "__main__":
    main()
