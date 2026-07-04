#!/usr/bin/env python3
"""Generate the piper-plus v7 reference fixtures for the gated Rust parity test
(crates/vokra-models/src/piper_plus/parity_v7.rs).

OFFLINE reference generator (onnxruntime) — NOT part of the runtime. The Vokra
runtime stays zero-external-dependency; onnx / onnxruntime / numpy are used here
only to dump committed reference tensors, exactly like the other tests/parity
suites use torch / onnxruntime.

Source model (private HF repo, HF_TOKEN required):
    ayousanz/piper-plus-zero-shot-multi-6lang-v7  ->  v7-epoch32-zs.onnx
A zero-shot multi-speaker(571) / multi-lang(6) / FiLM MB-iSTFT-VITS2 model.

Determinism: with scales=[noise_scale=0, length_scale=1, noise_w=0] the ONNX is
fully deterministic (two runs are bit-identical), so these fixtures are an exact
reference. The intermediates are exposed by appending their node-output names to
the graph outputs before running onnxruntime.

Usage:
    python3 gen_reference.py path/to/v7-epoch32-zs.onnx  [out_dir=.]

The fixed input (also recorded in manifest.txt): phoneme ids [1,2,...,13,0]
(T=14), lid=0, speaker_embedding=zeros(192) (zero-shot fallback; spk_proj(0)!=0),
prosody_features=zeros. Outputs: pcm, durations, and intermediates
g / m_p / logs_p / flow_z / dec_input / sdp_body, all as little-endian f32.
"""
import hashlib
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

# node-output names to expose as graph outputs (verified against the v7 graph)
INTERMEDIATES = {
    "g": "/Add_output_0",
    "m_p": "/enc_p/Split_output_0",
    "logs_p": "/enc_p/Split_output_1",
    "flow_z": "/flow/flows.0/Concat_output_0",
    "dec_input": "/Mul_9_output_0",
    "sdp_body": "/dp/proj/Conv_output_0",
}


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
    feed = {
        "input": np.array([[1] + list(range(2, 2 + t)) + [0]], dtype=np.int64),
        "input_lengths": np.array([t + 2], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # deterministic
        "speaker_embedding": np.zeros((1, 192), dtype=np.float32),
        "lid": np.array([0], dtype=np.int64),
        "prosody_features": np.zeros((1, t + 2, 3), dtype=np.int64),
    }
    names = ["output", "durations"] + list(INTERMEDIATES.values())
    res = sess.run(names, feed)
    out = {"pcm": res[0], "durations": res[1]}
    for i, key in enumerate(INTERMEDIATES):
        out[key] = res[2 + i]
    os.remove(tmp)

    with open(os.path.join(out_dir, "manifest.txt"), "w") as man:
        man.write("repo=ayousanz/piper-plus-zero-shot-multi-6lang-v7 file=v7-epoch32-zs.onnx\n")
        man.write("phoneme_ids=%s lid=0 scales=[0,1,0] spk_emb=zeros192 prosody=zeros\n" % feed["input"].tolist())
        man.write("sr=22050 gin=512 t_phonemes=%d\n" % (t + 2))
        for key in ["pcm", "durations", "m_p", "logs_p", "flow_z", "dec_input", "sdp_body", "g"]:
            arr = np.ascontiguousarray(out[key].ravel().astype("<f4"))
            arr.tofile(os.path.join(out_dir, key + ".f32"))
            man.write("%s shape=%s len=%d sha256=%s\n" % (
                key, list(out[key].shape), arr.size, hashlib.sha256(arr.tobytes()).hexdigest()[:16]))
    print("wrote fixtures to", out_dir)


if __name__ == "__main__":
    main()
