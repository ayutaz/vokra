#!/usr/bin/env python3
"""Generate MB-iSTFT-VITS2 (piper-plus) numerical-parity fixtures offline
(M0-07-T21).

Runs the distributed piper-plus voice through **onnxruntime** with the noise
scales zeroed (deterministic, docs/piper-plus-integration.md §5) and dumps the
final PCM, durations and the component-parity intermediates (enc_p m_p/logs_p
and the decoder-input latent z) as little-endian f32 files plus a manifest.

onnxruntime is used only here, offline — never in the runtime or CI
(FR-LD-05). Regenerate with:

    python gen_reference.py <voice.onnx> <config.json> <out_dir>

Fixtures are committed; the voice model itself is not (too large — the native
parity test is gated on $VOKRA_PIPER_GGUF, like Whisper's).
"""
import json
import struct
import sys

import numpy as np
import onnx
import onnxruntime as ort


# Internal graph tensors exposed as extra outputs for component parity
# (names verified against the distributed tsukuyomi graph).
INTERMEDIATES = {
    "m_p": "/enc_p/Split_output_0",
    "logs_p": "/enc_p/Split_output_1",
    "dec_input": "/Mul_9_output_0",  # z * y_mask, the decoder input
    "sdp_body": "/dp/proj/Conv_output_0",  # SDP body (proj output) feeding the flows
}

# A fixed phoneme-id sequence in piper multilingual framing
# (BOS ^=1, PAD _=0 interspersed, EOS $=2). Not real G2P output — a stable
# parity driver. JA-ish ids: k=32 o=14 N=25 n=57 i=11 t=38 w=63 a=10.
PHONEME_IDS = [1, 32, 0, 14, 0, 25, 0, 57, 0, 11, 0, 38, 0, 11, 0, 63, 0, 10, 0, 2]
LID = 0
NOISE_SCALE = 0.0
LENGTH_SCALE = 1.0
NOISE_W = 0.0


def write_f32(path, arr):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    return arr.size


def main():
    onnx_path, config_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(config_path) as f:
        config = json.load(f)
    piper_version = config.get("piper_version", "?")

    # Add the intermediate tensors as graph outputs.
    model = onnx.load(onnx_path)
    existing = {o.name for o in model.graph.output}
    for name in INTERMEDIATES.values():
        if name not in existing:
            model.graph.output.extend([onnx.helper.make_empty_tensor_value_info(name)])

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    t = len(PHONEME_IDS)
    feeds = {
        "input": np.array([PHONEME_IDS], dtype=np.int64),
        "input_lengths": np.array([t], dtype=np.int64),
        "scales": np.array([NOISE_SCALE, LENGTH_SCALE, NOISE_W], dtype=np.float32),
        "lid": np.array([LID], dtype=np.int64),
        "prosody_features": np.zeros((1, t, 3), dtype=np.int64),
        "speaker_embedding": np.zeros((1, 256), dtype=np.float32),
        "speaker_embedding_mask": np.zeros((1, 1), dtype=np.int64),
    }

    out_names = ["output", "durations"] + list(INTERMEDIATES.values())
    outputs = sess.run(out_names, feeds)
    named = dict(zip(out_names, outputs))

    pcm = named["output"].ravel()
    durations = named["durations"].ravel()
    m_p = named[INTERMEDIATES["m_p"]]
    logs_p = named[INTERMEDIATES["logs_p"]]
    dec_input = named[INTERMEDIATES["dec_input"]]

    n_pcm = write_f32(f"{out_dir}/pcm.f32", pcm)
    write_f32(f"{out_dir}/durations.f32", durations)
    write_f32(f"{out_dir}/m_p.f32", m_p)
    write_f32(f"{out_dir}/logs_p.f32", logs_p)
    write_f32(f"{out_dir}/dec_input.f32", dec_input)
    write_f32(f"{out_dir}/sdp_body.f32", named[INTERMEDIATES["sdp_body"]])

    # m_p / logs_p are [1, HIDDEN, T]; dec_input is [1, HIDDEN, T_frames].
    hidden = m_p.shape[1]
    t_frames = dec_input.shape[2]

    manifest = [
        "# piper-plus MB-iSTFT-VITS2 parity fixture (M0-07-T21)",
        "# Generated offline by gen_reference.py via onnxruntime (deterministic).",
        f"piper_version = {piper_version}",
        f"phoneme_ids = {' '.join(str(x) for x in PHONEME_IDS)}",
        f"lid = {LID}",
        f"noise_scale = {NOISE_SCALE}",
        f"length_scale = {LENGTH_SCALE}",
        f"noise_w = {NOISE_W}",
        f"hidden = {hidden}",
        f"t_phonemes = {t}",
        f"t_frames = {t_frames}",
        f"pcm_len = {n_pcm}",
        f"sample_rate = {config['audio']['sample_rate']}",
    ]
    with open(f"{out_dir}/manifest.txt", "w") as f:
        f.write("\n".join(manifest) + "\n")

    print(f"wrote fixtures to {out_dir}")
    print(f"  T_phonemes={t} T_frames={t_frames} pcm_len={n_pcm}")
    print(f"  durations={durations}")
    print(f"  m_p[:5]={m_p.ravel()[:5]}")
    print(f"  pcm range=[{pcm.min():.4f}, {pcm.max():.4f}] mean={pcm.mean():.5f}")


if __name__ == "__main__":
    main()
