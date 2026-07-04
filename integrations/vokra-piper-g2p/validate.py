#!/usr/bin/env python3
"""Offline validation for the real-G2P bridge (task C).

Proves two things that together establish "Vokra + real G2P reproduces
piper-plus's own inference for the same text":

  (A) **id-map wiring** — the encoder id map this bridge reconstructs by
      inverting the voice GGUF's `vokra.piper.phoneme_symbols` is *byte-identical*
      to the checkpoint's `config.json` `phoneme_id_map`, and the GGUF language
      order matches `language_id_map`. Since the bridge runs the *unmodified*
      `piper-plus-g2p` crate on top of that map, its `(ids, prosody, lid)` output
      equals piper-plus's by construction.

  (B) **end-to-end model parity** — feeding the bridge's real `(ids, prosody,
      lid)` for actual sentences to onnxruntime v7 reproduces the WAV the Vokra
      binary wrote (native path), within the FP16 reference tolerance. This
      re-confirms the native model consumes the real, non-zero-prosody G2P output
      exactly like the reference runtime — on real text, not a synthetic buffer.

OFFLINE tool (onnxruntime / numpy) — not part of the runtime. Needs the model
files (not committed): the converted voice GGUF, its `config.json`, and the v7
ONNX.

Usage:
    python3 validate.py --voice voice.gguf --config config.json \
        --onnx voice.onnx --bin target/release/vokra-piper-g2p
"""
import argparse
import ast
import struct
import subprocess
import sys

import numpy as np
import onnxruntime as ort


# --- minimal GGUF metadata reader (mirrors scratchpad/gguf_dump.py) ----------
def read_gguf_meta(path):
    d = open(path, "rb").read(8_000_000)
    assert d[:4] == b"GGUF", "not a GGUF file"
    off = [4]

    def u32():
        v = struct.unpack_from("<I", d, off[0])[0]; off[0] += 4; return v

    def u64():
        v = struct.unpack_from("<Q", d, off[0])[0]; off[0] += 8; return v

    def s():
        n = u64(); v = d[off[0]:off[0] + n].decode("utf8", "replace"); off[0] += n; return v

    def val(t):
        if t == 4: return u32()
        if t == 5: v = struct.unpack_from("<i", d, off[0])[0]; off[0] += 4; return v
        if t == 6: v = struct.unpack_from("<f", d, off[0])[0]; off[0] += 4; return v
        if t == 7: v = d[off[0]]; off[0] += 1; return bool(v)
        if t == 8: return s()
        if t == 10: return u64()
        if t == 11: v = struct.unpack_from("<q", d, off[0])[0]; off[0] += 8; return v
        if t == 12: v = struct.unpack_from("<d", d, off[0])[0]; off[0] += 8; return v
        if t == 0: v = d[off[0]]; off[0] += 1; return v
        if t == 1: v = struct.unpack_from("<b", d, off[0])[0]; off[0] += 1; return v
        if t == 2: v = struct.unpack_from("<H", d, off[0])[0]; off[0] += 2; return v
        if t == 3: v = struct.unpack_from("<h", d, off[0])[0]; off[0] += 2; return v
        if t == 9:
            et = u32(); n = u64(); return [val(et) for _ in range(n)]
        raise ValueError(f"gguf type {t}")

    ver = u32(); _tcount = u64(); kvcount = u64()  # noqa: F841
    meta = {}
    for _ in range(kvcount):
        k = s(); t = u32(); meta[k] = val(t)
    return meta


INTERMEDIATE_OUT = "output"  # graph pcm output name


def onnx_pcm(sess, ids, prosody, lid):
    seq = len(ids)
    pros = np.array(prosody, dtype=np.int64).reshape(1, seq, 3)
    feed = {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([seq], dtype=np.int64),
        "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # deterministic
        "speaker_embedding": np.zeros((1, 192), dtype=np.float32),
        "lid": np.array([lid], dtype=np.int64),
        "prosody_features": pros,
    }
    return sess.run([INTERMEDIATE_OUT], feed)[0].ravel().astype(np.float32)


def read_wav_mono_f32(path):
    b = open(path, "rb").read()
    # minimal PCM16 mono reader (matches the binary's writer)
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    p = 12
    data = None
    while p + 8 <= len(b):
        cid = b[p:p + 4]; sz = struct.unpack_from("<I", b, p + 4)[0]; body = p + 8
        if cid == b"data":
            data = np.frombuffer(b[body:body + sz], dtype="<i2").astype(np.float32) / 32768.0
        p = body + sz + (sz & 1)
    assert data is not None
    return data


def dump_g2p(binary, voice, text, lang):
    out = subprocess.run(
        [binary, "--voice", voice, "--text", text, "--lang", lang, "--dump"],
        capture_output=True, text=True, check=True,
    ).stdout
    ids = prosody = lid = None
    for line in out.splitlines():
        if line.startswith("ids="): ids = ast.literal_eval(line[4:])
        elif line.startswith("lid="): lid = int(line[4:])
        elif line.startswith("prosody="): prosody = ast.literal_eval(line[8:])
    return ids, prosody, lid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bin", required=True)
    args = ap.parse_args()

    import json
    cfg = json.load(open(args.config))
    id_map = cfg["phoneme_id_map"]          # symbol -> [id]
    lang_id_map = cfg["language_id_map"]     # code -> lid
    meta = read_gguf_meta(args.voice)
    symbols = meta["vokra.piper.phoneme_symbols"]   # id -> symbol
    lang_codes = meta["vokra.piper.language_codes"]  # lid -> code

    ok = True

    # (A) id-map wiring: invert GGUF symbols, compare to config phoneme_id_map.
    inverted = {}
    for i, sym in enumerate(symbols):
        if sym:
            inverted.setdefault(sym, [i])
    missing = {s: v for s, v in id_map.items() if inverted.get(s) != v}
    extra = {s: v for s, v in inverted.items() if id_map.get(s) != v}
    if missing or extra:
        ok = False
        print(f"[A] FAIL id-map: {len(missing)} mismatched config keys, {len(extra)} extra")
        for s, v in list(missing.items())[:5]:
            print(f"      config[{s!r}]={v}  inverted={inverted.get(s)}")
    else:
        print(f"[A] OK  id-map wiring: inverted GGUF symbols == config phoneme_id_map "
              f"({len(id_map)} symbols, byte-identical)")

    # language order
    lang_by_id = [c for c, _ in sorted(lang_id_map.items(), key=lambda kv: kv[1])]
    if lang_codes != lang_by_id:
        ok = False
        print(f"[A] FAIL language order: gguf {lang_codes} != config {lang_by_id}")
    else:
        print(f"[A] OK  language order: {lang_codes} (lid indices match config)")

    # (B) end-to-end model parity on real sentences.
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    id2sym = {v[0]: k for k, v in id_map.items()}
    cases = [("こんにちは、今日はいい天気ですね。", "ja"),
             ("Hello, this is a test of the Vokra engine.", "en")]
    for text, lang in cases:
        ids, prosody, lid = dump_g2p(args.bin, args.voice, text, lang)
        phones = "".join(id2sym.get(i, "·") for i in ids if i not in (0,))
        nz = sum(1 for p in prosody if p != [0, 0, 0])
        # native WAV
        wav = args.voice + f".{lang}.native.wav"
        subprocess.run([args.bin, "--voice", args.voice, "--text", text,
                        "--lang", lang, "--out", wav], capture_output=True, check=True)
        native = read_wav_mono_f32(wav)
        ref = onnx_pcm(sess, ids, prosody, lid)
        n = min(len(native), len(ref))
        if len(native) != len(ref):
            ok = False
            print(f"[B] FAIL {lang}: length native={len(native)} onnx={len(ref)}")
            continue
        d = float(np.abs(native - ref).max())
        corr = float(np.corrcoef(native, ref)[0, 1])
        status = "OK " if (d <= 0.05 and corr >= 0.999) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[B] {status} {lang}: {len(ids)} ids lid={lid} nz_prosody={nz} "
              f"| native vs onnxruntime max|Δ|={d:.6f} corr={corr:.6f}")
        print(f"        phones: {phones[:70]}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
