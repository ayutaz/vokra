#!/usr/bin/env python3
"""cc-25: 8 kHz (ctx288) real-speech evaluation — reference side.

Produces, for each (clip x onnx-variant):
  * an 8 kHz PCM16 WAV derived from the 16 kHz source with a numpy-only
    Kaiser-windowed-sinc anti-aliasing decimator (the same filter family
    gen_reference.py will carry, so the committed fixture is reproducible
    without adding scipy to the parity tooling deps),
  * the ORT official-wrapper (ctx288) per-frame probabilities,
  * upstream get_speech_timestamps segments at default parameters.

scipy is used ONLY to cross-check the numpy filter and to measure spectra; it
is not on the fixture-generation path.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy import signal

HOME = Path.home()
W = HOME / ".cache/vokra-eval/weights/silero-vad"
CORPUS = HOME / ".cache/vokra-eval/corpus"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

FRAME8, CTX8 = 256, 32


# --------------------------------------------------------------------------
# WAV I/O (mono PCM16, byte-for-byte what the Rust reader sees)
# --------------------------------------------------------------------------
def read_wav_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    b = path.read_bytes()
    assert b[0:4] == b"RIFF" and b[8:12] == b"WAVE", f"{path}: not RIFF/WAVE"
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid = b[pos:pos + 4]
        size = struct.unpack_from("<I", b, pos + 4)[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHI", b, body)
        elif cid == b"data":
            data = b[body:body + size]
        pos = body + size + (size & 1)
    assert fmt is not None and data is not None and fmt[0] == 1 and fmt[1] == 1
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, fmt[2]


def write_wav_pcm16(path: Path, rate: int, x: np.ndarray) -> None:
    pcm = np.clip(np.round(np.asarray(x, np.float64) * 32768.0), -32768, 32767).astype("<i2")
    data = pcm.tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    chunks = b"".join([b"fmt ", struct.pack("<I", len(fmt)), fmt,
                       b"data", struct.pack("<I", len(data)), data])
    riff = b"WAVE" + chunks
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)


# --------------------------------------------------------------------------
# numpy-only anti-aliased decimation by 2 (the fixture-generation path)
# --------------------------------------------------------------------------
def halfband_decimate(x: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz with a linear-phase Kaiser-windowed-sinc lowpass.

    41-tap FIR, cutoff at 0.5 x Nyquist (= 4 kHz), Kaiser beta 5.0, DC-normalised
    — the same design scipy.signal.resample_poly uses by default for down=2.
    Symmetric zero-padding removes the filter's group delay, so the output is
    time-aligned with the input; only then is every 2nd sample kept. Filtering
    BEFORE decimation is what makes this an 8 kHz signal rather than an aliased
    artifact.
    """
    half = 20
    ntaps = 2 * half + 1
    m = np.arange(ntaps) - half
    h = np.sinc(0.5 * m) * np.kaiser(ntaps, 5.0)
    h /= h.sum()
    xp = np.concatenate([np.zeros(half), np.asarray(x, np.float64), np.zeros(half)])
    return np.convolve(xp, h, mode="valid")[::2]


def band_power(x: np.ndarray, rate: int, lo: float, hi: float) -> float:
    f, p = signal.periodogram(np.asarray(x, np.float64), fs=rate, window="hann",
                              nfft=1 << 15, scaling="spectrum")
    return float(p[(f >= lo) & (f < hi)].sum())


# --------------------------------------------------------------------------
# ORT official-wrapper (ctx288) streaming
# --------------------------------------------------------------------------
def ort_ctx_probs(sess, pcm: np.ndarray, rate: int, frame: int, ctx: int) -> list[float]:
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, ctx), dtype=np.float32)
    probs = []
    for i in range(len(pcm) // frame):
        f = pcm[i * frame:(i + 1) * frame][None, :]
        x = np.concatenate([context, f], axis=1)
        out, state = sess.run(None, {"input": x, "state": state,
                                     "sr": np.array(rate, np.int64)})
        context = x[..., -ctx:]
        probs.append(float(out.reshape(-1)[0]))
    return probs


def segments(probs, rate, frame, audio_len):
    """Upstream get_speech_timestamps at defaults (mirror of parity.rs replica)."""
    TH, NEG = 0.5, 0.35
    min_speech, min_sil, pad = rate * 250 // 1000, rate * 100 // 1000, rate * 30 // 1000
    triggered, start, temp_end, spans = False, 0, 0, []
    for i, p in enumerate(probs):
        cur = i * frame
        if p >= TH and temp_end != 0:
            temp_end = 0
        if p >= TH and not triggered:
            triggered, start = True, cur
            continue
        if p < NEG and triggered:
            if temp_end == 0:
                temp_end = cur
            if cur - temp_end >= min_sil:
                if temp_end - start > min_speech:
                    spans.append([start, temp_end])
                temp_end, triggered = 0, False
    if triggered and audio_len - start > min_speech:
        spans.append([start, audio_len])
    n = len(spans)
    for i in range(n):
        if i == 0:
            spans[i][0] = max(0, spans[i][0] - pad)
        if i + 1 < n:
            gap = spans[i + 1][0] - spans[i][1]
            if gap < 2 * pad:
                spans[i][1] += gap // 2
                spans[i + 1][0] -= gap // 2
            else:
                spans[i][1] = min(spans[i][1] + pad, audio_len)
                spans[i + 1][0] = max(0, spans[i + 1][0] - pad)
        else:
            spans[i][1] = min(spans[i][1] + pad, audio_len)
    return [tuple(s) for s in spans]


def main() -> None:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sessions = {
        "master": ort.InferenceSession(str(W / "silero_vad_master.onnx"), so,
                                       providers=["CPUExecutionProvider"]),
        "v5.0": ort.InferenceSession(str(W / "silero_vad.onnx"), so,
                                     providers=["CPUExecutionProvider"]),
    }

    clips = {
        "jfk": CORPUS / "jfk-30s.wav",
        "libri0002": CORPUS / "1272-128104-0002.wav",
    }

    report: dict = {}
    for tag, src in clips.items():
        x16, rate = read_wav_pcm16_mono(src)
        assert rate == 16000

        y = halfband_decimate(x16)
        dst = OUT / f"{tag}-8k.wav"
        write_wav_pcm16(dst, 8000, y)
        x8, r8 = read_wav_pcm16_mono(dst)
        assert r8 == 8000

        # Legitimacy evidence.
        naive = x16[::2]
        sp = signal.resample_poly(x16.astype(np.float64), 1, 2)
        e = {
            "src_share_above_4k_pct": 100 * band_power(x16, 16000, 4000, 8000)
            / (band_power(x16, 16000, 0, 4000) + band_power(x16, 16000, 4000, 8000)),
            "vs_scipy_resample_poly_max_abs": float(np.max(np.abs(y - sp))),
            "vs_naive_decimation_max_abs": float(np.max(np.abs(y - naive))),
            "alias_band_3k6_4k_reject_db": float(
                10 * np.log10(band_power(naive, 8000, 3600, 4000)
                              / band_power(x8, 8000, 3600, 4000))),
        }
        f16, p16 = signal.periodogram(x16.astype(np.float64), fs=16000, window="hann",
                                      nfft=1 << 15, scaling="spectrum")
        f8, p8 = signal.periodogram(x8.astype(np.float64), fs=8000, window="hann",
                                    nfft=1 << 14, scaling="spectrum")
        grid = np.linspace(50, 3500, 400)
        db = 10 * np.log10(np.interp(grid, f8, p8) / np.interp(grid, f16, p16))
        e["passband_50_3500_median_db"] = float(np.median(db))
        e["passband_50_3500_p05_db"] = float(np.percentile(db, 5))
        e["passband_50_3500_p95_db"] = float(np.percentile(db, 95))

        entry = {"source": str(src), "src_samples": int(len(x16)),
                 "wav8k": str(dst), "samples_8k": int(len(x8)),
                 "duration_s": len(x8) / 8000.0, "resampling": e, "ort": {}}

        for vtag, sess in sessions.items():
            probs = ort_ctx_probs(sess, x8, 8000, FRAME8, CTX8)
            (OUT / f"probs_{tag}_8k_ctx_{vtag}.txt").write_text(
                "".join(f"{np.float32(v):.9g}\n" for v in probs))
            segs = segments(probs, 8000, FRAME8, len(probs) * FRAME8)
            entry["ort"][vtag] = {
                "frames": len(probs), "max_prob": max(probs), "min_prob": min(probs),
                "segments": segs,
                "speech_s": sum(b - a for a, b in segs) / 8000.0,
            }
            print(f"[{tag}/{vtag}] frames={len(probs)} max={max(probs):.4f} "
                  f"segments={len(segs)} speech={entry['ort'][vtag]['speech_s']:.2f}s")

        # 16 kHz counterpart from the same source, master only, for cross-rate context.
        probs16 = ort_ctx_probs(sessions["master"], x16, 16000, 512, 64)
        s16 = segments(probs16, 16000, 512, len(probs16) * 512)
        entry["ort16_master"] = {
            "frames": len(probs16), "max_prob": max(probs16), "segments": s16,
            "speech_s": sum(b - a for a, b in s16) / 16000.0,
        }
        print(f"[{tag}/16k-master] frames={len(probs16)} max={max(probs16):.4f} "
              f"segments={len(s16)} speech={entry['ort16_master']['speech_s']:.2f}s")

        report[tag] = entry
        print(f"[{tag}] resampling evidence: {json.dumps(e, indent=None)}")
        print()

    (OUT / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {OUT/'reference.json'}")


if __name__ == "__main__":
    main()
