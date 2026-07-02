#!/usr/bin/env python3
"""Generate the raw f32 PCM fixtures for the C ABI smoke tests (M0-09-T11).

The C smoke tests (tests/capi/smoke_*.c) must not parse WAV or use locale
dependent parsers (strtod is forbidden, NFR-RL-01), so audio input is supplied
as raw little-endian float32 PCM they can read with a single fread. This script
derives those fixtures from the committed M0-05 / M0-06 parity assets — it does
not download or create any new model input.

Outputs (committed alongside this script):

  vad_input_16k.f32  <- tests/parity/silero_vad/test_16k.wav
                        (IEEE float32 mono 16 kHz; the leading frames are kept:
                        20 * 512 = 10240 samples, enough to exercise push/poll)
  asr_input_16k.f32  <- tests/parity/whisper_base/input_pcm.f32
                        (already raw f32 mono 16 kHz; copied verbatim)

Run from the repo root:  python3 tests/capi/fixtures/gen_fixtures.py
"""

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent

VAD_WAV = ROOT / "tests/parity/silero_vad/test_16k.wav"
ASR_PCM = ROOT / "tests/parity/whisper_base/input_pcm.f32"

VAD_OUT = FIX / "vad_input_16k.f32"
ASR_OUT = FIX / "asr_input_16k.f32"

# Keep 20 frames of 512 samples @ 16 kHz for the VAD fixture.
VAD_SAMPLES = 20 * 512


def read_wav_mono_f32(path: Path) -> tuple[list[float], int]:
    """Minimal RIFF/WAVE reader for mono int16 (fmt 1) or float32 (fmt 3)."""
    data = path.read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"error: {path} is not a RIFF/WAVE file")
    fmt = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            audio_format, channels, sample_rate = struct.unpack_from("<HHI", body, 0)
            (bits,) = struct.unpack_from("<H", body, 14)
            fmt = (audio_format, channels, sample_rate, bits)
        elif cid == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        sys.exit(f"error: {path} missing fmt/data chunk")
    audio_format, channels, sample_rate, bits = fmt
    if channels != 1:
        sys.exit(f"error: {path} must be mono, got {channels} channels")
    if audio_format == 3 and bits == 32:  # IEEE float
        samples = list(struct.unpack("<%df" % (len(payload) // 4), payload))
    elif audio_format == 1 and bits == 16:  # int16 PCM
        ints = struct.unpack("<%dh" % (len(payload) // 2), payload)
        samples = [s / 32768.0 for s in ints]
    else:
        sys.exit(f"error: {path} unsupported format {audio_format}/{bits}-bit")
    return samples, sample_rate


def main() -> None:
    samples, rate = read_wav_mono_f32(VAD_WAV)
    if rate != 16000:
        sys.exit(f"error: {VAD_WAV} must be 16 kHz, got {rate}")
    samples = samples[:VAD_SAMPLES]
    VAD_OUT.write_bytes(struct.pack("<%df" % len(samples), *samples))
    print(f"wrote {VAD_OUT.name}: {len(samples)} f32 samples")

    if not ASR_PCM.exists():
        sys.exit(f"error: {ASR_PCM} not found")
    raw = ASR_PCM.read_bytes()
    if len(raw) % 4 != 0:
        sys.exit(f"error: {ASR_PCM} is not a whole number of f32 samples")
    ASR_OUT.write_bytes(raw)
    print(f"wrote {ASR_OUT.name}: {len(raw) // 4} f32 samples (verbatim copy)")


if __name__ == "__main__":
    main()
