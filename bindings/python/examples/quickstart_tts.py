#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTS quickstart — synthesize speech from UTF-8 text with Vokra.

This example is the minimal end-to-end TTS path exposed by the Python
binding: open a piper-plus GGUF voice, hand it a UTF-8 string, get back
mono ``f32`` PCM plus the model's output sample rate. It doubles as the
NFR-MT-07 doc-example CI target (``python -m py_compile`` on every CI
run), so it must remain importable without the prebuilt native library
on ``$PYTHONPATH`` — every FFI-touching import stays inside ``main``.

Design notes
------------
* Uses :class:`vokra.Session` as a context manager. ``__exit__`` calls
  ``vokra_session_destroy`` exactly once even if ``synthesize`` raises —
  the RAII contract that :mod:`vokra.session` documents. Callers who
  need explicit lifetime control can use ``session.close()`` instead.
* ``synthesize(text)`` takes UTF-8 text and returns
  ``(pcm: list[float], sample_rate: int)`` per :meth:`Session.synthesize`.
  Failures raise a :class:`vokra.VokraError` subclass — we never silently
  downgrade to CPU, retry, or emit fallback silence (FR-EX-08).
* Writes a 16-bit little-endian PCM WAV via the stdlib ``wave`` module.
  ``numpy`` is *not* imported — the binding is numpy-optional and this
  quickstart is the "no extra deps" reference. If you have numpy handy,
  ``np.clip(np.asarray(pcm) * 32767.0, -32768, 32767).astype('<i2')
  .tobytes()`` is a faster equivalent to the loop below.
* Locale-independent numeric marshalling: no ``float(str)`` calls, no
  ``strtod``. Sample rate flows through as an ``int`` from
  :meth:`Session.synthesize`, and PCM is scaled by an integer literal
  ``32767`` — safe under ``LC_NUMERIC=de_DE.UTF-8`` (NFR-RL-01).

Usage
-----
::

    python quickstart_tts.py path/to/piper-voice.gguf "Hello, world" out.wav

The output file is a 16-bit mono PCM WAV at the model's native sample
rate (piper-plus voices commonly emit 22050 Hz; the runtime reports the
actual rate — we never hard-code it). No resampling is applied on the
Python side; if you need a specific rate, run ``sox``/``ffmpeg`` on the
resulting WAV.
"""

from __future__ import annotations

import sys
import wave
from typing import Iterable, List


def _f32_to_s16_le_bytes(pcm: Iterable[float]) -> bytes:
    """Convert mono ``f32`` samples in ``[-1, 1]`` to 16-bit LE PCM bytes.

    Out-of-range samples are hard-clipped to the int16 range rather than
    wrapped — silent clipping matches what every audio-processing pipeline
    does at the DAC boundary, and wrapping would produce audible glitches.
    We build a ``bytearray`` and let ``struct``-style packing happen via
    ``int.to_bytes`` per-sample; this is slower than numpy but keeps the
    dependency footprint at zero, matching the ASR quickstart's ethos.
    """
    out = bytearray()
    # Local aliases shave a lookup per sample in the hot loop — this
    # matters for multi-second clips even in a "quickstart" script.
    _min, _max = -32768, 32767
    append = out.extend
    for s in pcm:
        # Scale by 32767 (not 32768) to keep +1.0 exactly representable
        # without overflow; the asymmetry of two's complement means
        # -1.0 * 32767 = -32767 still fits, and we clip anything below.
        v = int(s * 32767.0)
        if v > _max:
            v = _max
        elif v < _min:
            v = _min
        # ``signed=True`` handles the negative half without extra math.
        append(v.to_bytes(2, byteorder="little", signed=True))
    return bytes(out)


def write_wav_mono_s16(path: str, pcm: List[float], sample_rate: int) -> None:
    """Write mono ``f32`` PCM as a 16-bit little-endian WAV file.

    Only mono / 16-bit / little-endian is emitted — the stdlib ``wave``
    module hard-codes little-endian and we choose 16-bit as the least
    surprising interchange format. Callers who want float WAV should
    reach for ``soundfile`` (BSD-3) on top of this list.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(_f32_to_s16_le_bytes(pcm))


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print(
            'usage: quickstart_tts.py <voice.gguf> "<text>" <out.wav>',
            file=sys.stderr,
        )
        return 2

    voice_path, text, wav_path = argv[1], argv[2], argv[3]

    # Import inside ``main`` so ``python -m py_compile`` succeeds without
    # the native library being loadable (NFR-MT-07 doc-example gate). The
    # module-level ``import vokra`` in :mod:`vokra` itself does not touch
    # the CDLL — it only exposes metadata until ``Session.open`` is called.
    import vokra

    # ``Session.open`` loads the GGUF and returns a live handle. The
    # ``with`` block guarantees ``vokra_session_destroy`` runs exactly
    # once, even if ``synthesize`` raises a :class:`vokra.VokraError`.
    with vokra.Session.open(voice_path) as session:
        pcm, sample_rate = session.synthesize(text)

    # Refuse to write an empty WAV silently: the C ABI is allowed to
    # return zero samples for empty input, but writing a 44-byte header
    # with no data is more confusing than a loud error.
    if not pcm:
        print(
            f"quickstart_tts: synthesize returned 0 samples for text={text!r}",
            file=sys.stderr,
        )
        return 1

    write_wav_mono_s16(wav_path, pcm, sample_rate)

    # Report the shape on stderr so stdout stays free for shell pipelines
    # that may want to chain (``| xxd``, ``| play -``, etc.). Duration is
    # a rounded string, not ``locale.format_string`` — LC_NUMERIC-safe.
    duration_ms = int(len(pcm) * 1000 / sample_rate) if sample_rate > 0 else 0
    print(
        f"quickstart_tts: wrote {len(pcm)} samples "
        f"({duration_ms} ms @ {sample_rate} Hz) to {wav_path!r}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
