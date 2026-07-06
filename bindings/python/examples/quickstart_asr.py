#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ASR quickstart — transcribe mono ``f32`` PCM with Vokra.

This example is the minimal end-to-end ASR path exposed by the Python
binding: open a Whisper GGUF, feed it mono ``f32`` samples, get a
``str`` transcript. It doubles as the NFR-MT-07 doc-example CI target
(``python -m py_compile`` on every CI run), so keep it importable
without any prebuilt native lib on ``$PYTHONPATH``.

Design notes
------------
* Uses :class:`vokra.Session` as a context manager. ``__exit__`` calls
  ``vokra_session_destroy`` exactly once even if ``transcribe`` raises —
  the RAII contract that :mod:`vokra.session` documents. Callers who
  need explicit lifetime control can use ``session.close()`` instead.
* ``transcribe(pcm, sample_rate=...)`` takes the sample rate as a
  keyword-only int and returns a UTF-8 ``str``. Failures raise a
  :class:`vokra.VokraError` subclass — we never silently downgrade to
  CPU, retry, or return a fallback string (FR-EX-08).
* Reads a 16-bit little-endian WAV via the stdlib ``wave`` module, then
  converts to mono ``f32`` in ``[-1, 1]`` via a plain list comprehension.
  ``numpy`` is *not* imported — the binding is numpy-optional, and this
  quickstart is meant to be the "no extra deps" reference. If you have
  numpy handy, ``pcm = (np.frombuffer(raw, dtype=np.int16).astype(
  np.float32) / 32768.0).tolist()`` is a faster equivalent.
* Locale-independent numeric marshalling: no ``float(str)`` calls, no
  ``strtod``. Sample rate is a raw ``int`` from ``wave.getframerate()``
  and PCM is scaled by an integer literal ``32768`` — safe under
  ``LC_NUMERIC=de_DE.UTF-8`` (NFR-RL-01).

Usage
-----
::

    python quickstart_asr.py path/to/whisper-base.gguf path/to/audio.wav

The WAV must be 16-bit PCM, little-endian. Multi-channel input is
downmixed to mono by averaging channels; sample rate is passed
through verbatim — the runtime returns
``VOKRA_ERROR_INVALID_ARGUMENT`` if it does not match the model's
front-end rate (M0 does not resample; use ``vokra-cli convert`` or
``sox``/``ffmpeg`` upstream if you need 16 kHz).
"""

from __future__ import annotations

import sys
import wave
from typing import List, Tuple


def load_wav_mono_f32(path: str) -> Tuple[List[float], int]:
    """Load a 16-bit PCM WAV and return ``(mono_f32_samples, sample_rate)``.

    Multi-channel input is averaged into a single mono channel. Only
    ``sampwidth == 2`` (16-bit little-endian) is supported here; other
    widths are rejected with a clear ``ValueError`` rather than silently
    reinterpreted (the C ABI itself would happily accept whatever ``f32``
    we hand it, so we validate on the Python side to keep the failure
    mode readable).
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(
            f"quickstart_asr: expected 16-bit PCM (sampwidth=2), "
            f"got sampwidth={sampwidth} in {path!r}"
        )

    # Interpret ``raw`` as signed 16-bit little-endian samples. Using
    # ``int.from_bytes`` per-frame would be legible but painfully slow;
    # ``memoryview.cast('h')`` gives us zero-copy int16 access from the
    # stdlib with no extra dependency.
    mv = memoryview(raw).cast("h")  # signed 16-bit, native (LE on x86/arm)
    total_samples = len(mv)
    expected = n_frames * n_channels
    if total_samples != expected:
        raise ValueError(
            f"quickstart_asr: WAV frame count mismatch "
            f"({total_samples} samples vs {expected} expected)"
        )

    inv = 1.0 / 32768.0
    if n_channels == 1:
        pcm = [s * inv for s in mv]
    else:
        # Average channels frame-by-frame. This is a plain-Python loop
        # rather than a comprehension so the intent is obvious; if this
        # becomes a bottleneck the user should reach for numpy.
        pcm = []
        for frame_idx in range(n_frames):
            start = frame_idx * n_channels
            acc = 0
            for c in range(n_channels):
                acc += mv[start + c]
            pcm.append((acc / n_channels) * inv)

    return pcm, sample_rate


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: quickstart_asr.py <model.gguf> <audio.wav>",
            file=sys.stderr,
        )
        return 2

    model_path, wav_path = argv[1], argv[2]

    # Import inside ``main`` so ``python -m py_compile`` succeeds without
    # the native library being loadable (NFR-MT-07 doc-example gate). The
    # module-level ``import vokra`` in :mod:`vokra` itself does not touch
    # the CDLL — it only exposes metadata until ``Session.open`` is called.
    import vokra

    pcm, sample_rate = load_wav_mono_f32(wav_path)

    # ``Session.open`` loads the GGUF and returns a live handle. The
    # ``with`` block guarantees ``vokra_session_destroy`` runs exactly
    # once, even if ``transcribe`` raises a :class:`vokra.VokraError`.
    with vokra.Session.open(model_path) as session:
        transcript = session.transcribe(pcm, sample_rate=sample_rate)

    # Write the transcript to stdout unadorned so shell pipelines
    # (``| jq``, ``| tee``) get exactly what the model produced. Errors
    # would have raised before reaching this line.
    print(transcript)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
