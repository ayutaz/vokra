# SPDX-License-Identifier: Apache-2.0
"""Session RAII wrapper bound to `vokra_session_create_from_file` /
`vokra_session_destroy` (T08) plus the high-level ``transcribe`` /
``synthesize`` façade (T10).

The `Session` class owns a `VokraSession` handle (opaque `void *`) and
guarantees:

* deterministic release via `__enter__` / `__exit__` (context manager),
* explicit release via `close()`,
* best-effort release via `__del__` when GC runs on an orphan instance,
* double-free rejection: once released, the handle is set to ``None`` and
  every subsequent operation (including a second ``close()`` from
  ``__del__`` after an explicit ``close()``) becomes a no-op — the native
  ``vokra_session_destroy`` is invoked at most once,
* use-after-free rejection: any operation that needs the raw handle goes
  through ``Handle.raw()`` which raises ``RuntimeError`` on a closed
  handle. That is the single choke-point between Python and the C ABI —
  wrappers built on top of ``Session`` (T10 `transcribe` / `synthesize`,
  T11 `Stream`) MUST route through ``raw()`` and never cache the pointer.

The C ABI itself lives in ``include/vokra.h``; `_bindings.PROTOTYPES`
declares the `argtypes`/`restype` this module relies on. Loading the
native library is delegated to ``_native.load_library()`` (T09) — this
module accepts the loaded CDLL as an injected argument so unit tests can
exercise the RAII contract with a mock library (no prebuilt binary
required at test time) while the higher-level ``Session.open(path)`` in
production paths uses the real one.

Design notes
------------
- The class inherits from ``_handles.Handle`` and overrides ``_release``
  to call ``lib.vokra_session_destroy(handle)``. All the double-free /
  UAF invariants are enforced by the base class; this file adds only the
  session-specific ``create_from_file`` factory and destroy wiring.
- ``__init__`` takes the ``ctypes.CDLL`` and the raw handle so tests can
  build a session around a mocked library. Callers that just want to
  open a file use :meth:`open`, which loads the real library on demand.
- ``transcribe`` / ``synthesize`` (T10) are the high-level ASR / TTS
  entry points. Every non-OK status from the C ABI is translated to a
  typed ``VokraError`` subclass and re-raised **verbatim** (FR-EX-08):
  we never silently retry on CPU, downgrade quality, or fall back to a
  dummy string — the wrapper is a thin marshalling layer, not a policy
  layer. Callers who want fallback behaviour must code it themselves on
  top of these primitives.
- Thread-safety: the C ABI documents "single-thread ownership" for
  sessions. This wrapper does not add locking; concurrent calls into the
  same ``Session`` from multiple threads are undefined behaviour, same
  as calling the C ABI directly. In particular, ``vokra_last_error()``
  is thread-local, so ``raise_from_status`` **must** be invoked in the
  same call frame as the failing FFI call — enforced by construction
  here (we capture the message immediately after the status check).
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

from ._bindings import bind
from ._handles import Handle
from .errors import raise_from_status

# Status code for success, mirrored locally so we do not import the whole
# enum tree from ``_bindings`` (which pulls every prototype). The value is
# fixed by the C ABI (``VOKRA_OK == 0``) so a stray drift would fail the
# ``check-py-bindings.sh --check`` gate long before this constant lied.
_VOKRA_OK = 0


class Session(Handle):
    """RAII wrapper around a ``VokraSession`` handle.

    Instances are normally constructed via :meth:`open` which loads a
    GGUF model from disk. Direct construction (``Session(lib, handle)``)
    is reserved for tests and for the T10/T11 higher-level wrappers that
    already own a valid handle.
    """

    __slots__ = ("_lib",)

    def __init__(
        self,
        lib: ctypes.CDLL,
        handle: Optional[ctypes.c_void_p],
    ) -> None:
        """Wrap an already-created ``VokraSession`` handle.

        Parameters
        ----------
        lib:
            The loaded native CDLL. Must have ``vokra_session_destroy``
            attached with the correct ``argtypes`` — normally done by
            :func:`_bindings.bind` at load time. We do NOT call ``bind``
            here so callers can pre-attach prototypes once per process.
        handle:
            The raw ``ctypes.c_void_p`` returned by
            ``vokra_session_create_from_file`` (or a compatible native
            constructor). ``None`` marks the wrapper as already closed.
        """
        super().__init__(handle)
        # Keep a strong ref to the CDLL so ``_release`` can call the
        # destroy symbol even if the caller has dropped the module-level
        # library reference. The library itself is process-wide and
        # cheap to keep alive.
        self._lib = lib

    # -- factory --------------------------------------------------------------

    @classmethod
    def open(cls, path: str, *, lib: Optional[ctypes.CDLL] = None) -> "Session":
        """Open a GGUF model file and return a live ``Session``.

        ``lib`` is exposed for tests that want to inject a mock CDLL;
        production callers omit it and get the process-wide library
        loaded by ``_native.load_library()`` (T09).

        Fails loudly:
        * ``FileNotFoundError`` if ``path`` does not exist — we check
          before the FFI call so the native side never has to deal with
          missing files (LC_NUMERIC-safe path handling: we pass bytes,
          not floats).
        * ``RuntimeError`` if the native constructor returns non-OK. T09
          will refine this to the ``VokraError`` hierarchy without
          touching call sites.
        """
        # Defer the actual load to T09; for now we require an explicit
        # ``lib`` so this module has no import-time dependency on the
        # not-yet-implemented ``_native.load_library``.
        if lib is None:  # pragma: no cover - guarded by tests via injection
            from ._native import load_library  # type: ignore[import-not-found]

            lib = bind(load_library())

        if not os.path.exists(path):
            raise FileNotFoundError(f"vokra: model file not found: {path!r}")

        out = ctypes.c_void_p()
        # Encode as UTF-8 filesystem path; the C ABI takes ``const char *``
        # and Windows-side conversion happens inside the runtime. We
        # never call ``strtod`` / locale-sensitive parsers, satisfying
        # NFR-RL-01.
        status = lib.vokra_session_create_from_file(
            path.encode("utf-8"),
            ctypes.byref(out),
        )
        if status != _VOKRA_OK or not out.value:
            # Fetch the thread-local error string on the SAME frame the
            # status was observed — the C ABI documents ``vokra_last_error``
            # as thread-local and valid until the next FFI call on this
            # thread. T09 will convert this to VokraError.
            last = lib.vokra_last_error()
            msg = last.decode("utf-8", "replace") if last else ""
            raise RuntimeError(
                f"vokra_session_create_from_file failed "
                f"(status={status}): {msg!r}"
            )

        return cls(lib, out)

    # -- high-level ASR / TTS API (T10) ---------------------------------------

    def transcribe(
        self,
        pcm,
        sample_rate: int,
    ) -> str:
        """Transcribe mono ``f32`` PCM to a UTF-8 string.

        This is a thin wrapper around ``vokra_asr_transcribe``: the wrapper
        marshals a Python sequence of floats into a contiguous
        ``ctypes.c_float`` array, invokes the C ABI, converts any non-OK
        status into the matching ``VokraError`` subclass (FR-EX-08 — no
        silent fallback to CPU, no default string), and returns the
        transcript.

        Parameters
        ----------
        pcm:
            Mono ``f32`` samples. Accepts anything that supports the
            iterator/len protocol (list, tuple, ``array.array('f')``,
            ``numpy.ndarray`` via ``.tolist()`` or ``ascontiguousarray``).
            May be empty; in that case we pass a NULL PCM pointer with
            ``num_samples == 0`` as the C ABI explicitly allows.
        sample_rate:
            PCM sample rate in Hz. Must match the model's front-end rate;
            the runtime returns ``VOKRA_ERROR_INVALID_ARGUMENT`` on a
            mismatch (M0 does not resample). We do not clamp or default
            — the caller owns the choice.

        Returns
        -------
        str
            The UTF-8 transcript. Empty on genuinely empty input; not on
            error (errors raise).

        Raises
        ------
        VokraError
            The matching subclass for the ``vokra_status_t`` returned by
            the C ABI. Backend-unavailable / unsupported-op errors are
            re-raised verbatim so callers can distinguish "GPU not
            wired" from "invalid argument" (FR-EX-08).

        Notes
        -----
        We never touch ``float(str)`` or ``locale.setlocale`` — all
        numeric marshalling is via ``ctypes`` scalars, so this method is
        LC_NUMERIC-safe (NFR-RL-01).
        """
        lib = self._lib
        handle = self.raw()  # UAF gate

        # Build the ``c_float * n`` buffer. Building a fresh contiguous
        # array (rather than reinterpreting numpy memory) keeps the
        # binding pure-``ctypes`` — no numpy dependency at the FFI seam —
        # and pins the buffer inside this call frame so Python's GC
        # cannot reclaim it before the C ABI returns (see R4 in the
        # M2-12 risk register).
        num_samples = len(pcm) if pcm is not None else 0
        if num_samples == 0:
            pcm_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_float))
        else:
            buf = (ctypes.c_float * num_samples)(*pcm)
            pcm_ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))

        out_text = ctypes.c_char_p()
        status = lib.vokra_asr_transcribe(
            handle,
            pcm_ptr,
            ctypes.c_size_t(num_samples),
            ctypes.c_int32(int(sample_rate)),
            ctypes.byref(out_text),
        )
        if status != _VOKRA_OK:
            # Read the thread-local error *immediately* in the same call
            # frame so the message belongs to this failure (C ABI §3-c).
            last = lib.vokra_last_error()
            msg = last.decode("utf-8", "replace") if last else ""
            # ``raise_from_status`` never returns; ``out_text`` is
            # untouched per the C ABI contract so nothing to free.
            raise_from_status(status, message=msg)

        # Success: decode, then free the Vokra-owned buffer. We decode
        # before freeing so a decode failure does not leak the C-side
        # allocation.
        try:
            raw = out_text.value  # bytes (NUL-terminated) or None
            text = raw.decode("utf-8", "replace") if raw else ""
        finally:
            lib.vokra_string_free(out_text)
        return text

    def synthesize(self, text: str) -> Tuple[list, int]:
        """Synthesize speech from UTF-8 text.

        Thin wrapper around ``vokra_tts_synthesize``. On success the
        Vokra-owned PCM buffer is copied into a Python ``list[float]``
        and freed before returning, so the caller never has to worry
        about C-side ownership.

        Parameters
        ----------
        text:
            The UTF-8 text to synthesize. May be empty; the C ABI
            accepts an empty string (the model decides whether to
            return empty PCM or fail).

        Returns
        -------
        tuple[list[float], int]
            ``(pcm, sample_rate)`` where ``pcm`` is mono ``f32``
            samples in ``[-1, 1]`` as a plain Python list and
            ``sample_rate`` is the model's output rate in Hz.

        Raises
        ------
        VokraError
            The matching subclass for the ``vokra_status_t`` returned
            by the C ABI. Propagated verbatim (FR-EX-08).

        Notes
        -----
        We return a ``list[float]`` rather than a numpy array to keep
        the binding numpy-optional (``audio.py`` has a numpy-aware
        helper for callers who want zero-copy). Copying is done via
        ``list(buf)`` which uses the CPython buffer protocol under the
        hood — negligible for sub-second clips.
        """
        lib = self._lib
        handle = self.raw()  # UAF gate

        # The C ABI takes ``const char *``; UTF-8 encode explicitly so
        # non-ASCII scripts (JA/ZH/KO) round-trip losslessly.
        text_bytes = text.encode("utf-8") if text is not None else b""

        out_pcm = ctypes.POINTER(ctypes.c_float)()
        out_num_samples = ctypes.c_size_t(0)
        out_sample_rate = ctypes.c_int32(0)

        status = lib.vokra_tts_synthesize(
            handle,
            text_bytes,
            ctypes.byref(out_pcm),
            ctypes.byref(out_num_samples),
            ctypes.byref(out_sample_rate),
        )
        if status != _VOKRA_OK:
            last = lib.vokra_last_error()
            msg = last.decode("utf-8", "replace") if last else ""
            # All three out-params are untouched on error per the C ABI
            # contract, so nothing to free.
            raise_from_status(status, message=msg)

        # Success: copy PCM into a Python list, then free the C-side
        # buffer. Copy before free so an OOM in the copy does not leak.
        n = int(out_num_samples.value)
        sr = int(out_sample_rate.value)
        try:
            if n == 0 or not out_pcm:
                pcm_list: list = []
            else:
                # ``out_pcm[:n]`` slices the POINTER(c_float) into a
                # Python list of floats — the standard ctypes idiom.
                pcm_list = list(out_pcm[:n])
        finally:
            lib.vokra_audio_free(out_pcm, ctypes.c_size_t(n))
        return pcm_list, sr

    # -- lifecycle ------------------------------------------------------------

    def _release(self) -> None:
        """Call ``vokra_session_destroy`` exactly once.

        The base class guarantees this is only invoked while
        ``self._handle`` is non-``None`` — that is the invariant that
        blocks double-free. If ``vokra_session_destroy`` itself raises
        (it should not; the symbol has ``restype=None`` and never returns
        a status), the base class still clears the handle so a retry
        does not re-enter the C ABI.
        """
        # ``self._handle`` is guaranteed non-None here by the ``Handle``
        # base contract (``close()`` early-returns on ``None``).
        assert self._handle is not None
        self._lib.vokra_session_destroy(self._handle)

    def __repr__(self) -> str:
        state = "closed" if self._handle is None else f"raw=0x{int(self._handle.value or 0):x}"
        return f"<Session {state}>"
