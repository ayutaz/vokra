# SPDX-License-Identifier: Apache-2.0
"""Streaming inference handle wrapper with RAII lifecycle + push/poll (T08+T11).

The Vokra C ABI exposes streaming inference as an opaque handle opened
from a parent :class:`Session` via ``vokra_stream_open`` and released
with ``vokra_stream_destroy``. This module wraps that pair in a Python
:class:`Stream` class that is safe to use in ``with`` blocks, is
idempotent under double-close, and never leaves the native handle
leaked on the happy path.

Ownership contract (single-thread, caller-driven polling)
----------------------------------------------------------
A :class:`Stream` is **single-thread-owned**. The C ABI documents that
``vokra_last_error`` is thread-local and that ``push_pcm`` / ``poll`` /
``poll_events`` on the same handle must be serialised by the caller
(see ``docs/adr/ADR-* language-binding-conventions``). Do not share a
:class:`Stream` instance across threads: either fence it with your own
lock or open one stream per worker thread. Streams **do not** extend
the lifetime of their parent :class:`Session` from Python's
perspective — close the stream first, then the session, or use them
in nested ``with`` blocks.

The C ABI is deliberately **callback-free**: the runtime buffers
produced samples and events internally, and the caller drains them at
its own cadence via ``poll`` / ``poll_events``. That means the Python
wrapper does not need ``ctypes.CFUNCTYPE`` trampolines nor rooted
callback references — the whole surface is plain synchronous FFI
serialised by the caller thread (§D2/D7 of the M2-12 plan). If a
future ABI revision adds push callbacks (IF-01 is pre-1.0), the design
will need CFUNCTYPE + rooted references then, but as of the ABI at
time of writing there are none.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, List, Optional, Tuple

from ._bindings import (
    VOKRA_EVENT_SPEECH_PROB,
    VOKRA_EVENT_TOKEN,
    VOKRA_EVENT_UNKNOWN,
    vokra_event_t,
)
from ._handles import Handle

if TYPE_CHECKING:  # pragma: no cover
    from .session import Session


# Success status mirrored locally to keep this module free of a full
# enum import (matches the pattern in ``session.py``). The value is
# fixed by the C ABI (``VOKRA_OK == 0``); drift is caught by
# ``check-py-bindings.sh --check`` long before this constant lies.
_VOKRA_OK = 0


class Event(Tuple[int, int, float]):
    """Immutable 3-tuple view of ``vokra_event_t``.

    Fields are exposed both positionally (for cheap unpacking and
    ``isinstance(x, tuple)`` friendly assertions in tests) and by name
    via the ``kind`` / ``a`` / ``b`` properties so callers can write
    ``ev.kind == vokra.EVENT_TOKEN`` without indexing into a bare
    tuple. Kept as a subclass of :class:`tuple` (not a dataclass) so
    the object is hashable, small, and copyable without depending on
    any non-stdlib helper.

    The three C fields are:

    * ``kind`` — one of ``VOKRA_EVENT_UNKNOWN`` / ``_SPEECH_PROB`` /
      ``_TOKEN``. Unknown kinds are surfaced verbatim so a newer
      runtime pushing extra events does not silently vanish.
    * ``a`` — ``uint32``. For ``TOKEN`` events this is the token id;
      other kinds may reuse it as an integer payload.
    * ``b`` — ``float32``. For ``SPEECH_PROB`` this is the probability
      in ``[0.0, 1.0]``; other kinds may reuse it as a float payload.
    """

    __slots__ = ()

    def __new__(cls, kind: int, a: int, b: float) -> "Event":
        # ``tuple.__new__`` takes an iterable — building it from a
        # 3-tuple literal is the standard idiom and avoids the more
        # expensive ``*args`` path.
        return tuple.__new__(cls, (int(kind), int(a), float(b)))

    @property
    def kind(self) -> int:
        return self[0]

    @property
    def a(self) -> int:
        return self[1]

    @property
    def b(self) -> float:
        return self[2]

    def __repr__(self) -> str:
        name = {
            VOKRA_EVENT_UNKNOWN: "UNKNOWN",
            VOKRA_EVENT_SPEECH_PROB: "SPEECH_PROB",
            VOKRA_EVENT_TOKEN: "TOKEN",
        }.get(self.kind, f"kind={self.kind}")
        return f"Event({name}, a={self.a}, b={self.b:.6g})"


class Stream(Handle):
    """RAII wrapper around a ``vokra_stream_t`` opaque handle.

    Parameters
    ----------
    session:
        Parent :class:`Session` the stream is opened against. Must be
        live (not closed) at construction time; the C ABI does not
        retain the session, so the caller is responsible for keeping
        it alive for the stream's lifetime.
    sample_rate_hz:
        PCM sample rate the stream will receive via :meth:`push`.
        Passed straight to ``vokra_stream_open``; validation is done
        natively.
    lib:
        Loaded ``ctypes.CDLL`` with :mod:`_bindings` prototypes attached.
        Injected for testability so the wrapper works against a stub
        library without loading ``libvokra`` from disk (the T08/T11
        tests run before the native artifact is bundled into the
        wheel). When ``None``, the constructor refuses to touch native
        code and raises ``RuntimeError`` — production callers always
        pass a lib.
    """

    __slots__ = ("_lib",)

    def __init__(
        self,
        session: "Session",
        sample_rate_hz: int,
        *,
        lib: Optional[ctypes.CDLL] = None,
    ) -> None:
        if lib is None:
            # No implicit fallback (FR-EX-08): callers must inject the
            # already-bound library or the higher-level `vokra` module
            # must resolve it before constructing a Stream.
            raise RuntimeError(
                "Stream requires a bound ctypes library; pass lib=<CDLL>"
            )

        # Ensure the parent session hasn't been closed. ``.raw()`` on a
        # closed handle raises RuntimeError which we let propagate: the
        # C ABI would otherwise segfault on a NULL session pointer.
        session_ptr = session.raw()

        # Match `_bindings.PROTOTYPES['vokra_stream_open']`:
        # (session*, sample_rate: i32, out_stream**) -> status
        out_stream = ctypes.c_void_p()
        status = lib.vokra_stream_open(
            session_ptr,
            ctypes.c_int32(int(sample_rate_hz)),
            ctypes.byref(out_stream),
        )
        if status != _VOKRA_OK:
            # T09 will replace this with the VokraError hierarchy that
            # reads vokra_last_error() on the same thread. For T08/T11
            # we keep the failure loud but self-contained so this
            # module does not depend on errors.py yet.
            raise RuntimeError(
                f"vokra_stream_open failed (status={status}); "
                "see vokra_last_error() for detail"
            )
        if not out_stream.value:
            # Defensive: OK status but NULL out-pointer would be a
            # runtime contract violation. Fail loud rather than proceed
            # with a NULL handle that would later NULL-deref natively.
            raise RuntimeError(
                "vokra_stream_open returned VOKRA_OK but null handle"
            )

        # Initialise the RAII base class with the acquired handle.
        # Base class stores it in _handle and drives close()/__del__.
        super().__init__(handle=out_stream)
        self._lib = lib

    # -- push/poll ----------------------------------------------------------

    def push(self, pcm) -> None:
        """Push a chunk of ``float32`` mono PCM samples into the stream.

        ``pcm`` may be any Python sequence (``list`` / ``tuple`` /
        ``array.array('f')`` / a numpy ``float32`` array via
        ``numpy.ascontiguousarray``) that supports ``len()`` and
        elementwise ``float`` coercion. For zero-copy paths callers
        pass a numpy array of dtype ``float32`` — we rebuild a
        ``ctypes`` pointer from the numpy buffer inline so the source
        array is kept rooted for the duration of the FFI call (see
        risk R4 in the plan: source-array binding prevents UAF when
        the GC would otherwise reclaim mid-call).

        Empty pushes are allowed and succeed as a no-op at the ABI
        layer; we forward them unchanged so the runtime can observe
        zero-length keep-alives if it wants.
        """
        n = len(pcm)
        if n == 0:
            # ``(c_float * 0)()`` is a valid but degenerate array; the
            # runtime may treat a 0-length push as a heartbeat. We
            # still round-trip through the ABI so behaviour is
            # observable to tests.
            buf = (ctypes.c_float * 0)()
            ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))
        else:
            # Try the fast path first: numpy float32 arrays expose a
            # ``.ctypes.data_as`` view without copying. Fall back to
            # an eager ``(c_float * n)(*pcm)`` copy for plain Python
            # sequences. The eager copy is O(n) but keeps the wrapper
            # dependency-free for callers who do not have numpy.
            ctypes_iface = getattr(pcm, "ctypes", None)
            dtype_name = getattr(getattr(pcm, "dtype", None), "name", None)
            if (
                ctypes_iface is not None
                and dtype_name == "float32"
                and getattr(pcm, "flags", None) is not None
                and getattr(pcm.flags, "c_contiguous", False)
            ):
                # Zero-copy: numpy view. ``pcm`` stays rooted as the
                # local ``pcm`` reference until this method returns,
                # which is after the C call completes.
                ptr = ctypes_iface.data_as(ctypes.POINTER(ctypes.c_float))
                buf = pcm  # noqa: F841 — keep the source alive during the call
            else:
                # Copy path. ``(c_float * n)(*pcm)`` iterates once and
                # coerces each element via ``float()``, which is
                # locale-independent (Python floats do not go through
                # ``strtod``), satisfying NFR-RL-01.
                buf = (ctypes.c_float * n)(*(float(x) for x in pcm))
                ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))

        # Match `_bindings.PROTOTYPES['vokra_stream_push_pcm']`:
        # (stream*, pcm*, n_samples: size_t) -> status
        status = self._lib.vokra_stream_push_pcm(
            self.raw(),
            ptr,
            ctypes.c_size_t(n),
        )
        if status != _VOKRA_OK:
            raise RuntimeError(
                f"vokra_stream_push_pcm failed (status={status}); "
                "see vokra_last_error() for detail"
            )

    def poll(self, capacity: int = 64) -> List[float]:
        """Drain up to ``capacity`` produced audio samples from the stream.

        Returns a fresh ``list[float]`` of length ``[0, capacity]``.
        An empty list means the runtime has nothing to hand back right
        now — the caller should push more PCM and poll again. This is
        the caller-driven polling model documented in the C ABI
        report (§4): there is no callback trampoline, no background
        thread, and no rooted CFUNCTYPE — the entire loop is a plain
        synchronous FFI cycle serialised by the caller thread.

        ``capacity`` is the maximum number of samples the caller wants
        in this drain — the runtime writes at most that many and
        reports how many it actually produced via an out-parameter.
        We size the ctypes buffer to ``capacity`` and slice down to
        the produced length before returning.
        """
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        if capacity == 0:
            # Round-trip through the ABI so behaviour is observable:
            # some runtimes report an available-count even when the
            # caller passes a zero-length buffer. We ignore the count
            # here because the caller has explicitly asked for none.
            return []

        buf = (ctypes.c_float * capacity)()
        produced = ctypes.c_size_t(0)
        # Match `_bindings.PROTOTYPES['vokra_stream_poll']`:
        # (stream*, out_pcm*, capacity: size_t, out_produced*) -> status
        status = self._lib.vokra_stream_poll(
            self.raw(),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(capacity),
            ctypes.byref(produced),
        )
        if status != _VOKRA_OK:
            raise RuntimeError(
                f"vokra_stream_poll failed (status={status}); "
                "see vokra_last_error() for detail"
            )
        n = int(produced.value)
        if n < 0 or n > capacity:
            # Defensive: a broken runtime that reports produced >
            # capacity would let us read out-of-bounds if we blindly
            # sliced. Fail loud instead.
            raise RuntimeError(
                f"vokra_stream_poll returned produced={n} > capacity={capacity}"
            )
        # Slice the ctypes array to a Python list. ``list(buf[:n])``
        # copies the samples into GC-owned Python floats so the
        # ctypes buffer is safe to release when this method returns.
        return list(buf[:n])

    def poll_events(self, capacity: int = 64) -> List[Event]:
        """Drain up to ``capacity`` events (VAD probs, ASR tokens, …).

        Same caller-driven contract as :meth:`poll` — no callback, no
        background thread, no rooted references needed. Returns a
        fresh ``list[Event]`` of length ``[0, capacity]``. Each entry
        is an immutable :class:`Event` triple ``(kind, a, b)`` where
        ``kind`` is one of the ``VOKRA_EVENT_*`` constants.

        The runtime is free to interleave events across different
        streams (VAD, ASR beam) but this wrapper preserves the order
        the runtime returns them — we do not sort, dedupe, or filter
        by kind here so callers can build higher-level filters on top.
        """
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {capacity}")
        if capacity == 0:
            return []

        buf = (vokra_event_t * capacity)()
        produced = ctypes.c_size_t(0)
        # Match `_bindings.PROTOTYPES['vokra_stream_poll_events']`:
        # (stream*, out_events*, capacity: size_t, out_produced*) -> status
        status = self._lib.vokra_stream_poll_events(
            self.raw(),
            ctypes.cast(buf, ctypes.POINTER(vokra_event_t)),
            ctypes.c_size_t(capacity),
            ctypes.byref(produced),
        )
        if status != _VOKRA_OK:
            raise RuntimeError(
                f"vokra_stream_poll_events failed (status={status}); "
                "see vokra_last_error() for detail"
            )
        n = int(produced.value)
        if n < 0 or n > capacity:
            raise RuntimeError(
                f"vokra_stream_poll_events returned produced={n} > capacity={capacity}"
            )
        # Copy each event out of the ctypes struct array into an
        # ``Event`` tuple so the caller cannot accidentally see stale
        # memory after the buffer is released.
        return [Event(buf[i].kind, buf[i].a, buf[i].b) for i in range(n)]

    # -- RAII hook -----------------------------------------------------------

    def _release(self) -> None:
        """Invoke ``vokra_stream_destroy`` on the wrapped handle.

        Called at most once by :meth:`Handle.close` because the base
        clears ``self._handle`` after the first call, making later
        ``close()``/``__del__`` invocations no-ops.
        """
        # ``self._handle`` is guaranteed non-None here (close() early-
        # returns otherwise) but grab it defensively for the ctypes call.
        handle = self._handle
        if handle is None:
            return
        # vokra_stream_destroy returns void; PROTOTYPES restype is None.
        self._lib.vokra_stream_destroy(handle)
