# SPDX-License-Identifier: Apache-2.0
"""Session RAII contract tests (T08).

These tests verify the invariants that keep the C ABI safe from Python:

1. ``vokra_session_destroy`` is called **exactly once** per successful
   ``vokra_session_create_from_file``, regardless of whether release is
   triggered by ``__exit__``, ``close()``, or ``__del__``.
2. Double-close is idempotent (destroy is not re-invoked).
3. Use-after-free through ``raw()`` raises ``RuntimeError`` before
   reaching the C ABI.
4. A failed constructor does not leave a live handle behind (no destroy
   is called for a NULL / failed create).

No prebuilt native library is required: we inject a stub ``CDLL``-like
object that records calls, matching the ``_bindings.PROTOTYPES`` shape
that :func:`_bindings.bind` normally attaches at import time.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without installing the package. This
# mirrors what ``pip install -e .`` would do but keeps the tests runnable
# in a fresh checkout without a build step.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vokra.session import Session, _VOKRA_OK  # noqa: E402
from vokra._bindings import (  # noqa: E402
    VOKRA_EVENT_SPEECH_PROB,
    VOKRA_EVENT_TOKEN,
    vokra_event_t,
)
from vokra.stream import Event, Stream  # noqa: E402


# --- fake native library ----------------------------------------------------


class FakeLib:
    """Stand-in for the loaded ``libvokra`` CDLL.

    Records every ``vokra_session_destroy`` call so tests can assert the
    exact number of destroys and the pointer they targeted. The
    ``create`` side is scripted per-test via ``next_status`` /
    ``next_handle`` so we can exercise both success and failure paths
    without allocating real memory.
    """

    def __init__(self) -> None:
        self.destroy_calls: list[int] = []
        self.last_error_bytes: bytes = b""
        self.next_status: int = _VOKRA_OK
        self.next_handle: int = 0xDEADBEEF  # sentinel non-NULL pointer

        # ASR / TTS scripting (T10) -----------------------------------------
        # ``transcribe_status`` / ``synthesize_status`` gate error paths;
        # ``transcribe_text`` / ``synthesize_pcm`` script happy paths.
        self.transcribe_status: int = _VOKRA_OK
        self.transcribe_text: bytes = b"hello world"
        # (num_samples, sample_rate) recorded per call so tests can assert
        # marshalling shape.
        self.transcribe_calls: list[tuple[int, int]] = []
        self.string_free_calls: list = []

        self.synthesize_status: int = _VOKRA_OK
        self.synthesize_pcm: list[float] = [0.0, 0.25, -0.5, 0.75]
        self.synthesize_sample_rate: int = 22050
        self.synthesize_calls: list[bytes] = []
        # ``num_samples`` argument recorded on every ``vokra_audio_free``.
        self.audio_free_calls: list[int] = []

        # Keep C-side allocations alive between call and paired free so
        # the pointers we return stay valid until the wrapper frees them
        # — mirrors the real ABI's ownership contract.
        self._pcm_buffers: list = []
        self._text_buffers: list = []

    # -- symbols the wrapper actually calls ---------------------------------

    def vokra_session_create_from_file(self, path_bytes, out_pp):
        # ``out_pp`` is ``byref(c_void_p)``; write the sentinel unless we
        # were told to fail.
        if self.next_status != _VOKRA_OK:
            # Real ABI leaves out untouched on failure; we mirror that.
            return self.next_status
        # ``out_pp._obj`` is the underlying ``c_void_p`` (byref exposes it).
        out_pp._obj.value = self.next_handle
        return _VOKRA_OK

    def vokra_session_destroy(self, handle):
        # ``handle`` arrives as a ``c_void_p``; store the raw address so
        # tests can assert on identity, not object identity.
        val = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.destroy_calls.append(val)

    def vokra_last_error(self):
        return self.last_error_bytes or None

    # -- T10 ASR / TTS symbols ----------------------------------------------

    def vokra_asr_transcribe(self, session, pcm_ptr, num_samples, sample_rate, out_pp):
        n = num_samples.value if isinstance(num_samples, ctypes.c_size_t) else int(num_samples)
        sr = sample_rate.value if isinstance(sample_rate, ctypes.c_int32) else int(sample_rate)
        self.transcribe_calls.append((n, sr))
        if self.transcribe_status != _VOKRA_OK:
            return self.transcribe_status
        # Allocate a stable ``c_char_p`` (owned by us until string_free is
        # called) and write its bytes address into ``*out_pp``. Retain it
        # so it cannot be GC'd before the wrapper reads / frees it.
        text_buf = ctypes.c_char_p(self.transcribe_text)
        self._text_buffers.append(text_buf)
        # ``out_pp`` was passed as ``byref(c_char_p())``; ``_obj`` is the
        # inner ``c_char_p`` we assign to.
        out_pp._obj.value = text_buf.value
        return _VOKRA_OK

    def vokra_string_free(self, s):
        # ``s`` is a ``c_char_p``; record the bytes for assertion.
        val = s.value if hasattr(s, "value") else s
        self.string_free_calls.append(val)

    def vokra_tts_synthesize(self, session, text_bytes, out_pcm_pp, out_num_pp, out_sr_pp):
        self.synthesize_calls.append(text_bytes)
        if self.synthesize_status != _VOKRA_OK:
            return self.synthesize_status
        n = len(self.synthesize_pcm)
        # Always allocate at least 1 slot so ``addressof`` is defined; the
        # wrapper won't read past ``n``.
        buf = (
            (ctypes.c_float * max(n, 1))(*self.synthesize_pcm)
            if n
            else (ctypes.c_float * 1)()
        )
        self._pcm_buffers.append(buf)  # keep alive until audio_free
        # ``out_pcm_pp`` is ``byref(POINTER(c_float)())``; write the raw
        # buffer address into the inner POINTER's slot so that
        # ``out_pcm[:n]`` in the wrapper reads back our samples.
        ctypes.memmove(
            ctypes.addressof(out_pcm_pp._obj),
            ctypes.byref(ctypes.c_void_p(ctypes.addressof(buf))),
            ctypes.sizeof(ctypes.c_void_p),
        )
        out_num_pp._obj.value = n
        out_sr_pp._obj.value = self.synthesize_sample_rate
        return _VOKRA_OK

    def vokra_audio_free(self, pcm, num_samples):
        n = num_samples.value if isinstance(num_samples, ctypes.c_size_t) else int(num_samples)
        self.audio_free_calls.append(n)


@pytest.fixture()
def fake_lib() -> FakeLib:
    return FakeLib()


@pytest.fixture()
def tmp_model(tmp_path: Path) -> str:
    """Create a placeholder file so ``os.path.exists`` returns True.

    The FakeLib never opens it — the check exists purely to satisfy the
    ``FileNotFoundError`` guard in ``Session.open``.
    """
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\x00")
    return str(p)


# --- RAII: destroy called exactly once --------------------------------------


def test_context_manager_destroys_once(fake_lib: FakeLib, tmp_model: str) -> None:
    with Session.open(tmp_model, lib=fake_lib) as sess:
        assert not sess.closed
        assert sess.raw().value == fake_lib.next_handle
    # Exiting the `with` block must have called destroy exactly once.
    assert fake_lib.destroy_calls == [fake_lib.next_handle]
    assert sess.closed


def test_explicit_close_destroys_once(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    assert fake_lib.destroy_calls == [fake_lib.next_handle]
    assert sess.closed


def test_del_destroys_when_no_explicit_close(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    handle_val = fake_lib.next_handle
    del sess
    # Force finalizers to run — CPython usually reclaims on the last
    # decref, but be defensive on other implementations.
    gc.collect()
    assert fake_lib.destroy_calls == [handle_val]


# --- double-free rejection --------------------------------------------------


def test_double_close_is_noop(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    sess.close()  # must not re-invoke destroy
    sess.close()
    assert fake_lib.destroy_calls == [fake_lib.next_handle]


def test_close_then_del_is_noop(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    handle_val = fake_lib.next_handle
    del sess
    gc.collect()
    # __del__ triggers a second close(), which the Handle base class
    # short-circuits because _handle is already None.
    assert fake_lib.destroy_calls == [handle_val]


def test_context_manager_then_explicit_close(fake_lib: FakeLib, tmp_model: str) -> None:
    with Session.open(tmp_model, lib=fake_lib) as sess:
        pass
    sess.close()  # post-with explicit close is still a no-op
    assert fake_lib.destroy_calls == [fake_lib.next_handle]


# --- UAF rejection ----------------------------------------------------------


def test_raw_after_close_raises(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        sess.raw()


def test_reenter_context_after_close_raises(fake_lib: FakeLib, tmp_model: str) -> None:
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        with sess:
            pass  # pragma: no cover - body must not execute


# --- failure paths do not leak / spurious-destroy ---------------------------


def test_create_failure_raises_and_no_destroy(fake_lib: FakeLib, tmp_model: str) -> None:
    fake_lib.next_status = 5  # VOKRA_ERROR_INVALID_ARGUMENT
    fake_lib.last_error_bytes = b"synthetic failure"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        Session.open(tmp_model, lib=fake_lib)
    # A failed create must not schedule a destroy — there is no handle
    # to release, and calling destroy(NULL) is UB in C.
    assert fake_lib.destroy_calls == []


def test_missing_file_raises_before_ffi(fake_lib: FakeLib, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.gguf"
    with pytest.raises(FileNotFoundError):
        Session.open(str(missing), lib=fake_lib)
    # Guard fires before any FFI call, so nothing to destroy.
    assert fake_lib.destroy_calls == []


# --- ownership across GC boundary -------------------------------------------


def test_two_sessions_release_independently(fake_lib: FakeLib, tmp_model: str) -> None:
    fake_lib.next_handle = 0x1111
    s1 = Session.open(tmp_model, lib=fake_lib)
    fake_lib.next_handle = 0x2222
    s2 = Session.open(tmp_model, lib=fake_lib)

    s1.close()
    assert fake_lib.destroy_calls == [0x1111]
    s2.close()
    assert fake_lib.destroy_calls == [0x1111, 0x2222]


# --- Stream RAII (T08 c10) --------------------------------------------------
#
# Same contract as Session but bound to vokra_stream_open / vokra_stream_destroy.
# Stream is opened against a live parent Session; the C ABI does NOT retain
# the parent, so we assert that Stream operates without extending session
# lifetime and refuses to open against a closed session.


class FakeStreamLib(FakeLib):
    """FakeLib extended with the stream open/destroy pair.

    Records open + destroy calls independently from session-side calls so
    the test can assert 1:1 pairing across the whole RAII surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stream_open_calls: list[int] = []
        self.stream_destroy_calls: list[int] = []
        self.next_stream_status: int = _VOKRA_OK
        self.next_stream_handle: int = 0xC0FFEE00

    def vokra_stream_open(self, session_ptr, rate, out_pp):
        # session_ptr arrives as c_void_p; rate as c_int32. We only care
        # that both are truthy so the wrapper is invoking us correctly.
        if self.next_stream_status != _VOKRA_OK:
            return self.next_stream_status
        self.stream_open_calls.append(self.next_stream_handle)
        out_pp._obj.value = self.next_stream_handle
        return _VOKRA_OK

    def vokra_stream_destroy(self, handle):
        val = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.stream_destroy_calls.append(val)

    # -- push / poll (T11) ---------------------------------------------------
    #
    # These fields track everything the wrapper hands us so the T11 tests
    # can assert on payload, capacity, cardinality, and error passthrough
    # without touching a real libvokra build.

    def _init_pushpoll(self) -> None:
        if not hasattr(self, "push_calls"):
            self.push_calls: list[list[float]] = []
            self.pending_pcm: list[float] = []
            self.pending_events: list[Event] = []
            self.push_status: int = _VOKRA_OK
            self.poll_status: int = _VOKRA_OK
            self.poll_events_status: int = _VOKRA_OK
            # A broken runtime knob: if set > 0, poll reports this many
            # samples via out_produced regardless of what it actually
            # wrote — used to prove the wrapper catches produced > cap.
            self.forced_produced: int | None = None

    def vokra_stream_push_pcm(self, stream_ptr, pcm_ptr, n_samples):
        self._init_pushpoll()
        n = n_samples.value if isinstance(n_samples, ctypes.c_size_t) else int(n_samples)
        # Copy out of the wrapper's ctypes buffer before it goes out of
        # scope; the test asserts against these exact floats.
        chunk = [float(pcm_ptr[i]) for i in range(n)]
        self.push_calls.append(chunk)
        self.pending_pcm.extend(chunk)
        return self.push_status

    def vokra_stream_poll(self, stream_ptr, out_pcm, capacity, out_produced):
        self._init_pushpoll()
        if self.poll_status != _VOKRA_OK:
            return self.poll_status
        cap = capacity.value if isinstance(capacity, ctypes.c_size_t) else int(capacity)
        take = min(cap, len(self.pending_pcm))
        for i in range(take):
            out_pcm[i] = ctypes.c_float(self.pending_pcm[i])
        del self.pending_pcm[:take]
        out_produced._obj.value = (
            self.forced_produced if self.forced_produced is not None else take
        )
        return _VOKRA_OK

    def vokra_stream_poll_events(self, stream_ptr, out_ev, capacity, out_produced):
        self._init_pushpoll()
        if self.poll_events_status != _VOKRA_OK:
            return self.poll_events_status
        cap = capacity.value if isinstance(capacity, ctypes.c_size_t) else int(capacity)
        take = min(cap, len(self.pending_events))
        for i in range(take):
            ev = self.pending_events[i]
            out_ev[i].kind = int(ev.kind)
            out_ev[i].a = int(ev.a)
            out_ev[i].b = float(ev.b)
        del self.pending_events[:take]
        out_produced._obj.value = take
        return _VOKRA_OK


@pytest.fixture()
def stream_lib() -> FakeStreamLib:
    return FakeStreamLib()


def test_stream_lifecycle(stream_lib: FakeStreamLib, tmp_model: str) -> None:
    """End-to-end Stream RAII across with/close/__del__/exception paths.

    Verifies every invariant that keeps the C ABI safe:
    * with-block releases exactly once on normal exit
    * with-block releases exactly once on exception
    * explicit close() is idempotent under repeated invocation
    * __del__ after close() does not double-destroy
    * __del__ on an orphan wrapper releases
    * use-after-free surfaces as RuntimeError from raw()
    * opening against a closed Session raises before touching the C ABI
    * missing lib kwarg raises without touching the C ABI
    """
    # -- happy path: with-block ---------------------------------------
    session = Session.open(tmp_model, lib=stream_lib)
    stream_lib.next_stream_handle = 0x11110000
    with Stream(session, sample_rate_hz=16000, lib=stream_lib) as s:
        assert not s.closed
        assert s.raw().value == 0x11110000
        assert stream_lib.stream_open_calls == [0x11110000]
        assert stream_lib.stream_destroy_calls == []
    assert stream_lib.stream_destroy_calls == [0x11110000]
    assert s.closed
    with pytest.raises(RuntimeError, match="closed"):
        s.raw()

    # -- explicit close() is idempotent -------------------------------
    stream_lib.next_stream_handle = 0x22220000
    s2 = Stream(session, sample_rate_hz=24000, lib=stream_lib)
    s2.close()
    s2.close()
    s2.close()
    assert stream_lib.stream_destroy_calls == [0x11110000, 0x22220000]

    # -- __del__ after explicit close does not double-destroy ---------
    stream_lib.next_stream_handle = 0x33330000
    s3 = Stream(session, sample_rate_hz=48000, lib=stream_lib)
    s3.close()
    del s3
    gc.collect()
    assert stream_lib.stream_destroy_calls == [0x11110000, 0x22220000, 0x33330000]

    # -- __del__ on orphan releases -----------------------------------
    stream_lib.next_stream_handle = 0x44440000
    s4 = Stream(session, sample_rate_hz=16000, lib=stream_lib)
    del s4
    gc.collect()
    assert stream_lib.stream_destroy_calls[-1] == 0x44440000
    assert len(stream_lib.stream_destroy_calls) == 4

    # -- with-block releases on exception -----------------------------
    class _Boom(RuntimeError):
        pass

    stream_lib.next_stream_handle = 0x55550000
    with pytest.raises(_Boom):
        with Stream(session, sample_rate_hz=16000, lib=stream_lib):
            raise _Boom()
    assert stream_lib.stream_destroy_calls[-1] == 0x55550000
    assert len(stream_lib.stream_destroy_calls) == 5

    # -- opening against a closed Session refuses (no NULL to C ABI) --
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        Stream(session, sample_rate_hz=16000, lib=stream_lib)
    # No extra open should have reached the fake lib.
    assert len(stream_lib.stream_open_calls) == 5

    # -- open failure raises and leaves no dangling handle ------------
    session2 = Session.open(tmp_model, lib=stream_lib)
    stream_lib.next_stream_status = 5  # VOKRA_ERROR_INVALID_ARGUMENT
    with pytest.raises(RuntimeError, match="vokra_stream_open failed"):
        Stream(session2, sample_rate_hz=16000, lib=stream_lib)
    # Failed open must not have registered a destroy pairing.
    assert len(stream_lib.stream_destroy_calls) == 5
    session2.close()

    # -- missing lib kwarg refuses (no implicit fallback) -------------
    session3 = Session.open(tmp_model, lib=stream_lib)
    with pytest.raises(RuntimeError, match="requires a bound"):
        Stream(session3, sample_rate_hz=16000)
    session3.close()


# --- T10: Session.transcribe / Session.synthesize --------------------------


def test_transcribe(fake_lib: FakeLib, tmp_model: str) -> None:
    """``transcribe`` marshals PCM, returns the transcript, frees the buffer.

    Verifies the full happy-path contract:
    * PCM sample count and sample rate are marshalled correctly.
    * The Vokra-owned transcript pointer is freed exactly once via
      ``vokra_string_free`` (no memory leak, no double-free).
    * The returned string is a plain ``str`` decoded from UTF-8.
    * Non-OK status is re-raised **verbatim** as the matching
      ``VokraError`` subclass — FR-EX-08: no silent fallback to a
      default transcript, no downgrade to CPU. Empty PCM is passed
      through as ``num_samples == 0`` (the C ABI explicitly allows
      a NULL pcm pointer in that case).
    """
    from vokra.errors import VokraInvalidArgumentError

    with Session.open(tmp_model, lib=fake_lib) as sess:
        # -- happy path -----------------------------------------------------
        fake_lib.transcribe_text = b"the quick brown fox"
        pcm = [0.0, 0.1, -0.2, 0.3, -0.4]
        text = sess.transcribe(pcm, sample_rate=16000)
        assert text == "the quick brown fox"
        assert fake_lib.transcribe_calls == [(5, 16000)]
        # The C-side transcript must be freed exactly once so the wrapper
        # never leaks the Vokra-owned buffer.
        assert len(fake_lib.string_free_calls) == 1

        # -- empty PCM: n=0 with NULL pointer is valid ---------------------
        fake_lib.transcribe_text = b""
        text2 = sess.transcribe([], sample_rate=16000)
        assert text2 == ""
        assert fake_lib.transcribe_calls[-1] == (0, 16000)
        assert len(fake_lib.string_free_calls) == 2

        # -- non-OK status raises verbatim (FR-EX-08) ----------------------
        # 5 == VOKRA_ERROR_INVALID_ARGUMENT; the wrapper must re-raise the
        # typed exception and NOT return a fallback string.
        fake_lib.transcribe_status = 5
        fake_lib.last_error_bytes = b"sample rate mismatch"
        with pytest.raises(VokraInvalidArgumentError) as exc_info:
            sess.transcribe([0.0] * 4, sample_rate=8000)
        # Error message must include the thread-local detail read
        # synchronously in the same call frame.
        assert "sample rate mismatch" in str(exc_info.value)
        # No spurious free on the error path — the C ABI leaves out_text
        # untouched on non-OK status.
        assert len(fake_lib.string_free_calls) == 2


def test_synthesize(fake_lib: FakeLib, tmp_model: str) -> None:
    """``synthesize`` returns (pcm_list, sample_rate) and frees the buffer.

    Verifies:
    * UTF-8 text is passed through verbatim (round-trips non-ASCII).
    * The returned tuple is ``(list[float], int)`` — numpy-optional.
    * The Vokra-owned PCM buffer is freed exactly once via
      ``vokra_audio_free`` with the matching sample count.
    * Non-OK status is re-raised verbatim as the matching
      ``VokraError`` subclass (FR-EX-08).
    """
    from vokra.errors import VokraBackendUnavailableError

    with Session.open(tmp_model, lib=fake_lib) as sess:
        # -- happy path: verify samples + sample rate + free --------------
        fake_lib.synthesize_pcm = [0.0, 0.25, -0.5, 0.75, 1.0, -1.0]
        fake_lib.synthesize_sample_rate = 22050
        pcm, sr = sess.synthesize("hello vokra")
        assert isinstance(pcm, list)
        assert sr == 22050
        assert len(pcm) == 6
        # Verify sample fidelity — the marshalling round-trip must be
        # exact for f32 values that are representable (powers of 2).
        assert pcm == pytest.approx([0.0, 0.25, -0.5, 0.75, 1.0, -1.0])
        # Text encoded as UTF-8, NUL-terminated by ctypes.c_char_p
        # semantics; FakeLib records raw bytes.
        assert fake_lib.synthesize_calls == [b"hello vokra"]
        # PCM buffer freed exactly once with the matching sample count.
        assert fake_lib.audio_free_calls == [6]

        # -- non-ASCII round-trips via UTF-8 ------------------------------
        fake_lib.synthesize_pcm = [0.1, -0.1]
        pcm2, sr2 = sess.synthesize("こんにちは")
        assert sr2 == 22050
        assert len(pcm2) == 2
        # UTF-8 encoding of the JA greeting.
        assert fake_lib.synthesize_calls[-1] == "こんにちは".encode("utf-8")
        assert fake_lib.audio_free_calls == [6, 2]

        # -- non-OK status raises verbatim (FR-EX-08) ---------------------
        # 4 == VOKRA_ERROR_BACKEND_UNAVAILABLE — the caller must see the
        # typed error, NOT a silent CPU fallback.
        fake_lib.synthesize_status = 4
        fake_lib.last_error_bytes = b"metal backend not built"
        with pytest.raises(VokraBackendUnavailableError) as exc_info:
            sess.synthesize("boom")
        assert "metal backend not built" in str(exc_info.value)
        # No spurious free on the error path — all three out-params are
        # untouched on non-OK status per the C ABI contract.
        assert fake_lib.audio_free_calls == [6, 2]


# --- Stream push / poll / poll_events (T11) --------------------------------
#
# T11 extends the Stream surface with caller-driven push/poll/poll_events —
# no CFUNCTYPE, no background thread. These tests exercise:
#   * exact payload round-tripping across the ctypes boundary
#   * caller-supplied capacity is honoured and out_produced is respected
#   * status != OK from push / poll / poll_events becomes RuntimeError
#   * Event triple exposes kind/a/b positionally and by attribute
#   * poll on a closed stream raises via raw()'s UAF guard
#   * negative capacity is rejected before the FFI call
#   * zero capacity round-trips as an empty list without touching the ABI
#   * broken runtime that reports produced > capacity is caught defensively


def _open_stream(stream_lib: FakeStreamLib, tmp_model: str, srate: int = 16000):
    """Small helper: open a live Session + Stream against the fake lib."""
    stream_lib.next_stream_handle = 0xB0B00000
    session = Session.open(tmp_model, lib=stream_lib)
    s = Stream(session, sample_rate_hz=srate, lib=stream_lib)
    return session, s


def test_stream_push_roundtrips_payload(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    s.push([0.1, -0.2, 0.3])
    s.push([0.4, 0.5])
    # Empty push is allowed and shows up as a zero-length chunk so the
    # runtime can observe heartbeats.
    s.push([])
    assert stream_lib.push_calls == [
        pytest.approx([0.1, -0.2, 0.3]),
        pytest.approx([0.4, 0.5]),
        [],
    ]
    s.close()
    session.close()


def test_stream_poll_drains_pending_samples(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    s.push([1.0, 2.0, 3.0, 4.0, 5.0])
    # First poll takes only 2 — the rest stays queued for the next call.
    first = s.poll(capacity=2)
    assert first == pytest.approx([1.0, 2.0])
    # Second poll drains the remainder.
    rest = s.poll(capacity=64)
    assert rest == pytest.approx([3.0, 4.0, 5.0])
    # No samples left — polling now returns an empty list, not an error.
    assert s.poll(capacity=64) == []
    s.close()
    session.close()


def test_stream_poll_zero_capacity_is_noop(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    s.push([9.0])
    # capacity=0 short-circuits before the FFI call; the sample stays
    # queued for a subsequent poll.
    assert s.poll(capacity=0) == []
    assert stream_lib.pending_pcm == [9.0]
    assert s.poll(capacity=1) == pytest.approx([9.0])
    s.close()
    session.close()


def test_stream_poll_negative_capacity_raises(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    with pytest.raises(ValueError, match="capacity"):
        s.poll(capacity=-1)
    with pytest.raises(ValueError, match="capacity"):
        s.poll_events(capacity=-4)
    s.close()
    session.close()


def test_stream_push_status_failure_raises(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    stream_lib._init_pushpoll()
    stream_lib.push_status = 5  # VOKRA_ERROR_INVALID_ARGUMENT
    with pytest.raises(RuntimeError, match="vokra_stream_push_pcm failed"):
        s.push([0.1, 0.2])
    s.close()
    session.close()


def test_stream_poll_status_failure_raises(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    stream_lib._init_pushpoll()
    stream_lib.poll_status = 9  # VOKRA_ERROR_OTHER
    with pytest.raises(RuntimeError, match="vokra_stream_poll failed"):
        s.poll(capacity=8)
    stream_lib.poll_events_status = 9
    with pytest.raises(RuntimeError, match="vokra_stream_poll_events failed"):
        s.poll_events(capacity=8)
    s.close()
    session.close()


def test_stream_poll_defends_against_bogus_produced(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    """A broken runtime reporting produced > capacity must not slice OOB."""
    session, s = _open_stream(stream_lib, tmp_model)
    stream_lib._init_pushpoll()
    stream_lib.forced_produced = 999  # capacity will be 4
    s.push([1.0, 2.0])
    with pytest.raises(RuntimeError, match="produced=999 > capacity=4"):
        s.poll(capacity=4)
    s.close()
    session.close()


def test_stream_poll_events_roundtrips_kind_a_b(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    stream_lib._init_pushpoll()
    stream_lib.pending_events = [
        Event(VOKRA_EVENT_SPEECH_PROB, 0, 0.87),
        Event(VOKRA_EVENT_TOKEN, 12345, 0.0),
    ]
    out = s.poll_events(capacity=8)
    assert len(out) == 2
    # Positional unpack still works (Event is a tuple subclass).
    kind0, a0, b0 = out[0]
    assert kind0 == VOKRA_EVENT_SPEECH_PROB
    assert a0 == 0
    assert b0 == pytest.approx(0.87)
    # Attribute access mirrors the positional view.
    assert out[1].kind == VOKRA_EVENT_TOKEN
    assert out[1].a == 12345
    assert out[1].b == 0.0
    # Second drain returns empty.
    assert s.poll_events(capacity=8) == []
    s.close()
    session.close()


def test_stream_poll_after_close_raises_uaf(
    stream_lib: FakeStreamLib, tmp_model: str
) -> None:
    session, s = _open_stream(stream_lib, tmp_model)
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.push([0.0])
    with pytest.raises(RuntimeError, match="closed"):
        s.poll(capacity=4)
    with pytest.raises(RuntimeError, match="closed"):
        s.poll_events(capacity=4)
    session.close()


def test_event_repr_names_known_kinds() -> None:
    """Event.__repr__ should surface the kind name for debugging."""
    ev = Event(VOKRA_EVENT_SPEECH_PROB, 0, 0.5)
    r = repr(ev)
    assert "SPEECH_PROB" in r
    assert "a=0" in r
    tok = Event(VOKRA_EVENT_TOKEN, 42, 0.0)
    assert "TOKEN" in repr(tok)
    # An unknown kind falls back to the numeric label.
    other = Event(99, 0, 0.0)
    assert "kind=99" in repr(other)
