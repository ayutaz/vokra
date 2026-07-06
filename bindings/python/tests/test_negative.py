# SPDX-License-Identifier: Apache-2.0
"""Negative-path tests — T12 (M2-12 Python bindings).

Covers the failure modes the T08–T10 wrappers must reject explicitly,
without silent fallbacks (FR-EX-08) and without reaching the C ABI with
NULL / invalid arguments (which would be UB in C). The three families:

1. **Invalid GGUF paths** — ``Session.open`` guards ``os.path.exists``
   *before* the FFI call so the runtime never sees a missing file, and
   propagates C-ABI-reported I/O failures (``VOKRA_ERROR_IO``) verbatim.
   We check both branches: pre-FFI ``FileNotFoundError`` on absent files
   and the ``RuntimeError`` wrapper when the native side reports failure.

2. **Double-free rejection** — the ``Handle`` base clears its raw pointer
   after ``_release`` so ``vokra_session_destroy`` is invoked at most
   once even across ``__exit__`` + ``close()`` + ``__del__`` overlap.
   ``raw()`` becomes the UAF gate: any post-close operation
   (``transcribe``, ``synthesize``, re-entering the ``with`` block,
   opening a ``Stream``) raises ``RuntimeError("closed")`` before any
   FFI call. Test smoke lives in ``test_session.py``; here we focus on
   the *combinations* that pathological callers actually hit.

3. **Oversized PCM buffers** — ``transcribe`` marshals a Python sequence
   into ``(c_float * n)`` and passes ``n`` as ``c_size_t``. The wrapper
   itself imposes no length cap (the C ABI decides), so we assert that
   the length and pointer that reach the fake native side match what the
   wrapper allocated, and that a runtime-reported failure surfaces as a
   typed ``VokraError`` rather than being swallowed. This locks in the
   contract that oversized inputs are the *runtime's* decision, never a
   silent truncation on the Python side.

Zero-dep (NFR-DS-02): only stdlib + pytest. No numpy, no pyo3, no cffi.
LC_NUMERIC-safe (NFR-RL-01): no ``float(str)`` / ``strtod`` on the path.
The tests use the same ``FakeLib`` pattern as ``test_session.py`` so we
never need a prebuilt ``libvokra`` to run them — CI's
``python-wheel-build`` job (D12) is where the real library is exercised.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path
from typing import Optional

import pytest

# Make the src/ layout importable without ``pip install -e .``. Same
# shim as the sibling tests; keeps a fresh clone runnable with just
# ``pytest`` on PATH.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vokra import _bindings as _b  # noqa: E402
from vokra.errors import (  # noqa: E402
    VokraError,
    VokraInvalidArgumentError,
    VokraIoError,
    VokraModelLoadError,
)
from vokra.session import Session, _VOKRA_OK  # noqa: E402
from vokra.stream import Stream  # noqa: E402


# ---------------------------------------------------------------------------
# Fake native library — mirrors the shape used by test_session.py so the
# negative-path tests can share intent (recording calls) without pulling
# in an actual libvokra. Kept local (rather than a shared fixture module)
# to keep this file self-contained per the "one file per change" rule.
# ---------------------------------------------------------------------------


class FakeLib:
    """Minimal stub CDLL covering the negative-path surface.

    Records exactly which symbols were reached so tests can prove that
    the wrapper rejected bad input *before* the C ABI (the interesting
    contract) or that the C ABI's own rejection was propagated
    faithfully. The ``next_*`` cursors are set per-test to script the
    return value of each call.
    """

    def __init__(self) -> None:
        # Session lifecycle recording
        self.create_calls: list[tuple[bytes, int]] = []
        self.destroy_calls: list[int] = []

        # ASR call recording — (handle, num_samples, sample_rate).
        # ``ptr_nonnull`` captures whether we saw a real buffer or NULL,
        # which is how the wrapper signals "empty PCM" per session.py.
        self.transcribe_calls: list[tuple[int, int, int, bool]] = []

        # Scripted return values / handles / error messages.
        self.next_status: int = _VOKRA_OK
        self.next_handle: int = 0xA55E7100
        self.next_transcribe_status: int = _VOKRA_OK
        self.next_transcribe_text: bytes = b"ok"
        self.last_error_bytes: bytes = b""

        # Track the last size the wrapper asked us to free so we can
        # assert alloc/free symmetry from the Python side. Not consulted
        # for the negative paths (all fail before the audio_free call)
        # but useful if a future test extends the same fake.
        self.audio_free_calls: list[tuple[int, int]] = []
        self.string_free_calls: list[int] = []

    # -- vokra_session_create_from_file ---------------------------------

    def vokra_session_create_from_file(self, path_bytes, out_pp):
        # Record the path the wrapper passed so tests can assert utf-8
        # encoding, no truncation on non-ASCII, etc.
        self.create_calls.append((path_bytes, self.next_status))
        if self.next_status != _VOKRA_OK:
            # Real ABI leaves ``*out`` untouched on failure; mirror that
            # so a wrapper bug that reads ``out.value`` even on error
            # would surface as a stale pointer, not a fresh sentinel.
            return self.next_status
        out_pp._obj.value = self.next_handle
        return _VOKRA_OK

    def vokra_session_destroy(self, handle):
        val = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.destroy_calls.append(val)

    def vokra_last_error(self):
        # Return ``None`` on empty so the wrapper's ``if last:`` check
        # gives the same result as the real C ABI (NULL vs empty
        # c_char_p) — the wrapper decodes with ``errors='replace'``.
        return self.last_error_bytes or None

    # -- vokra_asr_transcribe -------------------------------------------

    def vokra_asr_transcribe(self, handle, pcm_ptr, num_samples, sample_rate, out_text_pp):
        # ``pcm_ptr`` is a POINTER(c_float); coerce to a bool via the
        # ctypes truthiness protocol (NULL -> False) rather than a raw
        # int cast which would need addressof().
        ptr_nonnull = bool(pcm_ptr)
        # ctypes scalars (c_size_t / c_int32) don't satisfy int() directly
        # in every CPython version — go through .value which is guaranteed
        # to return a plain Python int for numeric ctypes types. Handles
        # come through as c_void_p, so use .value there too.
        h_val = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        n_val = num_samples.value if hasattr(num_samples, "value") else int(num_samples)
        sr_val = sample_rate.value if hasattr(sample_rate, "value") else int(sample_rate)
        self.transcribe_calls.append(
            (int(h_val or 0), int(n_val), int(sr_val), ptr_nonnull)
        )
        if self.next_transcribe_status != _VOKRA_OK:
            return self.next_transcribe_status
        # Success: hand back a heap-owned c_char_p. We stash the bytes on
        # ``self`` so the wrapper's ``vokra_string_free`` sees the same
        # allocation the C ABI would have handed out.
        out_text_pp._obj.value = self.next_transcribe_text
        return _VOKRA_OK

    def vokra_string_free(self, ptr):
        # ``ptr`` arrives as a c_char_p; we cannot recover a stable id
        # from it (Python may have already dropped the bytes object), so
        # just record the fact of the free.
        self.string_free_calls.append(id(ptr))

    def vokra_audio_free(self, ptr, num_samples):
        self.audio_free_calls.append((id(ptr), int(num_samples)))


@pytest.fixture()
def fake_lib() -> FakeLib:
    return FakeLib()


@pytest.fixture()
def tmp_model(tmp_path: Path) -> str:
    """Create an empty placeholder file to satisfy the pre-FFI guard.

    ``Session.open`` checks ``os.path.exists`` before the FFI call so the
    file must actually exist on disk; contents are irrelevant since the
    FakeLib never opens it.
    """
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\x00")
    return str(p)


# ---------------------------------------------------------------------------
# 1. Invalid GGUF paths
# ---------------------------------------------------------------------------
#
# Two orthogonal branches:
#   (a) pre-FFI guard: ``os.path.exists`` returns False -> FileNotFoundError
#       raised *before* the C ABI is reached, so no destroy is scheduled.
#   (b) FFI branch:    the file exists but the runtime reports I/O failure
#       (VOKRA_ERROR_IO); the wrapper propagates the thread-local error
#       message verbatim and does *not* schedule a destroy (there is no
#       handle to release when create failed).


def test_missing_gguf_path_raises_before_ffi(fake_lib: FakeLib, tmp_path: Path) -> None:
    """A path that does not exist raises ``FileNotFoundError`` before FFI.

    The guard is intentionally in Python so the C ABI never has to
    handle a missing file — this also keeps ``vokra_last_error()`` out
    of the picture for the trivial "typo in the CLI" case.
    """
    missing = tmp_path / "no-such-model.gguf"
    with pytest.raises(FileNotFoundError, match="no-such-model.gguf"):
        Session.open(str(missing), lib=fake_lib)
    # Guard fires before any FFI call — nothing to record, nothing to free.
    assert fake_lib.create_calls == []
    assert fake_lib.destroy_calls == []


def test_directory_path_raises_before_ffi(fake_lib: FakeLib, tmp_path: Path) -> None:
    """Passing a directory path (real, but not a regular file) fails cleanly.

    ``os.path.exists`` is True on a directory, so this exercises the
    other side of the guard: if the runtime ever grows a "directory
    treated as file" bug, the wrapper still refuses with a typed error
    from the FFI rather than dereferencing a bad handle.

    We script the FakeLib to report ``VOKRA_ERROR_IO`` so this reflects
    the real runtime behaviour on such input.
    """
    fake_lib.next_status = _b.VOKRA_ERROR_IO
    fake_lib.last_error_bytes = b"expected regular file, got directory"
    # The directory itself must exist so ``os.path.exists`` returns True.
    with pytest.raises(RuntimeError, match="expected regular file"):
        Session.open(str(tmp_path), lib=fake_lib)
    # FFI *was* reached (dir passes the pre-FFI guard) but no handle was
    # produced, so destroy must not have been scheduled.
    assert len(fake_lib.create_calls) == 1
    assert fake_lib.destroy_calls == []


def test_empty_path_raises_before_ffi(fake_lib: FakeLib) -> None:
    """An empty string path fails the pre-FFI guard cleanly.

    ``os.path.exists("")`` returns False on every supported platform
    (Linux/macOS/Windows), so the wrapper rejects with FileNotFoundError
    without ever calling into the native library. This is the "silent
    default" trap: without the guard, some ctypes implementations pass
    ``b""`` as ``const char *`` which some libc opens as the CWD.
    """
    with pytest.raises(FileNotFoundError):
        Session.open("", lib=fake_lib)
    assert fake_lib.create_calls == []
    assert fake_lib.destroy_calls == []


def test_non_ascii_path_is_utf8_encoded_at_ffi(fake_lib: FakeLib, tmp_path: Path) -> None:
    """Non-ASCII paths reach the C ABI as UTF-8, not as native locale.

    Guards against a regression where the wrapper would use
    ``os.fsencode`` (locale-dependent on POSIX) instead of an explicit
    ``.encode("utf-8")``. This is the LC_NUMERIC-adjacent trap the C ABI
    is exposed to whenever a caller has ``LANG=C`` set.

    The FakeLib fails the create so we exercise the negative branch of
    a legitimate non-ASCII path, matching this file's theme.
    """
    non_ascii = tmp_path / "モデル.gguf"  # モデル.gguf
    non_ascii.write_bytes(b"\x00")
    fake_lib.next_status = _b.VOKRA_ERROR_MODEL_LOAD
    fake_lib.last_error_bytes = b"bad GGUF magic"
    with pytest.raises(RuntimeError, match="bad GGUF magic"):
        Session.open(str(non_ascii), lib=fake_lib)
    # Assert the encoding contract: the wrapper handed us UTF-8 bytes,
    # not the platform's default locale encoding. The exact expected
    # bytes are stable across all supported OS.
    assert len(fake_lib.create_calls) == 1
    seen_path_bytes = fake_lib.create_calls[0][0]
    assert seen_path_bytes == str(non_ascii).encode("utf-8"), (
        "Session.open must UTF-8 encode paths for the C ABI, regardless "
        "of the host locale (NFR-RL-01)."
    )
    assert fake_lib.destroy_calls == []


def test_ffi_reported_model_load_error_typed_by_t09(fake_lib: FakeLib, tmp_model: str) -> None:
    """A ``VOKRA_ERROR_MODEL_LOAD`` from the C ABI surfaces as a typed error.

    T09 wired ``raise_from_status`` to map every ``vokra_status_t`` to a
    matching ``VokraError`` subclass. ``Session.open`` (still on T08's
    ``RuntimeError`` path pending T09 refinement of the open path) uses
    ``raise_from_status`` indirectly via the higher-level wrappers, but
    the status-to-class mapping is exhaustive so we lock it in here from
    the caller's perspective: when the C ABI reports a specific status,
    a downstream ``raise_from_status`` call in the same session lifetime
    produces the *matching* subclass, not the base ``VokraError``.

    We drive this through ``transcribe`` (T10) which is the first API
    fully wired to ``raise_from_status``, so we open a session
    successfully then script the runtime to fail the ASR call. This
    catches a regression where the errors map might silently degrade to
    ``VokraOtherError`` on a valid status.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    try:
        fake_lib.next_transcribe_status = _b.VOKRA_ERROR_MODEL_LOAD
        fake_lib.last_error_bytes = b"invalid tokenizer chunk"
        with pytest.raises(VokraModelLoadError) as excinfo:
            sess.transcribe([0.0] * 16, sample_rate=16000)
        # Message is captured in the same call frame as the failing FFI
        # call, satisfying the thread-local ``vokra_last_error`` contract.
        assert "invalid tokenizer chunk" in str(excinfo.value)
        assert excinfo.value.status == _b.VOKRA_ERROR_MODEL_LOAD
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# 2. Double-free rejection
# ---------------------------------------------------------------------------
#
# The base ``Handle`` class already covers the trivial repeated-close
# case in ``test_session.py``. Here we assert the combinations pathological
# callers actually reach:
#   * post-close ``transcribe`` raises before the C ABI (UAF gate),
#   * post-close ``synthesize`` raises before the C ABI (UAF gate),
#   * post-close ``Stream(...)`` raises before ``vokra_stream_open``,
#   * ``__exit__`` + ``close()`` + ``__del__`` triple overlap -> destroy once.


def test_transcribe_after_close_raises_before_ffi(fake_lib: FakeLib, tmp_model: str) -> None:
    """``transcribe`` on a closed session raises before touching the C ABI.

    ``raw()`` is the single choke-point; if a future refactor caches the
    pointer in ``Session.__init__`` and re-uses it after close, this
    test fires because the fake would record a transcribe call.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        sess.transcribe([0.0] * 16, sample_rate=16000)
    # Confirm no C ABI call reached the fake — the UAF gate ran first.
    assert fake_lib.transcribe_calls == []


def test_synthesize_after_close_raises_before_ffi(fake_lib: FakeLib, tmp_model: str) -> None:
    """Same UAF gate on the TTS side.

    ``synthesize`` also routes through ``raw()`` so we assert the same
    contract for the second high-level entry point. Both branches must
    stay in lock-step: adding a new API that reads ``self._handle``
    directly without going through ``raw()`` would silently miss this
    check.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        sess.synthesize("hello")


def test_stream_open_after_session_close_raises(fake_lib: FakeLib, tmp_model: str) -> None:
    """Opening a Stream on a closed Session raises before ``vokra_stream_open``.

    A double-free bug here would be catastrophic: the runtime would see
    a NULL session pointer, and depending on the backend either crash or
    write into freed memory. We rely on ``Session.raw()`` failing first.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        Stream(sess, sample_rate_hz=16000, lib=fake_lib)


def test_triple_lifecycle_overlap_destroys_once(fake_lib: FakeLib, tmp_model: str) -> None:
    """``with`` + explicit ``close()`` + ``__del__`` = exactly one destroy.

    This is the worst-case overlap: a caller uses a ``with`` block, then
    calls ``close()`` on the (already-closed) sess reference the block
    left in scope, then lets GC finalize. The C ABI must see exactly one
    ``vokra_session_destroy`` — the base class's ``self._handle = None``
    after the first release is the invariant that guarantees this.
    """
    with Session.open(tmp_model, lib=fake_lib) as sess:
        handle_val = sess.raw().value
    # After ``with`` the wrapper is closed but still in scope.
    assert sess.closed
    sess.close()  # explicit close on already-closed sess: no-op
    sess.close()  # and again — still no-op
    # Drop the reference and force GC so ``__del__`` runs deterministically.
    del sess
    gc.collect()
    assert fake_lib.destroy_calls == [handle_val]


def test_reenter_context_after_close_raises(fake_lib: FakeLib, tmp_model: str) -> None:
    """``__enter__`` on a closed handle raises rather than replaying.

    The base class ``__enter__`` calls ``raw()`` first so a re-entered
    ``with`` on a closed handle fails loudly. Without this the second
    ``with`` block would succeed silently and the second ``__exit__``
    would no-op — hiding the caller's bug.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    sess.close()
    with pytest.raises(RuntimeError, match="closed"):
        with sess:  # pragma: no cover - body must not execute
            pass
    # Reject must have happened before any further FFI call.
    assert fake_lib.destroy_calls == [fake_lib.next_handle]


# ---------------------------------------------------------------------------
# 3. Oversized / mismatched PCM buffers
# ---------------------------------------------------------------------------
#
# The wrapper is intentionally NOT a policy layer: it does not impose
# a length cap or auto-resample. Instead it marshals whatever the caller
# passed into ``(c_float * n)`` + ``c_size_t`` and lets the C ABI decide.
# These tests assert that contract explicitly so a future "helpful"
# refactor (e.g. clamping n or normalising sample_rate) is caught.


def test_oversized_pcm_forwarded_verbatim_length(fake_lib: FakeLib, tmp_model: str) -> None:
    """Oversized PCM: length and pointer reach the C ABI unmodified.

    We pass a buffer larger than any realistic 30s @ 16kHz Whisper input
    (480_000 samples) — 10x that — and confirm the wrapper does not
    truncate on the Python side. The FakeLib scripts a success so the
    wrapper's alloc/free path also stays covered; the negative-path
    intent here is "no silent truncation", not "the runtime must accept".
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    try:
        oversized_n = 4_800_000  # 300s @ 16kHz — well over Whisper's 30s window
        pcm = [0.0] * oversized_n
        # Success path: wrapper allocates, calls the fake, decodes, frees.
        result = sess.transcribe(pcm, sample_rate=16000)
        assert result == "ok"
        assert len(fake_lib.transcribe_calls) == 1
        _handle_val, n_seen, sr_seen, ptr_nonnull = fake_lib.transcribe_calls[0]
        assert n_seen == oversized_n, (
            "Session.transcribe must forward num_samples verbatim to the "
            "C ABI; silent truncation would hide a real caller bug."
        )
        assert sr_seen == 16000
        assert ptr_nonnull, "non-empty PCM must reach the runtime as a real pointer"
        # And ``vokra_string_free`` ran on success.
        assert len(fake_lib.string_free_calls) == 1
    finally:
        sess.close()


def test_oversized_pcm_runtime_reject_is_typed_error(fake_lib: FakeLib, tmp_model: str) -> None:
    """When the runtime rejects an oversized buffer, we get a typed error.

    Real behaviour: for buffers beyond the model's supported context, the
    C ABI returns ``VOKRA_ERROR_INVALID_ARGUMENT`` with a descriptive
    message. The wrapper propagates that as ``VokraInvalidArgumentError``
    verbatim — no fallback, no truncation, no default string (FR-EX-08).
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    try:
        fake_lib.next_transcribe_status = _b.VOKRA_ERROR_INVALID_ARGUMENT
        fake_lib.last_error_bytes = b"pcm exceeds model context window"
        with pytest.raises(VokraInvalidArgumentError) as excinfo:
            sess.transcribe([0.0] * 10_000_000, sample_rate=16000)
        # Message and status are captured on the failing frame.
        assert "pcm exceeds model context window" in str(excinfo.value)
        assert excinfo.value.status == _b.VOKRA_ERROR_INVALID_ARGUMENT
        # We saw exactly one C ABI call — no retry on failure.
        assert len(fake_lib.transcribe_calls) == 1
        # And no string_free on failure (out_text was untouched per ABI).
        assert fake_lib.string_free_calls == []
    finally:
        sess.close()


def test_empty_pcm_forwards_null_pointer(fake_lib: FakeLib, tmp_model: str) -> None:
    """Empty PCM: wrapper hands NULL + 0 to the C ABI (not an empty buffer).

    The C ABI explicitly allows ``pcm_ptr = NULL`` when ``num_samples ==
    0``; the wrapper documents this in ``session.transcribe`` docstring.
    We assert the contract here because it is the boundary where
    "oversized" flips to "undersized" — both edges must be forwarded
    verbatim to the runtime rather than short-circuited on the Python
    side (which would hide a runtime bug on the ``0``-length path).
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    try:
        # Empty list, empty tuple, and ``None`` should all funnel to the
        # same NULL + 0 call — the wrapper normalises to a single path.
        for empty_input in ([], (), None):
            fake_lib.transcribe_calls.clear()
            fake_lib.next_transcribe_text = b""
            result = sess.transcribe(empty_input, sample_rate=16000)
            assert result == ""
            assert len(fake_lib.transcribe_calls) == 1
            _h, n_seen, sr_seen, ptr_nonnull = fake_lib.transcribe_calls[0]
            assert n_seen == 0
            assert sr_seen == 16000
            assert not ptr_nonnull, (
                "Empty PCM must be forwarded as NULL + 0; a non-NULL "
                f"empty buffer is a spec violation (input={empty_input!r})."
            )
    finally:
        sess.close()


def test_negative_sample_rate_reaches_runtime_verbatim(
    fake_lib: FakeLib, tmp_model: str
) -> None:
    """Negative sample_rate is *not* clamped on the Python side.

    The C ABI treats sample_rate as ``int32_t`` and expects the runtime
    to validate. The wrapper passes the caller's value through
    ``ctypes.c_int32(int(sample_rate))`` without any range check. A
    "helpful" refactor that clamps to unsigned would hide a caller bug
    (typo, unsigned overflow) — we lock in the contract that whatever
    the caller passed reaches the runtime unchanged, and the runtime's
    ``VOKRA_ERROR_INVALID_ARGUMENT`` is propagated as a typed error.
    """
    sess = Session.open(tmp_model, lib=fake_lib)
    try:
        fake_lib.next_transcribe_status = _b.VOKRA_ERROR_INVALID_ARGUMENT
        fake_lib.last_error_bytes = b"sample_rate must be positive"
        with pytest.raises(VokraInvalidArgumentError, match="sample_rate must be positive"):
            sess.transcribe([0.0] * 16, sample_rate=-16000)
        assert len(fake_lib.transcribe_calls) == 1
        # The wrapper forwarded the caller's value as-is (ctypes.c_int32
        # sign-extends within the 32-bit range).
        _h, n_seen, sr_seen, _ptr_nonnull = fake_lib.transcribe_calls[0]
        assert n_seen == 16
        assert sr_seen == -16000
    finally:
        sess.close()
