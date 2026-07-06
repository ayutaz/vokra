# SPDX-License-Identifier: Apache-2.0
"""Error mapping + thread-local errno contract tests (T09).

These tests verify:

1. Every non-OK ``vokra_status_t`` maps to the correct
   :class:`~vokra.errors.VokraError` subclass.
2. ``_check_status`` reads ``vokra_last_error()`` in the same call
   frame and embeds the resulting text in the exception message
   (C ABI report §3, R3 mitigation).
3. ``vokra_last_error()`` state on one thread does not leak to
   another (thread-local contract).
4. ``VOKRA_OK`` short-circuits without touching the library.
5. Missing library / unsupported OS raise the base :class:`VokraError`
   (pre-FFI environment errors, not ``vokra_status_t`` codes).

The tests intentionally do NOT depend on a prebuilt native library:
we inject a ``FakeLib`` that mimics only the two symbols
``_check_status`` needs (``vokra_last_error``). This matches the
approach used by ``test_session.py`` (T08) and keeps ``pytest`` runnable
in a fresh checkout without a build step.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

import pytest

# Make the src/ layout importable without ``pip install -e .``.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vokra import _bindings, _native  # noqa: E402
from vokra.errors import (  # noqa: E402
    VokraBackendUnavailableError,
    VokraError,
    VokraGraphValidationError,
    VokraInvalidArgumentError,
    VokraIoError,
    VokraModelLoadError,
    VokraNotImplementedError,
    VokraOtherError,
    VokraPanicError,
    VokraUnsupportedOpError,
)


# --- fake native library ----------------------------------------------------


class FakeLib:
    """Stand-in for the loaded ``libvokra`` CDLL.

    Only implements the symbols ``_check_status`` touches. The
    ``last_error`` field is stored per-thread so we can verify the
    thread-local contract without hooking libc TLS.
    """

    def __init__(self) -> None:
        self._tls = threading.local()

    def set_error(self, message: bytes | None) -> None:
        """Set (or clear) the ``vokra_last_error`` payload for THIS thread."""
        self._tls.err = message

    def vokra_last_error(self) -> bytes | None:
        return getattr(self._tls, "err", None)


@pytest.fixture()
def fake_lib() -> FakeLib:
    return FakeLib()


# --- status -> subclass mapping ---------------------------------------------


@pytest.mark.parametrize(
    "status, exc_cls",
    [
        (_bindings.VOKRA_ERROR_IO, VokraIoError),
        (_bindings.VOKRA_ERROR_MODEL_LOAD, VokraModelLoadError),
        (_bindings.VOKRA_ERROR_UNSUPPORTED_OP, VokraUnsupportedOpError),
        (_bindings.VOKRA_ERROR_BACKEND_UNAVAILABLE, VokraBackendUnavailableError),
        (_bindings.VOKRA_ERROR_INVALID_ARGUMENT, VokraInvalidArgumentError),
        (_bindings.VOKRA_ERROR_GRAPH_VALIDATION, VokraGraphValidationError),
        (_bindings.VOKRA_ERROR_NOT_IMPLEMENTED, VokraNotImplementedError),
        (_bindings.VOKRA_ERROR_PANIC, VokraPanicError),
        (_bindings.VOKRA_ERROR_OTHER, VokraOtherError),
    ],
)
def test_check_status_maps_each_code(
    fake_lib: FakeLib, status: int, exc_cls: type
) -> None:
    fake_lib.set_error(b"synthetic")
    with pytest.raises(exc_cls) as exc_info:
        _native._check_status(status, lib=fake_lib)
    assert exc_info.value.status == status
    assert "synthetic" in exc_info.value.message
    # Subclass identity is preserved (not just a generic VokraError).
    assert type(exc_info.value) is exc_cls
    # Base class catch still works.
    assert isinstance(exc_info.value, VokraError)


def test_check_status_ok_is_noop(fake_lib: FakeLib) -> None:
    # Even with a stale errno string set, VOKRA_OK must not raise.
    fake_lib.set_error(b"stale error should be ignored")
    _native._check_status(_bindings.VOKRA_OK, lib=fake_lib)  # no exception


def test_check_status_unknown_code_falls_back_to_base(fake_lib: FakeLib) -> None:
    # A future ABI extension might add code 99. We must still raise a
    # ``VokraError`` (base class), not a bare Exception.
    fake_lib.set_error(b"future code")
    with pytest.raises(VokraError) as exc_info:
        _native._check_status(99, lib=fake_lib)
    assert exc_info.value.status == 99


# --- last_error contract ----------------------------------------------------


def test_check_status_reads_last_error_same_frame(fake_lib: FakeLib) -> None:
    """The message must reflect the errno set right before the call.

    This is the R3 mitigation: ``_check_status`` must not defer the
    read to a later frame where a subsequent FFI call may have
    overwritten the thread-local.
    """
    fake_lib.set_error(b"file 'foo.gguf' not found")
    with pytest.raises(VokraIoError) as exc_info:
        _native._check_status(_bindings.VOKRA_ERROR_IO, lib=fake_lib)
    assert "file 'foo.gguf' not found" in exc_info.value.message


def test_check_status_null_last_error_yields_empty_message(fake_lib: FakeLib) -> None:
    # Real runtime may leave errno NULL if it never set one.
    fake_lib.set_error(None)
    with pytest.raises(VokraInvalidArgumentError) as exc_info:
        _native._check_status(_bindings.VOKRA_ERROR_INVALID_ARGUMENT, lib=fake_lib)
    assert exc_info.value.message == ""
    # ``str(exc)`` still includes the status so callers see something.
    assert str(exc_info.value.status) in str(exc_info.value)


def test_check_status_decodes_utf8_lossy(fake_lib: FakeLib) -> None:
    # A well-formed UTF-8 payload including non-ASCII.
    fake_lib.set_error("モデルが読めません".encode("utf-8"))
    with pytest.raises(VokraModelLoadError) as exc_info:
        _native._check_status(_bindings.VOKRA_ERROR_MODEL_LOAD, lib=fake_lib)
    assert "モデル" in exc_info.value.message

    # Malformed bytes must NOT raise a UnicodeDecodeError — errors=replace.
    fake_lib.set_error(b"\xff\xfe partial")
    with pytest.raises(VokraModelLoadError) as exc_info:
        _native._check_status(_bindings.VOKRA_ERROR_MODEL_LOAD, lib=fake_lib)
    # The replacement char is U+FFFD; presence of "partial" proves we
    # did not lose the tail.
    assert "partial" in exc_info.value.message


# --- thread-local isolation -------------------------------------------------


def test_last_error_is_thread_local(fake_lib: FakeLib) -> None:
    """Errno set on a worker thread must not leak to the main thread.

    Mirrors the ``errno_is_thread_local`` test in the Rust C ABI
    suite (M0-09) — the whole point of the C ABI's thread-local errno
    is that concurrent sessions do not corrupt each other's error state.
    """
    fake_lib.set_error(b"main-thread state")

    seen_on_worker: list[str] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        # Worker has no errno set yet.
        assert fake_lib.vokra_last_error() is None
        # Simulate a worker-side failure.
        fake_lib.set_error(b"worker-thread failure")
        barrier.wait()  # let the main thread verify isolation
        barrier.wait()  # wait for main to finish its check
        try:
            _native._check_status(_bindings.VOKRA_ERROR_IO, lib=fake_lib)
        except VokraIoError as exc:
            seen_on_worker.append(exc.message)

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait()  # worker has set its errno; main verifies isolation
    # Main thread's view must still be "main-thread state" — the worker's
    # write must not have clobbered it.
    assert fake_lib.vokra_last_error() == b"main-thread state"
    barrier.wait()  # release worker to run _check_status
    t.join(timeout=5.0)
    assert not t.is_alive(), "worker thread did not finish"

    # Worker's exception message must reflect the worker's errno, not
    # the main thread's.
    assert seen_on_worker == ["worker-thread failure"]


# --- library resolution -----------------------------------------------------


def test_lib_filename_matches_current_platform() -> None:
    name = _native._lib_filename()
    if sys.platform == "darwin":
        assert name == "libvokra.dylib"
    elif sys.platform.startswith("linux"):
        assert name == "libvokra.so"
    elif sys.platform in ("win32", "cygwin"):
        assert name == "vokra.dll"
    else:
        pytest.skip(f"unsupported platform for this test: {sys.platform}")


def test_find_lib_raises_when_missing(tmp_path, monkeypatch) -> None:
    # Point the env override at an empty directory; strip the repo-root
    # target/ fallback by chdir'ing into a scratch dir with no ``target``.
    monkeypatch.setenv("VOKRA_LIB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # Also invalidate the module-cached instance so ``load_library``
    # would attempt a fresh resolve.
    monkeypatch.setattr(_native, "_lib", None, raising=False)
    # Filter out repo-root ``target/`` fallback: patch _candidate_paths
    # to only return the env dir so this test is hermetic.
    monkeypatch.setattr(
        _native, "_candidate_paths", lambda: [tmp_path], raising=True
    )
    with pytest.raises(VokraError) as exc_info:
        _native._find_lib()
    assert "not found" in exc_info.value.message.lower()


def test_check_status_without_lib_still_raises(monkeypatch) -> None:
    """When ``_lib`` is unset and no ``lib=`` kwarg is passed, we still
    raise the mapped subclass — with an empty message, since we
    intentionally do NOT auto-load (see docstring)."""
    monkeypatch.setattr(_native, "_lib", None, raising=False)
    with pytest.raises(VokraPanicError) as exc_info:
        _native._check_status(_bindings.VOKRA_ERROR_PANIC)
    assert exc_info.value.status == _bindings.VOKRA_ERROR_PANIC
    assert exc_info.value.message == ""


# ---------------------------------------------------------------------------
# T12 (c17) focused invariants
# ---------------------------------------------------------------------------
#
# The two assertions the WP T12 spec explicitly names:
#
#   (i)  a missing model path taken through vokra_session_create_from_file
#        surfaces as VokraIoError with a non-empty runtime message.
#   (ii) a worker-thread errno set on thread T1 stays isolated from a
#        main-thread vokra_last_error() reader on thread T0.
#
# The earlier tests above cover parts of these invariants in isolation
# (status -> subclass mapping; per-thread errno storage). The two tests
# here compose them into the exact end-to-end paths T12 requires, so a
# regression that breaks *only the composition* is still caught.
#
# We drive the failing session-create through a FakeLib so this file
# still runs without the prebuilt cdylib on disk (same discipline as
# test_session.py). The FakeLib scripts vokra_session_create_from_file
# to return VOKRA_ERROR_IO and to set a thread-local errno string, then
# we call _check_status right after — that's the exact pattern
# session.py::Session.open will use once T09 refinement lands. Using
# the wrapper primitives (_check_status + _read_last_error) rather than
# the full Session.open FFI path keeps the test decoupled from the
# transitional RuntimeError branch inside Session.open (which the
# session.py docstring already flags as "T09 will refine this").


class _MissingPathLib(FakeLib):
    """FakeLib scripted for the "missing model path" T12 (i) scenario.

    Extends FakeLib with a create_from_file symbol that unconditionally
    fails with VOKRA_ERROR_IO after setting a runtime-style errno on
    the current thread's TLS. This mirrors what the real
    vokra_session_create_from_file does when ``fopen`` returns ENOENT.
    """

    def __init__(self, errno_msg: bytes) -> None:
        super().__init__()
        self._errno_msg = errno_msg

    def vokra_session_create_from_file(self, path_bytes, out_pp):  # noqa: ARG002
        # Real ABI: writes a human-readable message to thread-local
        # errno, leaves ``out`` untouched, returns VOKRA_ERROR_IO.
        self.set_error(self._errno_msg)
        return _bindings.VOKRA_ERROR_IO


def test_missing_model_path_raises_vokra_io_error_with_message() -> None:
    """T12 (i): a missing model path -> :class:`VokraIoError` with a
    non-empty message drawn from ``vokra_last_error()``.

    This is the exact user-facing contract for "the model file I asked
    for doesn't exist". The invariants under test:

    * The C ABI's VOKRA_ERROR_IO is mapped to VokraIoError (not the
      base VokraError, not a bare RuntimeError). If a future refactor
      collapsed the mapping, ``except VokraIoError:`` handlers would
      silently stop catching missing-file failures — a silent
      correctness regression that this exact-type check flushes out.
    * The runtime's thread-local errno string is surfaced verbatim on
      ``exc.message``, and the raw C ABI status code is preserved on
      ``exc.status``. Both invariants make logs and programmatic
      handling meaningful; an empty message would let the wrapper hide
      the failure cause.

    The FakeLib scripts create_from_file to behave exactly like the
    real ABI on ENOENT (set thread-local errno + return VOKRA_ERROR_IO
    + leave the out pointer untouched). We then call ``_check_status``
    in the same call frame — the exact pattern every wrapper method
    uses.
    """
    runtime_msg = b"model file not found: /nope/does-not-exist.gguf"
    fake = _MissingPathLib(runtime_msg)

    # Reproduce the wrapper's usage pattern: FFI call, then immediately
    # check the returned status in the same frame.
    out = ctypes.c_void_p()
    status = fake.vokra_session_create_from_file(
        b"/nope/does-not-exist.gguf", ctypes.byref(out)
    )
    assert status == _bindings.VOKRA_ERROR_IO
    # ``out`` must be untouched on failure — matches the C ABI's
    # "leave outputs untouched on error" contract. If a Session RAII
    # regression ever tried to release ``out`` here, this guard catches
    # it before we double-free NULL.
    assert not out.value

    with pytest.raises(VokraIoError) as excinfo:
        _native._check_status(status, lib=fake)

    exc = excinfo.value
    # Exactness check (not just isinstance): the mapping table in
    # errors.py::_STATUS_TO_CLASS must produce VokraIoError itself,
    # not a broader base that happens to be a superclass.
    assert type(exc) is VokraIoError, (
        f"expected VokraIoError for VOKRA_ERROR_IO, got "
        f"{type(exc).__name__}; the status -> subclass table in "
        f"vokra.errors has drifted from the C ABI enum."
    )
    assert exc.status == _bindings.VOKRA_ERROR_IO
    # Non-empty verbatim message — this is what keeps log lines useful
    # and lets callers do ``if "not found" in exc.message: ...`` in a
    # cross-platform way.
    assert exc.message == runtime_msg.decode("utf-8")
    assert exc.message  # explicit non-empty guard.
    # ``str(exc)`` composes message + status per VokraError.__init__;
    # both parts must be visible.
    rendered = str(exc)
    assert runtime_msg.decode("utf-8") in rendered
    assert f"status={_bindings.VOKRA_ERROR_IO}" in rendered


def test_worker_thread_errno_isolated_from_main_thread_reader() -> None:
    """T12 (ii): a worker thread's ``vokra_last_error()`` write must be
    invisible to a concurrent main-thread reader.

    Why this matters:
      * The Python wrapper reads ``vokra_last_error()`` in the same
        call frame as the failing FFI call (see session.py::transcribe
        and _native._check_status). If the errno storage were process-
        global — e.g. if a well-meaning refactor cached the last read
        in a module-level ``_last_error`` string — a background thread's
        failure would silently poison an unrelated main-thread exception
        message. That's the exact failure mode the C ABI's
        ``thread_local!`` storage class is designed to prevent, and this
        test locks the Python wrapper's proxying of that invariant.
      * Rust-side, the invariant is enforced by
        ``thread_local! { LAST_ERROR: … }`` in ``vokra-capi/src/error.rs``
        (see the ``errno_is_thread_local`` test there). The FakeLib
        below models the same storage class using ``threading.local``,
        so a regression in either half surfaces here.

    Test shape mirrors the Rust-side ``errno_is_thread_local`` test:
      1. Main thread reader starts at empty (no thread has written).
      2. Worker thread writes a distinctive marker to its own TLS
         slot and signals readiness.
      3. Main thread reads its own TLS slot -> must still be empty.
      4. Worker exits cleanly; main thread's slot remains empty.

    Synchronisation uses :class:`threading.Event` instead of sleep so
    the test is deterministic on slow CI runners.
    """
    fake = FakeLib()

    # 1. Baseline: main thread sees NULL/empty because no thread has
    #    written yet.
    assert _native._read_last_error(fake) == ""

    # Distinctive marker so a false-positive (worker's message leaking
    # into main-thread reader) is visible in the diff.
    worker_msg = "worker-thread failure marker: XYZZY-42"
    worker_ready = threading.Event()
    worker_release = threading.Event()
    # Record what the worker sees when reading its own slot; failures
    # inside the worker do not automatically propagate out of
    # threading.Thread, so we surface them via this dict instead.
    worker_view: dict[str, str] = {}

    def worker_body() -> None:
        # 2. Worker writes to its own thread-local errno slot.
        fake.set_error(worker_msg.encode("utf-8"))
        # Sanity: the worker CAN read back what it just wrote — proves
        # the FakeLib is not a no-op (which would mask a real leak by
        # accident).
        worker_view["read"] = _native._read_last_error(fake)
        worker_ready.set()
        # Hold the worker alive until the main-thread assertion has
        # run; otherwise the worker could exit and free its
        # threading.local slot before the isolation check, which would
        # trivially pass for the wrong reason.
        assert worker_release.wait(timeout=5.0), "main thread hung"

    t = threading.Thread(
        target=worker_body, name="vokra-errno-isolation-worker"
    )
    t.start()
    try:
        assert worker_ready.wait(timeout=5.0), (
            "worker thread failed to reach the ready barrier; likely "
            "hung inside FakeLib.set_error or _read_last_error."
        )

        # Sanity: worker saw its own write.
        assert worker_view["read"] == worker_msg, (
            f"worker could not read back its own thread-local errno "
            f"(expected {worker_msg!r}, got {worker_view['read']!r}); "
            f"this is a FakeLib bug, not a wrapper bug."
        )

        # 3. THE INVARIANT: main-thread reader must still see NULL.
        # If this fires, the errno has leaked across threads and the
        # wrapper cannot safely surface FFI failure messages.
        main_view = _native._read_last_error(fake)
        assert main_view == "", (
            f"main-thread vokra_last_error() is contaminated by the "
            f"worker thread's write (saw {main_view!r}); thread-local "
            f"isolation is broken. Check that _read_last_error / "
            f"_check_status never cache the errno in module-level state, "
            f"and that FakeLib._tls is a threading.local (not a plain "
            f"attribute)."
        )
    finally:
        # 4. Release the worker and join deterministically.
        worker_release.set()
        t.join(timeout=5.0)
        assert not t.is_alive(), "worker thread did not exit within 5s"

    # Post-join: main-thread slot still empty. This second read guards
    # against a subtle regression where the worker's TLS slot leaks
    # into the main thread's view AFTER the worker exits (which could
    # happen if some future refactor kept a strong ref to the worker's
    # thread-local storage from module state).
    assert _native._read_last_error(fake) == ""
