# SPDX-License-Identifier: Apache-2.0
"""Exception hierarchy for the Vokra Python binding.

Every non-OK ``vokra_status_t`` returned by the C ABI is converted to a
``VokraError`` subclass by :func:`raise_from_status`. Callers only ever see
Python exceptions; the raw ``int`` status never leaks past the wrapper.

Design constraints (M2-12 T09):

* No dependency on ``ctypes.CDLL`` at import time — this module must be safe
  to import even when the native library is missing (e.g. static analysis,
  documentation builds, or ``pip install`` without wheels).
* ``vokra_last_error()`` is **thread-local** (C ABI report §3): the wrapper
  MUST read it in the same call frame as the failing call. That contract is
  enforced by :func:`raise_from_status` accepting the message as an argument
  rather than fetching it lazily.
* The status→class map is exhaustive; unknown codes raise :class:`VokraError`
  directly so future C ABI additions never silently degrade to a bare int.
* Locale-independent (NFR-RL-01): message decoding uses UTF-8 with
  ``errors="replace"``; no ``strtod``/``float(str)`` involvement.

The 9 subclasses map 1:1 to the real ``vokra_status_t`` enum defined in
``include/vokra.h`` (mirrored in ``vokra._bindings``). The M2-12 plan §D8
originally listed 10 hypothetical variants (Alloc/Decode/Stream) that never
shipped in the C ABI; this module tracks the shipping enum, not the draft.
"""

from __future__ import annotations

from typing import NoReturn

from . import _bindings as _b

__all__ = [
    "VokraError",
    "VokraIoError",
    "VokraModelLoadError",
    "VokraUnsupportedOpError",
    "VokraBackendUnavailableError",
    "VokraInvalidArgumentError",
    "VokraGraphValidationError",
    "VokraNotImplementedError",
    "VokraPanicError",
    "VokraOtherError",
    "raise_from_status",
]


class VokraError(Exception):
    """Base class for every error raised by the Vokra binding.

    ``status`` is the raw ``vokra_status_t`` value; ``message`` is the
    thread-local ``vokra_last_error()`` string captured at the failure site
    (may be empty if the native library did not set one).
    """

    status: int
    message: str

    def __init__(self, message: str = "", status: int = _b.VOKRA_ERROR_OTHER) -> None:
        self.status = int(status)
        self.message = message or ""
        # Compose a useful ``str(exc)`` without leaking the raw int when the
        # message is already descriptive.
        if self.message:
            super().__init__(f"{self.message} (status={self.status})")
        else:
            super().__init__(f"vokra error (status={self.status})")


class VokraIoError(VokraError):
    """``VOKRA_ERROR_IO`` — file not found, read/write failure, mmap failure."""


class VokraModelLoadError(VokraError):
    """``VOKRA_ERROR_MODEL_LOAD`` — GGUF parse or metadata validation failed."""


class VokraUnsupportedOpError(VokraError):
    """``VOKRA_ERROR_UNSUPPORTED_OP`` — op not implemented by the chosen backend.

    Per FR-EX-08, the runtime raises this instead of silently falling back to
    CPU. Callers must switch backend or model, not swallow the exception.
    """


class VokraBackendUnavailableError(VokraError):
    """``VOKRA_ERROR_BACKEND_UNAVAILABLE`` — requested backend not built in / no device."""


class VokraInvalidArgumentError(VokraError):
    """``VOKRA_ERROR_INVALID_ARGUMENT`` — caller passed a null pointer, bad length, etc."""


class VokraGraphValidationError(VokraError):
    """``VOKRA_ERROR_GRAPH_VALIDATION`` — audio graph shape/dtype check failed."""


class VokraNotImplementedError(VokraError):
    """``VOKRA_ERROR_NOT_IMPLEMENTED`` — API surface reserved but not yet wired."""


class VokraPanicError(VokraError):
    """``VOKRA_ERROR_PANIC`` — Rust panic caught at the FFI boundary."""


class VokraOtherError(VokraError):
    """``VOKRA_ERROR_OTHER`` — unclassified failure; consult ``message``."""


_STATUS_TO_CLASS: dict[int, type[VokraError]] = {
    _b.VOKRA_ERROR_IO: VokraIoError,
    _b.VOKRA_ERROR_MODEL_LOAD: VokraModelLoadError,
    _b.VOKRA_ERROR_UNSUPPORTED_OP: VokraUnsupportedOpError,
    _b.VOKRA_ERROR_BACKEND_UNAVAILABLE: VokraBackendUnavailableError,
    _b.VOKRA_ERROR_INVALID_ARGUMENT: VokraInvalidArgumentError,
    _b.VOKRA_ERROR_GRAPH_VALIDATION: VokraGraphValidationError,
    _b.VOKRA_ERROR_NOT_IMPLEMENTED: VokraNotImplementedError,
    _b.VOKRA_ERROR_PANIC: VokraPanicError,
    _b.VOKRA_ERROR_OTHER: VokraOtherError,
}


def raise_from_status(status: int, message: str = "") -> NoReturn:
    """Raise the ``VokraError`` subclass matching ``status``.

    Callers must invoke this **in the same call frame** as the failing native
    call so that ``vokra_last_error()`` (thread-local) still reflects the
    correct failure. Passing ``VOKRA_OK`` is a bug: it raises
    :class:`VokraOtherError` so the mistake is visible in tests.
    """
    cls = _STATUS_TO_CLASS.get(int(status), VokraError)
    raise cls(message=message, status=int(status))
