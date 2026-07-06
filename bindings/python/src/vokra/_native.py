# SPDX-License-Identifier: Apache-2.0
"""Native library loader + status-check helpers (T09).

Responsibilities
----------------
1. Resolve the prebuilt ``libvokra`` shared object for the current
   ``(sys.platform, platform.machine())`` combination (D5/D10:
   linux x86_64/aarch64, macOS universal2, Windows x86_64).
2. Load it via ``ctypes.CDLL`` and attach the argtypes/restype declared
   in ``_bindings.PROTOTYPES`` (T07 codegen).
3. Provide :func:`_check_status` — the single choke-point every
   higher-level wrapper calls right after a ``vokra_status_t``-returning
   FFI call. ``_check_status`` reads ``vokra_last_error()`` in the
   **same Python call frame** so the thread-local errno contract
   (C ABI report §3) is respected (R3 mitigation).

Zero-dep (NFR-DS-02): only ``ctypes``, ``os``, ``pathlib``, ``platform``,
``sys``, ``threading`` — all stdlib. No numpy / no pyo3 / no cffi.

Locale-independence (NFR-RL-01): all marshaling is via ``ctypes.c_*``
primitives; we never call ``strtod`` / ``float(str)`` on any FFI
payload. See CLAUDE.md "LC_NUMERIC 罠".
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
from pathlib import Path
from typing import Optional

from . import _bindings
from .errors import VokraError, raise_from_status

__all__ = [
    "load_library",
    "load",
    "is_loaded",
    "_check_status",
    "_read_last_error",
    "_lib_filename",
    "_find_lib",
]

# ---------------------------------------------------------------------------
# Platform dispatch
# ---------------------------------------------------------------------------

# ``sys.platform`` prefix -> dylib basename. Windows exports ``vokra.dll``
# with no ``lib`` prefix (Rust's ``cdylib`` convention on MSVC); POSIX
# platforms use ``libvokra.{so,dylib}``.
_LIB_BASENAME = {
    "darwin": "libvokra.dylib",
    "linux": "libvokra.so",
    "win32": "vokra.dll",
    "cygwin": "vokra.dll",
}

# Machines we build wheels for. ``platform.machine()`` returns
# ``AMD64`` on Windows and ``x86_64`` on POSIX for the same arch, so
# we normalize both. ``arm64`` (macOS) and ``aarch64`` (Linux) are the
# same ISA under different vendor names.
_SUPPORTED_MACHINES = {
    "x86_64",
    "amd64",  # Windows spelling of x86_64
    "arm64",  # macOS spelling of aarch64
    "aarch64",  # Linux spelling
}


def _platform_key() -> str:
    """Normalized ``sys.platform`` -> {darwin, linux, win32}.

    Collapses ``linux2`` (very old CPython) into ``linux`` so the
    basename table stays small.
    """
    plat = sys.platform
    if plat.startswith("linux"):
        return "linux"
    return plat


def _lib_filename() -> str:
    """Return the expected native library filename for this host.

    Raises :class:`VokraError` when the host OS/arch pair is not in
    the supported tier-1 matrix (D10). This is a pre-FFI environment
    error, not a ``vokra_status_t``, so the base class is used.
    """
    key = _platform_key()
    name = _LIB_BASENAME.get(key)
    if name is None:
        raise VokraError(
            message=(
                f"unsupported OS for vokra native binding: sys.platform={sys.platform!r}, "
                f"machine={platform.machine()!r}. Supported: linux (x86_64/aarch64), "
                f"macOS (universal2), Windows (x86_64)."
            ),
        )
    machine = platform.machine().lower()
    if machine and machine not in _SUPPORTED_MACHINES:
        # Non-fatal warning path would hide a genuine mis-configuration;
        # fail loudly per FR-EX-08 (no silent fallback).
        raise VokraError(
            message=(
                f"unsupported CPU arch for vokra native binding: "
                f"platform.machine()={platform.machine()!r}. "
                f"Supported: {sorted(_SUPPORTED_MACHINES)!r}."
            ),
        )
    return name


def _candidate_paths() -> list[Path]:
    """Ordered directories to probe for the native library.

    Order (highest priority first):
      1. ``VOKRA_LIB_DIR`` env override — dev / CI escape hatch.
      2. Package-internal ``vokra/_lib/`` (populated by the wheel build,
         D5).
      3. Repo-relative ``target/{release,debug}/`` (source checkout /
         CI before wheel packaging).

    We do NOT search ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH``
    explicitly; passing a bare filename to ``ctypes.CDLL`` already
    delegates to the OS loader which honors those env vars. This
    helper returns explicit ``Path`` objects only.
    """
    out: list[Path] = []

    env_dir = os.environ.get("VOKRA_LIB_DIR")
    if env_dir:
        out.append(Path(env_dir).expanduser().resolve())

    # Package-internal (wheel install).
    out.append(Path(__file__).resolve().parent / "_lib")

    # Repo-checkout fallback — walk up looking for ``target/``.
    here = Path(__file__).resolve()
    for parent in here.parents:
        target = parent / "target"
        if target.is_dir():
            out.append(target / "release")
            out.append(target / "debug")
            break

    return out


def _find_lib() -> Path:
    """Locate the native library on disk, or raise :class:`VokraError`."""
    fname = _lib_filename()
    tried: list[str] = []
    for base in _candidate_paths():
        candidate = base / fname
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise VokraError(
        message=(
            "vokra native library not found. Searched: "
            + ", ".join(tried)
            + f" (looking for {fname!r}). Set VOKRA_LIB_DIR to override, or "
            "install a platform-tagged wheel that bundles the prebuilt library."
        ),
    )


# ---------------------------------------------------------------------------
# CDLL load — lazy, single-instance
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
_load_lock = threading.Lock()


def load_library() -> ctypes.CDLL:
    """Load the native library and attach ``_bindings.PROTOTYPES``.

    Idempotent: subsequent calls return the same ``CDLL`` instance so
    all wrappers share one ``vokra_last_error()`` thread-local table
    (the C ABI's errno lives inside the loaded library's data segment,
    so keeping a single ``CDLL`` handle is a correctness requirement,
    not just an optimization).

    Named ``load_library`` because ``session.py`` and ``stream.py``
    already import it under that name; :func:`load` is exported as an
    alias so newer call sites can use either spelling.
    """
    global _lib
    # Fast path without lock — the common case after first use.
    if _lib is not None:
        return _lib
    with _load_lock:
        if _lib is not None:  # pragma: no cover - double-checked locking
            return _lib
        path = _find_lib()
        try:
            # ``RTLD_LOCAL`` (the default) keeps our symbols out of the
            # global namespace so a host embedding multiple audio
            # runtimes cannot collide on ``vokra_*`` names.
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise VokraError(
                message=f"failed to dlopen vokra native library at {path}: {exc}",
            ) from exc

        try:
            _bindings.bind(lib)
        except AttributeError as exc:
            # A missing symbol means the loaded library was built from
            # a different revision than ``_bindings.py``. Fail loudly
            # (FR-EX-08) rather than let a later call segfault.
            raise VokraError(
                message=(
                    f"vokra native library at {path} is missing a symbol required by "
                    f"this binding: {exc}. Rebuild the library or reinstall the wheel."
                ),
            ) from exc

        _lib = lib
        return _lib


# Public alias so future call sites can use the shorter name.
load = load_library


def is_loaded() -> bool:
    """True if :func:`load_library` has succeeded at least once."""
    return _lib is not None


# ---------------------------------------------------------------------------
# Status check — call this in the SAME frame as the FFI call
# ---------------------------------------------------------------------------


def _read_last_error(lib: ctypes.CDLL) -> str:
    """Snapshot ``vokra_last_error()`` for the current thread.

    Returns ``""`` when the thread-local is NULL. The returned pointer
    is owned by the runtime (C ABI report §3: not the caller's to
    ``free``); we only decode it.

    Locale-independent decode (NFR-RL-01): UTF-8 with ``errors="replace"``.
    """
    try:
        raw = lib.vokra_last_error()
    except Exception:  # pragma: no cover - defensive
        # If the symbol was somehow never bound, we would rather surface
        # an empty message than mask the primary error we are reporting.
        return ""
    if not raw:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    # In case a future prototype tweak yields a c_void_p, coerce.
    try:
        return ctypes.string_at(raw).decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return ""


def _check_status(status: int, lib: Optional[ctypes.CDLL] = None) -> None:
    """Raise the mapped :class:`~vokra.errors.VokraError` subclass when
    ``status != VOKRA_OK``.

    MUST be called in the same Python frame as the originating FFI
    call so ``vokra_last_error()`` still reflects that call on the
    current thread (R3 mitigation, C ABI report §3).

    Parameters
    ----------
    status : int
        The ``vokra_status_t`` returned by the FFI call.
    lib : ctypes.CDLL, optional
        Pre-loaded library handle; defaults to the module-cached one
        (or, if that has not been populated yet, calls
        :func:`load_library`). Passing it explicitly avoids re-resolving
        in hot paths and lets tests inject a fake library.
    """
    if status == _bindings.VOKRA_OK:
        return

    # Resolve the library exactly once so we read ``vokra_last_error``
    # from the same instance that produced ``status``. We do NOT call
    # ``load_library()`` here if ``_lib`` is unset: a non-zero status
    # implies the caller already went through the library, so ``_lib``
    # should already be set. Falling back to a fresh load would create
    # a second CDLL instance with a different thread-local table and
    # silently lose the error message.
    if lib is None:
        lib = _lib

    message = _read_last_error(lib) if lib is not None else ""
    raise_from_status(status, message=message)
