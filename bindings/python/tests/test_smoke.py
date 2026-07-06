# SPDX-License-Identifier: Apache-2.0
"""Smoke test — T12 (M2-12 Python bindings).

Verifies the two most basic invariants of the packaged binding:

1. ``import vokra`` succeeds without side effects on a bare interpreter.
   The top-level ``__init__.py`` (per its own docstring) intentionally
   does NOT load the native library, so this must pass even in a
   source-only checkout with no ``libvokra.{dylib,so,dll}`` on disk.

2. Once the prebuilt shared library is discoverable (either bundled in
   ``vokra/_lib/`` for wheel installs, or found in ``target/{release,
   debug}/`` for a repo checkout, or via ``VOKRA_LIB_DIR``), the version
   metadata symbol ``vokra_version()`` is callable through ctypes and
   returns a non-empty UTF-8 string. This is the minimum evidence that
   the C ABI is reachable from Python (T13's byte-identical parity test
   builds on top of this).

When the native library cannot be located, the second assertion is
skipped rather than failed: a fresh sdist install with no wheel-bundled
lib is a legitimate configuration, and CI enforces the presence of the
lib in the ``python-wheel-build`` job (D12) — not here.

Zero-dep (NFR-DS-02): only stdlib + pytest. No numpy, no pyo3, no cffi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without ``pip install -e .`` so this
# test runs in a fresh clone with only ``pytest`` on the path. Mirrors
# the sys.path shim used by ``test_session.py``.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# 1. bare import
# ---------------------------------------------------------------------------


def test_import_vokra_succeeds() -> None:
    """``import vokra`` must not require the native lib.

    Per ``src/vokra/__init__.py`` the package top-level exposes only
    metadata (``__version__``) and defers CDLL load to first API use.
    This lets a source-only install ``pip install vokra`` (sdist)
    succeed on any platform even when the wheel-bundled shared library
    is absent — the failure only surfaces when the user actually calls
    into a Session/Stream API. This test locks that contract in.
    """
    import vokra  # noqa: F401 — the act of importing is the assertion.

    # The package must advertise a version string so downstream tools
    # (pip, uv, poetry) can pin it. Pre-1.0 the exact value is unstable
    # (IF-01) so we only check shape, not equality.
    assert isinstance(vokra.__version__, str)
    assert vokra.__version__  # non-empty


def test_import_does_not_load_native_lib() -> None:
    """A bare ``import vokra`` must NOT eagerly dlopen the library.

    This matches ``__init__.py``'s docstring guarantee: metadata-only
    exposure. If some future refactor accidentally hoists a ``load()``
    call to module top-level, this test fires — that would silently
    break sdist installs on platforms without a prebuilt lib.
    """
    # Fresh import path: drop any cached module so we observe the load
    # side effects of THIS import, not a leftover from another test.
    for mod in ("vokra", "vokra._native", "vokra._bindings"):
        sys.modules.pop(mod, None)

    import vokra  # noqa: F401

    # ``vokra._native`` is not imported by ``vokra/__init__.py``, so it
    # must be absent from ``sys.modules`` right after the bare import.
    # (If a later ticket adds an eager submodule import for a good
    # reason, update this assertion in the same PR that changes the
    # contract — do not silently loosen it.)
    assert "vokra._native" not in sys.modules, (
        "vokra top-level import must not eagerly load _native / dlopen "
        "the shared library; sdist installs on unsupported platforms "
        "would break."
    )


# ---------------------------------------------------------------------------
# 2. native version metadata reachable via ctypes
# ---------------------------------------------------------------------------


def test_vokra_version_symbol_returns_nonempty_string() -> None:
    """``vokra_version()`` through ctypes returns a non-empty UTF-8 str.

    This is the smallest possible end-to-end check that the C ABI is
    wired up correctly: cdylib present, symbol exported, ctypes
    argtypes/restype correct (``restype = c_char_p``), and no
    LC_NUMERIC-style locale trap on the way through.

    Skipped when the native library cannot be found on disk — a bare
    sdist install without a wheel-bundled lib and without a repo
    checkout is not a failure of THIS test's contract.
    """
    from vokra import _bindings, _native
    from vokra.errors import VokraError

    try:
        lib = _native.load()
    except VokraError as exc:
        # Two distinct "not present" flavors surface as VokraError:
        #   (a) platform not supported (D10 — linux/macos/windows only)
        #   (b) libvokra.{dylib,so,dll} not found on any candidate path
        # Both are legitimate skip conditions here; CI enforces presence
        # inside the python-wheel-build job (D12).
        pytest.skip(f"native libvokra not available for smoke test: {exc}")

    # ``bind()`` should already have been called by ``load()``; assert
    # it explicitly so a future regression that decouples the two
    # doesn't silently leave argtypes unset (which would crash with a
    # ctypes ArgumentError, not a clean AttributeError).
    assert _bindings.PROTOTYPES["vokra_version"] == (
        # (restype, argtypes)
        # int-like ctypes types compare by identity; keep the shape
        # check narrow so an unrelated prototype edit doesn't break it.
        lib.vokra_version.restype,
        tuple(lib.vokra_version.argtypes),
    ), "vokra_version prototype drifted between _bindings.py and CDLL bind()"

    raw = lib.vokra_version()

    # ctypes returns ``bytes`` when ``restype = c_char_p`` for a
    # nul-terminated string; ``None`` would mean the symbol handed back
    # NULL, which the runtime must never do for a static version string.
    assert raw is not None, "vokra_version() returned NULL"
    assert isinstance(raw, bytes)
    assert len(raw) > 0, "vokra_version() returned an empty C string"

    # Decode with a defensive fallback: the runtime writes ASCII in
    # practice (e.g. "0.1.0-dev"), but even a locale-shifted build
    # should not raise here — LC_NUMERIC-safe (NFR-RL-01).
    version = raw.decode("utf-8", errors="replace")
    assert version, "decoded vokra_version() is empty"
    # No newline / control-char pollution — the version string is
    # supposed to be a single line for logging convenience.
    assert "\n" not in version and "\r" not in version, (
        f"vokra_version() returned a multi-line string: {version!r}"
    )
