# SPDX-License-Identifier: Apache-2.0
"""Opaque C-ABI handle wrapper with RAII lifecycle (T08).

The Vokra C ABI (`include/vokra.h`) exposes sessions and streams as
opaque `void *` handles created by `vokra_*_create` / `vokra_*_open` and
released by `vokra_*_destroy`. This module provides a small base class
that concrete wrappers (`session.Session`, `stream.Stream`) subclass so
handle ownership is uniform and reasoned about in one place.

Design notes
------------
- Handles are stored as ``ctypes.c_void_p`` and are never dereferenced
  from Python; only the loaded native library (via `_bindings.PROTOTYPES`)
  touches the pointed-at memory. This keeps the wrapper pure-`ctypes`
  with no non-stdlib deps (NFR-DS-02).
- Lifecycle follows deterministic RAII:
    * `__enter__` / `__exit__` for `with` blocks
    * `close()` for explicit release
    * `__del__` as a best-effort finalizer when GC runs
  All three converge on a single ``_release()`` hook that subclasses
  override to call the correct `vokra_*_destroy` symbol. Double-close is
  idempotent — a released handle is set to ``None`` and later calls to
  `close()` become no-ops so `__del__` after an explicit `close()` will
  not double-free.
- Any operation on a closed handle raises ``RuntimeError`` from
  ``raw()``; that is the single check-point subclasses go through when
  they need the raw ``c_void_p`` to hand to a native call.
- ``__del__`` swallows exceptions from the destroy call because
  interpreter shutdown can null out module globals (including the
  loaded CDLL) before finalizers run; raising there would spam stderr
  without recovering anything. Explicit ``close()`` still propagates.

The base class stays deliberately small (no I/O, no logging, no
locking): thread-safety of individual handles is documented in the
C ABI as "single-thread ownership" and enforced by the higher-level
wrappers, not here.
"""

from __future__ import annotations

import ctypes
from typing import Optional


class Handle:
    """Base class for RAII-wrapped opaque C-ABI handles.

    Subclasses set ``self._handle`` (a ``ctypes.c_void_p`` or ``None``)
    in their constructor after the native ``*_create``/``*_open`` call
    succeeds, and override ``_release`` to invoke the paired destroy
    symbol. Instances of this base class with no subclass initialisation
    behave as an already-closed handle so trivial construction (e.g. the
    smoke test ``Handle()``) never touches the native library.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: Optional[ctypes.c_void_p] = None) -> None:
        # ``handle=None`` -> "already closed" sentinel. Callers that own
        # a real pointer should pass the ``c_void_p`` they got from the
        # native ``*_create`` call.
        self._handle: Optional[ctypes.c_void_p] = handle

    # -- ownership predicates ------------------------------------------------

    @property
    def closed(self) -> bool:
        """True once the paired destroy has run (or was never set)."""
        return self._handle is None

    def raw(self) -> ctypes.c_void_p:
        """Return the raw ``c_void_p`` for a native call.

        Raises ``RuntimeError`` if the handle is already released; this
        is the single choke-point that keeps use-after-free from
        reaching the C ABI.
        """
        if self._handle is None:
            raise RuntimeError(
                f"{type(self).__name__}: handle is closed (use-after-free)"
            )
        return self._handle

    # -- lifecycle -----------------------------------------------------------

    def _release(self) -> None:
        """Invoke the paired ``vokra_*_destroy`` symbol.

        Base implementation is a no-op so ``Handle()`` (the smoke test)
        can be constructed and closed without a loaded library.
        Concrete subclasses override this to call e.g.
        ``lib.vokra_session_destroy(self._handle)``.
        """
        # Intentionally empty; see docstring.

    def close(self) -> None:
        """Release the underlying native handle, once.

        Safe to call multiple times: after the first call the handle is
        cleared so subsequent calls (including the implicit ``__del__``
        or a second explicit close) are no-ops. Exceptions from
        ``_release`` propagate to the caller so mis-wired destroy
        symbols surface loudly during development.
        """
        if self._handle is None:
            return
        try:
            self._release()
        finally:
            # Clear even if _release raised so we do not try again from
            # __del__ / a repeated close(); the C ABI treats a failed
            # destroy as still-consumed for the caller.
            self._handle = None

    def __enter__(self) -> "Handle":
        # Fail early if someone re-enters an already-closed handle.
        self.raw()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Suppress no exceptions; propagate whatever the with-body raised
        # after releasing the handle.
        self.close()

    def __del__(self) -> None:
        # Best-effort finalizer. Interpreter shutdown can already have
        # torn down the CDLL / _bindings module by the time GC runs
        # against orphan handles, so we intentionally swallow errors
        # here — explicit close() remains the reliable release path.
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._handle is None else f"raw=0x{int(self._handle.value or 0):x}"
        return f"<{type(self).__name__} {state}>"
