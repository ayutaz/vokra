# SPDX-License-Identifier: Apache-2.0
"""Memory-leak regression test for Session handles (T12).

Creates and destroys 1000 ``Session`` instances via the same three
release paths ``test_session.py`` covers (context manager, explicit
``close()``, ``__del__`` on drop) and asserts:

1. ``vokra_session_destroy`` is called **exactly once per successful
   create** — the FakeLib call ledger must have length 1000. This is the
   authoritative check: the C ABI cannot leak if every create is paired
   with exactly one destroy.
2. Resident set size (RSS) growth is bounded to a small, generous cap.
   RSS is inherently noisy (allocator arenas, gc bookkeeping, JIT
   caches), so we probe once after a warm-up of 100 iterations and again
   after the full 1000, and require the delta stays under a per-handle
   budget. That budget is deliberately loose — the goal is to catch a
   linear leak (bytes per iter × 900 iters), not to profile allocator
   noise.
3. Python-side object count does not accumulate: after ``gc.collect()``
   the number of live ``Session`` instances tracked by GC returns to
   zero. This guards against a wrapper-side reference cycle that would
   pin ``ctypes.CDLL`` mocks or ``c_void_p`` objects even though the
   native destroy ran.

Design notes
------------
* We use the same ``FakeLib`` shape as ``test_session.py`` so the test
  runs without a prebuilt native library — the wrapper's RAII invariants
  are language-level and do not require real native memory to verify.
  A separate integration test (T13, GGUF-gated) exercises the real
  ``libvokra`` cdylib end-to-end.
* RSS is read via :mod:`resource` on POSIX. On Windows ``resource`` is
  unavailable; we fall back to the ledger + gc checks, which are the
  strong guarantees. The RSS check is a defense-in-depth belt on top of
  the ledger suspenders.
* ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` returns bytes on macOS
  and kilobytes on Linux (see :manpage:`getrusage(2)`); we normalise to
  bytes so the assertion threshold is stable across platforms.
* We iterate 1000 times per FR guidance in the plan (T12). At ~1 KB per
  Session wrapper (self + Handle base + c_void_p) the theoretical Python
  churn is ~1 MB, well within our 8 MB cap even accounting for gc arenas
  and pytest fixtures.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without installing the package
# (mirrors test_session.py so both files run in the same fresh checkout).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vokra.session import Session, _VOKRA_OK  # noqa: E402


# --- fake native library (mirrors test_session.py) --------------------------


class LedgerLib:
    """Records every create/destroy call for post-hoc balance checking.

    Unlike ``test_session.FakeLib`` this tracks *both* sides of the
    lifecycle so we can assert ``len(create) == len(destroy)`` — the
    invariant a leak would violate. Each create returns a fresh sentinel
    pointer so double-destroys would be visible in the ledger (same
    address twice).
    """

    def __init__(self) -> None:
        self.create_calls: list[int] = []
        self.destroy_calls: list[int] = []
        # Fresh non-NULL sentinel per create. Start above zero so any
        # accidental NULL from wrapper bugs is obvious.
        self._next_handle = 0x1000

    # -- symbols the wrapper actually calls ---------------------------------

    def vokra_session_create_from_file(self, path_bytes, out_pp):
        # Hand out a unique sentinel so a double-destroy shows up as a
        # duplicate in destroy_calls, and a leak shows up as
        # len(create) > len(destroy).
        h = self._next_handle
        self._next_handle += 1
        self.create_calls.append(h)
        out_pp._obj.value = h
        return _VOKRA_OK

    def vokra_session_destroy(self, handle):
        val = handle.value if isinstance(handle, ctypes.c_void_p) else int(handle)
        self.destroy_calls.append(val)

    def vokra_last_error(self):
        return None


@pytest.fixture()
def ledger_lib() -> LedgerLib:
    return LedgerLib()


@pytest.fixture()
def tmp_model(tmp_path: Path) -> str:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"\x00")
    return str(p)


# --- RSS probe (portable POSIX + graceful Windows fallback) -----------------


def _rss_bytes() -> int | None:
    """Return current process RSS in bytes, or None if unavailable.

    * Linux: ``ru_maxrss`` is in kilobytes — multiply by 1024.
    * macOS: ``ru_maxrss`` is in bytes.
    * Windows: :mod:`resource` is stdlib-absent — return None so the
      caller can skip the RSS assertion while keeping the ledger check.

    Note: ``ru_maxrss`` is a *high-water mark*, not current RSS, so it
    is monotonic within a process. That is actually what we want for a
    leak test: a leak makes the watermark climb; a non-leak plateaus.
    """
    try:
        import resource  # POSIX-only stdlib module
    except ImportError:
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: on Linux ru_maxrss is in KB and typically < 10^7;
    # on macOS it is in bytes and typically > 10^7. This mirrors the
    # convention documented in psutil's cross-platform notes.
    if sys.platform == "darwin":
        return int(ru)
    return int(ru) * 1024


# --- the leak test ----------------------------------------------------------


_ITERATIONS = 1000
# Generous per-handle budget: at ~1 KB of Python overhead per Session
# (self, Handle base, c_void_p, dict entry in the fake ledger) the
# theoretical churn is ~1 MB. We cap at 8 MB total delta to leave room
# for allocator noise (glibc arenas, gc gen-2 promotions) while still
# catching a leak that scales with iteration count. A real leak of even
# 32 KB/iter would blow through this by 4x.
_MAX_RSS_GROWTH_BYTES = 8 * 1024 * 1024


def test_1000_sessions_do_not_leak(ledger_lib: LedgerLib, tmp_model: str) -> None:
    """Full-cycle regression: 1000 create/destroy pairs, RSS bounded."""
    # Warm-up: let allocator arenas settle before we take a baseline.
    # Without this, the first ~100 iterations grow ru_maxrss just from
    # loading pytest fixtures and priming gc gen-2, which would give us
    # a false-positive delta.
    for _ in range(100):
        with Session.open(tmp_model, lib=ledger_lib):
            pass
    gc.collect()
    baseline_rss = _rss_bytes()

    # Main loop: mix the three release paths so we exercise the same
    # invariants that test_session.py verifies individually, but under
    # sustained load. Path selection is deterministic (i % 3) so a
    # failure is reproducible.
    for i in range(_ITERATIONS):
        path = i % 3
        if path == 0:
            # Context manager release
            with Session.open(tmp_model, lib=ledger_lib):
                pass
        elif path == 1:
            # Explicit close()
            sess = Session.open(tmp_model, lib=ledger_lib)
            sess.close()
        else:
            # Drop reference; __del__ + gc must release
            sess = Session.open(tmp_model, lib=ledger_lib)
            del sess

    # Force pending finalizers before any assertion — __del__ path
    # relies on GC running, and CPython's refcount usually reclaims on
    # last decref but PyPy / debug builds defer.
    gc.collect()

    # -- ledger check: the strong guarantee --------------------------------
    # Every create must be paired with exactly one destroy. Warm-up
    # contributes 100 pairs, main loop contributes _ITERATIONS pairs.
    expected = 100 + _ITERATIONS
    assert len(ledger_lib.create_calls) == expected, (
        f"expected {expected} creates, got {len(ledger_lib.create_calls)}"
    )
    assert len(ledger_lib.destroy_calls) == expected, (
        f"leak: {len(ledger_lib.create_calls)} creates but only "
        f"{len(ledger_lib.destroy_calls)} destroys "
        f"({len(ledger_lib.create_calls) - len(ledger_lib.destroy_calls)} handles leaked)"
    )
    # No duplicate destroys — a double-free would show the same sentinel
    # pointer twice in the ledger.
    assert len(set(ledger_lib.destroy_calls)) == len(ledger_lib.destroy_calls), (
        "double-destroy detected: same handle destroyed twice"
    )

    # -- gc check: no live Session instances after collection --------------
    live_sessions = [obj for obj in gc.get_objects() if isinstance(obj, Session)]
    assert live_sessions == [], (
        f"{len(live_sessions)} Session instance(s) still alive after gc.collect(); "
        f"a reference cycle is pinning them"
    )

    # -- RSS check (belt on top of ledger suspenders) ----------------------
    # Skip on platforms without `resource` (Windows). The ledger + gc
    # checks above are already sufficient to catch a wrapper-side leak;
    # RSS is a defense-in-depth signal for allocator-level leaks that
    # only manifest with a real native library (covered by T13).
    if baseline_rss is None:
        pytest.skip("resource module unavailable (Windows) — ledger check passed")
    final_rss = _rss_bytes()
    assert final_rss is not None  # POSIX branch, cannot be None here
    growth = final_rss - baseline_rss
    assert growth <= _MAX_RSS_GROWTH_BYTES, (
        f"RSS grew by {growth / (1024 * 1024):.2f} MB across "
        f"{_ITERATIONS} iterations (cap: "
        f"{_MAX_RSS_GROWTH_BYTES / (1024 * 1024):.0f} MB); "
        f"probable leak in Python wrapper"
    )
