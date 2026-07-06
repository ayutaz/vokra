"""Vokra — audio-specialized inference runtime (Python binding).

This is a T06 skeleton. Public API surface (Session, Stream, VokraError,
etc.) is populated by later tickets:

    T07  scripts/gen-py-bindings.py  -> _bindings.py
    T08  _handles.py, session.py, stream.py
    T09  errors.py
    T10  Session.transcribe / Session.synthesize
    T11  Stream.push / Stream.poll

Until then, importing `vokra` succeeds but exposes only metadata; the
native library is not loaded here so that a wheel built without the
prebuilt lib (source-only sdist install) still imports without crashing.

Pre-1.0: the underlying C ABI is not frozen. Pin an exact version.
See README.md and ADR-0003.
"""

__version__ = "0.1.0.dev0"
__all__ = ["__version__"]
