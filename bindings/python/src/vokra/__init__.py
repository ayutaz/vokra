"""Public Python API for the Vokra speech runtime.

Importing :mod:`vokra` exposes the high-level handle and error types but does
not load the native library. The first operation that needs native code (for
example :meth:`Session.open`) resolves and binds ``libvokra`` lazily. This
keeps documentation tooling and source-only installs importable while still
failing loudly at the first unavailable native operation.

Pre-1.0: the underlying C ABI is not frozen. Pin an exact version. See the
binding README and ADR-0003.
"""

from .errors import (
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
from .session import Session
from .stream import Event, Stream

try:
    from importlib.metadata import PackageNotFoundError, version as _distribution_version

    __version__ = _distribution_version("vokra")
except PackageNotFoundError:  # pragma: no cover - source-tree import without metadata
    __version__ = "0.1.0.dev0"
__all__ = [
    "__version__",
    "Session",
    "Stream",
    "Event",
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
]
