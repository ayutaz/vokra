"""Import-only compatibility for DeepFilterNet 0.5.6 on torchaudio 2.11+.

DeepFilterNet's :mod:`df.io` imports ``AudioMetaData`` from the legacy
``torchaudio.backend.common`` namespace at module import time.  Torchaudio
2.11 removed that namespace after moving its audio I/O surface to TorchCodec,
but DeepFilterNet's tensor-only ``init_df`` / ``enhance`` path does not call
``df.io.load_audio`` or otherwise use the metadata type.

Keep the upstream reference oracle importable by restoring only that type
namespace.  This deliberately does not implement ``torchaudio.info`` or
replace any model, resampler, audio I/O, or tensor operation.  A caller that
tries to use DeepFilterNet's removed legacy audio-I/O path will still fail
loudly instead of receiving an approximation.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import NamedTuple


class _AudioMetaData(NamedTuple):
    """Shape of torchaudio's removed public ``AudioMetaData`` value."""

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str


def install_deepfilternet_import_compat() -> None:
    """Provide the legacy metadata import only when torchaudio removed it."""

    try:
        from torchaudio.backend.common import AudioMetaData as _ExistingAudioMetaData  # noqa: F401

        return
    except ModuleNotFoundError as error:
        if error.name not in {"torchaudio.backend", "torchaudio.backend.common"}:
            raise

    import torchaudio

    if not hasattr(torchaudio, "functional"):
        raise RuntimeError(
            "torchaudio lacks the functional API required by the DFN3 oracle"
        )

    backend_module = ModuleType("torchaudio.backend")
    backend_module.__path__ = []  # type: ignore[attr-defined]  # mark as package
    common_module = ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]
    backend_module.common = common_module  # type: ignore[attr-defined]

    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module
