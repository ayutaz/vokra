"""Fail-loud import seams for unused text-phonemization helpers.

The pinned upstream module imports its text path at module import time even
when a packet already contains phoneme IDs.  Installing these tiny sentinels
keeps that unused optional import from widening the audited environment.  Any
attempt to construct or call the text/eSpeak path raises immediately; this is
not a silent fallback and cannot produce phonemes.
"""
from __future__ import annotations

import sys
import types
from typing import Any


class _ForbiddenUse:
    def __init__(self, package: str, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            f"Zonos packet reference forbids text phonemization; optional {package} path was used"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Zonos packet reference forbids text phonemization")

class _DeferredForbidden:
    """Permit an import-time helper object, but fail on every real use."""

    def __init__(self, package: str) -> None:
        self.package = package

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Zonos packet reference forbids text phonemization; optional {self.package}.{name} was used"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Zonos packet reference forbids text phonemization")

    def create(self) -> "_DeferredForbidden":
        # Sudachi's default argument is constructed while the official module
        # is imported; defer the loud failure until a text path actually uses
        # the tokenizer.
        return self

    def tokenize(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Zonos packet reference forbids text phonemization")


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    return module


def install() -> None:
    """Install only the source's unused phonemizer/text import sentinels."""
    phonemizer = _module("phonemizer")
    backend = _module("phonemizer.backend")
    backend.EspeakBackend = _ForbiddenUse  # type: ignore[attr-defined]
    phonemizer.backend = backend  # type: ignore[attr-defined]
    sys.modules["phonemizer"] = phonemizer
    sys.modules["phonemizer.backend"] = backend

    inflect = _module("inflect")
    inflect.engine = lambda *args, **kwargs: _DeferredForbidden("inflect")  # type: ignore[attr-defined]
    sys.modules["inflect"] = inflect

    kanjize = _module("kanjize")
    kanjize.number2kanji = lambda *args, **kwargs: _ForbiddenUse("kanjize")  # type: ignore[attr-defined]
    sys.modules["kanjize"] = kanjize

    sudachi = _module("sudachipy")
    sudachi.Dictionary = lambda *args, **kwargs: _DeferredForbidden("sudachipy")  # type: ignore[attr-defined]
    sudachi.SplitMode = types.SimpleNamespace(A=object())  # type: ignore[attr-defined]
    sys.modules["sudachipy"] = sudachi


def self_test() -> None:
    install()
    import phonemizer.backend

    try:
        phonemizer.backend.EspeakBackend("en-us")
    except RuntimeError as error:
        assert "forbids text phonemization" in str(error)
    else:
        raise AssertionError("eSpeak sentinel did not fail closed")
    from sudachipy import Dictionary

    assert Dictionary(dict="full").create() is not None
    try:
        Dictionary(dict="full").create().tokenize("text")
    except RuntimeError as error:
        assert "forbids text phonemization" in str(error)
    else:
        raise AssertionError("Sudachi sentinel did not fail closed")
    print("zonos import policy self-test: OK")


if __name__ == "__main__":
    self_test()
