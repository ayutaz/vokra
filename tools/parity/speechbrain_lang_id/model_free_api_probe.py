#!/usr/bin/env python3
"""Fail-closed, import-only SpeechBrain/TorchAudio API probe.

This probe deliberately does not acquire or execute a checkpoint.  The pinned
SpeechBrain 1.1.1 release contains the upstream guard for the removed
``torchaudio.list_audio_backends`` API and exposes the classifier methods used
by the official VoxLingua107 dumper.  No compatibility shim is installed: a
successful result must come from the fixed release itself.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from typing import Any, MutableMapping


EXPECTED_TORCH_VERSION = "2.13.0+cpu"
EXPECTED_TORCHAUDIO_VERSION = "2.11.0+cpu"
EXPECTED_SPEECHBRAIN_VERSION = "1.1.1"
SCHEMA = "vokra-speechbrain-lang-id-model-free-api-probe-v2"


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def force_offline_environment(environment: MutableMapping[str, str]) -> None:
    """Force every Hub client used by this import-only probe offline."""
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"


def _matches_exact_pinned_cpu_version(value: object, expected: str) -> bool:
    """Accept only the exact CPU wheel version recorded by the lock evidence."""
    if not isinstance(value, str):
        return False
    # TorchVersion is a str subclass with an overridden __eq__.  Call the
    # builtin implementation directly so only its actual text is accepted.
    return str.__eq__(value, expected) is True


def _version_block_status(
    torch_version: object, torchaudio_version: object
) -> str | None:
    """Return the fail-closed status for the two pinned CPU wheel versions."""
    if not _matches_exact_pinned_cpu_version(torch_version, EXPECTED_TORCH_VERSION):
        return "BLOCKED_UNEXPECTED_TORCH"
    if not _matches_exact_pinned_cpu_version(
        torchaudio_version, EXPECTED_TORCHAUDIO_VERSION
    ):
        return "BLOCKED_UNEXPECTED_TORCHAUDIO"
    return None


def self_test() -> None:
    offline = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    force_offline_environment(offline)
    assert offline == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    assert EXPECTED_SPEECHBRAIN_VERSION == "1.1.1"
    assert _version_block_status("2.13.0+cpu", "2.11.0+cpu") is None
    invalid_torch_versions = (
        "2.13.0",
        "2.13.01",
        "2.13.1+cpu",
        "2.13.0rc1",
        "2.13.0.dev0",
        "2.13.0+cu130",
        "2.13.0+cpu-junk",
    )
    for version in invalid_torch_versions:
        assert _version_block_status(version, EXPECTED_TORCHAUDIO_VERSION) == (
            "BLOCKED_UNEXPECTED_TORCH"
        )
    invalid_torchaudio_versions = (
        "2.11.0",
        "2.11.01",
        "2.11.1+cpu",
        "2.11.0rc1",
        "2.11.0.dev0",
        "2.11.0+cu130",
        "2.11.0+cpu-junk",
    )
    for version in invalid_torchaudio_versions:
        assert _version_block_status(EXPECTED_TORCH_VERSION, version) == (
            "BLOCKED_UNEXPECTED_TORCHAUDIO"
        )
    for unsupported in (None, 2.13, True, b"2.13.0+cpu"):
        assert _version_block_status(unsupported, EXPECTED_TORCHAUDIO_VERSION) == (
            "BLOCKED_UNEXPECTED_TORCH"
        )
    for unsupported in (None, 2.11, False, b"2.11.0+cpu"):
        assert _version_block_status(EXPECTED_TORCH_VERSION, unsupported) == (
            "BLOCKED_UNEXPECTED_TORCHAUDIO"
        )
    assert _version_block_status("2.13.0+cpu", "2.11.0+cpu-junk") == (
        "BLOCKED_UNEXPECTED_TORCHAUDIO"
    )

    class AlwaysEqual(str):
        def __eq__(self, other: object) -> bool:
            return True

    class AlwaysUnequal(str):
        def __eq__(self, other: object) -> bool:
            return False

    assert _version_block_status(
        AlwaysEqual(EXPECTED_TORCH_VERSION), EXPECTED_TORCHAUDIO_VERSION
    ) is None
    assert _version_block_status(
        AlwaysEqual("wrong-torch"), EXPECTED_TORCHAUDIO_VERSION
    ) == "BLOCKED_UNEXPECTED_TORCH"
    assert _version_block_status(
        AlwaysUnequal(EXPECTED_TORCH_VERSION), EXPECTED_TORCHAUDIO_VERSION
    ) is None
    assert _version_block_status(
        EXPECTED_TORCH_VERSION, AlwaysEqual(EXPECTED_TORCHAUDIO_VERSION)
    ) is None
    assert _version_block_status(
        EXPECTED_TORCH_VERSION, AlwaysEqual("wrong-torchaudio")
    ) == "BLOCKED_UNEXPECTED_TORCHAUDIO"
    assert _version_block_status(
        EXPECTED_TORCH_VERSION, AlwaysUnequal(EXPECTED_TORCHAUDIO_VERSION)
    ) is None
    print("model_free_api_probe.py self-test: PASS")


def probe() -> tuple[int, dict[str, Any]]:
    """Import only pinned packages and report an explicit fail-closed result."""
    force_offline_environment(os.environ)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "weights_requested": False,
        "model_instantiated": False,
        "model_executed": False,
        "hub_contacted": False,
        "platform": f"{platform.system()}-{platform.machine()}",
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        },
    }
    try:
        import torch
        import torchaudio
    except Exception as error:  # noqa: BLE001 - report import-only failure
        result["status"] = "BLOCKED_RUNTIME_IMPORT"
        result["error"] = _exception_record(error)
        return 2, result

    result["torch"] = torch.__version__
    result["torchaudio"] = torchaudio.__version__
    try:
        result["speechbrain"] = importlib.metadata.version("speechbrain")
    except importlib.metadata.PackageNotFoundError:
        result["status"] = "BLOCKED_MISSING_SPEECHBRAIN"
        return 2, result
    if result["speechbrain"] != EXPECTED_SPEECHBRAIN_VERSION:
        result["status"] = "BLOCKED_UNEXPECTED_SPEECHBRAIN"
        return 2, result
    version_status = _version_block_status(torch.__version__, torchaudio.__version__)
    if version_status is not None:
        result["status"] = version_status
        return 2, result

    try:
        import speechbrain  # noqa: F401 - validate the fixed release import
        from speechbrain.inference.classifiers import EncoderClassifier
    except Exception as error:  # noqa: BLE001 - report import-only failure
        result["status"] = "BLOCKED_SPEECHBRAIN_IMPORT"
        result["error"] = _exception_record(error)
        return 2, result

    result["speechbrain_import"] = "PASS"
    result["classifier_api"] = {
        "EncoderClassifier": "PASS",
        "encode_batch": callable(getattr(EncoderClassifier, "encode_batch", None)),
        "classify_batch": callable(getattr(EncoderClassifier, "classify_batch", None)),
    }
    if not all(
        (
            result["classifier_api"]["encode_batch"],
            result["classifier_api"]["classify_batch"],
        )
    ):
        result["status"] = "BLOCKED_MISSING_CLASSIFIER_API"
        return 2, result
    result["status"] = "MODEL_FREE_API_VALIDATED"
    return 0, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    rc, report = probe()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
