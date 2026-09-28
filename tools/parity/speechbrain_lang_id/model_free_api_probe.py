#!/usr/bin/env python3
"""Fail-closed, import-only SpeechBrain/TorchAudio API probe.

This probe deliberately does not acquire or execute a checkpoint.  SpeechBrain
1.0.3 imports a removed ``torchaudio.list_audio_backends`` symbol when paired
with the pinned TorchAudio 2.11.0+cpu wheel.  An in-memory shim is exercised
only to distinguish that known import seam from an unrelated import failure;
the production result remains blocked whenever the shim is required.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from types import ModuleType
from typing import Any, MutableMapping


EXPECTED_TORCH_PREFIX = "2.13.0"
EXPECTED_TORCHAUDIO_PREFIX = "2.11.0"
SCHEMA = "vokra-speechbrain-lang-id-model-free-api-probe-v1"


def install_audio_backend_compat(module: ModuleType) -> str:
    """Install only the known import-time shim and report its disposition."""
    if hasattr(module, "list_audio_backends"):
        return "native"
    module.list_audio_backends = lambda: []  # type: ignore[attr-defined]
    if not callable(module.list_audio_backends):  # type: ignore[attr-defined]
        raise RuntimeError("torchaudio compatibility shim was not callable")
    return "shimmed"


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def force_offline_environment(environment: MutableMapping[str, str]) -> None:
    """Force every Hub client used by this import-only probe offline."""
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"


def self_test() -> None:
    native = ModuleType("native")
    native.list_audio_backends = lambda: []  # type: ignore[attr-defined]
    assert install_audio_backend_compat(native) == "native"
    missing = ModuleType("missing")
    assert install_audio_backend_compat(missing) == "shimmed"
    assert missing.list_audio_backends() == []  # type: ignore[attr-defined]
    offline = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    force_offline_environment(offline)
    assert offline == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
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
    if not torch.__version__.startswith(EXPECTED_TORCH_PREFIX):
        result["status"] = "BLOCKED_UNEXPECTED_TORCH"
        return 2, result
    if not torchaudio.__version__.startswith(EXPECTED_TORCHAUDIO_PREFIX):
        result["status"] = "BLOCKED_UNEXPECTED_TORCHAUDIO"
        return 2, result

    native_api = hasattr(torchaudio, "list_audio_backends")
    result["torchaudio_list_audio_backends"] = "native" if native_api else "missing"
    if native_api:
        try:
            import speechbrain  # noqa: F401
        except Exception as error:  # noqa: BLE001 - record official import failure
            result["native_speechbrain_import"] = "BLOCKED"
            result["native_import_error"] = _exception_record(error)
        else:
            result["native_speechbrain_import"] = "PASS"
        if result["native_speechbrain_import"] != "PASS":
            result["status"] = "BLOCKED_SPEECHBRAIN_IMPORT"
            return 2, result
        result["status"] = "MODEL_FREE_API_VALIDATED"
        return 0, result

    # Do not import SpeechBrain before installing the shim: a failed import
    # leaves partially initialized submodules in sys.modules and would make a
    # later compatibility-only retry report a false circular-import failure.
    result["native_speechbrain_import"] = "NOT_ATTEMPTED_MISSING_API"
    result["native_import_error"] = {
        "type": "MissingTorchaudioAPI",
        "message": "torchaudio.list_audio_backends is absent",
    }
    shim = install_audio_backend_compat(torchaudio)
    result["compatibility_shim"] = shim
    try:
        import speechbrain  # noqa: F811,F401
    except Exception as error:  # noqa: BLE001 - record shimmed import failure
        result["shimmed_speechbrain_import"] = "BLOCKED"
        result["shimmed_import_error"] = _exception_record(error)
    else:
        result["shimmed_speechbrain_import"] = "PASS"
    result["status"] = "BLOCKED_COMPATIBILITY_SHIM_REQUIRED"
    return 2, result


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
