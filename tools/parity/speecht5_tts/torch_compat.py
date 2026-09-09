"""Fail-closed compatibility checks for the pinned SpeechT5 oracle.

Transformers 5.10.4 imports an unrelated FP8 module that names
``torch.float8_e8m0fnu``.  The exact approved torch 2.4.1+cpu wheel does not
export that name.  This module installs a narrow alias only after verifying
both installed distribution identities; it never changes model configuration
or selects an FP8/quantized execution route.
"""

from __future__ import annotations

import importlib.metadata as metadata
import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable


SUPPORTED_TORCH = "2.4.1+cpu"
SUPPORTED_TRANSFORMERS = "5.10.4"
MISSING = object()
COMPATIBILITY_SMOKE_SENTINEL = (
    "SPEECHT5_COMPATIBILITY_SMOKE status=PASS torch={torch} "
    "transformers={transformers} alias={alias} "
    "model_load=NOT_PERFORMED upload=NOT_PERFORMED"
)


def install_float8_import_compat(
    torch_module: Any,
    *,
    version_reader: Callable[[str], str] | None = None,
) -> str:
    """Validate exact package identities and install one import-only alias.

    ``version_reader`` is used only by the model-free self-test. Production
    callers use installed distribution metadata and cannot override it.
    """
    read_version = version_reader or metadata.version
    torch_version = read_version("torch")
    transformers_version = read_version("transformers")
    if torch_version != SUPPORTED_TORCH:
        raise RuntimeError(f"unsupported torch identity for FP8 import shim: {torch_version}")
    if transformers_version != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            f"unsupported Transformers identity for FP8 import shim: {transformers_version}"
        )

    dtype_type = getattr(torch_module, "dtype", MISSING)
    if not isinstance(dtype_type, type):
        raise RuntimeError("torch dtype type is unavailable; refusing compatibility mutation")
    existing = getattr(torch_module, "float8_e8m0fnu", MISSING)
    if existing is not MISSING:
        if not isinstance(existing, dtype_type):
            raise RuntimeError("unexpected existing torch.float8_e8m0fnu state")
        return "native"

    sentinel = getattr(torch_module, "float8_e4m3fn", MISSING)
    if sentinel is MISSING or not isinstance(sentinel, dtype_type):
        raise RuntimeError("approved torch FP8 sentinel is unavailable")
    setattr(torch_module, "float8_e8m0fnu", sentinel)
    if getattr(torch_module, "float8_e8m0fnu", MISSING) is not sentinel:
        raise RuntimeError("torch.float8_e8m0fnu compatibility mutation was not exact")
    return "shimmed"


def require_non_quantized_config(checkpoint: Path) -> None:
    """Reject quantized or fine-grained FP8 SpeechT5 configurations."""
    path = checkpoint / "config.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"SpeechT5 config is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"SpeechT5 config is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("SpeechT5 config root is not an object")

    def walk(node: Any, path_name: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = str(key).casefold()
                if ("quant" in lowered or "fp8" in lowered) and child not in (None, False, {}, [], ""):
                    raise RuntimeError(f"non-quantized SpeechT5 config contains route {path_name}.{key}")
                walk(child, f"{path_name}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path_name}[{index}]")

    walk(value, "config")


def _require_compatibility_smoke_host(
    *,
    environment: dict[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> None:
    env = environment if environment is not None else os.environ
    actual_system = system if system is not None else platform.system()
    actual_machine = machine if machine is not None else platform.machine()
    if env.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("VOKRA_PUBLISH_ON_VAST=1 is required for compatibility smoke")
    if actual_system != "Linux" or actual_machine != "x86_64":
        raise RuntimeError(
            "compatibility smoke requires Linux x86_64; "
            f"got {actual_system} {actual_machine}"
        )


def compatibility_smoke() -> int:
    """Import only the pinned API on an authorized VAST host.

    This deliberately does not inspect a checkpoint or instantiate a model.
    Offline environment flags make any accidental hub access fail closed while
    the package imports are exercised.
    """
    _require_compatibility_smoke_host()
    torch_version = metadata.version("torch")
    transformers_version = metadata.version("transformers")
    if torch_version != SUPPORTED_TORCH or transformers_version != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            "compatibility smoke package identity mismatch: "
            f"torch={torch_version} transformers={transformers_version}"
        )
    previous_offline = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    try:
        import torch

        alias_status = install_float8_import_compat(torch)
        import transformers
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
    finally:
        for key, value in previous_offline.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if torch.__version__ != SUPPORTED_TORCH:
        raise RuntimeError(f"torch runtime version drifted: {torch.__version__}")
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            f"Transformers runtime version drifted: {transformers.__version__}"
        )
    if getattr(transformers, "SpeechT5ForTextToSpeech", MISSING) is not SpeechT5ForTextToSpeech:
        raise RuntimeError("Transformers does not expose SpeechT5ForTextToSpeech exactly")
    if getattr(transformers, "SpeechT5Tokenizer", MISSING) is not SpeechT5Tokenizer:
        raise RuntimeError("Transformers does not expose SpeechT5Tokenizer exactly")
    if alias_status not in {"native", "shimmed"}:
        raise RuntimeError(f"unexpected FP8 compatibility status: {alias_status}")
    print(
        COMPATIBILITY_SMOKE_SENTINEL.format(
            torch=torch.__version__, transformers=transformers.__version__, alias=alias_status
        )
    )
    return 0


def self_test() -> None:
    class FakeDType:
        pass

    class FakeTorch:
        dtype = FakeDType

        def __init__(self, *, native: Any = MISSING, sentinel: Any = None) -> None:
            self.float8_e4m3fn = sentinel if sentinel is not None else FakeDType()
            if native is not MISSING:
                self.float8_e8m0fnu = native

    versions = {"torch": SUPPORTED_TORCH, "transformers": SUPPORTED_TRANSFORMERS}
    reader = versions.__getitem__
    shimmed = FakeTorch()
    sentinel = shimmed.float8_e4m3fn
    assert install_float8_import_compat(shimmed, version_reader=reader) == "shimmed"
    assert shimmed.float8_e8m0fnu is sentinel

    native_value = FakeDType()
    native = FakeTorch(native=native_value)
    assert install_float8_import_compat(native, version_reader=reader) == "native"
    assert native.float8_e8m0fnu is native_value

    wrong_versions = {"torch": "2.4.1", "transformers": SUPPORTED_TRANSFORMERS}
    rejected = FakeTorch()
    try:
        install_float8_import_compat(rejected, version_reader=wrong_versions.__getitem__)
    except RuntimeError:
        assert not hasattr(rejected, "float8_e8m0fnu")
    else:
        raise AssertionError("wrong torch identity was accepted")

    mutated = FakeTorch(native=object())
    try:
        install_float8_import_compat(mutated, version_reader=reader)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected native symbol state was accepted")

    class MutatingTorch(FakeTorch):
        def __setattr__(self, name: str, value: Any) -> None:
            if name == "float8_e8m0fnu":
                object.__setattr__(self, name, object())
            else:
                object.__setattr__(self, name, value)

    mutating = MutatingTorch()
    try:
        install_float8_import_compat(mutating, version_reader=reader)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mutating FP8 compatibility state was accepted")

    missing_sentinel = FakeTorch(sentinel=object())
    try:
        install_float8_import_compat(missing_sentinel, version_reader=reader)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown FP8 sentinel state was accepted")

    with tempfile.TemporaryDirectory(prefix="speecht5-config-selftest-") as directory:
        checkpoint = Path(directory)
        (checkpoint / "config.json").write_text('{"model_type":"speecht5"}\n', encoding="utf-8")
        require_non_quantized_config(checkpoint)
        (checkpoint / "config.json").write_text('{"quantization_config":{"bits":8}}\n', encoding="utf-8")
        try:
            require_non_quantized_config(checkpoint)
        except RuntimeError:
            pass
        else:
            raise AssertionError("quantized config was accepted")
        (checkpoint / "config.json").write_text('{"finegrained_fp8":true}\n', encoding="utf-8")
        try:
            require_non_quantized_config(checkpoint)
        except RuntimeError:
            pass
        else:
            raise AssertionError("fine-grained FP8 config was accepted")

    environment = {}
    try:
        _require_compatibility_smoke_host(environment=environment, system="Linux", machine="x86_64")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing VAST authorization was accepted")
    environment = {"VOKRA_PUBLISH_ON_VAST": "1"}
    try:
        _require_compatibility_smoke_host(environment=environment, system="Darwin", machine="x86_64")
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-Linux compatibility host was accepted")
    try:
        _require_compatibility_smoke_host(environment=environment, system="Linux", machine="arm64")
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-x86_64 compatibility host was accepted")

    print("speecht5 torch compatibility self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compatibility-smoke", action="store_true")
    args = parser.parse_args()
    if args.self_test == args.compatibility_smoke:
        parser.error("exactly one of --self-test or --compatibility-smoke is required")
    if args.self_test:
        self_test()
        return 0
    return compatibility_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
