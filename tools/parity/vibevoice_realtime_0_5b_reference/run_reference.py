#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python
"""Run a narrow official VibeVoice-Realtime-0.5B real-weight reference.

This runner deliberately imports Microsoft's pinned VibeVoice implementation and
calls its public ``forward_lm``/``forward_tts_lm`` methods.  It is not a Rust
mirror and it does not provide a fallback implementation.  The checkpoint and
the source checkout must be present on VAST; the maintainer Mac must not run
this command or receive the checkpoint.

The operation is intentionally small but meaningful: a deterministic four-token
text pass, the official TTS-LM splice plus EOS classifier, and the official
acoustic connector.  When CUDA is available, the same operation is measured on
CPU and CUDA and CUDA is selected only if its median is lower and its outputs
pass the provisional device-selection guard. That guard is not an independent
numerical reference or release gate.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

# Keep the authenticated upstream checkout clean across repeated VAST runs.
# The runner never needs Python bytecode artifacts in that checkout.
sys.dont_write_bytecode = True


SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
CHECKPOINT_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
CHECKPOINT_SHA256 = "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514"
TRANSFORMERS_PIN = "5.10.4"
QWEN2_FAST_MODULE = "transformers.models.qwen2.tokenization_qwen2_fast"
REGISTRATION_COMPATIBILITY = "SCOPED_VIBEVOICE_ACOUSTIC_TOKENIZER_OVERRIDE"
CONFIG_BYTES = 2117
CONFIG_SHA256 = "caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57"
EXPECTED_TENSOR_COUNT = 605
EXPECTED_HIDDEN_SIZE = 896
EXPECTED_ACOUSTIC_DIM = 64
SEED = 1234
# Provisional same-workload device-selection guard, not a release/parity gate.
CUDA_ATOL = 5e-2
# Only used to avoid division by zero in diagnostic relative-error reporting.
# It does not change the device-selection guard.
RELATIVE_EPS = 1e-6
FORMAT = "vokra-vibevoice-realtime-0.5b-official-reference-v1"


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected a regular file, got {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _source_identity(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError(f"official source root is not a regular directory: {source_root}")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("official source root must be a git checkout") from error
    if revision != SOURCE_REVISION:
        raise RuntimeError(f"official source revision mismatch: {revision} != {SOURCE_REVISION}")
    try:
        status = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not inspect official source checkout status") from error
    if status:
        raise RuntimeError("official source checkout is dirty; refusing unpinned source files")
    required = [
        source_root / "vibevoice" / "modular" / "modeling_vibevoice_streaming.py",
        source_root / "vibevoice" / "modular" / "modeling_vibevoice_streaming_inference.py",
        source_root / "vibevoice" / "modular" / "configuration_vibevoice_streaming.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"official source checkout is incomplete: {missing}")
    return {"repository": "microsoft/VibeVoice", "revision": revision, "root": str(source_root)}


def _assert_regular_output_dir(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise RuntimeError(f"output directory must not be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"output path is not a directory: {path}")


def _load_official(source_root: Path):
    """Import only the pinned upstream implementation; never mirror it."""

    sys.path.insert(0, str(source_root))
    try:
        from vibevoice.modular.configuration_vibevoice_streaming import (  # type: ignore
            VibeVoiceStreamingConfig,
        )
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (  # type: ignore
            VibeVoiceStreamingForConditionalGenerationInference,
        )
    except Exception as error:  # noqa: BLE001 - report the exact upstream import failure
        raise RuntimeError(
            "pinned official VibeVoice import failed; no mirror/fallback is permitted"
        ) from error
    return VibeVoiceStreamingConfig, VibeVoiceStreamingForConditionalGenerationInference


def _qwen2_fast_init_kwargs(
    vocab_file: str | None,
    merges_file: str | None,
    tokenizer_file: str | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Translate the removed Fast-tokenizer constructor names to Transformers 5."""

    translated = dict(kwargs)
    if vocab_file is not None:
        translated.setdefault("vocab", vocab_file)
        translated.setdefault("vocab_file", vocab_file)
    if merges_file is not None:
        translated.setdefault("merges", merges_file)
        translated.setdefault("merges_file", merges_file)
    if tokenizer_file is not None:
        translated.setdefault("tokenizer_file", tokenizer_file)
    return translated


def _install_qwen2_fast_shim() -> str:
    """Install only the removed Qwen2 fast-module compatibility namespace.

    Transformers 5 moved the fast implementation into ``Qwen2Tokenizer``
    backed by ``TokenizersBackend``. The pinned official source still imports
    the old module and constructor names. The shim maps only those names and
    translates file arguments; it does not replace tokenization behavior or
    provide a model fallback. The real import/API smoke remains mandatory.
    """

    try:
        importlib.import_module(QWEN2_FAST_MODULE)
        return "NATIVE"
    except ModuleNotFoundError as error:
        if error.name != QWEN2_FAST_MODULE:
            raise RuntimeError(
                f"Qwen2 fast tokenizer import failed for another missing dependency: {error.name}"
            ) from error

    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    class Qwen2TokenizerFast(Qwen2Tokenizer):
        """Compatibility name for the Transformers 5 TokenizersBackend class."""

        def __init__(
            self,
            vocab_file: str | None = None,
            merges_file: str | None = None,
            tokenizer_file: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                **_qwen2_fast_init_kwargs(vocab_file, merges_file, tokenizer_file, kwargs)
            )

    Qwen2TokenizerFast.__name__ = "Qwen2TokenizerFast"
    Qwen2TokenizerFast.__qualname__ = "Qwen2TokenizerFast"
    shim = types.ModuleType(QWEN2_FAST_MODULE)
    shim.Qwen2TokenizerFast = Qwen2TokenizerFast
    shim.__all__ = ["Qwen2TokenizerFast"]
    sys.modules[QWEN2_FAST_MODULE] = shim
    return "COMPATIBILITY_SHIM"


def _is_official_registration_pair(auto_model, cls, config_class, model_class) -> bool:
    return (
        cls is auto_model
        and config_class.__module__.startswith("vibevoice.")
        and model_class.__module__.startswith("vibevoice.")
        and config_class.__name__ == "VibeVoiceAcousticTokenizerConfig"
        and model_class.__name__ == "VibeVoiceAcousticTokenizerModel"
    )


@contextlib.contextmanager
def _official_registration_scope():
    """Allow only the known official/source class-name collision once.

    Transformers 5.10.4 ships a native ``VibeVoiceAcousticTokenizerConfig``
    with the same class name as the older pinned Microsoft source. Its auto
    mapping rejects the source registration even though the source is imported
    directly and never asks AutoModel to construct the native class. The
    scoped adapter permits exactly that source config/model pair with
    ``exist_ok=True`` and leaves every other registration unchanged.
    """

    from transformers.models.auto import AutoModel

    original_owner = next(base for base in AutoModel.__mro__ if "register" in base.__dict__)
    original_descriptor = original_owner.__dict__["register"]
    had_own_register = "register" in AutoModel.__dict__
    own_register = AutoModel.__dict__.get("register")
    original_function = original_descriptor.__func__

    def scoped_register(cls, config_class, model_class, exist_ok=False):
        source_pair = _is_official_registration_pair(
            AutoModel, cls, config_class, model_class
        )
        return original_function(cls, config_class, model_class, exist_ok=exist_ok or source_pair)

    AutoModel.register = classmethod(scoped_register)
    try:
        yield REGISTRATION_COMPATIBILITY
    finally:
        if had_own_register:
            AutoModel.register = own_register
        else:
            delattr(AutoModel, "register")


def _compatibility_check(source_root: Path) -> dict[str, Any]:
    """Import pinned upstream classes and inspect their model-free API.

    The upstream project currently declares a Transformers ``<5.0.0``
    requirement while this isolated environment uses patched ``5.10.4`` for
    GHSA-xrqw-3rrv-vx5w. This gate therefore checks the actual pinned source
    import and API surface before any checkpoint is constructed or loaded.
    """

    import transformers

    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(
            "Transformers compatibility check requires the isolated pin "
            f"{TRANSFORMERS_PIN}, got {transformers.__version__}"
        )
    qwen2_fast_import = _install_qwen2_fast_shim()
    with _official_registration_scope() as registration_compatibility:
        config_class, model_class = _load_official(source_root)
    required_config_attrs = {"model_type", "from_dict", "get_text_config"}
    missing_config = sorted(name for name in required_config_attrs if not hasattr(config_class, name))
    if missing_config:
        raise RuntimeError(
            f"official config API missing under Transformers {TRANSFORMERS_PIN}: {missing_config}"
        )
    required_model_attrs = {"forward_lm", "forward_tts_lm"}
    missing_model = sorted(name for name in required_model_attrs if not hasattr(model_class, name))
    if missing_model:
        raise RuntimeError(
            f"official model API missing under Transformers {TRANSFORMERS_PIN}: {missing_model}"
        )
    signatures = {
        "forward_lm": inspect.signature(model_class.forward_lm),
        "forward_tts_lm": inspect.signature(model_class.forward_tts_lm),
    }
    required_parameters = {
        "forward_lm": {"input_ids", "attention_mask", "use_cache", "return_dict"},
        "forward_tts_lm": {
            "input_ids",
            "attention_mask",
            "lm_last_hidden_state",
            "tts_text_masks",
            "use_cache",
            "return_dict",
        },
    }
    missing_parameters = {
        name: sorted(parameters - set(signatures[name].parameters))
        for name, parameters in required_parameters.items()
    }
    missing_parameters = {name: values for name, values in missing_parameters.items() if values}
    if missing_parameters:
        raise RuntimeError(
            f"official VibeVoice forward API is incompatible with the runner: {missing_parameters}"
        )
    from transformers.cache_utils import DynamicCache

    if "config" not in inspect.signature(DynamicCache.__init__).parameters:
        raise RuntimeError(
            "Transformers DynamicCache lacks the config parameter required by the pinned VibeVoice source"
        )
    result = {
        "status": "AUTHENTICATED_API_SMOKE",
        "transformers": TRANSFORMERS_PIN,
        "source_revision": SOURCE_REVISION,
        "qwen2_fast_import": qwen2_fast_import,
        "registration_compatibility": registration_compatibility,
        "official_config": f"{config_class.__module__}.{config_class.__name__}",
        "official_model": f"{model_class.__module__}.{model_class.__name__}",
        "forward_parameters": {
            name: sorted(signature.parameters) for name, signature in signatures.items()
        },
        "dynamic_cache_config_parameter": True,
        "model_download": "NO_MODEL_DOWNLOAD",
        "model_execution": "NO_MODEL_EXECUTION",
        "publication": "NO_UPLOAD",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_model(source_root: Path, config_path: Path, weights_path: Path):
    import torch
    from safetensors.torch import load_file

    config_class, model_class = _load_official(source_root)
    with config_path.open("r", encoding="utf-8") as stream:
        config_data = json.load(stream)
    config = config_class.from_dict(config_data)
    model = model_class(config)
    state_dict = load_file(str(weights_path), device="cpu")
    if len(state_dict) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(f"checkpoint tensor count mismatch: {len(state_dict)} != {EXPECTED_TENSOR_COUNT}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefix = "model.acoustic_tokenizer.encoder."
    missing_encoder = sorted(
        name for name in missing if name.startswith(allowed_missing_prefix)
    )
    disallowed_missing = sorted(
        name for name in missing if not name.startswith(allowed_missing_prefix)
    )
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "official checkpoint did not bind exactly: "
            f"missing={disallowed_missing!r}, unexpected={list(unexpected)!r}"
        )
    model.eval()
    missing_encoder_sha256 = hashlib.sha256(
        "\n".join(missing_encoder).encode("utf-8")
    ).hexdigest()
    return torch, model, {
        "allowed_prefix": allowed_missing_prefix,
        "count": len(missing_encoder),
        "names_sha256": missing_encoder_sha256,
    }


def _inputs(torch, device: str, hidden_size: int, floating_dtype) -> dict[str, Any]:
    # These IDs are deliberately an explicit deterministic probe, not a hidden
    # tokenizer dependency.  All are valid entries in the pinned Qwen vocabulary.
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    tts_text_masks = torch.ones_like(input_ids, dtype=torch.long)
    acoustic_latent = torch.linspace(
        -1.0, 1.0, EXPECTED_ACOUSTIC_DIM, dtype=floating_dtype, device=device
    ).reshape(1, 1, EXPECTED_ACOUSTIC_DIM)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tts_text_masks": tts_text_masks,
        "acoustic_latent": acoustic_latent,
    }


def _run_once(torch, model, device: str) -> dict[str, Any]:
    floating_dtype = next(model.acoustic_connector.parameters()).dtype
    inputs = _inputs(torch, device, EXPECTED_HIDDEN_SIZE, floating_dtype)
    with torch.inference_mode():
        lm = model.forward_lm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        tts = model.forward_tts_lm(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            lm_last_hidden_state=lm.last_hidden_state,
            tts_text_masks=inputs["tts_text_masks"],
            use_cache=False,
            return_dict=True,
        )
        acoustic = model.acoustic_connector(inputs["acoustic_latent"])
    return {
        "eos_logits": tts.logits.detach().to("cpu", dtype=torch.float32),
        "lm_last_hidden_state": lm.last_hidden_state.detach().to("cpu", dtype=torch.float32),
        "tts_last_hidden_state": tts.last_hidden_state.detach().to("cpu", dtype=torch.float32),
        "acoustic_connector": acoustic.detach().to("cpu", dtype=torch.float32),
    }


def _timed(torch, model, device: str, repeats: int, warmups: int) -> tuple[dict[str, Any], list[float]]:
    for _ in range(warmups):
        _run_once(torch, model, device)
        if device == "cuda":
            torch.cuda.synchronize()
    samples: list[float] = []
    output: dict[str, Any] | None = None
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = _run_once(torch, model, device)
        if device == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    assert output is not None
    return output, samples


def _compare_outputs(torch, cpu_output: dict[str, Any], cuda_output: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    global_max_abs = 0.0
    global_max_relative = 0.0
    global_finite = True
    for name in cpu_output:
        cpu_values = cpu_output[name]
        cuda_values = cuda_output[name]
        if tuple(cpu_values.shape) != tuple(cuda_values.shape):
            raise RuntimeError(
                f"CPU/CUDA output shape mismatch for {name}: "
                f"{tuple(cpu_values.shape)} != {tuple(cuda_values.shape)}"
            )
        cpu_finite = bool(torch.isfinite(cpu_values).all().item())
        cuda_finite = bool(torch.isfinite(cuda_values).all().item())
        finite = cpu_finite and cuda_finite
        difference = (cpu_values - cuda_values).abs()
        denominator = torch.maximum(
            torch.maximum(cpu_values.abs(), cuda_values.abs()),
            torch.full_like(cpu_values, RELATIVE_EPS),
        )
        relative = difference / denominator
        if finite:
            max_abs = float(difference.max().item())
            max_relative = float(relative.max().item())
            global_max_abs = max(global_max_abs, max_abs)
            global_max_relative = max(global_max_relative, max_relative)
        else:
            max_abs = None
            max_relative = None
        global_finite = global_finite and finite
        diagnostics[name] = {
            "shape": list(cpu_values.shape),
            "finite": finite,
            "cpu_finite": cpu_finite,
            "cuda_finite": cuda_finite,
            "max_abs_difference": max_abs,
            "max_relative_difference": max_relative,
            "differing_elements": int((difference != 0).sum().item()),
            "total_elements": cpu_values.numel(),
        }
    diagnostics["global"] = {
        "finite": global_finite,
        "max_abs_difference": global_max_abs if global_finite else None,
        "max_relative_difference": global_max_relative if global_finite else None,
        "relative_denominator_floor": RELATIVE_EPS,
    }
    return diagnostics


def _tensor_record(array, path: Path) -> dict[str, Any]:
    import numpy as np

    values = array.numpy()
    np.save(path, values, allow_pickle=False)
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "finite": bool(np.isfinite(values).all()),
        "nonzero": bool(np.any(values != 0.0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _record_outputs(output: dict[str, Any], output_dir: Path, prefix: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, values in output.items():
        filename = f"{prefix}_{name}.npy" if prefix else f"{name}.npy"
        records[name] = _tensor_record(values, output_dir / filename)
    return records


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    source_root = args.source_root.resolve()
    config_path = args.config.resolve()
    weights_path = args.weights.resolve()
    output_dir = args.output.resolve()
    source_identity = _source_identity(source_root)
    config_identity = _file_identity(config_path)
    weights_identity = _file_identity(weights_path)
    if config_identity["bytes"] != CONFIG_BYTES or config_identity["sha256"] != CONFIG_SHA256:
        raise RuntimeError(
            "checkpoint config identity mismatch: "
            f"{config_identity['bytes']} bytes/{config_identity['sha256']} != "
            f"{CONFIG_BYTES} bytes/{CONFIG_SHA256}"
        )
    if weights_identity["sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: {weights_identity['sha256']} != {CHECKPOINT_SHA256}"
        )
    if weights_identity["bytes"] < 2_000_000_000:
        raise RuntimeError("checkpoint is unexpectedly small; refusing partial/fixture weights")
    _assert_regular_output_dir(output_dir)
    _compatibility_check(source_root)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    torch, model, checkpoint_binding = _load_model(source_root, config_path, weights_path)
    compute_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    # The exact BF16 checkpoint has already been bound above.  This is a
    # deliberate compute-dtype cast of that same official model, not a second
    # checkpoint or a synthesized fixture.
    model.to(dtype=compute_dtype)
    cpu_output, cpu_samples = _timed(torch, model, "cpu", args.repeats, args.warmups)
    selected_device = "cpu"
    selected_output = cpu_output
    artifact_records: dict[str, Any] = {
        "cpu": _record_outputs(cpu_output, output_dir, "cpu")
    }
    gpu_evidence: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available() and not args.cpu_only:
        model.to("cuda")
        cuda_output, cuda_samples = _timed(torch, model, "cuda", args.repeats, args.warmups)
        artifact_records["cuda"] = _record_outputs(cuda_output, output_dir, "cuda")
        diagnostics = _compare_outputs(torch, cpu_output, cuda_output)
        max_abs = diagnostics["global"]["max_abs_difference"]
        cpu_median = statistics.median(cpu_samples)
        cuda_median = statistics.median(cuda_samples)
        parity_pass = bool(
            diagnostics["global"]["finite"]
            and max_abs is not None
            and max_abs <= CUDA_ATOL
        )
        gpu_evidence.update(
            {
                "device": torch.cuda.get_device_name(0),
                "cpu_seconds": cpu_samples,
                "cuda_seconds": cuda_samples,
                "cpu_median_seconds": cpu_median,
                "cuda_median_seconds": cuda_median,
                "max_abs_difference": max_abs,
                "atol": CUDA_ATOL,
                "diagnostics": diagnostics,
                "parity_pass": parity_pass,
                "faster": bool(cuda_median < cpu_median),
            }
        )
        if parity_pass and cuda_median < cpu_median:
            selected_device = "cuda"
            selected_output = cuda_output
        else:
            model.to("cpu")
    else:
        gpu_evidence["reason"] = "CUDA unavailable or --cpu-only"

    artifact_records["selected"] = _record_outputs(selected_output, output_dir, "")
    packet = {
        "format": FORMAT,
        "status": "REFERENCE_RUN_OPEN_NOT_RUST_PARITY",
        "execution": "official_microsoft_vibevoice_only",
        "source": source_identity,
        "checkpoint": {
            "repository": "microsoft/VibeVoice-Realtime-0.5B",
            "revision": CHECKPOINT_REVISION,
            **weights_identity,
            "expected_tensor_count": EXPECTED_TENSOR_COUNT,
            "allowed_missing": checkpoint_binding,
            "topology_note": (
                "The pinned Realtime checkpoint is decoder-only for the acoustic "
                "tokenizer; only model.acoustic_tokenizer.encoder.* is absent. "
                "The selected official text/TTS/EOS/acoustic-connector path does "
                "not call that encoder."
            ),
        },
        "config": config_identity,
        "official_classes": {
            "model": "vibevoice.modular.modeling_vibevoice_streaming_inference.VibeVoiceStreamingForConditionalGenerationInference",
            "calls": ["forward_lm", "forward_tts_lm", "model.acoustic_connector"],
        },
        "compute_dtype": args.dtype,
        "input": {
            "seed": SEED,
            "input_ids": [[1, 2, 3, 4]],
            "attention_mask": [[1, 1, 1, 1]],
            "tts_text_masks": [[1, 1, 1, 1]],
            "acoustic_latent": {"shape": [1, 1, EXPECTED_ACOUSTIC_DIM], "range": [-1.0, 1.0]},
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": sys.platform,
            "pid": os.getpid(),
        },
        "timing": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "cpu_seconds": cpu_samples,
            "selected_device": selected_device,
            "gpu": gpu_evidence,
        },
        "artifacts": artifact_records,
        "parity": "CUDA_COMPARED_WITH_FIXED_ATOL" if "parity_pass" in gpu_evidence else "CPU_ONLY",
        "publication": "NO_UPLOAD",
    }
    with (output_dir / "reference.json").open("w", encoding="utf-8") as stream:
        json.dump(packet, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


def _registration_scope_self_test() -> None:
    module_names = ("transformers", "transformers.models", "transformers.models.auto")
    missing = object()
    saved_modules = {name: sys.modules.get(name, missing) for name in module_names}
    calls: list[tuple[str, str, bool]] = []

    class BaseAutoModel:
        @classmethod
        def register(cls, config_class, model_class, exist_ok=False):
            calls.append((config_class.__name__, model_class.__name__, exist_ok))

    class FakeAutoModel(BaseAutoModel):
        pass

    exact_config = type("VibeVoiceAcousticTokenizerConfig", (), {})
    exact_model = type("VibeVoiceAcousticTokenizerModel", (), {})
    other_config = type("OtherConfig", (), {})
    other_model = type("OtherModel", (), {})
    exact_config.__module__ = "vibevoice.modular.configuration_vibevoice"
    exact_model.__module__ = "vibevoice.modular.modular_vibevoice_tokenizer"
    other_config.__module__ = "vibevoice.modular"
    other_model.__module__ = "vibevoice.modular"
    transformers_module = types.ModuleType("transformers")
    models_module = types.ModuleType("transformers.models")
    auto_module = types.ModuleType("transformers.models.auto")
    auto_module.AutoModel = FakeAutoModel
    sys.modules.update(
        {
            "transformers": transformers_module,
            "transformers.models": models_module,
            "transformers.models.auto": auto_module,
        }
    )
    original_register = FakeAutoModel.register
    try:
        assert "register" not in FakeAutoModel.__dict__
        with _official_registration_scope() as status:
            assert status == REGISTRATION_COMPATIBILITY
            FakeAutoModel.register(exact_config, exact_model)
            FakeAutoModel.register(other_config, other_model)
        assert calls == [
            ("VibeVoiceAcousticTokenizerConfig", "VibeVoiceAcousticTokenizerModel", True),
            ("OtherConfig", "OtherModel", False),
        ]
        assert "register" not in FakeAutoModel.__dict__
        assert FakeAutoModel.register == original_register
        try:
            with _official_registration_scope():
                raise RuntimeError("self-test exception")
        except RuntimeError as error:
            assert str(error) == "self-test exception"
        else:
            raise AssertionError("registration scope swallowed an exception")
        assert "register" not in FakeAutoModel.__dict__
        assert FakeAutoModel.register == original_register
    finally:
        for name, value in saved_modules.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def self_test() -> None:
    assert len(SOURCE_REVISION) == 40
    assert len(CHECKPOINT_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == 64
    assert TRANSFORMERS_PIN == "5.10.4"
    translated = _qwen2_fast_init_kwargs("vocab.json", "merges.txt", "tokenizer.json", {"unk_token": "<unk>"})
    assert translated["vocab"] == "vocab.json"
    assert translated["merges"] == "merges.txt"
    assert translated["tokenizer_file"] == "tokenizer.json"
    assert REGISTRATION_COMPATIBILITY == "SCOPED_VIBEVOICE_ACOUSTIC_TOKENIZER_OVERRIDE"
    _registration_scope_self_test()
    assert EXPECTED_TENSOR_COUNT == 605
    assert EXPECTED_HIDDEN_SIZE == 896
    assert EXPECTED_ACOUSTIC_DIM == 64
    print("vibevoice realtime official reference self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--compatibility-check",
        action="store_true",
        help="import the pinned official source and inspect its API without model construction",
    )
    parser.add_argument("--source-root", type=Path, help="pinned Microsoft/VibeVoice git checkout")
    parser.add_argument("--config", type=Path, help="pinned checkpoint config.json")
    parser.add_argument("--weights", type=Path, help="pinned model.safetensors")
    parser.add_argument("--output", type=Path, help="directory for the reference packet")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.compatibility_check:
        if args.source_root is None:
            parser.error("--compatibility-check requires --source-root")
        _source_identity(args.source_root.resolve())
        _compatibility_check(args.source_root.resolve())
        return
    if any(value is None for value in (args.source_root, args.config, args.weights, args.output)):
        parser.error("--source-root, --config, --weights, and --output are required")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
