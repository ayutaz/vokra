#!/usr/bin/env python3
"""VAST-only official SGMSE enhancement packet and native comparator.

The reference path calls the pinned SpeechBrain ``SGMSEEnhancement.enhance_batch``
waveform wrapper, which reaches the upstream ``ScoreModel.enhance`` sampler.
It captures every prior/corrector/predictor noise tensor consumed by that
official sampler and never reimplements the sampler in Python. The comparator
only consumes a verified packet and compares the raw native PCM output.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from sgmse_dump_reference import (
    CHECKPOINT_NAME,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_LICENSE_SPDX,
    EMA_ROUTE_STATUS,
    HYPERPARAMS_RAW,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_LICENSE_SHA256,
    SOURCE_LICENSE_SPDX,
    SOURCE_REVISION,
    SPEECHBRAIN_LICENSE_SHA256,
    SPEECHBRAIN_LICENSE_SPDX,
    SPEECHBRAIN_REVISION,
    SCORE_MODEL_CONFIG,
    atomic_rename_noreplace,
    cpu_model,
    load_score_model,
    path_overlaps,
    require_absent_output,
    require_clean_revision,
    require_disjoint_inputs,
    require_manifest_identity,
    require_vokra_checkout,
    require_source_file,
    sha256,
    verify_algorithm_source,
    verify_ema_route,
    verify_hyperparams_file,
)


PACKET_FORMAT = "vokra-sgmse-native-enhancement-reference-v1"
PACKET_STATUS = "REFERENCE_COMPLETE_NO_UPLOAD"
COMPARISON_FORMAT = "vokra-sgmse-native-enhancement-comparison-v1"
HARNESS_STATUS = "HARNESS_READY"
CPU_STATUS_BEFORE_RUN = "CPU_ENHANCEMENT_PARITY_NOT_RUN"
APPLE_STATUS = "APPLE_NOT_RUN"
PUBLICATION_STATUS = "NO_UPLOAD"
INPUT_WAV_SHA256 = "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"
INPUT_WAV_SIZE = 64044
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
FP32_ATOL = 0.01
NOISE_CALL_COUNT = 61
NOISE_CALLS_NAME = "noise_calls.txt"
NOISE_NAME = "noise.f32"
INPUT_NAME = "input_pcm.f32"
REFERENCE_NAME = "enhanced_pcm.f32"
RUN_LOG_NAME = "run.log"
MANIFEST_NAME = "manifest.json"
EXPECTED_PACKET_FILES = {
    MANIFEST_NAME,
    RUN_LOG_NAME,
    INPUT_NAME,
    REFERENCE_NAME,
    NOISE_NAME,
    NOISE_CALLS_NAME,
}
EXPECTED_NATIVE_FILES = {"enhanced_pcm.f32"}
ENHANCEMENT_SOURCE_FILE = "speechbrain/inference/enhancement.py"
ENHANCEMENT_SOURCE_MARKERS = ("class SGMSEEnhancement", "enhance_batch")
OFFICIAL_CALL_IDENTITY = "SGMSEEnhancement.enhance_batch -> ScoreModel.enhance"
SAMPLING_KEYS = (
    "sampler_type", "predictor", "corrector", "N", "corrector_steps", "snr"
)


def validate_official_enhance_signature(signature: inspect.Signature) -> None:
    """Pin the upstream model.py API without requiring a denoise parameter."""
    parameters = signature.parameters
    required = {
        "y", "sampler_type", "predictor", "corrector", "N",
        "corrector_steps", "snr", "timeit",
    }
    if not required.issubset(parameters) or "denoise" in parameters:
        raise ValueError(f"official ScoreModel.enhance signature drifted: {signature}")
    if parameters["timeit"].default is inspect.Parameter.empty or not any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    ):
        raise ValueError(f"official ScoreModel.enhance must expose timeit/**kwargs: {signature}")


def verify_official_enhancement_source(
    speechbrain_source: Path, inspection: dict[str, Any]
) -> dict[str, Any]:
    """Require the reviewed SpeechBrain waveform wrapper before importing it."""
    source_evidence = inspection.get("speechbrain_source", {})
    executable_files = source_evidence.get("executable_files")
    row = executable_files.get(ENHANCEMENT_SOURCE_FILE) if isinstance(executable_files, dict) else None
    if not isinstance(row, dict) or set(row) != {"sha256", "size", "required_markers"}:
        raise ValueError("inspection lacks the reviewed SpeechBrain enhancement wrapper")
    if (
        not isinstance(row["sha256"], str)
        or len(row["sha256"]) != 64
        or not isinstance(row["size"], int)
        or row["size"] <= 0
        or row["required_markers"] != {marker: True for marker in ENHANCEMENT_SOURCE_MARKERS}
    ):
        raise ValueError("SpeechBrain enhancement source evidence is malformed")
    return require_source_file(
        speechbrain_source,
        ENHANCEMENT_SOURCE_FILE,
        row["sha256"],
        ENHANCEMENT_SOURCE_MARKERS,
    )


def verify_speechbrain_source_manifest(
    speechbrain: dict[str, Any],
    ema_route: dict[str, Any],
    runtime: dict[str, Any],
    *,
    allow_missing_source_for_self_test: bool,
) -> None:
    """Re-hash the exact clean SpeechBrain checkout named by the packet.

    The model-free self-test explicitly opts into synthetic source rows. A
    generated VAST packet must carry a real clean checkout; its
    three executable source files are re-hashed at verification time rather
    than trusting hashes copied into the manifest.
    """
    source_path = Path(speechbrain["path"])
    if allow_missing_source_for_self_test and not source_path.exists():
        return
    if not source_path.is_absolute() or source_path.is_symlink():
        raise ValueError("SpeechBrain source checkout must be an absolute non-symlink directory")
    if speechbrain.get("clean") is not True:
        raise ValueError("SpeechBrain source checkout is not marked clean")
    observed = require_clean_revision(
        source_path, SPEECHBRAIN_REVISION, "SpeechBrain source"
    )
    if observed["path"] != str(source_path):
        raise ValueError("SpeechBrain source path was normalized unexpectedly")

    expected_files: dict[str, dict[str, Any]] = {}
    source_rows = speechbrain.get("files")
    if isinstance(source_rows, list):
        for row in source_rows:
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                expected_files[row["path"]] = row
    ema_files = ema_route.get("source_files")
    if isinstance(ema_files, dict):
        for row in ema_files.values():
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                expected_files[row["path"]] = row
    required = {
        ENHANCEMENT_SOURCE_FILE: ENHANCEMENT_SOURCE_MARKERS,
        "speechbrain/integrations/models/sgmse_plus.py": ("class ScoreModel",),
        "speechbrain/utils/parameter_transfer.py": (
            "class Pretrainer",
            "filename = name + PARAMFILE_EXT",
            "def load_collected",
        ),
    }
    if set(expected_files) != set(required):
        raise ValueError("SpeechBrain source file inventory is incomplete")
    for relative, markers in required.items():
        row = expected_files[relative]
        if (
            not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or not isinstance(row.get("size"), int)
            or row["size"] <= 0
        ):
            raise ValueError(f"SpeechBrain source evidence is malformed: {relative}")
        actual = require_source_file(source_path, relative, row["sha256"], markers)
        if actual["size"] != row["size"]:
            raise ValueError(f"SpeechBrain source size differs: {relative}")


def reviewed_sampling_config(hyperparams_evidence: dict[str, Any]) -> dict[str, Any]:
    """Parse the exact reviewed YAML sampling block, without a YAML import."""
    raw = hyperparams_evidence.get("raw")
    if hyperparams_evidence.get("sha256") != hashlib.sha256(HYPERPARAMS_RAW.encode()).hexdigest() or not isinstance(raw, str):
        raise ValueError("wrapper sampling requires the reviewed hyperparams evidence")
    canonical_match = re.search(r"^sampling:\n(?P<body>(?:  [^\n]+\n)+)\nmodules:", HYPERPARAMS_RAW, re.MULTILINE)
    observed_match = re.search(r"^sampling:\n(?P<body>(?:  [^\n]+\n)+)\nmodules:", raw, re.MULTILINE)
    if canonical_match is None or observed_match is None or observed_match.group("body") != canonical_match.group("body"):
        raise ValueError("reviewed hyperparams sampling block drifted")
    values: dict[str, Any] = {}
    for line in observed_match.group("body").splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator or key not in SAMPLING_KEYS or key in values:
            raise ValueError("reviewed hyperparams sampling entries are malformed")
        if key in {"N", "corrector_steps"}:
            values[key] = int(value)
        elif key == "snr":
            values[key] = float(value)
        else:
            values[key] = value
    if tuple(values) != SAMPLING_KEYS:
        raise ValueError("reviewed hyperparams sampling keys drifted")
    return values


def validate_wrapper_sampling(sampling: dict[str, Any]) -> None:
    expected = reviewed_sampling_config(
        {
            "sha256": hashlib.sha256(HYPERPARAMS_RAW.encode()).hexdigest(),
            "raw": HYPERPARAMS_RAW,
        }
    )
    if sampling != expected:
        raise ValueError("wrapper sampling does not match the reviewed hyperparams")


def build_official_enhancer(
    speechbrain_source: Path,
    score_model: Any,
    torch: Any,
    sampling: dict[str, Any],
) -> Any:
    """Construct SpeechBrain's pinned waveform-to-spectrogram wrapper."""
    sys.path.insert(0, str(speechbrain_source))
    from speechbrain.inference.enhancement import SGMSEEnhancement

    signature = inspect.signature(SGMSEEnhancement)
    required = {"modules", "hparams", "run_opts"}
    if not required.issubset(signature.parameters):
        raise ValueError(f"pinned SGMSEEnhancement constructor drifted: {signature}")
    class ReviewedHParams(dict[str, Any]):
        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as error:
                raise AttributeError(name) from error

    validate_wrapper_sampling(sampling)
    hparams = ReviewedHParams({
        "sample_rate": SAMPLE_RATE,
        "n_fft": 510,
        "hop_length": 128,
        "window_type": "hann",
        "transform_type": "exponent",
        "spec_factor": 0.15,
        "spec_abs_exponent": 0.5,
        "sampling": ReviewedHParams(sampling),
    })
    device = torch.device("cuda")
    score_model = score_model.to(device)
    enhancer = SGMSEEnhancement(
        modules={"score_model": score_model},
        hparams=hparams,
        run_opts={"device": "cuda"},
    )
    if hasattr(enhancer, "to"):
        enhancer = enhancer.to(device)
    model_devices = {str(parameter.device) for parameter in score_model.parameters()}
    if model_devices != {str(device)}:
        raise ValueError(f"ScoreModel was not placed on CUDA: {sorted(model_devices)}")
    if str(getattr(enhancer, "device", device)) != str(device):
        raise ValueError("SGMSEEnhancement did not select the CUDA device")
    return enhancer


def call_official_enhancer(
    enhancer: Any, score_model: Any, waveform: Any, torch: Any
) -> tuple[Any, list[dict[str, Any]]]:
    """Call only the official waveform wrapper and tap its score call."""
    original_score_enhance = score_model.enhance
    score_calls: list[dict[str, Any]] = []

    def capture_score_call(*args: Any, **kwargs: Any) -> Any:
        if not args:
            raise ValueError("official ScoreModel.enhance received no spectrogram")
        spectrogram = args[0]
        if not isinstance(spectrogram, torch.Tensor) or spectrogram.ndim != 4 or not torch.is_complex(spectrogram):
            raise ValueError("official ScoreModel.enhance did not receive [B,1,F,T] complex spectrogram")
        if spectrogram.shape[0] != 1 or spectrogram.shape[1] != 1:
            raise ValueError("official ScoreModel.enhance spectrogram batch/channel drifted")
        score_calls.append({"shape": [int(axis) for axis in spectrogram.shape], "dtype": str(spectrogram.dtype)})
        return original_score_enhance(*args, **kwargs)

    score_model.enhance = capture_score_call
    try:
        enhanced = enhancer.enhance_batch(waveform)
    finally:
        score_model.enhance = original_score_enhance
    if len(score_calls) != 1:
        raise ValueError(f"official wrapper invoked ScoreModel.enhance {len(score_calls)} times")
    return enhanced, score_calls


def validate_official_waveform_output(
    enhanced: Any, sample_count: int, torch: Any
) -> Any:
    """Validate the tensor returned by SpeechBrain's official wrapper."""
    if not isinstance(enhanced, torch.Tensor):
        raise ValueError("official SGMSEEnhancement.enhance_batch returned a non-tensor")
    enhanced_tensor = enhanced.detach().cpu().reshape(-1)
    if enhanced_tensor.numel() != sample_count or not bool(torch.isfinite(enhanced_tensor).all()):
        raise ValueError("official enhancement returned invalid PCM")
    return enhanced_tensor


def reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def serialize_noise_planes(real: list[float], imaginary: list[float] | None) -> list[float]:
    """Match Rust's two-plane score layout without discarding complex noise."""
    if imaginary is None:
        return list(real)
    if len(real) != len(imaginary):
        raise ValueError("complex noise planes have different lengths")
    return [*real, *imaginary]


def read_f32(path: Path, label: str) -> list[float]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or symlinked")
    raw = path.read_bytes()
    if len(raw) == 0 or len(raw) % 4:
        raise ValueError(f"{label} has invalid byte length")
    values = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite value")
    return values


def write_f32(path: Path, values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cannot write empty or non-finite f32 payload")
    with path.open("xb") as handle:
        for start in range(0, len(values), 8192):
            chunk = values[start : start + 8192]
            handle.write(struct.pack(f"<{len(chunk)}f", *chunk))
        handle.flush()
        os.fsync(handle.fileno())
    byte_count = len(values) * 4
    return {
        "path": path.name,
        "dtype": "float32",
        "shape": [len(values)],
        "count": len(values),
        "bytes": byte_count,
        "sha256": sha256(path),
    }


def require_exact_files(directory: Path, expected: set[str], label: str) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{label} directory is missing or symlinked")
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != expected:
        raise ValueError(f"{label} has an unexpected or missing file")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{label} contains a non-regular file")


def _read_wav_pcm(path: Path) -> list[float]:
    if path.is_symlink() or not path.is_file() or sha256(path) != INPUT_WAV_SHA256:
        raise ValueError("input WAV is not the reviewed exact fixture")
    if path.stat().st_size != INPUT_WAV_SIZE:
        raise ValueError("input WAV byte count differs from reviewed fixture")
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != SAMPLE_RATE
            or handle.getnchannels() != CHANNELS
            or handle.getsampwidth() != SAMPLE_WIDTH
            or handle.getcomptype() != "NONE"
        ):
            raise ValueError("input WAV is not mono 16-bit PCM at 16 kHz")
        raw = handle.readframes(handle.getnframes())
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    values = [sample / 32768.0 for sample in samples]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("input PCM is empty or non-finite")
    return values


def _validate_noise_calls(packet: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    calls = manifest.get("noise_calls")
    if not isinstance(calls, list) or len(calls) != NOISE_CALL_COUNT:
        raise ValueError("noise call count is not the fixed prior + 60-step contract")
    noise_path = packet / NOISE_NAME
    calls_path = packet / NOISE_CALLS_NAME
    raw = noise_path.read_bytes()
    lines = calls_path.read_text(encoding="ascii").splitlines()
    if len(lines) != len(calls):
        raise ValueError("noise call index file count mismatch")
    expected_offset = 0
    for index, (call, line) in enumerate(zip(calls, lines)):
        if not isinstance(call, dict) or set(call) != {
            "index", "kind", "step", "corrector", "count", "offset", "bytes",
            "sha256", "source_dtype", "source_shape", "serialized_layout"
        }:
            raise ValueError("noise call schema drifted")
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("noise call index line schema drifted")
        line_values = [int(value) for value in fields]
        if line_values != [
            call["index"],
            0 if call["kind"] == "prior" else 1,
            call["step"],
            int(call["corrector"]),
            call["count"],
            call["offset"],
        ]:
            raise ValueError("noise call index is not bound to manifest")
        expected_kind = "prior" if index == 0 else "corrector" if index % 2 else "predictor"
        expected_step = -1 if index == 0 else (index - 1) // 2
        expected_corrector = index != 0 and index % 2 == 1
        if (
            call["index"] != index
            or call["kind"] != expected_kind
            or call["step"] != expected_step
            or call["corrector"] is not expected_corrector
            or not isinstance(call["count"], int)
            or call["count"] <= 0
            or call["offset"] != expected_offset
            or call["bytes"] != call["count"] * 4
            or not isinstance(call["sha256"], str)
            or len(call["sha256"]) != 64
            or call["source_dtype"] not in {"torch.float32", "torch.complex64"}
            or not isinstance(call["source_shape"], list)
            or not call["source_shape"]
            or any(not isinstance(axis, int) or axis <= 0 for axis in call["source_shape"])
            or call["serialized_layout"] not in {"flat", "real_plane_then_imag_plane"}
        ):
            raise ValueError("noise call order/shape metadata is invalid")
        source_count = math.prod(call["source_shape"])
        if call["source_dtype"] == "torch.complex64":
            if call["serialized_layout"] != "real_plane_then_imag_plane" or call["count"] != source_count * 2:
                raise ValueError("complex noise was not serialized as two real planes")
        elif call["serialized_layout"] != "flat" or call["count"] != source_count:
            raise ValueError("real noise serialization metadata is invalid")
        segment = raw[call["offset"] : call["offset"] + call["bytes"]]
        if len(segment) != call["bytes"] or digest_bytes(segment) != call["sha256"]:
            raise ValueError("noise call payload hash mismatch")
        values = struct.unpack(f"<{call['count']}f", segment)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("noise payload contains a non-finite value")
        expected_offset += call["bytes"]
    if expected_offset != len(raw):
        raise ValueError("noise payload has trailing or truncated bytes")
    return calls


def verify_reference(
    packet: Path,
    vokra_root: Path | None = None,
    *,
    allow_missing_source_for_self_test: bool = False,
) -> dict[str, Any]:
    require_exact_files(packet, EXPECTED_PACKET_FILES, "reference packet")
    manifest = json.loads(
        (packet / MANIFEST_NAME).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json,
    )
    if not isinstance(manifest, dict) or set(manifest) != {
        "format", "status", "publication", "model_repository", "model_revision",
        "checkpoint", "source", "speechbrain_source", "licenses", "ema_route", "model", "input",
        "runtime", "artifacts", "noise_calls", "noise_payload", "tolerance",
        "identity", "run_log", "vokra",
    }:
        raise ValueError("reference packet manifest schema drifted")
    if (
        manifest["format"] != PACKET_FORMAT
        or manifest["status"] != PACKET_STATUS
        or manifest["publication"] != "NO_UPLOAD"
        or manifest["model_repository"] != MODEL_REPOSITORY
        or manifest["model_revision"] != MODEL_REVISION
    ):
        raise ValueError("reference packet identity/status mismatch")
    checkpoint = manifest["checkpoint"]
    if checkpoint != {
        "filename": CHECKPOINT_NAME,
        "size": CHECKPOINT_SIZE,
        "sha256": CHECKPOINT_SHA256,
        "license_spdx": CHECKPOINT_LICENSE_SPDX,
    }:
        raise ValueError("reference checkpoint identity mismatch")
    source = manifest["source"]
    speechbrain = manifest["speechbrain_source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"path", "revision", "clean", "repository", "license_spdx", "license_sha256", "files"}
        or source.get("revision") != SOURCE_REVISION
        or source.get("license_spdx") != SOURCE_LICENSE_SPDX
        or source.get("license_sha256") != SOURCE_LICENSE_SHA256
        or not isinstance(speechbrain, dict)
        or set(speechbrain) != {"path", "revision", "clean", "repository", "license_spdx", "license_sha256", "files"}
        or speechbrain.get("revision") != SPEECHBRAIN_REVISION
        or speechbrain.get("license_spdx") != SPEECHBRAIN_LICENSE_SPDX
        or speechbrain.get("license_sha256") != SPEECHBRAIN_LICENSE_SHA256
    ):
        raise ValueError("reference source/license identity mismatch")
    speechbrain_files = speechbrain["files"]
    if (
        not isinstance(speechbrain_files, list)
        or len(speechbrain_files) != 1
        or not isinstance(speechbrain_files[0], dict)
        or speechbrain_files[0].get("path") != ENHANCEMENT_SOURCE_FILE
        or speechbrain_files[0].get("markers") != {marker: True for marker in ENHANCEMENT_SOURCE_MARKERS}
        or speechbrain_files[0].get("sha256") != "019e79bb489ba4c7f1ddd681e0cc007d7034386636ffd156128ec85058476995"
        or speechbrain_files[0].get("size") != 11693
    ):
        raise ValueError("reference SpeechBrain enhancement source evidence mismatch")
    if manifest["licenses"] != {
        "algorithm": {"spdx": SOURCE_LICENSE_SPDX, "sha256": SOURCE_LICENSE_SHA256},
        "speechbrain": {"spdx": SPEECHBRAIN_LICENSE_SPDX, "sha256": SPEECHBRAIN_LICENSE_SHA256},
        "checkpoint": CHECKPOINT_LICENSE_SPDX,
    }:
        raise ValueError("reference license record mismatch")
    ema_route = manifest["ema_route"]
    source_files = ema_route.get("source_files") if isinstance(ema_route, dict) else None
    score_source = source_files.get("score_model") if isinstance(source_files, dict) else None
    transfer_source = source_files.get("parameter_transfer") if isinstance(source_files, dict) else None
    if (
        not isinstance(ema_route, dict)
        or ema_route.get("status") != EMA_ROUTE_STATUS
        or ema_route.get("unsafe_pickle_fallback") is not False
        or ema_route.get("parameter_load") != "strict_state_dict"
        or ema_route.get("loadable") != "score_model_ema"
        or ema_route.get("checkpoint_filename") != CHECKPOINT_NAME
        or not isinstance(score_source, dict)
        or score_source.get("path") != "speechbrain/integrations/models/sgmse_plus.py"
        or score_source.get("sha256") != "b70ecde1d7326282b339348c739e91413c6dbac07ef98d34b540be07d8e70935"
        or score_source.get("size") != 21777
        or not isinstance(transfer_source, dict)
        or transfer_source.get("path") != "speechbrain/utils/parameter_transfer.py"
        or not isinstance(transfer_source.get("sha256"), str)
        or len(transfer_source["sha256"]) != 64
        or not isinstance(transfer_source.get("size"), int)
        or transfer_source["size"] <= 0
    ):
        raise ValueError("reference EMA route evidence is missing or not strict")
    model = manifest["model"]
    if (
        not isinstance(model, dict)
        or model.get("load") != "torch.load(weights_only=True)+load_state_dict(strict=True)"
        or model.get("tensor_count") != 647
        or model.get("parameter_count") != 65_590_822
    ):
        raise ValueError("reference strict model-load evidence is missing or mismatched")
    if manifest["input"] != {
        "wav_filename": "ref-clip.wav",
        "wav_size": INPUT_WAV_SIZE,
        "wav_sha256": INPUT_WAV_SHA256,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sample_width": SAMPLE_WIDTH,
        "pcm_filename": INPUT_NAME,
    }:
        raise ValueError("reference input identity mismatch")
    if manifest["tolerance"] != {
        "metric": "waveform_max_abs_and_rmse",
        "max_abs": FP32_ATOL,
        "rmse": FP32_ATOL,
        "basis": "repository FP32_ATOL=0.01; preregistered before real run",
    }:
        raise ValueError("reference tolerance contract drifted")
    runtime = manifest["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"platform_system", "platform_machine", "platform_node", "cpu_model", "nproc", "torch_version", "numpy_version"}
        or runtime.get("platform_system") != "Linux"
        or runtime.get("platform_machine") != "x86_64"
        or not runtime.get("torch_version")
        or not runtime.get("numpy_version")
    ):
        raise ValueError("reference runtime is not VAST Linux x86_64")
    verify_speechbrain_source_manifest(
        speechbrain,
        ema_route,
        runtime,
        allow_missing_source_for_self_test=allow_missing_source_for_self_test,
    )
    vokra = manifest["vokra"]
    if not isinstance(vokra, dict) or set(vokra) != {
        "path", "commit", "clean", "tool_sha256", "uv_lock_sha256"
    }:
        raise ValueError("reference Vokra provenance schema drifted")
    if vokra.get("clean") is not True or not isinstance(vokra.get("commit"), str):
        raise ValueError("reference Vokra provenance is not clean")
    if vokra_root is not None:
        current = require_vokra_checkout(vokra_root)
        if (
            Path(vokra["path"]).resolve(strict=False) != vokra_root.resolve(strict=False)
            or vokra["commit"] != current["commit"]
            or vokra["tool_sha256"] != sha256(vokra_root / "tools/parity/sgmse_native_enhancement_parity.py")
            or vokra["uv_lock_sha256"] != sha256(vokra_root / "tools/parity/uv.lock")
        ):
            raise ValueError("reference Vokra checkout/tool/lock provenance mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {INPUT_NAME, REFERENCE_NAME, NOISE_NAME, NOISE_CALLS_NAME}:
        raise ValueError("reference artifact set mismatch")
    for filename, item in artifacts.items():
        path = packet / filename
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "dtype", "shape", "count", "bytes", "sha256"}
            or item.get("path") != filename
            or item.get("dtype") not in {"float32", "ascii"}
            or not isinstance(item.get("shape"), list)
            or not isinstance(item.get("count"), int)
            or item.get("count") <= 0
            or item.get("bytes") != path.stat().st_size
            or item.get("sha256") != sha256(path)
        ):
            raise ValueError(f"reference artifact identity mismatch: {filename}")
        if filename != NOISE_CALLS_NAME:
            if item["dtype"] != "float32" or item["shape"] != [item["count"]] or item["bytes"] != item["count"] * 4:
                raise ValueError(f"reference float artifact shape mismatch: {filename}")
        elif item["dtype"] != "ascii" or item["shape"] != [len((packet / filename).read_text(encoding="ascii").splitlines())]:
            raise ValueError("reference noise index shape mismatch")
    input_values = read_f32(packet / INPUT_NAME, "reference input PCM")
    output_values = read_f32(packet / REFERENCE_NAME, "reference enhanced PCM")
    if len(input_values) != len(output_values) or len(input_values) == 0:
        raise ValueError("reference PCM lengths differ or are empty")
    _validate_noise_calls(packet, manifest)
    payload = manifest["noise_payload"]
    if payload != {"filename": NOISE_NAME, "call_count": NOISE_CALL_COUNT, "dtype": "float32", "complete": True}:
        raise ValueError("noise payload is not complete")
    identity = manifest["identity"]
    if identity != {
        "reference_tool": "sgmse_native_enhancement_parity.py",
        "official_call": OFFICIAL_CALL_IDENTITY,
        "noise_capture": "torch.randn_like+torch.randn during one official enhance call",
        "self_test": "sgmse_native_enhancement_parity.py --self-test",
    }:
        raise ValueError("reference implementation identity mismatch")
    run_log = manifest["run_log"]
    if run_log != {"path": RUN_LOG_NAME, "size": (packet / RUN_LOG_NAME).stat().st_size, "sha256": sha256(packet / RUN_LOG_NAME)}:
        raise ValueError("reference run log identity mismatch")
    run_log_text = (packet / RUN_LOG_NAME).read_text(encoding="utf-8")
    if run_log_text.count("status=REFERENCE_COMPLETE_NO_UPLOAD") != 1:
        raise ValueError("reference run log completion marker is missing")
    return manifest


def compare_native(
    reference_dir: Path, native_dir: Path, vokra_root: Path
) -> dict[str, Any]:
    if not reference_dir.is_absolute() or not native_dir.is_absolute() or path_overlaps(reference_dir, native_dir):
        raise ValueError("reference and native paths must be absolute and disjoint")
    manifest = verify_reference(reference_dir, vokra_root)
    require_exact_files(native_dir, EXPECTED_NATIVE_FILES, "native enhancement output")
    expected = read_f32(reference_dir / REFERENCE_NAME, "reference enhanced PCM")
    actual = read_f32(native_dir / "enhanced_pcm.f32", "native enhanced PCM")
    if len(actual) != len(expected):
        raise ValueError("native/reference PCM lengths differ")
    deltas = [abs(left - right) for left, right in zip(actual, expected)]
    max_abs = max(deltas)
    rmse = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
    result = {
        "format": COMPARISON_FORMAT,
        "status": "CPU_ENHANCEMENT_PARITY_PASS" if max_abs <= FP32_ATOL and rmse <= FP32_ATOL else "CPU_ENHANCEMENT_PARITY_FAIL",
        "reference_status": manifest["status"],
        "publication": "NO_UPLOAD",
        "max_abs": max_abs,
        "rmse": rmse,
        "atol": FP32_ATOL,
        "samples": len(expected),
    }
    return result


def _run_official_reference(
    source: Path,
    speechbrain_source: Path,
    checkpoint: Path,
    hyperparams: Path,
    inspection_manifest: Path,
    input_wav: Path,
    output_dir: Path,
    vokra_root: Path,
) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("SGMSE enhancement reference is VAST/Linux x86_64-only")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ValueError("VOKRA_PUBLISH_ON_VAST=1 is required for reference execution")
    all_paths = (source, speechbrain_source, checkpoint, hyperparams, inspection_manifest, input_wav, output_dir, vokra_root)
    if any(not path.is_absolute() for path in all_paths):
        raise ValueError("all reference paths must be absolute")
    # The fixed input fixture is intentionally inside this authenticated
    # Vokra checkout; all mutable/model/source trees remain disjoint.
    require_disjoint_inputs(
        (vokra_root, inspection_manifest.parent, checkpoint.parent, source, speechbrain_source)
    )
    if not inspection_manifest.is_file() or inspection_manifest.is_symlink():
        raise ValueError("inspection manifest is missing or symlinked")
    inspection = json.loads(inspection_manifest.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json)
    require_manifest_identity(inspection)
    hyperparams_evidence = verify_hyperparams_file(hyperparams, inspection)
    if checkpoint.name != CHECKPOINT_NAME or checkpoint.stat().st_size != CHECKPOINT_SIZE or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the fixed identity")
    source_tree = require_clean_revision(source, SOURCE_REVISION, "SGMSE source")
    speechbrain_tree = require_clean_revision(speechbrain_source, SPEECHBRAIN_REVISION, "SpeechBrain source")
    vokra_tree = require_vokra_checkout(vokra_root)
    require_absent_output(output_dir, (vokra_root, inspection_manifest.parent, checkpoint.parent, source, speechbrain_source, input_wav))
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        import numpy as np
        import torch

        ema_route = verify_ema_route(hyperparams, speechbrain_source, inspection["safe_load"], inspection)
        enhancement_source = verify_official_enhancement_source(speechbrain_source, inspection)
        algorithm_files = verify_algorithm_source(source, inspection)
        model, model_evidence = load_score_model(source, speechbrain_source, checkpoint, SCORE_MODEL_CONFIG)
        if model_evidence["tensor_count"] != 647 or model_evidence["parameter_count"] != 65_590_822:
            raise ValueError("loaded checkpoint count differs from fixed evidence")
        if not torch.cuda.is_available():
            raise ValueError("official SGMSEEnhancement reference requires CUDA")
        sampling = reviewed_sampling_config(hyperparams_evidence | {"raw": HYPERPARAMS_RAW})
        enhancer = build_official_enhancer(speechbrain_source, model, torch, sampling)
        pcm = _read_wav_pcm(input_wav)
        input_artifact = write_f32(temporary / INPUT_NAME, pcm)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(20260901)
        np.random.seed(20260901)
        calls: list[dict[str, Any]] = []
        original_randn_like = torch.randn_like
        original_randn = torch.randn

        def capture_like(input_tensor: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_randn_like(input_tensor, *args, **kwargs)
            calls.append({"tensor": result.detach().cpu().contiguous()})
            return result

        def capture_randn(*args: Any, **kwargs: Any) -> Any:
            result = original_randn(*args, **kwargs)
            calls.append({"tensor": result.detach().cpu().contiguous()})
            return result

        torch.randn_like = capture_like
        torch.randn = capture_randn
        try:
            signature = inspect.signature(model.enhance)
            validate_official_enhance_signature(signature)
            waveform = torch.from_numpy(np.asarray(pcm, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                enhanced, score_calls = call_official_enhancer(enhancer, model, waveform, torch)
        finally:
            torch.randn_like = original_randn_like
            torch.randn = original_randn
        if len(calls) != NOISE_CALL_COUNT:
            raise ValueError(f"official enhance consumed {len(calls)} noise tensors, expected {NOISE_CALL_COUNT}")
        enhanced_tensor = validate_official_waveform_output(enhanced, len(pcm), torch)
        enhanced_values = enhanced_tensor.numpy().astype(np.float32).tolist()
        reference_artifact = write_f32(temporary / REFERENCE_NAME, enhanced_values)
        noise_path = temporary / NOISE_NAME
        noise_rows: list[dict[str, Any]] = []
        offset = 0
        call_lines: list[str] = []
        with noise_path.open("xb") as noise_handle:
            for index, captured in enumerate(calls):
                tensor = captured["tensor"]
                source_dtype = str(tensor.dtype)
                source_shape = [int(axis) for axis in tensor.shape]
                if source_dtype == "torch.complex64":
                    real = tensor.real.numpy().astype(np.float32).reshape(-1).tolist()
                    imaginary = tensor.imag.numpy().astype(np.float32).reshape(-1).tolist()
                    flattened = serialize_noise_planes(real, imaginary)
                    serialized_layout = "real_plane_then_imag_plane"
                elif source_dtype == "torch.float32":
                    flattened = tensor.numpy().astype(np.float32).reshape(-1).tolist()
                    serialized_layout = "flat"
                else:
                    raise ValueError(f"official noise dtype is unsupported: {source_dtype}")
                raw_digest = hashlib.sha256()
                for start in range(0, len(flattened), 8192):
                    chunk = struct.pack(
                        f"<{min(8192, len(flattened) - start)}f",
                        *flattened[start : start + 8192],
                    )
                    noise_handle.write(chunk)
                    raw_digest.update(chunk)
                kind = "prior" if index == 0 else "corrector" if index % 2 else "predictor"
                step = -1 if index == 0 else (index - 1) // 2
                corrector = index != 0 and index % 2 == 1
                byte_count = len(flattened) * 4
                noise_rows.append({"index": index, "kind": kind, "step": step, "corrector": corrector, "count": len(flattened), "offset": offset, "bytes": byte_count, "sha256": raw_digest.hexdigest(), "source_dtype": source_dtype, "source_shape": source_shape, "serialized_layout": serialized_layout})
                call_lines.append(f"{index} {0 if kind == 'prior' else 1} {step} {int(corrector)} {len(flattened)} {offset}")
                offset += byte_count
            noise_handle.flush()
            os.fsync(noise_handle.fileno())
        (temporary / NOISE_CALLS_NAME).write_text("\n".join(call_lines) + "\n", encoding="ascii")
        artifacts = {
            INPUT_NAME: input_artifact,
            REFERENCE_NAME: reference_artifact,
            NOISE_NAME: {
                "path": NOISE_NAME,
                "dtype": "float32",
                "shape": [offset // 4],
                "count": offset // 4,
                "bytes": offset,
                "sha256": sha256(noise_path),
            },
            NOISE_CALLS_NAME: {
                "path": NOISE_CALLS_NAME,
                "dtype": "ascii",
                "shape": [len(call_lines)],
                "count": len(call_lines),
                "bytes": (temporary / NOISE_CALLS_NAME).stat().st_size,
                "sha256": sha256(temporary / NOISE_CALLS_NAME),
            },
        }
        run_log = temporary / RUN_LOG_NAME
        run_log.write_text(
            "reference=vokra-sgmse-native-enhancement-reference-v1\n"
            "status=REFERENCE_COMPLETE_NO_UPLOAD\n"
            "official_call=SGMSEEnhancement.enhance_batch -> ScoreModel.enhance\n"
            "noise_capture=prior+corrector+predictor\n"
            "publication=NO_UPLOAD\n",
            encoding="utf-8",
        )
        manifest = {
            "format": PACKET_FORMAT,
            "status": PACKET_STATUS,
            "publication": "NO_UPLOAD",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "checkpoint": {"filename": checkpoint.name, "size": checkpoint.stat().st_size, "sha256": sha256(checkpoint), "license_spdx": CHECKPOINT_LICENSE_SPDX},
            "source": {**source_tree, "repository": "https://github.com/sp-uhh/sgmse.git", "revision": SOURCE_REVISION, "license_spdx": SOURCE_LICENSE_SPDX, "license_sha256": SOURCE_LICENSE_SHA256, "files": algorithm_files},
            "speechbrain_source": {**speechbrain_tree, "repository": "https://github.com/speechbrain/speechbrain.git", "revision": SPEECHBRAIN_REVISION, "license_spdx": SPEECHBRAIN_LICENSE_SPDX, "license_sha256": SPEECHBRAIN_LICENSE_SHA256, "files": [enhancement_source]},
            "licenses": {"algorithm": {"spdx": SOURCE_LICENSE_SPDX, "sha256": SOURCE_LICENSE_SHA256}, "speechbrain": {"spdx": SPEECHBRAIN_LICENSE_SPDX, "sha256": SPEECHBRAIN_LICENSE_SHA256}, "checkpoint": CHECKPOINT_LICENSE_SPDX},
            "ema_route": ema_route,
            "model": model_evidence,
            "input": {"wav_filename": input_wav.name, "wav_size": input_wav.stat().st_size, "wav_sha256": sha256(input_wav), "sample_rate": SAMPLE_RATE, "channels": CHANNELS, "sample_width": SAMPLE_WIDTH, "pcm_filename": INPUT_NAME},
            "runtime": {"platform_system": platform.system(), "platform_machine": platform.machine(), "platform_node": platform.node(), "cpu_model": cpu_model(), "nproc": os.cpu_count(), "torch_version": torch.__version__, "numpy_version": np.__version__},
            "artifacts": artifacts,
            "noise_calls": noise_rows,
            "noise_payload": {"filename": NOISE_NAME, "call_count": len(noise_rows), "dtype": "float32", "complete": True},
            "tolerance": {"metric": "waveform_max_abs_and_rmse", "max_abs": FP32_ATOL, "rmse": FP32_ATOL, "basis": "repository FP32_ATOL=0.01; preregistered before real run"},
            "identity": {"reference_tool": "sgmse_native_enhancement_parity.py", "official_call": OFFICIAL_CALL_IDENTITY, "noise_capture": "torch.randn_like+torch.randn during one official enhance call", "self_test": "sgmse_native_enhancement_parity.py --self-test"},
            "vokra": {**vokra_tree, "tool_sha256": sha256(vokra_root / "tools/parity/sgmse_native_enhancement_parity.py"), "uv_lock_sha256": sha256(vokra_root / "tools/parity/uv.lock")},
            "run_log": {"path": RUN_LOG_NAME, "size": run_log.stat().st_size, "sha256": sha256(run_log)},
        }
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_reference(temporary, vokra_root)
        if os.path.lexists(output_dir):
            raise ValueError("reference output appeared before no-replace publication")
        atomic_rename_noreplace(temporary, output_dir)
        temporary = Path()
        return manifest
    finally:
        if str(temporary) != ".":
            shutil.rmtree(temporary, ignore_errors=True)


def self_test() -> int:
    assert FP32_ATOL == 0.01
    assert NOISE_CALL_COUNT == 61
    assert PACKET_STATUS == "REFERENCE_COMPLETE_NO_UPLOAD"
    assert (HARNESS_STATUS, CPU_STATUS_BEFORE_RUN, APPLE_STATUS, PUBLICATION_STATUS) == (
        "HARNESS_READY",
        "CPU_ENHANCEMENT_PARITY_NOT_RUN",
        "APPLE_NOT_RUN",
        "NO_UPLOAD",
    )
    assert "ScoreModel.enhance" in "official ScoreModel.enhance"
    validate_official_enhance_signature(inspect.signature(
        lambda self, y, sampler_type, predictor, corrector, N, corrector_steps, snr, timeit=False, **kwargs: None
    ))
    try:
        validate_official_enhance_signature(inspect.signature(
            lambda self, y, sampler_type, predictor, corrector, N, corrector_steps, snr, denoise=False, timeit=False, **kwargs: None
        ))
    except ValueError:
        pass
    else:
        return 1
    sampling_evidence = {
        "sha256": hashlib.sha256(HYPERPARAMS_RAW.encode()).hexdigest(),
        "raw": HYPERPARAMS_RAW,
    }
    sampling = reviewed_sampling_config(sampling_evidence)
    validate_wrapper_sampling(sampling)

    class FakeHParams(dict[str, Any]):
        def __getattr__(self, name: str) -> Any:
            return self[name]

    class FakeEnhancementConstructor:
        def __init__(self, hparams: FakeHParams):
            self.sampling = hparams.sampling

    fake_constructor = FakeEnhancementConstructor(FakeHParams({"sampling": sampling}))
    assert fake_constructor.sampling == sampling
    for tampered_sampling in (
        {key: value for key, value in sampling.items() if key != "snr"},
        {**sampling, "N": sampling["N"] + 1},
    ):
        try:
            validate_wrapper_sampling(tampered_sampling)
        except ValueError:
            pass
        else:
            return 1
    try:
        reviewed_sampling_config({"sha256": sampling_evidence["sha256"], "raw": HYPERPARAMS_RAW.replace("sampling:\n", "")})
    except ValueError:
        pass
    else:
        return 1
    assert serialize_noise_planes([1.0, 2.0], [3.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]
    assert serialize_noise_planes([1.0, 2.0], None) == [1.0, 2.0]

    class FakeTensor:
        def __init__(self, shape: tuple[int, ...], complex_value: bool = False):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = "torch.complex64" if complex_value else "torch.float32"
            self._count = math.prod(shape)
            self.complex_value = complex_value

        def detach(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def reshape(self, *_shape: int) -> "FakeTensor":
            return FakeTensor((self._count,), self.complex_value)

        def numel(self) -> int:
            return self._count

    class FakeTorch:
        Tensor = FakeTensor

        class _Finite:
            def all(self) -> bool:
                return True

        @staticmethod
        def is_complex(value: FakeTensor) -> bool:
            return value.complex_value

        @staticmethod
        def isfinite(_value: FakeTensor) -> "FakeTorch._Finite":
            return FakeTorch._Finite()

    class FakeScoreModel:
        def enhance(self, value: FakeTensor) -> FakeTensor:
            if value.ndim != 4 or not value.complex_value:
                raise AssertionError("raw waveform reached ScoreModel.enhance")
            return value

    fake_score_model = FakeScoreModel()

    class FakeEnhancer:
        def enhance_batch(self, waveform: FakeTensor) -> FakeTensor:
            if waveform.shape != (1, 5):
                raise AssertionError("wrapper did not receive waveform input")
            fake_score_model.enhance(FakeTensor((1, 1, 2, 3), complex_value=True))
            return FakeTensor((1, 5))

    fake_output, fake_calls = call_official_enhancer(
        FakeEnhancer(), fake_score_model, FakeTensor((1, 5)), FakeTorch
    )
    assert fake_calls == [{"shape": [1, 1, 2, 3], "dtype": "torch.complex64"}]
    assert validate_official_waveform_output(fake_output, 5, FakeTorch).numel() == 5

    class BadEnhancer:
        def enhance_batch(self, waveform: FakeTensor) -> FakeTensor:
            return fake_score_model.enhance(waveform)

    try:
        call_official_enhancer(BadEnhancer(), fake_score_model, FakeTensor((1, 5)), FakeTorch)
    except ValueError:
        pass
    else:
        return 1

    with tempfile.TemporaryDirectory(prefix="sgmse-enhancement-self-test-") as directory:
        root = Path(directory)
        packet = root / "packet"
        packet.mkdir()
        (packet / INPUT_NAME).write_bytes(struct.pack("<f", 0.25))
        (packet / REFERENCE_NAME).write_bytes(struct.pack("<f", 0.5))
        noise_raw = bytearray()
        noise_calls = []
        call_lines = []
        for index in range(NOISE_CALL_COUNT):
            kind = "prior" if index == 0 else "corrector" if index % 2 else "predictor"
            step = -1 if index == 0 else (index - 1) // 2
            corrector = index != 0 and index % 2 == 1
            segment = struct.pack("<f", float(index + 1))
            noise_calls.append({
                "index": index, "kind": kind, "step": step, "corrector": corrector,
                "count": 1, "offset": len(noise_raw), "bytes": 4,
                "sha256": digest_bytes(segment), "source_dtype": "torch.float32",
                "source_shape": [1], "serialized_layout": "flat",
            })
            call_lines.append(f"{index} {0 if index == 0 else 1} {step} {int(corrector)} 1 {len(noise_raw)}")
            noise_raw.extend(segment)
        (packet / NOISE_NAME).write_bytes(noise_raw)
        (packet / NOISE_CALLS_NAME).write_text("\n".join(call_lines) + "\n", encoding="ascii")
        (packet / RUN_LOG_NAME).write_text(
            "reference=vokra-sgmse-native-enhancement-reference-v1\n"
            "status=REFERENCE_COMPLETE_NO_UPLOAD\npublication=NO_UPLOAD\n",
            encoding="utf-8",
        )
        artifacts = {
            INPUT_NAME: {"path": INPUT_NAME, "dtype": "float32", "shape": [1], "count": 1, "bytes": 4, "sha256": sha256(packet / INPUT_NAME)},
            REFERENCE_NAME: {"path": REFERENCE_NAME, "dtype": "float32", "shape": [1], "count": 1, "bytes": 4, "sha256": sha256(packet / REFERENCE_NAME)},
            NOISE_NAME: {"path": NOISE_NAME, "dtype": "float32", "shape": [NOISE_CALL_COUNT], "count": NOISE_CALL_COUNT, "bytes": NOISE_CALL_COUNT * 4, "sha256": sha256(packet / NOISE_NAME)},
            NOISE_CALLS_NAME: {"path": NOISE_CALLS_NAME, "dtype": "ascii", "shape": [NOISE_CALL_COUNT], "count": NOISE_CALL_COUNT, "bytes": (packet / NOISE_CALLS_NAME).stat().st_size, "sha256": sha256(packet / NOISE_CALLS_NAME)},
        }
        manifest = {
            "format": PACKET_FORMAT, "status": PACKET_STATUS, "publication": "NO_UPLOAD",
            "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION,
            "checkpoint": {"filename": CHECKPOINT_NAME, "size": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "license_spdx": CHECKPOINT_LICENSE_SPDX},
            "source": {"path": "/source", "revision": SOURCE_REVISION, "clean": True, "repository": "https://github.com/sp-uhh/sgmse.git", "license_spdx": SOURCE_LICENSE_SPDX, "license_sha256": SOURCE_LICENSE_SHA256, "files": []},
            "speechbrain_source": {"path": "/speechbrain", "revision": SPEECHBRAIN_REVISION, "clean": True, "repository": "https://github.com/speechbrain/speechbrain.git", "license_spdx": SPEECHBRAIN_LICENSE_SPDX, "license_sha256": SPEECHBRAIN_LICENSE_SHA256, "files": [{"path": ENHANCEMENT_SOURCE_FILE, "sha256": "019e79bb489ba4c7f1ddd681e0cc007d7034386636ffd156128ec85058476995", "size": 11693, "markers": {marker: True for marker in ENHANCEMENT_SOURCE_MARKERS}}]},
            "licenses": {"algorithm": {"spdx": SOURCE_LICENSE_SPDX, "sha256": SOURCE_LICENSE_SHA256}, "speechbrain": {"spdx": SPEECHBRAIN_LICENSE_SPDX, "sha256": SPEECHBRAIN_LICENSE_SHA256}, "checkpoint": CHECKPOINT_LICENSE_SPDX},
            "ema_route": {"status": EMA_ROUTE_STATUS, "loadable": "score_model_ema", "checkpoint_filename": CHECKPOINT_NAME, "parameter_load": "strict_state_dict", "unsafe_pickle_fallback": False, "source_files": {"score_model": {"path": "speechbrain/integrations/models/sgmse_plus.py", "sha256": "b70ecde1d7326282b339348c739e91413c6dbac07ef98d34b540be07d8e70935", "size": 21777}, "parameter_transfer": {"path": "speechbrain/utils/parameter_transfer.py", "sha256": "0" * 64, "size": 1}}},
            "model": {"load": "torch.load(weights_only=True)+load_state_dict(strict=True)", "tensor_count": 647, "parameter_count": 65_590_822},
            "input": {"wav_filename": "ref-clip.wav", "wav_size": INPUT_WAV_SIZE, "wav_sha256": INPUT_WAV_SHA256, "sample_rate": SAMPLE_RATE, "channels": CHANNELS, "sample_width": SAMPLE_WIDTH, "pcm_filename": INPUT_NAME},
            "runtime": {"platform_system": "Linux", "platform_machine": "x86_64", "platform_node": "self-test", "cpu_model": "self-test", "nproc": 1, "torch_version": "self-test", "numpy_version": "self-test"},
            "artifacts": artifacts, "noise_calls": noise_calls,
            "noise_payload": {"filename": NOISE_NAME, "call_count": NOISE_CALL_COUNT, "dtype": "float32", "complete": True},
            "tolerance": {"metric": "waveform_max_abs_and_rmse", "max_abs": FP32_ATOL, "rmse": FP32_ATOL, "basis": "repository FP32_ATOL=0.01; preregistered before real run"},
            "identity": {"reference_tool": "sgmse_native_enhancement_parity.py", "official_call": OFFICIAL_CALL_IDENTITY, "noise_capture": "torch.randn_like+torch.randn during one official enhance call", "self_test": "sgmse_native_enhancement_parity.py --self-test"},
            "vokra": {"path": "/source", "commit": "self-test", "clean": True, "tool_sha256": "0" * 64, "uv_lock_sha256": "0" * 64},
            "run_log": {"path": RUN_LOG_NAME, "size": (packet / RUN_LOG_NAME).stat().st_size, "sha256": sha256(packet / RUN_LOG_NAME)},
        }
        (packet / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        verify_for_self_test = lambda path: verify_reference(
            path, allow_missing_source_for_self_test=True
        )
        assert verify_for_self_test(packet)["status"] == PACKET_STATUS
        missing_checkout = root / "missing-speechbrain-checkout"
        candidate = root / "candidate-missing-speechbrain-checkout"
        shutil.copytree(packet, candidate)
        candidate_manifest = json.loads((candidate / MANIFEST_NAME).read_text())
        candidate_manifest["runtime"]["platform_node"] = "vast-self-test"
        candidate_manifest["speechbrain_source"]["path"] = str(missing_checkout)
        (candidate / MANIFEST_NAME).write_text(json.dumps(candidate_manifest), encoding="utf-8")
        try:
            verify_reference(candidate)
        except ValueError:
            pass
        else:
            return 1
        for field in ("ema_route", "model"):
            candidate = root / f"missing-{field}"
            shutil.copytree(packet, candidate)
            candidate_manifest = json.loads((candidate / MANIFEST_NAME).read_text())
            del candidate_manifest[field]
            (candidate / MANIFEST_NAME).write_text(json.dumps(candidate_manifest), encoding="utf-8")
            try:
                verify_for_self_test(candidate)
            except ValueError:
                pass
            else:
                return 1
        for mutation in ("truncate", "nonfinite", "wrong-order", "wrong-count"):
            candidate = root / f"candidate-{mutation}"
            shutil.copytree(packet, candidate)
            if mutation == "truncate":
                (candidate / NOISE_NAME).write_bytes(noise_raw[:-1])
            elif mutation == "nonfinite":
                bad = bytearray(noise_raw)
                bad[0:4] = struct.pack("<f", float("nan"))
                (candidate / NOISE_NAME).write_bytes(bad)
            else:
                candidate_manifest = json.loads((candidate / MANIFEST_NAME).read_text())
                if mutation == "wrong-order":
                    candidate_manifest["noise_calls"][1]["kind"] = "predictor"
                else:
                    candidate_manifest["noise_calls"][1]["count"] = 2
                (candidate / MANIFEST_NAME).write_text(json.dumps(candidate_manifest), encoding="utf-8")
            try:
                verify_for_self_test(candidate)
            except (ValueError, struct.error):
                pass
            else:
                return 1
        duplicate_manifest = root / "duplicate-manifest"
        duplicate_manifest.write_text('{"format":"a","format":"b"}', encoding="utf-8")
        try:
            json.loads(duplicate_manifest.read_text(), object_pairs_hook=reject_duplicate_json)
        except ValueError:
            pass
        else:
            return 1
        duplicate_log = root / "duplicate-log"
        shutil.copytree(packet, duplicate_log)
        duplicate_log_text = (duplicate_log / RUN_LOG_NAME).read_text(encoding="utf-8")
        (duplicate_log / RUN_LOG_NAME).write_text(
            duplicate_log_text + "status=REFERENCE_COMPLETE_NO_UPLOAD\n",
            encoding="utf-8",
        )
        duplicate_log_manifest = json.loads((duplicate_log / MANIFEST_NAME).read_text())
        duplicate_log_manifest["run_log"] = {"path": RUN_LOG_NAME, "size": (duplicate_log / RUN_LOG_NAME).stat().st_size, "sha256": sha256(duplicate_log / RUN_LOG_NAME)}
        (duplicate_log / MANIFEST_NAME).write_text(json.dumps(duplicate_log_manifest), encoding="utf-8")
        try:
            verify_reference(duplicate_log)
        except ValueError:
            pass
        else:
            return 1
        try:
            compare_native(packet, packet, root)
        except ValueError:
            pass
        else:
            return 1
        for name in EXPECTED_PACKET_FILES:
            if name not in {entry.name for entry in packet.iterdir()}:
                return 1
        assert path_overlaps(packet, packet / "child")
    print("sgmse_native_enhancement_parity self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--generate-reference", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--speechbrain-source-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hyperparams", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--input-wav", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.source_dir, args.speechbrain_source_dir, args.checkpoint, args.hyperparams, args.inspection_manifest, args.input_wav, args.output_dir, args.reference_dir, args.native_dir, args.vokra_root)) or args.generate_reference or args.compare or args.verify_reference:
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if args.generate_reference:
        values = (args.source_dir, args.speechbrain_source_dir, args.checkpoint, args.hyperparams, args.inspection_manifest, args.input_wav, args.output_dir, args.vokra_root)
        if any(value is None for value in values) or args.compare or args.verify_reference:
            parser.error("--generate-reference requires its complete input set and cannot combine with --compare")
        try:
            manifest = _run_official_reference(*values)  # type: ignore[arg-type]
        except Exception as error:
            print(f"sgmse enhancement reference BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
        return 0
    if args.verify_reference:
        if (
            args.reference_dir is None
            or args.vokra_root is None
            or args.compare
            or args.generate_reference
            or any(value is not None for value in (args.source_dir, args.speechbrain_source_dir, args.checkpoint, args.hyperparams, args.inspection_manifest, args.input_wav, args.output_dir, args.native_dir))
        ):
            parser.error("--verify-reference requires --reference-dir and --vokra-root only")
        try:
            manifest = verify_reference(args.reference_dir, args.vokra_root)
        except Exception as error:
            print(f"sgmse enhancement verifier BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps({"status": manifest["status"], "reference_dir": str(args.reference_dir)}, sort_keys=True))
        return 0
    if args.compare:
        if (
            args.reference_dir is None
            or args.native_dir is None
            or args.vokra_root is None
            or any(value is not None for value in (args.source_dir, args.speechbrain_source_dir, args.checkpoint, args.hyperparams, args.inspection_manifest, args.input_wav, args.output_dir))
        ):
            parser.error("--compare requires --reference-dir, --native-dir, and --vokra-root only")
        try:
            result = compare_native(args.reference_dir, args.native_dir, args.vokra_root)
        except Exception as error:
            print(f"sgmse enhancement comparator BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "CPU_ENHANCEMENT_PARITY_PASS" else 1
    parser.error("choose exactly one of --self-test, --generate-reference, --verify-reference, or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
