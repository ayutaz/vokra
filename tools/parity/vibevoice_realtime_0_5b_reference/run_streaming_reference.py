#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_realtime_0_5b_reference python
"""Run the official VibeVoice Realtime streaming path under explicit gates.

This module is an independent *caller* of Microsoft's pinned ``generate``
implementation.  It intentionally contains no copy of the generation
algorithm.  The real-weight path is VAST-only and refuses to import torch or
load a checkpoint/preset until the source, checkpoint, tokenizer, Carter
preset, bounded input, and an externally supplied owner scope are all
hash-bound.  The owner scope is not shipped by Vokra: voice consent,
disclaimer/watermark preservation, and permission to generate audio must be
provided by the owner for the exact identities being run.

The hooks are observational.  They record outputs returned by official
methods/modules, the initial diffusion noise, raw sampled latents, the
official decode input after its scale/bias transform, connector and decoder
chunks, EOS values, and cache lengths, then restore every patched
attribute in ``finally``.  They do not alter official return values or
implement a replacement sampler.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


SOURCE_REVISION = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
CHECKPOINT_REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
CHECKPOINT_BYTES = 2_035_332_888
CHECKPOINT_SHA256 = "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514"
CONFIG_BYTES = 2_117
CONFIG_SHA256 = "caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57"
PRESET_RELATIVE_PATH = "demo/voices/streaming_model/en-Carter_man.pt"
PRESET_BYTES = 4_256_002
PRESET_GIT_BLOB_SHA1 = "1d795ef667e6641eecb8b22452bb853b089bfdbe"
PRESET_PAYLOAD_SHA256 = "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897"
TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-0.5B"
TOKENIZER_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TOKENIZER_FILES = {
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "tokenizer_config.json": (7_228, "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9"),
    "tokenizer.json": (7_031_645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
}
FORMAT = "vokra-vibevoice-realtime-streaming-reference-v1"
SCOPE_SCHEMA = "vokra-vibevoice-realtime-streaming-execution-v1"
COMPATIBILITY_ROUTE = "TRUSTED_RUN_REFERENCE_COMPATIBILITY_AND_OFFICIAL_DYNAMICCACHE_DDP_CACHE_DATA"
NO_UPLOAD = "NO_UPLOAD"
CUDA_ATOL = 5e-2  # provisional device-selection guard, not a waveform release bound
MAX_TEXT_BYTES = 16_384
MAX_NEW_TOKENS = 64
MAX_DDPM_STEPS = 20


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _bounded_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise RuntimeError("bounded input text is empty or exceeds the maximum size")
    return value


def _validate_noise_budget(draw_count: int, max_new_tokens: int) -> None:
    if type(draw_count) is not int or type(max_new_tokens) is not int or draw_count < 0 or max_new_tokens < 1 or draw_count > max_new_tokens:
        raise RuntimeError("diffusion noise draw count exceeds the bounded generation budget")


def _json_no_duplicates(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=pairs)


def _reject_symlink_ancestors(path: Path, label: str) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise RuntimeError(f"{label} path or ancestor is a symlink: {current}")
        if current == current.parent:
            break
        current = current.parent


def _regular_file(path: Path, label: str) -> Path:
    _reject_symlink_ancestors(path, label)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _regular_dir(path: Path, label: str) -> Path:
    _reject_symlink_ancestors(path, label)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a regular non-symlink directory: {path}")
    return path.resolve()


def _platform_guard() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("streaming reference is VAST/Linux x86_64 only")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("streaming reference requires VOKRA_PUBLISH_ON_VAST=1")


def _source_identity(source_root: Path) -> dict[str, str]:
    root = _regular_dir(source_root, "official source root")
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
        origin = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("official source must be a clean git checkout") from error
    if head != SOURCE_REVISION or status:
        raise RuntimeError(f"official source identity is not clean at {SOURCE_REVISION}")
    if origin not in {
        "https://github.com/microsoft/VibeVoice.git",
        "git@github.com:microsoft/VibeVoice.git",
        "ssh://git@github.com/microsoft/VibeVoice.git",
    }:
        raise RuntimeError(f"unexpected official source origin: {origin!r}")
    required = (
        root / "vibevoice/modular/modeling_vibevoice_streaming_inference.py",
        root / "vibevoice/processor/vibevoice_streaming_processor.py",
        root / "demo/realtime_model_inference_from_file.py",
    )
    if any(not item.is_file() for item in required):
        raise RuntimeError("official source checkout is incomplete")
    return {"repository": "microsoft/VibeVoice", "revision": head, "origin": "https://github.com/microsoft/VibeVoice.git"}


def _vokra_identity(root: Path) -> dict[str, str]:
    root = _regular_dir(root, "Vokra checkout")
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Vokra checkout must be a clean git tree") from error
    if status:
        raise RuntimeError("Vokra checkout is dirty; refusing an unbound reference code tree")
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise RuntimeError("Vokra checkout returned a non-canonical Git identity")
    return {"vokra_head": head, "vokra_tree_sha1": tree}


def _identity(path: Path, expected_bytes: int | None = None, expected_sha256: str | None = None) -> dict[str, Any]:
    path = _regular_file(path, "authenticated artifact")
    size = path.stat().st_size
    sha = _sha256_file(path)
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(f"artifact byte count {size} != {expected_bytes}: {path}")
    if expected_sha256 is not None and sha != expected_sha256:
        raise RuntimeError(f"artifact SHA-256 {sha} != {expected_sha256}: {path}")
    return {"path": str(path), "bytes": size, "sha256": sha}


def _preset_identity(source_root: Path, preset_path: Path) -> dict[str, Any]:
    preset = _regular_file(preset_path, "fixed Carter preset")
    if preset.name != Path(PRESET_RELATIVE_PATH).name or preset.stat().st_size != PRESET_BYTES:
        raise RuntimeError("fixed Carter preset filename/size mismatch")
    try:
        blob = subprocess.check_output(
            ["git", "-C", str(source_root), "hash-object", "--", str(preset)], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not authenticate Carter Git blob") from error
    payload_sha = _sha256_file(preset)
    if blob != PRESET_GIT_BLOB_SHA1 or payload_sha != PRESET_PAYLOAD_SHA256:
        raise RuntimeError("fixed Carter preset hash mismatch")
    return {
        "relative_path": PRESET_RELATIVE_PATH,
        "bytes": PRESET_BYTES,
        "git_blob_sha1": blob,
        "payload_sha256": payload_sha,
    }


def _scope_payload(scope: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(scope)
    payload.pop("scope_sha256", None)
    return payload


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"todo", "pending", "tbd", "unknown"}:
        raise RuntimeError(f"owner scope {label} is missing or placeholder")
    return value.strip()


def _validate_scope(
    scope_path: Path,
    identities: dict[str, Any],
    text_sha256: str,
    expected_scope_sha256: str,
    script_sha256: str,
    uv_lock_sha256: str,
    trusted_runner_sha256: str,
) -> dict[str, Any]:
    scope_path = _regular_file(scope_path, "owner execution scope")
    if len(expected_scope_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_scope_sha256):
        raise RuntimeError("--owner-scope-sha256 must be a lowercase SHA-256")
    actual_file_sha256 = _sha256_file(scope_path)
    if actual_file_sha256 != expected_scope_sha256:
        raise RuntimeError("owner scope file does not match the externally supplied SHA-256")
    try:
        scope = _json_no_duplicates(scope_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError("owner execution scope must be valid JSON") from error
    if not isinstance(scope, dict) or scope.get("schema") != SCOPE_SCHEMA:
        raise RuntimeError("owner scope schema mismatch")
    supplied_scope_sha = _nonempty_text(scope.get("scope_sha256"), "scope_sha256")
    actual_scope_sha = _sha256_bytes(_canonical_json(_scope_payload(scope)))
    if supplied_scope_sha != actual_scope_sha:
        raise RuntimeError("owner scope hash does not authenticate its contents")
    if scope.get("publication") != NO_UPLOAD:
        raise RuntimeError("owner execution scope must remain NO_UPLOAD")
    _nonempty_text(scope.get("owner"), "owner")
    approved_at = _nonempty_text(scope.get("approved_at_utc"), "approved_at_utc")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", approved_at):
        raise RuntimeError("owner scope approved_at_utc must be an RFC3339 UTC timestamp")
    try:
        datetime.datetime.fromisoformat(approved_at[:-1] + "+00:00")
    except ValueError as error:
        raise RuntimeError("owner scope approved_at_utc is not an RFC3339 timestamp") from error
    if scope.get("decision") not in {"APPROVE_COMMERCIAL_EXECUTION", "APPROVE_RESEARCH_ONLY_EXECUTION"}:
        raise RuntimeError("owner scope does not approve execution")
    source = scope.get("source")
    checkpoint = scope.get("checkpoint")
    preset = scope.get("preset")
    tokenizer = scope.get("tokenizer")
    voice = scope.get("voice")
    execution = scope.get("execution")
    dependencies = scope.get("dependencies")
    expected = {
        "source": {"repository": "microsoft/VibeVoice", "revision": SOURCE_REVISION},
        "checkpoint": {"revision": CHECKPOINT_REVISION, "sha256": CHECKPOINT_SHA256, "bytes": CHECKPOINT_BYTES},
        "config": {"sha256": CONFIG_SHA256, "bytes": CONFIG_BYTES},
        "preset": {"relative_path": PRESET_RELATIVE_PATH, "bytes": PRESET_BYTES, "git_blob_sha1": PRESET_GIT_BLOB_SHA1, "payload_sha256": PRESET_PAYLOAD_SHA256},
        "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION},
    }
    provided = {"source": source, "checkpoint": checkpoint, "config": scope.get("config"), "preset": preset, "tokenizer": tokenizer}
    for name, values in expected.items():
        actual = provided[name]
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in values.items()):
            raise RuntimeError(f"owner scope {name} identity mismatch")
    for name in ("checkpoint", "config", "preset"):
        bound = provided[name]
        measured = identities.get(name)
        if not isinstance(measured, dict):
            raise RuntimeError(f"measured {name} identity is missing")
        measured_hash = measured.get("payload_sha256") if name == "preset" else measured.get("sha256")
        bound_hash = bound.get("payload_sha256") if name == "preset" else bound.get("sha256")
        if bound.get("bytes") != measured.get("bytes") or bound_hash != measured_hash:
            raise RuntimeError(f"owner scope {name} does not match the measured artifact")
    if tokenizer.get("files") != {name: sha for name, (_bytes, sha) in TOKENIZER_FILES.items()}:
        raise RuntimeError("owner scope tokenizer files are not the fixed native four-file contract")
    if not isinstance(voice, dict) or voice.get("consent") != "PROVED":
        raise RuntimeError("owner scope must prove Carter voice consent for this exact run")
    _nonempty_text(voice.get("consent_evidence_reference"), "voice.consent_evidence_reference")
    evidence_sha = voice.get("consent_evidence_sha256")
    if not isinstance(evidence_sha, str) or len(evidence_sha) != 64 or any(character not in "0123456789abcdef" for character in evidence_sha):
        raise RuntimeError("owner scope must hash-bind voice consent evidence")
    if voice.get("disclaimer") != "PRESERVE_UPSTREAM" or voice.get("watermark") != "DEFERRED_NO_EMBEDDED_CLAIM":
        raise RuntimeError("owner scope must preserve disclaimer and explicitly retain deferred watermark status")
    if not isinstance(execution, dict) or execution.get("model_forward") != "APPROVED" or execution.get("audio_generation") != "APPROVED":
        raise RuntimeError("owner scope does not approve model forward and audio generation")
    if not isinstance(dependencies, dict):
        raise RuntimeError("owner scope dependencies disposition is missing")
    if dependencies.get("uv_lock_sha256") != uv_lock_sha256:
        raise RuntimeError("owner scope dependencies do not bind the measured uv.lock")
    if dependencies.get("license_audit") != "APPROVED_FOR_REFERENCE_EXECUTION":
        raise RuntimeError("owner scope lacks an approved license disposition")
    if dependencies.get("security_disposition") != "APPROVED_FOR_REFERENCE_EXECUTION":
        raise RuntimeError("owner scope lacks an approved security disposition")
    if dependencies.get("use") != "REFERENCE_ONLY_NO_RUNTIME_REUSE":
        raise RuntimeError("owner scope dependency use is not reference-only")
    _nonempty_text(dependencies.get("evidence_reference"), "dependencies.evidence_reference")
    dependency_evidence_sha = dependencies.get("evidence_sha256")
    if not isinstance(dependency_evidence_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", dependency_evidence_sha):
        raise RuntimeError("owner scope dependencies must hash-bind disposition evidence")
    max_new_tokens = execution.get("max_new_tokens")
    ddpm_steps = execution.get("ddpm_steps")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise RuntimeError(f"owner scope max_new_tokens must be 1..{MAX_NEW_TOKENS}")
    if type(ddpm_steps) is not int or ddpm_steps != MAX_DDPM_STEPS:
        raise RuntimeError(f"owner scope ddpm_steps must be exactly {MAX_DDPM_STEPS}")
    cfg_scale = execution.get("cfg_scale")
    if type(cfg_scale) not in (int, float) or isinstance(cfg_scale, bool) or not math.isfinite(float(cfg_scale)) or float(cfg_scale) < 0:
        raise RuntimeError("owner scope cfg_scale must be a finite non-negative number")
    if type(execution.get("benchmark_warmups")) is not int or execution["benchmark_warmups"] != 1:
        raise RuntimeError("owner scope benchmark_warmups must be exactly 1")
    if type(execution.get("benchmark_repeats")) is not int or execution["benchmark_repeats"] != 3:
        raise RuntimeError("owner scope benchmark_repeats must be exactly 3")
    input_contract = scope.get("input")
    if not isinstance(input_contract, dict):
        raise RuntimeError("owner scope input contract must be an object")
    if input_contract.get("text_sha256") != text_sha256:
        raise RuntimeError("owner scope text hash mismatch")
    if input_contract.get("sample_rate") != 24_000:
        raise RuntimeError("owner scope must bind the official 24 kHz output")
    runtime = scope.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("reference_script_sha256") != script_sha256 or runtime.get("uv_lock_sha256") != uv_lock_sha256 or runtime.get("trusted_runner_sha256") != trusted_runner_sha256:
        raise RuntimeError("owner scope runtime hash binding mismatch")
    if runtime.get("compatibility_route") != COMPATIBILITY_ROUTE:
        raise RuntimeError("owner scope compatibility route mismatch")
    if runtime.get("seed") != 1234 or runtime.get("noise_policy") != "CONTROLLED_CPU_TAPE" or runtime.get("device_selection_policy") != "CUDA_IF_FULL_TRACE_GUARD_AND_MEDIAN_FASTER":
        raise RuntimeError("owner scope noise/device policy mismatch")
    if runtime.get("vokra_head") != identities.get("vokra_head") or runtime.get("vokra_tree_sha1") != identities.get("vokra_tree_sha1"):
        raise RuntimeError("owner scope Vokra HEAD/tree binding mismatch")
    return scope


def _hash_tokenizer_files(tokenizer_dir: Path, scope: dict[str, Any]) -> dict[str, Any]:
    files = scope["tokenizer"]["files"]
    entries = {item.name for item in tokenizer_dir.iterdir()}
    if entries != set(TOKENIZER_FILES):
        raise RuntimeError("tokenizer directory must contain exactly the four authenticated assets")
    identities: dict[str, Any] = {}
    for relative, expected_sha in sorted(files.items()):
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"invalid tokenizer file path in owner scope: {relative!r}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError(f"invalid tokenizer hash in owner scope: {relative!r}")
        expected_bytes, fixed_sha = TOKENIZER_FILES[relative]
        if expected_sha != fixed_sha:
            raise RuntimeError(f"tokenizer hash drift for {relative}")
        identities[relative] = _identity(tokenizer_dir / relative, expected_bytes, expected_sha)
    return identities


def _prepare_output(path: Path) -> Path:
    path = path.absolute()
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise RuntimeError(f"output ancestor is a symlink: {current}")
        current = current.parent
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output directory must not exist (no-clobber): {path}")
    path.mkdir(parents=True)
    return path


def _detach_cpu(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return value


def _record_tensor(value: Any, output_dir: Path, name: str, torch: Any) -> dict[str, Any]:
    import numpy as np

    if not torch.is_tensor(value):
        raise RuntimeError(f"trace {name} is not a tensor")
    cpu = _detach_cpu(value, torch)
    if not bool(torch.isfinite(cpu).all().item()):
        raise RuntimeError(f"trace {name} contains non-finite values")
    safe = name.replace("/", "_").replace(".", "_")
    path = output_dir / f"{safe}.npy"
    array = cpu.numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha256_file(path), "finite": True}


class _NoiseTape:
    def __init__(self, torch: Any, replay: list[Any] | None = None) -> None:
        self.torch = torch
        self.values: list[Any] = [] if replay is None else replay
        self.replay = replay is not None
        self.index = 0
        self.matching = True

    def draw(self, original: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        value = original(*args, **kwargs)
        if self.replay:
            if self.index >= len(self.values):
                raise RuntimeError("CUDA requested more diffusion draws than CPU reference")
            expected = self.values[self.index]
            self.index += 1
            if tuple(value.shape) != tuple(expected.shape):
                raise RuntimeError("CPU/CUDA diffusion draw shape mismatch")
            actual_cpu = value.detach().to(device="cpu", dtype=self.torch.float32)
            if not self.torch.equal(actual_cpu, expected):
                self.matching = False
                raise RuntimeError("CPU/CUDA diffusion initial noise diverged")
            return expected.to(device=value.device, dtype=value.dtype)
        self.values.append(value.detach().to(device="cpu", dtype=self.torch.float32).contiguous())
        self.index += 1
        return value

    def finish(self) -> None:
        if self.replay and self.index != len(self.values):
            raise RuntimeError("CUDA consumed fewer diffusion draws than CPU reference")
        if not self.matching:
            raise RuntimeError("diffusion noise tape did not match")


class _Trace:
    def __init__(self, output_dir: Path, torch: Any, noise_tape: _NoiseTape) -> None:
        self.output_dir = output_dir
        self.torch = torch
        self.noise_tape = noise_tape
        self.records: list[dict[str, Any]] = []
        self.counters: dict[str, int] = {}

    def record(self, stage: str, value: Any) -> None:
        ordinal = self.counters.get(stage, 0)
        self.counters[stage] = ordinal + 1
        self.records.append({"stage": stage, "ordinal": ordinal, "tensor": _record_tensor(value, self.output_dir, f"trace_{stage}_{ordinal}", self.torch)})

    def record_output(self, stage: str, output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            self.record(stage + ".hidden", hidden)
        cache = getattr(output, "past_key_values", None)
        if cache is not None:
            layers = getattr(cache, "layers", None)
            if not isinstance(layers, list):
                raise RuntimeError(f"trace {stage}.cache has no official layer list")
            lengths = []
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if (
                    not self.torch.is_tensor(key)
                    or not self.torch.is_tensor(value)
                    or key.ndim != 4
                    or value.ndim != 4
                ):
                    raise RuntimeError(f"trace {stage}.cache has an invalid official cache layer")
                lengths.append(int(key.shape[2]))
            cache_stage = stage + ".cache"
            ordinal = self.counters.get(cache_stage, 0)
            self.counters[cache_stage] = ordinal + 1
            self.records.append({"stage": cache_stage, "ordinal": ordinal, "cache_layers": len(layers), "cache_lengths": lengths})


@contextlib.contextmanager
def _observational_hooks(model: Any, trace: _Trace, cache_roles: dict[int, str]):
    """Patch only official instance methods/modules and restore them always."""

    originals: list[tuple[Any, str, Any]] = []
    handles: list[Any] = []

    def patch(obj: Any, name: str, replacement: Any) -> None:
        originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    original_forward_lm = model.forward_lm
    original_forward_tts = model.forward_tts_lm
    original_sample = model.sample_speech_tokens
    original_decode = model.model.acoustic_tokenizer.decode
    def forward_lm(*args: Any, **kwargs: Any) -> Any:
        result = original_forward_lm(*args, **kwargs)
        trace.record_output("lm.positive", result)
        return result

    def forward_tts(*args: Any, **kwargs: Any) -> Any:
        result = original_forward_tts(*args, **kwargs)
        role = cache_roles.get(id(kwargs.get("past_key_values")))
        if role is None:
            raise RuntimeError("official TTS cache role became ambiguous; refusing mislabeled trace")
        trace.record_output(role, result)
        returned_cache = getattr(result, "past_key_values", None)
        if returned_cache is not None:
            cache_roles[id(returned_cache)] = role
        return result

    def sample_speech(*args: Any, **kwargs: Any) -> Any:
        result = original_sample(*args, **kwargs)
        trace.record("speech.sampled_latent", result)
        return result

    def decode(*args: Any, **kwargs: Any) -> Any:
        if args:
            trace.record("acoustic.decode_input_unscaled", args[0])
        elif "latents" in kwargs:
            trace.record("acoustic.decode_input_unscaled", kwargs["latents"])
        result = original_decode(*args, **kwargs)
        trace.record("acoustic.decoder_chunk", result)
        return result

    def hook(stage: str, capture_input: bool = False):
        def callback(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if capture_input and _inputs:
                trace.record(stage + ".input", _inputs[0])
            trace.record(stage, output)

        return callback

    try:
        patch(model, "forward_lm", forward_lm)
        patch(model, "forward_tts_lm", forward_tts)
        patch(model, "sample_speech_tokens", sample_speech)
        patch(model.model.acoustic_tokenizer, "decode", decode)
        for module, stage, capture_input in (
            (model.model.acoustic_connector, "acoustic.connector", True),
            (model.tts_eos_classifier, "tts.eos", False),
            (model.model.prediction_head, "diffusion.prediction", False),
        ):
            handles.append(module.register_forward_hook(hook(stage, capture_input)))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


@contextlib.contextmanager
def _controlled_noise(torch: Any, tape: _NoiseTape):
    original = torch.randn

    def recorded_randn(*args: Any, **kwargs: Any) -> Any:
        return tape.draw(original, args, kwargs)

    torch.randn = recorded_randn
    try:
        yield
    finally:
        torch.randn = original


def _migrate_legacy_cache(cache: Any, branch: str, torch: Any, DynamicCache: Any) -> Any:
    """Migrate legacy pickle lists through Transformers' public constructor.

    The pinned Microsoft helper only adds ``layers`` wrappers whose API is not
    complete for Transformers 5.10.4 (notably it lacks ``get_seq_length`` and
    ``keys``/``values``).  This route never invents a cache class or tensor
    layout: ``DynamicCache(ddp_cache_data=...)`` is the official constructor,
    and every source tensor is compared after construction.
    """

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if not isinstance(key_cache, (list, tuple)) or not isinstance(value_cache, (list, tuple)):
        raise RuntimeError(f"{branch} preset cache lacks legacy key_cache/value_cache lists")
    if len(key_cache) != len(value_cache) or not key_cache:
        raise RuntimeError(f"{branch} preset cache has no complete legacy layer pairs")
    before: list[tuple[Any, Any]] = []
    for index, (key, value) in enumerate(zip(key_cache, value_cache, strict=True)):
        if (
            not torch.is_tensor(key)
            or not torch.is_tensor(value)
            or key.ndim != 4
            or value.ndim != 4
            or key.shape != value.shape
            or key.dtype != value.dtype
        ):
            raise RuntimeError(f"{branch} legacy cache layer {index} has invalid shape or dtype")
        before.append((key.detach().clone(), value.detach().clone()))
    try:
        migrated = DynamicCache(ddp_cache_data=tuple(before))
    except Exception as error:
        raise RuntimeError(f"{branch} official DynamicCache ddp_cache_data migration failed") from error
    layers = getattr(migrated, "layers", None)
    if not isinstance(layers, list) or len(layers) != len(before):
        raise RuntimeError(f"{branch} official DynamicCache migration changed layer count")
    for index, (old_key, old_value) in enumerate(before):
        layer = layers[index]
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        required = ("update", "get_mask_sizes", "get_seq_length", "get_max_cache_shape")
        if any(not callable(getattr(layer, name, None)) for name in required):
            raise RuntimeError(f"{branch} official DynamicCache layer {index} lacks required API")
        if (
            not torch.is_tensor(key)
            or not torch.is_tensor(value)
            or key.shape != old_key.shape
            or value.shape != old_value.shape
            or key.dtype != old_key.dtype
            or value.dtype != old_value.dtype
            or not torch.equal(key, old_key)
            or not torch.equal(value, old_value)
        ):
            raise RuntimeError(f"{branch} official DynamicCache migration changed tensor values/shape/dtype")
        expected_length = int(old_key.shape[2])
        if int(migrated.get_seq_length(index)) != expected_length:
            raise RuntimeError(f"{branch} official DynamicCache migration changed sequence length")
        mask_sizes = migrated.get_mask_sizes(0, index)
        if tuple(mask_sizes) != (expected_length, 0):
            raise RuntimeError(f"{branch} official DynamicCache migration returned incompatible mask sizes")
    return migrated


def _load_preset(path: Path, device: str, torch: Any, BaseModelOutputWithPast: Any, DynamicCache: Any) -> dict[str, Any]:

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        loaded = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(loaded, dict) or set(loaded) != {"lm", "tts_lm", "neg_lm", "neg_tts_lm"}:
        raise RuntimeError("Carter preset does not contain exactly four official branches")
    for branch, output in loaded.items():
        cache = getattr(output, "past_key_values", None)
        output.past_key_values = _migrate_legacy_cache(cache, branch, torch, DynamicCache)
    return loaded


def _cast_preset_to_f32(loaded: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Cast authenticated BF16 preset tensors exactly to the F32 compute dtype."""

    for branch, output in loaded.items():
        hidden = output.last_hidden_state
        if not torch.is_tensor(hidden) or not bool(torch.isfinite(hidden).all().item()):
            raise RuntimeError(f"{branch} preset hidden state is not finite")
        hidden_f32 = hidden.to(dtype=torch.float32)
        if not torch.equal(hidden_f32.to(dtype=hidden.dtype), hidden):
            raise RuntimeError(f"{branch} BF16->F32 hidden cast failed lossless round-trip")
        output.last_hidden_state = hidden_f32
        cache = output.past_key_values
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, list):
            raise RuntimeError(f"{branch} preset cache has no official layer list after migration")
        for index, layer in enumerate(layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            for tensor in (key, value):
                if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
                    raise RuntimeError(f"{branch} preset cache tensor is not finite")
                cast = tensor.to(dtype=torch.float32)
                if not torch.equal(cast.to(dtype=tensor.dtype), tensor):
                    raise RuntimeError(f"{branch} BF16->F32 cast failed lossless round-trip at layer {index}")
            layer.keys = key.to(dtype=torch.float32)
            layer.values = value.to(dtype=torch.float32)
            # DynamicLayer stores these derived attributes during lazy
            # initialization; keep them synchronized after the safe cast.
            layer.dtype = layer.keys.dtype
            layer.device = layer.keys.device
            if layer.dtype != torch.float32 or layer.device != layer.keys.device:
                raise RuntimeError(f"{branch} official cache layer metadata is stale after F32 cast")
    return loaded


def _move_inputs(inputs: Any, device: str, torch: Any) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: _move_inputs(value, device, torch) for key, value in inputs.items()}
    return inputs


def _run_once(
    *, model: Any, processor: Any, text: str, preset_path: Path, output_dir: Path | None, device: str,
    max_new_tokens: int, ddpm_steps: int, cfg_scale: float, noise_tape: _NoiseTape, torch: Any,
    record_trace: bool,
) -> tuple[dict[str, Any], list[float]]:
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    preset = _load_preset(preset_path, device, torch, BaseModelOutputWithPast, DynamicCache)
    preset = _cast_preset_to_f32(preset, torch)
    inputs = processor.process_input_with_cached_prompt(text=text, cached_prompt=preset, padding=True, return_tensors="pt", return_attention_mask=True)
    inputs = _move_inputs(inputs, device, torch)
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)
    trace_dir = None if output_dir is None else output_dir / device
    if record_trace and trace_dir is None:
        raise RuntimeError("trace output directory is required when record_trace is enabled")
    if trace_dir is not None:
        trace_dir.mkdir()
    trace = _Trace(trace_dir, torch, noise_tape) if record_trace else None
    if trace is not None:
        # The official generate call consumes these authenticated prefilled
        # outputs directly; record all four branches without invoking an extra
        # forward or changing the official call sequence.
        for branch, stage in (
            ("lm", "lm.positive.prefill"),
            ("neg_lm", "lm.negative.prefill"),
            ("tts_lm", "tts.positive.prefill"),
            ("neg_tts_lm", "tts.negative.prefill"),
        ):
            trace.record_output(stage, preset[branch])
    cache_roles = {
        id(preset["tts_lm"].past_key_values): "tts.positive",
        id(preset["neg_tts_lm"].past_key_values): "tts.negative",
    }
    hook_scope = _observational_hooks(model, trace, cache_roles) if record_trace else contextlib.nullcontext()
    # The owner scope fixes this seed; reset immediately before each official
    # generate call so CPU/CUDA use the same source RNG state.
    torch.manual_seed(1234)
    with _controlled_noise(torch, noise_tape), hook_scope:
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                show_progress_bar=False,
                all_prefilled_outputs=preset,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    noise_tape.finish()
    _validate_noise_budget(len(noise_tape.values), max_new_tokens)
    speech = result.speech_outputs[0] if result.speech_outputs else None
    if speech is None:
        raise RuntimeError("official generate returned no speech output")
    pcm_record = _record_tensor(speech, trace_dir, "pcm", torch) if record_trace else None
    noise_hashes = [_sha256_bytes(item.detach().to(device="cpu", dtype=torch.float32).numpy().tobytes()) for item in noise_tape.values]
    sequence_cpu = result.sequences.detach().to(device="cpu").contiguous()
    sequence_hash = _sha256_bytes(sequence_cpu.numpy().tobytes())
    noise_records = []
    if record_trace:
        noise_records = [_record_tensor(value, trace_dir, f"diffusion_initial_noise_{index}", torch) for index, value in enumerate(noise_tape.values)]
    packet = {"device": device, "seconds": elapsed, "pcm": pcm_record, "traces": [] if trace is None else trace.records, "diffusion_initial_noise": noise_records, "noise_draws": len(noise_tape.values), "noise_hashes": noise_hashes, "noise_matching": noise_tape.matching, "sequence_shape": list(sequence_cpu.shape), "sequence_dtype": str(sequence_cpu.dtype), "sequence_sha256": sequence_hash}
    if trace_dir is not None:
        (trace_dir / "trace.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return packet, [elapsed]


def _compare_packets(cpu: dict[str, Any], cuda: dict[str, Any], torch: Any, output_dir: Path) -> dict[str, Any]:
    # Only same-shaped finite trace arrays are compared.  This is a diagnostic
    # and provisional device-selection guard; no release waveform bound is
    # inferred from it.
    import numpy as np

    diagnostics: list[dict[str, Any]] = []
    if not cpu["traces"] or not cuda["traces"]:
        return {"guard_atol": CUDA_ATOL, "finite_and_within_guard": False, "reason": "empty trace", "trace_diagnostics": diagnostics}
    if cpu["sequence_shape"] != cuda["sequence_shape"] or cpu["sequence_dtype"] != cuda["sequence_dtype"] or cpu["sequence_sha256"] != cuda["sequence_sha256"]:
        return {"guard_atol": CUDA_ATOL, "finite_and_within_guard": False, "reason": "sequence mismatch", "trace_diagnostics": diagnostics}
    if cpu["noise_draws"] != cuda["noise_draws"] or cpu["noise_hashes"] != cuda["noise_hashes"] or not cpu["noise_matching"] or not cuda["noise_matching"]:
        return {"guard_atol": CUDA_ATOL, "finite_and_within_guard": False, "reason": "noise tape mismatch", "trace_diagnostics": diagnostics}
    def index(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        tensors: dict[str, Any] = {}
        caches: dict[str, Any] = {}
        for record in records:
            key = record["stage"] + f"/{record['ordinal']}"
            target = caches if "cache_lengths" in record else tensors
            if key in target:
                raise RuntimeError(f"duplicate trace stage identity: {key}")
            target[key] = record.get("tensor", record)
        return tensors, caches

    cpu_arrays, cpu_cache = index(cpu["traces"])
    cuda_arrays, cuda_cache = index(cuda["traces"])
    if set(cpu_arrays) != set(cuda_arrays) or set(cpu_cache) != set(cuda_cache):
        return {"guard_atol": CUDA_ATOL, "finite_and_within_guard": False, "reason": "trace stage/ordinal mismatch", "trace_diagnostics": diagnostics}
    if any(cpu_cache[key]["cache_layers"] != cuda_cache[key]["cache_layers"] or cpu_cache[key]["cache_lengths"] != cuda_cache[key]["cache_lengths"] for key in cpu_cache):
        return {"guard_atol": CUDA_ATOL, "finite_and_within_guard": False, "reason": "cache length mismatch", "trace_diagnostics": diagnostics}
    for key in sorted(set(cpu_arrays) & set(cuda_arrays)):
        left = np.load(output_dir / "cpu" / cpu_arrays[key]["file"], allow_pickle=False)
        right = np.load(output_dir / "cuda" / cuda_arrays[key]["file"], allow_pickle=False)
        if left.shape != right.shape:
            diagnostics.append({"stage": key, "shape_match": False})
            continue
        diff = np.abs(left - right)
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        diagnostics.append({"stage": key, "shape_match": True, "finite": finite, "max_abs": float(diff.max()) if diff.size else 0.0})
    left_pcm = np.load(output_dir / "cpu" / cpu["pcm"]["file"], allow_pickle=False)
    right_pcm = np.load(output_dir / "cuda" / cuda["pcm"]["file"], allow_pickle=False)
    pcm_diff = np.abs(left_pcm - right_pcm) if left_pcm.shape == right_pcm.shape else None
    pcm_finite = pcm_diff is not None and bool(np.isfinite(left_pcm).all() and np.isfinite(right_pcm).all())
    max_abs = max([item["max_abs"] for item in diagnostics if item.get("shape_match") and item.get("finite")] + ([float(pcm_diff.max())] if pcm_finite else [math.inf]))
    all_trace_pass = len(diagnostics) == len(cpu_arrays) and all(item.get("shape_match") and item.get("finite") for item in diagnostics)
    return {"guard_atol": CUDA_ATOL, "trace_diagnostics": diagnostics, "pcm_shape_match": pcm_diff is not None, "pcm_finite": pcm_finite, "pcm_max_abs": None if pcm_diff is None else float(pcm_diff.max()), "global_max_abs": max_abs, "finite_and_within_guard": all_trace_pass and pcm_finite and math.isfinite(max_abs) and max_abs <= CUDA_ATOL}


def _benchmark(
    *, model: Any, processor: Any, text: str, preset_path: Path, device: str, warmups: int, repeats: int,
    max_new_tokens: int, ddpm_steps: int, cfg_scale: float, tape_values: list[Any], torch: Any,
) -> list[float]:
    for _ in range(warmups):
        torch.manual_seed(1234)
        warmup_tape = _NoiseTape(torch, replay=tape_values)
        _run_once(model=model, processor=processor, text=text, preset_path=preset_path, output_dir=None, device=device, max_new_tokens=max_new_tokens, ddpm_steps=ddpm_steps, cfg_scale=cfg_scale, noise_tape=warmup_tape, torch=torch, record_trace=False)
    samples: list[float] = []
    for _ in range(repeats):
        torch.manual_seed(1234)
        tape = _NoiseTape(torch, replay=tape_values)
        packet, _ = _run_once(
            model=model,
            processor=processor,
            text=text,
            preset_path=preset_path,
            output_dir=None,
            device=device,
            max_new_tokens=max_new_tokens,
            ddpm_steps=ddpm_steps,
            cfg_scale=cfg_scale,
            noise_tape=tape,
            torch=torch,
            record_trace=False,
        )
        if not packet["noise_matching"]:
            raise RuntimeError(f"{device} benchmark did not consume the authenticated noise tape")
        samples.append(packet["seconds"])
    return samples


def _run(args: argparse.Namespace) -> None:
    _platform_guard()
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    text = _bounded_text(text)
    text_sha = _sha256_bytes(text.encode("utf-8"))
    source_root = _regular_dir(args.source_root, "official source root")
    source = _source_identity(source_root)
    checkpoint = _identity(args.weights, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    config = _identity(args.config, CONFIG_BYTES, CONFIG_SHA256)
    preset = _preset_identity(source_root, args.preset)
    script_identity = _identity(Path(__file__))
    lock_identity = _identity(Path(__file__).with_name("uv.lock"))
    trusted_runner_identity = _identity(Path(__file__).with_name("run_reference.py"))
    vokra_identity = _vokra_identity(Path(__file__).resolve().parents[3])
    scope = _validate_scope(
        args.owner_scope,
        {"source": source, "checkpoint": checkpoint, "config": config, "preset": preset, **vokra_identity},
        text_sha,
        args.owner_scope_sha256,
        script_identity["sha256"],
        lock_identity["sha256"],
        trusted_runner_identity["sha256"],
    )
    tokenizer_dir = _regular_dir(args.tokenizer_dir, "fixed tokenizer directory")
    tokenizer = _hash_tokenizer_files(tokenizer_dir, scope)
    output_dir = _prepare_output(args.output)

    # Imports and model construction are intentionally below every gate above.
    sys.path.insert(0, str(source_root))
    import run_reference as trusted_reference

    trusted_reference._compatibility_check(source_root)
    torch, model, checkpoint_binding = trusted_reference._load_model(source_root, args.config, args.weights)
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor, VibeVoiceTokenizerProcessor

    tokenizer_obj = VibeVoiceTextTokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)
    processor = VibeVoiceStreamingProcessor(tokenizer=tokenizer_obj, audio_processor=VibeVoiceTokenizerProcessor())
    model.eval().to(dtype=torch.float32)

    torch.manual_seed(1234)
    cpu_tape = _NoiseTape(torch)
    cpu_packet, _ = _run_once(model=model, processor=processor, text=text, preset_path=args.preset, output_dir=output_dir, device="cpu", max_new_tokens=scope["execution"]["max_new_tokens"], ddpm_steps=scope["execution"]["ddpm_steps"], cfg_scale=float(scope["execution"]["cfg_scale"]), noise_tape=cpu_tape, torch=torch, record_trace=True)
    packets = {"cpu": cpu_packet}
    cpu_samples = _benchmark(model=model, processor=processor, text=text, preset_path=args.preset, device="cpu", warmups=scope["execution"]["benchmark_warmups"], repeats=scope["execution"]["benchmark_repeats"], max_new_tokens=scope["execution"]["max_new_tokens"], ddpm_steps=scope["execution"]["ddpm_steps"], cfg_scale=float(scope["execution"]["cfg_scale"]), tape_values=cpu_tape.values, torch=torch)
    packets["cpu_timing"] = {"warmups": scope["execution"]["benchmark_warmups"], "samples": cpu_samples, "median_seconds": statistics.median(cpu_samples), "noise_replay": "CONTROLLED_CPU_TAPE"}
    if torch.cuda.is_available() and not args.cpu_only:
        model.to("cuda")
        torch.manual_seed(1234)
        cuda_tape = _NoiseTape(torch, replay=cpu_tape.values)
        cuda_packet, _ = _run_once(model=model, processor=processor, text=text, preset_path=args.preset, output_dir=output_dir, device="cuda", max_new_tokens=scope["execution"]["max_new_tokens"], ddpm_steps=scope["execution"]["ddpm_steps"], cfg_scale=float(scope["execution"]["cfg_scale"]), noise_tape=cuda_tape, torch=torch, record_trace=True)
        packets["cuda"] = cuda_packet
        comparison = _compare_packets(cpu_packet, cuda_packet, torch, output_dir)
        packets["cuda_comparison"] = comparison
        torch.manual_seed(1234)
        cuda_samples = _benchmark(model=model, processor=processor, text=text, preset_path=args.preset, device="cuda", warmups=scope["execution"]["benchmark_warmups"], repeats=scope["execution"]["benchmark_repeats"], max_new_tokens=scope["execution"]["max_new_tokens"], ddpm_steps=scope["execution"]["ddpm_steps"], cfg_scale=float(scope["execution"]["cfg_scale"]), tape_values=cpu_tape.values, torch=torch)
        packets["cuda_timing"] = {"warmups": scope["execution"]["benchmark_warmups"], "samples": cuda_samples, "median_seconds": statistics.median(cuda_samples), "noise_replay": "CONTROLLED_CPU_TAPE"}
        packets["selected_device"] = "cuda" if comparison["finite_and_within_guard"] and packets["cuda_timing"]["median_seconds"] < packets["cpu_timing"]["median_seconds"] else "cpu"
    else:
        packets["selected_device"] = "cpu"
        packets["cuda_reason"] = "CUDA unavailable or --cpu-only"
    packet = {"format": FORMAT, "status": "REFERENCE_RUN_OPEN_NOT_RUST_PARITY", "publication": NO_UPLOAD, "source": source, "checkpoint": checkpoint, "checkpoint_binding": checkpoint_binding, "config": config, "preset": preset, "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION, "files": tokenizer}, "owner_scope_sha256": scope["scope_sha256"], "owner_scope_file_sha256": args.owner_scope_sha256, "runtime": {**vokra_identity, "reference_script": script_identity, "uv_lock": lock_identity, "trusted_runner": trusted_runner_identity}, "input": {"text_sha256": text_sha, "bytes": len(text.encode("utf-8")), "sample_rate": 24_000}, "official_path": {"source_module": "vibevoice.modular.modeling_vibevoice_streaming_inference", "entrypoint": "VibeVoiceStreamingForConditionalGenerationInference.generate", "hooks_observational": True, "noise_controlled_replay": True, "preset_cast": "BF16_TO_F32_AFTER_EXACT_ROUNDTRIP"}, "packets": packets}
    (output_dir / "reference.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(packet, indent=2, sort_keys=True))


def _self_test() -> None:
    assert _sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    payload = {"schema": SCOPE_SCHEMA, "owner": "owner", "execution": {"max_new_tokens": 1}}
    encoded = _canonical_json(payload)
    assert _sha256_bytes(encoded) == _sha256_bytes(_canonical_json(payload))
    assert MAX_NEW_TOKENS == 64 and MAX_DDPM_STEPS == 20
    try:
        _json_no_duplicates('{"a":1,"a":2}')
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate JSON key was accepted")
    assert _bounded_text("hello") == "hello"
    try:
        _bounded_text(" ")
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty input was accepted")
    _validate_noise_budget(1, 1)
    try:
        _validate_noise_budget(2, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("noise budget overflow was accepted")

    with tempfile.TemporaryDirectory(dir=str(Path(tempfile.gettempdir()).resolve()), prefix="vokra-streaming-scope-") as temporary:
        root = Path(temporary)
        script_sha, lock_sha, helper_sha = "1" * 64, "2" * 64, "3" * 64
        identities = {
            "checkpoint": {"bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256},
            "config": {"bytes": CONFIG_BYTES, "sha256": CONFIG_SHA256},
            "preset": {"bytes": PRESET_BYTES, "payload_sha256": PRESET_PAYLOAD_SHA256},
            "vokra_head": "a" * 40,
            "vokra_tree_sha1": "b" * 40,
        }
        # This scope is an ephemeral validator fixture only; it is never an
        # owner approval record and must not be reused for a real run.
        scope = {
            "schema": SCOPE_SCHEMA,
            "owner": "model-free-test-owner",
            "approved_at_utc": "2026-09-30T00:00:00Z",
            "decision": "APPROVE_COMMERCIAL_EXECUTION",
            "publication": NO_UPLOAD,
            "source": {"repository": "microsoft/VibeVoice", "revision": SOURCE_REVISION},
            "checkpoint": {"revision": CHECKPOINT_REVISION, "sha256": CHECKPOINT_SHA256, "bytes": CHECKPOINT_BYTES},
            "config": {"sha256": CONFIG_SHA256, "bytes": CONFIG_BYTES},
            "preset": {"relative_path": PRESET_RELATIVE_PATH, "bytes": PRESET_BYTES, "git_blob_sha1": PRESET_GIT_BLOB_SHA1, "payload_sha256": PRESET_PAYLOAD_SHA256},
            "tokenizer": {"repository": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION, "files": {name: sha for name, (_bytes, sha) in TOKENIZER_FILES.items()}},
            "voice": {"consent": "PROVED", "consent_evidence_reference": "owner-record", "consent_evidence_sha256": "a" * 64, "disclaimer": "PRESERVE_UPSTREAM", "watermark": "DEFERRED_NO_EMBEDDED_CLAIM"},
            "execution": {"model_forward": "APPROVED", "audio_generation": "APPROVED", "max_new_tokens": 1, "ddpm_steps": 20, "cfg_scale": 3.0, "benchmark_warmups": 1, "benchmark_repeats": 3},
            "dependencies": {"uv_lock_sha256": lock_sha, "license_audit": "APPROVED_FOR_REFERENCE_EXECUTION", "security_disposition": "APPROVED_FOR_REFERENCE_EXECUTION", "use": "REFERENCE_ONLY_NO_RUNTIME_REUSE", "evidence_reference": "synthetic-model-free-test", "evidence_sha256": "b" * 64},
            "input": {"text_sha256": _sha256_bytes(b"hello"), "sample_rate": 24_000},
            "runtime": {"reference_script_sha256": script_sha, "uv_lock_sha256": lock_sha, "trusted_runner_sha256": helper_sha, "compatibility_route": COMPATIBILITY_ROUTE, "seed": 1234, "noise_policy": "CONTROLLED_CPU_TAPE", "device_selection_policy": "CUDA_IF_FULL_TRACE_GUARD_AND_MEDIAN_FASTER", "vokra_head": "a" * 40, "vokra_tree_sha1": "b" * 40},
        }
        scope["scope_sha256"] = _sha256_bytes(_canonical_json(scope))
        scope_path = root / "scope.json"
        raw = _canonical_json(scope)
        scope_path.write_bytes(raw)
        valid_args = (identities, _sha256_bytes(b"hello"), _sha256_bytes(raw), script_sha, lock_sha, helper_sha)
        _validate_scope(scope_path, *valid_args)

        def expect_scope_reject(mutator: Callable[[dict[str, Any]], None]) -> None:
            changed = copy.deepcopy(scope)
            mutator(changed)
            changed["scope_sha256"] = _sha256_bytes(_canonical_json(_scope_payload(changed)))
            changed_raw = _canonical_json(changed)
            scope_path.write_bytes(changed_raw)
            try:
                _validate_scope(scope_path, identities, _sha256_bytes(b"hello"), _sha256_bytes(changed_raw), script_sha, lock_sha, helper_sha)
            except RuntimeError:
                return
            raise AssertionError("tampered owner scope was accepted")

        expect_scope_reject(lambda value: value["config"].update(sha256="0" * 64))
        expect_scope_reject(lambda value: value["runtime"].update(reference_script_sha256="0" * 64))
        expect_scope_reject(lambda value: value["dependencies"].pop("license_audit"))
        expect_scope_reject(lambda value: value["dependencies"].update(security_disposition="PENDING"))
        expect_scope_reject(lambda value: value["dependencies"].update(evidence_sha256="B" * 64))
        expect_scope_reject(lambda value: value["runtime"].update(seed=42))
        expect_scope_reject(lambda value: value["execution"].update(cfg_scale=float("nan")))
        expect_scope_reject(lambda value: value["execution"].update(cfg_scale=True))
        expect_scope_reject(lambda value: value["execution"].update(ddpm_steps=19))
        expect_scope_reject(lambda value: value.update(approved_at_utc="2026-09-30 00:00:00Z"))

    class FakeHandle:
        def __init__(self, owner: list[Any], marker: Any): self.owner, self.marker = owner, marker
        def remove(self):
            if self.marker in self.owner:
                self.owner.remove(self.marker)

    class FakeModule:
        def __init__(self): self.hooks = []
        def register_forward_hook(self, callback): self.hooks.append(callback); return FakeHandle(self.hooks, callback)

    class FakeModel:
        def __init__(self):
            self.model = type("M", (), {"acoustic_tokenizer": type("A", (), {"decode": lambda *_a, **_k: object()})(), "acoustic_connector": FakeModule(), "prediction_head": FakeModule()})()
            self.tts_eos_classifier = FakeModule()
        def forward_lm(self, **_kwargs): return type("O", (), {})()
        def forward_tts_lm(self, **_kwargs): return type("O", (), {})()
        def sample_speech_tokens(self, *_a, **_k): return object()

    model = FakeModel()
    original = model.forward_lm
    try:
        with _observational_hooks(model, _Trace(Path("/tmp"), type("T", (), {})(), _NoiseTape(type("T", (), {})())), {}):
            raise RuntimeError("expected hook failure")
    except RuntimeError as error:
        assert str(error) == "expected hook failure"
    assert model.forward_lm.__func__ is original.__func__
    assert not model.model.acoustic_connector.hooks

    failed_install = FakeModel()
    failed_original = failed_install.forward_lm
    def fail_register(_callback):
        raise RuntimeError("install failure")
    failed_install.tts_eos_classifier.register_forward_hook = fail_register
    try:
        with _observational_hooks(failed_install, _Trace(Path("/tmp"), type("T", (), {})(), _NoiseTape(type("T", (), {})())), {}):
            raise AssertionError("hook installation unexpectedly succeeded")
    except RuntimeError as error:
        assert str(error) == "install failure"
    assert failed_install.forward_lm.__func__ is failed_original.__func__
    assert not failed_install.model.acoustic_connector.hooks
    try:
        _validate_scope(Path("/definitely/missing-owner-scope.json"), {}, "0" * 64, "0" * 64, "0" * 64, "0" * 64, "0" * 64)
    except RuntimeError as error:
        assert "regular" in str(error) or "scope" in str(error)
    else:
        raise AssertionError("missing owner scope did not fail closed")
    print("vibevoice streaming reference self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--owner-scope", type=Path)
    parser.add_argument("--owner-scope-sha256")
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    required = (args.source_root, args.weights, args.config, args.preset, args.tokenizer_dir, args.owner_scope, args.output, args.owner_scope_sha256)
    if any(value is None for value in required) or (args.text is None) == (args.text_file is None):
        parser.error("real run requires all artifact/scope paths, external scope SHA, and exactly one of --text/--text-file")
    _run(args)


if __name__ == "__main__":
    main()
