#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Run the pinned official Irodori pipeline for a tiny deterministic packet.

This is an adapter, not a Python mirror.  The pinned source/lock is recorded as
inspection evidence, but its DACVAE -> audiotools/librosa -> soxr/soundfile
closure is forbidden here.  The stdlib-only dependency gate therefore blocks
before source import, dependency synchronization, model access, or PCM; no
intermediate latents or fabricated parity evidence can be emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from irodori_inspect import (
    DACVAE_REVISION,
    SOURCE_LOCK_SHA256,
    SOURCE_PYPROJECT_SHA256,
    dependency_gate,
    inspect_dependency_lock,
)

UPSTREAM_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
MODEL_REPOSITORY = "Aratako/Irodori-TTS-500M-v3"
MODEL_REVISION = "236c1e56591279fc24e3c1bf6609fc06e48dde28"
CODEC_REPOSITORY = "Aratako/Semantic-DACVAE-Japanese-32dim"
CODEC_REVISION = "47376ee24834d7a05a48ebabfe3cde29b3c5e214"
TOKENIZER_REPOSITORY = "llm-jp/llm-jp-3-150m"
TOKENIZER_REVISION = "b112feef602fff752e4dac4c30af6a2c2fa41c7a"
MODEL_BYTES = 2_048_269_748
MODEL_SHA256 = "c4b8e7e982697664f829b7fb6bea307a25bd7ee013ad0d6114efc3e326acbd54"
FORMAT = "vokra-irodori-500m-v3-official-reference-v1"
INSPECTION_KEYS = frozenset({
    "model_identity", "source_identity", "codec_identity", "tokenizer_identity",
    "model_composite_roles", "historical_public_identity", "status", "evidence_stage",
    "inspection_status", "runtime_status", "cpu_status", "metal_status", "parity_status",
    "publication", "error", "blockers", "model", "model_readme", "model_safetensors",
    "public_gguf", "codec", "tokenizer", "source", "notes",
})
MODEL_PATHS = frozenset({
    ".gitattributes", "EMOJI_ANNOTATIONS.md", "README.md", "model.safetensors",
    "samples/clone_gen1.wav", "samples/clone_gen2.wav", "samples/clone_ref1.wav",
    "samples/clone_ref2.wav", "samples/emoji_sample1.wav", "samples/emoji_sample2.wav",
    "samples/emoji_sample3.wav", "samples/standard_sample1.wav", "samples/standard_sample2.wav",
})
CODEC_PATHS = frozenset({".gitattributes", "README.md", "weights.pth"})
TOKENIZER_PATHS = frozenset({
    ".gitattributes", "README.md", "config.json", "generation_config.json",
    "model.safetensors", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular local file: {path}")
    return path


def require_tokenizer(path: Path) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"tokenizer must be a regular local directory: {path}")
    required = {
        ".gitattributes", "README.md", "config.json", "generation_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    }
    members = {item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file() and not item.is_symlink()}
    if members != required:
        raise RuntimeError(
            "tokenizer must contain exactly the seven authenticated small assets "
            f"(missing={sorted(required - members)}, extra={sorted(members - required)})"
        )
    return path


def strict_json(data: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=pairs)


def write_tensor(output: Path, name: str, tensor: Any, *, dtype: str) -> dict[str, Any]:
    """Write an official tensor as a portable little-endian packet member."""
    torch = __import__("torch")
    if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0 or tensor.ndim == 0:
        raise RuntimeError(f"{name}: tensor must be nonempty")
    expected_suffix = {"f32": ".f32le", "u32": ".u32le", "u8": ".u8"}.get(dtype)
    if expected_suffix is None:
        raise ValueError(f"unsupported packet dtype: {dtype}")
    if Path(name).name != name or ".." in Path(name).parts or not name.endswith(expected_suffix):
        raise RuntimeError(f"{name}: unsafe packet member name")
    value = tensor.detach().to(device="cpu").contiguous()
    if any(not isinstance(dim, int) or dim <= 0 or dim > (1 << 31) for dim in value.shape):
        raise RuntimeError(f"{name}: shape must have bounded positive dimensions")
    if value.numel() > (1 << 31):
        raise RuntimeError(f"{name}: tensor is too large for an evidence packet")
    if dtype == "f32":
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"{name}: float tensor must be finite")
        raw = value.to(dtype=torch.float32).numpy().astype("<f4", copy=False).tobytes()
    elif dtype == "u32":
        if torch.is_floating_point(value):
            if not bool(torch.isfinite(value).all().item()) or float(value.min().item()) < 0 or float(value.max().item()) > 0xFFFFFFFF:
                raise RuntimeError(f"{name}: u32 value outside [0, 2^32)")
        integer = value.to(dtype=torch.int64)
        if torch.is_floating_point(value) and not bool(torch.equal(value, integer)):
            raise RuntimeError(f"{name}: u32 values must be integral")
        if int(integer.min().item()) < 0 or int(integer.max().item()) > 0xFFFFFFFF:
            raise RuntimeError(f"{name}: u32 value outside [0, 2^32)")
        raw = integer.numpy().astype("<u4", copy=False).tobytes()
    elif dtype == "u8":
        if torch.is_floating_point(value):
            if not bool(torch.isfinite(value).all().item()) or float(value.min().item()) < 0 or float(value.max().item()) > 0xFF:
                raise RuntimeError(f"{name}: u8 value outside [0, 255]")
        integer = value.to(dtype=torch.int64)
        if torch.is_floating_point(value) and not bool(torch.equal(value, integer)):
            raise RuntimeError(f"{name}: u8 values must be integral")
        if int(integer.min().item()) < 0 or int(integer.max().item()) > 0xFF:
            raise RuntimeError(f"{name}: u8 value outside [0, 255]")
        raw = integer.to(dtype=torch.uint8).numpy().tobytes()
    else:
        raise ValueError(f"unsupported packet dtype: {dtype}")
    width = {"f32": 4, "u32": 4, "u8": 1}[dtype]
    if len(raw) != value.numel() * width:
        raise RuntimeError(f"{name}: packet byte count does not match tensor shape")
    path = output / name
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{name}: duplicate/stale packet member")
    output.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": name, "dtype": dtype, "shape": list(value.shape),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require_source_environment(source: Path) -> None:
    """Inspect the fixed source lock, but never resolve or synchronize it."""
    for name in ("pyproject.toml", "uv.lock"):
        require_file(source / name, f"official source {name}")
    if hashlib.sha256((source / "pyproject.toml").read_bytes()).hexdigest() != SOURCE_PYPROJECT_SHA256:
        raise RuntimeError("official source pyproject.toml SHA-256 mismatch")
    inspect_dependency_lock(source / "uv.lock")


def install_tokenizer_link(checkpoint: Path, tokenizer: Path) -> tuple[Path, bool]:
    """Expose the authenticated local tokenizer through the source's resolver."""
    destination = checkpoint.parent / "tokenizer"
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or destination.resolve() != tokenizer.resolve():
            raise RuntimeError(f"checkpoint tokenizer path is not the authenticated snapshot: {destination}")
        return destination, False
    destination.symlink_to(tokenizer, target_is_directory=True)
    return destination, True


def verify_inspection_manifest(path: Path, checkpoint: Path, codec: Path, tokenizer: Path) -> None:
    """Cross-check local inputs against the authenticated inspection packet."""
    data = strict_json(path.read_text(encoding="utf-8"))
    if set(data) != INSPECTION_KEYS:
        raise RuntimeError("inspection manifest schema is not the exact authenticated schema")
    if data.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise RuntimeError("inspection manifest is not authenticated complete evidence")
    if any(data.get(key) != expected for key, expected in {
        "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "publication": "NO_UPLOAD",
        "runtime_status": "PARTIAL_RUNTIME_BLOCKED", "cpu_status": "UNSUPPORTED_FULL_TTS",
        "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN",
    }.items()):
        raise RuntimeError("inspection manifest does not retain fail-closed runtime/public status")
    model = data.get("model_identity", {})
    if (model.get("repository"), model.get("revision"), model.get("bytes"), model.get("sha256")) != (
        MODEL_REPOSITORY, MODEL_REVISION, MODEL_BYTES, MODEL_SHA256
    ) or checkpoint.stat().st_size != MODEL_BYTES or sha256(checkpoint) != MODEL_SHA256:
        raise RuntimeError("checkpoint does not match the authenticated model identity")
    model_files = data.get("model", {}).get("files")
    if not isinstance(model_files, list) or len(model_files) != len(MODEL_PATHS):
        raise RuntimeError("inspection packet model file cardinality mismatch")
    model_rows: dict[str, dict[str, Any]] = {}
    for row in model_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in model_rows:
            raise RuntimeError("inspection packet model file rows are malformed or duplicated")
        model_rows[row["path"]] = row
    if set(model_rows) != MODEL_PATHS:
        raise RuntimeError("inspection packet model file set mismatch")
    model_row = model_rows.get("model.safetensors")
    if not isinstance(model_row, dict) or model_row.get("bytes") != MODEL_BYTES or model_row.get("sha256") != MODEL_SHA256:
        raise RuntimeError("inspection packet does not authenticate the model payload")
    codec_identity = data.get("codec_identity", {})
    if (codec_identity.get("repository"), codec_identity.get("revision")) != (CODEC_REPOSITORY, CODEC_REVISION):
        raise RuntimeError("codec does not match the authenticated codec identity")
    codec_files = data.get("codec", {}).get("files")
    if not isinstance(codec_files, list) or len(codec_files) != len(CODEC_PATHS):
        raise RuntimeError("inspection packet codec file cardinality mismatch")
    codec_rows: dict[str, dict[str, Any]] = {}
    for row in codec_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in codec_rows:
            raise RuntimeError("inspection packet codec file rows are malformed or duplicated")
        codec_rows[row["path"]] = row
    if set(codec_rows) != CODEC_PATHS:
        raise RuntimeError("inspection packet codec file set mismatch")
    codec_row = codec_rows.get("weights.pth")
    if not isinstance(codec_row, dict) or codec.stat().st_size != codec_row.get("bytes") or sha256(codec) != codec_row.get("sha256"):
        raise RuntimeError("codec payload does not match authenticated inspection evidence")
    tokenizer_identity = data.get("tokenizer_identity", {})
    tokenizer_evidence = data.get("tokenizer", {})
    if (tokenizer_identity.get("repository"), tokenizer_identity.get("revision")) != (
        TOKENIZER_REPOSITORY, TOKENIZER_REVISION
    ) or tokenizer_evidence.get("tokenizer_status") != "AUTHENTICATED_SMALL_ASSETS":
        raise RuntimeError("tokenizer evidence is not the selected authenticated snapshot")
    require_tokenizer(tokenizer)
    token_files = tokenizer_evidence.get("files")
    if not isinstance(token_files, list) or len(token_files) != len(TOKENIZER_PATHS):
        raise RuntimeError("inspection packet tokenizer file cardinality mismatch")
    token_rows: dict[str, dict[str, Any]] = {}
    for row in token_files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row["path"] in token_rows:
            raise RuntimeError("inspection packet tokenizer file rows are malformed or duplicated")
        token_rows[row["path"]] = row
    if set(token_rows) != TOKENIZER_PATHS:
        raise RuntimeError("inspection packet tokenizer file set mismatch")
    for name in sorted(token_rows):
        if name == "model.safetensors":
            continue
        path_value = tokenizer / name
        if not path_value.is_file() or path_value.stat().st_size != token_rows[name].get("bytes") or sha256(path_value) != token_rows[name].get("sha256"):
            raise RuntimeError(f"tokenizer asset does not match authenticated inspection evidence: {name}")
    source_identity = data.get("source_identity", {})
    if (source_identity.get("repository"), source_identity.get("revision")) != (
        "https://github.com/Aratako/Irodori-TTS.git", UPSTREAM_REVISION
    ):
        raise RuntimeError("inspection source identity does not match the fixed origin/revision")
    if (data.get("source", {}).get("repository"), data.get("source", {}).get("revision")) != (
        "https://github.com/Aratako/Irodori-TTS.git", UPSTREAM_REVISION
    ):
        raise RuntimeError("inspection source revision does not match the fixed source")
    if data.get("source", {}).get("reference_environment", {}).get("status") != "AUTHENTICATED_SOURCE_LOCK":
        raise RuntimeError("inspection packet does not authenticate the fixed source uv.lock")
    environment = data["source"]["reference_environment"]
    if environment.get("lock_sha256") != SOURCE_LOCK_SHA256 or environment.get("pyproject_sha256") != SOURCE_PYPROJECT_SHA256:
        raise RuntimeError("inspection packet source digest binding mismatch")
    if environment.get("dependency_closure_status") != "AUTHENTICATED_SOURCE_LOCK_FORBIDDEN_CLOSURE":
        raise RuntimeError("inspection packet does not retain the forbidden dependency closure")


def tensor_capture(captures: dict[str, Any], output: Path, name: str, value: Any, dtype: str) -> None:
    captures.setdefault("artifacts", []).append(write_tensor(output, name, value, dtype=dtype))


def validate_complete_packet(captures: dict[str, Any], output: Path, args: argparse.Namespace) -> None:
    required = {
        "text_input_ids.u32le", "text_attention_mask.u8", "predicted_log_frames.f32le",
        "initial_noise.f32le", "official_u_schedule.f32le", "sampled_patched_latent.f32le",
        "official_pcm.f32le",
    }
    artifacts = captures.get("artifacts", [])
    names = [row.get("path") for row in artifacts if isinstance(row, dict)]
    if set(names) != required or len(names) != len(required):
        raise RuntimeError(f"official packet artifact set is incomplete: {sorted(set(names))}")
    if {item.name for item in output.iterdir()} != required:
        raise RuntimeError("official packet contains stale or orphan files")
    for row in artifacts:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or row.get("dtype") not in {"f32", "u32", "u8"}:
            raise RuntimeError("official packet artifact schema is invalid")
        path = output / row["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row.get("bytes") or path.stat().st_size <= 0:
            raise RuntimeError(f"official packet artifact is missing or inconsistent: {path.name}")
        shape = row.get("shape")
        if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            raise RuntimeError(f"official packet shape is invalid: {path.name}")
        if row.get("bytes") != (1 if row["dtype"] == "u8" else 4) * int(__import__("math").prod(shape)):
            raise RuntimeError(f"official packet shape/byte mismatch: {path.name}")
        if row.get("sha256") != sha256(path):
            raise RuntimeError(f"official packet hash mismatch: {path.name}")
    rng_records = captures.get("rng_records")
    if len(rng_records) != 1 or rng_records[0].get("operation") != "torch.randn":
        raise RuntimeError("official RF sampler RNG call order/count is not the expected single initial-noise call")
    if not captures.get("normalized_text") or not all(str(text).strip() for text in captures["normalized_text"]):
        raise RuntimeError("official tokenizer did not return nonempty normalized text")
    sampler = captures.get("sampler", {})
    if sampler.get("seed") != int(args.seed) or sampler.get("num_steps") != int(args.num_steps):
        raise RuntimeError("official sampler seed/step contract was not captured")
    schedule = next(row for row in artifacts if row["path"] == "official_u_schedule.f32le")
    if schedule.get("shape") != [int(args.num_steps) + 1]:
        raise RuntimeError("official schedule shape does not match num_steps")
    if captures.get("used_seed") != int(args.seed):
        raise RuntimeError("official runtime did not use the caller-provided seed")
    if captures.get("sample_rate_hz") != 48_000:
        raise RuntimeError("official codec sample rate is not the authenticated 48 kHz route")
    pcm = next(row for row in artifacts if row["path"] == "official_pcm.f32le")
    pcm_elements = int(__import__("math").prod(pcm["shape"])) if pcm.get("shape") else 0
    if pcm_elements <= 0 or pcm_elements > 1_440_000 * 2:
        raise RuntimeError("official PCM artifact is empty")


def checkout_revision(path: Path) -> None:
    if not (path / ".git").is_dir():
        raise RuntimeError("official source checkout must contain .git")
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"source revision {revision} != pinned {UPSTREAM_REVISION}")
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip():
        raise RuntimeError("official source checkout is dirty")
    origin = subprocess.check_output(
        ["git", "-C", str(path), "remote", "get-url", "origin"], text=True
    ).strip()
    if origin.removesuffix(".git") != "https://github.com/Aratako/Irodori-TTS":
        raise RuntimeError(f"official source origin mismatch: {origin}")


def write_manifest(output: Path, status: str, **extra: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT,
        "reference_status": status,
        "source": {
            "repository": "https://github.com/Aratako/Irodori-TTS.git",
            "revision": UPSTREAM_REVISION,
            "pyproject_sha256": SOURCE_PYPROJECT_SHA256,
            "uv_lock_sha256": SOURCE_LOCK_SHA256,
            "dacvae_revision": DACVAE_REVISION,
            "execution": "BLOCKED_BEFORE_SYNC_OR_IMPORT",
        },
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "codec": {"repository": CODEC_REPOSITORY, "revision": CODEC_REVISION},
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "source_config_note": "source names the repository only; this immutable snapshot is adapted evidence, not an upstream pin",
        },
        "artifacts": [],
        **extra,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> int:
    with __import__("tempfile").TemporaryDirectory() as directory:
        output = Path(directory)
        write_manifest(output, "REFERENCE_BLOCKED", blockers=["test"])
        data = strict_json((output / "manifest.json").read_text(encoding="utf-8"))
        assert data["reference_status"] == "REFERENCE_BLOCKED"
        assert data["tokenizer"]["revision"] == TOKENIZER_REVISION
        try:
            strict_json('{"x": 1, "x": 2}')
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate inspection JSON accepted")
        assert dependency_gate() == 2
    assert len(TOKENIZER_REVISION) == 40 and all(c in "0123456789abcdef" for c in TOKENIZER_REVISION)
    source = Path(__file__).read_text(encoding="utf-8")
    assert "AutoTokenizer" in source and "sample_euler_rf_cfg" in source
    assert "REFERENCE_EVIDENCE_COMPLETE" in source
    assert "official source seeds/allocates x_t internally" in source
    assert "require_source_environment" in source and "uv.lock" in source
    assert "BLOCKED_BEFORE_SYNC_OR_IMPORT" in source
    print("irodori_500m_v3_dump_reference.py self-test: OK")
    return 0


def run(args: argparse.Namespace) -> int:
    # Gate before resolving or creating the output path, even when called
    # directly rather than through a worker.
    if dependency_gate() != 0:
        return 2
    output = args.output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            print(f"Irodori official reference blocked: output must be absent or empty: {output}", file=sys.stderr)
            return 2
    else:
        output.mkdir(parents=True)
    captures: dict[str, Any] = {"artifacts": [], "rng_records": []}
    write_manifest(
        output,
        "REFERENCE_BLOCKED",
        blockers=[
            "authenticated source lock resolves DACVAE -> descript-audiotools -> librosa -> soxr/soundfile",
            "stdlib dependency gate blocked before source import and model execution",
        ],
    )
    return 2
    link: Path | None = None
    created_link = False
    try:
        upstream = args.upstream.resolve()
        checkout_revision(upstream)
        require_source_environment(upstream)
        checkpoint = require_file(args.checkpoint.resolve(), "checkpoint")
        codec = require_file(args.codec.resolve(), "codec weights")
        tokenizer = require_tokenizer(args.tokenizer.resolve())
        inspection_manifest = require_file(args.inspection_manifest.resolve(), "inspection manifest")
        verify_inspection_manifest(inspection_manifest, checkpoint, codec, tokenizer)
        link, created_link = install_tokenizer_link(checkpoint, tokenizer)

        # Avoid irodori_tts/__init__.py's training-only imports. The official
        # tokenizer.py calls AutoTokenizer.from_pretrained on this local path.
        # Every implementation below is still imported from the fixed official
        # checkout; this package shell only controls import side effects.
        sys.path.insert(0, str(upstream))
        package = types.ModuleType("irodori_tts")
        package.__path__ = [str(upstream / "irodori_tts")]
        sys.modules["irodori_tts"] = package
        import torch
        import irodori_tts.inference_runtime as official_runtime
        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

        key = RuntimeKey(
            checkpoint=str(checkpoint), model_device="cpu", codec_repo=str(codec),
            codec_device="cpu", model_precision="fp32", codec_precision="fp32",
            compile_model=False,
        )
        runtime = InferenceRuntime.from_key(key)

        def capture_batch(texts: Any, max_length: int | None = None) -> Any:
            result = original_batch(texts, max_length=max_length)
            captures["normalized_text"] = [str(text) for text in texts]
            captures["artifacts"].append(
                write_tensor(output, "text_input_ids.u32le", result[0], dtype="u32")
            )
            captures["artifacts"].append(
                write_tensor(output, "text_attention_mask.u8", result[1], dtype="u8")
            )
            return result

        original_batch = runtime.tokenizer.batch_encode
        runtime.tokenizer.batch_encode = capture_batch
        original_duration = runtime.model.predict_duration_log_frames

        def capture_duration(*call_args: Any, **call_kwargs: Any) -> Any:
            result = original_duration(*call_args, **call_kwargs)
            captures["artifacts"].append(
                write_tensor(output, "predicted_log_frames.f32le", result, dtype="f32")
            )
            return result

        runtime.model.predict_duration_log_frames = capture_duration
        original_sampler = official_runtime.sample_euler_rf_cfg
        original_randn = torch.randn
        original_linspace = torch.linspace

        def capture_randn(*call_args: Any, **call_kwargs: Any) -> Any:
            value = original_randn(*call_args, **call_kwargs)
            captures["rng_records"].append({
                "call_site": "irodori_tts.rf.sample_euler_rf_cfg",
                "operation": "torch.randn",
                "shape": list(value.shape), "device": str(value.device),
                "generator_supplied": call_kwargs.get("generator") is not None,
            })
            if len(captures["rng_records"]) == 1:
                captures["artifacts"].append(
                    write_tensor(output, "initial_noise.f32le", value, dtype="f32")
                )
            return value

        def capture_linspace(*call_args: Any, **call_kwargs: Any) -> Any:
            value = original_linspace(*call_args, **call_kwargs)
            captures["artifacts"].append(
                write_tensor(output, "official_u_schedule.f32le", value, dtype="f32")
            )
            return value

        def capture_sampler(*call_args: Any, **call_kwargs: Any) -> Any:
            captures["sampler"] = {
                key: value for key, value in call_kwargs.items()
                if key in {"sequence_length", "num_steps", "seed", "cfg_scale_text",
                           "cfg_scale_caption", "cfg_scale_speaker", "cfg_guidance_mode",
                           "cfg_min_t", "cfg_max_t", "t_schedule_mode", "sway_coeff"}
            }
            torch.randn = capture_randn
            torch.linspace = capture_linspace
            try:
                value = original_sampler(*call_args, **call_kwargs)
            finally:
                torch.randn = original_randn
                torch.linspace = original_linspace
            captures["artifacts"].append(
                write_tensor(output, "sampled_patched_latent.f32le", value, dtype="f32")
            )
            return value

        official_runtime.sample_euler_rf_cfg = capture_sampler
        request = SamplingRequest(
            text=args.text, no_ref=True, num_candidates=1, num_steps=args.num_steps,
            seconds=args.seconds, duration_scale=args.duration_scale,
            min_seconds=0.5, max_seconds=30.0, cfg_guidance_mode="independent",
            seed=args.seed, trim_tail=False,
        )
        try:
            result = runtime.synthesize(request)
        finally:
            runtime.tokenizer.batch_encode = original_batch
            runtime.model.predict_duration_log_frames = original_duration
            official_runtime.sample_euler_rf_cfg = original_sampler
            torch.randn = original_randn
            torch.linspace = original_linspace
        captures["artifacts"].append(
            write_tensor(output, "official_pcm.f32le", result.audio, dtype="f32")
        )
        captures["sample_rate_hz"] = int(result.sample_rate)
        captures["stage_timings"] = [[name, float(seconds)] for name, seconds in result.stage_timings]
        captures["used_seed"] = int(result.used_seed)
        captures["rng_contract"] = "official source seeds/allocates x_t internally"
        validate_complete_packet(captures, output, args)
        captures["status"] = "OFFICIAL_RUNTIME_EXECUTED"
        write_manifest(output, "REFERENCE_EVIDENCE_COMPLETE", **captures)
        print(f"Irodori official reference written to {output}")
        return 0
    except Exception as error:
        write_manifest(
            output,
            "REFERENCE_BLOCKED",
            error=f"{type(error).__name__}: {error}",
            artifacts=captures.get("artifacts", []),
            rng_records=captures.get("rng_records", []),
            blockers=[
                "official source environment must be run through its authenticated uv.lock",
                "checkpoint tensor-role mapping and codec topology evidence are not accepted for native binding",
            ],
        )
        print(f"Irodori official reference blocked: {error}", file=sys.stderr)
        return 2
    finally:
        if created_link and link is not None:
            link.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text", default="こんにちは。これは固定参照です。")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not all((args.upstream, args.checkpoint, args.codec, args.tokenizer, args.inspection_manifest, args.output)):
        parser.error("reference requires --upstream --checkpoint --codec --tokenizer --inspection-manifest --output")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
