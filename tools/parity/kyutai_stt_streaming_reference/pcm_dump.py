#!/usr/bin/env -S uv run --no-sync --project tools/parity --python 3.12 python
"""Capture an official Kyutai PCM -> Mimi -> LMGen reference packet.

This entry point is deliberately separate from ``dump.py``.  The latter starts
from an already-authenticated Mimi-code boundary; this module starts from one
authenticated raw PCM byte stream and invokes the official Mimi encoder before
every official LMGen step.  ``real`` is VAST-only and remains blocked while the
reviewed dependency-closure allowlist is empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any

from contract import (
    CONTEXT,
    HEX64,
    HF_REPOSITORY,
    HF_REVISION,
    MOSHI_REPOSITORY,
    MOSHI_REVISION,
    MODEL_TENSOR_MANIFEST_SHA256,
    N_Q,
    TEXT_CARD,
    authenticate_composite_model,
    git_identity,
    require_execution_readiness,
    require_pcm_source_packet,
    require_source_packet,
    real_path,
    _read_bounded,
)
from dump import (
    RingCapture,
    _MAX_ARTIFACT_BYTES,
    _MAX_LOGIT_ARTIFACT_BYTES,
    _artifact_record,
    _decoder_helpers,
    _tensor_bytes,
    _write_output,
    checkpoint_steps,
)


PCM_SCHEMA = "vokra-kyutai-stt-independent-pcm-reference-v1"
PCM_SCOPE = (
    "official raw PCM -> Mimi encode -> official Moshi LMGen streaming; "
    "includes main-LM RingKV evidence, no native parity or publication claim"
)
PCM_SAMPLE_RATE = 24_000
PCM_CHANNELS = 1
PCM_BYTES_PER_SAMPLE = 4
PCM_INPUT_DTYPE = "float32-le"
PCM_MAX_INPUT_BYTES = 64 * 1024 * 1024
PCM_MAX_FRAMES = CONTEXT + 2
PCM_LOGIT_BYTES_PER_STEP = TEXT_CARD * 4
PCM_MAX_TOKEN_VALUES = 4096


def _read_pcm_input(path: Path, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    """Read only canonical mono 24 kHz float32-le bytes, with a hard bound."""
    path = real_path(path, "raw PCM input")
    if not isinstance(expected_sha256, str) or HEX64.fullmatch(expected_sha256) is None:
        raise ValueError("raw PCM expected sha256 must be lowercase hexadecimal")
    size = path.stat().st_size
    if size > PCM_MAX_INPUT_BYTES:
        raise ValueError("raw PCM input exceeds its bounded byte budget")
    if size == 0 or size % PCM_BYTES_PER_SAMPLE != 0:
        raise ValueError("raw PCM input must be non-empty float32-le bytes")
    body = _read_bounded(
        path,
        "raw PCM input",
        PCM_MAX_INPUT_BYTES,
        expected_size=size,
    )
    if len(body) != size:
        raise ValueError("raw PCM input changed while reading")
    observed_sha256 = hashlib.sha256(body).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("raw PCM input sha256 mismatch")
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", body)):
        raise ValueError("raw PCM input contains non-finite float32 samples")
    return body, {
        "path": str(path),
        "bytes": len(body),
        "sha256": observed_sha256,
        "sha256_expected": expected_sha256,
        "sha256_verified": True,
        "sample_rate": PCM_SAMPLE_RATE,
        "channels": PCM_CHANNELS,
        "dtype": PCM_INPUT_DTYPE,
        "endianness": "little",
        "layout": "interleaved-mono",
        "samples": len(body) // PCM_BYTES_PER_SAMPLE,
    }


def _padding_plan(
    sample_count: int,
    *,
    sample_rate: int,
    frame_size: int,
    prefix_seconds: float,
    suffix_seconds: float,
) -> dict[str, Any]:
    """Reproduce the official evaluator's prefix/suffix/ceil framing."""
    if sample_count <= 0 or sample_rate <= 0 or frame_size <= 0:
        raise ValueError("PCM framing inputs must be positive")
    if not all(math.isfinite(value) and value >= 0 for value in (prefix_seconds, suffix_seconds)):
        raise ValueError("PCM padding seconds must be finite and non-negative")
    prefix_samples = int(prefix_seconds * sample_rate)
    suffix_samples = int(suffix_seconds * sample_rate)
    unrounded = sample_count + prefix_samples + suffix_samples
    padded_samples = unrounded
    if padded_samples % frame_size:
        padded_samples += frame_size - padded_samples % frame_size
    frames = padded_samples // frame_size
    if frames > PCM_MAX_FRAMES:
        raise ValueError("padded PCM exceeds the authenticated 377-frame boundary")
    return {
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "prefix_seconds": prefix_seconds,
        "suffix_seconds": suffix_seconds,
        "prefix_samples": prefix_samples,
        "suffix_samples": suffix_samples,
        "unrounded_samples": unrounded,
        "padded_samples": padded_samples,
        "frames": frames,
        "padding": "official evaluator prefix/suffix then ceil to frame_size",
    }


def _validate_codes(codes: Any, *, expected_codebooks: int) -> tuple[tuple[int, ...], str, bytes]:
    if expected_codebooks != N_Q or expected_codebooks != 32:
        raise ValueError("Mimi codebook count is not the authenticated 32")
    shape = tuple(int(dimension) for dimension in getattr(codes, "shape", ()))
    dtype = str(getattr(codes, "dtype", ""))
    if shape != (1, expected_codebooks, 1):
        raise ValueError(f"Mimi code shape is not (1,{expected_codebooks},1)")
    if dtype != "torch.int64":
        raise ValueError("Mimi codes must remain official torch.int64 at the LM boundary")
    values = codes.detach().cpu().reshape(-1).tolist()
    if any(not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < 2048 for item in values):
        raise ValueError("Mimi codes exceed the authenticated cardinality 2048")
    body = _tensor_bytes(codes)
    if len(body) != expected_codebooks * 8:
        raise ValueError("Mimi code bytes do not match the authenticated shape")
    return shape, dtype, body


def _validate_pcm_logits(value: Any, *, reserved_bytes: int) -> tuple[list[int], list[int], str, bytes]:
    source_shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    if source_shape != (1, 1, 1, TEXT_CARD):
        raise ValueError("official PCM logits must have shape [1,1,1,text_card]")
    numel = getattr(value, "numel", None)
    if not callable(numel) or int(numel()) != TEXT_CARD:
        raise ValueError("official PCM logits cardinality mismatch")
    isfinite = getattr(value, "isfinite", None)
    if not callable(isfinite):
        raise TypeError("official PCM logits expose no finite check")
    finite = isfinite()
    all_finite = getattr(finite, "all", None)
    item = getattr(all_finite(), "item", None) if callable(all_finite) else None
    if not callable(item) or not bool(item()):
        raise ValueError("official PCM logits contain non-finite values")
    if reserved_bytes + PCM_LOGIT_BYTES_PER_STEP > _MAX_LOGIT_ARTIFACT_BYTES:
        raise ValueError("official PCM logits exceed their bounded artifact")
    source_dtype = str(getattr(value, "dtype", ""))
    converted = value.detach().cpu().float().contiguous().reshape(1, TEXT_CARD)
    if tuple(int(dimension) for dimension in getattr(converted, "shape", ())) != (1, TEXT_CARD):
        raise ValueError("PCM logit comparison reshape is not [1,text_card]")
    if str(getattr(converted, "dtype", "")) != "torch.float32" or int(converted.element_size()) != 4:
        raise ValueError("PCM logit comparison export is not contiguous float32")
    body = _tensor_bytes(converted)
    if len(body) != PCM_LOGIT_BYTES_PER_STEP:
        raise ValueError("float32 PCM logit export size mismatch")
    return list(source_shape), [1, TEXT_CARD], source_dtype, body


def _tensor_values(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str = "torch.int64",
    upper_bound: int = TEXT_CARD,
) -> tuple[list[int], tuple[int, ...], str]:
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    if shape != expected_shape or str(getattr(value, "dtype", "")) != expected_dtype:
        raise ValueError(f"{label} shape/dtype is not the authenticated contract")
    numel = getattr(value, "numel", None)
    if not callable(numel) or int(numel()) > PCM_MAX_TOKEN_VALUES:
        raise ValueError(f"{label} exceeds bounded token metadata")
    values = value.detach().cpu().reshape(-1).tolist()
    if any(not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < upper_bound for item in values):
        raise ValueError(f"{label} contains an out-of-range token")
    return [int(item) for item in values], shape, str(getattr(value, "dtype", ""))


def _environment_fingerprint() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "toolchain": sys.version,
        "torch_cpu_capability": "unresolved-before-import",
        "dtype": "unresolved-before-model",
    }


def _import_official_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    """Import only the pinned official runtime; tests replace this seam."""
    sys.path.insert(0, str(args.moshi_source))
    sys.path.insert(0, str(args.moshi_source / "moshi"))
    sys.path.insert(0, str(args.dsm_source))
    try:
        import torch  # type: ignore
        from moshi.models import LMGen, loaders  # type: ignore
        from moshi.modules import transformer  # type: ignore
    except ImportError as error:
        raise SystemExit(f"official Moshi implementation is required: {error}") from error
    return torch, transformer, LMGen, loaders


def _build_checkpoint_info(loaders: Any, model_root: Path, decoder: Any) -> Any:
    """Use only local authenticated overrides; prohibit optional HF branches."""
    info = loaders.CheckpointInfo.from_hf_repo(
        HF_REPOSITORY,
        moshi_weights=model_root / decoder.MODEL_NAME,
        mimi_weights=model_root / decoder.MIMI_NAME,
        tokenizer=model_root / decoder.TOKENIZER_NAME,
        config_path=model_root / "config.json",
        revision=HF_REVISION,
    )
    raw_config = getattr(info, "raw_config", None)
    if not isinstance(raw_config, dict) or raw_config.get("model_type") != "stt":
        raise ValueError("authenticated Kyutai config is not the STT model")
    if raw_config.get("mimi_config_name") is not None or raw_config.get("lora_name") is not None:
        raise ValueError("optional Mimi/HF or LoRA config branch is not permitted")
    if getattr(info, "mimi_config", None) is not None or getattr(info, "lora_weights", None) is not None:
        raise ValueError("optional Mimi config or LoRA weights are not permitted")
    return info


def _check_artifact_budget(files: dict[str, Any], manifest: dict[str, Any]) -> None:
    if len(files) > 4096:
        raise ValueError("PCM reference artifact file count exceeds its bound")
    manifest_body = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    total_bytes = len(manifest_body) + sum(len(body) for body in files.values())
    if total_bytes > _MAX_ARTIFACT_BYTES:
        raise ValueError("PCM reference artifact budget exceeded before writer allocation")


def self_test() -> None:
    assert PCM_SCHEMA.endswith("-v1")
    assert PCM_SAMPLE_RATE == 24_000
    assert PCM_CHANNELS == 1
    plan = _padding_plan(
        10,
        sample_rate=24_000,
        frame_size=1_920,
        prefix_seconds=1.0,
        suffix_seconds=3.0,
    )
    assert plan["padded_samples"] % 1_920 == 0
    assert plan["frames"] == 51
    print("kyutai STT official PCM producer self-test PASS")


def real(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing PCM reference")
    checkout = Path.cwd()
    readiness = require_execution_readiness(
        args.approval_evidence,
        expected_head=args.expected_head,
        expected_approval_sha256=args.approval_sha256,
        checkout=checkout,
        dependency_closure_sha256=args.dependency_closure_sha256,
    )
    if args.source_packet is None:
        raise ValueError("official PCM capture requires an authenticated source packet")

    # Both contracts are required: the PCM role set does not replace the
    # narrower LM/transformer role authentication used by the existing oracle.
    pcm_source = require_pcm_source_packet(args.source_packet)
    lm_source = require_source_packet(args.source_packet)
    decoder = _decoder_helpers()
    decoder.require_clean_head(args.expected_head)
    dsm_identity = git_identity(args.dsm_source, decoder.DSM_REPOSITORY, decoder.DSM_REVISION)
    moshi_identity = git_identity(args.moshi_source, MOSHI_REPOSITORY, MOSHI_REVISION)
    source_contract = decoder.authenticate_streaming_source_contract(
        args.dsm_source, args.moshi_source
    )
    model_record = authenticate_composite_model(args.model)
    decoder_record = decoder.authenticate_model(args.model / decoder.MODEL_NAME, args.model / "config.json")
    if decoder_record.get("tensor_manifest_sha256") != MODEL_TENSOR_MANIFEST_SHA256:
        raise ValueError("authenticated model tensor manifest mismatch")
    model_record["tensor_manifest_sha256"] = decoder_record["tensor_manifest_sha256"]
    raw_pcm, pcm_input = _read_pcm_input(args.pcm_input, args.pcm_input_sha256)
    environment = _environment_fingerprint()

    torch, transformer, LMGen, loaders = _import_official_runtime(args)

    environment["torch_cpu_capability"] = getattr(
        torch.backends.cpu, "get_cpu_capability", lambda: "unknown"
    )()
    environment["dtype"] = str(torch.bfloat16)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("locked single-thread execution could not be established")
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("deterministic torch algorithms could not be enabled")

    info = _build_checkpoint_info(loaders, args.model, decoder)
    mimi = info.get_mimi(device="cpu")
    lm = info.get_moshi(device="cpu", dtype=torch.bfloat16)
    if (lm.dep_q, lm.n_q, lm.text_card, lm.dim, lm.context) != (0, N_Q, TEXT_CARD, 2048, CONTEXT):
        raise ValueError("official streaming LM axes are not the authenticated contract")
    if int(mimi.sample_rate) != PCM_SAMPLE_RATE or int(mimi.channels) != PCM_CHANNELS:
        raise ValueError("official Mimi geometry is not canonical mono 24 kHz")
    frame_size = int(mimi.frame_size)
    if frame_size != 1_920 or abs(float(mimi.frame_rate) - 12.5) > 0.0:
        raise ValueError("official Mimi frame geometry is not 1920 samples at 12.5 Hz")
    expected_codebooks = int(mimi.num_codebooks)
    if expected_codebooks != 32 or expected_codebooks != lm.num_codebooks - lm.dep_q - 1:
        raise ValueError("Mimi codebooks do not match official LM input contract")
    prefix_seconds = float(info.stt_config.get("audio_silence_prefix_seconds", 1.0))
    audio_delay_seconds = float(info.stt_config.get("audio_delay_seconds", 5.0))
    if prefix_seconds != 1.0 or audio_delay_seconds != 2.5:
        raise ValueError("authenticated STT padding must be prefix 1.0s and audio delay 2.5s")
    suffix_seconds = audio_delay_seconds + 0.5
    padding = _padding_plan(
        pcm_input["samples"],
        sample_rate=int(mimi.sample_rate),
        frame_size=frame_size,
        prefix_seconds=prefix_seconds,
        suffix_seconds=suffix_seconds,
    )
    if padding["frames"] != PCM_MAX_FRAMES:
        raise ValueError("official PCM capture requires exactly the authenticated 377 frames")
    pcm_tensor = torch.frombuffer(bytearray(raw_pcm), dtype=torch.float32).clone().view(1, 1, -1)
    pcm_tensor = torch.nn.functional.pad(
        pcm_tensor,
        (padding["prefix_samples"], padding["suffix_samples"]),
    )
    pcm_tensor = torch.nn.functional.pad(
        pcm_tensor,
        (0, padding["padded_samples"] - padding["unrounded_samples"]),
    )

    logits_bytes = bytearray()
    logits_events: list[dict[str, Any]] = []
    token_bytes = bytearray()
    token_events: list[dict[str, Any]] = []
    step_output_bytes = bytearray()
    step_events: list[dict[str, Any]] = []
    current_phase = "warmup"
    current_step = 0

    def on_logits(value: Any) -> None:
        source_shape, comparison_shape, source_dtype, body = _validate_pcm_logits(
            value, reserved_bytes=len(logits_bytes)
        )
        offset = len(logits_bytes)
        logits_bytes.extend(body)
        logits_events.append({
            "phase": current_phase,
            "step": current_step,
            "lm_call_ordinal": 0,
            "source_shape": source_shape,
            "comparison_shape": comparison_shape,
            "dtype": "torch.float32",
            "source_dtype": source_dtype,
            "conversion": "official logits converted to contiguous torch.float32 for comparison export",
            "offset": offset,
            "bytes": len(body),
        })

    def on_text(value: Any) -> None:
        values, shape, dtype = _tensor_values(
            value,
            label="official text token",
            expected_shape=(1,),
            upper_bound=TEXT_CARD,
        )
        offset = len(token_bytes)
        for item in values:
            token_bytes.extend(int(item).to_bytes(8, "little", signed=False))
        token_events.append({
            "phase": current_phase,
            "step": current_step,
            "lm_call_ordinal": 0,
            "shape": list(shape),
            "dtype": dtype,
            "values": values,
            "offset": offset,
            "bytes": len(values) * 8,
        })

    lm_gen = LMGen(
        lm,
        use_sampling=False,
        temp=0.0,
        temp_text=0.0,
        top_k_text=50,
        check=True,
        on_text_hook=on_text,
        on_text_logits_hook=on_logits,
    )
    capture = RingCapture(checkpoint_steps(CONTEXT))
    code_bytes = bytearray()
    try:
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            if lm_gen._streaming_state is None:
                raise ValueError("official LMGen did not initialize streaming state")
            capture.install(transformer, lm)
            capture.record_state_snapshot("initial", 0)
            for current_step in range(padding["frames"]):
                capture.phase = current_phase
                capture.step = current_step
                start = current_step * frame_size
                frame = pcm_tensor[:, :, start : start + frame_size]
                codes = mimi.encode(frame)
                code_shape, code_dtype, code_body = _validate_codes(
                    codes, expected_codebooks=expected_codebooks
                )
                code_offset = len(code_bytes)
                code_bytes.extend(code_body)
                output = lm_gen.step(codes)
                if output is None:
                    raise ValueError("official LMGen returned no streaming frame")
                output_record: dict[str, Any] = {"returned": output is not None}
                if output is not None:
                    output_values, output_shape, output_dtype = _tensor_values(
                        output,
                        label="official LM step output",
                        expected_shape=(1, 1, 1),
                        upper_bound=TEXT_CARD,
                    )
                    output_offset = len(step_output_bytes)
                    for item in output_values:
                        step_output_bytes.extend(int(item).to_bytes(8, "little", signed=False))
                    output_record.update({
                        "shape": list(output_shape),
                        "dtype": output_dtype,
                        "values": output_values,
                        "offset": output_offset,
                        "bytes": len(output_values) * 8,
                    })
                step_events.append({
                    "phase": current_phase,
                    "step": current_step,
                    "lm_call_ordinal": 0,
                    "pcm_offset_samples": start,
                    "pcm_samples": frame_size,
                    "codes": {"shape": list(code_shape), "dtype": code_dtype, "offset": code_offset, "bytes": len(code_body)},
                    "lm_output": output_record,
                })
            reset_mask = torch.ones(1, dtype=torch.bool, device=lm.device)
            mimi.reset_streaming(reset_mask)
            lm_gen.reset_streaming(reset_mask)
            current_phase = "after_reset"
            current_step = 0
            capture.record_state_snapshot(current_phase, current_step)
            capture.phase = current_phase
            capture.step = current_step
            frame = pcm_tensor[:, :, :frame_size]
            codes = mimi.encode(frame)
            code_shape, code_dtype, code_body = _validate_codes(
                codes, expected_codebooks=expected_codebooks
            )
            code_offset = len(code_bytes)
            code_bytes.extend(code_body)
            output = lm_gen.step(codes)
            if output is None:
                raise ValueError("official LMGen returned no post-reset frame")
            output_record = {"returned": output is not None}
            if output is not None:
                output_values, output_shape, output_dtype = _tensor_values(
                    output,
                    label="official post-reset LM step output",
                    expected_shape=(1, 1, 1),
                    upper_bound=TEXT_CARD,
                )
                output_offset = len(step_output_bytes)
                for item in output_values:
                    step_output_bytes.extend(int(item).to_bytes(8, "little", signed=False))
                output_record.update({"shape": list(output_shape), "dtype": output_dtype, "values": output_values, "offset": output_offset, "bytes": len(output_values) * 8})
            step_events.append({
                "phase": current_phase,
                "step": current_step,
                "lm_call_ordinal": 0,
                "pcm_offset_samples": 0,
                "pcm_samples": frame_size,
                "codes": {"shape": list(code_shape), "dtype": code_dtype, "offset": code_offset, "bytes": len(code_body)},
                "lm_output": output_record,
            })
    finally:
        capture.uninstall(transformer)

    capture.require_complete_coverage()
    if (
        len(step_events) != PCM_MAX_FRAMES + 1
        or len(logits_events) != PCM_MAX_FRAMES + 1
        or len(token_events) != PCM_MAX_FRAMES + 1
    ):
        raise RuntimeError("official PCM capture is missing a joint step, logits, or token event")
    files: dict[str, bytes] = {
        "input/raw_pcm.f32le": raw_pcm,
        "mimi/codes.i64": bytes(code_bytes),
        "lm/text_logits.f32": bytes(logits_bytes),
        "lm/text_tokens.i64": bytes(token_bytes),
        "lm/step_outputs.i64": bytes(step_output_bytes),
    }
    files.update({name: bytes(body) for name, body in capture.files.items()})
    ring_events = []
    for event in capture.events:
        enriched = dict(event)
        enriched["lm_call_ordinal"] = 0
        ring_events.append(enriched)
    kv_artifact_bytes = sum(
        len(body) for name, body in capture.files.items() if name.startswith("kv/")
    )
    manifest = {
        "format": PCM_SCHEMA,
        "status": "REFERENCE_READY",
        "scope": PCM_SCOPE,
        "claim_boundary": "official raw PCM -> Mimi -> LMGen only; no native parity/publication claim",
        "expected_head": args.expected_head,
        "dependency_closure_sha256": readiness["dependency_closure_sha256"],
        "approval_sha256": args.approval_sha256,
        "source": {"pcm": pcm_source, "lm": lm_source, "dsm": dsm_identity, "moshi": moshi_identity, "contract": source_contract},
        "model": model_record,
        "input": {"raw_pcm": pcm_input, "padding": padding},
        "mimi": {"device": "cpu", "sample_rate": int(mimi.sample_rate), "frame_rate": float(mimi.frame_rate), "frame_size": frame_size, "channels": int(mimi.channels), "num_codebooks": expected_codebooks, "dtype": str(next(mimi.parameters()).dtype)},
        "lm": {"device": "cpu", "n_q": int(lm.n_q), "dep_q": int(lm.dep_q), "text_card": int(lm.text_card), "dim": int(lm.dim), "context": int(lm.context), "weights_dtype": str(lm.dtype), "generation": {"use_sampling": False, "temp": 0.0, "temp_text": 0.0, "top_k_text": 50, "check": True}},
        "execution": {"implementation": "official CheckpointInfo.get_mimi -> MimiModel.encode -> LMGen.step", "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "publication": "NO_UPLOAD", "environment": environment},
        "coverage": {"warmup_frames": PCM_MAX_FRAMES, "post_reset_frames": 1, "mimi_encode_calls": len(step_events), "lm_step_calls": len(step_events), "logit_events": len(logits_events), "text_events": len(token_events)},
        "joint_steps": step_events,
        "text": {"logits": logits_events, "tokens": token_events},
        "ring_cache": {"capture": "RingCapture official RingKVCache.complete hook", "checkpoints": sorted(checkpoint_steps(CONTEXT)), "checkpoint_file_pattern": "kv/{phase}-step-{step:04d}-layer-{layer:02d}-{keys|values}.bin", "events": ring_events, "state_snapshots": capture.state_snapshots, "reset_mask": [True], "lm_calls": len(step_events), "calls_per_pcm_frame": 1, "kv_artifact_bytes": kv_artifact_bytes, "comparison": "retain official raw BF16 ring evidence; no native parity verdict"},
        "artifacts": _artifact_record(files),
        "artifact_budget_bytes": _MAX_ARTIFACT_BYTES,
    }
    _check_artifact_budget(files, manifest)
    _write_output(args.out, files, manifest)
    print(f"PCM reference written: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    real_parser = sub.add_parser("real")
    real_parser.add_argument("--model", type=Path, required=True)
    real_parser.add_argument("--dsm-source", type=Path, required=True)
    real_parser.add_argument("--moshi-source", type=Path, required=True)
    real_parser.add_argument("--source-packet", type=Path, required=True)
    real_parser.add_argument("--pcm-input", type=Path, required=True)
    real_parser.add_argument("--pcm-input-sha256", required=True)
    real_parser.add_argument("--out", type=Path, required=True)
    real_parser.add_argument("--expected-head", required=True)
    real_parser.add_argument("--approval-evidence", type=Path, required=True)
    real_parser.add_argument("--approval-sha256", required=True)
    real_parser.add_argument("--dependency-closure-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
    else:
        real(args)


if __name__ == "__main__":
    main()
