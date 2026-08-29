#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Run the pinned official VoxCPM-0.5B source with a deterministic packet.

The driver calls the real ``VoxCPMModel.generate`` API from the fixed source.
It does not mirror the model. The packet intentionally limits this evidence
run to one autoregressive feature step, so a temporary ``torch.randn`` hook
accepts exactly the one inference draw used by official ``UnifiedCFM.forward``.
Any call/shape/type drift or unconsumed draw is a hard reference error.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION = "38a76704ee67935ccbafbe5b6725e83dbb1e9305"
HF_REPOSITORY = "openbmb/VoxCPM-0.5B"
HF_REVISION = "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_ROLES = (
    "src/voxcpm/model/voxcpm.py",
    "src/voxcpm/modules/locdit/unified_cfm.py",
)
OFFICIAL_GENERATE_ARGUMENTS = (
    "target_text", "prompt_text", "prompt_wav_path", "min_len", "max_len",
    "inference_timesteps", "cfg_value", "seed",
)
OFFICIAL_RANDOM_DRAW_SHAPE = (1, 64, 2)
PACKET_KEYS = {
    "text", "prompt_text", "prompt_pcm", "sample_rate", "token_ids", "noise",
    "noise_dtype", "noise_device", "min_len", "max_len", "inference_timesteps",
    "cfg_value", "seed",
}


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate packet key: {key}")
        result[key] = value
    return result


def load_packet(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    if not isinstance(packet, dict) or set(packet) != PACKET_KEYS:
        raise RuntimeError(f"packet keys must be exactly {sorted(PACKET_KEYS)}")
    if not isinstance(packet["text"], str) or not packet["text"]:
        raise RuntimeError("packet text must be non-empty")
    if not isinstance(packet["prompt_text"], str):
        raise RuntimeError("packet prompt_text must be a string")
    if packet["sample_rate"] != 16_000:
        raise RuntimeError("VoxCPM-0.5B reference requires 16 kHz PCM")
    pcm = packet["prompt_pcm"]
    if not isinstance(pcm, list) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or abs(float(value)) > 1.0
        for value in pcm
    ):
        raise RuntimeError("prompt_pcm must contain finite normalized samples")
    token_calls = packet["token_ids"]
    if not isinstance(token_calls, list) or not token_calls or any(
        not isinstance(call, list) or not call or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in call
        )
        for call in token_calls
    ):
        raise RuntimeError("token_ids must be a non-empty list of expected tokenizer calls")
    noise = packet["noise"]
    if not isinstance(noise, list) or len(noise) != 128 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) for value in noise
    ):
        raise RuntimeError("noise must contain exactly 128 finite caller-owned values")
    if packet["noise_dtype"] not in {"torch.float32", "torch.float16", "torch.bfloat16"}:
        raise RuntimeError("noise_dtype must name a supported torch floating dtype")
    if packet["noise_device"] != "cpu":
        raise RuntimeError("reference source must run on the requested CPU device")
    for key in ("min_len", "max_len", "inference_timesteps", "seed"):
        if not isinstance(packet[key], int) or isinstance(packet[key], bool) or packet[key] < 0:
            raise RuntimeError(f"packet {key} must be a nonnegative integer")
    if packet["max_len"] == 0 or packet["inference_timesteps"] == 0:
        raise RuntimeError("packet max_len and inference_timesteps must be positive")
    if packet["max_len"] != 1 or packet["min_len"] > 1:
        raise RuntimeError("this packet contract permits exactly one autoregressive feature step")
    if not isinstance(packet["cfg_value"], (int, float)) or isinstance(packet["cfg_value"], bool) or not math.isfinite(float(packet["cfg_value"])):
        raise RuntimeError("packet cfg_value must be finite")
    return packet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REVISION:
        raise RuntimeError(f"official source HEAD mismatch: {head}")
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip().removesuffix(".git")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError(f"official source origin mismatch: {origin}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True):
        raise RuntimeError("official source checkout is dirty")
    missing = [role for role in SOURCE_ROLES if not (source / role).is_file()]
    if missing:
        raise RuntimeError(f"official source roles missing: {missing}")
    model_source = (source / SOURCE_ROLES[0]).read_text(encoding="utf-8")
    try:
        tree = ast.parse(model_source, filename=SOURCE_ROLES[0])
    except SyntaxError as error:
        raise RuntimeError(f"official VoxCPM model source is not parseable: {error}") from error
    generate_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_generate"]
    if len(generate_nodes) != 1:
        raise RuntimeError("fixed source must contain exactly one _generate function")
    generate_args = [arg.arg for arg in generate_nodes[0].args.args]
    public_generate_args = generate_args[1:] if generate_args and generate_args[0] == "self" else generate_args
    if public_generate_args[: len(OFFICIAL_GENERATE_ARGUMENTS)] != list(OFFICIAL_GENERATE_ARGUMENTS):
        raise RuntimeError(f"fixed _generate signature drift: {generate_args}")
    cfm_source = (source / SOURCE_ROLES[1]).read_text(encoding="utf-8")
    for marker in ("torch.randn", "self.in_channels", "n_timesteps"):
        if marker not in cfm_source:
            raise RuntimeError(f"fixed UnifiedCFM source lacks random-draw marker: {marker}")
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "roles": {role: sha256_file(source / role) for role in SOURCE_ROLES},
        "_generate_arguments": generate_args,
        "random_draw_source": "torch.randn((b, self.in_channels, t), ...)",
    }


def write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    pcm = bytearray()
    for sample in samples:
        pcm.extend(int(max(-1.0, min(1.0, sample)) * 32767.0).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(pcm))


def tensor_record(output: Path, name: str, value: Any) -> dict[str, Any]:
    import torch  # type: ignore
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"tap {name} returned {type(value).__name__}, not Tensor")
    value = value.detach().to("cpu").contiguous()
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"tap {name} returned non-finite values")
    # NumPy has no bfloat16 dtype on several VAST images.  Preserve the
    # original learned output bytes rather than silently converting it to
    # float32 for the evidence hash.
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    (output / f"{name}.bin").write_bytes(raw)
    return {"name": name, "shape": [int(axis) for axis in value.shape], "dtype": str(value.dtype), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def record_tensor_value(output: Path, name: str, value: Any) -> list[dict[str, Any]]:
    """Record tensors returned by a real source call without inventing a tap."""
    if value is None:
        return []
    if isinstance(value, tuple | list):
        records: list[dict[str, Any]] = []
        for index, child in enumerate(value):
            records.extend(record_tensor_value(output, f"{name}_{index:02d}", child))
        return records
    return [tensor_record(output, name, value)]


def run_official(source: Path, snapshot: Path, packet: dict[str, Any], output: Path) -> dict[str, Any]:
    source_record = source_identity(source)
    sys.path.insert(0, str(source / "src"))
    try:
        from voxcpm.model.voxcpm import VoxCPMModel  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"cannot import pinned official VoxCPMModel: {error}") from error
    constructor = getattr(VoxCPMModel, "from_local", None)
    if not callable(constructor):
        raise RuntimeError("pinned source lacks VoxCPMModel.from_local")
    model = constructor(str(snapshot), optimize=False, training=False, device="cpu")
    generate = getattr(model, "generate", None)
    tokenizer = getattr(model, "text_tokenizer", None)
    if not callable(generate) or not callable(tokenizer):
        raise RuntimeError("official model lacks generate/text_tokenizer seams")

    token_calls: list[list[int]] = []
    def capture_tokenizer(text: str) -> list[int]:
        values = tokenizer(text)
        if hasattr(values, "tolist"):
            values = values.tolist()
        values = [int(value) for value in values]
        token_calls.append(values)
        return values
    model.text_tokenizer = capture_tokenizer

    taps: list[dict[str, Any]] = []
    handles = []
    feature_module = getattr(model, "feat_decoder", None)
    if feature_module is None or not hasattr(feature_module, "register_forward_hook"):
        raise RuntimeError("official generated-feature tap module is unavailable")
    feature_calls = 0
    def feature_hook(_module: Any, _inputs: Any, result: Any) -> None:
        nonlocal feature_calls
        taps.extend(record_tensor_value(output, f"generated_features_{feature_calls:04d}", result))
        feature_calls += 1
    handles.append(feature_module.register_forward_hook(feature_hook))

    audio_vae = getattr(model, "audio_vae", None)
    encode = getattr(audio_vae, "encode", None)
    decode = getattr(audio_vae, "decode", None)
    if not callable(encode) or not callable(decode):
        raise RuntimeError("official AudioVAE encode/decode methods are unavailable")
    encode_calls = 0
    decode_calls = 0
    def capture_encode(*args: Any, **kwargs: Any) -> Any:
        nonlocal encode_calls
        result = encode(*args, **kwargs)
        taps.extend(record_tensor_value(output, "prompt_latent", result))
        encode_calls += 1
        return result
    def capture_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal decode_calls
        result = decode(*args, **kwargs)
        taps.extend(record_tensor_value(output, f"decoded_pcm_{decode_calls:04d}", result))
        decode_calls += 1
        return result
    setattr(audio_vae, "encode", capture_encode)
    setattr(audio_vae, "decode", capture_decode)

    import torch  # type: ignore
    original_randn = torch.randn
    draw_calls = 0
    draw_dtype: str | None = None
    draw_device: str | None = None
    def guarded_randn(*args: Any, **kwargs: Any) -> Any:
        nonlocal draw_calls, draw_dtype, draw_device
        draw_calls += 1
        if draw_calls != 1:
            raise RuntimeError(f"official UnifiedCFM requested unexpected randn call {draw_calls}")
        shape = args[0] if args else kwargs.get("size")
        if tuple(shape) != OFFICIAL_RANDOM_DRAW_SHAPE:
            raise RuntimeError(f"official UnifiedCFM draw shape drift: {shape}")
        dtype = kwargs.get("dtype", torch.float32)
        device = kwargs.get("device", "cpu")
        draw_dtype = str(dtype)
        draw_device = str(device)
        if draw_dtype != packet["noise_dtype"] or draw_device != packet["noise_device"]:
            raise RuntimeError(f"official UnifiedCFM draw type/device drift: {draw_dtype}/{draw_device}")
        dtype_by_name = {"torch.float32": torch.float32, "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16}
        return torch.tensor(packet["noise"], dtype=dtype_by_name[packet["noise_dtype"]], device=device).reshape(OFFICIAL_RANDOM_DRAW_SHAPE)

    wav_path: Path | None = None
    try:
        if packet["prompt_pcm"]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
            write_wav(wav_path, packet["prompt_pcm"], packet["sample_rate"])
        torch.randn = guarded_randn
        # ``generate`` forwards keyword arguments to the pinned ``_generate``
        # implementation, whose first parameter is ``target_text``.  Do not
        # use the tempting ``text=`` spelling: it is not an upstream keyword.
        result = generate(target_text=packet["text"], prompt_text=packet["prompt_text"], prompt_wav_path=str(wav_path) if wav_path else "", min_len=packet["min_len"], max_len=packet["max_len"], inference_timesteps=packet["inference_timesteps"], cfg_value=float(packet["cfg_value"]), seed=packet["seed"])
    finally:
        torch.randn = original_randn
        for handle in handles:
            handle.remove()
        setattr(audio_vae, "encode", encode)
        setattr(audio_vae, "decode", decode)
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    if draw_calls != 1:
        raise RuntimeError(f"official UnifiedCFM draw was not consumed exactly once: {draw_calls}")
    if token_calls != packet["token_ids"]:
        raise RuntimeError(f"official tokenizer output mismatch: expected {packet['token_ids']!r}, got {token_calls!r}")
    expected_encode_calls = 1 if packet["prompt_pcm"] else 0
    if encode_calls != expected_encode_calls:
        raise RuntimeError(f"official AudioVAE encode call count drift: expected {expected_encode_calls}, got {encode_calls}")
    if decode_calls != 1:
        raise RuntimeError(f"official AudioVAE decode call count drift: expected 1, got {decode_calls}")
    if feature_calls != 1:
        raise RuntimeError(f"official feature-generation call count drift: expected 1, got {feature_calls}")
    taps.append(tensor_record(output, "final_pcm", result))
    return {
        "source": source_record,
        "official_generate_arguments": list(OFFICIAL_GENERATE_ARGUMENTS),
        "random_draw": {"function": "torch.randn", "calls": draw_calls, "shape": list(OFFICIAL_RANDOM_DRAW_SHAPE), "dtype": draw_dtype, "device": draw_device},
        "tokenizer_calls": token_calls,
        "draw_calls": draw_calls,
        "taps": taps,
    }


def self_test() -> None:
    assert strict_pairs([("x", 1)]) == {"x": 1}
    try:
        strict_pairs([("x", 1), ("x", 2)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate packet keys must fail")
    assert "src/voxcpm/model/voxcpm.py" in SOURCE_ROLES
    assert "src/voxcpm/modules/locdit/unified_cfm.py" in SOURCE_ROLES
    assert OFFICIAL_GENERATE_ARGUMENTS[0] == "target_text"
    assert "token_ids" not in OFFICIAL_GENERATE_ARGUMENTS
    assert "prompt_pcm" not in OFFICIAL_GENERATE_ARGUMENTS
    assert "noise" not in OFFICIAL_GENERATE_ARGUMENTS
    assert OFFICIAL_RANDOM_DRAW_SHAPE == (1, 64, 2)
    packet = {
        "text": "hello", "prompt_text": "", "prompt_pcm": [], "sample_rate": 16_000,
        "token_ids": [[1], [2]], "noise": [0.0] * 128, "noise_dtype": "torch.float32", "noise_device": "cpu", "min_len": 0,
        "max_len": 1, "inference_timesteps": 1, "cfg_value": 2.0, "seed": 0,
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as handle:
        json.dump(packet, handle)
        packet_path = Path(handle.name)
    try:
        assert load_packet(packet_path)["text"] == "hello"
    finally:
        packet_path.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as handle:
        handle.write('{"text":"x","text":"y"}')
        duplicate_path = Path(handle.name)
    try:
        try:
            load_packet(duplicate_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate packet JSON keys must fail")
    finally:
        duplicate_path.unlink(missing_ok=True)
    print("voxcpm_0_5b_reference --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.source, args.snapshot, args.packet, args.output)):
        parser.error("--source, --snapshot, --packet and --output are required")
    try:
        packet = load_packet(args.packet)
        args.output.mkdir(parents=True, exist_ok=True)
        result = run_official(args.source, args.snapshot, packet, args.output)
        manifest = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "reference_status": "REFERENCE_EVIDENCE_COMPLETE", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "MEASURED_NOT_GATED", "publication": "NO_UPLOAD", "repository": HF_REPOSITORY, "revision": HF_REVISION, "packet_sha256": hashlib.sha256(args.packet.read_bytes()).hexdigest(), **result}
        (args.output / "packet.json").write_bytes(args.packet.read_bytes())
        (args.output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except Exception as error:  # noqa: BLE001
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "reference_status": "REFERENCE_ERROR", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "repository": HF_REPOSITORY, "revision": HF_REVISION, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "error": f"{type(error).__name__}: {error}"}
        (args.output / "manifest.json").write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"voxcpm_0_5b_reference: BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
