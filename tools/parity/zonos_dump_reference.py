#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Run the pinned upstream Zonos implementation for a typed packet.

This is an independent reference runner, not a Rust mirror.  It loads the
official ``Zonos`` implementation from the exact source checkout and the
fixed HF transformer snapshot, then asks the upstream generation method for a
deterministic greedy code sequence.  The packet contains the offline eSpeak
symbol IDs and raw conditioner controls; its projected-prefix compatibility
fields are parsed for integrity but are deliberately never consumed.

The official model also owns the DAC companion.  ``--pcm-output`` therefore
requires an already available official DAC cache; there is no fallback codec,
zero-fill output, or locally invented decoder.  A successful reference run is
``MEASURED_NOT_GATED`` until an independently reviewed Rust CPU/Metal bound is
registered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import struct
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION = "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
UPSTREAM_REPOSITORY = "Zyphra/Zonos-v0.1-transformer"
UPSTREAM_REVISION = "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e"
PACKET_MAGIC = b"ZONOSCP1"
PACKET_VERSION = 1
CODEBOOKS = 9
MASKED = 1025
MAX_PHONEMES = 1 << 20
MAX_PREFIX_VALUES = 1 << 24


def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"path contains a symlink ancestor: {path}")


def require_regular_input(path: Path, label: str) -> None:
    reject_symlink_ancestors(path)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} must be a regular non-symlink file")


def require_absent_output(path: Path, label: str) -> None:
    reject_symlink_ancestors(path)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{label} already exists or is symlinked")
    parent = path.parent
    while not parent.exists():
        if parent == parent.parent:
            raise RuntimeError(f"{label} has no existing parent")
        parent = parent.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(f"{label} parent is not a real directory")


def fixed_source(source: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("Zonos source root must be a real directory")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if git("rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("Zonos source HEAD is not the fixed official commit")
    if git("remote", "get-url", "origin") != SOURCE_REPOSITORY:
        raise RuntimeError("Zonos source origin is not the fixed official repository")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Zonos source checkout is dirty")
    for role in ("zonos/model.py", "zonos/conditioning.py", "zonos/codebook_pattern.py"):
        role_path = source / role
        if not role_path.is_file() or role_path.is_symlink():
            raise RuntimeError(f"official Zonos source role is missing: {role}")


def take(data: bytes, cursor: list[int], count: int) -> bytes:
    if count < 0 or cursor[0] > len(data) - count:
        raise ValueError("conditioning packet is truncated or overflows")
    start = cursor[0]
    cursor[0] += count
    return data[start : cursor[0]]


def u32(data: bytes, cursor: list[int]) -> int:
    return struct.unpack("<I", take(data, cursor, 4))[0]


def i32(data: bytes, cursor: list[int]) -> int:
    return struct.unpack("<i", take(data, cursor, 4))[0]


def f32(data: bytes, cursor: list[int]) -> float:
    value = struct.unpack("<f", take(data, cursor, 4))[0]
    if not math.isfinite(value):
        raise ValueError("conditioning packet contains a non-finite scalar")
    return value


def parse_packet(path: Path) -> dict[str, Any]:
    """Parse the v1 packet and authenticate its content digest.

    Layout intentionally mirrors only the published wire contract.  The
    conditional/unconditional projected arrays are retained in the digest but
    are not returned to the reference model, preventing prefix bypass.
    """
    require_regular_input(path, "conditioning packet")
    data = path.read_bytes()
    cursor = [0]
    if take(data, cursor, len(PACKET_MAGIC)) != PACKET_MAGIC:
        raise ValueError("conditioning packet magic mismatch")
    if u32(data, cursor) != PACKET_VERSION:
        raise ValueError("unsupported conditioning packet version")
    phoneme_count = u32(data, cursor)
    if not 2 <= phoneme_count <= MAX_PHONEMES:
        raise ValueError("phoneme count is outside the bounded packet contract")
    speaker = [f32(data, cursor) for _ in range(128)]
    emotion = [f32(data, cursor) for _ in range(8)]
    fmax, pitch_std, speaking_rate = (f32(data, cursor) for _ in range(3))
    language_id = i32(data, cursor)
    codebook_count = u32(data, cursor)
    prompt_frames = u32(data, cursor)
    if codebook_count != CODEBOOKS or prompt_frames > MAX_PHONEMES:
        raise ValueError("prompt code shape is not the fixed nine-codebook contract")
    conditional_count = u32(data, cursor)
    unconditional_count = u32(data, cursor)
    if (
        conditional_count == 0
        or conditional_count != unconditional_count
        or conditional_count > MAX_PREFIX_VALUES
        or conditional_count % 2048
    ):
        raise ValueError("projected-prefix compatibility shape is invalid")
    digest_offset = cursor[0]
    digest = take(data, cursor, 32)
    computed = hashlib.sha256(data[:digest_offset] + data[digest_offset + 32 :]).digest()
    if digest != computed:
        raise ValueError("conditioning packet content digest mismatch")
    phoneme_ids = [u32(data, cursor) for _ in range(phoneme_count)]
    if phoneme_ids[0] != 2 or phoneme_ids[-1] != 3 or any(
        token == 0 or token in (2, 3) or token >= 189 for token in phoneme_ids[1:-1]
    ):
        raise ValueError("phoneme packet is not BOS/UNK/symbol/EOS framed")
    prompt = [
        [u32(data, cursor) for _ in range(prompt_frames)] for _ in range(codebook_count)
    ]
    projected_count = conditional_count + unconditional_count
    projected = [f32(data, cursor) for _ in range(projected_count)]
    if cursor[0] != len(data) or not all(math.isfinite(value) for value in projected):
        raise ValueError("conditioning packet has trailing bytes or non-finite prefix values")
    if not 0 <= fmax <= 24_000 or not 0 <= pitch_std <= 400 or not 0 <= speaking_rate <= 40:
        raise ValueError("conditioner scalar is outside the fixed source domain")
    if not -1 <= language_id <= 126 or any(not 0 <= value <= 1 for value in emotion):
        raise ValueError("conditioner control is outside the fixed source domain")
    total = sum(emotion)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise ValueError("emotion must be the normalized source vector")
    if any(token > MASKED for row in prompt for token in row):
        raise ValueError("prompt code is outside the source codebook vocabulary")
    return {
        "digest": digest.hex(),
        "phoneme_ids": phoneme_ids,
        "speaker": speaker,
        "emotion": emotion,
        "fmax": fmax,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "language_id": language_id,
        "prompt_codes": prompt,
    }


def official_prefix(model: Any, packet: dict[str, Any], torch: Any) -> Any:
    """Use the official conditioner modules on raw packet fields."""
    prefix = model.prefix_conditioner
    device = model.device
    dtype = next(model.parameters()).dtype
    ids = torch.tensor(packet["phoneme_ids"], dtype=torch.long, device=device).view(1, -1)
    phoneme = prefix.conditioners[0].phoneme_embedder(ids).to(dtype)
    controls = [
        torch.tensor(packet["speaker"], dtype=dtype, device=device).view(1, 1, 128),
        torch.tensor(packet["emotion"], dtype=dtype, device=device).view(1, 1, 8),
        torch.tensor([[packet["fmax"]]], dtype=dtype, device=device),
        torch.tensor([[packet["pitch_std"]]], dtype=dtype, device=device),
        torch.tensor([[packet["speaking_rate"]]], dtype=dtype, device=device),
        torch.tensor([[packet["language_id"] + 1]], dtype=torch.long, device=device),
    ]
    conditional = [phoneme]
    unconditional = [phoneme]
    for conditioner, value in zip(prefix.conditioners[1:], controls):
        conditional.append(conditioner((value,)).to(dtype))
        unconditional.append(conditioner(None).to(dtype))
    cond = torch.cat(conditional, dim=-2)
    uncond = torch.cat(unconditional, dim=-2)
    return torch.cat((prefix.norm(prefix.project(cond)), prefix.norm(prefix.project(uncond))))


def run_reference(source: Path, snapshot: Path, packet_path: Path, output: Path, max_new_tokens: int, cfg_scale: float, pcm_output: Path | None) -> None:
    if pcm_output is None:
        raise RuntimeError("--pcm-output is required for the authenticated Zonos evidence record")
    require_absent_output(output, "codes output")
    require_absent_output(pcm_output, "PCM output")
    record_path = output.with_suffix(".json")
    require_absent_output(record_path, "reference record")
    if output.resolve(strict=False) == pcm_output.resolve(strict=False) or record_path.resolve(strict=False) in {output.resolve(strict=False), pcm_output.resolve(strict=False)}:
        raise RuntimeError("codes, PCM, and reference record outputs must be distinct")
    output_parent = output.parent.resolve()
    pcm_parent = pcm_output.parent.resolve()
    if output_parent != pcm_parent or record_path.parent.resolve() != output_parent:
        raise RuntimeError("codes, PCM, and reference record must share one evidence directory")
    fixed_source(source)
    packet = parse_packet(packet_path)
    config = snapshot / "config.json"
    weights = snapshot / "model.safetensors"
    require_regular_input(config, "upstream config")
    require_regular_input(weights, "upstream model weights")
    if not config.is_file() or not weights.is_file():
        raise RuntimeError("fixed upstream snapshot must contain config.json and model.safetensors")
    sys.path.insert(0, str(source))
    try:
        # The fixed source's DACAutoencoder normally resolves its own HF
        # companion.  Never let a reference run silently fetch a mutable
        # companion; VAST must pre-stage the reviewed DAC cache instead.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from zonos.model import Zonos

        # The official constructor/load path is used verbatim.  It may require
        # the already-authenticated DAC companion cache; no alternate loader
        # is permitted here.
        model = Zonos.from_local(str(config), str(weights), device="cpu")
        model.eval()
        prefix = official_prefix(model, packet, torch)
        import numpy as np

        prompt = torch.tensor(packet["prompt_codes"], dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            codes = model.generate(
                prefix,
                audio_prefix_codes=prompt if prompt.shape[-1] else None,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                sampling_params={"temperature": 0.0},
                progress_bar=False,
                disable_torch_compile=True,
            )
            pcm = model.autoencoder.decode(codes).detach().float().cpu().reshape(-1)
            if pcm.numel() == 0 or not bool(torch.isfinite(pcm).all()):
                raise RuntimeError("official DAC returned empty/non-finite PCM")
            pcm_output.parent.mkdir(parents=True, exist_ok=True)
            pcm_values = pcm.numpy().astype("<f4", copy=False)
            if pcm_output.suffix == ".f32le":
                pcm_output.write_bytes(pcm_values.tobytes(order="C"))
            else:
                np.save(pcm_output, pcm_values)
            codes_cpu = codes.detach().to("cpu")
        if codes_cpu.numel() == 0 or not bool(torch.isfinite(codes_cpu.float()).all()):
            raise RuntimeError("official generation returned empty/non-finite codes")
        output.parent.mkdir(parents=True, exist_ok=True)
        np_codes = codes_cpu.numpy().astype("int64", copy=False)
        if output.suffix == ".u32le":
            if (np_codes < 0).any() or (np_codes > MASKED).any():
                raise RuntimeError("official generation returned an out-of-range code")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(np_codes.astype("<u4", copy=False).tobytes(order="C"))
        else:
            np.save(output, np_codes)
        record = {
            "format": "vokra-zonos-reference-v1",
            "reference_status": "MEASURED_NOT_GATED",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "conditioning_packet_sha256": sha256_bytes(packet_path.read_bytes()),
            "conditioning_packet_content_digest": packet["digest"],
            # The record is consumed from its sibling evidence directory;
            # relative names let the inspector bind the file to that root.
            "codes_path": output.name,
            "codes_shape": list(np_codes.shape),
            "codes_sha256": sha256_bytes(output.read_bytes()),
            "pcm_path": pcm_output.name,
            "pcm_sha256": sha256_bytes(pcm_output.read_bytes()),
            "pcm_sample_rate": 44_100,
            "runtime_status": "REFERENCE_ONLY_NO_NATIVE_VERDICT",
            "publication": "NO_UPLOAD",
        }
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        sys.path.remove(str(source))


def self_test() -> None:
    assert len(SOURCE_REVISION) == len(UPSTREAM_REVISION) == 40
    assert PACKET_MAGIC == b"ZONOSCP1" and CODEBOOKS == 9 and MASKED == 1025
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=no_dupes)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys must fail")
    # A minimal packet cannot be accepted by accident: authentication and the
    # complete typed fields are mandatory before any official import.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        directory = str(Path(directory).resolve())
        path = Path(directory) / "empty.packet"
        path.write_bytes(b"")
        try:
            parse_packet(path)
        except ValueError:
            pass
        else:
            raise AssertionError("empty packet must fail closed")
        # Positive wire-contract fixture: it uses raw typed controls and a
        # minimal two-symbol phoneme sequence, while the projected arrays are
        # present only because v1 reserves them in the authenticated packet.
        body = bytearray(PACKET_MAGIC)
        body.extend(struct.pack("<II", PACKET_VERSION, 2))
        body.extend(struct.pack("<f", 0.0) * 128)
        body.extend(struct.pack("<f", 1.0))
        body.extend(struct.pack("<f", 0.0) * 7)
        body.extend(struct.pack("<fff", 22_050.0, 20.0, 15.0))
        body.extend(struct.pack("<iIII", 0, CODEBOOKS, 0, 2048))
        body.extend(struct.pack("<I", 2048))
        digest_offset = len(body)
        body.extend(b"\0" * 32)
        body.extend(struct.pack("<II", 2, 3))
        body.extend(struct.pack("<f", 0.0) * 4096)
        body[digest_offset : digest_offset + 32] = hashlib.sha256(
            body[:digest_offset] + body[digest_offset + 32 :]
        ).digest()
        valid = Path(directory) / "valid.packet"
        valid.write_bytes(body)
        parsed = parse_packet(valid)
        assert parsed["phoneme_ids"] == [2, 3]
        body[-1] ^= 1
        valid.write_bytes(body)
        try:
            parse_packet(valid)
        except ValueError:
            pass
        else:
            raise AssertionError("packet content mutation must fail digest authentication")
        output_root = Path(directory) / "outputs"
        output_root.mkdir()
        require_absent_output(output_root / "codes.u32le", "self-test output")
        (output_root / "existing.u32le").write_bytes(b"x")
        try:
            require_absent_output(output_root / "existing.u32le", "self-test output")
        except RuntimeError:
            pass
        else:
            raise AssertionError("existing output must not be overwritten")
        (output_root / "real").mkdir()
        (output_root / "link").symlink_to(output_root / "real", target_is_directory=True)
        try:
            require_absent_output(output_root / "link" / "new.u32le", "self-test output")
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlink output ancestors must fail closed")
    print("zonos_dump_reference.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--upstream-snapshot", type=Path)
    parser.add_argument("--conditioning-packet", type=Path)
    parser.add_argument("--codes-output", type=Path)
    parser.add_argument("--pcm-output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.source, args.upstream_snapshot, args.conditioning_packet, args.codes_output, args.pcm_output)
    if any(item is None for item in required):
        parser.error("--source, --upstream-snapshot, --conditioning-packet, --codes-output, and --pcm-output are required")
    if args.max_new_tokens <= 0 or not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        parser.error("generation controls must be positive and finite")
    try:
        run_reference(args.source, args.upstream_snapshot, args.conditioning_packet, args.codes_output, args.max_new_tokens, args.cfg_scale, args.pcm_output)
    except Exception as error:
        print(f"zonos reference blocker: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
