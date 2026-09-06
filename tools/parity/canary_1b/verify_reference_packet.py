#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Fail-closed verifier for the eight-file Canary official-reference packet.

This verifier is intentionally independent of NeMo.  It authenticates the
small packet produced by the official dumper before it can be consumed by an
Apple or VAST parity runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

DATA_FILES = tuple(
    f"reference-{case}.{suffix}"
    for case in ("en-en", "en-de")
    for suffix in ("json", "pcm.f32", "tokens.txt", "text.txt")
)
MANIFEST_NAME = "reference-manifest.sha256"
PACKET_DIGEST_NAME = "reference-packet.sha256"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_report(path: Path, variant: str, revision: str, checkpoint: str, audio_sha: str) -> None:
    case = path.name.removeprefix("reference-").removesuffix(".json")
    source, target = case.split("-", 1)
    expected_format = (
        "vokra-canary-1b-flash-nemo-reference-v1"
        if variant == "flash"
        else "vokra-canary-1b-v2-nemo-reference-v1"
    )
    report = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
    )
    if not isinstance(report, dict):
        raise ValueError(f"{path.name}: JSON root must be an object")
    expected_keys = {
        "format", "reference_implementation", "reference_package",
        "reference_source_audit_commit", "nemo_version", "torch_version",
        "environment", "upstream_hf", "upstream_revision", "checkpoint_sha256",
        "audio", "audio_sha256", "sample_rate", "sample_count",
        "source_language", "target_language", "taskname", "text", "tokens",
    }
    if set(report) != expected_keys:
        raise ValueError(f"{path.name}: JSON key closure changed")
    if report["format"] != expected_format:
        raise ValueError(f"{path.name}: unexpected report format")
    if (
        report["reference_implementation"]
        != "nemo.collections.asr.models.EncDecMultiTaskModel.restore_from"
        or report["reference_package"] != "nemo-toolkit[asr]==3.0.0"
        or report["reference_source_audit_commit"]
        != "837a31fa7a810a3de9e4826837e97dea837a5c42"
    ):
        raise ValueError(f"{path.name}: reference provenance drift")
    expected_hf = "nvidia/canary-1b-flash" if variant == "flash" else "nvidia/canary-1b-v2"
    if report["upstream_hf"] != expected_hf or report["upstream_revision"] != revision:
        raise ValueError(f"{path.name}: upstream identity drift")
    if report["checkpoint_sha256"] != checkpoint or report["audio_sha256"] != audio_sha:
        raise ValueError(f"{path.name}: checkpoint/audio identity drift")
    if report["audio"] != "tests/fixtures/audio/jfk-30s.wav":
        raise ValueError(f"{path.name}: audio path drift")
    if report["source_language"] != source or report["target_language"] != target:
        raise ValueError(f"{path.name}: language identity drift")
    if report["taskname"] != ("asr" if source == target else "ast"):
        raise ValueError(f"{path.name}: task identity drift")
    tokens = report["tokens"]
    text = report["text"]
    if not isinstance(tokens, list) or not tokens or any(
        type(token) is not int or token < 0 for token in tokens
    ):
        raise ValueError(f"{path.name}: invalid or empty token sequence")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{path.name}: text fixture must be nonempty")
    if (
        report["sample_rate"] != 16_000
        or type(report["sample_count"]) is not int
        or report["sample_count"] <= 0
    ):
        raise ValueError(f"{path.name}: audio metadata is invalid")
    stem = path.with_suffix("")
    token_sidecar = stem.with_suffix(".tokens.txt")
    text_sidecar = stem.with_suffix(".text.txt")
    pcm_sidecar = stem.with_suffix(".pcm.f32")
    sidecar_tokens = [int(value) for value in token_sidecar.read_text().split()]
    if sidecar_tokens != tokens or text_sidecar.read_text() != text + "\n":
        raise ValueError(f"{path.name}: sidecars do not match JSON")
    pcm = pcm_sidecar.read_bytes()
    if not pcm or len(pcm) % 4 or len(pcm) // 4 != report["sample_count"]:
        raise ValueError(f"{path.name}: PCM sidecar does not match sample_count")
    # Ensure the packet really contains little-endian f32 values, rather than
    # merely a byte count that happens to be divisible by four.
    values = [
        struct.unpack("<f", pcm[offset : offset + 4])[0]
        for offset in range(0, len(pcm), 4)
    ]
    if any(not math.isfinite(value) for value in values) or not any(
        value != 0.0 for value in values
    ):
        raise ValueError(f"{path.name}: PCM must contain finite, nonzero JFK samples")


def verify_packet(args: argparse.Namespace) -> None:
    directory = args.directory
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("reference directory is missing or symlinked")
    expected_names = set(DATA_FILES) | {MANIFEST_NAME, PACKET_DIGEST_NAME}
    entries = list(directory.iterdir())
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise ValueError("reference packet has missing, duplicate, symlinked, or extra entries")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"reference packet entry is not a regular file: {entry.name}")
    if not HEX64.fullmatch(args.manifest_sha256) or not HEX64.fullmatch(args.packet_sha256):
        raise ValueError("expected packet hashes must be lowercase SHA-256")
    if digest_file(directory / MANIFEST_NAME) != args.manifest_sha256:
        raise ValueError("reference manifest digest mismatch")
    lines = (directory / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(DATA_FILES):
        raise ValueError("reference manifest does not have exact data-file closure")
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) not in DATA_FILES or match.group(2) in seen:
            raise ValueError("reference manifest has malformed/extra/duplicate entry")
        name = match.group(2)
        if digest_file(directory / name) != match.group(1):
            raise ValueError(f"reference manifest hash mismatch: {name}")
        seen.add(name)
    if seen != set(DATA_FILES):
        raise ValueError("reference manifest is missing a data file")
    packet = b"".join((directory / name).read_bytes() for name in DATA_FILES)
    packet_digest_file = (directory / PACKET_DIGEST_NAME).read_text(encoding="utf-8")
    if packet_digest_file != args.packet_sha256 + "\n" or digest_bytes(packet) != args.packet_sha256:
        raise ValueError("reference packet digest mismatch")
    for name in ("reference-en-en.json", "reference-en-de.json"):
        parse_report(directory / name, args.variant, args.revision, args.checkpoint_sha256, args.audio_sha256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--variant", choices=("flash", "v2"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--packet-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character lowercase commit SHA")
    for name, value in (("checkpoint", args.checkpoint_sha256), ("audio", args.audio_sha256)):
        if not HEX64.fullmatch(value):
            parser.error(f"--{name}-sha256 must be lowercase SHA-256")
    try:
        verify_packet(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"reference packet verification failed: {error}", file=sys.stderr)
        return 1
    print("reference packet: exact authenticated closure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
