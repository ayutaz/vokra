#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Authenticate the independent SpeechBrain ECAPA reference packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

DATA = ("pcm.f32.bin", "features.f32.bin", "embedding.f32.bin", "manifest.json")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def self_test() -> int:
    try:
        unique([("one", 1)])
        try:
            unique([("one", 1), ("one", 2)])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON key was accepted")
        if not HEX64.fullmatch("a" * 64):
            raise AssertionError("SHA-256 validator rejected a valid digest")
        if HEX64.fullmatch("A" * 64):
            raise AssertionError("SHA-256 validator accepted uppercase input")
    except AssertionError as error:
        print(f"ECAPA reference verifier self-test failed: {error}", file=sys.stderr)
        return 1
    print("ECAPA reference verifier self-test: PASS")
    return 0


def require_finite(path: Path, *, nonzero: bool = False) -> None:
    values = [value[0] for value in struct.iter_unpack("<f", path.read_bytes())]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite or empty float payload: {path.name}")
    if nonzero and not any(value != 0.0 for value in values):
        raise ValueError(f"all-zero float payload: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--wav-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--packet-sha256")
    parser.add_argument("--pcm-sha256")
    parser.add_argument("--features-sha256")
    parser.add_argument("--embedding-sha256")
    parser.add_argument("--manifest-json-sha256")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    required = {
        "directory": args.directory,
        "revision": args.revision,
        "checkpoint-sha256": args.checkpoint_sha256,
        "wav-sha256": args.wav_sha256,
        "manifest-sha256": args.manifest_sha256,
        "packet-sha256": args.packet_sha256,
        "pcm-sha256": args.pcm_sha256,
        "features-sha256": args.features_sha256,
        "embedding-sha256": args.embedding_sha256,
        "manifest-json-sha256": args.manifest_json_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or any(
        not HEX64.fullmatch(value)
        for value in (
            args.checkpoint_sha256,
            args.wav_sha256,
            args.manifest_sha256,
            args.packet_sha256,
            args.pcm_sha256,
            args.features_sha256,
            args.embedding_sha256,
            args.manifest_json_sha256,
        )
    ):
        parser.error("identity arguments must be lowercase pinned hashes")
    try:
        directory = args.directory
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("reference directory missing or symlinked")
        names = set(DATA) | {"reference-manifest.sha256", "reference-packet.sha256"}
        entries = list(directory.iterdir())
        if {entry.name for entry in entries} != names or len(entries) != len(names):
            raise ValueError("reference packet has missing, duplicate, or extra entries")
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError("reference packet contains symlink/non-file")
        manifest = directory / "reference-manifest.sha256"
        if digest(manifest) != args.manifest_sha256:
            raise ValueError("reference manifest digest mismatch")
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(DATA):
            raise ValueError("reference manifest closure mismatch")
        seen: set[str] = set()
        for line in lines:
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if not match or match.group(2) not in DATA or match.group(2) in seen:
                raise ValueError("malformed/duplicate manifest entry")
            if digest(directory / match.group(2)) != match.group(1):
                raise ValueError(f"manifest hash mismatch: {match.group(2)}")
            seen.add(match.group(2))
        if seen != set(DATA):
            raise ValueError("reference manifest missing member")
        packet_sha = hashlib.sha256(b"".join((directory / name).read_bytes() for name in DATA)).hexdigest()
        if packet_sha != args.packet_sha256 or (directory / "reference-packet.sha256").read_text() != packet_sha + "\n":
            raise ValueError("reference packet digest mismatch")
        expected_member_hashes = {
            "pcm.f32.bin": args.pcm_sha256,
            "features.f32.bin": args.features_sha256,
            "embedding.f32.bin": args.embedding_sha256,
            "manifest.json": args.manifest_json_sha256,
        }
        for name, expected in expected_member_hashes.items():
            if digest(directory / name) != expected:
                raise ValueError(f"committed fixture hash mismatch: {name}")
        require_finite(directory / "pcm.f32.bin", nonzero=True)
        require_finite(directory / "features.f32.bin")
        require_finite(directory / "embedding.f32.bin")
        report = json.loads((directory / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=unique)
        if not isinstance(report, dict):
            raise ValueError("manifest JSON root must be object")
        expected_keys = {"format", "model_id", "revision", "source", "sample_rate", "pcm_samples", "raw_feature_shape", "feature_shape", "embedding_shape", "wav_sha256", "checkpoint_sha256", "python", "numpy", "torch", "torchaudio", "speechbrain"}
        if set(report) != expected_keys:
            raise ValueError("manifest JSON key closure mismatch")
        if (report["format"], report["model_id"], report["revision"], report["source"], report["sample_rate"], report["pcm_samples"], report["wav_sha256"], report["checkpoint_sha256"]) != ("vokra-ecapa-tdnn-reference-v1", "speechbrain/spkrec-ecapa-voxceleb", args.revision, "speechbrain/spkrec-ecapa-voxceleb", 16000, 52173, args.wav_sha256, args.checkpoint_sha256):
            raise ValueError("reference identity/metadata drift")
        if report["raw_feature_shape"] != [1, 327, 80] or report["feature_shape"] != [1, 327, 80] or report["embedding_shape"] != [1, 1, 192]:
            raise ValueError("reference shape metadata drift")
        expected_sizes = {"pcm.f32.bin": 52173 * 4, "features.f32.bin": 327 * 80 * 4, "embedding.f32.bin": 192 * 4}
        for name, size in expected_sizes.items():
            if (directory / name).stat().st_size != size:
                raise ValueError(f"reference size mismatch: {name}")
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"ECAPA reference packet verification failed: {error}", file=sys.stderr)
        return 1
    print("ECAPA reference packet: exact authenticated closure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
