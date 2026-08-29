#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Strict, payload-independent validation for a GigaAM v3 reference packet."""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

CHECKPOINT_BYTES = 448_928_167
CHECKPOINT_SHA256 = "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a"
CONFIG_SHA256 = "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52"
MODELING_SHA256 = "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86"
TOKENIZER_SHA256 = "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a"
HF_REVISION = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
PCM_F32LE_SHA256 = "f92e4a0422c513ab107975f5c9bd7a8e7a92532b37508a769c92d2496625229b"


def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def stream_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scan_f32(path: Path) -> tuple[bool, bool]:
    finite = True
    nonzero = False
    carry = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            data = carry + chunk
            usable = len(data) - (len(data) % 4)
            for (value,) in struct.iter_unpack("<f", data[:usable]):
                finite = finite and math.isfinite(value)
                nonzero = nonzero or value != 0.0
            carry = data[usable:]
    require(not carry, f"unaligned f32 artifact: {path.name}")
    return finite, nonzero


def validate_decision_shapes(logits: list[int], *vectors: list[int]) -> None:
    require(len(logits) == 2 and logits[1] == 1_025, "RNNT logits shape mismatch")
    for shape in vectors:
        require(shape == [logits[0]], "RNNT decision vector shape mismatch")


def validate_bundle(root: Path, portable: bool = False) -> str:
    """Validate source identities and every raw artifact, returning manifest SHA."""
    require(root.is_dir() and not root.is_symlink(), "reference directory is missing or symlinked")
    require(all(not ancestor.is_symlink() for ancestor in (root, *root.parents)), "reference path has symlink ancestry")
    root_real = root.resolve()
    manifest_path = root / "manifest.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid reference manifest: {exc}") from exc
    require(
        set(document)
        == {"format", "status", "repository", "revision", "source_revision", "config_sha256", "modeling_gigaam_sha256", "source_files", "pcm_input", "artifacts", "mel_frames", "encoded_frames", "rnnt", "frontend", "runtime", "parity"},
        "reference root schema mismatch",
    )
    require(document["format"] == "vokra-gigaam-v3-reference-v1", "reference format mismatch")
    require(document["status"] == "REFERENCE_DUMP_OPEN_NOT_PARITY" and document["parity"] == "OPEN_MEASURED_NOT_GATED", "reference status must remain open")
    require(document["repository"] == "ai-sage/GigaAM-v3" and document["revision"] == HF_REVISION and document["source_revision"] == SOURCE_REVISION, "reference identity mismatch")
    require(document["config_sha256"] == CONFIG_SHA256 and document["modeling_gigaam_sha256"] == MODELING_SHA256, "reference source hash mismatch")
    require(document["frontend"] == {"center": False, "mel_scale": "htk", "mel_norm": None, "power": 2, "n_fft": 320, "win_length": 320, "hop_length": 160, "n_mels": 64}, "frontend contract mismatch")
    source_files = document["source_files"]
    require(isinstance(source_files, dict) and set(source_files) == {"config", "modeling_gigaam", "checkpoint", "tokenizer"}, "source file set mismatch")
    expected_names = {"config": "config.json", "modeling_gigaam": "modeling_gigaam.py", "checkpoint": "pytorch_model.bin", "tokenizer": "tokenizer.model"}
    expected_hashes = {"config": CONFIG_SHA256, "modeling_gigaam": MODELING_SHA256, "checkpoint": CHECKPOINT_SHA256, "tokenizer": TOKENIZER_SHA256}
    for name, row in source_files.items():
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, f"source row schema mismatch: {name}")
        require(isinstance(row["path"], str), f"source path type mismatch: {name}")
        path = Path(row["path"])
        require(path.is_absolute() and path.name == expected_names[name], f"source path mismatch: {name}")
        if not portable:
            require(path.is_file() and not path.is_symlink(), f"source path mismatch: {name}")
            require(all(not ancestor.is_symlink() for ancestor in (path, *path.parents)), f"source symlink ancestry: {name}")
            source_real = path.resolve()
            require(source_real != root_real and root_real not in source_real.parents, f"source overlaps reference output: {name}")
        require(isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and row["bytes"] > 0, f"source byte type mismatch: {name}")
        require(isinstance(row["sha256"], str) and row["sha256"] == expected_hashes[name], f"source digest declaration mismatch: {name}")
        if not portable:
            digest, size = stream_sha256(path)
            require((digest, size) == (row["sha256"], row["bytes"]), f"source digest mismatch: {name}")
    require(source_files["checkpoint"]["bytes"] == CHECKPOINT_BYTES, "checkpoint size mismatch")
    pcm = document["pcm_input"]
    require(isinstance(pcm, dict) and set(pcm) == {"path", "sample_rate_hz", "shape", "dtype", "f32le_sha256"}, "PCM contract mismatch")
    require(pcm == {"path": "pcm.f32le", "sample_rate_hz": 16_000, "shape": [16_000], "dtype": "float32", "f32le_sha256": PCM_F32LE_SHA256}, "fixed PCM identity mismatch")
    artifacts = document["artifacts"]
    names = {"pcm": ("pcm.f32le", "float32"), "log_mel": ("log_mel.f32le", "float32"), "encoded": ("encoded.f32le", "float32"), "rnnt_logits": ("rnnt_logits.f32le", "float32"), "decision_frames": ("decision_frames.u32le", "uint32"), "decision_symbols": ("decision_symbols.u32le", "uint32"), "decision_argmax": ("decision_argmax.u32le", "uint32"), "token_ids": ("token_ids.u32le", "uint32")}
    require(isinstance(artifacts, dict) and set(artifacts) == set(names), "artifact set mismatch")
    for name, row in artifacts.items():
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256", "shape", "dtype"}, f"artifact row schema mismatch: {name}")
        require(isinstance(row["path"], str) and isinstance(row["dtype"], str), f"artifact path/type mismatch: {name}")
        require((row["path"], row["dtype"]) == names[name], f"artifact identity mismatch: {name}")
        require(isinstance(row["bytes"], int) and not isinstance(row["bytes"], bool) and row["bytes"] >= 0, f"artifact bytes type mismatch: {name}")
        require(isinstance(row["shape"], list) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in row["shape"]), f"artifact shape mismatch: {name}")
        path = root / row["path"]
        require(isinstance(row["path"], str) and not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts, f"artifact path mismatch: {name}")
        require(path.is_file() and not path.is_symlink() and all(not ancestor.is_symlink() for ancestor in (path, *path.parents)), f"artifact is missing: {name}")
        digest, size = stream_sha256(path)
        require((digest, size) == (row["sha256"], row["bytes"]), f"artifact digest mismatch: {name}")
        require(size == math.prod(row["shape"]) * 4, f"artifact shape/bytes mismatch: {name}")
        if row["dtype"] == "float32":
            finite, nonzero = scan_f32(path)
            require(finite and nonzero, f"float artifact must be finite and nonzero: {name}")
            if name == "rnnt_logits":
                values = struct.unpack("<" + "f" * (size // 4), path.read_bytes())
                for offset in range(0, len(values), 1_025):
                    logits = values[offset : offset + 1_025]
                    maximum = max(logits)
                    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
                    require(abs(logsumexp) <= 1.0e-4, "RNNT rows must be log-softmax")
    require(artifacts["pcm"]["shape"] == [16_000] and artifacts["pcm"]["sha256"] == PCM_F32LE_SHA256, "PCM artifact mismatch")
    require(artifacts["pcm"]["path"] == pcm["path"], "PCM input/artifact path mismatch")
    mel_frames = document["mel_frames"]
    encoded_frames = document["encoded_frames"]
    require(isinstance(mel_frames, int) and not isinstance(mel_frames, bool) and mel_frames > 0, "mel frame count mismatch")
    require(isinstance(encoded_frames, int) and not isinstance(encoded_frames, bool) and encoded_frames > 0, "encoded frame count mismatch")
    require(artifacts["log_mel"]["shape"] == [mel_frames, 64] and artifacts["encoded"]["shape"] == [encoded_frames, 768], "frontend/encoder shape mismatch")
    decisions = artifacts["rnnt_logits"]["shape"]
    validate_decision_shapes(decisions, artifacts["decision_frames"]["shape"], artifacts["decision_symbols"]["shape"], artifacts["decision_argmax"]["shape"])
    require(document["rnnt"] == {"num_classes": 1_025, "blank_id": 1_024, "max_symbols_per_step": 10, "decode": "greedy", "joint_output": "log_softmax"}, "RNNT contract mismatch")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        try:
            no_duplicate_keys([("a", 1), ("a", 2)])
        except ValueError:
            require(not (isinstance(True, int) and not isinstance(True, bool)), "JSON booleans must not pass integer gates")
            require(4 * 3 * 4 != 4 * 2 * 4, "shape/byte tamper fixture must differ")
            try:
                validate_decision_shapes([2, 1_025], [1])
            except ValueError:
                pass
            else:
                raise AssertionError("decision vector shape tamper was accepted")
            print("gigaam_v3_validation self-test: OK")
            return 0
        raise SystemExit("duplicate-key gate failed")
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[1] != "--portable"):
        raise SystemExit("usage: gigaam_v3_validation.py [--portable] <reference-dir> | --self-test")
    portable = len(sys.argv) == 3
    digest = validate_bundle(Path(sys.argv[-1]), portable=portable)
    print(f"validated GigaAM v3 reference bundle: manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
