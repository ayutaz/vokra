"""Strict, payload-independent validation for GigaAM reference artifacts.

The validator checks only the raw artifact files and manifest. It never loads a
model and keeps hashing streaming so it is safe to run before the parity leg.
"""
from __future__ import annotations

import hashlib
import copy
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_BYTES = 883170115
CHECKPOINT_SHA256 = "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728"
CONFIG_SHA256 = "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653"
MODELING_SHA256 = "6d02e640fbb5738ab11c030520a68654ef32f4ff363723db10534cf8b5d5c0e7"
PCM_SAMPLE_RATE_HZ = 16_000
PCM_SAMPLES = 16_000
PCM_F32LE_SHA256 = "f92e4a0422c513ab107975f5c9bd7a8e7a92532b37508a769c92d2496625229b"


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_reference_bundle(root: Path) -> str:
    """Validate a generated reference directory and return its manifest SHA."""
    manifest_path = root / "manifest.json"
    _require(root.is_dir() and not root.is_symlink(), "reference directory is missing or symlinked")
    root_real = root.resolve()
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid reference manifest: {exc}") from exc
    _require(
        set(document)
        == {
            "format",
            "status",
            "repository",
            "revision",
            "source_revision",
            "config_sha256",
            "modeling_gigaam_sha256",
            "source_files",
            "pcm_input",
            "artifacts",
            "encoded_length",
            "ctc",
            "runtime",
            "parity",
        },
        "reference manifest root schema mismatch",
    )
    _require(
        document["format"] == "vokra-gigaam-multilingual-reference-v1"
        and document["status"] == "REFERENCE_DUMP_OPEN_NOT_PARITY"
        and document["repository"] == "ai-sage/GigaAM-Multilingual"
        and document["revision"] == "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
        and document["source_revision"] == "7447938d791c4f3e643386ee22c33777004293a5"
        and document["config_sha256"] == CONFIG_SHA256
        and document["modeling_gigaam_sha256"] == MODELING_SHA256
        and document["parity"] == "OPEN_MEASURED_NOT_GATED",
        "reference identity/status mismatch",
    )
    _require(
        document["ctc"]
        == {
            "vocab_size": 71,
            "blank_id": 70,
            "collapse": "collapse_adjacent_repeat_then_remove_blank",
        },
        "reference CTC contract mismatch",
    )
    pcm_input = document["pcm_input"]
    _require(
        isinstance(pcm_input, dict)
        and set(pcm_input) == {"path", "npy_sha256", "f32le_sha256", "shape", "dtype", "sample_rate_hz"}
        and isinstance(pcm_input["path"], str)
        and pcm_input["path"].startswith("/")
        and isinstance(pcm_input["npy_sha256"], str)
        and len(pcm_input["npy_sha256"]) == 64
        and all(char in "0123456789abcdef" for char in pcm_input["npy_sha256"])
        and pcm_input["f32le_sha256"] == PCM_F32LE_SHA256
        and pcm_input["shape"] == [PCM_SAMPLES]
        and pcm_input["dtype"] == "float32"
        and pcm_input["sample_rate_hz"] == PCM_SAMPLE_RATE_HZ,
        "fixed PCM input contract mismatch",
    )
    source_files = document["source_files"]
    _require(
        isinstance(source_files, dict)
        and set(source_files) == {"config", "modeling_gigaam", "checkpoint"},
        "reference source file set mismatch",
    )
    source_paths: list[Path] = []
    for name, row in source_files.items():
        _require(
            isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
            f"reference source file row mismatch: {name}",
        )
        _require(
            isinstance(row["path"], str)
            and row["path"].startswith("/")
            and "\\" not in row["path"]
            and ".." not in Path(row["path"]).parts
            and isinstance(row["bytes"], int)
            and not isinstance(row["bytes"], bool)
            and row["bytes"] > 0
            and isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in row["sha256"]),
            f"reference source file row type mismatch: {name}",
        )
        expected_basename = {
            "config": "config.json",
            "modeling_gigaam": "modeling_gigaam.py",
            "checkpoint": "pytorch_model.bin",
        }[name]
        source_path = Path(row["path"])
        _require(
            source_path.name == expected_basename
            and source_path.is_file()
            and not source_path.is_symlink()
            and all(not ancestor.is_symlink() for ancestor in (source_path, *source_path.parents)),
            f"reference source file is missing, symlinked, or misnamed: {name}",
        )
        source_real = source_path.resolve()
        _require(
            source_real != root_real
            and root_real not in source_real.parents
            and source_real not in source_paths,
            f"reference source file overlaps output or another source: {name}",
        )
        source_paths.append(source_real)
        actual_sha, actual_bytes = stream_sha256(source_path)
        _require(
            actual_sha == row["sha256"] and actual_bytes == row["bytes"],
            f"reference source file digest/size mismatch: {name}",
        )
    _require(
        source_files["config"]["sha256"] == document["config_sha256"]
        and source_files["modeling_gigaam"]["sha256"] == document["modeling_gigaam_sha256"]
        and source_files["checkpoint"]["sha256"] == CHECKPOINT_SHA256
        and source_files["checkpoint"]["bytes"] == CHECKPOINT_BYTES,
        "reference source digest/size mismatch",
    )
    artifacts = document["artifacts"]
    expected = {
        "pcm": ("pcm.f32le", "float32"),
        "encoded": ("encoded.f32le", "float32"),
        "logits": ("logits.f32le", "float32"),
        "raw_argmax": ("raw_argmax.u32le", "uint32"),
        "token_ids": ("token_ids.u32le", "uint32"),
    }
    _require(isinstance(artifacts, dict) and set(artifacts) == set(expected), "reference artifact set mismatch")
    for name, row in artifacts.items():
        _require(
            isinstance(row, dict) and set(row) == {"path", "bytes", "sha256", "shape", "dtype"},
            f"reference artifact schema mismatch: {name}",
        )
        _require(
            (row["path"], row["dtype"]) == expected[name]
            and isinstance(row["bytes"], int)
            and not isinstance(row["bytes"], bool)
            and row["bytes"] >= 0
            and (row["bytes"] > 0 or name == "token_ids")
            and isinstance(row["shape"], list)
            and all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in row["shape"])
            and isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in row["sha256"]),
            f"reference artifact type mismatch: {name}",
        )
        artifact = root / row["path"]
        _require(artifact.is_file() and not artifact.is_symlink(), f"reference artifact missing: {name}")
        digest, size = stream_sha256(artifact)
        _require(digest == row["sha256"] and size == row["bytes"], f"reference artifact digest mismatch: {name}")
        _require(size == math.prod(row["shape"]) * 4, f"reference artifact shape/bytes mismatch: {name}")
        if name in {"pcm", "encoded", "logits"}:
            values = [value[0] for value in struct.iter_unpack("<f", artifact.read_bytes())]
            _require(all(math.isfinite(value) for value in values), f"reference artifact is non-finite: {name}")
            _require(any(value != 0.0 for value in values), f"reference artifact is all-zero: {name}")
            if name == "logits":
                _require(len(values) % 71 == 0, "reference logits class shape mismatch")
                for row_values in (values[offset : offset + 71] for offset in range(0, len(values), 71)):
                    maximum = max(row_values)
                    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in row_values))
                    _require(abs(logsumexp) <= 1e-4, "reference logits are not row-wise log-softmax")
    encoded_length = document["encoded_length"]
    _require(isinstance(encoded_length, int) and not isinstance(encoded_length, bool) and encoded_length > 0, "encoded length mismatch")
    _require(artifacts["pcm"]["shape"] and len(artifacts["pcm"]["shape"]) == 1, "PCM shape mismatch")
    _require(artifacts["pcm"]["shape"] == [PCM_SAMPLES] and artifacts["pcm"]["sha256"] == PCM_F32LE_SHA256, "fixed PCM artifact mismatch")
    _require(artifacts["encoded"]["shape"] == [encoded_length, 768], "encoded shape mismatch")
    _require(artifacts["logits"]["shape"] == [encoded_length, 71], "logits shape mismatch")
    _require(artifacts["raw_argmax"]["shape"] == [encoded_length], "argmax shape mismatch")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def self_test() -> None:
    """Exercise duplicate-key, bool-type, and shape/byte tamper rejection."""
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="gigaam-reference-self-test-", dir=temp_parent) as directory:
        base = Path(directory)
        root = base / "reference"
        root.mkdir()
        files = {
            "pcm": ("pcm.f32le", "float32", [PCM_SAMPLES]),
            "encoded": ("encoded.f32le", "float32", [1, 768]),
            "logits": ("logits.f32le", "float32", [1, 71]),
            "raw_argmax": ("raw_argmax.u32le", "uint32", [1]),
            "token_ids": ("token_ids.u32le", "uint32", [0]),
        }
        artifacts = {}
        for name, (filename, dtype, shape) in files.items():
            path = root / filename
            if name == "pcm":
                raw = struct.pack("<16000f", *[((index % 97) - 48) / 48.0 for index in range(16000)])
            elif name == "encoded":
                raw = struct.pack("<768f", 1.0, *([0.0] * 767))
            elif name == "logits":
                raw = struct.pack("<71f", *([-math.log(71.0)] * 71))
            else:
                raw = b"\0" * (math.prod(shape) * 4)
            path.write_bytes(raw)
            artifacts[name] = {"path": filename, "bytes": path.stat().st_size, "sha256": stream_sha256(path)[0], "shape": shape, "dtype": dtype}
        source_root = base / "source"
        source_root.mkdir()
        source_payloads = {
            "config": ("config.json", b"config"),
            "modeling_gigaam": ("modeling_gigaam.py", b"modeling"),
            "checkpoint": ("pytorch_model.bin", b"checkpoint"),
        }
        source = {}
        for name, (filename, payload) in source_payloads.items():
            path = source_root / filename
            path.write_bytes(payload)
            digest, size = stream_sha256(path)
            source[name] = {"path": str(path), "bytes": size, "sha256": digest}
        saved_constants = (CONFIG_SHA256, MODELING_SHA256, CHECKPOINT_SHA256, CHECKPOINT_BYTES)
        globals()["CONFIG_SHA256"] = source["config"]["sha256"]
        globals()["MODELING_SHA256"] = source["modeling_gigaam"]["sha256"]
        globals()["CHECKPOINT_SHA256"] = source["checkpoint"]["sha256"]
        globals()["CHECKPOINT_BYTES"] = source["checkpoint"]["bytes"]
        try:
            document = {"format": "vokra-gigaam-multilingual-reference-v1", "status": "REFERENCE_DUMP_OPEN_NOT_PARITY", "repository": "ai-sage/GigaAM-Multilingual", "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8", "source_revision": "7447938d791c4f3e643386ee22c33777004293a5", "config_sha256": CONFIG_SHA256, "modeling_gigaam_sha256": MODELING_SHA256, "source_files": source, "pcm_input": {"path": "/pcm.npy", "npy_sha256": "b" * 64, "f32le_sha256": PCM_F32LE_SHA256, "shape": [PCM_SAMPLES], "dtype": "float32", "sample_rate_hz": PCM_SAMPLE_RATE_HZ}, "artifacts": artifacts, "encoded_length": 1, "ctc": {"vocab_size": 71, "blank_id": 70, "collapse": "collapse_adjacent_repeat_then_remove_blank"}, "runtime": {}, "parity": "OPEN_MEASURED_NOT_GATED"}
            baseline = copy.deepcopy(document)
            (root / "manifest.json").write_text(json.dumps(baseline), encoding="utf-8")
            validate_reference_bundle(root)
            mutations = [
                ("artifact bytes", lambda doc: doc["artifacts"]["encoded"].update(bytes=True)),
                ("artifact shape", lambda doc: doc["artifacts"]["logits"].update(shape=[1, 70])),
                ("source path", lambda doc: doc["source_files"]["config"].update(path=str(base / "missing.json"))),
                ("source bytes", lambda doc: doc["source_files"]["config"].update(bytes=7)),
                ("source digest", lambda doc: doc["source_files"]["config"].update(sha256="a" * 64)),
                ("source basename", lambda doc: doc["source_files"]["config"].update(path=str(source_root / "wrong.json"))),
            ]
            symlink = source_root / "config-link.json"
            symlink.symlink_to(source_root / "config.json")
            mutations.append(("source symlink", lambda doc: doc["source_files"]["config"].update(path=str(symlink))))
            for label, mutate in mutations:
                candidate = copy.deepcopy(baseline)
                mutate(candidate)
                (root / "manifest.json").write_text(json.dumps(candidate), encoding="utf-8")
                try:
                    validate_reference_bundle(root)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"tampered {label} accepted")
            source_root.joinpath("config.json").write_bytes(b"tampered")
            (root / "manifest.json").write_text(json.dumps(baseline), encoding="utf-8")
            try:
                validate_reference_bundle(root)
            except ValueError:
                pass
            else:
                raise AssertionError("tampered source contents accepted")
            (root / "manifest.json").write_text('{"format":1,"format":1}', encoding="utf-8")
            try:
                validate_reference_bundle(root)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate manifest key accepted")
        finally:
            globals()["CONFIG_SHA256"], globals()["MODELING_SHA256"], globals()["CHECKPOINT_SHA256"], globals()["CHECKPOINT_BYTES"] = saved_constants


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("usage: gigaam_multilingual_validation.py --self-test")
    self_test()
    print("gigaam_multilingual_validation self-test: OK")
