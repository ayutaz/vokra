#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Official-source GigaAM Multilingual reference dumper.

This is intentionally a VAST-side tool. It imports the pinned Hugging Face
remote-code model and writes logits for an explicit PCM fixture; it never
claims parity by itself. Missing source/checkpoint/input or an unpinned model
revision aborts loudly.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

HF_REPOSITORY = "ai-sage/GigaAM-Multilingual"
HF_REVISION = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
CONFIG_SHA256 = "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653"
MODELING_SHA256 = "6d02e640fbb5738ab11c030520a68654ef32f4ff363723db10534cf8b5d5c0e7"
CHECKPOINT_BYTES = 883170115
CHECKPOINT_SHA256 = "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728"
PCM_SAMPLE_RATE_HZ = 16_000
PCM_SAMPLES = 16_000
PCM_F32LE_SHA256 = "f92e4a0422c513ab107975f5c9bd7a8e7a92532b37508a769c92d2496625229b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_symlink_ancestry(path: Path, label: str) -> None:
    """Reject symlinks on every component of an input or output path."""
    absolute = Path.cwd() / path if not path.is_absolute() else path
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise SystemExit(f"{label} has symlink ancestry: {ancestor}")


def write_f32(path: Path, values: "np.ndarray") -> dict[str, object]:
    """Write a contiguous little-endian float32 array and return its row."""
    import numpy as np

    raw = np.asarray(values, dtype="<f4", order="C").tobytes(order="C")
    path.write_bytes(raw)
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(values.shape),
        "dtype": "float32",
    }


def write_u32(path: Path, values: "np.ndarray") -> dict[str, object]:
    """Write a contiguous little-endian uint32 array and return its row."""
    import numpy as np

    raw = np.asarray(values, dtype="<u4", order="C").tobytes(order="C")
    path.write_bytes(raw)
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(values.shape),
        "dtype": "uint32",
    }


def decode_ctc(raw_ids: list[int], blank_id: int = 70) -> list[int]:
    """Collapse adjacent raw IDs, then remove CTC blank IDs.

    A blank resets the adjacent-repeat state before it is removed.  Therefore
    ``[a, a, blank, a]`` decodes to ``[a, a]``: the final ``a`` is separated
    from the first run by the blank.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for value in raw_ids:
        token = int(value)
        if token != previous:
            collapsed.append(token)
        previous = token
    return [token for token in collapsed if token != blank_id]


def self_test_wrapper_contract() -> None:
    """Pin the official wrapper-to-inner-head boundary without importing it."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "head"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "model"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "model"
        for node in ast.walk(tree)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pcm-npy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert len(HF_REVISION) == 40 and len(SOURCE_REVISION) == 40
        assert len(CONFIG_SHA256) == 64 and len(MODELING_SHA256) == 64
        assert CHECKPOINT_BYTES > 0
        assert decode_ctc([2, 2, 70, 2]) == [2, 2]
        self_test_wrapper_contract()
        print("sber_gigaam_multilingual_dump_reference self-test: OK")
        return 0
    if args.model_dir is None or args.pcm_npy is None or args.output is None:
        parser.error("--model-dir, --pcm-npy, and --output are required outside --self-test")
    reject_symlink_ancestry(args.model_dir, "model directory")
    reject_symlink_ancestry(args.pcm_npy, "PCM fixture")
    reject_symlink_ancestry(args.output, "reference output")
    if args.model_dir.is_symlink() or not args.model_dir.is_dir() or not (args.model_dir / "config.json").is_file():
        raise SystemExit("official model directory/config.json is missing")
    config_path = args.model_dir / "config.json"
    modeling_path = args.model_dir / "modeling_gigaam.py"
    checkpoint_path = args.model_dir / "pytorch_model.bin"
    for source_path, label in ((config_path, "config.json"), (modeling_path, "modeling_gigaam.py")):
        reject_symlink_ancestry(source_path, label)
    if config_path.is_symlink() or modeling_path.is_symlink() or not modeling_path.is_file():
        raise SystemExit("official modeling_gigaam.py is missing; refusing mirror implementation")
    reject_symlink_ancestry(checkpoint_path, "pytorch_model.bin")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise SystemExit("official pytorch_model.bin is missing or symlinked")
    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit("official pytorch_model.bin byte size mismatch")
    if args.pcm_npy.is_symlink() or not args.pcm_npy.is_file():
        raise SystemExit("PCM fixture is missing")
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("reference output directory must be absent and non-symlink")
    output_dir = args.output
    manifest_path = output_dir / "manifest.json"

    import numpy as np
    import torch
    from transformers import AutoModel

    if sha256(config_path) != CONFIG_SHA256:
        raise SystemExit("config.json SHA-256 mismatch")
    if sha256(modeling_path) != MODELING_SHA256:
        raise SystemExit("modeling_gigaam.py SHA-256 mismatch")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"invalid official config JSON: {exc}") from exc
    if config.get("_commit_hash") not in (None, HF_REVISION):
        raise SystemExit("model directory is not pinned to the authenticated revision")
    pcm = np.load(args.pcm_npy, allow_pickle=False)
    if pcm.ndim != 1 or pcm.size != PCM_SAMPLES or pcm.dtype != np.dtype("float32"):
        raise SystemExit(f"PCM fixture must be the fixed float32 length-{PCM_SAMPLES} array")
    if not np.isfinite(pcm).all() or not np.any(pcm != 0):
        raise SystemExit("PCM fixture must be finite and nonzero")
    pcm_sha256 = sha256(args.pcm_npy)
    checkpoint_sha256 = sha256(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise SystemExit("official pytorch_model.bin SHA-256 mismatch")
    model = AutoModel.from_pretrained(
        args.model_dir, revision=HF_REVISION, trust_remote_code=True, local_files_only=True
    ).eval()
    inner_model = getattr(model, "model", None)
    if type(model).__name__ != "GigaAMModel" or inner_model is None or not hasattr(inner_model, "head"):
        raise SystemExit("official GigaAMModel wrapper/inner head contract is missing")
    if not type(inner_model).__module__.endswith("modeling_gigaam"):
        raise SystemExit("inner head is not provided by the pinned modeling_gigaam module")
    waveform = torch.from_numpy(np.asarray(pcm, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([waveform.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        encoded, encoded_lengths = model.forward(waveform, lengths)
        logits = model.model.head(encoded)
    # The pinned encoder returns `[B, D, T]`; CTCHead transposes its Conv1d
    # output to `[B, T, 71]`. Keep the reference rows time-major too.
    encoded_np = np.asarray(encoded[0].transpose(0, 1).cpu().numpy(), dtype=np.float32)
    logits_np = np.asarray(logits[0].cpu().numpy(), dtype=np.float32)
    lengths_np = encoded_lengths.cpu().numpy()
    if logits_np.ndim != 2 or logits_np.shape[-1] != 71:
        raise SystemExit(f"official logits must have shape [T,71], found {logits_np.shape}")
    if encoded_np.ndim != 2 or encoded_np.shape[0] != logits_np.shape[0]:
        raise SystemExit(f"encoded/logits time shape mismatch: {encoded_np.shape}, {logits_np.shape}")
    if encoded_np.dtype.kind not in "fc" or logits_np.dtype.kind not in "fc":
        raise SystemExit("official encoded/logits must be floating arrays")
    if lengths_np.shape != (1,) or int(lengths_np[0]) != logits_np.shape[0]:
        raise SystemExit("encoded length does not match logits time axis")
    if not np.isfinite(encoded_np).all() or not np.isfinite(logits_np).all():
        raise SystemExit("official encoded/logits contain non-finite values")
    if not np.any(logits_np != 0):
        raise SystemExit("official logits are all zero")
    logsumexp = np.logaddexp.reduce(logits_np, axis=1)
    if not np.allclose(logsumexp, 0.0, atol=1e-4, rtol=0.0):
        raise SystemExit("official CTC head output is not row-wise log-softmax")
    pcm_f32 = np.asarray(pcm, dtype=np.float32)
    pcm_raw_sha256 = hashlib.sha256(np.asarray(pcm_f32, dtype="<f4").tobytes()).hexdigest()
    if pcm_raw_sha256 != PCM_F32LE_SHA256:
        raise SystemExit("fixed PCM f32le SHA-256 mismatch")
    raw_argmax = np.argmax(logits_np, axis=1).astype(np.uint32)
    token_ids = decode_ctc(raw_argmax.tolist())
    token_ids_np = np.asarray(token_ids, dtype=np.uint32)
    output_dir.mkdir(parents=True)
    artifacts = {
        "pcm": write_f32(output_dir / "pcm.f32le", pcm_f32),
        "encoded": write_f32(output_dir / "encoded.f32le", encoded_np),
        "logits": write_f32(output_dir / "logits.f32le", logits_np),
        "raw_argmax": write_u32(output_dir / "raw_argmax.u32le", raw_argmax),
        "token_ids": write_u32(output_dir / "token_ids.u32le", token_ids_np),
    }
    manifest_path.write_text(json.dumps({
        "format": "vokra-gigaam-multilingual-reference-v1",
        "status": "REFERENCE_DUMP_OPEN_NOT_PARITY",
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "config_sha256": CONFIG_SHA256,
        "modeling_gigaam_sha256": MODELING_SHA256,
        "source_files": {
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": CONFIG_SHA256},
            "modeling_gigaam": {"path": str(modeling_path), "bytes": modeling_path.stat().st_size, "sha256": MODELING_SHA256},
            "checkpoint": {"path": str(checkpoint_path), "bytes": CHECKPOINT_BYTES, "sha256": checkpoint_sha256},
        },
        "pcm_input": {"path": str(args.pcm_npy), "npy_sha256": pcm_sha256, "f32le_sha256": pcm_raw_sha256, "shape": list(pcm.shape), "dtype": str(pcm.dtype), "sample_rate_hz": PCM_SAMPLE_RATE_HZ},
        "artifacts": artifacts,
        "encoded_length": int(lengths_np[0]),
        "ctc": {"vocab_size": 71, "blank_id": 70, "collapse": "collapse_adjacent_repeat_then_remove_blank"},
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": __import__("transformers").__version__, "official_import": "transformers.AutoModel.from_pretrained(trust_remote_code=True)"},
        "parity": "OPEN_MEASURED_NOT_GATED",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote official raw reference artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
