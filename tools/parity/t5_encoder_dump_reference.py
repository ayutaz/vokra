#!/usr/bin/env python3
"""Dump an independent official T5-base encoder reference.

The oracle is the pinned ``transformers.T5EncoderModel.forward`` body.  This
script does not import Vokra and does not reproduce T5 equations.  It records
fixed token ids, an encoder key mask, the official FP32 hidden states, the
upstream revision supplied by the caller, and SHA-256 identities for every
checkpoint weight file visible in the local snapshot.

Run only through the repository's Python 3.12 parity environment, normally on
the VAST worker that downloaded the immutable snapshot::

    uv run --project tools/parity/t5_encoder --frozen --python 3.12 python \
        tools/parity/t5_encoder_dump_reference.py \
        --checkpoint /work/t5-base-snapshot \
        --source-revision <immutable-hf-commit> \
        --output-dir /work/t5-base-reference

The default source id is ``google-t5/t5-base``.  ``--source-revision`` is
mandatory so a mutable branch name cannot become parity provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import T5EncoderModel


TOKEN_IDS = np.asarray([71, 1234, 5, 0, 42, 9, 1], dtype="<u4")
ATTENTION_MASK = np.asarray([1, 1, 1, 0, 1, 1, 1], dtype="u1")

EXPECTED_CONFIG = {
    "vocab_size": 32_128,
    "d_model": 768,
    "d_kv": 64,
    "d_ff": 3_072,
    "num_layers": 12,
    "num_heads": 12,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1.0e-6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_identities(checkpoint: Path) -> dict[str, dict[str, object]]:
    weights = {
        *checkpoint.glob("*.safetensors"),
        *checkpoint.glob("pytorch_model*.bin"),
    }
    if not weights:
        raise SystemExit(
            f"checkpoint contains no *.safetensors or pytorch_model*.bin files: {checkpoint}"
        )
    config = checkpoint / "config.json"
    if not config.is_file():
        raise SystemExit(f"checkpoint contains no config.json: {checkpoint}")
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted({config, *weights})
    }


def _validate_config(model: T5EncoderModel) -> dict[str, object]:
    config = model.config
    observed: dict[str, object] = {
        key: getattr(config, key) for key in EXPECTED_CONFIG
    }
    for key, expected in EXPECTED_CONFIG.items():
        actual = observed[key]
        if isinstance(expected, float):
            matches = abs(float(actual) - expected) <= 1.0e-12
        else:
            matches = actual == expected
        if not matches:
            raise SystemExit(
                f"checkpoint is not canonical T5-base: config.{key}={actual!r}, expected {expected!r}"
            )
    dense_act_fn = getattr(config, "dense_act_fn", None)
    is_gated_act = bool(getattr(config, "is_gated_act", False))
    if dense_act_fn != "relu" or is_gated_act:
        raise SystemExit(
            "checkpoint is not the non-gated ReLU T5-base topology: "
            f"dense_act_fn={dense_act_fn!r}, is_gated_act={is_gated_act!r}"
        )
    observed["dense_act_fn"] = dense_act_fn
    observed["is_gated_act"] = is_gated_act
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-repo", default="google-t5/t5-base")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        parser.error(f"--checkpoint is not a directory: {checkpoint}")
    source_revision = args.source_revision.strip()
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        parser.error("--source-revision must be a 40-character lowercase commit id")
    source_repo = args.source_repo.strip()
    if not source_repo:
        parser.error("--source-repo must be non-empty")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output-dir exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"--output-dir must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = T5EncoderModel.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device="cpu", dtype=torch.float32)
    model.eval()
    config = _validate_config(model)

    input_ids = torch.from_numpy(TOKEN_IDS.astype(np.int64, copy=True))[None, :]
    attention_mask = torch.from_numpy(
        ATTENTION_MASK.astype(np.int64, copy=True)
    )[None, :]
    with torch.inference_mode():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state[0]
    output = hidden.detach().to(device="cpu", dtype=torch.float32).numpy()
    if output.shape != (TOKEN_IDS.size, EXPECTED_CONFIG["d_model"]):
        raise SystemExit(f"unexpected official output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise SystemExit("official T5 output contains a non-finite value")

    inputs_path = output_dir / "input_ids.u32"
    mask_path = output_dir / "attention_mask.u8"
    output_path = output_dir / "last_hidden_state.f32"
    TOKEN_IDS.tofile(inputs_path)
    ATTENTION_MASK.tofile(mask_path)
    output.astype("<f4", copy=False).tofile(output_path)

    fixture_files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (inputs_path, mask_path, output_path)
    }
    manifest = {
        "format": "vokra-t5-encoder-reference-v1",
        "oracle": "transformers.T5EncoderModel.forward",
        "source_repo": source_repo,
        "source_revision": source_revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "config": config,
        "shape": [int(TOKEN_IDS.size), int(EXPECTED_CONFIG["d_model"])],
        "checkpoint_files": _checkpoint_identities(checkpoint),
        "fixtures": fixture_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "t5_encoder_reference: "
        f"source={source_repo}@{source_revision} "
        f"shape={output.shape} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
