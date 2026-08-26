#!/usr/bin/env python3
"""Dump an independent official MusicGen delay-pattern reference.

The oracle is the pinned
``transformers.MusicgenForCausalLM.build_delay_pattern_mask`` and
``apply_delay_pattern_mask`` implementation.  This script imports those
methods directly; it neither imports Vokra nor reproduces the scheduling
equations.  No model or checkpoint is constructed.

Run through the repository's pinned Python 3.12 parity environment, normally
on the VAST worker::

    uv run --project tools/parity/t5_encoder --frozen --python 3.12 python \
        tools/parity/musicgen_delay_pattern_dump_reference.py \
        --output-dir /work/musicgen-delay-reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from transformers import MusicgenForCausalLM


EXPECTED_TRANSFORMERS_VERSION = "4.45.2"
PAD_TOKEN_ID = 2_048

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "mono_no_prompt",
        "batch_size": 1,
        "num_codebooks": 4,
        "prompt_len": 1,
        "max_length": 8,
        "audio_channels": 1,
        "input_ids": [PAD_TOKEN_ID] * 4,
    },
    {
        "name": "mono_prompted",
        "batch_size": 1,
        "num_codebooks": 4,
        "prompt_len": 3,
        "max_length": 8,
        "audio_channels": 1,
        "input_ids": [
            PAD_TOKEN_ID,
            10,
            11,
            PAD_TOKEN_ID,
            12,
            13,
            PAD_TOKEN_ID,
            14,
            15,
            PAD_TOKEN_ID,
            16,
            17,
        ],
        "apply_seq_len": 5,
        "generated_token": 99,
    },
    {
        "name": "stereo_no_prompt",
        "batch_size": 1,
        "num_codebooks": 4,
        "prompt_len": 1,
        "max_length": 5,
        "audio_channels": 2,
        "input_ids": [PAD_TOKEN_ID] * 4,
    },
    {
        "name": "mono_short_sequence",
        "batch_size": 1,
        "num_codebooks": 4,
        "prompt_len": 1,
        "max_length": 6,
        "audio_channels": 1,
        "input_ids": [PAD_TOKEN_ID] * 4,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_u32(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().to(device="cpu", dtype=torch.int64).numpy()
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint32).max):
        raise SystemExit(f"{path.name}: official output does not fit u32")
    values.astype("<u4", copy=False).tofile(path)


def _write_i64(path: Path, tensor: torch.Tensor) -> None:
    tensor.detach().to(device="cpu", dtype=torch.int64).numpy().astype(
        "<i8", copy=False
    ).tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        parser.error(
            "official delay-pattern oracle requires "
            f"transformers=={EXPECTED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output-dir exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"--output-dir must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest_cases: list[dict[str, object]] = []
    fixture_files: dict[str, dict[str, object]] = {}
    for case in CASES:
        rows = int(case["batch_size"]) * int(case["num_codebooks"])
        prompt_len = int(case["prompt_len"])
        input_ids = torch.tensor(case["input_ids"], dtype=torch.long).reshape(
            rows, prompt_len
        )
        # The unbound official method only reads these two fields. Avoiding a
        # model construction proves this fixture is scheduling-only and keeps
        # checkpoint/model artefacts out of the process.
        oracle_self = SimpleNamespace(
            num_codebooks=int(case["num_codebooks"]),
            config=SimpleNamespace(audio_channels=int(case["audio_channels"])),
        )
        prefix, pattern = MusicgenForCausalLM.build_delay_pattern_mask(
            oracle_self,
            input_ids,
            pad_token_id=PAD_TOKEN_ID,
            max_length=int(case["max_length"]),
        )
        prefix = prefix.contiguous()
        pattern = pattern.contiguous()

        stem = str(case["name"])
        input_path = output_dir / f"{stem}.input_ids.u32"
        prefix_path = output_dir / f"{stem}.prefix.u32"
        pattern_path = output_dir / f"{stem}.pattern.i64"
        _write_u32(input_path, input_ids)
        _write_u32(prefix_path, prefix)
        _write_i64(pattern_path, pattern)
        for path in (input_path, prefix_path, pattern_path):
            fixture_files[path.name] = _identity(path)

        manifest_case: dict[str, object] = {
            key: value for key, value in case.items() if key != "input_ids"
        }
        manifest_case["rows"] = rows
        manifest_case["prefix_len"] = int(prefix.shape[-1])
        manifest_case["files"] = {
            "input_ids": input_path.name,
            "prefix": prefix_path.name,
            "pattern": pattern_path.name,
        }
        if "apply_seq_len" in case:
            apply_seq_len = int(case["apply_seq_len"])
            generated = torch.full(
                (rows, apply_seq_len),
                int(case["generated_token"]),
                dtype=torch.long,
            )
            applied = MusicgenForCausalLM.apply_delay_pattern_mask(
                generated, pattern
            ).contiguous()
            applied_path = output_dir / f"{stem}.applied.u32"
            _write_u32(applied_path, applied)
            fixture_files[applied_path.name] = _identity(applied_path)
            manifest_case["files"]["applied"] = applied_path.name
        manifest_cases.append(manifest_case)

    manifest = {
        "format": "vokra-musicgen-delay-pattern-reference-v1",
        "oracle": (
            "transformers.MusicgenForCausalLM.build_delay_pattern_mask+"
            "apply_delay_pattern_mask"
        ),
        "source": (
            "github.com/huggingface/transformers/blob/v4.45.2/"
            "src/transformers/models/musicgen/modeling_musicgen.py"
        ),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "pad_token_id": PAD_TOKEN_ID,
        "cases": manifest_cases,
        "fixtures": fixture_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote official MusicGen delay-pattern reference: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
