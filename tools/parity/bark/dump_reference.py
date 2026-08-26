#!/usr/bin/env python3
"""Dump a pinned independent official Transformers Bark reference.

The oracle imports ``BarkModel`` from locked Transformers 4.31.0 and loads the
exact immutable Suno checkpoint supplied by the VAST worker. Vokra is never
imported and no Vokra forward is mirrored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import BarkModel
from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)


TRANSFORMERS_VERSION = "4.31.0"
TRANSFORMERS_SOURCE_REVISION = "e42587f596181396e1c4b63660abf0c736b10dae"
GENERATION_CONFIG_BYTES = 4_908
GENERATION_CONFIG_SHA256 = (
    "ab2969fcd40e085bc924ad99ad419c27f62f5acb61afac5de7490ab0c796b5b9"
)
SAMPLE_RATE = 24_000
FRAME_HOP = 320
CODEBOOKS = 8
CODEBOOK_SIZE = 1_024
MAX_SEMANTIC_TOKENS = 4
# Caller-visible GPT-2/Bark vocabulary IDs. Their linguistic content is not a
# parity premise; only this exact finite sequence is part of the fixture.
TEXT_TOKEN_IDS = [15496, 11, 428, 318, 257, 1332, 13]

VARIANTS: dict[str, dict[str, object]] = {
    "small": {
        "upstream_hf": "suno/bark-small",
        "upstream_revision": "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
        "checkpoint_bytes": 1_676_663_913,
        "checkpoint_sha256": "f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9",
        "config_bytes": 8_803,
        "config_sha256": "9d95e9c3027cd79cf5f762cc03a69b6393cea87c51e9dd6b998fde3a7f01510e",
        "hidden": 768,
        "heads": 12,
        "layers": 12,
    },
    "full": {
        "upstream_hf": "suno/bark",
        "upstream_revision": "70a8a7d34168586dc5d028fa9666aceade177992",
        "checkpoint_bytes": 4_486_643_861,
        "checkpoint_sha256": "4e3d407b9b3b619da184c85786c88e5e35f90f9089303e16db696ed0be477989",
        "config_bytes": 8_806,
        "config_sha256": "48be144c0232acd8c55786d1eea9161ae6c973f21ec4a2f02627c844065ea695",
        "hidden": 1_024,
        "heads": 16,
        "layers": 24,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing pinned input: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path.name} bytes {actual_bytes} != pinned {expected_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{path.name} SHA-256 {actual_sha256} != pinned {expected_sha256}"
        )


def verify_model_directory(model_dir: Path, identity: dict[str, object]) -> None:
    verify_file(
        model_dir / "pytorch_model.bin",
        int(identity["checkpoint_bytes"]),
        str(identity["checkpoint_sha256"]),
    )
    verify_file(
        model_dir / "config.json",
        int(identity["config_bytes"]),
        str(identity["config_sha256"]),
    )
    verify_file(
        model_dir / "generation_config.json",
        GENERATION_CONFIG_BYTES,
        GENERATION_CONFIG_SHA256,
    )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected = (
        int(identity["hidden"]),
        int(identity["heads"]),
        int(identity["layers"]),
    )
    for stage in ["semantic_config", "coarse_acoustics_config", "fine_acoustics_config"]:
        section = config[stage]
        actual = (
            int(section["hidden_size"]),
            int(section["num_heads"]),
            int(section["num_layers"]),
        )
        if actual != expected:
            raise RuntimeError(f"{stage} topology {actual} != pinned {expected}")
    codec = config["codec_config"]
    if (
        int(codec["sampling_rate"]) != SAMPLE_RATE
        or list(codec["upsampling_ratios"]) != [8, 5, 4, 2]
        or int(codec["codebook_size"]) != CODEBOOK_SIZE
        or int(codec["codebook_dim"]) != 128
    ):
        raise RuntimeError("codec topology differs from the pinned causal 24 kHz release")


def write_u32(path: Path, values: torch.Tensor | list[int]) -> None:
    array = np.asarray(
        values.detach().cpu().to(torch.int64).contiguous().numpy()
        if isinstance(values, torch.Tensor)
        else values,
        dtype=np.int64,
    )
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise RuntimeError(f"{path.name} contains a value outside u32")
    path.write_bytes(array.astype("<u4", copy=False).tobytes(order="C"))


def write_f32(path: Path, values: torch.Tensor) -> None:
    array = values.detach().cpu().to(torch.float32).contiguous().numpy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"{path.name} contains a non-finite value")
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def execution_environment() -> dict[str, object]:
    capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "torch_cpu_capability": (
            str(capability()) if callable(capability) else "unavailable"
        ),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers {transformers.__version__} != pinned {TRANSFORMERS_VERSION}"
        )
    identity = VARIANTS[args.variant]
    verify_model_directory(args.model_dir, identity)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x4241524B)
    model = BarkModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()

    semantic_config = BarkSemanticGenerationConfig(
        **model.generation_config.semantic_config
    )
    coarse_config = BarkCoarseGenerationConfig(
        **model.generation_config.coarse_acoustics_config
    )
    fine_config = BarkFineGenerationConfig(
        **model.generation_config.fine_acoustics_config
    )
    input_ids = torch.zeros((1, 256), dtype=torch.long)
    attention_mask = torch.zeros((1, 256), dtype=torch.long)
    input_ids[0, : len(TEXT_TOKEN_IDS)] = torch.tensor(TEXT_TOKEN_IDS)
    attention_mask[0, : len(TEXT_TOKEN_IDS)] = 1

    with torch.inference_mode():
        semantic = model.semantic.generate(
            input_ids,
            attention_mask=attention_mask,
            semantic_generation_config=semantic_config,
            do_sample=False,
            max_new_tokens=MAX_SEMANTIC_TOKENS,
        )
        coarse = model.coarse_acoustics.generate(
            semantic.clone(),
            semantic_generation_config=semantic_config,
            coarse_generation_config=coarse_config,
            codebook_size=int(model.generation_config.codebook_size),
            do_sample=False,
        )
        fine = model.fine_acoustics.generate(
            coarse,
            semantic_generation_config=semantic_config,
            coarse_generation_config=coarse_config,
            fine_generation_config=fine_config,
            codebook_size=int(model.generation_config.codebook_size),
            temperature=None,
        )
        pcm = model.codec_decode(fine).reshape(-1)

    if semantic.shape[0] != 1 or semantic.shape[1] == 0:
        raise RuntimeError(f"unexpected semantic shape {list(semantic.shape)}")
    if fine.ndim != 3 or list(fine.shape[:2]) != [1, CODEBOOKS]:
        raise RuntimeError(f"unexpected fine shape {list(fine.shape)}")
    frames = int(fine.shape[2])
    codes = fine[0].transpose(0, 1).contiguous()
    if codes.shape != (frames, CODEBOOKS):
        raise RuntimeError(f"unexpected frame-major code shape {list(codes.shape)}")
    if bool(torch.any(codes < 0)) or bool(torch.any(codes >= CODEBOOK_SIZE)):
        raise RuntimeError("official Bark emitted an out-of-range codec index")
    if pcm.numel() != frames * FRAME_HOP:
        raise RuntimeError(
            f"official PCM length {pcm.numel()} != {frames} * {FRAME_HOP}"
        )

    args.output.mkdir(parents=True, exist_ok=False)
    files = {
        "text_token_ids.u32le": args.output / "text_token_ids.u32le",
        "semantic_tokens.u32le": args.output / "semantic_tokens.u32le",
        "codes.u32le": args.output / "codes.u32le",
        "decoded_pcm.f32": args.output / "decoded_pcm.f32",
    }
    write_u32(files["text_token_ids.u32le"], TEXT_TOKEN_IDS)
    write_u32(files["semantic_tokens.u32le"], semantic.reshape(-1))
    write_u32(files["codes.u32le"], codes.reshape(-1))
    write_f32(files["decoded_pcm.f32"], pcm)

    manifest = {
        "format": "vokra-bark-transformers-4.31-reference-v1",
        "oracle": "official Transformers BarkModel semantic/coarse/fine generate plus codec_decode",
        "variant": args.variant,
        "upstream_hf": identity["upstream_hf"],
        "upstream_revision": identity["upstream_revision"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "config_sha256": identity["config_sha256"],
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_source_revision": TRANSFORMERS_SOURCE_REVISION,
        "max_semantic_tokens": MAX_SEMANTIC_TOKENS,
        "text_tokens": len(TEXT_TOKEN_IDS),
        "semantic_tokens": int(semantic.numel()),
        "frames": frames,
        "codebooks": CODEBOOKS,
        "codebook_size": CODEBOOK_SIZE,
        "sample_rate": SAMPLE_RATE,
        "frame_hop": FRAME_HOP,
        "environment": execution_environment(),
        "files": {name: sha256_file(path) for name, path in sorted(files.items())},
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
