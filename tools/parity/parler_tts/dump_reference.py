#!/usr/bin/env python3
"""Dump a pinned independent official Parler-TTS reference.

The oracle imports ``ParlerTTSForConditionalGeneration`` from the locked
official repository and runs its released text encoder, delayed generation,
and embedded DAC. Vokra is never imported and no Vokra forward is mirrored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import parler_tts
import torch
import transformers
from parler_tts import ParlerTTSForConditionalGeneration


PARLER_SOURCE_REVISION = "d108732cd57788ec86bc857d99a6cabd66663d68"
TRANSFORMERS_VERSION = "4.46.1"
SAMPLE_RATE = 44_100
FRAME_HOP = 512
NUM_CODEBOOKS = 9
CODEBOOK_SIZE = 1_024
MAX_FRAMES = 4
DESCRIPTION_TOKEN_IDS = [71, 1234, 1]
PROMPT_TOKEN_IDS = [12, 34, 1]

VARIANTS: dict[str, dict[str, object]] = {
    "english": {
        "upstream_hf": "parler-tts/parler-tts-mini-v1",
        "upstream_revision": "0392b9451a601e528fd863bbb0598431fee810d9",
        "checkpoint_bytes": 3_511_490_560,
        "checkpoint_sha256": "bc430eb6752b96ffb3f67036d1a6e207fbd031575a775716ffa64ef1eeb03692",
        "config_bytes": 6_930,
        "config_sha256": "d8d2afa72bf3b098263a073c4d4df18627b76e1eb454c48f60bc5f787b2433b1",
        "generation_bytes": 265,
        "generation_sha256": "77831b39a5e0c4dba09b4dcbe37ce082e10f94c646920b20678c9c5289e52440",
        "prompt_vocab": 32_128,
    },
    "multilingual": {
        "upstream_hf": "parler-tts/parler-tts-mini-multilingual-v1.1",
        "upstream_revision": "11b27d57855dec1ce0914ba1f12363bf2ea75ba3",
        "checkpoint_bytes": 3_751_321_772,
        "checkpoint_sha256": "79c64e3705e0ccce122988c7817f0d65efa3fd37625906d90765858bdab38412",
        "config_bytes": 7_467,
        "config_sha256": "06d4cb727521542cab6b26d3ad1c8517d51fd1f551600ec67a59575364e221c6",
        "generation_bytes": 218,
        "generation_sha256": "3bb518e78ea5f32fbbcfc7f0aaed388e7aefede474d2bf4b8cf4502fd6b27a92",
        "prompt_vocab": 90_714,
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


def verify_official_source() -> None:
    distribution = importlib.metadata.distribution("parler-tts")
    direct_url_raw = distribution.read_text("direct_url.json")
    if direct_url_raw is None:
        raise RuntimeError("parler-tts installation has no PEP 610 direct_url.json")
    direct_url = json.loads(direct_url_raw)
    actual_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if actual_revision != PARLER_SOURCE_REVISION:
        raise RuntimeError(
            f"parler-tts source {actual_revision!r} != pinned {PARLER_SOURCE_REVISION}"
        )


def verify_model_directory(model_dir: Path, identity: dict[str, object]) -> None:
    verify_file(
        model_dir / "model.safetensors",
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
        int(identity["generation_bytes"]),
        str(identity["generation_sha256"]),
    )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    decoder = config["decoder"]
    audio = config["audio_encoder"]
    expected = {
        "prompt_vocab": int(identity["prompt_vocab"]),
        "text_d_model": 1_024,
        "text_layers": 24,
        "decoder_hidden": 1_024,
        "decoder_layers": 24,
        "decoder_codebooks": NUM_CODEBOOKS,
        "decoder_vocab": 1_088,
        "codebook_size": CODEBOOK_SIZE,
        "sample_rate": SAMPLE_RATE,
    }
    actual = {
        "prompt_vocab": int(config["vocab_size"]),
        "text_d_model": int(config["text_encoder"]["d_model"]),
        "text_layers": int(config["text_encoder"]["num_layers"]),
        "decoder_hidden": int(decoder["hidden_size"]),
        "decoder_layers": int(decoder["num_hidden_layers"]),
        "decoder_codebooks": int(decoder["num_codebooks"]),
        "decoder_vocab": int(decoder["vocab_size"]),
        "codebook_size": int(audio["codebook_size"]),
        "sample_rate": int(audio["sampling_rate"]),
    }
    if actual != expected or bool(config["prompt_cross_attention"]):
        raise RuntimeError(
            f"Parler topology differs from pinned contract: {actual}, expected {expected}"
        )


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


def normalize_audio_codes(raw: torch.Tensor) -> torch.Tensor:
    codes = raw.detach().cpu().to(torch.int64)
    while codes.ndim > 2:
        if codes.shape[0] != 1:
            raise RuntimeError(f"unexpected official DAC code shape {list(codes.shape)}")
        codes = codes[0]
    if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS:
        raise RuntimeError(f"unexpected official DAC code shape {list(codes.shape)}")
    if bool(torch.any(codes < 0)) or bool(torch.any(codes >= CODEBOOK_SIZE)):
        raise RuntimeError("official Parler-TTS emitted an out-of-range DAC code")
    return codes.transpose(0, 1).contiguous()


def execution_environment() -> dict[str, object]:
    capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "parler_tts": str(parler_tts.__version__),
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
    verify_official_source()
    identity = VARIANTS[args.variant]
    verify_model_directory(args.model_dir, identity)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x5041524C)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).eval()

    description = torch.tensor([DESCRIPTION_TOKEN_IDS], dtype=torch.long)
    description_mask = torch.ones_like(description)
    prompt = torch.tensor([PROMPT_TOKEN_IDS], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt)
    captured_codes: list[torch.Tensor] = []
    original_decode = model.audio_encoder.decode

    def capture_decode(*decode_args: Any, **decode_kwargs: Any) -> Any:
        raw_codes = decode_kwargs.get("audio_codes")
        if raw_codes is None and decode_args:
            raw_codes = decode_args[0]
        if not isinstance(raw_codes, torch.Tensor):
            raise RuntimeError("official audio_encoder.decode received no tensor codes")
        captured_codes.append(raw_codes.detach().cpu().clone())
        return original_decode(*decode_args, **decode_kwargs)

    with torch.inference_mode():
        text_hidden = model.text_encoder(
            input_ids=description,
            attention_mask=description_mask,
            return_dict=True,
        ).last_hidden_state[0]
        model.audio_encoder.decode = capture_decode
        try:
            pcm = model.generate(
                description,
                attention_mask=description_mask,
                prompt_input_ids=prompt,
                prompt_attention_mask=prompt_mask,
                do_sample=False,
                min_new_tokens=0,
                max_length=MAX_FRAMES + NUM_CODEBOOKS,
            ).reshape(-1)
        finally:
            model.audio_encoder.decode = original_decode

    if len(captured_codes) != 1:
        raise RuntimeError(
            f"official DAC decode call count {len(captured_codes)} != expected 1"
        )
    codes = normalize_audio_codes(captured_codes[0])
    frames = int(codes.shape[0])
    if frames == 0 or frames > MAX_FRAMES:
        raise RuntimeError(f"official generated frame count {frames} is invalid")
    if pcm.numel() != frames * FRAME_HOP:
        raise RuntimeError(
            f"official PCM length {pcm.numel()} != {frames} * {FRAME_HOP}"
        )

    args.output.mkdir(parents=True, exist_ok=False)
    files = {
        "description_token_ids.u32le": args.output / "description_token_ids.u32le",
        "prompt_token_ids.u32le": args.output / "prompt_token_ids.u32le",
        "text_hidden.f32": args.output / "text_hidden.f32",
        "codes.u32le": args.output / "codes.u32le",
        "decoded_pcm.f32": args.output / "decoded_pcm.f32",
    }
    write_u32(files["description_token_ids.u32le"], DESCRIPTION_TOKEN_IDS)
    write_u32(files["prompt_token_ids.u32le"], PROMPT_TOKEN_IDS)
    write_f32(files["text_hidden.f32"], text_hidden)
    write_u32(files["codes.u32le"], codes.reshape(-1))
    write_f32(files["decoded_pcm.f32"], pcm)

    manifest = {
        "format": "vokra-parler-tts-official-reference-v1",
        "oracle": "official pinned ParlerTTSForConditionalGeneration text encoder, generate and embedded DAC",
        "variant": args.variant,
        "upstream_hf": identity["upstream_hf"],
        "upstream_revision": identity["upstream_revision"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "config_sha256": identity["config_sha256"],
        "generation_config_sha256": identity["generation_sha256"],
        "parler_source_revision": PARLER_SOURCE_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "description_tokens": len(DESCRIPTION_TOKEN_IDS),
        "prompt_tokens": len(PROMPT_TOKEN_IDS),
        "text_hidden": list(text_hidden.shape),
        "max_frames": MAX_FRAMES,
        "frames": frames,
        "codebooks": NUM_CODEBOOKS,
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
