#!/usr/bin/env python3
"""Dump an independent Transformers deepfake-classifier reference.

The oracle imports ``Wav2Vec2ForSequenceClassification`` from the upstream
recorded Transformers 4.41.2 release and strictly loads the immutable
``MelodyMachine/Deepfake-audio-detection-V2`` safetensors checkpoint. It does
not mirror the Rust graph. A deterministic raw-PCM fixture exercises the
official feature extractor, encoder, projector, mean pool, classifier and
softmax without adding an unrelated third-party audio file to provenance.

Run the real checkpoint only through the VAST worker. ``--self-test`` is
dependency-free and is safe on the maintainer Mac through ``uv run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

UPSTREAM_REPO = "MelodyMachine/Deepfake-audio-detection-V2"
UPSTREAM_REVISION = "de3cde5a29c449bb5268814e421b46bf6ebdcd72"
CHECKPOINT_FILE = "model.safetensors"
CHECKPOINT_BYTES = 378_302_360
CHECKPOINT_SHA256 = (
    "997d9ce59e63151d5e444a6fa7c863986d0e56d515f67321bd705ac3b01bc38c"
)
CONFIG_FILE = "config.json"
CONFIG_BYTES = 2_509
CONFIG_SHA256 = "a7ff31ca7ba4dc7fb5c4847d6dff0cb8daa1f0ec512e6ff8190664874c5b2806"
PREPROCESSOR_FILE = "preprocessor_config.json"
PREPROCESSOR_BYTES = 215
PREPROCESSOR_SHA256 = (
    "8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827"
)
TRANSFORMERS_VERSION = "5.5.0"
SAMPLE_RATE = 16_000
SAMPLES = 16_000
LABELS = ["fake", "real"]
SIGNAL_SEED = 0x6D2B79F5
# Filled from the pure-integer LCG below; it pins the raw little-endian f32
# fixture independently of NumPy, torch and the model checkpoint.
SIGNAL_SHA256 = "b95320de8c0182cc0a916dbbfe03fa8a1103e5a2ab71cb56e217f8e712f51585"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_hash: str, label: str) -> None:
    if not path.is_file():
        sys.exit(f"{label}: missing {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        sys.exit(f"{label}: {path} has {actual_size} bytes, expected {size}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        sys.exit(f"{label}: {path} SHA-256 {actual_hash}, expected {expected_hash}")


def deterministic_pcm() -> list[float]:
    """Return exact binary-rational f32 values without a platform RNG."""

    state = SIGNAL_SEED
    values: list[float] = []
    for _ in range(SAMPLES):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        signed_24 = (state >> 8) - 8_388_608
        values.append(signed_24 / 33_554_432.0)
    return values


def signal_bytes() -> bytes:
    return struct.pack(f"<{SAMPLES}f", *deterministic_pcm())


def tensor_value(value: Any):
    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    elif isinstance(value, (tuple, list)):
        value = value[0]
    return value.detach().to(device="cpu", dtype=value.dtype).contiguous()


def validate_official_contract(config: Any, extractor: Any) -> None:
    expected = {
        "model_type": "wav2vec2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3_072,
        "classifier_proj_size": 256,
        "num_labels": 2,
        "feat_extract_norm": "group",
        "do_stable_layer_norm": False,
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "use_weighted_layer_sum": False,
    }
    for key, wanted in expected.items():
        actual = getattr(config, key)
        if actual != wanted:
            sys.exit(f"config drift: {key}={actual!r}, expected {wanted!r}")
    labels = [config.id2label[index] for index in range(config.num_labels)]
    if labels != LABELS:
        sys.exit(f"config label order drifted: {labels!r}")
    if extractor.sampling_rate != SAMPLE_RATE:
        sys.exit(
            f"preprocessor sampling_rate={extractor.sampling_rate}, expected {SAMPLE_RATE}"
        )
    if extractor.do_normalize is not True or extractor.return_attention_mask is not False:
        sys.exit(
            "preprocessor must retain do_normalize=true and return_attention_mask=false"
        )


def dump(args: argparse.Namespace) -> None:
    checkpoint = args.input_dir / CHECKPOINT_FILE
    config_path = args.input_dir / CONFIG_FILE
    preprocessor = args.input_dir / PREPROCESSOR_FILE
    verify_file(checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "checkpoint")
    verify_file(config_path, CONFIG_BYTES, CONFIG_SHA256, "config")
    verify_file(
        preprocessor,
        PREPROCESSOR_BYTES,
        PREPROCESSOR_SHA256,
        "preprocessor config",
    )

    try:
        import numpy as np
        import safetensors
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import (
            Wav2Vec2Config,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForSequenceClassification,
        )
    except ImportError as error:
        sys.exit(f"missing locked parity dependency: {error}")

    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(
            f"Transformers {transformers.__version__} loaded, expected {TRANSFORMERS_VERSION}"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = Wav2Vec2Config.from_json_file(str(config_path))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.input_dir, local_files_only=True
    )
    validate_official_contract(config, extractor)
    model = Wav2Vec2ForSequenceClassification(config)
    incompatible = model.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        sys.exit(f"strict official checkpoint load mismatch: {incompatible}")
    model.eval()

    raw_bytes = signal_bytes()
    actual_signal_hash = hashlib.sha256(raw_bytes).hexdigest()
    if actual_signal_hash != SIGNAL_SHA256:
        sys.exit(
            f"deterministic PCM SHA-256 {actual_signal_hash}, expected {SIGNAL_SHA256}"
        )
    pcm = np.frombuffer(raw_bytes, dtype="<f4").copy()
    batch = extractor(
        pcm,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=False,
        return_tensors="pt",
    )
    if set(batch) != {"input_values"}:
        sys.exit(f"unexpected official feature-extractor fields: {sorted(batch)}")

    captured: dict[str, Any] = {}

    def capture_encoder(_module: Any, _inputs: Any, output: Any) -> None:
        captured["encoder_features"] = tensor_value(output)

    def capture_projector(_module: Any, _inputs: Any, output: Any) -> None:
        captured["projected_features"] = tensor_value(output)

    handles = [
        model.wav2vec2.encoder.register_forward_hook(capture_encoder),
        model.projector.register_forward_hook(capture_projector),
    ]
    with torch.inference_mode():
        output = model(**batch, return_dict=True)
        projected = captured["projected_features"]
        pooled = projected.mean(dim=1)
        scores = torch.softmax(output.logits, dim=-1)
    for handle in handles:
        handle.remove()

    outputs = {
        "input_pcm": torch.from_numpy(pcm),
        "normalized_pcm": batch["input_values"],
        "encoder_features": captured["encoder_features"],
        "projected_features": projected,
        "pooled_embedding": pooled,
        "logits": output.logits,
        "scores": scores,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_outputs: dict[str, dict[str, Any]] = {}
    for name, value in outputs.items():
        array = value.detach().cpu().numpy().astype("<f4", copy=False)
        destination = args.output_dir / f"{name}.f32"
        array.tofile(destination)
        manifest_outputs[name] = {
            "shape": list(array.shape),
            "elements": int(array.size),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    manifest = {
        "oracle": "official Transformers Wav2Vec2ForSequenceClassification imported without reimplementation",
        "transformers_version": transformers.__version__,
        "entrypoint": "transformers.Wav2Vec2ForSequenceClassification.forward",
        "checkpoint_repo": UPSTREAM_REPO,
        "checkpoint_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "preprocessor_sha256": PREPROCESSOR_SHA256,
        "sample_rate": SAMPLE_RATE,
        "signal_sha256": SIGNAL_SHA256,
        "labels": LABELS,
        "inference": {
            "feature_extractor": "official Wav2Vec2FeatureExtractor",
            "attention_mask": False,
            "pooling": "official mean over projected time axis",
            "classifier": "official model.classifier",
            "mode": "eval + torch.inference_mode",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
        },
        "outputs": manifest_outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"deepfake reference: {len(outputs)} tensors -> {args.output_dir}")


def self_test() -> None:
    payload = signal_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != SAMPLES * 4:
        raise AssertionError(f"signal bytes={len(payload)}, expected {SAMPLES * 4}")
    if actual != SIGNAL_SHA256:
        raise AssertionError(f"signal SHA-256 {actual}, expected {SIGNAL_SHA256}")
    if LABELS != ["fake", "real"] or len(UPSTREAM_REVISION) != 40:
        raise AssertionError("pinned immutable classifier contract drifted")
    print(f"deepfake_detection_dump_reference: self-test PASS sha256={actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.input_dir is None or args.output_dir is None):
        parser.error("--input-dir and --output-dir are required")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
