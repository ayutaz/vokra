#!/usr/bin/env python3
"""Dump independent FunASR emotion2vec+ Large reference tensors.

This oracle imports the official FunASR implementation from immutable commit
``2f7dcbad90e82e964ab381ad63ff5109dd92327d``.  It does not reimplement the
network.  Hooks capture the official ConvFeatureExtractionModel output,
projected local features, context encoder output, final features, utterance
mean, classifier logits, and softmax scores for the official example WAV.

Run through ``uv run --project tools/parity --python 3.12`` on vast.ai; the
checkpoint plus prepared GGUF exceeds the local aggregate-artifact threshold.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

FUNASR_REPO = "https://github.com/modelscope/FunASR.git"
FUNASR_REVISION = "2f7dcbad90e82e964ab381ad63ff5109dd92327d"
UPSTREAM_REPO = "emotion2vec/emotion2vec_plus_large"
UPSTREAM_REVISION = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
CHECKPOINT_BYTES = 1_945_790_254
CHECKPOINT_SHA256 = (
    "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4"
)
BLOCKED_UNSAFE_PICKLE = "BLOCKED_UNSAFE_PICKLE"
CONFIG_BYTES = 5_552
CONFIG_SHA256 = "f4fa0eb82cc78bfebb43c56d68791afb01788085a18897d20999af7bc45d51d3"
TOKENS_BYTES = 119
TOKENS_SHA256 = "866121e470057b847d7a50e9923509141fb2924392f53385a186482a1ec0fb7f"
EXAMPLE_BYTES = 131_376
EXAMPLE_SHA256 = "a4839eaaa3d54bd2db6eb48aa3d40def1b5c5004df3fd163a8dcd045097f8a23"
SAMPLE_RATE = 16_000
LABELS = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_hash: str, label: str) -> None:
    if path.stat().st_size != size:
        sys.exit(f"{label}: {path} has {path.stat().st_size} bytes, expected {size}")
    actual = sha256_file(path)
    if actual != expected_hash:
        sys.exit(f"{label}: {path} SHA-256 {actual}, expected {expected_hash}")


def verify_funasr_checkout(path: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != FUNASR_REVISION:
        sys.exit(
            f"FunASR checkout is {revision}, expected immutable {FUNASR_REVISION}"
        )


def tensor_value(value: Any):
    if isinstance(value, tuple):
        value = value[0]
    return value.detach().to(device="cpu", dtype=value.dtype).contiguous()


def _assert_checkpoint_loader_contract() -> None:
    """Keep checkpoint deserialization restricted against future regressions."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    load_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "load"
                and isinstance(function.value, ast.Name)
                and function.value.id == "torch"
            ):
                load_calls += 1
                weights_only = [
                    keyword
                    for keyword in node.keywords
                    if keyword.arg == "weights_only"
                ]
                assert len(weights_only) == 1
                assert isinstance(weights_only[0].value, ast.Constant)
                assert weights_only[0].value.value is True
            if (
                isinstance(function, ast.Name)
                and function.id == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "torch"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "load"
            ):
                raise AssertionError("torch.load monkeypatch is forbidden")
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "object"
                and isinstance(function.value, ast.Name)
                and function.value.id == "patch"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "torch"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "load"
            ):
                raise AssertionError("torch.load monkeypatch is forbidden")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "load"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "torch"
                ):
                    raise AssertionError("torch.load monkeypatch is forbidden")
        if isinstance(node, ast.ImportFrom):
            assert all(alias.name != "Unpickler" for alias in node.names)
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            assert not name.endswith("Unpickler")
    assert load_calls > 0


def dump(args: argparse.Namespace) -> None:
    verify_funasr_checkout(args.funasr_source)
    verify_file(args.checkpoint, CHECKPOINT_BYTES, CHECKPOINT_SHA256, "checkpoint")
    verify_file(args.config, CONFIG_BYTES, CONFIG_SHA256, "config")
    verify_file(args.tokens, TOKENS_BYTES, TOKENS_SHA256, "tokens")
    verify_file(args.wav, EXAMPLE_BYTES, EXAMPLE_SHA256, "official example WAV")

    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torch.nn.functional as functional
        from omegaconf import OmegaConf
    except ImportError as error:
        sys.exit(f"missing parity dependency: {error}")

    sys.path.insert(0, str(args.funasr_source))
    try:
        from funasr.models.emotion2vec.model import Emotion2vec
    except ImportError as error:
        sys.exit(f"cannot import official FunASR Emotion2vec: {error}")

    token_list = args.tokens.read_text(encoding="utf-8").splitlines()
    if token_list != LABELS:
        sys.exit(f"official token order changed: {token_list!r}")
    config = OmegaConf.load(args.config)
    model = Emotion2vec(
        model_conf=OmegaConf.to_container(config.model_conf, resolve=True),
        vocab_size=len(token_list),
    )
    try:
        raw = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - fail-closed deserialization boundary
        raise RuntimeError(
            f"{BLOCKED_UNSAFE_PICKLE}: emotion2vec checkpoint requires pickle "
            "globals outside PyTorch's restricted weights_only loader; unsafe "
            "fallback is forbidden"
        ) from error
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        sys.exit("checkpoint must contain a dict-valued top-level 'model'")
    incompatible = model.load_state_dict(raw["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        sys.exit(f"strict load mismatch: {incompatible}")
    model.eval()

    captured: dict[str, Any] = {}
    audio = model.modality_encoders["AUDIO"]

    def hook(name: str):
        def capture(_module, _inputs, output):
            captured[name] = tensor_value(output)

        return capture

    handles = [
        audio.local_encoder.register_forward_hook(hook("conv_features")),
        audio.project_features.register_forward_hook(hook("projected_features")),
        audio.context_encoder.register_forward_hook(hook("context_features")),
    ]
    waveform, sample_rate = sf.read(args.wav, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
        sys.exit(
            f"official WAV must be mono {SAMPLE_RATE} Hz, got shape={waveform.shape}, rate={sample_rate}"
        )
    source = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    normalized = functional.layer_norm(source, source.shape)
    with torch.inference_mode():
        result = model.extract_features(normalized.view(1, -1), padding_mask=None)
        final_features = result["x"]
        pooled = final_features.mean(dim=1)
        logits = model.proj(pooled)
        scores = torch.softmax(logits, dim=-1)
    for handle in handles:
        handle.remove()

    outputs = {
        "normalized_pcm": normalized,
        **captured,
        "final_features": final_features,
        "pooled_embedding": pooled,
        "logits": logits,
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
        "oracle": "official FunASR Emotion2vec imported without reimplementation",
        "funasr_repo": FUNASR_REPO,
        "funasr_revision": FUNASR_REVISION,
        "entrypoint": "funasr.models.emotion2vec.model.Emotion2vec.extract_features",
        "checkpoint_repo": UPSTREAM_REPO,
        "checkpoint_revision": UPSTREAM_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "tokens_sha256": TOKENS_SHA256,
        "wav_sha256": EXAMPLE_SHA256,
        "sample_rate": SAMPLE_RATE,
        "labels": LABELS,
        "inference": {
            "normalization": "torch.nn.functional.layer_norm(source, source.shape), eps=1e-5",
            "mask": False,
            "remove_extra_tokens": True,
            "pooling": "mean over final time axis",
            "classifier": "official model.proj followed by torch.softmax",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "outputs": manifest_outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"emotion2vec reference: {len(outputs)} tensors -> {args.output_dir}")


def self_test() -> None:
    assert len(LABELS) == 9
    assert LABELS[0] == "生气/angry"
    assert LABELS[-1] == "<unk>"
    assert len(FUNASR_REVISION) == 40
    assert len(CHECKPOINT_SHA256) == 64
    assert BLOCKED_UNSAFE_PICKLE == "BLOCKED_UNSAFE_PICKLE"
    _assert_checkpoint_loader_contract()
    print("emotion2vec_dump_reference: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funasr-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    required = [
        args.funasr_source,
        args.checkpoint,
        args.config,
        args.tokens,
        args.wav,
        args.output_dir,
    ]
    if not args.self_test and any(value is None for value in required):
        parser.error(
            "--funasr-source, --checkpoint, --config, --tokens, --wav, and --output-dir are required"
        )
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
