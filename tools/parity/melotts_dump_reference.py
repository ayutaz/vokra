#!/usr/bin/env python3
"""Dump an independent official-MeloTTS acoustic-core reference fixture.

The oracle is MyShell's pinned ``myshell-ai/MeloTTS`` source tree plus one of
the five official language checkpoints. Vokra, its GGUF, and its Rust
implementation are never imported. Raw-text normalization, G2P, and BERT
tokenization are kept outside this fixture: deterministic already-expanded
BERT features isolate the released acoustic graph from language-frontend
dependencies.

Run through the repository Python 3.12 environment::

    uv run --project tools/parity python tools/parity/melotts_dump_reference.py \
      --source-root /path/to/myshell-ai-MeloTTS-at-2091453 \
      --variant english \
      --output crates/vokra-models/tests/fixtures/melotts_english
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


SOURCE_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
SYMBOLS = ("_", "hh", "ah", "l", "ow", "_")
TONES = (7, 7, 8, 7, 9, 7)
LANGUAGE_IDS = (2, 2, 2, 2, 2, 2)
LENGTH_SCALE = 0.05


@dataclass(frozen=True)
class VariantSpec:
    slug: str
    upstream_hf: str
    upstream_revision: str
    checkpoint_sha256: str
    speaker_id: int


VARIANTS = {
    "english": VariantSpec(
        "english",
        "myshell-ai/MeloTTS-English",
        "bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef",
        "acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c",
        0,
    ),
    "chinese": VariantSpec(
        "chinese",
        "myshell-ai/MeloTTS-Chinese",
        "af5d207a364ea4208c6f589c89f57f88414bdd16",
        "a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093",
        1,
    ),
    "korean": VariantSpec(
        "korean",
        "myshell-ai/MeloTTS-Korean",
        "0207e5adfc90129a51b6b03d89be6d84360ed323",
        "48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0",
        0,
    ),
    "spanish": VariantSpec(
        "spanish",
        "myshell-ai/MeloTTS-Spanish",
        "dbb5496df39d11a66c1d5f5a9ca357c3c9fb95fb",
        "9077a7e7e5fd8e42f3f922641c401f1936971c08465a3e7ccb19d57a659e72ae",
        0,
    ),
    "japanese": VariantSpec(
        "japanese",
        "myshell-ai/MeloTTS-Japanese",
        "367f8795464b531b4e97c1515bddfc1243e60891",
        "96ae783e6ec0177aa810e2a645aec5d136a6f4992fdea26ee92b7b04d9688ad0",
        0,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_source_revision(source_root: Path) -> None:
    models = source_root / "melo" / "models.py"
    commons = source_root / "melo" / "commons.py"
    if not models.is_file() or not commons.is_file():
        raise RuntimeError(
            f"{source_root} is not a MeloTTS source tree (missing melo/models.py "
            "or melo/commons.py)"
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot verify MeloTTS source revision under {source_root}: {error}"
        ) from error
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"MeloTTS source revision {revision}, expected {SOURCE_REVISION}"
        )


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes(order="C"))


def write_u32(path: Path, values: list[int]) -> None:
    path.write_bytes(np.asarray(values, dtype="<u4").tobytes(order="C"))


def write_i32(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().to(torch.int32).contiguous().numpy()
    path.write_bytes(np.asarray(array, dtype="<i4").tobytes(order="C"))


def position_major(tensor: torch.Tensor) -> torch.Tensor:
    """Convert official ``[1, channels, time]`` into ``[time, channels]``."""

    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"expected [1, channels, time], got {tuple(tensor.shape)}")
    return tensor[0].transpose(0, 1).contiguous()


def load_model(source_root: Path, config_path: Path, checkpoint_path: Path):
    sys.path.insert(0, str(source_root))
    from melo.models import SynthesizerTrn  # type: ignore[import-not-found]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    train = config["train"]
    model_config = config["model"]
    model = SynthesizerTrn(
        len(config["symbols"]),
        data["filter_length"] // 2 + 1,
        train["segment_size"] // data["hop_length"],
        n_speakers=data["n_speakers"],
        num_tones=config["num_tones"],
        num_languages=config["num_languages"],
        **model_config,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("official checkpoint has no dictionary-valued `model` entry")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="english")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = VARIANTS[args.variant]

    source_root = args.source_root.resolve()
    require_source_revision(source_root)
    checkpoint_path = args.checkpoint or Path(
        hf_hub_download(
            repo_id=spec.upstream_hf,
            filename="checkpoint.pth",
            revision=spec.upstream_revision,
        )
    )
    config_path = args.config or Path(
        hf_hub_download(
            repo_id=spec.upstream_hf,
            filename="config.json",
            revision=spec.upstream_revision,
        )
    )
    if sha256(checkpoint_path) != spec.checkpoint_sha256:
        raise RuntimeError(
            f"official checkpoint SHA-256 mismatch: {sha256(checkpoint_path)} != "
            f"{spec.checkpoint_sha256}"
        )

    torch.set_num_threads(1)
    torch.manual_seed(0x4D454C4F)
    torch.use_deterministic_algorithms(True)
    model, config = load_model(source_root, config_path, checkpoint_path)

    symbol_to_id = {symbol: index for index, symbol in enumerate(config["symbols"])}
    phoneme_ids = [symbol_to_id[symbol] for symbol in SYMBOLS]
    sequence_len = len(phoneme_ids)
    if len(TONES) != sequence_len or len(LANGUAGE_IDS) != sequence_len:
        raise RuntimeError("fixture id axes have inconsistent lengths")

    phones = torch.tensor([phoneme_ids], dtype=torch.long)
    tones = torch.tensor([TONES], dtype=torch.long)
    languages = torch.tensor([LANGUAGE_IDS], dtype=torch.long)
    lengths = torch.tensor([sequence_len], dtype=torch.long)
    speaker = torch.tensor([spec.speaker_id], dtype=torch.long)
    bert = torch.zeros((1, 1024, sequence_len), dtype=torch.float32)
    positions = torch.arange(sequence_len, dtype=torch.float32).reshape(1, 1, -1) + 1
    dimensions = torch.arange(768, dtype=torch.float32).reshape(1, -1, 1) + 1
    ja_bert = (
        0.05 * torch.sin(positions * dimensions * 0.001)
        + 0.02 * torch.cos((positions + dimensions) * 0.007)
    ).contiguous()

    with torch.inference_mode():
        global_conditioning = model.emb_g(speaker).unsqueeze(-1)
        hidden, mean, log_scale, text_mask = model.enc_p(
            phones,
            lengths,
            tones,
            languages,
            bert,
            ja_bert,
            g=global_conditioning,
        )
        log_duration = model.dp(hidden, text_mask, g=global_conditioning)
        duration = torch.ceil(torch.exp(log_duration) * text_mask * LENGTH_SCALE)
        frame_lengths = torch.clamp_min(torch.sum(duration, dim=(1, 2)), 1).long()

        # Use the official source implementation for the duration path and
        # prior expansion.  Importing from the pinned tree keeps the oracle
        # independent from Vokra's length regulator.
        from melo import commons  # type: ignore[import-not-found]

        frame_mask = torch.unsqueeze(
            commons.sequence_mask(frame_lengths, None), 1
        ).to(text_mask.dtype)
        attention_mask = torch.unsqueeze(text_mask, 2) * torch.unsqueeze(
            frame_mask, -1
        )
        attention = commons.generate_path(duration, attention_mask)
        expanded_mean = torch.matmul(
            attention.squeeze(1), mean.transpose(1, 2)
        ).transpose(1, 2)
        expanded_log_scale = torch.matmul(
            attention.squeeze(1), log_scale.transpose(1, 2)
        ).transpose(1, 2)
        prior = expanded_mean
        decoder_latent = model.flow(
            prior, frame_mask, g=global_conditioning, reverse=True
        )
        pcm = model.dec(decoder_latent * frame_mask, g=global_conditioning)

    frame_count = int(frame_lengths.item())
    if frame_count <= 0 or pcm.shape != (1, 1, frame_count * 512):
        raise RuntimeError(
            f"unexpected frame/PCM geometry: frames={frame_count}, pcm={tuple(pcm.shape)}"
        )
    for label, tensor in {
        "hidden": hidden,
        "mean": mean,
        "log_scale": log_scale,
        "log_duration": log_duration,
        "expanded_mean": expanded_mean,
        "expanded_log_scale": expanded_log_scale,
        "decoder_latent": decoder_latent,
        "pcm": pcm,
    }.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"official {label} contains a non-finite value")

    args.output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit_f32(name: str, tensor: torch.Tensor) -> None:
        path = args.output / name
        write_f32(path, tensor)
        written.append(path)

    def emit_u32(name: str, values: list[int]) -> None:
        path = args.output / name
        write_u32(path, values)
        written.append(path)

    emit_u32("phoneme_ids.u32", phoneme_ids)
    emit_u32("tones.u32", list(TONES))
    emit_u32("language_ids.u32", list(LANGUAGE_IDS))
    emit_f32("bert_position_major.f32", bert[0].transpose(0, 1))
    emit_f32("ja_bert_position_major.f32", ja_bert[0].transpose(0, 1))
    emit_f32("speaker_conditioning.f32", global_conditioning.reshape(-1))
    emit_f32("hidden_position_major.f32", position_major(hidden))
    emit_f32("mean_position_major.f32", position_major(mean))
    emit_f32("log_scale_position_major.f32", position_major(log_scale))
    emit_f32("log_duration.f32", log_duration.reshape(-1))
    duration_path = args.output / "durations.i32"
    write_i32(duration_path, duration.reshape(-1))
    written.append(duration_path)
    emit_f32("expanded_mean_position_major.f32", position_major(expanded_mean))
    emit_f32(
        "expanded_log_scale_position_major.f32", position_major(expanded_log_scale)
    )
    emit_f32("decoder_latent_position_major.f32", position_major(decoder_latent))
    emit_f32("pcm.f32", pcm.reshape(-1))

    features_path = args.output / "features.vmf"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("melotts_pack_features.py")),
            "--variant",
            spec.slug,
            "--speaker-id",
            str(spec.speaker_id),
            "--phoneme-ids",
            str(args.output / "phoneme_ids.u32"),
            "--tones",
            str(args.output / "tones.u32"),
            "--language-ids",
            str(args.output / "language_ids.u32"),
            "--bert",
            str(args.output / "bert_position_major.f32"),
            "--ja-bert",
            str(args.output / "ja_bert_position_major.f32"),
            "--output",
            str(features_path),
        ],
        check=True,
    )
    written.append(features_path)

    manifest = {
        "format": "vokra-melotts-reference-v1",
        "oracle": "myshell-ai/MeloTTS official PyTorch acoustic modules",
        "source_revision": SOURCE_REVISION,
        "variant": spec.slug,
        "upstream_hf": spec.upstream_hf,
        "upstream_revision": spec.upstream_revision,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "config_sha256": sha256(config_path),
        "torch": torch.__version__,
        "symbols": list(SYMBOLS),
        "speaker_id": spec.speaker_id,
        "length_scale": LENGTH_SCALE,
        "sequence_len": sequence_len,
        "frame_count": frame_count,
        "pcm_samples": int(pcm.numel()),
        "files": {path.name: sha256(path) for path in sorted(written)},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
