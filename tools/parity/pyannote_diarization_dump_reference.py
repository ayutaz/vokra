#!/usr/bin/env python3
"""Dump an independent official pyannote.audio 3.1.1 diarization reference.

The oracle imports ``SpeakerDiarization.apply`` from the exact upstream source
checkout at ``6a972c0c4e95de04637d7221208736c64c8b972a``. It restores the two
immutable public Vokra weight GGUFs into the official PyanNet and WeSpeaker
classes, then runs the official pipeline on the first six seconds of the
committed Public Domain JFK fixture. No diarization, clustering, pooling, or
reconstruction equation is mirrored here.

The real run belongs on VAST through
``scripts/publish/vast-ai/run-pyannote-diarization-parity.sh``. ``--self-test``
uses only the Python standard library and is safe on the maintainer Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


PYANNOTE_AUDIO_REPO = "https://github.com/pyannote/pyannote-audio.git"
PYANNOTE_AUDIO_REVISION = "6a972c0c4e95de04637d7221208736c64c8b972a"
PYANNOTE_AUDIO_VERSION = "3.1.1"

PIPELINE_REPO = "vokra/pyannote-speaker-diarization-3.1"
PIPELINE_REVISION = "a2bc759121b1cf64d3fc669be9785af963eb54b4"
PIPELINE_FILE = "pyannote-speaker-diarization-3.1.gguf"
PIPELINE_BYTES = 1_728
PIPELINE_SHA256 = "6f2fe6d681d75fdde84768792f54725baf4e5e025f3a9c4af9618867a64e3a64"

SEGMENTATION_REPO = "vokra/pyannote-segmentation-3.0"
SEGMENTATION_REVISION = "50bf4e510e0c689668384aec0f866f02e0fcaea8"
SEGMENTATION_FILE = "pyannote-seg.gguf"
SEGMENTATION_BYTES = 5_898_272
SEGMENTATION_SHA256 = "22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1"
SEGMENTATION_TENSORS = 54

EMBEDDING_REPO = "vokra/pyannote-wespeaker-voxceleb-resnet34-lm"
EMBEDDING_REVISION = "8e27acd8a875088f1a7321f40610397bf964a446"
EMBEDDING_FILE = "pyannote-wespeaker.restamped.gguf"
EMBEDDING_BYTES = 26_584_064
EMBEDDING_SHA256 = "6dccbc026e9c32a8f99f3441e64f1ff52e36afb055442595c86cda8021c78c39"
EMBEDDING_TENSORS = 182

JFK_BYTES = 352_078
JFK_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
SAMPLE_RATE = 16_000
CLIP_SAMPLES = 96_000
CLIP_F32_SHA256 = "9be7750d40f483b720348391824ce3373a4aa8086079bc1e5aa47e118b87b472"
EXPECTED_CHUNKS = 3
EXPECTED_SEGMENTATION_FRAMES = 293
LOCAL_SPEAKERS = 3
EMBEDDING_DIM = 256

SOURCE_FILES = {
    "version.txt": (
        6,
        "6dc3b2dc3139b7d421b6fc26a7f853389a9b035ea4f2e9047b9516050dd6ac73",
    ),
    "LICENSE": (
        1_061,
        "a3b53644a76e70e289b25271b119c0a1aadaaf0db7a16225fb494fdc0e36c32a",
    ),
    "pyannote/audio/models/segmentation/PyanNet.py": (
        6_650,
        "8d576a0992d56f23f1c065e1ee211747649e3fd1494eee50e994aec7856f09ff",
    ),
    "pyannote/audio/models/blocks/sincnet.py": (
        3_383,
        "cb54c32a5e7965b2c068dedf9314168cf79de7e1f45f92740d380dee8e56db03",
    ),
    "pyannote/audio/core/model.py": (
        25_328,
        "e2f019a1f083db8c5a4956238c6f4e05dcda5a9ccfcd2343a926df88a54b951d",
    ),
    "pyannote/audio/core/task.py": (
        17_526,
        "a5903cd9e1e16ec96267a4b5ebe6d8786fec8a1ebad246fb84eca5a8094c47e2",
    ),
    "pyannote/audio/core/inference.py": (
        31_614,
        "e198c9bb964edaef0cea0e300fbc7b2ee9fd0ac331738441a44a16dbef8c5610",
    ),
    "pyannote/audio/core/pipeline.py": (
        12_375,
        "ff178646940a6e1cdcd79085df60b92c46bc3bc2ea2cb4610d59910123764970",
    ),
    "pyannote/audio/utils/powerset.py": (
        5_226,
        "7bf78a678197b3f48ca686be14e757af3a3e884a52817b29a35de036123b737c",
    ),
    "pyannote/audio/utils/signal.py": (
        11_524,
        "a19f2ef7c02f4fb187584bc475ec81b00d294a6d0ed9bed1d325639c72682ed5",
    ),
    "pyannote/audio/models/blocks/pooling.py": (
        4_720,
        "fb1fed1dd1157d8d3885c8a0c8b0cafad67850ea479a7e0879e453d86bcf0528",
    ),
    "pyannote/audio/models/embedding/wespeaker/resnet.py": (
        9_320,
        "a0a051280f7b9c110873dae17ecdcfce0c35173c858a3d0f6ad24c49f25b6f4d",
    ),
    "pyannote/audio/models/embedding/wespeaker/__init__.py": (
        7_335,
        "0881a5aa74b665b65b8d9d0b3a34ffc2487f6455cc85589227213271e271a90c",
    ),
    "pyannote/audio/pipelines/speaker_verification.py": (
        28_708,
        "7e1bf7fa41ef064e1fae556ca6fd032ce2369bddcf919ac79a28cf08e55839fd",
    ),
    "pyannote/audio/pipelines/clustering.py": (
        21_260,
        "260d739aa9b444633c09d6ea03c7dd92fc6dc31289ddf86547a77577d8a204c6",
    ),
    "pyannote/audio/pipelines/speaker_diarization.py": (
        24_973,
        "142e95702bf9aea9d520d8e0104a42d69b05034b915c860b2a948130b70a4026",
    ),
    "pyannote/audio/pipelines/utils/diarization.py": (
        8_697,
        "4ebb0eebfa3a713b56bc11178a48402c83fb6ff28f80f1ad978bba1fdbc48a5f",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, size: int, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label}: missing {path}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"{label}: {actual_size} bytes != expected {size}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label}: SHA-256 {actual_hash} != expected {expected_hash}"
        )


def verify_source_checkout(source: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PYANNOTE_AUDIO_REVISION:
        raise RuntimeError(
            f"pyannote.audio checkout is {revision}, expected {PYANNOTE_AUDIO_REVISION}"
        )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"pyannote.audio checkout is dirty:\n{dirty}")
    for relative, (size, expected_hash) in SOURCE_FILES.items():
        verify_file(source / relative, size, expected_hash, f"official source {relative}")
    version = (source / "version.txt").read_text(encoding="utf-8").strip()
    if version != PYANNOTE_AUDIO_VERSION:
        raise RuntimeError(
            f"official source version is {version}, expected {PYANNOTE_AUDIO_VERSION}"
        )


def read_jfk_clip(path: Path) -> tuple[bytes, bytes]:
    verify_file(path, JFK_BYTES, JFK_SHA256, "Public Domain JFK WAV")
    with wave.open(str(path), "rb") as reader:
        actual = (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.getcomptype(),
        )
        expected = (1, 2, SAMPLE_RATE, "NONE")
        if actual != expected:
            raise RuntimeError(f"JFK WAV format {actual!r} != expected {expected!r}")
        pcm16 = reader.readframes(CLIP_SAMPLES)
    if len(pcm16) != CLIP_SAMPLES * 2:
        raise RuntimeError(
            f"JFK clip has {len(pcm16) // 2} samples, expected {CLIP_SAMPLES}"
        )
    values = struct.unpack(f"<{CLIP_SAMPLES}h", pcm16)
    pcm_f32 = struct.pack(f"<{CLIP_SAMPLES}f", *(value / 32768.0 for value in values))
    actual_hash = hashlib.sha256(pcm_f32).hexdigest()
    if actual_hash != CLIP_F32_SHA256:
        raise RuntimeError(
            f"six-second f32 clip SHA-256 {actual_hash} != {CLIP_F32_SHA256}"
        )
    return pcm16, pcm_f32


def write_clip_wav(path: Path, pcm16: bytes) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm16)


def gguf_state(path: Path, expected_state: Any, expected_count: int, np: Any, torch: Any):
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise RuntimeError(f"missing locked parity dependency: {error}") from error

    reader = GGUFReader(str(path))
    by_name = {item.name: item for item in reader.tensors}
    if len(by_name) != len(reader.tensors):
        raise RuntimeError(f"{path}: duplicate tensor names")
    if len(by_name) != expected_count:
        raise RuntimeError(f"{path}: {len(by_name)} tensors != expected {expected_count}")
    expected_names = set(expected_state)
    actual_names = set(by_name)
    if actual_names != expected_names:
        raise RuntimeError(
            f"{path}: official/GGUF names differ: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    restored = {}
    for name, target in expected_state.items():
        item = by_name[name]
        if int(item.tensor_type) != 0:
            raise RuntimeError(f"{path}: {name} is not canonical F32")
        values = item.data.copy().reshape(-1).astype(np.float32, copy=False)
        expected_elements = int(target.numel())
        if values.size != expected_elements:
            raise RuntimeError(
                f"{path}: {name} has {values.size} values, expected {expected_elements}"
            )
        restored[name] = torch.from_numpy(
            values.reshape(tuple(target.shape)).copy()
        )
    return restored


def build_official_pipeline(source: Path, segmentation_gguf: Path, embedding_gguf: Path):
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(f"missing locked parity dependency: {error}") from error

    sys.path.insert(0, str(source))
    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
        from pyannote.audio.models.segmentation.PyanNet import PyanNet
        from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    except ImportError as error:
        raise RuntimeError(f"cannot import official pyannote.audio source: {error}") from error

    for symbol, relative in [
        (PyanNet, "pyannote/audio/models/segmentation/PyanNet.py"),
        (
            WeSpeakerResNet34,
            "pyannote/audio/models/embedding/wespeaker/__init__.py",
        ),
        (
            SpeakerDiarization,
            "pyannote/audio/pipelines/speaker_diarization.py",
        ),
    ]:
        imported = Path(inspect.getsourcefile(symbol) or "").resolve()
        expected = (source / relative).resolve()
        if imported != expected:
            raise RuntimeError(f"{symbol.__name__} imported from {imported}, expected {expected}")
    installed_version = importlib.metadata.version("pyannote.audio")
    if installed_version != PYANNOTE_AUDIO_VERSION:
        raise RuntimeError(
            f"installed pyannote.audio is {installed_version}, expected {PYANNOTE_AUDIO_VERSION}"
        )

    segmentation = PyanNet(
        sincnet={"stride": 10},
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.0,
        },
        linear={"hidden_size": 128, "num_layers": 2},
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )
    segmentation.specifications = Specifications(
        problem=Problem.MONO_LABEL_CLASSIFICATION,
        resolution=Resolution.FRAME,
        duration=5.0,
        classes=["speaker#1", "speaker#2", "speaker#3"],
        powerset_max_classes=2,
        permutation_invariant=True,
    )
    segmentation.build()
    restored_segmentation = gguf_state(
        segmentation_gguf,
        segmentation.state_dict(),
        SEGMENTATION_TENSORS,
        np,
        torch,
    )
    incompatible = segmentation.load_state_dict(restored_segmentation, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict official PyanNet state mismatch: {incompatible}")
    segmentation.eval()

    embedding = WeSpeakerResNet34()
    embedding.specifications = Specifications(
        problem=Problem.REPRESENTATION,
        resolution=Resolution.CHUNK,
        duration=5.0,
    )
    restored_embedding = gguf_state(
        embedding_gguf,
        embedding.state_dict(),
        EMBEDDING_TENSORS,
        np,
        torch,
    )
    incompatible = embedding.load_state_dict(restored_embedding, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict official WeSpeaker state mismatch: {incompatible}")
    embedding.eval()

    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        segmentation_step=0.1,
        embedding=embedding,
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
        embedding_batch_size=32,
        segmentation_batch_size=32,
    )
    pipeline.instantiate(
        {
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": 0.7045654963945799,
            },
        }
    )
    return pipeline, np, torch


def write_array(output: Path, name: str, value: Any, dtype: str, np: Any):
    array = np.asarray(value, dtype=dtype)
    destination = output / name
    array.tofile(destination)
    return {
        "shape": list(array.shape),
        "elements": int(array.size),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def dump(args: argparse.Namespace) -> None:
    verify_source_checkout(args.pyannote_source)
    verify_file(args.pipeline_gguf, PIPELINE_BYTES, PIPELINE_SHA256, "pipeline GGUF")
    verify_file(
        args.segmentation_gguf,
        SEGMENTATION_BYTES,
        SEGMENTATION_SHA256,
        "segmentation GGUF",
    )
    verify_file(
        args.embedding_gguf,
        EMBEDDING_BYTES,
        EMBEDDING_SHA256,
        "embedding GGUF",
    )
    pcm16, pcm_f32 = read_jfk_clip(args.wav)
    pipeline, np, torch = build_official_pipeline(
        args.pyannote_source, args.segmentation_gguf, args.embedding_gguf
    )

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    pcm = np.frombuffer(pcm_f32, dtype="<f4").copy()
    waveform = torch.from_numpy(pcm.copy()).reshape(1, -1)
    captured: dict[str, Any] = {}

    def hook(step_name: str, artifact: Any, **_kwargs: Any) -> None:
        if artifact is None:
            return
        if step_name in {"segmentation", "speaker_counting", "discrete_diarization"}:
            captured[step_name] = np.asarray(artifact.data).copy()
        elif step_name == "embeddings":
            values = np.asarray(artifact)
            if values.ndim == 3:
                captured[step_name] = values.copy()

    with torch.inference_mode():
        annotation = pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE, "uri": "jfk-6s"},
            hook=hook,
        )

    required = {
        "segmentation",
        "speaker_counting",
        "embeddings",
        "discrete_diarization",
    }
    if set(captured) != required:
        raise RuntimeError(
            f"official pipeline hooks changed: got {sorted(captured)}, expected {sorted(required)}"
        )
    if tuple(captured["segmentation"].shape) != (
        EXPECTED_CHUNKS,
        EXPECTED_SEGMENTATION_FRAMES,
        LOCAL_SPEAKERS,
    ):
        raise RuntimeError(
            f"official segmentation shape changed: {captured['segmentation'].shape}"
        )
    if tuple(captured["embeddings"].shape) != (
        EXPECTED_CHUNKS,
        LOCAL_SPEAKERS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(f"official embedding shape changed: {captured['embeddings'].shape}")
    for name, values in captured.items():
        if not bool(np.isfinite(values).all()):
            raise RuntimeError(f"official {name} contains non-finite values")

    segments: list[tuple[float, float, int]] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        prefix = "SPEAKER_"
        if not isinstance(label, str) or not label.startswith(prefix):
            raise RuntimeError(f"official annotation label is not canonical: {label!r}")
        speaker = int(label[len(prefix) :])
        segments.append((float(segment.start), float(segment.duration), speaker))
    segments.sort(key=lambda item: (item[0], item[2]))
    if not segments:
        raise RuntimeError("official pipeline emitted no speaker segment for the JFK clip")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input_pcm.f32").write_bytes(pcm_f32)
    write_clip_wav(args.output_dir / "input.wav", pcm16)
    segment_times = [value for start, duration, _ in segments for value in (start, duration)]
    segment_speakers = [speaker for _, _, speaker in segments]
    outputs = {
        "input_pcm.f32": {
            "shape": [CLIP_SAMPLES],
            "elements": CLIP_SAMPLES,
            "bytes": len(pcm_f32),
            "sha256": hashlib.sha256(pcm_f32).hexdigest(),
        },
        "segmentation.f32": write_array(
            args.output_dir,
            "segmentation.f32",
            captured["segmentation"],
            "<f4",
            np,
        ),
        "speaker_count.f32": write_array(
            args.output_dir,
            "speaker_count.f32",
            captured["speaker_counting"],
            "<f4",
            np,
        ),
        "embeddings.f32": write_array(
            args.output_dir,
            "embeddings.f32",
            captured["embeddings"],
            "<f4",
            np,
        ),
        "discrete_diarization.f32": write_array(
            args.output_dir,
            "discrete_diarization.f32",
            captured["discrete_diarization"],
            "<f4",
            np,
        ),
        "segments.f32": write_array(
            args.output_dir, "segments.f32", segment_times, "<f4", np
        ),
        "segment_speakers.u32": write_array(
            args.output_dir,
            "segment_speakers.u32",
            segment_speakers,
            "<u4",
            np,
        ),
    }
    outputs["input.wav"] = {
        "bytes": (args.output_dir / "input.wav").stat().st_size,
        "sha256": sha256_file(args.output_dir / "input.wav"),
    }
    manifest = {
        "format": "vokra-pyannote-diarization-reference-v1",
        "oracle": "official pyannote.audio 3.1.1 SpeakerDiarization.apply imported without reimplementation",
        "source_repo": PYANNOTE_AUDIO_REPO,
        "source_revision": PYANNOTE_AUDIO_REVISION,
        "source_files": {
            path: {"bytes": size, "sha256": digest}
            for path, (size, digest) in SOURCE_FILES.items()
        },
        "artifacts": {
            "pipeline": {
                "repo": PIPELINE_REPO,
                "revision": PIPELINE_REVISION,
                "file": PIPELINE_FILE,
                "bytes": PIPELINE_BYTES,
                "sha256": PIPELINE_SHA256,
            },
            "segmentation": {
                "repo": SEGMENTATION_REPO,
                "revision": SEGMENTATION_REVISION,
                "file": SEGMENTATION_FILE,
                "bytes": SEGMENTATION_BYTES,
                "sha256": SEGMENTATION_SHA256,
            },
            "embedding": {
                "repo": EMBEDDING_REPO,
                "revision": EMBEDDING_REVISION,
                "file": EMBEDDING_FILE,
                "bytes": EMBEDDING_BYTES,
                "sha256": EMBEDDING_SHA256,
            },
        },
        "input": {
            "source": "tests/fixtures/audio/jfk-30s.wav first six seconds",
            "source_sha256": JFK_SHA256,
            "sample_rate": SAMPLE_RATE,
            "samples": CLIP_SAMPLES,
            "pcm_f32_sha256": CLIP_F32_SHA256,
        },
        "pipeline_parameters": {
            "segmentation_step": 0.1,
            "segmentation_batch_size": 32,
            "embedding_batch_size": 32,
            "embedding_exclude_overlap": True,
            "segmentation_min_duration_off": 0.0,
            "clustering_method": "centroid",
            "clustering_min_cluster_size": 12,
            "clustering_threshold": 0.7045654963945799,
        },
        "segments": len(segments),
        "environment": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "pyannote_audio": importlib.metadata.version("pyannote.audio"),
            "gguf": importlib.metadata.version("gguf"),
        },
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PYANNOTE_DIARIZATION_OFFICIAL_REFERENCE "
        f"chunks={EXPECTED_CHUNKS} embeddings={EXPECTED_CHUNKS * LOCAL_SPEAKERS} "
        f"segments={len(segments)} verdict=PASS"
    )


def self_test(wav: Path) -> None:
    _pcm16, pcm_f32 = read_jfk_clip(wav)
    if len(pcm_f32) != CLIP_SAMPLES * 4:
        raise RuntimeError("deterministic clip length drifted")
    revisions = [
        PYANNOTE_AUDIO_REVISION,
        PIPELINE_REVISION,
        SEGMENTATION_REVISION,
        EMBEDDING_REVISION,
    ]
    if any(len(revision) != 40 for revision in revisions):
        raise RuntimeError("revision contract must use full 40-hex commits")
    hashes = [PIPELINE_SHA256, SEGMENTATION_SHA256, EMBEDDING_SHA256, JFK_SHA256]
    if any(len(digest) != 64 for digest in hashes):
        raise RuntimeError("artifact identity contract is incomplete")
    if SEGMENTATION_TENSORS != 54 or EMBEDDING_TENSORS != 182:
        raise RuntimeError("released tensor contract drifted")
    print("pyannote_diarization_dump_reference: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyannote-source", type=Path)
    parser.add_argument("--pipeline-gguf", type=Path)
    parser.add_argument("--segmentation-gguf", type=Path)
    parser.add_argument("--embedding-gguf", type=Path)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and any(
        value is None
        for value in (
            args.pyannote_source,
            args.pipeline_gguf,
            args.segmentation_gguf,
            args.embedding_gguf,
            args.output_dir,
        )
    ):
        parser.error(
            "--pyannote-source, all three --*-gguf paths, --wav, and --output-dir are required"
        )
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test(args.wav)
        return 0
    dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
