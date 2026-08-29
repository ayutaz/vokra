#!/usr/bin/env python3
"""Dump an independent omniASR-CTC-1B oracle using the pinned upstream code.

This tool intentionally imports the official ``omnilingual_asr`` loader and
fairseq2 model.  It never reimplements wav2vec2, has no fixture fallback, and
fails if either checkout or the loaded model cannot be authenticated.  The
large checkpoint/reference run belongs on VAST; ``--self-test`` is the only
safe local mode.

The output packet contains raw little-endian ``f32`` frontend/encoder/logit
arrays and ``u32`` greedy ids plus one duplicate-key-free JSON manifest.  The
manifest binds both official source checkouts, their content SHA-256 values,
the exact HF checkpoint/tokenizer identities, and the deterministic non-zero
PCM input.  No transcript text is used as an oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

OMNILINGUAL_REPOSITORY = "https://github.com/facebookresearch/omnilingual-asr"
OMNILINGUAL_REVISION = "a7fb36017a46eee8953f76bd628c174d51aefeef"
FAIRSEQ2_REPOSITORY = "https://github.com/facebookresearch/fairseq2"
FAIRSEQ2_REVISION = "8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca"
MODEL_ID = "facebook/omniASR-CTC-1B"
HF_REVISION = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
CHECKPOINT_SHA256 = "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c"
PREPARED_SHA256 = "cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5"
SAMPLE_RATE = 16_000
PCM_SAMPLES = 16_000
PCM_EPS = 1e-5
SCHEMA = "omniasr-ctc-reference-v1"

# These are the source files that establish the model card and the actual
# fairseq2 implementation boundary.  Their raw content hashes are checked
# against the reviewed pins below; Git blob SHA-1 is not accepted.
OMNILINGUAL_SOURCE_PATHS = (
    "src/omnilingual_asr/cards/models/rc_models.yaml",
    "src/omnilingual_asr/datasets/utils/audio.py",
    "src/omnilingual_asr/models/wav2vec2_asr/config.py",
    "src/omnilingual_asr/models/wav2vec2_ssl/config.py",
)
FAIRSEQ2_SOURCE_PATHS = (
    "src/fairseq2/models/wav2vec2/config.py",
    "src/fairseq2/models/wav2vec2/factory.py",
    "src/fairseq2/models/wav2vec2/feature_extractor.py",
    "src/fairseq2/models/wav2vec2/position_encoder.py",
    "src/fairseq2/models/wav2vec2/frontend.py",
    "src/fairseq2/models/wav2vec2/asr/config.py",
    "src/fairseq2/models/wav2vec2/asr/factory.py",
    "src/fairseq2/models/wav2vec2/asr/model.py",
    # Wav2Vec2AsrModel delegates its encoder blocks to these upstream
    # implementations.  They are part of the authenticated boundary; this
    # list is intentionally exact and is not a substitute for their source.
    "src/fairseq2/models/transformer/encoder.py",
    "src/fairseq2/models/transformer/encoder_layer.py",
    "src/fairseq2/models/transformer/ffn.py",
    "src/fairseq2/models/transformer/multihead_attention.py",
    "src/fairseq2/models/transformer/norm_order.py",
    "src/fairseq2/models/transformer/sdpa/base.py",
    "src/fairseq2/models/transformer/sdpa/default.py",
    "src/fairseq2/models/transformer/sdpa/torch.py",
    "src/fairseq2/nn/normalization.py",
    "src/fairseq2/nn/projection.py",
)

# These pins are SHA-256 of the exact raw bytes at the pinned commits.  A
# dynamically recorded digest is not accepted: it cannot authenticate that the
# worker used the reviewed source boundary.  Do not replace these with Git
# blob SHA-1 or guessed digests.
EXPECTED_OMNILINGUAL_SOURCE_SHA256: dict[str, str] = {
    "src/omnilingual_asr/cards/models/rc_models.yaml": "7c9a28b2a111f2e088a5b2be161dd68686a810cd7462241209c2c5e8a81a2913",
    "src/omnilingual_asr/datasets/utils/audio.py": "e4a36129233325f95ab342939ad294fe37ac4eadaff6366524d60dc7ab8ea69e",
    "src/omnilingual_asr/models/wav2vec2_asr/config.py": "94ee297b4ebb122967631d2739b329e3b0d8432e9bf4a63306e085834e382ff1",
    "src/omnilingual_asr/models/wav2vec2_ssl/config.py": "550c6840b9b594226959948b4a48eb0e696171e9c5ac4fc070a9ea2c3d346414",
}
EXPECTED_FAIRSEQ2_SOURCE_SHA256: dict[str, str] = {
    "src/fairseq2/models/wav2vec2/config.py": "e75143abfa8e208f2291258949c1af7875087514113c0c370fa915b56905bd22",
    "src/fairseq2/models/wav2vec2/factory.py": "de7bbbd70cf06eb99fb363ecd641b13825c50c66fb1694d1f3a866e722523b5a",
    "src/fairseq2/models/wav2vec2/feature_extractor.py": "37ccd7f2209f0cab58cdd9766f71dc5425a1a42399fc9fa4ebef094694427ec9",
    "src/fairseq2/models/wav2vec2/position_encoder.py": "630941cb76bd77fe383e027be872004f2bbc7666c5f4e4619ef7cd16795280f6",
    "src/fairseq2/models/wav2vec2/frontend.py": "80b43735da89510df292fd7c97b0ff32fdbc52431802f6c01b6ebd8b45ed73cc",
    "src/fairseq2/models/wav2vec2/asr/config.py": "4e199ebe027239d23b6351251b997877ed1e67b0bf854930c3b7e9afbc6f1f3c",
    "src/fairseq2/models/wav2vec2/asr/factory.py": "60a59c2f63ac14707565016e034bb729c3ee91973d076b41189bd13173119c16",
    "src/fairseq2/models/wav2vec2/asr/model.py": "42a9bc0f9d11eb88a1848827468b692c972107b7fe3068fcbffa7844a25a1f38",
    "src/fairseq2/models/transformer/encoder.py": "b828efb95036e32865e32f79da9178c1a3dff204c5448194fed52a6c07ba7352",
    "src/fairseq2/models/transformer/encoder_layer.py": "389e6a49c54680a30ff09c3fc1d23c37fd1465f6772b42e25b9da59d8411acfd",
    "src/fairseq2/models/transformer/ffn.py": "c2e60872d4c1500bdc4767d032ab7dc7b0e9d4881ef3c4fe6cfa6b4ca7d321cd",
    "src/fairseq2/models/transformer/multihead_attention.py": "35b54f73e71b052160a0ca0baca998af5543a711ed297ec85d9a5ea7d32f552c",
    "src/fairseq2/models/transformer/norm_order.py": "1c1d4a191707291e8123423b2ac999a2c2f7e71690c238f7b0ce1cf0dc8080c0",
    "src/fairseq2/models/transformer/sdpa/base.py": "2e004badcbf3be84cbc0e74a395c873f0d4febd038b130769e9a29d1f7c1c549",
    "src/fairseq2/models/transformer/sdpa/default.py": "0bb33d8f2fbf7063bc3402ad9e5a5a4c94ea2b08a04282a64300ed8de2451e8b",
    "src/fairseq2/models/transformer/sdpa/torch.py": "b1aa6d3ac26d225a2d7e18bf023615f9aa8538de3e903d1db6f57fe30af1fb61",
    "src/fairseq2/nn/normalization.py": "f8f019e06d7d39040ef394cc292e58ed88a704c215fcac3afe7d9cfc028de158",
    "src/fairseq2/nn/projection.py": "14d625c9ad142e2e148e23ef5479ec29538a48f2a1e8534704d02162d096e052",
}


def die(message: str) -> "NoReturn":
    raise SystemExit(f"omniasr_ctc_dump_reference: {message}")


def sha256_file(path: Path) -> tuple[str, int]:
    if not path.is_file() or path.is_symlink():
        die(f"required regular file is missing or symlinked: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def source_records(
    root: Path, paths: tuple[str, ...], expected_hashes: dict[str, str]
) -> list[dict[str, Any]]:
    records = []
    for raw in paths:
        path = root / raw
        digest, size = sha256_file(path)
        expected = expected_hashes.get(raw)
        if expected is None:
            die(f"source raw SHA-256 pin is missing for {raw}; VAST review required")
        if digest != expected:
            die(f"source raw SHA-256 mismatch for {raw}: {digest} != {expected}")
        records.append({"path": raw, "sha256": digest, "bytes": size})
    return records


def fixed_pcm() -> list[float]:
    # A deterministic finite signal with non-zero DC-free content.  This is
    # the sole input for both sides of parity and is deliberately generated,
    # not downloaded or replaced with a fixture.
    values = [
        0.35 * math.sin(2.0 * math.pi * 440.0 * i / SAMPLE_RATE)
        + 0.11 * math.sin(2.0 * math.pi * 997.0 * i / SAMPLE_RATE)
        for i in range(PCM_SAMPLES)
    ]
    # Establish the input as float32 before either serialization or the
    # official normalizer sees it.  This round-trip is equivalent to creating
    # a torch.float32 tensor and makes the bytes consumed by Rust identical to
    # the values passed to apply_audio_normalization below.
    values = list(struct.unpack("<" + "f" * len(values), struct.pack("<" + "f" * len(values), *values)))
    if not any(value != 0.0 for value in values) or not all(
        math.isfinite(value) for value in values
    ):
        die("fixed PCM is zero or non-finite")
    return values


def write_f32(path: Path, values: Any) -> tuple[str, int, list[int]]:
    import torch

    tensor = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    shape = [int(dim) for dim in tensor.shape]
    flat = tensor.reshape(-1).tolist()
    if not flat or not all(math.isfinite(float(value)) for value in flat):
        die(f"reference tensor is empty or non-finite: {path.name}")
    path.write_bytes(struct.pack("<" + "f" * len(flat), *flat))
    digest, size = sha256_file(path)
    return digest, size, shape


def write_u32(path: Path, values: Any) -> tuple[str, int, list[int]]:
    import torch

    tensor = values.detach().to(device="cpu", dtype=torch.int64).contiguous()
    shape = [int(dim) for dim in tensor.shape]
    flat = tensor.reshape(-1).tolist()
    if not flat or any(int(value) < 0 or int(value) >= 9812 for value in flat):
        die(f"reference token ids are empty or outside vocab: {path.name}")
    path.write_bytes(struct.pack("<" + "I" * len(flat), *[int(v) for v in flat]))
    digest, size = sha256_file(path)
    return digest, size, shape


def collapse_ctc_ids(values: list[int]) -> list[int]:
    """Apply CTC's adjacent-repeat collapse followed by blank removal.

    Blank removal happens after collapsing, so ``[a, blank, a]`` remains
    ``[a, a]`` while adjacent ``[a, a]`` becomes ``[a]``.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for value in values:
        if value != previous:
            collapsed.append(value)
        previous = value
    return [value for value in collapsed if value != 0]


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            die(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        die(f"invalid manifest {path}: {exc}")
    if not isinstance(value, dict):
        die("manifest root must be an object")
    return value


def validate_manifest(path: Path) -> None:
    manifest = load_strict_json(path)
    expected = {
        "schema",
        "status",
        "model",
        "source",
        "input",
        "artifacts",
        "comparison",
    }
    if set(manifest) != expected:
        die(f"manifest top-level keys differ: {sorted(manifest)}")
    if manifest["schema"] != SCHEMA or manifest["status"] != "REFERENCE_COMPLETE":
        die("manifest status/schema is not exact")
    model = manifest["model"]
    if not isinstance(model, dict):
        die("model identity must be an object")
    if model != {
        "id": MODEL_ID,
        "hf_revision": HF_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "prepared_sha256": PREPARED_SHA256,
        "checkpoint_bytes": model.get("checkpoint_bytes"),
        "tokenizer_sha256": model.get("tokenizer_sha256"),
        "tokenizer_bytes": model.get("tokenizer_bytes"),
        "dtype": "float32",
        "tensor_count": 807,
    }:
        die("model identity is not exact")
    for key in ("checkpoint_bytes", "tokenizer_bytes"):
        if not isinstance(model[key], int) or isinstance(model[key], bool) or model[key] <= 0:
            die(f"model byte count is invalid: {key}")
    for key in ("checkpoint_sha256", "prepared_sha256", "tokenizer_sha256"):
        if not isinstance(model[key], str) or not re.fullmatch(r"[0-9a-f]{64}", model[key]):
            die(f"model digest is invalid: {key}")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {"omnilingual_asr", "fairseq2"}:
        die("source identity is incomplete")
    for name, repository, revision in (
        ("omnilingual_asr", OMNILINGUAL_REPOSITORY, OMNILINGUAL_REVISION),
        ("fairseq2", FAIRSEQ2_REPOSITORY, FAIRSEQ2_REVISION),
    ):
        record = source[name]
        if not isinstance(record, dict):
            die(f"{name} source identity must be an object")
        if record.get("repository") != repository or record.get("revision") != revision:
            die(f"{name} source repository/revision mismatch")
        rows = record.get("files")
        expected_paths = set(OMNILINGUAL_SOURCE_PATHS if name == "omnilingual_asr" else FAIRSEQ2_SOURCE_PATHS)
        if not isinstance(rows, list) or len(rows) != len(expected_paths) or not rows:
            die(f"{name} source file hashes are missing")
        expected_hashes = (
            EXPECTED_OMNILINGUAL_SOURCE_SHA256
            if name == "omnilingual_asr"
            else EXPECTED_FAIRSEQ2_SOURCE_SHA256
        )
        if {row.get("path") for row in rows if isinstance(row, dict)} != expected_paths:
            die(f"{name} source path set is not exact")
        for row in rows:
            if set(row) != {"path", "sha256", "bytes"} or not isinstance(row["path"], str):
                die(f"{name} source row schema is not exact")
            if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
                die(f"{name} source row size is invalid")
            if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
                die(f"{name} source row digest is invalid")
            expected_hash = expected_hashes.get(row["path"])
            if expected_hash is None:
                die(f"{name} source raw SHA-256 pin is missing for {row['path']}")
            if row["sha256"] != expected_hash:
                die(f"{name} source raw SHA-256 differs from the reviewed pin for {row['path']}")
    inp = manifest["input"]
    if not isinstance(inp, dict):
        die("input contract must be an object")
    if set(inp) != {"sample_rate", "channels", "samples", "pcm_sha256", "dtype", "normalization"}:
        die("input schema is not exact")
    if inp != dict(inp, sample_rate=SAMPLE_RATE, channels=1, samples=PCM_SAMPLES, dtype="float32-le", normalization="torch_layer_norm_waveform_eps_1e-5"):
        die("input contract is not exact")
    if not isinstance(inp["pcm_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", inp["pcm_sha256"]):
        die("input PCM digest is invalid")
    artifacts = manifest["artifacts"]
    artifact_paths = {
        "pcm.f32le", "frontend.f32le", "encoder.f32le", "ctc_logits.f32le", "tokens.u32le"
    }
    if not isinstance(artifacts, list) or len(artifacts) != len(artifact_paths) or {
        row.get("path") for row in artifacts if isinstance(row, dict)
    } != artifact_paths:
        die("artifact set is not exact")
    for row in artifacts:
        if not isinstance(row, dict):
            die("artifact row must be an object")
        if set(row) != {"path", "sha256", "bytes", "dtype", "shape"}:
            die(f"artifact schema is not exact: {row.get('path')}")
        if not isinstance(row["path"], str) or not isinstance(row["dtype"], str) or row["dtype"] not in {"float32-le", "uint32-le"} or not isinstance(row["shape"], list) or not row["shape"]:
            die(f"artifact dtype/shape is invalid: {row.get('path')}")
        expected_dtype = "uint32-le" if row["path"] == "tokens.u32le" else "float32-le"
        if row["dtype"] != expected_dtype:
            die(f"artifact dtype is invalid: {row['path']}")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in row["shape"]):
            die(f"artifact shape dimensions are invalid: {row.get('path')}")
        if not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
            die(f"artifact byte count is invalid: {row.get('path')}")
        if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            die(f"artifact digest is invalid: {row.get('path')}")
        expected_rank = 1 if row["path"] in {"pcm.f32le", "tokens.u32le"} else 3
        if len(row["shape"]) != expected_rank:
            die(f"artifact rank is invalid: {row.get('path')}")
        if row["path"] == "pcm.f32le" and row["shape"] != [PCM_SAMPLES]:
            die("PCM shape is not exact")
        if row["path"] in {"frontend.f32le", "encoder.f32le"} and row["shape"][0] != 1:
            die(f"batch shape is not exact: {row['path']}")
        if row["path"] in {"frontend.f32le", "encoder.f32le"} and row["shape"][2] != 1280:
            die(f"model dimension is not exact: {row['path']}")
        if row["path"] == "ctc_logits.f32le" and row["shape"][-1] != 9812:
            die("CTC vocabulary shape is not exact")
        artifact_path = path.parent / row["path"]
        digest, size = sha256_file(artifact_path)
        if digest != row["sha256"] or size != row["bytes"]:
            die(f"artifact digest/size mismatch: {row['path']}")
        elements = 1
        for dim in row["shape"]:
            elements *= dim
        if elements * 4 != row["bytes"]:
            die(f"artifact shape does not account for byte length: {row['path']}")
    pcm_row = next(row for row in artifacts if row["path"] == "pcm.f32le")
    if pcm_row["sha256"] != inp["pcm_sha256"] or pcm_row["shape"] != [PCM_SAMPLES]:
        die("PCM artifact is not bound to the input contract")
    comparison = manifest["comparison"]
    if comparison != {
        "frontend_atol": 0.01,
        "encoder_atol": 0.01,
        "logits_atol": 0.01,
        "tokens": "exact",
        "status": "NOT_RUN_RUST",
    }:
        die("comparison contract is not fail-closed")


def run(args: argparse.Namespace) -> None:
    import torch
    # Importing the official model package is required to install its
    # fairseq2 asset-family/architecture registration before load_model().
    # This is intentionally an import-only side effect from the pinned
    # checkout, never a local replacement registration.
    import omnilingual_asr.models.wav2vec2_asr  # noqa: F401
    from fairseq2.assets import AssetCard
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout
    from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
    from omnilingual_asr.datasets.utils.audio import apply_audio_normalization

    checkpoint, checkpoint_bytes = sha256_file(args.checkpoint)
    if checkpoint != CHECKPOINT_SHA256:
        die(f"checkpoint SHA-256 mismatch: {checkpoint}")
    tokenizer, tokenizer_bytes = sha256_file(args.tokenizer)
    if tokenizer_bytes <= 0:
        die("tokenizer is empty")
    if args.model_card != "omniASR_CTC_1B":
        die("only the official omniASR_CTC_1B card is accepted")
    omni_files = source_records(
        args.omnilingual_src,
        OMNILINGUAL_SOURCE_PATHS,
        EXPECTED_OMNILINGUAL_SOURCE_SHA256,
    )
    fairseq_files = source_records(
        args.fairseq2_src,
        FAIRSEQ2_SOURCE_PATHS,
        EXPECTED_FAIRSEQ2_SOURCE_SHA256,
    )
    pcm = fixed_pcm()
    pcm_path = args.output_dir / "pcm.f32le"
    pcm_path.write_bytes(struct.pack("<" + "f" * len(pcm), *pcm))
    pcm_digest, pcm_bytes = sha256_file(pcm_path)
    # This must remain the upstream preprocessing boundary.  The pinned source
    # file and its recorded hash authenticate the exact `eps=1e-5`
    # implementation; this worker does not mirror it locally.
    pcm_tensor = torch.tensor(pcm, dtype=torch.float32)
    normalized = apply_audio_normalization(pcm_tensor)
    source = normalized.reshape(1, -1)
    seqs_layout = BatchLayout(source.shape, seq_lens=[len(pcm)], device=source.device)

    # The official fairseq2 model loader is the only model construction route.
    # A local AssetCard points that official loader at the already-authenticated
    # checkpoint; this avoids a second, unbound download while retaining the
    # upstream model family/architecture factory.
    card = AssetCard(
        name=args.model_card,
        metadata={
            "model_family": "wav2vec2_asr",
            "model_arch": "1b",
            "checkpoint": args.checkpoint.resolve().as_uri(),
            "tokenizer_ref": "omniASR_tokenizer",
        },
    )
    model = load_model(card, device=torch.device("cpu"), dtype=torch.float32)
    if not isinstance(model, Wav2Vec2AsrModel):
        die(f"official card resolved to unexpected model type: {type(model)!r}")
    model.eval()
    with torch.inference_mode():
        extracted, extracted_layout = model.encoder_frontend.extract_features(source, seqs_layout)
        frontend, frontend_layout = model.encoder_frontend.process_features(extracted, extracted_layout)
        encoded, encoded_layout = model.encoder(frontend, frontend_layout)
        logits = model.final_proj(encoded)
    if not isinstance(logits, torch.Tensor) or not isinstance(encoded, torch.Tensor) or not isinstance(frontend, torch.Tensor):
        die("official frontend/encoder/final projection did not return tensors")
    predicted = torch.argmax(logits, dim=-1)
    # The official pipeline's CTC post-processing is intentionally limited to
    # greedy ids: collapse adjacent repeats, then remove blank id 0.  No text
    # tokenizer is involved in this oracle.
    ids = predicted[0, : int(encoded_layout.seq_lens_pt[0].item())]
    ids = collapse_ctc_ids([int(value) for value in ids.tolist()])
    artifacts = []
    for name, values, writer in (
        ("frontend.f32le", frontend, write_f32),
        ("encoder.f32le", encoded, write_f32),
        ("ctc_logits.f32le", logits, write_f32),
        ("tokens.u32le", torch.tensor(ids, dtype=torch.int64), write_u32),
    ):
        digest, size, shape = writer(args.output_dir / name, values)
        artifacts.append({"path": name, "sha256": digest, "bytes": size, "dtype": "uint32-le" if name.endswith("u32le") else "float32-le", "shape": shape})
    manifest = {
        "schema": SCHEMA,
        "status": "REFERENCE_COMPLETE",
        "model": {"id": MODEL_ID, "hf_revision": HF_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "prepared_sha256": PREPARED_SHA256, "checkpoint_bytes": checkpoint_bytes, "tokenizer_sha256": tokenizer, "tokenizer_bytes": tokenizer_bytes, "dtype": "float32", "tensor_count": 807},
        "source": {
            "omnilingual_asr": {"repository": OMNILINGUAL_REPOSITORY, "revision": OMNILINGUAL_REVISION, "files": omni_files},
            "fairseq2": {"repository": FAIRSEQ2_REPOSITORY, "revision": FAIRSEQ2_REVISION, "files": fairseq_files},
        },
        "input": {"sample_rate": SAMPLE_RATE, "channels": 1, "samples": PCM_SAMPLES, "pcm_sha256": pcm_digest, "dtype": "float32-le", "normalization": "torch_layer_norm_waveform_eps_1e-5"},
        "artifacts": [{"path": "pcm.f32le", "sha256": pcm_digest, "bytes": pcm_bytes, "dtype": "float32-le", "shape": [PCM_SAMPLES]}] + artifacts,
        "comparison": {
            "frontend_atol": 0.01,
            "encoder_atol": 0.01,
            "logits_atol": 0.01,
            "tokens": "exact",
            "status": "NOT_RUN_RUST",
        },
    }
    # The packet schema includes PCM as an artifact, so validate after writing.
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    validate_manifest(manifest_path)
    print(manifest_path)


def self_test() -> None:
    assert OMNILINGUAL_REVISION == "a7fb36017a46eee8953f76bd628c174d51aefeef"
    assert FAIRSEQ2_REVISION == "8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca"
    pcm = fixed_pcm()
    assert len(pcm) == PCM_SAMPLES and any(value != 0.0 for value in pcm)
    assert all(math.isfinite(value) for value in pcm)
    assert no_duplicate_pairs([("a", 1)]) == {"a": 1}
    assert collapse_ctc_ids([3, 3, 0, 3, 0, 0, 4]) == [3, 3, 4]
    try:
        no_duplicate_pairs([("a", 1), ("a", 2)])
    except SystemExit:
        pass
    else:
        die("duplicate-key self-test did not fail")
    print("omniasr_ctc_dump_reference.py self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--omnilingual-src", type=Path)
    parser.add_argument("--fairseq2-src", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-card", default="omniASR_CTC_1B")
    args = parser.parse_args()
    if args.self_test:
        if args.validate_manifest is not None or any(value is not None for value in (args.checkpoint, args.tokenizer, args.omnilingual_src, args.fairseq2_src, args.output_dir)):
            die("--self-test accepts no work arguments")
        self_test()
        return
    if args.validate_manifest is not None:
        if any(value is not None for value in (args.checkpoint, args.tokenizer, args.omnilingual_src, args.fairseq2_src, args.output_dir)):
            die("--validate-manifest accepts no work arguments")
        validate_manifest(args.validate_manifest)
        print(f"validated {args.validate_manifest}")
        return
    if any(value is None for value in (args.checkpoint, args.tokenizer, args.omnilingual_src, args.fairseq2_src, args.output_dir)):
        parser.error("work mode requires checkpoint, tokenizer, both source checkouts, and output-dir")
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run(args)


if __name__ == "__main__":
    main()
