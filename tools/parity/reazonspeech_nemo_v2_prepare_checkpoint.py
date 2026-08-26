#!/usr/bin/env python3
"""Prepare the pinned ReazonSpeech NeMo v2 archive on VAST.

This offline sidecar authenticates the 2.48 GB NeMo archive, delegates pickle
loading to the repository's audited ``nemo_pt_to_safetensors.py`` bridge, and
extracts only the fixed model config and plaintext SentencePiece vocabulary.
It never uploads or publishes anything, and actual preparation refuses to run
outside a provisioned Linux/VAST environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
from pathlib import Path

UPSTREAM_HF = "reazon-research/reazonspeech-nemo-v2"
UPSTREAM_REVISION = "33693408be76b7cba9fd4a7546a0a8772430211b"
ARCHIVE_SIZE = 2_477_946_880
ARCHIVE_SHA256 = "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae"
MODEL_WEIGHTS_SIZE = 2_477_537_598
CONFIG_SIZE = 50_649
CONFIG_SHA256 = "88925d58533c40da62007ad39b8abd702646c7e81627dea5b15961c4ad4f9833"
VOCAB_SIZE_BYTES = 41_144
VOCAB_SHA256 = "989e4950cf53c0fee66f632cdd966bdd840b851a9e0e812322fd667e4b1c07bb"
FLOAT_TENSOR_COUNT = 965
COUNTER_COUNT = 24
COUNTER = re.compile(
    r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.num_batches_tracked$"
)


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "ReazonSpeech-NeMo-v2 preparation is Linux/VAST-only; refusing "
            f"the 2.48 GB checkpoint on {platform.system()}"
        )
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit(
            "VOKRA_PUBLISH_ON_VAST=1 is absent; run the repository VAST "
            "provisioner before large-model preparation"
        )


def validate_stripped_manifest(manifest: dict[str, object]) -> list[int]:
    if manifest.get("kept_count") != FLOAT_TENSOR_COUNT:
        raise ValueError(
            f"expected {FLOAT_TENSOR_COUNT} float tensors, got "
            f"{manifest.get('kept_count')}"
        )
    if manifest.get("dropped_count") != COUNTER_COUNT:
        raise ValueError(
            f"expected {COUNTER_COUNT} stripped counters, got "
            f"{manifest.get('dropped_count')}"
        )
    if manifest.get("unknown_stripped"):
        raise ValueError("unknown tensor dtypes were stripped")
    if manifest.get("shared_pairs"):
        raise ValueError("the pinned checkpoint unexpectedly contains shared tensors")
    if manifest.get("nemo_checkpoint_member") not in {
        "model_weights.ckpt",
        "./model_weights.ckpt",
    }:
        raise ValueError(
            "the preparation bridge did not select the root model_weights.ckpt"
        )

    layers: list[int] = []
    dropped = manifest.get("dropped_tensors")
    if not isinstance(dropped, list):
        raise ValueError("stripped manifest has no dropped_tensors list")
    for tensor in dropped:
        if not isinstance(tensor, dict):
            raise ValueError(f"malformed stripped tensor row: {tensor!r}")
        name = tensor.get("name")
        match = COUNTER.fullmatch(name) if isinstance(name, str) else None
        if (
            match is None
            or tensor.get("dtype") != "torch.int64"
            or tensor.get("shape") != []
        ):
            raise ValueError(f"unexpected stripped tensor: {tensor}")
        layers.append(int(match.group(1)))
    if sorted(layers) != list(range(COUNTER_COUNT)):
        raise ValueError(
            "expected one BatchNorm counter for every encoder layer 0..23; "
            f"got {sorted(layers)}"
        )
    return layers


def one_member(bundle: tarfile.TarFile, basename: str) -> tarfile.TarInfo:
    matches = [
        member
        for member in bundle.getmembers()
        if member.isfile() and Path(member.name).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {basename}, found {[item.name for item in matches]}"
        )
    return matches[0]


def read_member(bundle: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = bundle.extractfile(member)
    if stream is None:
        raise ValueError(f"could not read tar member {member.name}")
    return stream.read()


def extract_small_assets(archive: Path, output_dir: Path) -> tuple[Path, Path]:
    with tarfile.open(archive, "r:*") as bundle:
        config_member = one_member(bundle, "model_config.yaml")
        vocab_members = [
            member
            for member in bundle.getmembers()
            if member.isfile() and member.name.endswith("_tokenizer.vocab")
        ]
        if len(vocab_members) != 1:
            raise ValueError(
                "expected exactly one *_tokenizer.vocab member, found "
                f"{[item.name for item in vocab_members]}"
            )
        weights_member = one_member(bundle, "model_weights.ckpt")
        if weights_member.size != MODEL_WEIGHTS_SIZE:
            raise ValueError(
                f"model_weights.ckpt size {weights_member.size}, expected "
                f"{MODEL_WEIGHTS_SIZE}"
            )
        config = read_member(bundle, config_member)
        vocab = read_member(bundle, vocab_members[0])

    if len(config) != CONFIG_SIZE or digest_bytes(config) != CONFIG_SHA256:
        raise ValueError(
            "model_config.yaml size/hash does not match the pinned release"
        )
    if len(vocab) != VOCAB_SIZE_BYTES or digest_bytes(vocab) != VOCAB_SHA256:
        raise ValueError("tokenizer.vocab size/hash does not match the pinned release")
    lines = vocab.decode("utf-8").splitlines()
    if len(lines) != 3_000 or not lines[0].startswith("<unk>\t"):
        raise ValueError(
            "tokenizer.vocab must contain 3,000 scored pieces beginning with <unk>"
        )

    config_path = output_dir / "model_config.yaml"
    vocab_path = output_dir / "tokenizer.vocab"
    config_path.write_bytes(config)
    vocab_path.write_bytes(vocab)
    return config_path, vocab_path


def self_test() -> None:
    manifest: dict[str, object] = {
        "kept_count": FLOAT_TENSOR_COUNT,
        "dropped_count": COUNTER_COUNT,
        "unknown_stripped": [],
        "shared_pairs": [],
        "nemo_checkpoint_member": "./model_weights.ckpt",
        "dropped_tensors": [
            {
                "name": (
                    f"encoder.layers.{layer}.conv.batch_norm.num_batches_tracked"
                ),
                "dtype": "torch.int64",
                "shape": [],
            }
            for layer in range(COUNTER_COUNT)
        ],
    }
    assert validate_stripped_manifest(manifest) == list(range(COUNTER_COUNT))
    assert len(ARCHIVE_SHA256) == len(CONFIG_SHA256) == len(VOCAB_SHA256) == 64
    print("reazonspeech_nemo_v2_prepare_checkpoint: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Pinned reazonspeech-nemo-v2.nemo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless --self-test is used")

    require_vast()
    archive = args.input.resolve()
    if not archive.is_file():
        parser.error(f"input is not a regular file: {archive}")
    if archive.stat().st_size != ARCHIVE_SIZE:
        raise SystemExit(
            f"archive size {archive.stat().st_size} != pinned {ARCHIVE_SIZE} bytes"
        )
    archive_sha256 = digest_file(archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise SystemExit(
            f"archive SHA-256 {archive_sha256} != pinned {ARCHIVE_SHA256}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / "reazonspeech-nemo-v2.prepared.safetensors"
    helper = Path(__file__).resolve().with_name("nemo_pt_to_safetensors.py")
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--input",
            str(archive),
            "--output",
            str(prepared),
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode

    stripped_manifest_path = prepared.with_suffix(
        prepared.suffix + ".stripped-manifest.json"
    )
    stripped_manifest = json.loads(
        stripped_manifest_path.read_text(encoding="utf-8")
    )
    dropped_layers = validate_stripped_manifest(stripped_manifest)
    config_path, vocab_path = extract_small_assets(archive, output_dir)

    audit = {
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "archive_size": ARCHIVE_SIZE,
        "archive_sha256": archive_sha256,
        "prepared_safetensors": prepared.name,
        "prepared_sha256": digest_file(prepared),
        "float_tensor_count": FLOAT_TENSOR_COUNT,
        "dropped_counter_layers": dropped_layers,
        "model_config": config_path.name,
        "model_config_sha256": digest_file(config_path),
        "tokenizer_vocab": vocab_path.name,
        "tokenizer_vocab_sha256": digest_file(vocab_path),
    }
    (output_dir / "prepare-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
