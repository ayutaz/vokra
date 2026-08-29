#!/usr/bin/env python3
"""Prepare the pinned NVIDIA Canary-1B-v2 main checkpoint on VAST.

The public `.nemo` carries both a timestamp auxiliary checkpoint and the
eight-layer Transformer-AED main checkpoint. This sidecar authenticates the
6.36 GB archive, selects ``./model_weights.ckpt`` explicitly, strips only the
32 scalar BatchNorm counters, and extracts the exact released config and
16,384-line aggregate vocabulary. It never uploads or publishes anything.
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

UPSTREAM_HF = "nvidia/canary-1b-v2"
UPSTREAM_REVISION = "87bc52657add533cd0156b3fc1aef027280754bf"
ARCHIVE_SIZE = 6_358_958_080
ARCHIVE_SHA256 = "ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094"
MAIN_CHECKPOINT_MEMBER = "./model_weights.ckpt"
MAIN_CHECKPOINT_SIZE = 3_853_798_427
CONFIG_SHA256 = "202542a45eb4ad656a47044c5db8c02926259d7232b436d77ca6af21dc84deae"
VOCAB_SHA256 = "4d10723a8bef5b8b186c3d2bb1449c849cc25c6b811969a7d170261b0ceed178"
FLOAT_TENSOR_COUNT = 1_478
COUNTER_COUNT = 32
COUNTER = re.compile(
    r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.num_batches_tracked$"
)
EXPECTED_SHARED_PAIRS = (
    (
        "transf_decoder._embedding.token_embedding.weight",
        "log_softmax.mlp.layer0.weight",
    ),
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
            "Canary-1B-v2 preparation is Linux/VAST-only; refusing to load "
            f"the 6.36 GB archive on {platform.system()}"
        )
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SystemExit(
            "VOKRA_PUBLISH_ON_VAST=1 is absent; run the repository VAST "
            "provisioner before large-model preparation"
        )


def validate_stripped_manifest(manifest: dict[str, object]) -> list[int]:
    if manifest.get("kept_count") != FLOAT_TENSOR_COUNT:
        raise ValueError(
            f"expected {FLOAT_TENSOR_COUNT} float tensors, got {manifest.get('kept_count')}"
        )
    if manifest.get("dropped_count") != COUNTER_COUNT:
        raise ValueError(
            f"expected {COUNTER_COUNT} stripped counters, got {manifest.get('dropped_count')}"
        )
    if manifest.get("unknown_stripped"):
        raise ValueError("unknown tensor dtypes were stripped")
    expected_shared_pairs = [
        {"canonical": canonical, "cloned": cloned}
        for canonical, cloned in EXPECTED_SHARED_PAIRS
    ]
    if manifest.get("shared_pairs") != expected_shared_pairs:
        raise ValueError(
            "Canary-1B-v2 shared_pairs must contain exactly the pinned pair "
            f"{expected_shared_pairs}, got {manifest.get('shared_pairs')}"
        )
    if manifest.get("nemo_checkpoint_member") != MAIN_CHECKPOINT_MEMBER:
        raise ValueError(
            "preparation selected the wrong NeMo checkpoint member: "
            f"{manifest.get('nemo_checkpoint_member')!r}"
        )

    dropped = manifest.get("dropped_tensors")
    if not isinstance(dropped, list):
        raise ValueError("stripped manifest has no dropped_tensors list")
    layers: list[int] = []
    for tensor in dropped:
        if not isinstance(tensor, dict):
            raise ValueError(f"malformed stripped tensor row: {tensor!r}")
        match = COUNTER.fullmatch(str(tensor.get("name", "")))
        if (
            match is None
            or tensor.get("dtype") != "torch.int64"
            or tensor.get("shape") != []
        ):
            raise ValueError(f"unexpected stripped tensor: {tensor}")
        layers.append(int(match.group(1)))
    if sorted(layers) != list(range(COUNTER_COUNT)):
        raise ValueError(
            "expected one BatchNorm counter for each encoder layer 0..31; "
            f"got {sorted(layers)}"
        )
    return layers


def extract_small_assets(archive: Path, output_dir: Path) -> tuple[Path, Path, str]:
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        main = [member for member in members if member.name == MAIN_CHECKPOINT_MEMBER]
        if len(main) != 1 or main[0].size != MAIN_CHECKPOINT_SIZE:
            raise ValueError(
                "official main checkpoint member mismatch: "
                f"{[(member.name, member.size) for member in main]}"
            )

        configs = [
            member
            for member in members
            if member.isfile() and Path(member.name).name == "model_config.yaml"
        ]
        if len(configs) != 1:
            raise ValueError(
                f"expected one model_config.yaml, found {[member.name for member in configs]}"
            )
        config_stream = bundle.extractfile(configs[0])
        if config_stream is None:
            raise ValueError("could not read model_config.yaml")
        config = config_stream.read()
        if digest_bytes(config) != CONFIG_SHA256:
            raise ValueError("model_config.yaml does not match the pinned SHA-256")

        vocab_matches: list[tuple[str, bytes]] = []
        for member in members:
            if not member.isfile() or not member.name.endswith(".vocab"):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read tokenizer member {member.name}")
            payload = stream.read()
            if digest_bytes(payload) == VOCAB_SHA256:
                vocab_matches.append((member.name, payload))
        if len(vocab_matches) != 1:
            raise ValueError(
                "expected one tokenizer vocabulary matching the pinned SHA-256; "
                f"found {[name for name, _ in vocab_matches]}"
            )
        vocab_member, vocab = vocab_matches[0]
        if len(vocab.splitlines()) != 16_384:
            raise ValueError("released aggregate tokenizer must contain 16,384 lines")

    config_path = output_dir / "model_config.yaml"
    vocab_path = output_dir / "tokenizer.vocab"
    config_path.write_bytes(config)
    vocab_path.write_bytes(vocab)
    return config_path, vocab_path, vocab_member


def self_test() -> None:
    manifest: dict[str, object] = {
        "kept_count": FLOAT_TENSOR_COUNT,
        "dropped_count": COUNTER_COUNT,
        "unknown_stripped": [],
        "shared_pairs": [
            {"canonical": canonical, "cloned": cloned}
            for canonical, cloned in EXPECTED_SHARED_PAIRS
        ],
        "nemo_checkpoint_member": MAIN_CHECKPOINT_MEMBER,
        "dropped_tensors": [
            {
                "name": f"encoder.layers.{layer}.conv.batch_norm.num_batches_tracked",
                "dtype": "torch.int64",
                "shape": [],
            }
            for layer in range(COUNTER_COUNT)
        ],
    }
    assert validate_stripped_manifest(manifest) == list(range(COUNTER_COUNT))

    for invalid_pairs, label in (
        ([], "missing shared pair"),
        (
            [
                {
                    "canonical": EXPECTED_SHARED_PAIRS[0][1],
                    "cloned": EXPECTED_SHARED_PAIRS[0][0],
                }
            ],
            "reversed shared pair",
        ),
        (
            [
                {
                    "canonical": "transf_decoder.embedding.token_embedding.weight",
                    "cloned": EXPECTED_SHARED_PAIRS[0][1],
                }
            ],
            "aliased shared pair",
        ),
        (
            [
                {"canonical": canonical, "cloned": cloned}
                for canonical, cloned in EXPECTED_SHARED_PAIRS
            ]
            + [{"canonical": "unexpected", "cloned": "unexpected"}],
            "additional shared pair",
        ),
    ):
        invalid_manifest = dict(manifest)
        invalid_manifest["shared_pairs"] = invalid_pairs
        try:
            validate_stripped_manifest(invalid_manifest)
        except ValueError as error:
            assert "shared_pairs" in str(error), label
        else:
            raise AssertionError(f"{label} must fail")

    wrong = dict(manifest)
    wrong["nemo_checkpoint_member"] = "./timestamps_asr_model_weights.ckpt"
    try:
        validate_stripped_manifest(wrong)
    except ValueError as error:
        assert "wrong NeMo checkpoint" in str(error)
    else:
        raise AssertionError("timestamp auxiliary checkpoint must fail")
    print("canary_1b_v2_prepare_checkpoint: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Pinned canary-1b-v2.nemo")
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
    prepared = output_dir / "canary-1b-v2.prepared.safetensors"
    helper = Path(__file__).resolve().with_name("nemo_pt_to_safetensors.py")
    result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--input",
            str(archive),
            "--output",
            str(prepared),
            "--nemo-checkpoint-member",
            MAIN_CHECKPOINT_MEMBER,
        ],
        check=False,
    )
    if result.returncode:
        return result.returncode

    stripped_manifest_path = prepared.with_suffix(
        prepared.suffix + ".stripped-manifest.json"
    )
    stripped_manifest = json.loads(stripped_manifest_path.read_text(encoding="utf-8"))
    dropped_layers = validate_stripped_manifest(stripped_manifest)
    config_path, vocab_path, vocab_member = extract_small_assets(archive, output_dir)

    audit = {
        "upstream_hf": UPSTREAM_HF,
        "upstream_revision": UPSTREAM_REVISION,
        "archive_size": ARCHIVE_SIZE,
        "archive_sha256": archive_sha256,
        "main_checkpoint_member": MAIN_CHECKPOINT_MEMBER,
        "main_checkpoint_size": MAIN_CHECKPOINT_SIZE,
        "prepared_safetensors": prepared.name,
        "prepared_sha256": digest_file(prepared),
        "float_tensor_count": FLOAT_TENSOR_COUNT,
        "dropped_counter_layers": dropped_layers,
        "model_config": config_path.name,
        "model_config_sha256": digest_file(config_path),
        "tokenizer_vocab": vocab_path.name,
        "tokenizer_vocab_member": vocab_member,
        "tokenizer_vocab_sha256": digest_file(vocab_path),
    }
    (output_dir / "prepare-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
