#!/usr/bin/env -S uv run --project tools/parity --frozen --python 3.12 python
"""Prepare the pinned NVIDIA Canary-1B-Flash `.nemo` release on VAST.

This is a large-model sidecar, never a runtime dependency. It authenticates
the complete 3.54 GB archive, delegates PyTorch-pickle extraction to the
existing audited NeMo bridge, verifies the exact 1,374-float / 32-counter
partition, and reconstructs the five-tokenizer aggregate vocabulary only when
its byte SHA-256 matches the released contract.

Actual preparation is Linux/VAST-only and requires the marker installed by
``scripts/publish/vast-ai/provision.sh``::

    VOKRA_PUBLISH_ON_VAST=1 uv run --project tools/parity --python 3.12 python \
      tools/parity/canary_1b_flash_prepare_checkpoint.py \
      --input /workspace/canary-1b-flash.nemo --output-dir /workspace/canary

No upload or publication is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

UPSTREAM_HF = "nvidia/canary-1b-flash"
UPSTREAM_REVISION = "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e"
ARCHIVE_SIZE = 3_540_715_520
ARCHIVE_SHA256 = "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324"
CONFIG_SHA256 = "42d71aebc1f4b9f387a20902db71e00128b324ff5156bdac63897e1afad55ff9"
AGGREGATE_VOCAB_SHA256 = (
    "08cb29d15437dbd3f45c26046c2f5994b3b92c86a3aa4a6e27d253d40837db79"
)
FLOAT_TENSOR_COUNT = 1_374
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


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def write_text_exclusive(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def require_vast() -> None:
    if platform.system() != "Linux":
        raise SystemExit(
            "Canary-1B-Flash preparation is Linux/VAST-only; refusing to load "
            f"the 3.54 GB checkpoint on {platform.system()}"
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
            "Canary-1B-Flash shared_pairs must contain exactly the pinned pair "
            f"{expected_shared_pairs}, got {manifest.get('shared_pairs')}"
        )

    layers: list[int] = []
    dropped = manifest.get("dropped_tensors")
    if not isinstance(dropped, list):
        raise ValueError("stripped manifest has no dropped_tensors list")
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


def resolve_aggregate_vocab(
    members: dict[str, bytes], expected_sha256: str = AGGREGATE_VOCAB_SHA256
) -> tuple[list[str], bytes]:
    if len(members) != 5:
        raise ValueError(
            f"expected exactly five tokenizer .vocab members, found {sorted(members)}"
        )
    matches: list[tuple[list[str], bytes]] = []
    for order in itertools.permutations(sorted(members)):
        payload = b"".join(members[name] for name in order)
        if digest_bytes(payload) == expected_sha256:
            matches.append((list(order), payload))
    if len(matches) != 1:
        raise ValueError(
            "could not resolve one unique tokenizer order from the pinned aggregate "
            f"SHA-256; matches={len(matches)}"
        )
    order, payload = matches[0]
    counts = [len(members[name].splitlines()) for name in order]
    if counts != [1_152, 1_024, 1_024, 1_024, 1_024]:
        raise ValueError(
            "aggregate tokenizer component sizes must be 1152 + 4x1024 in "
            f"released order; got {counts} for {order}"
        )
    return order, payload


def extract_small_assets(archive: Path, output_dir: Path) -> tuple[list[str], Path, Path]:
    with tarfile.open(archive, "r:") as bundle:
        config_members = [
            member
            for member in bundle.getmembers()
            if member.isfile() and Path(member.name).name == "model_config.yaml"
        ]
        if len(config_members) != 1:
            raise ValueError(
                f"expected one model_config.yaml, found {[m.name for m in config_members]}"
            )
        config_stream = bundle.extractfile(config_members[0])
        if config_stream is None:
            raise ValueError("could not read model_config.yaml")
        config = config_stream.read()
        if digest_bytes(config) != CONFIG_SHA256:
            raise ValueError("model_config.yaml does not match the pinned released SHA-256")

        vocab: dict[str, bytes] = {}
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(".vocab"):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read tokenizer member {member.name}")
            vocab[member.name] = stream.read()

    order, aggregate = resolve_aggregate_vocab(vocab)
    config_path = output_dir / "model_config.yaml"
    aggregate_path = output_dir / "canary-1b-flash.aggregate.vocab"
    write_bytes_exclusive(config_path, config)
    write_bytes_exclusive(aggregate_path, aggregate)
    return order, config_path, aggregate_path


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="canary-flash-prepare-") as directory:
        path = Path(directory) / "sentinel"
        write_bytes_exclusive(path, b"one")
        try:
            write_bytes_exclusive(path, b"two")
        except FileExistsError:
            pass
        else:
            raise AssertionError("pre-existing output was clobbered")
    manifest: dict[str, object] = {
        "kept_count": FLOAT_TENSOR_COUNT,
        "dropped_count": COUNTER_COUNT,
        "unknown_stripped": [],
        "shared_pairs": [
            {"canonical": canonical, "cloned": cloned}
            for canonical, cloned in EXPECTED_SHARED_PAIRS
        ],
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

    members = {
        "spl.vocab": b"s\n",
        "en.vocab": b"e\n",
        "de.vocab": b"d\n",
        "es.vocab": b"x\n",
        "fr.vocab": b"f\n",
    }
    expected = digest_bytes(b"".join(members[name] for name in members))
    try:
        resolve_aggregate_vocab(members, expected)
    except ValueError as error:
        # The production component-size gate is expected to reject this tiny
        # payload after the order/hash resolver has succeeded.
        assert "component sizes" in str(error)
    else:
        raise AssertionError("tiny vocab must reach and fail the production size gate")
    print("canary_1b_flash_prepare_checkpoint: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Pinned canary-1b-flash.nemo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required unless --self-test is used")

    require_vast()
    if args.input.is_symlink():
        parser.error(f"input must not be a symlink: {args.input}")
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

    if args.output_dir.exists() and (args.output_dir.is_symlink() or not args.output_dir.is_dir()):
        parser.error(f"output directory must be a non-symlink directory: {args.output_dir}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / "canary-1b-flash.prepared.safetensors"
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
    stripped_manifest = json.loads(stripped_manifest_path.read_text(encoding="utf-8"))
    dropped_layers = validate_stripped_manifest(stripped_manifest)
    tokenizer_order, config_path, aggregate_path = extract_small_assets(
        archive, output_dir
    )

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
        "aggregate_vocab": aggregate_path.name,
        "aggregate_vocab_sha256": digest_file(aggregate_path),
        "aggregate_vocab_member_order": tokenizer_order,
    }
    write_text_exclusive(
        output_dir / "prepare-audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
