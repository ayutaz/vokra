#!/usr/bin/env python3
"""Dump an independent DNSMOS P.808 + P.835 reference JSONL.

The oracle imports ``ComputeScore`` from the exact pinned Microsoft
DNS-Challenge checkout and calls that official implementation with the two
audited ONNX files. It never imports Vokra and contains no mirror network.

Model execution belongs on VAST. ``--self-test`` is stdlib-only and validates
the pinned contract without importing ONNX Runtime, librosa, or soundfile.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


SOURCE_REPOSITORY = "https://github.com/microsoft/DNS-Challenge"
SOURCE_REVISION = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
SOURCE_RELATIVE = Path("DNSMOS/dnsmos_local.py")
SOURCE_BYTES = 6_491
SOURCE_SHA256 = "1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b"
LICENSE_RELATIVE = Path("LICENSE")
LICENSE_BYTES = 19_047
LICENSE_SHA256 = "d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f"
P808_BYTES = 224_860
P808_SHA256 = "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91"
P835_BYTES = 1_157_965
P835_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
SAMPLE_RATE = 16_000
ONNXRUNTIME_VERSION = "1.29.0"
NUMPY_VERSION = "2.3.5"
LIBROSA_VERSION = "0.11.0"
SOUNDFILE_VERSION = "0.14.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing pinned input: {path}")
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: {size} bytes != pinned {expected_bytes}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {digest} != pinned {expected_sha256}")


def git_output(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(checkout: Path) -> Path:
    checkout = checkout.resolve()
    revision = git_output(checkout, "rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"DNS-Challenge source revision {revision!r} != pinned {SOURCE_REVISION!r}"
        )
    source = checkout / SOURCE_RELATIVE
    validate_file(source, SOURCE_BYTES, SOURCE_SHA256)
    validate_file(checkout / LICENSE_RELATIVE, LICENSE_BYTES, LICENSE_SHA256)
    return source


def import_official(source: Path) -> ModuleType:
    module_name = "vokra_independent_dnsmos_reference"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001 - oracle failure must be loud
        raise RuntimeError(
            "could not import the pinned official dnsmos_local.py; a mirror fallback is forbidden"
        ) from error
    imported = Path(module.__file__).resolve()
    if imported != source.resolve():
        raise ValueError(f"imported DNSMOS from {imported}, expected {source}")
    if not hasattr(module, "ComputeScore"):
        raise ValueError("official dnsmos_local.py has no ComputeScore")
    if getattr(module, "SAMPLING_RATE", None) != SAMPLE_RATE:
        raise ValueError("official DNSMOS sample rate drifted from 16000")
    if getattr(module, "INPUT_LENGTH", None) != 9.01:
        raise ValueError("official DNSMOS input length drifted from 9.01 seconds")
    return module


def finite_number(row: dict[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"official DNSMOS produced non-finite {key}: {value!r}")
    return value


def dump(args: argparse.Namespace) -> None:
    import soundfile as sf

    if sf.__version__ != SOUNDFILE_VERSION:
        raise ValueError(
            f"soundfile {sf.__version__!r} != pinned {SOUNDFILE_VERSION!r}; run through tools/parity/uv.lock"
        )

    source = validate_source(args.source_dir)
    validate_file(args.p808, P808_BYTES, P808_SHA256)
    validate_file(args.p835, P835_BYTES, P835_SHA256)
    if not args.input_wav:
        raise ValueError("at least one --input-wav is required")
    basenames = [path.name for path in args.input_wav]
    if len(set(basenames)) != len(basenames):
        raise ValueError("--input-wav basenames must be unique")
    if args.output_jsonl.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output_jsonl}")

    for path in args.input_wav:
        info = sf.info(path)
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: {info.samplerate} Hz != required {SAMPLE_RATE} Hz"
            )
        if info.channels != 1:
            raise ValueError(f"{path}: {info.channels} channels != required mono")
        if info.frames <= 0:
            raise ValueError(f"{path}: empty audio is not a valid DNSMOS fixture")

    official = import_official(source)
    for package, actual, expected in [
        ("onnxruntime", official.ort.__version__, ONNXRUNTIME_VERSION),
        ("numpy", official.np.__version__, NUMPY_VERSION),
        ("librosa", official.librosa.__version__, LIBROSA_VERSION),
    ]:
        if actual != expected:
            raise ValueError(
                f"{package} {actual!r} != pinned {expected!r}; run through tools/parity/uv.lock"
            )
    scorer = official.ComputeScore(str(args.p835.resolve()), str(args.p808.resolve()))
    records = []
    for path in args.input_wav:
        result = scorer(str(path.resolve()), SAMPLE_RATE, False)
        records.append(
            {
                "wav": path.name,
                "samples": int(round(finite_number(result, "len_in_sec") * SAMPLE_RATE)),
                "num_hops": int(result["num_hops"]),
                "p808": finite_number(result, "P808_MOS"),
                "sig": finite_number(result, "SIG"),
                "bak": finite_number(result, "BAK"),
                "ovrl": finite_number(result, "OVRL"),
                "source_revision": SOURCE_REVISION,
                "source_sha256": SOURCE_SHA256,
                "p808_onnx_sha256": P808_SHA256,
                "p835_onnx_sha256": P835_SHA256,
                "onnxruntime_version": ONNXRUNTIME_VERSION,
                "numpy_version": NUMPY_VERSION,
                "librosa_version": LIBROSA_VERSION,
                "soundfile_version": SOUNDFILE_VERSION,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def self_test() -> None:
    assert SOURCE_REVISION == "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
    assert len(SOURCE_SHA256) == len(P808_SHA256) == len(P835_SHA256) == 64
    assert SOURCE_RELATIVE.as_posix() == "DNSMOS/dnsmos_local.py"
    assert P808_BYTES + P835_BYTES == 1_382_825
    assert SAMPLE_RATE * 9.01 == 144_160.0
    assert (ONNXRUNTIME_VERSION, NUMPY_VERSION, LIBROSA_VERSION, SOUNDFILE_VERSION) == (
        "1.29.0",
        "2.3.5",
        "0.11.0",
        "0.14.0",
    )
    print("dnsmos_score_reference self-test: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="pinned microsoft/DNS-Challenge Git checkout root",
    )
    parser.add_argument("--p808", type=Path, help="pinned model_v8.onnx")
    parser.add_argument("--p835", type=Path, help="pinned sig_bak_ovr.onnx")
    parser.add_argument("--input-wav", type=Path, action="append")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    missing = [
        flag
        for flag, value in [
            ("--source-dir", args.source_dir),
            ("--p808", args.p808),
            ("--p835", args.p835),
            ("--input-wav", args.input_wav),
            ("--output-jsonl", args.output_jsonl),
        ]
        if value is None
    ]
    if missing:
        parser.error(f"required unless --self-test: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        dump(args)


if __name__ == "__main__":
    main()
