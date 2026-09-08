#!/usr/bin/env python3
"""Run the official CLAP audio/text processor without loading a model.

This VAST-only stage is intentionally independent of ``ClapModel``.  It uses
only the pinned metadata snapshot, a deterministic PCM signal, and one fixed
text string, then writes comparable feature/ID/mask payloads atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "clap_dump_reference.py"
REPOSITORY = "laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
SAMPLE_RATE = 48_000
FIXED_TEXT = "a calm room with a distant piano"
SCHEMA = "vokra-clap-htsat-fused-source-only-reference-v1"
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fact(cls: Any) -> dict[str, str]:
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError(f"cannot locate official source for {cls.__name__}")
    path = Path(source).resolve()
    return {
        "class": cls.__name__,
        "source": str(path),
        "source_sha256": sha256_file(path),
    }


def write_atomic_no_replace(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"output already exists: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_metadata_snapshot(snapshot: Path) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(f"metadata snapshot is missing or symlinked: {snapshot}")
    for path in snapshot.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            raise RuntimeError(f"source-only snapshot contains weights: {path.name}")
    required = ("config.json", "preprocessor_config.json", *TOKENIZER_FILES)
    for name in required:
        path = snapshot / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"source-only snapshot is missing regular {name}")


def load_deterministic_pcm() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("vokra_clap_dump_reference", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared CLAP reference: {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deterministic_pcm()


def run(snapshot: Path, output_dir: Path) -> None:
    require_metadata_snapshot(snapshot)
    if output_dir.exists() or output_dir.is_symlink():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import transformers
    from transformers import ClapProcessor

    np.random.seed(0)
    processor = ClapProcessor.from_pretrained(snapshot, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or tokenizer.__class__.__name__ != "RobertaTokenizer":
        raise RuntimeError("official CLAP processor did not resolve RobertaTokenizer")
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None or feature_extractor.__class__.__name__ != "ClapFeatureExtractor":
        raise RuntimeError("official CLAP processor did not resolve ClapFeatureExtractor")
    pcm = load_deterministic_pcm()
    outputs = processor(
        audio=[pcm],
        text=[FIXED_TEXT],
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
    )
    required_keys = {"input_features", "is_longer", "input_ids", "attention_mask"}
    missing = sorted(required_keys - set(outputs))
    if missing:
        raise RuntimeError(f"official ClapProcessor omitted required source outputs: {missing}")

    arrays = {
        "pcm.f32": np.asarray(pcm, dtype="<f4"),
        "input_features.f32": np.asarray(outputs["input_features"], dtype="<f4"),
        "is_longer.u8": np.asarray(outputs["is_longer"], dtype="u1"),
        "input_ids.i64": np.asarray(outputs["input_ids"], dtype="<i8"),
        "attention_mask.u8": np.asarray(outputs["attention_mask"], dtype="u1"),
    }
    pcm_array = arrays["pcm.f32"]
    features_array = arrays["input_features.f32"]
    longer_array = arrays["is_longer.u8"]
    ids_array = arrays["input_ids.i64"]
    mask_array = arrays["attention_mask.u8"]
    if pcm_array.shape != (480_000,) or not np.isfinite(pcm_array).all():
        raise RuntimeError("source PCM is not the fixed finite [480000] little-endian f32 vector")
    if features_array.ndim < 2 or features_array.shape[0] != 1 or features_array.size == 0 or not np.isfinite(features_array).all():
        raise RuntimeError("official CLAP features are not finite, nonempty, batch=1")
    if longer_array.size == 0 or not np.isin(longer_array, [0, 1]).all():
        raise RuntimeError("official CLAP is_longer is empty or non-canonical")
    if ids_array.ndim < 2 or ids_array.shape[0] != 1 or ids_array.size == 0:
        raise RuntimeError("official CLAP input_ids are not nonempty batch=1")
    if mask_array.shape != ids_array.shape or mask_array.size == 0 or not np.isin(mask_array, [0, 1]).all():
        raise RuntimeError("official CLAP attention_mask is not binary and shape-matched")
    if not np.all((ids_array >= 0) & (ids_array < 50_265)):
        raise RuntimeError("official CLAP input_ids exceed the pinned vocabulary range")
    if not np.isfinite(arrays["pcm.f32"]).all() or not np.isfinite(arrays["input_features.f32"]).all():
        raise RuntimeError("official CLAP source output contains non-finite audio values")
    for filename, array in arrays.items():
        write_atomic_no_replace(output_dir / filename, array.tobytes(order="C"))

    processor_source = inspect.getsourcefile(ClapProcessor)
    if processor_source is None:
        raise RuntimeError("cannot locate official ClapProcessor source")
    tokenizer_source = inspect.getsourcefile(tokenizer.__class__)
    if tokenizer_source is None:
        raise RuntimeError("cannot locate official tokenizer source")
    metadata = {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_ONLY",
        "repository": REPOSITORY,
        "revision": REVISION,
        "fixed_text": FIXED_TEXT,
        "sample_rate": SAMPLE_RATE,
        "processor_class": ClapProcessor.__name__,
        "processor_source": str(Path(processor_source).resolve()),
        "processor_source_sha256": sha256_file(Path(processor_source).resolve()),
        "transformers_sources": {
            "feature_extractor": source_fact(feature_extractor.__class__),
            "processor": source_fact(processor.__class__),
            "tokenizer": source_fact(tokenizer.__class__),
        },
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_source": str(Path(tokenizer_source).resolve()),
        "tokenizer_files": {
            filename: sha256_file(snapshot / filename) for filename in TOKENIZER_FILES
        },
        "transformers_version": transformers.__version__,
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "execution": "SOURCE_ONLY_REFERENCE_EXECUTED",
        "outputs": {
            filename: {
                "dtype": "float32" if filename.endswith(".f32") else "int64" if filename.endswith(".i64") else "uint8",
                "endianness": "little",
                "itemsize": int(array.dtype.itemsize),
                "shape": list(array.shape),
                "byte_length": int(array.nbytes),
                "sha256": sha256_file(output_dir / filename),
            }
            for filename, array in arrays.items()
        },
        "files_sha256": {
            filename: sha256_file(output_dir / filename) for filename in sorted(arrays)
        },
    }
    write_atomic_no_replace(
        output_dir / "meta.json",
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def self_test() -> None:
    assert SCHEMA.endswith("-v1")
    assert REPOSITORY == "laion/clap-htsat-fused"
    assert len(REVISION) == 40 and all(char in "0123456789abcdef" for char in REVISION)
    assert SAMPLE_RATE == 48_000
    assert FIXED_TEXT
    source_text = Path(__file__).read_text(encoding="utf-8")
    assert "ClapProcessor" in source_text
    assert "audio=[pcm]" in source_text
    deprecated_keyword = "audio" + "s="
    assert deprecated_keyword not in source_text
    with tempfile.TemporaryDirectory(prefix="vokra-clap-source-only-") as temporary:
        root = Path(temporary)
        (root / "snapshot").mkdir()
        for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            (root / "snapshot" / name).write_text("{}\n", encoding="utf-8")
        (root / "snapshot" / "not-a-model.txt").write_text("ok\n", encoding="utf-8")
        require_metadata_snapshot(root / "snapshot")
        (root / "snapshot" / "forbidden.safetensors").write_bytes(b"not a checkpoint")
        try:
            require_metadata_snapshot(root / "snapshot")
        except RuntimeError as exc:
            assert "weights" in str(exc)
        else:
            raise AssertionError("source-only weight contamination was accepted")
        output = root / "output"
        output.mkdir()
        try:
            write_atomic_no_replace(output / "x", b"first")
            write_atomic_no_replace(output / "x", b"second")
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("source-only output replacement was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.snapshot is not None or args.output_dir is not None:
            parser.error("--self-test accepts no source paths")
        self_test()
        print("clap source-only reference self-test: OK")
        return 0
    if args.snapshot is None or args.output_dir is None:
        parser.error("normal runs require --snapshot and --output-dir")
    try:
        run(args.snapshot, args.output_dir)
    except Exception as exc:
        # Preserve an explicit blocked-stage packet for the VAST worker.  Do
        # not replace an existing meta.json or any caller-owned output.
        if args.output_dir.exists() and args.output_dir.is_dir() and not args.output_dir.is_symlink():
            blocked = args.output_dir / "meta.json"
            if not blocked.exists() and not blocked.is_symlink():
                write_atomic_no_replace(
                    blocked,
                    (
                        json.dumps(
                            {
                                "schema": SCHEMA,
                                "status": "BLOCKED_SOURCE_ONLY",
                                "repository": REPOSITORY,
                                "revision": REVISION,
                                "weights": "NOT_ACQUIRED",
                                "model_load": "NOT_PERFORMED",
                                "model_forward": "NOT_PERFORMED",
                                "execution": "SOURCE_ONLY_REFERENCE_BLOCKED",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
        print(f"CLAP_SOURCE_ONLY_REFERENCE BLOCKED_SOURCE_ONLY: {exc}", file=sys.stderr)
        return 2
    print(f"CLAP_SOURCE_ONLY_REFERENCE PASS_SOURCE_ONLY: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
