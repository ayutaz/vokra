#!/usr/bin/env python3
"""Dump an independent CLAP audio reference from official Transformers.

The model and processor are resolved from one immutable Hugging Face
revision. Tensor names and shapes are read from the loaded state dict (never
reconstructed here), which makes the resulting manifest the evidence needed
before a native GGUF binder can be enabled. This sidecar is VAST-only and is
not part of the Vokra runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


REPOSITORY = "laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
SAMPLE_RATE = 48_000
DUMPER_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> tuple[str, str]:
    from transformers import ClapModel

    source = inspect.getsourcefile(ClapModel)
    if source is None:
        raise RuntimeError("cannot locate official Transformers ClapModel source")
    path = Path(source).resolve()
    return str(path), sha256_file(path)


def deterministic_pcm() -> Any:
    import numpy as np

    samples = SAMPLE_RATE
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    signal = (
        0.31 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.17 * np.sin(2.0 * np.pi * 440.0 * time + 0.2)
        + 0.07 * np.sin(2.0 * np.pi * 880.0 * time + 0.7)
    )
    return np.ascontiguousarray(signal.astype(np.float32))


def self_test() -> None:
    """Check the deterministic, provenance-critical pieces without a model."""

    assert REPOSITORY == "laion/clap-htsat-fused"
    assert len(REVISION) == 40 and all(c in "0123456789abcdef" for c in REVISION)
    # Keep this contract check dependency-free: the real NumPy/Torch path is
    # intentionally imported only by ``dump`` on the VAST reference host.
    pcm = [
        0.31 * math.sin(2.0 * math.pi * 220.0 * index / SAMPLE_RATE)
        + 0.17 * math.sin(2.0 * math.pi * 440.0 * index / SAMPLE_RATE + 0.2)
        + 0.07 * math.sin(2.0 * math.pi * 880.0 * index / SAMPLE_RATE + 0.7)
        for index in range(SAMPLE_RATE)
    ]
    assert len(pcm) == SAMPLE_RATE and all(math.isfinite(value) for value in pcm)


def dump(model_dir: str | None, output_dir: Path) -> None:
    import numpy as np
    import torch
    import transformers
    from transformers import ClapModel, ClapProcessor

    output_dir.mkdir(parents=True, exist_ok=False)
    source_path, source_sha256 = source_hash()
    kwargs = {"revision": REVISION, "local_files_only": model_dir is not None}
    model_source = model_dir or REPOSITORY
    processor = ClapProcessor.from_pretrained(model_source, **kwargs)
    model = ClapModel.from_pretrained(model_source, torch_dtype=torch.float32, **kwargs)
    model.eval()
    resolved_revision = (
        Path(model_dir).resolve().name
        if model_dir is not None
        else getattr(model.config, "_commit_hash", None)
    )
    if resolved_revision != REVISION:
        raise RuntimeError(
            f"resolved snapshot revision is {resolved_revision!r}; expected {REVISION!r}"
        )

    state_dict = model.state_dict()
    tensor_manifest = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(state_dict.items())
    }
    pcm = deterministic_pcm()
    inputs = processor(audios=[pcm], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    embedding = outputs.audio_embeds[0].detach().cpu().to(torch.float32).numpy()
    if embedding.ndim != 1 or embedding.size != 512:
        raise RuntimeError(f"official audio embedding shape is {embedding.shape}, expected model output")
    np.asarray(pcm, dtype="<f4").tofile(output_dir / "pcm.f32")
    np.asarray(embedding, dtype="<f4").tofile(output_dir / "audio_embedding.f32")

    metadata = {
        "contract": "vokra-clap-htsat-fused-reference-v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "resolved_revision": resolved_revision,
        "sample_rate": SAMPLE_RATE,
        "dumper_version": DUMPER_VERSION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "runtime_platform": platform.platform(),
        "transformers_clap_model_source": source_path,
        "transformers_clap_model_source_sha256": source_sha256,
        "tensor_manifest": tensor_manifest,
        "model_config": model.config.to_dict(),
        "files_sha256": {},
        "parity_status": "INSPECTION_ONLY",
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "meta.json":
            metadata["files_sha256"][path.name] = sha256_file(path)
    (output_dir / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir", help="local snapshot at the pinned revision")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.model_dir is not None or args.output_dir is not None:
            parser.error("--self-test accepts no model or output arguments")
        self_test()
        print("clap_dump_reference self-test: OK")
        return 0
    if args.output_dir is None:
        parser.error("normal runs require --output-dir")
    dump(args.model_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
