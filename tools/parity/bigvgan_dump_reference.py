#!/usr/bin/env python3
"""Dump one real-weight NVIDIA BigVGAN mel-to-waveform reference.

The oracle imports upstream ``bigvgan.py`` from a caller-supplied checkout and
loads the official ``bigvgan_generator.pt``. A tiny stand-in for upstream's
plotting-heavy ``utils`` module supplies only the two functions imported by
``bigvgan.py``; no model math is mirrored here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch  # type: ignore[import-not-found]


SOURCE_REPOSITORY = "https://github.com/NVIDIA/BigVGAN"


class _UnsafePickle:
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_checkout(path: Path, revision: str) -> None:
    """Require an immutable, clean checkout of the audited upstream source."""
    if not (path / "bigvgan.py").is_file() or not (path / "env.py").is_file():
        raise SystemExit(f"bigvgan_dump: source checkout lacks bigvgan.py/env.py: {path}")
    try:
        resolved = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        origin = subprocess.check_output(
            ["git", "-C", str(path), "remote", "get-url", "origin"], text=True
        ).strip().removesuffix(".git")
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"bigvgan_dump: source is not a git checkout: {path}: {exc}") from exc
    if resolved != revision:
        raise SystemExit(f"bigvgan_dump: source revision {resolved!r} != required {revision!r}")
    if origin != SOURCE_REPOSITORY:
        raise SystemExit(f"bigvgan_dump: source origin {origin!r} != {SOURCE_REPOSITORY!r}")
    if dirty:
        raise SystemExit(f"bigvgan_dump: source checkout is dirty: {path}")


def safe_load(path: Path) -> object:
    # Safe tensor-only unpickling is deliberate. There is no unrestricted
    # pickle fallback for an untrusted or mismatched checkpoint.
    return torch.load(path, map_location="cpu", weights_only=True)


def self_test() -> None:
    """Exercise the safe-load contract without network or model artifacts."""
    with tempfile.TemporaryDirectory(prefix="bigvgan-dump-selftest-") as directory:
        root = Path(directory)
        safe = root / "safe.pt"
        torch.save({"generator": {"weight": torch.ones(1)}}, safe)
        loaded = safe_load(safe)
        if not isinstance(loaded, dict) or "generator" not in loaded:
            raise SystemExit("bigvgan_dump self-test: safe checkpoint did not load")

        unsafe = root / "unsafe.pt"
        torch.save(_UnsafePickle(), unsafe)
        try:
            safe_load(unsafe)
        except Exception as exc:  # noqa: BLE001 - any safe-loader refusal is expected
            if "Weights only load failed" not in str(exc) and "Unsupported global" not in str(exc):
                raise SystemExit(f"bigvgan_dump self-test: unexpected safe-load error: {exc}") from exc
        else:
            raise SystemExit("bigvgan_dump self-test: unsafe pickle was accepted")

    print("bigvgan_dump_reference.py self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config-sha256")
    parser.add_argument("--source-revision")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.upstream_dir, args.checkpoint, args.checkpoint_sha256, args.config_sha256, args.source_revision, args.config, args.output)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return
    required = {
        "--upstream-dir": args.upstream_dir,
        "--checkpoint": args.checkpoint,
        "--checkpoint-sha256": args.checkpoint_sha256,
        "--config-sha256": args.config_sha256,
        "--source-revision": args.source_revision,
        "--config": args.config,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    if not args.checkpoint.is_file():
        raise SystemExit(f"bigvgan_dump: checkpoint not found: {args.checkpoint}")
    if not isinstance(args.checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_sha256):
        raise SystemExit("bigvgan_dump: --checkpoint-sha256 must be 64 lowercase hex characters")
    actual_hash = sha256_of(args.checkpoint)
    if actual_hash != args.checkpoint_sha256:
        raise SystemExit(f"bigvgan_dump: checkpoint SHA-256 {actual_hash} != required {args.checkpoint_sha256}")
    if not isinstance(args.source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise SystemExit("bigvgan_dump: --source-revision must be a 40-character lowercase git revision")
    if not isinstance(args.config_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", args.config_sha256):
        raise SystemExit("bigvgan_dump: --config-sha256 must be 64 lowercase hex characters")
    if not args.config.is_file():
        raise SystemExit(f"bigvgan_dump: config not found: {args.config}")
    actual_config_hash = sha256_of(args.config)
    if actual_config_hash != args.config_sha256:
        raise SystemExit(f"bigvgan_dump: config SHA-256 {actual_config_hash} != required {args.config_sha256}")

    upstream = args.upstream_dir.resolve()
    verify_source_checkout(upstream, args.source_revision)
    sys.path.insert(0, str(upstream))

    # bigvgan.py imports only these symbols from utils.py. Avoid importing the
    # unrelated matplotlib/librosa plotting and dataset stack into the oracle.
    upstream_utils = types.ModuleType("utils")
    upstream_utils.get_padding = lambda kernel_size, dilation=1: int(
        (kernel_size * dilation - dilation) / 2
    )
    upstream_utils.init_weights = lambda _module, mean=0.0, std=0.01: None
    sys.modules["utils"] = upstream_utils

    from bigvgan import BigVGAN  # type: ignore[import-not-found]
    from env import AttrDict  # type: ignore[import-not-found]

    config = AttrDict(json.loads(args.config.read_text(encoding="utf-8")))
    generator = BigVGAN(config)
    checkpoint = safe_load(args.checkpoint)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise SystemExit("bigvgan_dump: safe checkpoint must be a dict containing generator")
    generator.load_state_dict(checkpoint["generator"])
    generator.remove_weight_norm()
    generator.eval()

    mel = torch.tensor(
        [((index * 17) % 31 - 15) / 20.0 for index in range(config.num_mels)],
        dtype=torch.float32,
    ).reshape(1, config.num_mels, 1)
    with torch.inference_mode():
        waveform = generator(mel).reshape(-1)

    lines = [
        "input," + ",".join(f"{value:.9g}" for value in mel.reshape(-1).tolist()),
        "output,"
        + ",".join(f"{value:.9g}" for value in waveform.tolist()),
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
