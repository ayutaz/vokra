#!/usr/bin/env python3
"""Prepare the pinned official DNS48 checkpoint for the Rust converter.

This offline-only sidecar validates the exact clean upstream source checkout
and the official checkpoint's size plus PyTorch SHA-256 filename prefix,
strict-loads it into the real ``denoiser.pretrained.dns48`` model, then writes
the 48 F32 tensors as safetensors. It never defines a mirror model.

Run checkpoint handling on VAST through the repository environment:

    uv run --project tools/parity python \
      tools/parity/facebook_denoiser_prepare_checkpoint.py \
      --source /root/denoiser --checkpoint /root/dns48-11decc9d8e3f0998.th \
      --output /root/facebook-denoiser.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


SOURCE_REVISION = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0"
CHECKPOINT_BYTES = 75_478_395
CHECKPOINT_SHA256_PREFIX = "11decc9d8e3f0998"
TENSOR_COUNT = 48
SOURCE_FILES = {
    Path("denoiser/demucs.py"): (
        17_080,
        "8e9c21935c647e24f31cefcc63a298cb2a1c25bc99aab44bbe63a7b5570836be",
    ),
    Path("denoiser/resample.py"): (
        2_187,
        "3e8ea258036660b7d33415794fe09ee010510f4d760bdfc5d5de268d6efb40f5",
    ),
    Path("denoiser/pretrained.py"): (
        3_070,
        "885ad1ddd6cee5d4ecf5b4bc32784ceee97dc37ae19570b7ce0f9869b360d108",
    ),
    Path("LICENSE"): (
        19_333,
        "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def validate_source(checkout: Path) -> Path:
    checkout = checkout.resolve()
    if git_output(checkout, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise ValueError(f"source checkout must be revision {SOURCE_REVISION}")
    if git_output(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official source checkout must be exactly clean")
    for relative, (expected_bytes, expected_sha256) in SOURCE_FILES.items():
        path = checkout / relative
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{relative}: unexpected byte length")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{relative}: unexpected SHA-256")
    return checkout


def validate_checkpoint(path: Path) -> str:
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError(
            f"checkpoint has {path.stat().st_size} bytes, expected {CHECKPOINT_BYTES}"
        )
    digest = sha256_file(path)
    if not digest.startswith(CHECKPOINT_SHA256_PREFIX):
        raise ValueError(
            f"checkpoint SHA-256 {digest} does not start with official prefix "
            f"{CHECKPOINT_SHA256_PREFIX}"
        )
    return digest


def prepare(source: Path, checkpoint: Path, output: Path) -> None:
    import torch
    from safetensors.torch import save_file

    source = validate_source(source)
    checkpoint_sha256 = validate_checkpoint(checkpoint)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    try:
        from denoiser.pretrained import dns48

        imported = Path(sys.modules[dns48.__module__].__file__).resolve()
    except Exception as error:  # noqa: BLE001 - loud official-import boundary
        raise RuntimeError(
            "could not import the pinned official denoiser package; mirror fallback forbidden"
        ) from error
    finally:
        sys.path.pop(0)
    expected_import = (source / "denoiser/pretrained.py").resolve()
    if imported != expected_import:
        raise ValueError(f"imported denoiser from {imported}, expected {expected_import}")

    model = dns48(pretrained=False).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"official checkpoint yielded {type(state)!r}, expected state dict")
    model.load_state_dict(state, strict=True)
    if len(state) != TENSOR_COUNT:
        raise ValueError(f"checkpoint has {len(state)} tensors, expected {TENSOR_COUNT}")
    non_f32 = sorted(
        name for name, value in state.items() if value.dtype != torch.float32
    )
    if non_f32:
        raise ValueError(f"checkpoint contains non-F32 tensors: {non_f32!r}")
    contiguous = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        contiguous,
        str(output),
        metadata={
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "model": "facebookresearch/denoiser dns48",
        },
    )
    print(f"wrote {output} ({output.stat().st_size} bytes, {len(contiguous)} tensors)")
    print(f"checkpoint_sha256={checkpoint_sha256}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not all([args.source, args.checkpoint, args.output]):
        parser.error("--source, --checkpoint and --output are required")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        if args.source is not None:
            validate_source(args.source)
        assert len(SOURCE_REVISION) == 40
        assert len(CHECKPOINT_SHA256_PREFIX) == 16
        assert len(SOURCE_FILES) == 4
        print("facebook_denoiser_prepare_checkpoint self-test: ok")
        return 0
    prepare(args.source, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
