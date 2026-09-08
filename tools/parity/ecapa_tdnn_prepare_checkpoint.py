#!/usr/bin/env python3
"""Prepare the official SpeechBrain ECAPA checkpoint for Rust conversion.

Only the 31 integer BatchNorm ``num_batches_tracked`` training counters are
removed. All 200 floating-point inference tensors retain their upstream names,
shapes, dtypes, and values; the Rust converter validates the full manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPECTED_FLOAT_TENSORS = 200
EXPECTED_TRAINING_COUNTERS = 31
UPSTREAM_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def reject_symlink_ancestry(path: Path | str, label: str) -> None:
    raw = os.fspath(path)
    if any(component in {".", ".."} for component in raw.split("/")):
        raise SystemExit(f"{label} must not contain lexical dot components")
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    for ancestor in (absolute, *absolute.parents):
        # macOS exposes /var as the system /private/var alias; it is not
        # user-controlled output redirection and is safe to traverse.
        if ancestor.is_symlink() and ancestor != Path("/var"):
            raise SystemExit(f"{label} has symlink ancestry: {ancestor}")


def validate_paths(input_path: Path | str, output_path: Path | str) -> None:
    reject_symlink_ancestry(input_path, "--input")
    reject_symlink_ancestry(output_path, "--output")
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_symlink() or not input_path.is_file():
        raise SystemExit("--input must be a regular non-symlink checkpoint")
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit("--output must be absent and non-symlink")


def cleanup_temp(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def save_no_clobber(tensors: dict[str, torch.Tensor], output: Path, metadata: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_ancestry(output, "--output")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise SystemExit("--output parent must be an existing regular non-symlink directory")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        save_file(tensors, temporary, metadata=metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-linking the completed temporary file is an atomic no-clobber
        # claim.  os.replace would silently overwrite a concurrent artifact.
        os.link(temporary, output)
    finally:
        cleanup_temp(temporary)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ecapa-prepare-self-test-") as directory:
        root = Path(directory)
        input_path = root / "input.ckpt"
        output_path = root / "output.safetensors"
        input_path.write_bytes(b"fixture")
        validate_paths(input_path, output_path)
        from unittest.mock import patch

        with patch.object(Path, "unlink", side_effect=PermissionError("test cleanup failure")):
            save_no_clobber({"fixture": torch.zeros(1)}, output_path, {"test": "cleanup"})
        assert output_path.read_bytes(), "final artifact was not published"
        for temporary_path in root.glob(".*.tmp"):
            os.unlink(temporary_path)
        existing = root / "existing.safetensors"
        existing.write_bytes(b"keep")
        try:
            validate_paths(input_path, existing)
        except SystemExit:
            pass
        else:
            raise AssertionError("existing output was accepted")
        assert existing.read_bytes() == b"keep"
        for path in (str(root) + "/./output", str(root) + "/../output"):
            try:
                validate_paths(input_path, path)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"lexical dot component accepted: {path}")
        real_parent = root / "real"
        real_parent.mkdir()
        symlink_parent = root / "link"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        try:
            validate_paths(input_path, symlink_parent / "output.safetensors")
        except SystemExit:
            pass
        else:
            raise AssertionError("symlink ancestor accepted")
    print("ecapa_tdnn_prepare_checkpoint self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        if args.input is not None or args.output is not None:
            parser.error("--self-test cannot be combined with --input/--output")
        self_test()
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")

    input_path = Path(args.input)
    output_path = Path(args.output)
    validate_paths(args.input, args.output)

    state = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"expected state-dict checkpoint, got {type(state).__name__}")

    counters = sorted(name for name in state if name.endswith(".num_batches_tracked"))
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in state.items()
        if name not in counters
    }
    if len(counters) != EXPECTED_TRAINING_COUNTERS:
        raise SystemExit(
            f"expected {EXPECTED_TRAINING_COUNTERS} BatchNorm counters, got {len(counters)}"
        )
    if len(tensors) != EXPECTED_FLOAT_TENSORS:
        raise SystemExit(
            f"expected {EXPECTED_FLOAT_TENSORS} inference tensors, got {len(tensors)}"
        )
    non_float = [name for name, value in tensors.items() if not value.is_floating_point()]
    if non_float:
        raise SystemExit(f"non-floating inference tensors remain: {non_float}")

    save_no_clobber(
        tensors,
        output_path,
        metadata={
            "source": "speechbrain/spkrec-ecapa-voxceleb",
            "revision": UPSTREAM_REVISION,
            "transform": "remove-batchnorm-num-batches-tracked-only",
        },
    )
    print(
        json.dumps(
            {
                "input_entries": len(state),
                "written_tensors": len(tensors),
                "removed_counters": counters,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
