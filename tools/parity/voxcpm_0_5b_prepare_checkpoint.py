#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Prepare an authenticated VoxCPM-0.5B composite evidence manifest.

The 0.5B release is a main ``pytorch_model.bin`` plus a separately loaded
``audiovae.pth`` state dict.  This tool deliberately writes evidence only;
it never emits a GGUF or a replacement model.  Both checkpoints are loaded
with PyTorch's restricted ``weights_only=True`` mode on VAST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HF_REPOSITORY = "openbmb/VoxCPM-0.5B"
HF_REVISION = "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REPOSITORY = "https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION = "38a76704ee67935ccbafbe5b6725e83dbb1e9305"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_state(path: Path, label: str) -> Any:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("torch is required on VAST for checkpoint preparation") from error
    unsafe = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if unsafe is not None and unsafe(str(path)):
        raise RuntimeError(f"{label} contains unsafe pickle globals")
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - checkpoint errors become blockers
        raise RuntimeError(f"{label}: torch.load(weights_only=True) failed: {error}") from error


def walk_tensors(value: Any, label: str, *, component: str, prefix: str = "") -> list[dict[str, Any]]:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("torch is required on VAST for checkpoint preparation") from error
    rows: list[dict[str, Any]] = []

    def visit(item: Any, name: str, depth: int) -> None:
        if depth > 32 or len(rows) >= 300_000:
            raise RuntimeError(f"{label}: bounded tensor walk exceeded")
        if isinstance(item, torch.Tensor):
            if item.layout != torch.strided:
                raise RuntimeError(f"{label}: unsupported tensor layout at {name}")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise RuntimeError(f"{label}: non-finite tensor at {name}")
            staged_name = f"{prefix}{name}"
            rows.append(
                {
                    "component": component,
                    "original_name": name,
                    "staged_name": staged_name,
                    "name": staged_name,
                    "shape": [int(axis) for axis in item.shape],
                    "dtype": str(item.dtype),
                    "elements": int(item.numel()),
                }
            )
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                if not isinstance(key, str):
                    raise RuntimeError(f"{label}: non-string state key")
                visit(item[key], f"{name}.{key}" if name else key, depth + 1)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]", depth + 1)
        elif item is None or isinstance(item, (str, int, float, bool)):
            return
        else:
            raise RuntimeError(f"{label}: unsupported state value {type(item).__name__}")

    visit(value, "", 0)
    if not rows:
        raise RuntimeError(f"{label}: empty tensor manifest")
    rows.sort(key=lambda row: row["name"])
    if len({row["name"] for row in rows}) != len(rows):
        raise RuntimeError(f"{label}: duplicate tensor names")
    return rows


def prepare(main: Path, audiovae: Path, output: Path) -> None:
    if not main.is_file() or not audiovae.is_file():
        raise RuntimeError("both pytorch_model.bin and audiovae.pth are required")
    main_value = load_state(main, "pytorch_model.bin")
    main_container = "state_dict" if isinstance(main_value, dict) and "state_dict" in main_value else "root"
    if main_container == "state_dict":
        main_value = main_value["state_dict"]
    main_rows = walk_tensors(main_value, "pytorch_model.bin", component="main")
    vae_value = load_state(audiovae, "audiovae.pth")
    vae_container = "state_dict" if isinstance(vae_value, dict) and "state_dict" in vae_value else "root"
    if isinstance(vae_value, dict) and "state_dict" in vae_value:
        vae_value = vae_value["state_dict"]
    vae_rows = walk_tensors(vae_value, "audiovae.pth", component="audiovae", prefix="audio_vae.")
    names = {row["staged_name"] for row in main_rows}
    overlap = names.intersection(row["staged_name"] for row in vae_rows)
    if overlap:
        raise RuntimeError(f"composite namespace collision: {sorted(overlap)[:3]}")
    combined = sorted(main_rows + vae_rows, key=lambda row: row["name"])
    result = {
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "preparation_status": "PREPARATION_EVIDENCE_COMPLETE",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "main_sha256": sha256_file(main),
        "audiovae_sha256": sha256_file(audiovae),
        "main": {"container_path": main_container, "tensor_count": len(main_rows), "manifest_sha256": canonical_manifest(main_rows), "tensors": main_rows},
        "audiovae": {"container_path": vae_container, "tensor_count": len(vae_rows), "manifest_sha256": canonical_manifest(vae_rows), "tensors": vae_rows},
        "composite": {"tensor_count": len(combined), "manifest_sha256": canonical_manifest(combined), "namespace": "audio_vae.", "rows_use_original_and_staged_names": True},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    rows = [{"name": "a", "shape": [2], "dtype": "torch.float32", "elements": 2}]
    assert canonical_manifest(rows) == canonical_manifest(rows)
    assert HF_REVISION == "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
    assert SOURCE_REVISION == "38a76704ee67935ccbafbe5b6725e83dbb1e9305"
    assert rows[0]["name"] not in {"audio_vae." + rows[0]["name"]}
    print("voxcpm_0_5b_prepare_checkpoint --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--main", type=Path)
    parser.add_argument("--audiovae", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not all((args.main, args.audiovae, args.output)):
        parser.error("--main, --audiovae and --output are required")
    try:
        prepare(args.main, args.audiovae, args.output)
    except Exception as error:  # noqa: BLE001 - preserve blocker evidence
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "BLOCKED",
            "evidence_stage": "INSPECTION_ONLY",
            "preparation_status": "PREPARATION_ERROR",
            "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
            "cpu_status": "UNSUPPORTED",
            "metal_status": "BLOCKED_BY_CPU",
            "parity_status": "NOT_RUN",
            "publication": "NO_UPLOAD",
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "error": f"{type(error).__name__}: {error}",
        }
        args.output.write_text(json.dumps(failure, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"voxcpm_0_5b_prepare_checkpoint: BLOCKED: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
