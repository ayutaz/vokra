#!/usr/bin/env python3
"""Verify a completed SGMSE VAST reference directory without model loading.

The verifier does not import upstream model code or load weights. It checks the
completion manifest, all fixture hashes, source checkout identity, and the
execution environment recorded by the reference worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sgmse_dump_reference import (
    CHECKPOINT_NAME,
    CHECKPOINT_LICENSE_SPDX,
    CHECKPOINT_PARAMETER_COUNT,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_TENSOR_COUNT,
    EXPECTED_HYPERPARAM_FACTS,
    HYPERPARAMS_NAME,
    HYPERPARAMS_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    REFERENCE_FORMAT,
    EMA_ROUTE_STATUS,
    SCORE_MODEL_CONFIG,
    SOURCE_LICENSE_SHA256,
    SOURCE_LICENSE_SPDX,
    SOURCE_REVISION,
    SPEECHBRAIN_LICENSE_SHA256,
    SPEECHBRAIN_LICENSE_SPDX,
    SPEECHBRAIN_REVISION,
    cpu_model,
    git_revision,
    path_overlaps,
    reject_duplicate_json,
)


EXPECTED_ARTIFACTS = {
    "input_noisy_real",
    "input_noisy_imag",
    "input_condition_real",
    "input_condition_imag",
    "score_real",
    "score_imag",
}
EXPECTED_SHAPE = [1, 1, 256, 16]
EXPECTED_COUNT = 4_096
EXPECTED_BYTES = EXPECTED_COUNT * 4
EXPECTED_SOURCE_REPOSITORY = "https://github.com/sp-uhh/sgmse.git"
EXPECTED_SPEECHBRAIN_REPOSITORY = "https://github.com/speechbrain/speechbrain.git"
EXPECTED_INPUT = {
    "seed": 20260901,
    "sample_rate": 16_000,
    "n_fft": 510,
    "frequency_bins": 256,
    "frames": 16,
    "forward_signature": "(x_t, y, t)",
}


def require_exact_output_files(output_dir: Path) -> None:
    expected = {"manifest.json", "run.log"} | {
        f"{name}.f32" for name in EXPECTED_ARTIFACTS
    }
    entries = list(output_dir.iterdir())
    if {entry.name for entry in entries} != expected:
        raise ValueError("completion output must contain exactly the eight reference files")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"completion output contains a non-regular file: {entry.name}")


def current_determinism_contract() -> dict[str, Any]:
    """Set and observe the contract; a verifier cannot observe another process."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    return {
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "numpy_seed": 20260901,
        "torch_seed": 20260901,
    }


def require_verifier_paths(
    manifest_path: Path, output_dir: Path, vokra_root: Path
) -> None:
    if not manifest_path.is_absolute() or not output_dir.is_absolute() or not vokra_root.is_absolute():
        raise ValueError("verifier paths must be absolute")
    expected_manifest_path = output_dir / "manifest.json"
    if manifest_path.resolve(strict=False) != expected_manifest_path.resolve(strict=False):
        raise ValueError("--manifest must be output-dir/manifest.json")
    if output_dir.resolve(strict=False) == vokra_root.resolve(strict=False) or path_overlaps(output_dir, vokra_root):
        raise ValueError("reference output and Vokra checkout must be disjoint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_revision(path: Path, expected: str, label: str) -> None:
    if not path.is_dir() or path.is_symlink() or git_revision(path) != expected:
        raise ValueError(f"{label} checkout revision is not exact")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{label} checkout is dirty")


def verify_manifest(
    manifest_path: Path, output_dir: Path, vokra_root: Path
) -> dict[str, Any]:
    require_verifier_paths(manifest_path, output_dir, vokra_root)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("reference output directory is missing or symlinked")
    require_exact_output_files(output_dir)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("completion manifest is missing or symlinked")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json,
    )
    if manifest.get("format") != REFERENCE_FORMAT:
        raise ValueError("completion format mismatch")
    if manifest.get("status") != "REFERENCE_COMPLETE_NO_UPLOAD":
        raise ValueError("completion status is not successful")
    if manifest.get("publication") != "NO_UPLOAD":
        raise ValueError("completion publication is not NO_UPLOAD")
    if manifest.get("fixture_payload") != "retained_for_native_parity":
        raise ValueError("fixture payload retention is not recorded")
    if manifest.get("blockers"):
        raise ValueError("completion manifest contains blockers")
    if manifest.get("model_repository") != MODEL_REPOSITORY:
        raise ValueError("completion model repository mismatch")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("completion model revision mismatch")
    if manifest.get("fixtures") != "VAST_ONLY":
        raise ValueError("completion fixture origin is not VAST_ONLY")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("completion checkpoint identity is missing")
    if checkpoint.get("filename") != CHECKPOINT_NAME:
        raise ValueError("completion checkpoint filename mismatch")
    if checkpoint.get("size") != CHECKPOINT_SIZE:
        raise ValueError("completion checkpoint size mismatch")
    if checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("completion checkpoint identity mismatch")
    hyperparams = manifest.get("hyperparams")
    if not isinstance(hyperparams, dict):
        raise ValueError("completion hyperparams evidence is missing")
    if hyperparams.get("filename") != HYPERPARAMS_NAME or hyperparams.get("sha256") != HYPERPARAMS_SHA256 or hyperparams.get("raw_sha256") != HYPERPARAMS_SHA256:
        raise ValueError("completion hyperparams SHA mismatch")
    hyperparams_path = Path(hyperparams.get("path", ""))
    if hyperparams_path.is_symlink() or not hyperparams_path.is_file() or sha256(hyperparams_path) != HYPERPARAMS_SHA256:
        raise ValueError("completion hyperparams file is missing or tampered")
    if hyperparams.get("raw_identity") != "fixed_reviewed_bytes":
        raise ValueError("completion hyperparams raw identity is not fixed")
    if hyperparams.get("constructor_kwargs") != SCORE_MODEL_CONFIG:
        raise ValueError("completion constructor kwargs differ from reviewed config")
    if hyperparams.get("constructor_keys") != sorted(SCORE_MODEL_CONFIG):
        raise ValueError("completion constructor key set differs from reviewed config")
    facts = hyperparams.get("facts")
    if facts != {name: True for name in EXPECTED_HYPERPARAM_FACTS}:
        raise ValueError("completion hyperparams facts are incomplete or mismatched")

    source = manifest.get("source")
    speechbrain = manifest.get("speechbrain_source")
    if not isinstance(source, dict) or not isinstance(speechbrain, dict):
        raise ValueError("completion source evidence is missing")
    if source.get("repository") != EXPECTED_SOURCE_REPOSITORY:
        raise ValueError("completion SGMSE source repository mismatch")
    if source.get("revision") != SOURCE_REVISION or source.get("clean") is not True:
        raise ValueError("completion SGMSE source identity mismatch")
    if source.get("license_spdx") != SOURCE_LICENSE_SPDX or source.get("license_sha256") != SOURCE_LICENSE_SHA256:
        raise ValueError("completion SGMSE license identity mismatch")
    if speechbrain.get("revision") != SPEECHBRAIN_REVISION or speechbrain.get("clean") is not True:
        raise ValueError("completion SpeechBrain source identity mismatch")
    if speechbrain.get("repository") != EXPECTED_SPEECHBRAIN_REPOSITORY:
        raise ValueError("completion SpeechBrain source repository mismatch")
    if speechbrain.get("license_spdx") != SPEECHBRAIN_LICENSE_SPDX or speechbrain.get("license_sha256") != SPEECHBRAIN_LICENSE_SHA256:
        raise ValueError("completion SpeechBrain license identity mismatch")
    licenses = manifest.get("licenses")
    if licenses != {
        "algorithm": {"spdx": SOURCE_LICENSE_SPDX, "sha256": SOURCE_LICENSE_SHA256},
        "speechbrain": {
            "spdx": SPEECHBRAIN_LICENSE_SPDX,
            "sha256": SPEECHBRAIN_LICENSE_SHA256,
        },
        "checkpoint": CHECKPOINT_LICENSE_SPDX,
    }:
        raise ValueError("completion license identity mismatch")

    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("tensor_count") != CHECKPOINT_TENSOR_COUNT or model.get("parameter_count") != CHECKPOINT_PARAMETER_COUNT:
        raise ValueError("completion model count mismatch")
    if model.get("constructor_kwargs") != SCORE_MODEL_CONFIG or model.get("constructor_keys") != sorted(SCORE_MODEL_CONFIG):
        raise ValueError("completion model constructor evidence mismatch")
    if model.get("load") != "torch.load(weights_only=True)+load_state_dict(strict=True)":
        raise ValueError("completion model load was not strict safe-load")
    ema_route = manifest.get("ema_route")
    if not isinstance(ema_route, dict) or ema_route.get("status") != EMA_ROUTE_STATUS or ema_route.get("loadable") != "score_model_ema" or ema_route.get("unsafe_pickle_fallback") is not False:
        raise ValueError("completion EMA route evidence mismatch")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != EXPECTED_ARTIFACTS:
        raise ValueError("completion artifact set mismatch")
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict) or metadata.get("shape") != EXPECTED_SHAPE or metadata.get("count") != EXPECTED_COUNT or metadata.get("dtype") != "float32":
            raise ValueError(f"completion artifact metadata mismatch: {name}")
        if metadata.get("bytes") != EXPECTED_BYTES:
            raise ValueError(f"completion artifact byte declaration mismatch: {name}")
        filename = metadata.get("path")
        if not isinstance(filename, str) or Path(filename).name != filename or filename != f"{name}.f32":
            raise ValueError(f"completion artifact path mismatch: {name}")
        artifact = output_dir / filename
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size != EXPECTED_BYTES
            or artifact.stat().st_size != metadata.get("bytes")
            or sha256(artifact) != metadata.get("sha256")
        ):
            raise ValueError(f"completion artifact hash mismatch: {name}")

    run_log = manifest.get("run_log")
    if not isinstance(run_log, dict) or run_log.get("path") != "run.log":
        raise ValueError("completion run log evidence is missing")
    run_log_path = output_dir / "run.log"
    if run_log_path.is_symlink() or not run_log_path.is_file() or run_log_path.stat().st_size > 8_192 or sha256(run_log_path) != run_log.get("sha256"):
        raise ValueError("completion run log is missing, oversized, or tampered")
    run_log_text = run_log_path.read_text(encoding="utf-8")
    for marker in (
        "status=REFERENCE_COMPLETE_NO_UPLOAD",
        "fixture_payload=retained_for_native_parity",
        "publication=NO_UPLOAD",
    ):
        if marker not in run_log_text:
            raise ValueError(f"completion run log is missing marker: {marker}")

    vokra = manifest.get("vokra")
    if not isinstance(vokra, dict) or vokra.get("clean") is not True or vokra.get("commit") != git_revision(vokra_root):
        raise ValueError("completion Vokra commit/clean identity mismatch")
    if Path(vokra.get("path", "")).resolve(strict=False) != vokra_root.resolve(strict=False):
        raise ValueError("completion Vokra path identity mismatch")
    root_status = subprocess.run(
        ["git", "-C", str(vokra_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if root_status.strip():
        raise ValueError("Vokra checkout is dirty during completion verification")
    lock_path = vokra_root / "tools/parity/uv.lock"
    if not lock_path.is_file() or sha256(lock_path) != vokra.get("uv_lock_sha256"):
        raise ValueError("completion parity uv.lock identity mismatch")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("completion runtime evidence is missing")
    if runtime.get("platform_system") != platform.system() or runtime.get("platform_machine") != platform.machine() or runtime.get("platform_node") != platform.node() or runtime.get("cpu_model") != cpu_model() or runtime.get("nproc") != os.cpu_count():
        raise ValueError("completion platform/CPU identity mismatch")
    if not runtime.get("torch_version") or not runtime.get("numpy_version"):
        raise ValueError("completion numerical library versions are missing")
    if runtime.get("determinism") != current_determinism_contract():
        raise ValueError("completion determinism settings mismatch")

    if manifest.get("input") != EXPECTED_INPUT:
        raise ValueError("completion input contract mismatch")

    identity = manifest.get("identity")
    tool_path = vokra_root / "tools/parity/sgmse_dump_reference.py"
    if (
        not isinstance(identity, dict)
        or identity.get("reference_format") != REFERENCE_FORMAT
        or identity.get("reference_tool") != tool_path.name
        or identity.get("reference_tool_sha256") != sha256(tool_path)
        or identity.get("self_test") != "sgmse_dump_reference.py --self-test"
    ):
        raise ValueError("completion reference identity mismatch")
    inspection = manifest.get("inspection_manifest")
    if not isinstance(inspection, dict) or inspection.get("sha256") != manifest.get("inspection_manifest_sha256"):
        raise ValueError("completion inspection manifest identity mismatch")
    inspection_path = Path(inspection.get("path", ""))
    if inspection_path.is_symlink() or not inspection_path.is_file() or sha256(inspection_path) != inspection.get("sha256"):
        raise ValueError("completion inspection manifest is missing or tampered")

    source_path = Path(source.get("path", ""))
    speechbrain_path = Path(speechbrain.get("path", ""))
    clean_revision(source_path, SOURCE_REVISION, "SGMSE source")
    clean_revision(speechbrain_path, SPEECHBRAIN_REVISION, "SpeechBrain source")
    return manifest


def self_test() -> None:
    assert EXPECTED_COUNT == 4_096
    assert EXPECTED_BYTES == 16_384
    assert EXPECTED_INPUT == {
        "seed": 20260901,
        "sample_rate": 16_000,
        "n_fft": 510,
        "frequency_bins": 256,
        "frames": 16,
        "forward_signature": "(x_t, y, t)",
    }
    assert EXPECTED_SOURCE_REPOSITORY == "https://github.com/sp-uhh/sgmse.git"
    assert EXPECTED_SPEECHBRAIN_REPOSITORY == "https://github.com/speechbrain/speechbrain.git"
    assert HYPERPARAMS_SHA256 == "5ebd87c6257537c3997c134b279d85cd7bebccce0e6d3fc68f7a36f15096aa51"
    assert set(EXPECTED_ARTIFACTS) == {
        "input_noisy_real",
        "input_noisy_imag",
        "input_condition_real",
        "input_condition_imag",
        "score_real",
        "score_imag",
    }
    try:
        json.loads('{"duplicate": 1, "duplicate": 2}', object_pairs_hook=reject_duplicate_json)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON object members were accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert not (root / "output").exists()
        output = root / "output"
        output.mkdir()
        for name in {"manifest.json", "run.log"} | {
            f"{item}.f32" for item in EXPECTED_ARTIFACTS
        }:
            (output / name).write_bytes(b"")
        require_exact_output_files(output)
        (output / "unexpected").write_bytes(b"")
        try:
            require_exact_output_files(output)
        except ValueError:
            pass
        else:
            raise AssertionError("extra completion output was accepted")
        vokra = root / "vokra"
        vokra.mkdir()
        require_verifier_paths(output / "manifest.json", output, vokra)
        try:
            require_verifier_paths(root / "wrong.json", output, vokra)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong --manifest path was accepted")
        try:
            require_verifier_paths(output / "manifest.json", vokra / "nested", vokra)
        except ValueError:
            pass
        else:
            raise AssertionError("overlapping output/Vokra paths were accepted")
    print("sgmse_verify_reference self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.manifest, args.output_dir, args.vokra_root)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.manifest, args.output_dir, args.vokra_root)):
        parser.error("normal verification requires --manifest, --output-dir, and --vokra-root")
    try:
        verify_manifest(
            args.manifest, args.output_dir, args.vokra_root  # type: ignore[arg-type]
        )
    except Exception as error:  # noqa: BLE001 - verifier is a VAST gate
        print(json.dumps({"status": "BLOCKED_REFERENCE_MANIFEST_INVALID", "error": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps({"status": "REFERENCE_MANIFEST_VERIFIED", "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
