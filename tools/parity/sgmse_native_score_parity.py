#!/usr/bin/env python3
"""Compare a Vokra SGMSE score dump with the independent VAST reference.

This is a fixture consumer, not a model implementation.  The reference
directory must be produced by ``sgmse_dump_reference.py`` from the pinned
upstream SpeechBrain/SGMSE source.  The native directory is deliberately a
small raw-f32 interchange boundary for a future Vokra native runner: it must
contain exactly ``score_real.f32`` and ``score_imag.f32`` and no other files.
No checkpoint is loaded and no reference values are generated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path
from typing import Any


REFERENCE_FORMAT = "vokra-sgmse-score-reference-v1"
REFERENCE_STATUS = "REFERENCE_COMPLETE_NO_UPLOAD"
MODEL_REPOSITORY = "speechbrain/sgmse-voicebank"
MODEL_REVISION = "8f4ff7b65284c49492a43349b8106e094ac0d365"
SOURCE_REPOSITORY = "https://github.com/sp-uhh/sgmse.git"
SOURCE_REVISION = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
SOURCE_LICENSE_SPDX = "mit"
SPEECHBRAIN_REPOSITORY = "https://github.com/speechbrain/speechbrain.git"
SPEECHBRAIN_REVISION = "2b3f4f44351fd08a627c4ab307de5c420351bc19"
SPEECHBRAIN_LICENSE_SPDX = "apache-2.0"
CHECKPOINT_LICENSE_SPDX = "apache-2.0"
REFERENCE_SHAPE = [1, 1, 256, 64]
REFERENCE_COUNT = 16_384
REFERENCE_BYTES = REFERENCE_COUNT * 4
FP32_ATOL = 0.01
SCORE_NAMES = ("score_real", "score_imag")
REFERENCE_ARTIFACT_NAMES = {
    "input_noisy_real",
    "input_noisy_imag",
    "input_condition_real",
    "input_condition_imag",
    "score_real",
    "score_imag",
}
NATIVE_ARTIFACT_NAMES = {f"{name}.f32" for name in SCORE_NAMES}
# The manifest intentionally carries run-specific Vokra commit, host, and
# absolute-path evidence, so its whole-file digest is not a stable identity.
# These six payload digests were byte-identical across three exact VAST runs
# and are the reviewed oracle for this consumer.  A manifest cannot authorize
# a different payload merely by rewriting its own `sha256` field.
REVIEWED_ARTIFACT_SHA256 = {
    "input_condition_imag": "37d4a9e7d1793aaef270cdbaddf69464fe3286171661c8f75380bd6f6e305893",
    "input_condition_real": "8fa96184edbec9c85856eebabd6ba6102fee3e30debaf7aee2fffffd1e9599ea",
    "input_noisy_imag": "a355948bcbafb8b89a3975d40ee333129216e730e153d8ef26d5419ed07f90ba",
    "input_noisy_real": "c62e324c7826c752b2a8b567d184bca31cd9e1dd6b1ac04885eb78f1ccf325fa",
    "score_imag": "ea029f909ed9eae729b2b52e51807847aece53ee574435b3b3c0f3bb713b25d5",
    "score_real": "f15e232711181167317c820b3e0c12f07fcad8f30cd431031d196aedabbda16b",
}
EXPECTED_INPUT = {
    "seed": 20260901,
    "sample_rate": 16_000,
    "n_fft": 510,
    "frequency_bins": 256,
    "frames": 64,
    "forward_signature": "(x_t, y, t)",
}


def reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object members instead of accepting the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or symlinked: {path}")


def require_exact_directory_files(directory: Path, expected: set[str], label: str) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{label} directory is missing or symlinked: {directory}")
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != expected:
        raise ValueError(f"{label} directory has unexpected or missing files")
    for entry in entries:
        require_regular_file(entry, f"{label} entry")


def paths_overlap(left: Path, right: Path) -> bool:
    """Return whether canonical paths are equal or one contains the other."""
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def read_f32(path: Path, label: str, count: int) -> list[float]:
    require_regular_file(path, label)
    if path.stat().st_size != count * 4:
        raise ValueError(f"{label} byte count mismatch")
    values: list[float] = []
    with path.open("rb") as handle:
        for raw in iter(lambda: handle.read(4), b""):
            if len(raw) != 4:
                raise ValueError(f"{label} has a truncated f32 value")
            value = struct.unpack("<f", raw)[0]
            if not math.isfinite(value):
                raise ValueError(f"{label} contains a non-finite value")
            values.append(value)
    if len(values) != count:
        raise ValueError(f"{label} element count mismatch")
    return values


def verify_reference(
    reference_dir: Path,
) -> dict[str, Any]:
    """Verify the fixed reference contract and return its manifest."""
    manifest_path = reference_dir / "manifest.json"
    require_regular_file(manifest_path, "reference manifest")
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json,
    )
    if manifest.get("format") != REFERENCE_FORMAT:
        raise ValueError("reference format mismatch")
    if manifest.get("status") != REFERENCE_STATUS:
        raise ValueError("reference is not a completed VAST run")
    if manifest.get("publication") != "NO_UPLOAD" or manifest.get("fixtures") != "VAST_ONLY":
        raise ValueError("reference publication/origin contract mismatch")
    if manifest.get("fixture_payload") != "retained_for_native_parity":
        raise ValueError("reference fixture payload was not retained")
    if manifest.get("model_repository") != MODEL_REPOSITORY or manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("reference model identity mismatch")
    source = manifest.get("source")
    speechbrain = manifest.get("speechbrain_source")
    if not isinstance(source, dict) or source.get("repository") != SOURCE_REPOSITORY or source.get("revision") != SOURCE_REVISION:
        raise ValueError("reference SGMSE source identity mismatch")
    if not isinstance(speechbrain, dict) or speechbrain.get("repository") != SPEECHBRAIN_REPOSITORY or speechbrain.get("revision") != SPEECHBRAIN_REVISION:
        raise ValueError("reference SpeechBrain source identity mismatch")
    if source.get("license_spdx") != SOURCE_LICENSE_SPDX or speechbrain.get("license_spdx") != SPEECHBRAIN_LICENSE_SPDX:
        raise ValueError("reference source license identity mismatch")
    licenses = manifest.get("licenses")
    if (
        not isinstance(licenses, dict)
        or not isinstance(licenses.get("algorithm"), dict)
        or licenses["algorithm"].get("spdx") != SOURCE_LICENSE_SPDX
        or not isinstance(licenses.get("speechbrain"), dict)
        or licenses["speechbrain"].get("spdx") != SPEECHBRAIN_LICENSE_SPDX
        or licenses.get("checkpoint") != CHECKPOINT_LICENSE_SPDX
    ):
        raise ValueError("reference license identity mismatch")
    if manifest.get("input") != EXPECTED_INPUT:
        raise ValueError("reference input contract mismatch")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("reference_format") != REFERENCE_FORMAT or identity.get("reference_tool") != "sgmse_dump_reference.py":
        raise ValueError("reference tool identity mismatch")
    ema_route = manifest.get("ema_route")
    if not isinstance(ema_route, dict) or ema_route.get("status") != "SOURCE_ROUTE_VERIFIED_STRICT_LOAD" or ema_route.get("unsafe_pickle_fallback") is not False:
        raise ValueError("reference EMA route identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REFERENCE_ARTIFACT_NAMES:
        raise ValueError("reference artifact set mismatch")
    require_exact_directory_files(
        reference_dir,
        {"manifest.json", "run.log"} | {f"{name}.f32" for name in REFERENCE_ARTIFACT_NAMES},
        "reference",
    )
    run_log = reference_dir / "run.log"
    run_log_evidence = manifest.get("run_log")
    if not isinstance(run_log_evidence, dict):
        raise ValueError("reference run log evidence is missing")
    if run_log_evidence.get("path") != "run.log":
        raise ValueError("reference run log path mismatch")
    if run_log_evidence.get("size") != run_log.stat().st_size or run_log_evidence.get("sha256") != sha256(run_log):
        raise ValueError("reference run log hash or size mismatch")
    if run_log.stat().st_size > 8_192:
        raise ValueError("reference run log is oversized")
    run_log_text = run_log.read_text(encoding="utf-8")
    for marker in (
        "status=REFERENCE_COMPLETE_NO_UPLOAD",
        "fixture_payload=retained_for_native_parity",
        "publication=NO_UPLOAD",
    ):
        if marker not in run_log_text:
            raise ValueError(f"reference run log is missing marker: {marker}")
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict) or metadata.get("shape") != REFERENCE_SHAPE or metadata.get("count") != REFERENCE_COUNT or metadata.get("bytes") != REFERENCE_BYTES or metadata.get("dtype") != "float32":
            raise ValueError(f"reference artifact metadata mismatch: {name}")
        expected_sha256 = REVIEWED_ARTIFACT_SHA256.get(name)
        if expected_sha256 is None or metadata.get("sha256") != expected_sha256:
            raise ValueError(f"reference artifact {name} is not a reviewed VAST payload")
        filename = metadata.get("path")
        if filename != f"{name}.f32":
            raise ValueError(f"reference artifact path mismatch: {name}")
        artifact = reference_dir / filename
        if artifact.stat().st_size != REFERENCE_BYTES or sha256(artifact) != expected_sha256:
            raise ValueError(f"reference artifact hash mismatch: {name}")
    return manifest


def compare_native(
    reference_dir: Path,
    native_dir: Path,
) -> dict[str, Any]:
    """Compare native score files against the authenticated VAST scores."""
    if not reference_dir.is_absolute() or not native_dir.is_absolute():
        raise ValueError("reference and native paths must be absolute")
    if paths_overlap(reference_dir, native_dir):
        raise ValueError("reference and native directories must be disjoint")
    manifest = verify_reference(reference_dir)
    require_exact_directory_files(native_dir, NATIVE_ARTIFACT_NAMES, "native score")
    metrics: dict[str, Any] = {}
    for name in SCORE_NAMES:
        expected = read_f32(reference_dir / f"{name}.f32", f"reference {name}", REFERENCE_COUNT)
        actual = read_f32(native_dir / f"{name}.f32", f"native {name}", REFERENCE_COUNT)
        differences = [abs(left - right) for left, right in zip(actual, expected)]
        max_abs = max(differences)
        max_index = differences.index(max_abs)
        metrics[name] = {
            "max_abs": max_abs,
            "max_index": max_index,
            "mean_abs": sum(differences) / len(differences),
            "atol": FP32_ATOL,
        }
        if max_abs > FP32_ATOL:
            raise ValueError(f"native {name} exceeds FP32 atol at index {max_index}: {max_abs}")
    return {"status": "SGMSE_NATIVE_SCORE_PARITY_PASS", "reference": manifest["identity"], "metrics": metrics}


def self_test() -> None:
    assert REFERENCE_SHAPE == [1, 1, 256, 64]
    assert REFERENCE_COUNT == 16_384
    assert REFERENCE_BYTES == 65_536
    assert FP32_ATOL == 0.01
    assert set(REVIEWED_ARTIFACT_SHA256) == REFERENCE_ARTIFACT_NAMES
    assert all(
        len(digest) == 64
        and digest.isascii()
        and digest.islower()
        and all(character in "0123456789abcdef" for character in digest)
        for digest in REVIEWED_ARTIFACT_SHA256.values()
    )
    try:
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=reject_duplicate_json)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON members were accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference = root / "reference"
        native = root / "native"
        reference.mkdir()
        native.mkdir()
        artifacts: dict[str, Any] = {}
        for name in REFERENCE_ARTIFACT_NAMES:
            path = reference / f"{name}.f32"
            path.write_bytes(b"\0" * REFERENCE_BYTES)
            artifacts[name] = {
                "path": path.name,
                "dtype": "float32",
                "shape": REFERENCE_SHAPE,
                "count": REFERENCE_COUNT,
                "bytes": REFERENCE_BYTES,
                "sha256": sha256(path),
            }
        # The fixture is synthetic and never leaves this self-test. Swap in
        # local digests only to exercise the exact verification wiring;
        # production calls retain the reviewed VAST constants above.
        reviewed_digests = REVIEWED_ARTIFACT_SHA256.copy()
        REVIEWED_ARTIFACT_SHA256.clear()
        REVIEWED_ARTIFACT_SHA256.update(
            {name: metadata["sha256"] for name, metadata in artifacts.items()}
        )
        manifest = {
            "format": REFERENCE_FORMAT,
            "status": REFERENCE_STATUS,
            "publication": "NO_UPLOAD",
            "fixtures": "VAST_ONLY",
            "fixture_payload": "retained_for_native_parity",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license_spdx": SOURCE_LICENSE_SPDX},
            "speechbrain_source": {"repository": SPEECHBRAIN_REPOSITORY, "revision": SPEECHBRAIN_REVISION, "license_spdx": SPEECHBRAIN_LICENSE_SPDX},
            "licenses": {
                "algorithm": {"spdx": SOURCE_LICENSE_SPDX},
                "speechbrain": {"spdx": SPEECHBRAIN_LICENSE_SPDX},
                "checkpoint": CHECKPOINT_LICENSE_SPDX,
            },
            "input": EXPECTED_INPUT,
            "artifacts": artifacts,
            "identity": {"reference_format": REFERENCE_FORMAT, "reference_tool": "sgmse_dump_reference.py"},
            "ema_route": {"status": "SOURCE_ROUTE_VERIFIED_STRICT_LOAD", "unsafe_pickle_fallback": False},
        }
        (reference / "run.log").write_text(
            "status=REFERENCE_COMPLETE_NO_UPLOAD\n"
            "fixture_payload=retained_for_native_parity\n"
            "publication=NO_UPLOAD\n",
            encoding="utf-8",
        )
        manifest["run_log"] = {
            "path": "run.log",
            "size": (reference / "run.log").stat().st_size,
            "sha256": sha256(reference / "run.log"),
        }
        (reference / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for name in SCORE_NAMES:
            (native / f"{name}.f32").write_bytes(b"\0" * REFERENCE_BYTES)
        result = compare_native(reference.resolve(), native.resolve())
        assert result["status"] == "SGMSE_NATIVE_SCORE_PARITY_PASS"

        # A rewritten manifest must not be able to bless altered reference
        # bytes: changing both the payload and its self-declared digest still
        # fails against the reviewed per-artifact allow-list.
        score_reference = reference / "score_real.f32"
        original_reference_score = score_reference.read_bytes()
        score_reference.write_bytes(b"\x01" + original_reference_score[1:])
        manifest["artifacts"]["score_real"]["sha256"] = sha256(score_reference)
        (reference / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            verify_reference(reference.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError("rewritten manifest authorized altered reference values")
        score_reference.write_bytes(original_reference_score)
        manifest["artifacts"]["score_real"]["sha256"] = artifacts["score_real"]["sha256"]
        (reference / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        run_log = reference / "run.log"
        original_log = run_log.read_bytes()
        run_log.write_bytes(original_log + b"tampered\n")
        try:
            compare_native(reference.resolve(), native.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError("run-log hash mismatch was accepted")
        run_log.write_bytes(original_log)
        native_score = native / "score_real.f32"
        original_score = native_score.read_bytes()
        native_score.write_bytes(struct.pack("<f", FP32_ATOL * 2) + original_score[4:])
        try:
            compare_native(reference.resolve(), native.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError("score tolerance mismatch was accepted")
        native_score.write_bytes(struct.pack("<f", math.nan) + original_score[4:])
        try:
            compare_native(reference.resolve(), native.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite native score was accepted")
        native_score.write_bytes(original_score)
        (native / "unexpected").write_bytes(b"")
        try:
            compare_native(reference.resolve(), native.resolve())
        except ValueError:
            pass
        else:
            raise AssertionError("extra native output was accepted")
        REVIEWED_ARTIFACT_SHA256.clear()
        REVIEWED_ARTIFACT_SHA256.update(reviewed_digests)
    print("sgmse_native_score_parity self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--native-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.reference_dir is not None or args.native_dir is not None:
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if args.reference_dir is None or args.native_dir is None:
        parser.error("normal parity requires --reference-dir and --native-dir")
    try:
        result = compare_native(args.reference_dir, args.native_dir)
    except Exception as error:  # noqa: BLE001 - parity gate must fail closed
        print(json.dumps({"status": "BLOCKED_SGMSE_NATIVE_SCORE_PARITY", "error": f"{type(error).__name__}: {error}"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
