"""microWakeWord host parity reference dumper (M5-03b Phase 4).

Offline sidecar tool (FR-LD-05: no Python / TFLite / TensorFlow ever enters
the runtime). Companion to ``prepare_checkpoint.py`` (Phase 1: TFLite → GGUF
weight extraction) — this Phase 4 script produces the reference artefacts
that the Rust host-parity harness
(``crates/vokra-kws-micro/tests/parity_microwakeword.rs``) reads to compare
against Vokra's [`vokra_kws_micro::features::FeatureExtractor`] and its
future INT8 forward chain.

# What this script emits

Given a source ``hey_jarvis.tflite`` (kahrendt/microWakeWord / ESPHome
micro-wake-word-models, Apache-2.0), this script writes into
``--output-dir`` the following files:

- ``input_pcm.bin`` — raw ``i16`` little-endian, exactly
    ``WINDOW_SAMPLES = 512`` samples (a 32 ms window @ 16 kHz), synthesised
    deterministically from a fixed seed. This is the PCM the Rust
    [`FeatureExtractor::compute_frame_f32`] consumes to reproduce the
    features side of the parity comparison.
- ``features_ref.bin`` — raw ``f32`` little-endian, exactly ``N_MELS = 40``
    floats. The reference log-mel features produced by a **numpy
    transcription** of the standard log-mel algorithm (Hann window +
    radix-2 FFT + HTK-convention mel filterbank + log10 with 1e-10 floor)
    against ``input_pcm.bin``.
- ``input_invocation_NN.bin`` — exact quantised ``int8`` bytes for one
    ``[1, 3, 40]`` invocation, with distinct frames per invocation.
- ``output_invocation_NN.bin`` / ``output_invocation_NN_f32.bin`` — exact
    raw ``uint8 [1, 1]`` output and affine dequantisation. The model is run
    through one persistent upstream interpreter and replayed in a fresh
    interpreter as a reset check.
- ``manifest.json`` — describes each artefact (name, path, shape,
    dtype, atol recommendation). Also carries the source ``.tflite``
    sha256 for provenance audit.

# What "reference" means here — honest boundary

The numpy reference for ``features_ref.bin`` is a **transcription** of
the same log-mel algorithm the Rust code implements — Hann window,
radix-2 FFT, HTK-convention mel filterbank, log10 with floor. This
validates *transcription faithfulness*: the Rust code implements the
standard algorithm it claims to implement, and matches an independent
numpy pass at the registered ``atol = 5e-2`` boundary on real inputs.

What this does **not** validate: bit-parity against the specific
training-time ``tf.signal`` mel front-end used to train the
microWakeWord checkpoints. Bit-parity against ``tf.signal`` would
require pulling ``tensorflow`` (~500 MB) into the sidecar's dep
footprint (currently 2 deps: ``numpy`` + ``ai-edge-litert``). The standard
log-mel algorithm is not compared with ``tf.signal.stft`` plus
``tf.signal.linear_to_mel_weight_matrix`` here. The registered 5e-2 boundary
applies only to this numpy transcription versus Rust f32 comparison.

The per-invocation output reference is the **real** upstream TFLite forward:
``ai_edge_litert.Interpreter`` runs the exact quantised MC-MobileNet
operations the checkpoint was trained with. That leg has no "transcription"
concern — it is the ground truth for the INT8 forward.

# NOT REFERENCED (clean-room)

- ``kahrendt/microWakeWord`` Python training code (Apache-2.0 — never
    vendored, never re-implemented; ``.tflite`` consumed as opaque
    black-box weights).
- ``esphome/esphome`` micro_wake_word component (GPL-3.0 — never
    imported, never inspected; see ``prepare_checkpoint.py``'s own
    NOT-REFERENCED list).

# Usage

::

    cd tools/parity/microwakeword-reference
    # VAST must first complete inspect.py's dependency/native-license audit.
    # The current result is BLOCKED_PENDING_VAST_EVIDENCE; do not generate
    # fixtures until the audit report explicitly permits it.
    uv run --no-project --offline --python 3.12 python inspect.py
    # Only after the VAST audit is PASS, and only with the owner-provided
    # regular (non-symlink) .tflite whose SHA is authenticated:
    # --output-dir must be a new absent sibling; existing paths are rejected.
    uv run python ../microwakeword/dump_reference.py \\
        --tflite-path ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \\
        --output-dir  ~/.cache/vokra-eval/fixtures/microwakeword \\
        --dependency-evidence /absolute/path/to/dependency-evidence.json \\
        --verbose

    The dependency-evidence file is the successful collection report from
    the same clean VAST environment. The dumper rechecks its fixed schema,
    hashes, platform, and installed distribution versions, then requires the
    Inspector's exact-owner-reviewed audit decision. The manifest keeps the
    raw collection status separate from the effective fixture decision;
    publication remains prohibited.

    # Point the Rust parity harness at both artefacts (the GGUF was
    # produced by prepare_checkpoint.py in a separate step):
    export VOKRA_KWS_REAL_GGUF=~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
    export VOKRA_KWS_REAL_FIXTURES=~/.cache/vokra-eval/fixtures/microwakeword
    CARGO_BUILD_JOBS=1 cargo test -p vokra-kws-micro \\
        --test parity_microwakeword -- --nocapture

Fails loudly on any anomaly (missing .tflite, dtype mismatch,
FeatureExtractor output length wrong, ...) rather than masking it —
FR-EX-08 posture, matches every other sidecar in ``tools/parity/``.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# Constants — must mirror ``crates/vokra-kws-micro/src/features.rs``
# and ``prepare_checkpoint.py`` exactly. A silent drift in any of these
# would misalign the numpy reference against the Rust
# ``FeatureExtractor``, and the parity harness would fail loudly (which
# is the whole point — this is a permanent guard, not a moving target).
# ----------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
HOP_MS: int = 10
WINDOW_MS: int = 32
N_MELS: int = 40

HOP_SAMPLES: int = SAMPLE_RATE * HOP_MS // 1000     # 160
WINDOW_SAMPLES: int = SAMPLE_RATE * WINDOW_MS // 1000  # 512
N_FFT: int = 512
N_BINS: int = N_FFT // 2 + 1                          # 257
LOG_MEL_EPSILON: float = 1e-10

# Deterministic PCM synthesis: a 440 Hz sine + light gaussian noise, so
# the reference has real spectral content across multiple mel bands
# (a pure sine would concentrate energy in one bin, hiding filterbank
# regressions; pure noise would flatten every band, hiding FFT
# regressions).
PCM_SEED: int = 0
PCM_SINE_HZ: float = 440.0
PCM_SINE_AMPLITUDE: float = 6000.0    # ~1/5 of int16 range → no clipping
PCM_NOISE_STDDEV: float = 200.0        # small vs sine → sine dominates

# Authenticated release contract. Keep this aligned with the converter's
# reviewed constants; a different model/topology must fail closed.
AUTHENTICATED_TFLITE_SHA256 = (
    "21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"
)
AUTHENTICATED_TFLITE_BYTES = 52272
INPUT_SHAPE = (1, 3, 40)
INPUT_DTYPE_NAME = "int8"
INPUT_SCALE = 0.10196078568696976
INPUT_ZERO_POINT = -128
OUTPUT_SHAPE = (1, 1)
OUTPUT_DTYPE_NAME = "uint8"
OUTPUT_SCALE = 1.0 / 256.0
OUTPUT_ZERO_POINT = 0
INVOCATION_COUNT = 4
FRAMES_PER_INVOCATION = INPUT_SHAPE[1]

DEPENDENCY_EVIDENCE_SCHEMA = "microwakeword-reference-dependency-evidence-v1"
DEPENDENCY_EVIDENCE_STATUS = "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED"
EXPECTED_REFERENCE_PROJECT_SHA256 = "2438d719428e497cc7f101429ba31fb5016e72737659d55aa0269d0824b1183d"
EXPECTED_REFERENCE_LOCK_SHA256 = "736fca6145c24984531ef11258cd64aebbb188fa8830300b09232cac0fe567f3"
EXPECTED_REFERENCE_DISTRIBUTIONS = {
    "ai-edge-litert": "2.1.5",
    "backports-strenum": "1.3.1",
    "flatbuffers": "25.12.19",
    "numpy": "2.5.2",
    "protobuf": "7.36.1",
    "tqdm": "4.70.0",
    "typing-extensions": "4.16.0",
}

# Compile-time contracts (mirror the Rust `const _:` asserts).
assert WINDOW_SAMPLES <= N_FFT, "WINDOW_SAMPLES must fit in N_FFT"
assert (N_FFT & (N_FFT - 1)) == 0, "N_FFT must be a power of two (radix-2)"


def sha256_of_file(path: Path) -> str:
    """Streamed hex sha256 of the file. Used to stamp the source
    ``.tflite`` provenance into the manifest so a future Rust-side
    fixture-integrity check can catch a drifted upstream."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DuplicateJsonKey(ValueError):
    """Raised when evidence JSON contains a duplicate object key."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate dependency-evidence key: {key}")
        result[key] = value
    return result


def load_dependency_evidence(path: Path) -> tuple[dict[str, Any], str, bytes]:
    """Load one regular evidence file and return its bytes digest.

    This parser is stdlib-only and runs before NumPy/LiteRT imports. It is a
    collection-evidence integrity gate, not a license or publication grant.
    """
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"--dependency-evidence must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, DuplicateJsonKey) as error:
        raise SystemExit(f"invalid dependency evidence JSON: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("dependency evidence root must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest(), payload


def _exact_keys(value: dict[str, Any], expected: tuple[str, ...], label: str) -> None:
    if set(value) != set(expected) or len(value) != len(expected):
        raise SystemExit(f"{label} keys drift: {sorted(value)}")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise SystemExit(f"{label} must be lowercase SHA-256 hex")
    return value


def _normalize_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_exact_versions(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(EXPECTED_REFERENCE_DISTRIBUTIONS):
        raise SystemExit(
            f"{label} must contain exactly {len(EXPECTED_REFERENCE_DISTRIBUTIONS)} audited distributions"
        )
    result: dict[str, str] = {}
    for name, expected in EXPECTED_REFERENCE_DISTRIBUTIONS.items():
        actual = value.get(name)
        if actual != expected:
            raise SystemExit(f"{label}.{name} version drift: {actual!r} != {expected!r}")
        result[name] = actual
    return result


def _metadata_license_declarations_present(metadata: dict[str, Any]) -> bool:
    fields = ("license", "license_expression", "license_classifiers")
    return all(isinstance(metadata.get(field), list) for field in fields) and any(
        isinstance(value, str)
        and value.strip()
        and value.strip().casefold() != "unknown"
        for field in fields
        for value in metadata[field]
    )


def validate_dependency_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate the collector's exact success contract without granting it permission."""
    _exact_keys(
        evidence,
        (
            "schema",
            "status",
            "publication_permitted",
            "fixture_generation_permitted",
            "owner_review_required",
            "platform",
            "project",
            "uv_lock",
            "lock",
            "installed_inventory",
            "installed_distributions",
            "failures",
        ),
        "dependency evidence",
    )
    if evidence["schema"] != DEPENDENCY_EVIDENCE_SCHEMA:
        raise SystemExit("dependency evidence schema drift")
    if evidence["status"] != DEPENDENCY_EVIDENCE_STATUS:
        raise SystemExit("dependency evidence collection did not succeed")
    if evidence["publication_permitted"] is not False or evidence["fixture_generation_permitted"] is not False:
        raise SystemExit("dependency evidence cannot grant publication or fixture permission")
    if evidence["owner_review_required"] is not True:
        raise SystemExit("dependency evidence must retain owner review requirement")
    if evidence["failures"] != []:
        raise SystemExit("dependency evidence contains collection failures")

    platform_data = evidence["platform"]
    if not isinstance(platform_data, dict):
        raise SystemExit("dependency evidence platform must be an object")
    _exact_keys(platform_data, ("system", "machine", "python"), "dependency evidence platform")
    if platform_data != {"system": "Linux", "machine": "x86_64", "python": "3.12"}:
        raise SystemExit(f"dependency evidence platform drift: {platform_data!r}")

    for key, expected in (("project", "pyproject.toml"), ("uv_lock", "uv.lock")):
        item = evidence[key]
        if not isinstance(item, dict):
            raise SystemExit(f"dependency evidence {key} must be an object")
        _exact_keys(item, ("path", "bytes", "sha256"), f"dependency evidence {key}")
        if item["path"] != expected or not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise SystemExit(f"dependency evidence {key} identity drift")
        expected_sha = EXPECTED_REFERENCE_PROJECT_SHA256 if key == "project" else EXPECTED_REFERENCE_LOCK_SHA256
        if _require_sha(item["sha256"], f"dependency evidence {key}.sha256") != expected_sha:
            raise SystemExit(f"dependency evidence {key} digest drift")

    lock_data = evidence["lock"]
    if not isinstance(lock_data, dict):
        raise SystemExit("dependency evidence lock must be an object")
    _exact_keys(
        lock_data,
        ("selected_platform", "resolution_markers", "selected_closure", "rows", "rows_sha256"),
        "dependency evidence lock",
    )
    if lock_data["selected_platform"] != "Linux x86_64 CPython 3.12":
        raise SystemExit("dependency evidence selected platform drift")
    if not isinstance(lock_data["resolution_markers"], list) or not isinstance(lock_data["rows"], list):
        raise SystemExit("dependency evidence lock rows/markers malformed")
    selected = lock_data["selected_closure"]
    if selected != sorted(["vokra-microwakeword-reference", *EXPECTED_REFERENCE_DISTRIBUTIONS]):
        raise SystemExit("dependency evidence selected closure drift")
    lock_rows_sha256 = _require_sha(lock_data["rows_sha256"], "dependency evidence lock.rows_sha256")
    if lock_rows_sha256 != _canonical_json_sha256(lock_data["rows"]):
        raise SystemExit("dependency evidence lock rows digest mismatch")

    inventory = evidence["installed_inventory"]
    if not isinstance(inventory, dict):
        raise SystemExit("dependency evidence installed_inventory must be an object")
    _exact_keys(inventory, ("status", "sha256", "entries", "failures"), "dependency evidence installed_inventory")
    if inventory["status"] != "PASS" or inventory["failures"] != []:
        raise SystemExit("dependency evidence installed inventory is not a successful collection")
    inventory_sha256 = _require_sha(inventory["sha256"], "dependency evidence installed_inventory.sha256")
    inventory_entries = inventory["entries"]
    if not isinstance(inventory_entries, list) or len(inventory_entries) != len(EXPECTED_REFERENCE_DISTRIBUTIONS):
        raise SystemExit(
            "dependency evidence installed inventory must contain exactly "
            f"{len(EXPECTED_REFERENCE_DISTRIBUTIONS)} entries"
        )
    inventory_names: set[str] = set()
    for entry in inventory_entries:
        if not isinstance(entry, dict):
            raise SystemExit("dependency evidence inventory entry must be an object")
        _exact_keys(
            entry,
            ("path", "metadata", "normalized_name", "name", "version", "status", "failures"),
            "dependency evidence inventory entry",
        )
        name = entry["normalized_name"]
        if not isinstance(name, str) or name not in EXPECTED_REFERENCE_DISTRIBUTIONS or name in inventory_names:
            raise SystemExit(f"dependency evidence inventory name drift: {name!r}")
        if entry["status"] != "VALID" or entry["failures"] != []:
            raise SystemExit(f"dependency evidence inventory entry is not valid: {name}")
        if entry["version"] != EXPECTED_REFERENCE_DISTRIBUTIONS[name]:
            raise SystemExit(f"dependency evidence inventory version drift: {name}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise SystemExit(f"dependency evidence inventory path malformed: {name}")
        metadata = entry["metadata"]
        if not isinstance(metadata, dict) or metadata.get("name") != [entry["name"]] or metadata.get("version") != [entry["version"]]:
            raise SystemExit(f"dependency evidence inventory metadata drift: {name}")
        inventory_names.add(name)
    if inventory_names != set(EXPECTED_REFERENCE_DISTRIBUTIONS):
        raise SystemExit("dependency evidence installed inventory set drift")
    if inventory_sha256 != _canonical_json_sha256(inventory_entries):
        raise SystemExit("dependency evidence installed inventory digest mismatch")

    installed = evidence["installed_distributions"]
    if not isinstance(installed, list) or len(installed) != len(EXPECTED_REFERENCE_DISTRIBUTIONS):
        raise SystemExit(
            "dependency evidence must contain exactly "
            f"{len(EXPECTED_REFERENCE_DISTRIBUTIONS)} installed distributions"
        )
    seen: set[str] = set()
    versions: dict[str, str] = {}
    required = (
        "expected_name",
        "expected_version",
        "status",
        "metadata",
        "record",
        "license_candidates",
        "native_payloads",
        "failures",
        "dist_info",
        "dist_info_path",
        "inventory_sha256",
    )
    for row in installed:
        if not isinstance(row, dict):
            raise SystemExit("dependency evidence installed row must be an object")
        _exact_keys(row, required, "dependency evidence installed row")
        name = row["expected_name"]
        version = row["expected_version"]
        if not isinstance(name, str) or name not in EXPECTED_REFERENCE_DISTRIBUTIONS or name in seen:
            raise SystemExit(f"dependency evidence installed name drift: {name!r}")
        if version != EXPECTED_REFERENCE_DISTRIBUTIONS[name]:
            raise SystemExit(f"dependency evidence installed version drift: {name}={version!r}")
        if row["status"] != DEPENDENCY_EVIDENCE_STATUS or row["failures"] != []:
            raise SystemExit(f"dependency evidence installed collection failed: {name}")
        if row["inventory_sha256"] != inventory_sha256:
            raise SystemExit(f"dependency evidence inventory digest mismatch: {name}")
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise SystemExit(f"dependency evidence installed metadata/record malformed: {name}")
        _exact_keys(
            metadata,
            (
                "path",
                "bytes",
                "sha256",
                "name",
                "version",
                "license",
                "license_expression",
                "license_file",
                "classifiers",
                "license_classifiers",
            ),
            f"dependency evidence metadata.{name}",
        )
        metadata_names = metadata.get("name")
        if (
            not isinstance(metadata_names, list)
            or len(metadata_names) != 1
            or not isinstance(metadata_names[0], str)
            or _normalize_distribution_name(metadata_names[0]) != name
            or metadata.get("version") != [version]
        ):
            raise SystemExit(f"dependency evidence installed metadata name/version drift: {name}")
        if not isinstance(metadata["bytes"], int) or metadata["bytes"] <= 0:
            raise SystemExit(f"dependency evidence metadata byte count malformed: {name}")
        _require_sha(metadata["sha256"], f"dependency evidence metadata.{name}.sha256")
        declaration_fields = ("license", "license_expression", "license_classifiers")
        for field in (*declaration_fields, "license_file", "classifiers"):
            if not isinstance(metadata[field], list) or not all(isinstance(item, str) for item in metadata[field]):
                raise SystemExit(f"dependency evidence metadata declaration malformed: {name}.{field}")
        record = row["record"]
        if not isinstance(record, dict):
            raise SystemExit(f"dependency evidence installed record malformed: {name}")
        _exact_keys(record, ("path", "bytes", "sha256", "entries", "entries_count", "entries_sha256"), f"dependency evidence record.{name}")
        if not isinstance(record["path"], str) or not record["path"] or not isinstance(record["bytes"], int) or record["bytes"] <= 0:
            raise SystemExit(f"dependency evidence RECORD identity malformed: {name}")
        entries = record["entries"]
        if not isinstance(entries, list) or not entries or record["entries_count"] != len(entries):
            raise SystemExit(f"dependency evidence installed RECORD is empty/malformed: {name}")
        _require_sha(record["sha256"], f"dependency evidence record.{name}.sha256")
        entries_sha256 = _require_sha(record["entries_sha256"], f"dependency evidence record.{name}.entries_sha256")
        if entries_sha256 != _canonical_json_sha256(entries):
            raise SystemExit(f"dependency evidence RECORD entries digest mismatch: {name}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise SystemExit(f"dependency evidence RECORD entry malformed: {name}")
            keys = set(entry)
            required_entry_keys = {"row", "declared", "actual", "validation", "errors"}
            if keys not in (required_entry_keys, required_entry_keys | {"resolved_path"}):
                raise SystemExit(f"dependency evidence RECORD entry keys drift: {name}")
            if not isinstance(entry["row"], int) or entry["row"] <= 0:
                raise SystemExit(f"dependency evidence RECORD row number malformed: {name}")
            declared = entry["declared"]
            if not isinstance(declared, dict) or set(declared) != {"path", "hash", "size"}:
                raise SystemExit(f"dependency evidence RECORD declaration malformed: {name}")
            if not isinstance(declared["path"], str) or not declared["path"]:
                raise SystemExit(f"dependency evidence RECORD path malformed: {name}")
            declared_hash = declared["hash"]
            if declared_hash is not None:
                if not isinstance(declared_hash, dict) or set(declared_hash) != {"algorithm", "value", "status"}:
                    raise SystemExit(f"dependency evidence RECORD hash declaration malformed: {name}")
                if not isinstance(declared_hash["status"], str) or not isinstance(declared_hash["algorithm"], (str, type(None))) or not isinstance(declared_hash["value"], (str, type(None))):
                    raise SystemExit(f"dependency evidence RECORD hash declaration types malformed: {name}")
            declared_size = declared["size"]
            if declared_size is not None:
                if not isinstance(declared_size, dict) or set(declared_size) != {"value", "status"}:
                    raise SystemExit(f"dependency evidence RECORD size declaration malformed: {name}")
                if not isinstance(declared_size["status"], str) or not isinstance(declared_size["value"], (int, str, type(None))):
                    raise SystemExit(f"dependency evidence RECORD size declaration types malformed: {name}")
            actual = entry["actual"]
            if actual is not None:
                if not isinstance(actual, dict) or set(actual) != {"sha256", "bytes"}:
                    raise SystemExit(f"dependency evidence RECORD actual identity malformed: {name}")
                _require_sha(actual["sha256"], f"dependency evidence RECORD actual.{name}.sha256")
                if not isinstance(actual["bytes"], int) or actual["bytes"] < 0:
                    raise SystemExit(f"dependency evidence RECORD actual byte count malformed: {name}")
            if entry["validation"] not in {"MATCH", "EMPTY_DECLARATION", "FAIL", "MALFORMED_ROW", "OVERSIZE"}:
                raise SystemExit(f"dependency evidence RECORD validation status malformed: {name}")
            if not isinstance(entry["errors"], list) or not all(isinstance(error, str) for error in entry["errors"]):
                raise SystemExit(f"dependency evidence RECORD errors malformed: {name}")
            if "resolved_path" in entry and (not isinstance(entry["resolved_path"], str) or not entry["resolved_path"]):
                raise SystemExit(f"dependency evidence RECORD resolved path malformed: {name}")
        candidates = row["license_candidates"]
        metadata_declared = _metadata_license_declarations_present(metadata)
        if not isinstance(candidates, list) or (not candidates and not metadata_declared):
            raise SystemExit(f"dependency evidence installed payload evidence malformed: {name}")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise SystemExit(f"dependency evidence license candidate malformed: {name}")
            _exact_keys(candidate, ("path", "bytes", "sha256"), f"dependency evidence license.{name}")
            if not isinstance(candidate["path"], str) or not candidate["path"] or not isinstance(candidate["bytes"], int) or candidate["bytes"] <= 0:
                raise SystemExit(f"dependency evidence license candidate identity malformed: {name}")
            _require_sha(candidate["sha256"], f"dependency evidence license.{name}.sha256")
        native_payloads = row["native_payloads"]
        if not isinstance(native_payloads, list):
            raise SystemExit(f"dependency evidence native payload evidence malformed: {name}")
        for native in native_payloads:
            if not isinstance(native, dict):
                raise SystemExit(f"dependency evidence native payload malformed: {name}")
            _exact_keys(native, ("path", "bytes", "sha256", "readelf"), f"dependency evidence native.{name}")
            if not isinstance(native["path"], str) or not native["path"] or not isinstance(native["bytes"], int) or native["bytes"] <= 0:
                raise SystemExit(f"dependency evidence native payload identity malformed: {name}")
            _require_sha(native["sha256"], f"dependency evidence native.{name}.sha256")
            if not isinstance(native["readelf"], dict) or not isinstance(native["readelf"].get("status"), str):
                raise SystemExit(f"dependency evidence native readelf malformed: {name}")
        if not isinstance(row["dist_info"], str) or not isinstance(row["dist_info_path"], str):
            raise SystemExit(f"dependency evidence installed dist-info malformed: {name}")
        seen.add(name)
        versions[name] = version
    if seen != set(EXPECTED_REFERENCE_DISTRIBUTIONS):
        raise SystemExit("dependency evidence installed distribution set drift")
    return {"platform": platform_data, "versions": versions}


def require_reference_runtime(evidence_versions: dict[str, str]) -> dict[str, Any]:
    """Verify the actual VAST interpreter only after all evidence gates pass."""
    import importlib.metadata
    import platform

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"reference runtime must be Python 3.12, got {sys.version_info[:3]}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("reference runtime must be Linux x86_64")
    expected_names = set(EXPECTED_REFERENCE_DISTRIBUTIONS)
    observed_names: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise SystemExit("installed reference distribution has no canonical Name metadata")
        observed_names.append(re.sub(r"[-_.]+", "-", name).casefold())
    if len(observed_names) != len(expected_names) or set(observed_names) != expected_names:
        raise SystemExit(
            "installed reference distribution set drift: "
            + ",".join(sorted(observed_names))
        )
    installed: dict[str, str] = {}
    for name, expected in EXPECTED_REFERENCE_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise SystemExit(f"required reference distribution is not installed: {name}") from error
        if actual != expected or actual != evidence_versions[name]:
            raise SystemExit(f"installed reference distribution drift: {name}={actual!r}")
        installed[name] = actual
    return {"system": "Linux", "machine": "x86_64", "python": "3.12", "installed_distributions": installed}


def _synthetic_dependency_evidence() -> dict[str, Any]:
    """Build a model-free success-shaped collector report for self-tests."""
    inventory_entries = [
        {
            "path": f"lib/python3.12/site-packages/{name}-{version}.dist-info",
            "metadata": {
                "path": f"lib/python3.12/site-packages/{name}-{version}.dist-info/METADATA",
                "bytes": 1,
                "sha256": "0" * 64,
                "name": [name],
                "version": [version],
                "license": ["BSD-3-Clause"],
                "license_expression": [],
                "license_file": ["LICENSE"],
                "classifiers": ["License :: OSI Approved :: BSD License"],
                "license_classifiers": ["License :: OSI Approved :: BSD License"],
            },
            "normalized_name": name,
            "name": name,
            "version": version,
            "status": "VALID",
            "failures": [],
        }
        for name, version in EXPECTED_REFERENCE_DISTRIBUTIONS.items()
    ]
    inventory_sha256 = _canonical_json_sha256(inventory_entries)
    lock_rows: list[dict[str, Any]] = []
    record_entries = [
        {
            "row": 1,
            "declared": {
                "path": "METADATA",
                "hash": {"algorithm": "sha256", "value": "0" * 64, "status": "VALID"},
                "size": {"value": 1, "status": "VALID"},
            },
            "actual": {"sha256": "0" * 64, "bytes": 1},
            "resolved_path": "lib/python3.12/site-packages/METADATA",
            "validation": "MATCH",
            "errors": [],
        }
    ]
    record = {
        "path": "RECORD",
        "bytes": 1,
        "sha256": "0" * 64,
        "entries": record_entries,
        "entries_count": len(record_entries),
        "entries_sha256": _canonical_json_sha256(record_entries),
    }
    rows = [
        {
            "expected_name": name,
            "expected_version": version,
            "status": DEPENDENCY_EVIDENCE_STATUS,
            "metadata": dict(inventory_entries[0]["metadata"] | {"name": [name], "version": [version]}),
            "record": record,
            "license_candidates": [{"path": "LICENSE", "bytes": 1, "sha256": "0" * 64}],
            "native_payloads": [],
            "failures": [],
            "dist_info": f"{name}-{version}.dist-info",
            "dist_info_path": f"lib/python3.12/site-packages/{name}-{version}.dist-info",
            "inventory_sha256": inventory_sha256,
        }
        for name, version in EXPECTED_REFERENCE_DISTRIBUTIONS.items()
    ]
    return {
        "schema": DEPENDENCY_EVIDENCE_SCHEMA,
        "status": DEPENDENCY_EVIDENCE_STATUS,
        "publication_permitted": False,
        "fixture_generation_permitted": False,
        "owner_review_required": True,
        "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12"},
        "project": {"path": "pyproject.toml", "bytes": 1, "sha256": EXPECTED_REFERENCE_PROJECT_SHA256},
        "uv_lock": {"path": "uv.lock", "bytes": 1, "sha256": EXPECTED_REFERENCE_LOCK_SHA256},
        "lock": {
            "selected_platform": "Linux x86_64 CPython 3.12",
            "resolution_markers": [],
            "selected_closure": sorted(["vokra-microwakeword-reference", *EXPECTED_REFERENCE_DISTRIBUTIONS]),
            "rows": [],
            "rows_sha256": _canonical_json_sha256(lock_rows),
        },
        "installed_inventory": {
            "status": "PASS",
            "sha256": inventory_sha256,
            "entries": inventory_entries,
            "failures": [],
        },
        "installed_distributions": rows,
        "failures": [],
    }


def require_reference_dependency_gate(dependency_evidence_bytes: bytes) -> dict[str, Any]:
    """Refuse fixture generation until the isolated dependency audit is PASS."""
    inspector_path = Path(__file__).parent.parent / "microwakeword-reference" / "inspect.py"
    project_path = inspector_path.parent / "pyproject.toml"
    lock_path = inspector_path.parent / "uv.lock"
    spec = importlib.util.spec_from_file_location("microwakeword_reference_inspector", inspector_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"reference dependency inspector unavailable: {inspector_path}")
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    report = inspector.inspect_documents(
        project_path.read_bytes(),
        lock_path.read_bytes(),
        dependency_evidence=dependency_evidence_bytes,
    )
    if report.get("status") != "PASS" or not report.get("fixture_generation_permitted", False):
        raise SystemExit(
            "reference fixture generation is blocked by dependency/license gate: "
            + ", ".join(report.get("failures", ["unknown gate failure"]))
        )
    return report


def _effective_dependency_manifest(
    raw_evidence: dict[str, Any],
    audit_report: dict[str, Any],
    dependency_contract: dict[str, Any],
    evidence_path: Path,
    evidence_sha256: str,
) -> dict[str, Any]:
    """Separate collector facts from the effective reviewed fixture decision."""
    return {
        "schema": DEPENDENCY_EVIDENCE_SCHEMA,
        "collection_status": raw_evidence["status"],
        "audit_status": audit_report["status"],
        "review_status": audit_report["dependency_evidence_status"],
        "publication_permitted": audit_report["publication_permitted"],
        "fixture_generation_permitted": audit_report["fixture_generation_permitted"],
        "collector_owner_review_required": raw_evidence["owner_review_required"],
        "failures": audit_report["failures"],
        "path": evidence_path.name,
        "sha256": evidence_sha256,
        "project_sha256": EXPECTED_REFERENCE_PROJECT_SHA256,
        "uv_lock_sha256": EXPECTED_REFERENCE_LOCK_SHA256,
        "platform": dependency_contract["platform"],
        "installed_distributions": dependency_contract["versions"],
    }


def synth_pcm(invocation: int = 0, frame: int = 0) -> np.ndarray:
    """Deterministic ``i16`` PCM frame, ``WINDOW_SAMPLES`` samples wide.

    The global frame index changes tone and seed so each streaming frame is
    distinct while remaining reproducible.
    """
    import numpy as np

    if invocation < 0 or frame < 0:
        raise ValueError("invocation/frame must be non-negative")
    global_frame = invocation * FRAMES_PER_INVOCATION + frame
    rng = np.random.default_rng(PCM_SEED + global_frame)
    t = np.arange(WINDOW_SAMPLES, dtype=np.float64) / float(SAMPLE_RATE)
    sine_hz = PCM_SINE_HZ + 37.0 * global_frame
    sine = PCM_SINE_AMPLITUDE * np.sin(2.0 * np.pi * sine_hz * t)
    noise = rng.normal(0.0, PCM_NOISE_STDDEV, size=WINDOW_SAMPLES)
    signal = sine + noise
    # Clip to int16 range (defensive — the amplitude budget above avoids
    # clipping in practice, but a stray parameter change should not
    # produce silent wrap-around).
    signal = np.clip(signal, -32768.0, 32767.0)
    return signal.astype(np.int16)


def quantize_features_int8(features: np.ndarray) -> np.ndarray:
    """Quantise one ``[3, 40]`` feature stack to authenticated ``[1,3,40]``."""
    import numpy as np

    if features.shape != INPUT_SHAPE[1:]:
        raise ValueError(f"features shape {features.shape} != {INPUT_SHAPE[1:]}")
    if features.dtype != np.float32 or not np.all(np.isfinite(features)):
        raise ValueError("features must be finite float32")
    scaled = features.astype(np.float64) / INPUT_SCALE
    rounded = np.where(scaled >= 0.0, scaled + 0.5, scaled - 0.5).astype(np.int64)
    values = rounded + INPUT_ZERO_POINT
    if np.any(values < -128) or np.any(values > 127):
        raise ValueError("quantised input does not fit int8")
    return values.astype(np.int8).reshape(INPUT_SHAPE)


def validate_tflite_contract(input_detail: dict[str, Any], output_detail: dict[str, Any]) -> None:
    """Fail-closed validation of the authenticated TFLite IO contract."""
    if tuple(int(x) for x in input_detail.get("shape", ())) != INPUT_SHAPE:
        raise ValueError(f"input shape is not authenticated: {input_detail.get('shape')}")
    input_dtype_name = getattr(input_detail.get("dtype"), "__name__", str(input_detail.get("dtype")))
    if input_dtype_name != INPUT_DTYPE_NAME:
        raise ValueError(f"input dtype is not int8: {input_detail.get('dtype')!r}")
    in_scale, in_zp = input_detail.get("quantization", (0.0, 0))
    if float(in_scale) != INPUT_SCALE or int(in_zp) != INPUT_ZERO_POINT:
        raise ValueError(f"input quantization drift: {(in_scale, in_zp)!r}")
    if tuple(int(x) for x in output_detail.get("shape", ())) != OUTPUT_SHAPE:
        raise ValueError(f"output shape is not authenticated: {output_detail.get('shape')}")
    output_dtype_name = getattr(output_detail.get("dtype"), "__name__", str(output_detail.get("dtype")))
    if output_dtype_name != OUTPUT_DTYPE_NAME:
        raise ValueError(f"output dtype is not uint8: {output_detail.get('dtype')!r}")
    out_scale, out_zp = output_detail.get("quantization", (0.0, 0))
    if float(out_scale) != OUTPUT_SCALE or int(out_zp) != OUTPUT_ZERO_POINT:
        raise ValueError(f"output quantization drift: {(out_scale, out_zp)!r}")


def hz_to_mel_f32(hz: np.float32) -> np.float32:
    """HTK-convention Hz → mel, computed in **float32** to match the Rust
    ``crates/vokra-kws-micro/src/features.rs::hz_to_mel`` bit-for-bit:
    ``mel = 2595 * log10(1 + hz / 700)``.

    The `np.float32(...)` casts pin every intermediate at f32 precision.
    A stray float64 promotion (e.g. `1.0 + hz / 700.0` where `hz` was
    promoted to f64 by scalar arithmetic) would silently shift the mel
    filterbank edges by ~1e-5 Hz and break bit-parity against Vokra
    (verified empirically — a float64 mel_points path fails Path B at
    band ~30 by 3e-2, within the registered 5e-2 boundary but still
    useful as a drift signal).
    """
    return np.float32(2595.0) * np.log10(np.float32(1.0) + hz / np.float32(700.0))


def mel_to_hz_f32(mel: np.float32) -> np.float32:
    """Inverse of :func:`hz_to_mel_f32`. Kept in **float32** for the same
    bit-parity reason. Same as the Rust ``mel_to_hz``:
    ``hz = 700 * (10^(mel / 2595) - 1)``.
    """
    return np.float32(700.0) * (np.float32(10.0) ** (mel / np.float32(2595.0)) - np.float32(1.0))


def hann_window(n: int) -> np.ndarray:
    """Symmetric Hann window matching the Rust ``hann_window``:
    ``w[i] = 0.5 * (1 - cos(2*pi*i / (n-1)))``.

    Equivalent to ``numpy.hanning(n)``. We spell it out explicitly (and
    keep every intermediate at f32) to make the correspondence to the
    Rust code unmistakable and to avoid a stray future refactor toward
    the periodic convention (``2*pi*i / n``) which would silently
    rescale every feature.
    """
    denom = np.float32(n - 1)
    i = np.arange(n, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)
    return (np.float32(0.5) * (np.float32(1.0) - np.cos(two_pi * i / denom))).astype(np.float32)


def mel_filterbank(n_mels: int, n_bins: int, sample_rate: int) -> np.ndarray:
    """Row-major ``[n_mels, n_bins]`` un-normalised triangular filterbank
    with HTK-convention mel spacing, ``fmin = 0``, ``fmax = sr / 2``.

    **f32 throughout** to match the Rust ``mel_filterbank`` bit-for-bit
    (the Rust code declares every intermediate as `f32`; using float64
    for the mel-point path in numpy shifts band edges by ~1e-5 Hz and
    accumulates a ~3e-2 log10 delta at high bands).
    """
    fmax = np.float32(0.5) * np.float32(sample_rate)
    mel_min = hz_to_mel_f32(np.float32(0.0))
    mel_max = hz_to_mel_f32(fmax)
    # (n_mels + 2) equally-spaced mel points. `np.linspace` with an
    # f32 dtype computes in f32 throughout.
    #
    # Vokra spells this as:
    #   mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1)
    # in f32. `np.linspace(mel_min, mel_max, n_mels+2, dtype=f32)` uses
    # a slightly different formula internally (`start + step * i` with
    # a precomputed `step`), which can differ at f32-ULP scale from the
    # Rust `(mel_max - mel_min) * i / n_mels_plus_1` form. Mirror the
    # Rust formula exactly to preserve bit-parity.
    denom = np.float32(n_mels + 1)
    span = mel_max - mel_min
    mel_points = np.array(
        [mel_min + span * np.float32(i) / denom for i in range(n_mels + 2)],
        dtype=np.float32,
    )
    bin_scale = np.float32(n_bins - 1) / fmax
    bin_pts = np.array(
        [mel_to_hz_f32(np.float32(mp)) * bin_scale for mp in mel_points],
        dtype=np.float32,
    )
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left = np.float32(bin_pts[m])
        center = np.float32(bin_pts[m + 1])
        right = np.float32(bin_pts[m + 2])
        for k in range(n_bins):
            kf = np.float32(k)
            if kf < left or kf > right:
                w = np.float32(0.0)
            elif kf <= center:
                if center == left:
                    w = np.float32(1.0)
                else:
                    w = (kf - left) / (center - left)
            elif center == right:
                w = np.float32(1.0)
            else:
                w = (right - kf) / (right - center)
            fb[m, k] = w
    return fb


def numpy_log_mel_features(pcm_i16: np.ndarray) -> np.ndarray:
    """Numpy reference log-mel feature extraction.

    Steps 1–5 of ``FeatureExtractor::compute_frame_f32``:

    1. i16 → f32 with symmetric Hann window applied (NO normalisation
        to [-1, 1] — matches Rust code exactly).
    2. Radix-2 FFT via ``np.fft.rfft`` (returns the one-sided spectrum;
        length ``N_FFT/2 + 1 = N_BINS``).
    3. Power spectrum ``|X[k]|²``.
    4. Row-major mel filterbank matmul (explicit Python loop, NOT
        ``fb @ power``, to match Rust's naive left-to-right accumulator
        order — numpy BLAS's pairwise / SIMD summation gives different
        f32 rounding at high bands, verified empirically).
    5. ``log10(max(mel_energy, LOG_MEL_EPSILON))``.

    # Precision honesty

    ``np.fft.rfft`` computes internally in float64 and casts to the
    input dtype at output. Vokra's Rust FFT is float32 throughout. The
    two agree bit-for-bit at low bands (< 1e-4 per-band |Δ|) but drift
    to ~3e-2 at high bands (~30) where the f32 rounding accumulates
    through log₂(N_FFT) = 9 butterfly stages. This is a real precision
    gap between the (higher-precision) numpy reference and the target-
    architecture-realistic (f32) Rust code — not a Rust bug. The
    parity harness's ``FEATURES_ATOL`` accepts this bound and catches
    regressions above it.

    A pure-f32 numpy transcription of Vokra's Cooley–Tukey radix-2 FFT
    would close this gap but would basically be running the Rust code
    in Python — the honest atol is the more useful posture.
    """
    assert pcm_i16.shape == (WINDOW_SAMPLES,), pcm_i16.shape
    assert pcm_i16.dtype == np.int16, pcm_i16.dtype

    # Step 1: i16 → f32 with Hann (no [-1, 1] normalisation).
    hann = hann_window(WINDOW_SAMPLES)
    windowed = pcm_i16.astype(np.float32) * hann

    # Zero-pad to N_FFT if the window is shorter (at default constants
    # WINDOW_SAMPLES == N_FFT, so this is a no-op; kept for parity with
    # the Rust code's implicit zero-padding).
    if WINDOW_SAMPLES < N_FFT:
        padded = np.zeros(N_FFT, dtype=np.float32)
        padded[:WINDOW_SAMPLES] = windowed
        windowed = padded

    # Step 2: real one-sided FFT.
    spec = np.fft.rfft(windowed, n=N_FFT)
    assert spec.shape == (N_BINS,), spec.shape

    # Step 3: power spectrum (|X[k]|²).
    power = (spec.real.astype(np.float32) ** 2) + (spec.imag.astype(np.float32) ** 2)

    # Step 4: filterbank matmul via explicit accumulator (matches Rust
    # naive left-to-right f32 summation order — see docstring above).
    fb = mel_filterbank(N_MELS, N_BINS, SAMPLE_RATE)
    mel_energy = np.zeros(N_MELS, dtype=np.float32)
    for m in range(N_MELS):
        acc = np.float32(0.0)
        row = fb[m]
        for k in range(N_BINS):
            acc = acc + row[k] * power[k]
        mel_energy[m] = acc

    # Step 5: log10(max(mel_energy, EPSILON)).
    clamped = np.maximum(mel_energy, np.float32(LOG_MEL_EPSILON))
    features = np.log10(clamped).astype(np.float32)
    assert features.shape == (N_MELS,), features.shape
    return features


def validate_final_output_path(path: Path) -> None:
    """Require a new final path; never delete or follow an existing path."""
    if path.is_symlink():
        raise SystemExit(f"refusing symlink output directory: {path}")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise SystemExit(f"output directory parent must be an existing real directory: {parent}")
    if path.exists():
        raise SystemExit(f"refusing existing output path (no-clobber): {path}")


def create_staging_output(final_path: Path) -> tuple[Path, Any]:
    """Create an owned sibling staging directory with failure cleanup."""
    validate_final_output_path(final_path)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_path.name}.staging-", dir=final_path.parent))

    def cleanup() -> None:
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging)

    atexit.register(cleanup)
    return staging, cleanup


def publish_staging(staging: Path, final_path: Path, cleanup: Any) -> None:
    """Publish staged files with an exclusive final-directory claim.

    The final directory is claimed with atomic ``mkdir``; each staged file is
    then hard-linked into it (which refuses an existing destination) and the
    manifest is linked last. A final directory without its manifest is never
    considered a fixture by the Rust gate.
    """
    lock_path = final_path.parent / f".{final_path.name}.publish.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise SystemExit(f"refusing concurrent/existing publish lock: {lock_path}") from error
    final_stat: os.stat_result | None = None
    lock_stat = os.fstat(fd)
    owner_marker = final_path / ".vokra-publish-owner"
    owner_token = os.urandom(16).hex().encode("ascii")

    def cleanup_owned_final() -> None:
        if final_stat is None:
            return
        try:
            current = final_path.lstat()
            marker = owner_marker.read_bytes()
        except (FileNotFoundError, OSError):
            return
        if (current.st_dev, current.st_ino) == (final_stat.st_dev, final_stat.st_ino) and marker == owner_token:
            shutil.rmtree(final_path)

    try:
        validate_final_output_path(final_path)
        children = list(staging.iterdir())
        if any(child.is_symlink() or not child.is_file() for child in children):
            raise SystemExit(f"staging contains unsafe non-regular artefact: {staging}")
        manifest = staging / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            raise SystemExit("staging manifest is missing or unsafe")
        try:
            os.mkdir(final_path, 0o700)
        except FileExistsError as error:
            raise SystemExit(f"refusing concurrent/existing final output: {final_path}") from error
        final_stat = final_path.lstat()
        with owner_marker.open("xb") as marker:
            marker.write(owner_token)
        for child in sorted(children, key=lambda item: item.name == "manifest.json"):
            destination = final_path / child.name
            try:
                os.link(child, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise SystemExit(f"refusing artefact destination collision: {destination}") from error
            child.unlink()
        staging.rmdir()
        marker_stat = owner_marker.lstat()
        current_marker = owner_marker.lstat()
        if (current_marker.st_dev, current_marker.st_ino) != (marker_stat.st_dev, marker_stat.st_ino):
            raise SystemExit("publish owner marker changed unexpectedly")
        owner_marker.unlink()
        atexit.unregister(cleanup)
    except BaseException:
        cleanup_owned_final()
        raise
    finally:
        os.close(fd)
        try:
            current_lock = lock_path.lstat()
        except FileNotFoundError:
            current_lock = None
        if current_lock is not None and (current_lock.st_dev, current_lock.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
            lock_path.unlink()


def require_regular_tflite(path: Path) -> None:
    if path.is_symlink():
        raise SystemExit(f"refusing symlink TFLite path: {path}")
    if not path.is_file():
        raise SystemExit(f"--tflite-path not found: {path}")


def require_real_cli_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("tflite_path", "output_dir", "dependency_evidence")
        if getattr(args, name, None) is None
    ]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))


def dump_le(arr: np.ndarray, path: Path) -> None:
    """Writes ``arr`` as little-endian raw bytes (no header).

    ``ndarray.tobytes()`` uses the native byte order; forcing to ``<f4``
    / ``<i2`` / ``<f8`` first pins the wire format across host
    endianness. Every consumer this file targets (the Rust parity
    harness) reads little-endian, matching the M5-03 IoT target family
    (thumbv8m is little-endian).
    """
    if path.is_symlink() or path.exists():
        raise SystemExit(f"refusing to overwrite existing artefact: {path}")
    # Map dtype → little-endian equivalent. Guard against surprise dtypes.
    if arr.dtype == np.int16:
        arr = arr.astype("<i2")
    elif arr.dtype == np.float32:
        arr = arr.astype("<f4")
    elif arr.dtype == np.int8:
        arr = arr.astype("<i1")  # trivially LE
    elif arr.dtype == np.uint8:
        arr = arr.astype("<u1")  # trivially LE
    else:
        raise SystemExit(f"dump_le: unsupported dtype {arr.dtype} for {path}")
    with path.open("xb") as output:
        output.write(np.ascontiguousarray(arr).tobytes())


def run_tflite_sequence(
    interp: Any, feature_sequence: np.ndarray, verbose: bool
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Run the independent upstream interpreter over a persistent sequence."""
    import numpy as np

    if feature_sequence.shape != (FRAMES_PER_INVOCATION * INVOCATION_COUNT, N_MELS):
        raise ValueError(f"unexpected feature sequence shape: {feature_sequence.shape}")
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("authenticated model must expose exactly one input and output")
    inp, out = input_details[0], output_details[0]
    validate_tflite_contract(inp, out)
    if verbose:
        print(f"  TFLite input : name={inp['name']!r} shape={list(inp['shape'])} dtype={inp['dtype']}", file=sys.stderr)
        print(f"  TFLite output: name={out['name']!r} shape={list(out['shape'])} dtype={out['dtype']}", file=sys.stderr)

    raw_outputs: list[np.ndarray] = []
    for invocation in range(INVOCATION_COUNT):
        frames = feature_sequence[
            invocation * FRAMES_PER_INVOCATION : (invocation + 1) * FRAMES_PER_INVOCATION
        ]
        input_tensor = quantize_features_int8(frames)
        interp.set_tensor(inp["index"], input_tensor)
        interp.invoke()
        raw = np.ascontiguousarray(interp.get_tensor(out["index"]))
        if raw.shape != OUTPUT_SHAPE or raw.dtype != np.uint8:
            raise ValueError(f"invocation {invocation}: output drift: {raw.shape} {raw.dtype}")
        raw_outputs.append(raw.copy())
        if verbose:
            print(f"  invocation {invocation:02d}: input bytes={input_tensor.nbytes} output={int(raw.reshape(-1)[0])}", file=sys.stderr)
    return raw_outputs, {
        "input_name": inp["name"],
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "int8",
        "input_scale": INPUT_SCALE,
        "input_zero_point": INPUT_ZERO_POINT,
        "output_name": out["name"],
        "output_shape": list(OUTPUT_SHAPE),
        "output_dtype": "uint8",
        "output_scale": OUTPUT_SCALE,
        "output_zero_point": OUTPUT_ZERO_POINT,
    }


def artifact(path: Path, name: str, shape: tuple[int, ...], dtype: str, role: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "name": name,
        "path": path.name,
        "shape": list(shape),
        "dtype": dtype,
        "byte_order": "little-endian",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "role": role,
    }


def self_test() -> int:
    """No-model contract checks; intentionally does not import LiteRT."""
    evidence = _synthetic_dependency_evidence()
    validated = validate_dependency_evidence(evidence)
    assert validated["platform"]["system"] == "Linux"
    assert validated["versions"] == EXPECTED_REFERENCE_DISTRIBUTIONS
    effective = _effective_dependency_manifest(
        evidence,
        {
            "status": "PASS",
            "dependency_evidence_status": "VALIDATED_EXACT_OWNER_REVIEWED",
            "publication_permitted": False,
            "fixture_generation_permitted": True,
            "failures": [],
        },
        validated,
        Path("dependency-evidence.json"),
        "0" * 64,
    )
    assert effective["audit_status"] == "PASS"
    assert effective["review_status"] == "VALIDATED_EXACT_OWNER_REVIEWED"
    assert effective["fixture_generation_permitted"] is True
    assert effective["publication_permitted"] is False
    backports_name = json.loads(json.dumps(evidence))
    backports_row = next(
        row for row in backports_name["installed_distributions"] if row["expected_name"] == "backports-strenum"
    )
    backports_row["metadata"]["name"] = ["backports.strenum"]
    validate_dependency_evidence(backports_name)
    non_equivalent_name = json.loads(json.dumps(backports_name))
    non_equivalent_row = next(
        row for row in non_equivalent_name["installed_distributions"] if row["expected_name"] == "backports-strenum"
    )
    non_equivalent_row["metadata"]["name"] = ["backports.strenum-extra"]
    try:
        validate_dependency_evidence(non_equivalent_name)
    except SystemExit:
        pass
    else:
        raise AssertionError("non-equivalent normalized distribution name was accepted")
    empty_success = json.loads(json.dumps(evidence))
    empty_success["installed_distributions"][0]["license_candidates"] = []
    try:
        validate_dependency_evidence(empty_success)
    except SystemExit as error:
        raise AssertionError("METADATA-declared empty license candidates were rejected") from error
    empty_failure = json.loads(json.dumps(empty_success))
    for field in ("license", "license_expression", "license_classifiers"):
        empty_failure["installed_distributions"][0]["metadata"][field] = []
    try:
        validate_dependency_evidence(empty_failure)
    except SystemExit:
        pass
    else:
        raise AssertionError("empty license evidence without METADATA declaration was accepted")
    only_file_failure = json.loads(json.dumps(empty_success))
    for field in ("license", "license_expression", "license_classifiers"):
        only_file_failure["installed_distributions"][0]["metadata"][field] = []
    assert only_file_failure["installed_distributions"][0]["metadata"]["license_file"] == ["LICENSE"]
    try:
        validate_dependency_evidence(only_file_failure)
    except SystemExit:
        pass
    else:
        raise AssertionError("License-File-only evidence was accepted")
    unknown_failure = json.loads(json.dumps(empty_success))
    unknown_failure["installed_distributions"][0]["license_candidates"] = []
    unknown_failure["installed_distributions"][0]["metadata"]["license"] = ["  UNKNOWN  "]
    unknown_failure["installed_distributions"][0]["metadata"]["license_expression"] = []
    unknown_failure["installed_distributions"][0]["metadata"]["license_classifiers"] = []
    try:
        validate_dependency_evidence(unknown_failure)
    except SystemExit:
        pass
    else:
        raise AssertionError("UNKNOWN metadata license declaration was accepted")
    inventory_tamper = json.loads(json.dumps(evidence))
    inventory_tamper["installed_inventory"]["entries"][0]["version"] = "9.9.9"
    try:
        validate_dependency_evidence(inventory_tamper)
    except SystemExit:
        pass
    else:
        raise AssertionError("installed inventory digest drift was accepted")
    record_tamper = json.loads(json.dumps(evidence))
    record_tamper["installed_distributions"][0]["record"]["entries"].append({"path": "tampered"})
    try:
        validate_dependency_evidence(record_tamper)
    except SystemExit:
        pass
    else:
        raise AssertionError("RECORD entries digest drift was accepted")
    lock_tamper = json.loads(json.dumps(evidence))
    lock_tamper["lock"]["rows"].append({"name": "tampered"})
    try:
        validate_dependency_evidence(lock_tamper)
    except SystemExit:
        pass
    else:
        raise AssertionError("lock rows digest drift was accepted")
    tampered = json.loads(json.dumps(evidence))
    tampered["project"]["sha256"] = "f" * 64
    try:
        validate_dependency_evidence(tampered)
    except SystemExit:
        pass
    else:
        raise AssertionError("dependency project digest drift was accepted")
    unknown = json.loads(json.dumps(evidence))
    unknown["unknown"] = True
    try:
        validate_dependency_evidence(unknown)
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown dependency evidence key was accepted")
    version_drift = json.loads(json.dumps(evidence))
    version_drift["installed_distributions"][0]["expected_version"] = "9.9.9"
    try:
        validate_dependency_evidence(version_drift)
    except SystemExit:
        pass
    else:
        raise AssertionError("dependency version drift was accepted")
    try:
        json.loads('{"schema":"one","schema":"two"}', object_pairs_hook=_object_without_duplicates)
    except DuplicateJsonKey:
        pass
    else:
        raise AssertionError("duplicate dependency evidence key was accepted")
    try:
        require_real_cli_args(argparse.Namespace(tflite_path=None, output_dir=None, dependency_evidence=None))
    except SystemExit:
        pass
    else:
        raise AssertionError("missing dependency-evidence CLI argument was accepted")
    good_in = {"shape": list(INPUT_SHAPE), "dtype": "int8", "quantization": (INPUT_SCALE, INPUT_ZERO_POINT)}
    good_out = {"shape": list(OUTPUT_SHAPE), "dtype": "uint8", "quantization": (OUTPUT_SCALE, OUTPUT_ZERO_POINT)}
    validate_tflite_contract(good_in, good_out)
    for bad_in, bad_out in (
        ({**good_in, "shape": [1, 40]}, good_out),
        ({**good_in, "dtype": "float32"}, good_out),
        ({**good_in, "quantization": (INPUT_SCALE, 0)}, good_out),
        (good_in, {**good_out, "shape": [1]}),
        (good_in, {**good_out, "dtype": "int8"}),
        (good_in, {**good_out, "quantization": (OUTPUT_SCALE, 1)}),
    ):
        try:
            validate_tflite_contract(bad_in, bad_out)
        except ValueError:
            pass
        else:
            raise AssertionError(f"contract drift accepted: {bad_in!r} / {bad_out!r}")
    # Exercise pure shape/quantisation constants and schedule invariants.
    assert INPUT_SHAPE == (1, 3, 40) and OUTPUT_SHAPE == (1, 1)
    assert INPUT_DTYPE_NAME == "int8" and OUTPUT_DTYPE_NAME == "uint8"
    assert INPUT_ZERO_POINT == -128 and OUTPUT_ZERO_POINT == 0
    assert OUTPUT_SCALE == 0.00390625
    assert INVOCATION_COUNT >= 2 and FRAMES_PER_INVOCATION == 3
    with tempfile.TemporaryDirectory(prefix="microwakeword-self-test-") as temporary:
        parent = Path(temporary)
        empty = parent / "empty"
        empty.mkdir()
        try:
            validate_final_output_path(empty)
        except SystemExit:
            pass
        else:
            raise AssertionError("existing output directory was accepted")
        new_final = parent / "new-output"
        staging, cleanup = create_staging_output(new_final)
        assert staging.parent == parent and staging.is_dir()
        cleanup()
        atexit.unregister(cleanup)
        published = parent / "published"
        staging, cleanup = create_staging_output(published)
        (staging / "partial-marker").write_text("complete staging", encoding="utf-8")
        (staging / "manifest.json").write_text("complete manifest", encoding="utf-8")
        publish_staging(staging, published, cleanup)
        assert (published / "partial-marker").read_text(encoding="utf-8") == "complete staging"
        assert sorted(child.name for child in published.iterdir()) == ["manifest.json", "partial-marker"]
        assert not staging.exists()
        shutil.rmtree(published)
        raced_final = parent / "raced-output"
        staging, cleanup = create_staging_output(raced_final)
        (staging / "manifest.json").write_text("{}\n", encoding="utf-8")
        raced_final.mkdir()
        try:
            publish_staging(staging, raced_final, cleanup)
        except SystemExit:
            pass
        else:
            raise AssertionError("publish replaced a raced final directory")
        assert raced_final.is_dir() and not (raced_final / ".vokra-publish-owner").exists()
        cleanup()
        atexit.unregister(cleanup)
        link = parent / "link"
        try:
            link.symlink_to(empty, target_is_directory=True)
        except OSError:
            pass  # Windows without symlink privileges; Linux VAST covers it.
        else:
            try:
                validate_final_output_path(link)
            except SystemExit:
                pass
            else:
                raise AssertionError("symlink output directory was accepted")
        source = parent / "source.tflite"
        source.write_bytes(b"placeholder")
        require_regular_tflite(source)
        evidence_path = parent / "dependency-evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        loaded, evidence_sha256, _ = load_dependency_evidence(evidence_path)
        validate_dependency_evidence(loaded)
        assert len(evidence_sha256) == 64
        evidence_link = parent / "dependency-evidence-link.json"
        try:
            evidence_link.symlink_to(evidence_path)
        except OSError:
            pass
        else:
            try:
                load_dependency_evidence(evidence_link)
            except SystemExit:
                pass
            else:
                raise AssertionError("symlink dependency evidence was accepted")
        source_link = parent / "source-link.tflite"
        try:
            source_link.symlink_to(source)
        except OSError:
            pass
        else:
            try:
                require_regular_tflite(source_link)
            except SystemExit:
                pass
            else:
                raise AssertionError("symlink TFLite input was accepted")
    print("microWakeWord reference self-test: PASS (no model/interpreter)", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="microWakeWord authenticated TFLite reference dumper.")
    ap.add_argument("--self-test", action="store_true", help="Run model-free contract checks.")
    ap.add_argument("--tflite-path", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--dependency-evidence", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        if any(
            value is not None
            for value in (args.tflite_path, args.output_dir, args.dependency_evidence)
        ) or args.verbose:
            ap.error("--self-test cannot be combined with real-input arguments")
        return self_test()
    try:
        require_real_cli_args(args)
    except SystemExit as error:
        ap.error(str(error))
    require_regular_tflite(args.tflite_path)
    dependency_evidence, dependency_evidence_sha256, dependency_evidence_bytes = load_dependency_evidence(args.dependency_evidence)
    dependency_contract = validate_dependency_evidence(dependency_evidence)
    audit_report = require_reference_dependency_gate(dependency_evidence_bytes)
    reference_environment = require_reference_runtime(dependency_contract["versions"])
    global np
    import numpy as np

    tflite_sha256 = sha256_of_file(args.tflite_path)
    if tflite_sha256 != AUTHENTICATED_TFLITE_SHA256:
        raise SystemExit(f"refusing unauthenticated TFLite SHA-256 {tflite_sha256}; expected {AUTHENTICATED_TFLITE_SHA256}")
    tflite_size = args.tflite_path.stat().st_size
    if tflite_size != AUTHENTICATED_TFLITE_BYTES:
        raise SystemExit(f"refusing authenticated SHA with unexpected byte size {tflite_size}; expected {AUTHENTICATED_TFLITE_BYTES}")
    final_output_dir = args.output_dir
    staging_output_dir, staging_cleanup = create_staging_output(final_output_dir)
    args.output_dir = staging_output_dir

    pcm_frames = [synth_pcm(i, f) for i in range(INVOCATION_COUNT) for f in range(FRAMES_PER_INVOCATION)]
    feature_sequence = np.stack([numpy_log_mel_features(pcm) for pcm in pcm_frames]).astype(np.float32)
    dump_le(pcm_frames[0], args.output_dir / "input_pcm.bin")
    dump_le(feature_sequence[0], args.output_dir / "features_ref.bin")
    for invocation in range(INVOCATION_COUNT):
        frames = feature_sequence[invocation * FRAMES_PER_INVOCATION : (invocation + 1) * FRAMES_PER_INVOCATION]
        dump_le(frames, args.output_dir / f"features_invocation_{invocation:02d}.bin")
        dump_le(quantize_features_int8(frames), args.output_dir / f"input_invocation_{invocation:02d}.bin")

    from ai_edge_litert.interpreter import Interpreter

    def new_interpreter() -> Any:
        fresh = Interpreter(model_path=str(args.tflite_path))
        fresh.allocate_tensors()
        return fresh

    raw_outputs, tflite_meta = run_tflite_sequence(new_interpreter(), feature_sequence, args.verbose)
    replay_outputs, _ = run_tflite_sequence(new_interpreter(), feature_sequence, False)
    if not all(np.array_equal(a, b) for a, b in zip(raw_outputs, replay_outputs, strict=True)):
        raise SystemExit("fresh-interpreter reset replay mismatch")
    output_f32 = [((raw.astype(np.float32) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE) for raw in raw_outputs]
    artefacts: list[dict[str, Any]] = [
        artifact(args.output_dir / "input_pcm.bin", "input_pcm", (WINDOW_SAMPLES,), "int16", "first frame for separate numpy/Rust frontend transcription"),
        artifact(args.output_dir / "features_ref.bin", "features_ref", (N_MELS,), "float32", "first frame numpy frontend transcription; not TFLite parity"),
    ]
    for invocation, (raw, dequant) in enumerate(zip(raw_outputs, output_f32, strict=True)):
        raw_path = args.output_dir / f"output_invocation_{invocation:02d}.bin"
        f32_path = args.output_dir / f"output_invocation_{invocation:02d}_f32.bin"
        artefacts.append(artifact(args.output_dir / f"features_invocation_{invocation:02d}.bin", f"features_invocation_{invocation:02d}", (FRAMES_PER_INVOCATION, N_MELS), "float32", "numpy frontend transcription used to form this invocation; not the TFLite oracle"))
        dump_le(raw, raw_path)
        dump_le(dequant, f32_path)
        artefacts.extend([
            artifact(args.output_dir / f"input_invocation_{invocation:02d}.bin", f"input_invocation_{invocation:02d}", INPUT_SHAPE, "int8", "exact quantised bytes fed to persistent upstream interpreter"),
            artifact(raw_path, f"output_invocation_{invocation:02d}", OUTPUT_SHAPE, "uint8", "raw upstream TFLite output"),
            artifact(f32_path, f"output_invocation_{invocation:02d}_f32", OUTPUT_SHAPE, "float32", "affine dequantisation of raw uint8 output"),
        ])
    concat_f32 = np.concatenate(output_f32, axis=0).astype(np.float32)
    dump_le(concat_f32, args.output_dir / "output_ref.bin")
    artefacts.append(artifact(args.output_dir / "output_ref.bin", "output_ref", (INVOCATION_COUNT, 1), "float32", "legacy aggregate of per-invocation dequantised outputs"))
    manifest: dict[str, Any] = {
        "schema": "microwakeword-reference-v2",
        "status": "REFERENCE_COMPLETE",
        "generator": "vokra tools/parity/microwakeword/dump_reference.py",
        "generator_version": "0.2.0-authenticated-streaming",
        "oracle": "ai_edge_litert.Interpreter running the pinned upstream TFLite; never a Vokra mirror",
        "source_tflite": str(args.tflite_path),
        "source_tflite_sha256": tflite_sha256,
        "source_tflite_bytes": tflite_size,
        "authenticated_model_sha256": AUTHENTICATED_TFLITE_SHA256,
        "constants": {"sample_rate": SAMPLE_RATE, "hop_ms": HOP_MS, "window_ms": WINDOW_MS, "n_mels": N_MELS, "hop_samples": HOP_SAMPLES, "window_samples": WINDOW_SAMPLES, "n_fft": N_FFT, "n_bins": N_BINS, "log_mel_epsilon": LOG_MEL_EPSILON},
        "pcm_synthesis": {"seed": PCM_SEED, "sine_hz": PCM_SINE_HZ, "sine_amplitude": PCM_SINE_AMPLITUDE, "noise_stddev": PCM_NOISE_STDDEV, "distinct_frame_schedule": "frequency += 37 Hz and seed += global frame index"},
        "authenticated_io": {"input": {"shape": list(INPUT_SHAPE), "dtype": "int8", "scale": INPUT_SCALE, "zero_point": INPUT_ZERO_POINT}, "output": {"shape": list(OUTPUT_SHAPE), "dtype": "uint8", "scale": OUTPUT_SCALE, "zero_point": OUTPUT_ZERO_POINT}},
        "persistent_sequence": {"invocation_count": INVOCATION_COUNT, "frames_per_invocation": FRAMES_PER_INVOCATION, "distinct_frames": True, "single_persistent_interpreter": True, "fresh_interpreter_reset_replay": {"status": "PASS", "invocation_count": INVOCATION_COUNT, "raw_outputs_match": True}},
        "artefacts": artefacts,
        "tflite_topology": tflite_meta,
        "frontend_parity_boundary": "features_ref is a numpy transcription kept separate from the independent TFLite oracle",
        "reference_environment": {
            "python": reference_environment["python"],
            "system": reference_environment["system"],
            "machine": reference_environment["machine"],
            "installed_distributions": reference_environment["installed_distributions"],
        },
        "dependency_evidence": _effective_dependency_manifest(
            dependency_evidence,
            audit_report,
            dependency_contract,
            args.dependency_evidence,
            dependency_evidence_sha256,
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise SystemExit(f"refusing to overwrite existing manifest: {manifest_path}")
    with manifest_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")
    publish_staging(staging_output_dir, final_output_dir, staging_cleanup)
    print(f"Wrote authenticated {INVOCATION_COUNT}-invocation fixture to {final_output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
