#!/usr/bin/env -S uv run --frozen --project tools/parity/vibevoice_1_5b_reference --python 3.12 python
"""Run the pinned Microsoft VibeVoice reference and persist small evidence.

This is intentionally an adapter around the upstream orphan commit.  It does
not contain a VibeVoice implementation and it never samples an implicit RNG:
the caller supplies token ids, prompt PCM, and Gaussian draws in a JSON
packet.  The large model is only loaded on the VAST worker.  If the exact
upstream API cannot consume the packet, this tool writes a blocked manifest
instead of silently substituting a local implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HF_REPOSITORY = "microsoft/VibeVoice-1.5B"
HF_REVISION = "142f4a5dda029212cda8b118e9d99c3da27018d8"
QWEN_REPOSITORY = "Qwen/Qwen2.5-1.5B"
QWEN_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
SOURCE_REPOSITORY = "https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION = "2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c"
SOURCE_ROLE = "vibevoice/modular/modeling_vibevoice_inference.py"
SOURCE_MODEL_ROLE = "vibevoice/modular/modeling_vibevoice.py"
SOURCE_ROLES = (
    "LICENSE", "pyproject.toml", "demo/inference_from_file.py",
    "vibevoice/modular/configuration_vibevoice.py",
    "vibevoice/modular/modeling_vibevoice.py",
    "vibevoice/modular/modeling_vibevoice_inference.py",
    "vibevoice/modular/modular_vibevoice_tokenizer.py",
    "vibevoice/modular/modular_vibevoice_text_tokenizer.py",
    "vibevoice/modular/modular_vibevoice_diffusion_head.py",
    "vibevoice/modular/streamer.py",
    "vibevoice/processor/vibevoice_processor.py",
    "vibevoice/processor/vibevoice_tokenizer_processor.py",
    "vibevoice/schedule/dpm_solver.py",
    "vibevoice/configs/qwen2.5_1.5b_64k.json",
)
SOURCE_LICENSE_BLOB = "269a8973689dbb250d355f516f8a30c1cc66b8e4"
SOURCE_ROLE_BLOBS: dict[str, str] = {
    "LICENSE": "269a8973689dbb250d355f516f8a30c1cc66b8e4",
    "pyproject.toml": "ece97ec7b9177119f4fdd1fb1f329b430876dd89",
    "demo/inference_from_file.py": "078b53a11f0e4bf617171101655c11d0d394e66b",
    "vibevoice/configs/qwen2.5_1.5b_64k.json": "febd05cd76d2a5df49c39fabcde478aa18e1ba78",
    "vibevoice/modular/configuration_vibevoice.py": "fcffcb93afae6358f57a155d6fb6eb009b69a706",
    "vibevoice/modular/modeling_vibevoice.py": "016a38979ef74e1ea9c5dc0405c8ac13feb0a0d5",
    "vibevoice/modular/modeling_vibevoice_inference.py": "7e10af4a2bd1f5ba4ec454942e4a87bb312aa091",
    "vibevoice/modular/modular_vibevoice_diffusion_head.py": "59de50fb2fe80d6b1ba5a50c9de1ef9cffc4f614",
    "vibevoice/modular/modular_vibevoice_text_tokenizer.py": "bfa7bdd18783d67d488371071cc6425ceb80b376",
    "vibevoice/modular/modular_vibevoice_tokenizer.py": "fbd5182f82ba61898a09b762ec20e6f34270d053",
    "vibevoice/modular/streamer.py": "7a76cb063ec1b48a9e6397f113b47663ae6c5799",
    "vibevoice/processor/vibevoice_processor.py": "66d0a9de2e2beb3eeeaf0bb5a5eb523d5f61acae",
    "vibevoice/processor/vibevoice_tokenizer_processor.py": "0d854b7842658dbb573b6623c05d1326a71221cf",
    "vibevoice/schedule/dpm_solver.py": "806241f4352465f50114b587e0db2c63bc73c24f",
}
FORMAT = "vokra-vibevoice-1-5b-reference-v1"
REFERENCE_PROJECT = Path(__file__).with_name("vibevoice_1_5b_reference")
REFERENCE_LOCK = REFERENCE_PROJECT / "uv.lock"
REFERENCE_LOCK_SHA256 = "a1aa0b371e5036a7f5bc72f2a5e1ba82ef21a6fa9ba8993e5612fb7612107806"
REFERENCE_PACKAGE_ROWS_SHA256 = "ae07242d3b0e4d8fdda8b7435956b835a996e003a6615660358a01dbfd9bddf6"
REFERENCE_PACKAGE_COUNT = 32
REFERENCE_PACKAGE_ROWS_SCHEMA = "package-resolution-and-dependency-markers-v2"
REFERENCE_LICENSE_ROWS_SHA256 = "6cca02093a2b76c728f0957193657f614e6f443e13805705423b384c5aa6c0ca"
REFERENCE_LICENSE_COUNT = 32
REFERENCE_LICENSE_STATUS = "BLOCKED_UNREVIEWED_TRANSITIVE"
REFERENCE_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
REFERENCE_RESOLUTION_MARKERS = ("sys_platform != 'darwin'", "sys_platform == 'darwin'")
REFERENCE_SOURCE_IMPORT_CLOSURE = ("torch", "numpy", "tqdm", "transformers", "diffusers.DPMSolverMultistepScheduler")
REFERENCE_EXCLUDED_OPTIONAL = ("accelerate", "gradio", "av", "aiortc", "ml-collections", "absl-py")
REFERENCE_EXCLUDED_UNUSED_OR_FORBIDDEN = ("soundfile", "libsndfile", "librosa", "soxr")
REFERENCE_LICENSE_MAP = "version-keyed name==version mapping emitted by vibevoice_1_5b_dump_reference.py from authenticated package_rows"
REFERENCE_LICENSE_BLOCKERS = (
    "certifi==2026.7.22 carries MPL-2.0 file-level notices outside the repository Apache/MIT/BSD allowlist; owner policy clearance is required",
    "filelock==3.32.4 declares Unlicense; owner policy clearance is required",
    "numpy==2.2.6 carries bundled OpenBLAS/LAPACK and GCC runtime notices including GPL/LGPL components; reference-only treatment is not owner-approved",
    "pyyaml==6.0.3 includes a native extension and requires separate source/binary review before use",
    "tqdm==4.67.1 contains MPL-2.0 files alongside MIT files; owner policy clearance is required",
    "typing-extensions==4.16.0 declares PSF-2.0 outside the repository Apache/MIT/BSD allowlist; owner policy clearance is required",
)
REFERENCE_LICENSE_PRIMARY_EVIDENCE = (
    "https://pypi.org/pypi/certifi/2026.7.22/json",
    "https://pypi.org/pypi/filelock/3.32.4/json",
    "https://pypi.org/pypi/numpy/2.2.6/json",
    "https://pypi.org/pypi/pyyaml/6.0.3/json",
    "https://pypi.org/pypi/tqdm/4.67.1/json",
    "https://pypi.org/pypi/typing-extensions/4.16.0/json",
)
MAX_TOKENS = 65_536
MAX_PCM_SAMPLES = 24_000 * 60 * 10
MAX_DRAWS = 10_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_lock_rows(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the complete, versioned lock identity used by the gate."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("dedicated VibeVoice lock package table is malformed")
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise RuntimeError("dedicated VibeVoice lock contains a malformed package row")
        source = package.get("source")
        if not isinstance(source, dict) or set(source) not in ({"registry"}, {"virtual"}) or not isinstance(next(iter(source.values())), str):
            raise RuntimeError("dedicated VibeVoice lock package source is malformed")
        markers = package.get("resolution-markers", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            raise RuntimeError("dedicated VibeVoice lock package markers are malformed")
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise RuntimeError("dedicated VibeVoice lock dependency table is malformed")
        canonical_dependencies: list[dict[str, Any]] = []
        for dependency in dependencies:
            if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
                raise RuntimeError("dedicated VibeVoice lock dependency row is malformed")
            if "version" in dependency and not isinstance(dependency["version"], str):
                raise RuntimeError("dedicated VibeVoice lock dependency version qualifier is malformed")
            if "marker" in dependency and not isinstance(dependency["marker"], str):
                raise RuntimeError("dedicated VibeVoice lock dependency marker qualifier is malformed")
            if "source" in dependency:
                dependency_source = dependency["source"]
                if not isinstance(dependency_source, dict) or not dependency_source or any(not isinstance(key, str) or not isinstance(value, str) for key, value in dependency_source.items()):
                    raise RuntimeError("dedicated VibeVoice lock dependency source qualifier is malformed")
            # Preserve every dependency-level qualifier (version/source/marker,
            # and any future uv qualifier) rather than reducing it to a name.
            canonical_dependencies.append({
                key: ({subkey: dependency[key][subkey] for subkey in sorted(dependency[key])} if key == "source" else dependency[key])
                for key in sorted(dependency)
            })
        canonical_dependencies.sort(key=lambda dependency: json.dumps(dependency, sort_keys=True, separators=(",", ":")))
        rows.append({
            "name": package["name"],
            "version": package["version"],
            "source": {key: source[key] for key in sorted(source)},
            "markers": sorted(markers),
            "dependencies": canonical_dependencies,
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["markers"]))


def lock_rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_license_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {"name", "version", "license", "primary_source", "route"}
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required or any(not isinstance(row[key], str) or not row[key] for key in required):
            raise RuntimeError("dedicated VibeVoice license conclusion row is malformed")
        result.append({key: row[key] for key in sorted(required)})
    return sorted(result, key=lambda row: (row["name"], row["version"], row["license"], row["primary_source"], row["route"]))


def license_rows_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def reference_lock_identity(lock_path: Path = REFERENCE_LOCK) -> dict[str, Any]:
    if not lock_path.is_file():
        raise RuntimeError("dedicated VibeVoice uv.lock is absent")
    digest = sha256(lock_path)
    if digest != REFERENCE_LOCK_SHA256:
        raise RuntimeError("dedicated VibeVoice uv.lock SHA-256 is not the reviewed identity")
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    if lock.get("requires-python") != "==3.12.*" or tuple(lock.get("resolution-markers", ())) != REFERENCE_RESOLUTION_MARKERS:
        raise RuntimeError("dedicated VibeVoice lock is not restricted to the reviewed Python 3.12 markers")
    with (REFERENCE_PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    project_metadata = project.get("tool", {}).get("vokra", {}).get("vibevoice_1_5b_reference", {})
    if tuple(project_metadata.get("source_import_closure", ())) != REFERENCE_SOURCE_IMPORT_CLOSURE:
        raise RuntimeError("dedicated VibeVoice source import-closure declaration drifted")
    if tuple(project_metadata.get("excluded_upstream_optional", ())) != REFERENCE_EXCLUDED_OPTIONAL:
        raise RuntimeError("dedicated VibeVoice optional dependency exclusion declaration drifted")
    if tuple(project_metadata.get("excluded_forbidden_or_unused", ())) != REFERENCE_EXCLUDED_UNUSED_OR_FORBIDDEN:
        raise RuntimeError("dedicated VibeVoice forbidden/unused dependency exclusion declaration drifted")
    if project_metadata.get("lock_package_rows_schema") != REFERENCE_PACKAGE_ROWS_SCHEMA:
        raise RuntimeError("dedicated VibeVoice lock package-row schema declaration drifted")
    rows = canonical_lock_rows(lock)
    if len(rows) != REFERENCE_PACKAGE_COUNT or lock_rows_sha256(rows) != REFERENCE_PACKAGE_ROWS_SHA256:
        raise RuntimeError("dedicated VibeVoice lock package rows are not the reviewed identity")
    names = {row["name"] for row in rows}
    forbidden = {"accelerate", "av", "aiortc", "librosa", "ml-collections", "gradio", "soundfile", "soxr", "triton"}
    if names & forbidden or any(name.startswith("nvidia-") for name in names):
        raise RuntimeError("dedicated VibeVoice lock includes excluded optional or forbidden packages")
    for row in rows:
        if row["name"] == "torch" and row["source"].get("registry") != REFERENCE_CPU_INDEX:
            raise RuntimeError("dedicated VibeVoice torch is not routed exclusively to the official CPU index")
    return {
        "path": str(lock_path),
        "sha256": digest,
        "python": lock["requires-python"],
        "resolution_markers": list(REFERENCE_RESOLUTION_MARKERS),
        "package_count": len(rows),
        "package_rows_sha256": REFERENCE_PACKAGE_ROWS_SHA256,
        "package_rows_schema": REFERENCE_PACKAGE_ROWS_SCHEMA,
        "package_rows": rows,
        "package_names": sorted(names),
        "cpu_index": REFERENCE_CPU_INDEX,
        "torch_distribution_versions": ["2.7.1", "2.7.1+cpu"],
    }


def license_audit_identity(lock_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Authenticate exact audit rows, then preserve the unresolved verdict."""
    if lock_record is None:
        lock_record = reference_lock_identity()
    with (REFERENCE_PROJECT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    metadata = project.get("tool", {}).get("vokra", {}).get("vibevoice_1_5b_reference", {}).get("license_audit", {})
    if not isinstance(metadata, dict) or metadata.get("status") != REFERENCE_LICENSE_STATUS:
        raise RuntimeError("dedicated VibeVoice license audit is missing or not fail-closed")
    package_names = set(lock_record.get("package_names", ()))
    if set(metadata.get("reviewed_packages", ())) != package_names:
        raise RuntimeError("dedicated VibeVoice license audit package inventory drifted")
    conclusion_rows = metadata.get("license_conclusions")
    if not isinstance(conclusion_rows, list) or len(conclusion_rows) != REFERENCE_LICENSE_COUNT:
        raise RuntimeError("dedicated VibeVoice versioned license conclusion inventory is incomplete")
    normalized = canonical_license_rows(conclusion_rows)
    if license_rows_sha256(normalized) != REFERENCE_LICENSE_ROWS_SHA256 or metadata.get("license_audit_rows_sha256") != REFERENCE_LICENSE_ROWS_SHA256:
        raise RuntimeError("dedicated VibeVoice license audit rows are not the reviewed identity")
    lock_keys = {(row["name"], row["version"]) for row in lock_record["package_rows"]}
    audit_keys = {(row["name"], row["version"]) for row in normalized}
    if lock_keys != audit_keys:
        raise RuntimeError("dedicated VibeVoice license rows do not cover every locked version")
    blockers = metadata.get("blockers")
    if not isinstance(blockers, list) or tuple(blockers) != REFERENCE_LICENSE_BLOCKERS:
        raise RuntimeError("dedicated VibeVoice license blockers drifted")
    evidence = metadata.get("primary_evidence")
    if not isinstance(evidence, list) or tuple(evidence) != REFERENCE_LICENSE_PRIMARY_EVIDENCE:
        raise RuntimeError("dedicated VibeVoice license primary evidence drifted")
    if metadata.get("license_conclusion_map") != REFERENCE_LICENSE_MAP or metadata.get("license_conclusion_count") != REFERENCE_LICENSE_COUNT:
        raise RuntimeError("dedicated VibeVoice license conclusion declaration drifted")
    if not isinstance(metadata.get("primary_metadata"), str) or not metadata["primary_metadata"]:
        raise RuntimeError("dedicated VibeVoice license primary metadata is missing")
    return {
        "status": metadata["status"],
        "reviewed_packages": sorted(package_names),
        "blockers": list(blockers),
        "primary_metadata": metadata.get("primary_metadata"),
        "primary_evidence": list(evidence),
        "license_conclusions": normalized,
        "license_audit_rows_sha256": REFERENCE_LICENSE_ROWS_SHA256,
        "license_conclusion_count": len(normalized),
    }


def reference_environment_identity() -> dict[str, Any]:
    lock = reference_lock_identity()
    audit = license_audit_identity(lock)
    return {"lock": lock, "license_audit": audit}


def require_license_clearance() -> dict[str, Any]:
    environment = reference_environment_identity()
    if environment["license_audit"]["status"] != "AUTHENTICATED_CLEAR":
        raise RuntimeError("dedicated VibeVoice dependency license audit is unresolved; uv sync/model acquisition/reference execution are blocked")
    return environment


def strict_json(value: str | bytes, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate key {key!r}")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=no_duplicates)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(source: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != SOURCE_REVISION:
        raise RuntimeError("official orphan source revision drift")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("official source checkout is dirty")
    origin = git(source, "remote", "get-url", "origin").rstrip("/")
    if origin.removesuffix(".git") != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("official source origin drift")
    rows: list[dict[str, Any]] = []
    tracked_output = git(source, "ls-files", "--stage")
    tracked: set[str] = set()
    for line in tracked_output.splitlines():
        metadata, role_name = line.split("\t", 1)
        mode, index_object, stage = metadata.split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise RuntimeError(f"upstream tracked entry is not stage-0 regular 100644/100755: {role_name}")
        if role_name in tracked:
            raise RuntimeError(f"duplicate upstream tracked entry: {role_name}")
        tracked.add(role_name)
        role = source / role_name
        if role.is_symlink() or not role.is_file() or (role.stat().st_mode & 0o7777) != int(mode[-4:], 8):
            raise RuntimeError(f"upstream tracked entry filesystem mode drift: {role_name}")
        head_object = git(source, "rev-parse", f"HEAD:{role_name}")
        working_object = git_blob_sha1(role)
        if index_object != head_object or index_object != working_object:
            raise RuntimeError(f"upstream tracked object drift: {role_name}")
        rows.append({"path": role_name, "mode": mode[-4:], "stage": 0, "bytes": role.stat().st_size, "index_object": index_object, "head_object": head_object, "working_git_blob_sha1": working_object})
    if set(SOURCE_ROLES) - tracked:
        raise RuntimeError(f"missing tracked upstream roles: {sorted(set(SOURCE_ROLES) - tracked)}")
    if set(SOURCE_ROLE_BLOBS) != set(SOURCE_ROLES):
        raise RuntimeError("complete fixed Microsoft role Git table is unavailable")
    for row in rows:
        if row["path"] in SOURCE_ROLE_BLOBS and row["head_object"].lower() != SOURCE_ROLE_BLOBS[row["path"]].lower():
            raise RuntimeError(f"fixed Microsoft role object drift: {row['path']}")
        if row["path"] in SOURCE_ROLE_BLOBS and row["mode"] != "0644":
            raise RuntimeError(f"fixed Microsoft role mode drift: {row['path']}")
    if git(source, "rev-parse", "HEAD:LICENSE").lower() != SOURCE_LICENSE_BLOB:
        raise RuntimeError("pinned source LICENSE object drift")
    source_text = "\n".join(
        (source / role_name).read_text(encoding="utf-8", errors="strict")
        for role_name in SOURCE_ROLES
        if role_name.endswith(".py")
    )
    import_closure = {
        "torch": "import torch" in source_text or "from torch" in source_text,
        "numpy": "import numpy" in source_text or "from numpy" in source_text,
        "tqdm": "tqdm" in source_text,
        "transformers": "transformers" in source_text,
        "diffusers.DPMSolverMultistepScheduler": "DPMSolverMultistepScheduler" in source_text and "diffusers" in source_text,
    }
    if not all(import_closure.values()):
        missing = [name for name, present in import_closure.items() if not present]
        raise RuntimeError(f"authenticated VibeVoice import closure proof is incomplete: {missing}")
    license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="strict")
    if "MIT License" not in license_text or "Permission is hereby granted, free of charge" not in license_text or "THE SOFTWARE IS PROVIDED \"AS IS\"" not in license_text:
        raise RuntimeError("pinned source MIT grant/warranty clauses are incomplete")
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "origin": origin,
        "tracked_file_count": len(rows),
        "tracked_files": rows,
        "roles": [row for row in rows if row["path"] in SOURCE_ROLES],
        "import_closure": import_closure,
        "license": {"path": "LICENSE", "git_blob_sha1": SOURCE_LICENSE_BLOB, "clauses": ["MIT License", "Permission is hereby granted, free of charge", "THE SOFTWARE IS PROVIDED \"AS IS\""]},
    }


def validate_packet(packet: Any) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("reference packet must be an object")
    required = {
        "token_ids",
        "processor_text",
        "speech_replacement_positions",
        "speech_input_mask",
        "speech_masks",
        "prompt_pcm",
        "prompt_sample_rate_hz",
        "guidance_scale",
        "max_generated_tokens",
        "seed",
        "random_draws",
    }
    if set(packet) != required:
        raise ValueError("reference packet fields must be exact")
    token_ids = packet["token_ids"]
    if not isinstance(token_ids, list) or not token_ids or len(token_ids) > MAX_TOKENS or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in token_ids):
        raise ValueError("token_ids must be bounded non-negative integers")
    if (
        not isinstance(packet["processor_text"], str)
        or not packet["processor_text"].strip()
        or len(packet["processor_text"]) > 16_384
        or "\x00" in packet["processor_text"]
    ):
        raise ValueError("processor_text must be a bounded UTF-8 text request")
    positions = packet["speech_replacement_positions"]
    if not isinstance(positions, list) or len(positions) > len(token_ids) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= len(token_ids) for x in positions) or len(set(positions)) != len(positions):
        raise ValueError("speech_replacement_positions must be unique in-range integers")
    input_mask = packet["speech_input_mask"]
    if (
        not isinstance(input_mask, list)
        or len(input_mask) != len(token_ids)
        or any(value not in (0, 1) or isinstance(value, bool) for value in input_mask)
        or [index for index, value in enumerate(input_mask) if value]
        != positions
    ):
        raise ValueError("speech_input_mask must exactly describe replacement positions")
    speech_masks = packet["speech_masks"]
    if (
        not isinstance(speech_masks, list)
        or not speech_masks
        or any(value not in (0, 1) or isinstance(value, bool) for value in speech_masks)
        or sum(speech_masks) != len(positions)
    ):
        raise ValueError("speech_masks must be the prepared latent-frame mask")
    pcm = packet["prompt_pcm"]
    if (
        not isinstance(pcm, list)
        or not pcm
        or len(pcm) > MAX_PCM_SAMPLES
        or len(pcm) % 3_200 != 0
        or any(
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not float(x) == float(x)
            or abs(float(x)) == float("inf")
            for x in pcm
        )
    ):
        raise ValueError("prompt_pcm must be a non-empty 3200-sample-aligned bounded finite array")
    if packet["prompt_sample_rate_hz"] != 24_000:
        raise ValueError("prompt_sample_rate_hz must be 24000")
    draws = packet["random_draws"]
    if not isinstance(draws, list) or not draws or len(draws) > MAX_DRAWS:
        raise ValueError("random_draws must be bounded records")
    names = ["prompt_latent", "diffusion_initial"]
    seen: list[str] = []
    for record in draws:
        if not isinstance(record, dict) or set(record) != {"name", "shape", "dtype", "values"}:
            raise ValueError("random draw records must have exact fields")
        name, shape, dtype, values = (record[key] for key in ("name", "shape", "dtype", "values"))
        if name not in names or (seen and names.index(name) < names.index(seen[-1])):
            raise ValueError("random draw records are not in official semantic order")
        if not isinstance(shape, list) or not shape or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in shape):
            raise ValueError("random draw shape is invalid")
        if dtype not in {"float32", "bfloat16", "float16"} or not isinstance(values, list):
            raise ValueError("random draw dtype/values are invalid")
        count = 1
        for dimension in shape:
            count *= dimension
        if count != len(values) or count > MAX_DRAWS or any(not isinstance(x, (int, float)) or isinstance(x, bool) or not float(x) == float(x) or abs(float(x)) == float("inf") for x in values):
            raise ValueError("random draw shape/value count mismatch")
        seen.append(name)
    if seen[0] != "prompt_latent" or seen[-1] != "diffusion_initial":
        raise ValueError("random draw records must cover prompt latent then diffusion calls")
    if not isinstance(packet["guidance_scale"], (int, float)) or isinstance(packet["guidance_scale"], bool) or not float(packet["guidance_scale"]) == float(packet["guidance_scale"]):
        raise ValueError("guidance_scale must be finite")
    if not isinstance(packet["max_generated_tokens"], int) or isinstance(packet["max_generated_tokens"], bool) or not 0 < packet["max_generated_tokens"] <= MAX_TOKENS:
        raise ValueError("max_generated_tokens must be bounded and positive")
    if not isinstance(packet["seed"], int) or isinstance(packet["seed"], bool) or packet["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    return packet


def generated_segment(sequence: list[int], prompt_length: int, max_generated_tokens: int) -> list[int]:
    """Return only tokens emitted after the caller's prompt.

    The upstream generation API may stop before its cap.  Slicing from the
    tail would incorrectly re-introduce prompt tokens in that case.
    """
    if prompt_length < 0 or len(sequence) < prompt_length:
        raise ValueError("official sequence is shorter than the input prompt")
    generated = sequence[prompt_length:]
    if len(generated) > max_generated_tokens:
        raise ValueError("official generation exceeded its requested cap")
    return generated


def validate_generate_signature(signature: inspect.Signature) -> None:
    """Require the fixed source API without inventing an ``input_ids`` arg."""
    parameters = signature.parameters
    required = {"inputs", "speech_tensors", "speech_masks", "speech_input_mask", "cfg_scale", "return_speech"}
    if not required.issubset(parameters):
        raise RuntimeError("pinned official generate signature is not the VibeVoice batch-one API")
    if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        raise RuntimeError("official generate must accept processor input_ids through **kwargs")


def diffusion_token_count(tokens: list[int]) -> int:
    """Count only speech diffusion decisions, excluding control markers."""
    return sum(token == 151_654 for token in tokens)


def processor_call_kwargs(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the exact official demo-shaped processor call.

    Keeping this boundary explicit makes it testable without importing the
    multi-gigabyte model.  The upstream API calls the argument
    ``voice_samples``; ``audio`` is a different Transformers convention.
    """
    import numpy as np

    return {
        "text": [packet["processor_text"]],
        "voice_samples": [[np.asarray(packet["prompt_pcm"], dtype=np.float32)]],
        "padding": True,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }


def generation_call_kwargs(prepared: Mapping[str, Any], tokenizer: Any, packet: dict[str, Any]) -> dict[str, Any]:
    """Build the official ``model.generate(**inputs, ...)`` call once."""
    kwargs = dict(prepared)
    if "inputs" in kwargs and "input_ids" in kwargs:
        raise RuntimeError("processor mapping contains conflicting inputs aliases")
    kwargs.update({
        "tokenizer": tokenizer,
        "max_new_tokens": packet["max_generated_tokens"],
        "return_speech": True,
        "cfg_scale": packet["guidance_scale"],
        "do_sample": False,
    })
    if "is_prefill" in kwargs:
        raise RuntimeError("internal is_prefill must not be passed to official generate")
    return kwargs


def prepared_inputs(source: Path, snapshot: Path, tokenizer: Any, packet: dict[str, Any], torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the pinned processor with the authenticated local tokenizer.

    The preprocessor's language-model name is a remote default.  Never let
    ``from_pretrained`` resolve that companion implicitly: instantiate the
    official feature extractor and processor classes from the local snapshot.
    API drift is a hard error, not a fallback to a mirror implementation.
    """
    processor_role = source / "vibevoice/processor/vibevoice_processor.py"
    if not processor_role.is_file():
        raise RuntimeError("pinned processor role is missing")
    processor_module = importlib.import_module("vibevoice.processor.vibevoice_processor")
    processor_cls = getattr(processor_module, "VibeVoiceProcessor", None)
    if processor_cls is None:
        raise RuntimeError("pinned source exposes no VibeVoiceProcessor")
    preprocessor = strict_json((snapshot / "preprocessor_config.json").read_bytes(), "preprocessor_config.json")
    if not isinstance(preprocessor, dict) or preprocessor.get("processor_class") != "VibeVoiceProcessor":
        raise RuntimeError("authenticated preprocessor config does not select VibeVoiceProcessor")
    feature_module = importlib.import_module("vibevoice.processor.vibevoice_tokenizer_processor")
    feature_cls = getattr(feature_module, "VibeVoiceTokenizerProcessor", None)
    feature_loader = getattr(feature_cls, "from_pretrained", None) if feature_cls is not None else None
    if not callable(feature_loader):
        raise RuntimeError("official tokenizer processor has no local from_pretrained loader")
    feature_extractor = feature_loader(str(snapshot), local_files_only=True)
    if not {"feature_extractor", "tokenizer"}.issubset(inspect.signature(processor_cls).parameters):
        raise RuntimeError("official processor cannot bind the authenticated local tokenizer")
    processor = processor_cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    prepared = processor(**processor_call_kwargs(packet))
    if not isinstance(prepared, Mapping):
        raise RuntimeError("official processor did not return a mapping")
    required = {"input_ids", "speech_tensors", "speech_masks", "speech_input_mask"}
    if not required.issubset(prepared):
        raise RuntimeError("official processor preparation fields are incomplete")
    actual_ids = prepared["input_ids"].detach().cpu().reshape(-1).tolist()
    actual_input_mask = [int(value) for value in prepared["speech_input_mask"].detach().cpu().reshape(-1).tolist()]
    actual_speech_masks = [int(value) for value in prepared["speech_masks"].detach().cpu().reshape(-1).tolist()]
    if actual_ids != packet["token_ids"]:
        raise RuntimeError("packet token_ids differ from official processor output")
    if actual_input_mask != packet["speech_input_mask"]:
        raise RuntimeError("packet speech_input_mask differs from official processor output")
    if actual_speech_masks != packet["speech_masks"]:
        raise RuntimeError("packet speech_masks differs from official processor output")
    tensors = prepared["speech_tensors"]
    if not isinstance(tensors, torch.Tensor) or tensors.numel() == 0 or not bool(torch.isfinite(tensors).all()):
        raise RuntimeError("official processor speech_tensors is invalid")
    payload = tensors.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return prepared, {
        "input_ids_shape": list(prepared["input_ids"].shape),
        "speech_tensors_shape": list(tensors.shape),
        "speech_tensors_dtype": str(tensors.dtype),
        "speech_tensors_sha256": hashlib.sha256(payload).hexdigest(),
        "speech_masks_shape": list(prepared["speech_masks"].shape),
        "speech_input_mask_shape": list(prepared["speech_input_mask"].shape),
    }


def blocked(output: Path, error: Exception, *, packet_sha256: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "reference_status": "REFERENCE_ERROR",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": HF_REPOSITORY, "requested_revision": HF_REVISION, "resolved_revision": None},
        "qwen_companion": {"repository": QWEN_REPOSITORY, "requested_revision": QWEN_REVISION, "resolved_revision": None},
        "source": {"repository": SOURCE_REPOSITORY, "requested_revision": SOURCE_REVISION, "resolved_revision": None},
        "error_type": type(error).__name__,
        "reason": str(error),
        "blockers": [str(error)],
    }
    if packet_sha256 is not None:
        manifest["input_packet_sha256"] = packet_sha256
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_reference(source: Path, snapshot: Path, qwen: Path, packet_path: Path, output: Path) -> int:
    environment = require_license_clearance()
    identity = source_identity(source)
    packet_bytes = packet_path.read_bytes()
    packet = validate_packet(strict_json(packet_bytes, "packet"))
    if not snapshot.is_dir() or not qwen.is_dir():
        raise RuntimeError("fixed HF and Qwen snapshots are required")
    # Import only the pinned upstream file.  No implementation is mirrored in
    # this module; absence of the expected official entry point is a blocker.
    role = source / SOURCE_ROLE
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("vibevoice.modular.modeling_vibevoice_inference")
    except Exception as import_error:
        spec = importlib.util.spec_from_file_location("vibevoice_official_inference", role)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load pinned upstream inference role") from import_error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    # This is the inference subclass used by the orphan release.  Do not fall
    # back to the base class: it has different generation/output semantics and
    # would make the resulting packet look authoritative while bypassing the
    # fixed processor path.
    cls = getattr(module, "VibeVoiceForConditionalGenerationInference", None)
    if cls is None:
        raise RuntimeError(
            "pinned inference role exposes no "
            "VibeVoiceForConditionalGenerationInference"
        )
    # The exact source route is intentionally called only when the upstream
    # object provides the expected generation API.  Any API drift fails closed
    # rather than using a locally-authored fallback.
    generate = getattr(cls, "generate", None)
    from_pretrained = getattr(cls, "from_pretrained", None)
    if generate is None or from_pretrained is None:
        raise RuntimeError("official VibeVoice generation API is unavailable")
    import torch

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(qwen), local_files_only=True, trust_remote_code=False
        )
        if len(tokenizer) != 151_936:
            raise RuntimeError("fixed Qwen tokenizer vocabulary size drift")
        special_ids = {
            "speech_start_id": ("<|vision_start|>", 151_652),
            "speech_end_id": ("<|vision_end|>", 151_653),
            "speech_diffusion_id": ("<|vision_pad|>", 151_654),
        }
        for name, (token, expected) in special_ids.items():
            if tokenizer.convert_tokens_to_ids(token) != expected:
                raise RuntimeError(f"fixed Qwen {name} drift")
        for token_id in packet["token_ids"]:
            if token_id >= len(tokenizer):
                raise RuntimeError("packet token id is outside the fixed Qwen vocabulary")
    except Exception as error:
        raise RuntimeError("fixed Qwen tokenizer could not authenticate packet ids") from error
    prepared, prepared_evidence = prepared_inputs(source, snapshot, tokenizer, packet, torch)
    model = from_pretrained(str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True)
    model.eval()
    validate_generate_signature(inspect.signature(model.generate))
    official_source = (source / SOURCE_ROLE).read_text(encoding="utf-8", errors="strict")
    if "multinomial" in official_source:
        raise RuntimeError("official token generation contains an uncontrolled multinomial path")
    # The upstream demo passes the processor mapping verbatim.  Do not add
    # both ``inputs`` and ``input_ids``: Transformers rejects that duplicate
    # source in _prepare_model_inputs.  is_prefill is internal upstream state.
    kwargs = generation_call_kwargs(prepared, tokenizer, packet)
    kwargs["do_sample"] = False
    # Consume the caller packet at the actual upstream random call sites.
    # A seeded Generator alone is insufficient evidence: it would ignore the
    # recorded prompt and diffusion draws.  Unexpected call shapes and unused
    # draws fail closed, so this adapter cannot silently become a mirror.
    draw_records = packet["random_draws"]
    draw_index = 0
    consumed_draws: list[dict[str, Any]] = []
    consumed_values: list[list[float]] = []

    def upstream_callsite() -> dict[str, Any]:
        frame = sys._getframe(2)
        while frame is not None:
            filename = Path(frame.f_code.co_filename).resolve()
            try:
                relative = filename.relative_to(source.resolve()).as_posix()
            except ValueError:
                frame = frame.f_back
                continue
            if not relative.startswith("vibevoice/"):
                raise RuntimeError("random draw originated outside pinned VibeVoice source")
            return {
                "path": relative,
                "function": frame.f_code.co_name,
                "line": frame.f_lineno,
            }
        raise RuntimeError("random draw had no pinned upstream source callsite")
    original_randn, original_randn_like = torch.randn, torch.randn_like
    official_diffusion_latents: list[Any] = []
    official_diffusion_latent_shapes: list[list[int]] = []
    original_scheduler_step = getattr(model.model.noise_scheduler, "step", None)
    if original_scheduler_step is None:
        raise RuntimeError("pinned official scheduler has no step method")

    def capture_scheduler_step(*args: Any, **call_kwargs: Any) -> Any:
        result = original_scheduler_step(*args, **call_kwargs)
        latent = getattr(result, "prev_sample", None)
        if isinstance(latent, torch.Tensor):
            official_diffusion_latents.append(latent.detach().cpu())
            official_diffusion_latent_shapes.append([int(value) for value in latent.shape])
        return result

    model.model.noise_scheduler.step = capture_scheduler_step

    def take(shape: tuple[int, ...], kwargs: dict[str, Any], actual_dtype: Any) -> Any:
        nonlocal draw_index
        count = 1
        for dimension in shape:
            count *= dimension
        if draw_index >= len(draw_records):
            raise RuntimeError(f"official random draw is not covered: {shape}")
        record = draw_records[draw_index]
        expected_shape = tuple(record["shape"])
        if shape != expected_shape or count != len(record["values"]):
            raise RuntimeError(f"official random draw {record['name']} shape mismatch: {shape} != {expected_shape}")
        expected_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[record["dtype"]]
        if actual_dtype != expected_dtype:
            raise RuntimeError(f"official random draw {record['name']} dtype mismatch")
        draw_index += 1
        callsite = upstream_callsite()
        if record["name"] == "prompt_latent" and not (
            callsite["path"].endswith("modular_vibevoice_tokenizer.py")
            and callsite["function"] == "sample"
        ):
            raise RuntimeError("prompt latent draw did not originate at the official tokenizer sample")
        if record["name"] == "diffusion_initial" and not (
            callsite["path"] == SOURCE_ROLE
            and callsite["function"] == "sample_speech_tokens"
        ):
            raise RuntimeError("diffusion draw did not originate at official sample_speech_tokens")
        consumed_draws.append(
            {
                "name": record["name"],
                "shape": list(shape),
                "dtype": record["dtype"],
                "callsite": callsite,
            }
        )
        result = torch.tensor(record["values"], dtype=expected_dtype).reshape(shape)
        device = kwargs.get("device")
        effective = result.detach().to(torch.float32).reshape(-1).cpu().tolist()
        consumed_values.append([float(value) for value in effective])
        return result.to(device=device) if device is not None else result

    def patched_randn(*shape: Any, **call_kwargs: Any) -> Any:
        if not shape and "size" in call_kwargs:
            shape_value = call_kwargs.pop("size")
            dimensions = tuple(int(x) for x in shape_value) if isinstance(shape_value, (tuple, list)) else (int(shape_value),)
        elif shape and isinstance(shape[0], (tuple, list)):
            dimensions = tuple(int(x) for x in shape[0])
        else:
            dimensions = tuple(int(x) for x in shape)
        return take(dimensions, call_kwargs, call_kwargs.get("dtype", torch.float32))

    def patched_randn_like(input_tensor: Any, **call_kwargs: Any) -> Any:
        return take(tuple(int(x) for x in input_tensor.shape), call_kwargs | {"device": input_tensor.device}, call_kwargs.get("dtype", input_tensor.dtype))

    torch.randn, torch.randn_like = patched_randn, patched_randn_like
    try:
        with torch.inference_mode():
            result = model.generate(**kwargs)
    finally:
        torch.randn, torch.randn_like = original_randn, original_randn_like
        model.model.noise_scheduler.step = original_scheduler_step
    if draw_index != len(draw_records):
        raise RuntimeError(f"official generation consumed {draw_index} of {len(draw_records)} caller draw records")
    sequences = getattr(result, "sequences", result)
    if (
        not isinstance(sequences, torch.Tensor)
        or sequences.numel() == 0
        or sequences.dtype not in (torch.int32, torch.int64)
        or sequences.ndim != 2
        or sequences.shape[0] != 1
    ):
        raise RuntimeError("official generation returned no integer token sequence")
    if not hasattr(result, "reach_max_step_sample"):
        raise RuntimeError("official result lacks reach_max_step_sample")
    sequence = sequences.detach().cpu().reshape(-1).to(torch.int64).tolist()
    prompt_length = len(packet["token_ids"])
    generated = generated_segment(sequence, prompt_length, packet["max_generated_tokens"])
    if any(not isinstance(token, int) or token < 0 or token >= 151_936 for token in generated):
        raise RuntimeError("official generated sequence has unsafe token ids")
    speech_outputs = getattr(result, "speech_outputs", None)
    if not isinstance(speech_outputs, list) or not speech_outputs or not isinstance(speech_outputs[0], torch.Tensor):
        raise RuntimeError("official generation returned no speech_outputs PCM")
    official_pcm = speech_outputs[0].detach().cpu().reshape(-1).to(torch.float32)
    if official_pcm.numel() == 0 or not bool(torch.isfinite(official_pcm).all()):
        raise RuntimeError("official speech_outputs PCM is empty or non-finite")
    output.mkdir(parents=True, exist_ok=True)
    (output / "token_ids.u32le").write_bytes(b"".join(int(x).to_bytes(4, "little") for x in packet["token_ids"]))
    import struct
    (output / "prompt_pcm.f32le").write_bytes(b"".join(struct.pack("<f", float(x)) for x in packet["prompt_pcm"]))
    draw_values: dict[str, list[float]] = {}
    native_diffusion_values: list[float] = []
    if len(consumed_values) != len(packet["random_draws"]):
        raise RuntimeError("official random draw records are incomplete")
    for record, values in zip(packet["random_draws"], consumed_values):
        draw_values.setdefault(record["name"], []).extend(values)
        if record["name"] == "diffusion_initial":
            shape = tuple(record["shape"])
            if len(shape) != 2 or shape[1] != 64 or shape[0] < 1:
                raise RuntimeError("official diffusion draw cannot be mapped to native batch-one draw")
            native_diffusion_values.extend(values[: shape[1]])
    for name, values in draw_values.items():
        (output / f"{name}.f32le").write_bytes(b"".join(struct.pack("<f", x) for x in values))
    generated_diffusion_count = diffusion_token_count(generated)
    if len(native_diffusion_values) != generated_diffusion_count * 64:
        raise RuntimeError("official diffusion draw count does not match generated speech tokens")
    (output / "diffusion_initial_native.f32le").write_bytes(
        b"".join(struct.pack("<f", x) for x in native_diffusion_values)
    )
    (output / "speech_input_mask.u8").write_bytes(bytes(packet["speech_input_mask"]))
    (output / "speech_masks.u8").write_bytes(bytes(packet["speech_masks"]))
    (output / "speech_replacement_positions.u32le").write_bytes(b"".join(int(x).to_bytes(4, "little") for x in packet["speech_replacement_positions"]))
    (output / "generated_tokens.u32le").write_bytes(b"".join(int(x).to_bytes(4, "little") for x in generated))
    (output / "official_pcm.f32le").write_bytes(b"".join(struct.pack("<f", float(x)) for x in official_pcm.tolist()))
    if official_diffusion_latents:
        latent_values = [float(x) for tensor in official_diffusion_latents for x in tensor.reshape(-1).tolist()]
        (output / "official_diffusion_latents.f32le").write_bytes(b"".join(struct.pack("<f", x) for x in latent_values))
    (output / "packet.json").write_bytes(packet_bytes)
    (output / "guidance-scale.txt").write_text(f"{float(packet['guidance_scale']):.9g}\n", encoding="ascii")
    (output / "max-generated-tokens.txt").write_text(f"{packet['max_generated_tokens']}\n", encoding="ascii")
    manifest = {
        "format": FORMAT,
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "reference_status": "REFERENCE_EVIDENCE_COMPLETE",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "cpu_status": "UNSUPPORTED",
        "metal_status": "BLOCKED_BY_CPU",
        "parity_status": "NOT_RUN",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": HF_REPOSITORY, "revision": HF_REVISION},
        "qwen_companion": {"repository": QWEN_REPOSITORY, "revision": QWEN_REVISION},
        "prepared_inputs": prepared_evidence,
        "reference_environment": environment,
        "source": {**identity, "model_role": SOURCE_MODEL_ROLE},
        "input_packet_sha256": sha256(packet_path),
        "seed": packet["seed"],
        "generated_tokens": {
            "count": len(generated),
            "prompt_length": prompt_length,
            "max_generated_tokens": packet["max_generated_tokens"],
            "shape": [1, len(generated)],
        },
        "reach_max_step_sample": bool(result.reach_max_step_sample),
        "official_pcm": {"sample_rate_hz": 24_000, "samples": int(official_pcm.numel()), "sha256": sha256(output / "official_pcm.f32le")},
        "diffusion_latents": {
            "steps": len(official_diffusion_latents),
            "captured": bool(official_diffusion_latents),
            "shapes": official_diffusion_latent_shapes,
        },
        "random_draws_consumed": consumed_draws,
        "diffusion_native_mapping": {
            "source_batch_row": 0,
            "width": 64,
            "draws": len(native_diffusion_values) // 64,
            "generated_diffusion_tokens": generated_diffusion_count,
        },
        "taps": ["official_generate.sequences", "official_generate.speech_outputs", "official_noise_scheduler.step.prev_sample"],
        "blockers": ["native Vokra CPU/Metal parity remains a separate staged test; no numeric PCM gate was registered"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("vibevoice reference completed through official generate; PCM parity remains MEASURED_NOT_GATED")
    return 0


def self_test() -> None:
    assert len(HF_REVISION) == len(SOURCE_REVISION) == len(QWEN_REVISION) == 40
    lock = reference_lock_identity()
    assert lock["python"] == "==3.12.*"
    assert lock["cpu_index"] == REFERENCE_CPU_INDEX
    assert lock["package_count"] == REFERENCE_PACKAGE_COUNT
    assert lock["package_rows_sha256"] == REFERENCE_PACKAGE_ROWS_SHA256
    assert lock["package_rows_schema"] == REFERENCE_PACKAGE_ROWS_SCHEMA
    assert len(lock["package_rows"]) == REFERENCE_PACKAGE_COUNT
    assert "diffusers" in lock["package_names"]
    assert "torch" in lock["package_names"] and "transformers" in lock["package_names"]
    assert "soundfile" not in lock["package_names"]
    assert "soxr" not in lock["package_names"]
    audit = license_audit_identity(lock)
    assert audit["status"] == REFERENCE_LICENSE_STATUS
    assert len(audit["license_conclusions"]) == REFERENCE_LICENSE_COUNT
    assert any("certifi" in blocker for blocker in audit["blockers"])
    assert any("typing-extensions" in blocker for blocker in audit["blockers"])
    tampered_rows = [dict(row) for row in lock["package_rows"]]
    tampered_rows[0] = {**tampered_rows[0], "source": {"registry": "https://example.invalid/simple"}}
    assert lock_rows_sha256(tampered_rows) != REFERENCE_PACKAGE_ROWS_SHA256
    dependency_index = next(index for index, row in enumerate(lock["package_rows"]) if row["dependencies"])
    tampered_dependencies = [dict(dependency) for dependency in lock["package_rows"][dependency_index]["dependencies"]]
    tampered_dependencies[0] = {**tampered_dependencies[0], "marker": "sys_platform == 'ios'"}
    tampered_dependency_rows = [dict(row) for row in lock["package_rows"]]
    tampered_dependency_rows[dependency_index] = {**tampered_dependency_rows[dependency_index], "dependencies": tampered_dependencies}
    assert lock_rows_sha256(tampered_dependency_rows) != REFERENCE_PACKAGE_ROWS_SHA256
    source_index = next(index for index, row in enumerate(lock["package_rows"]) if any("source" in dependency for dependency in row["dependencies"]))
    source_dependencies = [dict(dependency) for dependency in lock["package_rows"][source_index]["dependencies"]]
    source_dependency_index = next(index for index, dependency in enumerate(source_dependencies) if "source" in dependency)
    source_dependencies[source_dependency_index] = {**source_dependencies[source_dependency_index], "source": {"registry": "https://example.invalid/simple"}}
    tampered_source_rows = [dict(row) for row in lock["package_rows"]]
    tampered_source_rows[source_index] = {**tampered_source_rows[source_index], "dependencies": source_dependencies}
    assert lock_rows_sha256(tampered_source_rows) != REFERENCE_PACKAGE_ROWS_SHA256
    tampered_license_rows = [dict(row) for row in audit["license_conclusions"]]
    tampered_license_rows[0] = {**tampered_license_rows[0], "route": "redistributed"}
    assert license_rows_sha256(tampered_license_rows) != REFERENCE_LICENSE_ROWS_SHA256
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-lock-tamper-") as directory:
        tampered_lock = Path(directory) / "uv.lock"
        tampered_lock.write_bytes(REFERENCE_LOCK.read_bytes() + b"\n# tampered\n")
        try:
            reference_lock_identity(tampered_lock)
        except RuntimeError:
            pass
        else:
            raise AssertionError("accepted a tampered dedicated lock")
    base = {
        "token_ids": [151_652], "processor_text": "fixture",
        "speech_replacement_positions": [],
        "speech_input_mask": [0], "speech_masks": [0], "prompt_pcm": [0.0] * 3_200,
        "prompt_sample_rate_hz": 24_000,
        "random_draws": [
            {"name": "prompt_latent", "shape": [1, 64], "dtype": "float32", "values": [0.0] * 64},
            {"name": "diffusion_initial", "shape": [1, 64], "dtype": "float32", "values": [0.0] * 64},
        ],
        "guidance_scale": 1.0, "max_generated_tokens": 1, "seed": 7,
    }
    assert validate_packet(base)["token_ids"] == [151_652]
    call = processor_call_kwargs(base)
    assert set(call) == {"text", "voice_samples", "padding", "return_tensors", "return_attention_mask"}
    assert "audio" not in call and call["text"] == ["fixture"]
    assert call["voice_samples"][0][0].dtype.name == "float32"
    def official_generate_fixture(inputs, speech_tensors, speech_masks, speech_input_mask, cfg_scale=1.0, return_speech=False, **kwargs):
        return inputs, kwargs
    validate_generate_signature(inspect.signature(official_generate_fixture))
    try:
        validate_generate_signature(inspect.signature(lambda input_ids, speech_tensors, speech_masks, speech_input_mask, cfg_scale, return_speech: None))
    except RuntimeError:
        pass
    else:
        raise AssertionError("signature without official inputs/**kwargs boundary accepted")
    generation = generation_call_kwargs({"input_ids": "fixture", "speech_tensors": "x"}, object(), base)
    assert "is_prefill" not in generation and "inputs" not in generation and generation["do_sample"] is False
    try:
        generation_call_kwargs({"inputs": "x", "input_ids": "y"}, object(), base)
    except RuntimeError:
        pass
    else:
        raise AssertionError("conflicting generate input aliases accepted")
    assert generated_segment([151_652, 151_653], 1, 4) == [151_653]
    assert generated_segment([151_652], 1, 4) == []
    assert diffusion_token_count([151_652, 151_654, 151_653, 151_643]) == 1
    try:
        generated_segment([151_652, 1, 2], 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted generated-token cap drift")
    for key in ("token_ids", "prompt_pcm"):
        bad = dict(base); bad[key] = []
        try:
            validate_packet(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted malformed {key}")
    bad = dict(base); bad["prompt_pcm"] = [0.0] * 3_201
    try:
        validate_packet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted non-3200-aligned prompt PCM")
    bad = dict(base); bad["prompt_sample_rate_hz"] = 16_000
    try:
        validate_packet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted sample-rate drift")
    bad = dict(base); bad["speech_masks"] = [1]
    try:
        validate_packet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted latent-mask mismatch")
    bad = dict(base); bad["speech_input_mask"] = [1]
    try:
        validate_packet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted replacement-mask mismatch")
    bad = dict(base); bad["random_draws"] = [dict(base["random_draws"][0], dtype="int8"), *base["random_draws"][1:]]
    try:
        validate_packet(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted unsupported draw dtype")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="vokra-vibevoice-ref-") as directory:
        out = Path(directory) / "evidence"
        blocked(out, RuntimeError("fixture"), packet_sha256="0" * 64)
        manifest = strict_json((out / "manifest.json").read_bytes(), "manifest")
        assert manifest["status"] == "BLOCKED" and manifest["publication"] == "NO_UPLOAD"
        assert manifest["input_packet_sha256"] == "0" * 64
    print("vibevoice_1_5b_dump_reference.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--qwen-snapshot", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--license-audit", action="store_true")
    args = parser.parse_args()
    if args.license_audit:
        if args.self_test or any(value is not None for value in (args.source, args.snapshot, args.qwen_snapshot, args.packet, args.output)):
            parser.error("--license-audit accepts no paths or --self-test")
        try:
            environment = reference_environment_identity()
            print(json.dumps({"reference_environment": environment}, sort_keys=True))
            if environment["license_audit"]["status"] != "AUTHENTICATED_CLEAR":
                print("VibeVoice dependency license audit BLOCKED; no sync, acquisition, or reference execution is permitted", file=sys.stderr)
                return 2
            return 0
        except Exception as error:
            print(f"VibeVoice dependency license audit ERROR: {error}", file=sys.stderr)
            return 2
    if args.self_test:
        if any(value is not None for value in (args.source, args.snapshot, args.qwen_snapshot, args.packet, args.output)):
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if any(value is None for value in (args.source, args.snapshot, args.qwen_snapshot, args.packet, args.output)):
        parser.error("all reference paths are required")
    try:
        environment = reference_environment_identity()
        if environment["license_audit"]["status"] != "AUTHENTICATED_CLEAR":
            raise RuntimeError("dedicated VibeVoice dependency license audit is unresolved")
    except Exception as error:
        print(f"VibeVoice reference BLOCKED before output creation: {error}", file=sys.stderr)
        return 2
    packet_sha = None
    try:
        packet_sha = sha256(args.packet)
        return run_reference(args.source, args.snapshot, args.qwen_snapshot, args.packet, args.output)
    except Exception as error:
        blocked(args.output, error, packet_sha256=packet_sha)
        print(f"VibeVoice reference BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
