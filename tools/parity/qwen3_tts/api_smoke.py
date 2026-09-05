#!/usr/bin/env python3
"""Authenticated, local-only Qwen3-TTS Transformers API smoke.

This module intentionally does not contain a model implementation.  The
production path imports the official QwenLM source checkout and calls its
``Qwen3TTSModel`` wrapper with a fixed local 0.6B-Base snapshot.  ``--self-test``
and ``--validate-evidence`` are stdlib-only and never import torch, acquire a
snapshot, or access the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

SCHEMA = "vokra-qwen3-tts-api-smoke-v1"
SOURCE_REPOSITORY = "QwenLM/Qwen3-TTS"
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SOURCE_PACKAGE_VERSION = "0.1.1"
MODEL_REPOSITORY = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"
MODEL_CONFIG_BYTES = 4494
MODEL_CONFIG_SHA256 = "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011"
DECODER_REPOSITORY = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DECODER_REVISION = "a87c50897bb00837eb857d0538b29d117541d7f6"
DECODER_CHECKPOINT_SHA256 = "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
LOCK_SHA256 = "b5fd403808a15759c5b10331e4da759ad230847baa833e75abba36d53a3cfdd2"
TRANSFORMERS_VERSION = "5.10.4"
TEXT = "The Vokra API smoke packet is short and deterministic."
LANGUAGE = "English"
MAX_NEW_TOKENS = 2
MIN_NEW_TOKENS = 2
OUTPUT_SAMPLE_RATE = 24_000
CODEBOOKS = 16
REFERENCE_AUDIO_SHA256 = "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"

EVIDENCE_KEYS = {
    "schema", "status", "publication", "source", "model", "decoder",
    "lock", "approval", "vokra_checkout", "package_versions",
    "environment", "inputs", "call_checkpoints", "api", "error",
}
CHECKPOINTS = (
    "execution_host_verified",
    "vokra_checkout_verified",
    "approval_evidence_recorded",
    "source_revision_verified",
    "model_snapshot_verified",
    "decoder_snapshot_verified",
    "lock_verified",
    "official_imports_verified",
    "model_loaded_cpu",
    "official_wrapper_called",
    "official_decoder_completed",
    "output_shape_verified",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SmokeError(RuntimeError):
    """A fail-closed precondition or API smoke failure."""


def require_execution_host() -> None:
    """Enforce the VAST identity in the API worker as well as its shell."""
    if __import__("os").environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise SmokeError("VOKRA_PUBLISH_ON_VAST=1 is absent")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SmokeError("API smoke requires VAST Linux x86_64")


def require_execution_host_values(publish: str, system: str, machine: str) -> None:
    """Self-test seam for the host gate; production never supplies overrides."""
    if publish != "1" or system != "Linux" or machine != "x86_64":
        raise SmokeError("API smoke requires VAST Linux x86_64")


def reject_symlink_ancestry(path: Path) -> None:
    absolute = Path(path)
    if not absolute.is_absolute():
        raise SmokeError(f"path must be absolute: {path}")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink() and current != Path("/var"):
            raise SmokeError(f"path has symlink ancestry: {path}")


def canonical_path(path: Path) -> Path:
    reject_symlink_ancestry(path)
    return path.resolve(strict=False)


def paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = canonical_path(first)
    second_resolved = canonical_path(second)
    return first_resolved == second_resolved or first_resolved in second_resolved.parents or second_resolved in first_resolved.parents


def require_output_isolated(output: Path, protected: list[Path]) -> None:
    if not output.is_absolute():
        raise SmokeError("evidence output must be an absolute path")
    reject_symlink_ancestry(output)
    if output.exists() or output.is_symlink():
        raise SmokeError(f"evidence output must be absent: {output}")
    output_parent = output.parent
    if not output_parent.is_dir():
        raise SmokeError("evidence output parent must already exist")
    for path in protected:
        if paths_overlap(output, path):
            raise SmokeError(f"evidence output overlaps protected path: {path}")


def require_vokra_checkout(root: Path) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or not (root / ".git").exists() or (root / ".git").is_symlink():
        raise SmokeError("Vokra checkout is missing, non-absolute, or symlinked")
    reject_symlink_ancestry(root)
    head = git_head(root)
    if not HEX40.fullmatch(head):
        raise SmokeError("Vokra checkout HEAD is not an immutable commit")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    ).stdout
    if status:
        raise SmokeError("Vokra checkout is dirty")
    return {"root": str(root.resolve()), "head": head, "clean": True}


def require_approval(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SmokeError("approval evidence must be an absolute regular non-symlink file")
    reject_symlink_ancestry(path)
    if path.stat().st_size <= 0:
        raise SmokeError("approval evidence is empty")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def require_input_boundaries(args: argparse.Namespace) -> None:
    root = canonical_path(args.vokra_root)
    named = {
        "source": args.source_dir,
        "model": args.model_dir,
        "decoder": args.decoder_dir,
        "reference": args.reference_audio,
        "lock": args.lock,
        "approval": args.approval_evidence,
        "vokra": args.vokra_root,
    }
    canonical = {name: canonical_path(path) for name, path in named.items()}
    for name in ("source", "model", "decoder"):
        if canonical[name] == root or root in canonical[name].parents:
            raise SmokeError(f"{name} must be outside the Vokra checkout")
    for left_index, left in enumerate(named):
        for right in list(named)[left_index + 1:]:
            if left == "vokra" and right in {"reference", "lock"}:
                continue
            if right == "vokra" and left in {"reference", "lock"}:
                continue
            if canonical[left] == canonical[right] or canonical[left] in canonical[right].parents or canonical[right] in canonical[left].parents:
                raise SmokeError(f"protected path boundaries overlap: {left} and {right}")
    if not (canonical["reference"] == root or root in canonical["reference"].parents):
        raise SmokeError("reference audio must be inside the Vokra checkout")
    if not (canonical["lock"] == root or root in canonical["lock"].parents):
        raise SmokeError("uv.lock must be inside the Vokra checkout")


def run_license_gate(args: argparse.Namespace, approval: dict[str, Any]) -> dict[str, Any]:
    gate_path = args.license_gate
    manifest_path = args.manifest
    project_path = args.project
    for path in (gate_path, manifest_path, project_path):
        if path.is_symlink() or not path.is_file():
            raise SmokeError(f"license gate input is missing or symlinked: {path}")
    reject_symlink_ancestry(gate_path)
    reject_symlink_ancestry(manifest_path)
    reject_symlink_ancestry(project_path)
    try:
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SmokeError(f"license gate manifest is unreadable: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("approval_scope_sha256"), str) or not HEX64.fullmatch(manifest["approval_scope_sha256"]):
        raise SmokeError("license gate manifest lacks an authenticated approval scope")
    try:
        spec = importlib.util.spec_from_file_location("vokra_qwen3_tts_license_gate", gate_path)
        if spec is None or spec.loader is None:
            raise SmokeError("license gate module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(
            args.lock, manifest_path, project_path, args.approval_evidence,
            SOURCE_REVISION, DECODER_REVISION, DECODER_CHECKPOINT_SHA256,
            {
                "0.6b-base": "5d83992436eae1d760afd27aff78a71d676296fc",
                "0.6b-customvoice": "85e237c12c027371202489a0ec509ded67b5e4b5",
                "1.7b-base": "fd4b254389122332181a7c3db7f27e918eec64e3",
                "1.7b-customvoice": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            },
        )
    except SystemExit as error:
        raise SmokeError(f"license gate rejected approval evidence (exit {error.code})") from error
    signoffs: list[dict[str, str]] = []
    for row in list(manifest.get("review_rows", [])) + list(manifest.get("component_rows", [])):
        if isinstance(row, dict) and isinstance(row.get("approval_signer"), str) and isinstance(row.get("approval_digest"), str):
            scope = str(row.get("name") or row.get("component") or "")
            signoffs.append({"scope": scope, "signer": row["approval_signer"], "digest": row["approval_digest"]})
    if not signoffs:
        raise SmokeError("license gate returned without authenticated owner sign-offs")
    return {
        **approval,
        "license_gate": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "approval_scope_sha256": manifest["approval_scope_sha256"],
        "owner_signoffs": sorted(signoffs, key=lambda row: row["scope"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_loads(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def git_head(source_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SmokeError(f"official source revision is unreadable: {error}") from error


def require_source(source_dir: Path) -> dict[str, Any]:
    if source_dir.is_symlink() or not source_dir.is_dir() or not (source_dir / ".git").is_dir():
        raise SmokeError("official source checkout is missing or symlinked")
    revision = git_head(source_dir)
    if revision != SOURCE_REVISION:
        raise SmokeError(f"official source revision {revision!r} != pinned {SOURCE_REVISION!r}")
    try:
        metadata = tomllib.loads((source_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SmokeError(f"official source metadata is unreadable: {error}") from error
    version = metadata.get("project", {}).get("version")
    if version != SOURCE_PACKAGE_VERSION:
        raise SmokeError(f"official source package version {version!r} != {SOURCE_PACKAGE_VERSION!r}")
    package = source_dir / "qwen_tts" / "__init__.py"
    if not package.is_file() or package.is_symlink():
        raise SmokeError("official qwen_tts package is missing or symlinked")
    if subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout:
        raise SmokeError("official source checkout is dirty")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
            "resolved_revision": revision, "package_version": version}


def require_lock(lock_path: Path) -> dict[str, Any]:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise SmokeError("uv.lock is missing or symlinked")
    actual = sha256_file(lock_path)
    if actual != LOCK_SHA256:
        raise SmokeError(f"uv.lock SHA-256 {actual} != pinned {LOCK_SHA256}")
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SmokeError(f"uv.lock is unreadable: {error}") from error
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise SmokeError("uv.lock package table is malformed")
    if any(isinstance(package, dict) and package.get("name") == "setuptools" for package in packages):
        raise SmokeError("uv.lock contains forbidden setuptools")
    if any(
        isinstance(package, dict)
        and any(isinstance(dependency, dict) and dependency.get("name") == "setuptools" for dependency in package.get("dependencies", []))
        for package in packages
    ):
        raise SmokeError("uv.lock contains forbidden setuptools dependency")
    versions: dict[str, set[str]] = {}
    for package in packages:
        if isinstance(package, dict) and isinstance(package.get("name"), str) and isinstance(package.get("version"), str):
            versions.setdefault(package["name"], set()).add(package["version"])
    return {"sha256": actual, "path": str(lock_path), "locked_versions": versions}


def artifact(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"{label} is missing or symlinked: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_model(model_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise SmokeError("0.6B model snapshot is missing or symlinked")
    config = model_dir / "config.json"
    config_artifact = artifact(config, "model config")
    if (config_artifact["bytes"], config_artifact["sha256"]) != (MODEL_CONFIG_BYTES, MODEL_CONFIG_SHA256):
        raise SmokeError("0.6B model config identity drifted")
    try:
        config_data = strict_json_loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SmokeError(f"model config is invalid: {error}") from error
    if not isinstance(config_data, dict) or config_data.get("model_type") != "qwen3_tts" or config_data.get("tts_model_type") != "base":
        raise SmokeError("model config is not the fixed Qwen3-TTS Base contract")
    if not (model_dir / "speech_tokenizer").is_dir() or (model_dir / "speech_tokenizer").is_symlink():
        raise SmokeError("nested official 12-Hz decoder directory is missing or symlinked")
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if not safetensors or any(path.is_symlink() for path in safetensors):
        raise SmokeError("0.6B model safetensors checkpoint is missing or symlinked")
    if (model_dir / "model.safetensors.index.json").exists() or list(model_dir.glob("model-*.safetensors")):
        raise SmokeError("sharded 0.6B checkpoint is not accepted by the API smoke")
    files = [config_artifact] + [artifact(path, "model checkpoint") for path in safetensors]
    return {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
            "config_bytes": MODEL_CONFIG_BYTES, "config_sha256": MODEL_CONFIG_SHA256,
            "files": files}, files


def require_decoder(model_dir: Path, decoder_dir: Path) -> dict[str, Any]:
    if decoder_dir.is_symlink() or not decoder_dir.is_dir():
        raise SmokeError("standalone decoder snapshot is missing or symlinked")
    standalone = decoder_dir / "model.safetensors"
    nested = model_dir / "speech_tokenizer" / "model.safetensors"
    standalone_info = artifact(standalone, "standalone decoder checkpoint")
    nested_info = artifact(nested, "nested decoder checkpoint")
    if standalone_info["sha256"] != DECODER_CHECKPOINT_SHA256 or nested_info["sha256"] != DECODER_CHECKPOINT_SHA256:
        raise SmokeError("official decoder checkpoint SHA-256 drifted")
    if standalone_info["sha256"] != nested_info["sha256"]:
        raise SmokeError("nested and standalone decoder checkpoints differ")
    return {"repository": DECODER_REPOSITORY, "revision": DECODER_REVISION,
            "checkpoint_sha256": DECODER_CHECKPOINT_SHA256,
            "standalone": standalone_info, "nested": nested_info}


def expected_package_versions(lock: dict[str, Any]) -> dict[str, str]:
    names = ("accelerate", "einops", "librosa", "numpy", "soundfile", "torch", "torchaudio", "transformers")
    result: dict[str, str] = {}
    for name in names:
        versions = lock["locked_versions"].get(name)
        if not versions or len(versions) != 1:
            raise SmokeError(f"uv.lock has no unique version for {name}")
        result[name] = next(iter(versions))
    if result["transformers"] != TRANSFORMERS_VERSION:
        raise SmokeError("uv.lock Transformers version drifted")
    return result


def validate_evidence_data(data: Any) -> None:
    if not isinstance(data, dict) or set(data) != EVIDENCE_KEYS:
        raise SmokeError("API smoke evidence schema or unknown fields drifted")
    if data["schema"] != SCHEMA or data["publication"] != "NO_UPLOAD":
        raise SmokeError("API smoke evidence identity/publication policy drifted")
    if data["status"] not in {"PASS", "FAIL"} or not isinstance(data["error"], (str, type(None))):
        raise SmokeError("API smoke evidence status/error is malformed")
    checkpoints = data["call_checkpoints"]
    if not isinstance(checkpoints, list) or any(item not in CHECKPOINTS for item in checkpoints) or len(set(checkpoints)) != len(checkpoints):
        raise SmokeError("API smoke checkpoints are malformed or duplicated")
    if data["status"] == "PASS" and checkpoints != list(CHECKPOINTS):
        raise SmokeError("passing API smoke evidence lacks an exact checkpoint sequence")
    if data["status"] == "PASS" and data["error"] is not None:
        raise SmokeError("passing API smoke evidence contains an error")
    for key in ("source", "model", "decoder", "lock", "approval", "vokra_checkout", "package_versions", "environment", "inputs", "api"):
        if not isinstance(data[key], dict):
            raise SmokeError(f"API smoke evidence field is not an object: {key}")
    if data["source"] and set(data["source"]) != {"repository", "revision", "resolved_revision", "package_version"}:
        raise SmokeError("source evidence schema or unknown fields drifted")
    if data["model"] and set(data["model"]) != {"repository", "revision", "config_bytes", "config_sha256", "files"}:
        raise SmokeError("model evidence schema or unknown fields drifted")
    if data["decoder"] and set(data["decoder"]) != {"repository", "revision", "checkpoint_sha256", "standalone", "nested"}:
        raise SmokeError("decoder evidence schema or unknown fields drifted")
    if set(data["lock"]) != {"sha256", "path"}:
        raise SmokeError("lock evidence schema or unknown fields drifted")
    if set(data["approval"]) != {"path", "sha256", "license_gate", "manifest_sha256", "approval_scope_sha256", "owner_signoffs"}:
        raise SmokeError("approval evidence schema or unknown fields drifted")
    if set(data["vokra_checkout"]) != {"root", "head", "clean"}:
        raise SmokeError("Vokra checkout evidence schema or unknown fields drifted")
    if set(data["environment"]) != {"python", "platform", "machine", "device", "torch_threads"}:
        raise SmokeError("environment evidence schema or unknown fields drifted")
    if set(data["api"]) != {"method", "local_files_only", "dtype", "device_map", "model_device", "wrapper", "max_new_tokens", "min_new_tokens", "sample_rate", "samples", "code_packet_frames", "code_packet_codebooks"}:
        raise SmokeError("API evidence schema or unknown fields drifted")
    if set(data["inputs"]) - {"reference_audio"}:
        raise SmokeError("input evidence schema or unknown fields drifted")
    for container in (data["package_versions"], data["inputs"]):
        if any(not isinstance(key, str) for key in container):
            raise SmokeError("evidence object key is malformed")
    if data["model"]:
        files = data["model"].get("files")
        if not isinstance(files, list) or any(not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} for row in files):
            raise SmokeError("model artifact evidence is malformed")
    if data["decoder"]:
        for name in ("standalone", "nested"):
            row = data["decoder"].get(name)
            if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
                raise SmokeError("decoder artifact evidence is malformed")
    def validate_artifact_row(row: Any) -> None:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise SmokeError("artifact evidence schema drifted")
        if not isinstance(row["path"], str) or not row["path"] or not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0 or not isinstance(row["sha256"], str) or not HEX64.fullmatch(row["sha256"]):
            raise SmokeError("artifact evidence identity is malformed")
    if data["source"]:
        if data["source"] != {
            "repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
            "resolved_revision": SOURCE_REVISION, "package_version": SOURCE_PACKAGE_VERSION,
        }:
            raise SmokeError("source identity in evidence drifted")
    if data["model"]:
        if data["model"].get("repository") != MODEL_REPOSITORY or data["model"].get("revision") != MODEL_REVISION or data["model"].get("config_bytes") != MODEL_CONFIG_BYTES or data["model"].get("config_sha256") != MODEL_CONFIG_SHA256:
            raise SmokeError("model identity in evidence drifted")
        for row in data["model"]["files"]:
            validate_artifact_row(row)
    if data["decoder"]:
        if data["decoder"].get("repository") != DECODER_REPOSITORY or data["decoder"].get("revision") != DECODER_REVISION or data["decoder"].get("checkpoint_sha256") != DECODER_CHECKPOINT_SHA256:
            raise SmokeError("decoder identity in evidence drifted")
        validate_artifact_row(data["decoder"]["standalone"])
        validate_artifact_row(data["decoder"]["nested"])
    if data["lock"].get("sha256") not in {None, LOCK_SHA256}:
        raise SmokeError("lock identity in evidence drifted")
    if data["lock"].get("sha256") is not None and not HEX64.fullmatch(data["lock"]["sha256"]):
        raise SmokeError("lock SHA-256 in evidence is malformed")
    if data["approval"]["sha256"] is not None and not HEX64.fullmatch(data["approval"]["sha256"]):
        raise SmokeError("approval SHA-256 in evidence is malformed")
    if data["approval"]["path"] is not None and (not isinstance(data["approval"]["path"], str) or not data["approval"]["path"]):
        raise SmokeError("approval path in evidence is malformed")
    if data["vokra_checkout"]["head"] is not None and not HEX40.fullmatch(data["vokra_checkout"]["head"]):
        raise SmokeError("Vokra checkout HEAD in evidence is malformed")
    if data["vokra_checkout"]["root"] is not None and (not isinstance(data["vokra_checkout"]["root"], str) or not data["vokra_checkout"]["root"]):
        raise SmokeError("Vokra checkout root in evidence is malformed")
    if data["approval"]["manifest_sha256"] is not None and not HEX64.fullmatch(data["approval"]["manifest_sha256"]):
        raise SmokeError("license manifest SHA-256 in evidence is malformed")
    if data["approval"]["approval_scope_sha256"] is not None and not HEX64.fullmatch(data["approval"]["approval_scope_sha256"]):
        raise SmokeError("license approval scope SHA-256 in evidence is malformed")
    if not isinstance(data["approval"]["owner_signoffs"], list) or any(
        not isinstance(row, dict) or set(row) != {"scope", "signer", "digest"}
        or not isinstance(row["scope"], str) or not isinstance(row["signer"], str) or not HEX40.fullmatch(row["signer"])
        or not isinstance(row["digest"], str) or not HEX64.fullmatch(row["digest"])
        for row in data["approval"]["owner_signoffs"]
    ):
        raise SmokeError("license owner sign-offs in evidence are malformed")
    if data["approval"]["license_gate"] not in {None, "PASS"}:
        raise SmokeError("license gate status in evidence is malformed")
    if data["vokra_checkout"]["clean"] not in {None, True}:
        raise SmokeError("Vokra checkout evidence is not clean")
    if data["inputs"]:
        reference = data["inputs"].get("reference_audio")
        validate_artifact_row(reference)
        if reference["sha256"] != REFERENCE_AUDIO_SHA256:
            raise SmokeError("reference audio identity in evidence drifted")
    if data["status"] == "PASS":
        if not data["source"] or not data["model"] or not data["decoder"] or data["lock"].get("sha256") != LOCK_SHA256 or data["approval"]["sha256"] is None or data["approval"]["license_gate"] != "PASS" or data["approval"]["manifest_sha256"] is None or data["approval"]["approval_scope_sha256"] is None or not data["approval"]["owner_signoffs"] or data["vokra_checkout"]["clean"] is not True or "reference_audio" not in data["inputs"]:
            raise SmokeError("passing evidence lacks authenticated identities")
        if not isinstance(data["approval"]["path"], str) or not isinstance(data["vokra_checkout"]["root"], str) or not isinstance(data["vokra_checkout"]["head"], str):
            raise SmokeError("passing evidence lacks approval/Vokra checkout paths")
        required_packages = {"accelerate", "einops", "librosa", "numpy", "soundfile", "torch", "torchaudio", "transformers", "qwen_tts_source"}
        if set(data["package_versions"]) != required_packages or any(not isinstance(value, str) or not value for value in data["package_versions"].values()):
            raise SmokeError("passing evidence package versions are incomplete")
        if data["api"]["method"] != "Qwen3TTSModel.from_pretrained" or data["api"]["local_files_only"] is not True or data["api"]["dtype"] != "float32" or data["api"]["device_map"] != "cpu" or data["api"]["wrapper"] != "generate_voice_clone":
            raise SmokeError("passing evidence does not prove the fixed local-only API call")
    if data["status"] == "FAIL" and not data["error"]:
        raise SmokeError("failed API smoke evidence lacks an error")


def write_evidence(path: Path, data: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeError(f"evidence output must be absent: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_smoke(args: argparse.Namespace) -> int:
    require_execution_host()
    require_input_boundaries(args)
    vokra_checkout = require_vokra_checkout(args.vokra_root)
    approval = require_approval(args.approval_evidence)
    require_output_isolated(
        args.output,
        [args.source_dir, args.model_dir, args.decoder_dir, args.reference_audio,
         args.lock, args.approval_evidence, args.vokra_root, args.project,
         args.manifest, args.license_gate],
    )
    approval = run_license_gate(args, approval)
    checkpoints: list[str] = []
    source: dict[str, Any] = {}
    model: dict[str, Any] = {}
    decoder: dict[str, Any] = {}
    lock: dict[str, Any] = {}
    package_versions: dict[str, str] = {}
    api: dict[str, Any] = {
        "method": "Qwen3TTSModel.from_pretrained",
        "local_files_only": True, "dtype": "float32", "device_map": "cpu",
        "model_device": None, "wrapper": "generate_voice_clone",
        "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS,
        "sample_rate": None, "samples": None, "code_packet_frames": None,
        "code_packet_codebooks": None,
    }
    inputs: dict[str, Any] = {}
    error: str | None = None
    checkpoints.extend(("execution_host_verified", "vokra_checkout_verified", "approval_evidence_recorded", "license_gate_verified"))
    try:
        source = require_source(args.source_dir)
        checkpoints.append("source_revision_verified")
        model, _ = require_model(args.model_dir)
        checkpoints.append("model_snapshot_verified")
        decoder = require_decoder(args.model_dir, args.decoder_dir)
        checkpoints.append("decoder_snapshot_verified")
        lock = require_lock(args.lock)
        package_versions = expected_package_versions(lock)
        checkpoints.append("lock_verified")
        inputs["reference_audio"] = artifact(args.reference_audio, "reference audio")
        if inputs["reference_audio"]["sha256"] != REFERENCE_AUDIO_SHA256:
            raise SmokeError("reference audio SHA-256 drifted")

        import numpy
        import torch
        import transformers
        import qwen_tts
        from qwen_tts import Qwen3TTSModel

        actual_versions = {
            name: importlib.metadata.version(name)
            for name in ("accelerate", "einops", "librosa", "numpy", "soundfile", "torch", "torchaudio", "transformers")
        }
        if actual_versions != package_versions:
            raise SmokeError(f"installed package versions differ from uv.lock: {actual_versions!r} != {package_versions!r}")
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise SmokeError("imported Transformers version is not 5.10.4")
        imported_root = Path(qwen_tts.__file__).resolve().parents[1]
        if imported_root != args.source_dir.resolve():
            raise SmokeError(f"qwen_tts imported from {imported_root}, not the authenticated source")
        checkpoints.append("official_imports_verified")
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        torch.manual_seed(1234)
        numpy.random.seed(1234)
        tts = Qwen3TTSModel.from_pretrained(
            str(args.model_dir), local_files_only=True, dtype=torch.float32, device_map="cpu"
        )
        if getattr(tts, "device", None) is None or tts.device.type != "cpu":
            raise SmokeError(f"official model selected {getattr(tts, 'device', None)!r}, expected CPU")
        api["model_device"] = str(tts.device)
        checkpoints.append("model_loaded_cpu")

        prompt = tts.create_voice_clone_prompt(ref_audio=str(args.reference_audio), x_vector_only_mode=True)[0]
        captured: list[Any] = []
        decoder_model = tts.model.speech_tokenizer
        original_decode = decoder_model.decode

        def capture(packet: Any) -> Any:
            captured.append(packet[0]["audio_codes"].detach().cpu().clone())
            return original_decode(packet)

        decoder_model.decode = capture
        try:
            kwargs = tts._merge_generate_kwargs(
                do_sample=False, top_k=None, top_p=1.0, temperature=0.0,
                repetition_penalty=1.0, subtalker_dosample=False,
                subtalker_top_k=None, subtalker_top_p=1.0,
                subtalker_temperature=0.0, max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
            )
            wavs, sample_rate = tts.generate_voice_clone(
                TEXT, language=LANGUAGE, voice_clone_prompt=[prompt],
                non_streaming_mode=False, **kwargs
            )
        finally:
            decoder_model.decode = original_decode
        checkpoints.append("official_wrapper_called")
        if len(captured) != 1:
            raise SmokeError(f"official decoder hook captured {len(captured)} packets, expected one")
        codes = captured[0]
        if codes.ndim != 2 or codes.shape[1] != CODEBOOKS:
            raise SmokeError(f"official code packet shape={tuple(codes.shape)}, expected [frames,{CODEBOOKS}]")
        pcm = numpy.asarray(wavs[0], dtype=numpy.float32)
        if int(sample_rate) != OUTPUT_SAMPLE_RATE or pcm.size == 0 or not numpy.isfinite(pcm).all():
            raise SmokeError("official wrapper returned an invalid sample rate or PCM")
        checkpoints.append("official_decoder_completed")
        api.update({"sample_rate": int(sample_rate), "samples": int(pcm.size),
                    "code_packet_frames": int(codes.shape[0]),
                    "code_packet_codebooks": int(codes.shape[1])})
        checkpoints.append("output_shape_verified")
        package_versions = {**package_versions, "qwen_tts_source": SOURCE_PACKAGE_VERSION}
    except Exception as caught:  # evidence is retained even for a partial smoke
        error = f"{type(caught).__name__}: {caught}"

    data: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if error is None else "FAIL",
        "publication": "NO_UPLOAD",
        "source": source,
        "model": model,
        "decoder": decoder,
        "lock": {"sha256": lock.get("sha256"), "path": lock.get("path")},
        "approval": approval,
        "vokra_checkout": vokra_checkout,
        "package_versions": package_versions,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "device": "cpu", "torch_threads": 1},
        "inputs": inputs,
        "call_checkpoints": checkpoints,
        "api": api,
        "error": error,
    }
    validate_evidence_data(data)
    write_evidence(args.output, data)
    print(json.dumps(data, sort_keys=True), flush=True)
    return 0 if error is None else 2


def self_test() -> None:
    global LOCK_SHA256
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise SmokeError("self-test imported a model dependency")
    for values in (("0", "Linux", "x86_64"), ("1", "Darwin", "arm64"), ("1", "Linux", "aarch64")):
        try:
            require_execution_host_values(*values)
        except SmokeError:
            pass
        else:
            raise SmokeError(f"non-VAST host was accepted: {values!r}")
    require_execution_host_values("1", "Linux", "x86_64")
    if len(SOURCE_REVISION) != 40 or not HEX40.fullmatch(SOURCE_REVISION):
        raise SmokeError("source revision is not immutable")
    if len(MODEL_REVISION) != 40 or not HEX40.fullmatch(MODEL_REVISION):
        raise SmokeError("model revision is not immutable")
    if len(DECODER_REVISION) != 40 or not HEX40.fullmatch(DECODER_REVISION):
        raise SmokeError("decoder revision is not immutable")
    if not HEX64.fullmatch(LOCK_SHA256) or not HEX64.fullmatch(DECODER_CHECKPOINT_SHA256):
        raise SmokeError("fixed SHA-256 identity is malformed")
    with tempfile.TemporaryDirectory(prefix="qwen3-tts-api-smoke-self-test-") as directory:
        root = Path(directory)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"x"}', encoding="utf-8")
        try:
            strict_json_loads(duplicate.read_text(encoding="utf-8"))
        except ValueError:
            pass
        else:
            raise SmokeError("duplicate JSON keys were accepted")
        unknown = {key: None for key in EVIDENCE_KEYS} | {"unknown": True}
        try:
            validate_evidence_data(unknown)
        except SmokeError:
            pass
        else:
            raise SmokeError("unknown evidence field was accepted")
        failure = {
            "schema": SCHEMA, "status": "FAIL", "publication": "NO_UPLOAD",
            "source": {}, "model": {}, "decoder": {},
            "lock": {"sha256": None, "path": None},
            "approval": {"path": None, "sha256": None, "license_gate": None, "manifest_sha256": None, "approval_scope_sha256": None, "owner_signoffs": []},
            "vokra_checkout": {"root": None, "head": None, "clean": None},
            "package_versions": {},
            "environment": {"python": "3.12", "platform": "Linux", "machine": "x86_64", "device": "cpu", "torch_threads": 1},
            "inputs": {}, "call_checkpoints": [],
            "api": {
                "method": "Qwen3TTSModel.from_pretrained", "local_files_only": True,
                "dtype": "float32", "device_map": "cpu", "model_device": None,
                "wrapper": "generate_voice_clone", "max_new_tokens": 2,
                "min_new_tokens": 2, "sample_rate": None, "samples": None,
                "code_packet_frames": None, "code_packet_codebooks": None,
            },
            "error": "SmokeError: host gate",
        }
        validate_evidence_data(failure)
        missing = root / "missing-model"
        try:
            require_model(missing)
        except SmokeError:
            pass
        else:
            raise SmokeError("missing model checkpoint was accepted")
        bad_lock = root / "bad-uv.lock"
        bad_lock.write_text("version = 1\n", encoding="utf-8")
        try:
            require_lock(bad_lock)
        except SmokeError:
            pass
        else:
            raise SmokeError("lock SHA-256 drift was accepted")
        forbidden_lock = root / "forbidden-uv.lock"
        forbidden_lock.write_text(
            '[project]\nname = "test"\n\n[[package]]\nname = "setuptools"\nversion = "84.0.0"\n',
            encoding="utf-8",
        )
        original_lock_sha = LOCK_SHA256
        LOCK_SHA256 = sha256_file(forbidden_lock)
        try:
            require_lock(forbidden_lock)
        except SmokeError:
            pass
        else:
            raise SmokeError("setuptools package reintroduction was accepted")
        finally:
            LOCK_SHA256 = original_lock_sha
        forbidden_dependency_lock = root / "forbidden-dependency-uv.lock"
        forbidden_dependency_lock.write_text(
            '[project]\nname = "test"\n\n[[package]]\nname = "torch"\nversion = "2.7.1"\ndependencies = [{ name = "setuptools" }]\n',
            encoding="utf-8",
        )
        LOCK_SHA256 = sha256_file(forbidden_dependency_lock)
        try:
            require_lock(forbidden_dependency_lock)
        except SmokeError:
            pass
        else:
            raise SmokeError("setuptools dependency reintroduction was accepted")
        finally:
            LOCK_SHA256 = original_lock_sha
        protected = root / "protected"
        output_parent = root / "output"
        protected.mkdir()
        output_parent.mkdir()
        require_output_isolated(output_parent / "evidence.json", [protected])
        try:
            require_output_isolated(protected / "evidence.json", [protected])
        except SmokeError:
            pass
        else:
            raise SmokeError("output overlap with protected path was accepted")
        symlink_parent = root / "symlink-parent"
        symlink_parent.symlink_to(output_parent, target_is_directory=True)
        try:
            require_output_isolated(symlink_parent / "evidence.json", [protected])
        except SmokeError:
            pass
        else:
            raise SmokeError("symlink ancestry for output was accepted")
        existing = output_parent / "existing.json"
        existing.write_text("{}", encoding="utf-8")
        try:
            require_output_isolated(existing, [protected])
        except SmokeError:
            pass
        else:
            raise SmokeError("existing output was accepted")
        vokra_root = root / "vokra"
        vokra_root.mkdir()
        boundary_args = argparse.Namespace(
            source_dir=root / "source", model_dir=root / "model",
            decoder_dir=root / "decoder", reference_audio=vokra_root / "ref.wav",
            lock=vokra_root / "uv.lock", approval_evidence=root / "approval.json",
            vokra_root=vokra_root,
        )
        require_input_boundaries(boundary_args)
        boundary_args.model_dir = vokra_root / "model"
        try:
            require_input_boundaries(boundary_args)
        except SmokeError:
            pass
        else:
            raise SmokeError("model path inside Vokra checkout was accepted")
        boundary_args.model_dir = boundary_args.source_dir
        try:
            require_input_boundaries(boundary_args)
        except SmokeError:
            pass
        else:
            raise SmokeError("source/model path overlap was accepted")
        repository_root = Path(__file__).resolve().parents[3]
        gate_approval_path = root / "gate-approval.json"
        gate_approval_path.write_text("{}\n", encoding="utf-8")
        gate_args = argparse.Namespace(
            lock=repository_root / "tools/parity/qwen3_tts/uv.lock",
            project=repository_root / "tools/parity/qwen3_tts/pyproject.toml",
            manifest=repository_root / "tools/parity/qwen3_tts/license_gate_manifest.json",
            license_gate=repository_root / "tools/parity/qwen3_tts/license_gate.py",
            approval_evidence=gate_approval_path,
        )
        try:
            run_license_gate(gate_args, require_approval(gate_approval_path))
        except SmokeError:
            pass
        else:
            raise SmokeError("the unresolved license manifest was accepted")
        bad_source = root / "bad-source"
        bad_source.mkdir()
        (bad_source / ".git").mkdir()
        (bad_source / "pyproject.toml").write_text('[project]\nversion = "0.0.0"\n', encoding="utf-8")
        (bad_source / "qwen_tts").mkdir()
        (bad_source / "qwen_tts" / "__init__.py").write_text("", encoding="utf-8")
        try:
            require_source(bad_source)
        except SmokeError:
            pass
        else:
            raise SmokeError("source version/revision drift was accepted")
    print("qwen3_tts api smoke self-test: PASS")


def validate_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SmokeError("evidence file is missing or symlinked")
    validate_evidence_data(strict_json_loads(path.read_text(encoding="utf-8")))
    print("qwen3_tts api smoke evidence validation: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-evidence", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--decoder-dir", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--license-gate", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            if any(value is not None for value in (args.validate_evidence, args.source_dir, args.model_dir, args.decoder_dir, args.reference_audio, args.lock, args.project, args.manifest, args.license_gate, args.vokra_root, args.approval_evidence, args.output)):
                raise SmokeError("--self-test accepts no other arguments")
            self_test()
            return 0
        if args.validate_evidence is not None:
            if any(value is not None for value in (args.source_dir, args.model_dir, args.decoder_dir, args.reference_audio, args.lock, args.project, args.manifest, args.license_gate, args.vokra_root, args.approval_evidence, args.output)):
                raise SmokeError("--validate-evidence accepts no other arguments")
            validate_file(args.validate_evidence)
            return 0
        required = (args.source_dir, args.model_dir, args.decoder_dir, args.reference_audio, args.lock, args.project, args.manifest, args.license_gate, args.vokra_root, args.approval_evidence, args.output)
        if any(value is None for value in required):
            raise SmokeError("all production paths are required")
        return run_smoke(args)
    except SmokeError as error:
        print(f"qwen3_tts api smoke: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
