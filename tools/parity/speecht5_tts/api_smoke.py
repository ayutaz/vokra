#!/usr/bin/env python3
"""Authenticated Transformers SpeechT5 API smoke on VAST.

This module deliberately has a model-free ``--self-test`` path.  The real
path imports the pinned Transformers package only after checking the immutable
checkpoint, project bytes, and VAST host contract.  It calls the official
``SpeechT5ForTextToSpeech.generate_speech`` API with a deterministic speaker
embedding and writes a small, hashed mel output for handoff.  No Vokra code
is imported and there is no upload path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


UPSTREAM_HF = "microsoft/speecht5_tts"
UPSTREAM_REVISION = "30fcde30f19b87502b8435427b5f5068e401d5f6"
SOURCE_WEIGHT = "pytorch_model.bin"
SOURCE_WEIGHT_BYTES = 585_476_837
SOURCE_WEIGHT_SHA256 = "d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"
TOKENIZER_SHA256 = "7fcc48f3e225f627b1641db410ceb0c8649bd2b0c982e150b03f8be3728ab560"
EXPECTED_TRANSFORMERS = "5.10.4"
PREVIOUS_TRANSFORMERS = "transformers==5.5.0"
SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
SECURITY_FLOOR = "5.10.0"
VOCODER_REPO = "microsoft/speecht5_hifigan"
VOCODER_REVISION = "bb6f429406e86a9992357a972c0698b22043307d"
VOCODER_WEIGHT_SHA256 = "b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b"
SMOKE_TEXT = "Hi."
SMOKE_SEED = 0x5350_4545_4348_5435
SPEAKER_DIM = 512
LOCK_SHA256 = "418fb6b6516e0284b503ed20872e2dc6dd375aff918e253f3e7f9d27b62f904c"
PYPROJECT_SHA256 = "1e61ad26749c1ad5ba05fe139ef8bfcf4698e3b030cad6182e18309789779346"
PASS_EVIDENCE_KEYS = {
    "call", "call_checkpoint_sha256", "checkpoint_files", "environment", "format",
    "frames", "input_sha256", "lock_sha256", "mel_bins", "output_count",
    "output_sha256", "package_rows_sha256", "package_sha256", "previous_isolated_transformers_pin",
    "project_sha256", "publication", "reference_implementation", "reference_package",
    "revision_sha256", "status", "transformers_security_advisory", "transformers_security_patched_minimum",
    "upstream_hf", "upstream_revision", "upload", "vocoder", "vokra_head", "vokra_root",
    "vokra_clean", "approval_evidence_sha256", "approval_scope_sha256", "approval_signer", "project_dir",
    "preflight_gate", "preflight_gate_sha256", "preflight_manifest_sha256",
}
FAIL_EVIDENCE_KEYS = {
    "approval_evidence_sha256", "approval_scope_sha256", "approval_signer", "error",
    "error_type", "format", "publication", "stage", "status", "upload", "vokra_clean",
    "vokra_head", "vokra_root",
    "preflight_gate", "preflight_gate_sha256", "preflight_manifest_sha256",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 drifted: {actual} != {expected}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def strict_json_loads(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing, non-regular, or symlinked: {path}")


def require_absolute_no_symlink_path(path: Path, label: str, *, exists: bool) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute: {path}")
    if any(component in {".", ".."} for component in path.parts[1:]):
        raise RuntimeError(f"{label} must not contain . or .. components: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"{label} has a symlinked ancestor: {current}")
    if exists and not path.exists():
        raise RuntimeError(f"{label} does not exist: {path}")
    if not exists and path.exists():
        raise RuntimeError(f"{label} must be absent: {path}")
    return path


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def require_disjoint_paths(named_paths: dict[str, Path]) -> None:
    names = list(named_paths)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            if paths_overlap(named_paths[left_name], named_paths[right_name]):
                raise RuntimeError(f"{left_name} overlaps {right_name}")


def git_checkout_context(vokra_root: Path) -> dict[str, Any]:
    require_absolute_no_symlink_path(vokra_root, "--vokra-root", exists=True)
    if not (vokra_root / ".git").is_dir() or not (vokra_root / "Cargo.toml").is_file():
        raise RuntimeError(f"--vokra-root is not a Vokra checkout: {vokra_root}")
    try:
        head = subprocess.run(
            ["git", "-C", str(vokra_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(vokra_root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Vokra checkout inspection failed: {type(error).__name__}") from error
    if not HEX40.fullmatch(head):
        raise RuntimeError("Vokra HEAD is not an exact commit")
    if status:
        raise RuntimeError("Vokra checkout is dirty")
    return {"vokra_root": str(vokra_root), "vokra_head": head, "vokra_clean": True}


def require_vast() -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("VOKRA_PUBLISH_ON_VAST=1 is absent")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "SpeechT5 API smoke is VAST/Linux x86_64-only; got "
            f"{platform.system()} {platform.machine()}"
        )


def verify_project(project_dir: Path) -> tuple[str, str, str]:
    require_absolute_no_symlink_path(project_dir, "parity project", exists=True)
    project = project_dir / "pyproject.toml"
    lock = project_dir / "uv.lock"
    require_regular(project, "pyproject.toml")
    require_regular(lock, "uv.lock")
    project_digest = sha256_file(project)
    lock_digest = sha256_file(lock)
    require_digest(project_digest, PYPROJECT_SHA256, "pyproject")
    require_digest(lock_digest, LOCK_SHA256, "uv.lock")
    lock_data = tomllib.loads(lock.read_text(encoding="utf-8"))
    packages = lock_data.get("package")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError("uv.lock package rows are missing")
    rows = []
    for package in packages:
        if not isinstance(package, dict):
            raise RuntimeError("uv.lock package row is malformed")
        rows.append(
            {
                "name": package.get("name"),
                "version": package.get("version"),
                "source": package.get("source"),
                "resolution-markers": package.get("resolution-markers", []),
                "dependencies": package.get("dependencies", []),
            }
        )
    rows.sort(key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True)))
    return project_digest, lock_digest, sha256_bytes(canonical(rows))


def installed_package_digest() -> tuple[str, dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", distribution.name).casefold()
        if name in packages:
            raise RuntimeError(f"duplicate installed distribution: {name}")
        packages[name] = distribution.version
    if not packages:
        raise RuntimeError("installed package inventory is empty")
    return sha256_bytes(canonical(packages)), dict(sorted(packages.items()))


def validate_approval_file(project_dir: Path, path: Path) -> dict[str, str]:
    require_regular(path, "approval evidence")
    manifest_path = project_dir / "license_gate_manifest.json"
    require_regular(manifest_path, "license gate manifest")
    try:
        approval = strict_json_loads(path.read_text(encoding="utf-8"))
        manifest = strict_json_loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"approval evidence is not strict JSON: {error}") from error
    if not isinstance(approval, dict) or not isinstance(manifest, dict):
        raise RuntimeError("preflight approval or manifest is not an object")
    required = {"decision", "scope_sha256", "manifest_sha256", "signer", "digest"}
    if set(approval) != required:
        raise RuntimeError("authenticated preflight approval schema is not exact")
    if approval.get("decision") != "APPROVED":
        raise RuntimeError("preflight approval decision is not APPROVED")
    signer = approval.get("signer")
    scope = approval.get("scope_sha256")
    if not isinstance(signer, str) or not signer.strip():
        raise RuntimeError("approval signer is missing")
    if not isinstance(scope, str) or not HEX64.fullmatch(scope):
        raise RuntimeError("approval scope_sha256 must be lowercase SHA-256")
    if approval.get("digest") != scope or manifest.get("approval_scope_sha256") != scope:
        raise RuntimeError("preflight approval scope/digest does not match manifest")
    if approval.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("preflight approval manifest digest drifted")
    return {"approval_evidence_sha256": sha256_file(path), "approval_scope_sha256": scope, "approval_signer": signer}


def run_preflight_gate(project_dir: Path, approval_path: Path) -> dict[str, str]:
    """Run the repository's exact license gate before any checkpoint read."""
    gate_path = project_dir / "preflight_gate.py"
    manifest_path = project_dir / "license_gate_manifest.json"
    require_regular(gate_path, "preflight gate")
    require_regular(manifest_path, "license gate manifest")
    spec = importlib.util.spec_from_file_location("vokra_speecht5_preflight_gate", gate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("existing preflight gate cannot be loaded")
    gate = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gate)
        passed, reason = gate.validate(project_dir, manifest_path, approval_path)
    except Exception as error:
        raise RuntimeError(f"existing preflight gate failed: {type(error).__name__}") from error
    if passed is not True:
        raise RuntimeError(f"existing preflight gate blocked: {reason}")
    return {
        "preflight_gate": "PASS",
        "preflight_gate_sha256": sha256_file(gate_path),
        "preflight_manifest_sha256": sha256_file(manifest_path),
    }


def validate_evidence_document(path: Path, status: str) -> dict[str, Any]:
    require_regular(path, "API smoke evidence")
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RuntimeError(f"API smoke evidence is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("API smoke evidence is not an object")
    expected = PASS_EVIDENCE_KEYS if status == "PASS" else FAIL_EVIDENCE_KEYS if status == "FAIL" else set()
    if not expected or set(value) != expected:
        raise RuntimeError(f"API smoke {status} evidence schema is not exact")
    if value.get("format") != "vokra-speecht5-api-smoke-v1" or value.get("status") != status or value.get("publication") != "NO_UPLOAD" or value.get("upload") != "NOT_PERFORMED":
        raise RuntimeError("API smoke evidence status/publication drifted")
    if value.get("vokra_clean") is not True or not isinstance(value.get("vokra_root"), str) or not Path(value["vokra_root"]).is_absolute() or not HEX40.fullmatch(str(value.get("vokra_head"))):
        raise RuntimeError("API smoke evidence lacks a clean exact Vokra HEAD")
    if value.get("preflight_gate") != "PASS":
        raise RuntimeError("API smoke evidence lacks a passing existing preflight gate")
    for key in ("approval_evidence_sha256", "approval_scope_sha256", "preflight_gate_sha256", "preflight_manifest_sha256"):
        if not HEX64.fullmatch(str(value.get(key))):
            raise RuntimeError(f"API smoke evidence has invalid {key}")
    if not isinstance(value.get("approval_signer"), str) or not value["approval_signer"].strip():
        raise RuntimeError("API smoke evidence lacks approval signer")
    if status == "PASS":
        if not isinstance(value.get("project_dir"), str) or not Path(value["project_dir"]).is_absolute():
            raise RuntimeError("API smoke evidence lacks an absolute parity project")
        for key in ("input_sha256", "output_sha256", "call_checkpoint_sha256", "project_sha256", "lock_sha256", "package_rows_sha256", "package_sha256"):
            if not HEX64.fullmatch(str(value.get(key))):
                raise RuntimeError(f"API smoke evidence has invalid {key}")
    else:
        if not isinstance(value.get("stage"), str) or not value["stage"] or not isinstance(value.get("error_type"), str) or not value["error_type"] or not isinstance(value.get("error"), str) or "\n" in value["error"]:
            raise RuntimeError("API smoke failure evidence lacks stage/error type")
    return value


def validate_evidence_output(output_dir: Path, status: str) -> dict[str, Any]:
    """Validate the evidence.json stored in a shell-provided output directory."""
    require_absolute_no_symlink_path(output_dir, "evidence directory", exists=True)
    if not output_dir.is_dir():
        raise RuntimeError(f"evidence directory is not a directory: {output_dir}")
    return validate_evidence_document(output_dir / "evidence.json", status)


def speaker_values() -> list[float]:
    return [float((index % 17) - 8) / 8.0 for index in range(SPEAKER_DIM)]


def verify_checkpoint(checkpoint: Path) -> dict[str, dict[str, Any]]:
    require_absolute_no_symlink_path(checkpoint, "checkpoint", exists=True)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint directory is missing or symlinked: {checkpoint}")
    expected = {
        SOURCE_WEIGHT: (SOURCE_WEIGHT_BYTES, SOURCE_WEIGHT_SHA256),
        "spm_char.model": (238_473, TOKENIZER_SHA256),
        "config.json": (2_062, "2caf62dde93699a90cfc35ff2a8de27b02b479a0c98881cbc55f9682cc43e258"),
        "tokenizer_config.json": (232, "d589430c619db2d95ff0fa757a187b55ef5ea44eff7fb08a6fbf0e78e32a6247"),
        "added_tokens.json": (40, "74be21ecff0a1fb1f304fe7c72ab21e4f0c046f8359fdf2852eb1b80967069ad"),
        "special_tokens_map.json": (234, "2a098b61fe8ec4cfd7674832ca00b4268c07569743a4ad15c8164e8f60ebf981"),
    }
    verified: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_hash) in expected.items():
        path = checkpoint / name
        require_regular(path, f"pinned checkpoint file {name}")
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned checkpoint identity drifted for {name}: "
                f"bytes={actual_bytes} sha256={actual_hash}"
            )
        verified[name] = {"bytes": actual_bytes, "sha256": actual_hash}
    return verified


def write_f32(path: Path, values: Any) -> tuple[int, str]:
    payload = values.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    path.write_bytes(payload)
    return len(payload) // 4, sha256_bytes(payload)


def validate_preflight(
    vokra_root: Path, project_dir: Path, checkpoint: Path, output_dir: Path, approval_path: Path
) -> dict[str, Any]:
    require_vast()
    root_context = git_checkout_context(vokra_root)
    require_absolute_no_symlink_path(project_dir, "parity project", exists=True)
    try:
        project_dir.relative_to(vokra_root)
    except ValueError as error:
        raise RuntimeError("parity project is outside --vokra-root") from error
    require_absolute_no_symlink_path(approval_path, "approval evidence", exists=True)
    require_regular(approval_path, "approval evidence")
    gate_context = run_preflight_gate(project_dir, approval_path)
    approval = validate_approval_file(project_dir, approval_path)
    require_absolute_no_symlink_path(checkpoint, "checkpoint", exists=True)
    require_absolute_no_symlink_path(output_dir, "output directory", exists=False)
    require_disjoint_paths({"checkpoint": checkpoint, "output": output_dir, "approval": approval_path})
    if paths_overlap(vokra_root, checkpoint) or paths_overlap(vokra_root, output_dir) or paths_overlap(vokra_root, approval_path):
        raise RuntimeError("checkpoint/output/approval overlaps --vokra-root")
    if paths_overlap(project_dir, checkpoint) or paths_overlap(project_dir, output_dir) or paths_overlap(project_dir, approval_path):
        raise RuntimeError("checkpoint/output/approval overlaps parity project")
    return {**root_context, **gate_context, **approval, "project_dir": str(project_dir)}


def safe_error_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    for secret in ("HF_TOKEN", "HF", "TOKEN", "Authorization", "Bearer"):
        message = message.replace(secret, "[REDACTED]")
    return message[:500] or type(error).__name__


def write_failure_evidence(output_dir: Path, context: dict[str, Any], stage: str, error: Exception) -> Path:
    evidence = {
        "format": "vokra-speecht5-api-smoke-v1",
        "status": "FAIL",
        "publication": "NO_UPLOAD",
        "upload": "NOT_PERFORMED",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": safe_error_message(error),
        **{key: context[key] for key in ("vokra_root", "vokra_head", "vokra_clean", "preflight_gate", "preflight_gate_sha256", "preflight_manifest_sha256", "approval_evidence_sha256", "approval_scope_sha256", "approval_signer")},
    }
    path = output_dir / "evidence.json"
    path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def run(checkpoint: Path, project_dir: Path, output_dir: Path, approval_path: Path, vokra_root: Path, text: str) -> int:
    context = validate_preflight(vokra_root, project_dir, checkpoint, output_dir, approval_path)
    if not text or text != text.strip():
        raise RuntimeError("smoke text must be non-empty and have no surrounding whitespace")
    output_dir.mkdir(parents=True, exist_ok=False)
    stage = "third_party_import"
    try:
        project_sha, lock_sha, package_rows_sha = verify_project(project_dir)
        checkpoint_files = verify_checkpoint(checkpoint)
        stage = "third_party_import"
        # Imports are intentionally kept inside the post-preflight block.
        import numpy as np
        import torch
        import transformers
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Tokenizer
        stage = "model_load"
        if transformers.__version__ != EXPECTED_TRANSFORMERS:
            raise RuntimeError(f"transformers {transformers.__version__} != {EXPECTED_TRANSFORMERS}")
        if not hasattr(transformers, "SpeechT5ForTextToSpeech"):
            raise RuntimeError("locked Transformers package does not expose SpeechT5ForTextToSpeech")

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SMOKE_SEED)
        tokenizer = SpeechT5Tokenizer.from_pretrained(checkpoint, local_files_only=True)
        encoded = tokenizer(text, return_tensors="pt", return_attention_mask=True)
        input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
        attention_mask = encoded["attention_mask"].to(dtype=torch.long, device="cpu")
        tokens = [int(token) for token in input_ids[0].tolist()]
        if not tokens or tokens[-1] != 2:
            raise RuntimeError(f"official tokenizer emitted an invalid sequence: {tokens}")
        speaker = torch.tensor(speaker_values(), dtype=torch.float32).unsqueeze(0)
        input_record = {
            "text": text,
            "text_utf8_sha256": sha256_bytes(text.encode("utf-8")),
            "tokens": tokens,
            "tokens_sha256": sha256_bytes(np.asarray(tokens, dtype="<u4").tobytes()),
            "speaker_dim": SPEAKER_DIM,
            "speaker_sha256": sha256_bytes(np.asarray(speaker[0].tolist(), dtype="<f4").tobytes()),
            "seed": SMOKE_SEED,
        }
        input_sha = sha256_bytes(canonical(input_record))
        (output_dir / "input.json").write_text(json.dumps(input_record, sort_keys=True, indent=2) + "\n", encoding="utf-8")

        model = SpeechT5ForTextToSpeech.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=False
        ).eval().to(device="cpu", dtype=torch.float32)
        stage = "api_call"
        call_record = {
            "api": "transformers.models.speecht5.modeling_speecht5.SpeechT5ForTextToSpeech.generate_speech",
            "kwargs": {
                "attention_mask": "input_attention_mask",
                "maxlenratio": 20.0,
                "minlenratio": 0.0,
                "output_cross_attentions": False,
                "return_output_lengths": True,
                "speaker_embeddings": "input_speaker_embeddings",
                "threshold": 0.5,
                "vocoder": None,
            },
            "input_sha256": input_sha,
            "checkpoint_sha256": SOURCE_WEIGHT_SHA256,
        }
        call_checkpoint_sha = sha256_bytes(canonical(call_record))
        with torch.inference_mode():
            generated, generated_lengths = model.generate_speech(
                input_ids,
                speaker_embeddings=speaker,
                attention_mask=attention_mask,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=20.0,
                vocoder=None,
                output_cross_attentions=False,
                return_output_lengths=True,
            )
        if len(generated_lengths) != 1:
            raise RuntimeError(f"official generator returned lengths={generated_lengths!r}")
        frames = int(generated_lengths[0])
        if frames <= 0 or tuple(generated.shape) != (1, frames, 80):
            raise RuntimeError(f"unexpected official mel shape={tuple(generated.shape)} frames={frames}")
        output_count, output_sha = write_f32(output_dir / "output.f32", generated[0])
        evidence = {
            "format": "vokra-speecht5-api-smoke-v1",
            "status": "PASS",
            "publication": "NO_UPLOAD",
            "upload": "NOT_PERFORMED",
            "reference_implementation": call_record["api"],
            "reference_package": f"transformers=={transformers.__version__}",
            "previous_isolated_transformers_pin": PREVIOUS_TRANSFORMERS,
            "transformers_security_advisory": SECURITY_ADVISORY,
            "transformers_security_patched_minimum": SECURITY_FLOOR,
            "upstream_hf": UPSTREAM_HF,
            "upstream_revision": UPSTREAM_REVISION,
            "revision_sha256": sha256_bytes(UPSTREAM_REVISION.encode()),
            "checkpoint_files": checkpoint_files,
            "vocoder": {"repo": VOCODER_REPO, "revision": VOCODER_REVISION, "pytorch_model_sha256": VOCODER_WEIGHT_SHA256},
            "project_sha256": project_sha,
            "lock_sha256": lock_sha,
            "package_rows_sha256": package_rows_sha,
            "package_sha256": installed_package_digest()[0],
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "output_count": output_count,
            "frames": frames,
            "mel_bins": int(generated.shape[-1]),
            "call_checkpoint_sha256": call_checkpoint_sha,
            "call": call_record,
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "platform": platform.platform()},
            **context,
        }
        evidence_path = output_dir / "evidence.json"
        evidence_path.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        validate_evidence_document(evidence_path, "PASS")
    except Exception as error:
        path = write_failure_evidence(output_dir, context, stage, error)
        validate_evidence_document(path, "FAIL")
        digest = sha256_file(path)
        print(f"SPEECHT5_API_SMOKE status=FAIL publication=NO_UPLOAD stage={stage} "
              f"error_type={type(error).__name__} evidence_sha256={digest}")
        return 2
    print("SPEECHT5_API_SMOKE status=PASS publication=NO_UPLOAD "
          f"revision={UPSTREAM_REVISION} input_sha256={input_sha} output_sha256={output_sha} "
          f"call_checkpoint_sha256={call_checkpoint_sha}")
    return 0


def self_test() -> int:
    assert PREVIOUS_TRANSFORMERS == "transformers==5.5.0"
    assert EXPECTED_TRANSFORMERS == "5.10.4"
    assert SECURITY_ADVISORY == "GHSA-xrqw-3rrv-vx5w"
    try:
        require_digest("0" * 64, LOCK_SHA256, "uv.lock")
    except RuntimeError:
        pass
    else:
        raise AssertionError("lock digest drift was accepted")
    try:
        require_digest(PYPROJECT_SHA256, "0" * 64, "pyproject")
    except RuntimeError:
        pass
    else:
        raise AssertionError("project digest drift was accepted")
    try:
        strict_json_loads('{"decision":"APPROVED","decision":"NO"}')
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys were accepted")
    try:
        require_project(Path("/nonexistent/speecht5-project"))
    except (RuntimeError, FileNotFoundError):
        pass
    else:
        raise AssertionError("missing project was accepted")
    try:
        verify_checkpoint(Path("/nonexistent/speecht5-checkpoint"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing model checkpoint was accepted")
    with tempfile.TemporaryDirectory(prefix="speecht5-api-smoke-selftest-") as directory:
        root = Path(directory).resolve()
        existing = root / "existing"
        existing.mkdir()
        try:
            require_absolute_no_symlink_path(existing, "existing", exists=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("pre-existing output path was accepted")
        try:
            require_absolute_no_symlink_path(root / ".." / "dotdot", "dotdot", exists=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dotdot output path was accepted")
        link = root / "link"
        link.symlink_to(existing, target_is_directory=True)
        try:
            require_absolute_no_symlink_path(link / "child", "symlinked", exists=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("symlinked output ancestry was accepted")
        if not paths_overlap(root / "work", root / "work" / "nested"):
            raise AssertionError("path overlap was not detected")
        for left, right in ((root / "work", root / "approval.json"), (root / "evidence", root / "approval.json")):
            if paths_overlap(left, right):
                raise AssertionError("disjoint paths were unexpectedly overlapping")
        if not paths_overlap(root / "work", root / "work" / "approval.json"):
            raise AssertionError("approval path overlap was not detected")
        manifest_path = Path(__file__).resolve().parent / "license_gate_manifest.json"
        approval_path = root / "approval.json"
        manifest_digest = sha256_file(manifest_path)
        scope = strict_json_loads(manifest_path.read_text(encoding="utf-8"))["approval_scope_sha256"]
        valid_approval = {"decision": "APPROVED", "scope_sha256": scope, "manifest_sha256": manifest_digest, "signer": "self-test", "digest": scope}
        approval_path.write_text(json.dumps(valid_approval), encoding="utf-8")
        approval = validate_approval_file(manifest_path.parent, approval_path)
        if approval["approval_scope_sha256"] != scope:
            raise AssertionError("valid preflight approval was not recorded")
        try:
            run_preflight_gate(manifest_path.parent, approval_path)
        except RuntimeError as error:
            if "existing preflight gate blocked" not in str(error):
                raise AssertionError(f"current manifest gate failed unexpectedly: {error}") from error
        else:
            raise AssertionError("current pending manifest bypassed the existing preflight gate")
        tampered = json.loads(approval_path.read_text(encoding="utf-8"))
        tampered["scope_sha256"] = "0" * 64
        approval_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            validate_approval_file(manifest_path.parent, approval_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered preflight approval was accepted")
        unknown = dict(valid_approval)
        unknown["unknown"] = True
        approval_path.write_text(json.dumps(unknown), encoding="utf-8")
        try:
            validate_approval_file(manifest_path.parent, approval_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unknown preflight approval field was accepted")
        pass_doc: dict[str, Any] = {key: None for key in PASS_EVIDENCE_KEYS}
        pass_doc.update({"format": "vokra-speecht5-api-smoke-v1", "status": "PASS", "publication": "NO_UPLOAD", "upload": "NOT_PERFORMED", "vokra_clean": True, "vokra_head": "a" * 40, "vokra_root": str(root), "preflight_gate": "PASS", "preflight_gate_sha256": "3" * 64, "preflight_manifest_sha256": "4" * 64, "approval_evidence_sha256": "a" * 64, "approval_scope_sha256": "b" * 64, "approval_signer": "self-test", "project_dir": str(root), "input_sha256": "c" * 64, "output_sha256": "d" * 64, "call_checkpoint_sha256": "e" * 64, "project_sha256": "f" * 64, "lock_sha256": "0" * 64, "package_rows_sha256": "1" * 64, "package_sha256": "2" * 64})
        pass_dir = root / "pass"
        pass_dir.mkdir()
        pass_path = pass_dir / "evidence.json"
        pass_path.write_text(json.dumps(pass_doc), encoding="utf-8")
        original_argv = sys.argv[:]
        try:
            sys.argv = [str(Path(__file__)), "--validate-evidence", "--output-dir", str(pass_dir), "--status", "PASS"]
            if main() != 0:
                raise AssertionError("PASS evidence directory CLI validation failed")
        finally:
            sys.argv = original_argv
        pass_doc["unknown"] = True
        pass_path.write_text(json.dumps(pass_doc), encoding="utf-8")
        try:
            validate_evidence_document(pass_path, "PASS")
        except RuntimeError:
            pass
        else:
            raise AssertionError("unknown PASS evidence field was accepted")
        fail_doc: dict[str, Any] = {key: None for key in FAIL_EVIDENCE_KEYS}
        fail_doc.update({"format": "vokra-speecht5-api-smoke-v1", "status": "FAIL", "publication": "NO_UPLOAD", "upload": "NOT_PERFORMED", "stage": "model_load", "error_type": "RuntimeError", "error": "safe", "vokra_clean": True, "vokra_head": "a" * 40, "vokra_root": str(root), "preflight_gate": "PASS", "preflight_gate_sha256": "3" * 64, "preflight_manifest_sha256": "4" * 64, "approval_evidence_sha256": "a" * 64, "approval_scope_sha256": "b" * 64, "approval_signer": "self-test"})
        fail_dir = root / "fail"
        fail_dir.mkdir()
        fail_path = fail_dir / "evidence.json"
        fail_path.write_text(json.dumps(fail_doc), encoding="utf-8")
        original_argv = sys.argv[:]
        try:
            sys.argv = [str(Path(__file__)), "--validate-evidence", "--output-dir", str(fail_dir), "--status", "FAIL"]
            if main() != 0:
                raise AssertionError("FAIL evidence directory CLI validation failed")
        finally:
            sys.argv = original_argv
        dirty_root = root / "dirty-checkout"
        dirty_root.mkdir()
        subprocess.run(["git", "init", "-q", str(dirty_root)], check=True, capture_output=True)
        (dirty_root / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(dirty_root), "add", "Cargo.toml"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dirty_root), "-c", "user.name=self-test", "-c", "user.email=self-test@example.invalid", "commit", "-qm", "initial"], check=True, capture_output=True)
        (dirty_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        try:
            git_checkout_context(dirty_root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("dirty checkout was accepted")
    original = os.environ.get("VOKRA_PUBLISH_ON_VAST")
    os.environ.pop("VOKRA_PUBLISH_ON_VAST", None)
    try:
        require_vast()
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-VAST environment was accepted")
    if original is not None:
        os.environ["VOKRA_PUBLISH_ON_VAST"] = original
    assert len(speaker_values()) == SPEAKER_DIM
    print("speecht5_tts.api_smoke: self-test PASS (offline, no model, no network)")
    return 0


def require_project(path: Path) -> None:
    verify_project(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--approval-evidence", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--validate-approval", action="store_true")
    parser.add_argument("--validate-evidence", action="store_true")
    parser.add_argument("--status", choices=("PASS", "FAIL"))
    parser.add_argument("--text", default=SMOKE_TEXT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.project_dir, args.output_dir, args.approval_evidence, args.vokra_root, args.status)) or args.validate_approval or args.validate_evidence or args.text != SMOKE_TEXT:
            parser.error("--self-test accepts no production arguments")
        return self_test()
    if args.validate_evidence:
        if args.output_dir is None or args.status is None or any(value is not None for value in (args.checkpoint, args.project_dir, args.approval_evidence, args.vokra_root)) or args.validate_approval:
            parser.error("--validate-evidence requires only --output-dir and --status")
        try:
            validate_evidence_output(args.output_dir, args.status)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"speecht5 API smoke evidence: BLOCKED: {error}", file=sys.stderr)
            return 2
        print(f"SPEECHT5_API_SMOKE_EVIDENCE status={args.status} verdict=PASS")
        return 0
    if args.validate_approval:
        if args.project_dir is None or args.approval_evidence is None or args.vokra_root is None or any(value is not None for value in (args.checkpoint, args.output_dir, args.status)) or args.validate_evidence:
            parser.error("--validate-approval requires --vokra-root, --project-dir, and --approval-evidence")
        try:
            git_checkout_context(args.vokra_root)
            require_absolute_no_symlink_path(args.project_dir, "parity project", exists=True)
            try:
                args.project_dir.relative_to(args.vokra_root)
            except ValueError as error:
                raise RuntimeError("parity project is outside --vokra-root") from error
            require_absolute_no_symlink_path(args.approval_evidence, "approval evidence", exists=True)
            require_regular(args.approval_evidence, "approval evidence")
            gate = run_preflight_gate(args.project_dir, args.approval_evidence)
            scope = validate_approval_file(args.project_dir, args.approval_evidence)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"speecht5 API smoke approval: BLOCKED: {error}", file=sys.stderr)
            return 2
        print(f"SPEECHT5_API_SMOKE_APPROVAL status=PASS preflight_gate={gate['preflight_gate']} scope_sha256={scope['approval_scope_sha256']}")
        return 0
    if args.checkpoint is None or args.project_dir is None or args.output_dir is None or args.approval_evidence is None or args.vokra_root is None or args.validate_evidence:
        parser.error("--checkpoint, --project-dir, --output-dir, --approval-evidence, and --vokra-root are required")
    try:
        return run(args.checkpoint, args.project_dir, args.output_dir, args.approval_evidence, args.vokra_root, args.text)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"speecht5 API smoke: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
