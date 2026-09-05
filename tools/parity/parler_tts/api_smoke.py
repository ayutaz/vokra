#!/usr/bin/env python3
"""Authenticate the pinned Parler-TTS Transformers generate API on VAST.

This is deliberately a small, independent compatibility probe.  It loads one
of the exact local Parler checkpoints through the official
``ParlerTTSForConditionalGeneration`` class and calls its official greedy
``generate`` plus embedded DAC decode with fixed token IDs.  It does not
import Vokra or claim numerical parity.  ``--self-test`` is stdlib-only and
never imports a third-party package, reads a model, or uses the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REPOSITORY = "https://github.com/huggingface/parler-tts.git"
SOURCE_REVISION = "d108732cd57788ec86bc857d99a6cabd66663d68"
LOCK_SHA256 = "0b37648f20d26197ba4a5dbeac5e6336b57454b5f7d2306dd1ddcbf321952bac"
PYPROJECT_SHA256 = "bea3b5f3c5e83b7af88e37a156a3ac8df2eccc5a1883a5daa229eecd080f3a1e"
PACKAGE_ROWS_SHA256 = "fd2b296630195079d54a79a0c911dacda6249f196b9b68df78149d60c58012f8"
TRANSFORMERS_VERSION = "5.10.4"
TORCH_VERSION = "2.11.0+cpu"
TORCHAUDIO_VERSION = "2.11.0+cpu"
SECURITY_ADVISORY = "GHSA-xrqw-3rrv-vx5w"
SECURITY_PATCHED_MINIMUM = "5.10.0"
EXPECTED_ENTRYPOINT = "ParlerTTSForConditionalGeneration"
FORMAT = "vokra-parler-tts-transformers-api-smoke-v1"

DESCRIPTION_TOKEN_IDS = [71, 1234, 1]
DESCRIPTION_ATTENTION_MASK = [1, 1, 1]
PROMPT_TOKEN_IDS = [12, 34, 1]
PROMPT_ATTENTION_MASK = [1, 1, 1]
MAX_FRAMES = 4
FRAME_HOP = 512
NUM_CODEBOOKS = 9
CODEBOOK_SIZE = 1_024
PREFLIGHT_EVIDENCE_KEYS = {
    "schema", "decision", "scope_sha256", "manifest_sha256", "lock_sha256",
    "pyproject_sha256", "signer", "digest",
}

EXPECTED_VARIANTS: dict[str, dict[str, Any]] = {
    "english": {
        "upstream_repo": "parler-tts/parler-tts-mini-v1",
        "upstream_revision": "0392b9451a601e528fd863bbb0598431fee810d9",
        "checkpoint_bytes": 3_511_490_560,
        "checkpoint_sha256": "bc430eb6752b96ffb3f67036d1a6e207fbd031575a775716ffa64ef1eeb03692",
        "config_bytes": 6_930,
        "config_sha256": "d8d2afa72bf3b098263a073c4d4df18627b76e1eb454c48f60bc5f787b2433b1",
        "generation_bytes": 265,
        "generation_sha256": "77831b39a5e0c4dba09b4dcbe37ce082e10f94c646920b20678c9c5289e52440",
    },
    "multilingual": {
        "upstream_repo": "parler-tts/parler-tts-mini-multilingual-v1.1",
        "upstream_revision": "11b27d57855dec1ce0914ba1f12363bf2ea75ba3",
        "checkpoint_bytes": 3_751_321_772,
        "checkpoint_sha256": "79c64e3705e0ccce122988c7817f0d65efa3fd37625906d90765858bdab38412",
        "config_bytes": 7_467,
        "config_sha256": "06d4cb727521542cab6b26d3ad1c8517d51fd1f551600ec67a59575364e221c6",
        "generation_bytes": 218,
        "generation_sha256": "3bb518e78ea5f32fbbcfc7f0aaed388e7aefede474d2bf4b8cf4502fd6b27a92",
    },
}

OUTPUT_KEYS = {
    "format",
    "status",
    "publication",
    "variant",
    "source",
    "project",
    "packages",
    "model",
    "input",
    "output",
    "call",
    "environment",
    "checkout",
    "authorization",
    "preflight",
}
AUTHORIZATION_OUTPUT_KEYS = PREFLIGHT_EVIDENCE_KEYS | {"file_sha256"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} JSON schema is not exact")


def require_fixed(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} {actual!r} != fixed {expected!r}")


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys)


def require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing, non-regular, or symlinked: {path}")


def validate_authorization(value: Any) -> dict[str, Any]:
    require_exact_keys(value, PREFLIGHT_EVIDENCE_KEYS, "preflight approval evidence")
    if value["schema"] != "v1" or value["decision"] != "APPROVED":
        raise RuntimeError("preflight approval schema or decision is invalid")
    for key in ("scope_sha256", "manifest_sha256", "lock_sha256", "pyproject_sha256", "digest"):
        if not isinstance(value[key], str) or not HEX64.fullmatch(value[key]):
            raise RuntimeError(f"preflight approval {key} is not a SHA-256 digest")
    require_fixed(value["lock_sha256"], LOCK_SHA256, "preflight approval uv.lock hash")
    require_fixed(value["pyproject_sha256"], PYPROJECT_SHA256, "preflight approval pyproject hash")
    if value["digest"] != value["scope_sha256"]:
        raise RuntimeError("preflight approval digest is not bound to scope")
    if not isinstance(value["signer"], str) or not value["signer"].strip():
        raise RuntimeError("preflight approval signer is empty")
    return value


def load_authorization(path: Path) -> dict[str, Any]:
    require_regular_file(path, "operator authorization")
    return validate_authorization(load_json(path))


def run_preflight_gate(project: Path, manifest: Path, approval: Path, gate_path: Path) -> dict[str, str]:
    gate_path = no_symlink_path(gate_path, "--preflight-gate", must_exist=True)
    if gate_path != project / "preflight_gate.py":
        raise RuntimeError("--preflight-gate must be the checked-in Parler preflight_gate.py")
    spec = importlib.util.spec_from_file_location("vokra_parler_preflight_gate", gate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the checked-in Parler preflight gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate(project, manifest, approval)
    if not isinstance(result, tuple) or len(result) != 2 or result[0] is not True:
        reason = result[1] if isinstance(result, tuple) and len(result) == 2 else "invalid gate result"
        raise RuntimeError(f"preflight gate blocked API smoke: {reason}")
    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest),
        "approval_sha256": sha256_file(approval),
    }


def no_symlink_path(path: Path, label: str, *, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be absolute")
    normalized = Path(os.path.normpath(str(path)))
    current = normalized
    while True:
        if current.is_symlink():
            raise RuntimeError(f"{label} contains a symlink: {current}")
        if current.exists():
            break
        if current == current.parent:
            break
        current = current.parent
    if must_exist and (not normalized.exists() or normalized.is_symlink()):
        raise RuntimeError(f"{label} is missing or symlinked: {normalized}")
    return normalized


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or path_is_under(left, right) or path_is_under(right, left)


def verify_vokra_root(root: Path) -> dict[str, Any]:
    root = no_symlink_path(root, "--vokra-root", must_exist=True)
    if not root.is_dir() or not (root / ".git").is_dir() or (root / ".git").is_symlink():
        raise RuntimeError("--vokra-root is not a real git checkout")
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
        head = git_head(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot authenticate Vokra checkout: {exc}") from exc
    if Path(top) != root or status:
        raise RuntimeError("Vokra checkout must be clean, including untracked files")
    if not HEX40.fullmatch(head):
        raise RuntimeError("Vokra checkout HEAD is not an immutable revision")
    return {"root": str(root), "head": head, "clean": True}


def validate_io_paths(args: argparse.Namespace, checkout: dict[str, Any]) -> dict[str, Path]:
    root = Path(checkout["root"])
    project = no_symlink_path(args.project, "--project", must_exist=True)
    manifest = no_symlink_path(args.manifest, "--manifest", must_exist=True)
    source = no_symlink_path(args.source_dir, "--source-dir", must_exist=True)
    model = no_symlink_path(args.model_dir, "--model-dir", must_exist=True)
    approval = no_symlink_path(args.operator_evidence, "--operator-evidence", must_exist=True)
    output = no_symlink_path(args.output, "--output", must_exist=False)
    if not project.is_dir() or not manifest.is_file() or not source.is_dir() or not model.is_dir() or not approval.is_file():
        raise RuntimeError("API smoke input path is not the required real file/directory")
    if manifest.parent != project:
        raise RuntimeError("--manifest must be the fixed project manifest")
    if not path_is_under(project, root) or not path_is_under(manifest, project):
        raise RuntimeError("project/manifest must be contained by --vokra-root")
    if output.exists() or output.is_symlink():
        raise RuntimeError("--output must be absent before the API smoke")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise RuntimeError("--output parent must be an existing real directory")
    protected = {"source": source, "model": model, "approval": approval, "output": output}
    if any(path_is_under(path, root) for path in protected.values()):
        raise RuntimeError("source/model/approval/output must not be contained by --vokra-root")
    values = list(protected.items())
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1:]:
            if paths_overlap(left, right):
                raise RuntimeError(f"{left_name} and {right_name} paths overlap")
    return {"root": root, "project": project, "manifest": manifest, **protected}


def git_head(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect git revision at {path}: {exc}") from exc


def verify_source(source_dir: Path) -> None:
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise RuntimeError(f"official source checkout is not a real directory: {source_dir}")
    require_fixed(git_head(source_dir), SOURCE_REVISION, "official source revision")
    try:
        dirty = subprocess.run(
            ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect source checkout cleanliness: {exc}") from exc
    if dirty:
        raise RuntimeError("official source checkout is dirty")


def verify_project(project: Path, manifest_path: Path) -> dict[str, Any]:
    project_file = project / "pyproject.toml"
    lock_file = project / "uv.lock"
    require_regular_file(project_file, "pyproject.toml")
    require_regular_file(lock_file, "uv.lock")
    require_regular_file(manifest_path, "API smoke manifest")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("API smoke manifest must be a JSON object")
    if manifest.get("gate_version") != 1:
        raise RuntimeError("authenticated Parler manifest version drifted")
    require_fixed(manifest.get("lock_sha256", ""), LOCK_SHA256, "manifest uv.lock hash")
    require_fixed(sha256_file(lock_file), LOCK_SHA256, "uv.lock hash")
    require_fixed(manifest.get("pyproject_sha256", ""), PYPROJECT_SHA256, "manifest pyproject hash")
    require_fixed(sha256_file(project_file), PYPROJECT_SHA256, "pyproject.toml hash")
    package_rows_sha256 = manifest.get("package_rows_sha256")
    if package_rows_sha256 != PACKAGE_ROWS_SHA256:
        raise RuntimeError("authenticated package-row hash is missing or malformed")
    if manifest.get("source_identity") != {
        "repo": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "license": "Apache-2.0",
    }:
        raise RuntimeError("authenticated Parler source identity drifted")
    route = manifest.get("reference_route")
    if not isinstance(route, dict):
        raise RuntimeError("authenticated reference route is missing")
    expected_route = {
        "entrypoint": EXPECTED_ENTRYPOINT,
        "transformers": TRANSFORMERS_VERSION,
        "isolated_transformers_pin": TRANSFORMERS_VERSION,
        "transformers_security_advisory": SECURITY_ADVISORY,
        "transformers_security_patched_minimum": SECURITY_PATCHED_MINIMUM,
        "torch": TORCH_VERSION,
        "torchaudio": TORCHAUDIO_VERSION,
    }
    for key, value in expected_route.items():
        if route.get(key) != value:
            raise RuntimeError(f"authenticated route drifted at {key}")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or len(variants) != len(EXPECTED_VARIANTS):
        raise RuntimeError("authenticated Parler model identities drifted")
    for actual, variant in zip(variants, ("english", "multilingual")):
        expected = EXPECTED_VARIANTS[variant]
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"authenticated Parler {variant} model identity drifted")
    return {
        "pyproject_sha256": sha256_file(project_file),
        "uv_lock_sha256": sha256_file(lock_file),
        "package_rows_sha256": package_rows_sha256,
        "reference_route": route,
    }


def verify_model(model_dir: Path, identity: dict[str, Any]) -> dict[str, Any]:
    files = {
        "model.safetensors": (identity["checkpoint_bytes"], identity["checkpoint_sha256"]),
        "config.json": (identity["config_bytes"], identity["config_sha256"]),
        "generation_config.json": (identity["generation_bytes"], identity["generation_sha256"]),
    }
    hashes: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_hash) in files.items():
        path = model_dir / name
        require_regular_file(path, f"pinned model input {name}")
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise RuntimeError(f"pinned model input identity mismatch: {path}")
        hashes[name] = {"bytes": actual_bytes, "sha256": actual_hash}
    return hashes


def validate_model_evidence(value: Any, identity: dict[str, Any]) -> None:
    require_exact_keys(value, {"repository", "revision", "files", "load_checkpoint_sha256"}, "model evidence")
    require_fixed(value["repository"], identity["upstream_repo"], "model repository")
    require_fixed(value["revision"], identity["upstream_revision"], "model revision")
    expected_load = {"entrypoint": EXPECTED_ENTRYPOINT, "local_files_only": True, "torch_dtype": "float32", "model": identity}
    if value["load_checkpoint_sha256"] != canonical_sha256(expected_load):
        raise RuntimeError("model load checkpoint hash is malformed")
    require_exact_keys(value["files"], {"model.safetensors", "config.json", "generation_config.json"}, "model files")
    expected_files = {
        "model.safetensors": (identity["checkpoint_bytes"], identity["checkpoint_sha256"]),
        "config.json": (identity["config_bytes"], identity["config_sha256"]),
        "generation_config.json": (identity["generation_bytes"], identity["generation_sha256"]),
    }
    for name, item in value["files"].items():
        require_exact_keys(item, {"bytes", "sha256"}, f"model file {name}")
        expected_bytes, expected_hash = expected_files[name]
        if item["bytes"] != expected_bytes or item["sha256"] != expected_hash:
            raise RuntimeError(f"model evidence identity mismatch: {name}")


def validate_evidence_data(value: Any, checkout: dict[str, Any], approval_path: Path, preflight: dict[str, str]) -> None:
    require_exact_keys(value, OUTPUT_KEYS, "API smoke evidence")
    if value["format"] != FORMAT or value["status"] != "PASS" or value["publication"] != "NO_UPLOAD":
        raise RuntimeError("API smoke evidence status/publication is invalid")
    if value["variant"] not in EXPECTED_VARIANTS:
        raise RuntimeError("API smoke evidence variant is invalid")
    require_exact_keys(value["source"], {"repository", "revision"}, "source evidence")
    require_fixed(value["source"]["repository"], SOURCE_REPOSITORY, "source repository")
    require_fixed(value["source"]["revision"], SOURCE_REVISION, "source revision")
    require_exact_keys(value["checkout"], {"root", "head", "clean"}, "checkout evidence")
    if value["checkout"] != checkout or value["checkout"]["clean"] is not True:
        raise RuntimeError("checkout evidence does not describe the authenticated clean checkout")
    require_exact_keys(value["preflight"], {"status", "manifest_sha256", "approval_sha256"}, "preflight evidence")
    if value["preflight"] != preflight:
        raise RuntimeError("preflight evidence is not bound to the successful gate")
    require_exact_keys(value["project"], {"pyproject_sha256", "uv_lock_sha256", "package_rows_sha256", "reference_route"}, "project evidence")
    require_fixed(value["project"]["pyproject_sha256"], PYPROJECT_SHA256, "project pyproject hash")
    require_fixed(value["project"]["uv_lock_sha256"], LOCK_SHA256, "project uv.lock hash")
    require_fixed(value["project"]["package_rows_sha256"], PACKAGE_ROWS_SHA256, "project package-row hash")
    require_exact_keys(value["packages"], {"parler_tts", "torch", "torchaudio", "transformers", "inventory_sha256"}, "package evidence")
    for key, expected in (("torch", TORCH_VERSION), ("torchaudio", TORCHAUDIO_VERSION), ("transformers", TRANSFORMERS_VERSION)):
        require_fixed(value["packages"][key], expected, f"package {key}")
    inventory = {key: value["packages"][key] for key in ("parler_tts", "torch", "torchaudio", "transformers")}
    if value["packages"]["inventory_sha256"] != canonical_sha256(inventory):
        raise RuntimeError("package inventory hash is invalid")
    identity = EXPECTED_VARIANTS[value["variant"]]
    validate_model_evidence(value["model"], identity)
    require_exact_keys(value["input"], {"description_token_ids", "description_attention_mask", "prompt_token_ids", "prompt_attention_mask", "dtype", "shape", "sha256"}, "input evidence")
    expected_input = {"description_input_ids": DESCRIPTION_TOKEN_IDS, "description_attention_mask": DESCRIPTION_ATTENTION_MASK, "prompt_input_ids": PROMPT_TOKEN_IDS, "prompt_attention_mask": PROMPT_ATTENTION_MASK, "dtype": "int64", "shape": [1, 3]}
    if value["input"]["description_token_ids"] != DESCRIPTION_TOKEN_IDS or value["input"]["description_attention_mask"] != DESCRIPTION_ATTENTION_MASK or value["input"]["prompt_token_ids"] != PROMPT_TOKEN_IDS or value["input"]["prompt_attention_mask"] != PROMPT_ATTENTION_MASK or value["input"]["dtype"] != "int64" or value["input"]["shape"] != {"description": [1, 3], "prompt": [1, 3]} or value["input"]["sha256"] != canonical_sha256(expected_input):
        raise RuntimeError("fixed API smoke input evidence drifted")
    require_exact_keys(value["output"], {"codes", "pcm"}, "output evidence")
    for name, expected_dtype in (("codes", "uint32"), ("pcm", "float32")):
        item = value["output"][name]
        require_exact_keys(item, {"dtype", "shape", "bytes", "sha256", "finite"}, f"{name} output evidence")
        if item["dtype"] != expected_dtype or item["finite"] is not True or not isinstance(item["sha256"], str) or not HEX64.fullmatch(item["sha256"]) or not isinstance(item["bytes"], int) or item["bytes"] <= 0:
            raise RuntimeError(f"{name} output evidence is malformed")
    code_shape, pcm_shape = value["output"]["codes"]["shape"], value["output"]["pcm"]["shape"]
    if not isinstance(code_shape, list) or len(code_shape) != 2 or code_shape[1] != NUM_CODEBOOKS or not 0 < code_shape[0] <= MAX_FRAMES or pcm_shape != [code_shape[0] * FRAME_HOP] or value["output"]["codes"]["bytes"] != code_shape[0] * NUM_CODEBOOKS * 4 or value["output"]["pcm"]["bytes"] != pcm_shape[0] * 4:
        raise RuntimeError("API smoke output shapes are invalid")
    require_exact_keys(value["call"], {"method", "input_sha256", "description_shape", "prompt_shape", "input_dtype", "do_sample", "min_new_tokens", "max_length", "decoder", "checkpoint_hashes", "api_return", "decoder_calls"}, "call evidence")
    if value["call"]["method"] != "ParlerTTSForConditionalGeneration.generate" or value["call"]["input_sha256"] != value["input"]["sha256"] or value["call"]["description_shape"] != [1, 3] or value["call"]["prompt_shape"] != [1, 3] or value["call"]["input_dtype"] != "int64" or value["call"]["do_sample"] is not False or value["call"]["min_new_tokens"] != 0 or value["call"]["max_length"] != MAX_FRAMES + NUM_CODEBOOKS or value["call"]["decoder"] != "embedded_audio_encoder.decode" or value["call"]["api_return"] != "decoded_pcm" or value["call"]["decoder_calls"] != 1:
        raise RuntimeError("API smoke call evidence is invalid")
    require_exact_keys(value["call"]["checkpoint_hashes"], {"pre_call_sha256", "post_call_sha256"}, "call checkpoint hashes")
    call_checkpoint = {key: value["call"][key] for key in ("method", "input_sha256", "description_shape", "prompt_shape", "input_dtype", "do_sample", "min_new_tokens", "max_length", "decoder")}
    expected_pre = canonical_sha256(call_checkpoint)
    expected_post = canonical_sha256({**call_checkpoint, "codes_sha256": value["output"]["codes"]["sha256"], "pcm_sha256": value["output"]["pcm"]["sha256"]})
    if value["call"]["checkpoint_hashes"] != {"pre_call_sha256": expected_pre, "post_call_sha256": expected_post}:
        raise RuntimeError("call checkpoint hashes do not bind the recorded call and outputs")
    require_exact_keys(value["environment"], {"platform", "machine", "python", "torch_threads"}, "environment evidence")
    if value["environment"]["machine"] != "x86_64" or value["environment"]["torch_threads"] != 1:
        raise RuntimeError("runtime environment evidence is invalid")
    authorization = load_authorization(approval_path)
    expected_authorization = {**authorization, "file_sha256": sha256_file(approval_path)}
    if value["authorization"] != expected_authorization:
        raise RuntimeError("API smoke authorization evidence is not bound to its file")


def validate_evidence_file(path: Path, checkout: dict[str, Any], approval_path: Path, preflight: dict[str, str]) -> None:
    path = no_symlink_path(path, "--validate-evidence", must_exist=True)
    require_regular_file(path, "API smoke evidence")
    validate_evidence_data(load_json(path), checkout, approval_path, preflight)


def runtime_package_record(parler_tts: Any, torch: Any, transformers: Any) -> dict[str, str]:
    packages = {
        "parler_tts": str(getattr(parler_tts, "__version__", "unknown")),
        "torch": str(torch.__version__),
        "torchaudio": str(__import__("torchaudio").__version__),
        "transformers": str(transformers.__version__),
    }
    if packages["torch"] != TORCH_VERSION:
        raise RuntimeError(f"torch {packages['torch']} != pinned {TORCH_VERSION}")
    if packages["torchaudio"] != TORCHAUDIO_VERSION:
        raise RuntimeError(f"torchaudio {packages['torchaudio']} != pinned {TORCHAUDIO_VERSION}")
    if packages["transformers"] != TRANSFORMERS_VERSION:
        raise RuntimeError(f"transformers {packages['transformers']} != pinned {TRANSFORMERS_VERSION}")
    return packages


def verify_runtime_host() -> None:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise RuntimeError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Parler API smoke is VAST/Linux x86_64-only")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    # Keep all third-party imports below the explicit VAST and file gates.
    verify_runtime_host()
    checkout = verify_vokra_root(args.vokra_root)
    project = no_symlink_path(args.project, "--project", must_exist=True)
    manifest = no_symlink_path(args.manifest, "--manifest", must_exist=True)
    approval_path = no_symlink_path(args.operator_evidence, "--operator-evidence", must_exist=True)
    preflight = run_preflight_gate(project, manifest, approval_path, args.preflight_gate)
    io_paths = validate_io_paths(args, checkout)
    project = io_paths["project"]
    manifest_identity = verify_project(project, io_paths["manifest"])
    source_dir = io_paths["source"]
    verify_source(source_dir)
    authorization = load_authorization(io_paths["approval"])
    identity = EXPECTED_VARIANTS[args.variant]
    input_hash = canonical_sha256(
        {
            "description_input_ids": DESCRIPTION_TOKEN_IDS,
            "description_attention_mask": DESCRIPTION_ATTENTION_MASK,
            "prompt_input_ids": PROMPT_TOKEN_IDS,
            "prompt_attention_mask": PROMPT_ATTENTION_MASK,
            "dtype": "int64",
            "shape": [1, len(DESCRIPTION_TOKEN_IDS)],
        }
    )
    model_hashes = verify_model(io_paths["model"], identity)

    sys.path.insert(0, str(source_dir))
    import parler_tts  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    import torchaudio  # noqa: PLC0415
    from parler_tts import ParlerTTSForConditionalGeneration  # noqa: PLC0415

    package_dir = Path(parler_tts.__file__).resolve().parent
    if package_dir.parent != source_dir or package_dir.name != "parler_tts":
        raise RuntimeError(f"parler_tts imported from unauthenticated path: {package_dir}")
    packages = runtime_package_record(parler_tts, torch, transformers)
    if torchaudio.__version__ != packages["torchaudio"]:
        raise RuntimeError("torchaudio package identity changed during probe")
    if ParlerTTSForConditionalGeneration.__name__ != EXPECTED_ENTRYPOINT:
        raise RuntimeError("official Parler entrypoint name drifted")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0x5041524C)
    model_load_checkpoint = {
        "entrypoint": EXPECTED_ENTRYPOINT,
        "local_files_only": True,
        "torch_dtype": "float32",
        "model": identity,
    }
    load_checkpoint_sha256 = canonical_sha256(model_load_checkpoint)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        io_paths["model"], local_files_only=True, torch_dtype=torch.float32
    ).eval()
    call_checkpoint = {
        "method": "ParlerTTSForConditionalGeneration.generate",
        "input_sha256": input_hash,
        "description_shape": [1, len(DESCRIPTION_TOKEN_IDS)],
        "prompt_shape": [1, len(PROMPT_TOKEN_IDS)],
        "input_dtype": "int64",
        "do_sample": False,
        "min_new_tokens": 0,
        "max_length": MAX_FRAMES + NUM_CODEBOOKS,
        "decoder": "embedded_audio_encoder.decode",
    }
    pre_call_sha256 = canonical_sha256(call_checkpoint)
    description_ids = torch.tensor([DESCRIPTION_TOKEN_IDS], dtype=torch.long)
    description_mask = torch.tensor([DESCRIPTION_ATTENTION_MASK], dtype=torch.long)
    prompt_ids = torch.tensor([PROMPT_TOKEN_IDS], dtype=torch.long)
    prompt_mask = torch.tensor([PROMPT_ATTENTION_MASK], dtype=torch.long)
    captured_codes: list[Any] = []
    original_decode = model.audio_encoder.decode

    def capture_decode(*decode_args: Any, **decode_kwargs: Any) -> Any:
        raw_codes = decode_kwargs.get("audio_codes")
        if raw_codes is None and decode_args:
            raw_codes = decode_args[0]
        if not isinstance(raw_codes, torch.Tensor):
            raise RuntimeError("official DAC decode received no tensor codes")
        captured_codes.append(raw_codes.detach().cpu().clone())
        return original_decode(*decode_args, **decode_kwargs)

    model.audio_encoder.decode = capture_decode
    with torch.inference_mode():
        try:
            pcm = model.generate(
                description_ids,
                attention_mask=description_mask,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                do_sample=False,
                min_new_tokens=0,
                max_length=MAX_FRAMES + NUM_CODEBOOKS,
            ).reshape(-1)
        finally:
            model.audio_encoder.decode = original_decode
    if len(captured_codes) != 1:
        raise RuntimeError(f"official DAC decode call count {len(captured_codes)} != expected 1")
    codes = captured_codes[0].to(torch.int64)
    while codes.ndim > 2:
        if codes.shape[0] != 1:
            raise RuntimeError(f"unexpected official DAC code shape {list(codes.shape)}")
        codes = codes[0]
    if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS:
        raise RuntimeError(f"unexpected official DAC code shape {list(codes.shape)}")
    if bool(torch.any(codes < 0)) or bool(torch.any(codes >= CODEBOOK_SIZE)):
        raise RuntimeError("official Parler-TTS emitted an out-of-range DAC code")
    codes = codes.transpose(0, 1).contiguous()
    if codes.shape[0] == 0 or codes.shape[0] > MAX_FRAMES:
        raise RuntimeError(f"official generated frame count {codes.shape[0]} is invalid")
    code_bytes = codes.numpy().astype("<u4", copy=False).tobytes(order="C")
    pcm = pcm.detach().cpu().to(torch.float32).contiguous()
    pcm_bytes = pcm.numpy().astype("<f4", copy=False).tobytes(order="C")
    if not bool(torch.isfinite(pcm).all()):
        raise RuntimeError("official DAC returned non-finite PCM")
    if pcm.numel() != codes.shape[0] * FRAME_HOP:
        raise RuntimeError(f"official PCM length {pcm.numel()} != {codes.shape[0]} * {FRAME_HOP}")
    output = {
        "codes": {"dtype": "uint32", "shape": list(codes.shape), "bytes": len(code_bytes), "sha256": sha256_bytes(code_bytes), "finite": True},
        "pcm": {"dtype": "float32", "shape": list(pcm.shape), "bytes": len(pcm_bytes), "sha256": sha256_bytes(pcm_bytes), "finite": True},
    }
    post_call_sha256 = canonical_sha256({**call_checkpoint, "codes_sha256": output["codes"]["sha256"], "pcm_sha256": output["pcm"]["sha256"]})
    evidence = {
        "format": FORMAT,
        "status": "PASS",
        "publication": "NO_UPLOAD",
        "variant": args.variant,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "project": manifest_identity,
        "packages": {**packages, "inventory_sha256": canonical_sha256(packages)},
        "model": {
            "repository": identity["upstream_repo"],
            "revision": identity["upstream_revision"],
            "files": model_hashes,
            "load_checkpoint_sha256": load_checkpoint_sha256,
        },
        "input": {
            "description_token_ids": DESCRIPTION_TOKEN_IDS,
            "description_attention_mask": DESCRIPTION_ATTENTION_MASK,
            "prompt_token_ids": PROMPT_TOKEN_IDS,
            "prompt_attention_mask": PROMPT_ATTENTION_MASK,
            "dtype": "int64",
            "shape": {
                "description": [1, len(DESCRIPTION_TOKEN_IDS)],
                "prompt": [1, len(PROMPT_TOKEN_IDS)],
            },
            "sha256": input_hash,
        },
        "output": output,
        "call": {
            **call_checkpoint,
            "checkpoint_hashes": {
                "pre_call_sha256": pre_call_sha256,
                "post_call_sha256": post_call_sha256,
            },
            "api_return": "decoded_pcm",
            "decoder_calls": len(captured_codes),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch_threads": torch.get_num_threads(),
        },
        "checkout": checkout,
        "preflight": preflight,
        "authorization": {
            **authorization,
            "file_sha256": sha256_file(io_paths["approval"]),
        },
    }
    require_exact_keys(evidence, OUTPUT_KEYS, "API smoke evidence")
    io_paths["output"].write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    validate_evidence_file(io_paths["output"], checkout, io_paths["approval"], preflight)
    return evidence


def self_test() -> int:
    try:
        try:
            reject_duplicate_json_keys([("a", 1), ("a", 2)])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate JSON keys were accepted")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
            try:
                load_json(duplicate)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate JSON file was accepted")
        try:
            require_exact_keys({"status": "PASS", "unexpected": True}, {"status"}, "self-test")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown JSON field was accepted")
        authorization = {
            "schema": "v1",
            "decision": "APPROVED",
            "scope_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "lock_sha256": LOCK_SHA256,
            "pyproject_sha256": PYPROJECT_SHA256,
            "signer": "self-test",
            "digest": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "authorization.json"
            valid.write_text(json.dumps(authorization), encoding="utf-8")
            load_authorization(valid)
            tampered = dict(authorization)
            tampered["digest"] = "c" * 64
            tampered_path = root / "tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                load_authorization(tampered_path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("tampered operator authorization was accepted")
            duplicate_auth = root / "duplicate-authorization.json"
            duplicate_auth.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
            try:
                load_authorization(duplicate_auth)
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate operator authorization was accepted")
            tampered_evidence = root / "tampered-evidence.json"
            tampered_evidence.write_text('{"format":"wrong","status":"PASS"}', encoding="utf-8")
            try:
                validate_evidence_file(tampered_evidence, {"root": str(root), "head": "0" * 40, "clean": True}, valid, {"status": "PASS", "manifest_sha256": "b" * 64, "approval_sha256": "c" * 64})
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError("tampered evidence was accepted")
            try:
                run_preflight_gate(Path(__file__).resolve().parent, Path(__file__).resolve().parent / "license_gate_manifest.json", valid, Path(__file__).resolve().parent / "preflight_gate.py")
            except RuntimeError:
                pass
            else:
                raise AssertionError("pending production preflight gate was accepted")
        for actual, expected, label in (
            ("0" * 40, SOURCE_REVISION, "revision"),
            ("b" * 64, LOCK_SHA256, "lock"),
        ):
            try:
                require_fixed(actual, expected, label)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{label} drift was accepted")
        try:
            verify_model(Path("/definitely-missing-parler-model"), EXPECTED_VARIANTS["english"])
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing model was accepted")
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            subprocess.run(["git", "init", "-q", str(checkout)], check=True, capture_output=True)
            (checkout / "README").write_text("self-test\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(checkout), "add", "README"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(checkout), "-c", "user.name=self-test", "-c", "user.email=self-test@example.invalid", "commit", "-qm", "initial"], check=True, capture_output=True)
            (checkout / "untracked").write_text("dirty\n", encoding="utf-8")
            try:
                verify_vokra_root(checkout)
            except RuntimeError:
                pass
            else:
                raise AssertionError("dirty checkout was accepted")
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            output_parent = root / "output"
            output_parent.mkdir()
            output = output_parent / "evidence.json"
            if output.exists():
                raise AssertionError("self-test output unexpectedly exists")
            link = root / "link"
            link.symlink_to(output_parent, target_is_directory=True)
            try:
                no_symlink_path(link / "evidence.json", "self-test symlink", must_exist=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlinked output ancestry was accepted")
            try:
                no_symlink_path(Path("relative") / "x", "self-test relative", must_exist=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError("relative output was accepted")
            if output.exists():
                raise AssertionError("self-test clobbered output")
        print("parler api smoke self-test: PASS (offline, no model, no network)")
        return 0
    except (AssertionError, OSError, ValueError) as exc:
        print(f"parler api smoke self-test: FAIL: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--validate-evidence", type=Path)
    parser.add_argument("--variant", choices=sorted(EXPECTED_VARIANTS))
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--preflight-gate", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operator-evidence", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.contract_only or args.validate_evidence is not None or any(value is not None for value in (args.variant, args.project, args.manifest, args.source_dir, args.model_dir, args.output, args.operator_evidence, args.vokra_root, args.preflight_gate)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if args.validate_evidence is not None:
        if args.contract_only or args.variant is not None or args.project is not None or args.manifest is not None or args.source_dir is not None or args.model_dir is not None or args.output is not None or args.vokra_root is None or args.operator_evidence is None or args.preflight_gate is None:
            parser.error("--validate-evidence requires only --validate-evidence, --vokra-root, --preflight-gate, and --operator-evidence")
        checkout = verify_vokra_root(args.vokra_root)
        project = no_symlink_path(args.preflight_gate.parent, "preflight project", must_exist=True)
        if not path_is_under(project, Path(checkout["root"])):
            parser.error("preflight project must be contained by --vokra-root")
        approval = no_symlink_path(args.operator_evidence, "--operator-evidence", must_exist=True)
        manifest = project / "license_gate_manifest.json"
        preflight = run_preflight_gate(project, manifest, approval, args.preflight_gate)
        validate_evidence_file(args.validate_evidence, checkout, approval, preflight)
        print("parler api smoke evidence: PASS (offline, no model, no network)")
        return 0
    if args.contract_only:
        if args.project is None or args.manifest is None or args.operator_evidence is None or args.vokra_root is None or args.preflight_gate is None or args.variant is not None or args.source_dir is not None or args.model_dir is not None or args.output is not None:
            parser.error("--contract-only requires only --project, --manifest, --vokra-root, --preflight-gate, and --operator-evidence")
        if not args.project.is_absolute() or not args.manifest.is_absolute() or not args.operator_evidence.is_absolute() or not args.vokra_root.is_absolute() or not args.preflight_gate.is_absolute():
            parser.error("contract paths must be absolute")
        checkout = verify_vokra_root(args.vokra_root)
        project = no_symlink_path(args.project, "--project", must_exist=True)
        manifest = no_symlink_path(args.manifest, "--manifest", must_exist=True)
        approval = no_symlink_path(args.operator_evidence, "--operator-evidence", must_exist=True)
        if not path_is_under(project, Path(checkout["root"])) or manifest.parent != project:
            parser.error("contract project/manifest must be the checked-out Parler project")
        run_preflight_gate(project, manifest, approval, args.preflight_gate)
        verify_project(project, manifest)
        load_authorization(approval)
        print("parler api smoke contract: VALIDATED (offline, no model, no network)")
        return 0
    required = (args.variant, args.project, args.manifest, args.source_dir, args.model_dir, args.output, args.operator_evidence, args.vokra_root, args.preflight_gate)
    if any(value is None for value in required):
        parser.error("production mode requires --variant, --project, --manifest, --source-dir, --model-dir, --output, and --operator-evidence")
    for path in (args.project, args.manifest, args.source_dir, args.model_dir, args.output, args.operator_evidence, args.vokra_root, args.preflight_gate):
        if not path.is_absolute():
            parser.error("production paths must be absolute")
    run_probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
