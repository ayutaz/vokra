#!/usr/bin/env python3
"""Inspect the official SGMSE-VoiceBank checkpoint without creating weights.

The HF checkpoint is an untrusted pickle container.  Only
``torch.load(weights_only=True)`` is attempted; SpeechBrain custom classes and
any unsafe pickle fallback are intentionally forbidden.  This sidecar writes
an evidence manifest, never a safetensors or GGUF candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MODEL_REPOSITORY = "speechbrain/sgmse-voicebank"
MODEL_REVISION = "8f4ff7b65284c49492a43349b8106e094ac0d365"
CHECKPOINT_NAME = "score_model_ema.ckpt"
CHECKPOINT_SIZE = 262_593_305
CHECKPOINT_SHA256 = "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3"
SOURCE_REPOSITORY = "https://github.com/sp-uhh/sgmse.git"
SOURCE_REVISION = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
SOURCE_LICENSE_SPDX = "mit"
SPEECHBRAIN_REPOSITORY = "https://github.com/speechbrain/speechbrain.git"
SPEECHBRAIN_REVISION = "2b3f4f44351fd08a627c4ab307de5c420351bc19"
SPEECHBRAIN_VERSION = "1.0.3"
SPEECHBRAIN_SDIST_SHA256 = "fcab3c6e90012cecb1eed40ea235733b550137e73da6bfa2340ba191ec714052"
SPEECHBRAIN_WHEEL_SHA256 = "9859d4c1b1fb3af3b85523c0c89f52e45a04f305622ed55f31aa32dd2fba19e9"
EXECUTABLE_SPEECHBRAIN_FILES = (
    "speechbrain/integrations/models/sgmse_plus.py",
    "speechbrain/inference/enhancement.py",
)
EXECUTABLE_SPEECHBRAIN_MARKERS = {
    "speechbrain/integrations/models/sgmse_plus.py": ("class ScoreModel",),
    "speechbrain/inference/enhancement.py": ("class SGMSEEnhancement", "enhance_batch"),
}
ALGORITHM_SOURCE_ROLES = {
    "score_model": ("class ScoreModel", "ScoreModel("),
    "ncsnpp": ("NCSNpp", "ncsnpp_v2"),
    "sde": ("OUVESDE", "SDERegistry"),
    "sampler_predictor": ("reverse_diffusion",),
    "sampler_corrector": ("ald", "CorrectorRegistry"),
}
ALGORITHM_SOURCE_FIXED_FILES = {
    "sampler_predictor": "sgmse/sampling/predictors.py",
    "sampler_corrector": "sgmse/sampling/correctors.py",
}
LOCKED_DISTRIBUTION_BLOCKER = "BLOCKED_LOCKED_DISTRIBUTION_MISSING_SGMSE_INTEGRATION"
LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS = {
    "speechbrain/integrations/models/sgmse_plus.py": {
        "present": False,
        "required_markers": {},
    },
    "speechbrain/inference/enhancement.py": {
        "present": True,
        "required_markers": {"class SGMSEEnhancement": False},
    },
}
# The authenticated fixed HF revision contains exactly these two companions;
# example.wav is not present in that snapshot and must not be downloaded or
# treated as evidence.
COMPANION_FILES = ("README.md", ".gitattributes")

CONFIG_PATTERNS = {
    "sample_rate": r"^sample_rate:\s*16000\s*$",
    "n_fft": r"^n_fft:\s*510\s*$",
    "hop_length": r"^hop_length:\s*128\s*$",
    "window_type": r"^window_type:\s*hann\s*$",
    "sampler_type": r"^\s*sampler_type:\s*pc\s*$",
    "predictor": r"^\s*predictor:\s*reverse_diffusion\s*$",
    "corrector": r"^\s*corrector:\s*ald\s*$",
    "steps": r"^\s*N:\s*30\s*$",
    "corrector_steps": r"^\s*corrector_steps:\s*1\s*$",
    "snr": r"^\s*snr:\s*0\.5\s*$",
    "score_model": r"!new:speechbrain\.integrations\.models\.sgmse_plus\.ScoreModel",
    "backbone": r"^\s*backbone:\s*ncsnpp_v2\s*$",
    "sde": r"^\s*sde:\s*ouve\s*$",
    "theta": r"^\s*theta:\s*1\.5\s*$",
    "sigma_min": r"^\s*sigma_min:\s*0\.05\s*$",
    "sigma_max": r"^\s*sigma_max:\s*0\.5\s*$",
    "ema_decay": r"^\s*ema_decay:\s*0\.999\s*$",
    "network_scaling": r"^\s*network_scaling:\s*1/t\s*$",
    "c_in": r"^\s*c_in:\s*['\"]?1['\"]?\s*$",
    "c_out": r"^\s*c_out:\s*['\"]?1['\"]?\s*$",
    "c_skip": r"^\s*c_skip:\s*['\"]?0['\"]?\s*$",
    "sigma_data": r"^\s*sigma_data:\s*0\.1\s*$",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_facts(text: str) -> tuple[dict[str, bool], list[str]]:
    facts = {name: re.search(pattern, text, re.MULTILINE) is not None for name, pattern in CONFIG_PATTERNS.items()}
    return facts, [name for name, present in facts.items() if not present]


def tensor_contract_status(loaded: dict[str, Any]) -> str:
    """Return the only safe status a checkpoint tensor contract may claim.

    A successful ``weights_only`` load is not, by itself, native-model
    authentication.  The native binder remains closed until this inspector's
    concrete tensor manifest has been reviewed and bound to the model schema.
    """
    if loaded.get("safe_load_status") != "SAFE_LOADED":
        return "AUTHENTICATED_MANIFEST_REQUIRED"
    manifest = loaded.get("tensor_manifest")
    if not isinstance(manifest, dict) or not manifest:
        return "AUTHENTICATED_MANIFEST_REQUIRED"
    return "SAFE_LOADED_MANIFEST"


def tensor_manifest(value: Any, path: str = "") -> tuple[dict[str, dict[str, Any]], list[str], bool]:
    import torch

    tensors: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []
    finite = True
    if isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
        tensors[path or "<root>"] = {
            "shape": [int(axis) for axis in value.shape],
            "dtype": str(value.dtype),
            "count": int(value.numel()),
            "finite": finite,
        }
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            child, child_unsupported, child_finite = tensor_manifest(
                value[key], f"{path}.{key}" if path else str(key)
            )
            tensors.update(child)
            unsupported.extend(child_unsupported)
            finite = finite and child_finite
    elif isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            child, child_unsupported, child_finite = tensor_manifest(child_value, f"{path}[{index}]")
            tensors.update(child)
            unsupported.extend(child_unsupported)
            finite = finite and child_finite
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        unsupported.append(f"{path}:{type(value).__name__}")
    return tensors, unsupported, finite


def _tensor_map_candidates(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    import torch

    candidates: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if value and all(isinstance(item, torch.Tensor) for item in value.values()):
            candidates.append((path or "<root>", value))
        for key in sorted(value, key=str):
            candidates.extend(_tensor_map_candidates(value[key], f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            candidates.extend(_tensor_map_candidates(child, f"{path}[{index}]"))
    return candidates


def safe_load_manifest(path: Path) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "container_path": str(path),
        "torch_loader": "torch.load(weights_only=True, map_location='cpu')",
        "unsafe_pickle_fallback": False,
    }
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - preserve loud safe-load evidence
        result["safe_load_status"] = "BLOCKED_WEIGHTS_ONLY"
        result["safe_load_error"] = f"{type(error).__name__}: {error}"
        return result

    candidates = _tensor_map_candidates(raw)
    if len(candidates) != 1:
        result["safe_load_status"] = "BLOCKED_AMBIGUOUS_TENSOR_CONTAINER"
        result["candidate_paths"] = [candidate[0] for candidate in candidates]
        return result
    container_path, state_dict = candidates[0]
    tensors, unsupported, finite = tensor_manifest(state_dict, "")
    result["container_path"] = container_path
    result["ema_extraction"] = (
        "top_level_state_dict" if container_path == "<root>" else f"safe_tensor_map:{container_path}"
    )
    result["tensor_manifest"] = tensors
    result["tensor_count"] = len(tensors)
    result["parameter_count"] = sum(item["count"] for item in tensors.values())
    result["unsupported_objects"] = unsupported
    result["all_finite"] = finite
    if not tensors:
        result["safe_load_status"] = "BLOCKED_EMPTY_TENSOR_MANIFEST"
    elif not finite:
        result["safe_load_status"] = "BLOCKED_NONFINITE_TENSOR"
    else:
        result["safe_load_status"] = "SAFE_LOADED"
    return result


def source_identity(source_dir: Path) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "repository": SOURCE_REPOSITORY,
        "expected_revision": SOURCE_REVISION,
        "license_spdx": SOURCE_LICENSE_SPDX,
    }
    blockers: list[str] = []
    license_path = source_dir / "LICENSE"
    if not license_path.is_file():
        blockers.append(f"missing source license: {license_path}")
    else:
        result["license_sha256"] = sha256(license_path)
        result["license_text_is_mit"] = "MIT License" in license_path.read_text(encoding="utf-8", errors="replace")
        if not result["license_text_is_mit"]:
            blockers.append("source LICENSE does not contain the MIT notice")
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        commit = ""
        blockers.append(f"source revision unavailable: {error}")
    result["resolved_revision"] = commit
    if commit != SOURCE_REVISION:
        blockers.append(f"source revision {commit!r} != {SOURCE_REVISION!r}")
    return result, blockers


def speechbrain_source_identity(source_dir: Path) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {
        "repository": SPEECHBRAIN_REPOSITORY,
        "expected_revision": SPEECHBRAIN_REVISION,
        "locked_distribution": {
            "version": SPEECHBRAIN_VERSION,
            "sdist_sha256": SPEECHBRAIN_SDIST_SHA256,
            "wheel_sha256": SPEECHBRAIN_WHEEL_SHA256,
        },
    }
    blockers: list[str] = []
    license_path = source_dir / "LICENSE"
    if not license_path.is_file():
        blockers.append(f"missing SpeechBrain license: {license_path}")
    else:
        license_text = license_path.read_text(encoding="utf-8", errors="replace")
        result["license_sha256"] = sha256(license_path)
        result["license_spdx"] = "apache-2.0"
        result["license_text_is_apache"] = "Apache License" in license_text
        if not result["license_text_is_apache"]:
            blockers.append("SpeechBrain LICENSE does not contain the Apache notice")
    try:
        commit = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        commit = ""
        blockers.append(f"SpeechBrain revision unavailable: {error}")
    result["resolved_revision"] = commit
    if commit != SPEECHBRAIN_REVISION:
        blockers.append(f"SpeechBrain revision {commit!r} != {SPEECHBRAIN_REVISION!r}")

    files: dict[str, Any] = {}
    for relative in EXECUTABLE_SPEECHBRAIN_FILES:
        path = source_dir / relative
        if not path.is_file():
            blockers.append(f"missing executable SpeechBrain source: {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        markers = EXECUTABLE_SPEECHBRAIN_MARKERS[relative]
        marker_status = {marker: marker in text for marker in markers}
        files[relative] = {
            "sha256": sha256(path),
            "size": path.stat().st_size,
            "required_markers": marker_status,
        }
        for marker, present in marker_status.items():
            if not present:
                blockers.append(
                    f"SpeechBrain source {relative} is missing executable marker: {marker}"
                )
    result["executable_files"] = files
    distribution, distribution_blockers = locked_distribution_source_audit_from_environment()
    result["locked_distribution_audit"] = distribution
    blockers.extend(distribution_blockers)
    return result, blockers


def locked_distribution_source_audit(
    version: str,
    files: set[str],
    source_texts: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    """Audit the locked wheel without importing SpeechBrain runtime modules.

    The reviewed 1.0.3 distribution is intentionally a dependency blocker:
    its source tree lacks the SGMSE integration used by the authenticated
    checkpoint.  A complete-looking replacement is also blocked until its
    provenance is reviewed rather than silently treated as runtime-ready.
    """

    observed: dict[str, Any] = {}
    for relative, expected in LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS.items():
        present = relative in files
        markers = {
            marker: marker in source_texts.get(relative, "")
            for marker in expected["required_markers"]
        }
        observed[relative] = {"present": present, "required_markers": markers}
    result: dict[str, Any] = {
        "version": version,
        "expected_version": SPEECHBRAIN_VERSION,
        "expected_source_roles": LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS,
        "observed_source_roles": observed,
        "status": LOCKED_DISTRIBUTION_BLOCKER,
    }
    blockers = [
        f"{LOCKED_DISTRIBUTION_BLOCKER}: reviewed SpeechBrain {SPEECHBRAIN_VERSION} distribution lacks source-backed SGMSE integration"
    ]
    if version != SPEECHBRAIN_VERSION:
        blockers.append(f"locked SpeechBrain version {version!r} != {SPEECHBRAIN_VERSION!r}")
    expected_observed = {
        relative: {
            "present": expected["present"],
            "required_markers": {
                marker: expected_value
                for marker, expected_value in expected["required_markers"].items()
            },
        }
        for relative, expected in LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS.items()
    }
    if observed != expected_observed:
        result["status"] = "BLOCKED_LOCKED_DISTRIBUTION_SOURCE_AUDIT_MISMATCH"
        blockers.append("locked SpeechBrain source audit differs from the reviewed 1.0.3 evidence")
    return result, blockers


def locked_distribution_source_audit_from_environment() -> tuple[dict[str, Any], list[str]]:
    """Read package metadata/source markers without importing SpeechBrain."""

    try:
        from importlib.metadata import distribution

        package = distribution("speechbrain")
        files = {str(path).replace("\\", "/") for path in (package.files or ())}
        source_texts: dict[str, str] = {}
        for relative in LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS:
            if relative not in files:
                continue
            path = Path(package.locate_file(relative))
            if path.is_file() and not path.is_symlink():
                source_texts[relative] = path.read_text(encoding="utf-8", errors="replace")
        return locked_distribution_source_audit(package.version, files, source_texts)
    except Exception as error:  # noqa: BLE001 - dependency evidence must fail closed
        return (
            {
                "version": None,
                "expected_version": SPEECHBRAIN_VERSION,
                "status": "BLOCKED_LOCKED_DISTRIBUTION_UNAVAILABLE",
                "error": f"{type(error).__name__}: {error}",
            },
            [f"BLOCKED_LOCKED_DISTRIBUTION_UNAVAILABLE: {error}"],
        )


def algorithm_source_inventory(source_dir: Path) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION}
    blockers: list[str] = []
    inventory: dict[str, Any] = {}
    try:
        candidates = sorted(source_dir.rglob("*.py"))
    except OSError as error:
        return result, [f"algorithm source scan failed: {error}"]
    for role, markers in ALGORITHM_SOURCE_ROLES.items():
        matches = []
        role_candidates = candidates
        fixed_relative = ALGORITHM_SOURCE_FIXED_FILES.get(role)
        if fixed_relative is not None:
            role_candidates = [source_dir / fixed_relative]
        for path in role_candidates:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if all(marker in text for marker in markers):
                matches.append({"path": str(path.relative_to(source_dir)), "sha256": sha256(path)})
        inventory[role] = matches
        if not matches:
            blockers.append(f"algorithm source role {role!r} has no pinned implementation file")
    result["files_by_role"] = inventory
    return result, blockers


def companion_identity(companion_dir: Path) -> tuple[dict[str, Any], list[str]]:
    companions: dict[str, Any] = {}
    blockers: list[str] = []
    for filename in COMPANION_FILES:
        path = companion_dir / filename
        if not path.is_file():
            blockers.append(f"missing HF companion file: {path}")
            continue
        companions[filename] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return companions, blockers


def inspect(
    ckpt: Path,
    hyperparams: Path,
    companion_dir: Path,
    algorithm_source_dir: Path,
    speechbrain_source_dir: Path,
    manifest_path: Path,
) -> int:
    blockers: list[str] = []
    checkpoint: dict[str, Any] = {"filename": ckpt.name, "expected_filename": CHECKPOINT_NAME}
    if ckpt.name != CHECKPOINT_NAME:
        blockers.append(f"checkpoint filename {ckpt.name!r} != {CHECKPOINT_NAME!r}")
    if ckpt.is_file():
        checkpoint["size"] = ckpt.stat().st_size
        checkpoint["sha256"] = sha256(ckpt)
        checkpoint["expected_size"] = CHECKPOINT_SIZE
        checkpoint["expected_sha256"] = CHECKPOINT_SHA256
        if checkpoint["size"] != CHECKPOINT_SIZE or checkpoint["sha256"] != CHECKPOINT_SHA256:
            blockers.append("checkpoint exact size/SHA256 identity mismatch")
    else:
        blockers.append(f"missing checkpoint: {ckpt}")

    config: dict[str, Any] = {"filename": hyperparams.name}
    if hyperparams.is_file():
        text = hyperparams.read_text(encoding="utf-8")
        facts, missing = config_facts(text)
        config.update({"sha256": sha256(hyperparams), "facts": facts, "raw": text})
        blockers.extend(f"hyperparams missing or mismatched field: {name}" for name in missing)
    else:
        blockers.append(f"missing hyperparams: {hyperparams}")

    algorithm_source, algorithm_blockers = source_identity(algorithm_source_dir)
    blockers.extend(algorithm_blockers)
    algorithm_inventory, inventory_blockers = algorithm_source_inventory(algorithm_source_dir)
    blockers.extend(inventory_blockers)
    speechbrain_source, speechbrain_blockers = speechbrain_source_identity(speechbrain_source_dir)
    blockers.extend(speechbrain_blockers)
    companions, companion_blockers = companion_identity(companion_dir)
    blockers.extend(companion_blockers)
    loaded: dict[str, Any] = {}
    if ckpt.is_file() and checkpoint.get("size") == CHECKPOINT_SIZE and checkpoint.get("sha256") == CHECKPOINT_SHA256:
        loaded = safe_load_manifest(ckpt)
        if loaded.get("safe_load_status") != "SAFE_LOADED":
            blockers.append(f"checkpoint safe-load: {loaded.get('safe_load_status')}")
    else:
        loaded = {
            "safe_load_status": "BLOCKED_CHECKPOINT_IDENTITY",
            "unsafe_pickle_fallback": False,
        }

    manifest = {
        "format": "vokra-sgmse-voicebank-inspection-v1",
        "tensor_contract": {
            "format": "vokra-sgmse-tensor-contract-v1",
            # The inspector is the only producer of the checkpoint-specific
            # contract.  Keep this explicit until a real safe-loaded manifest
            # exists; a hand-written or historical 647-tensor list must never
            # close the converter gate.
            "status": tensor_contract_status(loaded),
            "source": "safe_load.tensor_manifest",
            "tensor_count": loaded.get("tensor_count"),
        },
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "weight_license_spdx": "apache-2.0",
        "checkpoint": checkpoint,
        "hyperparams": config,
        "companions": companions,
        "algorithm_source": {**algorithm_source, **algorithm_inventory},
        "speechbrain_source": speechbrain_source,
        "safe_load": loaded,
        "blockers": blockers,
        "runtime_status": "INSPECTION_ONLY",
        "parity_status": "INSPECTION_ONLY",
        "publication": "NO_UPLOAD",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blockers:
        print(f"SGMSE inspection blocked; evidence preserved at {manifest_path}", file=sys.stderr)
        return 2
    print(f"SGMSE inspection complete; evidence written to {manifest_path}")
    return 0


def self_test() -> None:
    assert MODEL_REVISION == "8f4ff7b65284c49492a43349b8106e094ac0d365"
    assert COMPANION_FILES == ("README.md", ".gitattributes")
    assert "example.wav" not in COMPANION_FILES
    assert CHECKPOINT_SIZE == 262_593_305
    assert len(CHECKPOINT_SHA256) == 64
    assert SOURCE_REVISION == "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
    assert SPEECHBRAIN_REVISION == "2b3f4f44351fd08a627c4ab307de5c420351bc19"
    assert SPEECHBRAIN_VERSION == "1.0.3"
    assert EXECUTABLE_SPEECHBRAIN_FILES[0].endswith("sgmse_plus.py")
    assert EXECUTABLE_SPEECHBRAIN_FILES[1].endswith("enhancement.py")
    assert "class SGMSEEnhancement" in EXECUTABLE_SPEECHBRAIN_MARKERS[EXECUTABLE_SPEECHBRAIN_FILES[1]]
    blocked_audit, blocked = locked_distribution_source_audit(
        SPEECHBRAIN_VERSION,
        {"speechbrain/inference/enhancement.py"},
        {"speechbrain/inference/enhancement.py": "def enhance_batch(self, noisy): pass"},
    )
    assert blocked_audit["status"] == LOCKED_DISTRIBUTION_BLOCKER
    assert any(LOCKED_DISTRIBUTION_BLOCKER in message for message in blocked)
    complete_audit, complete_blockers = locked_distribution_source_audit(
        SPEECHBRAIN_VERSION,
        set(LOCKED_DISTRIBUTION_SOURCE_EXPECTATIONS),
        {
            "speechbrain/integrations/models/sgmse_plus.py": "class ScoreModel: pass",
            "speechbrain/inference/enhancement.py": "class SGMSEEnhancement: pass",
        },
    )
    assert complete_audit["status"] == "BLOCKED_LOCKED_DISTRIBUTION_SOURCE_AUDIT_MISMATCH"
    assert complete_blockers
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        (source / "sgmse" / "sampling").mkdir(parents=True)
        (source / "score.py").write_text("class ScoreModel: pass\nScoreModel(\n", encoding="utf-8")
        (source / "ncsnpp.py").write_text("NCSNpp ncsnpp_v2\n", encoding="utf-8")
        (source / "sde.py").write_text("OUVESDE SDERegistry\n", encoding="utf-8")
        (source / "sgmse/sampling/predictors.py").write_text("reverse_diffusion\n", encoding="utf-8")
        (source / "sgmse/sampling/correctors.py").write_text("ald CorrectorRegistry\n", encoding="utf-8")
        inventory, inventory_blockers = algorithm_source_inventory(source)
        assert not inventory_blockers
        assert inventory["files_by_role"]["sampler_predictor"][0]["path"] == "sgmse/sampling/predictors.py"
        assert inventory["files_by_role"]["sampler_corrector"][0]["path"] == "sgmse/sampling/correctors.py"
    facts, missing = config_facts("sample_rate: 16000\nn_fft: 510\nhop_length: 128\nwindow_type: hann\n")
    assert facts["sample_rate"] and facts["n_fft"] and facts["hop_length"] and facts["window_type"]
    assert "sampler_type" in missing
    assert tensor_contract_status({"safe_load_status": "BLOCKED_WEIGHTS_ONLY"}) == "AUTHENTICATED_MANIFEST_REQUIRED"
    assert tensor_contract_status({"safe_load_status": "SAFE_LOADED"}) == "AUTHENTICATED_MANIFEST_REQUIRED"
    assert tensor_contract_status({"safe_load_status": "SAFE_LOADED", "tensor_manifest": {"x": {}}}) == "SAFE_LOADED_MANIFEST"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--hyperparams", type=Path)
    parser.add_argument("--companion-dir", type=Path)
    parser.add_argument("--algorithm-source-dir", type=Path)
    parser.add_argument("--speechbrain-source-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.ckpt, args.hyperparams, args.companion_dir, args.algorithm_source_dir, args.speechbrain_source_dir, args.manifest)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("sgmse_prepare_checkpoint self-test: OK")
        return 0
    if None in (args.ckpt, args.hyperparams, args.companion_dir, args.algorithm_source_dir, args.speechbrain_source_dir, args.manifest):
        parser.error("normal runs require --ckpt, --hyperparams, --companion-dir, --algorithm-source-dir, --speechbrain-source-dir, and --manifest")
    return inspect(
        args.ckpt,
        args.hyperparams,
        args.companion_dir,
        args.algorithm_source_dir,
        args.speechbrain_source_dir,
        args.manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
