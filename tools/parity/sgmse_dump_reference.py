#!/usr/bin/env python3
"""Run an independent SGMSE score reference on VAST.

This tool is deliberately separate from the runtime and converter.  It imports
the exact pinned upstream SpeechBrain/SGMSE source, safe-loads the fixed
checkpoint, and writes deterministic score tensors plus a machine-readable
manifest.  It never writes a GGUF, publishes a model, or uses an unsafe pickle
fallback.  ``--self-test`` exercises only static contracts and performs no
Torch/model import.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
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
CHECKPOINT_TENSOR_COUNT = 647
CHECKPOINT_PARAMETER_COUNT = 65_590_822
CHECKPOINT_LICENSE_SPDX = "apache-2.0"
SOURCE_REVISION = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
SOURCE_REPOSITORY = "https://github.com/sp-uhh/sgmse.git"
SOURCE_LICENSE_SPDX = "mit"
SOURCE_LICENSE_SHA256 = "8748956d2e5afe9dfc8311188b4119dacc7c5293b0561e7cca7a21cf80e54caa"
SPEECHBRAIN_REVISION = "2b3f4f44351fd08a627c4ab307de5c420351bc19"
SPEECHBRAIN_REPOSITORY = "https://github.com/speechbrain/speechbrain.git"
SPEECHBRAIN_LICENSE_SPDX = "apache-2.0"
SPEECHBRAIN_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
PAD_SPEC_SOURCE_FILE = "sgmse/util/other.py"
PAD_SPEC_SOURCE_SHA256 = "092efb6e7da82d11c0afa555e5b124dd950e1216237e1c165a3aea8d4551ffd0"
PAD_SPEC_MARKERS = (
    "def pad_spec",
    "T = Y.size(3)",
    "if T%64 !=0:",
    "num_pad = 64-T%64",
)
PAD_SPEC_SEMANTICS = "pad_spec_pads_time_axis_to_a_multiple_of_64"
REFERENCE_FRAMES = 64
HYPERPARAMS_NAME = "hyperparams.yaml"
HYPERPARAMS_SHA256 = "5ebd87c6257537c3997c134b279d85cd7bebccce0e6d3fc68f7a36f15096aa51"
HYPERPARAMS_RAW = """sample_rate: 16000
n_fft: 510
hop_length: 128
window_type: hann

transform_type: exponent
spec_factor: 0.15
spec_abs_exponent: 0.5

sampling:
  sampler_type: pc
  predictor: reverse_diffusion
  corrector: ald
  N: 30
  corrector_steps: 1
  snr: 0.5

modules:
  score_model: !new:speechbrain.integrations.models.sgmse_plus.ScoreModel
    backbone: ncsnpp_v2
    sde: ouve
    theta: 1.5
    sigma_min: 0.05
    sigma_max: 0.5
    lr: 0.0001
    ema_decay: 0.999
    t_eps: 0.03
    num_eval_files: 5
    loss_type: score_matching
    loss_weighting: sigma^2
    network_scaling: 1/t
    c_in: '1'
    c_out: '1'
    c_skip: '0'
    sigma_data: 0.1
    l1_weight: 0.001
    pesq_weight: 0.0
    N: 30
    corrector_steps: 1
    sampler_type: pc
    snr: 0.5
    sr: 16000

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
  loadables:
     score_model_ema: !ref <modules[score_model]>
"""
REFERENCE_FORMAT = "vokra-sgmse-score-reference-v1"
CONSTRUCTION_EVIDENCE_FORMAT = "vokra-sgmse-construction-evidence-v1"
REFERENCE_BLOCKER = "BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE"
INSPECTION_EMA_BLOCKER = "BLOCKED_EMA_SELECTION_UNVERIFIED"
EMA_ROUTE_STATUS = "SOURCE_ROUTE_VERIFIED_STRICT_LOAD"

SCORE_MODEL_FILE = "speechbrain/integrations/models/sgmse_plus.py"
PARAMETER_TRANSFER_FILE = "speechbrain/utils/parameter_transfer.py"
SCORE_MODEL_MARKERS = ("class ScoreModel",)
PARAMETER_TRANSFER_MARKERS = (
    "class Pretrainer",
    "filename = name + PARAMFILE_EXT",
    "def load_collected",
)
ALGORITHM_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "score_model": ("class ScoreModel",),
    "ncsnpp": ("NCSNpp", "ncsnpp_v2"),
    "sde": ("OUVESDE", "SDERegistry"),
    "sampler_predictor": ("reverse_diffusion",),
    "sampler_corrector": ("ald", "CorrectorRegistry"),
}

# Values are not guessed model dimensions: they are the exact hyperparameter
# facts already authenticated by the inspection manifest.  Constructor
# introspection below rejects any source revision whose API does not expose the
# expected names instead of silently adapting a different model.
SCORE_MODEL_CONFIG: dict[str, Any] = {
    "backbone": "ncsnpp_v2",
    "sde": "ouve",
    "theta": 1.5,
    "sigma_min": 0.05,
    "sigma_max": 0.5,
    "lr": 0.0001,
    "ema_decay": 0.999,
    "t_eps": 0.03,
    "num_eval_files": 5,
    "loss_type": "score_matching",
    "loss_weighting": "sigma^2",
    "network_scaling": "1/t",
    # These are strings in the pinned YAML and are interpreted by the
    # upstream ScoreModel's _c_in/_c_out/_c_skip route.  Do not coerce them to
    # integers: that would select a different (and failing) source branch.
    "c_in": "1",
    "c_out": "1",
    "c_skip": "0",
    "sigma_data": 0.1,
    "l1_weight": 0.001,
    "pesq_weight": 0.0,
    "N": 30,
    "corrector_steps": 1,
    "sampler_type": "pc",
    "snr": 0.5,
    "sr": 16_000,
}
SCORE_MODEL_CONFIG_KEYS = frozenset(SCORE_MODEL_CONFIG)
EXPECTED_HYPERPARAM_FACTS = frozenset(
    {
        "backbone",
        "c_in",
        "c_out",
        "c_skip",
        "corrector",
        "corrector_steps",
        "ema_decay",
        "hop_length",
        "n_fft",
        "network_scaling",
        "predictor",
        "sample_rate",
        "sampler_type",
        "score_model",
        "sde",
        "sigma_data",
        "sigma_max",
        "sigma_min",
        "snr",
        "steps",
        "theta",
        "window_type",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_duplicate_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object members instead of silently keeping the last."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_revision(path: Path, expected: str, label: str) -> None:
    observed = git_revision(path)
    if observed != expected:
        raise ValueError(f"{label} revision {observed!r} != {expected!r}")


def require_clean_revision(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} checkout is missing or symlinked")
    require_revision(path, expected, label)
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError(f"{label} checkout is dirty")
    return {"path": str(path), "revision": expected, "clean": True}


def path_overlaps(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Rename a private directory without ever replacing a target.

    Linux VAST workers use ``renameat2(RENAME_NOREPLACE)``.  Darwin's
    ``renameatx_np(RENAME_EXCL)`` provides the same contract for local static
    self-tests; unsupported hosts fail closed rather than falling back to a
    TOCTOU-prone existence check followed by rename.
    """
    system = platform.system()
    libc = ctypes.CDLL(None, use_errno=True)
    if system == "Linux":
        function = getattr(libc, "renameat2", None)
        flags = 1  # RENAME_NOREPLACE
        dirfd = -100  # AT_FDCWD
    elif system == "Darwin":
        function = getattr(libc, "renameatx_np", None)
        flags = 4  # RENAME_EXCL
        dirfd = -2  # AT_FDCWD
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")
    if function is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        dirfd,
        os.fsencode(source),
        dirfd,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def require_absent_output(output_dir: Path, forbidden: tuple[Path, ...]) -> None:
    if not output_dir.is_absolute() or any(not path.is_absolute() for path in forbidden):
        raise ValueError("reference paths must be absolute")
    if output_dir.exists() or output_dir.is_symlink() or os.path.lexists(output_dir):
        raise ValueError("reference output directory must be absent (no-clobber)")
    if any(path_overlaps(output_dir, path) for path in forbidden):
        raise ValueError("reference output path overlaps a protected input tree")
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise ValueError("reference output parent must be an existing real directory")


def require_disjoint_inputs(paths: tuple[Path, ...]) -> None:
    for index, left in enumerate(paths):
        if not left.is_absolute():
            raise ValueError("reference paths must be absolute")
        for right in paths[index + 1 :]:
            if path_overlaps(left, right):
                raise ValueError("reference input paths overlap protected trees")


def require_vokra_checkout(vokra_root: Path) -> dict[str, Any]:
    if not vokra_root.is_dir() or vokra_root.is_symlink() or not (vokra_root / ".git").exists():
        raise ValueError("Vokra checkout is missing or symlinked")
    commit = git_revision(vokra_root)
    status = subprocess.run(
        ["git", "-C", str(vokra_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Vokra checkout is dirty")
    return {"path": str(vokra_root), "commit": commit, "clean": True}


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "model\t")):
                _, _, value = line.partition(":")
                if value.strip():
                    return value.strip()
    return platform.processor() or "unknown"


def verify_hyperparams_file(
    hyperparams: Path, inspection: dict[str, Any]
) -> dict[str, Any]:
    if hyperparams.name != HYPERPARAMS_NAME or not hyperparams.is_file() or hyperparams.is_symlink():
        raise ValueError("hyperparams.yaml is missing, symlinked, or has the wrong name")
    raw_bytes = hyperparams.read_bytes()
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if observed_sha256 != HYPERPARAMS_SHA256 or raw_bytes != HYPERPARAMS_RAW.encode():
        raise ValueError("hyperparams.yaml bytes differ from the reviewed fixed identity")
    manifest_hyperparams = inspection.get("hyperparams")
    if not isinstance(manifest_hyperparams, dict):
        raise ValueError("inspection hyperparams evidence is missing")
    if (
        manifest_hyperparams.get("filename") != HYPERPARAMS_NAME
        or manifest_hyperparams.get("sha256") != HYPERPARAMS_SHA256
        or manifest_hyperparams.get("raw") != HYPERPARAMS_RAW
    ):
        raise ValueError("inspection hyperparams evidence does not match fixed bytes")
    facts = manifest_hyperparams.get("facts")
    expected_facts = {name: True for name in EXPECTED_HYPERPARAM_FACTS}
    if facts != expected_facts:
        raise ValueError("inspection hyperparams facts are incomplete or mismatched")
    return {
        "path": str(hyperparams),
        "filename": HYPERPARAMS_NAME,
        "sha256": observed_sha256,
        "raw_sha256": observed_sha256,
        "raw_identity": "fixed_reviewed_bytes",
        "constructor_kwargs": dict(SCORE_MODEL_CONFIG),
        "constructor_keys": sorted(SCORE_MODEL_CONFIG_KEYS),
        "facts": expected_facts,
    }


def require_manifest_identity(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != "vokra-sgmse-voicebank-inspection-v1":
        raise ValueError("inspection manifest format is not the pinned SGMSE format")
    if manifest.get("model_repository") != MODEL_REPOSITORY:
        raise ValueError("inspection manifest model repository mismatch")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("inspection manifest model revision mismatch")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("inspection manifest checkpoint identity mismatch")
    if checkpoint.get("filename") != CHECKPOINT_NAME:
        raise ValueError("inspection manifest checkpoint filename mismatch")
    if checkpoint.get("size") != CHECKPOINT_SIZE or checkpoint.get("expected_size") != CHECKPOINT_SIZE:
        raise ValueError("inspection manifest checkpoint size mismatch")
    if checkpoint.get("expected_sha256") != CHECKPOINT_SHA256:
        raise ValueError("inspection manifest checkpoint expected SHA mismatch")
    if manifest.get("weight_license_spdx") != CHECKPOINT_LICENSE_SPDX:
        raise ValueError("inspection checkpoint license identity mismatch")
    source = manifest.get("algorithm_source")
    if not isinstance(source, dict):
        raise ValueError("inspection algorithm source evidence is missing")
    if any(
        source.get(key) != SOURCE_REVISION
        for key in ("expected_revision", "resolved_revision", "revision")
    ):
        raise ValueError("inspection algorithm source revision identity mismatch")
    if source.get("license_spdx") != SOURCE_LICENSE_SPDX:
        raise ValueError("inspection algorithm source license identity mismatch")
    if source.get("license_sha256") != SOURCE_LICENSE_SHA256:
        raise ValueError("inspection algorithm license hash mismatch")
    speechbrain = manifest.get("speechbrain_source")
    if not isinstance(speechbrain, dict):
        raise ValueError("inspection SpeechBrain source evidence is missing")
    if any(
        speechbrain.get(key) != SPEECHBRAIN_REVISION
        for key in ("expected_revision", "resolved_revision")
    ):
        raise ValueError("inspection SpeechBrain source revision identity mismatch")
    if speechbrain.get("license_spdx") != SPEECHBRAIN_LICENSE_SPDX:
        raise ValueError("inspection SpeechBrain license identity mismatch")
    if speechbrain.get("license_sha256") != SPEECHBRAIN_LICENSE_SHA256:
        raise ValueError("inspection SpeechBrain license hash mismatch")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("inspection manifest blockers must be a list")
    required_blockers = {REFERENCE_BLOCKER, INSPECTION_EMA_BLOCKER}
    observed_codes = {
        item.split(":", 1)[0]
        for item in blockers
        if isinstance(item, str) and ":" in item
    }
    missing = sorted(required_blockers - observed_codes)
    if missing:
        raise ValueError(f"inspection manifest lacks expected fail-closed blockers: {missing}")


def require_source_file(
    root: Path,
    relative: str,
    expected_hash: str | None,
    markers: tuple[str, ...],
) -> dict[str, Any]:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"pinned source file is missing or symlinked: {relative}")
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise ValueError(f"{relative} is missing source markers: {missing}")
    digest = sha256(path)
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"{relative} hash differs from inspection evidence")
    return {
        "path": relative,
        "sha256": digest,
        "size": path.stat().st_size,
        "markers": {marker: True for marker in markers},
    }


def verify_pad_spec_source(algorithm_source: Path) -> dict[str, Any]:
    evidence = require_source_file(
        algorithm_source,
        PAD_SPEC_SOURCE_FILE,
        PAD_SPEC_SOURCE_SHA256,
        PAD_SPEC_MARKERS,
    )
    evidence["semantics"] = PAD_SPEC_SEMANTICS
    return evidence


def verify_ema_route(
    hyperparams: Path,
    speechbrain_source: Path,
    loaded: dict[str, Any],
    inspection: dict[str, Any],
) -> dict[str, Any]:
    """Verify source-level EMA selection without trusting filename alone."""
    text = hyperparams.read_text(encoding="utf-8")
    expected_mapping = "score_model_ema: !ref <modules[score_model]>"
    if expected_mapping not in text:
        raise ValueError("hyperparams lacks the exact score_model_ema loadable mapping")
    source_files = inspection.get("speechbrain_source", {}).get("executable_files", {})
    score_expected = source_files.get(SCORE_MODEL_FILE, {}).get("sha256")
    if not isinstance(score_expected, str) or len(score_expected) != 64:
        raise ValueError("inspection lacks the reviewed ScoreModel source hash")
    score_file = require_source_file(
        speechbrain_source, SCORE_MODEL_FILE, score_expected, SCORE_MODEL_MARKERS
    )
    transfer_file = require_source_file(
        speechbrain_source, PARAMETER_TRANSFER_FILE, None, PARAMETER_TRANSFER_MARKERS
    )
    if loaded.get("container_path") != "<root>":
        raise ValueError("safe-loaded checkpoint is not the exact root tensor map")
    if loaded.get("safe_load_status") != "SAFE_LOADED":
        raise ValueError("checkpoint was not safe-loaded")
    if loaded.get("ema_extraction") != "UNVERIFIED":
        raise ValueError("unexpected pre-existing EMA status in inspection evidence")
    return {
        "status": EMA_ROUTE_STATUS,
        "loadable": "score_model_ema",
        "checkpoint_filename": CHECKPOINT_NAME,
        "hyperparams_mapping": expected_mapping,
        "selection": "root_tensor_map_loaded_into_pinned_ScoreModel",
        "parameter_load": "strict_state_dict",
        "source_files": {
            "score_model": score_file,
            "parameter_transfer": transfer_file,
        },
        "unsafe_pickle_fallback": False,
    }


def verify_algorithm_source(
    algorithm_source: Path, inspection: dict[str, Any]
) -> list[dict[str, Any]]:
    inventory = inspection.get("algorithm_source", {}).get("files_by_role")
    if not isinstance(inventory, dict):
        raise ValueError("inspection algorithm source inventory is missing")
    verified: list[dict[str, Any]] = []
    for role, markers in ALGORITHM_ROLE_MARKERS.items():
        rows = inventory.get(role)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"inspection algorithm role {role!r} is empty")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError(f"inspection algorithm role {role!r} has malformed row")
            expected_hash = row.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(
                    f"inspection algorithm role {role!r} lacks a reviewed source hash"
                )
            verified.append(
                {
                    "role": role,
                    **require_source_file(
                        algorithm_source,
                        row["path"],
                        expected_hash,
                        markers,
                    ),
                }
            )
    return verified


def _class_identity(module: Any) -> str:
    cls = module.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _source_identity(cls: type[Any], roots: tuple[tuple[Path, str, str], ...]) -> dict[str, Any]:
    """Record where an instantiated module class came from.

    Classes loaded from either pinned checkout receive a file hash.  Torch
    framework classes are recorded as runtime classes because their source is
    outside the two authenticated trees; they are never treated as SGMSE
    implementation evidence.
    """
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    try:
        source_file = inspect.getsourcefile(cls)
    except (OSError, TypeError):
        source_file = None
    if source_file:
        resolved = Path(source_file).resolve(strict=False)
        for root, repository, revision in roots:
            try:
                relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
            except (OSError, ValueError):
                continue
            if relative and not Path(relative).is_absolute():
                return {
                    "kind": "pinned_checkout",
                    "repository": repository,
                    "revision": revision,
                    "path": relative,
                    "sha256": sha256(resolved),
                }
    # Only PyTorch framework classes may be represented without a pinned
    # checkout file. A third-party/custom class outside both authenticated
    # trees must stop the VAST run rather than being mislabeled as runtime.
    if not qualified.startswith("torch."):
        raise ValueError(f"module class is outside authenticated source trees: {qualified}")
    return {"kind": "runtime", "module": qualified}


def _direct_tensor_rows(module: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameters = []
    for name, tensor in module.named_parameters(recurse=False):
        parameters.append({
            "name": name,
            "shape": [int(axis) for axis in tensor.shape],
            "dtype": str(tensor.dtype),
        })
    buffers = []
    for name, tensor in module.named_buffers(recurse=False):
        buffers.append({
            "name": name,
            "shape": [int(axis) for axis in tensor.shape],
            "dtype": str(tensor.dtype),
        })
    return sorted(parameters, key=lambda row: row["name"]), sorted(
        buffers, key=lambda row: row["name"]
    )


def collect_construction_evidence(
    model: Any,
    loaded: dict[str, Any],
    algorithm_source: Path,
    speechbrain_source: Path,
    checkpoint: Path,
    inspection_manifest: Path,
) -> dict[str, Any]:
    """Capture exact source construction facts without assigning native roles."""
    import torch

    roots = (
        (algorithm_source, SOURCE_REPOSITORY, SOURCE_REVISION),
        (speechbrain_source, SPEECHBRAIN_REPOSITORY, SPEECHBRAIN_REVISION),
    )
    named_modules = []
    owner_rows: dict[str, tuple[str, str, str]] = {}
    for path, module in model.named_modules():
        parameters, buffers = _direct_tensor_rows(module)
        for row in parameters:
            full_name = f"{path}.{row['name']}" if path else row["name"]
            owner_rows[full_name] = (path, "parameter", row["name"])
        for row in buffers:
            full_name = f"{path}.{row['name']}" if path else row["name"]
            owner_rows[full_name] = (path, "buffer", row["name"])
        named_modules.append({
            "path": path,
            "class": _class_identity(module),
            "source": _source_identity(module.__class__, roots),
            "direct_parameters": parameters,
            "direct_buffers": buffers,
        })
    named_modules.sort(key=lambda row: row["path"])

    state_rows = []
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state_dict entry is not a tensor: {name}")
        owner = owner_rows.get(name)
        if owner is None:
            raise ValueError(f"state_dict entry has no direct module owner: {name}")
        owner_path, owner_kind, owner_name = owner
        state_rows.append({
            "name": name,
            "shape": [int(axis) for axis in tensor.shape],
            "dtype": str(tensor.dtype),
            "owner_path": owner_path,
            "owner_kind": owner_kind,
            "owner_name": owner_name,
        })

    all_modules = [
        (path, module)
        for path, module in model.named_modules()
        if path.endswith("all_modules") and isinstance(module, torch.nn.ModuleList)
    ]
    if len(all_modules) != 1:
        raise ValueError(f"expected exactly one NCSN++ all_modules ModuleList, found {len(all_modules)}")
    all_modules_path, module_list = all_modules[0]
    module_rows = []
    for ordinal, module in enumerate(module_list):
        parameters, buffers = _direct_tensor_rows(module)
        module_rows.append({
            "ordinal": ordinal,
            "class": _class_identity(module),
            "source": _source_identity(module.__class__, roots),
            "direct_parameters": parameters,
            "direct_buffers": buffers,
        })
    packet = {
        "format": CONSTRUCTION_EVIDENCE_FORMAT,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "speechbrain_repository": SPEECHBRAIN_REPOSITORY,
            "speechbrain_revision": SPEECHBRAIN_REVISION,
        },
        "checkpoint": {
            "filename": checkpoint.name,
            "size": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "tensor_count": loaded.get("tensor_count"),
            "state_tensor_numel": loaded.get("parameter_count"),
        },
        "inspection_manifest_sha256": sha256(inspection_manifest),
        "state_dict": {
            "count": len(state_rows),
            "state_tensor_numel": sum(math.prod(row["shape"]) for row in state_rows),
            "parameter_row_numel": sum(
                math.prod(row["shape"])
                for row in state_rows
                if row["owner_kind"] == "parameter"
            ),
            "buffer_row_numel": sum(
                math.prod(row["shape"])
                for row in state_rows
                if row["owner_kind"] == "buffer"
            ),
            # These are true model totals (deduplicated by PyTorch for shared
            # aliases), kept separate from state_dict row totals above.
            "parameter_numel": sum(int(tensor.numel()) for tensor in model.parameters()),
            "buffer_numel": sum(int(tensor.numel()) for tensor in model.buffers()),
            "rows": state_rows,
        },
        "named_modules": named_modules,
        "ncsnpp_all_modules": {
            "path": all_modules_path,
            "count": len(module_rows),
            "rows": module_rows,
        },
    }
    packet["canonical_sha256"] = construction_evidence_sha256(packet)
    return packet


def construction_evidence_sha256(evidence: dict[str, Any]) -> str:
    unsigned = {
        key: value for key, value in evidence.items() if key != "canonical_sha256"
    }
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_direct_rows(rows: Any, label: str) -> set[str]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows are not a list")
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype"}:
            raise ValueError(f"{label} row is malformed")
        if not isinstance(row["name"], str) or not row["name"] or not isinstance(row["dtype"], str):
            raise ValueError(f"{label} row identity is malformed")
        if (
            not isinstance(row["shape"], list)
            or any(not isinstance(axis, int) or isinstance(axis, bool) or axis <= 0 for axis in row["shape"])
        ):
            raise ValueError(f"{label} row shape is malformed")
        names.append(row["name"])
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError(f"{label} rows are reordered or duplicated")
    return set(names)


def validate_construction_evidence(
    evidence: Any,
    *,
    expected_checkpoint: dict[str, Any],
    expected_inspection_sha256: str,
) -> None:
    """Validate the source construction packet before it is consumed offline."""
    if not isinstance(evidence, dict) or set(evidence) != {
        "format", "source", "checkpoint", "inspection_manifest_sha256",
        "state_dict", "named_modules", "ncsnpp_all_modules",
        "canonical_sha256",
    }:
        raise ValueError("construction evidence envelope is malformed")
    if evidence["format"] != CONSTRUCTION_EVIDENCE_FORMAT:
        raise ValueError("construction evidence format mismatch")
    source = evidence["source"]
    if not isinstance(source, dict) or source != {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "speechbrain_repository": SPEECHBRAIN_REPOSITORY,
        "speechbrain_revision": SPEECHBRAIN_REVISION,
    }:
        raise ValueError("construction source identity mismatch")
    if evidence["inspection_manifest_sha256"] != expected_inspection_sha256:
        raise ValueError("construction inspection identity mismatch")
    if evidence["checkpoint"] != expected_checkpoint:
        raise ValueError("construction checkpoint identity mismatch")
    unsigned = {key: value for key, value in evidence.items() if key != "canonical_sha256"}
    expected_digest = construction_evidence_sha256(unsigned)
    if evidence["canonical_sha256"] != expected_digest:
        raise ValueError("construction evidence canonical digest mismatch")

    modules = evidence["named_modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("construction named_modules are missing")
    paths = []
    module_by_path: dict[str, dict[str, Any]] = {}
    for row in modules:
        if not isinstance(row, dict) or set(row) != {
            "path", "class", "source", "direct_parameters", "direct_buffers"
        }:
            raise ValueError("construction named_module row is malformed")
        path = row["path"]
        if not isinstance(path, str) or path in module_by_path:
            raise ValueError("construction named_module paths are duplicated")
        if not isinstance(row["class"], str) or not row["class"]:
            raise ValueError("construction named_module class is malformed")
        source = row["source"]
        if (
            not isinstance(source, dict)
            or source.get("kind") == "runtime"
            and (
                set(source) != {"kind", "module"}
                or source.get("module") != row["class"]
                or not isinstance(source.get("module"), str)
                or not source["module"].startswith("torch.")
            )
        ):
            raise ValueError("construction named_module source/class identity is invalid")
        _validate_direct_rows(row["direct_parameters"], f"module {path} parameters")
        _validate_direct_rows(row["direct_buffers"], f"module {path} buffers")
        paths.append(path)
        module_by_path[path] = row
    if paths != sorted(paths) or paths[0] != "":
        raise ValueError("construction named_modules are reordered or lack the root")

    all_modules = evidence["ncsnpp_all_modules"]
    if not isinstance(all_modules, dict) or set(all_modules) != {"path", "count", "rows"}:
        raise ValueError("construction all_modules packet is malformed")
    all_path = all_modules["path"]
    if not isinstance(all_path, str) or all_path not in module_by_path or not all_path.endswith("all_modules"):
        raise ValueError("construction all_modules owner path is invalid")
    all_rows = all_modules["rows"]
    if not isinstance(all_rows, list) or all_modules["count"] != len(all_rows):
        raise ValueError("construction all_modules count mismatch")
    for ordinal, row in enumerate(all_rows):
        if not isinstance(row, dict) or set(row) != {
            "ordinal", "class", "source", "direct_parameters", "direct_buffers"
        } or row["ordinal"] != ordinal:
            raise ValueError("construction all_modules rows are missing, duplicated, or reordered")
        if not isinstance(row["class"], str) or not row["class"]:
            raise ValueError("construction all_modules class is malformed")
        source = row["source"]
        if (
            not isinstance(source, dict)
            or source.get("kind") == "runtime"
            and (
                set(source) != {"kind", "module"}
                or source.get("module") != row["class"]
                or not isinstance(source.get("module"), str)
                or not source["module"].startswith("torch.")
            )
        ):
            raise ValueError("construction all_modules source/class identity is invalid")
        parameters = _validate_direct_rows(
            row["direct_parameters"], f"all_modules[{ordinal}] parameters"
        )
        buffers = _validate_direct_rows(
            row["direct_buffers"], f"all_modules[{ordinal}] buffers"
        )
        member_path = f"{all_path}.{ordinal}"
        member = module_by_path.get(member_path)
        if member is None or member["class"] != row["class"] or member["source"] != row["source"]:
            raise ValueError("construction all_modules member path/class/source mismatch")
        if (
            _validate_direct_rows(member["direct_parameters"], f"module {member_path} parameters")
            != parameters
            or _validate_direct_rows(member["direct_buffers"], f"module {member_path} buffers")
            != buffers
        ):
            raise ValueError("construction all_modules member parameter/buffer mismatch")

    state = evidence["state_dict"]
    if not isinstance(state, dict) or set(state) != {
        "count", "state_tensor_numel", "parameter_row_numel", "buffer_row_numel",
        "parameter_numel", "buffer_numel", "rows",
    }:
        raise ValueError("construction state_dict packet is malformed")
    rows = state["rows"]
    if not isinstance(rows, list) or state["count"] != len(rows) or rows != sorted(rows, key=lambda row: row.get("name", "")):
        raise ValueError("construction state_dict rows are missing or reordered")
    if len(rows) != expected_checkpoint["tensor_count"]:
        raise ValueError("construction state_dict tensor count mismatch")
    expected_names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "name", "shape", "dtype", "owner_path", "owner_kind", "owner_name"
        }:
            raise ValueError("construction state_dict row is malformed")
        name = row["name"]
        if not isinstance(name, str) or name in expected_names:
            raise ValueError("construction state_dict names are duplicated")
        expected_names.add(name)
        if row["owner_path"] not in module_by_path or row["owner_kind"] not in {"parameter", "buffer"}:
            raise ValueError("construction state_dict owner is invalid")
        owner = module_by_path[row["owner_path"]]
        direct = owner["direct_parameters"] if row["owner_kind"] == "parameter" else owner["direct_buffers"]
        matches = [item for item in direct if item["name"] == row["owner_name"]]
        if len(matches) != 1 or matches[0]["shape"] != row["shape"] or matches[0]["dtype"] != row["dtype"]:
            raise ValueError("construction state_dict owner/shape/dtype mismatch")
        expected_full_name = f"{row['owner_path']}.{row['owner_name']}" if row["owner_path"] else row["owner_name"]
        if expected_full_name != name:
            raise ValueError("construction state_dict owner name mismatch")
    state_tensor_numel = sum(
        math.prod(row["shape"]) for row in rows
    )
    if state["state_tensor_numel"] != state_tensor_numel or state_tensor_numel != expected_checkpoint["state_tensor_numel"]:
        raise ValueError("construction state_dict tensor numel differs from checkpoint evidence")
    parameter_rows = sum(
        math.prod(row["shape"]) for row in rows if row["owner_kind"] == "parameter"
    )
    buffer_rows = sum(
        math.prod(row["shape"]) for row in rows if row["owner_kind"] == "buffer"
    )
    if state["parameter_row_numel"] != parameter_rows or state["buffer_row_numel"] != buffer_rows:
        raise ValueError("construction state_dict parameter/buffer row totals mismatch")
    if any(
        not isinstance(state[key], int) or isinstance(state[key], bool) or state[key] < 0
        for key in ("parameter_numel", "buffer_numel")
    ) or state["parameter_numel"] > parameter_rows or state["buffer_numel"] > buffer_rows:
        raise ValueError("construction true parameter/buffer totals are invalid")


def load_score_model(
    algorithm_source: Path,
    speechbrain_source: Path,
    checkpoint: Path,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Instantiate and strictly load the pinned ScoreModel."""
    import torch

    # Both pinned source trees are required: the SpeechBrain integration owns
    # the ScoreModel wrapper while the SGMSE tree owns NCSN++ and OUVE.
    sys.path.insert(0, str(speechbrain_source))
    sys.path.insert(0, str(algorithm_source))
    from speechbrain.integrations.models.sgmse_plus import ScoreModel

    signature = inspect.signature(ScoreModel)
    parameters = signature.parameters
    required = [
        parameter.name
        for parameter in parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and parameter.default is parameter.empty
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"pinned ScoreModel requires unknown config fields: {missing}")
    accepts_var_keyword = any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    positional_only = [
        name
        for name in config
        if name in parameters and parameters[name].kind is parameters[name].POSITIONAL_ONLY
    ]
    unknown = [name for name in config if name not in parameters and not accepts_var_keyword]
    if positional_only or unknown:
        raise ValueError(
            "pinned ScoreModel constructor API drifted: "
            f"positional_only={positional_only}, unknown={unknown}"
        )
    kwargs = dict(config)
    model = ScoreModel(**kwargs)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not loaded or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in loaded.items()
    ):
        raise ValueError("safe-loaded checkpoint is not a non-empty tensor state dict")
    incompatible = model.load_state_dict(loaded, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "strict SGMSE state_dict binding failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    # Repeat through strict=True so this gate cannot be weakened by a future
    # refactor of the diagnostic check above.
    model.load_state_dict(loaded, strict=True)
    model.eval()
    return model, {
        "class": f"{ScoreModel.__module__}.{ScoreModel.__qualname__}",
        "signature": str(signature),
        "constructor_kwargs": kwargs,
        "constructor_keys": sorted(kwargs),
        "tensor_count": len(loaded),
        "parameter_count": sum(int(value.numel()) for value in loaded.values()),
        "load": "torch.load(weights_only=True)+load_state_dict(strict=True)",
    }


def run_reference(
    source: Path,
    speechbrain_source: Path,
    checkpoint: Path,
    hyperparams: Path,
    inspection_manifest: Path,
    output_dir: Path,
    vokra_root: Path,
) -> dict[str, Any]:
    """Generate into a private sibling and atomically publish once complete."""
    all_paths = (
        source,
        speechbrain_source,
        checkpoint,
        hyperparams,
        inspection_manifest,
        output_dir,
        vokra_root,
    )
    if any(not path.is_absolute() for path in all_paths):
        raise ValueError("all reference paths must be absolute")
    if not inspection_manifest.is_file() or inspection_manifest.is_symlink():
        raise ValueError("inspection manifest is missing or symlinked")
    if hyperparams.parent.resolve(strict=False) != checkpoint.parent.resolve(strict=False):
        raise ValueError("checkpoint and hyperparams must share the reviewed inspection directory")
    require_disjoint_inputs(
        (vokra_root, inspection_manifest.parent, checkpoint.parent, source, speechbrain_source)
    )
    inspection = json.loads(
        inspection_manifest.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json,
    )
    require_manifest_identity(inspection)
    verify_hyperparams_file(hyperparams, inspection)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("checkpoint is missing or symlinked")
    source_tree = require_clean_revision(source, SOURCE_REVISION, "SGMSE source")
    speechbrain_tree = require_clean_revision(
        speechbrain_source, SPEECHBRAIN_REVISION, "SpeechBrain source"
    )
    vokra_tree = require_vokra_checkout(vokra_root)
    require_absent_output(
        output_dir,
        (
            vokra_root,
            inspection_manifest,
            inspection_manifest.parent,
            hyperparams,
            checkpoint,
            checkpoint.parent,
            source,
            speechbrain_source,
        ),
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        result = _run_reference_into(
            source,
            speechbrain_source,
            checkpoint,
            hyperparams,
            inspection_manifest,
            temporary,
            vokra_root,
            source_tree,
            speechbrain_tree,
            vokra_tree,
        )
        publish_reference_directory(temporary, output_dir, result)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def publish_reference_directory(
    temporary: Path, output_dir: Path, result: dict[str, Any]
) -> None:
    """Atomically publish a complete private directory without clobbering."""
    manifest_path = temporary / "manifest.json"
    if not manifest_path.is_file() or not result.get("artifacts"):
        raise ValueError("reference manifest or artifacts are missing")
    expected_names = {"manifest.json", "run.log"}
    for metadata in result["artifacts"].values():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            raise ValueError("reference artifact metadata is malformed")
        artifact_name = metadata["path"]
        if Path(artifact_name).name != artifact_name:
            raise ValueError("reference artifact path must be a direct child")
        expected_names.add(artifact_name)
        artifact = temporary / metadata["path"]
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"reference artifact is missing or symlinked: {artifact}")
        if sha256(artifact) != metadata["sha256"]:
            raise ValueError(f"reference artifact hash mismatch: {artifact.name}")
    if {entry.name for entry in temporary.iterdir()} != expected_names:
        raise ValueError("reference output contains an unexpected or missing file")
    if any(entry.is_symlink() or not entry.is_file() for entry in temporary.iterdir()):
        raise ValueError("reference output contains a non-regular file")
    if os.path.lexists(output_dir):
        raise ValueError("reference output appeared during atomic publish")
    atomic_rename_noreplace(temporary, output_dir)


def _run_reference_into(
    source: Path,
    speechbrain_source: Path,
    checkpoint: Path,
    hyperparams: Path,
    inspection_manifest: Path,
    output_dir: Path,
    vokra_root: Path,
    source_tree: dict[str, Any],
    speechbrain_tree: dict[str, Any],
    vokra_tree: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = json.loads(
        inspection_manifest.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json,
    )
    require_manifest_identity(inspection)
    hyperparams_evidence = verify_hyperparams_file(hyperparams, inspection)
    if checkpoint.name != CHECKPOINT_NAME:
        raise ValueError("checkpoint filename mismatch")
    if checkpoint.stat().st_size != CHECKPOINT_SIZE or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity differs from fixed inspection evidence")
    loaded_manifest = inspection.get("safe_load")
    if not isinstance(loaded_manifest, dict):
        raise ValueError("inspection safe_load evidence is missing")
    ema_route = verify_ema_route(
        hyperparams, speechbrain_source, loaded_manifest, inspection
    )
    algorithm_files = verify_algorithm_source(source, inspection)
    pad_spec_file = verify_pad_spec_source(source)
    # Pin the CPU execution knobs before importing or invoking the upstream
    # model.  The fixture remains an x86/VAST reference, but its environment is
    # explicit enough to diagnose a future platform-dependent drift.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260901)
    np.random.seed(20260901)
    model, model_evidence = load_score_model(
        source,
        speechbrain_source, checkpoint, SCORE_MODEL_CONFIG
    )
    if (
        model_evidence.get("tensor_count") != CHECKPOINT_TENSOR_COUNT
        or model_evidence.get("parameter_count") != CHECKPOINT_PARAMETER_COUNT
    ):
        raise ValueError("strict model load count differs from reviewed checkpoint evidence")
    # Capture the source construction route after strict loading. This packet
    # is evidence for a later offline mapping review only; it deliberately
    # contains no Vokra role labels and cannot unlock REVIEWED_* constants.
    model_evidence["construction_evidence"] = collect_construction_evidence(
        model,
        loaded_manifest,
        source,
        speechbrain_source,
        checkpoint,
        inspection_manifest,
    )
    # Imports and model construction can mutate Torch's precision policy and
    # consume RNG state. Re-establish the forward-time contract immediately
    # after those operations so fixture bytes are reproducible.
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260901)
    np.random.seed(20260901)
    frequency_bins = 510 // 2 + 1
    # The pinned util/other.py pad_spec contract pads the time axis to a
    # multiple of 64; 64 is the smallest source-authenticated fixture block.
    frames = REFERENCE_FRAMES
    # The pinned NCSN++ route accepts complex [batch, channel, frequency,
    # frame] tensors.  The source ScoreModel/Backbone contract fixes one
    # complex channel; do not silently squeeze or broadcast this dimension.
    real = torch.randn((1, 1, frequency_bins, frames), dtype=torch.float32)
    imaginary = torch.randn((1, 1, frequency_bins, frames), dtype=torch.float32)
    noisy = torch.complex(real, imaginary)
    condition = torch.complex(real.flip(-1), imaginary.flip(-1))
    timestep = torch.tensor([0.5], dtype=torch.float32)
    forward_signature = inspect.signature(model.forward)
    parameters = [
        parameter
        for parameter in forward_signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
    ]
    names = [parameter.name for parameter in parameters]
    # This positional order is the pinned upstream ScoreModel contract
    # (``forward(self, x_t, y, t)``); keyword invocation below keeps the
    # fixture explicit while this check prevents an API drift from passing.
    if names != ["x_t", "y", "t"]:
        raise ValueError(
            "pinned ScoreModel.forward signature is not the reviewed x_t,y,t route: "
            f"{forward_signature}"
        )
    with torch.no_grad():
        score = model(x_t=noisy, t=timestep, y=condition)
    if not isinstance(score, torch.Tensor) or not torch.is_complex(score):
        raise ValueError("pinned ScoreModel returned a non-complex tensor")
    if tuple(score.shape) != tuple(noisy.shape):
        raise ValueError(f"score shape {tuple(score.shape)} != input shape {tuple(noisy.shape)}")
    if not bool(torch.isfinite(score.real).all() and torch.isfinite(score.imag).all()):
        raise ValueError("reference score contains a non-finite value")

    arrays = {
        "input_noisy_real": noisy.real,
        "input_noisy_imag": noisy.imag,
        "input_condition_real": condition.real,
        "input_condition_imag": condition.imag,
        "score_real": score.real,
        "score_imag": score.imag,
    }
    artifacts: dict[str, Any] = {}
    for name, tensor in arrays.items():
        path = output_dir / f"{name}.f32"
        tensor.detach().cpu().contiguous().numpy().astype(np.float32).tofile(path)
        artifacts[name] = {
            "path": path.name,
            "dtype": "float32",
            "shape": list(tensor.shape),
            "count": int(tensor.numel()),
            "bytes": int(tensor.numel()) * 4,
            "sha256": sha256(path),
        }
    run_log_text = (
        "reference=vokra-sgmse-score-reference-v1\n"
        "status=REFERENCE_COMPLETE_NO_UPLOAD\n"
        "fixture_payload=retained_for_native_parity\n"
        f"source_revision={SOURCE_REVISION}\n"
        f"speechbrain_revision={SPEECHBRAIN_REVISION}\n"
        f"hyperparams_sha256={HYPERPARAMS_SHA256}\n"
        "load=torch.load(weights_only=True)+load_state_dict(strict=True)\n"
        "publication=NO_UPLOAD\n"
    )
    run_log_path = output_dir / "run.log"
    run_log_path.write_text(run_log_text, encoding="utf-8")
    result = {
        "format": REFERENCE_FORMAT,
        "status": "REFERENCE_COMPLETE_NO_UPLOAD",
        "publication": "NO_UPLOAD",
        "inspection_manifest_sha256": sha256(inspection_manifest),
        "inspection_manifest": {
            "path": str(inspection_manifest),
            "sha256": sha256(inspection_manifest),
        },
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "checkpoint": {
            "filename": checkpoint.name,
            "size": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "source": {
            **source_tree,
            "repository": "https://github.com/sp-uhh/sgmse.git",
            "license_spdx": SOURCE_LICENSE_SPDX,
            "license_sha256": SOURCE_LICENSE_SHA256,
            "files": algorithm_files,
            "pad_spec": pad_spec_file,
        },
        "speechbrain_source": {
            **speechbrain_tree,
            "repository": "https://github.com/speechbrain/speechbrain.git",
            "license_spdx": SPEECHBRAIN_LICENSE_SPDX,
            "license_sha256": SPEECHBRAIN_LICENSE_SHA256,
        },
        "hyperparams": hyperparams_evidence,
        "licenses": {
            "algorithm": {
                "spdx": inspection.get("algorithm_source", {}).get("license_spdx"),
                "sha256": inspection.get("algorithm_source", {}).get("license_sha256"),
            },
            "speechbrain": {
                "spdx": inspection.get("speechbrain_source", {}).get("license_spdx"),
                "sha256": inspection.get("speechbrain_source", {}).get("license_sha256"),
            },
            "checkpoint": inspection.get("weight_license_spdx"),
        },
        "ema_route": ema_route,
        "model": model_evidence,
        "vokra": {
            **vokra_tree,
            "uv_lock_sha256": sha256(vokra_root / "tools/parity/uv.lock"),
        },
        "runtime": {
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "platform_node": platform.node(),
            "cpu_model": cpu_model(),
            "nproc": os.cpu_count(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "determinism": {
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                "numpy_seed": 20260901,
                "torch_seed": 20260901,
            },
        },
        "input": {
            "seed": 20260901,
            "sample_rate": 16_000,
            "n_fft": 510,
            "frequency_bins": frequency_bins,
            "frames": frames,
            "forward_signature": str(forward_signature),
        },
        "artifacts": artifacts,
        "fixtures": "VAST_ONLY",
        "fixture_payload": "retained_for_native_parity",
        "run_log": {
            "path": run_log_path.name,
            "size": run_log_path.stat().st_size,
            "sha256": sha256(run_log_path),
        },
        "identity": {
            "reference_format": REFERENCE_FORMAT,
            "reference_tool": Path(__file__).name,
            "reference_tool_sha256": sha256(Path(__file__).resolve()),
            "self_test": "sgmse_dump_reference.py --self-test",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def self_test() -> None:
    assert CHECKPOINT_NAME == "score_model_ema.ckpt"
    assert CHECKPOINT_SIZE == 262_593_305
    assert len(CHECKPOINT_SHA256) == 64
    assert REFERENCE_FORMAT == "vokra-sgmse-score-reference-v1"
    assert REFERENCE_BLOCKER == "BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE"
    assert INSPECTION_EMA_BLOCKER == "BLOCKED_EMA_SELECTION_UNVERIFIED"
    assert EMA_ROUTE_STATUS == "SOURCE_ROUTE_VERIFIED_STRICT_LOAD"
    assert PARAMETER_TRANSFER_MARKERS[1] == "filename = name + PARAMFILE_EXT"
    assert SCORE_MODEL_CONFIG["backbone"] == "ncsnpp_v2"
    assert SCORE_MODEL_CONFIG["sde"] == "ouve"
    assert SCORE_MODEL_CONFIG["c_in"] == "1"
    assert SCORE_MODEL_CONFIG["c_out"] == "1"
    assert SCORE_MODEL_CONFIG["c_skip"] == "0"
    assert SCORE_MODEL_CONFIG["N"] == 30
    assert REFERENCE_FRAMES == 64
    assert PAD_SPEC_SOURCE_FILE == "sgmse/util/other.py"
    assert PAD_SPEC_SOURCE_SHA256 == "092efb6e7da82d11c0afa555e5b124dd950e1216237e1c165a3aea8d4551ffd0"
    assert PAD_SPEC_SEMANTICS == "pad_spec_pads_time_axis_to_a_multiple_of_64"
    assert PAD_SPEC_MARKERS[-1] == "num_pad = 64-T%64"
    assert "reviewed x_t,y,t route" in inspect.getsource(_run_reference_into)
    assert "torch.load(weights_only=True)" in inspect.getsource(load_score_model)
    assert '"bytes": int(tensor.numel()) * 4' in inspect.getsource(_run_reference_into)
    assert CONSTRUCTION_EVIDENCE_FORMAT == "vokra-sgmse-construction-evidence-v1"
    # A dependency-free packet model stands in for a tiny ModuleList. The
    # actual collector runs only after the pinned model is loaded on VAST;
    # these checks exercise the same strict packet contract locally.
    toy_source = {"kind": "runtime", "module": "torch.nn.Module"}
    toy_module_list_source = {
        "kind": "runtime", "module": "torch.nn.modules.container.ModuleList"
    }
    toy_modules = [
        {"path": "", "class": "torch.nn.Module", "source": toy_source, "direct_parameters": [], "direct_buffers": []},
        {"path": "backbone", "class": "torch.nn.Module", "source": toy_source, "direct_parameters": [], "direct_buffers": []},
        {"path": "backbone.all_modules", "class": "torch.nn.modules.container.ModuleList", "source": toy_module_list_source, "direct_parameters": [], "direct_buffers": []},
        {"path": "backbone.all_modules.0", "class": "torch.nn.Module", "source": toy_source, "direct_parameters": [
            {"name": "bias", "shape": [1], "dtype": "torch.float32"},
            {"name": "weight", "shape": [1], "dtype": "torch.float32"},
        ], "direct_buffers": []},
    ]
    toy_state_rows = [
        {"name": "backbone.all_modules.0.bias", "shape": [1], "dtype": "torch.float32", "owner_path": "backbone.all_modules.0", "owner_kind": "parameter", "owner_name": "bias"},
        {"name": "backbone.all_modules.0.weight", "shape": [1], "dtype": "torch.float32", "owner_path": "backbone.all_modules.0", "owner_kind": "parameter", "owner_name": "weight"},
    ]
    toy_packet = {
        "format": CONSTRUCTION_EVIDENCE_FORMAT,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "speechbrain_repository": SPEECHBRAIN_REPOSITORY, "speechbrain_revision": SPEECHBRAIN_REVISION},
        "checkpoint": {"filename": CHECKPOINT_NAME, "size": CHECKPOINT_SIZE, "sha256": CHECKPOINT_SHA256, "tensor_count": 2, "state_tensor_numel": 2},
        "inspection_manifest_sha256": "0" * 64,
        "state_dict": {"count": 2, "state_tensor_numel": 2, "parameter_row_numel": 2, "buffer_row_numel": 0, "parameter_numel": 2, "buffer_numel": 0, "rows": toy_state_rows},
        "named_modules": toy_modules,
        "ncsnpp_all_modules": {"path": "backbone.all_modules", "count": 1, "rows": [{"ordinal": 0, "class": "torch.nn.Module", "source": toy_source, "direct_parameters": toy_modules[-1]["direct_parameters"], "direct_buffers": []}]},
    }
    toy_packet["canonical_sha256"] = construction_evidence_sha256(toy_packet)
    validate_construction_evidence(
        toy_packet,
        expected_checkpoint=toy_packet["checkpoint"],
        expected_inspection_sha256="0" * 64,
    )
    import copy
    for variant, mutate in (
        ("duplicate", lambda packet: packet["named_modules"].append(copy.deepcopy(packet["named_modules"][-1]))),
        ("missing", lambda packet: packet["named_modules"].pop()),
        ("extra", lambda packet: packet["named_modules"].append({"path": "z", "class": "toy.Extra", "source": toy_source, "direct_parameters": [], "direct_buffers": []})),
        ("reordered module", lambda packet: packet["named_modules"].reverse()),
        ("reordered parameter", lambda packet: packet["named_modules"][-1]["direct_parameters"].reverse()),
        ("tampered parameter", lambda packet: packet["named_modules"][-1]["direct_parameters"][0].update(shape=[2])),
    ):
        candidate = copy.deepcopy(toy_packet)
        mutate(candidate)
        try:
            validate_construction_evidence(
                candidate,
                expected_checkpoint=toy_packet["checkpoint"],
                expected_inspection_sha256="0" * 64,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"toy construction {variant} mutation was accepted")
    for variant, mutate in (
        (
            "arbitrary runtime class",
            lambda packet: packet["named_modules"][0]["source"].update(module="evil.Custom"),
        ),
        (
            "tampered runtime class",
            lambda packet: packet["named_modules"][0].update(**{"class": "torch.nn.Linear"}),
        ),
    ):
        candidate = copy.deepcopy(toy_packet)
        mutate(candidate)
        candidate["canonical_sha256"] = construction_evidence_sha256(candidate)
        try:
            validate_construction_evidence(
                candidate,
                expected_checkpoint=toy_packet["checkpoint"],
                expected_inspection_sha256="0" * 64,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"toy construction {variant} was accepted")
    reference_source = inspect.getsource(_run_reference_into)
    model_load_end = reference_source.index("model, model_evidence = load_score_model")
    forward_reassert = reference_source.index(
        "torch.use_deterministic_algorithms(True)", model_load_end
    )
    fixture_randn = reference_source.index("real = torch.randn", forward_reassert)
    assert model_load_end < forward_reassert < fixture_randn
    try:
        json.loads('{"duplicate": 1, "duplicate": 2}', object_pairs_hook=reject_duplicate_json)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON object members were accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "source.py"
        path.write_text(
            "class ScoreModel:\nfilename = name + PARAMFILE_EXT\ndef load_collected(): pass\n",
            encoding="utf-8",
        )
        assert sha256(path)
        blocked = root / "blocked"
        write_blocked_manifest(blocked, ValueError("fixture self-test"))
        blocked_manifest = json.loads(
            (blocked / "manifest.json").read_text(),
            object_pairs_hook=reject_duplicate_json,
        )
        assert blocked_manifest["status"] == REFERENCE_BLOCKER
        assert blocked_manifest["fixture_generation"] == "NOT_RUN"
        (blocked / "sentinel").write_text("keep", encoding="utf-8")
        write_blocked_manifest(blocked, ValueError("must not overwrite"))
        assert (blocked / "sentinel").read_text(encoding="utf-8") == "keep"
        candidate = root / "candidate.tmp"
        candidate.mkdir()
        (candidate / "manifest.json").write_text("{}", encoding="utf-8")
        try:
            publish_reference_directory(candidate, root / "published", {"artifacts": {}})
        except ValueError:
            pass
        else:
            raise AssertionError("partial reference directory was published")
        shutil.rmtree(candidate)
        successful = root / "successful.tmp"
        successful.mkdir()
        successful_artifact = successful / "score_real.f32"
        successful_artifact.write_bytes(b"fixture")
        (successful / "run.log").write_text("fixture\n", encoding="utf-8")
        (successful / "manifest.json").write_text("{}", encoding="utf-8")
        published = root / "published"
        publish_reference_directory(
            successful,
            published,
            {"artifacts": {"score_real": {"path": successful_artifact.name, "sha256": sha256(successful_artifact)}}},
        )
        assert published.is_dir() and (published / "score_real.f32").is_file()
        existing = root / "existing"
        import threading

        target_ready = threading.Event()

        def install_concurrent_target() -> None:
            target_ready.wait()
            existing.mkdir()
            (existing / "sentinel").write_text("keep", encoding="utf-8")

        target_thread = threading.Thread(target=install_concurrent_target)
        target_thread.start()
        target_ready.set()
        target_thread.join()
        candidate.mkdir()
        artifact = candidate / "score_real.f32"
        artifact.write_bytes(b"fixture")
        (candidate / "manifest.json").write_text("{}", encoding="utf-8")
        try:
            publish_reference_directory(
                candidate,
                existing,
                {"artifacts": {"score_real": {"path": artifact.name, "sha256": sha256(artifact)}}},
            )
        except ValueError:
            pass
        else:
            raise AssertionError("existing output was replaced")
        assert (existing / "sentinel").read_text(encoding="utf-8") == "keep"
        shutil.rmtree(candidate)
        primitive_source = root / "primitive.tmp"
        primitive_source.mkdir()
        try:
            atomic_rename_noreplace(primitive_source, existing)
        except OSError as error:
            assert error.errno == errno.EEXIST
        else:
            raise AssertionError("no-replace primitive accepted an existing target")
        assert (existing / "sentinel").read_text(encoding="utf-8") == "keep"
        shutil.rmtree(primitive_source)
        protected = root / "protected"
        protected.mkdir()
        symlink = root / "symlink"
        symlink.symlink_to(protected, target_is_directory=True)
        try:
            require_absent_output(symlink, (protected,))
        except ValueError:
            pass
        else:
            raise AssertionError("symlink output path was accepted")


def write_blocked_manifest(
    output_dir: Path, error: Exception, forbidden: tuple[Path, ...] = ()
) -> None:
    """Persist a fail-closed result without replacing existing evidence."""
    temporary: Path | None = None
    try:
        if not output_dir.is_absolute() or any(not path.is_absolute() for path in forbidden):
            return
        if any(path_overlaps(output_dir, path) for path in forbidden):
            return
        if output_dir.exists() or output_dir.is_symlink() or os.path.lexists(output_dir):
            return
        if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
            return
        temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.blocker-", dir=output_dir.parent))
        manifest_path = temporary / "manifest.json"
        payload = (
            json.dumps(
                {
                    "format": REFERENCE_FORMAT,
                    "status": REFERENCE_BLOCKER,
                    "publication": "NO_UPLOAD",
                    "blockers": [REFERENCE_BLOCKER],
                    "execution": "BLOCKED",
                    "fixture_generation": "NOT_RUN",
                    "error": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        if os.path.lexists(output_dir):
            return
        atomic_rename_noreplace(temporary, output_dir)
        temporary = None
    except OSError:
        # The original exception and non-zero exit remain authoritative; do
        # not hide a failed run behind best-effort evidence persistence.
        return
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--speechbrain-source-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hyperparams", type=Path)
    parser.add_argument("--inspection-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--vokra-root", type=Path)
    args = parser.parse_args()
    values = (
        args.source_dir,
        args.speechbrain_source_dir,
        args.checkpoint,
        args.hyperparams,
        args.inspection_manifest,
        args.output_dir,
        args.vokra_root,
    )
    if args.self_test:
        if any(value is not None for value in values):
            parser.error("--self-test accepts no other arguments")
        self_test()
        print("sgmse_dump_reference self-test: OK")
        return 0
    if any(value is None for value in values):
        parser.error(
            "normal runs require --source-dir, --speechbrain-source-dir, "
            "--checkpoint, --hyperparams, --inspection-manifest, --output-dir, "
            "and --vokra-root"
        )
    try:
        result = run_reference(*values)  # type: ignore[arg-type]
    except Exception as error:  # noqa: BLE001 - preserve a loud VAST blocker
        if args.output_dir is not None:
            forbidden = tuple(
                value
                for value in (
                    args.vokra_root,
                    args.inspection_manifest,
                    args.inspection_manifest.parent if args.inspection_manifest else None,
                    args.hyperparams,
                    args.hyperparams.parent if args.hyperparams else None,
                    args.checkpoint,
                    args.checkpoint.parent if args.checkpoint else None,
                    args.source_dir,
                    args.speechbrain_source_dir,
                )
                if value is not None
            )
            write_blocked_manifest(args.output_dir, error, forbidden)
        print(
            json.dumps(
                {
                    "format": REFERENCE_FORMAT,
                    "status": REFERENCE_BLOCKER,
                    "publication": "NO_UPLOAD",
                    "execution": "BLOCKED",
                    "fixture_generation": "NOT_RUN",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
