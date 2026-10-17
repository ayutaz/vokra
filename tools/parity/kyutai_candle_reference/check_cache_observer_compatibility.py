#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Remote-only source compatibility runner for the Moshi cache observer seam.

This is a deliberately uncalled runner.  It accepts only the exact eight-file
application delivery produced by the earlier source application, replays the
frozen applier in a disposable workspace, applies the already-authenticated
cache observer patch to the three actual Moshi sources, and then performs a
bounded source compatibility build.  It never loads a model or claims runtime,
PCM, numerical parity, legal approval, or publication readiness.

The test module exercises the source and orchestration contracts with temporary
fixtures and mocked stages.  It does not invoke this runner's compiler,
applier, network, or model paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any

import cache_observer_patch as observer
import check_cpu_compatibility as legacy


MAX_DEADLINE = 900
TARGET = legacy.TARGET
REMOTE_OPT_IN = "VOKRA_KYUTAI_CANDLE_OBSERVER_REMOTE_LINUX"
WORKFLOW_HEAD_ENV = "VOKRA_CANDLE_WORKFLOW_HEAD"
STATUS_PENDING = "SOURCE_COMPATIBILITY_REVIEW_PENDING"
STATUS_FAILED = "SOURCE_COMPATIBILITY_FAILED"
SOURCE_FILES = (
    "moshi-core/src/asr.rs",
    "moshi-core/src/transformer.rs",
    "moshi-core/src/lm.rs",
)
EXPECTED_REGISTRY_SOURCES = 90
PATCHED_PATHS = frozenset(SOURCE_FILES)
# Immutable legacy application contract, deliberately reused rather than
# restated or relaxed for the cache-observer candidate.
APPLICATION_FILES = legacy.APPLICATION_FILES
FINAL_LOCK_SHA256 = legacy.FINAL_LOCK_SHA256
SOURCE_RECEIPT_SHA256 = legacy.SOURCE_RECEIPT_SHA256
CPU_FINGERPRINT = legacy.CPU_FINGERPRINT
ROOT_MANIFEST_SHA256 = legacy.ROOT_MANIFEST_SHA256
EXPECTED_GATES = {
    "runtime": "NOT_RUN",
    "default_none": "NOT_RUN",
    "batched": "NOT_RUN",
    "pcm": "NOT_RUN",
    "real_weight": "NOT_RUN",
    "cpu": "NOT_RUN",
    "metal": "NOT_RUN",
    "raw_kv_capture": "NOT_RUN",
    "independent_pcm_reference": "NOT_RUN",
    "dependency_license_owner_review": "BLOCKED_PENDING_REVIEW",
    "model_weights": "NOT_ACQUIRED",
    "publication": "NO_UPLOAD",
}
EXPECTED_OBSERVER_GATES = {
    "remote_build": "NOT_RUN",
    "runtime": "NOT_RUN",
    "cpu": "NOT_RUN",
    "metal": "NOT_RUN",
    "raw_kv_capture": "NOT_RUN",
    "independent_pcm_reference": "NOT_RUN",
    "dependency_license_owner_review": "BLOCKED_PENDING_REVIEW",
    "model_weights": "NOT_ACQUIRED",
    "publication": "NO_UPLOAD",
}
EXPECTED_OUTPUT_KEYS = frozenset(("asr", "transformer", "lm"))
EXPECTED_CANDLE_API_PROVENANCE = {
    "repository": "huggingface/candle",
    "tag": "0.9.1",
    "commit": "cd96fa80da255e34f7b16b4ff98b6a31d557201b",
    "path_in_vcs": "candle-nn",
    "official_url": "https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/candle-nn/src/kv_cache.rs",
    "archive_sha256": "c1980d53280c8f9e2c6cbe1785855d7ff8010208b46e21252b978badf13ad69d",
    "used_accessors": [
        "RotatingKvCache::k",
        "RotatingKvCache::v",
        "RotatingKvCache::k_cache",
        "RotatingKvCache::v_cache",
        "RotatingCache::offset",
        "RotatingCache::dim",
        "RotatingCache::max_seq_len",
        "RotatingCache::current_seq_len",
    ],
    "license_approval": None,
}
MAX_RECEIPT_ARTIFACT_BYTES = 2 * 1024 * 1024


class CompatibilityError(RuntimeError):
    """Raised when a source-only compatibility prerequisite is not proven."""


class BoundedCommandError(CompatibilityError):
    """Wrapper retaining the bounded command result for the receipt."""

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def _sha256(path: Path) -> str:
    return legacy._sha256(path)


def _evidence_total(directory: Path) -> int:
    return legacy._evidence_total(directory)


def _artifact_ids(result: dict[str, Any], *, package_identities: dict[str, str] | None = None) -> list[str]:
    return legacy._artifact_ids(result, package_identities=package_identities)


def _metadata_package_identities(path: Path) -> dict[str, str]:
    return legacy._metadata_package_identities(path)


def _json(path: Path) -> dict[str, Any]:
    return legacy._json(path)


def _remaining(deadline: float) -> float:
    return legacy._remaining(deadline)


def _verify_application(root: Path) -> dict[str, Any]:
    """Reuse the immutable legacy exact-eight-file verifier by normal import."""
    return legacy._verify_application(root)


def _verify_application_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return legacy._verify_application_receipt(receipt)


def _verify_fixed_files(root: Path) -> None:
    legacy._verify_fixed_files(root)


def _validate_cpu_metadata(
    path: Path,
    *,
    expected_fingerprint: str | None = legacy.CPU_FINGERPRINT,
    role_counts: tuple[int, int, int] = (90, 14, 1),
) -> dict[str, Any]:
    return legacy._validate_cpu_metadata(
        path, expected_fingerprint=expected_fingerprint, role_counts=role_counts
    )


def _sanitized_env(*, cargo_home: Path, target_dir: Path, offline: bool, home: Path | None = None) -> dict[str, str]:
    return legacy._sanitized_env(
        cargo_home=cargo_home, target_dir=target_dir, offline=offline, home=home
    )


def _bind_trusted_toolchain(env: dict[str, str]) -> None:
    legacy._bind_trusted_toolchain(env)


def _run_stage(
    stage: str,
    *,
    receipt: dict[str, Any],
    evidence_dir: Path,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    deadline: float,
    max_output: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Delegate bounded process handling to the immutable checker helper."""
    return legacy._run_stage(
        stage,
        receipt=receipt,
        evidence_dir=evidence_dir,
        argv=argv,
        cwd=cwd,
        env=env,
        deadline=deadline,
        max_output=max_output,
    )


def _artifact(path: Path, data: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "git_blob_sha1": observer.git_blob_sha1(data),
        "raw_sha1": observer.raw_sha1(data),
        "bytes": len(data),
        "sha256": observer.sha256_bytes(data),
    }


def _regular_file(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise CompatibilityError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise CompatibilityError(f"{label} is missing: {path}") from exc
        if stat.S_ISLNK(mode):
            raise CompatibilityError(f"{label} has a symlink component: {current}")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise CompatibilityError(f"{label} is not a regular file: {path}")


def _require_fresh_output_root(output: Path, evidence_dir: Path, *, forbidden_roots: tuple[Path, ...] = ()) -> None:
    """Reject reused output/evidence roots before launching any child."""
    if not output.is_absolute() or not evidence_dir.is_absolute():
        raise CompatibilityError("output and evidence paths must be absolute")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise CompatibilityError("output parent is missing or unsafe")
    output_resolved = output.resolve(strict=False)
    evidence_resolved = evidence_dir.resolve(strict=False)
    for forbidden in forbidden_roots:
        forbidden_resolved = forbidden.resolve(strict=False)
        if (
            output_resolved == forbidden_resolved
            or forbidden_resolved in output_resolved.parents
            or output_resolved in forbidden_resolved.parents
            or evidence_resolved == forbidden_resolved
            or forbidden_resolved in evidence_resolved.parents
            or evidence_resolved in forbidden_resolved.parents
        ):
            raise CompatibilityError(f"output/evidence overlaps forbidden root: {forbidden}")
    for candidate, label in ((output, "compatibility output"), (evidence_dir, "evidence root")):
        if candidate.exists() or candidate.is_symlink():
            raise CompatibilityError(f"{label} already exists; refusing reuse: {candidate}")
        current = Path(candidate.anchor)
        for part in candidate.parts[1:-1]:
            current /= part
            if current.is_symlink():
                raise CompatibilityError(f"{label} parent has a symlink component: {current}")


def _require_fresh_workspace_roots(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        candidate = root / name
        if candidate.exists() or candidate.is_symlink():
            raise CompatibilityError(f"fresh workspace root already exists: {candidate}")


def _bounded_regular_bytes(path: Path, label: str) -> bytes:
    """Read a small receipt/artifact only after regular-file validation."""
    _regular_file(path, label)
    try:
        expected_size = path.stat().st_size
    except OSError as exc:
        raise CompatibilityError(f"{label} metadata is unreadable: {path}") from exc
    if expected_size > MAX_RECEIPT_ARTIFACT_BYTES:
        raise CompatibilityError(f"{label} exceeds the 2 MiB bound: {path}")
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_RECEIPT_ARTIFACT_BYTES + 1)
            observed_size = os.fstat(handle.fileno()).st_size
    except OSError as exc:
        raise CompatibilityError(f"{label} is unreadable: {path}") from exc
    if len(data) > MAX_RECEIPT_ARTIFACT_BYTES or len(data) != expected_size or observed_size != expected_size:
        raise CompatibilityError(f"{label} changed during bounded read: {path}")
    return data


def _bounded_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_bounded_regular_bytes(path, label))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompatibilityError(f"{label} is not bounded JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CompatibilityError(f"{label} root must be an object: {path}")
    return value


def _authenticate_actual_inputs(workspace: Path) -> dict[str, bytes]:
    projection = workspace / "projection"
    paths = {
        "asr": projection / SOURCE_FILES[0],
        "transformer": projection / SOURCE_FILES[1],
        "lm": projection / SOURCE_FILES[2],
        "license": projection / "LICENSE",
    }
    return {
        "asr": observer._authenticated(
            paths["asr"], observer.SOURCE_BLOB_SHA1, observer.SOURCE_BYTES, "actual ASR source"
        ),
        "transformer": observer._authenticated(
            paths["transformer"],
            observer.TRANSFORMER_BLOB_SHA1,
            observer.TRANSFORMER_BYTES,
            "actual transformer source",
        ),
        "lm": observer._authenticated(
            paths["lm"], observer.LM_BLOB_SHA1, observer.LM_BYTES, "actual LM source"
        ),
        "license": observer._authenticated(
            paths["license"],
            observer.LICENSE_BLOB_SHA1,
            observer.LICENSE_BYTES,
            "actual root Rust LICENSE",
        ),
    }


def _row_map(root: Path, *, deadline: float) -> dict[str, dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise CompatibilityError(f"source snapshot root is missing or unsafe: {root}")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        _remaining(deadline)
        if path.is_symlink():
            raise CompatibilityError(f"source snapshot contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            rows[relative] = {"bytes": size, "sha256": _sha256(path)}
    return rows


def _registry_rows(metadata_path: Path, cargo_home: Path, *, deadline: float) -> dict[str, dict[str, dict[str, Any]]]:
    metadata = legacy._cargo_metadata(_json(metadata_path))
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise CompatibilityError("source metadata packages are malformed")
    registry_packages = [
        package
        for package in packages
        if isinstance(package, dict)
        and isinstance(package.get("source"), str)
        and package["source"].startswith("registry+")
    ]
    if len(registry_packages) != EXPECTED_REGISTRY_SOURCES:
        raise CompatibilityError(
            f"registry source count changed: {len(registry_packages)} != {EXPECTED_REGISTRY_SOURCES}"
        )
    registry_root = cargo_home / "registry" / "src"
    if not registry_root.is_dir() or registry_root.is_symlink():
        raise CompatibilityError("authenticated Cargo registry source root is missing")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for package in registry_packages:
        _remaining(deadline)
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise CompatibilityError("registry source identity is malformed")
        identity = f"{name}@{version}"
        matches: list[Path] = []
        for index_dir in sorted(registry_root.iterdir()):
            _remaining(deadline)
            if index_dir.is_symlink() or not index_dir.is_dir():
                raise CompatibilityError("registry source index contains an invalid entry")
            candidate = index_dir / f"{name}-{version}"
            if candidate.exists():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise CompatibilityError(f"registry source identity is unsafe: {identity}")
                matches.append(candidate)
        if len(matches) != 1 or identity in result:
            raise CompatibilityError(f"registry source identity is ambiguous or missing: {identity}")
        result[identity] = _row_map(matches[0], deadline=deadline)
    return result


def _source_snapshot_detailed(*, metadata_path: Path, workspace: Path, cargo_home: Path, deadline: float) -> dict[str, Any]:
    return {
        "projection": _row_map(workspace / "projection", deadline=deadline),
        "vendor": _row_map(workspace / "vendor", deadline=deadline),
        "registry_sources": _registry_rows(metadata_path, cargo_home, deadline=deadline),
    }


def _snapshot_digest(rows: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    registry = snapshot["registry_sources"]
    return {
        "projection": {"files": len(snapshot["projection"]), "sha256": _snapshot_digest(snapshot["projection"])},
        "vendor": {"files": len(snapshot["vendor"]), "sha256": _snapshot_digest(snapshot["vendor"])},
        "registry_sources": {
            "packages": len(registry),
            "sha256": _snapshot_digest(
                {identity: _snapshot_digest(rows) for identity, rows in sorted(registry.items())}
            ),
        },
    }


def _assert_only_source_changes(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    patched_sha256: dict[str, str],
    patched_bytes: dict[str, int] | None = None,
) -> None:
    if set(patched_sha256) != PATCHED_PATHS:
        raise CompatibilityError("patched source hash map must contain exactly three source paths")
    if patched_bytes is not None and set(patched_bytes) != PATCHED_PATHS:
        raise CompatibilityError("patched source byte map must contain exactly three source paths")
    if before["vendor"] != after["vendor"]:
        raise CompatibilityError("vendor source snapshot changed outside the observer sources")
    if before["registry_sources"] != after["registry_sources"]:
        raise CompatibilityError("registry source snapshot changed outside the observer sources")
    before_projection = before["projection"]
    after_projection = after["projection"]
    if set(before_projection) != set(after_projection):
        raise CompatibilityError("projection file set changed during observer patching")
    changed = {
        path for path in before_projection if before_projection[path] != after_projection[path]
    }
    if changed != PATCHED_PATHS:
        raise CompatibilityError(f"projection changed outside exact observer sources: {sorted(changed)}")
    for path, expected_sha256 in patched_sha256.items():
        if after_projection[path]["sha256"] != expected_sha256:
            raise CompatibilityError(f"patched source hash mismatch in projection: {path}")
        if patched_bytes is not None and after_projection[path]["bytes"] != patched_bytes[path]:
            raise CompatibilityError(f"patched source byte count mismatch in projection: {path}")


def _validate_observer_receipt(
    receipt_path: Path,
    *,
    source_paths: dict[str, Path],
    source_bytes: dict[str, bytes],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    receipt = _bounded_json(receipt_path, "cache observer receipt")
    if receipt.get("schema") != "vokra-moshi-cache-observer-source-only-v2":
        raise CompatibilityError("cache observer receipt schema changed")
    if receipt.get("patch") != observer.PATCH_NAME:
        raise CompatibilityError("cache observer receipt patch identity changed")
    source_records = {
        "source": ("asr", "rust/moshi-core/src/asr.rs", observer.SOURCE_URL, observer.SOURCE_BLOB_SHA1, observer.SOURCE_BYTES),
        "transformer_source": ("transformer", "rust/moshi-core/src/transformer.rs", observer.TRANSFORMER_URL, observer.TRANSFORMER_BLOB_SHA1, observer.TRANSFORMER_BYTES),
        "lm_source": ("lm", "rust/moshi-core/src/lm.rs", observer.LM_URL, observer.LM_BLOB_SHA1, observer.LM_BYTES),
    }
    for key, expected in source_records.items():
        record = receipt.get(key)
        if not isinstance(record, dict):
            raise CompatibilityError(f"receipt source binding missing: {key}")
        if key == "source" and (record.get("repository") != "kyutai-labs/moshi" or record.get("commit") != observer.UPSTREAM_COMMIT):
            raise CompatibilityError(f"receipt source provenance mismatch: {key}")
        if record.get("path") != expected[1] or record.get("official_raw_url") != expected[2]:
            raise CompatibilityError(f"receipt source URL/path mismatch: {key}")
        if record.get("authenticated_input") != str(source_paths[expected[0]]):
            raise CompatibilityError(f"receipt authenticated path mismatch: {key}")
        if record.get("git_blob_sha1") != expected[3] or record.get("bytes") != expected[4]:
            raise CompatibilityError(f"receipt source pin mismatch: {key}")
    license_record = receipt.get("root_rust_license")
    if not isinstance(license_record, dict) or license_record.get("authenticated_input") != str(source_paths["license"]):
        raise CompatibilityError("receipt license binding mismatch")
    if license_record.get("path") != "rust/LICENSE" or license_record.get("official_raw_url") != observer.LICENSE_URL:
        raise CompatibilityError("receipt license URL/path mismatch")
    if license_record.get("git_blob_sha1") != observer.LICENSE_BLOB_SHA1 or license_record.get("bytes") != observer.LICENSE_BYTES:
        raise CompatibilityError("receipt license pin mismatch")
    if "license_approval" not in receipt or receipt.get("license_approval") is not None:
        raise CompatibilityError("receipt top-level license approval is not explicitly null")
    candle = receipt.get("candle_api_provenance")
    if candle != EXPECTED_CANDLE_API_PROVENANCE:
        raise CompatibilityError("receipt Candle API provenance is incomplete or changed")
    moshi = receipt.get("moshi_api_provenance")
    if moshi != {
        "repository": "kyutai-labs/moshi",
        "commit": observer.UPSTREAM_COMMIT,
        "path": "rust/moshi-core/src/kv_cache.rs",
        "official_raw_url": observer.MOSHI_KV_CACHE_URL,
        "used_accessors": ["KvCache::current_seq_len", "KvCache::positions"],
    }:
        raise CompatibilityError("receipt Moshi API provenance changed")
    if receipt.get("source_sha256") != observer.sha256_bytes(source_bytes["asr"]):
        raise CompatibilityError("receipt ASR source hash mismatch")
    if receipt.get("transformer_source_sha256") != observer.sha256_bytes(source_bytes["transformer"]):
        raise CompatibilityError("receipt transformer source hash mismatch")
    if receipt.get("lm_source_sha256") != observer.sha256_bytes(source_bytes["lm"]):
        raise CompatibilityError("receipt LM source hash mismatch")
    if receipt.get("license_sha256") != observer.sha256_bytes(source_bytes["license"]):
        raise CompatibilityError("receipt license hash mismatch")
    reversible = receipt.get("reversible")
    if reversible != {"asr": True, "transformer": True, "lm": True}:
        raise CompatibilityError("receipt reversal proof is incomplete")
    if set(output_paths) != EXPECTED_OUTPUT_KEYS:
        raise CompatibilityError("patched output paths must be exactly ASR, transformer, and LM")
    if receipt.get("gates") != EXPECTED_OBSERVER_GATES:
        raise CompatibilityError("cache observer gate state is incomplete or changed")
    bindings = receipt.get("patched_outputs")
    if not isinstance(bindings, dict) or set(bindings) != EXPECTED_OUTPUT_KEYS:
        raise CompatibilityError("receipt output bindings are missing")
    for key, path in output_paths.items():
        record = bindings.get(key)
        if not isinstance(record, dict):
            raise CompatibilityError(f"receipt output binding missing: {key}")
        data = _bounded_regular_bytes(path, f"patched {key} output")
        actual = _artifact(path, data)
        for field in ("path", "git_blob_sha1", "raw_sha1", "bytes", "sha256"):
            if record.get(field) != actual[field]:
                raise CompatibilityError(f"receipt output binding mismatch: {key}.{field}")
    return receipt


def _replace_exact_sources(workspace: Path, patched: dict[str, bytes]) -> None:
    projection = workspace / "projection"
    if set(patched) != PATCHED_PATHS:
        raise CompatibilityError("replacement set is not exactly the three Moshi source files")
    for relative, data in patched.items():
        target = projection / relative
        _regular_file(target, f"workspace source {relative}")
        mode = stat.S_IMODE(target.stat().st_mode)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.observer-", dir=target.parent)
        temporary_path = Path(temporary)
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise


def _run_api_probe(
    *,
    root: Path,
    target_dir: Path,
    deadline: float,
    env: dict[str, str],
    receipt: dict[str, Any],
    evidence_dir: Path,
) -> dict[str, Any]:
    rlib = legacy._find_moshi_rlib(target_dir)
    host_deps = legacy._validate_host_dependency_dir(target_dir)
    probe = root / "cache-observer-compatibility-api-probe.rs"
    probe.write_text(
        """extern crate moshi;
use moshi::{asr, candle, conditioner, lm, mimi::Mimi, transformer, StreamMask, StreamTensor};
use asr::{AsrItemObservation, AsrObservation, State};

fn original_mimi_and_state_apis(
    state: &mut State,
    mimi: &mut Mimi,
    pcm: candle::Tensor,
    condition: Option<&conditioner::Condition>,
    mask: &StreamMask,
    stream: &StreamTensor,
) -> candle::Result<Vec<asr::AsrMsg>> {
    let _encode: fn(&mut Mimi, &StreamTensor, &StreamMask) -> candle::Result<StreamTensor> = Mimi::encode_step;
    let _decode: fn(&mut Mimi, &StreamTensor, &StreamMask) -> candle::Result<StreamTensor> = Mimi::decode_step;
    let _idx: fn(&State) -> usize = State::model_step_idx;
    let _ = mimi.encode_step(stream, mask)?;
    state.step_pcm(pcm, condition, mask, |_items, _tensor, _tensors| {})
}

fn observer_setters_and_callbacks() {
    let _v1_setter: fn(&mut State, Option<asr::AsrObserver>) = State::set_observer;
    let _v2_setter: fn(&mut State, Option<asr::AsrObserverV2>) = State::set_observer_v2;
    let _v1_type: Option<asr::AsrObserver> = None;
    let _v2_type: Option<asr::AsrObserverV2> = None;
    let _full_v2_callback: Option<asr::AsrObserverV2> = Some(Box::new(
        |_observation: &AsrObservation,
         _mask: &StreamMask,
         _text_logits: &candle::Tensor,
         _transformer_out: &candle::Tensor,
         _selected_tokens: &[u32]| -> candle::Result<()> { Ok(()) },
    ));
}

fn item_metadata_types(item: &AsrItemObservation) {
    let _: usize = item.step_idx;
    let _: u32 = item.text_token;
}

fn raw_cache_types(raw: &transformer::RawKvObservation) {
    let _: Option<&candle::Tensor> = raw.k.as_ref();
    let _: Option<&candle::Tensor> = raw.v.as_ref();
    let _: usize = raw.k_offset;
    let _: usize = raw.v_offset;
    let _: usize = raw.k_dim;
    let _: usize = raw.v_dim;
    let _: usize = raw.k_max_seq_len;
    let _: usize = raw.v_max_seq_len;
    let _: usize = raw.k_current_seq_len;
    let _: usize = raw.v_current_seq_len;
}

fn cache_getters_and_observation_fields(
    lm_model: &lm::LmModel,
    transformer_model: &transformer::StreamingTransformer,
    observation: &AsrObservation,
) -> candle::Result<()> {
    let _lm_getter: fn(&lm::LmModel) -> candle::Result<lm::LmObservation> = lm::LmModel::cache_observation;
    let _transformer_getter: fn(&transformer::StreamingTransformer) -> candle::Result<transformer::CacheObservation> = transformer::StreamingTransformer::cache_observation;
    let _lm_observation: &lm::LmObservation = &observation.lm;
    let cache = transformer_model.cache_observation()?;
    let _: &Vec<usize> = &cache.upstream_current_seq_len;
    let _: &Vec<Vec<usize>> = &cache.upstream_positions_for_zero_query;
    let _: &Vec<usize> = &cache.last_reset_pos;
    for raw in &cache.raw_kv {
        raw_cache_types(raw);
    }
    let _: usize = observation.global_model_step_idx;
    for item in &observation.items {
        item_metadata_types(item);
    }
    let _: &transformer::CacheObservation = &observation.lm.transformer;
    let _ = lm_model;
    Ok(())
}
""",
        encoding="utf-8",
    )
    output = root / "cache-observer-compatibility-api-probe.rlib"
    command = legacy.NAMESPACE_PREFIX + [
        "rustc",
        "--crate-type",
        "lib",
        "--edition",
        "2021",
        "--target",
        TARGET,
        str(probe),
        "--extern",
        f"moshi={rlib}",
        "-L",
        f"dependency={rlib.parent}",
        "-L",
        f"dependency={host_deps}",
        "-o",
        str(output),
    ]
    result = _run_stage(
        "api_probe",
        receipt=receipt,
        evidence_dir=evidence_dir,
        argv=command,
        cwd=root,
        env=env,
        deadline=deadline,
        max_output=512 * 1024,
    )
    if not output.is_file() or output.stat().st_size > 16 * 1024 * 1024:
        raise CompatibilityError("API probe output is missing or oversized")
    return result


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    data = json.dumps(receipt, ensure_ascii=True, sort_keys=True, indent=2).encode() + b"\n"
    if len(data) > 2 * 1024 * 1024:
        raise CompatibilityError("compatibility receipt exceeds 2 MiB")
    observer._write_exclusive(path, data, "compatibility receipt")


def _initial_receipt() -> dict[str, Any]:
    return {
        "schema": "vokra-moshi-cache-observer-source-compatibility-v1",
        "status": STATUS_PENDING,
        "source_only": True,
        "approval": None,
        "model_execution": "NOT_AUTHORIZED",
        "upload": "NO_UPLOAD",
        "target": TARGET,
        "commands": [],
        "gates": dict(EXPECTED_GATES),
    }


def check(application_root: Path, output: Path, deadline_seconds: int) -> int:
    started = time.monotonic()
    receipt = _initial_receipt()
    evidence_dir = output.parent / "evidence"
    can_write_failure_receipt = False
    try:
        repo = Path(__file__).resolve().parents[3]
        _require_fresh_output_root(output, evidence_dir, forbidden_roots=(application_root, repo))
        can_write_failure_receipt = True
        if not isinstance(deadline_seconds, int) or not 0 < deadline_seconds <= MAX_DEADLINE:
            raise CompatibilityError(f"deadline_seconds must be in 1..{MAX_DEADLINE}")
        deadline = started + deadline_seconds
        if not sys.platform.startswith("linux"):
            raise CompatibilityError("cache observer compatibility requires Linux")
        if os.environ.get(REMOTE_OPT_IN) != "1":
            raise CompatibilityError(f"missing explicit remote opt-in marker: {REMOTE_OPT_IN}")
        workflow_head = os.environ.get(WORKFLOW_HEAD_ENV)
        if not workflow_head or len(workflow_head) != 40 or any(char not in "0123456789abcdef" for char in workflow_head.lower()):
            raise CompatibilityError("workflow checkout head is malformed")
        receipt["workflow_checkout_head"] = workflow_head
        if not application_root.is_absolute():
            raise CompatibilityError("application-root must be an explicit absolute path")
        application_receipt = _verify_application(application_root)
        receipt["application_proofs"] = {
            "source_files_pre": application_receipt["source_proof_pre"]["files_checked"],
            "source_files_post": application_receipt["source_proof_post"]["files_checked"],
            "unchanged_files_pre": application_receipt["source_proof_pre"]["unchanged_files"],
            "unchanged_files_post": application_receipt["source_proof_post"]["unchanged_files"],
            "member_checks_pre": sum(item.get("members_checked", 0) for item in application_receipt["member_premetadata_checks"]),
            "member_checks_post": sum(item.get("members_checked", 0) for item in application_receipt["member_postchecks"]),
            "non_manifest_pre": sum(item.get("non_manifest_members", 0) for item in application_receipt["member_premetadata_checks"]),
            "non_manifest_post": sum(item.get("non_manifest_members", 0) for item in application_receipt["member_postchecks"]),
        }
        if receipt["application_proofs"] != {
            "source_files_pre": 23,
            "source_files_post": 23,
            "unchanged_files_pre": 21,
            "unchanged_files_post": 21,
            "member_checks_pre": 308,
            "member_checks_post": 308,
            "non_manifest_pre": 292,
            "non_manifest_post": 292,
        }:
            raise CompatibilityError("application proof counts changed")
        _verify_fixed_files(repo)
        if _sha256(application_root / "final.lock") != legacy.FINAL_LOCK_SHA256:
            raise CompatibilityError("authenticated application final.lock mismatch")
        with tempfile.TemporaryDirectory(prefix="vokra-candle-cache-observer-") as temporary:
            root = Path(temporary)
            git_env = _sanitized_env(cargo_home=root / "git-cargo-home", target_dir=root / "git-target", offline=True, home=root / "git-home")
            checkout = _run_stage(
                "checkout_head",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=["git", "rev-parse", "HEAD"],
                cwd=repo,
                env=git_env,
                deadline=deadline,
                max_output=128,
            )
            if checkout.get("stdout", "").strip() != workflow_head:
                raise CompatibilityError("workflow checkout head does not match checked-out HEAD")
            receipt["checked_out_head"] = workflow_head
            packet = root / "packet"
            applied_output = root / "application-output"
            workspace = root / "workspace"
            _require_fresh_workspace_roots(root, ("packet", "application-output", "workspace"))
            applier_budget = min(480, max(1, int(_remaining(deadline)) - 2))
            applier = repo / "tools/parity/kyutai_candle_reference/apply_paste_overlay.py"
            application_home = root / "application-home"
            application_home.mkdir()
            application_env = _sanitized_env(cargo_home=root / "cargo-home", target_dir=root / "target", offline=False, home=application_home)
            _bind_trusted_toolchain(application_env)
            application_env["VOKRA_CANDLE_REMOTE_LINUX"] = "1"
            application_env[REMOTE_OPT_IN] = "1"
            _run_stage(
                "source_application",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=[sys.executable, str(applier), str(packet), str(applied_output), str(workspace), "--deadline-seconds", str(applier_budget)],
                cwd=repo,
                env=application_env,
                deadline=deadline,
            )
            if _sha256(applied_output / "source_receipt.json") != legacy.SOURCE_RECEIPT_SHA256:
                raise CompatibilityError("replayed source receipt changed")
            replay_source = _json(applied_output / "source_receipt.json")
            if not isinstance(replay_source.get("files"), dict) or len(replay_source["files"]) != 23:
                raise CompatibilityError("replayed source receipt file manifest is incomplete")
            replay_application = _verify_application_receipt(_json(applied_output / "application_receipt.json"))
            receipt["replayed_application_proofs"] = {
                "source_pre": replay_application["source_proof_pre"]["files_checked"],
                "source_post": replay_application["source_proof_post"]["files_checked"],
                "unchanged_pre": replay_application["source_proof_pre"]["unchanged_files"],
                "unchanged_post": replay_application["source_proof_post"]["unchanged_files"],
                "members_pre": sum(item.get("members_checked", 0) for item in replay_application["member_premetadata_checks"]),
                "members_post": sum(item.get("members_checked", 0) for item in replay_application["member_postchecks"]),
                "non_manifest_pre": sum(item.get("non_manifest_members", 0) for item in replay_application["member_premetadata_checks"]),
                "non_manifest_post": sum(item.get("non_manifest_members", 0) for item in replay_application["member_postchecks"]),
            }
            if _sha256(applied_output / "final.lock") != legacy.FINAL_LOCK_SHA256:
                raise CompatibilityError("replayed final lock changed")
            if _sha256(workspace / "projection" / "Cargo.lock") != legacy.FINAL_LOCK_SHA256:
                raise CompatibilityError("fresh applier Cargo.lock mismatch")
            baseline = _validate_cpu_metadata(applied_output / "baseline_cpu_metadata.json", expected_fingerprint=None, role_counts=(104, 0, 1))
            overlay = _validate_cpu_metadata(applied_output / "overlay_cpu_metadata.json")
            receipt["cpu_metadata"] = {"baseline": baseline, "overlay": overlay}
            if _sha256(workspace / "projection" / "Cargo.toml") != legacy.ROOT_MANIFEST_SHA256:
                raise CompatibilityError("fresh projection root manifest mismatch")
            source_bytes = _authenticate_actual_inputs(workspace)
            source_paths = {
                "asr": workspace / "projection" / SOURCE_FILES[0],
                "transformer": workspace / "projection" / SOURCE_FILES[1],
                "lm": workspace / "projection" / SOURCE_FILES[2],
                "license": workspace / "projection" / "LICENSE",
            }
            cargo_home = workspace / "cargo-home"
            before = _source_snapshot_detailed(
                metadata_path=applied_output / "overlay_cpu_metadata.json",
                workspace=workspace,
                cargo_home=cargo_home,
                deadline=deadline,
            )
            stage = root / "observer-staging"
            stage.mkdir()
            outputs = {key: stage / f"{key}.rs" for key in ("asr", "transformer", "lm")}
            receipt_path = stage / "cache-observer-receipt.json"
            observer.apply(
                source_paths["asr"],
                source_paths["transformer"],
                source_paths["lm"],
                source_paths["license"],
                outputs["asr"],
                outputs["transformer"],
                outputs["lm"],
                receipt_path,
            )
            patched_bytes = {key: outputs[key].read_bytes() for key in outputs}
            cache_receipt = _validate_observer_receipt(
                receipt_path,
                source_paths=source_paths,
                source_bytes=source_bytes,
                output_paths=outputs,
            )
            if observer.revert_asr_bytes(patched_bytes["asr"]) != source_bytes["asr"]:
                raise CompatibilityError("ASR byte reversal failed")
            if observer.revert_transformer_bytes(patched_bytes["transformer"]) != source_bytes["transformer"]:
                raise CompatibilityError("transformer byte reversal failed")
            if observer.revert_lm_bytes(patched_bytes["lm"]) != source_bytes["lm"]:
                raise CompatibilityError("LM byte reversal failed")
            _replace_exact_sources(
                workspace,
                {
                    SOURCE_FILES[0]: patched_bytes["asr"],
                    SOURCE_FILES[1]: patched_bytes["transformer"],
                    SOURCE_FILES[2]: patched_bytes["lm"],
                },
            )
            after = _source_snapshot_detailed(
                metadata_path=applied_output / "overlay_cpu_metadata.json",
                workspace=workspace,
                cargo_home=cargo_home,
                deadline=deadline,
            )
            patched_hashes = {
                SOURCE_FILES[0]: observer.sha256_bytes(patched_bytes["asr"]),
                SOURCE_FILES[1]: observer.sha256_bytes(patched_bytes["transformer"]),
                SOURCE_FILES[2]: observer.sha256_bytes(patched_bytes["lm"]),
            }
            patched_sizes = {
                SOURCE_FILES[0]: len(patched_bytes["asr"]),
                SOURCE_FILES[1]: len(patched_bytes["transformer"]),
                SOURCE_FILES[2]: len(patched_bytes["lm"]),
            }
            _assert_only_source_changes(before, after, patched_sha256=patched_hashes, patched_bytes=patched_sizes)
            receipt["cache_observer_receipt"] = cache_receipt
            receipt["source_snapshot_before_build_state"] = "PATCHED"
            receipt["source_snapshot_before_build"] = _snapshot_summary(after)
            receipt["source_snapshot_pre_patch"] = _snapshot_summary(before)
            receipt["patched_source_artifacts"] = {
                path: {"bytes": patched_sizes[path], "sha256": patched_hashes[path]}
                for path in SOURCE_FILES
            }
            target_dir = root / "target-build"
            build_home = root / "build-home"
            build_home.mkdir()
            build_env = _sanitized_env(cargo_home=cargo_home, target_dir=target_dir, offline=True, home=build_home)
            _bind_trusted_toolchain(build_env)
            build_env[REMOTE_OPT_IN] = "1"
            parent_net_ns = os.readlink("/proc/self/ns/net")
            build_env["VOKRA_PARENT_NET_NS"] = parent_net_ns
            preflight = _run_stage(
                "namespace_preflight",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=legacy.NAMESPACE_PREFIX + ["/bin/sh", "-c", "test \"$(readlink /proc/self/ns/net)\" != \"$VOKRA_PARENT_NET_NS\" && test \"$(awk '/NoNewPrivs/ {print $2}' /proc/self/status)\" = 1 && rustc --version && cargo --version"],
                cwd=root,
                env=build_env,
                deadline=deadline,
                max_output=64 * 1024,
            )
            if "rustc 1.99.0" not in preflight.get("stdout", "") or "cargo 1.99.0" not in preflight.get("stdout", ""):
                raise CompatibilityError("namespace preflight did not select Rust/Cargo 1.99.0")
            receipt["isolation"] = {"network_namespace": "unshare --net verified", "no_new_privs": True, "runner_uid_preserved": True}
            manifest = workspace / "projection" / "moshi-core" / "Cargo.toml"
            build = _run_stage(
                "cargo_build",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=legacy.NAMESPACE_PREFIX + ["cargo", "build", "--lib", "--message-format=json-render-diagnostics", "--locked", "--offline", "--no-default-features", "--target", TARGET, "--manifest-path", str(manifest)],
                cwd=workspace / "projection",
                env=build_env,
                deadline=deadline,
            )
            post_build_snapshot = _source_snapshot_detailed(
                metadata_path=applied_output / "overlay_cpu_metadata.json",
                workspace=workspace,
                cargo_home=cargo_home,
                deadline=deadline,
            )
            if post_build_snapshot != after:
                raise CompatibilityError("source, vendor, or registry tree changed during build")
            receipt["source_snapshot_post_build"] = _snapshot_summary(post_build_snapshot)
            receipt["compiler_artifacts"] = _artifact_ids(
                build,
                package_identities=_metadata_package_identities(applied_output / "overlay_cpu_metadata.json"),
            )
            if _sha256(workspace / "projection" / "Cargo.lock") != legacy.FINAL_LOCK_SHA256:
                raise CompatibilityError("Cargo.lock changed during build")
            receipt["post_build_final_lock_sha256"] = legacy.FINAL_LOCK_SHA256
            post_metadata_command = legacy.NAMESPACE_PREFIX + ["cargo", "metadata", "--locked", "--offline", "--format-version", "1", "--filter-platform", TARGET, "--no-default-features", "--manifest-path", str(manifest)]
            post_metadata_result = _run_stage(
                "post_build_cpu_metadata",
                receipt=receipt,
                evidence_dir=evidence_dir,
                argv=post_metadata_command,
                cwd=workspace / "projection",
                env=build_env,
                deadline=deadline,
            )
            post_metadata_path = root / "post-build-cpu-metadata.json"
            try:
                post_metadata = json.loads(post_metadata_result["stdout"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise CompatibilityError("post-build CPU metadata is not JSON") from exc
            post_metadata_path.write_text(json.dumps({"metadata": post_metadata}), encoding="utf-8")
            receipt["post_build_cpu_metadata"] = _validate_cpu_metadata(post_metadata_path)
            _run_api_probe(root=root, target_dir=target_dir, deadline=deadline, env=build_env, receipt=receipt, evidence_dir=evidence_dir)
            if _sha256(workspace / "projection" / "Cargo.lock") != legacy.FINAL_LOCK_SHA256:
                raise CompatibilityError("Cargo.lock changed during API probe")
            final_snapshot = _source_snapshot_detailed(
                metadata_path=applied_output / "overlay_cpu_metadata.json",
                workspace=workspace,
                cargo_home=cargo_home,
                deadline=deadline,
            )
            if final_snapshot != after:
                raise CompatibilityError("source, vendor, or registry tree changed after build/probe")
            if receipt.get("gates") != EXPECTED_GATES or receipt.get("approval") is not None or receipt.get("upload") != "NO_UPLOAD":
                raise CompatibilityError("compatibility receipt gate state changed")
            receipt["compiler"] = {"status": "PASS", "target": TARGET, "offline": True, "library": True, "no_default_features": True}
            receipt["status"] = STATUS_PENDING
            receipt["evidence_bytes"] = _evidence_total(evidence_dir)
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _remaining(deadline)
            _write_receipt(output, receipt)
            return 0
    except Exception as exc:
        receipt["status"] = STATUS_FAILED
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)[:4000]}
        if isinstance(exc, (BoundedCommandError, legacy.BoundedCommandError)):
            failed = {key: value for key, value in exc.result.items() if key not in {"stdout_bytes", "stderr_bytes", "stdout", "stderr"}}
            failed["stdout"] = str(exc.result.get("stdout", ""))[-32768:]
            failed["stderr"] = str(exc.result.get("stderr", ""))[-32768:]
            receipt["failed_command"] = failed
        if evidence_dir.is_dir():
            try:
                receipt["evidence_bytes"] = _evidence_total(evidence_dir)
            except (CompatibilityError, legacy.CompatibilityError) as bound_error:
                receipt["evidence_bound_error"] = str(bound_error)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if can_write_failure_receipt:
            try:
                _write_receipt(output, receipt)
            except Exception:
                pass
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=MAX_DEADLINE)
    args = parser.parse_args(argv)
    if args.deadline_seconds <= 0 or args.deadline_seconds > MAX_DEADLINE:
        parser.error(f"--deadline-seconds must be in 1..{MAX_DEADLINE}")
    return check(args.application_root, args.output, args.deadline_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
