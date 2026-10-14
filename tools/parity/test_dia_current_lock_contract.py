#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Offline causal tests for the current Dia lock evidence contract."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata as package_metadata
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

from dia_1_6b_reference.lock_contract import (
    EXPECTED_ACTIVE_ROWS,
    EXPECTED_ALL_ROWS,
    GATE_STATUS,
    LICENSE_CLASSIFICATION,
    LICENSE_REVIEW_STATUS,
    LockContractError,
    SCHEMA,
    build_lock_audit,
    validate_lock_audit_packet,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "tools" / "parity" / "dia_1_6b_reference"
DUMPER = ROOT / "tools" / "parity" / "dia_1_6b_dump_reference.py"
VALIDATOR = ROOT / "tools" / "parity" / "dia_1_6b_validate_evidence.py"


def expect_reject(packet) -> None:
    try:
        validate_lock_audit_packet(packet, PROJECT)
    except LockContractError:
        return
    raise AssertionError("stale or tampered current lock packet was accepted")


def test_current_packet_is_exact_and_unreviewed() -> None:
    packet = build_lock_audit(PROJECT)
    validate_lock_audit_packet(packet, PROJECT)
    assert packet["schema"] == SCHEMA
    assert packet["status"] == GATE_STATUS
    assert packet["license_classification"] == LICENSE_CLASSIFICATION
    assert packet["license_review_status"] == LICENSE_REVIEW_STATUS
    assert packet["all_lock_rows"]["package_count"] == EXPECTED_ALL_ROWS == 31
    assert packet["active_linux_x86_64_rows"]["package_count"] == EXPECTED_ACTIVE_ROWS == 27
    all_names = {row["name"] for row in packet["all_lock_rows"]["rows"]}
    active_names = {row["name"] for row in packet["active_linux_x86_64_rows"]["rows"]}
    assert "vokra-dia-1-6b-reference" in all_names
    assert "vokra-dia-1-6b-reference" not in active_names
    assert "colorama" in all_names
    assert "colorama" not in active_names
    assert {"torch", "torchaudio"} <= active_names
    assert "_REVIEWED" not in json.dumps(packet, sort_keys=True)


def test_old_contracts_and_each_integrity_binding_reject() -> None:
    packet = build_lock_audit(PROJECT)
    stale_schema = copy.deepcopy(packet)
    stale_schema["schema"] = "vokra-dia-uv-lock-license-audit-v1"
    expect_reject(stale_schema)

    stale_hash = copy.deepcopy(packet)
    stale_hash["uv_lock_sha256"] = "58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
    expect_reject(stale_hash)

    stale_project_hash = copy.deepcopy(packet)
    stale_project_hash["pyproject_sha256"] = "fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
    expect_reject(stale_project_hash)

    stale_direct = copy.deepcopy(packet)
    stale_direct["direct_dependency_versions"].pop("torchaudio")
    stale_direct["direct_dependency_versions"]["torch"] = "2.6.0+cpu"
    expect_reject(stale_direct)

    wrong_full_count = copy.deepcopy(packet)
    wrong_full_count["all_lock_rows"]["package_count"] = 29
    expect_reject(wrong_full_count)

    wrong_active_digest = copy.deepcopy(packet)
    wrong_active_digest["active_linux_x86_64_rows"]["rows_sha256"] = "0" * 64
    expect_reject(wrong_active_digest)

    malformed_digest = copy.deepcopy(packet)
    malformed_digest["active_linux_x86_64_rows"]["rows_sha256"] = []
    expect_reject(malformed_digest)

    malformed_count = copy.deepcopy(packet)
    malformed_count["all_lock_rows"]["package_count"] = True
    expect_reject(malformed_count)

    wrong_membership = copy.deepcopy(packet)
    inactive_row = next(row for row in wrong_membership["all_lock_rows"]["rows"] if row["name"] == "colorama")
    wrong_membership["active_linux_x86_64_rows"]["rows"][0] = copy.deepcopy(inactive_row)
    expect_reject(wrong_membership)

    wrong_marker = copy.deepcopy(packet)
    wrong_marker["all_lock_rows"]["rows"][0]["resolution_markers"].append("sys_platform == 'darwin'")
    expect_reject(wrong_marker)

    wrong_source = copy.deepcopy(packet)
    wrong_source["all_lock_rows"]["rows"][0]["source"] = {"registry": "https://example.invalid/simple"}
    expect_reject(wrong_source)

    malformed_source_value = copy.deepcopy(packet)
    malformed_source_value["all_lock_rows"]["rows"][0]["source"] = {"registry": []}
    expect_reject(malformed_source_value)

    duplicate = copy.deepcopy(packet)
    duplicate["all_lock_rows"]["rows"][1] = copy.deepcopy(duplicate["all_lock_rows"]["rows"][0])
    expect_reject(duplicate)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_torchaudio_seam_is_legacy_pending_and_not_compatibility_evidence() -> None:
    dumper = _load_module(DUMPER, "dia_current_lock_dumper_target")
    validator = _load_module(VALIDATOR, "dia_current_lock_validator_target")
    previous = sys.modules.pop("torchaudio", None)
    try:
        descriptor = dumper.install_torchaudio_fail_closed_stub()
        assert descriptor == validator.TORCHAUDIO_SEAM
        assert descriptor["locked_distribution"] == "torchaudio==2.11.0+cpu"
        assert descriptor["installed_distribution_status"] == "UNKNOWN_NOT_CHECKED"
        assert descriptor["compatibility_evidence"] == "LEGACY_UNVALIDATED_NO_COMPATIBILITY_EVIDENCE"
        assert descriptor["audio_prompt"] is None
        validator.require_audio_dependency_seam(descriptor)
        old = dict(descriptor)
        old["installed_distribution"] = False
        try:
            validator.require_audio_dependency_seam(old)
        except ValueError:
            pass
        else:
            raise AssertionError("historical absence descriptor was accepted")
    finally:
        sys.modules.pop("torchaudio", None)
        if previous is not None:
            sys.modules["torchaudio"] = previous


def test_matching_local_bytes_are_required_without_expected_override() -> None:
    packet = build_lock_audit(PROJECT)
    with tempfile.TemporaryDirectory(prefix="dia-lock-contract-") as directory:
        candidate = Path(directory) / "dia_1_6b_reference"
        candidate.mkdir()
        shutil.copy2(PROJECT / "pyproject.toml", candidate / "pyproject.toml")
        shutil.copy2(PROJECT / "uv.lock", candidate / "uv.lock")
        (candidate / "uv.lock").write_bytes((candidate / "uv.lock").read_bytes() + b"\n")
        rejected = False
        try:
            validate_lock_audit_packet(packet, candidate)
        except LockContractError:
            rejected = True
        assert rejected


def _valid_identity(validator, packet: dict) -> dict:
    return {
        "project": "dia_1_6b_reference",
        "python": "3.12",
        "uv_lock_sha256": validator.REFERENCE_PROJECT_LOCK_SHA256,
        "pyproject_sha256": validator.REFERENCE_PROJECT_PYPROJECT_SHA256,
        "lock_schema": "uv-lock-v1-python312",
        "use_torch_compile": False,
        "dependency_audit": copy.deepcopy(packet),
        "dependency_license_audit": "BLOCKED_UNREVIEWED_TRANSITIVE",
        "expected_versions": dict(validator.DIRECT_DEPENDENCY_VERSIONS),
        "actual_versions": dict(validator.DIRECT_DEPENDENCY_VERSIONS),
    }


def test_actual_consumer_rejects_stale_project_and_lock_packets() -> None:
    dumper = _load_module(DUMPER, "dia_current_consumer_dumper_target")
    validator = _load_module(VALIDATOR, "dia_current_consumer_validator_target")
    packet = dumper.lock_dependency_audit(PROJECT / "uv.lock")
    identity = _valid_identity(validator, packet)
    validator.require_reference_project(identity)

    for field, stale in (
        ("uv_lock_sha256", "58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"),
        ("pyproject_sha256", "fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"),
    ):
        candidate = copy.deepcopy(identity)
        candidate[field] = stale
        try:
            validator.require_reference_project(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"stale consumer identity accepted: {field}")

    old_full = copy.deepcopy(identity)
    old_full["dependency_audit"]["all_lock_rows"]["package_count"] = 29
    try:
        validator.require_reference_project(old_full)
    except ValueError:
        pass
    else:
        raise AssertionError("historical 29-row consumer packet accepted")

    old_direct = copy.deepcopy(identity)
    old_direct["expected_versions"].pop("torchaudio")
    old_direct["actual_versions"].pop("torchaudio")
    old_direct["expected_versions"]["torch"] = "2.6.0+cpu"
    old_direct["actual_versions"]["torch"] = "2.6.0+cpu"
    try:
        validator.require_reference_project(old_direct)
    except ValueError:
        pass
    else:
        raise AssertionError("historical six-direct dependency map accepted")

    for section in ("all_lock_rows", "active_linux_x86_64_rows"):
        tampered = copy.deepcopy(identity)
        tampered["dependency_audit"][section]["rows"][0]["version"] += ".tampered"
        try:
            validator.require_reference_project(tampered)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tampered {section} consumer rows accepted")


def test_dumper_identity_binds_labeled_runtime_versions_only() -> None:
    dumper = _load_module(DUMPER, "dia_current_metadata_dumper_target")
    expected = dict(dumper.DIRECT_DEPENDENCY_VERSIONS)
    original = package_metadata.version
    calls: list[str] = []

    def labeled_version(name: str) -> str:
        calls.append(name)
        return expected[name]

    package_metadata.version = labeled_version
    try:
        identity = dumper.reference_project_identity()
    finally:
        package_metadata.version = original
    assert identity["expected_versions"] == expected
    assert identity["actual_versions"] == expected
    assert set(calls) == set(expected)

    def missing_version(name: str) -> str:
        if name == "torchaudio":
            raise package_metadata.PackageNotFoundError(name)
        return expected[name]

    package_metadata.version = missing_version
    try:
        try:
            dumper.reference_project_identity()
        except RuntimeError as error:
            assert "torchaudio" in str(error)
        else:
            raise AssertionError("missing labeled distribution was accepted")
    finally:
        package_metadata.version = original

    def mismatched_version(name: str) -> str:
        return "2.6.0+cpu" if name == "torch" else expected[name]

    package_metadata.version = mismatched_version
    try:
        try:
            dumper.reference_project_identity()
        except RuntimeError as error:
            assert "torch" in str(error)
        else:
            raise AssertionError("mismatched labeled distribution was accepted")
    finally:
        package_metadata.version = original

    try:
        dumper.lock_dependency_audit(PROJECT / "pyproject.toml")
    except RuntimeError as error:
        assert "canonical regular uv.lock" in str(error)
    else:
        raise AssertionError("wrong lock filename was accepted")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="count", default=0)
    args = parser.parse_args(argv)
    if args.self_test != 1:
        parser.error("exactly one --self-test is required")
    return args


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    test_current_packet_is_exact_and_unreviewed()
    test_old_contracts_and_each_integrity_binding_reject()
    test_torchaudio_seam_is_legacy_pending_and_not_compatibility_evidence()
    test_matching_local_bytes_are_required_without_expected_override()
    test_actual_consumer_rejects_stale_project_and_lock_packets()
    test_dumper_identity_binds_labeled_runtime_versions_only()
    print("dia current lock contract tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
