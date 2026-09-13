#!/usr/bin/env -S uv run --no-project --offline --python 3.12
"""Fail-closed license/native evidence gate for the Zonos reference closure.

The supplied publisher archive was collected at exact closure-equivalent Vokra
head c82ed76e.  This gate records its exact reviewable facts and requires an
explicit detached-head opt-in when that captured commit is not in the current
branch graph.  It never downloads, imports, or executes a reference package or
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("license_gate_manifest.json")
PROJECT = Path(__file__).with_name("pyproject.toml")
LOCK = Path(__file__).with_name("uv.lock")
CONSTRAINTS = Path(__file__).with_name("numpy-build-constraints.txt")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_DISPOSITIONS = {
    "REVIEWED_PERMISSIVE",
    "POLICY_REVIEW_REQUIRED",
    "NATIVE_BUNDLE_REVIEW_REQUIRED",
    "EMBEDDED_LICENSE_REVIEW_REQUIRED",
    "VENDORED_LICENSE_REVIEW_REQUIRED",
}


class GateError(ValueError):
    """Raised for malformed, stale, or incomplete evidence."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error


def require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise GateError(f"{label} schema is not exact")
    return value


def validate_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    manifest = load_json(path)
    require_keys(
        manifest,
        {"schema", "status", "publication", "current_closure", "evidence", "package_rows", "native_boundaries", "first_party_project", "approval"},
        "manifest",
    )
    if manifest["schema"] != "vokra-zonos-dependency-license-gate-v1":
        raise GateError("manifest schema is not exact")
    if manifest["status"] != "BLOCKED_UNREVIEWED_TRANSITIVE" or manifest["publication"] != "NO_UPLOAD":
        raise GateError("manifest is not fail-closed")

    current = require_keys(manifest["current_closure"], {"project_sha256", "lock_sha256", "constraints_sha256"}, "current closure")
    for key, path_value in (("project_sha256", PROJECT), ("lock_sha256", LOCK), ("constraints_sha256", CONSTRAINTS)):
        if not HEX64.fullmatch(str(current[key])) or current[key] != file_digest(path_value):
            raise GateError(f"current closure identity mismatch: {key}")

    evidence = require_keys(
        manifest["evidence"],
        {"root_name", "audit_json_sha256", "checksum_verify_log_sha256", "head", "candidate_scope_sha256", "lock_rows_sha256", "active_lock_rows_sha256", "installed_closure_sha256", "native_files_sha256", "publisher_files_sha256", "numpy_record_sha256", "publisher_manifest_sha256", "publisher_archive_dir", "counts", "project_sha256", "lock_sha256", "numpy_wheel_sha256", "constraints_sha256"},
        "evidence",
    )
    for key in ("audit_json_sha256", "candidate_scope_sha256", "lock_rows_sha256", "active_lock_rows_sha256", "installed_closure_sha256", "native_files_sha256", "publisher_files_sha256", "numpy_record_sha256", "publisher_manifest_sha256", "project_sha256", "lock_sha256", "numpy_wheel_sha256", "constraints_sha256"):
        if not HEX64.fullmatch(str(evidence[key])):
            raise GateError(f"evidence digest is malformed: {key}")
    if evidence["checksum_verify_log_sha256"] is not None and not HEX64.fullmatch(str(evidence["checksum_verify_log_sha256"])):
        raise GateError("checksum verification log digest is malformed")
    if not HEX40.fullmatch(str(evidence["head"])):
        raise GateError("evidence HEAD is malformed")
    counts = require_keys(evidence["counts"], {"lock_packages", "active_packages", "installed_distributions", "publisher_files", "native_files"}, "evidence counts")
    if counts != {"lock_packages": 38, "active_packages": 34, "installed_distributions": 34, "publisher_files": 59, "native_files": 41}:
        raise GateError("evidence counts are not the captured closure")
    if evidence["constraints_sha256"] != current["constraints_sha256"]:
        raise GateError("constraint identity is not shared by current and captured closure")
    if evidence["publisher_archive_dir"] != "publisher-archive":
        raise GateError("publisher archive directory is not exact")

    rows = manifest["package_rows"]
    if not isinstance(rows, list) or len(rows) != 34:
        raise GateError("manifest must contain exactly 34 active package rows")
    ids: list[tuple[str, str]] = []
    for row in rows:
        package = require_keys(row, {"name", "version", "spdx", "disposition"}, "package row")
        if not all(isinstance(package[key], str) and package[key].strip() for key in ("name", "version", "spdx")):
            raise GateError("package row has an empty identity or SPDX expression")
        if package["disposition"] not in PACKAGE_DISPOSITIONS:
            raise GateError(f"unknown package disposition: {package['disposition']}")
        identity = (package["name"], package["version"])
        if identity in ids:
            raise GateError(f"duplicate package row: {identity}")
        ids.append(identity)
    if ids != sorted(ids):
        raise GateError("package rows must be in exact name/version order")

    native = require_keys(manifest["native_boundaries"], {"status", "count", "sha256", "root_counts", "numpy_policy_sha256", "numpy_runtime_config_sha256", "numpy_no_blas", "torch_and_torchaudio"}, "native boundaries")
    if native != {
        "status": "REVIEW_REQUIRED",
        "count": 41,
        "sha256": evidence["native_files_sha256"],
        "root_counts": {"hf_xet": 1, "markupsafe": 1, "numpy": 21, "regex": 1, "safetensors": 1, "tokenizers": 1, "torchaudio": 2, "torch": 12, "yaml": 1},
        "numpy_policy_sha256": "1353c6c72ce22ff1af2885d1cecde7cde486d23b89bc80e7e5c7913c30606a7c",
        "numpy_runtime_config_sha256": "8ed54f6c33c562d3a462dc1570abb79b28dc034d2dd4e7a7be91444bc2051fb4",
        "numpy_no_blas": "PASS_NO_FORBIDDEN_BLAS",
        "torch_and_torchaudio": "BUNDLED_NATIVE_COMPONENTS_REQUIRE_OWNER_REVIEW",
    }:
        raise GateError("native boundary policy is not exact")

    first_party = require_keys(manifest["first_party_project"], {"name", "spdx", "pyproject_sha256", "root_license_path", "root_license_sha256", "disposition"}, "first-party project")
    if first_party != {
        "name": "vokra-zonos-v0-1-reference",
        "spdx": "Apache-2.0",
        "pyproject_sha256": current["project_sha256"],
        "root_license_path": "LICENSE",
        "root_license_sha256": "e1fef339bc7071bae13a31ca873f95e3157665d68342f558e1dd4974ca5a4cee",
        "disposition": "FIRST_PARTY_NOT_INDEPENDENT_SCOPE",
    }:
        raise GateError("first-party project identity or disposition is not exact")
    if file_digest(ROOT / "LICENSE") != first_party["root_license_sha256"]:
        raise GateError("root LICENSE identity drifted")

    approval = require_keys(manifest["approval"], {"status", "signer", "digest"}, "approval")
    if approval != {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None}:
        raise GateError("owner approval must remain empty and fail-closed")
    return manifest


def _safe_relative(path_text: str) -> PurePosixPath:
    path = PurePosixPath(path_text)
    if not path_text or path.is_absolute() or "\\" in path_text or ".." in path.parts:
        raise GateError(f"unsafe evidence path: {path_text!r}")
    return path


def verify_checksums(root: Path, checksum_log_sha256: str | None) -> None:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise GateError("evidence SHA256SUMS is missing or symlinked")
    expected: dict[PurePosixPath, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise GateError("evidence SHA256SUMS has a malformed row")
        relative = _safe_relative(match.group(2))
        if relative in expected:
            raise GateError(f"duplicate evidence checksum row: {relative}")
        expected[relative] = match.group(1)
    actual: set[PurePosixPath] = set()
    for path in root.rglob("*"):
        if path == sums_path:
            continue
        if path.is_symlink():
            raise GateError(f"evidence contains a symlink: {path}")
        if path.is_file():
            actual.add(PurePosixPath(path.relative_to(root).as_posix()))
    # The verification log is written after ``sha256sum -c`` and therefore
    # cannot authenticate itself.  It is the sole explicitly allowed
    # post-verification file; every other regular file must be listed.
    allowed_unhashed = {PurePosixPath("checksum-verify.log")} if checksum_log_sha256 is not None else set()
    if set(expected) != actual - allowed_unhashed or not (actual - set(expected)) <= allowed_unhashed:
        raise GateError("evidence SHA256SUMS does not cover exactly every authenticated file")
    for relative, expected_digest in expected.items():
        path = root / Path(*relative.parts)
        if file_digest(path) != expected_digest:
            raise GateError(f"evidence checksum mismatch: {relative}")
    checksum_log = root / "checksum-verify.log"
    if checksum_log_sha256 is not None and (not checksum_log.is_file() or file_digest(checksum_log) != checksum_log_sha256):
        raise GateError("checksum verification log identity mismatch")


def validate_evidence(manifest: dict[str, Any], root: Path, *, allow_ancestor_head: bool = False) -> str:
    evidence = manifest["evidence"]
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.name != evidence["root_name"]:
        raise GateError("evidence directory is not the exact captured root")
    verify_checksums(root, evidence["checksum_verify_log_sha256"])
    report_path = root / "dependency-audit.json"
    if file_digest(report_path) != evidence["audit_json_sha256"]:
        raise GateError("audit JSON SHA-256 does not match manifest")
    report = load_json(report_path)
    if not isinstance(report, dict) or report.get("schema") != "vokra-zonos-dependency-audit-v1" or report.get("status") != "BLOCKED_UNREVIEWED_TRANSITIVE" or report.get("publication") != "NO_UPLOAD":
        raise GateError("captured audit status/publication is not fail-closed")
    scope = report.get("candidate_scope")
    if not isinstance(scope, dict) or report.get("candidate_scope_sha256") != evidence["candidate_scope_sha256"] or scope.get("schema") != "vokra-zonos-dependency-approval-scope-v1" or digest(scope) != evidence["candidate_scope_sha256"]:
        raise GateError("candidate scope identity is not exact")
    if report.get("execution_identity", {}).get("expected_head") != evidence["head"] or report.get("execution_identity", {}).get("actual_head") != evidence["head"]:
        raise GateError("captured execution HEAD is not exact")
    for key in ("model_access", "source_access", "checkpoint_access"):
        if scope.get(key) is not False:
            raise GateError(f"captured evidence crosses acquisition boundary: {key}")
    lock = report.get("lock")
    if not isinstance(lock, dict) or lock.get("package_count") != evidence["counts"]["lock_packages"] or not isinstance(lock.get("rows"), list) or len(lock["rows"]) != evidence["counts"]["lock_packages"] or lock.get("rows_sha256") != evidence["lock_rows_sha256"] or digest(lock["rows"]) != evidence["lock_rows_sha256"]:
        raise GateError("captured lock row digest mismatch")
    installed = report.get("installed")
    if not isinstance(installed, dict) or installed.get("status") != "COLLECTED" or installed.get("failures") != []:
        raise GateError("captured installed closure is incomplete")
    active = installed.get("active_lock_packages")
    distributions = installed.get("installed_distributions")
    native = installed.get("native_files")
    publisher = installed.get("publisher_license_notice_files")
    if not all(isinstance(value, list) for value in (active, distributions, native, publisher)):
        raise GateError("captured package/native/publisher rows are incomplete")
    if len(active) != evidence["counts"]["active_packages"] or digest(active) != evidence["active_lock_rows_sha256"]:
        raise GateError("active lock package set/order/hash mismatch")
    if len(distributions) != evidence["counts"]["installed_distributions"] or digest(distributions) != evidence["installed_closure_sha256"]:
        raise GateError("installed distribution set/order/hash mismatch")
    if len(native) != evidence["counts"]["native_files"] or digest(native) != evidence["native_files_sha256"]:
        raise GateError("native evidence set/order/hash mismatch")
    native_policy = installed.get("numpy_native_policy")
    runtime_config = installed.get("numpy_runtime_config")
    if not isinstance(native_policy, dict) or native_policy.get("schema") != "vokra-zonos-numpy-native-policy-v1" or native_policy.get("status") != "PASS_NO_FORBIDDEN_BLAS" or native_policy.get("native_file_count") != 21 or native_policy.get("forbidden_boundaries") != [] or digest(native_policy) != manifest["native_boundaries"]["numpy_policy_sha256"]:
        raise GateError("NumPy native policy evidence is not exact")
    if not isinstance(runtime_config, dict) or runtime_config.get("status") != "PASS_NO_FORBIDDEN_BLAS" or runtime_config.get("forbidden_boundaries") != [] or digest(runtime_config) != manifest["native_boundaries"]["numpy_runtime_config_sha256"]:
        raise GateError("NumPy runtime configuration evidence is not exact")
    root_counts: dict[str, int] = {}
    for row in native:
        path_text = row.get("path") if isinstance(row, dict) else None
        if not isinstance(path_text, str) or "/" not in path_text:
            raise GateError("native evidence path is malformed")
        root_name = path_text.split("/", 1)[0]
        root_counts[root_name] = root_counts.get(root_name, 0) + 1
    if root_counts != manifest["native_boundaries"]["root_counts"]:
        raise GateError("native evidence root classification mismatch")
    if len(publisher) != evidence["counts"]["publisher_files"] or digest(publisher) != evidence["publisher_files_sha256"]:
        raise GateError("publisher evidence set/order/hash mismatch")

    expected_ids = [(row["name"], row["version"]) for row in manifest["package_rows"]]
    captured_ids = sorted((row.get("name"), row.get("version")) for row in active if isinstance(row, dict))
    if captured_ids != expected_ids:
        raise GateError("manifest package rows do not match captured active rows")
    archive = installed.get("publisher_archive")
    archive_root = root / evidence["publisher_archive_dir"]
    archive_manifest = archive_root / "manifest.json"
    if not isinstance(archive, dict) or file_digest(archive_manifest) != evidence["publisher_manifest_sha256"]:
        raise GateError("publisher archive manifest identity mismatch")
    archive_rows = load_json(archive_manifest)
    if archive.get("files") != archive_rows or archive.get("manifest_sha256") != evidence["publisher_manifest_sha256"] or not isinstance(archive_rows, list):
        raise GateError("publisher archive rows are not bound")
    expected_archive_paths: set[PurePosixPath] = set()
    for row in archive_rows:
        if not isinstance(row, dict) or not all(key in row for key in ("distribution", "path", "archive_path", "bytes", "sha256")):
            raise GateError("publisher archive row is malformed")
        relative = _safe_relative(row["archive_path"])
        if relative in expected_archive_paths:
            raise GateError(f"duplicate publisher archive path: {relative}")
        expected_archive_paths.add(relative)
        payload = archive_root / Path(*relative.parts)
        if not payload.is_file() or payload.is_symlink() or payload.stat().st_size != row["bytes"] or file_digest(payload) != row["sha256"]:
            raise GateError(f"publisher archive payload mismatch: {relative}")
    actual_archive_paths = {
        PurePosixPath(path.relative_to(archive_root).as_posix())
        for path in archive_root.rglob("*")
        if path.is_file() and path != archive_manifest
    }
    if actual_archive_paths != expected_archive_paths:
        raise GateError("publisher archive contains unexpected or missing files")
    archived = {(row.get("distribution"), row.get("path"), row.get("bytes"), row.get("sha256")) for row in archive_rows}
    for row in publisher:
        identity = (row.get("distribution"), row.get("path"), row.get("bytes"), row.get("sha256"))
        if row.get("status") == "MISSING" or identity not in archived:
            raise GateError("publisher bytes are not all present in the legal archive")
    scope_digests = {
        "installed_closure_sha256": evidence["installed_closure_sha256"],
        "native_files_sha256": evidence["native_files_sha256"],
        "publisher_files_sha256": evidence["publisher_files_sha256"],
        "numpy_record_sha256": evidence["numpy_record_sha256"],
        "numpy_native_policy_sha256": manifest["native_boundaries"]["numpy_policy_sha256"],
        "numpy_runtime_config_sha256": manifest["native_boundaries"]["numpy_runtime_config_sha256"],
        "publisher_archive_manifest_sha256": evidence["publisher_manifest_sha256"],
    }
    for key, expected in scope_digests.items():
        if not isinstance(expected, str) or not HEX64.fullmatch(expected) or scope.get(key) != expected:
            raise GateError(f"candidate scope digest is not bound: {key}")
    if installed.get("digests", {}).get("installed_closure_sha256") != evidence["installed_closure_sha256"] or installed.get("digests", {}).get("native_files_sha256") != evidence["native_files_sha256"] or installed.get("digests", {}).get("publisher_files_sha256") != evidence["publisher_files_sha256"] or installed.get("digests", {}).get("numpy_record_sha256") != evidence["numpy_record_sha256"]:
        raise GateError("installed digest map is not bound")
    if report.get("project", {}).get("pyproject_sha256") != evidence["project_sha256"] or report.get("project", {}).get("uv_lock_sha256") != evidence["lock_sha256"] or report.get("project", {}).get("constraints_sha256") != evidence["constraints_sha256"]:
        raise GateError("captured project identity is not exact")
    if report.get("project", {}).get("expected_direct_versions", {}).get("torch") != "2.11.0+cpu" or report.get("project", {}).get("expected_direct_versions", {}).get("torchaudio") != "2.11.0+cpu":
        raise GateError("captured Torch CPU pair is not the supplied evidence closure")
    wheel_identity = scope.get("wheel_identity")
    if not isinstance(wheel_identity, dict) or wheel_identity.get("sha256") != evidence["numpy_wheel_sha256"]:
        raise GateError("NumPy no-BLAS wheel identity is not bound")
    if scope.get("native_files_sha256") != evidence["native_files_sha256"] or scope.get("publisher_files_sha256") != evidence["publisher_files_sha256"] or scope.get("publisher_archive_manifest_sha256") != evidence["publisher_manifest_sha256"]:
        raise GateError("candidate scope does not bind native/publisher evidence")

    current_head = _current_head()
    if evidence["head"] != current_head:
        if not allow_ancestor_head:
            raise GateError("captured evidence HEAD differs from the current checkout; explicit ancestor mode is required")
        try:
            subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", evidence["head"], "HEAD"], check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            # The VAST worker may validate a clean side branch whose commit is
            # not in the current branch graph.  Keep this opt-in and require
            # that the exact commit is present and carries identical
            # project/lock/constraint bytes before accepting its evidence.
            _validate_detached_commit(evidence["head"], evidence)
    if evidence["project_sha256"] != manifest["current_closure"]["project_sha256"] or evidence["lock_sha256"] != manifest["current_closure"]["lock_sha256"]:
        raise GateError("captured dependency closure is not current")
    return "captured publisher/native evidence is valid for the current dependency closure; owner/legal sign-off remains required"


def _current_head() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, ValueError):
        return ""


def _validate_detached_commit(expected_head: str, evidence: dict[str, Any]) -> None:
    try:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{expected_head}^{{commit}}"], check=True, capture_output=True)
        for relative, expected_digest in (("tools/parity/zonos_v0_1_reference/pyproject.toml", evidence["project_sha256"]), ("tools/parity/zonos_v0_1_reference/uv.lock", evidence["lock_sha256"]), ("tools/parity/zonos_v0_1_reference/numpy-build-constraints.txt", evidence["constraints_sha256"])):
            payload = subprocess.run(["git", "-C", str(ROOT), "show", f"{expected_head}:{relative}"], check=True, capture_output=True).stdout
            if hashlib.sha256(payload).hexdigest() != expected_digest:
                raise GateError(f"captured HEAD closure file differs: {relative}")
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateError("captured evidence HEAD commit or closure files are unavailable") from error


def self_test() -> None:
    manifest = validate_manifest()
    assert len(manifest["package_rows"]) == 34
    rows = {row["name"]: row for row in manifest["package_rows"]}
    assert rows["filelock"]["spdx"] == "MIT"
    assert rows["packaging"]["spdx"] == "Apache-2.0 OR BSD-2-Clause"
    assert rows["regex"]["spdx"] == "Apache-2.0 AND CNRI-Python"
    assert rows["setuptools"]["disposition"] == "VENDORED_LICENSE_REVIEW_REQUIRED"
    assert rows["torch"]["disposition"] == "NATIVE_BUNDLE_REVIEW_REQUIRED"
    assert rows["tqdm"]["disposition"] == "POLICY_REVIEW_REQUIRED"
    assert manifest["status"] == "BLOCKED_UNREVIEWED_TRANSITIVE"
    assert manifest["publication"] == "NO_UPLOAD"
    assert manifest["approval"]["signer"] is None
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})
    assert _safe_relative("publisher-archive/manifest.json").parts == ("publisher-archive", "manifest.json")
    for unsafe in ("", "/tmp/file", "../file", "a\\b"):
        try:
            _safe_relative(unsafe)
        except GateError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {unsafe!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--allow-ancestor-head", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = validate_manifest(args.manifest)
        if args.self_test:
            self_test()
            if args.evidence_dir is not None:
                print(f"zonos license gate evidence self-test: PASS ({validate_evidence(manifest, args.evidence_dir, allow_ancestor_head=args.allow_ancestor_head)})")
            print("zonos license gate self-test: PASS")
            return 0
        if args.evidence_dir is None:
            print("zonos license gate: BLOCKED (owner/legal sign-off and captured evidence are required)")
            return 2
        print(f"zonos license gate: BLOCKED ({validate_evidence(manifest, args.evidence_dir, allow_ancestor_head=args.allow_ancestor_head)})")
        return 2
    except (GateError, OSError) as error:
        print(f"zonos license gate: FAIL-CLOSED ({error})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
