#!/usr/bin/env python3
"""Offline/model-free SpeechBrain Lang-ID closure audit.

This audit authenticates only bytes that are already present in the checkout
(the dedicated uv project and the fixed WAV fixture).  Upstream checkpoint and
loader identities are checked as immutable manifest records, never fetched or
executed.  The intentionally blocked report is suitable for a VAST source/
license audit before an owner approval and real-weight run exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import tomllib
from pathlib import Path

import preflight_gate as gate


SCHEMA = "speechbrain-lang-id-model-free-audit-v1"
DECISION = "BLOCKED"
BLOCKERS = [
    "BLOCKED_PACKAGE_LICENSE_REVIEW: locked Python package license and native-bundled review rows remain unresolved",
    "BLOCKED_MODEL_LICENSE_REVIEW: model/source/fixture license rows remain owner-review-required",
    "BLOCKED_OWNER_SIGNOFF: approval.status is OWNER_SIGNOFF_REQUIRED and no owner evidence is accepted",
    "BLOCKED_REAL_WEIGHT_PARITY: checkpoint execution, conversion, CPU parity, and Metal parity were not run in model-free mode",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def fail(message: str) -> None:
    print(f"SpeechBrain Lang-ID model-free audit: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def regular_file(path: Path, label: str) -> None:
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:-1]:
            current /= part
            if current.is_symlink() or not current.is_dir():
                fail(f"{label} has an unsafe or non-directory ancestor")
        st = os.lstat(path)
    except OSError as error:
        fail(f"{label} is unreadable: {error}")
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        fail(f"{label} must be a regular non-symlink file")


def safe_output(path: Path) -> None:
    if not path.is_absolute() or any(part in ("", ".", "..") for part in path.parts[1:]):
        fail("audit output must be an absolute lexical-safe path")
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            fail("audit output has an unsafe or non-directory ancestor")
    if os.path.lexists(path):
        fail("audit output must not already exist")


def publish_no_replace(path: Path, payload: bytes) -> None:
    """Publish a small report without exposing a partial file or clobbering."""
    safe_output(path)
    parent = path.parent
    fd = -1
    temporary: Path | None = None
    try:
        for attempt in range(32):
            candidate = parent / f".{path.name}.tmp-{os.getpid()}-{attempt}"
            try:
                fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
            except FileExistsError:
                continue
            temporary = candidate
            break
        if fd < 0 or temporary is None:
            raise OSError("could not reserve unique audit temp file")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("zero-byte report write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = None
    except Exception:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def validate_manifest(manifest: object, lock_data: dict[str, object]) -> dict[str, object]:
    if not isinstance(manifest, dict):
        fail("manifest top-level value must be an object")
    if set(manifest) != gate.MANIFEST_KEYS:
        fail("manifest schema has missing or extra keys")
    if manifest.get("gate_version") != gate.GATE_VERSION:
        fail("manifest gate version differs")
    if manifest.get("upstream_repo") != gate.REPO or manifest.get("upstream_revision") != gate.REVISION:
        fail("fixed upstream repository/revision differs")
    if manifest.get("publication_decision") != "NO_UPLOAD":
        fail("publication decision is not NO_UPLOAD")
    if not gate.lock_artifacts_complete(lock_data):
        fail("uv lock artifact URLs/hashes/sizes are incomplete")
    rows = gate.lock_rows(lock_data)
    if manifest.get("package_rows") != rows or manifest.get("package_rows_sha256") != gate.canonical(rows):
        fail("locked package rows differ from the canonical uv.lock inventory")
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows) or manifest.get("package_review_rows_sha256") != gate.canonical(reviews):
        fail("package review rows are missing or tampered")
    expected = {(row["name"], row["version"]): row for row in rows}
    seen: set[tuple[str, str]] = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != gate.PACKAGE_REVIEW_SCHEMA:
            fail("package review row schema is not exact")
        key = (review.get("name"), review.get("version"))
        if key in seen or key not in expected or review.get("source") != expected[key]["source"]:
            fail("package review identity/source is not bound to uv.lock")
        seen.add(key)
    if seen != set(expected):
        fail("package review inventory does not cover uv.lock")
    licenses = manifest.get("license_rows")
    if not isinstance(licenses, list) or [row.get("id") for row in licenses if isinstance(row, dict)] != gate.LICENSE_IDS or any(not isinstance(row, dict) or set(row) != gate.LICENSE_SCHEMA for row in licenses) or manifest.get("license_rows_sha256") != gate.canonical(licenses):
        fail("license review rows are missing or tampered")
    payload = manifest.get("payload_identities")
    if payload != list(gate.IDENTITIES.values()) or manifest.get("payload_identities_sha256") != gate.canonical(payload):
        fail("recorded native payload identity table differs")
    source = manifest.get("source_identities")
    if source != list(gate.SOURCE_IDENTITIES) or manifest.get("source_identities_sha256") != gate.canonical(source):
        fail("recorded official source identity table differs")
    if manifest.get("contract") != gate.CONTRACT:
        fail("recorded native contract differs")
    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict) or fixture.get("path") != gate.FIXTURE_PATH or fixture.get("bytes") != gate.FIXTURE_BYTES or fixture.get("sha256") != gate.FIXTURE_SHA256:
        fail("recorded fixture identity differs")
    approval = manifest.get("approval")
    if not isinstance(approval, dict) or set(approval) != gate.APPROVAL_KEYS or approval.get("status") != "OWNER_SIGNOFF_REQUIRED" or approval.get("signer") is not None or approval.get("digest") is not None:
        fail("approval state is not the expected fail-closed owner-signoff state")
    return manifest


def audit(lock_path: Path, project_path: Path, manifest_path: Path, fixture_path: Path, output: Path | None) -> int:
    for path, label in ((lock_path, "uv.lock"), (project_path, "pyproject.toml"), (manifest_path, "license manifest"), (fixture_path, "fixture")):
        regular_file(path, label)
    lock_bytes = lock_path.read_bytes()
    project_bytes = project_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    fixture_bytes = fixture_path.read_bytes()
    if digest(lock_bytes) != gate.LOCK_SHA256 or digest(project_bytes) != gate.PROJECT_SHA256:
        fail("lock/project bytes differ from code-bound closure")
    if len(fixture_bytes) != gate.FIXTURE_BYTES or digest(fixture_bytes) != gate.FIXTURE_SHA256:
        fail("fixture bytes differ from the fixed provenance identity")
    try:
        lock_data = tomllib.loads(lock_bytes.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        fail(f"uv.lock is not valid TOML: {error}")
    try:
        manifest_value = load_json(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        fail(f"manifest JSON is unreadable: {error}")
    manifest = validate_manifest(manifest_value, lock_data)
    scope_sha256 = gate.scope(manifest)
    report = {
        "schema": SCHEMA,
        "decision": DECISION,
        "status": "BLOCKED_OWNER_REVIEW",
        "evidence_stage": "MODEL_FREE_SOURCE_LICENSE_FIXTURE_AUDIT",
        "no_upload": True,
        "upstream_repo": manifest["upstream_repo"],
        "upstream_revision": manifest["upstream_revision"],
        "scope_sha256": scope_sha256,
        "scope_status": "PENDING_REVIEW_NOT_OWNER_SIGNABLE",
        "scope_basis": "CURRENT_MANIFEST_INCLUDING_UNRESOLVED_REVIEW_ROWS",
        "owner_signoff_eligible": False,
        "checkout_inputs": {
            "uv_lock": {"bytes": len(lock_bytes), "sha256": digest(lock_bytes), "status": "VERIFIED"},
            "pyproject": {"bytes": len(project_bytes), "sha256": digest(project_bytes), "status": "VERIFIED"},
            "license_manifest": {"bytes": len(manifest_bytes), "sha256": digest(manifest_bytes), "status": "VERIFIED"},
            "fixture": {"path": gate.FIXTURE_PATH, "bytes": len(fixture_bytes), "sha256": digest(fixture_bytes), "status": "VERIFIED"},
        },
        "native_payload_identities": {"status": "RECORDED_IDENTITY_ONLY_NO_MATERIALIZATION", "rows": manifest["payload_identities"]},
        "official_source_identities": {"status": "RECORDED_IDENTITY_ONLY_NO_FETCH", "rows": manifest["source_identities"]},
        "source_license_bytes": {
            "status": "BLOCKED_NOT_MATERIALIZED",
            "reason": "No primary source/license byte bundle is present in the checkout; this route does not fetch or infer one",
            "required_before_approval": ["official source tree/license bytes", "locked package license bytes", "fixture provenance/license evidence"],
        },
        "contract": manifest["contract"],
        "publication": "NO_UPLOAD",
        "blockers": BLOCKERS,
        "next_vast_command": "run-speechbrain-lang-id-validation.sh --approval-evidence <owner-approved-json> --approval-evidence-sha256 <sha256> --expected-head <clean-40-hex-head>",
    }
    encoded = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    if output is not None:
        try:
            publish_no_replace(output, encoded)
        except OSError as error:
            fail(f"cannot publish model-free audit report: {error}")
    print(f"SpeechBrain Lang-ID model-free audit: BLOCKED_OWNER_REVIEW scope_sha256={scope_sha256} scope_status=PENDING_REVIEW_NOT_OWNER_SIGNABLE")
    if output is not None:
        print(f"audit_report={output}")
    for blocker in BLOCKERS:
        print(blocker)
    return 2


def self_test() -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = ["model_free_audit.py", "--self-test", "--output", "/tmp/should-not-be-used.json"]
        try:
            main()
        except SystemExit as error:
            assert error.code == 2
        else:
            raise SystemExit("self-test CLI accepted mixed arguments")
    finally:
        sys.argv = original_argv
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        lock = root / "uv.lock"
        project = root / "pyproject.toml"
        manifest_path = root / "manifest.json"
        fixture = root / "jfk-30s.wav"
        output = root / "audit.json"
        lock.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nwheels=[{url="https://example.test/demo-1.whl",hash="sha256:' + 'a' * 64 + '",size=1}]\n', encoding="utf-8")
        project.write_bytes(b"project")
        fixture.write_bytes(b"wav")
        old = (gate.LOCK_SHA256, gate.PROJECT_SHA256, gate.FIXTURE_BYTES, gate.FIXTURE_SHA256, gate.IDENTITIES, gate.SOURCE_IDENTITIES, gate.LICENSE_IDS, gate.CONTRACT, gate.FIXTURE_PATH)
        try:
            gate.LOCK_SHA256 = digest(lock.read_bytes())
            gate.PROJECT_SHA256 = digest(b"project")
            gate.FIXTURE_BYTES = 3
            gate.FIXTURE_SHA256 = digest(b"wav")
            gate.IDENTITIES = {"demo.ckpt": {"path": "demo.ckpt", "role": "checkpoint", "bytes": 3, "sha256": digest(b"abc"), "status": "REVIEWED"}}
            gate.SOURCE_IDENTITIES = ({"path": "source.py", "role": "official-loader-config", "bytes": 3, "sha256": digest(b"src"), "status": "REVIEWED"},)
            gate.LICENSE_IDS = ["model-apache"]
            gate.CONTRACT = {"n_mels": 60, "embedding_dim": 256, "class_count": 107, "sample_rate": 16000}
            rows = [{"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "dependencies": []}]
            reviews = [{"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "license": "UNRESOLVED", "status": "UNRESOLVED", "native_bundled_review": "OWNER_REVIEW_REQUIRED"}]
            licenses = [{"id": "model-apache", "license": "UNRESOLVED", "status": "UNRESOLVED", "conclusion": "OWNER_REVIEW_REQUIRED", "native_bundled_review": "OWNER_REVIEW_REQUIRED"}]
            manifest = {"gate_version": 1, "lock_sha256": gate.LOCK_SHA256, "project_sha256": gate.PROJECT_SHA256, "package_rows": rows, "package_rows_sha256": gate.canonical(rows), "package_review_rows": reviews, "package_review_rows_sha256": gate.canonical(reviews), "license_rows": licenses, "license_rows_sha256": gate.canonical(licenses), "upstream_repo": gate.REPO, "upstream_revision": gate.REVISION, "payload_identities": list(gate.IDENTITIES.values()), "payload_identities_sha256": gate.canonical(list(gate.IDENTITIES.values())), "source_identities": list(gate.SOURCE_IDENTITIES), "source_identities_sha256": gate.canonical(list(gate.SOURCE_IDENTITIES)), "fixture": {"path": gate.FIXTURE_PATH, "bytes": 3, "sha256": gate.FIXTURE_SHA256, "status": "REVIEWED"}, "contract": gate.CONTRACT, "publication_decision": "NO_UPLOAD", "approval": {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None}}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            assert audit(lock, project, manifest_path, fixture, output) == 2
            report = load_json(output)
            assert isinstance(report, dict) and report["decision"] == "BLOCKED" and report["no_upload"] is True
            assert report["scope_status"] == "PENDING_REVIEW_NOT_OWNER_SIGNABLE"
            assert report["scope_basis"] == "CURRENT_MANIFEST_INCLUDING_UNRESOLVED_REVIEW_ROWS"
            assert report["owner_signoff_eligible"] is False
            assert report["source_license_bytes"]["status"] == "BLOCKED_NOT_MATERIALIZED"
            assert report["scope_sha256"] == gate.scope(manifest)
            try:
                audit(lock, project, manifest_path, fixture, output)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted existing output")
            manifest_path.write_text('{"publication_decision":"NO_UPLOAD","publication_decision":"UPLOAD"}', encoding="utf-8")
            try:
                audit(lock, project, manifest_path, fixture, None)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted duplicate manifest key")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fixture.write_bytes(b"tampered")
            try:
                audit(lock, project, manifest_path, fixture, None)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted tampered fixture")
            fixture.write_bytes(b"wav")
            tampered = dict(manifest); tampered["approval"] = {"status": "OWNER_SIGNOFF_APPROVED", "signer": "fake-owner", "digest": "0" * 64}
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                audit(lock, project, manifest_path, fixture, None)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted tampered approval state")
            tampered = dict(manifest); tampered["publication_decision"] = "UPLOAD"
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                audit(lock, project, manifest_path, fixture, None)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted tampered publication decision")
            tampered = dict(manifest); tampered["package_rows"] = []
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            try:
                audit(lock, project, manifest_path, fixture, None)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise SystemExit("self-test accepted tampered package inventory")
        finally:
            gate.LOCK_SHA256, gate.PROJECT_SHA256, gate.FIXTURE_BYTES, gate.FIXTURE_SHA256, gate.IDENTITIES, gate.SOURCE_IDENTITIES, gate.LICENSE_IDS, gate.CONTRACT, gate.FIXTURE_PATH = old
    print("model_free_audit.py self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest, args.fixture, args.output)):
            parser.error("--self-test is exclusive with audit input/output arguments")
        self_test()
        return 0
    if not all((args.lock, args.project, args.manifest, args.fixture)):
        parser.error("--lock, --project, --manifest, and --fixture are required")
    return audit(args.lock, args.project, args.manifest, args.fixture, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
