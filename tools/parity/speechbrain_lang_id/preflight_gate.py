#!/usr/bin/env python3
"""Stdlib-only, offline, fail-closed SpeechBrain Lang-ID gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
import tomllib

GATE_VERSION = 1
REPO = "speechbrain/lang-id-voxlingua107-ecapa"
REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
LOCK_SHA256 = "1f16fbbfc147274bc9791332b023122a049a3b68db197d0ce2a04002640673fc"
PROJECT_SHA256 = "c11018b1a1a51f786c2036ff40c6a8b519f73db6309cd82d69a6ee989593608d"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_FILES = ("embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt")
SOURCE_IDENTITIES = (
    {"path": "hyperparams.yaml", "role": "official-loader-config", "bytes": 1519, "sha256": "88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f", "status": "REVIEWED"},
    {"path": "config.json", "role": "official-loader-metadata", "bytes": 51, "sha256": "a861f8fbc2e23c0fc0823b3c0fd2b3d1e839563c2d4e3f9663a1237cce62bc89", "status": "REVIEWED"},
)
SOURCE_IDENTITY_SCHEMA = {"path", "role", "bytes", "sha256", "status"}
LICENSE_IDS = ["model-apache", "source-code", "python-closure", "fixture"]
LICENSE_SCHEMA = {"id", "license", "status", "conclusion", "native_bundled_review"}
PACKAGE_REVIEW_SCHEMA = {"name", "version", "source", "license", "status", "native_bundled_review"}
IDENTITY_SCHEMA = {"path", "role", "bytes", "sha256", "status"}
IDENTITIES = {
    "embedding_model.ckpt": {"path": "embedding_model.ckpt", "role": "checkpoint", "bytes": 84474355, "sha256": "ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a", "status": "REVIEWED"},
    "classifier.ckpt": {"path": "classifier.ckpt", "role": "checkpoint", "bytes": 762555, "sha256": "a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc", "status": "REVIEWED"},
    "label_encoder.txt": {"path": "label_encoder.txt", "role": "labels", "bytes": 2204, "sha256": "9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f", "status": "REVIEWED"},
}
CONTRACT = {"n_mels": 60, "embedding_dim": 256, "class_count": 107, "sample_rate": 16000}
FIXTURE_PATH = "tests/fixtures/audio/jfk-30s.wav"
FIXTURE_BYTES = 352078
FIXTURE_SHA256 = "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
UNRESOLVED = {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
MANIFEST_KEYS = {"gate_version", "lock_sha256", "project_sha256", "package_rows", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "license_rows", "license_rows_sha256", "upstream_repo", "upstream_revision", "payload_identities", "payload_identities_sha256", "source_identities", "source_identities_sha256", "fixture", "contract", "publication_decision", "approval"}
APPROVAL_KEYS = {"status", "signer", "digest"}
EVIDENCE_KEYS = {"decision", "signer", "scope_sha256", "approval_digest", "manifest_sha256"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_reject_duplicate_keys)


def block(message: str) -> None:
    print(f"SpeechBrain Lang-ID gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        block(f"{label} must be a regular non-symlink file")


def resolved(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = "_".join(value.strip().casefold().split())
    return bool(normalized) and normalized not in UNRESOLVED


def lock_rows(lock: dict) -> list[dict]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        block("lock package table is missing or empty")
    rows = []
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not package["name"].strip() or not isinstance(package.get("version"), str) or not package["version"].strip():
            block("lock package row is malformed")
        rows.append({"name": package["name"], "version": package["version"], "source": package.get("source"), "dependencies": package.get("dependencies", [])})
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def lock_artifacts_complete(lock: dict) -> bool:
    """Require uv's immutable registry artifact metadata for every package."""
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        return False
    for package in packages:
        if not isinstance(package, dict):
            return False
        source = package.get("source")
        if not isinstance(source, dict) or not source or not all(isinstance(k, str) and isinstance(v, str) for k, v in source.items()):
            return False
        if source.get("virtual") == ".":
            continue
        if "virtual" in source:
            return False
        artifacts = []
        for key in ("sdist", "wheels"):
            value = package.get(key)
            if value is None:
                continue
            if key == "sdist":
                if not isinstance(value, dict):
                    return False
                artifacts.append(value)
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                artifacts.extend(value)
            else:
                return False
        if not artifacts:
            return False
        for artifact in artifacts:
            if not isinstance(artifact.get("url"), str) or not artifact["url"].startswith("https://") or not isinstance(artifact.get("size"), int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0 or not isinstance(artifact.get("hash"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                return False
    return True


def scope(manifest: dict) -> str:
    return canonical({
        "schema": "speechbrain-lang-id-approval-v1",
        "gate_version": manifest.get("gate_version"),
        "lock_sha256": manifest.get("lock_sha256"), "project_sha256": manifest.get("project_sha256"),
        "package_rows": manifest.get("package_rows"), "package_rows_sha256": manifest.get("package_rows_sha256"),
        "package_review_rows": manifest.get("package_review_rows"), "package_review_rows_sha256": manifest.get("package_review_rows_sha256"),
        "license_rows": manifest.get("license_rows"), "license_rows_sha256": manifest.get("license_rows_sha256"),
        "upstream_repo": manifest.get("upstream_repo"), "upstream_revision": manifest.get("upstream_revision"),
        "payload_identities": manifest.get("payload_identities"), "source_identities": manifest.get("source_identities"), "source_identities_sha256": manifest.get("source_identities_sha256"),
        "fixture": manifest.get("fixture"), "contract": manifest.get("contract"),
        "publication_decision": manifest.get("publication_decision"), "expected_decision": "APPROVED",
    })


def run(lock_path: Path, project_path: Path, manifest_path: Path, approval: Path | None) -> None:
    try:
        regular_file(lock_path, "lock")
        regular_file(project_path, "project")
        regular_file(manifest_path, "manifest")
        if approval is not None:
            regular_file(approval, "approval evidence")
        lock_bytes, project_bytes = lock_path.read_bytes(), project_path.read_bytes()
        manifest = load_json(manifest_path.read_text(encoding="utf-8"))
        lock_data = tomllib.loads(lock_bytes.decode())
        rows = lock_rows(lock_data)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        block(f"closure is unreadable: {error}")
    if not isinstance(manifest, dict):
        block("manifest JSON top-level value must be an object")
    if manifest.get("gate_version") != GATE_VERSION:
        block("unsupported gate version")
    if set(manifest) != MANIFEST_KEYS:
        block("manifest schema has missing or extra keys")
    if digest(lock_bytes) != LOCK_SHA256 or digest(project_bytes) != PROJECT_SHA256:
        block("lock/project bytes differ from code-bound closure")
    if manifest.get("lock_sha256") != LOCK_SHA256 or manifest.get("project_sha256") != PROJECT_SHA256:
        block("manifest lock/project hashes differ from code-bound closure")
    if not lock_artifacts_complete(lock_data):
        block("uv lock artifact URLs/hashes/sizes are incomplete")
    if manifest.get("package_rows") != rows or manifest.get("package_rows_sha256") != canonical(rows):
        block("canonical locked package rows drifted")
    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        block("every locked package needs a review row")
    expected = {(row["name"], row["version"]): row for row in rows}; seen = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != PACKAGE_REVIEW_SCHEMA:
            block("package review schema drifted")
        key = (review["name"], review["version"])
        if key in seen or key not in expected or review["source"] != expected[key]["source"]:
            block("package review identity/source drifted")
        seen.add(key)
        if review["status"] != "REVIEWED" or not resolved(review["license"]) or not resolved(review["native_bundled_review"]):
            block(f"package review unresolved: {key}")
    if seen != set(expected) or manifest.get("package_review_rows_sha256") != canonical(reviews):
        block("package review rows drifted")
    if manifest.get("upstream_repo") != REPO or manifest.get("upstream_revision") != REVISION:
        block("fixed official source identity drifted")
    payload = manifest.get("payload_identities")
    if not isinstance(payload, list) or [row.get("path") for row in payload if isinstance(row, dict)] != list(PAYLOAD_FILES) or any(not isinstance(row, dict) or set(row) != IDENTITY_SCHEMA for row in payload):
        block("checkpoint payload identity table is not the exact three-file schema")
    if manifest.get("payload_identities_sha256") != canonical(payload):
        block("checkpoint payload identity digest drifted")
    for row in payload:
        if row != IDENTITIES[row["path"]] or row["status"] != "REVIEWED" or not isinstance(row["bytes"], int) or row["bytes"] <= 0 or not isinstance(row["sha256"], str) or not HEX64.fullmatch(row["sha256"]):
            block(f"checkpoint identity unresolved: {row.get('path')}")
    source = manifest.get("source_identities")
    if source != list(SOURCE_IDENTITIES) or manifest.get("source_identities_sha256") != canonical(source) or not isinstance(source, list) or len(source) != len(SOURCE_IDENTITIES) or any(set(row) != SOURCE_IDENTITY_SCHEMA or row.get("status") != "REVIEWED" or not resolved(row.get("path")) or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0 or not HEX64.fullmatch(str(row.get("sha256", ""))) for row in source):
        block("official SpeechBrain source-code identity is unresolved")
    licenses = manifest.get("license_rows")
    if not isinstance(licenses, list) or [row.get("id") for row in licenses if isinstance(row, dict)] != LICENSE_IDS or any(not isinstance(row, dict) or set(row) != LICENSE_SCHEMA for row in licenses):
        block("license rows are missing, duplicated, reordered, or malformed")
    if manifest.get("license_rows_sha256") != canonical(licenses):
        block("license row digest drifted")
    if any(row["status"] != "REVIEWED" or not resolved(row["license"]) or not resolved(row["conclusion"]) or not resolved(row["native_bundled_review"]) for row in licenses):
        block("license/source/native closure review is unresolved")
    fixture = manifest.get("fixture")
    if not isinstance(fixture, dict) or set(fixture) != {"path", "bytes", "sha256", "status"} or fixture.get("path") != FIXTURE_PATH or fixture.get("bytes") != FIXTURE_BYTES or fixture.get("sha256") != FIXTURE_SHA256 or fixture.get("status") != "REVIEWED":
        block("fixed WAV fixture identity is unresolved")
    if manifest.get("contract") != CONTRACT or manifest.get("publication_decision") != "NO_UPLOAD":
        block("contract/publication decision drifted")
    approval_record = manifest.get("approval")
    if not isinstance(approval_record, dict) or set(approval_record) != APPROVAL_KEYS or approval_record.get("status") != "OWNER_SIGNOFF_APPROVED" or not resolved(approval_record.get("signer")) or not HEX64.fullmatch(str(approval_record.get("digest", ""))):
        block("owner signoff remains required")
    expected_scope = scope(manifest)
    if approval_record["digest"] != expected_scope or approval is None:
        block("approval scope/evidence is missing or not canonical")
    try:
        evidence = load_json(approval.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        block(f"approval evidence unreadable: {error}")
    if not isinstance(evidence, dict):
        block("approval evidence JSON top-level value must be an object")
    if set(evidence) != EVIDENCE_KEYS:
        block("approval evidence schema has missing or extra keys")
    if evidence.get("decision") != "APPROVED" or not resolved(evidence.get("signer")) or evidence.get("signer") != approval_record["signer"]:
        block("approval evidence signer/decision is not approved")
    if evidence.get("scope_sha256") != expected_scope or evidence.get("approval_digest") != expected_scope:
        block("approval evidence does not bind exact scope")
    if evidence.get("manifest_sha256") != digest(manifest_path.read_bytes()):
        block("approval evidence does not bind exact manifest bytes")
    print("SpeechBrain Lang-ID gate: PASS")


def self_test() -> None:
    # Production constants and manifest are intentionally unresolved. Exercise
    # the placeholder normalizer without importing any project dependency.
    for value in (None, "", " null ", "OWNER_REVIEW_REQUIRED", "pending review", "TODO"):
        if resolved(value):
            raise SystemExit(f"self-test resolved placeholder: {value!r}")
    if not resolved("license review ticket LANG-123"):
        raise SystemExit("self-test rejected a real review citation")
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw); lock = root / "uv.lock"; project = root / "pyproject.toml"; manifest_path = root / "manifest.json"; evidence = root / "approval.json"
        lock.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nwheels=[{url="https://example.test/demo-1.whl",hash="sha256:' + 'a' * 64 + '",size=1}]\n', encoding="utf-8"); project.write_text('[project]\nname="demo"\n', encoding="utf-8")
        global LOCK_SHA256, PROJECT_SHA256, PAYLOAD_FILES, IDENTITIES, SOURCE_IDENTITIES, LICENSE_IDS, CONTRACT, FIXTURE_PATH, FIXTURE_BYTES, FIXTURE_SHA256
        old = (LOCK_SHA256, PROJECT_SHA256, PAYLOAD_FILES, IDENTITIES, SOURCE_IDENTITIES, LICENSE_IDS, CONTRACT, FIXTURE_PATH, FIXTURE_BYTES, FIXTURE_SHA256)
        LOCK_SHA256, PROJECT_SHA256 = digest(lock.read_bytes()), digest(project.read_bytes()); PAYLOAD_FILES = ("demo.ckpt",); IDENTITIES = {"demo.ckpt": {"path":"demo.ckpt","role":"checkpoint","bytes":3,"sha256":digest(b"abc"),"status":"REVIEWED"}}; SOURCE_IDENTITIES = ({"path":"source.py","role":"official-loader-config","bytes":3,"sha256":digest(b"src"),"status":"REVIEWED"},); LICENSE_IDS = ["model-apache"]; CONTRACT = {"n_mels":60,"embedding_dim":256,"class_count":107,"sample_rate":16000}; FIXTURE_PATH, FIXTURE_BYTES, FIXTURE_SHA256 = "tests/fixtures/audio/jfk-30s.wav", 3, digest(b"wav")
        rows = lock_rows(tomllib.loads(lock.read_text())); reviews = [{"name":"demo","version":"1","source":{"registry":"https://pypi.org/simple"},"license":"MIT","status":"REVIEWED","native_bundled_review":"reviewed"}]; licenses = [{"id":"model-apache","license":"Apache-2.0","status":"REVIEWED","conclusion":"reviewed","native_bundled_review":"reviewed"}]; payload = list(IDENTITIES.values())
        assert lock_artifacts_complete(tomllib.loads(lock.read_text()))
        malformed_lock = tomllib.loads(lock.read_text()); malformed_lock["package"][0]["wheels"][0]["url"] = "file:///not-https"; assert not lock_artifacts_complete(malformed_lock)
        malformed_lock = tomllib.loads(lock.read_text()); malformed_lock["package"][0]["wheels"][0]["size"] = True; assert not lock_artifacts_complete(malformed_lock)
        malformed_lock = tomllib.loads(lock.read_text()); malformed_lock["package"][0]["wheels"] = {}; assert not lock_artifacts_complete(malformed_lock)
        manifest = {"gate_version":1,"lock_sha256":LOCK_SHA256,"project_sha256":PROJECT_SHA256,"package_rows":rows,"package_rows_sha256":canonical(rows),"package_review_rows":reviews,"package_review_rows_sha256":canonical(reviews),"license_rows":licenses,"license_rows_sha256":canonical(licenses),"upstream_repo":REPO,"upstream_revision":REVISION,"payload_identities":payload,"payload_identities_sha256":canonical(payload),"source_identities":list(SOURCE_IDENTITIES),"source_identities_sha256":canonical(list(SOURCE_IDENTITIES)),"fixture":{"path":"tests/fixtures/audio/jfk-30s.wav","bytes":3,"sha256":digest(b"wav"),"status":"REVIEWED"},"contract":CONTRACT,"publication_decision":"NO_UPLOAD","approval":{"status":"OWNER_SIGNOFF_APPROVED","signer":"test","digest":None}}
        manifest["approval"]["digest"] = scope(manifest); manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8"); evidence.write_text(json.dumps({"decision":"APPROVED","signer":"test","scope_sha256":manifest["approval"]["digest"],"approval_digest":manifest["approval"]["digest"],"manifest_sha256":digest(manifest_path.read_bytes())}), encoding="utf-8"); run(lock, project, manifest_path, evidence)
        baseline_manifest = manifest_path.read_text(encoding="utf-8"); baseline_evidence = evidence.read_text(encoding="utf-8")
        tampered_evidence = load_json(baseline_evidence); assert isinstance(tampered_evidence, dict); tampered_evidence["extra"] = "reject"; evidence.write_text(json.dumps(tampered_evidence), encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted extra approval evidence key")
        evidence.write_text(baseline_evidence, encoding="utf-8")
        evidence.write_text('{"decision":"APPROVED","decision":"APPROVED"}', encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted duplicate approval evidence key")
        evidence.write_text(baseline_evidence, encoding="utf-8")
        evidence.write_text('{"decision":"APPROVED","nested":{"scope":"ok","scope":"tampered"}}', encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted nested duplicate approval evidence key")
        evidence.write_text(baseline_evidence, encoding="utf-8")
        tampered_manifest = load_json(baseline_manifest); assert isinstance(tampered_manifest, dict); tampered_manifest["extra"] = "reject"; manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted extra manifest key")
        manifest_path.write_text(baseline_manifest, encoding="utf-8")
        manifest_path.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted duplicate manifest key")
        manifest_path.write_text('{"gate_version":1,"nested":{"scope":"ok","scope":"tampered"}}', encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted nested duplicate manifest key")
        manifest_path.write_text("[]", encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test leaked a non-object manifest exception")
        manifest_path.write_text(baseline_manifest, encoding="utf-8")
        tampered_evidence = load_json(baseline_evidence); assert isinstance(tampered_evidence, dict); tampered_evidence["signer"] = "OWNER_REVIEW_REQUIRED"; evidence.write_text(json.dumps(tampered_evidence), encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted placeholder signer")
        evidence.write_text(baseline_evidence, encoding="utf-8")
        manifest["package_review_rows"][0]["native_bundled_review"] = "OWNER_REVIEW_REQUIRED"; manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        try: run(lock, project, manifest_path, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted unresolved package review")
        manifest_path.write_text(baseline_manifest, encoding="utf-8")
        approval_link = root / "approval-link.json"; approval_link.symlink_to(evidence)
        try: run(lock, project, manifest_path, approval_link)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted symlinked approval evidence")
        manifest_link = root / "manifest-link.json"; manifest_link.symlink_to(manifest_path)
        try: run(lock, project, manifest_link, evidence)
        except SystemExit as error:
            if error.code != 2: raise
        else: raise SystemExit("self-test accepted symlinked manifest")
        LOCK_SHA256, PROJECT_SHA256, PAYLOAD_FILES, IDENTITIES, SOURCE_IDENTITIES, LICENSE_IDS, CONTRACT, FIXTURE_PATH, FIXTURE_BYTES, FIXTURE_SHA256 = old
    print("preflight_gate.py self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--lock", type=Path); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--approval", type=Path); args = parser.parse_args()
    if args.self_test: self_test()
    elif not all((args.lock, args.project, args.manifest)): parser.error("--lock, --project, and --manifest are required")
    else: run(args.lock, args.project, args.manifest, args.approval)
