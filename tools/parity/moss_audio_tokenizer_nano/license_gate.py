#!/usr/bin/env python3
"""Dependency-free, offline approval gate for the MOSS Audio Tokenizer Nano."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
import tomllib

REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
# These are code-bound after the staged files are finalized; a byte drift blocks.
LOCK_SHA256 = "d5580f6bc13c20169451b789311863f50b917b0f07b364e80fa6a0c26314e7a5"
PROJECT_SHA256 = "62266fe62f3a94bf8604bdc771a3185cea353a80bbb81b06fe56488bae11d6fc"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_FILES = (
    "LICENSE", "README.md", "config.json", "configuration_moss_audio_tokenizer.py",
    "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
)
# No authenticated byte/SHA evidence for Nano's fixed revision is present in
# this checkout. Nulls are intentional: production must remain blocked.
FILE_IDENTITIES = {
    name: {"path": name, "role": {"LICENSE":"license", "README.md":"documentation", "config.json":"config", "configuration_moss_audio_tokenizer.py":"source", "modeling_moss_audio_tokenizer.py":"source", "model.safetensors.index.json":"index", "model-00001-of-00001.safetensors":"weights"}[name], "bytes": None, "sha256": None, "status": "UNRESOLVED"}
    for name in PAYLOAD_FILES
}
ROUTE = {
    "status": "UNRESOLVED",
    "transformers_version": "5.5.0",
    "reason": "official Nano dataclass compatibility has no authenticated package route",
}
REFERENCE_CONTRACT = {
    "frames": 2, "quantizers": 16, "codebook_size": 1024,
    "sample_rate": 48000, "channels": 2, "frame_hop": 3840,
    # The official Nano decoder tap count/shapes are not authenticated yet.
    # Nulls are code-bound blockers, not permissive wildcards.
    "quantizer_shape": None, "decoder_tap_count": None,
    "decoder_tap_shapes": None,
}
SENTINELS = {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
LICENSE_SCHEMA = {"id", "license", "status", "conclusion", "native_bundled_review"}
LICENSE_IDS = ["source-apache", "weights-apache", "python-closure"]
PACKAGE_REVIEW_SCHEMA = {
    "name", "version", "source", "license", "status", "native_bundled_review",
}
MANIFEST_SCHEMA = {"gate_version", "lock_sha256", "project_sha256", "package_rows", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "license_rows", "license_rows_sha256", "model_rows", "model_rows_sha256", "upstream_repo", "upstream_revision", "reference_route", "reference_contract", "publication_decision", "approval"}
APPROVAL_SCHEMA = {"status", "signer", "digest"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    """Hash without materializing a potentially multi-gigabyte shard."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        blocked(f"cannot read snapshot file {path}: {error}")
    return digest.hexdigest()


def canon(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_json(text: str) -> object:
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=reject)


def resolved(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = "_".join(value.strip().casefold().split())
    return bool(normalized) and normalized not in SENTINELS


def lock_rows(lock: dict) -> list[dict]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("lock package table missing/empty")
    rows = []
    seen = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("malformed lock package row")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError("lock package name/version must be nonempty strings")
        key = (name, version)
        if key in seen:
            raise ValueError("duplicate lock package identity")
        seen.add(key)
        markers = package.get("resolution-markers", [])
        dependencies = package.get("dependencies", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers):
            raise ValueError("malformed lock resolution markers")
        if not isinstance(dependencies, list) or any(not isinstance(dep, dict) or not set(dep) <= {"name", "marker"} or not isinstance(dep.get("name"), str) or not dep["name"] or ("marker" in dep and not isinstance(dep["marker"], str)) for dep in dependencies):
            raise ValueError("malformed lock dependency row")
        rows.append({
            "name": name, "version": version, "source": package.get("source"),
            "resolution-markers": package.get("resolution-markers", []),
            "dependencies": package.get("dependencies", []),
        })
    return sorted(rows, key=lambda row: (row["name"], row["version"]))


def artifact_error(lock: dict) -> str | None:
    """Require resolver-pinned distribution metadata for every real package."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        return "package table is not a list"
    virtual_count = 0
    for package in packages:
        if not isinstance(package, dict):
            return "package row is not a table"
        source = package.get("source")
        if not isinstance(source, dict):
            return f"package {package.get('name')!r} has malformed source"
        if "sdist" in package and not isinstance(package["sdist"], dict):
            return f"package {package.get('name')!r} has malformed sdist"
        if "wheels" in package and not isinstance(package["wheels"], list):
            return f"package {package.get('name')!r} has malformed wheels"
        if source == {"virtual": "."}:
            if "sdist" in package or "wheels" in package:
                return "virtual project source cannot carry resolver artifacts"
            virtual_count += 1
            continue
        if set(source) != {"registry"} or not isinstance(source.get("registry"), str) or not source["registry"].startswith("https://"):
            return f"package {package.get('name')!r} has malformed registry source"
        artifacts = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            artifacts.append(sdist)
        wheels = package.get("wheels")
        if isinstance(wheels, list):
            artifacts.extend(wheels)
        if not artifacts:
            return f"package {package.get('name')!r} has no resolver artifacts"
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                return f"package {package.get('name')!r} has malformed artifact"
            url, digest, size = artifact.get("url"), artifact.get("hash"), artifact.get("size")
            if not isinstance(url, str) or not url.startswith("https://"):
                return f"package {package.get('name')!r} has invalid artifact URL"
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                return f"package {package.get('name')!r} has invalid artifact hash"
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                return f"package {package.get('name')!r} has invalid artifact size"
    if virtual_count != 1:
        return "lock must contain exactly one virtual project source"
    return None


def project_identity(project: bytes) -> tuple[str, str]:
    data = tomllib.loads(project.decode())
    metadata = data.get("project")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str) or not metadata["name"] or not isinstance(metadata.get("version"), str) or not metadata["version"]:
        raise ValueError("project must declare nonempty name and version")
    return metadata["name"], metadata["version"]


def scope(manifest: dict) -> str:
    return canon({
        "schema": "moss-audio-tokenizer-nano-approval-v1",
        "gate_version": manifest.get("gate_version"),
        "lock_sha256": manifest.get("lock_sha256"),
        "project_sha256": manifest.get("project_sha256"),
        "package_rows": manifest.get("package_rows"),
        "package_rows_sha256": manifest.get("package_rows_sha256"),
        "package_review_rows": manifest.get("package_review_rows"),
        "package_review_rows_sha256": manifest.get("package_review_rows_sha256"),
        "license_rows": manifest.get("license_rows"),
        "license_rows_sha256": manifest.get("license_rows_sha256"),
        "model_rows": manifest.get("model_rows"),
        "model_rows_sha256": manifest.get("model_rows_sha256"),
        "upstream_repo": manifest.get("upstream_repo"),
        "upstream_revision": manifest.get("upstream_revision"),
        "reference_route": manifest.get("reference_route"),
        "reference_contract": manifest.get("reference_contract"),
        "publication_decision": manifest.get("publication_decision"),
        "expected_decision": "APPROVED",
    })


def blocked(message: str) -> None:
    print(f"moss Nano license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def verify_snapshot(snapshot: Path, manifest_path: Path) -> None:
    if not snapshot.is_dir():
        blocked(f"snapshot directory is missing: {snapshot}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        blocked("snapshot manifest is missing or not a regular file")
    try:
        manifest = load_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        blocked(f"invalid snapshot manifest: {error}")
    rows = manifest.get("model_rows")
    if not isinstance(rows, list) or [row.get("path") for row in rows if isinstance(row, dict)] != list(PAYLOAD_FILES):
        blocked("snapshot identity table is not the exact seven-file contract")
    expected = {row["path"]: row for row in rows}
    entries = list(snapshot.iterdir())
    for entry in entries:
        if entry.is_symlink():
            blocked(f"snapshot contains a symlink: {entry.name}")
        if entry.name == ".cache":
            if not entry.is_dir():
                blocked("snapshot transport cache is not a directory")
            # Hugging Face may place transport metadata here. It is never an
            # accepted payload path and is intentionally not traversed.
            continue
    actual = sorted(entry.name for entry in entries if entry.name != ".cache")
    if actual != sorted(PAYLOAD_FILES):
        blocked(f"snapshot file inventory differs from fixed payload: {actual}")
    for name in PAYLOAD_FILES:
        row = expected[name]
        path = snapshot / name
        if path.is_symlink() or not path.is_file():
            blocked(f"snapshot payload is not a regular file: {name}")
        if row.get("status") != "REVIEWED" or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0 or not isinstance(row.get("sha256"), str) or not HEX64.fullmatch(row["sha256"]):
            blocked(f"snapshot identity unresolved: {name}")
        if path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            blocked(f"snapshot identity mismatch: {name}")
    print("moss Nano snapshot identity: PASS")


def run(lock_path: Path, project_path: Path, manifest_path: Path, approval: Path | None) -> None:
    for path, label in ((lock_path, "lock"), (project_path, "project"), (manifest_path, "manifest")):
        if path.is_symlink() or not path.is_file():
            blocked(f"{label} input is missing or not a regular file")
    try:
        lock_bytes = lock_path.read_bytes()
        project_bytes = project_path.read_bytes()
        manifest = load_json(manifest_path.read_text(encoding="utf-8"))
        lock = tomllib.loads(lock_bytes.decode())
        project_name, project_version = project_identity(project_bytes)
        rows = lock_rows(lock)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, json.JSONDecodeError, ValueError) as error:
        blocked(f"invalid closure: {error}")
    if (error := artifact_error(lock)) is not None:
        blocked(f"resolver artifact metadata: {error}")
    virtuals = [package for package in lock["package"] if isinstance(package, dict) and package.get("source") == {"virtual": "."}]
    if len(virtuals) != 1 or virtuals[0].get("name") != project_name or virtuals[0].get("version") != project_version:
        blocked("virtual project row does not bind to pyproject identity")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_SCHEMA:
        blocked("manifest schema drifted")
    if manifest.get("gate_version") != 1 or type(manifest.get("gate_version")) is not int:
        blocked("unsupported gate_version")
    if not HEX64.fullmatch(LOCK_SHA256) or not HEX64.fullmatch(PROJECT_SHA256):
        blocked("code-bound lock/project digest is not finalized")
    if sha(lock_bytes) != LOCK_SHA256 or sha(project_bytes) != PROJECT_SHA256:
        blocked("lock/project bytes differ from code-bound closure")
    if manifest.get("lock_sha256") != LOCK_SHA256 or manifest.get("project_sha256") != PROJECT_SHA256:
        blocked("manifest lock/project hashes differ from code-bound closure")
    if manifest.get("reference_route") != ROUTE or ROUTE["status"] != "REVIEWED":
        blocked("official Transformers compatibility route is not authenticated")
    if manifest.get("package_rows") != rows or manifest.get("package_rows_sha256") != canon(rows):
        blocked("canonical lock rows drifted")

    reviews = manifest.get("package_review_rows")
    if not isinstance(reviews, list) or len(reviews) != len(rows):
        blocked("every locked package needs a review row")
    actual = {(row["name"], row["version"]): row for row in rows}
    seen = set()
    for review in reviews:
        if not isinstance(review, dict):
            blocked("malformed package review row")
        if set(review) != PACKAGE_REVIEW_SCHEMA:
            blocked("package review row schema drifted")
        key = (review.get("name"), review.get("version"))
        if key in seen or key not in actual or review.get("source") != actual[key].get("source"):
            blocked("package review identity/source drifted")
        seen.add(key)
        if review.get("status") != "REVIEWED" or not resolved(review.get("license")) or not resolved(review.get("native_bundled_review")):
            blocked(f"package review unresolved: {key}")
    if seen != set(actual) or manifest.get("package_review_rows_sha256") != canon(reviews):
        blocked("package review rows drifted")

    if manifest.get("upstream_repo") != REPO or manifest.get("upstream_revision") != REVISION:
        blocked("fixed upstream identity drifted")
    model_rows = manifest.get("model_rows")
    if not isinstance(model_rows, list) or len(model_rows) != len(PAYLOAD_FILES):
        blocked("Nano payload identity table must contain exactly seven files")
    if [row.get("path") for row in model_rows if isinstance(row, dict)] != list(PAYLOAD_FILES):
        blocked("Nano payload files are missing, duplicated, reordered, or extra")
    if any(not isinstance(row, dict) or set(row) != {"path", "role", "bytes", "sha256", "status"} for row in model_rows):
        blocked("Nano payload identity schema drifted")
    if manifest.get("model_rows_sha256") != canon(model_rows):
        blocked("Nano payload identity digest drifted")
    for row in model_rows:
        expected = FILE_IDENTITIES[row["path"]]
        if row != expected or row["status"] != "REVIEWED" or type(row["bytes"]) is not int or row["bytes"] <= 0 or not isinstance(row["sha256"], str) or not HEX64.fullmatch(row["sha256"]):
            blocked(f"Nano payload identity unresolved or drifted: {row.get('path')}")

    licenses = manifest.get("license_rows")
    if not isinstance(licenses, list) or len(licenses) != 3 or [row.get("id") for row in licenses if isinstance(row, dict)] != LICENSE_IDS:
        blocked("license rows are missing, duplicated, reordered, or extra")
    if any(not isinstance(row, dict) or set(row) != LICENSE_SCHEMA for row in licenses):
        blocked("license row schema drifted")
    if manifest.get("license_rows_sha256") != canon(licenses):
        blocked("license rows digest drifted")
    if any(row["status"] != "REVIEWED" or not resolved(row["license"]) or not resolved(row["conclusion"]) or not resolved(row["native_bundled_review"]) for row in licenses):
        blocked("license conclusion/native disposition unresolved")
    contract = manifest.get("reference_contract")
    if not isinstance(contract, dict) or set(contract) != set(REFERENCE_CONTRACT) or any(type(contract[key]) is not int for key, value in REFERENCE_CONTRACT.items() if isinstance(value, int)) or contract != REFERENCE_CONTRACT:
        blocked("reference contract drifted")
    if manifest.get("publication_decision") != "NO_UPLOAD":
        blocked("publication decision is not NO_UPLOAD")
    approval_record = manifest.get("approval")
    if not isinstance(approval_record, dict) or set(approval_record) != APPROVAL_SCHEMA or approval_record.get("status") != "OWNER_SIGNOFF_APPROVED":
        blocked("owner signoff remains required")
    expected_scope = scope(manifest)
    if approval_record.get("digest") != expected_scope or not isinstance(approval_record.get("signer"), str) or not approval_record["signer"] or not HEX64.fullmatch(str(approval_record.get("digest"))):
        blocked("approval digest is not canonical")
    if approval is None or approval.is_symlink() or not approval.is_file():
        blocked("approval evidence missing or is not a regular file")
    try:
        evidence = load_json(approval.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        blocked(f"approval evidence unreadable: {error}")
    if not isinstance(evidence, dict) or set(evidence) != {"scope_schema", "scope_sha256", "approval_digest", "decision", "signer", "manifest_sha256"} or evidence.get("scope_schema") != "moss-audio-tokenizer-nano-approval-v1" or evidence.get("scope_sha256") != expected_scope or evidence.get("approval_digest") != expected_scope or evidence.get("decision") != "APPROVED" or evidence.get("signer") != approval_record["signer"] or evidence.get("manifest_sha256") != sha(manifest_path.read_bytes()):
        blocked("approval evidence does not bind canonical scope")
    print("moss Nano license gate: PASS")


def self_test() -> None:
    for value in (None, "", " null ", "OWNER_REVIEW_REQUIRED", "pending review", "TODO"):
        if resolved(value):
            raise SystemExit(f"self-test resolved placeholder: {value!r}")
    if not resolved("owner_review_required is a historical citation"):
        raise SystemExit("self-test rejected a longer citation")
    global LOCK_SHA256, PROJECT_SHA256, PAYLOAD_FILES, FILE_IDENTITIES, ROUTE
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); lock_path = root / "uv.lock"; project_path = root / "pyproject.toml"; manifest_path = root / "manifest.json"; evidence_path = root / "evidence.json"
        vector_path = root / "vector"
        vector_path.write_bytes(b"abc")
        if sha_file(vector_path) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
            raise SystemExit("self-test streaming SHA-256 known vector failed")
        virtual = '\n[[package]]\nname="demo"\nversion="0.1.0"\nsource={virtual="."}\n'
        lock_path.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/demo.tar.gz",hash="sha256:' + "a" * 64 + '",size=1}\n' + virtual, encoding="utf-8")
        project_path.write_text('[project]\nname="demo"\nversion="0.1.0"\n', encoding="utf-8")
        valid_lock = tomllib.loads(lock_path.read_text())
        for label, mutate in (("sdist", lambda value: value["package"][0].update(sdist="bad")), ("wheels", lambda value: value["package"][0].update(wheels={})), ("source", lambda value: value["package"][0].update(source="bad")), ("virtual-source", lambda value: value["package"][0].update(source={"virtual":"other"})), ("missing-virtual", lambda value: value["package"].pop()), ("duplicate-virtual", lambda value: value["package"].append(dict(value["package"][-1]))), ("duplicate-package", lambda value: value["package"].append(dict(value["package"][0]))), ("bool-size", lambda value: value["package"][0]["sdist"].update(size=True))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            rejected = artifact_error(candidate) is not None
            if label == "duplicate-package":
                try:
                    lock_rows(candidate)
                except ValueError:
                    rejected = True
            if not rejected:
                raise SystemExit(f"self-test accepted malformed lock: {label}")
        PAYLOAD_FILES = ("demo",)
        FILE_IDENTITIES = {"demo": {"path": "demo", "role": "upstream", "bytes": 3, "sha256": sha(b"abc"), "status": "REVIEWED"}}
        ROUTE = {"status": "REVIEWED", "transformers_version": "5.5.0", "reason": "owner evidence"}
        LOCK_SHA256, PROJECT_SHA256 = sha(lock_path.read_bytes()), sha(project_path.read_bytes())
        rows = lock_rows(tomllib.loads(lock_path.read_text()))
        review = [{"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "reviewed"}, {"name": "demo", "version": "0.1.0", "source": {"virtual": "."}, "license": "project", "status": "REVIEWED", "native_bundled_review": "reviewed"}]
        licenses = [{"id": ident, "license": "Apache-2.0", "status": "REVIEWED", "conclusion": "reviewed", "native_bundled_review": "reviewed"} for ident in LICENSE_IDS]
        model_rows = list(FILE_IDENTITIES.values())
        manifest = {"gate_version": 1, "lock_sha256": LOCK_SHA256, "project_sha256": PROJECT_SHA256, "package_rows": rows, "package_rows_sha256": canon(rows), "package_review_rows": review, "package_review_rows_sha256": canon(review), "license_rows": licenses, "license_rows_sha256": canon(licenses), "model_rows": model_rows, "model_rows_sha256": canon(model_rows), "upstream_repo": REPO, "upstream_revision": REVISION, "reference_route": ROUTE, "reference_contract": REFERENCE_CONTRACT, "publication_decision": "NO_UPLOAD", "approval": {"status": "OWNER_SIGNOFF_APPROVED", "signer": "owner", "digest": None}}
        manifest["approval"]["digest"] = scope(manifest); manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        evidence = {"scope_schema": "moss-audio-tokenizer-nano-approval-v1", "scope_sha256": manifest["approval"]["digest"], "approval_digest": manifest["approval"]["digest"], "decision": "APPROVED", "signer": "owner", "manifest_sha256": sha(manifest_path.read_bytes())}; evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        snapshot = root / "snapshot"; snapshot.mkdir(); (snapshot / "demo").write_bytes(b"abc")
        (snapshot / ".cache" / "huggingface").mkdir(parents=True)
        verify_snapshot(snapshot, manifest_path)
        for label, mutate in (
            ("snapshot-missing", lambda: (snapshot / "demo").unlink()),
            ("snapshot-extra", lambda: (snapshot / "extra").write_bytes(b"x")),
            ("snapshot-symlink", lambda: ((snapshot / "demo").unlink(), (snapshot / "demo").symlink_to(vector_path))),
            ("snapshot-hash", lambda: (snapshot / "demo").write_bytes(b"tampered")),
        ):
            if label == "snapshot-extra":
                (snapshot / "demo").write_bytes(b"abc")
            mutate()
            try:
                verify_snapshot(snapshot, manifest_path)
            except SystemExit as error:
                if error.code != 2: raise
            else:
                raise SystemExit(f"self-test accepted {label} tamper")
            for path in (snapshot / "extra", snapshot / "demo"):
                if path.is_symlink() or path.exists():
                    path.unlink()
            (snapshot / "demo").write_bytes(b"abc")
        run(lock_path, project_path, manifest_path, evidence_path)
        for input_path, label in ((lock_path, "lock-input"), (project_path, "project-input"), (manifest_path, "manifest-input")):
            target = root / (label + "-target"); target.write_bytes(b"input-target")
            original = input_path.read_bytes(); input_path.unlink(); input_path.symlink_to(target)
            try:
                run(lock_path, project_path, manifest_path, evidence_path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"self-test accepted symlink {label}")
            input_path.unlink(); input_path.write_bytes(original)
        for label, mutate in (("artifact", None), ("scope", lambda m: m["reference_contract"].update(frames=1)), ("model", lambda m: m["model_rows"][0].update(status="UNRESOLVED")), ("license", lambda m: m["license_rows"][0].update(conclusion="TODO")), ("publication", lambda m: m.update(publication_decision="UPLOAD")), ("arbitrary", lambda m: m["approval"].update(digest="a" * 64)), ("package-schema", lambda m: m["package_review_rows"][0].update(extra="drift")), ("manifest-schema", lambda m: m.update(extra=True)), ("approval-schema", lambda m: m["approval"].update(extra=True))):
            if label == "artifact":
                lock_path.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/demo.tar.gz",hash="sha256:' + "a" * 64 + '"}\n' + virtual, encoding="utf-8")
                try:
                    run(lock_path, project_path, manifest_path, evidence_path)
                except SystemExit as error:
                    if error.code != 2:
                        raise
                else:
                    raise SystemExit("self-test accepted artifact tamper")
                lock_path.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/demo.tar.gz",hash="sha256:' + "a" * 64 + '",size=1}\n' + virtual, encoding="utf-8")
                continue
            candidate = load_json(manifest_path.read_text()); mutate(candidate); manifest_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
            try:
                run(lock_path, project_path, manifest_path, evidence_path)
            except SystemExit as error:
                if error.code != 2: raise
            else:
                raise SystemExit(f"self-test accepted {label} tamper")
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        for label, mutate in (("evidence-scope", lambda value: value.update(scope_sha256="a" * 64)), ("evidence-signer", lambda value: value.update(signer="other")), ("evidence-decision", lambda value: value.update(decision="PENDING")), ("evidence-extra", lambda value: value.update(extra=True))):
            candidate = load_json(evidence_path.read_text(encoding="utf-8")); mutate(candidate); evidence_path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                run(lock_path, project_path, manifest_path, evidence_path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"self-test accepted {label} tamper")
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        evidence_path.unlink(); evidence_path.symlink_to(manifest_path)
        try:
            run(lock_path, project_path, manifest_path, evidence_path)
        except SystemExit as error:
            if error.code != 2:
                raise
        else:
            raise SystemExit("self-test accepted symlink evidence")
    print("license_gate.py self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-snapshot", action="store_true"); parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--lock", type=Path); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify_snapshot:
        if not args.snapshot or not args.manifest:
            parser.error("--verify-snapshot requires --snapshot and --manifest")
        verify_snapshot(args.snapshot, args.manifest)
    elif not all((args.lock, args.project, args.manifest)):
        parser.error("--lock, --project, and --manifest are required")
    else:
        run(args.lock, args.project, args.manifest, args.approval)
