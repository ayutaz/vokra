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
from urllib.parse import urlparse

REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
# These are code-bound after the staged files are finalized; a byte drift blocks.
LOCK_SHA256 = "29d49a9d88d73e185d3c125c3ac0c09baa83b51d3835e231ba634610bef3d8d1"
PROJECT_SHA256 = "ec4075893adeaed5e82d475284c14241a71a330100d07d387bc62befb7a1b6ce"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PAYLOAD_FILES = (
    ".gitattributes", "README.md", "__init__.py", "config.json", "configuration_moss_audio_tokenizer.py",
    "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json",
    "model-00001-of-00001.safetensors",
)
FILE_IDENTITIES = {
    ".gitattributes": {"path": ".gitattributes", "role": "_gitattributes", "bytes": 1519, "sha256": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 1519, "canonical_git_blob_sha1": "a6344aac8c09253b3b630fb776ae94478aa0275b", "git_blob_sha1": "a6344aac8c09253b3b630fb776ae94478aa0275b", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "README.md": {"path": "README.md", "role": "readme_md", "bytes": 12491, "sha256": "83cae343211c7c482433c9758f16fc9d17046a0cd450d185b473542e87e35ef8", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 12491, "canonical_git_blob_sha1": "a276b3e2e399916a0cda4c4145e48655ddd62faa", "git_blob_sha1": "a276b3e2e399916a0cda4c4145e48655ddd62faa", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "__init__.py": {"path": "__init__.py", "role": "__init___py", "bytes": 52, "sha256": "ed0b4910a5e53b1dfc3234cfcf96e2c6890d339f9134a3dfed8481d01b3768fd", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 52, "canonical_git_blob_sha1": "be87d805388ff76659c35e7273e8c3687b55eeff", "git_blob_sha1": "be87d805388ff76659c35e7273e8c3687b55eeff", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "config.json": {"path": "config.json", "role": "config_json", "bytes": 7385, "sha256": "b38892f8ba00efc18af2ad9eca999c7603f871548c2e7f99258cb1cefc70ee06", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 7385, "canonical_git_blob_sha1": "1789fbb15e81a782e83a726998b7c910aade93cf", "git_blob_sha1": "1789fbb15e81a782e83a726998b7c910aade93cf", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "configuration_moss_audio_tokenizer.py": {"path": "configuration_moss_audio_tokenizer.py", "role": "configuration", "bytes": 19249, "sha256": "b2d67dc4581e70f4b69b2d7eccefe32581d0c5192fe4d97fe1830e94a255b8aa", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 19249, "canonical_git_blob_sha1": "2efd84a7f1b9682d530616e434b13b8dec560136", "git_blob_sha1": "2efd84a7f1b9682d530616e434b13b8dec560136", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "modeling_moss_audio_tokenizer.py": {"path": "modeling_moss_audio_tokenizer.py", "role": "modeling", "bytes": 138814, "sha256": "b14af7c188944da5101adbd4aaa9c3617d66b83507f0efbd6eb416381a105930", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 138814, "canonical_git_blob_sha1": "17fc9d059a1dc0fede4cfd1dfed2cffaef70a4c1", "git_blob_sha1": "17fc9d059a1dc0fede4cfd1dfed2cffaef70a4c1", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "model.safetensors.index.json": {"path": "model.safetensors.index.json", "role": "model_safetensors_index_json", "bytes": 34346, "sha256": "2bf639e2c37c8502b1d0d92bf616b8108cb9975d42f587487a103682575b1d6d", "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 34346, "canonical_git_blob_sha1": "c1e82d52cc62a951fa73edf0df3980215c516832", "git_blob_sha1": "c1e82d52cc62a951fa73edf0df3980215c516832", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
    "model-00001-of-00001.safetensors": {"path": "model-00001-of-00001.safetensors", "role": "weights", "bytes": None, "sha256": None, "status": "AUTHENTICATED_SERVER_IDENTITY_ONLY", "materialized": False, "content_not_downloaded": True, "server_bytes": 87922568, "canonical_git_blob_sha1": None, "git_blob_sha1": None, "lfs_payload_sha256": "34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67", "lfs_pointer_git_blob_sha1": "0eb20a3607f11c3070a66415c638cdf79af3e82a"},
}
MODEL_INFO = {
    "id": REPO,
    "sha": REVISION,
    "private": False,
    "gated": False,
    "disabled": False,
    "cardData_license": "apache-2.0",
}
ROUTE = {
    "status": "UNRESOLVED",
    "transformers_version": "5.10.4",
    "previous_isolated_transformers_pin": "5.5.0",
    "isolated_transformers_pin": "transformers==5.10.4",
    "transformers_security_advisory": "GHSA-xrqw-3rrv-vx5w",
    "transformers_security_patched_minimum": "5.10.0",
    "transformers_compatibility_status": "BLOCKED_UNVERIFIED_API_SMOKE",
    "reason": "official Nano dataclass compatibility and API smoke remain unauthenticated; 5.5.0 is retained only as a previous isolated reference pin",
}
REFERENCE_CONTRACT = {
    "frames": 2, "quantizers": 16, "codebook_size": 1024,
    "sample_rate": 48000, "channels": 2, "frame_hop": 3840,
    # Shapes are authenticated by the model-free meta-device inspection. They
    # do not constitute a real-weight/API compatibility or parity approval.
    "quantizer_shape": "1x768x2", "decoder_tap_count": 9,
    "decoder_tap_shapes": [
        {"name": "decoder_0", "shape": "1x192x8"},
        {"name": "decoder_1", "shape": "1x768x8"},
        {"name": "decoder_2", "shape": "1x384x16"},
        {"name": "decoder_3", "shape": "1x768x16"},
        {"name": "decoder_4", "shape": "1x384x32"},
        {"name": "decoder_5", "shape": "1x768x32"},
        {"name": "decoder_6", "shape": "1x384x64"},
        {"name": "decoder_7", "shape": "1x240x64"},
        {"name": "decoder_8", "shape": "1x1x15360"},
    ],
}
SENTINELS = {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
LICENSE_SCHEMA = {"id", "license", "status", "conclusion", "native_bundled_review", "evidence"}
MATERIALIZED_MODEL_ROW_SCHEMA = {"path", "role", "bytes", "sha256", "status", "materialized", "content_not_downloaded", "server_size", "canonical_git_blob_sha1", "git_blob_sha1", "lfs_payload_sha256", "lfs_pointer_git_blob_sha1"}
SERVER_ONLY_MODEL_ROW_SCHEMA = {"path", "role", "bytes", "sha256", "status", "materialized", "content_not_downloaded", "server_bytes", "canonical_git_blob_sha1", "git_blob_sha1", "lfs_payload_sha256", "lfs_pointer_git_blob_sha1"}
LICENSE_IDS = ["source-apache", "weights-apache", "python-closure"]
PACKAGE_REVIEW_SCHEMA = {
    "name", "version", "source", "license", "status", "native_bundled_review",
}
MANIFEST_SCHEMA = {"gate_version", "lock_sha256", "project_sha256", "package_rows", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "license_rows", "license_rows_sha256", "model_rows", "model_rows_sha256", "upstream_repo", "upstream_revision", "model_info", "license_file_present", "reference_route", "reference_contract", "publication_decision", "approval"}
APPROVAL_SCHEMA = {"status", "signer", "digest"}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"dependencies", "name", "sdist", "source", "version", "wheels"}),
    frozenset({"name", "source", "version", "wheels"}),
    frozenset({"dependencies", "name", "source", "version", "wheels"}),
    frozenset({"dependencies", "metadata", "name", "source", "version"}),
}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
CPU_TORCH_VERSION = "2.7.1+cpu"
REGISTRY_HOSTS = {
    "https://pypi.org/simple": "files.pythonhosted.org",
    "https://download.pytorch.org/whl/cpu": "download-r2.pytorch.org",
}


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


def valid_model_row(row: object) -> bool:
    if not isinstance(row, dict) or not isinstance(row.get("materialized"), bool) or not isinstance(row.get("content_not_downloaded"), bool):
        return False
    if row["materialized"]:
        return (
            set(row) == MATERIALIZED_MODEL_ROW_SCHEMA
            and row["content_not_downloaded"] is False
            and isinstance(row.get("bytes"), int) and row["bytes"] > 0
            and row.get("server_size") == row["bytes"]
            and isinstance(row.get("sha256"), str) and HEX64.fullmatch(row["sha256"]) is not None
            and isinstance(row.get("canonical_git_blob_sha1"), str) and re.fullmatch(r"[0-9a-f]{40}", row["canonical_git_blob_sha1"]) is not None
            and row.get("git_blob_sha1") == row["canonical_git_blob_sha1"]
            and row.get("lfs_payload_sha256") is None and row.get("lfs_pointer_git_blob_sha1") is None
        )
    return (
        set(row) == SERVER_ONLY_MODEL_ROW_SCHEMA
        and row["content_not_downloaded"] is True
        and row.get("bytes") is None and row.get("sha256") is None
        and row.get("canonical_git_blob_sha1") is None and row.get("git_blob_sha1") is None
        and isinstance(row.get("server_bytes"), int) and row["server_bytes"] > 0
        and isinstance(row.get("lfs_payload_sha256"), str) and HEX64.fullmatch(row["lfs_payload_sha256"]) is not None
        and isinstance(row.get("lfs_pointer_git_blob_sha1"), str) and re.fullmatch(r"[0-9a-f]{40}", row["lfs_pointer_git_blob_sha1"]) is not None
    )


def lock_rows(lock: dict) -> list[dict]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or type(lock.get("version")) is not int or lock.get("revision") != 3 or type(lock.get("revision")) is not int:
        raise ValueError("lock top-level schema drifted")
    if not isinstance(lock.get("requires-python"), str) or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) for item in lock["supported-markers"]):
        raise ValueError("lock marker schema malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise ValueError("lock package table missing/empty")
    rows = []
    seen = set()
    for package in packages:
        if not isinstance(package, dict) or frozenset(package) not in PACKAGE_KEYS:
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
        if not isinstance(dependencies, list) or any(not isinstance(dep, dict) or set(dep) != {"name", "marker"} or not isinstance(dep.get("name"), str) or not dep["name"] or not isinstance(dep["marker"], str) for dep in dependencies):
            raise ValueError("malformed lock dependency row")
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}):
            raise ValueError("malformed lock source")
        if "registry" in source and source["registry"] not in REGISTRY_HOSTS:
            raise ValueError("unsupported lock registry")
        if "virtual" in source and source["virtual"] != ".":
            raise ValueError("malformed virtual source")
        if "virtual" in source:
            metadata = package.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list) or any(not isinstance(req, dict) or set(req) not in ({"name", "specifier"}, {"index", "name", "specifier"}) or not isinstance(req.get("name"), str) or not req["name"] or not isinstance(req.get("specifier"), str) or ("index" in req and req["index"] != "https://download.pytorch.org/whl/cpu") for req in metadata["requires-dist"]):
                raise ValueError("malformed virtual metadata")
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
        if frozenset(package) not in PACKAGE_KEYS:
            return f"package {package.get('name')!r} has an inexact row schema"
        if "sdist" in package and not isinstance(package["sdist"], dict):
            return f"package {package.get('name')!r} has malformed sdist"
        if "wheels" in package and not isinstance(package["wheels"], list):
            return f"package {package.get('name')!r} has malformed wheels"
        if source == {"virtual": "."}:
            if "sdist" in package or "wheels" in package:
                return "virtual project source cannot carry resolver artifacts"
            if set(package) != frozenset({"dependencies", "metadata", "name", "source", "version"}) or not isinstance(package.get("metadata"), dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                return "virtual project metadata schema is not exact"
            virtual_count += 1
            continue
        if set(source) != {"registry"} or source.get("registry") not in REGISTRY_HOSTS:
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
            # The official PyTorch CPU simple index currently omits size for
            # this exact wheel. Keep the missing byte fact fail-closed so the
            # VAST audit/owner review can resolve it without invention.
            cpu_torch_missing_size = isinstance(artifact, dict) and (
                package.get("name") == "torch"
                and package.get("version") == CPU_TORCH_VERSION
                and source == {"registry": CPU_TORCH_INDEX}
                and "size" not in artifact
                and set(artifact) == {"url", "hash", "upload-time"}
            )
            if not isinstance(artifact, dict) or (set(artifact) != ARTIFACT_KEYS and not cpu_torch_missing_size):
                return f"package {package.get('name')!r} has malformed artifact"
            parsed = None
            try:
                parsed = urlparse(artifact["url"])
                hostname = parsed.hostname
            except (TypeError, ValueError):
                hostname = None
            expected_host = REGISTRY_HOSTS[source["registry"]]
            if (parsed is None or not isinstance(artifact["url"], str) or parsed.scheme != "https" or hostname != expected_host or parsed.netloc != hostname or not parsed.path.startswith("/") or parsed.query or parsed.fragment):
                return f"package {package.get('name')!r} has invalid artifact URL"
            if not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                return f"package {package.get('name')!r} has invalid artifact hash"
            if not cpu_torch_missing_size and (not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0):
                return f"package {package.get('name')!r} has invalid artifact size"
            if not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip():
                return f"package {package.get('name')!r} has invalid artifact upload-time"
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
        "model_info": manifest.get("model_info"),
        "license_file_present": manifest.get("license_file_present"),
        "reference_route": manifest.get("reference_route"),
        "reference_contract": manifest.get("reference_contract"),
        "publication_decision": manifest.get("publication_decision"),
        "expected_decision": "APPROVED",
    })


def blocked(message: str) -> None:
    print(f"moss Nano license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def verify_snapshot(snapshot: Path, manifest_path: Path) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir():
        blocked(f"snapshot directory is missing: {snapshot}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        blocked("snapshot manifest is missing or not a regular file")
    try:
        manifest = load_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        blocked(f"invalid snapshot manifest: {error}")
    if not isinstance(manifest, dict):
        blocked("snapshot manifest top-level value must be an object")
    if manifest.get("model_info") != MODEL_INFO:
        blocked("snapshot manifest HF model_info identity drifted")
    if manifest.get("license_file_present") is not False:
        blocked("snapshot manifest did not record fixed revision LICENSE-file absence")
    rows = manifest.get("model_rows")
    if not isinstance(rows, list) or [row.get("path") for row in rows if isinstance(row, dict)] != list(PAYLOAD_FILES):
        blocked("snapshot identity table is not the exact seven-non-weight-plus-shard contract")
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
        if not valid_model_row(row) or row != FILE_IDENTITIES.get(name):
            blocked(f"snapshot identity unresolved or drifted: {name}")
        if row["materialized"] and (path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]):
            blocked(f"snapshot identity mismatch: {name}")
        if not row["materialized"] and (
            path.stat().st_size != row["server_bytes"]
            or sha_file(path) != row["lfs_payload_sha256"]
        ):
            blocked(f"server-only snapshot LFS identity mismatch: {name}")
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
    if manifest.get("model_info") != MODEL_INFO:
        blocked("HF model_info identity drifted")
    if manifest.get("license_file_present") is not False:
        blocked("fixed revision LICENSE-file absence was not recorded")
    model_rows = manifest.get("model_rows")
    if not isinstance(model_rows, list) or len(model_rows) != len(PAYLOAD_FILES):
        blocked("Nano payload identity table must contain exactly seven non-weight files plus one shard")
    if [row.get("path") for row in model_rows if isinstance(row, dict)] != list(PAYLOAD_FILES):
        blocked("Nano payload files are missing, duplicated, reordered, or extra")
    if any(not valid_model_row(row) for row in model_rows):
        blocked("Nano payload identity schema or materialization state drifted")
    if manifest.get("model_rows_sha256") != canon(model_rows):
        blocked("Nano payload identity digest drifted")
    for row in model_rows:
        expected = FILE_IDENTITIES[row["path"]]
        if row != expected:
            blocked(f"Nano payload identity unresolved or drifted: {row.get('path')}")

    licenses = manifest.get("license_rows")
    if not isinstance(licenses, list) or len(licenses) != 3 or [row.get("id") for row in licenses if isinstance(row, dict)] != LICENSE_IDS:
        blocked("license rows are missing, duplicated, reordered, or extra")
    if any(not isinstance(row, dict) or set(row) != LICENSE_SCHEMA for row in licenses):
        blocked("license row schema drifted")
    if manifest.get("license_rows_sha256") != canon(licenses):
        blocked("license rows digest drifted")
    if any(row["status"] != "REVIEWED" or not resolved(row["license"]) or not resolved(row["conclusion"]) or not resolved(row["native_bundled_review"]) or not resolved(row["evidence"]) for row in licenses):
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
        virtual = '\n[[package]]\nname="demo"\nversion="0.1.0"\nsource={virtual="."}\ndependencies=[]\n[package.metadata]\nrequires-dist=[]\n'
        lock_path.write_text('version=1\nrevision=3\nrequires-python="==3.12.*"\nresolution-markers=[]\nsupported-markers=[]\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/packages/demo.tar.gz",hash="sha256:' + "a" * 64 + '",size=1,upload-time="2026-01-01T00:00:00Z"}\nwheels=[{url="https://files.pythonhosted.org/packages/demo.whl",hash="sha256:' + "b" * 64 + '",size=2,upload-time="2026-01-01T00:00:00Z"}]\n' + virtual, encoding="utf-8")
        project_path.write_text('[project]\nname="demo"\nversion="0.1.0"\n', encoding="utf-8")
        valid_lock = tomllib.loads(lock_path.read_text())
        for label, mutate in (("top-extra", lambda value: value.update(unexpected=True)), ("top-missing", lambda value: value.pop("revision")), ("package-extra", lambda value: value["package"][0].update(unexpected=True))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            try:
                lock_rows(candidate)
            except ValueError:
                continue
            raise SystemExit(f"self-test accepted malformed lock schema: {label}")
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
        for field in ("url", "hash", "size", "upload-time"):
            candidate = load_json(json.dumps(valid_lock)); candidate["package"][0]["sdist"].pop(field)
            if artifact_error(candidate) is None:
                raise SystemExit(f"self-test accepted missing artifact field: {field}")
        for label, mutate in (("extra-artifact-field", lambda value: value["package"][0]["sdist"].update(extra=True)), ("empty-upload-time", lambda value: value["package"][0]["sdist"].update(**{"upload-time": " "})), ("evil-host", lambda value: value["package"][0]["sdist"].update(url="https://evil.example/packages/demo.tar.gz"))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            if artifact_error(candidate) is None:
                raise SystemExit(f"self-test accepted malformed artifact: {label}")
        PAYLOAD_FILES = ("demo", "weights")
        FILE_IDENTITIES = {
            "demo": {"path": "demo", "role": "upstream", "bytes": 3, "sha256": sha(b"abc"), "status": "AUTHENTICATED", "materialized": True, "content_not_downloaded": False, "server_size": 3, "canonical_git_blob_sha1": "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f", "git_blob_sha1": "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f", "lfs_payload_sha256": None, "lfs_pointer_git_blob_sha1": None},
            "weights": {"path": "weights", "role": "weights", "bytes": None, "sha256": None, "status": "AUTHENTICATED_SERVER_IDENTITY_ONLY", "materialized": False, "content_not_downloaded": True, "server_bytes": 6, "lfs_payload_sha256": "0844df8fee9eeadaa344bc7e1a7ae769602eba35e773c2093c8fb9c3e5ead14e", "canonical_git_blob_sha1": None, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": "ffffffffffffffffffffffffffffffffffffffff"},
        }
        ROUTE = {"status": "REVIEWED", "transformers_version": "5.10.4", "reason": "owner evidence"}
        LOCK_SHA256, PROJECT_SHA256 = sha(lock_path.read_bytes()), sha(project_path.read_bytes())
        rows = lock_rows(tomllib.loads(lock_path.read_text()))
        review = [{"name": "demo", "version": "1", "source": {"registry": "https://pypi.org/simple"}, "license": "MIT", "status": "REVIEWED", "native_bundled_review": "reviewed"}, {"name": "demo", "version": "0.1.0", "source": {"virtual": "."}, "license": "project", "status": "REVIEWED", "native_bundled_review": "reviewed"}]
        licenses = [{"id": ident, "license": "Apache-2.0", "status": "REVIEWED", "conclusion": "reviewed", "native_bundled_review": "reviewed", "evidence": "self-test owner evidence"} for ident in LICENSE_IDS]
        model_rows = list(FILE_IDENTITIES.values())
        manifest = {"gate_version": 1, "lock_sha256": LOCK_SHA256, "project_sha256": PROJECT_SHA256, "package_rows": rows, "package_rows_sha256": canon(rows), "package_review_rows": review, "package_review_rows_sha256": canon(review), "license_rows": licenses, "license_rows_sha256": canon(licenses), "model_rows": model_rows, "model_rows_sha256": canon(model_rows), "upstream_repo": REPO, "upstream_revision": REVISION, "model_info": MODEL_INFO, "license_file_present": False, "reference_route": ROUTE, "reference_contract": REFERENCE_CONTRACT, "publication_decision": "NO_UPLOAD", "approval": {"status": "OWNER_SIGNOFF_APPROVED", "signer": "owner", "digest": None}}
        manifest["approval"]["digest"] = scope(manifest); manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        evidence = {"scope_schema": "moss-audio-tokenizer-nano-approval-v1", "scope_sha256": manifest["approval"]["digest"], "approval_digest": manifest["approval"]["digest"], "decision": "APPROVED", "signer": "owner", "manifest_sha256": sha(manifest_path.read_bytes())}; evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        snapshot = root / "snapshot"; snapshot.mkdir(); (snapshot / "demo").write_bytes(b"abc"); (snapshot / "weights").write_bytes(b"weight")
        (snapshot / ".cache" / "huggingface").mkdir(parents=True)
        verify_snapshot(snapshot, manifest_path)
        snapshot_manifest = manifest_path.read_text(encoding="utf-8")
        for label, payload in (("snapshot-manifest-duplicate", '{"model_rows":[],"model_rows":[]}'), ("snapshot-manifest-non-object", "[]")):
            manifest_path.write_text(payload, encoding="utf-8")
            try:
                verify_snapshot(snapshot, manifest_path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"self-test accepted {label}")
            manifest_path.write_text(snapshot_manifest, encoding="utf-8")
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
        (snapshot / "weights").write_bytes(b"weighx")
        try:
            verify_snapshot(snapshot, manifest_path)
        except SystemExit as error:
            if error.code != 2:
                raise
        else:
            raise SystemExit("self-test accepted same-size server-only shard tamper")
        (snapshot / "weights").write_bytes(b"weight")
        verify_snapshot(snapshot, manifest_path)
        run(lock_path, project_path, manifest_path, evidence_path)
        baseline_manifest = manifest_path.read_text(encoding="utf-8")
        baseline_evidence = evidence_path.read_text(encoding="utf-8")
        for label, target, payload in (
            ("manifest-duplicate", manifest_path, '{"gate_version":1,"gate_version":1}'),
            ("manifest-nested-duplicate", manifest_path, '{"model_rows":{"path":"a","path":"b"}}'),
            ("evidence-duplicate", evidence_path, '{"decision":"APPROVED","decision":"PENDING"}'),
            ("evidence-nested-duplicate", evidence_path, '{"scope":{"a":1,"a":2}}'),
        ):
            target.write_text(payload, encoding="utf-8")
            try:
                run(lock_path, project_path, manifest_path, evidence_path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"self-test accepted {label}")
            target.write_text(baseline_manifest if target is manifest_path else baseline_evidence, encoding="utf-8")
        for label, target in (("manifest-non-object", manifest_path), ("evidence-non-object", evidence_path)):
            target.write_text("[]", encoding="utf-8")
            try:
                run(lock_path, project_path, manifest_path, evidence_path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise SystemExit(f"self-test accepted {label}")
            target.write_text(baseline_manifest if target is manifest_path else baseline_evidence, encoding="utf-8")
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
        for label, mutate in (("artifact", None), ("scope", lambda m: m["reference_contract"].update(frames=1)), ("model", lambda m: m["model_rows"][0].update(status="UNRESOLVED")), ("model-shard-materialized", lambda m: m["model_rows"][-1].update(materialized=True)), ("model-info", lambda m: m["model_info"].update(sha="a" * 40)), ("license-file", lambda m: m.update(license_file_present=True)), ("license", lambda m: m["license_rows"][0].update(conclusion="TODO")), ("publication", lambda m: m.update(publication_decision="UPLOAD")), ("arbitrary", lambda m: m["approval"].update(digest="a" * 64)), ("package-schema", lambda m: m["package_review_rows"][0].update(extra="drift")), ("manifest-schema", lambda m: m.update(extra=True)), ("approval-schema", lambda m: m["approval"].update(extra=True))):
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
    parser.add_argument("--lock", type=Path); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--approval-evidence", dest="approval", type=Path)
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
