#!/usr/bin/env python3
"""Audit the pinned CLAP API and metadata without acquiring model weights.

This audit is intentionally separate from ``clap_dump_reference.py``.  It
loads only the official Transformers configuration and feature-extractor
classes, and consumes metadata JSON files staged by the shell worker.  It
never calls ``from_pretrained`` for a model, imports ``ClapModel`` for a
forward, or accepts a checkpoint path.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import sys
import tempfile
import tomllib
import zipfile
from io import StringIO
from pathlib import Path
from typing import Any


MODEL_REPOSITORY = "laion/clap-htsat-fused"
MODEL_REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
# Compatibility aliases are kept for the shared local contract, but evidence
# always serializes model and Transformers identities as separate fields.
REPOSITORY = MODEL_REPOSITORY
REVISION = MODEL_REVISION
TRANSFORMERS_VERSION = "5.10.4"
TRANSFORMERS_WHEEL_SHA256 = (
    "8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e"
)
TRANSFORMERS_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/d7/f1/d66881f28d3e64002a21d043c7c8db306c0ad5a711c85337ff551bfbc040/transformers-5.10.4-py3-none-any.whl"
)
TRANSFORMERS_WHEEL_SIZE = 11_004_075
SCHEMA = "vokra-clap-htsat-fused-model-free-audit-v1"
SOURCE_CONTRACT_SCHEMA = "vokra-clap-htsat-fused-source-contract-v2"
EXPECTED_DEPENDENCY_AUDIT_STATUS = "PENDING_VAST_AUDIT"
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
SOURCE_SHA256_HEX_LENGTH = 64
SOURCE_STATUS_PENDING = "PENDING_VAST_WHEEL_BINDING"
SOURCE_STATUS_AUTHENTICATED = "AUTHENTICATED_LOCKED_WHEEL_SOURCE"
WHEEL_BINDING_STATUS = "PASS_LOCKED_WHEEL_BINDING"
SOURCE_REFERENCE_STATUS = "PASS_SOURCE_ONLY"
TOKENIZER_FILES = ("tokenizer_config.json", "vocab.json", "merges.txt")
TRANSFORMERS_SOURCE_MEMBERS = {
    "feature_extraction_clap": "transformers/models/clap/feature_extraction_clap.py",
    "processing_clap": "transformers/models/clap/processing_clap.py",
    "processing_utils": "transformers/processing_utils.py",
    "modeling_clap": "transformers/models/clap/modeling_clap.py",
    "configuration_clap": "transformers/models/clap/configuration_clap.py",
    "roberta_tokenizer": "transformers/models/roberta/tokenization_roberta.py",
}

# These are the official, model-independent preprocessing entrypoints.  The
# audit records their source identities rather than reimplementing or running
# the audio path.  A method disappearing or being renamed therefore blocks the
# packet instead of silently reducing the authenticated surface.
SOURCE_METHODS = {
    "audio": ("__call__", "_get_input_mel", "_np_extract_fbank_features"),
    "text": ("__init__",),
}


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def local_file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": git_blob_sha1(data),
    }


def normalize_card_data(info: Any) -> tuple[dict[str, Any], str]:
    """Normalize the public ModelInfo.card_data value and retain its source."""

    raw = getattr(info, "card_data", None)
    if raw is None:
        return {}, "info.card_data(None)"
    if isinstance(raw, dict):
        return dict(raw), "info.card_data"
    to_dict = getattr(raw, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return dict(converted), "info.card_data.to_dict"
    raise RuntimeError("ModelInfo.card_data is neither a dict nor to_dict mapping")


def read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing, symlinked, or empty: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def validate_dependency_inventory(path: Path) -> dict[str, Any]:
    inventory = read_json(path, "dependency license inventory")
    if inventory.get("schema") != "vokra-clap-htsat-fused-dependency-license-inventory-v1":
        raise RuntimeError("dependency license inventory schema drifted")
    if inventory.get("status") not in {"BLOCKED", "PENDING_OWNER_REVIEW"}:
        raise RuntimeError("dependency license inventory status is not fail-closed")
    if inventory.get("dependency_audit_status") != EXPECTED_DEPENDENCY_AUDIT_STATUS:
        raise RuntimeError(
            "dependency license inventory status is not the expected "
            f"{EXPECTED_DEPENDENCY_AUDIT_STATUS}"
        )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "status": inventory["status"],
        "dependency_audit_status": inventory["dependency_audit_status"],
        "findings": inventory.get("findings", []),
    }


def write_atomic_no_replace(path: Path, text: str) -> None:
    """Create a regular output file without ever replacing an existing path."""

    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"output already exists: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_no_weights(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"metadata directory is missing or symlinked: {directory}")
    forbidden = sorted(
        path.name
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    if forbidden:
        raise RuntimeError(f"model-free metadata directory contains weights: {forbidden}")


def load_reference_contract() -> tuple[dict[str, Any], Any, Any, Any, Any]:
    """Load the shared contract without importing torch or a model class."""

    contract_path = Path(__file__).resolve().parents[1] / "clap_dump_reference.py"
    spec = importlib.util.spec_from_file_location("vokra_clap_dump_reference", contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared CLAP contract: {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.PREPROCESSOR_CONTRACT,
        module.validate_preprocessor_contract,
        module.validate_model_config,
        module.validate_feature_extractor_serializer_contract,
        module.validate_tensor_manifest,
    )


def parse_record(record_text: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in csv.reader(StringIO(record_text)):
        if len(row) != 3 or not row[0]:
            raise RuntimeError("Transformers wheel RECORD row is malformed")
        path, digest, size = row
        if path in records:
            raise RuntimeError(f"Transformers wheel RECORD has duplicate path: {path}")
        if path.endswith(".dist-info/RECORD") and digest == "" and size == "":
            parsed_size = None
        else:
            try:
                parsed_size = int(size)
            except ValueError as exc:
                raise RuntimeError(f"Transformers wheel RECORD size is invalid: {path}") from exc
            if parsed_size < 0:
                raise RuntimeError(f"Transformers wheel RECORD size is negative: {path}")
        records[path] = {"hash": digest, "size": parsed_size}
    return records


def record_hash_for_bytes(value: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode("ascii").rstrip("=")
    return f"sha256={digest}"


def validate_record_identity(
    records: dict[str, dict[str, Any]], path: str, value: bytes, *, label: str
) -> dict[str, Any]:
    record = records.get(path)
    if record is None:
        raise RuntimeError(f"{label} RECORD is missing {path}")
    if record["size"] != len(value):
        raise RuntimeError(f"{label} RECORD size differs for {path}")
    expected_hash = record_hash_for_bytes(value)
    if record["hash"] != expected_hash:
        raise RuntimeError(f"{label} RECORD hash differs for {path}")
    return {
        "path": path,
        "size": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "record_hash": record["hash"],
        "record_size": record["size"],
    }


def validate_archive_member_name(name: str) -> None:
    if not name or "\\" in name or "\x00" in name or name.startswith("/"):
        raise RuntimeError(f"Transformers wheel member path is unsafe: {name!r}")
    parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"Transformers wheel member path is unsafe: {name!r}")


def canonical_tree_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_transformers_wheel(
    wheel_path: Path, expected_artifact: dict[str, Any]
) -> dict[str, Any]:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise RuntimeError(f"Transformers wheel is missing or symlinked: {wheel_path}")
    wheel_bytes_hash = sha256_file(wheel_path)
    expected_hash = expected_artifact.get("hash")
    expected_size = expected_artifact.get("size")
    if expected_hash != f"sha256:{TRANSFORMERS_WHEEL_SHA256}":
        raise RuntimeError("uv.lock Transformers wheel hash differs from fixed wheel identity")
    if expected_artifact.get("url") != TRANSFORMERS_WHEEL_URL or expected_size != TRANSFORMERS_WHEEL_SIZE:
        raise RuntimeError("uv.lock Transformers wheel URL or size differs from fixed identity")
    if wheel_bytes_hash != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("downloaded Transformers wheel SHA-256 differs from locked identity")
    if wheel_path.stat().st_size != TRANSFORMERS_WHEEL_SIZE:
        raise RuntimeError("downloaded Transformers wheel size differs from locked identity")

    try:
        archive = zipfile.ZipFile(wheel_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Transformers wheel is not a readable ZIP archive") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("Transformers wheel has duplicate archive members")
        for info in infos:
            validate_archive_member_name(info.filename)
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise RuntimeError(f"Transformers wheel contains a symlink member: {info.filename}")
        record_name = "transformers-5.10.4.dist-info/RECORD"
        if record_name not in names:
            raise RuntimeError("Transformers wheel RECORD is missing")
        records = parse_record(archive.read(record_name).decode("utf-8"))
        source_members = sorted(
            name for name in names if name.startswith("transformers/") and name.endswith(".py")
        )
        if not source_members:
            raise RuntimeError("Transformers wheel has no Python source members")
        archive_tree: list[dict[str, Any]] = []
        archive_by_path: dict[str, dict[str, Any]] = {}
        for member in source_members:
            data = archive.read(member)
            row = validate_record_identity(records, member, data, label="wheel")
            archive_tree.append(row)
            archive_by_path[member] = row
        archive_members: dict[str, dict[str, Any]] = {}
        for label, member in TRANSFORMERS_SOURCE_MEMBERS.items():
            if member not in names:
                raise RuntimeError(f"Transformers wheel is missing source member {member}")
            archive_members[label] = {
                **archive_by_path[member],
                "record": archive_by_path[member],
            }

    distribution = importlib.metadata.distribution("transformers")
    if distribution.version != TRANSFORMERS_VERSION:
        raise RuntimeError("installed Transformers distribution version drifted")
    installed_record_text = distribution.read_text("RECORD")
    if not installed_record_text:
        raise RuntimeError("installed Transformers distribution has no readable RECORD")
    installed_records = parse_record(installed_record_text)
    installed_tree: list[dict[str, Any]] = []
    installed_by_path: dict[str, dict[str, Any]] = {}
    for member in source_members:
        installed_path = Path(distribution.locate_file(member))
        if installed_path.is_symlink() or not installed_path.is_file():
            raise RuntimeError(f"installed Transformers source member is missing or symlinked: {member}")
        data = installed_path.read_bytes()
        installed_record = validate_record_identity(
            installed_records, member, data, label="installed"
        )
        archive_member = archive_by_path[member]
        if installed_record != archive_member:
            raise RuntimeError(f"installed Transformers source differs from wheel: {member}")
        installed_tree.append(installed_record)
        installed_by_path[member] = installed_record
    archive_tree_digest = canonical_tree_digest(archive_tree)
    installed_tree_digest = canonical_tree_digest(installed_tree)
    if archive_tree_digest != installed_tree_digest:
        raise RuntimeError("installed Transformers Python tree digest differs from wheel")
    installed_members: dict[str, dict[str, Any]] = {
        label: {
            **installed_by_path[member],
            "installed_path": str(Path(distribution.locate_file(member)).resolve()),
            "record": installed_by_path[member],
        }
        for label, member in TRANSFORMERS_SOURCE_MEMBERS.items()
    }
    return {
        "status": WHEEL_BINDING_STATUS,
        "wheel": {
            "path": str(wheel_path.resolve()),
            "filename": wheel_path.name,
            "url": TRANSFORMERS_WHEEL_URL,
            "size": TRANSFORMERS_WHEEL_SIZE,
            "sha256": TRANSFORMERS_WHEEL_SHA256,
        },
        "distribution": {
            "name": distribution.metadata.get("Name"),
            "version": distribution.version,
            "record": "RECORD",
        },
        "archive_members": archive_members,
        "installed_members": installed_members,
        "python_source_tree": {
            "member_count": len(source_members),
            "canonical_tree_sha256": archive_tree_digest,
            "archive_rows": archive_tree,
            "installed_rows": installed_tree,
        },
    }


def audit_dependencies(project_path: Path, lock_path: Path) -> dict[str, Any]:
    if project_path.is_symlink() or not project_path.is_file():
        raise RuntimeError(f"pyproject.toml is missing or symlinked: {project_path}")
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"uv.lock is missing or symlinked: {lock_path}")
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    expected_dependencies = {
        "numpy==2.3.5",
        "torch==2.7.1",
        "transformers==5.10.4",
    }
    actual_dependencies = set(project["project"]["dependencies"])
    if actual_dependencies != expected_dependencies:
        raise RuntimeError(
            f"CLAP dependency contract drifted: {sorted(actual_dependencies)}"
        )
    packages = {
        package["name"]: package
        for package in lock.get("package", [])
        if isinstance(package, dict) and "name" in package
    }
    for name, version in (("numpy", "2.3.5"), ("torch", "2.7.1+cpu"), ("transformers", "5.10.4")):
        if packages.get(name, {}).get("version") != version:
            raise RuntimeError(f"uv.lock {name} drifted: {packages.get(name)}")
    transformers_package = packages["transformers"]
    transformer_wheels = [
        wheel for wheel in transformers_package.get("wheels", [])
        if isinstance(wheel, dict) and wheel.get("url") == TRANSFORMERS_WHEEL_URL
    ]
    if len(transformer_wheels) != 1:
        raise RuntimeError("uv.lock must contain exactly one fixed Transformers wheel")
    transformers_artifact = transformer_wheels[0]
    torch_source = packages["torch"].get("source", {})
    if torch_source.get("registry") != "https://download.pytorch.org/whl/cpu":
        raise RuntimeError(f"CLAP torch source is not the pinned CPU index: {torch_source}")
    values = project["tool"]["vokra"]["clap_reference"]
    if values["isolated_transformers_pin"] != TRANSFORMERS_VERSION:
        raise RuntimeError("pyproject Transformers pin drifted")
    if values["transformers_wheel_sha256"] != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("pyproject Transformers wheel hash drifted")
    if values.get("source_contract_status") != SOURCE_STATUS_PENDING:
        raise RuntimeError("pyproject source contract must remain pending before VAST wheel binding")
    return {
        "project_sha256": sha256_file(project_path),
        "lock_sha256": sha256_file(lock_path),
        "dependencies": sorted(actual_dependencies),
        "locked_versions": {
            name: packages[name]["version"] for name in ("numpy", "torch", "transformers")
        },
        "torch_index": torch_source["registry"],
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "transformers_wheel_artifact": transformers_artifact,
        "source_contract_status": values["source_contract_status"],
        "license_evidence_status": values["license_status"],
        "dependency_audit_status": values["dependency_audit_status"],
        "publication": values["publication"],
    }


def source_fact(cls: Any) -> dict[str, Any]:
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError(f"cannot locate source for {cls.__name__}")
    path = Path(source).resolve()
    return {
        "class": cls.__name__,
        "source": str(path),
        "source_sha256": sha256_file(path),
        "signature": str(inspect.signature(cls)),
    }


def source_method_fact(owner: Any, method_name: str) -> dict[str, str]:
    method = getattr(owner, method_name, None)
    if method is None or not callable(method):
        raise RuntimeError(
            f"official CLAP source is missing callable {owner.__name__}.{method_name}"
        )
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"cannot inspect official CLAP source {owner.__name__}.{method_name}"
        ) from exc
    return {
        "signature": str(inspect.signature(method)),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def processor_call_fact(processor: Any, processor_base: Any) -> dict[str, Any]:
    if "__call__" in getattr(processor, "__dict__", {}):
        owner = processor
        owner_name = "ClapProcessor"
    elif "__call__" in getattr(processor_base, "__dict__", {}):
        owner = processor_base
        owner_name = "ProcessorMixin"
    else:
        raise RuntimeError("official CLAP processor has no own or base __call__ implementation")
    return {
        "owner": owner_name,
        "source": source_fact(owner),
        "method": source_method_fact(owner, "__call__"),
    }


def build_source_contract(
    feature_extractor: Any,
    processor: Any,
    processor_base: Any,
    tokenizer: Any,
    wheel_binding: dict[str, Any],
    source_reference: dict[str, Any],
) -> dict[str, Any]:
    contract = {
        "schema": SOURCE_CONTRACT_SCHEMA,
        "status": SOURCE_STATUS_AUTHENTICATED,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "wheel_binding": wheel_binding,
        "source_reference": source_reference,
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "audio": {
            "class": source_fact(feature_extractor),
            "methods": {
                name: source_method_fact(feature_extractor, name)
                for name in SOURCE_METHODS["audio"]
            },
            "semantic_surface": "official_audio_feature_extractor_entrypoints",
        },
        "text": {
            "class": source_fact(processor),
            "methods": {
                name: source_method_fact(processor, name)
                for name in SOURCE_METHODS["text"]
            },
            "base_class": source_fact(processor_base),
            "call": processor_call_fact(processor, processor_base),
            "semantic_surface": "official_text_processor_binding_entrypoints",
        },
        "tokenizer": {
            "class": source_fact(tokenizer),
            "archive_member": TRANSFORMERS_SOURCE_MEMBERS["roberta_tokenizer"],
            "semantic_surface": "official_roberta_tokenizer_entrypoint",
        },
        "execution": "SOURCE_ONLY_REFERENCE_EXECUTED",
    }
    validate_source_contract(contract)
    return contract


def validate_source_contract(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise RuntimeError("CLAP source contract is not an object")
    if contract.get("schema") != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("CLAP source contract schema drifted")
    if contract.get("status") != SOURCE_STATUS_AUTHENTICATED:
        raise RuntimeError("CLAP source contract is not authenticated")
    if contract.get("model_repository") != MODEL_REPOSITORY:
        raise RuntimeError("CLAP model repository identity drifted")
    if contract.get("model_revision") != MODEL_REVISION:
        raise RuntimeError("CLAP model revision identity drifted")
    if contract.get("transformers_version") != TRANSFORMERS_VERSION:
        raise RuntimeError("CLAP source contract Transformers version drifted")
    if contract.get("transformers_wheel_sha256") != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("CLAP Transformers wheel identity drifted")
    if contract.get("weights") != "NOT_ACQUIRED" or contract.get("model_load") != "NOT_PERFORMED" or contract.get("model_forward") != "NOT_PERFORMED":
        raise RuntimeError("CLAP source contract overclaims model execution")
    wheel_binding = contract.get("wheel_binding")
    if not isinstance(wheel_binding, dict) or wheel_binding.get("status") != WHEEL_BINDING_STATUS:
        raise RuntimeError("CLAP source contract wheel binding is not complete")
    wheel = wheel_binding.get("wheel")
    if not isinstance(wheel, dict) or wheel.get("sha256") != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("CLAP source contract wheel bytes are not fixed")
    archive_members = wheel_binding.get("archive_members")
    installed_members = wheel_binding.get("installed_members")
    if not isinstance(archive_members, dict) or not isinstance(installed_members, dict):
        raise RuntimeError("CLAP source contract wheel member evidence is missing")
    source_tree = wheel_binding.get("python_source_tree")
    if not isinstance(source_tree, dict) or not isinstance(source_tree.get("member_count"), int):
        raise RuntimeError("CLAP full Transformers Python tree evidence is missing")
    archive_rows = source_tree.get("archive_rows")
    installed_rows = source_tree.get("installed_rows")
    if not isinstance(archive_rows, list) or not isinstance(installed_rows, list):
        raise RuntimeError("CLAP full Transformers Python tree rows are missing")
    if len(archive_rows) != source_tree["member_count"] or archive_rows != installed_rows:
        raise RuntimeError("CLAP full Transformers Python tree differs between wheel and install")
    if source_tree.get("canonical_tree_sha256") != canonical_tree_digest(archive_rows):
        raise RuntimeError("CLAP full Transformers Python tree digest is invalid")
    tree_paths: list[str] = []
    for row in archive_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("CLAP full Transformers Python tree row is malformed")
        path = row["path"]
        validate_archive_member_name(path)
        if not path.startswith("transformers/") or not path.endswith(".py"):
            raise RuntimeError("CLAP full Transformers Python tree contains a non-source path")
        if not isinstance(row.get("size"), int) or row["size"] < 1:
            raise RuntimeError("CLAP full Transformers Python tree contains an empty source")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != SOURCE_SHA256_HEX_LENGTH:
            raise RuntimeError("CLAP full Transformers Python tree hash is malformed")
        tree_paths.append(path)
    if len(tree_paths) != len(set(tree_paths)):
        raise RuntimeError("CLAP full Transformers Python tree contains duplicate paths")
    tree_by_path = {row["path"]: row for row in archive_rows}
    for label, member in TRANSFORMERS_SOURCE_MEMBERS.items():
        archive = archive_members.get(label)
        installed = installed_members.get(label)
        if not isinstance(archive, dict) or archive.get("path") != member:
            raise RuntimeError(f"CLAP wheel archive member is missing: {member}")
        if not isinstance(installed, dict) or installed.get("path") != member:
            raise RuntimeError(f"CLAP installed source member is missing: {member}")
        tree_row = tree_by_path.get(member)
        if not isinstance(tree_row, dict):
            raise RuntimeError(f"CLAP selected source member is absent from the full tree: {member}")
        for evidence in (archive, installed):
            if any(evidence.get(key) != tree_row.get(key) for key in ("path", "size", "sha256")):
                raise RuntimeError(f"CLAP selected source member disagrees with the full tree: {member}")
        if archive.get("sha256") != installed.get("sha256"):
            raise RuntimeError(f"CLAP installed source differs from wheel: {member}")
    source_reference = contract.get("source_reference")
    if not isinstance(source_reference, dict) or source_reference.get("status") != SOURCE_REFERENCE_STATUS:
        raise RuntimeError("CLAP source contract lacks executed source-only reference")
    for field in ("weights", "model_load", "model_forward"):
        if source_reference.get(field) not in {
            "NOT_ACQUIRED" if field == "weights" else "NOT_PERFORMED"
        }:
            raise RuntimeError("CLAP source reference overclaims model execution")
    if contract.get("execution") != "SOURCE_ONLY_REFERENCE_EXECUTED":
        raise RuntimeError("CLAP source contract execution mode drifted")
    reference_sources = source_reference.get("transformers_sources")
    if not isinstance(reference_sources, dict) or set(reference_sources) != {
        "feature_extractor",
        "processor",
        "tokenizer",
    }:
        raise RuntimeError("CLAP source reference source facts are missing")
    for key, label, class_name in (
        ("feature_extractor", "feature_extraction_clap", "ClapFeatureExtractor"),
        ("processor", "processing_clap", "ClapProcessor"),
        ("tokenizer", "roberta_tokenizer", "RobertaTokenizer"),
    ):
        fact = reference_sources.get(key)
        member = TRANSFORMERS_SOURCE_MEMBERS[label]
        if not isinstance(fact, dict) or fact.get("class") != class_name:
            raise RuntimeError(f"CLAP source reference {key} source fact is malformed")
        if not isinstance(fact.get("source"), str) or not fact["source"].endswith(member):
            raise RuntimeError(f"CLAP source reference {key} source path is not wheel-bound")
        if fact.get("source_sha256") != archive_members[label]["sha256"]:
            raise RuntimeError(f"CLAP source reference {key} source differs from wheel")
    expected_surfaces = {
        "audio": "official_audio_feature_extractor_entrypoints",
        "text": "official_text_processor_binding_entrypoints",
    }
    for modality, method_names in SOURCE_METHODS.items():
        entry = contract.get(modality)
        if not isinstance(entry, dict) or entry.get("semantic_surface") != expected_surfaces[modality]:
            raise RuntimeError(f"CLAP {modality} source contract surface is malformed")
        cls = entry.get("class")
        if not isinstance(cls, dict) or not isinstance(cls.get("class"), str):
            raise RuntimeError(f"CLAP {modality} source class fact is malformed")
        for key in ("source", "source_sha256", "signature"):
            if not isinstance(cls.get(key), str) or not cls[key]:
                raise RuntimeError(f"CLAP {modality} source class fact is missing {key}")
        if len(cls["source_sha256"]) != SOURCE_SHA256_HEX_LENGTH or any(
            char not in "0123456789abcdef" for char in cls["source_sha256"]
        ):
            raise RuntimeError(f"CLAP {modality} source class hash is malformed")
        methods = entry.get("methods")
        if not isinstance(methods, dict) or set(methods) != set(method_names):
            raise RuntimeError(f"CLAP {modality} source method set drifted")
        for name in method_names:
            fact = methods[name]
            if not isinstance(fact, dict) or not isinstance(fact.get("signature"), str):
                raise RuntimeError(f"CLAP {modality}.{name} source fact is malformed")
            digest = fact.get("source_sha256")
            if not isinstance(digest, str) or len(digest) != SOURCE_SHA256_HEX_LENGTH or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise RuntimeError(f"CLAP {modality}.{name} source hash is malformed")
        if modality == "text":
            base_class = entry.get("base_class")
            if not isinstance(base_class, dict) or base_class.get("class") != "ProcessorMixin":
                raise RuntimeError("CLAP text processor base source fact is malformed")
            for key in ("source", "source_sha256", "signature"):
                if not isinstance(base_class.get(key), str) or not base_class[key]:
                    raise RuntimeError(f"CLAP text processor base source fact is missing {key}")
            base_hash = base_class["source_sha256"]
            if len(base_hash) != SOURCE_SHA256_HEX_LENGTH or any(
                char not in "0123456789abcdef" for char in base_hash
            ):
                raise RuntimeError("CLAP text processor base source hash is malformed")
            call = entry.get("call")
            if not isinstance(call, dict) or call.get("owner") not in {"ClapProcessor", "ProcessorMixin"}:
                raise RuntimeError("CLAP processor call owner is malformed")
            call_source = call.get("source")
            call_method = call.get("method")
            if not isinstance(call_source, dict) or not isinstance(call_method, dict):
                raise RuntimeError("CLAP processor call source fact is missing")
            call_owner = call["owner"]
            call_member = (
                TRANSFORMERS_SOURCE_MEMBERS["processing_clap"]
                if call_owner == "ClapProcessor"
                else TRANSFORMERS_SOURCE_MEMBERS["processing_utils"]
            )
            if call_source.get("class") != call_owner or not isinstance(call_source.get("source"), str) or not call_source["source"].endswith(call_member):
                raise RuntimeError("CLAP processor call source path is not wheel-bound")
            if call_source.get("source_sha256") != archive_members["processing_clap" if call_owner == "ClapProcessor" else "processing_utils"]["sha256"]:
                raise RuntimeError("CLAP processor call source differs from wheel")
            call_hash = call_method.get("source_sha256")
            if not isinstance(call_method.get("signature"), str) or not isinstance(call_hash, str) or len(call_hash) != SOURCE_SHA256_HEX_LENGTH or any(char not in "0123456789abcdef" for char in call_hash):
                raise RuntimeError("CLAP processor call method source fact is malformed")
    source_module_bindings = {
        "audio": ("feature_extraction_clap", "class"),
        "text": ("processing_clap", "class"),
    }
    for modality, (label, _) in source_module_bindings.items():
        source_class = contract[modality]["class"]
        expected_member = TRANSFORMERS_SOURCE_MEMBERS[label]
        if not source_class["source"].endswith(expected_member):
            raise RuntimeError(f"CLAP {modality} source path is not wheel-bound")
        if source_class["source_sha256"] != archive_members[label]["sha256"]:
            raise RuntimeError(f"CLAP {modality} source hash differs from wheel")
    text_base = contract["text"]["base_class"]
    if not text_base["source"].endswith(TRANSFORMERS_SOURCE_MEMBERS["processing_utils"]):
        raise RuntimeError("CLAP ProcessorMixin source path is not wheel-bound")
    if text_base["source_sha256"] != archive_members["processing_utils"]["sha256"]:
        raise RuntimeError("CLAP ProcessorMixin source hash differs from wheel")
    tokenizer = contract.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise RuntimeError("CLAP tokenizer source contract is missing")
    tokenizer_member = TRANSFORMERS_SOURCE_MEMBERS["roberta_tokenizer"]
    if tokenizer.get("archive_member") != tokenizer_member:
        raise RuntimeError("CLAP tokenizer archive member drifted")
    tokenizer_class = tokenizer.get("class")
    if not isinstance(tokenizer_class, dict) or tokenizer_class.get("class") != "RobertaTokenizer":
        raise RuntimeError("CLAP tokenizer source class is not RobertaTokenizer")
    tokenizer_source = tokenizer_class.get("source")
    if not isinstance(tokenizer_source, str) or not tokenizer_source.endswith(tokenizer_member):
        raise RuntimeError("CLAP tokenizer source path is not the fixed official path")
    if tokenizer_class.get("source_sha256") != archive_members["roberta_tokenizer"]["sha256"]:
        raise RuntimeError("CLAP tokenizer source does not match the locked wheel")


def validate_remote_identity(
    remote_identity: dict[str, Any],
    *,
    config_path: Path,
    preprocessor_path: Path,
    metadata_dir: Path | None = None,
) -> None:
    if remote_identity.get("repository") != REPOSITORY:
        raise RuntimeError("remote identity repository drifted")
    if remote_identity.get("requested_revision") != REVISION:
        raise RuntimeError("remote identity requested revision drifted")
    if remote_identity.get("resolved_revision") != REVISION:
        raise RuntimeError("HfApi resolved revision drifted")
    if remote_identity.get("metadata_api") != "HfApi.model_info/list_repo_files":
        raise RuntimeError("remote identity was not obtained from the metadata API")
    if remote_identity.get("tree_api") != "HfApi.list_repo_tree(expand=True)":
        raise RuntimeError("remote identity tree was not obtained from the metadata API")
    if remote_identity.get("card_data_source") not in {
        "info.card_data",
        "info.card_data.to_dict",
        "info.card_data(None)",
    }:
        raise RuntimeError("remote identity card_data source is invalid")
    if remote_identity.get("card_data_license_status") not in {"PRESENT", "MISSING"}:
        raise RuntimeError("remote identity cardData license status is invalid")
    if remote_identity.get("repo_license_file_status") not in {"PRESENT", "MISSING"}:
        raise RuntimeError("remote identity repository LICENSE status is invalid")
    if remote_identity.get("weights") != "NOT_ACQUIRED":
        raise RuntimeError("remote identity does not prove weights were not acquired")
    remote_files = remote_identity.get("remote_files")
    if not isinstance(remote_files, dict):
        raise RuntimeError("remote metadata tree is missing")
    files = [
        ("config.json", config_path),
        ("preprocessor_config.json", preprocessor_path),
    ]
    if metadata_dir is not None:
        files.extend((filename, metadata_dir / filename) for filename in TOKENIZER_FILES)
    for filename, path in files:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"materialized metadata file is missing or symlinked: {filename}")
        entry = remote_files.get(filename)
        if not isinstance(entry, dict):
            raise RuntimeError(f"remote metadata tree is missing {filename}")
        local = local_file_identity(path)
        if entry.get("local_size") != local["size"]:
            raise RuntimeError(f"materialized {filename} size differs from remote packet")
        if entry.get("local_sha256") != local["sha256"]:
            raise RuntimeError(f"materialized {filename} SHA-256 differs from remote packet")
        if entry.get("local_git_blob_sha1") != local["git_blob_sha1"]:
            raise RuntimeError(f"materialized {filename} Git blob differs from remote packet")
        if entry.get("remote_size") != local["size"]:
            raise RuntimeError(f"materialized {filename} size differs from remote tree")
        lfs_sha256 = entry.get("remote_lfs_sha256")
        blob_id = entry.get("remote_blob_id")
        if lfs_sha256:
            if lfs_sha256 != local["sha256"]:
                raise RuntimeError(f"materialized {filename} differs from remote LFS SHA-256")
            if entry.get("remote_lfs_size") != local["size"]:
                raise RuntimeError(f"materialized {filename} differs from remote LFS size")
            pointer = (
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{lfs_sha256}\nsize {local['size']}\n"
            ).encode("utf-8")
            if blob_id != git_blob_sha1(pointer):
                raise RuntimeError(f"remote LFS pointer Git blob differs for {filename}")
        elif blob_id:
            if blob_id != local["git_blob_sha1"]:
                raise RuntimeError(f"materialized {filename} differs from remote Git blob")
        else:
            raise RuntimeError(f"remote tree has no blob or LFS identity for {filename}")


def validate_output_shape_size(
    output_dir: Path,
    filename: str,
    spec: dict[str, Any],
    expected_hash: str,
) -> tuple[list[int], str, int]:
    expected_formats = {
        "pcm.f32": ("float32", 4),
        "input_features.f32": ("float32", 4),
        "is_longer.u8": ("uint8", 1),
        "input_ids.i64": ("int64", 8),
        "attention_mask.u8": ("uint8", 1),
    }
    if filename not in expected_formats:
        raise RuntimeError(f"unknown source-only output: {filename}")
    if not isinstance(spec, dict) or set(spec) != {"dtype", "endianness", "itemsize", "shape", "byte_length", "sha256"}:
        raise RuntimeError(f"source-only output metadata is incomplete: {filename}")
    expected_dtype, expected_itemsize = expected_formats[filename]
    if spec["dtype"] != expected_dtype or spec["endianness"] != "little" or spec["itemsize"] != expected_itemsize:
        raise RuntimeError(f"source-only output format drifted: {filename}")
    shape = spec["shape"]
    if not isinstance(shape, list) or not shape or any(not isinstance(axis, int) or axis <= 0 for axis in shape):
        raise RuntimeError(f"source-only output shape is malformed: {filename}")
    expected_bytes = math.prod(shape) * expected_itemsize
    if spec["byte_length"] != expected_bytes:
        raise RuntimeError(f"source-only output shape/byte length mismatch: {filename}")
    output = output_dir / filename
    if output.is_symlink() or not output.is_file() or output.stat().st_size != expected_bytes or sha256_file(output) != expected_hash or spec["sha256"] != expected_hash:
        raise RuntimeError(f"source-only output identity mismatch: {filename}")
    return shape, expected_dtype, expected_itemsize


def validate_input_ids_attention_mask_shapes(outputs: dict[str, Any]) -> None:
    if outputs["input_ids.i64"]["shape"] != outputs["attention_mask.u8"]["shape"]:
        raise RuntimeError("source-only input_ids and attention_mask shapes differ")


def validate_source_only_reference(
    path: Path, *, metadata_dir: Path, wheel_binding: dict[str, Any]
) -> dict[str, Any]:
    evidence = read_json(path, "source-only reference")
    if evidence.get("schema") != "vokra-clap-htsat-fused-source-only-reference-v1":
        raise RuntimeError("source-only reference schema drifted")
    if evidence.get("status") != SOURCE_REFERENCE_STATUS:
        raise RuntimeError("source-only reference did not complete")
    if evidence.get("repository") != MODEL_REPOSITORY or evidence.get("revision") != MODEL_REVISION:
        raise RuntimeError("source-only reference model identity drifted")
    if evidence.get("processor_class") != "ClapProcessor":
        raise RuntimeError("source-only reference processor drifted")
    if evidence.get("tokenizer_class") != "RobertaTokenizer":
        raise RuntimeError("source-only reference tokenizer drifted")
    if evidence.get("transformers_version") != TRANSFORMERS_VERSION:
        raise RuntimeError("source-only reference Transformers version drifted")
    tokenizer_source = evidence.get("tokenizer_source")
    if not isinstance(tokenizer_source, str) or not tokenizer_source.endswith(
        TRANSFORMERS_SOURCE_MEMBERS["roberta_tokenizer"]
    ):
        raise RuntimeError("source-only reference tokenizer source path drifted")
    source_facts = evidence.get("transformers_sources")
    if not isinstance(source_facts, dict) or set(source_facts) != {"feature_extractor", "processor", "tokenizer"}:
        raise RuntimeError("source-only reference source facts are missing")
    archive_members = wheel_binding.get("archive_members", {})
    for key, label, class_name in (
        ("feature_extractor", "feature_extraction_clap", "ClapFeatureExtractor"),
        ("processor", "processing_clap", "ClapProcessor"),
        ("tokenizer", "roberta_tokenizer", "RobertaTokenizer"),
    ):
        fact = source_facts.get(key)
        member = TRANSFORMERS_SOURCE_MEMBERS[label]
        if not isinstance(fact, dict) or fact.get("class") != class_name:
            raise RuntimeError(f"source-only {key} source fact is malformed")
        if not isinstance(fact.get("source"), str) or not fact["source"].endswith(member):
            raise RuntimeError(f"source-only {key} source path drifted")
        if fact.get("source_sha256") != archive_members.get(label, {}).get("sha256"):
            raise RuntimeError(f"source-only {key} source hash differs from wheel")
    if evidence.get("fixed_text") != "a calm room with a distant piano":
        raise RuntimeError("source-only reference fixed text drifted")
    if evidence.get("weights") != "NOT_ACQUIRED":
        raise RuntimeError("source-only reference overclaims weight status")
    if evidence.get("model_load") != "NOT_PERFORMED" or evidence.get("model_forward") != "NOT_PERFORMED":
        raise RuntimeError("source-only reference overclaims model execution")
    if evidence.get("execution") != "SOURCE_ONLY_REFERENCE_EXECUTED":
        raise RuntimeError("source-only reference execution status drifted")
    files = evidence.get("files_sha256")
    outputs = evidence.get("outputs")
    expected_files = {
        "pcm.f32",
        "input_features.f32",
        "is_longer.u8",
        "input_ids.i64",
        "attention_mask.u8",
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise RuntimeError("source-only reference output set drifted")
    if not isinstance(outputs, dict) or set(outputs) != expected_files:
        raise RuntimeError("source-only reference output metadata drifted")
    import numpy as np

    tokenizer_files = evidence.get("tokenizer_files")
    if not isinstance(tokenizer_files, dict) or set(tokenizer_files) != set(TOKENIZER_FILES):
        raise RuntimeError("source-only tokenizer file set drifted")
    for filename, expected_hash in tokenizer_files.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != SOURCE_SHA256_HEX_LENGTH:
            raise RuntimeError(f"source-only tokenizer hash is malformed: {filename}")
        tokenizer_file = metadata_dir / filename
        if not tokenizer_file.is_file() or tokenizer_file.is_symlink() or sha256_file(tokenizer_file) != expected_hash:
            raise RuntimeError(f"source-only tokenizer file is not staged: {filename}")
    for filename, expected_hash in files.items():
        if not isinstance(expected_hash, str) or len(expected_hash) != SOURCE_SHA256_HEX_LENGTH:
            raise RuntimeError(f"source-only output hash is malformed: {filename}")
        spec = outputs[filename]
        shape, _, expected_itemsize = validate_output_shape_size(
            path.parent, filename, spec, expected_hash
        )
        numpy_dtype = {
            "pcm.f32": "<f4",
            "input_features.f32": "<f4",
            "is_longer.u8": "u1",
            "input_ids.i64": "<i8",
            "attention_mask.u8": "u1",
        }[filename]
        expected_bytes = math.prod(shape) * expected_itemsize
        output = path.parent / filename
        values = np.frombuffer(output.read_bytes(), dtype=numpy_dtype)
        if tuple(values.shape) != (expected_bytes // expected_itemsize,):
            raise RuntimeError(f"source-only output byte payload is malformed: {filename}")
        values = values.reshape(shape)
        if filename in {"pcm.f32", "input_features.f32"} and not np.isfinite(values).all():
            raise RuntimeError(f"source-only floating payload is non-finite: {filename}")
        if filename == "pcm.f32" and shape != [480_000]:
            raise RuntimeError("source-only PCM shape is not [480000]")
        if filename == "input_features.f32" and (shape[0] != 1 or values.size == 0):
            raise RuntimeError("source-only features are not nonempty batch=1")
        if filename == "is_longer.u8" and (values.size == 0 or not np.isin(values, [0, 1]).all()):
            raise RuntimeError("source-only is_longer payload is not binary")
        if filename == "input_ids.i64" and (shape[0] != 1 or values.size == 0 or not np.all((values >= 0) & (values < 50_265))):
            raise RuntimeError("source-only input_ids payload is invalid")
        if filename == "attention_mask.u8" and (values.size == 0 or not np.isin(values, [0, 1]).all()):
            raise RuntimeError("source-only attention mask is not binary")
    validate_input_ids_attention_mask_shapes(outputs)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "status": evidence["status"],
        "repository": evidence["repository"],
        "revision": evidence["revision"],
        "fixed_text": evidence["fixed_text"],
        "tokenizer_class": evidence["tokenizer_class"],
        "tokenizer_source": evidence["tokenizer_source"],
        "tokenizer_files": tokenizer_files,
        "transformers_version": evidence["transformers_version"],
        "transformers_sources": source_facts,
        "files_sha256": files,
        "weights": evidence["weights"],
        "model_load": evidence["model_load"],
        "model_forward": evidence["model_forward"],
    }


def validate_expected_manifest(
    path: Path,
    *,
    config_path: Path,
    validate_manifest: Any,
    wheel_binding: dict[str, Any],
) -> dict[str, Any]:
    evidence = read_json(path, "source-derived expected manifest")
    if evidence.get("schema") != "vokra-clap-htsat-fused-source-derived-expected-manifest-v1":
        raise RuntimeError("source-derived expected manifest schema drifted")
    if evidence.get("status") != "SOURCE_DERIVED_EXPECTED_MANIFEST":
        raise RuntimeError("source-derived expected manifest is blocked or incomplete")
    if evidence.get("repository") != MODEL_REPOSITORY or evidence.get("revision") != MODEL_REVISION:
        raise RuntimeError("source-derived expected manifest model identity drifted")
    if evidence.get("manifest_kind") != "SOURCE_DERIVED_EXPECTED_MANIFEST":
        raise RuntimeError("source-derived manifest kind is not distinct from observed state")
    if evidence.get("weights") != "NOT_ACQUIRED" or evidence.get("model_load") != "NOT_PERFORMED" or evidence.get("model_forward") != "NOT_PERFORMED":
        raise RuntimeError("source-derived expected manifest overclaims model execution")
    if evidence.get("execution") != "META_CONSTRUCTION_ONLY":
        raise RuntimeError("source-derived expected manifest execution status drifted")
    if evidence.get("transformers_version") != TRANSFORMERS_VERSION:
        raise RuntimeError("source-derived expected manifest Transformers version drifted")
    sources = evidence.get("transformers_sources")
    if not isinstance(sources, dict) or set(sources) != {"config", "model"}:
        raise RuntimeError("source-derived expected manifest source facts are missing")
    archive_members = wheel_binding.get("archive_members", {})
    for key, label, class_name in (
        ("config", "configuration_clap", "ClapConfig"),
        ("model", "modeling_clap", "ClapModel"),
    ):
        fact = sources.get(key)
        member = TRANSFORMERS_SOURCE_MEMBERS[label]
        if not isinstance(fact, dict) or fact.get("class") != class_name:
            raise RuntimeError(f"source-derived {key} source fact is malformed")
        if not isinstance(fact.get("source"), str) or not fact["source"].endswith(member):
            raise RuntimeError(f"source-derived {key} source path drifted")
        if fact.get("source_sha256") != archive_members.get(label, {}).get("sha256"):
            raise RuntimeError(f"source-derived {key} source hash differs from wheel")
    expected_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if evidence.get("config_sha256") != expected_config_hash:
        raise RuntimeError("source-derived expected manifest config identity differs")
    manifest = evidence.get("tensor_manifest")
    roles = evidence.get("state_dict_roles")
    if not isinstance(manifest, dict) or not isinstance(roles, dict):
        raise RuntimeError("source-derived expected manifest payload is missing")
    validate_manifest(manifest, roles)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "status": evidence["status"],
        "manifest_kind": evidence["manifest_kind"],
        "tensor_count": len(manifest),
        "state_dict_roles": roles,
        "transformers_version": evidence["transformers_version"],
        "transformers_sources": sources,
    }


def audit(
    *,
    project_path: Path,
    lock_path: Path,
    config_path: Path,
    preprocessor_path: Path,
    remote_identity_path: Path,
    transformers_wheel_path: Path,
    source_reference_path: Path,
    expected_manifest_path: Path,
    dependency_inventory_path: Path | None = None,
) -> dict[str, Any]:
    if config_path.parent != preprocessor_path.parent:
        raise RuntimeError("config and preprocessor must come from one metadata snapshot")
    require_no_weights(config_path.parent)
    config_payload = read_json(config_path, "config.json")
    preprocessor_payload = read_json(preprocessor_path, "preprocessor_config.json")
    remote_identity = read_json(remote_identity_path, "remote identity")
    validate_remote_identity(
        remote_identity,
        config_path=config_path,
        preprocessor_path=preprocessor_path,
        metadata_dir=config_path.parent,
    )
    (
        preprocessor_contract,
        validate_preprocessor,
        validate_model_config,
        validate_serializer,
        validate_tensor_manifest,
    ) = load_reference_contract()

    from transformers import ClapConfig, ClapFeatureExtractor, ClapProcessor, RobertaTokenizer
    from transformers.processing_utils import ProcessorMixin

    config = ClapConfig.from_dict(config_payload)
    config_contract = validate_model_config(config.to_dict())
    raw_preprocessing = validate_preprocessor(preprocessor_payload)
    raw_processor_class = raw_preprocessing["processor_class"]
    if raw_processor_class != "ClapProcessor":
        raise RuntimeError(f"raw CLAP processor_class drifted: {raw_processor_class!r}")
    extractor = ClapFeatureExtractor(**preprocessor_payload)
    preprocessing_serialized = validate_serializer(extractor.to_dict())

    dependencies = audit_dependencies(project_path, lock_path)
    wheel_binding = validate_transformers_wheel(
        transformers_wheel_path, dependencies["transformers_wheel_artifact"]
    )
    source_reference = validate_source_only_reference(
        source_reference_path,
        metadata_dir=config_path.parent,
        wheel_binding=wheel_binding,
    )
    expected_manifest = validate_expected_manifest(
        expected_manifest_path,
        config_path=config_path,
        validate_manifest=validate_tensor_manifest,
        wheel_binding=wheel_binding,
    )
    dependency_inventory: dict[str, Any] | None = None
    if dependency_inventory_path is not None:
        dependency_inventory = validate_dependency_inventory(dependency_inventory_path)
    observed_transformers = importlib.metadata.version("transformers")
    if observed_transformers != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"installed Transformers version drifted: {observed_transformers!r}"
        )
    processor_source = source_fact(ClapProcessor)
    if processor_source["class"] != raw_processor_class:
        raise RuntimeError("raw processor_class is not bound to official ClapProcessor source")
    tokenizer = RobertaTokenizer
    source_contract = build_source_contract(
        ClapFeatureExtractor,
        ClapProcessor,
        ProcessorMixin,
        tokenizer,
        wheel_binding,
        source_reference,
    )
    return {
        "schema": SCHEMA,
        "status": "PASS_MODEL_FREE",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "publication": "NO_UPLOAD",
        "owner_approval": "PENDING_OWNER_APPROVAL",
        "metadata": {
            "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
            "preprocessor": {
                "path": preprocessor_path.name,
                "sha256": sha256_file(preprocessor_path),
            },
        },
        "config_contract": config_contract,
        "preprocessing_contract": {
            "raw_release_metadata": raw_preprocessing,
            "serializer_round_trip": preprocessing_serialized,
            "processor_class_binding": {
                "raw_processor_class": raw_processor_class,
                "api_class": processor_source["class"],
                "source": processor_source,
            },
        },
        "source_contract": source_contract,
        "source_reference": source_reference,
        "expected_manifest": expected_manifest,
        "api_facts": {
            "transformers_version": observed_transformers,
            "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
            "config": source_fact(ClapConfig),
            "feature_extractor": source_fact(ClapFeatureExtractor),
            "processor": processor_source,
            "tokenizer": source_fact(tokenizer),
        },
        "remote_identity": remote_identity,
        "dependency_license_contract": {
            "status": "PENDING",
            "license_evidence_status": dependencies["license_evidence_status"],
            "dependency_audit_status": dependencies["dependency_audit_status"],
            "facts": dependencies,
            "inventory": dependency_inventory,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def self_test() -> None:
    assert SCHEMA.endswith("-v1")
    assert len(REVISION) == 40 and all(char in "0123456789abcdef" for char in REVISION)
    assert TRANSFORMERS_VERSION == "5.10.4"
    assert len(TRANSFORMERS_WHEEL_SHA256) == 64
    assert WEIGHT_SUFFIXES == {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
    archive_hashes = {
        "feature_extraction_clap": "a" * 64,
        "processing_clap": "c" * 64,
        "processing_utils": "e" * 64,
        "modeling_clap": "h" * 64,
        "configuration_clap": "i" * 64,
        "roberta_tokenizer": "g" * 64,
    }
    archive_members = {
        label: {"path": member, "size": 1, "sha256": archive_hashes[label]}
        for label, member in TRANSFORMERS_SOURCE_MEMBERS.items()
    }
    tree_rows = [
        {"path": value["path"], "size": value["size"], "sha256": value["sha256"]}
        for value in archive_members.values()
    ]
    synthetic_source_contract = {
        "schema": SOURCE_CONTRACT_SCHEMA,
        "status": SOURCE_STATUS_AUTHENTICATED,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "wheel_binding": {
            "status": WHEEL_BINDING_STATUS,
            "wheel": {"sha256": TRANSFORMERS_WHEEL_SHA256},
            "archive_members": archive_members,
            "installed_members": json.loads(json.dumps(archive_members)),
            "python_source_tree": {
                "member_count": len(tree_rows),
                "canonical_tree_sha256": canonical_tree_digest(tree_rows),
                "archive_rows": tree_rows,
                "installed_rows": json.loads(json.dumps(tree_rows)),
            },
        },
        "source_reference": {
            "status": SOURCE_REFERENCE_STATUS,
            "transformers_sources": {
                "feature_extractor": {
                    "class": "ClapFeatureExtractor",
                    "source": "/wheel/transformers/models/clap/feature_extraction_clap.py",
                    "source_sha256": archive_hashes["feature_extraction_clap"],
                },
                "processor": {
                    "class": "ClapProcessor",
                    "source": "/wheel/transformers/models/clap/processing_clap.py",
                    "source_sha256": archive_hashes["processing_clap"],
                },
                "tokenizer": {
                    "class": "RobertaTokenizer",
                    "source": "/wheel/transformers/models/roberta/tokenization_roberta.py",
                    "source_sha256": archive_hashes["roberta_tokenizer"],
                },
            },
            "weights": "NOT_ACQUIRED",
            "model_load": "NOT_PERFORMED",
            "model_forward": "NOT_PERFORMED",
        },
        "weights": "NOT_ACQUIRED",
        "model_load": "NOT_PERFORMED",
        "model_forward": "NOT_PERFORMED",
        "audio": {
            "class": {
                "class": "ClapFeatureExtractor",
                "source": "/wheel/transformers/models/clap/feature_extraction_clap.py",
                "source_sha256": archive_hashes["feature_extraction_clap"],
                "signature": "()",
            },
            "methods": {
                name: {"signature": "()", "source_sha256": "b" * 64}
                for name in SOURCE_METHODS["audio"]
            },
            "semantic_surface": "official_audio_feature_extractor_entrypoints",
        },
        "text": {
            "class": {
                "class": "ClapProcessor",
                "source": "/wheel/transformers/models/clap/processing_clap.py",
                "source_sha256": archive_hashes["processing_clap"],
                "signature": "()",
            },
            "methods": {
                name: {"signature": "()", "source_sha256": "d" * 64}
                for name in SOURCE_METHODS["text"]
            },
            "base_class": {
                "class": "ProcessorMixin",
                "source": "/wheel/transformers/processing_utils.py",
                "source_sha256": archive_hashes["processing_utils"],
                "signature": "()",
            },
            "call": {
                "owner": "ProcessorMixin",
                "source": {
                    "class": "ProcessorMixin",
                    "source": "/wheel/transformers/processing_utils.py",
                    "source_sha256": archive_hashes["processing_utils"],
                    "signature": "()",
                },
                "method": {"signature": "()", "source_sha256": "f" * 64},
            },
            "semantic_surface": "official_text_processor_binding_entrypoints",
        },
        "tokenizer": {
            "class": {
                "class": "RobertaTokenizer",
                "source": "/wheel/transformers/models/roberta/tokenization_roberta.py",
                "source_sha256": archive_hashes["roberta_tokenizer"],
                "signature": "()",
            },
            "archive_member": TRANSFORMERS_SOURCE_MEMBERS["roberta_tokenizer"],
            "semantic_surface": "official_roberta_tokenizer_entrypoint",
        },
        "execution": "SOURCE_ONLY_REFERENCE_EXECUTED",
    }
    validate_source_contract(synthetic_source_contract)
    tampered_source_contract = json.loads(json.dumps(synthetic_source_contract))
    tampered_source_contract["audio"]["methods"]["__call__"]["source_sha256"] = "not-a-hash"
    try:
        validate_source_contract(tampered_source_contract)
    except RuntimeError as exc:
        assert "source hash" in str(exc)
    else:
        raise AssertionError("tampered source contract was accepted")
    for field, value in (
        ("model_revision", "0" * 40),
        ("status", SOURCE_STATUS_PENDING),
        ("transformers_version", "5.10.3"),
        ("execution", "SOURCE_HASH_ONLY_NO_PREPROCESSING_EXECUTION"),
    ):
        tampered_identity = json.loads(json.dumps(synthetic_source_contract))
        tampered_identity[field] = value
        try:
            validate_source_contract(tampered_identity)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"tampered source identity/status was accepted: {field}")
    tampered_member = json.loads(json.dumps(synthetic_source_contract))
    tampered_member["wheel_binding"]["archive_members"]["processing_clap"]["path"] = "replaced.py"
    try:
        validate_source_contract(tampered_member)
    except RuntimeError as exc:
        assert "archive member" in str(exc)
    else:
        raise AssertionError("replaced wheel source member was accepted")
    deleted_tree_member = json.loads(json.dumps(synthetic_source_contract))
    deleted_tree_member["wheel_binding"]["python_source_tree"]["archive_rows"].pop()
    try:
        validate_source_contract(deleted_tree_member)
    except RuntimeError as exc:
        assert "tree" in str(exc)
    else:
        raise AssertionError("deleted wheel source member was accepted")
    tampered_reference = json.loads(json.dumps(synthetic_source_contract))
    tampered_reference["source_reference"]["transformers_sources"]["processor"]["source_sha256"] = "0" * 64
    try:
        validate_source_contract(tampered_reference)
    except RuntimeError as exc:
        assert "source reference" in str(exc)
    else:
        raise AssertionError("tampered source-reference hash was accepted")
    missing_processor_call = json.loads(json.dumps(synthetic_source_contract))
    del missing_processor_call["text"]["call"]
    try:
        validate_source_contract(missing_processor_call)
    except RuntimeError as exc:
        assert "call" in str(exc)
    else:
        raise AssertionError("missing processor call source was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-clap-wheel-") as temporary:
        wheel = Path(temporary) / "transformers-5.10.4-py3-none-any.whl"
        wheel.write_bytes(b"tampered wheel")
        try:
            validate_transformers_wheel(
                wheel,
                {"url": TRANSFORMERS_WHEEL_URL, "hash": "sha256:" + "0" * 64, "size": TRANSFORMERS_WHEEL_SIZE},
            )
        except RuntimeError as exc:
            assert "fixed wheel identity" in str(exc)
        else:
            raise AssertionError("tampered fixed wheel identity was accepted")
    record_bytes = b"source.py,sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b"source").digest()).rstrip(b"=") + b",6\npackage.dist-info/RECORD,,\n"
    records = parse_record(record_bytes.decode("ascii"))
    assert validate_record_identity(records, "source.py", b"source", label="self-test")["size"] == 6
    try:
        validate_record_identity(records, "source.py", b"changed", label="self-test")
    except RuntimeError as exc:
        assert "RECORD" in str(exc)
    else:
        raise AssertionError("tampered RECORD identity was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-clap-payload-") as temporary:
        payload_dir = Path(temporary)
        payload = payload_dir / "is_longer.u8"
        payload.write_bytes(b"\x01")
        payload_hash = sha256_file(payload)
        valid_spec = {
            "dtype": "uint8",
            "endianness": "little",
            "itemsize": 1,
            "shape": [1],
            "byte_length": 1,
            "sha256": payload_hash,
        }
        assert validate_output_shape_size(payload_dir, payload.name, valid_spec, payload_hash)[0] == [1]
        zero_spec = dict(valid_spec)
        zero_spec["shape"] = [0]
        zero_spec["byte_length"] = 0
        try:
            validate_output_shape_size(payload_dir, payload.name, zero_spec, payload_hash)
        except RuntimeError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("zero/shape payload was accepted")
        mismatch_spec = dict(valid_spec)
        mismatch_spec["shape"] = [2]
        mismatch_spec["byte_length"] = 2
        try:
            validate_output_shape_size(payload_dir, payload.name, mismatch_spec, payload_hash)
        except RuntimeError as exc:
            assert "identity" in str(exc)
        else:
            raise AssertionError("mismatched payload size was accepted")
    valid_token_shapes = {
        "input_ids.i64": {"shape": [1, 4]},
        "attention_mask.u8": {"shape": [1, 4]},
    }
    validate_input_ids_attention_mask_shapes(valid_token_shapes)
    valid_token_shapes["attention_mask.u8"]["shape"] = [1, 3]
    try:
        validate_input_ids_attention_mask_shapes(valid_token_shapes)
    except RuntimeError as exc:
        assert "shapes differ" in str(exc)
    else:
        raise AssertionError("input_ids/attention_mask shape mismatch was accepted")
    preprocessor_contract, validate_preprocessor, _, validate_serializer, validate_tensor_manifest = load_reference_contract()
    raw = validate_preprocessor(dict(preprocessor_contract))
    assert raw["processor_class"] == "ClapProcessor"
    serialized = {
        key: value
        for key, value in raw.items()
        if key != "processor_class"
    }
    assert validate_serializer(serialized) == serialized
    try:
        tampered_serializer = dict(serialized)
        tampered_serializer["sampling_rate"] = 16_000
        validate_serializer(tampered_serializer)
    except RuntimeError as exc:
        assert "sampling_rate" in str(exc)
    else:
        raise AssertionError("serializer preprocessing drift was accepted")
    expected_roles = {
        "audio_tower": ["audio_model.x"],
        "text_tower": ["text_model.x"],
        "audio_projection": ["audio_projection.x"],
        "text_projection": ["text_projection.x"],
        "contrastive_scalar": ["logit_scale_a", "logit_scale_t"],
    }
    expected_tensor_manifest = {
        name: {
            "role": role,
            "shape": [] if role == "contrastive_scalar" else [1],
            "dtype": "torch.float32",
        }
        for role, names in expected_roles.items()
        for name in names
    }
    with tempfile.TemporaryDirectory(prefix="vokra-clap-expected-") as temporary:
        root = Path(temporary)
        config = root / "config.json"
        expected = root / "expected.json"
        config.write_text("{}\n", encoding="utf-8")
        expected_payload = {
            "schema": "vokra-clap-htsat-fused-source-derived-expected-manifest-v1",
            "status": "SOURCE_DERIVED_EXPECTED_MANIFEST",
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "transformers_sources": {
                "config": {"class": "ClapConfig", "source": "/wheel/transformers/models/clap/configuration_clap.py", "source_sha256": archive_hashes["configuration_clap"]},
                "model": {"class": "ClapModel", "source": "/wheel/transformers/models/clap/modeling_clap.py", "source_sha256": archive_hashes["modeling_clap"]},
            },
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weights": "NOT_ACQUIRED",
            "model_load": "NOT_PERFORMED",
            "model_forward": "NOT_PERFORMED",
            "execution": "META_CONSTRUCTION_ONLY",
            "manifest_kind": "SOURCE_DERIVED_EXPECTED_MANIFEST",
            "state_dict_roles": expected_roles,
            "tensor_manifest": expected_tensor_manifest,
        }
        expected.write_text(json.dumps(expected_payload) + "\n", encoding="utf-8")
        assert validate_expected_manifest(expected, config_path=config, validate_manifest=validate_tensor_manifest, wheel_binding=synthetic_source_contract["wheel_binding"])["status"] == "SOURCE_DERIVED_EXPECTED_MANIFEST"
        expected_payload["transformers_sources"]["model"]["source_sha256"] = "0" * 64
        expected.write_text(json.dumps(expected_payload) + "\n", encoding="utf-8")
        try:
            validate_expected_manifest(expected, config_path=config, validate_manifest=validate_tensor_manifest, wheel_binding=synthetic_source_contract["wheel_binding"])
        except RuntimeError as exc:
            assert "source hash" in str(exc)
        else:
            raise AssertionError("tampered expected model source was accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-clap-audit-") as temporary:
        root = Path(temporary)
        inventory = root / "dependency-inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "schema": "vokra-clap-htsat-fused-dependency-license-inventory-v1",
                    "status": "PENDING_OWNER_REVIEW",
                    "dependency_audit_status": EXPECTED_DEPENDENCY_AUDIT_STATUS,
                    "findings": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        assert validate_dependency_inventory(inventory)["dependency_audit_status"] == EXPECTED_DEPENDENCY_AUDIT_STATUS
        inventory.write_text(
            inventory.read_text(encoding="utf-8").replace(
                EXPECTED_DEPENDENCY_AUDIT_STATUS, "PENDING_OWNER_REVIEW"
            ),
            encoding="utf-8",
        )
        try:
            validate_dependency_inventory(inventory)
        except RuntimeError as exc:
            assert EXPECTED_DEPENDENCY_AUDIT_STATUS in str(exc)
        else:
            raise AssertionError("arbitrary dependency audit status was accepted")
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"role": "audio", "role": "text"}\n', encoding="utf-8")
        try:
            read_json(duplicate, "duplicate")
        except RuntimeError as exc:
            assert "duplicate JSON key" in str(exc)
        else:
            raise AssertionError("duplicate JSON keys were accepted")

        weights = root / "metadata"
        weights.mkdir()
        (weights / "config.json").write_text("{}\n", encoding="utf-8")
        (weights / "forbidden.safetensors").write_bytes(b"not a checkpoint")
        try:
            require_no_weights(weights)
        except RuntimeError as exc:
            assert "forbidden.safetensors" in str(exc)
        else:
            raise AssertionError("weight contamination was accepted")

        output = root / "evidence.json"
        write_atomic_no_replace(output, '{"status":"first"}\n')
        try:
            write_atomic_no_replace(output, '{"status":"second"}\n')
        except RuntimeError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("output replacement was accepted")
        assert output.read_text(encoding="utf-8") == '{"status":"first"}\n'

        config = root / "config.json"
        preprocessor = root / "preprocessor_config.json"
        config.write_bytes(b'{"config":true}\n')
        preprocessor.write_bytes(b'{"preprocessor":true}\n')
        remote_files = {}
        for filename, path in (("config.json", config), ("preprocessor_config.json", preprocessor)):
            local = local_file_identity(path)
            remote_files[filename] = {
                "remote_size": local["size"],
                "remote_blob_id": local["git_blob_sha1"],
                "remote_lfs_sha256": None,
                "remote_lfs_size": None,
                "local_size": local["size"],
                "local_sha256": local["sha256"],
                "local_git_blob_sha1": local["git_blob_sha1"],
            }
        remote = {
            "repository": REPOSITORY,
            "requested_revision": REVISION,
            "resolved_revision": REVISION,
            "metadata_api": "HfApi.model_info/list_repo_files",
            "tree_api": "HfApi.list_repo_tree(expand=True)",
            "card_data_source": "info.card_data",
            "card_data_license_status": "MISSING",
            "repo_license_file_status": "MISSING",
            "weights": "NOT_ACQUIRED",
            "remote_files": remote_files,
        }
        validate_remote_identity(remote, config_path=config, preprocessor_path=preprocessor)
        remote["remote_files"]["config.json"]["remote_size"] += 1
        try:
            validate_remote_identity(remote, config_path=config, preprocessor_path=preprocessor)
        except RuntimeError as exc:
            assert "size differs" in str(exc)
        else:
            raise AssertionError("remote/local metadata identity mismatch was accepted")

        class DictInfo:
            card_data = {"license": "apache-2.0"}

        class CardData:
            def to_dict(self) -> dict[str, str]:
                return {"license": "apache-2.0"}

        assert normalize_card_data(DictInfo()) == ({"license": "apache-2.0"}, "info.card_data")
        assert normalize_card_data(type("Info", (), {"card_data": CardData()})()) == (
            {"license": "apache-2.0"},
            "info.card_data.to_dict",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--wheel-binding-only", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preprocessor", type=Path)
    parser.add_argument("--remote-identity", type=Path)
    parser.add_argument("--transformers-wheel", type=Path)
    parser.add_argument("--source-reference", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--dependency-inventory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.wheel_binding_only or any(value is not None for value in (args.project, args.lock, args.config, args.preprocessor, args.remote_identity, args.transformers_wheel, args.source_reference, args.expected_manifest, args.dependency_inventory, args.output)):
            parser.error("--self-test accepts no audit paths")
        self_test()
        print("clap model-free audit self-test: OK")
        return 0
    if args.wheel_binding_only:
        if args.project is None or args.lock is None or args.transformers_wheel is None or any(value is not None for value in (args.config, args.preprocessor, args.remote_identity, args.source_reference, args.expected_manifest, args.dependency_inventory, args.output)):
            parser.error("--wheel-binding-only requires --project, --lock, and --transformers-wheel only")
        dependencies = audit_dependencies(args.project, args.lock)
        binding = validate_transformers_wheel(args.transformers_wheel, dependencies["transformers_wheel_artifact"])
        tree = binding["python_source_tree"]
        print(f"CLAP_WHEEL_BINDING {binding['status']}: count={tree['member_count']} digest={tree['canonical_tree_sha256']}")
        return 0
    required = (args.project, args.lock, args.config, args.preprocessor, args.remote_identity, args.transformers_wheel, args.source_reference, args.expected_manifest, args.output)
    if any(value is None for value in required):
        parser.error("normal runs require project/lock/config/preprocessor/remote-identity/wheel/source-reference/expected-manifest/output")
    assert args.output is not None
    assert args.remote_identity is not None
    assert args.transformers_wheel is not None
    assert args.source_reference is not None
    assert args.expected_manifest is not None
    evidence = audit(
        project_path=args.project,
        lock_path=args.lock,
        config_path=args.config,
        preprocessor_path=args.preprocessor,
        remote_identity_path=args.remote_identity,
        transformers_wheel_path=args.transformers_wheel,
        source_reference_path=args.source_reference,
        expected_manifest_path=args.expected_manifest,
        dependency_inventory_path=args.dependency_inventory,
    )
    try:
        write_atomic_no_replace(
            args.output, json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"CLAP_MODEL_FREE_AUDIT PASS_MODEL_FREE: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
