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
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


REPOSITORY = "laion/clap-htsat-fused"
REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"
TRANSFORMERS_VERSION = "5.10.4"
TRANSFORMERS_WHEEL_SHA256 = (
    "8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e"
)
SCHEMA = "vokra-clap-htsat-fused-model-free-audit-v1"
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}


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


def load_reference_contract() -> tuple[dict[str, Any], Any, Any, Any]:
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
    )


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
    torch_source = packages["torch"].get("source", {})
    if torch_source.get("registry") != "https://download.pytorch.org/whl/cpu":
        raise RuntimeError(f"CLAP torch source is not the pinned CPU index: {torch_source}")
    values = project["tool"]["vokra"]["clap_reference"]
    if values["isolated_transformers_pin"] != TRANSFORMERS_VERSION:
        raise RuntimeError("pyproject Transformers pin drifted")
    if values["transformers_wheel_sha256"] != TRANSFORMERS_WHEEL_SHA256:
        raise RuntimeError("pyproject Transformers wheel hash drifted")
    return {
        "project_sha256": sha256_file(project_path),
        "lock_sha256": sha256_file(lock_path),
        "dependencies": sorted(actual_dependencies),
        "locked_versions": {
            name: packages[name]["version"] for name in ("numpy", "torch", "transformers")
        },
        "torch_index": torch_source["registry"],
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
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


def validate_remote_identity(
    remote_identity: dict[str, Any],
    *,
    config_path: Path,
    preprocessor_path: Path,
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
    for filename, path in (
        ("config.json", config_path),
        ("preprocessor_config.json", preprocessor_path),
    ):
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


def audit(
    *,
    project_path: Path,
    lock_path: Path,
    config_path: Path,
    preprocessor_path: Path,
    remote_identity_path: Path,
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
    )
    (
        preprocessor_contract,
        validate_preprocessor,
        validate_model_config,
        validate_serializer,
    ) = load_reference_contract()

    from transformers import ClapConfig, ClapFeatureExtractor, ClapProcessor

    config = ClapConfig.from_dict(config_payload)
    config_contract = validate_model_config(config.to_dict())
    raw_preprocessing = validate_preprocessor(preprocessor_payload)
    raw_processor_class = raw_preprocessing["processor_class"]
    if raw_processor_class != "ClapProcessor":
        raise RuntimeError(f"raw CLAP processor_class drifted: {raw_processor_class!r}")
    extractor = ClapFeatureExtractor(**preprocessor_payload)
    preprocessing_serialized = validate_serializer(extractor.to_dict())

    dependencies = audit_dependencies(project_path, lock_path)
    observed_transformers = importlib.metadata.version("transformers")
    if observed_transformers != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"installed Transformers version drifted: {observed_transformers!r}"
        )
    processor_source = source_fact(ClapProcessor)
    if processor_source["class"] != raw_processor_class:
        raise RuntimeError("raw processor_class is not bound to official ClapProcessor source")
    return {
        "schema": SCHEMA,
        "status": "PASS_MODEL_FREE",
        "repository": REPOSITORY,
        "revision": REVISION,
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
        "api_facts": {
            "transformers_version": observed_transformers,
            "config": source_fact(ClapConfig),
            "feature_extractor": source_fact(ClapFeatureExtractor),
            "processor": processor_source,
        },
        "remote_identity": remote_identity,
        "dependency_license_contract": {
            "status": "PENDING",
            "license_evidence_status": dependencies["license_evidence_status"],
            "dependency_audit_status": dependencies["dependency_audit_status"],
            "facts": dependencies,
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
    preprocessor_contract, validate_preprocessor, _, validate_serializer = load_reference_contract()
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
    with tempfile.TemporaryDirectory(prefix="vokra-clap-audit-") as temporary:
        root = Path(temporary)
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
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preprocessor", type=Path)
    parser.add_argument("--remote-identity", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.lock, args.config, args.preprocessor, args.remote_identity, args.output)):
            parser.error("--self-test accepts no audit paths")
        self_test()
        print("clap model-free audit self-test: OK")
        return 0
    required = (args.project, args.lock, args.config, args.preprocessor, args.remote_identity, args.output)
    if any(value is None for value in required):
        parser.error("normal runs require --project, --lock, --config, --preprocessor, --remote-identity, and --output")
    assert args.output is not None
    assert args.remote_identity is not None
    evidence = audit(
        project_path=args.project,
        lock_path=args.lock,
        config_path=args.config,
        preprocessor_path=args.preprocessor,
        remote_identity_path=args.remote_identity,
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
