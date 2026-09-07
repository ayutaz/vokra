#!/usr/bin/env -S uv run --script
"""Model-free official Qwen3-TTS API probe.

This probe is deliberately separate from the real-weight API smoke.  It
imports the pinned QwenLM source and exercises only configuration,
tokenizer/processor construction, and wrapper API introspection.  It rejects
checkpoint files and never calls ``Qwen3TTSModel.from_pretrained``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
import types
from pathlib import Path
from typing import Any

SCHEMA = "vokra-qwen3-tts-model-free-api-smoke-v1"
SOURCE_REPOSITORY = "QwenLM/Qwen3-TTS"
SOURCE_URL = "https://github.com/QwenLM/Qwen3-TTS.git"
SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SOURCE_PACKAGE_VERSION = "0.1.1"
SOURCE_FILES = (
    "qwen_tts/__init__.py",
    "qwen_tts/core/models/configuration_qwen3_tts.py",
    "qwen_tts/core/models/processing_qwen3_tts.py",
    "qwen_tts/inference/qwen3_tts_model.py",
)
VARIANTS: dict[str, dict[str, Any]] = {
    "0.6b-base": {
        "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "config_bytes": 4494,
        "config_sha256": "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
        "tts_model_type": "base",
    },
    "0.6b-customvoice": {
        "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": "85e237c12c027371202489a0ec509ded67b5e4b5",
        "config_bytes": 4908,
        "config_sha256": "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
        "tts_model_type": "custom_voice",
    },
    "1.7b-base": {
        "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "config_bytes": 4494,
        "config_sha256": "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
        "tts_model_type": "base",
    },
    "1.7b-customvoice": {
        "repository": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "config_bytes": 4908,
        "config_sha256": "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
        "tts_model_type": "custom_voice",
    },
}
COMMON_ASSETS = {
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "tokenizer_config.json": (7344, "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670"),
    "generation_config.json": (245, "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa"),
}
PROJECT_SHA256 = "7ef84e96d4fb486aa4b6c922fbbe06cb42f8ab56108958106287ccd613ac100e"
LOCK_SHA256 = "b5fd403808a15759c5b10331e4da759ad230847baa833e75abba36d53a3cfdd2"
REQUIRED_DEPENDENCIES = {
    "accelerate==1.12.0", "einops==0.8.2", "librosa==1.0.0", "numpy==2.5.2",
    "soundfile==0.14.0", "torch==2.7.1", "torchaudio==2.11.0",
    "transformers==5.10.4",
}
FORBIDDEN_PACKAGES = {"gradio", "onnxruntime", "protobuf", "setuptools", "sox"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOX_SENTINEL_FILE = "/__vokra_import_only_sox_sentinel__.py"


class ProbeError(RuntimeError):
    """A fail-closed model-free probe failure."""


class SoxSentinelAccessError(ProbeError):
    """The forbidden optional sox module was accessed during import."""


class ApiProbeFailure(ProbeError):
    """An official API import/introspection failure with sentinel evidence."""

    def __init__(self, message: str, *, sentinel_installed: bool, accesses: int, metadata_reads: int) -> None:
        super().__init__(message)
        self.sentinel_installed = sentinel_installed
        self.accesses = accesses
        self.metadata_reads = metadata_reads


class _SoxSentinel(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("sox")
        self.accesses = 0
        self.metadata_reads = 0

    def __getattribute__(self, name: str) -> Any:
        if name == "__file__":
            reads = object.__getattribute__(self, "metadata_reads")
            object.__setattr__(self, "metadata_reads", reads + 1)
            return SOX_SENTINEL_FILE
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        self.accesses += 1
        raise SoxSentinelAccessError(f"forbidden sox access: {name}")


@contextmanager
def install_sox_sentinel() -> Any:
    """Provide import-only ``sox`` and restore ``sys.modules`` exactly."""

    module_name = "sox"
    if module_name in sys.modules:
        raise ProbeError("real or pre-existing sox module is installed")
    if importlib.util.find_spec(module_name) is not None:
        raise ProbeError("real sox package is installed")
    sentinel = _SoxSentinel()
    sys.modules[module_name] = sentinel
    try:
        yield sentinel
    finally:
        if sys.modules.get(module_name) is sentinel:
            del sys.modules[module_name]
        elif module_name in sys.modules:
            raise ProbeError("sox sentinel was overwritten during API probe")


def pending_approval() -> dict[str, Any]:
    return {
        "source_license": "PENDING_OWNER_APPROVAL",
        "model_license": "PENDING_OWNER_APPROVAL",
        "operator": "PENDING_OWNER_APPROVAL",
        "signer": None,
        "scope_sha256": None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def require_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ProbeError(f"{label} is missing, symlinked, or empty: {path}")


def require_clean_head(root: Path, expected_head: str) -> None:
    if not HEX40.fullmatch(expected_head):
        raise ProbeError("expected HEAD must be exactly 40 lowercase hex")
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual != expected_head:
        raise ProbeError(f"Vokra checkout HEAD drifted: {actual} != {expected_head}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise ProbeError("Vokra checkout is dirty")


def verify_project(project: Path) -> dict[str, Any]:
    pyproject = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    require_regular(pyproject, "Qwen3-TTS project")
    require_regular(lock_path, "Qwen3-TTS uv.lock")
    project_hash = sha256_file(pyproject)
    if project_hash != PROJECT_SHA256:
        raise ProbeError(f"project SHA-256 drifted: {project_hash}")
    lock_hash = sha256_file(lock_path)
    if lock_hash != LOCK_SHA256:
        raise ProbeError(f"uv.lock SHA-256 drifted: {lock_hash}")
    project_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = project_data.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or set(dependencies) != REQUIRED_DEPENDENCIES:
        raise ProbeError("Qwen3-TTS dependency closure is not exact")
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "manifest", "package"}:
        raise ProbeError("uv.lock top-level schema drifted")
    if lock["manifest"] != {"overrides": [{"name": "setuptools", "marker": "python_full_version < '0'"}]}:
        raise ProbeError("uv.lock manifest override drifted")
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ProbeError("uv.lock package table is malformed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ProbeError("uv.lock package row is malformed")
        name, version, source = package.get("name"), package.get("version"), package.get("source")
        if not isinstance(name, str) or not isinstance(version, str) or not isinstance(source, dict):
            raise ProbeError("uv.lock package identity is malformed")
        if source == {"virtual": "."}:
            continue
        if set(source) != {"registry"} or source.get("registry") not in {"https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"}:
            raise ProbeError(f"uv.lock package source is not approved: {name}")
        key = (name, version, source["registry"])
        if key in seen:
            raise ProbeError(f"duplicate locked package: {key}")
        seen.add(key)
        if name in FORBIDDEN_PACKAGES:
            raise ProbeError(f"forbidden package is locked: {name}")
        rows.append({"name": name, "version": version, "source": source})
    return {
        "project_sha256": project_hash,
        "lock_sha256": lock_hash,
        "packages": sorted(rows, key=lambda row: (row["name"], row["version"], row["source"]["registry"])),
    }


def verify_source(source: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir() or not (source / ".git").is_dir() or (source / ".git").is_symlink():
        raise ProbeError("official source checkout is missing or symlinked")
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual != SOURCE_REVISION:
        raise ProbeError(f"official source revision drifted: {actual}")
    if subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout:
        raise ProbeError("official source checkout is dirty")
    metadata = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata.get("project", {}).get("version")
    if version != SOURCE_PACKAGE_VERSION:
        raise ProbeError(f"official source package version drifted: {version!r}")
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = source / relative
        require_regular(path, f"official source {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"repository": SOURCE_REPOSITORY, "url": SOURCE_URL, "revision": SOURCE_REVISION,
            "package_version": SOURCE_PACKAGE_VERSION, "files": files}


def verify_metadata(snapshot: Path, variant: str) -> dict[str, Any]:
    expected = {"config.json": (VARIANTS[variant]["config_bytes"], VARIANTS[variant]["config_sha256"])}
    expected.update(COMMON_ASSETS)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ProbeError(f"metadata snapshot is missing or symlinked: {snapshot}")
    entries = list(snapshot.iterdir())
    cache = snapshot / ".cache"
    if cache in entries and (cache.is_symlink() or not cache.is_dir()):
        raise ProbeError("metadata snapshot .cache is invalid")
    visible = [entry for entry in entries if entry.name != ".cache"]
    nested_weights = [
        path for path in snapshot.rglob("*")
        if path.name.endswith(".safetensors") or path.name.startswith("model-")
    ]
    if nested_weights:
        raise ProbeError(f"checkpoint file reached model-free snapshot: {nested_weights[0]}")
    if sorted(entry.name for entry in visible) != sorted(expected):
        raise ProbeError("metadata snapshot contains an extra or missing file")
    records: dict[str, dict[str, Any]] = {}
    for name, (size, expected_hash) in expected.items():
        path = snapshot / name
        require_regular(path, f"metadata {name}")
        actual = sha256_file(path)
        if path.stat().st_size != size or actual != expected_hash:
            raise ProbeError(f"metadata identity drifted for {name}")
        records[name] = {"bytes": size, "sha256": actual}
    config = strict_json((snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_tts" or config.get("tts_model_type") != VARIANTS[variant]["tts_model_type"]:
        raise ProbeError(f"{variant} config contract is not exact")
    if any(path.name.endswith(".safetensors") or path.name.startswith("model-") for path in visible):
        raise ProbeError("checkpoint file reached model-free snapshot")
    return {"repository": VARIANTS[variant]["repository"], "revision": VARIANTS[variant]["revision"], "files": records,
            "model_type": config["model_type"], "tts_model_type": config["tts_model_type"],
            "checkpoint_files": "NONE_PRESENT"}


def api_probe(source: Path, snapshot: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source))
    sentinel: _SoxSentinel | None = None
    try:
        with install_sox_sentinel() as sentinel:
            import qwen_tts
            from qwen_tts import Qwen3TTSModel
            from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
            config = Qwen3TTSConfig.from_pretrained(str(snapshot), local_files_only=True)
            processor = Qwen3TTSProcessor.from_pretrained(str(snapshot), local_files_only=True)
            if processor is None or config.model_type != "qwen3_tts":
                raise ProbeError("official processor/config construction returned an invalid object")
            package_root = Path(qwen_tts.__file__).resolve().parents[1]
            if package_root != source.resolve():
                raise ProbeError(f"qwen_tts imported from unexpected path: {package_root}")
            versions = {
                name: importlib.metadata.version(name)
                for name in ("accelerate", "einops", "librosa", "numpy", "soundfile", "torch", "torchaudio", "transformers")
            }
            if versions["transformers"] != "5.10.4":
                raise ProbeError(f"Transformers runtime drifted: {versions['transformers']}")
            if sentinel.accesses != 0:
                raise ProbeError(f"forbidden sox sentinel was accessed {sentinel.accesses} time(s)")
            return {
                "imports": [
                    "qwen_tts.Qwen3TTSModel",
                    "qwen_tts.core.models.Qwen3TTSConfig",
                    "qwen_tts.core.models.Qwen3TTSProcessor",
                ],
                "package_versions": versions,
                "config_class": f"{Qwen3TTSConfig.__module__}.{Qwen3TTSConfig.__name__}",
                "processor_class": f"{Qwen3TTSProcessor.__module__}.{Qwen3TTSProcessor.__name__}",
                "wrapper_class": f"{Qwen3TTSModel.__module__}.{Qwen3TTSModel.__name__}",
                "config_from_pretrained": "CALLED_LOCAL_ONLY",
                "processor_from_pretrained": "CALLED_LOCAL_ONLY",
                "wrapper_from_pretrained": "NOT_CALLED",
                "wrapper_signature": str(inspect.signature(Qwen3TTSModel.from_pretrained)),
                "generate_voice_clone_signature": str(inspect.signature(Qwen3TTSModel.generate_voice_clone)),
                "checkpoint_load": "NOT_PERFORMED",
                "sox_sentinel": {
                    "installed": True,
                    "allowed_metadata": ["__file__"],
                    "metadata_reads": sentinel.metadata_reads,
                    "accesses": 0,
                },
            }
    except Exception as exc:  # noqa: BLE001 - API incompatibility is evidence, not a traceback
        raise ApiProbeFailure(
            str(exc),
            sentinel_installed=sentinel is not None,
            accesses=sentinel.accesses if sentinel is not None else 0,
            metadata_reads=sentinel.metadata_reads if sentinel is not None else 0,
        ) from None
    finally:
        if sys.path and sys.path[0] == str(source):
            sys.path.pop(0)


def write_output(path: Path, evidence: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise ProbeError("output must be an absolute path")
    if path.exists() or path.is_symlink():
        raise ProbeError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(evidence, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> int:
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise ProbeError("VOKRA_PUBLISH_ON_VAST=1 is required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ProbeError("model-free API smoke requires VAST Linux x86_64")
    root = Path(args.vokra_root)
    project = Path(args.project)
    source = Path(args.source_dir)
    snapshot_root = Path(args.snapshot_root)
    output = Path(args.output)
    require_clean_head(root, args.expected_head)
    project_record = verify_project(project)
    source_record = verify_source(source)
    variant_names = list(VARIANTS) if args.variant == "all" else [args.variant]
    metadata = {variant: verify_metadata(snapshot_root / variant, variant) for variant in variant_names}
    api: dict[str, Any] = {}
    try:
        for variant in variant_names:
            api[variant] = api_probe(source, snapshot_root / variant)
    except Exception as exc:  # noqa: BLE001 - API incompatibility is emitted atomically
        sentinel_installed = bool(getattr(exc, "sentinel_installed", False))
        sentinel_accesses = int(getattr(exc, "accesses", 0))
        sentinel_metadata_reads = int(getattr(exc, "metadata_reads", 0))
        blocked = {
            "schema": SCHEMA,
            "status": "BLOCKED_INCOMPATIBLE_API",
            "publication": "NO_UPLOAD",
            "expected_head": args.expected_head,
            "source": source_record,
            "variants": metadata,
            "project": project_record,
            "api": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "sox_sentinel": {
                    "installed": sentinel_installed,
                    "allowed_metadata": ["__file__"],
                    "metadata_reads": sentinel_metadata_reads,
                    "accesses": sentinel_accesses,
                },
            },
            "checkpoint_load": "NOT_PERFORMED",
            "approval": pending_approval(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
        }
        write_output(output, blocked)
        print(f"BLOCKED_INCOMPATIBLE_API: {exc}", file=sys.stderr)
        return 2
    sentinel_records = [record["sox_sentinel"] for record in api.values()]
    evidence = {
        "schema": SCHEMA,
        "status": "PASS_MODEL_FREE",
        "publication": "NO_UPLOAD",
        "expected_head": args.expected_head,
        "source": source_record,
        "variants": metadata,
        "project": project_record,
        "api": api,
        "sox_sentinel": {
            "installed": all(record["installed"] for record in sentinel_records),
            "allowed_metadata": ["__file__"],
            "metadata_reads": sum(record["metadata_reads"] for record in sentinel_records),
            "accesses": sum(record["accesses"] for record in sentinel_records),
        },
        "checkpoint_load": "NOT_PERFORMED",
        "approval": pending_approval(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    write_output(output, evidence)
    print("QWEN3_TTS_MODEL_FREE_API_SMOKE PASS_MODEL_FREE (no checkpoint load, no upload)")
    return 0


def self_test() -> int:
    try:
        assert all(HEX40.fullmatch(identity["revision"]) for identity in VARIANTS.values())
        assert all(HEX64.fullmatch(identity["config_sha256"]) for identity in VARIANTS.values())
        assert HEX64.fullmatch(PROJECT_SHA256) and HEX64.fullmatch(LOCK_SHA256)
        with tempfile.TemporaryDirectory(prefix="qwen3-tts-model-free-self-test-") as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"x":1,"x":2}', encoding="utf-8")
            try:
                strict_json(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
            else:
                raise AssertionError("duplicate JSON key accepted")
            if importlib.util.find_spec("sox") is not None:
                raise AssertionError("real sox package is installed")
            with install_sox_sentinel() as sentinel:
                assert sys.modules["sox"] is sentinel
                assert sentinel.__file__ == SOX_SENTINEL_FILE
                assert sentinel.metadata_reads == 1
                try:
                    sentinel.Transformer
                except SoxSentinelAccessError:
                    pass
                else:
                    raise AssertionError("sox sentinel allowed attribute access")
                assert sentinel.accesses == 1
            assert "sox" not in sys.modules
            prior = types.ModuleType("sox")
            sys.modules["sox"] = prior
            try:
                try:
                    with install_sox_sentinel():
                        raise AssertionError("pre-existing sox module was clobbered")
                except ProbeError:
                    pass
                assert sys.modules["sox"] is prior
            finally:
                del sys.modules["sox"]
            blocked_path = Path(directory) / "blocked.json"
            write_output(blocked_path, {
                "schema": SCHEMA,
                "status": "BLOCKED_INCOMPATIBLE_API",
                "publication": "NO_UPLOAD",
                "checkpoint_load": "NOT_PERFORMED",
                "api": {
                    "sox_sentinel": {
                        "installed": True,
                        "allowed_metadata": ["__file__"],
                        "metadata_reads": 1,
                        "accesses": 1,
                    }
                },
                "approval": pending_approval(),
            })
            blocked = strict_json(blocked_path.read_text(encoding="utf-8"))
            assert blocked["status"] == "BLOCKED_INCOMPATIBLE_API"
            assert blocked["publication"] == "NO_UPLOAD"
            assert blocked["checkpoint_load"] == "NOT_PERFORMED"
            assert blocked["approval"]["source_license"] == "PENDING_OWNER_APPROVAL"
            assert blocked["api"]["sox_sentinel"]["allowed_metadata"] == ["__file__"]
            assert blocked["api"]["sox_sentinel"]["accesses"] == 1
            assert not list(Path(directory).glob(".blocked.json.*.tmp"))
        probe_source = inspect.getsource(api_probe)
        assert "Qwen3TTSModel.from_pretrained(" not in probe_source
        assert '"wrapper_from_pretrained": "NOT_CALLED"' in probe_source
        print("qwen3_tts model-free API smoke self-test PASS (no model, no network)")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"qwen3_tts model-free API smoke self-test FAIL: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vokra-root", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.expected_head, args.variant, args.output)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    required = (args.vokra_root, args.project, args.source_dir, args.snapshot_root, args.expected_head, args.variant, args.output)
    if any(value is None for value in required):
        parser.error("all production paths are required")
    try:
        return run(args)
    except (ProbeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"qwen3_tts model-free API smoke: BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
