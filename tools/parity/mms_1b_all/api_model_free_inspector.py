#!/usr/bin/env python3
"""Record the official MMS adapter API without touching model checkpoints.

The normal path imports only the two official Transformers API classes and
inspects their signatures/source.  It never calls ``from_pretrained``,
``load_adapter``, or a forward method.  Evidence is generated only from a
clean checkout at the exact expected commit and is written atomically without
replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Callable

try:
    from . import license_gate
except ImportError:
    import license_gate


SCHEMA = "vokra-mms-1b-all-api-model-free-evidence-v2"
REPOSITORY = "facebook/mms-1b-all"
REVISION = "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPORT_KEYS = {
    "schema", "status", "publication", "upstream", "project_sha256", "lock_sha256",
    "expected_head", "head", "clean", "generator_sha256", "runtime", "api", "execution", "owner_review",
}


class InspectorError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_lexical_path(path: Path, label: str) -> None:
    raw = str(path).replace("\\", "/")
    if not path.is_absolute() or "//" in raw or any(part in {".", ".."} for part in raw.split("/")):
        raise InspectorError(f"{label} has unsafe lexical components: {path}")


def safe_existing_directory(path: Path, label: str) -> Path:
    reject_lexical_path(path, label)
    if path.is_symlink() or not path.is_dir():
        raise InspectorError(f"{label} is not a regular directory: {path}")
    current = path
    while True:
        if current.is_symlink():
            raise InspectorError(f"{label} has symlinked ancestry: {path}")
        if current.parent == current:
            break
        current = current.parent
    return path


def safe_output_parent(path: Path) -> Path:
    reject_lexical_path(path, "output")
    if path.name in {"", ".", ".."}:
        raise InspectorError(f"output filename is malformed: {path}")
    return safe_existing_directory(path.parent, "output parent")


def regular_file(path: Path) -> bool:
    absolute = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_file():
        return False
    return all(not parent.is_symlink() for parent in (absolute, *absolute.parents))


def write_atomic_no_replace(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise InspectorError(f"output already exists: {path}")
    safe_output_parent(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise InspectorError(f"output appeared during inspection: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_head(value: Any, label: str = "head") -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise InspectorError(f"{label} must be exactly 40 lowercase hexadecimal characters")
    return value


def validate_git_result(expected_head: str, head: str, status: str) -> dict[str, Any]:
    expected = validate_head(expected_head, "expected_head")
    actual = validate_head(head.strip(), "head")
    if actual != expected:
        raise InspectorError(f"checkout HEAD {actual} differs from expected {expected}")
    if status:
        raise InspectorError("checkout is dirty; model-free evidence requires a clean tree")
    return {"expected_head": expected, "head": actual, "clean": True}


def git_state(repo_root: Path, expected_head: str, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    safe_existing_directory(repo_root, "repository root")

    def invoke(*args: str) -> Any:
        result = runner(["git", *args], cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise InspectorError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    top = Path(invoke("rev-parse", "--show-toplevel").strip()).resolve()
    if top != repo_root.resolve():
        raise InspectorError(f"git root mismatch: {top} != {repo_root.resolve()}")
    head = invoke("rev-parse", "HEAD")
    status = invoke("status", "--porcelain", "--untracked-files=all")
    return validate_git_result(expected_head, head, status)


def load_project_lock(project_path: Path, lock_path: Path) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    if not regular_file(project_path) or not regular_file(lock_path):
        raise InspectorError("project or lock is missing, symlinked, or not a regular file")
    try:
        project_bytes = project_path.read_bytes()
        lock_bytes = lock_path.read_bytes()
        project = tomllib.loads(project_bytes.decode("utf-8"))
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        license_gate.project_schema(project)
        license_gate.lock_rows(lock)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as error:
        raise InspectorError(f"project/lock validation failed: {error}") from error
    return project_bytes, lock_bytes, project, lock


def source_record(function: Any, package_root: Path, *, load_adapter: bool = False) -> dict[str, str]:
    source_name = inspect.getsourcefile(function)
    if not isinstance(source_name, str):
        raise InspectorError(f"official API source path is unavailable: {function}")
    source_path = Path(source_name)
    if not regular_file(source_path):
        raise InspectorError(f"official API source is not a regular symlink-free file: {source_path}")
    try:
        relative = source_path.resolve().relative_to(package_root.resolve())
    except ValueError as error:
        raise InspectorError(f"official API source escaped Transformers package: {source_path}") from error
    logical = PurePosixPath("transformers", *relative.parts).as_posix()
    if logical.startswith("/") or any(part in {"", ".", ".."} for part in PurePosixPath(logical).parts):
        raise InspectorError(f"official API source path is unsafe: {logical}")
    record = {"path": logical, "sha256": file_digest(source_path)}
    if load_adapter:
        record["load_adapter_source_sha256"] = hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()
    validate_source_record(record, adapter=load_adapter)
    return record


def validate_source_record(record: Any, *, adapter: bool = False) -> None:
    expected = {"path", "sha256", "load_adapter_source_sha256"} if adapter else {"path", "sha256"}
    if not isinstance(record, dict) or set(record) != expected or not isinstance(record.get("path"), str):
        raise InspectorError("API source record schema is malformed")
    path = record["path"]
    if not path.startswith("transformers/") or path.startswith("/") or "//" in path or "\\" in path or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts):
        raise InspectorError("API source path is malformed")
    if not HEX64.fullmatch(str(record.get("sha256"))):
        raise InspectorError("API source hash is malformed")
    if adapter and not HEX64.fullmatch(str(record.get("load_adapter_source_sha256"))):
        raise InspectorError("API adapter source hash is malformed")


def validate_report_schema(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != REPORT_KEYS:
        raise InspectorError("API evidence schema is malformed")
    if value.get("schema") != SCHEMA or value.get("status") != "MODEL_FREE_API_VALIDATED" or value.get("publication") != "NO_UPLOAD" or value.get("clean") is not True:
        raise InspectorError("API evidence status is malformed")


def generate(project_path: Path, lock_path: Path, output: Path, repo_root: Path, expected_head: str) -> None:
    project_bytes, lock_bytes, _, _ = load_project_lock(project_path, lock_path)
    git_identity = git_state(repo_root, expected_head)
    # These are the only runtime API imports in this generator.  No checkpoint
    # API is called below; signatures/source inspection is model-free.
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    transformers_root = Path(inspect.getfile(AutoProcessor)).resolve().parents[2]
    auto_source = source_record(AutoProcessor.from_pretrained, transformers_root)
    adapter_source = source_record(Wav2Vec2ForCTC.load_adapter, transformers_root, load_adapter=True)
    report = {
        "schema": SCHEMA,
        "status": "MODEL_FREE_API_VALIDATED",
        "publication": "NO_UPLOAD",
        "upstream": {"repository": REPOSITORY, "revision": REVISION},
        "project_sha256": hashlib.sha256(project_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        **git_identity,
        "generator_sha256": file_digest(Path(__file__).resolve()),
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "transformers": __import__("transformers").__version__,
            "torch": importlib.metadata.version("torch"),
            "weights_acquired": False,
            "model_class_imported": True,
            "model_instantiated": False,
            "model_weights_loaded": False,
            "model_executed": False,
        },
        "api": {
            "auto_processor_from_pretrained_signature": str(inspect.signature(AutoProcessor.from_pretrained)),
            "auto_processor_source": auto_source,
            "wav2vec2_for_ctc_from_pretrained_signature": str(inspect.signature(Wav2Vec2ForCTC.from_pretrained)),
            "wav2vec2_for_ctc_load_adapter_signature": str(inspect.signature(Wav2Vec2ForCTC.load_adapter)),
            "wav2vec2_for_ctc_load_adapter_source": adapter_source,
            "target_lang_adapter_surface": "Wav2Vec2ForCTC.load_adapter(target_lang=language)",
        },
        "execution": {
            "api_import_only": True,
            "checkpoint_download": False,
            "checkpoint_load": False,
            "forward": False,
            "parity": "BLOCKED_PENDING_AUTHENTICATED_MANIFEST",
        },
        "owner_review": "PENDING_OWNER_APPROVAL",
    }
    validate_report_schema(report)
    write_atomic_no_replace(output, canonical(report) + "\n")


def self_test() -> None:
    valid = "a" * 40
    assert validate_git_result(valid, valid, "") == {"expected_head": valid, "head": valid, "clean": True}
    for bad in ("", "a" * 39, "A" * 40, "not-a-head"):
        try:
            validate_head(bad, "expected_head")
        except InspectorError:
            pass
        else:
            raise SystemExit(f"self-test accepted malformed head: {bad!r}")
    try:
        validate_git_result(valid, valid, " M dirty.py\n")
    except InspectorError:
        pass
    else:
        raise SystemExit("self-test accepted dirty checkout")
    try:
        validate_git_result(valid, "b" * 40, "")
    except InspectorError:
        pass
    else:
        raise SystemExit("self-test accepted HEAD mismatch")
    with tempfile.TemporaryDirectory(prefix="vokra-mms-api-git-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        class Result:
            def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
                self.stdout, self.stderr, self.returncode = stdout, stderr, returncode
        responses = iter((Result(f"{root}\n"), Result(f"{valid}\n"), Result(" M dirty.py\n")))
        def fake_runner(*_args: Any, **_kwargs: Any) -> Result:
            return next(responses)
        try:
            git_state(root, valid, runner=fake_runner)
        except InspectorError:
            pass
        else:
            raise SystemExit("self-test accepted mocked dirty checkout")
    validate_source_record({"path": "transformers/models/auto/processing_auto.py", "sha256": "a" * 64})
    for bad in (
        {"path": "../processing_auto.py", "sha256": "a" * 64},
        {"path": "transformers//processing_auto.py", "sha256": "a" * 64},
        {"path": "transformers/models/auto/processing_auto.py", "sha256": "bad"},
        {"path": "transformers/models/auto/processing_auto.py", "sha256": "a" * 64, "extra": True},
    ):
        try:
            validate_source_record(bad)
        except InspectorError:
            pass
        else:
            raise SystemExit("self-test accepted source path/hash/schema tamper")
    valid_report = {key: None for key in REPORT_KEYS}
    valid_report.update({"schema": SCHEMA, "status": "MODEL_FREE_API_VALIDATED", "publication": "NO_UPLOAD", "clean": True})
    validate_report_schema(valid_report)
    valid_report["unexpected"] = True
    try:
        validate_report_schema(valid_report)
    except InspectorError:
        pass
    else:
        raise SystemExit("self-test accepted evidence schema tamper")
    with tempfile.TemporaryDirectory(prefix="vokra-mms-api-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        output = root / "evidence.json"
        write_atomic_no_replace(output, "first\n")
        try:
            write_atomic_no_replace(output, "second\n")
        except InspectorError:
            pass
        else:
            raise SystemExit("self-test replaced existing evidence")
        linked_parent = root / "linked-parent"
        linked_parent.symlink_to(root, target_is_directory=True)
        try:
            write_atomic_no_replace(linked_parent / "nested.json", "third\n")
        except InspectorError:
            pass
        else:
            raise SystemExit("self-test accepted symlinked output parent")
        try:
            write_atomic_no_replace(root / "dot" / ".." / "unsafe.json", "fourth\n")
        except InspectorError:
            pass
        else:
            raise SystemExit("self-test accepted lexical dot output path")
    print("mms model-free API inspector self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    values = (args.project, args.lock, args.output, args.repo_root, args.expected_head)
    if args.self_test:
        if any(value is not None for value in values):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in values):
        parser.error("normal runs require --project, --lock, --output, --repo-root, and --expected-head")
    try:
        generate(args.project, args.lock, args.output, args.repo_root, args.expected_head)
    except (InspectorError, OSError, ImportError, ValueError) as error:
        print(f"mms model-free API inspector: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"mms model-free API evidence written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
