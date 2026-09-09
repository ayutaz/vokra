#!/usr/bin/env python3
"""Audit the frozen MMS Python closure without importing model code.

The audit runs only in the dedicated VAST environment.  It reads the exact
``uv.lock`` rows and the installed distribution metadata/files, recording
artifact identities, publisher license metadata, license-file bytes, and
native payload bytes.  These facts are not legal approval: every package and
native payload remains pending owner review and the report is therefore
fail-closed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

try:
    from . import license_gate
except ImportError:  # direct uv run from the repository root
    import license_gate


SCHEMA = "vokra-mms-1b-all-dependency-audit-v1"
OWNER_REVIEW = "PENDING_OWNER_APPROVAL"
LICENSE_NAMES = {"license", "licence", "copying", "notice", "copyright"}
NATIVE_SUFFIXES = {".so", ".dylib", ".dll", ".pyd"}
ELF_MAGIC = b"\x7fELF"
MAX_LICENSE_BYTES = 2 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPORT_KEYS = {
    "schema", "status", "publication", "owner_review", "project_sha256", "lock_sha256",
    "expected_head", "head", "clean", "audit_script_sha256", "package_rows",
    "package_rows_sha256", "package_facts", "package_facts_sha256", "distribution_inventory", "environment", "blockers",
}


class AuditError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def reject_lexical_path(path: Path, label: str) -> None:
    raw = str(path).replace("\\", "/")
    if not path.is_absolute() or "//" in raw or any(part in {".", ".."} for part in raw.split("/")):
        raise AuditError(f"{label} has unsafe lexical components: {path}")


def safe_existing_directory(path: Path, label: str) -> Path:
    reject_lexical_path(path, label)
    if path.is_symlink() or not path.is_dir():
        raise AuditError(f"{label} is not a regular directory: {path}")
    current = path
    while True:
        if current.is_symlink():
            raise AuditError(f"{label} has symlinked ancestry: {path}")
        if current.parent == current:
            break
        current = current.parent
    return path


def safe_output_parent(path: Path) -> Path:
    reject_lexical_path(path, "output")
    if path.name in {"", ".", ".."}:
        raise AuditError(f"output filename is malformed: {path}")
    return safe_existing_directory(path.parent, "output parent")


def validate_head(value: Any, label: str = "head") -> str:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise AuditError(f"{label} must be exactly 40 lowercase hexadecimal characters")
    return value


def validate_git_result(expected_head: str, head: str, status: str) -> dict[str, Any]:
    expected = validate_head(expected_head, "expected_head")
    actual = validate_head(head.strip(), "head")
    if actual != expected:
        raise AuditError(f"checkout HEAD {actual} differs from expected {expected}")
    if status:
        raise AuditError("checkout is dirty; dependency evidence requires a clean tree")
    return {"expected_head": expected, "head": actual, "clean": True}


def git_state(repo_root: Path, expected_head: str, *, runner: Any = subprocess.run) -> dict[str, Any]:
    safe_existing_directory(repo_root, "repository root")

    def invoke(*args: str) -> str:
        result = runner(["git", *args], cwd=repo_root, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AuditError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    top = Path(invoke("rev-parse", "--show-toplevel").strip()).resolve()
    if top != repo_root.resolve():
        raise AuditError(f"git root mismatch: {top} != {repo_root.resolve()}")
    return validate_git_result(expected_head, invoke("rev-parse", "HEAD"), invoke("status", "--porcelain", "--untracked-files=all"))


def validate_report_schema(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != REPORT_KEYS:
        raise AuditError("dependency audit evidence schema is malformed")
    if value.get("schema") != SCHEMA or value.get("status") != "BLOCKED" or value.get("publication") != "NO_UPLOAD" or value.get("clean") is not True:
        raise AuditError("dependency audit evidence status is malformed")


def write_atomic_no_replace(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise AuditError(f"output already exists: {path}")
    safe_output_parent(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuditError(f"output appeared during audit: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_lock(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        lock_bytes = path.read_bytes()
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        rows = license_gate.lock_rows(lock)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, TypeError, ValueError) as error:
        raise AuditError(f"cannot load authenticated lock: {error}") from error
    return lock, rows


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def distribution_identity(name: str, version: str) -> str:
    if not isinstance(name, str) or not name.strip() or not isinstance(version, str) or not version.strip():
        raise AuditError("installed distribution has malformed name/version")
    normalized = normalize_name(name)
    if not normalized or "==" in version:
        raise AuditError("installed distribution has malformed normalized identity")
    return f"{normalized}=={version}"


def distribution_inventory(rows: list[dict[str, Any]], distributions: Any = None) -> dict[str, Any]:
    expected = sorted(distribution_identity(str(row["name"]), str(row["version"])) for row in rows if row["source"] != {"virtual": "."})
    installed: list[str] = []
    for dist in (metadata.distributions() if distributions is None else distributions):
        name = dist.metadata.get("Name") or getattr(dist, "name", None)
        version = dist.metadata.get("Version") or getattr(dist, "version", None)
        installed.append(distribution_identity(name, version))
    expected_counts = Counter(expected)
    installed_counts = Counter(installed)
    missing = sorted((expected_counts - installed_counts).elements())
    unexpected = sorted((installed_counts - expected_counts).elements())
    duplicates = sorted(name for name, count in installed_counts.items() if count > 1)
    return {
        "expected": expected,
        "installed": sorted(installed),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "exact": not missing and not unexpected and not duplicates,
    }


def safe_relative(value: str) -> PurePosixPath:
    raw_parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise AuditError(f"unsafe installed path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AuditError(f"unsafe installed path: {value!r}")
    return path


def is_license_name(value: str) -> bool:
    base = value.casefold().replace("_", "-")
    return base in LICENSE_NAMES or any(base.startswith(f"{name}.") for name in LICENSE_NAMES)


def is_native_name(value: str) -> bool:
    lower = value.casefold()
    return any(lower.endswith(suffix) for suffix in NATIVE_SUFFIXES) or ".so." in lower


def safe_dist_path(dist: Any, entry: Any, *, prefix: Path) -> Path:
    """Resolve one inventory candidate without trusting RECORD traversal."""
    raw = Path(dist.locate_file(entry))
    raw_absolute = Path(os.path.normpath(str(raw.absolute())))
    prefix_absolute = Path(os.path.normpath(str(prefix.absolute())))
    if prefix_absolute.is_symlink() or not prefix_absolute.is_dir():
        raise AuditError(f"Python prefix is not a regular directory: {prefix}")
    try:
        raw_absolute.relative_to(prefix_absolute)
    except ValueError as error:
        raise AuditError(f"inventory candidate escapes sys.prefix: {raw}") from error
    current = raw_absolute
    while current != prefix_absolute:
        if current.is_symlink():
            raise AuditError(f"inventory candidate has symlinked ancestry: {raw}")
        if current == current.parent:
            raise AuditError(f"inventory candidate has unsafe ancestry: {raw}")
        current = current.parent
    try:
        resolved = raw_absolute.resolve(strict=True)
        prefix_real = prefix_absolute.resolve(strict=True)
        resolved.relative_to(prefix_real)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise AuditError(f"inventory candidate resolves outside sys.prefix: {raw}") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise AuditError(f"inventory candidate is not a regular file: {raw}")
    return resolved


def native_file(path: Path, relative: PurePosixPath) -> bool:
    if path.suffix.casefold() in NATIVE_SUFFIXES:
        return True
    try:
        mode = path.stat().st_mode
        if not (mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
            return False
        with path.open("rb") as handle:
            return handle.read(4) == ELF_MAGIC
    except OSError as error:
        raise AuditError(f"cannot inspect native candidate {relative}: {error}") from error


def file_fact(path: Path, relative: PurePosixPath) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"installed payload is not a regular file: {relative}")
    size = path.stat().st_size
    if size < 0:
        raise AuditError(f"installed payload has invalid size: {relative}")
    return {"path": relative.as_posix(), "bytes": size, "sha256": file_digest(path)}


def archive_evidence(row: dict[str, Any], archive_dir: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for kind in ("sdist", "wheels"):
        values = row.get(kind, [])
        if kind == "sdist":
            values = [values] if values else []
        for artifact in values:
            if "size" in artifact:
                continue
            url = str(artifact["url"])
            filename = Path(unquote(urlsplit(url).path)).name
            candidate = archive_dir / filename
            if not candidate.is_file() or candidate.is_symlink():
                raise AuditError(f"missing archive evidence for lock artifact: {url}")
            actual_size = candidate.stat().st_size
            actual_hash = file_digest(candidate)
            if actual_hash != str(artifact["hash"]).removeprefix("sha256:"):
                raise AuditError(f"archive hash mismatch for lock artifact: {url}")
            evidence.append({
                "kind": kind,
                "url": url,
                "path": filename,
                "bytes": actual_size,
                "sha256": actual_hash,
                "lock_size": None,
            })
    return evidence


def distribution_fact(row: dict[str, Any], archive_dir: Path) -> dict[str, Any]:
    name = str(row["name"])
    expected_version = str(row["version"])
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError as error:
        raise AuditError(f"locked package is not installed: {name}=={expected_version}") from error
    if dist.version != expected_version:
        raise AuditError(f"installed version drift: {name} {dist.version!r} != {expected_version!r}")
    prefix = Path(sys.prefix)
    if not prefix.is_dir() or prefix.is_symlink():
        raise AuditError(f"Python prefix is not regular: {prefix}")
    license_files: list[dict[str, Any]] = []
    native_files: list[dict[str, Any]] = []
    external_record_paths: list[str] = []
    file_inventory_errors: list[str] = []
    files = list(dist.files or [])
    for entry in sorted(files, key=lambda value: str(value)):
        raw_entry = str(entry).replace("\\", "/")
        name_candidate = is_license_name(PurePosixPath(raw_entry).name) or is_native_name(PurePosixPath(raw_entry).name)
        has_traversal = raw_entry.startswith("/") or any(part in {"", ".", ".."} for part in raw_entry.split("/"))
        # RECORD commonly contains console scripts outside site-packages. They
        # are not package payload and are ignored only when they are not a
        # license/native candidate; candidate paths are always strict-checked.
        if has_traversal and not name_candidate:
            external_record_paths.append(raw_entry)
            continue
        try:
            relative = PurePosixPath(raw_entry)
            if relative.is_absolute() or not relative.parts:
                raise AuditError(f"unsafe inventory path: {raw_entry!r}")
            resolved = safe_dist_path(dist, entry, prefix=prefix)
            if is_license_name(relative.name):
                fact = file_fact(resolved, relative)
                if fact["bytes"] > MAX_LICENSE_BYTES:
                    fact["too_large"] = True
                license_files.append(fact)
            if is_native_name(relative.name) or native_file(resolved, relative):
                native_files.append(file_fact(resolved, relative))
        except AuditError as error:
            file_inventory_errors.append(str(error))
    publisher_license = dist.metadata.get("License") or ""
    classifiers = sorted(dist.metadata.get_all("Classifier") or [])
    recovered_archives = archive_evidence(row, archive_dir)
    base = {
        "name": name,
        "version": expected_version,
        "source": row["source"],
        "artifacts": {key: row[key] for key in ("sdist", "wheels") if key in row},
        "artifact_identity_sha256": digest({key: row[key] for key in ("sdist", "wheels") if key in row}),
        "archive_evidence": recovered_archives,
        "artifact_status": "LOCK_SIZE_RECOVERED" if recovered_archives else "LOCK_ARTIFACT_EXACT",
        "publisher_license": publisher_license,
        "license_classifiers": classifiers,
        "license_files": license_files,
        "license_files_sha256": digest(license_files),
        "native_files": native_files,
        "native_files_sha256": digest(native_files),
        "external_record_paths": external_record_paths,
        "external_record_paths_sha256": digest(external_record_paths),
        "file_inventory_errors": sorted(set(file_inventory_errors)),
        "file_inventory_status": "UNKNOWN" if file_inventory_errors else "EXACT",
        "license_review": OWNER_REVIEW,
        "native_bundled_review": OWNER_REVIEW,
    }
    if file_inventory_errors:
        base["license_review"] = OWNER_REVIEW
        base["native_bundled_review"] = OWNER_REVIEW
    base["fact_sha256"] = digest(base)
    return base


def audit(project: Path, lock_path: Path, output: Path, archive_dir: Path, repo_root: Path, expected_head: str) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise AuditError("MMS dependency audit requires Linux x86_64 VAST")
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        raise AuditError("VOKRA_PUBLISH_ON_VAST=1 is required")
    git_identity = git_state(repo_root, expected_head)
    try:
        project_bytes = project.read_bytes()
        project_value = tomllib.loads(project_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise AuditError(f"cannot load pyproject: {error}") from error
    try:
        license_gate.project_schema(project_value)
        lock, rows = load_lock(lock_path)
    except (TypeError, ValueError, AuditError) as error:
        raise AuditError(str(error)) from error
    package_rows = [row for row in rows if row["source"] != {"virtual": "."}]
    root_rows = [row for row in rows if row["source"] == {"virtual": "."}]
    if len(root_rows) != 1:
        raise AuditError("dedicated lock must contain exactly one virtual project")
    safe_existing_directory(archive_dir, "archive evidence directory")
    inventory = distribution_inventory(rows)
    if inventory["exact"] is not True:
        raise AuditError(f"installed distribution inventory is not exact: {inventory}")
    facts = [distribution_fact(row, archive_dir) for row in package_rows]
    failures = [
        "publisher license metadata and bundled/native payloads require owner review",
        "no model weights were acquired or imported; runtime/parity remain blocked",
        "one PyTorch CPU lock wheel omitted size; exact archive bytes are recovered but owner review remains required",
    ]
    report = {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "publication": "NO_UPLOAD",
        "owner_review": OWNER_REVIEW,
        "project_sha256": file_digest(project),
        "lock_sha256": file_digest(lock_path),
        **git_identity,
        "audit_script_sha256": file_digest(Path(__file__).resolve()),
        "distribution_inventory": inventory,
        "package_rows": rows,
        "package_rows_sha256": digest(rows),
        "package_facts": facts,
        "package_facts_sha256": digest(facts),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "weights_acquired": False,
            "model_imported": False,
            "model_executed": False,
            "upload": "NO_UPLOAD",
        },
        "blockers": failures,
    }
    validate_report_schema(report)
    write_atomic_no_replace(output, canonical(report) + "\n")


def self_test() -> None:
    # Keep generated fact construction fail-closed if a future edit repeats a
    # literal dictionary key (Python otherwise silently keeps the last value).
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)]
            if len(keys) != len(set(keys)):
                raise SystemExit("self-test found duplicate literal dictionary key")
    assert digest({"x": 1}) == digest({"x": 1})
    valid_head = "a" * 40
    assert validate_git_result(valid_head, valid_head, "") == {"expected_head": valid_head, "head": valid_head, "clean": True}
    for bad in ("", "a" * 39, "A" * 40, "not-a-head"):
        try:
            validate_head(bad, "expected_head")
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted malformed head: {bad!r}")
    try:
        validate_git_result(valid_head, valid_head, " M dirty.py\n")
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted dirty checkout")
    with tempfile.TemporaryDirectory(prefix="vokra-mms-audit-git-", dir="/private/tmp") as temporary:
        root = Path(temporary)
        class Result:
            def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
                self.stdout, self.stderr, self.returncode = stdout, stderr, returncode
        responses = iter((Result(f"{root}\n"), Result(f"{valid_head}\n"), Result(" M dirty.py\n")))
        def fake_runner(*_args: Any, **_kwargs: Any) -> Result:
            return next(responses)
        try:
            git_state(root, valid_head, runner=fake_runner)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted mocked dirty checkout")
    valid_report = {key: None for key in REPORT_KEYS}
    valid_report.update({"schema": SCHEMA, "status": "BLOCKED", "publication": "NO_UPLOAD", "clean": True})
    validate_report_schema(valid_report)
    valid_report["unexpected"] = True
    try:
        validate_report_schema(valid_report)
    except AuditError:
        pass
    else:
        raise SystemExit("self-test accepted audit evidence schema tamper")
    rows = [
        {"name": "Demo-Pkg", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}},
        {"name": "second_pkg", "version": "2.0+cpu", "source": {"registry": "https://pypi.org/simple"}},
        {"name": "root", "source": {"virtual": "."}},
    ]
    class Installed:
        def __init__(self, name: str, version: str):
            self.metadata = {"Name": name, "Version": version}
    exact = distribution_inventory(rows, [Installed("demo_pkg", "1.0"), Installed("second-pkg", "2.0+cpu")])
    assert exact == {"expected": ["demo-pkg==1.0", "second-pkg==2.0+cpu"], "installed": ["demo-pkg==1.0", "second-pkg==2.0+cpu"], "missing": [], "unexpected": [], "duplicates": [], "exact": True}
    mismatch = distribution_inventory(rows, [Installed("demo_pkg", "1.1"), Installed("second-pkg", "2.0+cpu")])
    assert mismatch["missing"] == ["demo-pkg==1.0"] and mismatch["unexpected"] == ["demo-pkg==1.1"] and mismatch["exact"] is False
    missing = distribution_inventory(rows, [Installed("demo_pkg", "1.0")])
    assert missing["missing"] == ["second-pkg==2.0+cpu"] and missing["exact"] is False
    unexpected = distribution_inventory(rows, [Installed("demo_pkg", "1.0"), Installed("second-pkg", "2.0+cpu"), Installed("extra", "9")])
    assert unexpected["unexpected"] == ["extra==9"] and unexpected["exact"] is False
    duplicate = distribution_inventory(rows, [Installed("demo_pkg", "1.0"), Installed("second-pkg", "2.0+cpu"), Installed("second_pkg", "2.0+cpu")])
    assert duplicate["duplicates"] == ["second-pkg==2.0+cpu"] and duplicate["exact"] is False
    assert safe_relative("dist-info/METADATA").as_posix() == "dist-info/METADATA"
    for bad in ("/absolute", "../escape", "a/../b", "./dot"):
        try:
            safe_relative(bad)
        except AuditError:
            pass
        else:
            raise SystemExit(f"self-test accepted unsafe path: {bad}")
    assert is_license_name("LICENSE") and is_license_name("license.txt")
    assert not is_license_name("not-license.txt")
    assert not native_file(Path(__file__), PurePosixPath(__file__).name)
    with tempfile.TemporaryDirectory(prefix="vokra-mms-audit-", dir="/private/tmp") as temporary:
        root = Path(temporary)

        class SyntheticDistribution:
            def __init__(self, location: Path):
                self.location = location

            def locate_file(self, entry: str) -> Path:
                return self.location / entry

        dist_root = root / "lib" / "python3.12" / "site-packages" / "demo.dist-info"
        license_path = root / "lib" / "share" / "licenses" / "demo" / "LICENSE"
        license_path.parent.mkdir(parents=True)
        license_path.write_bytes(b"MIT\n")
        synthetic = SyntheticDistribution(dist_root)
        legal = safe_dist_path(
            synthetic, "../../../share/licenses/demo/LICENSE", prefix=root
        )
        assert legal == license_path.resolve()
        # Non-target console scripts are ignored before path resolution.
        assert not is_license_name("tool") and not is_native_name("tool")
        outside = root.parent / "mms-audit-outside-module.so"
        outside.write_bytes(b"native")
        try:
            safe_dist_path(synthetic, "../../../../../../mms-audit-outside-module.so", prefix=root)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted a prefix-external native candidate")
        symlink_target = root / "real-native"
        symlink_target.mkdir()
        symlink = root / "linked-native"
        symlink.symlink_to(symlink_target, target_is_directory=True)
        (symlink_target / "module.so").write_bytes(b"native")
        try:
            safe_dist_path(synthetic, "../../../../linked-native/module.so", prefix=root)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted symlinked native ancestry")
        archive_dir = root / "archives"
        archive_dir.mkdir()
        archive = archive_dir / "demo.whl"
        archive.write_bytes(b"archive")
        row = {
            "name": "demo",
            "version": "1",
            "source": {"registry": "https://pypi.org/simple"},
            "wheels": [{
                "url": "https://files.pythonhosted.org/packages/demo.whl",
                "hash": "sha256:" + file_digest(archive),
                "upload-time": "2026-01-01T00:00:00Z",
            }],
        }
        assert archive_evidence(row, archive_dir)[0]["bytes"] == archive.stat().st_size
        archive.write_bytes(b"tampered")
        try:
            archive_evidence(row, archive_dir)
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted archive hash tamper")
        output = root / "evidence.json"
        write_atomic_no_replace(output, "first\n")
        try:
            write_atomic_no_replace(output, "second\n")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test replaced an existing output")
        linked_parent = root / "linked-parent"
        linked_parent.symlink_to(root, target_is_directory=True)
        try:
            write_atomic_no_replace(linked_parent / "nested.json", "third\n")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted symlinked output parent")
        try:
            write_atomic_no_replace(root / "dot" / ".." / "unsafe.json", "fourth\n")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted lexical dot output path")
        try:
            safe_existing_directory(linked_parent, "archive evidence directory")
        except AuditError:
            pass
        else:
            raise SystemExit("self-test accepted symlinked archive directory")
    sample = {"name": "demo", "source": {"registry": "https://pypi.org/simple"}, "artifacts": {}}
    assert HEX64.fullmatch(digest(sample))
    print("mms dependency audit self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.project, args.lock, args.output, args.archive_dir, args.repo_root, args.expected_head)):
            parser.error("--self-test accepts no other arguments")
        self_test()
        return 0
    if any(value is None for value in (args.project, args.lock, args.output, args.archive_dir, args.repo_root, args.expected_head)):
        parser.error("normal runs require --project, --lock, --output, --archive-dir, --repo-root, and --expected-head")
    try:
        audit(args.project, args.lock, args.output, args.archive_dir, args.repo_root, args.expected_head)
    except (AuditError, OSError, ValueError) as error:
        print(f"mms dependency audit: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"mms dependency audit: BLOCKED pending owner review; evidence={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
