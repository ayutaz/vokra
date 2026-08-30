#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""VAST-only collection of locked artifact license/native evidence.

This module deliberately performs network I/O only when invoked by the VAST
worker.  Local self-tests use small synthetic archives and never resolve,
download, or import a dependency.  The resulting evidence remains BLOCKED
until ``audit.py`` validates every row and an owner signs off independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit

from audit import NATIVE_PACKAGES, active_closure, lock_rows, repository_license, sha256, supported_wheel, validate_spdx_expression

CHUNK_SIZE = 1 << 20
MAX_LICENSE_BYTES = 4 << 20
MAX_METADATA_BYTES = 1 << 20
# Native entries are hashed in bounded chunks and never copied to a second
# file.  2 GiB accommodates large CPU Torch extensions while bounding a
# malformed archive before it can consume unbounded local resources.
MAX_NATIVE_BYTES = 2 << 30
PRIMARY_LICENSE_NAMES = ("license", "copying")
NATIVE_SUFFIXES = (".so", ".dylib", ".dll", ".pyd")
SPDX_ALIASES = {
    "apache software license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "mit license": "MIT",
    "mit": "MIT",
    "isc license": "ISC",
    "isc": "ISC",
    "python software foundation license": "PSF-2.0",
    "python-2.0": "Python-2.0",
}
CLASSIFIER_ALIASES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: CNRI Python License": "CNRI-Python",
    "License :: OSI Approved :: ISC License": "ISC",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _license_member(name: str) -> bool:
    if not _safe_member(name):
        return False
    base = Path(name).name.lower()
    return any(base == token or base.startswith(token + ".") or base.startswith(token + "-") for token in PRIMARY_LICENSE_NAMES)


def _license_rank(name: str, kind: str) -> tuple[int, int, int, str]:
    """Rank distribution-owned license locations before bundled components.

    Wheels place the project license at the archive root or in the project's
    ``*.dist-info/licenses`` directory.  Sdists have one package-root prefix,
    so that prefix is ignored for the same decision.  Everything below those
    locations is retained as bundled evidence but cannot become the primary
    license merely because its basename is ``LICENSE``.
    """
    parts = Path(name).parts
    if kind == "sdist" and len(parts) > 1:
        parts = parts[1:]
    lowered = tuple(part.lower() for part in parts)
    base = lowered[-1]
    basename_rank = 0 if base == "license" else 1 if base.startswith("license.") or base.startswith("license-") else 2
    for index, part in enumerate(lowered[:-1]):
        if part.endswith(".dist-info") and index + 1 < len(lowered) and lowered[index + 1] == "licenses":
            return (0, len(lowered) - index - 2, basename_rank, name.lower())
    if len(lowered) == 1:
        return (1, 0, basename_rank, name.lower())
    if lowered[-2].endswith(".dist-info"):
        return (2, len(lowered), basename_rank, name.lower())
    return (3, len(lowered), basename_rank, name.lower())


def _bounded_declaration(value: str) -> str:
    value = value.replace("\x00", "\\0").strip()
    return value if len(value) <= 256 else value[:253] + "..."


def _spdx_expression(value: str) -> str:
    try:
        return validate_spdx_expression(value)
    except ValueError as error:
        raise ValueError(f"License-Expression {_bounded_declaration(value)!r}: {error}") from error


def _metadata_license(metadata: bytes, label: str = "METADATA") -> str:
    expressions: list[str] = []
    legacy: list[str] = []
    classifiers: list[str] = []
    declarations: list[str] = []
    for line in metadata.decode("utf-8", errors="replace").splitlines():
        lowered = line.lower()
        if lowered.startswith("license-expression:"):
            value = line.split(":", 1)[1].strip()
            expressions.append(value)
            declarations.append(f"License-Expression={_bounded_declaration(value)}")
        elif lowered.startswith("license:"):
            value = line.split(":", 1)[1].strip()
            legacy.append(value)
            declarations.append(f"License={_bounded_declaration(value)}")
        elif lowered.startswith("classifier:"):
            value = line.split(":", 1)[1].strip()
            if value.startswith("License ::"):
                classifiers.append(value)
                declarations.append(f"Classifier={_bounded_declaration(value)}")
    summary = ", ".join(declarations[:8]) or "<none>"
    if len(declarations) > 8:
        summary += f", ... ({len(declarations)} declarations)"
    if expressions:
        try:
            parsed = [_spdx_expression(value) for value in expressions]
        except ValueError as error:
            raise ValueError(f"{label} License-Expression rejected ({summary}): {error}") from error
        if len(set(parsed)) != 1:
            raise ValueError(f"{label} has ambiguous License-Expression declarations ({summary})")
        return parsed[0]
    if classifiers:
        mapped = [CLASSIFIER_ALIASES[value] for value in classifiers if value in CLASSIFIER_ALIASES]
        unknown = [value for value in classifiers if value not in CLASSIFIER_ALIASES]
        if unknown and not mapped:
            raise ValueError(f"{label} has no recognized classifier license ({summary})")
        if unknown and mapped:
            raise ValueError(f"{label} has ambiguous classifier license declarations ({summary})")
        if len(set(mapped)) != 1:
            raise ValueError(f"{label} has ambiguous classifier license declarations ({summary})")
        return mapped[0]
    parsed_legacy: list[str] = []
    for value in legacy:
        lowered = value.lower().strip()
        if lowered in SPDX_ALIASES:
            parsed_legacy.append(SPDX_ALIASES[lowered])
    if parsed_legacy and len(set(parsed_legacy)) == 1:
        return parsed_legacy[0]
    if parsed_legacy:
        raise ValueError(f"{label} has ambiguous legacy license declarations ({summary})")
    raise ValueError(f"{label} has no recognized license declarations ({summary})")


def _publisher_url(metadata: bytes) -> str | None:
    for line in metadata.decode("utf-8", errors="replace").splitlines():
        if line.lower().startswith("project-url:"):
            value = line.split(":", 1)[1].strip()
            if "," in value:
                label, url = (part.strip() for part in value.split(",", 1))
                if any(token in label.lower() for token in ("source", "repository", "home", "project")) and url.startswith("https://"):
                    return url
        if line.lower().startswith("home-page:"):
            url = line.split(":", 1)[1].strip()
            if url.startswith("https://"):
                return url
    return None


def _bounded_read(stream: BinaryIO, limit: int, label: str) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds the bounded evidence limit")
    return data


def _stream_digest(stream: BinaryIO, limit: int, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = stream.read(CHUNK_SIZE)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise ValueError(f"{label} exceeds the bounded evidence limit")
        digest.update(block)
    return digest.hexdigest(), total


def inspect_archive(path: Path, kind: str) -> tuple[bytes, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return primary license, SPDX, native payloads, and bundled licenses."""
    license_entries: list[tuple[str, bytes]] = []
    metadata_entries: list[tuple[str, bytes]] = []
    native_entries: list[dict[str, Any]] = []
    if kind == "wheel":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not _safe_member(info.filename):
                    raise ValueError("wheel contains an unsafe path")
                if info.is_dir():
                    continue
                lower = info.filename.lower()
                if _license_member(info.filename):
                    with archive.open(info) as stream:
                        license_entries.append((info.filename, _bounded_read(stream, MAX_LICENSE_BYTES, info.filename)))
                if lower.endswith(".dist-info/metadata"):
                    with archive.open(info) as stream:
                        metadata_entries.append((info.filename, _bounded_read(stream, MAX_METADATA_BYTES, info.filename)))
                if lower.endswith(NATIVE_SUFFIXES):
                    with archive.open(info) as stream:
                        digest, size = _stream_digest(stream, MAX_NATIVE_BYTES, info.filename)
                    native_entries.append({"name": info.filename, "sha256": digest, "bytes": size})
    elif kind == "sdist":
        with tarfile.open(path, mode="r:*") as archive:
            for info in archive.getmembers():
                if not _safe_member(info.name):
                    raise ValueError("sdist contains an unsafe path")
                if not info.isfile():
                    continue
                stream = archive.extractfile(info)
                if stream is None:
                    continue
                lower = info.name.lower()
                if _license_member(info.name):
                    license_entries.append((info.name, _bounded_read(stream, MAX_LICENSE_BYTES, info.name)))
                if lower.endswith((".dist-info/metadata", "/pkg-info", "pkg-info")):
                    stream.seek(0)
                    metadata_entries.append((info.name, _bounded_read(stream, MAX_METADATA_BYTES, info.name)))
                if lower.endswith(NATIVE_SUFFIXES):
                    stream.seek(0)
                    digest, size = _stream_digest(stream, MAX_NATIVE_BYTES, info.name)
                    native_entries.append({"name": info.name, "sha256": digest, "bytes": size})
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    if not license_entries:
        raise ValueError(f"{path.name} has no unambiguous license bytes")
    ranked = sorted(enumerate(license_entries), key=lambda item: _license_rank(item[1][0], kind))
    primary_index = ranked[0][0]
    primary_rank = _license_rank(license_entries[primary_index][0], kind)
    if primary_rank[0] == 3:
        raise ValueError(f"{path.name} has no distribution-owned primary license location")
    primary_entries = [item for item in ranked if _license_rank(item[1][0], kind) == primary_rank]
    if len(primary_entries) > 1 and any(item[1][1] != primary_entries[0][1] for item in primary_entries[1:]):
        raise ValueError(f"{path.name} has conflicting license files")
    license_bytes = license_entries[primary_index][1]
    bundled_licenses = [
        {"path": name, "sha256": _sha256_bytes(data), "bytes": len(data)}
        for index, (name, data) in enumerate(license_entries)
        if index != primary_index
    ]
    if not metadata_entries:
        raise ValueError(f"{path.name} has no metadata license declarations (metadata=<none>)")
    metadata_entries.sort(key=lambda item: item[0].lower())
    parsed_metadata: list[str] = []
    metadata_errors: list[str] = []
    for metadata_name, metadata in metadata_entries:
        try:
            parsed_metadata.append(_metadata_license(metadata, metadata_name))
        except ValueError as error:
            metadata_errors.append(str(error))
    if metadata_errors:
        details = "; ".join(metadata_errors[:8])
        if len(metadata_errors) > 8:
            details += f"; ... ({len(metadata_errors)} metadata files)"
        raise ValueError(f"{path.name} metadata license declarations rejected: {_bounded_declaration(details)}")
    if len(set(parsed_metadata)) != 1:
        declarations = ", ".join(f"{name}={value}" for (name, _), value in zip(metadata_entries, parsed_metadata))
        raise ValueError(f"{path.name} has ambiguous metadata license declarations ({_bounded_declaration(declarations)})")
    spdx = parsed_metadata[0]
    return license_bytes, spdx, native_entries, bundled_licenses


def _download(url: str, destination: Path) -> int:
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": "vokra-xy-tokenizer-dependency-audit/1"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as stream:
        final_url = response.geturl()
        if urlsplit(final_url).scheme != "https" or not urlsplit(final_url).netloc:
            raise ValueError("artifact download redirect did not remain HTTPS")
        while True:
            block = response.read(CHUNK_SIZE)
            if not block:
                break
            stream.write(block)
            digest.update(block)
            total += len(block)
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest.hexdigest() + "\n", encoding="ascii")
    return total


def _artifact_candidates(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    wheels = [artifact for artifact in row.get("wheels", []) if supported_wheel(artifact["url"])]
    return [("wheel", artifact) for artifact in wheels] + ([ ("sdist", row["sdist"]) ] if "sdist" in row else [])


def _collect_package(row: dict[str, Any], artifacts_dir: Path, licenses_dir: Path) -> dict[str, Any]:
    identity = (row["name"], row["version"])
    candidates = _artifact_candidates(row)
    if not candidates:
        raise ValueError("no compatible locked artifact")
    candidate_errors: list[str] = []
    license_path = licenses_dir / f"{row['name']}-{row['version']}"
    for kind, lock_artifact in candidates:
        artifact_path = artifacts_dir / f"{row['name']}-{row['version']}.{kind}"
        try:
            size = _download(lock_artifact["url"], artifact_path)
            actual_hash = "sha256:" + sha256(artifact_path)
            if actual_hash != lock_artifact["hash"] or ("size" in lock_artifact and size != lock_artifact["size"]) or size <= 0:
                raise ValueError("downloaded artifact does not match uv.lock")
            license_bytes, spdx, native_entries, bundled_licenses = inspect_archive(artifact_path, kind)
            if identity[0] in NATIVE_PACKAGES and not native_entries:
                raise ValueError("selected native package has no bundled native payload")
            license_path.write_bytes(license_bytes)
            payloads = [{"name": native["name"], "sha256": native["sha256"], "bytes": native["bytes"], "license_source": lock_artifact["url"], "license_revision": lock_artifact["hash"], "artifact_sha256": lock_artifact["hash"]} for native in native_entries]
            bundled = [{**license, "artifact_sha256": actual_hash} for license in bundled_licenses]
            if kind == "wheel":
                metadata = b""
                try:
                    with zipfile.ZipFile(artifact_path) as archive:
                        metadata_name = next(name for name in archive.namelist() if name.lower().endswith(".dist-info/metadata"))
                        with archive.open(metadata_name) as stream:
                            metadata = _bounded_read(stream, MAX_METADATA_BYTES, metadata_name)
                except (StopIteration, KeyError):
                    pass
                license_source = _publisher_url(metadata)
                if license_source is None:
                    raise ValueError("wheel lacks a primary publisher source URL")
            else:
                license_source = lock_artifact["url"]
            return {
                "name": identity[0], "version": identity[1],
                "license": {"kind": "locked-sdist" if kind == "sdist" else "publisher", "source": license_source, "revision": lock_artifact["hash"], "sha256": _sha256_bytes(license_bytes), "bytes": len(license_bytes), "spdx": spdx},
                "artifact": {"kind": kind, "url": lock_artifact["url"], "sha256": lock_artifact["hash"], "bytes": size},
                "native_payloads": payloads,
                "bundled_licenses": bundled,
            }
        except Exception as error:  # preserve each exact candidate's failure before fallback
            candidate_errors.append(f"{kind} {lock_artifact['url']}: {error}")
            artifact_path.unlink(missing_ok=True)
            artifact_path.with_suffix(artifact_path.suffix + ".sha256").unlink(missing_ok=True)
            license_path.unlink(missing_ok=True)
    raise ValueError("; ".join(candidate_errors))


def collect(project: Path, output: Path) -> dict[str, Any]:
    if not project.is_absolute() or project.is_symlink() or not project.is_dir() or any(parent.is_symlink() for parent in project.parents):
        raise ValueError("project must be an absolute non-symlink directory")
    if output.exists() or output.is_symlink() or not output.is_absolute() or any(parent.is_symlink() for parent in output.parents) or not output.parent.is_dir():
        raise ValueError("evidence output must be an absent absolute path")
    lock_data, all_rows = lock_rows(project / "uv.lock")
    active = active_closure(lock_data, all_rows)
    repository_license_path = repository_license(project)
    output.mkdir()
    artifacts_dir = output / "artifacts"
    licenses_dir = output / "license-bytes"
    for directory in (artifacts_dir, licenses_dir):
        directory.mkdir()
    package_evidence: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in active:
        identity = (row["name"], row["version"])
        try:
            if row["source"] == {"virtual": "."}:
                project_file = project / "pyproject.toml"
                project_bytes = project_file.read_bytes()
                license_bytes = repository_license_path.read_bytes()
                package_evidence.append({
                    "name": identity[0], "version": identity[1],
                    "license": {"kind": "local-file", "source": "LICENSE", "revision": sha256(repository_license_path), "sha256": _sha256_bytes(license_bytes), "bytes": len(license_bytes), "spdx": "Apache-2.0"},
                    "artifact": {"kind": "virtual-local", "url": "pyproject.toml", "sha256": _sha256_bytes(project_bytes), "bytes": len(project_bytes)},
                    "native_payloads": [], "bundled_licenses": [],
                })
            else:
                package_evidence.append(_collect_package(row, artifacts_dir, licenses_dir))
        except Exception as error:
            failures.append({"name": identity[0], "version": identity[1], "reason": str(error)})
    evidence = {"packages": package_evidence}
    (output / "license_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"status": "BLOCKED" if failures else "COLLECTED", "active_package_count": len(active), "successful_package_count": len(package_evidence), "failures": failures}
    (output / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "license_evidence": str(output / "license_evidence.json"), "collection_report": str(output / "collection_report.json")}


def self_test() -> None:
    global _collect_package, _download, active_closure, lock_rows, repository_license
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        root = Path(directory)
        wheel = root / "demo.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nLicense: MIT\nProject-URL: Source, https://example.invalid/demo\n")
            archive.writestr("demo-1.0.dist-info/LICENSE", b"MIT License\n")
            archive.writestr("demo-1.0.dist-info/COPYING", b"A different attribution/license copy is lower priority.\n")
            archive.writestr("demo-1.0.dist-info/NOTICE", b"Attribution text may differ from the license.\n")
            archive.writestr("demo-1.0.dist-info/COPYRIGHT", b"Copyright notices are not license bytes.\n")
            archive.writestr("demo/vendor/LICENSE", b"Bundled component license.\n")
            archive.writestr("demo/native.so", b"native")
        license_bytes, spdx, native, bundled = inspect_archive(wheel, "wheel")
        assert license_bytes == b"MIT License\n" and spdx == "MIT" and native[0]["name"] == "demo/native.so"
        assert {entry["path"] for entry in bundled} == {"demo-1.0.dist-info/COPYING", "demo/vendor/LICENSE"}
        assert _metadata_license(b"License-Expression: BSD-3-Clause\n") == "BSD-3-Clause"
        assert _metadata_license(b"License-Expression: MIT AND MPL-2.0 OR PSF-2.0\n") == "MIT AND MPL-2.0 OR PSF-2.0"
        assert _metadata_license(b"Classifier: License :: OSI Approved :: CNRI Python License\n") == "CNRI-Python"
        assert _metadata_license(b"License: Python Software Foundation License\n") == "PSF-2.0"
        for malformed in (
            b"License: BSD\n",
            b"License: BSD License\n",
            b"Classifier: License :: OSI Approved :: BSD License\n",
            b"License-Expression: GPL-3.0\n",
            b"License-Expression: LGPL-2.1-or-later\n",
            b"License-Expression: MIT XOR BSD-3-Clause\n",
            b"License-Expression: MIT\nLicense-Expression: BSD-3-Clause\n",
            b"Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)\n",
            b"Classifier: License :: OSI Approved :: MIT License\nClassifier: License :: OSI Approved :: BSD License\n",
        ):
            try:
                _metadata_license(malformed)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid or ambiguous metadata license accepted")
        metadata_conflict = root / "metadata-conflict.whl"
        with zipfile.ZipFile(metadata_conflict, "w") as archive:
            archive.writestr("a-1.0.dist-info/METADATA", "License-Expression: MIT\n")
            archive.writestr("b-1.0.dist-info/METADATA", "License-Expression: BSD-3-Clause\n")
            archive.writestr("LICENSE", b"MIT License\n")
        try:
            inspect_archive(metadata_conflict, "wheel")
        except ValueError as error:
            assert "ambiguous metadata" in str(error)
        else:
            raise AssertionError("ambiguous metadata declarations accepted")
        ambiguous = root / "ambiguous.whl"
        with zipfile.ZipFile(ambiguous, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA", "License: MIT\n")
            archive.writestr("demo-1.0.dist-info/LICENSE", b"one\n")
            archive.writestr("demo-1.0.dist-info/license", b"two\n")
        try:
            inspect_archive(ambiguous, "wheel")
        except ValueError as error:
            assert "conflicting" in str(error)
        else:
            raise AssertionError("same-priority license ambiguity accepted")
        artifacts_dir = root / "artifacts"
        licenses_dir = root / "licenses"
        artifacts_dir.mkdir()
        licenses_dir.mkdir()
        original_download = _download
        bad_wheel = root / "bad-source.whl"
        with zipfile.ZipFile(bad_wheel, "w") as archive:
            archive.writestr("demo-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nLicense: MIT\n")
            archive.writestr("demo-1.0.dist-info/LICENSE", b"stale candidate\n")

        def fake_download(url: str, destination: Path) -> int:
            data = bad_wheel.read_bytes() if "bad-source" in url else wheel.read_bytes()
            destination.write_bytes(data)
            return len(data)

        _download = fake_download
        try:
            fake_row = {"name": "demo", "version": "1.0", "wheels": [{"url": "https://example.invalid/bad-source-1.0-py3-none-manylinux_2_17_x86_64.whl", "hash": "sha256:" + _sha256_bytes(bad_wheel.read_bytes()), "size": bad_wheel.stat().st_size, "upload-time": "2026-01-01T00:00:00Z"}, {"url": "https://example.invalid/demo-1.0-py3-none-manylinux_2_17_x86_64.whl", "hash": "sha256:" + _sha256_bytes(wheel.read_bytes()), "size": wheel.stat().st_size, "upload-time": "2026-01-01T00:00:00Z"}]}
            collected = _collect_package(fake_row, artifacts_dir, licenses_dir)
            assert collected["artifact"]["bytes"] == wheel.stat().st_size and collected["native_payloads"][0]["artifact_sha256"] == collected["artifact"]["sha256"]
            assert (licenses_dir / "demo-1.0").read_bytes() == b"MIT License\n"
        finally:
            _download = original_download
        class FakeResponse:
            def __init__(self, payload: bytes, final_url: str) -> None:
                self.payload, self.final_url = payload, final_url

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def read(self, size: int = -1) -> bytes:
                payload, self.payload = self.payload, b""
                return payload

            def geturl(self) -> str:
                return self.final_url

        original_urlopen = urllib.request.urlopen
        seen_request: list[urllib.request.Request] = []

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            seen_request.append(request)
            return FakeResponse(b"downloaded", "https://download-r2.pytorch.org/whl/cpu/demo.whl")

        urllib.request.urlopen = fake_urlopen
        try:
            downloaded = root / "downloaded.whl"
            assert _download("https://download-r2.pytorch.org/whl/cpu/demo.whl", downloaded) == len(b"downloaded")
            assert seen_request[0].headers.get("User-agent") == "vokra-xy-tokenizer-dependency-audit/1"
        finally:
            urllib.request.urlopen = original_urlopen
        def fake_http_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            return FakeResponse(b"downloaded", "http://download-r2.pytorch.org/whl/cpu/demo.whl")

        urllib.request.urlopen = fake_http_urlopen
        try:
            try:
                _download("https://download-r2.pytorch.org/whl/cpu/demo.whl", root / "http-redirect.whl")
            except ValueError as error:
                assert "HTTPS" in str(error)
            else:
                raise AssertionError("HTTP artifact redirect accepted")
        finally:
            urllib.request.urlopen = original_urlopen
        project = root / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (project / "uv.lock").write_text("fake", encoding="utf-8")
        (project / "LICENSE").write_bytes(b"Apache License\n")
        output = root / "partial-report"
        original_lock_rows, original_active, original_repository_license, original_collect_package = lock_rows, active_closure, repository_license, _collect_package
        fake_rows = [
            {"name": "vokra-xy-tokenizer-reference", "version": "0.1.0", "source": {"virtual": "."}},
            {"name": "broken", "version": "1.0", "source": {"registry": "https://pypi.org/simple"}},
        ]

        def fake_lock_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return {}, fake_rows

        def fake_active(data: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return rows

        def fake_repository_license(path: Path) -> Path:
            return project / "LICENSE"

        def fake_collect_package(row: dict[str, Any], artifacts: Path, licenses: Path) -> dict[str, Any]:
            if row["name"] == "broken":
                raise ValueError("candidate one failed; candidate two failed")
            raise AssertionError("virtual row was not handled locally")

        lock_rows, active_closure, repository_license, _collect_package = fake_lock_rows, fake_active, fake_repository_license, fake_collect_package
        try:
            partial = collect(project, output)
            assert partial["status"] == "BLOCKED" and partial["active_package_count"] == 2 and partial["successful_package_count"] == 1 and len(partial["failures"]) == 1
            assert json.loads((output / "collection_report.json").read_text(encoding="utf-8"))["failures"][0]["name"] == "broken"
            virtual_evidence = json.loads((output / "license_evidence.json").read_text(encoding="utf-8"))["packages"][0]
            assert virtual_evidence["artifact"]["sha256"] == _sha256_bytes((project / "pyproject.toml").read_bytes())
        finally:
            lock_rows, active_closure, repository_license, _collect_package = original_lock_rows, original_active, original_repository_license, original_collect_package
        try:
            evaluate = inspect_archive
            bad = root / "bad.whl"
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr("demo-1.0.dist-info/METADATA", "License: UNKNOWN\n")
                archive.writestr("demo-1.0.dist-info/LICENSE", b"unknown\n")
            evaluate(bad, "wheel")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown license was accepted")
    print("xy_tokenizer_reference/collect_evidence.py self-test: OK (fake archive only)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if args.project is not None or args.output is not None:
            parser.error("--self-test accepts no paths")
        self_test()
        return 0
    if args.project is None or args.output is None:
        parser.error("--project and --output are required")
    result = collect(args.project, args.output)
    print(json.dumps(result, sort_keys=True))
    return 2 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
