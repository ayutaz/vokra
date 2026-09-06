#!/usr/bin/env python3
"""Static audit for the exact official qwen-asr Transformers backend wheel.

This module never imports or executes wheel code.  It authenticates the
immutable PyPI artifact, rejects duplicate/path/link members, and binds the
official backend source files and RECORD before a VAST dumper may load them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


WHEEL_URL = (
    "https://files.pythonhosted.org/packages/01/12/d3027a7e4dc2eea0b12a4bf8414a7109f055004e177166e01d8859d3ca0/"
    "qwen_asr-0.0.6-py3-none-any.whl"
)
WHEEL_BYTES = 141_603
WHEEL_SHA256 = "b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3"
LICENSE_MEMBER = "qwen_asr-0.0.6.dist-info/licenses/LICENSE"
LICENSE_BYTES = 11_343
LICENSE_SHA256 = "a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655"
RECORD_MEMBER = "qwen_asr-0.0.6.dist-info/RECORD"
RECORD_BYTES = 1_916
RECORD_SHA256 = "c389f21776ed93a8470f03617cb9387f800ecfca25bdf611d13c9f049d537d52"

BACKEND_MEMBERS = {
    "qwen_asr/core/transformers_backend/__init__.py": (807, "6de5cac018e0c994e04e855ef7fe9ed142383e88ffdcdf80463cbc7d5028b21d"),
    "qwen_asr/core/transformers_backend/configuration_qwen3_asr.py": (20_164, "acf6c3f1cb3dc1ea0df621a11f7df6ccca109be73b6bf51508b5052942f5bec1"),
    "qwen_asr/core/transformers_backend/modeling_qwen3_asr.py": (57_850, "2fb5d98da1933748f5117ee05ce4e7150c9ead8154fb8e25f7af3968b853adc7"),
    "qwen_asr/core/transformers_backend/processing_qwen3_asr.py": (8_545, "a61f1fbfc06e3dfc63a2b8d15e3b29e98b032792c3fb7d0e01c04ca9c2181ffb"),
    # The official wrapper supplies prompt construction; the dumper AST-lifts
    # only its two helper methods and never imports this module wholesale.
    "qwen_asr/inference/qwen3_asr.py": (31_163, "0b1770f8e907b6c5a0a1e9ebce037cb63f48555f3cd15eaf6ea2078e9df41a7b"),
    # parse_asr_output is AST-lifted from this exact support module; its
    # top-level librosa import is deliberately never executed.
    "qwen_asr/inference/utils.py": (13_915, "b37fb4140e6d9be85f2ecb1552ed16136eb98a57f44b402f8b0b213cb6df2a82"),
}

EXPECTED_MEMBERS = frozenset(
    {
        "qwen_asr/__init__.py",
        "qwen_asr/__main__.py",
        "qwen_asr/cli/demo.py",
        "qwen_asr/cli/demo_streaming.py",
        "qwen_asr/cli/serve.py",
        *BACKEND_MEMBERS,
        "qwen_asr/core/vllm_backend/__init__.py",
        "qwen_asr/core/vllm_backend/qwen3_asr.py",
        "qwen_asr/inference/qwen3_forced_aligner.py",
        "qwen_asr/inference/utils.py",
        "qwen_asr/inference/assets/korean_dict_jieba.dict",
        LICENSE_MEMBER,
        "qwen_asr-0.0.6.dist-info/METADATA",
        "qwen_asr-0.0.6.dist-info/WHEEL",
        "qwen_asr-0.0.6.dist-info/entry_points.txt",
        "qwen_asr-0.0.6.dist-info/top_level.txt",
        RECORD_MEMBER,
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> None:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe wheel member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe wheel member path: {name!r}")


def _regular_member(info: zipfile.ZipInfo) -> None:
    if info.is_dir():
        raise ValueError(f"wheel contains unexpected directory member: {info.filename}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode and mode != stat.S_IFREG:
        raise ValueError(f"wheel member is not regular: {info.filename}")
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted wheel member: {info.filename}")


def _validate_archive_members(infos: list[zipfile.ZipInfo]) -> set[str]:
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("wheel contains duplicate member names")
    for info in infos:
        _safe_member_name(info.filename)
        _regular_member(info)
    return set(names)


def audit_wheel(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("wheel must be a regular non-symlink file")
    size = path.stat().st_size
    digest = sha256_file(path)
    if (size, digest) != (WHEEL_BYTES, WHEEL_SHA256):
        raise ValueError(f"wheel identity drift: bytes={size} sha256={digest}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = _validate_archive_members(infos)
        if names != EXPECTED_MEMBERS:
            missing = sorted(EXPECTED_MEMBERS - names)
            extra = sorted(names - EXPECTED_MEMBERS)
            raise ValueError(f"wheel member set drift: missing={missing} extra={extra}")
        evidence: dict[str, dict[str, Any]] = {}
        for name, (expected_bytes, expected_hash) in BACKEND_MEMBERS.items():
            body = archive.read(name)
            actual = (len(body), sha256_bytes(body))
            if actual != (expected_bytes, expected_hash):
                raise ValueError(f"official backend member drift: {name}")
            evidence[name] = {"bytes": actual[0], "sha256": actual[1]}
        license_body = archive.read(LICENSE_MEMBER)
        if (len(license_body), sha256_bytes(license_body)) != (LICENSE_BYTES, LICENSE_SHA256):
            raise ValueError("wheel Apache LICENSE identity drift")
        record_body = archive.read(RECORD_MEMBER)
        if (len(record_body), sha256_bytes(record_body)) != (RECORD_BYTES, RECORD_SHA256):
            raise ValueError("wheel RECORD identity drift")
    return {
        "schema": "vokra-qwen3-asr-official-transformers-wheel-v1",
        "name": "qwen-asr",
        "version": "0.0.6",
        "url": WHEEL_URL,
        "bytes": size,
        "sha256": digest,
        "license": {"member": LICENSE_MEMBER, "bytes": LICENSE_BYTES, "sha256": LICENSE_SHA256, "spdx": "Apache-2.0"},
        "record": {"member": RECORD_MEMBER, "bytes": RECORD_BYTES, "sha256": RECORD_SHA256},
        "official_backend_members": evidence,
        "execution": "OFFICIAL_TRANSFORMERS_BACKEND_ONLY",
        "publication": "NO_UPLOAD",
    }


def extract_backend(path: Path, destination: Path) -> None:
    """Extract only authenticated backend/wrapper members into a new root."""
    report = audit_wheel(path)
    if destination.exists() or destination.is_symlink():
        raise ValueError("backend extraction destination must be absent")
    destination.mkdir()
    with zipfile.ZipFile(path) as archive:
        for name in (*BACKEND_MEMBERS,):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    if report["official_backend_members"].keys() != BACKEND_MEMBERS.keys():
        raise ValueError("backend extraction evidence mismatch")


def _write_report_no_replace(path: Path, report: dict[str, Any]) -> None:
    """Write small evidence atomically without overwriting an existing path."""
    if path.exists() or path.is_symlink():
        raise ValueError("wheel audit output must be absent and non-symlink")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("wheel audit output parent must be an existing directory")
    temporary = Path(tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)[1])
    try:
        payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except OSError as error:
        raise ValueError(f"wheel audit output publication failed: {error}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def self_test() -> int:
    source = Path("/private/tmp/qwen_asr-0.0.6-py3-none-any.whl")
    if source.is_file():
        audit_wheel(source)
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-wheel-test-") as directory:
        fixture = Path(directory) / "fixture.whl"
        fixture.write_bytes(b"not the reviewed wheel")
        tampered = Path(directory) / "tampered.whl"
        tampered.write_bytes(fixture.read_bytes() + b"x")
        try:
            audit_wheel(tampered)
        except ValueError:
            pass
        else:
            raise AssertionError("tampered wheel accepted")
    try:
        _validate_archive_members([zipfile.ZipInfo("qwen_asr/../escape.py")])
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe wheel member accepted")
    try:
        _validate_archive_members([zipfile.ZipInfo("a"), zipfile.ZipInfo("a")])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate wheel member accepted")
    names = _validate_archive_members([zipfile.ZipInfo("qwen_asr/unknown.py")])
    if names <= EXPECTED_MEMBERS:
        raise AssertionError("unknown wheel member set accepted")
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-wheel-output-test-") as directory:
        output = Path(directory) / "audit.json"
        _write_report_no_replace(output, {"schema": "self-test"})
        original = output.read_bytes()
        try:
            _write_report_no_replace(output, {"schema": "tampered"})
        except ValueError:
            pass
        else:
            raise AssertionError("existing wheel audit output was overwritten")
        if output.read_bytes() != original:
            raise AssertionError("existing wheel audit output changed")
        symlink = Path(directory) / "audit-link.json"
        symlink.symlink_to(output)
        try:
            _write_report_no_replace(symlink, {"schema": "tampered"})
        except ValueError:
            pass
        else:
            raise AssertionError("symlink wheel audit output was accepted")
    print("qwen3-asr wheel audit: self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.wheel is None or args.output is None:
        parser.error("--wheel and --output are required")
    report = audit_wheel(args.wheel)
    try:
        _write_report_no_replace(args.output, report)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
