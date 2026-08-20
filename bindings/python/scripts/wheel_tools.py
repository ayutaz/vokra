#!/usr/bin/env python3
"""Fail-loud validation and manifest tooling for Vokra native wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import struct
import sys
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path, PurePosixPath
from typing import NoReturn

TARGETS = {
    "linux-x86_64": {
        "native": "vokra/_lib/libvokra.so",
        "arch": "x86_64",
    },
    "macos-arm64": {
        "native": "vokra/_lib/libvokra.dylib",
        "arch": "arm64",
    },
    "macos-x86_64": {
        "native": "vokra/_lib/libvokra.dylib",
        "arch": "x86_64",
    },
    "windows-x86_64": {
        "native": "vokra/_lib/vokra.dll",
        "arch": "x86_64",
    },
}

_WHEEL_NAME_RE = re.compile(
    r"^vokra-(?P<version>[^-]+)-(?P<python>[^-]+)-"
    r"(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"wheel validation failed: {message}")


def _sha256_record(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _binary_arch(data: bytes) -> str:
    if data.startswith(b"\x7fELF"):
        if len(data) < 20 or data[4] != 2:
            _fail("native ELF is not a complete 64-bit object")
        endian = "<" if data[5] == 1 else ">" if data[5] == 2 else None
        if endian is None:
            _fail("native ELF has an invalid byte-order marker")
        machine = struct.unpack_from(f"{endian}H", data, 18)[0]
        return {62: "x86_64", 183: "aarch64"}.get(machine, f"elf-machine-{machine}")

    if data.startswith(b"\xcf\xfa\xed\xfe"):
        if len(data) < 8:
            _fail("native Mach-O header is truncated")
        cpu_type = struct.unpack_from("<I", data, 4)[0]
        return {
            0x01000007: "x86_64",
            0x0100000C: "arm64",
        }.get(cpu_type, f"macho-cpu-{cpu_type:#x}")

    if data.startswith((b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")):
        return "universal"

    if data.startswith(b"MZ"):
        if len(data) < 64:
            _fail("native PE header is truncated")
        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if pe_offset + 6 > len(data) or data[pe_offset : pe_offset + 4] != b"PE\0\0":
            _fail("native PE signature is missing")
        machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
        return {0x8664: "x86_64", 0xAA64: "arm64"}.get(machine, f"pe-machine-{machine:#x}")

    _fail("native library is not ELF, Mach-O, or PE/COFF")


def _validate_platform(target: str, platform_tag: str) -> None:
    if target == "linux-x86_64":
        components = platform_tag.split(".")
        if "manylinux_2_28_x86_64" not in components:
            _fail(
                "Linux wheel must include the manylinux_2_28_x86_64 policy tag; "
                f"got {platform_tag!r}"
            )
        if any(not part.startswith("manylinux_") or not part.endswith("_x86_64") for part in components):
            _fail(f"Linux wheel has an unexpected compound platform tag {platform_tag!r}")
        return

    expected = {
        "macos-arm64": "macosx_11_0_arm64",
        "macos-x86_64": "macosx_11_0_x86_64",
        "windows-x86_64": "win_amd64",
    }[target]
    if platform_tag != expected:
        _fail(f"{target} wheel platform is {platform_tag!r}, expected {expected!r}")


def _parse_record(record: bytes) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for row in csv.reader(io.StringIO(record.decode("utf-8"))):
        if len(row) != 3:
            _fail(f"RECORD row does not have three fields: {row!r}")
        path, digest, size = row
        if path in rows:
            _fail(f"duplicate RECORD path {path!r}")
        rows[path] = (digest, size)
    return rows


def verify_wheel(wheel: Path, target: str, version: str) -> dict[str, object]:
    if target not in TARGETS:
        _fail(f"unknown target {target!r}")
    match = _WHEEL_NAME_RE.fullmatch(wheel.name)
    if match is None:
        _fail(f"unexpected wheel filename {wheel.name!r}")
    if match["version"] != version:
        _fail(
            f"wheel filename version {match['version']!r} does not match expected {version!r}"
        )
    if (match["python"], match["abi"]) != ("py3", "none"):
        _fail(
            "ctypes wheel must be Python-ABI-independent (py3-none), got "
            f"{match['python']}-{match['abi']}"
        )
    _validate_platform(target, match["platform"])

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        files = [name for name in names if not name.endswith("/")]
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                _fail(f"unsafe archive member path {name!r}")
        if len(files) != len(set(files)):
            _fail("wheel contains duplicate archive members")

        dist_info = sorted({name.split("/", 1)[0] for name in files if ".dist-info/" in name})
        if len(dist_info) != 1:
            _fail(f"wheel must contain exactly one .dist-info directory, got {dist_info!r}")
        metadata_path = f"{dist_info[0]}/METADATA"
        wheel_path = f"{dist_info[0]}/WHEEL"
        record_path = f"{dist_info[0]}/RECORD"
        for required in (metadata_path, wheel_path, record_path):
            if required not in files:
                _fail(f"wheel is missing {required}")

        metadata = BytesParser(policy=compat32).parsebytes(archive.read(metadata_path))
        if metadata.get("Name") != "vokra":
            _fail(f"METADATA Name is {metadata.get('Name')!r}, expected 'vokra'")
        if metadata.get("Version") != version:
            _fail(
                f"METADATA Version is {metadata.get('Version')!r}, expected {version!r}"
            )
        if metadata.get("Requires-Python") != "<3.13,>=3.9":
            _fail(
                "METADATA Requires-Python must match the tested Python 3.9-3.12 "
                f"range, got {metadata.get('Requires-Python')!r}"
            )

        wheel_metadata = BytesParser(policy=compat32).parsebytes(archive.read(wheel_path))
        if wheel_metadata.get("Root-Is-Purelib", "").lower() != "false":
            _fail("WHEEL must declare Root-Is-Purelib: false")
        expected_tag = f"py3-none-{match['platform']}"
        tags = wheel_metadata.get_all("Tag", [])
        if expected_tag not in tags:
            _fail(f"WHEEL tags {tags!r} do not contain filename tag {expected_tag!r}")

        expected_native = TARGETS[target]["native"]
        native_members = [
            name for name in files if name.lower().endswith((".so", ".dylib", ".dll"))
        ]
        if native_members != [expected_native]:
            _fail(
                f"wheel must contain only {expected_native!r} as native payload; "
                f"got {native_members!r}"
            )
        native_data = archive.read(expected_native)
        arch = _binary_arch(native_data)
        if arch != TARGETS[target]["arch"]:
            _fail(
                f"native payload architecture is {arch!r}, expected "
                f"{TARGETS[target]['arch']!r} for {target}"
            )

        record = _parse_record(archive.read(record_path))
        if set(record) != set(files):
            missing = sorted(set(files) - set(record))
            extra = sorted(set(record) - set(files))
            _fail(f"RECORD member mismatch: missing={missing!r}, extra={extra!r}")
        for name in files:
            digest, size = record[name]
            if name == record_path:
                if digest or size:
                    _fail("RECORD's own row must have empty hash and size")
                continue
            data = archive.read(name)
            if digest != _sha256_record(data) or size != str(len(data)):
                _fail(f"RECORD hash/size mismatch for {name}")

    data = wheel.read_bytes()
    return {
        "filename": wheel.name,
        "target": target,
        "python_tag": "py3",
        "abi_tag": "none",
        "platform_tag": match["platform"],
        "native_path": expected_native,
        "native_arch": arch,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _infer_target(wheel: Path) -> str:
    name = wheel.name
    if "manylinux_2_28_x86_64" in name:
        return "linux-x86_64"
    if name.endswith("-macosx_11_0_arm64.whl"):
        return "macos-arm64"
    if name.endswith("-macosx_11_0_x86_64.whl"):
        return "macos-x86_64"
    if name.endswith("-win_amd64.whl"):
        return "windows-x86_64"
    _fail(f"cannot infer supported target from {name!r}")


def build_manifest(directory: Path, version: str) -> dict[str, object]:
    wheels = sorted(directory.rglob("*.whl"))
    if len(wheels) != len(TARGETS):
        _fail(f"expected {len(TARGETS)} wheels, found {len(wheels)} under {directory}")
    records = [verify_wheel(wheel, _infer_target(wheel), version) for wheel in wheels]
    targets = [record["target"] for record in records]
    if sorted(targets) != sorted(TARGETS):
        _fail(f"wheel target set is {targets!r}, expected {sorted(TARGETS)!r}")
    return {
        "schema_version": 1,
        "package": "vokra",
        "version": version,
        "wheel_count": len(records),
        "wheels": sorted(records, key=lambda item: str(item["target"])),
    }


def _normalized_version(raw: str) -> str:
    from packaging.version import InvalidVersion, Version

    candidate = raw.removeprefix("v")
    if "+" in candidate:
        _fail("local/build metadata (+...) is not permitted for release wheels")
    try:
        return str(Version(candidate))
    except InvalidVersion as exc:
        _fail(f"{raw!r} is not convertible to a PEP 440 version: {exc}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    normalize = sub.add_parser("normalize-version")
    normalize.add_argument("version")

    verify = sub.add_parser("verify")
    verify.add_argument("--wheel", type=Path, required=True)
    verify.add_argument("--target", choices=sorted(TARGETS), required=True)
    verify.add_argument("--version", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--directory", type=Path, required=True)
    manifest.add_argument("--version", required=True)
    manifest.add_argument("--output", type=Path)
    manifest.add_argument("--check", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "normalize-version":
        print(_normalized_version(args.version))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_wheel(args.wheel, args.target, args.version), sort_keys=True))
        return 0

    manifest = build_manifest(args.directory, args.version)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != rendered:
            _fail(f"manifest drift: {args.check} does not match the wheel set")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not args.output and not args.check:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
