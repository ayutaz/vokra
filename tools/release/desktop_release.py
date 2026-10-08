#!/usr/bin/env python3
"""Assemble and verify the Vokra desktop release payload.

This module is deliberately standard-library-only.  It verifies binary format
and machine identity from the payload bytes, rather than treating a filename as
proof of architecture.  The release workflow uses the same code for native
artifact handoff, deterministic Windows CLI packaging, and the final
all-assets/hash manifest gate.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile

from version_contract import SEMVER_RE


class DesktopReleaseError(ValueError):
    """A release payload failed a fail-closed validation."""


CHECKED_IN_HEADER = Path(__file__).resolve().parents[2] / "include" / "vokra.h"


def _read(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise DesktopReleaseError(f"payload is not a regular file: {path}")
    data = path.read_bytes()
    if not data:
        raise DesktopReleaseError(f"payload is empty: {path}")
    return data


def _elf_machine(data: bytes) -> int | None:
    if len(data) < 20 or data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    endian = "little" if data[5] == 1 else "big" if data[5] == 2 else None
    return int.from_bytes(data[18:20], endian) if endian else None


def _elf_has_interp(data: bytes) -> bool:
    """Return whether an ELF64 payload declares a PT_INTERP loader."""
    if len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2:
        raise DesktopReleaseError("malformed ELF64 header")
    endian = "little" if data[5] == 1 else "big" if data[5] == 2 else None
    if endian is None:
        raise DesktopReleaseError("malformed ELF64 byte order")
    phoff = int.from_bytes(data[32:40], endian)
    phentsize = int.from_bytes(data[54:56], endian)
    phnum = int.from_bytes(data[56:58], endian)
    if phnum == 0:
        return False
    if phentsize < 56:
        raise DesktopReleaseError("malformed ELF64 program header size")
    table_end = phoff + phentsize * phnum
    if phoff > len(data) or table_end > len(data):
        raise DesktopReleaseError("truncated ELF64 program header table")
    for index in range(phnum):
        start = phoff + index * phentsize
        if int.from_bytes(data[start:start + 4], endian) == 3:  # PT_INTERP
            return True
    return False


def _mach_cpu(data: bytes) -> int | None:
    if len(data) < 8:
        return None
    if data[:4] == b"\xcf\xfa\xed\xfe":
        return int.from_bytes(data[4:8], "little")
    if data[:4] == b"\xfe\xed\xfa\xcf":
        return int.from_bytes(data[4:8], "big")
    return None


def _pe_machine(data: bytes) -> int | None:
    if len(data) < 64 or data[:2] != b"MZ":
        return None
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    if pe_offset < 0 or pe_offset + 6 > len(data):
        return None
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        return None
    return int.from_bytes(data[pe_offset + 4:pe_offset + 6], "little")


def verify_binary(path: Path, target: str) -> None:
    """Verify an ELF, Mach-O, or PE payload has the declared machine."""
    data = _read(path)
    if target in ("elf-x86_64", "elf-x86_64-musl"):
        if _elf_machine(data) != 0x3E:
            raise DesktopReleaseError(f"{path}: expected ELF x86_64 binary")
        if target == "elf-x86_64-musl" and _elf_has_interp(data):
            raise DesktopReleaseError(f"{path}: expected static musl ELF without PT_INTERP")
    elif target == "mach-o-arm64":
        if _mach_cpu(data) != 0x0100000C:
            raise DesktopReleaseError(f"{path}: expected thin Mach-O arm64 binary")
    elif target == "pe-x86_64":
        if _pe_machine(data) != 0x8664:
            raise DesktopReleaseError(f"{path}: expected PE x86_64 binary")
    else:
        raise DesktopReleaseError(f"unknown binary target identity: {target}")


def native_smoke(path: Path, expected_version: str) -> None:
    """Load a C ABI library and call its model-free version entry point."""
    if not SEMVER_RE.fullmatch(expected_version):
        raise DesktopReleaseError(f"expected runtime version is not SemVer: {expected_version!r}")
    _read(path)
    try:
        library = ctypes.CDLL(str(path))
        version = library.vokra_version
        version.restype = ctypes.c_char_p
        raw = version()
    except (OSError, AttributeError) as exc:
        raise DesktopReleaseError(f"could not load C ABI library {path}: {exc}") from exc
    if not raw:
        raise DesktopReleaseError(f"vokra_version returned NULL: {path}")
    try:
        actual_version = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DesktopReleaseError(f"vokra_version returned non-UTF-8 data: {path}") from exc
    if actual_version != expected_version:
        raise DesktopReleaseError(
            f"vokra_version mismatch for {path}: {actual_version!r} != {expected_version!r}"
        )


def _safe_zip_member(name: str) -> None:
    # ZIP names are always slash-separated by the format.  Backslashes are
    # rejected too because Windows extraction APIs may treat them as separators.
    if not name or name.startswith(("/", "\\")) or "\\" in name:
        raise DesktopReleaseError(f"unsafe ZIP member path: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DesktopReleaseError(f"unsafe ZIP member path: {name!r}")


def verify_windows_cli_zip(path: Path) -> None:
    _read(path)
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "vokra-cli.exe":
                names = [info.filename for info in infos]
                raise DesktopReleaseError(
                    f"{path}: expected exactly vokra-cli.exe, got {names!r}"
                )
            info = infos[0]
            _safe_zip_member(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise DesktopReleaseError(f"{path}: ZIP member is a symlink")
            payload = archive.read(info)
            if not payload:
                raise DesktopReleaseError(f"{path}: vokra-cli.exe is empty")
            # Validate the member bytes, not merely the archive filename.
            if _pe_machine(payload) != 0x8664:
                raise DesktopReleaseError(
                    f"{path}: vokra-cli.exe is not a PE x86_64 binary"
                )
    except zipfile.BadZipFile as exc:
        raise DesktopReleaseError(f"{path}: invalid ZIP: {exc}") from exc


def _single_payload(root: Path, expected_name: str) -> Path:
    if not root.is_dir():
        raise DesktopReleaseError(f"payload directory missing: {root}")
    entries = sorted(root.rglob("*"))
    if any(entry.is_dir() for entry in entries):
        raise DesktopReleaseError(f"{root}: unexpected directory in payload")
    files = [entry for entry in entries if entry.is_file() or entry.is_symlink()]
    if len(files) != 1 or files[0].name != expected_name:
        names = [str(entry.relative_to(root)) for entry in files]
        raise DesktopReleaseError(
            f"{root}: expected one payload named {expected_name!r}, got {names!r}"
        )
    return files[0]


def _asset_names(version: str) -> dict[str, str]:
    if not SEMVER_RE.fullmatch(version):
        raise DesktopReleaseError(f"release version is not unprefixed SemVer: {version!r}")
    return {
        "capi-linux": f"libvokra-{version}-x86_64-linux.so",
        "capi-macos": f"libvokra-{version}-macos.dylib",
        "capi-windows": f"vokra-{version}-windows.dll",
        "cli-linux": f"vokra-cli-{version}-x86_64-linux",
        "cli-linux-musl": f"vokra-cli-{version}-x86_64-linux-musl",
        "cli-macos": f"vokra-cli-{version}-aarch64-macos",
        "cli-windows-zip": f"vokra-cli-{version}-x86_64-windows.zip",
        "header": "vokra.h",
    }


def _copy_binary(source: Path, destination: Path, target: str, executable: bool = False) -> None:
    verify_binary(source, target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o755 if executable else stat.S_IMODE(source.stat().st_mode))


def _verify_header(path: Path) -> None:
    data = _read(path)
    try:
        expected = CHECKED_IN_HEADER.read_bytes()
    except OSError as exc:
        raise DesktopReleaseError(f"checked-in C ABI header is unavailable: {CHECKED_IN_HEADER}") from exc
    if data != expected:
        raise DesktopReleaseError(f"header does not match the checked-in cbindgen contract: {path}")


def _write_windows_zip(source: Path, destination: Path) -> None:
    verify_binary(source, "pe-x86_64")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Fixed metadata and stored bytes make the archive byte-identical across
    # runs.  The member basename is part of the winget manifest contract.
    info = zipfile.ZipInfo("vokra-cli.exe", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.external_attr = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(info, source.read_bytes())


def assemble(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise DesktopReleaseError(f"assembly output must be empty: {out}")
    names = _asset_names(args.version)
    _copy_binary(Path(args.capi_linux), out / names["capi-linux"], "elf-x86_64")
    _copy_binary(Path(args.capi_macos), out / names["capi-macos"], "mach-o-arm64")
    _copy_binary(Path(args.capi_windows), out / names["capi-windows"], "pe-x86_64")
    _copy_binary(Path(args.cli_linux), out / names["cli-linux"], "elf-x86_64", executable=True)
    _copy_binary(Path(args.cli_linux_musl), out / names["cli-linux-musl"], "elf-x86_64-musl", executable=True)
    _copy_binary(Path(args.cli_macos), out / names["cli-macos"], "mach-o-arm64", executable=True)
    _write_windows_zip(Path(args.cli_windows), out / names["cli-windows-zip"])
    header = Path(args.header)
    _verify_header(header)
    shutil.copyfile(header, out / names["header"])
    out.joinpath(names["header"]).chmod(stat.S_IMODE(header.stat().st_mode))


def _primary_names(version: str) -> set[str]:
    return set(_asset_names(version).values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_primary(out: Path, version: str) -> None:
    names = _asset_names(version)
    expected_targets = {
        "capi-linux": "elf-x86_64",
        "capi-macos": "mach-o-arm64",
        "capi-windows": "pe-x86_64",
        "cli-linux": "elf-x86_64",
        "cli-linux-musl": "elf-x86_64-musl",
        "cli-macos": "mach-o-arm64",
    }
    for kind, target in expected_targets.items():
        path = out / names[kind]
        verify_binary(path, target)
        if kind.startswith("cli-") and not (path.stat().st_mode & stat.S_IXUSR):
            raise DesktopReleaseError(f"CLI payload is not executable: {path}")
    verify_windows_cli_zip(out / names["cli-windows-zip"])
    _verify_header(out / names["header"])


def _asset_metadata(version: str) -> dict[str, dict[str, str]]:
    names = _asset_names(version)
    return {
        names["capi-linux"]: {"target": "ELF x86_64 Linux", "role": "shared-library"},
        names["capi-macos"]: {"target": "thin Mach-O arm64 macOS", "role": "shared-library"},
        names["capi-windows"]: {"target": "PE x86_64 Windows", "role": "shared-library"},
        names["cli-linux"]: {"target": "ELF x86_64 Linux glibc", "role": "cli"},
        names["cli-linux-musl"]: {"target": "ELF x86_64 Linux musl (explicit target build)", "role": "cli"},
        names["cli-macos"]: {"target": "thin Mach-O arm64 macOS", "role": "cli"},
        names["cli-windows-zip"]: {"target": "ZIP containing PE x86_64 Windows CLI", "role": "cli"},
        names["header"]: {"target": "C ABI header", "role": "header"},
        "vokra-desktop-capi.spdx.json": {"target": "vokra-capi dependency closure", "role": "sbom"},
        "vokra-desktop-cli.spdx.json": {"target": "vokra-cli dependency closure", "role": "sbom"},
    }


def _verify_sbom(path: Path, package: str, doc_name: str) -> None:
    try:
        document = json.loads(_read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DesktopReleaseError(f"SBOM is not valid UTF-8 SPDX JSON: {path}") from exc
    if (
        not isinstance(document, dict)
        or document.get("SPDXID") != "SPDXRef-DOCUMENT"
        or document.get("spdxVersion") != "SPDX-2.3"
        or document.get("name") != doc_name
    ):
        raise DesktopReleaseError(f"SBOM schema/name mismatch: {path}")
    packages = document.get("packages")
    root_id = "SPDXRef-Package-" + package
    if not isinstance(packages, list) or not any(
        isinstance(item, dict)
        and item.get("SPDXID") == root_id
        and item.get("name") == package
        for item in packages
    ):
        raise DesktopReleaseError(f"SBOM does not describe its root package {package}: {path}")
    relationships = document.get("relationships")
    if not isinstance(relationships, list) or not any(
        isinstance(item, dict)
        and item.get("spdxElementId") == "SPDXRef-DOCUMENT"
        and item.get("relationshipType") == "DESCRIBES"
        and item.get("relatedSpdxElement") == root_id
        for item in relationships
    ):
        raise DesktopReleaseError(f"SBOM root relationship is missing: {path}")


def write_manifest(args: argparse.Namespace) -> None:
    out = Path(args.dist_dir)
    if not out.is_dir() or any(path.is_dir() or path.is_symlink() or not path.is_file() for path in out.iterdir()):
        raise DesktopReleaseError("desktop release directory contains a non-regular entry")
    _verify_primary(out, args.version)
    required = sorted(_primary_names(args.version))
    for name, package in (
        ("vokra-desktop-capi.spdx.json", "vokra-capi"),
        ("vokra-desktop-cli.spdx.json", "vokra-cli"),
    ):
        _verify_sbom(out / name, package, name.removesuffix(".spdx.json"))
        required.append(name)
    required = sorted(required)
    metadata = _asset_metadata(args.version)
    assets = [
        {"name": name, **metadata[name], "sha256": _sha256(out / name), "size": (out / name).stat().st_size}
        for name in required
    ]
    document = {
        "schema": "vokra-desktop-release-v1",
        "version": args.version,
        "assets": assets,
    }
    manifest = out / "vokra-desktop-manifest.json"
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
    for name in required + [manifest.name]:
        (out / f"{name}.sha256").write_text(
            f"{_sha256(out / name)}  {name}\n", encoding="utf-8"
        )


def verify_dist(args: argparse.Namespace) -> None:
    out = Path(args.dist_dir)
    _verify_primary(out, args.version)
    if not out.is_dir():
        raise DesktopReleaseError("desktop release directory is missing")
    entries = list(out.iterdir())
    if any(path.is_dir() or path.is_symlink() or not path.is_file() for path in entries):
        raise DesktopReleaseError("desktop release directory contains a non-regular entry")
    manifest_path = out / "vokra-desktop-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DesktopReleaseError(f"manifest missing: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DesktopReleaseError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"assets", "schema", "version"}:
        raise DesktopReleaseError("manifest top-level schema is invalid")
    if document["schema"] != "vokra-desktop-release-v1" or document["version"] != args.version:
        raise DesktopReleaseError("manifest schema/version mismatch")
    _verify_sbom(out / "vokra-desktop-capi.spdx.json", "vokra-capi", "vokra-desktop-capi")
    _verify_sbom(out / "vokra-desktop-cli.spdx.json", "vokra-cli", "vokra-desktop-cli")
    assets = document.get("assets")
    expected_assets = sorted(_primary_names(args.version) | {"vokra-desktop-capi.spdx.json", "vokra-desktop-cli.spdx.json"})
    if not isinstance(assets, list) or [item.get("name") if isinstance(item, dict) else None for item in assets] != expected_assets:
        raise DesktopReleaseError("manifest does not enumerate exactly the required assets")
    metadata = _asset_metadata(args.version)
    for item in assets:
        if not isinstance(item, dict) or set(item) != {"name", "target", "role", "sha256", "size"}:
            raise DesktopReleaseError("manifest asset entry has invalid schema")
        name = item["name"]
        if not isinstance(name, str) or name not in metadata:
            raise DesktopReleaseError("manifest asset entry has invalid name")
        if item["target"] != metadata[name]["target"] or item["role"] != metadata[name]["role"]:
            raise DesktopReleaseError(f"manifest identity mismatch: {name}")
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise DesktopReleaseError(f"manifest hash is malformed: {name}")
        if not isinstance(item["size"], int) or isinstance(item["size"], bool) or item["size"] <= 0:
            raise DesktopReleaseError(f"manifest size is malformed: {name}")
        path = out / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise DesktopReleaseError(f"manifest hash/size mismatch: {path}")
    expected_files = set(expected_assets) | {manifest_path.name}
    expected_files |= {f"{name}.sha256" for name in expected_files}
    actual_files = {path.name for path in entries}
    if actual_files != expected_files:
        raise DesktopReleaseError(f"desktop release file set mismatch: {sorted(actual_files ^ expected_files)}")
    for name in expected_assets + [manifest_path.name]:
        sidecar = out / f"{name}.sha256"
        lines = sidecar.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1 or len(lines[0].split()) != 2 or lines[0].split()[1] != name:
            raise DesktopReleaseError(f"malformed SHA256 sidecar: {sidecar}")
        digest = lines[0].split()[0]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise DesktopReleaseError(f"malformed SHA256 sidecar digest: {sidecar}")
        if digest != _sha256(out / name):
            raise DesktopReleaseError(f"SHA256 sidecar mismatch: {sidecar}")


def verify_binary_command(args: argparse.Namespace) -> None:
    targets = {
        "linux-x86_64": "elf-x86_64",
        "linux-x86_64-musl": "elf-x86_64-musl",
        "macos-arm64": "mach-o-arm64",
        "windows-x86_64": "pe-x86_64",
    }
    try:
        verify_binary(Path(args.path), targets[args.target])
    except KeyError as exc:
        raise DesktopReleaseError(f"unknown target: {args.target}") from exc


def native_smoke_command(args: argparse.Namespace) -> None:
    native_smoke(Path(args.path), args.expected_version)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    binary = sub.add_parser("verify-binary")
    binary.add_argument("path")
    binary.add_argument("--target", required=True, choices=("linux-x86_64", "linux-x86_64-musl", "macos-arm64", "windows-x86_64"))
    smoke = sub.add_parser("native-smoke")
    smoke.add_argument("path")
    smoke.add_argument("--expected-version", required=True)
    ass = sub.add_parser("assemble")
    ass.add_argument("--version", required=True)
    ass.add_argument("--out-dir", required=True)
    ass.add_argument("--header", required=True)
    for name in ("capi-linux", "capi-macos", "capi-windows", "cli-linux", "cli-linux-musl", "cli-macos", "cli-windows"):
        ass.add_argument(f"--{name}", required=True)
    man = sub.add_parser("manifest")
    man.add_argument("--version", required=True)
    man.add_argument("--dist-dir", required=True)
    ver = sub.add_parser("verify")
    ver.add_argument("--version", required=True)
    ver.add_argument("--dist-dir", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "verify-binary":
            verify_binary_command(args)
        elif args.command == "native-smoke":
            native_smoke_command(args)
        elif args.command == "assemble":
            assemble(args)
        elif args.command == "manifest":
            write_manifest(args)
        elif args.command == "verify":
            verify_dist(args)
        else:  # pragma: no cover - argparse enforces the subcommands
            raise DesktopReleaseError(f"unknown command: {args.command}")
    except (DesktopReleaseError, OSError, zipfile.BadZipFile) as exc:
        print(f"desktop-release: error: {exc}", file=sys.stderr)
        return 1
    print(f"desktop-release: {args.command} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
