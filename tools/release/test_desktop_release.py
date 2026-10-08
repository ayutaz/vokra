#!/usr/bin/env python3
"""Focused tests for desktop payload identity, packaging, and completeness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("desktop_release", HERE / "desktop_release.py")
assert SPEC and SPEC.loader
desktop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(desktop)


def elf(machine: int = 0x3E, interp: bool = False) -> bytes:
    data = bytearray(120 if interp else 64)
    data[:4] = b"\x7fELF"
    data[4] = 2
    data[5] = 1
    data[18:20] = machine.to_bytes(2, "little")
    if interp:
        data[32:40] = (64).to_bytes(8, "little")
        data[54:56] = (56).to_bytes(2, "little")
        data[56:58] = (1).to_bytes(2, "little")
        data[64:68] = (3).to_bytes(4, "little")
    return bytes(data)


def macho(cpu: int = 0x0100000C) -> bytes:
    return b"\xcf\xfa\xed\xfe" + cpu.to_bytes(4, "little") + bytes(56)


def pe(machine: int = 0x8664) -> bytes:
    data = bytearray(128)
    data[:2] = b"MZ"
    data[0x3C:0x40] = (64).to_bytes(4, "little")
    data[64:68] = b"PE\0\0"
    data[68:70] = machine.to_bytes(2, "little")
    return bytes(data)


def args_for(root: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        version="1.2.3",
        out_dir=str(out),
        header=str(root / "vokra.h"),
        capi_linux=str(root / "capi-linux"),
        capi_macos=str(root / "capi-macos"),
        capi_windows=str(root / "capi-windows"),
        cli_linux=str(root / "cli-linux"),
        cli_linux_musl=str(root / "cli-linux-musl"),
        cli_macos=str(root / "cli-macos"),
        cli_windows=str(root / "cli-windows"),
    )


def sbom(package: str, name: str) -> str:
    root_id = "SPDXRef-Package-" + package
    return json.dumps(
        {
            "spdxVersion": "SPDX-2.3",
            "name": name,
            "packages": [{"SPDXID": root_id, "name": package}],
            "SPDXID": "SPDXRef-DOCUMENT",
            "relationships": [
                {
                    "spdxElementId": "SPDXRef-DOCUMENT",
                    "relationshipType": "DESCRIBES",
                    "relatedSpdxElement": root_id,
                }
            ],
        }
    ) + "\n"


def make_valid_dist(root: Path) -> tuple[argparse.Namespace, Path]:
    for name, data in {
        "capi-linux": elf(),
        "capi-macos": macho(),
        "capi-windows": pe(),
        "cli-linux": elf(interp=True),
        "cli-linux-musl": elf(),
        "cli-macos": macho(),
        "cli-windows": pe(),
    }.items():
        (root / name).write_bytes(data)
    header = root / "vokra.h"
    header.write_bytes((HERE.parents[1] / "include" / "vokra.h").read_bytes())
    out = root / "dist"
    args = args_for(root, out)
    desktop.assemble(args)
    (out / "vokra-desktop-capi.spdx.json").write_text(
        sbom("vokra-capi", "vokra-desktop-capi"), encoding="utf-8"
    )
    (out / "vokra-desktop-cli.spdx.json").write_text(
        sbom("vokra-cli", "vokra-desktop-cli"), encoding="utf-8"
    )
    manifest_args = argparse.Namespace(version="1.2.3", dist_dir=str(out))
    desktop.write_manifest(manifest_args)
    return manifest_args, out


class DesktopReleaseTests(unittest.TestCase):
    def test_binary_identity_rejects_wrong_and_empty_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good"
            good.write_bytes(elf())
            desktop.verify_binary(good, "elf-x86_64")
            wrong = root / "wrong"
            wrong.write_bytes(elf(0xB7))
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_binary(wrong, "elf-x86_64")
            empty = root / "empty"
            empty.touch()
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_binary(empty, "elf-x86_64")

    def test_musl_identity_rejects_dynamic_elf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            static = root / "static"
            static.write_bytes(elf())
            desktop.verify_binary(static, "elf-x86_64-musl")
            dynamic = root / "dynamic"
            dynamic.write_bytes(elf(interp=True))
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_binary(dynamic, "elf-x86_64-musl")

    def test_musl_identity_rejects_malformed_program_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed"
            data = bytearray(64)
            data[:4] = b"\x7fELF"
            data[4] = 2
            data[5] = 1
            data[18:20] = (0x3E).to_bytes(2, "little")
            data[56:58] = (1).to_bytes(2, "little")
            data[54:56] = (4).to_bytes(2, "little")
            malformed.write_bytes(data)
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_binary(malformed, "elf-x86_64-musl")

    def test_native_smoke_checks_exact_capi_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "libvokra.so"
            path.write_bytes(elf())

            class FakeVersion:
                restype = None

                def __call__(self) -> bytes:
                    return b"0.3.0"

            class FakeLibrary:
                vokra_version = FakeVersion()

            original = desktop.ctypes.CDLL
            desktop.ctypes.CDLL = lambda _path: FakeLibrary()
            try:
                desktop.native_smoke(path, "0.3.0")
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.native_smoke(path, "0.3.1")
            finally:
                desktop.ctypes.CDLL = original

    def test_assemble_and_manifest_are_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "capi-linux": elf(),
                "capi-macos": macho(),
                "capi-windows": pe(),
                "cli-linux": elf(interp=True),
                "cli-linux-musl": elf(),
                "cli-macos": macho(),
                "cli-windows": pe(),
            }
            for name, data in files.items():
                (root / name).write_bytes(data)
            header = root / "vokra.h"
            header.write_bytes((HERE.parents[1] / "include" / "vokra.h").read_bytes())
            out = root / "dist"
            desktop.assemble(args_for(root, out))
            (out / "vokra-desktop-capi.spdx.json").write_text(
                sbom("vokra-capi", "vokra-desktop-capi"), encoding="utf-8"
            )
            (out / "vokra-desktop-cli.spdx.json").write_text(
                sbom("vokra-cli", "vokra-desktop-cli"), encoding="utf-8"
            )
            manifest_args = argparse.Namespace(version="1.2.3", dist_dir=str(out))
            desktop.write_manifest(manifest_args)
            desktop.verify_dist(manifest_args)
            self.assertTrue((out / "vokra-cli-1.2.3-x86_64-linux").stat().st_mode & stat.S_IXUSR)
            first_zip = (out / "vokra-cli-1.2.3-x86_64-windows.zip").read_bytes()
            second_zip = root / "second.zip"
            desktop._write_windows_zip(root / "cli-windows", second_zip)
            self.assertEqual(first_zip, second_zip.read_bytes())
            document = json.loads((out / "vokra-desktop-manifest.json").read_text())
            self.assertEqual(document["assets"], sorted(document["assets"], key=lambda item: item["name"]))
            self.assertEqual(document["assets"][0].keys(), {"name", "role", "sha256", "size", "target"})

    def test_missing_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, out = make_valid_dist(Path(tmp))
            (out / "vokra.h.sha256").unlink()
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_dist(args)

    def test_wrong_sidecar_name_and_hash_are_rejected_independently(self) -> None:
        for mode in ("name", "hash"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args, out = make_valid_dist(Path(tmp))
                expected = out / "vokra.h.sha256"
                if mode == "name":
                    expected.rename(out / "wrong-name.sha256")
                else:
                    expected.write_text("0" * 64 + "  vokra.h\n", encoding="utf-8")
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.verify_dist(args)

    def test_extra_regular_symlink_and_directory_are_rejected_independently(self) -> None:
        for mode in ("regular", "symlink", "directory"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args, out = make_valid_dist(Path(tmp))
                extra = out / "unexpected"
                if mode == "regular":
                    extra.write_bytes(b"unexpected")
                elif mode == "symlink":
                    os.symlink("vokra.h", extra)
                else:
                    extra.mkdir()
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.verify_dist(args)

    def test_manifest_and_sbom_symlinks_are_rejected_independently(self) -> None:
        for name in ("vokra-desktop-manifest.json", "vokra-desktop-cli.spdx.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args, out = make_valid_dist(Path(tmp))
                path = out / name
                path.unlink()
                os.symlink("vokra-desktop-capi.spdx.json", path)
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.verify_dist(args)

    def test_manifest_top_level_and_entry_schema_fail_independently(self) -> None:
        for mutation in ("top-level", "entry"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                args, out = make_valid_dist(Path(tmp))
                manifest_path = out / "vokra-desktop-manifest.json"
                if mutation == "top-level":
                    manifest_path.write_text("[]\n", encoding="utf-8")
                else:
                    document = json.loads(manifest_path.read_text())
                    document["assets"][0] = {"name": document["assets"][0]["name"]}
                    manifest_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.verify_dist(args)

    def test_manifest_cannot_hide_changed_invalid_sbom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args, out = make_valid_dist(Path(tmp))
            sbom_path = out / "vokra-desktop-cli.spdx.json"
            sbom_path.write_text("{}\n", encoding="utf-8")
            document = json.loads((out / "vokra-desktop-manifest.json").read_text())
            for item in document["assets"]:
                if item["name"] == sbom_path.name:
                    item["sha256"] = desktop._sha256(sbom_path)
                    item["size"] = sbom_path.stat().st_size
            manifest_path = out / "vokra-desktop-manifest.json"
            manifest_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
            (out / f"{sbom_path.name}.sha256").write_text(
                f"{desktop._sha256(sbom_path)}  {sbom_path.name}\n", encoding="utf-8"
            )
            (out / "vokra-desktop-manifest.json.sha256").write_text(
                f"{desktop._sha256(manifest_path)}  vokra-desktop-manifest.json\n", encoding="utf-8"
            )
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_dist(args)

    def test_missing_payload_and_unsafe_zip_member_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing"
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop._single_payload(missing, "vokra-cli.exe")
            archive_path = root / "unsafe.zip"
            info = zipfile.ZipInfo("../vokra-cli.exe", date_time=(1980, 1, 1, 0, 0, 0))
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(info, pe())
            with self.assertRaises(desktop.DesktopReleaseError):
                desktop.verify_windows_cli_zip(archive_path)

    def test_version_is_validated_before_path_construction(self) -> None:
        with self.assertRaises(desktop.DesktopReleaseError):
            desktop._asset_names("../escape")

    def test_sbom_schema_is_required(self) -> None:
        for replacement in ("{}\n", sbom("wrong-root", "vokra-desktop-cli")):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                args, out = make_valid_dist(Path(tmp))
                (out / "vokra-desktop-cli.spdx.json").write_text(replacement, encoding="utf-8")
                with self.assertRaises(desktop.DesktopReleaseError):
                    desktop.write_manifest(args)

    def test_cli_assembly_accepts_hyphenated_options_and_real_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in {
                "capi-linux": elf(),
                "capi-macos": macho(),
                "capi-windows": pe(),
                "cli-linux": elf(interp=True),
                "cli-linux-musl": elf(),
                "cli-macos": macho(),
                "cli-windows": pe(),
            }.items():
                (root / name).write_bytes(data)
            out = root / "dist"
            real_header = HERE.parents[1] / "include" / "vokra.h"
            argv = [
                "assemble", "--version", "1.2.3", "--out-dir", str(out),
                "--header", str(real_header),
                "--capi-linux", str(root / "capi-linux"),
                "--capi-macos", str(root / "capi-macos"),
                "--capi-windows", str(root / "capi-windows"),
                "--cli-linux", str(root / "cli-linux"),
                "--cli-linux-musl", str(root / "cli-linux-musl"),
                "--cli-macos", str(root / "cli-macos"),
                "--cli-windows", str(root / "cli-windows"),
            ]
            self.assertEqual(desktop.main(argv), 0)
            self.assertEqual((out / "vokra.h").read_bytes(), real_header.read_bytes())

    def test_windows_zip_has_exact_safe_member_and_stable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vokra-cli.exe"
            source.write_bytes(pe())
            one = root / "one.zip"
            two = root / "two.zip"
            desktop._write_windows_zip(source, one)
            desktop._write_windows_zip(source, two)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            desktop.verify_windows_cli_zip(one)
            with zipfile.ZipFile(one) as archive:
                self.assertEqual(archive.namelist(), ["vokra-cli.exe"])


if __name__ == "__main__":
    unittest.main()
