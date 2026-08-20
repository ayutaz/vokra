"""Unit tests for native-wheel tag, architecture, and manifest checks."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "wheel_tools.py"
_SPEC = importlib.util.spec_from_file_location("vokra_wheel_tools", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
wheel_tools = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wheel_tools)


def _record_hash(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


def _binary(target: str) -> bytes:
    if target == "linux-x86_64":
        data = bytearray(64)
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", data, 18, 62)
        return bytes(data)
    if target.startswith("macos-"):
        cpu = 0x0100000C if target == "macos-arm64" else 0x01000007
        return b"\xcf\xfa\xed\xfe" + struct.pack("<I", cpu) + bytes(56)
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, 0x8664)
    return bytes(data)


def _platform(target: str) -> str:
    return {
        "linux-x86_64": "manylinux_2_28_x86_64",
        "macos-arm64": "macosx_11_0_arm64",
        "macos-x86_64": "macosx_11_0_x86_64",
        "windows-x86_64": "win_amd64",
    }[target]


def _write_wheel(
    directory: Path,
    target: str,
    version: str = "1.2.3",
    *,
    binary_target: str | None = None,
) -> Path:
    platform = _platform(target)
    filename = directory / f"vokra-{version}-py3-none-{platform}.whl"
    dist_info = f"vokra-{version}.dist-info"
    native = wheel_tools.TARGETS[target]["native"]
    members = {
        "vokra/__init__.py": b"__version__ = 'fixture'\n",
        native: _binary(binary_target or target),
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.4\nName: vokra\nVersion: {version}\n"
            "Requires-Python: <3.13,>=3.9\n\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
            f"Tag: py3-none-{platform}\n\n"
        ).encode(),
    }
    record_path = f"{dist_info}/RECORD"
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for name, data in members.items():
        writer.writerow((name, _record_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    members[record_path] = stream.getvalue().encode()

    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return filename


@pytest.mark.parametrize("target", sorted(wheel_tools.TARGETS))
def test_verify_wheel_accepts_truthful_target(tmp_path: Path, target: str) -> None:
    wheel = _write_wheel(tmp_path, target)
    result = wheel_tools.verify_wheel(wheel, target, "1.2.3")
    assert result["target"] == target
    assert result["native_arch"] == wheel_tools.TARGETS[target]["arch"]
    assert len(result["sha256"]) == 64


def test_verify_wheel_rejects_false_architecture(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "macos-x86_64", binary_target="macos-arm64")
    with pytest.raises(SystemExit, match="architecture"):
        wheel_tools.verify_wheel(wheel, "macos-x86_64", "1.2.3")


def test_manifest_requires_all_four_targets(tmp_path: Path) -> None:
    for target in wheel_tools.TARGETS:
        _write_wheel(tmp_path, target)
    manifest = wheel_tools.build_manifest(tmp_path, "1.2.3")
    assert manifest["wheel_count"] == 4
    assert {row["target"] for row in manifest["wheels"]} == set(wheel_tools.TARGETS)
    json.dumps(manifest)

    next(wheel for wheel in tmp_path.glob("*.whl") if "win_amd64" in wheel.name).unlink()
    with pytest.raises(SystemExit, match="expected 4 wheels"):
        wheel_tools.build_manifest(tmp_path, "1.2.3")
