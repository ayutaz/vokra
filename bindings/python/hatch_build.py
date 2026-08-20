"""Hatchling hooks for Vokra's platform-specific ``ctypes`` wheel.

The Python layer is ABI-independent, but every wheel contains one native
``vokra-capi`` shared library.  The wheel therefore uses ``py3-none`` for the
Python/ABI fields and an explicit platform field supplied by the native build
job.  Guessing a CPython ABI tag or relabelling a host-built library is unsafe.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

_DEV_VERSION = "0.1.0.dev0"
_TAG_RE = re.compile(
    r"^py3-none-(?:"
    r"linux_(?:x86_64|aarch64)|"
    r"macosx_11_0_(?:arm64|x86_64)|"
    r"win_amd64"
    r")$"
)


def _host_platform_tag() -> str:
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    if sys.platform.startswith("linux") and machine in {"x86_64", "aarch64"}:
        return f"py3-none-linux_{machine}"
    if sys.platform == "darwin" and machine in {"arm64", "x86_64"}:
        return f"py3-none-macosx_11_0_{machine}"
    if sys.platform == "win32" and machine == "x86_64":
        return "py3-none-win_amd64"
    raise RuntimeError(
        "unsupported Vokra wheel build host: "
        f"sys.platform={sys.platform!r}, machine={platform.machine()!r}"
    )


def _native_library_name() -> str:
    if sys.platform.startswith("linux"):
        return "libvokra.so"
    if sys.platform == "darwin":
        return "libvokra.dylib"
    if sys.platform == "win32":
        return "vokra.dll"
    raise RuntimeError(f"unsupported Vokra wheel build OS: {sys.platform!r}")


class VokraMetadataHook(MetadataHookInterface):
    """Set package metadata from the release job without editing sources."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        metadata["version"] = os.environ.get("VOKRA_BUILD_VERSION", _DEV_VERSION)


class VokraWheelHook(BuildHookInterface):
    """Require one native library and emit a truthful platform wheel tag."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        native_name = _native_library_name()
        native_path = Path(self.root, "src", "vokra", "_lib", native_name)
        if not native_path.is_file() or native_path.stat().st_size == 0:
            raise RuntimeError(
                "Vokra wheel build requires a non-empty native library at "
                f"{native_path}; build vokra-capi for this host first"
            )

        tag = os.environ.get("VOKRA_WHEEL_TAG", _host_platform_tag())
        if not _TAG_RE.fullmatch(tag):
            raise RuntimeError(
                f"unsafe or unsupported VOKRA_WHEEL_TAG={tag!r}; expected an "
                "explicit py3-none platform tag for the current native build"
            )
        if tag != _host_platform_tag():
            raise RuntimeError(
                f"VOKRA_WHEEL_TAG={tag!r} does not match native build host "
                f"{_host_platform_tag()!r}; cross-retagging is forbidden"
            )

        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = tag


def get_metadata_hook() -> type[VokraMetadataHook]:
    return VokraMetadataHook


def get_build_hook() -> type[VokraWheelHook]:
    return VokraWheelHook
