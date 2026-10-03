#!/usr/bin/env python3
"""Fail-closed, model-free guard for the official XCodec2 decoder oracle.

This guard checks the dependency declaration and lock boundary before importing
the official decoder.  It never opens a GGUF or executes a model forward.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import sys
import tomllib
from pathlib import Path

import dependency_audit as audit


ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"
EXPECTED_OVERRIDES = [
    "torch==2.13.0",
    "torchaudio==2.11.0",
    "transformers ; python_version < '0'",
]
EXPECTED_LOCK_OVERRIDES = [
    {
        "name": "torch",
        "specifier": "==2.13.0",
        "index": "https://download.pytorch.org/whl/cpu",
    },
    {
        "name": "torchaudio",
        "specifier": "==2.11.0",
        "index": "https://download.pytorch.org/whl/cpu",
    },
    {
        "name": "transformers",
        "marker": "python_full_version < '0'",
    },
]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
EXPECTED_RESOLVED_VERSIONS = {
    "torch": {"2.13.0", "2.13.0+cpu"},
    "torchaudio": {"2.11.0", "2.11.0+cpu"},
}
MACOS_MARKER = "sys_platform == 'darwin'"
LINUX_MARKER = (
    "sys_platform != 'darwin' and sys_platform != 'emscripten' and "
    "sys_platform != 'win32'"
)
EXPECTED_PLATFORM_WHEELS = {
    "torch": {
        "2.13.0": ("macosx_14_0_arm64",),
        "2.13.0+cpu": ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64"),
    },
    "torchaudio": {
        "2.11.0": ("macosx_11_0_arm64",),
        "2.11.0+cpu": ("manylinux_2_28_x86_64", "manylinux_2_28_aarch64"),
    },
}
FORBIDDEN_LOCK_ROWS = ("transformers", "tokenizers", "typer", "shellingham")


def _parse_documents(pyproject_text: str, lock_text: str) -> tuple[dict, dict]:
    """Parse both TOML documents before inspecting any dependency fields."""

    return tomllib.loads(pyproject_text), tomllib.loads(lock_text)


def _validate_documents(pyproject: dict, lock: dict) -> None:
    tool_uv = pyproject.get("tool", {}).get("uv", {})
    if tool_uv.get("override-dependencies") != EXPECTED_OVERRIDES:
        raise AssertionError("pyproject lost the audited dependency overrides")
    indexes = tool_uv.get("index")
    if indexes != [
        {"name": "pytorch-cpu", "url": PYTORCH_CPU_INDEX, "explicit": True}
    ]:
        raise AssertionError("pyproject lost the explicit PyTorch CPU index")
    if tool_uv.get("sources") != {
        "torch": {"index": "pytorch-cpu"},
        "torchaudio": {"index": "pytorch-cpu"},
    }:
        raise AssertionError("pyproject lost the Torch/TorchAudio CPU sources")
    project_dependencies = set(pyproject.get("project", {}).get("dependencies", []))
    for declaration in (
        "torch==2.13.0",
        "torchaudio==2.11.0",
        "transformers==5.10.4",
        "xcodec2==0.1.5",
    ):
        if declaration not in project_dependencies:
            raise AssertionError(f"missing audited declaration: {declaration}")

    manifest = lock.get("manifest", {})
    if manifest.get("overrides") != EXPECTED_LOCK_OVERRIDES:
        raise AssertionError("uv.lock lost the audited dependency overrides")

    packages = lock.get("package", [])
    rows = {package["name"] for package in packages}
    unexpected = rows.intersection(FORBIDDEN_LOCK_ROWS)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise AssertionError(f"forbidden dependency lock row(s) reintroduced: {names}")

    for name, expected_versions in EXPECTED_RESOLVED_VERSIONS.items():
        resolved = [package for package in packages if package.get("name") == name]
        actual_versions = {package.get("version") for package in resolved}
        if actual_versions != expected_versions:
            raise AssertionError(
                f"{name} resolved versions {sorted(actual_versions)!r} != "
                f"{sorted(expected_versions)!r}"
            )
        if any(
            package.get("source", {}).get("registry") != PYTORCH_CPU_INDEX
            for package in resolved
        ):
            raise AssertionError(
                f"{name} is not fully sourced from the PyTorch CPU index"
            )
        for package in resolved:
            version = package["version"]
            markers = set(package.get("resolution-markers", []))
            if version.endswith("+cpu"):
                if LINUX_MARKER not in markers:
                    raise AssertionError(
                        f"{name} {version} lacks the Linux CPU resolution marker"
                    )
            elif MACOS_MARKER not in markers:
                raise AssertionError(
                    f"{name} {version} lacks the macOS resolution marker"
                )
            wheel_urls = {
                wheel["url"] for wheel in package.get("wheels", [])
            }
            for platform_token in EXPECTED_PLATFORM_WHEELS[name][version]:
                if not any(platform_token in url for url in wheel_urls):
                    raise AssertionError(
                        f"{name} {version} lacks an audited {platform_token} wheel"
                    )

    xcodec2 = [package for package in packages if package.get("name") == "xcodec2"]
    if len(xcodec2) != 1:
        raise AssertionError("uv.lock must contain exactly one xcodec2 package row")
    xcodec2_dependencies = xcodec2[0].get("dependencies", [])
    for name, expected_versions in EXPECTED_RESOLVED_VERSIONS.items():
        dependencies = [
            dependency
            for dependency in xcodec2_dependencies
            if dependency.get("name") == name
        ]
        if {
            dependency.get("version") for dependency in dependencies
        } != expected_versions:
            raise AssertionError(
                f"xcodec2 dependency edge for {name} does not follow the audited pair"
            )


def _validate_files(pyproject_text: str, lock_text: str) -> None:
    pyproject, lock = _parse_documents(pyproject_text, lock_text)
    _validate_documents(pyproject, lock)
    manifest = audit.strict_json(audit.MANIFEST)
    rows = audit.strict_json(audit.ROWS)
    expected = audit.source_digest_map()
    if manifest.get("project", {}).get("source_digests") != expected:
        raise AssertionError("license manifest helper source digest binding drifted")
    if rows.get("source_digests") != expected:
        raise AssertionError("dependency rows helper source digest binding drifted")


def _validate_installed_environment() -> None:
    installed = {
        distribution.metadata["Name"].lower()
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    unexpected = installed.intersection(FORBIDDEN_LOCK_ROWS)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise AssertionError(f"forbidden dependency installed: {names}")


def _validate_official_decoder() -> None:
    # The decoder module must remain usable without Transformers or its native
    # tokenizer/shellingham branch.  dump_reference performs the official
    # package/source hash checks before returning this class.
    for name in FORBIDDEN_LOCK_ROWS:
        sys.modules[name] = None
    from dump_reference import (
        import_official_decoder,
        validate_execution_authorization,
        validate_preimport_runtime,
    )

    validate_execution_authorization()
    decoder = import_official_decoder(validate_preimport_runtime())
    if decoder.__module__ != "xcodec2.vq.codec_decoder_vocos":
        raise AssertionError(f"unexpected decoder module: {decoder.__module__}")
    if decoder.__name__ != "CodecDecoderVocos":
        raise AssertionError(f"unexpected decoder API: {decoder.__name__}")


def _tamper_self_test(pyproject_text: str, lock_text: str) -> None:
    pyproject, lock = _parse_documents(pyproject_text, lock_text)

    def assert_rejected(
        label: str, tampered_pyproject: dict, tampered_lock: dict
    ) -> None:
        try:
            _validate_documents(tampered_pyproject, tampered_lock)
        except AssertionError:
            return
        raise AssertionError(f"tamper self-test accepted {label}")

    # Exercise the fail-closed branch in memory: a future lock regeneration
    # that adds any excluded package must make this guard fail.
    for name in FORBIDDEN_LOCK_ROWS:
        for quote in ('"', "'"):
            # Parsing this text specifically covers both TOML quote styles;
            # validation below then inspects the parsed package table.
            tampered_row = tomllib.loads(
                f"[[package]]\nname = {quote}{name}{quote}\n"
            )["package"][0]
            tampered_lock = copy.deepcopy(lock)
            tampered_lock.setdefault("package", []).append(tampered_row)
            assert_rejected(
                f"reintroduced {name} lock row",
                copy.deepcopy(pyproject),
                tampered_lock,
            )

    for index, label in ((0, "Torch"), (1, "TorchAudio")):
        tampered_pyproject = copy.deepcopy(pyproject)
        tampered_pyproject["tool"]["uv"]["override-dependencies"].pop(index)
        assert_rejected(
            f"missing {label} override", tampered_pyproject, copy.deepcopy(lock)
        )

    for index, replacement, label in (
        (0, "torch==2.5.0", "Torch"),
        (1, "torchaudio==2.5.0", "TorchAudio"),
    ):
        tampered_pyproject = copy.deepcopy(pyproject)
        tampered_pyproject["tool"]["uv"]["override-dependencies"][index] = replacement
        assert_rejected(
            f"weakened {label} override", tampered_pyproject, copy.deepcopy(lock)
        )

    tampered_pyproject = copy.deepcopy(pyproject)
    tampered_pyproject["tool"]["uv"]["index"][0]["url"] = "https://pypi.org/simple"
    assert_rejected("non-PyTorch index", tampered_pyproject, copy.deepcopy(lock))

    tampered_pyproject = copy.deepcopy(pyproject)
    tampered_pyproject["tool"]["uv"]["index"][0]["explicit"] = False
    assert_rejected("non-explicit PyTorch index", tampered_pyproject, copy.deepcopy(lock))

    for name in ("torch", "torchaudio"):
        tampered_pyproject = copy.deepcopy(pyproject)
        tampered_pyproject["tool"]["uv"]["sources"][name]["index"] = "pypi"
        assert_rejected(
            f"{name} source escape", tampered_pyproject, copy.deepcopy(lock)
        )

    tampered_pyproject = copy.deepcopy(pyproject)
    tampered_pyproject["tool"]["uv"]["override-dependencies"] = []
    assert_rejected(
        "weakened dependency override set", tampered_pyproject, copy.deepcopy(lock)
    )

    for index, label in ((0, "Torch"), (1, "TorchAudio")):
        tampered_lock = copy.deepcopy(lock)
        tampered_lock["manifest"]["overrides"].pop(index)
        assert_rejected(
            f"missing lock {label} override", copy.deepcopy(pyproject), tampered_lock
        )

    for index, replacement, label in (
        (0, "==2.5.0", "Torch"),
        (1, "==2.5.0", "TorchAudio"),
    ):
        tampered_lock = copy.deepcopy(lock)
        tampered_lock["manifest"]["overrides"][index]["specifier"] = replacement
        assert_rejected(
            f"weakened lock {label} override", copy.deepcopy(pyproject), tampered_lock
        )

    for index, label in ((0, "Torch"), (1, "TorchAudio")):
        tampered_lock = copy.deepcopy(lock)
        tampered_lock["manifest"]["overrides"][index]["index"] = (
            "https://pypi.org/simple"
        )
        assert_rejected(
            f"lock {label} source escape", copy.deepcopy(pyproject), tampered_lock
        )

    for name, versions in EXPECTED_RESOLVED_VERSIONS.items():
        for version in versions:
            tampered_lock = copy.deepcopy(lock)
            tampered_lock["package"] = [
                package
                for package in tampered_lock["package"]
                if not (
                    package.get("name") == name
                    and package.get("version") == version
                )
            ]
            assert_rejected(
                f"missing {name} {version} platform row",
                copy.deepcopy(pyproject),
                tampered_lock,
            )

    for name in EXPECTED_RESOLVED_VERSIONS:
        tampered_lock = copy.deepcopy(lock)
        package = next(
            package
            for package in tampered_lock["package"]
            if package.get("name") == name
            and package.get("version", "").endswith("+cpu")
        )
        package["source"]["registry"] = "https://pypi.org/simple"
        assert_rejected(
            f"{name} CPU package source escape",
            copy.deepcopy(pyproject),
            tampered_lock,
        )

    for name in EXPECTED_RESOLVED_VERSIONS:
        tampered_lock = copy.deepcopy(lock)
        package = next(
            package
            for package in tampered_lock["package"]
            if package.get("name") == "xcodec2"
        )
        dependency = next(
            dependency
            for dependency in package["dependencies"]
            if dependency.get("name") == name
        )
        dependency.pop("version", None)
        assert_rejected(
            f"weakened xcodec2 {name} dependency edge",
            copy.deepcopy(pyproject),
            tampered_lock,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also exercise fail-closed tamper cases in memory",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="validate the lock and tamper cases without importing packages",
    )
    args = parser.parse_args()

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    lock_text = LOCKFILE.read_text(encoding="utf-8")
    _validate_files(pyproject_text, lock_text)
    if args.self_test:
        _tamper_self_test(pyproject_text, lock_text)
    if args.documents_only:
        print("xcodec2 dependency guard: PASS (documents-only)")
        return 0
    _validate_installed_environment()
    _validate_official_decoder()
    print("xcodec2 dependency guard: PASS (decoder-only, model-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
