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


ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"
EXPECTED_OVERRIDE = "transformers ; python_version < '0'"
EXPECTED_LOCK_OVERRIDE = {
    "name": "transformers",
    "marker": "python_full_version < '0'",
}
FORBIDDEN_LOCK_ROWS = ("transformers", "tokenizers", "typer", "shellingham")


def _parse_documents(pyproject_text: str, lock_text: str) -> tuple[dict, dict]:
    """Parse both TOML documents before inspecting any dependency fields."""

    return tomllib.loads(pyproject_text), tomllib.loads(lock_text)


def _validate_documents(pyproject: dict, lock: dict) -> None:
    tool_uv = pyproject.get("tool", {}).get("uv", {})
    if tool_uv.get("override-dependencies") != [EXPECTED_OVERRIDE]:
        raise AssertionError("pyproject lost the impossible Transformers override")
    project_dependencies = set(pyproject.get("project", {}).get("dependencies", []))
    for declaration in ('transformers==5.10.4', 'xcodec2==0.1.5'):
        if declaration not in project_dependencies:
            raise AssertionError(f"missing audited declaration: {declaration}")

    manifest = lock.get("manifest", {})
    if manifest.get("overrides") != [EXPECTED_LOCK_OVERRIDE]:
        raise AssertionError("uv.lock lost the impossible Transformers override")

    rows = {package["name"] for package in lock.get("package", [])}
    unexpected = rows.intersection(FORBIDDEN_LOCK_ROWS)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise AssertionError(f"forbidden dependency lock row(s) reintroduced: {names}")


def _validate_files(pyproject_text: str, lock_text: str) -> None:
    pyproject, lock = _parse_documents(pyproject_text, lock_text)
    _validate_documents(pyproject, lock)


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
    from dump_reference import import_official_decoder

    decoder = import_official_decoder()
    if decoder.__module__ != "xcodec2.vq.codec_decoder_vocos":
        raise AssertionError(f"unexpected decoder module: {decoder.__module__}")
    if decoder.__name__ != "CodecDecoderVocos":
        raise AssertionError(f"unexpected decoder API: {decoder.__name__}")


def _tamper_self_test(pyproject_text: str, lock_text: str) -> None:
    pyproject, lock = _parse_documents(pyproject_text, lock_text)
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
            try:
                _validate_documents(pyproject, tampered_lock)
            except AssertionError:
                continue
            raise AssertionError(f"tamper self-test accepted reintroduced {name}")

    tampered_pyproject = copy.deepcopy(pyproject)
    tampered_pyproject["tool"]["uv"]["override-dependencies"] = []
    try:
        _validate_documents(tampered_pyproject, lock)
    except AssertionError:
        pass
    else:
        raise AssertionError("tamper self-test accepted a weakened pyproject override")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also exercise fail-closed tamper cases in memory",
    )
    args = parser.parse_args()

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    lock_text = LOCKFILE.read_text(encoding="utf-8")
    _validate_files(pyproject_text, lock_text)
    if args.self_test:
        _tamper_self_test(pyproject_text, lock_text)
    _validate_installed_environment()
    _validate_official_decoder()
    print("xcodec2 dependency guard: PASS (decoder-only, model-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
