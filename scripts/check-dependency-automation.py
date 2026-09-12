#!/usr/bin/env -S uv run --script
"""Keep Dependabot directory coverage aligned with tracked lockfiles."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


LOCKLESS_UV = {
    "tools/parity/audiogen_medium_reference",
    "tools/parity/audioldm2_reference",
    "tools/parity/cosyvoice2_reference",
    "tools/parity/cosyvoice3_reference",
    "tools/parity/firered_asr_llm_l",
    "tools/parity/higgs_audio_v3_tts_4b",
}


def section_directories(text: str, ecosystem: str) -> set[str]:
    lines = text.splitlines()
    in_section = False
    in_directories = False
    result: set[str] = set()
    for line in lines:
        if line.startswith("  - package-ecosystem:"):
            in_section = f'"{ecosystem}"' in line
            in_directories = False
            continue
        if in_section and line.startswith("    directories:"):
            in_directories = True
            continue
        if in_section and in_directories and line.startswith("      - "):
            value = line.strip()[2:].strip().strip('"\'')
            result.add(value.lstrip("/") or ".")
        elif in_section and in_directories:
            in_directories = False
        if in_section and line.startswith("  - "):
            break
    return result


def tracked_dirs(root: Path, filename: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", filename, f"**/{filename}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return {str(Path(path).parent) for path in result.stdout.splitlines() if path}


def validate(text: str, root: Path | None = None) -> list[str]:
    root = root or Path(__file__).resolve().parents[1]
    expected_uv = tracked_dirs(root, "uv.lock")
    expected_cargo = tracked_dirs(root, "Cargo.lock")
    expected_cargo.discard(".")
    uv = section_directories(text, "uv")
    cargo = section_directories(text, "cargo")
    errors: list[str] = []
    if uv != expected_uv:
        errors.append("uv Dependabot directories differ: missing=" + ",".join(sorted(expected_uv - uv)) + "; extra=" + ",".join(sorted(uv - expected_uv)))
    if cargo != expected_cargo:
        errors.append("Cargo Dependabot directories differ: missing=" + ",".join(sorted(expected_cargo - cargo)) + "; extra=" + ",".join(sorted(cargo - expected_cargo)))
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", "pyproject.toml", "*/pyproject.toml"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    pyprojects = {str(Path(path).parent) for path in result.stdout.splitlines() if path}
    lockless = pyprojects - expected_uv
    if lockless != LOCKLESS_UV:
        errors.append("lockless uv trees changed; update the deliberate allowlist")
    if uv & lockless:
        errors.append("lockless uv trees must not be enabled in Dependabot: " + ",".join(sorted(uv & lockless)))
    if "Cargo.lock" in cargo or "." in cargo:
        errors.append("root Cargo.lock must remain excluded from Dependabot")
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="vokra-dependabot-") as directory:
        root = Path(directory)
        (root / "a").mkdir()
        (root / "uv.lock").write_text("", encoding="utf-8")
        (root / "a/uv.lock").write_text("", encoding="utf-8")
        (root / "Cargo.lock").write_text("", encoding="utf-8")
        (root / "b").mkdir()
        (root / "b/Cargo.lock").write_text("", encoding="utf-8")
        for relative in LOCKLESS_UV:
            path = root / relative
            path.mkdir(parents=True, exist_ok=True)
            (path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "add",
                "uv.lock",
                "a/uv.lock",
                "Cargo.lock",
                "b/Cargo.lock",
                *[f"{path}/pyproject.toml" for path in LOCKLESS_UV],
            ],
            check=True,
        )
        good = '  - package-ecosystem: "cargo"\n    directories:\n      - "/b"\n  - package-ecosystem: "uv"\n    directories:\n      - "/"\n      - "/a"\n'
        assert not validate(good, root), "complete fixture must pass"
        bad = good.replace('      - "/a"', '      - "/missing"')
        assert validate(bad, root), "orphaned/missing lock coverage must fail"
        bad_root_cargo = good.replace('  - package-ecosystem: "uv"', '      - "/"\n  - package-ecosystem: "uv"')
        assert validate(bad_root_cargo, root), "root Cargo.lock must remain excluded"
    print("check-dependency-automation --self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    root = Path(__file__).resolve().parents[1]
    errors = validate((root / ".github/dependabot.yml").read_text(encoding="utf-8"), root)
    if errors:
        print("\n".join(f"check-dependency-automation: {item}" for item in errors))
        return 1
    print("Dependabot lock coverage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
