#!/usr/bin/env python3
"""Fail-closed dependency guard for the NanoCodec NeMo oracle.

This guard is deliberately model-free.  It checks the committed isolated
project/lock contract and, when given an exact NVIDIA-NeMo/Speech checkout,
checks the static TTS source boundary that justifies omitting only the unused
G2P/NLTK dependency branch while retaining the required SpeechLM2 -> PEFT ->
Accelerate import closure.  It never imports NeMo or opens a checkpoint.
"""

from __future__ import annotations

import argparse
import ast
from collections import deque
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Callable


PROJECT_NAME = "vokra-nanocodec-parity"
NEMO_COMMIT = "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef"
FORBIDDEN_ROWS = frozenset({"nltk"})
REQUIRED_ROWS = frozenset({"accelerate", "peft"})
IMPOSSIBLE_MARKER = "python_full_version < '0'"
IMPORT_RE = re.compile(r"\b(?:from|import)\s+nltk\b")


def fail(message: str) -> None:
    raise RuntimeError(f"nanocodec dependency guard: {message}")


def _name(row: object) -> str | None:
    return row.get("name") if isinstance(row, dict) and isinstance(row.get("name"), str) else None


def verify_project(project: Path) -> None:
    pyproject_path = project / "pyproject.toml"
    lock_path = project / "uv.lock"
    if not pyproject_path.is_file() or not lock_path.is_file():
        fail("pyproject.toml and uv.lock must both be regular files")

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_table = pyproject.get("project")
    if not isinstance(project_table, dict) or project_table.get("name") != PROJECT_NAME:
        fail("unexpected project identity")

    dependencies = project_table.get("dependencies")
    if dependencies != [
        "nemo_toolkit[tts] @ git+https://github.com/NVIDIA-NeMo/Speech.git@" + NEMO_COMMIT,
        "peft==0.20.0",
    ]:
        fail("the project must declare only the pinned NeMo source and explicit PEFT edge")

    uv = pyproject.get("tool", {}).get("uv", {})
    overrides = uv.get("override-dependencies") if isinstance(uv, dict) else None
    if not isinstance(overrides, list) or "nltk ; python_version < '0'" not in overrides:
        fail("pyproject must make NLTK unreachable with the impossible marker")

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    manifest = lock.get("manifest")
    if not isinstance(manifest, dict):
        fail("uv.lock manifest is missing")
    lock_overrides = manifest.get("overrides")
    if not isinstance(lock_overrides, list) or not any(
        isinstance(row, dict)
        and row.get("name") == "nltk"
        and row.get("marker") == IMPOSSIBLE_MARKER
        for row in lock_overrides
    ):
        fail("uv.lock does not retain the impossible NLTK override")

    package_rows = lock.get("package")
    if not isinstance(package_rows, list) or not package_rows:
        fail("uv.lock package table is missing")
    identities = {_name(row) for row in package_rows}
    present = sorted(FORBIDDEN_ROWS & identities)
    if present:
        fail("forbidden unused package rows remain: " + ", ".join(present))
    missing = sorted(REQUIRED_ROWS - identities)
    if missing:
        fail("required import-closure package rows are missing: " + ", ".join(missing))

    virtual = [row for row in package_rows if _name(row) == PROJECT_NAME]
    if len(virtual) != 1:
        fail("uv.lock must contain exactly one virtual project row")
    virtual_deps = virtual[0].get("dependencies", [])
    if not isinstance(virtual_deps, list) or [
        _name(row) for row in virtual_deps
    ] != ["nemo-toolkit", "peft"]:
        fail("virtual project dependency closure is missing explicit PEFT")

    nemo_rows = [row for row in package_rows if _name(row) == "nemo-toolkit"]
    if len(nemo_rows) != 1:
        fail("uv.lock must contain exactly one nemo-toolkit row")
    source = nemo_rows[0].get("source", {})
    source_text = str(source.get("git", "")) if isinstance(source, dict) else ""
    if NEMO_COMMIT not in source_text:
        fail("nemo-toolkit lock source is not pinned to the audited commit")
    peft_rows = [row for row in package_rows if _name(row) == "peft"]
    if len(peft_rows) != 1 or not any(
        _name(item) == "accelerate" for item in peft_rows[0].get("dependencies", [])
    ):
        fail("PEFT lock row does not retain its required Accelerate edge")

    for row in package_rows:
        if not isinstance(row, dict):
            fail("uv.lock contains a malformed package row")
        for field in ("dependencies", "optional-dependencies"):
            values = row.get(field, [])
            if isinstance(values, dict):
                values = [item for group in values.values() for item in group]
            if not isinstance(values, list):
                continue
            names = {_name(item) for item in values}
            leaked = sorted(FORBIDDEN_ROWS & names)
            if leaked:
                fail(f"{row.get('name')} retains forbidden dependency edges: {', '.join(leaked)}")


def _source_root(source_root: Path) -> Path:
    if (source_root / "nemo").is_dir():
        return source_root / "nemo"
    if source_root.name == "nemo" and source_root.is_dir():
        return source_root
    fail("--nemo-source-root must point to an NVIDIA-NeMo/Speech checkout")


def _git_root(source_root: Path) -> Path:
    candidate = source_root if (source_root / ".git").exists() else source_root.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", str(candidate), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        fail(f"cannot inspect the NeMo checkout with git: {exc}")
    if head.returncode != 0 or head.stdout.strip() != NEMO_COMMIT:
        fail(f"NeMo source HEAD is not the pinned commit {NEMO_COMMIT}")
    if status.returncode != 0 or status.stdout:
        fail("NeMo source checkout is not clean")
    return candidate


def _module_name(path: Path, nemo: Path) -> str:
    relative = path.relative_to(nemo).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(("nemo", *parts))


def _relative_import_targets(module: str, node: ast.ImportFrom) -> list[str]:
    package = module.split(".")[:-1]
    if node.level:
        package = package[: len(package) - node.level + 1]
    prefix = [*package, *(node.module.split(".") if node.module else [])]
    if node.module:
        base = ".".join(prefix)
        return [base, *(f"{base}.{alias.name}" for alias in node.names)]
    return [".".join([*prefix, alias.name]) for alias in node.names]


def _source_import_targets(module: str, tree: ast.AST) -> list[str]:
    targets: list[str] = []
    class ImportCollector(ast.NodeVisitor):
        """Collect imports executed while a module is being imported.

        Imports nested inside a function are intentionally excluded: NeMo's
        ``safe_instantiate`` helper has a guarded, function-local G2P import
        for unrelated text models.  It is not part of AudioCodecModel's
        import-time closure and would be a false positive here.
        """

        def __init__(self) -> None:
            self.function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_depth += 1
            self.generic_visit(node)
            self.function_depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Import(self, node: ast.Import) -> None:
            if self.function_depth == 0:
                targets.extend(alias.name for alias in node.names)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if self.function_depth > 0:
                return
            if node.level:
                targets.extend(_relative_import_targets(module, node))
            elif node.module:
                targets.append(node.module)
                targets.extend(f"{node.module}.{alias.name}" for alias in node.names)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                self.function_depth == 0
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                targets.append(node.args[0].value)
            self.generic_visit(node)

    ImportCollector().visit(tree)
    return targets


def _verify_import_boundary(nemo: Path) -> None:
    modules = {
        _module_name(path, nemo): path
        for path in nemo.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    seeds = (
        "nemo",
        "nemo.collections.tts",
        "nemo.collections.tts.models",
        "nemo.collections.tts.models.audio_codec",
    )
    queue: deque[tuple[str, tuple[str, ...]]] = deque(
        (seed, (seed,)) for seed in seeds if seed in modules
    )
    visited: set[str] = set()
    while queue:
        module, chain = queue.popleft()
        if module in visited:
            continue
        visited.add(module)
        path = modules[module]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            fail(f"cannot parse pinned source module {path}: {exc}")
        for target in _source_import_targets(module, tree):
            if target.startswith("nemo.collections.tts.g2p"):
                fail(
                    "AudioCodecModel import graph reaches G2P: "
                    + " -> ".join((*chain, target))
                )
            if target == "nltk":
                fail(
                    "AudioCodecModel import graph reaches a forbidden package: "
                    + " -> ".join((*chain, target))
                )
            if target in modules:
                queue.append((target, (*chain, target)))


def verify_source(source_root: Path) -> None:
    nemo = _source_root(source_root)
    _git_root(source_root)
    tts = nemo / "collections" / "tts"
    models = tts / "models"
    if not tts.is_dir() or not models.is_dir():
        fail("pinned source checkout has no nemo/collections/tts tree")

    # NLTK is used by the upstream English G2P implementation only.  The
    # AudioCodecModel import path enters tts.models and never imports tts.g2p.
    for path in sorted(tts.rglob("*.py")):
        relative = path.relative_to(tts)
        text = path.read_text(encoding="utf-8")
        if "g2p" not in relative.parts and IMPORT_RE.search(text):
            fail(f"forbidden NLTK import outside the excluded G2P subtree: {relative}")
    for path in sorted(models.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if IMPORT_RE.search(text):
            fail(f"AudioCodecModel import boundary imports a forbidden package: {path.relative_to(nemo)}")
    _verify_import_boundary(nemo)


def _expect_project_failure(project: Path, label: str) -> None:
    try:
        verify_project(project)
    except RuntimeError:
        return
    fail(f"tamper test unexpectedly passed: {label}")


def _tamper_test(project: Path, label: str, mutate: Callable[[Path], None]) -> None:
    with tempfile.TemporaryDirectory(prefix="nanocodec-guard-") as temporary:
        candidate = Path(temporary)
        shutil.copy2(project / "pyproject.toml", candidate / "pyproject.toml")
        shutil.copy2(project / "uv.lock", candidate / "uv.lock")
        mutate(candidate)
        _expect_project_failure(candidate, label)


def self_test(project: Path) -> None:
    verify_project(project)

    def reintroduce_nltk(candidate: Path) -> None:
        with (candidate / "uv.lock").open("a", encoding="utf-8") as stream:
            stream.write('\n[[package]]\nname = "nltk"\nversion = "3.10.0"\n')

    def remove_peft_accelerate(candidate: Path) -> None:
        lock_path = candidate / "uv.lock"
        text = lock_path.read_text(encoding="utf-8")
        start = text.index('[[package]]\nname = "peft"')
        end = text.find("\n[[package]]", start + 1)
        if end < 0:
            end = len(text)
        section = text[start:end]
        edge = '    { name = "accelerate" },\n'
        if edge not in section:
            fail("tamper fixture could not locate PEFT -> Accelerate edge")
        lock_path.write_text(text[:start] + section.replace(edge, "", 1) + text[end:], encoding="utf-8")

    def drift_nemo_commit(candidate: Path) -> None:
        pyproject_path = candidate / "pyproject.toml"
        text = pyproject_path.read_text(encoding="utf-8")
        pyproject_path.write_text(text.replace(NEMO_COMMIT, "0" * 40, 1), encoding="utf-8")

    _tamper_test(project, "NLTK lock row reintroduction", reintroduce_nltk)
    _tamper_test(project, "PEFT -> Accelerate edge removal", remove_peft_accelerate)
    _tamper_test(project, "NeMo commit drift", drift_nemo_commit)
    print("nanocodec dependency guard: PASS (project/lock model-free self-test)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).parent)
    parser.add_argument("--nemo-source-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(args.project.resolve())
            if args.nemo_source_root is not None:
                verify_source(args.nemo_source_root.resolve())
                print("nanocodec dependency guard: PASS (pinned NeMo source audit)")
        else:
            verify_project(args.project.resolve())
            if args.nemo_source_root is None:
                fail("full audit requires --nemo-source-root; refusing an unverified source boundary")
            verify_source(args.nemo_source_root.resolve())
            print("nanocodec dependency guard: PASS")
    except (OSError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
