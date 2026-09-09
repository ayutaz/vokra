#!/usr/bin/env python3
"""Offline security gate for the narrow GigaAM v3 reference closure.

This gate intentionally reads repository text only.  It does not import the
remote model, Lightning, Torch, or any other third-party package, and it never
downloads a checkpoint.  The fixed short-PCM route must remain independent of
the optional long-form pyannote/Lightning stack.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parent
PARITY = PROJECT.parent
PYPROJECT = PROJECT / "pyproject.toml"
LOCK = PROJECT / "uv.lock"
DUMPER = PARITY / "sber_gigaam_v3_dump_reference.py"
PREPARER = PARITY / "sber_gigaam_v3_prepare_checkpoint.py"
VALIDATOR = PARITY / "gigaam_v3_validation.py"
WORKER = ROOT / "scripts" / "publish" / "vast-ai" / "run-gigaam-v3-validation.sh"
BANNED_IDENTITIES = {"lightning", "pytorch-lightning", "pyannote-audio", "torchcodec"}
FORBIDDEN_CALLS = ("load_from_checkpoint", "_load_state", "_instantiator")


def fail(message: str) -> None:
    raise AssertionError(message)


def verify_loader_ast(source: str) -> None:
    """Require the one fixed, local-only AutoModel loader call.

    This deliberately inspects the syntax tree rather than searching source
    text: a docstring or comment must not be able to spoof this gate.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        fail(f"GigaAM dumper is not valid Python: {exc}")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "auto_model"
    ]
    if len(calls) != 1:
        fail(f"expected exactly one auto_model.from_pretrained call, found {len(calls)}")
    call = calls[0]
    if len(call.args) != 1 or any(keyword.arg is None for keyword in call.keywords):
        fail("AutoModel loader must have one model path and explicit keyword arguments")
    keyword_values: dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg in keyword_values:
            fail(f"AutoModel loader repeats keyword: {keyword.arg}")
        keyword_values[keyword.arg] = keyword.value
    required = {"revision", "trust_remote_code", "local_files_only"}
    if not required.issubset(keyword_values):
        fail("AutoModel loader lost a required fixed provenance keyword")
    revision = keyword_values["revision"]
    if not isinstance(revision, ast.Name) or revision.id != "HF_REVISION":
        fail("AutoModel loader revision is not the fixed HF_REVISION name")
    for name in ("trust_remote_code", "local_files_only"):
        value = keyword_values[name]
        if not isinstance(value, ast.Constant) or value.value is not True:
            fail(f"AutoModel loader {name} must be the literal True")


def verify_forbidden_ast(source: str, path: Path) -> None:
    """Reject executable references to the vulnerable Lightning APIs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        fail(f"invalid Python security input {path}: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            fail(f"forbidden Lightning name re-entered GigaAM route: {path}:{node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
            fail(f"forbidden Lightning attribute re-entered GigaAM route: {path}:{node.lineno}")


def run_ast_tamper_self_tests() -> None:
    """Exercise the loader gate without importing Python model dependencies."""
    good = """
def load(auto_model, model_dir):
    return auto_model.from_pretrained(
        model_dir,
        revision=HF_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
"""
    verify_loader_ast(good)
    tampered = {
        "wrong receiver": good.replace("auto_model.from_pretrained", "AutoModel.from_pretrained"),
        "mutable revision": good.replace("revision=HF_REVISION", 'revision="latest"'),
        "remote code disabled": good.replace("trust_remote_code=True", "trust_remote_code=False"),
        "local files disabled": good.replace("local_files_only=True", "local_files_only=False"),
        "duplicate loader": good + "\nauto_model.from_pretrained(model_dir)\n",
    }
    for label, source in tampered.items():
        try:
            verify_loader_ast(source)
        except AssertionError:
            continue
        fail(f"loader AST tamper case unexpectedly passed: {label}")


def main() -> int:
    run_ast_tamper_self_tests()
    for path in (PYPROJECT, LOCK, DUMPER, PREPARER, VALIDATOR, WORKER):
        if not path.is_file() or path.is_symlink():
            fail(f"missing or symlinked security input: {path}")
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = project.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list):
        fail("GigaAM dependencies are not a list")
    dependency_text = "\n".join(str(item) for item in dependencies).casefold()
    if any(identity in dependency_text for identity in BANNED_IDENTITIES):
        fail("optional pyannote/Lightning dependency re-entered the narrow project")
    lock_text = LOCK.read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    rows = lock.get("package", [])
    if not isinstance(rows, list):
        fail("GigaAM uv.lock package rows are malformed")
    locked_names = {str(row.get("name", "")).casefold() for row in rows if isinstance(row, dict)}
    if locked_names.intersection(BANNED_IDENTITIES):
        fail("vulnerable or optional Lightning package re-entered uv.lock")
    source = DUMPER.read_text(encoding="utf-8")
    if "AutoModel" not in source or "trust_remote_code=True" not in source or "local_files_only=True" not in source:
        fail("GigaAM dumper lost the fixed local-only remote-code route")
    if not re.search(r"HF_REVISION\s*=\s*\"[0-9a-f]{40}\"", source):
        fail("GigaAM dumper lost its immutable HF revision")
    for path in (DUMPER, PREPARER, VALIDATOR):
        verify_forbidden_ast(path.read_text(encoding="utf-8"), path)
    if any(call in WORKER.read_text(encoding="utf-8") for call in FORBIDDEN_CALLS):
        fail(f"forbidden Lightning checkpoint API re-entered GigaAM worker: {WORKER}")
    verify_loader_ast(source)
    print("gigaam_v3 security self-test: PASS (offline, no model, no network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
