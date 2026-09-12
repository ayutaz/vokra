#!/usr/bin/env python3
"""Create and validate the external, exact-HEAD BiCodec approval record.

The approval cannot be committed as a fixed JSON file: its ``git_commit`` is
required to equal the clean checkout used by the VAST worker, and committing
the record would change that checkout's HEAD.  This generator keeps the
owner-approved upstream identity immutable while binding the external record
to the exact HEAD immediately before the worker starts.

This is an offline metadata helper.  It never downloads, opens, or executes a
model/checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

IMMUTABLE = {
    "schema": "vokra-bicodec-approval-v1",
    "model": "SparkAudio/Spark-TTS-0.5B",
    "upstream_repo": "https://github.com/SparkAudio/Spark-TTS",
    "upstream_revision": "2f1ea9082400547242641f5271b6f941c9f439d1",
    "upstream_hf_revision": "642071559bfc6346c2359d19dcb6be3f9dd8a05d",
    "license_spdx": "cc-by-nc-sa-4.0",
    "checkpoint_sha256": "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec",
    "config_sha256": "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be",
    "no_upload": True,
    "decision": "RESEARCH_ONLY",
}
SIGNER = "yousan"
EXPECTED_KEYS = set(IMMUTABLE) | {"git_commit", "signer", "scope_sha256"}


def scope_digest(value: dict[str, Any]) -> str:
    scope = {key: value[key] for key in EXPECTED_KEYS if key not in {"scope_sha256", "signer"}}
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(head: str) -> dict[str, Any]:
    if not HEX40.fullmatch(head):
        raise ValueError("git_commit must be lowercase 40-hex")
    value = {**IMMUTABLE, "git_commit": head, "signer": SIGNER}
    value["scope_sha256"] = scope_digest(value)
    return value


def validate(value: Any, expected_head: str) -> None:
    if not HEX40.fullmatch(expected_head):
        raise ValueError("expected HEAD must be lowercase 40-hex")
    if not isinstance(value, dict) or set(value) != EXPECTED_KEYS:
        raise ValueError("approval schema is not exact")
    for key, expected in IMMUTABLE.items():
        if value.get(key) != expected:
            raise ValueError(f"approval identity drift: {key}")
    if value.get("git_commit") != expected_head:
        raise ValueError("approval git_commit does not match expected HEAD")
    if value.get("signer") != SIGNER:
        raise ValueError("approval signer is not the owner signer")
    if not HEX64.fullmatch(str(value.get("scope_sha256"))):
        raise ValueError("approval scope digest is not lowercase 64-hex")
    if value["scope_sha256"] != scope_digest(value):
        raise ValueError("approval scope digest mismatch")


def current_head(repo_root: Path) -> str:
    repo_root = repo_root.absolute()
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise ValueError("--repo-root must be an absolute regular directory")
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        if status:
            raise ValueError("repository worktree must be clean")
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot resolve repository HEAD: {error}") from error


def _reject_symlink_ancestry(path: Path) -> None:
    current = path.absolute().parent
    while True:
        if current.is_symlink():
            raise ValueError("output parent has symlink ancestry")
        if current == current.parent:
            return
        current = current.parent


def write_no_clobber(path: Path, value: dict[str, Any], repo_root: Path) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("--output must be an absent absolute path")
    repo_root = repo_root.absolute()
    output = path.absolute()
    if output == repo_root or repo_root in output.parents:
        raise ValueError("--output must be outside --repo-root")
    _reject_symlink_ancestry(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def self_test() -> int:
    head = "1" * 40
    value = build(head)
    validate(value, head)
    tampered = dict(value)
    tampered["git_commit"] = "2" * 40
    try:
        validate(tampered, head)
    except ValueError:
        pass
    else:
        raise AssertionError("stale git_commit accepted")
    tampered = dict(value)
    tampered["scope_sha256"] = "0" * 64
    try:
        validate(tampered, head)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered scope digest accepted")
    with tempfile.TemporaryDirectory(prefix="vokra-bicodec-approval-") as directory:
        root = Path(directory) / "repo"
        root.mkdir()
        subprocess.run(["git", "-C", str(root), "init", "--quiet"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "self-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "self-test"], check=True)
        marker = root / "marker"
        marker.write_text("clean\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "marker"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "--quiet", "-m", "self-test"], check=True)
        clean_head = current_head(root)
        if not HEX40.fullmatch(clean_head):
            raise AssertionError("clean temporary repository HEAD was not resolved")
        (root / "dirty").write_text("dirty\n", encoding="utf-8")
        try:
            current_head(root)
        except ValueError as error:
            if "clean" not in str(error):
                raise AssertionError("dirty worktree failure was not explicit") from error
        else:
            raise AssertionError("dirty worktree accepted")
        (root / "dirty").unlink()
        try:
            write_no_clobber(root / "inside" / "approval.json", value, root)
        except ValueError:
            pass
        else:
            raise AssertionError("repository-local output accepted")
        symlink_parent = Path(directory) / "symlink-parent"
        symlink_parent.symlink_to(Path(directory), target_is_directory=True)
        try:
            write_no_clobber(symlink_parent / "approval.json", value, root)
        except ValueError:
            pass
        else:
            raise AssertionError("symlink output ancestry accepted")
    print("bicodec_owner_approval.py self-test: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--expected-head")
    args = parser.parse_args(argv)
    if args.self_test:
        if any(value is not None for value in (args.repo_root, args.git_commit, args.output, args.validate, args.expected_head)):
            parser.error("--self-test accepts no other options")
        return self_test()
    if args.validate is not None:
        if args.output is not None or args.repo_root is not None or args.git_commit is not None or args.expected_head is None:
            parser.error("--validate requires only --expected-head")
        try:
            value = json.loads(args.validate.read_text(encoding="utf-8"))
            validate(value, args.expected_head)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"bicodec approval: BLOCKED: {error}", file=sys.stderr)
            return 2
        print("bicodec approval: PASS")
        return 0
    if args.output is None or args.repo_root is None:
        parser.error("--repo-root and --output are required")
    try:
        head = current_head(args.repo_root)
        if args.git_commit is not None and args.git_commit != head:
            raise ValueError("--git-commit does not match repository HEAD")
        if args.expected_head is not None and head != args.expected_head:
            raise ValueError("repository HEAD does not match --expected-head")
        value = build(head)
        write_no_clobber(args.output, value, args.repo_root)
        validate(value, head)
    except (OSError, ValueError) as error:
        print(f"bicodec approval: BLOCKED: {error}", file=sys.stderr)
        return 2
    print(f"bicodec approval: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
