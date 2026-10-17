#!/usr/bin/env python3
"""Bind a hosted RC smoke receipt to the verified desktop distribution.

This annotator only reads JSON and bytes.  It never executes a model or the
CLI; the hosted smoke job must have completed successfully before it is run.
The receipt is replaced atomically only after every identity and provenance
check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import tempfile

import desktop_release


class DistributionIdentityError(ValueError):
    """The smoke receipt and distribution cannot be bound safely."""


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
POSITIVE_DECIMAL_RE = re.compile(r"^[1-9][0-9]*$")
EXPECTED_LABELS = {"asr-jfk", "vad-jfk", "vad-silence"}
IDENTITY_SCHEMA = "vokra-rc-distribution-identity-v1"


def _regular_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise DistributionIdentityError(f"{label} is not a regular file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_head(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise DistributionIdentityError(f"{label} is not a lowercase 40-hex SHA")
    return value


def _require_positive_decimal(value: str, label: str) -> str:
    if not isinstance(value, str) or not POSITIVE_DECIMAL_RE.fullmatch(value):
        raise DistributionIdentityError(f"{label} is not a positive decimal identifier")
    return value


def _validate_receipt(receipt: object, source_head: str, cli: Path) -> None:
    if not isinstance(receipt, dict):
        raise DistributionIdentityError("receipt top-level value is not an object")
    if type(receipt.get("schema")) is not int or receipt["schema"] != 1:
        raise DistributionIdentityError("receipt schema is not integer schema 1")
    if receipt.get("status") != "PASS":
        raise DistributionIdentityError("receipt status is not PASS")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict):
        raise DistributionIdentityError("receipt runtime object is missing")
    if runtime.get("head") != source_head or runtime.get("expected_head") != source_head:
        raise DistributionIdentityError("receipt runtime head does not match source SHA")
    if runtime.get("platform") != "linux":
        raise DistributionIdentityError("receipt runtime platform is not linux")
    if runtime.get("github_actions") != "true":
        raise DistributionIdentityError("receipt runtime is not a hosted GitHub run")
    if runtime.get("marker") != "1":
        raise DistributionIdentityError("receipt runtime marker is missing")
    for flag in ("functional_smoke_only", "not_numerical_parity", "not_metal_or_performance"):
        if receipt.get(flag) is not True:
            raise DistributionIdentityError(f"receipt safety flag is not true: {flag}")
    semantic_checks = receipt.get("semantic_checks")
    if not isinstance(semantic_checks, dict):
        raise DistributionIdentityError("receipt semantic checks are missing")
    asr = semantic_checks.get("asr_jfk_keyphrase")
    vad = semantic_checks.get("vad")
    if (
        not isinstance(asr, dict)
        or asr.get("passed") is not True
        or not isinstance(vad, dict)
        or vad.get("passed") is not True
    ):
        raise DistributionIdentityError("receipt ASR/VAD semantic checks are not passed")

    commands = receipt.get("commands")
    if not isinstance(commands, list) or len(commands) != len(EXPECTED_LABELS):
        raise DistributionIdentityError("receipt must contain exactly three smoke commands")
    labels: set[str] = set()
    cli_name = str(cli)
    for command in commands:
        if not isinstance(command, dict):
            raise DistributionIdentityError("receipt command is not an object")
        label = command.get("label")
        if not isinstance(label, str) or label in labels or label not in EXPECTED_LABELS:
            raise DistributionIdentityError("receipt command labels are not the reviewed set")
        labels.add(label)
        if type(command.get("returncode")) is not int or command["returncode"] != 0:
            raise DistributionIdentityError(f"receipt command did not return zero: {label}")
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise DistributionIdentityError(f"receipt command argv is malformed: {label}")
        if argv[0] != cli_name:
            raise DistributionIdentityError(f"receipt command CLI path mismatch: {label}")
        if len(argv) < 2 or argv[1] != "run":
            raise DistributionIdentityError(f"receipt command is not a CLI run: {label}")
        backend_indices = [index for index, item in enumerate(argv) if item == "--backend"]
        if len(backend_indices) != 1:
            raise DistributionIdentityError(f"receipt command backend option is not unique: {label}")
        backend_index = backend_indices[0]
        if backend_index + 1 >= len(argv) or argv[backend_index + 1] != "cpu":
            raise DistributionIdentityError(f"receipt command is not exactly CPU backend: {label}")
    if labels != EXPECTED_LABELS:
        raise DistributionIdentityError("receipt command labels are incomplete")


def _manifest_cli(manifest: Path, version: str, cli: Path) -> tuple[int, str, str]:
    manifest = _regular_file(manifest, "manifest")
    cli = _regular_file(cli, "CLI")
    if manifest.name != "vokra-desktop-manifest.json":
        raise DistributionIdentityError("manifest filename is not the desktop manifest")
    if manifest.parent.resolve() != cli.parent.resolve():
        raise DistributionIdentityError("CLI is not beside the verified manifest")
    try:
        expected_cli = desktop_release._asset_names(version)["cli-linux"]
    except desktop_release.DesktopReleaseError as exc:
        raise DistributionIdentityError(str(exc)) from exc
    if cli.name != expected_cli:
        raise DistributionIdentityError("CLI filename does not match the RC version")

    try:
        desktop_release.verify_dist(
            argparse.Namespace(version=version, dist_dir=str(manifest.parent))
        )
    except desktop_release.DesktopReleaseError as exc:
        raise DistributionIdentityError(f"desktop manifest verification failed: {exc}") from exc
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributionIdentityError("desktop manifest cannot be read as JSON") from exc
    assets = document.get("assets") if isinstance(document, dict) else None
    matches = [item for item in assets or [] if isinstance(item, dict) and item.get("name") == cli.name]
    if len(matches) != 1:
        raise DistributionIdentityError("desktop manifest has no unique Linux CLI entry")
    entry = matches[0]
    actual_bytes = cli.stat().st_size
    actual_sha256 = _sha256(cli)
    if entry.get("size") != actual_bytes or entry.get("sha256") != actual_sha256:
        raise DistributionIdentityError("desktop manifest does not match the Linux CLI bytes")
    return actual_bytes, actual_sha256, _sha256(manifest)


def annotate(
    *,
    receipt_path: Path,
    manifest_path: Path,
    cli_path: Path,
    version: str,
    source_head: str,
    github_run_id: str,
    github_run_attempt: str,
) -> None:
    receipt_path = _regular_file(receipt_path, "receipt")
    source_head = _require_source_head(source_head, "source head")
    github_run_id = _require_positive_decimal(github_run_id, "GitHub run id")
    github_run_attempt = _require_positive_decimal(github_run_attempt, "GitHub run attempt")
    manifest_path = _regular_file(manifest_path, "manifest")
    cli_path = _regular_file(cli_path, "CLI")
    cli_path = cli_path.resolve()
    manifest_path = manifest_path.resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DistributionIdentityError("receipt cannot be read as JSON") from exc
    if isinstance(receipt, dict) and "distribution_identity" in receipt:
        raise DistributionIdentityError("receipt already has distribution_identity")
    _validate_receipt(receipt, source_head, cli_path)
    cli_bytes, cli_sha256, manifest_sha256 = _manifest_cli(manifest_path, version, cli_path)
    updated = dict(receipt)
    updated["distribution_identity"] = {
        "schema": IDENTITY_SCHEMA,
        "source_head": source_head,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
        "cli": {
            "filename": cli_path.name,
            "bytes": cli_bytes,
            "sha256": cli_sha256,
        },
        "manifest_sha256": manifest_sha256,
    }
    original_mode = stat.S_IMODE(receipt_path.stat().st_mode)
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.", suffix=".tmp", dir=receipt_path.parent
        )
        temporary = Path(temporary_name)
        with open(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(updated, stream, indent=2)
            stream.write("\n")
        temporary.chmod(original_mode)
        temporary.replace(receipt_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--github-run-id", required=True)
    parser.add_argument("--github-run-attempt", required=True)
    args = parser.parse_args(argv)
    try:
        annotate(
            receipt_path=args.receipt,
            manifest_path=args.manifest,
            cli_path=args.cli,
            version=args.version,
            source_head=args.source_head,
            github_run_id=args.github_run_id,
            github_run_attempt=args.github_run_attempt,
        )
    except (DistributionIdentityError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"rc-distribution-identity: annotated {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
