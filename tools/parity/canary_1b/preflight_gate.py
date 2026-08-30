#!/usr/bin/env python3
"""Offline approval gate for the two complete NVIDIA Canary-1B releases.

The gate is deliberately dependency-free and must be called before a worker
checks its host, creates scratch/evidence directories, invokes ``uv sync``,
downloads a checkpoint, or starts Cargo.  It authenticates a small,
owner-supplied JSON record against the immutable release identities recorded in
the manifest.  The gate does not download or inspect model payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


GATE_VERSION = 1
MANIFEST_SCHEMA = "vokra-canary-1b-gate-v1"
APPROVAL_SCHEMA = "vokra-canary-1b-approval-v1"
LICENSE_SPDX = "CC-BY-4.0"
PUBLICATION = "NO_UPLOAD"
ATTRIBUTION_REQUIRED = True
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {
    "",
    "anonymous",
    "example",
    "none",
    "owner-example",
    "owner_example",
    "null",
    "owner_review_required",
    "pending",
    "pending_review",
    "review_required",
    "self-test",
    "test",
    "todo",
    "unresolved",
}

# These are copied from the already audited preparation workers.  Keep the
# values here as a second, dependency-free check: changing the JSON manifest
# alone must never change the scope that an approval can unlock.
VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "canary-1b-flash",
        "model": "canary-1b-flash",
        "upstream_repo": "nvidia/canary-1b-flash",
        "upstream_revision": "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e",
        "archive_filename": "canary-1b-flash.nemo",
        "archive_bytes": 3_540_715_520,
        "archive_sha256": "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324",
        "main_checkpoint_member": None,
        "main_checkpoint_bytes": None,
        "attribution_required": True,
        "attribution_text": "This application uses NVIDIA Canary-1B-Flash (multilingual ASR / AST for English, German, Spanish and French). Model weights are licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-flash",
    },
    {
        "variant": "canary-1b-v2",
        "model": "canary-1b-v2",
        "upstream_repo": "nvidia/canary-1b-v2",
        "upstream_revision": "87bc52657add533cd0156b3fc1aef027280754bf",
        "archive_filename": "canary-1b-v2.nemo",
        "archive_bytes": 6_358_958_080,
        "archive_sha256": "ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094",
        "main_checkpoint_member": "./model_weights.ckpt",
        "main_checkpoint_bytes": 3_853_798_427,
        "attribution_required": True,
        "attribution_text": "This application uses NVIDIA Canary-1B-v2 (multilingual ASR / AST across 25 languages). Model weights are licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-v2",
    },
)

MANIFEST_KEYS = {"schema", "gate_version", "license_spdx", "publication", "variants"}
VARIANT_KEYS = {
    "variant",
    "model",
    "upstream_repo",
    "upstream_revision",
    "archive_filename",
    "archive_bytes",
    "archive_sha256",
    "main_checkpoint_member",
    "main_checkpoint_bytes",
    "attribution_required",
    "attribution_text",
}
APPROVAL_KEYS = {
    "schema",
    "variant",
    "model",
    "upstream_repo",
    "upstream_revision",
    "license_spdx",
    "archive_filename",
    "archive_bytes",
    "archive_sha256",
    "main_checkpoint_member",
    "main_checkpoint_bytes",
    "attribution_required",
    "attribution_text",
    "publication",
    "manifest_sha256",
    "no_upload",
    "decision",
    "signer",
    "scope_sha256",
}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return digest_bytes(encoded)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unreadable JSON {path}: {exc}") from exc


def reject_symlink_ancestry(path: Path) -> None:
    """Reject direct or intermediate symlinks, including a dangling leaf."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.root)
    for component in absolute.parts[1:]:
        current /= component
        # macOS exposes the writable temporary hierarchy through the standard
        # `/var` compatibility symlink.  It is an OS-owned anchor, not an
        # evidence alias; continue walking its children while rejecting every
        # user-controlled symlink below it.
        if current.is_symlink() and current != Path("/var"):
            raise ValueError(f"symlinked approval/gate path: {path}")


def require_regular(path: Path, label: str) -> bytes:
    reject_symlink_ancestry(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or not a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {exc}") from exc


def reviewed_signer(value: Any, *, allow_self_test: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"\s+", "_", value.strip().casefold())
    if not normalized:
        return False
    # The fixture signer is accepted only by the in-process self-test.  A
    # production invocation can never unlock with an example/test identity.
    if normalized == "owner-example":
        return allow_self_test
    if (
        "example" in normalized
        or "self-test" in normalized
        or "self_test" in normalized
        or re.search(r"(?:^|[-_])test(?:$|[-_])", normalized)
    ):
        return False
    if normalized in PLACEHOLDERS:
        return False
    return True


def expected_manifest() -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "gate_version": GATE_VERSION,
        "license_spdx": LICENSE_SPDX,
        "publication": PUBLICATION,
        "variants": [dict(variant) for variant in VARIANTS],
    }


def variant_for(name: str) -> dict[str, Any]:
    for variant in VARIANTS:
        if variant["variant"] == name:
            return dict(variant)
    raise ValueError(f"unknown Canary variant: {name}")


def approval_scope(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant["variant"],
        "model": variant["model"],
        "upstream_repo": variant["upstream_repo"],
        "upstream_revision": variant["upstream_revision"],
        "license_spdx": LICENSE_SPDX,
        "archive_filename": variant["archive_filename"],
        "archive_bytes": variant["archive_bytes"],
        "archive_sha256": variant["archive_sha256"],
        "main_checkpoint_member": variant["main_checkpoint_member"],
        "main_checkpoint_bytes": variant["main_checkpoint_bytes"],
        "attribution_required": ATTRIBUTION_REQUIRED,
        "attribution_text": variant["attribution_text"],
        "no_upload": True,
        "publication": PUBLICATION,
    }


def validate(
    manifest_path: Path,
    approval_path: Path,
    variant_name: str,
    *,
    allow_self_test_signer: bool = False,
) -> tuple[bool, str]:
    try:
        manifest_bytes = require_regular(manifest_path, "gate manifest")
        require_regular(approval_path, "external approval evidence")
        manifest = load_json(manifest_path)
        approval = load_json(approval_path)
        expected = expected_manifest()
        if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
            raise ValueError("gate manifest schema is not exact")
        variants = manifest.get("variants")
        if not isinstance(variants, list) or any(
            not isinstance(variant, dict) or set(variant) != VARIANT_KEYS for variant in variants
        ):
            raise ValueError("gate manifest variant schema is not exact")
        # Canonical JSON distinguishes JSON booleans from integers (Python's
        # ``True == 1`` would otherwise let a type-tampered manifest through).
        if canonical_digest(manifest) != canonical_digest(expected):
            raise ValueError("gate manifest identities or policy drifted")
        if not isinstance(approval, dict) or set(approval) != APPROVAL_KEYS:
            raise ValueError("external approval schema is not exact")
        variant = variant_for(variant_name)
        expected_approval = {
            "schema": APPROVAL_SCHEMA,
            **approval_scope(variant),
            "manifest_sha256": digest_bytes(manifest_bytes),
        }
        for key in (
            "schema",
            "variant",
            "model",
            "upstream_repo",
            "upstream_revision",
            "license_spdx",
            "archive_filename",
            "archive_bytes",
            "archive_sha256",
            "main_checkpoint_member",
            "main_checkpoint_bytes",
            "attribution_required",
            "attribution_text",
            "publication",
            "manifest_sha256",
        ):
            if approval.get(key) != expected_approval[key]:
                raise ValueError(f"approval identity mismatch: {key}")
        if approval.get("no_upload") is not True or approval.get("decision") != "APPROVED":
            raise ValueError("approval must explicitly be APPROVED with no_upload=true")
        if approval.get("attribution_required") is not True:
            raise ValueError("approval must explicitly require NVIDIA attribution")
        if not reviewed_signer(approval.get("signer"), allow_self_test=allow_self_test_signer):
            raise ValueError("approval signer is missing or a placeholder")
        if not isinstance(approval.get("scope_sha256"), str) or not HEX64.fullmatch(approval["scope_sha256"]):
            raise ValueError("approval scope_sha256 is not lowercase SHA-256")
        expected_scope = canonical_digest(approval_scope(variant))
        if approval["scope_sha256"] != expected_scope:
            raise ValueError("approval scope digest does not cover exact immutable identity")
        return True, "PASS"
    except (OSError, TypeError, ValueError) as exc:
        return False, str(exc)


def self_test() -> int:
    """Exercise duplicate/missing/symlink/tamper rejection without network."""
    with tempfile.TemporaryDirectory(prefix="vokra-canary-gate-") as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        approval_path = root / "approval.json"
        manifest = expected_manifest()
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        variant = variant_for("canary-1b-flash")
        scope = canonical_digest(approval_scope(variant))
        approval = {
            "schema": APPROVAL_SCHEMA,
            **approval_scope(variant),
            "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
            "no_upload": True,
            "decision": "APPROVED",
            "signer": "owner-example",
            "scope_sha256": scope,
        }
        approval_path.write_text(json.dumps(approval, separators=(",", ":")), encoding="utf-8")
        ok, reason = validate(
            manifest_path,
            approval_path,
            "canary-1b-flash",
            allow_self_test_signer=True,
        )
        if not ok:
            print(f"canary gate valid approval rejected: {reason}", file=sys.stderr)
            return 1
        if validate(manifest_path, approval_path, "canary-1b-flash")[0]:
            print("canary gate accepted self-test signer in production mode", file=sys.stderr)
            return 1

        def rejected(label: str, mutate: Any) -> bool:
            candidate = json.loads(approval_path.read_text(encoding="utf-8"))
            mutate(candidate)
            candidate_path = root / f"{label}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            return not validate(manifest_path, candidate_path, "canary-1b-flash")[0]

        checks = {
            "missing": lambda value: value.pop("archive_sha256"),
            "tampered": lambda value: value.update(archive_sha256="0" * 64),
            "scope": lambda value: value.update(scope_sha256="0" * 64),
            "signer": lambda value: value.update(signer="PENDING_REVIEW"),
            "signer-example-form": lambda value: value.update(signer="owner-example-review"),
            "decision": lambda value: value.update(decision="PENDING_REVIEW"),
            "upload": lambda value: value.update(no_upload=False),
            "attribution-required": lambda value: value.update(attribution_required=False),
            "attribution-type": lambda value: value.update(attribution_required=1),
            "attribution-text": lambda value: value.update(attribution_text="tampered attribution"),
        }
        for label, mutate in checks.items():
            if not rejected(label, mutate):
                print(f"canary gate accepted {label} approval tamper", file=sys.stderr)
                return 1

        tampered_manifest = root / "tampered-manifest.json"
        changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed_manifest["variants"][0]["attribution_text"] = "tampered attribution"
        tampered_manifest.write_text(json.dumps(changed_manifest), encoding="utf-8")
        if validate(tampered_manifest, approval_path, "canary-1b-flash")[0]:
            print("canary gate accepted a tampered manifest", file=sys.stderr)
            return 1

        duplicate = root / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        if not validate(manifest_path, duplicate, "canary-1b-flash")[1].startswith("unreadable JSON"):
            print("canary gate duplicate-key rejection lost", file=sys.stderr)
            return 1

        link = root / "approval-link.json"
        link.symlink_to(approval_path)
        if validate(manifest_path, link, "canary-1b-flash")[0]:
            print("canary gate accepted symlinked approval", file=sys.stderr)
            return 1

        v2 = root / "v2-approval.json"
        v2_doc = dict(approval)
        v2_doc.update(
            {
                **approval_scope(variant_for("canary-1b-v2")),
                "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
                "scope_sha256": canonical_digest(approval_scope(variant_for("canary-1b-v2"))),
            }
        )
        v2.write_text(json.dumps(v2_doc), encoding="utf-8")
        ok, reason = validate(
            manifest_path,
            v2,
            "canary-1b-v2",
            allow_self_test_signer=True,
        )
        if not ok:
            print(f"canary gate valid v2 approval rejected: {reason}", file=sys.stderr)
            return 1

    print("canary_1b preflight gate self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--variant", choices=[variant["variant"] for variant in VARIANTS])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.manifest, args.approval, args.variant)):
            parser.error("--self-test accepts no other arguments")
        return self_test()
    if any(value is None for value in (args.manifest, args.approval, args.variant)):
        parser.error("normal runs require --manifest, --approval, and --variant")
    ok, reason = validate(args.manifest, args.approval, args.variant)
    if ok:
        print("canary_1b preflight gate: PASS")
        return 0
    print(f"canary_1b preflight gate: BLOCKED: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
