#!/usr/bin/env -S uv run --no-project --python 3.12
"""Plan a fail-closed paste -> pastey dependency overlay.

This tool consumes only two authenticated, tracked source receipts.  It does
not edit Cargo.toml files, resolve dependencies, access the network, or claim
that an overlay has been applied or compiled.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import posixpath
import re
import stat
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PARENT_RECEIPT = ROOT / "docs/handoff/kyutai-candle-paste-parents-source-2026-10-15.json"
PASTEY_RECEIPT = ROOT / "docs/handoff/kyutai-candle-pastey-source-2026-10-15.json"
PARENT_RECEIPT_REF = "docs/handoff/kyutai-candle-paste-parents-source-2026-10-15.json"
PASTEY_RECEIPT_REF = "docs/handoff/kyutai-candle-pastey-source-2026-10-15.json"
PARENT_BYTES = 218910
PARENT_SHA256 = "234a530ba183472daf0a34444703b934df741a50520f4c48cf421be25ec3ff36"
PASTEY_BYTES = 176564
PASTEY_SHA256 = "d4ca85e59f16d9ae6dc36f406473b6c534fe86895c353c3aca18930bce6edba2"
PASTEY_ARCHIVE_SHA256 = "2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4"
PASTEY_VERSION = "0.2.3"
JSON_LIMIT = 2 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_PARENT_COUNT = 16
EXPECTED_CPU_COUNT = 14
EXPECTED_METAL_COUNT = 2
REQUIRED_STATUS = {
    "status": "COMPLETE",
    "collection_only": True,
    "approval": None,
    "compatibility": "NOT_TESTED",
    "model_execution": "NOT_AUTHORIZED",
    "upload": "NO_UPLOAD",
}
EXPECTED_PASTEY_STATUS = {**REQUIRED_STATUS, "license_review": "NOT_PERFORMED"}
REQUIREMENT_PAIRS = {"1": "^1", "1.0": "^1.0"}
REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
EXPECTED_BINDINGS = (
    ("gemm", "0.17.1", True), ("gemm", "0.18.2", True),
    ("gemm-c32", "0.17.1", True), ("gemm-c32", "0.18.2", True),
    ("gemm-c64", "0.17.1", True), ("gemm-c64", "0.18.2", True),
    ("gemm-common", "0.17.1", True), ("gemm-common", "0.18.2", True),
    ("gemm-f16", "0.17.1", True), ("gemm-f16", "0.18.2", True),
    ("gemm-f32", "0.17.1", True), ("gemm-f32", "0.18.2", True),
    ("gemm-f64", "0.17.1", True), ("gemm-f64", "0.18.2", True),
    ("metal", "0.27.0", False), ("metal", "0.29.0", False),
)


class OverlayError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_authenticated(path: Path, expected_bytes: int, expected_sha256: str) -> bytes:
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        st = os.fstat(fd)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise OverlayError(f"receipt unavailable: {path}") from exc
    try:
        if not stat.S_ISREG(st.st_mode) or st.st_size != expected_bytes:
            raise OverlayError(f"receipt type/size mismatch: {path}")
        chunks = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise OverlayError(f"receipt read failed: {path}") from exc
    finally:
        os.close(fd)
    if len(raw) != expected_bytes or sha256(raw) != expected_sha256:
        raise OverlayError(f"receipt SHA-256 mismatch: {path}")
    return raw


def assert_status(document: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if key not in document:
            raise OverlayError(f"{label} status boundary missing: {key}")
        actual = document[key]
        if (value is None and actual is not None) or (value is not None and type(actual) is not type(value)) or actual != value:
            raise OverlayError(f"{label} status boundary mismatch: {key}")


def parse_receipts(parent_path: Path = PARENT_RECEIPT, pastey_path: Path = PASTEY_RECEIPT) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_raw = read_authenticated(parent_path, PARENT_BYTES, PARENT_SHA256)
    pastey_raw = read_authenticated(pastey_path, PASTEY_BYTES, PASTEY_SHA256)
    try:
        parent = json.loads(parent_raw)
        pastey = json.loads(pastey_raw)
    except json.JSONDecodeError as exc:
        raise OverlayError("authenticated receipt is not valid JSON") from exc
    if not isinstance(parent, dict) or not isinstance(pastey, dict):
        raise OverlayError("authenticated receipts must be JSON objects")
    assert_status(parent, REQUIRED_STATUS, "parent")
    assert_status(pastey, EXPECTED_PASTEY_STATUS, "pastey")
    return parent, pastey


def metadata_paste_is_narrow(binding: dict[str, Any], manifest_literal: str | None = None) -> None:
    dep = binding.get("metadata_paste")
    if not isinstance(dep, dict):
        raise OverlayError(f"missing metadata paste declaration: {binding.get('name')}@{binding.get('version')}")
    expected = {
        "features": [], "kind": None, "optional": False, "registry": None, "rename": None,
        "target": None, "uses_default_features": True,
    }
    for key, value in expected.items():
        if dep.get(key) != value:
            raise OverlayError(f"unsupported paste metadata field {key}: {binding.get('name')}@{binding.get('version')}")
    req_literal = dep.get("req", "")[1:] if isinstance(dep.get("req"), str) and dep["req"].startswith("^") else None
    if req_literal not in REQUIREMENT_PAIRS or REQUIREMENT_PAIRS[req_literal] != dep.get("req"):
        raise OverlayError(f"unrecognized paste requirement pair: {binding.get('name')}@{binding.get('version')}")
    if manifest_literal is not None and req_literal != manifest_literal:
        raise OverlayError(f"paste metadata does not match manifest literal: {binding.get('name')}@{binding.get('version')}")


def manifest_text(parent: dict[str, Any], archive: dict[str, Any]) -> str:
    manifest = archive.get("manifest")
    text_store = parent.get("text_store")
    if not isinstance(manifest, dict) or not isinstance(text_store, dict):
        raise OverlayError(f"manifest/text store missing: {archive.get('name')}@{archive.get('version')}")
    ref = manifest.get("text_ref")
    text = text_store.get(ref)
    if not isinstance(text, str) or sha256(text.encode()) != manifest.get("sha256"):
        raise OverlayError(f"manifest text binding mismatch: {archive.get('name')}@{archive.get('version')}")
    return text


def parse_manifest(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise OverlayError(f"invalid manifest: {label}") from exc
    if not isinstance(parsed, dict):
        raise OverlayError(f"manifest is not a table: {label}")
    return parsed


def dependency_table(manifest: dict[str, Any], label: str) -> dict[str, Any]:
    deps = manifest.get("dependencies")
    if not isinstance(deps, dict) or "paste" not in deps or not isinstance(deps["paste"], dict):
        raise OverlayError(f"missing root dependencies.paste table: {label}")
    dep = deps["paste"]
    if set(dep) - {"version"}:
        raise OverlayError(f"unsupported paste dependency fields: {label}")
    if not isinstance(dep.get("version"), str) or dep["version"] not in REQUIREMENT_PAIRS:
        raise OverlayError(f"unsupported literal paste requirement: {label}")
    return dep


def validate_proposed_manifest(original: str, proposed: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    original_parsed = parse_manifest(original, label)
    dependency_table(original_parsed, label)
    proposed_parsed = parse_manifest(proposed, f"{label} proposed")
    expected = copy.deepcopy(original_parsed)
    expected["dependencies"]["paste"]["version"] = "=0.2.3"
    expected["dependencies"]["paste"]["package"] = "pastey"
    if proposed_parsed != expected:
        raise OverlayError(f"semantic manifest difference exceeds paste overlay: {label}")
    return original_parsed, proposed_parsed


def section_bounds(lines: list[str], header: str, label: str) -> tuple[int, int]:
    matches = [index for index, line in enumerate(lines) if line.rstrip("\r\n") == header]
    if len(matches) != 1:
        raise OverlayError(f"expected exactly one {header}: {label}")
    start = matches[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.match(r"^\s*\[", lines[index]):
            end = index
            break
    return start, end


def propose_manifest(original: str, label: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    parsed = parse_manifest(original, label)
    dep = dependency_table(parsed, label)
    lines = original.splitlines(keepends=True)
    start, end = section_bounds(lines, "[dependencies.paste]", label)
    version_lines = [i for i in range(start + 1, end) if re.match(r"^\s*version\s*=\s*\"(?:1|1\.0)\"\s*$", lines[i].rstrip("\r\n"))]
    if len(version_lines) != 1:
        raise OverlayError(f"expected one literal paste version line: {label}")
    if any(re.match(r"^\s*\[dependencies\.paste\.", line) for line in lines):
        raise OverlayError(f"nested paste table is unsupported: {label}")
    version_index = version_lines[0]
    old_line = lines[version_index]
    newline = "\r\n" if old_line.endswith("\r\n") else "\n" if old_line.endswith("\n") else ""
    prefix = old_line[: len(old_line) - len(old_line.lstrip())]
    lines[version_index] = f'{prefix}version = "=0.2.3"{newline}'
    lines.insert(version_index + 1, f'{prefix}package = "pastey"{newline}')
    proposed = "".join(lines)
    _, proposed_parsed = validate_proposed_manifest(original, proposed, label)
    return proposed, parsed, proposed_parsed


def unchanged_members(archive: dict[str, Any]) -> list[dict[str, Any]]:
    members = archive.get("regular_members")
    if not isinstance(members, list):
        raise OverlayError(f"regular member inventory missing: {archive.get('name')}@{archive.get('version')}")
    rows = []
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise OverlayError("malformed regular member inventory")
        if member["path"] == "Cargo.toml":
            continue
        rows.append({key: member[key] for key in ("path", "bytes", "sha256")})
    return rows


def validate_text_ref(row: dict[str, Any], text_store: dict[str, Any], label: str) -> None:
    if not isinstance(row, dict) or type(row.get("bytes")) is not int or row["bytes"] < 0:
        raise OverlayError(f"malformed text reference: {label}")
    if not isinstance(row.get("sha256"), str) or not SHA256_RE.fullmatch(row["sha256"]):
        raise OverlayError(f"incomplete text reference: {label}")
    if row.get("text_ref") != row["sha256"]:
        raise OverlayError(f"text reference/hash identity mismatch: {label}")
    text = text_store.get(row["text_ref"])
    if not isinstance(text, str) or len(text.encode("utf-8")) != row["bytes"] or sha256(text.encode("utf-8")) != row["sha256"]:
        raise OverlayError(f"text reference binding mismatch: {label}")


def validate_member_ref(row: dict[str, Any], label: str) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("path"), str) or type(row.get("bytes")) is not int or row["bytes"] < 0 or not isinstance(row.get("sha256"), str) or not SHA256_RE.fullmatch(row["sha256"]):
        raise OverlayError(f"malformed member reference: {label}")
    path = row["path"]
    if not path or "\\" in path or "\x00" in path or path.startswith("/") or path.startswith("//") or posixpath.normpath(path) != path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise OverlayError(f"unsafe/non-normalized member path: {label}")


def match_inventory_ref(row: dict[str, Any], inventory: dict[str, dict[str, Any]], label: str, expected_path: str | None = None) -> None:
    if expected_path is not None:
        member = inventory.get(expected_path)
        if member is None or member["bytes"] != row["bytes"] or member["sha256"] != row["sha256"]:
            raise OverlayError(f"referenced member does not match inventory: {label}")
        return
    matches = [member for member in inventory.values() if member["bytes"] == row["bytes"] and member["sha256"] == row["sha256"]]
    if len(matches) != 1:
        raise OverlayError(f"referenced member is missing or ambiguous in inventory: {label}")


def validate_archive_evidence(parent: dict[str, Any], archive: dict[str, Any]) -> str:
    if not isinstance(archive, dict):
        raise OverlayError("archive entry is not an object")
    name, version = archive.get("name"), archive.get("version")
    label = f"{name}@{version}"
    text_store = parent.get("text_store")
    if not isinstance(text_store, dict):
        raise OverlayError(f"archive/text store missing: {label}")
    manifest = archive.get("manifest")
    validate_text_ref(manifest, text_store, f"{label} manifest")
    original = text_store[manifest["text_ref"]]
    if archive.get("manifest_orig") is not None:
        validate_text_ref(archive["manifest_orig"], text_store, f"{label} Cargo.toml.orig")
        if archive.get("manifest_orig_text") != text_store[archive["manifest_orig"]["text_ref"]]:
            raise OverlayError(f"Cargo.toml.orig text binding mismatch: {label}")
    elif archive.get("manifest_orig_text") is not None:
        raise OverlayError(f"unexpected Cargo.toml.orig text: {label}")
    members = archive.get("regular_members")
    if not isinstance(members, list):
        raise OverlayError(f"regular member inventory missing: {label}")
    inventory: dict[str, dict[str, Any]] = {}
    for index, member in enumerate(members):
        validate_member_ref(member, f"{label} regular member {index}")
        if member["path"] in inventory:
            raise OverlayError(f"duplicate regular member path: {label} {member['path']}")
        inventory[member["path"]] = member
    match_inventory_ref(manifest, inventory, f"{label} Cargo.toml", "Cargo.toml")
    rust_sources = archive.get("rust_sources")
    if not isinstance(rust_sources, dict):
        raise OverlayError(f"Rust source inventory missing: {label}")
    for path, member in rust_sources.items():
        if not isinstance(path, str):
            raise OverlayError(f"malformed Rust source path: {label}")
        rust_member = {"path": path, **member} if isinstance(member, dict) else member
        validate_member_ref(rust_member, f"{label} Rust source {path}")
        match_inventory_ref(rust_member, inventory, f"{label} Rust source {path}", path)
    licenses = archive.get("license_members")
    if not isinstance(licenses, dict):
        raise OverlayError(f"license inventory missing: {label}")
    for path, member in licenses.items():
        if not isinstance(path, str):
            raise OverlayError(f"malformed license path: {label}")
        validate_text_ref(member, text_store, f"{label} license {path}")
        match_inventory_ref(member, inventory, f"{label} license {path}", path)
    if archive.get("manifest_orig") is not None:
        match_inventory_ref(archive["manifest_orig"], inventory, f"{label} Cargo.toml.orig", "Cargo.toml.orig")
    vcs_member = archive.get("vcs_member")
    if vcs_member is not None:
        validate_text_ref(vcs_member, text_store, f"{label} VCS member")
        match_inventory_ref(vcs_member, inventory, f"{label} VCS member")
    return original


def patch_snippet(bindings: list[dict[str, Any]]) -> str:
    rows = ["[patch.crates-io]"]
    keys: set[str] = set()
    for binding in bindings:
        name, version = binding["name"], binding["version"]
        key = re.sub(r"[^A-Za-z0-9_]", "_", f"{name}_v{version.replace('.', '_')}")
        if key in keys:
            raise OverlayError(f"duplicate isolated patch key: {key}")
        keys.add(key)
        rows.append(f'"{key}" = {{ package = "{name}", path = "vendor/{name}-{version}" }}')
    return "\n".join(rows) + "\n"


def build_plan(parent: dict[str, Any], pastey: dict[str, Any]) -> dict[str, Any]:
    assert_status(parent, REQUIRED_STATUS, "parent")
    assert_status(pastey, EXPECTED_PASTEY_STATUS, "pastey")
    identity = parent.get("identity")
    plan = parent.get("plan")
    archives = parent.get("archives")
    if identity != {"collected": 16, "expected": 16, "missing": []} or not isinstance(plan, dict) or not isinstance(archives, list) or len(archives) != 16:
        raise OverlayError("parent receipt is not the exact 16-package evidence")
    bindings = plan.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != EXPECTED_PARENT_COUNT:
        raise OverlayError("parent bindings count mismatch")
    if any(not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("version"), str) or type(row.get("cpu")) is not bool for row in bindings):
        raise OverlayError("parent binding schema mismatch")
    actual_bindings = tuple((row["name"], row["version"], row["cpu"]) for row in bindings)
    if actual_bindings != EXPECTED_BINDINGS:
        raise OverlayError("parent binding identities/order/CPU flags mismatch")
    if sum(bool(row.get("cpu")) for row in bindings) != EXPECTED_CPU_COUNT or sum(not bool(row.get("cpu")) for row in bindings) != EXPECTED_METAL_COUNT:
        raise OverlayError("parent CPU/Metal set mismatch")
    if any(not isinstance(archive, dict) for archive in archives):
        raise OverlayError("parent archive schema mismatch")
    archive_by_identity = {f"{a.get('name')}@{a.get('version')}": a for a in archives}
    if len(archive_by_identity) != len(archives):
        raise OverlayError("duplicate parent archive identity")
    proposals = []
    for binding in bindings:
        name, version = binding.get("name"), binding.get("version")
        key = f"{name}@{version}"
        archive = archive_by_identity.get(key)
        if archive is None or archive.get("archive_sha256") != binding.get("checksum") or binding.get("source") != REGISTRY_SOURCE:
            raise OverlayError(f"parent archive binding mismatch: {key}")
        original = validate_archive_evidence(parent, archive)
        proposed, original_parsed, proposed_parsed = propose_manifest(original, key)
        if original_parsed.get("package", {}).get("name") != name or original_parsed.get("package", {}).get("version") != version:
            raise OverlayError(f"manifest package identity mismatch: {key}")
        manifest_literal = original_parsed["dependencies"]["paste"]["version"]
        metadata_paste_is_narrow(binding, manifest_literal)
        original_bytes = original.encode("utf-8")
        proposed_bytes = proposed.encode("utf-8")
        diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), proposed.splitlines(keepends=True),
                                              fromfile=f"{key}/Cargo.toml", tofile=f"{key}/Cargo.toml.overlay"))
        proposals.append({
            "identity": key, "cpu": bool(binding["cpu"]), "source": binding.get("source"), "archive_sha256": binding.get("checksum"),
            "metadata_paste": binding["metadata_paste"],
            "original_manifest": {"sha256": archive["manifest"]["sha256"], "bytes": archive["manifest"]["bytes"], "text_ref": archive["manifest"]["text_ref"]},
            "proposed_manifest": {"text": proposed, "bytes": len(proposed_bytes), "sha256": sha256(proposed_bytes)},
            "unified_diff": diff, "original_manifest_parsed": original_parsed,
            "expected_unchanged_members": unchanged_members(archive),
            "expected_unchanged_rust_sources": archive.get("rust_sources", {}),
            "expected_unchanged_manifest_orig": None if archive.get("manifest_orig") is None else {
                "sha256": archive["manifest_orig"]["sha256"], "bytes": archive["manifest_orig"]["bytes"],
                "text_ref": archive["manifest_orig"]["text_ref"],
            },
            "license_members": archive.get("license_members", {}), "vcs_member": archive.get("vcs_member"),
            "application": "NOT_RUN", "source_unchanged": "NOT_CLAIMED_UNTIL_HOSTED_APPLICATION",
        })
    pastey_archive = pastey.get("archive", {})
    pastey_manifest = pastey.get("manifest", {})
    pastey_package = pastey_manifest.get("package", {})
    if pastey_archive.get("compressed_sha256") != PASTEY_ARCHIVE_SHA256 or pastey_package.get("name") != "pastey" or pastey_package.get("version") != PASTEY_VERSION or pastey_package.get("license") != "MIT OR Apache-2.0" or pastey_package.get("build") is not False or pastey_manifest.get("lib", {}).get("proc-macro") is not True:
        raise OverlayError("pastey source receipt identity/license/proc-macro boundary mismatch")
    snippet = patch_snippet(bindings)
    payload = {
        "status": "PLAN_ONLY", "collection_only": True, "approval": None, "compatibility": "NOT_TESTED",
        "license_review": "NOT_PERFORMED", "model_execution": "NOT_AUTHORIZED", "upload": "NO_UPLOAD",
        "application": "NOT_RUN", "source_unchanged": "NOT_CLAIMED_UNTIL_HOSTED_APPLICATION",
        "parent_receipt": {"path": PARENT_RECEIPT_REF, "bytes": PARENT_BYTES, "sha256": PARENT_SHA256},
        "pastey_receipt": {"path": PASTEY_RECEIPT_REF, "bytes": PASTEY_BYTES, "sha256": PASTEY_SHA256,
                           "archive_sha256": PASTEY_ARCHIVE_SHA256, "version": PASTEY_VERSION,
                           "license": pastey_package["license"], "proc_macro": True, "build": False},
        "parent_counts": {"total": len(proposals), "cpu": sum(p["cpu"] for p in proposals), "metal": sum(not p["cpu"] for p in proposals)},
        "proposals": proposals,
        "isolated_patch_crates_io": snippet,
        "root_patch_boundary": "FORBIDDEN: root paste -> pastey [patch.crates-io] would patch the paste identity, not the dependency package identity; use package=\"pastey\" on each dependency overlay.",
        "application_limits": {"json_bytes": JSON_LIMIT, "source_manifest_files_written": 0, "cargo_run": "NOT_RUN", "network": "NOT_RUN"},
    }
    return payload


def write_plan(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or os.path.lexists(path) or not path.parent.is_dir() or path.parent.is_symlink():
        raise OverlayError(f"output must be fresh absolute path with existing nonsymlink parent: {path}")
    raw = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    if len(raw) > JSON_LIMIT:
        raise OverlayError(f"plan exceeds {JSON_LIMIT} bytes")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.plan_only and args.output is None:
        parser.error("--output is required")
    try:
        parent, pastey = parse_receipts()
        payload = build_plan(parent, pastey)
        if args.output is None:
            print(json.dumps({"status": payload["status"], "parent_sha256": PARENT_SHA256,
                              "pastey_sha256": PASTEY_SHA256, "proposal_count": len(payload["proposals"]),
                              "bytes_bound": JSON_LIMIT}, sort_keys=True))
        else:
            write_plan(args.output, payload)
            print(json.dumps({"status": payload["status"], "output_sha256": sha256(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode()),
                              "proposal_count": len(payload["proposals"]), "output": str(args.output)}, sort_keys=True))
    except (OverlayError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
