#!/usr/bin/env python3
"""Offline, fail-closed license gate for the Dia reference environment.

The captured VAST report is evidence, not an approval.  This gate binds the
active Linux lock closure and every package's captured metadata/native facts
to immutable digests.  It never installs, imports, acquires, or executes a
model and cannot create an owner decision.
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
import tomllib

GATE_VERSION = 1
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPORT_SHA256 = "8ce645073916f8f1148ed572b476653fc602a2236fbeee90183b0b01b1c57cd8"
OWNER_SCOPE_FILE_SHA256 = "0d51a44f9494313d01b6cb0314d35215c48c311c0c071d8516a10617a2f224e3"
OWNER_SCOPE_SHA256 = "a5c90a401c7edeb39cb8036fab789ad82a117798d41101b3c60f23f6f3b97668"
EXPECTED_HEAD = "87da78dc7709075d9dc23b797fc978b9c678c777"
LOCK_SHA256 = "58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
PROJECT_SHA256 = "fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
ACTIVE_ROWS_SHA256 = "fe967f78eab4210df3586cd0166b7ad0a0705385aae22216cb381f23c50563e5"
ALL_ROWS_SHA256 = "f5f4e5320901463d6393bb8cc1c2561538b3dd41089d9f09207379496278c13b"
NATIVE_INVENTORY_SHA256 = "df48dac02d5f1f2803b8789b1759067ec3e1b6af67418b1d5ea82fcd4541643f"
PREPARATION_SHA256 = "1d45fb286a44a4903fc3508c43ad37fd2fac54d5a2d81b94b21086e283e6e019"

EXPECTED_FACTS = {
    "annotated-types@0.8.0": ("3d20afd6f216c3bb0c6668660434b5d0728f26d8ee092e442c0c3c4b38ee3f75", "MIT", "1", "0", "0"),
    "certifi@2026.7.22": ("32021a862de99411d36f71f52a8243e771770d599f30e151ca7bf57f5420ed92", "MPL-2.0", "1", "0", "0"),
    "charset-normalizer@3.5.1": ("9fde0575b4dc71668bdc2f65b1f331d0935f69625da895822b7bca3cf9a85a29", "MIT", "1", "2", "2"),
    "einops@0.8.2": ("49f45647cde9493df8f9ebff79fbe6ce968a1e714b9df18183ea1addd85b8721", "MIT", "1", "0", "0"),
    "filelock@3.32.4": ("269d8e61cae91df75de2a83c184a8b16b82a9d95f0500eff46740a22e8edb5d4", "MIT", "1", "0", "0"),
    "fsspec@2026.7.0": ("61dc2c720bd3867aa26e370e08e41f6402d27353211d84ac48091ffae7b39314", "BSD-3-Clause", "1", "0", "0"),
    "gguf@0.19.0": ("22bd6d57ca7f45f64d500a593bd6a16a033f856b8778a1964a083dd42a917a1e", "MIT", "1", "0", "0"),
    "huggingface-hub@0.30.2": ("54eefb61b082b08ddfe7f9c46290e8c437b8e748b37629a7817a6fa7bff5c618", "Apache-2.0", "1", "0", "0"),
    "idna@3.19": ("8c5f340dc7f0db84b44e26604fb8fcf67b7d54a359fedec98bc037cdbcba7271", "BSD-3-Clause", "1", "0", "0"),
    "jinja2@3.1.6": ("70fef3bad0ce6e0f2296ec751d80201f4d13594265f0d4125e6364b6ba3ab21e", "BSD-3-Clause", "1", "0", "0"),
    "markupsafe@3.0.3": ("563d6789527c6b18ed49e3f449926776c8f94b54095185ce1b3541efae6b2f0f", "BSD-3-Clause", "1", "1", "1"),
    "mpmath@1.3.0": ("fca2c48bd768039e407ba6d68d944974cd8358ed83832dbbf714ac53a4963693", "BSD", "1", "0", "0"),
    "networkx@3.6.1": ("b6edce1d1e81f508b56b8c768be10f64c0c79c3c4e515b7d14af02345fa4bf34", "BSD-3-Clause", "1", "0", "0"),
    "numpy@2.2.5": ("9c0a86a7efac5d1368ed6b11e1f380f2228a836d5b6cbf615bab35d4a10aa110", "BSD-3-Clause plus zlib/NCSA bundled notices", "4", "21", "21"),
    "packaging@26.3": ("e98dac4a0316d88b6515185cef22106ef780a8411d34387dd9d90279e01a06fa", "Apache-2.0 OR BSD-2-Clause", "3", "0", "0"),
    "pydantic@2.11.3": ("80c96f9c72c85f82d6e8e63d6558b922df7ad22a536af06c5c47f4b2eafeed0b", "MIT", "1", "0", "0"),
    "pydantic-core@2.33.1": ("dae7e23ba2bcb03f67dd043d06c7dab3d8e4647d323f53821574eefa3d54cc13", "MIT", "1", "1", "1"),
    "pyyaml@6.0.3": ("7c7a366ae846d0745061f7e77e26d81769ee92ab3eaf7dc88e237326c575a7a6", "MIT", "1", "1", "1"),
    "requests@2.34.2": ("ba82d533310f21aaa3e20446dad7430e5950f74804ce372194e9ffe936190425", "Apache-2.0", "2", "0", "0"),
    "setuptools@84.0.0": ("54434b779d0466451697468abfd200b9d3f1b2222066aade7efe9a42f72047cd", "MIT; vendored LGPL-3.0 and MPL/other notice material requires separate review", "17", "0", "0"),
    "sympy@1.13.1": ("e58ecea2c81e1f46599f728ef30a1e2b9267a343049193d2d764cd84e02b2a95", "BSD plus MIT latex2sympy notice", "2", "0", "0"),
    "torch@2.6.0+cpu": ("6c062b2635f617752b883898b36a58dc89ad34bc3928560e6b4ccb6e7e9f1019", "BSD-3-Clause; NOTICE/native third-party closure requires review", "2", "133", "133"),
    "tqdm@4.70.0": ("14e20af6bc89cb4f23458fd9da8ace69ce3f8ed4361a6fc78aa94f46b7226d66", "MPL-2.0 AND MIT", "1", "0", "0"),
    "typing-extensions@4.16.0": ("4d387755e825794d9871d853bfd0ad05f83b79a3c82ee44f9fd49a0fdd4bab04", "PSF-2.0", "1", "0", "0"),
    "typing-inspection@0.4.4": ("e76c43195e229aaa7d63abd1e70a133c3a3ba242a47ffa9a0e52b5300f16b277", "MIT", "1", "0", "0"),
    "urllib3@2.7.0": ("fab60fb2b94b24ee24534d193afb2bbeca7f4b42bc1df04655336993d3faee68", "MIT", "1", "0", "0"),
}
EXPECTED_LICENSE_ROWS = {
    "python-dependency-closure": ("FACTS_CAPTURED_OWNER_REVIEW_REQUIRED", "Captured metadata includes MIT, BSD, Apache-2.0, PSF-2.0, MPL-2.0, and bundled notices; classification is not performed.", "uv.lock and VAST dependency-audit.json"),
    "native-bundled-payloads": ("FACTS_CAPTURED_OWNER_REVIEW_REQUIRED", "159 native/bundled files are captured; NumPy/Torch and native package attribution remains owner/legal review.", "VAST native_inventory"),
    "dia-source-license": ("NOT_ACQUIRED_OWNER_REVIEW_REQUIRED", "NOT_ASSESSED; source acquisition was NONE", "VAST policy.model_source_facts"),
    "dia-model-weight": ("NOT_ACQUIRED_OWNER_REVIEW_REQUIRED", "NOT_ASSESSED; model acquisition was NONE", "VAST policy.model_source_facts"),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def block(message: str) -> None:
    print(f"dia license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)


def active_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    if set(lock) != {"version", "revision", "requires-python", "resolution-markers", "package"}:
        block("uv.lock schema drifted")
    if lock.get("version") != 1 or lock.get("revision") != 3 or lock.get("requires-python") != "==3.12.*":
        block("uv.lock resolver identity drifted")
    rows = []
    for row in lock.get("package", []):
        if row.get("source") == {"virtual": "."}:
            continue
        if row.get("name") == "colorama" or (row.get("name") == "torch" and row.get("version") == "2.6.0"):
            continue
        rows.append({"dependencies": row.get("dependencies", []), "name": row["name"], "resolution_markers": row.get("resolution-markers", []), "source": row["source"], "version": row["version"]})
    rows.sort(key=lambda row: (row["name"], row["version"]))
    if len(rows) != 26 or digest(rows) != ACTIVE_ROWS_SHA256:
        block("active Linux lock rows are not the captured 26-row closure")
    return rows


def validate_manifest(lock_path: Path, project_path: Path, manifest_path: Path) -> dict[str, Any]:
    for path in (lock_path, project_path, manifest_path):
        if path.is_symlink() or not path.is_file():
            block(f"non-regular closure input: {path}")
    try:
        lock_bytes = lock_path.read_bytes(); project_bytes = project_path.read_bytes(); manifest = load_json(manifest_path)
        lock = tomllib.loads(lock_bytes.decode()); project = tomllib.loads(project_bytes.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        block(f"closure input is unreadable: {exc}")
    expected_keys = {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "license_rows", "license_rows_sha256", "identities", "audit_evidence", "approval_scope_sha256", "approval", "publication"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys or manifest.get("gate_version") != GATE_VERSION:
        block("gate manifest schema/version drifted")
    if hashlib.sha256(lock_bytes).hexdigest() != LOCK_SHA256 or manifest["lock_sha256"] != LOCK_SHA256:
        block("uv.lock is not the captured lock")
    if hashlib.sha256(project_bytes).hexdigest() != PROJECT_SHA256 or manifest["project_sha256"] != PROJECT_SHA256:
        block("pyproject.toml is not the captured project")
    if project.get("project", {}).get("name") != "vokra-dia-1-6b-reference" or project.get("project", {}).get("version") != "0.1.0":
        block("project identity drifted")
    rows = active_rows(lock)
    review_rows = manifest["package_review_rows"]
    ids = [f"{row['name']}@{row['version']}" for row in rows]
    if not isinstance(review_rows, list) or len(review_rows) != len(rows) or digest(review_rows) != manifest["package_review_rows_sha256"]:
        block("package review set/hash is malformed")
    review_ids = [item.get("id") for item in review_rows]
    if sorted(review_ids) != sorted(ids) or len(set(review_ids)) != len(ids):
        block("package review identities do not match the exact active lock set")
    for item in review_rows:
        if set(item) != {"id", "status", "license", "native_bundled_review", "evidence"} or item["status"] != "REVIEWED_FACTS_OWNER_APPROVAL_REQUIRED":
            block(f"package review status/schema is unresolved: {item.get('id')!r}")
        expected = EXPECTED_FACTS.get(item["id"])
        ev = item.get("evidence")
        if expected is None or not isinstance(ev, dict) or set(ev) != {"facts_sha256", "publisher_file_count", "native_file_count", "bundled_library_count"} or tuple((ev["facts_sha256"], item["license"], str(ev["publisher_file_count"]), str(ev["native_file_count"]), str(ev["bundled_library_count"]))) != expected or not isinstance(item["native_bundled_review"], str) or not item["native_bundled_review"].strip():
            block(f"package evidence identity/hash drifted: {item.get('id')!r}")
    if manifest["package_rows_sha256"] != ACTIVE_ROWS_SHA256:
        block("package row digest is not the captured closure")
    audit = manifest["audit_evidence"]
    expected_audit = {"schema": "vokra-dia-dependency-audit-v1", "report_sha256": REPORT_SHA256, "owner_scope_file_sha256": OWNER_SCOPE_FILE_SHA256, "owner_scope_sha256": OWNER_SCOPE_SHA256, "expected_head": EXPECTED_HEAD, "package_count": 26, "native_file_count": 159, "publisher_evidence_entries": 50, "status": "FACTS_COLLECTED_GATE_BLOCKED", "dependency_license_audit": "BLOCKED_UNREVIEWED_TRANSITIVE", "publication": "NO_UPLOAD", "all_lock_rows_sha256": ALL_ROWS_SHA256, "native_inventory_sha256": NATIVE_INVENTORY_SHA256, "preparation_sha256": PREPARATION_SHA256}
    if audit != expected_audit:
        block("VAST evidence identity/counts drifted")
    identities = manifest["identities"]
    if identities != {"source_acquisition": "NONE", "model_acquisition": "NONE", "model_import": "NOT_PERFORMED", "execution": "NOT_PERFORMED", "torch_imported": False, "vokra_imported": False, "license_classification": "NOT_PERFORMED", "training_provenance": "OWNER_REVIEW_REQUIRED"}:
        block("source/model execution boundary drifted")
    license_rows = manifest["license_rows"]
    if not isinstance(license_rows, list) or [row.get("id") for row in license_rows] != list(EXPECTED_LICENSE_ROWS) or digest(license_rows) != manifest["license_rows_sha256"]:
        block("license disposition row set/hash is malformed")
    for row in license_rows:
        expected = EXPECTED_LICENSE_ROWS[row["id"]]
        if set(row) != {"id", "status", "license", "source", "payload_sha256"} or tuple(row[key] for key in ("status", "license", "source")) != expected or not HEX64.fullmatch(row["payload_sha256"]):
            block(f"license disposition is unresolved or tampered: {row.get('id')!r}")
    if manifest["publication"] != "NO_UPLOAD" or manifest["approval"] != {"status": "PENDING_OWNER_REVIEW", "signer": None, "digest": None}:
        block("owner/publication boundary is not fail-closed")
    scope = {"lock_sha256": manifest["lock_sha256"], "project_sha256": manifest["project_sha256"], "package_rows_sha256": manifest["package_rows_sha256"], "package_review_rows_sha256": manifest["package_review_rows_sha256"], "license_rows_sha256": manifest["license_rows_sha256"], "audit_evidence": audit, "identities": identities, "publication": manifest["publication"]}
    if manifest["approval_scope_sha256"] != digest(scope):
        block("approval scope is not bound to all evidence")
    return manifest


def run(lock_path: Path, project_path: Path, manifest_path: Path) -> None:
    validate_manifest(lock_path, project_path, manifest_path)
    block("owner/legal review is pending; publication remains NO_UPLOAD")


def self_test() -> None:
    root = Path(__file__).resolve().parent
    validate_manifest(root / "uv.lock", root / "pyproject.toml", root / "license_gate_manifest.json")
    production = load_json(root / "license_gate_manifest.json")
    with tempfile.TemporaryDirectory(prefix="dia-license-gate-") as directory:
        path = Path(directory) / "manifest.json"
        mutations = (
            lambda value: value["package_review_rows"][0]["evidence"].update(facts_sha256="0" * 64),
            lambda value: value["audit_evidence"].update(report_sha256="0" * 64),
            lambda value: value["approval"].update(status="OWNER_SIGNOFF_APPROVED"),
        )
        for mutate in mutations:
            candidate = json.loads(json.dumps(production)); mutate(candidate); path.write_text(json.dumps(candidate), encoding="utf-8")
            try:
                validate_manifest(root / "uv.lock", root / "pyproject.toml", path)
            except SystemExit as exc:
                if exc.code != 2: raise
            else:
                raise AssertionError("tampered Dia manifest was accepted")
        path.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
        try:
            validate_manifest(root / "uv.lock", root / "pyproject.toml", path)
        except SystemExit as exc:
            if exc.code != 2: raise
        else:
            raise AssertionError("duplicate JSON key was accepted")
    print("dia_1_6b_reference license_gate self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--lock", type=Path); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if any(value is None for value in (args.lock, args.project, args.manifest)): parser.error("--lock, --project, and --manifest are required")
    run(args.lock, args.project, args.manifest); return 0


if __name__ == "__main__":
    raise SystemExit(main())
