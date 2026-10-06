"""Deterministic, stdlib-only tests for the supplemental metadata auditor."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import package_metadata_audit as audit


def _git_tree(entries: list[tuple[str, str, str, str]]) -> str:
    payload = []
    for name, mode, kind, sha in entries:
        key = name.encode() + (b"/" if kind == "tree" else b"")
        git_mode = "40000" if kind == "tree" else mode
        payload.append((key, f"{git_mode} ".encode() + name.encode() + b"\0" + bytes.fromhex(sha)))
    body = b"".join(item[1] for item in sorted(payload, key=lambda item: item[0]))
    return hashlib.sha1(f"tree {len(body)}\0".encode() + body).hexdigest()


def _tree_capture(files: dict[str, bytes], *, extra: dict[str, bytes] | None = None) -> tuple[dict[str, Any], str]:
    all_files = dict(files)
    if extra:
        all_files.update(extra)
    blobs = {path: audit.git_blob_sha1(body) for path, body in all_files.items()}
    dirs = {""}
    for path in all_files:
        parts = path.split("/")
        for index in range(1, len(parts)):
            dirs.add("/".join(parts[:index]))
    tree_shas: dict[str, str] = {}
    rows_by_parent: dict[str, list[tuple[str, str, str, str]]] = {path: [] for path in dirs}
    for path, sha in blobs.items():
        parent, _, name = path.rpartition("/")
        rows_by_parent[parent].append((name, "100644", "blob", sha))
    for directory in sorted((value for value in dirs if value), key=lambda value: value.count("/"), reverse=True):
        parent, _, name = directory.rpartition("/")
        tree_sha = _git_tree(rows_by_parent[directory])
        tree_shas[directory] = tree_sha
        rows_by_parent[parent].append((name, "040000", "tree", tree_sha))
    root_sha = _git_tree(rows_by_parent[""])
    rows: list[dict[str, Any]] = []
    for path, body in all_files.items():
        rows.append({"path": path, "mode": "100644", "type": "blob", "sha": blobs[path], "size": len(body)})
    for directory, sha in tree_shas.items():
        rows.append({"path": directory, "mode": "040000", "type": "tree", "sha": sha})
    rows.sort(key=lambda row: row["path"])
    return {"sha": root_sha, "truncated": False, "tree": rows}, root_sha


def _blob_capture(body: bytes) -> dict[str, Any]:
    return {
        "sha": audit.git_blob_sha1(body),
        "size": len(body),
        "encoding": "base64",
        "content": base64.b64encode(body).decode("ascii"),
    }


def _commit(revision: str, tree: str) -> dict[str, Any]:
    return {"sha": revision, "commit": {"tree": {"sha": tree}}}


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pyproject = b'''[project]\nrequires-python = ">= 3.10,<3.15"\ndependencies = [\n  "torch >= 2.2.0, < 2.10",\n  "bitsandbytes >= 0.45, < 0.50.0; sys_platform == \'linux\'",\n  "sphn >= 0.2.0, < 0.3.0",\n]\n'''
    requirements = b"torch==2.2\nsphn==0.1.4\n"
    setup_cfg = b"[metadata]\nname = moshi\n"
    files = {
        "moshi/pyproject.toml": pyproject,
        "moshi/requirements.txt": requirements,
        "moshi/setup.cfg": setup_cfg,
    }
    moshi_tree, moshi_root = _tree_capture(files)
    dsm_tree, dsm_root = _tree_capture({"scripts/run.py": b"print('not executed')\n", "README.md": b"source\n"})
    moshi_spec = {
        "repository": "example/moshi",
        "revision": "1" * 40,
        "tree": moshi_root,
        "files": {
            path: {"git_blob_sha1": audit.git_blob_sha1(body), "bytes": len(body)}
            for path, body in files.items()
        },
    }
    dsm_spec = {"repository": "example/dsm", "revision": "2" * 40, "tree": dsm_root}
    blobs = {path: _blob_capture(body) for path, body in files.items()}
    return (
        _commit(moshi_spec["revision"], moshi_root),
        moshi_tree,
        blobs,
        _commit(dsm_spec["revision"], dsm_root),
        dsm_tree,
        {"moshi": moshi_spec, "dsm": dsm_spec},
    )


class PackageMetadataAuditTests(unittest.TestCase):
    def test_success_is_distinct_source_metadata_and_preserves_unknowns(self) -> None:
        mc, mt, blobs, dc, dt, specs = _fixture()
        receipt = audit._audit_captures_from_specs(mc, mt, blobs, dc, dt, moshi_spec=specs["moshi"], dsm_spec=specs["dsm"])
        self.assertEqual(receipt["schema"], audit.SCHEMA)
        self.assertEqual(receipt["status"], "SOURCE_METADATA_ONLY")
        self.assertTrue(receipt["no_upload"])
        self.assertFalse(receipt["model_activity"])
        self.assertFalse(receipt["owner_review"])
        self.assertEqual(receipt["reviewed_dependency_closure_sha256"], "")
        self.assertEqual(receipt["packaging"]["conflicts"][0]["status"], "PACKAGING_CONSTRAINT_CONFLICT")
        self.assertEqual(receipt["source_graph_preserved"]["candidate_status"], "CANDIDATE_UNKNOWN")
        self.assertEqual(
            receipt["source_graph_preserved"]["third_party_candidates"],
            ["bitsandbytes", "einops", "huggingface_hub", "numpy", "safetensors", "sentencepiece", "torch"],
        )
        self.assertEqual(receipt["source_graph_preserved"]["dynamic_import"]["status"], "DYNAMIC_IMPORT_LITERAL_AUTHENTICATED_SOURCE")

        for key in ("execution_closure", "runtime_execution", "dependency_resolution", "model_activity", "weight_activity", "audio_activity", "owner_review", "legal_approval", "license_approval", "native_approval"):
            with self.subTest(key=key):
                self.assertFalse(receipt[key])
        self.assertTrue(receipt["no_upload"])
        self.assertEqual(receipt["reviewed_dependency_closure_sha256"], "")
        self.assertEqual(receipt["execution"], "NOT_RUN")
        self.assertEqual(receipt["runtime_origin"], "UNKNOWN")
        self.assertEqual(receipt["license_status"], "UNKNOWN_DEPENDENCY_LICENSES")
        self.assertEqual(receipt["native_payload_status"], "UNKNOWN_NATIVE_PAYLOADS")

    def test_public_fixed_audit_rejects_synthetic_spec_overrides(self) -> None:
        mc, mt, blobs, dc, dt, specs = _fixture()
        with self.assertRaises(TypeError):
            audit.audit_captures(mc, mt, blobs, dc, dt, moshi_spec=specs["moshi"])  # type: ignore[call-arg]

    def test_reconstructs_nested_tree_and_rejects_content_tamper(self) -> None:
        body = {"a/b.txt": b"one", "a/c.txt": b"two"}
        capture, root = _tree_capture(body)
        self.assertEqual(audit._validate_tree(capture, root)["a"]["type"], "tree")
        mutated = json.loads(json.dumps(capture))
        mutated["tree"][0]["sha"] = "0" * 40
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(mutated, root)

    def test_recursive_tree_matches_real_git_write_tree_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "a" / "b.txt").write_bytes(b"one\n")
            (root / "a" / "c.txt").write_bytes(b"two\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "add", "a"], check=True, capture_output=True)
            tree_sha = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
            listing = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-t", tree_sha], text=True)
            rows = []
            for line in listing.splitlines():
                metadata, path = line.split("\t", 1)
                mode, kind, object_sha = metadata.split()
                row = {"path": path, "mode": mode, "type": kind, "sha": object_sha}
                if kind == "blob":
                    row["size"] = (root / path).stat().st_size
                rows.append(row)
            capture = {"sha": tree_sha, "truncated": False, "tree": rows}
            self.assertEqual(audit._validate_tree(capture, tree_sha)["a"]["sha"], next(row["sha"] for row in rows if row["path"] == "a"))

    def test_stale_revision_duplicate_path_symlink_and_truncation_fail_closed(self) -> None:
        mc, mt, blobs, dc, dt, specs = _fixture()
        stale = dict(mc)
        stale["sha"] = "3" * 40
        with self.assertRaisesRegex(audit.CaptureError, "revision"):
            audit._audit_captures_from_specs(stale, mt, blobs, dc, dt, moshi_spec=specs["moshi"], dsm_spec=specs["dsm"])
        duplicate = json.loads(json.dumps(mt))
        duplicate["tree"].append(dict(duplicate["tree"][0]))
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(duplicate, specs["moshi"]["tree"])
        symlink = json.loads(json.dumps(mt))
        symlink["tree"][0]["mode"] = "120000"
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(symlink, specs["moshi"]["tree"])
        boolean_size = json.loads(json.dumps(mt))
        next(row for row in boolean_size["tree"] if row["type"] == "blob")["size"] = True
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(boolean_size, specs["moshi"]["tree"])
        truncated = json.loads(json.dumps(mt))
        truncated["truncated"] = True
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(truncated, specs["moshi"]["tree"])
        deleted = json.loads(json.dumps(mt))
        deleted["tree"].pop()
        deleted["truncated"] = False
        with self.assertRaises(audit.CaptureError):
            audit._validate_tree(deleted, specs["moshi"]["tree"])

        bad_blob = {path: dict(value) for path, value in blobs.items()}
        bad_blob["moshi/requirements.txt"]["content"] = base64.b64encode(b"tampered\n").decode("ascii")
        with self.assertRaisesRegex(audit.CaptureError, "content hash"):
            audit._audit_captures_from_specs(mc, mt, bad_blob, dc, dt, moshi_spec=specs["moshi"], dsm_spec=specs["dsm"])
        requirements_row = next(row for row in mt["tree"] if row["path"] == "moshi/requirements.txt")
        float_size = dict(blobs["moshi/requirements.txt"], size=195.0)
        with self.assertRaisesRegex(audit.CaptureError, "size is not an integer"):
            audit._blob_body(float_size, "moshi/requirements.txt", specs["moshi"]["files"]["moshi/requirements.txt"], requirements_row)
        boolean_blob_size = dict(blobs["moshi/requirements.txt"], size=True)
        with self.assertRaisesRegex(audit.CaptureError, "size is not an integer"):
            audit._blob_body(boolean_blob_size, "moshi/requirements.txt", specs["moshi"]["files"]["moshi/requirements.txt"], requirements_row)
        contradictory_commit = dict(mc, tree={"sha": "f" * 40})
        with self.assertRaisesRegex(audit.CaptureError, "tree mismatch"):
            audit._validate_commit(contradictory_commit, specs["moshi"])

    def test_dsm_packaging_presence_is_not_absence_evidence(self) -> None:
        mc, _mt, blobs, dc, _dt, specs = _fixture()
        dsm_tree, root = _tree_capture({"requirements.txt": b"sphn==0.1.4\n"})
        dsm_spec = dict(specs["dsm"], tree=root)
        with self.assertRaisesRegex(audit.CaptureError, "unexpectedly present"):
            audit._audit_captures_from_specs(mc, _mt, blobs, _commit(specs["dsm"]["revision"], root), dsm_tree, moshi_spec=specs["moshi"], dsm_spec=dsm_spec)

    def test_unprovided_moshi_packaging_path_is_not_silently_dropped(self) -> None:
        mc, _mt, blobs, dc, dt, specs = _fixture()
        bodies = {
            "moshi/pyproject.toml": base64.b64decode(blobs["moshi/pyproject.toml"]["content"]),
            "moshi/requirements.txt": base64.b64decode(blobs["moshi/requirements.txt"]["content"]),
            "moshi/setup.cfg": base64.b64decode(blobs["moshi/setup.cfg"]["content"]),
        }
        tree, root = _tree_capture(bodies, extra={"moshi/setup.py": b"# not captured\n"})
        with self.assertRaisesRegex(audit.CaptureError, "packaging paths"):
            audit._audit_captures_from_specs(_commit(specs["moshi"]["revision"], root), tree, blobs, dc, dt, moshi_spec=dict(specs["moshi"], tree=root), dsm_spec=specs["dsm"])

    def test_json_capture_rejects_duplicate_keys_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"sha":"a","sha":"b"}', encoding="utf-8")
            with self.assertRaises(audit.CaptureError):
                audit._json_file(duplicate)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(audit.CaptureError):
                audit._json_file(link)

            output = root / "output.json"
            output.write_text("already present", encoding="utf-8")
            with self.assertRaises(audit.CaptureError):
                audit._write_new_file(output, "new\n")
            output_link = root / "output-link.json"
            output_link.symlink_to(output)
            with self.assertRaises(audit.CaptureError):
                audit._write_new_file(output_link, "new\n")
            grown = root / "grown.json"
            grown.write_text("{}", encoding="utf-8")
            with mock.patch.object(Path, "read_bytes", return_value=b"x" * (audit.MAX_CAPTURE_BYTES + 1)):
                with self.assertRaisesRegex(audit.CaptureError, "grew beyond bounded size"):
                    audit._json_file(grown)

    def test_real_fixed_capture_positive_when_explicitly_available(self) -> None:
        capture_root = Path(os.environ.get("PACKAGE_METADATA_CAPTURE_DIR", "/private/tmp/vokra-kyutai-package-metadata-20261007"))
        required = (
            "moshi-commit.json",
            "moshi-tree.json",
            "moshi-pyproject.json",
            "moshi-requirements.json",
            "moshi-setup.cfg.json",
            "dsm-commit.json",
            "dsm-tree.json",
        )
        if not all((capture_root / name).is_file() for name in required):
            self.skipTest("authorized fixed captures are not present")
        receipt = audit.audit_paths(
            moshi_commit=capture_root / "moshi-commit.json",
            moshi_tree=capture_root / "moshi-tree.json",
            moshi_pyproject=capture_root / "moshi-pyproject.json",
            moshi_requirements=capture_root / "moshi-requirements.json",
            moshi_setup_cfg=capture_root / "moshi-setup.cfg.json",
            dsm_commit=capture_root / "dsm-commit.json",
            dsm_tree=capture_root / "dsm-tree.json",
        )
        self.assertEqual(receipt["status"], "SOURCE_METADATA_ONLY")
        self.assertEqual([row["bytes"] for row in receipt["files"]], [1305, 195, 125])
        self.assertEqual(
            [(row["path"], row["git_blob_sha1"], row["sha256"]) for row in receipt["files"]],
            [
                ("moshi/pyproject.toml", "0a99f52ea834cdcfe1b07ec3cfcbb7e96083fb61", "ae98e527d44b74ee91f00880c1058ebeedbf84eacbce5d8278039b1823750258"),
                ("moshi/requirements.txt", "89cee794c64d70932c56be7f964f23aa921dd0fa", "fb1fb99ed6f035dea8cf89f1a7f7c228658fe3b20463540a1d908820ba0105bc"),
                ("moshi/setup.cfg", "4c7f6cd5b66032ce7b1d45a779b7692ad9c1d443", "c3abfef71f3a34a33122af0e4768b3d532099a5c0ccef7680dadb0d0d19369d0"),
            ],
        )
        self.assertEqual(
            receipt["repositories"][0]["unassessed_packaging_paths"],
            [
                "moshi_mlx/pyproject.toml",
                "moshi_mlx/requirements.txt",
                "moshi_mlx/setup.cfg",
                "requirements-dev.txt",
                "rust/mimi-pyo3/pyproject.toml",
                "rust/moshi-server/pyproject.toml",
                "rust/moshi-server/uv.lock",
                "scripts/setup.cfg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
