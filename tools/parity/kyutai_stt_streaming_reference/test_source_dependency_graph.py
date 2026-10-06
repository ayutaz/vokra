from __future__ import annotations

import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import source_dependency_graph as graph


def _blob_sha(body: bytes) -> str:
    return hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()


def _make_packet(root: Path, *, tamper: str | None = None, dynamic: bool = False, diamond: bool = False) -> Path:
    packet = root / "packet"
    packet.mkdir()
    source_files = {
        "tools/parity/kyutai_stt_streaming_reference/pcm_dump.py": (
            b"import torch\nfrom moshi.models import LMGen\nfrom .missing import no\n"
            if not dynamic
            else b"import importlib\nimportlib.import_module('torch')\n"
        ),
        "tools/parity/kyutai_stt_streaming_reference/contract.py": b"from moshi.models.lm import LMModel\n",
        "moshi/moshi/__init__.py": b"from . import models\n",
        "moshi/moshi/models/__init__.py": b"from .lm import LMModel\nfrom . import loaders\n",
        "moshi/moshi/models/lm.py": b"from ..modules import transformer\n",
        "moshi/moshi/models/loaders.py": b"from ..modules import streaming\n",
        "moshi/moshi/modules/__init__.py": b"",
        "moshi/moshi/modules/transformer.py": b"import math\nfrom . import cycle_a\n",
        "moshi/moshi/modules/streaming.py": b"import io\n",
        "moshi/moshi/modules/cycle_a.py": b"from . import cycle_b\n",
        "moshi/moshi/modules/cycle_b.py": b"from . import cycle_a\n",
        "scripts/stt_from_file_pytorch.py": b"import moshi.models\n",
    }
    if diamond:
        source_files["tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"] = b"import delay\n"
        source_files["tools/parity/kyutai_stt_streaming_reference/delay.py"] = b"import moshi.models\nimport numpy\n"
        source_files["roles/role.py"] = b"import moshi.models\nimport numpy\n"
    if tamper == "body":
        source_files["moshi/moshi/modules/transformer.py"] = b"import os\n"
    names: dict[str, bytes] = {
        "dsm-commit.json": b"",
        "dsm-tree.json": b"",
        "moshi-commit.json": b"",
        "moshi-tree.json": b"",
    }
    repos = []
    for prefix, paths, commit, tree, commit_name, tree_name in (
        (
            "dsm",
            {k: v for k, v in source_files.items() if k.startswith("scripts/")},
            "1" * 40,
            "2" * 40,
            "dsm-commit.json",
            "dsm-tree.json",
        ),
        (
            "moshi",
            {k: v for k, v in source_files.items() if k.startswith("moshi/") or k.startswith("tools/") or k.startswith("roles/")},
            "3" * 40,
            "4" * 40,
            "moshi-commit.json",
            "moshi-tree.json",
        ),
    ):
        names[commit_name] = json.dumps({"sha": commit, "commit": {"tree": {"sha": tree}}}, separators=(",", ":")).encode()
        rows = []
        receipt_rows = []
        for index, (path, body) in enumerate(sorted(paths.items())):
            packet_name = f"{prefix}-{index:04d}.py"
            names[packet_name] = body
            blob = _blob_sha(body)
            rows.append({"path": path, "mode": "100644", "type": "blob", "sha": blob, "size": len(body), "url": "https://example.invalid/blob"})
            receipt_rows.append({"bytes": len(body), "git_blob_sha1": blob, "kind": "python", "path": path, "raw_url": "https://example.invalid/raw", "receipt_name": packet_name, "sha256": hashlib.sha256(body).hexdigest()})
        tree_doc = {"sha": tree, "truncated": False, "tree": rows}
        names[tree_name] = json.dumps(tree_doc, separators=(",", ":")).encode()
        repos.append({"commit_receipt_name": commit_name, "commit_sha": commit, "files": receipt_rows, "repository": f"example/{prefix}", "revision": commit, "tree_receipt_name": tree_name, "tree_sha": tree, "tree_truncated": False})
    if tamper == "tree":
        tree_doc = json.loads(names["moshi-tree.json"])
        tree_doc["tree"][0]["sha"] = "f" * 40
        names["moshi-tree.json"] = json.dumps(tree_doc, separators=(",", ":")).encode()
    receipt = {
        "schema": "synthetic",
        "status": "SOURCE_ONLY_PREPARATION",
        "repositories": repos,
        "legal_approval": "NOT_DONE",
        "owner_review": "REQUIRED",
        "publication": "NO_UPLOAD",
    }
    names["source-receipt.json"] = json.dumps(receipt, separators=(",", ":")).encode()
    if tamper == "escape":
        receipt["repositories"][0]["files"][0]["path"] = "../escape.py"
        names["source-receipt.json"] = json.dumps(receipt, separators=(",", ":")).encode()
    if tamper == "duplicate-path":
        receipt["repositories"][0]["files"].append(dict(receipt["repositories"][0]["files"][0]))
        names["source-receipt.json"] = json.dumps(receipt, separators=(",", ":")).encode()
    entries = [{"name": name, "role": "authenticated-source", "size": len(body), "sha256": hashlib.sha256(body).hexdigest()} for name, body in sorted(names.items())]
    if tamper == "duplicate-name":
        entries.append(dict(entries[0]))
    names["manifest.json"] = json.dumps({"schema": "vokra-flat-source-receipt-v1", "status": "SOURCE_ONLY_PREPARATION", "files": entries}, separators=(",", ":")).encode()
    for name, body in names.items():
        (packet / name).write_bytes(body)
    return packet


class SourceDependencyGraphTests(unittest.TestCase):
    def test_graph_resolves_authenticated_source_and_external_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            packet = _make_packet(Path(raw))
            report = graph.build_graph(packet, ["vokra_pcm_dump=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)
        self.assertEqual(report["status"], graph.GRAPH_STATUS)
        self.assertEqual(report["execution_status"], graph.EXECUTION_STATUS)
        self.assertIn("torch", {row["name"] for row in report["external_candidates"]})
        modules = {row["module"] for row in report["authenticated_modules"]}
        self.assertIn("moshi.models", modules)
        self.assertIn("moshi.models.lm", modules)
        self.assertIn("moshi.models.loaders", modules)
        self.assertIn("moshi.modules.transformer", modules)
        self.assertIn("moshi.modules.streaming", modules)
        self.assertIn("moshi.modules.cycle_a", modules)
        self.assertIn("moshi.modules.cycle_b", modules)
        self.assertIn("moshi.models", {row["target"] for row in report["edges"] if row["target"]})
        self.assertEqual(len(report["graph_sha256"]), 64)

    def test_missing_root_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = graph.build_graph(_make_packet(Path(raw)), ["pcm=missing/pcm_dump.py"], synthetic=True)
        self.assertEqual(report["closure_status"], graph.REVIEW_STATUS)
        self.assertTrue(any(row["status"] == "MISSING_AUTHENTICATED_SOURCE" for row in report["unresolved"]))

    def test_dynamic_import_is_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = graph.build_graph(_make_packet(Path(raw), dynamic=True), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)
        self.assertTrue(any(row["status"] == "DYNAMIC_IMPORT_UNKNOWN" for row in report["unresolved"]))

    def test_relative_edge_and_cycle_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = graph.build_graph(_make_packet(Path(raw)), ["moshi.models.lm=moshi/moshi/models/lm.py"], synthetic=True)
        modules = {row["module"] for row in report["authenticated_modules"]}
        self.assertIn("moshi", modules)
        self.assertIn("moshi.models", modules)
        self.assertIn("moshi.modules", {edge["target"] for edge in report["edges"] if edge["target"]})
        self.assertIn("moshi.modules.transformer", {edge["target"] for edge in report["edges"] if edge["target"]})
        self.assertLessEqual(len(report["authenticated_modules"]), graph.MAX_GRAPH_MODULES)

    def test_plain_import_missing_internal_module_is_not_collapsed_to_symbol(self) -> None:
        target, resolution = graph._edge_target(
            "moshi.models",
            {"name": "moshi.does_not_exist", "kind": "import", "relative_level": 0},
            {"moshi": {"is_package": True}},
        )
        self.assertEqual(target, "moshi.does_not_exist")
        self.assertEqual(resolution, "missing-authenticated-source")

    def test_from_import_missing_internal_module_is_unresolved(self) -> None:
        target, resolution = graph._edge_target(
            "fixture",
            {"name": "moshi.does_not_exist", "kind": "from-import", "relative_level": 0},
            {"moshi": {"is_package": True}},
        )
        self.assertEqual(target, "moshi.does_not_exist")
        self.assertEqual(resolution, "missing-authenticated-source")

    def test_relative_import_beyond_package_root_is_blocked(self) -> None:
        self.assertEqual(
            graph._resolve_relative("moshi", "missing", 2, source_is_package=True),
            "<invalid-relative-import>",
        )

    def test_conditional_and_optional_handler_imports_keep_branch_labels(self) -> None:
        tree = ast.parse(
            "if FLAG:\n    import one\nelse:\n    import two\n"
            "try:\n    import three\nexcept ImportError:\n    import four\n"
            "else:\n    import five\nfinally:\n    import six\n"
        )
        visitor = graph._ImportVisitor("fixture")
        visitor.visit(tree)
        labels = {edge["name"]: edge["condition"] for edge in visitor.edges}
        self.assertEqual(labels["one"], "conditional")
        self.assertEqual(labels["two"], "conditional-else")
        self.assertEqual(labels["three"], "optional-import")
        self.assertEqual(labels["four"], "optional-import-handler")
        self.assertEqual(labels["five"], "try-else")
        self.assertEqual(labels["six"], "try-finally")

    def test_if_test_dynamic_import_is_unknown_and_keeps_conditional_label(self) -> None:
        tree = ast.parse(
            "if importlib.import_module('optional_backend'):\n"
            "    import branch\n"
        )
        visitor = graph._ImportVisitor("fixture")
        visitor.visit(tree)
        dynamic = [edge for edge in visitor.edges if edge["name"] == "<dynamic-import>"]
        self.assertEqual(len(dynamic), 1)
        self.assertEqual(dynamic[0]["kind"], "dynamic-import")
        self.assertEqual(dynamic[0]["condition"], "conditional")
        self.assertEqual(graph._edge_target("fixture", dynamic[0], {}), (None, "dynamic-import-unknown"))

    def test_except_type_dynamic_import_is_unknown_and_keeps_handler_label(self) -> None:
        tree = ast.parse(
            "try:\n"
            "    import body\n"
            "except importlib.import_module('optional_backend'):\n"
            "    import handler\n"
        )
        visitor = graph._ImportVisitor("fixture")
        visitor.visit(tree)
        dynamic = [edge for edge in visitor.edges if edge["name"] == "<dynamic-import>"]
        self.assertEqual(len(dynamic), 1)
        self.assertEqual(dynamic[0]["kind"], "dynamic-import")
        self.assertEqual(dynamic[0]["condition"], "try-block-handler")
        self.assertEqual(graph._edge_target("fixture", dynamic[0], {}), (None, "dynamic-import-unknown"))
        labels = {edge["name"]: edge["condition"] for edge in visitor.edges}
        self.assertEqual(labels["body"], "try-block")
        self.assertEqual(labels["handler"], "try-block-handler")

    def test_scope_fixed_point_unions_role_and_producer_diamond(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            packet = _make_packet(Path(raw), diamond=True)
            fixed_packet = {
                "status": "AUTHENTICATED_FIXED_PCM_SOURCE_PACKET",
                "manifest_sha256": "a" * 64,
                "dsm_tree_sha": "b" * 40,
                "moshi_tree_sha": "c" * 40,
                "role_contract": "synthetic-test",
            }
            fake_vokra = {"status": "AUTHENTICATED_VOKRA_GIT_SOURCE"}
            fake_rows = {"local/contract.py": {"body": b"synthetic contract"}}
            with mock.patch.object(graph, "_authenticate_upstream_packet", return_value=fixed_packet), \
                mock.patch.object(graph, "_authenticate_vokra_sources", return_value=(fake_vokra, fake_rows)), \
                mock.patch.object(graph, "UPSTREAM_ROLE_ROOTS", (("role", "roles/role.py"),)), \
                mock.patch.object(graph, "VOKRA_SOURCE_PATHS", ("local/pcm.py", "local/dump.py", "local/contract.py")):
                report = graph.build_graph(
                    packet,
                    ["pcm_dump=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"],
                    vokra_root=Path(raw),
                )
        shared = next(row for row in report["authenticated_modules"] if row["module"] == "moshi.models")
        self.assertEqual(set(shared["reachability_scopes"]), {"producer", "source-contract-role"})
        numpy = next(row for row in report["external_candidates"] if row["name"] == "numpy")
        self.assertEqual(set(numpy["reachability_scopes"]), {"producer", "source-contract-role"})

    def test_manifest_body_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            packet = _make_packet(Path(raw))
            (packet / "moshi-0000.py").write_bytes(b"tampered")
            with self.assertRaises(graph.GraphError):
                graph.build_graph(packet, ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)

    def test_tree_blob_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(graph.GraphError):
                graph.build_graph(_make_packet(Path(raw), tamper="tree"), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)

    def test_source_path_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(graph.GraphError):
                graph.build_graph(_make_packet(Path(raw), tamper="escape"), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)

    def test_duplicate_authenticated_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(graph.GraphError):
                graph.build_graph(_make_packet(Path(raw), tamper="duplicate-path"), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)

    def test_duplicate_manifest_name_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(graph.GraphError):
                graph.build_graph(_make_packet(Path(raw), tamper="duplicate-name"), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)

    def test_raw_dot_path_rejected(self) -> None:
        with self.assertRaises(graph.GraphError):
            graph._safe_rel("moshi//models.py", "synthetic path")

    def test_synthetic_receipt_cannot_claim_fixed_production_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(graph.GraphError):
                graph.build_graph(_make_packet(Path(raw)), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"])

    def test_swapped_fixed_packet_identity_is_rejected(self) -> None:
        class SwappedContract:
            PCM_SOURCE_PACKET_MANIFEST_SHA256 = "0" * 64

            @staticmethod
            def require_pcm_source_packet(_packet: Path) -> dict[str, str]:
                return {"manifest_sha256": "1" * 64}

        with tempfile.TemporaryDirectory() as raw:
            packet = _make_packet(Path(raw))
            with mock.patch.object(graph, "_contract_module", return_value=SwappedContract):
                with self.assertRaises(graph.GraphError):
                    graph._authenticate_upstream_packet(packet)

    def test_packet_validator_runs_only_after_authenticated_contract_bytes(self) -> None:
        class TrustedContract:
            PCM_SOURCE_PACKET_MANIFEST_SHA256 = "a" * 64

            @staticmethod
            def require_pcm_source_packet(_packet: Path) -> dict[str, str]:
                return {
                    "manifest_sha256": "a" * 64,
                    "dsm_tree_sha": "b" * 40,
                    "moshi_tree_sha": "c" * 40,
                }

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract_path = root / graph.VOKRA_SOURCE_PATHS[2]
            contract_path.parent.mkdir(parents=True)
            body = b"# authenticated contract fixture\n"
            contract_path.write_bytes(body)
            TrustedContract.__file__ = str(contract_path)
            with mock.patch.object(graph, "_contract_module", return_value=TrustedContract):
                result = graph._authenticate_upstream_packet(
                    root,
                    trusted_contract={"body": body},
                    vokra_root=root,
                )
                self.assertEqual(result["status"], "AUTHENTICATED_FIXED_PCM_SOURCE_PACKET")
                contract_path.write_bytes(b"# swapped contract fixture\n")
                with self.assertRaises(graph.GraphError):
                    graph._authenticate_upstream_packet(
                        root,
                        trusted_contract={"body": body},
                        vokra_root=root,
                    )

    def test_contract_loader_uses_selected_checkout_not_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selected = root / "selected-contract.py"
            neighbor = root / "neighbor-contract.py"
            selected_body = b"MARKER = 'selected'\n"
            selected.write_bytes(selected_body)
            neighbor.write_bytes(b"MARKER = 'neighbor'\n")
            cache = selected.parent / "__pycache__"
            cache.mkdir()
            (cache / "selected-contract.cpython-312.pyc").write_bytes(b"corrupt adjacent cache")
            loaded = graph._contract_module(selected, selected_body)
            self.assertEqual(loaded.MARKER, "selected")
            selected.write_bytes(neighbor.read_bytes())
            self.assertEqual(loaded.MARKER, "selected")

    def test_vokra_git_root_authenticates_fixed_head_and_rejects_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkout = Path(raw) / "checkout"
            for source_path in graph.VOKRA_SOURCE_PATHS:
                path = checkout / source_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((f"# {source_path}\n").encode())
            commands = [
                ["git", "init", "-q"],
                ["git", "config", "user.email", "graph-test@example.invalid"],
                ["git", "config", "user.name", "graph-test"],
                ["git", "remote", "add", "origin", graph.VOKRA_REPOSITORY],
                ["git", "add", *graph.VOKRA_SOURCE_PATHS],
                ["git", "commit", "-qm", "source graph fixture"],
            ]
            for command in commands:
                subprocess.run(command, cwd=checkout, check=True, capture_output=True)
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
            ).stdout.strip()
            auth, rows = graph._authenticate_vokra_sources(checkout, head)
            self.assertEqual(auth["status"], "AUTHENTICATED_VOKRA_GIT_SOURCE")
            self.assertEqual(set(rows), set(graph.VOKRA_SOURCE_PATHS))
            with self.assertRaises(graph.GraphError):
                graph._authenticate_vokra_sources(checkout, "0" * 40)
            (checkout / graph.VOKRA_SOURCE_PATHS[0]).write_bytes(b"# changed\n")
            with self.assertRaises(graph.GraphError):
                graph._authenticate_vokra_sources(checkout, head)

    def test_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "report.json"
            graph._write_exclusive(output, b"{}")
            with self.assertRaises(graph.GraphError):
                graph._write_exclusive(output, b"new")

    def test_output_symlink_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "target.json"
            output = root / "report.json"
            target.write_bytes(b"keep")
            os.symlink(target, output)
            with self.assertRaises(graph.GraphError):
                graph._write_exclusive(output, b"replace")
            self.assertEqual(target.read_bytes(), b"keep")

    def test_packet_manifest_is_stable_and_no_execution_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report_a = graph.build_graph(_make_packet(Path(raw)), ["pcm=tools/parity/kyutai_stt_streaming_reference/pcm_dump.py"], synthetic=True)
            self.assertEqual(report_a["publication"], "NO_UPLOAD")
            self.assertEqual(report_a["license_status"], "UNKNOWN_DEPENDENCY_LICENSES")
            self.assertEqual(report_a["native_status"], "UNKNOWN_NATIVE_PAYLOADS")
            self.assertEqual(report_a["source_receipt_status"], "SYNTHETIC_TEST_INPUT_NOT_AUTHENTICATED")
            self.assertTrue(all(row["source_kind"] == "synthetic-test-input" for row in report_a["authenticated_modules"]))
            self.assertNotIn("AUTHENTICATED_SOURCE_RECEIPT", json.dumps(report_a))
            self.assertNotIn("APPROVED", json.dumps(report_a))


if __name__ == "__main__":
    unittest.main()
