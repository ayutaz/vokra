#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Stdlib-only tests for the XCodec2 model-free evidence contract."""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

import collect_dependency_evidence as collector
import dependency_audit as audit
import dump_reference as reference


class DependencyAuditTests(unittest.TestCase):
    def _fake_runtime(self, root: Path):
        """Build real tiny distributions for the shared pre-import gate."""

        packages = {}
        source_bytes = {
            "xcodec2/vq/codec_decoder_vocos.py": b"decoder-source\n",
            "xcodec2/vq/bs_roformer5.py": b"transformer-source\n",
            "torchtune/modules/position_embeddings.py": b"rope-source\n",
        }
        for name, version in reference.EXPECTED_DISTRIBUTIONS.items():
            package_file = reference.EXPECTED_PACKAGE_FILES[name]
            dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
            files = {package_file: b"# synthetic package\n"}
            if name == "xcodec2":
                files.update({path: value for path, value in source_bytes.items() if path.startswith("xcodec2/")})
            if name == "torchtune":
                files["torchtune/modules/position_embeddings.py"] = source_bytes["torchtune/modules/position_embeddings.py"]
            if name == "torch":
                files["torch/libtiny.so"] = b"\x7fELFsynthetic"
            for relative, value in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            record_path = f"{dist_info}/RECORD"
            record = []
            for relative, value in sorted(files.items()):
                digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
                record.append(f"{relative},sha256={digest},{len(value)}")
            record.append(f"{record_path},,")
            record_file = root / record_path
            record_file.parent.mkdir(parents=True, exist_ok=True)
            record_file.write_text("\n".join(record) + "\n", encoding="utf-8")
            packages[name] = type(
                "FakeDistribution",
                (),
                {
                    "metadata": {"Name": name},
                    "version": version,
                    "files": [*files, record_path],
                    "locate_file": lambda self, relative, base=root: base / str(relative),
                },
            )()
        return packages, source_bytes

    def _with_fake_runtime(self, root: Path):
        packages, source_bytes = self._fake_runtime(root)
        original_distribution = reference.importlib.metadata.distribution
        original_constants = (
            reference.DECODER_SOURCE_SHA256,
            reference.TRANSFORMER_SOURCE_SHA256,
            reference.TORCHTUNE_ROPE_SHA256,
        )
        reference.importlib.metadata.distribution = lambda name: packages[name]
        reference.DECODER_SOURCE_SHA256 = hashlib.sha256(source_bytes["xcodec2/vq/codec_decoder_vocos.py"]).hexdigest()
        reference.TRANSFORMER_SOURCE_SHA256 = hashlib.sha256(source_bytes["xcodec2/vq/bs_roformer5.py"]).hexdigest()
        reference.TORCHTUNE_ROPE_SHA256 = hashlib.sha256(source_bytes["torchtune/modules/position_embeddings.py"]).hexdigest()
        return packages, original_distribution, original_constants

    def _restore_fake_runtime(self, original_distribution, original_constants):
        reference.importlib.metadata.distribution = original_distribution
        (
            reference.DECODER_SOURCE_SHA256,
            reference.TRANSFORMER_SOURCE_SHA256,
            reference.TORCHTUNE_ROPE_SHA256,
        ) = original_constants

    def test_linux_closure_and_fail_closed_documents(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = audit.reachable_lock_identities(lock)
        self.assertEqual(len(closure), 63)
        self.assertEqual(len(closure) - 1, 62)
        manifest = audit.strict_json(audit.MANIFEST)
        rows = audit.strict_json(audit.ROWS)
        self.assertEqual(manifest["publication"], "NO_UPLOAD")
        self.assertEqual(rows["status"], "BLOCKED_PENDING_PRIMARY_BYTES")

    def test_marker_target_is_linux_not_macos(self) -> None:
        self.assertTrue(audit.marker_reaches("sys_platform != 'darwin' and platform_machine == 'x86_64'"))
        self.assertFalse(audit.marker_reaches("sys_platform == 'darwin'"))
        for version, expected in (("3.8", False), ("3.9", True), ("3.10", True), ("3.12", True), ("3.13", True)):
            environment = dict(audit.TARGET_ENV)
            environment["python_version"] = version
            environment["python_full_version"] = version + ".0"
            self.assertEqual(audit.marker_reaches("python_version >= '3.9'", environment), expected)
            self.assertEqual(audit.marker_reaches("python_full_version < '3.10.0'", environment), version == "3.8" or version == "3.9")
        self.assertTrue(audit.marker_reaches("python_full_version == '3.12'", {**audit.TARGET_ENV, "python_full_version": "3.12.0"}))
        self.assertTrue(audit.marker_reaches("python_version >= '3.12.0'", {**audit.TARGET_ENV, "python_version": "3.12"}))

    def test_artifact_urls_and_dependency_edges_are_strictly_bound(self) -> None:
        for url in (
            "https://files.pythonhosted.org/packages/pkg.whl",
            "https://files.pythonhosted.org:443/packages/pkg.whl",
        ):
            audit._artifact_url_ok(url, torch=False)
        for url in (
            "http://files.pythonhosted.org/packages/pkg.whl",
            "https://files.pythonhosted.org:444/packages/pkg.whl",
            "https://files.pythonhosted.org/packages/pkg.whl?redirect=1",
            "https://files.pythonhosted.org",
            "https://evil.example/packages/pkg.whl",
            "https://@files.pythonhosted.org/packages/pkg.whl",
            "https://files.pythonhosted.org/packages/pkg.whl\n",
        ):
            with self.subTest(url=url), self.assertRaises(audit.AuditError, msg=url):
                audit._artifact_url_ok(url, torch=False)

        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        xcodec = next(row for row in lock["package"] if row.get("name") == "xcodec2")
        for edge in xcodec["dependencies"]:
            if edge.get("name") == "torch":
                edge["version"] = "2.5.0"
        with self.assertRaises(audit.AuditError):
            audit.reachable_lock_identities(lock)

    def test_activated_extra_reaches_aiohttp_subgraph(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = set(audit.reachable_lock_identities(lock))
        self.assertIn(("aiohttp", "3.14.3"), closure)
        self.assertIn(("yarl", "1.24.5"), closure)
        self.assertNotIn(("torch", "2.13.0"), closure)
        first = audit.reachable_lock_identities(lock)
        second = audit.reachable_lock_identities(lock)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_manifest_policy_and_rows_hash_tampering_is_rejected(self) -> None:
        lock = tomllib.loads(audit.LOCK.read_text(encoding="utf-8"))
        closure = audit.reachable_lock_identities(lock)
        manifest = audit.strict_json(audit.MANIFEST)
        rows = audit.strict_json(audit.ROWS)
        actual = {
            "pyproject_sha256": audit.sha256_file(audit.PYPROJECT),
            "uv_lock_sha256": audit.sha256_file(audit.LOCK),
            "dependency_audit_sha256": audit.sha256_file(audit.ROWS),
            "source_digests": audit.source_digest_map(),
        }
        policy_tampered = copy.deepcopy(manifest)
        policy_tampered["policy"]["automatic_exceptions"] = True
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(policy_tampered, rows, actual, closure)
        hash_tampered = dict(actual)
        hash_tampered["dependency_audit_sha256"] = "0" * 64
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(manifest, rows, hash_tampered, closure)
        source_tampered = copy.deepcopy(actual)
        source_tampered["source_digests"]["collect_dependency_evidence.py"] = "0" * 64
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(manifest, rows, source_tampered, closure)
        devendor_tampered = copy.deepcopy(manifest)
        devendor_tampered["setuptools_devendoring"]["upstream_contract"]["vendor_root"] = "setuptools/vendor"
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(devendor_tampered, rows, actual, closure)
        rows_tampered = copy.deepcopy(rows)
        rows_tampered["setuptools_devendoring"]["status"] = "APPROVED"
        with self.assertRaises(audit.AuditError):
            audit.validate_gate_documents(manifest, rows_tampered, actual, closure)

    def test_reference_defers_third_party_imports_until_preflight(self) -> None:
        tree = ast.parse(
            (Path(__file__).parent / "dump_reference.py").read_text(encoding="utf-8")
        )
        top_level_imports = {
            alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        top_level_from = {
            node.module.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("torch", top_level_imports | top_level_from)
        self.assertNotIn("numpy", top_level_imports | top_level_from)
        self.assertNotIn("gguf", top_level_imports | top_level_from)
        source = (Path(__file__).parent / "dump_reference.py").read_text(encoding="utf-8")
        self.assertLess(source.index("validate_patched_runtime()"), source.index("import numpy as np"))
        main_start = source.index("def main()")
        self.assertLess(
            source.index("    validate_execution_authorization()", main_start),
            source.index("    import numpy as np", main_start),
        )
        main_node = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        output_preflight = next(
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_new_output_paths"
        )
        runtime_preflight = next(
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_patched_runtime"
        )
        module_load = next(
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load_official_modules"
        )
        numpy_import = next(
            node.lineno
            for node in ast.walk(main_node)
            if isinstance(node, ast.Import)
            and any(alias.name == "numpy" for alias in node.names)
        )
        self.assertLess(output_preflight, runtime_preflight)
        self.assertLess(output_preflight, numpy_import)
        self.assertLess(output_preflight, module_load)

    def test_reference_inputs_and_outputs_are_bounded_and_non_overwriting(self) -> None:
        self.assertEqual(reference.AUDITED_GGUF_BYTES, 3_291_064_672)
        self.assertTrue(reference._bounded_size_ok(reference.AUDITED_GGUF_BYTES, reference.MAX_GGUF_BYTES))
        self.assertFalse(reference._bounded_size_ok(reference.AUDITED_GGUF_BYTES + 1, reference.MAX_GGUF_BYTES))
        with tempfile.TemporaryDirectory(prefix="xcodec2-reference-io-test-") as directory:
            root = Path(directory)
            codes = root / "codes.u32le"
            codes.write_bytes(b"\x00\x00\x00\x00")
            self.assertEqual(reference._validate_codes_file(codes), codes)
            misaligned = root / "misaligned.u32le"
            misaligned.write_bytes(b"\x00\x00\x00")
            with self.assertRaises(RuntimeError):
                reference._validate_codes_file(misaligned)
            oversized = root / "oversized"
            oversized.write_bytes(b"x" * 5)
            with self.assertRaises(RuntimeError):
                reference._regular_bounded_file(oversized, 4, "codes")
            link = root / "codes-link"
            link.symlink_to(codes)
            with self.assertRaises(RuntimeError):
                reference._regular_bounded_file(link, 4, "codes")

            with self.assertRaises(RuntimeError):
                reference._validate_reference_budget(
                    reference.MAX_REFERENCE_OUTPUT_BYTES // (reference.HIDDEN_DIM * 4) + 1
                )
            output = reference._safe_output_dir(root / "output")
            paths = reference._new_output_paths(output)
            self.assertEqual(tuple(path.name for path in paths), reference.REFERENCE_OUTPUT_NAMES)
            untouched = output / "untouched.bin"
            untouched.write_bytes(b"keep")
            existing = output / "features.f32"
            existing.write_bytes(b"keep")
            with self.assertRaises(RuntimeError):
                reference._new_output_paths(output)
            self.assertEqual(existing.read_bytes(), b"keep")
            existing.unlink()
            output_symlink = output / "features.f32"
            output_symlink.symlink_to("untouched.bin")
            with self.assertRaises(RuntimeError):
                reference._new_output_paths(output)
            output_symlink.unlink()
            target = output / "manifest.json"
            reference._atomic_write_bytes(target, b"first\n")
            self.assertEqual(target.read_bytes(), b"first\n")
            with self.assertRaises(RuntimeError):
                reference._atomic_write_bytes(target, b"second\n")
            target_link = output / "linked.json"
            target_link.symlink_to(target)
            with self.assertRaises(RuntimeError):
                reference._atomic_write_bytes(target_link, b"blocked\n")
            output_link = root / "output-link"
            output_link.symlink_to(output, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                reference._safe_output_dir(output_link)

            created: list[tuple[Path, int, int]] = []
            recoverable = output / "recoverable.bin"
            try:
                reference._atomic_write_bytes(recoverable, b"new output")
                reference._record_created_output(created, recoverable)
                raise RuntimeError("simulated later output failure")
            except RuntimeError:
                reference._remove_outputs_created_by_us(created)
            self.assertFalse(recoverable.exists())
            self.assertEqual(untouched.read_bytes(), b"keep")

    def test_atomic_writer_failure_and_link_race_are_recoverable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-atomic-failure-test-") as directory:
            output = Path(directory)
            sibling = output / "sibling.bin"
            sibling.write_bytes(b"preserve")
            partial = output / "partial.bin"

            def fail_after_partial_write(stream) -> None:
                stream.write(b"partial")
                raise RuntimeError("simulated writer failure")

            with self.assertRaises(RuntimeError):
                reference._atomic_write(partial, fail_after_partial_write)
            self.assertFalse(partial.exists())
            self.assertEqual(sibling.read_bytes(), b"preserve")
            self.assertEqual(list(output.glob(".partial.bin.*.tmp")), [])

            raced = output / "raced.bin"
            original_link = reference.os.link

            def create_race_target(source, destination):
                Path(destination).write_bytes(b"racer-owned")
                return original_link(source, destination)

            with mock.patch.object(reference.os, "link", side_effect=create_race_target):
                with self.assertRaises(FileExistsError):
                    reference._atomic_write(raced, lambda stream: stream.write(b"new"))
            self.assertEqual(raced.read_bytes(), b"racer-owned")
            self.assertEqual(list(output.glob(".raced.bin.*.tmp")), [])

    def test_archive_license_bytes_are_literal_and_traversal_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-audit-test-") as directory:
            path = Path(directory) / "demo.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("demo/LICENSE", b"MIT\n")
                archive.writestr("demo/large-native.bin", b"x" * (collector.MAX_MEMBER_BYTES + 1))
            evidence = collector.archive_license_files(path)
            self.assertEqual(evidence[0]["bytes"], 4)
            self.assertEqual(evidence[0]["sha256"], collector.sha256_bytes(b"MIT\n"))

    def test_wheel_installed_license_and_native_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-binding-test-") as directory:
            path = Path(directory) / "demo.whl"
            payloads = {"demo/LICENSE": b"MIT\n", "demo/libdemo.so": b"\x7fELFnative", "demo/omitted.so": b"not-in-publisher-record", "demo/RECORD": b"source-side record", "demo-1.dist-info/METADATA": b"Name: demo\n", "demo-1.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n"}
            record_rows = []
            for name, value in payloads.items():
                if name == "demo/omitted.so":
                    continue
                digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
                record_rows.append(f"{name},sha256={digest},{len(value)}")
            payloads["demo-1.dist-info/RECORD"] = ("\n".join(record_rows) + "\ndemo-1.dist-info/RECORD,,\n").encode()
            clean_path = Path(directory) / "clean.whl"
            with zipfile.ZipFile(clean_path, "w") as archive:
                for name, value in payloads.items():
                    if name != "demo/omitted.so":
                        archive.writestr(name, value)
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in payloads.items():
                    archive.writestr(name, value)
            # Installer-generated RECORD/WHEEL bytes are intentionally excluded;
            # this simulates a legitimate RECORD rewrite after installation.
            installed = {name: {"path": name, "bytes": len(value), "sha256": collector.sha256_bytes(value)} for name, value in payloads.items() if not collector.is_generated_installer_path(name)}
            binding, failures = collector.compare_wheel_payloads(clean_path, installed)
            self.assertEqual(binding["status"], "PAYLOAD_SCOPED_VERIFIED")
            self.assertFalse(failures)
            binding, failures = collector.compare_wheel_payloads(path, installed)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertTrue(any("omitted.so" in failure for failure in failures))
            self.assertFalse(collector.is_generated_installer_path("demo/RECORD"))
            removed = dict(installed)
            removed.pop("demo/LICENSE")
            binding, failures = collector.compare_wheel_payloads(clean_path, removed)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertTrue(any("missing publisher RECORD path" in failure for failure in failures))
            removed_native = dict(installed)
            removed_native.pop("demo/libdemo.so")
            _, native_failures = collector.compare_wheel_payloads(clean_path, removed_native)
            self.assertTrue(any("demo/libdemo.so" in failure for failure in native_failures))
            tampered = copy.deepcopy(installed)
            tampered["demo/LICENSE"]["sha256"] = "0" * 64
            tampered["demo/libdemo.so"]["bytes"] += 1
            binding, failures = collector.compare_wheel_payloads(clean_path, tampered)
            self.assertEqual(binding["status"], "BLOCKED")
            self.assertEqual(len(failures), 2)

    def test_collector_aggregate_caps_fail_closed(self) -> None:
        budget = collector.CollectionBudget()
        old_cap = collector.MAX_TOTAL_ARTIFACT_BYTES
        try:
            collector.MAX_TOTAL_ARTIFACT_BYTES = 4
            budget.add_artifact(4)
            with self.assertRaises(collector.EvidenceError):
                budget.add_artifact(1)
        finally:
            collector.MAX_TOTAL_ARTIFACT_BYTES = old_cap
        with tempfile.TemporaryDirectory(prefix="xcodec2-output-cap-") as directory:
            old_report_cap = collector.MAX_REPORT_BYTES
            try:
                collector.MAX_REPORT_BYTES = 16
                with self.assertRaises(collector.EvidenceError):
                    collector.write_atomic(Path(directory) / "report.json", {"value": "x" * 17})
            finally:
                collector.MAX_REPORT_BYTES = old_report_cap
        expired = collector.CollectionBudget(seconds=-1)
        body = b"payload"
        artifact = {
            "url": "https://files.pythonhosted.org/packages/expired.whl",
            "hash": "sha256:" + collector.sha256_bytes(body),
            "size": len(body),
        }
        with tempfile.TemporaryDirectory(prefix="xcodec2-timeout-") as directory:
            with self.assertRaises(collector.EvidenceError):
                collector.fetch_artifact(
                    artifact,
                    Path(directory),
                    lambda _url: (_ for _ in ()).throw(AssertionError("expired budget fetched")),
                    budget=expired,
                )

    def test_installed_hash_license_native_paths_honor_shared_deadline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-deadline-test-") as directory:
            root = Path(directory)
            (root / "demo").mkdir()
            (root / "demo/LICENSE").write_bytes(b"MIT\n")
            (root / "demo/libdemo.so").write_bytes(b"\x7fELFsynthetic")
            fake = type(
                "FakeDistribution",
                (),
                {
                    "files": ["demo/LICENSE", "demo/libdemo.so"],
                    "locate_file": lambda self, relative: root / str(relative),
                },
            )()
            expired = collector.CollectionBudget(seconds=-1)
            with self.assertRaises(collector.EvidenceError):
                collector.installed_license_files(fake, budget=expired)
            with self.assertRaises(collector.EvidenceError):
                collector.native_evidence(fake, budget=expired)
            with self.assertRaises(collector.EvidenceError):
                collector.installed_payload_hashes(fake, {"demo/libdemo.so"}, budget=expired)

    def test_response_timeout_refreshes_urllib_fp_socket_and_fails_closed_without_one(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.values = []

            def settimeout(self, value: float) -> None:
                self.values.append(value)

        socket = FakeSocket()
        response = SimpleNamespace(raw=None, fp=SimpleNamespace(raw=SimpleNamespace(_sock=socket)))
        self.assertTrue(collector.set_response_timeout(response, 1.25))
        self.assertEqual(socket.values, [1.25])
        self.assertFalse(collector.set_response_timeout(SimpleNamespace(raw=None, fp=None), 1.25))
        with self.assertRaises(collector.EvidenceError):
            collector.remaining_timeout(collector.CollectionBudget(seconds=-1))

    def test_decoder_preimport_source_has_no_boolean_bypass(self) -> None:
        source = (Path(__file__).parent / "dump_reference.py").read_text(encoding="utf-8")
        self.assertNotIn("prevalidated", source)
        self.assertIn("_PreimportProof", source)

    def test_preimport_gate_rejects_wrong_distribution_before_module_import(self) -> None:
        class FakeDistribution:
            metadata = {"Name": "numpy"}
            version = "0.0.0"
            files = []

            def locate_file(self, _relative: str) -> Path:
                return Path("/nonexistent")

        original = reference.importlib.metadata.distribution
        try:
            reference.importlib.metadata.distribution = lambda _name: FakeDistribution()
            with self.assertRaisesRegex(RuntimeError, "numpy version"):
                reference.validate_preimport_runtime()
        finally:
            reference.importlib.metadata.distribution = original
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("xcodec2", sys.modules)

    def test_preimport_gate_validates_real_tiny_record_and_origins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-preimport-real-") as directory:
            root = Path(directory)
            _, original_distribution, original_constants = self._with_fake_runtime(root)
            old_path = list(sys.path)
            removed = {name: sys.modules.pop(name, None) for name in reference.EXPECTED_MODULE_NAMES}
            try:
                sys.path.insert(0, str(root))
                proof = reference.validate_preimport_runtime()
                self.assertEqual(proof.versions["xcodec2"], "0.1.5")
                self.assertIn("torch/libtiny.so", proof.record_digests["torch"])
            finally:
                sys.path[:] = old_path
                for name, module in removed.items():
                    if module is not None:
                        sys.modules[name] = module
                self._restore_fake_runtime(original_distribution, original_constants)

    def test_preimport_gate_rejects_record_tamper_shadow_and_symlink_ancestor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-preimport-negative-") as directory:
            root = Path(directory)
            _, original_distribution, original_constants = self._with_fake_runtime(root)
            old_path = list(sys.path)
            removed = {name: sys.modules.pop(name, None) for name in reference.EXPECTED_MODULE_NAMES}
            try:
                sys.path.insert(0, str(root))
                (root / "numpy/__init__.py").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(RuntimeError, "RECORD bytes mismatch"):
                    reference.validate_preimport_runtime()
                self._restore_fake_runtime(original_distribution, original_constants)
                _, original_distribution, original_constants = self._with_fake_runtime(root)
                shadow = root / "shadow"
                (shadow / "numpy").mkdir(parents=True)
                (shadow / "numpy/__init__.py").write_bytes(b"shadow\n")
                sys.path.insert(0, str(shadow))
                with self.assertRaisesRegex(RuntimeError, "origin mismatch"):
                    reference.validate_preimport_runtime()
                sys.path.pop(0)
                self._restore_fake_runtime(original_distribution, original_constants)
                _, original_distribution, original_constants = self._with_fake_runtime(root)
                external = root / "external-numpy"
                (external / "numpy").mkdir(parents=True)
                (external / "numpy/__init__.py").write_bytes(b"external\n")
                (root / "numpy").rename(root / "numpy-real")
                (root / "numpy").symlink_to(external / "numpy", target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, "symlink ancestry"):
                    reference.validate_preimport_runtime()
            finally:
                sys.path[:] = old_path
                for name, module in removed.items():
                    if module is not None:
                        sys.modules[name] = module
                self._restore_fake_runtime(original_distribution, original_constants)

    def test_decoder_import_requires_preimport_proof(self) -> None:
        source = (Path(__file__).parent / "dump_reference.py").read_text(encoding="utf-8")
        self.assertNotIn("prevalidated", source)
        with self.assertRaises(TypeError):
            reference.import_official_decoder()  # type: ignore[call-arg]

    def test_blocked_manifest_stops_callable_before_official_import(self) -> None:
        proof = reference._PreimportProof({}, {}, reference._PREIMPORT_NONCE)
        original = reference.install_official_rope_import
        called = []
        try:
            reference.install_official_rope_import = lambda: called.append(True)
            with self.assertRaisesRegex(RuntimeError, "blocked by the existing audit/owner contract"):
                reference.import_official_decoder(proof)
        finally:
            reference.install_official_rope_import = original
        self.assertEqual(called, [])
        with self.assertRaisesRegex(RuntimeError, "blocked by the existing audit/owner contract"):
            reference.gguf_tensor(object(), ())
        self.assertNotIn("numpy", sys.modules)

    def test_malformed_audit_contract_stops_direct_tensor_gate(self) -> None:
        original = reference.audit.load_contract
        try:
            reference.audit.load_contract = lambda: {"manifest": {}}
            with self.assertRaisesRegex(RuntimeError, "blocked by the existing audit/owner contract"):
                reference.gguf_tensor(object(), ())
        finally:
            reference.audit.load_contract = original

    def test_collector_has_no_model_imports(self) -> None:
        for filename in ("dependency_audit.py", "collect_dependency_evidence.py"):
            tree = ast.parse((Path(__file__).parent / filename).read_text(encoding="utf-8"))
            imported = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported.update(
                node.module.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            )
            self.assertNotIn("torch", imported)
            self.assertNotIn("xcodec2", imported)

    def test_collector_cli_rejects_symlink_project_before_collection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="xcodec2-cli-path-test-") as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)
            old_argv = sys.argv
            old_collect = collector.collect
            old_write = collector.write_atomic
            try:
                sys.argv = ["collect_dependency_evidence.py", "--project", str(linked / "xcodec2"), "--output", str(root / "report.json"), "--expected-head", "0" * 40]
                collector.collect = lambda _head: (_ for _ in ()).throw(AssertionError("collection must not run"))
                collector.write_atomic = lambda _path, _report: (_ for _ in ()).throw(AssertionError("write must not run"))
                self.assertEqual(collector.main(), 2)
            finally:
                sys.argv = old_argv
                collector.collect = old_collect
                collector.write_atomic = old_write


if __name__ == "__main__":
    unittest.main()
