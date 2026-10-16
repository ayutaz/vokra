#!/usr/bin/env -S uv run --no-project --python 3.12
"""Offline stdlib tests for the authenticated paste overlay plan."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_paste_overlay.py")
SPEC = importlib.util.spec_from_file_location("prepare_paste_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


class PasteOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent, self.pastey = overlay.parse_receipts()

    def test_authenticated_sixteen_package_plan(self) -> None:
        payload = overlay.build_plan(self.parent, self.pastey)
        self.assertEqual(payload["status"], "PLAN_ONLY")
        self.assertEqual(payload["application"], "NOT_RUN")
        self.assertEqual(payload["compatibility"], "NOT_TESTED")
        self.assertEqual(payload["license_review"], "NOT_PERFORMED")
        self.assertIsNone(payload["approval"])
        self.assertEqual(payload["model_execution"], "NOT_AUTHORIZED")
        self.assertEqual(payload["upload"], "NO_UPLOAD")
        self.assertEqual(payload["parent_counts"], {"total": 16, "cpu": 14, "metal": 2})
        self.assertEqual(len(payload["proposals"]), 16)
        patch_lines = payload["isolated_patch_crates_io"].splitlines()[1:]
        self.assertEqual(len(patch_lines), 16)
        self.assertEqual(len({line.split('"', 2)[1] for line in patch_lines}), 16)
        self.assertIn('package = "gemm"', payload["isolated_patch_crates_io"])
        self.assertIn('package = "metal"', payload["isolated_patch_crates_io"])
        self.assertNotIn('package = "pastey"', payload["isolated_patch_crates_io"])

        for proposal in payload["proposals"]:
            original = proposal["original_manifest_parsed"]
            proposed = overlay.parse_manifest(proposal["proposed_manifest"]["text"], proposal["identity"])
            expected = copy.deepcopy(original)
            expected["dependencies"]["paste"]["version"] = "=0.2.3"
            expected["dependencies"]["paste"]["package"] = "pastey"
            self.assertEqual(proposed, expected)
            raw = proposal["proposed_manifest"]["text"].encode()
            self.assertEqual(proposal["proposed_manifest"]["bytes"], len(raw))
            self.assertEqual(proposal["proposed_manifest"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(proposal["expected_unchanged_members"])
            self.assertTrue(all(m["path"] != "Cargo.toml" for m in proposal["expected_unchanged_members"]))
            self.assertEqual(proposal["application"], "NOT_RUN")
            self.assertEqual(proposal["source_unchanged"], "NOT_CLAIMED_UNTIL_HOSTED_APPLICATION")

        with tempfile.TemporaryDirectory(prefix="paste-overlay-plan-") as directory:
            output = Path(directory) / "plan.json"
            overlay.write_plan(output, payload)
            raw = output.read_bytes()
            self.assertLessEqual(len(raw), overlay.JSON_LIMIT)
            reread = json.loads(raw)
            self.assertEqual(reread["parent_counts"], {"total": 16, "cpu": 14, "metal": 2})
            self.assertEqual(len(reread["proposals"]), 16)

    def test_manifest_rejects_alias_features_and_unknown_versions(self) -> None:
        base = '[package]\nname = "fixture"\nversion = "1.0.0"\n\n[dependencies.paste]\nversion = "1.0"\n'
        cases = (
            (base.replace('version = "1.0"', 'version = "1.0"\npackage = "pastey"'), "alias"),
            (base.replace('version = "1.0"', 'version = "1.0"\nfeatures = []'), "features"),
            (base.replace('version = "1.0"', 'version = "1.2"'), "version"),
            (base.replace('[dependencies.paste]', '[dependencies.paste]\n\n[dependencies.paste.foo]'), "nested"),
            (base.replace('[dependencies.paste]', '[dependencies.paste]\nversion = "1.0"\n\n[dependencies.paste]'), "duplicate"),
        )
        for text, label in cases:
            with self.subTest(label=label), self.assertRaises(overlay.OverlayError):
                overlay.propose_manifest(text, label)

    def test_semantic_difference_is_rejected_by_guard(self) -> None:
        payload = overlay.build_plan(self.parent, self.pastey)
        proposal = payload["proposals"][0]
        archive = next(a for a in self.parent["archives"] if f'{a["name"]}@{a["version"]}' == proposal["identity"])
        original = overlay.manifest_text(self.parent, archive)
        changed = proposal["proposed_manifest"]["text"].replace(
            'version = "=0.2.3"', 'version = "=0.2.3"\n\n[package.metadata.overlay-test]\nchanged = true'
        )
        with self.assertRaises(overlay.OverlayError):
            overlay.validate_proposed_manifest(original, changed, "unrelated")

    def test_structural_parent_mutations_fail_closed(self) -> None:
        cases = []
        mutated = copy.deepcopy(self.parent)
        mutated["plan"]["bindings"][0]["source"] = "path+forged"
        cases.append(("source", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["plan"]["bindings"][0]["cpu"] = False
        cases.append(("cpu flag", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["plan"]["bindings"][1] = copy.deepcopy(mutated["plan"]["bindings"][0])
        cases.append(("duplicate", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["plan"]["bindings"].pop()
        cases.append(("missing", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["archives"][0]["archive_sha256"] = "0" * 64
        cases.append(("forged archive", mutated))
        mutated = copy.deepcopy(self.parent)
        rust_path = next(iter(mutated["archives"][0]["rust_sources"]))
        mutated["archives"][0]["rust_sources"][rust_path]["sha256"] = "0" * 64
        cases.append(("Rust sha mismatch", mutated))
        mutated = copy.deepcopy(self.parent)
        rust_path = next(iter(mutated["archives"][0]["rust_sources"]))
        mutated["archives"][0]["regular_members"] = [m for m in mutated["archives"][0]["regular_members"] if m["path"] != rust_path]
        cases.append(("member missing", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["archives"][0]["regular_members"].append(copy.deepcopy(mutated["archives"][0]["regular_members"][0]))
        cases.append(("member duplicate", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["archives"][0]["regular_members"][0]["path"] = "../unsafe"
        cases.append(("unsafe member path", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["archives"][0]["manifest_orig"]["bytes"] += 1
        cases.append(("orig bytes mismatch", mutated))
        mutated = copy.deepcopy(self.parent)
        manifest = mutated["archives"][0]["manifest"]
        manifest["text_ref"] = next(ref for ref in mutated["text_store"] if ref != manifest["sha256"])
        cases.append(("aliased text ref", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated["status"] = "NOT_COMPLETE"
        cases.append(("status", mutated))
        mutated = copy.deepcopy(self.parent)
        mutated.pop("approval")
        cases.append(("missing approval", mutated))
        for label, candidate in cases:
            with self.subTest(label=label), self.assertRaises(overlay.OverlayError):
                overlay.build_plan(candidate, self.pastey)

    def test_receipt_hash_and_type_checks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paste-overlay-input-") as directory:
            root = Path(directory)
            body = root / "body.json"
            body.write_bytes(b"{}")
            with self.assertRaises(overlay.OverlayError):
                overlay.read_authenticated(body, 2, "0" * 64)
            link = root / "link.json"
            link.symlink_to(body)
            with self.assertRaises(overlay.OverlayError):
                overlay.read_authenticated(link, 2, hashlib.sha256(b"{}").hexdigest())

    def test_output_must_be_fresh_absolute_nonsymlink(self) -> None:
        payload = overlay.build_plan(self.parent, self.pastey)
        with tempfile.TemporaryDirectory(prefix="paste-overlay-output-") as directory:
            root = Path(directory)
            output = root / "plan.json"
            overlay.write_plan(output, payload)
            with self.assertRaises(overlay.OverlayError):
                overlay.write_plan(output, payload)
            with self.assertRaises(overlay.OverlayError):
                overlay.write_plan(Path("relative-plan.json"), payload)
            with self.assertRaises(overlay.OverlayError):
                overlay.write_plan(root / "missing" / "plan.json", payload)
            symlink_parent = root / "symlink-parent"
            symlink_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(overlay.OverlayError):
                overlay.write_plan(symlink_parent / "plan.json", payload)
