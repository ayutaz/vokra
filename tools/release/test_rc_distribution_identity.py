#!/usr/bin/env python3
"""Offline tests for the RC distribution identity annotator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rc_distribution_identity as identity
from test_desktop_release import make_valid_dist


SOURCE_HEAD = "a" * 40


def smoke_receipt(cli: Path) -> dict[str, object]:
    cli = cli.resolve()

    def command(label: str) -> dict[str, object]:
        return {
            "label": label,
            "returncode": 0,
            "argv": [
                str(cli),
                "run",
                "--model",
                f"/tmp/{label}.gguf",
                "--input",
                "/tmp/audio.wav",
                "--backend",
                "cpu",
            ],
        }

    return {
        "schema": 1,
        "status": "PASS",
        "runtime": {
            "head": SOURCE_HEAD,
            "expected_head": SOURCE_HEAD,
            "platform": "linux",
            "github_actions": "true",
            "marker": "1",
        },
        "functional_smoke_only": True,
        "not_numerical_parity": True,
        "not_metal_or_performance": True,
        "semantic_checks": {
            "asr_jfk_keyphrase": {"passed": True},
            "vad": {"passed": True},
        },
        "commands": [
            command("asr-jfk"),
            command("vad-jfk"),
            command("vad-silence"),
        ],
    }


class DistributionIdentityTests(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        _, dist = make_valid_dist(root)
        cli = dist / "vokra-cli-1.2.3-x86_64-linux"
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps(smoke_receipt(cli), indent=2) + "\n", encoding="utf-8")
        return temporary, receipt, dist / "vokra-desktop-manifest.json", cli

    def run_annotation(self, receipt: Path, manifest: Path, cli: Path) -> None:
        identity.annotate(
            receipt_path=receipt,
            manifest_path=manifest,
            cli_path=cli,
            version="1.2.3",
            source_head=SOURCE_HEAD,
            github_run_id="12345",
            github_run_attempt="2",
        )

    def test_annotation_binds_verified_payload_and_preserves_receipt(self) -> None:
        temporary, receipt, manifest, cli = self.fixture()
        try:
            before = json.loads(receipt.read_text(encoding="utf-8"))
            self.run_annotation(receipt, manifest, cli)
            after = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual({key: after[key] for key in before}, before)
            self.assertEqual(
                after["distribution_identity"],
                {
                    "schema": identity.IDENTITY_SCHEMA,
                    "source_head": SOURCE_HEAD,
                    "github_run_id": "12345",
                    "github_run_attempt": "2",
                    "cli": {
                        "filename": cli.name,
                        "bytes": cli.stat().st_size,
                        "sha256": identity._sha256(cli),
                    },
                    "manifest_sha256": identity._sha256(manifest),
                },
            )
        finally:
            temporary.cleanup()

    def test_rejects_existing_identity_without_rewriting_receipt(self) -> None:
        temporary, receipt, manifest, cli = self.fixture()
        try:
            original = json.loads(receipt.read_text(encoding="utf-8"))
            original["distribution_identity"] = {"schema": "old"}
            receipt.write_text(json.dumps(original) + "\n", encoding="utf-8")
            before = receipt.read_bytes()
            with self.assertRaises(identity.DistributionIdentityError):
                self.run_annotation(receipt, manifest, cli)
            self.assertEqual(receipt.read_bytes(), before)
        finally:
            temporary.cleanup()

    def test_rejects_source_run_and_attempt_mismatches(self) -> None:
        for field, value in (
            ("source_head", "b" * 40),
            ("source_head", "not-a-sha"),
            ("github_run_id", "0"),
            ("github_run_attempt", "not-a-run"),
        ):
            with self.subTest(field=field):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    kwargs = {
                        "receipt_path": receipt,
                        "manifest_path": manifest,
                        "cli_path": cli,
                        "version": "1.2.3",
                        "source_head": SOURCE_HEAD,
                        "github_run_id": "12345",
                        "github_run_attempt": "2",
                    }
                    kwargs[field] = value
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        identity.annotate(**kwargs)
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

    def test_rejects_status_and_safety_flag_mutations(self) -> None:
        mutations = {
            "status": lambda document: document.__setitem__("status", "FAIL"),
            "functional": lambda document: document.__setitem__("functional_smoke_only", False),
            "numerical": lambda document: document.__setitem__("not_numerical_parity", False),
            "metal": lambda document: document.__setitem__("not_metal_or_performance", False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                    mutate(document)
                    receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        self.run_annotation(receipt, manifest, cli)
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

    def test_rejects_missing_semantic_and_runtime_contracts(self) -> None:
        mutations = {
            "schema": lambda document: document.__setitem__("schema", 2),
            "schema-bool": lambda document: document.__setitem__("schema", True),
            "asr": lambda document: document["semantic_checks"].pop("asr_jfk_keyphrase"),
            "vad": lambda document: document["semantic_checks"].pop("vad"),
            "platform": lambda document: document["runtime"].__setitem__("platform", "darwin"),
            "github-actions": lambda document: document["runtime"].__setitem__("github_actions", "false"),
            "marker": lambda document: document["runtime"].__setitem__("marker", "0"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                    mutate(document)
                    receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        self.run_annotation(receipt, manifest, cli)
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

    def test_rejects_wrong_backend_and_cli_bytes(self) -> None:
        for mutation in ("backend", "bytes"):
            with self.subTest(mutation=mutation):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                    if mutation == "backend":
                        document["commands"][0]["argv"][-1] = "metal"
                    else:
                        cli.write_bytes(cli.read_bytes() + b"changed")
                    receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        self.run_annotation(receipt, manifest, cli)
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

    def test_rejects_manifest_parent_or_version_filename_mismatch(self) -> None:
        temporary, receipt, manifest, cli = self.fixture()
        try:
            wrong_cli = manifest.parent / "vokra-cli-9.9.9-x86_64-linux"
            wrong_cli.write_bytes(cli.read_bytes())
            document = json.loads(receipt.read_text(encoding="utf-8"))
            for command in document["commands"]:
                command["argv"][0] = str(wrong_cli.resolve())
            receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(identity.DistributionIdentityError, "CLI filename"):
                self.run_annotation(receipt, manifest, wrong_cli)

            outside = Path(temporary.name) / "outside"
            outside.mkdir()
            outside_cli = outside / cli.name
            outside_cli.write_bytes(cli.read_bytes())
            document = json.loads(receipt.read_text(encoding="utf-8"))
            for command in document["commands"]:
                command["argv"][0] = str(outside_cli.resolve())
            receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(identity.DistributionIdentityError, "not beside"):
                self.run_annotation(receipt, manifest, outside_cli)
        finally:
            temporary.cleanup()

    def test_rejects_missing_or_symlinked_distribution_inputs(self) -> None:
        for kind in ("missing-cli", "missing-manifest", "symlink-cli", "symlink-manifest"):
            with self.subTest(kind=kind):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    if kind == "missing-cli":
                        candidate = manifest.parent / "missing-cli"
                        call_cli = candidate
                        call_manifest = manifest
                    elif kind == "missing-manifest":
                        candidate = manifest.parent / "missing-manifest.json"
                        call_cli = cli
                        call_manifest = candidate
                    elif kind == "symlink-cli":
                        candidate = manifest.parent / "linked-cli"
                        os.symlink(cli, candidate)
                        call_cli = candidate
                        call_manifest = manifest
                    else:
                        candidate = manifest.parent / "linked-manifest.json"
                        os.symlink(manifest, candidate)
                        call_cli = cli
                        call_manifest = candidate
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        identity.annotate(
                            receipt_path=receipt,
                            manifest_path=call_manifest,
                            cli_path=call_cli,
                            version="1.2.3",
                            source_head=SOURCE_HEAD,
                            github_run_id="12345",
                            github_run_attempt="2",
                        )
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

    def test_rejects_wrong_argv_and_returncode_without_rewriting(self) -> None:
        for name in ("verb", "duplicate-backend", "trailing-backend", "duplicate-label", "returncode"):
            with self.subTest(name=name):
                temporary, receipt, manifest, cli = self.fixture()
                try:
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                    command = document["commands"][0]
                    if name == "verb":
                        command["argv"][1] = "help"
                    elif name == "duplicate-backend":
                        command["argv"] += ["--backend", "cpu"]
                    elif name == "trailing-backend":
                        command["argv"] += ["--backend"]
                    elif name == "duplicate-label":
                        document["commands"][1]["label"] = "asr-jfk"
                    else:
                        command["returncode"] = True
                    receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
                    before = receipt.read_bytes()
                    with self.assertRaises(identity.DistributionIdentityError):
                        self.run_annotation(receipt, manifest, cli)
                    self.assertEqual(receipt.read_bytes(), before)
                finally:
                    temporary.cleanup()

        temporary, receipt, manifest, cli = self.fixture()
        try:
            document = json.loads(receipt.read_text(encoding="utf-8"))
            document["commands"][0]["argv"][0] = str((manifest.parent / "other-cli").resolve())
            receipt.write_text(json.dumps(document) + "\n", encoding="utf-8")
            before = receipt.read_bytes()
            with self.assertRaisesRegex(identity.DistributionIdentityError, "CLI path mismatch"):
                self.run_annotation(receipt, manifest, cli)
            self.assertEqual(receipt.read_bytes(), before)
        finally:
            temporary.cleanup()

    def test_atomic_write_failure_preserves_receipt_and_cleans_temp(self) -> None:
        temporary, receipt, manifest, cli = self.fixture()
        try:
            before = receipt.read_bytes()
            with mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")):
                with self.assertRaises(OSError):
                    self.run_annotation(receipt, manifest, cli)
            self.assertEqual(receipt.read_bytes(), before)
            self.assertEqual(list(receipt.parent.glob(f".{receipt.name}.*.tmp")), [])
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
