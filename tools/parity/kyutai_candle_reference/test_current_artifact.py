#!/usr/bin/env -S uv run --no-project --python 3.12
"""Stdlib tests for the bounded current-run Candle evidence envelope."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import current_artifact


class CurrentArtifactTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        root = Path(temporary) / "application"
        root.mkdir()
        for name in current_artifact.MEMBER_NAMES:
            (root / name).write_bytes((name + "\n").encode())
        return root

    def _envelope(self, root: Path) -> dict:
        return current_artifact.build_envelope(
            root,
            repository=current_artifact.PRODUCTION_REPOSITORY,
            run_id=123,
            run_attempt=1,
            source_head="a" * 40,
            workflow_sha="b" * 40,
            run_ref="refs/pull/123/merge",
            caller="applied-advisory",
            artifact_id=456,
            artifact_name="kyutai-candle-paste-application-evidence-123-1",
            artifact_digest="c" * 64,
            artifact_size=4096,
        )

    def test_member_map_and_envelope_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            envelope = self._envelope(root)
            self.assertEqual(current_artifact.validate_envelope(envelope), envelope)
            measured, receipt_hash = current_artifact.member_map(root)
            self.assertEqual(measured, {"members": envelope["members"], "application_receipt_sha256": receipt_hash})

    def test_duplicate_unknown_and_mutated_fields_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            envelope = self._envelope(root)
            unknown = copy.deepcopy(envelope)
            unknown["extra"] = True
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.validate_envelope(unknown)
            mutated = copy.deepcopy(envelope)
            mutated["members"][0]["sha256"] = "0" * 64
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.validate_envelope(mutated)
            receipt_mutation = copy.deepcopy(envelope)
            receipt_mutation["application_receipt_sha256"] = "0" * 64
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.validate_envelope(receipt_mutation)
            duplicate = '{"schema": 1, "schema": 2}'
            with self.assertRaises(current_artifact.CurrentArtifactError):
                json.loads(duplicate, object_pairs_hook=current_artifact._reject_duplicate_json)

    def test_member_set_and_link_types_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            (root / "extra").write_bytes(b"unexpected")
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.member_map(root)
            (root / "extra").unlink()
            target = Path(temporary) / "outside"
            target.write_bytes(b"outside")
            (root / "source_receipt.json").unlink()
            (root / "source_receipt.json").symlink_to(target)
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.member_map(root)
            (root / "source_receipt.json").unlink()
            (root / "source_receipt.json").write_bytes(b"source_receipt.json\n")
            (root / "baseline_cpu_metadata.json").unlink()
            os.link(root / "source_receipt.json", root / "baseline_cpu_metadata.json")
            with self.assertRaises(current_artifact.CurrentArtifactError):
                current_artifact.member_map(root)

    def test_capture_binds_api_digest_size_and_workflow_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            listing = {
                "total_count": 1,
                "artifacts": [{
                    "id": 456,
                    "name": "kyutai-candle-paste-application-evidence-123-1",
                    "digest": "sha256:" + "c" * 64,
                    "size_in_bytes": 4096,
                    "expired": False,
                    "workflow_run": {"id": 123, "head_sha": "a" * 40},
                }],
            }
            with mock.patch.object(current_artifact, "_api_json", return_value=listing):
                envelope = current_artifact.capture_primary_artifact(
                    root=root,
                    repository=current_artifact.PRODUCTION_REPOSITORY,
                    token="test",
                    run_id=123,
                    run_attempt=1,
                    source_head="a" * 40,
                    workflow_sha="b" * 40,
                    run_ref="refs/pull/123/merge",
                    caller="applied-advisory",
                    artifact_id=456,
                    artifact_name="kyutai-candle-paste-application-evidence-123-1",
                    upload_digest="c" * 64,
                )
            self.assertEqual(envelope["artifact_digest"], "c" * 64)
            bad = dict(listing)
            bad["artifacts"] = [dict(listing["artifacts"][0], size_in_bytes=current_artifact.MAX_ARTIFACT_BYTES + 1)]
            with mock.patch.object(current_artifact, "_api_json", return_value=bad):
                with self.assertRaises(current_artifact.CurrentArtifactError):
                    current_artifact.capture_primary_artifact(
                        root=root,
                        repository=current_artifact.PRODUCTION_REPOSITORY,
                        token="test",
                        run_id=123,
                        run_attempt=1,
                        source_head="a" * 40,
                        workflow_sha="b" * 40,
                        run_ref="refs/pull/123/merge",
                        caller="applied-advisory",
                        artifact_id=456,
                        artifact_name="kyutai-candle-paste-application-evidence-123-1",
                        upload_digest="c" * 64,
                    )


if __name__ == "__main__":
    unittest.main()
