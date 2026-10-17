#!/usr/bin/env python3
"""Offline mocked tests for release_candidate_model_metadata.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("release_candidate_model_metadata.py")
SPEC = importlib.util.spec_from_file_location("release_candidate_model_metadata", SCRIPT)
assert SPEC and SPEC.loader
metadata = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metadata
SPEC.loader.exec_module(metadata)


def payload(
    name: str,
    repo: str,
    revision: str,
    filename: str = "candidate.gguf",
    size: int = 17,
) -> dict[str, object]:
    return {
        "id": repo,
        "sha": revision,
        "siblings": [
            {
                "rfilename": filename,
                "size": size,
                "lfs": {"sha256": "a" * 64, "size": size},
            }
        ],
    }


class MetadataTests(unittest.TestCase):
    def test_real_fetch_requires_hosted_github_actions(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(metadata.MetadataError):
                metadata.collect_metadata()

    def test_api_reader_is_bounded_and_rejects_non_hf_redirects(self) -> None:
        repo, revision = metadata.REVISIONS["whisper-base"]
        api_url = "https://huggingface.co/api/models/" + repo + "/revision/" + revision + "?blobs=true"
        payload_bytes = b'{"id":"' + repo.encode() + b'","sha":"' + revision.encode() + b'"}'

        class Response:
            def __init__(
                self,
                body: bytes,
                final_url: str,
                content_type: str = "application/json; charset=utf-8",
            ) -> None:
                self.body = body
                self.final_url = final_url
                self.content_type = content_type
                self.read_sizes: list[int] = []

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def geturl(self) -> str:
                return self.final_url

            @property
            def headers(self) -> dict[str, str]:
                return {"Content-Type": self.content_type}

            def read(self, size: int) -> bytes:
                self.read_sizes.append(size)
                chunk, self.body = self.body[:size], self.body[size:]
                return chunk

        response = Response(payload_bytes, api_url)
        seen: dict[str, object] = {}

        def opener(request: object, timeout: int) -> Response:
            seen["request"] = request
            seen["timeout"] = timeout
            return response

        result, byte_count, _ = metadata._fetch_json(
            api_url,
            opener=opener,
            clock=lambda: 0.0,
        )
        self.assertEqual(result["id"], "vokra/whisper-base")
        self.assertEqual(byte_count, len(payload_bytes))
        self.assertEqual(response.read_sizes[0], min(64 * 1024, metadata.MAX_API_BYTES + 1))
        request = seen["request"]
        self.assertEqual(seen["timeout"], metadata.SOCKET_TIMEOUT)
        self.assertIsNone(request.get_header("Authorization"))  # type: ignore[union-attr]

        with self.assertRaises(metadata.MetadataError):
            metadata._fetch_json(
                api_url,
                opener=lambda request, timeout: Response(payload_bytes, "https://cdn.example.invalid/api"),
                clock=lambda: 0.0,
            )

        with self.assertRaises(metadata.MetadataError):
            metadata._fetch_json(
                api_url,
                opener=lambda request, timeout: Response(payload_bytes, api_url, "text/plain"),
                clock=lambda: 0.0,
            )

        duplicate = Response(b'{"id":"x","id":"y"}', api_url)
        with self.assertRaises(metadata.MetadataError):
            metadata._fetch_json(api_url, opener=lambda request, timeout: duplicate, clock=lambda: 0.0)

        oversized = Response(b"x" * (metadata.MAX_API_BYTES + 1), api_url)
        with self.assertRaises(metadata.MetadataError):
            metadata._fetch_json(api_url, opener=lambda request, timeout: oversized, clock=lambda: 0.0)
        self.assertLessEqual(max(oversized.read_sizes), 64 * 1024)

        with self.assertRaises(metadata.MetadataError):
            metadata._fetch_json(
                api_url.replace("/api/models/", "/resolve/"),
                opener=opener,
                clock=lambda: 0.0,
            )
        with self.assertRaises(metadata.MetadataError):
            metadata._NoRedirectHandler().redirect_request(
                metadata.urllib.request.Request(api_url),
                None,
                302,
                "redirect",
                {},
                api_url,
            )

    def test_collects_only_fixed_revisions_and_validates_smoke_manifest(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> tuple[dict[str, object], int, str]:
            calls.append(url)
            name = "whisper-base" if "whisper-base" in url else "silero-vad-v5"
            repo, revision = metadata.REVISIONS[name]
            return payload(name, repo, revision, f"{name}.gguf"), 123, "b" * 64

        pins, identities = metadata.collect_metadata(fetcher=fetch)
        self.assertEqual(set(pins), set(metadata.REVISIONS))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("blobs=true" in url and "/revision/" in url for url in calls))
        self.assertEqual(identities[0]["response_bytes"], 123)
        self.assertNotIn("siblings", json.dumps(identities))

    def test_rejects_ambiguous_missing_lfs_or_wrong_revision(self) -> None:
        name = "whisper-base"
        repo, revision = metadata.REVISIONS[name]
        ambiguous = payload(name, repo, revision)
        ambiguous["siblings"] = [*ambiguous["siblings"], ambiguous["siblings"][0]]
        with self.assertRaises(metadata.MetadataError):
            metadata._extract_pin(name, repo, revision, ambiguous)

        missing_lfs = payload(name, repo, revision)
        missing_lfs["siblings"][0].pop("lfs")
        with self.assertRaises(metadata.MetadataError):
            metadata._extract_pin(name, repo, revision, missing_lfs)

        legacy_oid = payload(name, repo, revision)
        legacy_oid["siblings"][0]["lfs"] = {"oid": "a" * 64, "size": 17}
        with self.assertRaises(metadata.MetadataError):
            metadata._extract_pin(name, repo, revision, legacy_oid)

        missing_sha = payload(name, repo, revision)
        missing_sha["sha"] = None
        with self.assertRaises(metadata.MetadataError):
            metadata._extract_pin(name, repo, revision, missing_sha)

        wrong_revision = payload(name, repo, "0" * 40)
        with self.assertRaises(metadata.MetadataError):
            metadata._extract_pin(name, repo, revision, wrong_revision)

    def test_collect_rejects_unsafe_filename_and_combined_size(self) -> None:
        name = "whisper-base"
        repo, revision = metadata.REVISIONS[name]

        def unsafe_fetch(url: str) -> tuple[dict[str, object], int, str]:
            current = "whisper-base" if "whisper-base" in url else "silero-vad-v5"
            current_repo, current_revision = metadata.REVISIONS[current]
            filename = "../unsafe.gguf" if current == name else f"{current}.gguf"
            return payload(current, current_repo, current_revision, filename), 1, "a" * 64

        with self.assertRaises(metadata.MetadataError):
            metadata.collect_metadata(fetcher=unsafe_fetch)

        def oversized_fetch(url: str) -> tuple[dict[str, object], int, str]:
            current = "whisper-base" if "whisper-base" in url else "silero-vad-v5"
            current_repo, current_revision = metadata.REVISIONS[current]
            return payload(current, current_repo, current_revision, f"{current}.gguf", 1_500_000_000), 1, "a" * 64

        with self.assertRaises(metadata.MetadataError):
            metadata.collect_metadata(fetcher=oversized_fetch)

    def test_outputs_are_compact_and_do_not_include_raw_api_payload(self) -> None:
        pins = {
            name: {
                "repo": repo,
                "revision": revision,
                "filename": "candidate.gguf",
                "sha256": "a" * 64,
                "size_bytes": 17,
            }
            for name, (repo, revision) in metadata.REVISIONS.items()
        }
        identities = [{"name": name, "repo": value["repo"], "revision": value["revision"]} for name, value in pins.items()]
        with tempfile.TemporaryDirectory() as directory:
            pins_path = Path(directory) / "pins.json"
            receipt_path = Path(directory) / "receipt.json"
            metadata.write_outputs(
                pins_path,
                receipt_path,
                pins,
                identities,
                {"github_sha": "a" * 40, "github_run_id": "123"},
            )
            self.assertLessEqual(pins_path.stat().st_size + receipt_path.stat().st_size, metadata.MAX_OUTPUT_BYTES)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertNotIn("siblings", receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["source"], {"github_sha": "a" * 40, "github_run_id": "123"})

    def test_main_fails_without_emitting_pins_or_pass_receipt(self) -> None:
        name = "whisper-base"
        repo, revision = metadata.REVISIONS[name]
        bad = payload(name, repo, revision, "../unsafe.gguf")
        good_name = "silero-vad-v5"
        good_repo, good_revision = metadata.REVISIONS[good_name]

        def bad_fetch(url: str) -> tuple[dict[str, object], int, str]:
            if "whisper-base" in url:
                return bad, 1, "a" * 64
            return payload(good_name, good_repo, good_revision, "silero-vad-v5.gguf"), 1, "b" * 64

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"GITHUB_ACTIONS": "true", "GITHUB_SHA": "a" * 40, "GITHUB_RUN_ID": "123"},
            clear=True,
        ), mock.patch.object(metadata, "_fetch_json", side_effect=bad_fetch):
            pins_path = Path(directory) / "pins.json"
            receipt_path = Path(directory) / "receipt.json"
            self.assertEqual(
                metadata.main(["--pins-output", str(pins_path), "--receipt", str(receipt_path)]),
                1,
            )
            self.assertFalse(pins_path.exists())
            self.assertFalse(receipt_path.exists())


if __name__ == "__main__":
    unittest.main()
