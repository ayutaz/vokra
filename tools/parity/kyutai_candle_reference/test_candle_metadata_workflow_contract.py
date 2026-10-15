from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import collect_metadata as leaf


WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/kyutai-candle-source-metadata.yml"
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
WORKSPACE = '''[workspace]
members = ["moshi-core", "other"]
resolver = "2"
[workspace.package]
license = "MIT/Apache-2.0"
'''
CORE = '''[package]
name = "moshi"
version = "0.6.4"
'''
LICENSE = b"source-only license receipt fixture\n"
LOCK = '''version = 4
[[package]]
name = "moshi"
version = "0.6.4"
dependencies = ["memmap2"]

[[package]]
name = "memmap2"
version = "{version}"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "{checksum}"
'''
METADATA = json.dumps(
    {
        "packages": [
            {
                "id": "moshi-id", "name": "moshi", "version": "0.6.4", "source": None,
                "manifest_path": "moshi-core/Cargo.toml", "features": {"default": []},
                "dependencies": [{
                    "name": "memmap2", "source": "registry+https://github.com/rust-lang/crates.io-index",
                    "req": "^0.9.5", "kind": None, "optional": False, "rename": None,
                    "target": None, "features": [], "uses_default_features": True,
                }],
            },
            {
                "id": "memmap-id", "name": "memmap2", "version": "0.9.5",
                "source": "registry+https://github.com/rust-lang/crates.io-index",
                "manifest_path": "memmap/Cargo.toml", "features": {}, "dependencies": [],
            },
        ],
        "workspace_members": ["moshi-id"],
        "resolve": {
            "root": None,
            "nodes": [
                {"id": "moshi-id", "features": [], "deps": [{
                    "name": "memmap2", "pkg": "memmap-id", "dep_kinds": [{"kind": None, "target": None}],
                }]},
                {"id": "memmap-id", "features": [], "deps": []},
            ],
        },
    }
)


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


class CandleMetadataWorkflowContractTests(unittest.TestCase):
    def test_workflow_recovers_exact_seven_evidence_files_without_globs_or_caches(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        expected = [
            "source_receipt.json",
            "source/rust/Cargo.lock",
            "base/Cargo.lock",
            "patched/Cargo.lock",
            "initial_metadata.json",
            "whole_lock_metadata.json",
            "metadata.json",
        ]
        check_start = text.index("      - name: Check the bounded evidence set")
        check_end = text.index("      - name: Upload exact bounded evidence", check_start)
        check_block = text[check_start:check_end]
        check_paths = [
            candidate
            for line in check_block.splitlines()
            if (candidate := line.strip().rstrip("\\").strip().removesuffix("; do").strip()).endswith((".json", "Cargo.lock"))
        ]
        self.assertEqual(check_paths, expected)
        upload_start = text.index("      - name: Upload exact bounded evidence")
        upload_block = text[upload_start:]
        upload_paths = [line.strip().rsplit("}/", 1)[-1] for line in upload_block.splitlines() if "runner.temp" in line]
        self.assertEqual(upload_paths, expected)
        self.assertNotIn("*", check_block + upload_block)
        self.assertNotIn("cache", check_block.lower() + upload_block.lower())

    def test_checkout_is_exact_sparse_owned_allowlist_without_filter(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("      - name: Checkout")
        end = text.index("      - name: Install uv", start)
        checkout = text[start:end]
        sparse_start = checkout.index("          sparse-checkout: |\n") + len("          sparse-checkout: |\n")
        sparse_end = checkout.index("          sparse-checkout-cone-mode: false", sparse_start)
        entries = [line.strip() for line in checkout[sparse_start:sparse_end].splitlines() if line.strip()]
        self.assertEqual(
            entries,
            [
                ".github/workflows/kyutai-candle-source-metadata.yml",
                "tools/parity/kyutai_candle_reference/collect_metadata.py",
                "tools/parity/kyutai_candle_reference/test_collect_metadata.py",
                "tools/parity/kyutai_candle_reference/test_collect_metadata_diagnostics.py",
                "tools/parity/kyutai_candle_reference/test_candle_metadata_workflow_contract.py",
            ],
        )
        self.assertIn("persist-credentials: false", checkout)
        self.assertIn("sparse-checkout-cone-mode: false", checkout)
        self.assertNotIn("filter:", checkout)
        self.assertNotIn("silero-vad-v5.gguf", checkout)
        self.assertNotIn("license_gate_manifest.json", checkout)

    def test_workflow_leaves_fresh_output_creation_to_collector(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        start = text.index("      - name: Capture source-only Cargo metadata")
        end = text.index("      - name: Check the bounded evidence set", start)
        capture_step = text[start:end]
        self.assertIn('VOKRA_CANDLE_REMOTE_LINUX: "1"', capture_step)
        self.assertIn('test ! -e "$capture_root"', capture_step)
        self.assertIn('test ! -L "$capture_root"', capture_step)
        self.assertNotIn('mkdir "$capture_root"', capture_step)
        self.assertIn("--deadline-seconds 480", capture_step)
        self.assertIn("collect_metadata.py", capture_step)

    def test_fresh_boundary_rejects_precreated_output_before_source_fetch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-candle-contract-", dir=TEMP_ROOT) as temp:
            output = Path(temp) / "capture"
            output.mkdir()
            fetched = False

            def fetch(_url: str, _deadline: float) -> bytes:
                nonlocal fetched
                fetched = True
                raise AssertionError("source fetch must not start for a precreated output")

            with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                with self.assertRaisesRegex(leaf.CaptureError, "destination must not exist"):
                    leaf.capture(output, deadline_seconds=480, source_fetcher=fetch, platform_name="Linux")
            self.assertFalse(fetched)

    def test_exact_main_entry_with_mocked_source_and_cargo(self) -> None:
        original_files = leaf.SOURCE_FILES
        original_urls = leaf.SOURCE_URLS
        lock_base = LOCK.format(version="0.9.5", checksum="a" * 64).encode()
        lock_patched = LOCK.format(version="0.9.11", checksum="b" * 64).encode()
        source_data = {
            "Cargo.toml": WORKSPACE.encode(),
            "Cargo.lock": lock_base,
            "LICENSE": LICENSE,
            "README.md": b"readme\n",
            "moshi-core/Cargo.toml": CORE.encode(),
            "moshi-core/src/lib.rs": b"pub fn synthetic() {}\n",
        }
        leaf.SOURCE_FILES = {name: (_blob(data), len(data)) for name, data in source_data.items()}
        leaf.SOURCE_URLS = {name: f"https://example.invalid/test/{name}" for name in source_data}
        try:
            def fetch(url: str, _deadline: float) -> bytes:
                return source_data[url.split("/test/", 1)[1]]

            cargo_calls: list[list[str]] = []

            def which(name: str) -> str:
                return f"/synthetic/{name}"

            def run(argv: list[str], *, cwd: Path, env: dict[str, str], deadline: float, validate: bool = True) -> dict[str, object]:
                del env, deadline, validate
                if Path(argv[0]).name == "rustc":
                    return {"argv": argv, "returncode": 0, "stdout": "rustc 1.99.0\nhost: x86_64-unknown-linux-gnu\n", "stderr": ""}
                cargo_calls.append(argv)
                if argv[1] == "update":
                    (cwd / "Cargo.lock").write_bytes(lock_patched)
                return {"argv": argv, "returncode": 0, "stdout": METADATA if argv[1] == "metadata" else "tree\n", "stderr": ""}

            with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                with mock.patch.object(leaf.platform, "system", return_value="Linux"):
                    with mock.patch.object(leaf.shutil, "which", side_effect=which):
                        with mock.patch.object(leaf, "run_bounded_subprocess", side_effect=run):
                            with mock.patch.object(leaf, "_bounded_url_fetch", side_effect=fetch):
                                with tempfile.TemporaryDirectory(prefix="vokra-candle-main-", dir=TEMP_ROOT) as temp:
                                    output = Path(temp) / "capture"
                                    self.assertEqual(leaf.main([str(output), "--deadline-seconds", "480"]), 0)
                                    self.assertTrue((output / "metadata.json").is_file())
            self.assertEqual([argv[1] for argv in cargo_calls], ["metadata", "tree", "tree", "tree", "metadata", "update", "metadata", "tree", "tree", "tree"])
        finally:
            leaf.SOURCE_FILES = original_files
            leaf.SOURCE_URLS = original_urls


if __name__ == "__main__":
    unittest.main()
