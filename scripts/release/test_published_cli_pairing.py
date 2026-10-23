#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Offline metadata and workflow checks; never executes the smoke driver."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "tools/parity/release_candidate_published_cli_pairing_pins.json"
HELPER = ROOT / "scripts/release/run-published-cli-pairing.sh"
VALIDATOR = ROOT / "scripts/release/validate_published_cli_pairing.py"
WORKFLOW = ROOT / ".github/workflows/release-candidate-published-cli-smoke.yml"


class HostedPairingTests(unittest.TestCase):
    def test_pins_are_exact_two_model_contract(self) -> None:
        pins = json.loads(PINS.read_text(encoding="utf-8"))
        self.assertEqual(set(pins), {"whisper-base", "silero-vad-v5"})
        self.assertEqual(pins["whisper-base"]["size_bytes"], 290859488)
        self.assertEqual(pins["silero-vad-v5"]["size_bytes"], 2183744)
        self.assertTrue(all(set(row) == {"repo", "revision", "filename", "sha256", "size_bytes"} for row in pins.values()))

    def test_metadata_only_path_is_offline_and_driver_free(self) -> None:
        env = os.environ.copy()
        env["UV_PYTHON_DOWNLOADS"] = "never"
        env["UV_CACHE_DIR"] = tempfile.mkdtemp(prefix="vokra-uv-")
        result = subprocess.run(["bash", str(HELPER), "--metadata-only"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("METADATA_PASS", result.stdout)

    def _receipt(self) -> dict:
        pins = json.loads(PINS.read_text(encoding="utf-8"))
        reviewed = {name: {**row, "url": f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{row['filename']}"} for name, row in pins.items()}
        commands = [{"label": label, "returncode": 0, "argv": ["driver", "--backend", "cpu"], "stdout": f"{label} stdout", "stderr": ""} for label in ("asr-jfk", "vad-jfk", "vad-silence")]
        models = [{"name": name, **row, "url": reviewed[name]["url"], "actual_sha256": row["sha256"], "expected_sha256": row["sha256"], "actual_bytes": row["size_bytes"], "expected_bytes": row["size_bytes"]} for name, row in pins.items()]
        return {"status": "PASS", "schema": 1, "runtime": {"head": "HEAD", "expected_head": "HEAD", "platform": "linux", "github_actions": "true", "marker": "1"}, "commands": commands, "reviewed_pins": reviewed, "models": models, "audio": {"jfk": {"sha256": "fixture-jfk", "bytes": 123}, "generated_silence": {"sha256": "fixture-silence", "bytes": 456}}, "functional_only": True, "semantic_checks": {"asr_jfk_keyphrase": {"passed": True}, "vad": {"passed": True}}}

    def _validate_receipt(self, receipt: dict) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="vokra-receipt-") as td:
            root = Path(td)
            rec = root / "receipt.json"
            out = root / "out.json"
            rec.write_text(json.dumps(receipt), encoding="utf-8")
            env = os.environ.copy()
            env["UV_PYTHON_DOWNLOADS"] = "never"
            env["UV_CACHE_DIR"] = str(root / "uv-cache")
            return subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "python", str(VALIDATOR), "--pins", str(PINS), "--head", "HEAD", "--receipt", str(rec), "--output", str(out)], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

    def test_validator_rejects_model_hash_duplicate_command_and_runtime(self) -> None:
        cases = []
        bad_model = self._receipt()
        bad_model["models"][0]["actual_sha256"] = "0" * 64
        cases.append(bad_model)
        bad_command = self._receipt()
        bad_command["commands"][1]["label"] = "asr-jfk"
        cases.append(bad_command)
        bad_returncode = self._receipt()
        bad_returncode["commands"][2]["returncode"] = 1
        cases.append(bad_returncode)
        bad_url = self._receipt()
        bad_url["models"][0]["url"] = "https://example.invalid/model.gguf"
        cases.append(bad_url)
        bad_runtime = self._receipt()
        bad_runtime["runtime"]["github_actions"] = "false"
        cases.append(bad_runtime)
        bad_status = self._receipt()
        bad_status["status"] = "FAIL"
        cases.append(bad_status)
        for receipt in cases:
            self.assertNotEqual(self._validate_receipt(receipt).returncode, 0)

    def test_validator_success_retains_full_driver_receipt(self) -> None:
        receipt = self._receipt()
        with tempfile.TemporaryDirectory(prefix="vokra-retain-") as td:
            root = Path(td)
            rec = root / "receipt.json"
            out = root / "evidence.json"
            rec.write_text(json.dumps(receipt), encoding="utf-8")
            env = os.environ.copy()
            env["UV_PYTHON_DOWNLOADS"] = "never"
            env["UV_CACHE_DIR"] = str(root / "uv-cache")
            result = subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "python", str(VALIDATOR), "--pins", str(PINS), "--head", "HEAD", "--receipt", str(rec), "--output", str(out)], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(evidence["driver_receipt"], receipt)
            self.assertLessEqual(out.stat().st_size, 64 * 1024)

    def test_validator_rejects_pin_url(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-pins-") as td:
            bad = Path(td) / "pins.json"
            pins = json.loads(PINS.read_text(encoding="utf-8"))
            pins["whisper-base"]["url"] = "https://example.invalid/not-derived"
            bad.write_text(json.dumps(pins), encoding="utf-8")
            env = os.environ.copy()
            env["UV_PYTHON_DOWNLOADS"] = "never"
            env["UV_CACHE_DIR"] = str(Path(td) / "uv-cache")
            result = subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "python", str(VALIDATOR), "--metadata-only", "--pins", str(bad), "--head", "metadata"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)

    def test_declined_consent_writes_block_receipt_without_network(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vokra-runner-") as td:
            env = os.environ.copy()
            env.update({"GITHUB_ACTIONS": "true", "RUNNER_TEMP": td, "UV_PYTHON_DOWNLOADS": "never", "UV_CACHE_DIR": str(Path(td) / "uv-cache")})
            result = subprocess.run(["bash", str(HELPER), "--allow-models", "false"], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            evidence = Path(td) / "vokra-rc1-pairing-evidence/pairing-receipt.json"
            self.assertEqual(json.loads(evidence.read_text(encoding="utf-8"))["status"], "BLOCKED")

    def test_workflow_is_manual_sparse_read_only_and_uploads_only_json(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertNotIn("push:", text)
        self.assertIn("type: boolean", text)
        self.assertIn("ALLOW_MODELS: ${{ inputs.allow_models }}", text)
        self.assertIn('if test "$ALLOW_MODELS" = true; then exit 0; fi', text)
        self.assertIn("permissions:\n  contents: read", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("lfs: false", text)
        self.assertIn("sparse-checkout-cone-mode: false", text)
        self.assertIn("scripts/release/test_published_cli_pairing.py", text)
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", text)
        self.assertIn("astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7", text)
        self.assertIn("enable-cache: false", text)
        self.assertIn("run: uv run --no-project --python 3.12 python scripts/release/test_published_cli_pairing.py", text)
        self.assertNotIn("actions/cache", text)
        self.assertIn("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", text)
        self.assertIn("runner.temp }}/vokra-rc1-pairing-evidence/*.json", text)
        self.assertNotIn("--push", text)

    def test_sparse_checkout_projects_every_offline_test_input(self) -> None:
        lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
        marker = lines.index("          sparse-checkout: |")
        sparse = set()
        for line in lines[marker + 1:]:
            if line.startswith("            "):
                sparse.add(line.strip())
            else:
                break
        required = {
            path.relative_to(ROOT).as_posix()
            for path in (PINS, HELPER, VALIDATOR, WORKFLOW, Path(__file__).resolve())
        }
        self.assertTrue(required <= sparse)

        selected = "HostedPairingTests.test_workflow_is_manual_sparse_read_only_and_uploads_only_json"

        def run_projected(include_workflow: bool) -> subprocess.CompletedProcess[str]:
            with tempfile.TemporaryDirectory(prefix="vokra-projected-") as td:
                projected_root = Path(td)
                sources = [PINS, HELPER, VALIDATOR, Path(__file__).resolve()]
                if include_workflow:
                    sources.append(WORKFLOW)
                for source in sources:
                    destination = projected_root / source.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                env = os.environ.copy()
                env["UV_PYTHON_DOWNLOADS"] = "never"
                env["UV_CACHE_DIR"] = str(projected_root / "uv-cache")
                return subprocess.run(["uv", "run", "--no-project", "--python", "3.12", "python", str(projected_root / Path(__file__).relative_to(ROOT)), selected], cwd=projected_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

        negative = run_projected(include_workflow=False)
        self.assertNotEqual(negative.returncode, 0)
        self.assertIn("FileNotFoundError", negative.stderr)
        self.assertIn("release-candidate-published-cli-smoke.yml", negative.stderr)
        positive = run_projected(include_workflow=True)
        self.assertEqual(positive.returncode, 0, positive.stderr)


if __name__ == "__main__":
    unittest.main()
