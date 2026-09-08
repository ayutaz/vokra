from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("dump_reference.py")
SPEC = importlib.util.spec_from_file_location("moss_audio_dump_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DumperContractTests(unittest.TestCase):
    def test_model_loader_is_explicit_high_memory_ordinary_cpu_route(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        model_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        self.assertTrue(model_calls)
        model_call = next(
            node
            for node in model_calls
            if any(
                keyword.arg == "attn_implementation"
                for keyword in node.keywords
            )
        )
        keyword_values = {
            keyword.arg: keyword.value for keyword in model_call.keywords
        }
        self.assertIn("low_cpu_mem_usage", keyword_values)
        self.assertIsInstance(keyword_values["low_cpu_mem_usage"], ast.Constant)
        self.assertIs(keyword_values["low_cpu_mem_usage"].value, False)
        self.assertEqual(MODULE.MIN_REFERENCE_MEMORY_KIB, 120_000_000)

    def test_high_memory_guard_rejects_non_x86_vast_host(self) -> None:
        with (
            patch.dict(MODULE.os.environ, {"VOKRA_PUBLISH_ON_VAST": "1"}),
            patch.object(MODULE.platform, "system", return_value="Linux"),
            patch.object(MODULE.platform, "machine", return_value="aarch64"),
        ):
            with self.assertRaisesRegex(SystemExit, "x86_64"):
                MODULE.require_high_memory_vast_host()

    def test_variants_match_converter_identity(self) -> None:
        self.assertEqual(
            MODULE.VARIANTS["4b"].revision,
            "6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d",
        )
        self.assertEqual(MODULE.VARIANTS["4b"].hidden_size, 2560)
        self.assertEqual(MODULE.VARIANTS["4b"].tensor_count, 901)
        self.assertEqual(
            MODULE.VARIANTS["8b"].revision,
            "6521a39181b47a18f2d9f4b3acfb5bca7b76b57f",
        )
        self.assertEqual(MODULE.VARIANTS["8b"].hidden_size, 4096)
        self.assertEqual(MODULE.VARIANTS["8b"].tensor_count, 901)

    def test_manifest_is_sorted_and_rejects_ambiguous_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.txt"
            MODULE.write_manifest(path, {"z": 2, "a": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), "a=1\nz=2\n")
        for value in ["a=b", "a\nb", "a\rb"]:
            with self.assertRaises(ValueError):
                MODULE.safe_manifest_value(value)

    def test_config_identity_fails_loudly_when_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(SystemExit, "missing pinned file"):
                MODULE.require_model_identity(
                    Path(temporary), MODULE.VARIANTS["4b"]
                )

    def test_cli_refuses_unknown_variant(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--variant",
                    "16b",
                    "--model-dir",
                    "/tmp/model",
                    "--source-dir",
                    "/tmp/source",
                    "--audio",
                    "/tmp/audio.wav",
                    "--output",
                    "/tmp/out",
                ]
            )


if __name__ == "__main__":
    unittest.main()
