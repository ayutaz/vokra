from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dump_reference.py")
SPEC = importlib.util.spec_from_file_location("qwen3_asr_dump_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DumperContractTests(unittest.TestCase):
    def test_variants_match_converter_identity(self) -> None:
        self.assertEqual(MODULE.VARIANTS["0.6b"].revision, "5eb144179a02acc5e5ba31e748d22b0cf3e303b0")
        self.assertEqual(MODULE.VARIANTS["0.6b"].hidden_size, 1024)
        self.assertEqual(MODULE.VARIANTS["0.6b"].tensor_count, 612)
        self.assertEqual(MODULE.VARIANTS["1.7b"].revision, "7278e1e70fe206f11671096ffdd38061171dd6e5")
        self.assertEqual(MODULE.VARIANTS["1.7b"].hidden_size, 2048)
        self.assertEqual(MODULE.VARIANTS["1.7b"].tensor_count, 708)

    def test_manifest_is_sorted_and_rejects_ambiguous_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.txt"
            MODULE.write_manifest(path, {"z": 2, "a": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), "a=1\nz=2\n")
        for value in ["a=b", "a\nb", "a\rb"]:
            with self.assertRaises(ValueError):
                MODULE.safe_manifest_value(value)

    def test_cli_refuses_unknown_variant(self) -> None:
        with self.assertRaises(SystemExit):
            MODULE.parse_args(
                [
                    "--variant",
                    "9b",
                    "--model-dir",
                    "/tmp/model",
                    "--audio",
                    "/tmp/audio.wav",
                    "--output",
                    "/tmp/out",
                ]
            )


if __name__ == "__main__":
    unittest.main()
