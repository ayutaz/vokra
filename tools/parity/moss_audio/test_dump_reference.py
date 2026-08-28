from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dump_reference.py")
SPEC = importlib.util.spec_from_file_location("moss_audio_dump_reference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DumperContractTests(unittest.TestCase):
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
