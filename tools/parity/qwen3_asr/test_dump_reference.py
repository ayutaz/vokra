from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
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
                    "--wheel",
                    "/tmp/qwen_asr.whl",
                    "--audio",
                    "/tmp/audio.wav",
                    "--output",
                    "/tmp/out",
                ]
            )

    def test_direct_backend_path_has_no_legacy_wrapper_name(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertNotIn("asr", loaded)

    def test_main_refuses_non_vast_before_touching_inputs(self) -> None:
        args = [
            "--variant", "0.6b", "--model-dir", "/missing/model",
            "--wheel", "/missing/wheel.whl", "--audio", "/missing/audio.wav",
            "--output", "/missing/output",
        ]
        with mock.patch.dict(MODULE.os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as error:
                MODULE.main(args)
        self.assertIn("VOKRA_PUBLISH_ON_VAST", str(error.exception))

    def test_input_and_output_paths_reject_symlink_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(SystemExit):
                MODULE.require_no_symlink_ancestors(link / "model", "model")
            with self.assertRaises(SystemExit):
                MODULE.require_empty_output(link / "reference")

    def test_source_inventory_requires_exact_index_shards_and_unique_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"shard")
            (model_dir / "model.safetensors.index.json").write_text(
                '{"metadata": {"total_size": 5}, "weight_map": '
                '{"tensor": "model-00001-of-00001.safetensors"}}',
                encoding="utf-8",
            )
            inventory = MODULE.source_inventory(model_dir)
            self.assertIn("model-00001-of-00001.safetensors", inventory)
            (model_dir / "extra.safetensors").write_bytes(b"extra")
            with self.assertRaises(SystemExit):
                MODULE.source_inventory(model_dir)
            (model_dir / "extra.safetensors").unlink()
            (model_dir / "model.safetensors.index.json").write_text(
                '{"metadata": {}, "metadata": {}, "weight_map": '
                '{"tensor": "model-00001-of-00001.safetensors"}}',
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                MODULE.source_inventory(model_dir)


if __name__ == "__main__":
    unittest.main()
