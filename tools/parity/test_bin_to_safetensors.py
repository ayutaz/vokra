import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

import bin_to_safetensors


class ExactCounterSkipTest(unittest.TestCase):
    def test_exact_integer_scalar_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.bin"
            output = root / "model.safetensors"
            counter = "layer.batch_norm.num_batches_tracked"
            torch.save(
                {
                    "weight": torch.tensor([1.0, -2.0]),
                    counter: torch.tensor(7, dtype=torch.int64),
                },
                source,
            )

            count, params = bin_to_safetensors.convert_bin_to_safetensors(
                source, output, frozenset({counter})
            )

            self.assertEqual((count, params), (1, 2))
            self.assertEqual(set(load_file(output)), {"weight"})

    def test_missing_or_non_scalar_counter_fails_without_output(self) -> None:
        for state_dict, skip_name in (
            ({"weight": torch.tensor([1.0])}, "missing"),
            (
                {
                    "weight": torch.tensor([1.0]),
                    "counter": torch.tensor([7], dtype=torch.int64),
                },
                "counter",
            ),
        ):
            with self.subTest(skip_name=skip_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "model.bin"
                output = root / "model.safetensors"
                torch.save(state_dict, source)

                with self.assertRaises(SystemExit):
                    bin_to_safetensors.convert_bin_to_safetensors(
                        source, output, frozenset({skip_name})
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
