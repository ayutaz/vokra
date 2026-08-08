#!/usr/bin/env python3
"""Unit tests for Wave-4 CLEAN-ROOM-DEFAULTS shape-derivation in
``sbv2_prepare_checkpoint.py::_derive_shape_fields`` +
``build_config_side_car``.

Pure-stdlib and network-free: no HF download, no huggingface_hub,
no safetensors PyPI dependency — the tests hand-craft safetensors
header dicts (the same JSON blob shape ``read_safetensors_header``
returns after parsing) and assert what the derivation produces.

Run:  python3 -m unittest discover tools/parity -p "test_sbv2_prepare_*.py"
"""
from __future__ import annotations

import unittest
from pathlib import Path

# Import the module under test. Prep script isn't installed as a package
# so we add its directory to sys.path.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbv2_prepare_checkpoint as prep  # noqa: E402


def entry(shape: list[int]) -> dict:
    """A synthetic safetensors header entry with only the fields the
    derivation reads (shape). Data-offsets/dtype are irrelevant for
    shape-derivation tests."""
    return {"dtype": "F32", "shape": shape, "data_offsets": [0, 1]}


class ShapeDerivationTests(unittest.TestCase):
    def test_derives_all_available_fields_from_real_base_ckpt_shape(self):
        """A synthetic safetensors header modeled on the real
        litagin/Style-Bert-VITS2-2.0-base-JP-Extra ckpt. Every
        shape-derivable field must be recovered; the fallback
        (`emb_g.weight`) is NOT reached because the primary
        (`enc_p.encoder.spk_emb_linear.weight`) is present."""
        header = {
            "enc_p.emb.weight": entry([112, 192]),
            "enc_p.tone_emb.weight": entry([12, 192]),
            "enc_p.language_emb.weight": entry([3, 192]),
            "enc_p.encoder.spk_emb_linear.weight": entry([192, 512]),
            "emb_g.weight": entry([5, 512]),
            "dec.ups.0.weight_v": entry([512, 256, 16]),
            "dec.ups.1.weight_v": entry([256, 128, 16]),
            "dec.ups.2.weight_v": entry([128, 64, 8]),
            "dec.ups.3.weight_v": entry([64, 32, 2]),
            "dec.ups.4.weight_v": entry([32, 16, 2]),
        }
        config, provenance = prep._derive_shape_fields(header)
        self.assertEqual(config["n_vocab"], 112)
        self.assertEqual(config["n_tones"], 12)
        self.assertEqual(config["d_model"], 192)
        self.assertEqual(config["d_speaker"], 512)
        self.assertEqual(config["n_speakers"], 5)
        self.assertEqual(
            config["decoder_upsample_kernel_sizes"], [16, 16, 8, 2, 2]
        )
        # Every derived key must have a provenance entry.
        for k in config:
            self.assertIn(k, provenance)
            self.assertIn("safetensors shape", provenance[k])

    def test_falls_back_to_emb_g_for_d_speaker_when_spk_emb_linear_absent(self):
        """A fine-tune that dropped `spk_emb_linear` still gets a valid
        d_speaker from the emb_g fallback."""
        header = {
            "emb_g.weight": entry([1, 384]),
        }
        config, provenance = prep._derive_shape_fields(header)
        self.assertEqual(config["d_speaker"], 384)
        self.assertEqual(config["n_speakers"], 1)
        self.assertIn("fallback", provenance["d_speaker"])

    def test_empty_header_derives_nothing(self):
        """A header with no shape-derivable tensors must return empty
        dicts — never a fabricated placeholder."""
        config, provenance = prep._derive_shape_fields({})
        self.assertEqual(config, {})
        self.assertEqual(provenance, {})

    def test_kernel_probe_stops_at_first_gap(self):
        """A partial ladder (stages 0 + 1 present, stage 2 absent) stops
        the probe cleanly at 2 kernels — never extrapolates."""
        header = {
            "dec.ups.0.weight": entry([64, 32, 16]),
            "dec.ups.1.weight": entry([32, 16, 8]),
            # dec.ups.2.* absent.
        }
        config, _ = prep._derive_shape_fields(header)
        self.assertEqual(config["decoder_upsample_kernel_sizes"], [16, 8])

    def test_kernel_probe_prefers_weight_v_over_weight(self):
        """When both `.weight_v` (weight-normed) and `.weight` (plain)
        are present, the derivation picks weight_v (the raw pre-norm
        tensor)."""
        header = {
            "dec.ups.0.weight_v": entry([64, 32, 20]),
            "dec.ups.0.weight": entry([64, 32, 4]),  # stray, must be ignored
        }
        config, _ = prep._derive_shape_fields(header)
        self.assertEqual(config["decoder_upsample_kernel_sizes"], [20])

    def test_malformed_shape_entry_is_silently_skipped(self):
        """A tensor entry with a non-list shape (malformed safetensors)
        must be treated as absent — never a crash, never a fabricated
        default."""
        header = {
            "enc_p.emb.weight": {"dtype": "F32", "shape": "malformed", "data_offsets": [0, 1]},
        }
        config, _ = prep._derive_shape_fields(header)
        self.assertNotIn("n_vocab", config)
        self.assertNotIn("d_model", config)

    def test_build_config_side_car_honors_shape_derivation_over_clean_room(self):
        """When shape derivation gives d_speaker=512 (real ckpt) but
        clean-room defaults would have shipped 256 (VITS convention),
        the shape-derived value wins."""
        header = {
            "enc_p.encoder.spk_emb_linear.weight": entry([192, 512]),
            "emb_g.weight": entry([1, 512]),
            "enc_p.emb.weight": entry([112, 192]),
            "enc_p.tone_emb.weight": entry([12, 192]),
        }
        config, provenance, _ = prep.build_config_side_car(
            upstream={},
            use_clean_room=True,
            safetensors_header=header,
        )
        self.assertEqual(config["d_speaker"], 512)
        self.assertIn("safetensors shape", provenance["d_speaker"])
        self.assertEqual(config["n_vocab"], 112)
        self.assertIn("safetensors shape", provenance["n_vocab"])
        self.assertEqual(config["n_tones"], 12)

    def test_build_config_side_car_falls_back_to_clean_room_when_no_shape(self):
        """Without a safetensors header, clean-room defaults still apply
        (backward compat: pre-Wave-4 call sites keep working)."""
        config, provenance, _ = prep.build_config_side_car(
            upstream={},
            use_clean_room=True,
            safetensors_header=None,
        )
        # d_speaker from clean-room fallback = VITS default 256.
        self.assertEqual(config["d_speaker"], 256)
        self.assertIn("clean-room default", provenance["d_speaker"])

    def test_build_config_side_car_upstream_config_still_wins_when_no_shape(self):
        """Backward compat: upstream config takes precedence over
        clean-room defaults when no safetensors header supplied."""
        config, provenance, _ = prep.build_config_side_car(
            upstream={"model": {"gin_channels": 999}},
            use_clean_room=True,
            safetensors_header=None,
        )
        self.assertEqual(config["d_speaker"], 999)
        self.assertIn("upstream", provenance["d_speaker"])


if __name__ == "__main__":
    unittest.main()
