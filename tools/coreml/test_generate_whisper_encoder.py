"""Contract tests for the offline Whisper CoreML artifact builder.

These tests cover validation and byte-level artifact binding only. They are
not numerical parity: delegate numerical parity is measured against Vokra's
independently-tested Rust CPU encoder on a real Whisper GGUF.
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gguf import GGMLQuantizationType

from generate_whisper_encoder import (
    EncoderLayerWeights,
    EncoderWeights,
    LayerNormWeights,
    LinearWeights,
    MANIFEST_FORMAT,
    TensorReader,
    WhisperConfig,
    build_program,
    render_manifest,
    tree_sha256,
    validate_config,
)


def tiny_config(**overrides: object) -> WhisperConfig:
    values: dict[str, object] = {
        "arch": "whisper",
        "n_mels": 2,
        "n_audio_ctx": 2,
        "d_model": 4,
        "n_audio_head": 2,
        "n_audio_layer": 1,
        "ffn_dim": 8,
    }
    values.update(overrides)
    return WhisperConfig(**values)


class ConfigContractTests(unittest.TestCase):
    def test_valid_config_derives_fixed_delegate_shapes(self) -> None:
        config = validate_config(tiny_config())
        self.assertEqual(config.input_shape, (1, 2, 4))
        self.assertEqual(config.output_shape, (1, 2, 4))
        self.assertEqual(config.head_dim, 2)

    def test_foreign_arch_and_non_divisible_heads_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "vokra.model.arch"):
            validate_config(tiny_config(arch="whisper-medusa-v1"))
        with self.assertRaisesRegex(ValueError, "divide"):
            validate_config(tiny_config(d_model=5))

    def test_mil_program_covers_the_complete_encoder_shape(self) -> None:
        config = validate_config(tiny_config())
        zeros = lambda shape: np.zeros(shape, dtype=np.float32)
        norm = LayerNormWeights(zeros((4,)), zeros((4,)))
        attention = LinearWeights(zeros((4, 4)), zeros((4,)))
        key = LinearWeights(zeros((4, 4)), None)
        weights = EncoderWeights(
            conv1_weight=zeros((4, 2, 3)),
            conv1_bias=zeros((4,)),
            conv2_weight=zeros((4, 4, 3)),
            conv2_bias=zeros((4,)),
            position_embedding=zeros((2, 4)),
            layers=(
                EncoderLayerWeights(
                    attn_ln=norm,
                    q=attention,
                    k=key,
                    v=attention,
                    out=attention,
                    mlp_ln=norm,
                    fc1=LinearWeights(zeros((8, 4)), zeros((8,))),
                    fc2=LinearWeights(zeros((4, 8)), zeros((4,))),
                ),
            ),
            final_norm=norm,
        )
        program = build_program(config, weights)
        self.assertEqual(tuple(program.functions["main"].outputs[0].shape), (1, 2, 4))
        self.assertEqual(program.functions["main"].outputs[0].name, "encoder_hidden")
        operation_names = {op.name for op in program.functions["main"].operations}
        self.assertIn("encoder_layer_0_query_scale", operation_names)
        self.assertIn("encoder_layer_0_key_scale", operation_names)

    def test_tensor_reader_uses_vokra_logical_shape_not_gguf_numpy_view(self) -> None:
        # Vokra records logical dimensions in GGUF order.  The official gguf
        # Python reader exposes the same bytes through a reversed-rank NumPy
        # view; flattening and reshaping restores Vokra's logical row-major
        # tensor without transposing values.
        raw = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
        tensor = SimpleNamespace(
            name="weight",
            shape=np.array([4, 2, 3], dtype=np.uint64),
            data=raw,
            tensor_type=GGMLQuantizationType.F32,
        )
        value = TensorReader(SimpleNamespace(tensors=[tensor])).require(
            "weight", (4, 2, 3)
        )
        self.assertEqual(value.shape, (4, 2, 3))
        np.testing.assert_array_equal(value.reshape(-1), raw.reshape(-1))

        tensor.shape = np.array([3, 2, 4], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "logical shape"):
            TensorReader(SimpleNamespace(tensors=[tensor])).require(
                "weight", (4, 2, 3)
            )


class ManifestContractTests(unittest.TestCase):
    def test_tree_digest_is_sorted_and_domain_separated(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "z.bin").write_bytes(b"z")
            (root / "nested").mkdir()
            (root / "nested" / "a.bin").write_bytes(b"alpha")

            expected = hashlib.sha256()
            expected.update(b"vokra-coreml-tree-v1\0")
            for relative, payload in (("nested/a.bin", b"alpha"), ("z.bin", b"z")):
                encoded = relative.encode("utf-8")
                expected.update(struct.pack("<Q", len(encoded)))
                expected.update(encoded)
                expected.update(struct.pack("<Q", len(payload)))
                expected.update(payload)

            self.assertEqual(tree_sha256(root), expected.hexdigest())

    def test_manifest_is_stable_and_carries_full_binding_contract(self) -> None:
        text = render_manifest(
            config=validate_config(tiny_config()),
            source_gguf_sha256="1" * 64,
            compiled_tree_sha256="2" * 64,
            compute_precision="float16",
            coremltools_version="9.0",
        )
        self.assertEqual(text.splitlines()[0], f"format={MANIFEST_FORMAT}")
        self.assertIn("submodel=whisper_encoder\n", text)
        self.assertIn("input_name=log_mel\n", text)
        self.assertIn("output_name=encoder_hidden\n", text)
        self.assertIn("input_shape=1,2,4\n", text)
        self.assertIn("output_shape=1,2,4\n", text)
        self.assertTrue(text.endswith("coremltools_version=9.0\n"))

if __name__ == "__main__":
    unittest.main()
