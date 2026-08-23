"""Build a validated CoreML Whisper-encoder delegate from a Vokra GGUF.

This is an offline developer tool. Runtime code never imports Python,
coremltools, gguf, NumPy, protobuf, or an ONNX implementation. The generated
sidecar is bound to both the source GGUF and compiled ``.mlmodelc`` tree by
SHA-256 so the Rust runtime can fail closed on stale or mismatched artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import coremltools as ct
import gguf
import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types


MANIFEST_FORMAT: Final = "vokra-coreml-sidecar-v1"
INPUT_NAME: Final = "log_mel"
OUTPUT_NAME: Final = "encoder_hidden"
COMPILED_MODEL_NAME: Final = "whisper-encoder.mlmodelc"
PACKAGE_NAME: Final = "whisper-encoder.mlpackage"
MANIFEST_NAME: Final = "manifest.txt"
MINIMUM_DEPLOYMENT_TARGET: Final = "macOS14"
TREE_HASH_DOMAIN: Final = b"vokra-coreml-tree-v1\0"
COMPUTE_PRECISIONS: Final = frozenset({"float16", "float32"})

ACCEPTED_ARCHS: Final = frozenset(
    {"whisper", "crisper-whisper", "distil-whisper", "kotoba-whisper"}
)

KEY_ARCH: Final = "vokra.model.arch"
KEY_N_MELS: Final = "vokra.whisper.n_mels"
KEY_N_AUDIO_CTX: Final = "vokra.whisper.n_audio_ctx"
KEY_D_MODEL: Final = "vokra.whisper.n_audio_state"
KEY_N_AUDIO_HEAD: Final = "vokra.whisper.n_audio_head"
KEY_N_AUDIO_LAYER: Final = "vokra.whisper.n_audio_layer"
KEY_FFN_DIM: Final = "vokra.whisper.ffn_dim"


@dataclass(frozen=True)
class WhisperConfig:
    """Encoder-only subset of the Vokra Whisper metadata contract."""

    arch: str
    n_mels: int
    n_audio_ctx: int
    d_model: int
    n_audio_head: int
    n_audio_layer: int
    ffn_dim: int

    @property
    def n_frames(self) -> int:
        return self.n_audio_ctx * 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_audio_head

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (1, self.n_mels, self.n_frames)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.n_audio_ctx, self.d_model)


@dataclass(frozen=True)
class LinearWeights:
    weight: np.ndarray
    bias: np.ndarray | None


@dataclass(frozen=True)
class LayerNormWeights:
    gamma: np.ndarray
    beta: np.ndarray


@dataclass(frozen=True)
class EncoderLayerWeights:
    attn_ln: LayerNormWeights
    q: LinearWeights
    k: LinearWeights
    v: LinearWeights
    out: LinearWeights
    mlp_ln: LayerNormWeights
    fc1: LinearWeights
    fc2: LinearWeights


@dataclass(frozen=True)
class EncoderWeights:
    conv1_weight: np.ndarray
    conv1_bias: np.ndarray
    conv2_weight: np.ndarray
    conv2_bias: np.ndarray
    position_embedding: np.ndarray
    layers: tuple[EncoderLayerWeights, ...]
    final_norm: LayerNormWeights


def validate_config(config: WhisperConfig) -> WhisperConfig:
    """Validate the exact whole-encoder delegate boundary, failing closed."""

    if config.arch not in ACCEPTED_ARCHS:
        raise ValueError(
            f"{KEY_ARCH}={config.arch!r} is not one of {sorted(ACCEPTED_ARCHS)}; "
            "Whisper Medusa and foreign partial matches must use their own binder"
        )
    positive = {
        KEY_N_MELS: config.n_mels,
        KEY_N_AUDIO_CTX: config.n_audio_ctx,
        KEY_D_MODEL: config.d_model,
        KEY_N_AUDIO_HEAD: config.n_audio_head,
        KEY_N_AUDIO_LAYER: config.n_audio_layer,
        KEY_FFN_DIM: config.ffn_dim,
    }
    for key, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
    if config.d_model % config.n_audio_head != 0:
        raise ValueError(
            f"{KEY_N_AUDIO_HEAD} ({config.n_audio_head}) must divide "
            f"{KEY_D_MODEL} ({config.d_model})"
        )
    if config.n_frames // 2 != config.n_audio_ctx:
        raise ValueError("Whisper stride-2 encoder input/output length contract is invalid")
    return config


def _field_contents(reader: gguf.GGUFReader, key: str) -> object:
    field = reader.get_field(key)
    if field is None:
        raise ValueError(f"GGUF is missing required metadata key {key!r}")
    return field.contents()


def _metadata_int(reader: gguf.GGUFReader, key: str) -> int:
    value = _field_contents(reader, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"GGUF metadata {key!r} must be an integer, got {value!r}")
    return value


def read_config(reader: gguf.GGUFReader) -> WhisperConfig:
    arch = _field_contents(reader, KEY_ARCH)
    if not isinstance(arch, str):
        raise ValueError(f"GGUF metadata {KEY_ARCH!r} must be a string, got {arch!r}")
    return validate_config(
        WhisperConfig(
            arch=arch,
            n_mels=_metadata_int(reader, KEY_N_MELS),
            n_audio_ctx=_metadata_int(reader, KEY_N_AUDIO_CTX),
            d_model=_metadata_int(reader, KEY_D_MODEL),
            n_audio_head=_metadata_int(reader, KEY_N_AUDIO_HEAD),
            n_audio_layer=_metadata_int(reader, KEY_N_AUDIO_LAYER),
            ffn_dim=_metadata_int(reader, KEY_FFN_DIM),
        )
    )


class TensorReader:
    """Strict GGUF tensor lookup plus official gguf-package dequantization."""

    def __init__(self, reader: gguf.GGUFReader) -> None:
        self._tensors = {tensor.name: tensor for tensor in reader.tensors}

    def require(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise ValueError(f"GGUF is missing required Whisper encoder tensor {name!r}")
        logical_shape = tuple(int(axis) for axis in tensor.shape)
        if logical_shape != shape:
            raise ValueError(
                f"Whisper encoder tensor {name!r} logical shape {logical_shape} "
                f"!= expected {shape}"
            )
        try:
            value = gguf.dequantize(tensor.data, tensor.tensor_type)
        except (NotImplementedError, ValueError) as error:
            raise ValueError(
                f"cannot dequantize {name!r} from GGML type {tensor.tensor_type.name}: {error}"
            ) from error
        # The official gguf reader reverses the logical GGUF dimensions when
        # exposing its NumPy view. Vokra GGUFs keep HF row-major payload bytes
        # and record the HF logical shape in the header, so reshape the flat
        # dequantized stream back to that validated logical shape. This is a
        # reshape, not a transpose: changing element order would corrupt every
        # non-square Linear and Conv weight.
        value = np.ascontiguousarray(value, dtype=np.float32).reshape(shape)
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Whisper encoder tensor {name!r} contains non-finite values")
        return value


def _linear(
    tensors: TensorReader,
    prefix: str,
    in_features: int,
    out_features: int,
    *,
    bias: bool,
) -> LinearWeights:
    weight = tensors.require(f"{prefix}.weight", (out_features, in_features))
    bias_value = tensors.require(f"{prefix}.bias", (out_features,)) if bias else None
    return LinearWeights(weight, bias_value)


def _layer_norm(tensors: TensorReader, prefix: str, width: int) -> LayerNormWeights:
    return LayerNormWeights(
        gamma=tensors.require(f"{prefix}.weight", (width,)),
        beta=tensors.require(f"{prefix}.bias", (width,)),
    )


def load_encoder_weights(reader: gguf.GGUFReader, config: WhisperConfig) -> EncoderWeights:
    tensors = TensorReader(reader)
    d_model = config.d_model
    layers: list[EncoderLayerWeights] = []
    for index in range(config.n_audio_layer):
        prefix = f"model.encoder.layers.{index}"
        attention = f"{prefix}.self_attn"
        layers.append(
            EncoderLayerWeights(
                attn_ln=_layer_norm(tensors, f"{prefix}.self_attn_layer_norm", d_model),
                q=_linear(tensors, f"{attention}.q_proj", d_model, d_model, bias=True),
                k=_linear(tensors, f"{attention}.k_proj", d_model, d_model, bias=False),
                v=_linear(tensors, f"{attention}.v_proj", d_model, d_model, bias=True),
                out=_linear(tensors, f"{attention}.out_proj", d_model, d_model, bias=True),
                mlp_ln=_layer_norm(tensors, f"{prefix}.final_layer_norm", d_model),
                fc1=_linear(tensors, f"{prefix}.fc1", d_model, config.ffn_dim, bias=True),
                fc2=_linear(tensors, f"{prefix}.fc2", config.ffn_dim, d_model, bias=True),
            )
        )
    return EncoderWeights(
        conv1_weight=tensors.require(
            "model.encoder.conv1.weight", (d_model, config.n_mels, 3)
        ),
        conv1_bias=tensors.require("model.encoder.conv1.bias", (d_model,)),
        conv2_weight=tensors.require(
            "model.encoder.conv2.weight", (d_model, d_model, 3)
        ),
        conv2_bias=tensors.require("model.encoder.conv2.bias", (d_model,)),
        position_embedding=tensors.require(
            "model.encoder.embed_positions.weight", (config.n_audio_ctx, d_model)
        ),
        layers=tuple(layers),
        final_norm=_layer_norm(tensors, "model.encoder.layer_norm", d_model),
    )


def _mil_linear(x: object, weights: LinearWeights, name: str) -> object:
    kwargs: dict[str, object] = {"x": x, "weight": weights.weight, "name": name}
    if weights.bias is not None:
        kwargs["bias"] = weights.bias
    return mb.linear(**kwargs)


def _mil_layer_norm(x: object, weights: LayerNormWeights, name: str) -> object:
    return mb.layer_norm(
        x=x,
        axes=[-1],
        gamma=weights.gamma,
        beta=weights.beta,
        epsilon=np.float32(1.0e-5),
        name=name,
    )


def build_program(config: WhisperConfig, weights: EncoderWeights) -> object:
    """Construct the complete Whisper audio encoder as one MIL program."""

    n_ctx = config.n_audio_ctx
    d_model = config.d_model
    n_head = config.n_audio_head
    head_dim = config.head_dim

    @mb.program(
        input_specs=[mb.TensorSpec(shape=config.input_shape, dtype=types.fp32)],
        opset_version=ct.target.macOS14,
    )
    def whisper_encoder(log_mel):
        hidden = mb.conv(
            x=log_mel,
            weight=weights.conv1_weight,
            bias=weights.conv1_bias,
            strides=[1],
            pad_type="custom",
            pad=[1, 1],
            name="conv1",
        )
        hidden = mb.gelu(x=hidden, mode="EXACT", name="conv1_gelu")
        hidden = mb.conv(
            x=hidden,
            weight=weights.conv2_weight,
            bias=weights.conv2_bias,
            strides=[2],
            pad_type="custom",
            pad=[1, 1],
            name="conv2",
        )
        hidden = mb.gelu(x=hidden, mode="EXACT", name="conv2_gelu")
        hidden = mb.transpose(x=hidden, perm=[0, 2, 1], name="channels_to_hidden")
        hidden = mb.add(
            x=hidden,
            y=weights.position_embedding[np.newaxis, :, :],
            name="add_position_embedding",
        )

        for index, layer in enumerate(weights.layers):
            prefix = f"encoder_layer_{index}"
            residual = hidden
            normed = _mil_layer_norm(hidden, layer.attn_ln, f"{prefix}_attn_norm")
            query = _mil_linear(normed, layer.q, f"{prefix}_query")
            # Algebraically this matches Vokra's q * head_dim^-0.5, while
            # following OpenAI Whisper's numerically stable fp16 form: split
            # the scale equally across q and k before QK^T. Applying the full
            # scale to q alone creates large ANE fp16 errors on real speech.
            attention_scale = np.float32(head_dim**-0.25)
            query = mb.mul(
                x=query,
                y=attention_scale,
                name=f"{prefix}_query_scale",
            )
            key = _mil_linear(normed, layer.k, f"{prefix}_key")
            key = mb.mul(
                x=key,
                y=attention_scale,
                name=f"{prefix}_key_scale",
            )
            value = _mil_linear(normed, layer.v, f"{prefix}_value")

            query = mb.reshape(
                x=query,
                shape=[1, n_ctx, n_head, head_dim],
                name=f"{prefix}_query_heads",
            )
            key = mb.reshape(
                x=key,
                shape=[1, n_ctx, n_head, head_dim],
                name=f"{prefix}_key_heads",
            )
            value = mb.reshape(
                x=value,
                shape=[1, n_ctx, n_head, head_dim],
                name=f"{prefix}_value_heads",
            )
            query = mb.transpose(x=query, perm=[0, 2, 1, 3])
            key = mb.transpose(x=key, perm=[0, 2, 1, 3])
            value = mb.transpose(x=value, perm=[0, 2, 1, 3])

            scores = mb.matmul(
                x=query,
                y=key,
                transpose_y=True,
                name=f"{prefix}_attention_scores",
            )
            probabilities = mb.softmax(
                x=scores, axis=-1, name=f"{prefix}_attention_softmax"
            )
            attended = mb.matmul(
                x=probabilities, y=value, name=f"{prefix}_attention_values"
            )
            attended = mb.transpose(x=attended, perm=[0, 2, 1, 3])
            attended = mb.reshape(
                x=attended,
                shape=[1, n_ctx, d_model],
                name=f"{prefix}_merge_heads",
            )
            attended = _mil_linear(attended, layer.out, f"{prefix}_attention_output")
            hidden = mb.add(x=residual, y=attended, name=f"{prefix}_attention_residual")

            residual = hidden
            normed = _mil_layer_norm(hidden, layer.mlp_ln, f"{prefix}_mlp_norm")
            mlp = _mil_linear(normed, layer.fc1, f"{prefix}_fc1")
            mlp = mb.gelu(x=mlp, mode="EXACT", name=f"{prefix}_fc1_gelu")
            mlp = _mil_linear(mlp, layer.fc2, f"{prefix}_fc2")
            hidden = mb.add(x=residual, y=mlp, name=f"{prefix}_mlp_residual")

        return _mil_layer_norm(hidden, weights.final_norm, OUTPUT_NAME)

    return whisper_encoder


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a compiled model tree using the Rust runtime's v1 framing."""

    if not root.is_dir():
        raise ValueError(f"compiled model tree does not exist: {root}")
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"compiled model tree must not contain symlinks: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append((relative, path))
        elif not path.is_dir():
            raise ValueError(f"unsupported compiled model tree entry: {path}")
    if not files:
        raise ValueError(f"compiled model tree contains no files: {root}")

    digest = hashlib.sha256()
    digest.update(TREE_HASH_DOMAIN)
    for relative, path in sorted(files):
        encoded = relative.encode("utf-8")
        size = path.stat().st_size
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", size))
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def render_manifest(
    *,
    config: WhisperConfig,
    source_gguf_sha256: str,
    compiled_tree_sha256: str,
    compute_precision: str,
    coremltools_version: str,
) -> str:
    for label, digest in (
        ("source GGUF", source_gguf_sha256),
        ("compiled model tree", compiled_tree_sha256),
    ):
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{label} SHA-256 is not canonical lowercase hex: {digest!r}")
    if compute_precision not in COMPUTE_PRECISIONS:
        raise ValueError(f"unsupported CoreML compute precision: {compute_precision!r}")

    def shape(value: tuple[int, int, int]) -> str:
        return ",".join(str(axis) for axis in value)

    rows = (
        ("format", MANIFEST_FORMAT),
        ("submodel", "whisper_encoder"),
        ("model_arch", config.arch),
        ("source_gguf_sha256", source_gguf_sha256),
        ("compiled_model", COMPILED_MODEL_NAME),
        ("compiled_tree_sha256", compiled_tree_sha256),
        ("input_name", INPUT_NAME),
        ("output_name", OUTPUT_NAME),
        ("input_shape", shape(config.input_shape)),
        ("output_shape", shape(config.output_shape)),
        ("compute_precision", compute_precision),
        ("minimum_deployment_target", MINIMUM_DEPLOYMENT_TARGET),
        ("coremltools_version", coremltools_version),
    )
    return "".join(f"{key}={value}\n" for key, value in rows)


def build_sidecar(
    gguf_path: Path,
    output_dir: Path,
    *,
    compute_precision: str,
    keep_package: bool,
) -> None:
    if not gguf_path.is_file():
        raise ValueError(f"GGUF path is not a regular file: {gguf_path}")
    if (
        gguf_path.stat().st_size >= 2 * 1024 * 1024 * 1024
        and os.environ.get("VOKRA_ON_VAST") != "1"
    ):
        raise ValueError(
            "GGUF is at least 2 GiB; Vokra policy requires running this conversion on vast.ai "
            "with VOKRA_ON_VAST=1 set by the remote runbook"
        )
    if not output_dir.is_dir():
        raise ValueError(f"output directory must already exist: {output_dir}")
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    if compute_precision not in COMPUTE_PRECISIONS:
        raise ValueError(f"unsupported compute precision: {compute_precision}")

    source_digest = file_sha256(gguf_path)
    reader = gguf.GGUFReader(str(gguf_path), mode="r")
    config = read_config(reader)
    weights = load_encoder_weights(reader, config)
    program = build_program(config, weights)
    precision = (
        ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
    )
    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=precision,
    )
    model.author = "Vokra offline CoreML converter"
    model.license = "Weights retain the source GGUF model license"
    model.short_description = "Whole Whisper audio encoder delegate for Vokra"
    model.user_defined_metadata["vokra.manifest_format"] = MANIFEST_FORMAT
    model.user_defined_metadata["vokra.submodel"] = "whisper_encoder"
    model.user_defined_metadata["vokra.source_gguf_sha256"] = source_digest
    model.user_defined_metadata["vokra.model_arch"] = config.arch
    model.user_defined_metadata["vokra.compute_precision"] = compute_precision

    package = output_dir / PACKAGE_NAME
    model.save(str(package))
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(output_dir)],
        check=True,
    )
    compiled = output_dir / COMPILED_MODEL_NAME
    if not compiled.is_dir():
        raise RuntimeError(f"coremlcompiler did not produce {compiled}")

    compiled_digest = tree_sha256(compiled)
    manifest = render_manifest(
        config=config,
        source_gguf_sha256=source_digest,
        compiled_tree_sha256=compiled_digest,
        compute_precision=compute_precision,
        coremltools_version=ct.__version__,
    )
    (output_dir / MANIFEST_NAME).write_text(manifest, encoding="utf-8", newline="\n")
    if not keep_package:
        shutil.rmtree(package)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build <model.gguf>.coreml/ for Vokra's Whisper encoder delegate"
    )
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compute-precision",
        choices=tuple(sorted(COMPUTE_PRECISIONS)),
        default="float16",
    )
    parser.add_argument(
        "--keep-package",
        action="store_true",
        help="retain the portable .mlpackage in addition to the runtime .mlmodelc",
    )
    args = parser.parse_args()
    build_sidecar(
        args.gguf,
        args.output_dir,
        compute_precision=args.compute_precision,
        keep_package=args.keep_package,
    )
    print(args.output_dir / COMPILED_MODEL_NAME)


if __name__ == "__main__":
    main()
