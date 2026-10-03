#!/usr/bin/env python3
"""Dump an independent official X-Codec2 token-to-PCM reference.

The oracle imports ``CodecDecoderVocos`` from the official
``xcodec2==0.1.5`` PyPI package, restores the audited public Vokra GGUF into
those official modules, and calls the upstream FSQ + decoder forward. It never
imports Vokra or mirrors the forward equations.

The released source imports ``RotaryPositionalEmbeddings`` through
``torchtune.__init__``, which also imports unrelated torchao modules. This
tool loads the exact ``torchtune==0.3.1`` position-embedding source directly
after verifying its SHA-256, then exposes that official class at
``torchtune.modules``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import importlib
import importlib.metadata
import importlib.util
import json
from dataclasses import dataclass
import sys
import types
from pathlib import Path
from pathlib import PurePosixPath

import dependency_audit as audit


XCODEC2_VERSION = "0.1.5"
TORCH_VERSION = "2.13.0"
TORCHAUDIO_VERSION = "2.11.0"
XCODEC2_SDIST_SHA256 = (
    "dc1a73b32090706e65fb73b2469411bc27bb72048677a23b430ab21ad325e45b"
)
DECODER_SOURCE_SHA256 = (
    "8a770d35c4d90a3a82b38869b7b39bd6fab6ab7b2079a44915c7740549f19282"
)
TRANSFORMER_SOURCE_SHA256 = (
    "54786751f363ed6ea510c7a4a13d5c093cd392f79c545ce31dedcd745d6662d0"
)
GGUF_SHA256 = "7ab4b94006068226b0741930081f7e149316e045511c1cddb94769e7f598698e"
TORCHTUNE_VERSION = "0.3.1"
TORCHTUNE_ROPE_SHA256 = (
    "8d79a03e1334fe6ecaff14b1e6a2d554e7e6209c95db058846973de013f92b80"
)
VECTOR_QUANTIZE_VERSION = "1.17.8"
CODEBOOK_SIZE = 65_536
HOP_LENGTH = 320
HIDDEN_DIM = 1_024
EXPECTED_DISTRIBUTIONS = {
    "numpy": "2.0.2",
    "gguf": "0.19.0",
    "torch": TORCH_VERSION,
    "torchaudio": TORCHAUDIO_VERSION,
    "torchtune": TORCHTUNE_VERSION,
    "vector-quantize-pytorch": VECTOR_QUANTIZE_VERSION,
    "xcodec2": XCODEC2_VERSION,
}
EXPECTED_PACKAGE_FILES = {
    "numpy": "numpy/__init__.py",
    "gguf": "gguf/__init__.py",
    "torch": "torch/__init__.py",
    "torchaudio": "torchaudio/__init__.py",
    "torchtune": "torchtune/__init__.py",
    "vector-quantize-pytorch": "vector_quantize_pytorch/__init__.py",
    "xcodec2": "xcodec2/__init__.py",
}
EXPECTED_MODULE_NAMES = {
    "numpy",
    "gguf",
    "torch",
    "torchaudio",
    "torchtune",
    "vector_quantize_pytorch",
    "xcodec2",
}
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_RECORD_ENTRIES = 250_000
MAX_PREIMPORT_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class _PreimportProof:
    versions: dict[str, str]
    record_digests: dict[str, dict[str, tuple[str, int]]]
    nonce: object


_PREIMPORT_NONCE = object()


def validate_execution_authorization() -> None:
    """Honor the existing fail-closed audit contract before any import."""

    try:
        contract = audit.load_contract()
        manifest = contract["manifest"]
        rows = contract["rows"]
        dependency_contract = manifest["dependency_audit"]
        policy = manifest["policy"]
    except Exception as exc:
        raise RuntimeError(
            "official XCodec2 execution is blocked by the existing audit/owner contract"
        ) from exc
    blocked_status = {
        "BLOCKED_PENDING_PRIMARY_BYTES",
        "BLOCKED_FACTUAL_COLLECTION",
        "BLOCKED_OWNER_REVIEW",
    }
    unresolved_policy = (
        not isinstance(dependency_contract, dict)
        or dependency_contract.get("owner_review_required") is not False
        or not isinstance(policy, dict)
        or str(policy.get("license_classification", "")).startswith("UNRESOLVED")
        or str(policy.get("native_payload", "")).startswith("UNRESOLVED")
    )
    if (
        manifest.get("status") in blocked_status
        or rows.get("status") in blocked_status
        or unresolved_policy
    ):
        raise RuntimeError(
            "official XCodec2 execution is blocked by the existing audit/owner contract"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_path_under(root: Path, path: Path, relative: str) -> Path:
    """Resolve a distribution member without following symlink ancestry."""

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"distribution root is not a regular directory: {root}")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ) or "\\" in relative:
        raise RuntimeError(f"distribution member path is unsafe: {relative}")
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"distribution path has symlink ancestry: {relative}")
    if not current.is_file():
        raise RuntimeError(f"distribution path is not a regular file: {relative}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = current.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"distribution path escapes site root: {relative}") from exc
    return current


def _hash_bounded(path: Path, budget: list[int]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            size += len(chunk)
            budget[0] += len(chunk)
            if budget[0] > MAX_PREIMPORT_BYTES:
                raise RuntimeError("pre-import RECORD byte budget exceeded")
            digest.update(chunk)
    return digest.hexdigest(), size


def _parse_record(
    distribution: importlib.metadata.Distribution,
    record_path: str,
    budget: list[int],
) -> dict[str, tuple[str, int]]:
    root = Path(distribution.locate_file(""))
    record_file = _regular_path_under(root, Path(distribution.locate_file(record_path)), record_path)
    try:
        with record_file.open("rb") as handle:
            raw = handle.read(MAX_RECORD_BYTES + 1)
        if len(raw) > MAX_RECORD_BYTES:
            raise RuntimeError(f"publisher RECORD exceeds bound: {record_path}")
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"publisher RECORD is unreadable: {record_path}") from exc
    rows: dict[str, tuple[str, int]] = {}
    try:
        parsed = csv.reader(io.StringIO(text, newline=""))
        for index, row in enumerate(parsed, start=1):
            if index > MAX_RECORD_ENTRIES:
                raise RuntimeError("publisher RECORD entry bound exceeded")
            if len(row) != 3:
                raise RuntimeError(f"publisher RECORD row {index} is malformed")
            relative, encoded_hash, encoded_size = row
            if (
                not relative
                or relative in rows
                or PurePosixPath(relative).is_absolute()
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            ):
                raise RuntimeError(f"publisher RECORD path {relative!r} is invalid or duplicated")
            if relative == record_path:
                if encoded_hash or encoded_size:
                    raise RuntimeError("publisher RECORD self-row must be blank")
                rows[relative] = ("", -1)
                continue
            if not encoded_hash.startswith("sha256=") or not encoded_size.isdigit():
                raise RuntimeError(f"publisher RECORD row {relative!r} lacks hash/size")
            try:
                digest = base64.urlsafe_b64decode(encoded_hash.removeprefix("sha256=") + "===")
            except (ValueError, base64.binascii.Error) as exc:
                raise RuntimeError(f"publisher RECORD hash is malformed: {relative}") from exc
            if len(digest) != hashlib.sha256().digest_size:
                raise RuntimeError(f"publisher RECORD hash length is invalid: {relative}")
            rows[relative] = (digest.hex(), int(encoded_size))
    except csv.Error as exc:
        raise RuntimeError("publisher RECORD CSV is malformed") from exc
    if record_path not in rows:
        raise RuntimeError("publisher RECORD self-row is missing")
    for relative, (expected_hash, expected_size) in rows.items():
        if relative == record_path:
            continue
        actual_hash, actual_size = _hash_bounded(
            _regular_path_under(root, Path(distribution.locate_file(relative)), relative), budget
        )
        if actual_hash != expected_hash or actual_size != expected_size:
            raise RuntimeError(f"publisher RECORD bytes mismatch: {relative}")
    return rows


def _regular_distribution_file(
    distribution: importlib.metadata.Distribution, relative: str
) -> Path:
    """Resolve one RECORD-listed regular file without importing its package."""

    entries = {str(entry) for entry in (distribution.files or [])}
    if relative not in entries:
        raise RuntimeError(
            f"{distribution.metadata.get('Name', '<unknown>')} RECORD omits {relative}"
        )
    root = Path(distribution.locate_file(""))
    return _regular_path_under(root, Path(distribution.locate_file(relative)), relative)


def _check_top_level_origin(name: str, expected_path: Path) -> None:
    """Reject an earlier sys.path shadow before any package code executes."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        raise RuntimeError(f"cannot resolve audited module origin: {name}") from exc
    if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
        raise RuntimeError(f"audited module has no regular origin: {name}")
    try:
        origin = Path(spec.origin).resolve(strict=True)
        expected = expected_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"audited module origin is unreadable: {name}") from exc
    if origin != expected:
        raise RuntimeError(
            f"audited module origin mismatch for {name}: {origin} != {expected}"
        )


def validate_preimport_runtime() -> _PreimportProof:
    """Authenticate installed distribution/RECORD identities before imports.

    ``importlib.metadata`` reads metadata and RECORD only; it does not execute
    package code.  This is intentionally stronger than version checks while
    retaining the blocked license/owner gate in the separate audit manifest.
    """

    if any(name in sys.modules for name in EXPECTED_MODULE_NAMES):
        raise RuntimeError("audited runtime packages were imported before preflight")
    versions: dict[str, str] = {}
    record_digests: dict[str, dict[str, tuple[str, int]]] = {}
    budget = [0]
    for name, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"audited distribution is not installed: {name}") from exc
        actual_name = (distribution.metadata.get("Name") or "").casefold()
        if actual_name != name.casefold():
            raise RuntimeError(f"distribution name mismatch for {name}: {actual_name!r}")
        actual_version = distribution.version
        if actual_version.split("+", 1)[0] != expected:
            raise RuntimeError(f"{name} version {actual_version!r} != audited {expected!r}")
        # A missing RECORD means the installed payload cannot be bound to the
        # selected distribution, even when its metadata version looks right.
        dist_info = [
            str(entry)
            for entry in (distribution.files or [])
            if str(entry).endswith(".dist-info/RECORD")
        ]
        if len(dist_info) != 1:
            raise RuntimeError(f"{name} must expose exactly one RECORD")
        record = _parse_record(distribution, dist_info[0], budget)
        package_relative = EXPECTED_PACKAGE_FILES[name]
        package_path = _regular_distribution_file(distribution, package_relative)
        if package_relative not in record or record[package_relative][0] == "":
            raise RuntimeError(f"{name} RECORD omits package entry: {package_relative}")
        module_name = package_relative.split("/", 1)[0]
        _check_top_level_origin(module_name, package_path)
        record_digests[name] = record
        versions[name] = actual_version
    # These are the source files whose bytes are fixed by the existing oracle
    # contract; checking them before import prevents a same-version replacement.
    xcodec_dist = importlib.metadata.distribution("xcodec2")
    for relative, expected_hash in (
        ("xcodec2/vq/codec_decoder_vocos.py", DECODER_SOURCE_SHA256),
        ("xcodec2/vq/bs_roformer5.py", TRANSFORMER_SOURCE_SHA256),
    ):
        record = record_digests["xcodec2"]
        path = _regular_distribution_file(xcodec_dist, relative)
        if relative not in record or record[relative][0] != expected_hash:
            raise RuntimeError(f"xcodec2 source identity mismatch: {relative}")
    rope_dist = importlib.metadata.distribution("torchtune")
    rope_path = _regular_distribution_file(
        rope_dist, "torchtune/modules/position_embeddings.py"
    )
    rope_record = record_digests["torchtune"]
    if (
        "torchtune/modules/position_embeddings.py" not in rope_record
        or rope_record["torchtune/modules/position_embeddings.py"][0]
        != TORCHTUNE_ROPE_SHA256
    ):
        raise RuntimeError("torchtune official RoPE source SHA-256 mismatch")
    return _PreimportProof(versions, record_digests, _PREIMPORT_NONCE)


def validate_patched_runtime() -> _PreimportProof:
    """Require the reviewed Torch/TorchAudio pair before importing upstream."""

    return validate_preimport_runtime()


def install_official_rope_import() -> None:
    validate_execution_authorization()
    distribution = importlib.metadata.distribution("torchtune")
    source_path = Path(
        distribution.locate_file("torchtune/modules/position_embeddings.py")
    )
    if sha256_file(source_path) != TORCHTUNE_ROPE_SHA256:
        raise RuntimeError("torchtune official RoPE source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location(
        "_vokra_official_torchtune_position_embeddings", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official RoPE source {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torchtune_package = types.ModuleType("torchtune")
    torchtune_modules = types.ModuleType("torchtune.modules")
    torchtune_modules.RotaryPositionalEmbeddings = (
        module.RotaryPositionalEmbeddings
    )
    torchtune_package.modules = torchtune_modules
    sys.modules["torchtune"] = torchtune_package
    sys.modules["torchtune.modules"] = torchtune_modules


def import_official_decoder(proof: _PreimportProof):
    validate_execution_authorization()
    if not isinstance(proof, _PreimportProof) or proof.nonce is not _PREIMPORT_NONCE:
        raise RuntimeError("official decoder import requires a live pre-import proof")
    install_official_rope_import()
    module = importlib.import_module("xcodec2.vq.codec_decoder_vocos")
    decoder_source = Path(module.__file__ or "")
    transformer_source = decoder_source.with_name("bs_roformer5.py")
    if sha256_file(decoder_source) != DECODER_SOURCE_SHA256:
        raise RuntimeError("xcodec2 official decoder source SHA-256 mismatch")
    if sha256_file(transformer_source) != TRANSFORMER_SOURCE_SHA256:
        raise RuntimeError("xcodec2 official Transformer source SHA-256 mismatch")
    return module.CodecDecoderVocos


def _gguf_tensor_after_gate(item, expected_shape):
    import numpy as np
    import torch

    if int(item.tensor_type) != 0:
        raise TypeError(f"{item.name}: public X-Codec2 tensor is not F32")
    values = item.data.copy().reshape(-1).astype(np.float32, copy=False)
    expected_elements = int(np.prod(expected_shape, dtype=np.int64))
    if values.size != expected_elements:
        raise RuntimeError(
            f"{item.name}: {values.size} values != expected {expected_elements}"
        )
    return torch.from_numpy(values.reshape(tuple(expected_shape)).copy())


def gguf_tensor(item, expected_shape):
    """Decode one tensor only after the execution authorization gate."""

    validate_execution_authorization()
    return _gguf_tensor_after_gate(item, expected_shape)


def _required_after_gate(by_name: dict, name: str, shape):
    item = by_name.get(name)
    if item is None:
        raise RuntimeError(f"GGUF is missing official inference tensor {name!r}")
    return _gguf_tensor_after_gate(item, shape)


def required(by_name: dict, name: str, shape):
    """Resolve one tensor only after the execution authorization gate."""

    validate_execution_authorization()
    return _required_after_gate(by_name, name, shape)


def load_official_modules(gguf_path: Path, proof: _PreimportProof):
    decoder_class = import_official_decoder(proof)
    import torch
    from gguf import GGUFReader

    decoder = decoder_class(hop_length=HOP_LENGTH)
    fc_post_a = torch.nn.Linear(2_048, HIDDEN_DIM)
    reader = GGUFReader(str(gguf_path))
    by_name = {item.name: item for item in reader.tensors}

    loaded = {}
    defaulted = []
    for name, target in decoder.state_dict().items():
        item = by_name.get(f"generator.{name}")
        if item is None and name.startswith("quantizer.layers."):
            defaulted.append(name)
        elif item is None:
            raise RuntimeError(f"GGUF is missing official decoder tensor {name!r}")
        else:
            loaded[name] = _gguf_tensor_after_gate(item, target.shape)
    incompatible = decoder.load_state_dict(loaded, strict=False)
    if sorted(incompatible.missing_keys) != sorted(defaulted):
        raise RuntimeError(
            f"official decoder missing keys {incompatible.missing_keys!r} != {defaulted!r}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"official decoder unexpected keys: {incompatible.unexpected_keys}")

    fc_post_a.load_state_dict(
        {
            "weight": _required_after_gate(by_name, "fc_post_a.weight", fc_post_a.weight.shape),
            "bias": _required_after_gate(by_name, "fc_post_a.bias", fc_post_a.bias.shape),
        },
        strict=True,
    )
    decoder.eval()
    fc_post_a.eval()
    return decoder, fc_post_a, len(loaded), defaulted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Keep all third-party imports below the existing audit/owner gate and
    # installed identity gate. Neither gate accepts an environment override.
    validate_execution_authorization()
    preimport = validate_patched_runtime()
    torch_version = preimport.versions["torch"]
    torchaudio_version = preimport.versions["torchaudio"]
    import numpy as np
    import torch

    gguf_sha256 = sha256_file(args.gguf)
    if gguf_sha256 != GGUF_SHA256:
        raise RuntimeError(f"GGUF SHA-256 {gguf_sha256} != {GGUF_SHA256}")
    if importlib.metadata.version("vector-quantize-pytorch") != VECTOR_QUANTIZE_VERSION:
        raise RuntimeError("vector-quantize-pytorch version mismatch")

    codes = np.fromfile(args.codes, dtype="<u4")
    if codes.size == 0 or np.any(codes >= CODEBOOK_SIZE):
        raise RuntimeError(f"codes must be non-empty and each below {CODEBOOK_SIZE}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    decoder, fc_post_a, loaded_count, defaulted = load_official_modules(args.gguf, preimport)

    code_tensor = torch.from_numpy(codes.astype(np.int64)).reshape(1, 1, -1)
    with torch.inference_mode():
        features = decoder.quantizer.get_output_from_indices(
            code_tensor.transpose(1, 2)
        )
        features = features.transpose(1, 2)
        features = fc_post_a(features.transpose(1, 2)).transpose(1, 2)
        decoded = decoder(features.transpose(1, 2), vq=False)[0]
    expected_shape = (1, 1, int(codes.size) * HOP_LENGTH)
    if tuple(features.shape) != (1, HIDDEN_DIM, int(codes.size)):
        raise RuntimeError(f"unexpected feature shape {tuple(features.shape)}")
    if tuple(decoded.shape) != expected_shape:
        raise RuntimeError(f"unexpected decoded shape {tuple(decoded.shape)}")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("official decoder emitted non-finite PCM")

    args.output.mkdir(parents=True, exist_ok=True)
    codes_path = args.output / "codes.u32le"
    features_path = args.output / "features.f32"
    pcm_path = args.output / "decoded_pcm.f32"
    np.asarray(codes, dtype="<u4").tofile(codes_path)
    np.asarray(features.cpu().numpy(), dtype="<f4").tofile(features_path)
    np.asarray(decoded.cpu().numpy(), dtype="<f4").tofile(pcm_path)
    manifest = {
        "format": "vokra-xcodec2-reference-v1",
        "oracle": "official xcodec2==0.1.5 CodecDecoderVocos FSQ + forward",
        "source_distribution": "xcodec2==0.1.5",
        "source_distribution_sha256": XCODEC2_SDIST_SHA256,
        "decoder_source_sha256": DECODER_SOURCE_SHA256,
        "transformer_source_sha256": TRANSFORMER_SOURCE_SHA256,
        "gguf_sha256": gguf_sha256,
        "torchtune": TORCHTUNE_VERSION,
        "torchtune_rope_sha256": TORCHTUNE_ROPE_SHA256,
        "vector_quantize_pytorch": VECTOR_QUANTIZE_VERSION,
        "torch": torch_version,
        "torchaudio": torchaudio_version,
        "official_state_tensors_loaded": loaded_count + 2,
        "official_defaulted_deterministic_buffers": defaulted,
        "code_count": int(codes.size),
        "feature_shape": list(features.shape),
        "decoded_shape": list(decoded.shape),
        "files": {
            path.name: sha256_file(path)
            for path in (codes_path, features_path, pcm_path)
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
