#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Dump the authenticated Kyutai STT ``dep_q=0`` decoder component.

The real path imports the pinned Moshi implementation (which in turn carries
the delayed-streams-modeling decoder), rather than reimplementing its math.
It is intentionally a VAST-only operation: this file never downloads model
weights and refuses absent or unauthenticated source/checkpoint inputs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import re
from pathlib import Path
from typing import Any

HF_REPOSITORY = "kyutai/stt-2.6b-en"
HF_REVISION = "a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
MODEL_NAME = "model.safetensors"
MODEL_BYTES = 5_234_275_128
MODEL_SHA256 = "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2"
MODEL_TENSOR_MANIFEST_SHA256 = "e62488c9d16953010c758ec17f4c70e8ee30d348adfab3811eb5dfecb435d5df"
TORCH_VERSION = "2.13.0"
CONFIG_SHA256 = "b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9"
MIMI_NAME = "mimi-pytorch-e351c8d8@125.safetensors"
MIMI_BYTES = 384_644_900
MIMI_SHA256 = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
TOKENIZER_NAME = "tokenizer_en_audio_4000.model"
TOKENIZER_BYTES = 59_339
TOKENIZER_SHA256 = "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
DSM_REPOSITORY = "https://github.com/kyutai-labs/delayed-streams-modeling.git"
DSM_REVISION = "4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_REPOSITORY = "https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
N_Q = 32
TEXT_CARD = 4000
AUDIO_CARD = 2048
TEXT_TOKENS = [3, 17, 23, 29]
AUDIO_CODES = [[(frame * 37 + channel * 11) % AUDIO_CARD for channel in range(N_Q)] for frame in range(len(TEXT_TOKENS))]
OUTPUT_NAMES = ("input.json", "hidden.f32", "logits.f32", "manifest.json")
MOSHI_ROLES = (
    "moshi/moshi/models/lm.py",
    "moshi/moshi/models/lm_utils.py",
    "moshi/moshi/models/loaders.py",
    "moshi/moshi/utils/sampling.py",
    "moshi/moshi/modules/transformer.py",
)
DSM_ROLES = ("configs/config-stt-en-hf.toml", "scripts/stt_from_file_pytorch.py")
STREAMING_STATUS = "AUTHENTICATED_SOURCE_CONTRACT"
STREAMING_RUNTIME_STATUS = "BLOCKED_NOT_EXECUTED"
STREAMING_BLOCKERS = (
    "per-step feedback/state transition is not independently observed",
    "temperature-zero tie behavior is not independently observed",
    "context/window truncation and output ordering are not independently observed",
)
STREAMING_CONTRACT_SCHEMA = "vokra-kyutai-stt-streaming-source-contract-v1"
APPROVAL_SCHEMA = "vokra-kyutai-stt-decoder-approval-v1"
APPROVAL_SCOPE = "KYUTAI_STT_DECODER_PARITY"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def expected_tensor_manifest() -> list[dict[str, Any]]:
    """The exact 323 BF16 LM tensors, in upstream state-dict names."""
    rows: list[dict[str, Any]] = [{"name": "text_emb.weight", "dtype": "BF16", "shape": [4001, 2048]}]
    rows.extend(
        {"name": f"emb.{channel}.weight", "dtype": "BF16", "shape": [2049, 2048]}
        for channel in range(N_Q)
    )
    for layer in range(48):
        prefix = f"transformer.layers.{layer}"
        rows.extend([
            {"name": f"{prefix}.self_attn.in_proj_weight", "dtype": "BF16", "shape": [6144, 2048]},
            {"name": f"{prefix}.self_attn.out_proj.weight", "dtype": "BF16", "shape": [2048, 2048]},
            {"name": f"{prefix}.gating.linear_in.weight", "dtype": "BF16", "shape": [11264, 2048]},
            {"name": f"{prefix}.gating.linear_out.weight", "dtype": "BF16", "shape": [2048, 5632]},
            {"name": f"{prefix}.norm1.alpha", "dtype": "BF16", "shape": [2048]},
            {"name": f"{prefix}.norm2.alpha", "dtype": "BF16", "shape": [2048]},
        ])
    rows.extend(
        [
            {"name": "out_norm.alpha", "dtype": "BF16", "shape": [2048]},
            {"name": "text_linear.weight", "dtype": "BF16", "shape": [4000, 2048]},
        ]
    )
    assert len(rows) == 323
    return rows


def tensor_manifest_digest(rows: list[dict[str, Any]]) -> str:
    canonical = "".join(
        f"{row['name']}\0{row['dtype']}\0{','.join(str(x) for x in row['shape'])}\n"
        for row in rows
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def safetensors_header(path: Path) -> list[dict[str, Any]]:
    """Read and authenticate only the safetensors header, never payload data."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header length")
        header_bytes = int.from_bytes(prefix, "little")
        if header_bytes <= 0 or header_bytes > 64 * 1024 * 1024:
            raise ValueError("invalid safetensors header length")
        document = json.loads(stream.read(header_bytes).decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(document, dict):
        raise ValueError("safetensors header is not an object")
    if "__metadata__" in document:
        metadata = document.pop("__metadata__")
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("invalid safetensors metadata")
    expected = expected_tensor_manifest()
    expected_names = {row["name"] for row in expected}
    if set(document) != expected_names:
        raise ValueError("exact decoder tensor name set mismatch")
    payload_bytes = path.stat().st_size - 8 - header_bytes
    ranges: list[tuple[int, int, str]] = []
    for row in expected:
        descriptor = document.get(row["name"])
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid descriptor for {row['name']}")
        if descriptor["dtype"] != row["dtype"] or descriptor["shape"] != row["shape"]:
            raise ValueError(f"decoder tensor descriptor mismatch: {row['name']}")
        offsets = descriptor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in offsets):
            raise ValueError(f"decoder tensor offsets mismatch: {row['name']}")
        elements = 1
        for dimension in row["shape"]:
            elements *= dimension
        if offsets[1] - offsets[0] != elements * 2 or offsets[1] > payload_bytes or offsets[0] > offsets[1]:
            raise ValueError(f"decoder tensor bounds mismatch: {row['name']}")
        ranges.append((offsets[0], offsets[1], row["name"]))
    cursor = 0
    for start, end, _ in sorted(ranges):
        if start != cursor:
            raise ValueError("safetensors payload has a gap or overlap")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError("safetensors payload has a gap or trailing bytes")
    return expected


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _varint(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data) and shift < 64:
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
    raise ValueError("malformed SentencePiece varint")


def _skip_wire(data: bytes, cursor: int, wire: int) -> int:
    if wire == 0:
        return _varint(data, cursor)[1]
    if wire == 1:
        end = cursor + 8
    elif wire == 2:
        length, cursor = _varint(data, cursor)
        end = cursor + length
    elif wire == 5:
        end = cursor + 4
    else:
        raise ValueError(f"unsupported SentencePiece wire type {wire}")
    if end > len(data):
        raise ValueError("truncated SentencePiece field")
    return end


def _spm_piece(message: bytes) -> tuple[str, int]:
    piece: str | None = None
    # sentencepiece_model.proto is proto2 and declares `optional Type type =
    # 3 [default = NORMAL]`; materialize NORMAL when the field is absent.
    piece_type = 1
    cursor = 0
    while cursor < len(message):
        tag, cursor = _varint(message, cursor)
        field, wire = tag >> 3, tag & 7
        if field == 1 and wire == 2:
            length, cursor = _varint(message, cursor)
            end = cursor + length
            if end > len(message):
                raise ValueError("truncated SentencePiece piece")
            piece = message[cursor:end].decode("utf-8")
            cursor = end
        elif field == 3 and wire == 0:
            piece_type, cursor = _varint(message, cursor)
        else:
            cursor = _skip_wire(message, cursor, wire)
    if piece is None:
        raise ValueError("SentencePiece entry has no piece string")
    return piece, piece_type


def _spm_normalizer(message: bytes) -> tuple[bool, bool]:
    """Read the two SentencePiece NormalizerSpec bools.

    Proto2 defaults for both fields are true in the pinned
    ``sentencepiece_model.proto``. Unknown normalizer fields are skipped;
    this helper records only the flags that affect the decode contract.
    """
    add_dummy_prefix = True
    remove_extra_whitespaces = True
    cursor = 0
    while cursor < len(message):
        tag, cursor = _varint(message, cursor)
        field, wire = tag >> 3, tag & 7
        if field in (3, 4) and wire == 0:
            value, cursor = _varint(message, cursor)
            if field == 3:
                add_dummy_prefix = value != 0
            else:
                remove_extra_whitespaces = value != 0
        else:
            cursor = _skip_wire(message, cursor, wire)
    return add_dummy_prefix, remove_extra_whitespaces


def tokenizer_structure(path: Path) -> dict[str, Any]:
    """Inspect the authenticated SPM table without importing a tokenizer.

    The bytes are read only after ``authenticate_model`` has checked the
    fixed filename/size/digest.  This records source facts for the artifact,
    not handwritten tokenization behavior.
    """
    data = path.read_bytes()
    pieces: list[tuple[str, int]] = []
    add_dummy_prefix = True
    remove_extra_whitespaces = True
    denormalizer_present = False
    cursor = 0
    while cursor < len(data):
        tag, cursor = _varint(data, cursor)
        field, wire = tag >> 3, tag & 7
        if field == 1 and wire == 2:
            length, cursor = _varint(data, cursor)
            end = cursor + length
            if end > len(data):
                raise ValueError("truncated SentencePiece entry")
            pieces.append(_spm_piece(data[cursor:end]))
            cursor = end
        elif field == 3 and wire == 2:
            length, cursor = _varint(data, cursor)
            end = cursor + length
            if end > len(data):
                raise ValueError("truncated SentencePiece normalizer")
            add_dummy_prefix, remove_extra_whitespaces = _spm_normalizer(data[cursor:end])
            cursor = end
        elif field == 5 and wire == 2:
            length, cursor = _varint(data, cursor)
            end = cursor + length
            if end > len(data):
                raise ValueError("truncated SentencePiece denormalizer")
            denormalizer_present = True
            cursor = end
        else:
            cursor = _skip_wire(data, cursor, wire)
    if len(pieces) != TEXT_CARD:
        raise ValueError(f"tokenizer carries {len(pieces)} pieces, expected {TEXT_CARD}")
    expected = [("<unk>", 2), ("<s>", 3), ("</s>", 3), ("<pad>", 3)]
    if pieces[:4] != expected:
        raise ValueError(f"tokenizer specials mismatch: {pieces[:4]!r}")
    if not add_dummy_prefix or not remove_extra_whitespaces:
        raise ValueError("tokenizer normalizer flags are not the authenticated defaults")
    if denormalizer_present:
        raise ValueError("tokenizer denormalizer is present but unsupported")
    _validate_piece_types(pieces)
    canonical = bytearray(b"vokra.kyutai_stt.tokenizer.table.v1\0")
    for index, (piece, piece_type) in enumerate(pieces):
        encoded = piece.encode("utf-8")
        canonical.extend(index.to_bytes(4, "little"))
        canonical.extend(len(encoded).to_bytes(4, "little"))
        canonical.extend(encoded)
        canonical.extend(piece_type.to_bytes(4, "little"))
    table_sha256 = hashlib.sha256(canonical).hexdigest()
    boundary = next(((index, piece) for index, (piece, _) in enumerate(pieces) if "▁" in piece), None)
    byte_ids = [
        index
        for index, (piece, piece_type) in enumerate(pieces)
        if piece_type == 6 and len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")
    ]
    byte_example: dict[str, Any] | None = None
    for start in range(len(byte_ids)):
        for width in range(1, min(4, len(byte_ids) - start) + 1):
            ids = byte_ids[start : start + width]
            raw = bytes(int(pieces[index][0][3:5], 16) for index in ids)
            try:
                rendered = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            byte_example = {"ids": ids, "pieces": [pieces[index][0] for index in ids], "rendered": rendered}
            break
        if byte_example is not None:
            break
    return {
        "card": len(pieces),
        "specials": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
        "byte_fallback_count": len(byte_ids),
        # Internal table-integrity digest; whole-file SHA-256 remains the
        # external identity gate for both the raw sidecar and GGUF output.
        "table_sha256": table_sha256,
        "byte_example": byte_example,
        "boundary_example": None if boundary is None else {"id": boundary[0], "piece": boundary[1], "rendered": boundary[1].replace("▁", " ")},
        "suppressed_ids": [0, 3],
        "normalizer": {
            "add_dummy_prefix": add_dummy_prefix,
            "remove_extra_whitespaces": remove_extra_whitespaces,
        },
        "denormalizer_present": denormalizer_present,
    }


def _validate_piece_types(pieces: list[tuple[str, int]]) -> None:
    """Reject explicit zero/unknown SentencePiece enum values for Kyutai."""
    for index, (_, piece_type) in enumerate(pieces):
        if not 1 <= piece_type <= 6:
            raise ValueError(f"tokenizer piece {index} has invalid type {piece_type}; expected 1..6")


def approval_file(raw_path: Path | str) -> Path:
    raw = str(raw_path)
    if not raw or not raw.startswith("/") or "//" in raw or "/./" in raw or "/../" in raw or raw.endswith(("/.", "/..")):
        raise ValueError("approval path must be absolute and dot-free")
    path = Path(raw)
    current = Path("/")
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("approval path contains a symlink ancestor")
    if not path.is_file() or path.is_symlink():
        raise ValueError("approval evidence must be a regular file")
    return path


def validate_approval(path: Path | str, expected_head: str, expected_sha256: str, repo_root: Path | None = None) -> dict[str, Any]:
    if not HEX40.fullmatch(expected_head) or not HEX64.fullmatch(expected_sha256):
        raise ValueError("approval binding must use lowercase HEAD40 and SHA25664")
    path = approval_file(path)
    if repo_root is not None:
        root = repo_root.resolve()
        resolved = path.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError("approval evidence must be outside the checkout")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("approval evidence SHA-256 mismatch")
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    keys = {"schema", "status", "decision", "expected_head", "model_repository", "model_revision", "source_repository", "source_revision", "moshi_repository", "moshi_revision", "no_upload", "scope"}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError("approval schema is not exact")
    if data.get("no_upload") is not True:
        raise ValueError("approval no_upload must be a JSON boolean true")
    expected = {"schema": APPROVAL_SCHEMA, "status": "APPROVED", "decision": "APPROVED_FOR_NO_UPLOAD_PARITY", "expected_head": expected_head, "model_repository": HF_REPOSITORY, "model_revision": HF_REVISION, "source_repository": DSM_REPOSITORY, "source_revision": DSM_REVISION, "moshi_repository": MOSHI_REPOSITORY, "moshi_revision": MOSHI_REVISION, "no_upload": True, "scope": APPROVAL_SCOPE}
    if data != expected:
        raise ValueError("approval identity/scope mismatch")
    return data


def require_clean_head(expected_head: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True, stderr=subprocess.STDOUT)
    if dirty or head != expected_head:
        raise ValueError("checkout must be clean and match --expected-head")
    return root


def git_identity(root: Path, repository: str, revision: str, roles: tuple[str, ...]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT).strip()

    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("source checkout must be an absolute real directory")
    if git("remote", "get-url", "origin") != repository or git("rev-parse", "HEAD") != revision or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"source is not the exact clean checkout: {root}")
    rows = []
    for role in roles:
        path = root / role
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"authenticated source role is missing: {role}")
        rows.append({"path": role, "bytes": path.stat().st_size, "sha256": sha256(path), "git_blob_sha1": git("rev-parse", f"HEAD:{role}")})
    return {"repository": repository, "revision": revision, "roles": rows}


def _source_ast(root: Path, relative: str) -> ast.Module:
    """Parse one authenticated source role without importing or executing it."""
    path = root / relative
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(f"cannot parse authenticated source role {relative}: {error}") from error


def _source_symbol(tree: ast.AST, name: str, containing: tuple[str, ...] = ()) -> ast.AST:
    symbols = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if containing:
        symbols = [
            node
            for node in symbols
            if all(expression in {ast.unparse(child) for child in ast.walk(node)} for expression in containing)
        ]
    if len(symbols) != 1:
        raise ValueError(f"authenticated source symbol {name!r} is not unique")
    return symbols[0]


def _require_source_expressions(symbol: ast.AST, role: str, expressions: tuple[str, ...]) -> None:
    actual = {ast.unparse(node) for node in ast.walk(symbol)}
    missing = [expression for expression in expressions if expression not in actual]
    if missing:
        raise ValueError(f"authenticated source expression drift in {role}: {missing!r}")


def authenticate_streaming_source_contract(dsm_source: Path, moshi_source: Path) -> dict[str, Any]:
    """Authenticate streaming control flow from pinned upstream source bytes.

    This is deliberately an AST/expression gate, not a second implementation:
    it records only expressions present in the authenticated source.  It does
    not execute a model or assert a numerical result.  The temperature-zero
    tie rule remains blocked because ``torch.argmax`` is an external semantic.
    """
    dsm_identity = git_identity(dsm_source, DSM_REPOSITORY, DSM_REVISION, DSM_ROLES)
    moshi_identity = git_identity(moshi_source, MOSHI_REPOSITORY, MOSHI_REVISION, MOSHI_ROLES)
    dsm_tree = _source_ast(dsm_source, DSM_ROLES[1])
    lm_tree = _source_ast(moshi_source, MOSHI_ROLES[0])
    sampling_tree = _source_ast(moshi_source, MOSHI_ROLES[3])
    transformer_tree = _source_ast(moshi_source, MOSHI_ROLES[4])

    dsm_main = _source_symbol(dsm_tree, "main")
    _require_source_expressions(
        dsm_main,
        DSM_ROLES[1],
        (
            "mimi.streaming(1)",
            "lm_gen.streaming(1)",
            "audio_tokens = mimi.encode(audio_chunk)",
            "text_tokens = lm_gen.step(audio_tokens)",
            "text_tokens_accum.append(text_tokens)",
        ),
    )
    lm_init = _source_symbol(lm_tree, "_init_streaming_state")
    _require_source_expressions(
        lm_init,
        MOSHI_ROLES[0],
        (
            "cache = torch.full((batch_size, self.lm_model.num_codebooks, self.max_delay + 2), lm_model.ungenerated_token_id, device=lm_model.device, dtype=torch.long)",
            "state.exit_stack.enter_context(self.lm_model.streaming(batch_size))",
        ),
    )
    lm_step = _source_symbol(lm_tree, "_step")
    _require_source_expressions(
        lm_step,
        MOSHI_ROLES[0],
        (
            "state.cache.gather(dim=2, index=positions)",
            "state.graphed_main(input_, state.condition_sum, state.condition_cross)",
            "sample_token(text_logits.float(), self.use_sampling, self.temp_text, self.top_k_text)",
            "state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)",
            "scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])",
            "index = (state.offsets % CT)[:, None, None]",
            "index = (state.offsets[:, None, None] - self.max_delay + gen_delays_cuda[:, None]) % CT",
            "out = state.cache.gather(dim=2, index=index)",
        ),
    )
    sampling_fn = _source_symbol(sampling_tree, "sample_token")
    _require_source_expressions(
        sampling_fn,
        MOSHI_ROLES[3],
        (
            "use_sampling and temp > 0.0",
            "next_token = torch.argmax(logits, dim=-1, keepdim=True)",
        ),
    )
    ring_complete = _source_symbol(transformer_tree, "complete")
    _require_source_expressions(
        ring_complete,
        MOSHI_ROLES[4],
        (
            "indexes = indexes % self.capacity",
            "positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - self.capacity)",
            "invalid = indexes >= self.end_offset.view(-1, 1)",
        ),
    )
    attention_forward = _source_symbol(
        transformer_tree,
        "forward",
        ("attn_bias = attn_bias & (delta < self.context)", "F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)"),
    )
    _require_source_expressions(
        attention_forward,
        MOSHI_ROLES[4],
        (
            "attn_bias = attn_bias & (delta < self.context)",
            "F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)",
        ),
    )
    source_roles = {
        "dsm": list(DSM_ROLES),
        "moshi": list(MOSHI_ROLES),
    }
    contracts = [
        {
            "name": "per_step_feedback_and_state_transition",
            "status": "SOURCE_EXPRESSION_AUTHENTICATED",
            "roles": [DSM_ROLES[1], MOSHI_ROLES[0]],
            "expressions": [
                "mimi.streaming(1) -> lm_gen.streaming(1)",
                "mimi.encode(audio_chunk) -> lm_gen.step(audio_tokens)",
                "state.cache.gather(...) -> state.graphed_main(...) -> generated text token -> state.cache scatter",
            ],
        },
        {
            "name": "temperature_zero_selection",
            "status": "SOURCE_EXPRESSION_AUTHENTICATED",
            "roles": [MOSHI_ROLES[3], MOSHI_ROLES[0]],
            "expressions": ["if use_sampling and temp > 0.0", "torch.argmax(logits, dim=-1, keepdim=True)"],
            "tie_resolution": "BLOCKED_EXTERNAL_TORCH_SEMANTICS",
        },
        {
            "name": "context_window_and_output_order",
            "status": "SOURCE_EXPRESSION_AUTHENTICATED",
            "roles": [MOSHI_ROLES[4], MOSHI_ROLES[0], DSM_ROLES[1]],
            "expressions": [
                "RingKVCache indexes modulo self.capacity",
                "causal attention delta < self.context",
                "LMGen output gathered from state.cache using generated delays",
                "DSM appends one text_tokens result per input chunk in loop order",
            ],
        },
    ]
    return {
        "schema": STREAMING_CONTRACT_SCHEMA,
        "status": STREAMING_STATUS,
        "runtime_status": STREAMING_RUNTIME_STATUS,
        "source_identity": {"dsm": dsm_identity, "moshi": moshi_identity},
        "source_roles": source_roles,
        "contracts": contracts,
        "blockers": list(STREAMING_BLOCKERS),
        "claim_boundary": "source expressions only; no model execution, numerical parity, PCM transcription, or framework tie-rule claim",
    }


def authenticate_model(model: Path, config: Path) -> dict[str, Any]:
    root = model.parent
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("model snapshot must be an absolute real directory")
    expected_files = {
        MODEL_NAME: (MODEL_BYTES, MODEL_SHA256),
        "config.json": (1_257, CONFIG_SHA256),
        MIMI_NAME: (MIMI_BYTES, MIMI_SHA256),
        TOKENIZER_NAME: (TOKENIZER_BYTES, TOKENIZER_SHA256),
    }
    actual_names = sorted(path.name for path in root.iterdir())
    if actual_names != sorted(expected_files):
        raise ValueError(f"model snapshot must contain exactly {sorted(expected_files)!r}")
    for name, (size, expected_sha) in expected_files.items():
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size or sha256(path) != expected_sha:
            raise ValueError(f"{name} identity mismatch")
    tensor_manifest = safetensors_header(model)
    document = json.loads(config.read_text(encoding="utf-8"), object_pairs_hook=unique)
    expected = {"card": 2048, "n_q": 32, "dep_q": 0, "delays": [0] * 33, "dim": 2048, "text_card": 4000, "existing_text_padding_id": 3, "num_heads": 32, "num_layers": 48, "hidden_scale": 4.125, "causal": True, "layer_scale": None, "context": 375, "max_period": 100000.0, "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope", "depformer_dim": 1024, "depformer_num_heads": 16, "depformer_num_layers": 6, "depformer_dim_feedforward": None, "depformer_multi_linear": True, "depformer_pos_emb": "none", "depformer_weights_per_step": True, "conditioners": {}, "cross_attention": False, "model_id": {"sig": "dabcc802", "epoch": 50}, "lm_gen_config": {"temp": 0.0, "temp_text": 0.0, "top_k": 250, "top_k_text": 50}, "stt_config": {"audio_delay_seconds": 2.5, "audio_silence_prefix_seconds": 1.0}, "model_type": "stt", "mimi_name": "mimi-pytorch-e351c8d8@125.safetensors", "tokenizer_name": "tokenizer_en_audio_4000.model"}
    if document != expected:
        raise ValueError("Kyutai config axes mismatch")
    record = {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "path": MODEL_NAME,
        "bytes": MODEL_BYTES,
        "sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "mimi": {"path": MIMI_NAME, "bytes": MIMI_BYTES, "sha256": MIMI_SHA256},
        "tokenizer": {"path": TOKENIZER_NAME, "bytes": TOKENIZER_BYTES, "sha256": TOKENIZER_SHA256},
        "tensor_manifest_sha256": tensor_manifest_digest(tensor_manifest),
        "tensor_manifest": tensor_manifest,
    }
    record["tokenizer_structure"] = tokenizer_structure(root / TOKENIZER_NAME)
    return record


def write_output(out: Path, files: dict[str, bytes], manifest: dict[str, Any]) -> None:
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("reference output must be absent and its parent must exist")
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        os.chmod(temporary, 0o700)
        for name, body in files.items():
            (temporary / name).write_bytes(body)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, out)
        parent_fd = os.open(out.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def self_test() -> None:
    assert len(TEXT_TOKENS) == 4 and len(AUDIO_CODES) == 4 and all(len(row) == N_Q for row in AUDIO_CODES)
    assert STREAMING_STATUS == "AUTHENTICATED_SOURCE_CONTRACT"
    assert STREAMING_RUNTIME_STATUS == "BLOCKED_NOT_EXECUTED"
    assert all("not independently observed" in blocker for blocker in STREAMING_BLOCKERS)
    assert TOKENIZER_NAME == "tokenizer_en_audio_4000.model"
    assert TOKENIZER_BYTES == 59_339 and TOKENIZER_SHA256 == "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
    # Synthetic wire-level checks exercise only malformed-input handling; no
    # token/text value here is presented as upstream parity evidence.
    try:
        _varint(b"\x80", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated SentencePiece varint accepted")
    try:
        _spm_piece(b"\x18\x01")
    except ValueError:
        pass
    else:
        raise AssertionError("SentencePiece entry without piece accepted")
    assert _spm_piece(b"\x0a\x01x") == ("x", 1)
    assert _spm_piece(b"\x0a\x01x\x18\x00") == ("x", 0)
    assert _spm_piece(b"\x0a\x01x\x18\x05") == ("x", 5)
    assert _spm_piece(b"\x0a\x01x\x18\x06") == ("x", 6)
    try:
        _validate_piece_types([("x", 0)])
    except ValueError:
        pass
    else:
        raise AssertionError("explicit zero SentencePiece type accepted")
    assert _spm_normalizer(b"\x18\x00\x20\x01") == (False, True)
    assert _spm_normalizer(b"") == (True, True)
    try:
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=unique)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")
    assert MODEL_BYTES > 5_000_000_000 and MODEL_SHA256 == "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2"
    tensor_manifest = expected_tensor_manifest()
    assert len(tensor_manifest) == 323
    assert tensor_manifest_digest(tensor_manifest) == MODEL_TENSOR_MANIFEST_SHA256
    assert tensor_manifest[0] == {"name": "text_emb.weight", "dtype": "BF16", "shape": [4001, 2048]}
    assert tensor_manifest[-1] == {"name": "text_linear.weight", "dtype": "BF16", "shape": [4000, 2048]}
    for bad in ("", "/tmp/./approval.json", "/tmp/../approval.json", "relative.json"):
        try:
            approval_file(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe approval path accepted")
    with tempfile.TemporaryDirectory(prefix=".kyutai-approval-", dir=Path.cwd()) as directory:
        head = "a" * 40
        payload = {"schema": APPROVAL_SCHEMA, "status": "APPROVED", "decision": "APPROVED_FOR_NO_UPLOAD_PARITY", "expected_head": head, "model_repository": HF_REPOSITORY, "model_revision": HF_REVISION, "source_repository": DSM_REPOSITORY, "source_revision": DSM_REVISION, "moshi_repository": MOSHI_REPOSITORY, "moshi_revision": MOSHI_REVISION, "no_upload": True, "scope": APPROVAL_SCOPE}
        approval = Path(directory) / "approval.json"
        approval.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        digest = sha256(approval)
        validate_approval(approval, head, digest)
        for invalid in (1, 0, "true"):
            approval.write_text(json.dumps(dict(payload, no_upload=invalid), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            try: validate_approval(approval, head, sha256(approval))
            except ValueError: pass
            else: raise AssertionError("non-boolean no_upload accepted")
        approval.write_bytes(b"{\xff")
        try: validate_approval(approval, head, sha256(approval))
        except (UnicodeDecodeError, ValueError): pass
        else: raise AssertionError("malformed UTF-8 approval accepted")
        approval.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        try: validate_approval(approval, head, digest, Path.cwd())
        except ValueError: pass
        else: raise AssertionError("checkout-contained approval accepted")
        try: validate_approval(approval, head, "0" * 64)
        except ValueError: pass
        else: raise AssertionError("wrong approval SHA accepted")
        for key, value in (("expected_head", "b" * 40), ("scope", "WRONG"), ("model_revision", "0" * 40)):
            bad = dict(payload, **{key: value})
            approval.write_text(json.dumps(bad, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            try: validate_approval(approval, head, sha256(approval))
            except ValueError: pass
            else: raise AssertionError("invalid approval identity accepted")
        approval.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        try: validate_approval(approval, head, sha256(approval))
        except ValueError: pass
        else: raise AssertionError("duplicate approval key accepted")
        approval.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        link = Path(directory) / "approval-link.json"
        link.symlink_to(approval)
        try: validate_approval(link, head, digest)
        except ValueError: pass
        else: raise AssertionError("symlink approval accepted")
    print("kyutai STT decoder reference self-test PASS")


def real(args: argparse.Namespace) -> None:
    checkout = require_clean_head(args.expected_head)
    approval = validate_approval(args.approval_evidence, args.expected_head, args.approval_sha256, checkout)
    if not args.model.is_absolute() or not args.model.is_dir() or args.model.is_symlink():
        raise ValueError("model snapshot must be an absolute directory")
    model_record = authenticate_model(args.model / MODEL_NAME, args.model / "config.json")
    dsm_record = git_identity(args.dsm_source, DSM_REPOSITORY, DSM_REVISION, DSM_ROLES)
    moshi_record = git_identity(args.moshi_source, MOSHI_REPOSITORY, MOSHI_REVISION, MOSHI_ROLES)
    streaming_contract = authenticate_streaming_source_contract(args.dsm_source, args.moshi_source)
    sys.path.insert(0, str(args.moshi_source))
    sys.path.insert(0, str(args.moshi_source / "moshi"))
    sys.path.insert(0, str(args.dsm_source))
    try:
        import torch  # type: ignore
        from moshi.models import loaders  # type: ignore
    except ImportError as error:
        raise SystemExit(f"official Moshi implementation is required: {error}") from error
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError("locked single-thread execution could not be established")
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("deterministic torch algorithms could not be enabled")
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise ValueError(f"unexpected locked torch version: {torch.__version__}")
    # CheckpointInfo is the upstream config-population path. Every referenced
    # file is a previously authenticated local override, so this cannot reach
    # the Hub or silently instantiate Moshi's unrelated 7B defaults.
    info = loaders.CheckpointInfo.from_hf_repo(
        HF_REPOSITORY,
        moshi_weights=args.model / MODEL_NAME,
        mimi_weights=args.model / MIMI_NAME,
        tokenizer=args.model / TOKENIZER_NAME,
        config_path=args.model / "config.json",
        revision=HF_REVISION,
    )
    lm = info.get_moshi(device="cpu", dtype=torch.float32)
    if (lm.dep_q, lm.n_q, lm.text_card, lm.dim) != (0, N_Q, TEXT_CARD, 2048):
        raise ValueError("official decoder axes are not the authenticated dep_q=0 contract")
    # Moshi expects [batch, channels, time]; construct it explicitly.
    sequence = torch.zeros((1, 1 + N_Q, len(TEXT_TOKENS)), dtype=torch.long)
    sequence[0, 0, :] = torch.tensor(TEXT_TOKENS)
    sequence[0, 1:, :] = torch.tensor(AUDIO_CODES).transpose(0, 1)
    with torch.no_grad():
        hidden, logits = lm.forward_text(sequence)
    hidden = hidden[0].float().cpu().contiguous()
    logits = (logits[0, 0] if logits.ndim == 4 and logits.shape[1] == 1 else logits[0]).float().cpu().contiguous()
    if tuple(hidden.shape) != (len(TEXT_TOKENS), 2048) or tuple(logits.shape) != (len(TEXT_TOKENS), TEXT_CARD):
        raise ValueError("official decoder output shape mismatch")
    if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("official decoder produced non-finite output")
    if int(torch.count_nonzero(hidden)) == 0 or int(torch.count_nonzero(logits)) == 0:
        raise ValueError("official decoder output is vacuous")
    files = {
        "input.json": (json.dumps({"text_tokens": TEXT_TOKENS, "mimi_codes": AUDIO_CODES}, sort_keys=True, indent=2) + "\n").encode(),
        "hidden.f32": hidden.numpy().tobytes(),
        "logits.f32": logits.numpy().tobytes(),
    }
    artifacts = {
        name: {
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "dtype": "json" if name == "input.json" else "f32-le",
            "role": {
                "input.json": "input",
                "hidden.f32": "diagnostics-only",
                "logits.f32": "parity",
            }[name],
        }
        for name, body in files.items()
    }
    manifest = {"format": "vokra-kyutai-stt-decoder-reference-v2", "status": "REFERENCE_READY", "component": "decoder", "scope": "dep_q=0 text decoder plus source-authenticated tokenizer structure; no Mimi neural encoding, streaming sampling loop, or PCM transcription claim", "expected_head": args.expected_head, "approval_sha256": args.approval_sha256, "approval_decision": approval["decision"], "approval_scope": approval["scope"], "model": model_record, "sources": {"dsm": dsm_record, "moshi": moshi_record}, "streaming": {**streaming_contract, "runtime_transition": "FAIL_CLOSED"}, "config": {"n_q": N_Q, "dep_q": 0, "d_model": 2048, "text_card": TEXT_CARD, "audio_card": AUDIO_CARD, "tensor_count": 323, "tokenizer_schema": "sentencepiece-decode-v1"}, "packet": {"text_tokens": TEXT_TOKENS, "mimi_codes": AUDIO_CODES}, "execution": {"implementation": "official Moshi LMModel.forward_text", "dtype": "F32", "device": "cpu", "python_version": f"{sys.version_info.major}.{sys.version_info.minor}", "torch_version": torch.__version__.split("+")[0], "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "publication": "NO_UPLOAD"}, "artifacts": artifacts}
    write_output(args.out, files, manifest)
    print(f"reference written: {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("self-test")
    approval_parser = sub.add_parser("validate-approval")
    approval_parser.add_argument("--expected-head", required=True)
    approval_parser.add_argument("--approval-evidence", required=True)
    approval_parser.add_argument("--approval-sha256", required=True)
    source_parser = sub.add_parser("validate-source-contract")
    source_parser.add_argument("--dsm-source", type=Path, required=True)
    source_parser.add_argument("--moshi-source", type=Path, required=True)
    real_parser = sub.add_parser("real")
    real_parser.add_argument("--model", type=Path, required=True)
    real_parser.add_argument("--dsm-source", type=Path, required=True)
    real_parser.add_argument("--moshi-source", type=Path, required=True)
    real_parser.add_argument("--out", type=Path, required=True)
    real_parser.add_argument("--expected-head", required=True)
    real_parser.add_argument("--approval-evidence", required=True)
    real_parser.add_argument("--approval-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "validate-approval":
        checkout = require_clean_head(args.expected_head)
        validate_approval(args.approval_evidence, args.expected_head, args.approval_sha256, checkout)
        print("Kyutai STT decoder approval: PASS")
    elif args.mode == "validate-source-contract":
        print(json.dumps(authenticate_streaming_source_contract(args.dsm_source, args.moshi_source), sort_keys=True))
    else:
        real(args)


if __name__ == "__main__":
    main()
