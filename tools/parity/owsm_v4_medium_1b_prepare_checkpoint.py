#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Collect fail-closed OWSM checkpoint payload evidence.

The fixed checkpoint is loaded only with ``weights_only=True`` on the VAST
worker. This tool deliberately does not convert or rewrite a checkpoint: the
repository has no OWSM GGUF writer contract yet, so target names, dimensions,
dtype, normalization, and transposition remain explicit blocked fields rather
than guessed mappings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

HF_REPOSITORY = "espnet/owsm_v4_medium_1B"
HF_REVISION = "e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
SOURCE_REVISION = "cccc29023d43a3f504e28df7d1324bb4eb6daedd"
CHECKPOINT_SHA256 = "b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09"
CHECKPOINT_TENSOR_COUNT = 1172
INSPECTION_MANIFEST_SHA256 = "82de20eea3cf3a247624c76cd8e108e562addda0c8582577515cf88abb3053d9"
HISTORICAL_INSPECTION_LOG_SHA256 = "4df29428ea8ce381311c5e407d937b6a517750f4edcbc88b8c606cdef82dc93b"
BPE_SHA256 = "7ddb01f03dab493c18ab69391e98744c090f897890d8b529b30cae52a8d9eef4"
STATS_SHA256 = "00c22dba27594df1d8f74a491b20c6e6e8c17e92159f81dfd634f98c098654"
TOKEN_LIST_SHA256 = "e19396ec012b0294a11fe85c35e36a1d903bc83e60ea602ddf6cc59b7c0e92f9"
FORMAT = "vokra-owsm-v4-medium-1b-payload-evidence-v1"
WRITER_SOURCE = "crates/vokra-convert/src/models/owsm_v4_medium_1b.rs"
WRITER_STATUS = "MISSING_OWSM_GGUF_WRITER_CONTRACT"
BLOCKED_ACTION = "UNSPECIFIED_PENDING_WRITER_REVIEW"


class EvidenceError(RuntimeError):
    """A fail-closed checkpoint evidence validation error."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unique_names(names: list[str]) -> None:
    if any(not isinstance(name, str) or not name for name in names):
        raise EvidenceError("empty/non-string source tensor name")
    if len(set(names)) != len(names):
        raise EvidenceError("duplicate source tensor name")


def no_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require_little_endian(byteorder: str) -> None:
    if sys.byteorder != "little" or byteorder not in ("=", "<"):
        raise EvidenceError(f"unexpected source byte order: {byteorder}")


def canonical_tensor_bytes(tensor: Any, torch: Any, numpy: Any) -> tuple[bytes, str]:
    """Return little-endian C-order F32 bytes, refusing unsafe coercions."""
    if str(tensor.dtype) != "torch.float32":
        raise EvidenceError(f"non-F32 tensor: {tensor.dtype}")
    if getattr(tensor, "device", None) is None or tensor.device.type != "cpu":
        raise EvidenceError("tensor is not resident on CPU")
    if getattr(tensor, "layout", None) != torch.strided:
        raise EvidenceError("tensor does not use the dense strided layout")
    require_little_endian("=")
    # ``Tensor.numpy()`` may expose the source storage order for a view.  Make
    # the logical indexing order explicit first: ``contiguous()`` preserves
    # the tensor's shape and values while materializing row-major (C-order)
    # storage, including for legitimate transposed/sliced CPU views.
    logical = tensor.detach().contiguous()
    array = logical.numpy()
    if not array.flags.c_contiguous:
        raise EvidenceError("canonical NumPy materialization is non-contiguous")
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise EvidenceError(f"unexpected NumPy dtype: {array.dtype}")
    require_little_endian(array.dtype.byteorder)
    # Always copy into an explicitly little-endian C-order array.  This keeps
    # the payload hash independent of both the source view's strides and any
    # native-endian NumPy aliasing.
    little = numpy.array(array, dtype=numpy.dtype("<f4"), order="C", copy=True)
    if little.dtype.byteorder not in ("<", "=") or not little.flags.c_contiguous:
        raise EvidenceError("canonical little-endian conversion drift")
    return little.tobytes(order="C"), "little"


def walk_tensors(value: Any, torch: Any, path: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key or "/" in key or "\\" in key:
                raise EvidenceError(f"unsafe checkpoint key at {path!r}")
            rows.extend(walk_tensors(item, torch, f"{path}.{key}" if path else key))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(walk_tensors(item, torch, f"{path}[{index}]"))
        return rows
    if value is None or isinstance(value, (bool, int, float, str)):
        return []
    raise EvidenceError(f"unsupported checkpoint object at {path}: {type(value).__name__}")


def validate_structural_manifest(
    path: Path, rows: list[dict[str, Any]], *, expected_digest: str = INSPECTION_MANIFEST_SHA256
) -> None:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError("structural manifest must be a regular non-symlink file")
    actual_digest = sha256_file(path)
    if actual_digest != expected_digest:
        raise EvidenceError("structural manifest SHA-256 does not match the fixed reviewed identity")
    try:
        packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_json_pairs)
        expected = packet["model"]["checkpoint"]["tensors"]
    except Exception as error:
        raise EvidenceError(f"structural manifest blocked: {error}") from error
    if not isinstance(expected, list):
        raise EvidenceError("structural manifest tensor list is missing")
    expected_by_name: dict[str, dict[str, Any]] = {}
    for row in expected:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise EvidenceError("structural manifest has malformed tensor row")
        name = row["name"]
        if name in expected_by_name:
            raise EvidenceError(f"duplicate structural tensor name: {name}")
        expected_by_name[name] = row
    actual_by_name = {row["source_name"]: row for row in rows}
    if set(expected_by_name) != set(actual_by_name):
        raise EvidenceError("checkpoint source names differ from structural manifest")
    for name, expected_row in expected_by_name.items():
        actual = actual_by_name[name]
        if (
            expected_row.get("shape") != actual["source_shape"]
            or expected_row.get("dtype") != actual["source_dtype"]
            or expected_row.get("numel") != actual["source_numel"]
            or expected_row.get("finite") is not True
        ):
            raise EvidenceError(f"checkpoint structural row differs: {name}")


def load_rows(
    checkpoint: Path,
    structural_manifest: Path | None = None,
    *,
    require_fixed_identity: bool = True,
) -> list[dict[str, Any]]:
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise EvidenceError("checkpoint must be a regular non-symlink file")
    if sys.byteorder != "little":
        raise EvidenceError("host platform is not little-endian")
    if require_fixed_identity and sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise EvidenceError("checkpoint SHA-256 does not match the fixed VAST identity")
    try:
        import numpy
        import torch
    except ImportError as error:
        raise EvidenceError(f"reference dependencies unavailable: {error}") from error
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise EvidenceError(f"weights_only checkpoint load blocked: {error}") from error
    pairs = sorted(walk_tensors(state, torch), key=lambda pair: pair[0])
    validate_unique_names([name for name, _ in pairs])
    rows: list[dict[str, Any]] = []
    for source_name, tensor in pairs:
        shape = [int(axis) for axis in tensor.shape]
        numel = int(tensor.numel())
        expected_numel = 1
        for axis in shape:
            if axis < 0:
                raise EvidenceError(f"negative tensor dimension: {source_name}")
            expected_numel *= axis
        if expected_numel != numel:
            raise EvidenceError(f"numel/shape mismatch: {source_name}")
        finite = bool(torch.isfinite(tensor).all().item()) if tensor.is_floating_point() else False
        if not finite:
            raise EvidenceError(f"non-finite tensor: {source_name}")
        payload, byte_order = canonical_tensor_bytes(tensor, torch, numpy)
        if len(payload) != numel * 4:
            raise EvidenceError(f"F32 payload byte length mismatch: {source_name}")
        rows.append(
            {
                "source_name": source_name,
                "source_shape": shape,
                "source_dtype": str(tensor.dtype),
                "source_numel": numel,
                "source_finite": finite,
                "source_contiguous": bool(tensor.is_contiguous()),
                "source_byte_order": byte_order,
                "canonical_payload_bytes": len(payload),
                "canonical_payload_sha256": sha256_bytes(payload),
                "target_name": None,
                "target_dims": None,
                "target_dtype": None,
                "normalization_action": BLOCKED_ACTION,
                "transposition_action": BLOCKED_ACTION,
                "mapping_status": WRITER_STATUS,
            }
        )
    if len(rows) != CHECKPOINT_TENSOR_COUNT:
        raise EvidenceError(f"expected {CHECKPOINT_TENSOR_COUNT} tensors, found {len(rows)}")
    if structural_manifest is not None:
        validate_structural_manifest(structural_manifest, rows)
    return rows


def manifest_without_digest(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "manifest_sha256"}


def build_manifest(
    checkpoint: Path, rows: list[dict[str, Any]], structural_manifest: Path | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "BLOCKED_WRITER_CONTRACT",
        "completed_evidence": False,
        "blocked_evidence": True,
        "writer_contract": {
            "status": WRITER_STATUS,
            "source_of_truth": WRITER_SOURCE,
            "target_mapping": "NOT_SPECIFIED_NO_OWSM_WRITER",
            "normalization": "NOT_SPECIFIED_NO_OWSM_WRITER",
            "transposition": "NOT_SPECIFIED_NO_OWSM_WRITER",
            "publication": "NO_UPLOAD",
        },
        "source": {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "inspection_manifest_sha256": INSPECTION_MANIFEST_SHA256,
            "historical_inspection_log_sha256": HISTORICAL_INSPECTION_LOG_SHA256,
            "bpe_sha256": BPE_SHA256,
            "stats_sha256": STATS_SHA256,
            "token_list_sha256": TOKEN_LIST_SHA256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_path": checkpoint.name,
            "structural_manifest_sha256": sha256_file(structural_manifest)
            if structural_manifest is not None
            else None,
        },
        "tensor_count": len(rows),
        "tensors": rows,
    }
    payload["manifest_sha256"] = sha256_bytes(canonical_json(manifest_without_digest(payload)))
    return payload


def write_atomic_no_replace(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"refusing to overwrite existing evidence path: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise EvidenceError(f"evidence parent must be an existing real directory: {parent}")
    data = canonical_json(payload)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise EvidenceError(f"evidence publication raced with an existing path: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def verify_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError("payload manifest must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_json_pairs)
    except Exception as error:
        raise EvidenceError(f"payload manifest JSON blocked: {error}") from error
    expected = payload.get("manifest_sha256")
    if not isinstance(expected, str) or expected != sha256_bytes(canonical_json(manifest_without_digest(payload))):
        raise EvidenceError("payload manifest digest mismatch")
    if payload.get("status") != "BLOCKED_WRITER_CONTRACT" or payload.get("completed_evidence") is not False:
        raise EvidenceError("payload manifest status is not fail-closed")
    if payload.get("blocked_evidence") is not True:
        raise EvidenceError("payload manifest blocked status is missing")
    writer = payload.get("writer_contract")
    if not isinstance(writer, dict) or writer != {
        "status": WRITER_STATUS,
        "source_of_truth": WRITER_SOURCE,
        "target_mapping": "NOT_SPECIFIED_NO_OWSM_WRITER",
        "normalization": "NOT_SPECIFIED_NO_OWSM_WRITER",
        "transposition": "NOT_SPECIFIED_NO_OWSM_WRITER",
        "publication": "NO_UPLOAD",
    }:
        raise EvidenceError("payload manifest writer contract is not exact")
    source = payload.get("source")
    if not isinstance(source, dict) or any(
        source.get(key) != value
        for key, value in {
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "inspection_manifest_sha256": INSPECTION_MANIFEST_SHA256,
            "historical_inspection_log_sha256": HISTORICAL_INSPECTION_LOG_SHA256,
            "bpe_sha256": BPE_SHA256,
            "stats_sha256": STATS_SHA256,
            "token_list_sha256": TOKEN_LIST_SHA256,
        }.items()
    ):
        raise EvidenceError("payload manifest source identity is not exact")
    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != CHECKPOINT_TENSOR_COUNT:
        raise EvidenceError("payload manifest tensor count mismatch")
    validate_unique_names([row.get("source_name") for row in tensors])
    for row in tensors:
        required = {
            "source_name",
            "source_shape",
            "source_dtype",
            "source_numel",
            "source_finite",
            "source_contiguous",
            "source_byte_order",
            "canonical_payload_bytes",
            "canonical_payload_sha256",
            "target_name",
            "target_dims",
            "target_dtype",
            "normalization_action",
            "transposition_action",
            "mapping_status",
        }
        if set(row) != required:
            raise EvidenceError("payload manifest row fields differ from the evidence schema")
        if row.get("source_dtype") != "torch.float32" or row.get("source_byte_order") != "little":
            raise EvidenceError("payload manifest source dtype/byte order mismatch")
        if row.get("source_finite") is not True or not isinstance(row.get("source_contiguous"), bool):
            raise EvidenceError("payload manifest source tensor safety status mismatch")
        if not isinstance(row.get("canonical_payload_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", row["canonical_payload_sha256"]
        ):
            raise EvidenceError("payload manifest payload digest is malformed")
        if row.get("mapping_status") != WRITER_STATUS or row.get("target_name") is not None:
            raise EvidenceError("payload manifest invented target mapping")
    return payload


def build_error_manifest(error: Exception) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": "BLOCKED_EVIDENCE_COLLECTION",
        "completed_evidence": False,
        "blocked_evidence": True,
        "error": str(error),
        "writer_contract": {
            "status": WRITER_STATUS,
            "source_of_truth": WRITER_SOURCE,
            "publication": "NO_UPLOAD",
        },
    }


def self_test() -> None:
    assert len(HF_REVISION) == len(SOURCE_REVISION) == 40
    assert CHECKPOINT_TENSOR_COUNT == 1172
    assert len(CHECKPOINT_SHA256) == 64
    with tempfile.TemporaryDirectory(prefix="owsm-prepare-self-test-") as temporary:
        root = Path(temporary)
        checkpoint = root / "synthetic.pth"
        output = root / "payload-manifest.json"
        try:
            import torch
            import numpy
        except ImportError:
            print("owsm_v4_medium_1b_prepare_checkpoint self-test: SKIP (torch unavailable)")
            return
        payload_bytes, byte_order = canonical_tensor_bytes(
            torch.tensor([1.0, 2.0], dtype=torch.float32), torch, numpy
        )
        assert byte_order == "little"
        assert payload_bytes == bytes.fromhex("0000803f00000040")
        assert sha256_bytes(payload_bytes) == hashlib.sha256(payload_bytes).hexdigest()
        try:
            require_little_endian(">")
        except EvidenceError:
            pass
        else:
            raise AssertionError("big-endian source was accepted")
        torch.save({"encoder": {"weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}}, checkpoint)
        try:
            load_rows(checkpoint, require_fixed_identity=False)
        except EvidenceError as error:
            assert "expected 1172" in str(error)
        else:
            raise AssertionError("synthetic count mismatch was accepted")
        rows = [
            {
                "source_name": f"synthetic.weight.{index}",
                "source_shape": [2],
                "source_dtype": "torch.float32",
                "source_numel": 2,
                "source_finite": True,
                "source_contiguous": True,
                "source_byte_order": "little",
                "canonical_payload_bytes": 8,
                "canonical_payload_sha256": "0" * 64,
                "mapping_status": WRITER_STATUS,
                "target_name": None,
                "target_dims": None,
                "target_dtype": None,
                "normalization_action": BLOCKED_ACTION,
                "transposition_action": BLOCKED_ACTION,
            }
            for index in range(CHECKPOINT_TENSOR_COUNT)
        ]
        structural = root / "structural.json"
        structural.write_text(
            json.dumps(
                {
                    "model": {
                        "checkpoint": {
                            "tensors": [
                                {
                                    "name": rows[0]["source_name"],
                                    "shape": rows[0]["source_shape"],
                                    "dtype": rows[0]["source_dtype"],
                                    "numel": rows[0]["source_numel"],
                                    "finite": True,
                                }
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        validate_structural_manifest(structural, rows[:1], expected_digest=sha256_file(structural))
        try:
            validate_structural_manifest(structural, rows[:1], expected_digest="0" * 64)
        except EvidenceError as error:
            assert "SHA-256" in str(error)
        else:
            raise AssertionError("structural manifest digest mismatch was accepted")
        structural.write_text(
            structural.read_text(encoding="utf-8").replace("synthetic.weight.0", "synthetic.weight.missing"),
            encoding="utf-8",
        )
        try:
            validate_structural_manifest(structural, rows[:1], expected_digest=INSPECTION_MANIFEST_SHA256)
        except EvidenceError:
            pass
        else:
            raise AssertionError("structural name drift was accepted")
        # ``source_contiguous`` records the source fact; canonicalization is
        # still safe when that fact is false, so the manifest verifier must
        # not turn this informational field back into a rejection gate.
        rows[0]["source_contiguous"] = False
        payload = build_manifest(checkpoint, rows)
        write_atomic_no_replace(output, payload)
        assert verify_manifest(output)["blocked_evidence"]
        try:
            write_atomic_no_replace(output, payload)
        except EvidenceError:
            pass
        else:
            raise AssertionError("no-clobber publication was accepted")
        tampered = json.loads(output.read_text(encoding="utf-8"))
        tampered["status"] = "PASS"
        output.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            verify_manifest(output)
        except EvidenceError:
            pass
        else:
            raise AssertionError("tampered manifest was accepted")
        # A transposed view is a real-world non-contiguous CPU tensor.  The
        # canonical bytes must follow its logical C-order values, not the
        # underlying storage order exposed by a Fortran-order view.
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        noncontiguous = base.t()
        assert not noncontiguous.is_contiguous()
        payload_bytes, byte_order = canonical_tensor_bytes(noncontiguous, torch, numpy)
        trusted_logical = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=torch.float32
        ).numpy()
        trusted_bytes = numpy.array(
            trusted_logical, dtype=numpy.dtype("<f4"), order="C", copy=True
        ).tobytes(order="C")
        assert byte_order == "little"
        assert payload_bytes == trusted_bytes
        assert payload_bytes == bytes.fromhex(
            "0000803f00008040000000400000a040000040400000c040"
        )
        assert sha256_bytes(payload_bytes) == sha256_bytes(trusted_bytes)
        storage_order_bytes = noncontiguous.detach().numpy().tobytes(order="F")
        assert storage_order_bytes != payload_bytes

        noncontiguous_checkpoint = root / "noncontiguous.pth"
        torch.save({"x": noncontiguous}, noncontiguous_checkpoint)
        try:
            load_rows(noncontiguous_checkpoint, require_fixed_identity=False)
        except EvidenceError as error:
            assert "expected 1172" in str(error)
        else:
            raise AssertionError("synthetic tensor count mismatch was accepted")
        nonf32 = root / "nonf32.pth"
        torch.save({"x": torch.ones((1,), dtype=torch.float16)}, nonf32)
        try:
            load_rows(nonf32, require_fixed_identity=False)
        except EvidenceError as error:
            assert "non-F32" in str(error)
        else:
            raise AssertionError("non-F32 tensor was accepted")
        validate_unique_names(["a"])
        try:
            validate_unique_names(["a", "a"])
        except EvidenceError:
            pass
        else:
            raise AssertionError("duplicate names were accepted")
    print("owsm_v4_medium_1b_prepare_checkpoint self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect fail-closed OWSM payload evidence")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--structural-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.structural_manifest, args.output)):
            parser.error("--self-test accepts no checkpoint, structural manifest, or output")
        self_test()
        return 0
    if args.checkpoint is None or args.structural_manifest is None or args.output is None:
        parser.error("normal run requires --checkpoint, --structural-manifest, and --output")
    try:
        rows = load_rows(args.checkpoint, args.structural_manifest)
        manifest = build_manifest(args.checkpoint, rows, args.structural_manifest)
        write_atomic_no_replace(args.output, manifest)
        verify_manifest(args.output)
    except Exception as error:
        print(f"PAYLOAD_EVIDENCE_BLOCKED: {error}", file=sys.stderr)
        if args.output.parent.is_dir() and not args.output.exists() and not args.output.is_symlink():
            try:
                write_atomic_no_replace(args.output, build_error_manifest(error))
            except Exception as publication_error:
                print(f"PAYLOAD_EVIDENCE_ERROR_MANIFEST_BLOCKED: {publication_error}", file=sys.stderr)
        return 2
    print(f"PAYLOAD_EVIDENCE_COLLECTED_BUT_WRITER_BLOCKED: {args.output.name}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
