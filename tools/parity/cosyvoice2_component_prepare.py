#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Prepare an authenticated CosyVoice2 torch ZIP into deterministic safetensors.

This is a VAST-only bridge for the two pinned CosyVoice2 components. It reads
only the restricted ``data.pkl`` graph and the explicitly authenticated F32
storage members; it never imports torch, executes a model, or performs tensor
math. The output is a dense, sorted safetensors file written with a
no-replace atomic publication.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosyvoice2_component_inspect as inspection  # noqa: E402


FORMAT = "vokra-cosyvoice2-component-prepared-safetensors-v1"
PREPARED_STATUS = "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED"
MAX_COPY_BYTES = 8_000_000_000
CHUNK_BYTES = 1 << 20


class PreparationError(ValueError):
    """Input checkpoint is not an authenticated dense F32 component."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_storage_key(key: Any) -> str:
    if not isinstance(key, str) or not key or "/" in key or "\\" in key or key in {".", ".."}:
        raise PreparationError(f"unsafe storage key: {key!r}")
    return key


def _numel(shape: list[Any]) -> int:
    total = 1
    for dimension in shape:
        if type(dimension) is not int or dimension <= 0:
            raise PreparationError(f"tensor shape must contain positive integers: {shape!r}")
        total = total * dimension
        if total > 1 << 40:
            raise PreparationError("tensor element count exceeds bounded preparation limit")
    return total


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    return stride


def _expected_aliases(component: str) -> set[frozenset[str]]:
    if component == "llm":
        return {
            frozenset(
                {
                    "llm.model.lm_head.weight",
                    "llm.model.model.embed_tokens.weight",
                }
            )
        }
    return set()


def _validate_component_manifest(
    component: str,
    state: dict[str, Any],
    archive: dict[str, Any],
) -> None:
    expected = inspection.COMPONENTS[component]
    if state.get("format") != "vokra-pytorch-state-dict-manifest-v1":
        raise PreparationError("restricted manifest format mismatch")
    if state.get("data_pickle_sha256") != expected["data_pickle_sha256"]:
        raise PreparationError("data.pkl digest does not match authenticated component")
    if state.get("tensor_count") != expected["tensor_count"]:
        raise PreparationError("authenticated tensor count mismatch")
    tensors = state.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise PreparationError("restricted manifest has no tensors")
    if len(tensors) != expected.get("tensor_count", len(tensors)):
        raise PreparationError("authenticated tensor count mismatch")
    if state.get("manifest_sha256") != expected["manifest_sha256"]:
        raise PreparationError("name/shape manifest digest does not match authenticated component")
    if state.get("float_tensor_count") != expected["tensor_count"]:
        raise PreparationError("authenticated float tensor count mismatch")
    if state.get("float_manifest_sha256") != expected["manifest_sha256"]:
        raise PreparationError("float manifest digest does not match authenticated component")
    if state.get("storage_manifest_sha256") != expected["storage_manifest_sha256"]:
        raise PreparationError("storage manifest digest does not match authenticated component")
    if archive.get("member_count") != expected["member_count"]:
        raise PreparationError("ZIP member count does not match authenticated component")
    if archive.get("storage_member_count") != expected["storage_member_count"]:
        raise PreparationError("storage member count does not match authenticated component")

    storage_names: set[str] = set()
    storage_groups: dict[str, list[str]] = {}
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not name or name == "__metadata__" or "/" in name or "\\" in name or "\x00" in name:
            raise PreparationError(f"unsafe tensor name: {name!r}")
        if not isinstance(tensor, dict):
            raise PreparationError(f"invalid tensor record: {name!r}")
        if tensor.get("dtype") != "F32" or tensor.get("location") != "cpu":
            raise PreparationError(f"{name}: only CPU F32 tensors are supported")
        shape = tensor.get("shape")
        stride = tensor.get("stride")
        if not isinstance(shape, list) or not isinstance(stride, list) or stride != _contiguous_stride(shape):
            raise PreparationError(f"{name}: non-contiguous tensor view rejected")
        elements = _numel(shape)
        if tensor.get("storage_offset") != 0:
            raise PreparationError(f"{name}: non-zero storage offset is a view")
        if tensor.get("storage_numel") != elements:
            raise PreparationError(f"{name}: storage_numel does not equal dense tensor size")
        key = _safe_storage_key(tensor.get("storage_key"))
        storage_names.add(key)
        storage_groups.setdefault(key, []).append(name)
        max_address = sum(step * (dimension - 1) for step, dimension in zip(stride, shape))
        if max_address >= tensor["storage_numel"]:
            raise PreparationError(f"{name}: strided view exceeds storage bounds")

    if set(archive.get("storage_members", [])) != storage_names:
        raise PreparationError("ZIP storage members do not exactly match the manifest")
    allowed_aliases = _expected_aliases(component)
    for key, names in storage_groups.items():
        if len(names) > 1 and frozenset(names) not in allowed_aliases:
            raise PreparationError(f"storage {key} has an unauthenticated alias set")
        if len(names) > 1:
            first = tensors[names[0]]
            for name in names[1:]:
                if any(tensors[name].get(field) != first.get(field) for field in ("dtype", "shape", "stride", "storage_offset", "storage_numel", "location")):
                    raise PreparationError(f"storage {key} alias metadata differs")


def _storage_member_name(data_pickle_member: str, key: str) -> str:
    parent = str(PurePosixPath(data_pickle_member).parent)
    prefix = "" if parent == "." else parent + "/"
    return prefix + "data/" + key


def _flatten_metadata(value: Any, prefix: str) -> dict[str, str]:
    """Flatten authenticated provenance into deterministic scalar metadata."""
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise PreparationError("provenance keys must be non-empty strings")
            flattened.update(_flatten_metadata(value[key], f"{prefix}.{key}"))
        return flattened
    if isinstance(value, (str, int, float, bool)):
        return {prefix: str(value)}
    raise PreparationError(f"unsupported provenance value at {prefix}")


def _write_all(stream: Any, data: bytes) -> int:
    """Write all bytes, rejecting a short or non-progressing write."""
    written = 0
    while written < len(data):
        count = stream.write(data[written:])
        if type(count) is not int or count <= 0 or count > len(data) - written:
            raise PreparationError("short write while preparing output")
        written += count
    return written


def _fsync_directory_best_effort(parent: Path) -> None:
    try:
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _require_existing_directory(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise PreparationError(f"{label} must be an absolute directory")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or not current.is_dir():
            raise PreparationError(f"{label} must contain only existing real directories")


def _write_no_replace(output: Path, write_body: Any) -> tuple[int, str]:
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise PreparationError("prepared output must be an absent absolute path")
    parent = output.parent
    _require_existing_directory(parent, "prepared output parent")
    temporary: Path | None = None
    file = None
    for attempt in range(100):
        candidate = parent / f".{output.name}.vokra-cosyvoice2-prepared-{os.getpid()}-{attempt}"
        try:
            file = candidate.open("xb")
            temporary = candidate
            break
        except FileExistsError:
            continue
    if temporary is None or file is None:
        raise PreparationError("could not allocate temporary prepared output")
    digest = hashlib.sha256()
    count = 0
    try:
        count = write_body(file, digest)
        file.flush()
        os.fsync(file.fileno())
        file.close()
        os.link(temporary, output)
        _fsync_directory_best_effort(parent)
        try:
            temporary.unlink()
        except OSError:
            pass
        return count, digest.hexdigest()
    except Exception:
        try:
            file.close()
        except Exception:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _prepare_from_zip(
    checkpoint: Path,
    component: str,
    output: Path,
    report_path: Path | None,
    root: Path,
    expected: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw, archive_summary = inspection.read_data_pickle(checkpoint)
    parser = root / "tools/audit/torch_pickle_manifest.py"
    inspection.require_regular(parser.resolve(), "restricted parser")
    state = inspection.run_restricted_parser(
        raw,
        root,
        f"hf://{inspection.MODEL_REPOSITORY}@{inspection.MODEL_REVISION}:{inspection.COMPONENTS[component]['path']}/data.pkl",
    )
    production_expected = expected is None
    expected = expected or inspection.COMPONENTS[component]
    # Synthetic self-tests use a local spec; production always uses the
    # immutable component record above.
    if production_expected:
        _validate_component_manifest(component, state, archive_summary)
    else:
        _validate_test_manifest(state, archive_summary, expected)
    tensors = state["tensors"]
    data_pickle_member = archive_summary["data_pickle_member"]
    with zipfile.ZipFile(checkpoint) as archive:
        member_info = {info.filename: info for info in archive.infolist()}
        entries: dict[str, dict[str, Any]] = {}
        offset = 0
        for name, tensor in sorted(tensors.items()):
            if name == "__metadata__":
                raise PreparationError("reserved safetensors metadata name is not a tensor")
            size = tensor["storage_numel"] * 4
            entries[name] = {"dtype": "F32", "shape": tensor["shape"], "data_offsets": [offset, offset + size]}
            offset += size
        if offset > MAX_COPY_BYTES:
            raise PreparationError("prepared output exceeds bounded size")
        input_bytes = checkpoint.stat().st_size
        input_sha256 = inspection.sha256_file(checkpoint)
        metadata = {
            "vokra.preparation.format": FORMAT,
            "vokra.preparation.component": component,
            "vokra.preparation.raw_checkpoint_sha256": input_sha256,
            "vokra.preparation.input_bytes": str(input_bytes),
            "vokra.preparation.input_sha256": input_sha256,
            "vokra.preparation.data_pickle_sha256": state["data_pickle_sha256"],
            "vokra.preparation.tensor_manifest_sha256": state["manifest_sha256"],
            "vokra.preparation.storage_manifest_sha256": state["storage_manifest_sha256"],
            "vokra.preparation.status": PREPARED_STATUS,
        }
        if provenance:
            metadata.update(_flatten_metadata(provenance, "vokra.preparation.provenance"))
        header = {"__metadata__": metadata, **entries}
        header_bytes = json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        if len(header_bytes) > 1 << 30:
            raise PreparationError("safetensors header is too large")

        def write_body(stream: Any, digest: Any) -> int:
            written = 0
            prefix = header_bytes
            packet = len(prefix).to_bytes(8, "little") + prefix
            _write_all(stream, packet)
            digest.update(packet)
            written += len(packet)
            for name, tensor in sorted(tensors.items()):
                key = _safe_storage_key(tensor["storage_key"])
                member = _storage_member_name(data_pickle_member, key)
                info = member_info.get(member)
                if info is None or info.is_dir() or info.file_size != tensor["storage_numel"] * 4:
                    raise PreparationError(f"{name}: storage member size/path mismatch")
                with archive.open(info, "r") as source:
                    remaining = info.file_size
                    while remaining:
                        block = source.read(min(CHUNK_BYTES, remaining))
                        if not block:
                            raise PreparationError(f"{name}: storage member truncated")
                        _write_all(stream, block)
                        digest.update(block)
                        written += len(block)
                        remaining -= len(block)
                if remaining != 0:
                    raise PreparationError(f"{name}: storage member copy incomplete")
            if written > MAX_COPY_BYTES:
                raise PreparationError("prepared output exceeds bounded size")
            return written

        output_bytes, output_sha256 = _write_no_replace(output, write_body)
    report = {
        "format": FORMAT,
        "status": "PREPARED_SAFETENSORS_READY",
        "component": component,
        "output": {"path": str(output), "bytes": output_bytes, "sha256": output_sha256},
        "input": {"path": str(checkpoint), "bytes": input_bytes, "sha256": input_sha256},
        "checkpoint": {"data_pickle_sha256": state["data_pickle_sha256"], "tensor_count": state["tensor_count"], "manifest_sha256": state["manifest_sha256"], "storage_manifest_sha256": state["storage_manifest_sha256"]},
        "payload_reads": "DATA_PKL_AND_AUTHENTICATED_F32_STORAGE_ONLY",
        "execution": {"model_execution": "NOT_RUN", "torch_import": "NOT_RUN", "publication": "NO_UPLOAD"},
    }
    if provenance:
        report.update(provenance)
    if report_path is not None:
        created = output.stat()
        try:
            _write_report(report_path, report)
        except Exception:
            try:
                current = output.stat()
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    output.unlink()
            except OSError:
                pass
            raise
    return report


def _validate_test_manifest(state: dict[str, Any], archive: dict[str, Any], expected: dict[str, Any]) -> None:
    tensors = state.get("tensors")
    if not isinstance(tensors, dict) or len(tensors) != expected["tensor_count"]:
        raise PreparationError("synthetic tensor count mismatch")
    if state["manifest_sha256"] != expected["manifest_sha256"] or state["storage_manifest_sha256"] != expected["storage_manifest_sha256"]:
        raise PreparationError("synthetic manifest mismatch")
    if archive["storage_member_count"] != expected["storage_member_count"]:
        raise PreparationError("synthetic storage count mismatch")
    storage_groups: dict[str, list[str]] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or name == "__metadata__" or "/" in name or "\\" in name:
            raise PreparationError(f"synthetic unsafe tensor name: {name!r}")
        if tensor["dtype"] != "F32" or tensor["storage_offset"] != 0 or tensor["stride"] != _contiguous_stride(tensor["shape"]):
            raise PreparationError(f"synthetic non-contiguous tensor: {name}")
        if tensor["storage_numel"] != _numel(tensor["shape"]):
            raise PreparationError(f"synthetic storage size mismatch: {name}")
        storage_groups.setdefault(_safe_storage_key(tensor["storage_key"]), []).append(name)
    if set(archive["storage_members"]) != {tensor["storage_key"] for tensor in tensors.values()}:
        raise PreparationError("synthetic storage set mismatch")
    for key, names in storage_groups.items():
        if len(names) > 1 and frozenset(names) not in _expected_aliases("flow"):
            raise PreparationError(f"synthetic unauthenticated storage alias: {key}")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise PreparationError("preparation report must be an absent absolute path")
    _require_existing_directory(path.parent, "preparation report parent")
    body = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    def write_report(stream: Any, digest: Any) -> int:
        written = _write_all(stream, body)
        digest.update(body)
        return written

    _write_no_replace(path, write_report)


def _synthetic_pickle(shape: tuple[int, ...] = (2, 2), stride: tuple[int, ...] = (2, 1), offset: int = 0, key: str = "0", numel: int = 4) -> bytes:
    class Storage:
        pass

    class Tensor:
        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            storage = Storage()
            storage.key, storage.numel = key, numel
            return (_synthetic_rebuild, (storage, offset, shape, stride, False, None))

    def persistent_id(value: Any) -> Any:
        if isinstance(value, Storage):
            return ("storage", "FloatStorage", value.key, "cpu", value.numel)
        return None

    class Pickler(pickle.Pickler):
        def persistent_id(self, value: Any) -> Any:
            return persistent_id(value)

    stream = io.BytesIO()
    Pickler(stream, protocol=2).dump({"w": Tensor()})
    raw = stream.getvalue().replace(b"c__main__\n_synthetic_rebuild\n", b"ctorch._utils\n_rebuild_tensor_v2\n")
    return raw.replace(b"X\x0c\x00\x00\x00FloatStorage", b"ctorch\nFloatStorage\n")


def _synthetic_rebuild(*args: Any) -> None:
    del args


def _synthetic_zip(path: Path, pickle_bytes: bytes, storage: bytes = b"\0" * 16, duplicate_member: bool = False) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", pickle_bytes)
        if duplicate_member:
            archive.writestr("archive/data.pkl", pickle_bytes)
        archive.writestr("archive/data/0", storage)
        archive.writestr("archive/version", b"3\n")


def self_test() -> None:
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-component-prepare-", dir=temp_root) as temp:
        root = Path(__file__).resolve().parents[2]
        work = Path(temp)
        valid = work / "valid.pt"
        _synthetic_zip(valid, _synthetic_pickle())
        raw, archive = inspection.read_data_pickle(valid)
        state = inspection.run_restricted_parser(raw, root, "synthetic")
        expected = {"tensor_count": 1, "manifest_sha256": state["manifest_sha256"], "storage_manifest_sha256": state["storage_manifest_sha256"], "storage_member_count": 1}
        class PartialWriter:
            def __init__(self) -> None:
                self.data = bytearray()

            def write(self, data: bytes) -> int:
                count = max(1, len(data) // 2)
                self.data.extend(data[:count])
                return count

        partial = PartialWriter()
        assert _write_all(partial, b"short writes are completed") == len(b"short writes are completed")

        class StalledWriter:
            def write(self, data: bytes) -> int:
                del data
                return 0

        try:
            _write_all(StalledWriter(), b"stalled")
        except PreparationError:
            pass
        else:
            raise AssertionError("non-progressing short write accepted")

        class ExcessWriter:
            def write(self, data: bytes) -> int:
                return len(data) + 1

        try:
            _write_all(ExcessWriter(), b"excess")
        except PreparationError:
            pass
        else:
            raise AssertionError("over-reporting short write accepted")
        output = work / "prepared.safetensors"
        report = _prepare_from_zip(valid, "flow", output, None, root, expected)
        assert report["status"] == "PREPARED_SAFETENSORS_READY" and output.is_file()
        assert inspection.sha256_file(output) == report["output"]["sha256"]
        encoded = output.read_bytes()
        header_size = int.from_bytes(encoded[:8], "little")
        header = json.loads(encoded[8 : 8 + header_size])
        assert list(header) == sorted(header) and header["w"]["data_offsets"] == [0, 16]
        assert all(isinstance(value, str) for value in header["__metadata__"].values())
        assert encoded[8 + header_size :] == b"\0" * 16
        existing = work / "existing.safetensors"
        existing.write_bytes(b"original")
        try:
            _prepare_from_zip(valid, "flow", existing, None, root, expected)
        except PreparationError:
            pass
        else:
            raise AssertionError("existing output was replaced")
        failed_report = work / "existing-report.json"
        failed_report.write_bytes(b"keep")
        failed_output = work / "failed-report.safetensors"
        try:
            _prepare_from_zip(valid, "flow", failed_output, failed_report, root, expected)
        except PreparationError:
            assert not failed_output.exists(), "report failure left newly created output"
            assert failed_report.read_bytes() == b"keep", "existing report was modified"
        else:
            raise AssertionError("existing report did not fail no-replace")
        report_link_target = work / "report-target"
        report_link_target.mkdir()
        report_link = work / "report-link"
        report_link.symlink_to(report_link_target, target_is_directory=True)
        try:
            _write_report(report_link / "manifest.json", {"status": "test"})
        except PreparationError:
            pass
        else:
            raise AssertionError("symlink report parent accepted")
        bad_storage = work / "bad-storage.pt"
        _synthetic_zip(bad_storage, _synthetic_pickle(), b"\0" * 8)
        try:
            _prepare_from_zip(bad_storage, "flow", work / "bad.safetensors", None, root, expected)
        except (PreparationError, inspection.InspectionError):
            assert not (work / "bad.safetensors").exists(), "failed preparation left output"
        else:
            raise AssertionError("out-of-bounds storage accepted")
        duplicate = work / "duplicate.pt"
        _synthetic_zip(duplicate, _synthetic_pickle(), duplicate_member=True)
        try:
            inspection.read_data_pickle(duplicate)
        except inspection.InspectionError:
            pass
        else:
            raise AssertionError("duplicate ZIP member accepted")
        alias_state = dict(state)
        alias_state["tensors"] = dict(state["tensors"])
        alias_state["tensors"]["other"] = dict(state["tensors"]["w"])
        alias_state["tensor_count"] = 2
        alias_expected = dict(expected)
        alias_expected["tensor_count"] = 2
        try:
            _validate_test_manifest(alias_state, archive, alias_expected)
        except PreparationError:
            pass
        else:
            raise AssertionError("unauthenticated storage alias accepted")
        for kwargs, label in [
            ({"stride": (1, 1)}, "non-contiguous"),
            ({"offset": 1}, "non-zero offset"),
            ({"key": "../0"}, "path traversal"),
        ]:
            malformed = work / f"{label}.pt"
            values = {"shape": (2, 2), "stride": (2, 1), "offset": 0, "key": "0", "numel": 4}
            values.update(kwargs)
            _synthetic_zip(malformed, _synthetic_pickle(**values))
            try:
                _prepare_from_zip(malformed, "flow", work / f"{label}.safetensors", None, root, expected)
            except (PreparationError, inspection.InspectionError):
                pass
            else:
                raise AssertionError(f"{label} accepted")
        dtype_state = dict(state)
        dtype_state["tensors"] = {"w": {**state["tensors"]["w"], "dtype": "F64"}}
        try:
            _validate_test_manifest(dtype_state, archive, expected)
        except PreparationError:
            pass
        else:
            raise AssertionError("unsupported dtype accepted")
        reserved_state = dict(state)
        reserved_state["tensors"] = {"__metadata__": dict(state["tensors"]["w"])}
        try:
            _validate_test_manifest(reserved_state, archive, expected)
        except PreparationError:
            pass
        else:
            raise AssertionError("reserved safetensors metadata name accepted")
    print("cosyvoice2_component_prepare self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--component", choices=tuple(inspection.COMPONENTS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--qwen-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    names = ("--self-test", "--component", "--checkpoint", "--source", "--config", "--qwen-config", "--output", "--report")
    counts = {name: sum(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:]) for name in names}
    if any(count > 1 for count in counts.values()):
        parser.error("duplicate CLI option is not allowed")
    if args.self_test:
        if any(value is not None for value in (args.component, args.checkpoint, args.source, args.config, args.qwen_config, args.output, args.report)):
            parser.error("--self-test accepts no paths")
        return args
    if any(value is None for value in (args.component, args.checkpoint, args.source, args.config, args.output, args.report)):
        parser.error("normal preparation requires component, checkpoint, source, config, output, and report")
    inspection.validate_component_config_requirements(args.component, args.qwen_config)
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        try:
            self_test()
        except Exception as error:
            print(f"cosyvoice2_component_prepare self-test FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        print("Linux x86_64 VAST is required", file=sys.stderr)
        return 2
    if os.environ.get("VOKRA_PUBLISH_ON_VAST") != "1":
        print("VOKRA_PUBLISH_ON_VAST=1 is required", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[2]
    try:
        if not root.joinpath(".git").is_dir() or subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True):
            raise PreparationError("clean Vokra checkout is required")
        expected = inspection.COMPONENTS[args.component]
        inspection.require_regular(args.checkpoint, expected["path"])
        if args.checkpoint.stat().st_size != expected["bytes"] or inspection.sha256_file(args.checkpoint) != expected["sha256"]:
            raise PreparationError("raw checkpoint identity mismatch")
        config_record = inspection.authenticate_config(args.config)
        qwen_record = inspection.authenticate_qwen_config(args.qwen_config) if args.qwen_config is not None else None
        source_record = inspection.authenticate_source(args.source, args.component)
        provenance: dict[str, Any] = {
            "model_config": config_record,
            "official_source": source_record,
            "artifact": {
                "repository": inspection.MODEL_REPOSITORY,
                "revision": inspection.MODEL_REVISION,
                "path": expected["path"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            },
        }
        if qwen_record is not None:
            provenance["qwen_config"] = qwen_record
        report = _prepare_from_zip(args.checkpoint, args.component, args.output, args.report, root, provenance=provenance)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"cosyvoice2_component_prepare: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
