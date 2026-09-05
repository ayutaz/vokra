#!/usr/bin/env -S uv run --no-project --python 3.12 python
"""Inspect only the structural manifest in the CosyVoice2 HiFT checkpoint.

This is deliberately a small, standard-library-only VAST helper. It reads the
single ``data.pkl`` member from a verified PyTorch ZIP archive and invokes the
repository's restricted state-dict unpickler. Tensor storage members are never
opened, and no model, torch, or upstream Python package is imported.
"""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
MODEL_PATH = "hift.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_PATH}?download=true"
MODEL_BYTES = 83_390_254
MODEL_SHA256 = "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
GENERATOR_PATH = "cosyvoice/hifigan/generator.py"
# These identities are recorded in cosyvoice2_inspect.py for this exact
# source revision. Keeping them here avoids importing the composite inspector.
GENERATOR_SHA256 = "f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626"
GENERATOR_GIT_BLOB_SHA1 = "326a1a70ae7707662939c20493b3a8e4b0906216"
LICENSE_PATH = "LICENSE"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
FORMAT = "vokra-cosyvoice2-hift-inspection-v1"
PICKLE_SOURCE = f"hf://{MODEL_REPOSITORY}@{MODEL_REVISION}:{MODEL_PATH}/data.pkl"
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_DATA_PICKLE_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 100_000
MAX_DIMENSION = 1 << 40


class InspectionError(ValueError):
    """The artifact or authenticated source is not safe to inspect."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise InspectionError(f"git command failed: {args!r}: {error}") from error


def safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or name.startswith("/")
        or "\\" in name
        or ".." in path.parts
        or path.is_absolute()
        or any(not component for component in path.parts)
        or len(name) > 4096
    ):
        raise InspectionError(f"unsafe ZIP member: {name!r}")


def read_data_pickle(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Validate ZIP metadata and read exactly one bounded data.pkl member."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise InspectionError(f"checkpoint is not a readable ZIP: {error}") from error
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
            raise InspectionError("ZIP member count is outside the safe bound")
        names: set[str] = set()
        total_bytes = 0
        data_infos: list[zipfile.ZipInfo] = []
        for info in infos:
            name = info.filename
            safe_member_name(name)
            if name in names or info.is_dir() or info.flag_bits & 1:
                raise InspectionError(f"duplicate, directory, or encrypted ZIP member: {name!r}")
            mode = info.external_attr >> 16
            if mode not in (0, 0o600, 0o100644, 0o100755):
                raise InspectionError(f"non-regular ZIP member: {name!r}")
            if info.file_size < 0:
                raise InspectionError(f"negative ZIP member size: {name!r}")
            total_bytes += info.file_size
            if total_bytes > MAX_ARCHIVE_BYTES:
                raise InspectionError("ZIP uncompressed byte bound exceeded")
            names.add(name)
            if PurePosixPath(name).name == "data.pkl":
                data_infos.append(info)
        if len(data_infos) != 1:
            raise InspectionError(f"expected exactly one */data.pkl member, got {len(data_infos)}")
        data_info = data_infos[0]
        if data_info.file_size > MAX_DATA_PICKLE_BYTES:
            raise InspectionError("data.pkl exceeds bounded parser input size")
        data = bytearray()
        try:
            with archive.open(data_info, "r") as stream:
                while True:
                    block = stream.read(min(1 << 20, MAX_DATA_PICKLE_BYTES + 1 - len(data)))
                    if not block:
                        break
                    data.extend(block)
                    if len(data) > MAX_DATA_PICKLE_BYTES:
                        raise InspectionError("data.pkl read bound exceeded")
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            raise InspectionError(f"data.pkl read failed: {error}") from error
        if len(data) != data_info.file_size:
            raise InspectionError("data.pkl size changed while reading")
        prefix = str(PurePosixPath(data_info.filename).parent)
        prefix = "" if prefix == "." else prefix + "/"
        storage_members = {
            name[len(prefix + "data/") :]
            for name in names
            if name.startswith(prefix + "data/") and name != prefix + "data/"
        }
        return bytes(data), {
            "member_count": len(infos),
            "uncompressed_bytes": total_bytes,
            "data_pickle_member": data_info.filename,
            "data_pickle_bytes": data_info.file_size,
            "storage_member_count": len(storage_members),
            "storage_members": sorted(storage_members),
        }


def load_restricted_parser(root: Path) -> Any:
    parser_path = root / "tools/audit/torch_pickle_manifest.py"
    if not parser_path.is_file():
        raise InspectionError(f"restricted parser is missing: {parser_path}")
    spec = importlib.util.spec_from_file_location("vokra_torch_pickle_manifest", parser_path)
    if spec is None or spec.loader is None:
        raise InspectionError("could not load restricted parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inspect_checkpoint(path: Path, root: Path) -> dict[str, Any]:
    raw, archive = read_data_pickle(path)
    parser = load_restricted_parser(root)
    try:
        state_dict = parser.load_manifest(io.BytesIO(raw))
        if type(state_dict) not in (dict, collections.OrderedDict):
            raise InspectionError("restricted parser did not return a plain dict or OrderedDict")
        if len(state_dict) > MAX_TENSORS:
            raise InspectionError("tensor count bound exceeded")
        for name, tensor in state_dict.items():
            if not isinstance(name, str) or not name or "\x00" in name or "/" in name or "\\" in name:
                raise InspectionError(f"unsafe tensor name: {name!r}")
            if any(dimension > MAX_DIMENSION for dimension in tensor.shape):
                raise InspectionError(f"tensor dimension bound exceeded: {name}")
            if tensor.storage_numel > MAX_DIMENSION * MAX_DIMENSION:
                raise InspectionError(f"storage element bound exceeded: {name}")
            if tensor.storage_key not in archive["storage_members"]:
                raise InspectionError(f"tensor storage member is missing: {tensor.storage_key}")
        manifest = parser.render_manifest(state_dict, PICKLE_SOURCE, hashlib.sha256(raw).hexdigest())
    except Exception as error:
        if isinstance(error, InspectionError):
            raise
        raise InspectionError(f"restricted data.pkl parser rejected checkpoint: {error}") from error
    manifest["archive"] = archive
    manifest["payload_reads"] = "DATA_PKL_ONLY; TENSOR_STORAGE_MEMBERS_NOT_OPENED"
    return manifest


def authenticate_source(source: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise InspectionError(f"source checkout is missing: {source}")
    head = git(source, "rev-parse", "HEAD")
    origin = git(source, "remote", "get-url", "origin")
    clean = git(source, "status", "--porcelain", "--untracked-files=all")
    if head != SOURCE_REVISION or origin != SOURCE_REPOSITORY or clean:
        raise InspectionError("CosyVoice source identity or clean-checkout mismatch")
    role = source / GENERATOR_PATH
    if not role.is_file() or role.is_symlink():
        raise InspectionError("HiFT generator source role is missing or symlinked")
    role_sha = sha256_file(role)
    role_blob = git_blob_sha1(role)
    if role_sha != GENERATOR_SHA256 or role_blob != GENERATOR_GIT_BLOB_SHA1:
        raise InspectionError("HiFT generator source role hash mismatch")
    if "HiFTGenerator" not in role.read_text(encoding="utf-8", errors="replace"):
        raise InspectionError("HiFT generator source marker is missing")
    license_path = source / LICENSE_PATH
    if not license_path.is_file() or license_path.is_symlink() or sha256_file(license_path) != LICENSE_SHA256:
        raise InspectionError("CosyVoice Apache LICENSE hash mismatch")
    if "Apache License" not in license_path.read_text(encoding="utf-8", errors="replace"):
        raise InspectionError("CosyVoice Apache LICENSE marker is missing")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "resolved_revision": head, "clean": True, "role": GENERATOR_PATH, "role_sha256": role_sha, "role_git_blob_sha1": role_blob, "license": LICENSE_PATH, "license_sha256": LICENSE_SHA256}


def write_error(output: Path, message: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"format": FORMAT, "status": "BLOCKED", "inspection_status": "INSPECTION_ERROR", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "NOT_RUN", "metal_status": "NOT_RUN", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "blockers": [message]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect(checkpoint: Path, source: Path, output: Path, root: Path) -> int:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise InspectionError("inspection output must be absent or empty")
    if checkpoint.stat().st_size != MODEL_BYTES:
        raise InspectionError("hift.pt byte count does not match the pinned artifact")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != MODEL_SHA256:
        raise InspectionError("hift.pt SHA-256 does not match the pinned artifact")
    source_record = authenticate_source(source)
    checkpoint_manifest = inspect_checkpoint(checkpoint, root)
    payload = {"format": FORMAT, "status": "BLOCKED", "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "NOT_RUN", "metal_status": "NOT_RUN", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "artifact": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "path": MODEL_PATH, "url": MODEL_URL, "bytes": MODEL_BYTES, "sha256": actual_sha}, "official_source": source_record, "checkpoint": checkpoint_manifest, "blockers": ["Native HiFT binder is not implemented; this is structural evidence only.", "No CPU, Metal, or numerical parity execution was performed."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 2


class _SyntheticStorage:
    pass


def _synthetic_rebuild(*args: Any) -> None:
    del args


class _SyntheticTensor:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (_synthetic_rebuild, (_SyntheticStorage(), 0, (2, 2), (2, 1), False, None))


def synthetic_pickle() -> bytes:
    """Build one safe state-dict pickle without torch or checkpoint data."""
    class Pickler(pickle.Pickler):
        def persistent_id(self, value: Any) -> Any:
            if isinstance(value, _SyntheticStorage):
                return ("storage", "FloatStorage", "0", "cpu", 4)
            return None
    stream = io.BytesIO()
    Pickler(stream, protocol=2).dump(collections.OrderedDict([("w", _SyntheticTensor())]))
    raw = stream.getvalue().replace(
        f"c{__name__}\n_synthetic_rebuild\n".encode(),
        b"ctorch._utils\n_rebuild_tensor_v2\n",
    )
    return raw.replace(b"X\x0c\x00\x00\x00FloatStorage", b"ctorch\nFloatStorage\n")


def synthetic_plain_dict_pickle() -> bytes:
    """Build the same safe tensor record with a built-in dict root."""
    class Pickler(pickle.Pickler):
        def persistent_id(self, value: Any) -> Any:
            if isinstance(value, _SyntheticStorage):
                return ("storage", "FloatStorage", "0", "cpu", 4)
            return None

    stream = io.BytesIO()
    Pickler(stream, protocol=2).dump({"w": _SyntheticTensor()})
    raw = stream.getvalue().replace(
        f"c{__name__}\n_synthetic_rebuild\n".encode(),
        b"ctorch._utils\n_rebuild_tensor_v2\n",
    )
    return raw.replace(b"X\x0c\x00\x00\x00FloatStorage", b"ctorch\nFloatStorage\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-hift-self-test-") as temp:
        root = Path(temp)
        valid = root / "valid.pt"
        with zipfile.ZipFile(valid, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", synthetic_pickle())
            archive.writestr("archive/data/0", b"\0" * 16)
            archive.writestr("archive/version", b"3\n")
        raw, info = read_data_pickle(valid)
        assert raw == synthetic_pickle() and info["data_pickle_member"] == "archive/data.pkl"
        manifest = inspect_checkpoint(valid, Path(__file__).resolve().parents[2])
        assert manifest["tensor_count"] == 1 and manifest["tensors"]["w"]["shape"] == [2, 2]
        plain = root / "plain-dict.pt"
        with zipfile.ZipFile(plain, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", synthetic_plain_dict_pickle())
            archive.writestr("archive/data/0", b"\0" * 16)
        plain_manifest = inspect_checkpoint(plain, Path(__file__).resolve().parents[2])
        assert plain_manifest["tensor_count"] == 1
        unsafe = root / "unsafe.pt"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../archive/data.pkl", synthetic_pickle())
        try:
            read_data_pickle(unsafe)
        except InspectionError:
            pass
        else:
            raise AssertionError("unsafe ZIP member accepted")
        multiple = root / "multiple.pt"
        with zipfile.ZipFile(multiple, "w") as archive:
            archive.writestr("a/data.pkl", synthetic_pickle())
            archive.writestr("b/data.pkl", synthetic_pickle())
        try:
            read_data_pickle(multiple)
        except InspectionError:
            pass
        else:
            raise AssertionError("multiple data.pkl members accepted")
        malicious = root / "malicious.pt"
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("archive/data.pkl", b"\x80\x02cposix\nsystem\n.")
            archive.writestr("archive/data/0", b"")
        try:
            inspect_checkpoint(malicious, Path(__file__).resolve().parents[2])
        except InspectionError:
            pass
        else:
            raise AssertionError("malicious pickle global accepted")
    print("cosyvoice2_hift_inspect self-test: OK")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.checkpoint, args.source, args.output)):
            parser.error("--self-test accepts no paths")
        try:
            self_test()
        except Exception as error:
            print(f"cosyvoice2_hift_inspect self-test FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    if any(value is None for value in (args.checkpoint, args.source, args.output)):
        parser.error("normal run requires --checkpoint, --source, and --output")
    try:
        return inspect(args.checkpoint, args.source, args.output, Path(__file__).resolve().parents[2])
    except Exception as error:
        write_error(args.output, str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
