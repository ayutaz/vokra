#!/usr/bin/env python3
"""Audit a PyTorch ZIP checkpoint nested in a public HF tar via HTTP Range.

The large NeMo archive and tensor storage members are never downloaded.  The
tool walks the outer tar's 512-byte headers, presents the selected checkpoint
member as a bounded seekable Range view, reads the inner ZIP central directory
and its small ``data.pkl``, then delegates to ``torch_pickle_manifest``'s
restricted unpickler.  That unpickler imports no checkpoint-selected code and
returns only tensor names, dtypes, shapes, strides, and storage references.

Every network response is checked by ``hf_tar_contract.RangeFetcher`` for an
exact ``206 Content-Range``.  A per-read cap and a total-transfer budget make a
server/parser surprise fail before it can turn into a multi-gigabyte download.

Example::

    uv run --no-project --python 3.12 python \
      tools/audit/hf_tar_torch_contract.py \
      nvidia/canary-1b-v2 canary-1b-v2.nemo ./model_weights.ckpt \
      --revision <commit> --expected-archive-size 6358958080 \
      --expected-checkpoint-size 3853798427 --omit-tensors

This is structural metadata inspection, not model execution or numerical
parity.  It does not authenticate, upload, or write unless ``--output`` or
``--state-dict-output`` is explicitly supplied.  The latter writes the direct
``vokra-pytorch-state-dict-manifest-v1`` object used as a strict-checkpoint
fixture while the normal report keeps the Range-transfer audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any, Callable

from hf_tar_contract import (
    BLOCK_SIZE,
    RangeFetcher,
    TarContractError,
    list_tar_contract,
    manifest_sha256 as tar_manifest_sha256,
)
from torch_pickle_manifest import load_manifest, render_manifest


DEFAULT_MAX_SINGLE_READ = 16 * 1024 * 1024
DEFAULT_TRANSFER_BUDGET = 32 * 1024 * 1024
DEFAULT_MAX_DATA_PICKLE = 16 * 1024 * 1024


class RangeSlice(io.RawIOBase):
    """Seekable view onto one byte range of a larger exact-range source."""

    def __init__(
        self,
        read_at: Callable[[int, int], bytes],
        base_offset: int,
        size: int,
        *,
        max_single_read: int = DEFAULT_MAX_SINGLE_READ,
        transfer_budget: int = DEFAULT_TRANSFER_BUDGET,
    ) -> None:
        super().__init__()
        if base_offset < 0 or size <= 0:
            raise ValueError(
                f"invalid RangeSlice base_offset={base_offset}, size={size}"
            )
        if max_single_read <= 0 or transfer_budget <= 0:
            raise ValueError("RangeSlice limits must be positive")
        self._read_at = read_at
        self._base_offset = base_offset
        self._size = size
        self._position = 0
        self._max_single_read = max_single_read
        self._transfer_budget = transfer_budget
        self.bytes_received = 0
        self.read_count = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"unsupported seek whence {whence}")
        if position < 0 or position > self._size:
            raise ValueError(
                f"seek position {position} outside checkpoint slice 0..{self._size}"
            )
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = self._size - self._position
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        if size == 0:
            return b""
        if size > self._max_single_read:
            raise TarContractError(
                f"inner ZIP requested {size} bytes in one read; cap is "
                f"{self._max_single_read}"
            )
        if self.bytes_received + size > self._transfer_budget:
            raise TarContractError(
                f"inner ZIP transfer would exceed {self._transfer_budget}-byte budget"
            )
        payload = self._read_at(self._base_offset + self._position, size)
        if len(payload) != size:
            raise TarContractError(
                f"inner ZIP range returned {len(payload)} bytes, expected {size}"
            )
        self._position += size
        self.bytes_received += size
        self.read_count += 1
        return payload


def _normalize_member(name: str) -> str:
    normalized = posixpath.normpath(name)
    # A root directory entry named `./` is a normal tar preamble and cannot
    # match a checkpoint filename. Parent traversal and absolute paths remain
    # malformed even though this tool never extracts to the filesystem.
    if normalized == ".":
        return normalized
    if normalized == ".." or normalized.startswith(("../", "/")):
        raise TarContractError(f"unsafe tar member name {name!r}")
    return normalized.removeprefix("./")


def _find_tar_member(members: list[Any], requested: str) -> Any:
    wanted = _normalize_member(requested)
    matches = [member for member in members if _normalize_member(member.name) == wanted]
    if len(matches) != 1:
        raise TarContractError(
            f"tar member {requested!r} matched {[member.name for member in matches]}"
        )
    return matches[0]


def _find_data_pickle(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    matches = [
        info
        for info in infos
        if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")
    ]
    if len(matches) != 1:
        raise TarContractError(
            f"inner checkpoint has {len(matches)} data.pkl members: "
            f"{[info.filename for info in matches]}"
        )
    info = matches[0]
    if info.flag_bits & 0x1:
        raise TarContractError("inner data.pkl is encrypted")
    if info.compress_type != zipfile.ZIP_STORED:
        raise TarContractError(
            f"inner data.pkl compression={info.compress_type}; expected ZIP_STORED"
        )
    return info


def audit_nested_checkpoint(
    read_at: Callable[[int, int], bytes],
    archive_size: int,
    checkpoint_member: str,
    source_label: str,
    *,
    expected_checkpoint_size: int | None = None,
    max_data_pickle: int = DEFAULT_MAX_DATA_PICKLE,
    max_single_read: int = DEFAULT_MAX_SINGLE_READ,
    transfer_budget: int = DEFAULT_TRANSFER_BUDGET,
) -> tuple[dict[str, Any], RangeSlice]:
    members = list_tar_contract(read_at, archive_size)
    member = _find_tar_member(members, checkpoint_member)
    if expected_checkpoint_size is not None and member.size != expected_checkpoint_size:
        raise TarContractError(
            f"checkpoint member size {member.size} != expected "
            f"{expected_checkpoint_size}"
        )
    checkpoint = RangeSlice(
        read_at,
        member.payload_offset,
        member.size,
        max_single_read=max_single_read,
        transfer_budget=transfer_budget,
    )
    with zipfile.ZipFile(checkpoint, "r") as bundle:
        infos = bundle.infolist()
        data_info = _find_data_pickle(infos)
        if data_info.file_size <= 0 or data_info.file_size > max_data_pickle:
            raise TarContractError(
                f"data.pkl size {data_info.file_size} outside 1..{max_data_pickle}"
            )
        with bundle.open(data_info, "r") as stream:
            data_pickle = stream.read(max_data_pickle + 1)
        if len(data_pickle) != data_info.file_size:
            raise TarContractError(
                f"read {len(data_pickle)} data.pkl bytes, expected {data_info.file_size}"
            )

    pickle_sha256 = hashlib.sha256(data_pickle).hexdigest()
    state_dict = load_manifest(io.BytesIO(data_pickle))
    manifest = render_manifest(state_dict, source_label, pickle_sha256)
    report = {
        "format": "vokra-hf-tar-pytorch-contract-v1",
        "archive_size": archive_size,
        "tar_member_count": len(members),
        "tar_manifest_sha256": tar_manifest_sha256(members),
        "checkpoint_member": {
            "name": member.name,
            "size": member.size,
            "header_offset": member.header_offset,
            "payload_offset": member.payload_offset,
        },
        "inner_zip_member_count": len(infos),
        "data_pickle": {
            "name": data_info.filename,
            "size": data_info.file_size,
            "compressed_size": data_info.compress_size,
            "sha256": pickle_sha256,
        },
        "checkpoint_range_reads": checkpoint.read_count,
        "checkpoint_bytes_received": checkpoint.bytes_received,
        "state_dict": manifest,
    }
    return report, checkpoint


def _plain_state_dict_pickle() -> bytes:
    return (
        b"\x80\x02"
        b"ccollections\nOrderedDict\n)R("
        b"X\x06\x00\x00\x00weight"
        b"ctorch._utils\n_rebuild_tensor_v2\n("
        b"(X\x07\x00\x00\x00storage"
        b"ctorch\nFloatStorage\n"
        b"X\x01\x00\x00\x000"
        b"X\x03\x00\x00\x00cpu"
        b"K\x06tQ"
        b"K\x00"
        b"K\x02K\x03\x86"
        b"K\x03K\x01\x86"
        b"\x89"
        b"ccollections\nOrderedDict\n)R"
        b"tRu."
    )


def self_test() -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("archive/data.pkl", _plain_state_dict_pickle())
        bundle.writestr("archive/data/0", b"tensor-payload-must-not-be-read")
    checkpoint_bytes = inner.getvalue()

    import tarfile

    outer = io.BytesIO()
    with tarfile.open(fileobj=outer, mode="w") as archive:
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = len(checkpoint_bytes)
        archive.addfile(info, io.BytesIO(checkpoint_bytes))
    archive_bytes = outer.getvalue()
    requested: list[tuple[int, int]] = []

    def read_at(offset: int, size: int) -> bytes:
        requested.append((offset, size))
        return archive_bytes[offset : offset + size]

    report, checkpoint = audit_nested_checkpoint(
        read_at,
        len(archive_bytes),
        "model_weights.ckpt",
        "fixture@revision:model_weights.ckpt",
        expected_checkpoint_size=len(checkpoint_bytes),
    )
    assert report["state_dict"]["tensor_count"] == 1
    assert report["state_dict"]["tensors"]["weight"]["shape"] == [2, 3]
    assert checkpoint.bytes_received < len(checkpoint_bytes)
    # Exact payload marker never appears in any requested byte range.
    assert not any(
        b"tensor-payload-must-not-be-read"
        in archive_bytes[offset : offset + size]
        for offset, size in requested
    )
    print("hf_tar_torch_contract: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?")
    parser.add_argument("filename", nargs="?")
    parser.add_argument("checkpoint_member", nargs="?")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--expected-archive-size", type=int)
    parser.add_argument("--expected-checkpoint-size", type=int)
    parser.add_argument("--max-data-pickle", type=int, default=DEFAULT_MAX_DATA_PICKLE)
    parser.add_argument("--max-single-read", type=int, default=DEFAULT_MAX_SINGLE_READ)
    parser.add_argument("--transfer-budget", type=int, default=DEFAULT_TRANSFER_BUDGET)
    parser.add_argument("--tensor-regex")
    parser.add_argument("--omit-tensors", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dict-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.repo or not args.filename or not args.checkpoint_member:
        raise SystemExit(
            "repo, filename and checkpoint_member are required unless --self-test is used"
        )
    if args.omit_tensors and args.tensor_regex:
        raise SystemExit("--omit-tensors and --tensor-regex are mutually exclusive")

    fetcher = RangeFetcher(args.repo, args.revision, args.filename)
    first_header = fetcher.read(0, BLOCK_SIZE)
    assert fetcher.total_size is not None
    archive_size = fetcher.total_size
    if (
        args.expected_archive_size is not None
        and archive_size != args.expected_archive_size
    ):
        raise SystemExit(
            f"remote archive size {archive_size} != expected "
            f"{args.expected_archive_size}"
        )

    def read_at(offset: int, size: int) -> bytes:
        if offset == 0 and size == BLOCK_SIZE:
            return first_header
        return fetcher.read(offset, size)

    source_label = (
        f"{args.repo}@{args.revision}:{args.filename}!{args.checkpoint_member}"
    )
    report, _checkpoint = audit_nested_checkpoint(
        read_at,
        archive_size,
        args.checkpoint_member,
        source_label,
        expected_checkpoint_size=args.expected_checkpoint_size,
        max_data_pickle=args.max_data_pickle,
        max_single_read=args.max_single_read,
        transfer_budget=args.transfer_budget,
    )
    report["repo"] = args.repo
    report["revision"] = args.revision
    report["filename"] = args.filename
    report["http_range_request_count"] = fetcher.request_count
    report["http_bytes_received"] = fetcher.bytes_received

    if args.state_dict_output is not None:
        state_dict_body = (
            json.dumps(
                report["state_dict"], ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n"
        )
        args.state_dict_output.write_text(state_dict_body, encoding="utf-8")

    tensors = report["state_dict"]["tensors"]
    if args.omit_tensors:
        report["state_dict"]["tensors"] = {}
    elif args.tensor_regex:
        pattern = re.compile(args.tensor_regex)
        report["state_dict"]["tensors"] = {
            name: value for name, value in tensors.items() if pattern.search(name)
        }

    body = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
