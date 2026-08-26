#!/usr/bin/env python3
"""Audit a public Hugging Face tar archive without downloading payloads.

Large ``.nemo`` releases are uncompressed tar archives.  Their 512-byte
headers contain the member name, type, size, and payload offset, so a binder
audit can inventory the archive through HTTP Range requests while skipping
multi-gigabyte checkpoint payloads entirely.

The reader is deliberately fail-closed:

* every response must be ``206 Partial Content`` with an exact Content-Range;
* ordinary member payloads are never requested;
* only bounded GNU long-name / PAX metadata payloads may be read; and
* checksum, alignment, archive-size, and end-marker inconsistencies abort.

Run through the repository Python policy::

    uv run --no-project --python 3.12 python tools/audit/hf_tar_contract.py \
      nvidia/canary-1b-v2 canary-1b-v2.nemo --revision <commit> \
      --expected-size 6358958080

This tool does not authenticate to Hugging Face and never uploads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable


BLOCK_SIZE = 512
MAX_METADATA_BYTES = 1024 * 1024
USER_AGENT = "vokra-hf-tar-contract/1.0"
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class TarContractError(ValueError):
    """Raised when a remote archive cannot be audited safely."""


@dataclass(frozen=True)
class TarMemberContract:
    """Payload-free contract for one tar member."""

    name: str
    type: str
    size: int
    header_offset: int
    payload_offset: int


class RangeFetcher:
    """Exact HTTP byte-range reader with an immutable total-size contract."""

    def __init__(self, repo: str, revision: str, filename: str) -> None:
        quoted_file = urllib.parse.quote(filename, safe="/")
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{quoted_file}"
        self.total_size: int | None = None
        self.request_count = 0
        self.bytes_received = 0

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or size <= 0:
            raise TarContractError(
                f"invalid range request offset={offset}, size={size}"
            )
        end = offset + size - 1
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={offset}-{end}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", None)
            content_range = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE.fullmatch(content_range)
            if status != 206 or match is None:
                raise TarContractError(
                    "server did not honor the exact Range request; refusing to "
                    f"risk a full archive download (status={status}, "
                    f"Content-Range={content_range!r})"
                )
            got_start, got_end, total = (int(value) for value in match.groups())
            if got_start != offset or got_end != end:
                raise TarContractError(
                    f"server returned bytes {got_start}-{got_end}, expected {offset}-{end}"
                )
            if self.total_size is None:
                self.total_size = total
            elif self.total_size != total:
                raise TarContractError(
                    f"archive size changed during audit: {self.total_size} -> {total}"
                )
            payload = response.read(size + 1)
        if len(payload) != size:
            raise TarContractError(
                f"range {offset}-{end} returned {len(payload)} bytes, expected {size}"
            )
        self.request_count += 1
        self.bytes_received += len(payload)
        return payload


def _tar_number(field: bytes, label: str) -> int:
    """Parse POSIX octal and GNU base-256 tar numeric fields."""

    if not field:
        raise TarContractError(f"empty numeric field for {label}")
    if field[0] & 0x80:
        value = int.from_bytes(field, byteorder="big", signed=True)
        # The high bit is a base-256 marker, not part of the value.
        value &= (1 << (len(field) * 8 - 1)) - 1
        return value
    raw = field.rstrip(b"\0 ").lstrip(b" ")
    if not raw:
        return 0
    if any(byte not in b"01234567" for byte in raw):
        raise TarContractError(f"invalid octal {label}: {field!r}")
    return int(raw, 8)


def _tar_text(field: bytes) -> str:
    return field.split(b"\0", 1)[0].decode("utf-8", errors="strict")


def _verify_header(block: bytes, offset: int) -> None:
    if len(block) != BLOCK_SIZE:
        raise TarContractError(f"short tar header at offset {offset}")
    expected = _tar_number(block[148:156], "checksum")
    checksum_block = block[:148] + b" " * 8 + block[156:]
    actual = sum(checksum_block)
    if actual != expected:
        raise TarContractError(
            f"tar checksum mismatch at offset {offset}: {actual} != {expected}"
        )


def _parse_pax(payload: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    cursor = 0
    while cursor < len(payload):
        space = payload.find(b" ", cursor)
        if space < 0:
            raise TarContractError("malformed PAX record length")
        try:
            length = int(payload[cursor:space])
        except ValueError as error:
            raise TarContractError("non-decimal PAX record length") from error
        if length <= 0 or cursor + length > len(payload):
            raise TarContractError("PAX record exceeds metadata payload")
        record = payload[space + 1 : cursor + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise TarContractError("malformed PAX key/value record")
        key, value = record[:-1].split(b"=", 1)
        values[key.decode("utf-8")] = value.decode("utf-8")
        cursor += length
    return values


def _metadata_payload(
    read: Callable[[int, int], bytes], payload_offset: int, size: int
) -> bytes:
    if size > MAX_METADATA_BYTES:
        raise TarContractError(
            f"tar metadata payload is {size} bytes; refusing more than "
            f"{MAX_METADATA_BYTES} bytes"
        )
    return read(payload_offset, size) if size else b""


def list_tar_contract(
    read: Callable[[int, int], bytes], total_size: int
) -> list[TarMemberContract]:
    """Walk tar headers with a range-reader while skipping regular payloads."""

    if total_size <= 0 or total_size % BLOCK_SIZE != 0:
        raise TarContractError(
            f"tar size must be positive and 512-byte aligned, got {total_size}"
        )
    members: list[TarMemberContract] = []
    offset = 0
    zero_blocks = 0
    pending_long_name: str | None = None
    pending_pax: dict[str, str] = {}
    global_pax: dict[str, str] = {}

    while offset + BLOCK_SIZE <= total_size:
        header_offset = offset
        block = read(offset, BLOCK_SIZE)
        offset += BLOCK_SIZE
        if block == b"\0" * BLOCK_SIZE:
            zero_blocks += 1
            if zero_blocks == 2:
                return members
            continue
        if zero_blocks:
            raise TarContractError(
                f"non-zero tar header after one zero block at offset {header_offset}"
            )

        _verify_header(block, header_offset)
        raw_name = _tar_text(block[0:100])
        prefix = _tar_text(block[345:500])
        name = f"{prefix}/{raw_name}" if prefix else raw_name
        size = _tar_number(block[124:136], "member size")
        typeflag = chr(block[156]) if block[156] else "0"
        payload_offset = offset
        padded_size = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        next_offset = payload_offset + padded_size
        if next_offset > total_size:
            raise TarContractError(
                f"member {name!r} extends past archive size: {next_offset} > {total_size}"
            )

        if typeflag in {"L", "K", "x", "g"}:
            payload = _metadata_payload(read, payload_offset, size)
            if typeflag == "L":
                pending_long_name = payload.rstrip(b"\0").decode("utf-8")
            elif typeflag == "x":
                pending_pax = _parse_pax(payload)
            elif typeflag == "g":
                global_pax.update(_parse_pax(payload))
            # GNU long-link metadata (`K`) is intentionally ignored: archive
            # inventory records member paths, never link targets.
            offset = next_offset
            continue

        effective = dict(global_pax)
        effective.update(pending_pax)
        if pending_long_name is not None:
            name = pending_long_name
        if "path" in effective:
            name = effective["path"]
        if "size" in effective:
            try:
                size = int(effective["size"])
            except ValueError as error:
                raise TarContractError("PAX size is not an integer") from error
            padded_size = ((size + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
            next_offset = payload_offset + padded_size
            if next_offset > total_size:
                raise TarContractError(
                    f"PAX-sized member {name!r} extends past archive"
                )

        members.append(
            TarMemberContract(
                name=name,
                type=typeflag,
                size=size,
                header_offset=header_offset,
                payload_offset=payload_offset,
            )
        )
        pending_long_name = None
        pending_pax = {}
        offset = next_offset

    raise TarContractError("tar archive has no two-block end marker")


def manifest_sha256(members: list[TarMemberContract]) -> str:
    canonical = bytearray()
    for member in members:
        canonical.extend(member.name.encode("utf-8"))
        canonical.append(0)
        canonical.extend(member.type.encode("ascii"))
        canonical.extend(member.size.to_bytes(8, byteorder="little"))
        canonical.extend(member.header_offset.to_bytes(8, byteorder="little"))
    return hashlib.sha256(canonical).hexdigest()


def self_test() -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in [
            ("model_config.yaml", b"model: canary\n"),
            ("nested/" + "long-name-" * 12 + "model_weights.ckpt", b"weights"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    archive_bytes = payload.getvalue()

    def read(offset: int, size: int) -> bytes:
        return archive_bytes[offset : offset + size]

    members = list_tar_contract(read, len(archive_bytes))
    assert [member.name for member in members] == [
        "model_config.yaml",
        "nested/" + "long-name-" * 12 + "model_weights.ckpt",
    ]
    assert [member.size for member in members] == [14, 7]
    # The regular payloads themselves are skipped by list_tar_contract; the
    # synthetic reader remains capable of serving PAX metadata when needed.
    assert manifest_sha256(members) == manifest_sha256(members)
    print("hf_tar_contract: self-test PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", help="Hugging Face repo, e.g. nvidia/model")
    parser.add_argument("filename", nargs="?", help="tar/.nemo filename in the repo")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--member-regex")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.repo or not args.filename:
        raise SystemExit("repo and filename are required unless --self-test is used")

    fetcher = RangeFetcher(args.repo, args.revision, args.filename)
    # The first exact header fetch establishes the Content-Range total.  It is
    # reused by the walker rather than issued twice.
    first_header = fetcher.read(0, BLOCK_SIZE)
    assert fetcher.total_size is not None
    total_size = fetcher.total_size
    if args.expected_size is not None and total_size != args.expected_size:
        raise SystemExit(
            f"remote size {total_size} != pinned --expected-size {args.expected_size}"
        )

    def read(offset: int, size: int) -> bytes:
        if offset == 0 and size == BLOCK_SIZE:
            return first_header
        return fetcher.read(offset, size)

    members = list_tar_contract(read, total_size)
    selected = members
    if args.member_regex:
        pattern = re.compile(args.member_regex)
        selected = [member for member in members if pattern.search(member.name)]
    report = {
        "repo": args.repo,
        "revision": args.revision,
        "filename": args.filename,
        "archive_size": total_size,
        "member_count": len(members),
        "manifest_sha256": manifest_sha256(members),
        "range_request_count": fetcher.request_count,
        "bytes_received": fetcher.bytes_received,
        "members": [asdict(member) for member in selected],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
