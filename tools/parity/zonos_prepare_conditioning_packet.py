#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Prepare the bounded Zonos v1 offline conditioning packet.

The preparer owns only the raw typed controls and authenticated phoneme
symbol IDs.  eSpeak/phonemizer runs outside the runtime and must provide the
fixed BOS=2/EOS=3-framed IDs.  The two projected-prefix arrays are retained as
v1 compatibility fields and are deliberately zero-filled; native Zonos
recomputes both prefixes from its bound conditioner tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

MAGIC = b"ZONOSCP1"
VERSION = 1
CODEBOOKS = 9
PREFIX_VALUES = 2048


def reject_symlink_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"path contains a symlink ancestor: {path}")


def require_absent_output(path: Path) -> None:
    reject_symlink_ancestors(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"output already exists or is symlinked: {path}")
    parent = path.parent
    while not parent.exists():
        if parent == parent.parent:
            raise ValueError(f"output has no existing parent: {path}")
        parent = parent.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"output parent is not a real directory: {parent}")


def load_json(value: str, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be JSON") from error


def vector(value: str, length: int, label: str) -> list[float]:
    raw = load_json(value, label)
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    output = [float(item) for item in raw]
    if any(not math.isfinite(item) for item in output):
        raise ValueError(f"{label} must contain finite values")
    return output


def packet(args: argparse.Namespace) -> tuple[bytes, str]:
    phoneme_ids = load_json(args.phoneme_ids, "--phoneme-ids")
    if (
        not isinstance(phoneme_ids, list)
        or len(phoneme_ids) < 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in phoneme_ids)
        or phoneme_ids[0] != 2
        or phoneme_ids[-1] != 3
        or any(item == 0 or item in (2, 3) or not 0 <= item < 189 for item in phoneme_ids[1:-1])
    ):
        raise ValueError("phoneme IDs must be BOS=2, interior UNK/symbol, EOS=3")
    speaker = vector(args.speaker, 128, "--speaker")
    emotion = vector(args.emotion, 8, "--emotion")
    if any(value < 0 for value in emotion) or sum(emotion) <= 0:
        raise ValueError("emotion must be finite, non-negative, and non-zero")
    emotion = [value / sum(emotion) for value in emotion]
    fmax, pitch_std, speaking_rate = (float(value) for value in (args.fmax, args.pitch_std, args.speaking_rate))
    if not 0 <= fmax <= 24_000 or not 0 <= pitch_std <= 400 or not 0 <= speaking_rate <= 40:
        raise ValueError("scalar controls are outside the fixed source domain")
    if not -1 <= args.language_id <= 126:
        raise ValueError("--language-id must be in -1..126")

    body = bytearray(MAGIC)
    body.extend(struct.pack("<II", VERSION, len(phoneme_ids)))
    body.extend(struct.pack(f"<{len(speaker)}f", *speaker))
    body.extend(struct.pack(f"<{len(emotion)}f", *emotion))
    body.extend(struct.pack("<fff", fmax, pitch_std, speaking_rate))
    body.extend(struct.pack("<iIII", args.language_id, CODEBOOKS, 0, PREFIX_VALUES))
    body.extend(struct.pack("<I", PREFIX_VALUES))
    digest_offset = len(body)
    body.extend(b"\0" * 32)
    body.extend(struct.pack(f"<{len(phoneme_ids)}I", *phoneme_ids))
    body.extend(struct.pack("<f", 0.0) * (PREFIX_VALUES * 2))
    digest = hashlib.sha256(body[:digest_offset] + body[digest_offset + 32 :]).digest()
    body[digest_offset : digest_offset + 32] = digest
    return bytes(body), digest.hex()


def self_test() -> None:
    try:
        load_json('{"x":1,"x":2}', "duplicate")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys must fail")
    args = argparse.Namespace(
        phoneme_ids="[2,1,188,3]",
        speaker=json.dumps([0.0, 0.125] + [0.0] * 126),
        emotion="[1,2,3,4,5,6,7,8]",
        fmax=22_050.0,
        pitch_std=20.0,
        speaking_rate=15.0,
        language_id=0,
    )
    value, digest = packet(args)
    assert value.startswith(MAGIC) and len(digest) == 64
    digest_offset = 592
    assert value[digest_offset : digest_offset + 32] == hashlib.sha256(
        value[:digest_offset] + value[digest_offset + 32 :]
    ).digest()
    assert struct.unpack_from("<f", value, 16 + 4)[0] == 0.125
    normalized_emotion = struct.unpack_from("<8f", value, 16 + 128 * 4)
    assert normalized_emotion[0] < normalized_emotion[-1]
    assert abs(sum(normalized_emotion) - 1.0) < 1.0e-6
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="zonos-conditioner-selftest-") as directory:
        directory = str(Path(directory).resolve())
        output_root = Path(directory) / "outputs"
        output_root.mkdir()
        require_absent_output(output_root / "packet.bin")
        (output_root / "existing.bin").write_bytes(b"x")
        try:
            require_absent_output(output_root / "existing.bin")
        except ValueError:
            pass
        else:
            raise AssertionError("existing output must fail closed")
        (output_root / "real").mkdir()
        (output_root / "link").symlink_to(output_root / "real", target_is_directory=True)
        try:
            require_absent_output(output_root / "link" / "new.bin")
        except ValueError:
            pass
        else:
            raise AssertionError("symlink output ancestors must fail closed")
    tampered = bytearray(value)
    tampered[-1] ^= 1
    assert tampered[digest_offset : digest_offset + 32] != hashlib.sha256(
        tampered[:digest_offset] + tampered[digest_offset + 32 :]
    ).digest()
    print("zonos_prepare_conditioning_packet.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phoneme-ids")
    parser.add_argument("--speaker")
    parser.add_argument("--emotion")
    parser.add_argument("--fmax", type=float, default=22_050.0)
    parser.add_argument("--pitch-std", type=float, default=20.0)
    parser.add_argument("--speaking-rate", type=float, default=15.0)
    parser.add_argument("--language-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.phoneme_ids or not args.speaker or not args.emotion or args.output is None:
        parser.error("--phoneme-ids, --speaker, --emotion, and --output are required")
    try:
        require_absent_output(args.output)
        value, digest = packet(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(value)
        print(json.dumps({
            "packet": args.output.name,
            "packet_sha256": hashlib.sha256(value).hexdigest(),
            "content_digest": digest,
            "projected_prefix": "compatibility_only_zero_filled",
            "runtime_recomputes_prefix": True,
        }, sort_keys=True))
    except (OSError, ValueError, struct.error) as error:
        print(f"zonos conditioning packet blocker: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
