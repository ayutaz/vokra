#!/usr/bin/env python3
"""Pack precomputed MeloTTS acoustic inputs into ``VKRMELO1`` v1.

This is a dependency-free bridge from the raw little-endian arrays emitted by
an upstream language frontend or parity dumper to the stable container consumed
by ``vokra-cli run --model <melotts.gguf> --input <features.vmf>``. It does not
perform G2P, tokenization, or BERT inference and therefore cannot silently
invent data absent from the five public acoustic GGUFs.

Run through the repository Python policy::

    uv run --project tools/parity python tools/parity/melotts_pack_features.py \
      --variant english --speaker-id 0 \
      --phoneme-ids phoneme_ids.u32 --tones tones.u32 \
      --language-ids language_ids.u32 \
      --bert bert_position_major.f32 \
      --ja-bert ja_bert_position_major.f32 \
      --output features.vmf
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


MAGIC = b"VKRMELO1"
VERSION = 1
HEADER_LEN = 64
BERT_DIMENSION = 1_024
JA_BERT_DIMENSION = 768
VARIANTS = {
    "english": 0,
    "chinese": 1,
    "korean": 2,
    "spanish": 3,
    "japanese": 4,
}


def read_u32(path: Path, label: str) -> tuple[bytes, list[int]]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(
            f"{label}: {path} has {len(data)} bytes; expected a non-empty "
            "little-endian u32 array"
        )
    return data, [value[0] for value in struct.iter_unpack("<I", data)]


def read_f32(path: Path, label: str) -> tuple[bytes, list[float]]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(
            f"{label}: {path} has {len(data)} bytes; expected a non-empty "
            "little-endian f32 array"
        )
    values = [value[0] for value in struct.iter_unpack("<f", data)]
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{index}] is non-finite ({value})")
    return data, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--phoneme-ids", type=Path, required=True)
    parser.add_argument("--tones", type=Path, required=True)
    parser.add_argument("--language-ids", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--ja-bert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.speaker_id <= 0xFFFF_FFFF:
        raise ValueError(f"--speaker-id {args.speaker_id} is outside u32")
    phoneme_bytes, phoneme_ids = read_u32(args.phoneme_ids, "phoneme_ids")
    tone_bytes, tones = read_u32(args.tones, "tones")
    language_bytes, language_ids = read_u32(args.language_ids, "language_ids")
    bert_bytes, bert = read_f32(args.bert, "bert")
    ja_bert_bytes, ja_bert = read_f32(args.ja_bert, "ja_bert")
    sequence_len = len(phoneme_ids)
    if len(tones) != sequence_len or len(language_ids) != sequence_len:
        raise ValueError(
            "phoneme_ids, tones, and language_ids must have the same non-zero length; "
            f"got {sequence_len}, {len(tones)}, {len(language_ids)}"
        )
    if len(bert) != sequence_len * BERT_DIMENSION:
        raise ValueError(
            f"bert has {len(bert)} values; expected sequence_len*1024 = "
            f"{sequence_len * BERT_DIMENSION}"
        )
    if len(ja_bert) != sequence_len * JA_BERT_DIMENSION:
        raise ValueError(
            f"ja_bert has {len(ja_bert)} values; expected sequence_len*768 = "
            f"{sequence_len * JA_BERT_DIMENSION}"
        )
    if sequence_len > 0xFFFF_FFFF:
        raise ValueError(f"sequence length {sequence_len} is outside u32")

    header = struct.pack(
        "<8sHHIIIIII28s",
        MAGIC,
        VERSION,
        HEADER_LEN,
        0,
        VARIANTS[args.variant],
        sequence_len,
        args.speaker_id,
        BERT_DIMENSION,
        JA_BERT_DIMENSION,
        bytes(28),
    )
    if len(header) != HEADER_LEN:
        raise AssertionError(f"internal header length {len(header)} != {HEADER_LEN}")
    args.output.write_bytes(
        header
        + phoneme_bytes
        + tone_bytes
        + language_bytes
        + bert_bytes
        + ja_bert_bytes
    )
    print(
        f"wrote {args.output}: variant={args.variant}, speaker_id={args.speaker_id}, "
        f"sequence_len={sequence_len}, bytes={args.output.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
