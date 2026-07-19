#!/usr/bin/env python3
"""Build a truncated-but-REAL-weights moshiko safetensors (M4 cc-06).

Keeps every tensor except ``transformer.layers.{i >= keep-tm}.*`` and
``depformer.layers.{i >= keep-dt}.*`` (the 7B checkpoint is temporal 32 /
depformer 6). All kept tensor payloads are copied **byte-verbatim** from
the source — only the tensor SET shrinks, so `vokra-convert`'s
shape-driven ``derive()`` sees a self-consistent smaller model (layer
counts are shape-counted; d_model / n_q / dep_q / card / text_card / ffn
hidden all stay the real 7B values). This is the campaign-2 truncation
recipe (2026-07-17, ``truncate_moshiko.py`` scratch tool) promoted into
the repo so the parity-moshi-real workflow's truncated leg can rebuild
the exact artifact: with the same source checkpoint and the same
``--keep-tm/--keep-dt``, the output is deterministic (tensors ordered by
original data offset; compact JSON header padded to 8 bytes).

Why truncation exists at all: the FULL-7B torch-side reference dump
(``moshi_dump.py real``) casts the 14.32 GiB BF16 dict to fp32 —
~43 GiB peak — which no 16 GB machine or ubuntu-latest runner can hold.
The truncated model (tm=2/dt=2 → 1.11 B params) keeps every derived
hparam real while fitting the dump in ~9 GiB. The Vokra side needs no
truncation any more (streaming converter + mmap load, M4 cc-06); the
truncation serves the *reference* side only.

Bounded RAM: header rewrite + chunked byte-range copy (64 MiB chunks).
"""
import argparse
import json
import struct
from pathlib import Path

CHUNK = 64 * 1024 * 1024


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="upstream model.safetensors (read-only)")
    p.add_argument("--dst", type=Path, required=True,
                   help="truncated safetensors to write")
    p.add_argument("--keep-tm", type=int, default=2,
                   help="temporal transformer layers to keep (default 2)")
    p.add_argument("--keep-dt", type=int, default=2,
                   help="depformer layers to keep (default 2)")
    args = p.parse_args()

    def keep(name: str) -> bool:
        for prefix, k in (("transformer.layers.", args.keep_tm),
                          ("depformer.layers.", args.keep_dt)):
            if name.startswith(prefix):
                idx = int(name[len(prefix):].split(".")[0])
                return idx < k
        return True

    with open(args.src, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data_base = 8 + hlen
    meta = header.pop("__metadata__", None)

    kept = {k: v for k, v in header.items() if keep(k)}
    dropped = len(header) - len(kept)

    # Preserve upstream tensor order (by original offset) for determinism.
    order = sorted(kept.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header = {}
    if meta is not None:
        new_header["__metadata__"] = meta
    cursor = 0
    plan = []  # (src_start, length)
    for name, info in order:
        s, e = info["data_offsets"]
        length = e - s
        new_header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                            "data_offsets": [cursor, cursor + length]}
        plan.append((s, length))
        cursor += length

    hjson = json.dumps(new_header, separators=(",", ":")).encode()
    pad = (-(8 + len(hjson))) % 8  # safetensors aligns data to 8 bytes
    hjson += b" " * pad

    with open(args.src, "rb") as src, open(args.dst, "wb") as dst:
        dst.write(struct.pack("<Q", len(hjson)))
        dst.write(hjson)
        for s, length in plan:
            src.seek(data_base + s)
            remaining = length
            while remaining:
                chunk = src.read(min(CHUNK, remaining))
                assert chunk, "short read"
                dst.write(chunk)
                remaining -= len(chunk)

    print(f"kept {len(kept)} tensors ({dropped} dropped), data bytes "
          f"{cursor:,} ({cursor / 2**30:.2f} GiB), out={args.dst}")


if __name__ == "__main__":
    main()
