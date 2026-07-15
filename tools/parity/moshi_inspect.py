#!/usr/bin/env python3
"""Dump the Moshi (kyutai) checkpoint tensor manifest (M4-06 T02).

This is an **offline** inspection tool (FR-LD-05: no Python enters the
runtime). It fetches only the safetensors *header* (tensor name / dtype /
shape) via an HTTP Range request — the multi-GB weight payload is never
downloaded — and writes the manifest JSON that anchors every
`crates/vokra-models/src/moshi/` shape decision (発明禁止: the Rust
implementation transcribes, never invents).

stdlib only (zero new pins — the parity venv is not even required).

Committed fixture
-----------------

``tests/parity/moshi/moshiko_tensor_manifest.json`` (+ ``.sha256``) is the
2026-07-15 dump of::

    https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/model.safetensors

(355 tensors, all BF16 — ADR M4-06 §D2). Re-run this script to refresh:

    python3 tools/parity/moshi_inspect.py \
        https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/model.safetensors \
        tests/parity/moshi/moshiko_tensor_manifest.json

The safetensors format: first 8 bytes = u64 LE header length N, then N
bytes of JSON mapping tensor name -> {dtype, shape, data_offsets}.
"""

import hashlib
import json
import re
import struct
import sys
import urllib.request


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def summarize(manifest: dict) -> None:
    """Prints the collapsed per-pattern shape view used in ADR M4-06 §D2."""
    pats: dict[str, set] = {}
    for name, info in manifest.items():
        if name == "__metadata__":
            continue
        p = re.sub(r"\.\d+\.", ".{N}.", name)
        pats.setdefault(p, set()).add((tuple(info["shape"]), info["dtype"]))
    for p in sorted(pats):
        print(f"  {p} -> {sorted(pats[p])}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    url, out = sys.argv[1], sys.argv[2]
    head = fetch_range(url, 0, 7)
    (n,) = struct.unpack("<Q", head)
    raw = fetch_range(url, 8, 8 + n - 1)
    header = json.loads(raw.decode("utf-8"))
    manifest = {}
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            manifest["__metadata__"] = info
            continue
        manifest[name] = {"dtype": info["dtype"], "shape": info["shape"]}
    body = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(out + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{digest}  {out.rsplit('/', 1)[-1]}\n")
    n_tensors = len([k for k in manifest if k != "__metadata__"])
    print(f"tensors: {n_tensors}, sha256: {digest}", file=sys.stderr)
    summarize(manifest)


if __name__ == "__main__":
    main()
