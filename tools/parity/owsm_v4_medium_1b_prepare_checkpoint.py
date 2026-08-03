#!/usr/bin/env python3
"""Prepare an ESPnet OWSM v4 medium 1B checkpoint → safetensors
(coverage-audit 2026-08-03 Wave B ticket).

Offline sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). The upstream ``espnet/owsm_v4_medium_1B`` HF release ships
several artifacts, including:

* ``owsm_v4_medium.pth`` — the raw ESPnet torch state-dict (flat
  ``OrderedDict``) — the primary distribution format.
* Optional ``*.safetensors`` — a byte-parallel safetensors mirror
  ESPnet has started publishing alongside newer OWSM releases.

The Rust converter
(``crates/vokra-convert/src/models/owsm_v4_medium_1b.rs``) consumes
safetensors only, so this script bridges the two:

* When the input directory already contains a ``*.safetensors`` file
  it is passed through unchanged (a plain copy for determinism, so
  the downstream converter reads from a stable path).
* When only ``.pth`` / ``.pt`` files are present the state-dict is
  loaded with ``torch.load(..., weights_only=True)`` (the ESPnet
  release loads cleanly under the safe loader; a checkpoint that
  does not is refused rather than falling back to unsafe unpickling)
  and re-emitted as safetensors — dotted upstream tensor names
  preserved verbatim so ``OwsmV4Weights::from_gguf`` (a future
  runtime addition) can walk the same names.

Non-F32 / F16 / BF16 tensors are dropped explicitly (each reported)
— the vokra-core safetensors reader is float-only by design and any
training-only INT counter has no inference role. A silent drop would
break FR-EX-08, so every skip lands in the log.

Shared-tensor deduplication (safetensors requires unique storage
per tensor; ESPnet's tied embedding / output-projection can share
storage under torch's serialization) follows the pattern from
`memory:[[reference-safetensors-shared-tensor-dedup]]` — the first
name seen wins; every alias is recorded in a ``shared_pairs.json``
sidecar for audit trail.

Fails loudly on any anomaly (non-tensor entry, unexpected dtype)
rather than masking it — FR-EX-08 posture.

# Usage

::

    uv run python tools/parity/owsm_v4_medium_1b_prepare_checkpoint.py \\
        --input-dir ~/models/owsm-v4-medium-1b \\
        --output ~/models/owsm-v4-medium-1b/model.safetensors

Then:

::

    vokra-cli convert --model owsm-v4-medium-1b \\
        --input ~/models/owsm-v4-medium-1b/model.safetensors \\
        --output ~/gguf/owsm-v4-medium-1b.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from collections import OrderedDict
from pathlib import Path


# Every dtype the ESPnet OWSM release is known to ship. The runtime-side
# reader accepts F32 / F16 / BF16; INT counters (BatchNorm / step counter)
# are dropped explicitly.
DTYPE_MAP = {
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
}

# Dropped explicitly (with a log line each) — no inference role.
DROP_INT_DTYPES = {
    "torch.int8",
    "torch.int16",
    "torch.int32",
    "torch.int64",
    "torch.uint8",
    "torch.uint16",
    "torch.uint32",
    "torch.uint64",
    "torch.bool",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_safetensors_in_dir(input_dir: Path) -> Path | None:
    """Return the first .safetensors under ``input_dir`` (single-file
    convention), or None."""
    hits = sorted(input_dir.rglob("*.safetensors"))
    if not hits:
        return None
    # A single top-level safetensors is what the fast-path pass-through
    # is designed for; multi-file sharded releases are out of scope for
    # this skeleton and would demand an index.json walk.
    if len(hits) > 1:
        print(
            f"; NOTE: {len(hits)} .safetensors files found in {input_dir}; "
            f"passing through the first: {hits[0].name}",
            file=sys.stderr,
        )
    return hits[0]


def _find_pth_in_dir(input_dir: Path) -> Path | None:
    """Return the first .pth or .pt under ``input_dir``, or None."""
    for pattern in ("*.pth", "*.pt"):
        hits = sorted(input_dir.rglob(pattern))
        if hits:
            if len(hits) > 1:
                print(
                    f"; NOTE: {len(hits)} {pattern} files found; using first: "
                    f"{hits[0].name}",
                    file=sys.stderr,
                )
            return hits[0]
    return None


def _write_safetensors(
    out_path: Path,
    tensors: "OrderedDict[str, tuple[str, list[int], bytes]]",
) -> None:
    """Minimal safetensors writer (stdlib only): 8-byte LE header
    length + JSON header + contiguous little-endian tensor data.

    Mirrors ``tools/parity/dfn3_prepare_checkpoint.py::write_safetensors``.
    """
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _bridge_pth_to_safetensors(
    pth_path: Path, out_path: Path, shared_pairs_sidecar: Path
) -> None:
    """Load a torch state-dict pickle and write it as safetensors.

    Only F32 / F16 / BF16 tensors are written; INT counters are
    dropped explicitly (each reported). Any other dtype is a hard
    error, never a silent drop (FR-EX-08 posture).

    Shared-tensor deduplication follows the safetensors requirement
    (unique storage per tensor); every alias is recorded in
    ``shared_pairs_sidecar`` for audit trail.
    """
    import torch  # heavy import, deferred to the .pth path

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if not isinstance(state, (dict, OrderedDict)):
        raise SystemExit(
            f"checkpoint top level is {type(state).__name__}, expected a flat state dict"
        )

    out: "OrderedDict[str, tuple[str, list[int], bytes]]" = OrderedDict()
    seen_ptrs: dict[int, str] = {}
    shared_pairs: list[tuple[str, str]] = []
    n_written = 0
    n_dropped_int = 0
    n_shared = 0

    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise SystemExit(
                f"entry {name!r} is {type(t).__name__}, expected torch.Tensor"
            )
        dtype_str = str(t.dtype)
        if dtype_str in DROP_INT_DTYPES:
            print(f"; DROP int tensor {name} (dtype={dtype_str}, shape={list(t.shape)})")
            n_dropped_int += 1
            continue
        if dtype_str not in DTYPE_MAP:
            raise SystemExit(
                f"unsupported dtype {dtype_str} for tensor {name!r} — "
                f"expected one of {sorted(DTYPE_MAP)} or a droppable INT type"
            )
        # Safetensors requires unique storage per tensor. Torch's tied
        # weights / shared embeddings can point at the same buffer;
        # clone-per-name is the standard bridge (memory:
        # [[reference-safetensors-shared-tensor-dedup]]).
        try:
            ptr = t.untyped_storage().data_ptr()
        except AttributeError:
            # older torch shim
            ptr = t.storage().data_ptr()
        if ptr in seen_ptrs:
            first = seen_ptrs[ptr]
            shared_pairs.append((first, name))
            n_shared += 1
            # Clone so the alias gets its own storage in the output.
            source = t.detach().clone().contiguous().cpu()
        else:
            seen_ptrs[ptr] = name
            source = t.detach().contiguous().cpu()

        # BF16 has no native numpy dtype in older numpy; go through raw
        # bytes via torch tensor to stay dependency-light.
        data = bytes(source.view(torch.uint8).numpy().tobytes()) \
            if dtype_str == "torch.bfloat16" \
            else source.numpy().tobytes()

        out[name] = (DTYPE_MAP[dtype_str], list(source.shape), data)
        n_written += 1

    _write_safetensors(out_path, out)
    if shared_pairs:
        shared_pairs_sidecar.write_text(
            json.dumps({"shared_pairs": shared_pairs}, indent=2)
        )
    print(
        f"; owsm-v4-medium-1b prep: {n_written} tensors written, "
        f"{n_dropped_int} INT counters dropped, {n_shared} shared aliases"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare ESPnet OWSM v4 medium 1B checkpoint for vokra-convert."
    )
    ap.add_argument(
        "--input-dir",
        required=False,
        help="Directory containing the downloaded ESPnet checkpoint (looks for "
        "*.safetensors first, then *.pth / *.pt).",
    )
    ap.add_argument(
        "--input",
        required=False,
        help="Path to a single .safetensors or .pth / .pt file (bypasses "
        "--input-dir discovery).",
    )
    ap.add_argument("--output", required=True, help="Output .safetensors path.")
    args = ap.parse_args()

    if not args.input_dir and not args.input:
        raise SystemExit("must pass --input-dir or --input")

    if args.input:
        src = Path(args.input)
        if not src.is_file():
            raise SystemExit(f"--input {src} not found")
        if src.suffix == ".safetensors":
            safetensors_src = src
            pth_src: Path | None = None
        elif src.suffix in {".pth", ".pt"}:
            safetensors_src = None
            pth_src = src
        else:
            raise SystemExit(f"--input {src} has unsupported extension {src.suffix}")
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise SystemExit(f"--input-dir {input_dir} not found")
        safetensors_src = _find_safetensors_in_dir(input_dir)
        pth_src = None if safetensors_src else _find_pth_in_dir(input_dir)

    out_path = Path(args.output)
    if safetensors_src is not None:
        # Fast path: the HF release already ships safetensors.
        # Deterministic copy so the downstream converter can rely on a
        # stable output path even if the caller re-runs prep.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.resolve() == safetensors_src.resolve():
            # Already in place; nothing to do (idempotent).
            pass
        else:
            shutil.copyfile(safetensors_src, out_path)
        print(
            f"; owsm-v4-medium-1b prep: pass-through safetensors "
            f"{safetensors_src} → {out_path} (sha256 {_sha256_file(out_path)[:16]}…)"
        )
    elif pth_src is not None:
        shared_pairs = out_path.with_suffix(out_path.suffix + ".shared_pairs.json")
        _bridge_pth_to_safetensors(pth_src, out_path, shared_pairs)
        print(
            f"; owsm-v4-medium-1b prep: bridged {pth_src} → {out_path} "
            f"(sha256 {_sha256_file(out_path)[:16]}…)"
        )
    else:
        raise SystemExit(
            "no safetensors or .pth / .pt found under the input path. "
            "Ensure the HF download preserved the checkpoint (do not pass "
            "--exclude '*.safetensors' when the safetensors variant exists)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
