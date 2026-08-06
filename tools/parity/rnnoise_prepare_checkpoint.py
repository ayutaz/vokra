#!/usr/bin/env python3
"""Flatten Xiph RNNoise v0.2's `weights_blob_9.bin` release asset into a
safetensors file suitable for `vokra-cli convert --model rnnoise-v0.2`
(coverage-audit 2026-08-03 Wave A ticket).

Upstream ships RNNoise as a **compact C-array blob** (~90 KB
`weights_blob_9.bin` bundled in the v0.2 GitHub Release tarball at
`github.com/xiph/rnnoise/releases/tag/v0.2`, additionally embedded as
`src/rnn_data.c` in the source tree). Vokra's `vokra-convert` reads only
safetensors (keeps the runtime zero-dep — no C parser, no pickle
deserialization, NFR-DS-02), so this Python-side prep tool flattens the
blob to safetensors offline. **No C / Python enters the runtime** — the
prep script runs on the operator's box, once per checkpoint.

Same "prep to safetensors" contract as sibling converters that consume
non-safetensors upstream releases (mirror of
`tools/parity/dfn3_prepare_checkpoint.py` for DeepFilterNet3's
`.ckpt.best` torch pickle and `tools/parity/dac_prepare_checkpoint.py`
for DAC's `.pth`).

# License posture

Reads BSD-3-Clause data (`github.com/xiph/rnnoise/blob/main/COPYING`,
standard three-clause BSD, `Copyright (c) 2017-2024 Mozilla / Xiph.Org /
Jean-Marc Valin`) with pure-Python `numpy.frombuffer` (BSD-3) + writes
via `safetensors.save_file` (Apache-2.0). No AGPL / GPL source is read
or referenced.

# Loud-partial posture (real per-layer split is deferred to owner)

The Xiph on-disk layout of `weights_blob_9.bin` is a sequence of
**int8-quantized** dense / GRU layer tensors plus per-layer float32 scale
factors, encoded in the order Xiph's `training/dump_rnnoise_weights_blob.c`
writes them:

    input_dense          (42 -> 24)   Dense    int8 kernel + f32 bias
    vad_gru              (24 -> 24)   GRU      int8 kernel + int8 recurrent + f32 bias (3 gates)
    noise_gru            (24 -> 48)   GRU      (same layout)
    denoise_gru          (24 -> 96)   GRU      (same layout)
    denoise_output       (96 -> 22)   Dense    int8 kernel + f32 bias
    vad_output           (24 -> 1)    Dense    int8 kernel + f32 bias

Emitting each of the above as a named f32 safetensors tensor requires
walking Xiph's exact struct layout (per-layer scale factor, gate
ordering, sign convention), a task that must be pinned to a Xiph
reference-C forward for parity. That work is the **owner** deliverable
(`docs/license-audit.md` §3.1 sign-off queue for RNNoise v0.2) — the
matching runtime module lives at `crates/vokra-models/src/rnnoise/` and
is deliberately deferred until per-layer parity against the Xiph C
forward can be measured.

Until that lands, this tool emits the blob as a **single opaque tensor**
(`rnnoise.weights_blob_f32`) — a flat f32 array holding the raw bytes
reinterpreted as little-endian float32 (padded with zero-bytes if the
blob length is not a multiple of 4). This is a **placeholder**: the
resulting GGUF is intentionally not loadable by any runtime forward
(the future `RnnoiseWeights::from_gguf` walks per-layer tensor names,
not the opaque blob), but it does exercise the converter's provenance /
category / license stamping path so the publish pipe (`publish-one.sh`)
can validate its 5 gates against the artifact today. A big
`vokra.rnnoise.prep_status = "opaque-blob-placeholder"` marker on the
safetensors header (and by extension the GGUF) makes this state
unmissable to any downstream reader.

# Usage

    uv run python tools/parity/rnnoise_prepare_checkpoint.py \\
        --input ~/checkpoints/rnnoise-v0.2/rnnoise-0.2/models/weights_blob_9.bin \\
        --output ~/checkpoints/rnnoise-v0.2/model.safetensors

Then:

    ./target/release/vokra-cli convert --model rnnoise \\
        --input ~/checkpoints/rnnoise-v0.2/model.safetensors \\
        --output ~/gguf/rnnoise-v0.2.gguf

# FR-EX-08 loud-error posture

- Missing / non-file `--input` → refuse with a clear message.
- `--output` exists → refuse (never silently overwrite).
- Zero-byte `--input` → refuse (a real weights blob is ~90 KB).
- numpy / safetensors import fail → exit with `uv add <pkg>` hint.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LOG_PREFIX = "rnnoise_prepare_checkpoint:"
# Placeholder tensor name — the future per-layer split walks a different
# name schema (`input_dense.kernel`, `vad_gru.recurrent`, ...); the
# `_f32` suffix + the `prep_status` marker on the GGUF prevent a caller
# from mistaking this placeholder for a real per-layer emission.
PLACEHOLDER_TENSOR_NAME = "rnnoise.weights_blob_f32"
# Minimum blob size that could plausibly be a real RNNoise weight
# release. The v0.2 asset is ~90 KB; anything under 1 KB is either a
# truncated download or the wrong file (e.g. a README).
MIN_PLAUSIBLE_BLOB_BYTES = 1024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_blob_as_f32(blob_bytes: bytes) -> "tuple[object, int]":
    """Reinterpret `blob_bytes` as a flat little-endian f32 array.

    Pads with zero-bytes to the nearest 4-byte boundary so the resulting
    ndarray dtype is a whole number of f32 elements (numpy's own
    `frombuffer` requires that; a partial trailing element would be
    silently dropped otherwise). Returns `(ndarray, pad_bytes)`.
    """
    try:
        import numpy as np
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep numpy ({exc}); install with "
            "`uv add numpy` in tools/parity/"
        )

    pad_bytes = (4 - (len(blob_bytes) % 4)) % 4
    padded = blob_bytes + (b"\x00" * pad_bytes)
    arr = np.frombuffer(padded, dtype="<f4").copy()  # copy: numpy view is read-only
    return arr, pad_bytes


def write_safetensors(arr: "object", pad_bytes: int, output: Path) -> None:
    """Emit `arr` as a single named f32 tensor under
    `PLACEHOLDER_TENSOR_NAME`, plus a `__metadata__` header pinning the
    prep-tool provenance so the downstream converter can loudly refuse a
    placeholder blob if it ever tries to bind it against per-layer
    tensor names.

    `pad_bytes` is recorded so a future per-layer decomposition script
    can recover the original blob length exactly (`len(padded_arr)*4 -
    pad_bytes`) rather than having to look it up from the original
    `weights_blob_N.bin` filename.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        sys.exit(
            f"{LOG_PREFIX} missing Python dep safetensors ({exc}); install with "
            "`uv add safetensors` in tools/parity/"
        )

    metadata = {
        # The `prep_status` marker is the single-source-of-truth flag the
        # downstream runtime uses to refuse a placeholder blob (mirror of
        # the `dfn3_prepare_checkpoint` provenance header).
        "prep_status": "opaque-blob-placeholder",
        "prep_tool": "rnnoise_prepare_checkpoint.py",
        "prep_tool_version": "0.1.0",
        "prep_pad_bytes": str(pad_bytes),
        "upstream_url": "https://github.com/xiph/rnnoise/releases/tag/v0.2",
        "license_spdx": "bsd-3-clause",
        # The follow-up per-layer split is owner-tracked; the tensor
        # names that decomposition will emit are documented in the
        # module docstring, not here (this metadata is header-cheap).
        "next_step": (
            "Real per-layer split against Xiph reference C forward "
            "is the owner deliverable — see rnnoise_prepare_checkpoint.py "
            "module docstring for the target tensor-name schema."
        ),
    }
    save_file({PLACEHOLDER_TENSOR_NAME: arr}, str(output), metadata=metadata)


def prepare(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        sys.exit(f"{LOG_PREFIX} --input {input_path} does not exist.")
    if not input_path.is_file():
        sys.exit(f"{LOG_PREFIX} --input {input_path} is not a regular file.")
    if output_path.exists():
        sys.exit(
            f"{LOG_PREFIX} refusing to overwrite existing --output {output_path}. "
            "Remove it first or pick a different --output path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    blob = input_path.read_bytes()
    if len(blob) < MIN_PLAUSIBLE_BLOB_BYTES:
        sys.exit(
            f"{LOG_PREFIX} --input {input_path} is only {len(blob)} bytes — a "
            f"real RNNoise v0.2 weights blob is ~90 KB. Re-download the release "
            "tarball (github.com/xiph/rnnoise/releases/tag/v0.2) and try again."
        )

    print(
        f"{LOG_PREFIX} reading {input_path} ({len(blob):,} bytes, sha256 "
        f"{hashlib.sha256(blob).hexdigest()})"
    )

    arr, pad_bytes = flatten_blob_as_f32(blob)
    print(
        f"{LOG_PREFIX} emitting placeholder tensor {PLACEHOLDER_TENSOR_NAME!r} "
        f"({arr.shape[0]:,} f32 elements, {pad_bytes} pad bytes)"
    )
    print(
        f"{LOG_PREFIX} PLACEHOLDER: the resulting safetensors is an opaque-blob "
        "stand-in — see the module docstring for the real per-layer split "
        "(owner deliverable, tracked in docs/license-audit.md §3.1)."
    )

    write_safetensors(arr, pad_bytes, output_path)
    print(
        f"{LOG_PREFIX} wrote {output_path} "
        f"(sha256 {sha256_of(output_path)})"
    )
    print(f"{LOG_PREFIX} done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten Xiph RNNoise v0.2 weights_blob_N.bin into safetensors "
            "for vokra-cli convert --model rnnoise-v0.2. Emits an opaque-blob "
            "placeholder tensor today; real per-layer decomposition is the "
            "owner deliverable — see the module docstring."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Local path to weights_blob_9.bin from the v0.2 release tarball "
            "(github.com/xiph/rnnoise/releases/tag/v0.2)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Local .safetensors path to write. Refuses to overwrite an "
            "existing file. Parent directory is created if absent."
        ),
    )
    args = parser.parse_args()
    return prepare(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
