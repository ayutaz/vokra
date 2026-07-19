#!/usr/bin/env python3
"""Dump FSQ codec family reference fixtures — WavTokenizer / X-Codec 2
(M4-16 T10/T11).

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled
into the runtime). It writes the committed fixtures under
``tests/parity/fsq/{wavtokenizer,xcodec2}/`` that the Rust parity test
(``crates/vokra-ops/tests/parity_fsq_codec.rs``) compares against
(NFR-QL-01 design-wide FP32 ``atol = 0.01``; the measured per-fixture gaps
are far tighter — recorded in each manifest). CI never runs Python — the
fixtures are committed (kaldi_fbank / M4-04 RVQ precedent).

**No pretrained weight is downloaded or used** — both fixtures are
synthetic (fixed seed). Real-checkpoint parity is the flip-the-switch
harness of the future model-integration WPs (ADR M4-16 §D-g). Follows the
``mimi_dump.py`` (M4-04 §D-j) manifest / writer conventions; kept a
separate file because the FSQ family is a separate subgraph from RVQ
(FR-OP-31 vs FR-OP-30) and must not grow entangled with the RVQ dump paths.

Subcommands
-----------

``wavtokenizer``
    Reference = ``torch.nn.functional.embedding(codes, table)`` — the
    **verbatim call** WavTokenizer's decode reduces to
    (``encoder/quantization/core_vq.py``: ``EuclideanCodebook.dequantize =
    F.embedding(embed_ind, self.embed)``; released configs are
    ``num_quantizers: 1`` with an Identity ``project_out``, so the n_q=1
    residual loop is a single gather — ADR M4-16 §D-c). The upstream repo
    is not pip-packaged, so the fixture drives the exact torch API rather
    than an installed package; the manifest records this honestly.

    - ``codebook_table_sliced.f32`` — ``[rows, d_model]`` seeded
      ``torch.randn`` table (sliced vs the released 4096 × 512; a gather is
      complete op semantics as long as codes < rows).
    - ``codes.u32`` — ``[time]`` fixed-seed codes in ``[0, rows)``
      (single codebook — one code per timestep, FSQ-family layout).
    - ``decoded_features.f32`` — ``[time, d_model]`` reference gather.
    - ``manifest.txt`` — shapes / seed / sha256 / versions / honest notes.

``xcodec2``
    Reference = ``vector_quantize_pytorch.ResidualFSQ`` at the **exact
    xcodec2==0.1.5 pin (vector-quantize-pytorch==1.17.8)** driven through
    the same public API ``modeling_xcodec2.py::decode_code`` calls:
    ``ResidualFSQ(dim=d_model, levels=[4]*8, num_quantizers=1)
    .get_output_from_indices(indices)`` with a seeded random
    ``project_out`` Linear. The **levels tuple is the real [4; 8]**
    (effective vocab 4^8 = 65536 — FR-OP-31's "65k+"); only ``d_model`` is
    sliced (64 vs the released 2048) to keep the fixture small.

    - ``levels.u32`` — the levels tuple (machine-checked by the Rust test
      so a drifted regeneration cannot silently change the attrs).
    - ``out_proj_weight.f32`` — ``[d_model, n_dims]`` row-major (torch
      ``nn.Linear.weight`` layout ``[out, in]``, dumped verbatim).
    - ``out_proj_bias.f32`` — ``[d_model]``.
    - ``codes.u32`` — ``[time]`` fixed-seed codes in ``[0, 65536)``
      (guaranteed to reach the top quartile so the high mixed-radix dims
      are exercised).
    - ``decoded_features.f32`` — ``[time, d_model]`` reference decode.
    - ``manifest.txt``.

    ``--oracle numpy`` replaces the torch reference with a numpy float32
    transcription of the pinned formula (basis cumprod → per-dim level →
    ``(l - half)/half`` → ``W @ code + b``) for environments without the
    pins; the manifest records which oracle generated the fixture
    (fabricated pass 禁止 — the committed fixture uses the real pin).

Fixture filenames deliberately use ``.f32`` / ``.u32`` / ``.txt`` only —
never the weight extensions ``.safetensors/.gguf/.pth/.bin`` — so the
compliance scanners stay inert on them (asserted by the Rust test).

Usage
-----

::

    venv/bin/python tools/parity/fsq_dump.py wavtokenizer \\
        --out tests/parity/fsq/wavtokenizer --seed 0 --time 32 --rows 256 --d-model 64

    venv/bin/python tools/parity/fsq_dump.py xcodec2 \\
        --out tests/parity/fsq/xcodec2 --seed 0 --time 32 --d-model 64
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# The released X-Codec 2 levels tuple — vq/codec_decoder_vocos.py
# `ResidualFSQ(dim=vq_dim, levels=[4, 4, 4, 4, 4,4,4,4], num_quantizers=1)`
# (ADR M4-16 §D-c; verified 2026-07-15).
XCODEC2_LEVELS = [4, 4, 4, 4, 4, 4, 4, 4]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_f32(path: Path, arr) -> None:
    import numpy as np

    np.ascontiguousarray(arr, dtype=np.float32).astype("<f4").tofile(path)


def _write_u32(path: Path, arr) -> None:
    import numpy as np

    np.ascontiguousarray(arr, dtype=np.uint32).astype("<u4").tofile(path)


def _versions() -> str:
    import numpy as np

    parts = [f"numpy=={np.__version__}"]
    try:
        import torch

        parts.append(f"torch=={torch.__version__}")
    except ImportError:
        parts.append("torch==(absent)")
    try:
        from importlib.metadata import version

        parts.append(f"vector-quantize-pytorch=={version('vector-quantize-pytorch')}")
    except Exception:  # noqa: BLE001 — best-effort version report
        parts.append("vector-quantize-pytorch==(absent)")
    return " ".join(parts)


def _manifest(out_dir: Path, lines: list[str], files: list[str]) -> None:
    for name in files:
        lines.append(f"sha256 {name} {_sha256(out_dir / name)}")
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# wavtokenizer — single-codebook gather reference
# ---------------------------------------------------------------------------


def dump_wavtokenizer(args: argparse.Namespace) -> None:
    import numpy as np

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    table = rng.standard_normal((args.rows, args.d_model)).astype(np.float32)
    codes = rng.integers(0, args.rows, size=(args.time,), dtype=np.uint32)
    # Pin the boundary rows so the extreme indices are always exercised.
    codes[0] = 0
    codes[-1] = args.rows - 1

    if args.oracle == "torch":
        import torch

        decoded = (
            torch.nn.functional.embedding(
                torch.from_numpy(codes.astype(np.int64)), torch.from_numpy(table)
            )
            .numpy()
            .astype(np.float32)
        )
        oracle_note = (
            "reference: torch.nn.functional.embedding — the verbatim call in WavTokenizer "
            "encoder/quantization/core_vq.py EuclideanCodebook.dequantize (n_q=1, "
            "project_out=Identity; upstream repo is not pip-packaged so the exact torch API "
            "is driven directly)"
        )
    else:
        decoded = table[codes.astype(np.int64)]
        oracle_note = (
            "reference: numpy float32 gather (--oracle numpy fallback; formula-equivalent "
            "to torch F.embedding — regenerate with the torch oracle before committing)"
        )

    _write_f32(out_dir / "codebook_table_sliced.f32", table)
    _write_u32(out_dir / "codes.u32", codes)
    _write_f32(out_dir / "decoded_features.f32", decoded)

    _manifest(
        out_dir,
        [
            "M4-16 T10 WavTokenizer VQ reference fixture (SYNTHETIC weights)",
            oracle_note,
            f"versions: {_versions()}",
            "canonical released shape: vocab_size=4096 (vq_bins), d_model=512 (dimension) "
            "— fixture is sliced (gather semantics complete for codes < rows); "
            "num_quantizers=1 per released configs (ADR M4-16 §D-c)",
            "weights: NONE downloaded / used — table is seeded standard_normal "
            "(real-checkpoint parity = flip-the-switch, model-integration WP)",
            f"fixture shapes: codebook_table_sliced [({args.rows}, {args.d_model})]; "
            f"codes [({args.time},)] u32; decoded_features [({args.time}, {args.d_model})]",
            f"seed: {args.seed}; codes in [0, {args.rows}) with boundary rows pinned",
        ],
        ["codebook_table_sliced.f32", "codes.u32", "decoded_features.f32"],
    )


# ---------------------------------------------------------------------------
# xcodec2 — FSQ dequant + out-projection reference (pin 1.17.8)
# ---------------------------------------------------------------------------


def _numpy_fsq_reference(codes, levels, weight, bias):
    """Numpy float32 transcription of the pinned 1.17.8 decode formula
    (finite_scalar_quantization.py `indices_to_level_indices` +
    `_scale_and_shift_inverse`; residual_fsq.py `get_output_from_indices`
    with num_quantizers=1 → scale=(L−1)^0=1 → project_out)."""
    import numpy as np

    time = codes.shape[0]
    n_dims = len(levels)
    grid = np.zeros((time, n_dims), dtype=np.float32)
    for t in range(time):
        rem = int(codes[t])
        for d, level in enumerate(levels):
            level_index = rem % level
            rem //= level
            half = level // 2
            grid[t, d] = np.float32(level_index - half) / np.float32(half)
    return (grid @ weight.T + bias).astype(np.float32)


def dump_xcodec2(args: argparse.Namespace) -> None:
    import numpy as np

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = list(XCODEC2_LEVELS)
    n_dims = len(levels)
    vocab = 1
    for level in levels:
        vocab *= level

    rng = np.random.default_rng(args.seed)
    weight = rng.standard_normal((args.d_model, n_dims)).astype(np.float32)
    bias = rng.standard_normal((args.d_model,)).astype(np.float32)
    codes = rng.integers(0, vocab, size=(args.time,), dtype=np.uint32)
    # Pin the boundary codes so index 0 and the top of the 65536 range are
    # always exercised (the Rust test asserts the top quartile is reached).
    codes[0] = 0
    codes[-1] = vocab - 1

    if args.oracle == "torch":
        import torch
        from vector_quantize_pytorch import ResidualFSQ

        quantizer = ResidualFSQ(dim=args.d_model, levels=levels, num_quantizers=1)
        with torch.no_grad():
            quantizer.project_out.weight.copy_(torch.from_numpy(weight))
            quantizer.project_out.bias.copy_(torch.from_numpy(bias))
        quantizer = quantizer.eval()
        # Same public API modeling_xcodec2.py::decode_code drives:
        # indices shaped [batch, time, num_quantizers].
        indices = torch.from_numpy(codes.astype(np.int64)).reshape(1, args.time, 1)
        with torch.no_grad():
            decoded = quantizer.get_output_from_indices(indices)
        decoded = decoded.squeeze(0).numpy().astype(np.float32)
        oracle_note = (
            "reference: vector_quantize_pytorch.ResidualFSQ.get_output_from_indices — the "
            "exact module + public API modeling_xcodec2.py::decode_code calls, at the "
            "xcodec2==0.1.5 pin vector-quantize-pytorch==1.17.8 (MIT reference CODE; "
            "project_out weights are seeded synthetic)"
        )
    else:
        decoded = _numpy_fsq_reference(codes, levels, weight, bias)
        oracle_note = (
            "reference: numpy float32 transcription of the pinned 1.17.8 formula "
            "(--oracle numpy fallback — regenerate with the torch oracle before committing)"
        )

    _write_u32(out_dir / "levels.u32", np.asarray(levels, dtype=np.uint32))
    _write_f32(out_dir / "out_proj_weight.f32", weight)
    _write_f32(out_dir / "out_proj_bias.f32", bias)
    _write_u32(out_dir / "codes.u32", codes)
    _write_f32(out_dir / "decoded_features.f32", decoded)

    _manifest(
        out_dir,
        [
            "M4-16 T11 X-Codec 2 FSQ reference fixture (SYNTHETIC projection)",
            oracle_note,
            f"versions: {_versions()}",
            f"levels tuple: {levels} (the released X-Codec 2 value — "
            f"vq/codec_decoder_vocos.py; effective vocab = {vocab})",
            f"canonical released d_model: 2048 (vq_dim default) — fixture slices d_model "
            f"to {args.d_model}; the levels axis is NOT sliced",
            "weights: NONE downloaded / used — project_out is seeded standard_normal "
            "(real-checkpoint parity = flip-the-switch, model-integration WP)",
            f"fixture shapes: out_proj_weight [({args.d_model}, {n_dims})] (nn.Linear "
            f"[out, in] row-major); out_proj_bias [({args.d_model},)]; "
            f"codes [({args.time},)] u32 in [0, {vocab}) with boundary codes pinned; "
            f"decoded_features [({args.time}, {args.d_model})]",
            f"seed: {args.seed}",
        ],
        [
            "levels.u32",
            "out_proj_weight.f32",
            "out_proj_bias.f32",
            "codes.u32",
            "decoded_features.f32",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    wt = sub.add_parser("wavtokenizer", help="WavTokenizer single-codebook VQ fixture")
    wt.add_argument("--out", required=True)
    wt.add_argument("--seed", type=int, default=0)
    wt.add_argument("--time", type=int, default=32)
    wt.add_argument("--rows", type=int, default=256)
    wt.add_argument("--d-model", type=int, default=64)
    wt.add_argument("--oracle", choices=["torch", "numpy"], default="torch")
    wt.set_defaults(func=dump_wavtokenizer)

    xc = sub.add_parser("xcodec2", help="X-Codec 2 FSQ dequant fixture")
    xc.add_argument("--out", required=True)
    xc.add_argument("--seed", type=int, default=0)
    xc.add_argument("--time", type=int, default=32)
    xc.add_argument("--d-model", type=int, default=64)
    xc.add_argument("--oracle", choices=["torch", "numpy"], default="torch")
    xc.set_defaults(func=dump_xcodec2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
