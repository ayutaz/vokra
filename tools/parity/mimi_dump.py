#!/usr/bin/env python3
"""Dump RVQ codec reference fixtures — Mimi / DAC / EnCodec (M4-04 T13/T14/T15).

This is an **offline** tool (FR-LD-05: no Python / PyTorch is ever pulled into
the runtime). It writes the committed fixtures under ``tests/parity/{mimi,dac,
encodec}/`` that the Rust parity tests (``crates/vokra-ops/tests/parity_mimi_
reference.rs`` / ``parity_dac_reference.rs`` / ``parity_encodec_reference.rs``)
compare against at FP32 ``atol = 0.01`` (NFR-QL-01). CI never runs Python —
the fixtures are committed (kaldi_fbank precedent).

History: the M3-06 placeholder of this file deferred the Kyutai reference
dump ("lands with M3-09"); M4-04 resolves that deferral and, per ADR M4-04
§D-j, folds the DAC / EnCodec subcommands into this same file (shared
manifest/writer helpers) instead of a new ``rvq_dump.py``.

Subcommands
-----------

``mimi``
    Loads a **moshi-native** Mimi checkpoint (pinned:
    ``kyutai/moshiko-pytorch-bf16`` rev ``2bfc9ae6…``,
    ``tokenizer-e351c8d8-checkpoint125.safetensors``, weights CC-BY 4.0 —
    attribution note lands in the manifest) through the canonical
    ``moshi.models.loaders.get_mimi(..., num_codebooks=8)`` API, then dumps:

    - ``codebook_tables_sliced.f32`` — ``[8, rows, d_model]`` **effective
      (pre-projected)** tables: row ``i`` of codebook ``cb`` is
      ``output_proj(embedding[cb][i])`` computed **by the upstream modules
      themselves** (the ``EuclideanCodebook.embedding`` property =
      ``embedding_sum / clamp(cluster_usage, 1e-5)``, then the split's
      bias-free 1×1 conv). Sliced to the first ``rows`` entries so the
      committed fixture stays ≤ 1 MB — RVQ decode is a gather + sum, so a
      sliced table is a complete op-semantics fixture as long as the codes
      stay < ``rows``.
    - ``codes.u32`` — ``[time, 8]`` fixed-seed codes in ``[0, rows)``.
    - ``decoded_features.f32`` — ``[time, d_model]`` reference decode from
      the **public** ``mimi.quantizer.decode`` API (full tables; identical
      gather because every code < rows). NOTE the reference projects
      *after* the per-split sum while the Vokra op sums pre-projected rows —
      mathematically equal (bias-free linear proj), FP32-reassociation-
      different; the measured gap is recorded by the Rust test.
    - ``manifest.txt`` — shapes / seed / sha256 / package versions / pins.

``dac``
    Loads a DAC release ``weights_*.pth`` (pinned: 24 kHz / 8 kbps tag 0.0.4)
    with ``torch.load(weights_only=True)``, rebuilds the upstream
    ``dac.model.DAC(**metadata.kwargs)``, and dumps the **factorized** decode
    fixture (first ``--books`` quantizers, prefix decode = upstream
    variable-bitrate semantics):

    - ``codebook_tables_sliced.f32`` — ``[books, rows, codebook_dim]`` low-dim
      codebook rows (verbatim);
    - ``out_proj_weight.f32`` — ``[books, d_model, codebook_dim]`` the
      weight-normed conv's **effective** weight (read back from the upstream
      module after a forward, i.e. exactly what torch's weight_norm computes);
    - ``out_proj_bias.f32`` — ``[books, d_model]``;
    - ``codes.u32`` — ``[time, books]`` fixed-seed codes in ``[0, rows)``;
    - ``decoded_features.f32`` — ``[time, d_model]`` from the public
      ``model.quantizer.from_codes`` API;
    - ``manifest.txt``.

``encodec``
    **No pretrained weight is downloaded — FR-OP-32 permanent constraint.**
    Builds ``EncodecModel.encodec_model_24khz(pretrained=False)`` (MIT
    reference **code**, canonical 24 kHz config: n_q=32, bins=1024,
    dimension=128 — read from the constructed model, not hard-coded),
    seed-randomizes every codebook ``embed`` buffer, and dumps the synthetic
    fixture (first ``--books`` quantizers, codes < ``rows``):

    - ``codebook_tables_sliced.f32`` / ``codes.u32`` /
      ``decoded_features.f32`` (via the public ``quantizer.decode`` API,
      codes layout ``[K, B, T]``) / ``manifest.txt``.

Fixture filenames deliberately use ``.f32`` / ``.u32`` / ``.txt`` only —
never the weight extensions ``.safetensors/.gguf/.pth/.bin`` — so
``scripts/compliance/check-encodec-exclusion.sh`` filename matching stays
inert on them (verified in M4-04 T15).

Usage
-----

::

    venv/bin/python tools/parity/mimi_dump.py mimi \\
        --checkpoint /path/to/tokenizer-e351c8d8-checkpoint125.safetensors \\
        --out tests/parity/mimi --seed 0 --time 32 --rows 48

    venv/bin/python tools/parity/mimi_dump.py dac \\
        --checkpoint /path/to/weights_24khz.pth \\
        --out tests/parity/dac --seed 0 --time 32 --rows 192 --books 12

    venv/bin/python tools/parity/mimi_dump.py encodec \\
        --out tests/parity/encodec --seed 0 --time 32 --rows 128 --books 8
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


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


def _versions(*packages: str) -> list[str]:
    import importlib.metadata as md

    out = []
    for p in packages:
        try:
            out.append(f"{p}=={md.version(p)}")
        except Exception:  # noqa: BLE001
            out.append(f"{p}==<missing>")
    return out


def _finish_manifest(out: Path, lines: list[str], files: list[str]) -> None:
    for f in files:
        lines.append(f"sha256 {f} {_sha256(out / f)}")
    (out / "manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# mimi
# ---------------------------------------------------------------------------


def cmd_mimi(args: argparse.Namespace) -> int:
    import torch
    from moshi.models import loaders

    torch.manual_seed(args.seed)
    mimi = loaders.get_mimi(args.checkpoint, device="cpu", num_codebooks=args.books)
    mimi.eval()
    q = mimi.quantizer  # SplitResidualVectorQuantizer

    rows, time, books = args.rows, args.time, args.books
    n_acoustic = books - 1

    # Effective sliced tables via the upstream modules (semantic first).
    tables = []
    with torch.no_grad():
        for cb in range(books):
            if cb == 0:
                layer = q.rvq_first.vq.layers[0]
                proj = q.rvq_first.output_proj
            else:
                layer = q.rvq_rest.vq.layers[cb - 1]
                proj = q.rvq_rest.output_proj
            emb = layer._codebook.embedding[:rows]  # noqa: SLF001  [rows, dim]
            # 1x1 conv expects [B, C, T]: rows become the "time" axis.
            eff = proj(emb.t().unsqueeze(0)).squeeze(0).t()  # [rows, d_model]
            tables.append(eff)
        tables_t = torch.stack(tables)  # [books, rows, d_model]
        d_model = tables_t.shape[-1]

        gen = torch.Generator().manual_seed(args.seed)
        codes_bkt = torch.randint(0, rows, (1, books, time), generator=gen)  # [B, K, T]
        decoded = q.decode(codes_bkt)  # [1, d_model, T] — public API
        decoded_td = decoded.squeeze(0).t().contiguous()  # [T, d_model]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_f32(out / "codebook_tables_sliced.f32", tables_t.numpy())
    _write_u32(out / "codes.u32", codes_bkt.squeeze(0).t().contiguous().numpy())  # [T, K]
    _write_f32(out / "decoded_features.f32", decoded_td.numpy())

    lines = [
        "M4-04 T13 Mimi RVQ reference fixture (sliced)",
        f"reference: moshi (Kyutai) public API — SplitResidualVectorQuantizer.decode; {', '.join(_versions('moshi', 'torch', 'numpy'))}",
        "checkpoint pin: kyutai/moshiko-pytorch-bf16 rev 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 tokenizer-e351c8d8-checkpoint125.safetensors (ADR M4-04 §D-k)",
        f"checkpoint sha256: {_sha256(Path(args.checkpoint))}",
        "weights license: CC-BY 4.0 (Kyutai) — attribution: NOTICE §5 'Mimi Codec (Kyutai / Moshi authors)'; this fixture embeds sliced rows of those weights",
        f"shapes: codebook_tables_sliced [({books}, {rows}, {d_model})] f32; codes [({time}, {books})] u32; decoded_features [({time}, {d_model})] f32",
        f"seed: {args.seed}; codes in [0, {rows}); books = 1 semantic + {n_acoustic} acoustic (num_codebooks={books} prefix of the physical 32)",
        "note: reference projects AFTER the per-split residual sum; the Vokra op sums pre-projected rows — equal for the bias-free linear proj up to FP32 reassociation (measured bound recorded in parity_mimi_reference.rs)",
    ]
    _finish_manifest(
        out, lines, ["codebook_tables_sliced.f32", "codes.u32", "decoded_features.f32"]
    )
    return 0


# ---------------------------------------------------------------------------
# dac
# ---------------------------------------------------------------------------


def cmd_dac(args: argparse.Namespace) -> int:
    import math

    import torch
    from dac.model import DAC

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kwargs = payload["metadata"]["kwargs"]
    model = DAC(**kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    rows, time, books = args.rows, args.time, args.books
    n_codebooks_full = int(kwargs["n_codebooks"])
    if books > n_codebooks_full:
        print(f"--books {books} > checkpoint n_codebooks {n_codebooks_full}", file=sys.stderr)
        return 2

    with torch.no_grad():
        gen = torch.Generator().manual_seed(args.seed)
        codes_bnt = torch.randint(0, rows, (1, books, time), generator=gen)  # [B, N, T]
        # Public API prefix decode (variable-bitrate semantics) — also fires
        # the weight_norm pre-forward hooks so .weight below is current.
        z_q, _z_p, _codes = model.quantizer.from_codes(codes_bnt)
        decoded_td = z_q.squeeze(0).t().contiguous()  # [T, d_model]

        tables = []
        weights = []
        biases = []
        for i in range(books):
            vq = model.quantizer.quantizers[i]
            tables.append(vq.codebook.weight[:rows])  # [rows, codebook_dim]
            # Effective (weight-normed) conv weight as torch computed it in
            # the forward above: [d_model, codebook_dim, 1] -> squeeze.
            weights.append(vq.out_proj.weight.squeeze(-1))
            biases.append(vq.out_proj.bias)
        tables_t = torch.stack(tables)
        weights_t = torch.stack(weights)
        biases_t = torch.stack(biases)
        codebook_dim = tables_t.shape[-1]
        d_model = weights_t.shape[1]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_f32(out / "codebook_tables_sliced.f32", tables_t.numpy())
    _write_f32(out / "out_proj_weight.f32", weights_t.numpy())
    _write_f32(out / "out_proj_bias.f32", biases_t.numpy())
    _write_u32(out / "codes.u32", codes_bnt.squeeze(0).t().contiguous().numpy())
    _write_f32(out / "decoded_features.f32", decoded_td.numpy())

    hop = math.prod(kwargs["encoder_rates"])
    lines = [
        "M4-04 T14 DAC factorized RVQ reference fixture (sliced, projection included)",
        f"reference: descript-audio-codec (MIT) public API — ResidualVectorQuantize.from_codes; {', '.join(_versions('descript-audio-codec', 'torch', 'numpy'))}",
        "checkpoint pin: descriptinc/descript-audio-codec release tag 0.0.4 weights_24khz.pth (24 kHz / 8 kbps zoo-primary variant — ADR M4-04 §T02)",
        f"checkpoint sha256: {_sha256(Path(args.checkpoint))}",
        f"upstream DacRvqAttrs pin (fixture asserts these): n_codebooks={n_codebooks_full} codebook_size={kwargs['codebook_size']} codebook_dim={kwargs['codebook_dim']} d_model={d_model} sample_rate={kwargs['sample_rate']} hop={hop}",
        f"fixture subset: first {books} quantizers (prefix decode = upstream variable-bitrate semantics), rows sliced to {rows}",
        f"shapes: codebook_tables_sliced [({books}, {rows}, {codebook_dim})]; out_proj_weight [({books}, {d_model}, {codebook_dim})]; out_proj_bias [({books}, {d_model})]; codes [({time}, {books})] u32; decoded_features [({time}, {d_model})]",
        f"seed: {args.seed}; codes in [0, {rows})",
        "note: out_proj_weight is the weight-norm EFFECTIVE weight as torch computed it (g*v/||v||, dim=0); the Vokra converter folds the same formula offline",
    ]
    _finish_manifest(
        out,
        lines,
        [
            "codebook_tables_sliced.f32",
            "out_proj_weight.f32",
            "out_proj_bias.f32",
            "codes.u32",
            "decoded_features.f32",
        ],
    )
    return 0


# ---------------------------------------------------------------------------
# encodec (synthetic weights — FR-OP-32)
# ---------------------------------------------------------------------------


def cmd_encodec(args: argparse.Namespace) -> int:
    import torch
    from encodec import EncodecModel

    # pretrained=False — NO weight download, ever (FR-OP-32 permanent
    # constraint; the fixture is synthetic).
    model = EncodecModel.encodec_model_24khz(pretrained=False)
    q = model.quantizer
    n_q_full, bins, dim = q.n_q, q.bins, q.dimension

    rows, time, books = args.rows, args.time, args.books
    if books > n_q_full or rows > bins:
        print(
            f"--books {books} / --rows {rows} exceed canonical n_q={n_q_full} bins={bins}",
            file=sys.stderr,
        )
        return 2

    gen = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for layer in q.vq.layers:
            emb = torch.randn(bins, dim, generator=gen)
            layer._codebook.embed.copy_(emb)  # noqa: SLF001

        codes_kbt = torch.randint(0, rows, (books, 1, time), generator=gen)  # [K, B, T]
        decoded = q.decode(codes_kbt)  # [1, dim, T] — public API
        decoded_td = decoded.squeeze(0).t().contiguous()

        tables = torch.stack(
            [q.vq.layers[i]._codebook.embed[:rows] for i in range(books)]  # noqa: SLF001
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_f32(out / "codebook_tables_sliced.f32", tables.numpy())
    _write_u32(out / "codes.u32", codes_kbt.squeeze(1).t().contiguous().numpy())  # [T, K]
    _write_f32(out / "decoded_features.f32", decoded_td.numpy())

    lines = [
        "M4-04 T15 EnCodec RVQ reference fixture (SYNTHETIC weights — FR-OP-32)",
        f"reference: encodec (MIT code) public API — ResidualVectorQuantizer.decode on fixed-seed random codebooks; {', '.join(_versions('encodec', 'torch', 'numpy'))}",
        f"canonical config source: EncodecModel.encodec_model_24khz(pretrained=False) constructed model — n_q={n_q_full}, bins={bins}, dimension={dim} (read from the model object, not hard-coded)",
        "weights: NONE downloaded / used — codebooks are torch.randn with the seed below (pretrained EnCodec weights are CC-BY-NC 4.0 and permanently zoo-excluded; they are not used even for fixture generation)",
        f"fixture subset: first {books} quantizers, rows sliced to {rows}",
        f"shapes: codebook_tables_sliced [({books}, {rows}, {dim})]; codes [({time}, {books})] u32; decoded_features [({time}, {dim})]",
        f"seed: {args.seed}; codes in [0, {rows})",
    ]
    _finish_manifest(
        out, lines, ["codebook_tables_sliced.f32", "codes.u32", "decoded_features.f32"]
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mimi", help="Kyutai Mimi reference dump (real sliced weights)")
    p.add_argument("--checkpoint", required=True, help="moshi-native Mimi safetensors")
    p.add_argument("--out", type=Path, default=Path("tests/parity/mimi"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", type=int, default=32)
    p.add_argument("--rows", type=int, default=48)
    p.add_argument("--books", type=int, default=8)
    p.set_defaults(fn=cmd_mimi)

    p = sub.add_parser("dac", help="DAC reference dump (real sliced weights, projection included)")
    p.add_argument("--checkpoint", required=True, help="DAC release weights_*.pth")
    p.add_argument("--out", type=Path, default=Path("tests/parity/dac"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", type=int, default=32)
    p.add_argument("--rows", type=int, default=192)
    p.add_argument("--books", type=int, default=12)
    p.set_defaults(fn=cmd_dac)

    p = sub.add_parser("encodec", help="EnCodec op-path dump (synthetic weights only — FR-OP-32)")
    p.add_argument("--out", type=Path, default=Path("tests/parity/encodec"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", type=int, default=32)
    p.add_argument("--rows", type=int, default=128)
    p.add_argument("--books", type=int, default=8)
    p.set_defaults(fn=cmd_encodec)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
