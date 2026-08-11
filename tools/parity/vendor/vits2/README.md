# `p0p4k/vits2_pytorch` vendor (SBV2 v2 Blocker 2b)

**Status: VENDORED (2 `.py` files at pinned commit
`1f4f3790568180f8dec4419d5cad5d0877b034bb`, `2023-10-13T07:14:02Z`).**

## What this is

[`p0p4k/vits2_pytorch`](https://github.com/p0p4k/vits2_pytorch) is a
permissive-license (MIT) PyTorch port of VITS2 (Kim et al. 2023,
*VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech
with Adversarial Learning and Architecture Design*, arXiv:2307.16430).

- **Source**: <https://github.com/p0p4k/vits2_pytorch>
- **License**: MIT (see sibling LICENSE, fetched byte-identical from
  raw.githubusercontent.com/p0p4k/vits2_pytorch/1f4f3790568180f8dec4419d5cad5d0877b034bb/LICENSE)
- **Pinned commit**: `1f4f3790568180f8dec4419d5cad5d0877b034bb` (`2023-10-13T07:14:02Z`)

## Why vendored (not pip install)

No PyPI package exists for this repository. Vendoring the minimal
inference-only surface is the permissive-license-preserving way to
depend on it (aligned with `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §6).

## What ships here

| Target file    | Contains                                      | Upstream source                                                             | Feeds parity reference for |
|----------------|-----------------------------------------------|-----------------------------------------------------------------------------|----------------------------|
| attentions.py  | MultiHeadAttention (rel-pos), Encoder, FFN    | `p0p4k/vits2_pytorch/attentions.py` @ 1f4f379                              | Blocker 2b flow attention  |
| models.py      | TransformerCouplingLayer, TransformerCouplingBlock | `p0p4k/vits2_pytorch/models.py` @ 1f4f379, classes only (no train utils) | Blocker 2b flow coupling   |

## Per-file sha256 (verify with `shasum -a 256 <file>`)

| file          | sha256                                                             |
|---------------|--------------------------------------------------------------------|
| LICENSE       | <fill after Task 2>                                                |
| attentions.py | <fill after Task 2>                                                |
| models.py     | <fill after Task 2>                                                |

## Clean-room contract

Files here may be READ or DERIVED FROM by Vokra's clean-room SBV2 v2
port. NOT REFERENCED: `litagin02/Style-Bert-VITS2` (AGPL-3.0) or
`fishaudio/Bert-VITS2` (AGPL-3.0). Diffing intermediate tensors from
Python side (this dir) against Rust side (`crates/vokra-models/src/sbv2/`)
is the mechanism by which per-tensor atol calibration in
`crates/vokra-models/tests/sbv2_parity_atol_calibration.rs` acquires
its rationale bounds.
