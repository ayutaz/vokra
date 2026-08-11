# `p0p4k/vits2_pytorch` vendor (SBV2 v2 Blocker 2b)

**Status: VENDORED (4 `.py` files + LICENSE at pinned commit
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
| commons.py     | subsequent_mask, fused_add_tanh_sigmoid_multiply, convert_pad_shape | `p0p4k/vits2_pytorch/commons.py` @ 1f4f379 | audio processing utilities |
| modules.py     | WN (WaveNet encoder), Flip (coupling reversal) | `p0p4k/vits2_pytorch/modules.py` @ 1f4f379 | inference-only classes |
| attentions.py  | MultiHeadAttention (rel-pos), Encoder, FFN    | `p0p4k/vits2_pytorch/attentions.py` @ 1f4f379                              | Blocker 2b flow attention  |
| models.py      | TransformerCouplingLayer, TransformerCouplingBlock | `p0p4k/vits2_pytorch/models.py` @ 1f4f379, classes only (no train utils) | Blocker 2b flow coupling   |

## Per-file sha256 (verify with `shasum -a 256 <file>`)

| file          | sha256                                                             |
|---------------|--------------------------------------------------------------------|
| LICENSE       | 3d8165162cef96f686f02146ac2e4ae80db5797296a99c658befa424ee64727b |
| commons.py    | 633d0a7e7f721a9c61321fb208d7ee7722fa1de0126a2c44410045e639da97de |
| modules.py    | 28c7442ad39a91a28f07b388ec05b18b0bfad463ae1e05a6bf1c9569cca57611 |
| attentions.py | 891973c7bea578e606b6381f7db93821e711d6c11aebfc91ad6627a153eed8a1 |
| models.py     | b67f5ca0a27b7ebf8b4f7a72f80440d08c77ea60bc116474821368ae311dae56 |

## Clean-room contract

Files here may be READ or DERIVED FROM by Vokra's clean-room SBV2 v2
port. NOT REFERENCED: `litagin02/Style-Bert-VITS2` (AGPL-3.0) or
`fishaudio/Bert-VITS2` (AGPL-3.0). Diffing intermediate tensors from
Python side (this dir) against Rust side (`crates/vokra-models/src/sbv2/`)
is the mechanism by which per-tensor atol calibration in
`crates/vokra-models/tests/sbv2_parity_atol_calibration.rs` acquires
its rationale bounds.
