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
| commons.py     | utility helpers (subsequent_mask, fused_add_tanh_sigmoid_multiply, convert_pad_shape, etc) | `p0p4k/vits2_pytorch/commons.py` @ 1f4f379 | shared signal processing |
| modules.py     | LayerNorm, WN (WaveNet encoder), Flip (flow reversal) | `p0p4k/vits2_pytorch/modules.py` @ 1f4f379 (inference subset only) | encoder/flow primitives |
| attentions.py  | MultiHeadAttention (rel-pos), Encoder, FFN (only these 3)    | `p0p4k/vits2_pytorch/attentions.py` @ 1f4f379 (inference subset)                              | Blocker 2b flow attention  |
| models.py      | TransformerCouplingLayer, TransformerCouplingBlock (transformer flows only) | `p0p4k/vits2_pytorch/models.py` @ 1f4f379 (inference subset) | Blocker 2b flow coupling   |

## Per-file sha256 (verify with `shasum -a 256 <file>`)

| file          | sha256                                                             |
|---------------|--------------------------------------------------------------------|
| LICENSE       | 3d8165162cef96f686f02146ac2e4ae80db5797296a99c658befa424ee64727b |
| commons.py    | 633d0a7e7f721a9c61321fb208d7ee7722fa1de0126a2c44410045e639da97de |
| modules.py    | 220ca1025179acdec01487432b46db717d464b838f1472bd4a5480694f9e2027 |
| attentions.py | f57e452fef656b5e5bfd608ba9970a0cc6e6fc509aacb4eda21d5b8b18f1a2a6 |
| models.py     | b67f5ca0a27b7ebf8b4f7a72f80440d08c77ea60bc116474821368ae311dae56 |

## Verify integrity

Run from repo root:

```bash
cd tools/parity/vendor/vits2 && shasum -a 256 -c <(cat <<'EOF'
3d8165162cef96f686f02146ac2e4ae80db5797296a99c658befa424ee64727b  LICENSE
633d0a7e7f721a9c61321fb208d7ee7722fa1de0126a2c44410045e639da97de  commons.py
220ca1025179acdec01487432b46db717d464b838f1472bd4a5480694f9e2027  modules.py
f57e452fef656b5e5bfd608ba9970a0cc6e6fc509aacb4eda21d5b8b18f1a2a6  attentions.py
b67f5ca0a27b7ebf8b4f7a72f80440d08c77ea60bc116474821368ae311dae56  models.py
EOF
)
```

Expected: `LICENSE: OK / commons.py: OK / modules.py: OK / attentions.py: OK / models.py: OK`. Any mismatch means the vendored file has drifted from its pinned upstream — investigate before proceeding.

## Clean-room contract

Files here may be READ or DERIVED FROM by Vokra's clean-room SBV2 v2
port. NOT REFERENCED: `litagin02/Style-Bert-VITS2` (AGPL-3.0) or
`fishaudio/Bert-VITS2` (AGPL-3.0). Diffing intermediate tensors from
Python side (this dir) against Rust side (`crates/vokra-models/src/sbv2/`)
is the mechanism by which per-tensor atol calibration in
`crates/vokra-models/tests/sbv2_parity_atol_calibration.rs` acquires
its rationale bounds.
