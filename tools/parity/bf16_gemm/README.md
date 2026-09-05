# BF16 GEMM reference generator

This directory contains the isolated Python 3.12 reference environment for
the AVX-512 BF16 GEMM parity packet.  `dump_reference.py` imports PyTorch and
calls its real `torch.matmul` operation after rounding both float32 inputs to
BF16 and widening them back to float32.  It does not import Vokra, execute a
model, acquire weights, or implement a second GEMM kernel.

Run it only on the authorized VAST host after the exact commit has been
transferred and the `uv.lock` environment has been synchronized:

```bash
uv run --project tools/parity/bf16_gemm --locked \
  python tools/parity/bf16_gemm/dump_reference.py \
  --output tests/parity/bf16_gemm
```

The comparison bound is pre-registered as absolute `atol=1e-3` and
`rtol=0.0`, within the existing NFR-QL-01 FP32 ceiling.  It is not adjusted
from a passing or failing run; if the exact VAST run fails, investigate the
kernel or reference environment instead of widening the bound.  The generator
writes raw little-endian float32 tensors, a
strict JSON manifest, and `manifest.sha256`.  The committed fixture directory
was generated and checked on VAST instance `49972360` at exact Vokra HEAD
`7d0119c942ed6cef2f50a89917e1fbd177ed427e`.

## VAST evidence (2026-09-06)

The reference environment was resolved and synchronized on VAST with
`uv.lock` SHA-256
`b93aececa3fa3b7649e40c8b51b3c5a7a69cf3421e1f43b808fe1eac8d107b92` and
Torch `2.13.0+cpu`.  No model, weight, HF token, or upload was involved.
The host CPU was `AMD Ryzen 7 7700X 8-Core Processor` and exposed
`avx512_bf16` together with `avx512f`, `avx512dq`, `avx512bw`, and `avx512vl`;
PyTorch reported `AVX512`.

The generated `manifest.json` SHA-256 is
`b9e7b687ef6352b30f258b0b1c02695e724e32443665e74981cac66b025b1ba3`, and its
outer `manifest.sha256` file SHA-256 is
`1b1eb4a07bfb570492360a21611a70b8e91c8c17db4e3202b8aca4ffeeb68fb1`.
The three-case Rust test was run twice with
`CARGO_BUILD_JOBS=1 RUSTC_WRAPPER= cargo test -p vokra-backend-cpu --test
bf16_gemm_torch_parity -- --ignored --nocapture`; both runs were
`1 passed / 0 failed`.  The second run log SHA-256 is
`11e95e33bf9abe3e22a3deace90ad6424bd6b2aad74984cff994f7d2e893bca6`.

An independent output probe measured maximum absolute differences of
`7.629394531e-6` (`full_k32_m8_n64`), `3.814697266e-6`
(`tails_m3_n35_k65`), and `1.907348633e-6` (`tails_m9_n33_k31`), with global
maximum `7.629394531e-6`.  The pre-registered `atol=1e-3`, `rtol=0.0` was
unchanged.
