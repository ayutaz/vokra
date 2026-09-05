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
currently contains only this contract README; no reference values are claimed
until the remote PyTorch generation and hardware test have both completed.
