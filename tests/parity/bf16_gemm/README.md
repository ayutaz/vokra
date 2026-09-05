# AVX-512 BF16 GEMM parity fixture contract

This directory is intentionally empty of numerical fixtures in Phase 1.
The packet will be generated on the authorized VAST host by
`tools/parity/bf16_gemm/dump_reference.py` and must contain exactly:

```text
README.md
manifest.json
manifest.sha256
full_k32_m8_n64_a.f32
full_k32_m8_n64_b.f32
full_k32_m8_n64_output.f32
tails_m3_n35_k65_a.f32
tails_m3_n35_k65_b.f32
tails_m3_n35_k65_output.f32
tails_m9_n33_k31_a.f32
tails_m9_n33_k31_b.f32
tails_m9_n33_k31_output.f32
```

The manifest schema is `vokra-bf16-gemm-parity-v1`.  It must retain the exact
PyTorch version, generator identity, oracle expression, CPU/device and
little-endian float32 provenance, per-file byte count/shape/dtype/SHA-256, and
the outer SHA-256 pin in `manifest.sha256`.  Inputs are deterministic and do
not use a random seed.  The oracle is PyTorch `torch.matmul` over the widened
BF16 inputs, not a handwritten mirror of the Vokra kernel.

The comparison bound is pre-registered as `atol=1e-3`, `rtol=0.0` from the
existing NFR-QL-01 ceiling.  A VAST failure is a kernel/reference issue to
investigate; the bound must not be widened from the observed result.

The three cases cover a complete `k` block (`k=96`), `m/n/k` tails
(`3x35x65`), and an additional odd `m/n/k` shape (`9x33x31`).  The Rust
consumer is ignored until this exact packet has been generated.  A host
without AVX-512 BF16 must report a skip; it must never run the test through a
different ISA or silently fall back to scalar.
