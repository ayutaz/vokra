# M1-02 — K-quant loader correctness (Q4_K / Q5_K / Q6_K)

M1-02 adds GGUF direct-load of the K-quant super-block types `Q4_K` (tag 12),
`Q5_K` (tag 13) and `Q6_K` (tag 14). Their scalar, `unsafe`-free dequantizer
lives in `crates/vokra-core/src/gguf/quant/`, and the offline quantizer lives in
`crates/vokra-convert/src/quantize.rs`.

## Why this directory has no committed reference tensors

Loader correctness is proven entirely by **internal oracles** — no fabricated
reference numbers and no external artifact (per the repo's numerical-parity
rule). There is therefore nothing to download or commit here; the oracles are
the tests themselves:

1. **Analytic closed-form super-block** (`gguf/quant/q{4,5,6}_k.rs` tests):
   a super-block is hand-built in-test from chosen `d` / `dmin` / sub-scales /
   quants (packed with the exact inverse of ggml's `get_scale_min_k4`), then
   dequantized and asserted **bit-for-bit** against the closed form
   `y = d·sc·q − dmin·m` (Q4/Q5) or `y = d·sc·q` (Q6). This pins the on-disk
   byte layout with zero external reference.
2. **Constant-block oracle**: uniform quants + scales must dequantize to one
   known constant.
3. **Block-aware sizing** (`gguf/tensor.rs`, `reader.rs`, `writer.rs` tests):
   a 256-element `Q4_K` tensor is exactly 144 bytes; the reader/writer accept a
   correctly-sized payload and reject short / partial-block ones
   (`TensorSizeMismatch` / `BlockSizeMisaligned`).
4. **Quantize → dequant round-trip** (`vokra-convert/src/quantize.rs` tests):
   the in-repo quantizer encodes real (deterministic pseudo-random) `f32`
   vectors, the runtime dequantizer decodes them, and the per-block error is
   asserted `≤` one analytic quantization step plus the `f16` scale-storage
   slack. This is a fully-internal differential between two independent,
   hand-written implementations of the same format — no llama.cpp artifact
   required. A whole quantized Whisper GGUF is producible in-repo via
   `vokra-convert --model whisper-base --quantize q4_k …`.

## Optional external differential (follow-up, not required)

If a llama.cpp-produced `Q4_K` / `Q5_K` / `Q6_K` GGUF is ever wanted as an
extra cross-check, the intended shape is an **env-gated** test (e.g.
`VOKRA_KQUANT_GGUF` pointing at the GGUF plus a committed offline `.f32` dump)
that compares Vokra's dequant against it at a tolerance, and is `#[ignore]`d
when absent — mirroring the existing `whisper_base` / `piper_plus` parity
suites. It is not needed for M1-02's loader-correctness bar and is left as a
follow-up.

## Explicitly out of scope for M1-02

- IQ2 / other i-quant families (FR-QT-01 marks them 極小デバイス用).
- SIMD-accelerated dequant (belongs in `vokra-backend-cpu`, the
  `unsafe`-allowed crate; must stay bit-identical to the scalar reference).
- A quality-optimizing quantizer (M1-02 ships a valid, bounded-error quantizer,
  not ggml's `make_qkx2_quants` search).
