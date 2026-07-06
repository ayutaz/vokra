# M2-08 quant parity fixtures

Tiny hand-generated GGUF fixtures for `crates/vokra-core/tests/quant_parity.rs`
(T13). No PyTorch, no external checkpoint — the fixtures are materialised at
test time from analytic values (constant sub-scales, constant quants) so the
expected `dequantize → GEMM → F32` output has a closed-form reference.

The test writes a **whisper-like tiny MLP block** GGUF:

- one `Q4_K` weight tensor shaped `[QK_K=256, 1]` (one super-block, 144 bytes),
- one `F32` bias tensor shaped `[1]` (biases stay F32 per FR-QT-03),
- a `vokra.frontend.*` chunk minimum required for load,
- a tiny `F32` activation input shaped `[QK_K=256]`.

The `Q4_K` block is filled with:
- `d = 0x3C00` (1.0 in f16), `dmin = 0x0000` (0.0 in f16),
- eight identical 6-bit sub-scales `sc = 2`, eight identical sub-mins `m = 0`,
- 256 quants all equal to `q = 4`.

Every element therefore dequantises to `d·sc·q − dmin·m = 1.0·2·4 − 0 = 8.0`
(bit-exact — no rounding). The GEMM in the parity test performs a dot product
of a constant-`1.0` input with the constant-`8.0` weight, giving the closed-form
expected output `256 · 8.0 = 2048.0` (well within `atol=0.01` for the FP16 tier
per NFR-QL-01).

The fixture is intentionally not checked in as a binary: the test builds it
fresh in this directory each run, both because the byte layout is a data-format
specification (ggml `k_quants.h`) that must round-trip through the in-tree
writer, and because a materialised `.gguf` here would be redundant with the
code that constructs it.
