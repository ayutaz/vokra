# rng_torch fixtures — PyTorch-compatible Philox4x32-10 reference vectors

Byte-exact reference vectors for `vokra_core::rng::{TorchPhiloxState,
TorchRandnStream, torch_randn_f32}` consumed by

- `crates/vokra-core/tests/rng_philox_randn.rs` — one-shot Box-Muller
  sample-0 / sample-1 pins;
- `crates/vokra-core/tests/rng_torch_randn_e2e.rs` — end-to-end
  `torch_randn_f32(seed, &mut out)` fixture diff (this directory's
  `torch_randn_*.f32.bin` files).

## Provenance

- **Generator**: `tools/parity/torch_philox_dump.py`, a pure-Python port
  of ATen's `aten/src/ATen/core/PhiloxRNGEngine.h::randn` + counter
  increment. Runs under uv-managed Python 3.12 (`cd tools/parity && uv
  run python torch_philox_dump.py --self-test`).
- **Cross-check**: the dumper self-tests against Random123 v1.14 KAT
  vectors (independent implementation by DE Shaw Research,
  BSD-3-Clause) before every fixture emit — see the `--self-test` flag
  and the `RANDOM123_KATS` list in the dumper. This forecloses the
  "both implementations have the same bug so tests falsely pass"
  hazard.
- **Endianness**: all files are little-endian (all Vokra CI targets are
  x86_64 or aarch64 — both little-endian). f32 sample files are raw
  4-byte IEEE-754 little-endian; u32 word files are raw 4-byte
  little-endian.
- **Host**: fixtures generated on macOS (Apple Silicon), Python 3.12
  via uv, math.log/sqrt/cos → libm on macOS. Rust reads the same
  `std::f32::{ln, sqrt, cos}` which link to the same libm on the same
  host — the Box-Muller f32 chain is therefore bit-identical when
  regenerated on the same platform. Cross-platform bit-parity is
  documented as a risk in the module doc (Linux glibc libm vs macOS
  libm vs Windows msvcrt).

## Regeneration

```bash
cd tools/parity
uv run python torch_philox_dump.py --seed 0     --randn-samples 4    --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed0_k4.f32.bin
uv run python torch_philox_dump.py --seed 42    --randn-samples 100  --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed42_k100.f32.bin
uv run python torch_philox_dump.py --seed 12345 --randn-samples 1000 --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed12345_k1000.f32.bin
uv run python torch_philox_dump.py --seed 0     --n 8                --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_philox_seed0_n8.u32.bin
uv run python torch_philox_dump.py --seed 42    --n 8                --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_philox_seed42_n8.u32.bin
```

Regenerate ONLY when (a) the Philox algorithm itself changes upstream
(hasn't since ~torch 1.5), (b) `SCALE`'s bit pattern changes in
`normal_kernel.rs` (would require a code review and separate ADR), or
(c) a new libm-quality issue is discovered on the fixture-generation
host. Do NOT regenerate on every CI run — the fixtures are shipped in
the repo (~4.5 KB total across all five files) and gated with owner
review.

## Files

| File | Size | Content |
| ---- | ---- | ------- |
| `torch_philox_seed0_n8.u32.bin`      |   32 B | first 8 raw Philox u32 words for `TorchPhiloxState::from_manual_seed(0)`  |
| `torch_philox_seed42_n8.u32.bin`     |   32 B | first 8 raw Philox u32 words for seed=42 (seed diversity check)          |
| `torch_randn_seed0_k4.f32.bin`       |   16 B | `torch_randn_f32(0, &mut [f32; 4])` end-to-end (canonical smoke)         |
| `torch_randn_seed42_k100.f32.bin`    |  400 B | `torch_randn_f32(42, &mut [f32; 100])` (seed + length diversity)         |
| `torch_randn_seed12345_k1000.f32.bin`| 4000 B | `torch_randn_f32(12345, &mut [f32; 1000])` (stress: 1000 samples)        |
