# rng_torch fixtures — MT19937 + PhiloxRNGEngine.h reference vectors

Byte-exact reference vectors consumed by:

- `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs` — **the
  load-bearing anchor** — pins the seed=0 `TorchRandnStream` output to
  the exact bytes real `torch.randn(4, device='cpu')` produces. No
  fixture file needed (bytes are inlined as `u32` constants).
- `crates/vokra-core/tests/rng_torch_randn_e2e.rs` — end-to-end
  `torch_randn_f32(seed, &mut out)` fixture diff (this directory's
  `torch_randn_*.f32.bin` files, regenerated after the 2026-08-08
  MT19937 correction; see "Provenance" below).
- `crates/vokra-core/tests/rng_philox_randn.rs` — one-shot Box-Muller
  sample-0 / sample-1 pins for `philox_randn_sample` (kept as a Philox
  primitive — **not** torch.randn; see the Historical note below).

## 2026-08-08 correction (bisect wf_20fa0933-53d)

**Prior to this correction**, the `torch_randn_*.f32.bin` fixtures held
the byte output of `philox_randn_sample(TorchPhiloxState::next_block())`
in the belief that this reproduced `torch.randn(device='cpu')`. A
byte-level bisect against real `torch.randn(4)` seed=0 (bit patterns
`[0x3fc53f5c, 0xbe963c50, 0xc00b7149, 0x3f1184b6]`) found NO match at
any sample: CPU torch uses `at::mt19937_engine` +
`at::normal_distribution<double>`, not Philox. The
`PhiloxRNGEngine.h::randn` function the previous fixtures came from is
dead code inside torch (that header, lines 39-41, states it is "not used
anywhere except for tests in cpu_generator_test.cpp").

After the correction, `TorchRandnStream` uses
`vokra_core::rng::TorchMt19937Engine` with f64 Box-Muller and pair
caching — bit-exact against real `torch.randn(cpu)` at the seed=0
anchor. These fixtures were regenerated on 2026-08-08 from the new
algorithm and verified byte-for-byte by
`tools/parity/torch_randn_cpu_dump.py --self-test` before commit.

## Provenance (post-correction)

- **Generator**: `tools/parity/torch_randn_cpu_dump.py` — pure-Python
  port of `ATen/core/MT19937RNGEngine.h::mt19937_engine` +
  `ATen/core/DistributionsHelper.h:187-198`
  (`normal_distribution<double>`), independent of PyTorch (no `import
  torch`) so it can regenerate fixtures on a torch-less host.
- **Anchor**: the dumper's `--self-test` mode checks the first 8
  MT19937 tempered u32s at seed=0 against `[0x8c7f0aac, 0x97c4aa2f,
  0xb716a675, 0xd821ccc0, 0x9a4eb343, 0xdba252fb, 0x8b7d76c3,
  0xd8e57d67]` (hand-derived from the algorithm), then verifies the
  first 4 `torch.randn` bit patterns against the seed=0 anchor above.
  This is the shared-bug hazard defense — the Rust and Python impls
  agree because they both agree with **real torch**, not because they
  share a bug.
- **Endianness**: all files are little-endian (all Vokra CI targets are
  x86_64 or aarch64). f32 sample files are raw 4-byte IEEE-754
  little-endian; u32 word files are raw 4-byte little-endian.
- **Host**: fixtures generated on macOS (Apple Silicon), Python 3.12
  via uv, math.log1p/sqrt/sin/cos → libm on macOS. Rust reads the same
  `std::f64::{ln_1p, sqrt, sin, cos}` linked to the same libm on the
  same host — the f64 Box-Muller chain is bit-identical when
  regenerated on the same platform.
- **Fast-path bit-parity is NOT guaranteed even intra-arch** (2026-08-08,
  PR27-RNG-CROSS-ARCH audit gap): the `k=100` and `k=1000` fixtures
  drive `torch_randn_f32`'s `K >= 16` fast path (`normal_fill_16_scalar`)
  which uses **f32** `logf`/`cosf`/`sinf`. Rust's LLVM lowering may
  pick a different libm entry than CPython's `math` module even on the
  SAME macOS Apple Silicon host, producing ~1 ULP delta at 1-2 samples
  per 1000. The `rng_torch_randn_e2e.rs` test therefore applies a
  per-sample 2-ULP tolerance for these fixtures (bit-exact anchor is
  the `k=4` streaming-path test, which runs at f64 precision and holds
  cross-arch). See `docs/adr/sbv2-libm-strategy.md` for the impossibility
  of "match torch bit-exact on every host" and the justification for
  this tolerance in place of vendoring `rust-lang/libm` / RLIBM / SLEEF.

  **Status update 2026-08-09**: workspace-wide vendoring of
  `rust-lang/libm` / RLIBM / SLEEF stays **rejected** (ADR §3.1 / §3.2
  unchanged) — the tolerance justification above is still in force. But
  the SBV2 hot-path libm swap (WP-05, ~40h CC budget) has since been
  **ACCEPTED** as a scoped exception; see `docs/adr/sbv2-libm-strategy.md`
  §3.2.1 for the boundary between "workspace-wide vendoring, still
  rejected" and "SBV2-only in-tree hot-path swap, now authorized". The
  in-tree swap covers HiFi-GAN `tanh` / `sqrt` (WP-08 dominant term),
  DeBERTa transcendentals (WP-10), text-encoder / flow primitives
  (WP-11), and SbV2SDP sampling (WP-12 — this is the site that touches
  `TorchRandnStream::next_f32`'s Box-Muller). WP-12 may retire this
  per-sample 2-ULP tolerance on the SBV2 downstream consumers of the
  RNG (via per-arch byte-exact baseline pinning, WP-06); the fixtures
  themselves and the `rng_torch_randn_e2e.rs` test continue to describe
  the general RNG contract vs torch on **host** libm, which does not
  change.

## Regeneration

```bash
cd tools/parity
uv run python torch_randn_cpu_dump.py --self-test
uv run python torch_randn_cpu_dump.py --seed 0     --randn-samples 4    --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed0_k4.f32.bin
uv run python torch_randn_cpu_dump.py --seed 42    --randn-samples 100  --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed42_k100.f32.bin
uv run python torch_randn_cpu_dump.py --seed 12345 --randn-samples 1000 --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_randn_seed12345_k1000.f32.bin
# Kept-for-Philox-primitive vectors (u32 raw block output — used only by
# rng_philox_kat.rs / rng_philox_state.rs, NOT for torch.randn):
uv run python torch_philox_dump.py --seed 0  --n 8 --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_philox_seed0_n8.u32.bin
uv run python torch_philox_dump.py --seed 42 --n 8 --out ../../crates/vokra-core/tests/fixtures/rng_torch/torch_philox_seed42_n8.u32.bin
```

Regenerate ONLY when (a) the algorithm changes upstream (MT19937
hasn't changed since ~torch 0.3; Philox hasn't changed since ~torch
1.5), or (b) a new libm-quality issue is discovered on the fixture-
generation host. Do NOT regenerate on every CI run — the fixtures are
shipped in the repo (~4.5 KB total across all five files) and gated
with owner review.

## Files

| File | Size | Content |
| ---- | ---- | ------- |
| `torch_philox_seed0_n8.u32.bin`      |   32 B | first 8 raw Philox u32 words for `TorchPhiloxState::from_manual_seed(0)` — Philox primitive only, NOT torch.randn |
| `torch_philox_seed42_n8.u32.bin`     |   32 B | first 8 raw Philox u32 words for seed=42 (seed diversity check) — Philox primitive only, NOT torch.randn        |
| `torch_randn_seed0_k4.f32.bin`       |   16 B | `torch_randn_f32(0, &mut [f32; 4])` end-to-end (**MT19937** post-correction, byte-exact vs real `torch.randn(4)`) |
| `torch_randn_seed42_k100.f32.bin`    |  400 B | `torch_randn_f32(42, &mut [f32; 100])` (seed + length diversity)         |
| `torch_randn_seed12345_k1000.f32.bin`| 4000 B | `torch_randn_f32(12345, &mut [f32; 1000])` (stress: 1000 samples)        |
