//! Standard-normal sampling for two independent worlds:
//!
//! 1. [`TorchRandnStream`] + [`torch_randn_f32`] — bit-exact against
//!    `torch.manual_seed(N); torch.randn(K, device='cpu')` for `K < 16`
//!    (or non-contiguous tensors), by driving `at::mt19937_engine`
//!    (see [`super::mt19937`]) through `at::normal_distribution<double>`
//!    from `aten/src/ATen/core/DistributionsHelper.h:187-198` (torch
//!    source, BSD-3-Clause). This is the path SBV2's SDP noise buffer
//!    consumes when `RngMode::TorchCpuMt19937Parity` is selected.
//!
//! 2. [`philox_randn_sample`] — the Box-Muller transform of one
//!    Philox4x32-10 block, kept as an internal primitive that is
//!    audited against Random123 KAT vectors but **does NOT match**
//!    real `torch.randn` on any backend (CPU uses MT19937, CUDA uses
//!    `curandStatePhilox4_32_10_t` with different subsequence packing,
//!    MPS uses Metal shader Philox with hi-first `splitlong` and
//!    different constant swaps). The `PhiloxRNGEngine.h::randn`
//!    function this mirrors carries an upstream disclaimer that it is
//!    "not used anywhere except for tests in cpu_generator_test.cpp"
//!    (torch source PhiloxRNGEngine.h:39-41). See the historical note
//!    at the bottom of this file for why the previous "torch parity"
//!    claim was withdrawn.
//!
//! # CPU-torch normal draw contract (what `TorchRandnStream` mirrors)
//!
//! ```cpp
//! // ATen/core/DistributionsHelper.h::normal_distribution<double>::operator()
//! // (paraphrased for clarity, lines 187-198):
//! double sample() {
//!   double u1 = uniform_real<double>();          // consumes random64() #1
//!   double u2 = uniform_real<double>();          // consumes random64() #2
//!   double r     = std::sqrt(-2.0 * std::log1p(-u2));
//!   double theta = 2.0 * M_PI * u1;
//!   // ...cache r * std::sin(theta) into the paired-sample slot...
//!   return r * std::cos(theta);
//! }
//! ```
//!
//! where `uniform_real<double>()` is `((random64() & ((1<<53)-1)) as f64)
//! * (1.0 / (1<<53) as f64)` — the top 53 bits of a hi-first-packed
//! `random64()` word, mapped uniformly onto `[0, 1)` (torch's
//! `TransformationHelper.h:84-90`).
//!
//! # Pair caching and f64→f32 cast
//!
//! Every two Box-Muller-formula evaluations produce two samples (cos
//! and sin of the same `theta`); torch caches the sine in the
//! generator's paired-sample slot and returns it on the next call
//! (identity fold across the u1/u2 draws). The final cast to `f32` is
//! `static_cast<float>(double_result)` — round-to-nearest-even, the
//! same rounding Rust's `as f32` conversion applies.
//!
//! # SCALE / u32_to_uniform_f32_pytorch — legacy pipeline glue
//!
//! [`SCALE`] and [`u32_to_uniform_f32_pytorch`] remain exported because
//! they are used by the pre-existing `PhiloxRNGEngine.h::randn`
//! integration tests (`rng_uniform_transform.rs` +
//! `rng_philox_randn.rs`) — the primitive constant itself is still
//! bit-exact against `constexpr float scale = 4.6566127342e-10f;`
//! upstream and useful for any port of `PhiloxRNGEngine.h` bytes.
//! Their claim to be "torch parity" was withdrawn (see the historical
//! note below); today they are just what they are — an f32-precision
//! uniform bridge that ATen's `PhiloxRNGEngine.h::randn` uses
//! internally, and nothing more.
//!
//! # Historical note (2026-08-08, bisect wf_20fa0933-53d)
//!
//! Pre-`TorchCpuMt19937Parity`, this file's `TorchRandnStream` fed
//! `TorchPhiloxState::next_block()` output through
//! `philox_randn_sample`, in the belief that this reproduced
//! `torch.randn(device='cpu')`. A byte-level bisect against real
//! `torch.randn(4)` seed=0 (u32 samples `[0x3fc53f5c, 0xbe963c50,
//! 0xc00b7149, 0x3f1184b6]`) found NO match at any sample — CPU torch
//! uses `at::mt19937` + `at::normal_distribution<double>`, not Philox.
//! The Philox path was `PhiloxRNGEngine.h`'s `randn`, which is dead
//! code inside torch (its own header disclaims it). This module's
//! `TorchRandnStream` was rewritten to use
//! [`super::mt19937::TorchMt19937Engine`] + f64 Box-Muller with pair
//! caching, and the previous Philox-based primitives were rescoped to
//! be internal utilities (not torch-parity claims).

use super::NormalSource;
use super::mt19937::TorchMt19937Engine;

/// The u32 → uniform-`f32` scale factor from torch's `PhiloxRNGEngine.h`:
/// `4.6566127342e-10f`, which is bit-exactly the f32 with pattern
/// `0x2FFF_FFFF` (verified by `uniform_scale_is_bit_exact_c_constexpr` in
/// `crates/vokra-core/tests/rng_uniform_transform.rs`).
///
/// Not quite `2^-31` (which is bits `0x3000_0000`, a smidge larger). The
/// difference is one ULP at the f32 exponent boundary; torch picked the
/// smaller constant so `u_max * SCALE` cannot round up to 1.0.
///
/// The `excessive_precision` lint here is intentionally allowed: `4.656612_6e-10`
/// (clippy's suggested truncation) rounds to the same f32 bit pattern, but the
/// full torch literal is preserved verbatim so a future maintainer diffing this
/// against upstream `PhiloxRNGEngine.h` sees an obviously identical constant.
///
/// Kept even after the torch-parity rewrite because
/// [`u32_to_uniform_f32_pytorch`] (which uses it) is still exported for
/// the two integration tests that pin its exact behavior (see this
/// module's top-level doc §"SCALE / u32_to_uniform_f32_pytorch —
/// legacy pipeline glue").
#[allow(clippy::excessive_precision)]
pub const SCALE: f32 = 4.6566127342e-10;

/// Maps a `u32` PRNG output to a uniform `f32` in `[0, 1)` using torch's
/// exact bit sequence: mask off the sign bit, cast the low 31 bits to
/// `f32` (a lossless cast because 31 bits fit in the 24-bit mantissa
/// after rounding), multiply by [`SCALE`].
///
/// Bit-exactly matches torch's `uint32_t_to_uniform_float` inside
/// `PhiloxRNGEngine.h`. **Not** used by [`TorchRandnStream`] (which is
/// the MT19937 + `uniform_real<double>` path now); kept for the two
/// legacy integration tests (`rng_uniform_transform.rs`,
/// `rng_philox_randn.rs`) that pin its exact behavior.
#[inline]
#[must_use]
pub fn u32_to_uniform_f32_pytorch(v: u32) -> f32 {
    ((v & 0x7FFF_FFFF) as f32) * SCALE
}

/// Transforms one 128-bit Philox block into ONE standard-normal `f32`
/// sample via the Box-Muller transform torch's `PhiloxRNGEngine.h::randn`
/// uses:
///
/// ```text
/// u1 = 1 - u32_to_uniform_f32(block[0])   // shift (0, 1] so ln is finite
/// u2 = 1 - u32_to_uniform_f32(block[1])
/// r     = sqrt(-2 * ln(u1))
/// theta = 2 * pi * u2
/// return r * cos(theta)
/// ```
///
/// `block[2]` and `block[3]` are DELIBERATELY DISCARDED.
///
/// # Not a torch.randn parity path
///
/// **This does NOT reproduce `torch.randn` on any real torch backend.**
/// CPU torch uses MT19937 + `normal_distribution<double>` (see
/// [`TorchRandnStream`] and [`super::mt19937`] for the actual parity
/// path). The `PhiloxRNGEngine.h::randn` function this mirrors is dead
/// code inside torch — its own header (lines 39-41) states it is "not
/// used anywhere except for tests in cpu_generator_test.cpp".
///
/// Kept as an internal primitive because:
/// (a) it is audited against Random123 KAT vectors via
/// `crates/vokra-core/tests/rng_philox_kat.rs` +
/// `rng_philox_randn.rs`, so a future CUDA `curandStatePhilox4_32_10_t`
/// parity path can build on the same block function once the
/// subsequence/offset packing is settled;
/// (b) removing it would obsolete the KAT tests without a comparable
/// replacement.
#[inline]
#[must_use]
pub fn philox_randn_sample(block: [u32; 4]) -> f32 {
    // Shift the uniform sample from [0, 1) to (0, 1] so `ln(u1)` is finite
    // even for the (astronomically rare) `block[0] == 0` case. If u1 = 1.0
    // exactly (happens only when `block[0] == 0`), `ln(1.0)` = 0 and the
    // sample is 0.
    let u1 = 1.0_f32 - u32_to_uniform_f32_pytorch(block[0]);
    let u2 = 1.0_f32 - u32_to_uniform_f32_pytorch(block[1]);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * core::f32::consts::PI * u2;
    r * theta.cos()
    // block[2] and block[3] intentionally discarded — see doc above.
}

/// 53-bit mask for the mantissa-precision uniform draw
/// (`uniform_real<double>` in torch).
const MASK_53: u64 = (1u64 << 53) - 1;

/// Divisor for the mantissa-precision uniform draw: `2^53`.
/// A f64 with this magnitude fits an exact integer (since the mantissa
/// is 52 bits + implicit leading one = 53), so `MASK_53 as f64 / DIVISOR_53`
/// is representable exactly and the entire uniform mapping is exact-arithmetic.
#[allow(clippy::cast_precision_loss)]
const DIVISOR_53: f64 = (1u64 << 53) as f64;

/// Converts a 64-bit output from [`TorchMt19937Engine::random64`] to a
/// uniform `f64` in `[0, 1)` via ATen's `uniform_real<double>`
/// (TransformationHelper.h:84-90 + DistributionsHelper.h:100-114):
/// mask to 53 bits, divide by `2^53`.
///
/// Kept `#[inline]` and free-standing so it can be used to isolate a
/// bug in the uniform layer without a Box-Muller transform in the
/// same call — a future `torch.rand`-parity API can build on it
/// directly.
#[inline]
#[must_use]
fn u64_to_uniform_f64_torch_cpu(v: u64) -> f64 {
    #![allow(clippy::cast_precision_loss)]
    ((v & MASK_53) as f64) / DIVISOR_53
}

/// A streaming source of standard-normal `f32` samples that byte-matches
/// `torch.manual_seed(N); torch.randn(K, device='cpu')` for `K < 16`
/// (or non-contiguous tensors — see the [`super::mt19937`] module doc
/// for the SIMD `normal_fill` fast path caveat that fires only for
/// `K >= 16` contiguous).
///
/// Wraps a [`TorchMt19937Engine`] plus a `cached_sin: Option<f64>` slot
/// mirroring torch's paired-sample cache — every two Box-Muller
/// evaluations produce a `(cos, sin)` pair, and the sine is cached and
/// returned on the next call. Uses `f64` throughout the transcendental
/// pipeline (sqrt / log1p / sin / cos), casting to `f32` only at the
/// very end, exactly as `at::normal_distribution<double>` does — a
/// naïve `f32` pipeline (as the previous Philox-based implementation
/// used) computes `ln` and `cos` at f32 precision and rounds
/// differently for ~10-20% of inputs, cascading to visible sample
/// divergence.
///
/// Impls [`NormalSource`] so any consumer written against that trait
/// (e.g. `SbV2SDP::sample`) can be constructed with either this stream
/// or the pre-existing `GaussianSplitMix64` synthetic path without
/// touching the fill loop.
#[derive(Clone, Debug)]
pub struct TorchRandnStream {
    engine: TorchMt19937Engine,
    /// Sine half of the current Box-Muller pair; `Some(v)` means the
    /// next `next_f32()` returns `v as f32` without advancing the
    /// engine, exactly as torch's `normal_distribution::operator()`
    /// caches the paired sample.
    cached_sin: Option<f64>,
}

impl TorchRandnStream {
    /// Creates a stream seeded with `seed`, equivalent to
    /// `torch.manual_seed(seed); ... torch.randn(...)` on CPU.
    ///
    /// **Only the low 32 bits of `seed` participate**, matching CPU
    /// torch's own behavior — see [`TorchMt19937Engine::new`]'s doc
    /// for the rationale. The `u64` seed API is caller ergonomics;
    /// MT19937 is a 32-bit generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            engine: TorchMt19937Engine::new(seed),
            cached_sin: None,
        }
    }

    /// Returns the next standard-normal `f32` sample.
    ///
    /// # Algorithm
    ///
    /// 1. If a `cached_sin` is present, return it cast to `f32` and
    ///    clear the slot (matches torch's `normal_distribution` cache
    ///    check).
    /// 2. Otherwise, draw `u1 = uniform_real<double>()` (first
    ///    `random64()`), then `u2 = uniform_real<double>()` (second
    ///    `random64()`).
    /// 3. Compute `r = sqrt(-2 * log1p(-u2))` — the `-log1p(-u2)` form
    ///    is torch's exact primitive; it evaluates to `-ln(1 - u2)`
    ///    but is numerically more stable near `u2 = 0` (where the
    ///    naive `ln(1 - u2)` loses precision because `1 - u2 ≈ 1`).
    /// 4. Compute `theta = 2 * pi * u1`.
    /// 5. Cache `r * sin(theta)` for the next call.
    /// 6. Return `(r * cos(theta)) as f32`.
    ///
    /// All arithmetic is `f64` until the final cast. Bit-exact against
    /// `at::normal_distribution<double>` on CPU, verified by
    /// `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs`.
    pub fn next_f32(&mut self) -> f32 {
        if let Some(v) = self.cached_sin.take() {
            // The paired-sample cast to f32 must happen at the exact
            // moment the value would be returned — deferring it to
            // `self.cached_sin` insertion would round twice for the
            // cosine half.
            return v as f32;
        }
        // Order matters: u1 is drawn FIRST, u2 SECOND, and then u1
        // feeds `theta` while u2 feeds `r` — see this module's
        // top-level doc for the exact upstream contract this
        // reproduces (DistributionsHelper.h:187-198).
        let u1 = u64_to_uniform_f64_torch_cpu(self.engine.random64());
        let u2 = u64_to_uniform_f64_torch_cpu(self.engine.random64());
        // `log1p(-u2)` = ln(1 - u2), numerically stable near u2 = 0.
        // For u2 = 0 exactly this gives `ln(1) = 0`, so `r = 0` — a
        // valid (zero-magnitude) sample that matches torch's edge
        // behavior at the boundary.
        let r = (-2.0_f64 * (-u2).ln_1p()).sqrt();
        let theta = 2.0_f64 * core::f64::consts::PI * u1;
        // Cache the sine half in f64 so the deferred f32 cast rounds
        // once (matching torch's `static_cast<float>` at return time).
        self.cached_sin = Some(r * theta.sin());
        (r * theta.cos()) as f32
    }
}

/// [`TorchRandnStream`] impls the RNG-agnostic [`NormalSource`] boundary
/// so a call site written against the trait can choose between the
/// synthetic `GaussianSplitMix64` and this torch-parity stream at
/// construction time.
impl NormalSource for TorchRandnStream {
    fn next_normal(&mut self) -> f32 {
        self.next_f32()
    }
}

/// Fills `out` with `out.len()` standard-normal `f32` samples from a
/// fresh [`TorchRandnStream::new(seed)`], byte-exactly matching
/// `torch.manual_seed(seed); torch.randn(1, out.len(), device='cpu')`
/// bytes for `out.len() < 16` — verified by the fixture tests in
/// `crates/vokra-core/tests/rng_torch_randn_e2e.rs` at (0, 4), (42,
/// 100), and (12345, 1000).
///
/// The output is C-contiguous little-endian when serialised via
/// `f32::to_le_bytes`.
///
/// # Scope caveat
///
/// Contiguous `out.len() >= 16` calls to `torch.randn` on CPU dispatch
/// to the SIMD `normal_fill` fast path with a different formula and no
/// pair caching (see the [`super::mt19937`] module doc). This function
/// matches the small-N path only; SBV2's SDP noise buffer stays well
/// under 16 elements per timestep (`2 * text_seq_len`, typically 4-20
/// per test phoneme string), so the small-N path is the one that fires
/// in practice.
pub fn torch_randn_f32(seed: u64, out: &mut [f32]) {
    let mut stream = TorchRandnStream::new(seed);
    for v in out {
        *v = stream.next_f32();
    }
}
