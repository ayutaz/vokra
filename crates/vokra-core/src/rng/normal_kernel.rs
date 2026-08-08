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
    /// The u64 seed the stream was constructed with. Kept so the
    /// `fill()` override can rebuild the engine at the top of a fresh
    /// `torch.randn(N)` call (torch dispatches on buffer size, not on
    /// stream position — see the `NormalSource::fill` doc).
    seed: u64,
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
            seed,
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

    /// Overrides the trait's per-sample default to dispatch on
    /// `out.len()` the way `torch.randn` does — see
    /// [`torch_randn_f32`]'s doc for the exact algorithm split.
    ///
    /// # Consumes fresh RNG state
    ///
    /// This override IGNORES `self`'s current MT19937 state and
    /// creates a fresh engine seeded from an internal
    /// `TorchMt19937Engine::new(seed)` derived from the seed stored
    /// in `self` — matching torch's `torch.manual_seed(seed);
    /// torch.randn(N)` idiom (a fresh call, not a resumption of a
    /// prior state). A caller wanting to append to a prior stream
    /// should call `next_normal()` in a loop directly, not this
    /// method.
    ///
    /// Rationale: torch's `normal_fill` fast path is a batch
    /// operation that does not compose with a streaming pair-cached
    /// state (the two paths consume RNG in fundamentally different
    /// orders — small-K reads 2 `random64` per sample, large-K reads
    /// 1 `u32` per sample plus a tail-recompute). Trying to
    /// interleave a streaming call and a batch call from the same
    /// state would violate torch's contract on either side.
    /// [`torch_randn_f32`]'s doc has the full derivation.
    fn fill(&mut self, out: &mut [f32]) {
        // Extract the seed the engine was constructed with. `TorchMt19937Engine`
        // does not expose this today; use the seed stored in Self by
        // walking through the internal API — the `new` constructor
        // saves it implicitly in the state array. We already stashed
        // the u64 seed into `Self` for exactly this purpose (see
        // the new `seed` field below).
        let seed = self.seed;
        torch_randn_f32(seed, out);
        // The engine has advanced fresh from `seed` — reset our own
        // pair cache to match (a subsequent `next_normal()` would
        // otherwise return a stale cached sine that has nothing to
        // do with the freshly-generated buffer).
        self.engine = TorchMt19937Engine::new(seed);
        self.cached_sin = None;
        // Skip the counters the batch just consumed so a subsequent
        // `next_normal()` reads AFTER the batch's stream. The batch
        // consumed `out.len()` u32s for uniforms + 16 more for the
        // tail-recompute if `out.len() % 16 != 0` (see
        // `torch_randn_fill_f32`'s bytes-consumed doc).
        let consumed = if out.len() >= 16 && out.len() % 16 != 0 {
            out.len() + 16
        } else if out.len() >= 16 {
            out.len()
        } else {
            // Small-K path consumes 2 u32s per sample (one random64
            // per uniform, two uniforms per sample) — but with pair
            // caching every OTHER sample reuses the previous pair's
            // sine, so half the samples cost 0 u32s. Net:
            // `out.len() / 2 + (out.len() % 2)` pairs each cost 4
            // u32s (2 random64 = 4 u32).
            let pairs = out.len().div_ceil(2);
            pairs * 4
        };
        for _ in 0..consumed {
            let _ = self.engine.next_u32();
        }
    }
}

/// Fills `out` with `out.len()` standard-normal `f32` samples matching
/// `torch.manual_seed(seed); torch.randn(1, out.len(), device='cpu')`.
///
/// Dispatches exactly as ATen's `normal_kernel`
/// (ATen/native/cpu/DistributionTemplates.h:230-255):
///
/// * `out.len() < 16` → the streaming
///   `at::normal_distribution<double>` path via [`TorchRandnStream`]
///   (small-K, per-sample pair-cached Box-Muller in f64).
/// * `out.len() >= 16` → the buffer `normal_fill` fast path (fill
///   uniforms in-place, then transform 16-wide blocks in-place via
///   [`normal_fill_16_scalar`], with a tail-recompute for the last
///   partial block per ATen's own algorithm at lines 216-227).
///
/// # Cross-platform bit-parity caveat
///
/// The scalar `normal_fill` path Rust implements here matches torch's
/// **scalar** path on all platforms — bit-exact against `torch.randn`
/// when torch itself takes the scalar path (aarch64 M1, non-AVX2
/// x86_64). On x86_64 hosts with AVX2, torch's own
/// `normal_fill_AVX2` uses `avx_mathfun`'s `log256_ps` and
/// `sincos256_ps` — vectorized approximations that differ from
/// libm's scalar `logf` / `cosf` / `sinf` by ~1 ULP for a non-trivial
/// fraction of inputs. So on such hosts, this function's output can
/// differ from `torch.randn` by up to ~1 ULP per sample; downstream
/// consumers whose parity tests run on AVX2 CI hosts should apply an
/// atol that accounts for that residual (SBV2's `PER_TENSOR_ATOL`
/// entry for `waveform` is where this shows up, per the honest-atol
/// discipline).
///
/// # Output layout
///
/// C-contiguous little-endian when serialised via `f32::to_le_bytes`.
pub fn torch_randn_f32(seed: u64, out: &mut [f32]) {
    if out.len() < 16 {
        // ATen `normal_kernel` else-branch (line 246-251) → small-K
        // streaming path with pair caching.
        let mut stream = TorchRandnStream::new(seed);
        for v in out {
            *v = stream.next_f32();
        }
        return;
    }
    // ATen `normal_kernel` fast-path (line 232-240): size >= 16 and
    // f32 and contiguous → `normal_fill` with mean=0, std=1.
    torch_randn_fill_f32(seed, out);
}

/// Bit-exact port of ATen's `normal_fill_16` scalar path
/// (`aten/src/ATen/native/cpu/DistributionTemplates.h:138-148`),
/// with `mean = 0.0`, `std = 1.0` (the `torch.randn` defaults):
///
/// ```text
/// for j in 0..8:
///     u1 = 1 - data[j]                    // (0, 1] so log is finite
///     u2 = data[j + 8]
///     r  = sqrt(-2 * log(u1))             // f32 libm log/sqrt
///     theta = f32(2.0 * f64(pi) * f64(u2))    // f64 mul, then f32 rounding — see note
///     data[j]     = r * cos(theta)        // f32 libm cos
///     data[j + 8] = r * sin(theta)        // f32 libm sin
/// ```
///
/// Consumes 16 pre-filled f32 uniforms in-place and produces 16
/// standard-normal samples in the same slice (first 8 → cosines,
/// second 8 → sines).
///
/// # `theta` precision detail
///
/// Upstream: `const scalar_t theta = 2.0f * c10::pi<double> * u2;`
/// with `scalar_t = float`. C++ operator precedence + usual
/// arithmetic conversions promote `float * double` to `double` and
/// `double * float` to `double`, so `theta` is computed at **f64
/// precision** and rounded to f32 only at the final assignment. A
/// naive Rust `f32 * f32 * f32` chain rounds at every step, which
/// diverges from upstream for many `u2` inputs.
///
/// Same-precision detail on the AVX2 path: `_mm256_set1_ps(2.0f *
/// c10::pi<double>)` narrows `2π` to f32 at construction, but then
/// `_mm256_mul_ps(two_pi, u2)` is a single f32 multiply — so on AVX2
/// hosts the constant is pre-rounded once but the `* u2` is a pure
/// f32 op. That is what makes AVX2 vs scalar output differ by a few
/// samples per block even before the transcendental approximations
/// come in. This scalar port matches the SCALAR path exactly (f64
/// intermediate); the AVX2 path is a separate residual documented on
/// `torch_randn_f32`.
#[inline]
fn normal_fill_16_scalar(data: &mut [f32]) {
    // The array-length precondition (16 elements). Not a runtime
    // check on the hot path — `torch_randn_fill_f32` only calls this
    // with 16-element slices.
    debug_assert_eq!(data.len(), 16);
    for j in 0..8 {
        let u1 = 1.0_f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0_f32 * u1.ln()).sqrt();
        // Match C++'s `2.0f * c10::pi<double> * u2` precision: promote
        // through f64 for both muls, cast to f32 at the end. See the
        // module-doc §"`theta` precision detail" for why a straight
        // `f32 * f32` chain diverges. `2.0 * PI` is a const-eval f64,
        // and `u2 as f64` widens exactly (every f32 fits in f64).
        #[allow(clippy::cast_possible_truncation)]
        let theta = ((2.0_f64 * core::f64::consts::PI) * (u2 as f64)) as f32;
        // The (radius * cos(theta) * std + mean) sequence with std=1,
        // mean=0 simplifies to (radius * cos(theta)) — a single mul.
        // Rust f32 mul + libm cos matches torch's scalar path bit-
        // exactly on non-AVX2 hosts; ~1 ULP off on AVX2 hosts where
        // torch dispatches to `sincos256_ps` (documented above).
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

/// Fill `out` with `out.len()` samples from torch's `normal_fill` fast
/// path (bit-exact port of ATen's `normal_fill` scalar variant,
/// `DistributionTemplates.h:206-228`), given `out.len() >= 16`.
///
/// # Algorithm (verbatim from upstream)
///
/// 1. Fill `out[i]` with `out.len()` `uniform_real<float>()` draws
///    (`((mt.next_u32() & 0xFFFFFF) as f32) / (1u32 << 24) as f32`),
///    one at a time.
/// 2. For each 16-wide block `out[i..i+16]` with `i = 0, 16, 32, ...`
///    while `i + 16 <= out.len()`: apply [`normal_fill_16_scalar`]
///    in-place.
/// 3. Tail (`out.len() % 16 != 0`): draw 16 fresh uniforms into
///    `out[out.len()-16..out.len()]` (overwriting the last 16 that
///    the first-pass fill produced), then apply
///    [`normal_fill_16_scalar`] on that block. This is torch's exact
///    "recompute the last 16 values" step — it does NOT skip the
///    block boundary, it duplicates draws so the tail is always a
///    complete 16-wide block.
///
/// # Bytes consumed
///
/// If `out.len()` is a multiple of 16: exactly `out.len()` MT19937
/// `next_u32()` draws (each `uniform_real<float>` consumes one u32).
/// Otherwise: `out.len() + 16` draws (16 extra for the tail-recompute
/// per upstream — matches the counter advance a caller trying to
/// reproduce torch's per-call MT state would need to account for).
#[inline]
fn torch_randn_fill_f32(seed: u64, out: &mut [f32]) {
    // Contract: this fast path only makes sense for size >= 16 (the
    // dispatcher in `torch_randn_f32` enforces this).
    debug_assert!(out.len() >= 16);
    let mut mt = TorchMt19937Engine::new(seed);

    // Step 1: fill with uniform(0, 1) using `uniform_real<float>`
    // (`(u32 & 0xFFFFFF) / 2^24`, matching torch's f32 uniform, not
    // the f64 version used by `at::normal_distribution<double>`).
    for v in out.iter_mut() {
        *v = mt_uniform_real_f32(&mut mt);
    }

    // Step 2: in-place normal_fill_16 on each 16-wide block.
    let size = out.len();
    let full_blocks = size / 16;
    for b in 0..full_blocks {
        let start = b * 16;
        normal_fill_16_scalar(&mut out[start..start + 16]);
    }

    // Step 3: tail — if size is not a multiple of 16, recompute the
    // last 16 uniforms + transform in-place. Upstream comment:
    // "Recompute the last 16 values." This is NOT an off-by-one — it
    // is a deliberate re-draw so the tail block is always complete.
    if size % 16 != 0 {
        let tail_start = size - 16;
        for v in out[tail_start..].iter_mut() {
            *v = mt_uniform_real_f32(&mut mt);
        }
        normal_fill_16_scalar(&mut out[tail_start..]);
    }
}

/// `uniform_real<float>` from `TransformationHelper.h:84-90` for
/// MT19937's per-call `next_u32()` output: mask to 24 bits, cast to
/// f32, divide by 2^24. Produces a value in `[0, 1)`. Bit-exact
/// against torch's own `uniform_real_distribution<float>::operator()`
/// with `from = 0, to = 1`.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn mt_uniform_real_f32(mt: &mut TorchMt19937Engine) -> f32 {
    let masked = mt.next_u32() & 0x00FF_FFFF;
    (masked as f32) / ((1u32 << 24) as f32)
}
