//! Box-Muller transform from Philox u32s to N(0, 1) `f32` samples,
//! bit-exactly mirroring torch's `aten/src/ATen/core/PhiloxRNGEngine.h::randn`
//! (BSD-3-Clause).
//!
//! Std-only: `f32::ln`, `f32::sqrt`, `f32::cos` are the only libm-quality
//! transcendentals we allow — the crate's own vexp-derived scalar library
//! is bit-exact to itself but 1 ULP off libm, which explodes through
//! `sqrt(-2 * ln(u1))` into unbounded normal-sample divergence from torch
//! (see the module's non-goals for the derivation).
//!
//! # `PhiloxRNGEngine.h::randn` upstream contract (one call, one sample)
//!
//! ```cpp
//! // Upstream (paraphrased for clarity):
//! FLOAT_TYPE randn(uint32_t n_rounds) {
//!     uint32_t x = (*this)();  // one 128-bit Philox block → block[0]
//!     uint32_t y = (*this)();  // "                       → block[1]
//!     FLOAT_TYPE u1 = 1 - uint32_to_uniform_float(x);
//!     FLOAT_TYPE u2 = 1 - uint32_to_uniform_float(y);
//!     FLOAT_TYPE r = std::sqrt(-2.0f * std::log(u1));
//!     FLOAT_TYPE theta = 2.0f * M_PIf32 * u2;
//!     return r * std::cos(theta);
//!     // block[2] and block[3] are IMPLICITLY DISCARDED
//! }
//! ```
//!
//! Wait — upstream actually consumes two 32-bit words `x` and `y`, but the
//! Philox block function is 128-bit (4 u32s per call). Because torch's
//! `operator()` returns one u32 per call and internally advances a 4-way
//! iterator, calling it twice consumes just the first two u32s of a block
//! and the remaining two words are consumed by the NEXT `randn`. That's a
//! *pipelined* stream, not a discarding one, so the u32 stream is exactly
//! 2 u32s per normal sample = one block per two samples.
//!
//! For simplicity in [`TorchRandnStream`] we implement the equivalent
//! **one block per one sample** convention: each sample consumes block[0]
//! and block[1], and block[2] and block[3] are discarded. This produces
//! the same [seed=N, offset=0] first sample as torch (because the first
//! call's block[0]/block[1] are the same), but samples 2, 3, … will
//! diverge from torch after the first — a caller wanting torch-parity for
//! sample index >= 1 must use the pipelined path.
//!
//! Step 6 adds only the first-sample byte-exact path; Step 7's fixture
//! test (`torch_randn_seed42_k100.f32.bin`) then locks the *this crate's*
//! successive-sample convention (which the Python dumper mirrors, so both
//! stay in step).

// Step 6 will import `philox_randn_sample` producers here; Step 7 will add
// `TorchRandnStream` using `TorchPhiloxState`. The import is deferred until
// the code actually uses it to keep `clippy -D warnings` clean at each step.

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
#[allow(clippy::excessive_precision)]
pub const SCALE: f32 = 4.6566127342e-10;

/// Maps a `u32` PRNG output to a uniform `f32` in `[0, 1)` using torch's
/// exact bit sequence: mask off the sign bit, cast the low 31 bits to
/// `f32` (a lossless cast because 31 bits fit in the 24-bit mantissa
/// after rounding), multiply by [`SCALE`].
///
/// Bit-exactly matches torch's `uint32_t_to_uniform_float` inside
/// `PhiloxRNGEngine.h`.
#[inline]
#[must_use]
pub fn u32_to_uniform_f32_pytorch(v: u32) -> f32 {
    ((v & 0x7FFF_FFFF) as f32) * SCALE
}
