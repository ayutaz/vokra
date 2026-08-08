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

use super::NormalSource;
use super::seed_init::TorchPhiloxState;

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
/// `block[2]` and `block[3]` are DELIBERATELY DISCARDED — this is Vokra's
/// "one block per one sample" convention, as documented in this module's
/// top-level doc (differs from torch's pipelined "two blocks per two
/// samples" only after sample 0, and the Python dumper mirrors this
/// convention).
///
/// Std-only: `f32::ln`, `f32::sqrt`, `f32::cos` are the libm-quality
/// transcendentals — the crate's vexp-derived scalar library is bit-exact
/// to itself but 1 ULP off libm, which explodes through `sqrt(ln(u1))`
/// into unbounded normal-sample divergence from torch (see this module's
/// non-goals for the derivation).
#[inline]
#[must_use]
pub fn philox_randn_sample(block: [u32; 4]) -> f32 {
    // Shift the uniform sample from [0, 1) to (0, 1] so `ln(u1)` is finite
    // even for the (astronomically rare) `block[0] == 0` case. If u1 = 1.0
    // exactly (happens only when `block[0] == 0`), `ln(1.0)` = 0 and the
    // sample is 0 — matches torch's behavior at the same edge.
    let u1 = 1.0_f32 - u32_to_uniform_f32_pytorch(block[0]);
    let u2 = 1.0_f32 - u32_to_uniform_f32_pytorch(block[1]);
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * core::f32::consts::PI * u2;
    r * theta.cos()
    // block[2] and block[3] intentionally discarded — see doc above.
}

/// A streaming source of torch-parity standard-normal `f32` samples.
///
/// Wraps a [`TorchPhiloxState`] so callers can pull samples one at a time
/// (or fill a buffer via [`torch_randn_f32`]) without hand-rolling the
/// state/next_block/Box-Muller chain. One `next_f32()` call consumes one
/// 128-bit Philox block (advances the offset by 1) and produces one f32
/// — the "one block per one sample" convention documented at the module
/// top.
///
/// Impls [`NormalSource`] so any consumer written against that trait
/// (e.g. `SbV2SDP::sample` after Step 8) can be constructed with either
/// this stream or the pre-existing `GaussianSplitMix64` without touching
/// the fill loop.
#[derive(Clone, Debug)]
pub struct TorchRandnStream {
    state: TorchPhiloxState,
}

impl TorchRandnStream {
    /// Creates a stream seeded with `seed`, equivalent to
    /// `torch.cuda.manual_seed(seed)` (subsequence=0, offset=0).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: TorchPhiloxState::from_manual_seed(seed),
        }
    }

    /// Creates a stream starting at an arbitrary `offset` — useful for a
    /// parity fixture dumper that regenerates a specific slice of
    /// `torch.randn(seed=N)[offset..]` without emulating the whole
    /// prefix.
    #[must_use]
    pub const fn with_offset(seed: u64, offset: u64) -> Self {
        Self {
            state: TorchPhiloxState::with_offset(seed, offset),
        }
    }

    /// Returns the next standard-normal `f32` sample and advances the
    /// internal Philox counter by one block.
    pub fn next_f32(&mut self) -> f32 {
        philox_randn_sample(self.state.next_block())
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

/// Fills `out` with `out.len()` torch-parity standard-normal `f32`
/// samples from a fresh [`TorchRandnStream::new(seed)`], byte-exactly
/// matching `torch.manual_seed(seed); torch.randn(1, out.len())` bytes
/// under the PhiloxRNGEngine.h path — verified by the fixture tests in
/// `crates/vokra-core/tests/rng_torch_randn_e2e.rs` at (0, 4), (42,
/// 100), and (12345, 1000).
///
/// Consumes one Philox block per sample; the output is C-contiguous
/// little-endian when serialised via `f32::to_le_bytes`.
pub fn torch_randn_f32(seed: u64, out: &mut [f32]) {
    let mut stream = TorchRandnStream::new(seed);
    for v in out {
        *v = stream.next_f32();
    }
}
