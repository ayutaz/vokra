//! AVX-VNNI 256-bit INT8 group-sum core (M4-17-T13).
//!
//! `vpdpbusd` in its VEX (256-bit) encoding — Alder Lake 2021+, present on
//! both P- and E-cores. **The client INT8 main path**: Intel client parts
//! since Alder Lake fuse AVX-512 off platform-wide, so a hybrid CPU probes as
//! `avx512* = false, avxvnni = true` and lands on this tier (ADR M4-17
//! §(d)). Numerically it is the same computation as the 512-bit VNNI core in
//! [`super::avx512`] at half the width: exact per-16-element-group i32 sums
//! consumed by [`super::kquant`]'s shared combine — bit-identical to the
//! scalar-int8 reference by integer exactness (ADR M4-17 §(e)).
//!
//! The `_mm256_dpbusd_avx_epi32` intrinsic (std_detect feature `"avxvnni"`)
//! is stable on rustc 1.95 (verified by compile probe), so no inline asm is
//! needed on x86-64. Unsafe boundary as in [`super::avx2`] (NFR-RL-07).

use core::arch::x86_64::*;

/// # Safety
/// Requires `avx2,avxvnni`; `q.len() == x.len()`, a multiple of 32,
/// `sums.len() * 16 == q.len()`.
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn vnni256_group_sums_impl(q: &[u8], x: &[i8], sums: &mut [i32]) {
    // SAFETY: `avxvnni` (+AVX2) guaranteed by the caller's `supports` gate.
    // `base + 32 <= q.len() == x.len()` bounds every 32-byte load; the 8 i32
    // lanes are stored to a stack buffer and folded per 4 lanes into
    // `sums[2 * blk + g]`, in bounds by the length contract.
    unsafe {
        debug_assert_eq!(q.len(), x.len());
        debug_assert_eq!(q.len() % 32, 0);
        debug_assert_eq!(sums.len() * 16, q.len());
        let mut blk = 0;
        let mut base = 0;
        while base + 32 <= q.len() {
            let qv = _mm256_loadu_si256(q[base..].as_ptr() as *const _);
            let xv = _mm256_loadu_si256(x[base..].as_ptr() as *const _);
            // vpdpbusd (VEX): unsigned bytes (weights, 0..=63) × signed bytes
            // (activations) accumulated per 4-byte dword into 8 i32 lanes.
            let dp = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), qv, xv);
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr() as *mut _, dp);
            // Each 16-byte group spans 4 consecutive dword lanes (2 groups
            // per 256-bit block); integer adds are exact.
            for g in 0..2 {
                sums[2 * blk + g] =
                    lanes[4 * g] + lanes[4 * g + 1] + lanes[4 * g + 2] + lanes[4 * g + 3];
            }
            blk += 1;
            base += 32;
        }
    }
}

/// AVX-VNNI 256-bit per-group INT8 dot sums (M4-17-T13): `sums[g]` receives
/// `Σ_{t<16} q[16g+t] · x[16g+t]` as an exact i32.
///
/// Caller contract (checked): `q.len() == x.len()`, a multiple of 32,
/// `sums.len() * 16 == q.len()` (one K-quant super-block = 256 bytes = 8
/// ymm loads).
pub(crate) fn vnni256_group_sums(q: &[u8], x: &[i8], sums: &mut [i32]) {
    assert_eq!(q.len(), x.len(), "vnni256_group_sums length mismatch");
    assert_eq!(q.len() % 32, 0, "vnni256_group_sums needs whole ymm blocks");
    assert_eq!(sums.len() * 16, q.len(), "vnni256_group_sums sums mismatch");
    // SAFETY: reached only after `CpuFeatures::supports(AvxVnni256)` (the
    // caller in `super::kquant` gates on it); lengths asserted above.
    unsafe { vnni256_group_sums_impl(q, x, sums) }
}
