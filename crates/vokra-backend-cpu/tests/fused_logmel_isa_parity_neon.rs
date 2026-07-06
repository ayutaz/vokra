//! M2-04-T06 (NEON): Scalar / NEON parity for the fused log-mel inner kernel.
//!
//! Companion to `tests/fused_logmel_isa_parity.rs` (AVX2) — exercises the
//! four-lane `vfmaq_f32` mel accumulation plus `vlog10_neon` polynomial
//! approximation on AArch64. NEON is the ARMv8-A baseline so there is no
//! runtime feature-detect guard (CLAUDE.md "NEON (ARMv8-A baseline、常時対応)");
//! the file is `#![cfg(target_arch = "aarch64")]`-gated and collapses to
//! empty on non-AArch64 hosts.
//!
//! Tolerance is `atol=1e-5` (SIMD-log10 approximation ceiling), matching the
//! plan-spec bound and well inside the FP32 NFR-QL-01 parity ceiling
//! `atol=0.01`. This is *within-CPU-backend* dispatch, not the cross-backend
//! fallback rule FR-EX-08 — scalar / AVX2 / NEON compute the same op with the
//! same result within FP32 rounding.

#![cfg(target_arch = "aarch64")]

use vokra_backend_cpu::fused_logmel_test_probe_neon::{
    fused_logmel_apply_frame_neon as neon, fused_logmel_apply_frame_scalar as scalar,
};

/// Deterministic xorshift64* — no `rand` dependency (NFR-DS-02).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        // Positive spectrum-like magnitudes in [0, 1e4): the log-mel input is
        // always a non-negative power spectrogram.
        (bits as f32 / (1u32 << 24) as f32) * 1e4
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
    fn next_unit(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }
    fn weights(&mut self, n_mels: usize, n_bins: usize) -> Vec<f32> {
        (0..n_mels * n_bins).map(|_| self.next_unit()).collect()
    }
}

/// Whisper's log-mel geometry (n_mels=80, n_bins=201) — the actual shape the
/// production kernel sees at run time. The bin count `201` triggers both the
/// four-lane vector chunk (`50 * 4 = 200`) and the one-element scalar tail,
/// which is the exact ragged-tail case the NEON kernel is required to handle
/// bit-close to the scalar reference.
#[test]
fn parity_whisper_shape_scalar_vs_neon() {
    let mut rng = Rng::new(0xF10E_10E1);
    let (n_mels, n_bins) = (80usize, 201usize);
    let weights = rng.weights(n_mels, n_bins);
    let power = rng.vec(n_bins);
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_n = vec![0.0f32; n_mels];
    scalar(&weights, &power, n_mels, n_bins, 1e-10, &mut out_s);
    neon(&weights, &power, n_mels, n_bins, 1e-10, &mut out_n);
    let mut max_abs = 0.0f32;
    for (i, (&s, &a)) in out_s.iter().zip(&out_n).enumerate() {
        let d = (s - a).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d < 1e-5,
            "mel {i}: scalar={s}, neon={a}, |Δ|={d} exceeds atol=1e-5"
        );
    }
    eprintln!("max |Δ| = {max_abs:e} (n_mels={n_mels}, n_bins={n_bins})");
}

/// Silence — all-zero power spectrogram must clamp to `log10(floor)` in every
/// mel bin, and the NEON path must produce the same finite value as the
/// scalar path.
#[test]
fn parity_silence_clamps_to_floor_on_both_paths() {
    let mut rng = Rng::new(0xDEAD_BEEF);
    let (n_mels, n_bins) = (80usize, 201usize);
    let weights = rng.weights(n_mels, n_bins);
    let power = vec![0.0f32; n_bins];
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_n = vec![0.0f32; n_mels];
    let floor = 1e-10f32;
    scalar(&weights, &power, n_mels, n_bins, floor, &mut out_s);
    neon(&weights, &power, n_mels, n_bins, floor, &mut out_n);
    let want = floor.log10();
    for (i, (&s, &a)) in out_s.iter().zip(&out_n).enumerate() {
        assert!(s.is_finite(), "scalar mel {i} not finite: {s}");
        assert!(a.is_finite(), "neon mel {i} not finite: {a}");
        assert!((s - want).abs() < 1e-6, "scalar mel {i} = {s}, want {want}");
        assert!(
            (s - a).abs() < 1e-5,
            "silence mel {i}: scalar={s} vs neon={a}"
        );
    }
}

/// Small tail-only case: n_bins = 3 (< 4) hits *only* the scalar tail of the
/// NEON dot product, so the NEON kernel must not skip work when the vector
/// chunk count is zero.
#[test]
fn parity_scalar_tail_only() {
    let (n_mels, n_bins) = (16usize, 3usize);
    let mut rng = Rng::new(0x1234_5678);
    let weights = rng.weights(n_mels, n_bins);
    let power = rng.vec(n_bins);
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_n = vec![0.0f32; n_mels];
    scalar(&weights, &power, n_mels, n_bins, 1e-10, &mut out_s);
    neon(&weights, &power, n_mels, n_bins, 1e-10, &mut out_n);
    for (i, (&s, &a)) in out_s.iter().zip(&out_n).enumerate() {
        assert!(
            (s - a).abs() < 1e-5,
            "tail-only mel {i}: scalar={s} vs neon={a}"
        );
    }
}
