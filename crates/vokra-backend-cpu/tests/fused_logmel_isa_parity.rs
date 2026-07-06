//! M2-04-T06: Scalar / AVX2 (/ NEON — separate ticket) parity for the fused
//! log-mel inner kernel.
//!
//! Cross-checks the AVX2 fused-log-mel kernel (mel accumulation via
//! `_mm256_fmadd_ps` + `vlog10_avx2` polynomial approximation) against the
//! portable scalar reference bundled in the same module. Because the SIMD
//! kernels compute the same op with the same result within FP32 rounding —
//! this is *within-CPU-backend* dispatch, not the cross-backend fallback rule
//! FR-EX-08 — the tolerance is the ULP-level SIMD-log10 approximation error,
//! well inside the FP32 parity ceiling NFR-QL-01 `atol = 0.01`.
//!
//! NEON parity is delivered by the companion NEON ticket; this file exercises
//! only Scalar / AVX2 on x86-64. On non-x86 hosts, or on an x86-64 host that
//! lacks AVX2 + FMA, the AVX2 sub-tests short-circuit (never a silent pass:
//! the run prints an explicit `skip:` diagnostic).
//!
//! The kernel is invoked through the crate-internal `pub(crate)` symbol via
//! the integration test's placement inside the same crate build unit; on
//! non-x86 targets the AVX2 module is not compiled in and the file collapses
//! to a scalar-only smoke test.

#![cfg(target_arch = "x86_64")]

// Reach the `pub(crate)` symbols by re-exporting them through a probe module
// living at the crate root when tests build the library. This integration
// test file relies on public re-exports added to the library for
// test-instrumentation; those live below the crate's private tree.
use vokra_backend_cpu::fused_logmel_test_probe::{
    fused_logmel_apply_frame_avx2 as avx2, fused_logmel_apply_frame_scalar as scalar,
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
    // Uniform in [0, 1] for filterbank-like weights.
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
/// eight-lane vector chunk (`24 * 8 = 192`) and the nine-element scalar tail,
/// which is the exact ragged-tail case the AVX2 kernel is required to handle
/// bit-for-bit like the scalar reference.
#[test]
fn parity_whisper_shape_scalar_vs_avx2() {
    if !(std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
    {
        eprintln!("skip: host lacks avx2+fma; AVX2 parity check bypassed");
        return;
    }
    let mut rng = Rng::new(0xF10E_10E1);
    let (n_mels, n_bins) = (80usize, 201usize);
    let weights = rng.weights(n_mels, n_bins);
    let power = rng.vec(n_bins);
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_a = vec![0.0f32; n_mels];
    scalar(&weights, &power, n_mels, n_bins, 1e-10, &mut out_s);
    avx2(&weights, &power, n_mels, n_bins, 1e-10, &mut out_a);
    let mut max_abs = 0.0f32;
    for (i, (&s, &a)) in out_s.iter().zip(&out_a).enumerate() {
        let d = (s - a).abs();
        if d > max_abs {
            max_abs = d;
        }
        // Plan-spec atol=1e-5: SIMD log10 approximation ceiling (well under
        // FP32 NFR-QL-01 atol=0.01).
        assert!(
            d < 1e-5,
            "mel {i}: scalar={s}, avx2={a}, |Δ|={d} exceeds atol=1e-5"
        );
    }
    // Bookkeeping print — visible with `--nocapture`.
    eprintln!("max |Δ| = {max_abs:e} (n_mels={n_mels}, n_bins={n_bins})");
}

/// Silence — all-zero power spectrogram must clamp to `log10(floor)` in
/// every mel bin, and the AVX2 path must produce the same finite value as
/// the scalar path.
#[test]
fn parity_silence_clamps_to_floor_on_both_paths() {
    if !(std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
    {
        eprintln!("skip: host lacks avx2+fma");
        return;
    }
    let mut rng = Rng::new(0xDEAD_BEEF);
    let (n_mels, n_bins) = (80usize, 201usize);
    let weights = rng.weights(n_mels, n_bins);
    let power = vec![0.0f32; n_bins];
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_a = vec![0.0f32; n_mels];
    let floor = 1e-10f32;
    scalar(&weights, &power, n_mels, n_bins, floor, &mut out_s);
    avx2(&weights, &power, n_mels, n_bins, floor, &mut out_a);
    let want = floor.log10();
    for (i, (&s, &a)) in out_s.iter().zip(&out_a).enumerate() {
        assert!(s.is_finite(), "scalar mel {i} not finite: {s}");
        assert!(a.is_finite(), "avx2 mel {i} not finite: {a}");
        assert!((s - want).abs() < 1e-6, "scalar mel {i} = {s}, want {want}");
        assert!(
            (s - a).abs() < 1e-5,
            "silence mel {i}: scalar={s} vs avx2={a}"
        );
    }
}

/// Small tail-only case: n_bins = 5 (< 8) hits *only* the scalar tail of the
/// AVX2 dot product, so the AVX2 kernel must not skip work when the vector
/// chunk count is zero.
#[test]
fn parity_scalar_tail_only() {
    if !(std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
    {
        eprintln!("skip: host lacks avx2+fma");
        return;
    }
    let (n_mels, n_bins) = (16usize, 5usize);
    let mut rng = Rng::new(0x1234_5678);
    let weights = rng.weights(n_mels, n_bins);
    let power = rng.vec(n_bins);
    let mut out_s = vec![0.0f32; n_mels];
    let mut out_a = vec![0.0f32; n_mels];
    scalar(&weights, &power, n_mels, n_bins, 1e-10, &mut out_s);
    avx2(&weights, &power, n_mels, n_bins, 1e-10, &mut out_a);
    for (i, (&s, &a)) in out_s.iter().zip(&out_a).enumerate() {
        assert!(
            (s - a).abs() < 1e-5,
            "tail-only mel {i}: scalar={s} vs avx2={a}"
        );
    }
}
