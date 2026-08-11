//! WP-07 wiring pin: `vokra-ops` (and any other first-party crate) can call the
//! shared `vokra-math` transcendental primitives WITHOUT depending on
//! `vokra-backend-cpu`.
//!
//! # Why this test exists
//!
//! Before WP-07 the scalar `exp` / `tanh` / `sqrt` / `sin` / `cos` / `log` /
//! `log1p` primitives lived in
//! `crates/vokra-backend-cpu/src/kernels/scalar_transcendental.rs` (M5-03-T06
//! Wave 1 for `exp`/`tanh`/`sqrt`; WP-06 for the sin/cos/log/log1p family).
//! Callers outside `vokra-backend-cpu` — the SBV2 hot path (WP-05 owner
//! decision 2026-08-09), the HiFi-GAN vocoder in `vokra-ops::hifigan`, the
//! DeBERTa v2 BERT encoder in `vokra-bert::deberta_v2` — could not reach
//! those primitives without introducing a `vokra-ops -> vokra-backend-cpu`
//! dependency edge, which would flip the crate graph inside-out (backend on
//! top of ops, not the other way round) and pull the full CPU kernel tier
//! (AVX-512, dotprod / i8mm, K-quants, thread pool …) into every op crate.
//!
//! WP-07 extracts the primitives into the new tiny `vokra-math` crate whose
//! only dep is `core` (zero external deps, no `vokra-core`, no `vokra-*`
//! kernel crate). Any first-party crate can now depend on `vokra-math` and
//! call the primitives directly.
//!
//! This test proves the wiring: `vokra-ops` `[dev-dependencies]` lists
//! `vokra-math`, this test imports it, and the calls resolve. Under TDD
//! discipline it started RED (the crate did not exist) and turned GREEN when
//! the crate was created.
//!
//! Numerical accuracy is enforced by `vokra-math`'s own property tests (the
//! dense-sweep bounds `EXP_REL_TOL`, `SIN_ABS_TOL`, `LOG_REL_TOL`, etc.); this
//! test only asserts each function is *callable* and returns the
//! textbook fixed points — a linker / API smoke, not a numeric oracle.

#[test]
fn vokra_math_is_callable_from_a_non_backend_crate() {
    // exp(0) = 1 exactly (0 * anything = 0, Horner sums to `C0 = 1`).
    assert_eq!(vokra_math::exp(0.0), 1.0);
    // exp(0.5) is the specific call named in the WP-07 brief; a smoke check
    // against std bounds the answer without pinning ULP (that is
    // vokra-math's own job).
    let e_half = vokra_math::exp(0.5);
    assert!(
        (e_half - 0.5_f32.exp()).abs() < 1e-6,
        "vokra_math::exp(0.5) = {e_half}, std::f32::exp(0.5) = {}",
        0.5_f32.exp()
    );

    // The other WP-06 primitives are reachable from the same path.
    assert_eq!(vokra_math::tanh(0.0), 0.0);
    assert_eq!(vokra_math::sqrt(0.0), 0.0);
    assert_eq!(vokra_math::sin(0.0), 0.0);
    assert_eq!(vokra_math::cos(0.0), 1.0);
    assert_eq!(vokra_math::log(1.0), 0.0);
    assert_eq!(vokra_math::log1p(0.0), 0.0);
}
