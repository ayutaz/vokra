//! Placeholder smoke test that keeps the `parity` CI check green until the
//! real parity suites land (M0-04/05/06/07) — see `tests/parity/README.md`
//! and M0-01-T13.

/// FP32 absolute tolerance mandated by NFR-QL-01.
const FP32_ATOL: f64 = 0.01;

#[test]
fn parity_harness_smoke() {
    // Doubles as a check of the workspace numeric parsing policy
    // (NFR-RL-01): Rust's `str::parse` is locale-independent, unlike the
    // banned C `strtod`.
    let parsed: f64 = "0.01".parse().expect("locale-independent float parsing");
    assert!((parsed - FP32_ATOL).abs() < f64::EPSILON);
}
