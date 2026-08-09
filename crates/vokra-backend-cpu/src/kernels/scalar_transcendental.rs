//! Backward-compat re-export shim over [`vokra_math`] (WP-07).
//!
//! # What moved and why
//!
//! Before WP-07 the scalar `exp` / `tanh` / `sqrt` / `sin` / `cos` / `log` /
//! `log1p` primitives lived here in-line (Wave 1 for `exp`/`tanh`/`sqrt`,
//! M5-03-T06; WP-06 added the sin/cos/log/log1p family for the
//! Style-Bert-VITS2 v2 hot path, commit `2f1696c`). Non-backend crates —
//! `vokra-ops::hifigan`, `vokra-bert::deberta_v2`, and future SBV2 kernels —
//! also need these primitives, but they cannot depend on
//! `vokra-backend-cpu`: that edge would flip the crate graph inside-out
//! (backend on top of ops, not the other way round) and pull the full CPU
//! kernel tier — AVX-512, dotprod / i8mm, K-quants, thread pool — into every
//! op crate.
//!
//! WP-07 hoists the implementations into the new tiny [`vokra_math`] crate
//! whose only dependency is `core` (zero external deps, no `vokra-core`,
//! no `vokra-*` kernel crate). This module is now a thin `pub(crate) use`
//! re-export shim so any existing internal caller reaching them via
//! `crate::kernels::scalar_transcendental::{…}` keeps compiling unchanged.
//! New callers should name [`vokra_math`] directly instead of routing
//! through this module — no new caller inside `vokra-backend-cpu` should
//! grow here.
//!
//! # Property tests / accuracy bounds
//!
//! The dense-sweep property tests (`EXP_REL_TOL`, `TANH_ABS_TOL`,
//! `SQRT_REL_TOL`, `SIN_ABS_TOL`, `COS_ABS_TOL`, `LOG_REL_TOL`,
//! `LOG1P_ABS_TOL`) moved with the implementations to `vokra-math`'s own
//! `#[cfg(test)]` module and continue to run under
//! `cargo test -p vokra-math`. Nothing was silently loosened in the move
//! (workspace red line #3: bounds are architectural — chosen at ~2× the
//! observed max, well inside NFR-QL-01 `atol=0.01` — never widened to force
//! a pass).

// Backward-compat: keep the existing path resolvable for anything inside
// `vokra-backend-cpu` still writing `crate::kernels::scalar_transcendental::exp`.
// `pub(crate)` mirrors the pre-WP-07 visibility exactly, so nothing outside
// the crate gains new API surface through this shim (the fresh public API is
// `vokra_math::*` at the crate boundary).
#[allow(unused_imports)] // callers migrate incrementally to `vokra_math::*`.
pub(crate) use vokra_math::{cos, exp, log, log1p, sin, sqrt, tanh};

#[cfg(test)]
mod tests {
    //! Guard that the shim actually forwards to `vokra-math`, not to any local
    //! shadow that might sneak in. Cheap: the numeric property bounds are
    //! owned by `vokra-math`; this only pins that the wire still leads there
    //! (a linker / re-export canary).

    use super::*;

    #[test]
    fn shim_forwards_to_vokra_math() {
        // Identity fixed points. If the re-export ever accidentally rebinds
        // to a local stub, at least one of these would drift.
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(sqrt(0.0), 0.0);
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(log(1.0), 0.0);
        assert_eq!(log1p(0.0), 0.0);
        // And the source is definitely `vokra_math`, not a shadow module: a
        // direct call at the crate boundary gives the same answer as the
        // re-exported one (`exp` is deterministic bit-for-bit, so `==` here
        // is meaningful — not an approximate compare).
        assert_eq!(exp(0.5), vokra_math::exp(0.5));
    }
}
