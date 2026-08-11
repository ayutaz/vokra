//! Smoke check that `vokra_core::rng` remains a valid module path after the
//! Step 0 refactor that converted `rng.rs` → `rng/mod.rs`, and that the three
//! pre-existing public types (`SplitMix64`, `GaussianSplitMix64`,
//! `Xorshift64Star`) are still re-exportable from the module facade.
//!
//! Step 0 is a purely structural change: byte-identical contents are moved to
//! a sub-directory so sibling files (`philox_round.rs`, `philox_state.rs`,
//! `seed_init.rs`, `normal_kernel.rs`) can join it in Steps 1–7. If this file
//! stops compiling, the refactor accidentally changed a public symbol path —
//! `use vokra_core::rng::GaussianSplitMix64` is the sole existing use-site's
//! import (`crates/vokra-models/src/sbv2/duration.rs:58`) so preserving it is
//! the NFR-PT-01 cross-build non-interference contract for this step.
#![allow(unused_imports, dead_code)]

use vokra_core::rng::{GaussianSplitMix64, SplitMix64, Xorshift64Star};

#[test]
fn rng_facade_reexports_pre_existing_public_types() {
    // The `use` above is the assertion; this test body only needs to touch
    // each type once so the compiler cannot dead-code-eliminate the import.
    let _ = SplitMix64::new(0);
    let _ = GaussianSplitMix64::new(0);
    let _ = Xorshift64Star::new(0);
}
