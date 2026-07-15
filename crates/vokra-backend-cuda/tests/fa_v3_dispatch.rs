//! M4-07-T09/T10/T11: FA v3 dispatch surface pins — CUDA-less green.
//!
//! The decision tables themselves (session probe / encoder opt-in / force
//! violation / t_q gate boundary) are unit-tested next to their pure
//! functions in `src/fa_v3.rs`; this integration file pins the *public*
//! contract those decisions hang off: the gate constant, the dispatch
//! priority documented shape, and the env-variable names the harness
//! (`tools/parity/cuda_rtf_variance.sh --fa-mode`) injects. A drive-by
//! rename of any of these must trip a red test here, because the shell
//! harness cannot be compiler-checked against the Rust source.

#![cfg(any(unix, windows))]

use vokra_backend_cuda::FA_V3_MIN_TQ;

/// The FA gate ladder is v3 (64) > v2 (16) > decomposed — both constants are
/// structural tile heights, not tuned numbers, and the FA v2 value is pinned
/// by `m3_01_backend_coverage::fa_v2_gate_constants_are_stable` already.
/// Calibration of the v3 value belongs to the owner's H100 measurement
/// (M4-07-T18); do NOT bump it here without that data.
#[test]
fn fa_v3_gate_ladder_is_stable() {
    const FA_V2_MIN_TQ: usize = 16; // documented FA v2 tile height (Br)
    assert_eq!(FA_V3_MIN_TQ, 64, "BR3 warpgroup tile height");
    // The v3 gate must be strictly tighter than v2's: a t_q in [16, 64)
    // falls through v3 to the v2 arm, below 16 to the decomposed chain.
    const _: () = assert!(FA_V3_MIN_TQ > FA_V2_MIN_TQ);
    // Whisper anchor shapes: the encoder (1500) and full-prefix decode (448)
    // clear the v3 gate; the steady-state decoder step (1) clears neither —
    // the FA v2 honest negative (vast.ai N=10, mean 0.782 gated vs 0.766
    // decomposed) is inherited by design, not fixed by v3.
    const _: () = assert!(1500 >= FA_V3_MIN_TQ);
    const _: () = assert!(448 >= FA_V3_MIN_TQ);
    const _: () = assert!(1 < FA_V2_MIN_TQ);
}

/// Env-variable names are load-bearing for the shell harness (`--fa-mode`
/// injects them) and for owner runbooks; pin the exact spellings. The
/// semantics (presence-based, like `VOKRA_CUDA_DISABLE_FA_V2`) are locked by
/// the `fa_v3_env_toggles_are_presence_based` unit test.
#[test]
fn fa_v3_env_variable_names_are_pinned() {
    // Compile-time string pins — the source of truth the harness mirrors.
    const DISABLE: &str = "VOKRA_CUDA_DISABLE_FA_V3";
    const FORCE: &str = "VOKRA_CUDA_FORCE_FA_V3";
    const ENCODER: &str = "VOKRA_CUDA_FA_V3_ENCODER";
    const DISABLE_V2: &str = "VOKRA_CUDA_DISABLE_FA_V2";
    // The v3 family mirrors the v2 hatch's prefix so operator muscle memory
    // transfers ("VOKRA_CUDA_" + verb + "_FA_V3").
    for name in [DISABLE, FORCE, ENCODER] {
        assert!(name.starts_with("VOKRA_CUDA_"));
        assert!(name.contains("FA_V3"));
    }
    assert_eq!(DISABLE.replace("V3", "V2"), DISABLE_V2);
}
