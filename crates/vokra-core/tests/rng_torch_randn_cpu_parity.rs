//! **Byte-exact CPU `torch.randn` parity anchor** — the load-bearing test
//! that pins Vokra's [`vokra_core::rng::TorchRandnStream`] to the exact
//! bit pattern real `torch.randn(4, device='cpu')` produces at seed=0.
//!
//! # Why this test exists
//!
//! Before the rewrite documented in
//! `crates/vokra-core/src/rng/normal_kernel.rs`'s "Historical note",
//! `TorchRandnStream` drove `TorchPhiloxState::next_block()` through a
//! Philox4x32-10 + f32 Box-Muller pipeline in the belief that this
//! reproduced `torch.randn(device='cpu')`. A byte-level bisect
//! (`wf_20fa0933-53d`) found that CPU torch is actually MT19937 +
//! `at::normal_distribution<double>`, and the Philox path was `PhiloxRNGEngine.h`'s
//! own `randn`, which the torch header itself disclaims as "not used
//! anywhere except for tests in cpu_generator_test.cpp".
//!
//! The other RNG integration tests
//! (`rng_torch_randn_e2e.rs`, `rng_philox_randn.rs`,
//! `sbv2_sdp_torch_parity.rs`) diff Rust against a Python port of the
//! same algorithm — so a shared-bug hazard would let both sides drift
//! together and every fixture would "pass" while diverging from real
//! torch. This test breaks that hazard by pinning the **exact 16 raw
//! bytes** real `torch.randn` produces on a documented (seed, k) point,
//! without a Python round-trip anywhere in the loop:
//!
//! ```python
//! import torch, struct
//! torch.manual_seed(0)
//! print([hex(struct.unpack('<I', struct.pack('<f', v))[0]) for v in
//!        torch.randn(4).tolist()])
//! # ['0x3fc53f5c', '0xbe963c50', '0xc00b7149', '0x3f1184b6']
//! ```
//!
//! (Values also copied into the bisect report at the top of this file's
//! rationale.) If a future refactor of `TorchRandnStream` drifts even
//! one ULP, this test fails immediately — it does not require any HF
//! model download, any Python venv, or any CI-only fixture.
//!
//! # Scope caveat inherited from `TorchRandnStream`
//!
//! CPU `torch.randn(K)` for `K < 16` (or non-contiguous tensors)
//! matches this path exactly. Contiguous `K >= 16` dispatches to
//! torch's SIMD `normal_fill` fast path with a *different* formula and
//! no pair caching — the SBV2 SDP noise buffer stays well under 16
//! elements per timestep (`2 * text_seq_len`), so the small-K path is
//! the one that fires in practice.

// The four `expected_values` entries reproduce Python `torch.randn(4)`
// verbatim (f64 literals from `torch.randn(4).tolist()`); Rust narrows
// them to f32 at the array-literal seam. Clippy's excessive-precision
// warning is intentionally allowed so the values remain a byte-for-byte
// transcription of the reference report (a maintainer diffing the file
// sees the exact f64 the report captured, not a truncation).
#![allow(clippy::excessive_precision)]

use vokra_core::rng::torch_randn_f32;

/// The load-bearing anchor: `torch.manual_seed(0); torch.randn(4)` on
/// CPU produces four `f32` values with u32 bit patterns
/// `[0x3fc53f5c, 0xbe963c50, 0xc00b7149, 0x3f1184b6]` — copied verbatim
/// from the workflow-`wf_20fa0933-53d` bisect report's own measurement
/// of real torch (see this file's module doc).
#[test]
fn torch_randn_cpu_seed_0_k_4_matches_real_torch_bits() {
    let mut got = [0.0_f32; 4];
    torch_randn_f32(0, &mut got);

    // Real torch bit pattern — the anchor. If any of these four
    // asserts fails, `TorchRandnStream` has drifted from real
    // `torch.randn(4, device='cpu')` at seed=0.
    let expected_bits: [u32; 4] = [0x3FC5_3F5C, 0xBE96_3C50, 0xC00B_7149, 0x3F11_84B6];
    // Real torch f32 values — for a human-readable failure message.
    let expected_values: [f32; 4] = [
        1.5409960746765137,
        -0.293428897857666,
        -2.1787893772125244,
        0.5684312582015991,
    ];

    for i in 0..4 {
        assert_eq!(
            got[i].to_bits(),
            expected_bits[i],
            "torch_randn(seed=0)[{i}]: got {} ({:#010x}), expected {} ({:#010x}) — \
             CPU torch parity broken; see crates/vokra-core/src/rng/normal_kernel.rs \
             module doc §Historical note",
            got[i],
            got[i].to_bits(),
            expected_values[i],
            expected_bits[i],
        );
    }
}

/// Cache invariant: since Box-Muller produces samples in `(cos, sin)`
/// pairs, sample `k` and sample `k+1` are drawn from the SAME pair
/// (the sine is cached and returned on the next call). This means a
/// call for `k=2` (two samples, one full pair) must consume exactly
/// two `random64()` draws and produce two samples whose bytes match
/// the first two of the `k=4` call above. If a regression made the
/// cache drop the sine (or double-consume the engine), this test would
/// fail while the four-anchor test above might still pass at sample 0.
#[test]
fn torch_randn_pair_cache_preserves_second_sample_bytes() {
    // First two of the seed=0 anchor (the (cos, sin) pair from the
    // very first Box-Muller step).
    let expected_bits_pair0: [u32; 2] = [0x3FC5_3F5C, 0xBE96_3C50];

    let mut two = [0.0_f32; 2];
    torch_randn_f32(0, &mut two);
    for i in 0..2 {
        assert_eq!(
            two[i].to_bits(),
            expected_bits_pair0[i],
            "torch_randn(seed=0, k=2)[{i}]: got {:#010x}, expected {:#010x} — \
             pair cache broken (see TorchRandnStream::next_f32 doc)",
            two[i].to_bits(),
            expected_bits_pair0[i],
        );
    }

    // Sample 3 (index 2) is the cos half of the SECOND pair — freshly
    // drawn from the engine — so a k=3 call must match samples 0/1/2
    // of the k=4 call above. If pair caching mis-alternated, sample 2
    // here would show up as the sin of pair 0 instead.
    let expected_bits_three: [u32; 3] = [0x3FC5_3F5C, 0xBE96_3C50, 0xC00B_7149];
    let mut three = [0.0_f32; 3];
    torch_randn_f32(0, &mut three);
    for i in 0..3 {
        assert_eq!(
            three[i].to_bits(),
            expected_bits_three[i],
            "torch_randn(seed=0, k=3)[{i}]: got {:#010x}, expected {:#010x} — \
             pair-alternation broken between samples 1 and 2",
            three[i].to_bits(),
            expected_bits_three[i],
        );
    }
}

/// Cross-seed sanity: different seeds must produce different first
/// samples. A regression that accidentally seeded the engine with a
/// constant (dropping the input seed) would show as identical output
/// across all seeds — this test would catch it while the `k=4`
/// anchor's per-sample check would only detect it for seed=0.
#[test]
fn torch_randn_different_seeds_produce_different_first_samples() {
    let mut a = [0.0_f32; 1];
    let mut b = [0.0_f32; 1];
    torch_randn_f32(0, &mut a);
    torch_randn_f32(1, &mut b);
    assert_ne!(
        a[0].to_bits(),
        b[0].to_bits(),
        "seed=0 and seed=1 must produce different first samples — a match here \
         means the seed input was silently dropped somewhere in the pipeline"
    );
}
