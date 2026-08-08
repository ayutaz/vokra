//! `SbV2SynthRequest::rng_mode` — the SBV2 caller-facing switch between
//! the byte-exact `torch.randn` parity path (default) and the
//! backward-compatible synthetic path.
//!
//! # Backstory
//!
//! Every VITS-family flow (piper-plus, GPT-SoVITS, RVC, upcoming
//! CosyVoice2 CFM prior, Bark, F5-TTS — see rng/mod.rs) draws its
//! stochastic-duration-predictor noise from what upstream PyTorch code
//! calls `torch.randn`. On the CPU path (the default backend, and the
//! one SBV2's Python reference dumper uses via `torch.manual_seed(seed);
//! ... torch.randn(1, 2, T)`) that reduces to
//! `at::mt19937_engine` (Mersenne Twister) plus
//! `at::normal_distribution<double>` (Box-Muller in f64 with pair
//! caching) — fully specified in torch source (BSD-3-Clause) at
//! `ATen/core/MT19937RNGEngine.h` and
//! `ATen/core/DistributionsHelper.h:187-198`.
//!
//! # 2026-08-08 correction (bisect wf_20fa0933-53d)
//!
//! Prior to this correction, `SbV2Model::synthesize` drove a
//! `TorchRandnStream` backed by Philox4x32-10 in the belief that this
//! reproduced `torch.randn(device='cpu')`. A byte-level bisect against
//! real `torch.randn(4)` seed=0 (bit patterns `[0x3fc53f5c, 0xbe963c50,
//! 0xc00b7149, 0x3f1184b6]`) found NO match at any sample — CPU torch
//! uses MT19937, not Philox. The Philox path was `PhiloxRNGEngine.h`'s
//! own `randn`, which the torch header itself disclaims as "not used
//! anywhere except for tests in cpu_generator_test.cpp"
//! (PhiloxRNGEngine.h:39-41).
//!
//! [`vokra_core::rng::TorchRandnStream`] was rewritten to use
//! [`vokra_core::rng::TorchMt19937Engine`] with f64 Box-Muller and pair
//! caching, and its seed=0 anchor now passes bit-exactly against real
//! torch (see
//! `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs`).
//!
//! # Enum variant naming (historical)
//!
//! The variant name [`RngMode::PhiloxRngEnginePyTorchParity`] and the
//! GGUF metadata slug `"phyloxrngengine_10"` predate this correction
//! and are kept for on-disk compatibility with any already-emitted
//! metadata + to minimize churn in the seven downstream test files
//! that construct `SbV2SynthRequest` literals with an explicit
//! `rng_mode`. The name is now factually misleading — the underlying
//! algorithm is MT19937 + `at::normal_distribution<double>`, not
//! Philox — but the *behavior* (byte-exact against `torch.randn(cpu)`)
//! matches what a reader would want from a "PyTorch parity" flag. A
//! future PR will introduce a `TorchCpuMt19937Parity` alias and
//! deprecate the current spelling.
//!
//! # What this enum does
//!
//! [`RngMode::PhiloxRngEnginePyTorchParity`] (the [`Default`]) makes
//! `SbV2Model::synthesize` use `TorchRandnStream` so a fixture emitted
//! by `torch.manual_seed(N); torch.randn(1, 2, T)` byte-matches the
//! noise buffer this crate produces.
//! [`RngMode::GaussianSplitMix64Legacy`] preserves the pre-Step-10
//! behavior for existing synthetic tests whose duration outputs are
//! byte-frozen.
//!
//! # Why the default flipped
//!
//! Torch-parity is the desired M4-going-forward behavior — the whole
//! reason the RNG layer was refactored (Steps 1-8) is so real SBV2 v2
//! parity tests (Task 28 in the SBV2 design doc §12) can byte-diff
//! their SDP noise against a real reference dumper. The legacy path is
//! kept as an opt-in so pre-existing synthetic tests continue to hold
//! their byte-frozen assertions, but new callers should not need to
//! choose — the default IS what they want.

/// Selects which RNG family `SbV2Model::synthesize` uses for the
/// stochastic-duration-predictor's Gaussian noise draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngMode {
    /// Byte-exact `torch.manual_seed(N); torch.randn(...)` parity on
    /// the CPU backend, via [`vokra_core::rng::TorchRandnStream`].
    ///
    /// **Historical name, current algorithm**: this variant is called
    /// `PhiloxRngEnginePyTorchParity` for on-disk / on-metadata
    /// backward compatibility (the GGUF slug `"phyloxrngengine_10"`
    /// and the seven `SbV2SynthRequest` literals across
    /// `crates/vokra-models/tests/*.rs` construct with this name).
    /// The underlying algorithm is **not** Philox; it is
    /// `at::mt19937_engine` + `at::normal_distribution<double>` (f64
    /// Box-Muller with pair caching), which is what CPU torch
    /// actually uses. See the module doc's "2026-08-08 correction"
    /// section for the bisect report and the rewrite.
    ///
    /// The default.
    PhiloxRngEnginePyTorchParity,

    /// The pre-Step-10 behavior: Vokra's synthetic
    /// [`vokra_core::rng::GaussianSplitMix64`] (splitmix64 +
    /// Box-Muller). Existing synthetic-fixture tests keep their
    /// byte-frozen assertions by explicitly opting into this.
    GaussianSplitMix64Legacy,
}

impl Default for RngMode {
    /// The default is [`RngMode::PhiloxRngEnginePyTorchParity`] — the
    /// torch-parity path is the desired M4 forward behavior for new
    /// callers. Existing tests preserve their byte-frozen assertions
    /// by explicitly setting `rng_mode:
    /// RngMode::GaussianSplitMix64Legacy` in the request struct
    /// literal.
    fn default() -> Self {
        Self::PhiloxRngEnginePyTorchParity
    }
}

impl RngMode {
    /// Human-readable slug used in GGUF metadata (`vokra.sbv2.rng.torch_mode`)
    /// and in log lines. Stable across compiler versions so a producer /
    /// consumer contract remains diffable.
    ///
    /// **Note**: the `"phyloxrngengine_10"` slug is a historical
    /// artifact of the pre-2026-08-08 belief that CPU torch used
    /// Philox4x32-10 (see this module's doc). The underlying algorithm
    /// today is MT19937 + `normal_distribution<double>`. The slug is
    /// preserved so already-emitted GGUFs remain readable; a future
    /// producer wanting a truthful slug should introduce a new
    /// variant.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PhiloxRngEnginePyTorchParity => "phyloxrngengine_10",
            Self::GaussianSplitMix64Legacy => "gaussian_splitmix64_legacy",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_torch_parity() {
        assert_eq!(RngMode::default(), RngMode::PhiloxRngEnginePyTorchParity);
    }

    #[test]
    fn as_str_matches_expected_slugs() {
        assert_eq!(
            RngMode::PhiloxRngEnginePyTorchParity.as_str(),
            "phyloxrngengine_10"
        );
        assert_eq!(
            RngMode::GaussianSplitMix64Legacy.as_str(),
            "gaussian_splitmix64_legacy"
        );
    }
}
