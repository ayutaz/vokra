//! `SbV2SynthRequest::rng_mode` — the SBV2 caller-facing switch between
//! the byte-exact `torch.randn` parity path (default) and the
//! backward-compatible synthetic path.
//!
//! # Backstory
//!
//! Every VITS-family flow (piper-plus, GPT-SoVITS, RVC, upcoming
//! CosyVoice2 CFM prior, Bark, F5-TTS — see rng/mod.rs) draws its
//! stochastic-duration-predictor noise from what upstream PyTorch code
//! calls `torch.randn`. On the GPU path (torch.cuda.manual_seed) that
//! reduces to ATen's `PhiloxRNGEngine.h::randn`, whose algorithm is a
//! Philox4x32-10 + Box-Muller pipeline fully specified in torch source
//! (BSD-3-Clause).
//!
//! Vokra's Rust port lives in `vokra_core::rng::{TorchRandnStream,
//! philox4x32_10, ...}` and is byte-exact against a
//! Random123-KAT-audited Python dumper (see
//! `crates/vokra-core/tests/rng_torch_randn_e2e.rs`). Before this file
//! existed, `SbV2Model::synthesize` internally constructed a
//! `GaussianSplitMix64` (Vokra's pre-existing synthetic splitmix64 +
//! Box-Muller wrapper) — good for reproducible synthetic tests but NOT
//! byte-parity with any PyTorch reference.
//!
//! # What this enum does
//!
//! [`RngMode::PhiloxRngEnginePyTorchParity`] (the [`Default`]) makes
//! `SbV2Model::synthesize` use `TorchRandnStream` so a fixture emitted
//! by `torch.manual_seed(N); torch.randn(1, 2, T)` (via the Python
//! PhiloxRNGEngine.h port in `tools/parity/torch_philox_dump.py`)
//! byte-matches the noise buffer this crate produces.
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
    /// Byte-exact `torch.manual_seed(N); torch.randn(...)` parity via
    /// [`vokra_core::rng::TorchRandnStream`] (Philox4x32-10 +
    /// PhiloxRNGEngine.h Box-Muller). The default.
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
