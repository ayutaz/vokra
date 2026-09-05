//! Authenticated ChatTTS composite component contracts.
//!
//! These typed contracts are deliberately separate from [`super::ChatTts`].
//! Until VAST records the complete tensor manifest, no public loader may turn
//! these axes into a synthetic decoder or vocoder.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

/// Source DVAE decoder topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsDvaeConfig {
    /// Input latent width.
    pub input_dim: usize,
    /// Output latent width.
    pub output_dim: usize,
    /// Hidden convolution width.
    pub hidden_dim: usize,
    /// Residual block count.
    pub layers: usize,
    /// Grouped residual normalization width.
    pub batch_norm_channels: usize,
    /// Mel output bins.
    pub mel_bins: usize,
}

impl Default for ChatTtsDvaeConfig {
    fn default() -> Self {
        Self {
            input_dim: 512,
            output_dim: 512,
            hidden_dim: 256,
            layers: 12,
            batch_norm_channels: 128,
            mel_bins: 100,
        }
    }
}

/// Source GFSQ grouping contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsGfsqConfig {
    /// Quantizer latent dimension.
    pub dimension: usize,
    /// Per-group FSQ levels.
    pub levels: [usize; 4],
    /// Number of groups.
    pub groups: usize,
    /// Residual depth.
    pub residuals: usize,
}

impl Default for ChatTtsGfsqConfig {
    fn default() -> Self {
        Self {
            dimension: 1_024,
            levels: [5; 4],
            groups: 2,
            residuals: 2,
        }
    }
}

/// Source hidden-state decoder topology used by `use_decoder=true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsDecoderConfig {
    /// Decoder input width.
    pub input_dim: usize,
    /// Decoder output width.
    pub output_dim: usize,
    /// Hidden width.
    pub hidden_dim: usize,
    /// Residual block count.
    pub layers: usize,
    /// Batch-normalization channels.
    pub batch_norm_channels: usize,
}

impl Default for ChatTtsDecoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 384,
            output_dim: 384,
            hidden_dim: 512,
            layers: 12,
            batch_norm_channels: 128,
        }
    }
}

/// Source Vocos vocoder topology and sample-rate contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsVocosConfig {
    /// Output sample rate.
    pub sample_rate_hz: usize,
    /// STFT FFT size.
    pub n_fft: usize,
    /// STFT hop size.
    pub hop_length: usize,
    /// Mel bins consumed by Vocos.
    pub mel_bins: usize,
    /// Vocos backbone width.
    pub hidden_dim: usize,
    /// Vocos feed-forward width.
    pub intermediate_dim: usize,
    /// Vocos block count.
    pub layers: usize,
}

impl Default for ChatTtsVocosConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            n_fft: 1_024,
            hop_length: 256,
            mel_bins: 100,
            hidden_dim: 512,
            intermediate_dim: 1_536,
            layers: 8,
        }
    }
}

/// Composite component axes fixed by the source release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChatTtsComponentContract {
    /// DVAE decoder axes.
    pub dvae: ChatTtsDvaeConfig,
    /// GFSQ axes.
    pub gfsq: ChatTtsGfsqConfig,
    /// Optional hidden-state decoder axes.
    pub decoder: ChatTtsDecoderConfig,
    /// Vocos axes.
    pub vocos: ChatTtsVocosConfig,
}

impl ChatTtsComponentContract {
    /// Validates the cross-component dimensions without inventing weights.
    pub fn validate(&self) -> Result<()> {
        if self.dvae.input_dim != self.dvae.output_dim
            || self.dvae.mel_bins != self.vocos.mel_bins
            || self.gfsq.levels != [5; 4]
            || self.gfsq.groups != 2
            || self.gfsq.residuals != 2
        {
            return Err(VokraError::ModelLoad(
                "chattts: authenticated DVAE/GFSQ/Vocos axis contract mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Future native composite session, gated on a complete VAST manifest.
#[derive(Debug)]
pub struct ChatTtsCompositeSession {
    backend: BackendKind,
    contract: ChatTtsComponentContract,
}

impl ChatTtsCompositeSession {
    /// Rejects GPT-only or self-stamped bundles before any learned operation.
    pub fn from_authenticated_manifest(_manifest: &[u8], backend: BackendKind) -> Result<Self> {
        let _ = backend;
        Err(VokraError::UnsupportedOp(
            "chattts: full composite manifest is not yet authenticated; DVAE/Decoder/Vocos remain inspection-only".to_owned(),
        ))
    }

    /// Returns the selected backend for future complete learned-op dispatch.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns fixed source component axes.
    #[must_use]
    pub const fn contract(&self) -> ChatTtsComponentContract {
        self.contract
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_component_axes_are_cross_checked() {
        let contract = ChatTtsComponentContract::default();
        contract.validate().unwrap();
        assert_eq!(contract.vocos.sample_rate_hz, 24_000);
        assert_eq!(contract.gfsq.levels, [5; 4]);
        let mut altered = contract;
        altered.vocos.mel_bins = 80;
        assert!(altered.validate().is_err());
    }
}
