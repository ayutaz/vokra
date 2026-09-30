//! Authenticated Realtime acoustic latent-to-PCM boundary.
//!
//! Realtime uses the same causal acoustic decoder topology as the native
//! VibeVoice-1.5B implementation, but its checkpoint identity and scalar
//! latent factors are different.  This module preflights the selected
//! learned-op backend, authenticates the Realtime composite, then reuses the
//! 1.5B decoder's loader and streaming DSP.
//! It accepts one scaled 64-wide latent frame and emits one 3200-sample,
//! 24 kHz mono chunk.  Learned decoder operations stay on the selected
//! backend; host-side layout/scaling work is not a CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::vibevoice::{
    VIBEVOICE_TOKENIZER_HOT_OPS, VibeVoiceAcousticDecoder, VibeVoiceAcousticDecoderStream,
    VibeVoiceLatentScale,
};

use super::VibeVoiceStreamingCheckpoint;

/// Width of one Realtime acoustic latent frame.
pub const REALTIME_ACOUSTIC_LATENT_WIDTH: usize = 64;
/// Samples emitted for one Realtime acoustic latent frame at 24 kHz.
pub const REALTIME_ACOUSTIC_CHUNK_SAMPLES: usize = 3_200;

/// Realtime acoustic decoder bound to one authenticated GGUF.
///
/// The decoder supports the backends whose complete tokenizer-op registry is
/// wired through [`crate::compute::Compute`].  Constructing this handle on an
/// uncovered backend returns [`VokraError::UnsupportedOp`] rather than
/// silently using CPU.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeAcousticDecoder {
    decoder: VibeVoiceAcousticDecoder,
    latent_scale: VibeVoiceLatentScale,
}

impl VibeVoiceRealtimeAcousticDecoder {
    /// Authenticates the Realtime composite and binds its decoder/scalars.
    pub fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        // Keep this preflight ahead of tensor binding.  The shared decoder
        // uses Conv1d/GroupedConv1d/Gemm/Gelu/RmsNorm through Compute; its
        // ConvTranspose topology is expressed as a host layout transform
        // followed by the selected backend's Conv1d kernel.  CUDA currently
        // lacks the grouped-convolution seam, so it must fail closed here.
        require_realtime_acoustic_backend(backend)?;
        // This also reports a feature/device-unavailable Metal build before
        // the authenticated GGUF payload is bound.  The decoder's forward
        // path repeats this registry check at execution time.
        let _ = Compute::for_backend(backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        // This gate must precede the shared decoder/scalar loaders.  In
        // particular, a Realtime file must never be accepted by the 1.5B
        // `VibeVoiceCheckpoint` manifest just because the decoder shapes fit.
        VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        let decoder = VibeVoiceAcousticDecoder::from_realtime_tensors(file, backend)?;
        let latent_scale = VibeVoiceLatentScale::from_realtime_tensors(file)?;
        Ok(Self {
            decoder,
            latent_scale,
        })
    }

    /// Returns the explicitly selected learned-op backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.decoder.backend()
    }

    /// Creates independent causal state for one streaming sample.
    #[must_use]
    pub fn stream(&self) -> VibeVoiceRealtimeAcousticDecoderStream {
        VibeVoiceRealtimeAcousticDecoderStream {
            decoder: self.decoder.stream(),
            latent_scale: self.latent_scale,
        }
    }

    /// Decodes one scaled latent frame without retaining causal state.
    pub fn decode_scaled_latent(&self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
        let mut stream = self.stream();
        stream.decode_scaled_latent(scaled_latent)
    }
}

/// Independent causal state for a Realtime acoustic stream.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeAcousticDecoderStream {
    decoder: VibeVoiceAcousticDecoderStream,
    latent_scale: VibeVoiceLatentScale,
}

impl VibeVoiceRealtimeAcousticDecoderStream {
    /// Clears causal history before starting a new sample.
    pub fn reset(&mut self) {
        self.decoder.reset();
    }

    /// Decodes one scaled `[64]` latent frame to one 3200-sample chunk.
    ///
    /// The official source performs `latent / scaling_factor - bias_factor`
    /// before passing the result to the causal acoustic tokenizer decoder.
    pub fn decode_scaled_latent(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
        if scaled_latent.len() != REALTIME_ACOUSTIC_LATENT_WIDTH {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice-realtime acoustic decoder requires one scaled [{REALTIME_ACOUSTIC_LATENT_WIDTH}] latent frame"
            )));
        }
        let unscaled = self.latent_scale.unscale_generated(scaled_latent)?;
        let pcm = self.decoder.decode_chunk(&unscaled, 1)?;
        if pcm.len() != REALTIME_ACOUSTIC_CHUNK_SAMPLES {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime acoustic decoder emitted {} samples, expected {}",
                pcm.len(),
                REALTIME_ACOUSTIC_CHUNK_SAMPLES
            )));
        }
        Ok(pcm)
    }
}

fn require_realtime_acoustic_backend(backend: BackendKind) -> Result<()> {
    match backend {
        BackendKind::Cpu | BackendKind::Metal => Ok(()),
        _ => Err(VokraError::UnsupportedOp(format!(
            "vibevoice-realtime acoustic decoder: backend {backend:?} lacks complete tokenizer learned-op coverage; no CPU fallback"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_acoustic_contract_is_one_frame_and_one_chunk() {
        assert_eq!(REALTIME_ACOUSTIC_LATENT_WIDTH, 64);
        assert_eq!(REALTIME_ACOUSTIC_CHUNK_SAMPLES, 3_200);
    }

    #[test]
    fn metal_backend_selection_is_permitted_without_device_claim() {
        assert!(require_realtime_acoustic_backend(BackendKind::Metal).is_ok());
    }

    #[test]
    fn uncovered_backend_is_rejected_without_fallback() {
        let error = require_realtime_acoustic_backend(BackendKind::Cuda).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }
}
