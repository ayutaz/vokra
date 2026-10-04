//! Authenticated Realtime acoustic latent-to-PCM boundary.
//!
//! Realtime uses the same causal acoustic decoder topology as the native
//! VibeVoice-1.5B implementation, but its checkpoint identity and scalar
//! latent factors are different.  This module authenticates the Realtime
//! composite first, then reuses the 1.5B decoder's loader and streaming DSP.
//! It accepts one scaled 64-wide latent frame and emits one 3200-sample,
//! 24 kHz mono chunk.  No backend partitioning or CPU fallback is allowed.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::vibevoice::{
    VibeVoiceAcousticDecoder, VibeVoiceAcousticDecoderStream, VibeVoiceLatentScale,
};

use super::VibeVoiceStreamingCheckpoint;

/// Width of one Realtime acoustic latent frame.
pub const REALTIME_ACOUSTIC_LATENT_WIDTH: usize = 64;
/// Samples emitted for one Realtime acoustic latent frame at 24 kHz.
pub const REALTIME_ACOUSTIC_CHUNK_SAMPLES: usize = 3_200;

/// Realtime acoustic decoder bound to one authenticated GGUF.
///
/// The decoder is intentionally CPU-only until an independent Metal path and
/// its parity evidence exist.  Constructing this handle on another backend
/// returns [`VokraError::UnsupportedOp`] rather than silently using CPU.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeAcousticDecoder {
    decoder: VibeVoiceAcousticDecoder,
    latent_scale: VibeVoiceLatentScale,
}

impl VibeVoiceRealtimeAcousticDecoder {
    /// Authenticates the Realtime composite and binds its decoder/scalars.
    pub fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        // This gate must precede the shared decoder/scalar loaders.  In
        // particular, a Realtime file must never be accepted by the 1.5B
        // `VibeVoiceCheckpoint` manifest just because the decoder shapes fit.
        VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        require_cpu(backend)?;
        let decoder = VibeVoiceAcousticDecoder::from_realtime_tensors(file, backend)?;
        let latent_scale = VibeVoiceLatentScale::from_realtime_tensors(file)?;
        Ok(Self {
            decoder,
            latent_scale,
        })
    }

    /// Returns the backend selected for this decoder (currently CPU only).
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

fn require_cpu(backend: BackendKind) -> Result<()> {
    if backend != BackendKind::Cpu {
        return Err(VokraError::UnsupportedOp(format!(
            "vibevoice-realtime acoustic decoder: backend {backend:?} is unsupported; no CPU fallback"
        )));
    }
    Ok(())
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
    fn non_cpu_backend_is_rejected_without_fallback() {
        let error = require_cpu(BackendKind::Metal).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }
}
