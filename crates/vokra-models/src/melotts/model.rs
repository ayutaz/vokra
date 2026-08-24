//! End-to-end MeloTTS acoustic core over precomputed language features.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::rng::NormalSource;
use vokra_core::{Result, VokraError};

use crate::sbv2::duration::length_regulate;

use super::{
    INTER_CHANNELS, MeloDecoder, MeloDurationModel, MeloFlowModel, MeloTextEncoder,
    MeloTextFeatures, MeloTtsCheckpoint,
};

/// Runtime controls matching the official MeloTTS inference surface.
#[derive(Debug, Clone, Copy)]
pub struct MeloSynthesisOptions {
    /// Blend between deterministic (`0`) and stochastic (`1`) duration.
    pub sdp_ratio: f32,
    /// Standard-deviation multiplier for acoustic-prior sampling.
    pub noise_scale: f32,
    /// Standard-deviation multiplier for stochastic duration sampling.
    pub noise_scale_w: f32,
    /// Duration multiplier; values above one produce slower speech.
    pub length_scale: f32,
    /// Allocation guard applied after integer duration prediction.
    pub max_frames: usize,
}

impl Default for MeloSynthesisOptions {
    fn default() -> Self {
        Self {
            sdp_ratio: 0.2,
            noise_scale: 0.667,
            noise_scale_w: 0.8,
            length_scale: 1.0,
            max_frames: 16_384,
        }
    }
}

/// Native low-level synthesis result.
#[derive(Debug, Clone)]
pub struct MeloSynthesisOutput {
    /// Mono floating-point PCM in `[-1, 1]`.
    pub pcm: Vec<f32>,
    /// Output sampling rate (44,100 Hz for every official release).
    pub sample_rate: u32,
    /// Integer decoder-frame duration per input position.
    pub durations: Vec<i32>,
    /// Total decoder-frame count before 512x upsampling.
    pub frame_count: usize,
}

/// Loaded text, duration, flow and decoder stacks for one official release.
pub struct MeloTts {
    text_encoder: MeloTextEncoder,
    duration: MeloDurationModel,
    flow: MeloFlowModel,
    decoder: MeloDecoder,
}

impl MeloTts {
    pub(super) fn from_checkpoint(checkpoint: &MeloTtsCheckpoint, file: &GgufFile) -> Result<Self> {
        Ok(Self {
            text_encoder: checkpoint.load_text_encoder(file)?,
            duration: checkpoint.load_duration_model(file)?,
            flow: checkpoint.load_flow_model(file)?,
            decoder: checkpoint.load_decoder(file)?,
        })
    }

    /// Synthesizes PCM from already-tokenized phoneme/tone/language and BERT
    /// feature matrices. Raw-text frontend availability is intentionally a
    /// separate language-specific boundary.
    pub fn synthesize<R: NormalSource>(
        &self,
        features: MeloTextFeatures<'_>,
        options: MeloSynthesisOptions,
        rng: &mut R,
        backend: BackendKind,
    ) -> Result<MeloSynthesisOutput> {
        validate_options(options)?;
        let encoded = self.text_encoder.encode(features, backend)?;
        let durations = self.duration.predict(
            &encoded.hidden,
            encoded.sequence_len,
            &encoded.speaker_conditioning,
            options.sdp_ratio,
            options.noise_scale_w,
            options.length_scale,
            rng,
            backend,
        )?;
        let frame_count = durations.iter().try_fold(0usize, |total, duration| {
            let duration = usize::try_from(*duration).map_err(|_| {
                VokraError::InvalidArgument(format!(
                    "melotts synthesis: duration must be non-negative, got {duration}"
                ))
            })?;
            total.checked_add(duration).ok_or_else(|| {
                VokraError::InvalidArgument("melotts synthesis: frame count overflow".to_owned())
            })
        })?;
        if frame_count == 0 || frame_count > options.max_frames {
            return Err(VokraError::InvalidArgument(format!(
                "melotts synthesis: predicted {frame_count} frames, allowed range is 1..={}",
                options.max_frames
            )));
        }

        let channels = INTER_CHANNELS as usize;
        let mean = length_regulate(&encoded.mean, &durations, channels);
        let log_scale = length_regulate(&encoded.log_scale, &durations, channels);
        let mut prior = Vec::with_capacity(frame_count * channels);
        for (mean, log_scale) in mean.into_iter().zip(log_scale) {
            let deviation = if options.noise_scale == 0.0 {
                0.0
            } else {
                rng.next_normal() * options.noise_scale
            };
            let value = mean + deviation * vokra_math::exp(log_scale);
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "melotts synthesis: acoustic prior produced a non-finite value".to_owned(),
                ));
            }
            prior.push(value);
        }
        let decoder_latent =
            self.flow
                .inverse(&prior, frame_count, &encoded.speaker_conditioning, backend)?;
        let pcm = self.decoder.decode(
            &decoder_latent,
            frame_count,
            &encoded.speaker_conditioning,
            backend,
        )?;
        Ok(MeloSynthesisOutput {
            pcm,
            sample_rate: self.decoder.sample_rate(),
            durations,
            frame_count,
        })
    }
}

fn validate_options(options: MeloSynthesisOptions) -> Result<()> {
    if !options.noise_scale.is_finite() || options.noise_scale < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "melotts synthesis: noise_scale must be finite and non-negative, got {}",
            options.noise_scale
        )));
    }
    if options.max_frames == 0 {
        return Err(VokraError::InvalidArgument(
            "melotts synthesis: max_frames must be positive".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&options.sdp_ratio) || !options.sdp_ratio.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "melotts synthesis: sdp_ratio must be finite in [0, 1], got {}",
            options.sdp_ratio
        )));
    }
    if !options.noise_scale_w.is_finite() || options.noise_scale_w < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "melotts synthesis: noise_scale_w must be finite and non-negative, got {}",
            options.noise_scale_w
        )));
    }
    if !options.length_scale.is_finite() || options.length_scale <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "melotts synthesis: length_scale must be finite and positive, got {}",
            options.length_scale
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_defaults_are_valid() {
        validate_options(MeloSynthesisOptions::default()).unwrap();
    }

    #[test]
    fn allocation_guard_must_be_positive() {
        let options = MeloSynthesisOptions {
            max_frames: 0,
            ..MeloSynthesisOptions::default()
        };
        assert!(validate_options(options).is_err());
    }
}
