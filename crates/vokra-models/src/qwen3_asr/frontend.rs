//! Exact variable-length Whisper frontend used by Qwen3-ASR.
//!
//! Unlike OpenAI Whisper inference, the official Qwen3-ASR processor requests
//! `padding=true, truncation=false`. A single utterance is therefore not forced
//! to 30 seconds: the STFT emits `pcm.len() / 160` retained frames after the
//! Whisper trailing-frame drop. The signal-processing parameters themselves
//! remain the pinned 16 kHz / 400-point Hann / 160-hop / 128-band Slaney path.

use vokra_core::{FrontendPolicy, FrontendSpec, Result, VokraError};
use vokra_ops::{mel_attrs_from_spec, mel_filterbank, stft, stft_attrs_from_spec};

use super::SAMPLE_RATE;

pub(super) const N_FFT: usize = 400;
pub(super) const HOP_LENGTH: usize = 160;
pub(super) const N_MELS: usize = 128;
pub(super) const CONV_CHUNK_FRAMES: usize = 100;
const MIN_REFLECT_SAMPLES: usize = N_FFT / 2 + 1;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Qwen3AsrFeatures {
    /// Row-major `[N_MELS, frames]` normalized log-mel features.
    pub(super) values: Vec<f32>,
    /// Number of unpadded feature frames.
    pub(super) frames: usize,
}

pub(super) fn runtime_frontend_spec() -> FrontendSpec {
    FrontendSpec {
        n_fft: N_FFT as u32,
        hop: HOP_LENGTH as u32,
        win_length: N_FFT as u32,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: SAMPLE_RATE as f32 / 2.0,
        n_mels: N_MELS as u32,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: SAMPLE_RATE,
    }
}

pub(super) fn check_frontend_spec(file: &vokra_core::gguf::GgufFile) -> Result<()> {
    FrontendSpec::from_gguf(file)?.check_against(&runtime_frontend_spec(), FrontendPolicy::Fail)
}

/// Extracts the exact variable-length Qwen3-ASR log-mel matrix.
pub(super) fn extract(pcm: &[f32]) -> Result<Qwen3AsrFeatures> {
    if pcm.len() < MIN_REFLECT_SAMPLES {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_asr frontend: PCM has {} samples; reflect-padded n_fft={N_FFT} requires at least {MIN_REFLECT_SAMPLES}",
            pcm.len()
        )));
    }
    if let Some((index, value)) = pcm
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_asr frontend: PCM sample {index} is non-finite ({value})"
        )));
    }

    let frames = pcm.len() / HOP_LENGTH;
    let spec = runtime_frontend_spec();
    let stft_attrs = stft_attrs_from_spec(&spec)?;
    let mel_attrs = mel_attrs_from_spec(&spec)?;
    let spectrum = stft(pcm, &stft_attrs)?;
    if spectrum.frames < frames + 1 {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr frontend: STFT produced {} frames, expected at least {} before trailing-frame removal",
            spectrum.frames,
            frames + 1
        )));
    }
    let bins = spectrum.bins;
    let power = spectrum.power();
    let mel = mel_filterbank(&mel_attrs).apply(&power[..frames * bins], frames);

    let floor_log = 1.0e-10_f32.log10();
    let mut values = vec![floor_log; N_MELS * frames];
    let mut global_max = f32::NEG_INFINITY;
    for frame in 0..frames {
        for band in 0..N_MELS {
            let value = mel[frame * N_MELS + band].max(1.0e-10).log10();
            values[band * frames + frame] = value;
            global_max = global_max.max(value);
        }
    }
    let dynamic_floor = global_max - 8.0;
    for value in &mut values {
        *value = (value.max(dynamic_floor) + 4.0) / 4.0;
    }

    Ok(Qwen3AsrFeatures { values, frames })
}

/// Official three stride-2 convolution length transform, applied separately
/// to each 100-frame chunk.
pub(super) const fn encoded_frames(input_frames: usize) -> usize {
    let full_chunks = input_frames / CONV_CHUNK_FRAMES;
    let tail = input_frames % CONV_CHUNK_FRAMES;
    full_chunks * 13 + if tail == 0 { 0 } else { tail.div_ceil(8) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_length_matches_official_chunk_boundaries() {
        for (input, output) in [
            (1, 1),
            (8, 1),
            (9, 2),
            (99, 13),
            (100, 13),
            (101, 14),
            (799, 104),
            (800, 104),
            (3_000, 390),
        ] {
            assert_eq!(encoded_frames(input), output, "input frames={input}");
        }
    }

    #[test]
    fn silence_is_variable_length_and_finite() {
        let pcm = vec![0.0; HOP_LENGTH * 3];
        let features = extract(&pcm).expect("silent features");
        assert_eq!(features.frames, 3);
        assert_eq!(features.values.len(), N_MELS * 3);
        assert!(features.values.iter().all(|value| value.is_finite()));
        assert!(features.values.iter().all(|value| *value == -1.5));
    }

    #[test]
    fn invalid_waveforms_fail_before_stft() {
        assert!(extract(&[0.0; MIN_REFLECT_SAMPLES - 1]).is_err());
        let mut pcm = vec![0.0; MIN_REFLECT_SAMPLES];
        pcm[7] = f32::NAN;
        assert!(extract(&pcm).is_err());
    }
}
