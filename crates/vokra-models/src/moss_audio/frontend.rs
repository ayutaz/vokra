//! Exact variable-length Whisper frontend used by MOSS-Audio.

use vokra_core::{FrontendSpec, Result, VokraError};
use vokra_ops::{mel_attrs_from_spec, mel_filterbank, stft, stft_attrs_from_spec};

use super::SAMPLE_RATE;

pub(super) const N_FFT: usize = 400;
pub(super) const HOP_LENGTH: usize = 160;
pub(super) const N_MELS: usize = 128;
/// Official `n_window * 2` chunk size before the three stride-2 convs.
pub(super) const AUDIO_CHUNK_FRAMES: usize = 400;
const MIN_REFLECT_SAMPLES: usize = N_FFT / 2 + 1;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MossAudioFeatures {
    /// Row-major `[N_MELS, frames]` normalized log-mel features.
    pub(super) values: Vec<f32>,
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

pub(super) fn extract(pcm: &[f32]) -> Result<MossAudioFeatures> {
    if pcm.len() < MIN_REFLECT_SAMPLES {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio frontend: PCM has {} samples; reflect-padded n_fft={N_FFT} requires at least {MIN_REFLECT_SAMPLES}",
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
            "moss_audio frontend: PCM sample {index} is non-finite ({value})"
        )));
    }

    let frames = pcm.len() / HOP_LENGTH;
    let spec = runtime_frontend_spec();
    let stft_attrs = stft_attrs_from_spec(&spec)?;
    let mel_attrs = mel_attrs_from_spec(&spec)?;
    let spectrum = stft(pcm, &stft_attrs)?;
    if spectrum.frames < frames + 1 {
        return Err(VokraError::ModelLoad(format!(
            "moss_audio frontend: STFT produced {} frames, expected at least {} before trailing-frame removal",
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

    Ok(MossAudioFeatures { values, frames })
}

pub(super) const fn encoded_frames(input_frames: usize) -> usize {
    input_frames.div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_length_matches_official_formula() {
        for (input, output) in [(1, 1), (8, 1), (9, 2), (399, 50), (400, 50)] {
            assert_eq!(encoded_frames(input), output);
        }
    }

    #[test]
    fn silence_is_variable_length_and_finite() {
        let pcm = vec![0.0; HOP_LENGTH * 3];
        let features = extract(&pcm).expect("silent features");
        assert_eq!(features.frames, 3);
        assert_eq!(features.values.len(), N_MELS * 3);
        assert!(features.values.iter().all(|value| *value == -1.5));
    }
}
