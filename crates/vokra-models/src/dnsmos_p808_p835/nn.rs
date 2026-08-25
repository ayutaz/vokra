use vokra_core::ir::graph::{
    MelAttrs, MelInterp, MelNorm, MelScale, Normalization, PadMode, StftAttrs, Window,
    WindowSymmetry,
};
use vokra_core::{Result, VokraError};
use vokra_ops::{mel_filterbank, stft};

use crate::compute::Compute;

use super::weights::{Conv2d, Dense, DnsmosWeights, P808Weights, P835Weights};
use super::{
    INPUT_LENGTH_SAMPLES, P808_FRAMES, P808_HOP, P808_N_FFT, P808_N_MELS, P835_BINS, P835_FRAMES,
    P835_HOP, P835_WINDOW, SAMPLE_RATE,
};

const IM2COL_CHUNK: usize = 1024;
const KERNEL: usize = 3;
const PAD: usize = 1;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ForwardScore {
    pub(super) p808: Option<f32>,
    pub(super) p835: Option<(f32, f32, f32)>,
}

#[derive(Debug)]
struct Tensor4 {
    data: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
}

impl Tensor4 {
    fn new(data: Vec<f32>, channels: usize, height: usize, width: usize) -> Result<Self> {
        let expected = checked_product("Tensor4", &[channels, height, width])?;
        if data.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "dnsmos: Tensor4 has {} values, expected {expected} for [{channels},{height},{width}]",
                data.len()
            )));
        }
        Ok(Self {
            data,
            channels,
            height,
            width,
        })
    }

    fn index(&self, channel: usize, y: usize, x: usize) -> usize {
        (channel * self.height + y) * self.width + x
    }
}

pub(super) fn score(
    compute: &Compute,
    weights: &DnsmosWeights,
    pcm: &[f32],
    run_p808: bool,
    run_p835: bool,
) -> Result<ForwardScore> {
    if !run_p808 && !run_p835 {
        return Err(VokraError::InvalidArgument(
            "dnsmos: score requested no bundle variants".to_owned(),
        ));
    }
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "dnsmos: PCM must contain at least one 16 kHz sample".to_owned(),
        ));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "dnsmos: PCM sample {index} is not finite"
        )));
    }

    let audio = repeat_to_window(pcm);
    let whole_seconds = audio.len() / SAMPLE_RATE as usize;
    let hops = whole_seconds.saturating_sub(9).max(1);
    let mut p808_sum = 0.0f32;
    let mut sig_sum = 0.0f64;
    let mut bak_sum = 0.0f64;
    let mut ovr_sum = 0.0f64;
    for hop in 0..hops {
        let start = hop * SAMPLE_RATE as usize;
        let end = start + INPUT_LENGTH_SAMPLES;
        let segment = audio.get(start..end).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "dnsmos: internal chunk {hop} [{start}..{end}] exceeds repeated PCM length {}",
                audio.len()
            ))
        })?;
        if run_p808 {
            p808_sum += p808_forward(compute, &weights.p808, &segment[..end - start - P808_HOP])?;
        }
        if run_p835 {
            let [sig, bak, ovr] = p835_forward(compute, &weights.p835, segment)?;
            sig_sum += poly(sig, -0.083_972_78, 1.220_839_53, 0.005_243_9);
            bak_sum += poly(bak, -0.131_668_88, 1.609_155_14, -0.396_045_46);
            ovr_sum += poly(ovr, -0.067_662_83, 1.115_464_68, 0.046_025_35);
        }
    }
    let count_f32 = hops as f32;
    let count_f64 = hops as f64;
    Ok(ForwardScore {
        p808: run_p808.then_some(p808_sum / count_f32),
        p835: run_p835.then_some((
            (sig_sum / count_f64) as f32,
            (bak_sum / count_f64) as f32,
            (ovr_sum / count_f64) as f32,
        )),
    })
}

fn repeat_to_window(pcm: &[f32]) -> Vec<f32> {
    let mut audio = pcm.to_vec();
    while audio.len() < INPUT_LENGTH_SAMPLES {
        // Exact `audio = np.append(audio, audio)` behavior from the official
        // scorer. Do not truncate at one window: the overshoot changes how
        // many one-second-hop windows short clips contribute.
        audio.extend_from_within(..);
    }
    audio
}

fn p808_forward(compute: &Compute, weights: &P808Weights, pcm: &[f32]) -> Result<f32> {
    let attrs = StftAttrs {
        n_fft: P808_N_FFT,
        hop_length: P808_HOP,
        win_length: P808_N_FFT,
        window: Window::Hann,
        window_symmetry: WindowSymmetry::Periodic,
        center: true,
        pad_mode: PadMode::Reflect,
        normalization: Normalization::Backward,
        causal: false,
        real_input: true,
    };
    let spectrogram = stft(pcm, &attrs)?;
    if spectrogram.frames != P808_FRAMES || spectrogram.bins != P808_N_FFT / 2 + 1 {
        return Err(VokraError::InvalidArgument(format!(
            "dnsmos p808: frontend produced {}x{}, expected {P808_FRAMES}x{}",
            spectrogram.frames,
            spectrogram.bins,
            P808_N_FFT / 2 + 1
        )));
    }
    let mel = mel_filterbank(&MelAttrs {
        sample_rate: SAMPLE_RATE,
        n_fft: P808_N_FFT,
        n_mels: P808_N_MELS,
        fmin: 0.0,
        fmax: Some(SAMPLE_RATE as f32 / 2.0),
        scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
        interp: MelInterp::Hz,
    })
    .apply(&spectrogram.power(), spectrogram.frames);
    let features = power_to_db_ref_max(&mel);
    let mut value = Tensor4::new(features, 1, P808_FRAMES, P808_N_MELS)?;
    for (index, conv) in weights.conv.iter().enumerate() {
        value = conv2d(compute, &value, conv)?;
        relu(&mut value.data);
        if matches!(index, 0 | 1 | 3) {
            value = max_pool_2x2(&value)?;
        }
    }
    let pooled = global_max(&value);
    let output = dense_stack(compute, pooled, &weights.dense)?;
    Ok(output[0])
}

fn power_to_db_ref_max(power: &[f32]) -> Vec<f32> {
    const AMIN: f32 = 1.0e-10;
    const TOP_DB: f32 = 80.0;
    let reference = power.iter().copied().fold(0.0f32, f32::max).max(AMIN);
    let reference_db = 10.0 * reference.log10();
    let mut db: Vec<f32> = power
        .iter()
        .map(|&value| 10.0 * value.max(AMIN).log10() - reference_db)
        .collect();
    let maximum = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = maximum - TOP_DB;
    for value in &mut db {
        *value = (value.max(floor) + 40.0) / 40.0;
    }
    db
}

fn p835_forward(compute: &Compute, weights: &P835Weights, pcm: &[f32]) -> Result<[f32; 3]> {
    let mut frames = vec![0.0; P835_FRAMES * P835_WINDOW];
    for frame in 0..P835_FRAMES {
        let start = frame * P835_HOP;
        frames[frame * P835_WINDOW..(frame + 1) * P835_WINDOW]
            .copy_from_slice(&pcm[start..start + P835_WINDOW]);
    }
    let mut real = vec![0.0; P835_FRAMES * P835_BINS];
    let mut imag = vec![0.0; P835_FRAMES * P835_BINS];
    compute.gemm_f32(
        P835_FRAMES,
        P835_BINS,
        P835_WINDOW,
        &frames,
        &weights.stft_real_io,
        None,
        &mut real,
    )?;
    compute.gemm_f32(
        P835_FRAMES,
        P835_BINS,
        P835_WINDOW,
        &frames,
        &weights.stft_imag_io,
        None,
        &mut imag,
    )?;
    let log_power = real
        .iter()
        .zip(&imag)
        .map(|(&real, &imag)| {
            let magnitude = (real * real + imag * imag).sqrt();
            magnitude.powf(2.0).max(1.0e-12).ln() / std::f32::consts::LN_10
        })
        .collect();
    let mut value = Tensor4::new(log_power, 1, P835_FRAMES, P835_BINS)?;
    for (index, conv) in weights.conv.iter().enumerate() {
        value = conv2d(compute, &value, conv)?;
        relu(&mut value.data);
        if matches!(index, 3 | 4 | 5) {
            value = max_pool_2x2(&value)?;
        }
    }
    let output = dense_stack(compute, global_max(&value), &weights.dense)?;
    Ok([output[0], output[1], output[2]])
}

fn conv2d(compute: &Compute, input: &Tensor4, weights: &Conv2d) -> Result<Tensor4> {
    if input.channels != weights.input
        || weights.weight.len() != weights.output * weights.input * KERNEL * KERNEL
        || weights.bias.len() != weights.output
    {
        return Err(VokraError::ModelLoad(format!(
            "dnsmos: Conv2d shape mismatch, activation channels={}, weights {}->{}, kernel=3x3",
            input.channels, weights.input, weights.output
        )));
    }
    let positions = checked_product("conv positions", &[input.height, input.width])?;
    let patch = checked_product("conv patch", &[weights.input, KERNEL, KERNEL])?;
    let mut output = vec![0.0; weights.output * positions];
    for chunk_start in (0..positions).step_by(IM2COL_CHUNK) {
        let chunk = (positions - chunk_start).min(IM2COL_CHUNK);
        let mut columns = vec![0.0; patch * chunk];
        for input_channel in 0..weights.input {
            for kernel_y in 0..KERNEL {
                for kernel_x in 0..KERNEL {
                    let patch_index = (input_channel * KERNEL + kernel_y) * KERNEL + kernel_x;
                    for local in 0..chunk {
                        let position = chunk_start + local;
                        let output_y = position / input.width;
                        let output_x = position % input.width;
                        let source_y = output_y + kernel_y;
                        let source_x = output_x + kernel_x;
                        if source_y >= PAD
                            && source_x >= PAD
                            && source_y - PAD < input.height
                            && source_x - PAD < input.width
                        {
                            columns[patch_index * chunk + local] = input.data
                                [input.index(input_channel, source_y - PAD, source_x - PAD)];
                        }
                    }
                }
            }
        }
        let mut block = vec![0.0; weights.output * chunk];
        compute.gemm_f32(
            weights.output,
            chunk,
            patch,
            &weights.weight,
            &columns,
            None,
            &mut block,
        )?;
        for channel in 0..weights.output {
            let destination = channel * positions + chunk_start;
            for local in 0..chunk {
                output[destination + local] =
                    block[channel * chunk + local] + weights.bias[channel];
            }
        }
    }
    Tensor4::new(output, weights.output, input.height, input.width)
}

fn max_pool_2x2(input: &Tensor4) -> Result<Tensor4> {
    let height = input.height / 2;
    let width = input.width / 2;
    let mut output = vec![f32::NEG_INFINITY; input.channels * height * width];
    for channel in 0..input.channels {
        for y in 0..height {
            for x in 0..width {
                let mut maximum = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        maximum =
                            maximum.max(input.data[input.index(channel, 2 * y + dy, 2 * x + dx)]);
                    }
                }
                output[(channel * height + y) * width + x] = maximum;
            }
        }
    }
    Tensor4::new(output, input.channels, height, width)
}

fn global_max(input: &Tensor4) -> Vec<f32> {
    let positions = input.height * input.width;
    (0..input.channels)
        .map(|channel| {
            input.data[channel * positions..(channel + 1) * positions]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

fn dense_stack(compute: &Compute, mut input: Vec<f32>, layers: &[Dense]) -> Result<Vec<f32>> {
    for (index, layer) in layers.iter().enumerate() {
        if input.len() != layer.input
            || layer.weight_io.len() != layer.input * layer.output
            || layer.bias.len() != layer.output
        {
            return Err(VokraError::ModelLoad(format!(
                "dnsmos: Dense {index} shape mismatch: input={}, expected={}, output={}",
                input.len(),
                layer.input,
                layer.output
            )));
        }
        let mut output = vec![0.0; layer.output];
        compute.gemm_f32(
            1,
            layer.output,
            layer.input,
            &input,
            &layer.weight_io,
            Some(&layer.bias),
            &mut output,
        )?;
        if index + 1 != layers.len() {
            relu(&mut output);
        }
        input = output;
    }
    Ok(input)
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn poly(value: f32, a: f64, b: f64, c: f64) -> f64 {
    let value = f64::from(value);
    (a * value + b) * value + c
}

fn checked_product(label: &str, values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product.checked_mul(value).ok_or_else(|| {
            VokraError::InvalidArgument(format!("dnsmos: {label} shape product overflow"))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;

    #[test]
    fn repetition_matches_official_append_audio_loop() {
        assert_eq!(
            repeat_to_window(&[1.0, 2.0])[..8],
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
        assert_eq!(repeat_to_window(&[1.0, 2.0]).len(), 262_144);
        assert_eq!(repeat_to_window(&vec![0.0; 16_000]).len(), 256_000);
        assert_eq!(
            repeat_to_window(&vec![0.0; INPUT_LENGTH_SAMPLES]).len(),
            INPUT_LENGTH_SAMPLES
        );
    }

    #[test]
    fn tiled_conv_uses_selected_gemm_and_same_padding() {
        let input = Tensor4::new(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap();
        let weights = Conv2d {
            weight: vec![1.0; 9],
            bias: vec![0.5],
            input: 1,
            output: 1,
        };
        let output = conv2d(&Compute::cpu(), &input, &weights).unwrap();
        assert_eq!(output.data, vec![10.5; 4]);
    }

    #[test]
    fn regular_p835_polyfit_coefficients_are_not_clipped() {
        assert!((poly(2.0, -0.083_972_78, 1.220_839_53, 0.005_243_9) - 2.111_031_84).abs() < 1e-8);
    }
}
