//! Causal VibeVoice tokenizer encoder and acoustic decoder.
//!
//! The encoder is shared by the acoustic and semantic tokenizers, while the
//! acoustic decoder provides the authenticated latent-to-PCM path.  Full
//! composite generation and stochastic latent sampling remain separate
//! stages. Unlike the generic continuous VAE, this path preserves
//! VibeVoice's causal SConv1d layout and per-stage ConvNeXt-like blocks.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::hifigan::HifiGanComputeOps;
use crate::strict_checkpoint::{load_tensor, require_tensor_shape};
use vokra_ops::hifigan::HifiGanBackendOps;

const BASE: usize = 32;
const RATIOS: [usize; 6] = [2, 2, 4, 5, 5, 8];
const DOWN_STRIDES: [usize; 7] = [1, 2, 2, 4, 5, 5, 8];
const DOWN_KERNELS: [usize; 7] = [7, 4, 4, 8, 10, 10, 16];
const CHANNELS: [usize; 7] = [32, 64, 128, 256, 512, 1024, 2048];
const DEPTHS: [usize; 7] = [3, 3, 3, 3, 3, 3, 8];
const HOP: usize = 3_200;
const KERNEL: usize = 7;
const EPS: f32 = 1.0e-5;
const GAMMA: f32 = 1.0e-6;

/// Learned operations used by the causal tokenizer encoder.
pub const VIBEVOICE_TOKENIZER_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::Gemm,
    HotOp::Gelu,
    HotOp::RmsNorm,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerKind {
    Acoustic,
    Semantic,
}

impl TokenizerKind {
    fn prefix(self) -> &'static str {
        match self {
            Self::Acoustic => "acoustic_tokenizer",
            Self::Semantic => "semantic_tokenizer",
        }
    }

    const fn output_dim(self) -> usize {
        match self {
            Self::Acoustic => 64,
            Self::Semantic => 128,
        }
    }
}

#[derive(Debug, Clone)]
struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

#[derive(Debug, Clone)]
struct Block {
    norm: Vec<f32>,
    mixer: Conv,
    gamma: Vec<f32>,
    ffn_norm: Vec<f32>,
    linear1: Linear,
    linear2: Linear,
    ffn_gamma: Vec<f32>,
}

#[derive(Debug, Clone)]
struct EncoderWeights {
    downsamples: Vec<Conv>,
    stages: Vec<Vec<Block>>,
    head: Conv,
}

#[derive(Debug, Clone)]
struct TransposeConv {
    /// PyTorch ConvTranspose1d layout `[input, output, kernel]`.
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
}

#[derive(Debug, Clone)]
struct DecoderWeights {
    stem: Conv,
    stages: Vec<Vec<Block>>,
    upsamples: Vec<TransposeConv>,
    head: Conv,
}

/// Strict causal encoder for one authenticated VibeVoice tokenizer variant.
#[derive(Debug, Clone)]
pub struct VibeVoiceTokenizerEncoder {
    weights: Arc<EncoderWeights>,
    kind: TokenizerKind,
    backend: BackendKind,
}

/// Independent per-sample streaming state.  It is never shared between
/// acoustic/semantic branches or positive/negative generation branches.
#[derive(Debug, Clone)]
pub struct VibeVoiceTokenizerStream {
    encoder: VibeVoiceTokenizerEncoder,
    caches: Vec<Vec<f32>>,
}

/// Authenticated VibeVoice acoustic latent decoder.
///
/// The decoder accepts row-major `[frames, 64]` unscaled acoustic latents and
/// returns 24 kHz mono PCM.  Its weights are bound only after the complete
/// authenticated 1,204-tensor checkpoint contract has been verified.
#[derive(Debug, Clone)]
pub struct VibeVoiceAcousticDecoder {
    weights: Arc<DecoderWeights>,
    backend: BackendKind,
}

/// Independent streaming state for one acoustic decoder sample.
#[derive(Debug, Clone)]
pub struct VibeVoiceAcousticDecoderStream {
    decoder: VibeVoiceAcousticDecoder,
    caches: Vec<Vec<f32>>,
}

impl VibeVoiceTokenizerEncoder {
    /// Loads the authenticated acoustic tokenizer encoder.
    pub fn acoustic_from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Self::from_authenticated(file, backend, TokenizerKind::Acoustic)
    }

    /// Loads the authenticated semantic tokenizer encoder.
    pub fn semantic_from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Self::from_authenticated(file, backend, TokenizerKind::Semantic)
    }

    fn from_authenticated(
        file: &GgufFile,
        backend: BackendKind,
        kind: TokenizerKind,
    ) -> Result<Self> {
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        let prefix = format!("model.{}", kind.prefix());
        let mut downsamples = Vec::with_capacity(7);
        for index in 0..7 {
            downsamples.push(load_conv(
                file,
                &format!("{prefix}.encoder.downsample_layers.{index}.0.conv.conv"),
                if index == 0 { 1 } else { CHANNELS[index - 1] },
                CHANNELS[index],
                DOWN_KERNELS[index],
                DOWN_STRIDES[index],
                1,
            )?);
        }
        let mut stages = Vec::with_capacity(CHANNELS.len());
        for (stage, (&channels, &depth)) in CHANNELS.iter().zip(DEPTHS.iter()).enumerate() {
            let mut blocks = Vec::with_capacity(depth);
            for block in 0..depth {
                let p = format!("{prefix}.encoder.stages.{stage}.{block}");
                blocks.push(Block {
                    norm: load_raw(file, &format!("{p}.norm.weight"), channels)?,
                    mixer: load_conv(
                        file,
                        &format!("{p}.mixer.conv.conv.conv"),
                        channels,
                        channels,
                        KERNEL,
                        1,
                        channels,
                    )?,
                    gamma: load_raw(file, &format!("{p}.gamma"), channels)?,
                    ffn_norm: load_raw(file, &format!("{p}.ffn_norm.weight"), channels)?,
                    linear1: load_linear(
                        file,
                        &format!("{p}.ffn.linear1"),
                        channels,
                        4 * channels,
                    )?,
                    linear2: load_linear(
                        file,
                        &format!("{p}.ffn.linear2"),
                        4 * channels,
                        channels,
                    )?,
                    ffn_gamma: load_raw(file, &format!("{p}.ffn_gamma"), channels)?,
                });
            }
            stages.push(blocks);
        }
        let head = load_conv(
            file,
            &format!("{prefix}.encoder.head.conv.conv"),
            CHANNELS[6],
            kind.output_dim(),
            KERNEL,
            1,
            1,
        )?;
        let weights = EncoderWeights {
            downsamples,
            stages,
            head,
        };
        validate_weights(&weights, kind)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        Ok(Self {
            weights: Arc::new(weights),
            kind,
            backend,
        })
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the output width (64 acoustic or 128 semantic).
    #[must_use]
    pub const fn output_dim(&self) -> usize {
        self.kind.output_dim()
    }

    /// Encodes PCM into row-major `[frames, output_dim]` features.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice tokenizer encoder requires non-empty PCM".to_owned(),
            ));
        }
        finite("vibevoice tokenizer PCM", pcm)?;
        let compute = Compute::for_backend(self.backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        let channels = self.encode_nonstream(&compute, pcm)?;
        channel_major_to_rows(&channels.0, channels.1, self.output_dim())
    }

    /// Creates an independent streaming state for this encoder.
    #[must_use]
    pub fn stream(&self) -> VibeVoiceTokenizerStream {
        VibeVoiceTokenizerStream {
            encoder: self.clone(),
            caches: vec![Vec::new(); self.conv_count()],
        }
    }

    fn conv_count(&self) -> usize {
        self.weights.downsamples.len()
            + self
                .weights
                .stages
                .iter()
                .flat_map(|stage| stage.iter())
                .count()
            + 1
    }

    fn encode_nonstream(&self, compute: &Compute, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut x = causal_conv(compute, pcm, 1, &self.weights.downsamples[0])?;
        let mut time = (pcm.len() + DOWN_STRIDES[0] - 1) / DOWN_STRIDES[0];
        for (stage_index, blocks) in self.weights.stages.iter().enumerate() {
            let channels = CHANNELS[stage_index];
            for block in blocks {
                x = block_forward(compute, &x, channels, time, block)?;
            }
            if let Some(down) = self.weights.downsamples.get(stage_index + 1) {
                x = causal_conv(compute, &x, channels, down)?;
                time = (time + down.stride - 1) / down.stride;
            }
        }
        let output = causal_conv(compute, &x, CHANNELS[6], &self.weights.head)?;
        let frames = time;
        Ok((output, frames))
    }
}

impl VibeVoiceTokenizerStream {
    /// Zeros established causal history while preserving its buffer shape.
    pub fn set_to_zero(&mut self) {
        for cache in &mut self.caches {
            cache.fill(0.0);
        }
    }

    /// Clears causal history so a following sample cannot inherit PCM state.
    pub fn reset(&mut self) {
        self.caches.fill(Vec::new());
    }

    /// Encodes one causal PCM chunk, rolling back every cache on failure.
    pub fn encode_chunk(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice tokenizer stream requires non-empty PCM".to_owned(),
            ));
        }
        if pcm.len() % HOP != 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice tokenizer stream requires 3200-sample aligned chunks".to_owned(),
            ));
        }
        finite("vibevoice tokenizer stream PCM", pcm)?;
        let mut staged = self.caches.clone();
        let result = self.encode_chunk_inner(pcm, &mut staged);
        if result.is_ok() {
            self.caches = staged;
        }
        result
    }

    fn encode_chunk_inner(&self, pcm: &[f32], caches: &mut [Vec<f32>]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.encoder.backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        let mut cursor = 0;
        let mut x = causal_conv_stream(
            &compute,
            pcm,
            1,
            &self.encoder.weights.downsamples[0],
            &mut caches[cursor],
        )?;
        cursor += 1;
        let mut time = pcm.len();
        for (stage_index, blocks) in self.encoder.weights.stages.iter().enumerate() {
            let channels = CHANNELS[stage_index];
            for block in blocks {
                x = block_forward_stream(&compute, &x, channels, time, block, &mut caches[cursor])?;
                cursor += 1;
            }
            if let Some(down) = self.encoder.weights.downsamples.get(stage_index + 1) {
                x = causal_conv_stream(&compute, &x, channels, down, &mut caches[cursor])?;
                cursor += 1;
                time = (time + down.stride - 1) / down.stride;
            }
        }
        let output = causal_conv_stream(
            &compute,
            &x,
            CHANNELS[6],
            &self.encoder.weights.head,
            &mut caches[cursor],
        )?;
        channel_major_to_rows(&output, time, self.encoder.output_dim())
    }
}

impl VibeVoiceAcousticDecoder {
    /// Loads the strict acoustic decoder from an authenticated GGUF.
    pub fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        let prefix = "model.acoustic_tokenizer.decoder";
        let stem = load_conv(
            file,
            &format!("{prefix}.upsample_layers.0.0.conv.conv"),
            64,
            2048,
            7,
            1,
            1,
        )?;
        let channels = [2048, 1024, 512, 256, 128, 64, 32];
        let depths = [8, 3, 3, 3, 3, 3, 3];
        let mut stages = Vec::with_capacity(channels.len());
        for (stage, (&width, &depth)) in channels.iter().zip(depths.iter()).enumerate() {
            stages.push(load_blocks(file, prefix, stage, width, depth)?);
        }
        let transpose_spec = [
            (2048, 1024, 16, 8),
            (1024, 512, 10, 5),
            (512, 256, 10, 5),
            (256, 128, 8, 4),
            (128, 64, 4, 2),
            (64, 32, 4, 2),
        ];
        let mut upsamples = Vec::with_capacity(transpose_spec.len());
        for (index, &(input, output, kernel, stride)) in transpose_spec.iter().enumerate() {
            upsamples.push(load_transpose_conv(
                file,
                &format!("{prefix}.upsample_layers.{}.0.convtr.convtr", index + 1),
                input,
                output,
                kernel,
                stride,
            )?);
        }
        let head = load_conv(file, &format!("{prefix}.head.conv.conv"), 32, 1, 7, 1, 1)?;
        let weights = DecoderWeights {
            stem,
            stages,
            upsamples,
            head,
        };
        validate_decoder_weights(&weights)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        Ok(Self {
            weights: Arc::new(weights),
            backend,
        })
    }

    /// Returns the selected backend; unsupported backends fail during loading.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Decodes row-major unscaled `[frames, 64]` latents to 24 kHz mono PCM.
    pub fn decode(&self, latents: &[f32], frames: usize) -> Result<Vec<f32>> {
        if frames == 0 || latents.len() != frames * 64 {
            return Err(VokraError::InvalidArgument(
                "vibevoice acoustic decoder requires non-empty [frames,64] latents".to_owned(),
            ));
        }
        finite("vibevoice acoustic decoder latents", latents)?;
        let compute = Compute::for_backend(self.backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        let input = transpose_tc_to_ct(latents, frames, 64)?;
        let output = decode_channel_major(&compute, &input, frames, &self.weights)?;
        if output.len() != decoder_pcm_len(frames)? {
            return Err(VokraError::ModelLoad(
                "vibevoice acoustic decoder did not produce 3200 samples per latent frame"
                    .to_owned(),
            ));
        }
        Ok(output)
    }

    /// Creates an isolated stream for one stable sample.
    #[must_use]
    pub fn stream(&self) -> VibeVoiceAcousticDecoderStream {
        VibeVoiceAcousticDecoderStream {
            decoder: self.clone(),
            caches: vec![Vec::new(); self.decoder_cache_count()],
        }
    }

    fn decoder_cache_count(&self) -> usize {
        2 + self.weights.stages.iter().map(Vec::len).sum::<usize>() + self.weights.upsamples.len()
    }
}

impl VibeVoiceAcousticDecoderStream {
    /// Zeros established causal history while preserving its buffer shape.
    pub fn set_to_zero(&mut self) {
        for cache in &mut self.caches {
            cache.fill(0.0);
        }
    }

    /// Clears causal history so a following sample cannot inherit decoder state.
    pub fn reset(&mut self) {
        self.caches.fill(Vec::new());
    }

    /// Decodes one 3200-sample-aligned latent chunk, transactionally.
    pub fn decode_chunk(&mut self, latents: &[f32], frames: usize) -> Result<Vec<f32>> {
        if frames == 0 || latents.len() != frames * 64 {
            return Err(VokraError::InvalidArgument(
                "vibevoice acoustic decoder stream requires [frames,64] latents".to_owned(),
            ));
        }
        finite("vibevoice acoustic decoder stream latents", latents)?;
        let mut staged = self.caches.clone();
        let result = self.decode_chunk_inner(latents, frames, &mut staged);
        if result.is_ok() {
            self.caches = staged;
        }
        result
    }

    fn decode_chunk_inner(
        &self,
        latents: &[f32],
        frames: usize,
        caches: &mut [Vec<f32>],
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.decoder.backend, VIBEVOICE_TOKENIZER_HOT_OPS)?;
        let input = transpose_tc_to_ct(latents, frames, 64)?;
        let mut cursor = 0;
        let mut time = frames;
        let mut x = causal_conv_stream(
            &compute,
            &input,
            64,
            &self.decoder.weights.stem,
            &mut caches[cursor],
        )?;
        cursor += 1;
        for (stage, blocks) in self.decoder.weights.stages.iter().enumerate() {
            let channels = [2048, 1024, 512, 256, 128, 64, 32][stage];
            for block in blocks {
                x = block_forward_stream(&compute, &x, channels, time, block, &mut caches[cursor])?;
                cursor += 1;
            }
            if let Some(upsample) = self.decoder.weights.upsamples.get(stage) {
                x = transpose_conv_stream(&compute, &x, channels, upsample, &mut caches[cursor])?;
                cursor += 1;
                time = time.checked_mul(upsample.stride).ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "vibevoice decoder stream time extent overflows".to_owned(),
                    )
                })?;
            }
        }
        let output = causal_conv_stream(
            &compute,
            &x,
            32,
            &self.decoder.weights.head,
            &mut caches[cursor],
        )?;
        if output.len() != decoder_pcm_len(frames)? {
            return Err(VokraError::ModelLoad(
                "vibevoice acoustic decoder stream output length mismatch".to_owned(),
            ));
        }
        let output = output;
        finite("vibevoice acoustic decoder stream PCM", &output)?;
        Ok(output)
    }
}

fn decoder_pcm_len(frames: usize) -> Result<usize> {
    frames.checked_mul(HOP).ok_or_else(|| {
        VokraError::InvalidArgument("vibevoice decoder frame count overflows PCM length".to_owned())
    })
}

fn load_blocks(
    file: &GgufFile,
    prefix: &str,
    stage: usize,
    channels: usize,
    depth: usize,
) -> Result<Vec<Block>> {
    let mut blocks = Vec::with_capacity(depth);
    for block in 0..depth {
        let p = format!("{prefix}.stages.{stage}.{block}");
        blocks.push(Block {
            norm: load_raw(file, &format!("{p}.norm.weight"), channels)?,
            mixer: load_conv(
                file,
                &format!("{p}.mixer.conv.conv.conv"),
                channels,
                channels,
                KERNEL,
                1,
                channels,
            )?,
            gamma: load_raw(file, &format!("{p}.gamma"), channels)?,
            ffn_norm: load_raw(file, &format!("{p}.ffn_norm.weight"), channels)?,
            linear1: load_linear(file, &format!("{p}.ffn.linear1"), channels, 4 * channels)?,
            linear2: load_linear(file, &format!("{p}.ffn.linear2"), 4 * channels, channels)?,
            ffn_gamma: load_raw(file, &format!("{p}.ffn_gamma"), channels)?,
        });
    }
    Ok(blocks)
}

fn load_transpose_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
) -> Result<TransposeConv> {
    let weight = load_tensor(
        file,
        "vibevoice tokenizer decoder",
        &format!("{prefix}.weight"),
        &[input, output, kernel],
    )?;
    let bias = load_raw(file, &format!("{prefix}.bias"), output)?;
    Ok(TransposeConv {
        weight,
        bias,
        input,
        output,
        kernel,
        stride,
    })
}

fn validate_decoder_weights(weights: &DecoderWeights) -> Result<()> {
    let channels = [2048, 1024, 512, 256, 128, 64, 32];
    let depths = [8, 3, 3, 3, 3, 3, 3];
    if weights.stem.input != 64
        || weights.stem.output != channels[0]
        || weights.stem.kernel != 7
        || weights.stem.stride != 1
        || weights.upsamples.len() != 6
        || weights.stages.len() != channels.len()
        || weights.head.input != 32
        || weights.head.output != 1
        || weights.head.kernel != 7
        || weights.head.stride != 1
    {
        return Err(VokraError::ModelLoad(
            "vibevoice acoustic decoder fixed topology mismatch".to_owned(),
        ));
    }
    let specs = [
        (2048, 1024, 16, 8),
        (1024, 512, 10, 5),
        (512, 256, 10, 5),
        (256, 128, 8, 4),
        (128, 64, 4, 2),
        (64, 32, 4, 2),
    ];
    for (stage, blocks) in weights.stages.iter().enumerate() {
        if blocks.len() != depths[stage]
            || blocks.iter().any(|block| {
                block.norm.len() != channels[stage]
                    || block.gamma.len() != channels[stage]
                    || block.ffn_norm.len() != channels[stage]
                    || block.ffn_gamma.len() != channels[stage]
                    || block.mixer.input != channels[stage]
                    || block.mixer.output != channels[stage]
                    || block.mixer.groups != channels[stage]
                    || block.linear1.input != channels[stage]
                    || block.linear1.output != channels[stage] * 4
                    || block.linear2.input != channels[stage] * 4
                    || block.linear2.output != channels[stage]
            })
        {
            return Err(VokraError::ModelLoad(
                "vibevoice acoustic decoder stage shape mismatch".to_owned(),
            ));
        }
    }
    for (actual, expected) in weights.upsamples.iter().zip(specs) {
        if (actual.input, actual.output, actual.kernel, actual.stride) != expected {
            return Err(VokraError::ModelLoad(
                "vibevoice acoustic decoder ConvTranspose shape mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn decode_channel_major(
    compute: &Compute,
    input: &[f32],
    frames: usize,
    weights: &DecoderWeights,
) -> Result<Vec<f32>> {
    let mut x = causal_conv(compute, input, 64, &weights.stem)?;
    let mut time = frames;
    let channels = [2048, 1024, 512, 256, 128, 64, 32];
    for (stage, blocks) in weights.stages.iter().enumerate() {
        for block in blocks {
            x = block_forward(compute, &x, channels[stage], time, block)?;
        }
        if let Some(upsample) = weights.upsamples.get(stage) {
            x = transpose_conv(compute, &x, channels[stage], upsample)?;
            time = time.checked_mul(upsample.stride).ok_or_else(|| {
                VokraError::InvalidArgument("vibevoice decoder time extent overflows".to_owned())
            })?;
        }
    }
    let output = causal_conv(compute, &x, 32, &weights.head)?;
    if output.len() != time {
        return Err(VokraError::ModelLoad(
            "vibevoice acoustic decoder output length mismatch".to_owned(),
        ));
    }
    finite("vibevoice acoustic decoder PCM", &output)?;
    Ok(output)
}

fn transpose_conv(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    conv: &TransposeConv,
) -> Result<Vec<f32>> {
    if channels != conv.input || input.len() % channels != 0 || conv.kernel < conv.stride {
        return Err(VokraError::InvalidArgument(
            "vibevoice ConvTranspose input/topology mismatch".to_owned(),
        ));
    }
    let time = input.len() / channels;
    if time == 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice ConvTranspose input is empty".to_owned(),
        ));
    }
    let ops = HifiGanComputeOps { compute };
    let raw = ops.conv_transpose1d(
        input,
        conv.input,
        time,
        &conv.weight,
        conv.output,
        conv.kernel,
        Some(&conv.bias),
        conv.stride,
        0,
    )?;
    let raw_time = (time - 1) * conv.stride + conv.kernel;
    let target = time * conv.stride;
    if raw.len() != conv.output * raw_time || conv.kernel - conv.stride > raw_time {
        return Err(VokraError::ModelLoad(
            "vibevoice ConvTranspose raw output extent mismatch".to_owned(),
        ));
    }
    let trim = conv.kernel - conv.stride;
    let mut output = vec![0.0; conv.output * target];
    for channel in 0..conv.output {
        output[channel * target..(channel + 1) * target]
            .copy_from_slice(&raw[channel * raw_time..channel * raw_time + target]);
        debug_assert_eq!(trim + target, raw_time);
    }
    finite("vibevoice ConvTranspose", &output)?;
    Ok(output)
}

fn transpose_conv_stream(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    conv: &TransposeConv,
    cache: &mut Vec<f32>,
) -> Result<Vec<f32>> {
    if input.len() % channels != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice ConvTranspose stream channel mismatch".to_owned(),
        ));
    }
    let time = input.len() / channels;
    if time == 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice ConvTranspose stream input is empty".to_owned(),
        ));
    }
    let context = conv.kernel - 1;
    if cache.len() % channels != 0 || cache.len() > channels * context {
        return Err(VokraError::ModelLoad(
            "vibevoice ConvTranspose stream cache shape mismatch".to_owned(),
        ));
    }
    let cached_frames = cache.len() / channels;
    let has_cache = cached_frames != 0;
    let joined_time = cached_frames + time;
    let mut joined = vec![0.0; channels * joined_time];
    if has_cache {
        for channel in 0..channels {
            joined[channel * joined_time..channel * joined_time + cached_frames]
                .copy_from_slice(&cache[channel * cached_frames..(channel + 1) * cached_frames]);
            joined[channel * joined_time + cached_frames..(channel + 1) * joined_time]
                .copy_from_slice(&input[channel * time..(channel + 1) * time]);
        }
    } else {
        joined.copy_from_slice(input);
    }
    let full = transpose_conv(compute, &joined, channels, conv)?;
    let output_len = time * conv.stride;
    let output = if has_cache {
        let full_time = joined_time * conv.stride;
        let mut tail = vec![0.0; conv.output * output_len];
        for channel in 0..conv.output {
            tail[channel * output_len..(channel + 1) * output_len].copy_from_slice(
                &full[channel * full_time + (full_time - output_len)
                    ..channel * full_time + full_time],
            );
        }
        tail
    } else {
        full
    };
    let kept = joined_time.min(context);
    let mut next_cache = vec![0.0; channels * kept];
    for channel in 0..channels {
        next_cache[channel * kept..(channel + 1) * kept].copy_from_slice(
            &joined[channel * joined_time + joined_time - kept..(channel + 1) * joined_time],
        );
    }
    *cache = next_cache;
    Ok(output)
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
) -> Result<Conv> {
    let weight = load_tensor(
        file,
        "vibevoice tokenizer",
        &format!("{prefix}.weight"),
        &[output, input / groups, kernel],
    )?;
    let bias = load_raw(file, &format!("{prefix}.bias"), output)?;
    Ok(Conv {
        weight,
        bias,
        input,
        output,
        kernel,
        stride,
        groups,
    })
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    let weight = load_tensor(
        file,
        "vibevoice tokenizer",
        &format!("{prefix}.weight"),
        &[output, input],
    )?;
    let bias = load_raw(file, &format!("{prefix}.bias"), output)?;
    let mut transposed = vec![0.0; input * output];
    for row in 0..output {
        for col in 0..input {
            transposed[col * output + row] = weight[row * input + col];
        }
    }
    Ok(Linear {
        weight: transposed,
        bias,
        input,
        output,
    })
}

fn load_raw(file: &GgufFile, name: &str, width: usize) -> Result<Vec<f32>> {
    require_tensor_shape(file, "vibevoice tokenizer", name, &[width])?;
    load_tensor(file, "vibevoice tokenizer", name, &[width])
}

fn validate_weights(weights: &EncoderWeights, kind: TokenizerKind) -> Result<()> {
    if weights.downsamples.len() != DOWN_STRIDES.len()
        || weights.stages.len() != CHANNELS.len()
        || weights.head.input != CHANNELS[6]
        || weights.head.output != kind.output_dim()
    {
        return Err(VokraError::ModelLoad(
            "vibevoice tokenizer fixed encoder shape contract mismatch".to_owned(),
        ));
    }
    for (index, down) in weights.downsamples.iter().enumerate() {
        let expected_input = if index == 0 { 1 } else { CHANNELS[index - 1] };
        if down.input != expected_input
            || down.output != CHANNELS[index]
            || down.kernel != DOWN_KERNELS[index]
            || down.stride != DOWN_STRIDES[index]
            || down.groups != 1
        {
            return Err(VokraError::ModelLoad(
                "vibevoice tokenizer downsample shape contract mismatch".to_owned(),
            ));
        }
    }
    for (stage, blocks) in weights.stages.iter().enumerate() {
        if blocks.len() != DEPTHS[stage] {
            return Err(VokraError::ModelLoad(
                "vibevoice tokenizer stage depth contract mismatch".to_owned(),
            ));
        }
        let channels = CHANNELS[stage];
        for block in blocks {
            if block.norm.len() != channels
                || block.gamma.len() != channels
                || block.ffn_norm.len() != channels
                || block.ffn_gamma.len() != channels
                || block.mixer.input != channels
                || block.mixer.output != channels
                || block.mixer.groups != channels
            {
                return Err(VokraError::ModelLoad(
                    "vibevoice tokenizer block shape contract mismatch".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn causal_conv(compute: &Compute, input: &[f32], channels: usize, conv: &Conv) -> Result<Vec<f32>> {
    if input.len() % channels != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer channel-major input mismatch".to_owned(),
        ));
    }
    let time = input.len() / channels;
    let left = conv.kernel.saturating_sub(conv.stride);
    let right = extra_padding(time, conv.kernel, conv.stride);
    let mut padded = vec![0.0; channels * (left + time + right)];
    for channel in 0..channels {
        padded
            [channel * (left + time + right) + left..channel * (left + time + right) + left + time]
            .copy_from_slice(&input[channel * time..(channel + 1) * time]);
    }
    conv_apply(compute, &padded, channels, left + time + right, conv, 0)
}

fn causal_conv_stream(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    conv: &Conv,
    cache: &mut Vec<f32>,
) -> Result<Vec<f32>> {
    if input.len() % channels != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer stream channel mismatch".to_owned(),
        ));
    }
    let time = input.len() / channels;
    if time < conv.stride {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer stream chunk is shorter than convolution stride".to_owned(),
        ));
    }
    let context = conv.kernel.saturating_sub(conv.stride);
    if cache.len() != channels * context {
        cache.clear();
        cache.resize(channels * context, 0.0);
    }
    let mut joined = vec![0.0; channels * (context + time)];
    for channel in 0..channels {
        joined[channel * (context + time)..channel * (context + time) + context]
            .copy_from_slice(&cache[channel * context..(channel + 1) * context]);
        joined[channel * (context + time) + context..channel * (context + time) + context + time]
            .copy_from_slice(&input[channel * time..(channel + 1) * time]);
    }
    let output = conv_apply(compute, &joined, channels, context + time, conv, 0)?;
    for channel in 0..channels {
        cache[channel * context..(channel + 1) * context].copy_from_slice(
            &joined[channel * (context + time) + time..channel * (context + time) + time + context],
        );
    }
    Ok(output)
}

fn conv_apply(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    time: usize,
    conv: &Conv,
    padding: usize,
) -> Result<Vec<f32>> {
    if channels != conv.input
        || conv.stride == 0
        || conv.kernel == 0
        || conv.groups == 0
        || conv.input % conv.groups != 0
        || conv.output % conv.groups != 0
        || time + 2 * padding < conv.kernel
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer convolution shape/extent mismatch".to_owned(),
        ));
    }
    let out_time = (time + 2 * padding - conv.kernel) / conv.stride + 1;
    let mut output = vec![0.0; conv.output * out_time];
    if conv.groups == 1 {
        compute.conv1d_f32(
            input,
            channels,
            time,
            &conv.weight,
            conv.output,
            conv.kernel,
            Some(&conv.bias),
            conv.stride,
            padding,
            &mut output,
        )?;
    } else {
        compute.grouped_conv1d_f32(
            input,
            channels,
            time,
            &conv.weight,
            conv.output,
            conv.kernel,
            Some(&conv.bias),
            conv.stride,
            padding,
            conv.groups,
            &mut output,
        )?;
    }
    finite("vibevoice tokenizer convolution", &output)?;
    Ok(output)
}

fn block_forward(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    time: usize,
    block: &Block,
) -> Result<Vec<f32>> {
    let rows = transpose_ct_to_tc(input, channels, time)?;
    let normed = rms_rows(compute, &rows, &block.norm, time, channels)?;
    let mixed = causal_conv(
        compute,
        &transpose_tc_to_ct(&normed, time, channels)?,
        channels,
        &block.mixer,
    )?;
    let mut mixed_rows = transpose_ct_to_tc(&mixed, channels, time)?;
    for row in mixed_rows.chunks_exact_mut(channels) {
        for (value, gamma) in row.iter_mut().zip(&block.gamma) {
            *value *= *gamma;
        }
    }
    let mut residual = rows.clone();
    add_scaled_rows(&mut residual, &mixed_rows, 1.0)?;
    let normed = rms_rows(compute, &residual, &block.ffn_norm, time, channels)?;
    let mut first = block.linear1.apply(compute, &normed, time)?;
    let first_input = first.clone();
    compute.gelu_f32(&first_input, &mut first)?;
    let second = block.linear2.apply(compute, &first, time)?;
    let mut output = residual;
    for (row, values) in output
        .chunks_exact_mut(channels)
        .zip(second.chunks_exact(channels))
    {
        for ((dst, value), gamma) in row.iter_mut().zip(values).zip(&block.ffn_gamma) {
            *dst += value * gamma;
        }
    }
    transpose_tc_to_ct(&output, time, channels)
}

fn block_forward_stream(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    time: usize,
    block: &Block,
    cache: &mut Vec<f32>,
) -> Result<Vec<f32>> {
    let rows = transpose_ct_to_tc(input, channels, time)?;
    let normed = rms_rows(compute, &rows, &block.norm, time, channels)?;
    let mixed = causal_conv_stream(
        compute,
        &transpose_tc_to_ct(&normed, time, channels)?,
        channels,
        &block.mixer,
        cache,
    )?;
    let mut mixed_rows = transpose_ct_to_tc(&mixed, channels, time)?;
    for row in mixed_rows.chunks_exact_mut(channels) {
        for (value, gamma) in row.iter_mut().zip(&block.gamma) {
            *value *= *gamma;
        }
    }
    let mut residual = rows.clone();
    for (row, mixed) in residual
        .chunks_exact_mut(channels)
        .zip(mixed_rows.chunks_exact(channels))
    {
        for (dst, value) in row.iter_mut().zip(mixed) {
            *dst += value;
        }
    }
    let normed = rms_rows(compute, &residual, &block.ffn_norm, time, channels)?;
    let mut first = block.linear1.apply(compute, &normed, time)?;
    let first_input = first.clone();
    compute.gelu_f32(&first_input, &mut first)?;
    let second = block.linear2.apply(compute, &first, time)?;
    for (row, values) in residual
        .chunks_exact_mut(channels)
        .zip(second.chunks_exact(channels))
    {
        for ((dst, value), gamma) in row.iter_mut().zip(values).zip(&block.ffn_gamma) {
            *dst += value * gamma;
        }
    }
    transpose_tc_to_ct(&residual, time, channels)
}

impl Linear {
    fn apply(&self, compute: &Compute, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        if rows == 0 || input.len() != rows * self.input {
            return Err(VokraError::InvalidArgument(
                "vibevoice tokenizer linear shape mismatch".to_owned(),
            ));
        }
        let mut output = vec![0.0; rows * self.output];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight,
            Some(&self.bias),
            &mut output,
        )?;
        finite("vibevoice tokenizer linear", &output)?;
        Ok(output)
    }
}

fn rms_rows(
    compute: &Compute,
    input: &[f32],
    gamma: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, rows, width, gamma, EPS)?;
    finite("vibevoice tokenizer RMSNorm", &output)?;
    Ok(output)
}

fn add_scaled_rows(dst: &mut [f32], src: &[f32], scale: f32) -> Result<()> {
    if dst.len() != src.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer residual shape mismatch".to_owned(),
        ));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += scale * src;
    }
    finite("vibevoice tokenizer residual", dst)
}

fn transpose_ct_to_tc(input: &[f32], channels: usize, time: usize) -> Result<Vec<f32>> {
    if input.len() != channels * time {
        return Err(VokraError::InvalidArgument(
            "vibevoice tokenizer transpose shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for position in 0..time {
            output[position * channels + channel] = input[channel * time + position];
        }
    }
    Ok(output)
}

fn transpose_tc_to_ct(input: &[f32], time: usize, channels: usize) -> Result<Vec<f32>> {
    transpose_ct_to_tc(input, time, channels)
}

fn channel_major_to_rows(input: &[f32], time: usize, output_dim: usize) -> Result<Vec<f32>> {
    transpose_ct_to_tc(input, output_dim, time)
}

fn extra_padding(time: usize, _kernel: usize, stride: usize) -> usize {
    if stride == 0 {
        return 0;
    }
    (stride - time % stride) % stride
}

fn finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_contract_is_explicit() {
        assert_eq!(RATIOS, [2, 2, 4, 5, 5, 8]);
        assert_eq!(DOWN_STRIDES, [1, 2, 2, 4, 5, 5, 8]);
        assert_eq!(DOWN_KERNELS, [7, 4, 4, 8, 10, 10, 16]);
        assert_eq!(CHANNELS, [32, 64, 128, 256, 512, 1024, 2048]);
        assert_eq!(DEPTHS, [3, 3, 3, 3, 3, 3, 8]);
        assert_eq!(HOP, 3200);
        assert_eq!(GAMMA, 1.0e-6);
        let expected_downsample_shapes = [
            ("downsample_layers.0.0.conv.conv", [32, 1, 7]),
            ("downsample_layers.1.0.conv.conv", [64, 32, 4]),
            ("downsample_layers.2.0.conv.conv", [128, 64, 4]),
            ("downsample_layers.3.0.conv.conv", [256, 128, 8]),
            ("downsample_layers.4.0.conv.conv", [512, 256, 10]),
            ("downsample_layers.5.0.conv.conv", [1024, 512, 10]),
            ("downsample_layers.6.0.conv.conv", [2048, 1024, 16]),
        ];
        for (index, (name, shape)) in expected_downsample_shapes.iter().enumerate() {
            assert_eq!(*name, format!("downsample_layers.{index}.0.conv.conv"));
            assert_eq!(
                *shape,
                [
                    CHANNELS[index],
                    if index == 0 { 1 } else { CHANNELS[index - 1] },
                    DOWN_KERNELS[index]
                ]
            );
        }
    }

    #[test]
    fn extra_padding_produces_ceil_stride_output() {
        for time in 1..32 {
            let stride = 5;
            let kernel = 10;
            let left = kernel - stride;
            let right = extra_padding(time, kernel, stride);
            assert_eq!(
                (left + time + right - kernel) / stride + 1,
                time.div_ceil(stride)
            );
        }
    }

    #[test]
    fn tiny_causal_conv_scalar_oracle_and_stream_cache() {
        let conv = Conv {
            weight: vec![1.0, 2.0, 3.0, 4.0],
            bias: vec![0.5],
            input: 1,
            output: 1,
            kernel: 4,
            stride: 2,
            groups: 1,
        };
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = [
            0.5 + 1.0 * 3.0 + 2.0 * 4.0,
            0.5 + 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0,
            0.5 + 3.0 + 4.0 * 2.0 + 5.0 * 3.0 + 6.0 * 4.0,
        ];
        let actual = causal_conv(&Compute::cpu(), &input, 1, &conv).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        let mut cache = vec![];
        let first = causal_conv_stream(&Compute::cpu(), &input[..4], 1, &conv, &mut cache).unwrap();
        let second =
            causal_conv_stream(&Compute::cpu(), &input[4..], 1, &conv, &mut cache).unwrap();
        assert_eq!([first, second].concat(), actual);
    }

    #[test]
    fn transpose_and_rms_block_order_are_deterministic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            transpose_ct_to_tc(&input, 2, 2).unwrap(),
            [1.0, 3.0, 2.0, 4.0]
        );
        let gamma = [1.0, 2.0];
        let output = rms_rows(&Compute::cpu(), &[1.0, 2.0, 3.0, 4.0], &gamma, 2, 2).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn streaming_block_matches_nonstream_without_double_gamma() {
        let block = tiny_block();
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let compute = Compute::cpu();
        let expected = block_forward(&compute, &input, 1, 4, &block).unwrap();
        let mut cache = Vec::new();
        let first = block_forward_stream(&compute, &input[..2], 1, 2, &block, &mut cache).unwrap();
        let second = block_forward_stream(&compute, &input[2..], 1, 2, &block, &mut cache).unwrap();
        assert_eq!([first, second].concat(), expected);
    }

    #[test]
    fn decoder_topology_uses_exact_hf_names_and_shapes() {
        let channels = [2048, 1024, 512, 256, 128, 64, 32];
        let depths = [8, 3, 3, 3, 3, 3, 3];
        let upsample_names = [
            "decoder.upsample_layers.1.0.convtr.convtr",
            "decoder.upsample_layers.2.0.convtr.convtr",
            "decoder.upsample_layers.3.0.convtr.convtr",
            "decoder.upsample_layers.4.0.convtr.convtr",
            "decoder.upsample_layers.5.0.convtr.convtr",
            "decoder.upsample_layers.6.0.convtr.convtr",
        ];
        let transpose = [
            (2048, 1024, 16, 8),
            (1024, 512, 10, 5),
            (512, 256, 10, 5),
            (256, 128, 8, 4),
            (128, 64, 4, 2),
            (64, 32, 4, 2),
        ];
        assert_eq!((64, channels[0], 7), (64, 2048, 7));
        assert_eq!((channels[6], 1, 7), (32, 1, 7));
        assert_eq!(depths, [8, 3, 3, 3, 3, 3, 3]);
        for (index, (name, shape)) in upsample_names.iter().zip(transpose).enumerate() {
            assert_eq!(
                *name,
                format!("decoder.upsample_layers.{}.0.convtr.convtr", index + 1)
            );
            assert_eq!(shape.0, channels[index]);
            assert_eq!(shape.1, channels[index + 1]);
            assert!(shape.2 >= shape.3);
        }
    }

    #[test]
    fn conv_transpose_matches_independent_scalar_and_layout_oracle() {
        let conv = TransposeConv {
            // Raw PyTorch layout [input, output, kernel].
            weight: vec![
                1.0, 2.0, 3.0, 4.0, // input 0 -> outputs 0,1 are interleaved below
                0.5, 1.5, 2.5, 3.5, -1.0, -2.0, -3.0, -4.0, 0.25, 0.75, 1.25, 1.75,
            ],
            bias: vec![0.25, -0.5],
            input: 2,
            output: 2,
            kernel: 4,
            stride: 2,
        };
        let input = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0]; // [2 channels, 3 frames]
        let actual = transpose_conv(&Compute::cpu(), &input, 2, &conv).unwrap();
        let raw_time = (3 - 1) * 2 + 4;
        let mut raw = vec![0.0; 2 * raw_time];
        for output_channel in 0..2 {
            for position in 0..raw_time {
                raw[output_channel * raw_time + position] = conv.bias[output_channel];
            }
            for frame in 0..3 {
                for input_channel in 0..2 {
                    for tap in 0..4 {
                        let sample = input[input_channel * 3 + frame];
                        let weight = conv.weight[(input_channel * 2 + output_channel) * 4 + tap];
                        raw[output_channel * raw_time + frame * 2 + tap] += sample * weight;
                    }
                }
            }
        }
        let mut expected = Vec::new();
        for channel in 0..2 {
            expected.extend_from_slice(&raw[channel * raw_time..channel * raw_time + 3 * 2]);
        }
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn conv_transpose_stream_matches_nonstream_for_aligned_chunks() {
        let conv = TransposeConv {
            weight: vec![1.0, -0.5, 0.25, 0.75],
            bias: vec![0.1],
            input: 1,
            output: 1,
            kernel: 4,
            stride: 2,
        };
        let input = [1.0, 2.0, -1.0, 0.5];
        let compute = Compute::cpu();
        let expected = transpose_conv(&compute, &input, 1, &conv).unwrap();
        let mut cache = Vec::new();
        let first = transpose_conv_stream(&compute, &input[..1], 1, &conv, &mut cache).unwrap();
        assert_eq!(cache, vec![input[0]]);
        let second = transpose_conv_stream(&compute, &input[1..3], 1, &conv, &mut cache).unwrap();
        let third = transpose_conv_stream(&compute, &input[3..], 1, &conv, &mut cache).unwrap();
        let actual = [first, second, third].concat();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn one_latent_frame_is_exactly_3200_pcm_samples() {
        assert_eq!(HOP, 3200);
        assert_eq!(decoder_pcm_len(1).unwrap(), 3200);
    }

    #[test]
    fn decoder_stream_rejects_nonfinite_before_cache_mutation() {
        let conv = TransposeConv {
            weight: vec![1.0],
            bias: vec![0.0],
            input: 1,
            output: 1,
            kernel: 1,
            stride: 1,
        };
        let mut cache = Vec::new();
        let before = cache.clone();
        let result = transpose_conv_stream(&Compute::cpu(), &[f32::NAN], 1, &conv, &mut cache);
        assert!(result.is_err());
        assert_eq!(cache, before);
    }

    #[test]
    fn decoder_backend_registry_is_explicit() {
        assert!(Compute::for_backend(BackendKind::Cpu, VIBEVOICE_TOKENIZER_HOT_OPS).is_ok());
    }

    #[test]
    fn set_to_zero_preserves_established_cache_shapes() {
        let encoder = VibeVoiceTokenizerEncoder {
            weights: Arc::new(EncoderWeights {
                downsamples: Vec::new(),
                stages: Vec::new(),
                head: Conv {
                    weight: vec![],
                    bias: vec![],
                    input: 0,
                    output: 0,
                    kernel: 0,
                    stride: 0,
                    groups: 0,
                },
            }),
            kind: TokenizerKind::Acoustic,
            backend: BackendKind::Cpu,
        };
        let mut encoder_stream = VibeVoiceTokenizerStream {
            encoder,
            caches: vec![vec![1.0, -2.0], vec![3.0]],
        };
        let shapes = encoder_stream
            .caches
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        encoder_stream.set_to_zero();
        assert_eq!(
            encoder_stream
                .caches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            shapes
        );
        assert!(
            encoder_stream
                .caches
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );

        let decoder = VibeVoiceAcousticDecoder {
            weights: Arc::new(DecoderWeights {
                stem: Conv {
                    weight: vec![],
                    bias: vec![],
                    input: 0,
                    output: 0,
                    kernel: 0,
                    stride: 0,
                    groups: 0,
                },
                stages: Vec::new(),
                upsamples: Vec::new(),
                head: Conv {
                    weight: vec![],
                    bias: vec![],
                    input: 0,
                    output: 0,
                    kernel: 0,
                    stride: 0,
                    groups: 0,
                },
            }),
            backend: BackendKind::Cpu,
        };
        let mut decoder_stream = VibeVoiceAcousticDecoderStream {
            decoder,
            caches: vec![vec![4.0], vec![-5.0, 6.0]],
        };
        let shapes = decoder_stream
            .caches
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>();
        decoder_stream.set_to_zero();
        assert_eq!(
            decoder_stream
                .caches
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            shapes
        );
        assert!(
            decoder_stream
                .caches
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );
    }

    fn tiny_block() -> Block {
        Block {
            norm: vec![1.0],
            mixer: Conv {
                weight: vec![1.0],
                bias: vec![0.0],
                input: 1,
                output: 1,
                kernel: 1,
                stride: 1,
                groups: 1,
            },
            gamma: vec![0.5],
            ffn_norm: vec![1.0],
            linear1: Linear {
                weight: vec![1.0; 4],
                bias: vec![0.0; 4],
                input: 1,
                output: 4,
            },
            linear2: Linear {
                weight: vec![1.0; 4],
                bias: vec![0.0],
                input: 4,
                output: 1,
            },
            ffn_gamma: vec![0.25],
        }
    }
}
