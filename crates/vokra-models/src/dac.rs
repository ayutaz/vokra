//! Native Descript Audio Codec (DAC) token-to-PCM runtime.
//!
//! The public 16 kHz, 24 kHz, and 44.1 kHz GGUFs contain the complete
//! upstream encoder, factorized residual vector quantizer, and SEANet decoder.
//! This module binds that exact tensor manifest and executes the released
//! token-to-waveform path.  The existing [`crate::codec::DacCodecGguf`]
//! performs the codebook gather/projection; the decoder below mirrors
//! `descriptinc/descript-audio-codec/dac/model/dac.py` from the 1.0.0 release:
//! weight-normalized convolutions, plain Snake activations, three residual
//! units per upsample stage, and terminal `tanh`.
//!
//! All learned operations use [`crate::compute::Compute`].  DAC RVQ, Conv1D
//! (including the exact ConvTranspose1D layout transform), and Snake are
//! Metal-covered.  A backend missing any member of [`DAC_HOT_OPS`] is rejected
//! before inference; there is no CPU fallback.

use std::collections::{BTreeMap, BTreeSet};

use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};
use vokra_ops::hifigan::{HifiGanBackendOps, HifiGanConvPadding};

use crate::codec::DacCodecGguf;
use crate::compute::{Compute, HotOp};
use crate::hifigan::HifiGanComputeOps;

/// Public converter/runtime architecture tag.
pub const ARCH: &str = "dac";
/// Complete learned-op set for the released token-to-PCM path.
pub const DAC_HOT_OPS: &[HotOp] = &[HotOp::DacRvq, HotOp::Conv1d, HotOp::SnakeActivation];

const NAME: &str = "DAC (Descript Audio Codec)";
const ENCODER_DIM: usize = 64;
const LATENT_DIM: usize = 1024;
const DECODER_DIM: usize = 1536;
const CODEBOOK_SIZE: usize = 1024;
const CODEBOOK_DIM: usize = 8;

/// Exact released DAC variants represented by the three public GGUFs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DacVariant {
    /// The released 16 kHz, 12-codebook checkpoint.
    Khz16,
    /// The released 24 kHz, 32-codebook checkpoint.
    Khz24,
    /// The released 44.1 kHz, 9-codebook checkpoint.
    Khz44,
}

impl DacVariant {
    const fn sample_rate(self) -> u32 {
        match self {
            Self::Khz16 => 16_000,
            Self::Khz24 => 24_000,
            Self::Khz44 => 44_100,
        }
    }

    const fn hop_length(self) -> usize {
        match self {
            Self::Khz16 | Self::Khz24 => 320,
            Self::Khz44 => 512,
        }
    }

    const fn n_codebooks(self) -> usize {
        match self {
            Self::Khz16 => 12,
            Self::Khz24 => 32,
            Self::Khz44 => 9,
        }
    }

    const fn encoder_rates(self) -> &'static [usize; 4] {
        match self {
            Self::Khz16 | Self::Khz24 => &[2, 4, 5, 8],
            Self::Khz44 => &[2, 4, 8, 8],
        }
    }

    const fn decoder_rates(self) -> &'static [usize; 4] {
        match self {
            Self::Khz16 | Self::Khz24 => &[8, 5, 4, 2],
            Self::Khz44 => &[8, 8, 4, 2],
        }
    }
}

fn metadata_u32(file: &GgufFile, key: &str) -> Result<u32> {
    file.get(key)
        .and_then(|value| value.as_u64())
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "dac: required u32 metadata `{key}` is missing or invalid"
            ))
        })
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(|value| value.as_str());
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "dac: metadata `{key}` = {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn variant_from_metadata(file: &GgufFile) -> Result<DacVariant> {
    let rate = metadata_u32(file, "vokra.dac.sample_rate")?;
    let hop = metadata_u32(file, "vokra.dac.hop_length")? as usize;
    let books = metadata_u32(file, "vokra.dac.n_codebooks")? as usize;
    let variant = match (rate, hop, books) {
        (16_000, 320, 12) => DacVariant::Khz16,
        (24_000, 320, 32) => DacVariant::Khz24,
        (44_100, 512, 9) => DacVariant::Khz44,
        other => {
            return Err(VokraError::ModelLoad(format!(
                "dac: unsupported released variant (sample_rate, hop_length, n_codebooks) = {other:?}; expected (16000,320,12), (24000,320,32), or (44100,512,9)"
            )));
        }
    };
    if metadata_u32(file, "vokra.dac.codebook_size")? as usize != CODEBOOK_SIZE
        || metadata_u32(file, "vokra.dac.codebook_dim")? as usize != CODEBOOK_DIM
        || metadata_u32(file, "vokra.dac.d_model")? as usize != LATENT_DIM
    {
        return Err(VokraError::ModelLoad(
            "dac: codebook_size/codebook_dim/d_model metadata does not match the released 1024/8/1024 contract"
                .to_owned(),
        ));
    }
    Ok(variant)
}

type Manifest = BTreeMap<String, Vec<usize>>;

fn add(manifest: &mut Manifest, name: impl Into<String>, shape: &[usize]) {
    let previous = manifest.insert(name.into(), shape.to_vec());
    debug_assert!(previous.is_none(), "duplicate DAC manifest entry");
}

fn add_snake(manifest: &mut Manifest, name: &str, channels: usize) {
    add(manifest, name, &[1, channels, 1]);
}

fn add_conv1d(
    manifest: &mut Manifest,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) {
    add(manifest, format!("{prefix}.bias"), &[out_channels]);
    add(
        manifest,
        format!("{prefix}.weight_g"),
        &[out_channels, 1, 1],
    );
    add(
        manifest,
        format!("{prefix}.weight_v"),
        &[out_channels, in_channels, kernel],
    );
}

fn add_conv_transpose1d(
    manifest: &mut Manifest,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) {
    add(manifest, format!("{prefix}.bias"), &[out_channels]);
    add(manifest, format!("{prefix}.weight_g"), &[in_channels, 1, 1]);
    add(
        manifest,
        format!("{prefix}.weight_v"),
        &[in_channels, out_channels, kernel],
    );
}

fn add_residual(manifest: &mut Manifest, prefix: &str, channels: usize) {
    add_snake(manifest, &format!("{prefix}.block.0.alpha"), channels);
    add_conv1d(
        manifest,
        &format!("{prefix}.block.1"),
        channels,
        channels,
        7,
    );
    add_snake(manifest, &format!("{prefix}.block.2.alpha"), channels);
    add_conv1d(
        manifest,
        &format!("{prefix}.block.3"),
        channels,
        channels,
        1,
    );
}

fn expected_manifest(variant: DacVariant) -> Manifest {
    let mut manifest = Manifest::new();

    // Encoder: exact mirror of upstream Encoder/EncoderBlock.  It is not
    // executed by this decoder surface, but remains part of the public-file
    // identity and therefore participates in the strict manifest gate.
    add_conv1d(&mut manifest, "encoder.block.0", 1, ENCODER_DIM, 7);
    let mut channels = ENCODER_DIM;
    for (stage, &stride) in variant.encoder_rates().iter().enumerate() {
        let block = format!("encoder.block.{}.block", stage + 1);
        for residual in 0..3 {
            add_residual(&mut manifest, &format!("{block}.{residual}"), channels);
        }
        add_snake(&mut manifest, &format!("{block}.3.alpha"), channels);
        add_conv1d(
            &mut manifest,
            &format!("{block}.4"),
            channels,
            channels * 2,
            2 * stride,
        );
        channels *= 2;
    }
    add_snake(&mut manifest, "encoder.block.5.alpha", channels);
    add_conv1d(&mut manifest, "encoder.block.6", channels, LATENT_DIM, 3);

    // Decoder.
    add_conv1d(&mut manifest, "decoder.model.0", LATENT_DIM, DECODER_DIM, 7);
    for (stage, &stride) in variant.decoder_rates().iter().enumerate() {
        let in_channels = DECODER_DIM >> stage;
        let out_channels = in_channels / 2;
        let block = format!("decoder.model.{}.block", stage + 1);
        add_snake(&mut manifest, &format!("{block}.0.alpha"), in_channels);
        add_conv_transpose1d(
            &mut manifest,
            &format!("{block}.1"),
            in_channels,
            out_channels,
            2 * stride,
        );
        for residual in 0..3 {
            add_residual(
                &mut manifest,
                &format!("{block}.{}", residual + 2),
                out_channels,
            );
        }
    }
    add_snake(&mut manifest, "decoder.model.5.alpha", DECODER_DIM / 16);
    add_conv1d(&mut manifest, "decoder.model.6", DECODER_DIM / 16, 1, 7);

    // Raw upstream quantizer tensors plus converter-derived decode tensors.
    for index in 0..variant.n_codebooks() {
        let raw = format!("quantizer.quantizers.{index}");
        add_conv1d(
            &mut manifest,
            &format!("{raw}.in_proj"),
            LATENT_DIM,
            CODEBOOK_DIM,
            1,
        );
        add(
            &mut manifest,
            format!("{raw}.codebook.weight"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        );
        add_conv1d(
            &mut manifest,
            &format!("{raw}.out_proj"),
            CODEBOOK_DIM,
            LATENT_DIM,
            1,
        );
        let derived = format!("vokra.dac.quantizer.{index}");
        add(
            &mut manifest,
            format!("{derived}.codebook"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        );
        add(
            &mut manifest,
            format!("{derived}.out_proj_weight"),
            &[LATENT_DIM, CODEBOOK_DIM],
        );
        add(
            &mut manifest,
            format!("{derived}.out_proj_bias"),
            &[LATENT_DIM],
        );
    }
    manifest
}

fn validate_manifest(file: &GgufFile, variant: DacVariant) -> Result<()> {
    let expected = expected_manifest(variant);
    let actual: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|info| info.name.clone())
        .collect();
    let expected_names: BTreeSet<String> = expected.keys().cloned().collect();
    if actual != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual).take(8).collect();
        let extra: Vec<&String> = actual.difference(&expected_names).take(8).collect();
        return Err(VokraError::ModelLoad(format!(
            "dac: tensor manifest mismatch for {variant:?}: expected {}, found {}; missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual.len()
        )));
    }
    for (name, shape) in expected {
        let info = file.tensor_info(&name).expect("name set was checked above");
        let actual_shape: Vec<usize> = info
            .dimensions
            .iter()
            .map(|&dimension| dimension as usize)
            .collect();
        if info.dtype != GgmlType::F32 || actual_shape != shape {
            return Err(VokraError::ModelLoad(format!(
                "dac: tensor `{name}` is {:?} {actual_shape:?}, expected F32 {shape:?}",
                info.dtype
            )));
        }
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>> {
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("dac: tensor `{name}` decode failed: {error}"))
    })
}

fn fold_weight_norm(v: &[f32], g: &[f32], rows: usize, row_width: usize) -> Result<Vec<f32>> {
    if v.len() != rows * row_width || g.len() != rows {
        return Err(VokraError::ModelLoad(format!(
            "dac: weight-norm operands have lengths v={} g={}, expected {} and {rows}",
            v.len(),
            g.len(),
            rows * row_width
        )));
    }
    let mut weight = vec![0.0; v.len()];
    for row in 0..rows {
        let source = &v[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "dac: invalid weight-norm row {row}: norm={norm}, g={}",
                g[row]
            )));
        }
        let scale = g[row] / norm;
        for (destination, source) in weight[row * row_width..(row + 1) * row_width]
            .iter_mut()
            .zip(source)
        {
            *destination = *source * scale;
        }
    }
    Ok(weight)
}

#[derive(Debug, Clone)]
struct Conv1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
}

impl Conv1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
        padding: usize,
    ) -> Result<Self> {
        let g = tensor(file, &format!("{prefix}.weight_g"))?;
        let v = tensor(file, &format!("{prefix}.weight_v"))?;
        let weight = fold_weight_norm(&v, &g, out_channels, in_channels * kernel)?;
        Ok(Self {
            weight,
            bias: tensor(file, &format!("{prefix}.bias"))?,
            in_channels,
            out_channels,
            kernel,
            dilation,
            padding,
        })
    }

    fn forward(
        &self,
        ops: &impl HifiGanBackendOps,
        input: &[f32],
        time: usize,
    ) -> Result<Vec<f32>> {
        ops.conv1d(
            input,
            self.in_channels,
            time,
            &self.weight,
            self.out_channels,
            self.kernel,
            Some(&self.bias),
            1,
            self.dilation,
            self.padding,
            HifiGanConvPadding::Zero,
        )
    }
}

#[derive(Debug, Clone)]
struct ConvTranspose1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let kernel = 2 * stride;
        let g = tensor(file, &format!("{prefix}.weight_g"))?;
        let v = tensor(file, &format!("{prefix}.weight_v"))?;
        let weight = fold_weight_norm(&v, &g, in_channels, out_channels * kernel)?;
        Ok(Self {
            weight,
            bias: tensor(file, &format!("{prefix}.bias"))?,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding: stride.div_ceil(2),
        })
    }

    fn forward(
        &self,
        ops: &impl HifiGanBackendOps,
        input: &[f32],
        time: usize,
    ) -> Result<Vec<f32>> {
        ops.conv_transpose1d(
            input,
            self.in_channels,
            time,
            &self.weight,
            self.out_channels,
            self.kernel,
            Some(&self.bias),
            self.stride,
            self.padding,
        )
    }
}

#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    channels: usize,
}

impl Snake {
    fn load(file: &GgufFile, name: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            alpha: tensor(file, name)?,
            channels,
        })
    }

    fn forward(&self, compute: &Compute, input: &[f32], time: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0; input.len()];
        compute.snake_activation_f32(input, &self.alpha, self.channels, time, &mut output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    first_snake: Snake,
    first_conv: Conv1d,
    second_snake: Snake,
    second_conv: Conv1d,
}

impl ResidualUnit {
    fn load(file: &GgufFile, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            first_snake: Snake::load(file, &format!("{prefix}.block.0.alpha"), channels)?,
            first_conv: Conv1d::load(
                file,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                7,
                dilation,
                3 * dilation,
            )?,
            second_snake: Snake::load(file, &format!("{prefix}.block.2.alpha"), channels)?,
            second_conv: Conv1d::load(
                file,
                &format!("{prefix}.block.3"),
                channels,
                channels,
                1,
                1,
                0,
            )?,
        })
    }

    fn forward(
        &self,
        compute: &Compute,
        ops: &impl HifiGanBackendOps,
        input: Vec<f32>,
        time: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.first_snake.forward(compute, &input, time)?;
        let hidden = self.first_conv.forward(ops, &hidden, time)?;
        let hidden = self.second_snake.forward(compute, &hidden, time)?;
        let mut hidden = self.second_conv.forward(ops, &hidden, time)?;
        if hidden.len() != input.len() {
            return Err(VokraError::InvalidArgument(format!(
                "dac residual: branch length {} != skip length {}",
                hidden.len(),
                input.len()
            )));
        }
        for (destination, skip) in hidden.iter_mut().zip(input) {
            *destination += skip;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: Snake,
    upsample: ConvTranspose1d,
    residuals: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn load(
        file: &GgufFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            snake: Snake::load(file, &format!("{prefix}.0.alpha"), in_channels)?,
            upsample: ConvTranspose1d::load(
                file,
                &format!("{prefix}.1"),
                in_channels,
                out_channels,
                stride,
            )?,
            residuals: [
                ResidualUnit::load(file, &format!("{prefix}.2"), out_channels, 1)?,
                ResidualUnit::load(file, &format!("{prefix}.3"), out_channels, 3)?,
                ResidualUnit::load(file, &format!("{prefix}.4"), out_channels, 9)?,
            ],
        })
    }

    fn forward(
        &self,
        compute: &Compute,
        ops: &impl HifiGanBackendOps,
        input: Vec<f32>,
        time: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let hidden = self.snake.forward(compute, &input, time)?;
        let mut hidden = self.upsample.forward(ops, &hidden, time)?;
        if hidden.len() % self.upsample.out_channels != 0 {
            return Err(VokraError::InvalidArgument(
                "dac decoder: ConvTranspose1D output is not channel-aligned".to_owned(),
            ));
        }
        let output_time = hidden.len() / self.upsample.out_channels;
        for residual in &self.residuals {
            hidden = residual.forward(compute, ops, hidden, output_time)?;
        }
        Ok((hidden, output_time))
    }
}

/// Complete weight-normalized SEANet decoder.
#[derive(Debug, Clone)]
pub struct DacDecoder {
    pre: Conv1d,
    blocks: Vec<DecoderBlock>,
    post_snake: Snake,
    post: Conv1d,
}

impl DacDecoder {
    fn load(file: &GgufFile, variant: DacVariant) -> Result<Self> {
        let pre = Conv1d::load(file, "decoder.model.0", LATENT_DIM, DECODER_DIM, 7, 1, 3)?;
        let mut blocks = Vec::with_capacity(4);
        for (stage, &stride) in variant.decoder_rates().iter().enumerate() {
            let in_channels = DECODER_DIM >> stage;
            let out_channels = in_channels / 2;
            blocks.push(DecoderBlock::load(
                file,
                &format!("decoder.model.{}.block", stage + 1),
                in_channels,
                out_channels,
                stride,
            )?);
        }
        Ok(Self {
            pre,
            blocks,
            post_snake: Snake::load(file, "decoder.model.5.alpha", DECODER_DIM / 16)?,
            post: Conv1d::load(file, "decoder.model.6", DECODER_DIM / 16, 1, 7, 1, 3)?,
        })
    }

    fn forward_with_compute(&self, features: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if features.is_empty() || !features.len().is_multiple_of(LATENT_DIM) {
            return Err(VokraError::InvalidArgument(format!(
                "dac decoder: channel-major feature length {} must be a positive multiple of {LATENT_DIM}",
                features.len()
            )));
        }
        let ops = HifiGanComputeOps { compute };
        let mut time = features.len() / LATENT_DIM;
        let mut hidden = self.pre.forward(&ops, features, time)?;
        for block in &self.blocks {
            (hidden, time) = block.forward(compute, &ops, hidden, time)?;
        }
        hidden = self.post_snake.forward(compute, &hidden, time)?;
        let mut pcm = self.post.forward(&ops, &hidden, time)?;
        for sample in &mut pcm {
            *sample = sample.tanh();
        }
        Ok(pcm)
    }
}

/// Complete public DAC token-to-waveform model.
#[derive(Debug, Clone)]
pub struct Dac {
    variant: DacVariant,
    codec: DacCodecGguf,
    decoder: DacDecoder,
    backend: BackendKind,
}

impl Dac {
    /// Strictly binds one of the three released public DAC GGUF variants.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, ARCH)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "MIT")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        if file
            .get(chunks::KEY_PROVENANCE_SOURCE)
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
            .is_none()
        {
            return Err(VokraError::ModelLoad(format!(
                "dac: `{}` is missing or empty",
                chunks::KEY_PROVENANCE_SOURCE
            )));
        }
        let variant = variant_from_metadata(file)?;
        validate_manifest(file, variant)?;
        let codec = DacCodecGguf::from_gguf(file)?;
        let decoder = DacDecoder::load(file, variant)?;
        Ok(Self {
            variant,
            codec,
            decoder,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF file.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for RVQ, every convolution, and every Snake op.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    /// Returns the selected execution backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    /// Returns the exact released DAC variant bound from GGUF metadata.
    pub const fn variant(&self) -> DacVariant {
        self.variant
    }

    #[must_use]
    /// Returns the model waveform sample rate.
    pub const fn sample_rate(&self) -> u32 {
        self.variant.sample_rate()
    }

    #[must_use]
    /// Returns the encoder hop recorded by the released checkpoint.
    pub const fn hop_length(&self) -> usize {
        self.variant.hop_length()
    }

    #[must_use]
    /// Returns the required codebook columns per code frame.
    pub const fn n_codebooks(&self) -> usize {
        self.variant.n_codebooks()
    }

    /// Runs the complete factorized RVQ + SEANet decoder.
    ///
    /// `codes` is row-major `[frames, n_codebooks]`.  The 16/24 kHz upstream
    /// decoder contains an odd stride-5 ConvTranspose1D and therefore emits
    /// `frames * 320 - 8` samples; no hidden pad or trim is applied.
    pub fn decode_codes(&self, codes: &[u32]) -> Result<Vec<f32>> {
        let n_codebooks = self.n_codebooks();
        if codes.is_empty() || !codes.len().is_multiple_of(n_codebooks) {
            return Err(VokraError::InvalidArgument(format!(
                "dac: code length {} must be a positive multiple of n_codebooks {n_codebooks}",
                codes.len()
            )));
        }
        let frames = codes.len() / n_codebooks;
        let compute = Compute::for_backend(self.backend, DAC_HOT_OPS)?;
        let time_major = compute.dac_rvq_f32(
            codes,
            frames,
            &self.codec.tables,
            &self.codec.out_projs,
            &self.codec.attrs,
        )?;
        let mut channel_major = vec![0.0; time_major.len()];
        for frame in 0..frames {
            for channel in 0..LATENT_DIM {
                channel_major[channel * frames + frame] = time_major[frame * LATENT_DIM + channel];
            }
        }
        self.decoder.forward_with_compute(&channel_major, &compute)
    }

    /// Runs only the SEANet decoder on channel-major `[1024, frames]` input.
    pub fn decode_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, DAC_HOT_OPS)?;
        self.decoder.forward_with_compute(features, &compute)
    }

    /// Exact output extent of the released decoder for a positive frame count.
    pub fn output_samples(&self, frames: usize) -> Result<usize> {
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "dac: output extent requires at least one code frame".to_owned(),
            ));
        }
        self.variant
            .decoder_rates()
            .iter()
            .try_fold(frames, |length, &stride| {
                (length - 1)
                    .checked_mul(stride)
                    .and_then(|value| value.checked_add(2 * stride))
                    .and_then(|value| value.checked_sub(2 * stride.div_ceil(2)))
                    .ok_or_else(|| {
                        VokraError::InvalidArgument(
                            "dac: decoder output length overflow/underflow".to_owned(),
                        )
                    })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_manifests_have_exact_public_counts() {
        assert_eq!(expected_manifest(DacVariant::Khz44).len(), 328);
        assert_eq!(expected_manifest(DacVariant::Khz16).len(), 358);
        assert_eq!(expected_manifest(DacVariant::Khz24).len(), 558);
    }

    #[test]
    fn variant_rates_reproduce_metadata_hops() {
        for variant in [DacVariant::Khz16, DacVariant::Khz24, DacVariant::Khz44] {
            assert_eq!(
                variant.encoder_rates().iter().product::<usize>(),
                variant.hop_length()
            );
            assert_eq!(
                variant.decoder_rates().iter().product::<usize>(),
                variant.hop_length()
            );
        }
    }

    #[test]
    fn odd_stride_extent_is_not_silently_padded() {
        let extent = |variant: DacVariant, frames: usize| {
            variant
                .decoder_rates()
                .iter()
                .fold(frames, |length, &stride| {
                    (length - 1) * stride + 2 * stride - 2 * stride.div_ceil(2)
                })
        };
        assert_eq!(extent(DacVariant::Khz44, 3), 3 * 512);
        assert_eq!(extent(DacVariant::Khz16, 3), 3 * 320 - 8);
        assert_eq!(extent(DacVariant::Khz24, 3), 3 * 320 - 8);
    }

    #[test]
    fn weight_norm_fold_matches_hand_calculation() {
        let got = fold_weight_norm(&[3.0, 4.0, 0.0, 2.0], &[2.0, 3.0], 2, 2).unwrap();
        assert_eq!(got, vec![1.2, 1.6, 0.0, 3.0]);
    }
}
