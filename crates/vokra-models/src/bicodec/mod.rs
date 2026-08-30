//! Native SparkAudio BiCodec token detokenizer (CPU / Metal).
//!
//! This is the bounded decode-only half of the official Spark-TTS BiCodec.
//! It binds the complete authenticated 840-tensor GGUF manifest, while only
//! loading the 319 tensors reachable from semantic/global token detokenize.
//! Encoding is intentionally unsupported: the upstream encoder depends on an
//! external Wav2Vec2 feature extractor and is not part of this wave.
//!
//! Source semantics are transcribed from Spark-TTS at revision
//! `2f1ea9082400547242641f5271b6f941c9f439d1`.  The weight artifact remains
//! CC-BY-NC-SA-4.0 research-only; this runtime never publishes weights.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};

/// GGUF architecture tag for the SparkAudio BiCodec.
pub const ARCH: &str = "bicodec";
/// GGUF model name for the authenticated Spark-TTS BiCodec release.
pub const NAME: &str = "spark-tts-bicodec";
/// PCM sample rate emitted by the wave decoder, in hertz.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of PCM samples emitted per semantic token.
pub const FRAME_HOP: usize = 320;
/// Number of semantic codebook entries.
pub const SEMANTIC_VOCAB: u32 = 8_192;
/// Number of packed global FSQ token values.
pub const GLOBAL_VOCAB: u32 = 4_096;
/// Fixed number of global speaker tokens.
pub const GLOBAL_TOKENS: usize = 32;
/// Semantic codebook embedding width.
pub const SEMANTIC_DIM: usize = 8;
/// Decoder feature and waveform-generator model width.
pub const MODEL_DIM: usize = 1_024;
/// Speaker FSQ latent width before flattening.
pub const SPEAKER_LATENT_DIM: usize = 128;
/// Number of scalar FSQ digits in one global token.
pub const SPEAKER_CODE_DIM: usize = 6;
/// Number of levels in each scalar FSQ digit.
pub const SPEAKER_LEVELS: usize = 4;

const LABEL: &str = "bicodec";
const MANIFEST_SHA256: [u8; 32] = [
    0xf9, 0x1e, 0xc1, 0x99, 0x5d, 0xdc, 0xb7, 0x51, 0x15, 0x13, 0x0c, 0x61, 0x4d, 0xd7, 0x97, 0xf7,
    0xd1, 0x2c, 0x3e, 0x97, 0xa1, 0x50, 0x5e, 0xcb, 0x7a, 0x61, 0xc9, 0x5d, 0x76, 0x2e, 0xc8, 0x6c,
];

/// Compute operations required by the native BiCodec decode route.
pub const BICODEC_DECODE_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::Conv1dDilation,
    HotOp::ConvTranspose1d,
    HotOp::SnakeActivation,
    HotOp::Tanh,
];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: 840,
    manifest_sha256: MANIFEST_SHA256,
};

const EPS: f32 = 1.0e-6;

/// Strict native BiCodec decoder.
#[derive(Debug)]
pub struct Bicodec {
    backend: BackendKind,
    weight_license: LicenseClass,
    semantic_codebook: Vec<f32>,
    semantic_project: WnConv1d,
    speaker_project_out: Linear,
    speaker_project: Linear,
    prenet: Decoder,
    wave: WaveGenerator,
}

impl Bicodec {
    /// Bind an authenticated BiCodec GGUF and use the CPU backend.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_backend(file, BackendKind::Cpu)
    }

    /// Bind an authenticated BiCodec GGUF using one preflighted backend.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata, tensor manifest/dtypes, provenance, or
    /// backend hot-op coverage does not match the fixed release contract.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, BICODEC_DECODE_HOT_OPS)?;
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        if let Some(tensor) = file
            .tensors()
            .iter()
            .find(|tensor| tensor.dtype != GgmlType::F32)
        {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: tensor `{}` has dtype {:?}; authenticated BiCodec release requires exact F32 tensors",
                tensor.name, tensor.dtype
            )));
        }
        require_string(file, "vokra.model.category", "codec")?;
        require_string(
            file,
            "vokra.provenance.upstream_hf",
            "SparkAudio/Spark-TTS-0.5B",
        )?;
        require_string(
            file,
            "vokra.bicodec.upstream_revision",
            "642071559bfc6346c2359d19dcb6be3f9dd8a05d",
        )?;
        require_string(
            file,
            "vokra.bicodec.checkpoint_sha256",
            "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec",
        )?;
        require_string(
            file,
            "vokra.bicodec.config_sha256",
            "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be",
        )?;
        require_string(
            file,
            "vokra.bicodec.source_repository",
            "https://github.com/SparkAudio/Spark-TTS",
        )?;
        require_string(
            file,
            "vokra.bicodec.source_revision",
            "2f1ea9082400547242641f5271b6f941c9f439d1",
        )?;
        require_string(
            file,
            "vokra.bicodec.inspection_status",
            "NATIVE_DECODE_ONLY",
        )?;
        require_bool(file, "vokra.bicodec.input_authenticated", true)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "cc-by-nc-sa-4.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::NonCommercialShareAlike.as_str(),
        )?;
        require_u32(file, "vokra.bicodec.sample_rate", SAMPLE_RATE)?;
        require_u32(file, "vokra.bicodec.frame_hop", FRAME_HOP as u32)?;
        require_u32(file, "vokra.bicodec.semantic_vocab", SEMANTIC_VOCAB)?;
        require_u32(file, "vokra.bicodec.global_vocab", GLOBAL_VOCAB)?;
        require_u32(file, "vokra.bicodec.global_tokens", GLOBAL_TOKENS as u32)?;

        let semantic_codebook = load_tensor(
            file,
            LABEL,
            "quantizer.codebook.weight",
            &[SEMANTIC_VOCAB as usize, SEMANTIC_DIM],
        )?;
        let semantic_project = WnConv1d::load(
            file,
            "quantizer.out_project",
            SEMANTIC_DIM,
            MODEL_DIM,
            1,
            1,
            0,
            1,
        )?;
        let speaker_project_out = Linear::load(
            file,
            "speaker_encoder.quantizer.project_out",
            SPEAKER_CODE_DIM,
            SPEAKER_LATENT_DIM,
        )?;
        let speaker_project = Linear::load(
            file,
            "speaker_encoder.project",
            SPEAKER_LATENT_DIM * GLOBAL_TOKENS,
            MODEL_DIM,
        )?;
        let prenet = Decoder::load(file, "prenet", false)?;
        let wave = WaveGenerator::load(file)?;

        Ok(Self {
            backend,
            weight_license: checkpoint.weight_license(),
            semantic_codebook,
            semantic_project,
            speaker_project_out,
            speaker_project,
            prenet,
            wave,
        })
    }

    /// Open and bind an authenticated BiCodec GGUF from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Return the backend selected when this decoder was bound.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Return the authenticated non-commercial/share-alike weight class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Return the fixed output sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Decode semantic `[T]` and global `[32]` tokens to mono 16-kHz PCM.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token shapes/ranges, unsupported backend
    /// coverage, malformed intermediate extents, or non-finite output.
    pub fn decode(&self, semantic_tokens: &[u32], global_tokens: &[u32]) -> Result<Vec<f32>> {
        validate_tokens(semantic_tokens, global_tokens)?;
        let compute = Compute::for_backend(self.backend, BICODEC_DECODE_HOT_OPS)?;
        let semantic = self.semantic_decode(semantic_tokens, &compute)?;
        let speaker = self.speaker_decode(global_tokens, &compute)?;
        let feature_len = MODEL_DIM
            .checked_mul(semantic_tokens.len())
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: latent extent overflow"))
            })?;
        if semantic.len() != feature_len || speaker.len() != MODEL_DIM {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: detokenizer latent shape mismatch"
            )));
        }
        let mut features = self.prenet.forward(&semantic, &speaker, &compute)?;
        let time = semantic_tokens.len();
        if features.len() != feature_len {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: prenet output shape mismatch"
            )));
        }
        for channel in 0..MODEL_DIM {
            for position in 0..time {
                features[channel * time + position] += speaker[channel];
            }
        }
        let waveform = self.wave.forward(&features, time, &compute)?;
        let expected = time.checked_mul(FRAME_HOP).ok_or_else(|| {
            VokraError::InvalidArgument("bicodec: output length overflow".to_owned())
        })?;
        if waveform.len() != expected {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: waveform length {} != expected {expected}",
                waveform.len()
            )));
        }
        reject_non_finite("decoded waveform", &waveform)?;
        Ok(waveform)
    }

    /// Decode the two token streams; this is an alias for [`Self::decode`].
    pub fn detokenize(&self, semantic_tokens: &[u32], global_tokens: &[u32]) -> Result<Vec<f32>> {
        self.decode(semantic_tokens, global_tokens)
    }

    /// Reject encoding because the bounded implementation requires external
    /// Wav2Vec2 feature extraction.
    ///
    /// # Errors
    ///
    /// Always returns [`VokraError::UnsupportedOp`].
    pub fn encode(&self, _pcm: &[f32], _sample_rate: u32) -> Result<(Vec<u32>, Vec<u32>)> {
        Err(VokraError::UnsupportedOp("bicodec: encode is unavailable in the bounded wave; Wav2Vec2 feature extraction is external".to_owned()))
    }

    fn semantic_decode(&self, tokens: &[u32], compute: &Compute) -> Result<Vec<f32>> {
        let embedded_len = SEMANTIC_DIM.checked_mul(tokens.len()).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: semantic embedding extent overflow"))
        })?;
        let mut embedded = vec![0.0f32; embedded_len];
        for (position, &token) in tokens.iter().enumerate() {
            let row = token as usize * SEMANTIC_DIM;
            for dim in 0..SEMANTIC_DIM {
                embedded[dim * tokens.len() + position] = self.semantic_codebook[row + dim];
            }
        }
        self.semantic_project
            .forward(&embedded, tokens.len(), compute)
    }

    fn speaker_decode(&self, tokens: &[u32], compute: &Compute) -> Result<Vec<f32>> {
        let mut digits = vec![0.0f32; SPEAKER_CODE_DIM * GLOBAL_TOKENS];
        let basis = [1usize, 4, 16, 64, 256, 1024];
        for (position, &token) in tokens.iter().enumerate() {
            let value = token as usize;
            for dim in 0..SPEAKER_CODE_DIM {
                let digit = (value / basis[dim]) % SPEAKER_LEVELS;
                digits[dim * GLOBAL_TOKENS + position] = (digit as f32 - 2.0) / 2.0;
            }
        }
        let latent =
            self.speaker_project_out
                .forward_channel_major(&digits, GLOBAL_TOKENS, compute)?;
        let flattened = latent;
        self.speaker_project.forward(&flattened, 1, compute)
    }
}

fn validate_tokens(semantic: &[u32], global: &[u32]) -> Result<()> {
    if semantic.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: semantic token count must be non-zero (got {})",
            semantic.len()
        )));
    }
    if global.len() != GLOBAL_TOKENS {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: global token count {} != {GLOBAL_TOKENS}",
            global.len()
        )));
    }
    if let Some((position, value)) = semantic
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value >= SEMANTIC_VOCAB)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: semantic_tokens[{position}]={value} outside 0..{SEMANTIC_VOCAB}"
        )));
    }
    if let Some((position, value)) = global
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value >= GLOBAL_VOCAB)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{ARCH}: global_tokens[{position}]={value} outside 0..{GLOBAL_VOCAB}"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct Linear {
    input: usize,
    output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn load(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        let source = load_tensor(file, LABEL, &format!("{prefix}.weight"), &[output, input])?;
        let mut weight = vec![0.0f32; input * output];
        for out in 0..output {
            for inner in 0..input {
                weight[inner * output + out] = source[out * input + inner];
            }
        }
        Ok(Self {
            input,
            output,
            weight,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize, compute: &Compute) -> Result<Vec<f32>> {
        let expected = rows.checked_mul(self.input).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: linear input extent overflow"))
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: linear input shape mismatch"
            )));
        }
        let output_len = rows.checked_mul(self.output).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: linear output extent overflow"))
        })?;
        let mut output = vec![0.0f32; output_len];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight,
            Some(&self.bias),
            &mut output,
        )?;
        Ok(output)
    }

    fn forward_channel_major(
        &self,
        input: &[f32],
        time: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let expected = self.input.checked_mul(time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: linear channel-major extent overflow"))
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: linear channel-major shape mismatch"
            )));
        }
        let mut rows = vec![0.0f32; expected];
        for channel in 0..self.input {
            for t in 0..time {
                rows[t * self.input + channel] = input[channel * time + t];
            }
        }
        let rows_out = self.forward(&rows, time, compute)?;
        let output_len = self.output.checked_mul(time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: linear channel-major output overflow"))
        })?;
        let mut output = vec![0.0f32; output_len];
        for t in 0..time {
            for channel in 0..self.output {
                output[channel * time + t] = rows_out[t * self.output + channel];
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct WnConv1d {
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl WnConv1d {
    #[allow(clippy::too_many_arguments)] // Convolution geometry is an intrinsic authenticated parameter set.
    fn load(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let g = load_tensor(file, LABEL, &format!("{prefix}.weight_g"), &[output, 1, 1])?;
        let v = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.weight_v"),
            &[output, input, kernel],
        )?;
        let weight = fold_weight_norm(&g, &v, output, input * kernel, prefix)?;
        let bias = load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?;
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            padding,
            dilation,
            weight,
            bias,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        let effective = self
            .kernel
            .checked_sub(1)
            .and_then(|value| value.checked_mul(self.dilation))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: Conv1d effective kernel overflow"))
            })?;
        let padded = time
            .checked_add(self.padding.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: Conv1d padding extent overflow"))
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: Conv1d padded extent overflow"))
            })?;
        if self.stride == 0 || effective > padded {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: invalid Conv1d extent"
            )));
        }
        let out_time = (padded - effective) / self.stride + 1;
        let output_len = self.output.checked_mul(out_time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: Conv1d output extent overflow"))
        })?;
        let mut output = vec![0.0f32; output_len];
        compute.conv1d_f32_dilated(
            input,
            self.input,
            time,
            &self.weight,
            self.output,
            self.kernel,
            Some(&self.bias),
            self.stride,
            self.dilation,
            self.padding,
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct RawConv1d {
    input: usize,
    output: usize,
    kernel: usize,
    padding: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl RawConv1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self> {
        if groups == 0 || input % groups != 0 || output % groups != 0 {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: invalid grouped Conv1d shape for {prefix}: input={input}, output={output}, groups={groups}"
            )));
        }
        Ok(Self {
            input,
            output,
            kernel,
            padding,
            weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.weight"),
                &[output, input / groups, kernel],
            )?,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }
    fn forward(
        &self,
        input: &[f32],
        time: usize,
        groups: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let padded = time
            .checked_add(self.padding.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: Conv1d padding extent overflow"))
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: Conv1d padded extent overflow"))
            })?;
        if self.kernel == 0 || self.kernel > padded {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: invalid Conv1d extent"
            )));
        }
        let out_time = padded - self.kernel + 1;
        let output_len = self.output.checked_mul(out_time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: Conv1d output extent overflow"))
        })?;
        let mut output = vec![0.0f32; output_len];
        if groups == 1 {
            compute.conv1d_f32(
                input,
                self.input,
                time,
                &self.weight,
                self.output,
                self.kernel,
                Some(&self.bias),
                1,
                self.padding,
                &mut output,
            )?;
        } else {
            compute.grouped_conv1d_f32(
                input,
                self.input,
                time,
                &self.weight,
                self.output,
                self.kernel,
                Some(&self.bias),
                1,
                self.padding,
                groups,
                &mut output,
            )?;
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}
impl LayerNorm {
    fn load(file: &GgufFile, prefix: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: load_tensor(file, LABEL, &format!("{prefix}.weight"), &[dim])?,
            beta: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[dim])?,
        })
    }
    fn identity(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
        }
    }
    fn forward(
        &self,
        input: &[f32],
        dim: usize,
        time: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let rows = transpose_to_rows(input, dim, time);
        let mut output = vec![0.0f32; rows.len()];
        compute.layer_norm_f32(&rows, &mut output, time, dim, &self.gamma, &self.beta, EPS)?;
        Ok(transpose_from_rows(&output, dim, time))
    }
}

#[derive(Debug)]
struct AdaNorm {
    scale: Linear,
    shift: Linear,
    identity: LayerNorm,
}
impl AdaNorm {
    fn load(file: &GgufFile, prefix: &str, condition: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            scale: Linear::load(file, &format!("{prefix}.scale"), condition, dim)?,
            shift: Linear::load(file, &format!("{prefix}.shift"), condition, dim)?,
            identity: LayerNorm::identity(dim),
        })
    }
    fn forward(
        &self,
        input: &[f32],
        dim: usize,
        time: usize,
        condition: &[f32],
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let normalized = self.identity.forward(input, dim, time, compute)?;
        let scale = self.scale.forward(condition, 1, compute)?;
        let shift = self.shift.forward(condition, 1, compute)?;
        let mut output = normalized;
        for channel in 0..dim {
            for t in 0..time {
                let index = channel * time + t;
                output[index] = output[index] * scale[channel] + shift[channel];
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
enum Norm {
    Plain(LayerNorm),
    Adaptive(AdaNorm),
}
impl Norm {
    fn forward(
        &self,
        input: &[f32],
        dim: usize,
        time: usize,
        condition: Option<&[f32]>,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        match (self, condition) {
            (Self::Plain(norm), _) => norm.forward(input, dim, time, compute),
            (Self::Adaptive(norm), Some(cond)) => norm.forward(input, dim, time, cond, compute),
            (Self::Adaptive(_), None) => Err(VokraError::InvalidArgument(format!(
                "{ARCH}: conditioned Vocos block requires d-vector"
            ))),
        }
    }
}

#[derive(Debug)]
struct VocosBlock {
    dim: usize,
    depthwise: RawConv1d,
    norm: Norm,
    pw1: Linear,
    pw2: Linear,
    gamma: Vec<f32>,
}
impl VocosBlock {
    fn load(
        file: &GgufFile,
        prefix: &str,
        dim: usize,
        intermediate: usize,
        condition: Option<usize>,
    ) -> Result<Self> {
        let norm = match condition {
            Some(c) => Norm::Adaptive(AdaNorm::load(file, &format!("{prefix}.norm"), c, dim)?),
            None => Norm::Plain(LayerNorm::load(file, &format!("{prefix}.norm"), dim)?),
        };
        Ok(Self {
            dim,
            depthwise: RawConv1d::load(file, &format!("{prefix}.dwconv"), dim, dim, 7, 3, dim)?,
            norm,
            pw1: Linear::load(file, &format!("{prefix}.pwconv1"), dim, intermediate)?,
            pw2: Linear::load(file, &format!("{prefix}.pwconv2"), intermediate, dim)?,
            gamma: load_tensor(file, LABEL, &format!("{prefix}.gamma"), &[dim])?,
        })
    }
    fn forward(
        &self,
        input: &[f32],
        time: usize,
        condition: Option<&[f32]>,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let mut x = self.depthwise.forward(input, time, self.dim, compute)?;
        x = self.norm.forward(&x, self.dim, time, condition, compute)?;
        let rows = transpose_to_rows(&x, self.dim, time);
        let hidden = self.pw1.forward(&rows, time, compute)?;
        let mut activated = vec![0.0f32; hidden.len()];
        compute.gelu_f32(&hidden, &mut activated)?;
        let projected = self.pw2.forward(&activated, time, compute)?;
        let mut output = input.to_vec();
        for channel in 0..self.dim {
            for t in 0..time {
                output[channel * time + t] +=
                    self.gamma[channel] * projected[t * self.dim + channel];
            }
        }
        Ok(output)
    }
}

#[derive(Debug)]
struct Vocos {
    dim: usize,
    embed: RawConv1d,
    norm: Norm,
    blocks: Vec<VocosBlock>,
    final_norm: LayerNorm,
}
impl Vocos {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        dim: usize,
        intermediate: usize,
        layers: usize,
        condition: Option<usize>,
    ) -> Result<Self> {
        let norm = match condition {
            Some(c) => Norm::Adaptive(AdaNorm::load(file, &format!("{prefix}.norm"), c, dim)?),
            None => Norm::Plain(LayerNorm::load(file, &format!("{prefix}.norm"), dim)?),
        };
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            blocks.push(VocosBlock::load(
                file,
                &format!("{prefix}.convnext.{index}"),
                dim,
                intermediate,
                condition,
            )?);
        }
        Ok(Self {
            dim,
            embed: RawConv1d::load(file, &format!("{prefix}.embed"), input, dim, 7, 3, 1)?,
            norm,
            blocks,
            final_norm: LayerNorm::load(file, &format!("{prefix}.final_layer_norm"), dim)?,
        })
    }
    fn forward(
        &self,
        input: &[f32],
        time: usize,
        condition: Option<&[f32]>,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let mut x = self.embed.forward(input, time, 1, compute)?;
        x = self.norm.forward(&x, self.dim, time, condition, compute)?;
        for block in &self.blocks {
            x = block.forward(&x, time, condition, compute)?;
        }
        let normalized = self.final_norm.forward(&x, self.dim, time, compute)?;
        Ok(transpose_to_rows(&normalized, self.dim, time))
    }
}

#[derive(Debug)]
struct Decoder {
    linear_pre: Linear,
    sampling: Vec<Vocos>,
    backbone: Vocos,
    linear: Linear,
}
impl Decoder {
    fn load(file: &GgufFile, prefix: &str, _postnet: bool) -> Result<Self> {
        let mut sampling = Vec::with_capacity(2);
        for index in 0..2 {
            sampling.push(Vocos::load(
                file,
                &format!("{prefix}.downsample.{index}.1"),
                384,
                384,
                2048,
                2,
                None,
            )?);
        }
        Ok(Self {
            linear_pre: Linear::load(file, &format!("{prefix}.linear_pre"), 1024, 384)?,
            sampling,
            backbone: Vocos::load(
                file,
                &format!("{prefix}.vocos_backbone"),
                384,
                384,
                2048,
                12,
                Some(1024),
            )?,
            linear: Linear::load(file, &format!("{prefix}.linear"), 384, 1024)?,
        })
    }
    fn forward(&self, input: &[f32], condition: &[f32], compute: &Compute) -> Result<Vec<f32>> {
        if input.len() % MODEL_DIM != 0 || condition.len() != MODEL_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: decoder input/condition shape mismatch"
            )));
        }
        let time = input.len() / 1024;
        let rows = transpose_to_rows(input, 1024, time);
        let projected = self.linear_pre.forward(&rows, time, compute)?;
        let mut x = transpose_from_rows(&projected, 384, time);
        for sampling in &self.sampling {
            let mut scaled = x.clone();
            for value in &mut scaled {
                *value *= 3.0;
            }
            let sampled_rows = sampling.forward(&scaled, time, None, compute)?;
            x = transpose_from_rows(&sampled_rows, 384, time);
        }
        let backbone_rows = self.backbone.forward(&x, time, Some(condition), compute)?;
        let output = self.linear.forward(&backbone_rows, time, compute)?;
        Ok(transpose_from_rows(&output, 1024, time))
    }
}

#[derive(Debug)]
struct ResidualUnit {
    channels: usize,
    first: WnConv1d,
    second: WnConv1d,
    alpha1: Vec<f32>,
    alpha2: Vec<f32>,
}
impl ResidualUnit {
    fn load(file: &GgufFile, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            channels,
            alpha1: load_alpha(file, &format!("{prefix}.block.0.alpha"), channels)?,
            first: WnConv1d::load(
                file,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                7,
                1,
                3 * dilation,
                dilation,
            )?,
            alpha2: load_alpha(file, &format!("{prefix}.block.2.alpha"), channels)?,
            second: WnConv1d::load(
                file,
                &format!("{prefix}.block.3"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
        })
    }
    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut x = vec![0.0; input.len()];
        compute.snake_activation_f32(input, &self.alpha1, self.channels, time, &mut x)?;
        x = self.first.forward(&x, time, compute)?;
        let mut y = vec![0.0; x.len()];
        compute.snake_activation_f32(&x, &self.alpha2, self.channels, time, &mut y)?;
        y = self.second.forward(&y, time, compute)?;
        for (value, residual) in y.iter_mut().zip(input) {
            *value += residual;
        }
        Ok(y)
    }
}

#[derive(Debug)]
struct WaveStage {
    channels: usize,
    alpha: Vec<f32>,
    upsample: WnConvTranspose1d,
    residuals: Vec<ResidualUnit>,
}
impl WaveStage {
    fn load(
        file: &GgufFile,
        index: usize,
        channels: usize,
        next: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let prefix = format!("decoder.model.{index}");
        let residuals = [2usize, 3, 4]
            .into_iter()
            .zip([1usize, 3, 9])
            .map(|(block, dilation)| {
                ResidualUnit::load(file, &format!("{prefix}.block.{block}"), next, dilation)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            channels,
            alpha: load_alpha(file, &format!("{prefix}.block.0.alpha"), channels)?,
            upsample: WnConvTranspose1d::load(
                file,
                &format!("{prefix}.block.1"),
                channels,
                next,
                kernel,
                stride,
                (kernel - stride) / 2,
            )?,
            residuals,
        })
    }
    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<(Vec<f32>, usize)> {
        let mut activated = vec![0.0; input.len()];
        compute.snake_activation_f32(input, &self.alpha, self.channels, time, &mut activated)?;
        let output = self.upsample.forward(&activated, time, compute)?;
        let next_time = output.len() / self.upsample.output;
        let mut output = output;
        for residual in &self.residuals {
            output = residual.forward(&output, next_time, compute)?;
        }
        Ok((output, next_time))
    }
}

#[derive(Debug)]
struct WnConvTranspose1d {
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl WnConvTranspose1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let g = load_tensor(file, LABEL, &format!("{prefix}.weight_g"), &[input, 1, 1])?;
        let v = load_tensor(
            file,
            LABEL,
            &format!("{prefix}.weight_v"),
            &[input, output, kernel],
        )?;
        Ok(Self {
            input,
            output,
            kernel,
            stride,
            padding,
            weight: fold_weight_norm(&g, &v, input, output * kernel, prefix)?,
            bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        })
    }
    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        if time == 0 || self.stride == 0 || self.kernel == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: invalid ConvTranspose1d extent"
            )));
        }
        let full = time
            .checked_sub(1)
            .and_then(|value| value.checked_mul(self.stride))
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!("{ARCH}: ConvTranspose1d extent overflow"))
            })?;
        let trim = self.padding.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: ConvTranspose1d padding extent overflow"))
        })?;
        if trim >= full {
            return Err(VokraError::InvalidArgument(format!(
                "{ARCH}: invalid ConvTranspose1d padding"
            )));
        }
        let out_time = full - trim;
        let output_len = self.output.checked_mul(out_time).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{ARCH}: ConvTranspose1d output extent overflow"))
        })?;
        let mut output = vec![0.0; output_len];
        compute.conv_transpose1d_f32(
            input,
            self.input,
            time,
            &self.weight,
            self.output,
            self.kernel,
            Some(&self.bias),
            self.stride,
            self.padding,
            0,
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct WaveGenerator {
    initial: WnConv1d,
    stages: Vec<WaveStage>,
    final_alpha: Vec<f32>,
    final_conv: WnConv1d,
}
impl WaveGenerator {
    fn load(file: &GgufFile) -> Result<Self> {
        let initial = WnConv1d::load(file, "decoder.model.0", 1024, 1536, 7, 1, 3, 1)?;
        let stages = [
            (1usize, 1536usize, 768usize, 16usize, 8usize),
            (2, 768, 384, 11, 5),
            (3, 384, 192, 8, 4),
            (4, 192, 96, 4, 2),
        ]
        .into_iter()
        .map(|(index, input, output, kernel, stride)| {
            WaveStage::load(file, index, input, output, kernel, stride)
        })
        .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            initial,
            stages,
            final_alpha: load_alpha(file, "decoder.model.5.alpha", 96)?,
            final_conv: WnConv1d::load(file, "decoder.model.6", 96, 1, 7, 1, 3, 1)?,
        })
    }
    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        let mut x = self.initial.forward(input, time, compute)?;
        let mut current = time;
        for stage in &self.stages {
            (x, current) = stage.forward(&x, current, compute)?;
        }
        let mut activated = vec![0.0; x.len()];
        compute.snake_activation_f32(&x, &self.final_alpha, 96, current, &mut activated)?;
        let output = self.final_conv.forward(&activated, current, compute)?;
        let pre_tanh = output;
        let mut output = vec![0.0; pre_tanh.len()];
        compute.tanh_f32(&pre_tanh, &mut output)?;
        Ok(output)
    }
}

fn load_alpha(file: &GgufFile, name: &str, channels: usize) -> Result<Vec<f32>> {
    let values = load_tensor(file, LABEL, name, &[1, channels, 1])?;
    Ok((0..channels).map(|channel| values[channel]).collect())
}

fn fold_weight_norm(
    g: &[f32],
    v: &[f32],
    primary: usize,
    plane: usize,
    name: &str,
) -> Result<Vec<f32>> {
    if g.len() != primary || v.len() != primary * plane {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: weight norm {name} shape mismatch"
        )));
    }
    let mut output = vec![0.0; v.len()];
    for row in 0..primary {
        let source = &v[row * plane..(row + 1) * plane];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "{ARCH}: invalid weight norm {name} row {row}"
            )));
        }
        for (dst, value) in output[row * plane..(row + 1) * plane]
            .iter_mut()
            .zip(source)
        {
            *dst = *value * g[row] / norm;
        }
    }
    reject_non_finite(name, &output)?;
    Ok(output)
}

fn transpose_to_rows(input: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut rows = vec![0.0; input.len()];
    for channel in 0..channels {
        for t in 0..time {
            rows[t * channels + channel] = input[channel * time + t];
        }
    }
    rows
}
fn transpose_from_rows(rows: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows.len()];
    for t in 0..time {
        for channel in 0..channels {
            output[channel * time + t] = rows[t * channels + channel];
        }
    }
    output
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: {label} contains non-finite value {value} at {index}"
        )));
    }
    Ok(())
}
fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata {key}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}
fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}
fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(u64::from(expected)) {
        return Err(VokraError::ModelLoad(format!(
            "{ARCH}: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strict_checkpoint::sha256_bytes;

    #[test]
    fn authenticated_manifest_digest_matches_converter_identity() {
        assert_eq!(
            hex_digest(&MANIFEST_SHA256),
            "f91ec1995ddcb75115130c614dd797f7d12c3e97a1505ecb7a61c95d762ec86c"
        );
    }

    #[test]
    fn token_bounds_are_fail_closed() {
        assert!(validate_tokens(&[], &[0; GLOBAL_TOKENS]).is_err());
        assert!(validate_tokens(&[SEMANTIC_VOCAB], &[0; GLOBAL_TOKENS]).is_err());
        assert!(validate_tokens(&[0], &[GLOBAL_VOCAB; GLOBAL_TOKENS]).is_err());
        assert!(validate_tokens(&[0], &[0; GLOBAL_TOKENS - 1]).is_err());
    }
    #[test]
    fn fsq_basis_and_normalization_are_fixed() {
        let basis = [1usize, 4, 16, 64, 256, 1024];
        assert_eq!(basis[5] * SPEAKER_LEVELS, GLOBAL_VOCAB as usize);
        assert_eq!(basis[0], 1);
        assert_eq!(basis[1], SPEAKER_LEVELS);
        assert_eq!((4095 / basis[5]) % SPEAKER_LEVELS, 3);
        assert_eq!((4095 / basis[0]) % SPEAKER_LEVELS, 3);
        assert_eq!((0.0f32 - 2.0) / 2.0, -1.0);
        assert_eq!((3.0f32 - 2.0) / 2.0, 0.5);
    }

    #[test]
    fn weight_norm_folds_rows_over_all_non_output_axes() {
        let folded = fold_weight_norm(&[2.0, -1.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "test")
            .expect("non-zero synthetic rows are valid");
        // Row 0 is 2·[3,4]/sqrt(3²+4²) = [6/5,8/5]; row 1 is
        // -1·[5,12]/sqrt(5²+12²) = [-5/13,-12/13].
        assert_eq!(
            folded,
            vec![6.0 / 5.0, 8.0 / 5.0, -5.0 / 13.0, -12.0 / 13.0]
        );
    }

    #[test]
    fn global_flatten_is_latent_major_then_token_major() {
        let mut latent = vec![0.0f32; SPEAKER_LATENT_DIM * GLOBAL_TOKENS];
        for channel in 0..SPEAKER_LATENT_DIM {
            for token in 0..GLOBAL_TOKENS {
                latent[channel * GLOBAL_TOKENS + token] = (channel * GLOBAL_TOKENS + token) as f32;
            }
        }
        let flattened = latent.clone();
        assert_eq!(flattened[0], 0.0);
        assert_eq!(flattened[GLOBAL_TOKENS - 1], 31.0);
        assert_eq!(flattened[GLOBAL_TOKENS], 32.0);
        assert_eq!(flattened.len(), SPEAKER_LATENT_DIM * GLOBAL_TOKENS);
    }

    #[test]
    fn sampling_block_scale_is_exact_three() {
        let input = [1.0f32, -2.0, 4.0];
        let scaled: Vec<f32> = input.iter().map(|value| value * 3.0).collect();
        assert_eq!(scaled, vec![3.0, -6.0, 12.0]);
    }

    fn synthetic_reference_manifest(shape: &str) -> String {
        format!(
            "{{\"semantic_latent\": {{\"path\": \"semantic_latent.f32\", \"shape\": {shape}, \"dtype\": \"F32\", \"sha256\": \"deadbeef\"}}}}"
        )
    }

    fn assert_reference_record_rejected(manifest: &str) {
        assert!(
            std::panic::catch_unwind(|| reference_record(
                manifest,
                "semantic_latent",
                &[1, MODEL_DIM, 4]
            ))
            .is_err()
        );
    }

    #[test]
    fn reference_record_accepts_pretty_multiline_shape() {
        let manifest = r#"{
            "semantic_latent": {
                "path": "semantic_latent.f32",
                "shape": [
                    1,
                    1024,
                    4
                ],
                "dtype": "F32",
                "sha256": "deadbeef"
            }
        }"#;
        assert_eq!(
            reference_record(manifest, "semantic_latent", &[1, MODEL_DIM, 4]),
            ("semantic_latent.f32", "deadbeef")
        );
    }

    #[test]
    fn reference_record_rejects_malformed_duplicate_and_extra_shapes() {
        for shape in ["[1, 1024, -4]", "[1, 1024, 1.0]"] {
            assert_reference_record_rejected(&synthetic_reference_manifest(shape));
        }
        let duplicate_shape = r#"{
            "semantic_latent": {
                "path": "semantic_latent.f32",
                "shape": [1, 1024, 4],
                "shape": [1, 1024, 4],
                "dtype": "F32",
                "sha256": "deadbeef"
            }
        }"#;
        assert_reference_record_rejected(duplicate_shape);
        assert_reference_record_rejected(&synthetic_reference_manifest("[1, 1024, 4, 0]"));
    }

    #[test]
    fn measured_parity_bounds_cover_pass_and_fail_cases() {
        let semantic = parity_bounds("semantic_latent");
        assert!(parity_passes(
            "semantic_latent",
            &[semantic.rmse as f32 * 0.5],
            &[0.0]
        ));
        let mut max_spike_actual = vec![0.0f32; 64];
        max_spike_actual[0] = semantic.max_abs as f32 * 1.1;
        let max_spike_reference = vec![0.0f32; 64];
        let (max_spike, max_spike_rmse) = parity_metrics(&max_spike_actual, &max_spike_reference);
        assert!(max_spike > semantic.max_abs);
        assert!(max_spike_rmse < semantic.rmse);
        assert!(!parity_passes(
            "semantic_latent",
            &max_spike_actual,
            &max_spike_reference
        ));

        // Keep each element below the max bound while exceeding the RMSE
        // bound, so the systematic-drift gate is exercised independently.
        let prenet = parity_bounds("prenet_output");
        let rmse_actual = [prenet.rmse as f32 * 1.5, prenet.rmse as f32 * 1.5];
        let rmse_reference = [0.0, 0.0];
        let (rmse_max, rmse_value) = parity_metrics(&rmse_actual, &rmse_reference);
        assert!(rmse_max < prenet.max_abs);
        assert!(rmse_value > prenet.rmse);
        assert!(!parity_passes(
            "prenet_output",
            &rmse_actual,
            &rmse_reference
        ));
        assert!(!parity_passes("waveform", &[f32::NAN], &[0.0]));
        assert!(!parity_passes("waveform", &[0.0], &[]));
    }

    #[test]
    fn parity_backend_selector_accepts_only_cpu_or_metal() {
        assert_eq!(parse_parity_backend(Some("cpu")), Ok(BackendKind::Cpu));
        assert_eq!(parse_parity_backend(Some("metal")), Ok(BackendKind::Metal));
        assert!(parse_parity_backend(None).is_err());
        assert!(parse_parity_backend(Some("")).is_err());
        assert!(parse_parity_backend(Some("cuda")).is_err());
        assert!(parse_parity_backend(Some("CPU")).is_err());
    }

    #[test]
    fn parity_backend_pass_sentinel_is_exact_and_backend_specific() {
        assert_eq!(
            parity_backend_pass_sentinel(BackendKind::Cpu),
            "BICODEC_MEASURED_PARITY_BACKEND backend=cpu verdict=PASS"
        );
        assert_eq!(
            parity_backend_pass_sentinel(BackendKind::Metal),
            "BICODEC_MEASURED_PARITY_BACKEND backend=metal verdict=PASS"
        );
    }

    #[test]
    #[ignore = "VAST-only: requires authenticated GGUF and official reference outputs"]
    fn official_reference_measured_parity() {
        let backend = parity_backend_from_env();
        let Ok(gguf_path) = std::env::var("VOKRA_BICODEC_PARITY_GGUF") else {
            eprintln!("BiCodec measured parity skipped: VOKRA_BICODEC_PARITY_GGUF is unset");
            return;
        };
        let Ok(reference_dir) = std::env::var("VOKRA_BICODEC_PARITY_REFERENCE") else {
            eprintln!("BiCodec measured parity skipped: VOKRA_BICODEC_PARITY_REFERENCE is unset");
            return;
        };
        let manifest_path = std::path::Path::new(&reference_dir).join("manifest.json");
        let manifest = std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| panic!("read BiCodec reference manifest: {error}"));
        for required in [
            "vokra-bicodec-official-reference-v1",
            "SparkAudio/Spark-TTS BiCodec official source",
            "https://github.com/SparkAudio/Spark-TTS",
            "2f1ea9082400547242641f5271b6f941c9f439d1",
            "642071559bfc6346c2359d19dcb6be3f9dd8a05d",
            "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec",
            "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be",
            "none (fixed literal token vectors)",
            "\"upload\": \"none\"",
            "\"sample_rate\": 16000",
            "\"frame_hop\": 320",
            "\"semantic_vocab\": 8192",
            "\"semantic_codebook_dim\": 8",
            "\"semantic_latent_dim\": 1024",
            "\"global_vocab\": 4096",
            "\"global_tokens\": 32",
        ] {
            assert!(
                manifest.contains(required),
                "reference manifest lacks {required:?}"
            );
        }
        let gguf = GgufFile::open(&gguf_path).expect("authenticated BiCodec GGUF must open");
        let model = Bicodec::from_gguf_with_backend(&gguf, backend)
            .expect("authenticated BiCodec GGUF must bind selected backend");
        let semantic_values = [0, 1, 4_096, 8_191];
        let global_values = [
            0, 1, 4_095, 16, 255, 1_024, 2_048, 3_072, 0, 1, 4_095, 16, 255, 1_024, 2_048, 3_072,
            0, 1, 4_095, 16, 255, 1_024, 2_048, 3_072, 0, 1, 4_095, 16, 255, 1_024, 2_048, 3_072,
        ];
        assert_eq!(
            manifest_string(&manifest, "semantic_csv"),
            semantic_values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        assert_eq!(
            manifest_string(&manifest, "global_csv"),
            global_values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let compute = Compute::for_backend(backend, BICODEC_DECODE_HOT_OPS)
            .expect("selected BiCodec hot-op preflight");
        let semantic = model
            .semantic_decode(&semantic_values, &compute)
            .expect("semantic decode");
        let speaker = model
            .speaker_decode(&global_values, &compute)
            .expect("global FSQ decode");
        let prenet = model
            .prenet
            .forward(&semantic, &speaker, &compute)
            .expect("prenet decode");
        let mut conditioned = prenet.clone();
        for channel in 0..MODEL_DIM {
            for position in 0..semantic_values.len() {
                conditioned[channel * semantic_values.len() + position] += speaker[channel];
            }
        }
        let waveform = model
            .wave
            .forward(&conditioned, semantic_values.len(), &compute)
            .expect("wave decode");
        compare_reference_stage(
            &manifest,
            &reference_dir,
            "semantic_latent",
            &[1, MODEL_DIM, semantic_values.len()],
            &semantic,
        );
        compare_reference_stage(
            &manifest,
            &reference_dir,
            "d_vector",
            &[1, MODEL_DIM],
            &speaker,
        );
        compare_reference_stage(
            &manifest,
            &reference_dir,
            "prenet_output",
            &[1, MODEL_DIM, semantic_values.len()],
            &prenet,
        );
        compare_reference_stage(
            &manifest,
            &reference_dir,
            "waveform",
            &[1, 1, semantic_values.len() * FRAME_HOP],
            &waveform,
        );
        println!("{}", parity_backend_pass_sentinel(backend));
    }

    fn parity_backend_from_env() -> BackendKind {
        parse_parity_backend(std::env::var("VOKRA_BICODEC_PARITY_BACKEND").ok()).unwrap_or_else(
            |error| {
                panic!(
                    "VOKRA_BICODEC_PARITY_BACKEND is required and must be exactly cpu or metal: {error}"
                )
            },
        )
    }

    fn parse_parity_backend(value: Option<&str>) -> std::result::Result<BackendKind, &'static str> {
        match value {
            Some("cpu") => Ok(BackendKind::Cpu),
            Some("metal") => Ok(BackendKind::Metal),
            _ => return Err("backend must be exactly cpu or metal"),
        }
    }

    fn parity_backend_pass_sentinel(backend: BackendKind) -> String {
        let label = match backend {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
            _ => unreachable!("BiCodec parity selector only admits CPU or Metal"),
        };
        format!("BICODEC_MEASURED_PARITY_BACKEND backend={label} verdict=PASS")
    }

    fn compare_reference_stage(
        manifest: &str,
        reference_dir: &str,
        role: &str,
        shape: &[usize],
        actual: &[f32],
    ) {
        let (path, expected_sha) = reference_record(manifest, role, shape);
        let expected_path = match role {
            "semantic_latent" => "semantic_latent.f32",
            "d_vector" => "d_vector.f32",
            "prenet_output" => "prenet_output.f32",
            "waveform" => "waveform.f32",
            _ => panic!("unknown BiCodec reference role {role}"),
        };
        assert_eq!(path, expected_path, "reference {role} path");
        let bytes = std::fs::read(std::path::Path::new(reference_dir).join(path))
            .unwrap_or_else(|error| panic!("read BiCodec reference {role}: {error}"));
        let expected_elements = shape.iter().product::<usize>();
        assert_eq!(
            bytes.len(),
            expected_elements * 4,
            "reference {role} byte shape"
        );
        assert_eq!(
            hex_digest(&sha256_bytes(&bytes)),
            expected_sha,
            "reference {role} SHA-256"
        );
        let reference: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
            .collect();
        assert_eq!(
            actual.len(),
            reference.len(),
            "BiCodec {role} element count"
        );
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "BiCodec {role} actual is non-finite"
        );
        assert!(
            reference.iter().all(|value| value.is_finite()),
            "BiCodec {role} reference is non-finite"
        );
        let (max_abs, rmse) = parity_metrics(actual, &reference);
        let bounds = parity_bounds(role);
        assert!(
            max_abs <= bounds.max_abs,
            "BiCodec {role} max_abs={max_abs:.9e} exceeds bound {:.9e}",
            bounds.max_abs
        );
        assert!(
            rmse <= bounds.rmse,
            "BiCodec {role} rmse={rmse:.9e} exceeds bound {:.9e}",
            bounds.rmse
        );
        println!(
            "BICODEC_MEASURED_PARITY stage={role} elements={} max_abs={max_abs:.9e} rmse={rmse:.9e} max_abs_bound={:.9e} rmse_bound={:.9e} verdict=PASS",
            actual.len(),
            bounds.max_abs,
            bounds.rmse
        );
    }

    #[derive(Clone, Copy)]
    struct ParityBounds {
        max_abs: f64,
        rmse: f64,
    }

    fn parity_bounds(role: &str) -> ParityBounds {
        // These are stage-specific ceilings rounded upward from the exact-head
        // 641ac312 VAST measurements (max_abs/rmse): semantic
        // 1.907348633e-6/3.059676605e-7 -> 4.0e-6/7.0e-7; d-vector
        // 1.847743988e-6/2.558271893e-7 -> 4.0e-6/6.0e-7; prenet
        // 7.539987564e-6/1.297758877e-6 -> 16.0e-6/3.0e-6; waveform
        // 6.183981895e-7/1.134617555e-7 -> 1.5e-6/2.5e-7.  The
        // independent official-source reference was CPU F32; no generic
        // 0.01 tolerance is used because these measured bounds are tighter.
        match role {
            "semantic_latent" => ParityBounds {
                max_abs: 4.0e-6,
                rmse: 7.0e-7,
            },
            "d_vector" => ParityBounds {
                max_abs: 4.0e-6,
                rmse: 6.0e-7,
            },
            "prenet_output" => ParityBounds {
                max_abs: 16.0e-6,
                rmse: 3.0e-6,
            },
            "waveform" => ParityBounds {
                max_abs: 1.5e-6,
                rmse: 2.5e-7,
            },
            _ => panic!("unknown BiCodec parity stage {role}"),
        }
    }

    fn parity_passes(role: &str, actual: &[f32], reference: &[f32]) -> bool {
        if actual.len() != reference.len()
            || actual.iter().any(|value| !value.is_finite())
            || reference.iter().any(|value| !value.is_finite())
            || actual.is_empty()
        {
            return false;
        }
        let (max_abs, rmse) = parity_metrics(actual, reference);
        let bounds = parity_bounds(role);
        max_abs <= bounds.max_abs && rmse <= bounds.rmse
    }

    fn parity_metrics(actual: &[f32], reference: &[f32]) -> (f64, f64) {
        let mut max_abs = 0.0f64;
        let mut sum_squared = 0.0f64;
        for (left, right) in actual.iter().zip(reference) {
            let delta = f64::from(*left) - f64::from(*right);
            max_abs = max_abs.max(delta.abs());
            sum_squared += delta * delta;
        }
        let rmse = (sum_squared / actual.len() as f64).sqrt();
        (max_abs, rmse)
    }

    fn reference_record<'a>(
        manifest: &'a str,
        role: &str,
        expected_shape: &[usize],
    ) -> (&'a str, &'a str) {
        let role_marker = format!("\"{role}\"");
        assert_eq!(
            manifest.matches(&role_marker).count(),
            1,
            "reference tensor record must be unique: {role}"
        );
        let role_start = manifest
            .find(&role_marker)
            .expect("reference tensor record");
        let mut object_start = role_start + role_marker.len();
        object_start = skip_ascii_whitespace(manifest, object_start);
        assert_eq!(manifest.as_bytes().get(object_start), Some(&b':'));
        object_start = skip_ascii_whitespace(manifest, object_start + 1);
        assert_eq!(manifest.as_bytes().get(object_start), Some(&b'{'));
        let object_end = matching_object_end(manifest, object_start);
        let record = &manifest[object_start..=object_end];

        let path = string_field(record, "path");
        let dtype = string_field(record, "dtype");
        assert_eq!(dtype, "F32", "reference tensor dtype");
        let expected_sha = string_field(record, "sha256");
        shape_field(record, expected_shape);
        (path, expected_sha)
    }

    fn skip_ascii_whitespace(text: &str, mut index: usize) -> usize {
        while text
            .as_bytes()
            .get(index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            index += 1;
        }
        index
    }

    fn matching_object_end(text: &str, start: usize) -> usize {
        let bytes = text.as_bytes();
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for (index, byte) in bytes.iter().enumerate().skip(start) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if *byte == b'\\' {
                    escaped = true;
                } else if *byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match *byte {
                b'"' => in_string = true,
                b'{' => depth = depth.checked_add(1).expect("reference object depth"),
                b'}' => {
                    depth = depth.checked_sub(1).expect("reference object close");
                    if depth == 0 {
                        return index;
                    }
                }
                _ => {}
            }
        }
        panic!("reference tensor record has no matching object close")
    }

    fn field_value_start(record: &str, field: &str) -> usize {
        let marker = format!("\"{field}\"");
        assert_eq!(
            record.matches(&marker).count(),
            1,
            "reference field must be unique: {field}"
        );
        let marker_start = record.find(&marker).expect("reference field");
        let colon_start = skip_ascii_whitespace(record, marker_start + marker.len());
        assert_eq!(record.as_bytes().get(colon_start), Some(&b':'));
        skip_ascii_whitespace(record, colon_start + 1)
    }

    fn string_field<'a>(record: &'a str, field: &str) -> &'a str {
        let value_start = field_value_start(record, field);
        assert_eq!(record.as_bytes().get(value_start), Some(&b'"'));
        let value_end = record[value_start + 1..]
            .find('"')
            .map(|offset| value_start + 1 + offset)
            .expect("reference string field terminator");
        let terminator = skip_ascii_whitespace(record, value_end + 1);
        assert!(
            matches!(record.as_bytes().get(terminator), Some(b',' | b'}')),
            "reference string field terminator"
        );
        &record[value_start + 1..value_end]
    }

    fn shape_field(record: &str, expected: &[usize]) {
        let value_start = field_value_start(record, "shape");
        assert_eq!(record.as_bytes().get(value_start), Some(&b'['));
        let bytes = record.as_bytes();
        let mut index = value_start + 1;
        let mut values = Vec::new();
        let mut expect_value = true;
        loop {
            index = skip_ascii_whitespace(record, index);
            match bytes.get(index) {
                Some(b']') => {
                    assert!(
                        !expect_value || values.is_empty(),
                        "reference shape trailing comma"
                    );
                    index += 1;
                    break;
                }
                Some(byte) if expect_value && byte.is_ascii_digit() => {
                    let mut value = 0usize;
                    while let Some(byte) = bytes.get(index).copied() {
                        if !byte.is_ascii_digit() {
                            break;
                        }
                        value = value
                            .checked_mul(10)
                            .and_then(|current| current.checked_add(usize::from(byte - b'0')))
                            .expect("reference shape dimension overflow");
                        index += 1;
                    }
                    values.push(value);
                    expect_value = false;
                }
                _ => panic!("reference shape contains a non-ASCII-number value"),
            }
            index = skip_ascii_whitespace(record, index);
            match bytes.get(index) {
                Some(b',') if !expect_value => {
                    expect_value = true;
                    index += 1;
                }
                Some(b']') if !expect_value => {
                    index += 1;
                    break;
                }
                _ => panic!("reference shape requires comma or close"),
            }
        }
        index = skip_ascii_whitespace(record, index);
        assert!(
            matches!(bytes.get(index), Some(b',' | b'}')),
            "reference shape terminator"
        );
        assert_eq!(values, expected, "reference tensor shape");
    }

    fn hex_digest(bytes: &[u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn manifest_string<'a>(manifest: &'a str, key: &str) -> &'a str {
        let marker = format!("\"{key}\": \"");
        assert_eq!(
            manifest.matches(&marker).count(),
            1,
            "manifest key count: {key}"
        );
        let start = manifest.find(&marker).expect("manifest string field") + marker.len();
        let end = manifest[start..]
            .find('"')
            .expect("manifest string field terminator")
            + start;
        &manifest[start..end]
    }
}
