//! Native SpeechTokenizer residual-VQ token-to-waveform decoder.
//!
//! This module binds the exact public
//! `vokra/speechtokenizer@576865f9e04b1f046b5c6601813c288b6439a8b2`
//! artifact. The released decoder is an eight-codebook, 1024-dimensional
//! residual VQ followed by a non-causal, weight-normalized SEANet with a
//! two-layer residual LSTM. The complete 166-tensor checkpoint manifest is
//! authenticated before any inference payload is decoded.
//!
//! RVQ, Conv1D, transposed-convolution projection and LSTM projections use one
//! selected [`Compute`] backend. Weight-normalization folding, reflect padding,
//! ELU/gate activations, transposed-convolution scatter/trim and residual adds
//! are deterministic host tensor-layout glue. Unsupported backends fail at
//! whole-model preflight; no learned operation silently falls back to CPU.
//!
//! Primary source: `ZhangXInFD/SpeechTokenizer` commit
//! `30c96fb32a9fc06a2258c98119e237def051e46c`, canonical model repository
//! `OpenMOSS-Team/SpeechTokenizer` revision
//! `4d54939beef00572fa7bfe41ee35a335c3732f51`. Independent real-weight parity
//! is generated only by calling that official implementation through
//! `tools/parity/speechtokenizer/dump_reference.py` on VAST.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{CodebookTable, EncodecRvqAttrs};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

/// Runtime/converter architecture tag.
pub const ARCH: &str = "speechtokenizer";
/// Canonical released model identity in the public GGUF.
pub const NAME: &str = "speechtokenizer";
/// Historical upstream slug stamped into the immutable public artifact.
pub const PUBLIC_UPSTREAM_HF: &str = "fnlp/SpeechTokenizer";
/// Canonical upstream repository after the Hugging Face organization move.
pub const CANONICAL_UPSTREAM_HF: &str = "OpenMOSS-Team/SpeechTokenizer";
/// Immutable upstream checkpoint/config revision.
pub const UPSTREAM_REVISION: &str = "4d54939beef00572fa7bfe41ee35a335c3732f51";
/// Immutable official source revision.
pub const SOURCE_REVISION: &str = "30c96fb32a9fc06a2258c98119e237def051e46c";
/// Official PyTorch checkpoint SHA-256 from authenticated HF LFS metadata.
pub const CHECKPOINT_SHA256: &str =
    "d04593b6c9a4b475f91ca481141a6ef5b23e6ac112f347dd2b2717f193c1c728";
/// Official config SHA-256 at [`UPSTREAM_REVISION`].
pub const CONFIG_SHA256: &str = "ea343ad69ca7e70c8febf8fc4cda683b1c4b1c36709e5e577936ffb05d62e6eb";
/// Public Vokra GGUF revision authenticated by [`SPEC`].
pub const PUBLIC_REVISION: &str = "576865f9e04b1f046b5c6601813c288b6439a8b2";
/// Public GGUF SHA-256 from authenticated HF LFS metadata.
pub const PUBLIC_GGUF_SHA256: &str =
    "ebed5bcfcc4113b5fd2211cd363ab2e754b6afba8bd55162078ff7e7914ed83e";
/// Public GGUF byte length.
pub const PUBLIC_GGUF_BYTES: u64 = 481_857_952;

/// PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// PCM samples represented by one residual-VQ frame.
pub const FRAME_HOP: usize = 320;
/// Maximum residual quantizer count in the release.
pub const NUM_CODEBOOKS: usize = 8;
/// Entries in every codebook.
pub const CODEBOOK_SIZE: usize = 1_024;
/// Latent/codebook width.
pub const DIMENSION: usize = 1_024;

const CATEGORY: &str = "codec";
const LABEL: &str = "speechtokenizer";
const LSTM_LAYERS: usize = 2;
const RATIOS: [usize; 4] = [8, 5, 4, 2];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: 166,
    manifest_sha256: [
        0x48, 0xbc, 0x3b, 0xa1, 0xee, 0x88, 0xca, 0x59, 0x8e, 0x1b, 0x1d, 0xbe, 0x3d, 0x0e, 0x32,
        0x22, 0x19, 0xa7, 0xe8, 0x8e, 0x14, 0xd4, 0x5d, 0x6f, 0x3a, 0x25, 0xcb, 0x5d, 0xe9, 0xbf,
        0xe7, 0x99,
    ],
};

/// Complete learned-op inventory for released token-to-waveform execution.
pub const SPEECHTOKENIZER_DECODE_HOT_OPS: &[HotOp] =
    &[HotOp::EncodecRvq, HotOp::Conv1d, HotOp::Gemm, HotOp::Gemv];

/// Strict real-weight SpeechTokenizer decoder.
#[derive(Debug)]
pub struct SpeechTokenizer {
    backend: BackendKind,
    weight_license: LicenseClass,
    codebooks: Vec<CodebookTable>,
    decoder: SeanetDecoder,
}

impl SpeechTokenizer {
    /// Binds the exact public GGUF using the CPU backend.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_backend(file, BackendKind::Cpu)
    }

    /// Binds the exact public GGUF and preflights one backend for the complete
    /// decode graph before reading any learned payload.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, SPEECHTOKENIZER_DECODE_HOT_OPS)?;
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, "vokra.model.category", CATEGORY)?;
        require_string(file, "vokra.provenance.upstream_hf", PUBLIC_UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_SOURCE, PUBLIC_UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;

        let mut codebooks = Vec::with_capacity(NUM_CODEBOOKS);
        for index in 0..NUM_CODEBOOKS {
            let embedding = tensor(
                file,
                &format!("quantizer.vq.layers.{index}._codebook.embed"),
                &[CODEBOOK_SIZE, DIMENSION],
            )?;
            codebooks.push(CodebookTable::new(CODEBOOK_SIZE, DIMENSION, embedding)?);
        }

        Ok(Self {
            backend,
            weight_license: checkpoint.weight_license(),
            codebooks,
            decoder: SeanetDecoder::load(file)?,
        })
    }

    /// Opens and strictly binds a GGUF using the CPU backend.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for all learned operations. Decode performs the
    /// same complete coverage preflight before executing.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[must_use]
    pub const fn frame_hop(&self) -> usize {
        FRAME_HOP
    }

    #[must_use]
    pub const fn max_quantizers(&self) -> usize {
        NUM_CODEBOOKS
    }

    /// Decodes frame-major `[frames, num_quantizers]` u32 code IDs.
    ///
    /// The official decoder accepts a non-empty prefix of its eight residual
    /// codebooks. Missing codebooks are never silently padded.
    pub fn decode_frame_major(
        &self,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "speechtokenizer: frames must be > 0".to_owned(),
            ));
        }
        if !(1..=NUM_CODEBOOKS).contains(&num_quantizers) {
            return Err(VokraError::InvalidArgument(format!(
                "speechtokenizer: num_quantizers {num_quantizers} is outside 1..={NUM_CODEBOOKS}"
            )));
        }
        let expected = frames.checked_mul(num_quantizers).ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: frames * num_quantizers overflows usize".to_owned(),
            )
        })?;
        if codes.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "speechtokenizer: codes.len() {} != frames {frames} * num_quantizers \
                 {num_quantizers} = {expected}",
                codes.len()
            )));
        }

        let compute = Compute::for_backend(self.backend, SPEECHTOKENIZER_DECODE_HOT_OPS)?;
        let time_major = compute.encodec_rvq_f32(
            codes,
            frames,
            &self.codebooks[..num_quantizers],
            &EncodecRvqAttrs {
                n_codebooks: num_quantizers,
                codebook_size: CODEBOOK_SIZE,
                d_model: DIMENSION,
            },
        )?;
        let latent_len = DIMENSION.checked_mul(frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: latent element count overflows usize".to_owned(),
            )
        })?;
        let mut latent = vec![0.0f32; latent_len];
        for frame in 0..frames {
            for channel in 0..DIMENSION {
                latent[channel * frames + frame] = time_major[frame * DIMENSION + channel];
            }
        }

        let (pcm, samples) = self.decoder.forward(&latent, frames, &compute)?;
        let expected_samples = frames.checked_mul(FRAME_HOP).ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: output sample count overflows usize".to_owned(),
            )
        })?;
        if samples != expected_samples || pcm.len() != expected_samples {
            return Err(VokraError::InvalidArgument(format!(
                "speechtokenizer: decoder emitted {} values / {samples} samples, expected \
                 {expected_samples}",
                pcm.len()
            )));
        }
        reject_non_finite("decoded PCM", &pcm)?;
        Ok(pcm)
    }
}

#[derive(Debug)]
struct SeanetDecoder {
    initial: WeightNormConv1d,
    lstm: Lstm,
    upsample: Vec<WeightNormConvTranspose1d>,
    residuals: Vec<ResidualBlock>,
    final_conv: WeightNormConv1d,
}

impl SeanetDecoder {
    fn load(file: &GgufFile) -> Result<Self> {
        let initial =
            WeightNormConv1d::load(file, "decoder.model.0.conv.conv", DIMENSION, 1_024, 7)?;
        let mut upsample = Vec::with_capacity(RATIOS.len());
        let mut residuals = Vec::with_capacity(RATIOS.len());
        for (transpose, residual, channels, next, ratio) in [
            (3, 4, 1_024, 512, 8),
            (6, 7, 512, 256, 5),
            (9, 10, 256, 128, 4),
            (12, 13, 128, 64, 2),
        ] {
            upsample.push(WeightNormConvTranspose1d::load(
                file,
                &format!("decoder.model.{transpose}.convtr.convtr"),
                channels,
                next,
                ratio * 2,
                ratio,
            )?);
            residuals.push(ResidualBlock::load(
                file,
                &format!("decoder.model.{residual}"),
                next,
            )?);
        }
        Ok(Self {
            initial,
            lstm: Lstm::load(file, "decoder.model.1.lstm", DIMENSION, LSTM_LAYERS)?,
            upsample,
            residuals,
            final_conv: WeightNormConv1d::load(file, "decoder.model.15.conv.conv", 64, 1, 7)?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) = self.initial.forward(input, input_len, compute)?;
        hidden = self.lstm.forward(&hidden, time, compute)?;
        for stage in 0..RATIOS.len() {
            elu_inplace(&mut hidden);
            (hidden, time) = self.upsample[stage].forward(&hidden, time, compute)?;
            hidden = self.residuals[stage].forward(&hidden, time, compute)?;
        }
        elu_inplace(&mut hidden);
        self.final_conv.forward(&hidden, time, compute)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    first: WeightNormConv1d,
    second: WeightNormConv1d,
    shortcut: WeightNormConv1d,
}

impl ResidualBlock {
    fn load(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        let hidden = channels / 2;
        Ok(Self {
            first: WeightNormConv1d::load(
                file,
                &format!("{prefix}.block.1.conv.conv"),
                channels,
                hidden,
                3,
            )?,
            second: WeightNormConv1d::load(
                file,
                &format!("{prefix}.block.3.conv.conv"),
                hidden,
                channels,
                1,
            )?,
            shortcut: WeightNormConv1d::load(
                file,
                &format!("{prefix}.shortcut.conv.conv"),
                channels,
                channels,
                1,
            )?,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("residual input", input, self.first.input_channels, time)?;
        let (shortcut, shortcut_time) = self.shortcut.forward(input, time, compute)?;
        let mut hidden = input.to_vec();
        elu_inplace(&mut hidden);
        let (mut hidden, first_time) = self.first.forward(&hidden, time, compute)?;
        elu_inplace(&mut hidden);
        let (mut hidden, second_time) = self.second.forward(&hidden, first_time, compute)?;
        if shortcut_time != time
            || first_time != time
            || second_time != time
            || hidden.len() != shortcut.len()
        {
            return Err(VokraError::InvalidArgument(format!(
                "speechtokenizer: residual branch shape mismatch: shortcut={shortcut_time}, \
                 first={first_time}, second={second_time}, values={}/{}",
                hidden.len(),
                shortcut.len()
            )));
        }
        for (value, skip) in hidden.iter_mut().zip(shortcut) {
            *value += skip;
        }
        Ok(hidden)
    }
}

#[derive(Debug)]
struct WeightNormConv1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl WeightNormConv1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        let magnitude = tensor(
            file,
            &format!("{prefix}.weight_g"),
            &[output_channels, 1, 1],
        )?;
        let direction = tensor(
            file,
            &format!("{prefix}.weight_v"),
            &[output_channels, input_channels, kernel],
        )?;
        let row_width = input_channels.checked_mul(kernel).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "speechtokenizer: weight-norm `{prefix}` row width overflows usize"
            ))
        })?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            weight: fold_weight_norm(&magnitude, &direction, output_channels, row_width, prefix)?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout("conv1d input", input, self.input_channels, input_len)?;
        if input_len == 0 || self.kernel == 0 {
            return Err(VokraError::InvalidArgument(
                "speechtokenizer: conv1d requires non-empty input/kernel".to_owned(),
            ));
        }
        let padding_total = self.kernel - 1;
        let extra = extra_padding(input_len, self.kernel, 1, padding_total)?;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let padded = reflect_pad1d(
            input,
            self.input_channels,
            input_len,
            padding_left,
            padding_right + extra,
        )?;
        let padded_len = input_len
            .checked_add(padding_total)
            .and_then(|value| value.checked_add(extra))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: conv1d padded length overflows usize".to_owned(),
                )
            })?;
        let output_len = padded_len - self.kernel + 1;
        let output_values = self
            .output_channels
            .checked_mul(output_len)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: conv1d output size overflows usize".to_owned(),
                )
            })?;
        let mut output = vec![0.0f32; output_values];
        compute.conv1d_f32(
            &padded,
            self.input_channels,
            padded_len,
            &self.weight,
            self.output_channels,
            self.kernel,
            Some(&self.bias),
            1,
            0,
            &mut output,
        )?;
        Ok((output, output_len))
    }
}

#[derive(Debug)]
struct WeightNormConvTranspose1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl WeightNormConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        let magnitude = tensor(file, &format!("{prefix}.weight_g"), &[input_channels, 1, 1])?;
        let direction = tensor(
            file,
            &format!("{prefix}.weight_v"),
            &[input_channels, output_channels, kernel],
        )?;
        let row_width = output_channels.checked_mul(kernel).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "speechtokenizer: transpose weight-norm `{prefix}` row width overflows usize"
            ))
        })?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: fold_weight_norm(&magnitude, &direction, input_channels, row_width, prefix)?,
            bias: tensor(file, &format!("{prefix}.bias"), &[output_channels])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout(
            "conv-transpose input",
            input,
            self.input_channels,
            input_len,
        )?;
        if input_len == 0 || self.stride == 0 || self.kernel < self.stride {
            return Err(VokraError::InvalidArgument(format!(
                "speechtokenizer: invalid conv-transpose input_len={input_len}, kernel={}, stride={}",
                self.kernel, self.stride
            )));
        }
        let raw_len = (input_len - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: conv-transpose output length overflow".to_owned(),
                )
            })?;
        let input_values = input_len.checked_mul(self.input_channels).ok_or_else(|| {
            VokraError::InvalidArgument("speechtokenizer: transpose input size overflow".to_owned())
        })?;
        let mut time_major = vec![0.0f32; input_values];
        for time in 0..input_len {
            for channel in 0..self.input_channels {
                time_major[time * self.input_channels + channel] =
                    input[channel * input_len + time];
            }
        }
        let projected_width = self
            .output_channels
            .checked_mul(self.kernel)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: transpose projection width overflow".to_owned(),
                )
            })?;
        let projected_values = input_len.checked_mul(projected_width).ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: transpose projection size overflow".to_owned(),
            )
        })?;
        let mut projected = vec![0.0f32; projected_values];
        compute.gemm_f32(
            input_len,
            projected_width,
            self.input_channels,
            &time_major,
            &self.weight,
            None,
            &mut projected,
        )?;
        let raw_values = self.output_channels.checked_mul(raw_len).ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: conv-transpose raw size overflow".to_owned(),
            )
        })?;
        let mut raw = vec![0.0f32; raw_values];
        for channel in 0..self.output_channels {
            raw[channel * raw_len..(channel + 1) * raw_len].fill(self.bias[channel]);
        }
        for time in 0..input_len {
            let destination = time * self.stride;
            for channel in 0..self.output_channels {
                let source = time * projected_width + channel * self.kernel;
                for tap in 0..self.kernel {
                    raw[channel * raw_len + destination + tap] += projected[source + tap];
                }
            }
        }

        let padding_total = self.kernel - self.stride;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let output_len = raw_len
            .checked_sub(padding_left + padding_right)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: conv-transpose trim exceeds output".to_owned(),
                )
            })?;
        let output_values = self
            .output_channels
            .checked_mul(output_len)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: conv-transpose output size overflow".to_owned(),
                )
            })?;
        let mut output = vec![0.0f32; output_values];
        for channel in 0..self.output_channels {
            output[channel * output_len..(channel + 1) * output_len].copy_from_slice(
                &raw[channel * raw_len + padding_left
                    ..channel * raw_len + padding_left + output_len],
            );
        }
        Ok((output, output_len))
    }
}

#[derive(Debug)]
struct LstmLayer {
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

#[derive(Debug)]
struct Lstm {
    dimension: usize,
    layers: Vec<LstmLayer>,
}

impl Lstm {
    fn load(file: &GgufFile, prefix: &str, dimension: usize, layers: usize) -> Result<Self> {
        let gates = 4 * dimension;
        let mut bound = Vec::with_capacity(layers);
        for layer in 0..layers {
            bound.push(LstmLayer {
                weight_ih: tensor(
                    file,
                    &format!("{prefix}.weight_ih_l{layer}"),
                    &[gates, dimension],
                )?,
                weight_hh: tensor(
                    file,
                    &format!("{prefix}.weight_hh_l{layer}"),
                    &[gates, dimension],
                )?,
                bias_ih: tensor(file, &format!("{prefix}.bias_ih_l{layer}"), &[gates])?,
                bias_hh: tensor(file, &format!("{prefix}.bias_hh_l{layer}"), &[gates])?,
            });
        }
        Ok(Self {
            dimension,
            layers: bound,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("LSTM input", input, self.dimension, time)?;
        let residual = input;
        let gates = 4 * self.dimension;
        let mut layer_input = input.to_vec();
        for layer in &self.layers {
            let values = self.dimension.checked_mul(time).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "speechtokenizer: LSTM output size overflows usize".to_owned(),
                )
            })?;
            let mut output = vec![0.0f32; values];
            let mut hidden = vec![0.0f32; self.dimension];
            let mut cell = vec![0.0f32; self.dimension];
            let mut step_input = vec![0.0f32; self.dimension];
            let mut input_gates = vec![0.0f32; gates];
            let mut recurrent_gates = vec![0.0f32; gates];
            for step in 0..time {
                for dimension in 0..self.dimension {
                    step_input[dimension] = layer_input[dimension * time + step];
                }
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_ih,
                    &step_input,
                    Some(&layer.bias_ih),
                    &mut input_gates,
                )?;
                compute.gemv_f32(
                    gates,
                    self.dimension,
                    &layer.weight_hh,
                    &hidden,
                    Some(&layer.bias_hh),
                    &mut recurrent_gates,
                )?;
                for dimension in 0..self.dimension {
                    let input_gate = sigmoid(input_gates[dimension] + recurrent_gates[dimension]);
                    let forget_gate = sigmoid(
                        input_gates[self.dimension + dimension]
                            + recurrent_gates[self.dimension + dimension],
                    );
                    let candidate = (input_gates[2 * self.dimension + dimension]
                        + recurrent_gates[2 * self.dimension + dimension])
                        .tanh();
                    let output_gate = sigmoid(
                        input_gates[3 * self.dimension + dimension]
                            + recurrent_gates[3 * self.dimension + dimension],
                    );
                    cell[dimension] = forget_gate * cell[dimension] + input_gate * candidate;
                    hidden[dimension] = output_gate * cell[dimension].tanh();
                    output[dimension * time + step] = hidden[dimension];
                }
            }
            layer_input = output;
        }
        for (value, &skip) in layer_input.iter_mut().zip(residual) {
            *value += skip;
        }
        Ok(layer_input)
    }
}

fn fold_weight_norm(
    magnitude: &[f32],
    direction: &[f32],
    rows: usize,
    row_width: usize,
    label: &str,
) -> Result<Vec<f32>> {
    let expected = rows.checked_mul(row_width).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "speechtokenizer: weight-norm `{label}` size overflows usize"
        ))
    })?;
    if magnitude.len() != rows || direction.len() != expected {
        return Err(VokraError::ModelLoad(format!(
            "speechtokenizer: weight-norm `{label}` has magnitude/direction lengths {}/{}, \
             expected {rows}/{expected}",
            magnitude.len(),
            direction.len()
        )));
    }
    let mut output = vec![0.0f32; expected];
    for row in 0..rows {
        let source = &direction[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(VokraError::ModelLoad(format!(
                "speechtokenizer: weight-norm `{label}` row {row} has invalid L2 norm {norm}"
            )));
        }
        let scale = magnitude[row] / norm;
        for (destination, &value) in output[row * row_width..(row + 1) * row_width]
            .iter_mut()
            .zip(source)
        {
            *destination = value * scale;
        }
    }
    reject_non_finite(label, &output).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(output)
}

fn extra_padding(
    input_len: usize,
    kernel: usize,
    stride: usize,
    padding_total: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "speechtokenizer: convolution stride must be > 0".to_owned(),
        ));
    }
    let padded = input_len.checked_add(padding_total).ok_or_else(|| {
        VokraError::InvalidArgument(
            "speechtokenizer: convolution padded length overflow".to_owned(),
        )
    })?;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "speechtokenizer: padded length {padded} is smaller than kernel {kernel}"
        )));
    }
    let frames = (padded - kernel).div_ceil(stride) + 1;
    let ideal = (frames - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(kernel))
        .and_then(|value| value.checked_sub(padding_total))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: ideal convolution length overflow".to_owned(),
            )
        })?;
    ideal.checked_sub(input_len).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "speechtokenizer: ideal convolution length {ideal} is smaller than input {input_len}"
        ))
    })
}

fn reflect_pad1d(
    input: &[f32],
    channels: usize,
    length: usize,
    left: usize,
    right: usize,
) -> Result<Vec<f32>> {
    require_layout("reflect-pad input", input, channels, length)?;
    if length == 0 {
        return Err(VokraError::InvalidArgument(
            "speechtokenizer: reflect padding requires non-empty input".to_owned(),
        ));
    }
    let max_padding = left.max(right);
    let extra = if length <= max_padding {
        max_padding - length + 1
    } else {
        0
    };
    let base_len = length.checked_add(extra).ok_or_else(|| {
        VokraError::InvalidArgument(
            "speechtokenizer: reflect padding base length overflow".to_owned(),
        )
    })?;
    let padded_len = base_len
        .checked_add(left)
        .and_then(|value| value.checked_add(right))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "speechtokenizer: reflect padding output length overflow".to_owned(),
            )
        })?;
    let output_len = padded_len - extra;
    let output_values = channels.checked_mul(output_len).ok_or_else(|| {
        VokraError::InvalidArgument(
            "speechtokenizer: reflect padding output size overflow".to_owned(),
        )
    })?;
    let mut output = vec![0.0f32; output_values];
    for channel in 0..channels {
        for output_index in 0..output_len {
            let logical = output_index as isize - left as isize;
            let source = if logical < 0 {
                (-logical) as usize
            } else if logical >= base_len as isize {
                (2 * base_len as isize - logical - 2) as usize
            } else {
                logical as usize
            };
            output[channel * output_len + output_index] = if source < length {
                input[channel * length + source]
            } else {
                0.0
            };
        }
    }
    Ok(output)
}

fn tensor(file: &GgufFile, name: &str, dimensions: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("speechtokenizer: missing tensor `{name}`"))
    })?;
    let expected = dimensions
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "speechtokenizer: tensor `{name}` shape {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!(
            "speechtokenizer: tensor `{name}` decode failed: {error}"
        ))
    })?;
    reject_non_finite(name, &values).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(values)
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "speechtokenizer: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_layout(label: &str, values: &[f32], channels: usize, time: usize) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!("speechtokenizer: {label} shape overflows usize"))
    })?;
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "speechtokenizer: {label} has {} values, expected {channels}x{time}={expected}",
            values.len()
        )));
    }
    Ok(())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "speechtokenizer: {label} contains non-finite {value} at index {index}"
        )));
    }
    Ok(())
}

fn elu_inplace(values: &mut [f32]) {
    for value in values {
        if *value < 0.0 {
            *value = value.exp_m1();
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_release_contract_is_exact() {
        assert_eq!(RATIOS.iter().product::<usize>(), FRAME_HOP);
        assert_eq!(SAMPLE_RATE as usize / FRAME_HOP, 50);
        assert_eq!(SPEC.tensor_count, 166);
        assert_eq!(SPEECHTOKENIZER_DECODE_HOT_OPS.len(), 4);
    }

    #[test]
    fn complete_decode_hot_ops_are_cpu_and_metal_covered() {
        Compute::for_backend(BackendKind::Cpu, SPEECHTOKENIZER_DECODE_HOT_OPS)
            .expect("CPU covers the complete SpeechTokenizer decoder");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, SPEECHTOKENIZER_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("SpeechTokenizer decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn weight_norm_folds_each_primary_plane() {
        let actual = fold_weight_norm(&[2.0, 3.0], &[3.0, 4.0, 0.0, 2.0], 2, 2, "x")
            .expect("valid weight norm");
        assert_eq!(actual, vec![1.2, 1.6, 0.0, 3.0]);
    }

    #[test]
    fn reflect_padding_matches_official_short_input_helper() {
        let normal = reflect_pad1d(&[1.0, 2.0, 3.0, 4.0], 1, 4, 2, 2).unwrap();
        assert_eq!(normal, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
        let short = reflect_pad1d(&[7.0], 1, 1, 3, 3).unwrap();
        assert_eq!(short, vec![0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]);
    }
}
