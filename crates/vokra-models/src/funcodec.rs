//! Native FunCodec token-to-waveform decoder for Mac CPU and Metal.
//!
//! This module binds the exact public
//! `vokra/funcodec@ea8be2e051ede0365496e8cd3b24d732c8bc6ffb` artifact.  The
//! checkpoint is Alibaba DAMO's 16 kHz, 32-codebook, 320-sample-hop release:
//! an unfactorized residual VQ followed by a non-causal SEANet decoder with
//! one-group time GroupNorm and a two-layer residual LSTM.  The public GGUF
//! also preserves training-only discriminator and mel-loss tensors; the
//! complete 230-tensor name/shape manifest is authenticated before any
//! inference tensor is decoded.
//!
//! Every learned reduction uses one selected [`Compute`] backend.  RVQ fold,
//! Conv1D, GroupNorm, transposed-convolution projection and LSTM projections
//! therefore run on CPU or Metal as one whole-model choice.  Reflect padding,
//! ELU/gate activations, transposed-convolution scatter/trim and residual adds
//! are deterministic host tensor-layout glue.  Unsupported backends fail at
//! preflight; there is no per-op CPU fallback.
//!
//! Primary source: `modelscope/FunCodec` commit
//! `b467b73e4025a123a68e64de9ba445d6a57d1984` and the fixed upstream model
//! revision `ef9fbae4943cb272b8803e8a0f3c974fa1003b1f`.  Independent
//! real-weight parity is generated only from that official implementation by
//! `tools/parity/funcodec/dump_reference.py` on VAST.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{CodebookTable, EncodecRvqAttrs};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

/// Runtime/converter architecture tag.
pub const ARCH: &str = "funcodec";
/// Canonical released model identity.
pub const NAME: &str = "funcodec-encodec-zh-en-16k-nq32-ds320";
/// Upstream model repository.
pub const UPSTREAM_HF: &str =
    "alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch";
/// Immutable upstream model revision.
pub const UPSTREAM_REVISION: &str = "ef9fbae4943cb272b8803e8a0f3c974fa1003b1f";
/// Official source commit immediately preceding the model publication.
pub const SOURCE_REVISION: &str = "b467b73e4025a123a68e64de9ba445d6a57d1984";
/// Official model checkpoint SHA-256, read from authenticated HF LFS metadata.
pub const CHECKPOINT_SHA256: &str =
    "08dd881b74daa150c405418b613496e872bbad4edd2d3c1d6d94ecf7199ac42c";
/// Public Vokra GGUF revision authenticated by the manifest below.
pub const PUBLIC_REVISION: &str = "ea8be2e051ede0365496e8cd3b24d732c8bc6ffb";
/// Public GGUF SHA-256, read from authenticated HF LFS metadata.
pub const PUBLIC_GGUF_SHA256: &str =
    "b6fa6c903e23b1785f517f4e6c33c5d323a227a94ea757442e5d177d48d5781d";
/// Public GGUF byte length.
pub const PUBLIC_GGUF_BYTES: u64 = 95_072_832;

/// PCM sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of PCM samples represented by one codec frame.
pub const FRAME_HOP: usize = 320;
/// Maximum residual quantizer count in the release.
pub const NUM_CODEBOOKS: usize = 32;
/// Entries in each residual codebook.
pub const CODEBOOK_SIZE: usize = 1_024;
/// SEANet/RVQ latent width.
pub const DIMENSION: usize = 128;

const CATEGORY: &str = "codec";
const LABEL: &str = "funcodec";
const NUM_FILTERS: usize = 32;
const LSTM_DIMENSION: usize = 512;
const LSTM_LAYERS: usize = 2;
const RATIOS: [usize; 4] = [8, 5, 4, 2];
const GROUP_NORM_EPS: f32 = 1.0e-5;

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: 230,
    manifest_sha256: [
        0x0d, 0x6f, 0xc1, 0x11, 0xe5, 0xb4, 0x5e, 0xdb, 0xdc, 0xa3, 0xe7, 0x4b, 0x4d, 0x3b, 0x95,
        0x01, 0x9d, 0x4e, 0xd6, 0x32, 0x4d, 0x86, 0x14, 0x59, 0x01, 0x22, 0xd0, 0xe7, 0x1d, 0xd8,
        0x16, 0x52,
    ],
};

/// Complete learned-op inventory for released token-to-waveform execution.
pub const FUNCODEC_DECODE_HOT_OPS: &[HotOp] = &[
    HotOp::EncodecRvq,
    HotOp::Conv1d,
    HotOp::GroupNorm,
    HotOp::Gemm,
    HotOp::Gemv,
];

/// Strict real-weight FunCodec decoder.
#[derive(Debug)]
pub struct FunCodec {
    backend: BackendKind,
    weight_license: LicenseClass,
    codebooks: Vec<CodebookTable>,
    decoder: SeanetDecoder,
}

impl FunCodec {
    /// Binds the exact public GGUF using the CPU backend.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_backend(file, BackendKind::Cpu)
    }

    /// Binds the exact public GGUF and preflights one backend for the complete
    /// decode graph before reading any learned payload.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, FUNCODEC_DECODE_HOT_OPS)?;
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, "vokra.model.category", CATEGORY)?;
        require_string(file, "vokra.provenance.upstream_hf", UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_SOURCE, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;

        let embedding = tensor(
            file,
            "quantizer.rq.model.embed",
            &[NUM_CODEBOOKS, CODEBOOK_SIZE, DIMENSION],
        )?;
        let table_size = CODEBOOK_SIZE * DIMENSION;
        let mut codebooks = Vec::with_capacity(NUM_CODEBOOKS);
        for index in 0..NUM_CODEBOOKS {
            let start = index * table_size;
            codebooks.push(CodebookTable::new(
                CODEBOOK_SIZE,
                DIMENSION,
                embedding[start..start + table_size].to_vec(),
            )?);
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

    /// Selects one backend for every learned operation.  Use
    /// [`Self::from_gguf_with_backend`] when an eager coverage preflight is
    /// required before weight decoding.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the checkpoint weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the decoder waveform sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Returns the waveform samples represented by one codec frame.
    #[must_use]
    pub const fn frame_hop(&self) -> usize {
        FRAME_HOP
    }

    /// Maximum residual-codebook prefix accepted by the released decoder.
    #[must_use]
    pub const fn max_quantizers(&self) -> usize {
        NUM_CODEBOOKS
    }

    /// Decode frame-major `[frames, num_quantizers]` u32 code IDs.
    ///
    /// The official residual decoder accepts every non-empty prefix of its 32
    /// codebooks. `num_quantizers` is therefore explicit and must be in
    /// `1..=32`; missing codebooks are not silently padded.
    pub fn decode_frame_major(
        &self,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "funcodec: frames must be > 0".to_owned(),
            ));
        }
        if !(1..=NUM_CODEBOOKS).contains(&num_quantizers) {
            return Err(VokraError::InvalidArgument(format!(
                "funcodec: num_quantizers {num_quantizers} is outside 1..={NUM_CODEBOOKS}"
            )));
        }
        let expected = frames.checked_mul(num_quantizers).ok_or_else(|| {
            VokraError::InvalidArgument(
                "funcodec: frames * num_quantizers overflows usize".to_owned(),
            )
        })?;
        if codes.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "funcodec: codes.len() {} != frames {frames} * num_quantizers \
                 {num_quantizers} = {expected}",
                codes.len()
            )));
        }

        let compute = Compute::for_backend(self.backend, FUNCODEC_DECODE_HOT_OPS)?;
        let latent_time_major = compute.encodec_rvq_f32(
            codes,
            frames,
            &self.codebooks[..num_quantizers],
            &EncodecRvqAttrs {
                n_codebooks: num_quantizers,
                codebook_size: CODEBOOK_SIZE,
                d_model: DIMENSION,
            },
        )?;
        let mut latent = vec![0.0f32; latent_time_major.len()];
        for frame in 0..frames {
            for channel in 0..DIMENSION {
                latent[channel * frames + frame] = latent_time_major[frame * DIMENSION + channel];
            }
        }

        let (pcm, samples) = self.decoder.forward(&latent, frames, &compute)?;
        let expected_samples = frames.checked_mul(FRAME_HOP).ok_or_else(|| {
            VokraError::InvalidArgument("funcodec: output sample count overflows usize".to_owned())
        })?;
        if samples != expected_samples || pcm.len() != expected_samples {
            return Err(VokraError::InvalidArgument(format!(
                "funcodec: decoder emitted {} values / {samples} samples, expected \
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
    initial: NormConv1d,
    lstm: Lstm,
    upsample: Vec<NormConvTranspose1d>,
    residuals: Vec<ResidualBlock>,
    final_conv: NormConv1d,
}

impl SeanetDecoder {
    fn load(file: &GgufFile) -> Result<Self> {
        let initial = NormConv1d::load(file, "decoder.model.0.conv", DIMENSION, 512, 7)?;
        let lstm = Lstm::load(file, "decoder.model.1.lstm", LSTM_DIMENSION, LSTM_LAYERS)?;
        let mut upsample = Vec::with_capacity(RATIOS.len());
        let mut residuals = Vec::with_capacity(RATIOS.len());
        for (transpose, residual, channels, next, ratio) in [
            (3, 4, 512, 256, 8),
            (6, 7, 256, 128, 5),
            (9, 10, 128, 64, 4),
            (12, 13, 64, 32, 2),
        ] {
            upsample.push(NormConvTranspose1d::load(
                file,
                &format!("decoder.model.{transpose}.convtr"),
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
            lstm,
            upsample,
            residuals,
            final_conv: NormConv1d::load(file, "decoder.model.15.conv", NUM_FILTERS, 1, 7)?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) = self.initial.forward(input, input_len, 1, compute)?;
        hidden = self.lstm.forward(&hidden, time, compute)?;
        for stage in 0..RATIOS.len() {
            elu_inplace(&mut hidden);
            (hidden, time) = self.upsample[stage].forward(&hidden, time, compute)?;
            hidden = self.residuals[stage].forward(&hidden, time, compute)?;
        }
        elu_inplace(&mut hidden);
        self.final_conv.forward(&hidden, time, 1, compute)
    }
}

#[derive(Debug)]
struct ResidualBlock {
    first: NormConv1d,
    second: NormConv1d,
    shortcut: NormConv1d,
}

impl ResidualBlock {
    fn load(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        let hidden = channels / 2;
        Ok(Self {
            first: NormConv1d::load(file, &format!("{prefix}.block.1.conv"), channels, hidden, 3)?,
            second: NormConv1d::load(file, &format!("{prefix}.block.3.conv"), hidden, channels, 1)?,
            shortcut: NormConv1d::load(
                file,
                &format!("{prefix}.shortcut.conv"),
                channels,
                channels,
                1,
            )?,
        })
    }

    fn forward(&self, input: &[f32], time: usize, compute: &Compute) -> Result<Vec<f32>> {
        require_layout("residual input", input, self.first.input_channels, time)?;
        let (shortcut, shortcut_time) = self.shortcut.forward(input, time, 1, compute)?;
        let mut hidden = input.to_vec();
        elu_inplace(&mut hidden);
        let (mut hidden, first_time) = self.first.forward(&hidden, time, 1, compute)?;
        elu_inplace(&mut hidden);
        let (mut hidden, second_time) = self.second.forward(&hidden, first_time, 1, compute)?;
        if shortcut_time != time
            || first_time != time
            || second_time != time
            || hidden.len() != shortcut.len()
        {
            return Err(VokraError::InvalidArgument(format!(
                "funcodec: residual branch shape mismatch: shortcut={shortcut_time}, \
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
struct NormConv1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
}

impl NormConv1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            weight: tensor(
                file,
                &format!("{prefix}.conv.weight"),
                &[output_channels, input_channels, kernel],
            )?,
            bias: tensor(file, &format!("{prefix}.conv.bias"), &[output_channels])?,
            norm_weight: tensor(file, &format!("{prefix}.norm.weight"), &[output_channels])?,
            norm_bias: tensor(file, &format!("{prefix}.norm.bias"), &[output_channels])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        input_len: usize,
        stride: usize,
        compute: &Compute,
    ) -> Result<(Vec<f32>, usize)> {
        require_layout("conv1d input", input, self.input_channels, input_len)?;
        if input_len == 0 || self.kernel == 0 || stride == 0 {
            return Err(VokraError::InvalidArgument(
                "funcodec: conv1d requires non-empty input/kernel and stride > 0".to_owned(),
            ));
        }
        let padding_total = self
            .kernel
            .checked_sub(1)
            .and_then(|value| value.checked_sub(stride - 1))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "funcodec: conv1d kernel {} is smaller than stride {stride}",
                    self.kernel
                ))
            })?;
        let extra = extra_padding(input_len, self.kernel, stride, padding_total)?;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let padded = reflect_pad1d(
            input,
            self.input_channels,
            input_len,
            padding_left,
            padding_right + extra,
        )?;
        let padded_len = input_len + padding_total + extra;
        let output_len = (padded_len - self.kernel) / stride + 1;
        let mut convolved = vec![0.0f32; self.output_channels * output_len];
        compute.conv1d_f32(
            &padded,
            self.input_channels,
            padded_len,
            &self.weight,
            self.output_channels,
            self.kernel,
            Some(&self.bias),
            stride,
            0,
            &mut convolved,
        )?;
        let mut normalized = vec![0.0f32; convolved.len()];
        compute.group_norm_f32(
            &convolved,
            &mut normalized,
            self.output_channels,
            output_len,
            &self.norm_weight,
            &self.norm_bias,
            GROUP_NORM_EPS,
        )?;
        Ok((normalized, output_len))
    }
}

#[derive(Debug)]
struct NormConvTranspose1d {
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
}

impl NormConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            input_channels,
            output_channels,
            kernel,
            stride,
            weight: tensor(
                file,
                &format!("{prefix}.convtr.weight"),
                &[input_channels, output_channels, kernel],
            )?,
            bias: tensor(file, &format!("{prefix}.convtr.bias"), &[output_channels])?,
            norm_weight: tensor(file, &format!("{prefix}.norm.weight"), &[output_channels])?,
            norm_bias: tensor(file, &format!("{prefix}.norm.bias"), &[output_channels])?,
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
                "funcodec: invalid conv-transpose input_len={input_len}, kernel={}, stride={}",
                self.kernel, self.stride
            )));
        }
        let raw_len = (input_len - 1)
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.kernel))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "funcodec: conv-transpose output length overflow".to_owned(),
                )
            })?;
        let mut time_major = vec![0.0f32; input_len * self.input_channels];
        for time in 0..input_len {
            for channel in 0..self.input_channels {
                time_major[time * self.input_channels + channel] =
                    input[channel * input_len + time];
            }
        }
        let projected_width = self.output_channels * self.kernel;
        let mut projected = vec![0.0f32; input_len * projected_width];
        compute.gemm_f32(
            input_len,
            projected_width,
            self.input_channels,
            &time_major,
            &self.weight,
            None,
            &mut projected,
        )?;
        let mut raw = vec![0.0f32; self.output_channels * raw_len];
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

        // Official order is ConvTranspose -> GroupNorm -> asymmetric trim.
        let mut normalized = vec![0.0f32; raw.len()];
        compute.group_norm_f32(
            &raw,
            &mut normalized,
            self.output_channels,
            raw_len,
            &self.norm_weight,
            &self.norm_bias,
            GROUP_NORM_EPS,
        )?;
        let padding_total = self.kernel - self.stride;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let output_len = raw_len
            .checked_sub(padding_left + padding_right)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "funcodec: conv-transpose trim exceeds output".to_owned(),
                )
            })?;
        let mut output = vec![0.0f32; self.output_channels * output_len];
        for channel in 0..self.output_channels {
            output[channel * output_len..(channel + 1) * output_len].copy_from_slice(
                &normalized[channel * raw_len + padding_left
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
            let mut output = vec![0.0f32; self.dimension * time];
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

fn extra_padding(
    input_len: usize,
    kernel: usize,
    stride: usize,
    padding_total: usize,
) -> Result<usize> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "funcodec: convolution stride must be > 0".to_owned(),
        ));
    }
    let padded = input_len.checked_add(padding_total).ok_or_else(|| {
        VokraError::InvalidArgument("funcodec: convolution padded length overflow".to_owned())
    })?;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "funcodec: padded length {padded} is smaller than kernel {kernel}"
        )));
    }
    let frames = (padded - kernel).div_ceil(stride) + 1;
    let ideal = (frames - 1)
        .checked_mul(stride)
        .and_then(|value| value.checked_add(kernel))
        .and_then(|value| value.checked_sub(padding_total))
        .ok_or_else(|| {
            VokraError::InvalidArgument("funcodec: ideal convolution length overflow".to_owned())
        })?;
    ideal.checked_sub(input_len).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "funcodec: ideal convolution length {ideal} is smaller than input {input_len}"
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
            "funcodec: reflect padding requires non-empty input".to_owned(),
        ));
    }
    let max_padding = left.max(right);
    let extra = if length <= max_padding {
        max_padding - length + 1
    } else {
        0
    };
    let base_len = length.checked_add(extra).ok_or_else(|| {
        VokraError::InvalidArgument("funcodec: reflect padding base length overflow".to_owned())
    })?;
    let padded_len = base_len
        .checked_add(left)
        .and_then(|value| value.checked_add(right))
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "funcodec: reflect padding output length overflow".to_owned(),
            )
        })?;
    let output_len = padded_len - extra;
    let mut output = vec![0.0f32; channels * output_len];
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
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("funcodec: missing tensor `{name}`")))?;
    let expected = dimensions
        .iter()
        .map(|&dimension| dimension as u64)
        .collect::<Vec<_>>();
    if info.dimensions != expected {
        return Err(VokraError::ModelLoad(format!(
            "funcodec: tensor `{name}` shape {:?}, expected {expected:?}",
            info.dimensions
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("funcodec: tensor `{name}` decode failed: {error}"))
    })?;
    reject_non_finite(name, &values).map_err(|error| VokraError::ModelLoad(error.to_string()))?;
    Ok(values)
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "funcodec: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_layout(label: &str, values: &[f32], channels: usize, time: usize) -> Result<()> {
    let expected = channels.checked_mul(time).ok_or_else(|| {
        VokraError::InvalidArgument(format!("funcodec: {label} shape overflows usize"))
    })?;
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "funcodec: {label} has {} values, expected {channels}x{time}={expected}",
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
            "funcodec: {label} contains non-finite {value} at index {index}"
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
        assert_eq!(LSTM_DIMENSION, NUM_FILTERS << RATIOS.len());
        assert_eq!(SAMPLE_RATE as usize / FRAME_HOP, 50);
        assert_eq!(SPEC.tensor_count, 230);
        assert_eq!(FUNCODEC_DECODE_HOT_OPS.len(), 5);
    }

    #[test]
    fn complete_decode_hot_ops_are_cpu_and_metal_covered() {
        Compute::for_backend(BackendKind::Cpu, FUNCODEC_DECODE_HOT_OPS)
            .expect("CPU covers the complete FunCodec decoder");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, FUNCODEC_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("FunCodec decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn reflect_padding_matches_official_short_input_helper() {
        let normal = reflect_pad1d(&[1.0, 2.0, 3.0, 4.0], 1, 4, 2, 2).unwrap();
        assert_eq!(normal, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
        let short = reflect_pad1d(&[7.0], 1, 1, 3, 3).unwrap();
        assert_eq!(short, vec![0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn convolution_extra_padding_matches_official_formula() {
        assert_eq!(extra_padding(4, 7, 1, 6).unwrap(), 0);
        assert_eq!(extra_padding(5, 4, 2, 2).unwrap(), 1);
        assert!(extra_padding(5, 4, 0, 2).is_err());
    }

    #[test]
    fn structured_dropout_quantizer_bounds_are_explicit() {
        assert!((1..=NUM_CODEBOOKS).contains(&1));
        assert!((1..=NUM_CODEBOOKS).contains(&NUM_CODEBOOKS));
        assert!(!(1..=NUM_CODEBOOKS).contains(&0));
        assert!(!(1..=NUM_CODEBOOKS).contains(&(NUM_CODEBOOKS + 1)));
    }
}
