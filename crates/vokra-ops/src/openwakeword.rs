//! openWakeWord native inference primitives (SoTA plan KWS binder).
//!
//! openWakeWord (`dscripka/openWakeWord`, Apache-2.0 code) is a small
//! keyword-spotting family where each wake-word is a **tiny per-wake-word
//! DNN classifier** over a rolling sequence of **96-d speech embeddings**.
//! Release v0.5.1 uses an exact 512-point learned-DFT mel front end, a
//! 20-convolution frozen speech-embedding CNN, and (for the Alexa head)
//! `Flatten(16 × 96) → Linear(1536 → 128) → ReLU → Linear(128 → 128)
//! → ReLU → Linear(128 → 1) → Sigmoid`.
//!
//! # Scope of this module
//!
//! This module hosts all three numerical stages. ONNX is used only by the
//! offline reference/conversion tool; the runtime consumes flat GGUF tensors.
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape mismatch — wrong embedding width, empty hidden layer, out
//! bias length ≠ 1, weight length ≠ `out × in` — is a hard error
//! ([`vokra_core::VokraError::InvalidArgument`]) naming the offending
//! dimension. No silent zero-pad, no silent truncation, no silent
//! sigmoid-of-zero on missing weights.

use vokra_core::{Result, VokraError};

/// Exact v0.5.1 mel-front-end weights.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordMelspecWeights {
    /// Learned real DFT basis, `[257, 512]`.
    pub dft_real: Vec<f32>,
    /// Learned imaginary DFT basis, `[257, 512]`.
    pub dft_imag: Vec<f32>,
    /// Mel projection, `[257, 32]`.
    pub mel: Vec<f32>,
}

impl OpenwakewordMelspecWeights {
    /// Validates the fixed v0.5.1 topology.
    pub fn validate(&self) -> Result<()> {
        require_len("melspec.dft_real", &self.dft_real, 257 * 512)?;
        require_len("melspec.dft_imag", &self.dft_imag, 257 * 512)?;
        require_len("melspec.mel", &self.mel, 257 * 32)
    }
}

/// Runs the official v0.5.1 mel graph on PCM expressed in int16 amplitude
/// units (the upstream model casts PCM16 to f32 without normalising it).
/// Output is row-major `[frames, 32]`, including upstream's `/10 + 2`
/// compatibility transform.
pub fn openwakeword_melspectrogram(
    weights: &OpenwakewordMelspecWeights,
    pcm16: &[f32],
) -> Result<Vec<f32>> {
    weights.validate()?;
    if pcm16.len() < 512 {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword melspectrogram: {} samples, expected at least 512",
            pcm16.len()
        )));
    }
    if pcm16.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "openwakeword melspectrogram: PCM contains a non-finite sample".to_owned(),
        ));
    }

    let frames = (pcm16.len() - 512) / 160 + 1;
    let mut db = vec![0.0f32; frames * 32];
    let mut peak = f32::NEG_INFINITY;
    for frame in 0..frames {
        let input = &pcm16[frame * 160..frame * 160 + 512];
        let mut power = [0.0f32; 257];
        for (bin, cell) in power.iter_mut().enumerate() {
            let real_w = &weights.dft_real[bin * 512..(bin + 1) * 512];
            let imag_w = &weights.dft_imag[bin * 512..(bin + 1) * 512];
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for ((&sample, &rw), &iw) in input.iter().zip(real_w).zip(imag_w) {
                real += sample * rw;
                imag += sample * iw;
            }
            *cell = real * real + imag * imag;
        }
        for mel_bin in 0..32 {
            let mut value = 0.0f32;
            for (fft_bin, &p) in power.iter().enumerate() {
                value += p * weights.mel[fft_bin * 32 + mel_bin];
            }
            let value = 10.0 * value.max(1.0e-10).log10();
            db[frame * 32 + mel_bin] = value;
            peak = peak.max(value);
        }
    }
    let floor = peak - 80.0;
    for value in &mut db {
        *value = value.max(floor) / 10.0 + 2.0;
    }
    Ok(db)
}

/// One NCHW convolution in the frozen v0.5.1 embedding network.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordConv2dWeights {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// `[out_channels, in_channels, kernel_h, kernel_w]`.
    pub weight: Vec<f32>,
    /// `[out_channels]`; absent only on the final convolution.
    pub bias: Option<Vec<f32>>,
}

/// The exact 20-convolution Google speech-embedding bundle in v0.5.1.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordEmbeddingWeights {
    /// Convolutions in graph execution order.
    pub convs: Vec<OpenwakewordConv2dWeights>,
}

#[derive(Clone, Copy)]
struct ConvSpec {
    input: usize,
    output: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    pool: Option<(usize, usize)>,
}

const EMBEDDING_SPECS: [ConvSpec; 20] = [
    ConvSpec {
        input: 1,
        output: 24,
        kh: 3,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 24,
        output: 24,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 24,
        output: 24,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: Some((2, 2)),
    },
    ConvSpec {
        input: 24,
        output: 48,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 48,
        output: 48,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: None,
    },
    ConvSpec {
        input: 48,
        output: 48,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 48,
        output: 48,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: Some((1, 2)),
    },
    ConvSpec {
        input: 48,
        output: 72,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 72,
        output: 72,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: None,
    },
    ConvSpec {
        input: 72,
        output: 72,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 72,
        output: 72,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: Some((2, 2)),
    },
    ConvSpec {
        input: 72,
        output: 96,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: Some((1, 2)),
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 1,
        kw: 3,
        pad_h: 0,
        pad_w: 1,
        pool: None,
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: Some((2, 2)),
    },
    ConvSpec {
        input: 96,
        output: 96,
        kh: 3,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        pool: None,
    },
];

impl OpenwakewordEmbeddingWeights {
    /// Validates every layer against the fixed v0.5.1 embedding graph.
    pub fn validate(&self) -> Result<()> {
        if self.convs.len() != EMBEDDING_SPECS.len() {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword embedding: {} convs, expected 20",
                self.convs.len()
            )));
        }
        for (index, (conv, spec)) in self.convs.iter().zip(EMBEDDING_SPECS).enumerate() {
            if (
                conv.in_channels,
                conv.out_channels,
                conv.kernel_h,
                conv.kernel_w,
            ) != (spec.input, spec.output, spec.kh, spec.kw)
            {
                return Err(VokraError::InvalidArgument(format!(
                    "openwakeword embedding conv {index}: shape [{}, {}, {}, {}], expected [{}, {}, {}, {}]",
                    conv.out_channels,
                    conv.in_channels,
                    conv.kernel_h,
                    conv.kernel_w,
                    spec.output,
                    spec.input,
                    spec.kh,
                    spec.kw
                )));
            }
            require_len(
                &format!("embedding.conv{index}.weight"),
                &conv.weight,
                spec.output * spec.input * spec.kh * spec.kw,
            )?;
            match (&conv.bias, index == 19) {
                (None, true) => {}
                (Some(bias), false) => {
                    require_len(&format!("embedding.conv{index}.bias"), bias, spec.output)?;
                }
                _ => {
                    return Err(VokraError::InvalidArgument(format!(
                        "openwakeword embedding conv {index}: bias presence does not match v0.5.1 topology"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Runs one `[76, 32]` mel window through the exact v0.5.1 embedding CNN.
pub fn openwakeword_embedding_forward(
    weights: &OpenwakewordEmbeddingWeights,
    melspec: &[f32],
) -> Result<Vec<f32>> {
    weights.validate()?;
    require_len("embedding input", melspec, 76 * 32)?;
    let mut value = melspec.to_vec();
    let (mut height, mut width) = (76usize, 32usize);
    for (index, ((conv, spec), is_last)) in weights
        .convs
        .iter()
        .zip(EMBEDDING_SPECS)
        .zip((0..20).map(|i| i == 19))
        .enumerate()
    {
        let (next, out_h, out_w) = conv2d(&value, height, width, conv, spec.pad_h, spec.pad_w)?;
        value = next;
        height = out_h;
        width = out_w;
        if !is_last {
            for cell in &mut value {
                let leaky = if *cell >= 0.0 { *cell } else { *cell * 0.2 };
                *cell = leaky.max(-0.4);
            }
        }
        if let Some((pool_h, pool_w)) = spec.pool {
            let pooled = max_pool2d(&value, spec.output, height, width, pool_h, pool_w);
            value = pooled.0;
            height = pooled.1;
            width = pooled.2;
        }
        debug_assert_eq!(
            value.len(),
            spec.output * height * width,
            "embedding layer {index}"
        );
    }
    if height != 1 || width != 1 || value.len() != 96 {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword embedding topology ended at [96,{height},{width}] instead of [96,1,1]"
        )));
    }
    Ok(value)
}

/// One row-major affine layer in an official wake-word DNN head.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordDenseWeights {
    /// Input width.
    pub input_dim: usize,
    /// Output width.
    pub output_dim: usize,
    /// Row-major `[output_dim, input_dim]` weight.
    pub weight: Vec<f32>,
    /// Bias with `output_dim` elements.
    pub bias: Vec<f32>,
}

/// Variable-depth official DNN head over a rolling embedding window.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordDnnClassifierWeights {
    /// Number of rolling embeddings flattened into the first layer.
    pub input_frames: usize,
    /// Width of one embedding.
    pub embedding_dim: usize,
    /// Dense layers in graph execution order.
    pub layers: Vec<OpenwakewordDenseWeights>,
}

impl OpenwakewordDnnClassifierWeights {
    /// Validates the connected affine topology and binary output.
    pub fn validate(&self) -> Result<()> {
        if self.input_frames == 0 || self.embedding_dim == 0 || self.layers.is_empty() {
            return Err(VokraError::InvalidArgument(
                "openwakeword DNN: input_frames, embedding_dim, and layers must be non-zero"
                    .to_owned(),
            ));
        }
        let mut input = self.input_frames * self.embedding_dim;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.input_dim != input || layer.output_dim == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "openwakeword DNN layer {index}: {} -> {}, expected input {input}",
                    layer.input_dim, layer.output_dim
                )));
            }
            require_len(
                &format!("DNN layer {index} weight"),
                &layer.weight,
                layer.input_dim * layer.output_dim,
            )?;
            require_len(
                &format!("DNN layer {index} bias"),
                &layer.bias,
                layer.output_dim,
            )?;
            input = layer.output_dim;
        }
        if input != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword DNN: final output width is {input}, expected one binary logit"
            )));
        }
        Ok(())
    }
}

/// Runs the official ReLU-between-layers / final-sigmoid DNN.
pub fn openwakeword_dnn_classifier_forward(
    weights: &OpenwakewordDnnClassifierWeights,
    embeddings: &[f32],
) -> Result<f32> {
    weights.validate()?;
    require_len(
        "DNN embedding window",
        embeddings,
        weights.input_frames * weights.embedding_dim,
    )?;
    let mut value = embeddings.to_vec();
    for (index, layer) in weights.layers.iter().enumerate() {
        let mut output = vec![0.0f32; layer.output_dim];
        for (row, cell) in output.iter_mut().enumerate() {
            let mut acc = layer.bias[row];
            for (&weight, &input) in layer.weight
                [row * layer.input_dim..(row + 1) * layer.input_dim]
                .iter()
                .zip(&value)
            {
                acc += weight * input;
            }
            *cell = acc;
        }
        if index + 1 != weights.layers.len() {
            for cell in &mut output {
                *cell = cell.max(0.0);
            }
        }
        value = output;
    }
    Ok(0.5 * (0.5 * value[0]).tanh() + 0.5)
}

fn require_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword {name}: {} elements, expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn conv2d(
    input: &[f32],
    height: usize,
    width: usize,
    weights: &OpenwakewordConv2dWeights,
    pad_h: usize,
    pad_w: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    require_len("conv input", input, weights.in_channels * height * width)?;
    let padded_height = height.checked_add(2 * pad_h).ok_or_else(|| {
        VokraError::InvalidArgument("openwakeword conv: padded height overflow".to_owned())
    })?;
    let padded_width = width.checked_add(2 * pad_w).ok_or_else(|| {
        VokraError::InvalidArgument("openwakeword conv: padded width overflow".to_owned())
    })?;
    if weights.kernel_h > padded_height || weights.kernel_w > padded_width {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword conv: kernel [{}, {}] exceeds padded input [{padded_height}, {padded_width}]",
            weights.kernel_h, weights.kernel_w
        )));
    }
    let out_h = padded_height - weights.kernel_h + 1;
    let out_w = padded_width - weights.kernel_w + 1;
    let mut output = vec![0.0f32; weights.out_channels * out_h * out_w];
    for out_channel in 0..weights.out_channels {
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                let mut acc = weights.bias.as_ref().map_or(0.0, |bias| bias[out_channel]);
                for in_channel in 0..weights.in_channels {
                    for kernel_y in 0..weights.kernel_h {
                        let in_y = out_y + kernel_y;
                        if in_y < pad_h || in_y - pad_h >= height {
                            continue;
                        }
                        for kernel_x in 0..weights.kernel_w {
                            let in_x = out_x + kernel_x;
                            if in_x < pad_w || in_x - pad_w >= width {
                                continue;
                            }
                            let input_index =
                                (in_channel * height + in_y - pad_h) * width + in_x - pad_w;
                            let weight_index = (((out_channel * weights.in_channels + in_channel)
                                * weights.kernel_h
                                + kernel_y)
                                * weights.kernel_w)
                                + kernel_x;
                            acc += input[input_index] * weights.weight[weight_index];
                        }
                    }
                }
                output[(out_channel * out_h + out_y) * out_w + out_x] = acc;
            }
        }
    }
    Ok((output, out_h, out_w))
}

fn max_pool2d(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = height / kernel_h;
    let out_w = width / kernel_w;
    let mut output = vec![f32::NEG_INFINITY; channels * out_h * out_w];
    for channel in 0..channels {
        for out_y in 0..out_h {
            for out_x in 0..out_w {
                let mut maximum = f32::NEG_INFINITY;
                for kernel_y in 0..kernel_h {
                    for kernel_x in 0..kernel_w {
                        maximum = maximum.max(
                            input[(channel * height + out_y * kernel_h + kernel_y) * width
                                + out_x * kernel_w
                                + kernel_x],
                        );
                    }
                }
                output[(channel * out_h + out_y) * out_w + out_x] = maximum;
            }
        }
    }
    (output, out_h, out_w)
}

/// One per-wake-word MLP classifier weight bundle (`Linear` → ReLU →
/// `Linear` → Sigmoid, where the final Sigmoid is applied by the
/// [`openwakeword_classifier_forward`] caller).
///
/// Every field is required and must be self-consistent: the runtime
/// binder (`vokra_models::kws::openwakeword`) validates the shapes at
/// GGUF load time via [`Self::validate`], and
/// [`openwakeword_classifier_forward`] re-validates at forward time so a
/// hand-built bundle in a downstream crate cannot silently misforward.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordClassifierWeights {
    /// Embedding dimensionality (`96` in the reference release).
    pub embedding_dim: usize,
    /// Hidden layer width (per-wake-word; upstream defaults vary from
    /// `128` to `256`).
    pub hidden_dim: usize,
    /// First linear layer weight, row-major `[hidden_dim, embedding_dim]`.
    pub linear1_weight: Vec<f32>,
    /// First linear layer bias, `[hidden_dim]`.
    pub linear1_bias: Vec<f32>,
    /// Output linear layer weight, row-major `[1, hidden_dim]` (each
    /// wake-word is a binary classifier).
    pub linear2_weight: Vec<f32>,
    /// Output linear layer bias, `[1]`.
    pub linear2_bias: Vec<f32>,
}

impl OpenwakewordClassifierWeights {
    /// Validates the shape contract loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any dimension mismatch, empty
    /// hidden layer, or an out layer that is not a single-class binary
    /// classifier.
    pub fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "openwakeword classifier: embedding_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "openwakeword classifier: hidden_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        let l1_expected = self.hidden_dim * self.embedding_dim;
        if self.linear1_weight.len() != l1_expected {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear1_weight has {} elements, expected {} \
                 (hidden_dim={} * embedding_dim={})",
                self.linear1_weight.len(),
                l1_expected,
                self.hidden_dim,
                self.embedding_dim
            )));
        }
        if self.linear1_bias.len() != self.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear1_bias has {} elements, expected hidden_dim={}",
                self.linear1_bias.len(),
                self.hidden_dim
            )));
        }
        let l2_expected = self.hidden_dim;
        if self.linear2_weight.len() != l2_expected {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear2_weight has {} elements, expected {} \
                 (1 output class * hidden_dim={})",
                self.linear2_weight.len(),
                l2_expected,
                self.hidden_dim
            )));
        }
        if self.linear2_bias.len() != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear2_bias has {} elements, expected 1 (single \
                 binary output class per wake-word)",
                self.linear2_bias.len()
            )));
        }
        Ok(())
    }
}

/// Runs one classifier forward pass on a single embedding vector,
/// returning the sigmoid probability in `[0, 1]`.
///
/// Pipeline: `y = sigmoid(linear2_bias + linear2_weight ⋅ relu(linear1_bias
/// + linear1_weight ⋅ embedding))`.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if the embedding length does not
///   match `weights.embedding_dim`, or if [`OpenwakewordClassifierWeights::validate`]
///   rejects the bundle.
pub fn openwakeword_classifier_forward(
    weights: &OpenwakewordClassifierWeights,
    embedding: &[f32],
) -> Result<f32> {
    weights.validate()?;
    if embedding.len() != weights.embedding_dim {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword classifier: embedding has {} elements, expected embedding_dim={}",
            embedding.len(),
            weights.embedding_dim
        )));
    }

    // Layer 1: hidden = ReLU(linear1_bias + linear1_weight @ embedding).
    let hidden_dim = weights.hidden_dim;
    let embedding_dim = weights.embedding_dim;
    let mut hidden = vec![0.0f32; hidden_dim];
    for (h, cell) in hidden.iter_mut().enumerate() {
        let row = &weights.linear1_weight[h * embedding_dim..(h + 1) * embedding_dim];
        let mut acc = weights.linear1_bias[h];
        for (w, x) in row.iter().zip(embedding.iter()) {
            acc += w * x;
        }
        // ReLU.
        *cell = if acc > 0.0 { acc } else { 0.0 };
    }

    // Layer 2: logit = linear2_bias + linear2_weight @ hidden.
    let mut logit = weights.linear2_bias[0];
    for (w, h) in weights.linear2_weight.iter().zip(hidden.iter()) {
        logit += w * h;
    }

    // Sigmoid — numerically-stable form via tanh (avoids overflow in
    // `exp(-x)` for large-magnitude logits).
    Ok(0.5 * (0.5 * logit).tanh() + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canonical tiny bundle used across the shape / forward tests
    /// (embedding_dim=4, hidden_dim=3). All weights below produce a
    /// mathematically-tractable forward the tests can assert on directly.
    fn tiny_bundle() -> OpenwakewordClassifierWeights {
        OpenwakewordClassifierWeights {
            embedding_dim: 4,
            hidden_dim: 3,
            // Identity-ish layer 1: row 0 sums (positive inputs); rows
            // 1/2 subtract them (so ReLU zeros them for positive input).
            #[rustfmt::skip]
            linear1_weight: vec![
                 1.0,  1.0,  1.0,  1.0, // row 0: acc = sum(x)
                -1.0, -1.0, -1.0, -1.0, // row 1: acc = -sum(x) → 0 after ReLU
                -1.0, -1.0, -1.0, -1.0, // row 2: same
            ],
            linear1_bias: vec![0.0, 0.0, 0.0],
            // Layer 2 picks up row 0 only.
            linear2_weight: vec![1.0, 0.0, 0.0],
            linear2_bias: vec![0.0],
        }
    }

    #[test]
    fn native_melspectrogram_uses_learned_dft_and_mel_weights() {
        let mut dft_real = vec![0.0; 257 * 512];
        dft_real[..512].fill(1.0);
        let mut mel = vec![0.0; 257 * 32];
        mel[0] = 1.0;
        let weights = OpenwakewordMelspecWeights {
            dft_real,
            dft_imag: vec![0.0; 257 * 512],
            mel,
        };
        let output = openwakeword_melspectrogram(&weights, &[1.0; 512]).unwrap();
        assert_eq!(output.len(), 32);
        let expected_peak = (10.0 * (512.0f32 * 512.0).log10()) / 10.0 + 2.0;
        assert!((output[0] - expected_peak).abs() < 1.0e-5);
        assert!(output[1] < output[0] - 7.9);
    }

    #[test]
    fn native_dnn_flattens_embedding_window_and_runs_all_layers() {
        let weights = OpenwakewordDnnClassifierWeights {
            input_frames: 2,
            embedding_dim: 2,
            layers: vec![
                OpenwakewordDenseWeights {
                    input_dim: 4,
                    output_dim: 2,
                    weight: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    bias: vec![0.0, 0.0],
                },
                OpenwakewordDenseWeights {
                    input_dim: 2,
                    output_dim: 1,
                    weight: vec![1.0, 2.0],
                    bias: vec![0.0],
                },
            ],
        };
        let probability =
            openwakeword_dnn_classifier_forward(&weights, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let expected = 1.0 / (1.0 + (-5.0f32).exp());
        assert!((probability - expected).abs() < 1.0e-6);
    }

    #[test]
    fn native_embedding_rejects_incomplete_topology() {
        let error = OpenwakewordEmbeddingWeights { convs: Vec::new() }
            .validate()
            .expect_err("all 20 graph convolutions are required");
        assert!(error.to_string().contains("expected 20"));
    }

    #[test]
    fn forward_returns_sigmoid_of_positive_sum() {
        let w = tiny_bundle();
        // x = [1, 1, 1, 1] → sum = 4 → hidden[0] = 4, hidden[1..] = 0 →
        // logit = 4 → probability = sigmoid(4) ≈ 0.9820.
        let p = openwakeword_classifier_forward(&w, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let expected = 1.0f32 / (1.0 + (-4.0f32).exp());
        assert!(
            (p - expected).abs() < 1e-6,
            "expected sigmoid(4) = {expected}, got {p}"
        );
    }

    #[test]
    fn forward_probability_stays_in_unit_interval() {
        let w = tiny_bundle();
        // Extreme positive and negative inputs to exercise numeric
        // stability of the sigmoid.
        for magnitude in [-1000.0f32, -10.0, 0.0, 10.0, 1000.0] {
            let x = vec![magnitude; w.embedding_dim];
            let p = openwakeword_classifier_forward(&w, &x).unwrap();
            assert!(
                p.is_finite(),
                "probability must be finite (magnitude={magnitude}, got {p})"
            );
            assert!(
                (0.0..=1.0).contains(&p),
                "sigmoid must live in [0, 1] (magnitude={magnitude}, got {p})"
            );
        }
    }

    #[test]
    fn forward_rejects_wrong_embedding_length_loudly() {
        let w = tiny_bundle();
        let err = openwakeword_classifier_forward(&w, &[1.0, 1.0])
            .expect_err("embedding_dim mismatch must be a loud error (FR-EX-08)");
        let msg = err.to_string();
        assert!(
            msg.contains("embedding"),
            "error message must mention embedding: {msg}"
        );
        assert!(
            msg.contains("2"),
            "error message must mention actual length 2: {msg}"
        );
        assert!(
            msg.contains("4"),
            "error message must mention expected 4: {msg}"
        );
    }

    #[test]
    fn validate_rejects_zero_embedding_dim() {
        let mut w = tiny_bundle();
        w.embedding_dim = 0;
        assert!(matches!(w.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_rejects_zero_hidden_dim() {
        let mut w = tiny_bundle();
        w.hidden_dim = 0;
        assert!(matches!(w.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_rejects_wrong_linear1_shape() {
        let mut w = tiny_bundle();
        w.linear1_weight.pop();
        let err = w
            .validate()
            .expect_err("shape mismatch must be a loud error");
        assert!(err.to_string().contains("linear1_weight"));
    }

    #[test]
    fn validate_rejects_multi_class_out_layer() {
        // openwakeword's per-wake-word classifier is binary (single output
        // class). A `[2]` out bias is an architectural mismatch that must
        // be refused loudly rather than silently returning the first row.
        let mut w = tiny_bundle();
        w.linear2_bias = vec![0.0, 0.0];
        let err = w
            .validate()
            .expect_err("multi-class out layer must be a loud error");
        assert!(err.to_string().contains("linear2_bias"));
    }

    #[test]
    fn forward_negative_sum_relu_masks_to_bias_only() {
        let w = tiny_bundle();
        // x = [-1, -1, -1, -1] → sum = -4 → hidden[0] = 0 after ReLU →
        // logit = 0 → probability = sigmoid(0) = 0.5.
        let p = openwakeword_classifier_forward(&w, &[-1.0, -1.0, -1.0, -1.0]).unwrap();
        assert!(
            (p - 0.5).abs() < 1e-6,
            "ReLU-masked forward must return sigmoid(0) = 0.5, got {p}"
        );
    }
}
