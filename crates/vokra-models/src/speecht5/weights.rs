//! Exact 393-tensor SpeechT5 TTS weight binder.

use std::collections::BTreeSet;

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::strict_checkpoint::load_tensor;

use super::{
    DECODER_ATTENTION_HEADS, DECODER_FFN_DIM, DECODER_LAYERS, ENCODER_ATTENTION_HEADS,
    ENCODER_FFN_DIM, ENCODER_LAYERS, ENCODER_MAX_RELATIVE_POSITION, HIDDEN_SIZE,
    MAX_SPEECH_POSITIONS, MAX_TEXT_POSITIONS, NUM_MEL_BINS, REDUCTION_FACTOR,
    SPEAKER_EMBEDDING_DIM, SPEECH_DECODER_POSTNET_KERNEL, SPEECH_DECODER_POSTNET_LAYERS,
    SPEECH_DECODER_POSTNET_UNITS, SPEECH_DECODER_PRENET_LAYERS, SPEECH_DECODER_PRENET_UNITS,
    TENSOR_COUNT, VOCAB_SIZE,
};

const LABEL: &str = "SpeechT5-TTS";

#[derive(Debug, Clone)]
pub(super) struct Linear {
    pub(super) weight: Vec<f32>, // [in, out]
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
}

impl Linear {
    fn load(
        loader: &mut TensorLoader<'_>,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let source = loader.tensor(&format!("{prefix}.weight"), &[output, input])?;
        let weight = transpose_out_in(&source, input, output);
        let bias = loader.tensor(&format!("{prefix}.bias"), &[output])?;
        Ok(Self {
            weight,
            bias,
            input,
            output,
        })
    }

    pub(super) fn forward(
        &self,
        compute: &Compute,
        values: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        if rows == 0 || values.len() != rows * self.input {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: linear input length {} does not match rows={rows}, width={}",
                values.len(),
                self.input
            )));
        }
        let mut output = vec![0.0; rows * self.output];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            values,
            &self.weight,
            Some(&self.bias),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub(super) struct LayerNorm {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

impl LayerNorm {
    fn load(loader: &mut TensorLoader<'_>, prefix: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: loader.tensor(&format!("{prefix}.weight"), &[width])?,
            bias: loader.tensor(&format!("{prefix}.bias"), &[width])?,
        })
    }

    pub(super) fn forward(
        &self,
        compute: &Compute,
        values: &[f32],
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if rows == 0
            || values.len() != rows * width
            || self.weight.len() != width
            || self.bias.len() != width
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: layer-norm shape mismatch"
            )));
        }
        let mut output = vec![0.0; values.len()];
        compute.layer_norm_f32(
            values,
            &mut output,
            rows,
            width,
            &self.weight,
            &self.bias,
            eps,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    pub(super) q: Linear,
    pub(super) k: Linear,
    pub(super) v: Linear,
    pub(super) out: Linear,
    pub(super) heads: usize,
}

impl Attention {
    fn load(loader: &mut TensorLoader<'_>, prefix: &str, heads: usize) -> Result<Self> {
        Ok(Self {
            q: Linear::load(
                loader,
                &format!("{prefix}.q_proj"),
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            )?,
            k: Linear::load(
                loader,
                &format!("{prefix}.k_proj"),
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            )?,
            v: Linear::load(
                loader,
                &format!("{prefix}.v_proj"),
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            )?,
            out: Linear::load(
                loader,
                &format!("{prefix}.out_proj"),
                HIDDEN_SIZE,
                HIDDEN_SIZE,
            )?,
            heads,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    pub(super) intermediate: Linear,
    pub(super) output: Linear,
}

impl FeedForward {
    fn load(loader: &mut TensorLoader<'_>, prefix: &str, inner: usize) -> Result<Self> {
        Ok(Self {
            intermediate: Linear::load(
                loader,
                &format!("{prefix}.intermediate_dense"),
                HIDDEN_SIZE,
                inner,
            )?,
            output: Linear::load(
                loader,
                &format!("{prefix}.output_dense"),
                inner,
                HIDDEN_SIZE,
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct EncoderLayer {
    pub(super) attention: Attention,
    pub(super) attention_norm: LayerNorm,
    pub(super) feed_forward: FeedForward,
    pub(super) final_norm: LayerNorm,
}

#[derive(Debug, Clone)]
pub(super) struct EncoderWeights {
    pub(super) token_embedding: Vec<f32>,
    pub(super) position_alpha: f32,
    pub(super) positions: Vec<f32>,
    pub(super) relative_positions: Vec<f32>,
    pub(super) initial_norm: LayerNorm,
    pub(super) layers: Vec<EncoderLayer>,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    pub(super) self_attention: Attention,
    pub(super) self_attention_norm: LayerNorm,
    pub(super) cross_attention: Attention,
    pub(super) cross_attention_norm: LayerNorm,
    pub(super) feed_forward: FeedForward,
    pub(super) final_norm: LayerNorm,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderWeights {
    pub(super) prenet: Vec<Linear>,
    pub(super) final_layer: Linear,
    pub(super) position_alpha: f32,
    pub(super) positions: Vec<f32>,
    pub(super) speaker_projection: Linear,
    pub(super) layers: Vec<DecoderLayer>,
}

#[derive(Debug, Clone)]
pub(super) struct BatchNormConv {
    pub(super) conv_weight: Vec<f32>,
    pub(super) input_channels: usize,
    pub(super) output_channels: usize,
    pub(super) norm_weight: Vec<f32>,
    pub(super) norm_bias: Vec<f32>,
    pub(super) running_mean: Vec<f32>,
    pub(super) running_var: Vec<f32>,
    pub(super) activation: bool,
}

#[derive(Debug, Clone)]
pub(super) struct PostnetWeights {
    pub(super) feat_out: Linear,
    pub(super) prob_out: Linear,
    pub(super) layers: Vec<BatchNormConv>,
}

#[derive(Debug, Clone)]
pub(super) struct SpeechT5Weights {
    pub(super) encoder: EncoderWeights,
    pub(super) decoder: DecoderWeights,
    pub(super) postnet: PostnetWeights,
}

impl SpeechT5Weights {
    pub(super) fn load(file: &GgufFile) -> Result<Self> {
        let mut loader = TensorLoader::new(file);

        let encoder = EncoderWeights {
            token_embedding: loader.tensor(
                "speecht5.encoder.prenet.embed_tokens.weight",
                &[VOCAB_SIZE, HIDDEN_SIZE],
            )?,
            position_alpha: loader.scalar("speecht5.encoder.prenet.encode_positions.alpha")?,
            positions: loader.tensor(
                "speecht5.encoder.prenet.encode_positions.pe",
                &[1, MAX_TEXT_POSITIONS, HIDDEN_SIZE],
            )?,
            relative_positions: loader.tensor(
                "speecht5.encoder.wrapped_encoder.embed_positions.pe_k.weight",
                &[
                    2 * ENCODER_MAX_RELATIVE_POSITION,
                    HIDDEN_SIZE / ENCODER_ATTENTION_HEADS,
                ],
            )?,
            initial_norm: LayerNorm::load(
                &mut loader,
                "speecht5.encoder.wrapped_encoder.layer_norm",
                HIDDEN_SIZE,
            )?,
            layers: (0..ENCODER_LAYERS)
                .map(|layer| {
                    let prefix = format!("speecht5.encoder.wrapped_encoder.layers.{layer}");
                    Ok(EncoderLayer {
                        attention: Attention::load(
                            &mut loader,
                            &format!("{prefix}.attention"),
                            ENCODER_ATTENTION_HEADS,
                        )?,
                        attention_norm: LayerNorm::load(
                            &mut loader,
                            &format!("{prefix}.layer_norm"),
                            HIDDEN_SIZE,
                        )?,
                        feed_forward: FeedForward::load(
                            &mut loader,
                            &format!("{prefix}.feed_forward"),
                            ENCODER_FFN_DIM,
                        )?,
                        final_norm: LayerNorm::load(
                            &mut loader,
                            &format!("{prefix}.final_layer_norm"),
                            HIDDEN_SIZE,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };

        let decoder = DecoderWeights {
            prenet: (0..SPEECH_DECODER_PRENET_LAYERS)
                .map(|layer| {
                    Linear::load(
                        &mut loader,
                        &format!("speecht5.decoder.prenet.layers.{layer}"),
                        if layer == 0 {
                            NUM_MEL_BINS
                        } else {
                            SPEECH_DECODER_PRENET_UNITS
                        },
                        SPEECH_DECODER_PRENET_UNITS,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_layer: Linear::load(
                &mut loader,
                "speecht5.decoder.prenet.final_layer",
                SPEECH_DECODER_PRENET_UNITS,
                HIDDEN_SIZE,
            )?,
            position_alpha: loader.scalar("speecht5.decoder.prenet.encode_positions.alpha")?,
            positions: loader.tensor(
                "speecht5.decoder.prenet.encode_positions.pe",
                &[1, MAX_SPEECH_POSITIONS, HIDDEN_SIZE],
            )?,
            speaker_projection: Linear::load(
                &mut loader,
                "speecht5.decoder.prenet.speaker_embeds_layer",
                HIDDEN_SIZE + SPEAKER_EMBEDDING_DIM,
                HIDDEN_SIZE,
            )?,
            layers: (0..DECODER_LAYERS)
                .map(|layer| {
                    let prefix = format!("speecht5.decoder.wrapped_decoder.layers.{layer}");
                    Ok(DecoderLayer {
                        self_attention: Attention::load(
                            &mut loader,
                            &format!("{prefix}.self_attn"),
                            DECODER_ATTENTION_HEADS,
                        )?,
                        self_attention_norm: LayerNorm::load(
                            &mut loader,
                            &format!("{prefix}.self_attn_layer_norm"),
                            HIDDEN_SIZE,
                        )?,
                        cross_attention: Attention::load(
                            &mut loader,
                            &format!("{prefix}.encoder_attn"),
                            DECODER_ATTENTION_HEADS,
                        )?,
                        cross_attention_norm: LayerNorm::load(
                            &mut loader,
                            &format!("{prefix}.encoder_attn_layer_norm"),
                            HIDDEN_SIZE,
                        )?,
                        feed_forward: FeedForward::load(
                            &mut loader,
                            &format!("{prefix}.feed_forward"),
                            DECODER_FFN_DIM,
                        )?,
                        final_norm: LayerNorm::load(
                            &mut loader,
                            &format!("{prefix}.final_layer_norm"),
                            HIDDEN_SIZE,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };

        let postnet = PostnetWeights {
            feat_out: Linear::load(
                &mut loader,
                "speech_decoder_postnet.feat_out",
                HIDDEN_SIZE,
                NUM_MEL_BINS * REDUCTION_FACTOR,
            )?,
            prob_out: Linear::load(
                &mut loader,
                "speech_decoder_postnet.prob_out",
                HIDDEN_SIZE,
                REDUCTION_FACTOR,
            )?,
            layers: (0..SPEECH_DECODER_POSTNET_LAYERS)
                .map(|layer| {
                    let prefix = format!("speech_decoder_postnet.layers.{layer}");
                    let input_channels = if layer == 0 {
                        NUM_MEL_BINS
                    } else {
                        SPEECH_DECODER_POSTNET_UNITS
                    };
                    let output_channels = if layer + 1 == SPEECH_DECODER_POSTNET_LAYERS {
                        NUM_MEL_BINS
                    } else {
                        SPEECH_DECODER_POSTNET_UNITS
                    };
                    Ok(BatchNormConv {
                        conv_weight: loader.tensor(
                            &format!("{prefix}.conv.weight"),
                            &[
                                output_channels,
                                input_channels,
                                SPEECH_DECODER_POSTNET_KERNEL,
                            ],
                        )?,
                        input_channels,
                        output_channels,
                        norm_weight: loader
                            .tensor(&format!("{prefix}.batch_norm.weight"), &[output_channels])?,
                        norm_bias: loader
                            .tensor(&format!("{prefix}.batch_norm.bias"), &[output_channels])?,
                        running_mean: loader.tensor(
                            &format!("{prefix}.batch_norm.running_mean"),
                            &[output_channels],
                        )?,
                        running_var: loader.tensor(
                            &format!("{prefix}.batch_norm.running_var"),
                            &[output_channels],
                        )?,
                        activation: layer + 1 != SPEECH_DECODER_POSTNET_LAYERS,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };

        loader.finish()?;
        Ok(Self {
            encoder,
            decoder,
            postnet,
        })
    }
}

struct TensorLoader<'a> {
    file: &'a GgufFile,
    consumed: BTreeSet<String>,
}

impl<'a> TensorLoader<'a> {
    fn new(file: &'a GgufFile) -> Self {
        Self {
            file,
            consumed: BTreeSet::new(),
        }
    }

    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        if !self.consumed.insert(name.to_owned()) {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: tensor `{name}` was bound more than once"
            )));
        }
        load_tensor(self.file, LABEL, name, shape)
    }

    fn scalar(&mut self, name: &str) -> Result<f32> {
        let values = self.tensor(name, &[])?;
        values.first().copied().ok_or_else(|| {
            VokraError::ModelLoad(format!("{LABEL}: scalar tensor `{name}` decoded empty"))
        })
    }

    fn finish(self) -> Result<()> {
        if self.consumed.len() != TENSOR_COUNT {
            let missing = self
                .file
                .tensors()
                .iter()
                .map(|tensor| tensor.name.as_str())
                .find(|name| !self.consumed.contains(*name));
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: runtime consumed {} of {TENSOR_COUNT} tensors; first unconsumed={missing:?}",
                self.consumed.len()
            )));
        }
        Ok(())
    }
}

fn transpose_out_in(source: &[f32], input: usize, output: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; source.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = source[out * input + inner];
        }
    }
    transposed
}
