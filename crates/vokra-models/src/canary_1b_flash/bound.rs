//! Strict released-weight binding and CPU/Metal shared forward.

use std::cell::RefCell;
use std::collections::BTreeSet;

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::canary::CanaryConfig;
use crate::compute::Compute;
use crate::parakeet::{
    FastConformerConvNorm, ParakeetBoundEncoderBlock, ParakeetBoundNorm, ParakeetBoundSubsampling,
    ParakeetEncoderConfig, conformer_block_forward, parakeet_logmel, relative_positions,
    subsampling_forward, transpose_out_in,
};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};

/// Immutable release axes that differ between Canary Transformer-AED models.
///
/// The released Flash and v2 checkpoints share every executable tensor-name
/// pattern and forward operation. Decoder depth, vocabulary width and strict
/// checkpoint identity are data, not separate implementations.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CanaryAedReleaseSpec {
    strict: StrictCheckpointSpec,
    sample_rate: u32,
    vocab_size: usize,
    eos_token_id: u32,
}

impl CanaryAedReleaseSpec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        label: &'static str,
        arch: &'static str,
        model_name: &'static str,
        tensor_count: usize,
        manifest_sha256: [u8; 32],
        sample_rate: u32,
        vocab_size: usize,
        eos_token_id: u32,
    ) -> Self {
        Self {
            strict: StrictCheckpointSpec {
                label,
                arch,
                model_name,
                model_name_alias: None,
                tensor_count,
                manifest_sha256,
            },
            sample_rate,
            vocab_size,
            eos_token_id,
        }
    }

    const fn label(self) -> &'static str {
        self.strict.label
    }
}

#[derive(Debug, Clone)]
struct SelfAttentionWeights {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

#[derive(Debug, Clone)]
struct CrossAttentionWeights {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w_t: Vec<f32>,
    k_b: Vec<f32>,
    v_w_t: Vec<f32>,
    v_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

#[derive(Debug, Clone)]
struct DecoderBlockWeights {
    norm_self: ParakeetBoundNorm,
    self_attention: SelfAttentionWeights,
    norm_cross: ParakeetBoundNorm,
    cross_attention: CrossAttentionWeights,
    norm_ffn: ParakeetBoundNorm,
    ffn_in_w: Vec<f32>,
    ffn_in_b: Vec<f32>,
    ffn_out_w: Vec<f32>,
    ffn_out_b: Vec<f32>,
}

/// Fully authenticated released float checkpoint.
#[derive(Debug, Clone)]
pub(crate) struct CanaryBoundWeights {
    release: CanaryAedReleaseSpec,
    checkpoint: StrictCheckpoint,
    encoder_config: ParakeetEncoderConfig,
    subsampling: ParakeetBoundSubsampling,
    encoder: Vec<ParakeetBoundEncoderBlock>,
    token_embedding: Vec<f32>,
    position_embedding: Vec<f32>,
    embedding_norm: ParakeetBoundNorm,
    decoder: Vec<DecoderBlockWeights>,
    decoder_final_norm: ParakeetBoundNorm,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
}

impl CanaryBoundWeights {
    pub(crate) fn verify_manifest(file: &GgufFile, release: CanaryAedReleaseSpec) -> Result<()> {
        StrictCheckpoint::bind(file, release.strict).map(|_| ())
    }

    pub(crate) fn from_gguf(
        file: &GgufFile,
        config: &CanaryConfig,
        release: CanaryAedReleaseSpec,
    ) -> Result<Self> {
        config.validate_for_forward()?;
        let checkpoint = StrictCheckpoint::bind(file, release.strict)?;
        let label = release.label();
        let enc = encoder_config(config);
        let d = enc.d_model;
        let ffn = enc.ffn_dim;
        let channels = enc.subsampling_conv_channels;
        let kernel = enc.subsampling_conv_kernel_size;
        let consumed = RefCell::new(BTreeSet::new());
        let tensor = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
            let value = load_tensor(file, label, name, shape)?;
            consumed.borrow_mut().insert(name.to_owned());
            Ok(value)
        };

        let subsampling = ParakeetBoundSubsampling {
            conv0_w: tensor(
                "encoder.pre_encode.conv.0.weight",
                &[channels, 1, kernel, kernel],
            )?,
            conv0_b: tensor("encoder.pre_encode.conv.0.bias", &[channels])?,
            depthwise_w: [
                tensor(
                    "encoder.pre_encode.conv.2.weight",
                    &[channels, 1, kernel, kernel],
                )?,
                tensor(
                    "encoder.pre_encode.conv.5.weight",
                    &[channels, 1, kernel, kernel],
                )?,
            ],
            depthwise_b: [
                tensor("encoder.pre_encode.conv.2.bias", &[channels])?,
                tensor("encoder.pre_encode.conv.5.bias", &[channels])?,
            ],
            pointwise_w_t: [
                transpose_out_in(
                    tensor(
                        "encoder.pre_encode.conv.3.weight",
                        &[channels, channels, 1, 1],
                    )?,
                    channels,
                    channels,
                ),
                transpose_out_in(
                    tensor(
                        "encoder.pre_encode.conv.6.weight",
                        &[channels, channels, 1, 1],
                    )?,
                    channels,
                    channels,
                ),
            ],
            pointwise_b: [
                tensor("encoder.pre_encode.conv.3.bias", &[channels])?,
                tensor("encoder.pre_encode.conv.6.bias", &[channels])?,
            ],
            linear_w_t: transpose_out_in(
                tensor(
                    "encoder.pre_encode.out.weight",
                    &[d, channels * (enc.in_dim / enc.subsampling_factor)],
                )?,
                d,
                channels * (enc.in_dim / enc.subsampling_factor),
            ),
            linear_b: tensor("encoder.pre_encode.out.bias", &[d])?,
        };

        let mut encoder = Vec::with_capacity(enc.n_layer);
        for layer in 0..enc.n_layer {
            let prefix = format!("encoder.layers.{layer}");
            let norm = |name: &str| -> Result<ParakeetBoundNorm> {
                Ok(ParakeetBoundNorm {
                    weight: tensor(&format!("{prefix}.{name}.weight"), &[d])?,
                    bias: tensor(&format!("{prefix}.{name}.bias"), &[d])?,
                })
            };
            let ff_weight = |branch: &str, linear: usize, output: usize, input: usize| {
                tensor(
                    &format!("{prefix}.{branch}.linear{linear}.weight"),
                    &[output, input],
                )
                .map(|weight| transpose_out_in(weight, output, input))
            };
            let ff_bias = |branch: &str, linear: usize, output: usize| {
                tensor(&format!("{prefix}.{branch}.linear{linear}.bias"), &[output]).map(Some)
            };
            let attention_weight = |name: &str| {
                tensor(&format!("{prefix}.self_attn.{name}.weight"), &[d, d])
                    .map(|weight| transpose_out_in(weight, d, d))
            };
            let attention_bias =
                |name: &str| tensor(&format!("{prefix}.self_attn.{name}.bias"), &[d]).map(Some);
            encoder.push(ParakeetBoundEncoderBlock {
                ff1_w1_t: ff_weight("feed_forward1", 1, ffn, d)?,
                ff1_b1: ff_bias("feed_forward1", 1, ffn)?,
                ff1_w2_t: ff_weight("feed_forward1", 2, d, ffn)?,
                ff1_b2: ff_bias("feed_forward1", 2, d)?,
                ff2_w1_t: ff_weight("feed_forward2", 1, ffn, d)?,
                ff2_b1: ff_bias("feed_forward2", 1, ffn)?,
                ff2_w2_t: ff_weight("feed_forward2", 2, d, ffn)?,
                ff2_b2: ff_bias("feed_forward2", 2, d)?,
                norm_ff1: norm("norm_feed_forward1")?,
                norm_attn: norm("norm_self_att")?,
                norm_conv: norm("norm_conv")?,
                norm_ff2: norm("norm_feed_forward2")?,
                norm_out: norm("norm_out")?,
                q_w_t: attention_weight("linear_q")?,
                q_b: attention_bias("linear_q")?,
                k_w_t: attention_weight("linear_k")?,
                k_b: attention_bias("linear_k")?,
                v_w_t: attention_weight("linear_v")?,
                v_b: attention_bias("linear_v")?,
                o_w_t: attention_weight("linear_out")?,
                o_b: attention_bias("linear_out")?,
                relative_k_w_t: attention_weight("linear_pos")?,
                bias_u: tensor(
                    &format!("{prefix}.self_attn.pos_bias_u"),
                    &[enc.n_head, enc.head_dim()],
                )?,
                bias_v: tensor(
                    &format!("{prefix}.self_attn.pos_bias_v"),
                    &[enc.n_head, enc.head_dim()],
                )?,
                conv_pw1_w_t: transpose_out_in(
                    tensor(
                        &format!("{prefix}.conv.pointwise_conv1.weight"),
                        &[2 * d, d, 1],
                    )?,
                    2 * d,
                    d,
                ),
                conv_pw1_b: Some(tensor(
                    &format!("{prefix}.conv.pointwise_conv1.bias"),
                    &[2 * d],
                )?),
                conv_dw_w: tensor(
                    &format!("{prefix}.conv.depthwise_conv.weight"),
                    &[d, 1, enc.conv_kernel_size],
                )?,
                conv_dw_b: Some(tensor(&format!("{prefix}.conv.depthwise_conv.bias"), &[d])?),
                conv_inner_norm: FastConformerConvNorm::BatchNorm {
                    weight: tensor(&format!("{prefix}.conv.batch_norm.weight"), &[d])?,
                    bias: tensor(&format!("{prefix}.conv.batch_norm.bias"), &[d])?,
                    running_mean: tensor(&format!("{prefix}.conv.batch_norm.running_mean"), &[d])?,
                    running_var: tensor(&format!("{prefix}.conv.batch_norm.running_var"), &[d])?,
                },
                conv_pw2_w_t: transpose_out_in(
                    tensor(&format!("{prefix}.conv.pointwise_conv2.weight"), &[d, d, 1])?,
                    d,
                    d,
                ),
                conv_pw2_b: Some(tensor(
                    &format!("{prefix}.conv.pointwise_conv2.bias"),
                    &[d],
                )?),
            });
        }

        let decoder_d = config.decoder.d_model;
        let decoder_ffn = config.decoder.ffn_dim;
        let decoder_attention = |prefix: &str| -> Result<SelfAttentionWeights> {
            Ok(SelfAttentionWeights {
                q_w: tensor(
                    &format!("{prefix}.query_net.weight"),
                    &[decoder_d, decoder_d],
                )?,
                q_b: tensor(&format!("{prefix}.query_net.bias"), &[decoder_d])?,
                k_w: tensor(&format!("{prefix}.key_net.weight"), &[decoder_d, decoder_d])?,
                k_b: tensor(&format!("{prefix}.key_net.bias"), &[decoder_d])?,
                v_w: tensor(
                    &format!("{prefix}.value_net.weight"),
                    &[decoder_d, decoder_d],
                )?,
                v_b: tensor(&format!("{prefix}.value_net.bias"), &[decoder_d])?,
                out_w: tensor(
                    &format!("{prefix}.out_projection.weight"),
                    &[decoder_d, decoder_d],
                )?,
                out_b: tensor(&format!("{prefix}.out_projection.bias"), &[decoder_d])?,
            })
        };
        let cross_attention = |prefix: &str| -> Result<CrossAttentionWeights> {
            Ok(CrossAttentionWeights {
                q_w: tensor(
                    &format!("{prefix}.query_net.weight"),
                    &[decoder_d, decoder_d],
                )?,
                q_b: tensor(&format!("{prefix}.query_net.bias"), &[decoder_d])?,
                k_w_t: transpose_out_in(
                    tensor(&format!("{prefix}.key_net.weight"), &[decoder_d, d])?,
                    decoder_d,
                    d,
                ),
                k_b: tensor(&format!("{prefix}.key_net.bias"), &[decoder_d])?,
                v_w_t: transpose_out_in(
                    tensor(&format!("{prefix}.value_net.weight"), &[decoder_d, d])?,
                    decoder_d,
                    d,
                ),
                v_b: tensor(&format!("{prefix}.value_net.bias"), &[decoder_d])?,
                out_w: tensor(
                    &format!("{prefix}.out_projection.weight"),
                    &[decoder_d, decoder_d],
                )?,
                out_b: tensor(&format!("{prefix}.out_projection.bias"), &[decoder_d])?,
            })
        };
        let mut decoder = Vec::with_capacity(config.decoder.n_layer);
        for layer in 0..config.decoder.n_layer {
            let prefix = format!("transf_decoder._decoder.layers.{layer}");
            let norm = |name: &str| -> Result<ParakeetBoundNorm> {
                Ok(ParakeetBoundNorm {
                    weight: tensor(&format!("{prefix}.{name}.weight"), &[decoder_d])?,
                    bias: tensor(&format!("{prefix}.{name}.bias"), &[decoder_d])?,
                })
            };
            decoder.push(DecoderBlockWeights {
                norm_self: norm("layer_norm_1")?,
                self_attention: decoder_attention(&format!("{prefix}.first_sub_layer"))?,
                norm_cross: norm("layer_norm_2")?,
                cross_attention: cross_attention(&format!("{prefix}.second_sub_layer"))?,
                norm_ffn: norm("layer_norm_3")?,
                ffn_in_w: tensor(
                    &format!("{prefix}.third_sub_layer.dense_in.weight"),
                    &[decoder_ffn, decoder_d],
                )?,
                ffn_in_b: tensor(
                    &format!("{prefix}.third_sub_layer.dense_in.bias"),
                    &[decoder_ffn],
                )?,
                ffn_out_w: tensor(
                    &format!("{prefix}.third_sub_layer.dense_out.weight"),
                    &[decoder_d, decoder_ffn],
                )?,
                ffn_out_b: tensor(
                    &format!("{prefix}.third_sub_layer.dense_out.bias"),
                    &[decoder_d],
                )?,
            });
        }

        let bound = Self {
            release,
            checkpoint,
            encoder_config: enc,
            subsampling,
            encoder,
            token_embedding: tensor(
                "transf_decoder._embedding.token_embedding.weight",
                &[release.vocab_size, decoder_d],
            )?,
            position_embedding: tensor(
                "transf_decoder._embedding.position_embedding.pos_enc",
                &[config.decoder.max_sequence_length, decoder_d],
            )?,
            embedding_norm: ParakeetBoundNorm {
                weight: tensor("transf_decoder._embedding.layer_norm.weight", &[decoder_d])?,
                bias: tensor("transf_decoder._embedding.layer_norm.bias", &[decoder_d])?,
            },
            decoder,
            decoder_final_norm: ParakeetBoundNorm {
                weight: tensor(
                    "transf_decoder._decoder.final_layer_norm.weight",
                    &[decoder_d],
                )?,
                bias: tensor(
                    "transf_decoder._decoder.final_layer_norm.bias",
                    &[decoder_d],
                )?,
            },
            head_w: tensor(
                "log_softmax.mlp.layer0.weight",
                &[release.vocab_size, decoder_d],
            )?,
            head_b: tensor("log_softmax.mlp.layer0.bias", &[release.vocab_size])?,
        };
        verify_consumed_tensor_set(file, &consumed.into_inner(), label)?;
        Ok(bound)
    }

    pub(crate) const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    pub(crate) const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    pub(crate) fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    pub(crate) fn encode_pcm(&self, compute: &Compute, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (features, frames) =
            parakeet_logmel(pcm, self.release.sample_rate, self.encoder_config.in_dim)?;
        let (mut hidden, encoded_frames) = subsampling_forward(
            compute,
            &features,
            frames,
            self.encoder_config.in_dim,
            &self.subsampling,
            &self.encoder_config,
        )?;
        if encoded_frames > self.encoder_config.max_position_embeddings {
            return Err(VokraError::InvalidArgument(format!(
                "{} encoder produced {encoded_frames} frames, exceeding max_position_embeddings={}",
                self.release.label(),
                self.encoder_config.max_position_embeddings
            )));
        }
        let positions = relative_positions(encoded_frames, self.encoder_config.d_model);
        for block in &self.encoder {
            conformer_block_forward(
                compute,
                &mut hidden,
                encoded_frames,
                block,
                &positions,
                &self.encoder_config,
            )?;
        }
        Ok((hidden, encoded_frames))
    }

    pub(crate) fn decode_tokens(
        &self,
        compute: &Compute,
        encoder: &[f32],
        encoder_frames: usize,
        config: &CanaryConfig,
        prompt: &[u32],
        requested_max_new_tokens: Option<usize>,
    ) -> Result<Vec<u32>> {
        let capacity = config
            .decoder
            .max_sequence_length
            .checked_sub(prompt.len())
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{} decoder context is shorter than its prompt",
                    self.release.label()
                ))
            })?;
        let max_new_tokens = match requested_max_new_tokens {
            Some(value) if value > capacity => {
                return Err(VokraError::InvalidArgument(format!(
                    "{} max_new_tokens={value} exceeds decoder capacity {capacity} after the {}-token prompt",
                    self.release.label(),
                    prompt.len(),
                )));
            }
            Some(value) => value,
            None => default_max_new_tokens(
                encoder_frames,
                prompt.len(),
                config.decoder.max_sequence_length,
            ),
        };
        let mut state = DecoderState::new(compute, encoder, encoder_frames, self, config)?;
        let mut logits = Vec::new();
        for &token in prompt {
            logits = state.step(compute, token, self, config)?;
        }
        let mut output = Vec::new();
        for _ in 0..max_new_tokens {
            let token = argmax_finite(&logits)? as u32;
            if token == self.release.eos_token_id {
                break;
            }
            output.push(token);
            logits = state.step(compute, token, self, config)?;
        }
        Ok(output)
    }
}

/// NeMo computes an absolute target-sequence limit first, then subtracts the
/// already-materialized Canary2 prompt. Keeping the prompt outside this
/// subtraction would permit nine extra decoder steps on a no-EOS hypothesis.
fn default_max_new_tokens(
    encoder_frames: usize,
    prompt_tokens: usize,
    max_sequence_length: usize,
) -> usize {
    encoder_frames
        .saturating_add(50)
        .min(max_sequence_length)
        .saturating_sub(prompt_tokens)
}

fn verify_consumed_tensor_set(
    file: &GgufFile,
    consumed: &BTreeSet<String>,
    label: &str,
) -> Result<()> {
    const AUTHENTICATED_FRONTEND_BUFFERS: [&str; 2] = [
        "preprocessor.featurizer.fb",
        "preprocessor.featurizer.window",
    ];
    let executable = file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .filter(|name| !AUTHENTICATED_FRONTEND_BUFFERS.contains(name))
        .collect::<BTreeSet<_>>();
    let consumed_refs = consumed.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if executable != consumed_refs {
        let missing = executable
            .difference(&consumed_refs)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = consumed_refs
            .difference(&executable)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        return Err(VokraError::ModelLoad(format!(
            "{label}: executable tensor coverage mismatch: consumed={}, expected={} (missing={missing:?}, unexpected={unexpected:?}). Only the authenticated mel-filter and Hann-window buffers may be reproduced by the shared frontend",
            consumed_refs.len(),
            executable.len(),
        )));
    }
    Ok(())
}

fn encoder_config(config: &CanaryConfig) -> ParakeetEncoderConfig {
    let enc = &config.encoder;
    ParakeetEncoderConfig {
        n_layer: enc.n_layer,
        d_model: enc.d_model,
        n_head: enc.n_head,
        n_head_kv: enc.n_head_kv,
        ffn_dim: enc.ffn_dim,
        conv_kernel_size: enc.conv_kernel_size,
        in_dim: enc.in_dim,
        subsampling_factor: enc.subsampling_factor,
        subsampling_conv_kernel_size: enc.subsampling_conv_kernel_size,
        subsampling_conv_stride: enc.subsampling_conv_stride,
        subsampling_conv_channels: enc.subsampling_conv_channels,
        max_position_embeddings: enc.max_position_embeddings,
        attention_bias: enc.attention_bias,
        convolution_bias: enc.convolution_bias,
        scale_input: enc.scale_input,
    }
}

struct DecoderState {
    self_keys: Vec<Vec<f32>>,
    self_values: Vec<Vec<f32>>,
    cross_keys: Vec<Vec<f32>>,
    cross_values: Vec<Vec<f32>>,
    position: usize,
    encoder_frames: usize,
}

impl DecoderState {
    fn new(
        compute: &Compute,
        encoder: &[f32],
        encoder_frames: usize,
        weights: &CanaryBoundWeights,
        config: &CanaryConfig,
    ) -> Result<Self> {
        let enc_width = config.encoder.d_model;
        let dec_width = config.decoder.d_model;
        if encoder.len() != encoder_frames * enc_width || encoder_frames == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{} decoder encoder shape mismatch: values={}, frames={encoder_frames}, width={enc_width}",
                weights.release.label(),
                encoder.len()
            )));
        }
        let mut cross_keys = Vec::with_capacity(weights.decoder.len());
        let mut cross_values = Vec::with_capacity(weights.decoder.len());
        for block in &weights.decoder {
            let mut keys = vec![0.0; encoder_frames * dec_width];
            compute.gemm_f32(
                encoder_frames,
                dec_width,
                enc_width,
                encoder,
                &block.cross_attention.k_w_t,
                Some(&block.cross_attention.k_b),
                &mut keys,
            )?;
            let mut values = vec![0.0; encoder_frames * dec_width];
            compute.gemm_f32(
                encoder_frames,
                dec_width,
                enc_width,
                encoder,
                &block.cross_attention.v_w_t,
                Some(&block.cross_attention.v_b),
                &mut values,
            )?;
            cross_keys.push(keys);
            cross_values.push(values);
        }
        Ok(Self {
            self_keys: vec![Vec::new(); weights.decoder.len()],
            self_values: vec![Vec::new(); weights.decoder.len()],
            cross_keys,
            cross_values,
            position: 0,
            encoder_frames,
        })
    }

    fn step(
        &mut self,
        compute: &Compute,
        token: u32,
        weights: &CanaryBoundWeights,
        config: &CanaryConfig,
    ) -> Result<Vec<f32>> {
        let width = config.decoder.d_model;
        if token as usize >= weights.release.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "{} decoder token {token} outside 0..{}",
                weights.release.label(),
                weights.release.vocab_size,
            )));
        }
        if self.position >= config.decoder.max_sequence_length {
            return Err(VokraError::InvalidArgument(format!(
                "{} decoder position {} exceeds max_sequence_length={}",
                weights.release.label(),
                self.position,
                config.decoder.max_sequence_length
            )));
        }
        let token_offset = token as usize * width;
        let position_offset = self.position * width;
        let mut hidden = vec![0.0; width];
        for index in 0..width {
            hidden[index] = weights.token_embedding[token_offset + index]
                + weights.position_embedding[position_offset + index];
        }
        hidden = layer_norm(compute, &hidden, &weights.embedding_norm)?;

        for (layer, block) in weights.decoder.iter().enumerate() {
            let normalized = layer_norm(compute, &hidden, &block.norm_self)?;
            let q = linear(
                compute,
                &normalized,
                &block.self_attention.q_w,
                &block.self_attention.q_b,
            )?;
            let k = linear(
                compute,
                &normalized,
                &block.self_attention.k_w,
                &block.self_attention.k_b,
            )?;
            let v = linear(
                compute,
                &normalized,
                &block.self_attention.v_w,
                &block.self_attention.v_b,
            )?;
            self.self_keys[layer].extend_from_slice(&k);
            self.self_values[layer].extend_from_slice(&v);
            let context = attention_one_query(
                compute,
                &q,
                &self.self_keys[layer],
                &self.self_values[layer],
                self.position + 1,
                config.decoder.n_head,
            )?;
            let branch = linear(
                compute,
                &context,
                &block.self_attention.out_w,
                &block.self_attention.out_b,
            )?;
            add_assign(&mut hidden, &branch);

            let normalized = layer_norm(compute, &hidden, &block.norm_cross)?;
            let q = linear(
                compute,
                &normalized,
                &block.cross_attention.q_w,
                &block.cross_attention.q_b,
            )?;
            let context = attention_one_query(
                compute,
                &q,
                &self.cross_keys[layer],
                &self.cross_values[layer],
                self.encoder_frames,
                config.decoder.n_head,
            )?;
            let branch = linear(
                compute,
                &context,
                &block.cross_attention.out_w,
                &block.cross_attention.out_b,
            )?;
            add_assign(&mut hidden, &branch);

            let normalized = layer_norm(compute, &hidden, &block.norm_ffn)?;
            let expanded = linear(compute, &normalized, &block.ffn_in_w, &block.ffn_in_b)?;
            let mut activated = vec![0.0; expanded.len()];
            compute.relu_f32(&expanded, &mut activated)?;
            let branch = linear(compute, &activated, &block.ffn_out_w, &block.ffn_out_b)?;
            add_assign(&mut hidden, &branch);
        }
        hidden = layer_norm(compute, &hidden, &weights.decoder_final_norm)?;
        self.position += 1;
        linear(compute, &hidden, &weights.head_w, &weights.head_b)
    }
}

fn linear(compute: &Compute, input: &[f32], weight: &[f32], bias: &[f32]) -> Result<Vec<f32>> {
    let output = bias.len();
    if input.is_empty() || weight.len() != output * input.len() {
        return Err(VokraError::InvalidArgument(format!(
            "Canary Transformer-AED linear shape mismatch: input={}, weight={}, bias={}",
            input.len(),
            weight.len(),
            bias.len()
        )));
    }
    let mut result = vec![0.0; output];
    compute.gemv_f32(output, input.len(), weight, input, Some(bias), &mut result)?;
    Ok(result)
}

fn layer_norm(compute: &Compute, input: &[f32], norm: &ParakeetBoundNorm) -> Result<Vec<f32>> {
    let width = norm.weight.len();
    if input.len() != width || norm.bias.len() != width {
        return Err(VokraError::InvalidArgument(
            "Canary Transformer-AED decoder LayerNorm shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; width];
    compute.layer_norm_f32(input, &mut output, 1, width, &norm.weight, &norm.bias, 1e-5)?;
    Ok(output)
}

fn attention_one_query(
    compute: &Compute,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    positions: usize,
    heads: usize,
) -> Result<Vec<f32>> {
    let width = query.len();
    if heads == 0
        || width == 0
        || width % heads != 0
        || keys.len() != positions * width
        || values.len() != positions * width
    {
        return Err(VokraError::InvalidArgument(format!(
            "Canary Transformer-AED attention shape mismatch: q={}, keys={}, values={}, positions={positions}, heads={heads}",
            width,
            keys.len(),
            values.len()
        )));
    }
    let head_dim = width / heads;
    // NeMo deliberately divides Q and K independently by sqrt(sqrt(d))
    // before their matmul for numerical stability. Applying one 1/sqrt(d)
    // factor after the dot product is algebraically equivalent but not the
    // same FP32 rounding contract.
    let attention_scale = (head_dim as f32).sqrt().sqrt();
    let mut context = vec![0.0; width];
    for head in 0..heads {
        let head_offset = head * head_dim;
        let mut keys_head = vec![0.0; positions * head_dim];
        for position in 0..positions {
            for dim in 0..head_dim {
                keys_head[position * head_dim + dim] =
                    keys[position * width + head_offset + dim] / attention_scale;
            }
        }
        let scaled_query = query[head_offset..head_offset + head_dim]
            .iter()
            .map(|value| value / attention_scale)
            .collect::<Vec<_>>();
        let mut scores = vec![0.0; positions];
        compute.gemv_f32(
            positions,
            head_dim,
            &keys_head,
            &scaled_query,
            None,
            &mut scores,
        )?;
        let mut probabilities = vec![0.0; positions];
        compute.softmax_f32(&scores, &mut probabilities, 1, positions)?;
        let mut values_t = vec![0.0; head_dim * positions];
        for dim in 0..head_dim {
            for position in 0..positions {
                values_t[dim * positions + position] = values[position * width + head_offset + dim];
            }
        }
        compute.gemv_f32(
            head_dim,
            positions,
            &values_t,
            &probabilities,
            None,
            &mut context[head_offset..head_offset + head_dim],
        )?;
    }
    Ok(context)
}

fn add_assign(target: &mut [f32], branch: &[f32]) {
    debug_assert_eq!(target.len(), branch.len());
    for (target, branch) in target.iter_mut().zip(branch) {
        *target += branch;
    }
}

fn argmax_finite(values: &[f32]) -> Result<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "Canary Transformer-AED decoder produced non-finite logit at {index}: {value}"
            )));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index).ok_or_else(|| {
        VokraError::InvalidArgument("Canary Transformer-AED decoder produced no logits".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::default_max_new_tokens;

    #[test]
    fn nemo_generation_limit_subtracts_the_existing_prompt() {
        assert_eq!(default_max_new_tokens(100, 9, 1_024), 141);
        assert_eq!(default_max_new_tokens(2_000, 9, 1_024), 1_015);
        assert_eq!(default_max_new_tokens(0, 64, 32), 0);
    }
}
