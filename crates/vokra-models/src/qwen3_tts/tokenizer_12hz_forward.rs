//! Native mapped forward for the official Qwen3-TTS 12 Hz waveform decoder.
//!
//! Tensor payloads stay in the read-only GGUF mapping. Dense linear weights
//! are transposed lazily once because [`Compute::gemm_f32`] consumes `[in,
//! out]` while PyTorch stores `Linear.weight` as `[out, in]`; causal
//! transposed-convolution kernels are likewise flipped once and cached. All
//! learned hot operations go through [`Compute`], so selecting Metal never
//! hides a CPU execution fallback.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::{CodebookTable, Qwen3TtsCodecConfig};

use crate::compute::{Compute, HotOp};

use super::tokenizer_12hz::Qwen3TtsTokenizer12HzConfig;

const LABEL: &str = "qwen3_tts_tokenizer_12hz";
const CODEBOOK_EPSILON: f32 = 1e-5;
const CONVNEXT_NORM_EPSILON: f32 = 1e-6;

/// Complete learned-op inventory for the released decoder. Every entry has a
/// real CPU and Metal implementation; backend preflight rejects every other
/// incomplete backend before tensor pages are touched.
pub(super) const QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS: &[HotOp] = &[
    HotOp::Qwen3TtsCodec,
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Silu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeBeta,
];

/// Mapping-owning decoder payload. Clones share the mmap, normalized
/// codebooks, and lazily transformed weights.
#[derive(Clone)]
pub(super) struct MappedDecoder {
    inner: Arc<MappedDecoderInner>,
}

struct MappedDecoderInner {
    file: Arc<GgufFile>,
    codebooks: OnceLock<Vec<CodebookTable>>,
    transformed: Mutex<BTreeMap<String, Arc<Vec<f32>>>>,
}

impl core::fmt::Debug for MappedDecoder {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Qwen3TtsTokenizer12HzMappedDecoder")
            .field("tensors", &self.inner.file.tensors().len())
            .field(
                "codebooks_materialized",
                &self.inner.codebooks.get().is_some(),
            )
            .field(
                "transformed_weights",
                &self
                    .inner
                    .transformed
                    .lock()
                    .map(|weights| weights.len())
                    .unwrap_or_default(),
            )
            .finish()
    }
}

impl MappedDecoder {
    /// Validates that all authenticated F32 tensors can be borrowed directly.
    /// This checks dtype/alignment/bounds without copying or faulting every
    /// tensor page into resident memory.
    pub(super) fn bind(file: Arc<GgufFile>) -> Result<Self> {
        for tensor in file.tensors() {
            vokra_mmap::tensor_f32_view(&file, &tensor.name).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "{LABEL}: zero-copy view for `{}` failed: {error}",
                    tensor.name
                ))
            })?;
        }
        Ok(Self {
            inner: Arc::new(MappedDecoderInner {
                file,
                codebooks: OnceLock::new(),
                transformed: Mutex::new(BTreeMap::new()),
            }),
        })
    }

    pub(super) fn decode_codes(
        &self,
        backend: BackendKind,
        codes: &[Vec<u32>],
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(backend, QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS)?;
        let frames = codes[0].len();
        let expected_samples = frames
            .checked_mul(config.decode_upsample_rate)
            .ok_or_else(|| invalid("frame-to-sample length overflow"))?;
        let mut pcm = Vec::with_capacity(expected_samples);
        let mut start = 0;
        while start < frames {
            let end = (start + config.chunk_size).min(frames);
            let context = config.left_context.min(start);
            let chunk_start = start - context;
            let chunk_codes = codes
                .iter()
                .map(|row| row[chunk_start..end].to_vec())
                .collect::<Vec<_>>();
            let chunk = self.decode_chunk(&compute, &chunk_codes, config)?;
            let discard = context
                .checked_mul(config.decode_upsample_rate)
                .ok_or_else(|| invalid("left-context sample length overflow"))?;
            if discard > chunk.len() {
                return Err(invalid(format!(
                    "decoded chunk has {} samples but context requires discarding {discard}",
                    chunk.len()
                )));
            }
            pcm.extend_from_slice(&chunk[discard..]);
            start = end;
        }
        if pcm.len() != expected_samples {
            return Err(invalid(format!(
                "chunked decoder emitted {} samples, expected {expected_samples}",
                pcm.len()
            )));
        }
        reject_non_finite("decoded PCM", &pcm)?;
        Ok(pcm)
    }

    fn decode_chunk(
        &self,
        compute: &Compute,
        codes: &[Vec<u32>],
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let frames = codes[0].len();
        let mut hidden = self.quantizer_decode(compute, codes, config)?;
        hidden = frame_to_channel(&hidden, frames, config.codebook_dim)?;
        hidden = self.causal_conv(
            compute,
            &hidden,
            frames,
            config.codebook_dim,
            config.latent_dim,
            3,
            1,
            1,
            "decoder.pre_conv",
        )?;
        hidden = channel_to_frame(&hidden, config.latent_dim, frames)?;
        hidden = self.pre_transformer(compute, &hidden, frames, config)?;
        hidden = frame_to_channel(&hidden, frames, config.latent_dim)?;

        let mut time = frames;
        for (stage, &ratio) in config.upsampling_ratios.iter().enumerate() {
            let prefix = format!("decoder.upsample.{stage}");
            hidden = self.causal_transposed_conv(
                compute,
                &hidden,
                time,
                config.latent_dim,
                config.latent_dim,
                ratio,
                ratio,
                &format!("{prefix}.0"),
            )?;
            time = time
                .checked_mul(ratio)
                .ok_or_else(|| invalid("ConvNeXt upsample length overflow"))?;
            hidden = self.convnext(
                compute,
                &hidden,
                time,
                config.latent_dim,
                &format!("{prefix}.1"),
            )?;
        }

        hidden = self.causal_conv(
            compute,
            &hidden,
            time,
            config.latent_dim,
            config.decoder_dim,
            7,
            1,
            1,
            "decoder.decoder.0",
        )?;
        let mut channels = config.decoder_dim;
        for (stage, &rate) in config.upsample_rates.iter().enumerate() {
            let prefix = format!("decoder.decoder.{}.block", stage + 1);
            hidden = self.snake_beta(compute, &hidden, channels, time, &format!("{prefix}.0"))?;
            let output_channels = channels / 2;
            hidden = self.causal_transposed_conv(
                compute,
                &hidden,
                time,
                channels,
                output_channels,
                2 * rate,
                rate,
                &format!("{prefix}.1"),
            )?;
            time = time
                .checked_mul(rate)
                .ok_or_else(|| invalid("waveform upsample length overflow"))?;
            channels = output_channels;
            for (residual, dilation) in (2..5).zip([1, 3, 9]) {
                hidden = self.residual_unit(
                    compute,
                    &hidden,
                    channels,
                    time,
                    dilation,
                    &format!("{prefix}.{residual}"),
                )?;
            }
        }
        hidden = self.snake_beta(compute, &hidden, channels, time, "decoder.decoder.5")?;
        hidden = self.causal_conv(
            compute,
            &hidden,
            time,
            channels,
            1,
            7,
            1,
            1,
            "decoder.decoder.6",
        )?;
        for sample in &mut hidden {
            *sample = sample.clamp(-1.0, 1.0);
        }
        let expected = frames
            .checked_mul(config.decode_upsample_rate)
            .ok_or_else(|| invalid("chunk sample length overflow"))?;
        if time != expected || hidden.len() != expected {
            return Err(invalid(format!(
                "native graph emitted shape [1,{time}] / {} values, expected [1,{expected}]",
                hidden.len()
            )));
        }
        Ok(hidden)
    }

    fn quantizer_decode(
        &self,
        compute: &Compute,
        codes: &[Vec<u32>],
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let frames = codes[0].len();
        let tables = self.codebooks(config)?;
        let branch_config = |num_quantizers| Qwen3TtsCodecConfig {
            num_quantizers,
            num_semantic_quantizers: 0,
            codebook_size: config.codebook_size,
            semantic_codebook_size: config.codebook_size,
            codebook_dim: config.quantizer_dim,
            sample_rate: config.output_sample_rate,
            downsample_rate: config.decode_upsample_rate as u32,
        };
        let first = compute.qwen3_tts_codec_f32(
            &codes[..config.num_semantic_quantizers],
            &tables[..config.num_semantic_quantizers],
            &branch_config(config.num_semantic_quantizers),
        )?;
        let rest = compute.qwen3_tts_codec_f32(
            &codes[config.num_semantic_quantizers..],
            &tables[config.num_semantic_quantizers..],
            &branch_config(config.num_quantizers - config.num_semantic_quantizers),
        )?;
        let first = self.linear(
            compute,
            &first,
            frames,
            config.quantizer_dim,
            config.codebook_dim,
            "decoder.quantizer.rvq_first.output_proj.weight",
            None,
        )?;
        let rest = self.linear(
            compute,
            &rest,
            frames,
            config.quantizer_dim,
            config.codebook_dim,
            "decoder.quantizer.rvq_rest.output_proj.weight",
            None,
        )?;
        Ok(first
            .into_iter()
            .zip(rest)
            .map(|(semantic, acoustic)| semantic + acoustic)
            .collect())
    }

    fn codebooks(&self, config: &Qwen3TtsTokenizer12HzConfig) -> Result<&[CodebookTable]> {
        if self.inner.codebooks.get().is_none() {
            let mut tables = Vec::with_capacity(config.num_quantizers);
            for quantizer in 0..config.num_quantizers {
                let (branch, layer) = if quantizer == 0 {
                    ("rvq_first", 0)
                } else {
                    ("rvq_rest", quantizer - 1)
                };
                let prefix = format!("decoder.quantizer.{branch}.vq.layers.{layer}._codebook");
                let usage = self.tensor(&format!("{prefix}.cluster_usage"))?;
                let sums = self.tensor(&format!("{prefix}.embedding_sum"))?;
                let mut normalized = Vec::with_capacity(sums.len());
                for row in 0..config.codebook_size {
                    let denominator = usage[row].max(CODEBOOK_EPSILON);
                    let start = row * config.quantizer_dim;
                    normalized.extend(
                        sums[start..start + config.quantizer_dim]
                            .iter()
                            .map(|value| value / denominator),
                    );
                }
                reject_non_finite("normalized Euclidean codebook", &normalized)?;
                tables.push(CodebookTable::new(
                    config.codebook_size,
                    config.quantizer_dim,
                    normalized,
                )?);
            }
            // Another clone may win the race; both results are authenticated
            // and equal, so retaining the first completed set is sufficient.
            let _ = self.inner.codebooks.set(tables);
        }
        Ok(self
            .inner
            .codebooks
            .get()
            .expect("codebooks are initialized above"))
    }

    fn pre_transformer(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let mut hidden = self.linear(
            compute,
            input,
            frames,
            config.latent_dim,
            config.hidden_size,
            "decoder.pre_transformer.input_proj.weight",
            Some("decoder.pre_transformer.input_proj.bias"),
        )?;
        for layer in 0..config.num_hidden_layers {
            hidden = self.transformer_layer(compute, &hidden, frames, layer, config)?;
        }
        let gamma = self.tensor("decoder.pre_transformer.norm.weight")?;
        let mut normalized = vec![0.0; hidden.len()];
        compute.rms_norm_f32(
            &hidden,
            &mut normalized,
            frames,
            config.hidden_size,
            gamma,
            config.rms_norm_eps,
        )?;
        self.linear(
            compute,
            &normalized,
            frames,
            config.hidden_size,
            config.latent_dim,
            "decoder.pre_transformer.output_proj.weight",
            Some("decoder.pre_transformer.output_proj.bias"),
        )
    }

    fn transformer_layer(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        layer: usize,
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let prefix = format!("decoder.pre_transformer.layers.{layer}");
        let mut normalized = vec![0.0; input.len()];
        compute.rms_norm_f32(
            input,
            &mut normalized,
            frames,
            config.hidden_size,
            self.tensor(&format!("{prefix}.input_layernorm.weight"))?,
            config.rms_norm_eps,
        )?;
        let attention = self.attention(compute, &normalized, frames, &prefix, config)?;
        let attention_scale = self.tensor(&format!("{prefix}.self_attn_layer_scale.scale"))?;
        let mut hidden = input.to_vec();
        for frame in 0..frames {
            for channel in 0..config.hidden_size {
                let offset = frame * config.hidden_size + channel;
                hidden[offset] += attention[offset] * attention_scale[channel];
            }
        }

        compute.rms_norm_f32(
            &hidden,
            &mut normalized,
            frames,
            config.hidden_size,
            self.tensor(&format!("{prefix}.post_attention_layernorm.weight"))?,
            config.rms_norm_eps,
        )?;
        let gate = self.linear(
            compute,
            &normalized,
            frames,
            config.hidden_size,
            config.intermediate_size,
            &format!("{prefix}.mlp.gate_proj.weight"),
            None,
        )?;
        let up = self.linear(
            compute,
            &normalized,
            frames,
            config.hidden_size,
            config.intermediate_size,
            &format!("{prefix}.mlp.up_proj.weight"),
            None,
        )?;
        let mut activated = vec![0.0; gate.len()];
        compute.silu_f32(&gate, &mut activated)?;
        for (value, up) in activated.iter_mut().zip(up) {
            *value *= up;
        }
        let mlp = self.linear(
            compute,
            &activated,
            frames,
            config.intermediate_size,
            config.hidden_size,
            &format!("{prefix}.mlp.down_proj.weight"),
            None,
        )?;
        let mlp_scale = self.tensor(&format!("{prefix}.mlp_layer_scale.scale"))?;
        for frame in 0..frames {
            for channel in 0..config.hidden_size {
                let offset = frame * config.hidden_size + channel;
                hidden[offset] += mlp[offset] * mlp_scale[channel];
            }
        }
        Ok(hidden)
    }

    fn attention(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        prefix: &str,
        config: &Qwen3TtsTokenizer12HzConfig,
    ) -> Result<Vec<f32>> {
        let query_width = config.num_attention_heads * config.head_dim;
        let key_value_width = config.num_key_value_heads * config.head_dim;
        let mut query = self.linear(
            compute,
            input,
            frames,
            config.hidden_size,
            query_width,
            &format!("{prefix}.self_attn.q_proj.weight"),
            None,
        )?;
        let mut key = self.linear(
            compute,
            input,
            frames,
            config.hidden_size,
            key_value_width,
            &format!("{prefix}.self_attn.k_proj.weight"),
            None,
        )?;
        let value = self.linear(
            compute,
            input,
            frames,
            config.hidden_size,
            key_value_width,
            &format!("{prefix}.self_attn.v_proj.weight"),
            None,
        )?;
        apply_half_split_rope(
            &mut query,
            frames,
            config.num_attention_heads,
            config.head_dim,
            config.rope_theta,
        )?;
        apply_half_split_rope(
            &mut key,
            frames,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
        )?;

        let groups = config.num_attention_heads / config.num_key_value_heads;
        let mut merged = vec![0.0; frames * query_width];
        let scale = (config.head_dim as f32).sqrt().recip();
        for head in 0..config.num_attention_heads {
            let key_value_head = head / groups;
            let mut query_head = vec![0.0; frames * config.head_dim];
            let mut key_transposed = vec![0.0; config.head_dim * frames];
            let mut value_head = vec![0.0; frames * config.head_dim];
            for frame in 0..frames {
                let query_start = frame * query_width + head * config.head_dim;
                let key_value_start = frame * key_value_width + key_value_head * config.head_dim;
                query_head[frame * config.head_dim..(frame + 1) * config.head_dim]
                    .copy_from_slice(&query[query_start..query_start + config.head_dim]);
                value_head[frame * config.head_dim..(frame + 1) * config.head_dim]
                    .copy_from_slice(&value[key_value_start..key_value_start + config.head_dim]);
                for inner in 0..config.head_dim {
                    key_transposed[inner * frames + frame] = key[key_value_start + inner];
                }
            }
            let mut logits = vec![0.0; frames * frames];
            compute.gemm_f32(
                frames,
                frames,
                config.head_dim,
                &query_head,
                &key_transposed,
                None,
                &mut logits,
            )?;
            for query_index in 0..frames {
                for key_index in 0..frames {
                    let logit = &mut logits[query_index * frames + key_index];
                    if sliding_causal_visible(query_index, key_index, config.sliding_window) {
                        *logit *= scale;
                    } else {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            let mut probabilities = vec![0.0; logits.len()];
            compute.softmax_f32(&logits, &mut probabilities, frames, frames)?;
            let mut context = vec![0.0; frames * config.head_dim];
            compute.gemm_f32(
                frames,
                config.head_dim,
                frames,
                &probabilities,
                &value_head,
                None,
                &mut context,
            )?;
            for frame in 0..frames {
                let destination = frame * query_width + head * config.head_dim;
                merged[destination..destination + config.head_dim].copy_from_slice(
                    &context[frame * config.head_dim..(frame + 1) * config.head_dim],
                );
            }
        }
        self.linear(
            compute,
            &merged,
            frames,
            query_width,
            config.hidden_size,
            &format!("{prefix}.self_attn.o_proj.weight"),
            None,
        )
    }

    fn convnext(
        &self,
        compute: &Compute,
        input: &[f32],
        time: usize,
        channels: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let convolved = self.causal_conv(
            compute,
            input,
            time,
            channels,
            channels,
            7,
            1,
            channels,
            &format!("{prefix}.dwconv"),
        )?;
        let frame_major = channel_to_frame(&convolved, channels, time)?;
        let mut normalized = vec![0.0; frame_major.len()];
        compute.layer_norm_f32(
            &frame_major,
            &mut normalized,
            time,
            channels,
            self.tensor(&format!("{prefix}.norm.weight"))?,
            self.tensor(&format!("{prefix}.norm.bias"))?,
            CONVNEXT_NORM_EPSILON,
        )?;
        let expanded = self.linear(
            compute,
            &normalized,
            time,
            channels,
            4 * channels,
            &format!("{prefix}.pwconv1.weight"),
            Some(&format!("{prefix}.pwconv1.bias")),
        )?;
        let mut activated = vec![0.0; expanded.len()];
        compute.gelu_f32(&expanded, &mut activated)?;
        let projected = self.linear(
            compute,
            &activated,
            time,
            4 * channels,
            channels,
            &format!("{prefix}.pwconv2.weight"),
            Some(&format!("{prefix}.pwconv2.bias")),
        )?;
        let gamma = self.tensor(&format!("{prefix}.gamma"))?;
        let mut output = input.to_vec();
        for frame in 0..time {
            for channel in 0..channels {
                output[channel * time + frame] +=
                    projected[frame * channels + channel] * gamma[channel];
            }
        }
        Ok(output)
    }

    fn residual_unit(
        &self,
        compute: &Compute,
        input: &[f32],
        channels: usize,
        time: usize,
        dilation: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        let mut hidden =
            self.snake_beta(compute, input, channels, time, &format!("{prefix}.act1"))?;
        hidden = self.causal_conv(
            compute,
            &hidden,
            time,
            channels,
            channels,
            7,
            dilation,
            1,
            &format!("{prefix}.conv1"),
        )?;
        hidden = self.snake_beta(compute, &hidden, channels, time, &format!("{prefix}.act2"))?;
        hidden = self.causal_conv(
            compute,
            &hidden,
            time,
            channels,
            channels,
            1,
            1,
            1,
            &format!("{prefix}.conv2"),
        )?;
        Ok(hidden
            .into_iter()
            .zip(input)
            .map(|(value, residual)| value + residual)
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_conv(
        &self,
        compute: &Compute,
        input: &[f32],
        time: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        if time == 0
            || kernel == 0
            || dilation == 0
            || groups == 0
            || input.len() != input_channels * time
            || input_channels % groups != 0
            || output_channels % groups != 0
        {
            return Err(invalid(format!(
                "causal conv `{prefix}` shape mismatch: input={} channels={input_channels}->{output_channels} time={time} kernel={kernel} dilation={dilation} groups={groups}",
                input.len()
            )));
        }
        let effective_kernel = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid(format!("causal conv `{prefix}` kernel overflow")))?;
        let left_padding = effective_kernel - 1;
        let padded_time = time
            .checked_add(left_padding)
            .ok_or_else(|| invalid(format!("causal conv `{prefix}` padding overflow")))?;
        let mut padded = vec![0.0; input_channels * padded_time];
        for channel in 0..input_channels {
            padded[channel * padded_time + left_padding..(channel + 1) * padded_time]
                .copy_from_slice(&input[channel * time..(channel + 1) * time]);
        }
        let weight_name = format!("{prefix}.conv.weight");
        let weight = if dilation == 1 {
            None
        } else {
            Some(
                self.cached_transform(format!("dilated:{weight_name}:{dilation}"), || {
                    let source = self.tensor(&weight_name)?;
                    let inputs_per_group = input_channels / groups;
                    let mut expanded =
                        vec![0.0; output_channels * inputs_per_group * effective_kernel];
                    for output in 0..output_channels {
                        for input_channel in 0..inputs_per_group {
                            for tap in 0..kernel {
                                let source_offset =
                                    (output * inputs_per_group + input_channel) * kernel + tap;
                                let destination = (output * inputs_per_group + input_channel)
                                    * effective_kernel
                                    + tap * dilation;
                                expanded[destination] = source[source_offset];
                            }
                        }
                    }
                    Ok(expanded)
                })?,
            )
        };
        let direct_weight;
        let weight = if let Some(weight) = weight.as_ref() {
            weight.as_slice()
        } else {
            direct_weight = self.tensor(&weight_name)?;
            direct_weight
        };
        let bias_name = format!("{prefix}.conv.bias");
        let bias = self.tensor(&bias_name)?;
        let mut output = vec![0.0; output_channels * time];
        if groups == 1 {
            compute.conv1d_f32(
                &padded,
                input_channels,
                padded_time,
                weight,
                output_channels,
                effective_kernel,
                Some(bias),
                1,
                0,
                &mut output,
            )?;
        } else {
            compute.grouped_conv1d_f32(
                &padded,
                input_channels,
                padded_time,
                weight,
                output_channels,
                effective_kernel,
                Some(bias),
                1,
                0,
                groups,
                &mut output,
            )?;
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_transposed_conv(
        &self,
        compute: &Compute,
        input: &[f32],
        time: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        if time == 0 || stride == 0 || kernel < stride || input.len() != input_channels * time {
            return Err(invalid(format!(
                "causal ConvTranspose1d `{prefix}` shape mismatch: input={} channels={input_channels}->{output_channels} time={time} kernel={kernel} stride={stride}",
                input.len()
            )));
        }
        let expanded_time = (time - 1)
            .checked_mul(stride)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid(format!("ConvTranspose1d `{prefix}` expansion overflow")))?;
        let mut expanded = vec![0.0; input_channels * expanded_time];
        for channel in 0..input_channels {
            for frame in 0..time {
                expanded[channel * expanded_time + frame * stride] = input[channel * time + frame];
            }
        }
        let source_name = format!("{prefix}.conv.weight");
        let weight = self.cached_transform(format!("transposed:{source_name}"), || {
            let source = self.tensor(&source_name)?;
            let mut flipped = vec![0.0; output_channels * input_channels * kernel];
            for input_channel in 0..input_channels {
                for output_channel in 0..output_channels {
                    for tap in 0..kernel {
                        flipped[(output_channel * input_channels + input_channel) * kernel + tap] =
                            source[(input_channel * output_channels + output_channel) * kernel
                                + (kernel - 1 - tap)];
                    }
                }
            }
            Ok(flipped)
        })?;
        let raw_time = expanded_time
            .checked_add(kernel - 1)
            .ok_or_else(|| invalid(format!("ConvTranspose1d `{prefix}` output overflow")))?;
        let mut raw = vec![0.0; output_channels * raw_time];
        compute.conv1d_f32(
            &expanded,
            input_channels,
            expanded_time,
            &weight,
            output_channels,
            kernel,
            Some(self.tensor(&format!("{prefix}.conv.bias"))?),
            1,
            kernel - 1,
            &mut raw,
        )?;
        let output_time = time
            .checked_mul(stride)
            .ok_or_else(|| invalid(format!("ConvTranspose1d `{prefix}` trim overflow")))?;
        let mut output = vec![0.0; output_channels * output_time];
        for channel in 0..output_channels {
            output[channel * output_time..(channel + 1) * output_time]
                .copy_from_slice(&raw[channel * raw_time..channel * raw_time + output_time]);
        }
        Ok(output)
    }

    fn snake_beta(
        &self,
        compute: &Compute,
        input: &[f32],
        channels: usize,
        time: usize,
        prefix: &str,
    ) -> Result<Vec<f32>> {
        if input.len() != channels * time {
            return Err(invalid(format!(
                "SnakeBeta `{prefix}` input has {}, expected {channels}x{time}",
                input.len()
            )));
        }
        // The official module stores log(alpha) and log(beta).
        let alpha = self
            .tensor(&format!("{prefix}.alpha"))?
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        let beta = self
            .tensor(&format!("{prefix}.beta"))?
            .iter()
            .map(|value| value.exp())
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        compute.snake_beta_f32(input, &alpha, &beta, channels, time, &mut output)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        compute: &Compute,
        input: &[f32],
        rows: usize,
        input_width: usize,
        output_width: usize,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> Result<Vec<f32>> {
        if input.len() != rows * input_width {
            return Err(invalid(format!(
                "linear `{weight_name}` input has {}, expected {rows}x{input_width}",
                input.len()
            )));
        }
        let weight = self.cached_transform(format!("linear:{weight_name}"), || {
            let source = self.tensor(weight_name)?;
            if source.len() != output_width * input_width {
                return Err(invalid(format!(
                    "linear `{weight_name}` has {} values, expected {output_width}x{input_width}",
                    source.len()
                )));
            }
            let mut transposed = vec![0.0; source.len()];
            for output in 0..output_width {
                for input in 0..input_width {
                    transposed[input * output_width + output] =
                        source[output * input_width + input];
                }
            }
            Ok(transposed)
        })?;
        let bias = bias_name.map(|name| self.tensor(name)).transpose()?;
        let mut output = vec![0.0; rows * output_width];
        compute.gemm_f32(
            rows,
            output_width,
            input_width,
            input,
            &weight,
            bias,
            &mut output,
        )?;
        Ok(output)
    }

    fn cached_transform(
        &self,
        key: String,
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<Arc<Vec<f32>>> {
        if let Some(weight) = self
            .inner
            .transformed
            .lock()
            .map_err(|_| invalid("transformed-weight cache mutex poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(weight);
        }
        let built = Arc::new(build()?);
        let mut cache = self
            .inner
            .transformed
            .lock()
            .map_err(|_| invalid("transformed-weight cache mutex poisoned"))?;
        Ok(cache.entry(key).or_insert_with(|| built.clone()).clone())
    }

    fn tensor(&self, name: &str) -> Result<&[f32]> {
        vokra_mmap::tensor_f32_view(&self.inner.file, name).map_err(|error| {
            VokraError::ModelLoad(format!("{LABEL}: tensor `{name}` view failed: {error}"))
        })
    }
}

fn apply_half_split_rope(
    values: &mut [f32],
    frames: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<()> {
    if frames == 0
        || heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
        || values.len() != frames * heads * head_dim
    {
        return Err(invalid("RoPE shape or theta mismatch"));
    }
    let width = heads * head_dim;
    let half = head_dim / 2;
    for frame in 0..frames {
        for head in 0..heads {
            let start = frame * width + head * head_dim;
            for pair in 0..half {
                let exponent = (2 * pair) as f32 / head_dim as f32;
                let angle = frame as f32 / theta.powf(exponent);
                let cosine = angle.cos();
                let sine = angle.sin();
                let first = values[start + pair];
                let second = values[start + half + pair];
                values[start + pair] = first * cosine - second * sine;
                values[start + half + pair] = second * cosine + first * sine;
            }
        }
    }
    Ok(())
}

#[inline]
fn sliding_causal_visible(query: usize, key: usize, window: usize) -> bool {
    key <= query && query - key < window
}

fn frame_to_channel(input: &[f32], frames: usize, channels: usize) -> Result<Vec<f32>> {
    if input.len() != frames * channels {
        return Err(invalid(format!(
            "frame-to-channel transpose has {} values, expected {frames}x{channels}",
            input.len()
        )));
    }
    let mut output = vec![0.0; input.len()];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    Ok(output)
}

fn channel_to_frame(input: &[f32], channels: usize, frames: usize) -> Result<Vec<f32>> {
    if input.len() != channels * frames {
        return Err(invalid(format!(
            "channel-to-frame transpose has {} values, expected {channels}x{frames}",
            input.len()
        )));
    }
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for frame in 0..frames {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    Ok(output)
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid(format!(
            "{label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(format!("{LABEL}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_transposes_round_trip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel = frame_to_channel(&input, 2, 3).unwrap();
        assert_eq!(channel, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(channel_to_frame(&channel, 3, 2).unwrap(), input);
    }

    #[test]
    fn half_split_rope_keeps_position_zero_and_rotates_position_one() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        apply_half_split_rope(&mut values, 2, 1, 4, 10_000.0).unwrap();
        assert_eq!(&values[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_ne!(&values[4..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sliding_causal_window_includes_self_and_exact_window_width() {
        assert!(sliding_causal_visible(100, 100, 72));
        assert!(sliding_causal_visible(100, 29, 72));
        assert!(!sliding_causal_visible(100, 28, 72));
        assert!(!sliding_causal_visible(100, 101, 72));
    }
}
