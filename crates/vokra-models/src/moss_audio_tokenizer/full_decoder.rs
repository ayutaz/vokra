//! Mmap-backed native decoder for the public Full MOSS Audio Tokenizer.
//!
//! The released GGUF is about 7.1 GB of dense F32 weights.  This module keeps
//! the file mapping alive and materialises one Transformer layer (or one LFQ
//! projection) at a time.  It never creates a resident copy of the complete
//! checkpoint.  Every learned reduction is dispatched through [`Compute`],
//! so selecting Metal cannot silently execute a CPU kernel.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufTensorInfo};
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::mapped_weights::{MappedModel, mapped_info, transpose_widen, widen_into};

use super::MOSS_AUDIO_TOKENIZER_FULL_HOT_OPS;

const LABEL: &str = "moss_audio_tokenizer/mapped-decoder";
const CODEBOOK_SIZE: usize = 1_024;
const CODEBOOK_DIM: usize = 8;
const RVQ_DIM: usize = 512;
const LATENT_DIM: usize = 768;
const NUM_QUANTIZERS: usize = 32;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const ROPE_MAX_PERIOD: f32 = 10_000.0;
const ATTENTION_QUERY_TILE: usize = 32;

const MAPPED: MappedModel = MappedModel {
    name: LABEL,
    // Full has no safe resident route on the maintainer-class machine.  The
    // public contract is dense, and a non-dense derivative is rejected rather
    // than widened through a hidden resident/CPU fallback.
    resident_entry: "MossAudioTokenizer::open_mapped",
};

#[derive(Debug, Clone, Copy)]
struct StageSpec {
    module_index: usize,
    input_dim: usize,
    model_dim: usize,
    output_dim: usize,
    ffn_dim: usize,
    layers: usize,
    heads: usize,
    context: usize,
    patch_after: usize,
}

const FULL_STAGE_SPECS: [StageSpec; 4] = [
    StageSpec {
        module_index: 0,
        input_dim: 768,
        model_dim: 1_280,
        output_dim: 1_280,
        ffn_dim: 5_120,
        layers: 32,
        heads: 20,
        context: 125,
        patch_after: 2,
    },
    StageSpec {
        module_index: 2,
        input_dim: 640,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 250,
        patch_after: 2,
    },
    StageSpec {
        module_index: 4,
        input_dim: 384,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 500,
        patch_after: 2,
    },
    StageSpec {
        module_index: 6,
        input_dim: 384,
        model_dim: 768,
        output_dim: 240,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 1_000,
        patch_after: 240,
    },
];

// MOSS-Audio-Tokenizer-v2 starts from a 25 Hz interleaved codec stream.
// The five x2 patch-up stages followed by x240 reconstruct 7,680 interleaved
// samples per codec frame, i.e. 3,840 samples for each stereo channel.
const V2_STAGE_SPECS: [StageSpec; 6] = [
    StageSpec {
        module_index: 0,
        input_dim: 768,
        model_dim: 1_280,
        output_dim: 1_280,
        ffn_dim: 5_120,
        layers: 32,
        heads: 20,
        context: 250,
        patch_after: 2,
    },
    StageSpec {
        module_index: 2,
        input_dim: 640,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 500,
        patch_after: 2,
    },
    StageSpec {
        module_index: 4,
        input_dim: 384,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 800,
        patch_after: 2,
    },
    StageSpec {
        module_index: 6,
        input_dim: 384,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 800,
        patch_after: 2,
    },
    StageSpec {
        module_index: 8,
        input_dim: 384,
        model_dim: 768,
        output_dim: 768,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 800,
        patch_after: 2,
    },
    StageSpec {
        module_index: 10,
        input_dim: 384,
        model_dim: 768,
        output_dim: 240,
        ffn_dim: 3_072,
        layers: 12,
        heads: 12,
        context: 800,
        patch_after: 240,
    },
];

/// Mapping-owning Full decoder.  Tensor payloads remain lazy until decode.
#[derive(Clone)]
pub(super) struct MappedDecoder {
    mapped: Arc<FullMappedDescriptors>,
}

impl std::fmt::Debug for MappedDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedDecoder")
            .field("stages", &self.mapped.stages.len())
            .field("quantizers", &self.mapped.quantizer.codebooks.len())
            .finish()
    }
}

impl MappedDecoder {
    pub(super) fn bind_full(file: Arc<GgufFile>) -> Result<Self> {
        Self::bind(file, &FULL_STAGE_SPECS)
    }

    pub(super) fn bind_v2(file: Arc<GgufFile>) -> Result<Self> {
        Self::bind(file, &V2_STAGE_SPECS)
    }

    fn bind(file: Arc<GgufFile>, stage_specs: &'static [StageSpec]) -> Result<Self> {
        Ok(Self {
            mapped: Arc::new(FullMappedDescriptors::bind(file, stage_specs)?),
        })
    }

    pub(super) fn decode_frame_major(
        &self,
        backend: BackendKind,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        validate_codes(codes, frames, num_quantizers)?;
        let compute = Compute::for_backend(backend, MOSS_AUDIO_TOKENIZER_FULL_HOT_OPS)?;
        let mut values = self.quantizer_decode(&compute, codes, frames, num_quantizers)?;
        let mut time = frames;

        for stage in &self.mapped.stages {
            values = transformer_stage(&compute, self.mapped.file(), &values, time, stage)?;
            (values, time) = patch_up(&values, time, stage.spec.patch_after)?;
        }
        if values.len() != time {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: final decoder has {} values for {time} mono samples",
                values.len()
            )));
        }
        reject_non_finite("decoded waveform", &values)?;
        Ok(values)
    }

    fn quantizer_decode(
        &self,
        compute: &Compute,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        let file = self.mapped.file();
        let quantizer = &self.mapped.quantizer;
        let mut rvq = vec![0.0f32; frames * RVQ_DIM];
        let mut codebook = Vec::new();
        let mut embedded = vec![0.0f32; frames * CODEBOOK_DIM];

        for quantizer_index in 0..num_quantizers {
            widen_descriptor(file, &quantizer.codebooks[quantizer_index], &mut codebook)?;
            for frame in 0..frames {
                let code = codes[frame * num_quantizers + quantizer_index] as usize;
                embedded[frame * CODEBOOK_DIM..(frame + 1) * CODEBOOK_DIM]
                    .copy_from_slice(&codebook[code * CODEBOOK_DIM..(code + 1) * CODEBOOK_DIM]);
            }
            let projection = quantizer.projections[quantizer_index].materialize(file)?;
            let projected = projection.forward(compute, &embedded, frames)?;
            for (sum, value) in rvq.iter_mut().zip(projected) {
                *sum += value;
            }
        }

        quantizer
            .output_projection
            .materialize(file)?
            .forward(compute, &rvq, frames)
    }
}

#[derive(Debug)]
struct FullMappedDescriptors {
    file: Arc<GgufFile>,
    quantizer: QuantizerDescriptors,
    stages: Vec<StageDescriptors>,
}

impl FullMappedDescriptors {
    fn bind(file: Arc<GgufFile>, stage_specs: &'static [StageSpec]) -> Result<Self> {
        let quantizer = QuantizerDescriptors::bind(&file)?;
        let stages = stage_specs
            .iter()
            .copied()
            .map(|spec| StageDescriptors::bind(&file, spec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            file,
            quantizer,
            stages,
        })
    }

    fn file(&self) -> &GgufFile {
        &self.file
    }
}

#[derive(Debug)]
struct QuantizerDescriptors {
    codebooks: Vec<GgufTensorInfo>,
    projections: Vec<WeightNormLinearDescriptor>,
    output_projection: WeightNormLinearDescriptor,
}

impl QuantizerDescriptors {
    fn bind(file: &GgufFile) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);
        let mut projections = Vec::with_capacity(NUM_QUANTIZERS);
        for index in 0..NUM_QUANTIZERS {
            let prefix = format!("quantizer.quantizers.{index}");
            codebooks.push(full_mapped_info(
                file,
                &format!("{prefix}.codebook.weight"),
                CODEBOOK_SIZE * CODEBOOK_DIM,
                MAPPED,
            )?);
            projections.push(WeightNormLinearDescriptor::bind(
                file,
                &format!("{prefix}.out_proj"),
                CODEBOOK_DIM,
                RVQ_DIM,
            )?);
        }
        Ok(Self {
            codebooks,
            projections,
            output_projection: WeightNormLinearDescriptor::bind(
                file,
                "quantizer.output_proj",
                RVQ_DIM,
                LATENT_DIM,
            )?,
        })
    }
}

#[derive(Debug)]
struct StageDescriptors {
    spec: StageSpec,
    input_projection: DenseLinearDescriptor,
    layers: Vec<LayerDescriptors>,
    output_projection: Option<DenseLinearDescriptor>,
}

impl StageDescriptors {
    fn bind(file: &GgufFile, spec: StageSpec) -> Result<Self> {
        let prefix = format!("decoder.{}", spec.module_index);
        let input_projection = DenseLinearDescriptor::bind(
            file,
            &format!("{prefix}.input_proj.weight"),
            spec.input_dim,
            spec.model_dim,
            None,
        )?;
        let layers = (0..spec.layers)
            .map(|index| {
                LayerDescriptors::bind(file, &format!("{prefix}.transformer.layers.{index}"), spec)
            })
            .collect::<Result<Vec<_>>>()?;
        let output_projection = (spec.output_dim != spec.model_dim)
            .then(|| {
                DenseLinearDescriptor::bind(
                    file,
                    &format!("{prefix}.output_proj.weight"),
                    spec.model_dim,
                    spec.output_dim,
                    None,
                )
            })
            .transpose()?;
        Ok(Self {
            spec,
            input_projection,
            layers,
            output_projection,
        })
    }
}

#[derive(Debug)]
struct LayerDescriptors {
    norm1_weight: GgufTensorInfo,
    norm1_bias: GgufTensorInfo,
    attention_in: DenseLinearDescriptor,
    attention_out: DenseLinearDescriptor,
    layer_scale1: GgufTensorInfo,
    norm2_weight: GgufTensorInfo,
    norm2_bias: GgufTensorInfo,
    ffn_in: DenseLinearDescriptor,
    ffn_out: DenseLinearDescriptor,
    layer_scale2: GgufTensorInfo,
}

impl LayerDescriptors {
    fn bind(file: &GgufFile, prefix: &str, spec: StageSpec) -> Result<Self> {
        let vector =
            |suffix: &str| full_mapped_info(file, &format!("{prefix}.{suffix}"), spec.model_dim);
        Ok(Self {
            norm1_weight: vector("norm1.weight")?,
            norm1_bias: vector("norm1.bias")?,
            attention_in: DenseLinearDescriptor::bind(
                file,
                &format!("{prefix}.self_attn.in_projs.0.weight"),
                spec.model_dim,
                spec.model_dim * 3,
                None,
            )?,
            attention_out: DenseLinearDescriptor::bind(
                file,
                &format!("{prefix}.self_attn.out_projs.0.weight"),
                spec.model_dim,
                spec.model_dim,
                None,
            )?,
            layer_scale1: vector("layer_scale_1.scale")?,
            norm2_weight: vector("norm2.weight")?,
            norm2_bias: vector("norm2.bias")?,
            ffn_in: DenseLinearDescriptor::bind(
                file,
                &format!("{prefix}.linear1.weight"),
                spec.model_dim,
                spec.ffn_dim,
                None,
            )?,
            ffn_out: DenseLinearDescriptor::bind(
                file,
                &format!("{prefix}.linear2.weight"),
                spec.ffn_dim,
                spec.model_dim,
                None,
            )?,
            layer_scale2: vector("layer_scale_2.scale")?,
        })
    }

    fn materialize(&self, file: &GgufFile) -> Result<MaterializedLayer> {
        let mut norm1_weight = Vec::new();
        let mut norm1_bias = Vec::new();
        let mut layer_scale1 = Vec::new();
        let mut norm2_weight = Vec::new();
        let mut norm2_bias = Vec::new();
        let mut layer_scale2 = Vec::new();
        widen_descriptor(file, &self.norm1_weight, &mut norm1_weight)?;
        widen_descriptor(file, &self.norm1_bias, &mut norm1_bias)?;
        widen_descriptor(file, &self.layer_scale1, &mut layer_scale1)?;
        widen_descriptor(file, &self.norm2_weight, &mut norm2_weight)?;
        widen_descriptor(file, &self.norm2_bias, &mut norm2_bias)?;
        widen_descriptor(file, &self.layer_scale2, &mut layer_scale2)?;
        Ok(MaterializedLayer {
            norm1_weight,
            norm1_bias,
            attention_in: self.attention_in.materialize(file)?,
            attention_out: self.attention_out.materialize(file)?,
            layer_scale1,
            norm2_weight,
            norm2_bias,
            ffn_in: self.ffn_in.materialize(file)?,
            ffn_out: self.ffn_out.materialize(file)?,
            layer_scale2,
        })
    }
}

#[derive(Debug)]
struct DenseLinearDescriptor {
    weight: GgufTensorInfo,
    bias: Option<GgufTensorInfo>,
    input: usize,
    output: usize,
}

impl DenseLinearDescriptor {
    fn bind(
        file: &GgufFile,
        weight_name: &str,
        input: usize,
        output: usize,
        bias_name: Option<&str>,
    ) -> Result<Self> {
        Ok(Self {
            weight: full_mapped_info(file, weight_name, input * output)?,
            bias: bias_name
                .map(|name| full_mapped_info(file, name, output))
                .transpose()?,
            input,
            output,
        })
    }

    fn materialize(&self, file: &GgufFile) -> Result<Linear> {
        let mut weight_t = Vec::new();
        transpose_widen(
            file.tensor_bytes(&self.weight),
            self.weight.dtype,
            self.output,
            self.input,
            &mut weight_t,
            MAPPED,
        )?;
        let bias = self
            .bias
            .as_ref()
            .map(|info| {
                let mut values = Vec::new();
                widen_descriptor(file, info, &mut values)?;
                Ok::<Vec<f32>, VokraError>(values)
            })
            .transpose()?;
        Ok(Linear {
            input: self.input,
            output: self.output,
            weight_t,
            bias,
        })
    }
}

#[derive(Debug)]
struct WeightNormLinearDescriptor {
    magnitude: GgufTensorInfo,
    direction: GgufTensorInfo,
    bias: GgufTensorInfo,
    input: usize,
    output: usize,
}

impl WeightNormLinearDescriptor {
    fn bind(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            magnitude: full_mapped_info(
                file,
                &format!("{prefix}.parametrizations.weight.original0"),
                output,
            )?,
            direction: full_mapped_info(
                file,
                &format!("{prefix}.parametrizations.weight.original1"),
                output * input,
            )?,
            bias: full_mapped_info(file, &format!("{prefix}.bias"), output)?,
            input,
            output,
        })
    }

    fn materialize(&self, file: &GgufFile) -> Result<Linear> {
        let mut magnitude = Vec::new();
        let mut direction = Vec::new();
        let mut bias = Vec::new();
        widen_descriptor(file, &self.magnitude, &mut magnitude)?;
        widen_descriptor(file, &self.direction, &mut direction)?;
        widen_descriptor(file, &self.bias, &mut bias)?;
        let weight_t = fold_weight_norm_transposed(
            &magnitude,
            &direction,
            self.input,
            self.output,
            &self.direction.name,
        )?;
        Ok(Linear {
            input: self.input,
            output: self.output,
            weight_t,
            bias: Some(bias),
        })
    }
}

#[derive(Debug)]
struct Linear {
    input: usize,
    output: usize,
    /// Row-major `[input, output]`, ready for `Compute::gemm_f32`.
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn forward(&self, compute: &Compute, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        let expected = rows.checked_mul(self.input).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: linear input shape overflows: {rows} x {}",
                self.input
            ))
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: linear input has {} values, expected {rows} x {} = {expected}",
                input.len(),
                self.input
            )));
        }
        let output_len = rows.checked_mul(self.output).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: linear output shape overflows: {rows} x {}",
                self.output
            ))
        })?;
        let mut output = vec![0.0f32; output_len];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight_t,
            self.bias.as_deref(),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug)]
struct MaterializedLayer {
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    attention_in: Linear,
    attention_out: Linear,
    layer_scale1: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    ffn_in: Linear,
    ffn_out: Linear,
    layer_scale2: Vec<f32>,
}

fn transformer_stage(
    compute: &Compute,
    file: &GgufFile,
    input: &[f32],
    time: usize,
    stage: &StageDescriptors,
) -> Result<Vec<f32>> {
    let mut hidden = stage
        .input_projection
        .materialize(file)?
        .forward(compute, input, time)?;
    for descriptor in &stage.layers {
        let layer = descriptor.materialize(file)?;
        let mut normalized = vec![0.0f32; hidden.len()];
        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            time,
            stage.spec.model_dim,
            &layer.norm1_weight,
            &layer.norm1_bias,
            LAYER_NORM_EPS,
        )?;
        let attention = attention(
            compute,
            &normalized,
            time,
            stage.spec.model_dim,
            stage.spec.heads,
            stage.spec.context,
            &layer.attention_in,
            &layer.attention_out,
        )?;
        add_scaled_residual(
            &mut hidden,
            &attention,
            &layer.layer_scale1,
            stage.spec.model_dim,
        )?;

        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            time,
            stage.spec.model_dim,
            &layer.norm2_weight,
            &layer.norm2_bias,
            LAYER_NORM_EPS,
        )?;
        let feed_forward = layer.ffn_in.forward(compute, &normalized, time)?;
        let mut activated = vec![0.0f32; feed_forward.len()];
        compute.gelu_f32(&feed_forward, &mut activated)?;
        let feed_forward = layer.ffn_out.forward(compute, &activated, time)?;
        add_scaled_residual(
            &mut hidden,
            &feed_forward,
            &layer.layer_scale2,
            stage.spec.model_dim,
        )?;
    }
    match &stage.output_projection {
        Some(projection) => projection
            .materialize(file)?
            .forward(compute, &hidden, time),
        None => Ok(hidden),
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    input: &[f32],
    time: usize,
    model_dim: usize,
    heads: usize,
    context: usize,
    input_projection: &Linear,
    output_projection: &Linear,
) -> Result<Vec<f32>> {
    if heads == 0 || !model_dim.is_multiple_of(heads) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention model_dim {model_dim} is not divisible by heads {heads}"
        )));
    }
    let head_dim = model_dim / heads;
    let projected = input_projection.forward(compute, input, time)?;
    let mut joined = vec![0.0f32; time * model_dim];
    let scale = (head_dim as f32).sqrt().recip();

    for head in 0..heads {
        let mut query = vec![0.0f32; time * head_dim];
        let mut key = vec![0.0f32; time * head_dim];
        let mut value = vec![0.0f32; time * head_dim];
        for position in 0..time {
            let source = position * model_dim * 3 + head * head_dim;
            let target = position * head_dim;
            query[target..target + head_dim].copy_from_slice(&projected[source..source + head_dim]);
            key[target..target + head_dim]
                .copy_from_slice(&projected[source + model_dim..source + model_dim + head_dim]);
            value[target..target + head_dim].copy_from_slice(
                &projected[source + model_dim * 2..source + model_dim * 2 + head_dim],
            );
        }
        apply_rope(&mut query, &mut key, time, head_dim)?;

        for query_start in (0..time).step_by(ATTENTION_QUERY_TILE) {
            let query_end = (query_start + ATTENTION_QUERY_TILE).min(time);
            let query_rows = query_end - query_start;
            let key_start = query_start.saturating_add(1).saturating_sub(context);
            let key_end = query_end;
            let key_rows = key_end - key_start;

            let mut key_t = vec![0.0f32; head_dim * key_rows];
            for key_position in key_start..key_end {
                let relative_key = key_position - key_start;
                for dimension in 0..head_dim {
                    key_t[dimension * key_rows + relative_key] =
                        key[key_position * head_dim + dimension];
                }
            }
            let query_slice = &query[query_start * head_dim..query_end * head_dim];
            let mut logits = vec![0.0f32; query_rows * key_rows];
            compute.gemm_f32(
                query_rows,
                key_rows,
                head_dim,
                query_slice,
                &key_t,
                None,
                &mut logits,
            )?;
            apply_local_causal_mask(
                &mut logits,
                query_start,
                query_rows,
                key_start,
                key_rows,
                context,
                scale,
            )?;
            let mut probabilities = vec![0.0f32; logits.len()];
            compute.softmax_f32(&logits, &mut probabilities, query_rows, key_rows)?;
            let value_slice = &value[key_start * head_dim..key_end * head_dim];
            let mut attended = vec![0.0f32; query_rows * head_dim];
            compute.gemm_f32(
                query_rows,
                head_dim,
                key_rows,
                &probabilities,
                value_slice,
                None,
                &mut attended,
            )?;
            for relative_query in 0..query_rows {
                let position = query_start + relative_query;
                let source = relative_query * head_dim;
                let target = position * model_dim + head * head_dim;
                joined[target..target + head_dim]
                    .copy_from_slice(&attended[source..source + head_dim]);
            }
        }
    }
    output_projection.forward(compute, &joined, time)
}

fn patch_up(input: &[f32], time: usize, patch: usize) -> Result<(Vec<f32>, usize)> {
    if time == 0 || patch == 0 || !input.len().is_multiple_of(time) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up shape mismatch: values={}, time={time}, patch={patch}",
            input.len()
        )));
    }
    let packed_channels = input.len() / time;
    if !packed_channels.is_multiple_of(patch) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up channels {packed_channels} are not divisible by patch {patch}"
        )));
    }
    let channels = packed_channels / patch;
    let output_time = time.checked_mul(patch).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up time overflows: {time} * {patch}"
        ))
    })?;
    let output_len = output_time.checked_mul(channels).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up output shape overflows: {output_time} * {channels}"
        ))
    })?;
    let mut output = vec![0.0f32; output_len];
    for position in 0..time {
        for channel in 0..channels {
            for subframe in 0..patch {
                output[(position * patch + subframe) * channels + channel] =
                    input[position * packed_channels + channel * patch + subframe];
            }
        }
    }
    Ok((output, output_time))
}

fn apply_rope(query: &mut [f32], key: &mut [f32], time: usize, dim: usize) -> Result<()> {
    if dim == 0 || !dim.is_multiple_of(2) || query.len() != time * dim || key.len() != query.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: RoPE shape mismatch: q={}, k={}, time={time}, dim={dim}",
            query.len(),
            key.len()
        )));
    }
    for position in 0..time {
        for pair in 0..dim / 2 {
            let frequency = ((pair as f32) * (-ROPE_MAX_PERIOD.ln() * 2.0 / dim as f32)).exp();
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let index = position * dim + pair * 2;
            for values in [&mut *query, &mut *key] {
                let real = values[index];
                let imaginary = values[index + 1];
                values[index] = real * cos - imaginary * sin;
                values[index + 1] = real * sin + imaginary * cos;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_local_causal_mask(
    logits: &mut [f32],
    query_start: usize,
    query_rows: usize,
    key_start: usize,
    key_rows: usize,
    context: usize,
    scale: f32,
) -> Result<()> {
    if query_rows == 0
        || key_rows == 0
        || context == 0
        || logits.len() != query_rows * key_rows
        || !scale.is_finite()
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention-mask shape mismatch: logits={}, queries={query_rows}, keys={key_rows}, context={context}, scale={scale}",
            logits.len()
        )));
    }
    for query in 0..query_rows {
        let absolute_query = query_start + query;
        for key in 0..key_rows {
            let absolute_key = key_start + key;
            let value = &mut logits[query * key_rows + key];
            if absolute_key > absolute_query || absolute_query - absolute_key >= context {
                *value = f32::NEG_INFINITY;
            } else {
                *value *= scale;
            }
        }
    }
    Ok(())
}

fn add_scaled_residual(
    residual: &mut [f32],
    update: &[f32],
    scale: &[f32],
    dim: usize,
) -> Result<()> {
    if residual.len() != update.len()
        || dim == 0
        || !residual.len().is_multiple_of(dim)
        || scale.len() != dim
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: scaled residual shape mismatch: residual={}, update={}, scale={}, dim={dim}",
            residual.len(),
            update.len(),
            scale.len()
        )));
    }
    for (index, (residual, update)) in residual.iter_mut().zip(update).enumerate() {
        *residual += update * scale[index % dim];
    }
    Ok(())
}

fn fold_weight_norm_transposed(
    magnitude: &[f32],
    direction: &[f32],
    input: usize,
    output: usize,
    label: &str,
) -> Result<Vec<f32>> {
    if magnitude.len() != output || direction.len() != output * input {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: weight norm `{label}` shape mismatch: magnitude={}, direction={}, expected {output} and {}",
            magnitude.len(),
            direction.len(),
            output * input
        )));
    }
    let mut weight_t = vec![0.0f32; direction.len()];
    for out in 0..output {
        let row = &direction[out * input..(out + 1) * input];
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !magnitude[out].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: weight norm `{label}` row {out} has invalid magnitude {} or direction norm {norm}",
                magnitude[out]
            )));
        }
        let factor = magnitude[out] / norm;
        for inner in 0..input {
            weight_t[inner * output + out] = row[inner] * factor;
        }
    }
    Ok(weight_t)
}

fn widen_descriptor(file: &GgufFile, info: &GgufTensorInfo, output: &mut Vec<f32>) -> Result<()> {
    widen_into(file.tensor_bytes(info), info.dtype, output, MAPPED)
}

fn full_mapped_info(file: &GgufFile, name: &str, elements: usize) -> Result<GgufTensorInfo> {
    if let Some(info) = file.tensor_info(name)
        && !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` uses unsupported mapped dtype {:?}; the Full bounded-memory runtime accepts dense F32, F16 or BF16 only and has no resident or CPU fallback",
            info.dtype
        )));
    }
    mapped_info(file, name, elements, MAPPED)
}

fn validate_codes(codes: &[u32], frames: usize, num_quantizers: usize) -> Result<()> {
    if frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: frames must be > 0"
        )));
    }
    if !(1..=NUM_QUANTIZERS).contains(&num_quantizers) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: num_quantizers {num_quantizers} is outside 1..={NUM_QUANTIZERS}"
        )));
    }
    let expected = frames.checked_mul(num_quantizers).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: frames * num_quantizers overflows: {frames} * {num_quantizers}"
        ))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes.len() {} != frames {frames} * num_quantizers {num_quantizers} = {expected}",
            codes.len()
        )));
    }
    if let Some((index, code)) = codes
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| *code as usize >= CODEBOOK_SIZE)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
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
            "{LABEL}: {label}[{index}] is not finite ({value})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_stage_contract_reconstructs_1920_samples_per_frame() {
        assert_eq!(
            FULL_STAGE_SPECS
                .iter()
                .map(|stage| stage.patch_after)
                .product::<usize>(),
            1_920
        );
        assert_eq!(
            FULL_STAGE_SPECS
                .iter()
                .map(|stage| stage.layers)
                .sum::<usize>(),
            68
        );
    }

    #[test]
    fn v2_stage_contract_reconstructs_stereo_3840_samples_per_frame() {
        assert_eq!(
            V2_STAGE_SPECS
                .iter()
                .map(|stage| stage.patch_after)
                .product::<usize>(),
            7_680
        );
        assert_eq!(
            V2_STAGE_SPECS
                .iter()
                .map(|stage| stage.layers)
                .sum::<usize>(),
            92
        );
        assert_eq!(
            V2_STAGE_SPECS
                .iter()
                .map(|stage| stage.module_index)
                .collect::<Vec<_>>(),
            vec![0, 2, 4, 6, 8, 10]
        );
        assert_eq!(
            V2_STAGE_SPECS
                .iter()
                .map(|stage| stage.context)
                .collect::<Vec<_>>(),
            vec![250, 500, 800, 800, 800, 800]
        );
    }

    #[test]
    fn tiled_mask_matches_local_causal_window() {
        let mut logits = vec![1.0f32; 3 * 5];
        apply_local_causal_mask(&mut logits, 4, 3, 2, 5, 3, 0.5).unwrap();
        // Absolute query 4 can see keys 2,3,4 and not 5,6.
        assert_eq!(
            &logits[0..5],
            &[0.5, 0.5, 0.5, f32::NEG_INFINITY, f32::NEG_INFINITY]
        );
        // Absolute query 6 can see keys 4,5,6 and not stale keys 2,3.
        assert_eq!(
            &logits[10..15],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.5, 0.5, 0.5]
        );
    }

    #[test]
    fn patch_up_matches_upstream_reshape_permute() {
        let (output, time) = patch_up(&[1.0, 2.0, 3.0, 4.0], 1, 2).unwrap();
        assert_eq!(time, 2);
        assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn weight_norm_folds_and_transposes_rows() {
        let weight =
            fold_weight_norm_transposed(&[5.0, 13.0], &[3.0, 4.0, 5.0, 12.0], 2, 2, "x").unwrap();
        assert_eq!(weight, vec![3.0, 5.0, 4.0, 12.0]);
    }
}
