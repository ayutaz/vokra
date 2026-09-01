//! Native, checkpoint-independent FireRed AED building blocks.
//!
//! The loader deliberately keeps checkpoint name binding in `mod.rs`; these
//! helpers only implement descriptor-bound tensor semantics once the strict
//! model constructor supplies the operands. They dispatch every learned operation via
//! [`Compute`], so CPU and Metal use the same first-class seam and unsupported
//! backends return an error before execution.

use crate::compute::{Compute, HotOp};
use std::collections::BTreeMap;
use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

/// Hot operations required by the FireRed AED encoder and decoder paths.
/// The list is consumed by a model entry point before any work;
/// it intentionally contains no fallback-only operation.
pub const FIRERED_ASR_AED_HOT_OPS: &[HotOp] = &[
    HotOp::Conv2d,
    HotOp::Conv1d,
    HotOp::Gemm,
    HotOp::Gelu,
    HotOp::LayerNorm,
    HotOp::Relu,
    HotOp::Silu,
    HotOp::Softmax,
];

/// Owned, source-bound FireRed tensors in the layouts consumed by the native
/// kernels.
///
/// This is deliberately separate from the inspection binder in `mod.rs`:
/// loading a tensor table is an opt-in operation, and callers that only need
/// to inspect provenance or manifests do not pay the multi-gigabyte decode
/// cost. The constructor accepts only the complete, compiled 940-tensor
/// descriptor and rejects every missing, extra, non-F32, or non-finite
/// value before returning a handle. The public model constructor additionally
/// requires the exact converter provenance; this type is crate-private so it
/// cannot bypass that gate. No synthesized tensor is ever installed.
#[derive(Debug)]
pub(crate) struct FireRedRuntimeWeights {
    tensors: BTreeMap<String, Vec<f32>>,
}

impl FireRedRuntimeWeights {
    /// Decodes the exact 940-tensor release into owned f32 values.
    ///
    /// Linear tensors are transposed once from the upstream PyTorch
    /// `[out,in]` layout to the row-major Compute `[in,out]` layout. Conv
    /// tensors and embeddings retain their source layout. This method is
    /// intended for VAST/production hosts; tests should use the primitive
    /// native APIs with small synthetic operands instead.
    pub(crate) fn from_gguf(file: &GgufFile) -> Result<Self> {
        let encoder = super::expected_encoder_tensor_specs();
        let decoder = super::expected_decoder_tensor_specs();
        if file.tensors().len() != encoder.len() + decoder.len() {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l native operand bind requires {} tensors, got {}",
                encoder.len() + decoder.len(),
                file.tensors().len()
            )));
        }
        let mut expected = BTreeMap::new();
        for spec in encoder.iter() {
            expected.insert(
                spec.name.as_str(),
                (
                    &spec.source_shape,
                    encoder_runtime_layout(spec.name.as_str(), spec.native_layout),
                ),
            );
        }
        for spec in decoder.iter() {
            if expected
                .insert(
                    spec.name.as_str(),
                    (&spec.source_shape, decoder_layout(spec.native_layout)),
                )
                .is_some()
            {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand bind has duplicate descriptor `{}`",
                    spec.name
                )));
            }
        }
        if expected.len() != file.tensors().len() {
            return Err(VokraError::ModelLoad(
                "firered-asr-aed-l native operand descriptor count is not unique".to_owned(),
            ));
        }
        let mut tensors = BTreeMap::new();
        for info in file.tensors() {
            let Some((shape, layout)) = expected.get(info.name.as_str()).copied() else {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand bind found unexpected tensor `{}`",
                    info.name
                )));
            };
            if info.dtype != GgmlType::F32 {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand tensor `{}` has dtype {:?}, expected F32",
                    info.name, info.dtype
                )));
            }
            if info.dimensions.as_slice() != shape.as_slice() {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand tensor `{}` shape {:?}, expected {:?}",
                    info.name, info.dimensions, shape
                )));
            }
            let values = file.tensor_f32(&info.name).map_err(|error| {
                VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand tensor `{}` decode failed: {error}",
                    info.name
                ))
            })?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l native operand tensor `{}` contains a non-finite value",
                    info.name
                )));
            }
            let values = match layout {
                NativeLayout::Transpose2d => transpose_2d(&values, shape)?,
                NativeLayout::Direct => values,
            };
            tensors.insert(info.name.clone(), values);
        }
        if tensors.len() != file.tensors().len() {
            return Err(VokraError::ModelLoad(
                "firered-asr-aed-l native operand bind found duplicate tensor names".to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Borrows a source-name tensor after its load-time layout conversion.
    pub(crate) fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors.get(name).map(Vec::as_slice).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "firered-asr-aed-l native operand tensor `{name}` is absent"
            ))
        })
    }

    /// Runs the descriptor-bound encoder stem and all sixteen Conformer blocks
    /// on already-extracted `[frames, 80]` fbank/CMVN features.
    ///
    /// The upstream fbank implementation is Python/native-fbank code and is
    /// intentionally not linked into the runtime. Consequently this method
    /// takes the source feature matrix, not PCM. Padding masks are accepted
    /// only when every input frame is valid; accepting a guessed downsampling
    /// mask would create a silent alignment bug.
    pub(crate) fn encode_features(
        &self,
        compute: &Compute,
        features: &[f32],
        frames: usize,
        input_mask: &[bool],
    ) -> Result<Vec<f32>> {
        if frames < 7
            || features.len()
                != frames
                    .checked_mul(super::AUTHENTICATED_N_MELS as usize)
                    .ok_or_else(|| {
                        VokraError::InvalidArgument("FireRed fbank shape overflow".to_owned())
                    })?
            || input_mask.len() != frames
            || !input_mask.iter().all(|&valid| valid)
        {
            return Err(VokraError::InvalidArgument(
                "firered-asr-aed-l encoder expects [frames,80] features and an all-valid input mask"
                    .to_owned(),
            ));
        }
        let stem = FireRedConv2dSubsampling {
            out_channels: 32,
            d_model: super::AUTHENTICATED_ENCODER_D_MODEL as usize,
        };
        let conv0_w = self.tensor("encoder.input_preprocessor.conv.0.weight")?;
        let conv0_b = self.tensor("encoder.input_preprocessor.conv.0.bias")?;
        let conv1_w = self.tensor("encoder.input_preprocessor.conv.2.weight")?;
        let conv1_b = self.tensor("encoder.input_preprocessor.conv.2.bias")?;
        let out_w = self.tensor("encoder.input_preprocessor.out.weight")?;
        let out_b = self.tensor("encoder.input_preprocessor.out.bias")?;
        let (hidden, stem_frames) = stem.forward(
            compute,
            features,
            frames,
            super::AUTHENTICATED_N_MELS as usize,
            conv0_w,
            conv0_b,
            conv1_w,
            conv1_b,
            out_w,
            out_b,
        )?;
        let mask = vec![true; stem_frames];
        let positions = self.tensor("encoder.positional_encoding.pe")?;
        let position_width = super::AUTHENTICATED_ENCODER_D_MODEL as usize;
        let position_count = stem_frames
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed encoder position count overflow".to_owned())
            })?;
        let needed = position_count.checked_mul(position_width).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed encoder position shape overflow".to_owned())
        })?;
        let table_frames = positions.len().checked_div(position_width).ok_or_else(|| {
            VokraError::ModelLoad(
                "firered-asr-aed-l positional encoding width is invalid".to_owned(),
            )
        })?;
        let table_values = table_frames.checked_mul(position_width).ok_or_else(|| {
            VokraError::ModelLoad("firered-asr-aed-l positional table overflows".to_owned())
        })?;
        if table_frames % 2 != 1 || positions.len() != table_values {
            return Err(VokraError::ModelLoad(
                "firered-asr-aed-l positional encoding table must be an odd [2*max-1,d] buffer"
                    .to_owned(),
            ));
        }
        let center = table_frames / 2;
        let start = center
            .checked_add(1)
            .and_then(|v| v.checked_sub(stem_frames))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "FireRed encoder position window underflows authenticated table".to_owned(),
                )
            })?;
        let end = start.checked_add(position_count).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed encoder position window overflows".to_owned())
        })?;
        if end > table_frames {
            return Err(VokraError::ModelLoad(
                "firered-asr-aed-l positional encoding is shorter than the source window"
                    .to_owned(),
            ));
        }
        // The source buffer is [1, 9999, d_model] (the leading singleton is
        // absent from the flattened GGUF payload). RelPositionalEncoding
        // selects the centered [2*T-1, d_model] window for each input length.
        let begin = start.checked_mul(position_width).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed encoder position offset overflows".to_owned())
        })?;
        let end_offset = begin.checked_add(needed).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed encoder position end overflows".to_owned())
        })?;
        let positions = &positions[begin..end_offset];
        let mut layers = Vec::with_capacity(super::AUTHENTICATED_ENCODER_N_LAYER as usize);
        for layer in 0..super::AUTHENTICATED_ENCODER_N_LAYER {
            let prefix = format!("encoder.layer_stack.{layer}.");
            layers.push(FireRedConformerBlockWeights {
                ffn1_ln_gamma: self.tensor(&format!("{prefix}ffn1.net.0.weight"))?,
                ffn1_ln_beta: self.tensor(&format!("{prefix}ffn1.net.0.bias"))?,
                ffn1_expand_w_t: self.tensor(&format!("{prefix}ffn1.net.1.weight"))?,
                ffn1_expand_b: self.tensor(&format!("{prefix}ffn1.net.1.bias"))?,
                ffn1_project_w_t: self.tensor(&format!("{prefix}ffn1.net.4.weight"))?,
                ffn1_project_b: self.tensor(&format!("{prefix}ffn1.net.4.bias"))?,
                attention_positions: positions,
                attention_q_w_t: self.tensor(&format!("{prefix}mhsa.w_qs.weight"))?,
                attention_k_w_t: self.tensor(&format!("{prefix}mhsa.w_ks.weight"))?,
                attention_v_w_t: self.tensor(&format!("{prefix}mhsa.w_vs.weight"))?,
                attention_linear_pos_w_t: self
                    .tensor(&format!("{prefix}mhsa.linear_pos.weight"))?,
                attention_q_norm_gamma: self
                    .tensor(&format!("{prefix}mhsa.layer_norm_q.weight"))?,
                attention_q_norm_beta: self.tensor(&format!("{prefix}mhsa.layer_norm_q.bias"))?,
                attention_k_norm_gamma: self
                    .tensor(&format!("{prefix}mhsa.layer_norm_k.weight"))?,
                attention_k_norm_beta: self.tensor(&format!("{prefix}mhsa.layer_norm_k.bias"))?,
                attention_v_norm_gamma: self
                    .tensor(&format!("{prefix}mhsa.layer_norm_v.weight"))?,
                attention_v_norm_beta: self.tensor(&format!("{prefix}mhsa.layer_norm_v.bias"))?,
                attention_bias_u: self.tensor(&format!("{prefix}mhsa.pos_bias_u"))?,
                attention_bias_v: self.tensor(&format!("{prefix}mhsa.pos_bias_v"))?,
                attention_output_w_t: self.tensor(&format!("{prefix}mhsa.fc.weight"))?,
                conv_pointwise_in_w: self
                    .tensor(&format!("{prefix}conv.pointwise_conv1.weight"))?,
                conv_depthwise_w: self.tensor(&format!("{prefix}conv.depthwise_conv.weight"))?,
                conv_depthwise_ln_gamma: self.tensor(&format!("{prefix}conv.batch_norm.weight"))?,
                conv_depthwise_ln_beta: self.tensor(&format!("{prefix}conv.batch_norm.bias"))?,
                conv_pointwise_out_w: self
                    .tensor(&format!("{prefix}conv.pointwise_conv2.weight"))?,
                conv_pre_ln_gamma: self.tensor(&format!("{prefix}conv.pre_layer_norm.weight"))?,
                conv_pre_ln_beta: self.tensor(&format!("{prefix}conv.pre_layer_norm.bias"))?,
                ffn2_ln_gamma: self.tensor(&format!("{prefix}ffn2.net.0.weight"))?,
                ffn2_ln_beta: self.tensor(&format!("{prefix}ffn2.net.0.bias"))?,
                ffn2_expand_w_t: self.tensor(&format!("{prefix}ffn2.net.1.weight"))?,
                ffn2_expand_b: self.tensor(&format!("{prefix}ffn2.net.1.bias"))?,
                ffn2_project_w_t: self.tensor(&format!("{prefix}ffn2.net.4.weight"))?,
                ffn2_project_b: self.tensor(&format!("{prefix}ffn2.net.4.bias"))?,
                final_ln_gamma: self.tensor(&format!("{prefix}layer_norm.weight"))?,
                final_ln_beta: self.tensor(&format!("{prefix}layer_norm.bias"))?,
            });
        }
        FireRedConformerEncoder::authenticated().forward(
            compute,
            &hidden,
            stem_frames,
            &mask,
            &layers,
        )
    }

    /// Runs the descriptor-bound incremental decoder on encoder memory.
    ///
    /// This is a feature-to-token primitive, not a PCM transcription API:
    /// the caller must provide memory produced by [`Self::encode_features`]
    /// and the exact special-token ids from the bound checkpoint metadata.
    /// The returned vector contains generated ids (the supplied SOS id is
    /// not included) and includes EOS when it is selected. Beam search is
    /// deliberately not hidden behind this method; the pinned upstream
    /// search policy still needs an independent trace before it can be
    /// reproduced without guessing.
    pub(crate) fn decode_greedy(
        &self,
        compute: &Compute,
        memory: &[f32],
        source_frames: usize,
        source_mask: &[bool],
        sos_id: usize,
        eos_id: usize,
        max_len: usize,
    ) -> Result<Vec<usize>> {
        let d_model = super::AUTHENTICATED_DECODER_D_MODEL as usize;
        let inner_dim = super::AUTHENTICATED_DECODER_FFN_DIM as usize;
        let n_head = super::AUTHENTICATED_DECODER_N_HEAD as usize;
        let vocab_size = super::AUTHENTICATED_DECODER_VOCAB_SIZE as usize;
        let max_positions = super::AUTHENTICATED_DECODER_MAX_POSITIONS as usize;
        if source_frames == 0
            || memory.len()
                != source_frames.checked_mul(d_model).ok_or_else(|| {
                    VokraError::InvalidArgument("FireRed decoder memory shape overflow".to_owned())
                })?
            || source_mask.len() != source_frames
            || !source_mask.iter().any(|&valid| valid)
            || sos_id >= vocab_size
            || eos_id >= vocab_size
            || max_len == 0
            || max_len > max_positions
            || !all_finite(&[memory])
        {
            return Err(VokraError::InvalidArgument(
                "firered-asr-aed-l greedy decoder memory, ids, or length is invalid".to_owned(),
            ));
        }
        let embedding = self.tensor("decoder.tgt_word_emb.weight")?;
        let positional = self.tensor("decoder.positional_encoding.pe")?;
        let output_norm_gamma = self.tensor("decoder.layer_norm_out.weight")?;
        let output_norm_beta = self.tensor("decoder.layer_norm_out.bias")?;
        let projection = self.tensor("decoder.tgt_word_prj.weight")?;
        let embedding_op = FireRedDecoderEmbedding {
            vocab_size,
            d_model,
            max_positions,
        };
        let output_head = FireRedDecoderOutputHead {
            d_model,
            vocab_size,
        };
        let layer_op = FireRedDecoderLayer {
            d_model,
            inner_dim,
            n_head,
            source_dim: d_model,
        };
        let mut caches = vec![
            (Vec::<f32>::new(), Vec::<f32>::new());
            super::AUTHENTICATED_DECODER_N_LAYER as usize
        ];
        let mut cache_masks =
            vec![Vec::<bool>::new(); super::AUTHENTICATED_DECODER_N_LAYER as usize];
        let mut previous = sos_id;
        let mut generated = Vec::with_capacity(max_len);
        for step in 0..max_len {
            let mut hidden = embedding_op.forward(&[previous], &[step], embedding, positional)?;
            for layer in 0..super::AUTHENTICATED_DECODER_N_LAYER as usize {
                let weights = self.decoder_layer_weights(layer)?;
                let query_mask = [true];
                let output = layer_op.forward(
                    compute,
                    &hidden,
                    1,
                    &query_mask,
                    memory,
                    source_frames,
                    source_mask,
                    &caches[layer].0,
                    &caches[layer].1,
                    &cache_masks[layer],
                    weights,
                )?;
                hidden = output.output;
                caches[layer] = (output.key_cache, output.value_cache);
                cache_masks[layer].push(true);
            }
            let logits = output_head.forward(
                compute,
                &hidden,
                1,
                output_norm_gamma,
                output_norm_beta,
                projection,
            )?;
            let next = logits
                .chunks_exact(vocab_size)
                .next()
                .and_then(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(left_id, left), (right_id, right)| {
                            left.partial_cmp(right)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| right_id.cmp(left_id))
                        })
                        .map(|(id, _)| id)
                })
                .ok_or_else(|| {
                    VokraError::ModelLoad(
                        "firered-asr-aed-l decoder produced no vocabulary logits".to_owned(),
                    )
                })?;
            generated.push(next);
            previous = next;
            if next == eos_id {
                break;
            }
        }
        Ok(generated)
    }

    fn decoder_layer_weights(&self, layer: usize) -> Result<FireRedDecoderLayerWeights<'_>> {
        if layer >= super::AUTHENTICATED_DECODER_N_LAYER as usize {
            return Err(VokraError::InvalidArgument(format!(
                "firered-asr-aed-l decoder layer {layer} is outside the authenticated stack"
            )));
        }
        let prefix = format!("decoder.layer_stack.{layer}.");
        Ok(FireRedDecoderLayerWeights {
            self_norm_gamma: self.tensor(&format!("{prefix}self_attn_norm.weight"))?,
            self_norm_beta: self.tensor(&format!("{prefix}self_attn_norm.bias"))?,
            self_attention: FireRedDecoderSelfAttentionWeights {
                q_w_t: self.tensor(&format!("{prefix}self_attn.w_qs.weight"))?,
                q_b: self.tensor(&format!("{prefix}self_attn.w_qs.bias"))?,
                k_w_t: self.tensor(&format!("{prefix}self_attn.w_ks.weight"))?,
                v_w_t: self.tensor(&format!("{prefix}self_attn.w_vs.weight"))?,
                v_b: self.tensor(&format!("{prefix}self_attn.w_vs.bias"))?,
                output_w_t: self.tensor(&format!("{prefix}self_attn.fc.weight"))?,
                output_b: self.tensor(&format!("{prefix}self_attn.fc.bias"))?,
            },
            cross_norm_gamma: self.tensor(&format!("{prefix}cross_attn_norm.weight"))?,
            cross_norm_beta: self.tensor(&format!("{prefix}cross_attn_norm.bias"))?,
            cross_attention: FireRedDecoderCrossAttentionWeights {
                q_w_t: self.tensor(&format!("{prefix}cross_attn.w_qs.weight"))?,
                q_b: self.tensor(&format!("{prefix}cross_attn.w_qs.bias"))?,
                k_w_t: self.tensor(&format!("{prefix}cross_attn.w_ks.weight"))?,
                v_w_t: self.tensor(&format!("{prefix}cross_attn.w_vs.weight"))?,
                v_b: self.tensor(&format!("{prefix}cross_attn.w_vs.bias"))?,
                output_w_t: self.tensor(&format!("{prefix}cross_attn.fc.weight"))?,
                output_b: self.tensor(&format!("{prefix}cross_attn.fc.bias"))?,
            },
            mlp_norm_gamma: self.tensor(&format!("{prefix}mlp_norm.weight"))?,
            mlp_norm_beta: self.tensor(&format!("{prefix}mlp_norm.bias"))?,
            mlp_expand_w_t: self.tensor(&format!("{prefix}mlp.w_1.weight"))?,
            mlp_expand_b: self.tensor(&format!("{prefix}mlp.w_1.bias"))?,
            mlp_project_w_t: self.tensor(&format!("{prefix}mlp.w_2.weight"))?,
            mlp_project_b: self.tensor(&format!("{prefix}mlp.w_2.bias"))?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeLayout {
    Direct,
    Transpose2d,
}

fn decoder_layout(layout: super::FireRedDecoderNativeLayout) -> NativeLayout {
    match layout {
        super::FireRedDecoderNativeLayout::LinearOutInToComputeInOut
        | super::FireRedDecoderNativeLayout::ProjectionRows => NativeLayout::Transpose2d,
        _ => NativeLayout::Direct,
    }
}

fn encoder_layout(layout: super::FireRedEncoderNativeLayout) -> NativeLayout {
    match layout {
        super::FireRedEncoderNativeLayout::LinearOutInToComputeInOut => NativeLayout::Transpose2d,
        _ => NativeLayout::Direct,
    }
}

fn encoder_runtime_layout(name: &str, layout: super::FireRedEncoderNativeLayout) -> NativeLayout {
    if name == "encoder.input_preprocessor.out.weight" {
        // FireRedConv2dSubsampling consumes the raw PyTorch projection and
        // performs its one transpose inside the helper.
        NativeLayout::Direct
    } else {
        encoder_layout(layout)
    }
}

fn transpose_2d(values: &[f32], shape: &[u64]) -> Result<Vec<f32>> {
    if shape.len() != 2 {
        return Err(VokraError::ModelLoad(format!(
            "firered-asr-aed-l linear tensor has non-2D shape {shape:?}"
        )));
    }
    let rows = usize::try_from(shape[0])
        .map_err(|_| VokraError::ModelLoad("linear row count overflow".to_owned()))?;
    let cols = usize::try_from(shape[1])
        .map_err(|_| VokraError::ModelLoad("linear column count overflow".to_owned()))?;
    if values.len()
        != rows
            .checked_mul(cols)
            .ok_or_else(|| VokraError::ModelLoad("linear shape overflow".to_owned()))?
    {
        return Err(VokraError::ModelLoad(
            "linear tensor payload shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = values[row * cols + col];
        }
    }
    Ok(output)
}

/// Builds the fixed sinusoidal relative-position slice used by upstream
/// `RelPositionalEncoding.forward`.  The returned row-major buffer has shape
/// `[2 * frames - 1, d_model]` and is ordered exactly as the upstream `pe`
/// window (positive-to-negative relative positions).
pub fn relative_positional_encoding(
    d_model: usize,
    max_len: usize,
    frames: usize,
) -> Result<Vec<f32>> {
    if d_model == 0 || max_len == 0 || frames == 0 || frames > max_len {
        return Err(VokraError::InvalidArgument(format!(
            "FireRed relative positional encoding requires 0 < frames <= max_len and positive d_model, got d_model={d_model}, max_len={max_len}, frames={frames}"
        )));
    }
    let span = max_len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument("FireRed relative-position span overflow".to_owned())
        })?;
    let table_len = span.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative-position table overflow".to_owned())
    })?;
    let mut table = vec![0.0; table_len];
    for position in 0..max_len {
        for dimension in (0..d_model).step_by(2) {
            let div = (-(10000.0_f32.ln()) * dimension as f32 / d_model as f32).exp();
            table[position * d_model + dimension] = (position as f32 * div).sin();
            if dimension + 1 < d_model {
                table[position * d_model + dimension + 1] = (position as f32 * div).cos();
            }
        }
    }
    // Upstream concatenates flipped non-negative positions with negative
    // positions excluding zero.  Negating the sine terms is sufficient for
    // the negative half while cosine remains even.
    let positive = table.clone();
    for position in 0..max_len {
        let source = max_len - 1 - position;
        for dimension in 0..d_model {
            table[position * d_model + dimension] = positive[source * d_model + dimension];
        }
    }
    for position in 1..max_len {
        let source = position;
        let destination = max_len - 1 + position;
        for dimension in (0..d_model).step_by(2) {
            table[destination * d_model + dimension] = -positive[source * d_model + dimension];
            if dimension + 1 < d_model {
                table[destination * d_model + dimension + 1] =
                    positive[source * d_model + dimension + 1];
            }
        }
    }
    let start = max_len - frames;
    let window = frames
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            VokraError::InvalidArgument("FireRed relative-position window overflow".to_owned())
        })?;
    let begin = start.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative-position offset overflow".to_owned())
    })?;
    let end = begin
        .checked_add(window.checked_mul(d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed relative-position window overflow".to_owned())
        })?)
        .ok_or_else(|| {
            VokraError::InvalidArgument("FireRed relative-position offset overflow".to_owned())
        })?;
    let result = table[begin..end].to_vec();
    if result.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "FireRed relative-position table is non-finite".to_owned(),
        ));
    }
    Ok(result)
}

/// Exact Kaldi CMVN transform used by the pinned upstream `CMVN` class.
#[derive(Debug, Clone, PartialEq)]
pub struct FireRedCmvn {
    means: Vec<f32>,
    inverse_std: Vec<f32>,
}

impl FireRedCmvn {
    /// Builds CMVN from the upstream 2×(dim+1) row-major Kaldi stats matrix.
    /// The final column is the frame count; variance is floored at 1e-20.
    pub fn from_stats(stats: &[f32], dim: usize) -> Result<Self> {
        let width = dim.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed CMVN dimension overflow".to_owned())
        })?;
        let stats_len = width.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed CMVN stats shape overflow".to_owned())
        })?;
        if dim == 0 || stats.len() != stats_len {
            return Err(VokraError::InvalidArgument(format!(
                "FireRed CMVN expects a 2x{} stats matrix, got {} values",
                width,
                stats.len()
            )));
        }
        let count = stats[dim];
        if !count.is_finite() || count <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "FireRed CMVN frame count must be finite and positive".to_owned(),
            ));
        }
        let mut means = Vec::with_capacity(dim);
        let mut inverse_std = Vec::with_capacity(dim);
        for index in 0..dim {
            let mean = stats[index] / count;
            let variance = (stats[width + index] / count - mean * mean).max(1e-20);
            if !mean.is_finite() || !variance.is_finite() || variance <= 0.0 {
                return Err(VokraError::InvalidArgument(
                    "FireRed CMVN statistics are non-finite".to_owned(),
                ));
            }
            means.push(mean);
            let inverse = 1.0 / variance.sqrt();
            if !inverse.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "FireRed CMVN inverse standard deviation is non-finite".to_owned(),
                ));
            }
            inverse_std.push(inverse);
        }
        Ok(Self { means, inverse_std })
    }

    /// Applies `(x - means) * inverse_std` to row-major `[frames, dim]` data.
    pub fn apply(&self, values: &mut [f32], frames: usize) -> Result<()> {
        let dim = self.means.len();
        let expected = frames.checked_mul(dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed CMVN input shape overflow".to_owned())
        })?;
        if values.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "FireRed CMVN input has {} values, expected {}x{}",
                values.len(),
                frames,
                dim
            )));
        }
        for row in values.chunks_exact_mut(dim) {
            for (index, value) in row.iter_mut().enumerate() {
                *value = (*value - self.means[index]) * self.inverse_std[index];
                if !value.is_finite() {
                    return Err(VokraError::InvalidArgument(
                        "FireRed CMVN output is non-finite".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.means.len()
    }
}

/// FireRed's exact two-layer, unpadded stride-2 Conv2d subsampling stem.
#[derive(Debug, Clone, Copy)]
pub struct FireRedConv2dSubsampling {
    pub out_channels: usize,
    pub d_model: usize,
}

impl FireRedConv2dSubsampling {
    /// Applies Conv2d(1→C, 3, stride=2), ReLU, Conv2d(C→C, 3, stride=2),
    /// ReLU, flattening each time step's channel/frequency plane, and the
    /// final linear projection.  Weights use PyTorch layout `[out,in,h,w]`
    /// and `[out,in]`; the input is row-major `[frames, frequency]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        frequency: usize,
        conv0_w: &[f32],
        conv0_b: &[f32],
        conv1_w: &[f32],
        conv1_b: &[f32],
        out_w: &[f32],
        out_b: &[f32],
    ) -> Result<(Vec<f32>, usize)> {
        if self.out_channels == 0 || self.d_model == 0 || frames < 7 || frequency < 7 {
            return Err(VokraError::InvalidArgument(
                "FireRed Conv2d subsampling requires positive channels/model width and input axes >= 7"
                    .to_owned(),
            ));
        }
        let input_len = frames.checked_mul(frequency).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d input shape overflow".to_owned())
        })?;
        if input.len() != input_len {
            return Err(VokraError::InvalidArgument(
                "FireRed Conv2d input shape mismatch".to_owned(),
            ));
        }
        let first_h = (frames - 3) / 2 + 1;
        let first_w = (frequency - 3) / 2 + 1;
        let second_h = (first_h - 3) / 2 + 1;
        let second_w = (first_w - 3) / 2 + 1;
        let first_len = self
            .out_channels
            .checked_mul(first_h)
            .and_then(|value| value.checked_mul(first_w))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed Conv2d intermediate shape overflow".to_owned())
            })?;
        let second_len = self
            .out_channels
            .checked_mul(second_h)
            .and_then(|value| value.checked_mul(second_w))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed Conv2d intermediate shape overflow".to_owned())
            })?;
        let expected0 = self.out_channels.checked_mul(9).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d weight shape overflow".to_owned())
        })?;
        let expected1 = self
            .out_channels
            .checked_mul(self.out_channels)
            .and_then(|value| value.checked_mul(9))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed Conv2d weight shape overflow".to_owned())
            })?;
        let expected_out = self
            .d_model
            .checked_mul(self.out_channels)
            .and_then(|value| value.checked_mul(second_w))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed Conv2d projection shape overflow".to_owned())
            })?;
        if conv0_w.len() != expected0
            || conv0_b.len() != self.out_channels
            || conv1_w.len() != expected1
            || conv1_b.len() != self.out_channels
            || out_w.len() != expected_out
            || out_b.len() != self.d_model
        {
            return Err(VokraError::ModelLoad(
                "FireRed Conv2d subsampling operand shape does not match authenticated source topology"
                    .to_owned(),
            ));
        }
        let mut first = vec![0.0; first_len];
        compute.conv2d_f32(
            input,
            1,
            frames,
            frequency,
            conv0_w,
            self.out_channels,
            3,
            3,
            Some(conv0_b),
            (2, 2),
            (0, 0),
            (1, 1),
            1,
            &mut first,
        )?;
        let mut first_relu = vec![0.0; first.len()];
        compute.relu_f32(&first, &mut first_relu)?;
        let mut second = vec![0.0; second_len];
        compute.conv2d_f32(
            &first_relu,
            self.out_channels,
            first_h,
            first_w,
            conv1_w,
            self.out_channels,
            3,
            3,
            Some(conv1_b),
            (2, 2),
            (0, 0),
            (1, 1),
            1,
            &mut second,
        )?;
        let mut second_relu = vec![0.0; second.len()];
        compute.relu_f32(&second, &mut second_relu)?;

        // PyTorch Linear stores [d_model, channels*frequency].  Convert to
        // the Compute row-major [channels*frequency, d_model] contract.
        let projection_in = self.out_channels.checked_mul(second_w).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d projection shape overflow".to_owned())
        })?;
        let projection_len = projection_in.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d projection shape overflow".to_owned())
        })?;
        let mut out_w_t = vec![0.0; projection_len];
        for output in 0..self.d_model {
            for input_index in 0..projection_in {
                out_w_t[input_index * self.d_model + output] =
                    out_w[output * projection_in + input_index];
            }
        }
        let flattened_len = second_h.checked_mul(projection_in).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d flatten shape overflow".to_owned())
        })?;
        let mut flattened = vec![0.0; flattened_len];
        for time in 0..second_h {
            for channel in 0..self.out_channels {
                let src = (channel * second_h + time) * second_w;
                flattened[time * projection_in + channel * second_w
                    ..time * projection_in + (channel + 1) * second_w]
                    .copy_from_slice(&second_relu[src..src + second_w]);
            }
        }
        let output_len = second_h.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conv2d output shape overflow".to_owned())
        })?;
        let mut output = vec![0.0; output_len];
        compute.gemm_f32(
            second_h,
            self.d_model,
            projection_in,
            &flattened,
            &out_w_t,
            Some(out_b),
            &mut output,
        )?;
        Ok((output, second_h))
    }
}

/// Source-faithful relative-position multi-head attention projection and
/// score path.  [`Self::forward`] covers the exact
/// `matrix_ac + rel_shift(matrix_bd)` score construction.  The complete
/// source attention seam, including the bias-free output projection and
/// residual, is [`Self::forward_with_output`].
#[derive(Debug, Clone, Copy)]
pub struct FireRedRelativeAttention {
    pub d_model: usize,
    pub n_head: usize,
}

impl FireRedRelativeAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        positions: &[f32],
        frames: usize,
        q_w_t: &[f32],
        k_w_t: &[f32],
        v_w_t: &[f32],
        linear_pos_w_t: &[f32],
        q_norm_gamma: &[f32],
        q_norm_beta: &[f32],
        k_norm_gamma: &[f32],
        k_norm_beta: &[f32],
        v_norm_gamma: &[f32],
        v_norm_beta: &[f32],
        bias_u: &[f32],
        bias_v: &[f32],
    ) -> Result<Vec<f32>> {
        self.forward_with_mask(
            compute,
            input,
            positions,
            frames,
            q_w_t,
            k_w_t,
            v_w_t,
            linear_pos_w_t,
            q_norm_gamma,
            q_norm_beta,
            k_norm_gamma,
            k_norm_beta,
            v_norm_gamma,
            v_norm_beta,
            bias_u,
            bias_v,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Masked variant used by the encoder block; `key_mask` is `[frames]`
    /// and follows upstream `src_mask` key-column semantics.
    pub fn forward_with_mask(
        &self,
        compute: &Compute,
        input: &[f32],
        positions: &[f32],
        frames: usize,
        q_w_t: &[f32],
        k_w_t: &[f32],
        v_w_t: &[f32],
        linear_pos_w_t: &[f32],
        q_norm_gamma: &[f32],
        q_norm_beta: &[f32],
        k_norm_gamma: &[f32],
        k_norm_beta: &[f32],
        v_norm_gamma: &[f32],
        v_norm_beta: &[f32],
        bias_u: &[f32],
        bias_v: &[f32],
        key_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let positions_count = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "FireRed relative-attention position shape overflow".to_owned(),
                )
            })?;
        let input_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention input shape overflow".to_owned(),
            )
        })?;
        let position_len = positions_count.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention position shape overflow".to_owned(),
            )
        })?;
        let model_matrix_len = self.d_model.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention matrix shape overflow".to_owned(),
            )
        })?;
        if frames == 0
            || self.d_model == 0
            || self.n_head == 0
            || self.d_model % self.n_head != 0
            || input.len() != input_len
            || positions.len() != position_len
            || q_w_t.len() != model_matrix_len
            || k_w_t.len() != model_matrix_len
            || v_w_t.len() != model_matrix_len
            || linear_pos_w_t.len() != model_matrix_len
            || q_norm_gamma.len() != self.d_model
            || q_norm_beta.len() != self.d_model
            || k_norm_gamma.len() != self.d_model
            || k_norm_beta.len() != self.d_model
            || v_norm_gamma.len() != self.d_model
            || v_norm_beta.len() != self.d_model
            || bias_u.len() != self.d_model
            || bias_v.len() != self.d_model
            || key_mask.is_some_and(|mask| mask.len() != frames || !mask.iter().any(|&valid| valid))
            || !all_finite(&[
                input,
                positions,
                q_w_t,
                k_w_t,
                v_w_t,
                linear_pos_w_t,
                q_norm_gamma,
                q_norm_beta,
                k_norm_gamma,
                k_norm_beta,
                v_norm_gamma,
                v_norm_beta,
                bias_u,
                bias_v,
            ])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed relative-attention operand shape mismatch".to_owned(),
            ));
        }
        let head_dim = self.d_model / self.n_head;
        let head_values = frames.checked_mul(head_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed relative-attention head shape overflow".to_owned())
        })?;
        let relative_values = frames.checked_mul(positions_count).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention relative shape overflow".to_owned(),
            )
        })?;
        let output_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention output shape overflow".to_owned(),
            )
        })?;
        let mut q_norm = vec![0.0; input.len()];
        let mut k_norm = vec![0.0; input.len()];
        let mut v_norm = vec![0.0; input.len()];
        compute.layer_norm_f32(
            input,
            &mut q_norm,
            frames,
            self.d_model,
            q_norm_gamma,
            q_norm_beta,
            1e-5,
        )?;
        compute.layer_norm_f32(
            input,
            &mut k_norm,
            frames,
            self.d_model,
            k_norm_gamma,
            k_norm_beta,
            1e-5,
        )?;
        compute.layer_norm_f32(
            input,
            &mut v_norm,
            frames,
            self.d_model,
            v_norm_gamma,
            v_norm_beta,
            1e-5,
        )?;
        let mut q = vec![0.0; input.len()];
        let mut k = vec![0.0; input.len()];
        let mut v = vec![0.0; input.len()];
        compute.gemm_f32(
            frames,
            self.d_model,
            self.d_model,
            &q_norm,
            q_w_t,
            None,
            &mut q,
        )?;
        compute.gemm_f32(
            frames,
            self.d_model,
            self.d_model,
            &k_norm,
            k_w_t,
            None,
            &mut k,
        )?;
        compute.gemm_f32(
            frames,
            self.d_model,
            self.d_model,
            &v_norm,
            v_w_t,
            None,
            &mut v,
        )?;
        if q.iter()
            .chain(k.iter())
            .chain(v.iter())
            .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "FireRed relative-attention projection is non-finite".to_owned(),
            ));
        }
        let mut projected_pos = vec![0.0; positions.len()];
        compute.gemm_f32(
            positions_count,
            self.d_model,
            self.d_model,
            positions,
            linear_pos_w_t,
            None,
            &mut projected_pos,
        )?;
        if projected_pos.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "FireRed projected positions are non-finite".to_owned(),
            ));
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        if !scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "FireRed relative-attention scale is non-finite".to_owned(),
            ));
        }
        let head_scores = frames.checked_mul(frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention score shape overflow".to_owned(),
            )
        })?;
        let score_len = self.n_head.checked_mul(head_scores).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention score shape overflow".to_owned(),
            )
        })?;
        let mut scores = vec![0.0; score_len];
        for head in 0..self.n_head {
            let head_offset = head * head_dim;
            let mut q_u = vec![0.0; head_values];
            let mut q_v = vec![0.0; head_values];
            let mut k_t = vec![0.0; head_values];
            let mut pos_t = vec![
                0.0;
                head_dim.checked_mul(positions_count).ok_or_else(|| {
                    VokraError::InvalidArgument(
                        "FireRed relative-attention position head overflow".to_owned(),
                    )
                })?
            ];
            for row in 0..frames {
                for dim in 0..head_dim {
                    let hidden = head_offset + dim;
                    q_u[row * head_dim + dim] = q[row * self.d_model + hidden] + bias_u[hidden];
                    q_v[row * head_dim + dim] = q[row * self.d_model + hidden] + bias_v[hidden];
                    k_t[dim * frames + row] = k[row * self.d_model + hidden];
                }
            }
            for row in 0..positions_count {
                for dim in 0..head_dim {
                    pos_t[dim * positions_count + row] =
                        projected_pos[row * self.d_model + head_offset + dim];
                }
            }
            let mut content = vec![0.0; head_scores];
            compute.gemm_f32(frames, frames, head_dim, &q_u, &k_t, None, &mut content)?;
            let mut relative_raw = vec![0.0; relative_values];
            compute.gemm_f32(
                frames,
                positions_count,
                head_dim,
                &q_v,
                &pos_t,
                None,
                &mut relative_raw,
            )?;
            let relative = rel_shift(&relative_raw, frames, positions_count)?;
            for index in 0..head_scores {
                let value = (content[index] + relative[index]) * scale;
                if !value.is_finite() {
                    return Err(VokraError::InvalidArgument(
                        "FireRed relative-attention score is non-finite".to_owned(),
                    ));
                }
                scores[head * head_scores + index] = value;
            }
        }
        // Upstream passes `src_mask` to attention as a key mask. Keep the
        // sentinel finite (Compute::softmax must never receive NaN/−inf),
        // then explicitly zero and renormalize masked columns below.
        if let Some(mask) = key_mask {
            for head in 0..self.n_head {
                for query in 0..frames {
                    for key in 0..frames {
                        if !mask[key] {
                            scores[head * head_scores + query * frames + key] = -f32::MAX;
                        }
                    }
                }
            }
        }
        let mut probabilities = vec![0.0; scores.len()];
        let softmax_rows = self.n_head.checked_mul(frames).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention softmax shape overflow".to_owned(),
            )
        })?;
        compute.softmax_f32(&scores, &mut probabilities, softmax_rows, frames)?;
        if let Some(mask) = key_mask {
            for head in 0..self.n_head {
                for query in 0..frames {
                    let row_start = head * head_scores + query * frames;
                    let row = &mut probabilities[row_start..row_start + frames];
                    let mut valid_sum = 0.0f32;
                    for (key, probability) in row.iter_mut().enumerate() {
                        if mask[key] {
                            valid_sum += *probability;
                        } else {
                            *probability = 0.0;
                        }
                    }
                    if !valid_sum.is_finite() || valid_sum <= 0.0 {
                        return Err(VokraError::InvalidArgument(
                            "FireRed attention masked softmax has no finite valid mass".to_owned(),
                        ));
                    }
                    for (key, probability) in row.iter_mut().enumerate() {
                        if mask[key] {
                            *probability /= valid_sum;
                        }
                    }
                }
            }
        }
        let mut output = vec![0.0; output_len];
        for head in 0..self.n_head {
            let mut v_head = vec![0.0; head_values];
            for row in 0..frames {
                v_head[row * head_dim..(row + 1) * head_dim].copy_from_slice(
                    &v[row * self.d_model + head * head_dim
                        ..row * self.d_model + (head + 1) * head_dim],
                );
            }
            let mut context = vec![0.0; head_values];
            compute.gemm_f32(
                frames,
                head_dim,
                frames,
                &probabilities[head * head_scores..(head + 1) * head_scores],
                &v_head,
                None,
                &mut context,
            )?;
            for row in 0..frames {
                output[row * self.d_model + head * head_dim
                    ..row * self.d_model + (head + 1) * head_dim]
                    .copy_from_slice(&context[row * head_dim..(row + 1) * head_dim]);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "FireRed relative-attention output is non-finite".to_owned(),
            ));
        }
        Ok(output)
    }

    /// Completes the source `RelPosMultiHeadAttention` path with its
    /// bias-free `fc` projection and residual add.  The matrix is row-major
    /// `[d_model, d_model]` in Compute's transposed (`[in, out]`) layout.
    /// Dropout is intentionally absent: this is an inference-only seam and
    /// upstream eval-mode dropout is identity.
    pub fn forward_with_output(
        &self,
        compute: &Compute,
        input: &[f32],
        positions: &[f32],
        frames: usize,
        q_w_t: &[f32],
        k_w_t: &[f32],
        v_w_t: &[f32],
        linear_pos_w_t: &[f32],
        q_norm_gamma: &[f32],
        q_norm_beta: &[f32],
        k_norm_gamma: &[f32],
        k_norm_beta: &[f32],
        v_norm_gamma: &[f32],
        v_norm_beta: &[f32],
        bias_u: &[f32],
        bias_v: &[f32],
        output_w_t: &[f32],
    ) -> Result<Vec<f32>> {
        self.forward_with_output_mask(
            compute,
            input,
            positions,
            frames,
            q_w_t,
            k_w_t,
            v_w_t,
            linear_pos_w_t,
            q_norm_gamma,
            q_norm_beta,
            k_norm_gamma,
            k_norm_beta,
            v_norm_gamma,
            v_norm_beta,
            bias_u,
            bias_v,
            output_w_t,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Completes [`Self::forward_with_mask`] with the bias-free output
    /// projection and residual.
    pub fn forward_with_output_mask(
        &self,
        compute: &Compute,
        input: &[f32],
        positions: &[f32],
        frames: usize,
        q_w_t: &[f32],
        k_w_t: &[f32],
        v_w_t: &[f32],
        linear_pos_w_t: &[f32],
        q_norm_gamma: &[f32],
        q_norm_beta: &[f32],
        k_norm_gamma: &[f32],
        k_norm_beta: &[f32],
        v_norm_gamma: &[f32],
        v_norm_beta: &[f32],
        bias_u: &[f32],
        bias_v: &[f32],
        output_w_t: &[f32],
        key_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let matrix_len = self.d_model.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention output matrix shape overflow".to_owned(),
            )
        })?;
        if output_w_t.len() != matrix_len || output_w_t.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "FireRed relative-attention output projection shape or values are invalid"
                    .to_owned(),
            ));
        }
        let attention = self.forward_with_mask(
            compute,
            input,
            positions,
            frames,
            q_w_t,
            k_w_t,
            v_w_t,
            linear_pos_w_t,
            q_norm_gamma,
            q_norm_beta,
            k_norm_gamma,
            k_norm_beta,
            v_norm_gamma,
            v_norm_beta,
            bias_u,
            bias_v,
            key_mask,
        )?;
        let output_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed relative-attention output shape overflow".to_owned(),
            )
        })?;
        let mut projected = vec![0.0; output_len];
        compute.gemm_f32(
            frames,
            self.d_model,
            self.d_model,
            &attention,
            output_w_t,
            None,
            &mut projected,
        )?;
        for (value, residual) in projected.iter_mut().zip(input) {
            *value += residual;
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "FireRed relative-attention output residual is non-finite".to_owned(),
                ));
            }
        }
        Ok(projected)
    }
}

/// Implements the source `_rel_shift` reshape/pad index glue after the
/// relative GEMM. `x` is `[frames, 2*frames-1]`; the returned matrix is
/// `[frames, frames]`.
fn rel_shift(x: &[f32], frames: usize, positions_count: usize) -> Result<Vec<f32>> {
    let expected = frames.checked_mul(positions_count).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative shift shape overflow".to_owned())
    })?;
    if positions_count
        != frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .unwrap_or(0)
        || x.len() != expected
    {
        return Err(VokraError::InvalidArgument(
            "FireRed relative shift input shape mismatch".to_owned(),
        ));
    }
    let padded_width = positions_count.checked_add(1).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative shift padded shape overflow".to_owned())
    })?;
    let padded_len = frames.checked_mul(padded_width).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative shift padded shape overflow".to_owned())
    })?;
    let output_len = frames.checked_mul(frames).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed relative shift output overflow".to_owned())
    })?;
    let mut padded = vec![0.0; padded_len];
    for query in 0..frames {
        padded[query * padded_width + 1..(query + 1) * padded_width]
            .copy_from_slice(&x[query * positions_count..(query + 1) * positions_count]);
    }
    let mut result = vec![0.0; output_len];
    for query in 0..frames {
        for key in 0..frames {
            // F.pad(x, (1, 0)).view(..., positions_count + 1, frames),
            // drop the first row, reshape to (..., frames, positions_count),
            // then keep the first `frames` columns.
            let flattened = frames + query * positions_count + key;
            result[query * frames + key] = padded[flattened];
        }
    }
    Ok(result)
}

/// Inference-only source Conformer feed-forward module.  The module itself
/// includes its pre-LayerNorm and residual, exactly like upstream
/// `ConformerFeedForward`; the enclosing block applies the separate 0.5
/// half-step to the module result.
#[derive(Debug, Clone, Copy)]
pub struct FireRedConformerFeedForward {
    pub d_model: usize,
    /// Source `d_inner` (already four times `d_model` for this release).
    pub inner_dim: usize,
}

impl FireRedConformerFeedForward {
    /// Runs `LN -> Linear(d,d_inner) -> Swish -> Linear(d_inner,d) ->
    /// residual`, where `d_inner` is supplied directly by the release config.
    /// Weights are transposed row-major Compute matrices (`[in, out]`), and
    /// dropout is identity in this eval-only path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        ln_gamma: &[f32],
        ln_beta: &[f32],
        expand_w_t: &[f32],
        expand_b: &[f32],
        project_w_t: &[f32],
        project_b: &[f32],
    ) -> Result<Vec<f32>> {
        let inner_dim = self.inner_dim;
        let input_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed FFN input shape overflow".to_owned())
        })?;
        let expand_len = self.d_model.checked_mul(inner_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed FFN expand shape overflow".to_owned())
        })?;
        let project_len = inner_dim.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed FFN project shape overflow".to_owned())
        })?;
        if self.d_model == 0
            || inner_dim == 0
            || frames == 0
            || input.len() != input_len
            || ln_gamma.len() != self.d_model
            || ln_beta.len() != self.d_model
            || expand_w_t.len() != expand_len
            || expand_b.len() != inner_dim
            || project_w_t.len() != project_len
            || project_b.len() != self.d_model
            || !all_finite(&[
                input,
                ln_gamma,
                ln_beta,
                expand_w_t,
                expand_b,
                project_w_t,
                project_b,
            ])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed FFN operand shape or values are invalid".to_owned(),
            ));
        }
        let mut normalized = vec![0.0; input_len];
        compute.layer_norm_f32(
            input,
            &mut normalized,
            frames,
            self.d_model,
            ln_gamma,
            ln_beta,
            1e-5,
        )?;
        let mut expanded = vec![
            0.0;
            frames.checked_mul(inner_dim).ok_or_else(|| {
                VokraError::InvalidArgument("FireRed FFN activation shape overflow".to_owned())
            })?
        ];
        compute.gemm_f32(
            frames,
            inner_dim,
            self.d_model,
            &normalized,
            expand_w_t,
            Some(expand_b),
            &mut expanded,
        )?;
        let mut activated = vec![0.0; expanded.len()];
        compute.silu_f32(&expanded, &mut activated)?;
        let mut projected = vec![0.0; input_len];
        compute.gemm_f32(
            frames,
            self.d_model,
            inner_dim,
            &activated,
            project_w_t,
            Some(project_b),
            &mut projected,
        )?;
        for (value, residual) in projected.iter_mut().zip(input) {
            *value += residual;
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "FireRed FFN residual is non-finite".to_owned(),
                ));
            }
        }
        Ok(projected)
    }
}

/// Source Conformer convolution module.  Inputs and outputs are frame-major
/// `[frames, d_model]`; the internal Conv1d operands are channel-major
/// `[channels, frames]`, matching the upstream transpose.  The depthwise
/// groups are dispatched through the existing single-group Compute seam one
/// channel at a time, with no CPU fallback when the selected backend lacks it.
#[derive(Debug, Clone, Copy)]
pub struct FireRedConformerConvolution {
    pub d_model: usize,
    pub kernel_size: usize,
}

impl FireRedConformerConvolution {
    /// Runs the source `LN -> pointwise -> GLU -> depthwise -> LN -> Swish ->
    /// pointwise` path with eval-mode dropout identity and explicit masks.
    /// `pointwise_in_w`, `depthwise_w`, and `pointwise_out_w` use raw PyTorch
    /// Conv1d layout `[out_channels, in_channels, kernel]` (the pointwise
    /// kernels therefore have `kernel = 1`; depthwise has `in = out = 2d`).
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        input_mask: &[bool],
        pointwise_in_w: &[f32],
        depthwise_w: &[f32],
        depthwise_ln_gamma: &[f32],
        depthwise_ln_beta: &[f32],
        pointwise_out_w: &[f32],
        pre_ln_gamma: &[f32],
        pre_ln_beta: &[f32],
    ) -> Result<Vec<f32>> {
        let d = self.d_model;
        let two_d = d.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed convolution width overflow".to_owned())
        })?;
        let four_d = d.checked_mul(4).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed convolution width overflow".to_owned())
        })?;
        let input_len = frames.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed convolution input shape overflow".to_owned())
        })?;
        let pointwise_in_len = four_d.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed convolution pointwise shape overflow".to_owned())
        })?;
        let depthwise_len = two_d.checked_mul(self.kernel_size).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed depthwise shape overflow".to_owned())
        })?;
        let pointwise_out_len = d.checked_mul(two_d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed convolution pointwise shape overflow".to_owned())
        })?;
        if d == 0
            || frames == 0
            || self.kernel_size == 0
            || self.kernel_size % 2 == 0
            || input.len() != input_len
            || input_mask.len() != frames
            || pointwise_in_w.len() != pointwise_in_len
            || depthwise_w.len() != depthwise_len
            || depthwise_ln_gamma.len() != two_d
            || depthwise_ln_beta.len() != two_d
            || pointwise_out_w.len() != pointwise_out_len
            || pre_ln_gamma.len() != d
            || pre_ln_beta.len() != d
            || !all_finite(&[
                input,
                pointwise_in_w,
                depthwise_w,
                depthwise_ln_gamma,
                depthwise_ln_beta,
                pointwise_out_w,
                pre_ln_gamma,
                pre_ln_beta,
            ])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed convolution operand shape or values are invalid".to_owned(),
            ));
        }
        let mut normalized = vec![0.0; input_len];
        compute.layer_norm_f32(
            input,
            &mut normalized,
            frames,
            d,
            pre_ln_gamma,
            pre_ln_beta,
            1e-5,
        )?;
        for (frame, valid) in input_mask.iter().copied().enumerate() {
            if !valid {
                normalized[frame * d..(frame + 1) * d].fill(0.0);
            }
        }
        let normalized_channels = transpose_frame_to_channel(&normalized, frames, d)?;
        let mut pointwise = vec![
            0.0;
            frames.checked_mul(four_d).ok_or_else(|| {
                VokraError::InvalidArgument(
                    "FireRed convolution activation shape overflow".to_owned(),
                )
            })?
        ];
        compute.conv1d_f32(
            &normalized_channels,
            d,
            frames,
            pointwise_in_w,
            four_d,
            1,
            None,
            1,
            0,
            &mut pointwise,
        )?;
        let glu = glu_split(&pointwise, frames, two_d)?;
        let depthwise =
            depthwise_same(compute, &glu, two_d, frames, depthwise_w, self.kernel_size)?;
        let depthwise_frames = transpose_channel_to_frame(&depthwise, frames, two_d)?;
        let mut depthwise_norm = vec![0.0; depthwise_frames.len()];
        compute.layer_norm_f32(
            &depthwise_frames,
            &mut depthwise_norm,
            frames,
            two_d,
            depthwise_ln_gamma,
            depthwise_ln_beta,
            1e-5,
        )?;
        let mut activated = vec![0.0; depthwise_norm.len()];
        compute.silu_f32(&depthwise_norm, &mut activated)?;
        let activated_channels = transpose_frame_to_channel(&activated, frames, two_d)?;
        let mut projected = vec![0.0; frames * d];
        compute.conv1d_f32(
            &activated_channels,
            two_d,
            frames,
            pointwise_out_w,
            d,
            1,
            None,
            1,
            0,
            &mut projected,
        )?;
        let mut output = input.to_vec();
        for frame in 0..frames {
            for channel in 0..d {
                let index = frame * d + channel;
                if input_mask[frame] {
                    output[index] += projected[channel * frames + frame];
                } else {
                    output[index] = 0.0;
                }
                if !output[index].is_finite() {
                    return Err(VokraError::InvalidArgument(
                        "FireRed convolution residual is non-finite".to_owned(),
                    ));
                }
            }
        }
        Ok(output)
    }
}

/// Complete inference-only FireRed Conformer block.  The two feed-forward
/// modules retain their own upstream residuals, then the block applies the
/// explicit `0.5 * x + 0.5 * ffn(x)` half-step twice; this is intentionally
/// not simplified to a single residual expression so the source equations
/// stay auditable.
#[derive(Debug, Clone, Copy)]
pub struct FireRedConformerBlock {
    pub d_model: usize,
    /// Source `d_inner` width, not a multiplier. The pinned release uses
    /// `d_model = 1280` and `d_inner = 5120`.
    pub inner_dim: usize,
    pub n_head: usize,
    pub kernel_size: usize,
}

/// Pinned-source encoder stack orchestration for batch-one frame-major
/// activations.  The layer weights are supplied by the strict GGUF binder;
/// this type deliberately does not manufacture a checkpoint-name mapping.
#[derive(Debug, Clone, Copy)]
pub struct FireRedConformerEncoder {
    d_model: usize,
    inner_dim: usize,
    n_head: usize,
    kernel_size: usize,
}

/// Borrowed checkpoint operands for one Conformer block.  Tensor names and
/// dimensions remain the responsibility of the exact 940-field
/// binder; these fields do not invent a manifest.
#[derive(Clone, Copy)]
pub struct FireRedConformerBlockWeights<'a> {
    pub ffn1_ln_gamma: &'a [f32],
    pub ffn1_ln_beta: &'a [f32],
    pub ffn1_expand_w_t: &'a [f32],
    pub ffn1_expand_b: &'a [f32],
    pub ffn1_project_w_t: &'a [f32],
    pub ffn1_project_b: &'a [f32],
    pub attention_positions: &'a [f32],
    pub attention_q_w_t: &'a [f32],
    pub attention_k_w_t: &'a [f32],
    pub attention_v_w_t: &'a [f32],
    pub attention_linear_pos_w_t: &'a [f32],
    pub attention_q_norm_gamma: &'a [f32],
    pub attention_q_norm_beta: &'a [f32],
    pub attention_k_norm_gamma: &'a [f32],
    pub attention_k_norm_beta: &'a [f32],
    pub attention_v_norm_gamma: &'a [f32],
    pub attention_v_norm_beta: &'a [f32],
    pub attention_bias_u: &'a [f32],
    pub attention_bias_v: &'a [f32],
    pub attention_output_w_t: &'a [f32],
    pub conv_pointwise_in_w: &'a [f32],
    pub conv_depthwise_w: &'a [f32],
    pub conv_depthwise_ln_gamma: &'a [f32],
    pub conv_depthwise_ln_beta: &'a [f32],
    pub conv_pointwise_out_w: &'a [f32],
    pub conv_pre_ln_gamma: &'a [f32],
    pub conv_pre_ln_beta: &'a [f32],
    pub ffn2_ln_gamma: &'a [f32],
    pub ffn2_ln_beta: &'a [f32],
    pub ffn2_expand_w_t: &'a [f32],
    pub ffn2_expand_b: &'a [f32],
    pub ffn2_project_w_t: &'a [f32],
    pub ffn2_project_b: &'a [f32],
    pub final_ln_gamma: &'a [f32],
    pub final_ln_beta: &'a [f32],
}

impl FireRedConformerBlock {
    /// Runs one frame-major `[frames, d_model]` block for batch one. `mask`
    /// has one boolean per frame; invalid frames are zeroed at the convolution
    /// input/output and after final LayerNorm. All learned operations route
    /// through `Compute`.
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        mask: &[bool],
        weights: &FireRedConformerBlockWeights<'_>,
    ) -> Result<Vec<f32>> {
        if frames == 0 || mask.len() != frames {
            return Err(VokraError::InvalidArgument(
                "FireRed Conformer block frame/mask shape mismatch".to_owned(),
            ));
        }
        let input_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer block input shape overflow".to_owned())
        })?;
        if input.len() != input_len || self.d_model == 0 || self.inner_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "FireRed Conformer block input or configuration is invalid".to_owned(),
            ));
        }
        self.validate_operands(input, frames, mask, weights)?;
        let ffn = FireRedConformerFeedForward {
            d_model: self.d_model,
            inner_dim: self.inner_dim,
        };
        let first_ffn = ffn.forward(
            compute,
            input,
            frames,
            weights.ffn1_ln_gamma,
            weights.ffn1_ln_beta,
            weights.ffn1_expand_w_t,
            weights.ffn1_expand_b,
            weights.ffn1_project_w_t,
            weights.ffn1_project_b,
        )?;
        let mut after_ffn1 = half_residual(input, &first_ffn)?;
        let attention = FireRedRelativeAttention {
            d_model: self.d_model,
            n_head: self.n_head,
        };
        after_ffn1 = attention.forward_with_output_mask(
            compute,
            &after_ffn1,
            weights.attention_positions,
            frames,
            weights.attention_q_w_t,
            weights.attention_k_w_t,
            weights.attention_v_w_t,
            weights.attention_linear_pos_w_t,
            weights.attention_q_norm_gamma,
            weights.attention_q_norm_beta,
            weights.attention_k_norm_gamma,
            weights.attention_k_norm_beta,
            weights.attention_v_norm_gamma,
            weights.attention_v_norm_beta,
            weights.attention_bias_u,
            weights.attention_bias_v,
            weights.attention_output_w_t,
            Some(mask),
        )?;
        let convolution = FireRedConformerConvolution {
            d_model: self.d_model,
            kernel_size: self.kernel_size,
        };
        let after_conv = convolution.forward(
            compute,
            &after_ffn1,
            frames,
            mask,
            weights.conv_pointwise_in_w,
            weights.conv_depthwise_w,
            weights.conv_depthwise_ln_gamma,
            weights.conv_depthwise_ln_beta,
            weights.conv_pointwise_out_w,
            weights.conv_pre_ln_gamma,
            weights.conv_pre_ln_beta,
        )?;
        let second_ffn = ffn.forward(
            compute,
            &after_conv,
            frames,
            weights.ffn2_ln_gamma,
            weights.ffn2_ln_beta,
            weights.ffn2_expand_w_t,
            weights.ffn2_expand_b,
            weights.ffn2_project_w_t,
            weights.ffn2_project_b,
        )?;
        let before_final = half_residual(&after_conv, &second_ffn)?;
        let mut output = vec![0.0; input_len];
        compute.layer_norm_f32(
            &before_final,
            &mut output,
            frames,
            self.d_model,
            weights.final_ln_gamma,
            weights.final_ln_beta,
            1e-5,
        )?;
        for frame in 0..frames {
            if !mask[frame] {
                output[frame * self.d_model..(frame + 1) * self.d_model].fill(0.0);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "FireRed Conformer final output is non-finite".to_owned(),
            ));
        }
        Ok(output)
    }

    fn validate_operands(
        &self,
        input: &[f32],
        frames: usize,
        mask: &[bool],
        weights: &FireRedConformerBlockWeights<'_>,
    ) -> Result<()> {
        if self.n_head == 0
            || self.d_model % self.n_head != 0
            || self.kernel_size == 0
            || self.kernel_size % 2 == 0
            || mask.len() != frames
            || !mask.iter().any(|&valid| valid)
        {
            return Err(VokraError::InvalidArgument(
                "FireRed Conformer block configuration or mask is invalid".to_owned(),
            ));
        }
        let inner = self.inner_dim;
        let two_d = self.d_model.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer channel width overflow".to_owned())
        })?;
        let four_d = self.d_model.checked_mul(4).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer block width overflow".to_owned())
        })?;
        let position_count = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed Conformer positions overflow".to_owned())
            })?;
        let matrix = self.d_model.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer matrix shape overflow".to_owned())
        })?;
        let ffn_expand = self.d_model.checked_mul(inner).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer FFN shape overflow".to_owned())
        })?;
        let conv_in = four_d.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer convolution shape overflow".to_owned())
        })?;
        let conv_depthwise = self
            .d_model
            .checked_mul(2)
            .and_then(|width| width.checked_mul(self.kernel_size))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed depthwise shape overflow".to_owned())
            })?;
        let conv_out = two_d.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer convolution shape overflow".to_owned())
        })?;
        let positions = position_count.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer positions shape overflow".to_owned())
        })?;
        let d = self.d_model;
        let input_len = frames.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed Conformer input shape overflow".to_owned())
        })?;
        let arrays = [
            ("input", input, input_len),
            ("ffn1_ln_gamma", weights.ffn1_ln_gamma, d),
            ("ffn1_ln_beta", weights.ffn1_ln_beta, d),
            ("ffn1_expand_w_t", weights.ffn1_expand_w_t, ffn_expand),
            ("ffn1_expand_b", weights.ffn1_expand_b, inner),
            ("ffn1_project_w_t", weights.ffn1_project_w_t, ffn_expand),
            ("ffn1_project_b", weights.ffn1_project_b, d),
            (
                "attention_positions",
                weights.attention_positions,
                positions,
            ),
            ("attention_q_w_t", weights.attention_q_w_t, matrix),
            ("attention_k_w_t", weights.attention_k_w_t, matrix),
            ("attention_v_w_t", weights.attention_v_w_t, matrix),
            (
                "attention_linear_pos_w_t",
                weights.attention_linear_pos_w_t,
                matrix,
            ),
            ("attention_q_norm_gamma", weights.attention_q_norm_gamma, d),
            ("attention_q_norm_beta", weights.attention_q_norm_beta, d),
            ("attention_k_norm_gamma", weights.attention_k_norm_gamma, d),
            ("attention_k_norm_beta", weights.attention_k_norm_beta, d),
            ("attention_v_norm_gamma", weights.attention_v_norm_gamma, d),
            ("attention_v_norm_beta", weights.attention_v_norm_beta, d),
            ("attention_bias_u", weights.attention_bias_u, d),
            ("attention_bias_v", weights.attention_bias_v, d),
            ("attention_output_w_t", weights.attention_output_w_t, matrix),
            ("conv_pointwise_in_w", weights.conv_pointwise_in_w, conv_in),
            ("conv_depthwise_w", weights.conv_depthwise_w, conv_depthwise),
            (
                "conv_depthwise_ln_gamma",
                weights.conv_depthwise_ln_gamma,
                two_d,
            ),
            (
                "conv_depthwise_ln_beta",
                weights.conv_depthwise_ln_beta,
                two_d,
            ),
            (
                "conv_pointwise_out_w",
                weights.conv_pointwise_out_w,
                conv_out,
            ),
            ("conv_pre_ln_gamma", weights.conv_pre_ln_gamma, d),
            ("conv_pre_ln_beta", weights.conv_pre_ln_beta, d),
            ("ffn2_ln_gamma", weights.ffn2_ln_gamma, d),
            ("ffn2_ln_beta", weights.ffn2_ln_beta, d),
            ("ffn2_expand_w_t", weights.ffn2_expand_w_t, ffn_expand),
            ("ffn2_expand_b", weights.ffn2_expand_b, inner),
            ("ffn2_project_w_t", weights.ffn2_project_w_t, ffn_expand),
            ("ffn2_project_b", weights.ffn2_project_b, d),
            ("final_ln_gamma", weights.final_ln_gamma, d),
            ("final_ln_beta", weights.final_ln_beta, d),
        ];
        for (name, values, expected) in arrays {
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "FireRed Conformer {name} shape or values are invalid"
                )));
            }
        }
        Ok(())
    }
}

impl FireRedConformerEncoder {
    /// Constructs the pinned release geometry.  The depth and all tensor
    /// axes are intentionally not caller-controlled: this release has
    /// exactly sixteen source blocks with the authenticated dimensions.
    pub fn authenticated() -> Self {
        Self {
            d_model: super::AUTHENTICATED_ENCODER_D_MODEL as usize,
            inner_dim: super::AUTHENTICATED_ENCODER_FFN_DIM as usize,
            n_head: super::AUTHENTICATED_ENCODER_N_HEAD as usize,
            kernel_size: super::AUTHENTICATED_ENCODER_KERNEL_SIZE as usize,
        }
    }

    /// Runs all encoder blocks in source order and performs a complete
    /// operand preflight for every block before the first backend dispatch.
    /// `layers[0]` is the upstream block zero and so on; no name-based
    /// fallback or layer truncation is permitted.
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        frames: usize,
        mask: &[bool],
        layers: &[FireRedConformerBlockWeights<'_>],
    ) -> Result<Vec<f32>> {
        if layers.len() != super::AUTHENTICATED_ENCODER_N_LAYER as usize {
            return Err(VokraError::ModelLoad(format!(
                "FireRed Conformer encoder configuration/layer count is invalid: expected {} layers, got {}",
                super::AUTHENTICATED_ENCODER_N_LAYER,
                layers.len()
            )));
        }
        let block = FireRedConformerBlock {
            d_model: self.d_model,
            inner_dim: self.inner_dim,
            n_head: self.n_head,
            kernel_size: self.kernel_size,
        };
        // Validate every layer against the original input shape and mask
        // before invoking any Compute operation.  This makes a late layer's
        // missing/non-finite operand fail before layer zero can dispatch.
        for (index, weights) in layers.iter().enumerate() {
            block
                .validate_operands(input, frames, mask, weights)
                .map_err(|error| {
                    VokraError::ModelLoad(format!(
                        "FireRed Conformer encoder layer {index} preflight failed: {error}"
                    ))
                })?;
        }
        let mut output = input.to_vec();
        for weights in layers {
            output = block.forward(compute, &output, frames, mask, weights)?;
        }
        Ok(output)
    }
}

/// Borrowed projections for the source FireRed decoder self-attention. Matrix
/// weights use the Compute layout `[input, output]` (the source PyTorch
/// `Linear` tensors are `[output, input]`); the two absent source biases are
/// represented by the lack of a field, not by fabricated zeros.
#[derive(Clone, Copy)]
pub struct FireRedDecoderSelfAttentionWeights<'a> {
    pub q_w_t: &'a [f32],
    pub q_b: &'a [f32],
    pub k_w_t: &'a [f32],
    pub v_w_t: &'a [f32],
    pub v_b: &'a [f32],
    pub output_w_t: &'a [f32],
    pub output_b: &'a [f32],
}

/// Borrowed projections for decoder cross-attention. The source
/// `DecoderMultiHeadAttention` uses the decoder width for K/V input too;
/// unequal encoder/decoder widths are rejected by the native seam.
#[derive(Clone, Copy)]
pub struct FireRedDecoderCrossAttentionWeights<'a> {
    pub q_w_t: &'a [f32],
    pub q_b: &'a [f32],
    pub k_w_t: &'a [f32],
    pub v_w_t: &'a [f32],
    pub v_b: &'a [f32],
    pub output_w_t: &'a [f32],
    pub output_b: &'a [f32],
}

/// Source `DecoderLayer` operands. All normalisation uses PyTorch
/// `LayerNorm(..., eps=1e-5)` and all dropout is identity in this inference
/// only path.
#[derive(Clone, Copy)]
pub struct FireRedDecoderLayerWeights<'a> {
    pub self_norm_gamma: &'a [f32],
    pub self_norm_beta: &'a [f32],
    pub self_attention: FireRedDecoderSelfAttentionWeights<'a>,
    pub cross_norm_gamma: &'a [f32],
    pub cross_norm_beta: &'a [f32],
    pub cross_attention: FireRedDecoderCrossAttentionWeights<'a>,
    pub mlp_norm_gamma: &'a [f32],
    pub mlp_norm_beta: &'a [f32],
    pub mlp_expand_w_t: &'a [f32],
    pub mlp_expand_b: &'a [f32],
    pub mlp_project_w_t: &'a [f32],
    pub mlp_project_b: &'a [f32],
}

/// Result of one self-attention invocation, including projected K/V cache
/// rows. Cache rows are frame-major `[cached_frames, d_model]`, so a caller
/// can pass them to the next step without re-projecting old tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct FireRedDecoderAttentionOutput {
    pub output: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

/// Source self-attention with causal and padding-key masks. With a non-empty
/// cache, only the supplied current query rows are projected and the causal
/// boundary is `past_len + query_index`; this is the upstream last-query cache
/// path when `query_frames == 1`.
#[derive(Debug, Clone, Copy)]
pub struct FireRedDecoderSelfAttention {
    pub d_model: usize,
    pub n_head: usize,
}

impl FireRedDecoderSelfAttention {
    pub fn forward(
        &self,
        compute: &Compute,
        query: &[f32],
        query_frames: usize,
        past_k: &[f32],
        past_v: &[f32],
        key_mask: &[bool],
        weights: FireRedDecoderSelfAttentionWeights<'_>,
    ) -> Result<FireRedDecoderAttentionOutput> {
        let d = self.d_model;
        let query_len = query_frames.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder self-attention query overflow".to_owned())
        })?;
        if query_frames == 0 || d == 0 || query.len() != query_len || past_k.len() != past_v.len() {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder self-attention query/cache shape is invalid".to_owned(),
            ));
        }
        let past_frames = past_k.len().checked_div(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder self-attention cache overflow".to_owned())
        })?;
        if past_frames.checked_mul(d) != Some(past_k.len()) {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder self-attention cache is not frame-major".to_owned(),
            ));
        }
        let total_frames = past_frames.checked_add(query_frames).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder self-attention frame overflow".to_owned())
        })?;
        if key_mask.len() != total_frames || !key_mask.iter().any(|&valid| valid) {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder self-attention key mask is invalid or all-masked".to_owned(),
            ));
        }
        validate_attention_geometry(d, self.n_head)?;
        let matrix = d.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder self-attention matrix overflow".to_owned())
        })?;
        validate_linear_operands(query, query_frames, d, d, weights.q_w_t, Some(weights.q_b))?;
        validate_linear_operands(query, query_frames, d, d, weights.k_w_t, None)?;
        validate_linear_operands(query, query_frames, d, d, weights.v_w_t, Some(weights.v_b))?;
        if weights.output_w_t.len() != matrix
            || weights.output_b.len() != d
            || !all_finite(&[past_k, past_v, weights.output_w_t, weights.output_b])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder self-attention output/cache operands are invalid".to_owned(),
            ));
        }
        let q = linear(
            compute,
            query,
            query_frames,
            d,
            d,
            weights.q_w_t,
            Some(weights.q_b),
        )?;
        let current_k = linear(compute, query, query_frames, d, d, weights.k_w_t, None)?;
        let current_v = linear(
            compute,
            query,
            query_frames,
            d,
            d,
            weights.v_w_t,
            Some(weights.v_b),
        )?;
        let cache_len = total_frames.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder self-attention cache overflow".to_owned())
        })?;
        let mut keys = Vec::with_capacity(cache_len);
        keys.extend_from_slice(past_k);
        keys.extend_from_slice(&current_k);
        let mut values = Vec::with_capacity(cache_len);
        values.extend_from_slice(past_v);
        values.extend_from_slice(&current_v);
        let attended = scaled_dot_product_attention(
            compute,
            &q,
            query_frames,
            &keys,
            total_frames,
            &values,
            d,
            self.n_head,
            key_mask,
            Some(past_frames),
        )?;
        let output = linear(
            compute,
            &attended,
            query_frames,
            d,
            d,
            weights.output_w_t,
            Some(weights.output_b),
        )?;
        ensure_finite(&output, "FireRed decoder self-attention output")?;
        Ok(FireRedDecoderAttentionOutput {
            output,
            key_cache: keys,
            value_cache: values,
        })
    }
}

/// Source cross-attention: no causal mask, but every encoder key must pass
/// the supplied source/padding mask. Query rows are frame-major and K/V rows
/// are encoder-frame-major.
#[derive(Debug, Clone, Copy)]
pub struct FireRedDecoderCrossAttention {
    pub d_model: usize,
    pub source_dim: usize,
    pub n_head: usize,
}

impl FireRedDecoderCrossAttention {
    pub fn forward(
        &self,
        compute: &Compute,
        query: &[f32],
        query_frames: usize,
        memory: &[f32],
        source_frames: usize,
        source_mask: &[bool],
        query_mask: Option<&[bool]>,
        weights: FireRedDecoderCrossAttentionWeights<'_>,
    ) -> Result<Vec<f32>> {
        let d = self.d_model;
        let source_dim = self.source_dim;
        let query_len = query_frames.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder cross-attention query overflow".to_owned())
        })?;
        let memory_len = source_frames.checked_mul(source_dim).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed decoder cross-attention memory overflow".to_owned(),
            )
        })?;
        if query_frames == 0
            || source_frames == 0
            || query.len() != query_len
            || memory.len() != memory_len
            || source_mask.len() != source_frames
            || !source_mask.iter().any(|&valid| valid)
            || source_dim != d
            || query_mask
                .is_some_and(|mask| mask.len() != query_frames || !mask.iter().all(|&valid| valid))
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder cross-attention shape or mask is invalid".to_owned(),
            ));
        }
        validate_attention_geometry(d, self.n_head)?;
        validate_linear_operands(query, query_frames, d, d, weights.q_w_t, Some(weights.q_b))?;
        validate_linear_operands(memory, source_frames, source_dim, d, weights.k_w_t, None)?;
        validate_linear_operands(
            memory,
            source_frames,
            source_dim,
            d,
            weights.v_w_t,
            Some(weights.v_b),
        )?;
        let matrix = d.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument(
                "FireRed decoder cross-attention matrix overflow".to_owned(),
            )
        })?;
        if weights.output_w_t.len() != matrix
            || weights.output_b.len() != d
            || !all_finite(&[query, memory, weights.output_w_t, weights.output_b])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder cross-attention output operands are invalid".to_owned(),
            ));
        }
        let q = linear(
            compute,
            query,
            query_frames,
            d,
            d,
            weights.q_w_t,
            Some(weights.q_b),
        )?;
        let k = linear(
            compute,
            memory,
            source_frames,
            source_dim,
            d,
            weights.k_w_t,
            None,
        )?;
        let v = linear(
            compute,
            memory,
            source_frames,
            source_dim,
            d,
            weights.v_w_t,
            Some(weights.v_b),
        )?;
        let attended = scaled_dot_product_attention(
            compute,
            &q,
            query_frames,
            &k,
            source_frames,
            &v,
            d,
            self.n_head,
            source_mask,
            None,
        )?;
        let output = linear(
            compute,
            &attended,
            query_frames,
            d,
            d,
            weights.output_w_t,
            Some(weights.output_b),
        )?;
        ensure_finite(&output, "FireRed decoder cross-attention output")?;
        Ok(output)
    }
}

/// Complete inference-only pre-norm decoder layer. The caller supplies the
/// projected self-attention cache; passing one current row exercises the
/// upstream incremental last-query path. The method deliberately returns the
/// cache separately and does not mutate hidden state behind the caller's back.
#[derive(Debug, Clone, Copy)]
pub struct FireRedDecoderLayer {
    pub d_model: usize,
    pub inner_dim: usize,
    pub n_head: usize,
    pub source_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FireRedDecoderLayerOutput {
    pub output: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl FireRedDecoderLayer {
    pub fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        query_frames: usize,
        query_mask: &[bool],
        memory: &[f32],
        source_frames: usize,
        source_mask: &[bool],
        past_k: &[f32],
        past_v: &[f32],
        past_mask: &[bool],
        weights: FireRedDecoderLayerWeights<'_>,
    ) -> Result<FireRedDecoderLayerOutput> {
        self.validate(
            input,
            query_frames,
            query_mask,
            memory,
            source_frames,
            source_mask,
            past_k,
            past_v,
            past_mask,
            weights,
        )?;
        let d = self.d_model;
        let mut normalized = vec![0.0; input.len()];
        compute.layer_norm_f32(
            input,
            &mut normalized,
            query_frames,
            d,
            weights.self_norm_gamma,
            weights.self_norm_beta,
            1e-5,
        )?;
        let self_output = FireRedDecoderSelfAttention {
            d_model: d,
            n_head: self.n_head,
        }
        .forward(
            compute,
            &normalized,
            query_frames,
            past_k,
            past_v,
            &combined_cache_mask(past_mask, query_mask)?,
            weights.self_attention,
        )?;
        let mut residual = add_masked_residual(input, &self_output.output, query_mask, d)?;
        let mut cross_norm = vec![0.0; residual.len()];
        compute.layer_norm_f32(
            &residual,
            &mut cross_norm,
            query_frames,
            d,
            weights.cross_norm_gamma,
            weights.cross_norm_beta,
            1e-5,
        )?;
        let cross_output = FireRedDecoderCrossAttention {
            d_model: d,
            source_dim: self.source_dim,
            n_head: self.n_head,
        }
        .forward(
            compute,
            &cross_norm,
            query_frames,
            memory,
            source_frames,
            source_mask,
            Some(query_mask),
            weights.cross_attention,
        )?;
        residual = add_masked_residual(&residual, &cross_output, query_mask, d)?;
        let mut mlp_norm = vec![0.0; residual.len()];
        compute.layer_norm_f32(
            &residual,
            &mut mlp_norm,
            query_frames,
            d,
            weights.mlp_norm_gamma,
            weights.mlp_norm_beta,
            1e-5,
        )?;
        let expanded = linear(
            compute,
            &mlp_norm,
            query_frames,
            d,
            self.inner_dim,
            weights.mlp_expand_w_t,
            Some(weights.mlp_expand_b),
        )?;
        let mut activated = vec![0.0; expanded.len()];
        compute.gelu_f32(&expanded, &mut activated)?;
        let projected = linear(
            compute,
            &activated,
            query_frames,
            self.inner_dim,
            d,
            weights.mlp_project_w_t,
            Some(weights.mlp_project_b),
        )?;
        let output = add_masked_residual(&residual, &projected, query_mask, d)?;
        Ok(FireRedDecoderLayerOutput {
            output,
            key_cache: self_output.key_cache,
            value_cache: self_output.value_cache,
        })
    }

    fn validate(
        &self,
        input: &[f32],
        query_frames: usize,
        query_mask: &[bool],
        memory: &[f32],
        source_frames: usize,
        source_mask: &[bool],
        past_k: &[f32],
        past_v: &[f32],
        past_mask: &[bool],
        weights: FireRedDecoderLayerWeights<'_>,
    ) -> Result<()> {
        if self.d_model == 0
            || self.inner_dim == 0
            || self.source_dim == 0
            || query_frames == 0
            || query_mask.len() != query_frames
            || !query_mask.iter().all(|&valid| valid)
            || source_frames == 0
            || source_mask.len() != source_frames
            || !source_mask.iter().any(|&valid| valid)
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder layer shape or mask is invalid".to_owned(),
            ));
        }
        validate_attention_geometry(self.d_model, self.n_head)?;
        let input_len = query_frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder layer input overflow".to_owned())
        })?;
        let memory_len = source_frames.checked_mul(self.source_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder layer memory overflow".to_owned())
        })?;
        if input.len() != input_len
            || memory.len() != memory_len
            || past_k.len() != past_v.len()
            || past_k.len() % self.d_model != 0
            || past_mask.len() != past_k.len() / self.d_model
            || (past_mask.is_empty() && !past_k.is_empty())
            || self.source_dim != self.d_model
            || !all_finite(&[input, memory, past_k, past_v])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder layer input/cache shape or values are invalid".to_owned(),
            ));
        }
        let d = self.d_model;
        let inner = self.inner_dim;
        let matrix = d.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder layer matrix overflow".to_owned())
        })?;
        let qk = self.source_dim.checked_mul(d).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder cross matrix overflow".to_owned())
        })?;
        let mlp = d.checked_mul(inner).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder MLP matrix overflow".to_owned())
        })?;
        let vectors = [
            weights.self_norm_gamma,
            weights.self_norm_beta,
            weights.self_attention.q_b,
            weights.self_attention.v_b,
            weights.self_attention.output_b,
            weights.cross_norm_gamma,
            weights.cross_norm_beta,
            weights.cross_attention.q_b,
            weights.cross_attention.v_b,
            weights.cross_attention.output_b,
            weights.mlp_norm_gamma,
            weights.mlp_norm_beta,
            weights.mlp_expand_b,
            weights.mlp_project_b,
        ];
        if vectors
            .iter()
            .any(|values| values.iter().any(|value| !value.is_finite()))
            || vectors[0].len() != d
            || vectors[1].len() != d
            || vectors[2].len() != d
            || vectors[3].len() != d
            || vectors[4].len() != d
            || vectors[5].len() != d
            || vectors[6].len() != d
            || vectors[7].len() != d
            || vectors[8].len() != d
            || vectors[9].len() != d
            || vectors[10].len() != d
            || vectors[11].len() != d
            || vectors[12].len() != inner
            || vectors[13].len() != d
            || weights.self_attention.q_w_t.len() != matrix
            || weights.self_attention.k_w_t.len() != matrix
            || weights.self_attention.v_w_t.len() != matrix
            || weights.self_attention.output_w_t.len() != matrix
            || weights.cross_attention.q_w_t.len() != matrix
            || weights.cross_attention.k_w_t.len() != qk
            || weights.cross_attention.v_w_t.len() != qk
            || weights.cross_attention.output_w_t.len() != matrix
            || weights.mlp_expand_w_t.len() != mlp
            || weights.mlp_project_w_t.len() != mlp
            || !all_finite(&[
                weights.self_attention.q_w_t,
                weights.self_attention.k_w_t,
                weights.self_attention.v_w_t,
                weights.self_attention.output_w_t,
                weights.cross_attention.q_w_t,
                weights.cross_attention.k_w_t,
                weights.cross_attention.v_w_t,
                weights.cross_attention.output_w_t,
                weights.mlp_expand_w_t,
                weights.mlp_project_w_t,
            ])
        {
            return Err(VokraError::ModelLoad(
                "FireRed decoder layer learned operand preflight failed".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Embedding and fixed positional table path. `embedding` is source
/// `[vocab,d_model]`; `positional` is the flattened source
/// `[1,max_positions,d_model]` table. The source's embedding scale is always
/// `sqrt(d_model)` and is derived internally; callers cannot provide an
/// arbitrary scale.
#[derive(Debug, Clone, Copy)]
pub struct FireRedDecoderEmbedding {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_positions: usize,
}

impl FireRedDecoderEmbedding {
    pub fn forward(
        &self,
        token_ids: &[usize],
        positions: &[usize],
        embedding: &[f32],
        positional: &[f32],
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty()
            || token_ids.len() != positions.len()
            || self.vocab_size == 0
            || self.d_model == 0
            || self.max_positions == 0
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder embedding shape/scale is invalid".to_owned(),
            ));
        }
        let embedding_len = self.vocab_size.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder embedding table overflow".to_owned())
        })?;
        let positional_len = self
            .max_positions
            .checked_mul(self.d_model)
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed decoder positional table overflow".to_owned())
            })?;
        if embedding.len() != embedding_len
            || positional.len() != positional_len
            || !all_finite(&[embedding, positional])
            || token_ids.iter().any(|&id| id >= self.vocab_size)
            || positions
                .iter()
                .any(|&position| position >= self.max_positions)
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder embedding ids/table are invalid".to_owned(),
            ));
        }
        let output_len = token_ids.len().checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder embedding output overflow".to_owned())
        })?;
        let scale = (self.d_model as f32).sqrt();
        if !scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder embedding scale is non-finite".to_owned(),
            ));
        }
        let mut output = vec![0.0; output_len];
        for (row, (&token, &position)) in token_ids.iter().zip(positions).enumerate() {
            let token_start = token * self.d_model;
            let position_start = position * self.d_model;
            for channel in 0..self.d_model {
                output[row * self.d_model + channel] =
                    embedding[token_start + channel] * scale + positional[position_start + channel];
            }
        }
        ensure_finite(&output, "FireRed decoder embedding output")?;
        Ok(output)
    }
}

/// Final decoder LayerNorm and bias-free vocabulary projection. The source
/// projection rows are `[vocab,d_model]`; callers provide its transposed
/// Compute view `[d_model,vocab]`.
#[derive(Debug, Clone, Copy)]
pub struct FireRedDecoderOutputHead {
    pub d_model: usize,
    pub vocab_size: usize,
}

impl FireRedDecoderOutputHead {
    pub fn forward(
        &self,
        compute: &Compute,
        hidden: &[f32],
        frames: usize,
        norm_gamma: &[f32],
        norm_beta: &[f32],
        projection_w_t: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_len = frames.checked_mul(self.d_model).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder output hidden overflow".to_owned())
        })?;
        let matrix = self.d_model.checked_mul(self.vocab_size).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder output projection overflow".to_owned())
        })?;
        if frames == 0
            || self.d_model == 0
            || self.vocab_size == 0
            || hidden.len() != hidden_len
            || norm_gamma.len() != self.d_model
            || norm_beta.len() != self.d_model
            || projection_w_t.len() != matrix
            || !all_finite(&[hidden, norm_gamma, norm_beta, projection_w_t])
        {
            return Err(VokraError::InvalidArgument(
                "FireRed decoder output-head operands are invalid".to_owned(),
            ));
        }
        let mut normalized = vec![0.0; hidden_len];
        compute.layer_norm_f32(
            hidden,
            &mut normalized,
            frames,
            self.d_model,
            norm_gamma,
            norm_beta,
            1e-5,
        )?;
        linear(
            compute,
            &normalized,
            frames,
            self.d_model,
            self.vocab_size,
            projection_w_t,
            None,
        )
    }
}

/// Pure beam state transition helper. EOS or `max_len` marks a beam finished;
/// finished beams are immutable. Length normalisation is explicit and
/// caller-controlled so this helper does not silently invent a search policy.
#[derive(Debug, Clone, PartialEq)]
pub struct FireRedBeamState {
    pub tokens: Vec<usize>,
    pub score: f32,
    pub finished: bool,
}

impl FireRedBeamState {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            score: 0.0,
            finished: false,
        }
    }

    pub fn advance(
        &self,
        token: usize,
        log_probability: f32,
        eos_id: usize,
        max_len: usize,
    ) -> Result<Self> {
        if max_len == 0 || !log_probability.is_finite() || log_probability > 0.0 {
            return Err(VokraError::InvalidArgument(
                "FireRed beam transition has invalid score or max_len".to_owned(),
            ));
        }
        if self.finished || self.tokens.len() >= max_len {
            return Ok(self.clone());
        }
        let mut next = self.clone();
        next.tokens.push(token);
        next.score += log_probability;
        next.finished = token == eos_id || next.tokens.len() == max_len;
        if !next.score.is_finite() {
            return Err(VokraError::InvalidArgument(
                "FireRed beam score became non-finite".to_owned(),
            ));
        }
        Ok(next)
    }

    /// Returns the source GNMT-normalized score. `tokens` stores generated
    /// tokens only; its effective length is one SOS plus non-EOS tokens.
    pub fn ranked_score(&self, eos_id: usize, length_penalty: f32) -> Result<f32> {
        if !length_penalty.is_finite() || length_penalty < 0.0 {
            return Err(VokraError::InvalidArgument(
                "FireRed beam length penalty is invalid".to_owned(),
            ));
        }
        let non_eos = self.tokens.iter().filter(|&&token| token != eos_id).count();
        let length = (1usize.checked_add(non_eos).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed beam length overflow".to_owned())
        })?) as f32;
        let denominator = ((5.0 + length) / 6.0).powf(length_penalty);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "FireRed beam length normalization is non-finite".to_owned(),
            ));
        }
        Ok(self.score / denominator)
    }
}

/// Applies the source EOS score penalty before beam selection. The source
/// default is `1.0`; values outside `(0, 1]` are rejected instead of silently
/// changing search behavior.
pub fn apply_fire_red_eos_penalty(
    token_scores: &mut [f32],
    eos_id: usize,
    eos_penalty: f32,
) -> Result<()> {
    if eos_id >= token_scores.len()
        || !eos_penalty.is_finite()
        || eos_penalty <= 0.0
        || eos_penalty > 1.0
        || token_scores.iter().any(|score| !score.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "FireRed EOS penalty or token scores are invalid".to_owned(),
        ));
    }
    token_scores[eos_id] *= eos_penalty;
    ensure_finite(token_scores, "FireRed EOS-penalized token scores")
}

fn validate_attention_geometry(d_model: usize, n_head: usize) -> Result<()> {
    if d_model == 0 || n_head == 0 || d_model % n_head != 0 {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder attention geometry is invalid".to_owned(),
        ));
    }
    Ok(())
}

fn validate_linear_operands(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<()> {
    let input_len = rows.checked_mul(input_dim).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder linear input overflow".to_owned())
    })?;
    let weight_len = input_dim.checked_mul(output_dim).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder linear weight overflow".to_owned())
    })?;
    if rows == 0
        || input.len() != input_len
        || weight.len() != weight_len
        || bias.is_some_and(|values| values.len() != output_dim)
        || !all_finite(&[input, weight])
        || bias.is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder linear operands are invalid".to_owned(),
        ));
    }
    Ok(())
}

fn linear(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
) -> Result<Vec<f32>> {
    validate_linear_operands(input, rows, input_dim, output_dim, weight, bias)?;
    let output_len = rows.checked_mul(output_dim).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder linear output overflow".to_owned())
    })?;
    let mut output = vec![0.0; output_len];
    compute.gemm_f32(
        rows,
        output_dim,
        input_dim,
        input,
        weight,
        bias,
        &mut output,
    )?;
    ensure_finite(&output, "FireRed decoder linear output")?;
    Ok(output)
}

fn scaled_dot_product_attention(
    compute: &Compute,
    q: &[f32],
    query_frames: usize,
    k: &[f32],
    key_frames: usize,
    v: &[f32],
    d_model: usize,
    n_head: usize,
    key_mask: &[bool],
    causal_past: Option<usize>,
) -> Result<Vec<f32>> {
    validate_attention_geometry(d_model, n_head)?;
    let q_len = query_frames.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder attention query overflow".to_owned())
    })?;
    let kv_len = key_frames.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder attention key overflow".to_owned())
    })?;
    if query_frames == 0
        || key_frames == 0
        || q.len() != q_len
        || k.len() != kv_len
        || v.len() != kv_len
        || key_mask.len() != key_frames
        || !key_mask.iter().any(|&valid| valid)
        || causal_past.is_some_and(|past| past > key_frames)
        || !all_finite(&[q, k, v])
    {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder attention shape/mask is invalid".to_owned(),
        ));
    }
    let head_dim = d_model / n_head;
    let score_len = n_head
        .checked_mul(query_frames)
        .and_then(|value| value.checked_mul(key_frames))
        .ok_or_else(|| VokraError::InvalidArgument("FireRed decoder score overflow".to_owned()))?;
    let mut scores = vec![0.0; score_len];
    let scale = (head_dim as f32).sqrt().recip();
    if !scale.is_finite() {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder attention scale is non-finite".to_owned(),
        ));
    }
    for head in 0..n_head {
        let offset = head * head_dim;
        let q_head_len = query_frames.checked_mul(head_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder query-head overflow".to_owned())
        })?;
        let k_head_len = head_dim.checked_mul(key_frames).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder key-head overflow".to_owned())
        })?;
        let mut q_head = vec![0.0; q_head_len];
        let mut k_head = vec![0.0; k_head_len];
        for row in 0..query_frames {
            q_head[row * head_dim..(row + 1) * head_dim]
                .copy_from_slice(&q[row * d_model + offset..row * d_model + offset + head_dim]);
        }
        for row in 0..key_frames {
            for dim in 0..head_dim {
                k_head[dim * key_frames + row] = k[row * d_model + offset + dim];
            }
        }
        let score_start = head
            .checked_mul(query_frames)
            .and_then(|value| value.checked_mul(key_frames))
            .ok_or_else(|| {
                VokraError::InvalidArgument("FireRed decoder score overflow".to_owned())
            })?;
        compute.gemm_f32(
            query_frames,
            key_frames,
            head_dim,
            &q_head,
            &k_head,
            None,
            &mut scores[score_start
                ..score_start
                    + query_frames.checked_mul(key_frames).ok_or_else(|| {
                        VokraError::InvalidArgument("FireRed decoder score overflow".to_owned())
                    })?],
        )?;
        for query in 0..query_frames {
            for key in 0..key_frames {
                let causal_ok = causal_past.map_or(true, |past| key <= past + query);
                if !key_mask[key] || !causal_ok {
                    scores[score_start + query * key_frames + key] = -f32::MAX;
                } else {
                    scores[score_start + query * key_frames + key] *= scale;
                }
            }
        }
    }
    let rows = n_head.checked_mul(query_frames).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder softmax rows overflow".to_owned())
    })?;
    let mut probabilities = vec![0.0; score_len];
    compute.softmax_f32(&scores, &mut probabilities, rows, key_frames)?;
    for head in 0..n_head {
        for query in 0..query_frames {
            let start = (head * query_frames + query) * key_frames;
            let mut sum = 0.0f32;
            for key in 0..key_frames {
                let causal_ok = causal_past.map_or(true, |past| key <= past + query);
                if !key_mask[key] || !causal_ok {
                    probabilities[start + key] = 0.0;
                }
                sum += probabilities[start + key];
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(VokraError::InvalidArgument(
                    "FireRed decoder attention row is all-masked".to_owned(),
                ));
            }
            for key in 0..key_frames {
                probabilities[start + key] /= sum;
            }
        }
    }
    let output_len = query_frames.checked_mul(d_model).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed decoder attention output overflow".to_owned())
    })?;
    let mut output = vec![0.0; output_len];
    for head in 0..n_head {
        let v_head_len = key_frames.checked_mul(head_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder value-head overflow".to_owned())
        })?;
        let mut v_head = vec![0.0; v_head_len];
        for key in 0..key_frames {
            v_head[key * head_dim..(key + 1) * head_dim].copy_from_slice(
                &v[key * d_model + head * head_dim..key * d_model + (head + 1) * head_dim],
            );
        }
        let context_len = query_frames.checked_mul(head_dim).ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder context overflow".to_owned())
        })?;
        let mut context = vec![0.0; context_len];
        compute.gemm_f32(
            query_frames,
            head_dim,
            key_frames,
            &probabilities
                [head * query_frames * key_frames..(head + 1) * query_frames * key_frames],
            &v_head,
            None,
            &mut context,
        )?;
        for query in 0..query_frames {
            output[query * d_model + head * head_dim..query * d_model + (head + 1) * head_dim]
                .copy_from_slice(&context[query * head_dim..(query + 1) * head_dim]);
        }
    }
    ensure_finite(&output, "FireRed decoder attention output")?;
    Ok(output)
}

fn combined_cache_mask(past_mask: &[bool], query_mask: &[bool]) -> Result<Vec<bool>> {
    if query_mask.is_empty() {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder query mask is empty".to_owned(),
        ));
    }
    let total = past_mask
        .len()
        .checked_add(query_mask.len())
        .ok_or_else(|| {
            VokraError::InvalidArgument("FireRed decoder cache mask overflow".to_owned())
        })?;
    let mut mask = Vec::with_capacity(total);
    mask.extend_from_slice(past_mask);
    mask.extend_from_slice(query_mask);
    Ok(mask)
}

fn add_masked_residual(
    left: &[f32],
    right: &[f32],
    mask: &[bool],
    d_model: usize,
) -> Result<Vec<f32>> {
    if left.len() != right.len()
        || mask.len().checked_mul(d_model) != Some(left.len())
        || !mask.iter().all(|&valid| valid)
        || !all_finite(&[left, right])
    {
        return Err(VokraError::InvalidArgument(
            "FireRed decoder residual shape or values are invalid".to_owned(),
        ));
    }
    let mut output = vec![0.0; left.len()];
    for frame in 0..mask.len() {
        for channel in 0..d_model {
            let index = frame * d_model + channel;
            output[index] = left[index] + right[index];
        }
    }
    ensure_finite(&output, "FireRed decoder residual")?;
    Ok(output)
}

fn ensure_finite(values: &[f32], label: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "{label} is non-finite"
        )));
    }
    Ok(())
}

fn all_finite(slices: &[&[f32]]) -> bool {
    slices
        .iter()
        .flat_map(|slice| slice.iter())
        .all(|value| value.is_finite())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn glu_split(input: &[f32], frames: usize, channels: usize) -> Result<Vec<f32>> {
    let expected = channels
        .checked_mul(2)
        .and_then(|width| width.checked_mul(frames))
        .ok_or_else(|| VokraError::InvalidArgument("FireRed GLU shape overflow".to_owned()))?;
    if input.len() != expected || channels == 0 || frames == 0 {
        return Err(VokraError::InvalidArgument(
            "FireRed GLU input shape mismatch".to_owned(),
        ));
    }
    let output_len = channels.checked_mul(frames).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed GLU output shape overflow".to_owned())
    })?;
    let mut output = vec![0.0; output_len];
    for channel in 0..channels {
        for frame in 0..frames {
            let index = channel * frames + frame;
            output[index] = input[index] * sigmoid(input[(channel + channels) * frames + frame]);
        }
    }
    Ok(output)
}

fn depthwise_same(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    frames: usize,
    weight: &[f32],
    kernel_size: usize,
) -> Result<Vec<f32>> {
    let input_len = channels.checked_mul(frames).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed depthwise input shape overflow".to_owned())
    })?;
    let weight_len = channels.checked_mul(kernel_size).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed depthwise weight shape overflow".to_owned())
    })?;
    if channels == 0
        || frames == 0
        || kernel_size == 0
        || kernel_size % 2 == 0
        || input.len() != input_len
        || weight.len() != weight_len
        || !all_finite(&[input, weight])
    {
        return Err(VokraError::InvalidArgument(
            "FireRed depthwise operand shape or values are invalid".to_owned(),
        ));
    }
    let mut output = vec![0.0; input_len];
    let padding = kernel_size / 2;
    for channel in 0..channels {
        let input_start = channel * frames;
        let weight_start = channel * kernel_size;
        compute.conv1d_f32(
            &input[input_start..input_start + frames],
            1,
            frames,
            &weight[weight_start..weight_start + kernel_size],
            1,
            kernel_size,
            None,
            1,
            padding,
            &mut output[input_start..input_start + frames],
        )?;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "FireRed depthwise output is non-finite".to_owned(),
        ));
    }
    Ok(output)
}

fn half_residual(input: &[f32], branch_with_residual: &[f32]) -> Result<Vec<f32>> {
    if input.len() != branch_with_residual.len() {
        return Err(VokraError::InvalidArgument(
            "FireRed Conformer residual shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for ((value, input), branch) in output.iter_mut().zip(input).zip(branch_with_residual) {
        *value = 0.5 * *input + 0.5 * *branch;
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(
                "FireRed Conformer half residual is non-finite".to_owned(),
            ));
        }
    }
    Ok(output)
}

fn transpose_frame_to_channel(input: &[f32], frames: usize, channels: usize) -> Result<Vec<f32>> {
    let expected = frames.checked_mul(channels).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed transpose shape overflow".to_owned())
    })?;
    if input.len() != expected {
        return Err(VokraError::InvalidArgument(
            "FireRed frame-major input shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; expected];
    for frame in 0..frames {
        for channel in 0..channels {
            output[channel * frames + frame] = input[frame * channels + channel];
        }
    }
    Ok(output)
}

fn transpose_channel_to_frame(input: &[f32], frames: usize, channels: usize) -> Result<Vec<f32>> {
    let expected = frames.checked_mul(channels).ok_or_else(|| {
        VokraError::InvalidArgument("FireRed transpose shape overflow".to_owned())
    })?;
    if input.len() != expected {
        return Err(VokraError::InvalidArgument(
            "FireRed channel-major input shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; expected];
    for frame in 0..frames {
        for channel in 0..channels {
            output[frame * channels + channel] = input[channel * frames + frame];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_linear_binding_transposes_only_two_dimensional_weights() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_2d(&values, &[2, 3]).unwrap(),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert!(transpose_2d(&values, &[6]).is_err());
        assert!(transpose_2d(&values[..5], &[2, 3]).is_err());
    }

    #[test]
    fn cmvn_matches_upstream_formula() {
        // dim=2, count=4; means=(2, 3), variance=(1, 4).
        let cmvn = FireRedCmvn::from_stats(&[8.0, 12.0, 4.0, 20.0, 52.0, 4.0], 2).unwrap();
        let mut values = [3.0, 5.0, 1.0, 7.0];
        cmvn.apply(&mut values, 2).unwrap();
        assert_eq!(values, [1.0, 1.0, -1.0, 2.0]);
    }

    #[test]
    fn relative_positions_have_source_window_shape() {
        let positions = relative_positional_encoding(4, 8, 3).unwrap();
        assert_eq!(positions.len(), 5 * 4);
        assert_eq!(&positions[8..12], &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn conv2d_stem_uses_two_dense_stride_stages() {
        let compute = Compute::cpu();
        let stem = FireRedConv2dSubsampling {
            out_channels: 2,
            d_model: 3,
        };
        let input = vec![1.0; 11 * 9];
        let (output, frames) = stem
            .forward(
                &compute,
                &input,
                11,
                9,
                &vec![1.0; 2 * 9],
                &vec![0.0; 2],
                &vec![1.0; 2 * 2 * 9],
                &vec![0.0; 2],
                &vec![1.0; 3 * 2],
                &vec![0.0; 3],
            )
            .unwrap();
        assert_eq!(frames, 2);
        assert_eq!(output.len(), 2 * 3);
    }

    #[test]
    fn relative_attention_dispatches_cpu_without_fallback() {
        let compute = Compute::cpu();
        let attention = FireRedRelativeAttention {
            d_model: 4,
            n_head: 2,
        };
        let output = attention
            .forward(
                &compute,
                &vec![1.0; 3 * 4],
                &vec![0.0; 5 * 4],
                3,
                &vec![1.0; 4 * 4],
                &vec![1.0; 4 * 4],
                &vec![1.0; 4 * 4],
                &vec![1.0; 4 * 4],
                &vec![1.0; 4],
                &vec![0.0; 4],
                &vec![1.0; 4],
                &vec![0.0; 4],
                &vec![1.0; 4],
                &vec![0.0; 4],
                &vec![0.0; 4],
                &vec![0.0; 4],
            )
            .unwrap();
        assert_eq!(output.len(), 3 * 4);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn relative_attention_output_projection_adds_source_residual() {
        let compute = Compute::cpu();
        let attention = FireRedRelativeAttention {
            d_model: 2,
            n_head: 1,
        };
        let input = [1.0, -2.0, 3.0, 4.0];
        let output = attention
            .forward_with_output(
                &compute, &input, &[0.0; 6], 2, &[1.0; 4], &[1.0; 4], &[1.0; 4], &[1.0; 4],
                &[1.0; 2], &[0.0; 2], &[1.0; 2], &[0.0; 2], &[1.0; 2], &[0.0; 2], &[0.0; 2],
                &[0.0; 2], &[0.0; 4],
            )
            .unwrap();
        // A zero bias-free output projection leaves only the source residual.
        assert_eq!(output, input);
    }

    #[test]
    fn relative_attention_key_mask_excludes_extreme_masked_frame() {
        let compute = Compute::cpu();
        let attention = FireRedRelativeAttention {
            d_model: 2,
            n_head: 1,
        };
        let common = (
            &[0.0; 10][..],
            3usize,
            &[1.0, 0.0, 0.0, 1.0][..],
            &[1.0, 0.0, 0.0, 1.0][..],
            &[1.0, 0.0, 0.0, 1.0][..],
            &[1.0, 0.0, 0.0, 1.0][..],
            &[1.0, 1.0][..],
            &[0.0, 0.0][..],
            &[1.0, 1.0][..],
            &[0.0, 0.0][..],
            &[1.0, 1.0][..],
            &[0.0, 0.0][..],
            &[0.0, 0.0][..],
            &[0.0, 0.0][..],
        );
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut changed = input;
        changed[2..4].copy_from_slice(&[1000.0, -1000.0]);
        let run = |values: &[f32]| {
            attention.forward_with_mask(
                &compute,
                values,
                common.0,
                common.1,
                common.2,
                common.3,
                common.4,
                common.5,
                common.6,
                common.7,
                common.8,
                common.9,
                common.10,
                common.11,
                common.12,
                common.13,
                Some(&[true, false, true]),
            )
        };
        let baseline = run(&input).unwrap();
        let altered = run(&changed).unwrap();
        for frame in [0usize, 2] {
            for channel in 0..2 {
                let index = frame * 2 + channel;
                assert!((baseline[index] - altered[index]).abs() <= 1e-5);
            }
        }
        assert!(
            attention
                .forward_with_mask(
                    &compute,
                    &input,
                    common.0,
                    common.1,
                    common.2,
                    common.3,
                    common.4,
                    common.5,
                    common.6,
                    common.7,
                    common.8,
                    common.9,
                    common.10,
                    common.11,
                    common.12,
                    common.13,
                    Some(&[false, false, false]),
                )
                .is_err()
        );
    }

    #[test]
    fn feed_forward_keeps_internal_residual_and_half_step_is_explicit() {
        let compute = Compute::cpu();
        let ffn = FireRedConformerFeedForward {
            d_model: 2,
            inner_dim: 8,
        };
        let input = [1.0, 3.0, 2.0, 4.0];
        let ln_gamma = [1.25, 0.75];
        let ln_beta = [0.1, -0.2];
        let expand_w_t: Vec<f32> = (0..16).map(|index| 0.05 * (index as f32 + 1.0)).collect();
        let expand_b: Vec<f32> = (0..8).map(|index| -0.1 + index as f32 * 0.03).collect();
        let project_w_t: Vec<f32> = (0..16).map(|index| -0.04 + index as f32 * 0.02).collect();
        let project_b = [0.25, -0.35];
        let output = ffn
            .forward(
                &compute,
                &input,
                2,
                &ln_gamma,
                &ln_beta,
                &expand_w_t,
                &expand_b,
                &project_w_t,
                &project_b,
            )
            .unwrap();
        let mut expected = vec![0.0; 4];
        for frame in 0..2 {
            let row = &input[frame * 2..frame * 2 + 2];
            let mean = (row[0] + row[1]) / 2.0;
            let variance = ((row[0] - mean).powi(2) + (row[1] - mean).powi(2)) / 2.0;
            let inv_std = (variance + 1e-5).sqrt().recip();
            let normalized = [
                (row[0] - mean) * inv_std * ln_gamma[0] + ln_beta[0],
                (row[1] - mean) * inv_std * ln_gamma[1] + ln_beta[1],
            ];
            let mut hidden = [0.0; 8];
            for inner in 0..8 {
                hidden[inner] = expand_b[inner]
                    + normalized[0] * expand_w_t[inner]
                    + normalized[1] * expand_w_t[8 + inner];
                hidden[inner] *= sigmoid(hidden[inner]);
            }
            for channel in 0..2 {
                let mut value = project_b[channel];
                for inner in 0..8 {
                    value += hidden[inner] * project_w_t[inner * 2 + channel];
                }
                expected[frame * 2 + channel] = value + row[channel];
            }
        }
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
        let half = half_residual(&input, &output).unwrap();
        for ((actual, left), right) in half.iter().zip(input).zip(output) {
            assert!((*actual - (0.5 * left + 0.5 * right)).abs() <= 1e-6);
        }
    }

    #[test]
    fn glu_and_depthwise_same_padding_match_scalar_oracles() {
        let glu = glu_split(&[2.0, -1.0, 0.0, 1.0], 2, 1).unwrap();
        assert!((glu[0] - 1.0).abs() < 1.0e-6);
        assert!((glu[1] + 0.7310586).abs() < 1.0e-6);

        let depthwise =
            depthwise_same(&Compute::cpu(), &[1.0, 2.0, 3.0], 1, 3, &[1.0, 2.0, 1.0], 3).unwrap();
        assert_eq!(depthwise, [4.0, 8.0, 8.0]);
    }

    #[test]
    fn convolution_masks_residual_and_rejects_even_kernel() {
        let convolution = FireRedConformerConvolution {
            d_model: 1,
            kernel_size: 3,
        };
        let output = convolution
            .forward(
                &Compute::cpu(),
                &[1.0, 2.0, 3.0],
                3,
                &[true, false, true],
                &[0.0; 4],
                &[0.0; 6],
                &[1.0; 2],
                &[0.0; 2],
                &[0.0; 2],
                &[1.0],
                &[0.0],
            )
            .unwrap();
        assert_eq!(output, [1.0, 0.0, 3.0]);
        let even = FireRedConformerConvolution {
            d_model: 1,
            kernel_size: 2,
        };
        assert!(
            even.forward(
                &Compute::cpu(),
                &[1.0],
                1,
                &[true],
                &[0.0; 4],
                &[0.0; 4],
                &[1.0; 2],
                &[0.0; 2],
                &[0.0; 2],
                &[1.0],
                &[0.0],
            )
            .is_err()
        );
    }

    #[test]
    fn convolution_nonzero_path_matches_independent_scalar_oracle() {
        let convolution = FireRedConformerConvolution {
            d_model: 2,
            kernel_size: 3,
        };
        let input = [1.0, 2.0, 3.0, 5.0];
        let pre_gamma = [1.1, 0.9];
        let pre_beta = [0.1, -0.2];
        let pointwise_in: Vec<f32> = (0..16).map(|index| 0.02 + index as f32 * 0.01).collect();
        let depthwise: Vec<f32> = (0..12).map(|index| 0.05 + index as f32 * 0.01).collect();
        let depth_gamma = [1.0, 0.9, 1.1, 0.8];
        let depth_beta = [0.1, -0.1, 0.2, -0.2];
        let pointwise_out: Vec<f32> = (0..8).map(|index| -0.03 + index as f32 * 0.015).collect();
        let actual = convolution
            .forward(
                &Compute::cpu(),
                &input,
                2,
                &[true, true],
                &pointwise_in,
                &depthwise,
                &depth_gamma,
                &depth_beta,
                &pointwise_out,
                &pre_gamma,
                &pre_beta,
            )
            .unwrap();

        let frames = 2;
        let mut normalized = [0.0; 4];
        for frame in 0..frames {
            let row = &input[frame * 2..frame * 2 + 2];
            let mean = (row[0] + row[1]) / 2.0;
            let variance = ((row[0] - mean).powi(2) + (row[1] - mean).powi(2)) / 2.0;
            let inv_std = (variance + 1e-5).sqrt().recip();
            normalized[frame * 2] = (row[0] - mean) * inv_std * pre_gamma[0] + pre_beta[0];
            normalized[frame * 2 + 1] = (row[1] - mean) * inv_std * pre_gamma[1] + pre_beta[1];
        }
        let mut pointwise = [0.0; 16];
        for channel in 0..8 {
            for frame in 0..frames {
                pointwise[channel * frames + frame] = pointwise_in[channel * 2]
                    * normalized[frame * 2]
                    + pointwise_in[channel * 2 + 1] * normalized[frame * 2 + 1];
            }
        }
        let mut glu = [0.0; 8];
        for channel in 0..4 {
            for frame in 0..frames {
                let index = channel * frames + frame;
                glu[index] = pointwise[index] * sigmoid(pointwise[(channel + 4) * frames + frame]);
            }
        }
        let mut depth = [0.0; 8];
        for channel in 0..4 {
            for frame in 0..frames {
                let mut value = 0.0;
                for tap in 0..3 {
                    let source = frame as isize + tap as isize - 1;
                    if (0..frames as isize).contains(&source) {
                        value +=
                            glu[channel * frames + source as usize] * depthwise[channel * 3 + tap];
                    }
                }
                depth[channel * frames + frame] = value;
            }
        }
        let mut activated = [0.0; 8];
        for frame in 0..frames {
            let row = [
                depth[frame],
                depth[frames + frame],
                depth[2 * frames + frame],
                depth[3 * frames + frame],
            ];
            let mean = row.iter().sum::<f32>() / 4.0;
            let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / 4.0;
            let inv_std = (variance + 1e-5).sqrt().recip();
            for channel in 0..4 {
                let value =
                    (row[channel] - mean) * inv_std * depth_gamma[channel] + depth_beta[channel];
                activated[channel * frames + frame] = value * sigmoid(value);
            }
        }
        let mut expected = input;
        for frame in 0..frames {
            for channel in 0..2 {
                let mut value = 0.0;
                for inner in 0..4 {
                    value += activated[inner * frames + frame] * pointwise_out[channel * 4 + inner];
                }
                expected[frame * 2 + channel] += value;
            }
        }
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn conformer_block_runs_nontrivial_path_and_preflights_late_operands() {
        let compute = Compute::cpu();
        let block = FireRedConformerBlock {
            d_model: 2,
            inner_dim: 8,
            n_head: 1,
            kernel_size: 3,
        };
        let input = [0.5, -1.0, 2.0, 3.0];
        let mask = [true, false];
        let ln2 = [1.0, 0.9];
        let beta2 = [0.1, -0.1];
        let ffn_w = vec![0.03; 16];
        let ffn_b = vec![0.02; 8];
        let ffn_out_w = vec![0.04; 16];
        let ffn_out_b = [0.01, -0.02];
        let positions = vec![0.05; 6];
        let matrix = vec![0.02; 4];
        let attn_gamma = [1.0, 1.0];
        let attn_beta = [0.0, 0.0];
        let attn_bias = [0.01, -0.01];
        let conv_in = vec![0.03; 16];
        let depthwise = vec![0.02; 12];
        let conv_gamma = [1.0, 0.9, 1.0, 0.9];
        let conv_beta = [0.0, 0.1, 0.0, 0.1];
        let conv_out = vec![0.04; 8];
        let final_beta = [0.0, 0.0];
        let weights = FireRedConformerBlockWeights {
            ffn1_ln_gamma: &ln2,
            ffn1_ln_beta: &beta2,
            ffn1_expand_w_t: &ffn_w,
            ffn1_expand_b: &ffn_b,
            ffn1_project_w_t: &ffn_out_w,
            ffn1_project_b: &ffn_out_b,
            attention_positions: &positions,
            attention_q_w_t: &matrix,
            attention_k_w_t: &matrix,
            attention_v_w_t: &matrix,
            attention_linear_pos_w_t: &matrix,
            attention_q_norm_gamma: &attn_gamma,
            attention_q_norm_beta: &attn_beta,
            attention_k_norm_gamma: &attn_gamma,
            attention_k_norm_beta: &attn_beta,
            attention_v_norm_gamma: &attn_gamma,
            attention_v_norm_beta: &attn_beta,
            attention_bias_u: &attn_bias,
            attention_bias_v: &attn_bias,
            attention_output_w_t: &matrix,
            conv_pointwise_in_w: &conv_in,
            conv_depthwise_w: &depthwise,
            conv_depthwise_ln_gamma: &conv_gamma,
            conv_depthwise_ln_beta: &conv_beta,
            conv_pointwise_out_w: &conv_out,
            conv_pre_ln_gamma: &attn_gamma,
            conv_pre_ln_beta: &attn_beta,
            ffn2_ln_gamma: &ln2,
            ffn2_ln_beta: &beta2,
            ffn2_expand_w_t: &ffn_w,
            ffn2_expand_b: &ffn_b,
            ffn2_project_w_t: &ffn_out_w,
            ffn2_project_b: &ffn_out_b,
            final_ln_gamma: &attn_gamma,
            final_ln_beta: &final_beta,
        };
        let output = block.forward(&compute, &input, 2, &mask, &weights).unwrap();
        assert!(output[..2].iter().all(|value| value.is_finite()));
        assert_eq!(output[2..], [0.0, 0.0]);

        // A late final-LN corruption is rejected by the block preflight,
        // before FFN1 can dispatch any learned operation.
        let late_final_beta = [f32::NAN, 0.0];
        let late_weights = FireRedConformerBlockWeights {
            final_ln_beta: &late_final_beta,
            ..weights
        };
        assert!(
            block
                .validate_operands(&input, 2, &mask, &late_weights)
                .is_err()
        );
    }

    fn encoder_fixture_weights(
        final_ln_beta: &'static [f32; 1],
    ) -> FireRedConformerBlockWeights<'static> {
        static ONE: [f32; 1] = [1.0];
        static ZERO: [f32; 1] = [0.0];
        static EXPAND_W: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        static EXPAND_B: [f32; 4] = [0.01, 0.02, 0.03, 0.04];
        static PROJECT_W: [f32; 4] = [0.2, 0.3, 0.4, 0.5];
        static PROJECT_B: [f32; 1] = [0.1];
        static POSITION: [f32; 1] = [0.05];
        static CONV_IN: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        static DEPTHWISE: [f32; 2] = [0.5, 0.25];
        static DEPTH_GAMMA: [f32; 2] = [1.0, 1.0];
        static DEPTH_BETA: [f32; 2] = [0.0, 0.1];
        static CONV_OUT: [f32; 2] = [0.2, 0.3];

        FireRedConformerBlockWeights {
            ffn1_ln_gamma: &ONE,
            ffn1_ln_beta: &ZERO,
            ffn1_expand_w_t: &EXPAND_W,
            ffn1_expand_b: &EXPAND_B,
            ffn1_project_w_t: &PROJECT_W,
            ffn1_project_b: &PROJECT_B,
            attention_positions: &POSITION,
            attention_q_w_t: &ONE,
            attention_k_w_t: &ONE,
            attention_v_w_t: &ONE,
            attention_linear_pos_w_t: &ONE,
            attention_q_norm_gamma: &ONE,
            attention_q_norm_beta: &ZERO,
            attention_k_norm_gamma: &ONE,
            attention_k_norm_beta: &ZERO,
            attention_v_norm_gamma: &ONE,
            attention_v_norm_beta: &ZERO,
            attention_bias_u: &ZERO,
            attention_bias_v: &ZERO,
            attention_output_w_t: &ONE,
            conv_pointwise_in_w: &CONV_IN,
            conv_depthwise_w: &DEPTHWISE,
            conv_depthwise_ln_gamma: &DEPTH_GAMMA,
            conv_depthwise_ln_beta: &DEPTH_BETA,
            conv_pointwise_out_w: &CONV_OUT,
            conv_pre_ln_gamma: &ONE,
            conv_pre_ln_beta: &ZERO,
            ffn2_ln_gamma: &ONE,
            ffn2_ln_beta: &ZERO,
            ffn2_expand_w_t: &EXPAND_W,
            ffn2_expand_b: &EXPAND_B,
            ffn2_project_w_t: &PROJECT_W,
            ffn2_project_b: &PROJECT_B,
            final_ln_gamma: &ONE,
            final_ln_beta,
        }
    }

    #[test]
    fn encoder_runs_all_sixteen_ordered_blocks_and_preflights_late_layer() {
        static BAD_FINAL_BETA: [f32; 1] = [f32::NAN];
        static LAYER_BETAS: [[f32; 1]; 16] = [
            [0.0],
            [0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5],
            [0.6],
            [0.7],
            [0.8],
            [0.9],
            [1.0],
            [1.1],
            [1.2],
            [1.3],
            [1.4],
            [1.5],
        ];
        let pinned = FireRedConformerEncoder::authenticated();
        assert_eq!(pinned.d_model, 1_280);
        assert_eq!(pinned.inner_dim, 5_120);
        assert_eq!(pinned.n_head, 20);
        assert_eq!(pinned.kernel_size, 33);

        let layers: Vec<_> = (0..16)
            .map(|index| encoder_fixture_weights(&LAYER_BETAS[index]))
            .collect();
        // Test-only small geometry keeps this unit test cheap; production
        // callers can only construct the pinned authenticated geometry.
        let encoder = FireRedConformerEncoder {
            d_model: 1,
            inner_dim: 4,
            n_head: 1,
            kernel_size: 1,
        };
        let output = encoder
            .forward(&Compute::cpu(), &[1.0], 1, &[true], &layers)
            .unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());

        let block = FireRedConformerBlock {
            d_model: 1,
            inner_dim: 4,
            n_head: 1,
            kernel_size: 1,
        };
        let mut expected = vec![1.0];
        for weights in &layers {
            expected = block
                .forward(&Compute::cpu(), &expected, 1, &[true], weights)
                .unwrap();
        }
        assert_eq!(output, expected, "encoder must fold blocks in source order");

        let mut reverse = layers.clone();
        reverse.reverse();
        let reversed = encoder
            .forward(&Compute::cpu(), &[1.0], 1, &[true], &reverse)
            .unwrap();
        assert_ne!(
            output, reversed,
            "reversing distinct layers must change output"
        );

        let missing = &layers[..15];
        assert!(
            encoder
                .forward(&Compute::cpu(), &[1.0], 1, &[true], missing)
                .is_err()
        );

        let mut malformed = layers;
        malformed[15] = encoder_fixture_weights(&BAD_FINAL_BETA);
        let error = encoder
            .forward(&Compute::cpu(), &[1.0], 1, &[true], &malformed)
            .expect_err("late layer corruption must fail before layer zero dispatch");
        assert!(error.to_string().contains("layer 15 preflight"));
    }

    #[test]
    fn relative_shift_matches_upstream_index_oracle() {
        let frames = 3;
        let positions = 2 * frames - 1;
        let input: Vec<f32> = (0..frames * positions).map(|value| value as f32).collect();
        // This is the equivalent of upstream RelPosMultiHeadAttention._rel_shift:
        // reshape [B,H,Q,2Q-1], discard the first column, reshape and select the
        // first Q columns.  The explicit oracle keeps this test independent of
        // the compact native index implementation above.
        let mut padded = vec![0.0; frames * (positions + 1)];
        for query in 0..frames {
            padded[query * (positions + 1) + 1..(query + 1) * (positions + 1)]
                .copy_from_slice(&input[query * positions..(query + 1) * positions]);
        }
        let mut oracle = vec![0.0; frames * frames];
        for query in 0..frames {
            for key in 0..frames {
                oracle[query * frames + key] = padded[frames + query * positions + key];
            }
        }
        assert_eq!(rel_shift(&input, frames, positions).unwrap(), oracle);
    }

    #[test]
    fn decoder_self_attention_causal_and_incremental_cache_match() {
        let compute = Compute::cpu();
        let attention = FireRedDecoderSelfAttention {
            d_model: 2,
            n_head: 1,
        };
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0, 0.0];
        let weights = FireRedDecoderSelfAttentionWeights {
            q_w_t: &identity,
            q_b: &zero,
            k_w_t: &identity,
            v_w_t: &identity,
            v_b: &zero,
            output_w_t: &identity,
            output_b: &zero,
        };
        let input = [1.0, 0.0, 0.0, 2.0];
        let full = attention
            .forward(&compute, &input, 2, &[], &[], &[true, true], weights)
            .unwrap();
        // Query zero cannot see query one under the causal boundary.
        assert!((full.output[0] - 1.0).abs() < 1e-6);
        assert!(full.output[3] > 1.0 && full.output[3] < 2.0);
        let first = attention
            .forward(&compute, &input[..2], 1, &[], &[], &[true], weights)
            .unwrap();
        let second = attention
            .forward(
                &compute,
                &input[2..],
                1,
                &first.key_cache,
                &first.value_cache,
                &[true, true],
                weights,
            )
            .unwrap();
        assert_eq!(second.output.as_slice(), &full.output[2..]);
        assert_eq!(second.key_cache.len(), 4);
        assert!(
            attention
                .forward(&compute, &input[..2], 1, &[], &[], &[false], weights)
                .is_err()
        );
    }

    #[test]
    fn decoder_cross_attention_source_mask_and_embedding_are_fail_closed() {
        let compute = Compute::cpu();
        let attention = FireRedDecoderCrossAttention {
            d_model: 2,
            source_dim: 2,
            n_head: 1,
        };
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0, 0.0];
        let weights = FireRedDecoderCrossAttentionWeights {
            q_w_t: &identity,
            q_b: &zero,
            k_w_t: &identity,
            v_w_t: &identity,
            v_b: &zero,
            output_w_t: &identity,
            output_b: &zero,
        };
        let query = [1.0, 0.0];
        let memory = [1.0, 0.0, 1000.0, -1000.0];
        let baseline = attention
            .forward(
                &compute,
                &query,
                1,
                &memory,
                2,
                &[true, false],
                None,
                weights,
            )
            .unwrap();
        let altered = attention
            .forward(
                &compute,
                &query,
                1,
                &[1.0, 0.0, -1000.0, 1000.0],
                2,
                &[true, false],
                None,
                weights,
            )
            .unwrap();
        assert_eq!(baseline, altered);
        assert!(
            attention
                .forward(
                    &compute,
                    &query,
                    1,
                    &memory,
                    2,
                    &[false, false],
                    None,
                    weights,
                )
                .is_err()
        );

        let mismatched = FireRedDecoderCrossAttention {
            d_model: 2,
            source_dim: 1,
            n_head: 1,
        };
        let narrow_k = [1.0, 0.0];
        let narrow_v = [0.0, 1.0];
        let narrow = FireRedDecoderCrossAttentionWeights {
            q_w_t: &identity,
            q_b: &zero,
            k_w_t: &narrow_k,
            v_w_t: &narrow_v,
            v_b: &zero,
            output_w_t: &identity,
            output_b: &zero,
        };
        assert!(
            mismatched
                .forward(&compute, &query, 1, &[1.0], 1, &[true], None, narrow,)
                .is_err()
        );

        let embedding = FireRedDecoderEmbedding {
            vocab_size: 2,
            d_model: 2,
            max_positions: 2,
        };
        let values = embedding
            .forward(&[1], &[0], &[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2, 0.3, 0.4])
            .unwrap();
        assert!((values[0] - (3.0 * 2.0_f32.sqrt() + 0.1)).abs() < 1e-6);
        assert!((values[1] - (4.0 * 2.0_f32.sqrt() + 0.2)).abs() < 1e-6);
        assert!(
            embedding
                .forward(&[2], &[0], &[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2, 0.3, 0.4],)
                .is_err()
        );
    }

    #[test]
    fn decoder_layer_uses_prenorm_gelu_and_requires_valid_queries() {
        let compute = Compute::cpu();
        let layer = FireRedDecoderLayer {
            d_model: 2,
            inner_dim: 4,
            n_head: 1,
            source_dim: 2,
        };
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0, 0.0];
        let ones = [1.0, 1.0];
        let expand = [0.2, 0.1, 0.3, -0.2, 0.4, 0.2, -0.1, 0.5];
        let expand_b = [0.1, 0.0, -0.1, 0.2];
        let project = [0.1, 0.2, -0.3, 0.4, 0.2, -0.1, 0.3, 0.1];
        let self_attention = FireRedDecoderSelfAttentionWeights {
            q_w_t: &identity,
            q_b: &zero,
            k_w_t: &identity,
            v_w_t: &identity,
            v_b: &zero,
            output_w_t: &identity,
            output_b: &zero,
        };
        let cross_attention = FireRedDecoderCrossAttentionWeights {
            q_w_t: &identity,
            q_b: &zero,
            k_w_t: &identity,
            v_w_t: &identity,
            v_b: &zero,
            output_w_t: &identity,
            output_b: &zero,
        };
        let weights = FireRedDecoderLayerWeights {
            self_norm_gamma: &ones,
            self_norm_beta: &zero,
            self_attention,
            cross_norm_gamma: &ones,
            cross_norm_beta: &zero,
            cross_attention,
            mlp_norm_gamma: &ones,
            mlp_norm_beta: &zero,
            mlp_expand_w_t: &expand,
            mlp_expand_b: &expand_b,
            mlp_project_w_t: &project,
            mlp_project_b: &zero,
        };
        let output = layer
            .forward(
                &compute,
                &[1.0, 2.0, 3.0, 4.0],
                2,
                &[true, true],
                &[0.5, -0.5, 2.0, 1.0],
                2,
                &[true, true],
                &[],
                &[],
                &[],
                weights,
            )
            .unwrap();
        assert!(output.output.iter().all(|value| value.is_finite()));
        // Upstream target masks are key/padding masks; they do not zero query
        // rows. This native layer therefore rejects padded query rows rather
        // than adding a non-source query-zeroing convention.
        assert!(
            layer
                .forward(
                    &compute,
                    &[1.0, 2.0, 3.0, 4.0],
                    2,
                    &[true, false],
                    &[0.5, -0.5, 2.0, 1.0],
                    2,
                    &[true, true],
                    &[],
                    &[],
                    &[],
                    weights,
                )
                .is_err()
        );
    }

    #[test]
    fn decoder_output_head_and_beam_rules_are_explicit() {
        let head = FireRedDecoderOutputHead {
            d_model: 2,
            vocab_size: 3,
        };
        let logits = head
            .forward(
                &Compute::cpu(),
                &[1.0, 3.0],
                1,
                &[1.0, 1.0],
                &[0.0, 0.0],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|value| value.is_finite()));

        let beam = FireRedBeamState::new();
        let eos = beam.advance(7, -0.5, 7, 4).unwrap();
        assert!(eos.finished);
        assert_eq!(eos.tokens, vec![7]);
        assert_eq!(eos.advance(8, -0.5, 7, 4).unwrap(), eos);
        let maxed = beam
            .advance(1, -0.1, 7, 1)
            .unwrap()
            .advance(2, -0.1, 7, 1)
            .unwrap();
        assert_eq!(maxed.tokens, vec![1]);
        assert!(maxed.finished);
        assert!(beam.advance(1, 0.1, 7, 4).is_err());
        assert!(eos.ranked_score(7, 0.5).unwrap().is_finite());
        let mut scores = [-2.0, -1.0, -3.0];
        apply_fire_red_eos_penalty(&mut scores, 1, 1.0).unwrap();
        assert_eq!(scores, [-2.0, -1.0, -3.0]);
        apply_fire_red_eos_penalty(&mut scores, 1, 0.5).unwrap();
        assert_eq!(scores[1], -0.5);
        assert!(apply_fire_red_eos_penalty(&mut scores, 1, 0.0).is_err());
        assert!(apply_fire_red_eos_penalty(&mut scores, 1, 1.1).is_err());
        let ranked = FireRedBeamState {
            tokens: vec![3, 7],
            score: -6.0,
            finished: true,
        }
        .ranked_score(7, 1.0)
        .unwrap();
        assert!((ranked - (-6.0 / (7.0 / 6.0))).abs() < 1e-6);
    }

    #[test]
    fn decoder_hot_ops_are_preflighted_without_cpu_fallback() {
        for required in [HotOp::Gemm, HotOp::LayerNorm, HotOp::Softmax, HotOp::Gelu] {
            assert!(
                FIRERED_ASR_AED_HOT_OPS.contains(&required),
                "decoder op {required:?} must be in whole-model preflight inventory"
            );
        }
        // Vulkan is intentionally not a covered Compute seam for this set in
        // the current repository. Either an unavailable build/device or an
        // explicit coverage error is acceptable; Ok would mean an accidental
        // backend path, while CPU fallback would violate FR-EX-08.
        let result = Compute::for_backend(
            vokra_core::BackendKind::Vulkan,
            &[HotOp::Gemm, HotOp::LayerNorm, HotOp::Softmax, HotOp::Gelu],
        );
        assert!(
            matches!(
                result,
                Err(VokraError::UnsupportedOp(_)) | Err(VokraError::BackendUnavailable(_))
            ),
            "unsupported/unavailable backend must refuse rather than fall back"
        );
    }
}
