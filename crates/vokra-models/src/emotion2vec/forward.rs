use vokra_core::{Result, VokraError};

use crate::align::charsiu::{
    CharsiuBlock, CharsiuConfig, CharsiuPosConv, feature_projection_forward_with_compute,
    layer_norm_with_compute_inplace, linear_forward_with_compute,
};
use crate::compute::Compute;
use crate::wav2vec2_ctc::{
    reject_non_finite, transpose_channel_to_frame, transpose_frame_to_channel,
    waveform_frontend_with_compute,
};

use super::bound::Emotion2VecWeights;
use super::{
    EXTRA_TOKENS, FEATURE_DIM, FFN, HEADS, HIDDEN, LAYER_NORM_EPS, NUM_CLASSES, POSITION_GROUPS,
    POSITION_KERNEL, SAMPLE_RATE,
};

pub(super) struct ForwardResult {
    pub(super) final_features: Vec<f32>,
    pub(super) frames: usize,
    pub(super) logits: Vec<f32>,
    pub(super) scores: Vec<f32>,
}

#[derive(Debug, Default)]
pub(super) struct ForwardTaps {
    pub(super) normalized_pcm: Vec<f32>,
    pub(super) conv_features: Vec<f32>,
    pub(super) projected_features: Vec<f32>,
    pub(super) context_features: Vec<f32>,
    pub(super) final_features: Vec<f32>,
    pub(super) pooled_embedding: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) scores: Vec<f32>,
}

pub(super) fn run_forward(
    weights: &Emotion2VecWeights,
    pcm: &[f32],
    compute: &Compute,
    mut taps: Option<&mut ForwardTaps>,
    with_head: bool,
) -> Result<ForwardResult> {
    let normalized = normalize_waveform(pcm);
    if let Some(taps) = taps.as_deref_mut() {
        taps.normalized_pcm.clone_from(&normalized);
    }

    let conv_features =
        waveform_frontend_with_compute(&normalized, &weights.stem_attrs, &weights.stem, compute)?;
    let frames = conv_features.len() / FEATURE_DIM;
    if frames == 0 {
        return Err(VokraError::InvalidArgument(
            "emotion2vec: waveform stem produced zero frames".to_owned(),
        ));
    }
    if let Some(taps) = taps.as_deref_mut() {
        // The official hook observes ConvFeatureExtractionModel before
        // project_features' TransposeLast, i.e. [B, C, T]. The native stem
        // returns frame-major [T, C], so store this tap in official layout.
        taps.conv_features = transpose_frame_to_channel(&conv_features, frames, FEATURE_DIM);
    }

    let mut projected = feature_projection_forward_with_compute(
        &conv_features,
        frames,
        FEATURE_DIM,
        &weights.projection,
        HIDDEN,
        true,
        LAYER_NORM_EPS,
        compute,
    )?;
    if let Some(taps) = taps.as_deref_mut() {
        taps.projected_features.clone_from(&projected);
    }

    let position = positional_stack(&projected, frames, &weights.position, compute)?;
    for (value, positional) in projected.iter_mut().zip(position) {
        *value += positional;
    }

    let total_frames = frames + EXTRA_TOKENS;
    let mut hidden = Vec::with_capacity(total_frames * HIDDEN);
    hidden.extend_from_slice(&weights.extra_tokens);
    hidden.extend_from_slice(&projected);

    layer_norm_with_compute_inplace(
        &mut hidden,
        total_frames,
        HIDDEN,
        &weights.context_norm_gamma,
        &weights.context_norm_beta,
        LAYER_NORM_EPS,
        compute,
    )?;
    let config = encoder_config();
    let slopes = alibi_slopes(HEADS);
    for block in &weights.context_blocks {
        emotion_block_forward(
            &mut hidden,
            total_frames,
            &config,
            block,
            &slopes,
            &weights.alibi_scale,
            compute,
        )?;
    }
    if let Some(taps) = taps.as_deref_mut() {
        taps.context_features.clone_from(&hidden);
    }

    for block in &weights.global_blocks {
        emotion_block_forward(
            &mut hidden,
            total_frames,
            &config,
            block,
            &slopes,
            &weights.alibi_scale,
            compute,
        )?;
    }

    let final_features = hidden[EXTRA_TOKENS * HIDDEN..].to_vec();
    reject_non_finite("emotion2vec final features", &final_features)?;
    if let Some(taps) = taps.as_deref_mut() {
        taps.final_features.clone_from(&final_features);
    }
    if !with_head {
        return Ok(ForwardResult {
            final_features,
            frames,
            logits: Vec::new(),
            scores: Vec::new(),
        });
    }

    let pooled = mean_frames(&final_features, frames);
    if let Some(taps) = taps.as_deref_mut() {
        taps.pooled_embedding.clone_from(&pooled);
    }
    let logits = linear_forward_with_compute(
        &pooled,
        1,
        HIDDEN,
        &weights.head_weight,
        &weights.head_bias,
        NUM_CLASSES,
        compute,
    )?;
    let mut scores = vec![0.0f32; NUM_CLASSES];
    compute.softmax_f32(&logits, &mut scores, 1, NUM_CLASSES)?;
    reject_non_finite("emotion2vec logits", &logits)?;
    reject_non_finite("emotion2vec scores", &scores)?;
    if let Some(taps) = taps {
        taps.logits.clone_from(&logits);
        taps.scores.clone_from(&scores);
    }
    Ok(ForwardResult {
        final_features,
        frames,
        logits,
        scores,
    })
}

fn normalize_waveform(pcm: &[f32]) -> Vec<f32> {
    let mean = pcm.iter().copied().sum::<f32>() / pcm.len() as f32;
    let variance = pcm
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / pcm.len() as f32;
    let inverse = (variance + LAYER_NORM_EPS).sqrt().recip();
    pcm.iter().map(|value| (*value - mean) * inverse).collect()
}

fn positional_stack(
    hidden: &[f32],
    frames: usize,
    layers: &[CharsiuPosConv],
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut channel_major = transpose_frame_to_channel(hidden, frames, HIDDEN);
    let gamma = vec![1.0f32; HIDDEN];
    let beta = vec![0.0f32; HIDDEN];
    for layer in layers {
        let mut convolution = vec![0.0f32; HIDDEN * frames];
        compute.grouped_conv1d_f32(
            &channel_major,
            HIDDEN,
            frames,
            &layer.weight,
            HIDDEN,
            POSITION_KERNEL,
            Some(&layer.bias),
            1,
            POSITION_KERNEL / 2,
            POSITION_GROUPS,
            &mut convolution,
        )?;
        let mut frame_major = transpose_channel_to_frame(&convolution, HIDDEN, frames);
        layer_norm_with_compute_inplace(
            &mut frame_major,
            frames,
            HIDDEN,
            &gamma,
            &beta,
            LAYER_NORM_EPS,
            compute,
        )?;
        channel_major = transpose_frame_to_channel(&frame_major, frames, HIDDEN);
        let mut activated = vec![0.0f32; channel_major.len()];
        compute.gelu_f32(&channel_major, &mut activated)?;
        channel_major = activated;
    }
    Ok(transpose_channel_to_frame(&channel_major, HIDDEN, frames))
}

#[allow(clippy::too_many_arguments)]
fn emotion_block_forward(
    hidden: &mut [f32],
    frames: usize,
    config: &CharsiuConfig,
    block: &CharsiuBlock,
    slopes: &[f32],
    learned_scale: &[f32],
    compute: &Compute,
) -> Result<()> {
    let hidden_size = config.hidden_size;
    let head_dim = hidden_size / config.n_head;
    let attention_scale = (head_dim as f32).sqrt().recip();
    let q = linear_forward_with_compute(
        hidden,
        frames,
        hidden_size,
        &block.q_w,
        &block.q_b,
        hidden_size,
        compute,
    )?;
    let k = linear_forward_with_compute(
        hidden,
        frames,
        hidden_size,
        &block.k_w,
        &block.k_b,
        hidden_size,
        compute,
    )?;
    let v = linear_forward_with_compute(
        hidden,
        frames,
        hidden_size,
        &block.v_w,
        &block.v_b,
        hidden_size,
        compute,
    )?;

    let mut attention = vec![0.0f32; frames * hidden_size];
    let mut q_head = vec![0.0f32; frames * head_dim];
    let mut k_head_t = vec![0.0f32; head_dim * frames];
    let mut v_head = vec![0.0f32; frames * head_dim];
    let mut scores = vec![0.0f32; frames * frames];
    let mut probabilities = vec![0.0f32; frames * frames];
    let mut head_output = vec![0.0f32; frames * head_dim];
    for head in 0..config.n_head {
        for frame in 0..frames {
            let source = frame * hidden_size + head * head_dim;
            let destination = frame * head_dim;
            q_head[destination..destination + head_dim]
                .copy_from_slice(&q[source..source + head_dim]);
            v_head[destination..destination + head_dim]
                .copy_from_slice(&v[source..source + head_dim]);
            for dim in 0..head_dim {
                k_head_t[dim * frames + frame] = k[source + dim];
            }
        }
        compute.gemm_f32(
            frames,
            frames,
            head_dim,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        for query in 0..frames {
            for key in 0..frames {
                let index = query * frames + key;
                scores[index] = scores[index] * attention_scale
                    + alibi_bias(head, query, key, slopes, learned_scale);
            }
        }
        compute.softmax_f32(&scores, &mut probabilities, frames, frames)?;
        compute.gemm_f32(
            frames,
            head_dim,
            frames,
            &probabilities,
            &v_head,
            None,
            &mut head_output,
        )?;
        for frame in 0..frames {
            let source = frame * head_dim;
            let destination = frame * hidden_size + head * head_dim;
            attention[destination..destination + head_dim]
                .copy_from_slice(&head_output[source..source + head_dim]);
        }
    }

    let projected = linear_forward_with_compute(
        &attention,
        frames,
        hidden_size,
        &block.o_w,
        &block.o_b,
        hidden_size,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }
    layer_norm_with_compute_inplace(
        hidden,
        frames,
        hidden_size,
        &block.attn_norm_gamma,
        &block.attn_norm_beta,
        config.layer_norm_eps,
        compute,
    )?;

    let fc1 = linear_forward_with_compute(
        hidden,
        frames,
        hidden_size,
        &block.fc1_w,
        &block.fc1_b,
        config.ffn_dim,
        compute,
    )?;
    let mut activated = vec![0.0f32; fc1.len()];
    compute.gelu_f32(&fc1, &mut activated)?;
    let fc2 = linear_forward_with_compute(
        &activated,
        frames,
        config.ffn_dim,
        &block.fc2_w,
        &block.fc2_b,
        hidden_size,
        compute,
    )?;
    for (value, residual) in hidden.iter_mut().zip(fc2) {
        *value += residual;
    }
    layer_norm_with_compute_inplace(
        hidden,
        frames,
        hidden_size,
        &block.ffn_norm_gamma,
        &block.ffn_norm_beta,
        config.layer_norm_eps,
        compute,
    )
}

pub(super) fn alibi_slopes(heads: usize) -> Vec<f32> {
    assert!(heads.is_power_of_two());
    let start = 2.0_f32.powf(-2.0_f32.powf(3.0 - (heads as f32).log2()));
    (1..=heads).map(|power| start.powi(power as i32)).collect()
}

pub(super) fn alibi_bias(
    head: usize,
    query: usize,
    key: usize,
    slopes: &[f32],
    learned_scale: &[f32],
) -> f32 {
    if query < EXTRA_TOKENS || key < EXTRA_TOKENS {
        return 0.0;
    }
    let distance = query.abs_diff(key) as f32;
    -distance * slopes[head] * learned_scale[head].max(0.0)
}

fn mean_frames(values: &[f32], frames: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; HIDDEN];
    for frame in 0..frames {
        for hidden in 0..HIDDEN {
            output[hidden] += values[frame * HIDDEN + hidden];
        }
    }
    let inverse = 1.0f32 / frames as f32;
    for value in &mut output {
        *value *= inverse;
    }
    output
}

fn encoder_config() -> CharsiuConfig {
    CharsiuConfig {
        hidden_size: HIDDEN,
        n_layer: 0,
        n_head: HEADS,
        ffn_dim: FFN,
        vocab_size: NUM_CLASSES,
        silence_id: 0,
        pad_id: 0,
        sample_rate: SAMPLE_RATE,
        frame_shift_sec: 0.02,
        layer_norm_eps: LAYER_NORM_EPS,
        pos_conv_kernel: POSITION_KERNEL,
        pos_conv_groups: POSITION_GROUPS,
        silence_threshold: 0,
        feature_projection_has_layer_norm: true,
        stem_conv_bias: false,
    }
}
