use std::f32::consts::E;

use vokra_core::{BackendKind, Result, VokraError};

use crate::compute::{Compute, HotOp};

use super::MoonshineConfig;
use super::weights::{Attention, DecoderLayer, EncoderLayer, Linear, MoonshineWeights};

pub(super) const HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
];

pub(super) fn generate(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    backend: BackendKind,
    pcm: &[f32],
) -> Result<Vec<u32>> {
    generate_with_limit(weights, config, backend, pcm, config.max_positions)
}

pub(super) fn generate_with_limit(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    backend: BackendKind,
    pcm: &[f32],
    max_positions: usize,
) -> Result<Vec<u32>> {
    let compute = Compute::for_backend(backend, HOT_OPS)?;
    let (encoder, encoder_len) = encode(weights, config, &compute, pcm)?;
    let mut ids = vec![config.decoder_start_token_id];
    while ids.len() < max_positions {
        let hidden = decode(weights, config, &compute, &ids, &encoder, encoder_len)?;
        let last = &hidden[(ids.len() - 1) * config.hidden_size..ids.len() * config.hidden_size];
        let mut logits = vec![0.0; config.vocab_size];
        compute.gemv_f32(
            config.vocab_size,
            config.hidden_size,
            &weights.embedding,
            last,
            None,
            &mut logits,
        )?;
        let next = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| VokraError::ModelLoad("moonshine: empty logits".into()))?;
        if next == config.eos_token_id {
            break;
        }
        ids.push(next);
    }
    Ok(ids.into_iter().skip(1).collect())
}

pub(super) fn encode(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    compute: &Compute,
    pcm: &[f32],
) -> Result<(Vec<f32>, usize)> {
    let d = config.hidden_size;
    let n1 = conv_out_len(pcm.len(), 127, 64)?;
    let mut conv1 = vec![0.0; d * n1];
    compute.conv1d_f32(
        pcm,
        1,
        pcm.len(),
        &weights.conv1.w,
        d,
        127,
        None,
        64,
        0,
        &mut conv1,
    )?;
    conv1.iter_mut().for_each(|value| *value = value.tanh());
    group_norm(
        &mut conv1,
        d,
        n1,
        &weights.groupnorm_weight,
        &weights.groupnorm_bias,
    );

    let n2 = conv_out_len(n1, 7, 3)?;
    let mut conv2 = vec![0.0; 2 * d * n2];
    compute.conv1d_f32(
        &conv1,
        d,
        n1,
        &weights.conv2.w,
        2 * d,
        7,
        weights.conv2.b.as_deref(),
        3,
        0,
        &mut conv2,
    )?;
    let mut activated = vec![0.0; conv2.len()];
    compute.gelu_f32(&conv2, &mut activated)?;

    let n3 = conv_out_len(n2, 3, 2)?;
    let mut conv3 = vec![0.0; d * n3];
    compute.conv1d_f32(
        &activated,
        2 * d,
        n2,
        &weights.conv3.w,
        d,
        3,
        weights.conv3.b.as_deref(),
        2,
        0,
        &mut conv3,
    )?;
    let mut conv3_activated = vec![0.0; conv3.len()];
    compute.gelu_f32(&conv3, &mut conv3_activated)?;

    if n3 > config.max_positions {
        return Err(VokraError::InvalidArgument(format!(
            "moonshine: audio produces {n3} encoder positions, maximum is {}",
            config.max_positions
        )));
    }
    let mut hidden = vec![0.0; n3 * d];
    for channel in 0..d {
        for time in 0..n3 {
            hidden[time * d + channel] = conv3_activated[channel * n3 + time];
        }
    }
    for layer in &weights.encoder_layers {
        encoder_layer(compute, config, layer, &mut hidden, n3)?;
    }
    hidden = layer_norm(compute, &hidden, n3, d, &weights.encoder_norm)?;
    Ok((hidden, n3))
}

pub(super) fn decode(
    weights: &MoonshineWeights,
    config: &MoonshineConfig,
    compute: &Compute,
    ids: &[u32],
    encoder: &[f32],
    encoder_len: usize,
) -> Result<Vec<f32>> {
    let d = config.hidden_size;
    let mut hidden = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        let id = id as usize;
        if id >= config.vocab_size {
            return Err(VokraError::ModelLoad(format!(
                "moonshine: generated token {id} exceeds vocabulary"
            )));
        }
        hidden.extend_from_slice(&weights.embedding[id * d..(id + 1) * d]);
    }
    for layer in &weights.decoder_layers {
        decoder_layer(
            compute,
            config,
            layer,
            &mut hidden,
            ids.len(),
            encoder,
            encoder_len,
        )?;
    }
    layer_norm(compute, &hidden, ids.len(), d, &weights.decoder_norm)
}

fn encoder_layer(
    compute: &Compute,
    config: &MoonshineConfig,
    layer: &EncoderLayer,
    hidden: &mut [f32],
    rows: usize,
) -> Result<()> {
    let d = config.hidden_size;
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln1)?;
    let attended = attention(
        compute,
        config,
        &layer.attn,
        &normalized,
        rows,
        &normalized,
        rows,
        false,
        true,
    )?;
    residual_add(hidden, &attended);
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln2)?;
    let projected = linear_rows(
        compute,
        &layer.fc1,
        &normalized,
        rows,
        d,
        config.intermediate_size,
    )?;
    let mut activated = vec![0.0; projected.len()];
    compute.gelu_f32(&projected, &mut activated)?;
    let projected = linear_rows(
        compute,
        &layer.fc2,
        &activated,
        rows,
        config.intermediate_size,
        d,
    )?;
    residual_add(hidden, &projected);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decoder_layer(
    compute: &Compute,
    config: &MoonshineConfig,
    layer: &DecoderLayer,
    hidden: &mut [f32],
    rows: usize,
    encoder: &[f32],
    encoder_rows: usize,
) -> Result<()> {
    let d = config.hidden_size;
    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln1)?;
    let attended = attention(
        compute,
        config,
        &layer.self_attn,
        &normalized,
        rows,
        &normalized,
        rows,
        true,
        true,
    )?;
    residual_add(hidden, &attended);

    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln2)?;
    let attended = attention(
        compute,
        config,
        &layer.cross_attn,
        &normalized,
        rows,
        encoder,
        encoder_rows,
        false,
        false,
    )?;
    residual_add(hidden, &attended);

    let normalized = layer_norm(compute, hidden, rows, d, &layer.ln3)?;
    let gated = linear_rows(
        compute,
        &layer.fc1,
        &normalized,
        rows,
        d,
        2 * config.intermediate_size,
    )?;
    let ff = config.intermediate_size;
    let mut activated = vec![0.0; rows * ff];
    for row in 0..rows {
        let src = &gated[row * 2 * ff..(row + 1) * 2 * ff];
        let dst = &mut activated[row * ff..(row + 1) * ff];
        for index in 0..ff {
            let gate = src[ff + index];
            dst[index] = src[index] * gate / (1.0 + E.powf(-gate));
        }
    }
    let projected = linear_rows(compute, &layer.fc2, &activated, rows, ff, d)?;
    residual_add(hidden, &projected);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    config: &MoonshineConfig,
    weights: &Attention,
    queries: &[f32],
    query_rows: usize,
    keys_values: &[f32],
    key_rows: usize,
    causal: bool,
    rotary: bool,
) -> Result<Vec<f32>> {
    let d = config.hidden_size;
    let heads = config.attention_heads;
    let head_dim = d / heads;
    let mut q = linear_rows(compute, &weights.q, queries, query_rows, d, d)?;
    let mut k = linear_rows(compute, &weights.k, keys_values, key_rows, d, d)?;
    let v = linear_rows(compute, &weights.v, keys_values, key_rows, d, d)?;
    if rotary {
        apply_rope(
            &mut q,
            query_rows,
            heads,
            head_dim,
            config.rotary_dim,
            config.rope_theta,
        );
        apply_rope(
            &mut k,
            key_rows,
            heads,
            head_dim,
            config.rotary_dim,
            config.rope_theta,
        );
    }
    let mut context = vec![0.0; query_rows * d];
    let scale = (head_dim as f32).sqrt().recip();
    if causal && query_rows != key_rows {
        return Err(VokraError::InvalidArgument(format!(
            "moonshine: causal attention requires equal query/key rows, got {query_rows}/{key_rows}"
        )));
    }
    for head in 0..heads {
        // The model stores Q/K/V as [time, head, dim]. Pack one head into
        // row-major matrices so both Q*K^T and attention*V go through the
        // backend GEMM. The old scalar dot/value loops made a non-CPU
        // selection a partial host execution even though every projection and
        // softmax was already backend-dispatched.
        let mut q_head = vec![0.0; query_rows * head_dim];
        let mut k_head_t = vec![0.0; head_dim * key_rows];
        let mut v_head = vec![0.0; key_rows * head_dim];
        for query in 0..query_rows {
            for dim in 0..head_dim {
                q_head[query * head_dim + dim] = q[query * d + head * head_dim + dim] * scale;
            }
        }
        for key in 0..key_rows {
            for dim in 0..head_dim {
                k_head_t[dim * key_rows + key] = k[key * d + head * head_dim + dim];
                v_head[key * head_dim + dim] = v[key * d + head * head_dim + dim];
            }
        }

        let mut scores = vec![0.0; query_rows * key_rows];
        compute.gemm_f32(
            query_rows,
            key_rows,
            head_dim,
            &q_head,
            &k_head_t,
            None,
            &mut scores,
        )?;
        if causal {
            for query in 0..query_rows {
                scores[query * key_rows + query + 1..(query + 1) * key_rows]
                    .fill(f32::NEG_INFINITY);
            }
        }

        let mut probabilities = vec![0.0; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, query_rows, key_rows)?;
        let mut head_context = vec![0.0; query_rows * head_dim];
        compute.gemm_f32(
            query_rows,
            head_dim,
            key_rows,
            &probabilities,
            &v_head,
            None,
            &mut head_context,
        )?;
        for query in 0..query_rows {
            for dim in 0..head_dim {
                context[query * d + head * head_dim + dim] = head_context[query * head_dim + dim];
            }
        }
    }
    linear_rows(compute, &weights.o, &context, query_rows, d, d)
}

fn linear_rows(
    compute: &Compute,
    linear: &Linear,
    input: &[f32],
    rows: usize,
    input_size: usize,
    output_size: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; rows * output_size];
    for row in 0..rows {
        compute.gemv_f32(
            output_size,
            input_size,
            &linear.w,
            &input[row * input_size..(row + 1) * input_size],
            linear.b.as_deref(),
            &mut output[row * output_size..(row + 1) * output_size],
        )?;
    }
    Ok(output)
}

fn layer_norm(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
) -> Result<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    let bias = vec![0.0; cols];
    compute.layer_norm_f32(input, &mut output, rows, cols, weight, &bias, 1e-5)?;
    Ok(output)
}

fn group_norm(values: &mut [f32], channels: usize, time: usize, weight: &[f32], bias: &[f32]) {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    let inverse_std = (variance + 1e-5).sqrt().recip();
    for channel in 0..channels {
        for index in 0..time {
            let slot = channel * time + index;
            values[slot] = (values[slot] - mean) * inverse_std * weight[channel] + bias[channel];
        }
    }
}

fn apply_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) {
    let d = heads * head_dim;
    for position in 0..rows {
        for head in 0..heads {
            let base = position * d + head * head_dim;
            for pair in (0..rotary_dim).step_by(2) {
                let frequency = theta.powf(-(pair as f32) / rotary_dim as f32);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let left = values[base + pair];
                let right = values[base + pair + 1];
                values[base + pair] = left * cos - right * sin;
                values[base + pair + 1] = right * cos + left * sin;
            }
        }
    }
}

fn conv_out_len(input: usize, kernel: usize, stride: usize) -> Result<usize> {
    if input < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "moonshine: input length {input} is shorter than Conv1D kernel {kernel}"
        )));
    }
    Ok((input - kernel) / stride + 1)
}

fn residual_add(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_linear(width: usize) -> Linear {
        let mut w = vec![0.0; width * width];
        for index in 0..width {
            w[index * width + index] = 1.0;
        }
        Linear { w, b: None }
    }

    fn attention_fixture() -> (MoonshineConfig, Attention) {
        let config = MoonshineConfig {
            variant: super::super::MoonshineVariant::Tiny,
            hidden_size: 2,
            intermediate_size: 4,
            encoder_layers: 1,
            decoder_layers: 1,
            attention_heads: 1,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            max_positions: 4,
            vocab_size: 4,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            sample_rate: 16_000,
        };
        let linear = identity_linear(2);
        (
            config,
            Attention {
                q: linear.clone(),
                k: linear.clone(),
                v: linear.clone(),
                o: linear,
            },
        )
    }

    #[test]
    fn convolution_lengths_match_reference_formula() {
        assert_eq!(conv_out_len(16_000, 127, 64).unwrap(), 249);
        assert_eq!(conv_out_len(249, 7, 3).unwrap(), 81);
        assert_eq!(conv_out_len(81, 3, 2).unwrap(), 40);
    }

    #[test]
    fn rope_rotates_first_pair_only() {
        let mut values = vec![1.0, 0.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 1, 4, 2, 10_000.0);
        assert_eq!(values, vec![1.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn composed_attention_routes_qk_and_value_fold_through_gemm() {
        let (config, weights) = attention_fixture();
        let input = [1.0, 0.0, 0.0, 1.0];
        let compute = Compute::cpu();
        let actual = attention(
            &compute, &config, &weights, &input, 2, &input, 2, false, false,
        )
        .unwrap();

        // Independent closed-form result for softmax([1/sqrt(2), 0]).
        let high = (1.0_f32 / 2.0_f32.sqrt()).exp() / ((1.0_f32 / 2.0_f32.sqrt()).exp() + 1.0);
        let low = 1.0 - high;
        let expected = [high, low, low, high];
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-6),
            "actual={actual:?} expected={expected:?}"
        );
    }

    #[test]
    fn composed_attention_applies_the_causal_mask_before_softmax() {
        let (config, weights) = attention_fixture();
        let input = [1.0, 0.0, 0.0, 1.0];
        let actual = attention(
            &Compute::cpu(),
            &config,
            &weights,
            &input,
            2,
            &input,
            2,
            true,
            false,
        )
        .unwrap();
        let high = (1.0_f32 / 2.0_f32.sqrt()).exp() / ((1.0_f32 / 2.0_f32.sqrt()).exp() + 1.0);
        let low = 1.0 - high;
        let expected = [1.0, 0.0, low, high];
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-6),
            "actual={actual:?} expected={expected:?}"
        );
    }

    /// Real-weight CPU/Metal parity for both public Moonshine variants.
    ///
    /// The CPU path is already checked against the independent pinned
    /// Transformers fixture in `moonshine::tests`; this test treats that path
    /// as the oracle and verifies the newly composed Metal attention path at
    /// the encoder, decoder, tied-logit, and generated-token boundaries. It
    /// skips only when neither public artifact is supplied:
    ///
    /// ```text
    /// VOKRA_MOONSHINE_TINY_GGUF=moonshine-tiny.gguf \
    /// VOKRA_MOONSHINE_BASE_GGUF=moonshine-base.gguf \
    ///   cargo test -p vokra-models --features metal \
    ///     moonshine_real_weights_metal_match_cpu -- --nocapture
    /// ```
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn moonshine_real_weights_metal_match_cpu() {
        const ATOL: f32 = 0.01;

        fn max_abs(left: &[f32], right: &[f32]) -> f32 {
            assert_eq!(left.len(), right.len());
            left.iter()
                .zip(right)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f32::max)
        }

        let paths = [
            ("tiny", std::env::var("VOKRA_MOONSHINE_TINY_GGUF")),
            ("base", std::env::var("VOKRA_MOONSHINE_BASE_GGUF")),
        ];
        let mut ran = 0usize;
        for (variant, path) in paths {
            let Ok(path) = path else { continue };
            ran += 1;
            let file = vokra_core::gguf::GgufFile::open(path).expect("open Moonshine GGUF");
            let model = super::super::Moonshine::from_gguf(&file).expect("bind Moonshine GGUF");
            let pcm = (0..16_000)
                .map(|index| {
                    let x = index as f32;
                    0.08 * (x * 0.013).sin() + 0.03 * (x * 0.0037).cos()
                })
                .collect::<Vec<_>>();
            let cpu = Compute::for_backend(BackendKind::Cpu, HOT_OPS).expect("CPU compute");
            let metal = match Compute::for_backend(BackendKind::Metal, HOT_OPS) {
                Ok(compute) => compute,
                Err(VokraError::BackendUnavailable(error)) => {
                    eprintln!("skip Moonshine Metal parity: {error}");
                    return;
                }
                Err(error) => panic!("Moonshine Metal hot-op coverage is incomplete: {error}"),
            };

            let (cpu_encoder, cpu_rows) =
                encode(&model.weights, &model.config, &cpu, &pcm).expect("CPU encoder");
            let (metal_encoder, metal_rows) =
                encode(&model.weights, &model.config, &metal, &pcm).expect("Metal encoder");
            assert_eq!(cpu_rows, metal_rows);
            let encoder_error = max_abs(&cpu_encoder, &metal_encoder);
            assert!(
                encoder_error <= ATOL,
                "{variant} encoder CPU/Metal max_abs={encoder_error:e} > {ATOL}"
            );

            let ids = [model.config.decoder_start_token_id, 1_939, 29_889];
            let cpu_decoder = decode(
                &model.weights,
                &model.config,
                &cpu,
                &ids,
                &cpu_encoder,
                cpu_rows,
            )
            .expect("CPU decoder");
            let metal_decoder = decode(
                &model.weights,
                &model.config,
                &metal,
                &ids,
                &metal_encoder,
                metal_rows,
            )
            .expect("Metal decoder");
            let decoder_error = max_abs(&cpu_decoder, &metal_decoder);
            assert!(
                decoder_error <= ATOL,
                "{variant} decoder CPU/Metal max_abs={decoder_error:e} > {ATOL}"
            );

            let d = model.config.hidden_size;
            let mut cpu_logits = vec![0.0; model.config.vocab_size];
            let mut metal_logits = vec![0.0; model.config.vocab_size];
            cpu.gemv_f32(
                model.config.vocab_size,
                d,
                &model.weights.embedding,
                &cpu_decoder[cpu_decoder.len() - d..],
                None,
                &mut cpu_logits,
            )
            .expect("CPU logits");
            metal
                .gemv_f32(
                    model.config.vocab_size,
                    d,
                    &model.weights.embedding,
                    &metal_decoder[metal_decoder.len() - d..],
                    None,
                    &mut metal_logits,
                )
                .expect("Metal logits");
            let logit_error = max_abs(&cpu_logits, &metal_logits);
            assert!(
                logit_error <= ATOL,
                "{variant} logits CPU/Metal max_abs={logit_error:e} > {ATOL}"
            );

            let cpu_ids =
                generate_with_limit(&model.weights, &model.config, BackendKind::Cpu, &pcm, 5)
                    .expect("CPU generation");
            let metal_ids =
                generate_with_limit(&model.weights, &model.config, BackendKind::Metal, &pcm, 5)
                    .expect("Metal generation");
            assert_eq!(cpu_ids, metal_ids, "{variant} generated token mismatch");
            eprintln!(
                "Moonshine {variant} CPU/Metal: encoder={encoder_error:e} decoder={decoder_error:e} logits={logit_error:e}"
            );
        }
        if ran == 0 {
            eprintln!(
                "skip Moonshine Metal real-weight parity: set VOKRA_MOONSHINE_TINY_GGUF and/or VOKRA_MOONSHINE_BASE_GGUF"
            );
        }
    }
}
