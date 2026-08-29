//! VoxCPM-0.5B local encoder, local DiT, and UnifiedCFM sampler.
//!
//! The tensor-facing surfaces in this module are deliberately strict: source
//! linear matrices are bound with exact dimensions, and all learned kernels
//! route through the selected [`Compute`] backend.  This module does not own
//! the base/residual LM generation loop or AudioVAE decoding.

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::strict_checkpoint::load_tensor;

use super::{
    MiniCpm4BlockWeights, MiniCpm4Config, MiniCpm4Linear, MiniCpm4Stack, MiniCpm4StackWeights,
};

const LOCAL_HIDDEN: usize = 1024;
const LOCAL_FEATURE: usize = 64;
const LOCAL_LAYERS: usize = 4;
const SPECIAL_TOKEN_SHAPE: &[usize] = &[1, 1, 1, LOCAL_HIDDEN];
const LOCAL_BIAS_SHAPE: &[usize] = &[LOCAL_HIDDEN];
#[allow(dead_code)] // Used only by the dormant staged tensor-contract tests.
const LOCAL_TENSOR_RANK_CONTRACT: &[(&str, &[usize])] = &[
    (
        "feat_encoder.in_proj.weight",
        &[LOCAL_HIDDEN, LOCAL_FEATURE],
    ),
    ("feat_encoder.in_proj.bias", LOCAL_BIAS_SHAPE),
    ("feat_encoder.special_token", SPECIAL_TOKEN_SHAPE),
    (
        "feat_decoder.estimator.in_proj.weight",
        &[LOCAL_HIDDEN, LOCAL_FEATURE],
    ),
    (
        "feat_decoder.estimator.cond_proj.weight",
        &[LOCAL_HIDDEN, LOCAL_FEATURE],
    ),
    (
        "feat_decoder.estimator.out_proj.weight",
        &[LOCAL_FEATURE, LOCAL_HIDDEN],
    ),
];

/// Source-shaped local encoder: project flattened `[B*T,P,64]` groups (the
/// source input is `[B,T,P,64]`), prepend one learned special token to each
/// `(B,T)` group, run a noncausal four-layer stack, and return `[B*T,1024]`
/// rows for reshaping to `[B,T,1024]`.
#[derive(Debug, Clone)]
pub struct LocalEncoder {
    stack: MiniCpm4Stack,
    in_proj: MiniCpm4Linear,
    special_token: Vec<f32>,
}

impl LocalEncoder {
    /// Production binding remains closed until VAST records an immutable
    /// complete composite manifest and exact provenance contract.
    pub fn from_gguf(_file: &GgufFile) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "voxcpm2 local encoder: production GGUF binding is blocked until the exact complete composite name/shape manifest and source/tokenizer provenance are authenticated on VAST",
        ))
    }

    /// Crate-private staged loader; same-shaped input is not production
    /// authorization until the complete composite contract is landed.
    #[allow(dead_code)]
    pub(crate) fn from_staged_gguf(file: &GgufFile) -> Result<Self> {
        let stack = load_local_stack(file, "feat_encoder.encoder")?;
        Self::from_source(
            stack,
            load_tensor(
                file,
                "voxcpm2",
                "feat_encoder.in_proj.weight",
                &[LOCAL_HIDDEN, LOCAL_FEATURE],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_encoder.in_proj.bias",
                LOCAL_BIAS_SHAPE,
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_encoder.special_token",
                SPECIAL_TOKEN_SHAPE,
            )?,
        )
    }

    pub(crate) fn from_source(
        stack: MiniCpm4Stack,
        in_weight: Vec<f32>,
        in_bias: Vec<f32>,
        special_token: Vec<f32>,
    ) -> Result<Self> {
        validate_local_stack(&stack, "local encoder")?;
        let in_proj =
            MiniCpm4Linear::from_source(in_weight, Some(in_bias), LOCAL_HIDDEN, LOCAL_FEATURE)?;
        check_finite_len("local encoder special_token", &special_token, LOCAL_HIDDEN)?;
        Ok(Self {
            stack,
            in_proj,
            special_token,
        })
    }

    /// `input` is row-major `[groups=B*T, patches=P, 64]`; output is
    /// `[groups=B*T, 1024]`. Each group gets its own independent `P+1`
    /// attention sequence; B/T are not collapsed into one sequence.
    pub fn forward(
        &self,
        input: &[f32],
        groups: usize,
        patches: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if groups == 0 || patches == 0 {
            return Err(VokraError::InvalidArgument(
                "VoxCPM local encoder requires non-empty batch and frames".to_owned(),
            ));
        }
        let expected = groups
            .checked_mul(patches)
            .and_then(|n| n.checked_mul(LOCAL_FEATURE))
            .ok_or_else(|| {
                VokraError::InvalidArgument("local encoder shape overflow".to_owned())
            })?;
        if input.len() != expected || input.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "local encoder input must be finite [groups=B*T, patches=P, 64]".to_owned(),
            ));
        }
        let mut projected = vec![0.0; groups * patches * LOCAL_HIDDEN];
        self.in_proj
            .apply(compute, input, groups * patches, &mut projected)?;
        let mut output = vec![0.0; groups * LOCAL_HIDDEN];
        for group in 0..groups {
            let mut sequence = Vec::with_capacity((patches + 1) * LOCAL_HIDDEN);
            sequence.extend_from_slice(&self.special_token);
            sequence.extend_from_slice(
                &projected[group * patches * LOCAL_HIDDEN..(group + 1) * patches * LOCAL_HIDDEN],
            );
            let encoded = self.stack.forward(&sequence, patches + 1, false, compute)?;
            output[group * LOCAL_HIDDEN..(group + 1) * LOCAL_HIDDEN]
                .copy_from_slice(&encoded[..LOCAL_HIDDEN]);
        }
        Ok(output)
    }

    #[must_use]
    /// Access the validated MiniCPM-4 encoder stack.
    pub fn stack(&self) -> &MiniCpm4Stack {
        &self.stack
    }
}

/// Learned local DiT projections. The stack is the same generic MiniCPM-4
/// implementation as the local encoder, but the sequence is noncausal.
#[derive(Debug, Clone)]
pub struct LocalDitWeights {
    in_proj: MiniCpm4Linear,
    cond_proj: MiniCpm4Linear,
    time_linear_1: MiniCpm4Linear,
    time_linear_2: MiniCpm4Linear,
    delta_time_linear_1: MiniCpm4Linear,
    delta_time_linear_2: MiniCpm4Linear,
    out_proj: MiniCpm4Linear,
}

impl LocalDitWeights {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_source(
        in_weight: Vec<f32>,
        in_bias: Vec<f32>,
        cond_weight: Vec<f32>,
        cond_bias: Vec<f32>,
        time_linear_1_weight: Vec<f32>,
        time_linear_1_bias: Vec<f32>,
        time_linear_2_weight: Vec<f32>,
        time_linear_2_bias: Vec<f32>,
        delta_time_linear_1_weight: Vec<f32>,
        delta_time_linear_1_bias: Vec<f32>,
        delta_time_linear_2_weight: Vec<f32>,
        delta_time_linear_2_bias: Vec<f32>,
        out_weight: Vec<f32>,
        out_bias: Vec<f32>,
    ) -> Result<Self> {
        let linear =
            |weight, bias, out, input| MiniCpm4Linear::from_source(weight, Some(bias), out, input);
        Ok(Self {
            in_proj: linear(in_weight, in_bias, LOCAL_HIDDEN, LOCAL_FEATURE)?,
            cond_proj: linear(cond_weight, cond_bias, LOCAL_HIDDEN, LOCAL_FEATURE)?,
            time_linear_1: linear(
                time_linear_1_weight,
                time_linear_1_bias,
                LOCAL_HIDDEN,
                LOCAL_HIDDEN,
            )?,
            time_linear_2: linear(
                time_linear_2_weight,
                time_linear_2_bias,
                LOCAL_HIDDEN,
                LOCAL_HIDDEN,
            )?,
            delta_time_linear_1: linear(
                delta_time_linear_1_weight,
                delta_time_linear_1_bias,
                LOCAL_HIDDEN,
                LOCAL_HIDDEN,
            )?,
            delta_time_linear_2: linear(
                delta_time_linear_2_weight,
                delta_time_linear_2_bias,
                LOCAL_HIDDEN,
                LOCAL_HIDDEN,
            )?,
            out_proj: linear(out_weight, out_bias, LOCAL_FEATURE, LOCAL_HIDDEN)?,
        })
    }
}

/// Local DiT input/output adapter.  Inputs and outputs use channel-major
/// `[64, positions]`, matching the source feature protocol.
#[derive(Debug, Clone)]
pub struct LocalDit {
    stack: MiniCpm4Stack,
    weights: LocalDitWeights,
    mean_mode: bool,
}

impl LocalDit {
    /// Production binding remains closed until VAST records an immutable
    /// complete composite manifest and exact provenance contract.
    pub fn from_gguf(_file: &GgufFile) -> Result<Self> {
        Err(VokraError::NotImplemented(
            "voxcpm2 local DiT: production GGUF binding is blocked until the exact complete composite name/shape manifest and source/tokenizer provenance are authenticated on VAST",
        ))
    }

    /// Crate-private staged loader; same-shaped input is not production
    /// authorization until the complete composite contract is landed.
    #[allow(dead_code)]
    pub(crate) fn from_staged_gguf(file: &GgufFile) -> Result<Self> {
        let weights = LocalDitWeights::from_source(
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.in_proj.weight",
                &[LOCAL_HIDDEN, LOCAL_FEATURE],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.in_proj.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.cond_proj.weight",
                &[LOCAL_HIDDEN, LOCAL_FEATURE],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.cond_proj.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.time_mlp.linear_1.weight",
                &[LOCAL_HIDDEN, LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.time_mlp.linear_1.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.time_mlp.linear_2.weight",
                &[LOCAL_HIDDEN, LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.time_mlp.linear_2.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.delta_time_mlp.linear_1.weight",
                &[LOCAL_HIDDEN, LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.delta_time_mlp.linear_1.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.delta_time_mlp.linear_2.weight",
                &[LOCAL_HIDDEN, LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.delta_time_mlp.linear_2.bias",
                &[LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.out_proj.weight",
                &[LOCAL_FEATURE, LOCAL_HIDDEN],
            )?,
            load_tensor(
                file,
                "voxcpm2",
                "feat_decoder.estimator.out_proj.bias",
                &[LOCAL_FEATURE],
            )?,
        )?;
        Self::from_source(load_local_stack(file, "feat_decoder.decoder")?, weights)
    }

    pub(crate) fn from_source(stack: MiniCpm4Stack, weights: LocalDitWeights) -> Result<Self> {
        Self::from_source_with_mean_mode(stack, weights, false)
    }

    /// Bind the source's `mean_mode` flag explicitly. The 0.5B checkpoint
    /// uses `false`, which means the delta-time estimator receives zero.
    pub fn from_source_with_mean_mode(
        stack: MiniCpm4Stack,
        weights: LocalDitWeights,
        mean_mode: bool,
    ) -> Result<Self> {
        validate_local_stack(&stack, "local DiT")?;
        Ok(Self {
            stack,
            weights,
            mean_mode,
        })
    }

    /// Estimate a velocity for channel-major `x` conditioned on channel-major
    /// `cond`. `mu` is the source's 1024-wide conditioning state. The returned
    /// tensor is channel-major `[64, x_positions]`.
    #[allow(clippy::too_many_arguments)] // The source DiT call has one argument per conditioning axis.
    pub fn forward(
        &self,
        x: &[f32],
        x_positions: usize,
        cond: &[f32],
        cond_positions: usize,
        t: f32,
        delta_t: f32,
        mu: &[f32],
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        if x_positions == 0 || cond_positions == 0 {
            return Err(VokraError::InvalidArgument(
                "VoxCPM local DiT requires non-empty x and cond".to_owned(),
            ));
        }
        if x.len() != LOCAL_FEATURE * x_positions
            || cond.len() != LOCAL_FEATURE * cond_positions
            || mu.len() != LOCAL_HIDDEN
            || x.iter().chain(cond).chain(mu).any(|v| !v.is_finite())
            || !t.is_finite()
            || !delta_t.is_finite()
        {
            return Err(VokraError::InvalidArgument(
                "local DiT input shape/finiteness mismatch".to_owned(),
            ));
        }
        let x_rows = channel_major_to_rows(x, LOCAL_FEATURE, x_positions);
        let cond_rows = channel_major_to_rows(cond, LOCAL_FEATURE, cond_positions);
        let mut x_hidden = vec![0.0; x_positions * LOCAL_HIDDEN];
        let mut cond_hidden = vec![0.0; cond_positions * LOCAL_HIDDEN];
        self.weights
            .in_proj
            .apply(compute, &x_rows, x_positions, &mut x_hidden)?;
        self.weights
            .cond_proj
            .apply(compute, &cond_rows, cond_positions, &mut cond_hidden)?;

        let time = time_embedding(t, LOCAL_HIDDEN)?;
        let delta_time = time_embedding(if self.mean_mode { delta_t } else { 0.0 }, LOCAL_HIDDEN)?;
        let mut time_hidden = vec![0.0; LOCAL_HIDDEN];
        let mut delta_hidden = vec![0.0; LOCAL_HIDDEN];
        self.weights
            .time_linear_1
            .apply(compute, &time, 1, &mut time_hidden)?;
        silu(compute, &mut time_hidden)?;
        let time_input = time_hidden.clone();
        self.weights
            .time_linear_2
            .apply(compute, &time_input, 1, &mut time_hidden)?;
        self.weights
            .delta_time_linear_1
            .apply(compute, &delta_time, 1, &mut delta_hidden)?;
        silu(compute, &mut delta_hidden)?;
        let delta_input = delta_hidden.clone();
        self.weights
            .delta_time_linear_2
            .apply(compute, &delta_input, 1, &mut delta_hidden)?;
        let mut prefix = vec![0.0; LOCAL_HIDDEN];
        for i in 0..LOCAL_HIDDEN {
            prefix[i] = time_hidden[i] + delta_hidden[i] + mu[i];
        }
        let sequence = assemble_dit_sequence(&prefix, &cond_hidden, &x_hidden);
        let decoded =
            self.stack
                .forward(&sequence, 1 + cond_positions + x_positions, false, compute)?;
        let x_start = (1 + cond_positions) * LOCAL_HIDDEN;
        let mut output_rows = vec![0.0; x_positions * LOCAL_FEATURE];
        self.weights.out_proj.apply(
            compute,
            &decoded[x_start..x_start + x_positions * LOCAL_HIDDEN],
            x_positions,
            &mut output_rows,
        )?;
        Ok(rows_to_channel_major(
            &output_rows,
            LOCAL_FEATURE,
            x_positions,
        ))
    }
}

/// UnifiedCFM reverse Euler sampler. Noise is supplied by the caller; this
/// type never creates hidden RNG state or silently injects temperature noise.
#[derive(Debug, Clone, Copy)]
pub struct UnifiedCfm {
    steps: usize,
    sway_coefficient: f32,
    cfg_scale: f32,
}

fn load_local_stack(file: &GgufFile, prefix: &str) -> Result<MiniCpm4Stack> {
    let base = MiniCpm4Config::voxcpm_0_5b()?;
    let config = MiniCpm4Config::new_with_original_max_positions(
        base.hidden_dim(),
        base.ffn_dim(),
        LOCAL_LAYERS,
        base.n_heads(),
        base.n_kv_heads(),
        base.max_positions(),
        base.original_max_positions(),
        base.rope_theta(),
        base.rms_norm_eps(),
        false,
        base.rope_short_factor().to_vec(),
        base.rope_long_factor().to_vec(),
    )?;
    let mut blocks = Vec::with_capacity(LOCAL_LAYERS);
    for layer in 0..LOCAL_LAYERS {
        let stem = format!("{prefix}.layers.{layer}");
        let tensor = |suffix: &str, shape: &[usize]| {
            load_tensor(file, "voxcpm2", &format!("{stem}.{suffix}"), shape)
        };
        blocks.push(MiniCpm4BlockWeights::from_source(
            &config,
            tensor("input_layernorm.weight", &[LOCAL_HIDDEN])?,
            tensor("post_attention_layernorm.weight", &[LOCAL_HIDDEN])?,
            tensor("self_attn.q_proj.weight", &[LOCAL_HIDDEN, LOCAL_HIDDEN])?,
            tensor("self_attn.k_proj.weight", &[128, LOCAL_HIDDEN])?,
            tensor("self_attn.v_proj.weight", &[128, LOCAL_HIDDEN])?,
            tensor("self_attn.o_proj.weight", &[LOCAL_HIDDEN, LOCAL_HIDDEN])?,
            tensor("mlp.gate_proj.weight", &[4096, LOCAL_HIDDEN])?,
            tensor("mlp.up_proj.weight", &[4096, LOCAL_HIDDEN])?,
            tensor("mlp.down_proj.weight", &[LOCAL_HIDDEN, 4096])?,
        )?);
    }
    MiniCpm4Stack::new(
        config,
        MiniCpm4StackWeights::from_source(
            blocks,
            load_tensor(
                file,
                "voxcpm2",
                &format!("{prefix}.norm.weight"),
                &[LOCAL_HIDDEN],
            )?,
        ),
    )
}

impl UnifiedCfm {
    /// Canonical 0.5B source schedule: sway coefficient is one and CFG is
    /// supplied by the authenticated config/caller.
    pub fn source_default(steps: usize, cfg_scale: f32) -> Result<Self> {
        Self::new(steps, 1.0, cfg_scale)
    }

    /// Construct an explicit UnifiedCFM Euler schedule.
    pub fn new(steps: usize, sway_coefficient: f32, cfg_scale: f32) -> Result<Self> {
        if steps == 0 || !sway_coefficient.is_finite() || !cfg_scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "VoxCPM UnifiedCFM requires positive steps and finite schedule".to_owned(),
            ));
        }
        Ok(Self {
            steps,
            sway_coefficient,
            cfg_scale,
        })
    }

    #[must_use]
    /// Number of Euler integration steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    #[must_use]
    /// Timestep sway coefficient.
    pub fn sway_coefficient(&self) -> f32 {
        self.sway_coefficient
    }

    #[must_use]
    /// Classifier-free guidance scale.
    pub fn cfg_scale(&self) -> f32 {
        self.cfg_scale
    }

    fn t_span(&self) -> Vec<f32> {
        (0..=self.steps)
            .map(|i| {
                let t = 1.0 - i as f32 / self.steps as f32;
                t + self.sway_coefficient * ((core::f32::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
            })
            .collect()
    }

    /// Integrate with separate positive/negative estimators. Each callback
    /// receives the current (swayed) time and caller-owned state. The first
    /// `max(1, int((N+1)*0.04))` estimates are intentionally ignored, matching
    /// `UnifiedCFM.forward`.
    pub fn integrate<P, N>(
        &self,
        noise: &[f32],
        mut positive: P,
        mut negative: N,
    ) -> Result<Vec<f32>>
    where
        P: FnMut(f32, &[f32]) -> Result<Vec<f32>>,
        N: FnMut(f32, &[f32]) -> Result<Vec<f32>>,
    {
        if noise.is_empty() || noise.iter().any(|v| !v.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "UnifiedCFM noise must be finite and non-empty".to_owned(),
            ));
        }
        let mut state = noise.to_vec();
        let t_span = self.t_span();
        let zero_steps = ((self.steps + 1) * 4 / 100).max(1);
        for index in 1..=self.steps {
            let dt = t_span[index - 1] - t_span[index];
            if !dt.is_finite() || dt < 0.0 {
                return Err(VokraError::InvalidArgument(
                    "UnifiedCFM sway schedule is not finite and descending".to_owned(),
                ));
            }
            if index <= zero_steps {
                // UnifiedCFM intentionally does not evaluate the estimator
                // during zero-star warm-up steps.
                continue;
            }
            let pos = positive(t_span[index - 1], &state)?;
            let neg = negative(t_span[index - 1], &state)?;
            if pos.len() != state.len()
                || neg.len() != state.len()
                || pos.iter().chain(&neg).any(|v| !v.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "UnifiedCFM estimator shape/finiteness mismatch".to_owned(),
                ));
            }
            let dot = pos.iter().zip(&neg).map(|(p, n)| p * n).sum::<f32>();
            let neg_norm = neg.iter().map(|n| n * n).sum::<f32>();
            let optimized_scale = dot / (neg_norm + 1e-8);
            for ((value, p), n) in state.iter_mut().zip(pos).zip(neg) {
                let scaled_neg = optimized_scale * n;
                *value -= dt * (scaled_neg + self.cfg_scale * (p - scaled_neg));
            }
        }
        Ok(state)
    }
}

fn validate_local_stack(stack: &MiniCpm4Stack, label: &str) -> Result<()> {
    let config = stack.config();
    if config.hidden_dim() != LOCAL_HIDDEN
        || config.ffn_dim() != 4096
        || config.n_heads() != 16
        || config.n_kv_heads() != 2
        || stack.layer_count() != LOCAL_LAYERS
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label} requires a 4-layer MiniCPM-4 stack with 1024 hidden, 4096 FFN, 16 Q heads, and 2 KV heads"
        )));
    }
    Ok(())
}

fn check_finite_len(label: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected || values.iter().any(|x| !x.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "{label} expected {expected} finite values, got {}",
            values.len()
        )));
    }
    Ok(())
}

fn channel_major_to_rows(input: &[f32], channels: usize, positions: usize) -> Vec<f32> {
    let mut rows = vec![0.0; positions * channels];
    for channel in 0..channels {
        for position in 0..positions {
            rows[position * channels + channel] = input[channel * positions + position];
        }
    }
    rows
}

fn rows_to_channel_major(input: &[f32], channels: usize, positions: usize) -> Vec<f32> {
    let mut output = vec![0.0; channels * positions];
    for channel in 0..channels {
        for position in 0..positions {
            output[channel * positions + position] = input[position * channels + channel];
        }
    }
    output
}

fn assemble_dit_sequence(prefix: &[f32], cond: &[f32], x: &[f32]) -> Vec<f32> {
    let mut sequence = Vec::with_capacity(prefix.len() + cond.len() + x.len());
    sequence.extend_from_slice(prefix);
    sequence.extend_from_slice(cond);
    sequence.extend_from_slice(x);
    sequence
}

fn silu(compute: &Compute, values: &mut [f32]) -> Result<()> {
    let input = values.to_vec();
    compute.silu_f32(&input, values)
}

fn time_embedding(value: f32, dim: usize) -> Result<Vec<f32>> {
    if dim < 2 || !value.is_finite() {
        return Err(VokraError::InvalidArgument(
            "VoxCPM time embedding requires finite value and dimension >= 2".to_owned(),
        ));
    }
    let half = dim / 2;
    let denominator = (half - 1) as f32;
    let mut output = vec![0.0; dim];
    for index in 0..half {
        let frequency = if half == 1 {
            1.0
        } else {
            ((index as f32) * (-10_000.0f32.ln()) / denominator).exp()
        };
        let angle = value * 1000.0 * frequency;
        output[index] = angle.sin();
        output[half + index] = angle.cos();
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_major_assembly_and_slice_are_deterministic() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rows = channel_major_to_rows(&input, 2, 3);
        assert_eq!(rows, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(rows_to_channel_major(&rows, 2, 3), input);
    }

    #[test]
    fn local_tensor_rank_contract_pins_special_token_and_biases() {
        let special = LOCAL_TENSOR_RANK_CONTRACT
            .iter()
            .find(|(name, _)| *name == "feat_encoder.special_token")
            .map(|(_, shape)| *shape)
            .unwrap();
        assert_eq!(special, &[1, 1, 1, 1024]);
        assert_ne!(special, &[1024]);
        let bias = LOCAL_TENSOR_RANK_CONTRACT
            .iter()
            .find(|(name, _)| *name == "feat_encoder.in_proj.bias")
            .map(|(_, shape)| *shape)
            .unwrap();
        assert_eq!(bias, &[1024]);
        assert!(LOCAL_TENSOR_RANK_CONTRACT.iter().any(|(name, shape)| *name
            == "feat_decoder.estimator.cond_proj.weight"
            && shape.len() == 2
            && shape[0] == 1024
            && shape[1] == 64));
    }

    #[test]
    fn tiny_adapter_projection_matches_independent_scalar_oracle() {
        // This exercises the same Compute-backed adapter used by LocEnc/LocDiT
        // without allocating production-sized 64x1024 matrices.
        let linear = MiniCpm4Linear::from_source(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Some(vec![0.5, -0.5, 1.0]),
            3,
            2,
        )
        .unwrap();
        let channel_major = [2.0, 4.0, 3.0, 5.0]; // channels=2, positions=2
        let rows = channel_major_to_rows(&channel_major, 2, 2);
        let mut actual = vec![0.0; 6];
        linear
            .apply(&Compute::cpu(), &rows, 2, &mut actual)
            .unwrap();
        let expected = [
            0.5 + 1.0 * 2.0 + 2.0 * 3.0,
            -0.5 + 3.0 * 2.0 + 4.0 * 3.0,
            1.0 + 5.0 * 2.0 + 6.0 * 3.0,
            0.5 + 1.0 * 4.0 + 2.0 * 5.0,
            -0.5 + 3.0 * 4.0 + 4.0 * 5.0,
            1.0 + 5.0 * 4.0 + 6.0 * 5.0,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn time_embedding_matches_sin_then_cos_contract() {
        let actual = time_embedding(0.0, 4).unwrap();
        assert_eq!(actual, vec![0.0, 0.0, 1.0, 1.0]);
        let nonzero = time_embedding(0.001, 4).unwrap();
        assert!(nonzero[..2].iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn dit_prefix_and_x_slice_are_source_ordered() {
        let sequence = assemble_dit_sequence(&[9.0, 10.0], &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(sequence, vec![9.0, 10.0, 1.0, 2.0, 3.0, 4.0]);
        let x_start = 2 + 2;
        assert_eq!(&sequence[x_start..], &[3.0, 4.0]);
    }

    #[test]
    fn cfm_applies_zero_star_and_cfg_euler() {
        let flow = UnifiedCfm::new(2, 0.0, 1.0).unwrap();
        let positive_calls = std::cell::Cell::new(0);
        let negative_calls = std::cell::Cell::new(0);
        let first_time = std::cell::Cell::new(-1.0);
        let output = flow
            .integrate(
                &[0.0],
                |t, _x| {
                    positive_calls.set(positive_calls.get() + 1);
                    first_time.set(t);
                    Ok(vec![2.0])
                },
                |_, _x| {
                    negative_calls.set(negative_calls.get() + 1);
                    Ok(vec![1.0])
                },
            )
            .unwrap();
        assert!((output[0] + 1.0).abs() < 1e-6);
        assert_eq!(positive_calls.get(), 1);
        assert_eq!(negative_calls.get(), 1);
        assert!((first_time.get() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn cfm_requires_caller_noise_and_rejects_nonfinite() {
        let flow = UnifiedCfm::new(1, 0.0, 1.0).unwrap();
        assert!(
            flow.integrate(&[f32::NAN], |_t, _x| Ok(vec![0.0]), |_t, _x| Ok(vec![0.0]))
                .is_err()
        );
    }

    #[test]
    fn source_default_pins_unit_sway() {
        let flow = UnifiedCfm::source_default(10, 2.0).unwrap();
        assert_eq!(flow.sway_coefficient(), 1.0);
        assert_eq!(flow.cfg_scale(), 2.0);
    }
}
