//! VibeVoice-Realtime-0.5B diffusion prediction head.
//!
//! This is the single-step native equivalent of Microsoft's
//! `vibevoice/modular/modular_vibevoice_diffusion_head.py` at
//! [`super::SOURCE_REVISION`].  It intentionally stops at prediction: CFG,
//! scheduler stepping, acoustic decoding, and streaming state belong to the
//! composite and are not hidden in this module.
//!
//! The upstream module uses only no-bias Linear layers, RMSNorm, SiLU, and
//! elementwise AdaLN modulation.  Matrix, norm, and activation work is routed
//! through one explicit [`Compute`] backend.  A requested GPU backend that
//! lacks a required kernel is therefore an error; this module never falls
//! back to CPU.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{load_tensor, require_tensor_shape};

/// Learned operations used by the Realtime diffusion prediction head.
pub const VIBEVOICE_STREAMING_DIFFUSION_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::RmsNorm, HotOp::Silu];

const HIDDEN: usize = 896;
const LATENT: usize = 64;
const FFN: usize = 2_688; // hidden_size * head_ffn_ratio (896 * 3.0)
const LAYERS: usize = 4;
const TIMESTEP_EMBED: usize = 256;
const MAX_PERIOD: f32 = 10_000.0;
const EPS: f32 = 1.0e-5;

#[derive(Debug, Clone)]
struct Linear {
    /// Column-major `[input, output]` storage expected by `Compute::gemm_f32`.
    weight: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn apply(&self, compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice realtime diffusion linear input {}, expected {}",
                input.len(),
                self.input
            )));
        }
        let mut output = vec![0.0; self.output];
        compute.gemm_f32(
            1,
            self.output,
            self.input,
            input,
            &self.weight,
            None,
            &mut output,
        )?;
        finite("vibevoice realtime diffusion linear", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct HeadLayer {
    norm: Vec<f32>,
    modulation: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

#[derive(Debug, Clone)]
struct FinalLayer {
    modulation: Linear,
    output: Linear,
}

#[derive(Debug, Clone)]
struct Weights {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    timestep_first: Linear,
    timestep_second: Linear,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

/// Strict Realtime diffusion prediction head on one selected backend.
#[derive(Debug, Clone)]
pub struct VibeVoiceStreamingDiffusionHead {
    weights: Weights,
    backend: BackendKind,
}

impl VibeVoiceStreamingDiffusionHead {
    /// Loads the fixed-upstream `model.prediction_head.*` tensor names.
    ///
    /// The parent Realtime binder validates architecture identity and
    /// provenance before this method accepts any head tensors.  The names
    /// below are derived from the upstream module attributes at the pinned
    /// source revision.  Complete safetensors/GGUF header and payload
    /// authentication remains a VAST gate; this loader does not claim it.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        super::VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        let weights = Weights::from_gguf(file)?;
        validate_weights(&weights)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_STREAMING_DIFFUSION_HOT_OPS)?;
        Ok(Self { weights, backend })
    }

    /// Returns the explicitly selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Predicts one 64-wide v-prediction latent for one diffusion step.
    ///
    /// This is the exact upstream forward order:
    /// `noisy_images_proj`, timestep MLP, `cond_proj`, four AdaLN/SwiGLU
    /// residual blocks, and the affine-free final RMSNorm/output projection.
    /// It does not sample or apply a scheduler step.
    pub fn forward(
        &self,
        noisy_latent: &[f32],
        condition: &[f32],
        timestep: f32,
    ) -> Result<Vec<f32>> {
        if noisy_latent.len() != LATENT || condition.len() != HIDDEN {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime diffusion input must be latent[64] and condition[896]"
                    .to_owned(),
            ));
        }
        if !timestep.is_finite() {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime diffusion timestep must be finite".to_owned(),
            ));
        }
        finite("vibevoice realtime diffusion input", noisy_latent)?;
        finite("vibevoice realtime diffusion condition", condition)?;
        let compute = Compute::for_backend(self.backend, VIBEVOICE_STREAMING_DIFFUSION_HOT_OPS)?;
        forward_with_compute(&compute, &self.weights, noisy_latent, condition, timestep)
    }
}

fn forward_with_compute(
    compute: &Compute,
    weights: &Weights,
    noisy_latent: &[f32],
    condition: &[f32],
    timestep: f32,
) -> Result<Vec<f32>> {
    if noisy_latent.len() != weights.noisy_images_proj.input
        || condition.len() != weights.cond_proj.input
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime diffusion generic input shape mismatch".to_owned(),
        ));
    }
    finite("vibevoice realtime diffusion input", noisy_latent)?;
    finite("vibevoice realtime diffusion condition", condition)?;
    if !timestep.is_finite() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime diffusion timestep must be finite".to_owned(),
        ));
    }

    let mut x = weights.noisy_images_proj.apply(compute, noisy_latent)?;
    let t_frequency = timestep_embedding(timestep);
    let t_hidden = weights.timestep_first.apply(compute, &t_frequency)?;
    let t_hidden = silu(compute, &t_hidden)?;
    let t = weights.timestep_second.apply(compute, &t_hidden)?;
    let cond = weights.cond_proj.apply(compute, condition)?;
    if cond.len() != t.len() {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime diffusion condition/time width mismatch".to_owned(),
        ));
    }
    let c: Vec<f32> = cond.into_iter().zip(t).map(|(a, b)| a + b).collect();
    finite("vibevoice realtime diffusion condition sum", &c)?;
    let c_silu = silu(compute, &c)?;

    for layer in &weights.layers {
        let normed = rms(compute, &x, &layer.norm)?;
        let modulation = layer.modulation.apply(compute, &c_silu)?;
        let (shift, scale, gate) = split_modulation(modulation, x.len())?;
        let modulated = modulate(&normed, &shift, &scale)?;
        let gate_projection = layer.gate.apply(compute, &modulated)?;
        let up = layer.up.apply(compute, &modulated)?;
        let gate_projection = silu(compute, &gate_projection)?;
        let activated: Vec<f32> = gate_projection
            .into_iter()
            .zip(up)
            .map(|(gate, up)| gate * up)
            .collect();
        let update = layer.down.apply(compute, &activated)?;
        for ((value, update), gate) in x.iter_mut().zip(update).zip(gate) {
            *value += gate * update;
        }
        finite("vibevoice realtime diffusion residual", &x)?;
    }

    let normed = rms_unaffine(compute, &x)?;
    let modulation = weights.final_layer.modulation.apply(compute, &c_silu)?;
    let (shift, scale) = split_final_modulation(modulation, x.len())?;
    let modulated = modulate(&normed, &shift, &scale)?;
    weights.final_layer.output.apply(compute, &modulated)
}

impl Weights {
    fn from_gguf(file: &GgufFile) -> Result<Self> {
        // Names are the official `VibeVoiceStreamingModel.prediction_head`
        // attributes, not a generic same-shape alias.
        let p = "model.prediction_head";
        let noisy_images_proj =
            load_linear(file, &format!("{p}.noisy_images_proj"), LATENT, HIDDEN)?;
        let cond_proj = load_linear(file, &format!("{p}.cond_proj"), HIDDEN, HIDDEN)?;
        let timestep_first = load_linear(
            file,
            &format!("{p}.t_embedder.mlp.0"),
            TIMESTEP_EMBED,
            HIDDEN,
        )?;
        let timestep_second = load_linear(file, &format!("{p}.t_embedder.mlp.2"), HIDDEN, HIDDEN)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let p = format!("{p}.layers.{index}");
            layers.push(HeadLayer {
                norm: load_raw(file, &format!("{p}.norm.weight"), HIDDEN)?,
                modulation: load_linear(
                    file,
                    &format!("{p}.adaLN_modulation.1"),
                    HIDDEN,
                    3 * HIDDEN,
                )?,
                gate: load_linear(file, &format!("{p}.ffn.gate_proj"), HIDDEN, FFN)?,
                up: load_linear(file, &format!("{p}.ffn.up_proj"), HIDDEN, FFN)?,
                down: load_linear(file, &format!("{p}.ffn.down_proj"), FFN, HIDDEN)?,
            });
        }
        let p = format!("{p}.final_layer");
        Ok(Self {
            noisy_images_proj,
            cond_proj,
            timestep_first,
            timestep_second,
            layers,
            final_layer: FinalLayer {
                modulation: load_linear(
                    file,
                    &format!("{p}.adaLN_modulation.1"),
                    HIDDEN,
                    2 * HIDDEN,
                )?,
                output: load_linear(file, &format!("{p}.linear"), HIDDEN, LATENT)?,
            },
        })
    }
}

fn load_raw(file: &GgufFile, name: &str, width: usize) -> Result<Vec<f32>> {
    require_tensor_shape(file, "vibevoice realtime diffusion", name, &[width])?;
    load_tensor(file, "vibevoice realtime diffusion", name, &[width])
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    let name = format!("{prefix}.weight");
    require_tensor_shape(
        file,
        "vibevoice realtime diffusion",
        &name,
        &[output, input],
    )?;
    let raw = load_tensor(
        file,
        "vibevoice realtime diffusion",
        &name,
        &[output, input],
    )?;
    let mut weight = vec![0.0; input * output];
    for row in 0..output {
        for col in 0..input {
            weight[col * output + row] = raw[row * input + col];
        }
    }
    Ok(Linear {
        weight,
        input,
        output,
    })
}

fn validate_weights(weights: &Weights) -> Result<()> {
    if weights.layers.len() != LAYERS
        || !valid_linear(&weights.noisy_images_proj, LATENT, HIDDEN)
        || !valid_linear(&weights.cond_proj, HIDDEN, HIDDEN)
        || !valid_linear(&weights.timestep_first, TIMESTEP_EMBED, HIDDEN)
        || !valid_linear(&weights.timestep_second, HIDDEN, HIDDEN)
        || !valid_linear(&weights.final_layer.modulation, HIDDEN, 2 * HIDDEN)
        || !valid_linear(&weights.final_layer.output, HIDDEN, LATENT)
    {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime diffusion fixed shape contract mismatch".to_owned(),
        ));
    }
    for layer in &weights.layers {
        if layer.norm.len() != HIDDEN
            || !valid_linear(&layer.modulation, HIDDEN, 3 * HIDDEN)
            || !valid_linear(&layer.gate, HIDDEN, FFN)
            || !valid_linear(&layer.up, HIDDEN, FFN)
            || !valid_linear(&layer.down, FFN, HIDDEN)
        {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime diffusion layer shape contract mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn valid_linear(linear: &Linear, input: usize, output: usize) -> bool {
    linear.input == input && linear.output == output && linear.weight.len() == input * output
}

fn timestep_embedding(timestep: f32) -> Vec<f32> {
    let half = TIMESTEP_EMBED / 2;
    let mut output = Vec::with_capacity(TIMESTEP_EMBED);
    for index in 0..half {
        let frequency = (-MAX_PERIOD.ln() * index as f32 / half as f32).exp();
        output.push((timestep * frequency).cos());
    }
    for index in 0..half {
        let frequency = (-MAX_PERIOD.ln() * index as f32 / half as f32).exp();
        output.push((timestep * frequency).sin());
    }
    output
}

fn silu(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    let mut output = vec![0.0; input.len()];
    compute.silu_f32(input, &mut output)?;
    finite("vibevoice realtime diffusion SiLU", &output)?;
    Ok(output)
}

fn rms(compute: &Compute, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() != weight.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime diffusion RMSNorm shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, 1, input.len(), weight, EPS)?;
    finite("vibevoice realtime diffusion RMSNorm", &output)?;
    Ok(output)
}

fn rms_unaffine(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime diffusion final RMSNorm input is empty".to_owned(),
        ));
    }
    let unit = vec![1.0; input.len()];
    rms(compute, input, &unit)
}

fn modulate(input: &[f32], shift: &[f32], scale: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() != shift.len() || input.len() != scale.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime diffusion AdaLN shape mismatch".to_owned(),
        ));
    }
    let output: Vec<f32> = input
        .iter()
        .zip(shift)
        .zip(scale)
        .map(|((&x, &shift), &scale)| x * (1.0 + scale) + shift)
        .collect();
    finite("vibevoice realtime diffusion AdaLN", &output)?;
    Ok(output)
}

fn split_modulation(modulation: Vec<f32>, width: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if width == 0 || modulation.len() != 3 * width {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime diffusion AdaLN modulation width mismatch".to_owned(),
        ));
    }
    Ok((
        modulation[..width].to_vec(),
        modulation[width..2 * width].to_vec(),
        modulation[2 * width..].to_vec(),
    ))
}

fn split_final_modulation(modulation: Vec<f32>, width: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    if width == 0 || modulation.len() != 2 * width {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime diffusion final modulation width mismatch".to_owned(),
        ));
    }
    Ok((modulation[..width].to_vec(), modulation[width..].to_vec()))
}

fn finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_head_constants_match_authenticated_config() {
        assert_eq!(HIDDEN, 896);
        assert_eq!(FFN, 2_688);
        assert_eq!(LAYERS, 4);
        assert_eq!(LATENT, 64);
        assert_eq!(TIMESTEP_EMBED, 256);
    }

    #[test]
    fn timestep_embedding_is_cosine_then_sine() {
        let values = timestep_embedding(0.5);
        assert_eq!(values.len(), TIMESTEP_EMBED);
        assert!((values[0] - 0.5_f32.cos()).abs() < 1.0e-6);
        assert!((values[TIMESTEP_EMBED / 2] - 0.5_f32.sin()).abs() < 1.0e-6);
    }

    #[test]
    fn adaln_chunk_order_is_shift_scale_gate() {
        let mut modulation = vec![0.0; 3 * HIDDEN];
        modulation[0] = 1.0;
        modulation[HIDDEN] = 2.0;
        modulation[2 * HIDDEN] = 3.0;
        let (shift, scale, gate) = split_modulation(modulation, HIDDEN).unwrap();
        assert_eq!((shift[0], scale[0], gate[0]), (1.0, 2.0, 3.0));
        assert_eq!(modulate(&[2.0], &[3.0], &[4.0]).unwrap(), vec![13.0]);
    }

    #[test]
    fn malformed_modulation_fails_closed() {
        assert!(split_modulation(vec![0.0; 3 * HIDDEN - 1], HIDDEN).is_err());
        assert!(modulate(&[0.0], &[], &[0.0]).is_err());
    }

    #[test]
    fn tiny_complete_head_matches_independent_scalar_oracle() {
        // This exercises the complete native forward chain without loading a
        // model. The oracle below is deliberately scalar and does not call
        // any production helper, so the test catches ordering/shape mistakes
        // in GEMM, timestep conditioning, AdaLN, SwiGLU, and the residual.
        let weights = tiny_weights();
        let compute = Compute::cpu();
        let noisy = [0.7_f32, -0.2];
        let condition = [0.3_f32, 0.8];
        let actual = forward_with_compute(&compute, &weights, &noisy, &condition, 0.37).unwrap();
        let expected = scalar_oracle(&weights, &noisy, &condition, 0.37);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }
    }

    fn tiny_weights() -> Weights {
        let layer = HeadLayer {
            norm: vec![1.1, 0.9],
            modulation: tiny_linear(2, 6, 0.13),
            gate: tiny_linear(2, 3, 0.21),
            up: tiny_linear(2, 3, -0.17),
            down: tiny_linear(3, 2, 0.29),
        };
        Weights {
            noisy_images_proj: tiny_linear(2, 2, 0.31),
            cond_proj: tiny_linear(2, 2, -0.27),
            timestep_first: tiny_linear(256, 2, 0.07),
            timestep_second: tiny_linear(2, 2, 0.19),
            layers: vec![layer],
            final_layer: FinalLayer {
                modulation: tiny_linear(2, 4, -0.23),
                output: tiny_linear(2, 2, 0.37),
            },
        }
    }

    fn tiny_linear(input: usize, output: usize, seed: f32) -> Linear {
        let mut weight = vec![0.0; input * output];
        for row in 0..output {
            for col in 0..input {
                let value = seed + (row as f32 + 1.0) * 0.031 - (col as f32 + 1.0) * 0.017;
                weight[col * output + row] = value;
            }
        }
        Linear {
            weight,
            input,
            output,
        }
    }

    fn scalar_oracle(
        weights: &Weights,
        noisy: &[f32],
        condition: &[f32],
        timestep: f32,
    ) -> Vec<f32> {
        let mut x = scalar_linear(&weights.noisy_images_proj, noisy);
        let t_frequency = scalar_timestep_embedding(timestep);
        let t_hidden = scalar_silu(&scalar_linear(&weights.timestep_first, &t_frequency));
        let t = scalar_linear(&weights.timestep_second, &t_hidden);
        let c: Vec<f32> = scalar_linear(&weights.cond_proj, condition)
            .into_iter()
            .zip(t)
            .map(|(condition, timestep)| condition + timestep)
            .collect();
        let c_silu = scalar_silu(&c);

        for layer in &weights.layers {
            let normalized = scalar_rms(&x, &layer.norm);
            let modulation = scalar_linear(&layer.modulation, &c_silu);
            let width = x.len();
            let modulated: Vec<f32> = normalized
                .iter()
                .enumerate()
                .map(|(index, value)| value * (1.0 + modulation[width + index]) + modulation[index])
                .collect();
            let gate = scalar_silu(&scalar_linear(&layer.gate, &modulated));
            let up = scalar_linear(&layer.up, &modulated);
            let activated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| gate * up)
                .collect();
            let update = scalar_linear(&layer.down, &activated);
            for (index, value) in x.iter_mut().enumerate() {
                *value += modulation[2 * width + index] * update[index];
            }
        }

        let normalized = scalar_rms(&x, &vec![1.0; x.len()]);
        let modulation = scalar_linear(&weights.final_layer.modulation, &c_silu);
        let width = x.len();
        let modulated: Vec<f32> = normalized
            .iter()
            .enumerate()
            .map(|(index, value)| value * (1.0 + modulation[width + index]) + modulation[index])
            .collect();
        scalar_linear(&weights.final_layer.output, &modulated)
    }

    fn scalar_linear(linear: &Linear, input: &[f32]) -> Vec<f32> {
        (0..linear.output)
            .map(|output| {
                input
                    .iter()
                    .enumerate()
                    .map(|(index, value)| value * linear.weight[index * linear.output + output])
                    .sum()
            })
            .collect()
    }

    fn scalar_silu(input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect()
    }

    fn scalar_timestep_embedding(timestep: f32) -> Vec<f32> {
        let half = 256 / 2;
        let mut output = Vec::with_capacity(256);
        for index in 0..half {
            let frequency = (-10_000.0_f32.ln() * index as f32 / half as f32).exp();
            output.push((timestep * frequency).cos());
        }
        for index in 0..half {
            let frequency = (-10_000.0_f32.ln() * index as f32 / half as f32).exp();
            output.push((timestep * frequency).sin());
        }
        output
    }

    fn scalar_rms(input: &[f32], weight: &[f32]) -> Vec<f32> {
        const ORACLE_EPS: f32 = 1.0e-5;
        let inverse_rms = 1.0
            / (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32
                + ORACLE_EPS)
                .sqrt();
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse_rms * weight)
            .collect()
    }
}
