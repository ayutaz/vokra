//! VibeVoice-1.5B diffusion prediction head.
//!
//! This module binds only the authenticated `prediction_head.*` section of
//! the composite.  It deliberately does not provide a tokenizer, scheduler,
//! or PCM path.  All learned matrix, RMSNorm, and SiLU operations are routed
//! through one selected [`Compute`] backend; a missing backend kernel is an
//! explicit error rather than a CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{load_tensor, require_tensor_shape};

/// Learned operations required by the VibeVoice diffusion head.
pub const VIBEVOICE_DIFFUSION_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::RmsNorm, HotOp::Silu];

const HIDDEN: usize = 1_536;
const LATENT: usize = 64;
const FFN: usize = 4_608;
const LAYERS: usize = 4;
const TIMESTEP_EMBED: usize = 256;
const MAX_PERIOD: f32 = 10_000.0;
const EPS: f32 = 1.0e-5;

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

impl Linear {
    fn apply(&self, compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice diffusion linear input {}, expected {}",
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
            self.bias.as_deref(),
            &mut output,
        )?;
        finite("vibevoice diffusion linear", &output)?;
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

/// Strict VibeVoice prediction-head runtime on one selected backend.
#[derive(Debug, Clone)]
pub struct VibeVoiceDiffusionHead {
    weights: Weights,
    backend: BackendKind,
}

impl VibeVoiceDiffusionHead {
    /// Loads the authenticated `model.prediction_head.*` tensors.
    ///
    /// Authentication is enforced here; callers cannot bind an arbitrary
    /// same-shaped GGUF by constructing a private weight store themselves.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        let weights = Weights::from_gguf(file)?;
        validate_weights(&weights)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_DIFFUSION_HOT_OPS)?;
        Ok(Self { weights, backend })
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Predicts one 64-wide v-prediction latent from one condition and time.
    ///
    /// `noisy_latent` is the current 64-dimensional diffusion sample and
    /// `condition` is the 1536-dimensional hidden row from the Qwen2 LM.
    /// `timestep` is the raw scheduler timestep accepted by the official
    /// head (for example, a value near 999 at the start of inference).
    pub fn forward(
        &self,
        noisy_latent: &[f32],
        condition: &[f32],
        timestep: f32,
    ) -> Result<Vec<f32>> {
        if noisy_latent.len() != LATENT || condition.len() != HIDDEN {
            return Err(VokraError::InvalidArgument(
                "vibevoice diffusion input must be latent[64] and condition[1536]".to_owned(),
            ));
        }
        if !timestep.is_finite() {
            return Err(VokraError::InvalidArgument(
                "vibevoice diffusion timestep must be finite".to_owned(),
            ));
        }
        finite("vibevoice diffusion input", noisy_latent)?;
        finite("vibevoice diffusion condition", condition)?;
        let compute = Compute::for_backend(self.backend, VIBEVOICE_DIFFUSION_HOT_OPS)?;
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
            "vibevoice diffusion generic input shape mismatch".to_owned(),
        ));
    }
    finite("vibevoice diffusion generic input", noisy_latent)?;
    finite("vibevoice diffusion generic condition", condition)?;
    if !timestep.is_finite() {
        return Err(VokraError::InvalidArgument(
            "vibevoice diffusion timestep must be finite".to_owned(),
        ));
    }
    let mut x = weights.noisy_images_proj.apply(compute, noisy_latent)?;
    let t = timestep_embedding(timestep);
    let t = weights.timestep_first.apply(compute, &t)?;
    let t_input = t.clone();
    let mut t = vec![0.0; weights.timestep_first.output];
    compute.silu_f32(&t_input, &mut t)?;
    let t = weights.timestep_second.apply(compute, &t)?;
    let cond = weights.cond_proj.apply(compute, condition)?;
    if cond.len() != t.len() {
        return Err(VokraError::ModelLoad(
            "vibevoice diffusion condition/time width mismatch".to_owned(),
        ));
    }
    let c: Vec<f32> = cond.into_iter().zip(t).map(|(a, b)| a + b).collect();
    finite("vibevoice diffusion condition sum", &c)?;

    for layer in &weights.layers {
        let width = layer.modulation.output / 3;
        let normed = rms_dynamic(compute, &x, &layer.norm)?;
        let modulation = layer.modulation.apply(compute, &silu(compute, &c)?)?;
        let (shift, scale, gate) = split_modulation_dynamic(modulation, width)?;
        let modulated = modulate_dynamic(&normed, &shift, &scale, width)?;
        let mut gated = layer.gate.apply(compute, &modulated)?;
        let up = layer.up.apply(compute, &modulated)?;
        let gated_input = gated.clone();
        compute.silu_f32(&gated_input, &mut gated)?;
        for (value, up) in gated.iter_mut().zip(up) {
            *value *= up;
        }
        let update = layer.down.apply(compute, &gated)?;
        for ((value, update), gate) in x.iter_mut().zip(update).zip(gate) {
            *value += gate * update;
        }
        finite("vibevoice diffusion residual", &x)?;
    }

    let normed = rms_unaffine_dynamic(compute, &x)?;
    let modulation = weights
        .final_layer
        .modulation
        .apply(compute, &silu(compute, &c)?)?;
    let width = weights.final_layer.output.input;
    let (shift, scale) = split_final_modulation_dynamic(modulation, width)?;
    let modulated = modulate_dynamic(&normed, &shift, &scale, width)?;
    weights.final_layer.output.apply(compute, &modulated)
}

impl Weights {
    fn from_gguf(file: &GgufFile) -> Result<Self> {
        let p = "model.prediction_head";
        let noisy_images_proj = load_linear(
            file,
            &format!("{p}.noisy_images_proj"),
            LATENT,
            HIDDEN,
            false,
        )?;
        let cond_proj = load_linear(file, &format!("{p}.cond_proj"), HIDDEN, HIDDEN, false)?;
        let timestep_first = load_linear(
            file,
            &format!("{p}.t_embedder.mlp.0"),
            TIMESTEP_EMBED,
            HIDDEN,
            false,
        )?;
        let timestep_second = load_linear(
            file,
            &format!("{p}.t_embedder.mlp.2"),
            HIDDEN,
            HIDDEN,
            false,
        )?;
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
                    false,
                )?,
                gate: load_linear(file, &format!("{p}.ffn.gate_proj"), HIDDEN, FFN, false)?,
                up: load_linear(file, &format!("{p}.ffn.up_proj"), HIDDEN, FFN, false)?,
                down: load_linear(file, &format!("{p}.ffn.down_proj"), FFN, HIDDEN, false)?,
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
                    false,
                )?,
                output: load_linear(file, &format!("{p}.linear"), HIDDEN, LATENT, false)?,
            },
        })
    }
}

fn load_raw(file: &GgufFile, name: &str, width: usize) -> Result<Vec<f32>> {
    require_tensor_shape(file, "vibevoice diffusion", name, &[width])?;
    load_tensor(file, "vibevoice diffusion", name, &[width])
}

fn load_linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    with_bias: bool,
) -> Result<Linear> {
    let name = format!("{prefix}.weight");
    require_tensor_shape(file, "vibevoice diffusion", &name, &[output, input])?;
    let raw = load_tensor(file, "vibevoice diffusion", &name, &[output, input])?;
    let mut weight = vec![0.0; input * output];
    for row in 0..output {
        for col in 0..input {
            weight[col * output + row] = raw[row * input + col];
        }
    }
    let bias = if with_bias {
        Some(load_raw(file, &format!("{prefix}.bias"), output)?)
    } else {
        None
    };
    Ok(Linear {
        weight,
        bias,
        input,
        output,
    })
}

fn validate_weights(weights: &Weights) -> Result<()> {
    if weights.layers.len() != LAYERS {
        return Err(VokraError::ModelLoad(
            "vibevoice diffusion fixed shape contract mismatch".to_owned(),
        ));
    }
    require_linear(&weights.noisy_images_proj, LATENT, HIDDEN, false)?;
    require_linear(&weights.cond_proj, HIDDEN, HIDDEN, false)?;
    require_linear(&weights.timestep_first, TIMESTEP_EMBED, HIDDEN, false)?;
    require_linear(&weights.timestep_second, HIDDEN, HIDDEN, false)?;
    for layer in &weights.layers {
        if layer.norm.len() != HIDDEN {
            return Err(VokraError::ModelLoad(
                "vibevoice diffusion layer shape contract mismatch".to_owned(),
            ));
        }
        require_linear(&layer.modulation, HIDDEN, 3 * HIDDEN, false)?;
        require_linear(&layer.gate, HIDDEN, FFN, false)?;
        require_linear(&layer.up, HIDDEN, FFN, false)?;
        require_linear(&layer.down, FFN, HIDDEN, false)?;
    }
    require_linear(&weights.final_layer.modulation, HIDDEN, 2 * HIDDEN, false)?;
    require_linear(&weights.final_layer.output, HIDDEN, LATENT, false)?;
    Ok(())
}

fn require_linear(linear: &Linear, input: usize, output: usize, with_bias: bool) -> Result<()> {
    if linear.input != input
        || linear.output != output
        || linear.weight.len() != input * output
        || linear
            .bias
            .as_ref()
            .is_some_and(|bias| bias.len() != output)
        || linear.bias.is_some() != with_bias
    {
        return Err(VokraError::ModelLoad(
            "vibevoice diffusion linear shape/bias contract mismatch".to_owned(),
        ));
    }
    Ok(())
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
    finite("vibevoice diffusion SiLU", &output)?;
    Ok(output)
}

/// RMSNorm is evaluated in f32, matching the official
/// `RMSNorm._norm(x.float()).type_as(x)` path.  The Compute path already has
/// f32 inputs/outputs, so no additional cast is needed here.
fn rms_dynamic(compute: &Compute, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() != weight.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice diffusion RMSNorm shape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, 1, input.len(), weight, EPS)?;
    finite("vibevoice diffusion RMSNorm", &output)?;
    Ok(output)
}

fn rms_unaffine_dynamic(compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(VokraError::InvalidArgument(
            "vibevoice diffusion final RMSNorm input is empty".to_owned(),
        ));
    }
    let unit = vec![1.0; input.len()];
    let mut output = vec![0.0; input.len()];
    compute.rms_norm_f32(input, &mut output, 1, input.len(), &unit, EPS)?;
    finite("vibevoice diffusion final RMSNorm", &output)?;
    Ok(output)
}

#[allow(dead_code)] // staged until the authenticated VibeVoice composite is wired
fn modulate(input: &[f32], shift: &[f32], scale: &[f32]) -> Result<Vec<f32>> {
    modulate_dynamic(input, shift, scale, HIDDEN)
}

fn modulate_dynamic(input: &[f32], shift: &[f32], scale: &[f32], width: usize) -> Result<Vec<f32>> {
    if width == 0 || input.len() != width || shift.len() != width || scale.len() != width {
        return Err(VokraError::InvalidArgument(
            "vibevoice diffusion AdaLN shape mismatch".to_owned(),
        ));
    }
    let output: Vec<f32> = input
        .iter()
        .zip(shift)
        .zip(scale)
        .map(|((&x, &shift), &scale)| x * (1.0 + scale) + shift)
        .collect();
    finite("vibevoice diffusion AdaLN", &output)?;
    Ok(output)
}

#[allow(dead_code)] // staged until the authenticated VibeVoice composite is wired
fn split_modulation(modulation: Vec<f32>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    split_modulation_dynamic(modulation, HIDDEN)
}

fn split_modulation_dynamic(
    modulation: Vec<f32>,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if width == 0 || modulation.len() != 3 * width {
        return Err(VokraError::ModelLoad(
            "vibevoice diffusion final modulation width mismatch".to_owned(),
        ));
    }
    Ok((
        modulation[..width].to_vec(),
        modulation[width..2 * width].to_vec(),
        modulation[2 * width..].to_vec(),
    ))
}

fn split_final_modulation_dynamic(
    modulation: Vec<f32>,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if width == 0 || modulation.len() != 2 * width {
        return Err(VokraError::ModelLoad(
            "vibevoice diffusion final modulation width mismatch".to_owned(),
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
    fn timestep_embedding_is_cosine_then_sine() {
        let values = timestep_embedding(0.5);
        assert_eq!(values.len(), 256);
        assert!((values[0] - 0.5_f32.cos()).abs() < 1.0e-6);
        assert!((values[128] - 0.5_f32.sin()).abs() < 1.0e-6);
    }

    #[test]
    fn adaln_chunk_order_is_shift_scale_gate() {
        let input = vec![2.0; HIDDEN];
        let shift = vec![3.0; HIDDEN];
        let scale = vec![4.0; HIDDEN];
        let output = modulate(&input, &shift, &scale).unwrap();
        assert_eq!(output[0], 13.0);
        let mut modulation = vec![0.0; 3 * HIDDEN];
        modulation[0] = 1.0;
        modulation[HIDDEN] = 2.0;
        modulation[2 * HIDDEN] = 3.0;
        let (shift, scale, gate) = split_modulation(modulation).unwrap();
        assert_eq!((shift[0], scale[0], gate[0]), (1.0, 2.0, 3.0));
    }

    #[test]
    fn invalid_modulation_shapes_fail_closed() {
        assert!(modulate(&[0.0; HIDDEN], &[0.0; HIDDEN - 1], &[0.0; HIDDEN]).is_err());
    }

    #[test]
    fn tiny_linear_uses_compute_column_major_layout() {
        let linear = tiny_linear(2, 2, 0.31);
        let raw =
            |row: usize, col: usize| 0.31 + (row as f32 + 1.0) * 0.031 - (col as f32 + 1.0) * 0.017;
        // `tiny_linear` models load_linear's [output,input] GGUF rows after
        // transposing them into Compute's [input,output] layout.
        assert_eq!(
            linear.weight,
            vec![raw(0, 0), raw(1, 0), raw(0, 1), raw(1, 1)]
        );
    }

    #[test]
    fn tiny_complete_head_matches_independent_scalar_oracle() {
        let weights = tiny_weights();
        let compute = Compute::cpu();
        let noisy = [0.7_f32, -0.2];
        let condition = [0.3_f32, 0.8];
        let actual = forward_with_compute(&compute, &weights, &noisy, &condition, 0.37).unwrap();
        let expected = scalar_oracle(&weights, &noisy, &condition, 0.37);
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
                let raw = seed + (row as f32 + 1.0) * 0.031 - (col as f32 + 1.0) * 0.017;
                weight[col * output + row] = raw;
            }
        }
        Linear {
            weight,
            bias: None,
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
        let mut frequency = Vec::with_capacity(TIMESTEP_EMBED);
        let half = TIMESTEP_EMBED / 2;
        for index in 0..half {
            let f = (-MAX_PERIOD.ln() * index as f32 / half as f32).exp();
            frequency.push((timestep * f).cos());
        }
        for index in 0..half {
            let f = (-MAX_PERIOD.ln() * index as f32 / half as f32).exp();
            frequency.push((timestep * f).sin());
        }
        let t = scalar_silu(&scalar_linear(&weights.timestep_first, &frequency));
        let t = scalar_linear(&weights.timestep_second, &t);
        let c: Vec<f32> = scalar_linear(&weights.cond_proj, condition)
            .into_iter()
            .zip(t)
            .map(|(a, b)| a + b)
            .collect();
        for layer in &weights.layers {
            let normalized = scalar_rms(&x, &layer.norm);
            let modulation = scalar_linear(&layer.modulation, &scalar_silu(&c));
            let width = x.len();
            let modulated: Vec<f32> = normalized
                .iter()
                .enumerate()
                .map(|(i, value)| value * (1.0 + modulation[width + i]) + modulation[i])
                .collect();
            let gate = scalar_silu(&scalar_linear(&layer.gate, &modulated));
            let up = scalar_linear(&layer.up, &modulated);
            let activated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate, up)| gate * up)
                .collect();
            let update = scalar_linear(&layer.down, &activated);
            for (i, value) in x.iter_mut().enumerate() {
                *value += modulation[2 * width + i] * update[i];
            }
        }
        let normalized = scalar_rms_unaffine(&x);
        let modulation = scalar_linear(&weights.final_layer.modulation, &scalar_silu(&c));
        let width = x.len();
        let modulated: Vec<f32> = normalized
            .iter()
            .enumerate()
            .map(|(i, value)| value * (1.0 + modulation[width + i]) + modulation[i])
            .collect();
        scalar_linear(&weights.final_layer.output, &modulated)
    }

    fn scalar_linear(linear: &Linear, input: &[f32]) -> Vec<f32> {
        (0..linear.output)
            .map(|output| {
                let mut value = linear.bias.as_ref().map_or(0.0, |bias| bias[output]);
                for (index, input) in input.iter().enumerate() {
                    value += input * linear.weight[index * linear.output + output];
                }
                value
            })
            .collect()
    }

    fn scalar_silu(input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect()
    }

    fn scalar_rms(input: &[f32], weight: &[f32]) -> Vec<f32> {
        let inverse = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32
            + EPS)
            .sqrt()
            .recip();
        input
            .iter()
            .zip(weight)
            .map(|(value, weight)| value * inverse * weight)
            .collect()
    }

    fn scalar_rms_unaffine(input: &[f32]) -> Vec<f32> {
        let inverse = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32
            + EPS)
            .sqrt()
            .recip();
        input.iter().map(|value| value * inverse).collect()
    }
}
