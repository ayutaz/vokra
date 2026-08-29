//! VibeVoice acoustic/semantic connector projections.
//!
//! The connectors are part of the authenticated composite checkpoint.  They
//! are deliberately separate from the Qwen decoder and diffusion head: the
//! former consumes mixed prompt rows, while the latter consumes the resulting
//! condition.  No connector constructor accepts an unverified weight bag.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{load_tensor, require_tensor_shape};

const HIDDEN: usize = 1_536;
const ACOUSTIC_FEATURES: usize = 64;
const SEMANTIC_FEATURES: usize = 128;
const RMS_EPS: f32 = 1.0e-6;

/// Learned operations required by either VibeVoice connector.
pub const VIBEVOICE_CONNECTOR_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::RmsNorm];

#[derive(Debug, Clone)]
struct Linear {
    /// Compute layout `[input, output]`, transposed from GGUF rows.
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn apply(&self, compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice connector input {}, expected {}",
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
            Some(&self.bias),
            &mut output,
        )?;
        finite("vibevoice connector linear", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ConnectorWeights {
    fc1: Linear,
    norm: Vec<f32>,
    fc2: Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConnectorKind {
    Acoustic,
    Semantic,
}

/// A strict authenticated acoustic or semantic connector.
#[derive(Debug, Clone)]
pub struct SpeechConnector {
    weights: ConnectorWeights,
    input_features: usize,
    kind: ConnectorKind,
    backend: BackendKind,
}

impl SpeechConnector {
    /// Loads the fixed acoustic connector (`model.acoustic_connector.*`).
    pub fn acoustic_from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Self::from_authenticated_gguf(
            file,
            backend,
            "model.acoustic_connector",
            ACOUSTIC_FEATURES,
            ConnectorKind::Acoustic,
        )
    }

    /// Loads the fixed semantic connector (`model.semantic_connector.*`).
    pub fn semantic_from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Self::from_authenticated_gguf(
            file,
            backend,
            "model.semantic_connector",
            SEMANTIC_FEATURES,
            ConnectorKind::Semantic,
        )
    }

    fn from_authenticated_gguf(
        file: &GgufFile,
        backend: BackendKind,
        prefix: &str,
        input_features: usize,
        kind: ConnectorKind,
    ) -> Result<Self> {
        // This is intentionally inside the public construction path.  A
        // caller cannot authenticate once and then substitute a same-shaped
        // connector from another GGUF.
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        let weights = ConnectorWeights {
            fc1: load_linear(file, &format!("{prefix}.fc1"), input_features, HIDDEN)?,
            norm: load_raw(file, &format!("{prefix}.norm"), HIDDEN)?,
            fc2: load_linear(file, &format!("{prefix}.fc2"), HIDDEN, HIDDEN)?,
        };
        validate_weights(&weights, input_features)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_CONNECTOR_HOT_OPS)?;
        Ok(Self {
            weights,
            input_features,
            kind,
            backend,
        })
    }

    /// Returns whether this is the 64-wide acoustic or 128-wide semantic path.
    #[must_use]
    pub const fn input_features(&self) -> usize {
        self.input_features
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Applies `fc1 -> RMSNorm(eps=1e-6) -> fc2` to one feature row.
    pub fn forward(&self, features: &[f32]) -> Result<Vec<f32>> {
        if features.len() != self.input_features {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice connector feature width {}, expected {}",
                features.len(),
                self.input_features
            )));
        }
        finite("vibevoice connector input", features)?;
        let compute = Compute::for_backend(self.backend, VIBEVOICE_CONNECTOR_HOT_OPS)?;
        let output = connector_forward_with_compute(&compute, &self.weights, features)?;
        if output.len() != HIDDEN {
            return Err(VokraError::ModelLoad(
                "vibevoice connector output width mismatch".to_owned(),
            ));
        }
        Ok(output)
    }
}

fn connector_forward_with_compute(
    compute: &Compute,
    weights: &ConnectorWeights,
    features: &[f32],
) -> Result<Vec<f32>> {
    if features.len() != weights.fc1.input
        || weights.fc1.output != weights.norm.len()
        || weights.fc2.input != weights.norm.len()
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice connector generic shape mismatch".to_owned(),
        ));
    }
    finite("vibevoice connector input", features)?;
    let projected = weights.fc1.apply(compute, features)?;
    let mut normalized = vec![0.0; weights.norm.len()];
    compute.rms_norm_f32(
        &projected,
        &mut normalized,
        1,
        weights.norm.len(),
        &weights.norm,
        RMS_EPS,
    )?;
    finite("vibevoice connector RMSNorm", &normalized)?;
    weights.fc2.apply(compute, &normalized)
}

/// Authenticated scalar conversion between raw and scaled acoustic latents.
#[derive(Debug, Clone, Copy)]
pub struct VibeVoiceLatentScale {
    bias_factor: f32,
    scaling_factor: f32,
}

impl VibeVoiceLatentScale {
    /// Loads `model.speech_bias_factor` and `model.speech_scaling_factor`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        super::VibeVoiceCheckpoint::from_gguf(file)?;
        let bias_factor = load_scalar(file, "model.speech_bias_factor")?;
        let scaling_factor = load_scalar(file, "model.speech_scaling_factor")?;
        if !bias_factor.is_finite() || !scaling_factor.is_finite() || scaling_factor == 0.0 {
            return Err(VokraError::ModelLoad(
                "vibevoice latent scale contains non-finite or zero scaling factor".to_owned(),
            ));
        }
        Ok(Self {
            bias_factor,
            scaling_factor,
        })
    }

    /// Constructs validated factors for bounded tests only.
    #[cfg(test)]
    fn for_test(bias_factor: f32, scaling_factor: f32) -> Result<Self> {
        if !bias_factor.is_finite() || !scaling_factor.is_finite() || scaling_factor == 0.0 {
            return Err(VokraError::InvalidArgument(
                "test latent scale factors must be finite and nonzero".to_owned(),
            ));
        }
        Ok(Self {
            bias_factor,
            scaling_factor,
        })
    }

    /// Converts a raw acoustic latent to the scaled diffusion representation.
    pub fn scale_raw(&self, latent: &[f32]) -> Result<Vec<f32>> {
        finite("vibevoice raw acoustic latent", latent)?;
        let output: Vec<f32> = latent
            .iter()
            .map(|&value| (value + self.bias_factor) * self.scaling_factor)
            .collect();
        finite("vibevoice scaled acoustic latent", &output)?;
        Ok(output)
    }

    /// Converts a generated scaled diffusion latent back to tokenizer units.
    pub fn unscale_generated(&self, latent: &[f32]) -> Result<Vec<f32>> {
        finite("vibevoice generated scaled latent", latent)?;
        let output: Vec<f32> = latent
            .iter()
            .map(|&value| value / self.scaling_factor - self.bias_factor)
            .collect();
        finite("vibevoice unscaled acoustic latent", &output)?;
        Ok(output)
    }

    /// Returns the authenticated additive bias factor.
    #[must_use]
    pub const fn bias_factor(&self) -> f32 {
        self.bias_factor
    }

    /// Returns the authenticated nonzero scaling factor.
    #[must_use]
    pub const fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }
}

/// Combines acoustic and semantic connector outputs for the next LM row.
/// `semantic_features` is the already-computed 128-wide semantic mean from
/// the semantic tokenizer; this module intentionally does not implement that
/// streaming tokenizer yet.
pub fn combine_next_lm_embedding(
    acoustic_connector: &SpeechConnector,
    scaled_acoustic_latent: &[f32],
    semantic_connector: &SpeechConnector,
    semantic_features: &[f32],
) -> Result<Vec<f32>> {
    if acoustic_connector.kind != ConnectorKind::Acoustic
        || semantic_connector.kind != ConnectorKind::Semantic
        || acoustic_connector.backend != semantic_connector.backend
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice connector kind/backend mismatch".to_owned(),
        ));
    }
    let compute = Compute::for_backend(acoustic_connector.backend, VIBEVOICE_CONNECTOR_HOT_OPS)?;
    let acoustic = connector_forward_with_compute(
        &compute,
        &acoustic_connector.weights,
        scaled_acoustic_latent,
    )?;
    let semantic =
        connector_forward_with_compute(&compute, &semantic_connector.weights, semantic_features)?;
    if acoustic.len() != HIDDEN || semantic.len() != HIDDEN {
        return Err(VokraError::ModelLoad(
            "vibevoice connector combination width mismatch".to_owned(),
        ));
    }
    combine_connector_outputs(&acoustic, &semantic)
}

fn combine_connector_outputs(acoustic: &[f32], semantic: &[f32]) -> Result<Vec<f32>> {
    if acoustic.len() != semantic.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice connector output shape mismatch".to_owned(),
        ));
    }
    let output: Vec<f32> = acoustic
        .iter()
        .zip(semantic)
        .map(|(&left, &right)| left + right)
        .collect();
    finite("vibevoice next LM embedding", &output)?;
    Ok(output)
}

fn load_raw(file: &GgufFile, name: &str, width: usize) -> Result<Vec<f32>> {
    require_tensor_shape(file, "vibevoice connector", name, &[width])?;
    load_tensor(file, "vibevoice connector", name, &[width])
}

fn load_scalar(file: &GgufFile, name: &str) -> Result<f32> {
    require_tensor_shape(file, "vibevoice connector", name, &[])?;
    let values = load_tensor(file, "vibevoice connector", name, &[])?;
    let value = *values.first().ok_or_else(|| {
        VokraError::ModelLoad(format!("vibevoice connector scalar `{name}` is empty"))
    })?;
    if values.len() != 1 {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice connector scalar `{name}` decoded {} values",
            values.len()
        )));
    }
    Ok(value)
}

fn load_linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<Linear> {
    let raw = load_tensor(
        file,
        "vibevoice connector",
        &format!("{prefix}.weight"),
        &[output, input],
    )?;
    let bias = load_raw(file, &format!("{prefix}.bias"), output)?;
    let mut weight = vec![0.0; input * output];
    for row in 0..output {
        for col in 0..input {
            weight[col * output + row] = raw[row * input + col];
        }
    }
    Ok(Linear {
        weight,
        bias,
        input,
        output,
    })
}

fn validate_weights(weights: &ConnectorWeights, input_features: usize) -> Result<()> {
    if weights.fc1.input != input_features
        || weights.fc1.output != HIDDEN
        || weights.fc1.bias.len() != HIDDEN
        || weights.fc2.input != HIDDEN
        || weights.fc2.output != HIDDEN
        || weights.fc2.bias.len() != HIDDEN
        || weights.norm.len() != HIDDEN
    {
        return Err(VokraError::ModelLoad(
            "vibevoice connector fixed shape contract mismatch".to_owned(),
        ));
    }
    Ok(())
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
    fn scale_and_unscale_round_trip() {
        let scale = VibeVoiceLatentScale::for_test(0.25, 1.75).unwrap();
        let raw = [0.0, -1.0, 2.5];
        let scaled = scale.scale_raw(&raw).unwrap();
        let restored = scale.unscale_generated(&scaled).unwrap();
        for (actual, expected) in restored.iter().zip(raw) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!(VibeVoiceLatentScale::for_test(0.0, 0.0).is_err());
    }

    #[test]
    fn combine_requires_distinct_connector_kinds_and_backend() {
        let acoustic = tiny_connector(ACOUSTIC_FEATURES, 0.1);
        let semantic = tiny_connector(SEMANTIC_FEATURES, -0.2);

        let mut wrong = tiny_connector(ACOUSTIC_FEATURES, 0.3);
        wrong.kind = ConnectorKind::Semantic;
        assert!(
            combine_next_lm_embedding(
                &wrong,
                &[1.0; ACOUSTIC_FEATURES],
                &semantic,
                &[2.0; SEMANTIC_FEATURES]
            )
            .is_err()
        );

        let mut backend_mismatch = tiny_connector(SEMANTIC_FEATURES, -0.4);
        backend_mismatch.backend = BackendKind::Metal;
        assert!(
            combine_next_lm_embedding(
                &acoustic,
                &[1.0; ACOUSTIC_FEATURES],
                &backend_mismatch,
                &[2.0; SEMANTIC_FEATURES]
            )
            .is_err()
        );
    }

    fn tiny_connector(input_features: usize, seed: f32) -> SpeechConnector {
        let fc1 = tiny_linear(input_features, 4, seed);
        let fc2 = tiny_linear(4, 4, -seed);
        let weights = ConnectorWeights {
            fc1,
            norm: vec![1.0; 4],
            fc2,
        };
        SpeechConnector {
            weights,
            input_features,
            kind: if input_features == ACOUSTIC_FEATURES {
                ConnectorKind::Acoustic
            } else {
                ConnectorKind::Semantic
            },
            backend: BackendKind::Cpu,
        }
    }

    #[test]
    fn tiny_connector_matches_independent_scalar_oracle() {
        let weights = ConnectorWeights {
            fc1: tiny_linear(2, 3, 0.2),
            norm: vec![1.1, 0.9, 1.3],
            fc2: tiny_linear(3, 2, -0.4),
        };
        let input = [0.7, -0.25];
        let output = connector_forward_with_compute(&Compute::cpu(), &weights, &input).unwrap();
        let first = scalar_linear(&weights.fc1, &input);
        let mean_square = first.iter().map(|value| value * value).sum::<f32>() / 3.0;
        let inverse = (mean_square + RMS_EPS).sqrt().recip();
        let normalized: Vec<f32> = first
            .iter()
            .zip(&weights.norm)
            .map(|(value, gamma)| value * inverse * gamma)
            .collect();
        let expected = scalar_linear(&weights.fc2, &normalized);
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(
            combine_connector_outputs(&[1.0, -2.0], &[0.5, 3.0]).unwrap(),
            [1.5, 1.0]
        );
    }

    fn tiny_linear(input: usize, output: usize, seed: f32) -> Linear {
        let mut weight = vec![0.0; input * output];
        let bias = (0..output)
            .map(|index| seed + index as f32 * 0.01)
            .collect();
        for col in 0..input {
            for row in 0..output {
                weight[col * output + row] = seed + col as f32 * 0.003 + row as f32 * 0.007;
            }
        }
        Linear {
            weight,
            bias,
            input,
            output,
        }
    }

    fn scalar_linear(linear: &Linear, input: &[f32]) -> Vec<f32> {
        (0..linear.output)
            .map(|output| {
                linear.bias[output]
                    + input
                        .iter()
                        .enumerate()
                        .map(|(index, value)| value * linear.weight[index * linear.output + output])
                        .sum::<f32>()
            })
            .collect()
    }
}
