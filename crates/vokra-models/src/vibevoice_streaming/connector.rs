//! Native acoustic connector for Microsoft VibeVoice-Realtime-0.5B.
//!
//! The pinned Microsoft source defines `SpeechConnector` as two affine
//! projections around a Llama RMSNorm.  The Realtime checkpoint has one
//! acoustic connector only: `64 -> 896 -> 896`.  This is deliberately a
//! separate type from the 1.5B VibeVoice connector, whose language width is
//! 1536 and whose checkpoint contract is different.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::load_tensor;

/// Input width of the authenticated acoustic VAE latent.
pub const VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_INPUT_WIDTH: usize = 64;
/// Output width consumed by the Realtime Qwen language model.
pub const VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH: usize = 896;
/// LlamaRMSNorm epsilon from the pinned Microsoft source.
pub const VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_RMS_EPS: f32 = 1.0e-6;

/// Backend operations required by the acoustic connector.
pub const VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::RmsNorm];

const LABEL: &str = "vibevoice realtime acoustic connector";
const PREFIX: &str = "model.acoustic_connector";

#[derive(Debug, Clone)]
struct Linear {
    /// Row-major Compute layout `[input, output]`, transposed from GGUF.
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}

impl Linear {
    fn apply_rows(&self, compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
        if self.input == 0 || self.output == 0 || input.is_empty() || input.len() % self.input != 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: linear input shape mismatch: input={}, expected a non-empty multiple of {}",
                input.len(),
                self.input
            )));
        }
        finite("acoustic connector input", input)?;
        let rows = input.len() / self.input;
        let mut output = vec![0.0; rows * self.output];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight,
            Some(&self.bias),
            &mut output,
        )?;
        finite("acoustic connector linear", &output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ConnectorWeights {
    fc1: Linear,
    norm: Vec<f32>,
    fc2: Linear,
}

/// Authenticated native Realtime acoustic connector.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeAcousticConnector {
    weights: ConnectorWeights,
    backend: BackendKind,
}

impl VibeVoiceRealtimeAcousticConnector {
    /// Binds `model.acoustic_connector.*` after the complete Realtime
    /// checkpoint identity, namespace, shape, and dtype gate has passed.
    pub fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        // Authentication intentionally happens before decoding any connector
        // payload.  A caller cannot substitute a same-shaped connector from a
        // different GGUF or bypass the strict Realtime namespace gate.
        super::VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        let weights = ConnectorWeights {
            fc1: load_linear(
                file,
                "fc1",
                VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_INPUT_WIDTH,
                VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH,
            )?,
            norm: load_tensor(
                file,
                LABEL,
                "model.acoustic_connector.norm.weight",
                &[VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH],
            )?,
            fc2: load_linear(
                file,
                "fc2",
                VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH,
                VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH,
            )?,
        };
        validate_weights(&weights)?;
        let _ = Compute::for_backend(backend, VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_HOT_OPS)?;
        Ok(Self { weights, backend })
    }

    /// Returns the explicitly selected backend.  No CPU fallback is used.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Applies `fc1 -> LlamaRMSNorm(eps=1e-6) -> fc2` to one or more rows.
    ///
    /// The input is row-major `[rows, 64]`; the returned buffer is row-major
    /// `[rows, 896]`.  This method is intentionally limited to f32 host
    /// activations.  GGUF source tensors may be F32/F16/BF16, but quantized or
    /// otherwise unsupported tensor dtypes are rejected by the checkpoint
    /// binder before this object can be constructed.
    pub fn forward(&self, features: &[f32]) -> Result<Vec<f32>> {
        let compute =
            Compute::for_backend(self.backend, VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_HOT_OPS)?;
        connector_forward_with_compute(&compute, &self.weights, features)
    }
}

fn connector_forward_with_compute(
    compute: &Compute,
    weights: &ConnectorWeights,
    features: &[f32],
) -> Result<Vec<f32>> {
    let Some(fc1_weight_len) = weights.fc1.input.checked_mul(weights.fc1.output) else {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime acoustic connector fc1 weight shape overflows".to_owned(),
        ));
    };
    let Some(fc2_weight_len) = weights.fc2.input.checked_mul(weights.fc2.output) else {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime acoustic connector fc2 weight shape overflows".to_owned(),
        ));
    };
    if weights.fc1.input == 0
        || weights.fc1.output == 0
        || weights.fc1.output != weights.norm.len()
        || weights.fc1.bias.len() != weights.fc1.output
        || weights.fc1.weight.len() != fc1_weight_len
        || weights.fc2.input != weights.norm.len()
        || weights.fc2.output == 0
        || weights.fc2.bias.len() != weights.fc2.output
        || weights.fc2.weight.len() != fc2_weight_len
    {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime acoustic connector generic shape contract mismatch".to_owned(),
        ));
    }
    finite("acoustic connector input", features)?;
    let projected = weights.fc1.apply_rows(compute, features)?;
    let rows = features.len() / weights.fc1.input;
    let mut normalized = vec![0.0; projected.len()];
    compute.rms_norm_f32(
        &projected,
        &mut normalized,
        rows,
        weights.norm.len(),
        &weights.norm,
        VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_RMS_EPS,
    )?;
    finite("acoustic connector RMSNorm", &normalized)?;
    weights.fc2.apply_rows(compute, &normalized)
}

fn load_linear(file: &GgufFile, suffix: &str, input: usize, output: usize) -> Result<Linear> {
    let weight_name = format!("{PREFIX}.{suffix}.weight");
    let bias_name = format!("{PREFIX}.{suffix}.bias");
    let raw = load_tensor(file, LABEL, &weight_name, &[output, input])?;
    let bias = load_tensor(file, LABEL, &bias_name, &[output])?;
    let mut weight = vec![0.0; input * output];
    for row in 0..output {
        for column in 0..input {
            weight[column * output + row] = raw[row * input + column];
        }
    }
    Ok(Linear {
        weight,
        bias,
        input,
        output,
    })
}

fn validate_weights(weights: &ConnectorWeights) -> Result<()> {
    if weights.fc1.input != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_INPUT_WIDTH
        || weights.fc1.output != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
        || weights.fc1.bias.len() != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
        || weights.norm.len() != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
        || weights.fc2.input != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
        || weights.fc2.output != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
        || weights.fc2.bias.len() != VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_OUTPUT_WIDTH
    {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime acoustic connector fixed shape contract mismatch".to_owned(),
        ));
    }
    finite("acoustic connector fc1 weight", &weights.fc1.weight)?;
    finite("acoustic connector fc1 bias", &weights.fc1.bias)?;
    finite("acoustic connector norm", &weights.norm)?;
    finite("acoustic connector fc2 weight", &weights.fc2.weight)?;
    finite("acoustic connector fc2 bias", &weights.fc2.bias)?;
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
    use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};

    fn tiny_linear(input: usize, output: usize) -> Linear {
        let weight = (0..input * output)
            .map(|index| (index as f32 + 1.0) * 0.1)
            .collect();
        let bias = (0..output).map(|index| index as f32 * 0.25).collect();
        Linear {
            weight,
            bias,
            input,
            output,
        }
    }

    #[test]
    fn synthetic_connector_matches_fc1_rmsnorm_fc2_oracle() {
        let weights = ConnectorWeights {
            fc1: tiny_linear(2, 3),
            norm: vec![1.0, 0.5, 2.0],
            fc2: tiny_linear(3, 2),
        };
        let input = [0.5, -1.0];
        let output = connector_forward_with_compute(&Compute::cpu(), &weights, &input).unwrap();

        let projected = [
            0.1 * 0.5 - 0.4,
            0.2 * 0.5 - 0.5 + 0.25,
            0.3 * 0.5 - 0.6 + 0.5,
        ];
        let inverse = (projected.iter().map(|value| value * value).sum::<f32>() / 3.0
            + VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_RMS_EPS)
            .sqrt()
            .recip();
        let normalized = [
            projected[0] * inverse,
            projected[1] * inverse * 0.5,
            projected[2] * inverse * 2.0,
        ];
        let expected = [
            0.1 * normalized[0] + 0.3 * normalized[1] + 0.5 * normalized[2],
            0.2 * normalized[0] + 0.4 * normalized[1] + 0.6 * normalized[2] + 0.25,
        ];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn rejects_malformed_shape_and_nonfinite_input() {
        let weights = ConnectorWeights {
            fc1: tiny_linear(2, 3),
            norm: vec![1.0; 3],
            fc2: tiny_linear(3, 2),
        };
        let compute = Compute::cpu();
        assert!(connector_forward_with_compute(&compute, &weights, &[1.0]).is_err());
        assert!(connector_forward_with_compute(&compute, &weights, &[f32::NAN, 1.0]).is_err());
    }

    #[test]
    fn refuses_wrong_architecture_before_connector_decode() {
        let mut builder = GgufBuilder::new();
        builder
            .add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive")
            .add_string(chunks::KEY_MODEL_ARCH, "vibevoice");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error =
            VibeVoiceRealtimeAcousticConnector::from_gguf(&file, BackendKind::Cpu).unwrap_err();
        assert!(error.to_string().contains("vibevoice_streaming"));
    }
}
