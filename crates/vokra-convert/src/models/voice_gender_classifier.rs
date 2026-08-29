//! JaesungHuh voice-gender classifier safetensors → GGUF conversion.
//!
//! The 202-tensor checkpoint is not the 200-tensor SpeechBrain speaker ECAPA
//! release. Its names and topology are retained verbatim so an artifact
//! carrying this model can only bind through the dedicated
//! `voice_gender_classifier` runtime architecture.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "voice_gender_classifier";
pub const NAME: &str = "voice-gender-classifier";
pub const CATEGORY: &str = "classification";
pub const UPSTREAM_HF: &str = "JaesungHuh/voice-gender-classifier";
pub const UPSTREAM_REVISION: &str = "49bcbecfd929ba5a043bde645fdff1a375eb79c7";
pub const UPSTREAM_HF_REVISION: &str = "db1222153bd60337e900be22add7af180452adc0";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const TENSOR_COUNT: usize = 202;
const CLASS_COUNT: usize = 2;
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.voice_gender.upstream_revision";
const KEY_UPSTREAM_HF_REVISION: &str = "vokra.voice_gender.upstream_hf_revision";
const KEY_SAMPLE_RATE: &str = "vokra.voice_gender.sample_rate";
const KEY_N_MELS: &str = "vokra.voice_gender.n_mels";
const KEY_N_FFT: &str = "vokra.voice_gender.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.voice_gender.win_length";
const KEY_HOP_LENGTH: &str = "vokra.voice_gender.hop_length";
const KEY_F_MIN: &str = "vokra.voice_gender.f_min";
const KEY_F_MAX: &str = "vokra.voice_gender.f_max";
const KEY_TDNN_CHANNELS: &str = "vokra.voice_gender.tdnn_channels";
const KEY_MFA_CHANNELS: &str = "vokra.voice_gender.mfa_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.voice_gender.attention_channels";
const KEY_EMBED_DIM: &str = "vokra.voice_gender.embed_dim";
const KEY_CLASS_COUNT: &str = "vokra.voice_gender.class_count";
const KEY_LABELS: &str = "vokra.voice_gender.labels";
const KEY_FRONTEND: &str = "vokra.voice_gender.frontend";
const KEY_LAYOUT: &str = "vokra.voice_gender.artifact_layout";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counters emitted by the voice-gender safetensors conversion.
pub struct VoiceGenderClassifierReport {
    /// Number of tensors read from the source manifest.
    pub read: usize,
    /// Number of floating-point tensors written to GGUF.
    pub written: usize,
    /// Number of non-floating-point tensors skipped defensively.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors passed through without widening.
    pub bf16_passthrough: usize,
}

/// Converts the pinned 202-tensor voice-gender safetensors manifest to GGUF.
pub fn convert_voice_gender_classifier_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<VoiceGenderClassifierReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_manifest(&st)?;
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_CATEGORY, CATEGORY);
    let effective_license = canonical_license(license)?;
    let license_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut builder,
        license_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_UPSTREAM_HF_REVISION, UPSTREAM_HF_REVISION);
    builder.add_u32(KEY_SAMPLE_RATE, 16_000);
    builder.add_u32(KEY_N_MELS, 80);
    builder.add_u32(KEY_N_FFT, 512);
    builder.add_u32(KEY_WIN_LENGTH, 400);
    builder.add_u32(KEY_HOP_LENGTH, 160);
    builder.add_f32(KEY_F_MIN, 20.0);
    builder.add_f32(KEY_F_MAX, 7_600.0);
    builder.add_u32(KEY_TDNN_CHANNELS, 1_024);
    builder.add_u32(KEY_MFA_CHANNELS, 1_536);
    builder.add_u32(KEY_ATTENTION_CHANNELS, 256);
    builder.add_u32(KEY_EMBED_DIM, 192);
    builder.add_u32(KEY_CLASS_COUNT, CLASS_COUNT as u32);
    builder.add_string(KEY_LABELS, "male,female");
    builder.add_string(KEY_FRONTEND, "torchaudio-mel-v1");
    builder.add_string(KEY_LAYOUT, "voice-gender-classifier-202-v1");

    let mut report = VoiceGenderClassifierReport::default();
    for tensor in st.tensors() {
        report.read += 1;
        match tensor.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                builder.add_tensor(
                    &tensor.name,
                    tensor.dtype,
                    tensor.shape.clone(),
                    st.tensor_bytes(tensor).to_vec(),
                )?;
                report.written += 1;
                if tensor.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => report.skipped_non_float += 1,
        }
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Parse(error.to_string()))?;
    std::fs::write(output, output_bytes)?;
    Ok(report)
}

fn canonical_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    match license {
        None => Ok(DEFAULT_LICENSE_SPDX),
        Some(value) if value.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) => Ok(DEFAULT_LICENSE_SPDX),
        Some(value) => Err(ConvertError::Parse(format!(
            "{ARCH}: only the audited MIT weight license is accepted, got `{value}`"
        ))),
    }
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: expected exactly {TENSOR_COUNT} tensors, got {}",
            st.tensors().len()
        )));
    }
    if expected.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "{ARCH}: internal manifest has {} entries, expected {TENSOR_COUNT}",
            expected.len()
        )));
    }
    for (name, shape) in expected {
        let Some(tensor) = st.tensors().iter().find(|tensor| tensor.name == name) else {
            return Err(ConvertError::Parse(format!(
                "{ARCH}: missing tensor `{name}`"
            )));
        };
        let expected_shape = shape.iter().map(|&value| value as u64).collect::<Vec<_>>();
        if tensor.shape != expected_shape {
            return Err(ConvertError::Parse(format!(
                "{ARCH}: tensor `{name}` has shape {:?}, expected {expected_shape:?}",
                tensor.shape,
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> Vec<(String, Vec<usize>)> {
    let mut manifest = Vec::with_capacity(TENSOR_COUNT);
    push_conv(&mut manifest, "conv1", 80, 1_024, 5);
    push_norm(&mut manifest, "bn1", 1_024);
    for layer in 1..=3 {
        let prefix = format!("layer{layer}");
        push_tdnn(
            &mut manifest,
            &format!("{prefix}.conv1"),
            &format!("{prefix}.bn1"),
            1_024,
            1_024,
            1,
        );
        for inner in 0..7 {
            push_tdnn(
                &mut manifest,
                &format!("{prefix}.convs.{inner}"),
                &format!("{prefix}.bns.{inner}"),
                128,
                128,
                3,
            );
        }
        push_tdnn(
            &mut manifest,
            &format!("{prefix}.conv3"),
            &format!("{prefix}.bn3"),
            1_024,
            1_024,
            1,
        );
        push_conv(&mut manifest, &format!("{prefix}.se.se.1"), 1_024, 128, 1);
        push_conv(&mut manifest, &format!("{prefix}.se.se.3"), 128, 1_024, 1);
    }
    push_conv(&mut manifest, "layer4", 3_072, 1_536, 1);
    push_conv(&mut manifest, "attention.0", 4_608, 256, 1);
    push_norm(&mut manifest, "attention.2", 256);
    push_conv(&mut manifest, "attention.4", 256, 1_536, 1);
    push_norm(&mut manifest, "bn5", 3_072);
    push_linear(&mut manifest, "fc6", 3_072, 192);
    push_norm(&mut manifest, "bn6", 192);
    push_linear(&mut manifest, "fc7", 192, 2);
    manifest
}

fn push_tdnn(
    manifest: &mut Vec<(String, Vec<usize>)>,
    conv: &str,
    norm: &str,
    input: usize,
    output: usize,
    kernel: usize,
) {
    push_conv(manifest, conv, input, output, kernel);
    push_norm(manifest, norm, output);
}

fn push_conv(
    manifest: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) {
    manifest.push((format!("{prefix}.weight"), vec![output, input, kernel]));
    manifest.push((format!("{prefix}.bias"), vec![output]));
}

fn push_linear(
    manifest: &mut Vec<(String, Vec<usize>)>,
    prefix: &str,
    input: usize,
    output: usize,
) {
    manifest.push((format!("{prefix}.weight"), vec![output, input]));
    manifest.push((format!("{prefix}.bias"), vec![output]));
}

fn push_norm(manifest: &mut Vec<(String, Vec<usize>)>, prefix: &str, channels: usize) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        manifest.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_manifest_is_distinct_and_complete() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert!(
            manifest
                .iter()
                .any(|(name, shape)| { name == "fc7.weight" && shape == &[2, 192] })
        );
        assert!(
            manifest
                .iter()
                .any(|(name, shape)| { name == "attention.0.weight" && shape == &[256, 4_608, 1] })
        );
    }

    #[test]
    fn identity_constants_are_pinned() {
        assert_eq!(ARCH, "voice_gender_classifier");
        assert_eq!(UPSTREAM_REVISION.len(), 40);
        assert_eq!(CLASS_COUNT, 2);
    }

    #[test]
    fn license_override_is_mit_only_and_canonicalized() {
        assert_eq!(canonical_license(None).unwrap(), "mit");
        assert_eq!(canonical_license(Some("MIT")).unwrap(), "mit");
        assert!(canonical_license(Some("apache-2.0")).is_err());
    }
}
