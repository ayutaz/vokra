//! **MetricGAN+** (SpeechBrain speechbrain/metricgan-plus-voicebank,
//! apache-2.0) — safetensors → GGUF conversion (perceptual-metric-tuned
//! speech enhancement GAN).
//!
//! # Provenance
//!
//! - **HF path**: `speechbrain/metricgan-plus-voicebank` (SpeechBrain
//!   distribution — safetensors + a Python pipeline; the LICENSE at
//!   `github.com/speechbrain/speechbrain/blob/develop/LICENSE` is
//!   Apache-2.0 and covers the trained release).
//! - **License (SPDX)**: `apache-2.0` — permissive; no runtime-side
//!   attribution obligation under the Apache-2.0 grant.
//! - **Category**: `enhancement` (per implementer spec — MetricGAN+ is a
//!   generator-only speech enhancement model that optimises perceptual
//!   metrics such as PESQ).
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2 / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. The observability
//! counter [`MetricganPlusReport::bf16_passthrough`] records how many
//! BF16 tensors landed on this arm so a silent widen / downcast cannot
//! slip in undetected.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM
//! contract). Real-weight parity + a native
//! `MetricganPlus::from_gguf` forward path are deferred to owner
//! sign-off (`docs/license-audit.md` §3.1) — this converter provides the
//! byte-parallel GGUF surface only. The internal generator topology
//! (LSTM stack + spectral-mask head over log-magnitude STFT) is
//! intentionally NOT re-implemented on this pass: transcribing that
//! from the SpeechBrain source is a `loud-partial` sibling wave.
//!
//! # No ONNX (permanent)
//!
//! MetricGAN+ ships PyTorch checkpoints (safetensors); this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/metricgan_plus/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MetricGAN+ GGUFs. Distinct from every other
/// enhancement / denoise sibling — MetricGAN+'s generator-only surface
/// (no discriminator at inference) is a distinct load path from
/// dual-branch or U-Net topologies.
pub const ARCH: &str = "metricgan_plus";

/// `vokra.model.name` for the VoiceBank-tuned release.
pub const NAME: &str = "metricgan-plus-voicebank";

/// `vokra.model.category` — `enhancement`.
pub const CATEGORY: &str = "enhancement";

/// `vokra.provenance.upstream_hf` slug (`org/name`).
pub const UPSTREAM_HF: &str = "speechbrain/metricgan-plus-voicebank";

/// Default upstream weight license (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a MetricGAN+ conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MetricganPlusReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm.
    pub bf16_passthrough: usize,
}

/// File-based MetricGAN+ converter.
///
/// Reads `input` (upstream `speechbrain/metricgan-plus-voicebank`
/// safetensors), writes a Vokra GGUF to `output`.
///
/// `license` optionally overrides the default `apache-2.0` provenance
/// stamp (same override pattern as `wespeaker::convert_wespeaker_file`).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_metricgan_plus_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MetricganPlusReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "speechbrain/metricgan-plus-voicebank \
             (MetricGAN+ speech enhancement, VoiceBank, apache-2.0)",
        ),
    );

    let mut report = MetricganPlusReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-metricgan-plus-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn bf16_round_trips_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Real MetricGAN+ generator names live under `enhance_model.*` in
        // the SpeechBrain checkpoint — use a realistic prefix so a
        // future `from_gguf` walk can be tested against the same shape.
        let input_bytes = safetensors_one_bf16("enhance_model.blstm.weight_ih_l0", &[2, 3], &bf16);
        let input = write_temp("bf16-in", &input_bytes);
        let output = write_temp("bf16-out", &[]);

        let report =
            convert_metricgan_plus_file(&input, &output, None).expect("convert MetricGAN+");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("enhance_model.blstm.weight_ih_l0")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn f32_and_f16_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 4] = [0x3C00, 0xC000, 0xB800, 0x4200];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"enhance.a":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_len}]}},"enhance.b":{{"dtype":"F16","shape":[2,2],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        let input = write_temp("mixed-in", &input_bytes);
        let output = write_temp("mixed-out", &[]);

        let report = convert_metricgan_plus_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
