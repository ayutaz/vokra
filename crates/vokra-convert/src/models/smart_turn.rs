//! **smart-turn-v2** (pipecat-ai): safetensors → GGUF conversion
//! (TIER 1 F wave, 2026-07-30).
//!
//! Input: the upstream `pipecat-ai/smart-turn-v2` release — a small
//! turn-detection classifier used in Pipecat realtime conversation
//! pipelines to decide when a user has finished speaking (a VAD variant
//! specialized for dialogue turn boundaries rather than raw voice
//! activity). Output: a GGUF carrying every F32 / F16 / BF16 tensor
//! verbatim plus the `vokra.provenance.*` / `vokra.model.*` metadata
//! chunks a future `vokra-models::smart_turn::*` loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `pipecat-ai/smart-turn-v2` (fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - **SPDX**: `bsd-2-clause` (`LicenseClass::Permissive` — bsd token
//!   matches [`LicenseClass::from_license_str`] permissive family).
//! - **Category**: `vad` (turn-taking = VAD variant; recorded under
//!   `vokra.model.category` so the model-card generator + zoo tier gate
//!   classify without a per-model switch).
//!
//! # BF16 pass-through
//!
//! Mirror of `fsmn_vad` / `firered_vad` / `wespeaker` / `neucodec` /
//! `ecapa_tdnn`.

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` — distinct from the other VADs because
/// smart-turn is a small turn-boundary classifier, not a raw voice
/// activity detector.
pub const ARCH: &str = "smart_turn";

pub const NAME: &str = "smart-turn-v2";
pub const CATEGORY: &str = "vad";
pub const UPSTREAM_HF: &str = "pipecat-ai/smart-turn-v2";
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-2-clause";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SmartTurnReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_smart_turn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SmartTurnReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some("pipecat-ai/smart-turn-v2 (turn detection classifier, bsd-2-clause)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = SmartTurnReport::default();
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

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-smart-turn-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
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

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("turn_classifier.head.weight", &[2, 3], &bf16);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_smart_turn_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("turn_classifier.head.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "bsd-2-clause resolves to Permissive"
        );
    }
}
