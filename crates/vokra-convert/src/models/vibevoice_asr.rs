#![allow(clippy::doc_lazy_continuation)]
//! **VibeVoice-ASR** (`microsoft/VibeVoice-ASR`, **MIT**):
//! safetensors → GGUF conversion (Wave 6 residual, 2026-08-01).
//!
//! Microsoft VibeVoice sibling with **ASR head** (VibeVoiceForASRTraining
//! per config.json). Distinct arch tag `vibevoice_asr` vs sibling TTS
//! `vibevoice` (already published as vokra/vibevoice / vokra/vibevoice-
//! realtime-0.5b) — silently sharing arch would misroute runtime
//! dispatch to a wrong-head forward (TTS head expects encoder ID stream,
//! ASR head expects raw audio → text tokens).
//!
//! # Scale — vast.ai handoff (~16.5 GB, 8-shard safetensors)
//!
//! Full VibeVoice-Large 7B + ASR head. Above M1 iMac safe threshold per
//! memory `[[feedback-large-models-on-vast-ai]]`. Shard-merge via
//! `tools/parity/vibevoice_asr_prepare_checkpoint.py`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "vibevoice_asr";
pub const NAME: &str = "vibevoice-asr";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "microsoft/VibeVoice-ASR";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str =
    "microsoft/VibeVoice-ASR (Microsoft VibeVoice 7B ASR-head sibling, MIT)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VibeVoiceAsrReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_vibevoice_asr_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<VibeVoiceAsrReport, ConvertError> {
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
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = VibeVoiceAsrReport::default();
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
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-vibevoice-asr-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("acoustic_vae.encoder", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_vibevoice_asr_file(&inp, &outp, None).unwrap();
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |key: &str| g.get(key).and_then(|v| v.as_str()).unwrap().to_owned();
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("lm.embed", "BF16", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_vibevoice_asr_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
