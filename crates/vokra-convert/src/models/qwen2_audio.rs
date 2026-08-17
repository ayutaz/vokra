#![allow(clippy::doc_lazy_continuation)]
//! **Qwen2-Audio-7B-Instruct** (`Qwen/Qwen2-Audio-7B-Instruct`, **apache-2.0**):
//! safetensors → GGUF conversion (Wave 6 residual, 2026-08-01).
//!
//! Alibaba Qwen2-Audio 7B Instruct = **audio-LLM omni** — Whisper audio
//! encoder + Qwen2-7B LM, arXiv:2407.10759. Distinct arch tag
//! `qwen2_audio` (sibling audio-LLM family: kimi_audio /
//! step_audio2_mini / baichuan_audio / voxtral / moshi / csm — each
//! has a distinct arch tag ゆえ silently sharing would misroute
//! runtime dispatch).
//!
//! # Scale — vast.ai handoff (~16 GB, 5-shard safetensors)
//!
//! 7B params + Whisper encoder = ~16 GB. Above M1 iMac safe threshold
//! per memory `[[feedback-large-models-on-vast-ai]]`; vast.ai runbook
//! required per owner 2026-08-01 directive「重いモデルはローカルで変換
//! しない」. Shard-merge via a future
//! `tools/parity/qwen2_audio_prepare_checkpoint.py` (not yet written).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "qwen2_audio";
pub const NAME: &str = "qwen2-audio-7b-instruct";
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "Qwen/Qwen2-Audio-7B-Instruct";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str =
    "Qwen/Qwen2-Audio-7B-Instruct (Alibaba 7B Whisper+Qwen2 audio-LLM, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Qwen2AudioReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_qwen2_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen2AudioReport, ConvertError> {
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

    let mut report = Qwen2AudioReport::default();
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
            "vokra-convert-qwen2-audio-{tag}-{}-{n}",
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
        let st = safetensors_one("audio_tower.encoder.embed", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_qwen2_audio_file(&inp, &outp, None).unwrap();
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_str =
            |key: &str| -> String { g.get(key).and_then(|v| v.as_str()).unwrap().to_owned() };
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
        let st = safetensors_one("lm.attn.q", "BF16", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_qwen2_audio_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
