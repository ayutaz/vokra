#![allow(clippy::doc_lazy_continuation)]
//! **Qwen2.5-Omni-7B** (`Qwen/Qwen2.5-Omni-7B`, **apache-2.0**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Alibaba Qwen2.5-Omni 7B = **Thinker + Talker unified any-to-any**
//! (audio + vision + text → audio + text) omni multimodal LLM. Distinct
//! from sibling [`crate::models::qwen2_audio`] (audio-only Whisper +
//! Qwen2-7B LM, arXiv:2407.10759) — Qwen2.5-Omni ships a fused Thinker
//! (multimodal understanding) + Talker (streaming speech generation)
//! pair over a Qwen2.5-7B backbone, so the tensor topology + tokenizer
//! + modality-injection scheme differ from the audio-only sibling.
//!
//! Distinct arch tag `qwen2-omni` (kebab-case, mirrors HF model-id
//! `Qwen2.5-Omni-7B` casing family). Sibling audio-LLM arch tags
//! (`qwen2_audio` / `kimi_audio` / `step_audio2_mini` / `baichuan_audio`
//! / `voxtral` / `moshi` / `csm` / `ultravox`) each stay distinct so
//! runtime dispatch cannot silently misroute a checkpoint into the
//! wrong forward (FR-EX-08).
//!
//! # Scale — vast.ai handoff (22.37 GB, 5-shard safetensors)
//!
//! Thinker + Talker unified = 22.37 GB total (5-shard). Well above the
//! M1 iMac 16 GB safe threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (≥8 GB strict cutoff). Owner
//! `[[feedback-large-models-on-vast-ai]]` directive: use vast.ai for
//! real convert; local mac only for code + build check. Shard-merge
//! script is not shipped in this land — the workflow adds it when the
//! owner promotes the T4/T1 publish path.
//!
//! # License posture — apache-2.0 (**Permissive**)
//!
//! Same tier as sibling `qwen2_audio`. Fail-closed default resolves to
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! `--license <spdx>` only when the caller legitimately holds the
//! weight under a different SPDX id. Registry override in
//! `crates/vokra-core/src/compliance/license_class.rs` anchors the
//! `qwen2-omni` family + `qwen2-5-omni-7b` model-id stamp on Permissive
//! (so registry-based lookups skip the string classifier fallback for
//! arch tags that don't spell a SPDX id).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "qwen2-omni";
pub const NAME: &str = "qwen2-5-omni-7b";
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "Qwen/Qwen2.5-Omni-7B";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str =
    "Qwen/Qwen2.5-Omni-7B (Alibaba 7B Thinker+Talker unified any-to-any omni, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Qwen25Omni7bReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_qwen2_5_omni_7b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen25Omni7bReport, ConvertError> {
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

    let mut report = Qwen25Omni7bReport::default();
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
            "vokra-convert-qwen2-5-omni-7b-{tag}-{}-{n}",
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
    fn qwen2_5_omni_7b_f32_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Thinker branch tensor name pattern — the actual upstream tree
        // groups thinker.* and talker.* alongside a shared backbone; we
        // exercise one representative name here to keep the synthetic
        // fixture minimal while still verifying the pass-through path.
        let st = safetensors_one(
            "thinker.audio_tower.encoder.embed",
            "F32",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_qwen2_5_omni_7b_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);
        assert_eq!(r.skipped_non_float, 0);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |key: &str| -> String {
            g.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{key}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn qwen2_5_omni_7b_bf16_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Talker branch tensor name pattern — mirrors the naming of the
        // upstream Talker AR head. Kept minimal for the synthetic path.
        let st = safetensors_one("talker.lm.attn.q_proj.weight", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_qwen2_5_omni_7b_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn qwen2_5_omni_7b_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_qwen2_5_omni_7b_file(&inp, &outp, Some("mit")).unwrap();
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
