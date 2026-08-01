#![allow(clippy::doc_lazy_continuation)]
//! **HuBERT-Large-LS960** (`facebook/hubert-large-ls960-ft`, **apache-2.0**):
//! safetensors → GGUF conversion (Wave 7 residual, 2026-08-01).
//!
//! HuBERT (Hsu et al. 2021, arXiv:2106.07447 "HuBERT: Self-Supervised
//! Speech Representation Learning by Masked Prediction of Hidden
//! Units") = 317M-parameter self-supervised speech encoder + CTC head
//! fine-tuned on LibriSpeech 960h. Distinct from sibling
//! [`crate::models::wav2vec2_ctc`] family: HuBERT uses a **BERT-style
//! masked feature prediction** objective over k-means-clustered MFCC /
//! prior-iteration hidden states, whereas wav2vec 2.0 uses a
//! **contrastive masked convnet** objective with Gumbel-softmax
//! quantised negatives. The two topologies share the same 7-layer
//! Conv1D feature-extractor front-end + Transformer encoder body but
//! differ in the pretraining loss and the position of the CTC head —
//! `HubertForCTC` on top of `HubertModel`, not `Wav2Vec2ForCTC`. A
//! future native forward is expected to share ops with wav2vec2_ctc
//! (feature-extractor Conv1D, Transformer encoder, CTC greedy / beam
//! decode) but the arch tag stays distinct so runtime dispatch cannot
//! misroute a HuBERT checkpoint into a wav2vec2 loader (or vice
//! versa) silently (FR-EX-08).
//!
//! # License posture — apache-2.0 (**Permissive**)
//!
//! `facebook/hubert-large-ls960-ft` HF `cardData.license = apache-2.0`
//! (primary source: HF model card front-matter). Same permissive
//! posture as sibling wav2vec 2.0 family + Whisper family. No NC / SA
//! gate; publish workflow uses the default T1 (Commercial) path.
//!
//! # Scale — local convert safe (~1.26 GB)
//!
//! HuBERT-Large ships ~1.26 GB (317M params × 4 bytes fp32 = ~1.27 GB).
//! Well below the M1 iMac 16 GB safe local threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (the 8 GB strict cutoff is
//! not approached). Local convert on the owner iMac is safe.
//!
//! # Publish path (blocked on §3.1 sign-off — placeholder row registered)
//!
//! `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS` has a
//! placeholder entry mapping `vokra/hubert-large-ls960` to a `§3.1`
//! row heading (the row itself will be added to
//! `docs/license-audit.md` in a post-workflow batch — this crate does
//! NOT touch the audit doc). `publish-one.sh` fails closed at gate
//! time until the row exists with a ☑ sign-off; the runtime side is
//! decoupled (M2-13 gate looks at `vokra.provenance.weight_license`,
//! not at the audit doc row).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "hubert";
pub const NAME: &str = "hubert-large-ls960";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "facebook/hubert-large-ls960-ft";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const UPSTREAM_SOURCE: &str =
    "facebook/hubert-large-ls960-ft (Meta HuBERT-Large 317M CTC LibriSpeech 960h, apache-2.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HubertLargeLs960Report {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_hubert_large_ls960_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<HubertLargeLs960Report, ConvertError> {
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

    let mut report = HubertLargeLs960Report::default();
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
            "vokra-convert-hubert-large-ls960-{tag}-{}-{n}",
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
        let st = safetensors_one(
            "hubert.encoder.layer0.attn.q_proj",
            "F32",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_hubert_large_ls960_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

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
        // Default license is apache-2.0 (Permissive) — distinct from
        // sibling MusicGen-Small (cc-by-nc-4.0 NonCommercial).
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one(
            "hubert.encoder.layer0.attn.k_proj",
            "BF16",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_hubert_large_ls960_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_hubert_large_ls960_file(&inp, &outp, Some("mit")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.license").and_then(|v| v.as_str()),
            Some("mit")
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
