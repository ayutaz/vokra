#![allow(clippy::doc_lazy_continuation)]
//! **XTTS-v2** (`coqui/XTTS-v2`, **coqui-public-model-license**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Coqui's XTTS v2 = multilingual zero-shot voice-cloning TTS = GPT-2
//! backbone (~1.9 GB checkpoint) that autoregressively generates discrete
//! Mel VQ tokens (from a Discrete VAE / DVAE) conditioned on a speaker
//! conditioning module (Perceiver-style latent) + language embedding, then
//! decodes the tokens via a HiFi-GAN vocoder head. Distinct arch tag
//! `xtts` — the GPT-2 + DVAE + HiFi-GAN triple is a distinct topology from
//! sibling TTS families (piper-plus MB-iSTFT-VITS2, Kokoro
//! StyleTTS2-derived iSTFTNet, CosyVoice2 FSQ + Qwen2.5 + HiFTNet), so
//! FR-EX-08 (no silent op-shape misroute) requires the distinct arch tag.
//!
//! # License posture — Coqui Public Model License (**NonCommercial**)
//!
//! The XTTS-v2 weights ship under Coqui's own **coqui-public-model-license**
//! (a bespoke research-only / non-commercial license, not an SPDX-listed
//! identifier). The class here is [`LicenseClass::NonCommercial`] under
//! the same T4 (Research-only) tier as sibling X-Codec-2 (cc-by-nc-4.0,
//! 2026-07-28 T4 precedent) and MusicGen family (cc-by-nc-4.0). Publish
//! requires `publish-one.sh --allow-noncommercial` per T4 precedent;
//! `LicenseClass::NonCommercial` is fail-closed at the M2-13 runtime gate
//! (commercial-mode load refuses).
//!
//! Note: Coqui as a company shut down in Jan 2024. The upstream repo
//! `coqui/XTTS-v2` on HF is the primary source; downstream forks (e.g.
//! `idiap/coqui-ai-TTS`) inherit the same license.
//!
//! # Scale — local convert safe (~1.90 GB)
//!
//! XTTS-v2 ships ~1.90 GB (GPT-2 backbone + DVAE + HiFi-GAN vocoder head).
//! Below the vast.ai ≥8 GB cutoff per memory
//! `[[feedback-large-models-on-vast-ai]]` = local convert safe on M1 iMac
//! 16 GB (peak footprint ~4 GB for read + build). Sibling to the
//! `ultravox_v0_5_llama_3_2_1b.rs` (~1.83 GB) local-safe entry.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "xtts";
pub const NAME: &str = "xtts-v2";
pub const CATEGORY: &str = "tts";
pub const UPSTREAM_HF: &str = "coqui/XTTS-v2";
pub const DEFAULT_LICENSE_SPDX: &str = "coqui-public-model-license";

const UPSTREAM_SOURCE: &str = "coqui/XTTS-v2 (Coqui multilingual zero-shot GPT-2 + DVAE + HiFi-GAN, coqui-public-model-license)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct XttsV2Report {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_xtts_v2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<XttsV2Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // License posture: the caller MAY override via `--license <spdx>`
    // (e.g. a downstream fork re-licensed under a different NC clause);
    // the default matches the upstream Coqui Public Model License and
    // stamps `LicenseClass::NonCommercial` to make the M2-13 runtime gate
    // fail-closed on commercial-mode load. Anyone supplying a permissive
    // SPDX override should have primary-source justification captured in
    // `docs/license-audit.md` §3.1 before publish.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::NonCommercial),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = XttsV2Report::default();
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
            "vokra-convert-xtts-v2-{tag}-{}-{n}",
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
    fn xtts_v2_default_license_is_noncommercial_and_metadata_round_trips() {
        let inp = tmp_path("meta-in");
        let outp = tmp_path("meta-out");
        // Emit two tensors mirroring the XTTS-v2 topology surface: one GPT-2
        // backbone weight (F32) and one BF16 head weight (HiFi-GAN vocoder
        // weight-normalized layers frequently land here on downstream forks).
        let f32_payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut header_map = String::from("{");
        header_map.push_str(&format!(
            r#""gpt.emb.weight":{{"dtype":"F32","shape":[1,3],"data_offsets":[0,{}]}}"#,
            f32_payload.len()
        ));
        header_map.push_str(&format!(
            r#","vocoder.conv.weight":{{"dtype":"BF16","shape":[1,2],"data_offsets":[{},{}]}}"#,
            f32_payload.len(),
            f32_payload.len() + bf16_payload.len(),
        ));
        header_map.push('}');
        let mut st_bytes = Vec::new();
        st_bytes.extend_from_slice(&(header_map.len() as u64).to_le_bytes());
        st_bytes.extend_from_slice(header_map.as_bytes());
        st_bytes.extend_from_slice(&f32_payload);
        st_bytes.extend_from_slice(&bf16_payload);
        std::fs::write(&inp, &st_bytes).unwrap();

        let r = convert_xtts_v2_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 2);
        assert_eq!(r.written, 2);
        assert_eq!(r.bf16_passthrough, 1);
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
        // Weight-license stamp routes through stamp_provenance — the default
        // is NonCommercial (T4 tier fail-closed). The stamp writes the
        // class's lowercase display form ("non-commercial"), not the enum
        // variant identifier.
        assert_eq!(
            read_str("vokra.provenance.weight_license"),
            "non-commercial"
        );
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn xtts_v2_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("gpt.attn.q_proj.weight", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_xtts_v2_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn xtts_v2_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        // A downstream fork under MIT would flip the class to Permissive
        // (stamped in the class's lowercase display form).
        convert_xtts_v2_file(&inp, &outp, Some("mit")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap(),
            "permissive"
        );
        assert_eq!(
            g.get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap(),
            "mit"
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
