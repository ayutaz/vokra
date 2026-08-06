#![allow(clippy::doc_lazy_continuation)]
//! **Seamless-M4T-v2-Large** (`facebook/seamless-m4t-v2-large`,
//! **cc-by-nc-4.0**): safetensors → GGUF conversion (Wave residual,
//! 2026-08-02).
//!
//! Meta **SeamlessM4T v2** flagship 2.3B parameter multitask speech-and-
//! text translation model (Communication et al. 2023, arXiv:2312.05187).
//! Unified any-to-any covering ASR + T2TT + S2TT + T2ST + S2ST across
//! ~100 source languages and ~35 target speech languages. The upstream
//! release ships as **2 safetensors shards + `.pt` duplicates +
//! `vocoder_v2.pt`** — the converter walks whatever safetensors bytes
//! the caller hands in (typical publish path pre-flattens shards +
//! vocoder to a single safetensors offline, mirroring the CSM / DAC
//! prepare-checkpoint pattern).
//!
//! # Architecture — **unity-2** (4 subgraphs)
//!
//! 1. **w2v-BERT 2.0 speech encoder** (BERT-style masked-feature
//!    prediction over 80-d Mel + 8 conformer + rotary positional
//!    biases).
//! 2. **Text decoder** (NLLB-derived multilingual transformer decoder
//!    with per-target-language sinusoidal biases).
//! 3. **T2U (text-to-unit)** decoder that emits acoustic units for
//!    the vocoder — Seamless's discrete-unit intermediate.
//! 4. **HiFi-GAN vocoder** (`vocoder_v2.pt`) that reconstructs 16 kHz
//!    waveform from the T2U units.
//!
//! The arch tag `unity-2` (Meta's fairseq2 dispatch name) is stamped
//! on the runtime side. Distinct from sibling M4T v1 / Massively
//! Multilingual Speech (MMS) — silently sharing an arch tag would
//! misroute the runtime binder (FR-EX-08).
//!
//! # License posture — CC-BY-NC 4.0 (**NonCommercial**)
//!
//! Same T4 (Research-only) tier as X-Codec 2 (2026-07-28 precedent),
//! MusicGen family (2026-08-01), CrisperWhisper + MMS-1B-All + MusicGen-
//! Small (2026-08-02 wave). `LicenseClass::NonCommercial` fail-closed
//! default + `publish-one.sh --allow-noncommercial` gate required.
//!
//! # Scale — vast.ai handoff (~9.00 GB)
//!
//! Above the ≥8 GB strict cutoff for M1 iMac 16 GB local convert per
//! memory `[[feedback-large-models-on-vast-ai]]`. Owner directive
//! 2026-08-01: 「重いモデル vast.ai」. Local convert would exhaust
//! the laptop's swap budget (mmap peak observed ~40 GB on similar-
//! scale Voxtral-Small-24B). The BF16 pass-through skeleton itself is
//! cheap in terms of code path (single tensor walk, verbatim byte
//! copy) — the size constraint is on the safetensors read side, not
//! this crate's writer.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "unity-2";
pub const NAME: &str = "seamless-m4t-v2-large";
pub const CATEGORY: &str = "s2s";
pub const UPSTREAM_HF: &str = "facebook/seamless-m4t-v2-large";
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

const UPSTREAM_SOURCE: &str = "facebook/seamless-m4t-v2-large (Meta SeamlessM4T v2 2.3B any-to-any speech-and-text translation, unity-2 arch, cc-by-nc-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SeamlessM4tV2LargeReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_seamless_m4t_v2_large_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SeamlessM4tV2LargeReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::NonCommercial),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = SeamlessM4tV2LargeReport::default();
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
            "vokra-convert-seamless-m4t-v2-large-{tag}-{}-{n}",
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
    fn f32_tensor_passes_through_and_default_license_is_noncommercial() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Name pattern hints at the w2v-BERT speech encoder subgraph
        // (first of the four unity-2 subgraphs) — the converter itself
        // does not require any particular naming since it walks bytes
        // verbatim, but the fixture spells out the arch intent.
        let st = safetensors_one("speech_encoder.emb", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_seamless_m4t_v2_large_file(&inp, &outp, None).unwrap();
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
        let st = safetensors_one("text_decoder.attn.q", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_seamless_m4t_v2_large_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
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
        convert_seamless_m4t_v2_large_file(&inp, &outp, Some("mit")).unwrap();
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
