#![allow(clippy::doc_lazy_continuation)]
//! **Demucs (HT-Demucs)** (`facebook/demucs`, **MIT**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Demucs family entry — Meta's hybrid transformer source-separation model
//! (Rouard et al. 2023, "Hybrid Transformers for Music Source Separation",
//! arXiv:2211.08553). **htdemucs** is the hybrid variant that combines a
//! U-Net waveform branch with a spectrogram branch joined by cross-domain
//! self-attention. The default upstream release performs **4-source
//! separation** (drums / bass / other / vocals — MUSDB18 stem taxonomy).
//!
//! **Distinct from siblings [`crate::ModelKind::SepformerWsj02mix`] et al.
//! (`speechbrain/sepformer-*`) and [`crate::ModelKind::TigerSeparator`]**
//! (`JusperLee/TIGER-DnR`). SepFormer is a dual-path Transformer on the
//! waveform-only encoder + masker + decoder pipeline (Subakan et al. 2021)
//! targeting speech mixtures. TIGER is a time-frequency dual-branch model
//! for dialog/effects/music separation. HT-Demucs is a hybrid
//! waveform+spectrogram U-Net + transformer, targeting **music** source
//! separation. Silently sharing arch tags across these three would misroute
//! runtime dispatch at the separator masker head (different output
//! branching, different domain of the internal representation) — FR-EX-08
//! (no silent op-shape misroute) requires the distinct `demucs` arch tag.
//!
//! Category `separation` shared with the SepFormer / TIGER separator
//! siblings. The runtime binder (hybrid U-Net waveform branch +
//! spectrogram branch + cross-domain self-attention + `separate_masks`
//! audio op emit) is deferred to owner sign-off
//! (`docs/license-audit.md` §3.1) — this converter emits the BF16
//! pass-through skeleton only.
//!
//! # License posture — MIT (**Permissive**)
//!
//! Sibling to the first-party Whisper / piper-plus / Silero / CAM++ /
//! Moonshine Permissive posture. Upstream `github.com/facebookresearch/demucs`
//! LICENSE ships MIT; the HF mirror `facebook/demucs` was HTTP 401 during
//! the 2026-08-02 residual walk (rate-limit on the unauthenticated API),
//! so the SPDX id is anchored on the upstream GitHub `LICENSE` file as the
//! primary source per memory `[[feedback-license-signoff-primary-source]]`
//! (fail-closed default: never fill §3.1 sign-off from CC judgement alone
//! — that stays owner's).  Default license `mit` +
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! [`crate::convert_file_licensed`] `license` when the caller legitimately
//! holds a different SPDX id (the Whisper / kokoro / xcodec2 override
//! pattern).
//!
//! # Scale — local convert safe (~0.50 GB)
//!
//! HT-Demucs ships ~500 MB per HF cardData / upstream release manifest.
//! Well below the M1 iMac 16 GB safe local threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (≥8 GB is the strict cutoff for
//! vast.ai handoff, 4 GB the "重いモデル" soft cutoff per owner
//! 2026-08-01 directive). Local convert on M1 iMac is safe — no vast.ai
//! handoff needed.
//!
//! # BF16 pass-through skeleton
//!
//! Mirror of sibling `musicgen_small.rs` / `moonshine_base.rs` /
//! `hubert_large_ls960.rs` / `openwakeword.rs` skeleton. Every F32 / F16 /
//! BF16 tensor passes through verbatim; non-float tensors are skipped (no
//! quantisation applied at the converter boundary — quantisation is a
//! separate pass). Runtime binder (hybrid U-Net + spectrogram branch +
//! cross-domain attention + `separate_masks` op) deferred to owner
//! sign-off (`docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "demucs";
pub const NAME: &str = "demucs-htdemucs";
pub const CATEGORY: &str = "separation";
pub const UPSTREAM_HF: &str = "facebook/demucs";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str =
    "facebook/demucs (Meta hybrid transformer Demucs, htdemucs 4-source separation, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DemucsHtdemucsReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_demucs_htdemucs_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DemucsHtdemucsReport, ConvertError> {
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

    let mut report = DemucsHtdemucsReport::default();
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
            "vokra-convert-demucs-htdemucs-{tag}-{}-{n}",
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
    fn demucs_htdemucs_f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Demucs-specific: hybrid U-Net waveform branch encoder weight —
        // the tensor name is representative of the htdemucs family.
        let st = safetensors_one("encoder.0.conv.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_demucs_htdemucs_file(&inp, &outp, None).unwrap();
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
        // Permissive default (mit) — sibling to Moonshine / Whisper /
        // piper-plus / Silero / CAM++ first-party posture.
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn demucs_htdemucs_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        // BF16 payload — matches the SoTA plan skeleton contract: runtime
        // widens BF16 → F32 exactly at load, so the converter must not
        // touch the bytes.
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Cross-domain self-attention QKV in the transformer bottleneck —
        // representative of the hybrid transformer branch.
        let st = safetensors_one(
            "crosstransformer.layers.0.attn.qkv",
            "BF16",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_demucs_htdemucs_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn demucs_htdemucs_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("mask.weight", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        // Override with a distinct valid SPDX to prove the caller-supplied
        // license reaches the provenance stamp (mirrors the sibling
        // musicgen_small / moonshine_base override test).
        convert_demucs_htdemucs_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        let spdx = g
            .get("vokra.provenance.license")
            .and_then(|v| v.as_str())
            .expect("license stamp missing");
        assert_eq!(spdx, "apache-2.0");
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
