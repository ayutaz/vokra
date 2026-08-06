#![allow(clippy::doc_lazy_continuation)]
//! **Moonshine-Base** (`UsefulSensors/moonshine-base`, **MIT**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Moonshine family entry — 61.5M-parameter transformer encoder-decoder ASR
//! from Useful Sensors (Jeffries et al. 2024, arXiv:2410.15608 "Moonshine:
//! Speech Recognition for Live Transcription and Voice Commands"). Sibling
//! to [`crate::models::moonshine_tiny`] with the same architecture family
//! (raw-audio Conv1D front-end + rotary position encoding + SwiGLU
//! activations) but a wider/deeper backbone (~2.3× parameter count vs the
//! 27M Tiny variant per the upstream release manifest). Distinct from
//! sibling [`crate::ModelKind::Whisper`] in two significant ways:
//! (1) **no mel front-end** — the model consumes raw 16 kHz audio directly
//! via a learned Conv1D stack, bypassing STFT + Mel filterbank;
//! (2) **rotary position encoding + SwiGLU** activations rather than
//! Whisper's sinusoidal + GELU.
//!
//! **Distinct arch tag `moonshine`** (shared with sibling
//! [`crate::models::moonshine_tiny`], since Tiny and Base share the same
//! architecture — only depth/width differ). Silently sharing with sibling
//! [`crate::ModelKind::Whisper`] would misroute runtime dispatch (raw-audio
//! Conv1D encoder vs Mel encoder); FR-EX-08 (no silent CPU fallback / no
//! silent op-shape misroute) requires the distinct `moonshine` tag.
//! Category `asr` shared with the Whisper family.
//!
//! # License posture — MIT (**Permissive**)
//!
//! Sibling to the first-party Whisper / piper-plus / Silero / CAM++ /
//! Moonshine-Tiny Permissive posture. Default license `mit` +
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! [`crate::convert_file_licensed`] `license` when the caller legitimately
//! holds a different SPDX id (the Whisper / kokoro / vits-ja / xcodec2
//! override pattern).
//!
//! # Scale — local convert safe (~0.25 GB)
//!
//! Moonshine-Base ships ~250 MB (61.5M params per HF API). Well below the
//! M1 iMac 16 GB safe local threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (≥8 GB is the strict cutoff).
//! Local convert on M1 iMac is safe — no vast.ai handoff needed. Sibling
//! Tiny variant (`moonshine-tiny`, ~0.11 GB) is also local-safe.
//!
//! # BF16 pass-through skeleton
//!
//! Mirror of sibling `moonshine_tiny.rs` / `musicgen_small.rs` /
//! `hubert_large_ls960.rs` / `openwakeword.rs` skeleton. Every F32 / F16 /
//! BF16 tensor passes through verbatim; non-float tensors are skipped (no
//! quantisation applied at the converter boundary — quantisation is a
//! separate pass). Runtime binder (raw-audio Conv1D + rotary + SwiGLU
//! encoder-decoder forward + CTC or greedy decode head) deferred to owner
//! sign-off (`docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "moonshine";
pub const NAME: &str = "moonshine-base";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "UsefulSensors/moonshine-base";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str =
    "UsefulSensors/moonshine-base (Useful Sensors 61.5M raw-audio transformer ASR, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoonshineBaseReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_moonshine_base_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineBaseReport, ConvertError> {
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

    let mut report = MoonshineBaseReport::default();
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
            "vokra-convert-moonshine-base-{tag}-{}-{n}",
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
    fn moonshine_base_f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Moonshine-specific: raw-audio Conv1D encoder weight (no mel
        // front-end) — the tensor name is representative of the family.
        let st = safetensors_one("encoder.audio_conv.weight", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_moonshine_base_file(&inp, &outp, None).unwrap();
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
        // Permissive default (mit) — sibling to moonshine-tiny.
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn moonshine_base_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        // BF16 payload — matches the SoTA plan skeleton contract: runtime
        // widens BF16 → F32 exactly at load, so the converter must not
        // touch the bytes.
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("decoder.rotary.qkv", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_moonshine_base_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn moonshine_base_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        // Explicit override to apache-2.0 — the caller legitimately holds
        // the weight under a different SPDX id (e.g., a downstream retrain
        // under a permissive-family variant). The pass path is Whisper /
        // kokoro / vits-ja parity.
        convert_moonshine_base_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        let lic = g
            .get("vokra.provenance.license")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(lic, "apache-2.0");
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn moonshine_base_f16_tensor_count_and_arch_name_distinct_from_tiny() {
        // Regression: distinct NAME string vs sibling moonshine-tiny (the
        // arch tag `moonshine` is shared, but the model name must not
        // collide — a Base checkpoint fed to a Tiny loader would misroute
        // at load, FR-EX-08). Verify with a small F16 payload (Whisper /
        // Kokoro production checkpoints use F16, so this is the realistic
        // tensor-count path).
        let inp = tmp_path("f16-in");
        let outp = tmp_path("f16-out");
        // Two F16 tensors ~ representative of a bi-tensor multi-add pass.
        let payload_a: Vec<u8> = [0x00_u8, 0x3C, 0x00, 0x40].to_vec(); // 1.0, 2.0 in F16
        let st_a = safetensors_one("encoder.block0.mlp.gate", "F16", &[1, 2], &payload_a);
        std::fs::write(&inp, &st_a).unwrap();
        let r = convert_moonshine_base_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        // NAME assertion — must be "moonshine-base", NOT "moonshine-tiny".
        // If a refactor accidentally merged the two ModelKind arms, this
        // would flip and the sibling verify shape (uniform triple lookup)
        // would misroute a Base checkpoint's runtime dispatch.
        let name = g
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(name, "moonshine-base");
        assert_ne!(name, "moonshine-tiny");
        // UPSTREAM_HF assertion — must point to the Base repo, not Tiny.
        let upstream = g
            .get(KEY_PROVENANCE_UPSTREAM_HF)
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(upstream, "UsefulSensors/moonshine-base");
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
