#![allow(clippy::doc_lazy_continuation)]
//! **Moonshine-Tiny** (`UsefulSensors/moonshine-tiny`, **MIT**):
//! safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Moonshine family entry — 27M-parameter transformer encoder-decoder ASR
//! from Useful Sensors (Jeffries et al. 2024, arXiv:2410.15608 "Moonshine:
//! Speech Recognition for Live Transcription and Voice Commands"). Distinct
//! from sibling [`crate::ModelKind::Whisper`] in two significant ways:
//! (1) **no mel front-end** — the model consumes raw 16 kHz audio directly
//! via a learned Conv1D stack (stride-tuned to yield ~40 ms hop equivalent),
//! bypassing STFT + Mel filterbank; (2) **rotary position encoding + SwiGLU**
//! activations, rather than Whisper's sinusoidal + GELU. This makes it a
//! new arch that cannot silently share the `whisper` arch tag — a Moonshine
//! checkpoint fed to a Whisper loader would misroute at the audio-input
//! boundary (raw-audio Conv1D vs mel filterbank).
//!
//! **Distinct arch tag `moonshine`** — silently sharing with sibling
//! [`crate::ModelKind::Whisper`] would misroute runtime dispatch (raw-audio
//! Conv1D encoder vs Mel encoder); FR-EX-08 (no silent CPU fallback / no
//! silent op-shape misroute) requires a distinct tag. Category `asr` shared
//! with the Whisper family.
//!
//! # License posture — MIT (**Permissive**)
//!
//! Sibling to the first-party Whisper / piper-plus / Silero / CAM++
//! Permissive posture. Default license `mit` +
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! [`crate::convert_file_licensed`] `license` when the caller legitimately
//! holds a different SPDX id (the Whisper / kokoro / vits-ja / xcodec2
//! override pattern).
//!
//! # Scale — local convert safe (~0.11 GB)
//!
//! Moonshine-Tiny ships ~110 MB (27M params). Well below the M1 iMac
//! 16 GB safe local threshold per memory
//! `[[feedback-large-models-on-vast-ai]]` (≥8 GB is the strict cutoff).
//! Local convert on M1 iMac is safe — no vast.ai handoff needed for the
//! Tiny variant. (Future family sibling `moonshine-base` at 60M params
//! ~240 MB is also local-safe.)
//!
//! # BF16 pass-through skeleton
//!
//! Mirror of sibling `musicgen_small.rs` / `hubert_large_ls960.rs` /
//! `openwakeword.rs` skeleton. Every F32 / F16 / BF16 tensor passes through
//! verbatim; non-float tensors are skipped (no quantisation applied at the
//! converter boundary — quantisation is a separate pass). Runtime binder
//! (raw-audio Conv1D + rotary + SwiGLU encoder-decoder forward + CTC or
//! greedy decode head) deferred to owner sign-off
//! (`docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "moonshine";
pub const NAME: &str = "moonshine-tiny";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "UsefulSensors/moonshine-tiny";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str =
    "UsefulSensors/moonshine-tiny (Useful Sensors 27M raw-audio transformer ASR, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoonshineTinyReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_moonshine_tiny_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineTinyReport, ConvertError> {
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

    let mut report = MoonshineTinyReport::default();
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
            "vokra-convert-moonshine-tiny-{tag}-{}-{n}",
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
    fn moonshine_tiny_f32_tensor_passes_through_and_default_license_is_permissive() {
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
        let r = convert_moonshine_tiny_file(&inp, &outp, None).unwrap();
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
        // Permissive default (mit) — distinct from the NonCommercial
        // fail-closed default of the sibling musicgen_small skeleton.
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn moonshine_tiny_bf16_tensor_passes_through_verbatim() {
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
        let r = convert_moonshine_tiny_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn moonshine_tiny_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        // Explicit override to apache-2.0 — the caller legitimately holds
        // the weight under a different SPDX id (e.g., a downstream retrain
        // under a permissive-family variant). The pass path is Whisper /
        // kokoro / vits-ja parity.
        convert_moonshine_tiny_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        let lic = g
            .get("vokra.provenance.license")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(lic, "apache-2.0");
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
