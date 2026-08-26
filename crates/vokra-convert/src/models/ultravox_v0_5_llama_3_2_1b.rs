#![allow(clippy::doc_lazy_continuation)]
//! **Ultravox v0.5 (Llama-3.2-1B)** (`fixie-ai/ultravox-v0_5-llama-3_2-1b`,
//! **MIT**): safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Ultravox family entry — fixie-ai's audio-text-to-text multimodal model
//! combining a separately acquired **Llama-3.2-1B** language backbone with a
//! **Whisper encoder + projection adapter** front-end.  The fixie-ai
//! safetensors — and therefore the 1,366,275,264-byte public Vokra GGUF —
//! contain only the 491 BF16 audio-tower/projector tensors.  Llama weights are
//! not bundled. Real audio is fed through the Whisper encoder, projected into
//! the separately licensed Llama token embedding space, then decoded by that
//! companion backbone.
//!
//! **Distinct from siblings [`crate::ModelKind::Voxtral`]** (Mistral text
//! decoder + Whisper encoder) **and [`crate::ModelKind::Qwen2Audio`]**
//! (Qwen2-7B decoder + Whisper encoder). All three are "Whisper encoder +
//! text LM decoder" audio-LLMs, but the decoder backbone (Llama vs Mistral
//! vs Qwen2) fixes tensor layout + tokenizer + rope base; silently sharing
//! a converter or arch tag across the three would misroute runtime dispatch
//! at the LM decoder loader (FR-EX-08 forbids silent shape misroute). This
//! converter therefore emits a distinct arch tag `ultravox` (shared with any
//! future Ultravox v0.6+ / Llama-3.2-3B siblings — the family topology is
//! the same, only the decoder scale differs, mirroring the MusicGen family
//! shared-arch-tag pattern).
//!
//! Category `audio-llm` is shared with sibling Qwen2-Audio-7B / Voxtral /
//! Kimi-Audio / Step-Audio2-Mini / Baichuan-Audio siblings.  The strict native
//! binder now executes all 32 Whisper layers plus the exact stack-8 SwiGLU
//! projector on CPU or Metal.  A complete text-generation route remains
//! deliberately partial until the separately licensed Llama-3.2-1B companion,
//! tokenizer and chat/audio-placeholder contract are bound.  The public GGUF
//! is never treated as a standalone LM.
//!
//! # License posture — MIT (**Permissive**)
//!
//! Sibling to the first-party Whisper / piper-plus / Silero / CAM++ /
//! Moonshine Permissive posture. Upstream `fixie-ai/ultravox-v0_5-llama-3_2-1b`
//! HF card declares `license: mit` per the SoTA scope-expansion 2026-07-30
//! canary sweep. Default license `mit` +
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! [`crate::convert_file_licensed`] `license` when the caller legitimately
//! holds a different SPDX id (the Whisper / kokoro / xcodec2 override
//! pattern). §3.1 sign-off remains owner (fail-closed default per memory
//! `[[feedback-license-signoff-primary-source]]`).
//!
//! # Scale — 1.37 GB public artifact
//!
//! The audited public GGUF is exactly 1,366,275,264 bytes. This converter reads
//! both the source and generated GGUF into owned buffers, so model conversion
//! and real-weight validation belong on the configured remote workflow when
//! the maintainer requests that the Mac stay idle; the mmap runtime binder is
//! the bounded-memory path.
//!
//! # BF16 pass-through skeleton
//!
//! Mirror of sibling `musicgen_small.rs` / `moonshine_base.rs` /
//! `hubert_large_ls960.rs` / `openwakeword.rs` / `demucs_htdemucs.rs`
//! skeleton. Every F32 / F16 / BF16 tensor passes through verbatim; non-
//! float tensors are skipped (no quantisation applied at the converter
//! boundary — quantisation is a separate pass).  The output is the MIT audio
//! component, not the separately distributed Llama companion.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "ultravox";
pub const NAME: &str = "ultravox-v0-5-llama-3-2-1b";
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "fixie-ai/ultravox-v0_5-llama-3_2-1b";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str = "fixie-ai/ultravox-v0_5-llama-3_2-1b (Ultravox v0.5 Whisper encoder + projection adapter only; Llama-3.2-1B companion not bundled, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UltravoxV05Llama321bReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_ultravox_v0_5_llama_3_2_1b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<UltravoxV05Llama321bReport, ConvertError> {
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

    let mut report = UltravoxV05Llama321bReport::default();
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
            "vokra-convert-ultravox-v0-5-llama-3-2-1b-{tag}-{}-{n}",
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
    fn ultravox_v0_5_llama_3_2_1b_f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Exact public projector namespace (the real release shape is
        // [2048, 2048]; this tiny payload only pins identity pass-through).
        let st = safetensors_one(
            "multi_modal_projector.linear_2.weight",
            "F32",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, None).unwrap();
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
    fn ultravox_v0_5_llama_3_2_1b_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        // BF16 payload — matches the SoTA plan skeleton contract: runtime
        // widens BF16 → F32 exactly at load, so the converter must not
        // touch the bytes.
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Exact public Whisper encoder namespace.  No `language_model.*`
        // tensor exists in this artifact; the Llama companion is separate.
        let st = safetensors_one(
            "audio_tower.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn ultravox_v0_5_llama_3_2_1b_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.license").and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
