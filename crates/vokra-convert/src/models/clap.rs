//! **CLAP** (LAION contrastive language-audio pretraining):
//! safetensors → GGUF conversion (TIER 1 F wave, 2026-07-30).
//!
//! Input: the upstream `laion/clap-htsat-fused` release — a
//! Contrastive Language-Audio Pretraining model with an HTSAT audio
//! encoder + text encoder trained contrastively (Wu et al. 2023 —
//! `arXiv:2211.06687`). One of the highest-download HF audio releases
//! (8.1M+). Output: a GGUF carrying every F32 / F16 / BF16 tensor
//! verbatim plus the `vokra.provenance.*` / `vokra.model.*` metadata
//! chunks a future `vokra-models::clap::*` loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `laion/clap-htsat-fused` (fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`).
//! - **Category**: `classification` (audio-text embedding — the model
//!   surface is "return an embedding compatible with the paired text
//!   encoder"; downstream users project into an N-way classification by
//!   picking a text prompt vocabulary).
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `neucodec` / `ecapa_tdnn`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! CLAP has a two-tower topology (audio encoder + text encoder + fused
//! projection); the pass-through preserves both towers so a future
//! `ClapWeights::from_gguf` can walk each side. Real-weight parity is
//! deferred to owner sign-off (loud-partial precedent — internal
//! HTSAT + text encoder forward will land behind
//! `VokraError::UnsupportedOp`).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` — first `"clap"` in the converter tree.
/// Distinct from every sibling arch tag because CLAP's two-tower
/// contrastive topology (HTSAT audio encoder + text encoder + shared
/// embedding space) is unrelated to speaker-encoder / VAD / ASR / TTS
/// families.
pub const ARCH: &str = "clap";

pub const NAME: &str = "clap-htsat-fused";
pub const CATEGORY: &str = "classification";
pub const UPSTREAM_HF: &str = "laion/clap-htsat-fused";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClapReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_clap_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ClapReport, ConvertError> {
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
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "laion/clap-htsat-fused (contrastive language-audio pretraining, \
             HTSAT audio encoder + text encoder, apache-2.0)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = ClapReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-clap-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn safetensors_two_towers(
        audio_name: &str,
        audio_bf16: &[u8],
        text_name: &str,
        text_f32: &[u8],
    ) -> Vec<u8> {
        // CLAP is two-tower: pin both an audio-encoder and a text-encoder
        // tensor in the same fixture so the round-trip proves the
        // pass-through doesn't collapse one tower.
        let audio_len = audio_bf16.len();
        let total = audio_len + text_f32.len();
        let header = format!(
            r#"{{"{audio_name}":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{audio_len}]}},"{text_name}":{{"dtype":"F32","shape":[2,3],"data_offsets":[{audio_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(audio_bf16);
        out.extend_from_slice(text_f32);
        out
    }

    #[test]
    fn both_towers_pass_through_verbatim() {
        // Audio tower (HTSAT) — BF16.
        let audio_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let audio_bf16: Vec<u8> = audio_vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Text tower — F32.
        let text_vals: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let text_bytes: Vec<u8> = text_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let input_bytes = safetensors_two_towers(
            "audio_encoder.htsat.blocks.0.attn.qkv.weight",
            &audio_bf16,
            "text_encoder.blocks.0.attn.qkv.weight",
            &text_bytes,
        );
        let input = scratch_path("two-tower-in");
        let output = scratch_path("two-tower-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_clap_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors — one per tower");
        assert_eq!(report.written, 2, "both towers survive pass-through");
        assert_eq!(report.bf16_passthrough, 1, "only audio tower is BF16");

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out).expect("parse");

        let audio_info = file
            .tensor_info("audio_encoder.htsat.blocks.0.attn.qkv.weight")
            .expect("audio tower tensor present");
        assert_eq!(audio_info.dtype, GgmlType::BF16, "BF16 audio tower");
        assert_eq!(file.tensor_bytes(audio_info), audio_bf16.as_slice());

        let text_info = file
            .tensor_info("text_encoder.blocks.0.attn.qkv.weight")
            .expect("text tower tensor present");
        assert_eq!(text_info.dtype, GgmlType::F32, "F32 text tower");
        assert_eq!(file.tensor_bytes(text_info), text_bytes.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
    }
}
