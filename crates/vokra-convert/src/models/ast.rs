//! **AST (Audio Spectrogram Transformer)**: safetensors → GGUF
//! conversion (TIER 1 F wave, 2026-07-30).
//!
//! Input: the upstream `MIT/ast-finetuned-audioset-10-10-0.4593`
//! release — Gong et al. 2021 AST (`arXiv:2104.01778`), a ViT-style
//! transformer applied to the log-mel spectrogram then fine-tuned on
//! AudioSet (527 classes). Reference source =
//! `github.com/YuanGongND/ast` (BSD-3-Clause). Output: a GGUF carrying
//! every F32 / F16 / BF16 tensor verbatim plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future `vokra-models::ast::*`
//! loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `MIT/ast-finetuned-audioset-10-10-0.4593` (note: `MIT`
//!   is the ORGANIZATION on HuggingFace, not the SPDX license — the
//!   ORG published under BSD-3-Clause; fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - **SPDX**: `bsd-3-clause` (`LicenseClass::Permissive` — bsd token
//!   matches [`LicenseClass::from_license_str`]).
//! - **Category**: `classification` (recorded under
//!   `vokra.model.category`) — 527-class AudioSet classifier.
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `neucodec` / `ecapa_tdnn` / `clap`.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufMetadataValue, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for AST GGUFs. Distinct from every sibling arch
/// tag because AST is a ViT-style transformer over spectrograms — no
/// existing sibling shares the topology.
pub const ARCH: &str = "ast";

pub const NAME: &str = "ast-finetuned-audioset";
pub const CATEGORY: &str = "classification";
pub const UPSTREAM_HF: &str = "MIT/ast-finetuned-audioset-10-10-0.4593";
pub const DEFAULT_LICENSE_SPDX: &str = "bsd-3-clause";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PREFIX: &str = "vokra.ast.";

const AST_U32_AXES: [(&str, u32); 13] = [
    ("hidden_size", 768),
    ("num_hidden_layers", 12),
    ("num_attention_heads", 12),
    ("intermediate_size", 3_072),
    ("patch_size", 16),
    ("frequency_stride", 10),
    ("time_stride", 10),
    ("num_mel_bins", 128),
    ("max_length", 1_024),
    ("num_labels", 527),
    ("num_prefix_tokens", 2),
    ("sample_rate", 16_000),
    ("layer_norm_eps_scaled_1e12", 1),
];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AstReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_ast_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AstReport, ConvertError> {
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
            "MIT/ast-finetuned-audioset-10-10-0.4593 (Audio Spectrogram Transformer, \
             ViT over spectrogram, 527-class AudioSet fine-tune, bsd-3-clause)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    for (suffix, value) in AST_U32_AXES {
        b.add_u32(&format!("{KEY_PREFIX}{suffix}"), value);
    }
    b.add_bool("vokra.ast.qkv_bias", true);
    b.add_string("vokra.ast.hidden_act", "gelu");
    b.add_string("vokra.ast.window_type", "hanning");
    b.add_u32("vokra.ast.frame_length", 400);
    b.add_u32("vokra.ast.frame_shift", 160);
    b.add_u32("vokra.ast.low_freq_hz", 20);
    b.add_bool("vokra.ast.subtract_mean", false);
    b.add_metadata(
        "vokra.ast.normalize_mean",
        GgufMetadataValue::F64(-4.267_739_3),
    );
    b.add_metadata(
        "vokra.ast.normalize_std",
        GgufMetadataValue::F64(4.568_997_4),
    );

    let mut report = AstReport::default();
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
            "vokra-ast-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Realistic AST tensor name (ViT patch embedding).
        let input_bytes = safetensors_one_bf16(
            "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            &[2, 3],
            &bf16,
        );
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_ast_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info(
                "audio_spectrogram_transformer.encoder.layer.0.attention.attention.query.weight",
            )
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "bsd-3-clause resolves to Permissive"
        );
        assert_eq!(
            file.get("vokra.ast.max_length").and_then(|v| v.as_u64()),
            Some(1_024)
        );
        assert_eq!(
            file.get("vokra.ast.window_type").and_then(|v| v.as_str()),
            Some("hanning")
        );
    }
}
