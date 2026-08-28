//! **X-vector speaker embedding** (SpeechBrain): safetensors → GGUF
//! conversion (TIER 1 F wave, 2026-07-30).
//!
//! Input: a safetensors prepared from the upstream
//! `speechbrain/spkrec-xvect-voxceleb` `embedding_model.ckpt` by
//! `tools/parity/xvector_prepare_checkpoint.py` — a TDNN-based speaker
//! embedding network (Snyder et al. 2018 —
//! `arXiv:1710.10467`) trained on VoxCeleb. An alternative to
//! ECAPA-TDNN (`ecapa_tdnn.rs`) and CAM++ (`campplus.rs`) — same
//! functional surface (fbank → speaker embedding) but distinct
//! topology (plain TDNN stack + statistics pooling, no SE-Res2Blocks,
//! no D-TDNN). Output: a GGUF carrying one of the two released, exact tensor
//! manifests (32-tensor embedding-only or 46-tensor combined classifier
//! checkpoint) plus the provenance, model, and pinned frontend metadata
//! consumed by `vokra-models::xvector`.
//!
//! # Provenance
//!
//! - **HF path**: `speechbrain/spkrec-xvect-voxceleb` (fetched
//!   2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`) — per the
//!   SpeechBrain family license.
//! - **Category**: `speaker` (recorded under `vokra.model.category`) —
//!   TDNN speaker encoder.
//!
//! # Float pass-through
//!
//! F32 / F16 / BF16 inference tensors pass through byte-for-byte.  The five
//! integer BatchNorm training counters in the raw `.ckpt` are not runtime
//! weights and are removed by the preparation tool before this converter.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` — distinct from `ecapa_tdnn` / `campplus` /
/// `wespeaker`; X-vector uses a plain TDNN stack + statistics pooling
/// (no SE-Res2Blocks / no D-TDNN / no ResNet34), so silently sharing a
/// speaker-encoder arch tag would misroute the runtime dispatch.
pub const ARCH: &str = "xvector";

pub const NAME: &str = "spkrec-xvect-voxceleb";
pub const CATEGORY: &str = "speaker";
pub const UPSTREAM_HF: &str = "speechbrain/spkrec-xvect-voxceleb";
pub const UPSTREAM_REVISION: &str = "56895a2df401be4150a159f3a1c653f00051d477";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SAMPLE_RATE: &str = "vokra.xvector.sample_rate";
const KEY_N_MELS: &str = "vokra.xvector.n_mels";
const KEY_N_FFT: &str = "vokra.xvector.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.xvector.win_length";
const KEY_HOP_LENGTH: &str = "vokra.xvector.hop_length";
const KEY_EMBED_DIM: &str = "vokra.xvector.embed_dim";
const KEY_TDNN_BLOCKS: &str = "vokra.xvector.tdnn_blocks";
const KEY_BN_EPS: &str = "vokra.xvector.bn_eps";
const KEY_STATS_STD_EPS: &str = "vokra.xvector.stats_std_eps";
const KEY_FRONTEND: &str = "vokra.xvector.frontend";
const KEY_PADDING: &str = "vokra.xvector.padding";
const KEY_ARTIFACT_LAYOUT: &str = "vokra.xvector.artifact_layout";

const EMBEDDING_TENSORS: usize = 32;
const COMBINED_TENSORS: usize = 46;
const EMBED_DIM: u64 = 512;
const STATS_DIM: u64 = 3_000;
const CONV_BLOCKS: [(usize, u64, u64, u64); 5] = [
    (0, 24, 512, 5),
    (3, 512, 512, 3),
    (6, 512, 512, 3),
    (9, 512, 512, 1),
    (12, 512, 1_500, 1),
];
const NORM_BLOCKS: [usize; 5] = [2, 5, 8, 11, 14];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XVectorLayout {
    EmbeddingOnlyBare,
    CombinedPrefixed,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct XVectorReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_xvector_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<XVectorReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let layout = validate_manifest(&st)?;

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
        Some("speechbrain/spkrec-xvect-voxceleb (TDNN X-vector speaker encoder, apache-2.0)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_N_MELS, 24);
    b.add_u32(KEY_N_FFT, 400);
    b.add_u32(KEY_WIN_LENGTH, 400);
    b.add_u32(KEY_HOP_LENGTH, 160);
    b.add_u32(KEY_EMBED_DIM, EMBED_DIM as u32);
    b.add_u32(KEY_TDNN_BLOCKS, CONV_BLOCKS.len() as u32);
    b.add_f32(KEY_BN_EPS, 1.0e-5);
    b.add_f32(KEY_STATS_STD_EPS, 1.0e-5);
    b.add_string(KEY_FRONTEND, "speechbrain-fbank-v1");
    b.add_string(KEY_PADDING, "reflect-same");
    b.add_string(
        KEY_ARTIFACT_LAYOUT,
        match layout {
            XVectorLayout::EmbeddingOnlyBare => "embedding-only-bare-v1",
            XVectorLayout::CombinedPrefixed => "combined-prefixed-v1",
        },
    );

    let mut report = XVectorReport::default();
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

fn validate_manifest(st: &SafetensorsFile) -> Result<XVectorLayout, ConvertError> {
    let bare = st.tensor_info("blocks.0.conv.weight").is_some();
    let prefixed = st
        .tensor_info("embedding_model.blocks.0.conv.weight")
        .is_some();
    let layout = match (bare, prefixed, st.tensors().len()) {
        (true, false, EMBEDDING_TENSORS) => XVectorLayout::EmbeddingOnlyBare,
        (false, true, COMBINED_TENSORS) => XVectorLayout::CombinedPrefixed,
        _ => {
            return Err(ConvertError::Parse(format!(
                "xvector: unsupported tensor layout: count={}, bare_stem={bare}, prefixed_stem={prefixed}; expected exactly 32 bare embedding tensors or 46 combined tensors",
                st.tensors().len()
            )));
        }
    };
    let stem = match layout {
        XVectorLayout::EmbeddingOnlyBare => "",
        XVectorLayout::CombinedPrefixed => "embedding_model.",
    };
    for (name, expected) in expected_embedding_manifest(stem) {
        check_shape(st, &name, &expected)?;
    }
    if layout == XVectorLayout::CombinedPrefixed {
        for prefix in ["classifier.norm.norm", "classifier.DNN.block_0.norm.norm"] {
            for suffix in ["weight", "bias", "running_mean", "running_var"] {
                check_shape(st, &format!("{prefix}.{suffix}"), &[EMBED_DIM])?;
            }
        }
        check_shape(
            st,
            "classifier.DNN.block_0.linear.w.weight",
            &[EMBED_DIM, EMBED_DIM],
        )?;
        check_shape(st, "classifier.DNN.block_0.linear.w.bias", &[EMBED_DIM])?;
        check_shape(st, "classifier.out.w.weight", &[7_205, EMBED_DIM])?;
        check_shape(st, "classifier.out.w.bias", &[7_205])?;
        check_shape(st, "mean_var_norm_emb.glob_mean", &[EMBED_DIM])?;
        check_shape(st, "mean_var_norm_emb.glob_std", &[1])?;
    }
    for tensor in st.tensors() {
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(ConvertError::Parse(format!(
                "xvector: tensor `{}` uses unsupported dtype {:?}; every manifest tensor must be F32, F16, or BF16",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(layout)
}

fn expected_embedding_manifest(stem: &str) -> Vec<(String, Vec<u64>)> {
    let mut manifest = Vec::with_capacity(EMBEDDING_TENSORS);
    for ((block, input_channels, output_channels, kernel), norm_block) in
        CONV_BLOCKS.into_iter().zip(NORM_BLOCKS)
    {
        manifest.push((
            format!("{stem}blocks.{block}.conv.weight"),
            vec![output_channels, input_channels, kernel],
        ));
        manifest.push((
            format!("{stem}blocks.{block}.conv.bias"),
            vec![output_channels],
        ));
        for suffix in ["weight", "bias", "running_mean", "running_var"] {
            manifest.push((
                format!("{stem}blocks.{norm_block}.norm.{suffix}"),
                vec![output_channels],
            ));
        }
    }
    manifest.push((
        format!("{stem}blocks.16.w.weight"),
        vec![EMBED_DIM, STATS_DIM],
    ));
    manifest.push((format!("{stem}blocks.16.w.bias"), vec![EMBED_DIM]));
    debug_assert_eq!(manifest.len(), EMBEDDING_TENSORS);
    manifest
}

fn check_shape(st: &SafetensorsFile, name: &str, expected: &[u64]) -> Result<(), ConvertError> {
    let info = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!("xvector: required tensor `{name}` is missing"))
    })?;
    if info.shape != expected {
        return Err(ConvertError::Parse(format!(
            "xvector: tensor `{name}` has shape {:?}, expected {expected:?}",
            info.shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-xvector-{tag}-{}-{}.bin",
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
    fn partial_checkpoint_is_rejected_fail_closed() {
        let input_bytes =
            safetensors_one_bf16("embedding_model.blocks.0.conv.weight", &[1, 1, 1], &[0, 0]);
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let error = convert_xvector_file(&input, &output, None).unwrap_err();
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        assert!(error.to_string().contains("expected exactly 32"));
    }

    #[test]
    fn exact_embedding_manifest_has_32_unique_names() {
        let manifest = expected_embedding_manifest("embedding_model.");
        assert_eq!(manifest.len(), EMBEDDING_TENSORS);
        let names = manifest
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), EMBEDDING_TENSORS);
        assert_eq!(UPSTREAM_REVISION.len(), 40);
    }
}
