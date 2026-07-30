//! **SpeechBrain lang-ID family**: safetensors → GGUF conversion
//! (TIER 1 F wave, 2026-07-30).
//!
//! Covers two upstream variants that share ECAPA-TDNN backbone +
//! language classification head:
//!
//! - **F7** = `speechbrain/lang-id-voxlingua107-ecapa` — 107-language
//!   ID trained on VoxLingua107 (Valk & Alumäe 2021 —
//!   `arXiv:2011.12998`).
//! - **F9** = `speechbrain/lang-id-commonlanguage_ecapa` — variant
//!   trained on the CommonLanguage dataset (~45 languages).
//!
//! Both use the same ECAPA-TDNN backbone (SE-Res2Blocks + attentive
//! stat pooling → 192-d embedding → language head); the only
//! variant-carrying axis is the target vocabulary size (107 vs ~45),
//! which is a shape-derivable hparam not a topology change. Sharing
//! this one file keeps the byte-parallel pass-through logic single-
//! sourced; the caller distinguishes at dispatch time via
//! [`Variant::name`], which stamps `vokra.model.name` accordingly.
//!
//! # Provenance
//!
//! - **HF paths**:
//!   - `speechbrain/lang-id-voxlingua107-ecapa` (F7, canonical)
//!   - `speechbrain/lang-id-commonlanguage_ecapa` (F9, sibling)
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`) for both
//!   (per SpeechBrain family license `github.com/speechbrain/speechbrain/blob/develop/LICENSE`).
//! - **Category**: `classification` (language identification is a fixed
//!   N-way classifier — recorded under `vokra.model.category`).
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `ecapa_tdnn` / `clap`.

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SpeechBrain lang-ID GGUFs (shared across
/// both variants — the topology IS shared; only the target vocab
/// differs).
pub const ARCH: &str = "lang_id_ecapa";

/// Model name for the F7 (VoxLingua107) variant.
pub const NAME_VOXLINGUA107: &str = "lang-id-voxlingua107-ecapa";

/// Model name for the F9 (CommonLanguage) variant.
pub const NAME_COMMONLANGUAGE: &str = "lang-id-commonlanguage-ecapa";

pub const CATEGORY: &str = "classification";

/// Upstream HF path for F7.
pub const UPSTREAM_HF_VOXLINGUA107: &str = "speechbrain/lang-id-voxlingua107-ecapa";

/// Upstream HF path for F9.
pub const UPSTREAM_HF_COMMONLANGUAGE: &str = "speechbrain/lang-id-commonlanguage_ecapa";

pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Which upstream variant to stamp on the GGUF.
///
/// Both variants share the ECAPA-TDNN topology; only the target
/// language vocabulary differs (VoxLingua107 = 107 classes vs
/// CommonLanguage ≈ 45 classes — a shape-derivable head-width hparam,
/// not a topology change). The variant tag lets a downstream loader
/// tell them apart without a per-model dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// F7: `speechbrain/lang-id-voxlingua107-ecapa` (default).
    VoxLingua107,
    /// F9: `speechbrain/lang-id-commonlanguage_ecapa`.
    CommonLanguage,
}

impl Variant {
    /// The `vokra.model.name` value stamped for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::VoxLingua107 => NAME_VOXLINGUA107,
            Self::CommonLanguage => NAME_COMMONLANGUAGE,
        }
    }

    /// The upstream HF slug for this variant.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::VoxLingua107 => UPSTREAM_HF_VOXLINGUA107,
            Self::CommonLanguage => UPSTREAM_HF_COMMONLANGUAGE,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeechbrainLangIdReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

/// Variant-aware converter. Callers who want the F7 default use
/// [`convert_speechbrain_lang_id_file`]; the CLI dispatch routes F9
/// (`--model lang-id-commonlanguage`) here with
/// [`Variant::CommonLanguage`].
pub fn convert_speechbrain_lang_id_variant(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: Variant,
) -> Result<SpeechbrainLangIdReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    let source_note = match variant {
        Variant::VoxLingua107 => {
            "speechbrain/lang-id-voxlingua107-ecapa (ECAPA-TDNN + 107-class lang-id, apache-2.0)"
        }
        Variant::CommonLanguage => {
            "speechbrain/lang-id-commonlanguage_ecapa (ECAPA-TDNN + CommonLanguage lang-id, apache-2.0)"
        }
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(source_note),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());

    let mut report = SpeechbrainLangIdReport::default();
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

/// Default variant convenience (F7 = VoxLingua107).
pub fn convert_speechbrain_lang_id_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SpeechbrainLangIdReport, ConvertError> {
    convert_speechbrain_lang_id_variant(input, output, license, Variant::VoxLingua107)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-lang-id-{tag}-{}-{}.bin",
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

    /// Default (F7 / VoxLingua107) path stamps the VoxLingua107 name +
    /// upstream HF slug.
    #[test]
    fn default_variant_stamps_voxlingua107() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes =
            safetensors_one_bf16("embedding_model.blocks.0.tdnn.conv.weight", &[2, 3], &bf16);
        let input = scratch_path("f7-in");
        let output = scratch_path("f7-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_speechbrain_lang_id_file(&input, &output, None).expect("convert");
        assert_eq!(report.written, 1);

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_VOXLINGUA107)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_VOXLINGUA107)
        );
    }

    /// F9 / CommonLanguage explicitly stamps the CommonLanguage name +
    /// upstream HF slug — variant tag flows through
    /// [`convert_speechbrain_lang_id_variant`].
    #[test]
    fn commonlanguage_variant_stamps_own_name() {
        let bf16 = [0u8; 12];
        let input_bytes =
            safetensors_one_bf16("embedding_model.blocks.0.tdnn.conv.weight", &[2, 3], &bf16);
        let input = scratch_path("f9-in");
        let output = scratch_path("f9-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report =
            convert_speechbrain_lang_id_variant(&input, &output, None, Variant::CommonLanguage)
                .expect("convert");
        assert_eq!(report.written, 1);

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_COMMONLANGUAGE),
            "F9 stamps CommonLanguage name, not VoxLingua107"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_COMMONLANGUAGE),
            "F9 stamps CommonLanguage upstream HF slug"
        );
        // ARCH is shared — the topology is the same.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "both variants share the ARCH tag"
        );
    }
}
