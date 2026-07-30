//! **SepFormer** (SpeechBrain sepformer family, apache-2.0):
//! safetensors → GGUF conversion for the Transformer-based dual-path
//! source-separation / enhancement family.
//!
//! # Family (3 variants share this converter)
//!
//! - **`speechbrain/sepformer-wsj02mix`** — 2-speaker source
//!   separation trained on WSJ0-2mix.
//!   [`SepformerVariant::Wsj02mix`]. `vokra.model.category = "separation"`.
//! - **`speechbrain/sepformer-wham16k-enhancement`** — single-speaker
//!   speech enhancement (WHAM! 16 kHz).
//!   [`SepformerVariant::Wham16kEnhancement`]. `vokra.model.category = "enhancement"`.
//! - **`speechbrain/sepformer-whamr16k`** — joint dereverb + denoise
//!   (WHAMR! 16 kHz).
//!   [`SepformerVariant::Whamr16k`]. `vokra.model.category = "enhancement"`.
//!
//! All three variants share the same SepFormer architecture (encoder +
//! dual-path Transformer masker + decoder — Subakan et al. 2021 /
//! Chen et al. 2022 WHAMR extension); only the training data + head
//! count differ. One `sepformer.rs` converter therefore covers all
//! three — the caller passes a [`SepformerVariant`] and the emitted
//! GGUF's `vokra.model.name`, `vokra.provenance.upstream_hf`,
//! `vokra.model.category`, and `vokra.sepformer.variant` stamps reflect
//! the specific release, while `vokra.model.arch = "sepformer"` is
//! shared across all three (silently sharing would misroute a
//! downstream loader if the family ever diverged in tensor topology —
//! today they do not, but the shared arch tag + explicit variant tag
//! is the honest posture).
//!
//! # Provenance
//!
//! - **License (SPDX)**: `apache-2.0` for all three variants per HF
//!   model-card `cardData.license` (SpeechBrain ships Apache-2.0
//!   end-to-end — `github.com/speechbrain/speechbrain/blob/develop/LICENSE`).
//!   Class = `Permissive`.
//! - **Attribution**: none required by license.
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2 / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 (`GgmlType::BF16`); runtime widens
//! BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! Real-weight parity + a native `SepFormer::from_gguf` forward path
//! are deferred to owner sign-off (`docs/license-audit.md` §3.1) — this
//! converter provides the byte-parallel GGUF surface only. The internal
//! dual-path Transformer topology is intentionally NOT re-implemented
//! on this pass (`loud-partial` sibling wave).
//!
//! # No ONNX (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints (safetensors); this converter
//! **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for every SepFormer variant — the topology is
/// shared across `wsj02mix` / `wham16k-enhancement` / `whamr16k`.
pub const ARCH: &str = "sepformer";

/// Default upstream weight license (SPDX) — apache-2.0 for every
/// SepFormer variant per HF model-card `cardData.license`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.sepformer.variant`: `"wsj02mix"` / `"wham16k-enhancement"` /
/// `"whamr16k"`. Consumers pick a specific separation / enhancement
/// head without parsing free-text `vokra.model.name`.
pub const KEY_SEPFORMER_VARIANT: &str = "vokra.sepformer.variant";

/// Which SepFormer release the caller is converting.
///
/// All three variants share the [`ARCH`] tag `sepformer`; the category,
/// upstream HF slug, and variant tag differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SepformerVariant {
    /// `speechbrain/sepformer-wsj02mix`: 2-speaker source separation
    /// on WSJ0-2mix. Category = `separation`,
    /// `vokra.sepformer.variant = "wsj02mix"`.
    Wsj02mix,
    /// `speechbrain/sepformer-wham16k-enhancement`: single-speaker
    /// speech enhancement (WHAM! 16 kHz). Category = `enhancement`,
    /// `vokra.sepformer.variant = "wham16k-enhancement"`.
    Wham16kEnhancement,
    /// `speechbrain/sepformer-whamr16k`: joint dereverb + denoise
    /// (WHAMR! 16 kHz). Category = `enhancement`,
    /// `vokra.sepformer.variant = "whamr16k"`.
    Whamr16k,
}

impl SepformerVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wsj02mix => "sepformer-wsj02mix",
            Self::Wham16kEnhancement => "sepformer-wham16k-enhancement",
            Self::Whamr16k => "sepformer-whamr16k",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`).
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Wsj02mix => "speechbrain/sepformer-wsj02mix",
            Self::Wham16kEnhancement => "speechbrain/sepformer-wham16k-enhancement",
            Self::Whamr16k => "speechbrain/sepformer-whamr16k",
        }
    }

    /// The `vokra.sepformer.variant` tag.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Wsj02mix => "wsj02mix",
            Self::Wham16kEnhancement => "wham16k-enhancement",
            Self::Whamr16k => "whamr16k",
        }
    }

    /// The `vokra.model.category` value. Wsj02mix is a pure
    /// **source-separation** task (2 speakers out of 1 mixture);
    /// the other two are single-output **enhancement** tasks
    /// (dereverb / denoise), so a downstream that speaks either
    /// API can pick the correct load path from this alone.
    pub const fn category(self) -> &'static str {
        match self {
            Self::Wsj02mix => "separation",
            Self::Wham16kEnhancement | Self::Whamr16k => "enhancement",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp.
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Wsj02mix => {
                "speechbrain/sepformer-wsj02mix (SepFormer 2-speaker separation, WSJ0-2mix, apache-2.0)"
            }
            Self::Wham16kEnhancement => {
                "speechbrain/sepformer-wham16k-enhancement (SepFormer speech enhancement, WHAM! 16 kHz, apache-2.0)"
            }
            Self::Whamr16k => {
                "speechbrain/sepformer-whamr16k (SepFormer dereverb + denoise, WHAMR! 16 kHz, apache-2.0)"
            }
        }
    }
}

/// Outcome of a SepFormer conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SepformerReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm.
    pub bf16_passthrough: usize,
    /// Which SepFormer variant was written.
    pub variant: Option<SepformerVariant>,
}

/// File-based SepFormer converter.
///
/// Reads `input` (upstream `speechbrain/sepformer-*` safetensors),
/// writes a Vokra GGUF to `output` carrying every F32 / F16 / BF16
/// tensor verbatim under its upstream name + the `vokra.model.*` +
/// `vokra.provenance.*` metadata chunks + the
/// `vokra.sepformer.variant` tag.
///
/// `variant` selects which SepFormer release the input came from —
/// the GGUF's `vokra.model.name`, `vokra.provenance.upstream_hf`,
/// `vokra.model.category`, and `vokra.sepformer.variant` stamps
/// reflect it.
///
/// `license` optionally overrides the default `apache-2.0` provenance
/// stamp (same override pattern as
/// `wespeaker::convert_wespeaker_file`).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_sepformer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: SepformerVariant,
) -> Result<SepformerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, variant.category());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_SEPFORMER_VARIANT, variant.tag());

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );

    let mut report = SepformerReport {
        variant: Some(variant),
        ..SepformerReport::default()
    };
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

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
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

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sepformer-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// BF16 round-trip + Wsj02mix-specific stamps (category =
    /// `separation`, not the sibling variants' `enhancement`).
    #[test]
    fn wsj02mix_bf16_round_trip_and_separation_category() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[2, 3], &bf16);
        let input = write_temp("wsj02mix-in", &input_bytes);
        let output = write_temp("wsj02mix-out", &[]);

        let report = convert_sepformer_file(&input, &output, None, SepformerVariant::Wsj02mix)
            .expect("convert sepformer-wsj02mix");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Wsj02mix));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("masker.dpt.0.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        // Wsj02mix's category is `separation` — distinct from the
        // enhancement variants.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation")
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("wsj02mix")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-wsj02mix")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-wsj02mix")
        );
        // Arch is shared across every variant.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Wham16k-enhancement variant carries the `enhancement` category
    /// and its own model.name / upstream_hf.
    #[test]
    fn wham16k_enhancement_carries_enhancement_category() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("encoder.conv.weight", &[1, 2], &bf16);
        let input = write_temp("wham16k-in", &input_bytes);
        let output = write_temp("wham16k-out", &[]);

        convert_sepformer_file(&input, &output, None, SepformerVariant::Wham16kEnhancement)
            .expect("convert sepformer-wham16k-enhancement");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("enhancement"),
            "Wham16kEnhancement must NOT emit `separation`"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("wham16k-enhancement")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-wham16k-enhancement")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-wham16k-enhancement")
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Whamr16k variant carries `enhancement` category (dereverb +
    /// denoise = single-output enhancement task).
    #[test]
    fn whamr16k_carries_enhancement_category() {
        let values: [f32; 2] = [1.0, -1.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("decoder.conv.weight", &[1, 2], &bf16);
        let input = write_temp("whamr16k-in", &input_bytes);
        let output = write_temp("whamr16k-out", &[]);

        convert_sepformer_file(&input, &output, None, SepformerVariant::Whamr16k)
            .expect("convert sepformer-whamr16k");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("enhancement")
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("whamr16k")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-whamr16k")
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 mixed pass-through with BF16 counter at 0.
    #[test]
    fn f32_and_f16_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 4] = [0x3C00, 0xC000, 0xB800, 0x4200];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"a.w":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_len}]}},"b.w":{{"dtype":"F16","shape":[2,2],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        let input = write_temp("mixed-in", &input_bytes);
        let output = write_temp("mixed-out", &[]);

        let report =
            convert_sepformer_file(&input, &output, None, SepformerVariant::Wham16kEnhancement)
                .expect("convert mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override lands on the artifact — apache-2.0 default,
    /// override to mit stays Permissive.
    #[test]
    fn license_override_reaches_the_artifact() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.w", &[1, 2], &bf16);
        let input = write_temp("license-in", &input_bytes);
        let output = write_temp("license-out", &[]);

        convert_sepformer_file(&input, &output, Some("mit"), SepformerVariant::Wsj02mix)
            .expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
