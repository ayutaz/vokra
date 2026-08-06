//! **ChatTTS** (`2Noise/ChatTTS`, **cc-by-nc-4.0**):
//! safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave D T4).
//!
//! 2Noise's conversational TTS = GPT-style autoregressive decoder over
//! discrete speech tokens, tuned on 10 000+ hours of Chinese and
//! English conversational audio, with a decoder head that maps LM
//! tokens back to waveform. Distributed on HuggingFace at
//! `huggingface.co/2Noise/ChatTTS`. Weight license is **CC-BY-NC-4.0**
//! (research-only, T4 tier — X-Codec-2 / Sortformer diar 4spk
//! precedent), so publish requires `--allow-noncommercial` and the
//! runtime M2-13 gate refuses commercial-mode load.
//!
//! # BF16 pass-through (mirror of sensevoicesmall / xtts_v2 / neucodec)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30
//! ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream HF safetensors keys verbatim**
//! (sibling to sensevoicesmall / xtts_v2 / chatterbox — the runtime
//! `vokra-models::chattts` future binder can rely on the upstream key
//! set without a rename layer).
//!
//! # No ONNX (permanent)
//!
//! The upstream release ships safetensors + torch pickle; this
//! converter accepts safetensors only (never touches ONNX,
//! FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for ChatTTS GGUFs. Distinct from every
/// sibling TTS family — silently sharing an arch tag with piper-plus
/// (VITS2), Kokoro (iSTFTNet), CosyVoice2 (FSQ + HiFTNet), Chatterbox
/// (Llama_520M + S3Gen HiFT-GAN), or xtts (GPT-2 + DVAE + HiFi-GAN)
/// would mis-route the runtime dispatch. ChatTTS's GPT-style AR
/// decoder + discrete-speech-token vocab + decoder head is a distinct
/// topology.
pub const ARCH: &str = "chattts";

/// `vokra.model.name` value written for the canonical
/// `2Noise/ChatTTS` release.
pub const NAME: &str = "chattts";

/// `vokra.model.category` value written for every ChatTTS GGUF.
pub const CATEGORY: &str = "tts";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf`. Verified against
/// `huggingface.co/2Noise/ChatTTS`.
pub const UPSTREAM_HF: &str = "2Noise/ChatTTS";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream HF card — CC-BY-NC-4.0 (research-only, non-commercial,
/// T4 tier per X-Codec-2 precedent).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / xtts_v2 / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key — the primary
/// redistribution source HF slug (parallel to the GitHub-only
/// `vokra.provenance.upstream_url` sibling key). Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a ChatTTS conversion. Mirrors the sibling
/// BF16-passthrough converters' counter shape.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter).
    pub bf16_passthrough: usize,
}

/// Converts a ChatTTS safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning a [`ChatTtsReport`].
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is [`DEFAULT_LICENSE_SPDX`] (`"cc-by-nc-4.0"`)
/// which resolves to [`LicenseClass::NonCommercial`] (T4 fail-closed).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_chattts_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ChatTtsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "2Noise/ChatTTS (GPT-style autoregressive decoder over discrete speech \
             tokens, conversational TTS ZH/EN, CC-BY-NC-4.0 — owner §3.1 sign-off \
             required, publish requires --allow-noncommercial)",
        ),
    );

    let mut report = ChatTtsReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
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

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-chattts-{tag}-{}-{}.{ext}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        let header = r#"{"gpt.layers.0.attn.wq.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_chattts_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("gpt.layers.0.attn.wq.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_fail_closed() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"gpt.embed_tokens.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"decoder.head.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);

        let input_path = scratch_path("mixed-in", "safetensors");
        let output_path = scratch_path("mixed-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_chattts_file(&input_path, &output_path, None).expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("gpt.embed_tokens.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file.tensor_info("decoder.head.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
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
        // cc-by-nc-4.0 resolves to NonCommercial (T4 fail-closed).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
            "cc-by-nc-4.0 must resolve to NonCommercial (T4 fail-closed)"
        );
    }

    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"gpt.layers.0.attn.wq.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_chattts_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 reclassifies away from the NonCommercial default"
        );
    }
}
