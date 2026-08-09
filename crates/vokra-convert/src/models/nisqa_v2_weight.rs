//! **NISQA v2** (`gabrielmittag/NISQA`, **cc-by-nc-sa-4.0**):
//! safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave D T4).
//!
//! Non-intrusive speech quality assessment model (Mittag et al. 2021
//! "NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech
//! Quality Prediction with Crowdsourced Datasets", arXiv:2104.09494).
//! Predicts P.808-style MOS + 4 dimensions (Noisiness / Coloration /
//! Discontinuity / Loudness) from a single-channel audio stream.
//! Distributed via GitHub only (`github.com/gabrielmittag/NISQA`) — no
//! HF mirror, so provenance rides `vokra.provenance.upstream_url` (the
//! NKF-AEC / RNNoise / NSNet2 GitHub-native precedent). Weight license
//! is **cc-by-nc-sa-4.0** (research-only + share-alike, T4 tier), so
//! publish requires `--allow-noncommercial` AND downstream artefacts
//! inherit the SA cascade.
//!
//! # BF16 pass-through (mirror of nkf_aec / sensevoicesmall / dnsmos)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30
//! ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream torch state-dict keys
//! verbatim** — the `.tar` bundle is pre-flattened offline to
//! safetensors by
//! `tools/parity/nisqa_v2_weight_prepare_checkpoint.py` and this
//! converter accepts safetensors only (the DFN3 / DAC / DNSMOS
//! precedent — pickles never enter the runtime, FR-LD-05).
//!
//! # No ONNX (permanent)
//!
//! The upstream release ships torch pickle only; this converter
//! **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for NISQA v2 weight GGUFs. Distinct from
/// every sibling eval / MOS predictor family — silently sharing an
/// arch tag with `dnsmos` (Microsoft) or `utmos` (SaruLab) would
/// mis-route the runtime dispatch (each is a distinct topology).
pub const ARCH: &str = "nisqa_v2_weight";

/// `vokra.model.name` value written for the canonical
/// `gabrielmittag/NISQA` release.
pub const NAME: &str = "nisqa_v2_weight";

/// `vokra.model.category` value written for every NISQA v2 GGUF.
/// Sibling of DNSMOS / UTMOS22-strong (`eval` family — non-intrusive
/// MOS predictors).
pub const CATEGORY: &str = "eval";

/// Primary redistribution source (author's GitHub repository — there is
/// no HF mirror). Written under [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/gabrielmittag/NISQA";

/// Default upstream weight licence (SPDX). Verified against the
/// upstream repo — CC-BY-NC-SA-4.0 (research-only + share-alike,
/// T4 tier).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for GitHub-only releases (parallel to the HF-hosted
/// `vokra.provenance.upstream_hf` key). Same convention as NKF-AEC /
/// RNNoise / NSNet2 / DNSMOS.
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a NISQA v2 weight conversion. Mirrors the sibling
/// BF16-passthrough converters' counter shape.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NisqaV2WeightReport {
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

/// Converts a NISQA v2 safetensors checkpoint at `input`
/// (pre-flattened from the upstream torch `.tar` pickle by
/// `tools/parity/nisqa_v2_weight_prepare_checkpoint.py`) into a
/// Vokra-native GGUF at `output`, returning a [`NisqaV2WeightReport`].
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX`
/// (`"cc-by-nc-sa-4.0"`) which resolves to
/// [`LicenseClass::NonCommercialShareAlike`] (T4 fail-closed).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_nisqa_v2_weight_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NisqaV2WeightReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    let effective_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let effective_class = LicenseClass::from_license_str(effective_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "github.com/gabrielmittag/NISQA (non-intrusive speech quality \
             assessment CNN + self-attention, Mittag et al. 2021 arXiv:2104.09494, \
             CC-BY-NC-SA-4.0 — owner §3.1 sign-off required, publish requires \
             --allow-noncommercial + SA cascade obligation)",
        ),
    );

    let mut report = NisqaV2WeightReport::default();
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
            "vokra-nisqa-v2-weight-{tag}-{}-{}.{ext}",
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
        let header =
            r#"{"mos_head.linear.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_nisqa_v2_weight_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of nkf_aec / sensevoicesmall)"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("mos_head.linear.weight")
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
            r#"{{"cnn.stack.0.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"attn.qkv.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_nisqa_v2_weight_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file.tensor_info("cnn.stack.0.weight").expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file.tensor_info("attn.qkv.bias").expect("F16 tensor");
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
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        // cc-by-nc-sa-4.0 resolves to NonCommercialShareAlike
        // (fail-closed T4 tier + SA cascade obligation).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "cc-by-nc-sa-4.0 must resolve to NonCommercialShareAlike (T4 + SA cascade)"
        );
    }

    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"mos_head.linear.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_nisqa_v2_weight_file(&input_path, &output_path, Some("apache-2.0"))
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
            "apache-2.0 reclassifies away from the NonCommercialShareAlike default"
        );
    }
}
