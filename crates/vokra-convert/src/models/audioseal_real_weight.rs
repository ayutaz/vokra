//! **AudioSeal real weight** (`facebook/audioseal`, MIT): safetensors →
//! GGUF conversion (coverage-audit-2026-08-03 Wave A permissive
//! continuation, 2026-08-04).
//!
//! Input: the upstream Meta AudioSeal release on HF (San Roman et al.
//! 2024 ICML arXiv:2401.17264, "Proactive Detection of Voice Cloning
//! with Localized Watermarking"). AudioSeal is a **paired
//! Generator + Detector** audio-watermarking system for EU AI Act
//! Article 50 compliance (2026-08-02 applies): the Generator embeds
//! a 16-bit message into a waveform via a HiFi-GAN-style residual
//! stack; the Detector recovers the message via a
//! HiFi-GAN-mirror encoder + binary classification head. This
//! converter is a **BF16 pass-through skeleton for the real weight
//! payload** — it complements the earlier M5-05 config-only scaffold
//! (honest-UNMET at the time) by admitting the real Generator +
//! Detector state-dicts through the same tensor-verbatim posture the
//! sibling BF16-passthrough converters use. Callers pre-flatten the
//! upstream torch pickles (`generator_base.pth` + `detector_base.pth`)
//! to a single safetensors offline via
//! `tools/parity/audioseal_prepare_checkpoint.py` (the DFN3 / DAC /
//! CSM pickle-bridge pattern — no pickle enters the runtime,
//! FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime watermark
//! path binds against.
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]). Verified
//!   against `github.com/facebookresearch/audioseal/blob/main/LICENSE`.
//! - Category: **watermark** (audio watermarking / steganographic
//!   marking of AI-generated content — the first-class category for
//!   EU AI Act Article 50 compliance obligations; sibling of the
//!   deferred `synthid_embed` / `c2pa_manifest` op families).
//! - Notes: replaces the M5-05 config-only scaffold with a real
//!   weight-loading path; the runtime binder for Generator +
//!   Detector remains gated on M5-05 T04 ADR ratification per
//!   `docs/license-audit.md §3.1` (owner critical path).
//!
//! # BF16 pass-through (mirror of sensevoicesmall / neucodec /
//! # ecapa_tdnn / speaker_3d)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream AudioSeal state-dict keys
//! verbatim** (`generator.*` / `detector.*` per the AudioSeal class
//! layout at
//! `github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py`).
//! Real-weight parity binding to a future `vokra-models::watermark::
//! audioseal::{generator, detector}` runtime module (native HiFi-GAN
//! residual + 16-bit message embedding forward, and the mirror encoder
//! + binary classification head) is deferred to owner sign-off per
//! `docs/license-audit.md §3.1` (M5-05 T04 ADR ratification is the
//! critical-path unlocker per CLAUDE.md M5-05 note).
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "audioseal_real_weight"` is intentionally
//! distinct from every sibling watermark / codec arch tag. There is
//! no existing base `audioseal` arch tag today (M5-05 was
//! config-only); the explicit `_real_weight` suffix future-proofs
//! against a potential later `audioseal_config` scaffold split so
//! runtime dispatch always maps to the actual weight-loading path.
//!
//! # No ONNX (permanent)
//!
//! The upstream AudioSeal release ships PyTorch pickle files; this
//! converter **never** touches ONNX (FR-LD-05).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native Generator +
//! Detector forward is a follow-up wave gated on the M5-05 T04 ADR
//! ratification (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for AudioSeal real-weight GGUFs.
/// Intentionally explicit `_real_weight` suffix future-proofs against
/// a potential later config-scaffold split so runtime dispatch always
/// maps to the actual weight-loading path (M5-05 was config-only).
pub const ARCH: &str = "audioseal_real_weight";

/// `vokra.model.name` value written for the canonical
/// `facebook/audioseal` release.
pub const NAME: &str = "audioseal_real_weight";

/// `vokra.model.category` value written for every AudioSeal real-
/// weight GGUF. First-class `watermark` category for EU AI Act
/// Article 50 compliance obligations (2026-08-02 applies).
pub const CATEGORY: &str = "watermark";

/// Upstream HF repository slug (`org/name`).
pub const UPSTREAM_HF: &str = "facebook/audioseal";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/facebookresearch/audioseal/blob/main/LICENSE` (standard
/// MIT).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key. Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an AudioSeal real-weight conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AudiosealRealWeightReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16.
    pub bf16_passthrough: usize,
}

/// Converts an AudioSeal real-weight safetensors checkpoint at `input`
/// (pre-flattened from the paired upstream `generator_base.pth` +
/// `detector_base.pth` torch pickles by
/// `tools/parity/audioseal_prepare_checkpoint.py`) into a Vokra-native
/// GGUF at `output`, returning an [`AudiosealRealWeightReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license. The
/// default is `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_audioseal_real_weight_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AudiosealRealWeightReport, ConvertError> {
    // Load the whole checkpoint into memory — ~20 MB combined
    // (Generator + Detector, well below any streaming threshold).
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
            "facebook/audioseal (paired Generator + Detector 16-bit-message audio watermark for \
             EU AI Act Article 50 compliance, San Roman et al. 2024 ICML arXiv:2401.17264, MIT — \
             runtime binder gated on M5-05 T04 ADR ratification)",
        ),
    );

    let mut report = AudiosealRealWeightReport::default();
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
            "vokra-audioseal-real-weight-{tag}-{}-{}.{ext}",
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

    /// Pins the BF16 pass-through end-to-end.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12);
        // AudioSeal Generator 16-bit message embedding weight — the
        // upstream `facebook/audioseal` state-dict key convention
        // preserved verbatim through the
        // `audioseal_prepare_checkpoint.py` bridge.
        let header = r#"{"generator.embedding.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_audioseal_real_weight_file(&input_path, &output_path, None)
            .expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("generator.embedding.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    /// Pins F32 and F16 pass-through. MIT default → Permissive.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"generator.residual.0.conv.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"detector.classifier.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_audioseal_real_weight_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("generator.residual.0.conv.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("detector.classifier.bias")
            .expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

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
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"generator.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report =
            convert_audioseal_real_weight_file(&input_path, &output_path, Some("apache-2.0"))
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
        );
    }
}
