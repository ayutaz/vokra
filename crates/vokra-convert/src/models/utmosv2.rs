//! **UTMOSv2** (`sarulab-speech/UTMOSv2`, MIT): safetensors → GGUF
//! conversion (coverage-audit-2026-08-03 Wave A permissive continuation,
//! 2026-08-04).
//!
//! Input: the upstream `sarulab-speech/UTMOSv2` release — the direct
//! successor to UTMOS22-strong (Baba et al. arXiv:2409.09305, "UTMOSv2:
//! UTokyo-SaruLab MOS Prediction System for VoiceMOS Challenge 2024",
//! VoiceMOS Challenge 2024 SoTA). Reference-free MOS-TTS quality
//! estimator = wav2vec2-large SSL encoder + listener / domain
//! conditioning + improved Regressor head. Distinct from the existing
//! `Utmos` (UTMOS22-strong = wav2vec2-base) — the arch is a strict
//! upgrade path so the ModelKind is separate (existing UTMOS22-strong
//! stays landed for reproducibility and the M5-15 UTMOS un-defer
//! judgement path). Callers pre-flatten the upstream torch pickle to
//! safetensors offline via
//! `tools/parity/utmosv2_prepare_checkpoint.py` (the UTMOS22-strong
//! pickle-bridge pattern — no pickle enters the runtime, FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime eval path binds
//! against.
//!
//! # License
//!
//! - SPDX: **MIT** ([`vokra_core::LicenseClass::Permissive`]).
//!   Verified against `github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE`.
//! - Category: **eval** (reference-free MOS-TTS quality estimator —
//!   sibling of the existing `utmos` / `dnsmos` / `nisqa_v2_weight`
//!   / `torchaudio_squim` families; UTMOSv2 is the upgrade path for
//!   the existing `Utmos` variant).
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
//! GGUF tensor names are the **upstream `sarulab-speech/UTMOSv2`
//! state-dict keys verbatim** (the CSM / Kokoro / CosyVoice2 / Chatterbox
//! contract). Real-weight parity binding to a future
//! `vokra-eval::utmosv2` runtime module (native wav2vec2-large forward
//! + listener/domain conditioning + Regressor head) is deferred to
//! owner sign-off per `docs/license-audit.md §3.1`.
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "utmosv2"` is intentionally distinct from
//! sibling eval-family arch tags (`utmos` = UTMOS22-strong wav2vec2-
//! base; `dnsmos`; `nisqa_v2_weight`; `torchaudio_squim`). Silently
//! sharing an arch tag with UTMOS22-strong would mis-route the runtime
//! dispatch (the SSL encoder axis is wav2vec2-large vs base and the
//! Regressor head layout differs).
//!
//! # No ONNX (permanent)
//!
//! The upstream UTMOSv2 release ships PyTorch pickle files; this
//! converter **never** touches ONNX (FR-LD-05).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native SSL encoder +
//! listener/domain conditioning + Regressor head forward is a
//! follow-up wave, deferred to owner sign-off (see
//! `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for UTMOSv2 GGUFs. Intentionally distinct
/// from every sibling eval-family arch tag (`utmos` / `dnsmos` /
/// `nisqa_v2_weight` / `torchaudio_squim`) — UTMOSv2's wav2vec2-large
/// SSL encoder + listener/domain conditioning + improved Regressor
/// head is a distinct topology from UTMOS22-strong's wav2vec2-base
/// backbone, so silently sharing an arch tag would mis-route the
/// runtime dispatch.
pub const ARCH: &str = "utmosv2";

/// `vokra.model.name` value written for the canonical
/// `sarulab-speech/UTMOSv2` release.
pub const NAME: &str = "utmosv2";

/// `vokra.model.category` value written for every UTMOSv2 GGUF.
/// Sibling of `utmos` / `dnsmos` / `nisqa_v2_weight` /
/// `torchaudio_squim` (`eval` family).
pub const CATEGORY: &str = "eval";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the artifact
/// back to its serving location without parsing the free-text
/// `vokra.provenance.source`.
pub const UPSTREAM_HF: &str = "sarulab-speech/UTMOSv2";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE` (standard
/// MIT).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention (not yet centralized
/// in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key — the primary
/// redistribution source HF slug for models mirrored on the Hugging
/// Face hub. Parallel to `vokra.provenance.upstream_url` (the raw-URL
/// sibling key for GitHub-only releases). Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a UTMOSv2 conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// (`super::sensevoicesmall::SenseVoiceSmallReport`) — the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Utmosv2Report {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a UTMOSv2 safetensors checkpoint at `input` (pre-flattened
/// from the upstream torch pickle by
/// `tools/parity/utmosv2_prepare_checkpoint.py`) into a Vokra-native
/// GGUF at `output`, returning a [`Utmosv2Report`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"mit"`,
/// `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_utmosv2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Utmosv2Report, ConvertError> {
    // Load the whole checkpoint into memory — the UTMOSv2 release is
    // ~500 MB (well below the streaming-mandated Moshi 14 GiB tier).
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
            "sarulab-speech/UTMOSv2 (wav2vec2-large SSL encoder + listener/domain \
             conditioning + Regressor head, reference-free MOS-TTS quality estimator, \
             Baba et al. arXiv:2409.09305 = VoiceMOS Challenge 2024 SoTA, MIT)",
        ),
    );

    let mut report = Utmosv2Report::default();
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
            "vokra-utmosv2-{tag}-{}-{}.{ext}",
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
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        // UTMOSv2 Regressor head linear weight — the upstream
        // `sarulab-speech/UTMOSv2` state-dict key convention preserved
        // verbatim through the `utmosv2_prepare_checkpoint.py` bridge.
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

        let report = convert_utmosv2_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
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

    /// Pins that F32 and F16 tensors both ride the pass-through arm.
    /// MIT default resolves to Permissive.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"ssl_encoder.encoder.layers.0.norm.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"listener_head.embedding.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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
            convert_utmosv2_file(&input_path, &output_path, None).expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("ssl_encoder.encoder.layers.0.norm.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("listener_head.embedding.bias")
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
            r#"{"ssl_encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_utmosv2_file(&input_path, &output_path, Some("apache-2.0"))
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
