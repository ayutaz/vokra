//! **TEN-VAD** (`TEN-framework/ten-vad`, Apache-2.0 + BSD-3-Clause front-end):
//! safetensors → GGUF conversion (coverage-audit-2026-08-03 Wave A permissive
//! continuation, 2026-08-04).
//!
//! Input: the upstream `TEN-framework/ten-vad` release on GitHub — a
//! compact voice-activity-detection model (~306 KB ONNX bundle, small
//! LSTM/GRU backbone + LPCNet-derived DSP front-end) targeting real-time
//! edge inference. Positioned as a **~5.5x lighter** alternative to
//! Silero VAD v5 (upstream claim) for latency-constrained deployments
//! at 16 kHz. The upstream release is distributed as ONNX only —
//! `tools/parity/ten_vad_prepare_checkpoint.py` bridges the ONNX graph
//! to safetensors offline (NSNet2 / DNSMOS ONNX-bridge precedent — no
//! ONNX ever enters the runtime, NFR-DS-02 zero-dep + FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime VAD path binds
//! against.
//!
//! # License
//!
//! - SPDX: **Apache-2.0** ([`vokra_core::LicenseClass::Permissive`]).
//!   The upstream repo LICENSE is Apache-2.0.
//! - Category: **vad-kws** (voice-activity detection — sibling of the
//!   existing `SileroVad` FR-LD-06 1:1 subgraph and `FsmnVad`
//!   first-class op posture, positioned as a third alternative topology
//!   under the shared `vad-kws` umbrella covering VAD + KWS families).
//! - Notes: the LPCNet-derived DSP front-end bundled in the upstream
//!   distribution is BSD-3-Clause; NOTICE attribution for the LPCNet
//!   copyright is required when redistributing runtime binaries that
//!   embed the front-end.
//!
//! # BF16 pass-through (mirror of nkf_aec / facebook_denoiser /
//! # torchaudio_squim / sensevoicesmall)
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm on the same
//! match arm — no convert-time widening. BF16 is emitted as GGUF type
//! 30 ([`GgmlType::BF16`]); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`TenVadReport::bf16_passthrough`] guards against a silent
//! widen / downcast regression.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream ONNX-inspection-derived state
//! keys verbatim** (`encoder.*` / `decoder.*` / `feature_extractor.*`
//! per the NSNet2 ONNX-bridge convention). Real-weight parity binding
//! to a future `vokra-models::ten_vad` module (native LSTM + LPCNet
//! front-end forward) is deferred to owner sign-off per
//! `docs/license-audit.md §3.1`.
//!
//! # Provenance key choice: `vokra.provenance.upstream_url`
//!
//! TEN-VAD's primary redistribution source is the **GitHub repository**
//! at `github.com/TEN-framework/ten-vad` — there is no HF mirror. The
//! converter therefore stamps [`KEY_PROVENANCE_UPSTREAM_URL`] following
//! the parallel key naming convention the Wave A tickets established
//! for non-HF sources (`nkf-aec` / `torchaudio-squim` /
//! `facebook-denoiser`).
//!
//! # No ONNX (permanent) in the runtime
//!
//! The upstream TEN-VAD release ships an ONNX file; the offline bridge
//! `tools/parity/ten_vad_prepare_checkpoint.py` flattens the graph
//! tensors to safetensors so the runtime never touches the ONNX
//! (FR-LD-05, NFR-DS-02).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native LSTM + LPCNet-
//! inspired feature-extractor forward is a follow-up wave, deferred to
//! owner sign-off (see `docs/license-audit.md` §3.1).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for TEN-VAD GGUFs. Intentionally distinct
/// from every sibling VAD / KWS arch tag (`silero-vad` /
/// `fsmn_vad` / `openwakeword` / `openwakeword_op`) — TEN-VAD's LSTM +
/// LPCNet-inspired front-end is a distinct topology, so silently
/// sharing would mis-route the runtime dispatch.
pub const ARCH: &str = "ten_vad";

/// `vokra.model.name` value written for the canonical
/// `TEN-framework/ten-vad` release.
pub const NAME: &str = "ten_vad";

/// `vokra.model.category` value written for every TEN-VAD GGUF.
/// Sibling of `silero-vad` / `fsmn_vad` / `openwakeword` (`vad-kws`
/// umbrella covering VAD + KWS families).
pub const CATEGORY: &str = "vad-kws";

/// Primary redistribution source (author's GitHub repository — no HF
/// mirror). Written under [`KEY_PROVENANCE_UPSTREAM_URL`].
pub const UPSTREAM_URL: &str = "github.com/TEN-framework/ten-vad";

/// Default upstream weight licence (SPDX). Verified against
/// `github.com/TEN-framework/ten-vad/blob/main/LICENSE` (Apache-2.0
/// for the main project; the LPCNet-derived DSP front-end bundled in
/// the upstream distribution is BSD-3-Clause — NOTICE attribution
/// required when redistributing binaries embedding the front-end).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// nkf_aec / sensevoicesmall / funcodec convention (not yet centralized
/// in `vokra-core::gguf::chunks`).
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_url` — the primary redistribution source
/// URL for models whose canonical release is NOT on the Hugging Face
/// hub. Parallel to `vokra.provenance.upstream_hf` (the HF-hosted
/// sibling key); the Wave A tickets established the split so the
/// model-card generator can distinguish "there is an HF mirror" from
/// "the source is a raw URL". Local per the same convention as
/// [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

/// Outcome of a TEN-VAD conversion.
///
/// Mirrors the sibling BF16-passthrough converters' counter shape
/// ([`super::nkf_aec::NkfAecReport`],
/// [`super::torchaudio_squim::TorchaudioSquimReport`]) — the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TenVadReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for parity with the sibling converters).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. A silent
    /// widen / downcast regression would surface as this counter
    /// drifting away from the input BF16 count.
    pub bf16_passthrough: usize,
}

/// Converts a TEN-VAD safetensors checkpoint at `input` (pre-flattened
/// from the upstream ONNX by
/// `tools/parity/ten_vad_prepare_checkpoint.py`) into a Vokra-native
/// GGUF at `output`, returning a [`TenVadReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// ONNX-inspection-derived key; the `vokra.model.*` (arch / name /
/// category) and `vokra.provenance.*` (weight_license / license /
/// model_id / source / upstream_url) chunks are stamped for the
/// runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`, `Permissive`).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_ten_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<TenVadReport, ConvertError> {
    // Load the whole checkpoint into memory — the TEN-VAD bundle is
    // ~306 KB, well below the streaming-mandated Moshi 14 GiB tier,
    // so the simple `std::fs::read` posture applies trivially.
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
            "github.com/TEN-framework/ten-vad (compact ~306 KB LSTM/GRU VAD + \
             LPCNet-derived DSP front-end, Apache-2.0 main + BSD-3-Clause front-end — \
             NOTICE attribution required for LPCNet copyright when redistributing \
             binaries embedding the front-end)",
        ),
    );

    let mut report = TenVadReport::default();
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
            "vokra-ten-vad-{tag}-{}-{}.{ext}",
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

    /// Pins the BF16 pass-through end-to-end. Mirrors
    /// `nkf_aec::tests::bf16_tensor_passes_through_verbatim`.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload = bf16_bytes(&values);
        assert_eq!(payload.len(), 12, "6 elements × 2 bytes BF16");
        // TEN-VAD LSTM feature-extractor Conv1D weight — the ONNX-
        // inspection-derived state key convention preserved verbatim
        // through the `ten_vad_prepare_checkpoint.py` bridge.
        let header =
            r#"{"encoder.conv.0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&payload);

        let input_path = scratch_path("bf16-in", "safetensors");
        let output_path = scratch_path("bf16-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_ten_vad_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1, "one BF16 tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of nkf_aec / torchaudio_squim)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("encoder.conv.0.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// Pins that F32 and F16 tensors both ride the pass-through arm in
    /// the same conversion, and that the BF16 counter stays at 0.
    /// Also asserts the arch / name / category / provenance stamps
    /// land through the default (apache-2.0 → Permissive).
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        assert_eq!(f16_bytes.len(), 4);

        let header = format!(
            r#"{{"encoder.lstm.weight_ih_l0":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"decoder.linear.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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
            convert_ten_vad_file(&input_path, &output_path, None).expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2, "F32 and F16 must both pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 / F16 must NOT increment the BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("encoder.lstm.weight_ih_l0")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("decoder.linear.bias").expect("F16 tensor");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(f16_info.dimensions, vec![2]);
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
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL)
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
            "apache-2.0 must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"encoder.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);

        let input_path = scratch_path("lic-in", "safetensors");
        let output_path = scratch_path("lic-out", "gguf");
        std::fs::write(&input_path, &input_bytes).expect("write input");
        let _in_guard = TempFileGuard(input_path.clone());
        let _out_guard = TempFileGuard(output_path.clone());

        let report = convert_ten_vad_file(&input_path, &output_path, Some("mit"))
            .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override replaces the raw SPDX string"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "mit stays Permissive (same class as apache-2.0 default)",
        );
    }
}
