//! **openWakeWord op wiring** (`dscripka/openWakeWord`, Apache-2.0
//! code): safetensors → GGUF conversion (coverage-audit-2026-08-03
//! Wave A permissive continuation, 2026-08-04).
//!
//! Input: user-provided openWakeWord model checkpoints. This converter
//! is deliberately a **runtime-op wiring companion** to the existing
//! `Openwakeword` ModelKind (2026-08-02 Wave residual, custom-KWS
//! MLP/CNN over precomputed melspec). The `_op` suffix signals that
//! the model kind primarily exists so the first-class `kws` op family
//! (CLAUDE.md audio-dialect §Streaming / VAD / KWS, `FR-OP kws`,
//! Porcupine-compatible) has a distinct runtime-dispatch anchor that
//! is decoupled from the base `openwakeword` converter's arch tag —
//! user-provided weights (either official CC-BY-NC-SA-4.0 downloads
//! the user obtains under their own compliance judgement OR
//! self-trained Apache-2.0 weights) route through this op-wiring path
//! and reach the runtime `KwsSession::from_gguf` binder without
//! silently masquerading as the base official-checkpoint converter.
//! Callers pre-flatten the upstream ONNX / TFLite to safetensors
//! offline via `tools/parity/openwakeword_op_prepare_checkpoint.py`
//! (the NSNet2 / TEN-VAD ONNX-bridge precedent — no ONNX enters the
//! runtime, NFR-DS-02 zero-dep + FR-LD-05).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*`
//! and `vokra.provenance.*` metadata chunks the runtime `kws` op binds
//! against.
//!
//! # License
//!
//! - SPDX default: **Apache-2.0** ([`vokra_core::LicenseClass::Permissive`])
//!   — the Apache-2.0 code license of the upstream openWakeWord
//!   project. Official weights on the release page are
//!   CC-BY-NC-SA-4.0, so a caller who has downloaded them must
//!   override at the CLI boundary (`--license cc-by-nc-sa-4.0`); the
//!   fail-closed disposition then flips to NonCommercialShareAlike
//!   and publish gate refuses without `--allow-noncommercial`.
//! - Category: **vad-kws** (keyword-spotting / wake-word — sibling of
//!   `silero-vad` / `fsmn_vad` / `ten_vad` under the shared `vad-kws`
//!   umbrella covering VAD + KWS families; distinct from the base
//!   `openwakeword` converter's arch tag).
//! - Notes: **Vokra does not redistribute openWakeWord official
//!   weights** — the upstream repo's release-page CC-BY-NC-SA-4.0
//!   term is not compatible with Vokra's default commercial-mode
//!   redistribution policy. The `_op` runtime-wiring path is for
//!   user-provided weights only; no §3.1 sign-off is required for the
//!   op-wiring converter itself because Vokra does not publish
//!   op-wiring artefacts.
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
//! GGUF tensor names are the **upstream openWakeWord Keras / ONNX-
//! inspection-derived keys verbatim** (`melspec_model.*` /
//! `embedding_model.*` / `classifier.*` per the openWakeWord model
//! layout: precomputed melspec → embedding CNN → wake-word classifier
//! MLP). Real-weight parity binding to a future
//! `vokra-models::kws::openwakeword_op` runtime module is deferred to
//! owner sign-off per `docs/license-audit.md §3.1` (op-wiring runtime
//! binder is scoped for the same wave).
//!
//! # Arch tag distinctness
//!
//! `vokra.model.arch = "openwakeword_op"` is intentionally distinct
//! from the sibling `openwakeword` (base official-checkpoint
//! converter, 2026-08-02 Wave residual). The `_op` variant is the
//! runtime-op-wiring anchor that user-provided weights route through
//! — silently sharing an arch tag with the base ModelKind would
//! blur the op-wiring vs published-artefact boundary and hide the
//! distinct license-override contract from the runtime dispatch.
//!
//! # No ONNX (permanent) in the runtime
//!
//! The upstream openWakeWord release ships ONNX + TFLite; the offline
//! bridge `tools/parity/openwakeword_op_prepare_checkpoint.py`
//! flattens the graph tensors to safetensors so the runtime never
//! touches the ONNX (FR-LD-05, NFR-DS-02).
//!
//! # Wiring status
//!
//! This is the TDD skeleton (BF16 / F16 / F32 pass-through plus
//! provenance / category stamps). The runtime native KWS op binder
//! is a follow-up wave.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` value for openWakeWord op-wiring GGUFs.
/// Intentionally distinct from the sibling base `openwakeword` arch
/// tag (2026-08-02 Wave residual) — the `_op` variant is the runtime-
/// op-wiring anchor, decoupled from the base converter's arch tag so
/// the runtime dispatch sees the two as different topologies with
/// different license-override contracts.
pub const ARCH: &str = "openwakeword_op";

/// `vokra.model.name` value written for the canonical
/// `dscripka/openWakeWord` op-wiring release.
pub const NAME: &str = "openwakeword_op";

/// `vokra.model.category` value written for every openWakeWord op
/// GGUF. Sibling of `silero-vad` / `fsmn_vad` / `ten_vad` under the
/// shared `vad-kws` umbrella covering VAD + KWS families.
pub const CATEGORY: &str = "vad-kws";

/// Upstream HF repository slug (`org/name`) — canonical HF mirror of
/// the openWakeWord family. Note: Vokra does NOT redistribute the
/// upstream official CC-BY-NC-SA-4.0 weights; this slug is recorded
/// as provenance only.
pub const UPSTREAM_HF: &str = "dscripka/openWakeWord";

/// Default upstream code licence (SPDX) — Apache-2.0 code. Official
/// weights are CC-BY-NC-SA-4.0, but Vokra does not redistribute
/// them, and callers who supply their own Apache-2.0 self-trained
/// weights can keep the Permissive default; callers who redistribute
/// the CC-BY-NC-SA-4.0 official weights must override to
/// `--license cc-by-nc-sa-4.0` which flips the fail-closed publish
/// gate to NonCommercialShareAlike.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// convention the sibling BF16-passthrough converters use applies).

/// `vokra.model.category` metadata key. Local per the established
/// sensevoicesmall / nkf_aec / funcodec convention.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// `vokra.provenance.upstream_hf` metadata key. Local per the same
/// convention as [`KEY_MODEL_CATEGORY`].
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an openWakeWord op-wiring conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct OpenwakewordOpReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16.
    pub bf16_passthrough: usize,
}

/// Converts an openWakeWord op-wiring safetensors checkpoint at `input`
/// (pre-flattened from the upstream ONNX / TFLite by
/// `tools/parity/openwakeword_op_prepare_checkpoint.py`) into a
/// Vokra-native GGUF at `output`, returning an
/// [`OpenwakewordOpReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// key; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license. The
/// default is `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`)
/// — callers redistributing the upstream CC-BY-NC-SA-4.0 official
/// weights MUST override to `--license cc-by-nc-sa-4.0` which flips
/// the publish gate to NonCommercialShareAlike (fail-closed for
/// commercial mode).
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_openwakeword_op_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<OpenwakewordOpReport, ConvertError> {
    // Load the whole checkpoint into memory — openWakeWord models are
    // tiny (~10 KB – 500 KB each), far below any streaming threshold.
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
            "dscripka/openWakeWord op-wiring (custom-KWS MLP/CNN over precomputed melspec, \
             Apache-2.0 code / CC-BY-NC-SA-4.0 official weights — Vokra does not \
             redistribute official weights; user-provided weights only, override to \
             --license cc-by-nc-sa-4.0 when distributing official CC-BY-NC-SA-4.0)",
        ),
    );

    let mut report = OpenwakewordOpReport::default();
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
            "vokra-openwakeword-op-{tag}-{}-{}.{ext}",
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
        // openWakeWord melspec CNN embedding weight — the upstream
        // Keras / ONNX-inspection-derived key convention preserved
        // verbatim through the `openwakeword_op_prepare_checkpoint.py`
        // bridge.
        let header = r#"{"melspec_model.embedding.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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
            convert_openwakeword_op_file(&input_path, &output_path, None).expect("convert BF16");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("melspec_model.embedding.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
    }

    /// Pins F32 and F16 pass-through. Apache-2.0 default → Permissive.
    #[test]
    fn f32_and_f16_tensors_pass_through_and_default_license_is_permissive() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_patterns: [u16; 2] = [0x3C00, 0x4000];
        let f16_bytes: Vec<u8> = f16_patterns.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"embedding_model.conv.0.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"classifier.dense.bias":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}}}}"#,
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

        let report = convert_openwakeword_op_file(&input_path, &output_path, None)
            .expect("convert F32 + F16 mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let f32_info = file
            .tensor_info("embedding_model.conv.0.weight")
            .expect("F32 tensor");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file
            .tensor_info("classifier.dense.bias")
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
            "apache-2.0 must resolve to Permissive (T1 tier)"
        );
    }

    /// Pins the license override boundary — a caller redistributing
    /// the upstream CC-BY-NC-SA-4.0 official weights overrides to
    /// flip the fail-closed publish gate to NonCommercialShareAlike.
    #[test]
    fn license_override_replaces_default() {
        let f32_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header =
            r#"{"embedding_model.embed.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
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
            convert_openwakeword_op_file(&input_path, &output_path, Some("cc-by-nc-sa-4.0"))
                .expect("convert with override");
        assert_eq!(report.written, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-sa-4.0"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "cc-by-nc-sa-4.0 override flips the class from Permissive to NonCommercialShareAlike",
        );
    }
}
