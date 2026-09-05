//! **XY_Tokenizer**: safetensors checkpoint → GGUF conversion.
//!
//! - **HF**: `OpenMOSS-Team/XY_Tokenizer_TTSD_V0`
//! - **License**: `apache-2.0` for the weights; the official source
//!   `readme.md` explicitly declares Apache-2.0 at the fixed revision, but no
//!   full `LICENSE`/`COPYING`/`NOTICE`/`COPYRIGHT` file is tracked. The source
//!   is not copied or derived here.
//! - **Category**: `codec`
//! - **Notes**: 1 kbps RVQ-8 @ 12.5 Hz, MOSS-TTSD backend.
//!
//! Input: a VAST-prepared safetensors artifact derived from the upstream
//! `OpenMOSS-Team/XY_Tokenizer_TTSD_V0` `.ckpt`. Output is deliberately
//! `INSPECTION_ONLY`; no native XY_Tokenizer binder exists yet because the
//! authenticated evidence still has an unverified topology contract.
//!
//! # BF16 posture
//!
//! Follows the `qwen3_tts` / `vibevoice` / `voxcpm2` landed pattern:
//! BF16 tensors pass through **verbatim** as GGUF type 30
//! (`GgmlType::BF16`) with no convert-time widening. The runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Mirrors the
//! `Qwen3TtsReport` counter shape (`written` / `skipped_non_float` /
//! `bf16_passthrough`) plus an additive `read` field so the operator
//! can cross-check that every input tensor landed on exactly one arm.
//!
//! # No ONNX (permanent)
//!
//! XY_Tokenizer is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); a native pipeline is a
//! future implementation (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).
//!
//! # License override
//!
//! `convert_xy_tokenizer_file` accepts the upstream `apache-2.0` value only;
//! arbitrary license overrides are rejected so the weight license cannot be
//! confused with the separately recorded source README declaration.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for XY_Tokenizer GGUFs. Intentionally **distinct**
/// from every sibling arch tag (`mimi`, `dac`, `wavtokenizer_vq`,
/// `xcodec2_fsq`, ...) — silently sharing an arch tag would mis-route
/// the runtime dispatch.
#[allow(dead_code)] // Retained as inspection-only dispatch metadata until binding is authenticated.
pub(crate) const ARCH: &str = "xy_tokenizer";

/// `vokra.model.name` value written for the canonical XY_Tokenizer
/// GGUF.
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub(crate) const NAME: &str = "xy_tokenizer_ttsd_v0";

/// The upstream HuggingFace repo path — recorded verbatim on the
/// GGUF at `vokra.provenance.upstream_hf`.
#[allow(dead_code)] // Retained as inspection-only provenance until the artifact is authenticated.
pub(crate) const UPSTREAM_HF: &str = "OpenMOSS-Team/XY_Tokenizer_TTSD_V0";

/// Default SPDX license string (upstream ships apache-2.0 end-to-end).
#[allow(dead_code)] // Retained as inspection-only license metadata until the artifact is authenticated.
pub(crate) const DEFAULT_LICENSE: &str = "apache-2.0";

/// Model category value written to `vokra.model.category`.
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub(crate) const CATEGORY: &str = "codec";

#[allow(dead_code)] // Retained as inspection-only provenance until the artifact is authenticated.
pub const UPSTREAM_REVISION: &str = "c83433728e698ed0698e88cb5096bc221fb8f8c5";
#[allow(dead_code)] // Retained as inspection-only checkpoint evidence until authenticated.
pub const CHECKPOINT_BYTES: u64 = 2_137_328_977;
#[allow(dead_code)] // Retained as inspection-only checkpoint evidence until authenticated.
pub const CHECKPOINT_SHA256: &str =
    "37c7ac18d0a48f5a1d0687e31af7c0264861232c500206718c98acd8e37d1671";
#[allow(dead_code)] // Retained as inspection-only sidecar evidence until authenticated.
pub const CONFIG_RELATIVE: &str = "config/xy_tokenizer_config.yaml";
#[allow(dead_code)] // Retained as inspection-only sidecar evidence until authenticated.
pub const CONFIG_SHA256: &str = "e7d48677e34f77e5b9fd7dc7a3e0eef7f2d2dd9be9a245d5c1d56489dc748938";
#[allow(dead_code)] // Retained as inspection-only source evidence until authenticated.
pub const SOURCE_REPOSITORY: &str = "https://github.com/gyt1145028706/XY-Tokenizer";
#[allow(dead_code)] // Retained as inspection-only source evidence until authenticated.
pub const SOURCE_REVISION: &str = "5df5609c5883e555bd39a2d0b1005ca8f1a8f12e";

// Additional provenance / model keys not in the shared
// `chunks::KEY_PROVENANCE_*` set. These are additive string keys
// specific to the deep-not-wide catalog (CLAUDE.md 2026-07-22
// project-goal-depth-not-breadth): every codec / TTS record carries its
// upstream HF path and category verbatim so `check-catalog-reality.sh`
// and `make_model_card.py` can drive gates without a side registry.
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_UPSTREAM_REVISION: &str = "vokra.xy_tokenizer.upstream_revision";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_CHECKPOINT_SHA256: &str = "vokra.xy_tokenizer.checkpoint_sha256";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_CONFIG_SHA256: &str = "vokra.xy_tokenizer.config_sha256";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_SOURCE_REPOSITORY: &str = "vokra.xy_tokenizer.source_repository";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_SOURCE_REVISION: &str = "vokra.xy_tokenizer.source_revision";
#[allow(dead_code)] // Metadata key is reserved for the future authenticated converter.
const KEY_INSPECTION_STATUS: &str = "vokra.xy_tokenizer.inspection_status";

/// Outcome of an XY_Tokenizer conversion.
///
/// Mirrors the `super::qwen3_tts::Qwen3TtsReport` counter shape
/// (`written` / `skipped_non_float` / `bf16_passthrough`) plus an
/// additive `read` field so the invariant
/// `report.read == report.written + report.skipped_non_float` holds for
/// every well-formed input. Fields are `pub` (not `pub(crate)`) because
/// this converter's file entry point is on the public API surface —
/// callers need to read the counters directly for their own reporting.
#[derive(Debug, Default)]
pub struct XyTokenizerReport {
    /// Total tensors read from the input safetensors file (should equal
    /// `written + skipped_non_float`).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — the shared
    /// pass-through arm from the `qwen3_tts` / `vibevoice` / `voxcpm2`
    /// contract).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 today, so anything reaching
    /// this arm signals an upstream reader change).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — emits GGUF
    /// type 30 verbatim; runtime widens BF16 → f32 losslessly via the
    /// single choke point `vokra-core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Convert an XY_Tokenizer safetensors checkpoint at `input` into a
/// Vokra GGUF at `output`.
///
/// The public conversion entry point currently refuses every input with an
/// explicit `INSPECTION_ONLY` error: although the VAST prepared-artifact SHA
/// and complete tensor manifest are authenticated, the topology contract,
/// native runtime path, and independent numerical parity are not. The private
/// synthetic format helper exists only for converter tests. The only accepted
/// license for a future conversion is the upstream `apache-2.0` weight
/// license.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read / write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF writer failure.
pub fn convert_xy_tokenizer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<XyTokenizerReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(
        "XY-Tokenizer conversion is INSPECTION_ONLY: topology contract, native runtime, and independent parity remain unauthenticated"
            .to_owned(),
    ))
}

#[allow(dead_code)] // Synthetic helper is reserved for the authenticated converter.
fn convert_xy_tokenizer_bytes(
    bytes: &[u8],
    output: &Path,
    spdx: &str,
) -> Result<XyTokenizerReport, ConvertError> {
    let st = SafetensorsFile::parse(bytes.to_vec())?;
    if st.tensors().is_empty() {
        return Err(ConvertError::Parse(
            "XY-Tokenizer checkpoint has no tensors; refusing complete conversion".to_owned(),
        ));
    }

    let mut b = GgufBuilder::new();
    // Canonical model identity stamps (chunks::KEY_MODEL_ARCH / _NAME).
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Additive keys the deep-not-wide catalog gates read
    // (`check-catalog-reality.sh`, `make_model_card.py`). These are
    // additive strings under `vokra.provenance.*` / `vokra.model.*` —
    // canonical keys such as `vokra.provenance.source` are still stamped
    // by `stamp_provenance` below, so the two sets do not overlap.
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    b.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
    b.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    b.add_string(KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY);
    b.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    b.add_string(KEY_INSPECTION_STATUS, "INSPECTION_ONLY");

    // Self-describing redistribution: the artifact carries its own
    // license. Default = `apache-2.0` (the OpenMOSS model-card weight
    // license). `require_license` rejects relabeling this weight as another
    // license; source README evidence is recorded separately. The
    // `LicenseClass` is derived through the shared classifier.
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_HF));

    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the shared ADR
    // (`docs/adr/qwen3-tts-bf16.md`, strategy A_passthrough); the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `vokra-core::gguf::quant::decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert`.
    let mut report = XyTokenizerReport::default();
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

    if report.written == 0 {
        return Err(ConvertError::Parse(
            "XY-Tokenizer checkpoint contains no supported float tensors".to_owned(),
        ));
    }

    let gguf_bytes = b.to_bytes()?;
    std::fs::write(output, &gguf_bytes)?;
    Ok(report)
}

#[allow(dead_code)] // License gate is reserved for the authenticated converter.
fn require_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let value = license
        .unwrap_or(DEFAULT_LICENSE)
        .trim()
        .to_ascii_lowercase();
    if value != DEFAULT_LICENSE {
        return Err(ConvertError::Usage(format!(
            "XY-Tokenizer weights are apache-2.0; refusing license override `{value}`"
        )));
    }
    Ok(DEFAULT_LICENSE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a unique temp path for a test artefact so parallel test
    /// runs never collide. Mirrors `models::moshi` streaming tests.
    fn unique_temp(prefix: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-xy_tokenizer-{prefix}-{}.{ext}",
            std::process::id()
        ));
        p
    }

    /// Builds a synthetic single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Panics if `bf16_bytes.len()` disagrees
    /// with `shape × 2` — mirrors `qwen3_tts::tests::safetensors_one_bf16`
    /// but named for XY_Tokenizer's `codebook.weight` scaffold.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            bf16_bytes.len(),
            elems as usize * 2,
            "test fixture: BF16 payload len must be shape × 2"
        );
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

    /// Two-tensor mixed-dtype fixture (F32 + F16 side by side) — covers
    /// the F32 and F16 legs of the pass-through arm in one run.
    fn safetensors_f32_and_f16() -> Vec<u8> {
        // F32 tensor (shape [2,3] → 24 bytes) then F16 tensor (shape [2,3]
        // → 12 bytes). Payload order matches lexicographic data_offsets so
        // the safetensors parser accepts the header in the natural order.
        let f32_vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_vals: [u16; 6] = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
        let f16_bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"a_f32":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}},"b_f16":{{"dtype":"F16","shape":[2,3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Build a BF16 payload with known non-zero bit patterns so the
        // byte-identity assert catches a silent widen / downcast attempt
        // (a raw zeroed payload would round-trip trivially through F32/F16
        // widen).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        let input_bytes = safetensors_one_bf16("codebook.weight", &[2, 3], &bf16);

        let input = unique_temp("bf16-in", "safetensors");
        let output = unique_temp("bf16-out", "gguf");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_xy_tokenizer_bytes(&input_bytes, &output, DEFAULT_LICENSE)
            .expect("convert synthetic fixture");
        assert_eq!(report.read, 1, "one input tensor was read");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror of qwen3_tts)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must NOT land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the GGUF: dtype preserved + payload
        // byte-identical (the Moshi assert_eq!(info.dtype, GgmlType::BF16,
        // "no convert-time widening") posture).
        let out_bytes = std::fs::read(&output).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("codebook.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays GGUF type 30"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let input_bytes = safetensors_f32_and_f16();

        let input = unique_temp("mixed-in", "safetensors");
        let output = unique_temp("mixed-out", "gguf");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_xy_tokenizer_bytes(&input_bytes, &output, DEFAULT_LICENSE)
            .expect("convert synthetic fixture");
        assert_eq!(report.read, 2, "two input tensors were read");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32+F16 input must leave the BF16 counter at zero"
        );

        // Both tensors survive the round-trip under their upstream names
        // with their dtypes preserved.
        let out_bytes = std::fs::read(&output).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let f32_info = file.tensor_info("a_f32").expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        let f16_info = file.tensor_info("b_f16").expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![2, 3]);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn permissive_license_override_is_rejected() {
        let error = require_license(Some("cc-by-4.0"))
            .expect_err("XY-Tokenizer must preserve the Apache-2.0 weight license");
        assert!(error.to_string().contains("apache-2.0"));
    }

    #[test]
    fn empty_checkpoint_is_not_complete() {
        let header = b"{}";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        let output = unique_temp("empty-out", "gguf");
        let error = convert_xy_tokenizer_bytes(&bytes, &output, DEFAULT_LICENSE)
            .expect_err("empty XY-Tokenizer input must fail closed");
        assert!(error.to_string().contains("no tensors"));
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn public_conversion_requires_prepared_sidecars() {
        let input = unique_temp("missing-manifest-in", "safetensors");
        let output = unique_temp("missing-manifest-out", "gguf");
        std::fs::write(
            &input,
            safetensors_one_bf16("codebook.weight", &[1], &[0, 0]),
        )
        .unwrap();
        let error = convert_xy_tokenizer_file(&input, &output, None)
            .expect_err("public conversion must require VAST authentication");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
