//! **bicodec** (Spark-TTS): safetensors checkpoint → GGUF conversion.
//!
//! Input: the upstream `SparkAudio/Spark-TTS-0.5B` release
//! (`huggingface.co/SparkAudio/Spark-TTS-0.5B`). Output: a Vokra GGUF
//! carrying every float tensor plus the `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks the native bicodec runtime side
//! will read.
//!
//! # Model card
//!
//! - **HF path**: `SparkAudio/Spark-TTS-0.5B`
//! - **License SPDX**: `cc-by-nc-sa-4.0` (the upstream model-card license;
//!   research-only and share-alike)
//! - **Category**: `codec` — bicodec is the dual-stream token codec used
//!   inside the Spark-TTS pipeline: a **semantic** stream that carries
//!   linguistic content plus a **fixed-length global speaker token**
//!   stream that carries timbre / prosody in a single conditioning
//!   vector shared across the whole utterance.
//!
//! # BF16 posture
//!
//! Follows the qwen3_tts / vibevoice / voxcpm2 land pattern (accepted
//! 2026-07-25): F32 / F16 / BF16 all pass through **verbatim** under
//! their upstream safetensors names. BF16 is emitted as GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at load
//! via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). No convert-time
//! widening, no silent F16 downcast (FR-EX-08).
//!
//! # Real-weight parity
//!
//! Deferred to the owner sign-off queue (`docs/license-audit.md` §3.1).
//! This converter is a native-side skeleton that pins the metadata
//! contract (arch / name / category / upstream-HF / license) plus the
//! BF16 pass-through invariant so a future `bicodec::from_gguf` can
//! bind against the same upstream tensor names once real weights are
//! audited.
//!
//! # No ONNX (permanent)
//!
//! Spark-TTS is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! planned for a future native implementation in `crates/vokra-models/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4); the
//! current converter intentionally stops at inspection-only GGUF output.

use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for bicodec GGUFs — kept intentionally distinct
/// from every sibling codec so the runtime dispatch cannot silently
/// route a bicodec artifact through a DAC / Mimi / RVQ / FSQ decoder.
pub(crate) const ARCH: &str = "bicodec";

/// `vokra.model.name` value for the canonical Spark-TTS bicodec release.
pub(crate) const NAME: &str = "spark-tts-bicodec";

/// Immutable upstream identities used by the inspection-only conversion.
pub const UPSTREAM_HF_REVISION: &str = "642071559bfc6346c2359d19dcb6be3f9dd8a05d";
pub const CHECKPOINT_BYTES: u64 = 625_518_756;
pub const CHECKPOINT_SHA256: &str =
    "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec";
pub const CONFIG_BYTES: u64 = 1_164;
pub const CONFIG_SHA256: &str = "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/SparkAudio/Spark-TTS";
pub const OFFICIAL_SOURCE_REVISION: &str = "2f1ea9082400547242641f5271b6f941c9f439d1";

/// `vokra.model.category` key — codec bucket for the artifact.
///
/// Kept as a local constant rather than a `vokra_core::gguf::chunks::*`
/// re-export because no other converter has taken the category slot yet; the key name
/// itself is documented at
/// [`docs/tickets/sota-coverage-plan-2026-07-22.md`] (SoTA plan bookkeeping).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Category string written into [`KEY_MODEL_CATEGORY`].
const MODEL_CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` key — HF repo path of the upstream weight.
///
/// Same rationale as [`KEY_MODEL_CATEGORY`]: kept as a local constant
/// because it is not yet part of the shared `vokra-core::chunks` surface;
/// the value is the HF `owner/repo` slug (no leading `huggingface.co/`,
/// no trailing slash).
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Value written under [`KEY_PROVENANCE_UPSTREAM_HF`].
const PROVENANCE_UPSTREAM_HF: &str = "SparkAudio/Spark-TTS-0.5B";

/// Default weight-license SPDX. Only this exact CC-BY-NC-SA value is accepted
/// by [`convert_bicodec_file`]; permissive relabels are rejected.
const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// Advisory source note stamped alongside the license into the
/// `vokra.provenance.source` chunk.
const PROVENANCE_SOURCE_NOTE: &str =
    "SparkAudio/Spark-TTS-0.5B (cc-by-nc-sa-4.0; research-only, share-alike)";

const KEY_UPSTREAM_REVISION: &str = "vokra.bicodec.upstream_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.bicodec.checkpoint_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.bicodec.config_sha256";
const KEY_SOURCE_REPOSITORY: &str = "vokra.bicodec.source_repository";
const KEY_SOURCE_REVISION: &str = "vokra.bicodec.source_revision";
const KEY_INSPECTION_STATUS: &str = "vokra.bicodec.inspection_status";
const KEY_INPUT_AUTHENTICATED: &str = "vokra.bicodec.input_authenticated";

/// Outcome of a bicodec conversion.
///
/// Mirrors the field set on the sibling converters
/// (`qwen3_tts::Qwen3TtsReport` / `vibevoice::VibeVoiceReport` /
/// `voxcpm2::VoxCpm2Report`) with the additive `read` counter the
/// file-based entry point surfaces so the caller can distinguish a
/// zero-tensor safetensors file from a zero-write outcome caused by
/// every tensor being quantized (defensive — the safetensors reader
/// rejects unknown dtypes at parse time today, so anything reaching the
/// non-float arm is a real regression on that reader).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BicodecReport {
    /// Total upstream tensors observed in the safetensors input.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy arm per the ADR-A_passthrough posture).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens
    /// BF16 → f32 losslessly at load via `decode_bf16` (`bits << 16` is
    /// exact — BF16 is the top 16 bits of an f32).
    pub bf16_passthrough: usize,
}

/// Convert a Spark-TTS bicodec safetensors checkpoint into a Vokra GGUF.
///
/// `input` is the upstream safetensors path; the emitted GGUF is written
/// to `output`. `license` may repeat the raw SPDX string stamped into
/// `vokra.provenance.license`; the only accepted value is
/// `DEFAULT_LICENSE_SPDX` (`"cc-by-nc-sa-4.0"`), matching the SparkAudio
/// weight card at `huggingface.co/SparkAudio/Spark-TTS-0.5B`.
///
/// # Errors
///
/// - I/O reading `input` or writing `output` propagates as
///   [`ConvertError::Io`].
/// - Safetensors parse failure propagates as [`ConvertError::Parse`].
/// - GGUF serialization failure propagates as the `From<GgufError>`
///   impl on `ConvertError`.
pub fn convert_bicodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<BicodecReport, ConvertError> {
    let license_spdx = require_license(license)?;
    let bytes = std::fs::read(input)?;
    validate_staged_identity(input, &bytes)?;
    convert_bicodec_bytes(&bytes, output, license_spdx)
}

/// Convert a byte buffer after the public entry point has authenticated it.
/// This private helper keeps format-loop tests small; production callers must use
/// [`convert_bicodec_file`], which authenticates both the checkpoint and its
/// required sibling `config.yaml` first.
fn convert_bicodec_bytes(
    bytes: &[u8],
    output: &Path,
    license_spdx: &str,
) -> Result<BicodecReport, ConvertError> {
    let st = SafetensorsFile::parse(bytes.to_vec())?;
    if st.tensors().is_empty() {
        return Err(ConvertError::Parse(
            "BiCodec checkpoint has no tensors; refusing to claim a complete conversion".to_owned(),
        ));
    }

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, PROVENANCE_UPSTREAM_HF);
    b.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_HF_REVISION);
    b.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
    b.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    b.add_string(KEY_SOURCE_REPOSITORY, OFFICIAL_SOURCE_REPOSITORY);
    b.add_string(KEY_SOURCE_REVISION, OFFICIAL_SOURCE_REVISION);
    b.add_string(KEY_INSPECTION_STATUS, "INSPECTION_ONLY");
    b.add_bool(KEY_INPUT_AUTHENTICATED, true);

    // Self-describing redistribution: the artifact carries its own licence.
    // `require_license` has already restricted this value to the upstream
    // CC-BY-NC-SA-4.0 identity, so the compliance class cannot be relabelled
    // as permissive. `None` selects the same upstream default.
    let class = LicenseClass::from_license_str(license_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        class,
        license_spdx,
        Some(NAME),
        Some(PROVENANCE_SOURCE_NOTE),
    );

    let mut report = BicodecReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the qwen3_tts / vibevoice /
    // voxcpm2 ADR-A_passthrough posture; the runtime widens BF16 → f32
    // exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (`bits << 16`
    // is exact — BF16 is the top 16 bits of an f32).
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
            "BiCodec checkpoint contains no supported float tensors; refusing to claim a complete conversion"
                .to_owned(),
        ));
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(report)
}

fn require_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let value = license
        .unwrap_or(DEFAULT_LICENSE_SPDX)
        .trim()
        .to_ascii_lowercase();
    if value != DEFAULT_LICENSE_SPDX {
        return Err(ConvertError::Usage(format!(
            "BiCodec weights are cc-by-nc-sa-4.0; refusing license override `{value}`"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

/// Validate the fixed HF artifact when the input is staged beside its official
/// `config.yaml`. Synthetic unit fixtures intentionally omit that sidecar and
/// therefore exercise only the format loop; the VAST worker always supplies
/// the sidecar and cannot bypass this identity check.
fn validate_staged_identity(input: &Path, checkpoint: &[u8]) -> Result<(), ConvertError> {
    let config = input
        .parent()
        .map(|parent| parent.join("config.yaml"))
        .unwrap_or_else(|| PathBuf::from("config.yaml"));
    if !config.exists() {
        return Err(ConvertError::Parse(
            "BiCodec input must be accompanied by the authenticated config.yaml sidecar".to_owned(),
        ));
    }
    let checkpoint_sha =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(checkpoint));
    if checkpoint.len() as u64 != CHECKPOINT_BYTES || checkpoint_sha != CHECKPOINT_SHA256 {
        return Err(ConvertError::Parse(format!(
            "BiCodec checkpoint identity mismatch: bytes={} sha256={checkpoint_sha}",
            checkpoint.len()
        )));
    }
    let config_bytes = std::fs::read(&config)?;
    let config_sha =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(&config_bytes));
    if config_bytes.len() as u64 != CONFIG_BYTES || config_sha != CONFIG_SHA256 {
        return Err(ConvertError::Parse(format!(
            "BiCodec config identity mismatch: bytes={} sha256={config_sha}",
            config_bytes.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a single-BF16-tensor safetensors buffer with a caller-supplied
    /// raw payload. Panics if `bf16_bytes.len() != shape × 2` — that would
    /// declare an invalid safetensors header the reader would reject.
    fn safetensors_one_bf16(shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            bf16_bytes.len(),
            elems as usize * 2,
            "test fixture: payload len must match shape × 2 BF16"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"bicodec.embed.weight":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Builds a two-tensor safetensors buffer with one F32 tensor followed
    /// by one F16 tensor. Both payloads are the caller's raw bytes.
    fn safetensors_f32_then_f16(
        f32_bytes: &[u8],
        f32_shape: &[u64],
        f16_bytes: &[u8],
        f16_shape: &[u64],
    ) -> Vec<u8> {
        let f32_expected = (f32_shape.iter().product::<u64>() as usize) * 4;
        let f16_expected = (f16_shape.iter().product::<u64>() as usize) * 2;
        assert_eq!(f32_bytes.len(), f32_expected, "F32 fixture payload len");
        assert_eq!(f16_bytes.len(), f16_expected, "F16 fixture payload len");
        let f32_end = f32_bytes.len();
        let f16_end = f32_end + f16_bytes.len();
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"bicodec.semantic.weight":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_end}]}},"bicodec.global.weight":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_end},{f16_end}]}}}}"#,
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("vokra-bicodec-{tag}-{}.{ext}", std::process::id()));
        p
    }

    /// STEP 1 (RED) pin — a synthetic BF16 tensor must survive the file-based
    /// convert path byte-for-byte: same dtype (`GgmlType::BF16` = GGUF type
    /// 30), same shape, same payload bytes. Mirrors the qwen3_tts / vibevoice
    /// BF16 pass-through pins.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // BF16 payload with known non-zero bit patterns — a raw-zero fixture
        // would round-trip trivially through any silent widen / downcast.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let input_bytes = safetensors_one_bf16(&[2, 3], &bf16);

        let input = temp_path("bf16-in", "safetensors");
        let output = temp_path("bf16-out", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_bicodec_bytes(&input_bytes, &output, DEFAULT_LICENSE_SPDX)
            .expect("convert synthetic fixture");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (ADR A_passthrough)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the on-disk GGUF: dtype preserved, payload
        // byte-identical — the "no convert-time widening" posture pin.
        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("bicodec.embed.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Metadata contract the module docstring promises.
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
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(PROVENANCE_UPSTREAM_HF)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX),
            "default license must be CC-BY-NC-SA-4.0 when the `license` param is None"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str()),
            "default class must retain both the NC and share-alike obligations"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// STEP 1 (RED) pin — mixed-dtype loops don't collapse to one arm: an
    /// F32 tensor and an F16 tensor in the same input must **both** pass
    /// through, with counters {`written=2`, `bf16_passthrough=0`,
    /// `skipped_non_float=0`}. Guards against a naive `if bf16 { ... } else`
    /// refactor that would only emit one branch of the match.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_values: [f32; 6] = [7.0, -8.25, 0.5, -0.5, 1.5, -3.75];
        let f32_bytes: Vec<u8> = f32_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_values: [f32; 4] = [1.0, -1.0, 0.5, -0.5];
        // Round f32 → f16 (IEEE-754 half): sign 1 + exp 5 + mant 10.
        let f16_bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                let sign = ((bits >> 16) & 0x8000) as u16;
                let exp32 = ((bits >> 23) & 0xFF) as i32;
                let mant32 = bits & 0x7F_FFFF;
                let (exp16, mant16) = if exp32 == 0 {
                    (0u16, 0u16)
                } else if exp32 == 0xFF {
                    (0x1F, (mant32 >> 13) as u16)
                } else {
                    let e = exp32 - 127 + 15;
                    if e <= 0 {
                        (0, 0)
                    } else if e >= 0x1F {
                        (0x1F, 0)
                    } else {
                        (e as u16, (mant32 >> 13) as u16)
                    }
                };
                (sign | (exp16 << 10) | mant16).to_le_bytes()
            })
            .collect();

        let input_bytes = safetensors_f32_then_f16(&f32_bytes, &[2, 3], &f16_bytes, &[2, 2]);

        let input = temp_path("f32f16-in", "safetensors");
        let output = temp_path("f32f16-out", "gguf");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_bicodec_bytes(&input_bytes, &output, DEFAULT_LICENSE_SPDX)
            .expect("convert synthetic fixture");
        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(report.written, 2, "both F32 and F16 must pass through");
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only and F16-only tensors must leave the BF16 counter at 0"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Both tensors survive with dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");

        let f32_info = file
            .tensor_info("bicodec.semantic.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("bicodec.global.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![2, 2]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn permissive_license_override_is_rejected() {
        let input = temp_path("license-in", "safetensors");
        let output = temp_path("license-out", "gguf");
        std::fs::write(&input, safetensors_one_bf16(&[1], &[0, 0])).expect("write input");
        let error = require_license(Some("apache-2.0"))
            .expect_err("BiCodec must reject a permissive relabel");
        assert!(error.to_string().contains("cc-by-nc-sa-4.0"));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn empty_checkpoint_is_not_a_complete_conversion() {
        let input = temp_path("empty-in", "safetensors");
        let output = temp_path("empty-out", "gguf");
        let header = b"{}";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        std::fs::write(&input, &bytes).expect("write input");
        let error = convert_bicodec_bytes(&bytes, &output, DEFAULT_LICENSE_SPDX)
            .expect_err("empty BiCodec input must fail closed");
        assert!(error.to_string().contains("no tensors"));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn public_conversion_requires_config_sidecar() {
        let input = temp_path("missing-config-in", "safetensors");
        let output = temp_path("missing-config-out", "gguf");
        std::fs::write(&input, safetensors_one_bf16(&[1], &[0, 0])).expect("write input");
        let error = convert_bicodec_file(&input, &output, None)
            .expect_err("public conversion must require authenticated config");
        assert!(error.to_string().contains("config.yaml sidecar"));
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
