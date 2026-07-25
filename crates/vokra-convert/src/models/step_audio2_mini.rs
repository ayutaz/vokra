//! **StepFun Step-Audio-2-mini**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 3, 2026-07-25).
//!
//! - Upstream: `stepfun-ai/Step-Audio-2-mini`
//!   (`huggingface.co/stepfun-ai/Step-Audio-2-mini`).
//! - License: **apache-2.0** end-to-end (`Permissive`).
//! - Category: **s2s** (speech-to-speech). StepFun 8B S2S with a dual codebook
//!   (semantic 1024 + acoustic 4096) and a flow-matching mel decoder.
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.model.*` /
//! `vokra.provenance.*` / `vokra.schema.*` metadata chunks the runtime
//! consumes. This is the **skeleton** converter — real-weight parity is
//! deferred to owner (docs/license-audit.md §3.1 sign-off); every F32 / F16 /
//! BF16 tensor passes through verbatim under its upstream name.
//!
//! # BF16 posture
//!
//! Matches the sibling ADR (qwen3-tts / vibevoice / voxcpm2 / moshi /
//! voxtral): BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) with no convert-time widening. The runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of an f32 — `bits << 16` is exact). No silent widen / downcast
//! (FR-EX-08).
//!
//! # No side-car config
//!
//! Real-weight binding is a follow-up wave gated on the upstream
//! tensor-name manifest fetch; this converter takes no `--config` today
//! and passes every F32 / F16 / BF16 tensor through unchanged so a
//! future `StepAudio2MiniWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Step-Audio-2-mini is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Step-Audio-2-mini GGUFs. Intentionally distinct
/// from every sibling arch tag — Step-Audio-2-mini is a dual-codebook S2S
/// (semantic 1024 + acoustic 4096) with a flow-matching mel decoder, so
/// silently sharing an arch would mis-route the runtime dispatch.
pub(crate) const ARCH: &str = "step_audio2_mini";

/// `vokra.model.name` value written for the canonical Step-Audio-2-mini GGUF.
pub(crate) const NAME: &str = "step-audio-2-mini";

/// `vokra.model.category` value — Step-Audio-2-mini is a speech-to-speech
/// model. The category chunk is a taxonomy tag orthogonal to `arch`; the
/// runtime does not dispatch on it (arch does), but it is machine-readable
/// for model-zoo / catalog surfaces (see `docs/license-audit.md`).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const CATEGORY: &str = "s2s";

/// The default upstream weight license — `apache-2.0` end-to-end
/// (verified via HF model card `license: apache-2.0`). Callers can
/// override at the `convert_step_audio2_mini_file(_, _, license=Some(_))`
/// boundary when the source distribution declares a different SPDX id.
const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// Human-readable upstream source note stored in
/// `vokra.provenance.source` (`KEY_PROVENANCE_SOURCE`).
const UPSTREAM_SOURCE: &str = "stepfun-ai/Step-Audio-2-mini (apache-2.0 end-to-end)";

/// Outcome of a Step-Audio-2-mini conversion. Additive counters — a
/// non-zero value on any field is a positive report; a zero `written`
/// value means the input safetensors carried no float tensors and the
/// runtime will refuse to bind any weights (FR-EX-08).
#[derive(Debug, Default)]
pub struct StepAudio2MiniReport {
    /// Total tensors observed in the input safetensors.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (observability counter — the ADR pattern shared with qwen3-tts /
    /// vibevoice / voxcpm2 / moshi / voxtral so a latent silent-widen
    /// cannot slip in undetected).
    pub bf16_passthrough: usize,
}

/// Converts a Step-Audio-2-mini safetensors checkpoint into a Vokra GGUF
/// written to `output`, returning a [`StepAudio2MiniReport`].
///
/// Every F32 / F16 / BF16 tensor passes through **verbatim** under its
/// upstream name — no convert-time widening (the sibling ADR posture
/// shared with qwen3-tts / vibevoice / voxcpm2 / moshi / voxtral).
///
/// `license` optionally overrides the stamped weight license. `None`
/// keeps the built-in default (`apache-2.0`); `Some(spdx)` writes the
/// caller-supplied SPDX id and re-derives the [`LicenseClass`] via
/// [`LicenseClass::from_license_str`]. This mirrors the outer
/// `convert_file_licensed` license-override contract.
///
/// # Errors
///
/// - [`ConvertError::Io`] if the input cannot be read or the output
///   cannot be written.
/// - [`ConvertError::Parse`] if the safetensors header is malformed.
/// - [`ConvertError::Gguf`] if the GGUF cannot be assembled.
pub fn convert_step_audio2_mini_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<StepAudio2MiniReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // License stamp: caller override wins, otherwise the built-in default
    // (apache-2.0 end-to-end — the HF model card value). We re-derive the
    // canonical [`LicenseClass`] via [`LicenseClass::from_license_str`] so
    // the class chunk stays in agreement with the raw SPDX id chunk (no
    // silent divergence — FR-EX-08 / M2-13).
    let license_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let license_class = LicenseClass::from_license_str(license_spdx);
    vokra_core::stamp_provenance(
        &mut b,
        license_class,
        license_spdx,
        Some(NAME),
        Some(UPSTREAM_SOURCE),
    );

    let mut report = StepAudio2MiniReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the sibling ADR shared with
    // qwen3-tts / vibevoice / voxcpm2 / moshi / voxtral; the runtime widens
    // BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    /// A unique temp path — per-process id **plus** a monotonic counter so
    /// two tests in the same process never race on the same file.
    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-step-audio2-mini-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-declared dtype and raw payload.
    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// Encodes an f32 array as little-endian BF16 bytes (top 16 bits of
    /// the f32 pattern — the exact inverse of the runtime's
    /// `decode_bf16 : bits << 16`).
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// Encodes an f32 array as little-endian F16 bytes via IEEE-754 half
    /// truncation (no external dep). The bit-pattern here is only used
    /// for byte-identity round-trip through the converter — no runtime
    /// widening is exercised, so this simple round-trip-shaped payload
    /// suffices.
    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| {
                // Simple round-to-nearest-even, ignoring denormals / NaN
                // subtleties — the test inputs below are exactly-representable
                // half values (±1.0, ±2.0, ±0.5).
                let b = v.to_bits();
                let sign = ((b >> 16) & 0x8000) as u16;
                let exp = ((b >> 23) & 0xff) as i32;
                let mantissa = b & 0x007f_ffff;
                let h = if exp == 0 {
                    sign
                } else if exp == 0xff {
                    sign | 0x7c00 | ((mantissa >> 13) as u16)
                } else {
                    let e = exp - 127 + 15;
                    if e <= 0 {
                        sign
                    } else if e >= 0x1f {
                        sign | 0x7c00
                    } else {
                        sign | ((e as u16) << 10) | ((mantissa >> 13) as u16)
                    }
                };
                h.to_le_bytes()
            })
            .collect()
    }

    /// RED-phase red-line — the BF16 pass-through arm must emit GGUF
    /// type 30 (`GgmlType::BF16`) with byte-identical payload, mirror
    /// of the qwen3-tts / vibevoice / voxcpm2 / moshi / voxtral pin.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Build a BF16 payload with known non-zero bit patterns so a
        // byte-identity assert catches any silent widen / downcast (a
        // zeroed payload would round-trip trivially through F32/F16
        // widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16 = bf16_bytes(&values);
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let input_bytes = safetensors_one("model.embed_tokens.weight", "BF16", &[2, 3], &bf16);

        let input = tmp_path("bf16-in");
        let output = tmp_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_step_audio2_mini_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one input tensor observed");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // Round-trip through the GGUF file: dtype preserved, payload
        // byte-identical to input (Moshi's `assert_eq!(info.dtype,
        // GgmlType::BF16, "no convert-time widening")` posture).
        let file = GgufFile::open(&output).expect("load output gguf");
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    /// The mixed-dtype pass-through arm must accept BOTH F32 and F16 in
    /// one call and leave the BF16 counter untouched (additive-default
    /// invariant: `bf16_passthrough == 0` on an F32/F16-only input).
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Tensor 1: F32 [1,2] → 8 bytes @ [0..8).
        // Tensor 2: F16 [1,2] → 4 bytes @ [8..12).
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 8);
        let f16_vals: [f32; 2] = [1.0, -2.0];
        let f16 = f16_bytes(&f16_vals);
        assert_eq!(f16.len(), 4);

        let header = r#"{"model.a.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"model.b.weight":{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]}}"#;
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16);

        let input = tmp_path("f32f16-in");
        let output = tmp_path("f32f16-out");
        std::fs::write(&input, &input_bytes).expect("write input");

        let report = convert_step_audio2_mini_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two input tensors observed");
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
            "F32/F16-only input must leave the BF16 counter at the Default 0 (additive-default invariant)"
        );

        // Both tensors survive with their dtypes preserved.
        let file = GgufFile::open(&output).expect("load output gguf");
        let f32_info = file.tensor_info("model.a.weight").expect("F32 present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
        let f16_info = file.tensor_info("model.b.weight").expect("F16 present");
        assert_eq!(f16_info.dtype, GgmlType::F16);
        assert_eq!(file.tensor_bytes(f16_info), f16.as_slice());

        // Arch / name / category / provenance chunks must survive.
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
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
