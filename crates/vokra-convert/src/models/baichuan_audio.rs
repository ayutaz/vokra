// TDD skeleton per the task spec (2026-07-25) — the module exposes a
// `pub` surface (`BaichuanAudioReport` + `convert_baichuan_audio_file`)
// so an external caller can eventually reach it, but no lib.rs re-
// export is in scope here (`docs/tickets/` follow-up: register a
// `ModelKind::BaichuanAudio` arm + a `convert_baichuan_audio_file`
// wrapper). Until the wire-up lands, every item is only reachable from
// the in-module tests below — the dead-code lint fires against the
// public surface at the crate boundary. Suppress at the module level
// so the intent is auditable in one place rather than sprinkled per
// item.
#![allow(dead_code)]

//! **baichuan_audio (Baichuan-Audio / Baichuan Omni-1.5)**: safetensors
//! checkpoint → GGUF conversion.
//!
//! Input: the upstream `baichuan-inc/Baichuan-Audio` release. Output: a
//! Vokra GGUF carrying every float tensor plus the mandated
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks so the runtime
//! (and any downstream Baichuan-Audio implementation) can bind the
//! artifact loudly.
//!
//! - **HF path**: `baichuan-inc/Baichuan-Audio`
//! - **License** (SPDX): `apache-2.0` (Permissive)
//! - **Category**: `s2s` (Baichuan Omni-1.5 — Whisper-Large encoder +
//!   8-layer RVQ 12.5 Hz + Flow Matching mel + CosyVoice2 HiFi-GAN)
//!
//! # BF16 posture (ADR mirror)
//!
//! Every F32 / F16 / BF16 tensor passes through **verbatim** under its
//! upstream safetensors name. BF16 is emitted as GGUF type 30
//! (`GgmlType::BF16`) with no convert-time widening; the runtime widens
//! BF16 → f32 losslessly at load through the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = the top
//! 16 bits of an f32 — `bits << 16` is exact). Mirrors the accepted
//! `qwen3_tts` / `vibevoice` / `voxcpm2` / `moshi` posture
//! (2026-07-25).
//!
//! # Real-weight parity (deferred)
//!
//! Native Baichuan-Audio runtime binding + real-weight parity is
//! deferred to the owner track (`docs/license-audit.md` §3.1 sign-off,
//! plus a follow-up `vokra-models::baichuan_audio` port). This converter
//! is the TDD-first skeleton in the qwen3-tts / vibevoice / voxcpm2
//! pattern: every float tensor passes through verbatim so a future
//! `BaichuanAudioWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Baichuan-Audio is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for baichuan_audio GGUFs — kept as a distinct arch
/// tag so the future runtime dispatch cannot be silently mis-routed onto
/// a sibling model (Whisper / CosyVoice2 / Moshi are all near-neighbour
/// in the pipeline description).
pub(crate) const ARCH: &str = "baichuan_audio";

/// `vokra.model.name` value written for the canonical Baichuan-Audio
/// release.
pub(crate) const NAME: &str = "baichuan-audio";

/// `vokra.model.category` value ("s2s" per the task spec).
pub(crate) const CATEGORY: &str = "s2s";

/// Upstream Hugging Face repository path
/// (`huggingface.co/baichuan-inc/Baichuan-Audio`).
pub(crate) const UPSTREAM_HF: &str = "baichuan-inc/Baichuan-Audio";

/// SPDX identifier of the upstream weight licence.
pub(crate) const LICENSE_SPDX: &str = "apache-2.0";

// --- Metadata keys mandated by the task ---------------------------------
//
// The task requires two extra keys on top of the existing
// `stamp_provenance` surface: `vokra.provenance.upstream_hf` and
// `vokra.model.category`. Neither is exposed by `vokra-core::chunks`
// today; the sibling models write the HF path into `vokra.provenance.
// source` via `stamp_provenance` (which we ALSO do below, for symmetry
// with the sibling converters). Keeping the constants module-local
// avoids leaking a converter-only key into the shared chunk surface
// until the runtime side adopts them.

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a baichuan_audio conversion.
///
/// Field parity with `qwen3_tts` / `vibevoice` / `voxcpm2` counters plus
/// an explicit `read` totals field for the task spec:
/// `read = written + skipped_non_float` (defensive audit — the two
/// derived counters must agree with the total tensor-header walk).
#[derive(Debug, Default)]
pub struct BaichuanAudioReport {
    /// Total number of tensors seen in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 through the same
    /// byte-copy path — BF16 emits GGUF type 30 unchanged).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the shared
    /// safetensors reader accepts only F32 / F16 / BF16 today, so any
    /// non-zero here signals an upstream reader change).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Runtime widens BF16 → f32 losslessly via
    /// `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Converts a Baichuan-Audio safetensors checkpoint at `input` into a
/// Vokra GGUF at `output`.
///
/// The emitted GGUF carries the task-mandated metadata:
///
/// - `vokra.model.arch` = `"baichuan_audio"`
/// - `vokra.model.name` = `"baichuan-audio"`
/// - `vokra.model.category` = `"s2s"`
/// - `vokra.provenance.upstream_hf` = `"baichuan-inc/Baichuan-Audio"`
/// - `vokra.provenance.license` = the SPDX id (`"apache-2.0"` by
///   default, overridden by the `license` argument when `Some`)
/// - `vokra.provenance.weight_license` = the [`LicenseClass`] derived
///   from the SPDX id
///
/// Plus the unconditional `vokra.schema.version` / `vokra.schema.producer`
/// stamps [`GgufBuilder`] auto-adds on serialization.
///
/// `license`, when `Some`, overrides the built-in `apache-2.0` stamp —
/// the caller declares the licence of the actual distribution source
/// and the converter re-derives the [`LicenseClass`] from that string.
///
/// # Errors
///
/// [`ConvertError::Io`] on read/write failure, [`ConvertError::Parse`]
/// on a malformed safetensors buffer, [`ConvertError::Gguf`] on a GGUF
/// write failure.
pub fn convert_baichuan_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<BaichuanAudioReport, ConvertError> {
    // Full-buffer read: mirror the sibling in-memory `models::*::convert`
    // path. The streaming (moshi-style) path is a follow-up gated on the
    // real 7B+ checkpoint size — the skeleton contract only guarantees
    // the shape-and-provenance pass.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    let effective_spdx = license.unwrap_or(LICENSE_SPDX);
    write_metadata(&mut b, effective_spdx);

    let mut report = BaichuanAudioReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening
    // (mirror qwen3_tts / vibevoice / voxcpm2 / moshi ADR, 2026-07-25).
    // BF16 stays GGUF `BF16` (type 30); the runtime widens BF16 → f32
    // losslessly at load via `vokra-core::gguf::quant::decode_bf16`.
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

/// Writes the task-mandated metadata + `stamp_provenance` surface onto
/// `b` given the effective SPDX licence id.
///
/// - `vokra.model.arch` / `vokra.model.name` / `vokra.model.category`
/// - `vokra.provenance.upstream_hf` (task-mandated key)
/// - `vokra.provenance.weight_license` / `vokra.provenance.license` /
///   `vokra.provenance.model_id` / `vokra.provenance.source` (via
///   [`vokra_core::stamp_provenance`], so a `license` override at the
///   entry point flows to both the SPDX string and its derived
///   [`LicenseClass`] atomically — no split-brain).
///
/// `vokra.schema.version` and `vokra.schema.producer` are auto-added by
/// [`GgufBuilder`] at serialization time (no per-converter stamp).
fn write_metadata(b: &mut GgufBuilder, license_spdx: &str) {
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    let class = LicenseClass::from_license_str(license_spdx);
    vokra_core::stamp_provenance(b, class, license_spdx, Some(NAME), Some(UPSTREAM_HF));
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, chunks};

    /// Fresh per-test tempfile pair (unique enough via process id + tag).
    fn tmp_paths(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-baichuan-audio-{}-{}.safetensors",
            tag,
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-baichuan-audio-{}-{}.gguf",
            tag,
            std::process::id()
        ));
        (input, output)
    }

    /// Writes a synthetic single-BF16-tensor safetensors file at `path`
    /// carrying the caller-supplied raw payload.
    fn write_safetensors_one_bf16(path: &Path, bf16_bytes: &[u8]) {
        let header = format!(
            r#"{{"decoder.embed.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(bf16_bytes);
        std::fs::write(path, &bytes).unwrap();
    }

    /// Writes a synthetic two-tensor safetensors file (F32 followed by
    /// F16, non-zero patterns so a silent widen / downcast can't
    /// round-trip trivially).
    fn write_safetensors_f32_and_f16(path: &Path) {
        // 24 bytes F32 (6 f32) @ [0..24) + 12 bytes F16 (6 f16) @ [24..36)
        let header = r#"{"decoder.a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"decoder.b.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[24,36]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // F16 bit patterns: 1.0, -1.0, 0.5, -0.5, 2.0, -2.0.
        for pattern in [0x3C00u16, 0xBC00, 0x3800, 0xB800, 0x4000, 0xC000] {
            bytes.extend_from_slice(&pattern.to_le_bytes());
        }
        std::fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input, output) = tmp_paths("bf16-passthrough");
        // Non-zero BF16 patterns so a silent widen / downcast at convert
        // time cannot round-trip trivially — the payload byte-identity
        // assert below only bites when the pattern carries information.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        write_safetensors_one_bf16(&input, &bf16);

        let report = convert_baichuan_audio_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1, "one tensor in header");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts ADR)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );

        // Round-trip through the emitted GGUF: dtype preserved, payload
        // byte-identical (Moshi's `assert_eq!(info.dtype, GgmlType::BF16,
        // "no convert-time widening")` posture).
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("decoder.embed.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2u64, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Task-mandated metadata.
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
            Some(LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        // Schema stamps auto-added by GgufBuilder — assert their presence.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be present (auto-stamped)"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be present (auto-stamped)"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let (input, output) = tmp_paths("f32-f16-passthrough");
        write_safetensors_f32_and_f16(&input);

        let report = convert_baichuan_audio_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors in header");
        assert_eq!(report.written, 2, "both float tensors passed through");
        assert_eq!(report.skipped_non_float, 0, "no non-float tensor in input");
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 tensor in input, subset counter must stay at Default 0"
        );

        // Both tensors survive with their dtypes and shapes preserved.
        let out_bytes = std::fs::read(&output).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let a = file
            .tensor_info("decoder.a.weight")
            .expect("F32 tensor present");
        assert_eq!(a.dtype, GgmlType::F32);
        assert_eq!(a.dimensions, vec![2u64, 3]);
        assert_eq!(file.tensor_bytes(a).len(), 24);
        let b = file
            .tensor_info("decoder.b.weight")
            .expect("F16 tensor present");
        assert_eq!(b.dtype, GgmlType::F16);
        assert_eq!(b.dimensions, vec![2u64, 3]);
        assert_eq!(file.tensor_bytes(b).len(), 12);

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}
