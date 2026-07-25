//! **OpenVoice V2** (`myshell-ai/OpenVoiceV2`, MIT weight, category `vc`):
//! safetensors checkpoint → GGUF conversion.
//!
//! Input: the upstream `myshell-ai/OpenVoiceV2` release — a zero-shot
//! voice-conversion model requiring only ~10 s of reference audio to
//! clone a speaker's timbre. Output: a GGUF carrying every float tensor
//! (F32 / F16 / BF16 verbatim) plus the `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks needed for provenance tracking.
//!
//! # Category — voice conversion (VC)
//!
//! OpenVoice V2 is a **VC** model (`vokra.model.category = "vc"`), the
//! same class as RVC v2 / GPT-SoVITS. Per CLAUDE.md 設計判断 8 (ELVIS
//! Act / NO FAKES Act), voice-cloning targets belong in the
//! `vokra-voiceclone-experimental` **separate** repository, not in the
//! core vokra distribution. This converter is the entry point that
//! **prepares** the GGUF; the actual runtime consumer + distribution
//! flow live behind the voiceclone-experimental research-flag gate
//! (FR-CP-03 / M2-13).
//!
//! # BF16 pass-through
//!
//! Every F32 / F16 / BF16 tensor is emitted verbatim; BF16 lands on
//! GGUF type 30 (`GgmlType::BF16`) with no convert-time widening. The
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
//! top 16 bits of an f32 — `bits << 16` is exact). Mirror of the
//! qwen3-tts / vibevoice / voxcpm2 / moshi / voxtral pass-through arm,
//! per the ADR `docs/adr/qwen3-tts-bf16.md` (strategy A_passthrough,
//! Accepted 2026-07-25).
//!
//! # License — MIT (self-describing)
//!
//! Default weight license = **MIT** end-to-end
//! (`huggingface.co/myshell-ai/OpenVoiceV2` model card `license: mit`,
//! fetched 2026-07-25 — CLAUDE.md「ハルシネーション厳禁」). MIT is a
//! `Permissive` license class — same commercial verdict as apache-2.0
//! (no runtime-side attribution obligation), just a different SPDX
//! string. The `license: Option<&str>` parameter lets a caller override
//! the stamped license when the actual distribution source declares a
//! different SPDX (the `convert_file_licensed` posture: publishing must
//! state the license of the artifact being redistributed).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice
//! / VoxCPM contract). Real-weight parity is a follow-up wave gated on
//! the owner § 3.1 sign-off (docs/license-audit.md); this converter
//! passes every F32 / F16 / BF16 tensor through unchanged so a future
//! `OpenVoiceV2Weights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! OpenVoice V2 is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively downstream (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Scaffold posture (`#![allow(dead_code)]`)
//!
//! This module is a **skeleton** landing: [`convert_openvoice_v2_file`]
//! is public within the crate but not yet re-exported from `lib.rs` and
//! not yet dispatched from a `ModelKind::OpenvoiceV2` arm of
//! `convert_file_licensed`. Because `mod models;` is private, the
//! compiler cannot reach these items from outside the crate and marks
//! them "never used" under `-D warnings` even though the in-module
//! `#[cfg(test)]` block exercises them.
//!
//! The follow-up wave adds the `ModelKind::OpenvoiceV2` variant + the
//! `convert_openvoice_v2_file` re-export at the crate root + a
//! voiceclone-experimental gate hook (this converter's category is
//! `"vc"`, so distribution is gated per CLAUDE.md 設計判断 8); when
//! that wave lands, drop this `#![allow(dead_code)]` in the same commit
//! so any future item that goes unused in this module is caught again.

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for OpenVoice V2 GGUFs — kept distinct from every
/// sibling arch tag so silently sharing does not mis-route the runtime
/// dispatch.
pub(crate) const ARCH: &str = "openvoice_v2";
/// `vokra.model.name` value written for the canonical OpenVoice V2 GGUF.
pub(crate) const NAME: &str = "openvoice-v2";
/// `vokra.model.category` — voice conversion (VC) per CLAUDE.md モデル
/// 表. Silently sharing with TTS / ASR / VAD categories would misroute
/// the voiceclone-experimental gate at distribution time.
pub(crate) const CATEGORY: &str = "vc";
/// Upstream HuggingFace slug — recorded on
/// `vokra.provenance.upstream_hf` so the GGUF is self-describing about
/// where its tensors came from.
pub(crate) const UPSTREAM_HF: &str = "myshell-ai/OpenVoiceV2";
/// Default weight license SPDX (MIT — verified from the HF model card,
/// 2026-07-25). Overridable via the `license` parameter of
/// [`convert_openvoice_v2_file`].
pub(crate) const DEFAULT_LICENSE: &str = "mit";

/// Metadata key: upstream HuggingFace slug (raw string). Distinct from
/// `vokra.provenance.source` (which is a longer human-readable label);
/// this key is the machine-parseable HF path so a downstream tool can
/// re-fetch the weight without guessing.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// Metadata key: model category tag (`"vc"` / `"tts"` / `"asr"` /
/// `"vad"` / `"s2s"`). Enables downstream tools (model-zoo publisher,
/// voiceclone-experimental gate) to route by category without parsing
/// the arch tag.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Outcome of an OpenVoice V2 conversion.
///
/// Mirrors [`crate::models::qwen3_tts::Qwen3TtsReport`] plus a `read`
/// counter (total tensors visited, including non-float ones counted in
/// `skipped_non_float`) so the caller can distinguish "empty
/// safetensors" from "all-non-float safetensors" without inspecting
/// individual tensors.
#[derive(Debug, Default)]
pub struct OpenvoiceV2Report {
    /// Total tensors visited (float + non-float).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped. The safetensors reader accepts only
    /// F32 / F16 / BF16 at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so a non-zero
    /// value here signals a reader change upstream. Kept for symmetry
    /// with the sibling converters.
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 → f32
    /// losslessly at load via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    pub bf16_passthrough: usize,
}

/// File-based converter: reads a safetensors file at `input`, emits a
/// GGUF file at `output`, and returns an [`OpenvoiceV2Report`].
///
/// `license` overrides the default `mit` stamp (SPDX raw string; the
/// class is re-derived via [`vokra_core::LicenseClass::from_license_str`]).
/// Pass `None` to keep the built-in `mit` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] on read/write failure, [`ConvertError::Parse`]
/// on a malformed safetensors buffer, and any GGUF-writer failure.
pub fn convert_openvoice_v2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<OpenvoiceV2Report, ConvertError> {
    // Whole-file read: OpenVoice V2 is small enough (~200 MB checkpoint)
    // that the streaming Moshi/Voxtral path (bounded-memory contract) is
    // not required. If a future variant grows past ~5 GB, switch to
    // `SafetensorsFileReader` + `GgufStreamWriter` per the moshi.rs
    // pattern.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Non-provenance identity chunks (task-required additive keys).
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // License override: `None` keeps the default MIT stamp; `Some(spdx)`
    // re-derives the class from the caller-supplied SPDX id.
    // `LicenseClass::from_license_str` fail-closes to `Unknown` on an
    // unrecognized string (a research-flag gate then trips at load
    // time), so no silent permissive default can slip in.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = OpenvoiceV2Report::default();
    // Float tensors pass through verbatim — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per ADR A_passthrough; the
    // runtime widens BF16 → f32 exactly at load via the single choke
    // point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    // Mirrors `qwen3_tts::convert` / `vibevoice::convert` /
    // `voxcpm2::convert` / `moshi::convert`.
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
    std::fs::write(output, out_bytes)?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile, chunks};

    /// Builds a per-test unique temp file path (process id + test tag +
    /// nanosecond counter) so two parallel tests never collide on the
    /// same file.
    fn temp_path(tag: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-openvoice-v2-{}-{}-{}.{}",
            std::process::id(),
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            ext,
        ));
        p
    }

    /// Builds a single-tensor safetensors byte buffer.
    fn safetensors_single(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
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

    /// Exercises the BF16 pass-through arm (per ADR A_passthrough): a
    /// BF16 tensor with non-zero patterns must round-trip byte-identical
    /// as GGUF type 30 (no convert-time widening).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Six non-zero BF16 patterns so a silent widen / downcast is
        // detectable at the byte level (a zeroed payload round-trips
        // trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16_bytes.len(), 12, "6 elements × 2 bytes BF16 payload");
        let st_bytes = safetensors_single("openvoice.embed", "BF16", &[2, 3], &bf16_bytes);

        let input = temp_path("bf16", "safetensors");
        let output = temp_path("bf16", "gguf");
        std::fs::write(&input, &st_bytes).expect("write input");

        let report = convert_openvoice_v2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        // Round-trip: tensor present, BF16 dtype preserved, payload
        // byte-identical (no silent widen).
        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse output");
        let info = file.tensor_info("openvoice.embed").expect("tensor present");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_bytes.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Provenance + category + upstream_hf stamps are present.
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
            Some(DEFAULT_LICENSE)
        );

        // Cleanup (never fail the test on cleanup errors).
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 and F16 tensors both pass through the union match arm; the
    /// BF16 counter stays at the Default 0.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Two tensors, ordered so the header's `data_offsets` are
        // strictly increasing (matches how upstream safetensors emitters
        // pack payloads).
        //   openvoice.f32 — F32, [1,2] → 8 bytes at [0..8)
        //   openvoice.f16 — F16, [2,3] → 12 bytes at [8..20)
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 payload: use a placeholder non-zero pattern so the readback
        // does not accidentally match a widen path (any 12 non-zero bytes
        // work here — F16 bit-exactness is out of scope for this leg).
        let f16_bytes: Vec<u8> = (1..=12u8).collect();

        let header = format!(
            r#"{{"openvoice.f32":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_end}]}},"openvoice.f16":{{"dtype":"F16","shape":[2,3],"data_offsets":[{f32_end},{f16_end}]}}}}"#,
            f32_end = f32_bytes.len(),
            f16_end = f32_bytes.len() + f16_bytes.len(),
        );
        let mut st_bytes = Vec::new();
        st_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        st_bytes.extend_from_slice(header.as_bytes());
        st_bytes.extend_from_slice(&f32_bytes);
        st_bytes.extend_from_slice(&f16_bytes);

        let input = temp_path("f32f16", "safetensors");
        let output = temp_path("f32f16", "gguf");
        std::fs::write(&input, &st_bytes).expect("write input");

        let report = convert_openvoice_v2_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors visited");
        assert_eq!(report.written, 2, "both F32 and F16 pass through");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 tensor in this fixture → counter at Default 0"
        );

        // Both tensors survive round-trip with their upstream names +
        // dtypes preserved.
        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse output");
        let f32_info = file
            .tensor_info("openvoice.f32")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("openvoice.f16")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Cleanup.
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
