//! **kNN-VC (bshall/knn-vc)** — few-shot voice conversion via
//! WavLM speaker features + k-nearest-neighbours matching over a target
//! corpus + a HiFi-GAN prematched vocoder. Weight license: **MIT**.
//! Category: `vc`.
//!
//! Upstream reference: <https://huggingface.co/bshall/knn-vc>.
//!
//! This module is a Vokra-native converter that walks the upstream
//! safetensors checkpoint and emits every float tensor into a GGUF
//! **verbatim** (F32 / F16 / BF16 pass-through). BF16 is emitted as GGUF
//! type 30 (`GgmlType::BF16`) — the runtime widens BF16 → f32 losslessly
//! at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the top
//! 16 bits of an f32 — `bits << 16` is exact). The pattern mirrors
//! `qwen3_tts` / `vibevoice` / `voxcpm2`.
//!
//! # No ONNX (permanent)
//!
//! bshall/knn-vc is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in Vokra (whisper.cpp 型 self re-implementation,
//! CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for kNN-VC GGUFs — used by the runtime dispatch to
/// route into the WavLM + k-NN + HiFi-GAN pipeline. Intentionally
/// **distinct** from every sibling arch tag.
pub(crate) const ARCH: &str = "knn_vc";
/// `vokra.model.name` value written for the canonical bshall/knn-vc GGUF.
pub(crate) const NAME: &str = "knn-vc";
/// `vokra.model.category` — top-level model class (`"vc"` — voice
/// conversion).
pub(crate) const CATEGORY: &str = "vc";
/// Upstream HuggingFace path stamped into `vokra.provenance.upstream_hf`.
const UPSTREAM_HF: &str = "bshall/knn-vc";
/// Default weight license (SPDX) — MIT end-to-end per the upstream
/// LICENSE + model card.
const DEFAULT_LICENSE: &str = "mit";

// --- vokra.knn_vc.* metadata keys ---------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
// `vokra.provenance.license` is written by [`stamp_provenance`] itself
// (see `vokra-core::compliance::stamp_provenance`), so this converter
// does not repeat that key here — override happens by passing the caller
// -supplied SPDX string through the same helper.

/// Outcome of a kNN-VC conversion.
///
/// Mirrors [`super::qwen3_tts::Qwen3TtsReport`] counter shape (the SoTA
/// plan Phase 3 pattern) plus a `read` counter for the total number of
/// safetensors entries walked (BF16 pass-through campaign 2026-07-25).
#[derive(Debug, Default)]
pub struct KnnVcReport {
    /// Total number of safetensors entries walked (float + non-float).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path per the BF16 pass-through ADR,
    /// mirror of `qwen3_tts` / `moshi` / `voxtral` / `vibevoice`).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any tensor
    /// reaching this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `vokra-core::gguf::quant::decode_bf16`
    /// (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Converts a bshall/knn-vc safetensors checkpoint at `input` into a GGUF
/// written to `output`, returning a counter report.
///
/// `license` optionally overrides the raw SPDX string written to
/// `vokra.provenance.license`. When `None`, the default MIT SPDX string
/// is stamped (matches the upstream release).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// safetensors name (the CSM / Kokoro / CosyVoice2 / Chatterbox /
/// Qwen3-TTS / VoxCPM / VibeVoice contract) — no convert-time widening.
pub fn convert_knn_vc_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<KnnVcReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. The default is MIT (bshall/knn-vc `LICENSE` file + HF
    // model card `license: mit`); the caller may override at the
    // `convert_file --license <spdx>` boundary (memory
    // [[project-huggingface-vokra-publication]]) when the same
    // architecture ships with a differently-licensed retrained weight.
    // `LicenseClass::from_license_str` (`vokra-core::compliance::
    // license_class::from_license_str`) normalises the SPDX string
    // and classifies fail-closed to `Unknown` on anything unrecognised,
    // so an override never silently upgrades a stricter licence to a
    // permissive one.
    let raw_license = license.unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(raw_license);
    vokra_core::stamp_provenance(
        &mut b,
        class,
        raw_license,
        Some(NAME),
        Some("bshall/knn-vc (MIT end-to-end)"),
    );

    let mut report = KnnVcReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30); the runtime widens BF16 → f32
    // exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` (`crates/vokra-convert/src/models/qwen3_tts.rs`).
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
    use vokra_core::gguf::GgufFile;

    /// Builds a synthetic single-tensor safetensors buffer with a
    /// caller-supplied dtype tag + raw payload. Header entry is
    /// `"knn_vc.embed.weight"` — a name close enough to the WavLM /
    /// HiFi-GAN scaffold that a future real-weight walk finds it under
    /// the same key.
    fn safetensors_one(dtype: &str, shape: &[u64], bytes: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"knn_vc.embed.weight":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bytes);
        out
    }

    /// Builds a two-tensor safetensors buffer covering the F32 leg
    /// (`knn_vc.embed.weight`) and the F16 leg
    /// (`knn_vc.norm.weight`) in one file.
    fn safetensors_f32_and_f16(f32_bytes: &[u8], f16_bytes: &[u8]) -> Vec<u8> {
        let f32_end = f32_bytes.len();
        let f16_end = f32_end + f16_bytes.len();
        // The F32 tensor has shape [1,2] → 8 bytes; F16 has shape [3] →
        // 6 bytes. The test caller supplies both payloads.
        let header = format!(
            r#"{{"knn_vc.embed.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_end}]}},"knn_vc.norm.weight":{{"dtype":"F16","shape":[3],"data_offsets":[{f32_end},{f16_end}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// Allocates unique temporary paths for `input` and `output` under
    /// `std::env::temp_dir()` (the moshi / streaming-convert pattern —
    /// `crates/vokra-convert/src/models/moshi.rs:788-798`).
    fn temp_pair(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-knn_vc-{tag}-in-{}.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-knn_vc-{tag}-out-{}.gguf",
            std::process::id()
        ));
        (input, output)
    }

    /// Pins BF16 pass-through end-to-end through the file-level entry
    /// point: the tensor lands in the output GGUF as `GgmlType::BF16`
    /// with a **byte-identical** payload (no silent widen / downcast).
    ///
    /// This is the RED-phase test for STEP 1: with `unimplemented!()` in
    /// the body, the test panics; STEP 2 turns the assertions green.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Build a BF16 payload with known non-zero bit patterns so a
        // subsequent byte-identity assert catches any silent widen /
        // downcast attempt (a zeroed payload would round-trip trivially
        // through an accidental F32 / F16 widen).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let blob = safetensors_one("BF16", &[2, 3], &bf16);

        let (input, output) = temp_pair("bf16");
        std::fs::write(&input, &blob).expect("write input");
        let report =
            convert_knn_vc_file(&input, &output, None).expect("convert_knn_vc_file must succeed");
        let out_bytes = std::fs::read(&output).expect("read output");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        // Counter surface: BF16 lands in the pass-through arm and
        // increments both `written` and `bf16_passthrough`; nothing
        // reaches `skipped_non_float`.
        assert_eq!(report.read, 1, "one safetensors entry walked");
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        // GGUF round-trip: dtype preserved as GgmlType::BF16, payload
        // byte-identical to input (no convert-time widening).
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("knn_vc.embed.weight")
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
    }

    /// Pins the F32 + F16 pass-through arm: two float tensors of
    /// different dtypes land in `written` (=2) and both preserve their
    /// dtypes end-to-end; `bf16_passthrough` stays at the `Default` 0
    /// so the additive BF16 counter cannot silently contaminate the
    /// F32 / F16 legs.
    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Three F16 patterns: 1.0, 2.0, 3.0.
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let blob = safetensors_f32_and_f16(&f32_bytes, &f16_bytes);

        let (input, output) = temp_pair("f32f16");
        std::fs::write(&input, &blob).expect("write input");
        let report = convert_knn_vc_file(&input, &output, None).expect("convert must succeed");
        let out_bytes = std::fs::read(&output).expect("read output");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        assert_eq!(report.read, 2, "two safetensors entries walked");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "no BF16 in the input — the additive counter must stay at Default 0"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let f32_info = file
            .tensor_info("knn_vc.embed.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(f32_info.dimensions, vec![1, 2]);
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("knn_vc.norm.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(f16_info.dimensions, vec![3]);
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());
    }
}
