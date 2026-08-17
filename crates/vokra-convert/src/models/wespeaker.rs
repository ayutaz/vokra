//! **WeSpeaker** (`Wespeaker/wespeaker-voxceleb-resnet34-LM`, apache-2.0):
//! safetensors → GGUF conversion (SoTA follow-on, 2026-07-25).
//!
//! Input: the upstream `Wespeaker/wespeaker-voxceleb-resnet34-LM` release
//! — a ResNet34 speaker-embedding network trained on VoxCeleb with the
//! Large-Margin (LM) fine-tuning stage. Output: a GGUF carrying every
//! float tensor verbatim under its upstream safetensors name, plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native WeSpeaker loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `Wespeaker/wespeaker-voxceleb-resnet34-LM` (recorded
//!   under `vokra.provenance.upstream_hf`).
//! - SPDX: `apache-2.0` (`LicenseClass::Permissive`).
//! - Model category: `speaker` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters
//! listed above. No convert-time widening; runtime widens BF16 → f32
//! losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec contract). Real-weight binding is a follow-up
//! wave gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 / BF16 tensor through unchanged so
//! a future `WespeakerWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream `wespeaker` Python pipeline
//! is deferred to owner (`docs/license-audit.md` §3.1 sign-off) — this
//! converter provides the byte-parallel GGUF surface only.
//!
//! # No ONNX (permanent)
//!
//! WeSpeaker is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/wespeaker/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for WeSpeaker GGUFs.
pub(crate) const ARCH: &str = "wespeaker";

/// `vokra.model.name` value written for the canonical WeSpeaker GGUF.
pub(crate) const NAME: &str = "wespeaker-voxceleb-resnet34-lm";

/// Model-category tag written under `vokra.model.category`. `"speaker"`
/// distinguishes speaker-embedding / speaker-verification networks from
/// TTS / ASR / codec / vocoder siblings so downstream consumers can
/// pick a load path without inspecting the arch.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "speaker";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the free-text
/// `vokra.provenance.source`. The slug preserves upstream casing
/// (`Wespeaker/wespeaker-voxceleb-resnet34-LM`).
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const UPSTREAM_HF: &str = "Wespeaker/wespeaker-voxceleb-resnet34-LM";

/// Outcome of a WeSpeaker conversion.
///
/// Mirrors [`crate::models::qwen3_tts::Qwen3TtsReport`]'s counter set
/// (float pass-through + BF16 subset counter + non-float defensive
/// counter), plus a leading `read` count of every tensor observed in
/// the input safetensors header. `read == written + skipped_non_float`
/// is an invariant preserved by [`convert_wespeaker_file`].
#[derive(Debug, Default)]
pub struct WespeakerReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `qwen3_tts` /
    /// `vibevoice` / `voxcpm2` / `neucodec` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Mirrors
    /// `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
    /// `vibevoice::VibeVoiceReport::bf16_passthrough`.
    pub bf16_passthrough: usize,
}

/// File-based WeSpeaker converter (`vokra-cli convert --model wespeaker`).
///
/// Reads `input` (upstream
/// `Wespeaker/wespeaker-voxceleb-resnet34-LM` `model.safetensors`),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `apache-2.0` provenance stamp (Whisper / kokoro-family override
/// pattern — see `convert_file_licensed` in `lib.rs`); pass `None` to
/// keep the built-in `apache-2.0` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_wespeaker_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WespeakerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a load path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream
    // `Wespeaker/wespeaker-voxceleb-resnet34-LM` model card).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("apache-2.0".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "Wespeaker/wespeaker-voxceleb-resnet34-LM \
             (ResNet34 speaker encoder, VoxCeleb + Large-Margin, apache-2.0)",
        ),
    );

    let mut report = WespeakerReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as qwen3_tts /
    // vibevoice / voxcpm2 / neucodec; runtime widens BF16 → f32 exactly
    // at load via `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is
    // exact).
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            f32_elems as usize * 4,
            "F32 payload len must match shape × 4"
        );
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(
            f16_bytes.len(),
            f16_elems as usize * 2,
            "F16 payload len must match shape × 2"
        );
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
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-wespeaker-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror an actual upstream ResNet34 tensor name (e.g.
        // `speaker.resnet.layer1.0.conv1.weight`) so the round-trip
        // exercises a realistic string, not a synthetic one.
        let input_bytes =
            safetensors_one_bf16("speaker.resnet.layer1.0.conv1.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_wespeaker_file(&input_path, &output_path, None)
            .expect("convert_wespeaker_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2)"
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
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("speaker.resnet.layer1.0.conv1.weight")
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

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0 = 0x3C00, -2.0 = 0xC000,
        // -0.5 = 0xB800, 3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        // Six values for a [2,3] tensor = 12 bytes.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12, "6 elements × 2 bytes F16 payload");

        let input_bytes = safetensors_f32_then_f16(
            "speaker.dense.weight",
            &[1, 2],
            &f32_bytes,
            "speaker.embed.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_wespeaker_file(&input_path, &output_path, None)
            .expect("convert_wespeaker_file must accept a mixed F32/F16 checkpoint");

        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Round-trip carries both tensors with their dtypes preserved
        // AND the arch / provenance / category stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("speaker.dense.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("speaker.embed.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Provenance / category chunks landed (task-spec pins).
        use vokra_core::LicenseClass;
        use vokra_core::gguf::chunks;
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
