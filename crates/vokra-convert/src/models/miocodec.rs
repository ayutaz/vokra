//! **MioCodec** (`Aratako/MioCodec-25Hz-44.1kHz-v2`, MIT):
//! safetensors → GGUF conversion (hf-audio-gap-comprehensive-2026-07-30
//! §3.8 JA-vocoder complement wave, 2026-08-04).
//!
//! Input: the upstream `Aratako/MioCodec-25Hz-44.1kHz-v2` release
//! (`huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2`) — a
//! single-safetensors JA-focused 25 Hz / 44.1 kHz speech codec
//! (~132 M F32 params, ~528 MB on-disk, `pipeline: audio-to-audio`,
//! HF cardData `license: mit`). Fine-tuned from
//! `Aratako/MioCodec-25Hz-24kHz` on multilingual speech corpora
//! (`sarulab-speech/mls_sidon` + `mythicinfinity/Libriheavy-HQ` +
//! `nvidia/hifitts-2`), 11-language coverage
//! (`en / ja / nl / fr / de / it / pl / pt / es / ko / zh`,
//! arXiv:2507.21138). Output: a Vokra GGUF carrying every float
//! tensor plus the `vokra.model.*` / `vokra.provenance.*` metadata
//! chunks the future native MioCodec runtime side will read.
//!
//! # Model card
//!
//! - **HF path**: `Aratako/MioCodec-25Hz-44.1kHz-v2`
//! - **License SPDX**: `mit` (weight + code, end-to-end)
//! - **Category**: `codec` — MioCodec is an audio-to-audio speech
//!   codec / tokenizer (waveform → discrete tokens → waveform)
//!   trained on multilingual speech; JA is a first-class training
//!   language alongside 10 other tongues. Complements the existing
//!   Kokoro / piper-plus JA vocoder stack per the hf-audio-gap
//!   audit `docs/handoff/hf-audio-gap-comprehensive-2026-07-30.md`
//!   §3.8.
//! - **Base model**: `Aratako/MioCodec-25Hz-24kHz` (fine-tune source)
//! - **Distinct arch tag**: `miocodec` — silently sharing an arch tag
//!   with a sibling codec (`mimi` / `dac` / `wavtokenizer` /
//!   `xcodec2` / `neucodec` / `bicodec` / `funcodec` /
//!   `speechtokenizer` / `focalcodec` / `xy_tokenizer` / `snac` /
//!   `step_audio2_mini`) would mis-route the runtime dispatch —
//!   MioCodec is Aratako's own codec design distinct from every
//!   existing RVQ / FSQ / SoundStream / focal-modulation family
//!   (FR-EX-08).
//!
//! # BF16 posture
//!
//! Follows the bicodec / neucodec / focalcodec / xcodec2 landed
//! pattern: F32 / F16 / BF16 all pass through **verbatim** under
//! their upstream safetensors names. BF16 is emitted as GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). No convert-
//! time widening, no silent F16 downcast (FR-EX-08). Upstream v2 is
//! F32 (HF API-verified 2026-08-04: `"safetensors": {"parameters":
//! {"F32": 132016399}}`); the BF16 arm is defensive today for future
//! BF16-quantized derivatives.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the neucodec / bicodec / focalcodec / xcodec2 contract). Real-
//! weight binding is a follow-up wave gated on the upstream tensor-
//! name manifest fetch; this converter passes every F32 / F16 / BF16
//! tensor through unchanged so a future
//! `MioCodecWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Deferred to the owner sign-off queue (`docs/license-audit.md`
//! §3.1). This converter is a native-side skeleton that pins the
//! metadata contract (arch / name / category / upstream-HF / license)
//! plus the BF16 pass-through invariant so a future
//! `MioCodec::from_gguf` can bind against the same upstream tensor
//! names once real weights are audited.
//!
//! # No ONNX (permanent)
//!
//! MioCodec ships `model.safetensors` + `config.yaml` directly
//! (no torch-pickle prepare step, no ONNX mirror); this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/miocodec/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

// Skeleton-only allowance: the public API
// (`convert_miocodec_file`, `MioCodecReport`, `KEY_*` /
// `MODEL_CATEGORY` / `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX`) is
// exercised by the in-module tests + lib.rs `convert_file` dispatch;
// this attribute is removed once the runtime
// `MioCodecWeights::from_gguf` binding lands and starts consuming
// the constants directly.
#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MioCodec GGUFs — kept intentionally distinct
/// from every sibling codec so the runtime dispatch cannot silently
/// route a MioCodec artifact through a DAC / Mimi / RVQ / FSQ /
/// SoundStream / focal-modulation decoder.
pub const ARCH: &str = "miocodec";

/// `vokra.model.name` value for the canonical
/// `Aratako/MioCodec-25Hz-44.1kHz-v2` release. Matches the publish
/// repo slug spelling (`vokra/miocodec-25hz-44khz-v2` — HF repo naming
/// = dashes only, lowercase, dots stripped from `44.1` → `44khz`).
pub const NAME: &str = "miocodec-25hz-44khz-v2";

/// `vokra.model.category` key — codec bucket for the artifact.
///
/// Kept as a local constant rather than a `vokra_core::chunks::*`
/// re-export because it is not yet part of the shared
/// `vokra-core::chunks` surface (mirrors the bicodec / neucodec /
/// focalcodec convention).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Category string written into [`KEY_MODEL_CATEGORY`].
pub const MODEL_CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` key — HF repo path of the upstream
/// weight.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Value written under [`KEY_PROVENANCE_UPSTREAM_HF`]. Preserves the
/// upstream capitalization + dot in `44.1kHz` — the HF repo slug is
/// case-sensitive and the primary source for the model-card generator.
pub const UPSTREAM_HF: &str = "Aratako/MioCodec-25Hz-44.1kHz-v2";

/// Default weight-license SPDX. Verified 2026-08-04 via HF cardData
/// API primary source (`api/models/Aratako/MioCodec-25Hz-44.1kHz-v2`
/// → `license: mit`). May be overridden via the `license` argument to
/// [`convert_miocodec_file`] (the whisper / kokoro / neucodec
/// override pattern).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

/// Advisory source note stamped alongside the license into the
/// `vokra.provenance.source` chunk.
const PROVENANCE_SOURCE_NOTE: &str = "Aratako/MioCodec-25Hz-44.1kHz-v2 (JA-focused 25 Hz / 44.1 kHz \
     multilingual speech codec, MIT end-to-end; base \
     Aratako/MioCodec-25Hz-24kHz)";

/// Outcome of a MioCodec conversion.
///
/// Mirrors the field set on the sibling BF16-pass-through converters
/// (`super::bicodec::BicodecReport`,
/// `super::neucodec::NeucodecReport`,
/// `super::focalcodec::FocalcodecReport`) — the `read` counter lets
/// the caller distinguish a zero-tensor safetensors file from a
/// zero-write outcome caused by every tensor being quantized
/// (defensive — the safetensors reader rejects unknown dtypes at
/// parse time today, so anything reaching the non-float arm is a real
/// regression on that reader).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MioCodecReport {
    /// Total upstream tensors observed in the safetensors input.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three
    /// go through the same byte-copy arm per the accepted BF16 pass-
    /// through posture).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time;
    /// anything that reaches this arm is a quantized dtype the runtime
    /// is not expected to consume).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens
    /// BF16 → f32 losslessly at load via `decode_bf16` (`bits << 16`
    /// is exact — BF16 is the top 16 bits of an f32).
    pub bf16_passthrough: usize,
}

/// Convert an `Aratako/MioCodec-25Hz-44.1kHz-v2` safetensors
/// checkpoint into a Vokra GGUF.
///
/// `input` is the upstream `model.safetensors` path; the emitted GGUF
/// is written to `output`. `license` overrides the raw SPDX string
/// stamped into `vokra.provenance.license` — the default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`), matching the Aratako weight
/// card at `huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2`. Pass
/// `Some(other_spdx)` when the immediate redistribution source has
/// re-tagged the artifact (mirror of the neucodec / bicodec /
/// focalcodec override pattern).
///
/// # Errors
///
/// - I/O reading `input` or writing `output` propagates as
///   [`ConvertError::Io`].
/// - Safetensors parse failure propagates as [`ConvertError::Parse`].
/// - GGUF serialization failure propagates as the `From<GgufError>`
///   impl on `ConvertError`.
pub fn convert_miocodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MioCodecReport, ConvertError> {
    // MioCodec v2 is ~528 MB single-file safetensors (132M F32
    // params, HF-verified 2026-08-04). Well within the sibling non-
    // streaming BF16 pass-through posture (~1 order of magnitude
    // smaller than the streaming-mandated Moshi 14 GiB tier that
    // requires the `MappedTextBlocks` / `restamp_provenance` mmap
    // path), so a plain `std::fs::read` is safe on M1 iMac 16 GB.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // Self-describing redistribution: the artifact carries its own
    // licence. The `license` param overrides the raw SPDX string
    // (`vokra.provenance.license`) and — when overridden — re-derives
    // the class through `LicenseClass::from_license_str` so the
    // compliance gate stays honest (a caller who overrides to a
    // non-permissive SPDX would otherwise get a silent Permissive
    // verdict). `None` keeps the Aratako default (mit → Permissive)
    // that matches the upstream weight card.
    let license_spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = match license {
        Some(_) => LicenseClass::from_license_str(license_spdx),
        None => LicenseClass::Permissive,
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        license_spdx,
        Some(NAME),
        Some(PROVENANCE_SOURCE_NOTE),
    );

    let mut report = MioCodecReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the bicodec /
    // neucodec / focalcodec ADR-A_passthrough posture; the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
    // (`bits << 16` is exact — BF16 is the top 16 bits of an f32).
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

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Panics if
    /// `bf16_bytes.len() != shape × 2` — that would declare an invalid
    /// safetensors header the reader would reject.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
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
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Builds a single-F32-tensor safetensors buffer — matches upstream
    /// MioCodec v2 dtype (F32 verified via HF API 2026-08-04).
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            elems as usize * 4,
            "test fixture: payload len must match shape × 4 F32"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out
    }

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-miocodec-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// Upstream MioCodec-25Hz-44.1kHz-v2 is F32 (HF API verified
    /// 2026-08-04) — this test pins the primary code path.
    #[test]
    fn f32_tensor_passes_through_and_stamps_land() {
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from a speech codec's
        // encoder stack (bicodec / neucodec convention).
        let input_bytes = safetensors_one_f32(
            "codec.encoder.conv_layers.0.conv.weight",
            &[2, 3],
            &f32_bytes,
        );
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_miocodec_file(&input_path, &output_path, None)
            .expect("convert_miocodec_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("codec.encoder.conv_layers.0.conv.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category chunks landed.
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
            Some(DEFAULT_LICENSE_SPDX)
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
            Some(MODEL_CATEGORY),
            "vokra.model.category must be `codec`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Defensive test — future BF16-quantized derivatives should ride
    /// the same arm as the sibling BF16-pass-through converters
    /// (bicodec / neucodec / focalcodec / xcodec2).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity
        // assert catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        let input_bytes = safetensors_one_bf16("codec.quantizer.embed", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_miocodec_file(&input_path, &output_path, None).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("codec.quantizer.embed")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// The `license` override must flow through to the provenance stamp
    /// and re-derive the class (guards against a silent Permissive
    /// verdict when a caller ships under a non-permissive SPDX).
    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("codec.encoder.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_miocodec_file(&input_path, &output_path, Some("apache-2.0"))
            .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "apache-2.0 is Permissive class (same as MIT)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
