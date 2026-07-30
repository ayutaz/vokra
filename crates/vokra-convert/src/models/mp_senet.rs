//! **MP-SENet** (JacobLinCool/MP-SENet-DNS, mit) — safetensors → GGUF
//! conversion (dual-branch magnitude-phase speech-enhancement U-Net).
//!
//! # Provenance
//!
//! - **HF path**: `JacobLinCool/MP-SENet-DNS` (safetensors + a PyTorch
//!   pipeline; the upstream lineage is `yxlu0057/MP-SENet` on GitHub).
//! - **License (SPDX)**: `mit` — permissive; the JacobLinCool DNS-tuned
//!   re-release inherits the base `yxlu0057/MP-SENet` MIT LICENSE.
//! - **Category**: `denoise` (per implementer spec — MP-SENet is a
//!   single-channel speech enhancement / noise-suppression model, not a
//!   source-separation family).
//! - **Attribution**: none required by license (MIT permits
//!   redistribution without runtime-side attribution obligation).
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2 / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. The observability
//! counter [`MpSenetReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot slip
//! in undetected.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM
//! contract). Real-weight parity + a native `MpSenet::from_gguf` forward
//! path are deferred to owner sign-off (`docs/license-audit.md` §3.1) —
//! this converter provides the byte-parallel GGUF surface only. The
//! internal dual-branch (magnitude + phase) U-Net topology is
//! intentionally NOT re-implemented on this pass: transcribing that
//! architecture from the paper (arXiv:2305.13686) + upstream
//! `yxlu0057/MP-SENet` code is a `loud-partial` sibling wave.
//!
//! # No ONNX (permanent)
//!
//! MP-SENet is distributed as safetensors + PyTorch; this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/mp_senet/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MP-SENet GGUFs. Distinct from every other
/// denoise / enhancement sibling (`denoise` = DeepFilterNet3, TIGER =
/// `tiger_separator`) because the topology + I/O contract differs
/// (MP-SENet operates on the STFT magnitude + phase directly, unlike
/// DFN3's spectrum masking).
pub const ARCH: &str = "mp_senet";

/// `vokra.model.name` value written for the canonical DNS-tuned
/// release.
pub const NAME: &str = "mp-senet-dns";

/// `vokra.model.category` value — a downstream that speaks the
/// denoise API picks this without inspecting the arch.
pub const CATEGORY: &str = "denoise";

/// `vokra.provenance.upstream_hf` slug (`org/name`).
pub const UPSTREAM_HF: &str = "JacobLinCool/MP-SENet-DNS";

/// Default upstream weight license (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of an MP-SENet conversion. Mirrors the sibling converters'
/// counter shape (`ecapa_tdnn::EcapaTdnnReport`,
/// `wespeaker::WespeakerReport`, `neucodec::NeucodecReport`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MpSenetReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter.
    pub bf16_passthrough: usize,
}

/// File-based MP-SENet converter.
///
/// Reads `input` (upstream `JacobLinCool/MP-SENet-DNS`
/// `model.safetensors`), writes a Vokra GGUF to `output` carrying
/// every F32 / F16 / BF16 tensor verbatim under its upstream
/// safetensors name + the `vokra.model.*` + `vokra.provenance.*`
/// metadata chunks.
///
/// `license` optionally overrides the default `mit` provenance stamp
/// (same override pattern as `wespeaker::convert_wespeaker_file`).
/// `None` keeps the built-in `mit` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_mp_senet_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MpSenetReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "JacobLinCool/MP-SENet-DNS \
             (Magnitude-Phase Speech Enhancement Net, DNS-tuned, mit)",
        ),
    );

    let mut report = MpSenetReport::default();
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
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
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

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-mp-senet-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// BF16 round-trip byte-identity + every stamp lands.
    #[test]
    fn bf16_round_trips_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast could not
        // round-trip trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("mag_encoder.conv0.weight", &[2, 3], &bf16);
        let input = write_temp("bf16-in", &input_bytes);
        let output = write_temp("bf16-out", &[]);

        let report = convert_mp_senet_file(&input, &output, None).expect("convert MP-SENet");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("mag_encoder.conv0.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical (no silent widen)"
        );

        // Every stamp lands.
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
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 pass through with the BF16 counter staying at 0.
    #[test]
    fn f32_and_f16_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 4] = [0x3C00, 0xC000, 0xB800, 0x4200];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"mag.a":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_len}]}},"phase.b":{{"dtype":"F16","shape":[2,2],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        let input = write_temp("mixed-in", &input_bytes);
        let output = write_temp("mixed-out", &[]);

        let report = convert_mp_senet_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Override reaches the artifact — `apache-2.0` string lands
    /// (upstream is MIT so the class shift stays Permissive).
    #[test]
    fn license_override_reaches_the_artifact() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("mag.w", &[1, 2], &bf16);
        let input = write_temp("license-in", &input_bytes);
        let output = write_temp("license-out", &[]);

        convert_mp_senet_file(&input, &output, Some("apache-2.0")).expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
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

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
