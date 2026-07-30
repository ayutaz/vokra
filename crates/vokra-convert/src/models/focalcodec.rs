//! **FocalCodec** (`lucadellalib/focalcodec_50hz`, apache-2.0):
//! safetensors → GGUF conversion (SoTA plan Phase D6, 2026-07-30).
//!
//! Input: the upstream `lucadellalib/focalcodec_50hz` release — a
//! low-bitrate speech codec producing ~50 Hz single-codebook tokens
//! (arXiv:2502.04465). Unlike the sibling BigVGAN / HiFi-GAN vocoder
//! releases (torch pickle `.pt` + `config.json`), FocalCodec ships
//! `model.safetensors` + `config.json` **directly** so this converter
//! consumes safetensors natively without a prepare-checkpoint script.
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream safetensors name, plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native FocalCodec loader
//! will read.
//!
//! # Provenance
//!
//! - **HF path**: `lucadellalib/focalcodec_50hz` (recorded under
//!   `vokra.provenance.upstream_hf`).
//! - **License (SPDX)**: `apache-2.0` — verified 2026-07-30 via HF
//!   API cardData `license: apache-2.0`, CC-verified (CLAUDE.md
//!   「ハルシネーション厳禁」). Base model
//!   `microsoft/wavlm-large` is MIT (both compatible under
//!   `LicenseClass::Permissive`).
//! - **Category**: `codec` — audio codec (waveform → discrete tokens →
//!   waveform). Distinct from vocoder (mel → waveform only).
//!
//! # FocalCodec vs sibling codecs
//!
//! Distinct arch tag from every sibling codec (Mimi / DAC /
//! WavTokenizer / neucodec / step_audio2_mini / X-Codec 2 / FunCodec):
//! FocalCodec uses **focal modulation** based tokenization at 50 Hz
//! (single codebook, low-bitrate), distinct from every RVQ / FSQ /
//! SoundStream sibling. Silently sharing an arch tag would mis-route
//! the runtime dispatch.
//!
//! # BF16 pass-through (mirror of wespeaker / ecapa_tdnn / funcodec)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//! The upstream FocalCodec release is F32 (verified via HF API:
//! `"safetensors": {"parameters": {"I64": 13, "F32": 142128305}}`),
//! so the BF16 arm is defensive today; the counter stays for
//! future BF16-quantized derivative releases.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the FunCodec / X-Codec 2 / WeSpeaker contract). Real-weight parity
//! vs the upstream `focalcodec` Python reference is deferred to owner
//! (`docs/license-audit.md` §3.1 sign-off queue).
//!
//! # No ONNX (permanent)
//!
//! FocalCodec ships PyTorch safetensors + config.json; the converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in `crates/vokra-models/src/focalcodec/` when the codec
//! lands (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FocalCodec GGUFs.
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `neucodec`, `funcodec`, `xcodec2`,
/// `speechtokenizer`, `bicodec`, `xy_tokenizer`, `step_audio2_mini`)
/// because FocalCodec is a focal-modulation-based single-codebook
/// low-bitrate codec, not an RVQ / FSQ / SoundStream family member.
pub const ARCH: &str = "focalcodec";

/// `vokra.model.name` value written for the canonical
/// `lucadellalib/focalcodec_50hz` GGUF.
pub const NAME: &str = "focalcodec-50hz";

/// `vokra.model.category` value written for every FocalCodec GGUF.
pub const CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator.
pub const UPSTREAM_HF: &str = "lucadellalib/focalcodec_50hz";

/// Default upstream weight licence (SPDX). Verified 2026-07-30 via
/// HF API cardData `license: apache-2.0`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Outcome of a FocalCodec conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::funcodec::FuncodecReport`],
/// [`super::wespeaker::WespeakerReport`]) adapted to the
/// file-oriented `convert_focalcodec_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FocalcodecReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `lucadellalib/focalcodec_50hz` safetensors checkpoint
/// at `input` into a Vokra-native GGUF at `output`, returning a
/// [`FocalcodecReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// safetensors name; the `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf) chunks are stamped for the runtime compliance gate
/// (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`, `Permissive`) — the
/// upstream HF release ships apache-2.0.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_focalcodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FocalcodecReport, ConvertError> {
    // FocalCodec-50Hz is ~569 MB (142M params, F32) — still 1 order
    // of magnitude smaller than the streaming-mandated Moshi 14 GiB
    // tier, so the simple `std::fs::read` posture the sibling
    // non-streaming BF16 pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Default provenance stamp — Permissive apache-2.0 (upstream
    // `lucadellalib/focalcodec_50hz` model card verified 2026-07-30 via
    // HF API). The optional `license` argument overrides below.
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
            "lucadellalib/focalcodec_50hz (FocalCodec 50 Hz single-codebook \
             audio codec, apache-2.0; base wavlm-large is MIT)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = FocalcodecReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR; the runtime
    // widens BF16 → f32 exactly at load via the single choke point
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

    /// Builds a single-BF16-tensor safetensors buffer.
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected);
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

    /// Builds an F32 tensor safetensors buffer — matches upstream FocalCodec
    /// dtype (F32 verified via HF API 2026-07-30).
    fn safetensors_one_f32(name: &str, shape: &[u64], f32_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 4;
        assert_eq!(f32_bytes.len(), expected);
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

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-focalcodec-{kind}-{}-{}.bin",
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
    fn f32_tensor_passes_through_and_stamps_land() {
        // Upstream FocalCodec is F32 (HF API verified 2026-07-30) —
        // this test pins the primary code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from FocalCodec's
        // WavLM-based encoder (upstream module tree — codec.encoder.*).
        let input_bytes = safetensors_one_f32(
            "codec.encoder.feature_extractor.conv_layers.0.conv.weight",
            &[2, 3],
            &f32_bytes,
        );
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_focalcodec_file(&input_path, &output_path, None)
            .expect("convert_focalcodec_file must accept F32 checkpoint");
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
            .tensor_info("codec.encoder.feature_extractor.conv_layers.0.conv.weight")
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
            Some(CATEGORY),
            "vokra.model.category must be `codec`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (wespeaker / ecapa_tdnn / funcodec).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        let input_bytes = safetensors_one_bf16("codec.quantizer.embed", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_focalcodec_file(&input_path, &output_path, None).expect("convert");
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

    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("codec.encoder.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_focalcodec_file(&input_path, &output_path, Some("mit"))
            .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT is Permissive class (same as apache-2.0)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
