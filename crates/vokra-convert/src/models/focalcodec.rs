//! **FocalCodec** (`lucadellalib/focalcodec_{50hz,25hz,12_5hz}`,
//! apache-2.0): safetensors → GGUF conversion (SoTA plan Phase D6,
//! 2026-07-30; 25Hz / 12.5Hz variants added 2026-07-31).
//!
//! Input: an upstream `lucadellalib/focalcodec_50hz` /
//! `focalcodec_25hz` / `focalcodec_12_5hz` release — low-bitrate
//! speech codecs producing ~50 / 25 / 12.5 Hz single-codebook tokens
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
//! - **HF paths** (three variants share this single converter):
//!   - `lucadellalib/focalcodec_50hz`   (Hz50, canonical/default)
//!   - `lucadellalib/focalcodec_25hz`   (Hz25, ~577 MB F32)
//!   - `lucadellalib/focalcodec_12_5hz` (Hz12_5, ~581 MB F32)
//! - **License (SPDX)**: `apache-2.0` for all three variants —
//!   verified 2026-07-30 (50Hz) and 2026-07-31 (25Hz + 12.5Hz) via HF
//!   cardData API `license: apache-2.0`, CC-verified (CLAUDE.md
//!   「ハルシネーション厳禁」). Base model
//!   `microsoft/wavlm-large` is MIT (both compatible under
//!   `LicenseClass::Permissive`).
//! - **Category**: `codec` — audio codec (waveform → discrete tokens →
//!   waveform). Distinct from vocoder (mel → waveform only).
//! - **Variant tag** (`vokra.focalcodec.variant`): `"50hz"` /
//!   `"25hz"` / `"12_5hz"` so a consumer that needs to pick a specific
//!   frame rate can inspect this without parsing free-text
//!   `vokra.model.name` (mirrors `vokra.bigvgan.variant` +
//!   `vokra.tiger.variant`).
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

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for FocalCodec GGUFs. Shared across every
/// [`FocalcodecVariant`] — the topology is identical across `50hz` /
/// `25hz` / `12_5hz` (WavLM-Large encoder → FocalEncoder compressor →
/// BinarySphericalQuantizer 8192 → FocalDecoder → Vocos vocoder,
/// upstream `config.json` verified 2026-07-31); only the effective
/// token frame rate differs.
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `neucodec`, `funcodec`, `xcodec2`,
/// `speechtokenizer`, `bicodec`, `xy_tokenizer`, `step_audio2_mini`)
/// because FocalCodec is a focal-modulation-based single-codebook
/// low-bitrate codec, not an RVQ / FSQ / SoundStream family member.
pub const ARCH: &str = "focalcodec";

/// `vokra.model.name` value written for the canonical
/// `lucadellalib/focalcodec_50hz` GGUF (backward-compat alias — new
/// callers should use [`FocalcodecVariant::name`]).
#[allow(dead_code)]
pub const NAME: &str = "focalcodec-50hz";

/// `vokra.model.category` value written for every FocalCodec GGUF.
pub const CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` value for the canonical `50hz`
/// variant (backward-compat alias — new callers should use
/// [`FocalcodecVariant::upstream_hf`]).
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "lucadellalib/focalcodec_50hz";

/// Default upstream weight licence (SPDX). Verified 2026-07-30 (50Hz)
/// and 2026-07-31 (25Hz + 12.5Hz) via HF API cardData
/// `license: apache-2.0`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.focalcodec.variant`: `"50hz"` / `"25hz"` / `"12_5hz"`.
/// Consumers pick a specific frame-rate head without parsing free-text
/// `vokra.model.name` (mirrors [`super::bigvgan`] +
/// [`super::tiger`] discriminators).
pub const KEY_FOCALCODEC_VARIANT: &str = "vokra.focalcodec.variant";

/// Which FocalCodec release the caller is converting. Selects the
/// model name / upstream HF slug / variant tag written into the GGUF.
///
/// All three variants share [`ARCH`] `focalcodec` — the topology is
/// identical, only the effective token frame rate differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocalcodecVariant {
    /// `lucadellalib/focalcodec_50hz`: 50 Hz single-codebook tokens
    /// (canonical default, ~569 MB F32).
    /// `vokra.focalcodec.variant = "50hz"`.
    Hz50,
    /// `lucadellalib/focalcodec_25hz`: 25 Hz single-codebook tokens
    /// (~577 MB F32, verified 2026-07-31).
    /// `vokra.focalcodec.variant = "25hz"`.
    Hz25,
    /// `lucadellalib/focalcodec_12_5hz`: 12.5 Hz single-codebook
    /// tokens (~581 MB F32, verified 2026-07-31).
    /// `vokra.focalcodec.variant = "12_5hz"`.
    Hz12_5,
}

impl FocalcodecVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hz50 => "focalcodec-50hz",
            Self::Hz25 => "focalcodec-25hz",
            Self::Hz12_5 => "focalcodec-12-5hz",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// release — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Hz50 => "lucadellalib/focalcodec_50hz",
            Self::Hz25 => "lucadellalib/focalcodec_25hz",
            Self::Hz12_5 => "lucadellalib/focalcodec_12_5hz",
        }
    }

    /// The `vokra.focalcodec.variant` tag written under
    /// [`KEY_FOCALCODEC_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Hz50 => "50hz",
            Self::Hz25 => "25hz",
            Self::Hz12_5 => "12_5hz",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s `source`
    /// argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Hz50 => {
                "lucadellalib/focalcodec_50hz (FocalCodec 50 Hz single-codebook \
                 audio codec, apache-2.0; base wavlm-large is MIT)"
            }
            Self::Hz25 => {
                "lucadellalib/focalcodec_25hz (FocalCodec 25 Hz single-codebook \
                 audio codec, apache-2.0; base wavlm-large is MIT)"
            }
            Self::Hz12_5 => {
                "lucadellalib/focalcodec_12_5hz (FocalCodec 12.5 Hz single-codebook \
                 audio codec, apache-2.0; base wavlm-large is MIT)"
            }
        }
    }
}

/// Outcome of a FocalCodec conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::funcodec::FuncodecReport`],
/// [`super::wespeaker::WespeakerReport`],
/// [`super::tiger::TigerReport`]) adapted to the file-oriented
/// `convert_focalcodec_file` surface.
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
    /// Which FocalCodec variant was written.
    pub variant: Option<FocalcodecVariant>,
}

/// Converts a `lucadellalib/focalcodec_{50hz,25hz,12_5hz}`
/// safetensors checkpoint at `input` into a Vokra-native GGUF at
/// `output`, tagging the emitted GGUF as the supplied
/// [`FocalcodecVariant`] (mirror of `convert_tiger_file` /
/// `convert_bigvgan_file`).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// safetensors name; the `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_hf), and `vokra.focalcodec.variant` chunks are stamped for
/// the runtime compliance gate (FR-CP-03) and shape-checked config
/// dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"apache-2.0"`, `Permissive`) — every
/// upstream FocalCodec HF release ships apache-2.0.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_focalcodec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: FocalcodecVariant,
) -> Result<FocalcodecReport, ConvertError> {
    // FocalCodec-50Hz is ~569 MB (142M params, F32); the 25Hz and
    // 12.5Hz variants are ~577 MB / ~581 MB (144M / 145M F32,
    // HF-verified 2026-07-31). Still 1 order of magnitude smaller
    // than the streaming-mandated Moshi 14 GiB tier, so the simple
    // `std::fs::read` posture the sibling non-streaming BF16
    // pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_FOCALCODEC_VARIANT, variant.tag());

    // Default provenance stamp — Permissive apache-2.0 (every upstream
    // FocalCodec model card verified via HF API 2026-07-30 / -31). The
    // optional `license` argument overrides below.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());

    let mut report = FocalcodecReport {
        variant: Some(variant),
        ..FocalcodecReport::default()
    };
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

        let report =
            convert_focalcodec_file(&input_path, &output_path, None, FocalcodecVariant::Hz50)
                .expect("convert_focalcodec_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(FocalcodecVariant::Hz50));

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

        let report =
            convert_focalcodec_file(&input_path, &output_path, None, FocalcodecVariant::Hz50)
                .expect("convert");
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

        convert_focalcodec_file(
            &input_path,
            &output_path,
            Some("mit"),
            FocalcodecVariant::Hz50,
        )
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

    /// The 25Hz variant reuses the same converter body but the name /
    /// variant / upstream stamps differ. Silently sharing stamps would
    /// misroute a downstream loader that dispatches on
    /// `vokra.model.name` — this test guards the variant switch
    /// (`super::tiger::tests::speech_variant_emits_distinct_stamps`
    /// precedent).
    #[test]
    fn hz25_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("codec.encoder.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("hz25-in", &input_bytes);
        let output_path = write_temp("hz25-out", &[]);

        let report =
            convert_focalcodec_file(&input_path, &output_path, None, FocalcodecVariant::Hz25)
                .expect("convert 25Hz variant");
        assert_eq!(report.variant, Some(FocalcodecVariant::Hz25));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("focalcodec-25hz"),
            "Hz25 must emit its own model.name, not fall back to Hz50"
        );
        assert_eq!(
            file.get(KEY_FOCALCODEC_VARIANT).and_then(|v| v.as_str()),
            Some("25hz")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("lucadellalib/focalcodec_25hz")
        );
        // Arch + category are shared with Hz50 (same downstream dispatch).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// The 12.5Hz variant likewise emits its own name / variant /
    /// upstream stamps. The `12_5hz` tag uses an underscore (not a dot)
    /// so it stays a valid identifier in GGUF metadata and downstream
    /// slug tables (matches the upstream HF slug `focalcodec_12_5hz`).
    /// The **repo** slug is `focalcodec-12-5hz` (dashes only — HF repo
    /// naming convention, no dots), so the two spellings are
    /// intentional and pinned by this test.
    #[test]
    fn hz12_5_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [1.5_f32, -2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("codec.decoder.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("hz12_5-in", &input_bytes);
        let output_path = write_temp("hz12_5-out", &[]);

        let report =
            convert_focalcodec_file(&input_path, &output_path, None, FocalcodecVariant::Hz12_5)
                .expect("convert 12.5Hz variant");
        assert_eq!(report.variant, Some(FocalcodecVariant::Hz12_5));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("focalcodec-12-5hz"),
            "Hz12_5 must emit `focalcodec-12-5hz` (dashes only, HF repo slug spelling)"
        );
        assert_eq!(
            file.get(KEY_FOCALCODEC_VARIANT).and_then(|v| v.as_str()),
            Some("12_5hz"),
            "variant tag uses underscore to match upstream HF slug spelling"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("lucadellalib/focalcodec_12_5hz")
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// Every enum variant maps to a distinct `(name, tag, upstream_hf)`
    /// triple — a defensive pin against a copy-paste that would
    /// silently re-use the Hz50 strings for a new variant.
    #[test]
    fn every_variant_has_distinct_stamps() {
        let variants = [
            FocalcodecVariant::Hz50,
            FocalcodecVariant::Hz25,
            FocalcodecVariant::Hz12_5,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let a = variants[i];
                let b = variants[j];
                assert_ne!(a.name(), b.name(), "names must differ ({a:?} vs {b:?})");
                assert_ne!(a.tag(), b.tag(), "tags must differ ({a:?} vs {b:?})");
                assert_ne!(
                    a.upstream_hf(),
                    b.upstream_hf(),
                    "upstream_hf must differ ({a:?} vs {b:?})"
                );
            }
        }
    }
}
