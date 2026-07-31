#![allow(clippy::doc_lazy_continuation)]
//! **SNAC** (`hubertsiuzdak/snac_{24khz,44khz}`, MIT) — Multi-Scale
//! Neural Audio Codec (Siuzdak et al. 2024, arXiv:2410.14411):
//! safetensors → GGUF conversion (2026-08-01 Wave 3).
//!
//! Input: an upstream `hubertsiuzdak/snac_24khz` /
//! `hubertsiuzdak/snac_44khz` release — SNAC produces 3–4 hierarchical
//! RVQ levels at different frame rates (24 kHz variant emits ~12/23/47
//! Hz tokens across 3 codebooks; 44.1 kHz emits a 4-level stack).
//! Unlike sibling codecs that ship `model.safetensors` directly
//! (FocalCodec, WavTokenizer, neucodec), SNAC ships **only**
//! `pytorch_model.bin` + `config.json` for both variants (verified
//! 2026-08-01 via `https://huggingface.co/api/models/hubertsiuzdak/snac_24khz`
//! and sibling — siblings for each = `[".gitattributes", "README.md",
//! "config.json", "pytorch_model.bin"]`, no `model.safetensors` mirror).
//! Callers pre-flatten the pickle to safetensors offline via
//! `tools/parity/bin_to_safetensors.py --hf-repo hubertsiuzdak/snac_{24khz,44khz}`
//! (the same shared bridge tool `speecht5_hifigan` / DeBERTa v3 large /
//! VoxCPM-0.5B / Fun-CosyVoice3 use, and the reason is identical:
//! Vokra's Rust converter is safetensors-only by design so the runtime
//! never grows a pickle parser, keeping the NFR-DS-02 zero-dep posture).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream `snac.SNAC` state-dict name, plus the `vokra.provenance.*`
//! / `vokra.model.*` / `vokra.snac.variant` metadata chunks a future
//! native SNAC loader will read.
//!
//! # Provenance
//!
//! - **HF paths** (two variants share this single converter):
//!   - `hubertsiuzdak/snac_24khz` (Hz24, canonical / default, ~76 MB
//!     `pytorch_model.bin`, ~452k monthly downloads — primary
//!     consumer = Orpheus-TTS + MOSS voice family + CSM-1B-adjacent
//!     TTS stacks).
//!   - `hubertsiuzdak/snac_44khz` (Hz44, music-quality, ~208 MB
//!     `pytorch_model.bin`, ~1.3k monthly downloads).
//! - **License (SPDX)**: `mit` for both variants — verified
//!   2026-08-01 via HF cardData API `license: mit` for
//!   `hubertsiuzdak/snac_24khz` and `hubertsiuzdak/snac_44khz`,
//!   CC-verified (CLAUDE.md 「ハルシネーション厳禁」). Upstream
//!   `github.com/hubertsiuzdak/snac` LICENSE = MIT.
//! - **Category**: `codec` — audio codec (waveform → hierarchical
//!   RVQ tokens → waveform). Distinct from vocoder (mel → waveform
//!   only).
//! - **Variant tag** (`vokra.snac.variant`): `"24khz"` /
//!   `"44khz"` so a consumer that needs to pick a specific frame
//!   rate + RVQ depth can inspect this without parsing free-text
//!   `vokra.model.name` (mirrors `vokra.focalcodec.variant` +
//!   `vokra.bigvgan.variant`).
//!
//! # SNAC vs sibling codecs
//!
//! Distinct arch tag from every sibling codec (Mimi / DAC /
//! WavTokenizer / neucodec / step_audio2_mini / X-Codec 2 /
//! FunCodec / FocalCodec / SpeechTokenizer / bicodec /
//! XyTokenizer): SNAC is a **multi-scale RVQ** with hierarchical
//! codebooks at different frame rates (`vq_strides` config axis)
//! + noise-conditioned residual (`noise=true`) + optional
//! sliding-window local attention (`attn_window_size` on the 44 kHz
//! variant only), distinct from every flat-RVQ / FSQ /
//! SoundStream / focal-modulation sibling. Silently sharing an
//! arch tag would misroute the runtime dispatch.
//!
//! # BF16 pass-through (mirror of focalcodec / bigvgan / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through
//! arm — no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). Both
//! upstream SNAC releases are torch-native F32 pickles (config
//! carries no dtype override, and upstream
//! `snac.SNAC.from_pretrained` builds a default-dtype model), so
//! the BF16 arm is defensive today; the counter stays for future
//! BF16-quantized derivative releases.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream `snac.SNAC` state-dict
//! names verbatim** (the FunCodec / X-Codec 2 / WeSpeaker /
//! FocalCodec contract). Real-weight parity vs the upstream Python
//! reference is deferred to owner (`docs/license-audit.md` §3.1
//! sign-off queue).
//!
//! # No ONNX (permanent)
//!
//! SNAC ships PyTorch pickle checkpoints; this converter **never**
//! touches ONNX (FR-LD-05); the pipeline is re-implemented natively
//! in `crates/vokra-models/src/snac/` when the codec lands
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for SNAC GGUFs. Shared across both
/// [`SnacVariant`] entries — upstream's
/// `snac.SNAC.from_pretrained()` factory routes both `snac_24khz`
/// and `snac_44khz` to the same class; the topology (encoder /
/// decoder body, hierarchical RVQ head, optional noise / local
/// attention) is structurally identical, only the per-variant
/// config axes (`encoder_dim`, `decoder_dim`, `vq_strides` depth,
/// `attn_window_size`) differ. Splitting arch would misroute a
/// downstream loader that dispatches on `vokra.model.arch`.
///
/// Intentionally distinct from every sibling codec (`mimi`, `dac`,
/// `wavtokenizer`, `neucodec`, `funcodec`, `xcodec2`,
/// `speechtokenizer`, `bicodec`, `xy_tokenizer`, `focalcodec`,
/// `step_audio2_mini`) because SNAC is a multi-scale RVQ family
/// member, not a flat-RVQ / FSQ / SoundStream / focal-modulation
/// codec.
pub const ARCH: &str = "snac";

/// `vokra.model.name` value written for the canonical
/// `hubertsiuzdak/snac_24khz` GGUF (backward-compat alias — new
/// callers should use [`SnacVariant::name`]).
#[allow(dead_code)]
pub const NAME: &str = "snac-24khz";

/// `vokra.model.category` value written for every SNAC GGUF.
pub const CATEGORY: &str = "codec";

/// `vokra.provenance.upstream_hf` value for the canonical `24khz`
/// variant (backward-compat alias — new callers should use
/// [`SnacVariant::upstream_hf`]).
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "hubertsiuzdak/snac_24khz";

/// Default upstream weight licence (SPDX). Verified 2026-08-01 via
/// HF API cardData `license: mit` for both `hubertsiuzdak/snac_24khz`
/// and `hubertsiuzdak/snac_44khz`; upstream
/// `github.com/hubertsiuzdak/snac` LICENSE = MIT.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication
// rule the sibling BF16 pass-through converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.snac.variant`: `"24khz"` / `"44khz"`. Consumers pick a
/// specific frame-rate + RVQ-depth head without parsing free-text
/// `vokra.model.name` (mirrors [`super::focalcodec`] +
/// [`super::bigvgan`] discriminators).
pub const KEY_SNAC_VARIANT: &str = "vokra.snac.variant";

/// Which SNAC release the caller is converting. Selects the model
/// name / upstream HF slug / variant tag written into the GGUF.
///
/// Both variants share [`ARCH`] `snac` (see the [`ARCH`] docstring
/// for why: upstream `snac.SNAC.from_pretrained()` routes both to
/// the same class; the topology is structurally identical, only
/// per-variant config axes differ).
///
/// # Per-variant config axes
///
/// Primary source: HF `config.json` for each release, fetched
/// 2026-08-01.
///
/// | axis | Hz24 | Hz44 |
/// |---|---|---|
/// | `sampling_rate` | 24 000 | 44 100 |
/// | `encoder_dim` | 48 | 64 |
/// | `encoder_rates` | `[2, 4, 8, 8]` (512x downsample) | `[2, 3, 8, 8]` (384x downsample) |
/// | `decoder_dim` | 1024 | 1536 |
/// | `decoder_rates` | `[8, 8, 4, 2]` | `[8, 8, 3, 2]` |
/// | `codebook_size` | 4096 | 4096 |
/// | `codebook_dim` | 8 | 8 |
/// | `vq_strides` | `[4, 2, 1]` (3 RVQ levels) | `[8, 4, 2, 1]` (4 RVQ levels) |
/// | `attn_window_size` | `null` (no attention) | `32` (local attention) |
/// | `noise` | `true` | `true` |
/// | `depthwise` | `true` | `true` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnacVariant {
    /// `hubertsiuzdak/snac_24khz`: 24 kHz sample rate, 3
    /// hierarchical RVQ levels @ ~12/23/47 Hz, no attention
    /// (canonical / higher-download release, primary consumer =
    /// Orpheus-TTS + MOSS voice + CSM-1B-adjacent TTS stacks).
    /// `vokra.snac.variant = "24khz"`.
    Hz24,
    /// `hubertsiuzdak/snac_44khz`: 44.1 kHz sample rate, 4
    /// hierarchical RVQ levels, `attn_window_size=32` for local
    /// attention (music-quality variant, lower download volume).
    /// `vokra.snac.variant = "44khz"`.
    Hz44,
}

impl SnacVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hz24 => "snac-24khz",
            Self::Hz44 => "snac-44khz",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for
    /// this release — the primary redistribution source the
    /// model-card generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Hz24 => "hubertsiuzdak/snac_24khz",
            Self::Hz44 => "hubertsiuzdak/snac_44khz",
        }
    }

    /// The `vokra.snac.variant` tag written under
    /// [`KEY_SNAC_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Hz24 => "24khz",
            Self::Hz44 => "44khz",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s
    /// `source` argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Hz24 => {
                "hubertsiuzdak/snac_24khz (SNAC Multi-Scale Neural Audio Codec, \
                 24 kHz, 3 RVQ levels @ ~12/23/47 Hz, mit)"
            }
            Self::Hz44 => {
                "hubertsiuzdak/snac_44khz (SNAC Multi-Scale Neural Audio Codec, \
                 44.1 kHz music-quality, 4 RVQ levels + 32-frame local attention, mit)"
            }
        }
    }
}

/// Outcome of a SNAC conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// ([`super::focalcodec::FocalcodecReport`],
/// [`super::bigvgan::BigVGanReport`],
/// [`super::wespeaker::WespeakerReport`]) adapted to the
/// variant-taking `convert_snac_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SnacReport {
    /// Total tensors surfaced by the safetensors reader (before
    /// any dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the
    /// safetensors reader accepts only F32 / F16 / BF16 at parse
    /// time, so a non-zero here would signal a reader change
    /// upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset
    /// of [`Self::written`]). Additive observability counter — a
    /// latent silent widen / downcast cannot slip in undetected
    /// without this counter also drifting.
    pub bf16_passthrough: usize,
    /// Which SNAC variant was written.
    pub variant: Option<SnacVariant>,
}

/// Converts a `hubertsiuzdak/snac_{24khz,44khz}` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`,
/// tagging the emitted GGUF as the supplied [`SnacVariant`]
/// (mirror of `convert_focalcodec_file` / `convert_bigvgan_file`).
///
/// **Prerequisite**: SNAC ships pytorch_model.bin only (verified
/// 2026-08-01 via HF API for both variants). Callers pre-flatten
/// to safetensors via `tools/parity/bin_to_safetensors.py
/// --hf-repo hubertsiuzdak/snac_{24khz,44khz}` (the shared bridge
/// tool `speecht5_hifigan` uses) before invoking this converter —
/// no pickle parser enters the Vokra runtime (NFR-DS-02 /
/// FR-LD-05).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// `snac.SNAC` state-dict name; the `vokra.model.*` (arch / name /
/// category), `vokra.provenance.*` (weight_license / license /
/// model_id / source / upstream_hf), and `vokra.snac.variant`
/// chunks are stamped for the runtime compliance gate (FR-CP-03)
/// and shape-checked config dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// [`DEFAULT_LICENSE_SPDX`] (`"mit"`, `Permissive`) — both
/// upstream SNAC HF releases ship MIT.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
pub fn convert_snac_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: SnacVariant,
) -> Result<SnacReport, ConvertError> {
    // SNAC-24kHz is ~76 MB per upstream pytorch_model.bin; SNAC-44kHz
    // is ~208 MB (both verified 2026-08-01 via HF file listing).
    // Both are 2–3 orders of magnitude smaller than the streaming-
    // mandated Moshi 14 GiB tier, so the simple `std::fs::read`
    // posture the sibling non-streaming BF16 pass-through converters
    // use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_SNAC_VARIANT, variant.tag());

    // Default provenance stamp — Permissive MIT (every upstream
    // SNAC model card verified via HF API 2026-08-01). The optional
    // `license` argument overrides below.
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

    let mut report = SnacReport {
        variant: Some(variant),
        ..SnacReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR; the runtime widens BF16 → f32 exactly at load via the
    // single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    // decode_bf16`.
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

    /// Builds an F32 tensor safetensors buffer — matches upstream
    /// SNAC dtype (torch-native F32 pickles for both variants).
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
            "vokra-snac-{kind}-{}-{}.bin",
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
        // Upstream SNAC is F32 (both variants torch-native pickle) —
        // this test pins the primary code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from SNAC's encoder
        // body (upstream `snac.snac.Encoder.block.*` module tree).
        let input_bytes =
            safetensors_one_f32("encoder.block.0.block.0.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("f32-in", &input_bytes);
        let output_path = write_temp("f32-out", &[]);

        let report = convert_snac_file(&input_path, &output_path, None, SnacVariant::Hz24)
            .expect("convert_snac_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(SnacVariant::Hz24));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.block.0.block.0.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category / variant chunks landed.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
            "Hz24 must emit the canonical `snac-24khz` model name"
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
            Some(UPSTREAM_HF),
            "Hz24 must emit the canonical `hubertsiuzdak/snac_24khz` upstream slug"
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `codec`",
        );
        assert_eq!(
            file.get(KEY_SNAC_VARIANT).and_then(|v| v.as_str()),
            Some("24khz"),
            "vokra.snac.variant must be `24khz` for Hz24",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (focalcodec / wespeaker / ecapa_tdnn). Non-zero
        // BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a realistic upstream tensor name from SNAC's RVQ
        // head (upstream
        // `snac.vq.ResidualVectorQuantize.quantizers.*.codebook.weight`).
        let input_bytes =
            safetensors_one_bf16("quantizer.quantizers.0.codebook.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report =
            convert_snac_file(&input_path, &output_path, None, SnacVariant::Hz24).expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("quantizer.quantizers.0.codebook.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_flows_through() {
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes =
            safetensors_one_f32("encoder.block.0.block.0.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_snac_file(
            &input_path,
            &output_path,
            Some("apache-2.0"),
            SnacVariant::Hz24,
        )
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

    /// The Hz44 variant reuses the same converter body but the
    /// name / variant / upstream stamps differ. Silently sharing
    /// stamps would misroute a downstream loader that dispatches
    /// on `vokra.model.name` or the variant discriminator — this
    /// test guards the variant switch (the
    /// `super::focalcodec::tests::hz25_variant_emits_distinct_stamps`
    /// precedent).
    #[test]
    fn hz44_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("decoder.model.0.weight", &[1, 2], &f32_bytes);
        let input_path = write_temp("hz44-in", &input_bytes);
        let output_path = write_temp("hz44-out", &[]);

        let report = convert_snac_file(&input_path, &output_path, None, SnacVariant::Hz44)
            .expect("convert 44kHz variant");
        assert_eq!(report.variant, Some(SnacVariant::Hz44));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("snac-44khz"),
            "Hz44 must emit its own model.name, not fall back to Hz24"
        );
        assert_eq!(
            file.get(KEY_SNAC_VARIANT).and_then(|v| v.as_str()),
            Some("44khz")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("hubertsiuzdak/snac_44khz")
        );
        // Arch + category are shared with Hz24 (same downstream
        // dispatch — both variants route to the same `snac.SNAC`
        // class upstream).
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

    /// Every enum variant maps to a distinct `(name, tag,
    /// upstream_hf, source_description)` tuple — a defensive pin
    /// against a copy-paste that would silently re-use the Hz24
    /// strings for a new variant (matches the focalcodec
    /// `every_variant_has_distinct_stamps` precedent).
    #[test]
    fn every_variant_has_distinct_stamps() {
        let variants = [SnacVariant::Hz24, SnacVariant::Hz44];
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
                assert_ne!(
                    a.source_description(),
                    b.source_description(),
                    "source_description must differ ({a:?} vs {b:?})"
                );
            }
        }
    }
}
