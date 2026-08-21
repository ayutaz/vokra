#![allow(clippy::doc_lazy_continuation)]
//! **Vocos** (`charactr/vocos-mel-24khz`, `charactr/vocos-encodec-24khz`,
//! MIT): pytorch-pickle → safetensors → GGUF conversion (2026-08-01
//! wave).
//!
//! Input: an upstream `charactr/vocos-{mel,encodec}-24khz` release —
//! **the top-downloaded HF vocoder audio release** as of 2026-08-01
//! (2.85M mel-24khz downloads). Vocos is the Fourier-space vocoder
//! (Siuzdak 2023, arXiv:2306.00814) that decodes mel spectrograms or
//! EnCodec latents into 24 kHz PCM through a ConvNeXt 1D backbone +
//! **iSTFT head** — a fundamentally different topology from every
//! HiFi-GAN family sibling (`bigvgan` / `hifigan_vocoder` /
//! `speecht5_hifigan`) which upsample time-domain waveforms through
//! transposed-conv + MRF blocks. Silently sharing an arch tag with
//! the HiFi-GAN family would mis-route runtime dispatch to a
//! wrong-shape forward.
//!
//! Both upstream releases ship a **torch pickle** `pytorch_model.bin`
//! + `config.yaml` only (no `model.safetensors` mirror as of
//! 2026-08-01 — verified via `https://huggingface.co/api/models/
//! charactr/vocos-mel-24khz` + `.../vocos-encodec-24khz`); callers
//! pre-flatten to safetensors offline via
//! `tools/parity/bin_to_safetensors.py`. The dedicated
//! `tools/parity/vocos_prepare_checkpoint.py` wrapper pins each audited
//! upstream revision — the same pattern
//! SpeechT5-HiFi-GAN / DeBERTa v3 large / VoxCPM-0.5B use, and the
//! reason is the same: Vokra's Rust converter is safetensors-only by
//! design so the runtime never grows a pickle parser, keeping the
//! NFR-DS-02 zero-dep posture).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream state-dict name (`feature_extractor.*`, `backbone.*`,
//! `head.*`), plus the `vokra.provenance.*`, `vokra.model.*`, and
//! `vokra.vocos.*` metadata chunks the native Vocos loader reads.
//!
//! # Provenance
//!
//! - **HF paths** (two variants share this single converter):
//!   - `charactr/vocos-mel-24khz`     (mel-24khz, 2.85M downloads,
//!     `MelSpectrogramFeatures` frontend, 100 mel bands @ 24 kHz,
//!     ~52 MB).
//!   - `charactr/vocos-encodec-24khz` (encodec-24khz, ~161 MB,
//!     `EncodecFeatures` frontend, 128-d EnCodec latents @ 75 Hz).
//! - **License (SPDX)**: `mit` for both variants — verified 2026-08-01
//!   via HF cardData API `license: mit` on both repos (CLAUDE.md
//!   「ハルシネーション厳禁」). Redistribution + commercial use OK,
//!   no attribution obligation.
//! - **Category**: `vocoder` — spectrogram / latent → PCM waveform
//!   generator. Same category tag as sibling `bigvgan` /
//!   `hifigan_vocoder` / `speecht5_hifigan` (all `vocoder`), but
//!   distinct **arch** tag (`vocos` vs `bigvgan` / `hifigan_vocoder`
//!   / `speecht5_hifigan`).
//!
//! # Distinct arch from HiFi-GAN family
//!
//! `ARCH = "vocos"` is intentionally distinct from every HiFi-GAN
//! family sibling — even though the category tag is shared. Vocos
//! does NOT upsample a time-domain waveform through transposed-conv
//! + MRF (leaky_relu / snake activation stacks). Instead:
//!
//! - The feature extractor emits mel or EnCodec latents.
//! - A **ConvNeXt 1D backbone** (Vocos paper §3.2) processes the
//!   spectral representation entirely in a shift-invariant feature
//!   space.
//! - The **iSTFTHead** projects to `(magnitude, phase)` (or
//!   symmetric complex STFT coefficients), reconstructs the complex
//!   STFT, and inverts to PCM via a single inverse STFT — one
//!   time-domain upsample step, no MRF.
//!
//! Runtime dispatch checks `vokra.model.arch` before picking the
//! forward implementation; sharing tags across topologies would
//! mis-route to a wrong-shape kernel.
//!
//! # Variant identity
//!
//! Both variants share the eight-block ConvNeXt + iSTFT family, but their
//! widths, Fourier axes, padding, and normalization differ. Mel uses plain
//! LayerNorm; Encodec uses four-row bandwidth-conditioned AdaLayerNorm. The
//! [`VocosVariant`]
//! discriminator tags the emitted GGUF under `vokra.vocos.variant`
//! so the runtime can pick the correct feature-extractor bind
//! path; every hparam is a shape-derived value read at
//! `vokra-models` bind time (FR-EX-08 authoritative gate) — this
//! converter is the byte-parallel pass-through side of the wave.
//!
//! # BF16 pass-through (mirror of speecht5_hifigan / bigvgan / focalcodec)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm
//! — no convert-time widening. BF16 stays GGUF type 30
//! (`GgmlType::BF16`); the runtime widens BF16 → f32 losslessly at
//! load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`VocosReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot
//! slip in undetected. Upstream `config.yaml` pins F32 so BF16 is
//! not expected on the primary release, but the counter is kept for
//! the SKU-parity contract and for third-party BF16 re-quantizations.
//!
//! # Vocos quantization warning (CLAUDE.md 設計判断 §Vocos)
//!
//! CLAUDE.md pins Vocos as **INT8-fragile**: 「Vocos は量子化耐性弱
//! (INT8 崩壊) → fp16 必須」. The converter therefore never emits
//! INT8 (the K-quant path is Whisper-only per
//! `main.rs --quantize` guard); BF16 is expected to be safe (BF16
//! loss is mantissa-only, not activation-crushing INT8 saturation),
//! but no parity data yet — an owner-side follow-up when the
//! runtime binder lands.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict names verbatim**
//! (the sibling BF16 pass-through contract — CSM / Kokoro /
//! CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice / VoxCPM /
//! WeSpeaker / ECAPA-TDNN / hifigan_vocoder / speecht5_hifigan /
//! bigvgan / focalcodec). Real-weight parity against `vocos==0.1.0` is gated
//! in `crates/vokra-models/tests/parity_vocos_real.rs`; both official
//! variants pass the fixed `1e-5` bound.
//!
//! # No ONNX (permanent)
//!
//! Charactr AI ships PyTorch pickle checkpoints; this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in `crates/vokra-models/src/vocos/` and `vokra-ops::vocos`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Runtime handshake
//!
//! `Vocos::from_gguf` walks the complete 83-tensor Mel or 82-tensor Encodec
//! manifest and rejects missing, extra, renamed, or wrong-shaped tensors
//! before constructing the native forward.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Vocos GGUFs. Shared across every
/// [`VocosVariant`] — the releases share a ConvNeXt/iSTFT family but retain
/// distinct widths, normalization, padding, and Fourier axes.
///
/// Intentionally distinct from every HiFi-GAN family sibling
/// (`bigvgan`, `hifigan_vocoder`, `speecht5_hifigan`) — Vocos is a
/// Fourier-space vocoder, not a time-domain upsample+MRF vocoder.
pub const ARCH: &str = "vocos";

/// `vokra.model.category` value written for every Vocos GGUF. Same
/// category tag as sibling `bigvgan` / `hifigan_vocoder` /
/// `speecht5_hifigan` (all `vocoder`), used by the model-card
/// generator classifier.
pub const CATEGORY: &str = "vocoder";

/// Default upstream weight licence (SPDX). Both `charactr/vocos-
/// mel-24khz` and `charactr/vocos-encodec-24khz` ship MIT end-to-end
/// (Charactr AI code + trained weights); verified 2026-08-01 via HF
/// cardData API.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.vocos.variant`: `"mel_24khz"` / `"encodec_24khz"`.
/// Consumers pick a specific frontend feature extractor without
/// parsing free-text `vokra.model.name` (mirrors
/// `vokra.bigvgan.variant` + `vokra.focalcodec.variant`).
pub const KEY_VOCOS_VARIANT: &str = "vokra.vocos.variant";

/// Which Vocos release the caller is converting. Selects the model
/// name / upstream HF slug / variant tag written into the GGUF.
///
/// Both variants share [`ARCH`] `vocos`; the required variant tag selects the
/// exact manifest and numerical contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocosVariant {
    /// `charactr/vocos-mel-24khz`: `MelSpectrogramFeatures` frontend
    /// (100 mel bands @ 24 kHz sampling). Canonical / default —
    /// 2.85M downloads (HF audio-vocoder category top as of
    /// 2026-08-01).
    /// `vokra.vocos.variant = "mel_24khz"`.
    Mel24khz,
    /// `charactr/vocos-encodec-24khz`: `EncodecFeatures` frontend
    /// (128-d EnCodec RVQ latents @ 75 Hz → 24 kHz PCM).
    /// `vokra.vocos.variant = "encodec_24khz"`.
    Encodec24khz,
}

impl VocosVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mel24khz => "vocos-mel-24khz",
            Self::Encodec24khz => "vocos-encodec-24khz",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// release — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Mel24khz => "charactr/vocos-mel-24khz",
            Self::Encodec24khz => "charactr/vocos-encodec-24khz",
        }
    }

    /// The `vokra.vocos.variant` tag written under
    /// [`KEY_VOCOS_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Mel24khz => "mel_24khz",
            Self::Encodec24khz => "encodec_24khz",
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp (`stamp_provenance`'s `source`
    /// argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Mel24khz => {
                "charactr/vocos-mel-24khz (Vocos Fourier-space vocoder, \
                 MelSpectrogramFeatures 100-band @ 24 kHz, MIT)"
            }
            Self::Encodec24khz => {
                "charactr/vocos-encodec-24khz (Vocos Fourier-space vocoder, \
                 EncodecFeatures 128-d latents @ 24 kHz, MIT)"
            }
        }
    }
}

/// Outcome of a Vocos conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::speecht5_hifigan::Speecht5HifiganReport`,
/// `super::bigvgan::BigVGanReport`,
/// `super::focalcodec::FocalcodecReport`) adapted to the
/// file-oriented `convert_vocos_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VocosReport {
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
    /// silent widen / downcast cannot slip in undetected without
    /// this counter also drifting.
    pub bf16_passthrough: usize,
    /// Which Vocos variant was written.
    pub variant: Option<VocosVariant>,
}

/// Converts a `charactr/vocos-{mel,encodec}-24khz` safetensors
/// checkpoint at `input` into a Vokra-native GGUF at `output`,
/// tagging the emitted GGUF as the supplied [`VocosVariant`] (mirror
/// of `convert_focalcodec_file` / `convert_bigvgan_file`).
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` (arch / name / category),
/// `vokra.provenance.*` (weight_license / license / model_id /
/// source / upstream_hf), and `vokra.vocos.variant` chunks are
/// stamped for the runtime compliance gate (FR-CP-03) and
/// shape-checked config dispatch.
///
/// `license` optionally overrides the stamped weight license (raw
/// SPDX string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"mit"`, `Permissive`) — every upstream
/// Vocos HF release ships MIT (verified 2026-08-01 via HF API
/// cardData; the upstream `github.com/charactr-platform/vocos`
/// LICENSE is also standard MIT).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input.
pub fn convert_vocos_file(
    input: &Path,
    output: &Path,
    variant: VocosVariant,
    license: Option<&str>,
) -> Result<VocosReport, ConvertError> {
    // Vocos mel-24khz is ~52 MB and encodec-24khz is ~161 MB — both 2
    // orders of magnitude smaller than the streaming-mandated Moshi
    // 14 GiB tier, so the simple `std::fs::read` posture the sibling
    // non-streaming BF16 pass-through converters use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_VOCOS_VARIANT, variant.tag());

    // Default provenance stamp — Permissive MIT end-to-end (every
    // upstream Vocos model card verified via HF API 2026-08-01). The
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

    let mut report = VocosReport {
        variant: Some(variant),
        ..VocosReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted
    // ADR (mirror of wespeaker / ecapa_tdnn / hifigan_vocoder /
    // speecht5_hifigan / bigvgan / focalcodec); the runtime widens
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

    /// Builds an F32 tensor safetensors buffer — matches upstream Vocos
    /// dtype (F32 pinned by `config.yaml`, HF API verified 2026-08-01).
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
            "vokra-vocos-{kind}-{}-{}.bin",
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
    fn f32_tensor_passes_through_and_stamps_land_mel24khz() {
        // Upstream Vocos ships F32 (`config.yaml` `torch_dtype: float32`,
        // HF API verified 2026-08-01) — this test pins the primary
        // code path.
        let f32_vals: [f32; 6] = [0.5, -0.25, 1.5, -3.0, 42.0, 0.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Mirror a realistic upstream tensor name from Vocos's
        // `Vocos.state_dict()` walk — `backbone.norm.weight` is the
        // initial LayerNorm of the ConvNeXt 1D stack.
        let input_bytes = safetensors_one_f32("backbone.norm.weight", &[2, 3], &f32_bytes);
        let input_path = write_temp("mel24-in", &input_bytes);
        let output_path = write_temp("mel24-out", &[]);

        let report = convert_vocos_file(&input_path, &output_path, VocosVariant::Mel24khz, None)
            .expect("convert_vocos_file must accept F32 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32 does not increment BF16 counter"
        );
        assert_eq!(report.variant, Some(VocosVariant::Mel24khz));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("backbone.norm.weight")
            .expect("F32 tensor present in output");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), f32_bytes.as_slice());

        // Provenance / category / arch / variant chunks landed.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("vocos-mel-24khz")
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
            Some("charactr/vocos-mel-24khz")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `vocoder`"
        );
        assert_eq!(
            file.get(KEY_VOCOS_VARIANT).and_then(|v| v.as_str()),
            Some("mel_24khz")
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Defensive test — future BF16-quantized derivatives should
        // ride the same arm as the sibling BF16-pass-through
        // converters (speecht5_hifigan / bigvgan / focalcodec).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        // Mirror a realistic Vocos state_dict name — `head.out.weight`
        // is the final projection of the iSTFT head that emits the
        // 2×N_FFT packed magnitude+phase STFT coefficients.
        let input_bytes = safetensors_one_bf16("head.out.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_vocos_file(&input_path, &output_path, VocosVariant::Mel24khz, None)
            .expect("convert");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("head.out.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    /// The Encodec24khz variant reuses the same converter body but
    /// the name / variant / upstream stamps differ. Silently sharing
    /// stamps would misroute a downstream loader that dispatches on
    /// `vokra.model.name` — this test guards the variant switch
    /// (`super::focalcodec::tests::hz25_variant_emits_distinct_stamps`
    /// precedent).
    #[test]
    fn encodec24khz_variant_emits_distinct_stamps() {
        let f32_bytes: Vec<u8> = [7.0_f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // `feature_extractor.encodec.quantizer.vq.layers.0.codebook.embed`
        // is a realistic EncodecFeatures state-dict path.
        let input_bytes = safetensors_one_f32(
            "feature_extractor.encodec.quantizer.vq.layers.0.codebook.embed",
            &[1, 2],
            &f32_bytes,
        );
        let input_path = write_temp("enc24-in", &input_bytes);
        let output_path = write_temp("enc24-out", &[]);

        let report =
            convert_vocos_file(&input_path, &output_path, VocosVariant::Encodec24khz, None)
                .expect("convert Encodec24khz variant");
        assert_eq!(report.variant, Some(VocosVariant::Encodec24khz));

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("vocos-encodec-24khz"),
            "Encodec24khz must emit its own model.name, not fall back to Mel24khz"
        );
        assert_eq!(
            file.get(KEY_VOCOS_VARIANT).and_then(|v| v.as_str()),
            Some("encodec_24khz")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("charactr/vocos-encodec-24khz")
        );
        // Arch + category are shared with Mel24khz (same downstream
        // dispatch).
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

    #[test]
    fn license_override_flows_through() {
        // A user who trained on a permissive corpus (e.g. Apache-2.0)
        // supplies a different SPDX id at conversion time — the
        // override must land on KEY_PROVENANCE_LICENSE +
        // KEY_PROVENANCE_WEIGHT_LICENSE and the LicenseClass must be
        // re-derived by from_license_str.
        let f32_bytes: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_f32("backbone.pos_embed", &[1, 2], &f32_bytes);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        convert_vocos_file(
            &input_path,
            &output_path,
            VocosVariant::Mel24khz,
            Some("apache-2.0"),
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

    /// Every enum variant maps to a distinct
    /// `(name, tag, upstream_hf)` triple — a defensive pin against a
    /// copy-paste that would silently re-use the Mel24khz strings for
    /// a new variant.
    #[test]
    fn every_variant_has_distinct_stamps() {
        let variants = [VocosVariant::Mel24khz, VocosVariant::Encodec24khz];
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
