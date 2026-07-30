//! **MeloTTS** (`myshell-ai/MeloTTS-{English,Chinese,Korean}`, MIT):
//! safetensors → GGUF conversion (implementer C wave, 2026-07-30).
//!
//! Input: an upstream `myshell-ai/MeloTTS-<lang>` release — the upstream
//! ships `checkpoint.pth` (torch pickle); callers must offline-flatten to
//! safetensors first (the same pattern as CSM / DAC / DFN3 —
//! `crates/vokra-convert/src/models/{csm,dac,denoise}.rs`). Output: a
//! GGUF carrying every float tensor plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native MeloTTS loader will
//! read.
//!
//! # Architecture (primary source, 2026-07-30 CC fetch)
//!
//! MeloTTS is a **VITS2-family multilingual TTS** with a modified
//! duration predictor (`use_duration_discriminator=true` +
//! `use_noise_scaled_mas=true` +
//! `n_layers_trans_flow=3`). Every hparam below is transcribed verbatim
//! from `huggingface.co/myshell-ai/MeloTTS-{English,Chinese,Korean}/raw/
//! main/config.json` (fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
//!
//! Shared axes (all three language releases):
//! - `data.sampling_rate = 44100`
//! - `data.filter_length = 2048` (n_fft)
//! - `data.hop_length = 512`
//! - `data.n_speakers = 256`
//! - `model.inter_channels = 192`
//! - `model.hidden_channels = 192`
//! - `model.filter_channels = 768`
//! - `model.n_heads = 2`
//! - `model.n_layers = 6` (posterior encoder)
//! - `model.n_layers_trans_flow = 3` (Transformer flow layers)
//! - `model.gin_channels = 256` (global-conditioning speaker vector dim)
//! - `model.upsample_rates = [8, 8, 2, 2, 2]` → HiFi-GAN generator
//!   (product 512 = hop_length)
//! - `model.upsample_initial_channel = 512`
//! - `model.resblock = "1"` (HiFi-GAN ResBlock1)
//!
//! Language-specific axes (compile-time [`MeloVariant`] constants):
//! - **English**: `n_symbols = 178`, `num_tones = ?` (English config has
//!   no explicit `num_tones` — the CJK-tonal head is absent), `spk2id =
//!   {EN-US:0, EN-BR:1, EN_INDIA:2, EN-AU:3, EN-Default:4}` (5 speakers)
//! - **Chinese**: `n_symbols = 112`, `num_tones = 11`, `spk2id = {ZH:1}`
//!   (1 speaker)
//! - **Korean**: `n_symbols = 219`, `num_tones = 16`, `num_languages = 10`,
//!   `spk2id = {KR:0}` (1 speaker)
//!
//! The variant tag rides the GGUF at `vokra.melotts.variant` (one of
//! `"english"` / `"chinese"` / `"korean"`) so a runtime dispatcher can
//! resolve the tone-embedding and speaker table without inspecting the
//! tensor shapes. Every hparam listed above rides
//! `vokra.melotts.*` verbatim so a future `MeloTtsWeights::from_gguf`
//! reader can walk them without re-parsing the upstream `config.json`.
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `wespeaker` / `neucodec`)
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! upstream safetensors names. BF16 stays GGUF type 30
//! (`GgmlType::BF16`) — no convert-time widening; runtime widens BF16 →
//! f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec contract). Real-weight binding is a follow-up
//! wave gated on the upstream tensor-name manifest fetch; this converter
//! passes every F32 / F16 / BF16 tensor through unchanged so a future
//! `MeloTtsWeights::from_gguf` can walk the same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream MeloTTS Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off) — this
//! converter provides the byte-parallel GGUF surface only.
//!
//! # No ONNX (permanent)
//!
//! MeloTTS is distributed as a torch pickle (`checkpoint.pth`) + a
//! Python pipeline. Callers offline-flatten to safetensors first (mirror
//! of the CSM / DAC / DFN3 prepare-script pattern); this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/melotts/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for MeloTTS GGUFs. Distinct from `piper-plus`
/// (MB-iSTFT-VITS2) and `vits-ja` (plain VITS + HiFi-GAN) — silently
/// sharing an arch tag would misroute the runtime dispatch.
pub(crate) const ARCH: &str = "melotts";

/// Model category tag — `tts`, distinguishing MeloTTS from codec /
/// speaker / emotion models the deep-not-wide catalog also carries.
pub(crate) const CATEGORY: &str = "tts";

/// The MeloTTS release variants. Each pins the language-specific config
/// axes (`n_symbols` / `num_tones` / `num_languages` / `n_speakers` /
/// canonical HF release name) as compile-time constants so no invented
/// placeholder can slip into the GGUF metadata.
///
/// The shared VITS2 backbone axes (`hidden_channels=192` /
/// `filter_channels=768` / `n_heads=2` / `n_layers=6` /
/// `n_layers_trans_flow=3` / `gin_channels=256` /
/// `upsample_rates=[8,8,2,2,2]` / `upsample_initial_channel=512`) are
/// identical across the three releases and live at module scope
/// ([`SAMPLE_RATE`] / [`HIDDEN_CHANNELS`] / …).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeloVariant {
    /// `myshell-ai/MeloTTS-English` (MIT). 178 symbols, 5 speakers.
    /// English config has no `num_tones` (the CJK tone-embedding head is
    /// absent for English) — the metadata key is written as `0` to
    /// distinguish "the head is not part of the model" from an
    /// "unstated" positive value.
    English,
    /// `myshell-ai/MeloTTS-Chinese` (MIT). 112 symbols, 11 tones,
    /// 1 speaker (`ZH`).
    Chinese,
    /// `myshell-ai/MeloTTS-Korean` (MIT). 219 symbols, 16 tones,
    /// 10 languages, 1 speaker (`KR`).
    Korean,
}

impl MeloVariant {
    /// The canonical HF release slug for this variant (== the string
    /// stamped under `vokra.model.name`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::English => "melotts-english",
            Self::Chinese => "melotts-chinese",
            Self::Korean => "melotts-korean",
        }
    }

    /// The `vokra.melotts.variant` short tag written on the GGUF (one of
    /// `"english"` / `"chinese"` / `"korean"`).
    pub const fn variant_tag(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Chinese => "chinese",
            Self::Korean => "korean",
        }
    }

    /// Upstream HuggingFace repo slug — recorded verbatim under
    /// `vokra.provenance.upstream_hf`.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::English => "myshell-ai/MeloTTS-English",
            Self::Chinese => "myshell-ai/MeloTTS-Chinese",
            Self::Korean => "myshell-ai/MeloTTS-Korean",
        }
    }

    /// `data.n_symbols` — number of text symbols (transcribed from each
    /// `config.json`'s `symbols` array length).
    pub const fn n_symbols(self) -> u32 {
        match self {
            Self::English => 178,
            Self::Chinese => 112,
            Self::Korean => 219,
        }
    }

    /// `num_tones` — CJK tone-embedding vocab. English has no tone head
    /// (returns 0 = "the head is not part of the model", distinct from
    /// an unstated positive value).
    pub const fn num_tones(self) -> u32 {
        match self {
            Self::English => 0,
            Self::Chinese => 11,
            Self::Korean => 16,
        }
    }

    /// `num_languages` — multilingual embedding vocab (Korean release
    /// carries a 10-language cross-lingual head; English + Chinese
    /// releases are monolingual = 1).
    pub const fn num_languages(self) -> u32 {
        match self {
            Self::English => 1,
            Self::Chinese => 1,
            Self::Korean => 10,
        }
    }

    /// Number of speakers registered in the `spk2id` table (transcribed
    /// from each `config.json`).
    pub const fn n_speakers_active(self) -> u32 {
        match self {
            Self::English => 5, // EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default
            Self::Chinese => 1, // ZH
            Self::Korean => 1,  // KR
        }
    }
}

// ---- Shared VITS2 backbone axes (identical across the 3 variants) ---

/// `data.sampling_rate = 44100` (all variants).
pub(crate) const SAMPLE_RATE: u32 = 44_100;
/// `data.filter_length = 2048` (all variants). The n_fft of the mel
/// front-end.
pub(crate) const N_FFT: u32 = 2_048;
/// `data.hop_length = 512` (all variants). Product of `upsample_rates`
/// so the vocoder output aligns with the mel frame.
pub(crate) const HOP_LENGTH: u32 = 512;
/// `data.n_speakers = 256` (all variants, from the config). The
/// **capacity** of the speaker-embedding table (distinct from the number
/// of *registered* speakers in `spk2id`, which is per-variant).
pub(crate) const N_SPEAKERS_CAPACITY: u32 = 256;
/// `model.inter_channels = 192` (all variants).
pub(crate) const INTER_CHANNELS: u32 = 192;
/// `model.hidden_channels = 192` (all variants).
pub(crate) const HIDDEN_CHANNELS: u32 = 192;
/// `model.filter_channels = 768` (all variants).
pub(crate) const FILTER_CHANNELS: u32 = 768;
/// `model.n_heads = 2` (all variants).
pub(crate) const N_HEADS: u32 = 2;
/// `model.n_layers = 6` (all variants). Text-encoder Transformer depth.
pub(crate) const N_LAYERS: u32 = 6;
/// `model.n_layers_trans_flow = 3` (all variants). VITS2-specific
/// Transformer-based flow depth — the axis that distinguishes VITS2
/// from VITS.
pub(crate) const N_LAYERS_TRANS_FLOW: u32 = 3;
/// `model.gin_channels = 256` (all variants). Speaker-embedding
/// dimensionality piped through the flow / decoder as
/// global-conditioning.
pub(crate) const GIN_CHANNELS: u32 = 256;
/// `model.upsample_initial_channel = 512` (all variants). HiFi-GAN
/// generator input channel count.
pub(crate) const UPSAMPLE_INITIAL_CHANNEL: u32 = 512;
/// Product of `model.upsample_rates = [8, 8, 2, 2, 2]` = 512, must
/// equal `hop_length`.
pub(crate) const UPSAMPLE_TOTAL: u32 = 512;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_MELO_VARIANT: &str = "vokra.melotts.variant";
const KEY_MELO_SAMPLE_RATE: &str = "vokra.melotts.sample_rate";
const KEY_MELO_N_FFT: &str = "vokra.melotts.n_fft";
const KEY_MELO_HOP_LENGTH: &str = "vokra.melotts.hop_length";
const KEY_MELO_N_SPEAKERS_CAPACITY: &str = "vokra.melotts.n_speakers_capacity";
const KEY_MELO_N_SPEAKERS_ACTIVE: &str = "vokra.melotts.n_speakers_active";
const KEY_MELO_INTER_CHANNELS: &str = "vokra.melotts.inter_channels";
const KEY_MELO_HIDDEN_CHANNELS: &str = "vokra.melotts.hidden_channels";
const KEY_MELO_FILTER_CHANNELS: &str = "vokra.melotts.filter_channels";
const KEY_MELO_N_HEADS: &str = "vokra.melotts.n_heads";
const KEY_MELO_N_LAYERS: &str = "vokra.melotts.n_layers";
const KEY_MELO_N_LAYERS_TRANS_FLOW: &str = "vokra.melotts.n_layers_trans_flow";
const KEY_MELO_GIN_CHANNELS: &str = "vokra.melotts.gin_channels";
const KEY_MELO_UPSAMPLE_INITIAL_CHANNEL: &str = "vokra.melotts.upsample_initial_channel";
const KEY_MELO_UPSAMPLE_TOTAL: &str = "vokra.melotts.upsample_total";
const KEY_MELO_N_SYMBOLS: &str = "vokra.melotts.n_symbols";
const KEY_MELO_NUM_TONES: &str = "vokra.melotts.num_tones";
const KEY_MELO_NUM_LANGUAGES: &str = "vokra.melotts.num_languages";

/// Outcome of a MeloTTS conversion. Mirrors the sibling BF16
/// pass-through report shape (`written` / `skipped_non_float` /
/// `bf16_passthrough`) plus a leading `read` counter so the invariant
/// `read == written + skipped_non_float` holds for every well-formed
/// input.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MeloTtsReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so any tensor
    /// reaching this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]).
    pub bf16_passthrough: usize,
}

/// File-based MeloTTS converter (`vokra-cli convert --model
/// melotts-english`, `-chinese`, `-korean`).
///
/// Reads `input` (upstream `myshell-ai/MeloTTS-<lang>` flattened to
/// safetensors — the upstream ships `checkpoint.pth`, callers pre-flatten
/// offline mirror of the CSM / DAC / DFN3 pattern), writes a Vokra GGUF
/// to `output`. `variant` pins the language-specific axes; `license`
/// overrides the default `mit` provenance stamp (Whisper / kokoro-family
/// override pattern — see `convert_file_licensed` in `lib.rs`); pass
/// `None` to keep the built-in `mit` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_melotts_file(
    input: &Path,
    output: &Path,
    variant: MeloVariant,
    license: Option<&str>,
) -> Result<MeloTtsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, variant.upstream_hf());

    // Variant tag — the runtime dispatcher's fast path.
    b.add_string(KEY_MELO_VARIANT, variant.variant_tag());

    // Shared VITS2 backbone axes (all 3 variants).
    b.add_u32(KEY_MELO_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_MELO_N_FFT, N_FFT);
    b.add_u32(KEY_MELO_HOP_LENGTH, HOP_LENGTH);
    b.add_u32(KEY_MELO_N_SPEAKERS_CAPACITY, N_SPEAKERS_CAPACITY);
    b.add_u32(KEY_MELO_INTER_CHANNELS, INTER_CHANNELS);
    b.add_u32(KEY_MELO_HIDDEN_CHANNELS, HIDDEN_CHANNELS);
    b.add_u32(KEY_MELO_FILTER_CHANNELS, FILTER_CHANNELS);
    b.add_u32(KEY_MELO_N_HEADS, N_HEADS);
    b.add_u32(KEY_MELO_N_LAYERS, N_LAYERS);
    b.add_u32(KEY_MELO_N_LAYERS_TRANS_FLOW, N_LAYERS_TRANS_FLOW);
    b.add_u32(KEY_MELO_GIN_CHANNELS, GIN_CHANNELS);
    b.add_u32(KEY_MELO_UPSAMPLE_INITIAL_CHANNEL, UPSAMPLE_INITIAL_CHANNEL);
    b.add_u32(KEY_MELO_UPSAMPLE_TOTAL, UPSAMPLE_TOTAL);

    // Variant-specific axes.
    b.add_u32(KEY_MELO_N_SYMBOLS, variant.n_symbols());
    b.add_u32(KEY_MELO_NUM_TONES, variant.num_tones());
    b.add_u32(KEY_MELO_NUM_LANGUAGES, variant.num_languages());
    b.add_u32(KEY_MELO_N_SPEAKERS_ACTIVE, variant.n_speakers_active());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = mit (all 3 MeloTTS variants ship MIT per the
    // HF model-card front-matter, fetched 2026-07-30 — CLAUDE.md
    // 「ハルシネーション厳禁」). `license` overrides for callers who
    // obtained the weight under a different SPDX (see
    // `convert_file_licensed`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("mit".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.upstream_hf()),
    );

    let mut report = MeloTtsReport::default();
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

    /// Per-test scratch path in the system temp dir. Nanosecond suffix
    /// separates parallel `cargo test` runs.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-melotts-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Single-tensor BF16 safetensors buffer with a caller-supplied
    /// non-zero payload so a silent widen would flip a byte-identity
    /// assert (zero payloads round-trip trivially).
    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(
            bf16_bytes.len(),
            elems as usize * 2,
            "test fixture: BF16 payload len must be shape × 2"
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

    /// F32 + F16 mixed-dtype safetensors buffer.
    fn safetensors_f32_and_f16() -> Vec<u8> {
        let f32_vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_vals: [u16; 6] = [0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
        let f16_bytes: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = format!(
            r#"{{"enc_p.emb.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}},"flow.pre.weight":{{"dtype":"F16","shape":[2,3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + f16_bytes.len(),
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&f32_bytes);
        out.extend_from_slice(&f16_bytes);
        out
    }

    #[test]
    fn variant_metadata_is_stamped_english() {
        let (input_bytes, _) = safetensors_and_expected_bf16();
        let input = scratch_path("en-in");
        let output = scratch_path("en-out");
        std::fs::write(&input, &input_bytes).unwrap();

        convert_melotts_file(&input, &output, MeloVariant::English, None).expect("convert English");

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).expect("parse GGUF");

        // Variant identity + upstream provenance + arch dispatch tag.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(MeloVariant::English.name())
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_HF).and_then(|v| v.as_str()),
            Some(MeloVariant::English.upstream_hf())
        );
        assert_eq!(
            file.get(KEY_MELO_VARIANT).and_then(|v| v.as_str()),
            Some("english")
        );

        // Shared VITS2 axes.
        assert_eq!(
            file.get(KEY_MELO_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(44_100)
        );
        assert_eq!(
            file.get(KEY_MELO_INTER_CHANNELS).and_then(|v| v.as_u64()),
            Some(192)
        );
        assert_eq!(
            file.get(KEY_MELO_N_LAYERS_TRANS_FLOW)
                .and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            file.get(KEY_MELO_UPSAMPLE_TOTAL).and_then(|v| v.as_u64()),
            Some(u64::from(HOP_LENGTH))
        );

        // English-specific pins.
        assert_eq!(
            file.get(KEY_MELO_N_SYMBOLS).and_then(|v| v.as_u64()),
            Some(178)
        );
        assert_eq!(
            file.get(KEY_MELO_NUM_TONES).and_then(|v| v.as_u64()),
            Some(0),
            "English has no CJK tone head — the metadata pins 0 to make the absence explicit"
        );
        assert_eq!(
            file.get(KEY_MELO_NUM_LANGUAGES).and_then(|v| v.as_u64()),
            Some(1)
        );
        assert_eq!(
            file.get(KEY_MELO_N_SPEAKERS_ACTIVE)
                .and_then(|v| v.as_u64()),
            Some(5)
        );

        // Default license = mit (Permissive class).
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

    #[test]
    fn variant_metadata_is_stamped_chinese_and_korean() {
        for (v, expect_symbols, expect_tones, expect_langs, expect_active) in [
            (MeloVariant::Chinese, 112, 11, 1, 1),
            (MeloVariant::Korean, 219, 16, 10, 1),
        ] {
            let (input_bytes, _) = safetensors_and_expected_bf16();
            let input = scratch_path(&format!("{}-in", v.variant_tag()));
            let output = scratch_path(&format!("{}-out", v.variant_tag()));
            std::fs::write(&input, &input_bytes).unwrap();

            convert_melotts_file(&input, &output, v, None).unwrap_or_else(|_| {
                panic!("convert {} must succeed", v.variant_tag());
            });
            let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
            assert_eq!(
                file.get(chunks::KEY_MODEL_NAME).and_then(|s| s.as_str()),
                Some(v.name())
            );
            assert_eq!(
                file.get(KEY_MELO_VARIANT).and_then(|s| s.as_str()),
                Some(v.variant_tag())
            );
            assert_eq!(
                file.get(KEY_MELO_N_SYMBOLS).and_then(|s| s.as_u64()),
                Some(expect_symbols)
            );
            assert_eq!(
                file.get(KEY_MELO_NUM_TONES).and_then(|s| s.as_u64()),
                Some(expect_tones)
            );
            assert_eq!(
                file.get(KEY_MELO_NUM_LANGUAGES).and_then(|s| s.as_u64()),
                Some(expect_langs)
            );
            assert_eq!(
                file.get(KEY_MELO_N_SPEAKERS_ACTIVE)
                    .and_then(|s| s.as_u64()),
                Some(expect_active)
            );
            std::fs::remove_file(&input).ok();
            std::fs::remove_file(&output).ok();
        }
    }

    fn safetensors_and_expected_bf16() -> (Vec<u8>, Vec<u8>) {
        // Non-zero BF16 patterns so a byte-identity assert catches a
        // silent widen / downcast.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("enc_p.emb.weight", &[2, 3], &bf16);
        (input_bytes, bf16)
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (input_bytes, bf16_payload) = safetensors_and_expected_bf16();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_melotts_file(&input, &output, MeloVariant::English, None)
            .expect("convert must succeed");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let info = file
            .tensor_info("enc_p.emb.weight")
            .expect("BF16 tensor present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "BF16 stays GGUF type 30 — no convert-time widening"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16_payload.as_slice(),
            "BF16 payload byte-identical to input"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through() {
        let input_bytes = safetensors_f32_and_f16();
        let input = scratch_path("mix-in");
        let output = scratch_path("mix-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_melotts_file(&input, &output, MeloVariant::Chinese, None).unwrap();
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16-only input keeps the BF16 subset counter at 0"
        );

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let f32_info = file
            .tensor_info("enc_p.emb.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32);
        let f16_info = file
            .tensor_info("flow.pre.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn license_override_reclassifies_provenance() {
        let (input_bytes, _) = safetensors_and_expected_bf16();
        let input = scratch_path("lic-in");
        let output = scratch_path("lic-out");
        std::fs::write(&input, &input_bytes).unwrap();

        convert_melotts_file(&input, &output, MeloVariant::English, Some("cc-by-nc-4.0"))
            .expect("convert must accept license override");

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("cc-by-nc-4.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::NonCommercial.as_str()),
            "cc-by-nc-4.0 must classify as NonCommercial via from_license_str"
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
