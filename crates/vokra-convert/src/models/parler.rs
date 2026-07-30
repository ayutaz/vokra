//! **Parler-TTS family** (`parler-tts/parler-tts-mini-multilingual-v1.1`
//! and `ai4bharat/indic-parler-tts`, apache-2.0): safetensors → GGUF
//! conversion (implementer C wave, 2026-07-30).
//!
//! Input: an upstream Parler-TTS release — both variants ship
//! `model.safetensors` directly (no torch-pickle prepare step). Output:
//! a GGUF carrying every float tensor plus the `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks a future native Parler-TTS loader
//! will read.
//!
//! # Architecture (primary source, 2026-07-30 CC fetch)
//!
//! Parler-TTS is a **decoder-only LM over discrete DAC codes**
//! conditioned on a T5 text-encoder embedding of a natural-language
//! description ("A female speaker with a clear voice speaks slowly…").
//! The pipeline: text description → T5 encoder (`text_encoder.*`) →
//! cross-attention into Parler decoder (`decoder.*`) → 9 parallel
//! DAC codebooks (24 kHz DAC, delay-shifted pattern) → DAC decoder
//! → 44.1 kHz PCM.
//!
//! Every hparam below is transcribed verbatim from `huggingface.co/
//! parler-tts/parler-tts-mini-multilingual-v1.1/raw/main/config.json`
//! (fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - `model_type = "parler_tts"`
//! - `architectures = ["ParlerTTSForConditionalGeneration"]`
//! - **Text encoder** (T5-family): `d_model=1024`, `num_layers=24`,
//!   `num_heads=16`, `d_ff=2816`, `vocab_size=32128`
//! - **Audio encoder** (DAC, external, referenced only): `model_type="dac"`,
//!   `codebook_size=1024`, `sampling_rate=44100`
//! - **Decoder** (Parler-specific): `hidden_size=1024`,
//!   `num_hidden_layers=24`, `num_attention_heads=16`,
//!   `num_key_value_heads=16` (MHA), `ffn_dim=4096`,
//!   `num_codebooks=9`, `vocab_size=1088`
//! - `vocab_size = 90714` (top-level, description tokenizer merged
//!   with the decoder's audio-code alphabet)
//! - `prompt_cross_attention = false`
//!
//! **Variant axis** — the Indic Parler release (C6) shares the
//! architecture end-to-end. It is a fine-tune of the same Parler-TTS
//! decoder on Indic-language data; the tensor topology and every
//! primary hparam listed above are unchanged. The variant tag rides
//! `vokra.parler.variant` so a runtime dispatcher can pick a tokenizer
//! / language table without inspecting the tensor shapes.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! upstream safetensors names. BF16 stays GGUF type 30
//! (`GgmlType::BF16`) — no convert-time widening; runtime widens BF16 →
//! f32 losslessly via the shared choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # DAC dependency (external)
//!
//! Parler-TTS emits 9-codebook DAC token streams; the audio decoder
//! itself is the standalone `descript/dac_44khz` release the runtime
//! consumes via the shared `crate::models::dac` converter path
//! (`ModelKind::Dac`, `crates/vokra-convert/src/models/dac.rs`). This
//! converter therefore emits **only** the Parler LM + T5 tensor pack;
//! the caller wires a separate DAC GGUF at runtime bind time (mirror
//! of the CosyVoice2 + Mimi split — CosyVoice2's GGUF also does not
//! carry the codec).
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream Parler-TTS Python pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off).
//!
//! # No ONNX (permanent)
//!
//! Both Parler variants ship safetensors directly; this converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in a future `crates/vokra-models/src/parler/` module
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Parler-TTS GGUFs.
pub(crate) const ARCH: &str = "parler_tts";
/// Model category tag — `tts`.
pub(crate) const CATEGORY: &str = "tts";

/// The Parler-TTS release variants. Each pins the canonical HF release
/// slug + variant-specific tag (both share the same tensor topology).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParlerVariant {
    /// `parler-tts/parler-tts-mini-multilingual-v1.1` (apache-2.0).
    /// Multilingual base (8 European languages).
    MiniMultilingual,
    /// `ai4bharat/indic-parler-tts` (apache-2.0, gated=auto).
    /// Fine-tune on Indic languages (21 languages: Assamese, Bengali,
    /// Bodo, Chhattisgarhi, Dogri, English, Gujarati, Hindi, Kannada,
    /// Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi,
    /// Sanskrit, Sindhi, Tamil, Telugu, Urdu). Same tensor topology as
    /// the multilingual base — this is a fine-tune, not a re-arch.
    IndicParler,
}

impl ParlerVariant {
    /// Canonical `vokra.model.name` for this variant (matches the HF
    /// release slug tail).
    pub const fn name(self) -> &'static str {
        match self {
            Self::MiniMultilingual => "parler-tts-mini-multilingual-v1.1",
            Self::IndicParler => "indic-parler-tts",
        }
    }

    /// Short `vokra.parler.variant` tag written on the GGUF.
    pub const fn variant_tag(self) -> &'static str {
        match self {
            Self::MiniMultilingual => "mini-multilingual",
            Self::IndicParler => "indic",
        }
    }

    /// Upstream HF repo slug (recorded under
    /// `vokra.provenance.upstream_hf`).
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::MiniMultilingual => "parler-tts/parler-tts-mini-multilingual-v1.1",
            Self::IndicParler => "ai4bharat/indic-parler-tts",
        }
    }
}

// ---- Hparams (transcribed verbatim from upstream config.json) -----------

// Text encoder (T5-family):
pub(crate) const TEXT_ENCODER_D_MODEL: u32 = 1_024;
pub(crate) const TEXT_ENCODER_NUM_LAYERS: u32 = 24;
pub(crate) const TEXT_ENCODER_NUM_HEADS: u32 = 16;
pub(crate) const TEXT_ENCODER_D_FF: u32 = 2_816;
pub(crate) const TEXT_ENCODER_VOCAB_SIZE: u32 = 32_128;

// Audio encoder (DAC, external — recorded for provenance completeness):
pub(crate) const AUDIO_ENCODER_CODEBOOK_SIZE: u32 = 1_024;
pub(crate) const AUDIO_ENCODER_SAMPLING_RATE: u32 = 44_100;

// Decoder (Parler-specific):
pub(crate) const DECODER_HIDDEN_SIZE: u32 = 1_024;
pub(crate) const DECODER_NUM_HIDDEN_LAYERS: u32 = 24;
pub(crate) const DECODER_NUM_ATTENTION_HEADS: u32 = 16;
pub(crate) const DECODER_NUM_KEY_VALUE_HEADS: u32 = 16;
pub(crate) const DECODER_FFN_DIM: u32 = 4_096;
pub(crate) const DECODER_NUM_CODEBOOKS: u32 = 9;
pub(crate) const DECODER_VOCAB_SIZE: u32 = 1_088;

// Top-level:
pub(crate) const VOCAB_SIZE_TOP: u32 = 90_714;
pub(crate) const PROMPT_CROSS_ATTENTION: bool = false;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.parler.variant";

const KEY_TEXT_ENCODER_D_MODEL: &str = "vokra.parler.text_encoder.d_model";
const KEY_TEXT_ENCODER_NUM_LAYERS: &str = "vokra.parler.text_encoder.num_layers";
const KEY_TEXT_ENCODER_NUM_HEADS: &str = "vokra.parler.text_encoder.num_heads";
const KEY_TEXT_ENCODER_D_FF: &str = "vokra.parler.text_encoder.d_ff";
const KEY_TEXT_ENCODER_VOCAB_SIZE: &str = "vokra.parler.text_encoder.vocab_size";

const KEY_AUDIO_ENCODER_CODEBOOK_SIZE: &str = "vokra.parler.audio_encoder.codebook_size";
const KEY_AUDIO_ENCODER_SAMPLING_RATE: &str = "vokra.parler.audio_encoder.sampling_rate";

const KEY_DECODER_HIDDEN_SIZE: &str = "vokra.parler.decoder.hidden_size";
const KEY_DECODER_NUM_HIDDEN_LAYERS: &str = "vokra.parler.decoder.num_hidden_layers";
const KEY_DECODER_NUM_ATTENTION_HEADS: &str = "vokra.parler.decoder.num_attention_heads";
const KEY_DECODER_NUM_KEY_VALUE_HEADS: &str = "vokra.parler.decoder.num_key_value_heads";
const KEY_DECODER_FFN_DIM: &str = "vokra.parler.decoder.ffn_dim";
const KEY_DECODER_NUM_CODEBOOKS: &str = "vokra.parler.decoder.num_codebooks";
const KEY_DECODER_VOCAB_SIZE: &str = "vokra.parler.decoder.vocab_size";

const KEY_VOCAB_SIZE_TOP: &str = "vokra.parler.vocab_size";
const KEY_PROMPT_CROSS_ATTENTION: &str = "vokra.parler.prompt_cross_attention";

/// Outcome of a Parler-TTS conversion. Mirrors the shared BF16
/// pass-through report shape (`written` / `skipped_non_float` /
/// `bf16_passthrough`) plus a leading `read` counter.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParlerReport {
    /// Total tensors observed in the safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive — the safetensors reader
    /// accepts only float dtypes).
    pub skipped_non_float: usize,
    /// BF16 tensors on the pass-through arm.
    pub bf16_passthrough: usize,
}

/// File-based Parler-TTS converter (`vokra-cli convert --model parler-tts`
/// or `--model indic-parler-tts`).
///
/// Reads `input` (upstream Parler-TTS `model.safetensors`), writes a
/// Vokra GGUF to `output`. `variant` pins the release identity; `license`
/// overrides the default `apache-2.0` provenance stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures; [`ConvertError::Parse`] for
/// malformed safetensors input; [`ConvertError::Gguf`] for GGUF writer
/// failure.
pub fn convert_parler_file(
    input: &Path,
    output: &Path,
    variant: ParlerVariant,
    license: Option<&str>,
) -> Result<ParlerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_VARIANT, variant.variant_tag());

    // Text encoder axes.
    b.add_u32(KEY_TEXT_ENCODER_D_MODEL, TEXT_ENCODER_D_MODEL);
    b.add_u32(KEY_TEXT_ENCODER_NUM_LAYERS, TEXT_ENCODER_NUM_LAYERS);
    b.add_u32(KEY_TEXT_ENCODER_NUM_HEADS, TEXT_ENCODER_NUM_HEADS);
    b.add_u32(KEY_TEXT_ENCODER_D_FF, TEXT_ENCODER_D_FF);
    b.add_u32(KEY_TEXT_ENCODER_VOCAB_SIZE, TEXT_ENCODER_VOCAB_SIZE);

    // Audio encoder (external DAC) provenance axes.
    b.add_u32(KEY_AUDIO_ENCODER_CODEBOOK_SIZE, AUDIO_ENCODER_CODEBOOK_SIZE);
    b.add_u32(KEY_AUDIO_ENCODER_SAMPLING_RATE, AUDIO_ENCODER_SAMPLING_RATE);

    // Decoder axes.
    b.add_u32(KEY_DECODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE);
    b.add_u32(KEY_DECODER_NUM_HIDDEN_LAYERS, DECODER_NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_DECODER_NUM_ATTENTION_HEADS, DECODER_NUM_ATTENTION_HEADS);
    b.add_u32(KEY_DECODER_NUM_KEY_VALUE_HEADS, DECODER_NUM_KEY_VALUE_HEADS);
    b.add_u32(KEY_DECODER_FFN_DIM, DECODER_FFN_DIM);
    b.add_u32(KEY_DECODER_NUM_CODEBOOKS, DECODER_NUM_CODEBOOKS);
    b.add_u32(KEY_DECODER_VOCAB_SIZE, DECODER_VOCAB_SIZE);

    // Top-level.
    b.add_u32(KEY_VOCAB_SIZE_TOP, VOCAB_SIZE_TOP);
    b.add_bool(KEY_PROMPT_CROSS_ATTENTION, PROMPT_CROSS_ATTENTION);

    // Default license = apache-2.0 (both variants; upstream card
    // front-matter, fetched 2026-07-30). `IndicParler` is `gated=auto`
    // on HF — the gate is access control, the license itself is
    // apache-2.0 per the card.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("apache-2.0".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.upstream_hf()),
    );

    let mut report = ParlerReport::default();
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

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-parler-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
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

    fn synth_bf16() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        (
            safetensors_one_bf16(
                "decoder.model.decoder.layers.0.self_attn.q_proj.weight",
                &[2, 3],
                &bf16,
            ),
            bf16,
        )
    }

    #[test]
    fn bf16_passthrough_and_hparams() {
        let (input_bytes, bf16_payload) = synth_bf16();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report =
            convert_parler_file(&input, &output, ParlerVariant::MiniMultilingual, None).unwrap();
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let info = file
            .tensor_info("decoder.model.decoder.layers.0.self_attn.q_proj.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16_payload.as_slice());

        // Every one of the 15 hparam pins is transcribed correctly.
        for (k, expect) in [
            (KEY_TEXT_ENCODER_D_MODEL, 1_024u64),
            (KEY_TEXT_ENCODER_NUM_LAYERS, 24),
            (KEY_TEXT_ENCODER_NUM_HEADS, 16),
            (KEY_TEXT_ENCODER_D_FF, 2_816),
            (KEY_TEXT_ENCODER_VOCAB_SIZE, 32_128),
            (KEY_AUDIO_ENCODER_CODEBOOK_SIZE, 1_024),
            (KEY_AUDIO_ENCODER_SAMPLING_RATE, 44_100),
            (KEY_DECODER_HIDDEN_SIZE, 1_024),
            (KEY_DECODER_NUM_HIDDEN_LAYERS, 24),
            (KEY_DECODER_NUM_ATTENTION_HEADS, 16),
            (KEY_DECODER_NUM_KEY_VALUE_HEADS, 16),
            (KEY_DECODER_FFN_DIM, 4_096),
            (KEY_DECODER_NUM_CODEBOOKS, 9),
            (KEY_DECODER_VOCAB_SIZE, 1_088),
            (KEY_VOCAB_SIZE_TOP, 90_714),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(expect),
                "{k} pin"
            );
        }

        // Variant identity, provenance defaults.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(ParlerVariant::MiniMultilingual.name())
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("mini-multilingual")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn indic_variant_stamps_distinct_provenance() {
        let (input_bytes, _) = synth_bf16();
        let input = scratch_path("indic-in");
        let output = scratch_path("indic-out");
        std::fs::write(&input, &input_bytes).unwrap();

        convert_parler_file(&input, &output, ParlerVariant::IndicParler, None).unwrap();
        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(ParlerVariant::IndicParler.name())
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_HF).and_then(|v| v.as_str()),
            Some(ParlerVariant::IndicParler.upstream_hf())
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("indic")
        );
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
