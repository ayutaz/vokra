//! **Chatterbox-Multilingual**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 3, 2026-07-24).
//!
//! Input: the upstream `ResembleAI/chatterbox` T3 safetensors — for the
//! multilingual variant, that is `t3_mtl23ls_v{2,3}.safetensors` (the
//! `src/chatterbox/mtl_tts.py::MULTILINGUAL_T3_MODELS` table). Output: a
//! GGUF carrying every float tensor plus the `vokra.chatterbox.*` and
//! `vokra.model.*` / `vokra.provenance.*` metadata chunks the native
//! Chatterbox implementation (`crates/vokra-models/src/chatterbox/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.chatterbox.*`
//!   chunk group is transcribed **verbatim** from the primary source
//!   (`src/chatterbox/models/t3/llama_configs.py::LLAMA_520M_CONFIG_DICT`
//!   for the backbone, `src/chatterbox/models/t3/modules/t3_config.py`
//!   for the T3 wrapper, `src/chatterbox/models/s3gen/const.py::S3GEN_SR`
//!   for the sample rate). No axis is invented; any tensor whose shape
//!   disagrees with these values fails the runtime shape gate loudly
//!   (FR-EX-08, `ChatterboxConfig::validate_for_forward`).
//! - **No side-car config** — Chatterbox does not ship a `config.json`;
//!   the T3 wrapper stores every hparam in Python code, and the release
//!   uses the same Llama_520M / T3 config for both the English-only
//!   default and every multilingual checkpoint (only `text_tokens_dict_size`
//!   changes). The converter picks the variant based on the caller-
//!   provided `variant` argument (`ChatterboxVariant::Multilingual` is the
//!   default), which the runtime cross-checks against the checkpoint's
//!   text-embedding row count when it binds real weights.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 contract). Real-weight binding is a
//! follow-up wave gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 tensor through unchanged so a future
//! `ChatterboxWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Chatterbox is distributed as safetensors + a Python pipeline; the
//! pipeline is re-implemented natively in
//! `crates/vokra-models/src/chatterbox/` (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4). This converter never touches
//! ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Chatterbox GGUFs — kept in sync with the runtime
/// constant `vokra-models::chatterbox::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "chatterbox";
/// `vokra.model.name` written for the canonical multilingual T3 GGUF.
pub(crate) const NAME_MULTILINGUAL: &str = "chatterbox-multilingual-v3";
/// `vokra.model.name` written for the canonical English-only T3 GGUF.
pub(crate) const NAME_ENGLISH: &str = "chatterbox-english";

// --- vokra.chatterbox.* keys (kept as constants in the converter; the
// runtime duplicates the strings in `crates/vokra-models/src/chatterbox/mod.rs`
// — the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro converters use applies) ---

const KEY_SAMPLE_RATE: &str = "vokra.chatterbox.sample_rate";

// T3 vocab / prompt axes
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.chatterbox.arch.text_vocab_size";
const KEY_SPEECH_VOCAB_SIZE: &str = "vokra.chatterbox.arch.speech_vocab_size";
const KEY_MAX_TEXT_TOKENS: &str = "vokra.chatterbox.arch.max_text_tokens";
const KEY_MAX_SPEECH_TOKENS: &str = "vokra.chatterbox.arch.max_speech_tokens";
const KEY_SPEAKER_EMBED_SIZE: &str = "vokra.chatterbox.arch.speaker_embed_size";

// Llama backbone axes
const KEY_HIDDEN_DIM: &str = "vokra.chatterbox.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.chatterbox.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.chatterbox.arch.n_head";
const KEY_N_HEAD_KV: &str = "vokra.chatterbox.arch.n_head_kv";
const KEY_HEAD_DIM: &str = "vokra.chatterbox.arch.head_dim";
const KEY_FFN_DIM: &str = "vokra.chatterbox.arch.ffn_dim";

// Norm / RoPE
const KEY_ROPE_BASE: &str = "vokra.chatterbox.arch.rope_base";
const KEY_RMS_NORM_EPS: &str = "vokra.chatterbox.arch.rms_norm_eps";

// Variant tag (multilingual vs english-only) — surfaces on the report and
// lets a runtime discriminate without walking every tensor.
const KEY_VARIANT: &str = "vokra.chatterbox.variant";

// --- Transcribed constants (primary source: the chatterbox Python source
// tree, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」) ------------

/// PCM sample rate — `src/chatterbox/models/s3gen/const.py::S3GEN_SR`.
const CHATTERBOX_SAMPLE_RATE: u32 = 24_000;

// T3 (Token-to-Token TTS) axes — `src/chatterbox/models/t3/modules/t3_config.py`.
/// `T3Config.multilingual()` sets `text_tokens_dict_size = 2454`.
const TEXT_VOCAB_MULTILINGUAL: u32 = 2454;
/// `T3Config.english_only()` sets `text_tokens_dict_size = 704`.
const TEXT_VOCAB_ENGLISH: u32 = 704;
/// `T3Config.speech_tokens_dict_size = 8194` (both variants).
const SPEECH_VOCAB_SIZE: u32 = 8194;
/// `T3Config.max_text_tokens = 2048`.
const MAX_TEXT_TOKENS: u32 = 2048;
/// `T3Config.max_speech_tokens = 4096`.
const MAX_SPEECH_TOKENS: u32 = 4096;
/// `T3Config.speaker_embed_size = 256`.
const SPEAKER_EMBED_SIZE: u32 = 256;

// Llama_520M axes — `src/chatterbox/models/t3/llama_configs.py::LLAMA_520M_CONFIG_DICT`.
const HIDDEN_DIM: u32 = 1024;
const N_LAYER: u32 = 30;
const N_HEAD: u32 = 16;
/// MHA (`num_key_value_heads == num_attention_heads`).
const N_HEAD_KV: u32 = 16;
const HEAD_DIM: u32 = 64;
const FFN_DIM: u32 = 4096;

// Norm / RoPE
const ROPE_BASE: f32 = 500_000.0;
const RMS_NORM_EPS: f32 = 1e-5;

/// Which Chatterbox T3 variant this GGUF represents. The two variants share
/// the Llama_520M backbone byte-for-byte — only `text_vocab_size` differs
/// (2454 multilingual vs 704 English-only), so this tag is what the runtime
/// checks to route the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatterboxVariant {
    /// `t3_mtl23ls_v{2,3}.safetensors` — 23 languages
    /// (`mtl_tts.py::SUPPORTED_LANGUAGES`). Default because the canonical
    /// Phase 3 landing is the multilingual T3.
    #[default]
    Multilingual,
    /// Default `t3.safetensors` — English-only baseline.
    ///
    /// The dispatch path in `vokra-convert::convert_file_licensed`
    /// intentionally does not surface an English CLI arm yet (the Phase 3
    /// landing is multilingual-first); this variant is produced from
    /// tests via [`convert_variant`] and is kept in the public enum so
    /// the runtime discriminator stays two-valued for a future
    /// `--variant english` flag.
    #[allow(dead_code)]
    English,
}

impl ChatterboxVariant {
    fn text_vocab(self) -> u32 {
        match self {
            Self::Multilingual => TEXT_VOCAB_MULTILINGUAL,
            Self::English => TEXT_VOCAB_ENGLISH,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Multilingual => NAME_MULTILINGUAL,
            Self::English => NAME_ENGLISH,
        }
    }

    /// Wire tag written into `vokra.chatterbox.variant`.
    fn tag(self) -> &'static str {
        match self {
            Self::Multilingual => "multilingual",
            Self::English => "english",
        }
    }
}

/// Outcome of a Chatterbox conversion.
#[derive(Debug, Default)]
pub(crate) struct ChatterboxReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub(crate) bf16_passthrough: usize,
    /// Variant this conversion labelled.
    pub(crate) variant: ChatterboxVariant,
    /// Operator-facing diagnostics (never fail the conversion — the runtime
    /// is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Chatterbox safetensors buffer into a populated GGUF builder,
/// tagging the emitted GGUF as the **multilingual** T3 variant.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.chatterbox.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive` (MIT).
///
/// # No side-car config
///
/// Chatterbox does not ship a `config.json` — every hparam is transcribed
/// from the Python source and is fixed across the whole release, so the
/// converter takes no config path.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ChatterboxReport), ConvertError> {
    convert_variant(bytes, ChatterboxVariant::Multilingual)
}

/// Converts a Chatterbox safetensors buffer with an explicit variant tag.
/// The English-only path is exposed for symmetry with the runtime's
/// `ChatterboxConfig::chatterbox_english`; the multilingual path is the
/// canonical Phase 3 landing.
pub(crate) fn convert_variant(
    bytes: Vec<u8>,
    variant: ChatterboxVariant,
) -> Result<(GgufBuilder, ChatterboxReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    write_hparams(&mut b, variant);
    // Self-describing redistribution: the artifact carries its own licence.
    // Chatterbox ships MIT per `github.com/resemble-ai/chatterbox/LICENSE`
    // (Copyright (c) 2025 Resemble AI, fetched 2026-07-24 — CLAUDE.md
    // 「ハルシネーション厳禁」).
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some(variant.name()),
        Some("ResembleAI/chatterbox (MIT — Copyright (c) 2025 Resemble AI)"),
    );

    let mut report = ChatterboxReport {
        variant,
        ..ChatterboxReport::default()
    };
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2 + moshi + voxtral): a Chatterbox T3
            // safetensors served in BF16 (`torch_dtype: bfloat16`) hits
            // this arm. Emit as GGUF type 30 verbatim; runtime widens on
            // load via `decode_bf16` (exact, `bits << 16`).
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
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). The \
             upstream Chatterbox release ships safetensors directly; a real \
             conversion needs a real `t3_mtl23ls_v{2,3}.safetensors` (or the \
             English-only `t3.safetensors`) as input. The BF16 pass-through \
             path is now wired (2026-07-25), so this state is only reachable \
             when the release contains no F32 / F16 / BF16 float tensors at all."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.chatterbox.*` chunk group from the transcribed
/// constants above (primary source: the chatterbox Python source tree).
fn write_hparams(b: &mut GgufBuilder, variant: ChatterboxVariant) {
    b.add_u32(KEY_SAMPLE_RATE, CHATTERBOX_SAMPLE_RATE);
    b.add_string(KEY_VARIANT, variant.tag());

    // T3 axes
    b.add_u32(KEY_TEXT_VOCAB_SIZE, variant.text_vocab());
    b.add_u32(KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE);
    b.add_u32(KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS);
    b.add_u32(KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS);
    b.add_u32(KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE);

    // Llama_520M backbone
    b.add_u32(KEY_HIDDEN_DIM, HIDDEN_DIM);
    b.add_u32(KEY_N_LAYER, N_LAYER);
    b.add_u32(KEY_N_HEAD, N_HEAD);
    b.add_u32(KEY_N_HEAD_KV, N_HEAD_KV);
    b.add_u32(KEY_HEAD_DIM, HEAD_DIM);
    b.add_u32(KEY_FFN_DIM, FFN_DIM);

    // Norm / RoPE
    b.add_f32(KEY_ROPE_BASE, ROPE_BASE);
    b.add_f32(KEY_RMS_NORM_EPS, RMS_NORM_EPS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the report
        // counts a non-zero write. The tensor name mirrors an upstream T3
        // scaffold name (`text_emb.weight`).
        let header = r#"{"text_emb.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    /// A single F16 tensor at the top of the file (shape [2,3] → 6 elements ×
    /// 2 bytes = 12 bytes). A real T3 safetensors is likely served in F16,
    /// so the F16 leg of the union match arm must be reachable.
    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header = r#"{"text_emb.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor — the safetensors reader accepts BF16 (per M4-06
    /// moshiko), so BF16 tensors reach `convert()` and MUST land in
    /// `skipped_non_float`, not silently dropped.
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"text_emb.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_f32(file: &GgufFile, key: &str) -> f32 {
        match file.get(key) {
            Some(GgufMetadataValue::F32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::chatterbox::EXPECTED_ARCH`.
        assert_eq!(ARCH, "chatterbox");
    }

    #[test]
    fn name_strings_match_hf_model_ids() {
        // Kept in sync with `huggingface.co/ResembleAI/chatterbox` and its
        // `t3_mtl23ls_v3.safetensors` (multilingual) / default `t3.safetensors`
        // (English-only) variants.
        assert_eq!(NAME_MULTILINGUAL, "chatterbox-multilingual-v3");
        assert_eq!(NAME_ENGLISH, "chatterbox-english");
    }

    /// The transcribed constants must equal the primary-source values —
    /// changing any of these silently mis-shapes the LLM backbone.
    #[test]
    fn transcribed_constants_match_primary_source() {
        assert_eq!(CHATTERBOX_SAMPLE_RATE, 24_000);
        assert_eq!(TEXT_VOCAB_MULTILINGUAL, 2454);
        assert_eq!(TEXT_VOCAB_ENGLISH, 704);
        assert_eq!(SPEECH_VOCAB_SIZE, 8194);
        assert_eq!(MAX_TEXT_TOKENS, 2048);
        assert_eq!(MAX_SPEECH_TOKENS, 4096);
        assert_eq!(SPEAKER_EMBED_SIZE, 256);
        assert_eq!(HIDDEN_DIM, 1024);
        assert_eq!(N_LAYER, 30);
        assert_eq!(N_HEAD, 16);
        assert_eq!(N_HEAD_KV, 16);
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(FFN_DIM, 4096);
        // MHA algebra
        assert_eq!(HIDDEN_DIM, N_HEAD * HEAD_DIM);
        assert_eq!(N_HEAD_KV, N_HEAD, "Llama_520M is MHA, not GQA");
        assert!((ROPE_BASE - 500_000.0).abs() < 1e-3);
        assert!((RMS_NORM_EPS - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.variant, ChatterboxVariant::Multilingual);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MULTILINGUAL)
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("multilingual")
        );

        // Every transcribed U32 hparam round-trips verbatim under the
        // CosyVoice3-style `vokra.chatterbox.*` prefix.
        for (key, want) in [
            (KEY_SAMPLE_RATE, CHATTERBOX_SAMPLE_RATE),
            (KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_MULTILINGUAL),
            (KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE),
            (KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS),
            (KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS),
            (KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE),
            (KEY_HIDDEN_DIM, HIDDEN_DIM),
            (KEY_N_LAYER, N_LAYER),
            (KEY_N_HEAD, N_HEAD),
            (KEY_N_HEAD_KV, N_HEAD_KV),
            (KEY_HEAD_DIM, HEAD_DIM),
            (KEY_FFN_DIM, FFN_DIM),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }
        assert!((get_f32(&file, KEY_ROPE_BASE) - ROPE_BASE).abs() < 1e-1);
        assert!((get_f32(&file, KEY_RMS_NORM_EPS) - RMS_NORM_EPS).abs() < 1e-12);

        // Provenance: MIT permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_MULTILINGUAL)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("MIT")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn english_variant_swaps_only_text_vocab_and_name() {
        let (builder, report) =
            convert_variant(minimal_safetensors_one_f32(), ChatterboxVariant::English)
                .expect("convert");
        assert_eq!(report.variant, ChatterboxVariant::English);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_ENGLISH)
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("english")
        );
        // Text-vocab flips to 704…
        assert_eq!(get_u32(&file, KEY_TEXT_VOCAB_SIZE), TEXT_VOCAB_ENGLISH);
        // …but every other axis matches the multilingual path.
        assert_eq!(get_u32(&file, KEY_SPEECH_VOCAB_SIZE), SPEECH_VOCAB_SIZE);
        assert_eq!(get_u32(&file, KEY_HIDDEN_DIM), HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_N_LAYER), N_LAYER);
        assert_eq!(get_u32(&file, KEY_N_HEAD), N_HEAD);
        assert_eq!(get_u32(&file, KEY_HEAD_DIM), HEAD_DIM);
        assert_eq!(get_u32(&file, KEY_FFN_DIM), FFN_DIM);
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        // Empty safetensors → the runtime's `ChatterboxWeights::from_gguf`
        // would fail loudly at bind time, but the converter itself succeeds
        // and reports the situation so the operator sees it now.
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union match
    /// arm. A real Chatterbox T3 checkpoint is likely served in F16 or BF16;
    /// a typo dropping `| GgmlType::F16` would silently bin every F16 tensor
    /// into `skipped_non_float`.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1, "F16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "F16 must not land in the skipped counter"
        );

        // The tensor survives the round trip under its upstream name and
        // preserves its F16 dtype (payload is 12 bytes = 6 × F16).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("text_emb.weight").expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for the
    /// Chatterbox T3 release candidate, `torch_dtype: bfloat16`) must
    /// reach the pass-through arm, emit as GGUF type 30 verbatim, and
    /// increment `bf16_passthrough`. Mirror of qwen3-tts / vibevoice /
    /// voxcpm2's `bf16_tensor_passes_through_verbatim` and moshi's
    /// `assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture.
    /// Removing the pin outright would let a latent silent-widen slip in
    /// undetected; rewriting to the passes-through invariant keeps the
    /// regression guard.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm and increment `written`"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        // The tensor survives the round trip under its upstream name and
        // preserves its BF16 dtype (no convert-time widening — runtime
        // widens on load via `decode_bf16`, exact via `bits << 16`).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("text_emb.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "BF16 payload = 6 elements × 2 bytes = 12 bytes"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation. A malformed
    /// input must surface as `Err(ConvertError::Parse(_))`, not as a silently
    /// empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer — shorter than the mandatory 8-byte header
        // length prefix, so `SafetensorsFile::parse` returns `Truncated`.
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 2: declared header length runs off the end of the buffer.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 3: valid length prefix but malformed JSON body.
        let bad_json = b"{not-json";
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("malformed JSON must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// Every `vokra.chatterbox.*` key uses the same prefix — a regression
    /// where a key crossed into another model's namespace (e.g.
    /// `vokra.chatterbox2.*`) would still round-trip in isolation but would
    /// misroute at the runtime dispatch layer.
    #[test]
    fn every_metadata_key_carries_the_chatterbox_prefix() {
        for key in [
            KEY_SAMPLE_RATE,
            KEY_TEXT_VOCAB_SIZE,
            KEY_SPEECH_VOCAB_SIZE,
            KEY_MAX_TEXT_TOKENS,
            KEY_MAX_SPEECH_TOKENS,
            KEY_SPEAKER_EMBED_SIZE,
            KEY_HIDDEN_DIM,
            KEY_N_LAYER,
            KEY_N_HEAD,
            KEY_N_HEAD_KV,
            KEY_HEAD_DIM,
            KEY_FFN_DIM,
            KEY_ROPE_BASE,
            KEY_RMS_NORM_EPS,
            KEY_VARIANT,
        ] {
            assert!(
                key.starts_with("vokra.chatterbox."),
                "{key} must live under the vokra.chatterbox.* prefix"
            );
        }
    }
}
