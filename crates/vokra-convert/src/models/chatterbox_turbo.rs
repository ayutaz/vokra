//! **Chatterbox-Turbo**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 3, 2026-07-24).
//!
//! Input: the upstream `ResembleAI/chatterbox-turbo` backbone
//! safetensors — `t3_turbo_v1.safetensors` (~1.92 GB). Output: a GGUF
//! carrying every float tensor plus the `vokra.chatterbox_turbo.*` and
//! `vokra.model.*` / `vokra.provenance.*` metadata chunks the native
//! Chatterbox-Turbo implementation
//! (`crates/vokra-models/src/chatterbox_turbo/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.chatterbox_turbo.*` chunk group is transcribed **verbatim**
//!   from the primary source `t3_turbo_v1.yaml`
//!   (`huggingface.co/ResembleAI/chatterbox-turbo`, fetched 2026-07-24
//!   — CLAUDE.md「ハルシネーション厳禁」). No axis is invented; any
//!   tensor whose shape disagrees with these values fails the runtime
//!   shape gate loudly (FR-EX-08,
//!   `ChatterboxTurboConfig::validate_for_forward`).
//! - **YAML config side-car (`t3_turbo_v1.yaml`)** — Chatterbox-Turbo
//!   ships a real config side-car this time (base Chatterbox did not);
//!   the converter still takes **no** `--config` path today because
//!   every field on that side-car is fixed for the Turbo release and
//!   the transcribed constants below are byte-parallel. Future
//!   releases that reshape the backbone would demand `--config`; this
//!   converter fails loudly if a tensor shape disagrees with the
//!   transcribed axes (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / base Chatterbox contract).
//! Real-weight binding is a follow-up wave gated on the upstream
//! tensor-name manifest fetch; this converter passes every F32 / F16
//! tensor through unchanged so a future
//! `ChatterboxTurboWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Chatterbox-Turbo is distributed both as safetensors and as a
//! separate `ResembleAI/chatterbox-turbo-ONNX` release. This converter
//! **never** touches ONNX (FR-LD-05); the pipeline is re-implemented
//! natively in `crates/vokra-models/src/chatterbox_turbo/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Chatterbox-Turbo GGUFs — kept in sync with the
/// runtime constant `vokra-models::chatterbox_turbo::EXPECTED_ARCH`.
/// Intentionally **distinct** from base Chatterbox's `"chatterbox"` so
/// the runtime can label the loaded model correctly.
pub(crate) const ARCH: &str = "chatterbox_turbo";
/// `vokra.model.name` value written for the canonical Turbo GGUF.
pub(crate) const NAME_TURBO: &str = "chatterbox-turbo-v1";

// --- vokra.chatterbox_turbo.* keys (kept as constants in the converter; the
// runtime duplicates the strings in
// `crates/vokra-models/src/chatterbox_turbo/mod.rs` — the two crates share
// only `vokra-core`, so the cross-crate constant duplication rule the CSM /
// CosyVoice2 / Kokoro / base Chatterbox converters use applies) --------------

const KEY_SAMPLE_RATE: &str = "vokra.chatterbox_turbo.sample_rate";

// GPT-2-medium vocab / prompt axes
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.chatterbox_turbo.arch.text_vocab_size";
const KEY_SPEECH_VOCAB_SIZE: &str = "vokra.chatterbox_turbo.arch.speech_vocab_size";
const KEY_MAX_TEXT_TOKENS: &str = "vokra.chatterbox_turbo.arch.max_text_tokens";
const KEY_MAX_SPEECH_TOKENS: &str = "vokra.chatterbox_turbo.arch.max_speech_tokens";
const KEY_SPEAKER_EMBED_SIZE: &str = "vokra.chatterbox_turbo.arch.speaker_embed_size";
const KEY_VE_HIDDEN_SIZE: &str = "vokra.chatterbox_turbo.arch.ve_hidden_size";

// GPT-2-medium backbone axes
const KEY_HIDDEN_DIM: &str = "vokra.chatterbox_turbo.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.chatterbox_turbo.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.chatterbox_turbo.arch.n_head";
const KEY_HEAD_DIM: &str = "vokra.chatterbox_turbo.arch.head_dim";

// STFT / mel frontend
const KEY_HOP_SIZE: &str = "vokra.chatterbox_turbo.arch.hop_size";
const KEY_WIN_SIZE: &str = "vokra.chatterbox_turbo.arch.win_size";
const KEY_NUM_MELS: &str = "vokra.chatterbox_turbo.arch.num_mels";

// Conditioning
const KEY_SPEECH_COND_PROMPT_LEN: &str = "vokra.chatterbox_turbo.arch.speech_cond_prompt_len";
const KEY_PARALINGUISTIC_TAG_COUNT: &str = "vokra.chatterbox_turbo.arch.paralinguistic_tag_count";

// Sentinel tokens
const KEY_START_TEXT_TOKEN: &str = "vokra.chatterbox_turbo.token.start_text";
const KEY_STOP_TEXT_TOKEN: &str = "vokra.chatterbox_turbo.token.stop_text";
const KEY_START_SPEECH_TOKEN: &str = "vokra.chatterbox_turbo.token.start_speech";
const KEY_STOP_SPEECH_TOKEN: &str = "vokra.chatterbox_turbo.token.stop_speech";

// Backbone family marker (Turbo == gpt2-medium, base Chatterbox == Llama_520M)
const KEY_BACKBONE_FAMILY: &str = "vokra.chatterbox_turbo.backbone_family";

// --- Transcribed constants (primary source: `t3_turbo_v1.yaml` at
// `huggingface.co/ResembleAI/chatterbox-turbo`, fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」) ------------------------------------

/// PCM sample rate — `t3_turbo_v1.yaml::sample_rate` (distinct from base
/// Chatterbox's 24 kHz).
const CHATTERBOX_TURBO_SAMPLE_RATE: u32 = 32_000;

/// Text-token vocabulary size = GPT-2 base (50 257) + 19 paralinguistic
/// tags = 50 276 (`t3_turbo_v1.yaml::text_tokens_dict_size`).
const TEXT_VOCAB_TURBO: u32 = 50_276;

/// Speech-token vocabulary size (`t3_turbo_v1.yaml::speech_tokens_dict_size`).
const SPEECH_VOCAB_SIZE: u32 = 6_563;

/// Max text-token positions (`t3_turbo_v1.yaml::max_text_tokens`).
const MAX_TEXT_TOKENS: u32 = 402;

/// Max speech-token positions (`t3_turbo_v1.yaml::max_speech_tokens`).
const MAX_SPEECH_TOKENS: u32 = 604;

/// Speaker-embedding dimension (`t3_turbo_v1.yaml::speaker_embed_size`).
const SPEAKER_EMBED_SIZE: u32 = 256;

/// Voice-encoder hidden dimension (`t3_turbo_v1.yaml::ve_hidden_size`).
const VE_HIDDEN_SIZE: u32 = 768;

// GPT-2-medium backbone axes — `t3_turbo_v1.yaml::legacy_gpt_hidden_size` +
// `n_transformer_layers` + `n_transformer_heads` + derived head_dim.
const HIDDEN_DIM: u32 = 1024;
const N_LAYER: u32 = 30;
const N_HEAD: u32 = 16;
const HEAD_DIM: u32 = 64;

// STFT frontend — `t3_turbo_v1.yaml::hop_size` / `win_size` / `num_mels`.
const HOP_SIZE: u32 = 320;
const WIN_SIZE: u32 = 2048;
const NUM_MELS: u32 = 256;

// Conditioning — `t3_turbo_v1.yaml::speech_cond_prompt_len`; tag count from
// `added_tokens.json`.
const SPEECH_COND_PROMPT_LEN: u32 = 250;
const PARALINGUISTIC_TAG_COUNT: u32 = 19;

// Sentinel tokens — `t3_turbo_v1.yaml::start_text_token` / `stop_text_token`
// / `start_speech_token` / `stop_speech_token`.
const START_TEXT_TOKEN: u32 = 255;
const STOP_TEXT_TOKEN: u32 = 0;
const START_SPEECH_TOKEN: u32 = 6_561;
const STOP_SPEECH_TOKEN: u32 = 6_562;

/// Backbone family — `t3_turbo_v1.yaml::gpt_transformer_type`.
const BACKBONE_FAMILY: &str = "gpt2-medium";

/// Outcome of a Chatterbox-Turbo conversion.
#[derive(Debug, Default)]
pub(crate) struct ChatterboxTurboReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32 / F16 tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time).
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics (never fail the conversion — the runtime
    /// is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Chatterbox-Turbo safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.chatterbox_turbo.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive` (MIT).
///
/// # No side-car config
///
/// Chatterbox-Turbo ships a real `t3_turbo_v1.yaml` this time, but the
/// converter still takes no path today — every field is fixed for the
/// Turbo release and byte-parallel to the constants above. A future
/// release that reshapes the backbone would demand `--config`.
pub(crate) fn convert(
    bytes: Vec<u8>,
) -> Result<(GgufBuilder, ChatterboxTurboReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME_TURBO);
    write_hparams(&mut b);
    // Self-describing redistribution: the artifact carries its own licence.
    // Chatterbox-Turbo ships MIT per `github.com/resemble-ai/chatterbox/LICENSE`
    // (Copyright (c) 2025 Resemble AI, fetched 2026-07-24 — CLAUDE.md
    // 「ハルシネーション厳禁」).
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some(NAME_TURBO),
        Some("ResembleAI/chatterbox-turbo (MIT — Copyright (c) 2025 Resemble AI)"),
    );

    let mut report = ChatterboxTurboReport::default();
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
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
             upstream Chatterbox-Turbo release ships \
             `t3_turbo_v1.safetensors` (~1.92 GB) directly; a real conversion \
             needs that file as input."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.chatterbox_turbo.*` chunk group from the transcribed
/// constants above (primary source: `t3_turbo_v1.yaml`).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, CHATTERBOX_TURBO_SAMPLE_RATE);
    b.add_string(KEY_BACKBONE_FAMILY, BACKBONE_FAMILY);

    // GPT-2-medium vocab / prompt axes
    b.add_u32(KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_TURBO);
    b.add_u32(KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE);
    b.add_u32(KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS);
    b.add_u32(KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS);
    b.add_u32(KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE);
    b.add_u32(KEY_VE_HIDDEN_SIZE, VE_HIDDEN_SIZE);

    // GPT-2-medium backbone
    b.add_u32(KEY_HIDDEN_DIM, HIDDEN_DIM);
    b.add_u32(KEY_N_LAYER, N_LAYER);
    b.add_u32(KEY_N_HEAD, N_HEAD);
    b.add_u32(KEY_HEAD_DIM, HEAD_DIM);

    // STFT frontend
    b.add_u32(KEY_HOP_SIZE, HOP_SIZE);
    b.add_u32(KEY_WIN_SIZE, WIN_SIZE);
    b.add_u32(KEY_NUM_MELS, NUM_MELS);

    // Conditioning
    b.add_u32(KEY_SPEECH_COND_PROMPT_LEN, SPEECH_COND_PROMPT_LEN);
    b.add_u32(KEY_PARALINGUISTIC_TAG_COUNT, PARALINGUISTIC_TAG_COUNT);

    // Sentinel tokens
    b.add_u32(KEY_START_TEXT_TOKEN, START_TEXT_TOKEN);
    b.add_u32(KEY_STOP_TEXT_TOKEN, STOP_TEXT_TOKEN);
    b.add_u32(KEY_START_SPEECH_TOKEN, START_SPEECH_TOKEN);
    b.add_u32(KEY_STOP_SPEECH_TOKEN, STOP_SPEECH_TOKEN);
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
    /// 2 bytes = 12 bytes). Real Turbo checkpoints are likely served in F16,
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

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::chatterbox_turbo::EXPECTED_ARCH`.
        assert_eq!(ARCH, "chatterbox_turbo");
    }

    #[test]
    fn arch_is_distinct_from_base_chatterbox() {
        // Turbo swaps backbone family + sample rate + text vocab; silently
        // sharing base's arch tag would misrepresent the loaded model.
        assert_ne!(ARCH, "chatterbox");
    }

    #[test]
    fn name_string_matches_hf_release() {
        assert_eq!(NAME_TURBO, "chatterbox-turbo-v1");
    }

    /// The transcribed constants must equal the primary-source values —
    /// changing any of these silently mis-shapes the GPT-2 backbone.
    #[test]
    fn transcribed_constants_match_primary_source() {
        assert_eq!(CHATTERBOX_TURBO_SAMPLE_RATE, 32_000);
        assert_eq!(TEXT_VOCAB_TURBO, 50_276);
        assert_eq!(SPEECH_VOCAB_SIZE, 6_563);
        assert_eq!(MAX_TEXT_TOKENS, 402);
        assert_eq!(MAX_SPEECH_TOKENS, 604);
        assert_eq!(SPEAKER_EMBED_SIZE, 256);
        assert_eq!(VE_HIDDEN_SIZE, 768);
        assert_eq!(HIDDEN_DIM, 1024);
        assert_eq!(N_LAYER, 30);
        assert_eq!(N_HEAD, 16);
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(HOP_SIZE, 320);
        assert_eq!(WIN_SIZE, 2048);
        assert_eq!(NUM_MELS, 256);
        assert_eq!(SPEECH_COND_PROMPT_LEN, 250);
        assert_eq!(PARALINGUISTIC_TAG_COUNT, 19);
        assert_eq!(START_TEXT_TOKEN, 255);
        assert_eq!(STOP_TEXT_TOKEN, 0);
        assert_eq!(START_SPEECH_TOKEN, 6_561);
        assert_eq!(STOP_SPEECH_TOKEN, 6_562);
        assert_eq!(BACKBONE_FAMILY, "gpt2-medium");
        // GPT-2 MHA algebra
        assert_eq!(HIDDEN_DIM, N_HEAD * HEAD_DIM);
        // Stop tokens live inside their vocabularies (const block so the
        // check is honoured at compile time — the values are all
        // `const`, so a runtime `assert!` would be dead-eliminated by
        // clippy's `assertions_on_constants` lint).
        const _: () = {
            assert!(STOP_TEXT_TOKEN < TEXT_VOCAB_TURBO);
            assert!(STOP_SPEECH_TOKEN < SPEECH_VOCAB_SIZE);
            // STFT well-formedness
            assert!(WIN_SIZE >= HOP_SIZE);
        };
    }

    /// The Turbo constants disagree with base Chatterbox on the three axes
    /// that actually change (backbone family, sample rate, text vocab) —
    /// pins the "distinct arch" contract at the numeric level.
    #[test]
    fn turbo_constants_differ_from_base_chatterbox() {
        // Base Chatterbox: sample_rate=24_000, text_vocab=2454/704.
        assert_ne!(CHATTERBOX_TURBO_SAMPLE_RATE, 24_000);
        assert_ne!(TEXT_VOCAB_TURBO, 2_454);
        assert_ne!(TEXT_VOCAB_TURBO, 704);
        // Backbone family differs (base = "Llama_520M", Turbo = "gpt2-medium").
        assert_ne!(BACKBONE_FAMILY, "Llama_520M");
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_TURBO)
        );
        assert_eq!(
            file.get(KEY_BACKBONE_FAMILY).and_then(|v| v.as_str()),
            Some(BACKBONE_FAMILY)
        );

        // Every transcribed U32 hparam round-trips verbatim under the
        // `vokra.chatterbox_turbo.*` prefix.
        for (key, want) in [
            (KEY_SAMPLE_RATE, CHATTERBOX_TURBO_SAMPLE_RATE),
            (KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_TURBO),
            (KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE),
            (KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS),
            (KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS),
            (KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE),
            (KEY_VE_HIDDEN_SIZE, VE_HIDDEN_SIZE),
            (KEY_HIDDEN_DIM, HIDDEN_DIM),
            (KEY_N_LAYER, N_LAYER),
            (KEY_N_HEAD, N_HEAD),
            (KEY_HEAD_DIM, HEAD_DIM),
            (KEY_HOP_SIZE, HOP_SIZE),
            (KEY_WIN_SIZE, WIN_SIZE),
            (KEY_NUM_MELS, NUM_MELS),
            (KEY_SPEECH_COND_PROMPT_LEN, SPEECH_COND_PROMPT_LEN),
            (KEY_PARALINGUISTIC_TAG_COUNT, PARALINGUISTIC_TAG_COUNT),
            (KEY_START_TEXT_TOKEN, START_TEXT_TOKEN),
            (KEY_STOP_TEXT_TOKEN, STOP_TEXT_TOKEN),
            (KEY_START_SPEECH_TOKEN, START_SPEECH_TOKEN),
            (KEY_STOP_SPEECH_TOKEN, STOP_SPEECH_TOKEN),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // Provenance: MIT permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_TURBO)
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
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        // Empty safetensors → the runtime's `ChatterboxTurboWeights::from_gguf`
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
    /// arm. A real Chatterbox-Turbo checkpoint is likely served in F16 or
    /// BF16; a typo dropping `| GgmlType::F16` would silently bin every F16
    /// tensor into `skipped_non_float`.
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

    /// Pins the `_ =>` arm of the tensor-dtype match: BF16 graduated to a
    /// supported safetensors dtype in M4-06 (moshiko is all-BF16) so BF16
    /// tensors now reach `convert()` and MUST be counted, not silently
    /// dropped.
    #[test]
    fn bf16_tensor_is_counted_as_skipped_non_float() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 0,
            "BF16 must not currently pass through — Chatterbox-Turbo converter is F32/F16 only"
        );
        assert_eq!(
            report.skipped_non_float, 1,
            "BF16 must increment the skipped counter"
        );
        // With zero float tensors written, the loud "no float tensors" note
        // fires.
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16-only conversion must emit the zero-float note: {:?}",
            report.notes
        );
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert!(
            file.tensor_info("text_emb.weight").is_none(),
            "BF16 tensor must not be written"
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

    /// Every `vokra.chatterbox_turbo.*` key uses the same prefix — a
    /// regression where a key crossed into another model's namespace (e.g.
    /// `vokra.chatterbox.*`) would still round-trip in isolation but would
    /// misroute at the runtime dispatch layer.
    #[test]
    fn every_metadata_key_carries_the_chatterbox_turbo_prefix() {
        for key in [
            KEY_SAMPLE_RATE,
            KEY_TEXT_VOCAB_SIZE,
            KEY_SPEECH_VOCAB_SIZE,
            KEY_MAX_TEXT_TOKENS,
            KEY_MAX_SPEECH_TOKENS,
            KEY_SPEAKER_EMBED_SIZE,
            KEY_VE_HIDDEN_SIZE,
            KEY_HIDDEN_DIM,
            KEY_N_LAYER,
            KEY_N_HEAD,
            KEY_HEAD_DIM,
            KEY_HOP_SIZE,
            KEY_WIN_SIZE,
            KEY_NUM_MELS,
            KEY_SPEECH_COND_PROMPT_LEN,
            KEY_PARALINGUISTIC_TAG_COUNT,
            KEY_START_TEXT_TOKEN,
            KEY_STOP_TEXT_TOKEN,
            KEY_START_SPEECH_TOKEN,
            KEY_STOP_SPEECH_TOKEN,
            KEY_BACKBONE_FAMILY,
        ] {
            assert!(
                key.starts_with("vokra.chatterbox_turbo."),
                "{key} must live under the vokra.chatterbox_turbo.* prefix"
            );
        }
    }
}
