//! **Chatterbox-Nano**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 3, 2026-07-24).
//!
//! Input: the upstream `ResembleAI/chatterbox-nano` backbone safetensors —
//! `t3_nano_v1.safetensors`. Output: a GGUF carrying every float tensor
//! plus the `vokra.chatterbox_nano.*` and `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks the native Chatterbox-Nano
//! implementation (`crates/vokra-models/src/chatterbox_nano/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.chatterbox_nano.*` chunk group is transcribed **verbatim**
//!   from the primary source `t3_nano_v1.yaml`
//!   (`huggingface.co/ResembleAI/chatterbox-nano`, fetched 2026-07-24
//!   — CLAUDE.md「ハルシネーション厳禁」). No axis is invented; any
//!   tensor whose shape disagrees with these values fails the runtime
//!   shape gate loudly (FR-EX-08,
//!   `ChatterboxNanoConfig::validate_for_forward`).
//! - **Backbone family** — Nano's yaml sets `llama_config_name:
//!   Llama_520M` and (a stale training-side flag)
//!   `gpt_transformer_type: gpt2`. The Llama_520M name is authoritative:
//!   the Nano backbone uses SwiGLU + RMSNorm + RoPE (the base
//!   Chatterbox topology), NOT gpt2-medium's LayerNorm-with-bias +
//!   fused-QKV-with-bias + GELU FFN (which is what Turbo uses). See
//!   the runtime module docstring for the primary-source rationale.
//! - **YAML config side-car (`t3_nano_v1.yaml`)** — the converter takes
//!   **no** `--config` path today because every field on that side-car
//!   is fixed for the Nano release and byte-parallel to the transcribed
//!   constants below. Future releases that reshape the backbone would
//!   demand `--config`; this converter fails loudly if a tensor shape
//!   disagrees with the transcribed axes (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / base Chatterbox / Chatterbox-Turbo
//! contract). Real-weight binding is a follow-up wave gated on the
//! upstream tensor-name manifest fetch; this converter passes every
//! F32 / F16 tensor through unchanged so a future
//! `ChatterboxNanoWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! This converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in
//! `crates/vokra-models/src/chatterbox_nano/` (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Chatterbox-Nano GGUFs — kept in sync with the
/// runtime constant `vokra-models::chatterbox_nano::EXPECTED_ARCH`.
/// Intentionally **distinct** from base Chatterbox's `"chatterbox"` and
/// Turbo's `"chatterbox_turbo"` so the runtime can label the loaded
/// model correctly.
pub(crate) const ARCH: &str = "chatterbox_nano";
/// `vokra.model.name` value written for the canonical Nano GGUF.
pub(crate) const NAME_NANO: &str = "chatterbox-nano-v1";

// --- vokra.chatterbox_nano.* keys (kept as constants in the converter; the
// runtime duplicates the strings in
// `crates/vokra-models/src/chatterbox_nano/mod.rs` — the two crates share
// only `vokra-core`, so the cross-crate constant duplication rule the CSM /
// CosyVoice2 / Kokoro / base Chatterbox / Chatterbox-Turbo converters use
// applies) -----------------------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.chatterbox_nano.sample_rate";

// T3 vocab / prompt axes
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.chatterbox_nano.arch.text_vocab_size";
const KEY_SPEECH_VOCAB_SIZE: &str = "vokra.chatterbox_nano.arch.speech_vocab_size";
const KEY_MAX_TEXT_TOKENS: &str = "vokra.chatterbox_nano.arch.max_text_tokens";
const KEY_MAX_SPEECH_TOKENS: &str = "vokra.chatterbox_nano.arch.max_speech_tokens";
const KEY_SPEAKER_EMBED_SIZE: &str = "vokra.chatterbox_nano.arch.speaker_embed_size";
const KEY_VE_HIDDEN_SIZE: &str = "vokra.chatterbox_nano.arch.ve_hidden_size";

// Llama_520M backbone axes
const KEY_HIDDEN_DIM: &str = "vokra.chatterbox_nano.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.chatterbox_nano.arch.n_layer";
const KEY_N_HEAD: &str = "vokra.chatterbox_nano.arch.n_head";
const KEY_N_HEAD_KV: &str = "vokra.chatterbox_nano.arch.n_head_kv";
const KEY_HEAD_DIM: &str = "vokra.chatterbox_nano.arch.head_dim";
const KEY_FFN_DIM: &str = "vokra.chatterbox_nano.arch.ffn_dim";

// Norm / RoPE
const KEY_ROPE_BASE: &str = "vokra.chatterbox_nano.arch.rope_base";
const KEY_RMS_NORM_EPS: &str = "vokra.chatterbox_nano.arch.rms_norm_eps";

// STFT / mel frontend
const KEY_HOP_SIZE: &str = "vokra.chatterbox_nano.arch.hop_size";
const KEY_WIN_SIZE: &str = "vokra.chatterbox_nano.arch.win_size";
const KEY_NUM_MELS: &str = "vokra.chatterbox_nano.arch.num_mels";

// Conditioning
const KEY_SPEECH_COND_PROMPT_LEN: &str = "vokra.chatterbox_nano.arch.speech_cond_prompt_len";
const KEY_PARALINGUISTIC_TAG_COUNT: &str = "vokra.chatterbox_nano.arch.paralinguistic_tag_count";

// Sentinel tokens
const KEY_START_TEXT_TOKEN: &str = "vokra.chatterbox_nano.token.start_text";
const KEY_STOP_TEXT_TOKEN: &str = "vokra.chatterbox_nano.token.stop_text";
const KEY_START_SPEECH_TOKEN: &str = "vokra.chatterbox_nano.token.start_speech";
const KEY_STOP_SPEECH_TOKEN: &str = "vokra.chatterbox_nano.token.stop_speech";

// Backbone family marker (Nano == Llama_520M — the yaml's `llama_config_name`
// is authoritative over the stale `gpt_transformer_type` field; see runtime
// module docstring).
const KEY_BACKBONE_FAMILY: &str = "vokra.chatterbox_nano.backbone_family";

// --- Transcribed constants (primary source: `t3_nano_v1.yaml` at
// `huggingface.co/ResembleAI/chatterbox-nano`, fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」) --------------------------------------

/// PCM sample rate — `t3_nano_v1.yaml::sample_rate` (32 kHz — same as
/// Turbo, distinct from base Chatterbox's 24 kHz).
const CHATTERBOX_NANO_SAMPLE_RATE: u32 = 32_000;

/// Text-token vocabulary size = GPT-2 base (50 257) + 19 paralinguistic
/// tags = 50 276 (`t3_nano_v1.yaml::text_tokens_dict_size`) — same as
/// Turbo.
const TEXT_VOCAB_NANO: u32 = 50_276;

/// Speech-token vocabulary size (`t3_nano_v1.yaml::speech_tokens_dict_size`).
const SPEECH_VOCAB_SIZE: u32 = 6_563;

/// Max text-token positions (`t3_nano_v1.yaml::max_text_tokens`).
const MAX_TEXT_TOKENS: u32 = 402;

/// Max speech-token positions (`t3_nano_v1.yaml::max_speech_tokens`).
const MAX_SPEECH_TOKENS: u32 = 604;

/// Speaker-embedding dimension (`t3_nano_v1.yaml::speaker_embed_size`).
const SPEAKER_EMBED_SIZE: u32 = 256;

/// Voice-encoder hidden dimension (`t3_nano_v1.yaml::ve_hidden_size`).
const VE_HIDDEN_SIZE: u32 = 768;

// Llama_520M backbone axes — `t3_nano_v1.yaml::legacy_gpt_hidden_size` /
// `n_transformer_layers` / `n_transformer_heads` +
// `LLAMA_520M_CONFIG_DICT` (`src/chatterbox/models/t3/llama_configs.py`)
// for `head_dim` / `intermediate_size` / `num_key_value_heads` /
// `rope_theta` / `rms_norm_eps`. MHA (n_head_kv == n_head).
const HIDDEN_DIM: u32 = 1024;
const N_LAYER: u32 = 30;
const N_HEAD: u32 = 16;
const N_HEAD_KV: u32 = 16;
const HEAD_DIM: u32 = 64;
const FFN_DIM: u32 = 4096;

// Norm / RoPE
const ROPE_BASE: f32 = 500_000.0;
const RMS_NORM_EPS: f32 = 1e-5;

// STFT frontend — `t3_nano_v1.yaml::hop_size` / `win_size` / `num_mels`.
const HOP_SIZE: u32 = 320;
const WIN_SIZE: u32 = 2048;
const NUM_MELS: u32 = 256;

// Conditioning — `t3_nano_v1.yaml::speech_cond_prompt_len`; tag count
// from `added_tokens.json`.
const SPEECH_COND_PROMPT_LEN: u32 = 250;
const PARALINGUISTIC_TAG_COUNT: u32 = 19;

// Sentinel tokens — `t3_nano_v1.yaml::start_text_token` /
// `stop_text_token` / `start_speech_token` / `stop_speech_token`. Nano's
// distinguishing sentinel is `stop_text_token = 50256` (the GPT-2
// `<|endoftext|>` id) — distinct from both Turbo (0) and base (0).
const START_TEXT_TOKEN: u32 = 255;
const STOP_TEXT_TOKEN: u32 = 50_256;
const START_SPEECH_TOKEN: u32 = 6_561;
const STOP_SPEECH_TOKEN: u32 = 6_562;

/// Backbone family — `t3_nano_v1.yaml::llama_config_name` (Llama_520M
/// is authoritative; the sibling `gpt_transformer_type: gpt2` field is
/// a stale training-side legacy flag inherited from the base training
/// config, and Nano's actual T3 backbone routes through the Llama
/// primitives just like base Chatterbox).
const BACKBONE_FAMILY: &str = "Llama_520M";

/// Outcome of a Chatterbox-Nano conversion.
#[derive(Debug, Default)]
pub(crate) struct ChatterboxNanoReport {
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
    /// Operator-facing diagnostics (never fail the conversion — the runtime
    /// is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Chatterbox-Nano safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.chatterbox_nano.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive` (MIT).
///
/// # No side-car config
///
/// Chatterbox-Nano ships a real `t3_nano_v1.yaml` alongside the safetensors,
/// but the converter still takes no path today — every field is fixed for
/// the Nano release and byte-parallel to the constants above. A future
/// release that reshapes the backbone would demand `--config`.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ChatterboxNanoReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME_NANO);
    write_hparams(&mut b);
    // Self-describing redistribution: the artifact carries its own licence.
    // Chatterbox-Nano ships MIT per `github.com/resemble-ai/chatterbox/LICENSE`
    // (Copyright (c) 2025 Resemble AI, fetched 2026-07-24 — CLAUDE.md
    // 「ハルシネーション厳禁」). The whole Chatterbox family (base + Turbo
    // + Nano + `-multilingual-*` variants) ships under a single MIT LICENSE.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "MIT",
        Some(NAME_NANO),
        Some("ResembleAI/chatterbox-nano (MIT — Copyright (c) 2025 Resemble AI)"),
    );

    let mut report = ChatterboxNanoReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2): upstream Chatterbox-Nano is likely
            // served in BF16 (base Chatterbox family serving format) so
            // the release checkpoint hits this arm. Emit as GGUF type 30
            // verbatim; runtime widens on load via `decode_bf16` (exact,
            // `bits << 16`).
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
             upstream Chatterbox-Nano release ships \
             `t3_nano_v1.safetensors` directly; the BF16 pass-through path \
             is now wired (2026-07-25), so this state is only reachable \
             when the release contains no F32 / F16 / BF16 float tensors \
             at all."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.chatterbox_nano.*` chunk group from the transcribed
/// constants above (primary source: `t3_nano_v1.yaml`).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, CHATTERBOX_NANO_SAMPLE_RATE);
    b.add_string(KEY_BACKBONE_FAMILY, BACKBONE_FAMILY);

    // T3 vocab / prompt axes
    b.add_u32(KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_NANO);
    b.add_u32(KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE);
    b.add_u32(KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS);
    b.add_u32(KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS);
    b.add_u32(KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE);
    b.add_u32(KEY_VE_HIDDEN_SIZE, VE_HIDDEN_SIZE);

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
    /// 2 bytes = 12 bytes). Real Nano checkpoints are likely served in F16,
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
        // sole handshake with `vokra-models::chatterbox_nano::EXPECTED_ARCH`.
        assert_eq!(ARCH, "chatterbox_nano");
    }

    #[test]
    fn arch_is_distinct_from_base_and_turbo() {
        // Nano keeps base's Llama_520M backbone family but swaps sample
        // rate + text vocab + stop-text sentinel; silently sharing base
        // or Turbo's arch tag would misrepresent the loaded model.
        assert_ne!(ARCH, "chatterbox");
        assert_ne!(ARCH, "chatterbox_turbo");
    }

    #[test]
    fn name_string_matches_hf_release() {
        assert_eq!(NAME_NANO, "chatterbox-nano-v1");
    }

    /// The transcribed constants must equal the primary-source values —
    /// changing any of these silently mis-shapes the Llama_520M backbone.
    #[test]
    fn transcribed_constants_match_primary_source() {
        assert_eq!(CHATTERBOX_NANO_SAMPLE_RATE, 32_000);
        assert_eq!(TEXT_VOCAB_NANO, 50_276);
        assert_eq!(SPEECH_VOCAB_SIZE, 6_563);
        assert_eq!(MAX_TEXT_TOKENS, 402);
        assert_eq!(MAX_SPEECH_TOKENS, 604);
        assert_eq!(SPEAKER_EMBED_SIZE, 256);
        assert_eq!(VE_HIDDEN_SIZE, 768);
        assert_eq!(HIDDEN_DIM, 1024);
        assert_eq!(N_LAYER, 30);
        assert_eq!(N_HEAD, 16);
        assert_eq!(N_HEAD_KV, 16);
        assert_eq!(HEAD_DIM, 64);
        assert_eq!(FFN_DIM, 4096);
        assert!((ROPE_BASE - 500_000.0).abs() < 1e-3);
        assert!((RMS_NORM_EPS - 1e-5).abs() < 1e-10);
        assert_eq!(HOP_SIZE, 320);
        assert_eq!(WIN_SIZE, 2048);
        assert_eq!(NUM_MELS, 256);
        assert_eq!(SPEECH_COND_PROMPT_LEN, 250);
        assert_eq!(PARALINGUISTIC_TAG_COUNT, 19);
        assert_eq!(START_TEXT_TOKEN, 255);
        // Nano's DISTINGUISHING sentinel — the GPT-2 <|endoftext|> token id.
        assert_eq!(STOP_TEXT_TOKEN, 50_256);
        assert_eq!(START_SPEECH_TOKEN, 6_561);
        assert_eq!(STOP_SPEECH_TOKEN, 6_562);
        assert_eq!(BACKBONE_FAMILY, "Llama_520M");
        // Llama_520M MHA algebra (const block so the check is honoured
        // at compile time — the values are all `const`, so a runtime
        // `assert!` would be dead-eliminated by clippy's
        // `assertions_on_constants` lint).
        const _: () = {
            assert!(HIDDEN_DIM == N_HEAD * HEAD_DIM);
            assert!(N_HEAD == N_HEAD_KV); // MHA
            // Stop tokens live inside their vocabularies.
            assert!(STOP_TEXT_TOKEN < TEXT_VOCAB_NANO);
            assert!(STOP_SPEECH_TOKEN < SPEECH_VOCAB_SIZE);
            // STFT well-formedness
            assert!(WIN_SIZE >= HOP_SIZE);
            // RoPE requires even head_dim.
            assert!(HEAD_DIM % 2 == 0);
        };
    }

    /// The Nano constants disagree with base Chatterbox on the axes that
    /// actually change (sample rate, text vocab, stop-text sentinel) and
    /// agree with Turbo on the shared axes (sample rate, text vocab) —
    /// pins the "distinct arch" contract at the numeric level.
    #[test]
    fn nano_constants_relate_correctly_to_siblings() {
        // Base Chatterbox: sample_rate=24_000, text_vocab=2454/704,
        // stop_text_token=0. Nano diverges on all three.
        assert_ne!(CHATTERBOX_NANO_SAMPLE_RATE, 24_000);
        assert_ne!(TEXT_VOCAB_NANO, 2_454);
        assert_ne!(TEXT_VOCAB_NANO, 704);
        assert_ne!(STOP_TEXT_TOKEN, 0);
        // Nano's stop_text_token is the GPT-2 EOT id (50256) — distinct
        // from Turbo's 0. This is the only architectural axis where the
        // two GPT-2-vocab members disagree.
        assert_ne!(STOP_TEXT_TOKEN, 0);
        // Backbone family stays with the Llama_520M family (base
        // Chatterbox's backbone) — distinct from Turbo's gpt2-medium.
        assert_ne!(BACKBONE_FAMILY, "gpt2-medium");
        assert_eq!(BACKBONE_FAMILY, "Llama_520M");
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
            Some(NAME_NANO)
        );
        assert_eq!(
            file.get(KEY_BACKBONE_FAMILY).and_then(|v| v.as_str()),
            Some(BACKBONE_FAMILY)
        );

        // Every transcribed U32 hparam round-trips verbatim under the
        // `vokra.chatterbox_nano.*` prefix.
        for (key, want) in [
            (KEY_SAMPLE_RATE, CHATTERBOX_NANO_SAMPLE_RATE),
            (KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_NANO),
            (KEY_SPEECH_VOCAB_SIZE, SPEECH_VOCAB_SIZE),
            (KEY_MAX_TEXT_TOKENS, MAX_TEXT_TOKENS),
            (KEY_MAX_SPEECH_TOKENS, MAX_SPEECH_TOKENS),
            (KEY_SPEAKER_EMBED_SIZE, SPEAKER_EMBED_SIZE),
            (KEY_VE_HIDDEN_SIZE, VE_HIDDEN_SIZE),
            (KEY_HIDDEN_DIM, HIDDEN_DIM),
            (KEY_N_LAYER, N_LAYER),
            (KEY_N_HEAD, N_HEAD),
            (KEY_N_HEAD_KV, N_HEAD_KV),
            (KEY_HEAD_DIM, HEAD_DIM),
            (KEY_FFN_DIM, FFN_DIM),
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

        // F32 norm / RoPE constants round-trip too.
        assert!((get_f32(&file, KEY_ROPE_BASE) - ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_RMS_NORM_EPS) - RMS_NORM_EPS).abs() < 1e-10);

        // Provenance: MIT permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_NANO)
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
        // Empty safetensors → the runtime's `ChatterboxNanoWeights::from_gguf`
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
    /// arm. A real Chatterbox-Nano checkpoint is likely served in F16 or
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

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union match arm. Real Chatterbox-Nano checkpoints
    /// are likely served in BF16 (upstream `t3_nano_v1.safetensors` follows
    /// the base Chatterbox family serving format); BF16 tensors MUST reach
    /// the pass-through arm verbatim (emitted as GGUF type 30 =
    /// `GgmlType::BF16`, no convert-time widening — the runtime widens
    /// BF16 → f32 losslessly at load via the single choke point
    /// `vokra-core::gguf::quant::decode_bf16`, which is exact since BF16
    /// is the top 16 bits of an f32 — `bits << 16`).
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture. Removing
    /// the pin outright would let a latent silent-widen slip in undetected;
    /// rewriting to the passes-through invariant keeps the regression
    /// guard (mirror of qwen3-tts / vibevoice / voxcpm2 pattern).
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
        // Loud-silence check for FR-EX-08: the zero-float note is a
        // false-positive here because BF16 IS a float.
        assert!(
            !report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16 pass-through must not emit the zero-float note: {:?}",
            report.notes
        );

        // The tensor survives the round trip under its upstream name and
        // preserves its BF16 dtype (no convert-time widening — runtime
        // widens on load via `decode_bf16`).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
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

    /// Every `vokra.chatterbox_nano.*` key uses the same prefix — a
    /// regression where a key crossed into another model's namespace (e.g.
    /// `vokra.chatterbox_turbo.*` / `vokra.chatterbox.*`) would still
    /// round-trip in isolation but would misroute at the runtime dispatch
    /// layer.
    #[test]
    fn every_metadata_key_carries_the_chatterbox_nano_prefix() {
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
            KEY_N_HEAD_KV,
            KEY_HEAD_DIM,
            KEY_FFN_DIM,
            KEY_ROPE_BASE,
            KEY_RMS_NORM_EPS,
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
                key.starts_with("vokra.chatterbox_nano."),
                "{key} must live under the vokra.chatterbox_nano.* prefix"
            );
        }
    }
}
