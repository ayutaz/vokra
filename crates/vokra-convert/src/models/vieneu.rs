//! **VieNeu-TTS-v3-Turbo** (`pnnbao-ump/VieNeu-TTS-v3-Turbo`,
//! apache-2.0): safetensors → GGUF conversion (implementer C wave,
//! 2026-07-30).
//!
//! Input: the upstream `pnnbao-ump/VieNeu-TTS-v3-Turbo` release —
//! ships `model.safetensors` directly (no torch-pickle prepare step).
//! Output: a GGUF carrying every float tensor plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native VieNeu-TTS loader will read.
//!
//! # Architecture (primary source, 2026-07-30 CC fetch)
//!
//! VieNeu-TTS-v3-Turbo is a **novel hierarchical AR TTS** architecture
//! (`architectures = ["VieNeuV3TurboForTTS"]`, `model_type =
//! "vieneu_v3_turbo"`) — NOT a VITS / StyleTTS / Piper fork. The
//! primary source config `huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/
//! raw/main/config.json` (fetched 2026-07-30 — CLAUDE.md「ハルシネー
//! ション厳禁」) describes:
//!
//! **Backbone** (LLM-family text-conditioning + acoustic prefill):
//! - `hidden_size = 768`, `num_hidden_layers = 12`
//! - `num_attention_heads = 12`, `num_key_value_heads = 4` (GQA ratio 3)
//! - `head_dim = 64`, `intermediate_size = 3072`
//! - `rms_norm_eps = 1e-6`, `rope_theta = 10000.0`
//! - `max_position_embeddings = 1024`, `tie_word_embeddings = false`
//!
//! **Local decoder** (acoustic head — small transformer, 2 layers):
//! - `local_num_hidden_layers = 2`, `local_num_attention_heads = 8`
//! - `local_intermediate_size = 2048`, `head_dim = 64`
//! - The upstream config notes `_local_rope_theta_note` = "DEPRECATED /
//!   unused: acoustic decoder now uses a learned slot-position
//!   embedding, not RoPE" — so the local decoder is **positioned by a
//!   learned slot embedding**, distinct from the backbone's RoPE.
//!
//! **Audio codec** (external, referenced but not shipped):
//! - `audio_tokenizer_pretrained_name_or_path =
//!   "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"` (must be fetched
//!   separately by the runtime — mirror of Parler-TTS's external DAC
//!   dependency)
//! - `audio_sample_rate = 48000` (48 kHz output — a distinguishing
//!   sentinel; other Vokra TTS models are 24 kHz / 32 kHz / 44.1 kHz)
//! - `audio_vocab_size = 1024`, `n_vq = 16` (16 quantizers per frame)
//!
//! **Token layout** (reserved slots + emotion tokens):
//! - `text_vocab_size = 419`
//! - `num_reserved_tokens = 30` (ids 13..42, inserted after emotion_4)
//! - `emotion_{0..4}_token_id = 8..12` (5 emotion control tokens)
//! - `audio_ref_slot_token_id = 7`, `audio_pad_token_id = 1024`
//! - `text_prompt_{start,end}_token_id = 3, 4`
//! - `speech_generation_{start,end}_token_id = 5, 6`
//! - `bos = 1`, `eos = 2`, `pad = 0`, `unk = 43`
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 tensors pass through **verbatim**. BF16 stays GGUF
//! type 30 — no convert-time widening; runtime widens BF16 → f32
//! losslessly via `decode_bf16`. Note: this checkpoint's own
//! `config.json` declares `dtype = "float32"` so BF16 tensors are not
//! expected on the wire from the *upstream* release — but a downstream
//! caller who quantized offline can still pass BF16 through this
//! converter unchanged.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! Real-weight binding is a follow-up wave gated on the upstream
//! tensor-name manifest fetch.
//!
//! # Feature scope — safetensors only, no ONNX
//!
//! The upstream release ships **both** safetensors (`model.safetensors`
//! at 32-bit float32) and multiple ONNX subgraph packs
//! (`onnx/vieneu_{prefill,decode_step,acoustic_cached}.onnx` +
//! `denoiser.onnx` + `speaker_encoder.onnx`). This converter **only**
//! consumes the safetensors path (FR-LD-05); the ONNX subgraphs and the
//! external denoiser / speaker encoder are out of scope for this
//! module (the denoiser is separately handled by DeepFilterNet3, and
//! speaker encoders are `campplus` / `ecapa-tdnn` / `wespeaker`).
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream Python pipeline is deferred to
//! owner (`docs/license-audit.md` §3.1 sign-off).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub(crate) const ARCH: &str = "vieneu_v3_turbo";
pub(crate) const NAME: &str = "vieneu-tts-v3-turbo";
pub(crate) const CATEGORY: &str = "tts";
pub(crate) const UPSTREAM_HF: &str = "pnnbao-ump/VieNeu-TTS-v3-Turbo";
pub(crate) const DEFAULT_LICENSE: &str = "apache-2.0";

// ---- Hparams (transcribed verbatim from upstream config.json) -----------

// Backbone (LLM-family):
pub(crate) const HIDDEN_SIZE: u32 = 768;
pub(crate) const NUM_HIDDEN_LAYERS: u32 = 12;
pub(crate) const NUM_ATTENTION_HEADS: u32 = 12;
pub(crate) const NUM_KEY_VALUE_HEADS: u32 = 4;
pub(crate) const HEAD_DIM: u32 = 64;
pub(crate) const INTERMEDIATE_SIZE: u32 = 3_072;
pub(crate) const MAX_POSITION_EMBEDDINGS: u32 = 1_024;
pub(crate) const ROPE_THETA: f32 = 10_000.0;
pub(crate) const RMS_NORM_EPS: f32 = 1e-6;
pub(crate) const TIE_WORD_EMBEDDINGS: bool = false;

// Local decoder (acoustic head):
pub(crate) const LOCAL_NUM_HIDDEN_LAYERS: u32 = 2;
pub(crate) const LOCAL_NUM_ATTENTION_HEADS: u32 = 8;
pub(crate) const LOCAL_INTERMEDIATE_SIZE: u32 = 2_048;

// Audio codec + rate:
pub(crate) const AUDIO_SAMPLE_RATE: u32 = 48_000;
pub(crate) const AUDIO_VOCAB_SIZE: u32 = 1_024;
pub(crate) const N_VQ: u32 = 16;
pub(crate) const AUDIO_TOKENIZER: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano";

// Text vocab + reserved tokens:
pub(crate) const TEXT_VOCAB_SIZE: u32 = 419;
pub(crate) const NUM_RESERVED_TOKENS: u32 = 30;
pub(crate) const RESERVED_TOKEN_START: u32 = 13;

// Special token ids:
pub(crate) const BOS_TOKEN_ID: u32 = 1;
pub(crate) const EOS_TOKEN_ID: u32 = 2;
pub(crate) const PAD_TOKEN_ID: u32 = 0;
pub(crate) const UNK_TOKEN_ID: u32 = 43;
pub(crate) const TEXT_PROMPT_START_TOKEN_ID: u32 = 3;
pub(crate) const TEXT_PROMPT_END_TOKEN_ID: u32 = 4;
pub(crate) const SPEECH_GENERATION_START_TOKEN_ID: u32 = 5;
pub(crate) const SPEECH_GENERATION_END_TOKEN_ID: u32 = 6;
pub(crate) const AUDIO_REF_SLOT_TOKEN_ID: u32 = 7;
pub(crate) const AUDIO_PAD_TOKEN_ID: u32 = 1_024;
pub(crate) const EMOTION_0_TOKEN_ID: u32 = 8;
pub(crate) const EMOTION_1_TOKEN_ID: u32 = 9;
pub(crate) const EMOTION_2_TOKEN_ID: u32 = 10;
pub(crate) const EMOTION_3_TOKEN_ID: u32 = 11;
pub(crate) const EMOTION_4_TOKEN_ID: u32 = 12;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const KEY_HIDDEN_SIZE: &str = "vokra.vieneu.hidden_size";
const KEY_NUM_HIDDEN_LAYERS: &str = "vokra.vieneu.num_hidden_layers";
const KEY_NUM_ATTENTION_HEADS: &str = "vokra.vieneu.num_attention_heads";
const KEY_NUM_KEY_VALUE_HEADS: &str = "vokra.vieneu.num_key_value_heads";
const KEY_HEAD_DIM: &str = "vokra.vieneu.head_dim";
const KEY_INTERMEDIATE_SIZE: &str = "vokra.vieneu.intermediate_size";
const KEY_MAX_POSITION_EMBEDDINGS: &str = "vokra.vieneu.max_position_embeddings";
const KEY_ROPE_THETA: &str = "vokra.vieneu.rope_theta";
const KEY_RMS_NORM_EPS: &str = "vokra.vieneu.rms_norm_eps";
const KEY_TIE_WORD_EMBEDDINGS: &str = "vokra.vieneu.tie_word_embeddings";

const KEY_LOCAL_NUM_HIDDEN_LAYERS: &str = "vokra.vieneu.local_num_hidden_layers";
const KEY_LOCAL_NUM_ATTENTION_HEADS: &str = "vokra.vieneu.local_num_attention_heads";
const KEY_LOCAL_INTERMEDIATE_SIZE: &str = "vokra.vieneu.local_intermediate_size";

const KEY_AUDIO_SAMPLE_RATE: &str = "vokra.vieneu.audio_sample_rate";
const KEY_AUDIO_VOCAB_SIZE: &str = "vokra.vieneu.audio_vocab_size";
const KEY_N_VQ: &str = "vokra.vieneu.n_vq";
const KEY_AUDIO_TOKENIZER: &str = "vokra.vieneu.audio_tokenizer_ref";

const KEY_TEXT_VOCAB_SIZE: &str = "vokra.vieneu.text_vocab_size";
const KEY_NUM_RESERVED_TOKENS: &str = "vokra.vieneu.num_reserved_tokens";
const KEY_RESERVED_TOKEN_START: &str = "vokra.vieneu.reserved_token_start";

const KEY_BOS_TOKEN_ID: &str = "vokra.vieneu.bos_token_id";
const KEY_EOS_TOKEN_ID: &str = "vokra.vieneu.eos_token_id";
const KEY_PAD_TOKEN_ID: &str = "vokra.vieneu.pad_token_id";
const KEY_UNK_TOKEN_ID: &str = "vokra.vieneu.unk_token_id";
const KEY_TEXT_PROMPT_START_TOKEN_ID: &str = "vokra.vieneu.text_prompt_start_token_id";
const KEY_TEXT_PROMPT_END_TOKEN_ID: &str = "vokra.vieneu.text_prompt_end_token_id";
const KEY_SPEECH_GENERATION_START_TOKEN_ID: &str = "vokra.vieneu.speech_generation_start_token_id";
const KEY_SPEECH_GENERATION_END_TOKEN_ID: &str = "vokra.vieneu.speech_generation_end_token_id";
const KEY_AUDIO_REF_SLOT_TOKEN_ID: &str = "vokra.vieneu.audio_ref_slot_token_id";
const KEY_AUDIO_PAD_TOKEN_ID: &str = "vokra.vieneu.audio_pad_token_id";
const KEY_EMOTION_0_TOKEN_ID: &str = "vokra.vieneu.emotion_0_token_id";
const KEY_EMOTION_1_TOKEN_ID: &str = "vokra.vieneu.emotion_1_token_id";
const KEY_EMOTION_2_TOKEN_ID: &str = "vokra.vieneu.emotion_2_token_id";
const KEY_EMOTION_3_TOKEN_ID: &str = "vokra.vieneu.emotion_3_token_id";
const KEY_EMOTION_4_TOKEN_ID: &str = "vokra.vieneu.emotion_4_token_id";

/// Outcome of a VieNeu-TTS conversion. Mirrors the shared BF16
/// pass-through report shape.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VieNeuReport {
    /// Total tensors observed in the safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors on the pass-through arm.
    pub bf16_passthrough: usize,
}

/// File-based VieNeu-TTS converter (`vokra-cli convert --model
/// vieneu-tts-v3-turbo`).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O; [`ConvertError::Parse`] for malformed
/// safetensors; [`ConvertError::Gguf`] for GGUF writer failure.
pub fn convert_vieneu_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<VieNeuReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);

    // Backbone hparams.
    b.add_u32(KEY_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS);
    b.add_u32(KEY_NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_HEADS);
    b.add_u32(KEY_HEAD_DIM, HEAD_DIM);
    b.add_u32(KEY_INTERMEDIATE_SIZE, INTERMEDIATE_SIZE);
    b.add_u32(KEY_MAX_POSITION_EMBEDDINGS, MAX_POSITION_EMBEDDINGS);
    b.add_f32(KEY_ROPE_THETA, ROPE_THETA);
    b.add_f32(KEY_RMS_NORM_EPS, RMS_NORM_EPS);
    b.add_bool(KEY_TIE_WORD_EMBEDDINGS, TIE_WORD_EMBEDDINGS);

    // Local decoder hparams.
    b.add_u32(KEY_LOCAL_NUM_HIDDEN_LAYERS, LOCAL_NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_LOCAL_NUM_ATTENTION_HEADS, LOCAL_NUM_ATTENTION_HEADS);
    b.add_u32(KEY_LOCAL_INTERMEDIATE_SIZE, LOCAL_INTERMEDIATE_SIZE);

    // Audio codec refs.
    b.add_u32(KEY_AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_RATE);
    b.add_u32(KEY_AUDIO_VOCAB_SIZE, AUDIO_VOCAB_SIZE);
    b.add_u32(KEY_N_VQ, N_VQ);
    b.add_string(KEY_AUDIO_TOKENIZER, AUDIO_TOKENIZER);

    // Text vocab layout.
    b.add_u32(KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_SIZE);
    b.add_u32(KEY_NUM_RESERVED_TOKENS, NUM_RESERVED_TOKENS);
    b.add_u32(KEY_RESERVED_TOKEN_START, RESERVED_TOKEN_START);

    // Special / emotion token ids.
    b.add_u32(KEY_BOS_TOKEN_ID, BOS_TOKEN_ID);
    b.add_u32(KEY_EOS_TOKEN_ID, EOS_TOKEN_ID);
    b.add_u32(KEY_PAD_TOKEN_ID, PAD_TOKEN_ID);
    b.add_u32(KEY_UNK_TOKEN_ID, UNK_TOKEN_ID);
    b.add_u32(KEY_TEXT_PROMPT_START_TOKEN_ID, TEXT_PROMPT_START_TOKEN_ID);
    b.add_u32(KEY_TEXT_PROMPT_END_TOKEN_ID, TEXT_PROMPT_END_TOKEN_ID);
    b.add_u32(
        KEY_SPEECH_GENERATION_START_TOKEN_ID,
        SPEECH_GENERATION_START_TOKEN_ID,
    );
    b.add_u32(
        KEY_SPEECH_GENERATION_END_TOKEN_ID,
        SPEECH_GENERATION_END_TOKEN_ID,
    );
    b.add_u32(KEY_AUDIO_REF_SLOT_TOKEN_ID, AUDIO_REF_SLOT_TOKEN_ID);
    b.add_u32(KEY_AUDIO_PAD_TOKEN_ID, AUDIO_PAD_TOKEN_ID);
    b.add_u32(KEY_EMOTION_0_TOKEN_ID, EMOTION_0_TOKEN_ID);
    b.add_u32(KEY_EMOTION_1_TOKEN_ID, EMOTION_1_TOKEN_ID);
    b.add_u32(KEY_EMOTION_2_TOKEN_ID, EMOTION_2_TOKEN_ID);
    b.add_u32(KEY_EMOTION_3_TOKEN_ID, EMOTION_3_TOKEN_ID);
    b.add_u32(KEY_EMOTION_4_TOKEN_ID, EMOTION_4_TOKEN_ID);

    // Default license = apache-2.0 (HF card front-matter, fetched
    // 2026-07-30).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_HF));

    let mut report = VieNeuReport::default();
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
            "vokra-vieneu-{}-{}-{}.bin",
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
            safetensors_one_bf16("model.embed_tokens.weight", &[2, 3], &bf16),
            bf16,
        )
    }

    #[test]
    fn bf16_passthrough_and_hparams() {
        let (input_bytes, bf16_payload) = synth_bf16();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_vieneu_file(&input, &output, None).unwrap();
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16_payload.as_slice());

        // Every hparam pin matches the transcribed config.json.
        for (k, expect) in [
            (KEY_HIDDEN_SIZE, 768u64),
            (KEY_NUM_HIDDEN_LAYERS, 12),
            (KEY_NUM_ATTENTION_HEADS, 12),
            (KEY_NUM_KEY_VALUE_HEADS, 4),
            (KEY_HEAD_DIM, 64),
            (KEY_INTERMEDIATE_SIZE, 3_072),
            (KEY_MAX_POSITION_EMBEDDINGS, 1_024),
            (KEY_LOCAL_NUM_HIDDEN_LAYERS, 2),
            (KEY_LOCAL_NUM_ATTENTION_HEADS, 8),
            (KEY_LOCAL_INTERMEDIATE_SIZE, 2_048),
            (KEY_AUDIO_SAMPLE_RATE, 48_000),
            (KEY_AUDIO_VOCAB_SIZE, 1_024),
            (KEY_N_VQ, 16),
            (KEY_TEXT_VOCAB_SIZE, 419),
            (KEY_NUM_RESERVED_TOKENS, 30),
            (KEY_RESERVED_TOKEN_START, 13),
            (KEY_BOS_TOKEN_ID, 1),
            (KEY_EOS_TOKEN_ID, 2),
            (KEY_PAD_TOKEN_ID, 0),
            (KEY_UNK_TOKEN_ID, 43),
            (KEY_AUDIO_REF_SLOT_TOKEN_ID, 7),
            (KEY_AUDIO_PAD_TOKEN_ID, 1_024),
            (KEY_EMOTION_0_TOKEN_ID, 8),
            (KEY_EMOTION_4_TOKEN_ID, 12),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(expect),
                "{k} pin"
            );
        }

        // Float-typed rope_theta / rms_norm_eps.
        assert_eq!(
            file.get(KEY_ROPE_THETA).and_then(|v| v.as_f64()),
            Some(f64::from(10_000.0f32))
        );
        assert!(
            file.get(KEY_RMS_NORM_EPS)
                .and_then(|v| v.as_f64())
                .is_some(),
            "rms_norm_eps must be stamped as an f32"
        );

        // Audio tokenizer ref recorded.
        assert_eq!(
            file.get(KEY_AUDIO_TOKENIZER).and_then(|v| v.as_str()),
            Some(AUDIO_TOKENIZER)
        );

        // Arch identity + provenance defaults.
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
            Some(DEFAULT_LICENSE)
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
