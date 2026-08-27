//! **Qwen3-TTS-12Hz released family**: safetensors checkpoint → GGUF
//! conversion (SoTA plan Phase 3, 2026-07-24; contract corrected
//! 2026-08-27).
//!
//! Input: one of the exact official 0.6B/1.7B Base, CustomVoice or
//! VoiceDesign manifests. Output: a GGUF carrying every float tensor plus
//! the `vokra.qwen3_tts.*` and `vokra.model.*` / `vokra.provenance.*`
//! metadata chunks the native Qwen3-TTS implementation
//! (`crates/vokra-models/src/qwen3_tts/`) reads.
//!
//! # BF16 handling — exact pass-through
//!
//! **Decision**: BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`). There is no convert-time widening; runtime widens
//! only a requested tensor through the canonical BF16 decoder. The current
//! converter uses the shared in-memory safetensors/builder path, so real
//! conversion is VAST-only under the repository's aggregate-artifact
//! threshold. This module does not claim a streaming writer path.
//!
//! F16 downcast and silent BF16 → F32 conversion remain forbidden. Tests
//! pin the dtype and payload bytes symmetrically with the F16/F32 paths.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.qwen3_tts.*` chunk group is transcribed **verbatim** from
//!   the primary source `config.json`
//!   (`huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`,
//!   fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). No axis
//!   is invented; any tensor whose shape disagrees with these values
//!   fails the runtime shape gate loudly (FR-EX-08,
//!   `Qwen3TtsConfig::validate_for_forward`).
//! - **Two sub-configs** — Qwen3-TTS splits its `config.json` into a
//!   `talker.*` (main AR LM) block and a `code_predictor.*` (5-layer
//!   parallel head that slots 16 codebook rows per step) block. Both
//!   are transcribed in full.
//! - **Codec handshake** — `num_code_groups` on both the talker and
//!   the code predictor must equal
//!   [`vokra_ops::qwen3_tts_codec::Qwen3TtsCodecConfig::num_quantizers`]
//!   on the shared seam (16 for every released variant). Silently
//!   drifting the two would drop or duplicate codebook rows at
//!   decode time.
//!
//! # No side-car config
//!
//! Qwen3-TTS ships a real upstream `config.json`, but this converter
//! takes **no** `--config` path today because every field is fixed for
//! each released variant and byte-parallel to the transcribed
//! constants below. The variant is selected by the [`Qwen3TtsVariant`]
//! caller argument (dispatched from the CLI's `--model` alias walk in
//! `crates/vokra-convert/src/lib.rs`):
//!
//! - [`Qwen3TtsVariant::_0_6B_Base`] — `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
//!   (talker hidden=1024, ffn=3072). Primary source
//!   `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`
//!   fetched 2026-07-24.
//! - [`Qwen3TtsVariant::_0_6B_CustomVoice`] —
//!   `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (same 1024/3072 talker,
//!   fixed speaker ids, no speaker encoder). The generic 0.6B converter
//!   distinguishes this 402-tensor manifest from Base's 478 tensors.
//! - [`Qwen3TtsVariant::_1_7B_Base`] —
//!   `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (talker hidden=2048,
//!   ffn=6144; the un-fine-tuned 1.7B backbone that the CustomVoice /
//!   VoiceDesign 1.7B siblings fine-tune from). Primary source
//!   `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json`
//!   fetched 2026-08-01.
//! - [`Qwen3TtsVariant::_1_7B_CustomVoice`] —
//!   `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (talker hidden=2048,
//!   ffn=6144; identical talker + code-predictor axes to the 1.7B-Base
//!   sibling, but no Base-only speaker encoder; the fine-tune target is
//!   `tts_model_type = "custom_voice"`). Primary source
//!   `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/raw/main/config.json`
//!   fetched 2026-07-30.
//! - [`Qwen3TtsVariant::_1_7B_VoiceDesign`] —
//!   `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (identical talker axes to
//!   CustomVoice; distinct HF release + `tts_model_type = "voice_design"`
//!   vs `"custom_voice"`; distinct NAME stamp). Primary source
//!   `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/config.json`
//!   fetched 2026-07-30.
//!
//! A future variant that reshapes the backbone further (2B / 7B) would
//! extend the [`Qwen3TtsVariant`] enum with its own constants; this
//! converter fails loudly if a tensor shape disagrees with the
//! selected variant's transcribed axes (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox contract). The runtime's
//! `Qwen3TtsCheckpoint::from_gguf` validates the exact official
//! 402/404/478/480-tensor variant manifest before any block is decoded.
//!
//! # BF16 posture
//!
//! The upstream Qwen3-TTS releases are served in **BF16**. BF16 tensors pass
//! through **verbatim** as GGUF
//! type 30 (`GgmlType::BF16`) with no convert-time widening; the
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16
//! is the top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`Qwen3TtsReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm (rewrite of the M4-06 posture pin per
//! the ADR's symmetric-rewrite red-line).
//!
//! # No ONNX (permanent)
//!
//! Qwen3-TTS is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/qwen3_tts/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Qwen3-TTS GGUFs — kept in sync with the
/// runtime constant `vokra-models::qwen3_tts::EXPECTED_ARCH`.
/// Intentionally **distinct** from the Qwen-family sibling arch tags
/// (`"cosyvoice2"` / `"cosyvoice3"`) because Qwen3-TTS is codec-LM,
/// not vocoder-LM — the terminal step is `qwen3_tts_codec`, NOT
/// `HiFTChain`. Silently sharing an arch tag would mis-route the
/// runtime dispatch.
pub(crate) const ARCH: &str = "qwen3_tts";
/// `vokra.model.name` value written for the canonical Qwen3-TTS-0.6B-Base
/// GGUF.
pub(crate) const NAME: &str = "qwen3-tts-12hz-0.6b-base";

/// `vokra.model.name` value written for the
/// `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` variant.
pub(crate) const NAME_0_6B_CUSTOM_VOICE: &str = "qwen3-tts-12hz-0.6b-customvoice";

/// `vokra.model.name` value written for the
/// `Qwen/Qwen3-TTS-12Hz-1.7B-Base` variant (un-fine-tuned 1.7B backbone;
/// added 2026-08-01, Wave 4).
pub(crate) const NAME_1_7B_BASE: &str = "qwen3-tts-12hz-1.7b-base";

/// `vokra.model.name` value written for the
/// `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` variant.
pub(crate) const NAME_1_7B_CUSTOM_VOICE: &str = "qwen3-tts-12hz-1.7b-customvoice";

/// `vokra.model.name` value written for the
/// `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` variant.
pub(crate) const NAME_1_7B_VOICE_DESIGN: &str = "qwen3-tts-12hz-1.7b-voicedesign";

const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.qwen3_tts.source_revision";
pub(crate) const KEY_CONFIG_JSON: &str = "vokra.qwen3_tts.config_json";
pub(crate) const KEY_TOKENIZER_VOCAB: &str = "vokra.qwen3_tts.tokenizer.vocab_json";
pub(crate) const KEY_TOKENIZER_MERGES: &str = "vokra.qwen3_tts.tokenizer.merges_txt";
pub(crate) const KEY_TOKENIZER_CONFIG: &str = "vokra.qwen3_tts.tokenizer.config_json";
pub(crate) const KEY_GENERATION_CONFIG: &str = "vokra.qwen3_tts.generation.config_json";

const VOCAB_FILE: ExactSidecar = ExactSidecar {
    name: "vocab.json",
    bytes: 2_776_833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
};
const MERGES_FILE: ExactSidecar = ExactSidecar {
    name: "merges.txt",
    bytes: 1_671_839,
    sha256: "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
};
const TOKENIZER_CONFIG_FILE: ExactSidecar = ExactSidecar {
    name: "tokenizer_config.json",
    bytes: 7_344,
    sha256: "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
};
const GENERATION_CONFIG_FILE: ExactSidecar = ExactSidecar {
    name: "generation_config.json",
    bytes: 245,
    sha256: "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
};

// --- vokra.qwen3_tts.* metadata keys (kept as constants in the converter;
// the runtime side lives in `crates/vokra-models/src/qwen3_tts/mod.rs` —
// the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox family
// converters use applies) -----------------------------------------------------

// Top-level (speaker encoder + sample rate)
const KEY_SAMPLE_RATE: &str = "vokra.qwen3_tts.sample_rate";
const KEY_SPEAKER_EMBED_DIM: &str = "vokra.qwen3_tts.speaker_embed_dim";
const KEY_HAS_SPEAKER_ENCODER: &str = "vokra.qwen3_tts.has_speaker_encoder";
const KEY_TTS_MODEL_SIZE: &str = "vokra.qwen3_tts.tts_model_size";
const KEY_TTS_MODEL_TYPE: &str = "vokra.qwen3_tts.tts_model_type";

// Talker (main AR LM) axes — config.json.talker.*
const KEY_TALKER_HIDDEN_DIM: &str = "vokra.qwen3_tts.talker.hidden_dim";
const KEY_TALKER_N_LAYER: &str = "vokra.qwen3_tts.talker.n_layer";
const KEY_TALKER_N_HEAD: &str = "vokra.qwen3_tts.talker.n_head";
const KEY_TALKER_N_HEAD_KV: &str = "vokra.qwen3_tts.talker.n_head_kv";
const KEY_TALKER_HEAD_DIM: &str = "vokra.qwen3_tts.talker.head_dim";
const KEY_TALKER_FFN_DIM: &str = "vokra.qwen3_tts.talker.ffn_dim";
const KEY_TALKER_VOCAB_SIZE: &str = "vokra.qwen3_tts.talker.vocab_size";
const KEY_TALKER_TEXT_VOCAB_SIZE: &str = "vokra.qwen3_tts.talker.text_vocab_size";
const KEY_TALKER_MAX_POSITIONS: &str = "vokra.qwen3_tts.talker.max_position_embeddings";
const KEY_TALKER_ROPE_BASE: &str = "vokra.qwen3_tts.talker.rope_base";
const KEY_TALKER_RMS_NORM_EPS: &str = "vokra.qwen3_tts.talker.rms_norm_eps";
const KEY_TALKER_POS_ID_PER_SEC: &str = "vokra.qwen3_tts.talker.position_id_per_seconds";
const KEY_TALKER_NUM_CODE_GROUPS: &str = "vokra.qwen3_tts.talker.num_code_groups";
const KEY_TALKER_TEXT_HIDDEN_SIZE: &str = "vokra.qwen3_tts.talker.text_hidden_size";

// Code predictor axes — config.json.code_predictor.*
const KEY_CP_HIDDEN_DIM: &str = "vokra.qwen3_tts.code_predictor.hidden_dim";
const KEY_CP_N_LAYER: &str = "vokra.qwen3_tts.code_predictor.n_layer";
const KEY_CP_N_HEAD: &str = "vokra.qwen3_tts.code_predictor.n_head";
const KEY_CP_N_HEAD_KV: &str = "vokra.qwen3_tts.code_predictor.n_head_kv";
const KEY_CP_HEAD_DIM: &str = "vokra.qwen3_tts.code_predictor.head_dim";
const KEY_CP_FFN_DIM: &str = "vokra.qwen3_tts.code_predictor.ffn_dim";
const KEY_CP_VOCAB_SIZE: &str = "vokra.qwen3_tts.code_predictor.vocab_size";
const KEY_CP_ROPE_BASE: &str = "vokra.qwen3_tts.code_predictor.rope_base";
const KEY_CP_RMS_NORM_EPS: &str = "vokra.qwen3_tts.code_predictor.rms_norm_eps";
const KEY_CP_NUM_CODE_GROUPS: &str = "vokra.qwen3_tts.code_predictor.num_code_groups";

// Model family marker (Qwen3-TTS is Qwen3-flavour — same op inventory as
// Qwen2 but wider head split + rope base 1_000_000).
const KEY_MODEL_FAMILY: &str = "vokra.qwen3_tts.model_family";

// --- Transcribed constants (primary source: `config.json` at
// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`, fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」) ------------------------------------

/// PCM sample rate the speaker encoder consumes (Hz). Fixed by
/// `README.md` — "Speaker Encoder: 24kHz sample rate, 1024-dim
/// encoding" — and shared with the codec's `sample_rate = 24_000` on
/// the [`vokra_ops::qwen3_tts_codec`] seam.
const QWEN3_TTS_SAMPLE_RATE: u32 = 24_000;

/// Speaker embedding width for 0.6B-Base
/// (`speaker_encoder_config.enc_dim = 1024`).
const QWEN3_TTS_0_6B_SPEAKER_EMBED_DIM: u32 = 1024;
/// Speaker embedding width for 1.7B-Base
/// (`speaker_encoder_config.enc_dim = 2048`).
const QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM: u32 = 2048;

// Talker (config.json.talker.*)
const TALKER_HIDDEN_DIM: u32 = 1024;
const TALKER_N_LAYER: u32 = 28;
const TALKER_N_HEAD: u32 = 16;
const TALKER_N_HEAD_KV: u32 = 8;
const TALKER_HEAD_DIM: u32 = 128;
const TALKER_FFN_DIM: u32 = 3072;
const TALKER_VOCAB_SIZE: u32 = 3072;
const TALKER_TEXT_VOCAB_SIZE: u32 = 151_936;
const TALKER_MAX_POSITIONS: u32 = 32_768;
const TALKER_ROPE_BASE: f32 = 1_000_000.0;
const TALKER_RMS_NORM_EPS: f32 = 1e-6;
const TALKER_POS_ID_PER_SEC: u32 = 13;
const TALKER_NUM_CODE_GROUPS: u32 = 16;
const TALKER_TEXT_HIDDEN_SIZE: u32 = 2048;

// Code predictor (config.json.code_predictor.*)
const CP_HIDDEN_DIM: u32 = 1024;
const CP_N_LAYER: u32 = 5;
const CP_N_HEAD: u32 = 16;
const CP_N_HEAD_KV: u32 = 8;
const CP_HEAD_DIM: u32 = 128;
const CP_FFN_DIM: u32 = 3072;
const CP_VOCAB_SIZE: u32 = 2048;
const CP_ROPE_BASE: f32 = 1_000_000.0;
const CP_RMS_NORM_EPS: f32 = 1e-6;
const CP_NUM_CODE_GROUPS: u32 = 16;

/// Model family marker — Qwen3 (`config.json.model_type = "qwen3_tts"`
/// and `config.json.architectures = ["Qwen3TTSForConditionalGeneration"]`).
/// Recorded so the runtime can distinguish Qwen3-TTS from
/// Qwen2-derived siblings (CosyVoice2/3) at telemetry time.
const MODEL_FAMILY: &str = "qwen3";

// --- 1.7B variant talker axes (primary source:
// `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/raw/main/config.json`
// + `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/config.json`
// fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
//
// The two 1.7B releases (CustomVoice + VoiceDesign) share **identical**
// talker + code-predictor axes; only `tts_model_type` and the NAME
// stamp differ. So a single set of "1_7B" talker constants covers both
// and each ModelKind arm stamps its own NAME via the variant selector.
//
// Axes that MATCH 0.6B are declared as constants here (rather than
// being unused) so:
//   (a) the 1.7B primary-source ingest is recorded in one place, in
//       full, right next to the 0.6B ingest — a reviewer diffing the
//       two sets sees exactly which axes actually widened (hidden /
//       ffn / text_hidden), rather than having to know that "silence
//       here means matches 0.6B";
//   (b) `transcribed_1_7b_constants_match_primary_source` regression-
//       pins every 1.7B axis against the 0.6B constant it currently
//       matches. If a future variant landed a widened axis that we
//       forgot to plumb into `write_hparams` via a selector, this pin
//       would catch the drift at test time.
//
// Constants that match 0.6B carry `#[allow(dead_code)]` because the
// production `write_hparams` path routes those axes to the 0.6B
// constant already (they're equal — routing either constant gives the
// same GGUF bytes); the constants exist purely for review + regression
// pinning above. The `_HIDDEN_DIM` / `_FFN_DIM` / `_TEXT_HIDDEN_SIZE`
// constants that DO differ from 0.6B are used by the variant selectors
// below and carry no attribute.
// ---------------------------------------------------------------------

/// Talker `hidden_size` for the 1.7B variants (both CustomVoice and
/// VoiceDesign). `config.json.talker_config.hidden_size = 2048`.
const TALKER_1_7B_HIDDEN_DIM: u32 = 2048;
/// Talker `intermediate_size` (SwiGLU FFN inner dim) for the 1.7B
/// variants. `config.json.talker_config.intermediate_size = 6144`.
const TALKER_1_7B_FFN_DIM: u32 = 6144;
/// Talker `num_hidden_layers` for the 1.7B variants. Same as 0.6B (28).
#[allow(dead_code)]
const TALKER_1_7B_N_LAYER: u32 = 28;
/// Talker `num_attention_heads` for the 1.7B variants. Same as 0.6B (16).
#[allow(dead_code)]
const TALKER_1_7B_N_HEAD: u32 = 16;
/// Talker `num_key_value_heads` for the 1.7B variants. Same as 0.6B (8).
#[allow(dead_code)]
const TALKER_1_7B_N_HEAD_KV: u32 = 8;
/// Talker `head_dim` for the 1.7B variants. Same as 0.6B (128).
#[allow(dead_code)]
const TALKER_1_7B_HEAD_DIM: u32 = 128;
/// Talker per-codebook speech-token vocabulary for the 1.7B variants.
/// Same as 0.6B (3072).
#[allow(dead_code)]
const TALKER_1_7B_VOCAB_SIZE: u32 = 3072;
/// Talker `text_vocab_size` for the 1.7B variants. Same as 0.6B
/// (151 936 — the Qwen3 base tokenizer).
#[allow(dead_code)]
const TALKER_1_7B_TEXT_VOCAB_SIZE: u32 = 151_936;
/// Talker `max_position_embeddings` for the 1.7B variants. Same as
/// 0.6B (32 768).
#[allow(dead_code)]
const TALKER_1_7B_MAX_POSITIONS: u32 = 32_768;
/// Talker RoPE base θ for the 1.7B variants. Same as 0.6B (1 000 000).
#[allow(dead_code)]
const TALKER_1_7B_ROPE_BASE: f32 = 1_000_000.0;
/// Talker RMSNorm ε for the 1.7B variants. Same as 0.6B (1e-6).
#[allow(dead_code)]
const TALKER_1_7B_RMS_NORM_EPS: f32 = 1e-6;
/// Talker `position_id_per_seconds` for the 1.7B variants. Same as
/// 0.6B (13 — 12.5 Hz codec + slack).
#[allow(dead_code)]
const TALKER_1_7B_POS_ID_PER_SEC: u32 = 13;
/// Talker `num_code_groups` for the 1.7B variants. Same as 0.6B (16 —
/// must match `Qwen3TtsCodecConfig::num_quantizers`).
#[allow(dead_code)]
const TALKER_1_7B_NUM_CODE_GROUPS: u32 = 16;
/// Talker `text_hidden_size` for the 1.7B variants. Same as 0.6B
/// (2048 — the width of the text encoder that feeds the talker; note
/// this now equals the 1.7B talker hidden, so the projection is
/// identity-sized).
const TALKER_1_7B_TEXT_HIDDEN_SIZE: u32 = 2048;

// Code predictor axes for the 1.7B variants — IDENTICAL to 0.6B EXCEPT
// `max_position_embeddings` (0.6B not tracked, 1.7B raised to 65 536 —
// but the current metadata schema does not carry the CP max positions,
// so no new key today). All other CP axes match the 0.6B constants
// declared above. If a future runtime binder needs the CP max
// positions, a `KEY_CP_MAX_POSITIONS` chunk should be added
// symmetrically to both 0.6B and 1.7B constants and populated with
// `32_768` / `65_536` respectively.

/// Which Qwen3-TTS release variant to stamp into the emitted GGUF.
///
/// Each variant selects a distinct set of talker axes and a distinct
/// `vokra.model.name` string. The code-predictor axes are identical
/// across every released variant.
///
/// The variant names begin with a numeric-size prefix (`_0_6B_Base`,
/// `_1_7B_Base`, `_1_7B_CustomVoice`, `_1_7B_VoiceDesign`) so the enum
/// reads verbatim against the upstream HF release ids (`0.6B-Base`,
/// `1.7B-Base`, `1.7B-CustomVoice`, `1.7B-VoiceDesign`). Rust's default
/// `non_camel_case_types` lint would rename them to something like
/// `_0_6bBase` and lose that fidelity, so it is silenced for this enum
/// only — this is a deliberate deviation from the workspace style
/// guide, kept confined to this one type.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3TtsVariant {
    /// `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — the original 0.6B release.
    /// Talker hidden=1024, ffn=3072.
    _0_6B_Base,
    /// `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` — fixed-speaker variant.
    /// Its official config has no speaker encoder and its checkpoint omits
    /// all 76 `speaker_encoder.*` tensors.
    _0_6B_CustomVoice,
    /// `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — the un-fine-tuned 1.7B backbone
    /// that the CustomVoice / VoiceDesign 1.7B siblings fine-tune from
    /// (added 2026-08-01, Wave 4). Talker axes are byte-identical to
    /// the two 1.7B fine-tuned siblings (hidden=2048, ffn=6144); only
    /// the HF release id + `vokra.model.name` stamp differ. Distinct
    /// arm rather than a slug-only add on `_1_7B_CustomVoice` because
    /// this is the untuned base checkpoint (its `tts_model_type` is
    /// distinct from `"custom_voice"` / `"voice_design"`) and a
    /// downstream that ships all three GGUFs side-by-side needs
    /// distinguishable provenance stamps.
    _1_7B_Base,
    /// `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` — 1.7B variant tuned for
    /// zero-shot voice cloning (`tts_model_type = "custom_voice"`).
    /// Talker hidden=2048, ffn=6144.
    _1_7B_CustomVoice,
    /// `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — 1.7B variant tuned for
    /// text-prompt voice-design synthesis (`tts_model_type =
    /// "voice_design"`). Identical talker axes to CustomVoice; distinct
    /// HF release + NAME stamp only.
    _1_7B_VoiceDesign,
}

impl Qwen3TtsVariant {
    /// The `vokra.model.name` string stamped for this variant.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::_0_6B_Base => NAME,
            Self::_0_6B_CustomVoice => NAME_0_6B_CUSTOM_VOICE,
            Self::_1_7B_Base => NAME_1_7B_BASE,
            Self::_1_7B_CustomVoice => NAME_1_7B_CUSTOM_VOICE,
            Self::_1_7B_VoiceDesign => NAME_1_7B_VOICE_DESIGN,
        }
    }

    pub(crate) const fn upstream_hf(self) -> &'static str {
        match self {
            Self::_0_6B_Base => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::_0_6B_CustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::_1_7B_Base => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::_1_7B_CustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::_1_7B_VoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        }
    }

    pub(crate) const fn source_revision(self) -> &'static str {
        match self {
            Self::_0_6B_Base => "5d83992436eae1d760afd27aff78a71d676296fc",
            Self::_0_6B_CustomVoice => "85e237c12c027371202489a0ec509ded67b5e4b5",
            Self::_1_7B_Base => "fd4b254389122332181a7c3db7f27e918eec64e3",
            Self::_1_7B_CustomVoice => "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            Self::_1_7B_VoiceDesign => "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        }
    }

    const fn config_file(self) -> ExactSidecar {
        match self {
            Self::_0_6B_Base => ExactSidecar {
                name: "config.json",
                bytes: 4_494,
                sha256: "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
            },
            Self::_0_6B_CustomVoice => ExactSidecar {
                name: "config.json",
                bytes: 4_908,
                sha256: "81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455",
            },
            Self::_1_7B_Base => ExactSidecar {
                name: "config.json",
                bytes: 4_494,
                sha256: "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
            },
            Self::_1_7B_CustomVoice => ExactSidecar {
                name: "config.json",
                bytes: 4_908,
                sha256: "17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9",
            },
            Self::_1_7B_VoiceDesign => ExactSidecar {
                name: "config.json",
                bytes: 4_421,
                sha256: "aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0",
            },
        }
    }

    /// Talker hidden dimension for this variant. Every 1.7B variant
    /// (Base / CustomVoice / VoiceDesign) shares the widened axis
    /// `TALKER_1_7B_HIDDEN_DIM = 2048`.
    pub(crate) const fn talker_hidden_dim(self) -> u32 {
        match self {
            Self::_0_6B_Base | Self::_0_6B_CustomVoice => TALKER_HIDDEN_DIM,
            Self::_1_7B_Base | Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => {
                TALKER_1_7B_HIDDEN_DIM
            }
        }
    }

    /// Talker SwiGLU inner (`intermediate_size`) for this variant.
    /// Every 1.7B variant shares `TALKER_1_7B_FFN_DIM = 6144`.
    pub(crate) const fn talker_ffn_dim(self) -> u32 {
        match self {
            Self::_0_6B_Base | Self::_0_6B_CustomVoice => TALKER_FFN_DIM,
            Self::_1_7B_Base | Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => {
                TALKER_1_7B_FFN_DIM
            }
        }
    }

    /// Talker `text_hidden_size` for this variant. Every 1.7B variant
    /// shares `TALKER_1_7B_TEXT_HIDDEN_SIZE = 2048` (identity-sized
    /// against the widened talker hidden).
    pub(crate) const fn talker_text_hidden_size(self) -> u32 {
        match self {
            Self::_0_6B_Base | Self::_0_6B_CustomVoice => TALKER_TEXT_HIDDEN_SIZE,
            Self::_1_7B_Base | Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => {
                TALKER_1_7B_TEXT_HIDDEN_SIZE
            }
        }
    }

    /// Exact upstream size marker from `config.json.tts_model_size`.
    const fn model_size(self) -> &'static str {
        match self {
            Self::_0_6B_Base | Self::_0_6B_CustomVoice => "0b6",
            Self::_1_7B_Base | Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => "1b7",
        }
    }

    /// Exact upstream mode marker from `config.json.tts_model_type`.
    const fn model_type(self) -> &'static str {
        match self {
            Self::_0_6B_Base | Self::_1_7B_Base => "base",
            Self::_0_6B_CustomVoice | Self::_1_7B_CustomVoice => "custom_voice",
            Self::_1_7B_VoiceDesign => "voice_design",
        }
    }

    /// Only Base checkpoints instantiate `Qwen3TTSSpeakerEncoder` in the
    /// official implementation.
    const fn has_speaker_encoder(self) -> bool {
        matches!(self, Self::_0_6B_Base | Self::_1_7B_Base)
    }

    /// Exact speaker-encoder output width. Non-Base variants have no speaker
    /// encoder and therefore stamp zero instead of inventing an axis.
    const fn speaker_embed_dim(self) -> u32 {
        match self {
            Self::_0_6B_Base => QWEN3_TTS_0_6B_SPEAKER_EMBED_DIM,
            Self::_1_7B_Base => QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM,
            Self::_0_6B_CustomVoice | Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => 0,
        }
    }

    const fn expected_tensor_count(self) -> usize {
        match self {
            Self::_0_6B_Base => 478,
            Self::_0_6B_CustomVoice => 402,
            Self::_1_7B_Base => 480,
            Self::_1_7B_CustomVoice | Self::_1_7B_VoiceDesign => 404,
        }
    }
}

/// Outcome of a Qwen3-TTS conversion.
#[derive(Debug, Default)]
pub(crate) struct Qwen3TtsReport {
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub(crate) written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling converters and to
    /// surface the "no float tensors" loud note when zero writes
    /// occur).
    pub(crate) skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — the ADR
    /// (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough, Accepted
    /// 2026-07-25) demands it as the symmetric rewrite of the M4-06
    /// posture pin so a latent silent-widen cannot slip in
    /// undetected. Mirrors `moshi::MoshiReport::bf16_passthrough`.
    pub(crate) bf16_passthrough: usize,
    /// Metadata entries written after fixed-revision release assets are
    /// embedded. Builder-only fixture conversion leaves this at zero.
    pub(crate) metadata_count: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Qwen3-TTS safetensors buffer into a populated GGUF
/// builder, using the canonical [`Qwen3TtsVariant::_0_6B_Base`] variant.
///
/// This is the backward-compatible entry point that mirrors the
/// pre-1.7B-variant signature; new callers should prefer
/// [`convert_variant`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream name; the
/// `vokra.qwen3_tts.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive`
/// (apache-2.0 — end-to-end).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, Qwen3TtsReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let variant = detect_0_6b_variant(&st)?;
    convert_parsed(st, variant)
}

/// Converts a Qwen3-TTS safetensors buffer into a populated GGUF builder
/// for the given release [`Qwen3TtsVariant`].
///
/// The talker axes, model type and `vokra.model.name` stamp are
/// variant-selected. Base-only speaker presence and width are also selected
/// exactly (1024 for 0.6B-Base, 2048 for 1.7B-Base, absent otherwise).
/// Every F32 / F16 / BF16 tensor passes through verbatim (ADR
/// A_passthrough). Provenance = `Permissive` (apache-2.0 end-to-end)
/// for every variant.
pub(crate) fn convert_variant(
    bytes: Vec<u8>,
    variant: Qwen3TtsVariant,
) -> Result<(GgufBuilder, Qwen3TtsReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&st, variant)?;
    convert_parsed(st, variant)
}

pub(crate) fn convert_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen3TtsReport, ConvertError> {
    validate_license(license, None)?;
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let variant = detect_0_6b_variant(&st)?;
    convert_parsed_file(st, input, output, variant)
}

pub(crate) fn convert_file_with_variant(
    input: &Path,
    output: &Path,
    variant: Qwen3TtsVariant,
    license: Option<&str>,
) -> Result<Qwen3TtsReport, ConvertError> {
    validate_license(license, Some(variant))?;
    let st = SafetensorsFile::parse(std::fs::read(input)?)?;
    validate_checkpoint(&st, variant)?;
    convert_parsed_file(st, input, output, variant)
}

fn validate_license(
    license: Option<&str>,
    variant: Option<Qwen3TtsVariant>,
) -> Result<(), ConvertError> {
    if let Some(value) = license
        && !value.is_empty()
        && !value.eq_ignore_ascii_case("apache-2.0")
    {
        let release = variant
            .map(Qwen3TtsVariant::upstream_hf)
            .unwrap_or("the exact detected Qwen3-TTS 0.6B release");
        return Err(ConvertError::Usage(format!(
            "qwen3-tts: {release} has pinned Apache-2.0 weights and sidecars; refusing conflicting --license {value:?}"
        )));
    }
    Ok(())
}

fn convert_parsed_file(
    st: SafetensorsFile,
    input: &Path,
    output: &Path,
    variant: Qwen3TtsVariant,
) -> Result<Qwen3TtsReport, ConvertError> {
    let (mut builder, mut report) = convert_parsed(st, variant)?;
    ReleaseAssets::load(input, variant)?.embed(&mut builder);
    report.metadata_count = builder.metadata_count();
    let bytes = builder.to_bytes()?;
    std::fs::write(output, bytes)?;
    Ok(report)
}

#[derive(Debug, Clone, Copy)]
struct ExactSidecar {
    name: &'static str,
    bytes: usize,
    sha256: &'static str,
}

#[derive(Debug)]
struct ReleaseAssets {
    config: Vec<u8>,
    vocab: Vec<u8>,
    merges: Vec<u8>,
    tokenizer_config: Vec<u8>,
    generation_config: Vec<u8>,
}

impl ReleaseAssets {
    fn load(input: &Path, variant: Qwen3TtsVariant) -> Result<Self, ConvertError> {
        let directory = input.parent().unwrap_or_else(|| Path::new("."));
        Ok(Self {
            config: read_exact_sidecar(directory, variant.config_file(), variant)?,
            vocab: read_exact_sidecar(directory, VOCAB_FILE, variant)?,
            merges: read_exact_sidecar(directory, MERGES_FILE, variant)?,
            tokenizer_config: read_exact_sidecar(directory, TOKENIZER_CONFIG_FILE, variant)?,
            generation_config: read_exact_sidecar(directory, GENERATION_CONFIG_FILE, variant)?,
        })
    }

    fn embed(&self, builder: &mut GgufBuilder) {
        add_u8_array(builder, KEY_CONFIG_JSON, &self.config);
        add_u8_array(builder, KEY_TOKENIZER_VOCAB, &self.vocab);
        add_u8_array(builder, KEY_TOKENIZER_MERGES, &self.merges);
        add_u8_array(builder, KEY_TOKENIZER_CONFIG, &self.tokenizer_config);
        add_u8_array(builder, KEY_GENERATION_CONFIG, &self.generation_config);
    }
}

fn read_exact_sidecar(
    directory: &Path,
    spec: ExactSidecar,
    variant: Qwen3TtsVariant,
) -> Result<Vec<u8>, ConvertError> {
    let path = directory.join(spec.name);
    let bytes = std::fs::read(&path).map_err(|error| {
        ConvertError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "qwen3-tts: reading required {}@{} sidecar {}: {error}",
                variant.upstream_hf(),
                variant.source_revision(),
                path.display()
            ),
        ))
    })?;
    if bytes.len() != spec.bytes {
        return Err(ConvertError::Parse(format!(
            "qwen3-tts: {}@{} sidecar {} is {} bytes, expected exactly {}",
            variant.upstream_hf(),
            variant.source_revision(),
            spec.name,
            bytes.len(),
            spec.bytes
        )));
    }
    let actual =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(&bytes));
    if actual != spec.sha256 {
        return Err(ConvertError::Parse(format!(
            "qwen3-tts: {}@{} sidecar {} SHA-256 {actual}, expected {}",
            variant.upstream_hf(),
            variant.source_revision(),
            spec.name,
            spec.sha256
        )));
    }
    Ok(bytes)
}

fn add_u8_array(builder: &mut GgufBuilder, key: &str, bytes: &[u8]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().copied().map(GgufMetadataValue::U8).collect(),
        }),
    );
}

fn convert_parsed(
    st: SafetensorsFile,
    variant: Qwen3TtsVariant,
) -> Result<(GgufBuilder, Qwen3TtsReport), ConvertError> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, variant.source_revision());
    b.add_string(KEY_SOURCE_REVISION, variant.source_revision());
    write_hparams(&mut b, variant);
    // Self-describing redistribution: the artifact carries its own licence.
    // Every Qwen3-TTS release ships `apache-2.0` end-to-end
    // (huggingface.co/Qwen/Qwen3-TTS-12Hz-{0.6B-Base,1.7B-Base,
    // 1.7B-CustomVoice,1.7B-VoiceDesign} model-card YAML front matter
    // `license: apache-2.0`, fetched 2026-07-24 / 2026-07-30 / 2026-08-01
    // — CLAUDE.md「ハルシネーション厳禁」). The whole release — LM +
    // codec + tokenizer + speaker encoder — carries a single apache-2.0
    // grant.
    let source = format!(
        "{}@{} (apache-2.0 end-to-end)",
        variant.upstream_hf(),
        variant.source_revision()
    );
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(variant.name()),
        Some(&source),
    );

    let mut report = Qwen3TtsReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `moshi::convert` (`crates/vokra-convert/src/models/moshi.rs`).
    for t in st.tensors() {
        match t.dtype {
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
             upstream Qwen3-TTS release ships \
             `model.safetensors` in BF16 (1.83 GB for 0.6B, ~3.8 GB for 1.7B); \
             the converter passes BF16 tensors through verbatim (ADR A_passthrough), \
             so a zero-write outcome here means the safetensors file itself was empty."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.qwen3_tts.*` chunk group from the transcribed
/// constants above (primary source: `config.json`), using the variant-
/// selected talker axes.
fn write_hparams(b: &mut GgufBuilder, variant: Qwen3TtsVariant) {
    b.add_u32(KEY_SAMPLE_RATE, QWEN3_TTS_SAMPLE_RATE);
    b.add_u32(KEY_SPEAKER_EMBED_DIM, variant.speaker_embed_dim());
    b.add_bool(KEY_HAS_SPEAKER_ENCODER, variant.has_speaker_encoder());
    b.add_string(KEY_TTS_MODEL_SIZE, variant.model_size());
    b.add_string(KEY_TTS_MODEL_TYPE, variant.model_type());
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);

    // Talker — variant-selected axes.
    b.add_u32(KEY_TALKER_HIDDEN_DIM, variant.talker_hidden_dim());
    b.add_u32(KEY_TALKER_N_LAYER, TALKER_N_LAYER);
    b.add_u32(KEY_TALKER_N_HEAD, TALKER_N_HEAD);
    b.add_u32(KEY_TALKER_N_HEAD_KV, TALKER_N_HEAD_KV);
    b.add_u32(KEY_TALKER_HEAD_DIM, TALKER_HEAD_DIM);
    b.add_u32(KEY_TALKER_FFN_DIM, variant.talker_ffn_dim());
    b.add_u32(KEY_TALKER_VOCAB_SIZE, TALKER_VOCAB_SIZE);
    b.add_u32(KEY_TALKER_TEXT_VOCAB_SIZE, TALKER_TEXT_VOCAB_SIZE);
    b.add_u32(KEY_TALKER_MAX_POSITIONS, TALKER_MAX_POSITIONS);
    b.add_f32(KEY_TALKER_ROPE_BASE, TALKER_ROPE_BASE);
    b.add_f32(KEY_TALKER_RMS_NORM_EPS, TALKER_RMS_NORM_EPS);
    b.add_u32(KEY_TALKER_POS_ID_PER_SEC, TALKER_POS_ID_PER_SEC);
    b.add_u32(KEY_TALKER_NUM_CODE_GROUPS, TALKER_NUM_CODE_GROUPS);
    b.add_u32(
        KEY_TALKER_TEXT_HIDDEN_SIZE,
        variant.talker_text_hidden_size(),
    );

    // Code predictor — identical across every released variant.
    b.add_u32(KEY_CP_HIDDEN_DIM, CP_HIDDEN_DIM);
    b.add_u32(KEY_CP_N_LAYER, CP_N_LAYER);
    b.add_u32(KEY_CP_N_HEAD, CP_N_HEAD);
    b.add_u32(KEY_CP_N_HEAD_KV, CP_N_HEAD_KV);
    b.add_u32(KEY_CP_HEAD_DIM, CP_HEAD_DIM);
    b.add_u32(KEY_CP_FFN_DIM, CP_FFN_DIM);
    b.add_u32(KEY_CP_VOCAB_SIZE, CP_VOCAB_SIZE);
    b.add_f32(KEY_CP_ROPE_BASE, CP_ROPE_BASE);
    b.add_f32(KEY_CP_RMS_NORM_EPS, CP_RMS_NORM_EPS);
    b.add_u32(KEY_CP_NUM_CODE_GROUPS, CP_NUM_CODE_GROUPS);
}

fn actual_manifest(st: &SafetensorsFile) -> BTreeMap<String, Vec<u64>> {
    st.tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect()
}

/// Auto-detects only the two 0.6B releases accepted by the generic
/// `ModelKind::Qwen3Tts` route. The two official manifests differ by the 76
/// Base-only speaker-encoder tensors, so no filename or user assertion is
/// trusted for this decision.
fn detect_0_6b_variant(st: &SafetensorsFile) -> Result<Qwen3TtsVariant, ConvertError> {
    let actual = actual_manifest(st);
    for variant in [
        Qwen3TtsVariant::_0_6B_Base,
        Qwen3TtsVariant::_0_6B_CustomVoice,
    ] {
        if actual == expected_manifest(variant) {
            return Ok(variant);
        }
    }
    Err(ConvertError::Parse(format!(
        "qwen3-tts: generic 0.6B route requires the exact official 478-tensor Base or 402-tensor CustomVoice manifest; found {} tensors. Select an explicit 1.7B model kind for a 1.7B checkpoint",
        actual.len()
    )))
}

fn validate_checkpoint(st: &SafetensorsFile, variant: Qwen3TtsVariant) -> Result<(), ConvertError> {
    let expected = expected_manifest(variant);
    debug_assert_eq!(expected.len(), variant.expected_tensor_count());
    let actual = actual_manifest(st);
    if actual == expected {
        return Ok(());
    }
    let missing = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .take(6)
        .collect::<Vec<_>>();
    let extra = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .take(6)
        .collect::<Vec<_>>();
    let wrong_shape = expected
        .iter()
        .filter_map(|(name, shape)| {
            actual
                .get(name)
                .filter(|actual_shape| *actual_shape != shape)
                .map(|actual_shape| (name, actual_shape, shape))
        })
        .take(6)
        .collect::<Vec<_>>();
    Err(ConvertError::Parse(format!(
        "qwen3-tts {} checkpoint manifest mismatch: expected {} tensors, found {}; missing={missing:?}, extra={extra:?}, wrong_shape={wrong_shape:?}",
        variant.name(),
        expected.len(),
        actual.len()
    )))
}

fn expected_manifest(variant: Qwen3TtsVariant) -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    if variant.has_speaker_encoder() {
        add_speaker_manifest(&mut out, variant.speaker_embed_dim() as u64);
    }

    let talker_hidden = variant.talker_hidden_dim() as u64;
    let talker_ffn = variant.talker_ffn_dim() as u64;
    add_stack_manifest(
        &mut out,
        "talker.model",
        talker_hidden,
        TALKER_N_LAYER as usize,
        talker_ffn,
    );
    out.insert(
        "talker.model.text_embedding.weight".into(),
        vec![
            TALKER_TEXT_VOCAB_SIZE as u64,
            TALKER_TEXT_HIDDEN_SIZE as u64,
        ],
    );
    out.insert(
        "talker.model.codec_embedding.weight".into(),
        vec![TALKER_VOCAB_SIZE as u64, talker_hidden],
    );
    out.insert("talker.model.norm.weight".into(), vec![talker_hidden]);
    out.insert(
        "talker.text_projection.linear_fc1.weight".into(),
        vec![
            TALKER_TEXT_HIDDEN_SIZE as u64,
            TALKER_TEXT_HIDDEN_SIZE as u64,
        ],
    );
    out.insert(
        "talker.text_projection.linear_fc1.bias".into(),
        vec![TALKER_TEXT_HIDDEN_SIZE as u64],
    );
    out.insert(
        "talker.text_projection.linear_fc2.weight".into(),
        vec![talker_hidden, TALKER_TEXT_HIDDEN_SIZE as u64],
    );
    out.insert(
        "talker.text_projection.linear_fc2.bias".into(),
        vec![talker_hidden],
    );
    out.insert(
        "talker.codec_head.weight".into(),
        vec![TALKER_VOCAB_SIZE as u64, talker_hidden],
    );

    add_stack_manifest(
        &mut out,
        "talker.code_predictor.model",
        CP_HIDDEN_DIM as u64,
        CP_N_LAYER as usize,
        CP_FFN_DIM as u64,
    );
    out.insert(
        "talker.code_predictor.model.norm.weight".into(),
        vec![CP_HIDDEN_DIM as u64],
    );
    for group in 0..CP_NUM_CODE_GROUPS.saturating_sub(1) as usize {
        // The official code constructs these embeddings with
        // `embedding_dim=talker_config.hidden_size`, not CP hidden size.
        out.insert(
            format!("talker.code_predictor.model.codec_embedding.{group}.weight"),
            vec![CP_VOCAB_SIZE as u64, talker_hidden],
        );
        out.insert(
            format!("talker.code_predictor.lm_head.{group}.weight"),
            vec![CP_VOCAB_SIZE as u64, CP_HIDDEN_DIM as u64],
        );
    }
    if talker_hidden != CP_HIDDEN_DIM as u64 {
        out.insert(
            "talker.code_predictor.small_to_mtp_projection.weight".into(),
            vec![CP_HIDDEN_DIM as u64, talker_hidden],
        );
        out.insert(
            "talker.code_predictor.small_to_mtp_projection.bias".into(),
            vec![CP_HIDDEN_DIM as u64],
        );
    }
    out
}

fn add_stack_manifest(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    hidden: u64,
    layers: usize,
    ffn: u64,
) {
    let query = TALKER_N_HEAD as u64 * TALKER_HEAD_DIM as u64;
    let key_value = TALKER_N_HEAD_KV as u64 * TALKER_HEAD_DIM as u64;
    for layer in 0..layers {
        let prefix = format!("{prefix}.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.q_norm.weight", vec![TALKER_HEAD_DIM as u64]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.k_norm.weight", vec![TALKER_HEAD_DIM as u64]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("mlp.gate_proj.weight", vec![ffn, hidden]),
            ("mlp.up_proj.weight", vec![ffn, hidden]),
            ("mlp.down_proj.weight", vec![hidden, ffn]),
        ] {
            out.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
}

fn add_speaker_manifest(out: &mut BTreeMap<String, Vec<u64>>, embedding_dim: u64) {
    out.insert(
        "speaker_encoder.blocks.0.conv.weight".into(),
        vec![512, 128, 5],
    );
    out.insert("speaker_encoder.blocks.0.conv.bias".into(), vec![512]);
    for block in 1..=3 {
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn1.conv.weight"),
            vec![512, 512, 1],
        );
        out.insert(
            format!("speaker_encoder.blocks.{block}.tdnn1.conv.bias"),
            vec![512],
        );
        for sub in 0..7 {
            out.insert(
                format!("speaker_encoder.blocks.{block}.res2net_block.blocks.{sub}.conv.weight"),
                vec![64, 64, 3],
            );
            out.insert(
                format!("speaker_encoder.blocks.{block}.res2net_block.blocks.{sub}.conv.bias"),
                vec![64],
            );
        }
        for (suffix, shape) in [
            ("tdnn2.conv.weight", vec![512, 512, 1]),
            ("tdnn2.conv.bias", vec![512]),
            ("se_block.conv1.weight", vec![128, 512, 1]),
            ("se_block.conv1.bias", vec![128]),
            ("se_block.conv2.weight", vec![512, 128, 1]),
            ("se_block.conv2.bias", vec![512]),
        ] {
            out.insert(format!("speaker_encoder.blocks.{block}.{suffix}"), shape);
        }
    }
    for (name, shape) in [
        ("speaker_encoder.mfa.conv.weight", vec![1536, 1536, 1]),
        ("speaker_encoder.mfa.conv.bias", vec![1536]),
        ("speaker_encoder.asp.tdnn.conv.weight", vec![128, 4608, 1]),
        ("speaker_encoder.asp.tdnn.conv.bias", vec![128]),
        ("speaker_encoder.asp.conv.weight", vec![1536, 128, 1]),
        ("speaker_encoder.asp.conv.bias", vec![1536]),
        ("speaker_encoder.fc.weight", vec![embedding_dim, 3072, 1]),
        ("speaker_encoder.fc.bias", vec![embedding_dim]),
    ] {
        out.insert(name.into(), shape);
    }
}

#[cfg(test)]
fn convert_variant_fixture(
    bytes: Vec<u8>,
    variant: Qwen3TtsVariant,
) -> Result<(GgufBuilder, Qwen3TtsReport), ConvertError> {
    convert_parsed(SafetensorsFile::parse(bytes)?, variant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the
        // report counts a non-zero write. The tensor name mirrors an
        // upstream Qwen3-TTS scaffold name.
        let header =
            r#"{"talker.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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
    /// 2 bytes = 12 bytes).
    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header =
            r#"{"talker.embed_tokens.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn scratch_directory(label: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-qwen3-tts-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir(&path).expect("create scratch directory");
        path
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
        // sole handshake with `vokra-models::qwen3_tts::EXPECTED_ARCH`.
        assert_eq!(ARCH, "qwen3_tts");
    }

    #[test]
    fn arch_is_distinct_from_qwen_family_siblings() {
        // Qwen3-TTS shares the Qwen family with CosyVoice2/3 but is
        // codec-LM (qwen3_tts_codec terminal), not vocoder-LM (HiFTChain
        // terminal). Silently sharing an arch tag would mis-route runtime
        // dispatch.
        assert_ne!(ARCH, "cosyvoice2");
        assert_ne!(ARCH, "cosyvoice3");
    }

    #[test]
    fn name_string_matches_hf_release() {
        assert_eq!(NAME, "qwen3-tts-12hz-0.6b-base");
    }

    /// The transcribed constants must equal the primary-source values
    /// (config.json / README.md) — changing any of these silently
    /// mis-shapes the Qwen3 backbone / codec handshake.
    #[test]
    fn transcribed_constants_match_primary_source() {
        // Speaker encoder + sample rate (README.md).
        assert_eq!(QWEN3_TTS_SAMPLE_RATE, 24_000);
        assert_eq!(QWEN3_TTS_0_6B_SPEAKER_EMBED_DIM, 1024);
        assert_eq!(QWEN3_TTS_1_7B_SPEAKER_EMBED_DIM, 2048);

        // Talker (config.json.talker.*).
        assert_eq!(TALKER_HIDDEN_DIM, 1024);
        assert_eq!(TALKER_N_LAYER, 28);
        assert_eq!(TALKER_N_HEAD, 16);
        assert_eq!(TALKER_N_HEAD_KV, 8);
        assert_eq!(TALKER_HEAD_DIM, 128);
        assert_eq!(TALKER_FFN_DIM, 3072);
        assert_eq!(TALKER_VOCAB_SIZE, 3072);
        assert_eq!(TALKER_TEXT_VOCAB_SIZE, 151_936);
        assert_eq!(TALKER_MAX_POSITIONS, 32_768);
        assert!((TALKER_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
        assert!((TALKER_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
        assert_eq!(TALKER_POS_ID_PER_SEC, 13);
        assert_eq!(TALKER_NUM_CODE_GROUPS, 16);
        assert_eq!(TALKER_TEXT_HIDDEN_SIZE, 2048);

        // Code predictor (config.json.code_predictor.*).
        assert_eq!(CP_HIDDEN_DIM, 1024);
        assert_eq!(CP_N_LAYER, 5);
        assert_eq!(CP_N_HEAD, 16);
        assert_eq!(CP_N_HEAD_KV, 8);
        assert_eq!(CP_HEAD_DIM, 128);
        assert_eq!(CP_FFN_DIM, 3072);
        assert_eq!(CP_VOCAB_SIZE, 2048);
        assert!((CP_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
        assert!((CP_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
        assert_eq!(CP_NUM_CODE_GROUPS, 16);

        assert_eq!(MODEL_FAMILY, "qwen3");

        // Compile-time algebra: Qwen3 GQA + RoPE + codec handshake pins.
        const _: () = {
            // GQA (both stacks)
            assert!(TALKER_N_HEAD % TALKER_N_HEAD_KV == 0);
            assert!(CP_N_HEAD % CP_N_HEAD_KV == 0);
            // RoPE evenness
            assert!(TALKER_HEAD_DIM % 2 == 0);
            assert!(CP_HEAD_DIM % 2 == 0);
            // Codec handshake — talker slots = code predictor slots.
            assert!(TALKER_NUM_CODE_GROUPS == CP_NUM_CODE_GROUPS);
            // Semantic (talker) vocab >= acoustic (code predictor) vocab.
            assert!(TALKER_VOCAB_SIZE >= CP_VOCAB_SIZE);
        };
    }

    /// The Qwen3-TTS codec handshake must match
    /// `vokra_ops::qwen3_tts_codec::Qwen3TtsCodecConfig::qwen3_tts_12hz`
    /// on `num_quantizers`. Drifting the two would drop or duplicate
    /// codebook rows silently.
    #[test]
    fn num_code_groups_matches_shared_codec_seam() {
        let codec = vokra_ops::qwen3_tts_codec::Qwen3TtsCodecConfig::qwen3_tts_12hz();
        assert_eq!(TALKER_NUM_CODE_GROUPS as usize, codec.num_quantizers);
        assert_eq!(CP_NUM_CODE_GROUPS as usize, codec.num_quantizers);
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) =
            convert_variant_fixture(minimal_safetensors_one_f32(), Qwen3TtsVariant::_0_6B_Base)
                .expect("convert fixture");
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
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_FAMILY).and_then(|v| v.as_str()),
            Some(MODEL_FAMILY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|value| value.as_str()),
            Some(Qwen3TtsVariant::_0_6B_Base.upstream_hf())
        );
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(Qwen3TtsVariant::_0_6B_Base.source_revision())
        );

        // Every transcribed U32 hparam round-trips verbatim under the
        // `vokra.qwen3_tts.*` prefix.
        for (key, want) in [
            (KEY_SAMPLE_RATE, QWEN3_TTS_SAMPLE_RATE),
            (KEY_SPEAKER_EMBED_DIM, QWEN3_TTS_0_6B_SPEAKER_EMBED_DIM),
            (KEY_TALKER_HIDDEN_DIM, TALKER_HIDDEN_DIM),
            (KEY_TALKER_N_LAYER, TALKER_N_LAYER),
            (KEY_TALKER_N_HEAD, TALKER_N_HEAD),
            (KEY_TALKER_N_HEAD_KV, TALKER_N_HEAD_KV),
            (KEY_TALKER_HEAD_DIM, TALKER_HEAD_DIM),
            (KEY_TALKER_FFN_DIM, TALKER_FFN_DIM),
            (KEY_TALKER_VOCAB_SIZE, TALKER_VOCAB_SIZE),
            (KEY_TALKER_TEXT_VOCAB_SIZE, TALKER_TEXT_VOCAB_SIZE),
            (KEY_TALKER_MAX_POSITIONS, TALKER_MAX_POSITIONS),
            (KEY_TALKER_POS_ID_PER_SEC, TALKER_POS_ID_PER_SEC),
            (KEY_TALKER_NUM_CODE_GROUPS, TALKER_NUM_CODE_GROUPS),
            (KEY_TALKER_TEXT_HIDDEN_SIZE, TALKER_TEXT_HIDDEN_SIZE),
            (KEY_CP_HIDDEN_DIM, CP_HIDDEN_DIM),
            (KEY_CP_N_LAYER, CP_N_LAYER),
            (KEY_CP_N_HEAD, CP_N_HEAD),
            (KEY_CP_N_HEAD_KV, CP_N_HEAD_KV),
            (KEY_CP_HEAD_DIM, CP_HEAD_DIM),
            (KEY_CP_FFN_DIM, CP_FFN_DIM),
            (KEY_CP_VOCAB_SIZE, CP_VOCAB_SIZE),
            (KEY_CP_NUM_CODE_GROUPS, CP_NUM_CODE_GROUPS),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // F32 norm / RoPE constants round-trip too.
        assert!((get_f32(&file, KEY_TALKER_ROPE_BASE) - TALKER_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_TALKER_RMS_NORM_EPS) - TALKER_RMS_NORM_EPS).abs() < 1e-12);
        assert!((get_f32(&file, KEY_CP_ROPE_BASE) - CP_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_CP_RMS_NORM_EPS) - CP_RMS_NORM_EPS).abs() < 1e-12);

        // Provenance: apache-2.0 permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn release_assets_are_embedded_as_one_complete_group() {
        let (mut builder, _) =
            convert_variant_fixture(minimal_safetensors_one_f32(), Qwen3TtsVariant::_0_6B_Base)
                .expect("convert fixture");
        let before = builder.metadata_count();
        let assets = ReleaseAssets {
            config: b"config".to_vec(),
            vocab: b"vocab".to_vec(),
            merges: b"merges".to_vec(),
            tokenizer_config: b"tokenizer".to_vec(),
            generation_config: b"generation".to_vec(),
        };
        assets.embed(&mut builder);
        assert_eq!(builder.metadata_count(), before + 5);
        let file =
            GgufFile::parse(builder.to_bytes().expect("serialize assets")).expect("parse assets");
        for (key, expected) in [
            (KEY_CONFIG_JSON, b"config".as_slice()),
            (KEY_TOKENIZER_VOCAB, b"vocab".as_slice()),
            (KEY_TOKENIZER_MERGES, b"merges".as_slice()),
            (KEY_TOKENIZER_CONFIG, b"tokenizer".as_slice()),
            (KEY_GENERATION_CONFIG, b"generation".as_slice()),
        ] {
            let GgufMetadataValue::Array(array) = file.get(key).expect("embedded asset") else {
                panic!("{key} must be an array");
            };
            let actual = array
                .values
                .iter()
                .map(|value| match value {
                    GgufMetadataValue::U8(byte) => *byte,
                    other => panic!("{key} contains non-U8 {other:?}"),
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn sidecar_reader_rejects_size_and_hash_drift() {
        let directory = scratch_directory("sidecars");
        let path = directory.join("tiny.txt");
        let variant = Qwen3TtsVariant::_1_7B_VoiceDesign;
        let spec = ExactSidecar {
            name: "tiny.txt",
            bytes: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        std::fs::write(&path, b"abc").expect("write exact sidecar");
        assert_eq!(
            read_exact_sidecar(&directory, spec, variant).expect("exact sidecar"),
            b"abc"
        );

        std::fs::write(&path, b"ab").expect("write short sidecar");
        assert!(
            read_exact_sidecar(&directory, spec, variant)
                .expect_err("size drift")
                .to_string()
                .contains("expected exactly 3")
        );
        std::fs::write(&path, b"abd").expect("write hash drift sidecar");
        assert!(
            read_exact_sidecar(&directory, spec, variant)
                .expect_err("hash drift")
                .to_string()
                .contains("SHA-256")
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        // Empty safetensors → the runtime's `Qwen3TtsWeights::from_gguf`
        // would fail loudly at bind time, but the converter itself
        // succeeds and reports the situation so the operator sees it now.
        let (_, report) = convert_variant_fixture(
            minimal_safetensors_no_tensors(),
            Qwen3TtsVariant::_0_6B_Base,
        )
        .expect("convert fixture");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union
    /// match arm.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) =
            convert_variant_fixture(minimal_safetensors_one_f16(), Qwen3TtsVariant::_0_6B_Base)
                .expect("convert fixture");
        assert_eq!(report.written, 1, "F16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "F16 must not land in the skipped counter"
        );

        // The tensor survives the round trip under its upstream name and
        // preserves its F16 dtype (payload is 12 bytes = 6 × F16).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("talker.embed_tokens.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Mirrors [`f16_tensor_passes_through_verbatim`] for BF16 — the
    /// upstream serving format for Qwen3-TTS-0.6B. Per the ADR
    /// (docs/adr/qwen3-tts-bf16.md, Accepted 2026-07-25, strategy
    /// A_passthrough), BF16 must reach the pass-through arm verbatim
    /// (emitted as GGUF type 30 = `GgmlType::BF16`, no convert-time
    /// widening — the runtime widens BF16 → f32 losslessly at load via
    /// the single choke point `vokra-core::gguf::quant::decode_bf16`).
    /// Rewrite of the M4-06 posture pin
    /// `bf16_tensor_is_counted_as_skipped_non_float` — the ADR's
    /// red-line demands the symmetric rewrite so a latent silent-widen
    /// cannot slip in undetected.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Build a BF16 payload with known non-zero bit patterns so a
        // subsequent byte-identity assert catches any silent widen /
        // downcast attempt (the raw zeroed payload of the shared
        // fixture would round-trip trivially through F32/F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&bf16);

        let (builder, report) =
            convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base).expect("convert fixture");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (ADR A_passthrough)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter after ADR A_passthrough"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );
        // Loud-silence check for FR-EX-08: the zero-float note is a
        // false-positive here because BF16 IS a float.
        assert!(
            !report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16 pass-through must not emit the zero-float note: {:?}",
            report.notes
        );

        // Round-trip through the GGUF: dtype preserved, payload
        // byte-identical (Moshi's assert_eq!(info.dtype, GgmlType::BF16,
        // "no convert-time widening") posture — the safetensors.rs:728-738
        // pin pattern).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("talker.embed_tokens.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
    }

    /// Pins the "mixed-dtype loops don't collapse to one arm" contract:
    /// a BF16 tensor and an F32 tensor in the same safetensors input
    /// must **both** pass through with their dtypes preserved. Guards
    /// against a regression where a naive `if bf16 { ... } else` refactor
    /// would only emit one branch of the match.
    #[test]
    fn bf16_and_f32_mixed_pass_through_side_by_side() {
        // Header declares tensors in order:
        //   talker.embed_tokens.weight — BF16, [2,3] → 12 bytes @ [0..12)
        //   talker.other.weight        — F32,  [1,2] →  8 bytes @ [12..20)
        // Safetensors sorts entries by data_offsets lexicographically
        // (the reader tolerates any JSON order), so the payload appends
        // BF16 first, then F32.
        let bf16_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let header = r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"talker.other.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[12,20]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&bf16);
        input.extend_from_slice(&f32_bytes);

        let (builder, report) =
            convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base).expect("convert fixture");
        assert_eq!(
            report.written, 2,
            "both BF16 and F32 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "only the BF16 tensor increments the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Both tensors survive the round-trip with their upstream names
        // and dtypes preserved.
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let bf16_info = file
            .tensor_info("talker.embed_tokens.weight")
            .expect("BF16 tensor present");
        assert_eq!(bf16_info.dtype, GgmlType::BF16, "BF16 stays BF16");
        assert_eq!(file.tensor_bytes(bf16_info), bf16.as_slice());

        let f32_info = file
            .tensor_info("talker.other.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());
    }

    /// Regression guard for the additive [`Qwen3TtsReport::bf16_passthrough`]
    /// field: on an F32-only input the new counter defaults to `0` (via
    /// the `#[derive(Default)]` on the report), proving the additive
    /// field does not shift or contaminate the other counters.
    #[test]
    fn bf16_passthrough_report_field_is_additive_default_zero() {
        let (_, report) =
            convert_variant_fixture(minimal_safetensors_one_f32(), Qwen3TtsVariant::_0_6B_Base)
                .expect("convert fixture");
        assert_eq!(report.written, 1, "F32 tensor still counted verbatim");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only input must leave the BF16 counter at the Default 0"
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation.
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer.
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

    /// Every `vokra.qwen3_tts.*` key uses the same prefix — a regression
    /// where a key crossed into another model's namespace (e.g.
    /// `vokra.cosyvoice3.*`) would still round-trip in isolation but
    /// would misroute at the runtime dispatch layer.
    #[test]
    fn every_metadata_key_carries_the_qwen3_tts_prefix() {
        for key in [
            KEY_SAMPLE_RATE,
            KEY_SPEAKER_EMBED_DIM,
            KEY_TALKER_HIDDEN_DIM,
            KEY_TALKER_N_LAYER,
            KEY_TALKER_N_HEAD,
            KEY_TALKER_N_HEAD_KV,
            KEY_TALKER_HEAD_DIM,
            KEY_TALKER_FFN_DIM,
            KEY_TALKER_VOCAB_SIZE,
            KEY_TALKER_TEXT_VOCAB_SIZE,
            KEY_TALKER_MAX_POSITIONS,
            KEY_TALKER_ROPE_BASE,
            KEY_TALKER_RMS_NORM_EPS,
            KEY_TALKER_POS_ID_PER_SEC,
            KEY_TALKER_NUM_CODE_GROUPS,
            KEY_TALKER_TEXT_HIDDEN_SIZE,
            KEY_CP_HIDDEN_DIM,
            KEY_CP_N_LAYER,
            KEY_CP_N_HEAD,
            KEY_CP_N_HEAD_KV,
            KEY_CP_HEAD_DIM,
            KEY_CP_FFN_DIM,
            KEY_CP_VOCAB_SIZE,
            KEY_CP_ROPE_BASE,
            KEY_CP_RMS_NORM_EPS,
            KEY_CP_NUM_CODE_GROUPS,
            KEY_MODEL_FAMILY,
        ] {
            assert!(
                key.starts_with("vokra.qwen3_tts."),
                "{key} must live under the vokra.qwen3_tts.* prefix"
            );
        }
    }

    // ─── Adversarial BF16 coverage ────────────────────────────────────────
    //
    // The ADR (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough) rests on
    // three properties of the runtime widening `bits << 16`:
    //   (a) every BF16 bit pattern round-trips as a byte-identical BF16
    //       payload through the converter (no silent widen / downcast);
    //   (b) `decode_bf16` widens **every** IEEE-754 corner (±Inf, quiet /
    //       signaling NaN, subnormals) to an f32 whose bit pattern equals
    //       the BF16 pattern shifted left 16 (i.e. mathematically exact);
    //   (c) the safetensors parser rejects malformed BF16 payloads *loudly*
    //       (FR-EX-08 no silent truncation) — odd byte counts, byte spans
    //       that disagree with `shape × 2`, empty tensors, and payloads that
    //       run past the data region all fail with a parse error rather
    //       than passing a truncated / wrong-shape tensor to the runtime.
    //
    // These tests pin (a), (b) and (c) end-to-end through `convert()` (not
    // through `decode_bf16` directly — that function is private to
    // `vokra-core::gguf::quant`, so the round-trip happens through
    // `GgufFile::tensor_f32`, which routes through `dequantize` →
    // `decode_bf16` per the ADR's single-choke-point invariant).

    /// Builds a synthetic single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Panics if `bf16_bytes.len()` disagrees
    /// with `shape × 2`.
    fn safetensors_one_bf16(shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"talker.embed_tokens.weight":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    /// Encodes 16-bit BF16 patterns as little-endian bytes (pass-through
    /// helper; deliberately does *not* touch f32 so a converter regression
    /// that silently widens can't sneak past by matching the same "value").
    fn bf16_pattern_bytes(patterns: &[u16]) -> Vec<u8> {
        patterns.iter().flat_map(|p| p.to_le_bytes()).collect()
    }

    /// Attack 1 — BF16 quiet NaN (0x7FC0) and signalling NaN (0x7FA0) both
    /// survive the pass-through as byte-identical bytes AND decode to
    /// F32 patterns whose bits equal `pattern << 16` (which is `is_nan()`
    /// by construction for any BF16 with `exp = 0xFF` and non-zero
    /// mantissa). A silent BF16 → F32 widen at convert time would still
    /// round-trip *values* (NaN in → NaN out), so this test asserts on the
    /// dtype (`GgmlType::BF16`) AND the raw bytes AND the decoded f32 bit
    /// pattern — three concentric fences.
    #[test]
    fn bf16_nan_bit_patterns_survive_pass_through_and_decode_as_nan() {
        // 0x7FC0: sign 0, exp 0xFF, mantissa 0b1000000 (MSB set) → quiet NaN.
        // 0x7FA0: sign 0, exp 0xFF, mantissa 0b0100000 (bit 5 set)  → signalling NaN.
        // 0xFFC1: sign 1, exp 0xFF, mantissa non-zero               → quiet NaN with payload.
        let patterns: [u16; 3] = [0x7FC0, 0x7FA0, 0xFFC1];
        let bytes = bf16_pattern_bytes(&patterns);
        let input = safetensors_one_bf16(&[3], &bytes);

        let (builder, report) =
            convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base).expect("convert fixture");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("talker.embed_tokens.weight").unwrap();
        // Fence 1: dtype stayed BF16 (no convert-time widening).
        assert_eq!(info.dtype, GgmlType::BF16);
        // Fence 2: raw bytes byte-identical to input.
        assert_eq!(file.tensor_bytes(info), bytes.as_slice());
        // Fence 3: `decode_bf16` (via tensor_f32) widens to the exact
        // f32 pattern `bits << 16`, and every result is is_nan().
        let decoded = file.tensor_f32("talker.embed_tokens.weight").unwrap();
        assert_eq!(decoded.len(), 3);
        for (i, (dec, pat)) in decoded.iter().zip(patterns.iter()).enumerate() {
            let want_bits = (u32::from(*pat)) << 16;
            assert_eq!(
                dec.to_bits(),
                want_bits,
                "element {i}: BF16 0x{pat:04X} must decode to f32 bits 0x{want_bits:08X}"
            );
            assert!(
                dec.is_nan(),
                "element {i}: BF16 0x{pat:04X} widened to non-NaN f32 (bits 0x{:08X})",
                dec.to_bits()
            );
        }
    }

    /// Attack 2 — BF16 ±Infinity (0x7F80 / 0xFF80) survive as bytes AND
    /// decode to `f32::INFINITY` / `f32::NEG_INFINITY` exactly.
    #[test]
    fn bf16_positive_and_negative_infinity_survive_and_decode_correctly() {
        // 0x7F80: sign 0, exp 0xFF, mantissa 0 → +∞.
        // 0xFF80: sign 1, exp 0xFF, mantissa 0 → -∞.
        let patterns: [u16; 2] = [0x7F80, 0xFF80];
        let bytes = bf16_pattern_bytes(&patterns);
        let input = safetensors_one_bf16(&[2], &bytes);

        let (builder, report) =
            convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base).expect("convert fixture");
        assert_eq!(report.bf16_passthrough, 1);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("talker.embed_tokens.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bytes.as_slice());
        let decoded = file.tensor_f32("talker.embed_tokens.weight").unwrap();
        assert_eq!(decoded, vec![f32::INFINITY, f32::NEG_INFINITY]);
        // Bit-exact: `bits << 16` for 0x7F80 is 0x7F800000 = f32::INFINITY.
        assert_eq!(decoded[0].to_bits(), f32::INFINITY.to_bits());
        assert_eq!(decoded[1].to_bits(), f32::NEG_INFINITY.to_bits());
    }

    /// Attack 3 — BF16 subnormals (0x0001 = smallest positive subnormal;
    /// 0x0080 = smallest positive normal; the pair covers the subnormal /
    /// normal boundary in both directions). The `bits << 16` widen turns
    /// a BF16 subnormal into an f32 subnormal with the **same mathematical
    /// value** (both formats' subnormal formula reduces to `mantissa ×
    /// 2^-(bias + mantissa_bits)`, and the shift preserves this). Some CPUs
    /// flush subnormals to zero (FTZ / DAZ), so this test also asserts the
    /// decode does NOT flush.
    #[test]
    fn bf16_subnormals_survive_and_decode_without_flush_to_zero() {
        // 0x0001: sign 0, exp 0x00, mantissa 0b0000001 → smallest positive
        //         BF16 subnormal = 2^-133.
        // 0x0080: sign 0, exp 0x01, mantissa 0        → smallest positive
        //         BF16 normal = 2^-126.
        // 0x007F: sign 0, exp 0x00, mantissa 0b1111111 → largest positive
        //         BF16 subnormal = (2^7 - 1) × 2^-133.
        // 0x8001: sign 1, exp 0x00, mantissa 0b0000001 → -smallest subnormal.
        let patterns: [u16; 4] = [0x0001, 0x0080, 0x007F, 0x8001];
        let bytes = bf16_pattern_bytes(&patterns);
        let input = safetensors_one_bf16(&[4], &bytes);

        let (builder, report) =
            convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base).expect("convert fixture");
        assert_eq!(report.bf16_passthrough, 1);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("talker.embed_tokens.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bytes.as_slice());

        let decoded = file.tensor_f32("talker.embed_tokens.weight").unwrap();
        for (i, (dec, pat)) in decoded.iter().zip(patterns.iter()).enumerate() {
            let want_bits = (u32::from(*pat)) << 16;
            assert_eq!(
                dec.to_bits(),
                want_bits,
                "element {i}: BF16 0x{pat:04X} must decode to f32 bits 0x{want_bits:08X} \
                 (subnormals must not be flushed by the widen)"
            );
        }
        // Bit-verified mathematical values.
        assert!(decoded[0] > 0.0, "smallest BF16 subnormal decodes to > 0.0");
        assert_eq!(decoded[0], f32::from_bits(0x0001_0000));
        assert_eq!(decoded[1], f32::from_bits(0x0080_0000)); // 2^-126
        assert_eq!(decoded[3], -f32::from_bits(0x0001_0000));
        // 0x0080 is the smallest positive f32 normal (= f32::MIN_POSITIVE).
        assert_eq!(decoded[1], f32::MIN_POSITIVE);
    }

    /// Attack 4 — a BF16 tensor whose `data_offsets` span is not a multiple
    /// of 2 (odd byte count) must be rejected at parse time, NOT silently
    /// truncated to floor(bytes / 2) elements. Building a header where
    /// shape=[3] BF16 (needs 6 bytes) is declared with a 5-byte data span:
    /// the parser must fail because `end - begin = 5 ≠ 6 = shape × 2`.
    /// The alternative — silently emitting a 2-element BF16 tensor
    /// (chunks_exact discards the odd byte) — would mis-shape the tensor
    /// at runtime, violating FR-EX-08.
    #[test]
    fn bf16_odd_byte_span_is_rejected_at_parse_not_silently_truncated() {
        let header =
            r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[3],"data_offsets":[0,5]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&[0u8; 5]);

        let err = convert(input).expect_err("odd BF16 byte span must not silently truncate");
        match err {
            ConvertError::Parse(msg) => {
                // `SafetensorsError::BadEntry` explicitly names the mismatch;
                // pinning the substring makes silent-truncation regressions
                // (e.g. dropping the byte-span check in parse_header_entries)
                // trip loudly instead of degrading to a mis-shaped tensor.
                assert!(
                    msg.contains("byte span") && msg.contains("does not match shape/dtype"),
                    "expected byte-span mismatch diagnostic, got: {msg}"
                );
            }
            other => panic!("expected ConvertError::Parse, got {other:?}"),
        }
    }

    /// Attack 5 — an empty BF16 tensor (`shape=[0]`, 0 payload bytes)
    /// converts cleanly: it lands on the pass-through arm, increments the
    /// BF16 counter, and round-trips through the GGUF as a zero-byte BF16
    /// tensor. Regression guard for a naive `if bytes.is_empty()` early
    /// return that would misclassify the empty tensor as skipped_non_float.
    #[test]
    fn bf16_empty_tensor_shape_zero_converts_cleanly() {
        let header =
            r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[0],"data_offsets":[0,0]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());

        let (builder, report) = convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base)
            .expect("empty BF16 fixture must convert cleanly");
        assert_eq!(report.written, 1);
        assert_eq!(
            report.bf16_passthrough, 1,
            "empty BF16 must still increment the BF16 counter (it IS a BF16 tensor)"
        );
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("talker.embed_tokens.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![0]);
        assert_eq!(file.tensor_bytes(info).len(), 0);
        // Decode: `dequantize(BF16, &[], 0)` returns Ok(vec![]).
        assert_eq!(
            file.tensor_f32("talker.embed_tokens.weight").unwrap(),
            Vec::<f32>::new()
        );
    }

    /// Attack 6 — a BF16 tensor whose declared byte span is exactly half
    /// of `shape × 2` (100 bytes for [10, 10] instead of 200) must be
    /// rejected at parse time. Mirrors Attack 4 but at a *matching-parity*
    /// scale (both spans are even), so it targets the shape check
    /// specifically rather than any odd-byte heuristic.
    #[test]
    fn bf16_shape_dtype_mismatch_half_size_is_rejected() {
        let header = r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[10,10],"data_offsets":[0,100]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&[0u8; 100]);

        let err = convert(input).expect_err("100 bytes for a 10×10 BF16 tensor must be rejected");
        match err {
            ConvertError::Parse(msg) => {
                assert!(
                    msg.contains("byte span") && msg.contains("100"),
                    "expected byte-span mismatch diagnostic naming the 100 bytes, got: {msg}"
                );
                // The expected size is 200; the diagnostic should say so.
                assert!(
                    msg.contains("200"),
                    "expected diagnostic to mention the 200-byte expected size, got: {msg}"
                );
            }
            other => panic!("expected ConvertError::Parse, got {other:?}"),
        }
    }

    /// Attack 7 — a `[1024, 1024]` BF16 tensor (2 MiB payload) round-trips
    /// byte-identically. Bytes are populated with a deterministic linear
    /// congruential pattern so any single-bit drop / duplication / offset
    /// shift trips loudly. Also exercises the alignment padding in
    /// `GgufBuilder::to_bytes` at a realistic weight-tensor scale.
    #[test]
    fn bf16_large_tensor_1024x1024_round_trips_bit_identically() {
        const ROWS: usize = 1024;
        const COLS: usize = 1024;
        const N_BYTES: usize = ROWS * COLS * 2;
        // Deterministic LCG (Numerical Recipes constants): every byte is a
        // function of its index, so a shift / drop / duplication produces
        // a byte-level diff the assert_eq will report precisely.
        let mut state: u32 = 0xDEAD_BEEF;
        let mut bytes = Vec::with_capacity(N_BYTES);
        for _ in 0..N_BYTES {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            bytes.push((state >> 24) as u8);
        }
        assert_eq!(bytes.len(), N_BYTES);
        let input = safetensors_one_bf16(&[ROWS as u64, COLS as u64], &bytes);

        let (builder, report) = convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base)
            .expect("large BF16 fixture must convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file.tensor_info("talker.embed_tokens.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![ROWS as u64, COLS as u64]);
        assert_eq!(file.tensor_bytes(info).len(), N_BYTES);
        assert_eq!(
            file.tensor_bytes(info),
            bytes.as_slice(),
            "2 MiB BF16 payload must survive round-trip byte-identically"
        );
    }

    /// Attack 8 — report accounting invariant:
    /// `report.written + report.skipped_non_float == n_input_tensors` for
    /// **every** mixed input the safetensors reader will accept. Today the
    /// reader accepts only F32 / F16 / BF16, so this reduces to
    /// `report.written == n_input_tensors` on any well-formed input; the
    /// test pins the invariant so a future dtype (I8 / I32 codebook ids
    /// etc.) that lands on the skipped arm still preserves it, catching
    /// a regression that would double-count or drop a tensor entirely.
    #[test]
    fn report_written_plus_skipped_equals_input_tensor_count() {
        // Three tensors of three different accepted dtypes in one file.
        // Header lists them lexicographically by data_offsets (F32 first,
        // then BF16, then F16 — matches how the payload is laid out below).
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16_bytes: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let header = format!(
            r#"{{"a_f32":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"b_bf16":{{"dtype":"BF16","shape":[2,3],"data_offsets":[{},{}]}},"c_f16":{{"dtype":"F16","shape":[3],"data_offsets":[{},{}]}}}}"#,
            f32_bytes.len(),
            f32_bytes.len(),
            f32_bytes.len() + bf16_bytes.len(),
            f32_bytes.len() + bf16_bytes.len(),
            f32_bytes.len() + bf16_bytes.len() + f16_bytes.len(),
        );
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);
        input.extend_from_slice(&bf16_bytes);
        input.extend_from_slice(&f16_bytes);

        let (_, report) = convert_variant_fixture(input, Qwen3TtsVariant::_0_6B_Base)
            .expect("mixed dtype fixture must convert");
        let n_input_tensors: usize = 3;
        assert_eq!(
            report.written + report.skipped_non_float,
            n_input_tensors,
            "invariant: every input tensor lands on exactly one arm \
             (written={}, skipped_non_float={}, n_input={n_input_tensors})",
            report.written,
            report.skipped_non_float,
        );
        // Sharpen the invariant: today all three dtypes are floats, so
        // written==3 and skipped==0 (breaking down which side of the
        // invariant each tensor lands on).
        assert_eq!(report.written, 3);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "exactly one BF16 tensor in the mixed input"
        );
    }

    // ─── 1.7B variant coverage ────────────────────────────────────────────
    //
    // The 1.7B variants (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` and
    // `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) share IDENTICAL talker +
    // code-predictor axes and differ only in `tts_model_type`
    // (`"custom_voice"` vs `"voice_design"`) and their HF release id +
    // `vokra.model.name` stamp. These tests pin:
    //   (a) the 1.7B talker constants match primary source (verbatim
    //       transcription — CLAUDE.md「ハルシネーション厳禁」);
    //   (b) `Qwen3TtsVariant::name()` stamps the correct HF release name
    //       for each variant;
    //   (c) `convert_variant()` emits the variant-selected talker hidden
    //       + FFN + text_hidden while selecting Base-only speaker metadata;
    //   (d) `convert()` authenticates either exact 0.6B Base or CustomVoice
    //       rather than trusting an alias or filename.

    /// Primary-source pin for the 1.7B talker axes.
    /// Sources fetched 2026-07-30:
    ///   - `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/raw/main/config.json`
    ///   - `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/config.json`
    #[test]
    fn transcribed_1_7b_constants_match_primary_source() {
        // Talker axes that DIFFER from 0.6B.
        assert_eq!(TALKER_1_7B_HIDDEN_DIM, 2048);
        assert_eq!(TALKER_1_7B_FFN_DIM, 6144);
        assert_eq!(TALKER_1_7B_TEXT_HIDDEN_SIZE, 2048);
        // Talker axes that MATCH 0.6B — pinned to guarantee the invariant.
        assert_eq!(TALKER_1_7B_N_LAYER, TALKER_N_LAYER);
        assert_eq!(TALKER_1_7B_N_HEAD, TALKER_N_HEAD);
        assert_eq!(TALKER_1_7B_N_HEAD_KV, TALKER_N_HEAD_KV);
        assert_eq!(TALKER_1_7B_HEAD_DIM, TALKER_HEAD_DIM);
        assert_eq!(TALKER_1_7B_VOCAB_SIZE, TALKER_VOCAB_SIZE);
        assert_eq!(TALKER_1_7B_TEXT_VOCAB_SIZE, TALKER_TEXT_VOCAB_SIZE);
        assert_eq!(TALKER_1_7B_MAX_POSITIONS, TALKER_MAX_POSITIONS);
        assert!((TALKER_1_7B_ROPE_BASE - TALKER_ROPE_BASE).abs() < 1e-3);
        assert!((TALKER_1_7B_RMS_NORM_EPS - TALKER_RMS_NORM_EPS).abs() < 1e-12);
        assert_eq!(TALKER_1_7B_POS_ID_PER_SEC, TALKER_POS_ID_PER_SEC);
        assert_eq!(TALKER_1_7B_NUM_CODE_GROUPS, TALKER_NUM_CODE_GROUPS);

        // Compile-time algebra: GQA + RoPE + codec handshake pins for 1.7B.
        const _: () = {
            assert!(TALKER_1_7B_N_HEAD % TALKER_1_7B_N_HEAD_KV == 0);
            assert!(TALKER_1_7B_HEAD_DIM % 2 == 0);
            assert!(TALKER_1_7B_NUM_CODE_GROUPS == CP_NUM_CODE_GROUPS);
            // Talker vocab (semantic) >= code predictor vocab (acoustic).
            assert!(TALKER_1_7B_VOCAB_SIZE >= CP_VOCAB_SIZE);
        };
    }

    #[test]
    fn variant_name_stamps_hf_release_id() {
        assert_eq!(
            Qwen3TtsVariant::_0_6B_Base.name(),
            "qwen3-tts-12hz-0.6b-base"
        );
        assert_eq!(
            Qwen3TtsVariant::_0_6B_CustomVoice.name(),
            "qwen3-tts-12hz-0.6b-customvoice"
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_Base.name(),
            "qwen3-tts-12hz-1.7b-base"
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_CustomVoice.name(),
            "qwen3-tts-12hz-1.7b-customvoice"
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_VoiceDesign.name(),
            "qwen3-tts-12hz-1.7b-voicedesign"
        );
        // The three 1.7B variants must have PAIRWISE DISTINCT NAME stamps so
        // a downstream that ships them side-by-side can tell them apart.
        assert_ne!(
            Qwen3TtsVariant::_1_7B_Base.name(),
            Qwen3TtsVariant::_1_7B_CustomVoice.name(),
        );
        assert_ne!(
            Qwen3TtsVariant::_1_7B_Base.name(),
            Qwen3TtsVariant::_1_7B_VoiceDesign.name(),
        );
        assert_ne!(
            Qwen3TtsVariant::_1_7B_CustomVoice.name(),
            Qwen3TtsVariant::_1_7B_VoiceDesign.name(),
        );
    }

    #[test]
    fn variant_selectors_return_variant_specific_talker_axes() {
        assert_eq!(
            Qwen3TtsVariant::_0_6B_Base.talker_hidden_dim(),
            TALKER_HIDDEN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_0_6B_CustomVoice.talker_hidden_dim(),
            TALKER_HIDDEN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_Base.talker_hidden_dim(),
            TALKER_1_7B_HIDDEN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_CustomVoice.talker_hidden_dim(),
            TALKER_1_7B_HIDDEN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_VoiceDesign.talker_hidden_dim(),
            TALKER_1_7B_HIDDEN_DIM
        );

        assert_eq!(Qwen3TtsVariant::_0_6B_Base.talker_ffn_dim(), TALKER_FFN_DIM);
        assert_eq!(
            Qwen3TtsVariant::_0_6B_CustomVoice.talker_ffn_dim(),
            TALKER_FFN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_Base.talker_ffn_dim(),
            TALKER_1_7B_FFN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_CustomVoice.talker_ffn_dim(),
            TALKER_1_7B_FFN_DIM
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_VoiceDesign.talker_ffn_dim(),
            TALKER_1_7B_FFN_DIM
        );

        assert_eq!(
            Qwen3TtsVariant::_0_6B_Base.talker_text_hidden_size(),
            TALKER_TEXT_HIDDEN_SIZE
        );
        assert_eq!(
            Qwen3TtsVariant::_0_6B_CustomVoice.talker_text_hidden_size(),
            TALKER_TEXT_HIDDEN_SIZE
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_Base.talker_text_hidden_size(),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_CustomVoice.talker_text_hidden_size(),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
        assert_eq!(
            Qwen3TtsVariant::_1_7B_VoiceDesign.talker_text_hidden_size(),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
    }

    /// The Base metadata path remains stable while production `convert`
    /// performs exact Base/CustomVoice manifest detection first.
    #[test]
    fn base_variant_still_targets_0_6b_base() {
        let (builder, _) =
            convert_variant_fixture(minimal_safetensors_one_f32(), Qwen3TtsVariant::_0_6B_Base)
                .expect("convert fixture");
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
            "the explicit Base fixture must stamp the 0.6B-Base name"
        );
        assert_eq!(get_u32(&file, KEY_TALKER_HIDDEN_DIM), TALKER_HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_TALKER_FFN_DIM), TALKER_FFN_DIM);
    }

    #[test]
    fn production_entry_points_reject_shape_fixtures() {
        let input = minimal_safetensors_one_f32();
        let generic = convert(input.clone()).expect_err("generic route must authenticate manifest");
        assert!(
            generic
                .to_string()
                .contains("478-tensor Base or 402-tensor CustomVoice"),
            "unexpected generic error: {generic}"
        );

        let selected = convert_variant(input, Qwen3TtsVariant::_1_7B_Base)
            .expect_err("selected route must authenticate manifest");
        assert!(
            selected
                .to_string()
                .contains("expected 480 tensors, found 1"),
            "unexpected selected error: {selected}"
        );
    }

    fn round_trip_variant(variant: Qwen3TtsVariant) -> GgufFile {
        let (builder, report) = convert_variant_fixture(minimal_safetensors_one_f32(), variant)
            .expect("convert fixture");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        let out = builder.to_bytes().expect("serialize");
        GgufFile::parse(out).expect("parse")
    }

    #[test]
    fn official_variant_manifests_match_live_header_contracts() {
        let base_06 = expected_manifest(Qwen3TtsVariant::_0_6B_Base);
        let custom_06 = expected_manifest(Qwen3TtsVariant::_0_6B_CustomVoice);
        let base_17 = expected_manifest(Qwen3TtsVariant::_1_7B_Base);
        let custom_17 = expected_manifest(Qwen3TtsVariant::_1_7B_CustomVoice);
        let design_17 = expected_manifest(Qwen3TtsVariant::_1_7B_VoiceDesign);

        assert_eq!(base_06.len(), 478);
        assert_eq!(custom_06.len(), 402);
        assert_eq!(base_17.len(), 480);
        assert_eq!(custom_17.len(), 404);
        assert_eq!(design_17, custom_17);
        assert_eq!(base_17["speaker_encoder.fc.weight"], [2048, 3072, 1]);
        assert!(!custom_17.contains_key("speaker_encoder.fc.weight"));
        assert_eq!(
            base_17["talker.code_predictor.model.codec_embedding.0.weight"],
            [2048, 2048]
        );
        assert_eq!(
            base_17["talker.code_predictor.small_to_mtp_projection.weight"],
            [1024, 2048]
        );
        assert_eq!(
            base_17["talker.code_predictor.small_to_mtp_projection.bias"],
            [1024]
        );
    }

    #[test]
    fn customvoice_0_6b_stamps_faithful_variant_metadata() {
        let file = round_trip_variant(Qwen3TtsVariant::_0_6B_CustomVoice);
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME)
                .and_then(|value| value.as_str()),
            Some(NAME_0_6B_CUSTOM_VOICE)
        );
        assert_eq!(get_u32(&file, KEY_SPEAKER_EMBED_DIM), 0);
        assert_eq!(
            file.get(KEY_HAS_SPEAKER_ENCODER)
                .and_then(|value| value.as_bool()),
            Some(false)
        );
        assert_eq!(
            file.get(KEY_TTS_MODEL_SIZE)
                .and_then(|value| value.as_str()),
            Some("0b6")
        );
        assert_eq!(
            file.get(KEY_TTS_MODEL_TYPE)
                .and_then(|value| value.as_str()),
            Some("custom_voice")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|value| value.as_str()),
            Some(NAME_0_6B_CUSTOM_VOICE)
        );
    }

    /// Mirror of the CustomVoice / VoiceDesign tests below for the
    /// `_1_7B_Base` variant added 2026-08-01 (Wave 4). Pins the same
    /// three-fence contract: arch tag is Qwen3-TTS, name stamp is the
    /// un-fine-tuned 1.7B-Base id, talker axes are the widened 1.7B set
    /// while every invariant axis (code predictor, sample rate, speaker
    /// embed, RoPE, RMSNorm, codec handshake) matches the 0.6B sibling.
    /// Provenance is apache-2.0 permissive under the 1.7B-Base NAME
    /// (distinct from every other variant).
    #[test]
    fn convert_variant_1_7b_base_emits_the_1_7b_axes_and_base_name() {
        let file = round_trip_variant(Qwen3TtsVariant::_1_7B_Base);
        // Arch tag never changes with variant — every variant is Qwen3-TTS.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // Variant-selected name.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_1_7B_BASE)
        );
        // Variant-selected talker axes — 1.7B widened set.
        assert_eq!(
            get_u32(&file, KEY_TALKER_HIDDEN_DIM),
            TALKER_1_7B_HIDDEN_DIM
        );
        assert_eq!(get_u32(&file, KEY_TALKER_FFN_DIM), TALKER_1_7B_FFN_DIM);
        assert_eq!(
            get_u32(&file, KEY_TALKER_TEXT_HIDDEN_SIZE),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
        // Invariant axes: shared with 0.6B.
        assert_eq!(get_u32(&file, KEY_TALKER_N_LAYER), TALKER_N_LAYER);
        assert_eq!(get_u32(&file, KEY_TALKER_N_HEAD), TALKER_N_HEAD);
        assert_eq!(get_u32(&file, KEY_TALKER_N_HEAD_KV), TALKER_N_HEAD_KV);
        assert_eq!(get_u32(&file, KEY_TALKER_HEAD_DIM), TALKER_HEAD_DIM);
        assert_eq!(get_u32(&file, KEY_TALKER_VOCAB_SIZE), TALKER_VOCAB_SIZE);
        assert_eq!(
            get_u32(&file, KEY_TALKER_TEXT_VOCAB_SIZE),
            TALKER_TEXT_VOCAB_SIZE
        );
        assert_eq!(
            get_u32(&file, KEY_TALKER_MAX_POSITIONS),
            TALKER_MAX_POSITIONS
        );
        assert!((get_f32(&file, KEY_TALKER_ROPE_BASE) - TALKER_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_TALKER_RMS_NORM_EPS) - TALKER_RMS_NORM_EPS).abs() < 1e-12);
        assert_eq!(
            get_u32(&file, KEY_TALKER_POS_ID_PER_SEC),
            TALKER_POS_ID_PER_SEC
        );
        assert_eq!(
            get_u32(&file, KEY_TALKER_NUM_CODE_GROUPS),
            TALKER_NUM_CODE_GROUPS
        );
        // Code predictor identical across every released variant.
        assert_eq!(get_u32(&file, KEY_CP_HIDDEN_DIM), CP_HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_CP_N_LAYER), CP_N_LAYER);
        assert_eq!(get_u32(&file, KEY_CP_FFN_DIM), CP_FFN_DIM);
        assert_eq!(get_u32(&file, KEY_CP_VOCAB_SIZE), CP_VOCAB_SIZE);
        // Provenance: apache-2.0 permissive under the 1.7B-Base NAME
        // (distinct from every sibling — a downstream that ships all
        // three 1.7B GGUFs side-by-side must be able to tell them apart
        // by `vokra.provenance.model_id`).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_1_7B_BASE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn convert_variant_1_7b_customvoice_emits_the_1_7b_axes_and_customvoice_name() {
        let file = round_trip_variant(Qwen3TtsVariant::_1_7B_CustomVoice);
        // Arch tag never changes with variant — every variant is Qwen3-TTS.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // Variant-selected name.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_1_7B_CUSTOM_VOICE)
        );
        // Variant-selected talker axes.
        assert_eq!(
            get_u32(&file, KEY_TALKER_HIDDEN_DIM),
            TALKER_1_7B_HIDDEN_DIM
        );
        assert_eq!(get_u32(&file, KEY_TALKER_FFN_DIM), TALKER_1_7B_FFN_DIM);
        assert_eq!(
            get_u32(&file, KEY_TALKER_TEXT_HIDDEN_SIZE),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
        // Invariant axes: shared with 0.6B.
        assert_eq!(get_u32(&file, KEY_TALKER_N_LAYER), TALKER_N_LAYER);
        assert_eq!(get_u32(&file, KEY_TALKER_N_HEAD), TALKER_N_HEAD);
        assert_eq!(get_u32(&file, KEY_TALKER_N_HEAD_KV), TALKER_N_HEAD_KV);
        assert_eq!(get_u32(&file, KEY_TALKER_HEAD_DIM), TALKER_HEAD_DIM);
        assert_eq!(get_u32(&file, KEY_TALKER_VOCAB_SIZE), TALKER_VOCAB_SIZE);
        assert_eq!(
            get_u32(&file, KEY_TALKER_TEXT_VOCAB_SIZE),
            TALKER_TEXT_VOCAB_SIZE
        );
        assert_eq!(
            get_u32(&file, KEY_TALKER_MAX_POSITIONS),
            TALKER_MAX_POSITIONS
        );
        assert!((get_f32(&file, KEY_TALKER_ROPE_BASE) - TALKER_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_TALKER_RMS_NORM_EPS) - TALKER_RMS_NORM_EPS).abs() < 1e-12);
        assert_eq!(
            get_u32(&file, KEY_TALKER_POS_ID_PER_SEC),
            TALKER_POS_ID_PER_SEC
        );
        assert_eq!(
            get_u32(&file, KEY_TALKER_NUM_CODE_GROUPS),
            TALKER_NUM_CODE_GROUPS
        );
        // Code predictor identical across every released variant.
        assert_eq!(get_u32(&file, KEY_CP_HIDDEN_DIM), CP_HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_CP_N_LAYER), CP_N_LAYER);
        assert_eq!(get_u32(&file, KEY_CP_FFN_DIM), CP_FFN_DIM);
        assert_eq!(get_u32(&file, KEY_CP_VOCAB_SIZE), CP_VOCAB_SIZE);
        // Provenance: apache-2.0 permissive under the variant-specific NAME.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_1_7B_CUSTOM_VOICE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn convert_variant_1_7b_voicedesign_emits_the_1_7b_axes_and_voicedesign_name() {
        let file = round_trip_variant(Qwen3TtsVariant::_1_7B_VoiceDesign);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_1_7B_VOICE_DESIGN)
        );
        // Variant-selected axes — same values as CustomVoice.
        assert_eq!(
            get_u32(&file, KEY_TALKER_HIDDEN_DIM),
            TALKER_1_7B_HIDDEN_DIM
        );
        assert_eq!(get_u32(&file, KEY_TALKER_FFN_DIM), TALKER_1_7B_FFN_DIM);
        assert_eq!(
            get_u32(&file, KEY_TALKER_TEXT_HIDDEN_SIZE),
            TALKER_1_7B_TEXT_HIDDEN_SIZE
        );
        // Provenance: still apache-2.0 permissive, but under the
        // VoiceDesign NAME (distinct from CustomVoice).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME_1_7B_VOICE_DESIGN)
        );
    }

    /// Every 1.7B GGUF (Base + CustomVoice + VoiceDesign) built from the
    /// same synthetic safetensors input share talker + code-predictor axes.
    /// Base alone carries the official 2048-wide speaker encoder;
    /// CustomVoice and VoiceDesign carry none, so that axis is deliberately
    /// excluded from the shared set.
    #[test]
    fn all_1_7b_variants_share_talker_and_cp_axes() {
        let base_17b = round_trip_variant(Qwen3TtsVariant::_1_7B_Base);
        let cv = round_trip_variant(Qwen3TtsVariant::_1_7B_CustomVoice);
        let vd = round_trip_variant(Qwen3TtsVariant::_1_7B_VoiceDesign);

        for key in [
            KEY_TALKER_HIDDEN_DIM,
            KEY_TALKER_N_LAYER,
            KEY_TALKER_N_HEAD,
            KEY_TALKER_N_HEAD_KV,
            KEY_TALKER_HEAD_DIM,
            KEY_TALKER_FFN_DIM,
            KEY_TALKER_VOCAB_SIZE,
            KEY_TALKER_TEXT_VOCAB_SIZE,
            KEY_TALKER_MAX_POSITIONS,
            KEY_TALKER_POS_ID_PER_SEC,
            KEY_TALKER_NUM_CODE_GROUPS,
            KEY_TALKER_TEXT_HIDDEN_SIZE,
            KEY_CP_HIDDEN_DIM,
            KEY_CP_N_LAYER,
            KEY_CP_N_HEAD,
            KEY_CP_N_HEAD_KV,
            KEY_CP_HEAD_DIM,
            KEY_CP_FFN_DIM,
            KEY_CP_VOCAB_SIZE,
            KEY_CP_NUM_CODE_GROUPS,
            KEY_SAMPLE_RATE,
        ] {
            assert_eq!(
                get_u32(&base_17b, key),
                get_u32(&cv, key),
                "{key} must match across 1.7B Base and CustomVoice"
            );
            assert_eq!(
                get_u32(&cv, key),
                get_u32(&vd, key),
                "{key} must match across 1.7B CustomVoice and VoiceDesign"
            );
        }
        assert_eq!(get_u32(&base_17b, KEY_SPEAKER_EMBED_DIM), 2048);
        assert_eq!(get_u32(&cv, KEY_SPEAKER_EMBED_DIM), 0);
        assert_eq!(get_u32(&vd, KEY_SPEAKER_EMBED_DIM), 0);
        assert_eq!(
            base_17b
                .get(KEY_HAS_SPEAKER_ENCODER)
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            cv.get(KEY_HAS_SPEAKER_ENCODER)
                .and_then(|value| value.as_bool()),
            Some(false)
        );
        assert_eq!(
            vd.get(KEY_HAS_SPEAKER_ENCODER)
                .and_then(|value| value.as_bool()),
            Some(false)
        );
        // NAME must be PAIRWISE distinct.
        let name_base_17b = base_17b
            .get(chunks::KEY_MODEL_NAME)
            .and_then(|v| v.as_str());
        let name_cv = cv.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str());
        let name_vd = vd.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str());
        assert_ne!(name_base_17b, name_cv);
        assert_ne!(name_base_17b, name_vd);
        assert_ne!(name_cv, name_vd);
        // But 0.6B and 1.7B talker hidden MUST differ (proves the variant
        // enum actually selects distinct axes).
        let base_06b = round_trip_variant(Qwen3TtsVariant::_0_6B_Base);
        assert_ne!(
            get_u32(&base_06b, KEY_TALKER_HIDDEN_DIM),
            get_u32(&base_17b, KEY_TALKER_HIDDEN_DIM),
            "0.6B and 1.7B talker hidden dims MUST differ"
        );
        assert_ne!(
            get_u32(&base_06b, KEY_TALKER_FFN_DIM),
            get_u32(&base_17b, KEY_TALKER_FFN_DIM),
            "0.6B and 1.7B talker FFN dims MUST differ"
        );
    }

    /// BF16 pass-through works uniformly for every 1.7B variant. The
    /// upstream releases ship BF16 (`safetensors.parameters.BF16 =
    /// 1_916_676_352` per the HF model cards fetched 2026-07-30 /
    /// 2026-08-01 — Base / CustomVoice / VoiceDesign all ship the same
    /// BF16 tensor count), so this arm is the real production path. The
    /// `_1_7B_Base` variant (added 2026-08-01, Wave 4) is included in
    /// the sweep so the BF16 posture is pinned across every 1.7B
    /// release the converter dispatches.
    #[test]
    fn bf16_pass_through_works_for_1_7b_variants() {
        // Reuse the fixture builder pattern from the top of the test module.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let header = r#"{"talker.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&bf16);

        for variant in [
            Qwen3TtsVariant::_1_7B_Base,
            Qwen3TtsVariant::_1_7B_CustomVoice,
            Qwen3TtsVariant::_1_7B_VoiceDesign,
        ] {
            let (builder, report) =
                convert_variant_fixture(input.clone(), variant).expect("BF16 convert fixture");
            assert_eq!(report.written, 1);
            assert_eq!(report.bf16_passthrough, 1);
            assert_eq!(report.skipped_non_float, 0);
            let out = builder.to_bytes().expect("serialize");
            let file = GgufFile::parse(out).expect("parse");
            let info = file
                .tensor_info("talker.embed_tokens.weight")
                .expect("BF16 tensor present");
            assert_eq!(info.dtype, GgmlType::BF16, "{variant:?}: BF16 stays BF16");
            assert_eq!(file.tensor_bytes(info), bf16.as_slice());
        }
    }
}
