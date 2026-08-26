//! **MOSS-TTS** (`OpenMOSS-Team/MOSS-TTS*`, apache-2.0): safetensors →
//! GGUF conversion (SoTA follow-on, added 2026-07-30).
//!
//! Input: one of the OpenMOSS TTS release safetensors — the family
//! spans four sibling checkpoints that share a single arch tag
//! (`moss_tts`) but differ in backbone family, audio-tokenizer axes
//! and per-frame codebook count. Output: a GGUF carrying every
//! float tensor verbatim under its upstream safetensors name, plus
//! the `vokra.moss_tts.*`, `vokra.model.*` and `vokra.provenance.*`
//! metadata chunks a future native MOSS-TTS loader will read.
//!
//! # Family coverage — variant selectors
//!
//! [`MossTtsVariant`] selects the per-release constants. The variants
//! were transcribed verbatim from upstream `config.json` files fetched
//! 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」):
//!
//! - [`MossTtsVariant::Delay`] —
//!   `OpenMOSS-Team/MOSS-TTS` and `OpenMOSS-Team/MOSS-TTS-v1.5` (both
//!   `model_type = "moss_tts_delay"`; `n_vq = 32`; `audio_vocab_size =
//!   1024`; `sampling_rate = 24_000`; language backbone = Qwen3-8B
//!   with hidden=4096 / ffn=12288 / n_layer=36 / n_head=32 /
//!   n_head_kv=8 / head_dim=128 / vocab=155_648 / rope_theta=1_000_000
//!   / rms_norm_eps=1e-6). The two releases share identical axes and
//!   differ only in language-coverage training data + release id + the
//!   `vokra.model.name` stamp.
//! - [`MossTtsVariant::Nano`] — `OpenMOSS-Team/MOSS-TTS-Nano-100M`
//!   (`model_type = "moss_tts_nano"`; `n_vq = 16`; `audio_vocab_size =
//!   1024`; `audio_tokenizer_sample_rate = 48_000`; language backbone
//!   = a GPT-2 flavour with `hidden_size = 768` /
//!   `gpt2_config.n_layer = 12` / `gpt2_config.n_head = 12` /
//!   `gpt2_config.n_positions = 32_768` / vocab = 16_384 / rotary
//!   positions with base 10_000 / LayerNorm epsilon 1e-5;
//!   `local_transformer_layers = 1`). Ships as a torch pickle
//!   `pytorch_model.bin`, not safetensors — callers pre-bridge with
//!   `tools/parity/bin_to_safetensors.py` (the OpenBMB VoxCPM
//!   precedent, `docs/license-audit.md` row 281 sign-off note).
//! - [`MossTtsVariant::Local`] —
//!   `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
//!   (`model_type = "moss_tts_local"`; `n_vq = 12`; `audio_vocab_size =
//!   1024`; `sampling_rate = 48_000`; language backbone = a Qwen3-
//!   flavour with hidden=2560 / ffn=9728 / n_layer=36 / n_head=32 /
//!   n_head_kv=8 / head_dim=128 / vocab=151_936 / rope_theta=1_000_000
//!   / rms_norm_eps=1e-6; plus a `gpt2_config` local head).
//!
//! # HF / licence / category
//!
//! - Upstream HF (recorded under `vokra.provenance.upstream_hf` +
//!   `vokra.model.name`): `OpenMOSS-Team/MOSS-TTS[-v1.5|-Nano-100M|
//!   -Local-Transformer-v1.5]`.
//! - SPDX: `apache-2.0` for every variant (`cardData.license =
//!   "apache-2.0"` on every HF model card, fetched 2026-07-30 via
//!   `curl https://huggingface.co/api/models/<id>` —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - Model category: `tts` (recorded under `vokra.model.category`).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `wespeaker` /
//! `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`). No convert-time widening; runtime widens
//! BF16 → f32 losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Wespeaker / Neucodec contract). Real-weight binding
//! into a `MossTtsWeights::from_gguf` is a follow-up wave gated on the
//! upstream tensor-name manifest fetch; this converter passes every
//! F32 / F16 / BF16 tensor through unchanged so a future native loader
//! can walk the same names.
//!
//! # Real-weight parity + runtime forward
//!
//! Real-weight parity and the native runtime forward are deferred to
//! owner (`docs/license-audit.md` §3.1 sign-off). This converter
//! provides the byte-parallel GGUF surface only — a "loud-partial"
//! landing per the RMVPE / Charsiu / pyannote precedent.
//!
//! # No ONNX (permanent)
//!
//! MOSS-TTS is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/moss_tts/`.
//!
//! # Vast.ai
//!
//! The Delay variants (`MOSS-TTS` + `MOSS-TTS-v1.5`, both ~17 GB BF16
//! across 4 safetensors shards) and the Local variant (~9 GB BF16) all
//! exceed the repository's 2 GB local-artifact ceiling and therefore belong
//! on vast.ai. This non-streaming converter must not run on the maintainer Mac
//! for those variants.
//! The Nano variant fits locally but ships as a torch pickle
//! `pytorch_model.bin` and needs a bridge pass first. Owner runs the
//! actual conversion on vast.ai per the model-publish runbook
//! (`docs/handoff/vast-ai-large-model-publish.md`).

// Skeleton-only allowance: the public API (`convert_moss_tts_file`,
// `MossTtsReport`, `MossTtsVariant`, the `KEY_*` / `MODEL_CATEGORY` /
// `UPSTREAM_HF_*` constants) is exercised by the in-module tests and
// wired to the CLI + `ModelKind` + `pub use` re-export in `lib.rs` in
// the same commit — this attribute is a no-op once every symbol is
// referenced from `lib.rs` and can be removed then. Kept for
// safety while the `lib.rs` wiring is landing in parallel.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::{FrontendSpec, LicenseClass};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

// ─── Family-wide constants ───────────────────────────────────────────
//
// The four MOSS-TTS releases share ONE arch tag (`moss_tts`) but
// distinct NAMEs, backbones and audio-tokenizer axes. The variant is
// carried into the emitted GGUF via `vokra.moss_tts.variant` so a
// runtime dispatcher can pick the right forward pipeline without
// re-reading every hparam.

/// `vokra.model.arch` for MOSS-TTS GGUFs.
pub(crate) const ARCH: &str = "moss_tts";

/// Dedicated architecture tag for the MOSS-Audio understanding models.
///
/// The historical public 4B/8B GGUFs were emitted through this MOSS-TTS
/// converter and therefore carry `moss_tts`.  New conversions must use the
/// upstream `model_type = "moss_audio"`; the runtime may admit the historical
/// files only through their exact 901-tensor manifest hashes.
pub(crate) const MOSS_AUDIO_ARCH: &str = "moss_audio";

const MOSS_AUDIO_SOURCE_CODE_REVISION: &str = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883";
const MOSS_AUDIO_CONFIGURATION_SHA256: &str =
    "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd";
const MOSS_AUDIO_MODELING_SHA256: &str =
    "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c";
const MOSS_AUDIO_PROCESSING_SHA256: &str =
    "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6";

/// Model-category tag written under `vokra.model.category`. `"tts"`
/// distinguishes MOSS-TTS from speaker / codec / ASR siblings so a
/// downstream can pick a load path without inspecting the arch.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "tts";
/// Category stamp for the MOSS-Audio-4B-Instruct sibling
/// (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0). Distinct from
/// the sibling `tts` variants because the "-Instruct" release is an
/// audio-LLM (custom `configuration_moss_audio.py` module), matching
/// the `s2s` category the sibling audio-LLM converters
/// (`kimi_audio` / `baichuan_audio` / `step_audio2_mini`) already
/// stamp. Selected per-variant via [`MossTtsVariant::category`] so the
/// existing 4 `tts` variants keep their current stamp byte-for-byte.
pub(crate) const MODEL_CATEGORY_S2S: &str = "s2s";

/// Upstream HF repository slug written under `vokra.provenance.upstream_hf`
/// — preserves upstream casing.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Per-variant HF release slugs.
pub(crate) const UPSTREAM_HF_MOSS_TTS: &str = "OpenMOSS-Team/MOSS-TTS";
pub(crate) const UPSTREAM_HF_MOSS_TTS_V15: &str = "OpenMOSS-Team/MOSS-TTS-v1.5";
pub(crate) const UPSTREAM_HF_MOSS_TTS_NANO: &str = "OpenMOSS-Team/MOSS-TTS-Nano-100M";
pub(crate) const UPSTREAM_HF_MOSS_TTS_LOCAL: &str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5";
/// VoiceGenerator is a smaller `moss_tts_delay` release.  It shares the
/// generation algorithm with Delay, but not the 8B tensor axes.
pub(crate) const UPSTREAM_HF_MOSS_VOICE_GENERATOR: &str = "OpenMOSS-Team/MOSS-VoiceGenerator";
/// MOSS-Audio-4B-Instruct HF release slug
/// (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0). Distinct
/// upstream sibling that shares the moss_tts family arch tag (this
/// converter is reused per the parent workflow's REUSE HINT), but is
/// a 4B audio-LLM custom-code release
/// (`configuration_moss_audio.py`, `trust_remote_code=True`) rather
/// than one of the four `moss_tts_*` (Delay / Nano / Local) tts
/// releases.
pub(crate) const UPSTREAM_HF_MOSS_AUDIO_4B_INSTRUCT: &str = "OpenMOSS-Team/MOSS-Audio-4B-Instruct";
/// MOSS-Audio-8B-Instruct HF release slug
/// (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`, apache-2.0). Larger
/// sibling of the 4B audio-LLM release (**4 shards ~9.05 GB BF16** per
/// parent workflow manifest 2026-08-02) with the same custom-code
/// architecture (`configuration_moss_audio.py`,
/// `trust_remote_code=True`). Requires vast.ai for a downloading
/// conversion (memory `[[feedback-large-models-on-vast-ai]]`).
pub(crate) const UPSTREAM_HF_MOSS_AUDIO_8B_INSTRUCT: &str = "OpenMOSS-Team/MOSS-Audio-8B-Instruct";

/// Per-variant `vokra.model.name` stamps (canonical, lower-cased HF slug
/// tail — mirrors the Qwen3-TTS / Chatterbox naming convention).
pub(crate) const NAME_MOSS_TTS: &str = "moss-tts";
pub(crate) const NAME_MOSS_TTS_V15: &str = "moss-tts-v1.5";
pub(crate) const NAME_MOSS_TTS_NANO: &str = "moss-tts-nano-100m";
pub(crate) const NAME_MOSS_TTS_LOCAL: &str = "moss-tts-local-transformer-v1.5";
/// `vokra.model.name` for the distinct Qwen3-1.7B VoiceGenerator release.
pub(crate) const NAME_MOSS_VOICE_GENERATOR: &str = "moss-voice-generator";
/// `vokra.model.name` stamp for MOSS-Audio-4B-Instruct — the
/// lower-cased HF slug tail, matching the sibling naming convention.
pub(crate) const NAME_MOSS_AUDIO_4B_INSTRUCT: &str = "moss-audio-4b-instruct";
/// `vokra.model.name` stamp for MOSS-Audio-8B-Instruct — the
/// lower-cased HF slug tail, matching the sibling naming convention.
pub(crate) const NAME_MOSS_AUDIO_8B_INSTRUCT: &str = "moss-audio-8b-instruct";

// ─── vokra.moss_tts.* metadata keys ──────────────────────────────────

/// Sub-arch discriminator (`"delay" | "nano" | "local"`) — matches the
/// upstream `config.json.model_type` fragment after the shared
/// `moss_tts_` prefix. Emitted so a runtime dispatcher can pick the
/// correct forward pipeline (delay vs nano vs local codec chain)
/// without re-parsing every hparam.
const KEY_MOSS_VARIANT: &str = "vokra.moss_tts.variant";

/// Number of parallel codebook streams per audio frame
/// (`config.json.n_vq`). 32 for Delay, 16 for Nano, 12 for Local.
const KEY_MOSS_N_VQ: &str = "vokra.moss_tts.n_vq";

/// Per-codebook audio vocabulary size (`config.json.audio_vocab_size`).
/// 1024 for every released variant; kept as a per-variant key so a
/// future release that widens the codec can differ without silently
/// mis-shaping the decoder.
const KEY_MOSS_AUDIO_VOCAB_SIZE: &str = "vokra.moss_tts.audio_vocab_size";

/// Output PCM sample rate (Hz). 24000 for Delay,
/// 48000 for Nano (`audio_tokenizer_sample_rate` — Nano puts the rate
/// under the audio-tokenizer sub-config, not at the top level) and
/// Local (`config.json.sampling_rate`).
const KEY_MOSS_SAMPLE_RATE: &str = "vokra.moss_tts.sample_rate";

/// Language backbone hidden dimension. 4096 for Delay (Qwen3-8B),
/// 768 for Nano (GPT-2 flavour), 2560 for Local (Qwen3-flavour).
const KEY_MOSS_LLM_HIDDEN_DIM: &str = "vokra.moss_tts.llm.hidden_dim";

/// Language backbone FFN inner dimension. 12288 for Delay, N/A for
/// Nano (GPT-2 defaults to 4·hidden internally), 9728 for Local.
const KEY_MOSS_LLM_FFN_DIM: &str = "vokra.moss_tts.llm.ffn_dim";

/// Language backbone transformer block count. 36 for Delay + Local,
/// 12 for Nano (Nano's is stored under `gpt2_config.n_layer`).
const KEY_MOSS_LLM_N_LAYER: &str = "vokra.moss_tts.llm.n_layer";

/// Language backbone attention head count. 32 for Delay + Local,
/// 12 for Nano.
const KEY_MOSS_LLM_N_HEAD: &str = "vokra.moss_tts.llm.n_head";

/// Language backbone KV head count for GQA. 8 for Delay + Local,
/// N/A for Nano (GPT-2 is MHA — same as `n_head`).
const KEY_MOSS_LLM_N_HEAD_KV: &str = "vokra.moss_tts.llm.n_head_kv";

/// Language backbone attention head dimension. 128 for Delay + Local,
/// 64 for Nano (`hidden_size / n_head = 768 / 12 = 64`).
const KEY_MOSS_LLM_HEAD_DIM: &str = "vokra.moss_tts.llm.head_dim";

/// Language backbone token vocabulary size. 155648 for Delay
/// (Qwen3-8B extended), 16384 for Nano, 151936 for Local.
const KEY_MOSS_LLM_VOCAB_SIZE: &str = "vokra.moss_tts.llm.vocab_size";

/// Language backbone RoPE base θ (`rope_theta`). 1_000_000 for
/// Delay + Local. The custom GPT-2 implementation used by Nano also
/// uses RoPE, with `gpt2_config.position_embedding_type = "rope"` and
/// `gpt2_config.rope_base = 10_000`.
const KEY_MOSS_LLM_ROPE_BASE: &str = "vokra.moss_tts.llm.rope_base";

/// Language backbone RMSNorm ε (`rms_norm_eps`). 1e-6 for Delay +
/// Local. GPT-2 (Nano) uses LayerNorm, so the Nano key is written as
/// `0.0` sentinel — the runtime binder must consult the backbone
/// family before applying this.
const KEY_MOSS_LLM_RMS_NORM_EPS: &str = "vokra.moss_tts.llm.rms_norm_eps";

/// Language backbone family — string discriminator so a future runtime
/// can pick RoPE vs learned pos, RMSNorm vs LayerNorm, GQA vs MHA
/// without inspecting the arch label. `"qwen3"` for Delay + Local,
/// `"gpt2"` for Nano.
const KEY_MOSS_LLM_FAMILY: &str = "vokra.moss_tts.llm.family";

/// Exact upstream revision used for corrected release contracts.
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
/// SHA-256 of the released Nano `pytorch_model.bin` LFS object.
const KEY_PROVENANCE_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
/// SHA-256 of the pinned Nano `config.json`.
const KEY_MOSS_CONFIG_SHA256: &str = "vokra.moss_tts.config_sha256";
/// Positional encoding selected by the custom GPT-2 implementation.
const KEY_MOSS_POSITION_EMBEDDING_TYPE: &str = "vokra.moss_tts.llm.position_embedding_type";
/// GPT-2 LayerNorm epsilon. Distinct from the legacy RMSNorm-only key.
const KEY_MOSS_LAYER_NORM_EPS: &str = "vokra.moss_tts.llm.layer_norm_eps";
/// Maximum global sequence length.
const KEY_MOSS_MAX_POSITION_EMBEDDINGS: &str = "vokra.moss_tts.llm.max_position_embeddings";
/// Number of autoregressive per-frame local-transformer blocks.
const KEY_MOSS_LOCAL_TRANSFORMER_LAYERS: &str = "vokra.moss_tts.local_transformer_layers";
/// Text/prompt and audio framing token IDs required by Nano generation.
const KEY_MOSS_PAD_TOKEN_ID: &str = "vokra.moss_tts.pad_token_id";
const KEY_MOSS_IM_START_TOKEN_ID: &str = "vokra.moss_tts.im_start_token_id";
const KEY_MOSS_IM_END_TOKEN_ID: &str = "vokra.moss_tts.im_end_token_id";
const KEY_MOSS_AUDIO_START_TOKEN_ID: &str = "vokra.moss_tts.audio_start_token_id";
const KEY_MOSS_AUDIO_END_TOKEN_ID: &str = "vokra.moss_tts.audio_end_token_id";
const KEY_MOSS_AUDIO_USER_SLOT_TOKEN_ID: &str = "vokra.moss_tts.audio_user_slot_token_id";
const KEY_MOSS_AUDIO_ASSISTANT_SLOT_TOKEN_ID: &str = "vokra.moss_tts.audio_assistant_slot_token_id";
const KEY_MOSS_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID: &str =
    "vokra.moss_tts.audio_assistant_gen_slot_token_id";
const KEY_MOSS_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID: &str =
    "vokra.moss_tts.audio_assistant_delay_slot_token_id";
const KEY_MOSS_AUDIO_PAD_TOKEN_ID: &str = "vokra.moss_tts.audio_pad_token_id";
/// SHA-256 pins for custom model/processor source when the release requires it.
const KEY_MOSS_MODELING_SOURCE_SHA256: &str = "vokra.moss_tts.modeling_source_sha256";
const KEY_MOSS_PROCESSING_SOURCE_SHA256: &str = "vokra.moss_tts.processing_source_sha256";
/// SHA-256 pins for the remaining fixed-revision Local custom-code files.
const KEY_MOSS_CONFIGURATION_SOURCE_SHA256: &str = "vokra.moss_tts.configuration_source_sha256";
const KEY_MOSS_QWEN3_DECODER_SOURCE_SHA256: &str = "vokra.moss_tts.qwen3_decoder_source_sha256";
const KEY_MOSS_GPT2_DECODER_SOURCE_SHA256: &str = "vokra.moss_tts.gpt2_decoder_source_sha256";
const KEY_MOSS_PROCESSOR_CONFIG_SHA256: &str = "vokra.moss_tts.processor_config_sha256";
/// Exact decoder companion selected by the release config.
const KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM: &str = "vokra.moss_tts.audio_tokenizer_upstream_hf";
/// Local GPT-2 decoder contract. These axes are deliberately distinct from
/// the global Qwen3 `llm.*` keys above.
const KEY_MOSS_LOCAL_HIDDEN_DIM: &str = "vokra.moss_tts.local_transformer.hidden_dim";
const KEY_MOSS_LOCAL_FFN_DIM: &str = "vokra.moss_tts.local_transformer.ffn_dim";
const KEY_MOSS_LOCAL_N_HEAD: &str = "vokra.moss_tts.local_transformer.n_head";
const KEY_MOSS_LOCAL_HEAD_DIM: &str = "vokra.moss_tts.local_transformer.head_dim";
const KEY_MOSS_LOCAL_POSITION_EMBEDDING_TYPE: &str =
    "vokra.moss_tts.local_transformer.position_embedding_type";
const KEY_MOSS_LOCAL_ROPE_BASE: &str = "vokra.moss_tts.local_transformer.rope_base";
const KEY_MOSS_LOCAL_LAYER_NORM_EPS: &str = "vokra.moss_tts.local_transformer.layer_norm_eps";
const KEY_MOSS_LOCAL_ACTIVATION: &str = "vokra.moss_tts.local_transformer.activation";
const KEY_MOSS_LOCAL_TEXT_HEAD_MODE: &str = "vokra.moss_tts.local_text_head_mode";
const KEY_MOSS_LOCAL_STATIC_KV_CACHE: &str = "vokra.moss_tts.local_transformer.use_static_kv_cache";

// ─── Per-variant transcribed constants ───────────────────────────────

/// The Delay variants share these axes (Delay = "MOSS-TTS" + "MOSS-TTS-v1.5").
/// Both are `model_type = "moss_tts_delay"` with a Qwen3-8B backbone.
///
/// Primary source:
///   - `huggingface.co/OpenMOSS-Team/MOSS-TTS/raw/main/config.json`
///     (fetched 2026-07-30)
///   - `huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5/raw/main/config.json`
///     (fetched 2026-07-30)
const DELAY_N_VQ: u32 = 32;
const DELAY_AUDIO_VOCAB_SIZE: u32 = 1024;
const DELAY_SAMPLE_RATE: u32 = 24_000;
const DELAY_LLM_HIDDEN: u32 = 4096;
const DELAY_LLM_FFN: u32 = 12_288;
const DELAY_LLM_N_LAYER: u32 = 36;
const DELAY_LLM_N_HEAD: u32 = 32;
const DELAY_LLM_N_HEAD_KV: u32 = 8;
const DELAY_LLM_HEAD_DIM: u32 = 128;
const DELAY_LLM_VOCAB: u32 = 155_648;
const DELAY_LLM_ROPE_BASE: f32 = 1_000_000.0;
const DELAY_LLM_RMS_NORM_EPS: f32 = 1e-6;

/// MOSS-VoiceGenerator axes from the official `config.json` at revision
/// `97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4` (fetched 2026-08-26).
/// The upstream class is still `MossTTSDelayModel`, but the checkpoint is a
/// Qwen3-1.7B / 16-codebook topology and must never inherit the 8B Delay
/// constants merely because `model_type = "moss_tts_delay"` is shared.
const VOICE_N_VQ: u32 = 16;
const VOICE_AUDIO_VOCAB_SIZE: u32 = 1_024;
const VOICE_SAMPLE_RATE: u32 = 24_000;
const VOICE_LLM_HIDDEN: u32 = 2_048;
const VOICE_LLM_FFN: u32 = 6_144;
const VOICE_LLM_N_LAYER: u32 = 28;
const VOICE_LLM_N_HEAD: u32 = 16;
const VOICE_LLM_N_HEAD_KV: u32 = 8;
const VOICE_LLM_HEAD_DIM: u32 = 128;
const VOICE_LLM_VOCAB: u32 = 155_648;
const VOICE_LLM_ROPE_BASE: f32 = 1_000_000.0;
const VOICE_LLM_RMS_NORM_EPS: f32 = 1e-6;
const VOICE_UPSTREAM_REVISION: &str = "97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4";
const VOICE_CONFIG_SHA256: &str =
    "5b6ccfbf309a5844c130d09c9b5fa8b9eef55db27f1b7072695483b6f5524685";
const VOICE_MODELING_SOURCE_SHA256: &str =
    "666d7320f93ce6b1c1f6ed4dba6fd4b9520a082a90fa7a17211efd83247d28a0";
const VOICE_PROCESSING_SOURCE_SHA256: &str =
    "16dda5233f9f752518d07a6b780d6555945b48547fba0b4e7faf6eb2c4ed0038";
const VOICE_POSITION_EMBEDDING_TYPE: &str = "rope";
const VOICE_MAX_POSITION_EMBEDDINGS: u32 = 40_960;
const VOICE_PAD_TOKEN_ID: u32 = 151_643;
const VOICE_IM_START_TOKEN_ID: u32 = 151_644;
const VOICE_IM_END_TOKEN_ID: u32 = 151_645;
const VOICE_AUDIO_START_TOKEN_ID: u32 = 151_652;
const VOICE_AUDIO_END_TOKEN_ID: u32 = 151_653;
const VOICE_AUDIO_USER_SLOT_TOKEN_ID: u32 = 151_654;
const VOICE_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID: u32 = 151_656;
const VOICE_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID: u32 = 151_662;
const VOICE_AUDIO_PAD_TOKEN_ID: u32 = 1_024;
const VOICE_AUDIO_TOKENIZER_UPSTREAM: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";

/// Nano axes. GPT-2 backbone — no RoPE, no RMSNorm; sentinels written
/// for those keys so the runtime binder can tell "not applicable"
/// apart from silent default (FR-EX-08).
///
/// Primary source:
///   - `huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/raw/main/config.json`
///     (fetched 2026-07-30; `hidden_size` at top level, transformer
///     shape under `gpt2_config`, audio rate under
///     `audio_tokenizer_sample_rate`)
const NANO_N_VQ: u32 = 16;
const NANO_AUDIO_VOCAB_SIZE: u32 = 1024;
const NANO_SAMPLE_RATE: u32 = 48_000;
const NANO_LLM_HIDDEN: u32 = 768;
/// GPT-2 defaults `n_inner = 4 * n_embd` when not overridden. Recorded
/// as the resolved value (`4 * 768 = 3072`) so downstream matches the
/// on-tensor shape.
const NANO_LLM_FFN: u32 = 3072;
const NANO_LLM_N_LAYER: u32 = 12;
const NANO_LLM_N_HEAD: u32 = 12;
/// GPT-2 is MHA — no GQA split. Recorded as `n_head` for uniformity.
const NANO_LLM_N_HEAD_KV: u32 = 12;
/// `hidden_size / n_head = 768 / 12 = 64`.
const NANO_LLM_HEAD_DIM: u32 = 64;
const NANO_LLM_VOCAB: u32 = 16_384;
/// Custom GPT-2 RoPE base from the pinned `config.json`.
const NANO_LLM_ROPE_BASE: f32 = 10_000.0;
/// GPT-2 uses LayerNorm — no RMSNorm ε. Sentinel `0.0`.
const NANO_LLM_RMS_NORM_EPS: f32 = 0.0;
const NANO_UPSTREAM_REVISION: &str = "44502f80dbf9743528fa921cc544d662c685ebec";
const NANO_CHECKPOINT_SHA256: &str =
    "24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5";
const NANO_CONFIG_SHA256: &str = "ba36b08c80d4ae0805a2bab32b6ac90ec0d1815d01d3854ba42811db1d5bde99";
const NANO_POSITION_EMBEDDING_TYPE: &str = "rope";
const NANO_LAYER_NORM_EPS: f32 = 1e-5;
const NANO_MAX_POSITION_EMBEDDINGS: u32 = 32_768;
const NANO_LOCAL_TRANSFORMER_LAYERS: u32 = 1;
const NANO_PAD_TOKEN_ID: u32 = 3;
const NANO_IM_START_TOKEN_ID: u32 = 4;
const NANO_IM_END_TOKEN_ID: u32 = 5;
const NANO_AUDIO_START_TOKEN_ID: u32 = 6;
const NANO_AUDIO_END_TOKEN_ID: u32 = 7;
const NANO_AUDIO_USER_SLOT_TOKEN_ID: u32 = 8;
const NANO_AUDIO_ASSISTANT_SLOT_TOKEN_ID: u32 = 9;
const NANO_AUDIO_PAD_TOKEN_ID: u32 = 1_024;
const NANO_AUDIO_TOKENIZER_UPSTREAM: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano";

/// Local axes. Qwen3-flavour backbone at 2.5 B scale.
///
/// Primary source:
///   - `huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5/raw/main/config.json`
///     (fetched 2026-07-30; language backbone under `qwen3_config`)
const LOCAL_N_VQ: u32 = 12;
const LOCAL_AUDIO_VOCAB_SIZE: u32 = 1024;
const LOCAL_SAMPLE_RATE: u32 = 48_000;
const LOCAL_LLM_HIDDEN: u32 = 2560;
const LOCAL_LLM_FFN: u32 = 9728;
const LOCAL_LLM_N_LAYER: u32 = 36;
const LOCAL_LLM_N_HEAD: u32 = 32;
const LOCAL_LLM_N_HEAD_KV: u32 = 8;
const LOCAL_LLM_HEAD_DIM: u32 = 128;
const LOCAL_LLM_VOCAB: u32 = 151_936;
const LOCAL_LLM_ROPE_BASE: f32 = 1_000_000.0;
const LOCAL_LLM_RMS_NORM_EPS: f32 = 1e-6;
const LOCAL_UPSTREAM_REVISION: &str = "be7766a6735b98bd793f7c79fb720b4d0f5d13b8";
const LOCAL_CONFIG_SHA256: &str =
    "826f81f163b1b557ad13f83c4f35008f4fee5a6cb6311b4316ff3dbb25149411";
const LOCAL_CONFIGURATION_SOURCE_SHA256: &str =
    "ab6debcb92032cb9dc91ae80aed77dbadd2e59848208baef2b062bd6def3f3be";
const LOCAL_MODELING_SOURCE_SHA256: &str =
    "b0a66211943ae580b087f3e71495fea2f455701a4f6c29b6d3562218f7668c5f";
const LOCAL_PROCESSING_SOURCE_SHA256: &str =
    "3fc5616b1ec3408162b7d859a7696725a40525313b20f9b31a06ee55c93bd7ad";
const LOCAL_GPT2_DECODER_SOURCE_SHA256: &str =
    "f2e877104669f1e6c7cd34680f0da1a8a159e032123ee56b660b63929b6c8989";
const LOCAL_QWEN3_DECODER_SOURCE_SHA256: &str =
    "100163bd7ecf31a59bafacc0b032ace9339edc992a3eb4cc80662502e04e46f0";
const LOCAL_PROCESSOR_CONFIG_SHA256: &str =
    "db574bfebad009e05193196a63a4eeecd353eeca177ccfff28b9379d595d88b7";
const LOCAL_POSITION_EMBEDDING_TYPE: &str = "rope";
const LOCAL_MAX_POSITION_EMBEDDINGS: u32 = 32_768;
const LOCAL_TRANSFORMER_LAYERS: u32 = 1;
const LOCAL_TRANSFORMER_HIDDEN: u32 = 2_560;
const LOCAL_TRANSFORMER_FFN: u32 = 9_728;
const LOCAL_TRANSFORMER_N_HEAD: u32 = 32;
const LOCAL_TRANSFORMER_HEAD_DIM: u32 = 80;
const LOCAL_TRANSFORMER_POSITION_EMBEDDING_TYPE: &str = "rope";
const LOCAL_TRANSFORMER_ROPE_BASE: f32 = 1_000_000.0;
const LOCAL_TRANSFORMER_LAYER_NORM_EPS: f32 = 1e-6;
const LOCAL_TRANSFORMER_ACTIVATION: &str = "silu";
const LOCAL_TEXT_HEAD_MODE: &str = "binary";
const LOCAL_PAD_TOKEN_ID: u32 = 151_643;
const LOCAL_IM_START_TOKEN_ID: u32 = 151_644;
const LOCAL_IM_END_TOKEN_ID: u32 = 151_645;
const LOCAL_AUDIO_START_TOKEN_ID: u32 = 151_669;
const LOCAL_AUDIO_END_TOKEN_ID: u32 = 151_670;
const LOCAL_AUDIO_USER_SLOT_TOKEN_ID: u32 = 151_654;
const LOCAL_AUDIO_ASSISTANT_SLOT_TOKEN_ID: u32 = 151_656;
const LOCAL_AUDIO_PAD_TOKEN_ID: u32 = 1_024;
const LOCAL_AUDIO_TOKENIZER_UPSTREAM: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2";

// ─── Variant enum + selectors ────────────────────────────────────────

/// Which MOSS-TTS release variant to stamp into the emitted GGUF.
///
/// Each variant selects a distinct set of hparams and a distinct
/// `vokra.model.name` stamp. The tag written under
/// `vokra.moss_tts.variant` (via [`Self::sub_arch`]) matches the
/// suffix after `moss_tts_` in upstream `config.json.model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MossTtsVariant {
    /// `OpenMOSS-Team/MOSS-TTS` — the base 8.49B `moss_tts_delay` release.
    Delay,
    /// `OpenMOSS-Team/MOSS-TTS-v1.5` — a sibling of `Delay` with the
    /// same axes; distinct language coverage and NAME stamp.
    DelayV15,
    /// `OpenMOSS-Team/MOSS-TTS-Nano-100M` — the small `moss_tts_nano`
    /// GPT-2-backbone variant. Ships as a torch pickle
    /// `pytorch_model.bin` — callers pre-bridge to safetensors.
    Nano,
    /// `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` — the mid-scale
    /// `moss_tts_local` release (Qwen3-flavour 2.5B + GPT-2 local
    /// head).
    Local,
    /// `OpenMOSS-Team/MOSS-VoiceGenerator` — the Qwen3-1.7B
    /// `moss_tts_delay` release with 28 layers and 16 audio codebooks.
    /// It reuses Delay generation semantics, but has a distinct exact tensor
    /// contract and provenance identity.
    VoiceGenerator,
    /// `OpenMOSS-Team/MOSS-Audio-4B-Instruct` — the 5.22B-parameter
    /// audio-understanding model.  It uses a 32-layer Whisper-style audio
    /// encoder, four GatedMLP adapters (one primary plus three DeepStack
    /// injections), and a 36-layer Qwen3 text decoder with hidden size 2560.
    /// The exact axes come from upstream revision
    /// `6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d` and the official native
    /// implementation pinned by `MOSS_AUDIO_SOURCE_CODE_REVISION`.
    AudioInstruct4b,
    /// `OpenMOSS-Team/MOSS-Audio-8B-Instruct` — the 9.05B-parameter
    /// sibling.  The audio tower and adapters are identical to the 4B
    /// release; its Qwen3 decoder widens to hidden size 4096 and FFN size
    /// 12288.  The exact axes come from upstream revision
    /// `6521a39181b47a18f2d9f4b3acfb5bca7b76b57f`.
    AudioInstruct8b,
}

impl MossTtsVariant {
    pub(crate) const fn arch(self) -> &'static str {
        match self {
            Self::AudioInstruct4b | Self::AudioInstruct8b => MOSS_AUDIO_ARCH,
            _ => ARCH,
        }
    }

    pub(crate) const fn is_audio_instruct(self) -> bool {
        matches!(self, Self::AudioInstruct4b | Self::AudioInstruct8b)
    }

    /// Sub-arch tag written under `vokra.moss_tts.variant`.
    pub(crate) const fn sub_arch(self) -> &'static str {
        match self {
            Self::Delay | Self::DelayV15 => "delay",
            Self::Nano => "nano",
            Self::Local => "local",
            Self::VoiceGenerator => "voice_generator",
            // Dedicated release tag under the `vokra.moss_audio.*` group.
            Self::AudioInstruct4b => "4b_instruct",
            Self::AudioInstruct8b => "8b_instruct",
        }
    }

    /// `vokra.model.name` stamp for this variant.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Delay => NAME_MOSS_TTS,
            Self::DelayV15 => NAME_MOSS_TTS_V15,
            Self::Nano => NAME_MOSS_TTS_NANO,
            Self::Local => NAME_MOSS_TTS_LOCAL,
            Self::VoiceGenerator => NAME_MOSS_VOICE_GENERATOR,
            Self::AudioInstruct4b => NAME_MOSS_AUDIO_4B_INSTRUCT,
            Self::AudioInstruct8b => NAME_MOSS_AUDIO_8B_INSTRUCT,
        }
    }

    /// Upstream HF repository slug for this variant.
    pub(crate) const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Delay => UPSTREAM_HF_MOSS_TTS,
            Self::DelayV15 => UPSTREAM_HF_MOSS_TTS_V15,
            Self::Nano => UPSTREAM_HF_MOSS_TTS_NANO,
            Self::Local => UPSTREAM_HF_MOSS_TTS_LOCAL,
            Self::VoiceGenerator => UPSTREAM_HF_MOSS_VOICE_GENERATOR,
            Self::AudioInstruct4b => UPSTREAM_HF_MOSS_AUDIO_4B_INSTRUCT,
            Self::AudioInstruct8b => UPSTREAM_HF_MOSS_AUDIO_8B_INSTRUCT,
        }
    }

    pub(crate) const fn upstream_revision(self) -> Option<&'static str> {
        match self {
            Self::AudioInstruct4b => Some("6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d"),
            Self::AudioInstruct8b => Some("6521a39181b47a18f2d9f4b3acfb5bca7b76b57f"),
            _ => None,
        }
    }

    pub(crate) const fn config_sha256(self) -> Option<&'static str> {
        match self {
            Self::AudioInstruct4b => {
                Some("e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa")
            }
            Self::AudioInstruct8b => {
                Some("535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536")
            }
            _ => None,
        }
    }

    pub(crate) const fn tensor_manifest_sha256(self) -> Option<&'static str> {
        match self {
            Self::AudioInstruct4b => {
                Some("4db8bfa2a54b7541dc092b73919771fdefa952ea1b054ce10845e9d2bcd6fadc")
            }
            Self::AudioInstruct8b => {
                Some("76c1275dabd9a3baf0189f5fc335a6c192c472e96bc363cc3a64ad2d37a5f83a")
            }
            _ => None,
        }
    }

    const fn provenance_source(self) -> &'static str {
        match self {
            Self::Delay => "OpenMOSS-Team/MOSS-TTS (moss_tts_delay, Qwen3-8B backbone, apache-2.0)",
            Self::DelayV15 => {
                "OpenMOSS-Team/MOSS-TTS-v1.5 (moss_tts_delay, Qwen3-8B backbone, apache-2.0)"
            }
            Self::Nano => {
                "OpenMOSS-Team/MOSS-TTS-Nano-100M (moss_tts_nano, GPT-2 backbone, apache-2.0)"
            }
            Self::Local => {
                "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (moss_tts_local, Qwen3-2.5B backbone, apache-2.0)"
            }
            Self::VoiceGenerator => {
                "OpenMOSS-Team/MOSS-VoiceGenerator (moss_tts_delay, Qwen3-1.7B backbone, apache-2.0)"
            }
            Self::AudioInstruct4b => {
                "OpenMOSS-Team/MOSS-Audio-4B-Instruct@6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d (MossAudioModel, apache-2.0)"
            }
            Self::AudioInstruct8b => {
                "OpenMOSS-Team/MOSS-Audio-8B-Instruct@6521a39181b47a18f2d9f4b3acfb5bca7b76b57f (MossAudioModel, apache-2.0)"
            }
        }
    }

    /// Model category stamp for this variant. `tts` for the four
    /// `moss_tts_*` sibling releases; `s2s` for the audio-LLM
    /// [`Self::AudioInstruct4b`] sibling (matching the category the
    /// sibling audio-LLM converters `kimi_audio` / `baichuan_audio` /
    /// `step_audio2_mini` already stamp).
    pub(crate) const fn category(self) -> &'static str {
        match self {
            Self::Delay | Self::DelayV15 | Self::Nano | Self::Local | Self::VoiceGenerator => {
                MODEL_CATEGORY
            }
            Self::AudioInstruct4b | Self::AudioInstruct8b => MODEL_CATEGORY_S2S,
        }
    }

    /// Backbone family tag written under `vokra.moss_tts.llm.family`.
    pub(crate) const fn llm_family(self) -> &'static str {
        match self {
            Self::Delay
            | Self::DelayV15
            | Self::Local
            | Self::VoiceGenerator
            | Self::AudioInstruct4b
            | Self::AudioInstruct8b => "qwen3",
            Self::Nano => "gpt2",
        }
    }

    // Per-variant hparam selectors. Each routes to the transcribed
    // constants above; the selector method is a compile-time-checked
    // wall between the enum and the constants, so a hypothetical
    // future variant that reshapes an axis has to update the selector
    // (the compiler enforces total match coverage).

    pub(crate) const fn n_vq(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_N_VQ,
            Self::VoiceGenerator => VOICE_N_VQ,
            Self::Nano => NANO_N_VQ,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_N_VQ,
        }
    }
    pub(crate) const fn audio_vocab_size(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_AUDIO_VOCAB_SIZE,
            Self::VoiceGenerator => VOICE_AUDIO_VOCAB_SIZE,
            Self::Nano => NANO_AUDIO_VOCAB_SIZE,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_AUDIO_VOCAB_SIZE,
        }
    }
    pub(crate) const fn sample_rate(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_SAMPLE_RATE,
            Self::VoiceGenerator => VOICE_SAMPLE_RATE,
            Self::Nano => NANO_SAMPLE_RATE,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_SAMPLE_RATE,
        }
    }
    pub(crate) const fn llm_hidden_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_HIDDEN,
            Self::VoiceGenerator => VOICE_LLM_HIDDEN,
            Self::Nano => NANO_LLM_HIDDEN,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_HIDDEN,
        }
    }
    pub(crate) const fn llm_ffn_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_FFN,
            Self::VoiceGenerator => VOICE_LLM_FFN,
            Self::Nano => NANO_LLM_FFN,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_FFN,
        }
    }
    pub(crate) const fn llm_n_layer(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_LAYER,
            Self::VoiceGenerator => VOICE_LLM_N_LAYER,
            Self::Nano => NANO_LLM_N_LAYER,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_N_LAYER,
        }
    }
    pub(crate) const fn llm_n_head(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_HEAD,
            Self::VoiceGenerator => VOICE_LLM_N_HEAD,
            Self::Nano => NANO_LLM_N_HEAD,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_N_HEAD,
        }
    }
    pub(crate) const fn llm_n_head_kv(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_HEAD_KV,
            Self::VoiceGenerator => VOICE_LLM_N_HEAD_KV,
            Self::Nano => NANO_LLM_N_HEAD_KV,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_N_HEAD_KV,
        }
    }
    pub(crate) const fn llm_head_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_HEAD_DIM,
            Self::VoiceGenerator => VOICE_LLM_HEAD_DIM,
            Self::Nano => NANO_LLM_HEAD_DIM,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_HEAD_DIM,
        }
    }
    pub(crate) const fn llm_vocab_size(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_VOCAB,
            Self::VoiceGenerator => VOICE_LLM_VOCAB,
            Self::Nano => NANO_LLM_VOCAB,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_VOCAB,
        }
    }
    pub(crate) const fn llm_rope_base(self) -> f32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_ROPE_BASE,
            Self::VoiceGenerator => VOICE_LLM_ROPE_BASE,
            Self::Nano => NANO_LLM_ROPE_BASE,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_ROPE_BASE,
        }
    }
    pub(crate) const fn llm_rms_norm_eps(self) -> f32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_RMS_NORM_EPS,
            Self::VoiceGenerator => VOICE_LLM_RMS_NORM_EPS,
            Self::Nano => NANO_LLM_RMS_NORM_EPS,
            Self::Local | Self::AudioInstruct4b | Self::AudioInstruct8b => LOCAL_LLM_RMS_NORM_EPS,
        }
    }
}

// ─── Report ──────────────────────────────────────────────────────────

/// Outcome of a MOSS-TTS conversion.
///
/// Mirrors [`crate::models::wespeaker::WespeakerReport`]'s counter set
/// (leading `read` count of tensors observed in the input header +
/// float pass-through + BF16 subset counter + non-float defensive
/// counter). `read == written + skipped_non_float` is an invariant.
#[derive(Debug, Default)]
pub struct MossTtsReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling `wespeaker` /
    /// `qwen3_tts` / `vibevoice` / `voxcpm2` / `neucodec` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]).
    pub bf16_passthrough: usize,
}

// ─── convert / convert_variant / convert_moss_tts_file ───────────────

/// Byte-based converter — used by tests and by the file-based helper.
///
/// The `input` bytes must be a valid safetensors buffer.
pub(crate) fn convert_variant(
    bytes: Vec<u8>,
    variant: MossTtsVariant,
) -> Result<(GgufBuilder, MossTtsReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    if variant.is_audio_instruct() {
        validate_moss_audio_manifest(&st, variant)?;
    }

    let mut b = metadata_builder(variant);
    let mut report = MossTtsReport::default();
    for t in st.tensors() {
        report.read += 1;
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
    Ok((b, report))
}

pub(crate) fn metadata_builder(variant: MossTtsVariant) -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, variant.arch());
    builder.add_string(chunks::KEY_MODEL_NAME, variant.name());
    builder.add_string(KEY_MODEL_CATEGORY, variant.category());
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    write_hparams(&mut builder, variant);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(variant.name()),
        Some(variant.provenance_source()),
    );
    builder
}

/// File-based MOSS-TTS converter (`vokra-cli convert --model moss-tts[-*]`).
///
/// Reads `input`, writes a Vokra GGUF to `output`. `license` overrides
/// the default `apache-2.0` provenance stamp (Whisper / kokoro-family
/// override pattern — see `convert_file_licensed` in `lib.rs`); pass
/// `None` to keep the built-in `apache-2.0` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_moss_tts_file(
    input: &Path,
    output: &Path,
    variant: MossTtsVariant,
    license: Option<&str>,
) -> Result<MossTtsReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (mut builder, report) = convert_variant(bytes, variant)?;
    // Apply the caller-supplied SPDX override on top of the built-in
    // apache-2.0 stamp (mirror of `convert_wespeaker_file`). We
    // re-stamp by writing the override provenance chunk over the
    // existing one — `stamp_provenance` is additive so calling it
    // twice replaces the earlier value in serialisation order.
    if let Some(spdx) = license.filter(|s| !s.is_empty()) {
        let class = LicenseClass::from_license_str(spdx);
        vokra_core::stamp_provenance(
            &mut builder,
            class,
            spdx,
            Some(variant.name()),
            Some(match variant {
                MossTtsVariant::Delay => "OpenMOSS-Team/MOSS-TTS",
                MossTtsVariant::DelayV15 => "OpenMOSS-Team/MOSS-TTS-v1.5",
                MossTtsVariant::Nano => "OpenMOSS-Team/MOSS-TTS-Nano-100M",
                MossTtsVariant::Local => "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
                MossTtsVariant::VoiceGenerator => "OpenMOSS-Team/MOSS-VoiceGenerator",
                MossTtsVariant::AudioInstruct4b => "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
                MossTtsVariant::AudioInstruct8b => "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
            }),
        );
    }

    let out_bytes = builder
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

// ─── Internal: write hparams for the selected variant ────────────────

fn write_hparams(b: &mut GgufBuilder, variant: MossTtsVariant) {
    if variant.is_audio_instruct() {
        write_moss_audio_hparams(b, variant);
        return;
    }
    b.add_string(KEY_MOSS_VARIANT, variant.sub_arch());
    b.add_u32(KEY_MOSS_N_VQ, variant.n_vq());
    b.add_u32(KEY_MOSS_AUDIO_VOCAB_SIZE, variant.audio_vocab_size());
    b.add_u32(KEY_MOSS_SAMPLE_RATE, variant.sample_rate());
    b.add_string(KEY_MOSS_LLM_FAMILY, variant.llm_family());
    b.add_u32(KEY_MOSS_LLM_HIDDEN_DIM, variant.llm_hidden_dim());
    b.add_u32(KEY_MOSS_LLM_FFN_DIM, variant.llm_ffn_dim());
    b.add_u32(KEY_MOSS_LLM_N_LAYER, variant.llm_n_layer());
    b.add_u32(KEY_MOSS_LLM_N_HEAD, variant.llm_n_head());
    b.add_u32(KEY_MOSS_LLM_N_HEAD_KV, variant.llm_n_head_kv());
    b.add_u32(KEY_MOSS_LLM_HEAD_DIM, variant.llm_head_dim());
    b.add_u32(KEY_MOSS_LLM_VOCAB_SIZE, variant.llm_vocab_size());
    b.add_f32(KEY_MOSS_LLM_ROPE_BASE, variant.llm_rope_base());
    b.add_f32(KEY_MOSS_LLM_RMS_NORM_EPS, variant.llm_rms_norm_eps());
    if variant == MossTtsVariant::Nano {
        b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, NANO_UPSTREAM_REVISION);
        b.add_string(KEY_PROVENANCE_CHECKPOINT_SHA256, NANO_CHECKPOINT_SHA256);
        b.add_string(KEY_MOSS_CONFIG_SHA256, NANO_CONFIG_SHA256);
        b.add_string(
            KEY_MOSS_POSITION_EMBEDDING_TYPE,
            NANO_POSITION_EMBEDDING_TYPE,
        );
        b.add_f32(KEY_MOSS_LAYER_NORM_EPS, NANO_LAYER_NORM_EPS);
        b.add_u32(
            KEY_MOSS_MAX_POSITION_EMBEDDINGS,
            NANO_MAX_POSITION_EMBEDDINGS,
        );
        b.add_u32(
            KEY_MOSS_LOCAL_TRANSFORMER_LAYERS,
            NANO_LOCAL_TRANSFORMER_LAYERS,
        );
        b.add_u32(KEY_MOSS_PAD_TOKEN_ID, NANO_PAD_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_START_TOKEN_ID, NANO_IM_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_END_TOKEN_ID, NANO_IM_END_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_START_TOKEN_ID, NANO_AUDIO_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_END_TOKEN_ID, NANO_AUDIO_END_TOKEN_ID);
        b.add_u32(
            KEY_MOSS_AUDIO_USER_SLOT_TOKEN_ID,
            NANO_AUDIO_USER_SLOT_TOKEN_ID,
        );
        b.add_u32(
            KEY_MOSS_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
            NANO_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
        );
        b.add_u32(KEY_MOSS_AUDIO_PAD_TOKEN_ID, NANO_AUDIO_PAD_TOKEN_ID);
        b.add_string(
            KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM,
            NANO_AUDIO_TOKENIZER_UPSTREAM,
        );
    } else if variant == MossTtsVariant::Local {
        b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, LOCAL_UPSTREAM_REVISION);
        b.add_string(KEY_MOSS_CONFIG_SHA256, LOCAL_CONFIG_SHA256);
        b.add_string(
            KEY_MOSS_CONFIGURATION_SOURCE_SHA256,
            LOCAL_CONFIGURATION_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_MODELING_SOURCE_SHA256,
            LOCAL_MODELING_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_PROCESSING_SOURCE_SHA256,
            LOCAL_PROCESSING_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_QWEN3_DECODER_SOURCE_SHA256,
            LOCAL_QWEN3_DECODER_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_GPT2_DECODER_SOURCE_SHA256,
            LOCAL_GPT2_DECODER_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_PROCESSOR_CONFIG_SHA256,
            LOCAL_PROCESSOR_CONFIG_SHA256,
        );
        b.add_string(
            KEY_MOSS_POSITION_EMBEDDING_TYPE,
            LOCAL_POSITION_EMBEDDING_TYPE,
        );
        b.add_u32(
            KEY_MOSS_MAX_POSITION_EMBEDDINGS,
            LOCAL_MAX_POSITION_EMBEDDINGS,
        );
        b.add_u32(KEY_MOSS_LOCAL_TRANSFORMER_LAYERS, LOCAL_TRANSFORMER_LAYERS);
        b.add_u32(KEY_MOSS_LOCAL_HIDDEN_DIM, LOCAL_TRANSFORMER_HIDDEN);
        b.add_u32(KEY_MOSS_LOCAL_FFN_DIM, LOCAL_TRANSFORMER_FFN);
        b.add_u32(KEY_MOSS_LOCAL_N_HEAD, LOCAL_TRANSFORMER_N_HEAD);
        b.add_u32(KEY_MOSS_LOCAL_HEAD_DIM, LOCAL_TRANSFORMER_HEAD_DIM);
        b.add_string(
            KEY_MOSS_LOCAL_POSITION_EMBEDDING_TYPE,
            LOCAL_TRANSFORMER_POSITION_EMBEDDING_TYPE,
        );
        b.add_f32(KEY_MOSS_LOCAL_ROPE_BASE, LOCAL_TRANSFORMER_ROPE_BASE);
        b.add_f32(
            KEY_MOSS_LOCAL_LAYER_NORM_EPS,
            LOCAL_TRANSFORMER_LAYER_NORM_EPS,
        );
        b.add_string(KEY_MOSS_LOCAL_ACTIVATION, LOCAL_TRANSFORMER_ACTIVATION);
        b.add_string(KEY_MOSS_LOCAL_TEXT_HEAD_MODE, LOCAL_TEXT_HEAD_MODE);
        b.add_bool(KEY_MOSS_LOCAL_STATIC_KV_CACHE, true);
        b.add_u32(KEY_MOSS_PAD_TOKEN_ID, LOCAL_PAD_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_START_TOKEN_ID, LOCAL_IM_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_END_TOKEN_ID, LOCAL_IM_END_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_START_TOKEN_ID, LOCAL_AUDIO_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_END_TOKEN_ID, LOCAL_AUDIO_END_TOKEN_ID);
        b.add_u32(
            KEY_MOSS_AUDIO_USER_SLOT_TOKEN_ID,
            LOCAL_AUDIO_USER_SLOT_TOKEN_ID,
        );
        b.add_u32(
            KEY_MOSS_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
            LOCAL_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
        );
        b.add_u32(
            KEY_MOSS_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
            LOCAL_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
        );
        b.add_u32(KEY_MOSS_AUDIO_PAD_TOKEN_ID, LOCAL_AUDIO_PAD_TOKEN_ID);
        b.add_string(
            KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM,
            LOCAL_AUDIO_TOKENIZER_UPSTREAM,
        );
    } else if variant == MossTtsVariant::VoiceGenerator {
        b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, VOICE_UPSTREAM_REVISION);
        b.add_string(KEY_MOSS_CONFIG_SHA256, VOICE_CONFIG_SHA256);
        b.add_string(
            KEY_MOSS_MODELING_SOURCE_SHA256,
            VOICE_MODELING_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_PROCESSING_SOURCE_SHA256,
            VOICE_PROCESSING_SOURCE_SHA256,
        );
        b.add_string(
            KEY_MOSS_POSITION_EMBEDDING_TYPE,
            VOICE_POSITION_EMBEDDING_TYPE,
        );
        b.add_u32(
            KEY_MOSS_MAX_POSITION_EMBEDDINGS,
            VOICE_MAX_POSITION_EMBEDDINGS,
        );
        b.add_u32(KEY_MOSS_PAD_TOKEN_ID, VOICE_PAD_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_START_TOKEN_ID, VOICE_IM_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_IM_END_TOKEN_ID, VOICE_IM_END_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_START_TOKEN_ID, VOICE_AUDIO_START_TOKEN_ID);
        b.add_u32(KEY_MOSS_AUDIO_END_TOKEN_ID, VOICE_AUDIO_END_TOKEN_ID);
        b.add_u32(
            KEY_MOSS_AUDIO_USER_SLOT_TOKEN_ID,
            VOICE_AUDIO_USER_SLOT_TOKEN_ID,
        );
        b.add_u32(
            KEY_MOSS_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
            VOICE_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
        );
        b.add_u32(
            KEY_MOSS_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
            VOICE_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
        );
        b.add_u32(KEY_MOSS_AUDIO_PAD_TOKEN_ID, VOICE_AUDIO_PAD_TOKEN_ID);
        b.add_string(
            KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM,
            VOICE_AUDIO_TOKENIZER_UPSTREAM,
        );
    }
}

fn write_moss_audio_hparams(b: &mut GgufBuilder, variant: MossTtsVariant) {
    let (text_hidden, text_ffn) = match variant {
        MossTtsVariant::AudioInstruct4b => (2_560, 9_728),
        MossTtsVariant::AudioInstruct8b => (4_096, 12_288),
        _ => unreachable!("MOSS-Audio metadata requires an audio-instruct variant"),
    };
    let upstream_revision = variant
        .upstream_revision()
        .expect("audio-instruct variants pin an upstream revision");
    let config_sha256 = variant
        .config_sha256()
        .expect("audio-instruct variants pin config.json");
    let manifest_sha256 = variant
        .tensor_manifest_sha256()
        .expect("audio-instruct variants pin the public manifest");

    b.add_string("vokra.provenance.upstream_revision", upstream_revision);
    b.add_string("vokra.moss_audio.variant", variant.sub_arch());
    b.add_string("vokra.moss_audio.source_revision", upstream_revision);
    b.add_string(
        "vokra.moss_audio.source_code_revision",
        MOSS_AUDIO_SOURCE_CODE_REVISION,
    );
    b.add_string("vokra.moss_audio.config_sha256", config_sha256);
    b.add_string(
        "vokra.moss_audio.configuration_source_sha256",
        MOSS_AUDIO_CONFIGURATION_SHA256,
    );
    b.add_string(
        "vokra.moss_audio.modeling_source_sha256",
        MOSS_AUDIO_MODELING_SHA256,
    );
    b.add_string(
        "vokra.moss_audio.processing_source_sha256",
        MOSS_AUDIO_PROCESSING_SHA256,
    );
    b.add_string("vokra.moss_audio.tensor_manifest_sha256", manifest_sha256);

    b.add_u32("vokra.moss_audio.audio.d_model", 1_280);
    b.add_u32("vokra.moss_audio.audio.output_dim", 1_280);
    b.add_u32("vokra.moss_audio.audio.n_mels", 128);
    b.add_u32("vokra.moss_audio.audio.n_layer", 32);
    b.add_u32("vokra.moss_audio.audio.n_head", 20);
    b.add_u32("vokra.moss_audio.audio.ffn_dim", 5_120);
    b.add_u32("vokra.moss_audio.audio.downsample_rate", 8);
    b.add_u32("vokra.moss_audio.audio.downsample_hidden_size", 480);
    b.add_u32("vokra.moss_audio.audio.attention_window_size", 100);
    b.add_u32("vokra.moss_audio.audio.max_source_positions", 1_500);
    b.add_u32("vokra.moss_audio.audio.n_window", 200);
    b.add_u32("vokra.moss_audio.audio.conv_chunksize", 64);
    b.add_f32("vokra.moss_audio.audio.layer_norm_eps", 1.0e-5);
    b.add_string("vokra.moss_audio.audio.activation", "gelu");
    add_u32_array(
        b,
        "vokra.moss_audio.audio.deepstack_layer_indexes",
        &[8, 16, 24],
    );

    b.add_u32("vokra.moss_audio.adapter_hidden_size", 8_192);
    b.add_u32("vokra.moss_audio.deepstack_num_inject_layers", 3);
    b.add_u32("vokra.moss_audio.text.hidden_size", text_hidden);
    b.add_u32("vokra.moss_audio.text.ffn_dim", text_ffn);
    b.add_u32("vokra.moss_audio.text.n_layer", 36);
    b.add_u32("vokra.moss_audio.text.n_head", 32);
    b.add_u32("vokra.moss_audio.text.n_head_kv", 8);
    b.add_u32("vokra.moss_audio.text.head_dim", 128);
    b.add_u32("vokra.moss_audio.text.max_position_embeddings", 40_960);
    b.add_u32("vokra.moss_audio.text.vocab_size", 151_936);
    b.add_f32("vokra.moss_audio.text.rope_theta", 1_000_000.0);
    b.add_f32("vokra.moss_audio.text.rms_norm_eps", 1.0e-6);
    b.add_bool("vokra.moss_audio.text.tie_word_embeddings", false);
    b.add_bool("vokra.moss_audio.text.attention_bias", false);
    b.add_u32("vokra.moss_audio.token.audio", 151_654);
    b.add_u32("vokra.moss_audio.token.audio_start", 151_669);
    b.add_u32("vokra.moss_audio.token.audio_end", 151_670);
    b.add_u32("vokra.moss_audio.token.bos", 151_643);
    b.add_u32("vokra.moss_audio.token.eos", 151_645);

    moss_audio_frontend_spec().write_into(b);
}

fn moss_audio_frontend_spec() -> FrontendSpec {
    FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: 8_000.0,
        n_mels: 128,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: 16_000,
    }
}

fn add_u32_array(builder: &mut GgufBuilder, key: &str, values: &[u32]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
        }),
    );
}

pub(crate) fn expected_moss_audio_manifest(variant: MossTtsVariant) -> BTreeMap<String, Vec<u64>> {
    let (text_hidden, text_ffn) = match variant {
        MossTtsVariant::AudioInstruct4b => (2_560_u64, 9_728_u64),
        MossTtsVariant::AudioInstruct8b => (4_096_u64, 12_288_u64),
        _ => return BTreeMap::new(),
    };
    let audio_hidden = 1_280_u64;
    let audio_ffn = 5_120_u64;
    let adapter_hidden = 8_192_u64;
    let query_width = 4_096_u64;
    let key_value_width = 1_024_u64;
    let mut tensors = BTreeMap::new();
    let mut insert = |name: String, shape: &[u64]| {
        let old = tensors.insert(name, shape.to_vec());
        debug_assert!(old.is_none());
    };

    insert("audio_encoder.conv1.weight".into(), &[480, 1, 3, 3]);
    insert("audio_encoder.conv1.bias".into(), &[480]);
    for index in 2..=3 {
        insert(
            format!("audio_encoder.conv{index}.weight"),
            &[480, 480, 3, 3],
        );
        insert(format!("audio_encoder.conv{index}.bias"), &[480]);
    }
    insert(
        "audio_encoder.stem_proj.weight".into(),
        &[audio_hidden, 7_680],
    );
    insert("audio_encoder.stem_proj.bias".into(), &[audio_hidden]);
    for layer in 0..32 {
        let prefix = format!("audio_encoder.layers.{layer}");
        insert(
            format!("{prefix}.self_attn_layer_norm.weight"),
            &[audio_hidden],
        );
        insert(
            format!("{prefix}.self_attn_layer_norm.bias"),
            &[audio_hidden],
        );
        for projection in ["q_proj", "v_proj"] {
            insert(
                format!("{prefix}.self_attn.{projection}.weight"),
                &[audio_hidden, audio_hidden],
            );
            insert(
                format!("{prefix}.self_attn.{projection}.bias"),
                &[audio_hidden],
            );
        }
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.self_attn.out_proj.weight"),
            &[audio_hidden, audio_hidden],
        );
        insert(format!("{prefix}.self_attn.out_proj.bias"), &[audio_hidden]);
        insert(format!("{prefix}.final_layer_norm.weight"), &[audio_hidden]);
        insert(format!("{prefix}.final_layer_norm.bias"), &[audio_hidden]);
        insert(format!("{prefix}.fc1.weight"), &[audio_ffn, audio_hidden]);
        insert(format!("{prefix}.fc1.bias"), &[audio_ffn]);
        insert(format!("{prefix}.fc2.weight"), &[audio_hidden, audio_ffn]);
        insert(format!("{prefix}.fc2.bias"), &[audio_hidden]);
    }
    insert("audio_encoder.layer_norm.weight".into(), &[audio_hidden]);
    insert("audio_encoder.layer_norm.bias".into(), &[audio_hidden]);

    for prefix in std::iter::once("audio_adapter".to_owned())
        .chain((0..3).map(|index| format!("deepstack_audio_merger_list.{index}")))
    {
        insert(
            format!("{prefix}.gate_proj.weight"),
            &[adapter_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.up_proj.weight"),
            &[adapter_hidden, audio_hidden],
        );
        insert(
            format!("{prefix}.down_proj.weight"),
            &[text_hidden, adapter_hidden],
        );
    }

    insert(
        "language_model.embed_tokens.weight".into(),
        &[151_936, text_hidden],
    );
    for layer in 0..36 {
        let prefix = format!("language_model.layers.{layer}");
        insert(format!("{prefix}.input_layernorm.weight"), &[text_hidden]);
        insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[query_width, text_hidden],
        );
        insert(format!("{prefix}.self_attn.q_norm.weight"), &[128]);
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[key_value_width, text_hidden],
        );
        insert(format!("{prefix}.self_attn.k_norm.weight"), &[128]);
        insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[key_value_width, text_hidden],
        );
        insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            &[text_hidden, query_width],
        );
        insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[text_hidden],
        );
        insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            &[text_ffn, text_hidden],
        );
        insert(
            format!("{prefix}.mlp.up_proj.weight"),
            &[text_ffn, text_hidden],
        );
        insert(
            format!("{prefix}.mlp.down_proj.weight"),
            &[text_hidden, text_ffn],
        );
    }
    insert("language_model.norm.weight".into(), &[text_hidden]);
    insert("lm_head.weight".into(), &[151_936, text_hidden]);
    debug_assert_eq!(tensors.len(), 901);
    tensors
}

fn validate_moss_audio_manifest(
    checkpoint: &SafetensorsFile,
    variant: MossTtsVariant,
) -> Result<(), ConvertError> {
    let expected = expected_moss_audio_manifest(variant);
    let observed: BTreeMap<_, _> = checkpoint
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect();
    if observed == expected {
        return Ok(());
    }
    let missing: Vec<_> = expected
        .keys()
        .filter(|name| !observed.contains_key(*name))
        .take(5)
        .cloned()
        .collect();
    let extra: Vec<_> = observed
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .take(5)
        .cloned()
        .collect();
    let wrong_shape: Vec<_> = expected
        .iter()
        .filter_map(|(name, shape)| {
            observed
                .get(name)
                .filter(|actual| *actual != shape)
                .map(|actual| format!("{name}: {actual:?} != {shape:?}"))
        })
        .take(5)
        .collect();
    Err(ConvertError::Parse(format!(
        "{}: strict MOSS-Audio manifest mismatch: found {} tensors, expected 901; missing={missing:?}, extra={extra:?}, wrong_shape={wrong_shape:?}",
        variant.name(),
        observed.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};

    // ─── Fixtures ────────────────────────────────────────────────────

    fn safetensors_one_f32(name: &str) -> Vec<u8> {
        // 6 elements × 4 bytes = 24 bytes payload.
        let header =
            format!(r#"{{"{name}":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}}}"#);
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
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
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
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

    fn round_trip(variant: MossTtsVariant) -> GgufFile {
        if variant.is_audio_instruct() {
            let mut builder = metadata_builder(variant);
            builder
                .add_tensor("embed.weight", GgmlType::F32, vec![2, 3], vec![0; 24])
                .expect("synthetic metadata fixture");
            return GgufFile::parse(builder.to_bytes().expect("serialize")).expect("parse");
        }
        let (builder, report) =
            convert_variant(safetensors_one_f32("embed.weight"), variant).expect("convert_variant");
        assert_eq!(report.written, 1, "F32 tensor must land on written arm");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.read, 1);
        let out = builder.to_bytes().expect("serialize");
        GgufFile::parse(out).expect("parse")
    }

    // ─── Constants pinned against primary source ─────────────────────

    #[test]
    fn delay_constants_match_primary_source() {
        // huggingface.co/OpenMOSS-Team/MOSS-TTS/raw/main/config.json
        // + huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5/raw/main/config.json
        // fetched 2026-07-30.
        assert_eq!(DELAY_N_VQ, 32);
        assert_eq!(DELAY_AUDIO_VOCAB_SIZE, 1024);
        assert_eq!(DELAY_SAMPLE_RATE, 24_000);
        // language_config (Qwen3-8B)
        assert_eq!(DELAY_LLM_HIDDEN, 4096);
        assert_eq!(DELAY_LLM_FFN, 12_288);
        assert_eq!(DELAY_LLM_N_LAYER, 36);
        assert_eq!(DELAY_LLM_N_HEAD, 32);
        assert_eq!(DELAY_LLM_N_HEAD_KV, 8);
        assert_eq!(DELAY_LLM_HEAD_DIM, 128);
        assert_eq!(DELAY_LLM_VOCAB, 155_648);
        assert!((DELAY_LLM_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
        assert!((DELAY_LLM_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
        // Compile-time algebra: GQA well-formedness + even head_dim.
        const _: () = {
            assert!(DELAY_LLM_N_HEAD % DELAY_LLM_N_HEAD_KV == 0);
            assert!(DELAY_LLM_HEAD_DIM % 2 == 0);
        };
    }

    #[test]
    fn nano_constants_match_primary_source() {
        // huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/raw/main/config.json
        // fetched 2026-07-30.
        assert_eq!(NANO_N_VQ, 16);
        assert_eq!(NANO_AUDIO_VOCAB_SIZE, 1024);
        assert_eq!(NANO_SAMPLE_RATE, 48_000); // audio_tokenizer_sample_rate
        assert_eq!(NANO_LLM_HIDDEN, 768); // top-level hidden_size
        assert_eq!(NANO_LLM_FFN, 3072); // 4 * 768 (GPT-2 default n_inner)
        assert_eq!(NANO_LLM_N_LAYER, 12); // gpt2_config.n_layer
        assert_eq!(NANO_LLM_N_HEAD, 12); // gpt2_config.n_head
        assert_eq!(NANO_LLM_N_HEAD_KV, 12); // MHA (== n_head)
        assert_eq!(NANO_LLM_HEAD_DIM, 64); // 768 / 12
        assert_eq!(NANO_LLM_VOCAB, 16_384);
        // The custom GPT-2 uses RoPE but retains LayerNorm.
        assert_eq!(NANO_LLM_ROPE_BASE, 10_000.0);
        assert_eq!(NANO_LLM_RMS_NORM_EPS, 0.0);
        assert_eq!(NANO_POSITION_EMBEDDING_TYPE, "rope");
        assert!((NANO_LAYER_NORM_EPS - 1e-5).abs() < 1e-12);
        assert_eq!(NANO_MAX_POSITION_EMBEDDINGS, 32_768);
        assert_eq!(NANO_LOCAL_TRANSFORMER_LAYERS, 1);
    }

    #[test]
    fn local_constants_match_primary_source() {
        // huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5/raw/main/config.json
        // fetched 2026-07-30.
        assert_eq!(LOCAL_N_VQ, 12);
        assert_eq!(LOCAL_AUDIO_VOCAB_SIZE, 1024);
        assert_eq!(LOCAL_SAMPLE_RATE, 48_000);
        assert_eq!(LOCAL_LLM_HIDDEN, 2560);
        assert_eq!(LOCAL_LLM_FFN, 9728);
        assert_eq!(LOCAL_LLM_N_LAYER, 36);
        assert_eq!(LOCAL_LLM_N_HEAD, 32);
        assert_eq!(LOCAL_LLM_N_HEAD_KV, 8);
        assert_eq!(LOCAL_LLM_HEAD_DIM, 128);
        assert_eq!(LOCAL_LLM_VOCAB, 151_936);
        assert!((LOCAL_LLM_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
        assert!((LOCAL_LLM_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn voice_generator_constants_match_primary_source() {
        // OpenMOSS-Team/MOSS-VoiceGenerator config.json at
        // 97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4, fetched 2026-08-26.
        assert_eq!(VOICE_N_VQ, 16);
        assert_eq!(VOICE_AUDIO_VOCAB_SIZE, 1_024);
        assert_eq!(VOICE_SAMPLE_RATE, 24_000);
        assert_eq!(VOICE_LLM_HIDDEN, 2_048);
        assert_eq!(VOICE_LLM_FFN, 6_144);
        assert_eq!(VOICE_LLM_N_LAYER, 28);
        assert_eq!(VOICE_LLM_N_HEAD, 16);
        assert_eq!(VOICE_LLM_N_HEAD_KV, 8);
        assert_eq!(VOICE_LLM_HEAD_DIM, 128);
        assert_eq!(VOICE_LLM_VOCAB, 155_648);
        assert_eq!(VOICE_LLM_ROPE_BASE, 1_000_000.0);
        assert!((VOICE_LLM_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
    }

    // ─── Variant selectors ───────────────────────────────────────────

    #[test]
    fn variant_sub_arch_returns_config_json_suffix() {
        // Must match upstream config.json.model_type suffix after
        // "moss_tts_".
        assert_eq!(MossTtsVariant::Delay.sub_arch(), "delay");
        assert_eq!(MossTtsVariant::DelayV15.sub_arch(), "delay");
        assert_eq!(MossTtsVariant::Nano.sub_arch(), "nano");
        assert_eq!(MossTtsVariant::Local.sub_arch(), "local");
        assert_eq!(MossTtsVariant::VoiceGenerator.sub_arch(), "voice_generator");
    }

    #[test]
    fn variant_name_stamps_are_distinct_and_lowercase() {
        // Each variant carries a unique NAME; a runtime that ships the
        // GGUFs side-by-side must be able to tell them apart. The four
        // sibling `moss_tts_*` variants start with `moss-tts`; the
        // audio-LLM sibling [`MossTtsVariant::AudioInstruct4b`] starts
        // with `moss-audio` (distinct HF release family — reuses this
        // converter per the parent workflow's REUSE HINT but keeps
        // its own upstream identity in the emitted GGUF).
        let names = [
            MossTtsVariant::Delay.name(),
            MossTtsVariant::DelayV15.name(),
            MossTtsVariant::Nano.name(),
            MossTtsVariant::Local.name(),
            MossTtsVariant::VoiceGenerator.name(),
            MossTtsVariant::AudioInstruct4b.name(),
            MossTtsVariant::AudioInstruct8b.name(),
        ];
        for n in names.iter() {
            assert_eq!(n.to_ascii_lowercase(), *n, "NAME must be lower-case: {n}");
            assert!(
                n.starts_with("moss-tts")
                    || n.starts_with("moss-audio-")
                    || n.starts_with("moss-voice-"),
                "NAME must start with moss-tts, moss-audio-, or moss-voice-: {n}"
            );
        }
        // Distinctness: 7 variants, 7 unique names.
        let mut seen: Vec<&str> = names.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 7, "every variant must have a unique NAME");
    }

    #[test]
    fn variant_upstream_hf_slugs_preserve_upstream_casing() {
        // The provenance stamp must reproduce the HF slug verbatim so
        // a downstream can trace the artifact.
        assert_eq!(
            MossTtsVariant::Delay.upstream_hf(),
            "OpenMOSS-Team/MOSS-TTS"
        );
        assert_eq!(
            MossTtsVariant::DelayV15.upstream_hf(),
            "OpenMOSS-Team/MOSS-TTS-v1.5"
        );
        assert_eq!(
            MossTtsVariant::Nano.upstream_hf(),
            "OpenMOSS-Team/MOSS-TTS-Nano-100M"
        );
        assert_eq!(
            MossTtsVariant::Local.upstream_hf(),
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
        );
        assert_eq!(
            MossTtsVariant::VoiceGenerator.upstream_hf(),
            "OpenMOSS-Team/MOSS-VoiceGenerator"
        );
    }

    #[test]
    fn variant_llm_family_discriminates_qwen3_from_gpt2() {
        assert_eq!(MossTtsVariant::Delay.llm_family(), "qwen3");
        assert_eq!(MossTtsVariant::DelayV15.llm_family(), "qwen3");
        assert_eq!(MossTtsVariant::Nano.llm_family(), "gpt2");
        assert_eq!(MossTtsVariant::Local.llm_family(), "qwen3");
        assert_eq!(MossTtsVariant::VoiceGenerator.llm_family(), "qwen3");
    }

    #[test]
    fn delay_and_delayv15_share_every_selector_except_name() {
        // MOSS-TTS and MOSS-TTS-v1.5 share axes byte-for-byte and
        // differ only in training-data language coverage + NAME
        // stamp. Every selector method must agree.
        let d = MossTtsVariant::Delay;
        let v = MossTtsVariant::DelayV15;
        assert_eq!(d.sub_arch(), v.sub_arch());
        assert_eq!(d.llm_family(), v.llm_family());
        assert_eq!(d.n_vq(), v.n_vq());
        assert_eq!(d.audio_vocab_size(), v.audio_vocab_size());
        assert_eq!(d.sample_rate(), v.sample_rate());
        assert_eq!(d.llm_hidden_dim(), v.llm_hidden_dim());
        assert_eq!(d.llm_ffn_dim(), v.llm_ffn_dim());
        assert_eq!(d.llm_n_layer(), v.llm_n_layer());
        assert_eq!(d.llm_n_head(), v.llm_n_head());
        assert_eq!(d.llm_n_head_kv(), v.llm_n_head_kv());
        assert_eq!(d.llm_head_dim(), v.llm_head_dim());
        assert_eq!(d.llm_vocab_size(), v.llm_vocab_size());
        assert_eq!(d.llm_rope_base(), v.llm_rope_base());
        assert_eq!(d.llm_rms_norm_eps(), v.llm_rms_norm_eps());
        // But NAME differs.
        assert_ne!(d.name(), v.name());
    }

    // ─── Round-trip GGUF emission per variant ────────────────────────

    #[test]
    fn delay_round_trip_emits_qwen3_8b_axes() {
        let file = round_trip(MossTtsVariant::Delay);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_TTS)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_MOSS_TTS)
        );
        assert_eq!(
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("delay")
        );
        assert_eq!(
            file.get(KEY_MOSS_LLM_FAMILY).and_then(|v| v.as_str()),
            Some("qwen3")
        );
        assert_eq!(get_u32(&file, KEY_MOSS_N_VQ), DELAY_N_VQ);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM), DELAY_LLM_HIDDEN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_FFN_DIM), DELAY_LLM_FFN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_N_LAYER), DELAY_LLM_N_LAYER);
        assert!((get_f32(&file, KEY_MOSS_LLM_ROPE_BASE) - DELAY_LLM_ROPE_BASE).abs() < 1e-3);
        // Provenance: apache-2.0 permissive.
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
        // Tensor written verbatim.
        let info = file.tensor_info("embed.weight").expect("F32 present");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![2, 3]);
    }

    #[test]
    fn nano_round_trip_emits_gpt2_axes_and_family_sentinel() {
        let file = round_trip(MossTtsVariant::Nano);
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_TTS_NANO)
        );
        assert_eq!(
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("nano")
        );
        assert_eq!(
            file.get(KEY_MOSS_LLM_FAMILY).and_then(|v| v.as_str()),
            Some("gpt2")
        );
        assert_eq!(get_u32(&file, KEY_MOSS_N_VQ), NANO_N_VQ);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM), NANO_LLM_HIDDEN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_N_LAYER), NANO_LLM_N_LAYER);
        // Custom GPT-2 RoPE plus LayerNorm (not RMSNorm).
        assert_eq!(get_f32(&file, KEY_MOSS_LLM_ROPE_BASE), 10_000.0);
        assert_eq!(get_f32(&file, KEY_MOSS_LLM_RMS_NORM_EPS), 0.0);
        assert_eq!(
            file.get(KEY_MOSS_POSITION_EMBEDDING_TYPE)
                .and_then(|v| v.as_str()),
            Some("rope")
        );
        assert!((get_f32(&file, KEY_MOSS_LAYER_NORM_EPS) - 1e-5).abs() < 1e-12);
        assert_eq!(get_u32(&file, KEY_MOSS_MAX_POSITION_EMBEDDINGS), 32_768);
        assert_eq!(get_u32(&file, KEY_MOSS_LOCAL_TRANSFORMER_LAYERS), 1);
        assert_eq!(get_u32(&file, KEY_MOSS_AUDIO_PAD_TOKEN_ID), 1_024);
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(|v| v.as_str()),
            Some(NANO_UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM)
                .and_then(|v| v.as_str()),
            Some(NANO_AUDIO_TOKENIZER_UPSTREAM)
        );
        // MHA — n_head == n_head_kv.
        assert_eq!(
            get_u32(&file, KEY_MOSS_LLM_N_HEAD),
            get_u32(&file, KEY_MOSS_LLM_N_HEAD_KV)
        );
    }

    #[test]
    fn local_round_trip_emits_qwen3_2_5b_axes() {
        let file = round_trip(MossTtsVariant::Local);
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_TTS_LOCAL)
        );
        assert_eq!(
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("local")
        );
        assert_eq!(
            file.get(KEY_MOSS_LLM_FAMILY).and_then(|v| v.as_str()),
            Some("qwen3")
        );
        assert_eq!(get_u32(&file, KEY_MOSS_N_VQ), LOCAL_N_VQ);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM), LOCAL_LLM_HIDDEN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_FFN_DIM), LOCAL_LLM_FFN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_VOCAB_SIZE), LOCAL_LLM_VOCAB);
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(GgufMetadataValue::as_str),
            Some(LOCAL_UPSTREAM_REVISION)
        );
        for (key, expected) in [
            (KEY_MOSS_CONFIG_SHA256, LOCAL_CONFIG_SHA256),
            (
                KEY_MOSS_CONFIGURATION_SOURCE_SHA256,
                LOCAL_CONFIGURATION_SOURCE_SHA256,
            ),
            (
                KEY_MOSS_MODELING_SOURCE_SHA256,
                LOCAL_MODELING_SOURCE_SHA256,
            ),
            (
                KEY_MOSS_PROCESSING_SOURCE_SHA256,
                LOCAL_PROCESSING_SOURCE_SHA256,
            ),
            (
                KEY_MOSS_QWEN3_DECODER_SOURCE_SHA256,
                LOCAL_QWEN3_DECODER_SOURCE_SHA256,
            ),
            (
                KEY_MOSS_GPT2_DECODER_SOURCE_SHA256,
                LOCAL_GPT2_DECODER_SOURCE_SHA256,
            ),
            (
                KEY_MOSS_PROCESSOR_CONFIG_SHA256,
                LOCAL_PROCESSOR_CONFIG_SHA256,
            ),
        ] {
            assert_eq!(
                file.get(key).and_then(GgufMetadataValue::as_str),
                Some(expected),
                "fixed Local provenance key {key}"
            );
        }
        assert_eq!(
            get_u32(&file, KEY_MOSS_MAX_POSITION_EMBEDDINGS),
            LOCAL_MAX_POSITION_EMBEDDINGS
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_LOCAL_TRANSFORMER_LAYERS),
            LOCAL_TRANSFORMER_LAYERS
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_LOCAL_HIDDEN_DIM),
            LOCAL_TRANSFORMER_HIDDEN
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_LOCAL_FFN_DIM),
            LOCAL_TRANSFORMER_FFN
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_LOCAL_N_HEAD),
            LOCAL_TRANSFORMER_N_HEAD
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_LOCAL_HEAD_DIM),
            LOCAL_TRANSFORMER_HEAD_DIM
        );
        assert_eq!(
            get_f32(&file, KEY_MOSS_LOCAL_ROPE_BASE),
            LOCAL_TRANSFORMER_ROPE_BASE
        );
        assert_eq!(
            get_f32(&file, KEY_MOSS_LOCAL_LAYER_NORM_EPS),
            LOCAL_TRANSFORMER_LAYER_NORM_EPS
        );
        assert_eq!(
            file.get(KEY_MOSS_LOCAL_TEXT_HEAD_MODE)
                .and_then(GgufMetadataValue::as_str),
            Some(LOCAL_TEXT_HEAD_MODE)
        );
        assert_eq!(
            file.get(KEY_MOSS_LOCAL_STATIC_KV_CACHE)
                .and_then(GgufMetadataValue::as_bool),
            Some(true)
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_AUDIO_START_TOKEN_ID),
            LOCAL_AUDIO_START_TOKEN_ID
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_AUDIO_END_TOKEN_ID),
            LOCAL_AUDIO_END_TOKEN_ID
        );
        assert_eq!(
            file.get(KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM)
                .and_then(GgufMetadataValue::as_str),
            Some(LOCAL_AUDIO_TOKENIZER_UPSTREAM)
        );
    }

    #[test]
    fn voice_generator_round_trip_emits_1_7b_axes_and_identity() {
        let file = round_trip(MossTtsVariant::VoiceGenerator);
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_VOICE_GENERATOR)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_MOSS_VOICE_GENERATOR)
        );
        assert_eq!(
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("voice_generator")
        );
        assert_eq!(get_u32(&file, KEY_MOSS_N_VQ), VOICE_N_VQ);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM), VOICE_LLM_HIDDEN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_FFN_DIM), VOICE_LLM_FFN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_N_LAYER), VOICE_LLM_N_LAYER);
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(|value| value.as_str()),
            Some(VOICE_UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_MOSS_CONFIG_SHA256)
                .and_then(|value| value.as_str()),
            Some(VOICE_CONFIG_SHA256)
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_MAX_POSITION_EMBEDDINGS),
            VOICE_MAX_POSITION_EMBEDDINGS
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID),
            VOICE_AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
        );
        assert_eq!(
            get_u32(&file, KEY_MOSS_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID),
            VOICE_AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID
        );
        assert_eq!(
            file.get(KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM)
                .and_then(|value| value.as_str()),
            Some(VOICE_AUDIO_TOKENIZER_UPSTREAM)
        );
        assert_ne!(
            get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM),
            DELAY_LLM_HIDDEN,
            "VoiceGenerator must never inherit the 8B Delay axes"
        );
    }

    #[test]
    fn audio_instruct_4b_round_trip_stamps_audio_llm_provenance() {
        // MOSS-Audio-4B-Instruct shares only the pass-through writer with
        // MOSS-TTS; metadata and runtime topology stay on `moss_audio`.
        // The provenance triple (NAME + upstream_hf + license = Permissive)
        // + category = `s2s` + the dedicated `moss_audio` metadata group are
        // the invariants a runtime dispatcher relies on to route this
        // artifact away from the MOSS-TTS family.
        let file = round_trip(MossTtsVariant::AudioInstruct4b);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(MOSS_AUDIO_ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_AUDIO_4B_INSTRUCT)
        );
        // Category diverges from the sibling tts variants — this
        // release is an audio-LLM, matching kimi_audio / baichuan_audio /
        // step_audio2_mini which all stamp `s2s`.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY_S2S)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_MOSS_AUDIO_4B_INSTRUCT)
        );
        // Distinct sub-arch tag lets a downstream dispatcher tell this
        // sibling apart from the four tts variants.
        assert_eq!(
            file.get("vokra.moss_audio.variant")
                .and_then(|v| v.as_str()),
            Some("4b_instruct")
        );
        assert_eq!(
            file.get("vokra.moss_audio.source_revision")
                .and_then(|v| v.as_str()),
            Some("6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d")
        );
        assert_eq!(get_u32(&file, "vokra.moss_audio.text.hidden_size"), 2_560);
        assert_eq!(get_u32(&file, "vokra.moss_audio.audio.n_layer"), 32);
        // Provenance license: apache-2.0 Permissive.
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
        // Sibling tts variants keep their `tts` category unchanged.
        for tts_variant in [
            MossTtsVariant::Delay,
            MossTtsVariant::DelayV15,
            MossTtsVariant::Nano,
            MossTtsVariant::Local,
            MossTtsVariant::VoiceGenerator,
        ] {
            let sibling = round_trip(tts_variant);
            assert_eq!(
                sibling.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
                Some(MODEL_CATEGORY),
                "{tts_variant:?}: tts category must be preserved byte-for-byte"
            );
        }
        // Tensor written verbatim.
        let info = file.tensor_info("embed.weight").expect("F32 present");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![2, 3]);
    }

    #[test]
    fn delay_v15_round_trip_stamps_v15_name_but_shares_delay_axes() {
        let file = round_trip(MossTtsVariant::DelayV15);
        // NAME is the v1.5 stamp.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME_MOSS_TTS_V15)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF_MOSS_TTS_V15)
        );
        // Sub-arch still "delay" (both `moss_tts_delay`).
        assert_eq!(
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("delay")
        );
        // Axes identical to the base `Delay` variant.
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_HIDDEN_DIM), DELAY_LLM_HIDDEN);
        assert_eq!(get_u32(&file, KEY_MOSS_LLM_FFN_DIM), DELAY_LLM_FFN);
    }

    // ─── BF16 pass-through ───────────────────────────────────────────

    #[test]
    fn bf16_pass_through_works_for_every_variant() {
        // Real MOSS-TTS releases ship BF16 (Delay + v1.5 + Local +
        // MOSS-Audio-4B-Instruct — 3 shards ~8 GB BF16 per parent
        // task manifest 2026-08-02).
        // Every variant must land BF16 on the pass-through arm with
        // byte-identical payload.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input = safetensors_one_bf16("weight", &[2, 3], &bf16);

        for variant in [
            MossTtsVariant::Delay,
            MossTtsVariant::DelayV15,
            MossTtsVariant::Nano,
            MossTtsVariant::Local,
            MossTtsVariant::VoiceGenerator,
        ] {
            let (builder, report) = convert_variant(input.clone(), variant).expect("BF16 convert");
            assert_eq!(report.read, 1);
            assert_eq!(report.written, 1);
            assert_eq!(report.skipped_non_float, 0);
            assert_eq!(report.bf16_passthrough, 1);
            let out = builder.to_bytes().expect("serialize");
            let file = GgufFile::parse(out).expect("parse");
            let info = file.tensor_info("weight").expect("BF16 tensor present");
            assert_eq!(info.dtype, GgmlType::BF16, "{variant:?}: BF16 stays BF16");
            assert_eq!(
                file.tensor_bytes(info),
                bf16.as_slice(),
                "{variant:?}: BF16 payload must be byte-identical"
            );
        }
    }

    #[test]
    fn read_written_skipped_invariant_holds_on_a_mixed_input() {
        // F32 + BF16 + F16 side by side — the invariant
        // `read == written + skipped_non_float` must hold for every
        // variant.
        let f32_bytes: Vec<u8> = [7.0f32, -8.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = [1.0f32, -2.5, 0.15625, 3.5, -0.5, 42.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = [0x3C00u16, 0x4000, 0x4200]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"a":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"b":{{"dtype":"BF16","shape":[2,3],"data_offsets":[{},{}]}},"c":{{"dtype":"F16","shape":[3],"data_offsets":[{},{}]}}}}"#,
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

        for variant in [
            MossTtsVariant::Delay,
            MossTtsVariant::Nano,
            MossTtsVariant::Local,
            MossTtsVariant::VoiceGenerator,
        ] {
            let (_, report) = convert_variant(input.clone(), variant).expect("mixed convert");
            assert_eq!(
                report.read,
                report.written + report.skipped_non_float,
                "{variant:?}: read == written + skipped invariant"
            );
            assert_eq!(report.written, 3, "{variant:?}: three floats");
            assert_eq!(report.bf16_passthrough, 1, "{variant:?}: one BF16");
        }
    }

    #[test]
    fn moss_audio_manifests_match_range_audited_public_headers() {
        for variant in [
            MossTtsVariant::AudioInstruct4b,
            MossTtsVariant::AudioInstruct8b,
        ] {
            let manifest = expected_moss_audio_manifest(variant);
            assert_eq!(manifest.len(), 901);
            let digest = crate::models::canary_1b_flash::manifest_sha256(&manifest);
            let digest = digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(digest, variant.tensor_manifest_sha256().unwrap());
        }
    }

    #[test]
    fn moss_audio_converter_rejects_a_count_only_placeholder() {
        let input = safetensors_one_f32("embed.weight");
        let error = convert_variant(input, MossTtsVariant::AudioInstruct4b)
            .expect_err("MOSS-Audio requires the complete 901-tensor contract");
        let ConvertError::Parse(message) = error else {
            panic!("expected strict manifest parse error, got {error:?}");
        };
        assert!(message.contains("expected 901"));
        assert!(message.contains("missing="));
    }

    // ─── Errors ─────────────────────────────────────────────────────

    #[test]
    fn malformed_input_returns_parse_error() {
        // Empty buffer.
        for variant in [MossTtsVariant::Delay, MossTtsVariant::Nano] {
            let err = convert_variant(Vec::new(), variant).expect_err("empty buffer");
            assert!(
                matches!(err, ConvertError::Parse(_)),
                "expected ConvertError::Parse, got {err:?}"
            );
        }
    }

    // ─── Metadata namespace hygiene ─────────────────────────────────

    #[test]
    fn every_moss_tts_metadata_key_is_namespaced() {
        // Guard: any typo that broke the namespace (e.g.
        // `vokra.moss.n_vq`) would misroute in downstream dispatchers.
        for key in [
            KEY_MOSS_VARIANT,
            KEY_MOSS_N_VQ,
            KEY_MOSS_AUDIO_VOCAB_SIZE,
            KEY_MOSS_SAMPLE_RATE,
            KEY_MOSS_LLM_FAMILY,
            KEY_MOSS_LLM_HIDDEN_DIM,
            KEY_MOSS_LLM_FFN_DIM,
            KEY_MOSS_LLM_N_LAYER,
            KEY_MOSS_LLM_N_HEAD,
            KEY_MOSS_LLM_N_HEAD_KV,
            KEY_MOSS_LLM_HEAD_DIM,
            KEY_MOSS_LLM_VOCAB_SIZE,
            KEY_MOSS_LLM_ROPE_BASE,
            KEY_MOSS_LLM_RMS_NORM_EPS,
            KEY_MOSS_CONFIG_SHA256,
            KEY_MOSS_POSITION_EMBEDDING_TYPE,
            KEY_MOSS_LAYER_NORM_EPS,
            KEY_MOSS_MAX_POSITION_EMBEDDINGS,
            KEY_MOSS_LOCAL_TRANSFORMER_LAYERS,
            KEY_MOSS_PAD_TOKEN_ID,
            KEY_MOSS_IM_START_TOKEN_ID,
            KEY_MOSS_IM_END_TOKEN_ID,
            KEY_MOSS_AUDIO_START_TOKEN_ID,
            KEY_MOSS_AUDIO_END_TOKEN_ID,
            KEY_MOSS_AUDIO_USER_SLOT_TOKEN_ID,
            KEY_MOSS_AUDIO_ASSISTANT_SLOT_TOKEN_ID,
            KEY_MOSS_AUDIO_PAD_TOKEN_ID,
            KEY_MOSS_AUDIO_TOKENIZER_UPSTREAM,
        ] {
            assert!(
                key.starts_with("vokra.moss_tts."),
                "{key} must live under the vokra.moss_tts.* prefix"
            );
        }
    }
}
