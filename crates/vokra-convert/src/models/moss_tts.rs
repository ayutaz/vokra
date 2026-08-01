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
//!   `gpt2_config.n_positions = 32_768` / vocab = 16_384;
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
//! re-implemented natively in a future `crates/vokra-models/src/moss_tts/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Vast.ai
//!
//! The Delay variants (`MOSS-TTS` + `MOSS-TTS-v1.5`, both ~17 GB BF16
//! across 4 safetensors shards) and the Local variant (~9 GB BF16) all
//! exceed the M1 iMac 16 GB dev machine's whole-file
//! `std::fs::read` capacity per memory
//! [[feedback-large-models-on-vast-ai]] (>8 GB safetensors → vast.ai).
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

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

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
/// MOSS-Audio-4B-Instruct HF release slug
/// (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0). Distinct
/// upstream sibling that shares the moss_tts family arch tag (this
/// converter is reused per the parent workflow's REUSE HINT), but is
/// a 4B audio-LLM custom-code release
/// (`configuration_moss_audio.py`, `trust_remote_code=True`) rather
/// than one of the four `moss_tts_*` (Delay / Nano / Local) tts
/// releases.
pub(crate) const UPSTREAM_HF_MOSS_AUDIO_4B_INSTRUCT: &str = "OpenMOSS-Team/MOSS-Audio-4B-Instruct";

/// Per-variant `vokra.model.name` stamps (canonical, lower-cased HF slug
/// tail — mirrors the Qwen3-TTS / Chatterbox naming convention).
pub(crate) const NAME_MOSS_TTS: &str = "moss-tts";
pub(crate) const NAME_MOSS_TTS_V15: &str = "moss-tts-v1.5";
pub(crate) const NAME_MOSS_TTS_NANO: &str = "moss-tts-nano-100m";
pub(crate) const NAME_MOSS_TTS_LOCAL: &str = "moss-tts-local-transformer-v1.5";
/// `vokra.model.name` stamp for MOSS-Audio-4B-Instruct — the
/// lower-cased HF slug tail, matching the sibling naming convention.
pub(crate) const NAME_MOSS_AUDIO_4B_INSTRUCT: &str = "moss-audio-4b-instruct";

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
/// Delay + Local. GPT-2 (Nano) uses **learned** positional embeddings
/// and therefore has no RoPE base; the Nano key is written as `0`
/// sentinel so a runtime binder can tell "not applicable" apart from
/// "silently defaulted" (FR-EX-08).
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
/// GPT-2 uses learned positional embeddings — no RoPE base. Sentinel
/// `0.0` so the runtime binder can tell "N/A" apart from "silently
/// defaulted" (FR-EX-08).
const NANO_LLM_ROPE_BASE: f32 = 0.0;
/// GPT-2 uses LayerNorm — no RMSNorm ε. Sentinel `0.0`.
const NANO_LLM_RMS_NORM_EPS: f32 = 0.0;

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
    /// `OpenMOSS-Team/MOSS-Audio-4B-Instruct` — a distinct 4B
    /// **audio-LLM** sibling (custom `configuration_moss_audio.py`
    /// module, `trust_remote_code=True`), reusing this converter per
    /// the parent workflow's REUSE HINT rather than a fresh
    /// `models/*.rs` module.
    ///
    /// **PLACEHOLDER HPARAMS**: The 4B model is not one of the four
    /// `moss_tts_{delay,nano,local}` releases whose axes are
    /// primary-source-transcribed here. The selector methods route
    /// AudioInstruct4b to the sibling `Local` (Qwen3-flavour 2.5B)
    /// axes as the closest-family placeholder while the parent-workflow
    /// task discipline forbids downloading the ~8 GB safetensors + the
    /// upstream `configuration_moss_audio.py` for transcription. A
    /// follow-up wave must land the true axes (config.json +
    /// `configuration_moss_audio.py` inspection) before any downstream
    /// loader can trust the emitted hparams. The **provenance** stamp
    /// (NAME + upstream_hf + license = apache-2.0 Permissive +
    /// category = `s2s`) is faithful — only the axis hparams are
    /// placeholder. The distinct `vokra.moss_tts.variant = "audio_4b"`
    /// tag lets a runtime dispatcher recognise this artifact and
    /// refuse to bind the placeholder axes until the follow-up lands.
    AudioInstruct4b,
}

impl MossTtsVariant {
    /// Sub-arch tag written under `vokra.moss_tts.variant`.
    pub(crate) const fn sub_arch(self) -> &'static str {
        match self {
            Self::Delay | Self::DelayV15 => "delay",
            Self::Nano => "nano",
            Self::Local => "local",
            // Distinct tag so a runtime dispatcher can distinguish the
            // audio-LLM sibling from the four `moss_tts_*` tts releases
            // and refuse to bind placeholder axes.
            Self::AudioInstruct4b => "audio_4b",
        }
    }

    /// `vokra.model.name` stamp for this variant.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Delay => NAME_MOSS_TTS,
            Self::DelayV15 => NAME_MOSS_TTS_V15,
            Self::Nano => NAME_MOSS_TTS_NANO,
            Self::Local => NAME_MOSS_TTS_LOCAL,
            Self::AudioInstruct4b => NAME_MOSS_AUDIO_4B_INSTRUCT,
        }
    }

    /// Upstream HF repository slug for this variant.
    pub(crate) const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Delay => UPSTREAM_HF_MOSS_TTS,
            Self::DelayV15 => UPSTREAM_HF_MOSS_TTS_V15,
            Self::Nano => UPSTREAM_HF_MOSS_TTS_NANO,
            Self::Local => UPSTREAM_HF_MOSS_TTS_LOCAL,
            Self::AudioInstruct4b => UPSTREAM_HF_MOSS_AUDIO_4B_INSTRUCT,
        }
    }

    /// Model category stamp for this variant. `tts` for the four
    /// `moss_tts_*` sibling releases; `s2s` for the audio-LLM
    /// [`Self::AudioInstruct4b`] sibling (matching the category the
    /// sibling audio-LLM converters `kimi_audio` / `baichuan_audio` /
    /// `step_audio2_mini` already stamp).
    pub(crate) const fn category(self) -> &'static str {
        match self {
            Self::Delay | Self::DelayV15 | Self::Nano | Self::Local => MODEL_CATEGORY,
            Self::AudioInstruct4b => MODEL_CATEGORY_S2S,
        }
    }

    /// Backbone family tag written under `vokra.moss_tts.llm.family`.
    pub(crate) const fn llm_family(self) -> &'static str {
        match self {
            Self::Delay | Self::DelayV15 | Self::Local | Self::AudioInstruct4b => "qwen3",
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
            Self::Nano => NANO_N_VQ,
            Self::Local | Self::AudioInstruct4b => LOCAL_N_VQ,
        }
    }
    pub(crate) const fn audio_vocab_size(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_AUDIO_VOCAB_SIZE,
            Self::Nano => NANO_AUDIO_VOCAB_SIZE,
            Self::Local | Self::AudioInstruct4b => LOCAL_AUDIO_VOCAB_SIZE,
        }
    }
    pub(crate) const fn sample_rate(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_SAMPLE_RATE,
            Self::Nano => NANO_SAMPLE_RATE,
            Self::Local | Self::AudioInstruct4b => LOCAL_SAMPLE_RATE,
        }
    }
    pub(crate) const fn llm_hidden_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_HIDDEN,
            Self::Nano => NANO_LLM_HIDDEN,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_HIDDEN,
        }
    }
    pub(crate) const fn llm_ffn_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_FFN,
            Self::Nano => NANO_LLM_FFN,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_FFN,
        }
    }
    pub(crate) const fn llm_n_layer(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_LAYER,
            Self::Nano => NANO_LLM_N_LAYER,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_N_LAYER,
        }
    }
    pub(crate) const fn llm_n_head(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_HEAD,
            Self::Nano => NANO_LLM_N_HEAD,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_N_HEAD,
        }
    }
    pub(crate) const fn llm_n_head_kv(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_N_HEAD_KV,
            Self::Nano => NANO_LLM_N_HEAD_KV,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_N_HEAD_KV,
        }
    }
    pub(crate) const fn llm_head_dim(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_HEAD_DIM,
            Self::Nano => NANO_LLM_HEAD_DIM,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_HEAD_DIM,
        }
    }
    pub(crate) const fn llm_vocab_size(self) -> u32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_VOCAB,
            Self::Nano => NANO_LLM_VOCAB,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_VOCAB,
        }
    }
    pub(crate) const fn llm_rope_base(self) -> f32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_ROPE_BASE,
            Self::Nano => NANO_LLM_ROPE_BASE,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_ROPE_BASE,
        }
    }
    pub(crate) const fn llm_rms_norm_eps(self) -> f32 {
        match self {
            Self::Delay | Self::DelayV15 => DELAY_LLM_RMS_NORM_EPS,
            Self::Nano => NANO_LLM_RMS_NORM_EPS,
            Self::Local | Self::AudioInstruct4b => LOCAL_LLM_RMS_NORM_EPS,
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

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, variant.category());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    write_hparams(&mut b, variant);
    // Self-describing redistribution: the artifact carries its own
    // licence. Every MOSS-TTS release ships `apache-2.0`
    // (`huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS[-v1.5|
    // -Nano-100M|-Local-Transformer-v1.5]` `cardData.license =
    // "apache-2.0"` fetched 2026-07-30 — CLAUDE.md
    // 「ハルシネーション厳禁」).
    let source = match variant {
        MossTtsVariant::Delay => {
            "OpenMOSS-Team/MOSS-TTS (moss_tts_delay, Qwen3-8B backbone, apache-2.0)"
        }
        MossTtsVariant::DelayV15 => {
            "OpenMOSS-Team/MOSS-TTS-v1.5 (moss_tts_delay, Qwen3-8B backbone, apache-2.0)"
        }
        MossTtsVariant::Nano => {
            "OpenMOSS-Team/MOSS-TTS-Nano-100M (moss_tts_nano, GPT-2 backbone, apache-2.0)"
        }
        MossTtsVariant::Local => {
            "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (moss_tts_local, Qwen3-2.5B backbone, apache-2.0)"
        }
        MossTtsVariant::AudioInstruct4b => {
            "OpenMOSS-Team/MOSS-Audio-4B-Instruct (configuration_moss_audio.py custom-code audio-LLM, apache-2.0; placeholder axes = Local family)"
        }
    };
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(variant.name()),
        Some(source),
    );

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
                MossTtsVariant::AudioInstruct4b => "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
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
        // GPT-2 sentinels for RoPE + RMSNorm (learned pos + LayerNorm).
        assert_eq!(NANO_LLM_ROPE_BASE, 0.0);
        assert_eq!(NANO_LLM_RMS_NORM_EPS, 0.0);
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

    // ─── Variant selectors ───────────────────────────────────────────

    #[test]
    fn variant_sub_arch_returns_config_json_suffix() {
        // Must match upstream config.json.model_type suffix after
        // "moss_tts_".
        assert_eq!(MossTtsVariant::Delay.sub_arch(), "delay");
        assert_eq!(MossTtsVariant::DelayV15.sub_arch(), "delay");
        assert_eq!(MossTtsVariant::Nano.sub_arch(), "nano");
        assert_eq!(MossTtsVariant::Local.sub_arch(), "local");
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
            MossTtsVariant::AudioInstruct4b.name(),
        ];
        for n in names.iter() {
            assert_eq!(n.to_ascii_lowercase(), *n, "NAME must be lower-case: {n}");
            assert!(
                n.starts_with("moss-tts") || n.starts_with("moss-audio-"),
                "NAME must start with moss-tts or moss-audio-: {n}"
            );
        }
        // Distinctness: 5 variants, 5 unique names.
        let mut seen: Vec<&str> = names.to_vec();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 5, "every variant must have a unique NAME");
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
    }

    #[test]
    fn variant_llm_family_discriminates_qwen3_from_gpt2() {
        assert_eq!(MossTtsVariant::Delay.llm_family(), "qwen3");
        assert_eq!(MossTtsVariant::DelayV15.llm_family(), "qwen3");
        assert_eq!(MossTtsVariant::Nano.llm_family(), "gpt2");
        assert_eq!(MossTtsVariant::Local.llm_family(), "qwen3");
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
        // GPT-2 sentinels for RoPE + RMSNorm.
        assert_eq!(get_f32(&file, KEY_MOSS_LLM_ROPE_BASE), 0.0);
        assert_eq!(get_f32(&file, KEY_MOSS_LLM_RMS_NORM_EPS), 0.0);
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
    }

    #[test]
    fn audio_instruct_4b_round_trip_stamps_audio_llm_provenance() {
        // MOSS-Audio-4B-Instruct reuses this converter per the parent
        // workflow's REUSE HINT (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`,
        // apache-2.0, 3 shards ~8 GB BF16, custom_code=True via
        // configuration_moss_audio.py — parent task manifest 2026-08-02).
        // The provenance triple (NAME + upstream_hf + license = Permissive)
        // + category = `s2s` + `vokra.moss_tts.variant = "audio_4b"` sub-arch
        // tag are the invariants a runtime dispatcher relies on to route
        // this artifact away from the four `moss_tts_*` tts variants and
        // refuse to bind the placeholder axes until the follow-up wave
        // lands the primary-source hparam transcription.
        let file = round_trip(MossTtsVariant::AudioInstruct4b);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
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
            file.get(KEY_MOSS_VARIANT).and_then(|v| v.as_str()),
            Some("audio_4b")
        );
        assert_eq!(
            file.get(KEY_MOSS_LLM_FAMILY).and_then(|v| v.as_str()),
            Some("qwen3")
        );
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
            MossTtsVariant::AudioInstruct4b,
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
            MossTtsVariant::AudioInstruct4b,
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
        ] {
            assert!(
                key.starts_with("vokra.moss_tts."),
                "{key} must live under the vokra.moss_tts.* prefix"
            );
        }
    }
}
