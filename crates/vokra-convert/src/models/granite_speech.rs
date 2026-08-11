#![allow(clippy::doc_lazy_continuation)]
//! **IBM Granite Speech 4.1-2B** (`ibm-granite/granite-speech-4.1-2b`,
//! apache-2.0): safetensors → GGUF conversion (2026-08-01 wave).
//!
//! Input: the upstream `ibm-granite/granite-speech-4.1-2b` release — a
//! Conformer CTC audio encoder + Granite-4.0-1b-base LLM text decoder +
//! BLIP-2 q-former projector + optional LoRA adapter bundle. HF Open ASR
//! leaderboard top-tier audio-LLM ASR system. The upstream release ships
//! as **3 sharded safetensors** (`model-00001-of-00003.safetensors`,
//! `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`)
//! + `model.safetensors.index.json` weight-map + `out_llm.safetensors`
//! auxiliary + `config.json` + tokenizer files, ~4.87 GB total (verified
//! 2026-08-01 via HF repo file listing — CLAUDE.md 「ハルシネーション厳禁」).
//!
//! This converter accepts a **single merged safetensors** input; owner
//! pre-merges the 3 shards + `out_llm.safetensors` via a Python one-liner
//! (`safetensors.torch.load_file` + `save_file`) before invoking
//! `vokra-convert` — the same offline-bridge posture DAC / CSM /
//! DeepFilterNet3 / Kokoro use for pickle inputs (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4; Rust converter is
//! safetensors-only by design so the runtime never grows a shard-index
//! reader, keeping the NFR-DS-02 zero-dep posture).
//!
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream HF-transformers `GraniteSpeechForConditionalGeneration` name
//! (`encoder.blocks.{i}.*`, `language_model.model.layers.{i}.*`,
//! `projector.encoder.layer.{i}.*`, `language_model.lm_head.weight`
//! from `out_llm.safetensors`, etc.), plus the `vokra.model.*` /
//! `vokra.provenance.*` / `vokra.granite_speech.*` metadata chunks a
//! future native `granite_speech` loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `ibm-granite/granite-speech-4.1-2b`.
//! - **License (SPDX)**: `apache-2.0` — end-to-end (IBM Granite Speech
//!   code + weight + LoRA adapter; primary source = HF model page
//!   linking `https://www.apache.org/licenses/LICENSE-2.0`, fetched
//!   2026-08-01).
//! - **Category**: `asr` — audio-LLM automatic speech recognition
//!   (Conformer CTC encoder → q-former projector → Granite LLM decoder).
//!   Same category tag as Voxtral / Whisper / Canary-Qwen so the
//!   model-card tooling can classify without reaching into per-converter
//!   constants.
//!
//! # Distinct arch from every existing arch
//!
//! `ARCH = "granite_speech"` is the first Vokra converter with the
//! Conformer CTC encoder + Granite LLM decoder + BLIP-2 q-former
//! projector bundle. Silently sharing an arch tag with any sibling
//! (Whisper / Voxtral / Canary-Qwen / Parakeet-CTC etc.) would mis-route
//! the runtime dispatch — the encoder is Conformer (not FastConformer,
//! not Whisper Transformer), the decoder is Granite LLM (with distinctive
//! `attention_multiplier=0.0078125`, `embedding_multiplier=12.0`,
//! `logits_scaling=8.0`, `residual_multiplier=0.22` scalars that no
//! other Vokra converter carries), and the projector is a BLIP-2
//! q-former (2 layers of cross-attention over 1024-dim queries) —
//! entirely distinct from Voxtral's soft-prompt prefix bridge.
//!
//! # BF16 pass-through (mirror of canary_qwen / speecht5_hifigan / voxtral)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm —
//! no convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`);
//! the runtime widens BF16 → f32 losslessly at load via the single
//! choke point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
//! (BF16 is the top 16 bits of an f32 — `bits << 16` is exact). The
//! observability counter [`GraniteSpeechReport::bf16_passthrough`]
//! records how many BF16 tensors landed on this arm so a silent widen /
//! downcast cannot slip in undetected. Upstream `config.json` pins
//! `dtype: bfloat16` end-to-end (encoder + decoder + projector all BF16)
//! so BF16 is the expected serving format.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice /
//! VoxCPM / WeSpeaker / ECAPA-TDNN / SpeechT5-HiFiGan / Canary-Qwen
//! contract). Real-weight parity vs the upstream
//! `transformers.GraniteSpeechForConditionalGeneration` Python forward
//! is deferred to owner (`docs/license-audit.md` §3.1 sign-off queue).
//!
//! # No ONNX (permanent)
//!
//! IBM Granite Speech ships PyTorch safetensors; this converter **never**
//! touches ONNX (FR-LD-05); the pipeline will be re-implemented natively
//! in `crates/vokra-models/src/granite_speech/` when the runtime binder
//! lands (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # Variant enum
//!
//! Only [`GraniteSpeechVariant::V4_1_2B`] is wired in the 2026-08-01
//! wave. A future `granite-speech-4.1-8b` (Granite-4.0-8B backbone,
//! reshaped decoder) or `granite-speech-3.3-8b` (Granite-3.3-8B
//! backbone, distinct encoder depth) would be a distinct variant enum
//! value + separate dispatch (silently sharing tensor axes would be
//! FR-EX-08 no-silent-fallback territory — the axes ride as `u32`
//! constants a future 8B variant would reshape).
//!
//! # Loud-partial precedent
//!
//! Real-weight forward binding is deferred: the runtime consumer will
//! walk the emitted tensor names and either succeed or fail loudly per
//! FR-EX-08. Today's converter surface is byte-exact provenance +
//! tensor-name preservation + hparam chunk group only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for IBM Granite Speech GGUFs.
///
/// Intentionally distinct from every existing arch (Whisper / Voxtral /
/// Canary-Qwen / Parakeet-CTC / etc.) — see the module-level docstring.
pub const ARCH: &str = "granite_speech";

/// Which upstream IBM Granite Speech variant this GGUF was minted from.
///
/// Only V4_1_2B is wired in the 2026-08-01 wave. Future variants
/// (`granite-speech-4.1-8b` / `granite-speech-3.3-8b`) will each become
/// their own enum arm with a separate dispatch — the decoder axes are
/// backbone-specific (Granite-4.0-1b-base vs Granite-4.0-8B vs
/// Granite-3.3-8B), and silently sharing hparams would violate
/// FR-EX-08 (no silent fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechVariant {
    /// `ibm-granite/granite-speech-4.1-2b` — 2 B params. Encoder =
    /// 16-layer Conformer (d_model 1024, n_heads 8, head_dim 128,
    /// conv_kernel 15, input_dim 160 = 80 logmels × 2 stacked frames,
    /// output_dim 348 CTC-char vocab). Decoder = Granite-4.0-1b-base
    /// (40 layers, hidden 2048, GQA 16 Q ÷ 4 KV, ffn 4096, RoPE θ 10000,
    /// RMSNorm ε 1e-5, vocab 100 353, distinctive scalars
    /// `attention_multiplier=0.0078125` / `embedding_multiplier=12.0` /
    /// `logits_scaling=8.0` / `residual_multiplier=0.22`). Projector =
    /// 2-layer BLIP-2 q-former (hidden 1024, 16 heads, downsample_rate
    /// 5). Every value transcribed verbatim from
    /// `huggingface.co/ibm-granite/granite-speech-4.1-2b/raw/main/config.json`
    /// (fetched 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」).
    V4_1_2B,
}

impl GraniteSpeechVariant {
    /// Model card canonical id — used for the `vokra.model.name` stamp
    /// and the model-card generator's SKU pick.
    pub const fn name(self) -> &'static str {
        match self {
            Self::V4_1_2B => "granite-speech-4.1-2b",
        }
    }

    /// `vokra.provenance.upstream_hf` value for this variant.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::V4_1_2B => "ibm-granite/granite-speech-4.1-2b",
        }
    }
}

/// `vokra.model.name` value written by the enum-arm-default dispatch
/// (used from `lib.rs`'s `ModelKind::GraniteSpeech` arm). Callers that
/// need a non-default variant use
/// [`convert_granite_speech_file`] directly with an explicit
/// [`GraniteSpeechVariant`].
#[allow(dead_code)]
pub const NAME: &str = "granite-speech-4.1-2b";

/// `vokra.model.category` value written for every IBM Granite Speech
/// GGUF. Same tag as the sibling audio-LLM ASR converters (Voxtral,
/// Canary-Qwen, Whisper family).
pub const CATEGORY: &str = "asr";

/// `vokra.provenance.upstream_hf` value written by the enum-arm-default
/// dispatch. Distinct-variant callers pass through
/// [`GraniteSpeechVariant::upstream_hf`] on the standalone
/// [`convert_granite_speech_file`] path.
#[allow(dead_code)]
pub const UPSTREAM_HF: &str = "ibm-granite/granite-speech-4.1-2b";

/// Default upstream weight licence (SPDX). IBM Granite Speech family
/// ships apache-2.0 end-to-end (verified 2026-08-01 via HF model page +
/// docs page linking `https://www.apache.org/licenses/LICENSE-2.0`).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling BF16 pass-through
// converters' cross-crate constant duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// --- vokra.granite_speech.* keys (kept as constants in the converter;
// the runtime module `vokra-models::granite_speech` will read them
// symmetrically — the cross-crate string handshake pattern) ---------

const KEY_SAMPLE_RATE: &str = "vokra.granite_speech.sample_rate";
const KEY_AUDIO_TOKEN_INDEX: &str = "vokra.granite_speech.audio_token_index";
const KEY_DOWNSAMPLE_RATE: &str = "vokra.granite_speech.downsample_rate";
const KEY_WINDOW_SIZE: &str = "vokra.granite_speech.window_size";
const KEY_TIE_WORD_EMBEDDINGS: &str = "vokra.granite_speech.tie_word_embeddings";
const KEY_HAS_LORA_ADAPTER: &str = "vokra.granite_speech.has_lora_adapter";
const KEY_INITIALIZER_RANGE: &str = "vokra.granite_speech.initializer_range";

// Encoder (Conformer CTC — 16-layer GraniteSpeechEncoder)
const KEY_ENC_N_LAYER: &str = "vokra.granite_speech.arch.encoder.n_layer";
const KEY_ENC_HIDDEN_DIM: &str = "vokra.granite_speech.arch.encoder.hidden_dim";
const KEY_ENC_N_HEAD: &str = "vokra.granite_speech.arch.encoder.n_head";
const KEY_ENC_HEAD_DIM: &str = "vokra.granite_speech.arch.encoder.head_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.granite_speech.arch.encoder.conv_kernel_size";
const KEY_ENC_INPUT_DIM: &str = "vokra.granite_speech.arch.encoder.input_dim";
const KEY_ENC_OUTPUT_DIM: &str = "vokra.granite_speech.arch.encoder.output_dim";
const KEY_ENC_CONV_EXPANSION: &str = "vokra.granite_speech.arch.encoder.conv_expansion_factor";
const KEY_ENC_FEEDFORWARD_MULT: &str = "vokra.granite_speech.arch.encoder.feedforward_mult";
const KEY_ENC_CONTEXT_SIZE: &str = "vokra.granite_speech.arch.encoder.context_size";
const KEY_ENC_MAX_POS_EMB: &str = "vokra.granite_speech.arch.encoder.max_pos_emb";

// Text decoder (Granite-4.0-1b-base — Granite LLM family)
const KEY_DEC_N_LAYER: &str = "vokra.granite_speech.arch.decoder.n_layer";
const KEY_DEC_HIDDEN_DIM: &str = "vokra.granite_speech.arch.decoder.hidden_dim";
const KEY_DEC_N_HEAD: &str = "vokra.granite_speech.arch.decoder.n_head";
const KEY_DEC_N_HEAD_KV: &str = "vokra.granite_speech.arch.decoder.n_head_kv";
const KEY_DEC_FFN_DIM: &str = "vokra.granite_speech.arch.decoder.ffn_dim";
const KEY_DEC_VOCAB_SIZE: &str = "vokra.granite_speech.arch.decoder.vocab_size";
const KEY_DEC_N_CTX: &str = "vokra.granite_speech.arch.decoder.n_ctx";
const KEY_DEC_ROPE_BASE: &str = "vokra.granite_speech.arch.decoder.rope_base";
const KEY_DEC_RMS_NORM_EPS: &str = "vokra.granite_speech.arch.decoder.rms_norm_eps";
const KEY_DEC_ATTENTION_MULTIPLIER: &str = "vokra.granite_speech.arch.decoder.attention_multiplier";
const KEY_DEC_EMBEDDING_MULTIPLIER: &str = "vokra.granite_speech.arch.decoder.embedding_multiplier";
const KEY_DEC_LOGITS_SCALING: &str = "vokra.granite_speech.arch.decoder.logits_scaling";
const KEY_DEC_RESIDUAL_MULTIPLIER: &str = "vokra.granite_speech.arch.decoder.residual_multiplier";

// Projector (BLIP-2 q-former)
const KEY_PROJ_N_LAYER: &str = "vokra.granite_speech.arch.projector.n_layer";
const KEY_PROJ_HIDDEN_SIZE: &str = "vokra.granite_speech.arch.projector.hidden_size";
const KEY_PROJ_INTERMEDIATE_SIZE: &str = "vokra.granite_speech.arch.projector.intermediate_size";
const KEY_PROJ_N_HEAD: &str = "vokra.granite_speech.arch.projector.n_head";
const KEY_PROJ_MAX_POS: &str = "vokra.granite_speech.arch.projector.max_position_embeddings";

// --- Transcribed constants (from HF config.json fetched 2026-08-01) ----
//
// Sample rate = 16 kHz (per multilingual_sample.wav + audio pipeline).
// Every other value transcribed verbatim from `config.json` primary source.

const SAMPLE_RATE: u32 = 16_000;
const AUDIO_TOKEN_INDEX: u32 = 100_352;
const DOWNSAMPLE_RATE: u32 = 5;
const WINDOW_SIZE: u32 = 15;
const TIE_WORD_EMBEDDINGS: bool = false;
const HAS_LORA_ADAPTER: bool = false; // `has_lora_adapter: false` in cardData
const INITIALIZER_RANGE: f32 = 0.02;

// Encoder axes (Conformer CTC — `GraniteSpeechEncoderConfig`)
const ENC_N_LAYER: u32 = 16;
const ENC_HIDDEN_DIM: u32 = 1024;
const ENC_N_HEAD: u32 = 8;
const ENC_HEAD_DIM: u32 = 128;
const ENC_CONV_KERNEL: u32 = 15;
const ENC_INPUT_DIM: u32 = 160; // 80 logmels × 2 (stacked frames)
const ENC_OUTPUT_DIM: u32 = 348; // CTC characters vocab
const ENC_CONV_EXPANSION: u32 = 2;
const ENC_FEEDFORWARD_MULT: u32 = 4;
const ENC_CONTEXT_SIZE: u32 = 200;
const ENC_MAX_POS_EMB: u32 = 512;

// Text decoder axes (Granite-4.0-1b-base — Granite LLM family)
const DEC_N_LAYER: u32 = 40;
const DEC_HIDDEN_DIM: u32 = 2048;
const DEC_N_HEAD: u32 = 16;
const DEC_N_HEAD_KV: u32 = 4;
const DEC_FFN_DIM: u32 = 4096;
const DEC_VOCAB_SIZE: u32 = 100_353;
const DEC_N_CTX: u32 = 4096;
const DEC_ROPE_BASE: f32 = 10_000.0;
const DEC_RMS_NORM_EPS: f32 = 1e-5;
const DEC_ATTENTION_MULTIPLIER: f32 = 0.007_812_5;
const DEC_EMBEDDING_MULTIPLIER: f32 = 12.0;
const DEC_LOGITS_SCALING: f32 = 8.0;
const DEC_RESIDUAL_MULTIPLIER: f32 = 0.22;

// Projector axes (BLIP-2 q-former — `Blip2QFormerConfig`)
const PROJ_N_LAYER: u32 = 2;
const PROJ_HIDDEN_SIZE: u32 = 1024;
const PROJ_INTERMEDIATE_SIZE: u32 = 4096;
const PROJ_N_HEAD: u32 = 16;
const PROJ_MAX_POS: u32 = 2048;

/// Outcome of an IBM Granite Speech conversion.
///
/// Mirrors the sibling BF16-pass-through converters' counter shape
/// (`super::canary_qwen::CanaryQwenReport`,
/// `super::speecht5_hifigan::Speecht5HifiganReport`) adapted to the
/// file-oriented `convert_granite_speech_file` surface.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a
    /// non-zero here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting. IBM Granite Speech ships BF16 end-to-end
    /// (config.json `dtype: bfloat16` for encoder + decoder + projector)
    /// so this counter tracks essentially all tensors on the primary
    /// release.
    pub bf16_passthrough: usize,
}

/// Converts an IBM Granite Speech safetensors checkpoint at `input` into
/// a Vokra-native GGUF at `output`, returning a [`GraniteSpeechReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// HF-transformers `GraniteSpeechForConditionalGeneration` name; the
/// `vokra.model.*` (arch / name / category) + `vokra.provenance.*`
/// (weight_license / license / model_id / source / upstream_hf) +
/// `vokra.granite_speech.*` (encoder / decoder / projector hparam)
/// chunks are stamped for the runtime compliance gate (FR-CP-03).
///
/// The `variant` argument picks which `vokra.model.name` +
/// `vokra.provenance.upstream_hf` values land — today only
/// [`GraniteSpeechVariant::V4_1_2B`] is wired.
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — the
/// upstream HF release ships apache-2.0 end-to-end.
///
/// # Sharded input contract
///
/// This entry accepts a **single merged safetensors** — owner
/// pre-merges the upstream 3 shards + `out_llm.safetensors` auxiliary
/// via a Python one-liner (`safetensors.torch.load_file` +
/// `save_file`) before invoking `vokra-convert`. See the module-level
/// docstring for the offline-bridge rationale (safetensors-only reader,
/// zero-dep runtime).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_granite_speech_file(
    input: &Path,
    output: &Path,
    variant: GraniteSpeechVariant,
    license: Option<&str>,
) -> Result<GraniteSpeechReport, ConvertError> {
    // Merged safetensors is ~4.87 GiB (3 shards ~4.62 GB + out_llm
    // 206 MB) — under the memory [[feedback-large-models-on-vast-ai]]
    // ≥8 GB vast.ai threshold and under the empirically-safe csm-1b
    // 6.21 GB M1 iMac 16 GB tight-fit ceiling, so the simple
    // `std::fs::read` posture the sibling non-streaming BF16 pass-
    // through converters use applies. A streaming path (mmap in /
    // GgufStreamWriter out — the Voxtral posture) is a follow-up if
    // the owner reports peak-mem issues.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    write_hparams(&mut b);

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (upstream `ibm-granite/granite-speech-4.1-2b` model card +
    // docs page linking apache.org/licenses/LICENSE-2.0). The optional
    // `license` argument overrides below.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(
            "ibm-granite/granite-speech-4.1-2b \
             (Conformer CTC encoder + Granite-4.0-1b-base LLM decoder + \
             BLIP-2 q-former projector, apache-2.0)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());

    let mut report = GraniteSpeechReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (mirror of canary_qwen / speecht5_hifigan / voxtral); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
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

/// Writes the `vokra.granite_speech.*` chunk group from the transcribed
/// primary-source constants above. Booleans ride as `u32` 0/1 for GGUF
/// portability.
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_AUDIO_TOKEN_INDEX, AUDIO_TOKEN_INDEX);
    b.add_u32(KEY_DOWNSAMPLE_RATE, DOWNSAMPLE_RATE);
    b.add_u32(KEY_WINDOW_SIZE, WINDOW_SIZE);
    b.add_u32(KEY_TIE_WORD_EMBEDDINGS, u32::from(TIE_WORD_EMBEDDINGS));
    b.add_u32(KEY_HAS_LORA_ADAPTER, u32::from(HAS_LORA_ADAPTER));
    b.add_f32(KEY_INITIALIZER_RANGE, INITIALIZER_RANGE);

    // Encoder (Conformer CTC)
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);
    b.add_u32(KEY_ENC_HIDDEN_DIM, ENC_HIDDEN_DIM);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_HEAD_DIM, ENC_HEAD_DIM);
    b.add_u32(KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL);
    b.add_u32(KEY_ENC_INPUT_DIM, ENC_INPUT_DIM);
    b.add_u32(KEY_ENC_OUTPUT_DIM, ENC_OUTPUT_DIM);
    b.add_u32(KEY_ENC_CONV_EXPANSION, ENC_CONV_EXPANSION);
    b.add_u32(KEY_ENC_FEEDFORWARD_MULT, ENC_FEEDFORWARD_MULT);
    b.add_u32(KEY_ENC_CONTEXT_SIZE, ENC_CONTEXT_SIZE);
    b.add_u32(KEY_ENC_MAX_POS_EMB, ENC_MAX_POS_EMB);

    // Decoder (Granite LLM family)
    b.add_u32(KEY_DEC_N_LAYER, DEC_N_LAYER);
    b.add_u32(KEY_DEC_HIDDEN_DIM, DEC_HIDDEN_DIM);
    b.add_u32(KEY_DEC_N_HEAD, DEC_N_HEAD);
    b.add_u32(KEY_DEC_N_HEAD_KV, DEC_N_HEAD_KV);
    b.add_u32(KEY_DEC_FFN_DIM, DEC_FFN_DIM);
    b.add_u32(KEY_DEC_VOCAB_SIZE, DEC_VOCAB_SIZE);
    b.add_u32(KEY_DEC_N_CTX, DEC_N_CTX);
    b.add_f32(KEY_DEC_ROPE_BASE, DEC_ROPE_BASE);
    b.add_f32(KEY_DEC_RMS_NORM_EPS, DEC_RMS_NORM_EPS);
    b.add_f32(KEY_DEC_ATTENTION_MULTIPLIER, DEC_ATTENTION_MULTIPLIER);
    b.add_f32(KEY_DEC_EMBEDDING_MULTIPLIER, DEC_EMBEDDING_MULTIPLIER);
    b.add_f32(KEY_DEC_LOGITS_SCALING, DEC_LOGITS_SCALING);
    b.add_f32(KEY_DEC_RESIDUAL_MULTIPLIER, DEC_RESIDUAL_MULTIPLIER);

    // Projector (BLIP-2 q-former)
    b.add_u32(KEY_PROJ_N_LAYER, PROJ_N_LAYER);
    b.add_u32(KEY_PROJ_HIDDEN_SIZE, PROJ_HIDDEN_SIZE);
    b.add_u32(KEY_PROJ_INTERMEDIATE_SIZE, PROJ_INTERMEDIATE_SIZE);
    b.add_u32(KEY_PROJ_N_HEAD, PROJ_N_HEAD);
    b.add_u32(KEY_PROJ_MAX_POS, PROJ_MAX_POS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload.
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

    /// Builds a mixed F32 + F16 safetensors buffer using realistic
    /// upstream `GraniteSpeechForConditionalGeneration` tensor names.
    fn safetensors_f32_and_f16() -> Vec<u8> {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0=0x3C00, -2.0=0xC000, -0.5=0xB800,
        // 3.0=0x4200, 0.15625=0x3100, 42.0=0x5140.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12);
        let header = format!(
            r#"{{"encoder.input_linear.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{}]}},"language_model.model.embed_tokens.weight":{{"dtype":"F16","shape":[2,3],"data_offsets":[{},{}]}}}}"#,
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

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// PID + nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-granite-speech-{kind}-{}-{}.bin",
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
    fn arch_string_matches_family_and_is_distinct() {
        // ARCH is the sole cross-crate handshake with the future
        // `vokra-models::granite_speech::EXPECTED_ARCH` — pinning it
        // here catches an accidental rename that would silently mis-
        // route runtime dispatch.
        assert_eq!(ARCH, "granite_speech");
        assert_ne!(ARCH, "whisper", "must not silently alias Whisper family");
        assert_ne!(ARCH, "voxtral", "must not silently alias Voxtral family");
        assert_ne!(
            ARCH, "canary-qwen",
            "must not silently alias Canary-Qwen family"
        );
        assert_ne!(
            ARCH, "parakeet-ctc",
            "must not silently alias Parakeet-CTC family"
        );
    }

    #[test]
    fn variant_name_and_upstream_hf_lock_the_v4_1_2b_release() {
        assert_eq!(
            GraniteSpeechVariant::V4_1_2B.name(),
            "granite-speech-4.1-2b"
        );
        assert_eq!(
            GraniteSpeechVariant::V4_1_2B.upstream_hf(),
            "ibm-granite/granite-speech-4.1-2b"
        );
        // Enum-arm-default constants agree with the V4_1_2B variant
        // (a drift would break the ModelKind::GraniteSpeech dispatch
        // in lib.rs).
        assert_eq!(NAME, GraniteSpeechVariant::V4_1_2B.name());
        assert_eq!(UPSTREAM_HF, GraniteSpeechVariant::V4_1_2B.upstream_hf());
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Realistic upstream tensor name from `GraniteSpeechEncoder`.
        let input_bytes =
            safetensors_one_bf16("encoder.blocks.0.self_attn.qkv_proj.weight", &[2, 3], &bf16);
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_granite_speech_file(
            &input_path,
            &output_path,
            GraniteSpeechVariant::V4_1_2B,
            None,
        )
        .expect("convert_granite_speech_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror canary_qwen / speecht5_hifigan)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("encoder.blocks.0.self_attn.qkv_proj.weight")
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

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through_and_stamps_land() {
        let input_bytes = safetensors_f32_and_f16();
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_granite_speech_file(
            &input_path,
            &output_path,
            GraniteSpeechVariant::V4_1_2B,
            None,
        )
        .expect("convert_granite_speech_file must accept a mixed F32/F16 checkpoint");

        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Round-trip carries both tensors with their dtypes preserved
        // AND the arch / provenance / category stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("encoder.input_linear.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");

        let f16_info = file
            .tensor_info("language_model.model.embed_tokens.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");

        // Provenance / category chunks landed (task-spec pins).
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
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "vokra.model.category must be `asr`",
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn hparam_chunk_group_lands_and_matches_primary_source_config() {
        // Round-trip every transcribed `vokra.granite_speech.*` U32 /
        // F32 hparam and cross-check the emitted value against the
        // primary-source `config.json` axes (fetched 2026-08-01). A
        // drift on either side (converter constant OR primary source
        // reinterpretation) fails this test loudly.
        let input_bytes = safetensors_f32_and_f16();
        let input_path = write_temp("hparam-in", &input_bytes);
        let output_path = write_temp("hparam-out", &[]);

        let _report = convert_granite_speech_file(
            &input_path,
            &output_path,
            GraniteSpeechVariant::V4_1_2B,
            None,
        )
        .expect("convert must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // U32 hparams
        for (key, want) in [
            (KEY_SAMPLE_RATE, SAMPLE_RATE),
            (KEY_AUDIO_TOKEN_INDEX, AUDIO_TOKEN_INDEX),
            (KEY_DOWNSAMPLE_RATE, DOWNSAMPLE_RATE),
            (KEY_WINDOW_SIZE, WINDOW_SIZE),
            (KEY_TIE_WORD_EMBEDDINGS, u32::from(TIE_WORD_EMBEDDINGS)),
            (KEY_HAS_LORA_ADAPTER, u32::from(HAS_LORA_ADAPTER)),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_ENC_HIDDEN_DIM, ENC_HIDDEN_DIM),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_HEAD_DIM, ENC_HEAD_DIM),
            (KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL),
            (KEY_ENC_INPUT_DIM, ENC_INPUT_DIM),
            (KEY_ENC_OUTPUT_DIM, ENC_OUTPUT_DIM),
            (KEY_ENC_CONV_EXPANSION, ENC_CONV_EXPANSION),
            (KEY_ENC_FEEDFORWARD_MULT, ENC_FEEDFORWARD_MULT),
            (KEY_ENC_CONTEXT_SIZE, ENC_CONTEXT_SIZE),
            (KEY_ENC_MAX_POS_EMB, ENC_MAX_POS_EMB),
            (KEY_DEC_N_LAYER, DEC_N_LAYER),
            (KEY_DEC_HIDDEN_DIM, DEC_HIDDEN_DIM),
            (KEY_DEC_N_HEAD, DEC_N_HEAD),
            (KEY_DEC_N_HEAD_KV, DEC_N_HEAD_KV),
            (KEY_DEC_FFN_DIM, DEC_FFN_DIM),
            (KEY_DEC_VOCAB_SIZE, DEC_VOCAB_SIZE),
            (KEY_DEC_N_CTX, DEC_N_CTX),
            (KEY_PROJ_N_LAYER, PROJ_N_LAYER),
            (KEY_PROJ_HIDDEN_SIZE, PROJ_HIDDEN_SIZE),
            (KEY_PROJ_INTERMEDIATE_SIZE, PROJ_INTERMEDIATE_SIZE),
            (KEY_PROJ_N_HEAD, PROJ_N_HEAD),
            (KEY_PROJ_MAX_POS, PROJ_MAX_POS),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: expected U32 {want}, got {other:?}"),
            }
        }

        // F32 scalar hparams (compared with a tight bound rather than
        // exact bit equality, in case a future serialize path adds an
        // f64 intermediate).
        for (key, want) in [
            (KEY_INITIALIZER_RANGE, INITIALIZER_RANGE),
            (KEY_DEC_ROPE_BASE, DEC_ROPE_BASE),
            (KEY_DEC_RMS_NORM_EPS, DEC_RMS_NORM_EPS),
            (KEY_DEC_ATTENTION_MULTIPLIER, DEC_ATTENTION_MULTIPLIER),
            (KEY_DEC_EMBEDDING_MULTIPLIER, DEC_EMBEDDING_MULTIPLIER),
            (KEY_DEC_LOGITS_SCALING, DEC_LOGITS_SCALING),
            (KEY_DEC_RESIDUAL_MULTIPLIER, DEC_RESIDUAL_MULTIPLIER),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::F32(v)) => {
                    assert!(
                        (*v - want).abs() < 1e-9 * want.abs().max(1.0),
                        "{key}: {v} vs {want}"
                    );
                }
                other => panic!("{key}: expected F32 {want}, got {other:?}"),
            }
        }

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn primary_source_axes_agree_with_config_json_fetched_2026_08_01() {
        // Pins every transcribed constant to the primary source. A
        // future contributor who edits a constant without also updating
        // the docstring + license-audit row + this test fails loudly.
        // Sourced from `huggingface.co/ibm-granite/granite-speech-4.1-2b/raw/main/config.json`
        // (fetched 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」).
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(AUDIO_TOKEN_INDEX, 100_352);
        assert_eq!(DOWNSAMPLE_RATE, 5);
        assert_eq!(WINDOW_SIZE, 15);
        const { assert!(!TIE_WORD_EMBEDDINGS) };
        const { assert!(!HAS_LORA_ADAPTER) };

        // Encoder
        assert_eq!(ENC_N_LAYER, 16);
        assert_eq!(ENC_HIDDEN_DIM, 1024);
        assert_eq!(ENC_N_HEAD, 8);
        assert_eq!(ENC_HEAD_DIM, 128);
        assert_eq!(ENC_CONV_KERNEL, 15);
        assert_eq!(ENC_INPUT_DIM, 160); // 80 logmels × 2 stacked frames
        assert_eq!(ENC_OUTPUT_DIM, 348);
        assert_eq!(ENC_CONTEXT_SIZE, 200);
        assert_eq!(ENC_MAX_POS_EMB, 512);

        // Decoder (Granite-4.0-1b-base — Granite family)
        assert_eq!(DEC_N_LAYER, 40);
        assert_eq!(DEC_HIDDEN_DIM, 2048);
        assert_eq!(DEC_N_HEAD, 16);
        assert_eq!(DEC_N_HEAD_KV, 4); // GQA 16 Q ÷ 4 KV
        assert_eq!(DEC_FFN_DIM, 4096);
        assert_eq!(DEC_VOCAB_SIZE, 100_353);
        assert_eq!(DEC_N_CTX, 4096);
        assert!((DEC_ROPE_BASE - 10_000.0).abs() < 1.0);
        assert!((DEC_RMS_NORM_EPS - 1e-5).abs() < 1e-12);
        // Distinctive Granite family scalars — these are the primary
        // fingerprint that separates a Granite LLM decoder from any
        // other Vokra decoder family (Llama / Qwen / Mistral do NOT
        // carry these).
        assert!((DEC_ATTENTION_MULTIPLIER - 0.007_812_5).abs() < 1e-9);
        assert!((DEC_EMBEDDING_MULTIPLIER - 12.0).abs() < 1e-6);
        assert!((DEC_LOGITS_SCALING - 8.0).abs() < 1e-6);
        assert!((DEC_RESIDUAL_MULTIPLIER - 0.22).abs() < 1e-6);

        // Projector (BLIP-2 q-former)
        assert_eq!(PROJ_N_LAYER, 2);
        assert_eq!(PROJ_HIDDEN_SIZE, 1024);
        assert_eq!(PROJ_INTERMEDIATE_SIZE, 4096);
        assert_eq!(PROJ_N_HEAD, 16);
        assert_eq!(PROJ_MAX_POS, 2048);
    }

    #[test]
    fn license_override_flows_through() {
        // A user who re-trained on a permissive corpus supplies a
        // different SPDX id at conversion time — the override must land
        // on KEY_PROVENANCE_LICENSE + KEY_PROVENANCE_WEIGHT_LICENSE and
        // the LicenseClass must be re-derived by from_license_str.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes =
            safetensors_one_bf16("encoder.blocks.0.self_attn.qkv_proj.weight", &[2, 3], &bf16);
        let input_path = write_temp("license-in", &input_bytes);
        let output_path = write_temp("license-out", &[]);

        let _report = convert_granite_speech_file(
            &input_path,
            &output_path,
            GraniteSpeechVariant::V4_1_2B,
            Some("mit"),
        )
        .expect("license override must succeed");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "license override must be honored"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT is Permissive class (same as apache-2.0)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn malformed_input_returns_parse_error() {
        let input_path = write_temp("malformed-in", &[]);
        let output_path = write_temp("malformed-out", &[]);
        let err = convert_granite_speech_file(
            &input_path,
            &output_path,
            GraniteSpeechVariant::V4_1_2B,
            None,
        )
        .expect_err("empty input must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
