//! Alibaba **Qwen3-ASR** family (`Qwen/Qwen3-ASR-0.6B` and
//! `Qwen/Qwen3-ASR-1.7B`, apache-2.0): safetensors → GGUF conversion
//! (SoTA plan Phase 5 ASR fleet, 2026-07-30).
//!
//! Input: the upstream `Qwen/Qwen3-ASR-{0.6B,1.7B}` release — a
//! Qwen3-flavour multilingual ASR checkpoint (`Qwen3ASRForConditionalGeneration`
//! architecture per the HF API `architectures` field, primary source
//! `huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json`). Output:
//! a GGUF carrying the exact release-specific BF16 tensor manifest under its
//! upstream safetensors names, plus the `vokra.qwen3_asr.*` hparam/identity
//! chunk group and `vokra.provenance.*` / `vokra.model.*` metadata consumed by
//! the strict native Qwen3-ASR binder.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`
//!   (recorded under `vokra.provenance.upstream_hf` together with the exact
//!   immutable revision). Both share the
//!   `qwen3_asr` `model_type` and the `Qwen3ASRForConditionalGeneration`
//!   architecture; the two sizes differ **only in the axes** captured
//!   in the `vokra.qwen3_asr.*` hparam chunk, so a single converter
//!   entry point handles both by dispatching on the `Variant` passed by
//!   the CLI.
//! - SPDX: `apache-2.0` (`LicenseClass::Permissive`), per both HF
//!   model cards (`cardData.license: apache-2.0`, CC-verified via the
//!   HF API on 2026-07-30 for both sizes).
//! - Model category: `asr` (recorded under `vokra.model.category`).
//!
//! # Architecture summary (primary source: HF `config.json`)
//!
//! The upstream `config.json` splits into a `thinker_config` with an
//! `audio_config` (audio encoder) and a `text_config` (Qwen3 decoder
//! LM). Both sizes are transcribed **verbatim** from the primary sources:
//!
//! - **0.6B** (`huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json`):
//!   - Audio encoder: `d_model=896`, `encoder_layers=18`,
//!     `encoder_attention_heads=14`, `encoder_ffn_dim=3584`,
//!     `num_mel_bins=128`, `max_source_positions=1500`,
//!     `output_dim=1024`, `downsample_hidden_size=480`,
//!     `conv_chunksize=500`, `n_window=50`, `n_window_infer=800`.
//!   - Text decoder (Qwen3): `hidden_size=1024`, `num_hidden_layers=28`,
//!     `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`,
//!     `intermediate_size=3072`, `max_position_embeddings=65536`,
//!     `rope_theta=1000000`, `rms_norm_eps=1e-06`, `vocab_size=151936`,
//!     `tie_word_embeddings=true`, `hidden_act="silu"`,
//!     `attention_bias=false`.
//!   - Adapter tokens: `audio_start_token_id=151669`,
//!     `audio_end_token_id=151670`, `audio_token_id=151676`.
//!   - `dtype="bfloat16"` (BF16 pass-through on the release path).
//!
//! - **1.7B** (`huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json`):
//!   - Audio encoder: `d_model=1024`, `encoder_layers=24`,
//!     `encoder_attention_heads=16`, `encoder_ffn_dim=4096`,
//!     `num_mel_bins=128`, `max_source_positions=1500`,
//!     `output_dim=2048`, `downsample_hidden_size=480`,
//!     `conv_chunksize=500`, `n_window=50`, `n_window_infer=800`.
//!   - Text decoder (Qwen3): `hidden_size=2048`, `num_hidden_layers=28`,
//!     `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=128`,
//!     `intermediate_size=6144`, `max_position_embeddings=65536`,
//!     `rope_theta=1000000`, `rms_norm_eps=1e-06`, `vocab_size=151936`,
//!     `tie_word_embeddings=true`, `hidden_act="silu"`,
//!     `attention_bias=false`.
//!   - Same adapter tokens as 0.6B — the tokenizer / adapter tokens
//!     are architecture-independent.
//!
//! Both configs are BF16 (`torch_dtype`/`dtype = "bfloat16"` per the
//! JSONs), so the release checkpoints hit the BF16 pass-through arm
//! (no convert-time widening).
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! Every required tensor must be BF16 and is emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters.
//! No convert-time widening; runtime widens BF16 → f32 losslessly via
//! the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). F32, F16, missing,
//! additional, renamed or reshaped tensors are rejected before output begins.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec / Wespeaker contract). The 612-tensor 0.6B and
//! 708-tensor 1.7B manifests are generated independently from their audited
//! release topology and SHA-pinned. The runtime binder checks the same exact
//! names and shapes without eagerly decoding the payload.
//!
//! # Real-weight parity
//!
//! Real-weight conversion and CPU parity against the upstream Qwen HF pipeline
//! are staged for VAST because each release exceeds the maintainer-Mac memory
//! threshold. Apple-device Metal parity is a separate pending gate; neither is
//! inferred from header-only or synthetic tests.
//!
//! # No ONNX (permanent)
//!
//! Qwen3-ASR is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05). The strict binder lives in
//! `crates/vokra-models/src/qwen3_asr/`; its log-mel frontend, audio encoder,
//! projector, fixed-revision Qwen2 BPE/chat contract and bounded-memory
//! 28-layer Qwen3 decode loop are native there. Conversion authenticates and
//! embeds all five tokenizer/chat/generation sidecars; no mutable runtime
//! download is permitted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufStreamWriter, GgufTensorDecl,
    GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};
use vokra_core::{FrontendSpec, LicenseClass};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFileReader};

/// `vokra.model.arch` for Qwen3-ASR GGUFs.
pub(crate) const ARCH: &str = "qwen3_asr";

/// Model-category tag written under `vokra.model.category` — distinguishes
/// ASR-family models from TTS / codec / vocoder / speaker siblings so
/// downstream consumers can pick a load path without inspecting the arch.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "asr";

/// Upstream-HF slug key (`vokra.provenance.upstream_hf`). Value depends on
/// the [`Variant`] the caller chose.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub(crate) const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
pub(crate) const KEY_SOURCE_REVISION: &str = "vokra.qwen3_asr.source_revision";
pub(crate) const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.qwen3_asr.tensor_manifest_sha256";

// -- Fixed-revision tokenizer / generation sidecars --------------------
// Both released sizes carry byte-identical files at the revisions pinned by
// `VariantAxes`. They are required together so an executable GGUF is
// self-contained and cannot silently pick up a mutable tokenizer from disk.
pub(crate) const KEY_TOKENIZER_VOCAB: &str = "vokra.qwen3_asr.tokenizer.vocab_json";
pub(crate) const KEY_TOKENIZER_MERGES: &str = "vokra.qwen3_asr.tokenizer.merges_txt";
pub(crate) const KEY_TOKENIZER_CONFIG: &str = "vokra.qwen3_asr.tokenizer.config_json";
pub(crate) const KEY_CHAT_TEMPLATE: &str = "vokra.qwen3_asr.tokenizer.chat_template_json";
pub(crate) const KEY_GENERATION_CONFIG: &str = "vokra.qwen3_asr.generation.config_json";

const VOCAB_FILE: ExactSidecar = ExactSidecar {
    name: "vocab.json",
    bytes: 2_776_833,
    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
};
const MERGES_FILE: ExactSidecar = ExactSidecar {
    name: "merges.txt",
    bytes: 1_671_853,
    sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
};
const TOKENIZER_CONFIG_FILE: ExactSidecar = ExactSidecar {
    name: "tokenizer_config.json",
    bytes: 12_487,
    sha256: "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
};
const CHAT_TEMPLATE_FILE: ExactSidecar = ExactSidecar {
    name: "chat_template.json",
    bytes: 1_161,
    sha256: "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
};
const GENERATION_CONFIG_FILE: ExactSidecar = ExactSidecar {
    name: "generation_config.json",
    bytes: 142,
    sha256: "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c",
};

// -- `vokra.qwen3_asr.*` audio-encoder hparam keys ----------------------
pub(crate) const KEY_AUDIO_D_MODEL: &str = "vokra.qwen3_asr.audio.d_model";
pub(crate) const KEY_AUDIO_N_LAYER: &str = "vokra.qwen3_asr.audio.n_layer";
pub(crate) const KEY_AUDIO_N_HEAD: &str = "vokra.qwen3_asr.audio.n_head";
pub(crate) const KEY_AUDIO_FFN_DIM: &str = "vokra.qwen3_asr.audio.ffn_dim";
pub(crate) const KEY_AUDIO_N_MELS: &str = "vokra.qwen3_asr.audio.n_mels";
pub(crate) const KEY_AUDIO_MAX_SOURCE_POSITIONS: &str =
    "vokra.qwen3_asr.audio.max_source_positions";
pub(crate) const KEY_AUDIO_OUTPUT_DIM: &str = "vokra.qwen3_asr.audio.output_dim";
pub(crate) const KEY_AUDIO_DOWNSAMPLE_HIDDEN_SIZE: &str =
    "vokra.qwen3_asr.audio.downsample_hidden_size";
pub(crate) const KEY_AUDIO_CONV_CHUNKSIZE: &str = "vokra.qwen3_asr.audio.conv_chunksize";
pub(crate) const KEY_AUDIO_N_WINDOW: &str = "vokra.qwen3_asr.audio.n_window";
pub(crate) const KEY_AUDIO_N_WINDOW_INFER: &str = "vokra.qwen3_asr.audio.n_window_infer";
pub(crate) const KEY_AUDIO_LAYER_NORM_EPS: &str = "vokra.qwen3_asr.audio.layer_norm_eps";
pub(crate) const KEY_AUDIO_ACTIVATION_FUNCTION: &str = "vokra.qwen3_asr.audio.activation_function";
pub(crate) const KEY_AUDIO_SCALE_EMBEDDING: &str = "vokra.qwen3_asr.audio.scale_embedding";

// -- `vokra.qwen3_asr.*` text-decoder (Qwen3 LM) hparam keys ------------
pub(crate) const KEY_TEXT_HIDDEN_SIZE: &str = "vokra.qwen3_asr.text.hidden_size";
pub(crate) const KEY_TEXT_N_LAYER: &str = "vokra.qwen3_asr.text.n_layer";
pub(crate) const KEY_TEXT_N_HEAD: &str = "vokra.qwen3_asr.text.n_head";
pub(crate) const KEY_TEXT_N_KV_HEAD: &str = "vokra.qwen3_asr.text.n_kv_head";
pub(crate) const KEY_TEXT_HEAD_DIM: &str = "vokra.qwen3_asr.text.head_dim";
pub(crate) const KEY_TEXT_FFN_DIM: &str = "vokra.qwen3_asr.text.ffn_dim";
pub(crate) const KEY_TEXT_MAX_POSITION_EMBEDDINGS: &str =
    "vokra.qwen3_asr.text.max_position_embeddings";
pub(crate) const KEY_TEXT_ROPE_THETA: &str = "vokra.qwen3_asr.text.rope_theta";
pub(crate) const KEY_TEXT_RMS_NORM_EPS: &str = "vokra.qwen3_asr.text.rms_norm_eps";
pub(crate) const KEY_TEXT_VOCAB_SIZE: &str = "vokra.qwen3_asr.text.vocab_size";
pub(crate) const KEY_TEXT_TIE_WORD_EMBEDDINGS: &str = "vokra.qwen3_asr.text.tie_word_embeddings";
pub(crate) const KEY_TEXT_ATTENTION_BIAS: &str = "vokra.qwen3_asr.text.attention_bias";

// -- `vokra.qwen3_asr.*` adapter-token keys -----------------------------
pub(crate) const KEY_AUDIO_START_TOKEN_ID: &str = "vokra.qwen3_asr.audio_start_token_id";
pub(crate) const KEY_AUDIO_END_TOKEN_ID: &str = "vokra.qwen3_asr.audio_end_token_id";
pub(crate) const KEY_AUDIO_TOKEN_ID: &str = "vokra.qwen3_asr.audio_token_id";

/// Which Qwen3-ASR release the converter is bound to. The two sizes
/// share arch, category, provenance stamps, and BF16 pass-through
/// posture; only the axes captured in the hparam chunk change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `Qwen/Qwen3-ASR-0.6B`. Audio encoder 18 × d=896 × 14h × ffn=3584,
    /// text decoder Qwen3 28 × d=1024 × 16Q ÷ 8KV × head_dim=128 ×
    /// ffn=3072.
    B06,
    /// `Qwen/Qwen3-ASR-1.7B`. Audio encoder 24 × d=1024 × 16h × ffn=4096,
    /// text decoder Qwen3 28 × d=2048 × 16Q ÷ 8KV × head_dim=128 ×
    /// ffn=6144.
    B17,
}

/// Per-variant axes transcribed verbatim from the primary-source
/// `config.json`.
#[derive(Debug, Clone, Copy)]
struct VariantAxes {
    // Common labels.
    name: &'static str,
    upstream_hf: &'static str,
    source_revision: &'static str,
    tensor_count: usize,
    tensor_manifest_sha256: &'static str,
    // Audio encoder axes.
    audio_d_model: u32,
    audio_n_layer: u32,
    audio_n_head: u32,
    audio_ffn_dim: u32,
    audio_n_mels: u32,
    audio_max_source_positions: u32,
    audio_output_dim: u32,
    audio_downsample_hidden_size: u32,
    audio_conv_chunksize: u32,
    audio_n_window: u32,
    audio_n_window_infer: u32,
    // Text decoder (Qwen3 LM) axes.
    text_hidden_size: u32,
    text_n_layer: u32,
    text_n_head: u32,
    text_n_kv_head: u32,
    text_head_dim: u32,
    text_ffn_dim: u32,
    text_max_position_embeddings: u32,
    text_rope_theta: f32,
    text_rms_norm_eps: f32,
    text_vocab_size: u32,
    text_tie_word_embeddings: bool,
    text_attention_bias: bool,
    // Adapter tokens (shared across both sizes today; kept per-variant
    // so a future release that shifts them cannot silently misroute).
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_token_id: u32,
}

impl Variant {
    fn axes(self) -> VariantAxes {
        match self {
            // Primary source: huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::B06 => VariantAxes {
                name: "qwen3-asr-0.6b",
                upstream_hf: "Qwen/Qwen3-ASR-0.6B",
                source_revision: "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
                tensor_count: 612,
                tensor_manifest_sha256: "8ff041c01225c0c743af7386978ca516afc633e000b181f4d49d775b8e99f91b",
                audio_d_model: 896,
                audio_n_layer: 18,
                audio_n_head: 14,
                audio_ffn_dim: 3584,
                audio_n_mels: 128,
                audio_max_source_positions: 1500,
                audio_output_dim: 1024,
                audio_downsample_hidden_size: 480,
                audio_conv_chunksize: 500,
                audio_n_window: 50,
                audio_n_window_infer: 800,
                text_hidden_size: 1024,
                text_n_layer: 28,
                text_n_head: 16,
                text_n_kv_head: 8,
                text_head_dim: 128,
                text_ffn_dim: 3072,
                text_max_position_embeddings: 65536,
                text_rope_theta: 1_000_000.0,
                text_rms_norm_eps: 1e-6,
                text_vocab_size: 151_936,
                text_tie_word_embeddings: true,
                text_attention_bias: false,
                audio_start_token_id: 151_669,
                audio_end_token_id: 151_670,
                audio_token_id: 151_676,
            },
            // Primary source: huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::B17 => VariantAxes {
                name: "qwen3-asr-1.7b",
                upstream_hf: "Qwen/Qwen3-ASR-1.7B",
                source_revision: "7278e1e70fe206f11671096ffdd38061171dd6e5",
                tensor_count: 708,
                tensor_manifest_sha256: "9136bf1de42a3248fb1ea55877dced6113a8b1e5a98fcae08b01b67f10a523ee",
                audio_d_model: 1024,
                audio_n_layer: 24,
                audio_n_head: 16,
                audio_ffn_dim: 4096,
                audio_n_mels: 128,
                audio_max_source_positions: 1500,
                audio_output_dim: 2048,
                audio_downsample_hidden_size: 480,
                audio_conv_chunksize: 500,
                audio_n_window: 50,
                audio_n_window_infer: 800,
                text_hidden_size: 2048,
                text_n_layer: 28,
                text_n_head: 16,
                text_n_kv_head: 8,
                text_head_dim: 128,
                text_ffn_dim: 6144,
                text_max_position_embeddings: 65536,
                text_rope_theta: 1_000_000.0,
                text_rms_norm_eps: 1e-6,
                text_vocab_size: 151_936,
                text_tie_word_embeddings: true,
                text_attention_bias: false,
                // Same as 0.6B — the tokenizer/adapter tokens are
                // architecture-independent.
                audio_start_token_id: 151_669,
                audio_end_token_id: 151_670,
                audio_token_id: 151_676,
            },
        }
    }
}

/// Outcome of a Qwen3-ASR conversion.
#[derive(Debug, Default)]
pub struct Qwen3AsrReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Canonical BF16 tensors written verbatim.
    pub written: usize,
    /// BF16 tensors copied byte-for-byte; equal to [`Self::written`] on
    /// success.
    pub bf16_passthrough: usize,
    /// Metadata entries stamped into the GGUF.
    pub metadata_count: usize,
}

/// Variant-taking file-based Qwen3-ASR converter — the CLI dispatch
/// arm picks the [`Variant`] from the `--model` string
/// (`qwen3-asr-0.6b` / `qwen3-asr-1.7b`).
///
/// Reads `input` as either the upstream single `model.safetensors` or a
/// `model.safetensors.index.json` plus its referenced shards. The exact
/// `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.json`
/// and `generation_config.json` from the same pinned release directory are
/// mandatory and embedded verbatim. The converter then streams a Vokra GGUF
/// to `output`. `license` may repeat the audited `apache-2.0` license but
/// cannot override it with a conflicting value.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing `output`;
/// [`ConvertError::Parse`] for malformed, partial, foreign, wrong-dtype or
/// inconsistent sharded input; [`ConvertError::Usage`] for a conflicting
/// license; [`ConvertError::Gguf`] if GGUF serialization fails.
pub fn convert_qwen3_asr_file_with_variant(
    input: &Path,
    output: &Path,
    variant: Variant,
    license: Option<&str>,
) -> Result<Qwen3AsrReport, ConvertError> {
    if let Some(value) = license
        && !value.is_empty()
        && !value.eq_ignore_ascii_case("apache-2.0")
    {
        return Err(ConvertError::Usage(format!(
            "qwen3-asr: {}@{} has pinned Apache-2.0 weights; refusing conflicting --license {value:?}",
            variant.axes().upstream_hf,
            variant.axes().source_revision
        )));
    }

    let axes = variant.axes();
    let mut checkpoint = CheckpointReader::open(input)?;
    checkpoint.validate(variant)?;
    let tokenizer = TokenizerAssets::load(input, &axes)?;
    let mut b = metadata_builder(&axes);
    tokenizer.embed(&mut b);
    let metadata_count = b.metadata_count();
    let expected = expected_manifest(variant);
    let decls = expected
        .keys()
        .map(|name| {
            let tensor = checkpoint.tensor(name).expect("validated manifest entry");
            GgufTensorDecl {
                name: name.clone(),
                dtype: tensor.info.dtype,
                dimensions: tensor.info.shape.clone(),
            }
        })
        .collect::<Vec<_>>();

    let output_file = std::fs::File::create(output)?;
    let mut writer = GgufStreamWriter::begin(std::io::BufWriter::new(output_file), &b, &decls)?;
    let mut payload = Vec::new();
    for declaration in &decls {
        checkpoint.read_tensor_into(&declaration.name, &mut payload)?;
        writer.write_tensor(&declaration.name, &payload)?;
    }
    drop(payload);
    let output_file = writer
        .finish()?
        .into_inner()
        .map_err(|error| ConvertError::Io(error.into_error()))?;
    output_file.sync_all().map_err(ConvertError::Io)?;

    Ok(Qwen3AsrReport {
        read: decls.len(),
        written: decls.len(),
        bf16_passthrough: decls.len(),
        metadata_count,
    })
}

fn metadata_builder(axes: &VariantAxes) -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, axes.name);
    builder.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, axes.upstream_hf);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, axes.source_revision);
    builder.add_string(KEY_SOURCE_REVISION, axes.source_revision);
    builder.add_string(KEY_TENSOR_MANIFEST_SHA256, axes.tensor_manifest_sha256);

    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(axes.name),
        Some(&format!(
            "{}@{} exact {}-BF16-tensor Qwen3-ASR checkpoint",
            axes.upstream_hf, axes.source_revision, axes.tensor_count
        )),
    );
    frontend_spec().write_into(&mut builder);
    write_hparams(&mut builder, axes);
    builder
}

#[derive(Debug, Clone, Copy)]
struct ExactSidecar {
    name: &'static str,
    bytes: usize,
    sha256: &'static str,
}

#[derive(Debug)]
struct TokenizerAssets {
    vocab: Vec<u8>,
    merges: Vec<u8>,
    tokenizer_config: Vec<u8>,
    chat_template: Vec<u8>,
    generation_config: Vec<u8>,
}

impl TokenizerAssets {
    fn load(input: &Path, axes: &VariantAxes) -> Result<Self, ConvertError> {
        let directory = input.parent().unwrap_or_else(|| Path::new("."));
        Ok(Self {
            vocab: read_exact_sidecar(directory, VOCAB_FILE, axes)?,
            merges: read_exact_sidecar(directory, MERGES_FILE, axes)?,
            tokenizer_config: read_exact_sidecar(directory, TOKENIZER_CONFIG_FILE, axes)?,
            chat_template: read_exact_sidecar(directory, CHAT_TEMPLATE_FILE, axes)?,
            generation_config: read_exact_sidecar(directory, GENERATION_CONFIG_FILE, axes)?,
        })
    }

    fn embed(&self, builder: &mut GgufBuilder) {
        add_u8_array(builder, KEY_TOKENIZER_VOCAB, &self.vocab);
        add_u8_array(builder, KEY_TOKENIZER_MERGES, &self.merges);
        add_u8_array(builder, KEY_TOKENIZER_CONFIG, &self.tokenizer_config);
        add_u8_array(builder, KEY_CHAT_TEMPLATE, &self.chat_template);
        add_u8_array(builder, KEY_GENERATION_CONFIG, &self.generation_config);
    }
}

fn read_exact_sidecar(
    directory: &Path,
    spec: ExactSidecar,
    axes: &VariantAxes,
) -> Result<Vec<u8>, ConvertError> {
    let path = directory.join(spec.name);
    let bytes = std::fs::read(&path).map_err(|error| {
        ConvertError::Io(std::io::Error::new(
            error.kind(),
            format!(
                "qwen3-asr: reading required {}@{} sidecar {}: {error}",
                axes.upstream_hf,
                axes.source_revision,
                path.display()
            ),
        ))
    })?;
    if bytes.len() != spec.bytes {
        return Err(parse_error(format!(
            "{}@{} sidecar {} is {} bytes, expected exactly {}",
            axes.upstream_hf,
            axes.source_revision,
            spec.name,
            bytes.len(),
            spec.bytes
        )));
    }
    let actual =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(&bytes));
    if actual != spec.sha256 {
        return Err(parse_error(format!(
            "{}@{} sidecar {} SHA-256 {actual}, expected {}",
            axes.upstream_hf, axes.source_revision, spec.name, spec.sha256
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

/// Exact `WhisperFeatureExtractor` parameters pinned by both Qwen3-ASR
/// releases. The processor overrides `padding=true, truncation=false`, which
/// changes only the waveform length presented to this frontend, not these
/// signal-processing axes.
pub(crate) fn frontend_spec() -> FrontendSpec {
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

#[derive(Debug)]
struct ResolvedSources {
    files: Vec<(String, PathBuf)>,
    weight_map: Option<BTreeMap<String, String>>,
}

fn resolve_sources(input: &Path) -> Result<ResolvedSources, ConvertError> {
    let is_index = input
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".index.json"));
    if !is_index {
        let name = input
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| input.display().to_string());
        return Ok(ResolvedSources {
            files: vec![(name, input.to_path_buf())],
            weight_map: None,
        });
    }

    let bytes = std::fs::read(input)?;
    let root = json::parse(&bytes).map_err(|error| {
        parse_error(format!(
            "shard index {} is malformed: {error}",
            input.display()
        ))
    })?;
    let entries = root
        .get("weight_map")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            parse_error(format!(
                "shard index {} has no `weight_map` object",
                input.display()
            ))
        })?;
    if entries.is_empty() {
        return Err(parse_error(format!(
            "shard index {} has an empty `weight_map`",
            input.display()
        )));
    }

    let directory = input.parent().unwrap_or_else(|| Path::new("."));
    let mut weight_map = BTreeMap::new();
    let mut shard_names = BTreeSet::new();
    for (tensor, value) in entries {
        let shard = value.as_str().ok_or_else(|| {
            parse_error(format!(
                "shard index {} maps tensor {tensor:?} to a non-string value",
                input.display()
            ))
        })?;
        let mut components = Path::new(shard).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(parse_error(format!(
                "shard index {} uses unsafe/non-local shard path {shard:?}",
                input.display()
            )));
        }
        if weight_map
            .insert(tensor.clone(), shard.to_owned())
            .is_some()
        {
            return Err(parse_error(format!(
                "shard index {} repeats tensor key {tensor:?}",
                input.display()
            )));
        }
        shard_names.insert(shard.to_owned());
    }

    let files = shard_names
        .into_iter()
        .map(|name| {
            let path = directory.join(&name);
            if !path.is_file() {
                return Err(parse_error(format!(
                    "shard index {} references missing file {}",
                    input.display(),
                    path.display()
                )));
            }
            Ok((name, path))
        })
        .collect::<Result<Vec<_>, ConvertError>>()?;
    Ok(ResolvedSources {
        files,
        weight_map: Some(weight_map),
    })
}

#[derive(Debug, Clone)]
struct LocatedTensor {
    reader: usize,
    source_name: String,
    info: SafeTensorInfo,
}

#[derive(Debug)]
struct CheckpointReader {
    readers: Vec<SafetensorsFileReader>,
    tensors: BTreeMap<String, LocatedTensor>,
    weight_map: Option<BTreeMap<String, String>>,
}

impl CheckpointReader {
    fn open(input: &Path) -> Result<Self, ConvertError> {
        let resolved = resolve_sources(input)?;
        let mut readers = Vec::with_capacity(resolved.files.len());
        let mut tensors = BTreeMap::new();
        for (source_name, path) in resolved.files {
            let reader = SafetensorsFileReader::open(&path).map_err(|error| {
                parse_error(format!(
                    "opening Qwen3-ASR shard {}: {error}",
                    path.display()
                ))
            })?;
            let reader_index = readers.len();
            for info in reader.tensors() {
                let located = LocatedTensor {
                    reader: reader_index,
                    source_name: source_name.clone(),
                    info: info.clone(),
                };
                if tensors.insert(info.name.clone(), located).is_some() {
                    return Err(parse_error(format!(
                        "tensor {:?} appears in more than one safetensors shard",
                        info.name
                    )));
                }
            }
            readers.push(reader);
        }
        Ok(Self {
            readers,
            tensors,
            weight_map: resolved.weight_map,
        })
    }

    fn tensor(&self, name: &str) -> Option<&LocatedTensor> {
        self.tensors.get(name)
    }

    fn validate(&self, variant: Variant) -> Result<(), ConvertError> {
        let observed = self
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), (tensor.info.dtype, tensor.info.shape.clone())))
            .collect::<BTreeMap<_, _>>();
        validate_observed_manifest(&observed, variant)?;

        if let Some(weight_map) = &self.weight_map {
            let axes = variant.axes();
            if weight_map.len() != axes.tensor_count {
                return Err(parse_error(format!(
                    "{} shard index declares {} tensors, expected exactly {}",
                    axes.name,
                    weight_map.len(),
                    axes.tensor_count
                )));
            }
            for (name, tensor) in &self.tensors {
                let declared = weight_map.get(name).ok_or_else(|| {
                    parse_error(format!("{} shard index omits tensor {name:?}", axes.name))
                })?;
                if declared != &tensor.source_name {
                    return Err(parse_error(format!(
                        "{} shard index maps tensor {name:?} to {declared:?}, but the tensor is stored in {:?}",
                        axes.name, tensor.source_name
                    )));
                }
            }
            if let Some(name) = weight_map
                .keys()
                .find(|name| !self.tensors.contains_key(*name))
            {
                return Err(parse_error(format!(
                    "{} shard index declares absent tensor {name:?}",
                    axes.name
                )));
            }
        }
        Ok(())
    }

    fn read_tensor_into(&mut self, name: &str, payload: &mut Vec<u8>) -> Result<(), ConvertError> {
        let reader = self
            .tensors
            .get(name)
            .ok_or_else(|| parse_error(format!("validated tensor {name:?} disappeared")))?
            .reader;
        self.readers[reader]
            .read_tensor_into(name, payload)
            .map_err(|error| parse_error(format!("reading tensor {name:?}: {error}")))
    }
}

fn validate_observed_manifest(
    observed: &BTreeMap<String, (GgmlType, Vec<u64>)>,
    variant: Variant,
) -> Result<(), ConvertError> {
    let axes = variant.axes();
    let expected = expected_manifest(variant);
    if observed.len() != axes.tensor_count {
        return Err(parse_error(format!(
            "{} checkpoint has {} tensors, expected exactly {} at revision {}",
            axes.name,
            observed.len(),
            axes.tensor_count,
            axes.source_revision
        )));
    }
    for (name, (dtype, shape)) in observed {
        let expected_shape = expected.get(name).ok_or_else(|| {
            parse_error(format!(
                "{} checkpoint contains unexpected tensor {name:?}",
                axes.name
            ))
        })?;
        if *dtype != GgmlType::BF16 {
            return Err(parse_error(format!(
                "{} tensor {name:?} is {dtype:?}, expected canonical BF16",
                axes.name
            )));
        }
        if shape != expected_shape {
            return Err(parse_error(format!(
                "{} tensor {name:?} has shape {shape:?}, expected {expected_shape:?}",
                axes.name
            )));
        }
    }
    if let Some(name) = expected.keys().find(|name| !observed.contains_key(*name)) {
        return Err(parse_error(format!(
            "{} checkpoint is missing required tensor {name:?}",
            axes.name
        )));
    }
    let actual_hash = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(&expected),
    );
    if actual_hash != axes.tensor_manifest_sha256 {
        return Err(parse_error(format!(
            "{} internal expected-manifest SHA-256 {actual_hash} does not match pinned {}",
            axes.name, axes.tensor_manifest_sha256
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("qwen3-asr: {}", message.into()))
}

pub(crate) fn expected_manifest(variant: Variant) -> BTreeMap<String, Vec<u64>> {
    let axes = variant.axes();
    let mut tensors = BTreeMap::new();
    let mut insert = |name: String, shape: &[u64]| {
        debug_assert!(tensors.insert(name, shape.to_vec()).is_none());
    };

    let conv = u64::from(axes.audio_downsample_hidden_size);
    insert("thinker.audio_tower.conv2d1.bias".into(), &[conv]);
    insert(
        "thinker.audio_tower.conv2d1.weight".into(),
        &[conv, 1, 3, 3],
    );
    for index in 2..=3 {
        insert(format!("thinker.audio_tower.conv2d{index}.bias"), &[conv]);
        insert(
            format!("thinker.audio_tower.conv2d{index}.weight"),
            &[conv, conv, 3, 3],
        );
    }
    let audio_dim = u64::from(axes.audio_d_model);
    let audio_ffn = u64::from(axes.audio_ffn_dim);
    insert(
        "thinker.audio_tower.conv_out.weight".into(),
        &[audio_dim, 7_680],
    );
    for layer in 0..axes.audio_n_layer {
        let prefix = format!("thinker.audio_tower.layers.{layer}");
        insert(format!("{prefix}.fc1.bias"), &[audio_ffn]);
        insert(format!("{prefix}.fc1.weight"), &[audio_ffn, audio_dim]);
        insert(format!("{prefix}.fc2.bias"), &[audio_dim]);
        insert(format!("{prefix}.fc2.weight"), &[audio_dim, audio_ffn]);
        for norm in ["final_layer_norm", "self_attn_layer_norm"] {
            insert(format!("{prefix}.{norm}.bias"), &[audio_dim]);
            insert(format!("{prefix}.{norm}.weight"), &[audio_dim]);
        }
        for projection in ["k_proj", "out_proj", "q_proj", "v_proj"] {
            insert(
                format!("{prefix}.self_attn.{projection}.bias"),
                &[audio_dim],
            );
            insert(
                format!("{prefix}.self_attn.{projection}.weight"),
                &[audio_dim, audio_dim],
            );
        }
    }
    insert("thinker.audio_tower.ln_post.bias".into(), &[audio_dim]);
    insert("thinker.audio_tower.ln_post.weight".into(), &[audio_dim]);
    insert("thinker.audio_tower.proj1.bias".into(), &[audio_dim]);
    insert(
        "thinker.audio_tower.proj1.weight".into(),
        &[audio_dim, audio_dim],
    );
    insert(
        "thinker.audio_tower.proj2.bias".into(),
        &[u64::from(axes.audio_output_dim)],
    );
    insert(
        "thinker.audio_tower.proj2.weight".into(),
        &[u64::from(axes.audio_output_dim), audio_dim],
    );

    let hidden = u64::from(axes.text_hidden_size);
    let query_width = u64::from(axes.text_n_head * axes.text_head_dim);
    let key_value_width = u64::from(axes.text_n_kv_head * axes.text_head_dim);
    let text_ffn = u64::from(axes.text_ffn_dim);
    let vocab = u64::from(axes.text_vocab_size);
    insert("thinker.lm_head.weight".into(), &[vocab, hidden]);
    insert("thinker.model.embed_tokens.weight".into(), &[vocab, hidden]);
    for layer in 0..axes.text_n_layer {
        let prefix = format!("thinker.model.layers.{layer}");
        insert(format!("{prefix}.input_layernorm.weight"), &[hidden]);
        insert(
            format!("{prefix}.mlp.down_proj.weight"),
            &[hidden, text_ffn],
        );
        insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            &[text_ffn, hidden],
        );
        insert(format!("{prefix}.mlp.up_proj.weight"), &[text_ffn, hidden]);
        insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[hidden],
        );
        insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            &[u64::from(axes.text_head_dim)],
        );
        insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[key_value_width, hidden],
        );
        insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            &[hidden, query_width],
        );
        insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            &[u64::from(axes.text_head_dim)],
        );
        insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[query_width, hidden],
        );
        insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[key_value_width, hidden],
        );
    }
    insert("thinker.model.norm.weight".into(), &[hidden]);
    debug_assert_eq!(tensors.len(), axes.tensor_count);
    tensors
}

/// Default file-based converter — 1.7B variant (the flagship release).
/// The CLI's `qwen3-asr-0.6b` slug routes to
/// [`convert_qwen3_asr_file_with_variant`] with [`Variant::B06`].
pub fn convert_qwen3_asr_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen3AsrReport, ConvertError> {
    convert_qwen3_asr_file_with_variant(input, output, Variant::B17, license)
}

fn write_hparams(b: &mut GgufBuilder, axes: &VariantAxes) {
    // Audio encoder.
    b.add_u32(KEY_AUDIO_D_MODEL, axes.audio_d_model);
    b.add_u32(KEY_AUDIO_N_LAYER, axes.audio_n_layer);
    b.add_u32(KEY_AUDIO_N_HEAD, axes.audio_n_head);
    b.add_u32(KEY_AUDIO_FFN_DIM, axes.audio_ffn_dim);
    b.add_u32(KEY_AUDIO_N_MELS, axes.audio_n_mels);
    b.add_u32(
        KEY_AUDIO_MAX_SOURCE_POSITIONS,
        axes.audio_max_source_positions,
    );
    b.add_u32(KEY_AUDIO_OUTPUT_DIM, axes.audio_output_dim);
    b.add_u32(
        KEY_AUDIO_DOWNSAMPLE_HIDDEN_SIZE,
        axes.audio_downsample_hidden_size,
    );
    b.add_u32(KEY_AUDIO_CONV_CHUNKSIZE, axes.audio_conv_chunksize);
    b.add_u32(KEY_AUDIO_N_WINDOW, axes.audio_n_window);
    b.add_u32(KEY_AUDIO_N_WINDOW_INFER, axes.audio_n_window_infer);
    // `nn.LayerNorm` is constructed without an override in the pinned model
    // source, so PyTorch's exact default epsilon is part of the graph contract.
    b.add_f32(KEY_AUDIO_LAYER_NORM_EPS, 1.0e-5);
    b.add_string(KEY_AUDIO_ACTIVATION_FUNCTION, "gelu");
    b.add_bool(KEY_AUDIO_SCALE_EMBEDDING, false);
    // Text decoder (Qwen3 LM).
    b.add_u32(KEY_TEXT_HIDDEN_SIZE, axes.text_hidden_size);
    b.add_u32(KEY_TEXT_N_LAYER, axes.text_n_layer);
    b.add_u32(KEY_TEXT_N_HEAD, axes.text_n_head);
    b.add_u32(KEY_TEXT_N_KV_HEAD, axes.text_n_kv_head);
    b.add_u32(KEY_TEXT_HEAD_DIM, axes.text_head_dim);
    b.add_u32(KEY_TEXT_FFN_DIM, axes.text_ffn_dim);
    b.add_u32(
        KEY_TEXT_MAX_POSITION_EMBEDDINGS,
        axes.text_max_position_embeddings,
    );
    b.add_f32(KEY_TEXT_ROPE_THETA, axes.text_rope_theta);
    b.add_f32(KEY_TEXT_RMS_NORM_EPS, axes.text_rms_norm_eps);
    b.add_u32(KEY_TEXT_VOCAB_SIZE, axes.text_vocab_size);
    b.add_bool(KEY_TEXT_TIE_WORD_EMBEDDINGS, axes.text_tie_word_embeddings);
    b.add_bool(KEY_TEXT_ATTENTION_BIAS, axes.text_attention_bias);
    // Adapter tokens.
    b.add_u32(KEY_AUDIO_START_TOKEN_ID, axes.audio_start_token_id);
    b.add_u32(KEY_AUDIO_END_TOKEN_ID, axes.audio_end_token_id);
    b.add_u32(KEY_AUDIO_TOKEN_ID, axes.audio_token_id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn observed(variant: Variant) -> BTreeMap<String, (GgmlType, Vec<u64>)> {
        expected_manifest(variant)
            .into_iter()
            .map(|(name, shape)| (name, (GgmlType::BF16, shape)))
            .collect()
    }

    fn scratch_directory(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-qwen3-asr-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir(&path).expect("create scratch directory");
        path
    }

    #[test]
    fn generated_manifests_match_range_audited_public_headers() {
        for variant in [Variant::B06, Variant::B17] {
            let axes = variant.axes();
            let manifest = expected_manifest(variant);
            assert_eq!(manifest.len(), axes.tensor_count);
            assert_eq!(
                crate::models::canary_1b_flash::hex(
                    &crate::models::canary_1b_flash::manifest_sha256(&manifest)
                ),
                axes.tensor_manifest_sha256
            );
            validate_observed_manifest(&observed(variant), variant)
                .expect("range-audited exact manifest");
        }
    }

    #[test]
    fn strict_manifest_rejects_shape_dtype_and_membership_drift() {
        let mut wrong_shape = observed(Variant::B06);
        wrong_shape
            .get_mut("thinker.audio_tower.conv2d1.bias")
            .expect("known tensor")
            .1[0] += 1;
        assert!(
            validate_observed_manifest(&wrong_shape, Variant::B06)
                .expect_err("shape drift")
                .to_string()
                .contains("shape")
        );

        let mut wrong_dtype = observed(Variant::B17);
        wrong_dtype
            .get_mut("thinker.model.norm.weight")
            .expect("known tensor")
            .0 = GgmlType::F16;
        assert!(
            validate_observed_manifest(&wrong_dtype, Variant::B17)
                .expect_err("dtype drift")
                .to_string()
                .contains("canonical BF16")
        );

        let mut wrong_member = observed(Variant::B06);
        wrong_member.remove("thinker.model.norm.weight");
        wrong_member.insert("foreign.weight".into(), (GgmlType::BF16, vec![1]));
        assert!(
            validate_observed_manifest(&wrong_member, Variant::B06)
                .expect_err("membership drift")
                .to_string()
                .contains("unexpected tensor")
        );
    }

    #[test]
    fn metadata_pins_revision_manifest_axes_and_license() {
        for variant in [Variant::B06, Variant::B17] {
            let axes = variant.axes();
            let builder = metadata_builder(&axes);
            // 53 explicit Qwen/provenance/frontend entries plus the writer's
            // two unspoofable schema stamp entries (version and producer).
            assert_eq!(builder.metadata_count(), 55);
            let bytes = builder.to_bytes().expect("serialize metadata");
            let file = GgufFile::parse(bytes).expect("parse metadata-only GGUF");
            assert_eq!(
                file.get(chunks::KEY_MODEL_NAME)
                    .and_then(|value| value.as_str()),
                Some(axes.name)
            );
            assert_eq!(
                file.get(KEY_PROVENANCE_UPSTREAM_HF)
                    .and_then(|value| value.as_str()),
                Some(axes.upstream_hf)
            );
            assert_eq!(
                file.get(KEY_SOURCE_REVISION)
                    .and_then(|value| value.as_str()),
                Some(axes.source_revision)
            );
            assert_eq!(
                file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                    .and_then(|value| value.as_str()),
                Some(axes.source_revision)
            );
            assert_eq!(
                file.get(KEY_TENSOR_MANIFEST_SHA256)
                    .and_then(|value| value.as_str()),
                Some(axes.tensor_manifest_sha256)
            );
            assert_eq!(
                file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                    .and_then(|value| value.as_str()),
                Some(LicenseClass::Permissive.as_str())
            );
            assert_eq!(
                file.get(KEY_AUDIO_N_LAYER).and_then(|value| value.as_u64()),
                Some(u64::from(axes.audio_n_layer))
            );
            assert_eq!(
                FrontendSpec::from_gguf(&file).expect("frontend metadata"),
                frontend_spec()
            );
            assert_eq!(
                file.get(KEY_AUDIO_LAYER_NORM_EPS)
                    .and_then(|value| value.as_f64()),
                Some(f64::from(1.0e-5_f32))
            );
            assert_eq!(
                file.get(KEY_AUDIO_ACTIVATION_FUNCTION)
                    .and_then(|value| value.as_str()),
                Some("gelu")
            );
            assert_eq!(
                file.get(KEY_AUDIO_SCALE_EMBEDDING)
                    .and_then(|value| value.as_bool()),
                Some(false)
            );
        }
    }

    #[test]
    fn tokenizer_assets_are_embedded_as_one_complete_group() {
        let axes = Variant::B06.axes();
        let mut builder = metadata_builder(&axes);
        let assets = TokenizerAssets {
            vocab: b"vocab".to_vec(),
            merges: b"merges".to_vec(),
            tokenizer_config: b"tokenizer".to_vec(),
            chat_template: b"chat".to_vec(),
            generation_config: b"generation".to_vec(),
        };
        assets.embed(&mut builder);
        assert_eq!(builder.metadata_count(), 60);
        let file =
            GgufFile::parse(builder.to_bytes().expect("serialize assets")).expect("parse assets");
        for (key, expected) in [
            (KEY_TOKENIZER_VOCAB, b"vocab".as_slice()),
            (KEY_TOKENIZER_MERGES, b"merges".as_slice()),
            (KEY_TOKENIZER_CONFIG, b"tokenizer".as_slice()),
            (KEY_CHAT_TEMPLATE, b"chat".as_slice()),
            (KEY_GENERATION_CONFIG, b"generation".as_slice()),
        ] {
            let GgufMetadataValue::Array(array) = file.get(key).expect("embedded key") else {
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
        let axes = Variant::B17.axes();
        let spec = ExactSidecar {
            name: "tiny.txt",
            bytes: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        std::fs::write(&path, b"abc").expect("write exact sidecar");
        assert_eq!(
            read_exact_sidecar(&directory, spec, &axes).expect("exact sidecar"),
            b"abc"
        );

        std::fs::write(&path, b"ab").expect("write short sidecar");
        assert!(
            read_exact_sidecar(&directory, spec, &axes)
                .expect_err("size drift")
                .to_string()
                .contains("expected exactly 3")
        );
        std::fs::write(&path, b"abd").expect("write hash-drift sidecar");
        assert!(
            read_exact_sidecar(&directory, spec, &axes)
                .expect_err("hash drift")
                .to_string()
                .contains("SHA-256")
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn shard_index_is_local_deduplicated_and_deterministic() {
        let directory = scratch_directory("index");
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            std::fs::write(directory.join(shard), []).expect("create shard placeholder");
        }
        let index = directory.join("model.safetensors.index.json");
        std::fs::write(
            &index,
            br#"{"weight_map":{"tensor.z":"model-00002-of-00002.safetensors","tensor.a":"model-00001-of-00002.safetensors","tensor.b":"model-00001-of-00002.safetensors"}}"#,
        )
        .expect("write index");
        let resolved = resolve_sources(&index).expect("resolve safe shard index");
        assert_eq!(resolved.files.len(), 2);
        assert_eq!(resolved.files[0].0, "model-00001-of-00002.safetensors");
        assert_eq!(resolved.files[1].0, "model-00002-of-00002.safetensors");
        assert_eq!(resolved.weight_map.expect("weight map").len(), 3);

        std::fs::write(
            &index,
            br#"{"weight_map":{"tensor":"../outside.safetensors"}}"#,
        )
        .expect("write unsafe index");
        assert!(
            resolve_sources(&index)
                .expect_err("path traversal")
                .to_string()
                .contains("unsafe/non-local")
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn conflicting_license_is_rejected_before_checkpoint_io() {
        let error = convert_qwen3_asr_file_with_variant(
            Path::new("does-not-exist.safetensors"),
            Path::new("unused.gguf"),
            Variant::B17,
            Some("MIT"),
        )
        .expect_err("conflicting license");
        assert!(matches!(error, ConvertError::Usage(_)));
        assert!(error.to_string().contains("pinned Apache-2.0"));
    }
}
