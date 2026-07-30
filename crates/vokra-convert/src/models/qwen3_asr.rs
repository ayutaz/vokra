//! Alibaba **Qwen3-ASR** family (`Qwen/Qwen3-ASR-0.6B` and
//! `Qwen/Qwen3-ASR-1.7B`, apache-2.0): safetensors → GGUF conversion
//! (SoTA plan Phase 5 ASR fleet, 2026-07-30).
//!
//! Input: the upstream `Qwen/Qwen3-ASR-{0.6B,1.7B}` release — a
//! Qwen3-flavour multilingual ASR checkpoint (`Qwen3ASRForConditionalGeneration`
//! architecture per the HF API `architectures` field, primary source
//! `huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json`). Output:
//! a GGUF carrying every float tensor verbatim under its upstream
//! safetensors name, plus the `vokra.qwen3_asr.*` hparam chunk group
//! and `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native Qwen3-ASR loader will read.
//!
//! # HF / licence / category
//!
//! - Upstream HF: `Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B`
//!   (recorded under `vokra.provenance.upstream_hf`). Both share the
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
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters.
//! No convert-time widening; runtime widens BF16 → f32 losslessly via
//! the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec / Wespeaker contract). Real-weight binding is
//! a follow-up wave gated on the upstream tensor-name manifest fetch;
//! this converter passes every F32 / F16 / BF16 tensor through
//! unchanged so a future `Qwen3AsrWeights::from_gguf` can walk the
//! same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream Qwen HF pipeline is
//! deferred to owner (`docs/license-audit.md` §3.1 sign-off) — this
//! converter provides the byte-parallel GGUF surface only.
//!
//! # No ONNX (permanent)
//!
//! Qwen3-ASR is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/qwen3_asr/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

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
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`); kept for
    /// symmetry with the sibling `qwen3_tts` / `vibevoice` / `voxcpm2` /
    /// `neucodec` / `wespeaker` reports).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). The upstream Qwen3-ASR release is BF16
    /// (`dtype="bfloat16"` in `config.json`), so a real checkpoint
    /// increments this counter for essentially every tensor.
    pub bf16_passthrough: usize,
}

/// Variant-taking file-based Qwen3-ASR converter — the CLI dispatch
/// arm picks the [`Variant`] from the `--model` string
/// (`qwen3-asr-0.6b` / `qwen3-asr-1.7b`).
///
/// Reads `input` (the upstream `Qwen/Qwen3-ASR-{0.6B,1.7B}`
/// `model.safetensors`), writes a Vokra GGUF to `output`. `license`
/// overrides the default `apache-2.0` provenance stamp (Whisper /
/// kokoro-family override pattern — see `convert_file_licensed` in
/// `lib.rs`); pass `None` to keep the built-in stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_qwen3_asr_file_with_variant(
    input: &Path,
    output: &Path,
    variant: Variant,
    license: Option<&str>,
) -> Result<Qwen3AsrReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let axes = variant.axes();

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, axes.name);
    // Category / upstream-HF stamps — not covered by `stamp_provenance`
    // (which handles the SPDX + class + model_id + source group only),
    // so written directly. Consumers pick a load path by category and
    // trace the artifact back to its serving location by upstream_hf.
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, axes.upstream_hf);

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream HF model-card, verified
    // 2026-07-30 via `cardData.license`). `license` overrides for
    // callers who obtained the weight under a different SPDX (see
    // `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("apache-2.0".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(axes.name),
        Some(&format!(
            "{} (Qwen3-flavour multilingual ASR: audio encoder + Qwen3 LM decoder, apache-2.0)",
            axes.upstream_hf
        )),
    );

    // Write the `vokra.qwen3_asr.*` hparam chunk — every axis
    // transcribed verbatim from the primary-source config.json (never
    // invented; the runtime rejects `0` sentinels loudly per FR-EX-08).
    write_hparams(&mut b, &axes);

    let mut report = Qwen3AsrReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30), same posture as
    // qwen3_tts / vibevoice / voxcpm2 / wespeaker / neucodec; runtime
    // widens BF16 → f32 exactly at load via
    // `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
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

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Mirrors the neucodec / wespeaker
    /// test helper.
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16).
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            f32_elems as usize * 4,
            "F32 payload len must match shape × 4"
        );
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(
            f16_bytes.len(),
            f16_elems as usize * 2,
            "F16 payload len must match shape × 2"
        );
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-qwen3-asr-{kind}-{}-{}.bin",
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
    fn bf16_tensor_passes_through_verbatim_1_7b() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a plausible upstream Qwen3-ASR tensor name (audio
        // encoder q_proj) so the round-trip exercises a realistic
        // string.
        let input_bytes = safetensors_one_bf16(
            "thinker.audio_tower.encoder.layers.0.self_attn.q_proj.weight",
            &[2, 3],
            &bf16,
        );
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_qwen3_asr_file(&input_path, &output_path, None)
            .expect("convert_qwen3_asr_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2)"
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
            .tensor_info("thinker.audio_tower.encoder.layers.0.self_attn.q_proj.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
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
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let input_bytes = safetensors_f32_then_f16(
            "thinker.text.embed_tokens.weight",
            &[1, 2],
            &f32_bytes,
            "thinker.audio_tower.conv1.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report =
            convert_qwen3_asr_file_with_variant(&input_path, &output_path, Variant::B17, None)
                .expect("mixed F32/F16 must convert");

        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output_path).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse output");

        // Provenance / category chunks landed for the 1.7B default.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("qwen3-asr-1.7b")
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
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("Qwen/Qwen3-ASR-1.7B")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn hparam_chunk_pins_0_6b_axes() {
        // A tiny synthetic checkpoint just to exercise the hparam write
        // path. The interesting assertions are the hparam values, which
        // must match the 0.6B config.json verbatim.
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);
        let input_path = write_temp("0.6b-hparam-in", &bytes);
        let output_path = write_temp("0.6b-hparam-out", &[]);

        let report =
            convert_qwen3_asr_file_with_variant(&input_path, &output_path, Variant::B06, None)
                .expect("0.6b conversion must succeed");
        assert_eq!(report.written, 1);

        let out = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out).expect("parse");

        // Model name / upstream_hf reflect the 0.6B variant.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("qwen3-asr-0.6b")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("Qwen/Qwen3-ASR-0.6B")
        );

        // Audio encoder axes — every value transcribed from
        // huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json.
        assert_eq!(
            file.get(KEY_AUDIO_D_MODEL).and_then(|v| v.as_u64()),
            Some(896)
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_LAYER).and_then(|v| v.as_u64()),
            Some(18)
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_HEAD).and_then(|v| v.as_u64()),
            Some(14)
        );
        assert_eq!(
            file.get(KEY_AUDIO_FFN_DIM).and_then(|v| v.as_u64()),
            Some(3584)
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_MELS).and_then(|v| v.as_u64()),
            Some(128)
        );
        assert_eq!(
            file.get(KEY_AUDIO_MAX_SOURCE_POSITIONS)
                .and_then(|v| v.as_u64()),
            Some(1500)
        );
        assert_eq!(
            file.get(KEY_AUDIO_OUTPUT_DIM).and_then(|v| v.as_u64()),
            Some(1024)
        );
        assert_eq!(
            file.get(KEY_AUDIO_DOWNSAMPLE_HIDDEN_SIZE)
                .and_then(|v| v.as_u64()),
            Some(480)
        );
        assert_eq!(
            file.get(KEY_AUDIO_CONV_CHUNKSIZE).and_then(|v| v.as_u64()),
            Some(500)
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_WINDOW).and_then(|v| v.as_u64()),
            Some(50)
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_WINDOW_INFER).and_then(|v| v.as_u64()),
            Some(800)
        );

        // Text decoder (Qwen3 LM) axes.
        assert_eq!(
            file.get(KEY_TEXT_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(1024)
        );
        assert_eq!(
            file.get(KEY_TEXT_N_LAYER).and_then(|v| v.as_u64()),
            Some(28)
        );
        assert_eq!(file.get(KEY_TEXT_N_HEAD).and_then(|v| v.as_u64()), Some(16));
        assert_eq!(
            file.get(KEY_TEXT_N_KV_HEAD).and_then(|v| v.as_u64()),
            Some(8)
        );
        assert_eq!(
            file.get(KEY_TEXT_HEAD_DIM).and_then(|v| v.as_u64()),
            Some(128)
        );
        assert_eq!(
            file.get(KEY_TEXT_FFN_DIM).and_then(|v| v.as_u64()),
            Some(3072)
        );
        assert_eq!(
            file.get(KEY_TEXT_MAX_POSITION_EMBEDDINGS)
                .and_then(|v| v.as_u64()),
            Some(65536)
        );
        assert_eq!(
            file.get(KEY_TEXT_VOCAB_SIZE).and_then(|v| v.as_u64()),
            Some(151_936)
        );

        // Adapter tokens.
        assert_eq!(
            file.get(KEY_AUDIO_START_TOKEN_ID).and_then(|v| v.as_u64()),
            Some(151_669)
        );
        assert_eq!(
            file.get(KEY_AUDIO_END_TOKEN_ID).and_then(|v| v.as_u64()),
            Some(151_670)
        );
        assert_eq!(
            file.get(KEY_AUDIO_TOKEN_ID).and_then(|v| v.as_u64()),
            Some(151_676)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn hparam_chunk_pins_1_7b_axes() {
        // Only the axes that DIFFER from 0.6B — the 0.6b test above
        // pins the shared ones. Different-vs-same axes are the entire
        // reason both variants exist.
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);
        let input_path = write_temp("1.7b-hparam-in", &bytes);
        let output_path = write_temp("1.7b-hparam-out", &[]);

        let _ = convert_qwen3_asr_file_with_variant(&input_path, &output_path, Variant::B17, None)
            .expect("1.7b conversion must succeed");

        let out = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out).expect("parse");

        // Audio encoder axes that differ from 0.6B.
        assert_eq!(
            file.get(KEY_AUDIO_D_MODEL).and_then(|v| v.as_u64()),
            Some(1024),
            "1.7B audio d_model = 1024 (0.6B = 896)"
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_LAYER).and_then(|v| v.as_u64()),
            Some(24),
            "1.7B audio n_layer = 24 (0.6B = 18)"
        );
        assert_eq!(
            file.get(KEY_AUDIO_N_HEAD).and_then(|v| v.as_u64()),
            Some(16),
            "1.7B audio n_head = 16 (0.6B = 14)"
        );
        assert_eq!(
            file.get(KEY_AUDIO_FFN_DIM).and_then(|v| v.as_u64()),
            Some(4096),
            "1.7B audio ffn_dim = 4096 (0.6B = 3584)"
        );
        assert_eq!(
            file.get(KEY_AUDIO_OUTPUT_DIM).and_then(|v| v.as_u64()),
            Some(2048),
            "1.7B audio output_dim = 2048 (0.6B = 1024)"
        );

        // Text decoder axes that differ from 0.6B.
        assert_eq!(
            file.get(KEY_TEXT_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(2048),
            "1.7B text hidden_size = 2048 (0.6B = 1024)"
        );
        assert_eq!(
            file.get(KEY_TEXT_FFN_DIM).and_then(|v| v.as_u64()),
            Some(6144),
            "1.7B text ffn_dim = 6144 (0.6B = 3072)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn license_override_replaces_stamp() {
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);
        let input_path = write_temp("lic-in", &bytes);
        let output_path = write_temp("lic-out", &[]);

        // Override to a fictional MIT variant to prove the flag flows
        // through. Real callers ride the apache-2.0 default; this only
        // exercises the override plumbing.
        let _ = convert_qwen3_asr_file(&input_path, &output_path, Some("MIT"))
            .expect("license override must succeed");

        let out = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out).expect("parse");

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("MIT")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT still classifies as Permissive"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
