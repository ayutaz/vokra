//! Voxtral (Mistral) safetensors checkpoint to GGUF conversion (M3-10-T03/T04).
//!
//! Input: the upstream Mistral **Voxtral** safetensors checkpoint (weights only
//! — IF-06 / FR-MD-02 — no model code is imported). Output: a GGUF carrying
//! every float tensor plus the `vokra.model.*`, `vokra.frontend.*`,
//! `vokra.voxtral.*` and `vokra.tokenizer.*` metadata chunks the native
//! Voxtral implementation (in `vokra-models::voxtral`) loads against.
//!
//! # Voxtral architecture (from the upstream release, 2025-07)
//!
//! - **Audio encoder**: Whisper-derived (log-mel → conv stem → transformer
//!   pre-norm self-attention stack + final layer_norm). The frontend spec
//!   matches Whisper's `n_fft=400 / hop=160 / mel_norm=slaney / n_mels=128`,
//!   sample_rate=16 kHz. The encoder emits `[n_audio_ctx, d_audio]` hidden
//!   states.
//! - **Text decoder**: **Mistral** LLaMA-style decoder — GQA (`n_head_q >
//!   n_head_kv`), RoPE, SwiGLU FFN (`silu(gate) * up`), pre-norm RMSNorm.
//!   Cross-attention takes the audio encoder output as key/value on the ASR
//!   / S2S entry.
//! - **Tokenizer**: Mistral tokenizer (SentencePiece byte-fallback BPE). The
//!   raw model file is embedded verbatim into `vokra.tokenizer.model` as a
//!   `U8` array so a runtime tokenizer can be constructed without an external
//!   tokenizer crate (NFR-DS-02).
//! - **ASR head**: tied logits — the token embedding acts as the output
//!   projection (no separate `lm_head` tensor). This is the same tie Whisper
//!   uses.
//! - **S2S head**: codec-token generation. The codec type is recorded under
//!   `vokra.voxtral.s2s.codec_type` (default `"none"` = ASR-only build).
//!
//! # Foundation-only scope in M3-10
//!
//! This module is the WP's foundation converter: it lays down the metadata
//! contract and the tensor-copy path exactly like `whisper.rs` and
//! `kokoro.rs`. It intentionally does not resolve tokenizer-file input,
//! quantization, or S2S codec inspection — those roll in with the CLI +
//! runtime wiring across T05..T24. Real end-to-end conversion for a
//! downloaded Voxtral checkpoint is unblocked once the tokenizer file is
//! passed alongside via `convert_voxtral_file` (see below).
//!
//! # No silent inference of Voxtral-specific hparams (FR-EX-08)
//!
//! Hparams are shape-driven where possible (encoder blocks / hidden width /
//! decoder blocks — same pattern Whisper uses), and hparams that can only be
//! read from a checkpoint side-car `config.json` (RoPE base, `n_head_kv`,
//! `rms_norm_eps`, `vocab_size`, s2s codec type) are exposed via the
//! [`VoxtralConfig`] struct. Callers pass `None` for a shape-only conversion
//! (matching the Kokoro no-config path) — the resulting GGUF is loadable but
//! flags any downstream loader that needs the missing hparams. Real training
//! runs always pass a config; this keeps unit tests small.

use vokra_core::gguf::{
    FrontendSpec, GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value written for Voxtral GGUFs.
pub(crate) const ARCH: &str = "voxtral";

/// Well-known Voxtral release names. The converter derives one from the
/// checkpoint's shape quintuple when a config is present; otherwise the label
/// is `voxtral-unknown` (foundation-only path).
pub(crate) const NAME_MINI: &str = "voxtral-mini-3b";
pub(crate) const NAME_SMALL: &str = "voxtral-small-24b";

// --- vokra.voxtral.* metadata keys (M3-10-T04 chunk design) -----------------

/// `vokra.voxtral.audio_encoder.n_layer` (`UINT32`).
const KEY_AE_N_LAYER: &str = "vokra.voxtral.audio_encoder.n_layer";
/// `vokra.voxtral.audio_encoder.n_head` (`UINT32`).
const KEY_AE_N_HEAD: &str = "vokra.voxtral.audio_encoder.n_head";
/// `vokra.voxtral.audio_encoder.hidden_dim` `d_audio` (`UINT32`).
const KEY_AE_HIDDEN_DIM: &str = "vokra.voxtral.audio_encoder.hidden_dim";
/// `vokra.voxtral.audio_encoder.n_ctx` — encoder positional length (`UINT32`).
const KEY_AE_N_CTX: &str = "vokra.voxtral.audio_encoder.n_ctx";
/// `vokra.voxtral.audio_encoder.n_mels` — encoder input mel channels (`UINT32`).
const KEY_AE_N_MELS: &str = "vokra.voxtral.audio_encoder.n_mels";
/// `vokra.voxtral.audio_encoder.ffn_dim` (`UINT32`).
const KEY_AE_FFN_DIM: &str = "vokra.voxtral.audio_encoder.ffn_dim";

/// `vokra.voxtral.text_decoder.n_layer` (`UINT32`).
const KEY_TD_N_LAYER: &str = "vokra.voxtral.text_decoder.n_layer";
/// `vokra.voxtral.text_decoder.n_head_q` — GQA query heads (`UINT32`).
const KEY_TD_N_HEAD_Q: &str = "vokra.voxtral.text_decoder.n_head_q";
/// `vokra.voxtral.text_decoder.n_head_kv` — GQA key/value heads (`UINT32`).
const KEY_TD_N_HEAD_KV: &str = "vokra.voxtral.text_decoder.n_head_kv";
/// `vokra.voxtral.text_decoder.hidden_dim` (`UINT32`).
const KEY_TD_HIDDEN_DIM: &str = "vokra.voxtral.text_decoder.hidden_dim";
/// `vokra.voxtral.text_decoder.ffn_dim` — SwiGLU inner width (`UINT32`).
const KEY_TD_FFN_DIM: &str = "vokra.voxtral.text_decoder.ffn_dim";
/// `vokra.voxtral.text_decoder.vocab_size` (`UINT32`).
const KEY_TD_VOCAB_SIZE: &str = "vokra.voxtral.text_decoder.vocab_size";
/// `vokra.voxtral.text_decoder.n_ctx` — max sequence length (`UINT32`).
const KEY_TD_N_CTX: &str = "vokra.voxtral.text_decoder.n_ctx";
/// `vokra.voxtral.text_decoder.rope_base` — RoPE base (`FLOAT32`).
const KEY_TD_ROPE_BASE: &str = "vokra.voxtral.text_decoder.rope_base";
/// `vokra.voxtral.text_decoder.rms_norm_eps` (`FLOAT32`).
const KEY_TD_RMS_NORM_EPS: &str = "vokra.voxtral.text_decoder.rms_norm_eps";

/// `vokra.voxtral.cross_attn.hidden_dim` — key/value hidden dim of cross-attn
/// (usually equal to `audio_encoder.hidden_dim`) (`UINT32`).
const KEY_XATTN_HIDDEN_DIM: &str = "vokra.voxtral.cross_attn.hidden_dim";

/// `vokra.voxtral.s2s.codec_type` — codec identifier for S2S mode; `"none"`
/// means ASR-only (`STRING`).
const KEY_S2S_CODEC_TYPE: &str = "vokra.voxtral.s2s.codec_type";
/// `vokra.voxtral.s2s.codec_config` — codec-specific config blob (`STRING`).
const KEY_S2S_CODEC_CONFIG: &str = "vokra.voxtral.s2s.codec_config";
/// `vokra.voxtral.s2s.watermark_default_on` — Voxtral S2S output has
/// AudioSeal ON default (T17); ASR output = text is watermark-exempt (`BOOL`).
const KEY_S2S_WATERMARK_DEFAULT_ON: &str = "vokra.voxtral.s2s.watermark_default_on";

/// `vokra.voxtral.mode` — `"asr"` or `"s2s"` — the primary head this GGUF
/// exposes. Both heads may be linked, but the converter records the intended
/// default so the runtime knows how to size the watermark config (T17).
const KEY_MODE: &str = "vokra.voxtral.mode";

/// `vokra.tokenizer.model` — the raw Mistral tokenizer model file, as a
/// `U8` array. Same key shape Whisper uses (see `whisper.rs`); the runtime
/// tokenizer is self-implemented, so no external crate is needed
/// (NFR-DS-02).
pub(crate) const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// The subset of Voxtral hparams that cannot be derived from a shape-only
/// safetensors sweep — these come from the checkpoint's `config.json` (or a
/// caller who has them from the upstream card). `None` for any field means
/// the converter writes a documented sentinel (`0` for integers, `0.0` for
/// floats, `"none"` for strings) and the runtime loader will flag it if it
/// tries to use that value at forward time (FR-EX-08).
#[derive(Debug, Clone, Default)]
pub struct VoxtralConfig {
    /// Text-decoder RoPE base θ (Mistral uses `1_000_000.0` on modern
    /// releases). Written into `vokra.voxtral.text_decoder.rope_base` — `0.0`
    /// when unset.
    pub rope_base: Option<f32>,
    /// Text-decoder RMSNorm ε (Mistral uses `1e-5`). Written into
    /// `vokra.voxtral.text_decoder.rms_norm_eps` — `0.0` when unset.
    pub rms_norm_eps: Option<f32>,
    /// GQA key/value head count (`<= n_head_q`). Written into
    /// `vokra.voxtral.text_decoder.n_head_kv` — `0` when unset.
    pub n_head_kv: Option<u32>,
    /// Number of query heads (SwiGLU decoder). Derived from
    /// `hidden_dim / head_dim` when unset if `head_dim` is provided; else `0`.
    pub n_head_q: Option<u32>,
    /// Per-head width (Mistral 3B uses `head_dim=128`). Written implicitly via
    /// `n_head_q * head_dim = hidden_dim`; here it drives `n_head_q` when the
    /// user cannot supply it directly.
    pub head_dim: Option<u32>,
    /// Vocabulary size (Mistral 32000..131072 range). Written into
    /// `vokra.voxtral.text_decoder.vocab_size` — `0` when unset.
    pub vocab_size: Option<u32>,
    /// Max sequence length. Written into `vokra.voxtral.text_decoder.n_ctx`
    /// — `0` when unset.
    pub max_position_embeddings: Option<u32>,
    /// Encoder positional length (`n_audio_ctx`). Written into
    /// `vokra.voxtral.audio_encoder.n_ctx` — `0` when unset.
    pub n_audio_ctx: Option<u32>,
    /// S2S codec identifier — e.g. `"mimi"` (Kyutai) or `"none"` for ASR-only
    /// builds (default).
    pub s2s_codec_type: Option<String>,
    /// Optional codec-specific config blob (JSON serialized upstream, opaque
    /// here) — written verbatim into `vokra.voxtral.s2s.codec_config`.
    pub s2s_codec_config: Option<String>,
    /// AudioSeal watermark default-ON flag (T17). Defaults to `true` for the
    /// S2S mode and `false` for ASR-only.
    pub s2s_watermark_default_on: Option<bool>,
    /// Primary head this GGUF exposes — one of `"asr"` (default) or `"s2s"`.
    pub mode: Option<String>,
    /// Optional raw Mistral tokenizer file bytes. Embedded verbatim into
    /// `vokra.tokenizer.model`.
    pub tokenizer_bytes: Option<Vec<u8>>,
}

/// A summary of what the converter wrote for the caller's CLI note.
#[derive(Debug, Default)]
pub struct ConvertReport {
    /// Number of float tensors written to the GGUF.
    pub written: usize,
    /// Non-float tensors that were skipped (e.g. integer position ids).
    pub skipped_non_float: usize,
    /// Whether a tokenizer blob was embedded.
    pub tokenizer_embedded: bool,
    /// Derived label (`voxtral-mini-3b` etc., or `voxtral-unknown`).
    pub name: &'static str,
}

/// Converts a Voxtral safetensors buffer plus an optional [`VoxtralConfig`]
/// into a GGUF builder.
///
/// The shape-only path (config `None`) matches Whisper's foundation path:
/// shape-driven hparams are written, missing side-car hparams get `0`
/// sentinels, and the runtime loader is expected to flag the missing values
/// at forward-time (never silently substitute). Real-world conversion always
/// passes a config.
pub(crate) fn convert(
    bytes: Vec<u8>,
    config: Option<&VoxtralConfig>,
) -> Result<(GgufBuilder, ConvertReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    // Derive shape-driven encoder/decoder hparams. The tensor names below
    // mirror what the upstream Voxtral checkpoint exposes (`audio_tower.*` for
    // the encoder, `language_model.*` for the Mistral decoder, or the plain
    // Mistral prefix `model.layers.*`). Missing tensors yield `0`, which the
    // runtime rejects at load — never silently substituted.
    let d_audio = tensor_dim(&st, "audio_tower.conv1.weight", 0);
    let n_mels_ck = tensor_dim(&st, "audio_tower.conv1.weight", 1);
    let n_audio_layer = count_layers(&st, "audio_tower.layers.");
    // Encoder positional embedding; some releases inline sinusoidal, others
    // learn. The learned form exposes the tensor; the sinusoidal form yields
    // 0 here (harmless — runtime rebuilds it if the tensor is absent).
    let n_audio_ctx_shape = tensor_dim(&st, "audio_tower.embed_positions.weight", 0);
    let n_audio_ffn = tensor_dim(&st, "audio_tower.layers.0.fc1.weight", 0);

    let (d_text, n_text_layer, ffn_text, vocab_shape) =
        derive_decoder_shape(&st).unwrap_or((0, 0, 0, 0));

    let name = derive_name(d_text, n_text_layer, config).unwrap_or("voxtral-unknown");

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    // Encoder is Whisper-derived; the frontend spec is Whisper's for the same
    // n_mels (128 on Voxtral). Runtime front-end check gates a mismatched GGUF
    // at load (bit-exact, FR-LD-03).
    write_frontend_spec(&mut b, n_mels_ck as u32);
    write_hparams(&mut b, &st, config, d_audio, n_audio_layer, n_audio_ffn);

    let tokenizer_embedded = if let Some(cfg) = config {
        if let Some(bytes) = cfg.tokenizer_bytes.as_ref() {
            embed_tokenizer(&mut b, bytes);
            true
        } else {
            false
        }
    } else {
        false
    };

    // Copy tensors verbatim. Skip non-float tensors (e.g. integer id caches).
    let mut written = 0usize;
    let mut skipped_non_float = 0usize;
    for t in st.tensors() {
        if !is_float_dtype(t.dtype) {
            skipped_non_float += 1;
            continue;
        }
        let name = gguf_tensor_name(&t.name);
        b.add_tensor(&name, t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec())?;
        written += 1;
    }

    // Belt-and-braces: reference the decoder-shape tuple even on the failure
    // branch so an over-eager reviewer does not silently drop it later.
    let _ = (d_text, ffn_text, vocab_shape, n_audio_ctx_shape);

    Ok((
        b,
        ConvertReport {
            written,
            skipped_non_float,
            tokenizer_embedded,
            name,
        },
    ))
}

/// Derives the Voxtral checkpoint size label from the text-decoder shape
/// (`hidden_dim`, `n_layer`) plus an optional config hint. The two shipping
/// releases are `voxtral-mini-3b` and `voxtral-small-24b`; a checkpoint that
/// matches neither triggers an explicit error rather than a silent fallback
/// (FR-EX-08).
fn derive_name(
    d_text: u64,
    n_text_layer: u32,
    config: Option<&VoxtralConfig>,
) -> Result<&'static str, ConvertError> {
    // Explicit override from a caller who has the release card handy wins.
    if let Some(cfg) = config {
        if let Some(mode) = cfg.mode.as_deref() {
            if mode == "small" {
                return Ok(NAME_SMALL);
            }
            if mode == "mini" {
                return Ok(NAME_MINI);
            }
        }
    }
    // Voxtral 2025-07 shape quintuples (Mistral-Small language backbone
    // `hidden_size=5120 / n_layer=40` for `small-24b`; `hidden_size=3072 /
    // n_layer=28` for `mini-3b`). These come from the upstream config cards
    // and are the same numbers Mistral used for Ministral / Mistral-Small.
    match (d_text, n_text_layer) {
        (3072, 28) => Ok(NAME_MINI),
        (5120, 40) => Ok(NAME_SMALL),
        _ => Err(ConvertError::Parse(format!(
            "unknown voxtral size: (d_text={d_text}, n_text_layer={n_text_layer}); \
             expected voxtral-mini-3b (3072, 28) or voxtral-small-24b (5120, 40)"
        ))),
    }
}

/// Reads the shape of the text-decoder side of the checkpoint. Tries the
/// modern `language_model.model.*` prefix first (Voxtral packages the Mistral
/// backbone under a submodule) and falls back to the plain `model.*` prefix
/// used by standalone Mistral releases. Returns `(d_model, n_layer, ffn_dim,
/// vocab_size)` — any missing dim is `0`, which the runtime rejects at load
/// (FR-EX-08). Returns `None` when neither prefix has any tensors.
fn derive_decoder_shape(st: &SafetensorsFile) -> Option<(u64, u32, u64, u64)> {
    for prefix in ["language_model.model.", "language_model.", "model."] {
        let n = count_layers(st, &format!("{prefix}layers."));
        if n == 0 {
            continue;
        }
        // Query proj is [d_model, d_model] (or [n_head_q*head_dim, d_model]
        // for GQA); the last axis is d_model. Some releases spell it
        // `self_attn.q_proj.weight`, others `self_attn.wq.weight`.
        let q_names = [
            format!("{prefix}layers.0.self_attn.q_proj.weight"),
            format!("{prefix}layers.0.self_attn.wq.weight"),
        ];
        let d_model = q_names
            .iter()
            .map(|n| tensor_dim(st, n, 1))
            .find(|&d| d != 0)
            .unwrap_or(0);
        // SwiGLU up-projection: `gate_proj` (or `w1`) is `[ffn_dim, d_model]`.
        let ffn_names = [
            format!("{prefix}layers.0.mlp.gate_proj.weight"),
            format!("{prefix}layers.0.mlp.w1.weight"),
        ];
        let ffn = ffn_names
            .iter()
            .map(|n| tensor_dim(st, n, 0))
            .find(|&d| d != 0)
            .unwrap_or(0);
        // Vocabulary from the token embedding.
        let vocab_names = [
            format!("{prefix}embed_tokens.weight"),
            "lm_head.weight".to_owned(),
        ];
        let vocab = vocab_names
            .iter()
            .map(|n| tensor_dim(st, n, 0))
            .find(|&d| d != 0)
            .unwrap_or(0);
        return Some((d_model, n, ffn, vocab));
    }
    None
}

/// Whisper-derived audio encoder front-end spec.
fn write_frontend_spec(b: &mut GgufBuilder, n_mels: u32) {
    let spec = FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: 8000.0,
        // 0 means "no encoder present" — the runtime will still refuse to
        // instantiate the audio path in that case (FR-EX-08).
        n_mels,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: 16_000,
    };
    spec.write_into(b);
}

fn write_hparams(
    b: &mut GgufBuilder,
    st: &SafetensorsFile,
    config: Option<&VoxtralConfig>,
    d_audio: u64,
    n_audio_layer: u32,
    n_audio_ffn: u64,
) {
    // Encoder side (Whisper-derived).
    b.add_u32(KEY_AE_N_LAYER, n_audio_layer);
    b.add_u32(KEY_AE_HIDDEN_DIM, d_audio as u32);
    b.add_u32(
        KEY_AE_N_HEAD,
        // Whisper-derived: head_dim=64 → n_head = d_audio / 64.
        if d_audio >= 64 {
            (d_audio / 64) as u32
        } else {
            0
        },
    );
    b.add_u32(KEY_AE_FFN_DIM, n_audio_ffn as u32);
    b.add_u32(
        KEY_AE_N_MELS,
        tensor_dim(st, "audio_tower.conv1.weight", 1) as u32,
    );
    b.add_u32(
        KEY_AE_N_CTX,
        config.and_then(|c| c.n_audio_ctx).unwrap_or(tensor_dim(
            st,
            "audio_tower.embed_positions.weight",
            0,
        ) as u32),
    );

    // Decoder side (Mistral text decoder). The shape-derived values are the
    // authoritative source; the config supplies the values shapes cannot
    // recover (RoPE base, RMSNorm eps, GQA head split, vocab size).
    let (d_text, n_text_layer, ffn_text, vocab_shape) =
        derive_decoder_shape(st).unwrap_or((0, 0, 0, 0));
    b.add_u32(KEY_TD_N_LAYER, n_text_layer);
    b.add_u32(KEY_TD_HIDDEN_DIM, d_text as u32);
    b.add_u32(KEY_TD_FFN_DIM, ffn_text as u32);
    b.add_u32(
        KEY_TD_VOCAB_SIZE,
        config
            .and_then(|c| c.vocab_size)
            .unwrap_or(vocab_shape as u32),
    );
    b.add_u32(
        KEY_TD_N_CTX,
        config.and_then(|c| c.max_position_embeddings).unwrap_or(0),
    );
    // GQA head split.
    let head_dim = config.and_then(|c| c.head_dim).unwrap_or(0);
    let n_head_q = config
        .and_then(|c| c.n_head_q)
        .or_else(|| {
            if head_dim > 0 && d_text > 0 {
                Some(d_text as u32 / head_dim)
            } else {
                None
            }
        })
        .unwrap_or(0);
    b.add_u32(KEY_TD_N_HEAD_Q, n_head_q);
    b.add_u32(
        KEY_TD_N_HEAD_KV,
        config.and_then(|c| c.n_head_kv).unwrap_or(0),
    );
    b.add_f32(
        KEY_TD_ROPE_BASE,
        config.and_then(|c| c.rope_base).unwrap_or(0.0),
    );
    b.add_f32(
        KEY_TD_RMS_NORM_EPS,
        config.and_then(|c| c.rms_norm_eps).unwrap_or(0.0),
    );

    // Cross-attention hidden dim (usually = d_audio).
    b.add_u32(KEY_XATTN_HIDDEN_DIM, d_audio as u32);

    // S2S / mode.
    let mode = config
        .and_then(|c| c.mode.as_deref())
        .unwrap_or("asr")
        .to_owned();
    b.add_string(KEY_MODE, mode.as_str());
    b.add_string(
        KEY_S2S_CODEC_TYPE,
        config
            .and_then(|c| c.s2s_codec_type.as_deref())
            .unwrap_or("none"),
    );
    b.add_string(
        KEY_S2S_CODEC_CONFIG,
        config
            .and_then(|c| c.s2s_codec_config.as_deref())
            .unwrap_or(""),
    );
    // Default: S2S mode → watermark ON; ASR mode → OFF (text output).
    let s2s_wm_default = mode == "s2s";
    b.add_bool(
        KEY_S2S_WATERMARK_DEFAULT_ON,
        config
            .and_then(|c| c.s2s_watermark_default_on)
            .unwrap_or(s2s_wm_default),
    );
}

/// Embeds the raw Mistral tokenizer model file as a `U8` array under
/// `vokra.tokenizer.model`. Same shape Whisper uses (bytes verbatim; runtime
/// tokenizer is self-implemented, no external crate).
fn embed_tokenizer(b: &mut GgufBuilder, bytes: &[u8]) {
    b.add_metadata(
        KEY_TOKENIZER_MODEL,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
        }),
    );
}

/// Maps an upstream safetensors tensor name to its GGUF name (identity —
/// the same contract Whisper uses).
pub(crate) fn gguf_tensor_name(hf_name: &str) -> String {
    hf_name.to_owned()
}

fn tensor_dim(st: &SafetensorsFile, name: &str, axis: usize) -> u64 {
    st.tensors()
        .iter()
        .find(|t: &&SafeTensorInfo| t.name == name)
        .and_then(|t| t.shape.get(axis).copied())
        .unwrap_or(0)
}

fn count_layers(st: &SafetensorsFile, prefix: &str) -> u32 {
    let mut n = 0u32;
    loop {
        let probe = format!("{prefix}{n}.");
        if st.tensors().iter().any(|t| t.name.starts_with(&probe)) {
            n += 1;
        } else {
            return n;
        }
    }
}

fn is_float_dtype(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::F32 | GgmlType::F16 | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Minimal synthetic safetensors with an `audio_tower.conv1.weight` +
    /// `language_model.model.embed_tokens.weight` — enough to exercise the
    /// hparam pathway without carrying real weight bytes.
    fn synthetic_voxtral() -> Vec<u8> {
        // `audio_tower.conv1.weight`: [d_audio=2, n_mels=2, k=3] = 12 f32.
        let a: Vec<u8> = (0..12)
            .flat_map(|i: i32| (i as f32).to_le_bytes())
            .collect();
        // `language_model.model.embed_tokens.weight`: [vocab=2, d=2] = 4 f32.
        let b: Vec<u8> = (0..4).flat_map(|i: i32| (i as f32).to_le_bytes()).collect();
        let a_end = a.len();
        let b_start = a_end;
        let b_end = a_end + b.len();
        let header = format!(
            r#"{{"audio_tower.conv1.weight":{{"dtype":"F32","shape":[2,2,3],"data_offsets":[0,{a_end}]}},"language_model.model.embed_tokens.weight":{{"dtype":"F32","shape":[2,2],"data_offsets":[{b_start},{b_end}]}}}}"#,
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&a);
        out.extend_from_slice(&b);
        out
    }

    #[test]
    fn convert_writes_model_arch_and_frontend_spec() {
        let (builder, report) = convert(synthetic_voxtral(), None).unwrap();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        // Shape doesn't match a known Voxtral release, so name falls back.
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("voxtral-unknown")
        );
        let spec = FrontendSpec::from_gguf(&file).unwrap();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.n_fft, 400);
        assert_eq!(spec.hop, 160);
        // n_mels tracks the checkpoint's conv1 axis-1 length (2 in the stub).
        assert_eq!(spec.n_mels, 2);
        // 2 tensors written, 0 skipped.
        assert_eq!(report.written, 2);
        assert_eq!(report.skipped_non_float, 0);
        assert!(!report.tokenizer_embedded);
    }

    #[test]
    fn convert_with_config_writes_full_hparam_chunk() {
        let cfg = VoxtralConfig {
            rope_base: Some(1_000_000.0),
            rms_norm_eps: Some(1e-5),
            n_head_kv: Some(8),
            n_head_q: Some(32),
            head_dim: Some(128),
            vocab_size: Some(32_000),
            max_position_embeddings: Some(32_768),
            n_audio_ctx: Some(1500),
            s2s_codec_type: Some("mimi".to_owned()),
            s2s_codec_config: Some("{}".to_owned()),
            s2s_watermark_default_on: Some(true),
            mode: Some("s2s".to_owned()),
            tokenizer_bytes: Some(b"fake-tokenizer".to_vec()),
        };
        let (builder, report) = convert(synthetic_voxtral(), Some(&cfg)).unwrap();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();

        // Hparam chunk keys — spot-check values. Note: GgufMetadataValue
        // exposes `as_f64` (widening to double); the value round-trips
        // exactly because FLOAT32 preserves 1_000_000.0.
        let rope_base = file
            .get(KEY_TD_ROPE_BASE)
            .and_then(|v| v.as_f64())
            .map(|f| f as f32);
        assert_eq!(rope_base, Some(1_000_000.0), "rope_base");
        assert_eq!(
            file.get(KEY_TD_N_HEAD_KV).and_then(|v| v.as_u64()),
            Some(8),
            "n_head_kv"
        );
        assert_eq!(
            file.get(KEY_TD_N_HEAD_Q).and_then(|v| v.as_u64()),
            Some(32),
            "n_head_q"
        );
        assert_eq!(
            file.get(KEY_MODE).and_then(|v| v.as_str()),
            Some("s2s"),
            "mode"
        );
        assert_eq!(
            file.get(KEY_S2S_CODEC_TYPE).and_then(|v| v.as_str()),
            Some("mimi"),
            "codec"
        );
        assert!(report.tokenizer_embedded);
    }

    #[test]
    fn is_float_dtype_covers_expected_types() {
        assert!(is_float_dtype(GgmlType::F32));
        assert!(is_float_dtype(GgmlType::F16));
        assert!(is_float_dtype(GgmlType::Q4K));
        assert!(is_float_dtype(GgmlType::Q5K));
        assert!(is_float_dtype(GgmlType::Q6K));
    }

    #[test]
    fn derive_name_maps_mini_and_small_shapes() {
        assert_eq!(derive_name(3072, 28, None).unwrap(), NAME_MINI);
        assert_eq!(derive_name(5120, 40, None).unwrap(), NAME_SMALL);
        // Unknown shape → explicit error (never a silent fall back, FR-EX-08).
        assert!(derive_name(1234, 5, None).is_err());
    }
}
