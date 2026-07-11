//! Voxtral hyperparameters read from GGUF `vokra.voxtral.*` metadata.
//!
//! Nothing here is hard-coded (FR-LD-02 / FR-MD-02): every field is read
//! from a metadata key `vokra-convert` wrote from the checkpoint's tensor
//! shapes (audio-encoder side) or the caller's `VoxtralConfig` side-car
//! (text-decoder side: RoPE base, RMSNorm ε, GQA head split, vocab size).
//!
//! The key strings are **duplicated verbatim** from
//! `vokra-convert/src/models/voxtral.rs` because the two crates cannot
//! depend on each other. Centralising them in `vokra_core::gguf::chunks` is
//! a follow-up (same posture as `whisper::config`).

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

// ----- Audio-encoder side -------------------------------------------------

const KEY_AE_N_LAYER: &str = "vokra.voxtral.audio_encoder.n_layer";
const KEY_AE_N_HEAD: &str = "vokra.voxtral.audio_encoder.n_head";
const KEY_AE_HIDDEN_DIM: &str = "vokra.voxtral.audio_encoder.hidden_dim";
const KEY_AE_N_CTX: &str = "vokra.voxtral.audio_encoder.n_ctx";
const KEY_AE_N_MELS: &str = "vokra.voxtral.audio_encoder.n_mels";
const KEY_AE_FFN_DIM: &str = "vokra.voxtral.audio_encoder.ffn_dim";

// ----- Text-decoder side --------------------------------------------------

const KEY_TD_N_LAYER: &str = "vokra.voxtral.text_decoder.n_layer";
const KEY_TD_N_HEAD_Q: &str = "vokra.voxtral.text_decoder.n_head_q";
const KEY_TD_N_HEAD_KV: &str = "vokra.voxtral.text_decoder.n_head_kv";
const KEY_TD_HIDDEN_DIM: &str = "vokra.voxtral.text_decoder.hidden_dim";
const KEY_TD_FFN_DIM: &str = "vokra.voxtral.text_decoder.ffn_dim";
const KEY_TD_VOCAB_SIZE: &str = "vokra.voxtral.text_decoder.vocab_size";
const KEY_TD_N_CTX: &str = "vokra.voxtral.text_decoder.n_ctx";
const KEY_TD_ROPE_BASE: &str = "vokra.voxtral.text_decoder.rope_base";
const KEY_TD_RMS_NORM_EPS: &str = "vokra.voxtral.text_decoder.rms_norm_eps";

// ----- Cross-attn / S2S / mode --------------------------------------------

const KEY_XATTN_HIDDEN_DIM: &str = "vokra.voxtral.cross_attn.hidden_dim";
const KEY_MODE: &str = "vokra.voxtral.mode";
const KEY_S2S_CODEC_TYPE: &str = "vokra.voxtral.s2s.codec_type";

/// Audio encoder hparams (Whisper-derived).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioEncoderConfig {
    /// Encoder transformer block count.
    pub n_layer: usize,
    /// Encoder attention heads.
    pub n_head: usize,
    /// Encoder hidden width `d_audio`.
    pub hidden_dim: usize,
    /// Encoder positional length (`n_audio_ctx`, 1500 for a Whisper-derived
    /// encoder).
    pub n_ctx: usize,
    /// Mel input channels (128 for Voxtral).
    pub n_mels: usize,
    /// Encoder feed-forward inner width.
    pub ffn_dim: usize,
}

impl AudioEncoderConfig {
    /// Per-head width. Whisper-derived: `hidden_dim / n_head`. Returns `0`
    /// when `n_head == 0` (the shape-only converter sentinel) so callers
    /// can pass it to downstream layer-shape checks without a panic.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim.checked_div(self.n_head).unwrap_or(0)
    }
}

/// Text decoder hparams (Mistral).
#[derive(Debug, Clone, PartialEq)]
pub struct TextDecoderConfig {
    /// Decoder block count.
    pub n_layer: usize,
    /// Query attention heads (`>= n_head_kv` — GQA).
    pub n_head_q: usize,
    /// Key/value attention heads (`n_head_q % n_head_kv == 0` — GQA).
    pub n_head_kv: usize,
    /// Decoder hidden width `d_text`.
    pub hidden_dim: usize,
    /// SwiGLU FFN inner width.
    pub ffn_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Max sequence length the decoder supports.
    pub n_ctx: usize,
    /// RoPE base θ (Mistral ships `1_000_000.0` on modern releases).
    pub rope_base: f32,
    /// RMSNorm epsilon (Mistral ships `1e-5`).
    pub rms_norm_eps: f32,
}

impl TextDecoderConfig {
    /// Per-head width (`hidden_dim / n_head_q`). Mistral variants ship a
    /// fixed `head_dim=128`, but the value is derived from the shape here
    /// so the loader validates the config self-consistently. Returns `0`
    /// when `n_head_q == 0` (the shape-only converter sentinel).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_dim.checked_div(self.n_head_q).unwrap_or(0)
    }
}

/// Voxtral top-level config.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralConfig {
    /// Audio encoder sub-config.
    pub audio: AudioEncoderConfig,
    /// Text decoder sub-config.
    pub text: TextDecoderConfig,
    /// Cross-attention hidden dim (usually equal to `audio.hidden_dim`).
    pub cross_attn_hidden_dim: usize,
    /// `"asr"` (default) or `"s2s"`.
    pub mode: String,
    /// S2S codec identifier (`"none"` = ASR-only build).
    pub s2s_codec_type: String,
}

impl VoxtralConfig {
    /// Reads and validates the config from a parsed GGUF file.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when a key is missing, has the wrong type
    /// or the head split does not divide `hidden_dim`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // Audio encoder.
        let ae_n_layer = req_u32(file, KEY_AE_N_LAYER)?;
        let ae_hidden_dim = req_u32(file, KEY_AE_HIDDEN_DIM)?;
        let ae_n_head = req_u32(file, KEY_AE_N_HEAD)?;
        let ae_n_ctx = opt_u32(file, KEY_AE_N_CTX);
        let ae_n_mels = req_u32(file, KEY_AE_N_MELS)?;
        let ae_ffn_dim = opt_u32(file, KEY_AE_FFN_DIM);

        if ae_n_head != 0 && ae_hidden_dim % ae_n_head != 0 {
            return Err(bad(format!(
                "audio_encoder.n_head ({ae_n_head}) must divide audio_encoder.hidden_dim ({ae_hidden_dim})"
            )));
        }

        // Text decoder.
        let td_n_layer = req_u32(file, KEY_TD_N_LAYER)?;
        let td_hidden_dim = req_u32(file, KEY_TD_HIDDEN_DIM)?;
        let td_ffn_dim = req_u32(file, KEY_TD_FFN_DIM)?;
        let td_vocab = opt_u32(file, KEY_TD_VOCAB_SIZE);
        let td_n_ctx = opt_u32(file, KEY_TD_N_CTX);
        let td_n_head_q = opt_u32(file, KEY_TD_N_HEAD_Q);
        let td_n_head_kv = opt_u32(file, KEY_TD_N_HEAD_KV);
        let td_rope_base = opt_f32(file, KEY_TD_ROPE_BASE);
        let td_rms_norm_eps = opt_f32(file, KEY_TD_RMS_NORM_EPS);

        if td_n_head_q != 0 && td_hidden_dim != 0 && td_hidden_dim % td_n_head_q != 0 {
            return Err(bad(format!(
                "text_decoder.n_head_q ({td_n_head_q}) must divide text_decoder.hidden_dim ({td_hidden_dim})"
            )));
        }
        if td_n_head_kv != 0 && td_n_head_q != 0 && td_n_head_q % td_n_head_kv != 0 {
            return Err(bad(format!(
                "text_decoder.n_head_kv ({td_n_head_kv}) must divide text_decoder.n_head_q ({td_n_head_q}) for GQA"
            )));
        }

        // Cross-attn / mode / codec.
        let xattn_hidden_dim = opt_u32(file, KEY_XATTN_HIDDEN_DIM);
        let mode = opt_string(file, KEY_MODE).unwrap_or_else(|| "asr".to_owned());
        let s2s_codec_type =
            opt_string(file, KEY_S2S_CODEC_TYPE).unwrap_or_else(|| "none".to_owned());

        Ok(Self {
            audio: AudioEncoderConfig {
                n_layer: ae_n_layer as usize,
                n_head: ae_n_head as usize,
                hidden_dim: ae_hidden_dim as usize,
                n_ctx: ae_n_ctx as usize,
                n_mels: ae_n_mels as usize,
                ffn_dim: ae_ffn_dim as usize,
            },
            text: TextDecoderConfig {
                n_layer: td_n_layer as usize,
                n_head_q: td_n_head_q as usize,
                n_head_kv: td_n_head_kv as usize,
                hidden_dim: td_hidden_dim as usize,
                ffn_dim: td_ffn_dim as usize,
                vocab_size: td_vocab as usize,
                n_ctx: td_n_ctx as usize,
                rope_base: td_rope_base,
                rms_norm_eps: td_rms_norm_eps,
            },
            cross_attn_hidden_dim: xattn_hidden_dim as usize,
            mode,
            s2s_codec_type,
        })
    }
}

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("voxtral config: {msg}"))
}

fn req_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad(format!("metadata key `{key}` is not a u32-range integer"))),
        None => Err(bad(format!("missing metadata key `{key}`"))),
    }
}

fn opt_u32(file: &GgufFile, key: &str) -> u32 {
    file.get(key)
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0)
}

fn opt_f32(file: &GgufFile, key: &str) -> f32 {
    file.get(key)
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.0)
}

fn opt_string(file: &GgufFile, key: &str) -> Option<String> {
    file.get(key).and_then(|v| v.as_str()).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn valid_builder() -> GgufBuilder {
        let mut b = GgufBuilder::new();
        // Audio encoder.
        b.add_u32(KEY_AE_N_LAYER, 24);
        b.add_u32(KEY_AE_HIDDEN_DIM, 1024);
        b.add_u32(KEY_AE_N_HEAD, 16);
        b.add_u32(KEY_AE_N_CTX, 1500);
        b.add_u32(KEY_AE_N_MELS, 128);
        b.add_u32(KEY_AE_FFN_DIM, 4096);
        // Text decoder.
        b.add_u32(KEY_TD_N_LAYER, 28);
        b.add_u32(KEY_TD_HIDDEN_DIM, 3072);
        b.add_u32(KEY_TD_FFN_DIM, 8192);
        b.add_u32(KEY_TD_VOCAB_SIZE, 32_000);
        b.add_u32(KEY_TD_N_CTX, 32_768);
        b.add_u32(KEY_TD_N_HEAD_Q, 24);
        b.add_u32(KEY_TD_N_HEAD_KV, 8);
        b.add_f32(KEY_TD_ROPE_BASE, 1_000_000.0);
        b.add_f32(KEY_TD_RMS_NORM_EPS, 1e-5);
        // Cross-attn / mode / codec.
        b.add_u32(KEY_XATTN_HIDDEN_DIM, 1024);
        b.add_string(KEY_MODE, "asr");
        b.add_string(KEY_S2S_CODEC_TYPE, "none");
        b
    }

    #[test]
    fn reads_voxtral_mini_hparams() {
        let file = GgufFile::parse(valid_builder().to_bytes().unwrap()).unwrap();
        let cfg = VoxtralConfig::from_gguf(&file).unwrap();
        assert_eq!(cfg.audio.n_layer, 24);
        assert_eq!(cfg.audio.hidden_dim, 1024);
        assert_eq!(cfg.audio.head_dim(), 64);
        assert_eq!(cfg.text.n_layer, 28);
        assert_eq!(cfg.text.n_head_q, 24);
        assert_eq!(cfg.text.n_head_kv, 8);
        assert_eq!(cfg.text.head_dim(), 128);
        assert!((cfg.text.rope_base - 1_000_000.0).abs() < 1.0);
        assert_eq!(cfg.mode, "asr");
    }

    #[test]
    fn missing_required_key_is_model_load_error() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_AE_HIDDEN_DIM, 1024); // AE_N_LAYER missing.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralConfig::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn gqa_head_split_must_divide_query_heads() {
        let mut b = valid_builder();
        // 7 does not divide 24 query heads → reject.
        b.add_u32(KEY_TD_N_HEAD_KV, 7);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralConfig::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn audio_encoder_head_split_must_divide_hidden_dim() {
        let mut b = valid_builder();
        b.add_u32(KEY_AE_N_HEAD, 7); // 1024 % 7 != 0.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralConfig::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
