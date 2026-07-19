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
const KEY_TD_HEAD_DIM: &str = "vokra.voxtral.text_decoder.head_dim";
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
    /// Explicit per-head width. Mistral **decouples** this from
    /// `hidden_dim / n_head_q` — the shipping `Voxtral-Mini-3B-2507` has
    /// `hidden_dim = 3072` but 32 query heads × `head_dim = 128` (a
    /// 4096-wide Q projection). `0` means "derive as `hidden_dim /
    /// n_head_q`" — the pre-M4 behaviour, kept so GGUFs converted before
    /// the `vokra.voxtral.text_decoder.head_dim` key existed still load.
    /// Read through [`Self::head_dim`], never directly.
    pub head_dim: usize,
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
    /// Per-head width — the explicit `head_dim` metadata when present, else
    /// derived as `hidden_dim / n_head_q` (pre-M4 GGUF compatibility).
    /// Returns `0` on the shape-only converter sentinel path (`n_head_q ==
    /// 0` with no explicit value), which every forward entry point rejects.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_dim.checked_div(self.n_head_q).unwrap_or(0)
        }
    }

    /// Q projection output width: `n_head_q * head_dim()`. Equals
    /// `hidden_dim` only when the checkpoint ties `head_dim` to
    /// `hidden_dim / n_head_q` (NOT true of the shipping Voxtral mini).
    #[must_use]
    pub fn q_hidden(&self) -> usize {
        self.n_head_q * self.head_dim()
    }

    /// K/V projection output width: `n_head_kv * head_dim()` — also the
    /// per-layer KV cache row width.
    #[must_use]
    pub fn kv_hidden(&self) -> usize {
        self.n_head_kv * self.head_dim()
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
        let td_head_dim = opt_u32(file, KEY_TD_HEAD_DIM);
        let td_rope_base = opt_f32(file, KEY_TD_ROPE_BASE);
        let td_rms_norm_eps = opt_f32(file, KEY_TD_RMS_NORM_EPS);

        // The divisibility requirement only concerns the DERIVED head width
        // (pre-M4 GGUFs without the explicit head_dim key). With an explicit
        // head_dim, `hidden_dim / n_head_q` is never consulted — Mistral
        // decouples the two (mini-3b: 3072 hidden, 32 heads x 128).
        if td_head_dim == 0
            && td_n_head_q != 0
            && td_hidden_dim != 0
            && td_hidden_dim % td_n_head_q != 0
        {
            return Err(bad(format!(
                "text_decoder.n_head_q ({td_n_head_q}) must divide text_decoder.hidden_dim \
                 ({td_hidden_dim}) when no explicit text_decoder.head_dim is present — \
                 re-convert with a converter that writes {KEY_TD_HEAD_DIM}"
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
                head_dim: td_head_dim as usize,
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
    fn reads_legacy_hparams_without_head_dim_key() {
        // A pre-M4 GGUF (no `text_decoder.head_dim` key) derives head_dim as
        // `hidden_dim / n_head_q` — 3072 / 24 = 128 in this fixture.
        let file = GgufFile::parse(valid_builder().to_bytes().unwrap()).unwrap();
        let cfg = VoxtralConfig::from_gguf(&file).unwrap();
        assert_eq!(cfg.audio.n_layer, 24);
        assert_eq!(cfg.audio.hidden_dim, 1024);
        assert_eq!(cfg.audio.head_dim(), 64);
        assert_eq!(cfg.text.n_layer, 28);
        assert_eq!(cfg.text.n_head_q, 24);
        assert_eq!(cfg.text.n_head_kv, 8);
        assert_eq!(cfg.text.head_dim, 0, "no explicit key → 0 sentinel field");
        assert_eq!(cfg.text.head_dim(), 128, "derived hidden_dim / n_head_q");
        assert_eq!(cfg.text.q_hidden(), 3072);
        assert_eq!(cfg.text.kv_hidden(), 1024);
        assert!((cfg.text.rope_base - 1_000_000.0).abs() < 1.0);
        assert_eq!(cfg.mode, "asr");
    }

    #[test]
    fn reads_real_mini_hparams_with_explicit_head_dim() {
        // The shipping Voxtral-Mini-3B-2507 shape: hidden 3072, 32 query
        // heads x head_dim 128 (Q projection 4096-wide — head_dim decoupled
        // from hidden_dim / n_head_q, which would be 96), 8 KV heads, 30
        // layers (config.json, 2026-07-16 real-weight eval).
        let mut b = valid_builder();
        b.add_u32(KEY_TD_N_LAYER, 30);
        b.add_u32(KEY_TD_N_HEAD_Q, 32);
        b.add_u32(KEY_TD_N_HEAD_KV, 8);
        b.add_u32(KEY_TD_HEAD_DIM, 128);
        b.add_u32(KEY_TD_VOCAB_SIZE, 131_072);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let cfg = VoxtralConfig::from_gguf(&file).unwrap();
        assert_eq!(cfg.text.n_layer, 30);
        assert_eq!(cfg.text.n_head_q, 32);
        assert_eq!(cfg.text.n_head_kv, 8);
        assert_eq!(cfg.text.head_dim(), 128, "explicit key wins");
        assert_eq!(cfg.text.q_hidden(), 4096, "32 x 128 != hidden_dim 3072");
        assert_eq!(cfg.text.kv_hidden(), 1024, "8 x 128");
    }

    #[test]
    fn explicit_head_dim_lifts_the_divisibility_requirement() {
        // hidden_dim 3070 is NOT divisible by 32 query heads — legal when an
        // explicit head_dim decouples the two, rejected when it must be
        // derived.
        let mut b = valid_builder();
        b.add_u32(KEY_TD_HIDDEN_DIM, 3070);
        b.add_u32(KEY_TD_N_HEAD_Q, 32);
        b.add_u32(KEY_TD_HEAD_DIM, 128);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(VoxtralConfig::from_gguf(&file).is_ok());

        let mut b = valid_builder();
        b.add_u32(KEY_TD_HIDDEN_DIM, 3070);
        b.add_u32(KEY_TD_N_HEAD_Q, 32);
        // No head_dim key → the derived path must reject the non-divisible
        // split rather than silently flooring 3070 / 32.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralConfig::from_gguf(&file),
            Err(VokraError::ModelLoad(_))
        ));
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
