//! Voxtral S2S head (M3-10-T14) — audio → text tokens → codec tokens → audio.
//!
//! # Foundation scope
//!
//! The S2S head reuses the audio encoder + text decoder from the ASR path
//! and adds the codec token → waveform stage on the output. It also
//! enforces the AudioSeal watermark posture the ticket calls out
//! (T17): S2S output is **audio**, so the default is watermark ON;
//! ASR output is text and the flag is irrelevant.
//!
//! # No codec bundled
//!
//! The specific codec the S2S mode uses is recorded in
//! `vokra.voxtral.s2s.codec_type` on the GGUF (default `"none"` for
//! ASR-only builds). Wiring the real Mimi decoder is a follow-up
//! (M3-06 mimi_rvq is a separate WP), so this file lands the header +
//! the mode / watermark gate; the actual codec decode is a
//! [`VokraError::NotImplemented`] until Mimi lands.

use vokra_core::{Result, VokraError};

use super::{AudioEncoder, TextDecoder, VoxtralConfig};

/// Watermark configuration for a Voxtral S2S output stream. The Voxtral
/// ASR mode produces text and never sets this; the S2S mode produces
/// audio and defaults to watermark ON per T17 + `docs/legal-compliance.md`
/// §1.4 (EU AI Act Article 50 deployer disclosure).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S2sWatermarkConfig {
    /// Whether to embed the AudioSeal watermark on the S2S output stream.
    pub enabled: bool,
}

impl S2sWatermarkConfig {
    /// Default posture from a loaded [`VoxtralConfig`]: `enabled = true`
    /// for `mode == "s2s"`, otherwise `enabled = false`.
    ///
    /// Real deployers must still surface a disclosure — this is only the
    /// technical default. See `docs/legal-compliance.md` §1.4.
    #[must_use]
    pub fn from_config(cfg: &VoxtralConfig) -> Self {
        Self {
            enabled: cfg.mode == "s2s",
        }
    }
}

/// A synthesis head that turns encoded audio into a codec-token stream
/// and, ultimately, a waveform. Foundation-only.
pub struct S2sHead<'m> {
    config: &'m VoxtralConfig,
    audio: &'m AudioEncoder,
    text: &'m TextDecoder,
    watermark: S2sWatermarkConfig,
}

impl<'m> S2sHead<'m> {
    /// Bundle a config + encoder + decoder into a runnable S2S head. The
    /// watermark default is derived from the config (`mode == "s2s"` →
    /// ON).
    #[must_use]
    pub fn new(config: &'m VoxtralConfig, audio: &'m AudioEncoder, text: &'m TextDecoder) -> Self {
        let watermark = S2sWatermarkConfig::from_config(config);
        Self {
            config,
            audio,
            text,
            watermark,
        }
    }

    /// Overrides the default watermark configuration. Deployers who need
    /// to disable the watermark for internal testing pipelines pass this
    /// with `enabled = false`; production deploys must still comply with
    /// EU AI Act Article 50.
    #[must_use]
    pub fn with_watermark(mut self, wm: S2sWatermarkConfig) -> Self {
        self.watermark = wm;
        self
    }

    /// The active watermark configuration.
    #[must_use]
    pub fn watermark(&self) -> S2sWatermarkConfig {
        self.watermark
    }

    /// S2S synthesize — foundation stub.
    ///
    /// # Foundation contract
    ///
    /// - Validates the config's `mode == "s2s"`;
    /// - validates the codec identifier is one we recognise (currently just
    ///   `"mimi"` — the M3-06 mimi_rvq WP handles the real decode) or
    ///   `"none"` (rejected here — an S2S mode with `codec_type = none` is
    ///   a mis-configured GGUF, FR-EX-08);
    /// - returns [`VokraError::NotImplemented`] pointing at the follow-up
    ///   ticket. No fabricated waveform output.
    pub fn synthesize(&self, _log_mel: &[f32], _n_frames: usize) -> Result<Vec<f32>> {
        if self.config.mode != "s2s" {
            return Err(VokraError::ModelLoad(format!(
                "voxtral s2s_head.synthesize: config.mode is `{}` — expected `s2s`",
                self.config.mode
            )));
        }
        if self.config.s2s_codec_type == "none" {
            return Err(VokraError::ModelLoad(
                "voxtral s2s_head.synthesize: config.s2s_codec_type is `none` but mode is \
                 `s2s` — the GGUF is mis-configured. Re-convert with a `s2s_codec_type` \
                 (e.g. `mimi`) in the VoxtralConfig side-car (FR-EX-08)."
                    .to_owned(),
            ));
        }
        if self.config.s2s_codec_type != "mimi" {
            return Err(VokraError::UnsupportedOp(format!(
                "voxtral s2s_head.synthesize: codec `{}` is not implemented yet — only \
                 `mimi` is expected to land as M3-06 mimi_rvq; other codecs (EnCodec, DAC) \
                 are M4+ (docs/license-audit.md keeps EnCodec CC-BY-NC out of the zoo).",
                self.config.s2s_codec_type
            )));
        }
        // Reference the loaded structures so future full implementations can
        // reach in without changing the entry-point signature.
        let _ = (&self.audio, &self.text);
        Err(VokraError::NotImplemented(
            "voxtral::s2s_head::synthesize: Mimi codec decode + AudioSeal watermark embed are \
             follow-up tickets (M3-10-T14 + M3-06 mimi_rvq + AudioSeal integration). This \
             entry point is the header only — do NOT interpret a return as silence.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};

    fn s2s_config(codec: &str) -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 4,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                head_dim: 0,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 4,
                n_ctx: 4,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "s2s".to_owned(),
            s2s_codec_type: codec.to_owned(),
        }
    }

    fn empty_encoder() -> AudioEncoder {
        AudioEncoder {
            conv1_w: Vec::new(),
            conv1_b: Vec::new(),
            conv2_w: Vec::new(),
            conv2_b: Vec::new(),
            pos_emb: Vec::new(),
            has_learned_pos_emb: false,
            layers: Vec::new(),
            ln_post: crate::voxtral::test_support::identity_ln(0),
        }
    }

    fn empty_decoder() -> TextDecoder {
        TextDecoder {
            token_emb: Vec::new(),
            lm_head: None,
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
            mapped: None,
        }
    }

    #[test]
    fn watermark_default_is_on_for_s2s_mode() {
        let cfg = s2s_config("mimi");
        let wm = S2sWatermarkConfig::from_config(&cfg);
        assert!(wm.enabled, "S2S mode default MUST be watermark ON");
    }

    #[test]
    fn watermark_default_is_off_for_asr_mode() {
        let mut cfg = s2s_config("none");
        cfg.mode = "asr".to_owned();
        let wm = S2sWatermarkConfig::from_config(&cfg);
        assert!(!wm.enabled, "ASR mode is text output — no waveform to mark");
    }

    #[test]
    fn synthesize_rejects_none_codec_in_s2s_mode() {
        let cfg = s2s_config("none");
        let audio = empty_encoder();
        let text = empty_decoder();
        let head = S2sHead::new(&cfg, &audio, &text);
        assert!(matches!(
            head.synthesize(&[], 0),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn synthesize_rejects_non_s2s_mode() {
        let mut cfg = s2s_config("mimi");
        cfg.mode = "asr".to_owned();
        let audio = empty_encoder();
        let text = empty_decoder();
        let head = S2sHead::new(&cfg, &audio, &text);
        assert!(matches!(
            head.synthesize(&[], 0),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn synthesize_mimi_returns_not_implemented() {
        let cfg = s2s_config("mimi");
        let audio = empty_encoder();
        let text = empty_decoder();
        let head = S2sHead::new(&cfg, &audio, &text);
        // Honest posture: implementation deferred, but this must be an
        // explicit NotImplemented — never a fabricated silence output.
        assert!(matches!(
            head.synthesize(&[], 0),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn synthesize_unknown_codec_is_unsupported_op() {
        let cfg = s2s_config("encodec");
        let audio = empty_encoder();
        let text = empty_decoder();
        let head = S2sHead::new(&cfg, &audio, &text);
        assert!(matches!(
            head.synthesize(&[], 0),
            Err(VokraError::UnsupportedOp(_))
        ));
    }
}
