//! StyleTTS 2 — Li et al., "StyleTTS 2: Towards Human-Level TTS through
//! Style Diffusion and Adversarial Training with Large Speech Language
//! Models" (NeurIPS 2023, arXiv:2306.07691). Config-only scaffold for the
//! architecture; **the released pretrained weights are voice-consent
//! gated and therefore fail-closed** (`LicenseClass::Unknown` — never
//! bound at load).
//!
//! # What StyleTTS 2 is (primary source)
//!
//! Upstream `github.com/yl4579/StyleTTS2` (MIT code) implements a
//! diffusion-based TTS with the following top-level pipeline (from
//! `models.py` + `Modules/*.py` + the paper §3):
//!
//! - **Text encoder** — a `TextEncoder` stack (5 residual 1D-conv
//!   blocks + LayerNorm + LSTM) mapping phoneme ids to
//!   `[T_text, d_hid]`. `d_hid` is `hidden_dim=512` in the released
//!   `Models/LibriTTS/config.yml` and `Models/LJSpeech/config.yml`.
//! - **Style predictor + Style diffusion sampler** — a `StyleDiffusion`
//!   module (K-diffusion 2nd-order Heun sampler over a
//!   `UNet1DConditionModel`) samples a **style vector**
//!   `s ∈ R^style_dim` from a reference / prompt embedding.
//!   `style_dim=128` in the released configs.
//! - **Prosody / duration** — a `DurationPredictor` conditions on the
//!   text hidden + style via `AdaIN` + LSTM, producing per-phoneme
//!   duration and F0 / N (energy). Upsampling to frame rate happens
//!   through the standard monotonic aligner (`length_regulator`).
//! - **Acoustic decoder** — an `iSTFTNet` head (StyleTTS 2 uses the
//!   Kaneko et al. 2022 iSTFTNet decoder, arXiv:2203.02395 — NOT
//!   plain HiFi-GAN). The final `synthesize` returns 24 kHz PCM
//!   (LJSpeech, `sample_rate=24000` in `config.yml`).
//!
//! # Why StyleTTS 2 weights are **not** distributed by Vokra
//!
//! The upstream README (`github.com/yl4579/StyleTTS2/blob/main/README.md`)
//! explicitly conditions weight use on **voice consent + disclosure**:
//!
//! > "Before using these pre-trained models, you agree to inform the
//! > listeners that the speech samples are synthesized by the pre-trained
//! > models, unless you have the permission to use the voice you
//! > synthesize."
//! >
//! > "only use voices whose speakers grant the permission to have their
//! > voice cloned..."
//!
//! This is a **usage agreement**, not a standard SPDX permissive license.
//! The registry (`vokra-core::LicenseClass::from_id`) resolves
//! `styletts2` / `styletts-2` to [`LicenseClass::Unknown`], which fails
//! closed under M2-13 — the runtime refuses to load StyleTTS 2 GGUFs
//! outside `--i-understand-risks --research-only` mode
//! (`docs/license-audit.md` §3.1 sign-off for StyleTTS 2 remains
//! **empty**, and CC does not pre-fill it — 2026-07-23 audit outcome:
//! `☑ Rejected 2026-07-23 yousan`, weight redistribution declined).
//!
//! # What this module actually provides
//!
//! - The [`StyleTts2Config`] surface — every hparam transcribed verbatim
//!   from `Models/LibriTTS/config.yml` + `Models/LJSpeech/config.yml`
//!   (fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」). Callers
//!   can construct the config and reason about the architecture without
//!   ever holding weights.
//! - A [`StyleTts2Tts`] engine handle whose [`StyleTts2Tts::synthesize`]
//!   returns [`VokraError::NotImplemented`] naming the blocker
//!   ("StyleTTS 2 weights are voice-consent gated; see
//!   docs/license-audit.md §3.1"). This is the standard skeleton contract
//!   (zonos / vits-ja / omniASR-CTC — the "config-first,
//!   weight-follow-up" pattern), specialized for the case where the
//!   follow-up wave is a **licensing decision**, not an implementation
//!   push.
//!
//! # Architecture independence
//!
//! Copyright protects **expression**, not architecture. This module
//! re-implements the architecture from primary sources (upstream code
//! comments + the NeurIPS paper) and does not ship any weight file; the
//! architectural axes here are safe to depend on. A user who trained
//! their own StyleTTS 2 on a corpus with a permissive weight license
//! could override `--license <spdx>` at conversion time (the vits-ja
//! escape hatch), but the default posture is fail-closed.

use std::path::Path;

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// StyleTTS 2 architecture hparams (LJSpeech / LibriTTS release defaults).
///
/// Every field is transcribed **verbatim** from the primary sources
/// `github.com/yl4579/StyleTTS2/tree/main/Models/LJSpeech/config.yml`
/// (single-speaker) and `Models/LibriTTS/config.yml` (multi-speaker,
/// diffusion sampler enabled) — fetched 2026-07-30 (CLAUDE.md
/// 「ハルシネーション厳禁」).
///
/// The tokenizer / phoneme table is out of scope for the config surface —
/// upstream uses `phonemizer` (which routes to eSpeak-NG, **GPL-3.0** —
/// forbidden by Vokra) so a Vokra deployment would source phonemes
/// through `integrations/vokra-piper-g2p` (whisper.cpp 型 clean-room
/// substitution, CLAUDE.md 設計判断 4).
#[derive(Debug, Clone, PartialEq)]
pub struct StyleTts2Config {
    /// Output PCM sample rate. **24 000 Hz** on both LJSpeech and
    /// LibriTTS (`config.yml`: `sample_rate: 24000`).
    pub sample_rate_hz: u32,
    /// Style vector dimension. **128** on both released variants
    /// (`config.yml`: `style_dim: 128`).
    pub style_dim: usize,
    /// Residual / hidden width fed through the text encoder + prosody
    /// stack. **512** on both released variants (`config.yml`:
    /// `hidden_dim: 512`).
    pub hidden_dim: usize,
    /// Number of Mel bins the iSTFTNet decoder consumes. **80** on both
    /// released variants (`config.yml`: `n_mels: 80`).
    pub n_mels: usize,
    /// Text encoder — number of residual 1D-conv blocks.
    /// `model_params.text_encoder.n_layer` — **3** on the released
    /// configs (LJSpeech / LibriTTS).
    pub text_encoder_n_layer: usize,
    /// Style predictor / duration LSTM hidden width. Upstream shares
    /// `hidden_dim` with the residual width; kept as its own field so a
    /// downstream variant can decouple.
    pub predictor_hidden_dim: usize,
    /// Style diffusion sampler — number of Heun steps. `diffusion.steps`
    /// on the LibriTTS config; **5** (2nd-order Heun ≈ 10 UNet1D
    /// forwards). The LJSpeech single-speaker variant disables the
    /// diffusion sampler (`diffusion.embedding_mask_proba: 0` and the
    /// reference embedding is deterministic).
    pub diffusion_steps: usize,
    /// Whether the checkpoint carries a trained style diffusion sampler.
    /// `false` for the single-speaker LJSpeech release, `true` for
    /// LibriTTS.
    pub uses_style_diffusion: bool,
    /// iSTFTNet decoder — number of channels at the entry. **512** on
    /// both released variants (`config.yml`:
    /// `model_params.decoder.dim_in: 512`).
    pub decoder_dim_in: usize,
    /// iSTFTNet decoder — resblock kernel sizes (`resblock_kernel_sizes:
    /// [3, 7, 11]` in `config.yml`).
    pub decoder_resblock_kernels: [usize; 3],
    /// iSTFTNet decoder — upsample rates (`upsample_rates: [10, 6]` in
    /// `config.yml`).
    pub decoder_upsample_rates: [usize; 2],
    /// iSTFTNet decoder — upsample kernel sizes (`upsample_kernel_sizes:
    /// [20, 12]` in `config.yml`).
    pub decoder_upsample_kernels: [usize; 2],
    /// iSTFTNet decoder — number of post-net iSTFT frequency bins
    /// (`gen_istft_n_fft: 20` in `config.yml`).
    pub decoder_gen_istft_n_fft: usize,
    /// iSTFTNet decoder — post-net iSTFT hop length
    /// (`gen_istft_hop_size: 5` in `config.yml`).
    pub decoder_gen_istft_hop_size: usize,
}

impl StyleTts2Config {
    /// The LJSpeech single-speaker release defaults
    /// (`Models/LJSpeech/config.yml`).
    #[must_use]
    pub fn ljspeech() -> Self {
        Self {
            sample_rate_hz: 24_000,
            style_dim: 128,
            hidden_dim: 512,
            n_mels: 80,
            text_encoder_n_layer: 3,
            predictor_hidden_dim: 512,
            diffusion_steps: 0,
            uses_style_diffusion: false,
            decoder_dim_in: 512,
            decoder_resblock_kernels: [3, 7, 11],
            decoder_upsample_rates: [10, 6],
            decoder_upsample_kernels: [20, 12],
            decoder_gen_istft_n_fft: 20,
            decoder_gen_istft_hop_size: 5,
        }
    }

    /// The LibriTTS multi-speaker release defaults
    /// (`Models/LibriTTS/config.yml`) — diffusion sampler active.
    #[must_use]
    pub fn libritts() -> Self {
        Self {
            uses_style_diffusion: true,
            diffusion_steps: 5,
            ..Self::ljspeech()
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// (mirrors the omniASR-CTC / VoxCPM / VibeVoice validate contract).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any required axis is `0` or
    /// obviously inconsistent (e.g. `hidden_dim == 0`).
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.sample_rate_hz == 0 {
            return Err(VokraError::InvalidArgument(
                "styletts2: sample_rate_hz must be > 0".to_owned(),
            ));
        }
        for (name, v) in [
            ("style_dim", self.style_dim),
            ("hidden_dim", self.hidden_dim),
            ("n_mels", self.n_mels),
            ("text_encoder_n_layer", self.text_encoder_n_layer),
            ("predictor_hidden_dim", self.predictor_hidden_dim),
            ("decoder_dim_in", self.decoder_dim_in),
            ("decoder_gen_istft_n_fft", self.decoder_gen_istft_n_fft),
            (
                "decoder_gen_istft_hop_size",
                self.decoder_gen_istft_hop_size,
            ),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "styletts2: {name} must be > 0 (got 0-placeholder)"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine handle
// ---------------------------------------------------------------------------

/// StyleTTS 2 engine handle.
///
/// Construction succeeds against a validated [`StyleTts2Config`]; every
/// [`Self::synthesize`] call returns [`VokraError::NotImplemented`]
/// naming the weight-license blocker (FR-EX-08 — never silent-produce a
/// hallucinated waveform from an unbound checkpoint).
#[derive(Debug, Clone)]
pub struct StyleTts2Tts {
    cfg: StyleTts2Config,
}

impl StyleTts2Tts {
    /// Assembles an engine from `cfg`. Cross-checks the config axes so a
    /// mismatched pair fails loudly here rather than deep inside a
    /// forward.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    pub fn new(cfg: StyleTts2Config) -> Result<Self> {
        cfg.validate_for_forward()?;
        Ok(Self { cfg })
    }

    /// Returns the resolved config.
    pub fn config(&self) -> &StyleTts2Config {
        &self.cfg
    }

    /// Loads a StyleTTS 2 GGUF from disk — **always fail-closed** on the
    /// weight side.
    ///
    /// The registered upstream weight license is a **voice-consent /
    /// disclosure usage agreement**, not a standard SPDX permissive
    /// license (`docs/license-audit.md` §3 StyleTTS 2 row +
    /// `LicenseClass::Unknown` in the registry). A user who legitimately
    /// holds a StyleTTS 2 checkpoint under a distinct SPDX id (e.g. a
    /// re-training on a permissive corpus) would build via the
    /// `--license <spdx>` override at conversion time and hit a **future
    /// wave** that wires this method to a real `StyleTts2Weights::from_gguf`;
    /// for now the entry point is deliberately absent to avoid silent
    /// production use.
    ///
    /// # Errors
    ///
    /// Always returns [`VokraError::NotImplemented`] naming
    /// `docs/license-audit.md` §3.1 — the sign-off queue that owner has
    /// **not** filled for StyleTTS 2 (fail-closed).
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let _ = path; // deliberately unused — see below
        Err(VokraError::NotImplemented(
            "styletts2: from_gguf is intentionally not wired. The upstream \
             yl4579/StyleTTS2 weights ride a voice-consent / disclosure usage \
             agreement (README §Pre-trained Model) — NOT a standard SPDX \
             permissive license — so vokra-core::LicenseClass::from_id resolves \
             `styletts2` / `styletts-2` to LicenseClass::Unknown (fail-closed, \
             M2-13). Vokra does NOT distribute StyleTTS 2 checkpoints; \
             docs/license-audit.md §3.1 StyleTTS 2 sign-off is currently \
             `☑ Rejected 2026-07-23 yousan` (weight redistribution declined). \
             The architecture surface (StyleTts2Config, StyleTts2Tts::new) is \
             here so downstream research callers who hold their own weight (a \
             re-training on a permissive corpus) can override the license at \
             conversion time and bind through a follow-up wave — but the \
             default posture is unimplemented, not silently-loaded.",
        ))
    }

    /// Synthesizes 24 kHz PCM from the input phoneme sequence.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `phoneme_ids` is empty.
    /// - [`VokraError::NotImplemented`] otherwise — the runtime path
    ///   is unimplemented deliberately because the weights are voice-
    ///   consent gated (see [`Self::from_gguf`]).
    pub fn synthesize(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "styletts2 synthesize: phoneme_ids is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "styletts2 synthesize: the runtime forward is intentionally not \
             wired. StyleTTS 2 weights are voice-consent gated \
             (LicenseClass::Unknown → fail-closed, M2-13); \
             docs/license-audit.md §3.1 StyleTTS 2 sign-off is \
             `☑ Rejected 2026-07-23 yousan`. See StyleTts2Tts::from_gguf for \
             the licensing rationale and the escape hatch (a downstream \
             re-training on a permissive corpus overrides at `--license \
             <spdx>` conversion time, bound by a future wave).",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ljspeech_defaults_match_upstream_config_yaml() {
        // Every axis transcribed from
        // github.com/yl4579/StyleTTS2/blob/main/Models/LJSpeech/config.yml
        // (fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」).
        let c = StyleTts2Config::ljspeech();
        assert_eq!(c.sample_rate_hz, 24_000);
        assert_eq!(c.style_dim, 128);
        assert_eq!(c.hidden_dim, 512);
        assert_eq!(c.n_mels, 80);
        assert_eq!(c.text_encoder_n_layer, 3);
        assert_eq!(c.predictor_hidden_dim, 512);
        assert_eq!(c.decoder_dim_in, 512);
        assert_eq!(c.decoder_resblock_kernels, [3, 7, 11]);
        assert_eq!(c.decoder_upsample_rates, [10, 6]);
        assert_eq!(c.decoder_upsample_kernels, [20, 12]);
        assert_eq!(c.decoder_gen_istft_n_fft, 20);
        assert_eq!(c.decoder_gen_istft_hop_size, 5);
        // Single-speaker → no style diffusion sampler.
        assert!(!c.uses_style_diffusion);
        assert_eq!(c.diffusion_steps, 0);
    }

    #[test]
    fn libritts_variant_enables_style_diffusion() {
        let c = StyleTts2Config::libritts();
        // Shares the architecture with LJSpeech but activates the
        // diffusion sampler.
        assert!(c.uses_style_diffusion);
        assert_eq!(c.diffusion_steps, 5);
        // Every other axis is identical to LJSpeech's.
        assert_eq!(c.hidden_dim, StyleTts2Config::ljspeech().hidden_dim);
        assert_eq!(c.style_dim, StyleTts2Config::ljspeech().style_dim);
    }

    #[test]
    fn validate_rejects_zero_placeholder() {
        let mut c = StyleTts2Config::ljspeech();
        c.hidden_dim = 0;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_rejects_ill_formed_config() {
        let mut c = StyleTts2Config::ljspeech();
        c.style_dim = 0;
        assert!(matches!(
            StyleTts2Tts::new(c),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn from_gguf_is_fail_closed_with_licence_reason() {
        // The path may or may not exist — the point is that the entry
        // returns NotImplemented naming the license blocker regardless.
        let missing = Path::new("/nonexistent/styletts2-does-not-exist.gguf");
        let err = StyleTts2Tts::from_gguf(missing).expect_err("must be NotImplemented");
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("voice-consent"),
                    "message must name the licensing blocker: {msg}",
                );
                assert!(
                    msg.contains("docs/license-audit.md"),
                    "message must point at the sign-off doc: {msg}",
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn synthesize_rejects_empty_input() {
        let tts = StyleTts2Tts::new(StyleTts2Config::ljspeech()).unwrap();
        let err = tts
            .synthesize(&[])
            .expect_err("empty phoneme sequence must fail loudly");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn synthesize_is_fail_closed_with_licence_reason() {
        let tts = StyleTts2Tts::new(StyleTts2Config::ljspeech()).unwrap();
        let err = tts
            .synthesize(&[1, 2, 3])
            .expect_err("must be NotImplemented");
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("voice-consent gated"),
                    "message must name the licensing blocker: {msg}",
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn config_returns_bound_axes() {
        let tts = StyleTts2Tts::new(StyleTts2Config::ljspeech()).unwrap();
        assert_eq!(tts.config().sample_rate_hz, 24_000);
        assert_eq!(tts.config().style_dim, 128);
    }
}
