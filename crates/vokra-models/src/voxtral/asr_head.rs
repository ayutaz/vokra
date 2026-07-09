//! Voxtral ASR head (M3-10-T13) — audio → text.
//!
//! # Foundation scope
//!
//! The ASR head owns the *logic* that ties the audio encoder output to the
//! Mistral text decoder for a greedy transcribe:
//!
//! 1. run [`AudioEncoder::forward`](super::audio_encoder::forward) on the
//!    log-mel input, yielding `[n_ctx, d_audio]`;
//! 2. seed the decoder with the special-token prefix (mode-specific — the
//!    upstream Voxtral card ships an ASR prefix on the `s2s` token stream);
//! 3. drive the causal decode loop, sourcing keys/values for cross-attention
//!    from the encoder output;
//! 4. stop on the `<eos>` token or a max-new-token cap;
//! 5. detokenize using the embedded Mistral tokenizer bytes.
//!
//! The full decode loop is a downstream ticket; **this file lands the
//! header** (types, entry-point signatures, and a smoke test that verifies
//! the wiring compiles + accepts the expected shapes on synthetic input).
//! No fabricated pass — the smoke test does not assert a specific token
//! sequence.

use vokra_core::{Result, VokraError};

use super::{AudioEncoder, AudioEncoderOutput, TextDecoder, VoxtralConfig};

/// A greedy ASR head. Holds references — it is a thin adaptor, not a
/// stateful engine.
///
/// Construct once (cheap) and call [`AsrHead::transcribe`] per utterance;
/// the caller owns the log-mel front-end + tokenization.
pub struct AsrHead<'m> {
    config: &'m VoxtralConfig,
    audio: &'m AudioEncoder,
    text: &'m TextDecoder,
}

impl<'m> AsrHead<'m> {
    /// Bundle a config + encoder + decoder into a runnable ASR head.
    #[must_use]
    pub fn new(config: &'m VoxtralConfig, audio: &'m AudioEncoder, text: &'m TextDecoder) -> Self {
        Self {
            config,
            audio,
            text,
        }
    }

    /// Runs the audio encoder on `log_mel` (`[n_mels, n_frames]`) and
    /// returns the encoder hidden state — the shape the future full decode
    /// will consume.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on a zero-sentinel config;
    /// [`VokraError::InvalidArgument`] on shape mismatch.
    pub fn encode(
        &self,
        compute: &crate::compute::Compute,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<AudioEncoderOutput> {
        super::audio_encoder::forward(compute, self.config, self.audio, log_mel, n_frames)
    }

    /// Greedy transcribe — foundation stub.
    ///
    /// # Foundation contract
    ///
    /// This entry point:
    /// - validates the config's `mode` is `"asr"` (or `"s2s"` — S2S also
    ///   produces text as its inner stream);
    /// - runs the encoder;
    /// - returns [`VokraError::NotImplemented`] with a message pointing to
    ///   the follow-up tickets (T13+) rather than fabricating output.
    ///
    /// This is deliberate: honest about scope, per FR-EX-08, no silent stub
    /// pass.
    pub fn transcribe(
        &self,
        compute: &crate::compute::Compute,
        log_mel: &[f32],
        n_frames: usize,
    ) -> Result<Vec<u32>> {
        // Config-mode gate — surface the mismatch clearly.
        if self.config.mode != "asr" && self.config.mode != "s2s" {
            return Err(VokraError::ModelLoad(format!(
                "voxtral asr_head.transcribe: config.mode is `{}` — expected `asr` or `s2s`",
                self.config.mode
            )));
        }
        // Exercise the encoder — its errors surface to the caller.
        let _enc = self.encode(compute, log_mel, n_frames)?;
        // Deferral: full autoregressive decode + tokenizer decode is
        // downstream. Return a clear NotImplemented rather than a
        // fabricated pass (FR-EX-08).
        Err(VokraError::NotImplemented(
            "voxtral::asr_head::transcribe: full autoregressive decode is a follow-up ticket \
             (M3-10-T13+). The encoder wiring is verified by this call; the tokenizer + KV \
             cache + cross-attention full-forward land alongside a real Voxtral checkpoint \
             parity dump (T19+).",
        ))
    }

    /// Returns the number of text-decoder blocks — surfaces a piece of the
    /// loaded model to the caller for e2e wiring checks.
    #[must_use]
    pub fn n_decoder_layer(&self) -> usize {
        self.text.n_layer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};

    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 8,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 8,
                n_ctx: 8,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    #[test]
    fn transcribe_returns_not_implemented_for_now() {
        // Foundation posture: the entry point is wired but returns a clear
        // NotImplemented pointing at the follow-up (FR-EX-08 — no fabricated
        // pass). The encoder wiring must still compile / not panic.
        let cfg = tiny_config();
        let ae = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let td = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![1.0f32; cfg.audio.n_mels * n_frames];
        assert!(matches!(
            head.transcribe(&Compute::cpu(), &log_mel, n_frames),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn transcribe_rejects_unknown_mode() {
        let mut cfg = tiny_config();
        cfg.mode = "wat".to_owned();
        let ae = AudioEncoder {
            conv1_w: Vec::new(),
            conv1_b: Vec::new(),
            conv2_w: Vec::new(),
            conv2_b: Vec::new(),
            pos_emb: Vec::new(),
            has_learned_pos_emb: false,
        };
        let td = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        let head = AsrHead::new(&cfg, &ae, &td);
        assert!(matches!(
            head.transcribe(&Compute::cpu(), &[], 0),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
