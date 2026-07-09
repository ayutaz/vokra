//! Voxtral ASR head (M3-10-T13) — audio → text.
//!
//! # Scope (autoregressive greedy)
//!
//! The ASR head owns the logic that ties the audio encoder output to the
//! Mistral text decoder for a greedy transcribe:
//!
//! 1. run [`AudioEncoder::forward`](super::audio_encoder::forward) on the
//!    log-mel input, yielding `[n_ctx, d_audio]`;
//! 2. seed the decoder with the config-supplied special-token prefix
//!    (`bos_id`);
//! 3. drive the causal decode loop through a [`TextDecoderSession`]
//!    (self-attention + KV cache append + RoPE + causal mask + GQA);
//! 4. stop on the `<eos>` token or a max-new-token cap;
//! 5. return the generated token id sequence (the caller detokenises via
//!    [`VoxtralTokenizer`]).
//!
//! # Audio conditioning — honest scope
//!
//! Upstream Voxtral projects the audio-encoder hidden state through an
//! **audio adapter** (unloaded in the current GGUF converter output) and
//! consumes it as a soft-prefix sequence at the start of the text decoder.
//! The adapter weights are **not** in the shape-only converter path, so
//! this head runs the encoder (for shape / dispatch coverage) and then
//! greedy-decodes from `bos_id` with **no audio conditioning wired**. The
//! returned tokens therefore reflect the language-model prior — real ASR
//! quality requires the adapter, which is a downstream ticket
//! (T19+ / real-checkpoint parity). This entry point deliberately never
//! fabricates a "transcript that sounds like the audio" (FR-EX-08):
//! either the caller sees the LM-prior output honestly, or they see an
//! explicit error.
//!
//! `bos_id` / `eos_id` default to Mistral's shipped ids (`1` / `2`) — every
//! Voxtral variant on HuggingFace inherits these from the Mistral base
//! tokenizer.
//!
//! [`TextDecoderSession`]: super::TextDecoderSession
//! [`VoxtralTokenizer`]: super::VoxtralTokenizer

use vokra_core::{BackendKind, Result, VokraError};

use super::text_decoder_session::{DEFAULT_MAX_NEW_TOKENS, TextDecoderSession, greedy_decode};
use super::{AudioEncoder, AudioEncoderOutput, TextDecoder, VoxtralConfig};

/// The Mistral BOS token id (shipped `<s>` = 1 across every Mistral tokenizer
/// release Voxtral inherits from).
pub const MISTRAL_BOS_ID: u32 = 1;
/// The Mistral EOS token id (shipped `</s>` = 2).
pub const MISTRAL_EOS_ID: u32 = 2;

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

    /// Greedy transcribe: audio → token id sequence.
    ///
    /// Steps:
    /// 1. validates `config.mode` is `"asr"` or `"s2s"` (S2S also produces
    ///    text as its inner stream);
    /// 2. runs the audio encoder end-to-end (shape / dispatch coverage);
    /// 3. constructs a [`TextDecoderSession`] on `backend` and greedy-decodes
    ///    from `bos_id` until `eos_id` or `max_new_tokens`.
    ///
    /// See the module doc's "Audio conditioning" note — until the audio
    /// adapter follow-up ticket lands, the returned tokens are the LM
    /// prior's continuation of `bos_id`, not audio-conditioned ASR.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a mismatched `config.mode` or a
    ///   0-sentinel config;
    /// - [`VokraError::InvalidArgument`] on a shape mismatch in `log_mel` or
    ///   an out-of-range token id;
    /// - [`VokraError::UnsupportedOp`] / [`VokraError::BackendUnavailable`]
    ///   on a backend that does not cover the Voxtral hot ops.
    pub fn transcribe(
        &self,
        backend: BackendKind,
        log_mel: &[f32],
        n_frames: usize,
        bos_id: u32,
        eos_id: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        // Config-mode gate — surface the mismatch clearly.
        if self.config.mode != "asr" && self.config.mode != "s2s" {
            return Err(VokraError::ModelLoad(format!(
                "voxtral asr_head.transcribe: config.mode is `{}` — expected `asr` or `s2s`",
                self.config.mode
            )));
        }
        // Config bounds gate: bos/eos must be in vocab.
        let vocab = self.config.text.vocab_size as u32;
        if bos_id >= vocab {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral asr_head.transcribe: bos_id {bos_id} >= vocab_size {vocab}"
            )));
        }
        // eos_id being out-of-vocab is TOLERATED (matches Whisper's greedy —
        // an unreachable eos just means the decode runs to max_new_tokens).

        // Build the compute seam via a lightweight Compute for the encoder
        // (the decoder session builds its own with the same backend).
        let compute = crate::compute::Compute::for_backend(backend, super::VOXTRAL_HOT_OPS)?;
        let _encoder_out = self.encode(&compute, log_mel, n_frames)?;

        // Autoregressive greedy over the loaded Mistral text decoder.
        let mut session = TextDecoderSession::new(self.config, self.text, backend)?;
        let cap = if max_new_tokens == 0 {
            DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };
        greedy_decode(&mut session, &[bos_id], eos_id, cap)
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
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};

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
                n_ctx: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    fn tiny_encoder(cfg: &VoxtralConfig) -> AudioEncoder {
        AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        }
    }

    /// Build a tiny hand-crafted TextDecoder with deterministic non-zero
    /// weights (same seed pattern as text_decoder_session::tests::tiny_decoder).
    /// Reused across the AsrHead greedy-decode tests.
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let head_dim = d / cfg.text.n_head_q;
        let kv_hidden = cfg.text.n_head_kv * head_dim;

        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }
        let blocks = (0..cfg.text.n_layer)
            .map(|_| DecoderBlock {
                attn_norm_gamma: vec![1.0f32; d],
                attn: GqaAttention {
                    q: linear(d, d, 0.10),
                    k: linear(d, kv_hidden, -0.07),
                    v: linear(d, kv_hidden, 0.05),
                    o: linear(d, d, -0.04),
                },
                ffn_norm_gamma: vec![1.0f32; d],
                ffn: SwiGluFfn {
                    gate: linear(d, ffn, 0.06),
                    up: linear(d, ffn, -0.02),
                    down: linear(ffn, d, 0.03),
                },
            })
            .collect();
        TextDecoder {
            token_emb,
            blocks,
            final_norm_gamma: vec![1.0f32; d],
            prefix: "",
        }
    }

    #[test]
    fn transcribe_returns_a_non_empty_token_sequence() {
        // With deterministic non-zero weights + eos outside the vocab so the
        // greedy loop never terminates early, transcribe must return exactly
        // `max_new_tokens` tokens, each in-vocab. This is the "engine is
        // registered and runs" oracle (no fabricated string — the caller
        // detokenises separately).
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        // eos = vocab+100 => unreachable => loop runs to max_new.
        let out = head
            .transcribe(
                BackendKind::Cpu,
                &log_mel,
                n_frames,
                /*bos*/ 1,
                /*eos*/ cfg.text.vocab_size as u32 + 100,
                /*max_new*/ 3,
            )
            .unwrap();
        assert_eq!(out.len(), 3, "must respect max_new");
        assert!(
            out.iter().all(|&t| (t as usize) < cfg.text.vocab_size),
            "every token must be in-vocab: {out:?}"
        );
    }

    #[test]
    fn transcribe_stops_on_eos() {
        // With eos = 0 and the deterministic weights above, the greedy loop
        // may or may not emit 0 as the first token; either way the returned
        // sequence must not exceed max_new_tokens. When it stops early on
        // eos, the eos token IS included in the result.
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let out = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 0, 10)
            .unwrap();
        assert!(out.len() <= 10, "must respect max_new");
        if let Some(&last) = out.last() {
            if last == 0 {
                // stopped on eos: eos IS included, no tokens after it.
                assert!(out.iter().take(out.len() - 1).all(|&t| t != 0));
            }
        }
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
            head.transcribe(BackendKind::Cpu, &[], 0, 1, 2, 4),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_rejects_out_of_range_bos() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.0f32; cfg.audio.n_mels * n_frames];
        let bos = cfg.text.vocab_size as u32; // >= vocab
        assert!(matches!(
            head.transcribe(BackendKind::Cpu, &log_mel, n_frames, bos, 2, 4),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_is_deterministic_across_calls() {
        let cfg = tiny_config();
        let ae = tiny_encoder(&cfg);
        let td = tiny_decoder(&cfg);
        let head = AsrHead::new(&cfg, &ae, &td);
        let n_frames = 8;
        let log_mel = vec![0.5f32; cfg.audio.n_mels * n_frames];
        let a = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 9999, 4)
            .unwrap();
        let b = head
            .transcribe(BackendKind::Cpu, &log_mel, n_frames, 1, 9999, 4)
            .unwrap();
        assert_eq!(a, b, "greedy decode must be deterministic");
    }

    #[test]
    fn mistral_default_ids_match_upstream_shipping_values() {
        // Guard against accidental rename; the caller-side layer relies on
        // these being the Mistral defaults (1 = <s>, 2 = </s>).
        assert_eq!(MISTRAL_BOS_ID, 1);
        assert_eq!(MISTRAL_EOS_ID, 2);
    }
}
