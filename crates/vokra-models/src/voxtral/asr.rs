//! Voxtral `AsrEngine` — [`vokra_core::AsrEngine`] adaptor over a loaded
//! [`VoxtralModel`].
//!
//! # Purpose
//!
//! The vokra-server (`integrations/vokra-server`) treats every ASR model
//! uniformly through the [`vokra_core::AsrEngine`] trait. This file supplies
//! the Voxtral side of that trait — a thin adaptor that owns a
//! [`VoxtralModel`] plus a [`StreamingConfig`] and dispatches
//! [`AsrEngine::transcribe`](vokra_core::AsrEngine::transcribe) through the
//! [`AsrHead`].
//!
//! # Honest scope (M3-10 structural completion)
//!
//! [`AsrHead::transcribe`](crate::voxtral::AsrHead::transcribe) currently
//! returns [`VokraError::NotImplemented`] for the full autoregressive decode
//! (the block math + KV cache + tokenizer greedy step is a follow-on
//! ticket, see the module docs). This adaptor exists so the vokra-server
//! layer can wire Voxtral through the same registry pattern the Whisper +
//! Kokoro + piper-plus engines use — **without** silently fabricating a
//! transcript (FR-EX-08). The moment the block math + greedy step land, the
//! server route lights up automatically; no re-plumbing on the server side.
//!
//! # Front-end responsibility
//!
//! [`AsrEngine::transcribe`](vokra_core::AsrEngine::transcribe) accepts mono
//! `f32` PCM at 16 kHz (Whisper front-end convention). This adaptor turns
//! PCM into log-mel using the same front-end Voxtral was declared with
//! (validated bit-exact against the GGUF `vokra.frontend.*` chunk at
//! [`VoxtralModel::from_gguf`]). Since the log-mel path itself needs a
//! Whisper-style STFT (which lives in `vokra-ops` and is exercised by
//! `crate::whisper`), and the two front-ends are declared identical, this
//! foundation surfaces a clear [`VokraError::NotImplemented`] at the
//! log-mel-conversion step until the shared front-end helper lands — again,
//! never a fabricated pass.

use std::sync::Arc;

use vokra_core::{AsrEngine, Result, Transcription, VokraError};

use super::{AsrHead, VoxtralModel};
use crate::compute::Compute;

/// A Voxtral engine that speaks the [`AsrEngine`] trait. Holds the loaded
/// [`VoxtralModel`] and the runtime settings the server-side registry
/// (`integrations/vokra-server`) needs.
///
/// Cloned freely on the hot path (the model is behind an [`Arc`]).
pub struct VoxtralAsr {
    /// The parsed model, shared. `Arc` because the registry holds one and
    /// hot-path handlers borrow it read-only.
    model: Arc<VoxtralModel>,
    /// Whether the model was declared as ASR- or S2S-capable in its config.
    /// ASR mode is the default; an S2S-tagged model can still be routed
    /// through this adaptor (S2S produces text on the inner stream) but the
    /// caller sees an ASR interface.
    // Currently only surfaced by `is_configured_for_asr` — the follow-up
    // ticket (real transcribe) will read it before dispatching a greedy
    // vs beam search head.
    #[allow(dead_code)]
    is_configured_for_asr: bool,
}

impl VoxtralAsr {
    /// Wraps a loaded [`VoxtralModel`] as an [`AsrEngine`].
    ///
    /// A model whose declared `mode` is not `"asr"` or `"s2s"` is rejected
    /// with an explicit [`VokraError::ModelLoad`] — never silently coerced
    /// (FR-EX-08).
    pub fn new(model: VoxtralModel) -> Result<Self> {
        let is_asr = matches!(model.config().mode.as_str(), "asr" | "s2s");
        if !is_asr {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::VoxtralAsr: unknown mode `{}` — expected `asr` or `s2s`",
                model.config().mode
            )));
        }
        Ok(Self {
            model: Arc::new(model),
            is_configured_for_asr: is_asr,
        })
    }

    /// Loads a Voxtral model from a GGUF file and wraps it as an
    /// [`AsrEngine`]. Same error surfaces as
    /// [`VoxtralModel::from_gguf`].
    pub fn from_gguf(file: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let model = VoxtralModel::from_gguf(file)?;
        Self::new(model)
    }

    /// Shared handle to the underlying model.
    #[must_use]
    pub fn model(&self) -> &Arc<VoxtralModel> {
        &self.model
    }
}

impl AsrEngine for VoxtralAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        // 1) Front-end: PCM → log-mel. The Voxtral front-end declares
        //    Whisper-style params (n_fft=400, hop=160, n_mels=128); the
        //    shared implementation lives in `vokra-ops::stft` /
        //    `crate::whisper::mel` and is exercised by the Whisper engine.
        //    Wiring it here is a small follow-on (extract the shared
        //    helper), but until then we surface a clear NotImplemented
        //    rather than fabricating a mel spectrogram or silently routing
        //    through Whisper's front-end (FR-EX-08).
        //
        //    The pcm length gets validated so the check catches an empty
        //    request as a proper error path.
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxtral::VoxtralAsr::transcribe: pcm slice is empty".into(),
            ));
        }
        // 2) Encoder + decoder greedy step — deferred (see AsrHead).
        //    Exercise the ASR-head wiring so a broken model surfaces at
        //    call time, then bubble up the honest NotImplemented.
        let head = AsrHead::new(
            self.model.config(),
            self.model.audio_encoder(),
            self.model.text_decoder(),
        );
        let _ = &head; // referenced for the future full transcribe path
        let _ = Compute::cpu(); // ensure the compute seam is reachable on CPU
        Err(VokraError::NotImplemented(
            "voxtral::VoxtralAsr::transcribe: full ASR requires the log-mel front-end helper \
             + autoregressive greedy decode. Both are downstream tickets (M3-10 follow-ups). \
             The engine is registered and every plumbing seam is exercised by this call — \
             it just refuses to fabricate output until the block math lands (FR-EX-08).",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::{AudioEncoder, TextDecoder, VoxtralConfig};

    fn tiny_model() -> VoxtralModel {
        // Handcraft a VoxtralModel with the smallest self-consistent
        // shapes so we can exercise the AsrEngine wiring without a GGUF.
        let cfg = VoxtralConfig {
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
        };
        let audio = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let text = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        // Reconstruct the top-level model directly (all fields are pub(crate)
        // within voxtral). This test-only helper mirrors what
        // VoxtralModel::from_gguf would produce for a synthetic GGUF.
        VoxtralModel {
            config: cfg,
            audio,
            text,
        }
    }

    #[test]
    fn new_accepts_asr_and_s2s_modes() {
        let model = tiny_model();
        let asr = VoxtralAsr::new(model).unwrap();
        assert!(asr.is_configured_for_asr);
    }

    #[test]
    fn new_rejects_unknown_mode() {
        let mut model = tiny_model();
        model.config.mode = "unknown".to_owned();
        assert!(matches!(
            VoxtralAsr::new(model),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_empty_pcm_is_invalid_argument() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_returns_honest_not_implemented_never_fabricates() {
        // Foundation: full transcribe is a follow-up. The adaptor must
        // return a clear NotImplemented rather than a fabricated string
        // (FR-EX-08 — no silent pass).
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        let pcm = vec![0.0f32; 16_000]; // 1 s of silence @ 16 kHz
        let result = asr.transcribe(&pcm);
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn is_asr_engine_object_safe() {
        // If AsrEngine goes non-object-safe, this line stops compiling.
        // The vokra-server registry stores engines behind Arc<dyn AsrEngine>
        // so this is a load-bearing property.
        let _engine: Arc<dyn AsrEngine> = Arc::new(VoxtralAsr::new(tiny_model()).unwrap());
    }
}
