//! `AsrEngine` implementation wiring Whisper into `session.asr().transcribe()`.
//!
//! [`WhisperAsr`] owns a loaded [`WhisperModel`] and an optional
//! [`WhisperTokenizer`]. [`AsrEngine::transcribe`] runs PCM → log-mel → encoder
//! → greedy decode → detokenize. Beam search (a model-independent host
//! function, [`vokra_core::decode`]) is offered through
//! [`WhisperAsr::transcribe_tokens_beam`]; the trait method uses greedy
//! (equivalent to `beam_width = 1`).
//!
//! # Hand-off to M0-09 (C ABI)
//!
//! The Rust surface M0-09 wraps is: [`WhisperAsr::from_gguf`] /
//! [`WhisperAsr::with_tokenizer`] to build, and the
//! [`AsrEngine::transcribe`](vokra_core::AsrEngine::transcribe) trait method
//! (plus [`WhisperAsr::transcribe_tokens`] for a tokenizer-free id stream).
//! C-ABI export (`vokra_asr_transcribe`, FR-API-01) is out of this WP's scope.

use std::sync::Arc;

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::GgufFile;
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, Result, VokraError};

use super::WhisperModel;
use super::greedy::{DEFAULT_MAX_NEW_TOKENS, greedy_decode};
use super::tokenizer::WhisperTokenizer;
use crate::compute::Compute;
use crate::whisper::WHISPER_HOT_OPS;
use crate::whisper::beam_glue::{WhisperBeamScorer, WhisperLogitsSource};
use vokra_core::decode::{BeamSearchConfig, SamplerConfig, beam_search, sample_sequence};

/// Whisper ASR engine: a loaded model plus an optional detokenizer.
///
/// The model is held behind an [`Arc`] so a per-utterance [`DecoderState`] (and
/// the beam scorer) can own it without a lifetime; cloning the handle is cheap.
///
/// [`DecoderState`]: super::decoder::DecoderState
pub struct WhisperAsr {
    model: Arc<WhisperModel>,
    tokenizer: Option<WhisperTokenizer>,
    /// Backend selector (`Copy`; the engine never holds a live `!Send`
    /// backend, so it stays `Send + Sync`). A [`Compute`] is built from it at
    /// each transcribe entry (M2-01 Phase 3).
    backend_kind: BackendKind,
}

impl WhisperAsr {
    /// Loads the model from `file`. The tokenizer is attached separately with
    /// [`with_tokenizer`](Self::with_tokenizer) (the M0 converter does not embed
    /// it — see [`WhisperTokenizer`]); an attempt is still made to read an
    /// embedded `vokra.tokenizer.model` blob for forward compatibility. The
    /// backend defaults to [`BackendKind::Cpu`].
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model = Arc::new(WhisperModel::from_gguf(file)?);
        let tokenizer = WhisperTokenizer::from_gguf(file, model.config().eot).ok();
        Ok(Self {
            model,
            tokenizer,
            backend_kind: BackendKind::Cpu,
        })
    }

    /// Attaches a detokenizer (from the parity fixture / sidecar in M0).
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: WhisperTokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Selects the backend the transcription forward runs on (default
    /// [`BackendKind::Cpu`]).
    ///
    /// Whisper needs softmax / layer-norm / GELU / conv1d / GEMV on the backend
    /// as well as GEMM, which the Metal slice does not yet cover, so
    /// `with_backend(BackendKind::Metal)` makes the transcribe entries an
    /// explicit [`VokraError::UnsupportedOp`] until those Metal kernels land
    /// (Phase 4) — never a silent CPU fall back (FR-EX-08).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend_kind = backend;
        self
    }

    /// The loaded model (encoder / decoder forwards, config).
    pub fn model(&self) -> &WhisperModel {
        &self.model
    }

    /// Transcribes `pcm` to the raw generated token id sequence (greedy),
    /// without detokenizing. Useful when no tokenizer is attached and for
    /// token-level parity tests.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        // Build the backend dispatcher once at the entry (errors here for a
        // backend that does not cover the Whisper op set — e.g. Metal, Phase 4),
        // and run the encoder on it; the decoder steps rebuild it from the same
        // backend selection. The CPU path is bit-identical to before the seam.
        let compute = Compute::for_backend(self.backend_kind, WHISPER_HOT_OPS)?;
        let encoder = self.model.encode_pcm_with(&compute, pcm)?;
        let mut state = self
            .model
            .decoder_with_backend(&encoder, self.backend_kind)?;
        let cfg = self.model.config();
        greedy_decode(
            &mut state,
            &cfg.decoder_start_ids,
            cfg.eot,
            DEFAULT_MAX_NEW_TOKENS,
        )
    }

    /// Transcribes `pcm` with beam search, returning the best token id
    /// sequence. `beam_width = 1` matches [`transcribe_tokens`](Self::transcribe_tokens).
    pub fn transcribe_tokens_beam(
        &self,
        pcm: &[f32],
        config: &BeamSearchConfig,
    ) -> Result<Vec<u32>> {
        // Metal is rejected here (Whisper op set uncovered until Phase 4); the
        // scorer's per-step decoder runs on the CPU backend as before.
        let compute = Compute::for_backend(self.backend_kind, WHISPER_HOT_OPS)?;
        let encoder = self.model.encode_pcm_with(&compute, pcm)?;
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&self.model), &encoder)?;
        let cfg = self.model.config();
        let hyps = beam_search(&mut scorer, &cfg.decoder_start_ids, cfg.eot, config)?;
        hyps.into_iter().next().map(|h| h.tokens).ok_or_else(|| {
            VokraError::ModelLoad("whisper beam search produced no hypothesis".into())
        })
    }

    /// Transcribes `pcm` with stochastic sampling (temperature / top-k / top-p /
    /// repetition penalty over the model-independent
    /// [`Sampler`](vokra_core::decode::Sampler)), returning the generated token
    /// id sequence. A `temperature == 0` config is greedy and reproduces
    /// [`transcribe_tokens`](Self::transcribe_tokens) token-for-token.
    pub fn transcribe_tokens_sampled(
        &self,
        pcm: &[f32],
        config: &SamplerConfig,
    ) -> Result<Vec<u32>> {
        let compute = Compute::for_backend(self.backend_kind, WHISPER_HOT_OPS)?;
        let encoder = self.model.encode_pcm_with(&compute, pcm)?;
        let mut source = WhisperLogitsSource::new(Arc::clone(&self.model), &encoder)?;
        let cfg = self.model.config();
        sample_sequence(
            &mut source,
            &cfg.decoder_start_ids,
            cfg.eot,
            config,
            DEFAULT_MAX_NEW_TOKENS,
        )
    }

    /// Detokenizes `ids`, or renders them as a bracketed id list when no
    /// tokenizer is attached (keeps the demo useful without a vocabulary).
    fn render(&self, ids: &[u32]) -> Result<String> {
        match &self.tokenizer {
            Some(t) => t.decode(ids),
            None => Ok(format!(
                "[no tokenizer; token ids: {}]",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            )),
        }
    }
}

impl AsrEngine for WhisperAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let ids = self.transcribe_tokens(pcm)?;
        Ok(Transcription::new(self.render(&ids)?))
    }
}
