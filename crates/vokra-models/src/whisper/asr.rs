//! `AsrEngine` implementation wiring Whisper into `session.asr().transcribe()`.
//!
//! [`WhisperAsr`] owns a loaded [`WhisperModel`] and an optional
//! [`WhisperTokenizer`]. [`AsrEngine::transcribe`] runs PCM → log-mel → encoder
//! → greedy decode → detokenize. Beam search (a model-independent host
//! function, [`vokra_core::decode`]) is offered through
//! [`WhisperAsr::transcribe_tokens_beam`] (top-1) and
//! [`WhisperAsr::transcribe_tokens_beam_nbest`] (full n-best); the trait
//! method uses greedy (equivalent to `beam_width = 1`).
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
use crate::whisper::beam_glue::{self, WhisperBeamScorer, WhisperLogitsSource};
use vokra_core::decode::{
    BeamHypothesis, BeamSearchConfig, SamplerConfig, beam_search, sample_sequence,
};

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
        Self::from_gguf_with(file, super::WhisperLoadOptions::default())
    }

    /// [`from_gguf`](Self::from_gguf) with the M5-15 fused-quant load options.
    ///
    /// With [`WhisperLoadOptions::fused_quant_weights`](super::WhisperLoadOptions::fused_quant_weights)
    /// the K-quantized projections run the fused INT8 kernels instead of being
    /// dequantized at load. **CPU-only and not bit-identical to the dequant
    /// route** — this is the entry point the M5-15-T10 WER comparison drives,
    /// which is why it exists on the ASR surface and not only on
    /// [`WhisperModel`].
    ///
    /// # Errors
    ///
    /// As [`from_gguf`](Self::from_gguf).
    pub fn from_gguf_with(file: &GgufFile, opts: super::WhisperLoadOptions) -> Result<Self> {
        let model = Arc::new(WhisperModel::from_gguf_with(file, opts)?);
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
    ///
    /// This is the top-1 collapse of
    /// [`transcribe_tokens_beam_nbest`](Self::transcribe_tokens_beam_nbest);
    /// callers that want the full ranked n-best list (vokra-server's
    /// beam surface, M3-15) should use the n-best variant directly.
    pub fn transcribe_tokens_beam(
        &self,
        pcm: &[f32],
        config: &BeamSearchConfig,
    ) -> Result<Vec<u32>> {
        let hyps = self.transcribe_tokens_beam_nbest(pcm, config)?;
        hyps.into_iter().next().map(|h| h.tokens).ok_or_else(|| {
            VokraError::ModelLoad("whisper beam search produced no hypothesis".into())
        })
    }

    /// Transcribes `pcm` with beam search, returning **the full n-best list**
    /// of hypotheses (up to [`BeamSearchConfig::n_best`]), ranked descending
    /// by `normalized_score`.
    ///
    /// Contract:
    /// * `config.beam_width == 1` and `config.n_best == 1` reproduce
    ///   [`transcribe_tokens`](Self::transcribe_tokens) as the head element.
    /// * `config.length_normalization` maps to HF `length_penalty` (α in the
    ///   `score / len^α` ranking).
    /// * Hypotheses are unique by token sequence (the model-independent
    ///   [`beam_search`] dedupes).
    ///
    /// FR-EX-08: hypotheses are returned raw (untruncated to top-1). The
    /// caller decides whether to collapse to the top-1 (legacy
    /// [`transcribe_tokens_beam`](Self::transcribe_tokens_beam) shape) or
    /// surface the ranked alternatives (vokra-server n-best endpoint,
    /// M3-15).
    ///
    /// # Errors
    ///
    /// * [`VokraError::UnsupportedOp`] when the selected backend does not
    ///   cover the Whisper op set (e.g. Metal today) — the same guard the
    ///   greedy [`transcribe_tokens`](Self::transcribe_tokens) enforces
    ///   (no silent CPU fall back, FR-EX-08).
    /// * Any error surfaced by
    ///   [`beam_search`](vokra_core::decode::beam_search) itself (empty
    ///   prefix, zero widths, `word_timestamps` enabled, …).
    pub fn transcribe_tokens_beam_nbest(
        &self,
        pcm: &[f32],
        config: &BeamSearchConfig,
    ) -> Result<Vec<BeamHypothesis>> {
        // Metal is rejected here (Whisper op set uncovered until Phase 4); the
        // scorer's per-step decoder runs on the CPU backend as before.
        let compute = Compute::for_backend(self.backend_kind, WHISPER_HOT_OPS)?;
        let encoder = self.model.encode_pcm_with(&compute, pcm)?;
        let cfg = self.model.config();
        // The clip's TRUE (unpadded) audio-position count for the word-
        // timestamp alignment (openai timing.py:208 `weights[:, :, :
        // num_frames // 2]`); the mel front-end zero-pads to the 30 s window
        // and the alignment must not span the padding (campaign-2 P2).
        let n_valid_audio = beam_glue::valid_audio_positions(pcm.len());
        // Attach the detokenizer when present so `word_timestamps` alignment
        // merges subword timings into per-word timings (M4-20); without a
        // tokenizer the alignment stays per-token. The two scorer types differ
        // only in the borrowed-tokenizer lifetime, so drive `beam_search` in
        // each arm rather than unifying to a trait object.
        match &self.tokenizer {
            Some(tok) => {
                let mut scorer = WhisperBeamScorer::with_tokenizer(
                    Arc::clone(&self.model),
                    &encoder,
                    tok,
                    n_valid_audio,
                )?;
                beam_search(&mut scorer, &cfg.decoder_start_ids, cfg.eot, config)
            }
            None => {
                let mut scorer =
                    WhisperBeamScorer::new(Arc::clone(&self.model), &encoder, n_valid_audio)?;
                beam_search(&mut scorer, &cfg.decoder_start_ids, cfg.eot, config)
            }
        }
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

    /// Whether an embedded / sidecar detokenizer is attached.
    ///
    /// Callers that need to surface an explicit error when a beam-search
    /// path would otherwise fall back to the bracketed id string (vokra-
    /// server's n-best endpoint prefers a hard `UnsupportedOp` over
    /// fabricated `"[no tokenizer; …]"` text) can gate on this.
    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Detokenizes `ids`, or renders them as a bracketed id list when no
    /// tokenizer is attached (keeps the demo useful without a vocabulary).
    ///
    /// Public so vokra-server's beam surface (M3-15) can detokenize each
    /// hypothesis returned by
    /// [`transcribe_tokens_beam_nbest`](Self::transcribe_tokens_beam_nbest)
    /// without re-implementing the fallback shape.
    pub fn render_ids(&self, ids: &[u32]) -> Result<String> {
        match &self.tokenizer {
            Some(t) => t.decode(ids),
            None => Ok(format!(
                "[no tokenizer; token ids: {}]",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            )),
        }
    }

    /// Test-only constructor from an already-loaded [`WhisperModel`], so the
    /// unit tests below can build a `WhisperAsr` on top of the tiny synthetic
    /// fixture without a real GGUF blob.
    ///
    /// Not part of the public API (compiled only under `cfg(test)`). Do not
    /// use for production code paths — [`from_gguf`](Self::from_gguf) is the
    /// only supported constructor at runtime, so the compliance /
    /// research-flag / provenance gates always fire (FR-CP-03).
    #[cfg(test)]
    pub(crate) fn from_model_for_test(model: Arc<WhisperModel>) -> Self {
        Self {
            model,
            tokenizer: None,
            backend_kind: BackendKind::Cpu,
        }
    }
}

impl AsrEngine for WhisperAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let ids = self.transcribe_tokens(pcm)?;
        Ok(Transcription::new(self.render_ids(&ids)?))
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the beam-search surface (M3-15 Wave 11 — Whisper beam
    //! wiring into `vokra-server`).
    //!
    //! Uses the `whisper::decoder::test_support::{tiny_model, tiny_encoder}`
    //! synthetic fixture so the tests run without a real GGUF. The full
    //! PCM→encoder path is exercised elsewhere; here we drive the underlying
    //! primitive `beam_search(&mut WhisperBeamScorer, ...)` that
    //! [`WhisperAsr::transcribe_tokens_beam_nbest`] composes internally, so
    //! the n-best return shape is guarded with a deterministic scorer (no
    //! PCM required, no encoder work needed).

    use super::*;
    use crate::whisper::beam_glue::WhisperBeamScorer;
    use crate::whisper::decoder::test_support::{tiny_encoder, tiny_model};
    use std::sync::Arc;
    use vokra_core::decode::{BeamHypothesis, BeamSearchConfig, beam_search};

    /// The n-best contract: `transcribe_tokens_beam_nbest` returns a ranked
    /// list of `BeamHypothesis` (up to `config.n_best`), sorted descending
    /// by `normalized_score`, with unique token sequences.
    ///
    /// Composed from the same primitive
    /// (`WhisperBeamScorer` + `beam_search`) that
    /// `transcribe_tokens_beam_nbest` uses internally after the encoder
    /// forward, so this test also guards the wrapper's return-shape contract
    /// (the wrapper does no post-processing after `beam_search`; see the
    /// implementation above).
    #[test]
    fn transcribe_tokens_beam_nbest_returns_multiple_hypotheses() {
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;

        // beam_width = 3, n_best = 3, small max_new so the tiny vocab
        // (n_vocab = 3, eot = 0) exhausts before running out of budget.
        let mut cfg = BeamSearchConfig::new(3, 6);
        cfg.n_best = 3;

        let hyps: Vec<BeamHypothesis> = beam_search(&mut scorer, &start, eot, &cfg).unwrap();
        assert!(!hyps.is_empty(), "beam search must return at least one hyp");
        assert!(hyps.len() <= 3, "n_best cap");

        // Ranked descending by normalized_score (the value the n-best sort
        // uses).
        for w in hyps.windows(2) {
            assert!(
                w[0].normalized_score >= w[1].normalized_score,
                "n-best must be ranked descending: {} vs {}",
                w[0].normalized_score,
                w[1].normalized_score,
            );
        }

        // Unique token sequences (beam_search dedupes).
        for i in 0..hyps.len() {
            for j in (i + 1)..hyps.len() {
                assert_ne!(
                    hyps[i].tokens, hyps[j].tokens,
                    "beam_search must dedupe by tokens",
                );
            }
        }
    }

    /// `transcribe_tokens_beam` (top-1 wrapper) equals the head of
    /// `transcribe_tokens_beam_nbest` — the wrapper only collapses to
    /// the top-1, it does not re-run the search or reorder.
    #[test]
    fn transcribe_tokens_beam_top1_matches_nbest_head() {
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer1 = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let mut scorer2 = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;

        let mut cfg = BeamSearchConfig::new(3, 6);
        cfg.n_best = 3;

        let nbest = beam_search(&mut scorer1, &start, eot, &cfg).unwrap();
        // Simulate the top-1 collapse `transcribe_tokens_beam` does — one
        // more `beam_search` on a fresh scorer with the same config, then
        // `.into_iter().next()`.
        let top1_via_second_call = beam_search(&mut scorer2, &start, eot, &cfg)
            .unwrap()
            .into_iter()
            .next()
            .map(|h| h.tokens)
            .unwrap();
        assert_eq!(top1_via_second_call, nbest.first().unwrap().tokens);
    }

    /// [`WhisperAsr::has_tokenizer`] reports `false` when built from the
    /// tiny model fixture (no tokenizer attached), and `render_ids` falls
    /// back to the `[no tokenizer; …]` string form. The n-best endpoint in
    /// vokra-server gates on `has_tokenizer` to surface an explicit
    /// `UnsupportedOp` rather than serve the bracketed fallback (FR-EX-08).
    #[test]
    fn has_tokenizer_and_render_ids_fallback_are_wired() {
        let model = tiny_model(0);
        let asr = WhisperAsr::from_model_for_test(model);
        assert!(!asr.has_tokenizer());
        let rendered = asr.render_ids(&[1, 2]).unwrap();
        assert!(
            rendered.contains("no tokenizer"),
            "fallback must name the missing tokenizer; got {rendered}",
        );
        assert!(rendered.contains('1') && rendered.contains('2'));
    }
}
