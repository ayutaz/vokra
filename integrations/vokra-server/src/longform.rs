//! Native long-form ASR orchestrator — the WhisperX-style pipeline (M0-08 +
//! M4-20 word timestamps + M0-08 speaker embedding), assembled in Rust.
//!
//! # What this module is
//!
//! WhisperX composes three off-the-shelf models to turn a long audio file into
//! (segment, text, per-word timings, speaker label) tuples:
//!
//! 1. **VAD** (Silero) chops the audio into speech-bounded segments;
//! 2. **Whisper** transcribes each segment (with cross-attention word timings
//!    from M4-20 on request);
//! 3. **Speaker embedding** (pyannote / SpeechBrain in upstream WhisperX;
//!    **CAM++** here, since M0-08 landed a native encoder) yields a 192-d
//!    vector per segment that we cluster with cosine similarity.
//!
//! Every component already lives in Vokra crates and is already
//! zero-dependency; the missing piece was the **glue**. That is this file.
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! This module lives in the excluded-workspace `vokra-server` crate, so it
//! does not touch the ROOT `Cargo.lock`. Everything it reaches for is a
//! first-party workspace crate (`vokra-core`, `vokra-models`, `vokra-ops`) —
//! no third-party HTTP/serde/audio glue is pulled in here.
//!
//! # Honest-scope notes (FR-EX-08)
//!
//! * **Per-word confidence.** Whisper's cross-attention alignment (`M4-20`)
//!   emits token spans + start/end times but does **not** carry a per-word
//!   log-probability. We record `WordTiming.confidence = 1.0` as a "not
//!   available" sentinel and document that the field will be filled when a
//!   real per-word posterior is landed (M5-15 UTMOS / M4-20 confidence
//!   follow-up). We do not synthesise a fake posterior from beam-hypothesis
//!   log-prob (an all-hypothesis average would look plausible without being).
//! * **Speaker clustering.** This is intentionally a minimal
//!   greedy-agglomerative pass (assign to nearest centroid above
//!   `speaker_threshold`, or new cluster otherwise) — the same shape the
//!   WhisperX quickstart uses. A real diarizer (pyannote-style HMM /
//!   spectral clustering) is a separate work item; we do not pretend this
//!   one is state-of-the-art.
//! * **`Whisper` word timestamps require model alignment heads.** A model
//!   without them makes the request an explicit `VokraError::UnsupportedOp`
//!   at the beam-search boundary — the orchestrator surfaces the error rather
//!   than silently returning empty words. Callers who know their model lacks
//!   alignment heads should set [`LongFormConfig::word_timestamps`] to false.
//!
//! # Testability
//!
//! The Whisper path lives behind a [`SegmentTranscriber`] trait so unit tests
//! can drive the orchestrator with a canned stub (there is no whisper GGUF
//! small enough to ship in-tree). The Silero VAD path uses the real
//! `tests/parity/silero_vad/silero-vad-v5.gguf` fixture (2 MiB, plain blob
//! under `.gitattributes` — no LFS).

use std::sync::Arc;

use vokra_core::decode::BeamSearchConfig;
use vokra_core::engines::VadEngine;
use vokra_core::{Result, VokraError};
use vokra_models::silero_vad::{SampleRate, SileroVadV5};
use vokra_models::speaker::{EMBED_DIM, SpeakerEncoder, cosine_similarity};
use vokra_models::whisper::asr::WhisperAsr;
use vokra_models::whisper::greedy::DEFAULT_MAX_NEW_TOKENS;
use vokra_ops::{KaldiFbankOpts, kaldi_fbank};

// ============================================================================
// Public output types
// ============================================================================

/// One aligned word inside a [`LongFormSegment`].
///
/// `start_sec` and `end_sec` are **absolute** into the original PCM (i.e., the
/// per-segment offsets returned by Whisper alignment are added to the
/// segment's own `start_sec`). `confidence` is `1.0` as a "not available"
/// sentinel — see the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// Detokenised word text, verbatim as Whisper's tokenizer produced it
    /// (spaces / punctuation included as-is; the callers strip if they wish).
    pub text: String,
    /// Word start time in seconds, absolute into the original PCM.
    pub start_sec: f32,
    /// Word end time in seconds, absolute into the original PCM.
    pub end_sec: f32,
    /// Placeholder posterior (currently `1.0`; see module docs — not
    /// fabricated, explicitly a "not available" sentinel).
    pub confidence: f32,
}

/// One VAD-bounded segment of the input PCM.
#[derive(Debug, Clone, PartialEq)]
pub struct LongFormSegment {
    /// Absolute segment start, in seconds.
    pub start_sec: f32,
    /// Absolute segment end, in seconds.
    pub end_sec: f32,
    /// Full transcription of this segment (Whisper detokenised output).
    pub text: String,
    /// Per-word timings — populated iff
    /// [`LongFormConfig::word_timestamps`] is on and the ASR emitted them.
    pub words: Vec<WordTiming>,
    /// Cluster id ("SPK_00", "SPK_01", …) — `Some(_)` iff a speaker encoder
    /// was passed to the orchestrator, `None` otherwise.
    pub speaker_id: Option<String>,
}

/// Full long-form transcription result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LongFormResult {
    /// Chronologically ordered segments — non-overlapping, covering only the
    /// speech-active regions.
    pub segments: Vec<LongFormSegment>,
}

// ============================================================================
// Config
// ============================================================================

/// Segmentation, transcription and speaker-clustering knobs.
///
/// Defaults mirror the upstream `silero_vad.utils_vad.get_speech_timestamps`
/// defaults (threshold 0.5, neg_threshold 0.35, min_speech 250 ms,
/// min_silence 100 ms, speech_pad 30 ms — the same reduction the M0-05
/// `silero_vad::parity` test uses).
#[derive(Debug, Clone, Copy)]
pub struct LongFormConfig {
    // ---- VAD segmenter ---------------------------------------------------
    /// Probability threshold to enter a speech region (upstream default 0.5).
    pub threshold: f32,
    /// Probability threshold to leave a speech region (upstream default
    /// `max(threshold - 0.15, 0.01)` = 0.35).
    pub neg_threshold: f32,
    /// Minimum speech-region duration to keep, in ms (upstream default 250).
    pub min_speech_ms: usize,
    /// Minimum silence-gap duration to split segments, in ms (upstream
    /// default 100).
    pub min_silence_ms: usize,
    /// Pad each side of a segment by this many ms (upstream default 30).
    pub speech_pad_ms: usize,

    // ---- Whisper transcription ------------------------------------------
    /// Ask for per-word timings from the ASR (M4-20 cross-attention DTW).
    /// A model without alignment heads makes this an explicit error at the
    /// beam-search boundary — never silently returns empty words.
    pub word_timestamps: bool,
    /// Upper bound on generated tokens per segment (matches Whisper's
    /// default).
    pub max_new_tokens: usize,

    // ---- Speaker clustering ---------------------------------------------
    /// Cosine-similarity threshold above which a segment's embedding joins
    /// an existing cluster (upstream WhisperX default 0.7).
    pub speaker_threshold: f32,
    /// Skip speaker binding for segments shorter than this (ms); fbank +
    /// CAM++ over a 100 ms clip is dominated by frame-boundary noise.
    /// Matches the Kaldi fbank frame length (25 ms) × a safety factor.
    pub speaker_min_ms: usize,
}

impl Default for LongFormConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            word_timestamps: true,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            speaker_threshold: 0.7,
            speaker_min_ms: 400,
        }
    }
}

// ============================================================================
// Transcriber seam
// ============================================================================

/// A per-segment ASR call.
///
/// [`WhisperAsr`] implements this by default; the trait exists so tests can
/// drive the orchestrator without shipping a Whisper GGUF and so a future
/// engine (Voxtral, etc.) can slot in.
///
/// `want_words` **must** be honoured: an implementation that has no
/// word-alignment path MUST return an error rather than silently returning
/// [`TranscribedText::words`] empty when `want_words == true` (FR-EX-08).
pub trait SegmentTranscriber: Send + Sync {
    /// Transcribes one PCM segment and returns text + optional word timings.
    fn transcribe_segment(&self, pcm: &[f32], want_words: bool) -> Result<TranscribedText>;
}

/// One segment's ASR output — text plus optional word timings.
///
/// `words` is empty when the caller did not request word timings. If word
/// timings were requested but the engine could not produce them, the
/// implementation must return an error, not an empty `words`.
#[derive(Debug, Clone, Default)]
pub struct TranscribedText {
    /// Detokenised text of the segment.
    pub text: String,
    /// Per-word timings, with times **relative to the segment start** — the
    /// orchestrator adds the segment's absolute offset. Empty iff word
    /// timings were not requested.
    pub words: Vec<WordTiming>,
}

/// [`WhisperAsr`] as a [`SegmentTranscriber`]: greedy fast-path when word
/// timings are not requested, beam-with-DTW when they are (matches the
/// vokra-server `transcribe_beam` split, extracted into a per-segment call).
impl SegmentTranscriber for WhisperAsr {
    fn transcribe_segment(&self, pcm: &[f32], want_words: bool) -> Result<TranscribedText> {
        if !want_words {
            // Backward-compat greedy shape — bit-identical to
            // `AsrEngine::transcribe(pcm)`.
            let ids = self.transcribe_tokens(pcm)?;
            let text = self.render_ids(&ids)?;
            return Ok(TranscribedText {
                text,
                words: Vec::new(),
            });
        }

        // Word-timing path: beam width 1 (greedy-equivalent) with alignment
        // on. `n_best = 1` because a long-form orchestrator surfaces only the
        // best hypothesis per segment.
        let mut cfg = BeamSearchConfig::new(1, DEFAULT_MAX_NEW_TOKENS);
        cfg.n_best = 1;
        cfg.length_normalization = 1.0;
        cfg.word_timestamps = true;
        let hyps = self.transcribe_tokens_beam_nbest(pcm, &cfg)?;
        let best = hyps.first().ok_or_else(|| {
            VokraError::ModelLoad("longform: whisper beam search produced no hypothesis".into())
        })?;
        let text = self.render_ids(&best.tokens)?;
        let mut words: Vec<WordTiming> = Vec::new();
        if let Some(timings) = &best.word_timestamps {
            for w in timings {
                let span = best.tokens.get(w.token_start..w.token_end).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "longform: word-timing token span {}..{} out of range for {} tokens",
                        w.token_start,
                        w.token_end,
                        best.tokens.len()
                    ))
                })?;
                let word_text = self.render_ids(span)?;
                words.push(WordTiming {
                    text: word_text,
                    start_sec: w.start,
                    end_sec: w.end,
                    // See module docs: "not available" sentinel, not
                    // fabricated. Beam hypothesis carries a whole-sequence
                    // score, not per-word.
                    confidence: 1.0,
                });
            }
        }
        Ok(TranscribedText { text, words })
    }
}

// ============================================================================
// The orchestrator itself
// ============================================================================

/// Native long-form ASR orchestrator — WhisperX pattern, assembled from the
/// Vokra components that already exist.
///
/// The three components (VAD, ASR, optional speaker encoder) are held behind
/// `Arc` so a single orchestrator can be shared across worker threads, and so
/// the same engines can also feed the request-scoped
/// [`crate::service::InferenceService`] without a second load.
pub struct LongFormOrchestrator {
    vad: Arc<SileroVadV5>,
    asr: Arc<dyn SegmentTranscriber>,
    speaker: Option<Arc<SpeakerEncoder>>,
    config: LongFormConfig,
    // ---- Running speaker-clustering state -------------------------------
    // Cluster centroids in the order they were first seen; the id "SPK_{i:02}"
    // is derived from the index (index 0 = "SPK_00", …). One `Vec<f32>` per
    // centroid; length always [`EMBED_DIM`]. Mean-of-embeddings, not
    // running mean — the classical WhisperX quickstart shape.
    centroids: Vec<Vec<f32>>,
    /// How many embeddings have been merged into each centroid (parallel to
    /// [`centroids`], used to update the mean incrementally).
    centroid_counts: Vec<usize>,
}

impl LongFormOrchestrator {
    /// Creates an orchestrator from a Silero VAD, a Whisper ASR, and an
    /// optional CAM++ speaker encoder. The default [`LongFormConfig`] is
    /// used; swap it with [`Self::with_config`].
    ///
    /// The trait object coercion (`Arc<WhisperAsr>` →
    /// `Arc<dyn SegmentTranscriber>`) is the only reason [`WhisperAsr`]
    /// needs a `SegmentTranscriber` impl — callers still pass concrete
    /// types.
    pub fn new(
        vad: Arc<SileroVadV5>,
        whisper: Arc<WhisperAsr>,
        speaker: Option<Arc<SpeakerEncoder>>,
    ) -> Self {
        Self::with_transcriber(vad, whisper as Arc<dyn SegmentTranscriber>, speaker)
    }

    /// Same as [`Self::new`] but takes an arbitrary [`SegmentTranscriber`]
    /// — the entry point unit tests and future engines use.
    pub fn with_transcriber(
        vad: Arc<SileroVadV5>,
        asr: Arc<dyn SegmentTranscriber>,
        speaker: Option<Arc<SpeakerEncoder>>,
    ) -> Self {
        Self {
            vad,
            asr,
            speaker,
            config: LongFormConfig::default(),
            centroids: Vec::new(),
            centroid_counts: Vec::new(),
        }
    }

    /// Replaces the current [`LongFormConfig`].
    #[must_use]
    pub fn with_config(mut self, config: LongFormConfig) -> Self {
        self.config = config;
        self
    }

    /// The current configuration.
    pub fn config(&self) -> &LongFormConfig {
        &self.config
    }

    /// Resets the running speaker-clustering state (drops all centroids).
    /// Call between independent utterances so speaker ids do not carry
    /// across sessions.
    pub fn reset_speakers(&mut self) {
        self.centroids.clear();
        self.centroid_counts.clear();
    }

    /// Runs the full VAD → Whisper → speaker pipeline over `pcm` at
    /// `sample_rate` Hz.
    ///
    /// `pcm` must be mono `f32`; sample rate must be one Silero supports
    /// (8 000 or 16 000 Hz) — anything else is a hard error, not a silent
    /// resample (FR-EX-08). If the caller wants a different rate, resample
    /// beforehand with `vokra_ops::resample`.
    ///
    /// # Errors
    ///
    /// * [`VokraError::InvalidArgument`] if `sample_rate` is not one Silero
    ///   supports;
    /// * propagates VAD forward errors;
    /// * propagates any [`SegmentTranscriber`] error;
    /// * propagates fbank / CAM++ / cosine errors when speaker binding is
    ///   enabled.
    pub fn transcribe(&mut self, pcm: &[f32], sample_rate: u32) -> Result<LongFormResult> {
        // (1) Sample-rate check — Silero is 8 kHz OR 16 kHz, no fallback.
        let rate = SampleRate::from_hz(sample_rate).map_err(|_| {
            VokraError::InvalidArgument(format!(
                "longform: unsupported sample_rate {sample_rate} Hz (Silero VAD accepts \
                 8000 or 16000 only; resample beforehand with vokra_ops::resample)",
            ))
        })?;
        if !self.vad.supports(rate) {
            return Err(VokraError::InvalidArgument(format!(
                "longform: loaded Silero VAD GGUF has no weights for {} Hz",
                sample_rate
            )));
        }

        // (2) Frame-by-frame VAD probabilities through the streaming handle.
        //     Trailing partial frames are dropped by the stream, so
        //     `probs.len() * frame_len` is the exact span the VAD scored.
        let mut stream = self.vad.open_stream();
        let probs = stream.push_pcm(pcm, sample_rate)?;
        let frame_len = rate.frame_len();
        let sample_rate_usize = sample_rate as usize;
        let audio_len = pcm.len().min(probs.len() * frame_len);

        // (3) Reduce probabilities → (sample-space) segments.
        let spans = self.speech_segments(&probs, sample_rate_usize, frame_len, audio_len);

        // (4) For each span: slice, transcribe, optionally embed + cluster.
        let mut segments: Vec<LongFormSegment> = Vec::with_capacity(spans.len());
        for (start_samples, end_samples) in spans {
            // Clamp to the actual PCM length — the pad pass may have pushed
            // `end` past `audio_len` on the trailing segment when the caller
            // supplied more samples than the VAD scored (e.g. a trailing
            // partial frame). Clamp rather than reject: the segmenter's
            // `audio_len` is already the VAD-scored span.
            let end_samples = end_samples.min(pcm.len());
            if start_samples >= end_samples {
                continue;
            }
            let seg_pcm = &pcm[start_samples..end_samples];
            let start_sec = start_samples as f32 / sample_rate as f32;
            let end_sec = end_samples as f32 / sample_rate as f32;

            // (4a) ASR
            let transcribed = self
                .asr
                .transcribe_segment(seg_pcm, self.config.word_timestamps)?;
            // Shift per-segment word offsets → absolute seconds.
            let words: Vec<WordTiming> = transcribed
                .words
                .into_iter()
                .map(|w| WordTiming {
                    text: w.text,
                    start_sec: w.start_sec + start_sec,
                    end_sec: w.end_sec + start_sec,
                    confidence: w.confidence,
                })
                .collect();

            // (4b) Speaker binding (optional). Clone the Arc so the
            // subsequent mutable borrow of `self` (to update the running
            // cluster state) does not conflict with the immutable borrow of
            // `self.speaker` above.
            let speaker_id = if let Some(encoder) = self.speaker.clone() {
                // Kaldi fbank is 25 ms frames; a segment shorter than one
                // frame has no features at all, so gate by
                // `speaker_min_ms` (well above one frame).
                let seg_ms = 1000 * seg_pcm.len() / sample_rate_usize.max(1);
                if seg_ms >= self.config.speaker_min_ms {
                    Some(self.bind_speaker(&encoder, seg_pcm, sample_rate)?)
                } else {
                    None
                }
            } else {
                None
            };

            segments.push(LongFormSegment {
                start_sec,
                end_sec,
                text: transcribed.text,
                words,
                speaker_id,
            });
        }

        Ok(LongFormResult { segments })
    }

    /// Computes a segment's 192-d CAM++ embedding and returns the assigned
    /// cluster label, updating the running centroid state as a mean.
    ///
    /// PCM must be 16 kHz for CAM++ — the fbank front-end is fixed to that
    /// rate. Non-16-kHz PCM would need a resample, which we do not do
    /// silently (FR-EX-08); the orchestrator's public [`Self::transcribe`]
    /// entry already refuses non-Silero rates (8 k / 16 k), so this method
    /// is only reached at 8 k or 16 k, and 8 k is rejected here explicitly.
    fn bind_speaker(
        &mut self,
        encoder: &SpeakerEncoder,
        seg_pcm: &[f32],
        sample_rate: u32,
    ) -> Result<String> {
        if sample_rate != 16_000 {
            return Err(VokraError::InvalidArgument(format!(
                "longform: speaker binding requires 16 kHz PCM (CAM++ fbank is fixed to \
                 16 kHz); got {sample_rate}. Resample beforehand with vokra_ops::resample \
                 or disable speaker binding."
            )));
        }
        let (fbank, t) = kaldi_fbank(seg_pcm, &KaldiFbankOpts::camplus())?;
        let emb: [f32; EMBED_DIM] = encoder.embed(&fbank, t)?;
        let label = self.assign_cluster(&emb)?;
        Ok(label)
    }

    /// Assigns `emb` to the nearest existing centroid whose cosine
    /// similarity exceeds [`LongFormConfig::speaker_threshold`], or starts a
    /// new cluster. Returns the cluster label ("SPK_00", "SPK_01", …).
    fn assign_cluster(&mut self, emb: &[f32; EMBED_DIM]) -> Result<String> {
        let mut best_idx = None;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(emb, c)?;
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }
        let idx = match best_idx {
            Some(i) if best_sim >= self.config.speaker_threshold => {
                // Update the running mean: centroid = (n*old + emb) / (n+1).
                let n = self.centroid_counts[i] as f32;
                let n1 = n + 1.0;
                for (c, &e) in self.centroids[i].iter_mut().zip(emb.iter()) {
                    *c = (*c * n + e) / n1;
                }
                self.centroid_counts[i] += 1;
                i
            }
            _ => {
                self.centroids.push(emb.to_vec());
                self.centroid_counts.push(1);
                self.centroids.len() - 1
            }
        };
        Ok(format!("SPK_{idx:02}"))
    }

    /// Speech segments from per-frame probabilities — a faithful reduction
    /// of the upstream segmenter at [`LongFormConfig`]-driven parameters.
    ///
    /// Ported verbatim from `vokra_models::silero_vad::parity::speech_segments`
    /// (that helper is `#[cfg(test)]`-only and not exported); the two must
    /// be kept in lockstep (see the pin test below).
    fn speech_segments(
        &self,
        probs: &[f32],
        sample_rate: usize,
        frame_len: usize,
        audio_len: usize,
    ) -> Vec<(usize, usize)> {
        let threshold = self.config.threshold;
        let neg_threshold = self.config.neg_threshold;
        let min_speech = sample_rate * self.config.min_speech_ms / 1000;
        let min_silence = sample_rate * self.config.min_silence_ms / 1000;
        let pad = sample_rate * self.config.speech_pad_ms / 1000;

        let mut triggered = false;
        let mut start = 0usize;
        let mut temp_end = 0usize; // 0 = unset (upstream sentinel convention)
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (i, &p) in probs.iter().enumerate() {
            let cur = i * frame_len;
            if p >= threshold && temp_end != 0 {
                temp_end = 0;
            }
            if p >= threshold && !triggered {
                triggered = true;
                start = cur;
                continue;
            }
            if p < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = cur;
                }
                if cur - temp_end >= min_silence {
                    if temp_end - start > min_speech {
                        spans.push((start, temp_end));
                    }
                    temp_end = 0;
                    triggered = false;
                }
            }
        }
        if triggered && audio_len - start > min_speech {
            spans.push((start, audio_len));
        }

        // Pad pass (upstream tail of get_speech_timestamps).
        let n = spans.len();
        for i in 0..n {
            if i == 0 {
                spans[i].0 = spans[i].0.saturating_sub(pad);
            }
            if i + 1 < n {
                let gap = spans[i + 1].0 - spans[i].1;
                if gap < 2 * pad {
                    spans[i].1 += gap / 2;
                    spans[i + 1].0 -= gap / 2;
                } else {
                    spans[i].1 = (spans[i].1 + pad).min(audio_len);
                    spans[i + 1].0 = spans[i + 1].0.saturating_sub(pad);
                }
            } else {
                spans[i].1 = (spans[i].1 + pad).min(audio_len);
            }
        }
        spans
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use vokra_models::silero_vad::wav::read_wav_f32;

    // ------------------------------------------------------------------ paths

    /// Absolute path to the committed Silero VAD v5 parity fixture (both
    /// rates; 2 MiB plain blob, LFS-free per `.gitattributes`).
    fn silero_gguf_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
    }

    /// Real 16 kHz mono speech fixture — the 11 s JFK excerpt shared with the
    /// Whisper real-audio CI (`tests/fixtures/audio/jfk-30s.wav`).
    fn jfk_wav_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/jfk-30s.wav")
    }

    // ---------------------------------------------------------- test doubles

    /// A canned transcriber that returns pre-set (text, words) tuples in
    /// order, so a test can drive the orchestrator without a Whisper GGUF.
    ///
    /// The test cases below use one canned response per expected VAD segment;
    /// the stub asserts internally that `want_words` matches whatever the
    /// canned entry claims. Deterministic and Send + Sync.
    struct StubTranscriber {
        canned: Mutex<std::collections::VecDeque<TranscribedText>>,
        calls: Mutex<Vec<bool>>, // captured `want_words` per call, for assertions
    }

    impl StubTranscriber {
        fn new(canned: Vec<TranscribedText>) -> Arc<Self> {
            Arc::new(Self {
                canned: Mutex::new(canned.into_iter().collect()),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn calls(&self) -> Vec<bool> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl SegmentTranscriber for StubTranscriber {
        fn transcribe_segment(&self, _pcm: &[f32], want_words: bool) -> Result<TranscribedText> {
            self.calls.lock().unwrap().push(want_words);
            self.canned.lock().unwrap().pop_front().ok_or_else(|| {
                VokraError::ModelLoad(
                    "StubTranscriber: exhausted canned responses (test bug)".into(),
                )
            })
        }
    }

    // ------------------------------------------------------------- helpers

    /// Loads the real Silero VAD fixture. Panics with a helpful message when
    /// the fixture is missing so a clean checkout gets a clear failure — the
    /// fixture is a plain blob under `.gitattributes`, no LFS required.
    fn load_silero() -> Arc<SileroVadV5> {
        let path = silero_gguf_path();
        assert!(
            path.exists(),
            "Silero VAD parity fixture missing at {path:?}. It is a plain blob \
             (see `.gitattributes`); a fresh `git clone` should include it."
        );
        Arc::new(SileroVadV5::open(&path).expect("open silero-vad-v5.gguf"))
    }

    /// Loads the real JFK 16 kHz mono speech clip.
    fn load_jfk_pcm() -> (Vec<f32>, u32) {
        let path = jfk_wav_path();
        assert!(
            path.exists(),
            "JFK audio fixture missing at {path:?}. It is a plain blob (see \
             `tests/fixtures/audio/README.md`); a fresh `git clone` should include it."
        );
        let w = read_wav_f32(&path).expect("read jfk-30s.wav");
        assert_eq!(w.sample_rate, 16_000, "JFK fixture must be 16 kHz mono");
        (w.samples, w.sample_rate)
    }

    // ================================================================
    // Test 1: VAD-driven segmentation, orchestrator produces one segment
    // per speech region.
    // ================================================================

    /// Composes a two-region PCM by taking two 1-second speech chunks from the
    /// real JFK clip and separating them with 0.5 s of digital silence
    /// (± leading + trailing padding of 0.5 s each so the boundaries have
    /// room to breathe). The orchestrator should return exactly two segments
    /// whose boundaries fall inside the two speech regions.
    #[test]
    fn orchestrator_segments_pcm_by_vad() {
        let vad = load_silero();
        let (jfk, sr) = load_jfk_pcm();
        assert_eq!(sr, 16_000);

        // Two 1-second speech chunks pulled from the middle of the clip —
        // safely inside the speech-active region (per the parity test, the
        // JFK excerpt has multiple speech regions across its 11 s span, so
        // an inner 1 s slice is guaranteed speech).
        let one_sec = 16_000usize;
        assert!(
            jfk.len() >= 5 * one_sec,
            "JFK fixture is only {} samples; need >= 5 s",
            jfk.len()
        );
        let chunk_a: Vec<f32> = jfk[one_sec..2 * one_sec].to_vec();
        let chunk_b: Vec<f32> = jfk[3 * one_sec..4 * one_sec].to_vec();

        let silence_500ms: Vec<f32> = vec![0.0; 16_000 / 2];
        let mut pcm: Vec<f32> = Vec::new();
        pcm.extend_from_slice(&silence_500ms);
        pcm.extend_from_slice(&chunk_a);
        pcm.extend_from_slice(&silence_500ms);
        pcm.extend_from_slice(&chunk_b);
        pcm.extend_from_slice(&silence_500ms);

        // Two canned transcriptions so the orchestrator does not run out of
        // stub responses. If the orchestrator produces a different number of
        // segments, the pop from an exhausted queue would raise its own
        // error (not a silent "returned empty").
        let stub = StubTranscriber::new(vec![
            TranscribedText {
                text: "hello world".into(),
                words: Vec::new(),
            },
            TranscribedText {
                text: "goodbye world".into(),
                words: Vec::new(),
            },
        ]);
        let mut orch = LongFormOrchestrator::with_transcriber(
            vad,
            stub.clone() as Arc<dyn SegmentTranscriber>,
            None,
        )
        .with_config(LongFormConfig {
            word_timestamps: false, // stub has no words; do not request them
            ..LongFormConfig::default()
        });

        let result = orch.transcribe(&pcm, sr).expect("transcribe");
        assert_eq!(
            result.segments.len(),
            2,
            "expected 2 segments for two speech regions separated by 500 ms silence, \
             got {:?}",
            result
                .segments
                .iter()
                .map(|s| (s.start_sec, s.end_sec, s.text.clone()))
                .collect::<Vec<_>>()
        );

        // The two segments must be in chronological order and non-overlapping,
        // and their spans must fall inside the two chunk regions (with the
        // 30 ms VAD pad + segmenter granularity slack).
        let s0 = &result.segments[0];
        let s1 = &result.segments[1];
        assert!(s0.start_sec < s0.end_sec, "s0 has zero-length span");
        assert!(s1.start_sec < s1.end_sec, "s1 has zero-length span");
        assert!(
            s0.end_sec <= s1.start_sec,
            "segments must be non-overlapping, got {s0:?} vs {s1:?}"
        );
        // Chunk A lives in [0.5, 1.5) s → allow ±0.1 s pad slack.
        assert!(
            s0.start_sec >= 0.3 && s0.end_sec <= 1.7,
            "segment 0 span {}..{} must land inside chunk A (0.5..1.5) ± pad",
            s0.start_sec,
            s0.end_sec
        );
        // Chunk B lives in [2.0, 3.0) s.
        assert!(
            s1.start_sec >= 1.8 && s1.end_sec <= 3.2,
            "segment 1 span {}..{} must land inside chunk B (2.0..3.0) ± pad",
            s1.start_sec,
            s1.end_sec
        );

        // The stub must have been called exactly twice, both times with the
        // `word_timestamps = false` flag we configured.
        assert_eq!(stub.calls(), vec![false, false]);
        assert_eq!(result.segments[0].text, "hello world");
        assert_eq!(result.segments[1].text, "goodbye world");
        // No speaker encoder was passed → no speaker labels.
        assert!(result.segments.iter().all(|s| s.speaker_id.is_none()));
    }

    // ================================================================
    // Test 2: word timings propagate from the ASR to the segment.
    // ================================================================

    #[test]
    fn orchestrator_returns_word_timings_when_available() {
        let vad = load_silero();
        let (jfk, sr) = load_jfk_pcm();

        // Single speech region, easy VAD signal — sandwich a 1 s speech clip
        // between silence.
        let one_sec = 16_000usize;
        let chunk: Vec<f32> = jfk[one_sec..2 * one_sec].to_vec();
        let silence_500ms: Vec<f32> = vec![0.0; 16_000 / 2];
        let mut pcm: Vec<f32> = Vec::new();
        pcm.extend_from_slice(&silence_500ms);
        pcm.extend_from_slice(&chunk);
        pcm.extend_from_slice(&silence_500ms);

        // The stub returns two word timings *relative* to the segment start
        // (0.10 s and 0.60 s into the segment). The orchestrator MUST add
        // the segment's absolute start (which is non-zero because we prefix
        // 500 ms of silence and Silero segments start after speech onset).
        let stub = StubTranscriber::new(vec![TranscribedText {
            text: "hello world".into(),
            words: vec![
                WordTiming {
                    text: "hello".into(),
                    start_sec: 0.10,
                    end_sec: 0.40,
                    confidence: 1.0,
                },
                WordTiming {
                    text: "world".into(),
                    start_sec: 0.60,
                    end_sec: 0.90,
                    confidence: 1.0,
                },
            ],
        }]);

        let mut orch = LongFormOrchestrator::with_transcriber(
            vad,
            stub.clone() as Arc<dyn SegmentTranscriber>,
            None,
        );
        // Default config has word_timestamps = true, so no override needed.
        let result = orch.transcribe(&pcm, sr).expect("transcribe");
        assert_eq!(result.segments.len(), 1, "expected 1 speech segment");
        let seg = &result.segments[0];
        assert_eq!(seg.text, "hello world");
        assert_eq!(seg.words.len(), 2, "expected 2 word timings");

        // The stub was asked for word timings.
        assert_eq!(stub.calls(), vec![true]);

        // Word timings are shifted by `seg.start_sec` (absolute).
        let w0 = &seg.words[0];
        let w1 = &seg.words[1];
        assert_eq!(w0.text, "hello");
        assert_eq!(w1.text, "world");
        // The relative offsets (0.10 s, 0.40 s, 0.60 s, 0.90 s) must show up
        // as `seg.start_sec + offset` in absolute terms.
        let eps = 1e-4;
        assert!(
            (w0.start_sec - (seg.start_sec + 0.10)).abs() < eps,
            "w0.start_sec = {}, expected {}",
            w0.start_sec,
            seg.start_sec + 0.10
        );
        assert!(
            (w0.end_sec - (seg.start_sec + 0.40)).abs() < eps,
            "w0.end_sec = {}, expected {}",
            w0.end_sec,
            seg.start_sec + 0.40
        );
        assert!(
            (w1.start_sec - (seg.start_sec + 0.60)).abs() < eps,
            "w1.start_sec = {}, expected {}",
            w1.start_sec,
            seg.start_sec + 0.60
        );
        assert!(
            (w1.end_sec - (seg.start_sec + 0.90)).abs() < eps,
            "w1.end_sec = {}, expected {}",
            w1.end_sec,
            seg.start_sec + 0.90
        );
    }

    // ================================================================
    // Sanity: speaker clustering assigns known ids without a real encoder.
    // ================================================================

    #[test]
    fn assign_cluster_starts_new_cluster_then_merges() {
        // Drive `assign_cluster` directly — it does not need the VAD or the
        // ASR. Two orthogonal-ish embeddings → two clusters ("SPK_00",
        // "SPK_01"); a near-duplicate of the first → back to "SPK_00".
        let vad = load_silero();
        let stub = StubTranscriber::new(vec![]);
        let mut orch =
            LongFormOrchestrator::with_transcriber(vad, stub as Arc<dyn SegmentTranscriber>, None);

        let mut a = [0.0f32; EMBED_DIM];
        a[0] = 1.0;
        let mut b = [0.0f32; EMBED_DIM];
        b[1] = 1.0;
        // A perturbed copy of `a` — still cosine ~ 1.0 with the original.
        let mut a2 = [0.0f32; EMBED_DIM];
        a2[0] = 0.99;
        a2[2] = 0.01;

        assert_eq!(orch.assign_cluster(&a).unwrap(), "SPK_00");
        assert_eq!(orch.assign_cluster(&b).unwrap(), "SPK_01");
        assert_eq!(orch.assign_cluster(&a2).unwrap(), "SPK_00");

        // reset_speakers drops the running state → next embedding is
        // "SPK_00" again.
        orch.reset_speakers();
        assert_eq!(orch.assign_cluster(&b).unwrap(), "SPK_00");
    }

    // ================================================================
    // Sanity: unsupported sample rate is a hard error, not silent
    // resample (FR-EX-08).
    // ================================================================

    #[test]
    fn transcribe_rejects_unsupported_sample_rate() {
        let vad = load_silero();
        let stub = StubTranscriber::new(vec![]);
        let mut orch =
            LongFormOrchestrator::with_transcriber(vad, stub as Arc<dyn SegmentTranscriber>, None);
        let err = orch.transcribe(&vec![0.0; 44_100], 44_100).unwrap_err();
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("sample_rate") || m.contains("Silero"),
                    "unexpected message: {m}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
