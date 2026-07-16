//! Adapters binding the Whisper decoder to the model-independent decoding
//! traits ([`LogitsSource`] / [`BeamScorer`], M0-06-T23 + M1-04).
//!
//! [`WhisperLogitsSource`] answers the raw-logits question — "logits of the next
//! token given this prefix" — by running the decoder over the full prefix. It is
//! the primitive both the [`Sampler`](vokra_core::decode::Sampler) and beam
//! search consume; [`WhisperBeamScorer`] is a **thin adapter** layering the
//! `log_softmax` that [`beam_search`](vokra_core::decode::beam_search) wants on
//! top of it.
//!
//! # Per-beam KV cache (M0: recompute)
//!
//! Each query hands the full token sequence, so the source **recomputes** from a
//! reset cache every call: correctness-first, no cache aliasing between beams.
//! Efficient per-beam cache reuse / reordering (FR-EX-02 / M1-04) is a later
//! optimization behind the same interface.

use std::sync::Arc;

use vokra_core::decode::word_timing::{
    AlignmentParams, CrossAttention, WordTiming, token_alignment, words_from_alignment,
};
use vokra_core::decode::{BeamScorer, LogitsSource};
use vokra_core::{Result, VokraError};

use super::WhisperModel;
use super::decoder::DecoderState;
use super::encoder::EncoderOutput;
use super::tokenizer::WhisperTokenizer;

/// Whisper audio-chunk duration in seconds (`N_SAMPLES = 480000` at
/// `SAMPLE_RATE = 16000` → 30 s; openai-whisper `audio.py`). Each audio token
/// spans `WHISPER_CHUNK_SECONDS / n_audio_ctx` (0.02 s for base's 1500 frames,
/// ADR M4-20 §D-2 / §D-3).
const WHISPER_CHUNK_SECONDS: f32 = 30.0;

/// [`LogitsSource`] over a Whisper decoder bound to one encoder output.
///
/// Owns its [`DecoderState`] (which owns the model via an [`Arc`]), so the
/// source carries no lifetime and can drive greedy, sampled or beam decoding.
pub struct WhisperLogitsSource {
    state: DecoderState,
    vocab: usize,
}

impl WhisperLogitsSource {
    /// Builds a source for `encoder`'s audio (precomputes cross-attention K/V).
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        let vocab = model.config().n_vocab;
        let state = model.decoder(encoder)?;
        Ok(Self { state, vocab })
    }
}

impl LogitsSource for WhisperLogitsSource {
    fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.state.reset();
        self.state.step_last(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// [`BeamScorer`] over a Whisper decoder: a thin `log_softmax` adapter on top of
/// [`WhisperLogitsSource`].
///
/// An optional borrowed [`WhisperTokenizer`] drives the subword→word merge in
/// [`align_words`](BeamScorer::align_words): with a tokenizer the alignment
/// returns one [`WordTiming`] per **word**; without one it returns Whisper's
/// per-token internal timing (unchanged M4-20 (a) behaviour). The borrow lives
/// no longer than the transcribe call that builds the scorer.
pub struct WhisperBeamScorer<'t> {
    source: WhisperLogitsSource,
    tokenizer: Option<&'t WhisperTokenizer>,
}

impl WhisperBeamScorer<'static> {
    /// Builds a scorer for `encoder`'s audio (precomputes cross-attention K/V)
    /// with **no** tokenizer — `align_words` yields per-token timings.
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        Ok(Self {
            source: WhisperLogitsSource::new(model, encoder)?,
            tokenizer: None,
        })
    }
}

impl<'t> WhisperBeamScorer<'t> {
    /// Builds a scorer that merges subword timings into word timings using
    /// `tokenizer` (M4-20, FR-OP-40). `align_words` then returns one
    /// [`WordTiming`] per word.
    pub(crate) fn with_tokenizer(
        model: Arc<WhisperModel>,
        encoder: &EncoderOutput,
        tokenizer: &'t WhisperTokenizer,
    ) -> Result<Self> {
        Ok(Self {
            source: WhisperLogitsSource::new(model, encoder)?,
            tokenizer: Some(tokenizer),
        })
    }
}

impl BeamScorer for WhisperBeamScorer<'_> {
    fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(log_softmax(&self.source.logits(tokens)?))
    }

    fn vocab_size(&self) -> usize {
        self.source.vocab_size()
    }

    /// Word-level timestamps via Whisper cross-attention DTW (M4-20,
    /// FR-OP-40). Returns `Ok(None)` when the model carries no alignment-head
    /// blob (`vokra.whisper.alignment_heads`) — [`beam_search`] then raises the
    /// explicit FR-EX-08 error (never a silent no-op, ADR M4-20 §D-3).
    ///
    /// When a tokenizer was attached
    /// ([`with_tokenizer`](WhisperBeamScorer::with_tokenizer)), the per-token
    /// alignment is merged into per-**word** timings
    /// ([`WhisperTokenizer::word_token_lens`] +
    /// [`words_from_alignment`]); without one it stays per-token.
    fn align_words(&mut self, tokens: &[u32]) -> Result<Option<Vec<WordTiming>>> {
        whisper_word_timings(&mut self.source.state, tokens, self.tokenizer)
    }
}

/// Computes Whisper word timings for `tokens` (the full best hypothesis,
/// including the forced prefix and the trailing token), or `Ok(None)` when the
/// model supplies no alignment heads.
///
/// The alignment logic (DTW / median filter / normalize / jumps) lives in
/// [`vokra_core::decode::word_timing`]; this function is the minimal Whisper
/// *consuming* wiring (ADR M4-20 §D-3): capture the selected alignment heads'
/// cross-attention, strip the forced prefix and trailing token (openai-whisper
/// `matrix[len(sot):-1]`), and hand the weight stack to the core to get one
/// start time per content token.
///
/// When `tokenizer` is `Some`, those per-token times are merged into per-**word**
/// timings ([`WhisperTokenizer::word_token_lens`] +
/// [`words_from_alignment`]) — the subword→word grouping openai-whisper does in
/// `split_to_word_tokens`. When `None`, the result is per-token (exactly
/// Whisper's internal timing before that merge). Either way the returned
/// `token_start` / `token_end` are absolute indices into `tokens`.
fn whisper_word_timings(
    state: &mut DecoderState,
    tokens: &[u32],
    tokenizer: Option<&WhisperTokenizer>,
) -> Result<Option<Vec<WordTiming>>> {
    // Clone the config so the immutable borrow does not conflict with the
    // `&mut state` capture call below.
    let cfg = state.config().clone();
    if cfg.alignment_heads.is_empty() {
        // No alignment-head blob → this model cannot produce word timestamps.
        // `None` makes beam_search raise the explicit FR-EX-08 error.
        return Ok(None);
    }
    let n_head = cfg.n_text_head;
    let n_layer = cfg.n_text_layer;
    let n_prefix = cfg.decoder_start_ids.len();
    let t = tokens.len();
    // Content tokens = strip the forced prefix and the trailing token
    // (openai-whisper matrix[len(sot):-1]). Fewer than one content token → no
    // words to align (an empty, still-valid alignment; not an error).
    if t <= n_prefix + 1 {
        return Ok(Some(Vec::new()));
    }
    let text_lo = n_prefix;
    let n_text = (t - 1) - text_lo;

    // Validate the alignment heads against the captured shape (FR-EX-08:
    // an out-of-range head is an explicit error, never silently skipped).
    for &(l, h) in &cfg.alignment_heads {
        if l >= n_layer || h >= n_head {
            return Err(VokraError::InvalidArgument(format!(
                "whisper align: alignment head ({l},{h}) out of range \
                 (n_text_layer {n_layer}, n_text_head {n_head})"
            )));
        }
    }

    // Capture cross-attention [n_layer, n_head, t, n_ctx].
    let captured = state.cross_attention_weights(tokens)?;
    let per_layer = n_head * t;
    let n_ctx = captured.len() / (n_layer * per_layer);
    debug_assert_eq!(captured.len(), n_layer * per_layer * n_ctx);

    // Stack the selected heads → [n_selected, n_text, n_ctx].
    let n_sel = cfg.alignment_heads.len();
    let mut weights = vec![0.0f32; n_sel * n_text * n_ctx];
    for (s, &(l, h)) in cfg.alignment_heads.iter().enumerate() {
        for ti in 0..n_text {
            let src = ((l * n_head + h) * t + (text_lo + ti)) * n_ctx;
            let dst = (s * n_text + ti) * n_ctx;
            weights[dst..dst + n_ctx].copy_from_slice(&captured[src..src + n_ctx]);
        }
    }
    let attn = CrossAttention {
        weights,
        n_head: n_sel,
        n_text,
        n_audio: n_ctx,
    };

    // Whisper audio-token rate: the 30 s window / the model's frame count.
    let dt = WHISPER_CHUNK_SECONDS / cfg.n_audio_ctx as f32;
    let params = AlignmentParams {
        median_filter_width: 7,
        audio_time_per_token: dt,
    };
    let times = token_alignment(&attn, &params)?; // per-content-token start times
    let final_time = n_ctx as f32 * dt;

    let out = match tokenizer {
        // Subword→word merge (openai split_to_word_tokens). The content tokens
        // are `tokens[text_lo .. text_lo + n_text]`; `word_token_lens` sums to
        // `n_text` (== `times.len()`), so `words_from_alignment` accepts it.
        Some(tok) => {
            let content = &tokens[text_lo..text_lo + n_text];
            let word_lens = tok.word_token_lens(content)?;
            let mut words = words_from_alignment(&times, &word_lens, final_time)?;
            // `words_from_alignment` indexes into the content slice (0-based);
            // shift back to absolute indices into `tokens` for the caller.
            for w in &mut words {
                w.token_start += text_lo;
                w.token_end += text_lo;
            }
            words
        }
        // No tokenizer → per-token timing (Whisper's internal granularity).
        None => {
            let mut out = Vec::with_capacity(n_text);
            for i in 0..n_text {
                let start = times[i];
                let end = if i + 1 < n_text {
                    times[i + 1]
                } else {
                    final_time
                }
                .max(start);
                out.push(WordTiming {
                    token_start: text_lo + i,
                    token_end: text_lo + i + 1,
                    start,
                    end,
                });
            }
            out
        }
    };
    Ok(Some(out))
}

/// Numerically stable `log_softmax` over a logits vector.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let log_sum = max as f64 + sum.ln();
    logits
        .iter()
        .map(|&l| (l as f64 - log_sum) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::decode::{BeamSearchConfig, SamplerConfig, beam_search, sample_sequence};

    use crate::whisper::decoder::test_support::{tiny_encoder, tiny_model, tiny_model_aligned};
    use crate::whisper::greedy::greedy_decode;

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, -1.0]);
        let s: f64 = lp.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "sum {s}");
        // Monotonic: larger logit → larger log-prob.
        assert!(lp[2] > lp[1] && lp[1] > lp[0] && lp[0] > lp[3]);
    }

    /// Temperature-0 sampling through the [`WhisperLogitsSource`] must reproduce
    /// the incremental greedy decoder token-for-token. This is the CI-runnable
    /// oracle for the sampled-transcribe wiring: the recompute-per-step source
    /// (reset + `step_last` on the full prefix) and the incremental greedy loop
    /// agree because reset+replay is bit-identical to incremental decoding.
    #[test]
    fn greedy_sampling_over_logits_source_matches_greedy_decode() {
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;

        let mut st = model.decoder(&enc).unwrap();
        let greedy = greedy_decode(&mut st, &start, eot, 6).unwrap();

        let mut src = WhisperLogitsSource::new(Arc::clone(&model), &enc).unwrap();
        let sampled = sample_sequence(&mut src, &start, eot, &SamplerConfig::greedy(), 6).unwrap();

        assert_eq!(
            greedy, sampled,
            "temperature-0 sampling must equal greedy decode"
        );
    }

    // ---- M4-20 (a): word-timestamp wiring (synthetic model, structural) -----

    /// A model with **no** alignment heads reports `Ok(None)` from
    /// `align_words`, so `beam_search` with `word_timestamps` raises the
    /// explicit FR-EX-08 error — never a silent no-op.
    #[test]
    fn no_alignment_heads_makes_word_timestamps_explicit_error() {
        let model = tiny_model(2); // tiny_cfg → empty alignment_heads
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc).unwrap();
        // Direct: the scorer supplies no alignment.
        assert!(scorer.align_words(&[1, 2, 0]).unwrap().is_none());

        // End-to-end through beam_search: explicit UnsupportedOp.
        let mut cfg = BeamSearchConfig::new(2, 6);
        cfg.word_timestamps = true;
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;
        match beam_search(&mut scorer, &start, eot, &cfg) {
            Err(VokraError::UnsupportedOp(_)) => {}
            other => panic!("expected UnsupportedOp without alignment heads, got {other:?}"),
        }
    }

    /// A model WITH alignment heads runs the full capture → DTW → WordTiming
    /// stack (synthetic weights). The numeric timestamps against real
    /// openai-whisper need a real checkpoint (owner, GGUF-gated); here we assert
    /// the **structural** contract: one timing per content token, spans in
    /// order, times within the audio window, monotone non-decreasing starts.
    #[test]
    fn alignment_heads_produce_structural_word_timings() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let n_ctx = 4usize;
        let dt = WHISPER_CHUNK_SECONDS / model.config().n_audio_ctx as f32;
        let window = n_ctx as f32 * dt;

        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc).unwrap();
        // tokens = prefix (start_ids, len 1) + 3 content tokens + trailing eot.
        let tokens = [1u32, 2, 1, 2, 0];
        let n_prefix = model.config().decoder_start_ids.len();
        let expected = tokens.len() - 1 - n_prefix; // strip prefix + trailing

        let timings = scorer
            .align_words(&tokens)
            .unwrap()
            .expect("aligned model returns Some");
        assert_eq!(timings.len(), expected, "one timing per content token");

        for (k, w) in timings.iter().enumerate() {
            assert_eq!(w.token_start, n_prefix + k);
            assert_eq!(w.token_end, n_prefix + k + 1);
            assert!(w.start <= w.end, "start <= end: {w:?}");
            assert!(
                (0.0..=window + 1e-3).contains(&w.start) && (0.0..=window + 1e-3).contains(&w.end),
                "times within [0, window={window}]: {w:?}"
            );
            if k > 0 {
                assert!(
                    w.start >= timings[k - 1].start - 1e-6,
                    "word starts must be non-decreasing: {timings:?}"
                );
            }
        }
    }

    /// A hypothesis with no content tokens (prefix + trailing only) aligns to an
    /// empty — but still `Some` — list (not an error, not a silent None).
    #[test]
    fn aligned_model_with_no_content_tokens_is_empty_not_none() {
        let model = tiny_model_aligned(1);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc).unwrap();
        // prefix len 1 + trailing eot only → zero content tokens.
        let timings = scorer.align_words(&[1u32, 0]).unwrap();
        assert_eq!(timings, Some(Vec::new()));
    }

    /// Builds a synthetic tokenizer over the tiny model's 3-token vocab
    /// (ids 0/1/2) with a chosen token→string map so the subword→word merge
    /// is exercised without a real checkpoint.
    #[cfg(test)]
    fn tiny_tokenizer(entries: &[(u8, &[u8])]) -> WhisperTokenizer {
        let mut v = Vec::new();
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (sp, bytes) in entries {
            v.push(*sp);
            v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        // eot = 0 (matches `tiny_cfg`).
        WhisperTokenizer::from_bytes(&v, 0).unwrap()
    }

    /// With a tokenizer attached, `align_words` merges subword timings into
    /// per-**word** timings (M4-20 (a) — the tokenizer follow-up). The three
    /// content tokens `[2, 1, 2]` decode (id 2 = `" hel"`, id 1 = `"lo"`,
    /// id 0 = special/eot) to `" hel" | "lo" | " hel"`, which
    /// `word_token_lens` groups as `[2, 1]` (`" hello"` then `" hel"`). So the
    /// alignment must return **2** timings — one per word, not one per token —
    /// with contiguous absolute token spans in order.
    #[test]
    fn tokenizer_merges_subword_timings_into_words() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let tok = tiny_tokenizer(&[
            (1, b""),     // id 0: special (eot)
            (0, b"lo"),   // id 1: continuation, no leading space
            (0, b" hel"), // id 2: leading space → word start
        ]);
        let mut scorer = WhisperBeamScorer::with_tokenizer(Arc::clone(&model), &enc, &tok).unwrap();

        // prefix (len 1) + content [2, 1, 2] + trailing eot.
        let tokens = [1u32, 2, 1, 2, 0];
        let n_prefix = model.config().decoder_start_ids.len();
        let n_ctx = 4usize;
        let dt = WHISPER_CHUNK_SECONDS / model.config().n_audio_ctx as f32;
        let window = n_ctx as f32 * dt;

        let timings = scorer
            .align_words(&tokens)
            .unwrap()
            .expect("aligned model returns Some");

        // Two WORDS (`[2, 1]`), not three tokens — this is the merge.
        assert_eq!(timings.len(), 2, "one timing per word: {timings:?}");
        // Absolute, contiguous, ordered spans covering the content tokens.
        assert_eq!(timings[0].token_start, n_prefix); // 1
        assert_eq!(timings[0].token_end, n_prefix + 2); // 3  (tokens 1..3)
        assert_eq!(timings[1].token_start, n_prefix + 2); // 3
        assert_eq!(timings[1].token_end, n_prefix + 3); // 4  (token 3)
        for (k, w) in timings.iter().enumerate() {
            assert!(w.start <= w.end, "start <= end: {w:?}");
            assert!(
                (0.0..=window + 1e-3).contains(&w.start) && (0.0..=window + 1e-3).contains(&w.end),
                "times within [0, window={window}]: {w:?}"
            );
            if k > 0 {
                assert!(
                    w.start >= timings[k - 1].start - 1e-6,
                    "word starts must be non-decreasing: {timings:?}"
                );
            }
        }
    }

    /// Regression: a tokenizer whose vocab makes every content token its own
    /// word (each id has a leading space) yields one timing per token — the
    /// merge collapses to the per-token result, matching the no-tokenizer path.
    #[test]
    fn tokenizer_one_word_per_token_matches_per_token_path() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        // Every non-special id starts with a space → every subword is its own
        // word.
        let tok = tiny_tokenizer(&[(1, b""), (0, b" a"), (0, b" b")]);
        let tokens = [1u32, 2, 1, 2, 0];

        let mut with_tok =
            WhisperBeamScorer::with_tokenizer(Arc::clone(&model), &enc, &tok).unwrap();
        let mut without = WhisperBeamScorer::new(Arc::clone(&model), &enc).unwrap();

        let merged = with_tok.align_words(&tokens).unwrap().unwrap();
        let per_token = without.align_words(&tokens).unwrap().unwrap();
        assert_eq!(
            merged, per_token,
            "one-word-per-token vocab must equal the per-token alignment",
        );
    }
}
