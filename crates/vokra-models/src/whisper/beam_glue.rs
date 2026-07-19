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
    APPEND_PUNCTUATIONS, AlignmentParams, CrossAttention, PREPEND_PUNCTUATIONS, WordTiming,
    merge_punctuations, token_alignment, words_from_alignment,
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

/// Valid (non-padding) cross-attention audio positions for a `pcm_len`-sample
/// clip: openai-whisper restricts the alignment weights to
/// `weights[:, :, : num_frames // 2]` (timing.py:208) where `num_frames` is
/// the clip's **unpadded** mel frame count for the window
/// (`segment_size = min(N_FRAMES, content_frames)`, transcribe.py:283, with
/// `content_frames = len(audio) // HOP_LENGTH`). The `// 2` maps mel frames
/// (100 / s) to encoder audio positions (50 / s, conv2 stride 2). Without this
/// restriction the DTW path is forced to span the zero-padded tail of the
/// 30 s window and the trailing words leak toward 30 s (campaign-2 P2).
pub(crate) fn valid_audio_positions(pcm_len: usize) -> usize {
    use super::mel::{HOP, N_FRAMES};
    N_FRAMES.min(pcm_len / HOP) / 2
}

/// Snapshot-cache capacity: two beam generations (parents + children) for
/// beam widths up to ~16, plus slack. Overflow evicts the oldest entry;
/// eviction only costs a replay, never correctness (M5-14-T13).
const MAX_KV_SNAPSHOTS: usize = 40;

/// [`LogitsSource`] over a Whisper decoder bound to one encoder output.
///
/// Owns its [`DecoderState`] (which owns the model via an [`Arc`]), so the
/// source carries no lifetime and can drive greedy, sampled or beam decoding.
///
/// # Per-beam incremental KV (M5-14-T13)
///
/// The M0 posture recomputed the whole prefix from a reset cache on every
/// query — O(len²) token-forwards per hypothesis over a beam decode. The
/// source now keeps a small cache of `(committed tokens → self-KV snapshot)`
/// pairs ([`DecoderState::selfkv_snapshot`], the Voxtral
/// `TextDecoderKvSnapshot` branch-primitive pattern): a query whose tokens
/// extend a cached entry by exactly one token restores that snapshot and
/// steps ONLY the new token. Restoring a byte-identical KV cache and
/// stepping is **bit-identical** to reset + full replay (the KV rows a
/// replay would recompute are projections of the same inputs — the
/// `incremental_source_bitwise_matches_full_recompute` oracle pins `==`),
/// so beam / sampled results are unchanged — beam_width = 1 in particular
/// now runs the exact greedy step sequence. Queries with no cached parent
/// (the first prefix query, an evicted parent, a device-session-backed
/// state) fall back to the old reset + replay, which is always correct.
pub struct WhisperLogitsSource {
    state: DecoderState,
    vocab: usize,
    /// `(committed token sequence, self-KV snapshot after those tokens)`.
    kv_snaps: Vec<(Vec<u32>, vokra_core::KvCache)>,
}

impl WhisperLogitsSource {
    /// Builds a source for `encoder`'s audio (precomputes cross-attention K/V).
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        let vocab = model.config().n_vocab;
        let state = model.decoder(encoder)?;
        Ok(Self {
            state,
            vocab,
            kv_snaps: Vec::new(),
        })
    }

    /// Records the state's current self-KV under `tokens` and evicts stale
    /// generations (anything shorter than `tokens.len() - 1` can never be a
    /// parent of a later query in a monotonically-growing search).
    fn remember(&mut self, tokens: &[u32]) {
        if !self.state.kv_branching_supported() {
            return;
        }
        let n = tokens.len();
        self.kv_snaps.retain(|(k, _)| k.len() + 1 >= n);
        if self.kv_snaps.iter().any(|(k, _)| k == tokens) {
            return;
        }
        if self.kv_snaps.len() >= MAX_KV_SNAPSHOTS {
            self.kv_snaps.remove(0);
        }
        self.kv_snaps
            .push((tokens.to_vec(), self.state.selfkv_snapshot()));
    }
}

impl LogitsSource for WhisperLogitsSource {
    fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Incremental path: restore the parent's KV, step the last token.
        if self.state.kv_branching_supported()
            && let Some(n) = tokens.len().checked_sub(1)
            && n > 0
        {
            let parent = &tokens[..n];
            if let Some(idx) = self
                .kv_snaps
                .iter()
                .position(|(k, _)| k.len() == n && k == parent)
            {
                let (_, snap) = &self.kv_snaps[idx];
                self.state.selfkv_restore(snap);
                let out = self.state.step_last(&tokens[n..])?;
                self.remember(tokens);
                return Ok(out);
            }
        }
        // Fallback: the M0 reset + full-prefix replay (always correct).
        self.state.reset();
        let out = self.state.step_last(tokens)?;
        self.remember(tokens);
        Ok(out)
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
    /// Valid (non-padding) audio positions for the alignment column
    /// restriction ([`valid_audio_positions`]; openai timing.py:208). The
    /// entry point derives it from the clip's true PCM length; capped at the
    /// captured window inside [`whisper_word_timings`].
    n_valid_audio: usize,
}

impl WhisperBeamScorer<'static> {
    /// Builds a scorer for `encoder`'s audio (precomputes cross-attention K/V)
    /// with **no** tokenizer — `align_words` yields per-token timings.
    /// `n_valid_audio` is the clip's valid (non-padding) audio-position count
    /// ([`valid_audio_positions`]).
    pub(crate) fn new(
        model: Arc<WhisperModel>,
        encoder: &EncoderOutput,
        n_valid_audio: usize,
    ) -> Result<Self> {
        Ok(Self {
            source: WhisperLogitsSource::new(model, encoder)?,
            tokenizer: None,
            n_valid_audio,
        })
    }
}

impl<'t> WhisperBeamScorer<'t> {
    /// Builds a scorer that merges subword timings into word timings using
    /// `tokenizer` (M4-20, FR-OP-40). `align_words` then returns one
    /// [`WordTiming`] per word. `n_valid_audio` as in
    /// [`new`](WhisperBeamScorer::new).
    pub(crate) fn with_tokenizer(
        model: Arc<WhisperModel>,
        encoder: &EncoderOutput,
        tokenizer: &'t WhisperTokenizer,
        n_valid_audio: usize,
    ) -> Result<Self> {
        Ok(Self {
            source: WhisperLogitsSource::new(model, encoder)?,
            tokenizer: Some(tokenizer),
            n_valid_audio,
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
        whisper_word_timings(
            &mut self.source.state,
            tokens,
            self.tokenizer,
            self.n_valid_audio,
        )
    }
}

/// Computes Whisper word timings for `tokens` (the full best hypothesis,
/// including the forced prefix and the trailing token), or `Ok(None)` when the
/// model supplies no alignment heads.
///
/// The alignment logic (DTW / median filter / normalize / jumps) lives in
/// [`vokra_core::decode::word_timing`]; this function is the minimal Whisper
/// *consuming* wiring (ADR M4-20 §D-3), mirroring openai-whisper
/// `timing.py::find_alignment`:
///
/// * **Row window** (timing.py:215 `matrix[len(tokenizer.sot_sequence) : -1]`):
///   the kept rows are the query positions `n_prefix - 1 .. t - 1` — the
///   `n_text + 1` **emission rows** of `[c_0 .. c_{n-1}, eot]`. The first kept
///   row's input is the last forced-prefix token (`<|notimestamps|>`); its
///   attention is where `c_0` is emitted. `decoder_start_ids` INCLUDES
///   `<|notimestamps|>` while openai's `sot_sequence` EXCLUDES it, hence
///   `n_prefix - 1` here == `len(sot_sequence)` there. Slicing at `n_prefix`
///   instead was the campaign-2 P1 off-by-one: every word start landed at the
///   previous word's emission (≈ the reference word's END, mean 212-443 ms).
/// * **Column restriction** (timing.py:208-209): only the clip's valid
///   (non-padding) audio positions enter the alignment; the model's softmaxed
///   probabilities are renormalized over the kept columns (softmax over a
///   prefix of the logits == full softmax restricted + renormalized).
/// * **Terminal bound** (timing.py:226/231): the eot row's arrival bounds the
///   LAST word's end — never the padded-window end `n_ctx * dt` (the
///   campaign-2 "final word ends at 30.000 s" pad leak).
/// * **Punctuation merge** (timing.py:245): with a tokenizer, bare punctuation
///   words are folded into their neighbours per the upstream default sets.
///
/// When `tokenizer` is `Some`, per-token times are merged into per-**word**
/// timings ([`WhisperTokenizer::word_token_lens`] + [`words_from_alignment`] +
/// [`merge_punctuations`]) — the subword→word grouping openai-whisper does in
/// `split_to_word_tokens` plus its punctuation merge. When `None`, the result
/// is per-token (Whisper's internal timing before those merges). Either way
/// the returned `token_start` / `token_end` are absolute indices into
/// `tokens`.
///
/// A hypothesis that stopped at the token budget (no trailing eot) is aligned
/// the same way; its final row is then the last generated token's emission,
/// which still bounds the earlier tokens (degenerate, not the openai path —
/// openai always appends eot to the forced sequence).
fn whisper_word_timings(
    state: &mut DecoderState,
    tokens: &[u32],
    tokenizer: Option<&WhisperTokenizer>,
    n_valid_audio: usize,
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
    if n_prefix == 0 {
        return Err(VokraError::InvalidArgument(
            "whisper align: empty decoder_start_ids (no forced prefix, so no \
             first-emission row exists)"
                .into(),
        ));
    }
    let t = tokens.len();
    // No content tokens between the forced prefix and the trailing token → no
    // words to align (an empty, still-valid alignment; not an error) — openai
    // find_alignment returns [] for empty text_tokens.
    if t <= n_prefix + 1 {
        return Ok(Some(Vec::new()));
    }
    // Emission-row window (timing.py:215): rows `emit_lo .. t - 1`, i.e. the
    // n_text + 1 rows emitting [c_0 .. c_{n-1}, <terminal>]. Content tokens
    // themselves start one later, at `n_prefix`.
    let emit_lo = n_prefix - 1;
    let n_rows = (t - 1) - emit_lo;
    let n_text = n_rows - 1;

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

    // Valid-column restriction (timing.py:208 `weights[:, :, : num_frames//2]`).
    // The capture is the model's softmax over the FULL padded window, so the
    // kept columns are renormalized per row — bit-for-bit the same value as
    // softmaxing the restricted logits (timing.py:209) up to f32 rounding.
    let n_audio = n_valid_audio.min(n_ctx);
    if n_audio == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "whisper align: no valid audio positions (clip shorter than one \
             audio token; n_valid_audio {n_valid_audio}, window {n_ctx})"
        )));
    }

    // Stack the selected heads' emission rows → [n_selected, n_rows, n_audio].
    let n_sel = cfg.alignment_heads.len();
    let mut weights = vec![0.0f32; n_sel * n_rows * n_audio];
    for (s, &(l, h)) in cfg.alignment_heads.iter().enumerate() {
        for ri in 0..n_rows {
            let src = ((l * n_head + h) * t + (emit_lo + ri)) * n_ctx;
            let dst = (s * n_rows + ri) * n_audio;
            let row = &captured[src..src + n_audio];
            let sum: f64 = row.iter().map(|&w| w as f64).sum();
            if sum <= 0.0 {
                // All cross-attention mass sits beyond the valid frames —
                // renormalizing would fabricate NaN/Inf weights (FR-EX-08).
                // (A NaN sum skips this guard but is still rejected by
                // `CrossAttention::validate`'s finiteness check.)
                return Err(VokraError::InvalidArgument(format!(
                    "whisper align: cross-attention mass vanished in the valid \
                     audio window (head ({l},{h}), emission row {ri})"
                )));
            }
            for j in 0..n_audio {
                weights[dst + j] = (row[j] as f64 / sum) as f32;
            }
        }
    }
    let attn = CrossAttention {
        weights,
        n_head: n_sel,
        n_text: n_rows,
        n_audio,
    };

    // Whisper audio-token rate: the 30 s window / the model's frame count.
    let dt = WHISPER_CHUNK_SECONDS / cfg.n_audio_ctx as f32;
    let params = AlignmentParams {
        median_filter_width: 7,
        audio_time_per_token: dt,
    };
    // n_rows arrival times: [0 .. n_text) = per-content-token starts; the last
    // entry is the terminal (eot) emission arrival that bounds the last word
    // (timing.py:226/231 `jump_times[word_boundaries[1:]]`).
    let times = token_alignment(&attn, &params)?;
    let eot_time = times[n_rows - 1];

    let out = match tokenizer {
        // Subword→word merge (openai split_to_word_tokens). The content tokens
        // are `tokens[n_prefix .. n_prefix + n_text]`; `word_token_lens` sums
        // to `n_text`, so `words_from_alignment` accepts the content times.
        Some(tok) => {
            let content = &tokens[n_prefix..n_prefix + n_text];
            let word_lens = tok.word_token_lens(content)?;
            let mut words = words_from_alignment(&times[..n_text], &word_lens, eot_time)?;
            // `words_from_alignment` indexes into the content slice (0-based);
            // shift back to absolute indices into `tokens` for the caller.
            for w in &mut words {
                w.token_start += n_prefix;
                w.token_end += n_prefix;
            }
            // Punctuation merge (timing.py:245, upstream default sets): bare
            // punctuation words fold into their neighbours; spans widen, the
            // absorbing word's own start/end stay.
            let mut texts = Vec::with_capacity(words.len());
            for w in &words {
                texts.push(tok.decode(&tokens[w.token_start..w.token_end])?);
            }
            merge_punctuations(
                &mut words,
                &mut texts,
                PREPEND_PUNCTUATIONS,
                APPEND_PUNCTUATIONS,
            )?;
            words
        }
        // No tokenizer → per-token timing (Whisper's internal granularity).
        // Each token ends at the NEXT emission arrival; the last content
        // token is bounded by the terminal (eot) row.
        None => {
            let mut out = Vec::with_capacity(n_text);
            for i in 0..n_text {
                let start = times[i];
                let end = times[i + 1].max(start);
                out.push(WordTiming {
                    token_start: n_prefix + i,
                    token_end: n_prefix + i + 1,
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

    /// M5-14-T13 bit-identity oracle: for a beam-shaped query schedule —
    /// a first (batched) prefix query, sibling children of one parent
    /// queried interleaved (as `beam_search` does), and a cold jump whose
    /// parent generation was evicted — the incremental-KV source must
    /// reproduce **exactly** (`==`) the greedy-style decode of the same
    /// hypothesis: the first query's tokens as one batched `step_into`,
    /// every later token as a single-token step. (The old per-query
    /// full-prefix recompute is only ulp-close to this — batched and
    /// stepped forwards differ in the last bit, which is why the legacy
    /// full-vs-cached oracle carries a 1e-4 tolerance — so greedy-style
    /// stepping is the reference the T13 gate is defined against.)
    #[test]
    fn incremental_source_bitwise_matches_greedy_style_stepping() {
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);

        // (query, batched-head length): the head replays as ONE batched
        // step — query 0 seeds the cache; the final [1, 2] re-query has no
        // len-1 parent left (generational eviction) so the source replays
        // it batched (head = 2).
        let queries: [(&[u32], usize); 7] = [
            (&[1, 2], 2),
            (&[1, 2, 1], 2),
            (&[1, 2, 0], 2),
            (&[1, 2, 1, 2], 2),
            (&[1, 2, 0, 1], 2),
            (&[1, 2, 1, 2, 0], 2),
            (&[1, 2], 2),
        ];

        let mut inc = WhisperLogitsSource::new(Arc::clone(&model), &enc).unwrap();
        for (q, head) in queries {
            let got = inc.logits(q).unwrap();

            // Greedy-style reference: batched head + single-token steps.
            let mut st = model.decoder(&enc).unwrap();
            st.reset();
            st.step_into(&q[..head]).unwrap();
            for tok in &q[head..] {
                st.step_into(std::slice::from_ref(tok)).unwrap();
            }
            let want = st.last_logits_row().to_vec();
            assert_eq!(got, want, "query {q:?}: incremental != greedy-style");
        }
    }

    /// M5-14-T13 regression pin: beam search over the incremental-KV scorer
    /// must select the same hypotheses as over an M0-style always-recompute
    /// scorer (reset + full-prefix replay per query) on the synthetic
    /// fixture, at widths 1 and 2. (On this fixture the tiny logit spread
    /// collapses under `log_softmax` f32 rounding, so token selection is
    /// tie-break-driven — identical inputs ⇒ identical selection either
    /// way. The real-weight beam-1 ≡ greedy transcript gate runs on the
    /// converted GGUF — Wave-2 acceptance spot-check.)
    #[test]
    fn beam_selection_unchanged_vs_recompute_scorer() {
        /// The M0 posture, kept as the selection reference.
        struct RecomputeScorer {
            state: DecoderState,
            vocab: usize,
        }
        impl BeamScorer for RecomputeScorer {
            fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
                self.state.reset();
                Ok(log_softmax(&self.state.step_last(tokens)?))
            }
            fn vocab_size(&self) -> usize {
                self.vocab
            }
        }

        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;
        for width in [1usize, 2] {
            let cfg = BeamSearchConfig::new(width, 6);

            let mut old = RecomputeScorer {
                state: model.decoder(&enc).unwrap(),
                vocab: model.config().n_vocab,
            };
            let want = beam_search(&mut old, &start, eot, &cfg).unwrap();

            let mut new = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
            let got = beam_search(&mut new, &start, eot, &cfg).unwrap();

            let want_tokens: Vec<_> = want.iter().map(|h| h.tokens.clone()).collect();
            let got_tokens: Vec<_> = got.iter().map(|h| h.tokens.clone()).collect();
            assert_eq!(got_tokens, want_tokens, "width {width}: selection changed");
        }
    }

    // ---- M4-20 (a): word-timestamp wiring (synthetic model, structural) -----

    /// A model with **no** alignment heads reports `Ok(None)` from
    /// `align_words`, so `beam_search` with `word_timestamps` raises the
    /// explicit FR-EX-08 error — never a silent no-op.
    #[test]
    fn no_alignment_heads_makes_word_timestamps_explicit_error() {
        let model = tiny_model(2); // tiny_cfg → empty alignment_heads
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
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

        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
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
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
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
        let mut scorer =
            WhisperBeamScorer::with_tokenizer(Arc::clone(&model), &enc, &tok, enc.n_ctx).unwrap();

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

    /// The DTW row window must start at the **first emission row** (query
    /// position `n_prefix - 1`, whose input is the last forced-prefix token)
    /// and include the **eot emission row** as the terminal bound — openai
    /// timing.py:215 `matrix[len(tokenizer.sot_sequence) : -1]` keeps the
    /// `n_text + 1` rows emitting `[c_0 .. c_{n-1}, eot]`. The reference here
    /// recomputes the alignment from the same captured weights with those row
    /// indices hardcoded for this fixture (`n_prefix = 1`, rows {0,1,2,3});
    /// the scorer must reproduce it exactly. Regression pin for the campaign-2
    /// off-by-one (every word start landed at the previous word's end).
    #[test]
    fn alignment_rows_start_at_first_emission_row_with_eot_bound() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        // prefix (len 1) + content [2, 1, 2] + trailing eot.
        let tokens = [1u32, 2, 1, 2, 0];
        let cfg = model.config().clone();
        let n_head = cfg.n_text_head;
        let t = tokens.len();
        let n_ctx = 4usize;

        // Reference: capture, then stack the alignment heads over the
        // explicitly-indexed emission rows {0, 1, 2, 3} (= n_prefix-1 .. t-1),
        // renormalizing each kept row over the kept columns (timing.py:208-209
        // restricts columns before the softmax; over the model's softmaxed
        // probabilities that is a per-row renormalization).
        let mut st = model.decoder(&enc).unwrap();
        let captured = st.cross_attention_weights(&tokens).unwrap();
        let rows = [0usize, 1, 2, 3];
        let n_rows = rows.len();
        let n_sel = cfg.alignment_heads.len();
        let mut weights = vec![0.0f32; n_sel * n_rows * n_ctx];
        for (s, &(l, h)) in cfg.alignment_heads.iter().enumerate() {
            for (ri, &row) in rows.iter().enumerate() {
                let src = ((l * n_head + h) * t + row) * n_ctx;
                let dst = (s * n_rows + ri) * n_ctx;
                let mut sum = 0.0f64;
                for j in 0..n_ctx {
                    sum += captured[src + j] as f64;
                }
                for j in 0..n_ctx {
                    weights[dst + j] = (captured[src + j] as f64 / sum) as f32;
                }
            }
        }
        let attn = CrossAttention {
            weights,
            n_head: n_sel,
            n_text: n_rows,
            n_audio: n_ctx,
        };
        let dt = WHISPER_CHUNK_SECONDS / cfg.n_audio_ctx as f32;
        let params = AlignmentParams {
            median_filter_width: 7,
            audio_time_per_token: dt,
        };
        // 4 entries: content starts [0..3] + the eot-row arrival [3].
        let ref_times = token_alignment(&attn, &params).unwrap();

        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let got = scorer.align_words(&tokens).unwrap().unwrap();
        assert_eq!(got.len(), 3, "one timing per content token: {got:?}");
        for (i, w) in got.iter().enumerate() {
            assert_eq!(
                w.start, ref_times[i],
                "content token {i} start must be its OWN emission-row arrival \
                 (openai timing.py:215 row window), got {got:?} want {ref_times:?}"
            );
            let want_end = ref_times[i + 1].max(ref_times[i]);
            assert_eq!(
                w.end, want_end,
                "content token {i} end must be the next emission-row arrival \
                 (eot row bounds the last), got {got:?} want {ref_times:?}"
            );
        }
    }

    /// [`valid_audio_positions`] mirrors openai `num_frames // 2` exactly
    /// (transcribe.py:283 `segment_size = min(N_FRAMES, content_frames)` with
    /// `content_frames = len(audio) // HOP_LENGTH`, then timing.py:208 `// 2`).
    #[test]
    fn valid_audio_positions_mirrors_openai_num_frames_over_two() {
        use crate::whisper::mel::{HOP, N_FRAMES, N_SAMPLES};
        // 11.0 s at 16 kHz (jfk): 1100 mel frames → 550 audio positions.
        assert_eq!(valid_audio_positions(176_000), 550);
        // ≥ 30 s clips cap at the full window (N_FRAMES / 2 = n_audio_ctx).
        assert_eq!(valid_audio_positions(N_SAMPLES), N_FRAMES / 2);
        assert_eq!(valid_audio_positions(10 * N_SAMPLES), N_FRAMES / 2);
        // Floor semantics on both divisions.
        assert_eq!(valid_audio_positions(HOP * 3 - 1), 1);
        assert_eq!(valid_audio_positions(0), 0);
    }

    /// With the columns restricted to the clip's valid frames, every aligned
    /// time must sit within the valid region `[0, (n_valid - 1) * dt]` — the
    /// zero-padded tail of the 30 s window is unreachable (campaign-2: an
    /// interior word overshot the true audio length, and DTW was forced to
    /// span all 1500 padded columns).
    #[test]
    fn restricted_valid_frames_bound_all_times() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let dt = WHISPER_CHUNK_SECONDS / model.config().n_audio_ctx as f32;
        // 2 valid of 4 window frames → max reachable time = 1 * dt.
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, 2).unwrap();
        let got = scorer.align_words(&[1u32, 2, 1, 2, 0]).unwrap().unwrap();
        assert_eq!(got.len(), 3);
        for w in &got {
            assert!(
                w.start <= dt + 1e-6 && w.end <= dt + 1e-6,
                "times must be within the valid-frame region [0, {dt}]: {w:?}"
            );
        }
    }

    /// A clip too short for even one valid audio position is an explicit
    /// error (FR-EX-08), never a fabricated alignment over pure padding.
    #[test]
    fn zero_valid_frames_is_explicit_error() {
        let model = tiny_model_aligned(1);
        let enc = tiny_encoder(model.config().d_model, 4);
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, 0).unwrap();
        match scorer.align_words(&[1u32, 2, 0]) {
            Err(VokraError::InvalidArgument(_)) => {}
            other => panic!("expected InvalidArgument for zero valid frames, got {other:?}"),
        }
    }

    /// The final content token's end must be the **eot-emission arrival**
    /// (≤ `(n_frames - 1) * dt`), never the padded-window end `n_ctx * dt` —
    /// the campaign-2 "final word ends at 30.000 s on every clip" pad leak.
    #[test]
    fn per_token_last_end_is_eot_arrival_not_padded_window_end() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let dt = WHISPER_CHUNK_SECONDS / model.config().n_audio_ctx as f32;
        let window = 4.0 * dt;
        let mut scorer = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let got = scorer.align_words(&[1u32, 2, 1, 2, 0]).unwrap().unwrap();
        let last = got.last().expect("non-empty alignment");
        assert!(
            last.end <= window - dt + 1e-6,
            "last end {} must be bounded by the eot-row arrival (max frame \
             index n_frames-1 → {}), not the padded-window end {window}",
            last.end,
            window - dt
        );
    }

    /// With a tokenizer, a bare trailing punctuation "word" must fold into the
    /// preceding word (openai timing.py:245 `merge_punctuations`, append set),
    /// keeping the absorbing word's own start/end and extending its token
    /// span. Content = [" hi", "."] → ONE word covering both tokens.
    #[test]
    fn tokenizer_path_merges_appended_punctuation() {
        let model = tiny_model_aligned(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let tok = tiny_tokenizer(&[
            (1, b""),    // id 0: special (eot)
            (0, b"."),   // id 1: bare punctuation, no leading space
            (0, b" hi"), // id 2: leading space → word start
        ]);
        // prefix (len 1) + content [2, 1] + trailing eot.
        let tokens = [1u32, 2, 1, 0];

        let mut merged =
            WhisperBeamScorer::with_tokenizer(Arc::clone(&model), &enc, &tok, enc.n_ctx).unwrap();
        let got = merged.align_words(&tokens).unwrap().unwrap();
        assert_eq!(
            got.len(),
            1,
            "trailing '.' must fold into the preceding word: {got:?}"
        );
        assert_eq!(got[0].token_start, 1, "span starts at the word token");
        assert_eq!(got[0].token_end, 3, "span extends over the merged '.'");

        // The merged word keeps the WORD's own start/end (upstream merge does
        // not touch times; the punctuation's arrival is dropped). Cross-check
        // against the per-token alignment: the word covered content token 0
        // before the merge, so its end is content token 0's end.
        let mut per_token = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();
        let pt = per_token.align_words(&tokens).unwrap().unwrap();
        assert_eq!(got[0].start, pt[0].start, "merge must not move the start");
        assert_eq!(got[0].end, pt[0].end, "merge must not extend the end");
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
            WhisperBeamScorer::with_tokenizer(Arc::clone(&model), &enc, &tok, enc.n_ctx).unwrap();
        let mut without = WhisperBeamScorer::new(Arc::clone(&model), &enc, enc.n_ctx).unwrap();

        let merged = with_tok.align_words(&tokens).unwrap().unwrap();
        let per_token = without.align_words(&tokens).unwrap().unwrap();
        assert_eq!(
            merged, per_token,
            "one-word-per-token vocab must equal the per-token alignment",
        );
    }
}
