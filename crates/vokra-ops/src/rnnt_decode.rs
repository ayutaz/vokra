//! # rnnt_decode — RNN-T / TDT decoding primitive
//!
//! SoTA plan Phase 2 ASR primitive: greedy + beam + TDT (Time-Duration
//! Transducer) decoding for `parakeet-rnnt-1.1b` and the `parakeet-tdt` v2 /
//! v3 / 1.1b family (CC-BY-4.0).
//!
//! ## Upstream references
//!
//! Ported / cross-referenced against these NVIDIA NeMo modules
//! (Apache-2.0), citing the classical (non label-looping) decoders — label
//! looping (~1500× RTFx) is a deferred follow-up:
//!
//! - `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py`
//!   — `GreedyRNNTInfer._greedy_decode()` (~L394-500) for the greedy outer /
//!   inner loop shape; `_greedy_decode_blank_as_pad_loop_frames()` (~L1097)
//!   for the batched frame-loop view (we serialize the batch axis here).
//! - `nemo/collections/asr/parts/submodules/tdt_beam_decoding.py`
//!   — `default_beam_search()` (L369-467) for the TDT beam expansion and
//!   `merge_duplicate_hypotheses()` (L476-497) for the recombination rule.
//!
//! ## Design constraints
//!
//! * **No prediction / joint network here.** This is a decoding primitive:
//!   the caller supplies the pre-materialized joint frame per timestep as
//!   `encoder_out` (i.e. one log-prob vector per time index, laid out
//!   row-major). Consumers with a real prediction network (Parakeet) drive
//!   this by materializing joint outputs upstream, exactly the shape a
//!   stateless-per-frame joint (or a label-looping cache flush) already
//!   produces.
//! * **Fail-loud on shape / attribute mismatches** (FR-EX-08). No silent
//!   truncation, no default vocab, no NaN pass-through — every degenerate
//!   input becomes an explicit [`VokraError::InvalidArgument`].
//! * **Zero third-party deps** (NFR-DS-02): the module is scalar Rust with
//!   `f32`, no `unsafe`, no SIMD intrinsics.
//!
//! ## Input layout
//!
//! Row-major `f32` slice, one contiguous logit vector per timestep:
//!
//! | Decoder            | Stride per timestep       | Total floats                |
//! |--------------------|---------------------------|-----------------------------|
//! | `Greedy`, `Beam`   | `vocab_size + 1`          | `T · (V + 1)`               |
//! | `Tdt`              | `vocab_size + 1 + D`      | `T · (V + 1 + D)`           |
//!
//! Where `V = vocab_size`, `T = num_timesteps`, and `D =
//! duration_bins.len()`. The vocabulary occupies indices `0..V` and the
//! blank symbol lives at `blank_id ∈ [0, V]`. For TDT the trailing `D`
//! floats are the duration head's log-probabilities in the order the
//! caller listed the bins.
//!
//! The values are treated as (log-)probabilities: the primitive **does not
//! log-softmax** them. Callers pass whatever their joint head produced;
//! scoring is `+=` accumulation across the emitted path.

use vokra_core::{Result, VokraError};

/// Which decoder to run.
///
/// * [`Self::Greedy`] — classical RNN-T greedy: for every timestep, emit the
///   argmax over `V + 1`. If the argmax is `blank_id`, no token is emitted
///   for this frame. Corresponds to the outer for-loop of NeMo's
///   `GreedyRNNTInfer._greedy_decode()` (~L394-500) with
///   `max_symbols_per_step = 1` (a stateless per-frame joint cannot advance
///   its own logits without a prediction-network state update).
/// * [`Self::Beam`] — classical (frame-synchronous) RNN-T beam search. Every
///   frame, every kept hypothesis expands by every candidate in `V + 1`;
///   the top `beam_size` accumulated scores survive. See NeMo's
///   `default_beam_search()` (L389-461) for the outer loop shape (this
///   primitive drops the duration cartesian product used by TDT).
/// * [`Self::Tdt`] — Time-Duration Transducer greedy: separate argmaxes over
///   the vocabulary head (`V + 1`) and the duration head (`D`) per frame.
///   The chosen duration determines the timestep skip. `duration_bins`
///   lists the bin values in the order the head predicts them (e.g. the
///   Parakeet TDT `[0, 1, 2, 3, 4]` set). A zero duration keeps the frame
///   pointer, matching NeMo's TDT beam
///   `if duration == 0: hyps.append(new_hyp)` branch (L450-451); the
///   [`RnntAttrs::max_symbols_per_step`] cap prevents infinite zero-duration
///   loops (matches NeMo's `min_non_zero_duration_idx` fallback L465).
#[derive(Debug, Clone, PartialEq)]
pub enum RnntDecoderKind {
    /// Frame-synchronous greedy decode (single best hypothesis).
    Greedy,
    /// Frame-synchronous beam search.
    Beam {
        /// Number of hypotheses kept per frame (`>= 1`).
        beam_size: usize,
    },
    /// TDT greedy with a duration head.
    Tdt {
        /// Duration bins in head-output order (e.g. `[0, 1, 2, 3, 4]`).
        /// Must contain at least one entry and at least one non-zero value.
        duration_bins: Vec<u32>,
    },
}

/// Decoder attributes.
///
/// The blank symbol is encoded as an index into the `V + 1` vocab head.
/// Common conventions:
///
/// * NeMo defaults to `blank_id = vocab_size` (blank appended after the
///   vocab), e.g. Parakeet-TDT-1.1b uses `blank_id = 1024` for a 1024-piece
///   BPE vocabulary.
/// * Some checkpoints use `blank_id = 0`. Either is accepted here so long
///   as `blank_id <= vocab_size` (i.e. inside the `V + 1` head).
#[derive(Debug, Clone)]
pub struct RnntAttrs {
    /// Number of encoder frames the joint has been materialized over. Must
    /// match the `T` axis of `encoder_out`.
    pub num_timesteps: usize,
    /// Vocabulary size excluding the blank symbol (`V`; the vocab head is
    /// `V + 1` wide).
    pub vocab_size: usize,
    /// Blank symbol index inside `[0, vocab_size]`.
    pub blank_id: u32,
    /// Zero-duration emission cap for TDT (matches NeMo
    /// `max_symbols_per_step`; default value 10 is the NeMo greedy default).
    /// Ignored for `Greedy` / `Beam`, which are implicit `max = 1`.
    pub max_symbols_per_step: usize,
    /// Which decoder to run.
    pub kind: RnntDecoderKind,
}

impl RnntAttrs {
    /// Constructs greedy attrs with NeMo-style defaults (`blank_id =
    /// vocab_size`, `max_symbols_per_step = 10`).
    pub fn greedy(num_timesteps: usize, vocab_size: usize) -> Self {
        Self {
            num_timesteps,
            vocab_size,
            blank_id: vocab_size as u32,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Greedy,
        }
    }

    /// Constructs beam attrs of the given width, NeMo-style defaults.
    pub fn beam(num_timesteps: usize, vocab_size: usize, beam_size: usize) -> Self {
        Self {
            num_timesteps,
            vocab_size,
            blank_id: vocab_size as u32,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Beam { beam_size },
        }
    }

    /// Constructs TDT attrs, NeMo-style defaults.
    pub fn tdt(num_timesteps: usize, vocab_size: usize, duration_bins: Vec<u32>) -> Self {
        Self {
            num_timesteps,
            vocab_size,
            blank_id: vocab_size as u32,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Tdt { duration_bins },
        }
    }
}

/// One decode hypothesis.
///
/// Shape mirrors NeMo's `Hypothesis` (`nemo/collections/asr/parts/utils/
/// rnnt_utils.py`): `y_sequence` → [`Self::tokens`], `timestamp` →
/// [`Self::timestamps`], `score` → [`Self::score`], `last_frame` →
/// [`Self::last_frame`]. The prediction-network `dec_state` field is
/// intentionally absent — this primitive is stateless.
#[derive(Debug, Clone, PartialEq)]
pub struct RnntHypothesis {
    /// Emitted non-blank tokens, in emission order.
    pub tokens: Vec<u32>,
    /// Frame index at which each of `tokens` was emitted. Same length as
    /// `tokens`.
    pub timestamps: Vec<usize>,
    /// Cumulative log-probability along the emitted path (sum of every
    /// per-frame vocab argmax value the decoder took, plus every duration
    /// argmax for TDT). Not length-normalized — the caller applies any
    /// normalization it wants downstream.
    pub score: f32,
    /// One-past-the-last frame consumed. Equals [`RnntAttrs::num_timesteps`]
    /// when the decoder walked the full input.
    pub last_frame: usize,
}

/// Runs the requested RNN-T / TDT decoder over `encoder_out`.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
///
/// * `num_timesteps == 0` or `vocab_size == 0`;
/// * `blank_id > vocab_size` (i.e. outside the `V + 1` vocab head);
/// * `max_symbols_per_step == 0`;
/// * `RnntDecoderKind::Beam { beam_size: 0 }`;
/// * `RnntDecoderKind::Tdt` with an empty `duration_bins` slice, or one
///   whose entries are all `0` (would deadlock the decoder loop);
/// * `encoder_out.len()` mismatching the expected `T · stride` size;
/// * any `NaN` in `encoder_out` (fail-loud vs. `partial_cmp` returning
///   `None` during argmax — FR-EX-08).
///
/// # Returns
///
/// * `Greedy` / `Tdt`: exactly one [`RnntHypothesis`].
/// * `Beam`: up to `beam_size` hypotheses, sorted best-first by
///   accumulated `score` (unnormalized — matches NeMo
///   `sort_nbest(score_norm=False)` at
///   `nemo/collections/asr/parts/submodules/tdt_beam_decoding.py:558-560`).
pub fn rnnt_decode(encoder_out: &[f32], attrs: &RnntAttrs) -> Result<Vec<RnntHypothesis>> {
    validate_attrs(attrs)?;

    let stride = expected_stride(attrs);
    let expected_len = attrs.num_timesteps.checked_mul(stride).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "rnnt_decode: num_timesteps ({}) * stride ({}) overflows usize",
            attrs.num_timesteps, stride
        ))
    })?;
    if encoder_out.len() != expected_len {
        return Err(VokraError::InvalidArgument(format!(
            "rnnt_decode: encoder_out length {} does not match expected T*stride = {} \
             (T={}, stride={})",
            encoder_out.len(),
            expected_len,
            attrs.num_timesteps,
            stride
        )));
    }
    if let Some(idx) = encoder_out.iter().position(|v| v.is_nan()) {
        return Err(VokraError::InvalidArgument(format!(
            "rnnt_decode: encoder_out contains NaN at index {idx}"
        )));
    }

    match &attrs.kind {
        RnntDecoderKind::Greedy => Ok(vec![decode_greedy(encoder_out, attrs, stride)]),
        RnntDecoderKind::Beam { beam_size } => decode_beam(encoder_out, attrs, stride, *beam_size),
        RnntDecoderKind::Tdt { duration_bins } => {
            Ok(vec![decode_tdt(encoder_out, attrs, stride, duration_bins)?])
        }
    }
}

// ---- helpers ---------------------------------------------------------------

/// Expected stride (per-timestep float count) implied by `attrs.kind`.
fn expected_stride(attrs: &RnntAttrs) -> usize {
    let vocab_head = attrs.vocab_size + 1;
    match &attrs.kind {
        RnntDecoderKind::Greedy | RnntDecoderKind::Beam { .. } => vocab_head,
        RnntDecoderKind::Tdt { duration_bins } => vocab_head + duration_bins.len(),
    }
}

/// Structural validation of the attrs before touching `encoder_out`.
fn validate_attrs(attrs: &RnntAttrs) -> Result<()> {
    if attrs.num_timesteps == 0 {
        return Err(VokraError::InvalidArgument(
            "rnnt_decode: num_timesteps must be >= 1".into(),
        ));
    }
    if attrs.vocab_size == 0 {
        return Err(VokraError::InvalidArgument(
            "rnnt_decode: vocab_size must be >= 1".into(),
        ));
    }
    if (attrs.blank_id as usize) > attrs.vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "rnnt_decode: blank_id ({}) must be <= vocab_size ({}) so it fits the V+1 head",
            attrs.blank_id, attrs.vocab_size
        )));
    }
    if attrs.max_symbols_per_step == 0 {
        return Err(VokraError::InvalidArgument(
            "rnnt_decode: max_symbols_per_step must be >= 1".into(),
        ));
    }
    match &attrs.kind {
        RnntDecoderKind::Greedy => Ok(()),
        RnntDecoderKind::Beam { beam_size } => {
            if *beam_size == 0 {
                return Err(VokraError::InvalidArgument(
                    "rnnt_decode: Beam { beam_size } must be >= 1".into(),
                ));
            }
            Ok(())
        }
        RnntDecoderKind::Tdt { duration_bins } => {
            if duration_bins.is_empty() {
                return Err(VokraError::InvalidArgument(
                    "rnnt_decode: Tdt requires at least one duration bin".into(),
                ));
            }
            if duration_bins.iter().all(|&d| d == 0) {
                return Err(VokraError::InvalidArgument(
                    "rnnt_decode: Tdt duration_bins must contain at least one non-zero value \
                     (all-zero bins would deadlock the frame pointer)"
                        .into(),
                ));
            }
            Ok(())
        }
    }
}

/// Argmax over a slice; NaNs are already filtered out by the caller so
/// `partial_cmp` always returns `Some`. Returns `(index, value)`.
///
/// A separate helper (rather than `iter().enumerate().max_by(...)`) keeps
/// the tie-break rule explicit: on equal values, the **lower index wins**,
/// matching PyTorch `argmax` (the reference `k = logp.max(0)` in
/// `rnnt_greedy_decoding.py` L436) and NumPy `argmax`.
fn argmax_f32(values: &[f32]) -> (usize, f32) {
    debug_assert!(!values.is_empty(), "argmax on empty slice");
    let mut best_idx = 0;
    let mut best_val = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_val {
            best_idx = i;
            best_val = v;
        }
    }
    (best_idx, best_val)
}

/// Greedy decode (single hypothesis).
///
/// Frame-synchronous, one emission per frame: at each `t`, the vocab-head
/// argmax is either the blank (no emission) or a token index that is
/// pushed onto the hypothesis. Score accumulates the picked log-prob per
/// frame (blank or non-blank alike — matches NeMo greedy summing every
/// frame's `v` regardless of blank).
fn decode_greedy(encoder_out: &[f32], attrs: &RnntAttrs, stride: usize) -> RnntHypothesis {
    let mut tokens = Vec::new();
    let mut timestamps = Vec::new();
    let mut score = 0.0_f32;
    for t in 0..attrs.num_timesteps {
        let frame = &encoder_out[t * stride..(t + 1) * stride];
        let (arg, val) = argmax_f32(frame);
        score += val;
        if (arg as u32) != attrs.blank_id {
            tokens.push(arg as u32);
            timestamps.push(t);
        }
    }
    RnntHypothesis {
        tokens,
        timestamps,
        score,
        last_frame: attrs.num_timesteps,
    }
}

/// Frame-synchronous beam search (top-K over the flat set of expansions).
///
/// At each frame `t`, every kept hypothesis expands by every candidate in
/// the `V + 1` head; the top `beam_size` accumulated scores survive. The
/// global-top-K rule matches HuggingFace `beam_search` semantics (already
/// re-used at `vokra_core::decode::beam_search`) and avoids the per-hyp
/// pre-top-K bias.
fn decode_beam(
    encoder_out: &[f32],
    attrs: &RnntAttrs,
    stride: usize,
    beam_size: usize,
) -> Result<Vec<RnntHypothesis>> {
    // beam_size==0 is rejected by validate_attrs; guarded here for clarity.
    debug_assert!(beam_size >= 1);
    let mut beams: Vec<RnntHypothesis> = vec![RnntHypothesis {
        tokens: Vec::new(),
        timestamps: Vec::new(),
        score: 0.0,
        last_frame: 0,
    }];
    for t in 0..attrs.num_timesteps {
        let frame = &encoder_out[t * stride..(t + 1) * stride];
        let mut cands: Vec<RnntHypothesis> = Vec::with_capacity(beams.len() * stride);
        for hyp in &beams {
            for (k, &lp) in frame.iter().enumerate() {
                let mut new_tokens = hyp.tokens.clone();
                let mut new_timestamps = hyp.timestamps.clone();
                if (k as u32) != attrs.blank_id {
                    new_tokens.push(k as u32);
                    new_timestamps.push(t);
                }
                cands.push(RnntHypothesis {
                    tokens: new_tokens,
                    timestamps: new_timestamps,
                    score: hyp.score + lp,
                    last_frame: t + 1,
                });
            }
        }
        // Sort candidates by descending score. NaN was already rejected up
        // front, so `partial_cmp` is always `Some(_)`. `partial_cmp` on
        // `+inf` / `-inf` is total, matching NeMo's `max()` semantics.
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .expect("nan pre-filtered at rnnt_decode entry")
        });
        cands.truncate(beam_size);
        beams = cands;
    }
    Ok(beams)
}

/// TDT greedy: joint vocab + duration head, per NeMo tdt_beam_decoding.
///
/// Per frame the decoder takes both argmaxes (vocab over `V + 1`, duration
/// over `D`). Score accumulates `vocab_lp + duration_lp` (the two heads
/// contribute independently — matches the NeMo `cartesian_prod(...).sum`
/// on L419-420). The chosen duration bin decides how far to skip:
///
/// * `duration > 0` — advance `t += duration`, reset the zero-duration
///   counter (matches TDT beam's `kept_hyps.append(new_hyp)` at L453);
/// * `duration == 0` — stay on the same frame **only** if we just emitted a
///   non-blank (allows multi-emit RNN-T style up to `max_symbols_per_step`).
///   On a blank frame the decoder force-advances by one to avoid infinite
///   loops (mirrors NeMo's `min_non_zero_duration_idx` fallback L462-467).
fn decode_tdt(
    encoder_out: &[f32],
    attrs: &RnntAttrs,
    stride: usize,
    duration_bins: &[u32],
) -> Result<RnntHypothesis> {
    let vocab_head = attrs.vocab_size + 1;
    let mut tokens = Vec::new();
    let mut timestamps = Vec::new();
    let mut score = 0.0_f32;
    let mut t: usize = 0;
    let mut zero_dur_streak: usize = 0;
    while t < attrs.num_timesteps {
        let frame = &encoder_out[t * stride..(t + 1) * stride];
        let (vocab_arg, vocab_val) = argmax_f32(&frame[..vocab_head]);
        let (dur_arg, dur_val) = argmax_f32(&frame[vocab_head..]);
        score += vocab_val + dur_val;

        let is_blank = (vocab_arg as u32) == attrs.blank_id;
        if !is_blank {
            tokens.push(vocab_arg as u32);
            timestamps.push(t);
        }

        // Compute the effective duration skip. `dur_arg` is a valid index
        // (argmax on non-empty duration head).
        let raw_dur = duration_bins[dur_arg];
        let effective_step: usize = if raw_dur > 0 {
            zero_dur_streak = 0;
            raw_dur as usize
        } else if is_blank {
            // Blank with zero duration would deadlock — force +1.
            zero_dur_streak = 0;
            1
        } else {
            // Non-blank with zero duration: multi-emit up to the cap.
            zero_dur_streak = zero_dur_streak.saturating_add(1);
            if zero_dur_streak >= attrs.max_symbols_per_step {
                zero_dur_streak = 0;
                1
            } else {
                0
            }
        };
        // `t += 0` is legal (multi-emit); `checked_add` guards against
        // pathological `duration_bins` such as `[u32::MAX]`.
        t = t.checked_add(effective_step).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "rnnt_decode: TDT step overflow at t={t}, dur={effective_step}"
            ))
        })?;
    }
    let last_frame = attrs.num_timesteps;
    Ok(RnntHypothesis {
        tokens,
        timestamps,
        score,
        last_frame,
    })
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A tiny hand-crafted logits builder: vocab_size = 3, blank = 3
    // (NeMo-style trailing blank). Layout `[T, 4]` for Greedy/Beam and
    // `[T, 4 + D]` for TDT, both row-major.
    fn frame(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    fn flatten(frames: &[Vec<f32>]) -> Vec<f32> {
        frames.iter().flatten().copied().collect()
    }

    // ---- Greedy happy path ------------------------------------------------

    #[test]
    fn greedy_emits_argmax_and_skips_blank() {
        // Frames: t=0 -> token 1 (val 2.0), t=1 -> blank (val 5.0),
        // t=2 -> token 0 (val 3.0). Score = 2.0 + 5.0 + 3.0 = 10.0.
        let enc = flatten(&[
            frame(&[0.0, 2.0, 0.0, 0.5]),
            frame(&[0.0, 0.0, 0.0, 5.0]),
            frame(&[3.0, 0.0, 0.0, 0.0]),
        ]);
        let attrs = RnntAttrs::greedy(3, 3);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tokens, vec![1, 0]);
        assert_eq!(out[0].timestamps, vec![0, 2]);
        assert!((out[0].score - 10.0).abs() < 1e-6);
        assert_eq!(out[0].last_frame, 3);
    }

    #[test]
    fn greedy_all_blank_returns_empty_sequence() {
        // Every frame's argmax is the blank (index 3).
        let enc = flatten(&[frame(&[0.0, 0.0, 0.0, 1.0]), frame(&[0.0, 0.0, 0.0, 1.0])]);
        let attrs = RnntAttrs::greedy(2, 3);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert!(out[0].tokens.is_empty());
        assert!(out[0].timestamps.is_empty());
        assert_eq!(out[0].last_frame, 2);
    }

    #[test]
    fn greedy_argmax_lower_index_wins_on_ties() {
        // Two equal max values at indices 0 and 2. NumPy / PyTorch pick
        // the lower index; our helper mirrors that (strict `>` update).
        let enc = flatten(&[frame(&[1.0, 0.0, 1.0, 0.0])]);
        let attrs = RnntAttrs::greedy(1, 3);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert_eq!(out[0].tokens, vec![0]);
    }

    // ---- Beam happy path --------------------------------------------------

    #[test]
    fn beam_size_one_matches_greedy() {
        // A beam of width 1 is greedy: same tokens, same score.
        let enc = flatten(&[
            frame(&[0.0, 2.0, 0.0, 0.5]),
            frame(&[0.0, 0.0, 0.0, 5.0]),
            frame(&[3.0, 0.0, 0.0, 0.0]),
        ]);
        let g = &rnnt_decode(&enc, &RnntAttrs::greedy(3, 3)).unwrap()[0];
        let b = &rnnt_decode(&enc, &RnntAttrs::beam(3, 3, 1)).unwrap()[0];
        assert_eq!(b.tokens, g.tokens);
        assert_eq!(b.timestamps, g.timestamps);
        assert!((b.score - g.score).abs() < 1e-6);
    }

    #[test]
    fn beam_returns_top_k_sorted_by_score() {
        // T=2, V=3 (head width 4, blank=3). Beam=3 gets at most 3 hyps.
        // Craft distinct per-frame argmaxes so the top hypotheses are
        // easy to enumerate by hand.
        // Frame 0 log-probs (ln): pick token 0 dominant.
        // Frame 1: pick token 1 dominant. Best two paths keep 0-then-1 and
        // 0-then-blank in that order.
        let enc = flatten(&[frame(&[3.0, 0.5, 0.0, 0.1]), frame(&[0.0, 4.0, 0.5, 1.0])]);
        let out = rnnt_decode(&enc, &RnntAttrs::beam(2, 3, 3)).unwrap();
        assert!(out.len() <= 3);
        assert!(out.len() >= 2);
        // Highest-scoring hypothesis: [0, 1] with score 3.0 + 4.0 = 7.0.
        assert_eq!(out[0].tokens, vec![0, 1]);
        assert!((out[0].score - 7.0).abs() < 1e-6);
        // Scores must be non-increasing.
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn beam_records_frame_indices_correctly() {
        // Emitted tokens carry the frame they came from.
        let enc = flatten(&[
            frame(&[0.0, 3.0, 0.0, 0.0]), // token 1 at t=0
            frame(&[0.0, 0.0, 0.0, 5.0]), // blank at t=1
            frame(&[2.0, 0.0, 0.0, 0.0]), // token 0 at t=2
        ]);
        let out = rnnt_decode(&enc, &RnntAttrs::beam(3, 3, 1)).unwrap();
        assert_eq!(out[0].tokens, vec![1, 0]);
        assert_eq!(out[0].timestamps, vec![0, 2]);
    }

    // ---- TDT happy path ---------------------------------------------------

    #[test]
    fn tdt_uses_duration_head_to_skip_frames() {
        // TDT layout: vocab head (V+1 = 4) + duration head (D = 3, bins
        // = [0, 1, 2]).
        // t=0: vocab argmax = 1 (val 4.0), dur argmax = 2 (bin=2, val 2.0).
        //   emit token 1, advance to t=2.
        // t=2: vocab argmax = 2 (val 3.0), dur argmax = 1 (bin=1, val 1.0).
        //   emit token 2, advance to t=3. done (num_timesteps=3).
        let bins = vec![0u32, 1, 2];
        let f0 = frame(&[0.0, 4.0, 0.0, 0.0, /*durs*/ 0.0, 1.0, 2.0]);
        let f1 = frame(&[9.0, 9.0, 9.0, 9.0, /*durs*/ 0.0, 0.0, 0.0]); // skipped
        let f2 = frame(&[0.0, 0.0, 3.0, 0.0, /*durs*/ 0.0, 1.0, 0.0]);
        let enc = flatten(&[f0, f1, f2]);

        let attrs = RnntAttrs::tdt(3, 3, bins);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tokens, vec![1, 2]);
        assert_eq!(out[0].timestamps, vec![0, 2]);
        // Score: (4+2) + (3+1) = 10.0
        assert!((out[0].score - 10.0).abs() < 1e-6);
        assert_eq!(out[0].last_frame, 3);
    }

    #[test]
    fn tdt_zero_duration_multi_emit_capped_by_max_symbols_per_step() {
        // Duration argmax is 0 (bin index 0 = value 0) forever. Vocab
        // argmax is token 0 (non-blank). Without the cap this would spin
        // on t=0. With max_symbols_per_step = 2, on the 2nd zero-duration
        // frame we force-advance.
        let bins = vec![0u32, 1];
        // stride = 4 + 2 = 6
        let f = frame(&[5.0, 0.0, 0.0, 0.0, /*durs*/ 3.0, 0.0]);
        let enc = flatten(&[f.clone(), f.clone()]); // 2 timesteps
        let mut attrs = RnntAttrs::tdt(2, 3, bins);
        attrs.max_symbols_per_step = 2;
        let out = rnnt_decode(&enc, &attrs).unwrap();
        // First emission at t=0 (streak=1), second at t=0 (streak=2 hits
        // cap, force +1), third at t=1 (streak=1 again), fourth at t=1
        // (streak=2 hits cap, force +1). t reaches num_timesteps=2, exit.
        assert_eq!(out[0].tokens, vec![0, 0, 0, 0]);
        assert_eq!(out[0].timestamps, vec![0, 0, 1, 1]);
        assert_eq!(out[0].last_frame, 2);
    }

    #[test]
    fn tdt_blank_with_zero_duration_force_advances() {
        // A blank whose selected duration bin is `0` would deadlock — the
        // decoder forces +1 to match NeMo's `min_non_zero_duration_idx`
        // fallback. We use bins `[0, 1]` (validator accepts, has non-zero)
        // and steer the duration argmax onto bin index 0 (value 0) via
        // higher logit; vocab argmax lands on the blank (index 3).
        let bins = vec![0u32, 1];
        // stride = 4 + 2 = 6. vocab argmax = blank (index 3, val 2.0);
        // duration argmax = bin index 0 (val 3.0 vs 0.0) → raw_dur = 0.
        let f = frame(&[0.0, 0.0, 0.0, 2.0, /*durs*/ 3.0, 0.0]);
        let enc = flatten(&[f.clone(), f]); // 2 frames
        let attrs = RnntAttrs::tdt(2, 3, bins);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        // No emissions (both blanks), forced +1 each frame, exit cleanly.
        assert!(out[0].tokens.is_empty());
        assert_eq!(out[0].last_frame, 2);
    }

    // ---- Validation errors ------------------------------------------------

    #[test]
    fn empty_encoder_out_or_num_timesteps_is_error() {
        // num_timesteps = 0 is an explicit error, regardless of encoder_out length.
        let attrs = RnntAttrs {
            num_timesteps: 0,
            vocab_size: 3,
            blank_id: 3,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Greedy,
        };
        assert!(matches!(
            rnnt_decode(&[], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));

        // num_timesteps > 0 but encoder_out empty → shape mismatch.
        let attrs = RnntAttrs::greedy(2, 3);
        assert!(matches!(
            rnnt_decode(&[], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn shape_mismatch_is_reported_explicitly() {
        // T=2 needs 8 floats (2 * 4); we pass 7.
        let attrs = RnntAttrs::greedy(2, 3);
        let enc = vec![0.0f32; 7];
        let err = rnnt_decode(&enc, &attrs).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("length 7"), "msg: {msg}");
                assert!(msg.contains("expected"), "msg: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn zero_vocab_size_is_error() {
        let attrs = RnntAttrs {
            num_timesteps: 1,
            vocab_size: 0,
            blank_id: 0,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Greedy,
        };
        assert!(matches!(
            rnnt_decode(&[], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn blank_id_out_of_range_is_error() {
        // vocab_size = 3 → blank_id must be in [0, 3]. 4 is out.
        let attrs = RnntAttrs {
            num_timesteps: 1,
            vocab_size: 3,
            blank_id: 4,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Greedy,
        };
        assert!(matches!(
            rnnt_decode(&[0.0; 4], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zero_max_symbols_per_step_is_error() {
        let attrs = RnntAttrs {
            num_timesteps: 1,
            vocab_size: 3,
            blank_id: 3,
            max_symbols_per_step: 0,
            kind: RnntDecoderKind::Greedy,
        };
        assert!(matches!(
            rnnt_decode(&[0.0; 4], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn zero_beam_size_is_error() {
        let attrs = RnntAttrs {
            num_timesteps: 1,
            vocab_size: 3,
            blank_id: 3,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Beam { beam_size: 0 },
        };
        assert!(matches!(
            rnnt_decode(&[0.0; 4], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tdt_empty_duration_bins_is_error() {
        let attrs = RnntAttrs::tdt(1, 3, vec![]);
        // Zero duration bins → stride = V+1 = 4, but validation catches
        // it before shape check.
        assert!(matches!(
            rnnt_decode(&[0.0; 4], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn tdt_all_zero_duration_bins_is_error() {
        // All-zero durations would deadlock — force an explicit error.
        let attrs = RnntAttrs::tdt(1, 3, vec![0, 0, 0]);
        assert!(matches!(
            rnnt_decode(&[0.0; 7], &attrs),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn nan_in_encoder_out_is_fail_loud() {
        let attrs = RnntAttrs::greedy(2, 3);
        let mut enc = vec![0.0f32; 8];
        enc[5] = f32::NAN;
        let err = rnnt_decode(&enc, &attrs).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(msg.contains("NaN"), "msg: {msg}");
                assert!(msg.contains("index 5"), "msg: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    // ---- Numerical edge cases ---------------------------------------------

    #[test]
    fn neg_inf_logits_are_valid_log_probs() {
        // A `-inf` log-prob (i.e. probability 0) is legitimate — a
        // completely suppressed token. It must not trigger the NaN filter,
        // and the greedy argmax must still pick some other index.
        let enc = flatten(&[frame(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY, -1.0])]);
        let attrs = RnntAttrs::greedy(1, 3);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert_eq!(out[0].tokens, vec![1]);
    }

    #[test]
    fn blank_id_at_zero_works_end_to_end() {
        // Some checkpoints use `blank_id = 0` (blank at index 0, vocab at
        // 1..V+1). Verify the decoder respects the placement.
        let enc = flatten(&[
            frame(&[3.0, 0.0, 0.0, 0.0]), // blank at t=0
            frame(&[0.0, 2.0, 0.0, 0.0]), // token 1 at t=1
        ]);
        let attrs = RnntAttrs {
            num_timesteps: 2,
            vocab_size: 3,
            blank_id: 0,
            max_symbols_per_step: 10,
            kind: RnntDecoderKind::Greedy,
        };
        let out = rnnt_decode(&enc, &attrs).unwrap();
        assert_eq!(out[0].tokens, vec![1]);
        assert_eq!(out[0].timestamps, vec![1]);
    }

    #[test]
    fn tdt_duration_overshoots_num_timesteps_terminates() {
        // duration bin = 100 overshoots T=2. The while loop must exit
        // cleanly on the very next check.
        let bins = vec![100u32];
        // stride = 4 + 1 = 5
        let f = frame(&[0.0, 2.0, 0.0, 0.0, /*durs*/ 1.0]);
        let enc = flatten(&[f, frame(&[0.0; 5])]);
        let attrs = RnntAttrs::tdt(2, 3, bins);
        let out = rnnt_decode(&enc, &attrs).unwrap();
        // Emit at t=0, jump to t=100, loop exits.
        assert_eq!(out[0].tokens, vec![1]);
        assert_eq!(out[0].timestamps, vec![0]);
        assert_eq!(out[0].last_frame, 2);
    }

    // ---- Attribute constructors ------------------------------------------

    #[test]
    fn attr_constructors_default_blank_and_cap() {
        let g = RnntAttrs::greedy(4, 128);
        assert_eq!(g.blank_id, 128);
        assert_eq!(g.max_symbols_per_step, 10);
        assert!(matches!(g.kind, RnntDecoderKind::Greedy));

        let b = RnntAttrs::beam(4, 128, 8);
        assert!(matches!(b.kind, RnntDecoderKind::Beam { beam_size: 8 }));

        let t = RnntAttrs::tdt(4, 128, vec![0, 1, 2, 3, 4]);
        assert!(matches!(t.kind, RnntDecoderKind::Tdt { .. }));
    }

    #[test]
    fn hypothesis_last_frame_reflects_walk_completion() {
        // For a fully-walked greedy decode, last_frame == num_timesteps.
        let enc = flatten(&[frame(&[0.0, 0.0, 0.0, 1.0])]);
        let out = rnnt_decode(&enc, &RnntAttrs::greedy(1, 3)).unwrap();
        assert_eq!(out[0].last_frame, 1);
    }
}
