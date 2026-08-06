//! Hybrid CTC / attention decoding + LSTM LM shallow fusion
//! (SoTA plan JA-ASR-3 primitive).
//!
//! # Runtime function — NOT a graph node (FR-EX-10 / FR-OP-40)
//!
//! [`hybrid_ctc_attention_decode`] is a host-side runtime function, not an
//! `OpKind` variant. Same posture as [`crate::ctc_decode::ctc_decode_beam`]
//! and [`vokra_core::decode::beam_search`]: encoding a hybrid rescorer as
//! a graph op would freeze `ctc_weight` / `lm_weight` / `beam_width` into
//! the model at conversion time, precisely the axes callers change most
//! often.
//!
//! # What this primitive is
//!
//! ESPnet-style hybrid rescoring — the attention decoder produces the
//! candidate token sequence (autoregressive beam search) and CTC provides
//! a **prefix score** at every beam-expansion step that gets **combined in
//! log space** with the attention log-prob:
//!
//! ```text
//! joint_score(prefix)
//!     = (1 - ctc_weight) · attn_logprob(prefix)
//!     + ctc_weight       · ctc_prefix_logprob(prefix)
//!     + lm_weight        · lstm_lm_logprob(prefix)     // shallow fusion (optional)
//! ```
//!
//! - `attn_logprob` is the running sum of the per-step log-probabilities the
//!   attention decoder emitted along the beam;
//! - `ctc_prefix_logprob` is the acoustic CTC log-probability that the
//!   prefix aligns to any CTC path over the encoder timeline
//!   (Watanabe et al. 2017, [Hybrid CTC/Attention Architecture for
//!   End-to-End Speech Recognition][watanabe-2017]);
//! - `lstm_lm_logprob` is the optional single-layer LSTM shallow-fusion
//!   score (ESPnet's default `LM Fusion Model`).
//!
//! For the CTC prefix score, this primitive uses the **truncated
//! prefix-beam approximation** — a `T×|labels|` DP over the CTC log-probs
//! where the last-column blank vs non-blank split lets us extend the beam
//! by one token at a time. This matches ESPnet's
//! `CTCPrefixScorer.final_score()` interface (score-per-candidate rather
//! than joint alignment path).
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! Scalar Rust, no BLAS, no `serde`, no third-party crate. The LSTM LM
//! forward computes a single time-step per beam-expansion call using
//! caller-supplied gate weights (`i` / `f` / `g` / `o` split); this is the
//! same shape ESPnet's `SequentialRNNLM` exposes.
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Every degenerate input becomes a loud [`VokraError::InvalidArgument`]:
//! shape mismatches on `ctc_log_probs`, `attn_next_step_fn` returning the
//! wrong vocab width, zero beam width, `ctc_weight` outside `[0, 1]`,
//! non-finite `lm_weight`, empty candidate list, or an SOS/EOS id ≥ vocab.
//!
//! [watanabe-2017]: https://arxiv.org/abs/1609.06773

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Signature of the attention decoder's next-step callback.
///
/// Given the current prefix (SOS-prepended) the callback returns a `Vec<f32>`
/// of length `vocab_size` — the per-token log-probability the attention
/// decoder assigned to extending this prefix by one token. Cached decoder
/// state lives inside the closure (typically a `RefCell<KVCache>`).
///
/// Kept as a top-level alias to satisfy `clippy::type_complexity` on
/// [`HybridCtcAttentionAttrs`].
pub type AttnNextStepFn<'a> = dyn Fn(&[u32]) -> Vec<f32> + 'a;

/// Signature of the optional LSTM LM shallow-fusion callback.
///
/// Given the current prefix (SOS-prepended) and a candidate token id,
/// returns an `f32` log-probability contribution. The LSTM state is the
/// caller's problem (they close over an interior-mutable `RefCell`).
pub type LmScoreFn<'a> = dyn Fn(&[u32], u32) -> f32 + 'a;

/// Hybrid decoding attributes.
pub struct HybridCtcAttentionAttrs<'a> {
    /// Number of encoder timesteps (rows) in `ctc_log_probs`.
    pub time: usize,
    /// Vocabulary size — every log-prob row must be this wide.
    pub vocab: usize,
    /// CTC blank token id (`0 <= blank_id < vocab`).
    pub blank_id: usize,
    /// SOS token id — prepended to every candidate prefix before the
    /// attention next-step callback and the LM callback fire. Must satisfy
    /// `sos_id < vocab`.
    pub sos_id: u32,
    /// EOS token id — the beam terminates a hypothesis when it emits this
    /// token. Must satisfy `eos_id < vocab`.
    pub eos_id: u32,
    /// Beam width (kept per step, `>= 1`).
    pub beam_width: usize,
    /// Maximum output length (safety cap; the beam terminates all live
    /// hypotheses when it hits `max_len` even if none emitted EOS).
    pub max_len: usize,
    /// CTC weight `α ∈ [0, 1]`. `0` = attention-only, `1` = CTC-only.
    /// Upstream ESPnet defaults to `0.3` for JA ASR.
    pub ctc_weight: f32,
    /// LSTM LM shallow-fusion weight (added to the ranking score as
    /// `lm_weight * sum lm_score`). `0.0` disables it even when
    /// `lm_score_fn` is Some. Must be finite.
    pub lm_weight: f32,
    /// Attention decoder callback (mandatory).
    pub attn_next_step_fn: &'a AttnNextStepFn<'a>,
    /// Optional LSTM LM callback for shallow fusion.
    pub lm_score_fn: Option<&'a LmScoreFn<'a>>,
}

/// Per-hypothesis result.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridHypothesis {
    /// Emitted token sequence (labels only, no blanks, EOS stripped).
    pub tokens: Vec<u32>,
    /// Cumulative joint score:
    /// `(1-α) · attn + α · ctc + lm_weight · lm`.
    pub score: f32,
}

/// Runs hybrid CTC / attention decoding with optional LSTM LM shallow
/// fusion.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any of:
/// - `time == 0` or `vocab == 0`;
/// - `blank_id >= vocab`, `sos_id >= vocab`, or `eos_id >= vocab`;
/// - `beam_width == 0` or `max_len == 0`;
/// - `ctc_log_probs.len() != time * vocab`;
/// - `ctc_weight` outside `[0, 1]` or non-finite `lm_weight`;
/// - `attn_next_step_fn` returns the wrong vocab width (checked on first
///   call).
///
/// # Returns
///
/// Up to `beam_width` completed hypotheses, sorted by `score` descending.
/// A hypothesis is "completed" iff it emitted EOS **or** it hit `max_len`
/// without emitting EOS (the latter case keeps the running joint score
/// but truncates the sequence to `max_len`).
pub fn hybrid_ctc_attention_decode(
    ctc_log_probs: &[f32],
    attrs: &HybridCtcAttentionAttrs,
) -> Result<Vec<HybridHypothesis>> {
    validate_attrs(ctc_log_probs, attrs)?;

    let HybridCtcAttentionAttrs {
        time,
        vocab,
        blank_id,
        sos_id,
        eos_id,
        beam_width,
        max_len,
        ctc_weight,
        lm_weight,
        attn_next_step_fn,
        lm_score_fn,
    } = *attrs;

    // Precompute the CTC prefix scorer helper.
    let ctc = CtcPrefixScorer::new(ctc_log_probs, time, vocab, blank_id);

    // Live beam: hypotheses currently being extended.
    // Each carries the running attention log-prob sum, the running LM log-prob
    // sum, and its CTC prefix state (last (pb, pnb) row over the CTC time
    // axis).
    let initial_ctc_state = ctc.initial_state();
    let initial_joint_rank = ctc_weight * initial_ctc_state.total;
    let mut live: Vec<BeamState> = vec![BeamState {
        tokens: Vec::new(),
        attn_score: 0.0,
        lm_score: 0.0,
        ctc_state: initial_ctc_state,
        joint_rank: initial_joint_rank,
    }];

    // Completed beam: hypotheses that emitted EOS or hit max_len.
    let mut completed: Vec<HybridHypothesis> = Vec::new();

    let attn_weight = 1.0 - ctc_weight;

    for _step in 0..max_len {
        if live.is_empty() {
            break;
        }
        // Extend every live hypothesis by every candidate token.
        // Candidates: full vocab. (An optional per-step top-K optimisation
        // is a follow-up — the primitive stays correct rather than fast.)
        let mut next: Vec<BeamState> = Vec::with_capacity(live.len() * vocab);
        for state in &live {
            // Attention step: fresh log-prob distribution over the vocab.
            let mut prefix_with_sos: Vec<u32> = Vec::with_capacity(state.tokens.len() + 1);
            prefix_with_sos.push(sos_id);
            prefix_with_sos.extend_from_slice(&state.tokens);
            let attn_lp = attn_next_step_fn(&prefix_with_sos);
            if attn_lp.len() != vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "hybrid_ctc_attention_decode: attn_next_step_fn returned {} log-probs, \
                     expected vocab={vocab} (FR-EX-08)",
                    attn_lp.len()
                )));
            }
            for tok in 0..(vocab as u32) {
                let attn_delta = attn_lp[tok as usize];
                if !attn_delta.is_finite() {
                    continue;
                }
                let (ctc_delta, ctc_state) = ctc.extend(&state.ctc_state, tok);
                let lm_delta = match lm_score_fn {
                    Some(f) if lm_weight != 0.0 => f(&prefix_with_sos, tok),
                    _ => 0.0,
                };
                let mut new_tokens = state.tokens.clone();
                new_tokens.push(tok);
                // Joint score — Watanabe/Hori hybrid interpolation on the
                // per-token step. `ctc_delta` is the total CTC prefix
                // log-prob so far (from `CtcPrefixScorer::extend`), so it
                // already integrates the whole path; the attn/lm sums are
                // running per-token additions. The joint rank uses the
                // whole triple so the pruning respects the CTC/LM
                // contribution (upstream ESPnet: `hyps_att[i]["ctc"]` +
                // `hyps_att[i]["att"] * (1-alpha)`).
                let joint_rank = attn_weight * (state.attn_score + attn_delta)
                    + ctc_weight * ctc_delta
                    + lm_weight * (state.lm_score + lm_delta);
                let new_state = BeamState {
                    tokens: new_tokens,
                    attn_score: state.attn_score + attn_delta,
                    lm_score: state.lm_score + lm_delta,
                    ctc_state,
                    joint_rank,
                };
                if tok == eos_id {
                    // Completed hypothesis — drop the EOS from the emitted
                    // tokens and record it.
                    let mut final_tokens = new_state.tokens.clone();
                    final_tokens.pop();
                    completed.push(HybridHypothesis {
                        tokens: final_tokens,
                        score: joint_rank,
                    });
                } else {
                    next.push(new_state);
                }
            }
        }

        // Prune to top-`beam_width` by joint rank score (attn + ctc + lm
        // combined). Missing the CTC term here would silently drop
        // CTC-only signal at pruning time, which is exactly the bug
        // ESPnet's `join_ctc_attention` avoids by carrying the total
        // rescored score per beam entry.
        next.sort_by(|a, b| {
            b.joint_rank
                .partial_cmp(&a.joint_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next.truncate(beam_width);
        live = next;
    }

    // Any live beam that never emitted EOS but consumed the full max_len
    // budget graduates to completed with its running score. This matches
    // ESPnet's behaviour where a length-constrained beam still surfaces a
    // best-effort hypothesis rather than silently dropping (FR-EX-08).
    for state in live {
        completed.push(HybridHypothesis {
            tokens: state.tokens,
            score: state.joint_rank,
        });
    }

    completed.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    completed.truncate(beam_width);
    Ok(completed)
}

// ---------------------------------------------------------------------------
// LSTM LM shallow-fusion utility
// ---------------------------------------------------------------------------

/// Single-layer LSTM cell — the shape ESPnet's `SequentialRNNLM` exposes.
///
/// Callers wire a stateful shallow-fusion closure over this cell (with a
/// per-beam `RefCell<LstmLmState>`) to plug into
/// [`HybridCtcAttentionAttrs::lm_score_fn`]. Kept as a helper so callers
/// wanting a stateful LM do not need to reimplement the gate arithmetic;
/// stateless callers ignore this and pass their own callback.
#[derive(Debug, Clone)]
pub struct LstmLmCell {
    /// Row-major `[4 · hidden, input]` — input projection (concat of i / f
    /// / g / o gates, matching PyTorch's `LSTMCell.weight_ih_l0`).
    pub w_ih: Vec<f32>,
    /// `[4 · hidden]`.
    pub b_ih: Vec<f32>,
    /// Row-major `[4 · hidden, hidden]` — recurrent projection (concat of
    /// i / f / g / o gates, `LSTMCell.weight_hh_l0`).
    pub w_hh: Vec<f32>,
    /// `[4 · hidden]`.
    pub b_hh: Vec<f32>,
    /// Row-major `[vocab, embed_dim]` — token embedding table.
    pub emb: Vec<f32>,
    /// Row-major `[vocab, hidden]` — output projection to log-softmax.
    pub w_out: Vec<f32>,
    /// `[vocab]`.
    pub b_out: Vec<f32>,
    /// Hidden dim.
    pub hidden: usize,
    /// Vocab size.
    pub vocab: usize,
    /// Embed dim (= input width to the LSTM cell).
    pub embed_dim: usize,
}

/// Per-beam LSTM state.
#[derive(Debug, Clone)]
pub struct LstmLmState {
    /// `[hidden]` hidden state.
    pub h: Vec<f32>,
    /// `[hidden]` cell state.
    pub c: Vec<f32>,
}

impl LstmLmCell {
    /// Zero-initialised hidden + cell state.
    pub fn zero_state(&self) -> LstmLmState {
        LstmLmState {
            h: vec![0.0; self.hidden],
            c: vec![0.0; self.hidden],
        }
    }

    /// Advance one step: consume `token`, update `state`, return the
    /// log-probability of the caller's next-token candidate `candidate`.
    ///
    /// The caller typically pre-runs the prefix (`step()` for every
    /// history token) into a per-beam state, then queries per-candidate.
    /// This function does both — it accepts the token to feed *into* the
    /// LSTM (advancing the state), then computes the log-softmax
    /// distribution and returns the entry for `candidate`.
    pub fn step(&self, state: &mut LstmLmState, token: u32, candidate: u32) -> Result<f32> {
        if (token as usize) >= self.vocab {
            return Err(VokraError::InvalidArgument(format!(
                "LstmLmCell::step: token {token} >= vocab {} (FR-EX-08)",
                self.vocab
            )));
        }
        if (candidate as usize) >= self.vocab {
            return Err(VokraError::InvalidArgument(format!(
                "LstmLmCell::step: candidate {candidate} >= vocab {} (FR-EX-08)",
                self.vocab
            )));
        }
        // Embed the input token.
        let emb_off = (token as usize) * self.embed_dim;
        let x = &self.emb[emb_off..emb_off + self.embed_dim];

        // Compute gates: gates = W_ih @ x + b_ih + W_hh @ h + b_hh.
        let four_h = 4 * self.hidden;
        let mut gates = vec![0.0f32; four_h];
        for (o, slot) in gates.iter_mut().enumerate().take(four_h) {
            let mut acc = self.b_ih[o] + self.b_hh[o];
            let w_ih_row = &self.w_ih[o * self.embed_dim..(o + 1) * self.embed_dim];
            for (i, &xi) in x.iter().enumerate().take(self.embed_dim) {
                acc += w_ih_row[i] * xi;
            }
            let w_hh_row = &self.w_hh[o * self.hidden..(o + 1) * self.hidden];
            for (i, &hi) in state.h.iter().enumerate().take(self.hidden) {
                acc += w_hh_row[i] * hi;
            }
            *slot = acc;
        }

        // Split gates into (i, f, g, o) and apply activations.
        // Standard PyTorch layout: [i; f; g; o].
        let hidden = self.hidden;
        let (i_slice, rest) = gates.split_at(hidden);
        let (f_slice, rest) = rest.split_at(hidden);
        let (g_slice, o_slice) = rest.split_at(hidden);
        let mut new_c = vec![0.0f32; hidden];
        let mut new_h = vec![0.0f32; hidden];
        for k in 0..hidden {
            let it = sigmoid(i_slice[k]);
            let ft = sigmoid(f_slice[k]);
            let gt = g_slice[k].tanh();
            let ot = sigmoid(o_slice[k]);
            new_c[k] = ft * state.c[k] + it * gt;
            new_h[k] = ot * new_c[k].tanh();
        }
        state.c = new_c;
        state.h = new_h;

        // Output projection + log-softmax over vocab. Only need the entry
        // for `candidate`, but the log-softmax normaliser needs the full
        // row.
        let mut logits = vec![0.0f32; self.vocab];
        for (o, slot) in logits.iter_mut().enumerate().take(self.vocab) {
            let mut acc = self.b_out[o];
            let w_row = &self.w_out[o * self.hidden..(o + 1) * self.hidden];
            for (i, &hi) in state.h.iter().enumerate().take(self.hidden) {
                acc += w_row[i] * hi;
            }
            *slot = acc;
        }
        // log-softmax.
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            return Ok(f32::NEG_INFINITY);
        }
        let sum_exp: f32 = logits.iter().map(|&x| (x - m).exp()).sum();
        let log_z = m + sum_exp.ln();
        Ok(logits[candidate as usize] - log_z)
    }
}

// ---------------------------------------------------------------------------
// CTC prefix scorer (truncated Watanabe / Hori prefix beam DP)
// ---------------------------------------------------------------------------

/// State the CTC prefix scorer carries per beam entry.
#[derive(Debug, Clone)]
struct CtcPrefixState {
    /// Per-timestep pair `(pb, pnb)` of log-probabilities that a CTC path
    /// terminating at time `t` produces the prefix with either a trailing
    /// blank (`pb`) or trailing non-blank (`pnb`). Length `time`.
    per_step: Vec<(f32, f32)>,
    /// Running joint log-prob of the prefix aligning to any CTC path over
    /// the full encoder timeline. `log_add(per_step[T-1])`.
    total: f32,
    /// Last token emitted (for the "repeat vs new label" split); `None`
    /// for the empty prefix.
    last: Option<u32>,
}

struct CtcPrefixScorer<'a> {
    log_probs: &'a [f32],
    time: usize,
    vocab: usize,
    blank_id: usize,
}

impl<'a> CtcPrefixScorer<'a> {
    fn new(log_probs: &'a [f32], time: usize, vocab: usize, blank_id: usize) -> Self {
        Self {
            log_probs,
            time,
            vocab,
            blank_id,
        }
    }

    /// The empty prefix's state: at time `t`, `pb = sum_{0..=t} lp[·, blank]`
    /// (path is all-blank), `pnb = -inf` (empty prefix has no label).
    fn initial_state(&self) -> CtcPrefixState {
        let mut per_step = Vec::with_capacity(self.time);
        let mut running_pb = 0.0f32; // log 1 = 0 for the "before t=0" position
        for t in 0..self.time {
            let lp_blank = self.log_probs[t * self.vocab + self.blank_id];
            running_pb += lp_blank;
            per_step.push((running_pb, f32::NEG_INFINITY));
        }
        let total = if self.time == 0 {
            0.0
        } else {
            let (pb, _pnb) = per_step[self.time - 1];
            pb
        };
        CtcPrefixState {
            per_step,
            total,
            last: None,
        }
    }

    /// Extend the prefix with token `tok`, returning `(prefix_logprob,
    /// new_state)`. Uses the standard Hannun prefix-beam recursion:
    ///
    /// - If `tok == blank`: `pb'[t] = log_add(pb[t-1] + lp[t, blank],
    ///   pnb[t-1] + lp[t, blank])`, `pnb'[t] = pnb[t-1] + lp[t, blank]`.
    ///   (Blank extension doesn't change the label sequence; this
    ///   primitive treats "extend with blank" as a no-op token, but
    ///   emitting blank as a candidate is not the callers' usual path —
    ///   they filter blank out of `candidate` selection.)
    /// - Otherwise: `pnb'[t] = ...` per Hannun 2014 eqs. 3-4, and
    ///   `pb'[t] = pb[t-1] + lp[t, blank]`.
    fn extend(&self, state: &CtcPrefixState, tok: u32) -> (f32, CtcPrefixState) {
        if self.time == 0 {
            return (f32::NEG_INFINITY, state.clone());
        }
        // Special-case blank: extension is a no-op on labels. Return the
        // running total unchanged; callers should not emit blank as a
        // label anyway.
        if tok as usize == self.blank_id {
            return (state.total, state.clone());
        }
        let mut per_step = vec![(f32::NEG_INFINITY, f32::NEG_INFINITY); self.time];
        for t in 0..self.time {
            let lp_tok = self.log_probs[t * self.vocab + tok as usize];
            let lp_blank = self.log_probs[t * self.vocab + self.blank_id];
            // pnb'[t]:
            //   - "same label as prefix's tail, so must have been through blank":
            //     pb_prev + lp[t, tok]
            //   - "different label / new emission":
            //     log_add(pb_prev, pnb_prev) + lp[t, tok]
            let (pb_prev, pnb_prev) = if t == 0 {
                // "before t=0": empty prefix has pb=0, pnb=-inf; extended prefixes
                // start from the anchor position (which is the empty prefix's step
                // stored in `state.per_step[0]` conceptually, but the recursion
                // reads t-1). For the standard Watanabe formulation we treat the
                // t=0 case as extending from the empty-prefix anchor:
                (0.0, f32::NEG_INFINITY)
            } else {
                per_step[t - 1]
            };
            let mixed = log_add(pb_prev, pnb_prev);
            let pnb_new = if state.last == Some(tok) {
                // Repeat vs new-label split — the "new label" branch requires a
                // blank in between (uses pb_prev only).
                log_add(pb_prev + lp_tok, state.per_step[t].1 + lp_tok)
            } else {
                mixed + lp_tok
            };
            // pb'[t] = pb_prev + lp[t, blank] + pnb_prev + lp[t, blank] (log-add)
            let pb_new = log_add(pb_prev + lp_blank, pnb_prev + lp_blank);
            per_step[t] = (pb_new, pnb_new);
        }
        let (pb_last, pnb_last) = per_step[self.time - 1];
        let total = log_add(pb_last, pnb_last);
        let new_state = CtcPrefixState {
            per_step,
            total,
            last: Some(tok),
        };
        (total, new_state)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BeamState {
    tokens: Vec<u32>,
    attn_score: f32,
    lm_score: f32,
    ctc_state: CtcPrefixState,
    /// Cached joint rank at the last extension:
    /// `attn_weight · attn_score + ctc_weight · ctc_total + lm_weight · lm_score`.
    /// The initial state has `joint_rank = attn_weight · 0 + ctc_weight
    /// · initial_ctc_total + lm_weight · 0`; every extend fills this in.
    joint_rank: f32,
}

fn validate_attrs(ctc_log_probs: &[f32], attrs: &HybridCtcAttentionAttrs) -> Result<()> {
    if attrs.time == 0 {
        return Err(VokraError::InvalidArgument(
            "hybrid_ctc_attention_decode: time must be >= 1".to_owned(),
        ));
    }
    if attrs.vocab == 0 {
        return Err(VokraError::InvalidArgument(
            "hybrid_ctc_attention_decode: vocab must be >= 1".to_owned(),
        ));
    }
    if attrs.blank_id >= attrs.vocab {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: blank_id {} >= vocab {} (FR-EX-08)",
            attrs.blank_id, attrs.vocab
        )));
    }
    if (attrs.sos_id as usize) >= attrs.vocab {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: sos_id {} >= vocab {} (FR-EX-08)",
            attrs.sos_id, attrs.vocab
        )));
    }
    if (attrs.eos_id as usize) >= attrs.vocab {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: eos_id {} >= vocab {} (FR-EX-08)",
            attrs.eos_id, attrs.vocab
        )));
    }
    if attrs.beam_width == 0 {
        return Err(VokraError::InvalidArgument(
            "hybrid_ctc_attention_decode: beam_width must be >= 1".to_owned(),
        ));
    }
    if attrs.max_len == 0 {
        return Err(VokraError::InvalidArgument(
            "hybrid_ctc_attention_decode: max_len must be >= 1".to_owned(),
        ));
    }
    if !attrs.ctc_weight.is_finite() || !(0.0..=1.0).contains(&attrs.ctc_weight) {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: ctc_weight must be finite and in [0, 1], got {}",
            attrs.ctc_weight
        )));
    }
    if !attrs.lm_weight.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: lm_weight must be finite, got {}",
            attrs.lm_weight
        )));
    }
    let expected = attrs.time.checked_mul(attrs.vocab).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: time {} * vocab {} overflows usize",
            attrs.time, attrs.vocab
        ))
    })?;
    if ctc_log_probs.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "hybrid_ctc_attention_decode: ctc_log_probs.len() {} != time*vocab {} \
             (time={}, vocab={})",
            ctc_log_probs.len(),
            expected,
            attrs.time,
            attrs.vocab
        )));
    }
    Ok(())
}

/// Numerically-stable `log(exp(a) + exp(b))`.
fn log_add(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    max + (min - max).exp().ln_1p()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a peaky log-prob matrix that assigns most of the mass to a
    /// specific target token at each timestep.
    fn peaky_log_probs(targets: &[u32], vocab: usize, blank_id: usize) -> Vec<f32> {
        let time = targets.len();
        let mut out = vec![-10.0f32; time * vocab];
        for (t, &tok) in targets.iter().enumerate() {
            out[t * vocab + tok as usize] = 0.0; // near log 1
            if tok as usize != blank_id {
                out[t * vocab + blank_id] = -5.0; // small blank leakage
            }
        }
        out
    }

    // ---- Happy path -----------------------------------------------------

    /// Attention decoder that always assigns log_prob = 0 (log 1) to token `1`
    /// and log-inf to everything else. With attention-only weighting we
    /// should get a beam of "1" tokens up to max_len (or EOS).
    #[test]
    fn attention_only_greedy_returns_expected_tokens() {
        let vocab = 4;
        let blank = 0;
        let sos = 3;
        let eos = 2;
        let time = 4;
        let ctc_lp = vec![0.0f32; time * vocab];
        let attn_lp = move |_prefix: &[u32]| {
            let mut lp = vec![f32::NEG_INFINITY; vocab];
            lp[1] = 0.0;
            lp[eos as usize] = -0.1; // slightly less than "1", won't fire yet
            lp
        };
        let boxed: Box<AttnNextStepFn> = Box::new(attn_lp);
        let attrs = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.0, // attention-only
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let hyps = hybrid_ctc_attention_decode(&ctc_lp, &attrs).unwrap();
        assert!(!hyps.is_empty(), "expected at least one hypothesis");
        // Top hypothesis must be all "1" tokens up to max_len.
        assert_eq!(hyps[0].tokens, vec![1, 1, 1]);
    }

    /// Attention decoder that always emits EOS with log 1 — beam must
    /// terminate immediately and emit an empty token list.
    #[test]
    fn eos_at_step_zero_returns_empty_hypothesis() {
        let vocab = 3;
        let blank = 0;
        let sos = 2;
        let eos = 1;
        let time = 3;
        let ctc_lp = vec![0.0f32; time * vocab];
        let attn_lp = move |_prefix: &[u32]| {
            let mut lp = vec![f32::NEG_INFINITY; vocab];
            lp[eos as usize] = 0.0;
            lp
        };
        let boxed: Box<AttnNextStepFn> = Box::new(attn_lp);
        let attrs = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 2,
            max_len: 5,
            ctc_weight: 0.0,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let hyps = hybrid_ctc_attention_decode(&ctc_lp, &attrs).unwrap();
        assert!(!hyps.is_empty());
        // Top hypothesis emitted EOS immediately → empty token list.
        assert!(hyps[0].tokens.is_empty(), "got {:?}", hyps[0].tokens);
    }

    /// LM shallow fusion: an LM that strongly prefers token `3` over
    /// token `1` must flip the top hypothesis when `lm_weight > 0`.
    /// (Uses vocab=5 with sos=4 / eos=0 so tokens 1 and 3 are both
    /// non-SOS non-EOS non-blank candidates the beam can extend by.)
    #[test]
    fn lm_shallow_fusion_reranks_toward_lm_preference() {
        let vocab = 5;
        let blank = 2;
        let sos = 4;
        let eos = 0;
        let time = 2;
        let ctc_lp = vec![0.0f32; time * vocab];
        // Attention slightly prefers "1" over "3". EOS = 0 gets a very
        // negative log-prob so it never wins on acoustic alone.
        let attn_lp = move |_prefix: &[u32]| {
            let mut lp = vec![f32::NEG_INFINITY; vocab];
            lp[1] = -0.5;
            lp[3] = -0.6;
            lp[eos as usize] = -5.0;
            lp
        };
        let attn_boxed: Box<AttnNextStepFn> = Box::new(attn_lp);
        // Baseline: attention + LM disabled, expect "1" to win.
        let attrs_baseline = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 2,
            max_len: 2,
            ctc_weight: 0.0,
            lm_weight: 0.0,
            attn_next_step_fn: &*attn_boxed,
            lm_score_fn: None,
        };
        let baseline = hybrid_ctc_attention_decode(&ctc_lp, &attrs_baseline).unwrap();
        assert!(
            baseline[0].tokens.contains(&1),
            "baseline expected token 1, got {:?}",
            baseline[0].tokens
        );

        // With LM strongly preferring "3", the top hypothesis must flip.
        let lm = move |_prefix: &[u32], c: u32| -> f32 { if c == 3 { 3.0 } else { 0.0 } };
        let lm_boxed: Box<LmScoreFn> = Box::new(lm);
        let attrs_lm = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 2,
            max_len: 2,
            ctc_weight: 0.0,
            lm_weight: 1.0,
            attn_next_step_fn: &*attn_boxed,
            lm_score_fn: Some(&*lm_boxed),
        };
        let boosted = hybrid_ctc_attention_decode(&ctc_lp, &attrs_lm).unwrap();
        assert!(
            boosted[0].tokens.contains(&3),
            "LM boost must rerank toward token 3, got {:?}",
            boosted[0].tokens
        );
    }

    /// CTC weight = 1 (CTC-only) with a peaky CTC distribution and an
    /// attention decoder that emits every non-EOS token uniformly — the
    /// top hypothesis must be shaped by CTC alone.
    ///
    /// Uses vocab=5 with sos=4/eos=0/blank=2 so tokens 1 and 3 are
    /// candidate labels the beam can extend by. Peaky CTC prefers token
    /// 1 at every step.
    #[test]
    fn ctc_only_with_peaky_ctc_shapes_the_beam() {
        let vocab = 5;
        let blank = 2;
        let sos = 4;
        let eos = 0;
        let time = 3;
        // Peaky CTC on token 1 across the whole encoder timeline.
        let ctc_lp = peaky_log_probs(&[1, 1, 1], vocab, blank);
        // Attention: uniform over non-EOS non-SOS tokens; EOS gets a
        // very negative score so it doesn't fire prematurely.
        let attn_lp = move |_prefix: &[u32]| {
            let mut lp = vec![0.0f32; vocab];
            lp[eos as usize] = -20.0; // never wins on attn alone
            lp[sos as usize] = f32::NEG_INFINITY; // never re-emit SOS
            lp
        };
        let boxed: Box<AttnNextStepFn> = Box::new(attn_lp);
        let attrs = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 3,
            max_len: 3,
            ctc_weight: 1.0, // CTC-only
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let hyps = hybrid_ctc_attention_decode(&ctc_lp, &attrs).unwrap();
        assert!(!hyps.is_empty());
        // Top hypothesis must contain token 1 (the CTC-peaky token) —
        // CTC-only ranking should push non-peaky tokens (like 3) out of
        // the beam.
        assert!(
            hyps[0].tokens.contains(&1),
            "CTC-only must surface the peaky token, got {:?}",
            hyps[0].tokens
        );
    }

    // ---- Shape / arg validation -----------------------------------------

    #[test]
    fn rejects_zero_time() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let attrs = HybridCtcAttentionAttrs {
            time: 0,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.5,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&[], &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_zero_beam_width() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 0,
            max_len: 3,
            ctc_weight: 0.5,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_zero_max_len() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 2,
            max_len: 0,
            ctc_weight: 0.5,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_shape_mismatch_on_ctc_log_probs() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 5]; // Wrong length: 3*4 = 12 expected.
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.5,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_ctc_weight_out_of_range() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        for bad in [-0.1f32, 1.5, f32::NAN] {
            let attrs = HybridCtcAttentionAttrs {
                time: 3,
                vocab: 4,
                blank_id: 0,
                sos_id: 3,
                eos_id: 2,
                beam_width: 2,
                max_len: 3,
                ctc_weight: bad,
                lm_weight: 0.0,
                attn_next_step_fn: &*boxed,
                lm_score_fn: None,
            };
            let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
            assert!(matches!(err, VokraError::InvalidArgument(_)));
        }
    }

    #[test]
    fn rejects_non_finite_lm_weight() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.5,
            lm_weight: f32::NAN,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_sos_or_eos_out_of_range() {
        let attn = |_p: &[u32]| vec![0.0f32; 4];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 4, // >= vocab
            eos_id: 2,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.5,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn rejects_attn_callback_wrong_vocab_width() {
        // Attention returns vocab=3 but attrs.vocab=4 → loud error on first
        // call.
        let attn = |_p: &[u32]| vec![0.0f32; 3];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let ctc = vec![0.0f32; 3 * 4];
        let attrs = HybridCtcAttentionAttrs {
            time: 3,
            vocab: 4,
            blank_id: 0,
            sos_id: 3,
            eos_id: 2,
            beam_width: 2,
            max_len: 3,
            ctc_weight: 0.0,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let err = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap_err();
        match err {
            VokraError::InvalidArgument(m) => assert!(m.contains("attn_next_step_fn returned")),
            _ => panic!("expected InvalidArgument"),
        }
    }

    // ---- LSTM LM cell ---------------------------------------------------

    fn tiny_lm_cell(hidden: usize, vocab: usize, embed: usize) -> LstmLmCell {
        let four_h = 4 * hidden;
        LstmLmCell {
            w_ih: vec![0.01; four_h * embed],
            b_ih: vec![0.0; four_h],
            w_hh: vec![0.01; four_h * hidden],
            b_hh: vec![0.0; four_h],
            emb: vec![0.1; vocab * embed],
            w_out: vec![0.05; vocab * hidden],
            b_out: vec![0.0; vocab],
            hidden,
            vocab,
            embed_dim: embed,
        }
    }

    #[test]
    fn lstm_lm_step_updates_state_and_returns_finite_logprob() {
        let cell = tiny_lm_cell(4, 5, 3);
        let mut state = cell.zero_state();
        let lp = cell.step(&mut state, 2u32, 3u32).unwrap();
        assert!(lp.is_finite(), "log-prob must be finite");
        // Hidden state must have changed from all-zero after one step.
        assert!(state.h.iter().any(|&v| v != 0.0));
        // Cell state must also have changed.
        assert!(state.c.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn lstm_lm_step_rejects_out_of_range_token() {
        let cell = tiny_lm_cell(4, 5, 3);
        let mut state = cell.zero_state();
        let err = cell.step(&mut state, 6u32, 3u32).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn lstm_lm_step_rejects_out_of_range_candidate() {
        let cell = tiny_lm_cell(4, 5, 3);
        let mut state = cell.zero_state();
        let err = cell.step(&mut state, 2u32, 5u32).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn lstm_lm_log_softmax_sums_to_one() {
        // Iterate over every candidate; sum of exp(log-probs) must be ~1.
        let cell = tiny_lm_cell(4, 5, 3);
        let mut sum = 0.0f32;
        for candidate in 0..(cell.vocab as u32) {
            let mut state = cell.zero_state();
            let lp = cell.step(&mut state, 0u32, candidate).unwrap();
            sum += lp.exp();
        }
        // Small allowance for f32 rounding.
        assert!((sum - 1.0).abs() < 1e-4, "sum of exp(log-probs) = {sum}");
    }

    // ---- log_add helper -------------------------------------------------

    #[test]
    fn log_add_handles_neg_infinity() {
        assert_eq!(log_add(f32::NEG_INFINITY, 0.5), 0.5);
        assert_eq!(log_add(0.5, f32::NEG_INFINITY), 0.5);
    }

    #[test]
    fn log_add_matches_log_of_sum() {
        // log_add(log 2, log 3) = log 5.
        let a = 2f32.ln();
        let b = 3f32.ln();
        let expected = 5f32.ln();
        assert!((log_add(a, b) - expected).abs() < 1e-6);
    }

    // ---- Determinism ----------------------------------------------------

    #[test]
    fn hybrid_decode_is_deterministic_under_fixed_callbacks() {
        let vocab = 3;
        let blank = 0;
        let sos = 2;
        let eos = 1;
        let time = 3;
        let ctc = vec![0.0f32; time * vocab];
        let attn = |_p: &[u32]| vec![-0.5f32, 0.5, -1.0];
        let boxed: Box<AttnNextStepFn> = Box::new(attn);
        let attrs = HybridCtcAttentionAttrs {
            time,
            vocab,
            blank_id: blank,
            sos_id: sos,
            eos_id: eos,
            beam_width: 3,
            max_len: 4,
            ctc_weight: 0.3,
            lm_weight: 0.0,
            attn_next_step_fn: &*boxed,
            lm_score_fn: None,
        };
        let a = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap();
        let b = hybrid_ctc_attention_decode(&ctc, &attrs).unwrap();
        assert_eq!(
            a, b,
            "hybrid decode must be deterministic under fixed inputs"
        );
    }
}
