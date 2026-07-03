//! Reusable, heap-stable scratch buffers for the Whisper forward pass
//! (FR-EX-05, hot-path malloc elimination).
//!
//! This is the whisper.cpp "reused compute buffer" pattern, expressed in 100%
//! safe Rust. Every transient buffer the encoder / decoder used to allocate per
//! call (`vec![0.0; …]` inside the sibling `nn` module) is promoted to a field
//! here, sized **once** to its worst case with [`Vec::with_capacity`] and
//! thereafter only `clear()`-ed and `resize()`-d to the current step's
//! dimensions.
//!
//! # Why this is provably zero-alloc at steady state (no `unsafe`)
//!
//! A [`Vec`] reallocates **iff** the requested length exceeds its current
//! capacity. `with_reserve` reserves each field to the maximum extent a bounded
//! Whisper decode can reach (the query/prefix width times the text-context
//! length), and every per-step `ensure` does
//! `clear(); resize(n, 0.0)` with `n ≤ capacity`. `clear()` keeps the capacity;
//! `resize` within capacity never reallocates. So after construction the
//! autoregressive step loop performs **no heap traffic** for these buffers — a
//! property the capacity-stability oracle in the `super::decoder` tests asserts
//! directly (a `Vec::capacity()` that never changes across N steps *is* the
//! proof no reallocation occurred). No raw pointers, no `MaybeUninit`, no
//! `unsafe`: `Vec::resize` / `clear` do all the work.
//!
//! # Distinct fields = simultaneously live, no aliasing
//!
//! Every buffer that is read while another is written is a **separate field**,
//! so the borrow checker permits the disjoint `&mut scratch.a` / `&scratch.b`
//! borrows the `*_into` helpers need — no `RefCell`, no split-at tricks. The
//! fields are `pub(crate)` precisely so the sibling `nn` / `decoder` / `encoder`
//! modules can take those disjoint field borrows at the call site. The field
//! set mirrors the locals of the former allocating `nn` functions one-for-one,
//! so the arithmetic (and thus the accumulation order, and thus the bit-exact
//! result) is unchanged.

/// `clear()` then `resize(n, 0.0)`: the zero-alloc reset-and-size primitive.
///
/// Within the capacity reserved by a `with_reserve` this never reallocates
/// (see the module docs). Kept as a free fn so every field uses the identical,
/// audited sizing path.
#[inline]
pub(crate) fn resize_zeroed(buf: &mut Vec<f32>, n: usize) {
    buf.clear();
    buf.resize(n, 0.0);
}

/// Head width `d / n_head`, guarding the zero-head configuration the synthetic
/// tiny-decoder tests use (`n_audio_layer = 0` ⇒ `n_head` may be `0`; the
/// attention scratch then reserves nothing and is never driven).
#[inline]
fn head_dim(d: usize, n_head: usize) -> usize {
    d.checked_div(n_head).unwrap_or(0)
}

/// Per-head multi-head-attention scratch: the eight buffers the former
/// `attention_from_kv` allocated on every call, now reused across heads, layers
/// and decode steps.
///
/// Shapes (row-major), for `t_q` queries attending `t_kv` keys with hidden
/// width `d` split into `n_head` heads of `hd = d / n_head`: `q`,`context` are
/// `[t_q, d]`; `scores`,`probs` are `[t_q, t_kv]`; the per-head slices
/// `qh`,`ctx_h` are `[t_q, hd]`, `kh_t` is `[hd, t_kv]` and `vh` is `[t_kv, hd]`.
pub(crate) struct AttnScratch {
    /// Scaled query projection `[t_q, d]`.
    pub(crate) q: Vec<f32>,
    /// Per-head context concatenated back to `[t_q, d]`.
    pub(crate) context: Vec<f32>,
    /// Attention scores `[t_q, t_kv]` (reused per head).
    pub(crate) scores: Vec<f32>,
    /// Softmax probabilities `[t_q, t_kv]` (reused per head).
    pub(crate) probs: Vec<f32>,
    /// This head's query slice `[t_q, hd]`.
    pub(crate) qh: Vec<f32>,
    /// This head's key slice, transposed to `[hd, t_kv]`.
    pub(crate) kh_t: Vec<f32>,
    /// This head's value slice `[t_kv, hd]`.
    pub(crate) vh: Vec<f32>,
    /// This head's context `[t_q, hd]` before scatter into `context`.
    pub(crate) ctx_h: Vec<f32>,
}

impl AttnScratch {
    /// Reserves capacity for at most `t_q_max` queries attending `t_kv_max`
    /// keys, so [`ensure`](Self::ensure) never reallocates within those bounds.
    pub(crate) fn with_reserve(t_q_max: usize, t_kv_max: usize, d: usize, n_head: usize) -> Self {
        let hd = head_dim(d, n_head);
        Self {
            q: Vec::with_capacity(t_q_max * d),
            context: Vec::with_capacity(t_q_max * d),
            scores: Vec::with_capacity(t_q_max * t_kv_max),
            probs: Vec::with_capacity(t_q_max * t_kv_max),
            qh: Vec::with_capacity(t_q_max * hd),
            kh_t: Vec::with_capacity(hd * t_kv_max),
            vh: Vec::with_capacity(t_kv_max * hd),
            ctx_h: Vec::with_capacity(t_q_max * hd),
        }
    }

    /// Sizes every field to this call's dimensions (`clear(); resize`). No
    /// reallocation while `t_q ≤ t_q_max` and `t_kv ≤ t_kv_max` of the
    /// [`with_reserve`](Self::with_reserve) that built it.
    pub(crate) fn ensure(&mut self, t_q: usize, t_kv: usize, d: usize, n_head: usize) {
        let hd = head_dim(d, n_head);
        resize_zeroed(&mut self.q, t_q * d);
        resize_zeroed(&mut self.context, t_q * d);
        resize_zeroed(&mut self.scores, t_q * t_kv);
        resize_zeroed(&mut self.probs, t_q * t_kv);
        resize_zeroed(&mut self.qh, t_q * hd);
        resize_zeroed(&mut self.kh_t, hd * t_kv);
        resize_zeroed(&mut self.vh, t_kv * hd);
        resize_zeroed(&mut self.ctx_h, t_q * hd);
    }

    /// The capacities of every field, for the zero-alloc capacity-stability
    /// oracle (a field that reallocated would report a larger capacity).
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> [usize; 8] {
        [
            self.q.capacity(),
            self.context.capacity(),
            self.scores.capacity(),
            self.probs.capacity(),
            self.qh.capacity(),
            self.kh_t.capacity(),
            self.vh.capacity(),
            self.ctx_h.capacity(),
        ]
    }
}

/// One transformer block's reusable buffers: the pre-norm output, the
/// self-attention key/value projections that feed the KV cache, the pre-residual
/// sub-block output, the two MLP buffers, and the nested attention scratch.
///
/// Shared verbatim by the decoder's self/cross/MLP sub-blocks (all `[t, d]`
/// residual-shaped) and by the encoder's self-attention blocks.
pub(crate) struct BlockScratch {
    /// Pre-norm (LayerNorm) output `[t, d]`; the attention/MLP input.
    pub(crate) ln: Vec<f32>,
    /// Self-attention key projection `[t, d]` before it is appended to the KV
    /// cache (unused by cross-attention, which has precomputed K/V).
    pub(crate) k: Vec<f32>,
    /// Self-attention value projection `[t, d]` before the KV-cache append.
    pub(crate) v: Vec<f32>,
    /// A sub-block's output `[t, d]` before the residual `add_assign` into the
    /// hidden state (reused across self-attn, cross-attn and the MLP).
    pub(crate) block_out: Vec<f32>,
    /// MLP hidden `[t, ffn_dim]` (the `fc1` output).
    pub(crate) mlp_h: Vec<f32>,
    /// MLP activation `[t, ffn_dim]` (the GELU output).
    pub(crate) mlp_a: Vec<f32>,
    /// Nested multi-head-attention scratch.
    pub(crate) attn: AttnScratch,
}

impl BlockScratch {
    /// Reserves for at most `t_q_max` residual rows attending `t_kv_max` keys,
    /// with hidden width `d`, MLP width `ff` and `n_head` heads.
    pub(crate) fn with_reserve(
        t_q_max: usize,
        t_kv_max: usize,
        d: usize,
        ff: usize,
        n_head: usize,
    ) -> Self {
        Self {
            ln: Vec::with_capacity(t_q_max * d),
            k: Vec::with_capacity(t_q_max * d),
            v: Vec::with_capacity(t_q_max * d),
            block_out: Vec::with_capacity(t_q_max * d),
            mlp_h: Vec::with_capacity(t_q_max * ff),
            mlp_a: Vec::with_capacity(t_q_max * ff),
            attn: AttnScratch::with_reserve(t_q_max, t_kv_max, d, n_head),
        }
    }

    /// Sizes the residual-shaped fields to `t` rows of width `d` and the MLP
    /// buffers to width `ff`. The attention scratch is sized on demand by
    /// `nn::attention_from_kv_into` (its `t_kv` differs between self- and
    /// cross-attention within the same block).
    pub(crate) fn ensure_residual(&mut self, t: usize, d: usize, ff: usize) {
        resize_zeroed(&mut self.ln, t * d);
        resize_zeroed(&mut self.k, t * d);
        resize_zeroed(&mut self.v, t * d);
        resize_zeroed(&mut self.block_out, t * d);
        resize_zeroed(&mut self.mlp_h, t * ff);
        resize_zeroed(&mut self.mlp_a, t * ff);
    }

    /// Every field capacity, for the capacity-stability oracle.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> Vec<usize> {
        let mut c = vec![
            self.ln.capacity(),
            self.k.capacity(),
            self.v.capacity(),
            self.block_out.capacity(),
            self.mlp_h.capacity(),
            self.mlp_a.capacity(),
        ];
        c.extend_from_slice(&self.attn.capacities());
        c
    }
}

/// Tied-logits-head scratch: the transpose of the hidden state, the
/// `[n_vocab, t]` GEMM output and the transposed `[t, n_vocab]` logits.
///
/// Not reserved to the full text context (that would be `n_vocab * n_text_ctx`
/// floats — tens of megabytes); reserved to the query/prefix width, which is
/// the largest single decode step on the greedy hot path (`t = 1` after the
/// prefix). Larger one-shot calls (e.g. a full-window beam recompute) resize up
/// once, which is outside the steady-state loop the oracle measures.
pub(crate) struct LogitsScratch {
    /// Hidden state transposed to `[d, t]`.
    pub(crate) h_t: Vec<f32>,
    /// `token_emb @ h_t` as `[n_vocab, t]`.
    pub(crate) logits_t: Vec<f32>,
    /// Final logits `[t, n_vocab]`.
    pub(crate) out: Vec<f32>,
}

impl LogitsScratch {
    /// Reserves for at most `t_max` positions over a `[n_vocab, d]` head.
    pub(crate) fn with_reserve(t_max: usize, d: usize, n_vocab: usize) -> Self {
        Self {
            h_t: Vec::with_capacity(d * t_max),
            logits_t: Vec::with_capacity(n_vocab * t_max),
            out: Vec::with_capacity(t_max * n_vocab),
        }
    }

    /// Sizes the three fields for `t` positions over `[n_vocab, d]`.
    pub(crate) fn ensure(&mut self, t: usize, d: usize, n_vocab: usize) {
        resize_zeroed(&mut self.h_t, d * t);
        resize_zeroed(&mut self.logits_t, n_vocab * t);
        resize_zeroed(&mut self.out, t * n_vocab);
    }

    /// Field capacities, for the capacity-stability oracle.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> [usize; 3] {
        [
            self.h_t.capacity(),
            self.logits_t.capacity(),
            self.out.capacity(),
        ]
    }
}

/// Encoder self-attention block scratch: one [`BlockScratch`] reused across
/// every `n_audio_layer` block of a single `encode` call, sized once to the
/// audio context length (the encoder is bidirectional, so `t_q == t_kv == t`
/// and nothing grows between blocks).
pub(crate) struct EncoderScratch {
    /// The wrapped per-block scratch, reused across all encoder layers.
    pub(crate) block: BlockScratch,
}

impl EncoderScratch {
    /// Reserves for `t` audio positions of width `d`, MLP width `ff`, `n_head`
    /// heads. `t_q_max == t_kv_max == t` because encoder attention is full and
    /// non-causal.
    pub(crate) fn with_reserve(t: usize, d: usize, ff: usize, n_head: usize) -> Self {
        Self {
            block: BlockScratch::with_reserve(t, t, d, ff, n_head),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_within_reserve_keeps_capacity() {
        // Reserve for the worst case, then size down and back up within it:
        // capacity must never move (the zero-alloc invariant in miniature).
        let mut a = AttnScratch::with_reserve(4, 8, 6, 2);
        let cap0 = a.capacities();
        a.ensure(1, 8, 6, 2); // a smaller-than-max step
        assert_eq!(a.capacities(), cap0, "sizing down must not reallocate");
        a.ensure(4, 8, 6, 2); // the max step
        assert_eq!(
            a.capacities(),
            cap0,
            "sizing to the reserve must not reallocate"
        );
    }

    #[test]
    fn resize_zeroed_clears_and_grows_to_len() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        resize_zeroed(&mut v, 5);
        assert_eq!(v, [0.0, 0.0, 0.0, 0.0, 0.0]);
        resize_zeroed(&mut v, 2);
        assert_eq!(v, [0.0, 0.0]);
    }

    #[test]
    fn head_dim_guards_zero_heads() {
        assert_eq!(head_dim(8, 0), 0);
        assert_eq!(head_dim(8, 2), 4);
    }

    #[test]
    fn block_scratch_residual_sizing_is_stable_within_reserve() {
        let mut b = BlockScratch::with_reserve(4, 8, 6, 12, 2);
        let cap0 = b.capacities();
        b.ensure_residual(1, 6, 12);
        b.attn.ensure(1, 5, 6, 2);
        assert_eq!(b.capacities(), cap0);
    }

    #[test]
    fn logits_scratch_stable_within_reserve() {
        let mut l = LogitsScratch::with_reserve(4, 6, 10);
        let cap0 = l.capacities();
        l.ensure(1, 6, 10);
        assert_eq!(l.capacities(), cap0);
        l.ensure(4, 6, 10);
        assert_eq!(l.capacities(), cap0);
    }
}
