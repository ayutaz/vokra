//! SBV2 text encoder + BERT bridge: phoneme/tone/word-boundary embedding
//! sum → transformer stack ([`SbV2TextEncoder`]), and the DeBERTa hidden
//! state → text-hidden-length additive contribution ([`BertBridge`]).
//! (Clean-room comment: see `mod.rs` — the transformer block below
//! follows the generic "Attention Is All You Need" structure
//! (arXiv:1706.03762, Vaswani et al. 2017), no SBV2/BV2 source
//! referenced.)
//!
//! `TransformerBlock` reuse decision (Task 17): **(B) new local
//! [`SbV2TransformerBlock`]**, not a `piper_plus` reuse — the only
//! candidate there, `piper_plus::text_encoder::TextEncoder`'s inlined
//! `attention`/`ffn` methods, is `pub(super)`-scoped to `piper_plus`
//! (unreachable from `sbv2`), has no standalone block type at all
//! (weights live as 4 parallel `Vec`s directly on `TextEncoder`, not a
//! `Vec<Block>`), and its `forward` takes a `&Compute` handle plus
//! VITS-specific windowed relative-position attention state that this
//! task neither needs nor wants to depend on. See `task-17-report.md`
//! for the full comparison.
//!
//! # Layout convention
//!
//! Every buffer in this module is flat, row-major, **position-major**:
//! a `[rows, cols]` buffer addresses row `r` as
//! `buf[r * cols .. (r + 1) * cols]`. This matches
//! [`StyleVectorInjector`](super::style::StyleVectorInjector)'s
//! `[seq_len, d_target]` convention (not `piper_plus`'s channel-major
//! `[channels, time]` layout).

/// Numerical-stability epsilon for [`SbV2TransformerBlock`]'s two
/// LayerNorm stages. A standard default, not model-specific.
const LN_EPS: f32 = 1e-5;

/// SBV2 text encoder: sums three per-position embedding lookups
/// (phoneme id, pitch-accent tone, word-boundary flag) into a `[seq_len,
/// d_model]` hidden state, then applies `transformer_layers` in order.
///
/// All three embedding tables are flat, row-major `[rows, d_model]`
/// buffers (`table[id * d_model .. (id + 1) * d_model]` addresses row
/// `id`), and [`forward`](SbV2TextEncoder::forward)'s output is likewise
/// flat, row-major `[seq_len, d_model]` — see the module-level layout
/// note.
pub struct SbV2TextEncoder {
    /// Phoneme-id embedding table, row-major `[n_vocab, d_model]`.
    phoneme_embed: Vec<f32>,
    /// Pitch-accent tone embedding table, row-major `[n_tones, d_model]`.
    tone_embed: Vec<f32>,
    /// Word-boundary flag embedding table, row-major `[2, d_model]` (row
    /// 0 = not a word start, row 1 = word start).
    wb_embed: Vec<f32>,
    /// Transformer stack, applied to the summed embedding in order. An
    /// empty stack is a legitimate, exercised no-op configuration (see
    /// this crate's `tests/sbv2_text_encoder.rs`) — not a silent
    /// fallback, since it is the documented behavior of a documented
    /// empty configuration, not an undocumented skip of requested work.
    transformer_layers: Vec<SbV2TransformerBlock>,
    /// Hidden (model) dimension shared by every embedding table and
    /// every transformer block.
    d_model: usize,
    /// Phoneme vocabulary size (`phoneme_embed.len() == n_vocab *
    /// d_model`).
    n_vocab: usize,
    /// Pitch-accent tone count (`tone_embed.len() == n_tones *
    /// d_model`).
    n_tones: usize,
}

impl SbV2TextEncoder {
    /// Builds an encoder from pre-trained embedding tables and a
    /// (possibly empty) transformer stack.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, so only in debug builds — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs for why this crate uses `debug_assert!` rather than
    /// `Result` for constructor shape checks) if `phoneme_embed.len() !=
    /// n_vocab * d_model`, `tone_embed.len() != n_tones * d_model`, or
    /// `wb_embed.len() != 2 * d_model`.
    pub fn from_weights(
        phoneme_embed: Vec<f32>,
        tone_embed: Vec<f32>,
        wb_embed: Vec<f32>,
        transformer_layers: Vec<SbV2TransformerBlock>,
        d_model: usize,
        n_vocab: usize,
        n_tones: usize,
    ) -> Self {
        debug_assert_eq!(
            phoneme_embed.len(),
            n_vocab * d_model,
            "phoneme_embed must be [n_vocab, d_model]"
        );
        debug_assert_eq!(
            tone_embed.len(),
            n_tones * d_model,
            "tone_embed must be [n_tones, d_model]"
        );
        debug_assert_eq!(wb_embed.len(), 2 * d_model, "wb_embed must be [2, d_model]");
        Self {
            phoneme_embed,
            tone_embed,
            wb_embed,
            transformer_layers,
            d_model,
            n_vocab,
            n_tones,
        }
    }

    /// Encodes one phoneme sequence: `phoneme_embed[id] + tone_embed[tone]
    /// + wb_embed[word_boundary]` summed per position, then
    /// `transformer_layers` applied in order.
    ///
    /// Returns a flat, row-major `[phoneme_ids.len(), d_model]` buffer
    /// (`out[i * d_model .. (i + 1) * d_model]` is position `i`'s hidden
    /// vector).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `tones.len()` or
    /// `word_boundaries.len()` differ from `phoneme_ids.len()`, if any
    /// `phoneme_ids` entry is `>= self.n_vocab`, or if any `tones` entry
    /// is `>= self.n_tones`.
    pub fn forward(&self, phoneme_ids: &[u16], tones: &[u8], word_boundaries: &[bool]) -> Vec<f32> {
        debug_assert_eq!(
            phoneme_ids.len(),
            tones.len(),
            "phoneme_ids and tones must be the same length"
        );
        debug_assert_eq!(
            phoneme_ids.len(),
            word_boundaries.len(),
            "phoneme_ids and word_boundaries must be the same length"
        );
        debug_assert!(
            phoneme_ids.iter().all(|&id| (id as usize) < self.n_vocab),
            "phoneme id out of range"
        );
        debug_assert!(
            tones.iter().all(|&t| (t as usize) < self.n_tones),
            "tone out of range"
        );

        let d_model = self.d_model;
        let seq_len = phoneme_ids.len();
        let mut hidden = vec![0.0_f32; seq_len * d_model];
        for ((out_row, &id), (&tone, &wb)) in hidden
            .chunks_exact_mut(d_model)
            .zip(phoneme_ids.iter())
            .zip(tones.iter().zip(word_boundaries.iter()))
        {
            let ph_row = &self.phoneme_embed[id as usize * d_model..(id as usize + 1) * d_model];
            let tone_row = &self.tone_embed[tone as usize * d_model..(tone as usize + 1) * d_model];
            let wb_row = &self.wb_embed[wb as usize * d_model..(wb as usize + 1) * d_model];
            for ((o, (&p, &t)), &w) in out_row
                .iter_mut()
                .zip(ph_row.iter().zip(tone_row.iter()))
                .zip(wb_row.iter())
            {
                *o = p + t + w;
            }
        }

        for block in &self.transformer_layers {
            block.forward(&mut hidden, seq_len);
        }

        hidden
    }
}

/// A single self-attention + FFN transformer block, applied in place to
/// a `[seq_len, d_model]` hidden buffer (see the module-level layout
/// note).
///
/// Structure: `hidden = LN1(hidden + SelfAttn(hidden))`, then
/// `hidden = LN2(hidden + FFN(hidden))` — the generic post-LayerNorm
/// Transformer block ("Attention Is All You Need", arXiv:1706.03762),
/// **single-head**, full (unmasked, non-causal) self-attention, no
/// relative-position bias, bias-free attention projections. See the
/// module doc's `TransformerBlock` reuse decision for why this is a new,
/// independent type rather than a `piper_plus` reuse, and
/// `task-17-report.md` for why the architecture is intentionally
/// minimal (real per-tensor weight loading and any architecture
/// correction against the upstream checkpoint — head count, norm
/// placement, FFN width/activation — land in Task 24-27).
pub struct SbV2TransformerBlock {
    /// Query projection, row-major `[d_model, d_model]` (`q[o] = Σ_i
    /// w_q[o, i] · x[i]`, no bias term).
    w_q: Vec<f32>,
    /// Key projection, row-major `[d_model, d_model]`, no bias term.
    w_k: Vec<f32>,
    /// Value projection, row-major `[d_model, d_model]`, no bias term.
    w_v: Vec<f32>,
    /// Output projection, row-major `[d_model, d_model]`, no bias term.
    w_o: Vec<f32>,
    /// First LayerNorm's per-channel scale (applied after the
    /// self-attention residual), `[d_model]`.
    ln1_gamma: Vec<f32>,
    /// First LayerNorm's per-channel bias, `[d_model]`.
    ln1_beta: Vec<f32>,
    /// FFN up-projection, row-major `[d_ff, d_model]`.
    ffn_w1: Vec<f32>,
    /// FFN up-projection bias, `[d_ff]`.
    ffn_b1: Vec<f32>,
    /// FFN down-projection, row-major `[d_model, d_ff]`.
    ffn_w2: Vec<f32>,
    /// FFN down-projection bias, `[d_model]`.
    ffn_b2: Vec<f32>,
    /// Second LayerNorm's per-channel scale (applied after the FFN
    /// residual), `[d_model]`.
    ln2_gamma: Vec<f32>,
    /// Second LayerNorm's per-channel bias, `[d_model]`.
    ln2_beta: Vec<f32>,
    /// Hidden (model) dimension.
    d_model: usize,
    /// FFN inner width.
    d_ff: usize,
}

impl SbV2TransformerBlock {
    /// Builds a block from pre-trained weights. Crate-internal: no
    /// caller constructs a non-empty `transformer_layers` stack yet
    /// (Task 24-27 loads real weights from GGUF and will call this).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if any weight/bias buffer's length
    /// doesn't match the shape documented on its field above.
    #[allow(clippy::too_many_arguments)] // one arg per weight tensor, mirrors the struct's fields
    #[allow(dead_code)] // consumed by the Task 24-27 real weight-load wiring
    pub(crate) fn new(
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        ln1_gamma: Vec<f32>,
        ln1_beta: Vec<f32>,
        ffn_w1: Vec<f32>,
        ffn_b1: Vec<f32>,
        ffn_w2: Vec<f32>,
        ffn_b2: Vec<f32>,
        ln2_gamma: Vec<f32>,
        ln2_beta: Vec<f32>,
        d_model: usize,
        d_ff: usize,
    ) -> Self {
        debug_assert_eq!(
            w_q.len(),
            d_model * d_model,
            "w_q must be [d_model, d_model]"
        );
        debug_assert_eq!(
            w_k.len(),
            d_model * d_model,
            "w_k must be [d_model, d_model]"
        );
        debug_assert_eq!(
            w_v.len(),
            d_model * d_model,
            "w_v must be [d_model, d_model]"
        );
        debug_assert_eq!(
            w_o.len(),
            d_model * d_model,
            "w_o must be [d_model, d_model]"
        );
        debug_assert_eq!(ln1_gamma.len(), d_model, "ln1_gamma must be [d_model]");
        debug_assert_eq!(ln1_beta.len(), d_model, "ln1_beta must be [d_model]");
        debug_assert_eq!(
            ffn_w1.len(),
            d_ff * d_model,
            "ffn_w1 must be [d_ff, d_model]"
        );
        debug_assert_eq!(ffn_b1.len(), d_ff, "ffn_b1 must be [d_ff]");
        debug_assert_eq!(
            ffn_w2.len(),
            d_model * d_ff,
            "ffn_w2 must be [d_model, d_ff]"
        );
        debug_assert_eq!(ffn_b2.len(), d_model, "ffn_b2 must be [d_model]");
        debug_assert_eq!(ln2_gamma.len(), d_model, "ln2_gamma must be [d_model]");
        debug_assert_eq!(ln2_beta.len(), d_model, "ln2_beta must be [d_model]");
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            ln1_gamma,
            ln1_beta,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
            ln2_gamma,
            ln2_beta,
            d_model,
            d_ff,
        }
    }

    /// Applies this block in place to `hidden` (`[seq_len, d_model]`
    /// row-major).
    fn forward(&self, hidden: &mut [f32], seq_len: usize) {
        debug_assert_eq!(
            hidden.len(),
            seq_len * self.d_model,
            "hidden must be [seq_len, d_model]"
        );

        let attn_out = self.self_attention(hidden);
        add_residual_inplace(hidden, &attn_out);
        layer_norm_rows_inplace(hidden, self.d_model, &self.ln1_gamma, &self.ln1_beta);

        let ffn_out = self.ffn(hidden);
        add_residual_inplace(hidden, &ffn_out);
        layer_norm_rows_inplace(hidden, self.d_model, &self.ln2_gamma, &self.ln2_beta);
    }

    /// Single-head scaled dot-product self-attention (full, non-causal).
    fn self_attention(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.d_model;
        let q = linear_rows(hidden, d, &self.w_q, d);
        let k = linear_rows(hidden, d, &self.w_k, d);
        let v = linear_rows(hidden, d, &self.w_v, d);
        let scale = 1.0_f32 / (d as f32).sqrt();

        let mut ctx = vec![0.0_f32; hidden.len()];
        let mut scores = vec![0.0_f32; hidden.len() / d];
        for (qi, ctx_row) in q.chunks_exact(d).zip(ctx.chunks_exact_mut(d)) {
            for (s, kj) in scores.iter_mut().zip(k.chunks_exact(d)) {
                *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax_inplace(&mut scores);
            for (&p, vj) in scores.iter().zip(v.chunks_exact(d)) {
                for (o, &vjd) in ctx_row.iter_mut().zip(vj) {
                    *o += p * vjd;
                }
            }
        }
        linear_rows(&ctx, d, &self.w_o, d)
    }

    /// Position-wise FFN: `ffn_w2 · ReLU(ffn_w1 · x + ffn_b1) + ffn_b2`.
    fn ffn(&self, hidden: &[f32]) -> Vec<f32> {
        let mut h = linear_rows_biased(hidden, self.d_model, &self.ffn_w1, &self.ffn_b1, self.d_ff);
        for v in &mut h {
            *v = v.max(0.0);
        }
        linear_rows_biased(&h, self.d_ff, &self.ffn_w2, &self.ffn_b2, self.d_model)
    }
}

/// DeBERTa-hidden → text-hidden-length additive bridge: projects a
/// `[bert_seq_len, d_bert]` DeBERTa v2/v3 output through a `1×1` conv (a
/// per-position linear map) to `[bert_seq_len, d_target]`, then
/// nearest-neighbor-interpolates along the sequence axis to
/// `[text_seq_len, d_target]`.
///
/// [`forward`](BertBridge::forward) returns only the
/// projected+interpolated contribution — it does **not** add it to any
/// text hidden state itself. The caller (`SbV2Model`, wired in Task 23)
/// performs that additive step, which keeps this function pure and
/// independently testable: zero `conv_weight`/`conv_bias` makes
/// `forward` return an all-zero contribution (the additive identity), so
/// a caller can disable the BERT contribution entirely by zeroing this
/// bridge's weights (see `bert_bridge_zero_weights_returns_zeros` in
/// `tests/sbv2_text_encoder.rs`).
pub struct BertBridge {
    /// `1×1` conv weight, row-major `[d_target, d_bert]` (same
    /// convention as
    /// [`StyleVectorInjector`](super::style::StyleVectorInjector)'s
    /// projections: `y[o] = Σ_i conv_weight[o, i] · x[i] +
    /// conv_bias[o]`).
    conv_weight: Vec<f32>,
    /// `1×1` conv bias, `[d_target]`.
    conv_bias: Vec<f32>,
    /// Input (DeBERTa) hidden dimension.
    d_bert: usize,
    /// Output (text hidden) dimension.
    d_target: usize,
}

impl BertBridge {
    /// Builds a bridge from a pre-trained `1×1` conv projection.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `conv_weight.len() != d_target *
    /// d_bert` or `conv_bias.len() != d_target`.
    pub fn from_conv(
        conv_weight: Vec<f32>,
        conv_bias: Vec<f32>,
        d_bert: usize,
        d_target: usize,
    ) -> Self {
        debug_assert_eq!(
            conv_weight.len(),
            d_target * d_bert,
            "conv_weight must be [d_target, d_bert]"
        );
        debug_assert_eq!(conv_bias.len(), d_target, "conv_bias must be [d_target]");
        Self {
            conv_weight,
            conv_bias,
            d_bert,
            d_target,
        }
    }

    /// Projects `bert_hidden` (`[bert_seq_len, d_bert]` row-major) to
    /// `[bert_seq_len, d_target]`, then nearest-neighbor-interpolates
    /// along the sequence axis to `[text_seq_len, d_target]` (row-major,
    /// flat `Vec<f32>` of length `text_seq_len * d_target`).
    ///
    /// Interpolation maps text position `t` to source (BERT) position
    /// `s = min((t * bert_seq_len) / text_seq_len, bert_seq_len - 1)`
    /// (integer division — nearest-neighbor, not linear/bilinear).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) in debug builds if `bert_seq_len ==
    /// 0` (an empty BERT sequence has no source position for the
    /// nearest-neighbor interpolation above to read from — the
    /// `bert_seq_len.saturating_sub(1)` clamp below prevents `usize`
    /// underflow but not the resulting empty-slice indexing) or if
    /// `bert_hidden.len() != bert_seq_len * self.d_bert`.
    pub fn forward(
        &self,
        bert_hidden: &[f32],
        text_seq_len: usize,
        bert_seq_len: usize,
    ) -> Vec<f32> {
        debug_assert!(
            bert_seq_len > 0,
            "BertBridge::forward requires a non-empty bert sequence (bert_seq_len == 0 \
             has no source position for the nearest-neighbor interpolation to read from)"
        );
        debug_assert_eq!(
            bert_hidden.len(),
            bert_seq_len * self.d_bert,
            "bert_hidden must be [bert_seq_len, d_bert]"
        );

        let d_target = self.d_target;
        let projected = linear_rows_biased(
            bert_hidden,
            self.d_bert,
            &self.conv_weight,
            &self.conv_bias,
            d_target,
        );

        let mut out = vec![0.0_f32; text_seq_len * d_target];
        for (t, out_row) in out.chunks_exact_mut(d_target).enumerate() {
            let s = (t * bert_seq_len / text_seq_len).min(bert_seq_len.saturating_sub(1));
            let src = &projected[s * d_target..(s + 1) * d_target];
            out_row.copy_from_slice(src);
        }
        out
    }
}

/// Applies a bias-free `[out_dim, in_dim]` row-major linear map to each
/// `in_dim`-wide row of `x`, producing a flat `[rows, out_dim]` buffer
/// (`rows = x.len() / in_dim`).
fn linear_rows(x: &[f32], in_dim: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
    let rows = x.len() / in_dim;
    let mut out = vec![0.0_f32; rows * out_dim];
    for (xi, oi) in x.chunks_exact(in_dim).zip(out.chunks_exact_mut(out_dim)) {
        for (o, wrow) in oi.iter_mut().zip(w.chunks_exact(in_dim)) {
            *o = wrow.iter().zip(xi).map(|(a, b)| a * b).sum();
        }
    }
    out
}

/// As [`linear_rows`], plus a per-output-channel bias `b` (`[out_dim]`)
/// added to every row.
fn linear_rows_biased(x: &[f32], in_dim: usize, w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = linear_rows(x, in_dim, w, out_dim);
    for row in out.chunks_exact_mut(out_dim) {
        for (o, &bi) in row.iter_mut().zip(b) {
            *o += bi;
        }
    }
    out
}

/// `a += b`, element-wise, in place.
fn add_residual_inplace(a: &mut [f32], b: &[f32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Numerically-stable softmax over `x`, in place.
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// LayerNorm over the last axis (`d_model` channels) of each row of a
/// `[rows, d_model]` buffer, in place.
fn layer_norm_rows_inplace(x: &mut [f32], d_model: usize, gamma: &[f32], beta: &[f32]) {
    for row in x.chunks_exact_mut(d_model) {
        let mean = row.iter().sum::<f32>() / d_model as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d_model as f32;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        for (v, (&g, &b)) in row.iter_mut().zip(gamma.iter().zip(beta.iter())) {
            *v = (*v - mean) * inv_std * g + b;
        }
    }
}
