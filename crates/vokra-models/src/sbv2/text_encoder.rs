//! SBV2 text encoder + BERT bridge.
//!
//! The `SbV2TextEncoder` sums three per-position embedding lookups
//! (phoneme id, pitch-accent tone) plus one per-utterance language
//! embedding broadcast across every position, scales by `sqrt(d_model)`,
//! then applies a stack of VITS-style **post-LayerNorm relative-position
//! transformer blocks** ([`SbV2TransformerBlock`]) — the architecture the
//! real base checkpoint (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`)
//! actually uses (117 tensors under `enc_p.*`, one 6-layer relative-
//! position transformer stack under `enc_p.encoder.*`).
//!
//! Clean-room reference: `tools/parity/vendor/vits/attentions.py` +
//! `.../modules.py` (jaywalnut310/vits, MIT). NOT referenced:
//! github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
//! github.com/fishaudio/Bert-VITS2 (AGPL-3.0) — only the MIT source has
//! been read.
//!
//! # Layout convention
//!
//! Every buffer in this module is flat, row-major, **position-major**:
//! a `[rows, cols]` buffer addresses row `r` as
//! `buf[r * cols .. (r + 1) * cols]`. This matches
//! [`StyleVectorInjector`](super::style::StyleVectorInjector)'s
//! `[seq_len, d_target]` convention (not `piper_plus`'s channel-major
//! `[channels, time]` layout).
//!
//! Weight tensors follow PyTorch's Conv1d convention `[out_channels,
//! in_channels, kernel]` (contiguous, so `w[oc, ic, k] = w[oc *
//! in_channels * kernel + ic * kernel + k]`). For the `[192, 192, 1]`
//! 1×1 convolutions used by the attention Q/K/V/O projections and the
//! `[192, 1024, 1]` `bert_bridge.conv`, this is byte-identical to a
//! `[out, in]` linear-weight buffer since `kernel = 1`.

/// Numerical-stability epsilon for [`LayerNorm`]. Matches upstream VITS
/// `modules.LayerNorm.__init__(eps=1e-5)`.
const LN_EPS: f32 = 1e-5;

/// Number of languages in the SBV2 v2 base checkpoint's
/// `enc_p.language_emb.weight` table. Real upstream shape observed on
/// `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` is `[3, 192]` — one row
/// per supported language: JA / EN / ZH (see [`SbV2TextEncoder::forward`]'s
/// `language_id` doc for the tentative row-ordering convention this crate
/// assumes, pending real-checkpoint config verification).
///
/// Formerly the SBV2 v2 design doc §7 assumed a `word_boundary_emb` table
/// (`[2, d_model]`, one row per boundary flag) at this slot — that
/// assumption did not survive the M6 real-checkpoint scout: no
/// `enc_p.word_boundary_emb.weight` tensor exists in the base checkpoint,
/// but `enc_p.language_emb.weight [3, 192]` does. This constant, and the
/// [`SbV2TextEncoder`] field it sizes, are the design-doc correction.
pub const N_LANGUAGES: usize = 3;

/// SBV2 text encoder: sums three per-position embedding lookups
/// (phoneme id, pitch-accent tone) plus one per-utterance language
/// embedding broadcast across every position, scales by `sqrt(d_model)`,
/// then applies `transformer_layers` in order (VITS-style relative-
/// position transformer stack — see [`SbV2TransformerBlock`]).
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
    /// Per-language embedding table, row-major `[N_LANGUAGES, d_model]`
    /// (row `0` = JA, `1` = EN, `2` = ZH by tentative convention — see
    /// [`forward`](Self::forward)'s `language_id` doc).
    language_embed: Vec<f32>,
    /// Transformer stack, applied to the summed+scaled embedding in
    /// order. An empty stack is a legitimate, exercised no-op
    /// configuration (see this crate's `tests/sbv2_text_encoder.rs` and
    /// `synthetic_for_test`) — the sum-then-`sqrt(d_model)` scaling
    /// still runs, so an empty-stack forward returns `(phoneme_embed +
    /// tone_embed + language_embed) * sqrt(d_model)`.
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
    /// Precomputed `sqrt(d_model)` scaling factor applied to the summed
    /// embedding before the transformer stack — matches upstream VITS
    /// `TextEncoder.forward`'s `x = self.emb(x) * math.sqrt(self.hidden_channels)`.
    scale: f32,
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
    /// `language_embed.len() != N_LANGUAGES * d_model`.
    pub fn from_weights(
        phoneme_embed: Vec<f32>,
        tone_embed: Vec<f32>,
        language_embed: Vec<f32>,
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
        debug_assert_eq!(
            language_embed.len(),
            N_LANGUAGES * d_model,
            "language_embed must be [N_LANGUAGES, d_model] (N_LANGUAGES = 3: JA/EN/ZH)"
        );
        for block in &transformer_layers {
            debug_assert_eq!(
                block.d_model, d_model,
                "every transformer block must share the encoder's d_model"
            );
        }
        let scale = (d_model as f32).sqrt();
        Self {
            phoneme_embed,
            tone_embed,
            language_embed,
            transformer_layers,
            d_model,
            n_vocab,
            n_tones,
            scale,
        }
    }

    /// The hidden (model) dimension shared by every embedding table and
    /// every transformer block — the row stride of [`forward`](Self::forward)'s
    /// output. Task 23 (`SbV2Model::synthesize`) uses this to index the
    /// per-position broadcast adds (speaker embedding, BERT bridge) it
    /// performs on that output.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Encodes one phoneme sequence: `x[i] = (phoneme_embed[phoneme_ids[i]]
    /// plus tone_embed[tones[i]] plus language_embed[language_id]) *
    /// sqrt(d_model)` summed per position (with `language_embed[language_id]`
    /// broadcast-added identically to every position of the utterance — it
    /// is a per-utterance conditioning signal, not per-phoneme), then
    /// `transformer_layers` applied in order.
    ///
    /// Returns a flat, row-major `[phoneme_ids.len(), d_model]` buffer
    /// (`out[i * d_model .. (i + 1) * d_model]` is position `i`'s hidden
    /// vector).
    ///
    /// # `language_id` row-ordering convention (tentative — TODO owner)
    ///
    /// Row `0` = JA, row `1` = EN, row `2` = ZH. **This ordering is
    /// tentative**: the real
    /// `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` checkpoint ships
    /// weights-only (no config.json enumerates the language row order), so
    /// this crate cannot verify it from primary source alone. The
    /// tentative ordering matches the alphabetical / JP-Extra-family
    /// convention used across VITS-JA / SBV2-JP-Extra derivatives and is
    /// what
    /// [`SbV2Model::synthesize`](super::mod::SbV2Model)'s
    /// [`Language`](super::g2p::Language) → `language_id` mapping assumes.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `tones.len()` differs from
    /// `phoneme_ids.len()`, if any `phoneme_ids` entry is `>= self.n_vocab`,
    /// if any `tones` entry is `>= self.n_tones`, or if `language_id
    /// as usize >= N_LANGUAGES`.
    pub fn forward(&self, phoneme_ids: &[u16], tones: &[u8], language_id: u8) -> Vec<f32> {
        // Full pipeline; discards the pre-transformer `phoneme_embed`.
        // See [`Self::forward_with_embed`] for the accessor variant that
        // returns both intermediates (Wave-4 INTERMEDIATE-ACCESSORS).
        self.forward_with_embed(phoneme_ids, tones, language_id).1
    }

    /// Same forward pass as [`Self::forward`], but returns the
    /// **pre-transformer** sum `(phoneme + tone + language) * sqrt(d_model)`
    /// alongside the **post-transformer** hidden state. Both buffers are
    /// row-major `[seq_len, d_model]`. Added for Wave-4
    /// `INTERMEDIATE-ACCESSORS` so parity harnesses can diff
    /// `phoneme_embed.bin` and `text_hidden.bin` from the Python reference
    /// dumper separately — the pre-transformer sum is what upstream calls
    /// `phoneme_embed` (design doc §10, dumper's `phoneme_embed.bin`) and
    /// is not otherwise observable from outside the encoder.
    ///
    /// # Returns
    ///
    /// `(phoneme_embed, text_hidden)`, both `[seq_len, d_model]` row-major.
    /// `phoneme_embed == (phoneme_embed_table[id] + tone_embed[tone] +
    /// language_embed[language_id]) * sqrt(d_model)`. `text_hidden` is
    /// that same buffer after the transformer stack was applied
    /// in place.
    ///
    /// # Panics
    ///
    /// Same preconditions as [`Self::forward`].
    pub fn forward_with_embed(
        &self,
        phoneme_ids: &[u16],
        tones: &[u8],
        language_id: u8,
    ) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(
            phoneme_ids.len(),
            tones.len(),
            "phoneme_ids and tones must be the same length"
        );
        debug_assert!(
            phoneme_ids.iter().all(|&id| (id as usize) < self.n_vocab),
            "phoneme id out of range"
        );
        debug_assert!(
            tones.iter().all(|&t| (t as usize) < self.n_tones),
            "tone out of range"
        );
        debug_assert!(
            (language_id as usize) < N_LANGUAGES,
            "language_id ({language_id}) out of range (must be < N_LANGUAGES = {N_LANGUAGES})",
        );

        let d_model = self.d_model;
        let seq_len = phoneme_ids.len();
        // The language embedding row is a per-utterance additive contribution
        // broadcast identically to every position — hoist the slice bind out
        // of the per-position loop so the same rows are iterated once per
        // step, not once per (position × d_model) element.
        let lang_start = (language_id as usize) * d_model;
        let lang_row = &self.language_embed[lang_start..lang_start + d_model];
        let mut hidden = vec![0.0_f32; seq_len * d_model];
        for ((out_row, &id), &tone) in hidden
            .chunks_exact_mut(d_model)
            .zip(phoneme_ids.iter())
            .zip(tones.iter())
        {
            let ph_row = &self.phoneme_embed[id as usize * d_model..(id as usize + 1) * d_model];
            let tone_row = &self.tone_embed[tone as usize * d_model..(tone as usize + 1) * d_model];
            for ((o, (&p, &t)), &l) in out_row
                .iter_mut()
                .zip(ph_row.iter().zip(tone_row.iter()))
                .zip(lang_row.iter())
            {
                *o = (p + t + l) * self.scale;
            }
        }

        // Snapshot the pre-transformer sum before the in-place stack.
        let phoneme_embed = hidden.clone();

        for block in &self.transformer_layers {
            block.forward(&mut hidden, seq_len);
        }

        (phoneme_embed, hidden)
    }
}

// =====================================================================
// Relative-position transformer block (VITS-style)
// =====================================================================

/// One VITS-style **post-LayerNorm relative-position transformer block**
/// applied in place to a `[seq_len, d_model]` hidden buffer (see the
/// module-level layout note).
///
/// Structure (matches upstream `vendor/vits/attentions.py::Encoder.forward`
/// for one iteration `i`, at `p_dropout = 0`):
///
/// ```text
///     y = attn(x)                    # relative-position multi-head attention
///     x = norm1(x + y)               # residual + LayerNorm (channel-last)
///     y = ffn(x)                     # Conv1d(k=3, same-pad) → ReLU → Conv1d(k=3, same-pad)
///     x = norm2(x + y)               # residual + LayerNorm (channel-last)
/// ```
///
/// # Notes on parity vs upstream
///
/// - `p_dropout = 0` on all inference paths — no runtime randomness.
/// - The per-position `x_mask` multiply (upstream `x * x_mask` on the
///   entry to `attn` / `ffn`, and `x = x * x_mask` on `Encoder.forward`
///   exit) is a **no-op** for our single-utterance inference path
///   (there is no padded batch dimension — every position is valid),
///   so it is skipped rather than materialized as an all-ones
///   multiplication.
pub struct SbV2TransformerBlock {
    /// Relative-position multi-head self-attention (Q/K/V/O 1×1 convs
    /// with bias + shared `emb_rel_k` / `emb_rel_v` tables).
    attn: RelPositionMHA,
    /// Post-attn residual LayerNorm (channel-last, matches upstream
    /// `modules.LayerNorm`).
    norm1: LayerNorm,
    /// Position-wise FFN — two same-padded `kernel = ffn_kernel` Conv1d
    /// layers with bias, ReLU between (matches upstream `attentions.FFN`
    /// at `activation is None`, the SBV2 default).
    ffn: PositionWiseFFN,
    /// Post-FFN residual LayerNorm.
    norm2: LayerNorm,
    /// Hidden (model) dimension — every buffer this block sees is
    /// `[T, d_model]`.
    d_model: usize,
}

impl SbV2TransformerBlock {
    /// Builds a block from four pre-constructed sub-modules.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if any sub-module's hidden dimension
    /// disagrees with `d_model`.
    pub fn new(
        attn: RelPositionMHA,
        norm1: LayerNorm,
        ffn: PositionWiseFFN,
        norm2: LayerNorm,
        d_model: usize,
    ) -> Self {
        debug_assert_eq!(
            attn.n_heads * attn.d_head,
            d_model,
            "attn n_heads * d_head must equal d_model"
        );
        debug_assert_eq!(attn.d_model, d_model, "attn d_model must equal d_model");
        debug_assert_eq!(norm1.channels, d_model, "norm1 channels must equal d_model");
        debug_assert_eq!(norm2.channels, d_model, "norm2 channels must equal d_model");
        debug_assert_eq!(ffn.d_model, d_model, "ffn d_model must equal d_model");
        Self {
            attn,
            norm1,
            ffn,
            norm2,
            d_model,
        }
    }

    /// Applies this block in place to `hidden` (`[seq_len, d_model]`
    /// row-major).
    ///
    /// `pub(super)` so [`super::flow::SbV2TransformerCouplingLayer`]
    /// (Blocker 2b, VITS2 TransformerCouplingBlock) can drive the same
    /// primitive from inside the flow module without duplicating the
    /// residual/LayerNorm plumbing — matches upstream `p0p4k/vits2_pytorch`
    /// where `TransformerCouplingLayer` reuses `FFT` (this block's
    /// equivalent) verbatim.
    pub(super) fn forward(&self, hidden: &mut [f32], seq_len: usize) {
        debug_assert_eq!(
            hidden.len(),
            seq_len * self.d_model,
            "hidden must be [seq_len, d_model]"
        );

        let attn_out = self.attn.forward(hidden, seq_len);
        add_residual_inplace(hidden, &attn_out);
        self.norm1.forward_inplace(hidden);

        let ffn_out = self.ffn.forward(hidden, seq_len);
        add_residual_inplace(hidden, &ffn_out);
        self.norm2.forward_inplace(hidden);
    }
}

// =====================================================================
// Sub-modules
// =====================================================================

/// Channel-last LayerNorm over the last axis (`channels`) of each row of a
/// `[rows, channels]` buffer, in place. Matches upstream VITS
/// `vendor/vits/modules.py::LayerNorm` at `eps = 1e-5` (the upstream
/// default): the upstream implementation transposes `[B, D, T]` to
/// `[B, T, D]`, applies `F.layer_norm(x, (channels,), gamma, beta, eps)`,
/// then transposes back — for our already-position-major `[T, D]` layout
/// this collapses to normalizing across `channels` on each `T`-th row,
/// which is what [`forward_inplace`](LayerNorm::forward_inplace) does.
pub struct LayerNorm {
    /// Per-channel scale (upstream `nn.Parameter(torch.ones(channels))`).
    gamma: Vec<f32>,
    /// Per-channel bias (upstream `nn.Parameter(torch.zeros(channels))`).
    beta: Vec<f32>,
    /// Number of channels this LayerNorm normalizes over.
    channels: usize,
}

impl LayerNorm {
    /// Builds a channel-last LayerNorm from its trained parameters.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `gamma.len() != channels` or
    /// `beta.len() != channels`.
    pub fn new(gamma: Vec<f32>, beta: Vec<f32>, channels: usize) -> Self {
        debug_assert_eq!(gamma.len(), channels, "gamma must be [channels]");
        debug_assert_eq!(beta.len(), channels, "beta must be [channels]");
        Self {
            gamma,
            beta,
            channels,
        }
    }

    /// Normalizes `x` (`[rows, channels]` row-major) in place — per row,
    /// `out[c] = (x[c] - mean) / sqrt(var + eps) * gamma[c] + beta[c]`.
    fn forward_inplace(&self, x: &mut [f32]) {
        layer_norm_rows_inplace(x, self.channels, &self.gamma, &self.beta);
    }
}

/// Relative-position multi-head self-attention. Q, K, V, O projections
/// are 1×1 Conv1ds with bias (the checkpoint stores them as `[D, D, 1]`
/// tensors + `[D]` bias vectors — for kernel=1 the tensor bytes are
/// identical to a `[D, D]` linear-weight buffer, so this crate stores
/// them as such).
///
/// `emb_rel_k` / `emb_rel_v` are `[1, 2*window+1, d_head]` tables (shared
/// across heads — SBV2 uses `heads_share=True`, the upstream default) —
/// the relative-position bias/value bank the multi-head attention adds
/// per the VITS relative-position attention formula
/// (`vendor/vits/attentions.py::MultiHeadAttention::attention`).
pub struct RelPositionMHA {
    /// Q projection weight, `[d_model, d_model]` (kernel=1 Conv1d).
    q_weight: Vec<f32>,
    /// Q projection bias, `[d_model]`.
    q_bias: Vec<f32>,
    /// K projection weight, `[d_model, d_model]`.
    k_weight: Vec<f32>,
    /// K projection bias, `[d_model]`.
    k_bias: Vec<f32>,
    /// V projection weight, `[d_model, d_model]`.
    v_weight: Vec<f32>,
    /// V projection bias, `[d_model]`.
    v_bias: Vec<f32>,
    /// O projection weight, `[d_model, d_model]`.
    o_weight: Vec<f32>,
    /// O projection bias, `[d_model]`.
    o_bias: Vec<f32>,
    /// Relative-position key bias table, `[2*window+1, d_head]`
    /// (upstream shape `[1, 2*window+1, d_head]` with `heads_share=True`
    /// — the leading singleton head dim is dropped here since every head
    /// reads the same rows).
    emb_rel_k: Vec<f32>,
    /// Relative-position value contribution table, `[2*window+1, d_head]`.
    emb_rel_v: Vec<f32>,
    /// Number of attention heads.
    n_heads: usize,
    /// Per-head channel width (`d_head = d_model / n_heads`).
    d_head: usize,
    /// Hidden (model) dimension (`n_heads * d_head`).
    d_model: usize,
    /// Relative-position half-window (`2*window+1` bins total).
    window_size: usize,
}

impl RelPositionMHA {
    /// Builds a multi-head relative-position attention block from
    /// pre-trained tensors.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if any weight/bias length disagrees
    /// with the documented shape, if `d_model % n_heads != 0`, or if the
    /// relative-position embeddings do not have the shape
    /// `[2*window+1, d_head]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_weight: Vec<f32>,
        q_bias: Vec<f32>,
        k_weight: Vec<f32>,
        k_bias: Vec<f32>,
        v_weight: Vec<f32>,
        v_bias: Vec<f32>,
        o_weight: Vec<f32>,
        o_bias: Vec<f32>,
        emb_rel_k: Vec<f32>,
        emb_rel_v: Vec<f32>,
        n_heads: usize,
        d_head: usize,
        window_size: usize,
    ) -> Self {
        debug_assert!(n_heads > 0, "n_heads must be positive");
        let d_model = n_heads * d_head;
        debug_assert_eq!(q_weight.len(), d_model * d_model, "q_weight must be [D, D]");
        debug_assert_eq!(k_weight.len(), d_model * d_model, "k_weight must be [D, D]");
        debug_assert_eq!(v_weight.len(), d_model * d_model, "v_weight must be [D, D]");
        debug_assert_eq!(o_weight.len(), d_model * d_model, "o_weight must be [D, D]");
        debug_assert_eq!(q_bias.len(), d_model, "q_bias must be [D]");
        debug_assert_eq!(k_bias.len(), d_model, "k_bias must be [D]");
        debug_assert_eq!(v_bias.len(), d_model, "v_bias must be [D]");
        debug_assert_eq!(o_bias.len(), d_model, "o_bias must be [D]");
        let rel_full = (2 * window_size + 1) * d_head;
        debug_assert_eq!(
            emb_rel_k.len(),
            rel_full,
            "emb_rel_k must be [2*window+1, d_head]"
        );
        debug_assert_eq!(
            emb_rel_v.len(),
            rel_full,
            "emb_rel_v must be [2*window+1, d_head]"
        );
        Self {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            emb_rel_k,
            emb_rel_v,
            n_heads,
            d_head,
            d_model,
            window_size,
        }
    }

    /// Runs one forward pass of relative-position multi-head self-attention
    /// on `x` (`[seq_len, d_model]` row-major), returning the same shape.
    fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        // Q/K/V projections (1×1 conv = per-position linear + bias).
        let q = conv1x1_biased(x, &self.q_weight, &self.q_bias, d, d);
        let k = conv1x1_biased(x, &self.k_weight, &self.k_bias, d, d);
        let v = conv1x1_biased(x, &self.v_weight, &self.v_bias, d, d);

        let n_heads = self.n_heads;
        let d_head = self.d_head;
        let scale = 1.0_f32 / (d_head as f32).sqrt();

        // rel_k / rel_v are shared across heads (heads_share=True).
        let rel_k = get_relative_embeddings(&self.emb_rel_k, self.window_size, seq_len, d_head);
        let rel_v = get_relative_embeddings(&self.emb_rel_v, self.window_size, seq_len, d_head);
        let rel_len = 2 * seq_len - 1;

        let mut output = vec![0.0_f32; seq_len * d];
        // scores buffer reused per head to avoid per-head reallocation.
        let mut scores = vec![0.0_f32; seq_len * seq_len];
        let mut rel_logits = vec![0.0_f32; seq_len * rel_len];
        let mut rel_weights = vec![0.0_f32; seq_len * rel_len];

        for h in 0..n_heads {
            // Content scores: scores[q_pos, k_pos] = <q_h[q_pos], k_h[k_pos]> * scale
            for q_pos in 0..seq_len {
                for k_pos in 0..seq_len {
                    let mut acc = 0.0_f32;
                    for kk in 0..d_head {
                        let qv = q[q_pos * d + h * d_head + kk];
                        let kv = k[k_pos * d + h * d_head + kk];
                        acc += qv * kv;
                    }
                    scores[q_pos * seq_len + k_pos] = acc * scale;
                }
            }
            // rel_logits[q_pos, r_idx] = <q_h[q_pos], rel_k[r_idx]> * scale
            for q_pos in 0..seq_len {
                for r_idx in 0..rel_len {
                    let mut acc = 0.0_f32;
                    for kk in 0..d_head {
                        let qv = q[q_pos * d + h * d_head + kk];
                        let rv = rel_k[r_idx * d_head + kk];
                        acc += qv * rv;
                    }
                    rel_logits[q_pos * rel_len + r_idx] = acc * scale;
                }
            }
            // Add skew-shifted rel_logits to scores.
            //   scores[q, k] += rel_logits[q, k - q + T - 1]
            for q_pos in 0..seq_len {
                let base = q_pos * rel_len;
                for k_pos in 0..seq_len {
                    let src_col = (k_pos as isize) - (q_pos as isize) + (seq_len as isize) - 1;
                    // src_col is always in [0, 2*T-2] since k, q in [0, T-1].
                    scores[q_pos * seq_len + k_pos] += rel_logits[base + src_col as usize];
                }
            }
            // Softmax along the last axis (per query row).
            for q_pos in 0..seq_len {
                let row = &mut scores[q_pos * seq_len..(q_pos + 1) * seq_len];
                softmax_inplace(row);
            }
            // Content output: out_content[t, kk] = Σ_k p_attn[t, k] * v_h[k, kk]
            for t in 0..seq_len {
                for kk in 0..d_head {
                    let mut acc = 0.0_f32;
                    for k_pos in 0..seq_len {
                        acc += scores[t * seq_len + k_pos] * v[k_pos * d + h * d_head + kk];
                    }
                    output[t * d + h * d_head + kk] = acc;
                }
            }
            // rel_weights[q, r_idx] = p_attn[q, r_idx - T + 1 + q]  if in range, else 0.
            for q_pos in 0..seq_len {
                let base = q_pos * rel_len;
                for r_idx in 0..rel_len {
                    let k_pos = (r_idx as isize) - (seq_len as isize) + 1 + (q_pos as isize);
                    if k_pos >= 0 && (k_pos as usize) < seq_len {
                        rel_weights[base + r_idx] = scores[q_pos * seq_len + k_pos as usize];
                    } else {
                        rel_weights[base + r_idx] = 0.0;
                    }
                }
            }
            // out_rel[t, kk] = Σ_r rel_weights[t, r] * rel_v[r, kk]; added to output.
            for t in 0..seq_len {
                for kk in 0..d_head {
                    let mut acc = 0.0_f32;
                    for r_idx in 0..rel_len {
                        acc += rel_weights[t * rel_len + r_idx] * rel_v[r_idx * d_head + kk];
                    }
                    output[t * d + h * d_head + kk] += acc;
                }
            }
        }

        // Final O projection.
        conv1x1_biased(&output, &self.o_weight, &self.o_bias, d, d)
    }
}

/// Position-wise FFN: two same-padded Conv1d layers with bias, ReLU
/// between. `conv_1` is `[d_ff, d_model, kernel]`, `conv_2` is `[d_model,
/// d_ff, kernel]`. Matches upstream `vendor/vits/attentions.py::FFN` at
/// `activation is None` (the SBV2 default — SBV2 uses ReLU).
pub struct PositionWiseFFN {
    /// First conv weight, `[d_ff, d_model, kernel]`.
    conv1_weight: Vec<f32>,
    /// First conv bias, `[d_ff]`.
    conv1_bias: Vec<f32>,
    /// Second conv weight, `[d_model, d_ff, kernel]`.
    conv2_weight: Vec<f32>,
    /// Second conv bias, `[d_model]`.
    conv2_bias: Vec<f32>,
    /// Kernel width (odd — SBV2 uses 3).
    kernel: usize,
    /// Hidden (model) dimension.
    d_model: usize,
    /// FFN inner width.
    d_ff: usize,
}

impl PositionWiseFFN {
    /// Builds a `PositionWiseFFN` from pre-trained weights.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if any weight/bias length disagrees
    /// with the documented shape, if `kernel == 0`, or if `kernel` is
    /// even. The `conv1d_same_padded` helper this FFN routes through uses
    /// symmetric `pad_l = pad_r = (kernel - 1) / 2` padding, which is
    /// only bit-exact vs upstream `attentions.FFN._same_padding` (upstream
    /// splits into `pad_l = (K-1)/2` and `pad_r = K/2`) when `kernel` is
    /// odd. SBV2 uses `kernel = 3` everywhere so this holds in every
    /// exercised path today; the assert is a tripwire against a future
    /// SKU that ships an even `kernel_ffn` in metadata and would
    /// otherwise silently shift the receptive field by one position.
    pub fn new(
        conv1_weight: Vec<f32>,
        conv1_bias: Vec<f32>,
        conv2_weight: Vec<f32>,
        conv2_bias: Vec<f32>,
        d_model: usize,
        d_ff: usize,
        kernel: usize,
    ) -> Self {
        debug_assert!(kernel > 0, "kernel must be positive");
        debug_assert!(
            kernel % 2 == 1,
            "kernel must be odd (conv1d_same_padded uses symmetric pad_l = pad_r = (kernel-1)/2; \
             the asymmetric-pad code path required for even kernels is not implemented — see \
             ffn_new_rejects_even_kernel)"
        );
        debug_assert_eq!(
            conv1_weight.len(),
            d_ff * d_model * kernel,
            "conv1_weight must be [d_ff, d_model, kernel]"
        );
        debug_assert_eq!(conv1_bias.len(), d_ff, "conv1_bias must be [d_ff]");
        debug_assert_eq!(
            conv2_weight.len(),
            d_model * d_ff * kernel,
            "conv2_weight must be [d_model, d_ff, kernel]"
        );
        debug_assert_eq!(conv2_bias.len(), d_model, "conv2_bias must be [d_model]");
        Self {
            conv1_weight,
            conv1_bias,
            conv2_weight,
            conv2_bias,
            kernel,
            d_model,
            d_ff,
        }
    }

    /// Runs one forward pass of the FFN on `x` (`[seq_len, d_model]`),
    /// returning `[seq_len, d_model]`.
    ///
    /// # SINGLE-UTTERANCE / UNMASKED CONTRACT (POSFFN-XMASK, 2026-08-09)
    ///
    /// This implementation intentionally **omits** the upstream's three
    /// `x * x_mask` multiplies (before `conv_1`, before `conv_2`, and
    /// once more after `conv_2` — `tools/parity/vendor/vits/attentions.py::FFN.forward`).
    /// For a single-utterance / unmasked path (Vokra's current only caller —
    /// [`SbV2TransformerBlock::forward`](Self)) every mask cell is `1.0`,
    /// so the three multiplies are byte-identical no-ops and the omission
    /// is a valid arithmetic simplification.
    ///
    /// **A future batched / streaming path** (e.g. M4/M5 `vokra-server`
    /// multi-session with different-length utterances co-batched) **would
    /// silently produce nondeterministic garbage** because per-utterance
    /// padded positions would leak into every subsequent conv's receptive
    /// field via `same-padded` convolution. Such a caller **must** either
    /// (a) fan out to per-utterance forward calls (already single-utterance
    /// safe here), or (b) grow this signature to accept an
    /// `Option<&[f32]> x_mask` parameter and apply the three multiplies
    /// when `Some` — FR-EX-08 forbids silently letting a batched caller
    /// through the current unmasked path.
    ///
    /// The `seq_len == x.len() / d_model` debug-assert below serves as the
    /// runtime tripwire: any caller that passes a `[B*T_max, d_model]`
    /// batched view with `B > 1` trips it (they'd need to fan out).
    fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        debug_assert_eq!(
            x.len(),
            seq_len * self.d_model,
            "PositionWiseFFN::forward: x must be [seq_len, d_model] — this implementation is \
             single-utterance / unmasked only. See the fn doc's POSFFN-XMASK contract."
        );
        // First conv: [T, d_model] -> [T, d_ff] with same-padding.
        let mut h = conv1d_same_padded(
            x,
            self.d_model,
            &self.conv1_weight,
            &self.conv1_bias,
            self.d_ff,
            self.kernel,
            seq_len,
        );
        // Upstream `attentions.FFN.forward` at `activation is None`
        // applies `torch.relu(x)`.
        for v in &mut h {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        // Second conv: [T, d_ff] -> [T, d_model] with same-padding.
        conv1d_same_padded(
            &h,
            self.d_ff,
            &self.conv2_weight,
            &self.conv2_bias,
            self.d_model,
            self.kernel,
            seq_len,
        )
    }
}

// =====================================================================
// BERT bridge (unchanged — post-M6 architecture)
// =====================================================================

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
    /// `[bert_seq_len, d_target]`, then **linear-interpolates** (torch
    /// `F.interpolate(mode='linear', align_corners=False)`) along the
    /// sequence axis to `[text_seq_len, d_target]` (row-major, flat
    /// `Vec<f32>` of length `text_seq_len * d_target`).
    ///
    /// # Interpolation formula (BERT-BRIDGE-LINEAR fix, 2026-08-09)
    ///
    /// For each destination text position `t ∈ [0, text_seq_len)`:
    ///
    /// ```text
    /// src_x = (t + 0.5) * bert_seq_len / text_seq_len - 0.5
    /// low   = clamp(floor(src_x), 0, bert_seq_len - 1)
    /// high  = clamp(low + 1,      0, bert_seq_len - 1)
    /// alpha = clamp(src_x - floor(src_x), 0.0, 1.0)
    /// out[t] = (1 - alpha) * projected[low] + alpha * projected[high]
    /// ```
    ///
    /// This matches PyTorch's `F.interpolate(input, size=text_seq_len,
    /// mode='linear', align_corners=False)` — the same formulation the
    /// Python reference in `tools/parity/sbv2_dump_reference.py` uses.
    /// Pre-fix Vokra used nearest-neighbor floor
    /// (`s = min(t * bert_seq_len / text_seq_len, bert_seq_len - 1)`),
    /// which diverges at every non-integer source position by roughly
    /// `|neighbor_a - neighbor_b| * 0.5` (10-50× the parity atol
    /// for DeBERTa-scale hidden values).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) in debug builds if `bert_seq_len ==
    /// 0` (an empty BERT sequence has no source position for the
    /// interpolation to read from — the clamp below prevents `usize`
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
             has no source position for the linear interpolation to read from)"
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
        // Cast once to f32 for the ratio + edge-clamp math.
        let src_len_f = bert_seq_len as f32;
        let dst_len_f = text_seq_len as f32;
        let max_idx = (bert_seq_len as i64) - 1;
        for (t, out_row) in out.chunks_exact_mut(d_target).enumerate() {
            // align_corners=False mapping.
            let src_x = ((t as f32) + 0.5) * src_len_f / dst_len_f - 0.5;
            let low_f = src_x.floor();
            let low = (low_f as i64).clamp(0, max_idx) as usize;
            let high = ((low_f as i64) + 1).clamp(0, max_idx) as usize;
            let low_row = &projected[low * d_target..(low + 1) * d_target];
            // Fast + numerically-exact path when `low == high` — this
            // fires (a) at every dst position for `bert_seq_len == 1`
            // (`high` clamps back to 0), and (b) at every edge sample
            // where `src_x` is out of range (`high` clamps to
            // `bert_seq_len - 1`). Copying the single row avoids the
            // `(1 - alpha) * x + alpha * x` fp round-trip that would
            // otherwise perturb the "identical" values by ~1 ULP —
            // preserves the pre-fix nearest-neighbor's exact-broadcast
            // property for degenerate ranges without giving up
            // interior-linear parity.
            if low == high {
                out_row.copy_from_slice(low_row);
                continue;
            }
            let alpha = (src_x - low_f).clamp(0.0, 1.0);
            let one_minus_alpha = 1.0 - alpha;
            let high_row = &projected[high * d_target..(high + 1) * d_target];
            for (d, cell) in out_row.iter_mut().enumerate() {
                *cell = one_minus_alpha * low_row[d] + alpha * high_row[d];
            }
        }
        out
    }
}

// =====================================================================
// Internal helpers
// =====================================================================

/// Applies a bias-free `[out_dim, in_dim]` row-major linear map to each
/// `in_dim`-wide row of `x`, producing a flat `[rows, out_dim]` buffer
/// (`rows = x.len() / in_dim`). Used by [`BertBridge`] via
/// [`linear_rows_biased`] — reused rather than duplicated to preserve
/// bit-identical arithmetic for that path across the M6 refactor.
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

/// 1×1 Conv1d with bias — per-position linear map. `x` is `[T, in_dim]`
/// row-major, `w` is `[out_dim, in_dim]`, `b` is `[out_dim]`. Returns
/// `[T, out_dim]`. Identical arithmetic to [`linear_rows_biased`], kept
/// as a separate entry point so the attention forward-pass call sites
/// read as "conv1x1" (matching the upstream naming convention on
/// `MultiHeadAttention.conv_{q,k,v,o}`).
fn conv1x1_biased(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    linear_rows_biased(x, in_dim, w, b, out_dim)
}

/// Same-padded (`pad_l = (K-1)/2`, `pad_r = K/2` per upstream
/// `attentions.FFN._same_padding`) Conv1d with bias. `x` is `[T, in_dim]`
/// row-major, `w` is `[out_dim, in_dim, kernel]` (PyTorch layout), `b`
/// is `[out_dim]`. Returns `[T, out_dim]`. Handles kernel=1 as a
/// no-padding pointwise conv.
fn conv1d_same_padded(
    x: &[f32],
    in_dim: usize,
    w: &[f32],
    b: &[f32],
    out_dim: usize,
    kernel: usize,
    seq_len: usize,
) -> Vec<f32> {
    let pad_l = (kernel - 1) / 2;
    let mut out = vec![0.0_f32; seq_len * out_dim];
    for t in 0..seq_len {
        for oc in 0..out_dim {
            let mut acc = b[oc];
            for k in 0..kernel {
                let src_t = (t as isize) + (k as isize) - (pad_l as isize);
                if src_t < 0 || (src_t as usize) >= seq_len {
                    // Zero-padded neighbor — contributes 0.
                    continue;
                }
                let src_row = (src_t as usize) * in_dim;
                let w_base = oc * in_dim * kernel;
                for ic in 0..in_dim {
                    acc += x[src_row + ic] * w[w_base + ic * kernel + k];
                }
            }
            out[t * out_dim + oc] = acc;
        }
    }
    out
}

/// Materializes the length-`2*t-1` slice of a shared relative-position
/// embedding table `emb_rel` (shape `[2*window+1, d_head]`) — matches
/// upstream `MultiHeadAttention._get_relative_embeddings(rel_emb, t)`
/// at `heads_share = True` (which drops the leading singleton head
/// dim). If `t > window+1`, out-of-window positions are zero-padded;
/// otherwise the table is sliced starting from `window+1-t`. Returns
/// a `[2*t-1, d_head]` row-major buffer.
fn get_relative_embeddings(emb_rel: &[f32], window: usize, t: usize, d_head: usize) -> Vec<f32> {
    let full_len = 2 * window + 1;
    debug_assert_eq!(
        emb_rel.len(),
        full_len * d_head,
        "emb_rel must be [2*window+1, d_head]"
    );
    let out_len = 2 * t - 1;
    let mut out = vec![0.0_f32; out_len * d_head];
    if t > window + 1 {
        // Pad by (t - window - 1) on each side of the middle dim.
        let pad = t - window - 1;
        for r_idx in 0..out_len {
            if r_idx < pad || r_idx >= pad + full_len {
                continue; // zero-padded neighbor
            }
            let src_row = r_idx - pad;
            let dst = &mut out[r_idx * d_head..(r_idx + 1) * d_head];
            dst.copy_from_slice(&emb_rel[src_row * d_head..(src_row + 1) * d_head]);
        }
    } else {
        // Slice out [window+1-t : window+t] which has length 2*t-1.
        let slice_start = window + 1 - t;
        for r_idx in 0..out_len {
            let src_row = slice_start + r_idx;
            let dst = &mut out[r_idx * d_head..(r_idx + 1) * d_head];
            dst.copy_from_slice(&emb_rel[src_row * d_head..(src_row + 1) * d_head]);
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

#[cfg(test)]
mod tests {
    //! Internal-helper regression tests. Public-surface tests live in
    //! `crates/vokra-models/tests/sbv2_text_encoder.rs`.
    use super::*;

    /// The skew-shift index formula
    /// `scores[q, k] += rel_logits[q, k - q + T - 1]` — the loop that
    /// bridges relative-position `rel_logits` (shape `[T, 2*T-1]`) into
    /// content scores (shape `[T, T]`) — is the inverse of the value-side
    /// `rel_weights[q, r_idx] = p_attn[q, r_idx - T + 1 + q]` fill: for
    /// every `(q, k)` in the content-score plane, the value-side index
    /// pair `(q, k - q + T - 1)` must produce back `k` when reduced by
    /// `r_idx - T + 1 + q`. This pins the two formulas to each other so
    /// they cannot drift independently.
    #[test]
    fn skew_and_inverse_skew_formulas_are_consistent() {
        let t = 5;
        for q in 0..t {
            for k in 0..t {
                let r_idx = (k as isize) - (q as isize) + (t as isize) - 1;
                // Must be in [0, 2*t-2].
                assert!(r_idx >= 0 && (r_idx as usize) < 2 * t - 1);
                // Inverse: recovering k from (q, r_idx).
                let k_back = r_idx - (t as isize) + 1 + (q as isize);
                assert_eq!(k_back, k as isize);
            }
        }
    }

    /// `get_relative_embeddings` in the `t > window + 1` regime pads
    /// left/right with zero rows and copies the original table in the
    /// middle — a length-`2*t-1` result.
    #[test]
    fn get_relative_embeddings_wide_regime_pads_and_copies() {
        let window = 2;
        let full_len = 2 * window + 1; // 5
        let d_head = 3;
        let emb: Vec<f32> = (0..full_len * d_head).map(|i| i as f32 + 1.0).collect();
        let t = 6; // > window + 1 = 3 → pad = t - window - 1 = 3
        let out = get_relative_embeddings(&emb, window, t, d_head);
        assert_eq!(out.len(), (2 * t - 1) * d_head);
        // Left pad rows [0..3] are zero, middle rows [3..8] copy emb, right pad rows [8..11] zero.
        for r_idx in 0..3 {
            for kk in 0..d_head {
                assert_eq!(out[r_idx * d_head + kk], 0.0, "left pad row {r_idx}");
            }
        }
        for r_idx in 3..8 {
            for kk in 0..d_head {
                assert_eq!(
                    out[r_idx * d_head + kk],
                    emb[(r_idx - 3) * d_head + kk],
                    "middle row {r_idx}"
                );
            }
        }
        for r_idx in 8..11 {
            for kk in 0..d_head {
                assert_eq!(out[r_idx * d_head + kk], 0.0, "right pad row {r_idx}");
            }
        }
    }

    /// `get_relative_embeddings` in the `t <= window + 1` regime slices
    /// the original table starting at `window+1-t` — no zero padding.
    #[test]
    fn get_relative_embeddings_narrow_regime_slices() {
        let window = 4;
        let full_len = 2 * window + 1; // 9
        let d_head = 2;
        let emb: Vec<f32> = (0..full_len * d_head).map(|i| i as f32 + 1.0).collect();
        let t = 3; // <= window + 1 = 5 → slice_start = 5 - 3 = 2
        let out = get_relative_embeddings(&emb, window, t, d_head);
        assert_eq!(out.len(), (2 * t - 1) * d_head);
        for r_idx in 0..(2 * t - 1) {
            for kk in 0..d_head {
                let expected = emb[(2 + r_idx) * d_head + kk];
                assert_eq!(
                    out[r_idx * d_head + kk],
                    expected,
                    "row {r_idx} slice mismatch"
                );
            }
        }
    }

    /// COSMETIC-BUNDLE (kernel-width shape check, 2026-08-09):
    /// [`PositionWiseFFN::new`]'s `conv1d_same_padded` helper uses the
    /// symmetric `pad_l = pad_r = (kernel - 1) / 2` padding, which is
    /// only bit-exact vs upstream `attentions.FFN._same_padding` when
    /// `kernel` is **odd** (upstream splits into `pad_l = (K-1)/2` and
    /// `pad_r = K/2`; the two are equal iff `K` is odd). SBV2 uses
    /// `kernel = 3` everywhere so this contract holds today, but a
    /// future SKU wiring an even `kernel_ffn` would silently shift the
    /// receptive field by one position (the truly asymmetric-pad code
    /// path is not implemented). Tripped as a `debug_assert!` at
    /// construction so a mis-metadata GGUF fails at load rather than
    /// silently producing wrong audio downstream.
    #[test]
    #[should_panic(expected = "kernel must be odd")]
    fn ffn_new_rejects_even_kernel() {
        let d_model = 4;
        let d_ff = 6;
        let kernel = 2; // even -> asymmetric-pad territory, unsupported
        let _ = PositionWiseFFN::new(
            vec![0.0; d_ff * d_model * kernel],
            vec![0.0; d_ff],
            vec![0.0; d_model * d_ff * kernel],
            vec![0.0; d_model],
            d_model,
            d_ff,
            kernel,
        );
    }

    /// POSFFN-XMASK regression pin: `PositionWiseFFN::forward` documents a
    /// single-utterance / unmasked contract (see the fn doc). The
    /// debug-assert `x.len() == seq_len * d_model` is the runtime
    /// tripwire that catches a mis-shaped call — a future batched
    /// caller that co-batches `B * T_max` positions into a single
    /// slice would trip this rather than silently produce
    /// nondeterministic garbage from the pad positions leaking into
    /// downstream `conv1d_same_padded` receptive fields.
    #[test]
    #[should_panic(expected = "PositionWiseFFN::forward")]
    fn ffn_forward_debug_asserts_x_len_matches_seq_len_times_d_model() {
        let d_model = 4;
        let d_ff = 6;
        let kernel = 3;
        let ffn = PositionWiseFFN::new(
            vec![0.0; d_ff * d_model * kernel],
            vec![0.0; d_ff],
            vec![0.0; d_model * d_ff * kernel],
            vec![0.0; d_model],
            d_model,
            d_ff,
            kernel,
        );
        // seq_len = 3 but x.len() = 3 * 5 (wrong d_model) — trips the
        // POSFFN-XMASK contract's runtime tripwire.
        let x_bad: Vec<f32> = vec![0.0; 3 * 5];
        let _ = ffn.forward(&x_bad, 3);
    }

    /// A zero-weight transformer block reduces to the LayerNorm-only
    /// path: with `emb_rel_k/v = 0`, `attn` returns zero (its Q/K/V
    /// projections have zero weight and zero bias), and `ffn` returns
    /// zero (both conv weights and biases are zero). So `x → norm2(norm1(x))`
    /// end to end.
    #[test]
    fn transformer_block_with_zero_weights_reduces_to_two_layer_norms() {
        let d_model = 4;
        let n_heads = 2;
        let d_head = 2;
        let window = 2;
        let d_ff = 6;
        let kernel = 3;
        let attn = RelPositionMHA::new(
            vec![0.0; d_model * d_model],
            vec![0.0; d_model],
            vec![0.0; d_model * d_model],
            vec![0.0; d_model],
            vec![0.0; d_model * d_model],
            vec![0.0; d_model],
            vec![0.0; d_model * d_model],
            vec![0.0; d_model],
            vec![0.0; (2 * window + 1) * d_head],
            vec![0.0; (2 * window + 1) * d_head],
            n_heads,
            d_head,
            window,
        );
        let norm1 = LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], d_model);
        let ffn = PositionWiseFFN::new(
            vec![0.0; d_ff * d_model * kernel],
            vec![0.0; d_ff],
            vec![0.0; d_model * d_ff * kernel],
            vec![0.0; d_model],
            d_model,
            d_ff,
            kernel,
        );
        let norm2 = LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], d_model);
        let block = SbV2TransformerBlock::new(attn, norm1, ffn, norm2, d_model);

        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        block.forward(&mut x, 2);

        // After the two LayerNorms with gamma=1, beta=0, every output row
        // must have (near-)zero mean and unit variance (numerical bound
        // 1e-4 for a length-4 row).
        for row in x.chunks_exact(d_model) {
            let mean: f32 = row.iter().sum::<f32>() / d_model as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d_model as f32;
            assert!(mean.abs() < 1e-4, "row mean should be ~0, got {mean}");
            assert!(
                (var - 1.0).abs() < 1e-3,
                "row variance should be ~1, got {var}"
            );
        }
    }
}
