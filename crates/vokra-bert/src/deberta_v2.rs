//! DeBERTa v2 encoder — clean-room per arXiv:2006.03654.
//!
//! # References (permissive only)
//!
//! - He, Liu, Gao, Chen 2021 (arXiv:2006.03654)
//! - microsoft/DeBERTa (MIT)
//! - HuggingFace transformers `deberta_v2` (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use vokra_core::gguf::GgufFile;
use vokra_core::VokraError;

/// Precomputed `sqrt(2/π)`, used by the tanh-approximation GELU on
/// [`FfnBlock::forward`]'s per-hidden-unit inner loop. Hoisted out of the
/// loop so the constant is not re-derived per activation. Pinned against
/// runtime `(2.0_f32 / std::f32::consts::PI).sqrt()` by
/// `const_hoist_tests::sqrt_two_over_pi_matches_runtime_within_1_ulp`
/// (drift detector); the value the pre-hoist code produced remains the
/// mathematical reference under `docs/adr/sbv2-libm-strategy.md` §3.
///
/// The literal keeps its full 16-digit form so a reader can recognise it as
/// `sqrt(2/π)` at a glance; f32 discards the tail past ~7 significant digits
/// via round-to-nearest, so `SQRT_TWO_OVER_PI.to_bits()` is what the drift
/// detector actually compares.
#[allow(clippy::excessive_precision)]
const SQRT_TWO_OVER_PI: f32 = 0.797_884_560_802_865_4_f32;

/// Log-scale relative position bucket per DeBERTa v2 (§3.2, "disentangled
/// attention"). Positions closer to `q` get finer buckets; positions far
/// away get log-spaced buckets that saturate at `bucket_size - 1`.
///
/// # Arguments
/// - `q_pos`, `k_pos`: absolute positions of query and key
/// - `bucket_size`: number of buckets (typically 256)
/// - `max_dist`: distance beyond which all positions share the last bucket
///
/// # Algorithm (arXiv:2006.03654 eq. after §3.2)
///
/// rel = q_pos - k_pos
/// sign = sign(rel)
/// mid = bucket_size / 2
/// if |rel| < mid: bucket = mid + rel   # linear near-region
/// else: bucket = mid + sign * (mid + log(|rel|/mid) / log(max_dist/mid) * mid)
///                            .clamp(0, bucket_size - 1)
pub fn relative_position_bucket(q_pos: i32, k_pos: i32, bucket_size: i32, max_dist: i32) -> i32 {
    let rel = q_pos - k_pos;

    // Special case: same position → bucket 0
    if rel == 0 {
        return 0;
    }

    let sign = if rel > 0 { 1 } else { -1 };
    let mid = bucket_size / 2;
    let abs_rel = rel.abs();
    if abs_rel < mid {
        (mid + rel).clamp(0, bucket_size - 1)
    } else {
        let log_ratio = (abs_rel as f32 / mid as f32).ln() / (max_dist as f32 / mid as f32).ln();
        let far = mid + (log_ratio * mid as f32) as i32;
        let bucketed = mid + sign * far.min(mid);
        bucketed.clamp(0, bucket_size - 1)
    }
}

/// Weight bundle for one disentangled attention layer.
/// All matrices are row-major, stored as (out_dim × in_dim) flattened.
///
/// # Position-aware biases (`bq_pos` / `bk_pos`)
///
/// Upstream DeBERTa v2/v3 `DisentangledSelfAttention` (HuggingFace
/// transformers `deberta_v2`, Apache-2.0) applies a distinct
/// `nn.Linear` (with bias) to the relative-position embeddings when
/// `share_att_key=False`. When `share_att_key=True` (the standard for
/// `ku-nlp/deberta-v2-large-japanese-char-wwm`), the same content Q/K
/// projections — biases included — are reused for the position path;
/// upstream carries only one `query_proj.bias` / `key_proj.bias` and no
/// separate `pos_query_proj.bias` / `pos_key_proj.bias`.
///
/// `bq_pos` / `bk_pos` are [`Option`] so a GGUF built from either
/// upstream style loads cleanly:
///
/// - **share_att_key=True** (or any pre-WP-15 GGUF): `None`.
///   [`DisentangledAttention::forward`] transparently falls back to
///   `bq` / `bk` (semantically identical to what upstream computes —
///   the *same* projection with its *same* bias).
/// - **share_att_key=False**: `Some(...)` carries the distinct
///   position-projection bias; forward uses it in place of `bq` / `bk`.
///
/// The Rust `wq_pos` / `wk_pos` weight fields are always populated
/// (either by verbatim upstream tensors under `share_att_key=False` or
/// by the converter duplicating `wq` / `wk` under `share_att_key=True`),
/// so *weight* symmetry is enforced at load time; the *bias* is the one
/// piece the loader honestly cannot always fill.
pub struct AttnWeights {
    pub wq: Vec<f32>, // [d_model, d_model]
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wq_pos: Vec<f32>, // position-aware Q projection
    pub wk_pos: Vec<f32>, // position-aware K projection
    pub w_out: Vec<f32>,
    pub pos_embed: Vec<f32>, // [n_pos_buckets, d_model]
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub bout: Vec<f32>,
    /// Optional distinct bias for the position-aware Q projection. See
    /// the struct-level "Position-aware biases" section for the
    /// two-config-style contract.
    pub bq_pos: Option<Vec<f32>>,
    /// Optional distinct bias for the position-aware K projection. See
    /// the struct-level "Position-aware biases" section for the
    /// two-config-style contract.
    pub bk_pos: Option<Vec<f32>>,
}

/// DeBERTa v2 disentangled attention (arXiv:2006.03654 §3.2).
///
/// Decomposes attention score into three terms and drops P2P (v2's
/// own simplification vs. v1):
///
/// - **C2C** (content-to-content): standard `Q · K^T`.
/// - **C2P** (content-to-position): `Q_i · K_pos[bucket(i, j)]`.
/// - **P2C** (position-to-content): `Q_pos[bucket(j, i)] · K_j`.
///
/// Final score = `(C2C + C2P + P2C) / sqrt(3 * head_dim)`.
pub struct DisentangledAttention {
    w: AttnWeights,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    n_pos_buckets: i32,
    max_pos_dist: i32,
}

impl DisentangledAttention {
    pub fn new(
        w: AttnWeights,
        d_model: usize,
        n_heads: usize,
        head_dim: usize,
        n_pos_buckets: i32,
        max_pos_dist: i32,
    ) -> Self {
        assert_eq!(d_model, n_heads * head_dim);
        Self {
            w,
            d_model,
            n_heads,
            head_dim,
            n_pos_buckets,
            max_pos_dist,
        }
    }

    /// Forward pass. `hidden` is [seq_len, d_model] flat.
    /// Returns [seq_len, d_model] flat.
    ///
    /// Score = (C2C + C2P + P2C) / sqrt(3 * head_dim) per DeBERTa v2 scaling.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        assert_eq!(hidden.len(), seq_len * self.d_model);
        // 1. Q, K, V from content
        let q = self.matmul_bias(hidden, &self.w.wq, &self.w.bq, seq_len);
        let k = self.matmul_bias(hidden, &self.w.wk, &self.w.bk, seq_len);
        let v = self.matmul_bias(hidden, &self.w.wv, &self.w.bv, seq_len);
        // 2. position-aware Q_p, K_p from pos_embed (fresh per relative pos).
        // `bq_pos` / `bk_pos` carry the distinct position-projection bias
        // when upstream is share_att_key=False; fall back to `bq` / `bk`
        // for share_att_key=True (or pre-WP-15 GGUFs that never stamped a
        // separate tensor) — see [`AttnWeights`] "Position-aware biases".
        let bq_pos = self.w.bq_pos.as_deref().unwrap_or(self.w.bq.as_slice());
        let bk_pos = self.w.bk_pos.as_deref().unwrap_or(self.w.bk.as_slice());
        let q_p = self.matmul_bias(
            &self.w.pos_embed,
            &self.w.wq_pos,
            bq_pos,
            self.n_pos_buckets as usize,
        );
        let k_p = self.matmul_bias(
            &self.w.pos_embed,
            &self.w.wk_pos,
            bk_pos,
            self.n_pos_buckets as usize,
        );

        // 3. Multi-head split
        let scale = 1.0 / ((3 * self.head_dim) as f32).sqrt();
        let mut out = vec![0.0_f32; seq_len * self.d_model];

        for head in 0..self.n_heads {
            let ho = head * self.head_dim;
            // 3a. Compute scores per (q_i, k_j)
            let mut scores = vec![0.0_f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // C2C
                    let mut s = 0.0;
                    for d in 0..self.head_dim {
                        s += q[i * self.d_model + ho + d] * k[j * self.d_model + ho + d];
                    }
                    // C2P: q_i · k_p[bucket(i,j)]
                    let bucket = relative_position_bucket(
                        i as i32,
                        j as i32,
                        self.n_pos_buckets,
                        self.max_pos_dist,
                    ) as usize;
                    for d in 0..self.head_dim {
                        s += q[i * self.d_model + ho + d] * k_p[bucket * self.d_model + ho + d];
                    }
                    // P2C: q_p[bucket(j,i)] · k_j (rev direction)
                    let bucket_rev = relative_position_bucket(
                        j as i32,
                        i as i32,
                        self.n_pos_buckets,
                        self.max_pos_dist,
                    ) as usize;
                    for d in 0..self.head_dim {
                        s += q_p[bucket_rev * self.d_model + ho + d] * k[j * self.d_model + ho + d];
                    }
                    scores[i * seq_len + j] = s * scale;
                }
            }
            // 3b. Softmax per row
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let max_v = scores[row_start..row_start + seq_len]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for j in 0..seq_len {
                    scores[row_start + j] = (scores[row_start + j] - max_v).exp();
                    sum += scores[row_start + j];
                }
                for j in 0..seq_len {
                    scores[row_start + j] /= sum;
                }
            }
            // 3c. Weighted V sum → per-head output
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut acc = 0.0;
                    for j in 0..seq_len {
                        acc += scores[i * seq_len + j] * v[j * self.d_model + ho + d];
                    }
                    out[i * self.d_model + ho + d] = acc;
                }
            }
        }
        // 4. Output projection
        self.matmul_bias(&out, &self.w.w_out, &self.w.bout, seq_len)
    }

    /// Naive matmul: y[i,o] = sum_d x[i,d] * w[o,d] + b[o]. Row-major.
    /// (Optimization = Stage B follow-up if hot path shows up. Correctness first.)
    fn matmul_bias(&self, x: &[f32], w: &[f32], b: &[f32], n_rows: usize) -> Vec<f32> {
        let d_in = self.d_model;
        let d_out = self.d_model;
        let mut y = vec![0.0_f32; n_rows * d_out];
        for i in 0..n_rows {
            for o in 0..d_out {
                let mut acc = b[o];
                for d in 0..d_in {
                    acc += x[i * d_in + d] * w[o * d_in + d];
                }
                y[i * d_out + o] = acc;
            }
        }
        y
    }
}

/// FFN block: x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)
/// per BERT/DeBERTa convention.
pub struct FfnBlock {
    w1: Vec<f32>,
    b1: Vec<f32>, // [d_ff, d_model]
    w2: Vec<f32>,
    b2: Vec<f32>, // [d_model, d_ff]
    d_model: usize,
    d_ff: usize,
}

impl FfnBlock {
    pub fn new(
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        d_model: usize,
        d_ff: usize,
    ) -> Self {
        assert_eq!(w1.len(), d_ff * d_model);
        assert_eq!(w2.len(), d_model * d_ff);
        Self {
            w1,
            b1,
            w2,
            b2,
            d_model,
            d_ff,
        }
    }

    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        // Linear 1
        let mut h = vec![0.0_f32; seq_len * self.d_ff];
        for i in 0..seq_len {
            for o in 0..self.d_ff {
                let mut a = self.b1[o];
                for d in 0..self.d_model {
                    a += x[i * self.d_model + d] * self.w1[o * self.d_model + d];
                }
                // GELU (Hendrycks approx): 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let g = 0.5 * a * (1.0 + (SQRT_TWO_OVER_PI * (a + 0.044715 * a * a * a)).tanh());
                h[i * self.d_ff + o] = g;
            }
        }
        // Linear 2
        let mut y = vec![0.0_f32; seq_len * self.d_model];
        for i in 0..seq_len {
            for o in 0..self.d_model {
                let mut a = self.b2[o];
                for d in 0..self.d_ff {
                    a += h[i * self.d_ff + d] * self.w2[o * self.d_ff + d];
                }
                y[i * self.d_model + o] = a;
            }
        }
        y
    }
}

/// Per-row LayerNorm: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(gamma: Vec<f32>, beta: Vec<f32>, eps: f32) -> Self {
        assert_eq!(gamma.len(), beta.len());
        Self { gamma, beta, eps }
    }

    /// `x` is [seq_len, d] flat. Normalizes each row independently.
    pub fn forward(&self, x: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
        assert_eq!(x.len(), seq_len * d);
        assert_eq!(self.gamma.len(), d);
        let mut y = vec![0.0_f32; x.len()];
        for i in 0..seq_len {
            let row = &x[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + self.eps).sqrt();
            for j in 0..d {
                y[i * d + j] = (row[j] - mean) * inv * self.gamma[j] + self.beta[j];
            }
        }
        y
    }
}

/// One DeBERTa v2 transformer block, **post-norm** order — matches
/// HuggingFace transformers `DebertaV2SelfOutput.forward` +
/// `DebertaV2Output.forward` (both apply LayerNorm AFTER the residual
/// add: `hidden_states = LayerNorm(hidden_states + input_tensor)`).
///
/// The converter maps upstream `attention.output.LayerNorm.*` → [`ln1`]
/// and `output.LayerNorm.*` → [`ln2`] (see
/// `crates/vokra-convert/src/models/deberta_v2.rs`); [`ln1`] fires after
/// the attention residual, [`ln2`] after the FFN residual, matching
/// upstream verbatim.
///
/// Historical: prior to the 2026-08-09 parity-CI fix (run 31314913038,
/// `bert_hidden_ja` max |Δ| = 1.368e2) this block computed
/// `h = hidden + attn(ln1(hidden))` — pre-norm — while the converter
/// loaded the post-norm tensors into the pre-norm slots, so weights were
/// correct but the graph topology drifted. Accumulated per-op drift over
/// 24 large-variant layers reached the observed magnitude. See
/// `crates/vokra-bert/tests/deberta_v2_synthetic.rs::encoder_layer_forward_uses_post_norm`
/// for the pin.
///
/// [`ln1`]: EncoderLayer::ln1
/// [`ln2`]: EncoderLayer::ln2
pub struct EncoderLayer {
    pub attn: DisentangledAttention,
    pub ffn: FfnBlock,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
}

impl EncoderLayer {
    /// Post-norm order (matches HuggingFace `DebertaV2SelfOutput.forward`
    /// + `DebertaV2Output.forward`):
    ///
    /// ```text
    /// h = ln1(hidden + attn(hidden))
    /// y = ln2(h      + ffn(h))
    /// ```
    ///
    /// LayerNorm fires **after** the residual add — do not confuse with
    /// the pre-norm variant `h = hidden + attn(ln1(hidden))` used by
    /// GPT-2 / LLaMA family models. See the struct-level doc for the
    /// parity-CI fix history and the citation-pin test.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.ln1.gamma.len();
        // Attention path: attn(hidden) + residual, then LayerNorm.
        let attn_out = self.attn.forward(hidden, seq_len);
        let mut h = vec![0.0_f32; hidden.len()];
        for i in 0..hidden.len() {
            h[i] = hidden[i] + attn_out[i];
        }
        let h = self.ln1.forward(&h, seq_len, d);
        // FFN path: ffn(h) + residual, then LayerNorm.
        let ffn_out = self.ffn.forward(&h, seq_len);
        let mut y = vec![0.0_f32; hidden.len()];
        for i in 0..hidden.len() {
            y[i] = h[i] + ffn_out[i];
        }
        self.ln2.forward(&y, seq_len, d)
    }
}

/// Full DeBERTa v2 encoder: token embedding lookup → embed LayerNorm →
/// N-layer transformer stack.
pub struct DebertaV2Encoder {
    layers: Vec<EncoderLayer>,
    embed: Vec<f32>, // [vocab, d_model]
    embed_ln: LayerNorm,
    d_model: usize,
    vocab_size: usize,
}

impl DebertaV2Encoder {
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let seq_len = ids.len();
        let mut hidden = vec![0.0_f32; seq_len * self.d_model];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < self.vocab_size,
                "token id {id} out of vocab {}",
                self.vocab_size
            );
            for d in 0..self.d_model {
                hidden[i * self.d_model + d] = self.embed[id * self.d_model + d];
            }
        }
        hidden = self.embed_ln.forward(&hidden, seq_len, self.d_model);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }
        hidden
    }

    pub fn get_d_model(&self) -> usize {
        self.d_model
    }

    /// Test-only probe (`#[doc(hidden)]`, WP-15): reports whether layer
    /// `i` loaded a distinct `bq_pos` / `bk_pos` bias tensor for the
    /// disentangled position-aware Q/K projections. Panics on
    /// out-of-range `i`. Used by `from_gguf` loader tests to prove the
    /// optional bias load path fires when the tensor is present.
    #[doc(hidden)]
    pub fn probe_layer_has_pos_biases(&self, i: usize) -> (bool, bool) {
        let attn = &self.layers[i].attn;
        (attn.w.bq_pos.is_some(), attn.w.bk_pos.is_some())
    }

    /// Builds a `DebertaV2Encoder` with deterministic synthetic weights, for
    /// structure/shape tests only (no real checkpoint involved).
    #[doc(hidden)]
    pub fn synthetic_for_test(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        vocab: usize,
        n_pos_buckets: i32,
    ) -> Self {
        let head_dim = d_model / n_heads;
        let make_layer = || {
            let w = AttnWeights {
                wq: vec![0.01; d_model * d_model],
                wk: vec![0.01; d_model * d_model],
                wv: vec![0.01; d_model * d_model],
                wq_pos: vec![0.01; d_model * d_model],
                wk_pos: vec![0.01; d_model * d_model],
                w_out: vec![0.01; d_model * d_model],
                pos_embed: vec![0.001; n_pos_buckets as usize * d_model],
                bq: vec![0.0; d_model],
                bk: vec![0.0; d_model],
                bv: vec![0.0; d_model],
                bout: vec![0.0; d_model],
                bq_pos: None,
                bk_pos: None,
            };
            EncoderLayer {
                attn: DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets, 512),
                ffn: FfnBlock::new(
                    vec![0.01; 4 * d_model * d_model],
                    vec![0.0; 4 * d_model],
                    vec![0.01; d_model * 4 * d_model],
                    vec![0.0; d_model],
                    d_model,
                    4 * d_model,
                ),
                ln1: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
                ln2: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
            }
        };
        Self {
            layers: (0..n_layers).map(|_| make_layer()).collect(),
            embed: vec![0.01; vocab * d_model],
            embed_ln: LayerNorm::new(vec![1.0; d_model], vec![0.0; d_model], 1e-7),
            d_model,
            vocab_size: vocab,
        }
    }
}

impl DebertaV2Encoder {
    /// Loads a `DebertaV2Encoder` from a GGUF file written by the SBV2
    /// converter.
    ///
    /// # Metadata keys (`vokra.bert.deberta_v2.*`)
    ///
    /// - `n_layers` (required), `vocab_size` (required)
    /// - `d_model` (default 1024), `n_heads` (default 16),
    ///   `n_pos_buckets` (default 512), `max_pos_dist` (default 512)
    ///
    /// # Tensor names
    ///
    /// - `bert.embed.weight`, `bert.embed.ln.{gamma,beta}`
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,wq_pos,wk_pos,w_out,pos_embed}.weight`
    /// - `bert.encoder.layer.<i>.attn.{wq,wk,wv,w_out}.bias`
    /// - `bert.encoder.layer.<i>.attn.{wq_pos,wk_pos}.bias` — **optional**
    ///   (WP-15). Loaded into [`AttnWeights::bq_pos`] / [`AttnWeights::bk_pos`]
    ///   when present; when absent, forward falls back to the content
    ///   biases `bq` / `bk` (backward-compat with pre-WP-15 GGUFs and
    ///   upstream `share_att_key=True` configs — see the [`AttnWeights`]
    ///   struct-level "Position-aware biases" section).
    /// - `bert.encoder.layer.<i>.ffn.{w1,w2}.{weight,bias}`
    /// - `bert.encoder.layer.<i>.ln{1,2}.{gamma,beta}`
    pub fn from_gguf(g: &GgufFile) -> Result<Self, VokraError> {
        let meta_u32 =
            |key: &str| -> Option<u32> { g.get(key).and_then(|v| v.as_u64()).map(|u| u as u32) };
        let require_u32 = |key: &str| -> Result<u32, VokraError> {
            meta_u32(key)
                .ok_or_else(|| VokraError::ModelLoad(format!("missing GGUF metadata key: {key}")))
        };

        let n_layers = require_u32("vokra.bert.deberta_v2.n_layers")? as usize;
        let d_model = meta_u32("vokra.bert.deberta_v2.d_model").unwrap_or(1024) as usize;
        let n_heads = meta_u32("vokra.bert.deberta_v2.n_heads").unwrap_or(16) as usize;
        let vocab_size = require_u32("vokra.bert.deberta_v2.vocab_size")? as usize;
        let n_pos_buckets = meta_u32("vokra.bert.deberta_v2.n_pos_buckets").unwrap_or(512) as i32;
        let max_pos_dist = meta_u32("vokra.bert.deberta_v2.max_pos_dist").unwrap_or(512) as i32;

        if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
            return Err(VokraError::ModelLoad(format!(
                "vokra.bert.deberta_v2: d_model ({d_model}) not divisible by n_heads ({n_heads})"
            )));
        }
        let head_dim = d_model / n_heads;

        let load_tensor_f32 = |name: &str| -> Result<Vec<f32>, VokraError> {
            g.tensor_f32(name)
                .map_err(|e| VokraError::ModelLoad(format!("{name}: {e}")))
        };
        // WP-15: `wq_pos.bias` / `wk_pos.bias` are optional (see
        // `AttnWeights` "Position-aware biases"). `tensor_info` probes
        // presence without incurring a Read error; only tensors that
        // actually exist are load-attempted, and any dtype / shape error
        // during the actual load still surfaces loudly (FR-EX-08).
        let load_optional_tensor_f32 = |name: &str| -> Result<Option<Vec<f32>>, VokraError> {
            if g.tensor_info(name).is_some() {
                Ok(Some(load_tensor_f32(name)?))
            } else {
                Ok(None)
            }
        };

        let embed = load_tensor_f32("bert.embed.weight")?;
        let embed_ln = LayerNorm::new(
            load_tensor_f32("bert.embed.ln.gamma")?,
            load_tensor_f32("bert.embed.ln.beta")?,
            1e-7,
        );

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("bert.encoder.layer.{i}");
            let w = AttnWeights {
                wq: load_tensor_f32(&format!("{p}.attn.wq.weight"))?,
                wk: load_tensor_f32(&format!("{p}.attn.wk.weight"))?,
                wv: load_tensor_f32(&format!("{p}.attn.wv.weight"))?,
                wq_pos: load_tensor_f32(&format!("{p}.attn.wq_pos.weight"))?,
                wk_pos: load_tensor_f32(&format!("{p}.attn.wk_pos.weight"))?,
                w_out: load_tensor_f32(&format!("{p}.attn.w_out.weight"))?,
                pos_embed: load_tensor_f32(&format!("{p}.attn.pos_embed.weight"))?,
                bq: load_tensor_f32(&format!("{p}.attn.wq.bias"))?,
                bk: load_tensor_f32(&format!("{p}.attn.wk.bias"))?,
                bv: load_tensor_f32(&format!("{p}.attn.wv.bias"))?,
                bout: load_tensor_f32(&format!("{p}.attn.w_out.bias"))?,
                bq_pos: load_optional_tensor_f32(&format!("{p}.attn.wq_pos.bias"))?,
                bk_pos: load_optional_tensor_f32(&format!("{p}.attn.wk_pos.bias"))?,
            };
            let ffn = FfnBlock::new(
                load_tensor_f32(&format!("{p}.ffn.w1.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w1.bias"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.weight"))?,
                load_tensor_f32(&format!("{p}.ffn.w2.bias"))?,
                d_model,
                4 * d_model,
            );
            let ln1 = LayerNorm::new(
                load_tensor_f32(&format!("{p}.ln1.gamma"))?,
                load_tensor_f32(&format!("{p}.ln1.beta"))?,
                1e-7,
            );
            let ln2 = LayerNorm::new(
                load_tensor_f32(&format!("{p}.ln2.gamma"))?,
                load_tensor_f32(&format!("{p}.ln2.beta"))?,
                1e-7,
            );
            layers.push(EncoderLayer {
                attn: DisentangledAttention::new(
                    w,
                    d_model,
                    n_heads,
                    head_dim,
                    n_pos_buckets,
                    max_pos_dist,
                ),
                ffn,
                ln1,
                ln2,
            });
        }

        Ok(Self {
            layers,
            embed,
            embed_ln,
            d_model,
            vocab_size,
        })
    }
}

#[cfg(test)]
mod const_hoist_tests {
    use super::SQRT_TWO_OVER_PI;

    /// Drift detector for the hoisted `sqrt(2/π)` constant used by the
    /// tanh-approximation GELU on the FFN hot path (`FfnBlock::forward`).
    ///
    /// The runtime side re-derives the value from the same primitives the
    /// pre-hoist code used (`f32::sqrt` on `2.0 / std::f32::consts::PI`); if a
    /// future edit ever changes the literal by more than one f32 ULP the
    /// hoisted fast path would silently diverge from the mathematically-
    /// intended value. 1 ULP = adjacent f32 bit patterns for positive finite
    /// values, so a `to_bits` distance of at most 1 is the tightest
    /// implementable bound.
    ///
    /// See `docs/adr/sbv2-libm-strategy.md` §3.1 for the parity contract this
    /// pin defends (host-libm sqrt is the reference implementation, not any
    /// vendored transcendental).
    #[test]
    fn sqrt_two_over_pi_matches_runtime_within_1_ulp() {
        let runtime = (2.0_f32 / std::f32::consts::PI).sqrt();
        let hoisted_bits = SQRT_TWO_OVER_PI.to_bits() as i64;
        let runtime_bits = runtime.to_bits() as i64;
        let ulp_distance = (hoisted_bits - runtime_bits).unsigned_abs();
        assert!(
            ulp_distance <= 1,
            "SQRT_TWO_OVER_PI = {SQRT_TWO_OVER_PI:e} (bits {:#x}), \
             runtime = {runtime:e} (bits {:#x}), differ by {ulp_distance} ULP",
            SQRT_TWO_OVER_PI.to_bits(),
            runtime.to_bits(),
        );
    }
}
