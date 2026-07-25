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
        // 2. position-aware Q_p, K_p from pos_embed (fresh per relative pos)
        let q_p = self.matmul_bias(
            &self.w.pos_embed,
            &self.w.wq_pos,
            &self.w.bq,
            self.n_pos_buckets as usize,
        );
        let k_p = self.matmul_bias(
            &self.w.pos_embed,
            &self.w.wk_pos,
            &self.w.bk,
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
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                let g = 0.5 * a * (1.0 + (c * (a + 0.044715 * a * a * a)).tanh());
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
