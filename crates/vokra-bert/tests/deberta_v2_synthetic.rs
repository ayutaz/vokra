//! DeBERTa v2 synthetic-weight structure tests.
//! Clean-room per arXiv:2006.03654 + HF transformers deberta_v2 (Apache-2.0).

use vokra_bert::deberta_v2::{
    relative_position_bucket, AttnWeights, DebertaV2Encoder, DisentangledAttention, EncoderLayer,
    FfnBlock, LayerNorm,
};

/// Root-cause pin (parity-CI HEAD a4cf654, `bert_hidden_ja` residual
/// 11.15 after Wave-1 R2 rel_embeddings pre-normalize fix): `q = k` must
/// map to the **middle** bucket (= `att_span` = `bucket_size / 2`), NOT
/// to bucket 0. This matches HuggingFace transformers
/// `make_log_bucket_position` + `disentangled_attention_bias` verbatim:
///
/// ```python
/// # transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py
/// # make_log_bucket_position(relative_pos=0, bucket_size, max_position):
/// #   sign = 0
/// #   mid = bucket_size // 2
/// #   abs_pos = mid - 1  (because rel is in (-mid, mid))
/// #   bucket_pos = relative_pos (=0) since abs_pos <= mid
/// # disentangled_attention_bias:
/// #   c2p_pos = clamp(relative_pos + att_span, 0, 2*att_span - 1)
/// #          = clamp(0 + att_span, ...) = att_span
/// ```
/// (`att_span` in HF == `bucket_size` == our `bucket_size` argument's
/// half — the Rust API packs the `+ att_span` shift INTO the bucket
/// function itself so callers get a positive-only index into
/// `pos_embed`.)
///
/// The pre-fix `if rel == 0 { return 0; }` special case in
/// [`vokra_bert::deberta_v2::relative_position_bucket`] returned bucket
/// 0 for the diagonal — the wrong row of the learned `pos_embed`. For a
/// 24-layer DeBERTa v2 large with seq_len = 5 this mis-picks the diagonal
/// on every attention row of every layer, driving the ~11.15 residual
/// the previous agent's Layer-Bisect harness was landed to localise
/// (see `tests/parity_deberta_v2_layer_bisect.rs`).
#[test]
fn bucket_for_same_position_is_att_span() {
    // For any `bucket_size`, `q == k` (rel = 0) must map to `bucket_size / 2`
    // — the "middle" bucket. HuggingFace's `make_log_bucket_position`
    // returns 0 for rel=0, then `+ att_span` offset in
    // `disentangled_attention_bias` shifts it to `att_span` = `bucket_size / 2`
    // in our unified API.
    assert_eq!(relative_position_bucket(5, 5, 256, 512), 128);
    assert_eq!(relative_position_bucket(0, 0, 8, 8), 4);
    // For the JA config: n_pos_buckets=512 → mid=256.
    assert_eq!(relative_position_bucket(3, 3, 512, 512), 256);
    assert_eq!(relative_position_bucket(0, 0, 512, 512), 256);
}

#[test]
fn bucket_symmetric_around_zero() {
    // relative pos -1 と +1 は異なる bucket に分かれる
    let b_pos = relative_position_bucket(6, 5, 256, 512); // rel = +1
    let b_neg = relative_position_bucket(5, 6, 256, 512); // rel = -1
    assert_ne!(b_pos, b_neg);
}

#[test]
fn bucket_saturates_at_max_dist() {
    // rel >> max_dist → bucket = bucket_size - 1（最大）
    let b_far = relative_position_bucket(0, 10_000, 256, 512);
    let b_last = relative_position_bucket(0, 100_000, 256, 512);
    assert_eq!(b_far, b_last);
}

/// Root-cause pin (parity-CI HEAD a4cf654, `bert_hidden_ja` residual
/// 11.15 after Wave-1 R2 rel_embeddings pre-normalize fix, companion to
/// [`bucket_for_same_position_is_att_span`]): the P2C term of the
/// disentangled attention MUST index `pos_embed` (`Q^p`) using the SAME
/// bucket as C2P — `bucket(i, j) = mid + (i - j)` — NOT the sign-swapped
/// `bucket(j, i) = mid + (j - i)`.
///
/// This matches HuggingFace transformers
/// `disentangled_attention_bias`'s effective indexing (after the
/// `p2c_pos = clamp(-r_pos + att_span, ...)` gather + `.transpose(-1,-2)`
/// dance):
///
/// ```python
/// # transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py
/// # p2c contribution to score[b, i, j] =
/// #   key_layer[b, j, :] · pos_query_layer[b, (i - j) + att_span, :]
/// ```
///
/// and matches microsoft/DeBERTa `disentangled_attention.py` verbatim
/// (which reuses `c2p_pos` for the P2C gather with `dim=-2` — clearer
/// evidence the two terms share the SAME bucket index):
///
/// ```python
/// # microsoft/DeBERTa/DeBERTa/deberta/disentangled_attention.py
/// # p2c_att = torch.bmm(pos_query_layer * scale, key_layer.transpose(-1, -2))
/// # p2c_att = torch.gather(p2c_att, dim=-2, index=c2p_pos)   # same c2p_pos
/// # score += p2c_att
/// ```
///
/// The pre-fix `bucket_rev = relative_position_bucket(j, i, ...)` call in
/// [`DisentangledAttention::forward`] picked the wrong `Q^p` row for
/// every off-diagonal (i, j) pair — for seq_len = 5 JA this is 20 wrong
/// row lookups per attention head per layer, driving a large fraction
/// of the ~11.15 residual alongside the diagonal `rel == 0` bug pinned
/// by [`bucket_for_same_position_is_att_span`].
///
/// Constructive proof: two identical attention modules, one with the
/// current (buggy) P2C indexing and one with the fixed indexing, must
/// produce **different** outputs on a non-symmetric input (a purely
/// symmetric input like all-ones can't distinguish because the bug is
/// symmetric in i and j — i.e., `pos_embed[mid + (i-j)]` and
/// `pos_embed[mid + (j-i)]` differ IFF the position embeddings are
/// asymmetric across `mid`, which non-uniform embeddings guarantee).
///
/// This test is a **behavior differential** against a scenario the
/// bugged path is architecturally forced to compute the same as the
/// fixed path — impossible without either (a) all-zero position
/// embeddings or (b) position embeddings that are perfectly
/// symmetric around bucket `mid`. Neither the JA nor the EN checkpoint
/// satisfies (a) or (b); the test setup uses asymmetric `pos_embed` to
/// mirror that.
#[test]
fn p2c_bucket_index_matches_c2p_not_reversed() {
    // Non-symmetric setup: seq_len = 2 with hidden[0] != hidden[1] and
    // asymmetric pos_embed. C2P and P2C then each read a different
    // `pos_embed` row per (i, j) pair; if P2C is buggy and picks
    // `mid + (j - i)` instead of `mid + (i - j)`, the resulting sum in
    // `score[i, j]` differs from what HF/Microsoft's reference computes.
    //
    // The empirical proof: this test's expected values are derived from
    // the corrected math (see the block comment in the test body). Under
    // the buggy P2C indexing, the output MUST differ from the expected
    // values. Under the fixed P2C indexing, they match.
    let d_model = 2;
    let n_heads = 1;
    let head_dim = d_model / n_heads;
    let n_pos_buckets: i32 = 8; // mid = 4
    let max_pos_dist: i32 = 8;
    let seq_len = 2;

    // Sanity: for (i=0, j=1), rel = -1 → bucket_c2p = mid + rel = 3.
    //         For (i=1, j=0), rel = +1 → bucket_c2p = mid + rel = 5.
    // The buggy P2C used relative_position_bucket(j, i, ...):
    //   For (i=0, j=1) → rel = j - i = +1 → bucket = 5 (should be 3).
    //   For (i=1, j=0) → rel = j - i = -1 → bucket = 3 (should be 5).
    // → swap of pos_embed[3] and pos_embed[5] for these two pairs.
    assert_eq!(
        relative_position_bucket(0, 1, n_pos_buckets, max_pos_dist),
        3
    );
    assert_eq!(
        relative_position_bucket(1, 0, n_pos_buckets, max_pos_dist),
        5
    );
    assert_eq!(
        relative_position_bucket(1, 0, n_pos_buckets, max_pos_dist),
        5
    );
    assert_eq!(
        relative_position_bucket(0, 1, n_pos_buckets, max_pos_dist),
        3
    );

    // Asymmetric pos_embed: row `bucket` = [bucket, -bucket] so
    // pos_embed[3] = [3, -3], pos_embed[5] = [5, -5]. Swapping these two
    // buckets in P2C's lookup DOES change the C2C + C2P + P2C sum for
    // both (0,1) and (1,0) score entries.
    let mut pos_embed = vec![0.0_f32; n_pos_buckets as usize * d_model];
    for bucket in 0..(n_pos_buckets as usize) {
        pos_embed[bucket * d_model] = bucket as f32;
        pos_embed[bucket * d_model + 1] = -(bucket as f32);
    }

    // Non-zero content path: wq/wk/wv/wq_pos/wk_pos identity, hidden
    // distinguishable per token. w_out identity so we can trace the
    // final projection.
    let identity = |d: usize| -> Vec<f32> {
        let mut m = vec![0.0_f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        m
    };
    let w = AttnWeights {
        wq: identity(d_model),
        wk: identity(d_model),
        wv: identity(d_model),
        wq_pos: identity(d_model),
        wk_pos: identity(d_model),
        w_out: identity(d_model),
        pos_embed,
        bq: vec![0.0_f32; d_model],
        bk: vec![0.0_f32; d_model],
        bv: vec![0.0_f32; d_model],
        bout: vec![0.0_f32; d_model],
    };
    let attn =
        DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets, max_pos_dist);
    // Hidden: token 0 = [1, 0], token 1 = [0, 1] — distinguishable, and
    // combined with pos_embed asymmetry drives the P2C bucket mistake
    // through to the final output.
    let hidden: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let out = attn.forward(&hidden, seq_len);

    // Under the FIXED P2C indexing, and with the FIXED rel==0 mapping
    // (bucket = 4 for the diagonal), computed by hand (see the paper
    // trace on the parity report for a4cf654):
    //
    //   scale = 1/sqrt(3*2) = 1/sqrt(6) ≈ 0.4082
    //
    //   Q, K, V after `x @ I + 0` are just the input hidden.
    //   Q_p, K_p after `pos_embed @ I + 0` are just pos_embed.
    //
    //   For pair (0, 0):
    //     C2C = q[0] · k[0] = [1,0]·[1,0] = 1
    //     bucket_c2p = mid = 4  → k_p[4] = [4, -4]
    //     C2P = q[0] · k_p[4]  = [1,0]·[4,-4] = 4
    //     bucket_p2c = mid = 4 (SAME as C2P after fix) → q_p[4] = [4, -4]
    //     P2C = q_p[4] · k[0]  = [4,-4]·[1,0] = 4
    //     score[0,0] = (1 + 4 + 4) * 0.4082 = 9 * 0.4082 = 3.6742
    //
    //   For pair (0, 1) — off-diagonal:
    //     C2C = q[0] · k[1] = [1,0]·[0,1] = 0
    //     bucket_c2p = mid + (0-1) = 3  → k_p[3] = [3, -3]
    //     C2P = q[0] · k_p[3]  = [1,0]·[3,-3] = 3
    //     bucket_p2c (FIXED, same as c2p) = 3 → q_p[3] = [3, -3]
    //     P2C = q_p[3] · k[1]  = [3,-3]·[0,1] = -3
    //     score[0,1] = (0 + 3 + (-3)) * 0.4082 = 0
    //
    //   For pair (1, 0):
    //     C2C = q[1] · k[0] = [0,1]·[1,0] = 0
    //     bucket_c2p = mid + (1-0) = 5  → k_p[5] = [5, -5]
    //     C2P = q[1] · k_p[5]  = [0,1]·[5,-5] = -5
    //     bucket_p2c (FIXED) = 5 → q_p[5] = [5, -5]
    //     P2C = q_p[5] · k[0]  = [5,-5]·[1,0] = 5
    //     score[1,0] = (0 + (-5) + 5) * 0.4082 = 0
    //
    //   For pair (1, 1):
    //     C2C = q[1] · k[1] = 1
    //     bucket = mid = 4 → C2P = q[1] · k_p[4] = [0,1]·[4,-4] = -4
    //     P2C = q_p[4] · k[1] = [4,-4]·[0,1] = -4
    //     score[1,1] = (1 + (-4) + (-4)) * 0.4082 = -7 * 0.4082 = -2.858
    //
    //   softmax row 0: [3.6742, 0]. shift: [0, -3.6742]. exp: [1, 0.02535].
    //     sum = 1.02535. probs: [0.9753, 0.02472].
    //   out row 0 = 0.9753 * V[0] + 0.02472 * V[1] = 0.9753 * [1,0] + 0.02472 * [0,1]
    //             = [0.9753, 0.02472]
    //
    //   softmax row 1: [0, -2.858]. shift: [0, -2.858]. exp: [1, 0.05733].
    //     sum = 1.05733. probs: [0.9458, 0.05423].
    //   out row 1 = 0.9458 * [1,0] + 0.05423 * [0,1] = [0.9458, 0.05423]
    //
    //   Final projection: w_out = I, bout = 0 → out unchanged.
    let expected: [f32; 4] = [0.9753, 0.02472, 0.9458, 0.05423];
    for (i, (&actual, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 5e-3,
            "P2C-indexing pin: output[{i}] = {actual:e}, expected ≈ {want} — a mismatch \
             here means either (a) rel==0 still maps to bucket 0 (the pre-fix diagonal \
             bug), or (b) P2C still uses `relative_position_bucket(j, i, ...)` instead \
             of `(i, j, ...)`. The hand-derived expected values are the HF/Microsoft \
             reference (see the block comment above)."
        );
    }
}

fn synthetic_attn(d_model: usize, n_heads: usize, _seq_len: usize) -> DisentangledAttention {
    let head_dim = d_model / n_heads;
    let n_pos_buckets = 512;
    // 決定的 weight: 全て 0.01 の identity 相当
    let w = AttnWeights {
        wq: vec![0.01_f32; d_model * d_model],
        wk: vec![0.01_f32; d_model * d_model],
        wv: vec![0.01_f32; d_model * d_model],
        wq_pos: vec![0.01_f32; d_model * d_model],
        wk_pos: vec![0.01_f32; d_model * d_model],
        w_out: vec![0.01_f32; d_model * d_model],
        pos_embed: vec![0.001_f32; n_pos_buckets * d_model],
        bq: vec![0.0_f32; d_model],
        bk: vec![0.0_f32; d_model],
        bv: vec![0.0_f32; d_model],
        bout: vec![0.0_f32; d_model],
        bq_pos: None,
        bk_pos: None,
    };
    DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets as i32, 512)
}

#[test]
fn attention_output_shape_matches_input() {
    let d_model = 8;
    let attn = synthetic_attn(d_model, 2, 4);
    let hidden = vec![0.1_f32; 4 * d_model];
    let out = attn.forward(&hidden, 4);
    assert_eq!(out.len(), 4 * d_model);
}

#[test]
fn attention_is_deterministic() {
    let attn = synthetic_attn(8, 2, 4);
    let hidden = vec![0.1_f32; 4 * 8];
    let o1 = attn.forward(&hidden, 4);
    let o2 = attn.forward(&hidden, 4);
    assert_eq!(o1, o2);
}

#[test]
fn ffn_shape_and_determinism() {
    let d_model = 8;
    let d_ff = 32;
    let ffn = FfnBlock::new(
        vec![0.01_f32; d_ff * d_model],
        vec![0.0; d_ff],
        vec![0.01_f32; d_model * d_ff],
        vec![0.0; d_model],
        d_model,
        d_ff,
    );
    let x = vec![0.1_f32; 4 * d_model];
    let y1 = ffn.forward(&x, 4);
    let y2 = ffn.forward(&x, 4);
    assert_eq!(y1.len(), 4 * d_model);
    assert_eq!(y1, y2);
}

#[test]
fn ffn_gelu_activates() {
    let d_model = 4;
    let d_ff = 8;
    // 正の入力 → gelu 正、負の入力 → gelu 小
    let ffn = FfnBlock::new(
        vec![1.0_f32; d_ff * d_model],
        vec![0.0; d_ff],
        vec![1.0_f32; d_model * d_ff],
        vec![0.0; d_model],
        d_model,
        d_ff,
    );
    let y_pos = ffn.forward(&vec![1.0_f32; d_model], 1);
    let y_neg = ffn.forward(&vec![-1.0_f32; d_model], 1);
    assert!(y_pos[0].abs() > y_neg[0].abs());
}

#[test]
fn layer_norm_zero_mean_unit_var() {
    let ln = LayerNorm::new(vec![1.0; 4], vec![0.0; 4], 1e-7);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = ln.forward(&x, 1, 4);
    let mean: f32 = y.iter().sum::<f32>() / 4.0;
    let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5, "mean {mean}");
    assert!((var - 1.0).abs() < 1e-3, "var {var}");
}

#[test]
fn encoder_stack_forward_shape() {
    // 2-layer, d_model=8, vocab=16, seq=3
    let enc = DebertaV2Encoder::synthetic_for_test(2, 8, 2, 16, 512);
    let out = enc.forward(&[1, 2, 3]);
    assert_eq!(out.len(), 3 * 8);
}

/// WP-15 regression pin: [`AttnWeights`] carries dedicated `bq_pos` /
/// `bk_pos` optional bias fields, and when populated the disentangled
/// attention forward routes them into the *position*-aware Q/K
/// projections (not the *content* biases `bq`/`bk`). Prior to WP-15 the
/// struct had no such fields, so `Q_p` / `K_p` were computed with `bq` /
/// `bk` — a silent mis-attribution when an upstream checkpoint carries a
/// distinct position bias. Constructive proof: two instances that differ
/// **only** in `bq_pos` must yield distinct outputs (position bias flows
/// through `Q_p` → scores → softmax → weighted V-sum).
#[test]
fn pos_projection_honors_dedicated_q_bias() {
    let d_model = 8;
    let n_heads = 2;
    let head_dim = d_model / n_heads;
    let n_pos_buckets: i32 = 8;
    let seq_len = 4;
    let hidden: Vec<f32> = (0..seq_len * d_model)
        .map(|i| 0.1 + 0.01 * i as f32)
        .collect();
    let pos_embed: Vec<f32> = (0..(n_pos_buckets as usize * d_model))
        .map(|i| 0.01 + 0.001 * i as f32)
        .collect();

    let make = |bq_pos: Option<Vec<f32>>| {
        let w = AttnWeights {
            wq: vec![0.02_f32; d_model * d_model],
            wk: vec![0.03_f32; d_model * d_model],
            wv: vec![0.04_f32; d_model * d_model],
            wq_pos: vec![0.05_f32; d_model * d_model],
            wk_pos: vec![0.06_f32; d_model * d_model],
            w_out: vec![0.07_f32; d_model * d_model],
            pos_embed: pos_embed.clone(),
            bq: vec![0.0_f32; d_model],
            bk: vec![0.0_f32; d_model],
            bv: vec![0.0_f32; d_model],
            bout: vec![0.0_f32; d_model],
            bq_pos,
            bk_pos: None,
        };
        DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets, 32)
    };
    let out_none = make(None).forward(&hidden, seq_len);
    let out_some = make(Some(vec![0.5_f32; d_model])).forward(&hidden, seq_len);

    let diff: f32 = out_none
        .iter()
        .zip(&out_some)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "bq_pos = Some must change forward output; sum(|diff|) = {diff}"
    );
}

/// Companion to [`pos_projection_honors_dedicated_q_bias`] for the K
/// side, but written as a **storage / round-trip** pin rather than a
/// behavior differential. A `bk_pos` addend to `k_p` (row-broadcast
/// across every relative-position bucket) shifts C2P scores by the same
/// per-row constant across all columns j — softmax is invariant to
/// per-row translations, so `bk_pos` **cannot** move the attention
/// weights or the final output in standard softmax attention (this
/// argument holds equally for the content-K bias `bk`, which is why HF
/// DeBERTa v2 keeps the tensor in the checkpoint even though it is
/// mathematically inert). We pin that `bk_pos` at least round-trips
/// through [`AttnWeights`] as a `pub` field — the loader-side proof
/// that a real GGUF's `wk_pos.bias` is not silently dropped lives in
/// the loader tests.
#[test]
fn bk_pos_round_trips_through_attn_weights() {
    let d_model = 4;
    let bk_pos_val: Vec<f32> = vec![0.7, 0.8, 0.9, 1.0];
    let w = AttnWeights {
        wq: vec![0.0; d_model * d_model],
        wk: vec![0.0; d_model * d_model],
        wv: vec![0.0; d_model * d_model],
        wq_pos: vec![0.0; d_model * d_model],
        wk_pos: vec![0.0; d_model * d_model],
        w_out: vec![0.0; d_model * d_model],
        pos_embed: vec![0.0; 4 * d_model],
        bq: vec![0.0; d_model],
        bk: vec![0.0; d_model],
        bv: vec![0.0; d_model],
        bout: vec![0.0; d_model],
        bq_pos: None,
        bk_pos: Some(bk_pos_val.clone()),
    };
    assert_eq!(w.bk_pos.as_deref(), Some(bk_pos_val.as_slice()));
    assert!(w.bq_pos.is_none());
}

/// Backward-compat pin: when a checkpoint does *not* stamp
/// `wq_pos.bias` / `wk_pos.bias`, the loader leaves the fields as
/// `None` and the forward falls back to the content biases `bq` / `bk`
/// (this matches upstream `share_att_key=True` semantics — the same
/// content projection is applied to both content and position
/// embeddings). Structural check: an `AttnWeights` built with the
/// documented default (`bq_pos: None`) must be forward-safe, and the
/// resulting output must be **bit-identical** to one where `bq_pos` /
/// `bk_pos` explicitly repeat `bq` / `bk`.
#[test]
fn none_pos_bias_matches_content_bias_fallback() {
    let d_model = 8;
    let n_heads = 2;
    let head_dim = d_model / n_heads;
    let n_pos_buckets: i32 = 8;
    let seq_len = 4;
    let hidden: Vec<f32> = (0..seq_len * d_model)
        .map(|i| 0.1 + 0.01 * i as f32)
        .collect();
    let pos_embed: Vec<f32> = (0..(n_pos_buckets as usize * d_model))
        .map(|i| 0.01 + 0.001 * i as f32)
        .collect();
    let bq_content = vec![0.11_f32; d_model];
    let bk_content = vec![0.13_f32; d_model];

    let make = |bq_pos: Option<Vec<f32>>, bk_pos: Option<Vec<f32>>| {
        let w = AttnWeights {
            wq: vec![0.02_f32; d_model * d_model],
            wk: vec![0.03_f32; d_model * d_model],
            wv: vec![0.04_f32; d_model * d_model],
            wq_pos: vec![0.05_f32; d_model * d_model],
            wk_pos: vec![0.06_f32; d_model * d_model],
            w_out: vec![0.07_f32; d_model * d_model],
            pos_embed: pos_embed.clone(),
            bq: bq_content.clone(),
            bk: bk_content.clone(),
            bv: vec![0.0_f32; d_model],
            bout: vec![0.0_f32; d_model],
            bq_pos,
            bk_pos,
        };
        DisentangledAttention::new(w, d_model, n_heads, head_dim, n_pos_buckets, 32)
    };
    let out_none = make(None, None).forward(&hidden, seq_len);
    let out_dupe =
        make(Some(bq_content.clone()), Some(bk_content.clone())).forward(&hidden, seq_len);
    assert_eq!(
        out_none, out_dupe,
        "None must be bit-identical to Some(bq)/Some(bk) — the documented backward-compat fallback"
    );
}

/// Parity-CI root-cause pin (2026-08-09, bert_hidden_ja Δ 1.368e2 on run
/// 31314913038, HEAD 427bfd3): `EncoderLayer::forward` MUST apply
/// LayerNorm AFTER the residual add (post-norm), matching HuggingFace
/// transformers `DebertaV2SelfOutput.forward` / `DebertaV2Output.forward`
/// verbatim:
///
/// ```python
/// def forward(self, hidden_states, input_tensor):
///     hidden_states = self.dense(hidden_states)
///     hidden_states = self.dropout(hidden_states)
///     hidden_states = self.LayerNorm(hidden_states + input_tensor)  # LN AFTER residual
///     return hidden_states
/// ```
/// (`transformers/src/transformers/models/deberta_v2/modeling_deberta_v2.py`,
/// verified 2026-08-09 via WebFetch of upstream `main`.)
///
/// The pre-fix `EncoderLayer::forward` applied LN BEFORE the residual
/// (`h = hidden + attn(ln1(hidden))`), producing the ~136.8 accumulated
/// per-layer drift over 24 large-variant DeBERTa v2 layers CI reported —
/// weights loaded correctly (the converter maps `attention.output.LayerNorm.*`
/// → `ln1`, `output.LayerNorm.*` → `ln2`), only the graph topology
/// differed.
///
/// Construction: a single-position input `[1, 2, 3, 4]` with all-zero
/// attn+ffn weights + gamma=1/beta=0 LN produces different results
/// under pre-norm vs post-norm:
///
/// - **pre-norm** (buggy): `h = hidden + attn(ln(hidden)) = hidden + 0 =
///   hidden`; `y = h + ffn(ln(h)) = hidden + 0 = [1,2,3,4]` (unchanged).
/// - **post-norm** (fixed): `h = ln(hidden + attn(hidden)) = ln(hidden + 0)
///   = ln([1,2,3,4]) = [-1.342, -0.447, 0.447, 1.342]`; `y = ln(h + ffn(h))
///   = ln(h + 0) = ln(h) = h` (LN is idempotent on already-normalized
///   input up to a tiny eps effect).
///
/// So the differentiating assertion is `output != hidden` — pre-norm
/// returns hidden verbatim, post-norm returns the normalized values. We
/// pin the exact normalized values too so a future refactor that
/// accidentally uses different LN placement (e.g. "sandwich" norm) trips
/// on the specific expected numbers, not just the inequality.
#[test]
fn encoder_layer_forward_uses_post_norm() {
    let d_model = 4;
    let n_heads = 1;
    let head_dim = d_model / n_heads;
    let n_pos_buckets: i32 = 8;
    let seq_len = 1;

    // All-zero attn weights + biases → attn.forward returns all zeros
    // regardless of input (w_out · anything = 0, bout = 0). This is what
    // makes the "residual + LN(residual)" vs "residual" distinction bare.
    let attn_weights = AttnWeights {
        wq: vec![0.0_f32; d_model * d_model],
        wk: vec![0.0_f32; d_model * d_model],
        wv: vec![0.0_f32; d_model * d_model],
        wq_pos: vec![0.0_f32; d_model * d_model],
        wk_pos: vec![0.0_f32; d_model * d_model],
        w_out: vec![0.0_f32; d_model * d_model],
        pos_embed: vec![0.0_f32; n_pos_buckets as usize * d_model],
        bq: vec![0.0_f32; d_model],
        bk: vec![0.0_f32; d_model],
        bv: vec![0.0_f32; d_model],
        bout: vec![0.0_f32; d_model],
        bq_pos: None,
        bk_pos: None,
    };
    let attn =
        DisentangledAttention::new(attn_weights, d_model, n_heads, head_dim, n_pos_buckets, 32);
    // Same story: all-zero FFN produces zero output for any input.
    let d_ff = 4 * d_model;
    let ffn = FfnBlock::new(
        vec![0.0_f32; d_ff * d_model],
        vec![0.0_f32; d_ff],
        vec![0.0_f32; d_model * d_ff],
        vec![0.0_f32; d_model],
        d_model,
        d_ff,
    );
    // gamma=1, beta=0 → LN is pure normalization (mean 0, unit variance).
    let ln1 = LayerNorm::new(vec![1.0_f32; d_model], vec![0.0_f32; d_model], 1e-7);
    let ln2 = LayerNorm::new(vec![1.0_f32; d_model], vec![0.0_f32; d_model], 1e-7);
    let layer = EncoderLayer {
        attn,
        ffn,
        ln1,
        ln2,
    };

    // Concrete input with distinguishable per-element values so a
    // "returns input unchanged" pre-norm output is obvious vs the
    // "returns LN(input)" post-norm output.
    let hidden: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let out = layer.forward(&hidden, seq_len);

    // Pre-norm (pre-fix) would return `hidden` unchanged (all zeros pass
    // through both residual add slots). Post-norm must not.
    assert_ne!(
        out, hidden,
        "post-norm EncoderLayer must not return the input verbatim when attn/ffn are zero — \
         pre-norm arithmetic returns `hidden + 0 = hidden` (the exact bug parity CI's \
         bert_hidden_ja Δ 1.368e2 pinned)"
    );

    // Expected post-norm output for `hidden = [1,2,3,4]`:
    //   step 1: h = ln1(hidden + attn(hidden)) = ln1([1,2,3,4] + 0) = ln1([1,2,3,4])
    //     mean = 2.5, var = 1.25, std = sqrt(1.25 + 1e-7) ≈ 1.11803
    //     ln1([1,2,3,4]) = [(x - 2.5) / 1.11803] = [-1.34164, -0.44721, 0.44721, 1.34164]
    //   step 2: y = ln2(h + ffn(h)) = ln2(h + 0) = ln2(h)
    //     h is already mean 0, var 1 → ln2(h) = h (up to eps effect)
    //   so y ≈ [-1.34164, -0.44721, 0.44721, 1.34164]
    let expected: [f32; 4] = [
        -1.341_640_8_f32,
        -0.447_213_6_f32,
        0.447_213_6_f32,
        1.341_640_8_f32,
    ];
    for (i, (&actual, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 5e-3,
            "post-norm output[{i}] = {actual}, expected ≈ {want} (= ln(ln([1,2,3,4]))[{i}] \
             under gamma=1/beta=0; a mismatch here means LN placement changed to something \
             other than the HuggingFace DebertaV2SelfOutput.forward pattern)"
        );
    }
}
