//! DeBERTa v2 synthetic-weight structure tests.
//! Clean-room per arXiv:2006.03654 + HF transformers deberta_v2 (Apache-2.0).

use vokra_bert::deberta_v2::{
    relative_position_bucket, AttnWeights, DebertaV2Encoder, DisentangledAttention, FfnBlock,
    LayerNorm,
};

#[test]
fn bucket_zero_for_same_position() {
    // q = k → bucket = 0
    assert_eq!(relative_position_bucket(5, 5, 256, 512), 0);
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
