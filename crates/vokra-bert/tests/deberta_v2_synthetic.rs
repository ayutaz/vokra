//! DeBERTa v2 synthetic-weight structure tests.
//! Clean-room per arXiv:2006.03654 + HF transformers deberta_v2 (Apache-2.0).

use vokra_bert::deberta_v2::{
    relative_position_bucket, AttnWeights, DisentangledAttention, FfnBlock,
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
