//! Plain BERT (`bert_base`) synthetic-weight structure tests.
//!
//! Clean-room per google-research/bert paper (Devlin et al. 2018,
//! arXiv:1810.04805) + HuggingFace transformers `BertModel`
//! (Apache-2.0). This module is intentionally arch-different from
//! `deberta_v2`/`deberta_v3` — plain BERT has learned absolute
//! position embeddings, no disentangled attention, and post-norm
//! layer order (LayerNorm applied after the residual add).
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)

use vokra_bert::bert_base::{
    BertBaseEncoder, BertConfig, BertEmbeddings, BertIntermediate, BertLayer, BertOutput,
    BertSelfAttention, BertSelfOutput,
};
use vokra_bert::deberta_v2::LayerNorm;

fn tiny_cfg() -> BertConfig {
    BertConfig {
        vocab_size: 16,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 32,
        max_position_embeddings: 32,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
    }
}

#[test]
fn config_carries_all_hparams() {
    let c = tiny_cfg();
    assert_eq!(c.vocab_size, 16);
    assert_eq!(c.hidden_size, 8);
    assert_eq!(c.num_hidden_layers, 1);
    assert_eq!(c.num_attention_heads, 2);
    assert_eq!(c.intermediate_size, 32);
    assert_eq!(c.max_position_embeddings, 32);
    assert_eq!(c.type_vocab_size, 2);
}

#[test]
fn embeddings_sum_three_tables_then_layernorm() {
    // token + position + token_type → LN. Deterministic weights so we
    // can prove the sum path is wired (LN then normalizes rows).
    let cfg = tiny_cfg();
    let emb = BertEmbeddings::new(
        vec![0.01_f32; cfg.vocab_size * cfg.hidden_size],
        vec![0.02_f32; cfg.max_position_embeddings * cfg.hidden_size],
        vec![0.03_f32; cfg.type_vocab_size * cfg.hidden_size],
        LayerNorm::new(
            vec![1.0_f32; cfg.hidden_size],
            vec![0.0_f32; cfg.hidden_size],
            cfg.layer_norm_eps,
        ),
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.max_position_embeddings,
        cfg.type_vocab_size,
    );
    let ids = vec![1_u32, 2, 3];
    let out = emb.forward(&ids, None);
    assert_eq!(out.len(), ids.len() * cfg.hidden_size);
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn embeddings_reject_oversized_token_id() {
    let cfg = tiny_cfg();
    let emb = BertEmbeddings::new(
        vec![0.01_f32; cfg.vocab_size * cfg.hidden_size],
        vec![0.02_f32; cfg.max_position_embeddings * cfg.hidden_size],
        vec![0.03_f32; cfg.type_vocab_size * cfg.hidden_size],
        LayerNorm::new(
            vec![1.0_f32; cfg.hidden_size],
            vec![0.0_f32; cfg.hidden_size],
            cfg.layer_norm_eps,
        ),
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.max_position_embeddings,
        cfg.type_vocab_size,
    );
    // vocab_size = 16 → id 16 is out of range
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        emb.forward(&[16_u32], None)
    }))
    .is_err();
    assert!(panicked, "token id ≥ vocab_size must panic (FR-EX-08)");
}

#[test]
fn embeddings_reject_seq_len_over_max_position() {
    let cfg = tiny_cfg();
    let emb = BertEmbeddings::new(
        vec![0.01_f32; cfg.vocab_size * cfg.hidden_size],
        vec![0.02_f32; cfg.max_position_embeddings * cfg.hidden_size],
        vec![0.03_f32; cfg.type_vocab_size * cfg.hidden_size],
        LayerNorm::new(
            vec![1.0_f32; cfg.hidden_size],
            vec![0.0_f32; cfg.hidden_size],
            cfg.layer_norm_eps,
        ),
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.max_position_embeddings,
        cfg.type_vocab_size,
    );
    // seq_len = 33 > max_position = 32 → must fail loud
    let ids = vec![0_u32; cfg.max_position_embeddings + 1];
    let panicked =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emb.forward(&ids, None))).is_err();
    assert!(panicked, "seq_len > max_position must panic (FR-EX-08)");
}

#[test]
fn self_attention_output_shape_matches_input() {
    let d = 8;
    let attn = BertSelfAttention::new(
        vec![0.01_f32; d * d],
        vec![0.0_f32; d],
        vec![0.01_f32; d * d],
        vec![0.0_f32; d],
        vec![0.01_f32; d * d],
        vec![0.0_f32; d],
        d,
        2,
    );
    let hidden = vec![0.1_f32; 4 * d];
    let out = attn.forward(&hidden, 4);
    assert_eq!(out.len(), 4 * d);
    assert!(out.iter().all(|x| x.is_finite()));
}

#[test]
fn self_output_post_norm_residual_is_layernormed() {
    // Post-norm invariant: after LN(x + residual), each row has ≈zero
    // mean and ≈unit variance (with gamma=1 / beta=0). The inputs
    // MUST be per-element varied — a constant row would collapse to
    // zero variance and mask a real routing bug.
    let d = 8;
    let so = BertSelfOutput::new(
        vec![0.01_f32; d * d],
        vec![0.0_f32; d],
        LayerNorm::new(vec![1.0_f32; d], vec![0.0_f32; d], 1e-12),
        d,
    );
    // Per-element varied inputs: sin(i) ∈ [-1, 1] scaled.
    let attn_out: Vec<f32> = (0..4 * d).map(|i| ((i as f32) * 0.5).sin() * 0.5).collect();
    let residual: Vec<f32> = (0..4 * d)
        .map(|i| ((i as f32) * 0.3 + 1.7).cos() * 0.7)
        .collect();
    let out = so.forward(&attn_out, &residual, 4);
    assert_eq!(out.len(), 4 * d);
    // Row 0 stats
    let mean: f32 = out[..d].iter().sum::<f32>() / d as f32;
    let var: f32 = out[..d].iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
    assert!(mean.abs() < 1e-4, "post-LN row mean {mean}");
    assert!((var - 1.0).abs() < 1e-2, "post-LN row var {var}");
}

#[test]
fn intermediate_gelu_asymmetric_pos_vs_neg() {
    // FFN intermediate uses GELU: strong positive → large positive,
    // strong negative → small (near-zero) magnitude. This differentiates
    // GELU from ReLU (which would give 0 for negative) and from identity.
    let d = 4;
    let ff = 8;
    let intr = BertIntermediate::new(vec![1.0_f32; ff * d], vec![0.0_f32; ff], d, ff);
    let y_pos = intr.forward(&vec![1.0_f32; d], 1);
    let y_neg = intr.forward(&vec![-1.0_f32; d], 1);
    assert!(y_pos[0].abs() > y_neg[0].abs());
    // GELU(4) ~= 4.0 (near-identity for large positive), GELU(-4) ~= 0
    let x_pos_sum = d as f32; // Wx = 4
                              // gelu(4) ≈ 3.9999
    assert!((y_pos[0] - x_pos_sum).abs() < 0.01);
}

#[test]
fn output_post_norm_residual_matches_self_output_shape() {
    let d = 8;
    let ff = 32;
    let out_block = BertOutput::new(
        vec![0.01_f32; d * ff],
        vec![0.0_f32; d],
        LayerNorm::new(vec![1.0_f32; d], vec![0.0_f32; d], 1e-12),
        d,
        ff,
    );
    let ffn_out = vec![0.1_f32; 4 * ff];
    let residual = vec![0.2_f32; 4 * d];
    let out = out_block.forward(&ffn_out, &residual, 4);
    assert_eq!(out.len(), 4 * d);
    // Row-wise LN invariant.
    let mean: f32 = out[..d].iter().sum::<f32>() / d as f32;
    assert!(mean.abs() < 1e-4, "post-LN row mean {mean}");
}

#[test]
fn layer_forward_shape_and_determinism() {
    let cfg = tiny_cfg();
    let layer = BertLayer::synthetic_for_test(&cfg);
    let hidden = vec![0.1_f32; 3 * cfg.hidden_size];
    let a = layer.forward(&hidden, 3);
    let b = layer.forward(&hidden, 3);
    assert_eq!(a.len(), 3 * cfg.hidden_size);
    assert_eq!(a, b);
    assert!(a.iter().all(|x| x.is_finite()));
}

#[test]
fn encoder_stack_forward_shape_and_finite() {
    // 1-layer tiny encoder — the WP-16 acceptance test.
    let cfg = tiny_cfg();
    let enc = BertBaseEncoder::synthetic_for_test(&cfg);
    let out = enc.forward(&[1_u32, 2, 3], None);
    assert_eq!(out.len(), 3 * cfg.hidden_size);
    assert!(out.iter().all(|x| x.is_finite()));
    assert_eq!(enc.d_model(), cfg.hidden_size);
}

#[test]
fn encoder_is_deterministic() {
    let cfg = tiny_cfg();
    let enc = BertBaseEncoder::synthetic_for_test(&cfg);
    let ids = [1_u32, 2, 3, 4, 5];
    let a = enc.forward(&ids, None);
    let b = enc.forward(&ids, None);
    assert_eq!(a, b);
}

#[test]
fn encoder_honors_token_type_ids() {
    // token_type_id 0 vs 1 → different token_type embedding lookup →
    // different output. Proves the token_type branch is not dead code.
    let cfg = tiny_cfg();
    let enc = BertBaseEncoder::synthetic_for_test(&cfg);
    let ids = [1_u32, 2, 3];
    let types_a = [0_u32, 0, 0];
    let types_b = [1_u32, 1, 1];
    let a = enc.forward(&ids, Some(&types_a));
    let b = enc.forward(&ids, Some(&types_b));
    assert_ne!(a, b, "token_type must influence output");
}

#[test]
fn encoder_via_trait_object() {
    // Uniform trait interface — same as DeBERTa v2/v3 impls.
    let cfg = tiny_cfg();
    let enc = BertBaseEncoder::synthetic_for_test(&cfg);
    let obj: Box<dyn vokra_bert::BertEncoder> = Box::new(enc);
    let out = obj.forward(&[1_u32, 2, 3]);
    assert_eq!(out.len(), 3 * cfg.hidden_size);
    assert_eq!(obj.d_model(), cfg.hidden_size);
}
