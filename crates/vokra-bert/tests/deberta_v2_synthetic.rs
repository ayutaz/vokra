//! DeBERTa v2 synthetic-weight structure tests.
//! Clean-room per arXiv:2006.03654 + HF transformers deberta_v2 (Apache-2.0).

use vokra_bert::deberta_v2::relative_position_bucket;

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
