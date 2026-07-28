use vokra_bert::{deberta_v2::DebertaV2Encoder, deberta_v3::DebertaV3Encoder, BertEncoder};

fn use_via_trait(enc: &dyn BertEncoder) -> usize {
    let out = enc.forward(&[1, 2]);
    assert_eq!(out.len(), 2 * enc.d_model());
    enc.d_model()
}

#[test]
fn v2_and_v3_share_trait() {
    let v2 = DebertaV2Encoder::synthetic_for_test(1, 8, 2, 16, 512);
    let v3 = DebertaV3Encoder::synthetic_for_test(1, 8, 2, 16, 512);
    assert_eq!(use_via_trait(&v2), 8);
    assert_eq!(use_via_trait(&v3), 8);
}
