use vokra_convert::ModelKind;

#[test]
fn deberta_variants_exist() {
    let _ = ModelKind::DebertaV2;
    let _ = ModelKind::DebertaV3;
}
