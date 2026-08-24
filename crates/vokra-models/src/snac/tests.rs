use super::runtime::expected_manifest;
use super::*;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile, chunks};

#[test]
fn released_variant_configs_match_upstream() {
    let hz24 = SnacConfig::for_variant(SnacVariant::Hz24);
    assert_eq!(hz24.sample_rate, 24_000);
    assert_eq!(hz24.active_vq_strides(), [4, 2, 1]);
    assert_eq!(hz24.n_stages, 3);

    let hz44 = SnacConfig::for_variant(SnacVariant::Hz44);
    assert_eq!(hz44.sample_rate, 44_100);
    assert_eq!(hz44.active_vq_strides(), [8, 4, 2, 1]);
    assert_eq!(hz44.n_stages, 4);
}

#[test]
fn released_manifests_have_revisioned_public_counts() {
    assert_eq!(expected_manifest(SnacVariant::Hz24).len(), 269);
    assert_eq!(expected_manifest(SnacVariant::Hz44).len(), 286);
}

#[test]
fn hz44_manifest_contains_both_attention_blocks_and_four_codebooks() {
    let manifest = expected_manifest(SnacVariant::Hz44);
    assert_eq!(
        manifest.get("encoder.block.5.to_qkv.weight"),
        Some(&vec![3072, 1024])
    );
    assert_eq!(
        manifest.get("decoder.model.2.to_qkv.weight"),
        Some(&vec![4608, 1536])
    );
    assert_eq!(
        manifest.get("quantizer.quantizers.3.codebook.weight"),
        Some(&vec![4096, 8])
    );
}

#[test]
fn hz24_manifest_has_no_attention_and_three_codebooks() {
    let manifest = expected_manifest(SnacVariant::Hz24);
    assert!(!manifest.contains_key("encoder.block.5.to_qkv.weight"));
    assert!(!manifest.contains_key("decoder.model.2.to_qkv.weight"));
    assert!(manifest.contains_key("quantizer.quantizers.2.codebook.weight"));
    assert!(!manifest.contains_key("quantizer.quantizers.3.codebook.weight"));
}

#[test]
fn odd_stride_decoder_manifest_pins_output_padding_weight_shape() {
    let manifest = expected_manifest(SnacVariant::Hz44);
    assert_eq!(
        manifest.get("decoder.model.5.block.1.parametrizations.weight.original1"),
        Some(&vec![384, 192, 6])
    );
}

fn tiny_file(arch: &str, variant: &str) -> GgufFile {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, arch);
    builder.add_string(KEY_SNAC_VARIANT, variant);
    builder.add_string(chunks::KEY_MODEL_NAME, "snac-24khz");
    builder.add_string("vokra.model.category", "codec");
    builder.add_string(chunks::KEY_PROVENANCE_MODEL_ID, "snac-24khz");
    builder.add_string(chunks::KEY_PROVENANCE_LICENSE, "mit");
    builder.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        vokra_core::LicenseClass::Permissive.as_str(),
    );
    builder.add_string("vokra.provenance.upstream_hf", "hubertsiuzdak/snac_24khz");
    builder.add_string(chunks::KEY_PROVENANCE_SOURCE, "test source");
    builder
        .add_tensor(
            "unexpected",
            GgmlType::F32,
            vec![1],
            0.0_f32.to_le_bytes().to_vec(),
        )
        .unwrap();
    GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
}

#[test]
fn strict_binder_rejects_wrong_arch_before_tensor_load() {
    let error = Snac::from_gguf(&tiny_file("dac", VARIANT_TAG_HZ24)).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains("vokra.model.arch"));
}

#[test]
fn strict_binder_rejects_unknown_variant() {
    let error = Snac::from_gguf(&tiny_file(ARCH, "16khz")).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains(KEY_SNAC_VARIANT));
}

#[test]
fn strict_binder_rejects_partial_tensor_manifest() {
    let error = Snac::from_gguf(&tiny_file(ARCH, VARIANT_TAG_HZ24)).unwrap_err();
    assert!(matches!(error, vokra_core::VokraError::ModelLoad(_)));
    assert!(error.to_string().contains("tensor manifest mismatch"));
}
