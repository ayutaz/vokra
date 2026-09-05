//! Active fail-closed dispatch tests for Sortformer-Diar-4spk-v1.

use std::path::PathBuf;

use vokra_convert::{
    ModelKind, convert_file, convert_file_licensed, convert_sortformer_diar_4spk_v1_file,
};

fn tmp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "vokra-sortformer-inspection-{tag}-{}",
        std::process::id()
    ));
    path
}

fn assert_inspection_only(error: impl std::fmt::Display) {
    let message = error.to_string();
    assert!(message.contains("INSPECTION_ONLY"), "{message}");
}

fn arbitrary_input() -> Vec<u8> {
    b"not-an-authenticated-sortformer-checkpoint".to_vec()
}

#[test]
fn direct_route_is_fail_closed_for_arbitrary_input_and_every_license() {
    for license in [None, Some("cc-by-nc-4.0"), Some("apache-2.0")] {
        let input = tmp_path("direct-input");
        let output = tmp_path("direct-output");
        std::fs::write(&input, arbitrary_input()).expect("write input");
        let error = convert_sortformer_diar_4spk_v1_file(&input, &output, license)
            .expect_err("inspection-only converter must reject");
        assert_inspection_only(error);
        assert!(
            !output.exists(),
            "fail-closed converter must not create output"
        );
        std::fs::remove_file(input).ok();
    }
}

#[test]
fn dispatch_route_is_fail_closed_and_does_not_create_output() {
    let input = tmp_path("dispatch-input");
    let output = tmp_path("dispatch-output");
    std::fs::write(&input, arbitrary_input()).expect("write input");
    let error = convert_file(ModelKind::SortformerDiar4spkV1, &input, &output)
        .expect_err("dispatch must preserve inspection-only refusal");
    assert_inspection_only(error);
    assert!(!output.exists(), "dispatch refusal must not create output");
    std::fs::remove_file(input).ok();
}

#[test]
fn licensed_dispatch_route_rejects_canonical_and_permissive_labels() {
    for license in ["cc-by-nc-4.0", "apache-2.0"] {
        let input = tmp_path("licensed-input");
        let output = tmp_path("licensed-output");
        std::fs::write(&input, arbitrary_input()).expect("write input");
        let error = convert_file_licensed(
            ModelKind::SortformerDiar4spkV1,
            &input,
            &output,
            Some(license),
        )
        .expect_err("licensed dispatch must preserve inspection-only refusal");
        assert_inspection_only(error);
        assert!(!output.exists(), "licensed refusal must not create output");
        std::fs::remove_file(input).ok();
    }
}
