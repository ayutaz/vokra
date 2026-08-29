//! Contract tests for the fail-closed OWSM v4 medium 1B conversion boundary.

use std::path::Path;

use vokra_convert::convert_owsm_v4_medium_1b_file;

#[test]
fn arbitrary_input_is_refused_and_output_is_not_created() {
    let output = std::env::temp_dir().join(format!(
        "vokra-owsm-v4-medium-1b-{}.gguf",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&output);
    let error =
        convert_owsm_v4_medium_1b_file(Path::new("/does/not/exist.safetensors"), &output, None)
            .unwrap_err()
            .to_string();
    assert!(error.contains("INSPECTION_ONLY"), "{error}");
    assert!(!output.exists());
}

#[test]
fn every_license_override_is_refused() {
    for license in [Some("cc-by-4.0"), Some("apache-2.0"), Some("mit")] {
        let error = convert_owsm_v4_medium_1b_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/owsm-v4-medium-1b.gguf"),
            license,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
    }
}
