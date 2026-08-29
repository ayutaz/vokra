//! Contract tests for the GigaAM Multilingual 71-class CTC inspection gate.

use std::path::Path;

use vokra_convert::convert_sber_gigaam_multilingual_file;

#[test]
fn multilingual_arbitrary_input_and_license_override_are_refused_without_output() {
    let output =
        std::env::temp_dir().join(format!("gigaam-multilingual-{}.gguf", std::process::id()));
    const MISSING_SIDECAR_ERROR: &str = "I/O error: No such file or directory (os error 2)";
    for license in [None, Some("mit"), Some("MIT")] {
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_multilingual_file(Path::new("missing"), &output, license)
            .unwrap_err()
            .to_string();
        assert_eq!(error, MISSING_SIDECAR_ERROR);
        assert!(!output.exists());
    }
    const INCOMPATIBLE_LICENSE: &str =
        "usage error: GigaAM Multilingual weights are fixed MIT; license override must be `mit`";
    for license in [Some("apache-2.0"), Some("cc-by-4.0")] {
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_multilingual_file(Path::new("missing"), &output, license)
            .unwrap_err()
            .to_string();
        assert_eq!(error, INCOMPATIBLE_LICENSE);
        assert!(!output.exists());
    }
}
