//! Contract tests for the GigaAM Multilingual 71-class CTC inspection gate.

use std::path::Path;

use vokra_convert::convert_sber_gigaam_multilingual_file;

#[test]
fn multilingual_arbitrary_input_and_license_override_are_refused_without_output() {
    let output =
        std::env::temp_dir().join(format!("gigaam-multilingual-{}.gguf", std::process::id()));
    let _ = std::fs::remove_file(&output);
    for license in [None, Some("mit"), Some("apache-2.0"), Some("cc-by-4.0")] {
        let error = convert_sber_gigaam_multilingual_file(Path::new("missing"), &output, license)
            .unwrap_err()
            .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(!output.exists());
    }
}
