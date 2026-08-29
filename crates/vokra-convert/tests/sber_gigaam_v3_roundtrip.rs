//! Contract tests for the GigaAM v3 RNNT inspection-only boundary.

use std::path::Path;

use vokra_convert::convert_sber_gigaam_v3_file;

#[test]
fn v3_arbitrary_input_and_license_override_are_refused_without_output() {
    let output = std::env::temp_dir().join(format!("gigaam-v3-{}.gguf", std::process::id()));
    const PREPARED_SHA_BLOCKER: &str = "usage error: GigaAM v3 prepared SHA-256 is not independently authenticated; obtain VAST evidence first";
    for license in [None, Some("mit"), Some("MIT")] {
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_v3_file(Path::new("missing"), &output, license)
            .unwrap_err()
            .to_string();
        assert_eq!(error, PREPARED_SHA_BLOCKER);
        assert!(!output.exists());
    }
    const INCOMPATIBLE_LICENSE: &str =
        "usage error: GigaAM v3 weights are fixed MIT; license override must be `mit`";
    for license in [Some("apache-2.0"), Some("cc-by-4.0")] {
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_v3_file(Path::new("missing"), &output, license)
            .unwrap_err()
            .to_string();
        assert_eq!(error, INCOMPATIBLE_LICENSE);
        assert!(!output.exists());
    }
}
