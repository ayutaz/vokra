//! FireRedASR-AED-L inspection-only dispatch contracts.

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file, convert_file_licensed};

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "vokra-firered-inspection-{tag}-{}",
        std::process::id()
    ))
}

#[test]
fn aliases_dispatch_to_distinct_model_kind() {
    for alias in [
        "firered-asr-aed-l",
        "firered_asr_aed_l",
        "fireredteam/firered-asr-aed-l",
        "FireRedTeam/FireRedASR-AED-L",
    ] {
        assert_eq!(ModelKind::from_arg(alias), Some(ModelKind::FireredAsrAedL));
    }
    assert_eq!(ModelKind::FireredAsrAedL.as_arg(), "firered-asr-aed-l");
}

#[test]
fn direct_dispatch_and_license_dispatch_refuse_without_output() {
    let input = temp_path("input");
    let output = temp_path("output");
    std::fs::write(&input, b"arbitrary checkpoint").expect("input");
    std::fs::remove_file(&output).ok();
    for result in [
        convert_file(ModelKind::FireredAsrAedL, &input, &output),
        convert_file_licensed(
            ModelKind::FireredAsrAedL,
            &input,
            &output,
            Some("apache-2.0"),
        ),
    ] {
        let error = result.expect_err("FireRed conversion must be inspection-only");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert!(!output.exists());
    }
    std::fs::remove_file(input).ok();
    std::fs::remove_file(output).ok();
}
