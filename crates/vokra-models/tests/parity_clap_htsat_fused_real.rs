//! CLAP real-artifact gate.
//!
//! This test is intentionally fail-closed while the VAST tensor manifest is
//! absent. It never treats the metadata-only binder as a CPU or Metal
//! implementation and therefore cannot manufacture a parity PASS.

use vokra_models::clap::Clap;

#[test]
fn clap_real_artifact_stays_inspection_only_without_manifest() {
    let Some(path) = std::env::var_os("VOKRA_CLAP_REAL_GGUF") else {
        eprintln!(
            "CLAP_METAL_VS_CPU INSPECTION_ONLY: GGUF not supplied; no hardware parity measured"
        );
        return;
    };
    let error = Clap::from_path(path).expect_err("unverified CLAP GGUF must not bind");
    let message = error.to_string();
    assert!(
        message.contains("inspection-only"),
        "unexpected gate: {message}"
    );
    assert!(
        message.contains("tensor-name/shape manifest"),
        "unexpected gate: {message}"
    );
    eprintln!(
        "CLAP_METAL_VS_CPU INSPECTION_ONLY: native binder is manifest-gated; no hardware parity measured"
    );
}
