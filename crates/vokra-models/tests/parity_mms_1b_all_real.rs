//! MMS-1B-All real-artifact gate.
//!
//! The current public Vokra artifact is adapter-only. Keep this test loud
//! until a VAST-produced full backbone + selected language adapter artifact
//! and official oracle fixture have been audited.

use std::path::Path;
use vokra_models::wav2vec2_ctc::Wav2Vec2Ctc;

#[test]
fn mms_1b_all_stays_inspection_only_until_full_manifest() {
    let Some(path) = std::env::var_os("VOKRA_MMS_1B_ALL_GGUF") else {
        eprintln!("MMS_1B_ALL INSPECTION_ONLY: full GGUF not supplied");
        return;
    };
    let error = Wav2Vec2Ctc::from_gguf(Path::new(&path)).expect_err(
        "adapter-only or unaudited MMS artifact must not bind as a complete checkpoint",
    );
    let message = error.to_string();
    assert!(
        message.contains("inspection-only"),
        "unexpected MMS result: {message}"
    );
    assert!(
        message.contains("CC-BY-NC-4.0"),
        "license gate missing: {message}"
    );
    assert!(
        message.contains("load_adapter"),
        "adapter contract missing: {message}"
    );
    eprintln!("MMS_1B_ALL INSPECTION_ONLY: no CPU/Metal parity measured");
}
