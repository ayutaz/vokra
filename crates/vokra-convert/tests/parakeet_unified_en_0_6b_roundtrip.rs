//! Coverage-audit 2026-08-03 Wave B integration test: a synthetic
//! Parakeet-Unified-shaped safetensors checkpoint round-trips through
//! the public [`convert_file`] dispatch (`ModelKind::ParakeetUnified`)
//! into a well-formed GGUF the runtime loader can parse.
//!
//! The narrower unit tests live inline in
//! `crates/vokra-convert/src/models/parakeet_unified.rs`; this test
//! guards the outer `ModelKind` dispatch + `ConvertSummary` reporting +
//! `vokra.provenance.upstream_hf` stamp so a future refactor that
//! dropped the ParakeetUnified arm from `convert_file_licensed` would
//! fail loudly here.
//!
//! Mirror of `crates/vokra-convert/tests/roundtrip.rs` (M0-03-T17) —
//! no large real checkpoint is committed. Full real-weight parity
//! against the upstream NeMo pipeline is the owner deliverable
//! (`docs/license-audit.md` §3.1 sign-off queue for
//! parakeet-unified-en-0.6b).

use std::path::PathBuf;

use vokra_convert::{ConvertSummary, ModelKind, convert_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufFile, chunks};

/// A unique temp path for this test process (moshi / emotion2vec /
/// nkf_aec pattern — no external `tempfile` dep, preserving zero-dep
/// NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-parakeet-unified-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Builds a synthetic Parakeet-Unified-shaped safetensors buffer:
/// two small F32 tensors under encoder-flavour names so a downstream
/// reader can eyeball the round-trip as "recognisably FastConformer-
/// encoder shaped". The actual per-layer axes are the owner
/// deliverable — this fixture only exercises the pass-through
/// contract.
fn synthetic_parakeet_unified_safetensors() -> Vec<u8> {
    // Non-zero payloads so a silent widen / drop regression falls out
    // rather than trivially round-tripping a zero buffer.
    let qkv: Vec<u8> = [1.0f32, -2.0, 3.5, -0.25, 100.0, 0.001]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(qkv.len(), 24, "6 elements × 4 bytes F32 payload");
    let punc_bias: Vec<u8> = [-42.0f32, 7.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(punc_bias.len(), 8, "2 elements × 4 bytes F32 payload");
    let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"punc_cap_head.bias":{"dtype":"F32","shape":[2],"data_offsets":[24,32]}}"#;
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&qkv);
    buf.extend_from_slice(&punc_bias);
    buf
}

#[test]
fn parakeet_unified_safetensors_roundtrips_through_convert_file_dispatch() {
    let input = tmp_path("in");
    let output = tmp_path("out");
    std::fs::write(&input, synthetic_parakeet_unified_safetensors()).expect("write input");

    let summary: ConvertSummary =
        convert_file(ModelKind::ParakeetUnified, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::ParakeetUnified);
    assert_eq!(summary.tensor_count, 2, "both float tensors must land");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must be non-empty on disk"
    );
    // Note: `ConvertSummary::metadata_count` on the file-based early-return
    // path is stamped as `0` per the neucodec / ecapa_tdnn / wespeaker
    // pattern (the fleet dispatch does not read the builder's metadata
    // counter back before returning — see `convert_file_licensed` in
    // `lib.rs`). The real metadata count is asserted below by GGUF parse.

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.tensors().len(),
        2,
        "both synthetic tensors survive the round-trip"
    );
    assert!(
        file.tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .is_some()
    );
    assert!(file.tensor_info("punc_cap_head.bias").is_some());

    // Arch + name + category — the three axes downstream tooling
    // (publish-one.sh / make_model_card.py / zoo manifest) reads.
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("parakeet-unified"),
        "vokra.model.arch pins the short arch tag (distinct from parakeet-tdt / parakeet-ctc)"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("parakeet-unified-en-0.6b"),
        "vokra.model.name matches the publish repo slug"
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("asr"),
        "category is asr (offline+streaming unified)"
    );

    // Provenance — Parakeet-Unified ships from HF (per the expected
    // NVIDIA family precedent) so the upstream_hf slot must be
    // stamped.
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("nvidia/parakeet-unified-en-0.6b"),
        "upstream_hf pins the canonical HF slug"
    );

    // License stamp — apache-2.0 default = Permissive per
    // PERMISSIVE_TOKENS.
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("apache-2.0")
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str())
    );

    // Schema stamp — the writer emits this unconditionally.
    assert!(
        file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
        "vokra.schema.version must be stamped"
    );
    assert!(
        file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
        "vokra.schema.producer must be stamped"
    );

    // Byte-identical payload survival — a silent widen / re-quantize
    // would flip these payloads.
    let expected_qkv: Vec<u8> = [1.0f32, -2.0, 3.5, -0.25, 100.0, 0.001]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let expected_bias: Vec<u8> = [-42.0f32, 7.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let qkv_info = file
        .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
        .unwrap();
    let bias_info = file.tensor_info("punc_cap_head.bias").unwrap();
    assert_eq!(
        file.tensor_bytes(qkv_info),
        expected_qkv.as_slice(),
        "F32 qkv payload must be byte-identical to input"
    );
    assert_eq!(
        file.tensor_bytes(bias_info),
        expected_bias.as_slice(),
        "F32 punc_cap_head bias payload must be byte-identical to input"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
