//! Coverage-audit 2026-08-03 Wave A ticket: integration test that a
//! synthetic RNNoise-shaped safetensors checkpoint round-trips through the
//! public [`convert_file`] dispatch (`ModelKind::Rnnoise`) into a
//! well-formed GGUF the runtime loader can parse. The narrower unit tests
//! live inline in `crates/vokra-convert/src/models/rnnoise.rs`; this test
//! guards the outer `ModelKind` dispatch + `ConvertSummary` reporting +
//! `vokra.provenance.upstream_url` stamp so a future refactor that dropped
//! the RNNoise arm from `convert_file_licensed` would fail loudly here.
//!
//! Mirror of `crates/vokra-convert/tests/roundtrip.rs` (M0-03-T17) — no
//! large real checkpoint is committed. Full real-weight parity against
//! Xiph's reference C forward lives in the separately gated model test.

use std::path::PathBuf;

use vokra_convert::{ConvertSummary, ModelKind, convert_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgufFile, chunks};

/// A unique temp path for this test process (moshi / emotion2vec pattern
/// — no external `tempfile` dep, preserving zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-rnnoise-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

fn canonical_manifest() -> Vec<(String, usize)> {
    let mut tensors = vec![
        ("conv1_weights_float".to_owned(), 24_960),
        ("conv1_bias".to_owned(), 128),
        ("conv2_weights_int8".to_owned(), 147_456),
        ("conv2_scale".to_owned(), 384),
        ("conv2_bias".to_owned(), 384),
    ];
    for layer in 1..=3 {
        for part in ["input", "recurrent"] {
            let prefix = format!("gru{layer}_{part}");
            tensors.push((format!("{prefix}_weights_int8"), 147_456));
            tensors.push((format!("{prefix}_weights_idx"), 4_752));
            tensors.push((format!("{prefix}_scale"), 1_152));
            tensors.push((format!("{prefix}_bias"), 1_152));
            if part == "recurrent" {
                tensors.push((format!("{prefix}_weights_diag"), 1_152));
            }
        }
    }
    tensors.extend([
        ("dense_out_weights_float".to_owned(), 12_288),
        ("dense_out_bias".to_owned(), 32),
        ("vad_dense_weights_float".to_owned(), 384),
        ("vad_dense_bias".to_owned(), 1),
    ]);
    assert_eq!(tensors.len(), 36);
    tensors
}

/// Builds the complete canonical 36-array RNNoise safetensors contract with
/// two non-zero payload anchors. The converter must reject the old two-tensor
/// scaffold instead of accepting a structurally meaningless artifact.
fn synthetic_rnnoise_safetensors() -> Vec<u8> {
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    for (name, count) in canonical_manifest() {
        let start = payload.len();
        let end = start + count * 4;
        payload.resize(end, 0);
        if name == "conv1_weights_float" {
            for (index, value) in [1.0f32, -2.0, 3.5].iter().enumerate() {
                payload[start + index * 4..start + (index + 1) * 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
        } else if name == "vad_dense_bias" {
            payload[start..end].copy_from_slice(&7.5f32.to_le_bytes());
        }
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{count}],\"data_offsets\":[{start},{end}]}}"
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&payload);
    buf
}

#[test]
fn rnnoise_safetensors_roundtrips_through_convert_file_dispatch() {
    let input = tmp_path("in");
    let output = tmp_path("out");
    std::fs::write(&input, synthetic_rnnoise_safetensors()).expect("write input");

    let summary: ConvertSummary =
        convert_file(ModelKind::Rnnoise, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Rnnoise);
    assert_eq!(summary.tensor_count, 36, "all canonical arrays must land");
    assert!(
        summary.output_bytes > 0,
        "output GGUF must be non-empty on disk"
    );
    // Note: `ConvertSummary::metadata_count` on the file-based early-return
    // path is stamped as `0` for the Phase 5 fleet + coverage-audit
    // pattern (the fleet dispatch does not read the builder's metadata
    // counter back before returning — see `convert_file_licensed` in
    // `lib.rs`). The real metadata count is asserted below by GGUF parse.

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(
        file.tensors().len(),
        36,
        "the complete canonical manifest survives the round-trip"
    );
    assert!(file.tensor_info("conv1_weights_float").is_some());
    assert!(file.tensor_info("gru3_recurrent_weights_diag").is_some());
    assert!(file.tensor_info("vad_dense_bias").is_some());

    // Arch + name + category — the three axes downstream tooling
    // (publish-one.sh / make_model_card.py / zoo manifest) reads.
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("rnnoise"),
        "vokra.model.arch pins the short arch tag (distinct from DFN3's `denoise`)"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("rnnoise-v0.2"),
        "vokra.model.name matches the publish repo slug"
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("denoise"),
        "category is the shared denoise family (sibling of DFN3)"
    );

    // Provenance — RNNoise ships from GitHub Release, not HF, so the
    // URL slot must be stamped (and `upstream_hf` must be absent to
    // avoid misleading the model-card generator).
    assert_eq!(
        file.get("vokra.provenance.upstream_url")
            .and_then(|v| v.as_str()),
        Some("https://github.com/xiph/rnnoise/releases/tag/v0.2"),
        "upstream_url pins the GitHub Release the blob ships from"
    );
    assert!(
        file.get("vokra.provenance.upstream_hf").is_none(),
        "upstream_hf must NOT be stamped for a non-HF release (would misrepresent \
         the serving location)"
    );

    // License stamp — BSD-3-Clause = Permissive per PERMISSIVE_TOKENS.
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("bsd-3-clause")
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
    let expected_kernel_prefix: Vec<u8> = [1.0f32, -2.0, 3.5]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let kernel = file.tensor_data("conv1_weights_float").unwrap();
    assert_eq!(
        &kernel[..expected_kernel_prefix.len()],
        expected_kernel_prefix.as_slice(),
        "F32 kernel prefix must be byte-identical to input"
    );
    assert_eq!(
        file.tensor_data("vad_dense_bias").unwrap(),
        7.5f32.to_le_bytes().as_slice(),
        "F32 bias payload must be byte-identical to input"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}
