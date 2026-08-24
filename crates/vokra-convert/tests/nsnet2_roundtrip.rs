//! Round-trip test (Coverage-audit 2026-08-03 Wave A): a synthetic exact
//! NSNet2 initializer manifest is written to disk, run through
//! [`vokra_convert::convert_file`] (via [`ModelKind::Nsnet2`]) as the
//! CLI would, and the resulting GGUF is loaded back with the runtime
//! loader. This mirrors the whisper / campplus round-trip in
//! `tests/roundtrip.rs` — no large real ONNX is committed to the repo;
//! real-model E2E is a manual run driven by
//! `tools/parity/nsnet2_prepare_checkpoint.py` + `vokra-cli convert`.
//!
//! The purpose is threefold:
//!
//! 1. Prove that the wiring landed: `--model nsnet2` reaches
//!    [`models::nsnet2::convert_nsnet2_file`] via `ModelKind::Nsnet2`.
//! 2. Prove that the artifact carries the provenance chunks the publish
//!    gate (`scripts/publish/publish-one.sh`) needs (arch / name / license /
//!    category / upstream_url).
//! 3. Guard against a regression where a future rename in `models/mod.rs`
//!    or a dropped alias in `ModelKind::from_arg` silently misroutes
//!    `nsnet2` back to a different converter path (FR-EX-08 — never a
//!    silent default).

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile, chunks};

/// Unique per-test scratch path (PID + nanosecond timestamp — the
/// emotion2vec / ecapa_tdnn test pattern; no external `tempfile` dep so
/// the zero-dep NFR-DS-02 invariant is preserved even in the test tree).
fn scratch_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-nsnet2-it-{}-{}-{}.bin",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

const OFFICIAL_TENSORS: &[(&str, &[usize])] = &[
    ("172", &[161, 400]),
    ("fc_in.0.bias", &[400]),
    ("192", &[1, 1200, 400]),
    ("193", &[1, 1200, 400]),
    ("194", &[1, 2400]),
    ("212", &[1, 1200, 400]),
    ("213", &[1, 1200, 400]),
    ("214", &[1, 2400]),
    ("215", &[400, 600]),
    ("fc_out.0.bias", &[600]),
    ("216", &[600, 600]),
    ("fc_out.2.bias", &[600]),
    ("217", &[600, 161]),
    ("fc_out.4.bias", &[161]),
];

/// Synthetic exact-manifest safetensors buffer. Initializer `172` carries
/// non-zero sentinels so the test independently verifies its
/// `[161,400]` → `[400,161]` transpose and semantic rename.
fn synthetic_nsnet2_safetensors() -> (Vec<u8>, Vec<u8>) {
    let mut entries = Vec::with_capacity(OFFICIAL_TENSORS.len());
    let mut payload = Vec::new();
    let mut expected_fc_in = vec![0u8; 400 * 161 * 4];
    for &(name, shape) in OFFICIAL_TENSORS {
        let start = payload.len();
        let elements = shape.iter().product::<usize>();
        let mut tensor = vec![0u8; elements * 4];
        if name == "172" {
            for (row, col, value) in [
                (0usize, 0usize, 1.0f32),
                (0, 399, -2.5),
                (7, 23, 0.15625),
                (160, 0, 3.5),
                (160, 399, 42.0),
            ] {
                let source = (row * 400 + col) * 4;
                tensor[source..source + 4].copy_from_slice(&value.to_le_bytes());
                let target = (col * 161 + row) * 4;
                expected_fc_in[target..target + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        payload.extend_from_slice(&tensor);
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
            payload.len(),
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    buf.extend_from_slice(&payload);
    (buf, expected_fc_in)
}

/// Pins the end-to-end wiring: `ModelKind::Nsnet2` reaches the NSNet2
/// converter, the tensor survives the round-trip with dtype + bytes
/// preserved, and the artifact carries the provenance chunks the
/// publish gate needs.
#[test]
fn nsnet2_safetensors_roundtrips_through_convert_file() {
    let (input_bytes, payload) = synthetic_nsnet2_safetensors();
    let input = scratch_path("in");
    let output = scratch_path("out");
    std::fs::write(&input, &input_bytes).expect("write safetensors input");

    let summary = convert_file(ModelKind::Nsnet2, &input, &output).expect("convert");
    assert_eq!(summary.model, ModelKind::Nsnet2);
    assert_eq!(summary.tensor_count, 14, "official tensor manifest written");
    assert_eq!(
        summary.notes.len(),
        1,
        "one summary note surfaced for the strict conversion report"
    );
    assert!(
        summary.notes[0].starts_with("nsnet2:"),
        "note is namespaced by the nsnet2 arm — dispatch reached the intended converter"
    );

    let file = GgufFile::open(&output).expect("load output gguf");
    assert_eq!(file.tensors().len(), 14);
    let info = file
        .tensor_info("fc_in.weight")
        .expect("numeric initializer renamed to semantic runtime name");
    assert_eq!(info.dtype, GgmlType::F32);
    assert_eq!(info.dimensions, vec![400, 161]);
    assert_eq!(
        file.tensor_bytes(info),
        payload.as_slice(),
        "MatMul initializer transposed independently into runtime layout"
    );

    // Provenance chunks the publish gate reads.
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("nsnet2"),
        "vokra.model.arch pinned to `nsnet2` (distinct from `denoise` = DFN3)"
    );
    assert_eq!(
        file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
        Some("nsnet2-20ms-baseline"),
        "vokra.model.name pinned to the upstream ONNX stem"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("mit"),
        "MIT default license stamped verbatim"
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str()),
        "MIT normalises to Permissive"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

/// Alias dispatch pin: every CLI-visible spelling for NSNet2 must reach
/// `ModelKind::Nsnet2`. A silent dispatch onto a different converter
/// (or `None` = "unknown model", FR-EX-08) would produce either a wrong
/// GGUF or a hard usage error — both of which this test surfaces on the
/// spot.
#[test]
fn nsnet2_aliases_dispatch_to_the_intended_variant() {
    for alias in [
        "nsnet2",
        "nsnet2-baseline",
        "nsnet2-20ms",
        "nsnet2-20ms-baseline",
        "microsoft/nsnet2",
        "microsoft/nsnet2-baseline",
    ] {
        let parsed = ModelKind::from_arg(alias)
            .unwrap_or_else(|| panic!("--model {alias} must dispatch to ModelKind::Nsnet2"));
        assert_eq!(
            parsed,
            ModelKind::Nsnet2,
            "--model {alias} routed to {parsed:?} but the alias table says Nsnet2"
        );
    }
}
