//! pyannote/segmentation-3.0 real-checkpoint smoke and CPU/Metal parity.
//!
//! Tests needing the immutable public GGUF are gated on [`GGUF_ENV`] and skip
//! cleanly when unset. Once supplied, strict identity, topology, default-on
//! CPU execution, and (on an Apple Metal build) CPU/Metal output parity are
//! hard requirements. No environment variable enables the forward.
//!
//! Set `PARITY_PYANNOTE_REAL_GGUF` to
//! `vokra/pyannote-segmentation-3.0@50bf4e510e0c689668384aec0f866f02e0fcaea8`
//! `pyannote-seg.gguf` (5,898,272 bytes; SHA-256
//! `22ff05fddf19e69c8d9aac8daa6d99014e6718bcd8d8c527d26da677d00c63f1`).
//! The VAST worker validates those values before invoking this suite.
//!
//! This smoke suite does not fabricate an upstream fixture or tolerance.
//! Independent official `pyannote.audio==3.0.0` probability parity is a
//! separate VAST suite; the default FP32 bound remains 0.01 until actual
//! measurements prove whether the implementation passes or needs diagnosis.

use std::env;
use std::path::Path;

use vokra_models::pyannote::{PyanNet, PyanNetConfig, decode_powerset};

/// Exact real GGUF supplied by the VAST worker. Absent means a clean skip;
/// present means all binding and execution failures are hard.
const GGUF_ENV: &str = "PARITY_PYANNOTE_REAL_GGUF";

#[test]
fn decode_powerset_matches_primary_source_mapping_table() {
    let expected: [(usize, Vec<usize>); 7] = [
        (0, vec![]),
        (1, vec![0]),
        (2, vec![1]),
        (3, vec![2]),
        (4, vec![0, 1]),
        (5, vec![0, 2]),
        (6, vec![1, 2]),
    ];
    for (argmax_class, expected_active) in expected {
        let mut row = vec![0.0f32; 7];
        row[argmax_class] = 1.0;
        let out = decode_powerset(&[row], 7, 16_000, 10);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].active_speakers, expected_active,
            "argmax={argmax_class} must decode to {expected_active:?}"
        );
    }
}

#[test]
fn parity_pyannote_public_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping real pyannote-segmentation smoke; this is a clean skip, never a fabricated pass"
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let model = PyanNet::from_gguf(path).unwrap_or_else(|error| {
        panic!(
            "pyannote-segmentation GGUF at {gguf_path} failed strict load: {error:?} (opted-in means hard failure; FR-EX-08)"
        )
    });
    let config = model.config();
    assert_eq!(model.model_name(), vokra_models::pyannote::NAME);
    assert_eq!(model.weight_license(), vokra_core::LicenseClass::Permissive);
    assert_eq!(config.sample_rate, 16_000);
    assert_eq!(config.lstm_num_layers, 4);
    assert_eq!(config.num_powerset_classes, 7);

    // 0.1 s deterministic 440 Hz input is long enough for three released
    // SincNet frames without making the four-layer scalar CPU oracle costly.
    let sample_rate = config.sample_rate as f32;
    let pcm: Vec<f32> = (0..1_600)
        .map(|index| (2.0 * std::f32::consts::PI * 440.0 * index as f32 / sample_rate).sin())
        .collect();
    let cpu = model.segment(&pcm).unwrap_or_else(|error| {
        panic!("pyannote-segmentation CPU forward failed at {gguf_path}: {error:?} (FR-EX-08)")
    });
    assert_eq!(cpu.len(), model.num_frames(pcm.len()));
    for (frame, row) in cpu.iter().enumerate() {
        assert_eq!(row.len(), config.num_powerset_classes as usize);
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "frame {frame} softmax sum {sum} != 1"
        );
        for (class, &probability) in row.iter().enumerate() {
            assert!(
                probability.is_finite() && (0.0..=1.0).contains(&probability),
                "frame {frame} class {class} probability {probability} is invalid"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use vokra_core::BackendKind;

        let metal = PyanNet::from_gguf_with_backend(path, BackendKind::Metal)
            .expect("bind and preflight exact PyanNet release for Metal")
            .segment(&pcm)
            .expect("run complete PyanNet Metal forward");
        assert_eq!(metal.len(), cpu.len());
        let max_abs = cpu
            .iter()
            .flatten()
            .zip(metal.iter().flatten())
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.01,
            "PyanNet CPU/Metal powerset max_abs={max_abs:.9e} exceeds 0.01"
        );
    }

    eprintln!(
        "pyannote-segmentation strict GGUF: sr={}, layers={}, classes={}, frames={}, legacy_metadata_repaired={}",
        config.sample_rate,
        config.lstm_num_layers,
        config.num_powerset_classes,
        cpu.len(),
        model.legacy_metadata_repaired(),
    );
}

#[test]
fn pyannet_config_default_constants_match_released_topology() {
    let config = PyanNetConfig::default();
    assert_eq!(config.sample_rate, 16_000);
    assert_eq!(config.sincnet_stride, 10);
    assert_eq!(config.lstm_hidden_size, 128);
    assert_eq!(config.lstm_num_layers, 4);
    assert!(config.lstm_bidirectional);
    assert!(config.lstm_monolithic);
    assert_eq!(config.linear_hidden_size, 128);
    assert_eq!(config.linear_num_layers, 2);
    assert_eq!(config.num_powerset_classes, 7);
}
