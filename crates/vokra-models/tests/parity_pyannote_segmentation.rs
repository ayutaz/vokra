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
//! [`parity_pyannote_official_probabilities`] consumes the independent dump
//! made by `tools/parity/pyannote_segmentation_dump_reference.py`. The oracle
//! imports the pinned official `PyanNet.forward`; the standard FP32 absolute
//! bound is 0.01 and any opted-in mismatch is a hard failure.

use std::env;
use std::fs;
use std::path::Path;

use vokra_models::pyannote::{PyanNet, PyanNetConfig, decode_powerset};

/// Exact real GGUF supplied by the VAST worker. Absent means a clean skip;
/// present means all binding and execution failures are hard.
const GGUF_ENV: &str = "PARITY_PYANNOTE_REAL_GGUF";
/// Directory emitted by the independent official `PyanNet.forward` dumper.
const REFERENCE_DIR_ENV: &str = "PARITY_PYANNOTE_REFERENCE_DIR";
const OFFICIAL_FP32_ATOL: f32 = 0.01;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned f32 reference {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

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
#[ignore = "requires the immutable public GGUF and independent official VAST dump"]
fn parity_pyannote_official_probabilities() {
    let gguf_path = env::var(GGUF_ENV)
        .unwrap_or_else(|_| panic!("{GGUF_ENV} must point at the immutable public PyanNet GGUF"));
    let reference_dir = env::var(REFERENCE_DIR_ENV).unwrap_or_else(|_| {
        panic!("{REFERENCE_DIR_ENV} must point at the independent official dump")
    });
    let reference_dir = Path::new(&reference_dir);
    let pcm = read_f32(&reference_dir.join("input_pcm.f32"));
    let expected = read_f32(&reference_dir.join("probabilities.f32"));
    assert_eq!(pcm.len(), 1_600, "official deterministic PCM length");
    assert_eq!(expected.len(), 3 * 7, "official probability shape [1,3,7]");
    assert!(
        expected.iter().all(|value| value.is_finite()),
        "official reference probabilities must be finite"
    );

    let model = PyanNet::from_gguf(Path::new(&gguf_path))
        .unwrap_or_else(|error| panic!("strict public PyanNet bind failed: {error:?}"));
    let actual = model
        .segment(&pcm)
        .unwrap_or_else(|error| panic!("native PyanNet CPU forward failed: {error:?}"));
    let actual: Vec<f32> = actual.into_iter().flatten().collect();
    assert_eq!(actual.len(), expected.len(), "official probability length");
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "native probabilities must be finite"
    );

    let mut max_abs = 0.0f32;
    let mut absolute_sum = 0.0f64;
    for (&native, &official) in actual.iter().zip(&expected) {
        let difference = (native - official).abs();
        max_abs = max_abs.max(difference);
        absolute_sum += f64::from(difference);
    }
    let mean_abs = absolute_sum / actual.len() as f64;
    assert!(
        max_abs <= OFFICIAL_FP32_ATOL,
        "PyanNet CPU vs official max_abs={max_abs:.9e} exceeds {OFFICIAL_FP32_ATOL:.9e}"
    );
    eprintln!(
        "PYANNOTE_OFFICIAL_PARITY backend=cpu max_abs={max_abs:.9e} mean_abs={mean_abs:.9e} bound={OFFICIAL_FP32_ATOL:.9e} frames=3 classes=7 verdict=PASS"
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
