//! Exact pyannote/speaker-diarization-3.1 three-artifact parity.
//!
//! Real files are opt-in so ordinary local tests never download or execute a
//! model. The VAST worker verifies all three immutable public identities, runs
//! the independent official `SpeakerDiarization.apply` dumper, then enables
//! the ignored end-to-end comparison below.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use vokra_core::BackendKind;
use vokra_models::pyannote::diarization::PyannoteSpeakerDiarization31;

const PIPELINE_ENV: &str = "PARITY_PYANNOTE_DIARIZATION_PIPELINE_GGUF";
const SEGMENTATION_ENV: &str = "PARITY_PYANNOTE_DIARIZATION_SEGMENTATION_GGUF";
const EMBEDDING_ENV: &str = "PARITY_PYANNOTE_DIARIZATION_EMBEDDING_GGUF";
const REFERENCE_DIR_ENV: &str = "PARITY_PYANNOTE_DIARIZATION_REFERENCE_DIR";
const TIME_ATOL_SECONDS: f32 = 1.0e-5;

fn three_artifacts() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let pipeline = env::var_os(PIPELINE_ENV);
    let segmentation = env::var_os(SEGMENTATION_ENV);
    let embedding = env::var_os(EMBEDDING_ENV);
    match (pipeline, segmentation, embedding) {
        (None, None, None) => None,
        (Some(pipeline), Some(segmentation), Some(embedding)) => {
            Some((pipeline.into(), segmentation.into(), embedding.into()))
        }
        _ => panic!(
            "{PIPELINE_ENV}, {SEGMENTATION_ENV}, and {EMBEDDING_ENV} must be all set or all unset"
        ),
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned f32 file {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "unaligned u32 file {path:?}");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn strict_public_pipeline_binds_all_three_artifacts() {
    let Some((pipeline, segmentation, embedding)) = three_artifacts() else {
        eprintln!(
            "pyannote diarization GGUF envs unset — skipping strict real-file bind; this is a clean skip"
        );
        return;
    };
    let model =
        PyannoteSpeakerDiarization31::open(pipeline, segmentation, embedding, BackendKind::Cpu)
            .expect("strictly bind exact pipeline, PyanNet, and WeSpeaker artifacts");
    assert_eq!(model.backend(), BackendKind::Cpu);
    assert_eq!(model.config().segmentation_batch_size, 32);
    assert_eq!(model.config().embedding_batch_size, 32);
    assert!(model.config().embedding_exclude_overlap);
    assert_eq!(model.config().clustering_min_cluster_size, 12);
}

#[test]
#[ignore = "requires three immutable public GGUFs and the independent official VAST dump"]
fn parity_pyannote_diarization_official_segments() {
    let (pipeline, segmentation, embedding) = three_artifacts().unwrap_or_else(|| {
        panic!("all three pyannote diarization GGUF environment variables are required")
    });
    let reference_dir = PathBuf::from(
        env::var_os(REFERENCE_DIR_ENV)
            .unwrap_or_else(|| panic!("{REFERENCE_DIR_ENV} must point at the official dump")),
    );
    let pcm = read_f32(&reference_dir.join("input_pcm.f32"));
    assert_eq!(pcm.len(), 96_000, "official six-second input length");
    let expected_times = read_f32(&reference_dir.join("segments.f32"));
    let expected_speakers = read_u32(&reference_dir.join("segment_speakers.u32"));
    assert_eq!(expected_times.len(), 2 * expected_speakers.len());
    assert!(
        !expected_speakers.is_empty(),
        "official reference must exercise at least one speaker turn"
    );

    let model =
        PyannoteSpeakerDiarization31::open(&pipeline, &segmentation, &embedding, BackendKind::Cpu)
            .expect("strict official-parity bind");
    let actual = model
        .diarize(&pcm, 16_000)
        .expect("native exact diarization CPU forward");
    assert_eq!(
        actual.len(),
        expected_speakers.len(),
        "official/native speaker-turn count"
    );

    let mut max_time_abs = 0.0f32;
    for (index, segment) in actual.iter().enumerate() {
        let expected_start = expected_times[2 * index];
        let expected_duration = expected_times[2 * index + 1];
        let start_abs = (segment.start_s - expected_start).abs();
        let duration_abs = (segment.duration_s - expected_duration).abs();
        max_time_abs = max_time_abs.max(start_abs).max(duration_abs);
        assert_eq!(
            segment.speaker_id as u32, expected_speakers[index],
            "speaker label at turn {index}"
        );
        assert!(
            start_abs <= TIME_ATOL_SECONDS,
            "turn {index} start {} vs official {expected_start} differs by {start_abs}",
            segment.start_s
        );
        assert!(
            duration_abs <= TIME_ATOL_SECONDS,
            "turn {index} duration {} vs official {expected_duration} differs by {duration_abs}",
            segment.duration_s
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let metal = PyannoteSpeakerDiarization31::open(
            pipeline,
            segmentation,
            embedding,
            BackendKind::Metal,
        )
        .expect("strict three-artifact Metal bind and eager preflight")
        .diarize(&pcm, 16_000)
        .expect("complete pyannote diarization Metal forward");
        assert_eq!(metal.len(), actual.len(), "CPU/Metal speaker-turn count");
        for (index, (cpu, gpu)) in actual.iter().zip(&metal).enumerate() {
            assert_eq!(cpu.speaker_id, gpu.speaker_id, "CPU/Metal label {index}");
            assert!(
                (cpu.start_s - gpu.start_s).abs() <= TIME_ATOL_SECONDS,
                "CPU/Metal start mismatch at turn {index}"
            );
            assert!(
                (cpu.duration_s - gpu.duration_s).abs() <= TIME_ATOL_SECONDS,
                "CPU/Metal duration mismatch at turn {index}"
            );
        }
    }

    eprintln!(
        "PYANNOTE_DIARIZATION_OFFICIAL_PARITY backend=cpu turns={} max_time_abs={max_time_abs:.9e} time_bound={TIME_ATOL_SECONDS:.9e} verdict=PASS",
        actual.len(),
    );
}
