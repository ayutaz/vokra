//! Strict binding smoke test for the five public MusicGen/AudioGen GGUFs.
//!
//! These artifacts are all at or above the repository's large-model boundary,
//! so this test must run on VAST rather than the maintainer Mac. Point any of
//! the documented environment variables at the corresponding fixed-revision
//! public file; unset rows skip cleanly. Setting an environment variable opts
//! that row into a hard failure on metadata or complete tensor-manifest drift.

use std::path::PathBuf;

use vokra_core::CompliancePolicy;
use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::audiocraft_encodec::FRAME_HOP;
use vokra_models::audiocraft_lm::AudioCraftGenerationConfig;
use vokra_models::audiogen::{AudioGen, AudioGenArtifactLayout};
use vokra_models::musicgen::{
    MusicGen, MusicGenArtifactLayout, MusicGenCompanion, MusicGenVariant, NAME_LARGE, NAME_MEDIUM,
    NAME_MELODY, NAME_SMALL,
};

struct MusicGenRow {
    env: &'static str,
    name: &'static str,
    variant: MusicGenVariant,
    layout: MusicGenArtifactLayout,
    tensors: usize,
}

const MUSICGEN_ROWS: &[MusicGenRow] = &[
    MusicGenRow {
        env: "VOKRA_MUSICGEN_SMALL_GGUF",
        name: NAME_SMALL,
        variant: MusicGenVariant::Small,
        layout: MusicGenArtifactLayout::TransformersComposite,
        tensors: 612,
    },
    MusicGenRow {
        env: "VOKRA_MUSICGEN_MEDIUM_GGUF",
        name: NAME_MEDIUM,
        variant: MusicGenVariant::Medium,
        layout: MusicGenArtifactLayout::AudioCraftLm,
        tensors: 588,
    },
    MusicGenRow {
        env: "VOKRA_MUSICGEN_LARGE_GGUF",
        name: NAME_LARGE,
        variant: MusicGenVariant::Large,
        layout: MusicGenArtifactLayout::AudioCraftLm,
        tensors: 588,
    },
    MusicGenRow {
        env: "VOKRA_MUSICGEN_MELODY_GGUF",
        name: NAME_MELODY,
        variant: MusicGenVariant::Melody,
        layout: MusicGenArtifactLayout::TransformersComposite,
        tensors: 710,
    },
];

#[test]
fn public_musicgen_and_audiogen_artifacts_match_strict_runtime_contracts() {
    let mut opted_in = 0usize;
    for row in MUSICGEN_ROWS {
        let Some(path) = std::env::var_os(row.env).map(PathBuf::from) else {
            continue;
        };
        opted_in += 1;
        let file = GgufFile::open(&path)
            .unwrap_or_else(|error| panic!("open {}={path:?}: {error}", row.env));
        let model = MusicGen::from_gguf(&file)
            .unwrap_or_else(|error| panic!("strictly bind {}={path:?}: {error}", row.env));
        assert_eq!(model.variant(), row.variant, "{} variant", row.name);
        assert_eq!(model.artifact_layout(), row.layout, "{} layout", row.name);
        assert_eq!(
            model.tensor_count(),
            row.tensors,
            "{} tensor count",
            row.name
        );
    }

    if let Some(path) = std::env::var_os("VOKRA_AUDIOGEN_MEDIUM_GGUF").map(PathBuf::from) {
        opted_in += 1;
        let file = GgufFile::open(&path)
            .unwrap_or_else(|error| panic!("open VOKRA_AUDIOGEN_MEDIUM_GGUF={path:?}: {error}"));
        let model = AudioGen::from_gguf(&file).unwrap_or_else(|error| {
            panic!("strictly bind VOKRA_AUDIOGEN_MEDIUM_GGUF={path:?}: {error}")
        });
        assert_eq!(
            model.artifact_layout(),
            AudioGenArtifactLayout::AudioCraftLm
        );
        assert_eq!(model.tensor_count(), 588);
    }

    if opted_in == 0 {
        eprintln!(
            "skipping public MusicGen/AudioGen strict contracts: set one or more of \
             VOKRA_MUSICGEN_{{SMALL,MEDIUM,LARGE,MELODY}}_GGUF or \
             VOKRA_AUDIOGEN_MEDIUM_GGUF on VAST"
        );
    }
}

/// Real-weight end-to-end route smoke for the two public LM-only releases.
///
/// This is deliberately ignored because the three fixed public artifacts total
/// more than 12 GB. Run it on VAST for CPU, or on a disposable external Apple
/// Silicon host for Metal. It is not an independent AudioCraft numerical
/// oracle: it proves strict binding, same-backend composition and a complete
/// finite T5 -> target LM -> companion EnCodec route only.
#[test]
#[ignore = "requires exact public MusicGen Small + Medium/Large real weights"]
fn public_musicgen_lm_only_companion_generates_finite_pcm() {
    const FRAMES: usize = 3;

    let backend_name =
        std::env::var("VOKRA_MUSICGEN_ROUTE_BACKEND").unwrap_or_else(|_| "cpu".to_owned());
    let backend = match backend_name.as_str() {
        "cpu" => BackendKind::Cpu,
        "metal" => {
            assert!(
                cfg!(feature = "metal"),
                "VOKRA_MUSICGEN_ROUTE_BACKEND=metal requires --features metal"
            );
            BackendKind::Metal
        }
        other => panic!("VOKRA_MUSICGEN_ROUTE_BACKEND must be cpu or metal, got {other:?}"),
    };
    let small_path = std::env::var_os("VOKRA_MUSICGEN_SMALL_GGUF")
        .map(PathBuf::from)
        .expect("VOKRA_MUSICGEN_SMALL_GGUF is required for the companion route smoke");
    let policy = CompliancePolicy::strict().with_research_license(true);
    let companion =
        MusicGenCompanion::from_path_with_policy_and_backend(&small_path, &policy, backend)
            .unwrap_or_else(|error| panic!("bind exact Small companion {small_path:?}: {error}"));
    assert_eq!(companion.backend(), backend);

    let targets = [
        (
            "VOKRA_MUSICGEN_MEDIUM_GGUF",
            NAME_MEDIUM,
            MusicGenVariant::Medium,
        ),
        (
            "VOKRA_MUSICGEN_LARGE_GGUF",
            NAME_LARGE,
            MusicGenVariant::Large,
        ),
    ];
    let mut exercised = 0usize;
    for (env_name, release_name, expected_variant) in targets {
        let Some(target_path) = std::env::var_os(env_name).map(PathBuf::from) else {
            continue;
        };
        exercised += 1;
        let target = MusicGen::from_path_with_policy_and_backend(&target_path, &policy, backend)
            .unwrap_or_else(|error| {
                panic!("bind exact {release_name} target {target_path:?}: {error}")
            });
        assert_eq!(target.variant(), expected_variant);
        assert_eq!(
            target.artifact_layout(),
            MusicGenArtifactLayout::AudioCraftLm
        );
        assert_eq!(target.backend(), backend);

        let pcm = target
            .generate_from_token_ids_with_companion(
                &companion,
                &[1],
                None,
                &[0],
                None,
                &AudioCraftGenerationConfig::greedy(FRAMES),
            )
            .unwrap_or_else(|error| {
                panic!("generate {release_name} with exact Small companion: {error}")
            });
        assert_eq!(pcm.len(), FRAMES * FRAME_HOP, "{release_name} PCM length");
        assert!(
            pcm.iter().all(|sample| sample.is_finite()),
            "{release_name} emitted non-finite PCM"
        );
        let max_abs = pcm.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!(max_abs > 0.0, "{release_name} emitted only zero PCM");
        println!(
            "MUSICGEN_COMPANION_ROUTE backend={backend_name} target={release_name} \
             frames={FRAMES} samples={} max_abs={max_abs:.9e} verdict=PASS",
            pcm.len()
        );
    }
    assert!(
        exercised > 0,
        "set VOKRA_MUSICGEN_MEDIUM_GGUF and/or VOKRA_MUSICGEN_LARGE_GGUF"
    );
}
