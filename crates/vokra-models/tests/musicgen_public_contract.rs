//! Strict binding smoke test for the five public MusicGen/AudioGen GGUFs.
//!
//! These artifacts are all at or above the repository's large-model boundary,
//! so this test must run on VAST rather than the maintainer Mac. Point any of
//! the documented environment variables at the corresponding fixed-revision
//! public file; unset rows skip cleanly. Setting an environment variable opts
//! that row into a hard failure on metadata or complete tensor-manifest drift.

use std::path::PathBuf;

use vokra_core::gguf::GgufFile;
use vokra_models::audiogen::{AudioGen, AudioGenArtifactLayout};
use vokra_models::musicgen::{
    MusicGen, MusicGenArtifactLayout, MusicGenVariant, NAME_LARGE, NAME_MEDIUM, NAME_MELODY,
    NAME_SMALL,
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
