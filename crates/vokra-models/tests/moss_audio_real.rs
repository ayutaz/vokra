//! Gated real-checkpoint parity for both MOSS-Audio Instruct releases.
//!
//! The reference directory is produced only on VAST by
//! `tools/parity/moss_audio/dump_reference.py`, which imports the exact
//! official OpenMOSS source commit and calls its model/processor classes.
//! Unset GGUF/reference variables skip honestly; no synthetic model or
//! fabricated numeric fixture is hidden here.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::BackendKind;
use vokra_models::moss_audio::{
    DEFAULT_USER_PROMPT, MossAudio, MossAudioGenerationOptions, MossAudioVariant, SAMPLE_RATE,
};

const REFERENCE_SCHEMA: &str = "vokra-moss-audio-reference-v1";
const SOURCE_CODE_REVISION: &str = "5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883";
const CONFIGURATION_SOURCE_SHA256: &str =
    "e597dca441ff7fb58a5ec43186fafdfce19f31dada4955b4910059baa5d52ebd";
const MODELING_SOURCE_SHA256: &str =
    "a52513e518c68a0ba7c636a1ab0e12f7755ceebd0ae033235dc5e2551bfcbf9c";
const PROCESSING_SOURCE_SHA256: &str =
    "05fb788cbdc6482eded8d70f7d2f524bc0cdca47d001acab5661c11f02cc6fe6";
const REFERENCE_AUDIO_SHA256: &str =
    "241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a";
const FP32_ATOL: f32 = 0.01;

#[derive(Debug)]
struct Reference {
    pcm: Vec<f32>,
    prompt_ids: Vec<u32>,
    primary_audio: Vec<f32>,
    deepstack_audio: [Vec<f32>; 3],
    audio_frames: usize,
    hidden_size: usize,
    prompt: String,
    max_new_tokens: usize,
    generated_ids: Vec<u32>,
    result_text: String,
}

#[derive(Debug)]
struct Actual {
    primary_audio: Vec<f32>,
    deepstack_audio: [Vec<f32>; 3],
    audio_frames: usize,
    hidden_size: usize,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    result_text: String,
}

fn read_manifest(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).expect("read MOSS-Audio reference manifest");
    let mut values = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("manifest line {} has no '=': {line:?}", line_number + 1));
        assert!(!key.is_empty(), "empty manifest key");
        assert!(
            values.insert(key.to_owned(), value.to_owned()).is_none(),
            "duplicate manifest key {key:?}"
        );
    }
    values
}

fn manifest_value<'a>(values: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    values
        .get(key)
        .unwrap_or_else(|| panic!("reference manifest is missing {key:?}"))
}

fn manifest_usize(values: &BTreeMap<String, String>, key: &str) -> usize {
    manifest_value(values, key)
        .parse()
        .unwrap_or_else(|_| panic!("reference manifest {key:?} is not usize"))
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?} is truncated");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{path:?} contains a non-finite reference value"
    );
    values
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?} is truncated");
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], reference: &[f32], label: &str) {
    assert_eq!(actual.len(), reference.len(), "{label} length");
    assert!(!actual.is_empty(), "{label} must not be empty");
    let mut worst_index = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (index, (&left, &right)) in actual.iter().zip(reference).enumerate() {
        assert!(
            left.is_finite(),
            "{label}: runtime value {index} is non-finite"
        );
        let delta = (left - right).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            worst_index = index;
        }
    }
    let mean_abs = sum_abs / actual.len() as f64;
    eprintln!(
        "MOSS_AUDIO_PARITY {label} max_abs={max_abs:.9e} index={worst_index} actual={:.9e} reference={:.9e} mean_abs={mean_abs:.9e} atol={FP32_ATOL:.9e}",
        actual[worst_index], reference[worst_index]
    );
    assert!(
        max_abs <= FP32_ATOL,
        "{label}: max_abs={max_abs} at {worst_index} exceeds FP32 atol={FP32_ATOL}; diagnose the worst element instead of widening the bound"
    );
}

fn variant_slug(variant: MossAudioVariant) -> &'static str {
    match variant {
        MossAudioVariant::B4Instruct => "4b",
        MossAudioVariant::B8Instruct => "8b",
    }
}

fn tensor_count(_variant: MossAudioVariant) -> usize {
    901
}

impl Reference {
    fn load(directory: &Path, variant: MossAudioVariant) -> Self {
        let manifest = read_manifest(&directory.join("manifest.txt"));
        assert_eq!(manifest_value(&manifest, "schema"), REFERENCE_SCHEMA);
        assert_eq!(manifest_value(&manifest, "variant"), variant_slug(variant));
        assert_eq!(
            manifest_value(&manifest, "model_name"),
            variant.model_name()
        );
        assert_eq!(
            manifest_value(&manifest, "upstream_repo"),
            variant.upstream_hf()
        );
        assert_eq!(
            manifest_value(&manifest, "upstream_revision"),
            variant.upstream_revision()
        );
        assert_eq!(
            manifest_value(&manifest, "source_code_revision"),
            SOURCE_CODE_REVISION
        );
        assert_eq!(
            manifest_value(&manifest, "configuration_source_sha256"),
            CONFIGURATION_SOURCE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "modeling_source_sha256"),
            MODELING_SOURCE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "processing_source_sha256"),
            PROCESSING_SOURCE_SHA256
        );
        assert_eq!(
            manifest_value(&manifest, "config_sha256"),
            variant.config_sha256()
        );
        assert_eq!(
            manifest_value(&manifest, "source_audio_sha256"),
            REFERENCE_AUDIO_SHA256
        );
        assert_eq!(manifest_value(&manifest, "transformers_version"), "4.57.1");
        assert!(
            manifest_value(&manifest, "torch_version").starts_with("2.9.1"),
            "reference must use the pinned Torch 2.9.1 line"
        );
        assert_eq!(
            manifest_usize(&manifest, "sample_rate"),
            SAMPLE_RATE as usize
        );
        assert_eq!(
            manifest_usize(&manifest, "tensor_count"),
            tensor_count(variant)
        );

        let pcm = read_f32(&directory.join("pcm.f32le"));
        let prompt_ids = read_u32(&directory.join("prompt_ids.u32le"));
        let primary_audio = read_f32(&directory.join("primary_audio.f32le"));
        let deepstack_audio = [
            read_f32(&directory.join("deepstack_audio_0.f32le")),
            read_f32(&directory.join("deepstack_audio_1.f32le")),
            read_f32(&directory.join("deepstack_audio_2.f32le")),
        ];
        let generated_ids = read_u32(&directory.join("generated_ids.u32le"));
        let audio_frames = manifest_usize(&manifest, "audio_frames");
        let hidden_size = manifest_usize(&manifest, "hidden_size");
        let expected_values = audio_frames * hidden_size;
        assert_eq!(pcm.len(), manifest_usize(&manifest, "pcm_samples"));
        assert_eq!(prompt_ids.len(), manifest_usize(&manifest, "prompt_tokens"));
        assert_eq!(
            generated_ids.len(),
            manifest_usize(&manifest, "generated_tokens")
        );
        assert_eq!(primary_audio.len(), expected_values);
        for (index, values) in deepstack_audio.iter().enumerate() {
            assert_eq!(values.len(), expected_values, "DeepStack reference {index}");
        }
        assert_eq!(
            hidden_size,
            variant.config().text.hidden_size as usize,
            "reference hidden width"
        );
        let prompt =
            std::fs::read_to_string(directory.join("prompt.txt")).expect("read prompt text");
        assert_eq!(prompt, DEFAULT_USER_PROMPT, "official example prompt");
        let result_text =
            std::fs::read_to_string(directory.join("result_text.txt")).expect("read result text");
        let max_new_tokens = manifest_usize(&manifest, "max_new_tokens");
        assert!(
            (1..=16).contains(&max_new_tokens),
            "reference max_new_tokens must remain in 1..=16"
        );
        Self {
            pcm,
            prompt_ids,
            primary_audio,
            deepstack_audio,
            audio_frames,
            hidden_size,
            prompt,
            max_new_tokens,
            generated_ids,
            result_text,
        }
    }
}

fn execute(
    gguf: &Path,
    reference: &Reference,
    variant: MossAudioVariant,
    backend: BackendKind,
) -> Actual {
    let model = MossAudio::open_mapped(gguf, backend)
        .unwrap_or_else(|error| panic!("open mapped MOSS-Audio on {backend:?}: {error}"));
    assert_eq!(model.backend(), backend);
    assert_eq!(model.checkpoint().variant(), variant);
    assert_eq!(model.checkpoint().tensor_count(), tensor_count(variant));
    assert_eq!(model.checkpoint().model_name(), variant.model_name());
    assert!(model.checkpoint().has_text_tokenizer());

    let audio = model
        .encode_audio(&reference.pcm, SAMPLE_RATE)
        .unwrap_or_else(|error| panic!("encode MOSS-Audio on {backend:?}: {error}"));
    let prompt_ids = model
        .tokenizer()
        .expect("authenticated MOSS-Audio tokenizer")
        .prompt_ids(audio.frames(), &reference.prompt)
        .expect("build official MOSS-Audio prompt");
    let options = MossAudioGenerationOptions::new(reference.max_new_tokens);
    let generated_ids = model
        .generate_tokens(&reference.pcm, SAMPLE_RATE, &prompt_ids, &options)
        .unwrap_or_else(|error| panic!("generate MOSS-Audio on {backend:?}: {error}"))
        .into_token_ids();
    let result_text = model
        .tokenizer()
        .expect("authenticated MOSS-Audio tokenizer")
        .decode_generated_ids(&generated_ids)
        .expect("decode generated MOSS-Audio ids");
    Actual {
        primary_audio: audio.values().to_vec(),
        deepstack_audio: audio.deepstack_values().map(|values| values.to_vec()),
        audio_frames: audio.frames(),
        hidden_size: audio.hidden_size(),
        prompt_ids,
        generated_ids,
        result_text,
    }
}

fn parity_from_env(gguf_variable: &str, reference_variable: &str, variant: MossAudioVariant) {
    let (Ok(gguf), Ok(reference_dir)) = (
        std::env::var(gguf_variable),
        std::env::var(reference_variable),
    ) else {
        eprintln!(
            "skip MOSS-Audio official parity: set both {gguf_variable} and {reference_variable}"
        );
        return;
    };
    let reference = Reference::load(Path::new(&reference_dir), variant);
    let actual = execute(Path::new(&gguf), &reference, variant, BackendKind::Cpu);
    assert_eq!(actual.audio_frames, reference.audio_frames);
    assert_eq!(actual.hidden_size, reference.hidden_size);
    assert_close(
        &actual.primary_audio,
        &reference.primary_audio,
        &format!("{} primary_audio CPU_vs_official", variant.model_name()),
    );
    for index in 0..3 {
        assert_close(
            &actual.deepstack_audio[index],
            &reference.deepstack_audio[index],
            &format!(
                "{} deepstack_audio_{index} CPU_vs_official",
                variant.model_name()
            ),
        );
    }
    assert_eq!(
        actual.prompt_ids, reference.prompt_ids,
        "official prompt ids"
    );
    assert_eq!(
        actual.generated_ids, reference.generated_ids,
        "official greedy token ids"
    );
    assert_eq!(
        actual.result_text, reference.result_text,
        "official decoded text"
    );
    eprintln!(
        "MOSS_AUDIO_PARITY {} CPU_vs_official token_ids=exact text=exact PASS",
        variant.model_name()
    );
}

#[test]
fn moss_audio_4b_cpu_matches_official_reference() {
    parity_from_env(
        "VOKRA_MOSS_AUDIO_4B_GGUF",
        "VOKRA_MOSS_AUDIO_4B_REFERENCE_DIR",
        MossAudioVariant::B4Instruct,
    );
}

#[test]
fn moss_audio_8b_cpu_matches_official_reference() {
    parity_from_env(
        "VOKRA_MOSS_AUDIO_8B_GGUF",
        "VOKRA_MOSS_AUDIO_8B_REFERENCE_DIR",
        MossAudioVariant::B8Instruct,
    );
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn moss_audio_real_metal_matches_cpu_exact_greedy() {
    use std::path::PathBuf;

    use vokra_core::VokraError;
    use vokra_models::Compute;
    use vokra_models::moss_audio::MOSS_AUDIO_HOT_OPS;

    match Compute::for_backend(BackendKind::Metal, MOSS_AUDIO_HOT_OPS) {
        Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
        Err(VokraError::BackendUnavailable(error)) => {
            eprintln!("skip MOSS-Audio Metal parity: no Metal device ({error})");
            return;
        }
        Err(error) => panic!("MOSS-Audio claims Metal coverage but preflight failed: {error}"),
    }

    let cases = [
        (
            "VOKRA_MOSS_AUDIO_4B_GGUF",
            "VOKRA_MOSS_AUDIO_4B_REFERENCE_DIR",
            MossAudioVariant::B4Instruct,
        ),
        (
            "VOKRA_MOSS_AUDIO_8B_GGUF",
            "VOKRA_MOSS_AUDIO_8B_REFERENCE_DIR",
            MossAudioVariant::B8Instruct,
        ),
    ];
    for (gguf_variable, reference_variable, variant) in cases {
        let (Ok(gguf), Ok(reference_dir)) = (
            std::env::var(gguf_variable),
            std::env::var(reference_variable),
        ) else {
            eprintln!(
                "skip {} Metal parity: set both {gguf_variable} and {reference_variable}",
                variant.model_name()
            );
            continue;
        };
        let gguf = PathBuf::from(gguf);
        let reference = Reference::load(Path::new(&reference_dir), variant);
        let cpu = execute(&gguf, &reference, variant, BackendKind::Cpu);
        let metal = execute(&gguf, &reference, variant, BackendKind::Metal);
        assert_eq!(metal.audio_frames, cpu.audio_frames);
        assert_eq!(metal.hidden_size, cpu.hidden_size);
        assert_close(
            &metal.primary_audio,
            &cpu.primary_audio,
            &format!("{} primary_audio Metal_vs_CPU", variant.model_name()),
        );
        for index in 0..3 {
            assert_close(
                &metal.deepstack_audio[index],
                &cpu.deepstack_audio[index],
                &format!(
                    "{} deepstack_audio_{index} Metal_vs_CPU",
                    variant.model_name()
                ),
            );
        }
        assert_eq!(metal.prompt_ids, cpu.prompt_ids, "Metal prompt ids");
        assert_eq!(
            metal.generated_ids, cpu.generated_ids,
            "Metal greedy token ids must exactly match CPU"
        );
        assert_eq!(metal.result_text, cpu.result_text, "Metal decoded text");
        eprintln!(
            "MOSS_AUDIO_PARITY {} Metal_vs_CPU token_ids=exact text=exact PASS",
            variant.model_name()
        );
    }
}
