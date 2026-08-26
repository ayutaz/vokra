//! Gated real-checkpoint parity for both released Qwen3-ASR variants.
//!
//! The reference directory is produced only on VAST by
//! `tools/parity/qwen3_asr/dump_reference.py`, which imports the official
//! pinned `qwen-asr` package. Unset GGUF/reference variables skip honestly;
//! no fixture, checkpoint, download, or synthetic number is hidden here.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::BackendKind;
use vokra_models::qwen3_asr::{
    Qwen3Asr, Qwen3AsrCheckpoint, Qwen3AsrGenerationOptions, Qwen3AsrTranscription,
    Qwen3AsrVariant, SAMPLE_RATE,
};

const REFERENCE_SCHEMA: &str = "vokra-qwen3-asr-reference-v1";
const FP32_ATOL: f32 = 0.01;

#[derive(Debug)]
struct Reference {
    pcm: Vec<f32>,
    prompt_ids: Vec<u32>,
    audio_embeddings: Vec<f32>,
    audio_frames: usize,
    hidden_size: usize,
    context: String,
    forced_language: Option<String>,
    max_new_tokens: usize,
    generated_ids: Vec<u32>,
    result_language: String,
    result_text: String,
}

#[derive(Debug)]
struct Actual {
    audio_embeddings: Vec<f32>,
    audio_frames: usize,
    hidden_size: usize,
    prompt_ids: Vec<u32>,
    transcription: Qwen3AsrTranscription,
}

fn read_manifest(path: &Path) -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(path).expect("read Qwen3-ASR reference manifest");
    let mut values = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("manifest line {} has no '=': {line:?}", line_number + 1));
        assert!(
            !key.is_empty(),
            "empty manifest key at line {}",
            line_number + 1
        );
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
        "QWEN3_ASR_PARITY {label} max_abs={max_abs:.9e} index={worst_index} actual={:.9e} reference={:.9e} mean_abs={mean_abs:.9e} atol={FP32_ATOL:.9e}",
        actual[worst_index], reference[worst_index]
    );
    assert!(
        max_abs <= FP32_ATOL,
        "{label}: max_abs={max_abs} at {worst_index} exceeds FP32 atol={FP32_ATOL}; do not widen the bound to fit an observation"
    );
}

fn expected_variant_slug(variant: Qwen3AsrVariant) -> &'static str {
    match variant {
        Qwen3AsrVariant::B06 => "0.6b",
        Qwen3AsrVariant::B17 => "1.7b",
    }
}

impl Reference {
    fn load(directory: &Path, variant: Qwen3AsrVariant) -> Self {
        let manifest = read_manifest(&directory.join("manifest.txt"));
        assert_eq!(manifest_value(&manifest, "schema"), REFERENCE_SCHEMA);
        assert_eq!(
            manifest_value(&manifest, "variant"),
            expected_variant_slug(variant)
        );
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
            variant.source_revision()
        );
        assert_eq!(manifest_value(&manifest, "qwen_asr_version"), "0.0.6");
        assert_eq!(manifest_value(&manifest, "transformers_version"), "4.57.6");
        assert_eq!(
            manifest_usize(&manifest, "sample_rate"),
            SAMPLE_RATE as usize
        );
        assert_eq!(
            manifest_usize(&manifest, "tensor_count"),
            variant.tensor_count()
        );

        let pcm = read_f32(&directory.join("pcm.f32le"));
        let prompt_ids = read_u32(&directory.join("prompt_ids.u32le"));
        let audio_embeddings = read_f32(&directory.join("audio_embeddings.f32le"));
        let generated_ids = read_u32(&directory.join("generated_ids.u32le"));
        let audio_frames = manifest_usize(&manifest, "audio_frames");
        let hidden_size = manifest_usize(&manifest, "hidden_size");
        assert_eq!(pcm.len(), manifest_usize(&manifest, "pcm_samples"));
        assert_eq!(prompt_ids.len(), manifest_usize(&manifest, "prompt_tokens"));
        assert_eq!(
            generated_ids.len(),
            manifest_usize(&manifest, "generated_tokens")
        );
        assert_eq!(
            audio_embeddings.len(),
            audio_frames * hidden_size,
            "official projected-audio shape"
        );
        assert_eq!(
            hidden_size,
            variant.config().text.hidden_size as usize,
            "reference hidden width"
        );
        let context =
            std::fs::read_to_string(directory.join("context.txt")).expect("read reference context");
        let forced_language = std::fs::read_to_string(directory.join("forced_language.txt"))
            .expect("read forced language");
        let forced_language = (!forced_language.is_empty()).then_some(forced_language);
        let result_language = std::fs::read_to_string(directory.join("result_language.txt"))
            .expect("read reference result language");
        let result_text = std::fs::read_to_string(directory.join("result_text.txt"))
            .expect("read reference result text");
        Self {
            pcm,
            prompt_ids,
            audio_embeddings,
            audio_frames,
            hidden_size,
            context,
            forced_language,
            max_new_tokens: manifest_usize(&manifest, "max_new_tokens"),
            generated_ids,
            result_language,
            result_text,
        }
    }

    fn options(&self) -> Qwen3AsrGenerationOptions {
        let mut options = Qwen3AsrGenerationOptions::default();
        options.context = self.context.clone();
        options.language = self.forced_language.clone();
        options.max_new_tokens = self.max_new_tokens;
        options
    }
}

fn execute(
    gguf: &Path,
    reference: &Reference,
    variant: Qwen3AsrVariant,
    backend: BackendKind,
) -> Actual {
    let model = Qwen3Asr::open_mapped(gguf, backend)
        .unwrap_or_else(|error| panic!("open mapped Qwen3-ASR on {backend:?}: {error}"));
    assert_eq!(model.backend(), backend);
    assert_eq!(model.checkpoint().variant(), variant);
    assert_eq!(model.checkpoint().tensor_count(), variant.tensor_count());
    assert_eq!(model.checkpoint().model_name(), variant.model_name());

    let audio = model
        .encode_audio(&reference.pcm, SAMPLE_RATE)
        .unwrap_or_else(|error| panic!("encode Qwen3-ASR audio on {backend:?}: {error}"));
    let prompt_ids = model
        .tokenizer()
        .prompt_ids(
            audio.frames(),
            Some(&reference.context),
            reference.forced_language.as_deref(),
        )
        .expect("build authenticated Qwen3-ASR prompt");
    let transcription = model
        .transcribe_with_options(&reference.pcm, SAMPLE_RATE, &reference.options())
        .unwrap_or_else(|error| panic!("transcribe Qwen3-ASR on {backend:?}: {error}"));
    Actual {
        audio_embeddings: audio.values().to_vec(),
        audio_frames: audio.frames(),
        hidden_size: audio.hidden_size(),
        prompt_ids,
        transcription,
    }
}

fn bind_from_env(variable: &str, expected: Qwen3AsrVariant) {
    let Ok(path) = std::env::var(variable) else {
        eprintln!("skip Qwen3-ASR strict bind: set {variable}");
        return;
    };
    let file = vokra_mmap::open_gguf(&path).expect("open Qwen3-ASR GGUF through mmap");
    let checkpoint = Qwen3AsrCheckpoint::from_gguf(&file).expect("strict Qwen3-ASR bind");
    assert_eq!(checkpoint.variant(), expected);
    assert_eq!(checkpoint.tensor_count(), expected.tensor_count());
    assert_eq!(checkpoint.model_name(), expected.model_name());
}

fn parity_from_env(gguf_variable: &str, reference_variable: &str, expected: Qwen3AsrVariant) {
    let (Ok(gguf), Ok(reference_dir)) = (
        std::env::var(gguf_variable),
        std::env::var(reference_variable),
    ) else {
        eprintln!(
            "skip Qwen3-ASR official parity: set both {gguf_variable} and {reference_variable}"
        );
        return;
    };
    let reference = Reference::load(Path::new(&reference_dir), expected);
    let actual = execute(Path::new(&gguf), &reference, expected, BackendKind::Cpu);
    assert_eq!(actual.audio_frames, reference.audio_frames);
    assert_eq!(actual.hidden_size, reference.hidden_size);
    assert_close(
        &actual.audio_embeddings,
        &reference.audio_embeddings,
        &format!("{} projected_audio CPU_vs_official", expected.model_name()),
    );
    assert_eq!(
        actual.prompt_ids, reference.prompt_ids,
        "official prompt ids"
    );
    assert_eq!(
        actual.transcription.token_ids, reference.generated_ids,
        "official greedy token ids"
    );
    assert_eq!(
        actual.transcription.language, reference.result_language,
        "official parsed language"
    );
    assert_eq!(
        actual.transcription.text, reference.result_text,
        "official parsed text"
    );
    eprintln!(
        "QWEN3_ASR_PARITY {} CPU_vs_official token_ids=exact text=exact PASS",
        expected.model_name()
    );
}

#[test]
fn qwen3_asr_0_6b_strict_public_contract() {
    bind_from_env("VOKRA_QWEN3_ASR_0_6B_GGUF", Qwen3AsrVariant::B06);
}

#[test]
fn qwen3_asr_1_7b_strict_public_contract() {
    bind_from_env("VOKRA_QWEN3_ASR_1_7B_GGUF", Qwen3AsrVariant::B17);
}

#[test]
fn qwen3_asr_0_6b_cpu_matches_official_reference() {
    parity_from_env(
        "VOKRA_QWEN3_ASR_0_6B_GGUF",
        "VOKRA_QWEN3_ASR_0_6B_REFERENCE_DIR",
        Qwen3AsrVariant::B06,
    );
}

#[test]
fn qwen3_asr_1_7b_cpu_matches_official_reference() {
    parity_from_env(
        "VOKRA_QWEN3_ASR_1_7B_GGUF",
        "VOKRA_QWEN3_ASR_1_7B_REFERENCE_DIR",
        Qwen3AsrVariant::B17,
    );
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn qwen3_asr_real_metal_matches_cpu_exact_greedy() {
    use std::path::PathBuf;

    use vokra_core::VokraError;
    use vokra_models::Compute;
    use vokra_models::qwen3_asr::QWEN3_ASR_HOT_OPS;

    match Compute::for_backend(BackendKind::Metal, QWEN3_ASR_HOT_OPS) {
        Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
        Err(VokraError::BackendUnavailable(error)) => {
            eprintln!("skip Qwen3-ASR Metal parity: no Metal device ({error})");
            return;
        }
        Err(error) => panic!("Qwen3-ASR claims Metal coverage but preflight failed: {error}"),
    }

    let cases = [
        (
            "VOKRA_QWEN3_ASR_0_6B_GGUF",
            "VOKRA_QWEN3_ASR_0_6B_REFERENCE_DIR",
            Qwen3AsrVariant::B06,
        ),
        (
            "VOKRA_QWEN3_ASR_1_7B_GGUF",
            "VOKRA_QWEN3_ASR_1_7B_REFERENCE_DIR",
            Qwen3AsrVariant::B17,
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
            &metal.audio_embeddings,
            &cpu.audio_embeddings,
            &format!("{} projected_audio Metal_vs_CPU", variant.model_name()),
        );
        assert_eq!(metal.prompt_ids, cpu.prompt_ids, "Metal prompt ids");
        assert_eq!(
            metal.transcription.token_ids, cpu.transcription.token_ids,
            "Metal greedy token ids must exactly match CPU"
        );
        assert_eq!(
            metal.transcription.language, cpu.transcription.language,
            "Metal parsed language"
        );
        assert_eq!(
            metal.transcription.text, cpu.transcription.text,
            "Metal text"
        );
        eprintln!(
            "QWEN3_ASR_PARITY {} Metal_vs_CPU token_ids=exact text=exact PASS",
            variant.model_name()
        );
    }
}
