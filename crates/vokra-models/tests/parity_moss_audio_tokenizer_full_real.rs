//! VAST-only real-weight parity for MOSS Audio Tokenizer Full.
//!
//! Generate the independent fixture with
//! `tools/parity/moss_audio_tokenizer_dump_reference.py --variant full`, then
//! run this ignored test with both environment paths set. The 7 GB checkpoint
//! is always opened through `vokra-mmap`; it is never copied into a resident
//! test buffer.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_models::moss_audio_tokenizer::{MossAudioTokenizer, MossAudioTokenizerVariant};

const OFFICIAL_REVISION: &str = "10cda397411ce6ddb802173f8d8a6c9fee3b845e";
const OFFICIAL_MODEL_SOURCE_SHA256: &str =
    "65cae7744845f1b8ac65957e918cea508efe331a38e87b882b7530b6c8d7caa5";
const OFFICIAL_CONFIG_SOURCE_SHA256: &str =
    "349b7ff7e1b3f160f9c80df9a0311672b326b8b73e90459122fb39e6878962bf";
const FP32_ATOL: f32 = 0.01;

struct Reference {
    frames: usize,
    num_quantizers: usize,
    codes: Vec<u32>,
    audio: Vec<f32>,
}

fn parse_usize(value: Option<&str>, label: &str) -> usize {
    value
        .unwrap_or_else(|| panic!("MOSS Full reference is missing {label}"))
        .parse()
        .unwrap_or_else(|error| panic!("MOSS Full reference {label} is invalid: {error}"))
}

fn load_reference(path: &Path) -> Reference {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read MOSS Full reference {}: {error}", path.display()));
    let source = format!("source,full,OpenMOSS-Team/MOSS-Audio-Tokenizer,{OFFICIAL_REVISION}");
    assert!(
        text.lines().any(|line| line == source),
        "MOSS Full reference lost its pinned official source"
    );
    for (kind, digest) in [
        ("model", OFFICIAL_MODEL_SOURCE_SHA256),
        ("config", OFFICIAL_CONFIG_SOURCE_SHA256),
    ] {
        assert!(
            text.lines().any(|line| {
                line.starts_with(&format!("source_file,{kind},"))
                    && line.ends_with(&format!(",{digest}"))
            }),
            "MOSS Full reference lost its pinned {kind} source digest"
        );
    }
    assert!(
        text.lines().any(|line| line.starts_with("runtime,torch-")),
        "MOSS Full reference must record its Torch environment"
    );
    assert!(
        text.lines()
            .any(|line| line.starts_with("environment,cpu,")),
        "MOSS Full reference must record CPU/ISA context"
    );
    assert!(
        text.lines()
            .any(|line| line.starts_with("environment,device,")),
        "MOSS Full reference must record the execution device"
    );
    for label in [
        "quantizer",
        "decoder_0",
        "decoder_1",
        "decoder_2",
        "decoder_3",
        "decoder_4",
        "decoder_5",
        "decoder_6",
        "decoder_7",
        "audio",
    ] {
        assert!(
            text.lines()
                .any(|line| line.starts_with(&format!("tensor,{label},"))),
            "MOSS Full reference is missing official tap {label}"
        );
    }

    let contract = text
        .lines()
        .find(|line| line.starts_with("contract,"))
        .expect("MOSS Full reference is missing contract");
    let mut contract = contract.split(',');
    assert_eq!(contract.next(), Some("contract"));
    let frames = parse_usize(contract.next(), "frames");
    let num_quantizers = parse_usize(contract.next(), "num_quantizers");
    assert_eq!(parse_usize(contract.next(), "codebook_size"), 1_024);
    assert_eq!(parse_usize(contract.next(), "sample_rate"), 24_000);
    assert_eq!(parse_usize(contract.next(), "channels"), 1);
    assert_eq!(parse_usize(contract.next(), "frame_hop"), 1_920);
    assert_eq!(contract.next(), None, "unexpected MOSS Full contract field");
    assert!(frames > 0);
    assert!((1..=32).contains(&num_quantizers));

    let codes = text
        .lines()
        .find_map(|line| line.strip_prefix("codes,"))
        .expect("MOSS Full reference is missing codes")
        .split(',')
        .map(|value| value.parse().expect("invalid MOSS Full reference code"))
        .collect::<Vec<u32>>();
    assert_eq!(codes.len(), frames * num_quantizers);
    assert!(codes.iter().all(|code| *code < 1_024));

    let audio_line = text
        .lines()
        .find(|line| line.starts_with("tensor,audio,"))
        .expect("MOSS Full reference is missing audio");
    let mut audio_fields = audio_line.split(',');
    assert_eq!(audio_fields.next(), Some("tensor"));
    assert_eq!(audio_fields.next(), Some("audio"));
    let expected_shape = format!("1x1x{}", frames * 1_920);
    assert_eq!(audio_fields.next(), Some(expected_shape.as_str()));
    let audio = audio_fields
        .map(|value| value.parse().expect("invalid MOSS Full reference sample"))
        .collect::<Vec<f32>>();
    assert_eq!(audio.len(), frames * 1_920);
    assert!(audio.iter().all(|sample| sample.is_finite()));

    Reference {
        frames,
        num_quantizers,
        codes,
        audio,
    }
}

fn max_abs(actual: &[f32], expected: &[f32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty MOSS Full waveform")
}

#[test]
#[ignore = "requires the 7 GB public GGUF and an independently generated VAST reference"]
fn official_full_decode_matches_cpu_and_optional_metal() {
    let gguf_path = PathBuf::from(
        std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_FULL_GGUF")
            .expect("set VOKRA_MOSS_AUDIO_TOKENIZER_FULL_GGUF for ignored real parity"),
    );
    let reference_path = PathBuf::from(
        std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_FULL_REFERENCE")
            .expect("set VOKRA_MOSS_AUDIO_TOKENIZER_FULL_REFERENCE for ignored real parity"),
    );
    let reference = load_reference(&reference_path);
    let file = Arc::new(vokra_mmap::open_gguf(&gguf_path).expect("mmap MOSS Full GGUF"));
    let cpu = MossAudioTokenizer::from_gguf_mapped_with_backend(file, BackendKind::Cpu)
        .expect("strict mapping-backed MOSS Full CPU bind");
    assert_eq!(cpu.variant(), MossAudioTokenizerVariant::Full);
    let cpu_audio = cpu
        .decode_frame_major(&reference.codes, reference.frames, reference.num_quantizers)
        .expect("MOSS Full CPU decode");
    assert_eq!(cpu_audio.sample_rate, 24_000);
    assert_eq!(cpu_audio.channels, 1);
    assert_eq!(cpu_audio.samples_per_channel, reference.frames * 1_920);
    let (index, delta) = max_abs(&cpu_audio.pcm, &reference.audio);
    eprintln!(
        "MOSS_AUDIO_TOKENIZER_FULL_PARITY backend=cpu max_abs={delta:.9e} index={index} actual={:.9e} reference={:.9e} bound={FP32_ATOL:.9e}",
        cpu_audio.pcm[index], reference.audio[index]
    );
    assert!(
        delta <= FP32_ATOL,
        "MOSS Full CPU/official max_abs={delta:.9e} exceeds {FP32_ATOL:.9e}"
    );

    if std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_FULL_METAL_PARITY").is_some() {
        let metal = cpu.clone().with_backend(BackendKind::Metal);
        let metal_audio = metal
            .decode_frame_major(&reference.codes, reference.frames, reference.num_quantizers)
            .expect("MOSS Full Metal decode");
        let (index, delta) = max_abs(&metal_audio.pcm, &cpu_audio.pcm);
        eprintln!(
            "MOSS_AUDIO_TOKENIZER_FULL_PARITY backend=metal max_abs={delta:.9e} index={index} metal={:.9e} cpu={:.9e} bound={FP32_ATOL:.9e}",
            metal_audio.pcm[index], cpu_audio.pcm[index]
        );
        assert!(
            delta <= FP32_ATOL,
            "MOSS Full Metal/CPU max_abs={delta:.9e} exceeds {FP32_ATOL:.9e}"
        );
    }
}
