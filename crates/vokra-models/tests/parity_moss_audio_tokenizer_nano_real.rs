//! VAST-only real-weight validation for the corrected MOSS Audio Tokenizer Nano.
//!
//! The CSV input is produced by the independent upstream custom-code oracle
//! (`tools/parity/moss_audio_tokenizer_dump_reference.py --variant nano`).
//! There is no reviewed Nano numeric bound yet, so this test deliberately
//! records measurements and never emits a numeric PASS.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_models::moss_audio_tokenizer::{MossAudioTokenizer, MossAudioTokenizerVariant};

const OFFICIAL_REPO: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano";
const OFFICIAL_REVISION: &str = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68";
const CODEBOOK_SIZE: usize = 1_024;
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: usize = 2;
const FRAME_HOP: usize = 3_840;
const MAX_QUANTIZERS: usize = 16;

struct Reference {
    frames: usize,
    num_quantizers: usize,
    codes: Vec<u32>,
    interleaved_audio: Vec<f32>,
}

fn parse_usize(value: Option<&str>, label: &str) -> usize {
    value
        .unwrap_or_else(|| panic!("MOSS Nano reference is missing {label}"))
        .parse()
        .unwrap_or_else(|error| panic!("MOSS Nano reference {label} is invalid: {error}"))
}

fn parse_f32(value: &str, label: &str) -> f32 {
    value
        .parse()
        .unwrap_or_else(|error| panic!("MOSS Nano reference {label} is invalid: {error}"))
}

fn load_reference(path: &Path) -> Reference {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read MOSS Nano reference {}: {error}", path.display()));
    let source = format!("source,nano,{OFFICIAL_REPO},{OFFICIAL_REVISION}");
    assert!(
        text.lines().any(|line| line == source),
        "MOSS Nano reference lost its pinned official source"
    );
    assert!(
        text.lines().any(|line| line.starts_with("runtime,torch-")),
        "MOSS Nano reference must record its Torch environment"
    );
    assert!(
        text.lines()
            .any(|line| line.starts_with("environment,cpu,")),
        "MOSS Nano reference must record CPU/ISA context"
    );
    assert!(
        text.lines()
            .any(|line| line.starts_with("environment,device,")),
        "MOSS Nano reference must record execution device"
    );
    for label in ["quantizer", "audio"] {
        assert!(
            text.lines()
                .any(|line| line.starts_with(&format!("tensor,{label},"))),
            "MOSS Nano reference is missing official tap {label}"
        );
    }
    for label in ["model", "config"] {
        assert!(
            text.lines()
                .any(|line| line.starts_with(&format!("source_file,{label},"))),
            "MOSS Nano reference is missing upstream {label} source identity"
        );
    }

    let contract = text
        .lines()
        .find(|line| line.starts_with("contract,"))
        .expect("MOSS Nano reference is missing contract");
    let mut fields = contract.split(',');
    assert_eq!(fields.next(), Some("contract"));
    let frames = parse_usize(fields.next(), "frames");
    let num_quantizers = parse_usize(fields.next(), "num_quantizers");
    assert_eq!(parse_usize(fields.next(), "codebook_size"), CODEBOOK_SIZE);
    assert_eq!(
        parse_usize(fields.next(), "sample_rate"),
        SAMPLE_RATE as usize
    );
    assert_eq!(parse_usize(fields.next(), "channels"), CHANNELS);
    assert_eq!(parse_usize(fields.next(), "frame_hop"), FRAME_HOP);
    assert_eq!(fields.next(), None, "unexpected MOSS Nano contract field");
    assert!(frames > 0);
    assert!((1..=MAX_QUANTIZERS).contains(&num_quantizers));

    let codes = text
        .lines()
        .find_map(|line| line.strip_prefix("codes,"))
        .expect("MOSS Nano reference is missing codes")
        .split(',')
        .map(|value| value.parse().expect("invalid MOSS Nano reference code"))
        .collect::<Vec<u32>>();
    assert_eq!(codes.len(), frames * num_quantizers);
    assert!(codes.iter().all(|code| *code < CODEBOOK_SIZE as u32));

    let audio_line = text
        .lines()
        .find(|line| line.starts_with("tensor,audio,"))
        .expect("MOSS Nano reference is missing restored stereo audio");
    let mut audio_fields = audio_line.split(',');
    assert_eq!(audio_fields.next(), Some("tensor"));
    assert_eq!(audio_fields.next(), Some("audio"));
    let samples_per_channel = frames * FRAME_HOP;
    let expected_shape = format!("1x{CHANNELS}x{samples_per_channel}");
    assert_eq!(audio_fields.next(), Some(expected_shape.as_str()));
    let channel_major = audio_fields
        .enumerate()
        .map(|(index, value)| parse_f32(value, &format!("audio[{index}]")))
        .collect::<Vec<f32>>();
    assert_eq!(channel_major.len(), CHANNELS * samples_per_channel);
    assert!(channel_major.iter().all(|sample| sample.is_finite()));

    let mut interleaved_audio = Vec::with_capacity(channel_major.len());
    for sample in 0..samples_per_channel {
        for channel in 0..CHANNELS {
            interleaved_audio.push(channel_major[channel * samples_per_channel + sample]);
        }
    }
    Reference {
        frames,
        num_quantizers,
        codes,
        interleaved_audio,
    }
}

fn numeric_error(actual: &[f32], expected: &[f32]) -> (usize, f32, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut max_index = 0;
    let mut max_abs = 0.0_f32;
    let mut sum_squared = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite());
        let delta = (actual - expected).abs();
        if delta.total_cmp(&max_abs).is_gt() {
            max_index = index;
            max_abs = delta;
        }
        sum_squared += f64::from(delta) * f64::from(delta);
    }
    let rms = (sum_squared / actual.len() as f64).sqrt();
    (max_index, max_abs, rms)
}

#[test]
#[ignore = "requires the corrected Nano GGUF and independent official reference"]
fn official_nano_decode_matches_cpu_and_optional_metal() {
    let gguf_path = PathBuf::from(
        std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_NANO_GGUF")
            .expect("set VOKRA_MOSS_AUDIO_TOKENIZER_NANO_GGUF for ignored real validation"),
    );
    let reference_path = PathBuf::from(
        std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE")
            .expect("set VOKRA_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE for ignored real validation"),
    );
    let reference = load_reference(&reference_path);
    let file = Arc::new(vokra_mmap::open_gguf(&gguf_path).expect("mmap MOSS Nano GGUF"));
    let cpu = MossAudioTokenizer::from_gguf_mapped_with_backend(file, BackendKind::Cpu)
        .expect("strict mapping-backed MOSS Nano CPU bind");
    assert_eq!(cpu.variant(), MossAudioTokenizerVariant::Nano);
    assert!(
        !cpu.requires_metadata_repair(),
        "validation must use the correctly stamped replacement, not the historical public Nano file"
    );
    let cpu_audio = cpu
        .decode_frame_major(&reference.codes, reference.frames, reference.num_quantizers)
        .expect("MOSS Nano CPU decode");
    assert_eq!(cpu_audio.sample_rate, SAMPLE_RATE);
    assert_eq!(cpu_audio.channels, CHANNELS);
    assert_eq!(cpu_audio.samples_per_channel, reference.frames * FRAME_HOP);
    let (index, max_abs, rms) = numeric_error(&cpu_audio.pcm, &reference.interleaved_audio);
    eprintln!(
        "MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs={max_abs:.9e} rms={rms:.9e} index={index} actual={:.9e} reference={:.9e}",
        cpu_audio.pcm[index], reference.interleaved_audio[index]
    );

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if std::env::var_os("VOKRA_MOSS_AUDIO_TOKENIZER_NANO_METAL_MEASUREMENT").is_some() {
        let metal = MossAudioTokenizer::open_mapped_with_backend(&gguf_path, BackendKind::Metal)
            .expect("strict mapping-backed MOSS Nano Metal bind");
        assert_eq!(metal.variant(), MossAudioTokenizerVariant::Nano);
        assert!(!metal.requires_metadata_repair());
        let metal_audio = metal
            .decode_frame_major(&reference.codes, reference.frames, reference.num_quantizers)
            .expect("MOSS Nano Metal decode");
        assert_eq!(metal_audio.pcm.len(), cpu_audio.pcm.len());
        let (index, max_abs, rms) = numeric_error(&metal_audio.pcm, &cpu_audio.pcm);
        eprintln!(
            "MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs={max_abs:.9e} rms={rms:.9e} index={index} metal={:.9e} cpu={:.9e}",
            metal_audio.pcm[index], cpu_audio.pcm[index]
        );
    }
}
