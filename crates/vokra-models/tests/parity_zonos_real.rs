//! VAST/Apple real-file Zonos code and PCM validation.
//!
//! This test is intentionally ignored locally.  It requires the fixed public
//! Zonos GGUF, the independently bound DAC 44.1-kHz GGUF, an authenticated v1
//! conditioning packet, and raw outputs from `zonos_dump_reference.py`.  The
//! code sequence is an exact discrete gate; PCM metrics remain
//! `MEASURED_NOT_GATED` until a reviewed bound is supplied by the worker.

use std::fs;
use std::path::Path;

use vokra_core::BackendKind;
use vokra_models::dac::Dac;
use vokra_models::zonos::{ZonosConditioningPacket, ZonosSamplingParams, ZonosTts};

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must name a VAST/Apple evidence file"))
}

fn digest(name: &str) -> [u8; 32] {
    let value = required(name);
    let bytes = value.as_bytes();
    assert_eq!(bytes.len(), 64, "{name} must be a 64-digit SHA-256 digest");
    let mut output = [0u8; 32];
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .unwrap_or_else(|_| panic!("{name} contains non-hex characters"));
    }
    output
}

fn f32_file(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert!(!bytes.is_empty() && bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn u32_file(path: &Path) -> Vec<u32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert!(!bytes.is_empty() && bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires fixed Zonos GGUF/DAC, authenticated packet, and VAST/Apple reference outputs"]
fn zonos_real_cpu_codes_and_pcm_boundary() {
    let model_path = required("VOKRA_ZONOS_GGUF");
    let dac_path = required("VOKRA_ZONOS_DAC_GGUF");
    let packet_path = required("VOKRA_ZONOS_CONDITIONING_PACKET");
    let expected_codes_path = required("VOKRA_ZONOS_REFERENCE_CODES");
    let expected_pcm_path = required("VOKRA_ZONOS_REFERENCE_PCM");
    let max_steps: usize = required("VOKRA_ZONOS_MAX_STEPS")
        .parse()
        .expect("VOKRA_ZONOS_MAX_STEPS must be a positive integer");
    assert!(max_steps > 0);
    let packet_bytes = fs::read(&packet_path).expect("read authenticated conditioning packet");
    let packet =
        ZonosConditioningPacket::parse(&packet_bytes, digest("VOKRA_ZONOS_PACKET_SHA256"), 2048)
            .expect("authenticated conditioning packet");
    let model = vokra_mmap::open_gguf(Path::new(&model_path)).expect("open Zonos GGUF");
    let dac_file = vokra_mmap::open_gguf(Path::new(&dac_path)).expect("open DAC GGUF");
    let backend = match std::env::var("VOKRA_ZONOS_BACKEND").as_deref() {
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok("metal") => BackendKind::Metal,
        Ok(other) => panic!("unsupported VOKRA_ZONOS_BACKEND={other}"),
    };
    let dac = Dac::from_gguf(&dac_file)
        .expect("strict DAC bind")
        .with_backend(backend);
    let tts = ZonosTts::from_gguf(&model)
        .expect("strict 246-tensor Zonos bind")
        .with_dac(dac)
        .expect("44.1-kHz nine-codebook DAC bind")
        .with_backend(backend);
    let codes = tts
        .generate_codes_with_sampling(&packet, max_steps, 2.0, &ZonosSamplingParams::greedy(), &[])
        .expect("native CPU Zonos code generation");
    let actual_codes: Vec<u32> = codes.iter().flatten().copied().collect();
    let expected_codes = u32_file(Path::new(&expected_codes_path));
    assert_eq!(
        actual_codes, expected_codes,
        "Zonos CPU codes differ from official reference"
    );
    let actual_pcm = tts.decode_codes(&codes).expect("native CPU DAC PCM decode");
    let expected_pcm = f32_file(Path::new(&expected_pcm_path));
    assert_eq!(actual_pcm.len(), expected_pcm.len(), "PCM lengths differ");
    assert!(actual_pcm.iter().all(|value| value.is_finite()));
    let max_abs = actual_pcm
        .iter()
        .zip(&expected_pcm)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = actual_pcm
        .iter()
        .zip(&expected_pcm)
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual_pcm.len() as f32;
    if let Ok(bound) = std::env::var("VOKRA_ZONOS_REGISTERED_PCM_MAX_ABS") {
        let bound: f32 = bound
            .parse()
            .expect("registered PCM bound must be finite f32");
        assert!(bound.is_finite() && bound >= 0.0);
        assert!(
            max_abs <= bound,
            "Zonos PCM max_abs={max_abs:e} > bound={bound:e}"
        );
    }
    eprintln!(
        "ZONOS_{}_REFERENCE codes=EXACT pcm_max_abs={max_abs:e} pcm_mean_abs={mean_abs:e} verdict=MEASURED_NOT_GATED",
        match backend {
            BackendKind::Cpu => "CPU",
            BackendKind::Metal => "METAL",
            _ => "OTHER",
        }
    );
}
