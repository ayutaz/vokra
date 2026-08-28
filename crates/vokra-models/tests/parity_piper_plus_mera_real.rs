//! Public Mera piper-plus GGUF against its independent official ONNX oracle.
//!
//! The small committed fixtures in `tests/parity/piper_plus_mera/` were dumped
//! by onnxruntime 1.23.2 from the fixed upstream revision recorded in the
//! manifest. `VOKRA_PIPER_MERA_GGUF` points at the separately published Vokra
//! GGUF; CI skips cleanly when the artifact is absent.

use std::collections::HashMap;
use std::path::PathBuf;

use vokra_core::BackendKind;
use vokra_models::piper_plus::PiperPlusTts;

const ATOL: f32 = 0.01;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("piper_plus_mera")
}

fn manifest() -> HashMap<String, String> {
    std::fs::read_to_string(fixture_dir().join("manifest.txt"))
        .expect("read Mera parity manifest")
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#'))
                .then(|| line.split_once('='))
                .flatten()
                .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        })
        .collect()
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: incomplete f32 fixture");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    assert!(actual.iter().all(|value| value.is_finite()));
    let (worst_index, worst) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (a, b))| (index, (a - b).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    eprintln!(
        "Mera official ONNX parity {label}: max|delta|={worst:.6e} at {worst_index} (atol={ATOL})"
    );
    assert!(
        worst <= ATOL,
        "{label}: max|delta|={worst} at {worst_index} exceeds {ATOL}"
    );
}

#[test]
fn mera_public_gguf_matches_official_onnx() {
    let Ok(path) = std::env::var("VOKRA_PIPER_MERA_GGUF") else {
        eprintln!("skipping Mera official ONNX parity: VOKRA_PIPER_MERA_GGUF unset");
        return;
    };
    let voice = PiperPlusTts::from_path(path).expect("load public Mera GGUF");
    let meta = manifest();
    let ids = meta["phoneme_ids"]
        .split_whitespace()
        .map(|token| token.parse::<i64>().expect("phoneme id"))
        .collect::<Vec<_>>();
    let lid = meta["lid"].parse::<i64>().expect("lid");
    let length_scale = meta["length_scale"].parse::<f32>().expect("length scale");

    let actual = voice
        .synthesize_with_intermediates(&ids, lid, BackendKind::Cpu, None, None, length_scale)
        .expect("Mera deterministic CPU synthesis");
    assert_eq!(actual.t_phonemes, meta["t_phonemes"].parse().unwrap());
    assert_eq!(actual.t_frames, meta["t_frames"].parse().unwrap());
    assert_eq!(actual.pcm.sample_rate, meta["sample_rate"].parse().unwrap());
    assert_close(&actual.m_p, &read_f32("m_p.f32"), "encoder m_p");
    assert_close(&actual.logs_p, &read_f32("logs_p.f32"), "encoder logs_p");
    assert_close(&actual.z, &read_f32("dec_input.f32"), "flow latent");
    assert_close(&actual.pcm.samples, &read_f32("pcm.f32"), "waveform PCM");
}
