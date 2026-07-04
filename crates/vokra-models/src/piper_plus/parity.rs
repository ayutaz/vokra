//! Numerical parity vs the piper-plus onnxruntime reference (M0-07-T13/T22).
//!
//! Fixtures live in `tests/parity/piper_plus/` and are regenerated offline by
//! `gen_reference.py` (onnxruntime, deterministic — noise scales zeroed, see
//! `docs/piper-plus-integration.md` §5). The reference is FP16 onnxruntime
//! (which casts to FP32 for every op), compared against this FP32 native
//! implementation.
//!
//! The voice GGUF is far too large to commit (~77 MB FP32), so these tests are
//! **gated on the `VOKRA_PIPER_GGUF` environment variable** and skip cleanly
//! when it is unset (e.g. in CI) — exactly like the Whisper parity tests
//! (`VOKRA_WHISPER_GGUF`). Set it to the path of a converted tsukuyomi voice to
//! run them locally:
//!
//! ```text
//! cargo run -p vokra-convert -- --model piper-plus \
//!     --input tsukuyomi-6lang-fp16.onnx --config config.json --output voice.gguf
//! VOKRA_PIPER_GGUF=voice.gguf cargo test -p vokra-models piper
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::PiperPlusTts;
use vokra_core::{Session, SynthesisRequest};

/// FP32 parity bound (NFR-QL-01).
const ATOL: f32 = 0.01;

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/vokra-models.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("piper_plus")
}

/// Reads a little-endian f32 fixture file.
fn read_f32(name: &str) -> Vec<f32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parses the `key = value` manifest.
fn manifest() -> HashMap<String, String> {
    let text = std::fs::read_to_string(fixtures_dir().join("manifest.txt")).expect("manifest.txt");
    let mut m = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            m.insert(k.trim().to_owned(), v.trim().to_owned());
        }
    }
    m
}

fn phoneme_ids(m: &HashMap<String, String>) -> Vec<i64> {
    m["phoneme_ids"]
        .split_whitespace()
        .map(|t| t.parse().expect("phoneme id"))
        .collect()
}

/// Loads the voice named by `$VOKRA_PIPER_GGUF`, or `None` to skip (CI).
fn load_voice() -> Option<PiperPlusTts> {
    let path = std::env::var("VOKRA_PIPER_GGUF").ok()?;
    Some(PiperPlusTts::from_path(&path).expect("load piper voice GGUF"))
}

/// Largest absolute difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn encoder_m_p_logs_p_parity() {
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper encoder parity: set VOKRA_PIPER_GGUF to run");
        return;
    };
    let m = manifest();
    let ids = phoneme_ids(&m);
    let lid: i64 = m["lid"].parse().unwrap();

    let out = voice.encode(&ids, lid).expect("encode");
    let ref_m_p = read_f32("m_p.f32");
    let ref_logs_p = read_f32("logs_p.f32");

    let dm = max_abs_diff(&out.m_p, &ref_m_p);
    let dl = max_abs_diff(&out.logs_p, &ref_logs_p);
    eprintln!("encoder parity: max|Δm_p|={dm:.6}, max|Δlogs_p|={dl:.6} (atol={ATOL})");
    assert!(dm <= ATOL, "m_p parity {dm} exceeds atol {ATOL}");
    assert!(dl <= ATOL, "logs_p parity {dl} exceeds atol {ATOL}");
}

#[test]
fn duration_parity() {
    // Native encoder + stochastic duration predictor (deterministic, noise_w=0)
    // vs the reference durations. Also checks integer w_ceil matches exactly.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper duration parity: set VOKRA_PIPER_GGUF to run");
        return;
    };
    let m = manifest();
    let ids = phoneme_ids(&m);
    let lid: i64 = m["lid"].parse().unwrap();
    let length_scale: f32 = m["length_scale"].parse().unwrap();

    // Isolate the SDP body (proj output) from the spline flows.
    let (body, _) = voice.sdp_body(&ids, lid).expect("sdp body");
    let ref_body = read_f32("sdp_body.f32");
    let db = max_abs_diff(&body, &ref_body);
    eprintln!("sdp body parity: max|Δbody|={db:.6}");

    let dur = voice.durations(&ids, lid, length_scale).expect("durations");
    let ref_dur = read_f32("durations.f32");
    let d = max_abs_diff(&dur, &ref_dur);

    // Integer frame counts (ceil) must agree exactly (drives length regulation).
    let w_ceil: Vec<usize> = dur.iter().map(|x| x.ceil() as usize).collect();
    let ref_ceil: Vec<usize> = ref_dur.iter().map(|x| x.ceil() as usize).collect();
    eprintln!(
        "duration parity: max|Δdur|={d:.6} w_ceil_match={} (atol={ATOL})",
        w_ceil == ref_ceil
    );
    assert!(d <= ATOL, "duration parity {d} exceeds atol {ATOL}");
    assert_eq!(w_ceil, ref_ceil, "ceil(durations) must match exactly");
}

#[test]
fn flow_latent_parity() {
    // Reference m_p + durations → length-regulate → reverse flow → z, compared
    // against the reference decoder-input latent (post-flow z·y_mask). Covers
    // length regulation (T15) + the flow (T16/T17) together.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper flow parity: set VOKRA_PIPER_GGUF to run");
        return;
    };
    let m = manifest();
    let lid: i64 = m["lid"].parse().unwrap();
    let hidden: usize = m["hidden"].parse().unwrap();
    let t_phonemes: usize = m["t_phonemes"].parse().unwrap();
    let t_frames: usize = m["t_frames"].parse().unwrap();

    let ref_m_p = read_f32("m_p.f32");
    let durations = read_f32("durations.f32");
    let ref_dec_input = read_f32("dec_input.f32");
    // w_ceil = ceil(durations) (length_scale already applied in durations).
    let w_ceil: Vec<usize> = durations.iter().map(|d| d.ceil() as usize).collect();
    assert_eq!(
        w_ceil.iter().sum::<usize>(),
        t_frames,
        "sum(w_ceil)=t_frames"
    );

    let (z, frames) = voice.expand_and_flow(&ref_m_p, t_phonemes, &w_ceil, lid);
    assert_eq!(frames, t_frames);
    assert_eq!(z.len(), hidden * t_frames);
    let d = max_abs_diff(&z, &ref_dec_input);
    eprintln!("flow parity: max|Δz|={d:.6} (atol={ATOL})");
    assert!(d <= ATOL, "flow latent parity {d} exceeds atol {ATOL}");
}

#[test]
fn decoder_pcm_parity() {
    // Feed the reference decoder-input latent (post-flow z·y_mask) through the
    // native MB-iSTFT decoder and compare the PCM. Isolates decoder + istft op
    // + PQMF from the flow/duration stages.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper decoder parity: set VOKRA_PIPER_GGUF to run");
        return;
    };
    let m = manifest();
    let lid: i64 = m["lid"].parse().unwrap();
    let hidden: usize = m["hidden"].parse().unwrap();
    let t_frames: usize = m["t_frames"].parse().unwrap();

    let dec_input = read_f32("dec_input.f32");
    assert_eq!(dec_input.len(), hidden * t_frames, "dec_input shape");
    let ref_pcm = read_f32("pcm.f32");

    let pcm = voice.decode(&dec_input, t_frames, lid).expect("decode");
    assert_eq!(pcm.len(), ref_pcm.len(), "pcm length");
    let d = max_abs_diff(&pcm, &ref_pcm);
    eprintln!("decoder parity: max|Δpcm|={d:.6} (atol={ATOL})");
    assert!(d <= ATOL, "decoder PCM parity {d} exceeds atol {ATOL}");
}

#[test]
fn e2e_pcm_parity() {
    // Full native path: phoneme ids → PCM, deterministic (noise scales 0),
    // vs the onnxruntime reference PCM. The WP completion criterion (T22).
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper e2e parity: set VOKRA_PIPER_GGUF to run");
        return;
    };
    let m = manifest();
    let ids = phoneme_ids(&m);
    let lid: i64 = m["lid"].parse().unwrap();
    let length_scale: f32 = m["length_scale"].parse().unwrap();

    let audio = voice
        .synthesize_phonemes(&ids, lid, None, None, 0.0, length_scale, 0.0)
        .expect("synthesize");
    let ref_pcm = read_f32("pcm.f32");
    assert_eq!(audio.samples.len(), ref_pcm.len(), "pcm length");
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "PCM has NaN/Inf"
    );
    let d = max_abs_diff(&audio.samples, &ref_pcm);
    eprintln!(
        "e2e parity: max|Δpcm|={d:.6}, len={} (atol={ATOL})",
        audio.samples.len()
    );
    assert!(d <= ATOL, "e2e PCM parity {d} exceeds atol {ATOL}");
}

#[test]
fn session_tts_api_smoke() {
    // The demo's path (M0-07-T20/T23): inject the native voice as a Session TTS
    // engine and synthesize through `session.tts()`. Text → phoneme ids uses
    // the placeholder tokenizer; asserts a non-empty, finite, correct-rate PCM.
    let path = match std::env::var("VOKRA_PIPER_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("skipping piper session smoke: set VOKRA_PIPER_GGUF to run");
            return;
        }
    };
    let model = PiperPlusTts::from_path(&path).expect("load voice");
    let sample_rate = model.config().sample_rate;
    let session = Session::from_file(&path)
        .build()
        .expect("session")
        .with_tts_engine(Arc::new(model));

    // Vowel phonemes are literal symbols the placeholder tokenizer maps.
    let audio = session
        .tts()
        .synthesize_request(&SynthesisRequest::new("aiueo").deterministic())
        .expect("synthesize via session");
    assert!(!audio.samples.is_empty(), "empty PCM");
    assert_eq!(audio.sample_rate, sample_rate);
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "NaN/Inf in PCM"
    );
    eprintln!(
        "session TTS smoke: {} samples @ {} Hz",
        audio.samples.len(),
        audio.sample_rate
    );
}
