//! piper-plus v7 **non-zero prosody** parity (gated, `VOKRA_PIPER_V7_GGUF`).
//!
//! The sibling [`super::parity_v7`] suite feeds `prosody = zeros`, so the JA
//! prosody projection only ever contributes its **bias**
//! (`ProsodyProj::channels` returns `weight @ 0 + bias`). That leaves
//! `prosody_proj.weight` — a `[3 → 16]` matrix that turns the per-phoneme
//! `(A1, A2, A3)` accent triples into duration-predictor conditioning —
//! completely unexercised by parity.
//!
//! This suite closes that gap. It feeds a fixed **non-zero** integer prosody
//! buffer (`tests/parity/piper_plus_v7_prosody/prosody.i64`, the same bytes the
//! onnxruntime reference used) with `lid = 0` (Japanese — the only language the
//! v7 graph gates prosody on, `Equal(lid, 0)`). Non-zero prosody changes the
//! predicted durations (the reference length regulates to **47** frames here vs
//! 27 for zero prosody), so matching the reference `pcm` proves the whole native
//! prosody path: `prosody_proj.weight @ features` → `x_dp` → SDP → durations →
//! length regulation → flow → decoder → PCM.
//!
//! Reference: `tests/parity/piper_plus_v7_prosody/` (onnxruntime, FP16 weights,
//! `scales = [0, 1, 0]`; see its `gen_reference.py` / `manifest.txt`). Runs only
//! when `VOKRA_PIPER_V7_GGUF` points at the converted v7 voice GGUF, and skips
//! cleanly otherwise so CI stays green.

use std::path::PathBuf;

use super::PiperPlusTts;
use super::config::HIDDEN;

/// Per-sample PCM tolerance for the FP16 onnxruntime reference.
const PCM_ATOL: f32 = 0.05;

/// Language id of the fixed reference input (`ja` — prosody gate ON).
const LID: i64 = 0;

/// `scales = [0, 1, 0]` → `length_scale = 1`, noise off (deterministic).
const LENGTH_SCALE: f32 = 1.0;

/// `sum(ceil(durations))` the length regulator must produce for the **non-zero**
/// prosody input — 47, up from the zero-prosody 27 (`manifest.txt`). That the
/// native duration path reaches 47 is itself proof prosody flows into the SDP.
const T_FRAMES: usize = 47;

/// The fixed v7 reference phoneme ids: `[1, 2, …, 13, 0]` (T = 14), identical to
/// the sibling suite (`manifest.txt`).
fn phoneme_ids() -> Vec<i64> {
    (1..=13).chain(std::iter::once(0)).collect()
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("piper_plus_v7_prosody")
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

/// Reads the committed non-zero prosody buffer as a flattened `[T · 3]` `i64`
/// slice — the exact bytes fed to the onnxruntime reference, so the native run
/// is byte-identical on its inputs.
fn read_prosody() -> Vec<i64> {
    let path = fixtures_dir().join("prosody.i64");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 8, 0, "prosody.i64: not a whole number of i64");
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Loads the voice named by `$VOKRA_PIPER_V7_GGUF`, or `None` to skip (CI).
fn load_voice() -> Option<PiperPlusTts> {
    let path = std::env::var("VOKRA_PIPER_V7_GGUF").ok()?;
    Some(PiperPlusTts::from_path(&path).expect("load piper v7 voice GGUF"))
}

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

fn rms_error_and_correlation(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut se, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        se += (x - y) * (x - y);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let rms = (se / a.len() as f64).sqrt() as f32;
    let corr = if na > 0.0 && nb > 0.0 {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    } else {
        0.0
    };
    (rms, corr)
}

#[test]
fn v7_prosody_reference_regulates_to_47_frames() {
    // Fixture sanity (non-gated — reads only committed bytes): the non-zero
    // prosody reference length-regulates to T_FRAMES (47), up from the
    // zero-prosody sibling's 27. This is the premise the e2e test relies on —
    // a native model that ignored `prosody_proj.weight` or the JA gate would
    // land on 27 and fail the length assertion in `v7_prosody_e2e_pcm_parity`.
    let dec_input = read_f32("dec_input.f32");
    assert_eq!(
        dec_input.len(),
        HIDDEN * T_FRAMES,
        "reference dec_input must be [HIDDEN, 47] for the non-zero prosody input"
    );
    // pcm frames = T_FRAMES * hop (256 for the MB-iSTFT head) → 12032 samples.
    let pcm = read_f32("pcm.f32");
    assert_eq!(pcm.len(), T_FRAMES * 256, "reference pcm is T_FRAMES · hop");
}

#[test]
fn v7_prosody_decoder_pcm_parity() {
    // Decoder isolated on the non-zero-prosody reference latent. Confirms the
    // longer (47-frame) `dec_input` decodes to the reference `pcm`.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 prosody decoder parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let dec_input = read_f32("dec_input.f32");
    assert_eq!(
        dec_input.len(),
        HIDDEN * T_FRAMES,
        "dec_input [HIDDEN, T_FRAMES]"
    );
    let pcm = voice.decode(&dec_input, T_FRAMES, LID).expect("decode");
    let ref_pcm = read_f32("pcm.f32");
    assert_eq!(pcm.len(), ref_pcm.len(), "pcm length");
    assert!(pcm.iter().all(|s| s.is_finite()), "PCM has NaN/Inf");
    let d = max_abs_diff(&pcm, &ref_pcm);
    let (rms_err, corr) = rms_error_and_correlation(&pcm, &ref_pcm);
    eprintln!(
        "v7 prosody decoder pcm parity: max|Δ|={d:.6} rms_err={rms_err:.6} corr={corr:.6} len={}",
        pcm.len()
    );
    assert!(
        d <= PCM_ATOL,
        "decoder PCM parity {d} exceeds atol {PCM_ATOL}"
    );
    assert!(corr >= 0.999, "PCM correlation {corr} below 0.999");
    assert!(rms_err <= 0.01, "PCM rms error {rms_err} exceeds 0.01");
}

#[test]
fn v7_prosody_e2e_pcm_parity() {
    // The headline proof: the full native path with **non-zero** prosody fed
    // through the public `synthesize_phonemes` API. `prosody_proj.weight @
    // features` (gated on `lid == 0`) alters `x_dp`, so the native SDP must
    // predict the reference durations (regulating to 47 frames) and the whole
    // chain must reproduce the reference `pcm`. A native model that dropped the
    // weight term, mis-shaped the `[3 → 16]` matrix, or skipped the JA gate would
    // either mismatch the length (47 vs 27) or drift past PCM_ATOL. This is what
    // the zero-prosody sibling suite could not check.
    let Some(voice) = load_voice() else {
        eprintln!("skipping piper v7 prosody e2e parity: set VOKRA_PIPER_V7_GGUF to run");
        return;
    };
    let ids = phoneme_ids();
    let prosody = read_prosody();
    assert_eq!(prosody.len(), ids.len() * 3, "prosody is [T · 3] flattened");
    // None speaker embedding → zeros (matches the reference `spk_emb=zeros192`);
    // Some(prosody) drives the JA prosody projection with real, non-zero triples.
    let audio = voice
        .synthesize_phonemes(&ids, LID, None, Some(&prosody), 0.0, LENGTH_SCALE, 0.0)
        .expect("synthesize");
    let ref_pcm = read_f32("pcm.f32");
    assert_eq!(
        audio.samples.len(),
        ref_pcm.len(),
        "pcm length (native durations must consume prosody → 47 frames)"
    );
    assert!(
        audio.samples.iter().all(|s| s.is_finite()),
        "PCM has NaN/Inf"
    );
    let d = max_abs_diff(&audio.samples, &ref_pcm);
    let (rms_err, corr) = rms_error_and_correlation(&audio.samples, &ref_pcm);
    eprintln!(
        "v7 prosody e2e pcm parity: max|Δ|={d:.6} rms_err={rms_err:.6} corr={corr:.6} len={}",
        audio.samples.len()
    );
    assert!(d <= PCM_ATOL, "e2e PCM parity {d} exceeds atol {PCM_ATOL}");
    assert!(corr >= 0.999, "PCM correlation {corr} below 0.999");
    assert!(rms_err <= 0.01, "PCM rms error {rms_err} exceeds 0.01");
}
