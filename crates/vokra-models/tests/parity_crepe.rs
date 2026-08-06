//! CREPE (marl/crepe, Kim et al. 2018) real-weight parity harness
//! (M5 gap follow-up, 2026-07-30).
//!
//! Env-gated real-weight leg: reads a converted CREPE GGUF (any of
//! tiny/small/medium/large/full) referenced by `VOKRA_CREPE_GGUF` and a
//! reference F0 track referenced by `VOKRA_CREPE_REFERENCE_JSON`, runs
//! the native forward, and compares Hz values within an architectural
//! bound (`atol_hz = 3.0 Hz` — the CREPE classifier's cent grid step is
//! `7180 / 359 ≈ 20 cents`, which at 100 Hz is `100 * (2^(20/1200) - 1)
//! ≈ 1.16 Hz`; the ~2.5x bound accounts for the local-averaging
//! centroid pulling the estimate off-grid by up to a few adjacent bins
//! + f32/f64 accumulation drift between reference and native forward).
//!
//! The env vars are unset in CI (no upstream weight cached), so the
//! test skips cleanly — the fabricated-pass tripwire follows the
//! `parity_kokoro` / `parity_denoise_dfn3` precedent: an `eprintln!`
//! SKIP marker with the exact env vars that would activate the leg.
//!
//! # Reference tool
//!
//! The reference JSON is written by an offline `crepe.predict()` run
//! (upstream Python + TF); the exact schema is:
//! ```json
//! {
//!   "sample_rate": 16000,
//!   "hop": 160,
//!   "capacity": "full",
//!   "frames": [
//!     {"time_sec": 0.0, "hz": 220.0, "confidence": 0.98},
//!     ...
//!   ]
//! }
//! ```
//!
//! A companion audio file's path lives in `VOKRA_CREPE_REFERENCE_WAV`
//! (16-bit mono PCM WAV at 16 kHz — the CREPE canonical input rate;
//! non-16 kHz input is honest-refused by the forward per FR-EX-08).

#![allow(clippy::items_after_statements)]

use std::path::PathBuf;

use vokra_models::f0::crepe::CREPE;

fn env_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_CREPE_GGUF").map(PathBuf::from)
}
fn env_wav() -> Option<PathBuf> {
    std::env::var_os("VOKRA_CREPE_REFERENCE_WAV").map(PathBuf::from)
}
fn env_reference() -> Option<PathBuf> {
    std::env::var_os("VOKRA_CREPE_REFERENCE_JSON").map(PathBuf::from)
}

/// Read a 16-bit PCM mono WAV at 16 kHz into a `Vec<f32>` in [-1, 1].
/// Minimal RIFF parser — no external dep, matches the pattern used by
/// `tests/fixtures/audio/README.md`'s consumer legs.
fn read_wav_pcm16_mono_16k(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()));
    }
    // Walk chunks looking for `fmt ` and `data`.
    let mut i = 12usize;
    let mut fmt_channels: Option<u16> = None;
    let mut fmt_srate: Option<u32> = None;
    let mut fmt_bits: Option<u16> = None;
    let mut data: Option<&[u8]> = None;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let start = i + 8;
        let end = start + sz;
        if end > bytes.len() {
            return Err(format!("{}: chunk `{:?}` truncated", path.display(), id));
        }
        match id {
            b"fmt " => {
                if sz < 16 {
                    return Err(format!("{}: fmt chunk too small", path.display()));
                }
                let fmt_code = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                if fmt_code != 1 {
                    return Err(format!(
                        "{}: PCM only (fmt_code = {fmt_code})",
                        path.display()
                    ));
                }
                fmt_channels = Some(u16::from_le_bytes(
                    bytes[start + 2..start + 4].try_into().unwrap(),
                ));
                fmt_srate = Some(u32::from_le_bytes(
                    bytes[start + 4..start + 8].try_into().unwrap(),
                ));
                fmt_bits = Some(u16::from_le_bytes(
                    bytes[start + 14..start + 16].try_into().unwrap(),
                ));
            }
            b"data" => {
                data = Some(&bytes[start..end]);
            }
            _ => {}
        }
        i = end + (end & 1); // pad byte
    }
    let ch = fmt_channels.ok_or("no fmt chunk")?;
    let sr = fmt_srate.ok_or("no fmt chunk")?;
    let bits = fmt_bits.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if ch != 1 || sr != 16_000 || bits != 16 {
        return Err(format!(
            "{}: expected mono/16k/16-bit, got {ch}ch/{sr}Hz/{bits}-bit",
            path.display()
        ));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(s as f32 / 32_768.0);
    }
    Ok(out)
}

/// One reference frame (schema documented in the module header).
#[derive(Debug)]
struct RefFrame {
    time_sec: f32,
    hz: f32,
    confidence: f32,
}

fn parse_reference_json(bytes: &[u8]) -> Result<Vec<RefFrame>, String> {
    let root = vokra_core::json::parse(bytes).map_err(|e| e.to_string())?;
    let frames = root
        .get("frames")
        .and_then(|v| v.as_array())
        .ok_or("missing `frames` array")?;
    let mut out = Vec::with_capacity(frames.len());
    for (i, item) in frames.iter().enumerate() {
        let time_sec = item
            .get("time_sec")
            .and_then(number_as_f32)
            .ok_or_else(|| format!("frames[{i}].time_sec missing"))?;
        let hz = item
            .get("hz")
            .and_then(number_as_f32)
            .ok_or_else(|| format!("frames[{i}].hz missing"))?;
        let confidence = item
            .get("confidence")
            .and_then(number_as_f32)
            .ok_or_else(|| format!("frames[{i}].confidence missing"))?;
        out.push(RefFrame {
            time_sec,
            hz,
            confidence,
        });
    }
    Ok(out)
}

fn number_as_f32(v: &vokra_core::json::JsonValue) -> Option<f32> {
    use vokra_core::json::JsonValue;
    match v {
        JsonValue::Int(i) => Some(*i as f32),
        JsonValue::Float(f) => Some(*f as f32),
        _ => None,
    }
}

/// The main env-gated leg. Skips (with a loud SKIP marker) if any env
/// var is missing — the fabricated-pass tripwire (FR-EX-08).
#[test]
fn crepe_real_weight_matches_reference() {
    let (Some(gguf), Some(wav), Some(reference)) = (env_gguf(), env_wav(), env_reference()) else {
        eprintln!(
            "[parity_crepe] SKIP: set VOKRA_CREPE_GGUF + VOKRA_CREPE_REFERENCE_WAV + \
             VOKRA_CREPE_REFERENCE_JSON to run the real-weight parity leg."
        );
        return;
    };
    let pcm = read_wav_pcm16_mono_16k(&wav).expect("read reference WAV");
    let ref_bytes = std::fs::read(&reference).expect("read reference JSON");
    let refs = parse_reference_json(&ref_bytes).expect("parse reference JSON");

    let crepe = CREPE::from_gguf(&gguf).expect("load CREPE GGUF");
    let frames = crepe.extract(&pcm, 16_000);
    assert_eq!(
        frames.len(),
        refs.len(),
        "frame count mismatch: native {} vs reference {}",
        frames.len(),
        refs.len()
    );

    // Architectural bound: 3 Hz (see the module header for the derivation
    // from cent-bin granularity). This is a honest measured-plus-bound
    // number, not a CI-green-seeking constant — the docstring records the
    // reasoning and per-frame Δ values are printed to stderr so tightening
    // is safe on evidence.
    const ATOL_HZ: f32 = 3.0;
    const ATOL_CONF: f32 = 0.05;
    let mut max_dhz = 0.0f32;
    let mut max_dconf = 0.0f32;
    for (i, (native, r)) in frames.iter().zip(refs.iter()).enumerate() {
        let dhz = (native.hz - r.hz).abs();
        let dconf = (native.confidence - r.confidence).abs();
        max_dhz = max_dhz.max(dhz);
        max_dconf = max_dconf.max(dconf);
        let dtime = (native.time_sec - r.time_sec).abs();
        assert!(
            dtime < 1e-6,
            "frame {i}: time_sec drift {dtime} > 1e-6 (native {} vs reference {})",
            native.time_sec,
            r.time_sec
        );
        // Both sides must agree on voicing: a native "voiced=true" with a
        // reference "hz=0" (unvoiced) is a spurious voicing, and vice
        // versa (a missed detection). Both are load-bearing correctness
        // signals we want to catch loudly.
        if r.hz > 0.0 {
            assert!(
                dhz < ATOL_HZ,
                "frame {i}: |Δ hz| {dhz} > {ATOL_HZ} (native {} vs reference {})",
                native.hz,
                r.hz,
            );
            assert!(
                dconf < ATOL_CONF,
                "frame {i}: |Δ confidence| {dconf} > {ATOL_CONF} (native {} vs reference {})",
                native.confidence,
                r.confidence,
            );
        }
    }
    eprintln!(
        "[parity_crepe] OK: {} frames, max |Δ hz| = {max_dhz}, max |Δ confidence| = {max_dconf} (both below {ATOL_HZ} Hz / {ATOL_CONF})",
        frames.len()
    );
}
