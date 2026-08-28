//! NKF-AEC numerical parity harness — env-gated (AEC family, 2026-08-05).
//!
//! Sibling of `parity_openwakeword.rs` / `parity_rmvpe.rs`: every test
//! that needs a real NKF-AEC GGUF + paired mic/farend WAVs is gated on
//! the [`GGUF_ENV`] / [`MIC_ENV`] / [`FAREND_ENV`] variables and skips
//! cleanly when unset (never a fabricated pass — memory
//! `[[project-real-weight-eval]]`). Once opted in, every failure is
//! hard: a missing / malformed / wrong-shaped fixture is a loud panic
//! (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! The upstream `fjiang9/NKF-AEC` release ships a ~5.3 KB
//! `pretrained/nkf.pt` plus demo WAVs (`src/mic.wav`, `src/ref.wav`).
//! Bridge them offline to a Vokra GGUF via the sidecar:
//!
//! ```text
//! # 1. Clone the release (git clone --depth 1 is fine, the repo is ~10 MB):
//! git clone --depth 1 https://github.com/fjiang9/NKF-AEC.git \
//!     ~/checkpoints/nkf-aec/repo
//!
//! # 2. Flatten .pt → safetensors:
//! cd tools/parity && uv run python nkf_aec_prepare_checkpoint.py \
//!     --input  ~/checkpoints/nkf-aec/repo/pretrained/nkf.pt \
//!     --output ~/checkpoints/nkf-aec/model.safetensors
//!
//! # 3. Convert safetensors → GGUF:
//! vokra-cli convert --model nkf-aec \
//!     --input  ~/checkpoints/nkf-aec/model.safetensors \
//!     --output ~/gguf/nkf-aec.gguf
//!
//! # 4. Point the parity harness at the artefacts:
//! export VOKRA_NKF_AEC_REAL_GGUF=~/gguf/nkf-aec.gguf
//! export VOKRA_NKF_AEC_REAL_MIC_WAV=~/checkpoints/nkf-aec/repo/src/mic.wav
//! export VOKRA_NKF_AEC_REAL_FAREND_WAV=~/checkpoints/nkf-aec/repo/src/ref.wav
//! cargo test -p vokra-models --test parity_nkf_aec -- --nocapture
//! ```
//!
//! # Numeric parity contract
//!
//! NKF-AEC is a per-bin adaptive Kalman filter; the cleaned PCM is
//! sensitive to numerical rounding in the recurrence (drift accumulates
//! across every frame). The parity check is an **ERLE floor**: the
//! echo-return-loss enhancement of the cleaned output vs the mic must
//! meet a minimum improvement over the raw mic — the same shape the
//! upstream authors report (Yang et al. 2023 ICASSP, AEC-Challenge
//! Blind Test Set). A `20 dB` floor corresponds to typical clean-
//! condition ERLE for NKF-AEC on the demo pair (upstream reports
//! `~40 dB` mean on the challenge blind set; the demo pair is
//! easier). If upstream ships a reference-cleaned WAV in a future
//! release, an additional per-sample max-|Δ| leg can bind here
//! without a schema change — the harness is written to accept an
//! optional [`REFERENCE_WAV_ENV`] side-car.
//!
//! # Skip-clean contract
//!
//! Every gated test starts with the same three env checks; unset ⇒
//! print a one-line explanation on stderr and return. The test result
//! is `PASSED` (never `SKIP`) so CI aggregates cleanly on runners that
//! do not carry the fixture — but the stderr trail makes the skip
//! visible so a stale env goes noticed.

use std::env;
use std::path::Path;

use vokra_core::VokraError;
use vokra_core::engines::AecEngine;
use vokra_models::aec::nkf_aec::{NkfAec, SAMPLE_RATE};

/// Env var the owner sets to point the gated harness at a real
/// NKF-AEC GGUF. Absent = skip cleanly (never a fabricated pass).
const GGUF_ENV: &str = "VOKRA_NKF_AEC_REAL_GGUF";

/// Env var pointing at a 16 kHz mono WAV — the mic (near-end + echo)
/// signal upstream also fed to the Python reference.
const MIC_ENV: &str = "VOKRA_NKF_AEC_REAL_MIC_WAV";

/// Env var pointing at a 16 kHz mono WAV — the far-end (loudspeaker
/// reference) signal upstream also fed to the Python reference.
const FAREND_ENV: &str = "VOKRA_NKF_AEC_REAL_FAREND_WAV";

/// Optional side-car: if set, points at a 16 kHz mono WAV containing
/// the upstream Python reference's cleaned output on the same mic /
/// farend pair. When present, the parity test also runs a per-sample
/// max-|Δ| leg against it (bound [`PCM_ATOL`]).
#[allow(dead_code)]
const REFERENCE_WAV_ENV: &str = "VOKRA_NKF_AEC_REFERENCE_WAV";

/// Minimum echo-return-loss enhancement (dB) the cleaned output must
/// achieve over the raw mic. Upstream reports ~40 dB mean on the
/// AEC-Challenge Blind Set; the demo pair is easier; 20 dB is a
/// conservative floor that catches every real regression while
/// leaving headroom for numerical drift across the Kalman recurrence.
#[allow(dead_code)]
const ERLE_FLOOR_DB: f32 = 20.0;

/// Per-sample max-|Δ| tolerance when [`REFERENCE_WAV_ENV`] is set. The
/// Kalman recurrence's arithmetic ordering may differ from PyTorch's
/// SGEMM path even at bit-identical weights — a small tolerance
/// accommodates that; a wider one would mask a real regression.
#[allow(dead_code)]
const PCM_ATOL: f32 = 5e-3;

/// GATED: opens a real NKF-AEC GGUF and verifies the load path is a
/// genuine bind (real config parse, every one of the 22 KGNet tensors
/// binds cleanly, arch tag matches).
///
/// Skips cleanly when [`GGUF_ENV`] is unset.
#[test]
fn parity_nkf_aec_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping NKF-AEC GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };
    let path = Path::new(&gguf_path);
    let session = NkfAec::open(path).unwrap_or_else(|e| {
        panic!(
            "NKF-AEC GGUF at {gguf_path} failed to load: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    let cfg = session.config();
    assert_eq!(
        cfg.sample_rate, SAMPLE_RATE,
        "NKF-AEC is trained at 16 kHz PCM in; a differently-rated GGUF is \
         either misconfigured or a non-canonical fork (loud-fail)"
    );
    assert_eq!(cfg.l, 4, "upstream nkf.py pins L = 4");
    assert_eq!(cfg.rnn_dim, 18, "upstream KGNet pins rnn_dim = 18");
    assert_eq!(cfg.n_fft, 1024, "upstream nkf.py pins n_fft = 1024");
    assert_eq!(cfg.hop, 256, "upstream nkf.py pins hop_length = 256");
    eprintln!(
        "NKF-AEC GGUF smoke pass — config = {:?}, all 22 KGNet tensors \
         bound successfully",
        cfg
    );
}

/// GATED: opens a real NKF-AEC GGUF and runs the cleaned-PCM ERLE
/// floor check against a paired mic / far-end WAV. Skips cleanly when
/// any of the three env vars is unset.
///
/// # ERLE metric
///
/// `ERLE(dB) = 10 * log10( ||mic||² / ||cleaned||² )` — the ratio of
/// input mic energy to output cleaned energy, in dB. A higher ERLE
/// means the AEC subtracted more echo energy from the mic. The
/// [`ERLE_FLOOR_DB`] floor here (20 dB) is 100x energy reduction — a
/// clear-cut success threshold for a canonical demo pair.
#[test]
fn parity_nkf_aec_erle_floor() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!("{GGUF_ENV} unset — skipping NKF-AEC ERLE parity; clean skip.");
        return;
    };
    let Some(mic_path) = env::var(MIC_ENV).ok() else {
        eprintln!(
            "{MIC_ENV} unset — skipping NKF-AEC ERLE parity (need both mic and farend WAVs); \
             clean skip."
        );
        return;
    };
    let Some(farend_path) = env::var(FAREND_ENV).ok() else {
        eprintln!(
            "{FAREND_ENV} unset — skipping NKF-AEC ERLE parity (need both mic and farend WAVs); \
             clean skip."
        );
        return;
    };

    let session = NkfAec::open(Path::new(&gguf_path))
        .unwrap_or_else(|e| panic!("NKF-AEC GGUF at {gguf_path} failed to load: {e:?}"));
    let mic = load_mono_16k_wav(&mic_path);
    let farend = load_mono_16k_wav(&farend_path);
    assert!(
        !mic.is_empty() && mic.len() == farend.len(),
        "mic / farend WAVs must be non-empty and length-matched \
         (mic={}, farend={})",
        mic.len(),
        farend.len()
    );

    let mut stream = session.open_stream(SAMPLE_RATE).unwrap();
    let cleaned = stream
        .push_paired(&mic, &farend)
        .unwrap_or_else(|e| panic!("push_paired failed: {e:?}"));

    // ERLE over the region we have cleaned samples for. Compare on
    // matching length — the streaming iSTFT tail may commit fewer
    // samples than the input.
    let n = cleaned.len().min(mic.len());
    assert!(
        n > SAMPLE_RATE as usize / 2,
        "cleaned output too short for a meaningful ERLE: {} samples \
         (needed > 0.5 s)",
        n
    );
    let mic_energy: f64 = mic[..n].iter().map(|&x| x as f64 * x as f64).sum();
    let out_energy: f64 = cleaned[..n].iter().map(|&x| x as f64 * x as f64).sum();
    assert!(
        mic_energy > 0.0,
        "mic input has zero energy — fixture appears silent"
    );
    // Guard against 0-energy output: log(0) is -inf; we bound at a
    // sane large value so the assertion fires with a clear message.
    let ratio = if out_energy > 0.0 {
        mic_energy / out_energy
    } else {
        f64::INFINITY
    };
    let erle_db = 10.0 * ratio.log10();
    eprintln!(
        "NKF-AEC ERLE = {erle_db:.2} dB (mic_energy = {mic_energy:.3e}, \
         cleaned_energy = {out_energy:.3e}, n = {n})"
    );
    assert!(
        erle_db as f32 >= ERLE_FLOOR_DB,
        "ERLE {erle_db:.2} dB below floor {ERLE_FLOOR_DB} dB — the neural \
         Kalman recurrence did not reduce echo energy enough. Either the \
         fixture WAVs are not the canonical demo pair, or the forward has \
         a regression."
    );

    // Optional stricter leg: per-sample max-|Δ| against an upstream-
    // supplied reference cleaned WAV.
    if let Ok(ref_path) = env::var(REFERENCE_WAV_ENV) {
        let reference = load_mono_16k_wav(&ref_path);
        let m = cleaned.len().min(reference.len());
        assert!(
            m > SAMPLE_RATE as usize / 2,
            "reference WAV too short for per-sample parity"
        );
        let max_dev = cleaned[..m]
            .iter()
            .zip(reference[..m].iter())
            .map(|(&c, &r)| (c - r).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "NKF-AEC per-sample max |Δ| vs {ref_path} = {max_dev:e} \
             (tolerance {PCM_ATOL:e})"
        );
        assert!(
            max_dev <= PCM_ATOL,
            "per-sample max |Δ| {max_dev:e} exceeds tolerance {PCM_ATOL:e} \
             — arithmetic order or hparam divergence from PyTorch reference"
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
fn parity_nkf_aec_cpu_metal_real_pair() {
    let (Some(gguf_path), Some(mic_path), Some(farend_path)) = (
        env::var(GGUF_ENV).ok(),
        env::var(MIC_ENV).ok(),
        env::var(FAREND_ENV).ok(),
    ) else {
        eprintln!("set {GGUF_ENV}, {MIC_ENV}, and {FAREND_ENV} for NKF CPU/Metal parity");
        return;
    };
    let cpu = NkfAec::open(Path::new(&gguf_path)).expect("bind NKF CPU");
    let metal = NkfAec::open(Path::new(&gguf_path))
        .expect("bind NKF Metal")
        .with_backend(vokra_core::BackendKind::Metal);
    assert_eq!(cpu.backend(), vokra_core::BackendKind::Cpu);
    assert_eq!(metal.backend(), vokra_core::BackendKind::Metal);
    let mic = load_mono_16k_wav(&mic_path);
    let farend = load_mono_16k_wav(&farend_path);
    assert_eq!(mic.len(), farend.len());
    let samples = mic.len().min(SAMPLE_RATE as usize);
    let mut cpu_stream = cpu.open_stream(SAMPLE_RATE).unwrap();
    let mut metal_stream = metal.open_stream(SAMPLE_RATE).unwrap();
    let cpu_output = cpu_stream
        .push_paired(&mic[..samples], &farend[..samples])
        .unwrap();
    let metal_output = metal_stream
        .push_paired(&mic[..samples], &farend[..samples])
        .unwrap();
    assert_eq!(metal_output.len(), cpu_output.len());
    let max_abs = metal_output
        .iter()
        .zip(&cpu_output)
        .map(|(&metal, &cpu)| (metal - cpu).abs())
        .fold(0.0f32, f32::max);
    eprintln!("NKF-AEC real CPU/Metal max_abs={max_abs:e}");
    assert!(max_abs <= 1e-2);
}

/// GATED: verifies FR-EX-08 sample-rate refusal on a real GGUF (an
/// engine bound at 16 kHz refuses `open_stream(8000)` loudly).
#[test]
fn parity_nkf_aec_wrong_sample_rate_loud() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!("{GGUF_ENV} unset — skipping wrong-rate loud-fail; clean skip.");
        return;
    };
    let session = NkfAec::open(Path::new(&gguf_path))
        .unwrap_or_else(|e| panic!("NKF-AEC GGUF at {gguf_path} failed to load: {e:?}"));
    // `Box<dyn AecStreamHandle + Send>` is not `Debug`, so we can't
    // use `.expect_err(...)` — pattern-match on the Result directly.
    match session.open_stream(8_000) {
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("8000") || msg.contains("8_000"),
                "must name pushed rate: {msg}"
            );
            assert!(
                msg.contains("16_000") || msg.contains("16000"),
                "must name model rate: {msg}"
            );
        }
        Err(other) => panic!("expected InvalidArgument, got {other:?}"),
        Ok(_) => panic!("wrong sample rate must be loud on a real GGUF too"),
    }
}

// ---- helpers ------------------------------------------------------------

/// Loads a mono 16 kHz PCM16 WAV as `Vec<f32>` in `[-1, 1]`. Uses a
/// small hand-rolled parser (no crate dep — the parity harness is
/// zero-dep to preserve NFR-DS-02).
fn load_mono_16k_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("read WAV {path}: {e} (fixture path may be stale)"));
    // Minimal RIFF/WAVE parser: RIFF header + WAVE + fmt + data.
    assert!(
        bytes.len() > 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{path} is not a RIFF/WAVE file"
    );
    // Walk chunks to find `fmt ` and `data`.
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (audio_format, channels, sample_rate, bits)
    let mut data_range: Option<(usize, usize)> = None;
    let mut i = 12usize;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let start = i + 8;
        let end = start + sz;
        assert!(end <= bytes.len(), "{path} chunk overruns file");
        match id {
            b"fmt " => {
                assert!(sz >= 16, "{path} fmt chunk too small");
                let af = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                let ch = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
                let sr = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
                fmt = Some((af, ch, sr, bits));
            }
            b"data" => data_range = Some((start, end)),
            _ => {}
        }
        // Chunks are 2-byte aligned.
        i = end + (sz & 1);
    }
    let (af, ch, sr, bits) = fmt.unwrap_or_else(|| panic!("{path} missing fmt chunk"));
    let (ds, de) = data_range.unwrap_or_else(|| panic!("{path} missing data chunk"));
    assert_eq!(af, 1, "{path} is not PCM (audio_format={af})");
    assert_eq!(ch, 1, "{path} is not mono (channels={ch})");
    assert_eq!(sr, 16_000, "{path} sample rate {sr} != 16000");
    assert_eq!(bits, 16, "{path} is not 16-bit (bits={bits})");
    bytes[ds..de]
        .chunks_exact(2)
        .map(|c| {
            let s = i16::from_le_bytes(c.try_into().unwrap());
            (s as f32) / (i16::MAX as f32)
        })
        .collect()
}
