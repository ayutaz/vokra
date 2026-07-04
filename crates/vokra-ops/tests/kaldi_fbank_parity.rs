//! Bit-exact parity for the CAM++ Kaldi fbank front-end (M0-08, NFR-QL-01).
//!
//! `vokra_ops::kaldi_fbank(KaldiFbankOpts::camplus())` is validated against an
//! independent Kaldi fbank oracle: `torchaudio.compliance.kaldi.fbank` (a BSD
//! PyTorch port of Kaldi `compute-fbank-feats`) with the exact CAM++ parameters,
//! followed by the per-utterance CMN vokra applies. The reference is
//! cross-checked by a from-scratch numpy Kaldi reimplementation (they agree to
//! ~9.3e-5; see `tests/parity/kaldi_fbank/manifest.txt`).
//!
//! Both fixtures — `input.f32` (the raw PCM) and `fbank_ref.f32` (the reference
//! features) — are committed, so this test runs in plain `cargo test` with **no
//! Python** and is **not** gated (unlike the CAM++ *network* parity, which needs
//! the 27 MB GGUF). Regenerate the fixtures with:
//!
//! ```text
//! python tests/parity/kaldi_fbank/gen_reference.py
//! ```
//!
//! The vokra (f32) vs oracle peak error was **measured at 9.3e-5** on log-mel
//! energies of O(10) — the same order as the torchaudio-f32-vs-numpy-f64
//! cross-check (9.3e-5), i.e. vokra agrees with the Kaldi reference to within
//! float32 rounding. The bound below is set at ~2x that. Frame count and shape
//! must match the reference exactly (no tolerance on shape).

use vokra_ops::{KaldiFbankOpts, kaldi_fbank};

/// Peak absolute error allowed between vokra and the Kaldi oracle on the CMN'd
/// log-mel features (measured 9.3e-5; bound ~2x, and 50x under the design-wide
/// FP32 `atol = 0.01`). Dominated by vokra's f32 FFT + f32 mel accumulation vs
/// the oracle's higher-precision path.
const ATOL: f32 = 2e-4;

/// CAM++ fbank geometry: 1 s @ 16 kHz -> `1 + (16000-400)/160 = 98` frames × 80.
const N_FRAMES: usize = 98;
const N_MELS: usize = 80;

fn fixtures_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/vokra-ops.
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("kaldi_fbank")
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

#[test]
fn kaldi_fbank_matches_torchaudio_kaldi_oracle() {
    let pcm = read_f32("input.f32");
    assert_eq!(pcm.len(), 16_000, "input fixture is 1 s @ 16 kHz");

    let (feats, t) = kaldi_fbank(&pcm, &KaldiFbankOpts::camplus()).expect("kaldi_fbank");

    // Frame geometry must be exact — no tolerance on shape.
    assert_eq!(t, N_FRAMES, "snip-edges frame count");
    assert_eq!(feats.len(), N_FRAMES * N_MELS, "feature matrix shape");

    let want = read_f32("fbank_ref.f32");
    assert_eq!(
        feats.len(),
        want.len(),
        "reference length {} != vokra {}",
        want.len(),
        feats.len()
    );

    let (mut max_abs, mut argmax) = (0.0f32, 0usize);
    for (i, (a, b)) in feats.iter().zip(&want).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }
    eprintln!(
        "kaldi_fbank parity: max|Δ|={max_abs:.3e} at [frame {}, bin {}] (atol={ATOL:.0e}, \
         {} values)",
        argmax / N_MELS,
        argmax % N_MELS,
        feats.len()
    );
    assert!(
        max_abs <= ATOL,
        "kaldi_fbank parity {max_abs:.3e} exceeds atol {ATOL:.0e}"
    );

    // Every value finite (no NaN/Inf leaked through the log floor or CMN).
    assert!(
        feats.iter().all(|v| v.is_finite()),
        "non-finite fbank value"
    );
}
