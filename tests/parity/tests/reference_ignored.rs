//! Parity shells for the reference implementations that are not installed in
//! this environment (M0-04-T13 / T16, path (b)).
//!
//! torch (STFT), librosa (mel) and scipy (DCT) fixtures are NOT committed
//! because those libraries are absent here and fixture numbers must never be
//! invented. These tests are therefore `#[ignore]`: they run only with
//! `cargo test -- --ignored` AND after the fixtures are generated in an
//! environment that has the reference library (see
//! `tests/parity/gen_parity_fixtures.py {stft,mel,dct}`). When the fixtures are
//! absent the body skips cleanly, so `--ignored` never fails spuriously.
//!
//! The comparison logic is fully implemented so the suite is exercise-ready the
//! moment the fixtures land — this is a real harness, not a stub.

use vokra_core::ir::graph::{DctAttrs, MelAttrs, MelInterp, MelNorm, MelScale, StftAttrs};
use vokra_ops::dct::dct;
use vokra_ops::mel::MelFilterbank;
use vokra_ops::stft::stft;
use vokra_parity::{assert_close, load_dir};

#[test]
#[ignore = "requires torch fixtures: python3 tests/parity/gen_parity_fixtures.py stft"]
fn stft_parity_against_torch() {
    let fixtures = load_dir("stft");
    if fixtures.is_empty() {
        eprintln!("stft fixtures absent (torch not installed at generation time) — skipping");
        return;
    }
    for f in &fixtures {
        let n_fft = f.usize("n_fft");
        let hop = f.usize("hop");
        let signal = f.floats("signal");
        // StftAttrs::new defaults (Hann/periodic, center, reflect, backward,
        // real_input) match torch.stft(..., normalized=False, return_complex).
        let spec = stft(&signal, &StftAttrs::new(n_fft, hop)).expect("stft");
        assert_eq!(spec.frames, f.usize("frames"), "frame count");
        assert_eq!(spec.bins, f.usize("bins"), "bin count");
        let ctx = format!("{:?}", f.path);
        assert_close(&spec.re, &f.floats("out_re"), &format!("{ctx} re"));
        assert_close(&spec.im, &f.floats("out_im"), &format!("{ctx} im"));
    }
}

#[test]
#[ignore = "requires librosa fixtures: python3 tests/parity/gen_parity_fixtures.py mel"]
fn mel_parity_against_librosa() {
    let fixtures = load_dir("mel");
    if fixtures.is_empty() {
        eprintln!("mel fixtures absent (librosa not installed at generation time) — skipping");
        return;
    }
    for f in &fixtures {
        let attrs = MelAttrs {
            sample_rate: f.usize("sample_rate") as u32,
            n_fft: f.usize("n_fft"),
            n_mels: f.usize("n_mels"),
            fmin: f.f64("fmin") as f32,
            fmax: Some(f.f64("fmax") as f32),
            scale: if f.usize("htk") == 1 {
                MelScale::Htk
            } else {
                MelScale::Slaney
            },
            norm: MelNorm::Slaney,
            // librosa reference uses Hz-domain triangular ramps.
            interp: MelInterp::Hz,
        };
        let fb = MelFilterbank::new(&attrs);
        assert_eq!(fb.n_freqs, f.usize("n_freqs"), "n_freqs");
        assert_close(&fb.weights, &f.floats("weights"), &format!("{:?}", f.path));
    }
}

#[test]
#[ignore = "requires scipy fixtures: python3 tests/parity/gen_parity_fixtures.py dct"]
fn dct_parity_against_scipy() {
    let fixtures = load_dir("dct");
    if fixtures.is_empty() {
        eprintln!("dct fixtures absent (scipy not installed at generation time) — skipping");
        return;
    }
    for f in &fixtures {
        let n = f.usize("n");
        let x = f.floats("in");
        // scipy.fft.dct(type=2, norm="ortho") == DctAttrs::new (ortho DCT-II).
        let got = dct(&x, 1, n, &DctAttrs::new());
        assert_close(&got, &f.floats("out"), &format!("{:?}", f.path));
    }
}
