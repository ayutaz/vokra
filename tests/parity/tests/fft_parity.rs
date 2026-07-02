//! FFT / RFFT numerical parity against numpy (M0-04-T08).
//!
//! numpy.fft is a mature, independent DFT reference (numpy is always available;
//! see `parity-requirements.txt`). Covers c2c forward/inverse over power-of-two
//! / composite / small lengths × all three normalizations, plus r2c. Tolerance
//! is NFR-QL-01 FP32 `atol = 0.01` (numpy computes in f64; vokra in f32).

use vokra_core::ir::graph::Normalization;
use vokra_ops::fft::{Complex32, FftPlan, RealFftPlan, norm_scale};
use vokra_parity::{assert_close, load_dir};

fn parse_norm(s: &str) -> Normalization {
    match s {
        "backward" => Normalization::Backward,
        "ortho" => Normalization::Ortho,
        "forward" => Normalization::Forward,
        other => panic!("unknown normalization {other:?}"),
    }
}

#[test]
fn fft_parity_against_numpy() {
    let fixtures = load_dir("fft");
    assert!(
        !fixtures.is_empty(),
        "no fft fixtures; run: python3 tests/parity/gen_parity_fixtures.py fft"
    );
    for f in &fixtures {
        let n = f.usize("n");
        let norm = parse_norm(f.get("norm"));
        let in_re = f.floats("in_re");
        let in_im = f.floats("in_im");
        let x: Vec<Complex32> = in_re
            .iter()
            .zip(&in_im)
            .map(|(r, i)| Complex32::new(*r, *i))
            .collect();

        let plan = FftPlan::new(n);
        let (raw, forward) = match f.get("kind") {
            "fft_forward" => (plan.forward_raw(&x), true),
            "fft_inverse" => (plan.inverse_raw(&x), false),
            other => panic!("unexpected fft kind {other:?}"),
        };
        let scale = norm_scale(norm, n, forward);
        let got_re: Vec<f32> = raw.iter().map(|c| c.re * scale).collect();
        let got_im: Vec<f32> = raw.iter().map(|c| c.im * scale).collect();

        let ctx = format!("{:?}", f.path);
        assert_close(&got_re, &f.floats("out_re"), &format!("{ctx} re"));
        assert_close(&got_im, &f.floats("out_im"), &format!("{ctx} im"));
    }
}

#[test]
fn rfft_parity_against_numpy() {
    let fixtures = load_dir("rfft");
    assert!(
        !fixtures.is_empty(),
        "no rfft fixtures; run: python3 tests/parity/gen_parity_fixtures.py fft"
    );
    for f in &fixtures {
        let n = f.usize("n");
        let real = f.floats("in_re");
        let spec = RealFftPlan::new(n).forward(&real);
        let got_re: Vec<f32> = spec.iter().map(|c| c.re).collect();
        let got_im: Vec<f32> = spec.iter().map(|c| c.im).collect();
        let ctx = format!("{:?}", f.path);
        assert_close(&got_re, &f.floats("out_re"), &format!("{ctx} re"));
        assert_close(&got_im, &f.floats("out_im"), &format!("{ctx} im"));
    }
}
