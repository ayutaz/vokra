//! FFT / RFFT / STFT micro-benchmarks (M0-04-T19).
//!
//! `harness = false` (see Cargo.toml): a plain `fn main()` using `std::time`,
//! rather than the unstable libtest `#[bench]` harness or an external bench
//! crate (criterion is a third-party dependency; the workspace keeps zero of
//! those). There is no perf gate here — M0-04 sets no latency target, and the
//! 5% regression check (NFR-PF-13) is wired by M0-01's CI. The one comparison
//! that matters for FR-OP-01 is real-input (RFFT) vs full complex FFT, which
//! should show the ~2× advantage motivating the `real_input` attribute.

use std::hint::black_box;
use std::time::Instant;

use vokra_core::ir::graph::StftAttrs;
use vokra_ops::fft::{Complex32, FftPlan, RealFftPlan};
use vokra_ops::stft;

/// Times `f` over `iters` iterations (after a short warm-up) and reports the
/// average per-iteration duration in microseconds.
fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    for _ in 0..iters.min(16) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_secs_f64() * 1e6 / f64::from(iters);
    println!("{name:<28} {per:>10.3} us/iter  ({iters} iters)");
}

fn main() {
    println!("vokra-ops M0-04 FFT/RFFT/STFT benchmarks\n");

    // Complex-to-complex FFT vs real-input FFT at n = 1024.
    let n = 1024;
    let cplan = FftPlan::new(n);
    let cinput: Vec<Complex32> = (0..n)
        .map(|k| Complex32::new((k as f32 * 0.01).sin(), 0.0))
        .collect();
    bench("c2c fft n=1024", 20_000, || {
        black_box(cplan.forward_raw(black_box(&cinput)));
    });

    let rplan = RealFftPlan::new(n);
    let rinput: Vec<f32> = (0..n).map(|k| (k as f32 * 0.01).sin()).collect();
    bench("r2c rfft n=1024", 20_000, || {
        black_box(rplan.forward(black_box(&rinput)));
    });

    // A composite length (5-smooth) and a Bluestein length for context.
    let comp = FftPlan::new(1000);
    let cinput2: Vec<Complex32> = (0..1000)
        .map(|k| Complex32::new(k as f32 * 0.001, 0.0))
        .collect();
    bench("c2c fft n=1000", 20_000, || {
        black_box(comp.forward_raw(black_box(&cinput2)));
    });

    let prime = FftPlan::new(1021); // prime -> Bluestein
    let cinput3: Vec<Complex32> = (0..1021)
        .map(|k| Complex32::new(k as f32 * 0.001, 0.0))
        .collect();
    bench("c2c fft n=1021 (bluestein)", 10_000, || {
        black_box(prime.forward_raw(black_box(&cinput3)));
    });

    // Whisper-shaped STFT: 1 s @ 16 kHz, n_fft=400, hop=160.
    let signal: Vec<f32> = (0..16_000).map(|t| (t as f32 * 0.02).sin()).collect();
    let attrs = StftAttrs::new(400, 160);
    bench("stft 16k n_fft=400 hop=160", 500, || {
        black_box(stft(black_box(&signal), black_box(&attrs)).unwrap());
    });

    println!("\nnote: c2c vs r2c at the same n shows the RFFT ~2x advantage");
    println!("(FR-OP-01 real_input). Record these numbers in the PR description.");
}
