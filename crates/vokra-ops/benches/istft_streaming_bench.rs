//! Streaming iSTFT micro-benchmarks (M2-05-T11).
//!
//! `harness = false` (see Cargo.toml): a plain `fn main()` using `std::time`,
//! no external bench crate (the workspace keeps zero third-party deps). No perf
//! gate lives here — the 5% regression check (NFR-PF-13) is wired by M0-01's CI
//! over the recorded numbers. What this compares is the batch `istft` against
//! the chunked streaming reconstruction at several chunk granularities: the
//! streamed total does the same overlap-add work, so per-chunk overhead (the
//! tail carry-over + flush) is what these numbers isolate. Record them in the PR
//! description.
//!
//! The steady-state zero-allocation property (FR-EX-05) is proved separately by
//! the `buffer_capacity_is_stable_across_chunks` unit test (the tail buffers are
//! reserved once in `new` and never reallocate); the per-frame inverse FFT
//! allocates exactly as the batch op does, so the two are a fair comparison.

use std::hint::black_box;
use std::time::Instant;

use vokra_ops::Spectrogram;
use vokra_ops::attrs::{IstftAttrs, IstftStreamingAttrs, StftAttrs};
use vokra_ops::{IstftStreamingState, istft};

/// Times `f` over `iters` iterations (after a short warm-up), reporting the
/// average per-iteration duration in microseconds.
fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    for _ in 0..iters.min(8) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let per = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    println!("{name:<34} {per:>10.3} us/iter  ({iters} iters)");
}

/// Streams `spec` in fixed `chunk` frame counts (push + finish).
fn run_stream(spec: &Spectrogram, attrs: &IstftStreamingAttrs, chunk: usize) -> Vec<f32> {
    let mut state = IstftStreamingState::new(attrs).unwrap();
    let mut out = Vec::new();
    let b = spec.bins;
    let mut f = 0;
    while f < spec.frames {
        let take = chunk.min(spec.frames - f);
        let sub = Spectrogram {
            frames: take,
            bins: b,
            re: spec.re[f * b..(f + take) * b].to_vec(),
            im: spec.im[f * b..(f + take) * b].to_vec(),
        };
        out.extend(state.push(&sub).unwrap());
        f += take;
    }
    out.extend(state.finish());
    out
}

fn main() {
    println!("vokra-ops M2-05 istft_streaming benchmarks\n");

    // A vocoder-shaped reconstruction: ~2 s of audio, n_fft=1024, hop=256,
    // real half-spectrum (iSTFTNet / Vocos head layout).
    let (n_fft, hop) = (1024usize, 256usize);
    let signal: Vec<f32> = (0..32_000).map(|t| (t as f32 * 0.02).sin()).collect();
    let sa = StftAttrs::new(n_fft, hop);
    let spec = vokra_ops::stft(&signal, &sa).unwrap();
    let ia = IstftAttrs::new(n_fft, hop);
    let attrs = IstftStreamingAttrs::from_istft(ia.clone());
    println!(
        "frames={} bins={} (n_fft={n_fft} hop={hop})\n",
        spec.frames, spec.bins
    );

    bench("batch istft", 300, || {
        black_box(istft(black_box(&spec), black_box(&ia)).unwrap());
    });
    for &chunk in &[1usize, 4, 16, 64] {
        bench(&format!("streaming chunk={chunk} frame(s)"), 300, || {
            black_box(run_stream(black_box(&spec), black_box(&attrs), chunk));
        });
    }

    println!("\nnote: streamed total == batch bit-for-bit (istft_streaming_parity).");
    println!("Record these numbers in the PR description (NFR-PF-13 gate on CI).");
}
