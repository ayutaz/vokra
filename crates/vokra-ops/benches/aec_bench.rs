//! AEC per-frame micro-benchmark (M4-03-T11).
//!
//! `harness = false` (see Cargo.toml): a plain `fn main()` on `std::time`,
//! no external bench crate (zero-dep — the fft_bench / istft_streaming_bench
//! pattern). **No absolute performance gate lives here** (measurements are
//! machine-dependent; the relative-regression thinking of the M3-01 gates
//! applies): the numbers are recorded in the PR description.
//!
//! What it reports per configuration: mean µs per `Aec::process` frame
//! (steady state, queue path included) and the fraction of the real-time
//! budget one frame is allowed (`frame_size / sample_rate` seconds) — the
//! "RTF hook" of the ticket. A value ≪ 1.0 means comfortable real-time
//! headroom on this machine.

use std::hint::black_box;
use std::time::Instant;

use vokra_core::stream::aec_ref_queue;
use vokra_ops::{Aec, AecAttrs};

fn bench_config(name: &str, attrs: AecAttrs, frames: u32) {
    let n = attrs.frame_size;
    let mut aec = Aec::new(&attrs).expect("aec builds");
    let (mut tx, mut rx) = aec_ref_queue(8 * attrs.filter_length, attrs.sample_rate).unwrap();

    let far: Vec<f32> = (0..n)
        .map(|i| ((i * 37 + 5) % 199) as f32 / 400.0 - 0.25)
        .collect();
    let mic: Vec<f32> = (0..n)
        .map(|i| ((i * 53 + 11) % 211) as f32 / 500.0 - 0.2)
        .collect();
    let mut out = vec![0.0f32; n];

    // Warm-up (adaptation transient + caches).
    for f in 0..32u64 {
        let pos = f * n as u64;
        tx.push(&far, pos).unwrap();
        aec.process(&mic, pos, &mut rx, &mut out).unwrap();
    }

    let start = Instant::now();
    for f in 32..(32 + u64::from(frames)) {
        let pos = f * n as u64;
        tx.push(&far, pos).unwrap();
        aec.process(&mic, pos, &mut rx, &mut out).unwrap();
        black_box(&out);
    }
    let per_frame_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(frames);
    let budget_us = 1e6 * attrs.frame_size as f64 / f64::from(attrs.sample_rate);
    println!(
        "{name:<34} {per_frame_us:>9.2} us/frame  budget {budget_us:>8.1} us  \
         load {:>6.3}x  ({frames} frames)",
        per_frame_us / budget_us
    );
}

fn main() {
    println!("aec_bench — SpeexDSP MDF port, queue path included (lower is better)");
    bench_config(
        "8 kHz  frame 128  tail 1024 (M=8)",
        AecAttrs {
            sample_rate: 8_000,
            frame_size: 128,
            filter_length: 1024,
        },
        2000,
    );
    bench_config(
        "16 kHz frame 256  tail 2048 (M=8)",
        AecAttrs {
            sample_rate: 16_000,
            frame_size: 256,
            filter_length: 2048,
        },
        2000,
    );
    bench_config(
        "24 kHz frame 480  tail 4800 (M=10)",
        AecAttrs {
            sample_rate: 24_000,
            frame_size: 480,
            filter_length: 4800,
        },
        1000,
    );
    bench_config(
        "48 kHz frame 512  tail 4096 (M=8)",
        AecAttrs {
            sample_rate: 48_000,
            frame_size: 512,
            filter_length: 4096,
        },
        1000,
    );
}
