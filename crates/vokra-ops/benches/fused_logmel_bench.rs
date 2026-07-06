//! Fused-vs-unfused log-mel micro-benchmarks (M2-04-T11; FR-OP-01/03,
//! NFR-PF-13).
//!
//! `harness = false` (see Cargo.toml): a plain `fn main()` using `std::time`,
//! no external bench crate — the workspace keeps zero third-party
//! dependencies (NFR-DS-02). This follows the `fft_bench.rs:11-30` pattern
//! (warmup `iters.min(16)`, timed loop, mean µs/iter) and reports
//! `unfused_us / fused_us / speedup` per input case.
//!
//! The comparison isolates the M2-04 fused single-pass kernel
//! (`vokra_ops::fused_log_mel_scalar`) against the pre-fusion imperative
//! reference path (`stft` → `power` → `MelFilterbank::apply` →
//! `log10` → transpose), the same one that lives in
//! `vokra_models::whisper::mel::log_mel` under
//! `VOKRA_DISABLE_FUSION=1`. Both kernels see the identical
//! `StftAttrs` / `MelAttrs` from the runtime frontend spec, so the delta is
//! purely the fusion effect (eliminated `re`+`im`+`power`+`mel[frames, n_mels]`
//! intermediates + reused FFT scratch).
//!
//! Three inputs cover the Whisper front-end profile:
//!
//! 1. **1 s silence** — very short input; pad path dominates, fusion win is
//!    smaller (fewer real frames of arithmetic per call);
//! 2. **30 s sine at 16 kHz** — matches Whisper's `N_SAMPLES = 480 000`,
//!    the fixed 30 s clip length; steady-state harmonic input;
//! 3. **30 s pink-noise-like sum-of-sines** — broadband energy across all mel
//!    bands, exercising the mel filterbank projection at full capacity.
//!
//! For the 30 s cases the bench also prints RTF (`elapsed_s / 30.0`) so the
//! wall-clock cost is visible in units the CLI report understands. This
//! bench emits no perf gate — NFR-PF-13's 5 % regression check runs from
//! CI's `bench-regression` job over the recorded numbers (T12).
//!
//! # Note on the `VOKRA_DISABLE_FUSION` env
//!
//! `vokra_models::whisper::mel::log_mel` resolves `VOKRA_DISABLE_FUSION`
//! **exactly once** on the first call and locks it for the process lifetime
//! (see the fusion toggle in `mel.rs`), so a single bench process cannot flip
//! the toggle mid-run through the public `log_mel` entry point. This bench
//! therefore drives the two paths *directly* via the underlying primitives —
//! the measurement is faithful to the fused-vs-unfused contract, and the
//! result reflects exactly what `log_mel` computes under each env setting.

use std::hint::black_box;
use std::time::Instant;

use vokra_core::ir::graph::StftAttrs;
use vokra_ops::mel::MelFilterbank;
use vokra_ops::{
    fused_log_mel_scalar, mel_attrs_from_spec, mel_filterbank, stft, stft_attrs_from_spec,
};

use vokra_models::whisper::mel::{N_FRAMES, N_SAMPLES, SAMPLE_RATE, runtime_frontend_spec};

/// Result of one timed loop: mean per-iteration in microseconds.
fn bench<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    for _ in 0..iters.min(16) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
}

/// Unfused reference log-mel path — a direct inline of the imperative
/// pre-fusion code that lives in `vokra_models::whisper::mel::log_mel` under
/// `VOKRA_DISABLE_FUSION=1`. Kept physically close to that reference so the
/// bench measures the same arithmetic. Returns `[n_mels, N_FRAMES]` row-major.
fn unfused_log_mel_ref(
    pcm: &[f32],
    stft_attrs: &StftAttrs,
    fb: &MelFilterbank,
    n_mels: usize,
) -> Vec<f32> {
    // Pad / trim to 30 s (Whisper's fixed clip).
    let mut buf = vec![0.0f32; N_SAMPLES];
    let n = pcm.len().min(N_SAMPLES);
    buf[..n].copy_from_slice(&pcm[..n]);

    // STFT → power, drop the trailing frame.
    let stft_out = stft(&buf, stft_attrs).expect("valid whisper STFT attrs");
    let bins = stft_out.bins;
    let frames = stft_out.frames.min(N_FRAMES + 1);
    let kept = frames.min(N_FRAMES);
    let power = stft_out.power();

    // Mel projection → [kept, n_mels].
    let mel = fb.apply(&power[..kept * bins], kept);

    // log10 + dynamic-range compression + normalization, transposed to
    // [n_mels, N_FRAMES]. Frames beyond `kept` stay at the log floor.
    let floor_log = 1e-10f32.log10();
    let mut out = vec![floor_log; n_mels * N_FRAMES];
    let mut gmax = f32::NEG_INFINITY;
    for t in 0..kept {
        for m in 0..n_mels {
            let l = mel[t * n_mels + m].max(1e-10).log10();
            out[m * N_FRAMES + t] = l;
            if l > gmax {
                gmax = l;
            }
        }
    }
    if gmax == f32::NEG_INFINITY {
        return out;
    }
    let dyn_floor = gmax - 8.0;
    for v in &mut out {
        *v = (v.max(dyn_floor) + 4.0) / 4.0;
    }
    out
}

/// Runs one case (unfused + fused, mean µs/iter, speedup, optional RTF).
fn run_case(
    label: &str,
    pcm: &[f32],
    iters: u32,
    stft_attrs: &StftAttrs,
    fb: &MelFilterbank,
    n_mels: usize,
    duration_s: Option<f64>,
) {
    // Prime the fused kernel's output buffer once per case; the kernel
    // requires `out.len() == n_mels * N_FRAMES` on every call.
    let mut fused_out = vec![1e-10f32.log10(); n_mels * N_FRAMES];

    // Unfused reference: the pre-M2-04 imperative path (matches
    // `whisper::mel::log_mel` under `VOKRA_DISABLE_FUSION=1`). The bench calls
    // the primitives directly rather than routing through `log_mel` because
    // that function once-locks the env toggle, which would prevent A/B in a
    // single process — see the module header.
    let unfused_us = bench(iters, || {
        let out = unfused_log_mel_ref(black_box(pcm), stft_attrs, fb, n_mels);
        black_box(out);
    });

    // Fused single-pass kernel.
    let fused_us = bench(iters, || {
        // Reset floor for a clean run (matches `log_mel`'s pre-fill).
        for v in fused_out.iter_mut() {
            *v = 1e-10f32.log10();
        }
        fused_log_mel_scalar(black_box(pcm), stft_attrs, fb, N_FRAMES, &mut fused_out);
        black_box(&fused_out);
    });

    let speedup = unfused_us / fused_us;
    print!(
        "{label:<40} unfused {unfused_us:>10.2} us  fused {fused_us:>10.2} us  speedup {speedup:>5.2}x",
    );
    if let Some(secs) = duration_s {
        // RTF = wall-clock per call / real-time audio duration.
        let rtf_unfused = (unfused_us * 1e-6) / secs;
        let rtf_fused = (fused_us * 1e-6) / secs;
        print!("  rtf unfused {rtf_unfused:.5}  rtf fused {rtf_fused:.5}");
    }
    println!();
}

fn main() {
    println!("vokra-ops M2-04-T11 fused-vs-unfused log-mel benchmarks\n");

    // Whisper's fixed front-end: n_fft=400, hop=160, Slaney mel, n_mels=80.
    let n_mels: usize = 80;
    let spec = runtime_frontend_spec(n_mels);
    let stft_attrs = stft_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");
    let mel_attrs = mel_attrs_from_spec(&spec).expect("runtime frontend spec is well-formed");
    let fb = mel_filterbank(&mel_attrs);

    // Case 1: 1 s silence @ 16 kHz. Padded to 30 s, so the STFT grid is
    // identical to a 30 s clip; only the audio energy differs.
    let silence: Vec<f32> = vec![0.0; SAMPLE_RATE as usize];
    run_case("1s silence", &silence, 20, &stft_attrs, &fb, n_mels, None);

    // Case 2: 30 s sine @ 16 kHz, matches Whisper's N_SAMPLES = 480 000.
    // A 440 Hz tone lights up a narrow band of mel filters.
    let sine: Vec<f32> = (0..N_SAMPLES)
        .map(|t| {
            let x = t as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 440.0 * x).sin() * 0.5
        })
        .collect();
    run_case(
        "30s sine 440Hz",
        &sine,
        10,
        &stft_attrs,
        &fb,
        n_mels,
        Some(30.0),
    );

    // Case 3: 30 s pink-noise-like sum-of-sines with 1/f-ish amplitudes.
    // Broadband energy across the mel bands exercises the full filterbank
    // projection cost (no free zeros in the accumulate).
    let freqs: [f32; 12] = [
        60.0, 120.0, 240.0, 480.0, 960.0, 1_500.0, 2_400.0, 3_600.0, 5_000.0, 6_500.0, 7_000.0,
        7_800.0,
    ];
    let pink: Vec<f32> = (0..N_SAMPLES)
        .map(|t| {
            let x = t as f32 / SAMPLE_RATE as f32;
            let mut acc = 0.0f32;
            for &f in &freqs {
                // 1/sqrt(f) amplitude approximates pink noise power law.
                let a = 1.0 / f.sqrt();
                acc += a * (2.0 * std::f32::consts::PI * f * x).sin();
            }
            acc * 0.3
        })
        .collect();
    run_case(
        "30s pink-noise sum-of-sines",
        &pink,
        10,
        &stft_attrs,
        &fb,
        n_mels,
        Some(30.0),
    );

    println!("\nnote: fused kernel eliminates re/im/power/[frames,n_mels] intermediates");
    println!("and reuses one FFT scratch across ~3001 frames. Record these numbers in the PR.");
}
