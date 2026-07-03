//! Reference RTF bench for the piper-plus native TTS (M1-01-D).
//!
//! `harness = false` + `std::time::Instant` — no external bench crate, so the
//! workspace stays dependency-free (NFR-DS-02); it mirrors the M0-08
//! `vokra-backend-cpu` bench style. It measures **real-time factor**
//! `RTF = synth_time / audio_duration` for a fixed deterministic utterance and
//! **asserts nothing**: M0/M1 define the RTF gate as a bench measurement, not a
//! unit assert, and the `< 0.5` confirmation on real hardware is M1-01-F
//! (NFR-PF-02). The dominant decoder/flow convolutions run through the
//! `vokra-backend-cpu` SIMD GEMM after M1-01-D (`piper_plus/nn.rs`).
//!
//! The voice GGUF is large and uncommitted, so this is gated on
//! `VOKRA_PIPER_GGUF` (like the parity tests) and prints a skip when it is
//! unset — safe to build/run in CI without a checkpoint.
//!
//! ```text
//! VOKRA_PIPER_GGUF=voice.gguf cargo bench -p vokra-models --bench piper_rtf
//! ```

use std::hint::black_box;
use std::time::Instant;

use vokra_models::piper_plus::PiperPlusTts;

fn main() {
    let Ok(path) = std::env::var("VOKRA_PIPER_GGUF") else {
        eprintln!("piper_rtf: set VOKRA_PIPER_GGUF=<voice.gguf> to measure (skipped)");
        return;
    };
    let voice = match PiperPlusTts::from_path(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("piper_rtf: failed to load {path}: {e}");
            return;
        }
    };
    let sample_rate = voice.config().sample_rate as f64;
    let length_scale = voice.config().length_scale;

    // A fixed, deterministic utterance built from literal vowel phonemes the
    // placeholder tokenizer maps (repeated so the run is long enough to time).
    let ids = voice.tokenize(&"aiueo ".repeat(20));
    if ids.is_empty() {
        eprintln!("piper_rtf: voice has no vowel phonemes for the fixed utterance (skipped)");
        return;
    }
    let ids: &[i64] = &ids;

    // Deterministic (noise scales 0) so the measurement is reproducible.
    let synth = |ids: &[i64]| {
        voice
            .synthesize_phonemes(ids, 0, 0.0, length_scale, 0.0)
            .expect("synthesize")
    };

    // Warm-up (page-in weights, prime the ISA dispatch table).
    let warm = synth(ids);
    let samples = warm.samples.len();
    black_box(&warm.samples);

    let iters = 5u32;
    let start = Instant::now();
    for _ in 0..iters {
        let audio = synth(black_box(ids));
        black_box(&audio.samples);
    }
    let synth_s = start.elapsed().as_secs_f64() / f64::from(iters);
    let audio_s = samples as f64 / sample_rate;
    let rtf = if audio_s > 0.0 {
        synth_s / audio_s
    } else {
        f64::NAN
    };

    println!("vokra-models piper-plus RTF bench (reference only; no gate)");
    println!(
        "  {samples} samples, {audio_s:.3}s audio @ {sample_rate:.0} Hz, {synth_s:.4}s/synth  ->  RTF {rtf:.4}"
    );
}
