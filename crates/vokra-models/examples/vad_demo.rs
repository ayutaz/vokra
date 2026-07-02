//! Silero VAD v5 demo (M0-05-T10): stream a mono WAV through the native VAD and
//! print per-frame speech probability plus the detected speech segments.
//!
//! ```text
//! cargo run -p vokra-models --example vad_demo -- [WAV] [THRESHOLD] [GGUF]
//! ```
//!
//! Defaults: the committed 16 kHz parity fixture, threshold 0.5, the corrected
//! both-rate fixture GGUF. WAV must be mono 8 kHz or 16 kHz (float32 or int16);
//! resampling is out of M0 scope (FR-OP-04 is M1), so any other rate is an
//! explicit error. Processing time per frame is printed for information only —
//! M0 defines no VAD performance gate.

use std::path::PathBuf;
use std::time::Instant;

use vokra_core::engines::VadEngine;
use vokra_models::silero_vad::wav::read_wav_f32;
use vokra_models::silero_vad::{SampleRate, SileroVadV5};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/parity/silero_vad")
        .join(name)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("vad_demo error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let wav_path = args
        .next()
        .map_or_else(|| fixture("test_16k.wav"), PathBuf::from);
    let threshold: f32 = args.next().map_or(Ok(0.5), |s| s.parse())?;
    let gguf_path = args
        .next()
        .map_or_else(|| fixture("silero-vad-v5.gguf"), PathBuf::from);

    let wav = read_wav_f32(&wav_path)?;
    let rate = SampleRate::from_hz(wav.sample_rate)?;
    let frame_len = rate.frame_len();
    let frame_secs = frame_len as f32 / wav.sample_rate as f32;

    println!(
        "loaded {} ({} Hz, {} samples, {:.2} s)",
        wav_path.display(),
        wav.sample_rate,
        wav.samples.len(),
        wav.samples.len() as f32 / wav.sample_rate as f32,
    );

    let model = SileroVadV5::open(&gguf_path)?;
    let mut stream = model.open_stream();

    let started = Instant::now();
    let probs = stream.push_pcm(&wav.samples, wav.sample_rate)?;
    let elapsed = started.elapsed();

    // Per-frame probability trace.
    println!("frame   time(s)   prob   speech?");
    for (i, &p) in probs.iter().enumerate() {
        let t = i as f32 * frame_secs;
        let mark = if p >= threshold { "*" } else { " " };
        println!("{i:5}  {t:7.3}  {p:6.4}   {mark}");
    }

    // Contiguous speech segments (>= threshold).
    let mut segments = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &p) in probs.iter().enumerate() {
        match (start, p >= threshold) {
            (None, true) => start = Some(i),
            (Some(s), false) => {
                segments.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        segments.push((s, probs.len()));
    }

    println!("\nspeech segments (threshold {threshold:.2}):");
    if segments.is_empty() {
        println!("  (none above threshold)");
    } else {
        for (s, e) in &segments {
            println!(
                "  {:.3}s .. {:.3}s ({} frames)",
                *s as f32 * frame_secs,
                *e as f32 * frame_secs,
                e - s
            );
        }
    }

    let audio_secs = probs.len() as f32 * frame_secs;
    let rtf = elapsed.as_secs_f32() / audio_secs.max(f32::EPSILON);
    println!(
        "\nprocessed {} frames in {:.2} ms ({:.3} ms/frame, RTF {:.4}) — informational only",
        probs.len(),
        elapsed.as_secs_f32() * 1e3,
        elapsed.as_secs_f32() * 1e3 / probs.len().max(1) as f32,
        rtf,
    );
    Ok(())
}
