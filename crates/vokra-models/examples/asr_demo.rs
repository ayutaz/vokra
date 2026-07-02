//! Whisper ASR demo (M0-06-T25).
//!
//! Transcribes a 16-bit PCM WAV to text on the CPU backend:
//!
//! ```text
//! cargo run -p vokra-models --release --example asr_demo -- \
//!     whisper-base.gguf input.wav [tokenizer.bin]
//! ```
//!
//! - the model is loaded from a GGUF (never ONNX — FR-LD-05);
//! - the optional third argument is the tokenizer blob produced by
//!   `tools/parity/dump_whisper_reference.py` (the M0 converter does not embed
//!   the vocabulary — see `whisper::tokenizer`); without it the demo prints the
//!   raw token ids;
//! - the WAV reader is a tiny std-only parser (no external crates —
//!   NFR-DS-02 / cargo-deny GPL-free); it accepts 16-bit PCM, mono or
//!   down-mixed stereo, and requires the model sample rate (`resample` is
//!   FR-OP-04 / M1, so a mismatch is an explicit error, not silent resampling);
//! - numeric parsing/formatting uses Rust's locale-independent APIs only
//!   (never `strtod` — NFR-RL-01).
//!
//! The selected CPU ISA path and a reference RTF are printed to stderr; the
//! transcript goes to stdout.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use vokra_backend_cpu::active_isa;
use vokra_core::AsrEngine; // brings `transcribe` into scope
use vokra_core::gguf::{FrontendSpec, GgufFile};
use vokra_models::whisper::{WhisperAsr, WhisperTokenizer};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: {} <model.gguf> <input.wav> [tokenizer.bin]",
            args.first().map(String::as_str).unwrap_or("asr_demo")
        );
        return ExitCode::from(2);
    }
    match run(&args[1], &args[2], args.get(3).map(String::as_str)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(model_path: &str, wav_path: &str, tokenizer_path: Option<&str>) -> Result<(), String> {
    let file = GgufFile::open(model_path).map_err(|e| format!("open GGUF `{model_path}`: {e}"))?;

    // Model sample rate from the frontend spec (bit-exact check is M1-03).
    let spec = FrontendSpec::from_gguf(&file).map_err(|e| format!("read frontend spec: {e}"))?;

    let mut asr = WhisperAsr::from_gguf(&file).map_err(|e| format!("load model: {e}"))?;
    if let Some(tk) = tokenizer_path {
        let blob = std::fs::read(tk).map_err(|e| format!("read tokenizer `{tk}`: {e}"))?;
        let tok = WhisperTokenizer::from_bytes(&blob, spec_eot(&file))
            .map_err(|e| format!("parse tokenizer: {e}"))?;
        asr = asr.with_tokenizer(tok);
    }

    let (pcm, sample_rate) = read_wav_pcm(wav_path)?;
    if sample_rate != spec.sample_rate {
        return Err(format!(
            "WAV sample rate {sample_rate} Hz != model {} Hz (resample is M1 / FR-OP-04; \
             re-sample the input offline)",
            spec.sample_rate
        ));
    }

    eprintln!("cpu ISA path: {:?}", active_isa());
    let audio_secs = pcm.len() as f64 / sample_rate as f64;

    let t0 = Instant::now();
    let transcription = asr
        .transcribe(&pcm)
        .map_err(|e| format!("transcribe: {e}"))?;
    let elapsed = t0.elapsed().as_secs_f64();

    println!("{}", transcription.text);
    let rtf = if audio_secs > 0.0 {
        elapsed / audio_secs
    } else {
        f64::NAN
    };
    eprintln!(
        "audio {audio_secs:.2}s, wall {elapsed:.2}s, RTF {rtf:.3} (reference only; the \
         NFR-PF-01 RTF < 0.3 gate is M1)"
    );
    Ok(())
}

/// Reads `vokra.whisper.eot` for the tokenizer (defaults to the multilingual
/// end-of-transcript id if absent).
fn spec_eot(file: &GgufFile) -> u32 {
    file.get("vokra.whisper.eot")
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(50257)
}

/// Minimal RIFF/WAVE reader: 16-bit integer PCM, mono or stereo (down-mixed).
///
/// Returns `(mono f32 samples in [-1, 1], sample_rate_hz)`. Every field is
/// bounds-checked; malformed input is a descriptive `Err`, never a panic.
fn read_wav_pcm(path: &str) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(Path::new(path)).map_err(|e| format!("read `{path}`: {e}"))?;
    let ctx = |m: &str| format!("WAV `{path}`: {m}");

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(ctx("not a RIFF/WAVE file"));
    }

    let mut fmt: Option<Fmt> = None;
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| ctx("chunk size runs past end of file"))?;
        match id {
            b"fmt " => fmt = Some(parse_fmt(&bytes[body_start..body_end]).map_err(|m| ctx(&m))?),
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        // Chunks are word-aligned (padded to even length).
        pos = body_end + (size & 1);
    }

    let fmt = fmt.ok_or_else(|| ctx("missing fmt chunk"))?;
    let data = data.ok_or_else(|| ctx("missing data chunk"))?;
    if fmt.audio_format != 1 {
        return Err(ctx("only integer PCM (format 1) is supported"));
    }
    if fmt.bits_per_sample != 16 {
        return Err(ctx("only 16-bit PCM is supported"));
    }
    if fmt.channels == 0 {
        return Err(ctx("zero channels"));
    }

    let ch = fmt.channels as usize;
    let frame = 2 * ch; // bytes per multi-channel frame
    let mut pcm = Vec::with_capacity(data.len() / frame.max(1));
    for f in data.chunks_exact(frame) {
        // Average the channels to mono.
        let mut acc = 0i32;
        for c in 0..ch {
            let s = i16::from_le_bytes([f[2 * c], f[2 * c + 1]]);
            acc += s as i32;
        }
        let avg = acc as f32 / ch as f32;
        pcm.push(avg / 32768.0);
    }
    Ok((pcm, fmt.sample_rate))
}

struct Fmt {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn parse_fmt(b: &[u8]) -> Result<Fmt, String> {
    if b.len() < 16 {
        return Err("fmt chunk too small".to_owned());
    }
    Ok(Fmt {
        audio_format: u16::from_le_bytes([b[0], b[1]]),
        channels: u16::from_le_bytes([b[2], b[3]]),
        sample_rate: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        bits_per_sample: u16::from_le_bytes([b[14], b[15]]),
    })
}
