//! piper-plus native TTS demo (M0-07-T23): text → WAV, through the Vokra API.
//!
//! ```text
//! cargo run -p vokra-models --example tts_demo -- \
//!     --gguf voice.gguf --text "aiueo" --output out.wav [--language ja] [--deterministic]
//! ```
//!
//! The path is the real Vokra API: a [`Session`] with the piper-plus native TTS
//! injected via `with_tts_engine`, driven by `session.tts()`. **No onnxruntime
//! is used** (M0 core hypothesis, FR-LD-05) — the voice is a GGUF converted
//! offline by `vokra-convert`. Text → phoneme ids uses the placeholder
//! tokenizer built into the model (mirrors `vokra_piper_plus::MockPhonemizer`;
//! the real 8-language G2P reuse is M0-07-T09). The WAV header is written by
//! hand (16-bit PCM mono) to avoid an audio-file crate (NFR-DS-02).
//!
//! M0 note: the demo output carries **no watermark** — AudioSeal / C2PA
//! (FR-OP-90/91) land in M1-07 (milestones.md §4.2 note 1).

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use vokra_core::{Session, SynthesisRequest};
use vokra_models::piper_plus::PiperPlusTts;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match Opts::parse(&args) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("error: {msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };

    // Load the voice natively and inject it as the session's TTS engine.
    let model = match PiperPlusTts::from_path(&opts.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to load voice `{}`: {e}", opts.gguf);
            return ExitCode::FAILURE;
        }
    };
    let sample_rate = model.config().sample_rate;
    let session = match Session::from_file(&opts.gguf).build() {
        Ok(s) => s.with_tts_engine(Arc::new(model)),
        Err(e) => {
            eprintln!("error: failed to open session: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut request = SynthesisRequest::new(&opts.text);
    if let Some(lang) = &opts.language {
        request = request.with_language(lang);
    }
    if opts.deterministic {
        request = request.deterministic();
    }

    let start = Instant::now();
    let audio = match session.tts().synthesize_request(&request) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: synthesis failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let elapsed = start.elapsed().as_secs_f64();

    let duration = audio.samples.len() as f64 / f64::from(sample_rate);
    let rtf = if duration > 0.0 {
        elapsed / duration
    } else {
        0.0
    };
    println!(
        "synthesized {} samples ({duration:.2}s @ {sample_rate} Hz) in {elapsed:.3}s — RTF {rtf:.3}",
        audio.samples.len()
    );

    let wav = wav_pcm16(&audio.samples, sample_rate);
    if let Err(e) = std::fs::write(&opts.output, &wav) {
        eprintln!("error: failed to write `{}`: {e}", opts.output);
        return ExitCode::FAILURE;
    }
    println!("wrote {} ({} bytes)", opts.output, wav.len());
    ExitCode::SUCCESS
}

const USAGE: &str = "\
tts_demo — piper-plus native TTS: text → WAV (M0-07, no onnxruntime)

USAGE:
    cargo run -p vokra-models --example tts_demo -- \\
        --gguf <voice.gguf> --text <text> --output <out.wav> [--language <ja|en|...>] [--deterministic]
";

struct Opts {
    gguf: String,
    text: String,
    output: String,
    language: Option<String>,
    deterministic: bool,
}

impl Opts {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut gguf = None;
        let mut text = None;
        let mut output = "out.wav".to_owned();
        let mut language = None;
        let mut deterministic = false;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--gguf" => {
                    gguf = Some(next(args, &mut i, "--gguf")?);
                }
                "--text" => {
                    text = Some(next(args, &mut i, "--text")?);
                }
                "--output" => {
                    output = next(args, &mut i, "--output")?;
                }
                "--language" => {
                    language = Some(next(args, &mut i, "--language")?);
                }
                "--deterministic" => {
                    deterministic = true;
                    i += 1;
                }
                other => return Err(format!("unexpected argument `{other}`")),
            }
        }
        Ok(Self {
            gguf: gguf.ok_or("--gguf is required")?,
            text: text.ok_or("--text is required")?,
            output,
            language,
            deterministic,
        })
    }
}

fn next(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    let v = args
        .get(*i + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?
        .clone();
    *i += 2;
    Ok(v)
}

/// Encodes mono f32 PCM in `[-1, 1]` as a 16-bit PCM WAV byte buffer.
fn wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bits_per_sample = 16u16;
    let channels = 1u16;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_len = (samples.len() * 2) as u32;

    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
