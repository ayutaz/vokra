//! `vokra-misaki-g2p` — text → speech with the **real** upstream misaki G2P
//! driving Vokra's native Kokoro-82M TTS.
//!
//! ```text
//! # Synthesize with an on-disk voice name (voicepack row selected by name):
//! vokra-misaki-g2p --kokoro kokoro-82m.gguf --text "Hello world" \
//!     --lang en --voice af_bella --out hello.wav
//!
//! # Japanese with an explicit venv interpreter (recommended: `pip install misaki[ja]`):
//! vokra-misaki-g2p --kokoro kokoro-82m.gguf --text "こんにちは" \
//!     --lang ja --voice jf_alpha --python .venv/bin/python --out hi.wav
//!
//! # Dump the phoneme id sequence without synthesizing:
//! vokra-misaki-g2p --kokoro kokoro-82m.gguf --text "Hello" --lang en --dump
//! ```
//!
//! This binary lives OUTSIDE the zero-dependency workspace (see `Cargo.toml`):
//! it is the opt-in bridge that runs the Python misaki. The Kokoro runtime it
//! drives stays zero-dependency.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use vokra_misaki_g2p::{MisakiG2p, MisakiLang};
use vokra_models::kokoro::KokoroTts;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    kokoro: PathBuf,
    text: String,
    lang: MisakiLang,
    voice: Option<String>,
    python: Option<PathBuf>,
    dump: bool,
    out: Option<PathBuf>,
    noise: f32,
    length: f32,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let kokoro = KokoroTts::from_path(&args.kokoro)
        .map_err(|e| format!("load kokoro {:?}: {e}", args.kokoro))?;
    let g2p = MisakiG2p::from_kokoro(&kokoro, args.python.clone())
        .map_err(|e| format!("build G2P: {e}"))?;

    // text → phoneme string (raw misaki output) → phoneme ids (Kokoro's table).
    let phonemes = g2p
        .phonemize_string(&args.text, args.lang)
        .map_err(|e| format!("misaki phonemize: {e}"))?;
    let ids = g2p
        .phonemes_to_ids(&phonemes)
        .map_err(|e| format!("phoneme id lookup: {e}"))?;

    eprintln!(
        "misaki: text={} chars, phonemes={:?} ({} chars), ids={} tokens, symbol_table={} entries",
        args.text.chars().count(),
        phonemes,
        phonemes.chars().count(),
        ids.len(),
        g2p.symbol_count(),
    );

    if args.dump {
        // Machine-readable diagnostic on stdout, no synthesis.
        print!("[");
        for (i, id) in ids.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{id}");
        }
        println!("]");
        return Ok(());
    }

    let out = args
        .out
        .as_ref()
        .ok_or_else(|| "no --out path given (required unless --dump)".to_owned())?;

    // Feed the ids to Kokoro's low-level synth path. `voice` picks a row of the
    // voicepack when it is stacked into the GGUF; `style_override = None` lets
    // the voice name resolve the style vector (M2-07-T02).
    let audio = kokoro
        .synthesize_phonemes(&ids, args.voice.as_deref(), None, args.noise, args.length)
        .map_err(|e| format!("kokoro synthesize: {e}"))?;

    write_wav_16bit(out, &audio.samples, audio.sample_rate)
        .map_err(|e| format!("write wav {out:?}: {e}"))?;

    eprintln!(
        "wrote {} samples @ {} Hz to {:?}",
        audio.samples.len(),
        audio.sample_rate,
        out,
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut kokoro: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut lang: Option<MisakiLang> = None;
    let mut voice: Option<String> = None;
    let mut python: Option<PathBuf> = None;
    let mut dump = false;
    let mut out: Option<PathBuf> = None;
    let mut noise = 0.0f32;
    let mut length = 1.0f32;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--kokoro" | "--voice-gguf" => {
                kokoro = Some(PathBuf::from(next_val(&mut args, &a)?));
            }
            "--text" => {
                text = Some(next_val(&mut args, &a)?);
            }
            "--lang" => {
                lang =
                    Some(MisakiLang::parse(&next_val(&mut args, &a)?).map_err(|e| e.to_string())?);
            }
            "--voice" => {
                voice = Some(next_val(&mut args, &a)?);
            }
            "--python" => {
                python = Some(PathBuf::from(next_val(&mut args, &a)?));
            }
            "--dump" => {
                dump = true;
            }
            "--out" => {
                out = Some(PathBuf::from(next_val(&mut args, &a)?));
            }
            "--noise-scale" => {
                noise = next_val(&mut args, &a)?
                    .parse::<f32>()
                    .map_err(|e| format!("--noise-scale: {e}"))?;
            }
            "--length-scale" => {
                length = next_val(&mut args, &a)?
                    .parse::<f32>()
                    .map_err(|e| format!("--length-scale: {e}"))?;
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown flag {other:?}; try --help"));
            }
        }
    }

    Ok(Args {
        kokoro: kokoro.ok_or_else(|| "--kokoro is required".to_owned())?,
        text: text.ok_or_else(|| "--text is required".to_owned())?,
        lang: lang.ok_or_else(|| "--lang is required".to_owned())?,
        voice,
        python,
        dump,
        out,
        noise,
        length,
    })
}

fn next_val(args: &mut std::iter::Skip<std::env::Args>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn print_help() {
    eprintln!(
        r#"vokra-misaki-g2p — text → Kokoro TTS with the upstream misaki G2P.

USAGE:
    vokra-misaki-g2p --kokoro <PATH> --text <STR> --lang <LANG> [OPTIONS]

REQUIRED:
    --kokoro <PATH>          Kokoro GGUF converted with `--config`
    --text <STR>             text to synthesize
    --lang <LANG>            en / en-gb / ja / zh / ko

VOICE:
    --voice <NAME>           voicepack row name (kokoro_v1_0 requires --stack-voicepack
                             at convert time; a bare voice GGUF resolves style_override
                             through this flag)

OUTPUT:
    --out <PATH>             write 16-bit PCM WAV (required unless --dump)
    --dump                   print phoneme id list on stdout, do not synthesize

MODEL:
    --noise-scale <F>        stochastic SineGen dither (default 0.0)
    --length-scale <F>       tempo scaling on the per-phoneme sigmoid.sum
                             (default 1.0; larger = slower)

PYTHON:
    --python <PATH>          interpreter that has `misaki` importable
                             (default: python3 on PATH)

MISCELLANEOUS:
    -h, --help               this text
"#
    );
}

/// Minimal RIFF/WAVE writer — 16-bit signed PCM, mono. Emits the standard
/// 44-byte header; sample count is bounded so multiplying by 2 cannot
/// overflow on any host with plausible memory.
fn write_wav_16bit(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_samples: u32 = u32::try_from(samples.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "too many samples"))?;
    let data_size = n_samples.checked_mul(2).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "data size overflow")
    })?;
    let riff_size = data_size.checked_add(36).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "RIFF size overflow")
    })?;

    // RIFF header.
    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;

    // fmt chunk (PCM 16-bit, mono).
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // chunk size
    w.write_all(&1u16.to_le_bytes())?; // format = PCM
    w.write_all(&1u16.to_le_bytes())?; // channels
    w.write_all(&sample_rate.to_le_bytes())?; // sample rate
    let byte_rate = sample_rate.checked_mul(2).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "byte rate overflow")
    })?;
    w.write_all(&byte_rate.to_le_bytes())?; // byte rate
    w.write_all(&2u16.to_le_bytes())?; // block align
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk.
    w.write_all(b"data")?;
    w.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        // Clip and round-to-nearest at the i16 boundary.
        let clipped = s.clamp(-1.0, 1.0);
        let quantized = (clipped * 32767.0).round() as i16;
        w.write_all(&quantized.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
