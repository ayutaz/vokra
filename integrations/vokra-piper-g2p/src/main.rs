//! `vokra-piper-g2p` — text → speech with the **real** 8-language piper-plus
//! G2P driving Vokra's native piper-plus TTS.
//!
//! ```text
//! vokra-piper-g2p --voice voice.gguf --text "こんにちは" --lang ja --out hello.wav
//! # zero-shot voice cloning from a reference utterance (CAM++):
//! vokra-piper-g2p --voice voice.gguf --text "Hello" --lang en \
//!     --ref ref.wav --speaker-gguf campplus.gguf --out cloned.wav
//! # dump the phonemizer output (ids / prosody / lid) without synthesizing:
//! vokra-piper-g2p --voice voice.gguf --text "こんにちは" --lang ja --dump
//! ```
//!
//! This binary lives OUTSIDE the zero-dependency workspace (see the crate
//! `Cargo.toml`): it is the opt-in bridge that links the third-party G2P. The
//! Vokra runtime it calls stays zero-dependency.

use std::path::PathBuf;
use std::process::ExitCode;

use vokra_core::SynthesisRequest;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::speaker::SpeakerEncoder;
use vokra_piper_g2p::PiperPlusG2p;

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
    voice: PathBuf,
    text: String,
    lang: Option<String>,
    reference: Option<PathBuf>,
    speaker_gguf: Option<PathBuf>,
    out: PathBuf,
    dump: bool,
    stochastic: bool,
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    let voice = PiperPlusTts::from_path(&args.voice)
        .map_err(|e| format!("load voice {:?}: {e}", args.voice))?;
    let g2p = PiperPlusG2p::from_voice(&voice).map_err(|e| format!("build G2P: {e}"))?;

    // Optional zero-shot speaker embedding from a reference utterance (CAM++).
    let speaker_embedding = match (&args.reference, &args.speaker_gguf) {
        (Some(ref_wav), Some(spk_gguf)) => {
            let encoder = SpeakerEncoder::from_path(spk_gguf)
                .map_err(|e| format!("load speaker encoder {spk_gguf:?}: {e}"))?;
            let (pcm, sr) = read_wav_mono(ref_wav)?;
            let emb = voice
                .embed_reference(&encoder, &pcm, sr)
                .map_err(|e| format!("embed reference: {e}"))?;
            eprintln!(
                "reference {:?}: {} samples @ {sr} Hz → speaker embedding [{}]",
                ref_wav,
                pcm.len(),
                emb.len()
            );
            Some(emb)
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err("--ref and --speaker-gguf must be given together".to_string());
        }
        (None, None) => None,
    };

    // Show what the real G2P produced (also the reproducible parity surface).
    let utt = {
        use vokra_piper_plus::Phonemizer;
        g2p.phonemize_full(&args.text)
            .map_err(|e| format!("phonemize: {e}"))?
    };
    let nonzero_prosody = utt.prosody.iter().filter(|p| **p != [0, 0, 0]).count();
    eprintln!(
        "G2P: {} phoneme ids, lid={}, {} non-zero prosody triples",
        utt.ids.len(),
        utt.lid,
        nonzero_prosody
    );
    if args.dump {
        println!("ids={:?}", utt.ids);
        println!("lid={}", utt.lid);
        println!("prosody={:?}", utt.prosody);
        return Ok(());
    }

    let mut request = SynthesisRequest::new(&args.text);
    if let Some(lang) = &args.lang {
        request = request.with_language(lang.clone());
    }
    if let Some(emb) = speaker_embedding {
        request = request.with_speaker_embedding(emb);
    }
    if !args.stochastic {
        request = request.deterministic();
    }

    let audio = voice
        .synthesize_full(&request, &g2p)
        .map_err(|e| format!("synthesize: {e}"))?;
    write_wav_mono(&args.out, &audio.samples, audio.sample_rate)?;
    eprintln!(
        "wrote {:?}: {} samples @ {} Hz ({:.2}s)",
        args.out,
        audio.samples.len(),
        audio.sample_rate,
        audio.samples.len() as f32 / audio.sample_rate as f32
    );
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut voice = None;
    let mut text = None;
    let mut lang = None;
    let mut reference = None;
    let mut speaker_gguf = None;
    let mut out = PathBuf::from("out.wav");
    let mut dump = false;
    let mut stochastic = false;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Reads the value that must follow a flag at `argv[i]`, advancing `i` past it.
    fn value(argv: &[String], i: &mut usize, name: &str) -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} requires a value"))
    }
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--voice" => voice = Some(PathBuf::from(value(&argv, &mut i, "--voice")?)),
            "--text" => text = Some(value(&argv, &mut i, "--text")?),
            "--lang" => lang = Some(value(&argv, &mut i, "--lang")?),
            "--ref" => reference = Some(PathBuf::from(value(&argv, &mut i, "--ref")?)),
            "--speaker-gguf" => {
                speaker_gguf = Some(PathBuf::from(value(&argv, &mut i, "--speaker-gguf")?))
            }
            "--out" => out = PathBuf::from(value(&argv, &mut i, "--out")?),
            "--dump" => dump = true,
            "--stochastic" => stochastic = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other} (see --help)")),
        }
        i += 1;
    }

    Ok(Args {
        voice: voice.ok_or("--voice <voice.gguf> is required")?,
        text: text.ok_or("--text <string> is required")?,
        lang,
        reference,
        speaker_gguf,
        out,
        dump,
        stochastic,
    })
}

fn print_usage() {
    eprintln!(
        "vokra-piper-g2p — real piper-plus G2P → Vokra native TTS\n\n\
         USAGE:\n  vokra-piper-g2p --voice <gguf> --text <str> [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20 --voice <path>         voice GGUF (converted piper-plus model)   [required]\n\
         \x20 --text  <string>       text to synthesize                        [required]\n\
         \x20 --lang  <code>         language hint (ja/en/zh/es/fr/pt); else auto-detect\n\
         \x20 --ref   <wav>          reference utterance for zero-shot cloning (needs --speaker-gguf)\n\
         \x20 --speaker-gguf <path>  CAM++ speaker-encoder GGUF\n\
         \x20 --out   <wav>          output WAV (default out.wav)\n\
         \x20 --dump                 print G2P ids/prosody/lid and exit (no synthesis)\n\
         \x20 --stochastic           enable noise (default is deterministic)"
    );
}

// --- Minimal WAV I/O (mono; PCM16 or IEEE-float32 in, PCM16 out) -----------

fn read_wav_mono(path: &PathBuf) -> Result<(Vec<f32>, u32), String> {
    let b = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    if b.len() < 44 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{path:?}: not a RIFF/WAVE file"));
    }
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);

    // Walk chunks to find `fmt ` and `data` (skip any others, e.g. LIST/fact).
    let (mut fmt_off, mut data_off, mut data_len) = (None, None, 0usize);
    let mut p = 12;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let sz = u32le(p + 4) as usize;
        let body = p + 8;
        if id == b"fmt " {
            fmt_off = Some(body);
        } else if id == b"data" {
            data_off = Some(body);
            data_len = sz.min(b.len().saturating_sub(body));
        }
        p = body + sz + (sz & 1); // chunks are word-aligned
    }
    let fmt = fmt_off.ok_or_else(|| format!("{path:?}: missing fmt chunk"))?;
    let data = data_off.ok_or_else(|| format!("{path:?}: missing data chunk"))?;

    let format = u16le(fmt); // 1 = PCM, 3 = IEEE float
    let channels = u16le(fmt + 2).max(1) as usize;
    let sample_rate = u32le(fmt + 4);
    let bits = u16le(fmt + 14) as usize;

    let mut mono: Vec<f32> = Vec::new();
    match (format, bits) {
        (1, 16) => {
            let frame = 2 * channels;
            for f in b[data..data + data_len].chunks_exact(frame) {
                // average channels → mono
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let s = i16::from_le_bytes([f[2 * c], f[2 * c + 1]]);
                    acc += s as f32 / 32768.0;
                }
                mono.push(acc / channels as f32);
            }
        }
        (3, 32) => {
            let frame = 4 * channels;
            for f in b[data..data + data_len].chunks_exact(frame) {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    let s =
                        f32::from_le_bytes([f[4 * c], f[4 * c + 1], f[4 * c + 2], f[4 * c + 3]]);
                    acc += s;
                }
                mono.push(acc / channels as f32);
            }
        }
        _ => {
            return Err(format!(
                "{path:?}: unsupported WAV format={format} bits={bits} (need PCM16 or float32)"
            ));
        }
    }
    Ok((mono, sample_rate))
}

fn write_wav_mono(path: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    let riff_len = 36 + data_len as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("write {path:?}: {e}"))
}
