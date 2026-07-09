//! `vokra-cli run` — load a GGUF and run its task on an input (M1-10a).
//!
//! ```text
//! vokra-cli run --model vad.gguf     --input speech.wav
//! vokra-cli run --model whisper.gguf --input speech.wav
//! vokra-cli run --model voice.gguf   --text "hello vokra" [--output out.wav]
//! ```
//!
//! The task is detected from the model architecture (see [`crate::engine`]);
//! VAD prints per-frame speech-probability summary, ASR prints the transcript,
//! and TTS writes a WAV (or reports the sample count when `--output` is absent).

use std::process::ExitCode;

use vokra_core::Session;

use crate::engine::{self, ModelTask};
use crate::wav;

pub(crate) const USAGE: &str = "\
vokra-cli run — load a GGUF and run VAD / ASR / TTS

USAGE:
    vokra-cli run --model <model.gguf> [--input <in.wav>] [--text <string>] [--output <out.wav>]

OPTIONS:
    --model <path>    GGUF model file (arch selects VAD / ASR / TTS)
    --input <path>    mono WAV input (required for VAD and ASR)
    --text <string>   text to synthesize (required for TTS)
    --output <path>   WAV file for the TTS output (optional)
    -h, --help        print this help
";

/// Parsed `run` arguments.
struct RunArgs {
    model: String,
    input: Option<String>,
    text: Option<String>,
    output: Option<String>,
}

fn parse_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(args.get(i + 1).ok_or("--model requires a value")?.clone());
                i += 2;
            }
            "--input" => {
                input = Some(args.get(i + 1).ok_or("--input requires a value")?.clone());
                i += 2;
            }
            "--text" => {
                text = Some(args.get(i + 1).ok_or("--text requires a value")?.clone());
                i += 2;
            }
            "--output" => {
                output = Some(args.get(i + 1).ok_or("--output requires a value")?.clone());
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    Ok(RunArgs {
        model: model.ok_or("--model is required")?,
        input,
        text,
        output,
    })
}

/// Entry point for `vokra-cli run`.
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let a = parse_args(args)?;
    let (session, task) = engine::load_session(&a.model)?;

    match task {
        ModelTask::Vad => {
            let path = a
                .input
                .as_deref()
                .ok_or("run (VAD): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let probs = run_vad(&session, &clip.samples, clip.sample_rate)?;
            let n = probs.len();
            let speech = probs.iter().filter(|&&p| p >= 0.5).count();
            let mean = if n == 0 {
                0.0
            } else {
                probs.iter().sum::<f32>() / n as f32
            };
            println!("vad: {n} frames, speech_frames={speech}, mean_prob={mean:.4}");
        }
        ModelTask::Asr => {
            let path = a
                .input
                .as_deref()
                .ok_or("run (ASR): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let text = run_asr(&session, &clip.samples)?;
            println!("asr: {text}");
        }
        ModelTask::Tts => {
            let text = a
                .text
                .as_deref()
                .ok_or("run (TTS): --text <string> is required")?;
            let audio = session.tts().synthesize(text).map_err(|e| e.to_string())?;
            match a.output.as_deref() {
                Some(out) => {
                    wav::write_wav(out, &audio.samples, audio.sample_rate)?;
                    println!(
                        "tts: wrote {} samples @ {} Hz -> {out}",
                        audio.samples.len(),
                        audio.sample_rate
                    );
                }
                None => {
                    let secs = audio.samples.len() as f64 / f64::from(audio.sample_rate);
                    println!(
                        "tts: {} samples, {secs:.3}s @ {} Hz (no --output; audio discarded)",
                        audio.samples.len(),
                        audio.sample_rate
                    );
                }
            }
        }
        // `mel-frontend` is a bench-only task (M2-04-T11) — it isolates the
        // Whisper log-mel path so the fused / unfused RTF isn't polluted by
        // encoder / decoder time. `vokra-cli run` has no analogous end-user
        // output, so reject rather than silently print something (FR-EX-08).
        ModelTask::MelFrontend => {
            return Err(
                "run: task `mel-frontend` is not supported (bench-only, see `vokra-cli bench --task mel-frontend`)"
                    .to_owned(),
            );
        }
        // Same posture for `cosyvoice2-synthetic` (M3-09-T24): bench-only
        // scaffold task. A real CosyVoice2 checkpoint's TTS run lands with
        // T07/T08 (LLM backbone forward) + T14/T15 (streaming pipeline
        // wired to a user-facing API) — that follow-on adds a
        // `ModelTask::Cosyvoice2` arm alongside `Tts` for the arch dispatch.
        ModelTask::Cosyvoice2Synthetic => {
            return Err(
                "run: task `cosyvoice2-synthetic` is not supported (bench-only, see \
                 `vokra-cli bench --task cosyvoice2-synthetic`)"
                    .to_owned(),
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Pushes the whole clip through a fresh VAD stream and returns the per-frame
/// speech probabilities.
fn run_vad(session: &Session, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
    let mut handle = session.open_vad_stream().map_err(|e| e.to_string())?;
    handle.push_pcm(pcm, sample_rate).map_err(|e| e.to_string())
}

/// Transcribes the clip and returns the recognized text.
fn run_asr(session: &Session, pcm: &[f32]) -> Result<String, String> {
    Ok(session
        .asr()
        .transcribe(pcm)
        .map_err(|e| e.to_string())?
        .text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    fn silero_fixture() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn parses_a_full_run_invocation() {
        let a = parse_args(&args(&[
            "--model", "m.gguf", "--input", "in.wav", "--output", "o.wav",
        ]))
        .expect("valid");
        assert_eq!(a.model, "m.gguf");
        assert_eq!(a.input.as_deref(), Some("in.wav"));
        assert_eq!(a.output.as_deref(), Some("o.wav"));
        assert_eq!(a.text, None);
    }

    #[test]
    fn rejects_missing_model_and_dangling_flag_and_stray_arg() {
        assert_eq!(
            parse_args(&args(&["--input", "x.wav"])).err().unwrap(),
            "--model is required"
        );
        assert_eq!(
            parse_args(&args(&["--model"])).err().unwrap(),
            "--model requires a value"
        );
        assert!(
            parse_args(&args(&["--bogus"]))
                .err()
                .unwrap()
                .contains("unexpected argument")
        );
    }

    #[test]
    fn run_vad_over_committed_fixture_yields_frames() {
        let (session, task) = engine::load_session(&silero_fixture()).expect("silero loads");
        assert_eq!(task, ModelTask::Vad);
        // 1 s of silence at 16 kHz completes several fixed-size frames.
        let pcm = vec![0.0f32; 16_000];
        let probs = run_vad(&session, &pcm, 16_000).expect("vad runs");
        assert!(!probs.is_empty(), "1 s of audio should complete >= 1 frame");
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
    }
}
