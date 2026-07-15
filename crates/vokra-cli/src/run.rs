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
                  [--beam-size <N>] [--length-penalty <α>] [--no-repeat-ngram <N>]
                  [--fixture-tokenizer] [--interrupt-after <N>] [--deterministic]

OPTIONS:
    --model <path>              GGUF model file (arch selects VAD / ASR / TTS / S2S)
    --input <path>              mono WAV input (required for VAD and ASR;
                                optional recorded context audio for S2S —
                                the explicit AEC bypass path, FR-OP-60)
    --text <string>             text to synthesize (TTS) / the reply text CSM
                                speaks (S2S — caller-supplied, the model does
                                not generate text)
    --output <path>             WAV file for the TTS / S2S output (optional)
    --beam-size <N>             ASR beam-search width (default 1 = greedy).
                                Currently only honored for `voxtral` arch —
                                other archs error out on --beam-size > 1
                                rather than silently ignoring the flag
                                (FR-EX-08).
    --length-penalty <α>        GNMT length-penalty exponent for beam search
                                (default 0.6). See `voxtral::BeamConfig`.
    --no-repeat-ngram <N>       Block repeated n-grams of length N during
                                beam search (default 0 = disabled).
    --fixture-tokenizer         S2S only: swap the (T29-gated) embedded
                                tokenizer for the explicit fixture byte
                                tokenizer — host-only smoke, linguistically
                                meaningless output (never inferred, FR-EX-08)
    --interrupt-after <N>       S2S only: stream frames and barge-in
                                (M3-14 semantics) after N frames — the T19
                                interrupt demo path
    --deterministic             S2S only: temperature-0 sampling
                                (reproducible smoke / parity anchor)
    -h, --help                  print this help
";

/// Parsed `run` arguments.
struct RunArgs {
    model: String,
    input: Option<String>,
    text: Option<String>,
    output: Option<String>,
    /// Beam-search width (default 1 = greedy). Only honored for `voxtral`
    /// arch — other archs error out on `> 1` rather than silently ignoring
    /// (FR-EX-08).
    beam_size: usize,
    /// GNMT length-penalty exponent (default 0.6, per `BeamConfig`).
    length_penalty: f32,
    /// Block repeated n-grams of this length during beam search
    /// (default 0 = disabled).
    no_repeat_ngram: usize,
    /// S2S: explicit fixture-tokenizer opt-in (host-only smoke).
    fixture_tokenizer: bool,
    /// S2S: barge-in after N streamed frames (T19 demo).
    interrupt_after: Option<usize>,
    /// S2S: deterministic (temperature-0) sampling.
    deterministic: bool,
}

fn parse_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut output: Option<String> = None;
    // Beam-search defaults: greedy (beam_size = 1). Length-penalty 0.6 is
    // only meaningful when beam_size > 1; the default is arbitrary but
    // matches `voxtral::BeamConfig::with_beam_size` so the same value flows
    // through if the user only passes `--beam-size`.
    let mut beam_size: usize = 1;
    let mut length_penalty: f32 = 0.6;
    let mut no_repeat_ngram: usize = 0;
    let mut fixture_tokenizer = false;
    let mut interrupt_after: Option<usize> = None;
    let mut deterministic = false;

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
            "--beam-size" => {
                let v = args.get(i + 1).ok_or("--beam-size requires a value")?;
                beam_size = v
                    .parse()
                    .map_err(|e| format!("--beam-size must be an unsigned integer: {e}"))?;
                if beam_size == 0 {
                    return Err("--beam-size must be >= 1".to_owned());
                }
                i += 2;
            }
            "--length-penalty" => {
                let v = args.get(i + 1).ok_or("--length-penalty requires a value")?;
                length_penalty = v
                    .parse()
                    .map_err(|e| format!("--length-penalty must be a float: {e}"))?;
                if !length_penalty.is_finite() || length_penalty < 0.0 {
                    return Err(format!(
                        "--length-penalty must be a non-negative finite float (got {length_penalty})"
                    ));
                }
                i += 2;
            }
            "--no-repeat-ngram" => {
                let v = args
                    .get(i + 1)
                    .ok_or("--no-repeat-ngram requires a value")?;
                no_repeat_ngram = v
                    .parse()
                    .map_err(|e| format!("--no-repeat-ngram must be an unsigned integer: {e}"))?;
                i += 2;
            }
            "--fixture-tokenizer" => {
                fixture_tokenizer = true;
                i += 1;
            }
            "--interrupt-after" => {
                let v = args
                    .get(i + 1)
                    .ok_or("--interrupt-after requires a value")?;
                interrupt_after =
                    Some(v.parse().map_err(|e| {
                        format!("--interrupt-after must be an unsigned integer: {e}")
                    })?);
                i += 2;
            }
            "--deterministic" => {
                deterministic = true;
                i += 1;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    Ok(RunArgs {
        model: model.ok_or("--model is required")?,
        input,
        text,
        output,
        beam_size,
        length_penalty,
        no_repeat_ngram,
        fixture_tokenizer,
        interrupt_after,
        deterministic,
    })
}

/// Entry point for `vokra-cli run`.
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let a = parse_args(args)?;
    let hint = a
        .fixture_tokenizer
        .then_some(engine::TaskHint::CsmFixtureTokenizer);
    let (session, task) =
        engine::load_session_with_backend(&a.model, vokra_core::BackendKind::Cpu, hint)?;

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
            // Beam-search wiring for ASR (M3-10 Voxtral). The current
            // `engine.rs` arch dispatch only wires Whisper on the ASR
            // path — Voxtral's arch string is not yet routed here (that
            // lands with a follow-up ticket alongside `ARCH_VOXTRAL`).
            // Whisper's beam-search entry point lives on a separate
            // `WhisperBeamScorer` and is not exposed through the shared
            // `AsrEngine` trait, so passing `--beam-size > 1` today would
            // be silently ignored on the Whisper path.
            //
            // FR-EX-08 posture: rather than silently dropping the flag,
            // hard-error when a beam-only flag is set on an arch whose
            // dispatch does not honor it.
            if a.beam_size > 1 || a.no_repeat_ngram > 0 {
                return Err(
                    "run (ASR): --beam-size > 1 / --no-repeat-ngram are only supported for the \
                     Voxtral arch. The current build's arch dispatch does not route Voxtral \
                     through `vokra-cli run` yet (M3-10 follow-up). Run the Voxtral beam decode \
                     via the `voxtral::VoxtralAsr::transcribe_beam` API directly or wait for the \
                     arch wiring."
                        .to_owned(),
                );
            }
            // Length-penalty defaults to 0.6 (matching `BeamConfig`). A
            // user who explicitly set --length-penalty AND beam_size = 1
            // is passing a flag that has no effect (greedy ignores the
            // penalty); we detect that combination by comparing to the
            // parser default. Rather than surfacing that as a hard error
            // (which would trip normal users who explored the flag), we
            // print an informational note and continue.
            #[allow(clippy::float_cmp)]
            if a.beam_size == 1 && a.length_penalty != 0.6 {
                eprintln!(
                    "run (ASR): note — --length-penalty is only applied when --beam-size > 1 \
                     (greedy ignores the length penalty)."
                );
            }
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
        ModelTask::S2s => {
            run_s2s(&session, &a)?;
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

/// The S2S (Sesame CSM) demo path — T20: recorded-file dialog turn through
/// the injected `S2sEngine` (batch) or, with `--interrupt-after`, the
/// streaming loop + M3-14-contract barge-in demo (T19).
fn run_s2s(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::DialogRequest;

    let text = a.text.as_deref().ok_or(
        "run (S2S): --text <reply text> is required — CSM speaks caller-supplied \
                text (it does not generate a reply; ADR M4-05 §D1-(b))",
    )?;
    let mut request = DialogRequest::new(text);
    if a.deterministic {
        request = request.deterministic();
    }
    if let Some(path) = a.input.as_deref() {
        let clip = wav::read_wav(path)?;
        request = request.with_input_audio(clip.samples);
    }

    if let Some(after) = a.interrupt_after {
        // Streaming + barge-in demo. The engine handle is only reachable
        // through the facade for batch dialog; the streaming surface is a
        // Rust API on the concrete engine, so this arm rebuilds it from
        // the model path (same GGUF, same synthesized bridge).
        use vokra_models::csm::{CsmEngine, CsmStreamConfig, EchoPath, FixtureByteTokenizer};
        let engine = CsmEngine::from_path(&a.model).map_err(|e| e.to_string())?;
        let engine = if a.fixture_tokenizer {
            let vocab = engine.config().text_vocab_size;
            engine
                .with_tokenizer(std::sync::Arc::new(
                    FixtureByteTokenizer::new(vocab).map_err(|e| e.to_string())?,
                ))
                .map_err(|e| e.to_string())?
        } else {
            engine
        };
        let engine = engine.with_echo_path(EchoPath::BypassRecordedInput);
        let mut stream = engine
            .open_stream(
                &request,
                Some(CsmStreamConfig {
                    max_frames: after * 4 + 8,
                }),
            )
            .map_err(|e| e.to_string())?;
        let handle = stream.interrupt_handle();
        let mut sink: Vec<vokra_core::StreamEvent> = Vec::new();
        let mut pcm = Vec::new();
        let mut frames = 0usize;
        while let Some(chunk) = stream.next_frame(&mut sink).map_err(|e| e.to_string())? {
            pcm.extend_from_slice(chunk);
            frames += 1;
            if frames == after {
                handle.interrupt();
            }
        }
        println!(
            "s2s: streamed {frames} frames ({} samples), stopped = {:?} (barge-in after {after})",
            pcm.len(),
            stream.stopped()
        );
        if let Some(out) = a.output.as_deref() {
            let sr = engine.config().sample_rate;
            wav::write_wav(out, &pcm, sr)?;
            println!("s2s: wrote {} samples @ {sr} Hz -> {out}", pcm.len());
        }
        return Ok(());
    }

    let turn = session
        .s2s()
        .dialog_request(&request)
        .map_err(|e| e.to_string())?;
    let audio = turn
        .audio
        .ok_or("run (S2S): the engine returned no audio")?;
    match a.output.as_deref() {
        Some(out) => {
            wav::write_wav(out, &audio.samples, audio.sample_rate)?;
            println!(
                "s2s: \"{}\" -> {} samples @ {} Hz -> {out}",
                turn.text,
                audio.samples.len(),
                audio.sample_rate
            );
        }
        None => {
            let secs = audio.samples.len() as f64 / f64::from(audio.sample_rate);
            println!(
                "s2s: \"{}\" -> {} samples, {secs:.3}s @ {} Hz (no --output; audio discarded)",
                turn.text,
                audio.samples.len(),
                audio.sample_rate
            );
        }
    }
    Ok(())
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

    /// Writes a synthesized-fixture CSM GGUF (tiny shape config + mimi
    /// chunk + provenance + a placeholder tokenizer blob) into a temp file
    /// and returns its path — the M4-05-T20 host-only smoke input.
    fn csm_fixture_gguf(tag: &str) -> std::path::PathBuf {
        use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};
        use vokra_models::csm::CsmConfig;
        use vokra_models::mimi::MimiNeuralConfig;
        let cfg = CsmConfig::tiny_for_tests();
        let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
        mimi_cfg.quantizer.n_q = cfg.n_codebooks;
        mimi_cfg.quantizer.bins = cfg.audio_vocab_size;
        let mut fixed = cfg.clone();
        fixed.sample_rate = mimi_cfg.sample_rate;
        fixed.frame_rate_mhz = mimi_cfg.frame_rate_mhz;
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "csm");
        vokra_core::stamp_provenance(
            &mut b,
            vokra_core::LicenseClass::Permissive,
            "Apache-2.0",
            Some("sesame/csm-1b"),
            None,
        );
        fixed.write_gguf_metadata(&mut b);
        mimi_cfg.write_gguf_metadata(&mut b);
        b.add_metadata(
            "vokra.tokenizer.model",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: vec![GgufMetadataValue::U8(1)],
            }),
        );
        let path = std::env::temp_dir().join(format!(
            "vokra-cli-csm-smoke-{tag}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, b.to_bytes().expect("serialize")).expect("write fixture");
        path
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
        // Defaults for beam-search flags.
        assert_eq!(a.beam_size, 1);
        assert!((a.length_penalty - 0.6).abs() < 1e-6);
        assert_eq!(a.no_repeat_ngram, 0);
    }

    #[test]
    fn parses_beam_search_flags() {
        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "in.wav",
            "--beam-size",
            "5",
            "--length-penalty",
            "1.2",
            "--no-repeat-ngram",
            "3",
        ]))
        .expect("valid");
        assert_eq!(a.beam_size, 5);
        assert!((a.length_penalty - 1.2).abs() < 1e-6);
        assert_eq!(a.no_repeat_ngram, 3);
    }

    #[test]
    fn rejects_bad_beam_size_and_length_penalty() {
        // --beam-size = 0 is rejected (matches BeamConfig invariant).
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size", "0"]))
                .err()
                .unwrap()
                .contains("beam-size must be >= 1")
        );
        // --beam-size non-integer.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size", "nope"]))
                .err()
                .unwrap()
                .contains("--beam-size")
        );
        // --length-penalty negative.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty", "-1"]))
                .err()
                .unwrap()
                .contains("--length-penalty")
        );
        // --length-penalty NaN.
        assert!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty", "nan"]))
                .err()
                .unwrap()
                .contains("--length-penalty")
        );
        // dangling values.
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--beam-size"]))
                .err()
                .unwrap(),
            "--beam-size requires a value"
        );
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--length-penalty"]))
                .err()
                .unwrap(),
            "--length-penalty requires a value"
        );
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--no-repeat-ngram"]))
                .err()
                .unwrap(),
            "--no-repeat-ngram requires a value"
        );
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

    #[test]
    fn s2s_host_only_smoke_batch_dialog_writes_a_wav() {
        // T20: explicit CPU backend, synthesized-fixture GGUF, explicit
        // fixture tokenizer (opt-in flag) → e2e run + WAV out.
        let model = csm_fixture_gguf("batch");
        let out = std::env::temp_dir().join(format!(
            "vokra-cli-csm-smoke-out-{}.wav",
            std::process::id()
        ));
        let code = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "host only smoke",
            "--fixture-tokenizer",
            "--deterministic",
            "--output",
            out.to_str().unwrap(),
        ]))
        .expect("s2s smoke runs");
        assert_eq!(code, ExitCode::SUCCESS);
        let clip = wav::read_wav(out.to_str().unwrap()).expect("output WAV parses");
        assert!(!clip.samples.is_empty());
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn s2s_streaming_barge_in_demo_stops_after_n_frames() {
        let model = csm_fixture_gguf("interrupt");
        let code = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "interrupt me",
            "--fixture-tokenizer",
            "--deterministic",
            "--interrupt-after",
            "2",
        ]))
        .expect("s2s barge-in demo runs");
        assert_eq!(code, ExitCode::SUCCESS);
        let _ = std::fs::remove_file(&model);
    }

    #[test]
    fn s2s_without_text_is_a_contract_error() {
        let model = csm_fixture_gguf("no-text");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--fixture-tokenizer",
        ]))
        .unwrap_err();
        assert!(err.contains("--text"), "actionable: {err}");
        let _ = std::fs::remove_file(&model);
    }

    #[test]
    fn s2s_gguf_tokenizer_without_fixture_flag_fails_loudly() {
        // Without --fixture-tokenizer the embedded (T29-gated) tokenizer is
        // honest: encode = NotImplemented — never a silent byte fallback.
        let model = csm_fixture_gguf("honest");
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--text",
            "should fail loudly",
        ]))
        .unwrap_err();
        assert!(
            err.contains("not implemented") || err.contains("T29"),
            "{err}"
        );
        let _ = std::fs::remove_file(&model);
    }
}
