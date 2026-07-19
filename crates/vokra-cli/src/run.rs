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
vokra-cli run — load a GGUF and run VAD / ASR / TTS / speaker embedding

USAGE:
    vokra-cli run --model <model.gguf> [--input <in.wav>] [--text <string>] [--output <out.wav>]
                  [--backend cpu|metal|cuda] [--beam-size <N>] [--length-penalty <α>]
                  [--no-repeat-ngram <N>] [--fixture-tokenizer] [--interrupt-after <N>]
                  [--deterministic]
    vokra-cli run --model <campplus.gguf> --input <a.wav> [--compare <b.wav>]

OPTIONS:
    --model <path>              GGUF model file (arch selects VAD / ASR / TTS / S2S /
                                speaker embedding)
    --input <path>              mono WAV input (required for VAD, ASR and speaker;
                                optional recorded context audio for S2S —
                                the explicit AEC bypass path, FR-OP-60)
    --backend <name>            cpu | metal | cuda — backend for the model's hot
                                ops [default cpu]. Mirrors `bench --backend`:
                                honored by the ASR (Whisper) and speaker (CAM++)
                                paths; VAD / TTS / S2S engines run on the CPU
                                regardless (same as bench). metal/cuda need the
                                CLI built with that feature — an unavailable
                                backend fails loudly at inference, never
                                silently on CPU (FR-EX-08).
    --compare <path>            speaker (campplus) only: second WAV; prints the
                                cosine similarity of the two 192-d embeddings
                                (speaker_verify, FR-OP-81)
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
    --duplex                    Moshi only: continuous full-duplex demo —
                                push mic frames from --input, pull model
                                frames, print the inner monologue (M4-06)
    --echo-sim <gain>           Moshi duplex only: mix the previous model
                                frame into the next mic frame at <gain>
                                (0..1) — the synthetic echo path the AEC
                                cancels (T26; without it the session runs
                                the explicit recorded-input AEC opt-out)
    --mimi <path>               Moshi only: standalone Mimi codec GGUF
                                (from `vokra-cli convert --model mimi`) —
                                binds the REAL codec on both duplex ends
                                instead of the synthesized bridge; a bind
                                failure is a hard error (FR-EX-08)
    -h, --help                  print this help
";

/// Parsed `run` arguments.
struct RunArgs {
    model: String,
    input: Option<String>,
    text: Option<String>,
    output: Option<String>,
    /// Backend the model's hot ops run on (mirrors `bench --backend`).
    /// Honored by the ASR (Whisper) and speaker (CAM++) paths; VAD / TTS /
    /// S2S engines run on the CPU regardless (the engine dispatch is not
    /// backend-parameterised for them — same as bench). An unavailable
    /// backend fails loudly at inference (FR-EX-08), never silently on CPU.
    backend: vokra_core::BackendKind,
    /// Speaker (campplus) only: second WAV for the cosine-similarity
    /// comparison. Any other task rejects the flag loudly (FR-EX-08).
    compare: Option<String>,
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
    /// Moshi (M4-06): continuous full-duplex push/pull demo (T26).
    duplex: bool,
    /// Moshi duplex: synthetic echo attenuation — the previous model
    /// frame is mixed into the next mic frame at this gain, exercising
    /// the AEC path end to end (T26 合成 echo 経路).
    echo_sim: Option<f32>,
    /// Moshi only: standalone Mimi codec side-car GGUF — binds the real
    /// codec ends instead of the synthesized bridge (hard error on any
    /// bind failure; rejected loudly on every other arch — FR-EX-08).
    mimi: Option<String>,
}

fn parse_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut output: Option<String> = None;
    let mut backend = vokra_core::BackendKind::Cpu;
    let mut compare: Option<String> = None;
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
    let mut duplex = false;
    let mut echo_sim: Option<f32> = None;
    let mut mimi: Option<String> = None;

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
            "--backend" => {
                let v = args.get(i + 1).ok_or("--backend requires a value")?;
                backend = crate::bench::parse_backend(v)?;
                i += 2;
            }
            "--compare" => {
                compare = Some(args.get(i + 1).ok_or("--compare requires a value")?.clone());
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
            "--duplex" => {
                duplex = true;
                i += 1;
            }
            "--echo-sim" => {
                let v = args.get(i + 1).ok_or("--echo-sim requires a value")?;
                let g: f32 = v
                    .parse()
                    .map_err(|e| format!("--echo-sim must be a float gain: {e}"))?;
                if !g.is_finite() || !(0.0..=1.0).contains(&g) {
                    return Err(format!("--echo-sim gain must be in [0, 1] (got {g})"));
                }
                echo_sim = Some(g);
                i += 2;
            }
            "--mimi" => {
                let v = args.get(i + 1).ok_or("--mimi requires a GGUF path")?;
                mimi = Some(v.clone());
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
        backend,
        compare,
        beam_size,
        length_penalty,
        no_repeat_ngram,
        fixture_tokenizer,
        interrupt_after,
        deterministic,
        duplex,
        echo_sim,
        mimi,
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
        engine::load_session_with_backend_and_mimi(&a.model, a.backend, hint, a.mimi.as_deref())?;

    // `--compare` belongs to the speaker (campplus) task only. Reject it on
    // every other arch rather than silently ignoring the flag (FR-EX-08).
    if a.compare.is_some() && task != ModelTask::Speaker {
        return Err(
            "run: --compare is only supported for the speaker (campplus) arch — \
             it compares two speaker embeddings (FR-OP-81)"
                .to_owned(),
        );
    }

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
        ModelTask::S2sDuplex => {
            run_s2s_duplex(&session, &a)?;
        }
        ModelTask::S2s => {
            run_s2s(&session, &a)?;
        }
        ModelTask::Speaker => {
            run_speaker(&session, &a)?;
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

/// The speaker-embedding demo path (CAM++ / M0-08, FR-OP-81): Kaldi fbank
/// (CAM++ config incl. CMN) → 192-d embedding; prints the L2-norm, and with
/// `--compare <b.wav>` the cosine similarity of the two embeddings via
/// [`vokra_models::speaker::speaker_verify`] (threshold-free: the operating
/// point is the caller's, ADR M4-20 §D-4).
///
/// The encoder binds here from the session's GGUF (the [`Session`] facade has
/// no speaker engine slot) and honors `--backend`: CAM++ dispatches GEMM only,
/// so a GEMM-covering backend (Metal) runs the whole forward on the GPU; an
/// unavailable backend errors at embed time (FR-EX-08).
fn run_speaker(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_models::speaker::{EMBED_DIM, SpeakerEncoder, speaker_verify};
    use vokra_ops::kaldi_fbank::{KaldiFbankOpts, kaldi_fbank};

    let input = a
        .input
        .as_deref()
        .ok_or("run (speaker): --input <a.wav> is required")?;
    let encoder = SpeakerEncoder::from_gguf(session.gguf())
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let opts = KaldiFbankOpts::camplus();

    // wav → fbank → embedding for one clip. The CAM++ fbank recipe is fixed
    // at 16 kHz (`KaldiFbankOpts::camplus`); a mismatched WAV is an explicit
    // error — silently feeding a 44.1/48 kHz clip through a 16 kHz filterbank
    // would produce a garbage embedding with no diagnostic (FR-EX-08).
    let embed_clip = |path: &str| -> Result<[f32; EMBED_DIM], String> {
        let clip = wav::read_wav(path)?;
        if clip.sample_rate != opts.sample_rate {
            return Err(format!(
                "run (speaker): {path}: expected a {} Hz mono WAV (the CAM++ Kaldi-fbank \
                 recipe), got {} Hz — resample offline first",
                opts.sample_rate, clip.sample_rate
            ));
        }
        let (fbank, frames) = kaldi_fbank(&clip.samples, &opts).map_err(|e| e.to_string())?;
        let emb = encoder.embed(&fbank, frames).map_err(|e| e.to_string())?;
        let l2 = emb
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        println!("speaker: {path}: frames={frames} dim={EMBED_DIM} l2_norm={l2:.6}");
        Ok(emb)
    };

    let emb_a = embed_clip(input)?;
    if let Some(compare) = a.compare.as_deref() {
        let emb_b = embed_clip(compare)?;
        let result = speaker_verify(&emb_a, &emb_b, None).map_err(|e| e.to_string())?;
        println!("speaker: cosine_similarity={:.6}", result.similarity);
    }
    Ok(())
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

/// The Moshi full-duplex demo path (M4-06-T26): file-driven mic frames
/// pushed through the facade duplex handle, model frames pulled per push,
/// with an optional synthetic echo path (`--echo-sim <gain>` — the
/// previous model frame mixes into the next mic frame; the session runs
/// its AEC against the pull-stamped reference queue) and the barge-in
/// demo (`--interrupt-after <frames>` flushes via the cross-thread
/// handle). The machine-checkable asserts (T26 (a)〜(d)) run inline:
/// full-length processing without underrun/panic, monotone reference
/// tags (a violating push would error loudly), flush-on-interrupt, and
/// deterministic reproduction under `--deterministic`. 知覚品質 / 実機音響
/// は T30 owner 検収 (合成 echo は近似 — spec 明記の切り離し).
fn run_s2s_duplex(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::DuplexSessionConfig;

    if a.text.is_some() {
        return Err(
            "run (Moshi duplex): --text is not accepted — Moshi GENERATES its own \
             reply (inner monologue); the transcript prints at the end"
                .to_owned(),
        );
    }
    let input_path = a.input.as_deref().ok_or(
        "run (Moshi duplex): --input <user.wav> is required (mic-side audio, mono, \
         at the model sample rate)",
    )?;
    let clip = wav::read_wav(input_path)?;

    if !a.duplex {
        // Batch turn through the facade (the run_s2s analog): push the
        // whole utterance, collect the reply + monologue transcript.
        let mut request = vokra_core::DialogRequest::new("").with_input_audio(clip.samples);
        if a.deterministic {
            request = request.deterministic();
        }
        let turn = session
            .s2s()
            .dialog_request(&request)
            .map_err(|e| e.to_string())?;
        let audio = turn
            .audio
            .ok_or("run (Moshi): the engine returned no audio")?;
        println!(
            "s2s (moshi): {} samples @ {} Hz, monologue: \"{}\" (use --duplex for \
             the continuous push/pull demo)",
            audio.samples.len(),
            audio.sample_rate,
            turn.text
        );
        if let Some(out) = a.output.as_deref() {
            wav::write_wav(out, &audio.samples, audio.sample_rate)?;
            println!("s2s (moshi): wrote -> {out}");
        }
        return Ok(());
    }

    let mut cfg = DuplexSessionConfig::new();
    if a.deterministic {
        cfg = cfg.deterministic();
    }
    if a.echo_sim.is_none() {
        // Recorded-file input with no echo path: the explicit opt-out
        // (never silent — the session records the citable warning).
        cfg = cfg.with_aec_disabled_explicitly();
    }
    let mut handle = session.s2s().duplex_with(&cfg).map_err(|e| e.to_string())?;
    let hop = handle.frame_hop();
    let sr = handle.sample_rate();
    // The synthetic echo arrives one pull→push cycle late; compensate the
    // reference clock accordingly (the T17 playback-offset knob).
    if a.echo_sim.is_some() {
        let mut cfg2 = cfg.clone().with_playback_offset_samples(hop as u64);
        if a.deterministic {
            cfg2 = cfg2.deterministic();
        }
        handle = session
            .s2s()
            .duplex_with(&cfg2)
            .map_err(|e| e.to_string())?;
    }

    let n_frames = clip.samples.len() / hop;
    if n_frames == 0 {
        return Err(format!(
            "run (Moshi duplex): input shorter than one frame ({} samples < hop {hop})",
            clip.samples.len()
        ));
    }
    let interrupt_handle = handle.interrupt_handle();
    let mut mic = vec![0.0f32; hop];
    let mut echo: Vec<f32> = vec![0.0; hop];
    let mut pcm: Vec<f32> = Vec::with_capacity(n_frames * hop);
    let mut emitted = 0usize;
    let mut interrupted_at: Option<usize> = None;
    for f in 0..n_frames {
        mic.copy_from_slice(&clip.samples[f * hop..(f + 1) * hop]);
        if let Some(gain) = a.echo_sim {
            for (m, e) in mic.iter_mut().zip(echo.iter()) {
                *m += gain * *e;
            }
        }
        // (a) the pipeline processes the whole input without underrun.
        handle.push_mic_frame(&mic).map_err(|e| e.to_string())?;
        // (b) pulls stamp monotone reference tags — a violation errors.
        while let Some(frame) = handle.pull_model_frame().map_err(|e| e.to_string())? {
            if a.echo_sim.is_some() {
                echo.copy_from_slice(&frame);
            }
            pcm.extend_from_slice(&frame);
            emitted += 1;
        }
        if let Some(after) = a.interrupt_after {
            if f + 1 == after && interrupted_at.is_none() {
                // (c) cross-thread barge-in: pending output flushes.
                interrupt_handle.interrupt();
                let flushed = handle.pull_model_frame().map_err(|e| e.to_string())?;
                if flushed.is_some() {
                    return Err("duplex barge-in: pending output survived the \
                         interrupt (flush contract violated)"
                        .to_owned());
                }
                interrupted_at = Some(f + 1);
                echo.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
    let text = handle.monologue_text().map_err(|e| e.to_string())?;
    for w in handle.warnings() {
        eprintln!("duplex warning: {w}");
    }
    println!(
        "s2s-duplex: {n_frames} mic frames -> {emitted} model frames ({} samples) @ {sr} Hz{}{}",
        pcm.len(),
        interrupted_at
            .map(|f| format!(", barge-in after {f}"))
            .unwrap_or_default(),
        a.echo_sim
            .map(|g| format!(", echo-sim gain {g} (AEC active)"))
            .unwrap_or_default(),
    );
    println!("s2s-duplex monologue: \"{text}\"");
    if let Some(out) = a.output.as_deref() {
        wav::write_wav(out, &pcm, sr)?;
        println!("s2s-duplex: wrote {} samples @ {sr} Hz -> {out}", pcm.len());
    }
    Ok(())
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

    /// `--mimi <path>` parses into `RunArgs::mimi` (Moshi real-codec
    /// side-car); a bare `--mimi` is a loud parse error.
    #[test]
    fn parse_accepts_mimi_sidecar_path() {
        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "mic.wav",
            "--duplex",
            "--mimi",
            "codec.gguf",
        ]))
        .expect("parses");
        assert_eq!(a.mimi.as_deref(), Some("codec.gguf"));
        assert!(a.duplex);
        let err = match parse_args(&args(&["--model", "m.gguf", "--mimi"])) {
            Err(e) => e,
            Ok(_) => panic!("bare --mimi must be rejected"),
        };
        assert!(err.contains("--mimi requires a GGUF path"), "got: {err}");
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

    // ---- --backend (bench-surface mirror) + --compare (speaker) ----------

    /// `--backend` parses exactly like `bench --backend` (shared
    /// `parse_backend`): default cpu, metal/cuda/vulkan accepted at parse
    /// time (availability is an inference-time explicit error, FR-EX-08).
    #[test]
    fn parses_backend_flag_with_cpu_default() {
        use vokra_core::BackendKind;
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.backend, BackendKind::Cpu);
        for (name, kind) in [
            ("cpu", BackendKind::Cpu),
            ("metal", BackendKind::Metal),
            ("cuda", BackendKind::Cuda),
            ("vulkan", BackendKind::Vulkan),
        ] {
            let a = parse_args(&args(&["--model", "m.gguf", "--backend", name]))
                .unwrap_or_else(|e| panic!("--backend {name} should parse: {e}"));
            assert_eq!(a.backend, kind, "--backend {name}");
        }
    }

    #[test]
    fn rejects_unknown_backend_and_dangling_backend() {
        let err = parse_args(&args(&["--model", "m.gguf", "--backend", "npu"]))
            .err()
            .unwrap();
        assert!(err.contains("unknown --backend"), "got: {err}");
        assert_eq!(
            parse_args(&args(&["--model", "m.gguf", "--backend"]))
                .err()
                .unwrap(),
            "--backend requires a value"
        );
    }

    #[test]
    fn parses_compare_flag_and_rejects_dangling_compare() {
        let a = parse_args(&args(&[
            "--model",
            "spk.gguf",
            "--input",
            "a.wav",
            "--compare",
            "b.wav",
        ]))
        .expect("valid");
        assert_eq!(a.compare.as_deref(), Some("b.wav"));
        // Default: no compare.
        let a = parse_args(&args(&["--model", "spk.gguf"])).expect("valid");
        assert_eq!(a.compare, None);
        assert_eq!(
            parse_args(&args(&["--model", "spk.gguf", "--compare"]))
                .err()
                .unwrap(),
            "--compare requires a value"
        );
    }

    /// The help text documents the new surface (Fix A + Fix C of the
    /// campaign-2 cli-enablers leg).
    #[test]
    fn help_text_documents_backend_compare_and_speaker() {
        assert!(USAGE.contains("--backend"), "USAGE lists --backend");
        assert!(
            USAGE.contains("cpu | metal | cuda"),
            "USAGE lists the backend names"
        );
        assert!(USAGE.contains("--compare"), "USAGE lists --compare");
        assert!(USAGE.contains("speaker"), "USAGE mentions the speaker task");
        assert!(USAGE.contains("campplus"), "USAGE names the campplus arch");
    }

    /// `--compare` on a non-speaker arch is an explicit contract error
    /// (FR-EX-08: never silently ignore a user flag).
    #[test]
    fn compare_on_non_speaker_arch_is_rejected() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--compare",
            "b.wav",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--compare is only supported for the speaker"),
            "got: {err}"
        );
    }

    /// A campplus-arch GGUF whose tensors do not bind fails loudly at the
    /// encoder bind inside the Speaker arm (the engine dispatch itself
    /// returns a bare session — see `engine::tests`).
    #[test]
    fn speaker_metadata_only_gguf_fails_loudly_at_encoder_bind() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let dir = std::env::temp_dir();
        let model = dir.join(format!("vokra-cli-spk-meta-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let in_wav = dir.join(format!("vokra-cli-spk-meta-{}.wav", std::process::id()));
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 16_000).unwrap();
        let err = main(&args(&[
            "--model",
            model.to_str().unwrap(),
            "--input",
            in_wav.to_str().unwrap(),
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&model);
        let _ = std::fs::remove_file(&in_wav);
        // The bind error names the missing tensor / weight, not a panic.
        assert!(!err.is_empty(), "loud bind error expected");
    }

    /// Speaker task without `--input` is a contract error.
    #[test]
    fn speaker_without_input_is_a_contract_error() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-spk-noinput-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(err.contains("--input"), "actionable: {err}");
    }

    /// Real-GGUF gated e2e (mirrors the `speaker::parity` gating): set
    /// `VOKRA_CAMPLUS_GGUF` to a converted CAM++ GGUF to run; skips clean
    /// when unset (CI stays green, no fabricated pass).
    #[test]
    fn speaker_real_gguf_e2e_identical_inputs_gated() {
        let Ok(model) = std::env::var("VOKRA_CAMPLUS_GGUF") else {
            eprintln!("skipping speaker CLI e2e: set VOKRA_CAMPLUS_GGUF to run");
            return;
        };
        let dir = std::env::temp_dir();
        let in_wav = dir.join(format!("vokra-cli-spk-e2e-{}.wav", std::process::id()));
        // 1 s of deterministic pseudo-speech at 16 kHz (multi-tone, enough
        // frames for the CAM++ receptive field).
        let samples: Vec<f32> = (0..16_000)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.3 * (t * std::f32::consts::TAU * 220.0).sin()
                    + 0.2 * (t * std::f32::consts::TAU * 660.0).sin()
            })
            .collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 16_000).unwrap();
        // Identical inputs → the run succeeds (cosine 1.0 prints to stdout;
        // the numeric assertion rides the campaign harness, which captures
        // stdout of the release binary).
        let code = main(&args(&[
            "--model",
            &model,
            "--input",
            in_wav.to_str().unwrap(),
            "--compare",
            in_wav.to_str().unwrap(),
        ]))
        .expect("speaker e2e runs");
        assert_eq!(code, ExitCode::SUCCESS);
        // A non-16 kHz clip is an explicit error (no silent resample).
        let wav8k = dir.join(format!("vokra-cli-spk-e2e8k-{}.wav", std::process::id()));
        wav::write_wav(wav8k.to_str().unwrap(), &samples[..8000], 8_000).unwrap();
        let err = main(&args(&[
            "--model",
            &model,
            "--input",
            wav8k.to_str().unwrap(),
        ]))
        .unwrap_err();
        assert!(
            err.contains("16000 Hz") || err.contains("16 kHz"),
            "got: {err}"
        );
        let _ = std::fs::remove_file(&in_wav);
        let _ = std::fs::remove_file(&wav8k);
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

    /// M4-06-T26: the full duplex demo path over a converted synthetic
    /// checkpoint — echo-sim (AEC active), barge-in flush, deterministic
    /// reproduction, and the attribution banner side of the load (the
    /// engine dispatch prints it; here we assert the session carries the
    /// resolved info).
    #[test]
    fn moshi_duplex_demo_smoke_with_echo_sim_and_barge_in() {
        let dir = std::env::temp_dir().join(format!("vokra-cli-moshi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        let in_wav = dir.join("user.wav");
        let out_wav = dir.join("model.wav");
        std::fs::write(&ckpt, moshi_fixture::synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, moshi_fixture::spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");

        // 4 frames of pseudo-speech at the converted (real-constant) rates.
        let (session, task) =
            engine::load_session(gguf.to_str().unwrap()).expect("moshi session loads");
        assert_eq!(task, ModelTask::S2sDuplex);
        assert!(
            session.attribution().is_some(),
            "FR-MD-09: the loader resolves the attribution surface"
        );
        let hop = 1920usize; // 24 kHz / 12.5 Hz (converter constants)
        let samples: Vec<f32> = (0..hop * 4)
            .map(|i| ((i as f32) * 0.01).sin() * 0.2)
            .collect();
        wav::write_wav(in_wav.to_str().unwrap(), &samples, 24_000).unwrap();

        let run_once = || {
            let args: Vec<String> = [
                "--model",
                gguf.to_str().unwrap(),
                "--input",
                in_wav.to_str().unwrap(),
                "--output",
                out_wav.to_str().unwrap(),
                "--duplex",
                "--echo-sim",
                "0.5",
                "--interrupt-after",
                "2",
                "--deterministic",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            main(&args).expect("duplex demo runs");
            wav::read_wav(out_wav.to_str().unwrap()).expect("output wav")
        };
        let a = run_once();
        let b = run_once();
        assert!(!a.samples.is_empty(), "model frames were pulled");
        assert_eq!(a.samples, b.samples, "T26 (d): deterministic reproduction");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Moshi duplex argument contract: --text is rejected (the model
    /// generates its reply), --input is required.
    #[test]
    fn moshi_duplex_rejects_text_and_requires_input() {
        let dir = std::env::temp_dir().join(format!("vokra-cli-moshi-neg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        std::fs::write(&ckpt, moshi_fixture::synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, moshi_fixture::spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");
        // A tokenizer-less conversion fails loudly at LOAD (monologue
        // decode is load-bearing) — pin that posture too.
        let bare = dir.join("bare.gguf");
        vokra_convert::convert_moshi_file(&ckpt, None, &bare).expect("convert bare");
        let err = engine::load_session(bare.to_str().unwrap()).unwrap_err();
        assert!(err.contains("vokra.tokenizer.model"), "loud: {err}");
        let base = ["--model".to_string(), gguf.to_str().unwrap().to_string()];
        let mut with_text: Vec<String> = base.to_vec();
        with_text.extend(["--text".into(), "scripted".into()]);
        let err = main(&with_text).unwrap_err();
        assert!(err.contains("GENERATES"), "contract: {err}");
        let err = main(&base).unwrap_err();
        assert!(
            err.contains("--input") || err.contains("--duplex") || err.contains("required"),
            "actionable: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Shared synthetic Moshi checkpoint fixtures (the converter/e2e
    /// wire shapes — MoshiConfig::tiny_for_tests).
    mod moshi_fixture {
        pub(super) fn spm_blob(n: usize) -> Vec<u8> {
            fn varint(mut v: u64, out: &mut Vec<u8>) {
                loop {
                    let mut b = (v & 0x7f) as u8;
                    v >>= 7;
                    if v != 0 {
                        b |= 0x80;
                    }
                    out.push(b);
                    if v == 0 {
                        break;
                    }
                }
            }
            let mut blob = Vec::new();
            for i in 0..n {
                let piece = format!("\u{2581}p{i}");
                let mut msg = Vec::new();
                msg.push(0x0a);
                varint(piece.len() as u64, &mut msg);
                msg.extend_from_slice(piece.as_bytes());
                msg.push(0x18);
                msg.push(0x01);
                blob.push(0x0a);
                varint(msg.len() as u64, &mut blob);
                blob.extend_from_slice(&msg);
            }
            blob
        }

        pub(super) fn synthetic_checkpoint() -> Vec<u8> {
            let mut entries: Vec<(String, Vec<u64>)> = Vec::new();
            let (d, text, card) = (16u64, 13u64, 9u64);
            let (h_tm, d_dt, h_dt) = (8u64, 8u64, 6u64);
            entries.push(("text_emb.weight".into(), vec![text + 1, d]));
            entries.push(("text_linear.weight".into(), vec![text, d]));
            entries.push(("out_norm.alpha".into(), vec![1, 1, d]));
            for k in 0..4 {
                entries.push((format!("emb.{k}.weight"), vec![card + 1, d]));
            }
            for i in 0..2 {
                let p = format!("transformer.layers.{i}");
                entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d]));
                entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d]));
                entries.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * d, d]));
                entries.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d]));
                entries.push((format!("{p}.gating.linear_in.weight"), vec![2 * h_tm, d]));
                entries.push((format!("{p}.gating.linear_out.weight"), vec![d, h_tm]));
            }
            for cb in 0..2 {
                entries.push((format!("depformer_in.{cb}.weight"), vec![d_dt, d]));
                entries.push((format!("linears.{cb}.weight"), vec![card, d_dt]));
            }
            entries.push(("depformer_text_emb.weight".into(), vec![text + 1, d_dt]));
            entries.push(("depformer_emb.0.weight".into(), vec![card + 1, d_dt]));
            for i in 0..2 {
                let p = format!("depformer.layers.{i}");
                entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d_dt]));
                entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d_dt]));
                entries.push((
                    format!("{p}.self_attn.in_proj_weight"),
                    vec![2 * 3 * d_dt, d_dt],
                ));
                entries.push((
                    format!("{p}.self_attn.out_proj.weight"),
                    vec![2 * d_dt, d_dt],
                ));
                for s in 0..2 {
                    entries.push((
                        format!("{p}.gating.{s}.linear_in.weight"),
                        vec![2 * h_dt, d_dt],
                    ));
                    entries.push((
                        format!("{p}.gating.{s}.linear_out.weight"),
                        vec![d_dt, h_dt],
                    ));
                }
            }
            let mut header = String::from("{");
            let mut data: Vec<u8> = Vec::new();
            let mut lcg = 0x9876_5432u32;
            for (i, (name, shape)) in entries.iter().enumerate() {
                let n: u64 = shape.iter().product();
                let start = data.len();
                for _ in 0..n {
                    lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                    let frac = (lcg >> 16) as u16 & 0x007F;
                    let sign = ((lcg >> 8) as u16) & 0x8000;
                    data.extend_from_slice(&(sign | 0x3E00 | frac).to_le_bytes());
                }
                let end = data.len();
                if i > 0 {
                    header.push(',');
                }
                header.push_str(&format!(
                    "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
                    shape
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            header.push('}');
            let mut blob = Vec::new();
            blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
            blob.extend_from_slice(header.as_bytes());
            blob.extend_from_slice(&data);
            blob
        }
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
