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
    vokra-cli run --model <whisper.gguf> --input <in.wav> --word-timestamps
    vokra-cli run --model <voxtral.gguf> --input <in.wav> [--language <code>] [--bare-prompt]
    vokra-cli run --model <campplus.gguf> --input <a.wav> [--compare <b.wav>]
    vokra-cli run --model <kokoro.gguf> --text <phonemes> --style <s.f32> [--output <out.wav>]

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
                                not generate text). For `kokoro` this is
                                PHONEME content, not graphemes: either a misaki
                                IPA string (each char looked up in the GGUF's
                                vokra.kokoro.phoneme_symbols table) or the
                                piper raw-id form (1 2 3 / 1,2,3 — content
                                ids only, sentinels are added). Kokoro has no
                                G2P bridge in-tree, so unmappable input is an
                                error rather than a silent drop.
    --voice <name>              kokoro only: voice name from the GGUF's
                                vokra.kokoro.voice_names. The name resolves,
                                but mapping it to a style row is NOT
                                implemented yet (M2-07-T02), so this cannot
                                synthesize on any GGUF — use --style.
    --style <path>              kokoro only: raw little-endian f32 style
                                vector, style_dim or 2*style_dim floats. The
                                2*style_dim form is upstream's full ref_s row
                                ([:style_dim] conditions the decoder,
                                [style_dim:] the prosody predictor). Takes
                                precedence over --voice.
    --length-scale <s>          kokoro only: duration multiplier (reciprocal
                                of upstream `speed`) [default 1.0]
    --output <path>             WAV file for the TTS / S2S output (optional)
    --beam-size <N>             ASR beam-search width (default 1 = greedy).
                                Honored for `voxtral` (n-best beam) and, with
                                --word-timestamps, for `whisper`. An arch whose
                                dispatch does not honor it errors out rather
                                than silently ignoring the flag (FR-EX-08).
    --length-penalty <α>        GNMT length-penalty exponent for beam search
                                (default 0.6). See `voxtral::BeamConfig`.
    --no-repeat-ngram <N>       Block repeated n-grams of length N during
                                beam search (default 0 = disabled).
    --word-timestamps           whisper only: emit per-word start/end times
                                (cross-attention DTW alignment, M4-20) after
                                the transcript, as `word<TAB>start<TAB>end`
                                lines. Requires the GGUF to carry
                                `vokra.whisper.alignment_heads`; a model
                                without them is an explicit error, never a
                                silent empty list.
    --language <code>           voxtral only: transcription language for the
                                trained prompt's `lang:<code>` segment
                                (lowercase ISO 639, default `en`). Pass
                                `auto` to omit the segment and let the model
                                infer the language.
    --bare-prompt               voxtral only: decode from the bare
                                soft-prefix + BOS layout instead of the
                                trained transcription prompt. Honest LM
                                continuation conditioned on the audio — NOT
                                a transcript (see AsrPromptLayout).
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
    /// Whisper only (cc-19): emit per-word timestamps after the transcript.
    /// Any other arch rejects the flag loudly (FR-EX-08).
    word_timestamps: bool,
    /// Voxtral only: the raw `--language` value. `None` = flag absent (keep
    /// the engine default, `en`); `Some("auto")` = omit the `lang:` segment
    /// entirely; `Some(code)` = that code.
    language: Option<String>,
    /// Voxtral only: opt into the bare soft-prefix + BOS layout.
    bare_prompt: bool,
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
    /// Kokoro only (cc-24): voice name from `vokra.kokoro.voice_names`.
    /// Rejected loudly on every other arch (FR-EX-08).
    voice: Option<String>,
    /// Kokoro only (cc-24): path to a raw little-endian f32 style vector
    /// (`style_dim` or `2·style_dim` floats). Takes precedence over
    /// `--voice`, matching `KokoroTts::synthesize_phonemes`.
    style: Option<String>,
    /// Kokoro only (cc-24): duration multiplier, the reciprocal of
    /// upstream's `speed`. Defaults to 1.0 = upstream default.
    length_scale: f32,
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
    let mut word_timestamps = false;
    let mut language: Option<String> = None;
    let mut bare_prompt = false;
    let mut fixture_tokenizer = false;
    let mut interrupt_after: Option<usize> = None;
    let mut deterministic = false;
    let mut duplex = false;
    let mut echo_sim: Option<f32> = None;
    let mut mimi: Option<String> = None;
    let mut voice: Option<String> = None;
    let mut style: Option<String> = None;
    let mut length_scale: f32 = 1.0;

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
            "--word-timestamps" => {
                word_timestamps = true;
                i += 1;
            }
            "--language" => {
                let v = args.get(i + 1).ok_or("--language requires a value")?;
                if v.is_empty() {
                    return Err("--language must not be empty (use `auto` to omit the \
                                prompt's lang: segment)"
                        .to_owned());
                }
                language = Some(v.clone());
                i += 2;
            }
            "--bare-prompt" => {
                bare_prompt = true;
                i += 1;
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
            "--voice" => {
                let v = args.get(i + 1).ok_or("--voice requires a name")?;
                if v.is_empty() {
                    return Err("--voice must not be empty".to_owned());
                }
                voice = Some(v.clone());
                i += 2;
            }
            "--style" => {
                let v = args.get(i + 1).ok_or("--style requires a path")?;
                style = Some(v.clone());
                i += 2;
            }
            "--length-scale" => {
                let v = args.get(i + 1).ok_or("--length-scale requires a value")?;
                length_scale = v
                    .parse()
                    .map_err(|e| format!("--length-scale must be a float: {e}"))?;
                if !length_scale.is_finite() || length_scale <= 0.0 {
                    return Err(format!(
                        "--length-scale must be a positive finite float (got {length_scale})"
                    ));
                }
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
        word_timestamps,
        language,
        bare_prompt,
        fixture_tokenizer,
        interrupt_after,
        deterministic,
        duplex,
        echo_sim,
        mimi,
        voice,
        style,
        length_scale,
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
    // `--word-timestamps` is a Whisper-only surface (cross-attention DTW,
    // M4-20); `--language` / `--bare-prompt` are Voxtral prompt knobs. Each
    // is rejected off its own arch rather than silently ignored (FR-EX-08).
    if a.word_timestamps && task != ModelTask::Asr {
        return Err(
            "run: --word-timestamps is only supported for the whisper arch — it needs the \
             cross-attention alignment heads (M4-20). Voxtral has no such alignment."
                .to_owned(),
        );
    }
    if (a.language.is_some() || a.bare_prompt) && task != ModelTask::AsrVoxtral {
        return Err(
            "run: --language / --bare-prompt are only supported for the voxtral arch — they \
             select the trained transcription prompt's `lang:` segment and layout"
                .to_owned(),
        );
    }
    // `--voice` / `--style` / `--length-scale` are Kokoro style-conditioning
    // knobs (cc-24). Rejected off that arch rather than silently ignored
    // (FR-EX-08) — a dropped style would change the speaker without saying so.
    if (a.voice.is_some() || a.style.is_some()) && task != ModelTask::TtsKokoro {
        return Err(
            "run: --voice / --style are only supported for the kokoro arch — they select the \
             style vector that conditions its decoder and prosody predictor"
                .to_owned(),
        );
    }
    if a.length_scale != 1.0 && task != ModelTask::TtsKokoro {
        return Err(
            "run: --length-scale is only supported for the kokoro arch (piper-plus exposes its \
             own scales through the engine API, not the CLI)"
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
            // Whisper's beam-search entry point lives on the concrete
            // `WhisperAsr` (n-best + alignment), not on the shared
            // `AsrEngine` trait the session injects. `--word-timestamps`
            // therefore routes through the beam surface (cc-19); without
            // it, a beam-only flag on the Whisper path would be silently
            // dropped, so it is a hard error instead (FR-EX-08).
            if a.word_timestamps {
                run_whisper_word_timestamps(&a.model, a.backend, &clip.samples, &a)?;
                return Ok(ExitCode::SUCCESS);
            }
            if a.beam_size > 1 || a.no_repeat_ngram > 0 {
                return Err(
                    "run (ASR): --beam-size > 1 / --no-repeat-ngram are only honored on the \
                     whisper path together with --word-timestamps (which routes through the \
                     beam/alignment surface), or on the `voxtral` arch. Add \
                     --word-timestamps, or run a voxtral GGUF."
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
        ModelTask::AsrVoxtral => {
            run_voxtral(&session, &a)?;
        }
        ModelTask::Tts => {
            let text = a
                .text
                .as_deref()
                .ok_or("run (TTS): --text <string> is required")?;
            let audio = session.tts().synthesize(text).map_err(|e| e.to_string())?;
            emit_audio(
                "tts",
                &audio.samples,
                audio.sample_rate,
                a.output.as_deref(),
            )?;
        }
        ModelTask::TtsKokoro => {
            run_kokoro(&a)?;
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

/// Writes synthesized PCM to `--output`, or reports its duration when the flag
/// is absent. Shared by the piper-plus and Kokoro TTS arms so both report the
/// same way. `label` prefixes the line (`tts` / `kokoro`).
fn emit_audio(
    label: &str,
    samples: &[f32],
    sample_rate: u32,
    output: Option<&str>,
) -> Result<(), String> {
    match output {
        Some(out) => {
            wav::write_wav(out, samples, sample_rate)?;
            println!(
                "{label}: wrote {} samples @ {sample_rate} Hz -> {out}",
                samples.len()
            );
        }
        None => {
            let secs = samples.len() as f64 / f64::from(sample_rate);
            println!(
                "{label}: {} samples, {secs:.3}s @ {sample_rate} Hz (no --output; audio discarded)",
                samples.len()
            );
        }
    }
    Ok(())
}

/// Maps `--text` to Kokoro phoneme ids, wrapped in the id-0 sentinels
/// upstream's tokenizer adds (`input_ids = [0, *content, 0]`, `kokoro==0.9.4`
/// `pipeline.py`).
///
/// Two input forms, mirroring `PassthroughPhonemizer`'s content/framing split
/// on the piper-plus side (`vokra-piper-plus::phonemizer`):
///
/// - **symbol form** (`"həlˈO wˈɜːld"`) — each `char` is looked up in the
///   GGUF's `vokra.kokoro.phoneme_symbols` table (index = id). Every symbol in
///   the shipped 178-entry table is a single `char`, so a per-`char` lookup is
///   exact.
/// - **raw-id form** (`"1 2 3"` or `"1,2,3"`) — the piper raw-id syntax,
///   whitespace- or comma-separated. Reproduces an exact upstream tokenization
///   (e.g. replaying a parity dump) without routing IPA through a shell. Ids
///   are **content only**; the sentinels are added here, as in piper's
///   `parse_content` / `phonemize` split.
///
/// # Disambiguation is verified, not assumed
///
/// The raw-id form is selected when every token is ASCII digits. That is only
/// unambiguous while no phoneme symbol is itself a digit — true of the shipped
/// misaki table, but checked against the actual table at run time rather than
/// trusted, so a future table that adds a digit symbol is a loud error instead
/// of a silent misreading of the caller's input.
///
/// # Unmappable input is an error
///
/// Both upstream Kokoro and this crate's `PiperPlusTts::tokenize` silently drop
/// symbols they cannot map. This route does not: dropping a phoneme changes the
/// utterance with no signal to the caller, which is the silent-fallback shape
/// FR-EX-08 forbids. The message names every offending character so a caller
/// can see whether they passed graphemes by mistake — the most likely error,
/// since there is no G2P bridge in-tree to convert them.
pub(crate) fn kokoro_phoneme_ids(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    let content = if is_id_sequence(text) {
        kokoro_content_from_ids(text, symbols)?
    } else {
        kokoro_content_from_symbols(text, symbols)?
    };
    if content.is_empty() {
        return Err("run (kokoro): --text produced no phonemes".to_owned());
    }
    // Upstream wraps the content in id 0. Index 0 of the table is the
    // empty/pad entry, so the sentinels are pushed positionally rather than
    // looked up.
    let mut ids = Vec::with_capacity(content.len() + 2);
    ids.push(0);
    ids.extend_from_slice(&content);
    ids.push(0);
    Ok(ids)
}

/// Whether `text` is the piper raw-id form: at least one token, every token
/// non-empty ASCII digits, split on whitespace or `,`.
fn is_id_sequence(text: &str) -> bool {
    let mut any = false;
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        any = true;
        if !tok.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    any
}

/// Parses the raw-id form into **content** ids (no sentinels).
fn kokoro_content_from_ids(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    // The digit heuristic is only sound while no symbol is a bare digit; verify
    // against this GGUF's actual table instead of assuming (FR-EX-08).
    let digit_symbols: Vec<&String> = symbols
        .iter()
        .filter(|s| s.len() == 1 && s.as_bytes()[0].is_ascii_digit())
        .collect();
    if !digit_symbols.is_empty() {
        return Err(format!(
            "run (kokoro): --text looks like a raw id sequence, but this voice's \
             phoneme_symbols table contains digit symbol(s) {digit_symbols:?}, so the \
             raw-id and symbol forms are ambiguous for this model — the input cannot be \
             interpreted without guessing (FR-EX-08)"
        ));
    }
    let mut ids = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let id: i64 = tok
            .parse()
            .map_err(|_| format!("run (kokoro): `{tok}` is not a phoneme id"))?;
        // Bound against the real table so an out-of-range id fails here rather
        // than indexing past the embedding rows downstream.
        if id <= 0 || id as usize >= symbols.len() {
            return Err(format!(
                "run (kokoro): phoneme id {id} out of range — --text takes CONTENT ids in \
                 1..{} (id 0 is the pad sentinel and is added automatically, as in the \
                 piper raw-id path)",
                symbols.len()
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Parses the symbol form into **content** ids (no sentinels).
fn kokoro_content_from_symbols(text: &str, symbols: &[String]) -> Result<Vec<i64>, String> {
    let mut ids = Vec::with_capacity(text.chars().count());
    let mut unknown: Vec<char> = Vec::new();
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        let needle = ch.encode_utf8(&mut buf);
        match symbols.iter().position(|s| s == needle) {
            // Index 0 is the pad sentinel; a table whose entry 0 is empty can
            // never match here, but guard anyway so a fixture that puts a real
            // symbol at 0 cannot inject an extra sentinel mid-sequence.
            Some(0) | None => unknown.push(ch),
            Some(id) => ids.push(id as i64),
        }
    }
    if !unknown.is_empty() {
        unknown.sort_unstable();
        unknown.dedup();
        return Err(format!(
            "run (kokoro): --text contains {} character(s) absent from this voice's \
             vokra.kokoro.phoneme_symbols table: {unknown:?}. --text takes misaki IPA \
             PHONEMES, not graphemes (there is no G2P bridge in-tree); dropping them \
             silently would change the utterance (FR-EX-08)",
            unknown.len()
        ));
    }
    Ok(ids)
}

/// Reads a raw little-endian f32 style vector from `path`.
///
/// The file must be a whole number of f32s and match either `style_dim` or
/// `2·style_dim` — the two lengths `KokoroTts::synthesize_phonemes` accepts.
/// Both checks happen here so a truncated dump is named as such rather than
/// surfacing as a shape error from inside the prosody predictor.
pub(crate) fn read_style_vector(path: &str, style_dim: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("--style {path}: {e}"))?;
    // `%` rather than `usize::is_multiple_of`: this crate inherits the
    // workspace MSRV (1.85) and that method is stable only since 1.87.
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "--style {path}: {} bytes is not a whole number of f32s",
            bytes.len()
        ));
    }
    let n = bytes.len() / 4;
    if n != style_dim && n != 2 * style_dim {
        return Err(format!(
            "--style {path}: {n} floats — expected style_dim ({style_dim}) or 2*style_dim \
             ({}) for a full upstream ref_s row",
            2 * style_dim
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The Kokoro-82M synthesis path (cc-24).
///
/// `--text` is a misaki phoneme string (see [`kokoro_phoneme_ids`]); the style
/// comes from `--style` (a raw f32 dump of an upstream voicepack row) or
/// `--voice` (a name from the GGUF's table).
///
/// # Why the engine is rebuilt here
///
/// The reachable synthesis surface is the concrete
/// [`vokra_models::kokoro::KokoroTts::synthesize_phonemes`], not the
/// [`vokra_core::TtsEngine`] trait — Kokoro's `synthesize` is a hard
/// `NotImplemented` pending a misaki G2P bridge. The arch dispatch therefore
/// hands back a bare session and the concrete engine binds once, here, from the
/// model path (the `ModelTask::Speaker` / `ModelTask::AsrVoxtral` pattern).
fn run_kokoro(a: &RunArgs) -> Result<(), String> {
    use vokra_models::kokoro::KokoroTts;

    let text = a
        .text
        .as_deref()
        .ok_or("run (kokoro): --text <phonemes> is required")?;
    let tts = KokoroTts::from_path(&a.model)
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    let config = tts.config();
    let ids = kokoro_phoneme_ids(text, &config.phoneme_symbols)?;

    // Style resolution mirrors `synthesize_phonemes`: an explicit override wins
    // over a name. Neither present is an error — there is no neutral default
    // style, and silently substituting zeros would synthesize in a voice the
    // caller never asked for (FR-EX-08).
    let style = match a.style.as_deref() {
        Some(path) => Some(read_style_vector(path, config.style_dim)?),
        None => None,
    };
    if style.is_none() && a.voice.is_none() {
        return Err(format!(
            "run (kokoro): a style is required — pass --style <f32 dump> (style_dim {} or \
             2*style_dim {}), or --voice <name> from {:?}",
            config.style_dim,
            2 * config.style_dim,
            config.voice_names,
        ));
    }

    let audio = tts
        .synthesize_phonemes(
            &ids,
            a.voice.as_deref(),
            style.as_deref(),
            0.0,
            a.length_scale,
        )
        .map_err(|e| {
            // `--voice` hits a hard `NotImplemented` in the model layer
            // (`synthesize_phonemes` — the voice → style-row lookup is
            // M2-07-T02). Append the actionable workaround rather than just
            // propagating the bare "not implemented".
            if a.voice.is_some() && style.is_none() {
                format!(
                    "{e}\nnote: the voice name resolves against \
                     `vokra.kokoro.voice_names`, but mapping it to a style row is not \
                     implemented yet, so `--voice` cannot synthesize on ANY Kokoro GGUF \
                     — including one whose voicepack rows were stacked in at conversion \
                     time (`tools/parity/kokoro_prepare_checkpoint.py --stack-voicepack`, \
                     off by default; upstream ships the rows as separate `voices/*.pt`). \
                     Until then use --style: export the row upstream would have picked, \
                     `voicepack[len(phonemes) - 1]` ({} f32, little-endian), and pass \
                     the file.",
                    2 * config.style_dim
                )
            } else {
                e.to_string()
            }
        })?;

    println!(
        "kokoro: {} phoneme ids (incl. 2 sentinels), style {}",
        ids.len(),
        match (&style, a.voice.as_deref()) {
            (Some(s), _) => format!("override ({} f32)", s.len()),
            (None, Some(v)) => format!("voice `{v}`"),
            (None, None) => unreachable!("checked above"),
        }
    );
    emit_audio(
        "kokoro",
        &audio.samples,
        audio.sample_rate,
        a.output.as_deref(),
    )
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

/// The Voxtral ASR path (P2 cc-10). Binds the concrete
/// [`vokra_models::voxtral::VoxtralAsr`] from the session's (mmap-backed)
/// GGUF exactly once — see [`ModelTask::AsrVoxtral`] for why the engine is
/// not injected by the dispatch — then greedy- or beam-decodes.
///
/// Prompt layout: the trained transcription wrapper by default (built at
/// runtime from the GGUF's embedded tekken vocab), with `--language` picking
/// the `lang:<code>` segment (`auto` omits it) and `--bare-prompt` opting
/// into the honest LM-continuation layout.
fn run_voxtral(session: &Session, a: &RunArgs) -> Result<(), String> {
    use vokra_core::AsrEngine;
    use vokra_models::voxtral::{AsrPromptLayout, VoxtralAsr};

    let path = a
        .input
        .as_deref()
        .ok_or("run (voxtral ASR): --input <in.wav> is required")?;
    let clip = wav::read_wav(path)?;

    let mut asr = VoxtralAsr::from_gguf(session.gguf())
        .map_err(|e| e.to_string())?
        .with_backend(a.backend);
    if a.bare_prompt {
        asr = asr.with_prompt_layout(AsrPromptLayout::BareSoftPrefix);
    }
    // `--language auto` = omit the prompt's `lang:` segment (upstream
    // `TranscriptionRequest.language = None`); an absent flag keeps the
    // engine default (`en`).
    match a.language.as_deref() {
        None => {}
        Some("auto") => asr = asr.with_language(None),
        Some(code) => asr = asr.with_language(Some(code.to_owned())),
    }

    if a.beam_size > 1 {
        let beams = asr
            .transcribe_beam_with_config_overrides(
                &clip.samples,
                a.beam_size,
                a.length_penalty,
                a.no_repeat_ngram,
                /*max_new_tokens*/ 0,
            )
            .map_err(|e| e.to_string())?;
        let best = beams
            .first()
            .ok_or("run (voxtral ASR): beam search produced no hypothesis")?;
        println!("asr: {}", best.text);
        for (i, b) in beams.iter().enumerate() {
            println!(
                "asr-alt[{i}]: score={:.4} logp={:.4} {}",
                b.result.length_normalized_score, b.result.log_prob, b.text
            );
        }
        return Ok(());
    }
    // `--no-repeat-ngram` only bites inside beam search; saying so beats
    // silently dropping it (FR-EX-08 spirit — the flag parses, so the user
    // gets a diagnostic rather than a wrong assumption).
    if a.no_repeat_ngram > 0 {
        eprintln!(
            "run (voxtral ASR): note — --no-repeat-ngram applies to beam search only \
             (greedy ignores it); pass --beam-size > 1 to use it."
        );
    }
    let text = asr
        .transcribe(&clip.samples)
        .map_err(|e| e.to_string())?
        .text;
    println!("asr: {text}");
    Ok(())
}

/// Whisper `--word-timestamps` (cc-19 CLI half): routes through the concrete
/// [`vokra_models::whisper::WhisperAsr`] beam/alignment surface (word
/// timestamps come from cross-attention DTW over the best hypothesis, M4-20
/// — the greedy `AsrEngine::transcribe` produces no alignment), prints the
/// transcript then one `word<TAB>start<TAB>end` line per word.
///
/// A GGUF without `vokra.whisper.alignment_heads` is an explicit error
/// raised inside `beam_search` — never an empty word list (FR-EX-08).
fn run_whisper_word_timestamps(
    model_path: &str,
    backend: vokra_core::BackendKind,
    pcm: &[f32],
    a: &RunArgs,
) -> Result<(), String> {
    use vokra_core::decode::BeamSearchConfig;
    use vokra_models::whisper::WhisperAsr;

    // Re-open the GGUF for the concrete engine: `Session` lends its file by
    // reference and `WhisperAsr::from_gguf` takes one, so this binds against
    // the same mmap-backed parse the dispatch already validated.
    let gguf = vokra_mmap::open_gguf(model_path).map_err(|e| e.to_string())?;
    let asr = WhisperAsr::from_gguf(&gguf)
        .map_err(|e| e.to_string())?
        .with_backend(backend);
    if !asr.has_tokenizer() {
        return Err(
            "run (whisper --word-timestamps): the GGUF embeds no tokenizer \
             (`vokra.tokenizer.model`), so word spans cannot be rendered to text. \
             Re-convert with the tokenizer chunk."
                .to_owned(),
        );
    }

    let mut cfg = BeamSearchConfig::greedy(vokra_models::whisper::greedy::DEFAULT_MAX_NEW_TOKENS);
    cfg.word_timestamps = true;
    // `--beam-size` (and its companions) ride the same surface: width 1 is
    // the greedy-equivalent alignment run.
    cfg.beam_width = a.beam_size.max(1);
    if a.beam_size > 1 {
        cfg.length_normalization = a.length_penalty;
        cfg.no_repeat_ngram_size = a.no_repeat_ngram;
    }
    let hyps = asr
        .transcribe_tokens_beam_nbest(pcm, &cfg)
        .map_err(|e| e.to_string())?;
    let best = hyps
        .first()
        .ok_or("run (whisper --word-timestamps): beam search produced no hypothesis")?;
    let text = asr.render_ids(&best.tokens).map_err(|e| e.to_string())?;
    println!("asr: {text}");

    // `beam_search` raises an explicit error when word timestamps were
    // requested on a model without alignment heads, so reaching here with
    // `None` would mean the driver silently skipped the alignment — surface
    // that rather than printing zero words as if the clip had none.
    let timings = best.word_timestamps.as_ref().ok_or(
        "run (whisper --word-timestamps): the decoder returned no alignment for the best \
         hypothesis (expected cross-attention word timings)",
    )?;
    for w in timings {
        let span = best.tokens.get(w.token_start..w.token_end).ok_or(
            "run (whisper --word-timestamps): word span out of range for the \
                    hypothesis tokens",
        )?;
        let word = asr.render_ids(span).map_err(|e| e.to_string())?;
        println!("word\t{word}\t{:.3}\t{:.3}", w.start, w.end);
    }
    Ok(())
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

    // ---- P2 cc-10 / cc-19: voxtral route + whisper word timestamps -------

    #[test]
    fn parses_word_timestamps_language_and_bare_prompt_flags() {
        let a = parse_args(&args(&["--model", "m.gguf", "--input", "in.wav"])).expect("valid");
        assert!(!a.word_timestamps);
        assert_eq!(a.language, None);
        assert!(!a.bare_prompt);

        let a = parse_args(&args(&[
            "--model",
            "m.gguf",
            "--input",
            "in.wav",
            "--word-timestamps",
        ]))
        .expect("valid");
        assert!(a.word_timestamps);

        let a = parse_args(&args(&[
            "--model",
            "v.gguf",
            "--input",
            "in.wav",
            "--language",
            "fr",
            "--bare-prompt",
        ]))
        .expect("valid");
        assert_eq!(a.language.as_deref(), Some("fr"));
        assert!(a.bare_prompt);

        // `auto` is carried verbatim; the run arm maps it to "omit the
        // lang: segment".
        let a = parse_args(&args(&["--model", "v.gguf", "--language", "auto"])).expect("valid");
        assert_eq!(a.language.as_deref(), Some("auto"));
    }

    #[test]
    fn rejects_dangling_and_empty_language() {
        assert_eq!(
            parse_args(&args(&["--model", "v.gguf", "--language"]))
                .err()
                .unwrap(),
            "--language requires a value"
        );
        assert!(
            parse_args(&args(&["--model", "v.gguf", "--language", ""]))
                .err()
                .unwrap()
                .contains("must not be empty")
        );
    }

    /// `--word-timestamps` off the whisper arch is an explicit contract
    /// error (FR-EX-08 — Voxtral has no cross-attention alignment).
    #[test]
    fn word_timestamps_on_non_whisper_arch_is_rejected() {
        let err = main(&args(&[
            "--model",
            &silero_fixture(),
            "--input",
            "unused.wav",
            "--word-timestamps",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--word-timestamps is only supported for the whisper"),
            "got: {err}"
        );
    }

    /// `--language` / `--bare-prompt` off the voxtral arch likewise.
    #[test]
    fn voxtral_prompt_flags_on_other_arch_are_rejected() {
        for flag in [
            vec!["--language".to_owned(), "fr".to_owned()],
            vec!["--bare-prompt".to_owned()],
        ] {
            let mut argv = args(&["--model", &silero_fixture(), "--input", "unused.wav"]);
            argv.extend(flag);
            let err = main(&argv).unwrap_err();
            assert!(
                err.contains("--language / --bare-prompt are only supported for the voxtral"),
                "got: {err}"
            );
        }
    }

    /// A metadata-only `voxtral` GGUF reaches the run arm (dispatch is
    /// bare by design) and then fails loudly when the concrete engine binds
    /// — never a silent success (FR-EX-08).
    #[test]
    fn voxtral_metadata_only_gguf_fails_loudly_at_engine_bind() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let dir = std::env::temp_dir();
        let model = dir.join(format!("vokra-cli-vox-meta-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let in_wav = dir.join(format!("vokra-cli-vox-meta-{}.wav", std::process::id()));
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
        assert!(!err.is_empty(), "loud bind error expected");
    }

    /// Voxtral without `--input` is a contract error naming the flag.
    #[test]
    fn voxtral_without_input_is_a_contract_error() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let model =
            std::env::temp_dir().join(format!("vokra-cli-vox-noinput-{}.gguf", std::process::id()));
        std::fs::write(&model, &bytes).unwrap();
        let err = main(&args(&["--model", model.to_str().unwrap()])).unwrap_err();
        let _ = std::fs::remove_file(&model);
        assert!(err.contains("--input"), "actionable: {err}");
    }

    /// Real-GGUF gated e2e for the voxtral CLI route (P2 cc-10): set
    /// `VOKRA_VOXTRAL_GGUF` (+ optional `VOKRA_VOXTRAL_WAV`) to run; skips
    /// clean when unset. Prints the transcript to stdout via the run arm —
    /// the numeric/text assertion rides the models-crate e2e test
    /// (`voxtral_transcription_prompt.rs`), this one proves the CLI wiring.
    #[test]
    fn voxtral_real_gguf_cli_route_gated() {
        let Ok(model) = std::env::var("VOKRA_VOXTRAL_GGUF") else {
            eprintln!("skipping voxtral CLI e2e: set VOKRA_VOXTRAL_GGUF to run");
            return;
        };
        let wav = std::env::var("VOKRA_VOXTRAL_WAV").unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/audio/jfk-30s.wav")
                .to_string_lossy()
                .into_owned()
        });
        let code = main(&args(&["--model", &model, "--input", &wav])).expect("voxtral CLI runs");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    /// Real-GGUF gated check for `--word-timestamps` (cc-19): set
    /// `VOKRA_WHISPER_GGUF` (+ optional `VOKRA_WHISPER_WAV`).
    #[test]
    fn whisper_word_timestamps_cli_route_gated() {
        let Ok(model) = std::env::var("VOKRA_WHISPER_GGUF") else {
            eprintln!("skipping whisper word-timestamps CLI e2e: set VOKRA_WHISPER_GGUF to run");
            return;
        };
        let wav = std::env::var("VOKRA_WHISPER_WAV").unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/audio/jfk-30s.wav")
                .to_string_lossy()
                .into_owned()
        });
        let code = main(&args(&[
            "--model",
            &model,
            "--input",
            &wav,
            "--word-timestamps",
        ]))
        .expect("whisper --word-timestamps runs");
        assert_eq!(code, ExitCode::SUCCESS);
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
        // P2 cc-10 / cc-19 surface.
        assert!(
            USAGE.contains("--word-timestamps"),
            "USAGE lists --word-timestamps"
        );
        assert!(USAGE.contains("--language"), "USAGE lists --language");
        assert!(USAGE.contains("--bare-prompt"), "USAGE lists --bare-prompt");
        assert!(USAGE.contains("voxtral"), "USAGE names the voxtral arch");
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

    // ---- cc-24: kokoro phoneme-id route -----------------------------------

    /// The misaki table shipped in a real Kokoro GGUF: index = id, entry 0 is
    /// the (unaddressable) pad sentinel. Trimmed to the symbols these tests
    /// need; the real voice carries 178.
    fn kokoro_symbols() -> Vec<String> {
        ["", "ð", "ə", " ", "k", "w", "ɪ"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    #[test]
    fn kokoro_tokenizer_wraps_ids_in_upstream_sentinels() {
        // Upstream builds `input_ids = [0, *ids, 0]` (kokoro==0.9.4).
        let ids = kokoro_phoneme_ids("ðə", &kokoro_symbols()).expect("all symbols known");
        assert_eq!(ids, vec![0, 1, 2, 0]);
    }

    #[test]
    fn kokoro_tokenizer_maps_every_char_by_table_position() {
        let ids = kokoro_phoneme_ids("ðə kwɪ", &kokoro_symbols()).expect("all symbols known");
        // ð=1 ə=2 space=3 k=4 w=5 ɪ=6, wrapped in the sentinels.
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 0]);
    }

    #[test]
    fn kokoro_tokenizer_rejects_unknown_characters_instead_of_dropping_them() {
        // FR-EX-08: silently dropping an unmappable phoneme would change the
        // utterance with no signal. The message must name the offenders and
        // point at the missing G2P bridge.
        let err = kokoro_phoneme_ids("ðəZ🎵", &kokoro_symbols()).unwrap_err();
        assert!(err.contains('Z') && err.contains('🎵'), "names them: {err}");
        assert!(
            err.contains("PHONEMES"),
            "explains the input contract: {err}"
        );
    }

    #[test]
    fn kokoro_tokenizer_rejects_empty_phoneme_text() {
        let err = kokoro_phoneme_ids("", &kokoro_symbols()).unwrap_err();
        assert!(err.contains("no phonemes"), "{err}");
    }

    #[test]
    fn kokoro_tokenizer_accepts_the_piper_raw_id_form() {
        // Content ids only; the sentinels are added here, as in piper's
        // `parse_content` / `phonemize` split. Whitespace and comma separated
        // forms are equivalent.
        let want = vec![0, 1, 2, 3, 0];
        assert_eq!(
            kokoro_phoneme_ids("1 2 3", &kokoro_symbols()).unwrap(),
            want
        );
        assert_eq!(
            kokoro_phoneme_ids("1,2,3", &kokoro_symbols()).unwrap(),
            want
        );
        assert_eq!(
            kokoro_phoneme_ids(" 1,  2 ,3 ", &kokoro_symbols()).unwrap(),
            want
        );
    }

    #[test]
    fn kokoro_raw_id_form_agrees_with_the_symbol_form() {
        // The two spellings of the same utterance must tokenize identically —
        // otherwise one of them is silently synthesizing something else.
        let syms = kokoro_symbols();
        assert_eq!(
            kokoro_phoneme_ids("ðə kwɪ", &syms).unwrap(),
            kokoro_phoneme_ids("1 2 3 4 5 6", &syms).unwrap()
        );
    }

    #[test]
    fn kokoro_raw_id_form_rejects_out_of_range_and_the_pad_sentinel() {
        let syms = kokoro_symbols(); // 7 entries → content ids are 1..7
        for bad in ["0", "7", "99"] {
            let err = kokoro_phoneme_ids(bad, &syms).unwrap_err();
            assert!(err.contains("out of range"), "{bad}: {err}");
        }
    }

    #[test]
    fn kokoro_raw_id_form_is_refused_when_a_digit_is_itself_a_symbol() {
        // The digit heuristic is only sound while no symbol is a bare digit.
        // A table that breaks that must produce a loud ambiguity error rather
        // than silently picking one reading (FR-EX-08).
        let mut syms = kokoro_symbols();
        syms.push("2".to_owned());
        let err = kokoro_phoneme_ids("1 2 3", &syms).unwrap_err();
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn kokoro_style_vector_accepts_both_upstream_widths() {
        let dir = std::env::temp_dir();
        for n in [4usize, 8] {
            let p = dir.join(format!("vokra-cli-style-{n}-{}.f32", std::process::id()));
            let bytes: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
            std::fs::write(&p, &bytes).unwrap();
            // style_dim = 4 → accepts 4 (single) and 8 (full ref_s row).
            let v = read_style_vector(p.to_str().unwrap(), 4).expect("width accepted");
            assert_eq!(v.len(), n);
            assert_eq!(v[0], 0.0);
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn kokoro_style_vector_rejects_wrong_width_and_ragged_files() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("vokra-cli-style-bad-{}.f32", std::process::id()));

        // Wrong float count (5 is neither style_dim nor 2*style_dim).
        std::fs::write(
            &p,
            (0..5)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let err = read_style_vector(p.to_str().unwrap(), 4).unwrap_err();
        assert!(err.contains("5 floats"), "{err}");

        // Not a whole number of f32s.
        std::fs::write(&p, [0u8; 6]).unwrap();
        let err = read_style_vector(p.to_str().unwrap(), 4).unwrap_err();
        assert!(err.contains("whole number of f32s"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn kokoro_style_flags_are_rejected_off_the_kokoro_arch() {
        // FR-EX-08: a style knob silently ignored on another arch would change
        // nothing about the output while implying it had.
        let model = silero_fixture();
        // `--input` is any readable path here: the arch guard fires before the
        // task itself runs, so the fixture doubles as the input.
        for flag in [["--voice", "af_heart"], ["--style", "/nonexistent.f32"]] {
            let err = main(&args(&[
                "--model", &model, "--input", &model, flag[0], flag[1],
            ]))
            .unwrap_err();
            assert!(
                err.contains("only supported for the kokoro arch"),
                "{flag:?}: {err}"
            );
        }
    }

    #[test]
    fn kokoro_length_scale_is_rejected_off_the_kokoro_arch() {
        let model = silero_fixture();
        let err = main(&args(&[
            "--model",
            &model,
            "--input",
            &model,
            "--length-scale",
            "1.5",
        ]))
        .unwrap_err();
        assert!(err.contains("only supported for the kokoro arch"), "{err}");
    }

    #[test]
    fn kokoro_length_scale_rejects_non_positive_values() {
        for bad in ["0", "-1.0", "nan"] {
            let err = parse_args(&args(&["--model", "m.gguf", "--length-scale", bad]))
                .map(|_| ())
                .unwrap_err();
            assert!(err.contains("positive finite"), "{bad}: {err}");
        }
    }
}
