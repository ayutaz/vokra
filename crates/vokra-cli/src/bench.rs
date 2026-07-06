//! `vokra-cli bench` — measure RTF / TTFA / jitter / p50-p95-p99 (M1-10a) and
//! the relative perf-regression gate (M1-10b, NFR-PF-13).
//!
//! ```text
//! vokra-cli bench --model vad.gguf --input speech.wav --iters 20 --warmup 3
//! vokra-cli bench --model voice.gguf --text "hello" --format json --baseline base.json
//! ```
//!
//! Timing uses `std::time::Instant` only (no external bench crate — NFR-DS-02),
//! mirroring the `vokra-ops` / `vokra-models` `harness = false` benches. With
//! `--baseline <report.json>` the measured RTF is compared against the baseline
//! and a **>5% relative** regression sets a non-zero exit code (NFR-PF-13).
//!
//! The **absolute** NFR-PF-01 (Whisper base RTF < 1.0) / NFR-PF-02 (piper-plus
//! RTF < 0.5) thresholds are intentionally NOT asserted here: they require the
//! real full models and a stable measurement lab (blocked), so only the
//! relative-regression scaffold ships in this WP.

use std::process::ExitCode;
use std::time::Instant;

use vokra_core::BackendKind;
use vokra_core::quant::{
    DegradationReport, QuantPolicy, verify_hifigan_int8 as core_verify_hifigan_int8,
};

use crate::engine::{self, ModelTask, TaskHint};
use crate::report::{self, BenchReport, RegressionCheck};
use crate::wav;

/// Relative regression tolerance for the M1-10b gate (NFR-PF-13: 5%).
const REGRESSION_THRESHOLD: f64 = 0.05;

/// Default utterance for TTS bench when `--text` is not supplied.
const DEFAULT_BENCH_TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// Output serialization format for the bench report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Flat `key=value` line.
    Kv,
    /// Compact JSON object.
    Json,
}

/// Parsed `bench` arguments.
struct BenchArgs {
    model: String,
    input: Option<String>,
    text: Option<String>,
    iters: usize,
    warmup: usize,
    format: Format,
    baseline: Option<String>,
    /// Backend the model's hot ops run on (ASR only today). Default CPU; `metal`
    /// / `cuda` require the CLI built with the matching feature, else inference
    /// fails with an explicit unsupported-backend error (no silent CPU fallback).
    backend: BackendKind,
    /// Optional task override — only `mel-frontend` (Whisper log-mel only,
    /// M2-04-T11) is recognized today. Absent → default arch → task mapping.
    task_hint: Option<TaskHint>,
}

/// Parses a `--backend` value. The variants always exist (core enum); whether
/// they are *usable* depends on the CLI's compiled features — an unavailable
/// backend fails loudly at inference, never silently on CPU.
fn parse_backend(v: &str) -> Result<BackendKind, String> {
    match v {
        "cpu" => Ok(BackendKind::Cpu),
        "metal" => Ok(BackendKind::Metal),
        "cuda" => Ok(BackendKind::Cuda),
        other => Err(format!("unknown --backend `{other}` (cpu | metal | cuda)")),
    }
}

/// Parses a `--task` value into an optional [`TaskHint`]. Unknown values are
/// rejected loudly (no silent fallback to the arch default — FR-EX-08).
fn parse_task_hint(v: &str) -> Result<TaskHint, String> {
    match v {
        "mel-frontend" => Ok(TaskHint::MelFrontend),
        other => Err(format!("unknown --task `{other}` (mel-frontend)")),
    }
}

/// Returns `true` if the loaded model exercises the HiFi-GAN op.
///
/// M2-08-T12 gate: only piper-plus (MB-iSTFT-VITS2's HiFi-GAN ResBlock2
/// stack, `vokra-models::piper_plus::decoder`) uses HiFi-GAN today. Whisper,
/// Silero VAD, and the mel-frontend task do not, so they short-circuit the
/// verify gate. Kokoro (M2-07, future) uses an iSTFTNet head that is
/// deliberately *not* registered as HiFi-GAN (CLAUDE.md レビュアー A 修正) —
/// the registry marker (`quant::HIFIGAN_GENERATOR_OP`) matches piper-plus's
/// TTS decoder only.
fn hifigan_in_use(task: ModelTask) -> bool {
    matches!(task, ModelTask::Tts)
}

/// Runs the HiFi-GAN INT8 opt-in verification gate at bench construction
/// (M2-08-T12).
///
/// Three branches (`vokra_core::quant::verify_hifigan_int8`):
/// - opt-in + eval pass → `Ok(())`
/// - opt-in + eval not run → `VokraError::HifiganInt8VerifyMissing`
/// - opt-in + eval fail → `VokraError::HifiganInt8DegradationExceeded`
///
/// `policy` is `None` until the runtime chunk reader lands (T05 landing pad
/// in `vokra_core::quant::chunk`); until then bench uses the safe default
/// (no opt-in) and the gate is a no-op. The signature accepts an optional
/// [`DegradationReport`] so a follow-up ticket can wire
/// `vokra-cli bench --check-degradation …` into this seam without touching
/// the branching logic.
fn verify_hifigan_int8(
    task: ModelTask,
    policy: Option<&QuantPolicy>,
    report: Option<&DegradationReport>,
) -> Result<(), String> {
    let Some(policy) = policy else {
        // No policy attached (chunk reader not yet wired): the safe default
        // is `no opt-in` per M2-08 (see `resolve::default_vocoder_safe`), so
        // the gate is a no-op.
        return Ok(());
    };
    core_verify_hifigan_int8(policy, hifigan_in_use(task), report).map_err(|e| e.to_string())
}

/// A finished bench run: the report plus the optional regression verdict.
struct BenchOutcome {
    report: BenchReport,
    regression: Option<RegressionCheck>,
}

pub(crate) const USAGE: &str = "\
vokra-cli bench — measure RTF / TTFA / jitter / p50-p95-p99

USAGE:
    vokra-cli bench --model <model.gguf> [--input <in.wav>] [--text <string>]
                    [--iters <n>] [--warmup <n>] [--format kv|json] [--baseline <report.json>]

OPTIONS:
    --model <path>       GGUF model file (arch selects VAD / ASR / TTS)
    --input <path>       mono WAV input (required for VAD and ASR)
    --text <string>      text to synthesize (TTS; defaults to a fixed phrase)
    --iters <n>          timed iterations           [default 10]
    --warmup <n>         untimed warm-up iterations [default 3]
    --format kv|json     report format              [default kv]
    --baseline <path>    a previous JSON report; a >5% RTF regression exits non-zero
    --backend <name>     cpu | metal | cuda — ASR hot ops backend [default cpu]
                         (metal/cuda need the CLI built with that feature)
    --task <name>        override the arch default task. Today the only value
                         is `mel-frontend` (Whisper only): benches the log-mel
                         front-end alone (M2-04-T11), so the fused vs unfused
                         RTF isn't polluted by encoder / decoder time.
    -h, --help           print this help
";

fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut iters: usize = 10;
    let mut warmup: usize = 3;
    let mut format = Format::Kv;
    let mut baseline: Option<String> = None;
    let mut backend = BackendKind::Cpu;
    let mut task_hint: Option<TaskHint> = None;

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
            "--iters" => {
                let v = args.get(i + 1).ok_or("--iters requires a value")?;
                iters = v.parse().map_err(|_| format!("invalid --iters `{v}`"))?;
                i += 2;
            }
            "--warmup" => {
                let v = args.get(i + 1).ok_or("--warmup requires a value")?;
                warmup = v.parse().map_err(|_| format!("invalid --warmup `{v}`"))?;
                i += 2;
            }
            "--format" => {
                let v = args.get(i + 1).ok_or("--format requires a value")?;
                format = match v.as_str() {
                    "kv" => Format::Kv,
                    "json" => Format::Json,
                    other => return Err(format!("unknown --format `{other}` (kv | json)")),
                };
                i += 2;
            }
            "--baseline" => {
                baseline = Some(
                    args.get(i + 1)
                        .ok_or("--baseline requires a value")?
                        .clone(),
                );
                i += 2;
            }
            "--backend" => {
                let v = args.get(i + 1).ok_or("--backend requires a value")?;
                backend = parse_backend(v)?;
                i += 2;
            }
            "--task" => {
                let v = args.get(i + 1).ok_or("--task requires a value")?;
                task_hint = Some(parse_task_hint(v)?);
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    if iters == 0 {
        return Err("--iters must be > 0".to_owned());
    }
    Ok(BenchArgs {
        model: model.ok_or("--model is required")?,
        input,
        text,
        iters,
        warmup,
        format,
        baseline,
        backend,
        task_hint,
    })
}

/// Runs `f` for `warmup` untimed iterations, then `iters` timed iterations,
/// returning the per-iteration wall-clock latency in seconds.
///
/// Propagates the first error from `f` (a failed inference aborts the bench).
fn time_iters<F>(warmup: usize, iters: usize, mut f: F) -> Result<Vec<f64>, String>
where
    F: FnMut() -> Result<(), String>,
{
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    Ok(samples)
}

/// GGUF metadata key holding the Whisper mel-band count. Kept in sync with the
/// converter (see `vokra-models::whisper::mod.rs`).
const KEY_WHISPER_N_MELS: &str = "vokra.whisper.n_mels";

/// Loads the model, times the task and (optionally) checks the baseline.
fn execute(args: &BenchArgs) -> Result<BenchOutcome, String> {
    let (session, task) =
        engine::load_session_with_backend(&args.model, args.backend, args.task_hint)?;

    // M2-08-T12: HiFi-GAN INT8 opt-in verify gate. `None` policy short-circuits
    // (the `vokra.quant.*` chunk reader is a T05 landing pad — see
    // `vokra_core::quant::chunk`). When T05 lands this reads
    // `QuantPolicy::from_gguf(session.gguf())?` and threads a caller-supplied
    // `DegradationReport` (`--check-degradation`) into the same seam.
    verify_hifigan_int8(task, None, None)?;

    let (task_name, audio_seconds, samples) = match task {
        ModelTask::Vad => {
            let path = args
                .input
                .as_deref()
                .ok_or("bench (VAD): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let sr = clip.sample_rate;
            let pcm = clip.samples;
            let audio_seconds = pcm.len() as f64 / f64::from(sr);
            let samples = time_iters(args.warmup, args.iters, || {
                let mut handle = session.open_vad_stream().map_err(|e| e.to_string())?;
                handle.push_pcm(&pcm, sr).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("vad", audio_seconds, samples)
        }
        ModelTask::Asr => {
            let path = args
                .input
                .as_deref()
                .ok_or("bench (ASR): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let audio_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
            let pcm = clip.samples;
            let samples = time_iters(args.warmup, args.iters, || {
                session.asr().transcribe(&pcm).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("asr", audio_seconds, samples)
        }
        ModelTask::Tts => {
            let text = args.text.as_deref().unwrap_or(DEFAULT_BENCH_TEXT);
            // One synth up front to learn the output length (RTF denominator).
            let first = session.tts().synthesize(text).map_err(|e| e.to_string())?;
            let audio_seconds = first.samples.len() as f64 / f64::from(first.sample_rate);
            let samples = time_iters(args.warmup, args.iters, || {
                session.tts().synthesize(text).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("tts", audio_seconds, samples)
        }
        ModelTask::MelFrontend => {
            // M2-04-T11: bench the Whisper log-mel front-end alone. Running
            // `whisper::mel::log_mel` directly against the input WAV keeps the
            // measurement isolated to the fused / unfused STFT → power → mel →
            // log10 → transpose path (M2-04-T08 toggle) — no encoder / decoder
            // time leaks into the RTF. `n_mels` is read from the GGUF exactly
            // the way `WhisperConfig` reads it, so the bench and the full ASR
            // path exercise the same front-end shape.
            let path = args
                .input
                .as_deref()
                .ok_or("bench (mel-frontend): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let audio_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
            let pcm = clip.samples;
            let n_mels = session
                .gguf()
                .get(KEY_WHISPER_N_MELS)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("GGUF is missing the `{KEY_WHISPER_N_MELS}` metadata key"))?
                as usize;
            let samples = time_iters(args.warmup, args.iters, || {
                // `black_box` prevents LLVM from noticing the result is
                // unused and dead-code-eliminating the whole call. Same trick
                // used by the ops-crate benches (see `fft_bench.rs`).
                let out = vokra_models::whisper::mel::log_mel(&pcm, n_mels);
                std::hint::black_box(out);
                Ok(())
            })?;
            ("mel-frontend", audio_seconds, samples)
        }
    };

    let stats = report::summarize(&samples).ok_or("no timing samples (iters must be > 0)")?;
    let rtf = if audio_seconds > 0.0 {
        stats.mean / audio_seconds
    } else {
        0.0
    };
    let report = BenchReport {
        task: task_name.to_owned(),
        iters: args.iters,
        warmup: args.warmup,
        audio_seconds,
        rtf,
        // Non-streaming: the first audio chunk is the whole clip (mean latency).
        ttfa_ms: stats.mean * 1e3,
        latency: stats,
    };

    let regression = match &args.baseline {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("reading baseline {path}: {e}"))?;
            let baseline_rtf = report::parse_baseline_rtf(&bytes)?;
            Some(report::compare(
                baseline_rtf,
                report.rtf,
                REGRESSION_THRESHOLD,
            ))
        }
        None => None,
    };

    Ok(BenchOutcome { report, regression })
}

/// Entry point for `vokra-cli bench`.
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let a = parse_args(args)?;
    let outcome = execute(&a)?;

    match a.format {
        Format::Kv => println!("{}", outcome.report.to_kv()),
        Format::Json => println!("{}", outcome.report.to_json()),
    }

    if let Some(reg) = &outcome.regression {
        println!(
            "regression: metric=rtf baseline={:.6} current={:.6} ratio={:.4} \
             threshold={:.2} regressed={}",
            reg.baseline, reg.current, reg.ratio, reg.threshold, reg.regressed
        );
        if reg.regressed {
            eprintln!(
                "bench: RTF regressed by more than {:.0}% vs baseline (NFR-PF-13)",
                reg.threshold * 100.0
            );
            return Ok(ExitCode::from(3));
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::quant::{CalibrationRef, DEGRADATION_THRESHOLD, QuantPolicy, QuantScheme};

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    // ----- M2-08-T12: HiFi-GAN INT8 opt-in verify gate --------------------

    fn opt_in_policy() -> QuantPolicy {
        QuantPolicy::new(QuantScheme::Fp16)
            .with_hifigan_int8_opt_in(CalibrationRef::new("hifigan-int8-cal-v1"))
    }

    #[test]
    fn hifigan_int8_verify_no_policy_short_circuits() {
        // No `vokra.quant.*` chunk (today's steady state — T05 landing pad):
        // the gate is a no-op even on a TTS session (piper-plus HiFi-GAN).
        verify_hifigan_int8(ModelTask::Tts, None, None).unwrap();
        verify_hifigan_int8(ModelTask::Asr, None, None).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_but_no_tts_short_circuits() {
        // A Whisper session with an opt-in policy (e.g. shared policy across
        // a batch of models) does not exercise the HiFi-GAN op — the gate
        // short-circuits without demanding a report.
        let policy = opt_in_policy();
        verify_hifigan_int8(ModelTask::Asr, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::Vad, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::MelFrontend, Some(&policy), None).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_without_report_errors_verify_missing() {
        // Branch (b): opt-in + piper-plus TTS + no attached report → hard
        // error mentioning verify-missing.
        let policy = opt_in_policy();
        let err = verify_hifigan_int8(ModelTask::Tts, Some(&policy), None).unwrap_err();
        assert!(
            err.contains("hifigan int8 verify missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_with_passing_report_ok() {
        // Branch (a): opt-in + piper-plus TTS + passing report → allowed.
        let policy = opt_in_policy();
        let report = DegradationReport::from_mel_loss(1.0, 1.02, DEGRADATION_THRESHOLD);
        assert!(report.passes_5pct_gate);
        verify_hifigan_int8(ModelTask::Tts, Some(&policy), Some(&report)).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_with_failing_report_errors_degradation_exceeded() {
        // Branch (c): opt-in + piper-plus TTS + failing report → hard error
        // mentioning degradation-exceeded and carrying the delta.
        let policy = opt_in_policy();
        let report = DegradationReport::from_mel_loss(1.0, 1.20, DEGRADATION_THRESHOLD);
        assert!(!report.passes_5pct_gate);
        let err = verify_hifigan_int8(ModelTask::Tts, Some(&policy), Some(&report)).unwrap_err();
        assert!(
            err.contains("hifigan int8 degradation exceeded"),
            "unexpected error: {err}"
        );
        assert!(err.contains("threshold 0.0500"), "unexpected error: {err}");
    }

    #[test]
    fn hifigan_int8_verify_no_opt_in_never_errors() {
        // Policy present but opt-in is off → gate is a no-op regardless of
        // task or report.
        let policy = QuantPolicy::new(QuantScheme::Fp16);
        verify_hifigan_int8(ModelTask::Tts, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::Asr, Some(&policy), None).unwrap();
    }

    fn silero_fixture() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn parses_defaults_and_overrides() {
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.iters, 10);
        assert_eq!(a.warmup, 3);
        assert_eq!(a.format, Format::Kv);

        let a = parse_args(&args(&[
            "--model", "m.gguf", "--iters", "50", "--warmup", "5", "--format", "json",
        ]))
        .expect("valid");
        assert_eq!(a.iters, 50);
        assert_eq!(a.warmup, 5);
        assert_eq!(a.format, Format::Json);
    }

    #[test]
    fn rejects_bad_args() {
        assert_eq!(
            parse_args(&args(&["--iters", "10"])).err().unwrap(),
            "--model is required"
        );
        assert!(
            parse_args(&args(&["--model", "m", "--iters", "xyz"]))
                .err()
                .unwrap()
                .contains("invalid --iters")
        );
        assert!(
            parse_args(&args(&["--model", "m", "--format", "yaml"]))
                .err()
                .unwrap()
                .contains("unknown --format")
        );
        assert_eq!(
            parse_args(&args(&["--model", "m", "--iters", "0"]))
                .err()
                .unwrap(),
            "--iters must be > 0"
        );
        assert!(
            parse_args(&args(&["--model", "m", "--stray"]))
                .err()
                .unwrap()
                .contains("unexpected argument")
        );
        assert!(
            parse_args(&args(&["--model", "m", "--task", "asr"]))
                .err()
                .unwrap()
                .contains("unknown --task")
        );
    }

    #[test]
    fn parses_task_mel_frontend_hint() {
        let a = parse_args(&args(&["--model", "m.gguf", "--task", "mel-frontend"]))
            .expect("valid --task");
        assert_eq!(a.task_hint, Some(TaskHint::MelFrontend));
        // Default: no hint.
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.task_hint, None);
    }

    #[test]
    fn time_iters_warms_up_then_times_each_iteration() {
        let mut calls = 0u32;
        let samples = time_iters(2, 5, || {
            calls += 1;
            Ok(())
        })
        .expect("no error");
        assert_eq!(calls, 7, "2 warm-up + 5 timed calls");
        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|s| s.is_finite() && *s >= 0.0));
    }

    #[test]
    fn time_iters_propagates_errors() {
        let r = time_iters(0, 3, || Err("boom".to_owned()));
        assert_eq!(r.err().unwrap(), "boom");
    }

    /// Builds a bare Whisper GGUF sufficient for the `mel-frontend` task (arch
    /// key + `n_mels` metadata). The bench arm skips weight loading and only
    /// needs `vokra.whisper.n_mels`, so no encoder / decoder tensors are
    /// required — this keeps the test asset synthetic and in-repo (no external
    /// fixture needed for M2-04-T11's routing test).
    fn write_bare_whisper_gguf() -> std::path::PathBuf {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "whisper");
        b.add_u32("vokra.whisper.n_mels", 80);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-bench-melfront-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn bench_mel_frontend_runs_log_mel_and_reports_rtf() {
        // 1 s of silence @ 16 kHz — `log_mel` pads to 30 s internally.
        let mut wav_path = std::env::temp_dir();
        wav_path.push(format!("vokra-cli-bench-mel-{}.wav", std::process::id()));
        wav::write_wav(&wav_path, &vec![0.0f32; 16_000], 16_000).expect("write wav");
        let gguf_path = write_bare_whisper_gguf();

        let a = BenchArgs {
            model: gguf_path.to_string_lossy().into_owned(),
            input: Some(wav_path.to_string_lossy().into_owned()),
            text: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::MelFrontend),
        };
        let outcome = execute(&a).expect("bench runs");
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&gguf_path);

        assert_eq!(outcome.report.task, "mel-frontend");
        assert_eq!(outcome.report.iters, 1);
        assert_eq!(outcome.report.latency.count, 1);
        assert!(outcome.report.audio_seconds > 0.0);
        assert!(outcome.report.rtf.is_finite() && outcome.report.rtf >= 0.0);
        assert!(outcome.regression.is_none());
    }

    #[test]
    fn bench_mel_frontend_rejects_non_whisper_arch() {
        // The hint is Whisper-only: pointing it at Silero must fail loudly
        // (FR-EX-08: no silent fallback to VAD).
        let a = BenchArgs {
            model: silero_fixture(),
            input: None,
            text: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::MelFrontend),
        };
        let err = match execute(&a) {
            Err(e) => e,
            Ok(_) => panic!("hint on non-whisper arch must be rejected"),
        };
        assert!(
            err.contains("only supported on arch `whisper`"),
            "got: {err}"
        );
    }

    #[test]
    fn bench_vad_over_committed_fixture_reports_well_formed_stats() {
        // Internal-oracle only: committed Silero GGUF + a generated silence WAV.
        let mut wav_path = std::env::temp_dir();
        wav_path.push(format!("vokra-cli-bench-{}.wav", std::process::id()));
        wav::write_wav(&wav_path, &vec![0.0f32; 16_000], 16_000).expect("write wav");

        let a = BenchArgs {
            model: silero_fixture(),
            input: Some(wav_path.to_string_lossy().into_owned()),
            text: None,
            iters: 2,
            warmup: 1,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: None,
        };
        let outcome = execute(&a).expect("bench runs");
        let _ = std::fs::remove_file(&wav_path);

        assert_eq!(outcome.report.task, "vad");
        assert_eq!(outcome.report.iters, 2);
        assert_eq!(outcome.report.latency.count, 2);
        assert!(outcome.report.audio_seconds > 0.0);
        assert!(outcome.report.rtf.is_finite() && outcome.report.rtf >= 0.0);
        assert!(outcome.regression.is_none());
        // The report serializes to parseable JSON with a readable-back rtf.
        let rtf = report::parse_baseline_rtf(outcome.report.to_json().as_bytes()).unwrap();
        assert!(rtf.is_finite());
    }
}
