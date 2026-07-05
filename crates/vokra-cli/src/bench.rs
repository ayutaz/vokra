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

use crate::engine::{self, ModelTask};
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

/// Loads the model, times the task and (optionally) checks the baseline.
fn execute(args: &BenchArgs) -> Result<BenchOutcome, String> {
    let (session, task) = engine::load_session_with_backend(&args.model, args.backend)?;

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
