//! `vokra-eval` — run one evaluation metric over a hypothesis/reference pair or
//! a manifest of them and print the scores (M1-09a; FR-TL-03).
//!
//! ```text
//! vokra-eval wer      --hyp "a x c" --ref "a b c"
//! vokra-eval cer      --hyp-file hyp.txt --ref-file ref.txt
//! vokra-eval mel-loss --hyp hyp.wav --ref ref.wav [--sample-rate 22050 --n-fft 1024 --hop 256 --n-mels 80]
//! vokra-eval wer      --manifest pairs.txt      # batch: per-item + aggregate mean
//! ```
//!
//! Output is a `key=value` report line per result (and one aggregate line in
//! manifest mode). Neural MOS metrics (UTMOS/DNSMOS, M1-09b) are not wired up
//! here — they need model weights.

use std::process::ExitCode;

use vokra_eval::manifest::Manifest;
use vokra_eval::metrics::{AudioRefMetric, Cer, MelLoss, Metric, TextMetric, Wer};
use vokra_eval::wav;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("vokra-eval: {msg}");
            ExitCode::from(2)
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum MetricKind {
    Wer,
    Cer,
    MelLoss,
}

struct Cli {
    metric: MetricKind,
    hyp: Option<String>,
    reference: Option<String>,
    hyp_file: Option<String>,
    ref_file: Option<String>,
    manifest: Option<String>,
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => {
            print_usage();
            return Err("no metric given".to_owned());
        }
        Some("-h") | Some("--help") => {
            print_usage();
            return Ok(());
        }
        _ => {}
    }
    let cli = parse_cli(&args)?;
    match cli.metric {
        MetricKind::Wer => run_text(&cli, Wer),
        MetricKind::Cer => run_text(&cli, Cer),
        MetricKind::MelLoss => run_audio(&cli),
    }
}

/// Reads the value that follows flag `key`, advancing `i` past it.
fn take<'a>(args: &'a [String], i: &mut usize, key: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| format!("flag `{key}` needs a value"))
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let metric = match args[0].as_str() {
        "wer" => MetricKind::Wer,
        "cer" => MetricKind::Cer,
        "mel-loss" | "mel_loss" => MetricKind::MelLoss,
        other => {
            return Err(format!(
                "unknown metric `{other}` (expected wer|cer|mel-loss)"
            ));
        }
    };
    let mut cli = Cli {
        metric,
        hyp: None,
        reference: None,
        hyp_file: None,
        ref_file: None,
        manifest: None,
        // librosa-style TTS defaults; override for a model's own FrontendSpec.
        sample_rate: 22_050,
        n_fft: 1_024,
        hop: 256,
        n_mels: 80,
    };
    let mut i = 1usize;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "--hyp" => cli.hyp = Some(take(args, &mut i, key)?.to_owned()),
            "--ref" => cli.reference = Some(take(args, &mut i, key)?.to_owned()),
            "--hyp-file" => cli.hyp_file = Some(take(args, &mut i, key)?.to_owned()),
            "--ref-file" => cli.ref_file = Some(take(args, &mut i, key)?.to_owned()),
            "--manifest" => cli.manifest = Some(take(args, &mut i, key)?.to_owned()),
            "--sample-rate" => {
                cli.sample_rate = take(args, &mut i, key)?
                    .parse()
                    .map_err(|_| "invalid --sample-rate".to_owned())?;
            }
            "--n-fft" => {
                cli.n_fft = take(args, &mut i, key)?
                    .parse()
                    .map_err(|_| "invalid --n-fft".to_owned())?;
            }
            "--hop" => {
                cli.hop = take(args, &mut i, key)?
                    .parse()
                    .map_err(|_| "invalid --hop".to_owned())?;
            }
            "--n-mels" => {
                cli.n_mels = take(args, &mut i, key)?
                    .parse()
                    .map_err(|_| "invalid --n-mels".to_owned())?;
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
        i += 1;
    }
    Ok(cli)
}

fn text_input(literal: Option<&str>, file: Option<&str>, which: &str) -> Result<String, String> {
    if let Some(t) = literal {
        return Ok(t.to_owned());
    }
    if let Some(f) = file {
        return std::fs::read_to_string(f).map_err(|e| format!("reading --{which}-file {f}: {e}"));
    }
    Err(format!(
        "missing --{which} (or --{which}-file / --manifest)"
    ))
}

fn run_text<M: TextMetric>(cli: &Cli, metric: M) -> Result<(), String> {
    if let Some(path) = &cli.manifest {
        let man = Manifest::load(path).map_err(|e| format!("reading manifest {path}: {e}"))?;
        let mut sum = 0.0f64;
        for (idx, rec) in man.records.iter().enumerate() {
            let hyp = rec
                .get("hyp")
                .ok_or_else(|| format!("manifest record at line {} has no `hyp`", rec.line))?;
            let reference = rec
                .get("ref")
                .ok_or_else(|| format!("manifest record at line {} has no `ref`", rec.line))?;
            let score = metric.eval_text(hyp, reference);
            println!("item={idx} metric={} score={score:.6}", metric.name());
            sum += score;
        }
        report_aggregate(metric.name(), man.records.len(), sum);
    } else {
        let hyp = text_input(cli.hyp.as_deref(), cli.hyp_file.as_deref(), "hyp")?;
        let reference = text_input(cli.reference.as_deref(), cli.ref_file.as_deref(), "ref")?;
        let score = metric.eval_text(&hyp, &reference);
        println!("metric={} score={score:.6}", metric.name());
    }
    Ok(())
}

fn run_audio(cli: &Cli) -> Result<(), String> {
    let metric = MelLoss::new(cli.sample_rate, cli.n_fft, cli.hop, cli.n_mels);
    if let Some(path) = &cli.manifest {
        let man = Manifest::load(path).map_err(|e| format!("reading manifest {path}: {e}"))?;
        let mut sum = 0.0f64;
        for (idx, rec) in man.records.iter().enumerate() {
            let hp = rec
                .get("hyp_wav")
                .ok_or_else(|| format!("manifest record at line {} has no `hyp_wav`", rec.line))?;
            let rp = rec
                .get("ref_wav")
                .ok_or_else(|| format!("manifest record at line {} has no `ref_wav`", rec.line))?;
            let score = eval_wav_pair(&metric, hp, rp)?;
            println!("item={idx} metric={} score={score:.6}", metric.name());
            sum += score;
        }
        report_aggregate(metric.name(), man.records.len(), sum);
    } else {
        let hp = cli
            .hyp
            .as_deref()
            .ok_or("missing --hyp <hyp.wav> (or --manifest)")?;
        let rp = cli
            .reference
            .as_deref()
            .ok_or("missing --ref <ref.wav> (or --manifest)")?;
        let score = eval_wav_pair(&metric, hp, rp)?;
        println!("metric={} score={score:.6}", metric.name());
    }
    Ok(())
}

fn eval_wav_pair(metric: &MelLoss, hyp_path: &str, ref_path: &str) -> Result<f64, String> {
    let h = wav::read_wav(hyp_path).map_err(|e| format!("reading {hyp_path}: {e}"))?;
    let r = wav::read_wav(ref_path).map_err(|e| format!("reading {ref_path}: {e}"))?;
    if h.sample_rate != r.sample_rate {
        return Err(format!(
            "sample-rate mismatch: {hyp_path} is {} Hz, {ref_path} is {} Hz",
            h.sample_rate, r.sample_rate
        ));
    }
    metric
        .eval_audio(&h.samples, &r.samples, h.sample_rate)
        .map_err(|e| e.to_string())
}

fn report_aggregate(name: &str, count: usize, sum: f64) {
    let mean = if count == 0 { 0.0 } else { sum / count as f64 };
    println!("metric={name} count={count} mean={mean:.6}");
}

fn print_usage() {
    eprintln!(
        "vokra-eval — Vokra evaluation metrics (M1-09a)\n\
\n\
USAGE:\n\
    vokra-eval <metric> [--hyp <in> --ref <in> | --hyp-file <f> --ref-file <f> | --manifest <f>]\n\
\n\
METRICS:\n\
    wer        word error rate  (edit distance over whitespace tokens / ref words)\n\
    cer        char error rate  (edit distance over Unicode chars / ref chars)\n\
    mel-loss   mean L1 over log10-mel spectrograms of two WAV clips\n\
\n\
INPUTS:\n\
    wer / cer : --hyp/--ref take literal text (or --hyp-file/--ref-file for text files)\n\
    mel-loss  : --hyp/--ref take mono WAV paths (float32 or int16)\n\
    --manifest <f> : batch mode; each blank-line-separated record has\n\
                     `hyp`/`ref` (text) or `hyp_wav`/`ref_wav` (audio) keys\n\
\n\
MEL-LOSS FRONT-END (must match the model's FrontendSpec; defaults are TTS-style):\n\
    --sample-rate <hz>  [22050]   --n-fft <n> [1024]   --hop <n> [256]   --n-mels <n> [80]\n\
\n\
Prints `metric=<name> score=<v>` per result and an aggregate mean in manifest mode.\n\
Neural MOS metrics (UTMOS/DNSMOS) are M1-09b (blocked on weights) and not available here."
    );
}
