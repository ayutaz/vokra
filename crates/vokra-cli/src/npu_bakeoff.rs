//! Same-machine, same-session CPU/delegate Whisper encoder bakeoff.
//!
//! This command is intentionally separate from the end-to-end ASR benchmark:
//! it measures exactly the submodel delegated to ANE/Hexagon, excludes model
//! loading and log-mel extraction, alternates CPU/delegate order, and compares
//! every hidden-state value against the first-party Rust CPU oracle.

use std::process::ExitCode;

const USAGE: &str = "\
vokra-cli npu-bakeoff — same-session Whisper encoder CPU/delegate gate

USAGE:
    vokra-cli npu-bakeoff --model MODEL.gguf --input AUDIO.wav \\
        --delegate coreml [--warmup N] [--iters N] [--atol F] [--min-speedup F]

DEFAULTS:
    --warmup 2 --iters 10 --atol 0.01 --min-speedup 2.0

The primary performance verdict uses median (p50) CPU/delegate latency. Model
loading, delegate compilation and log-mel extraction are excluded. The command
returns failure when numerical parity or the speed threshold fails.
";

#[derive(Debug, Clone, PartialEq)]
struct Options {
    model: String,
    input: String,
    delegate: String,
    warmup: usize,
    iterations: usize,
    atol: f32,
    min_speedup: f64,
}

fn parse(args: &[String]) -> Result<Option<Options>, String> {
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
    {
        return Ok(None);
    }
    let mut model = None;
    let mut input = None;
    let mut delegate = None;
    let mut warmup = 2usize;
    let mut iterations = 10usize;
    let mut atol = 0.01f32;
    let mut min_speedup = 2.0f64;
    let mut index = 0usize;
    while index < args.len() {
        let flag = &args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("npu-bakeoff: `{flag}` requires a value\n\n{USAGE}"))?;
        match flag.as_str() {
            "--model" => model = Some(value.clone()),
            "--input" => input = Some(value.clone()),
            "--delegate" => delegate = Some(value.clone()),
            "--warmup" => {
                warmup = value
                    .parse()
                    .map_err(|_| format!("npu-bakeoff: invalid --warmup `{value}`"))?;
            }
            "--iters" => {
                iterations = value
                    .parse()
                    .map_err(|_| format!("npu-bakeoff: invalid --iters `{value}`"))?;
            }
            "--atol" => {
                atol = value
                    .parse()
                    .map_err(|_| format!("npu-bakeoff: invalid --atol `{value}`"))?;
            }
            "--min-speedup" => {
                min_speedup = value
                    .parse()
                    .map_err(|_| format!("npu-bakeoff: invalid --min-speedup `{value}`"))?;
            }
            _ => return Err(format!("npu-bakeoff: unknown option `{flag}`\n\n{USAGE}")),
        }
        index += 2;
    }
    if warmup == 0 || iterations == 0 {
        return Err("npu-bakeoff: --warmup and --iters must both be at least 1".to_owned());
    }
    if !atol.is_finite() || atol < 0.0 {
        return Err("npu-bakeoff: --atol must be a finite non-negative number".to_owned());
    }
    if !min_speedup.is_finite() || min_speedup <= 0.0 {
        return Err("npu-bakeoff: --min-speedup must be a finite positive number".to_owned());
    }
    let delegate = delegate.ok_or_else(|| "npu-bakeoff: missing --delegate".to_owned())?;
    if !matches!(delegate.as_str(), "coreml" | "qnn") {
        return Err(format!(
            "npu-bakeoff: --delegate must be `coreml` or `qnn`, got `{delegate}`"
        ));
    }
    Ok(Some(Options {
        model: model.ok_or_else(|| "npu-bakeoff: missing --model".to_owned())?,
        input: input.ok_or_else(|| "npu-bakeoff: missing --input".to_owned())?,
        delegate,
        warmup,
        iterations,
        atol,
        min_speedup,
    }))
}

pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    let Some(options) = parse(args)? else {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    };
    execute(options)
}

#[cfg(feature = "coreml")]
fn execute(options: Options) -> Result<ExitCode, String> {
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_models::whisper::mel::{N_FRAMES, SAMPLE_RATE};
    use vokra_models::whisper::{CoreMlArtifact, CoreMlBackend, WhisperModel};

    if cfg!(debug_assertions) {
        return Err(
            "npu-bakeoff: performance verdicts require an optimized binary; rerun with `cargo run --release -p vokra-cli --features coreml -- npu-bakeoff ...`"
                .to_owned(),
        );
    }

    if options.delegate == "qnn" {
        return Err(
            "npu-bakeoff: QNN whole-Whisper-encoder graph execution is not implemented yet; no CPU fallback was used"
                .to_owned(),
        );
    }

    let wav = crate::wav::read_wav(&options.input)?;
    if wav.sample_rate != SAMPLE_RATE {
        return Err(format!(
            "npu-bakeoff: Whisper requires {SAMPLE_RATE} Hz mono PCM, got {} Hz",
            wav.sample_rate
        ));
    }
    let gguf = vokra_mmap::open_gguf(&options.model).map_err(|error| error.to_string())?;
    let model_arch = gguf
        .get(KEY_MODEL_ARCH)
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("npu-bakeoff: GGUF is missing `{KEY_MODEL_ARCH}`"))?
        .to_owned();
    let model = WhisperModel::from_gguf(&gguf).map_err(|error| error.to_string())?;
    let config = model.config();
    let artifact = CoreMlArtifact::from_whisper_sidecar(
        &options.model,
        &model_arch,
        [1, config.n_mels, N_FRAMES],
        [1, config.n_audio_ctx, config.d_model],
    )
    .map_err(|error| error.to_string())?;
    let log_mel = model.log_mel(&wav.samples);
    let report = CoreMlBackend::with_thread_local_artifact(&artifact, |delegate| {
        model.bakeoff_encoder_delegate(
            delegate,
            &log_mel,
            N_FRAMES,
            options.warmup,
            options.iterations,
            options.atol,
        )
    })
    .map_err(|error| error.to_string())?;

    let cpu = crate::report::summarize(&report.cpu_seconds)
        .ok_or_else(|| "npu-bakeoff: no CPU latency samples".to_owned())?;
    let delegated = crate::report::summarize(&report.delegate_seconds)
        .ok_or_else(|| "npu-bakeoff: no delegate latency samples".to_owned())?;
    if delegated.p50 <= 0.0 || delegated.p95 <= 0.0 {
        return Err(
            "npu-bakeoff: delegate latency was zero; refusing fabricated speedup".to_owned(),
        );
    }
    let p50_speedup = cpu.p50 / delegated.p50;
    let p95_speedup = cpu.p95 / delegated.p95;
    let parity_pass = report.max_abs_error <= options.atol;
    let speed_pass = p50_speedup >= options.min_speedup;
    let pass = parity_pass && speed_pass;

    println!("format=vokra-npu-bakeoff-v1");
    println!("delegate={}", report.delegate_name);
    println!("model_arch={model_arch}");
    println!("build_profile=release");
    println!(
        "source_gguf_sha256={}",
        artifact.source_gguf_sha256().ok_or_else(|| {
            "npu-bakeoff: verified artifact lost its source digest binding".to_owned()
        })?
    );
    println!(
        "compiled_tree_sha256={}",
        artifact.compiled_tree_sha256().ok_or_else(|| {
            "npu-bakeoff: verified artifact lost its tree digest binding".to_owned()
        })?
    );
    println!("same_process=true");
    println!("same_model_instance=true");
    println!("same_delegate_session=true");
    println!("same_input_features=true");
    println!("timed_scope=whisper_encoder_only");
    println!("cpu_baseline=first_party_rust_cpu");
    println!("warmup={}", options.warmup);
    println!("iterations={}", options.iterations);
    println!("compared_values={}", report.compared_values);
    println!("atol={:.9}", options.atol);
    println!("max_abs_error={:.9}", report.max_abs_error);
    println!("mean_abs_error={:.9}", report.mean_abs_error);
    println!("values_over_atol={}", report.values_over_atol);
    println!("max_error_iteration={}", report.max_error_iteration);
    println!("max_error_index={}", report.max_error_index);
    println!(
        "max_error_audio_position={}",
        report.max_error_index / config.d_model
    );
    println!(
        "max_error_hidden_channel={}",
        report.max_error_index % config.d_model
    );
    println!("max_error_cpu_value={:.9}", report.max_error_cpu_value);
    println!(
        "max_error_delegate_value={:.9}",
        report.max_error_delegate_value
    );
    println!(
        "parity_verdict={}",
        if parity_pass { "PASS" } else { "FAIL" }
    );
    println!("cpu_p50_ms={:.6}", cpu.p50 * 1_000.0);
    println!("cpu_p95_ms={:.6}", cpu.p95 * 1_000.0);
    println!("cpu_mean_ms={:.6}", cpu.mean * 1_000.0);
    println!("cpu_stddev_ms={:.6}", cpu.stddev * 1_000.0);
    println!("cpu_cv={:.9}", cpu.stddev / cpu.mean);
    println!("delegate_p50_ms={:.6}", delegated.p50 * 1_000.0);
    println!("delegate_p95_ms={:.6}", delegated.p95 * 1_000.0);
    println!("delegate_mean_ms={:.6}", delegated.mean * 1_000.0);
    println!("delegate_stddev_ms={:.6}", delegated.stddev * 1_000.0);
    println!("delegate_cv={:.9}", delegated.stddev / delegated.mean);
    println!("speedup_basis=p50");
    println!("speedup_p50={p50_speedup:.6}");
    println!("speedup_p95={p95_speedup:.6}");
    println!("minimum_speedup={:.6}", options.min_speedup);
    println!("speed_verdict={}", if speed_pass { "PASS" } else { "FAIL" });
    println!("verdict={}", if pass { "PASS" } else { "FAIL" });
    Ok(if pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

#[cfg(not(feature = "coreml"))]
fn execute(options: Options) -> Result<ExitCode, String> {
    Err(format!(
        "npu-bakeoff: delegate `{}` is unavailable because vokra-cli was built without `--features coreml`; no CPU fallback was used",
        options.delegate
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_defaults_and_rejects_zero_measurements() {
        let parsed = parse(&strings(&[
            "--model",
            "model.gguf",
            "--input",
            "audio.wav",
            "--delegate",
            "coreml",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.warmup, 2);
        assert_eq!(parsed.iterations, 10);
        assert_eq!(parsed.atol, 0.01);
        assert_eq!(parsed.min_speedup, 2.0);

        let err = parse(&strings(&[
            "--model",
            "model.gguf",
            "--input",
            "audio.wav",
            "--delegate",
            "coreml",
            "--iters",
            "0",
        ]))
        .unwrap_err();
        assert!(err.contains("at least 1"));
    }
}
