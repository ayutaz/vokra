//! `vokra-convert` command-line entry point (M0-03, FR-TL-01).
//!
//! ```text
//! vokra-convert --model <whisper|silero-vad> --input <ckpt> --output <out.gguf>
//! ```
//!
//! `whisper` auto-detects the size (base / small / medium / large-v3 / turbo)
//! from the checkpoint tensor shapes (M2-06-T06); `whisper-base` is kept as a
//! backward-compatible alias.
//!
//! After writing the GGUF, the tool re-opens it with the runtime loader and
//! prints a verification line, giving direct evidence that the output is
//! mmap-loadable and that its `vokra.*` chunks read back (the M0-03-T13 /
//! M0-03-T16 local-run checks).

use std::path::PathBuf;
use std::process::ExitCode;

use vokra_convert::{ModelKind, convert_file, convert_file_quantized, convert_piper_plus_file};
use vokra_core::gguf::{FrontendSpec, GgmlType, GgufFile};

const USAGE: &str = "\
vokra-convert — convert an upstream checkpoint to Vokra GGUF (M0-03, FR-TL-01)

USAGE:
    vokra-convert --model <whisper|silero-vad|campplus|kokoro> --input <checkpoint> --output <out.gguf>
    vokra-convert --model piper-plus --input <voice.onnx> --config <config.json> --output <out.gguf>

OPTIONS:
    --model <kind>     whisper (safetensors; size auto-detected from
                       checkpoint tensor shapes: base/small/medium/large-v3/
                       turbo — unknown shapes error out, no silent fallback
                       per FR-EX-08), silero-vad (ONNX), campplus (CAM++
                       speaker-encoder ONNX), kokoro (Kokoro-82M StyleTTS 2
                       派生 iSTFTNet safetensors) or piper-plus (MB-iSTFT-VITS2
                       voice: ONNX + config.json). `whisper-base` is accepted
                       as a backward-compatible alias for `whisper` (size is
                       still derived from the checkpoint, not the flag).
    --input <path>     upstream checkpoint file
    --config <path>    piper-plus config.json (piper-plus only)
    --output <path>    GGUF file to write
    --quantize <kind>  K-quantize large weight matrices: q4_k | q5_k | q6_k
                       (whisper only; biases/norms stay F32)
    -h, --help         print this help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let parsed = match parse_args(&args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("error: {msg}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let Parsed {
        model,
        input,
        config,
        output,
        quant,
    } = parsed;

    let result = match model {
        ModelKind::PiperPlus => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            match &config {
                Some(config) => convert_piper_plus_file(&input, config, &output),
                None => {
                    eprintln!(
                        "error: --model piper-plus requires --config <config.json>\n\n{USAGE}"
                    );
                    return ExitCode::from(2);
                }
            }
        }
        _ => match quant {
            Some(q) => convert_file_quantized(model, &input, &output, q),
            None => convert_file(model, &input, &output),
        },
    };

    match result {
        Ok(summary) => {
            println!(
                "converted {model}: {} tensors, {} metadata keys, {} bytes -> {}",
                summary.tensor_count,
                summary.metadata_count,
                summary.output_bytes,
                output.display()
            );
            for note in &summary.notes {
                println!("  note: {note}");
            }
            if let Err(code) = verify(model, &output) {
                return code;
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Parsed {
    model: ModelKind,
    input: PathBuf,
    config: Option<PathBuf>,
    output: PathBuf,
    quant: Option<GgmlType>,
}

/// Parses the `--quantize` argument into a K-quant target dtype.
fn parse_quant(s: &str) -> Option<GgmlType> {
    match s {
        "q4_k" | "q4k" => Some(GgmlType::Q4K),
        "q5_k" | "q5k" => Some(GgmlType::Q5K),
        "q6_k" | "q6k" => Some(GgmlType::Q6K),
        _ => None,
    }
}

fn parse_args(args: &[String]) -> Result<Parsed, String> {
    let mut model: Option<ModelKind> = None;
    let mut input: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut quant: Option<GgmlType> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let v = args.get(i + 1).ok_or("--model requires a value")?;
                model = Some(ModelKind::from_arg(v).ok_or_else(|| {
                    format!(
                        "unknown model `{v}` (whisper [alias: whisper-base] | silero-vad | piper-plus | campplus | kokoro)"
                    )
                })?);
                i += 2;
            }
            "--input" => {
                input = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--input requires a value")?,
                ));
                i += 2;
            }
            "--config" => {
                config = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--config requires a value")?,
                ));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--output requires a value")?,
                ));
                i += 2;
            }
            "--quantize" => {
                let v = args.get(i + 1).ok_or("--quantize requires a value")?;
                quant = Some(
                    parse_quant(v)
                        .ok_or_else(|| format!("unknown --quantize `{v}` (q4_k | q5_k | q6_k)"))?,
                );
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    Ok(Parsed {
        model: model.ok_or("--model is required")?,
        input: input.ok_or("--input is required")?,
        config,
        output: output.ok_or("--output is required")?,
        quant,
    })
}

/// Re-opens the produced GGUF through the runtime loader and prints a
/// verification line. Returns `Err(code)` if the output does not load.
fn verify(model: ModelKind, output: &PathBuf) -> Result<(), ExitCode> {
    let file = match GgufFile::open(output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: output GGUF failed to load back: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    print!(
        "verified load: version {}, alignment {}, {} tensors, {} metadata keys",
        file.version(),
        file.alignment(),
        file.tensors().len(),
        file.metadata().len()
    );
    match model {
        ModelKind::Whisper => match FrontendSpec::from_gguf(&file) {
            Ok(spec) => println!(
                "; frontend n_fft={} hop={} n_mels={} sample_rate={}",
                spec.n_fft, spec.hop, spec.n_mels, spec.sample_rate
            ),
            Err(e) => {
                println!();
                eprintln!("error: frontend_spec did not read back: {e}");
                return Err(ExitCode::FAILURE);
            }
        },
        ModelKind::SileroVad => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!("; arch={arch}");
        }
        ModelKind::PiperPlus => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.piper.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_sym = file
                .get("vokra.piper.num_symbols")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("; arch={arch} sample_rate={sr} num_symbols={n_sym}");
        }
        ModelKind::CamPlus => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let embed = file
                .get("vokra.campplus.embed_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let blocks = file
                .get("vokra.campplus.block_config")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.values
                        .iter()
                        .filter_map(|v| v.as_u64())
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            println!("; arch={arch} embed_dim={embed} block_config=[{blocks}]");
        }
        ModelKind::Kokoro => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.kokoro.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let style_dim = file
                .get("vokra.kokoro.style_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let num_voices = file
                .get("vokra.kokoro.num_voices")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} sample_rate={sr} style_dim={style_dim} num_voices={num_voices}"
            );
        }
        ModelKind::CosyVoice2 => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.cosyvoice2.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.cosyvoice2.arch.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.cosyvoice2.arch.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden_dim = file
                .get("vokra.cosyvoice2.arch.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} sample_rate={sr} n_layer={n_layer} n_head={n_head} \
                 hidden_dim={hidden_dim}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns a `&str` slice into the owned `Vec<String>` `parse_args` expects.
    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Extracts the error message from a `parse_args` result (`Parsed` is not
    /// `Debug`, so `unwrap_err` cannot be used directly).
    fn err_of(r: Result<Parsed, String>) -> String {
        match r {
            Ok(_) => panic!("expected parse_args to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_full_valid_invocation() {
        let parsed = parse_args(&args(&[
            "--model", "whisper", "--input", "i", "--output", "o",
        ]))
        .expect("valid args");
        assert_eq!(parsed.model, ModelKind::Whisper);
        assert_eq!(parsed.input, PathBuf::from("i"));
        assert_eq!(parsed.output, PathBuf::from("o"));
        assert_eq!(parsed.config, None);
        assert_eq!(parsed.quant, None);
    }

    /// The legacy `whisper-base` label from pre-M2-06 CLI invocations must
    /// keep dispatching to the same size-detecting path as the canonical
    /// `whisper`. Both should resolve to `ModelKind::Whisper` (M2-06-T06).
    #[test]
    fn whisper_base_alias_dispatches_to_same_kind_as_whisper() {
        let via_whisper = parse_args(&args(&[
            "--model", "whisper", "--input", "i", "--output", "o",
        ]))
        .expect("valid args (whisper)");
        let via_alias = parse_args(&args(&[
            "--model",
            "whisper-base",
            "--input",
            "i",
            "--output",
            "o",
        ]))
        .expect("valid args (whisper-base alias)");
        assert_eq!(via_whisper.model, ModelKind::Whisper);
        assert_eq!(via_alias.model, ModelKind::Whisper);
        assert_eq!(via_whisper.model, via_alias.model);
    }

    #[test]
    fn parses_quantize_flag() {
        let parsed = parse_args(&args(&[
            "--model",
            "whisper-base",
            "--input",
            "i",
            "--output",
            "o",
            "--quantize",
            "q5_k",
        ]))
        .expect("valid args");
        assert_eq!(parsed.quant, Some(GgmlType::Q5K));
    }

    #[test]
    fn rejects_unknown_quantize_value() {
        let err = err_of(parse_args(&args(&[
            "--model",
            "whisper-base",
            "--input",
            "i",
            "--output",
            "o",
            "--quantize",
            "q9_k",
        ])));
        assert!(err.contains("unknown --quantize"), "got: {err}");
    }

    #[test]
    fn parses_piper_plus_with_config() {
        let parsed = parse_args(&args(&[
            "--model",
            "piper-plus",
            "--input",
            "v.onnx",
            "--config",
            "c.json",
            "--output",
            "o",
        ]))
        .expect("valid piper args");
        assert_eq!(parsed.model, ModelKind::PiperPlus);
        assert_eq!(parsed.config, Some(PathBuf::from("c.json")));
    }

    #[test]
    fn rejects_unknown_model() {
        let err = err_of(parse_args(&args(&[
            "--model", "bogus", "--input", "i", "--output", "o",
        ])));
        assert!(err.contains("unknown model"), "got: {err}");
    }

    #[test]
    fn rejects_flag_without_value() {
        let err = err_of(parse_args(&args(&["--model"])));
        assert_eq!(err, "--model requires a value");
    }

    #[test]
    fn rejects_unexpected_argument() {
        let err = err_of(parse_args(&args(&["--stray"])));
        assert!(err.contains("unexpected argument"), "got: {err}");
    }

    #[test]
    fn requires_each_mandatory_field() {
        // Missing --model (present --input/--output).
        assert_eq!(
            err_of(parse_args(&args(&["--input", "i", "--output", "o"]))),
            "--model is required"
        );
        // Missing --input.
        assert_eq!(
            err_of(parse_args(&args(&[
                "--model",
                "whisper-base",
                "--output",
                "o"
            ]))),
            "--input is required"
        );
        // Missing --output.
        assert_eq!(
            err_of(parse_args(&args(&[
                "--model",
                "whisper-base",
                "--input",
                "i"
            ]))),
            "--output is required"
        );
    }
}
