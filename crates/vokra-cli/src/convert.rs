//! `vokra-cli convert` — delegate to the offline `vokra-convert` library (M1-10a).
//!
//! A thin front-end over the first-party `vokra-convert` crate so the umbrella
//! CLI can drive the offline checkpoint → GGUF conversion without duplicating
//! its logic. The standalone `vokra-convert` binary is kept (it is the
//! dependency-isolation boundary for the ONNX/protobuf handling); this delegate
//! just re-exposes the same `--model/--input/--config/--output/--quantize`
//! surface and calls the library entry points.

use std::path::PathBuf;
use std::process::ExitCode;

use vokra_convert::{ModelKind, convert_file, convert_file_quantized, convert_piper_plus_file};
use vokra_core::gguf::GgmlType;

pub(crate) const USAGE: &str = "\
vokra-cli convert — convert an upstream checkpoint to Vokra GGUF (offline tool)

USAGE:
    vokra-cli convert --model <whisper-base|silero-vad> --input <ckpt> --output <out.gguf>
    vokra-cli convert --model piper-plus --input <voice.onnx> --config <config.json> --output <out.gguf>

OPTIONS:
    --model <kind>     whisper-base | silero-vad | piper-plus
    --input <path>     upstream checkpoint file
    --config <path>    piper-plus config.json (piper-plus only)
    --output <path>    GGUF file to write
    --quantize <kind>  K-quantize weight matrices: q4_k | q5_k | q6_k (whisper-base only)
    -h, --help         print this help
";

/// Parsed `convert` arguments.
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
                    format!("unknown model `{v}` (whisper-base | silero-vad | piper-plus)")
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

/// Entry point for `vokra-cli convert`.
pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let p = parse_args(args)?;
    let model = p.model; // ModelKind is Copy; reused after the move into convert_*.

    let result = match model {
        ModelKind::PiperPlus => {
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper-base".to_owned());
            }
            match &p.config {
                Some(config) => convert_piper_plus_file(&p.input, config, &p.output),
                None => {
                    return Err("--model piper-plus requires --config <config.json>".to_owned());
                }
            }
        }
        _ => match p.quant {
            Some(q) => convert_file_quantized(model, &p.input, &p.output, q),
            None => convert_file(model, &p.input, &p.output),
        },
    };

    match result {
        Ok(summary) => {
            println!(
                "converted {model}: {} tensors, {} metadata keys, {} bytes -> {}",
                summary.tensor_count,
                summary.metadata_count,
                summary.output_bytes,
                p.output.display()
            );
            for note in &summary.notes {
                println!("  note: {note}");
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    fn err_of(r: Result<Parsed, String>) -> String {
        match r {
            Ok(_) => panic!("expected parse_args to fail"),
            Err(e) => e,
        }
    }

    #[test]
    fn parses_whisper_with_quantize() {
        let p = parse_args(&args(&[
            "--model",
            "whisper-base",
            "--input",
            "i",
            "--output",
            "o",
            "--quantize",
            "q5_k",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::Whisper);
        assert_eq!(p.input, PathBuf::from("i"));
        assert_eq!(p.output, PathBuf::from("o"));
        assert_eq!(p.quant, Some(GgmlType::Q5K));
    }

    #[test]
    fn parses_piper_plus_with_config() {
        let p = parse_args(&args(&[
            "--model",
            "piper-plus",
            "--input",
            "v.onnx",
            "--config",
            "c.json",
            "--output",
            "o",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::PiperPlus);
        assert_eq!(p.config, Some(PathBuf::from("c.json")));
    }

    #[test]
    fn rejects_unknown_model_and_quant_and_missing_fields() {
        assert!(
            err_of(parse_args(&args(&[
                "--model", "bogus", "--input", "i", "--output", "o"
            ])))
            .contains("unknown model")
        );
        assert!(
            err_of(parse_args(&args(&[
                "--model",
                "whisper-base",
                "--input",
                "i",
                "--output",
                "o",
                "--quantize",
                "q9_k",
            ])))
            .contains("unknown --quantize")
        );
        assert_eq!(
            err_of(parse_args(&args(&["--input", "i", "--output", "o"]))),
            "--model is required"
        );
        assert_eq!(
            err_of(parse_args(&args(&["--model"]))),
            "--model requires a value"
        );
    }
}
