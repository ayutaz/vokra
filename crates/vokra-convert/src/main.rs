//! `vokra-convert` command-line entry point (M0-03, FR-TL-01).
//!
//! ```text
//! vokra-convert --model <whisper-base|silero-vad> --input <ckpt> --output <out.gguf>
//! ```
//!
//! After writing the GGUF, the tool re-opens it with the runtime loader and
//! prints a verification line, giving direct evidence that the output is
//! mmap-loadable and that its `vokra.*` chunks read back (the M0-03-T13 /
//! M0-03-T16 local-run checks).

use std::path::PathBuf;
use std::process::ExitCode;

use vokra_convert::{ModelKind, convert_file};
use vokra_core::gguf::{FrontendSpec, GgufFile};

const USAGE: &str = "\
vokra-convert — convert an upstream checkpoint to Vokra GGUF (M0-03, FR-TL-01)

USAGE:
    vokra-convert --model <whisper-base|silero-vad> --input <checkpoint> --output <out.gguf>

OPTIONS:
    --model <kind>     whisper-base (safetensors) or silero-vad (ONNX)
    --input <path>     upstream checkpoint file
    --output <path>    GGUF file to write
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
    let (model, input, output) = parsed;

    match convert_file(model, &input, &output) {
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

type ParsedArgs = (ModelKind, PathBuf, PathBuf);

fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut model: Option<ModelKind> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let v = args.get(i + 1).ok_or("--model requires a value")?;
                model =
                    Some(ModelKind::from_arg(v).ok_or_else(|| {
                        format!("unknown model `{v}` (whisper-base | silero-vad)")
                    })?);
                i += 2;
            }
            "--input" => {
                input = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--input requires a value")?,
                ));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--output requires a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    Ok((
        model.ok_or("--model is required")?,
        input.ok_or("--input is required")?,
        output.ok_or("--output is required")?,
    ))
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
        ModelKind::WhisperBase => match FrontendSpec::from_gguf(&file) {
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
    }
    Ok(())
}
