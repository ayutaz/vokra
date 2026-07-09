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

use vokra_convert::{
    ModelKind, PolicyPreset, VoxtralConfig, convert_file, convert_file_quantized,
    convert_file_with_policy, convert_kokoro_file, convert_piper_plus_file,
    convert_voxtral_file_with_adapter_config,
};
use vokra_core::gguf::GgmlType;

pub(crate) const USAGE: &str = "\
vokra-cli convert — convert an upstream checkpoint to Vokra GGUF (offline tool)

USAGE:
    vokra-cli convert --model <whisper-base|silero-vad|campplus> --input <ckpt> --output <out.gguf>
    vokra-cli convert --model piper-plus --input <voice.onnx> --config <config.json> --output <out.gguf>
    vokra-cli convert --model kokoro --input <ckpt.safetensors> [--config <config.json>] --output <out.gguf>
    vokra-cli convert --model voxtral --input <ckpt.safetensors> [--adapter-config <adapter.json>] --output <out.gguf>

OPTIONS:
    --model <kind>            whisper-base | silero-vad | piper-plus | campplus | kokoro | voxtral
    --input <path>            upstream checkpoint file
    --config <path>           piper-plus config.json (piper-plus only) OR Kokoro
                              config.json (misaki phoneme symbols + voice names;
                              omit to emit the p0..p_{n-1} placeholder table)
    --adapter-config <path>   Voxtral audio-adapter side-car JSON (M3-10 Wave 8):
                              writes `vokra.voxtral.adapter.*` metadata so the
                              runtime binds the checkpoint's adapter tensors
                              and routes ASR through the audio-conditioned
                              soft-prefix path (see docs/tickets/m3/M3-10*.md).
                              Omit for the honest LM-continuation path.
    --output <path>           GGUF file to write
    --quantize <kind>         K-quantize weight matrices: q4_k | q5_k | q6_k (whisper only)
                              Alias for --policy-preset whisper_q4_k (when kind=q4_k).
    --policy-preset <preset>  M2-08 quantization policy preset (whisper only):
                              vocoder_safe (default) | whisper_q4_k | fp16
    -h, --help                print this help
";

/// Parsed `convert` arguments.
struct Parsed {
    model: ModelKind,
    input: PathBuf,
    config: Option<PathBuf>,
    /// M3-10 Wave 8 — Voxtral only. When present, `convert` routes through
    /// [`convert_voxtral_file_with_adapter_config`] and emits the adapter
    /// metadata chunk into the GGUF so the runtime binds real adapter tensors
    /// and does audio-conditioned ASR.
    adapter_config: Option<PathBuf>,
    output: PathBuf,
    quant: Option<GgmlType>,
    policy: Option<PolicyPreset>,
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
    let mut adapter_config: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut quant: Option<GgmlType> = None;
    let mut policy: Option<PolicyPreset> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let v = args.get(i + 1).ok_or("--model requires a value")?;
                model = Some(ModelKind::from_arg(v).ok_or_else(|| {
                    format!(
                        "unknown model `{v}` \
                         (whisper-base | silero-vad | piper-plus | campplus | kokoro | voxtral)"
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
            "--adapter-config" => {
                adapter_config = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--adapter-config requires a value")?,
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
            "--policy-preset" => {
                let v = args.get(i + 1).ok_or("--policy-preset requires a value")?;
                policy = Some(PolicyPreset::from_arg(v).ok_or_else(|| {
                    format!("unknown --policy-preset `{v}` (vocoder_safe | whisper_q4_k | fp16)")
                })?);
                i += 2;
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    if quant.is_some() && policy.is_some() {
        return Err("--quantize and --policy-preset are mutually exclusive".to_owned());
    }

    Ok(Parsed {
        model: model.ok_or("--model is required")?,
        input: input.ok_or("--input is required")?,
        config,
        adapter_config,
        output: output.ok_or("--output is required")?,
        quant,
        policy,
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
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            match &p.config {
                Some(config) => convert_piper_plus_file(&p.input, config, &p.output),
                None => {
                    return Err("--model piper-plus requires --config <config.json>".to_owned());
                }
            }
        }
        ModelKind::Kokoro => {
            // Kokoro is whisper-only for quantization surface in M2-08 (T06);
            // reject the flag rather than silently ignoring it.
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            match &p.config {
                // Real misaki phoneme table + voice list wired in.
                Some(config) => convert_kokoro_file(&p.input, config, &p.output),
                // Backward-compatible placeholder path: emits `p{i}` symbols
                // and an empty voice_names array (matches the M2-07 T06
                // roundtrip test contract).
                None => convert_file(model, &p.input, &p.output),
            }
        }
        ModelKind::Voxtral => {
            // Voxtral is whisper-only for quantization surface; reject rather
            // than silently ignoring.
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            match (&p.config, &p.adapter_config) {
                // M3-10 Wave 8: adapter-conditioned convert. The base config
                // JSON path is currently unused for Voxtral (VoxtralConfig is
                // populated by the future --side-car path in T04 follow-up);
                // for now, the audio-adapter path uses an empty base config
                // plus the adapter JSON side-car. When --side-car lands, the
                // Voxtral path will merge both.
                (_, Some(adapter_json)) => convert_voxtral_file_with_adapter_config(
                    &p.input,
                    &VoxtralConfig::default(),
                    adapter_json,
                    &p.output,
                ),
                // No adapter → shape-only conversion (honest LM-continuation
                // posture, Wave 7 semantic). Same behavior as the pre-Wave-8
                // path.
                (_, None) => convert_file(model, &p.input, &p.output),
            }
        }
        _ => {
            // Ticket precedence: an explicit --policy-preset wins; else the
            // legacy --quantize q4_k alias maps to the whisper_q4_k preset;
            // else fall through to convert_file_quantized (Q5/Q6 legacy
            // shapes) or the plain byte-exact path.
            if let Some(preset) = p.policy {
                convert_file_with_policy(model, &p.input, &p.output, preset)
            } else if let Some(q) = p.quant {
                if q == GgmlType::Q4K {
                    // Backward-compat alias per T06 spec.
                    convert_file_with_policy(model, &p.input, &p.output, PolicyPreset::WhisperQ4K)
                } else {
                    convert_file_quantized(model, &p.input, &p.output, q)
                }
            } else {
                convert_file(model, &p.input, &p.output)
            }
        }
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
    fn parses_kokoro_with_config() {
        // Config-driven Kokoro path (M2-07-T17-fixup #3): the CLI accepts
        // `--config <path>.json` so the misaki phoneme table + voice list get
        // wired into the emitted GGUF verbatim. The plain `--input`-only path
        // still works (see the placeholder-path roundtrip test).
        let p = parse_args(&args(&[
            "--model",
            "kokoro",
            "--input",
            "kokoro.safetensors",
            "--config",
            "c.json",
            "--output",
            "o.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::Kokoro);
        assert_eq!(p.input, PathBuf::from("kokoro.safetensors"));
        assert_eq!(p.config, Some(PathBuf::from("c.json")));
        assert_eq!(p.output, PathBuf::from("o.gguf"));
        assert!(p.quant.is_none());
        assert!(p.policy.is_none());
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

    #[test]
    fn parses_voxtral_with_adapter_config() {
        // M3-10 Wave 8: the voxtral path accepts an `--adapter-config
        // adapter.json` argument that, at run time, emits the
        // `vokra.voxtral.adapter.*` metadata chunk so the runtime binds real
        // adapter tensors and does audio-conditioned ASR.
        let p = parse_args(&args(&[
            "--model",
            "voxtral",
            "--input",
            "voxtral.safetensors",
            "--adapter-config",
            "adapter.json",
            "--output",
            "voxtral.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::Voxtral);
        assert_eq!(p.input, PathBuf::from("voxtral.safetensors"));
        assert_eq!(p.adapter_config, Some(PathBuf::from("adapter.json")));
        assert_eq!(p.output, PathBuf::from("voxtral.gguf"));
    }

    #[test]
    fn parses_voxtral_without_adapter_config_is_ok() {
        // No `--adapter-config` → shape-only convert path (honest
        // LM-continuation Wave 7 posture).
        let p = parse_args(&args(&[
            "--model",
            "voxtral",
            "--input",
            "voxtral.safetensors",
            "--output",
            "voxtral.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::Voxtral);
        assert!(p.adapter_config.is_none());
    }

    #[test]
    fn adapter_config_requires_value() {
        assert!(
            err_of(parse_args(&args(&[
                "--model",
                "voxtral",
                "--input",
                "i",
                "--adapter-config",
            ])))
            .contains("--adapter-config requires a value")
        );
    }
}
