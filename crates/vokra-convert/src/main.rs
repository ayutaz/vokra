//! `vokra-convert` command-line entry point (M0-03, FR-TL-01).
//!
//! ```text
//! vokra-convert --model <whisper|silero-vad|piper-plus|campplus|kokoro|cosyvoice2|voxtral|mimi|dac|csm|moshi|denoise>
//!               --input <ckpt> [--config <side-car>] --output <out.gguf>
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

use vokra_convert::{
    ModelKind, convert_cosyvoice2_file, convert_csm_file, convert_dac_file, convert_file_licensed,
    convert_file_quantized, convert_moshi_file, convert_piper_plus_file, convert_utmos_file,
};
use vokra_core::gguf::{FrontendSpec, GgmlType};

const USAGE: &str = "\
vokra-convert — convert an upstream checkpoint to Vokra GGUF (M0-03, FR-TL-01)

USAGE:
    vokra-convert --model <whisper|silero-vad|campplus|kokoro|voxtral|mimi|denoise> --input <checkpoint> --output <out.gguf>
    vokra-convert --model piper-plus --input <voice.onnx> --config <config.json> --output <out.gguf>
    vokra-convert --model dac --input <prepared.safetensors> --config <config.json> --output <out.gguf>
    vokra-convert --model utmos --input <prepared.safetensors> --config <config.json> --output <out.gguf>
    vokra-convert --model <cosyvoice2|csm|moshi> --input <ckpt.safetensors> [--config <side-car>] --output <out.gguf>

OPTIONS:
    --model <kind>     whisper (safetensors; size auto-detected from
                       checkpoint tensor shapes: base/small/medium/large-v3/
                       turbo — unknown shapes error out, no silent fallback
                       per FR-EX-08), silero-vad (ONNX), campplus (CAM++
                       speaker-encoder ONNX), kokoro (Kokoro-82M StyleTTS 2
                       派生 iSTFTNet safetensors), piper-plus (MB-iSTFT-VITS2
                       voice: ONNX + config.json), cosyvoice2 (CosyVoice2-0.5B
                       LLM safetensors), voxtral (Mistral Voxtral safetensors;
                       shape-only here — the config-aware / adapter path is
                       `vokra-cli convert`), mimi (Kyutai Mimi codec
                       safetensors), dac (prepared DAC safetensors +
                       config.json), csm (Sesame CSM-1B safetensors) or
                       moshi (Kyutai Moshi safetensors). `whisper-base` is
                       accepted as a backward-compatible alias for `whisper`
                       (size is still derived from the checkpoint, not the
                       flag).
    --input <path>     upstream checkpoint file
    --config <path>    piper-plus config.json (piper-plus, required) OR the
                       DAC prepare-script config.json (dac, required — from
                       tools/parity/dac_prepare_checkpoint.py) OR the
                       upstream HF config.json for cosyvoice2 (Qwen2
                       schema; supplies the attention head split +
                       rope_theta/rms_norm_eps/n_ctx that tensor shapes
                       cannot determine) OR the raw Llama-3.2 tokenizer
                       file (csm; optional — without it the runtime text
                       path fails loudly) OR the raw SentencePiece
                       tokenizer file (moshi; optional — without it the
                       monologue decode fails loudly)
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
        license,
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
        ModelKind::Dac => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            match &config {
                Some(config) => convert_dac_file(&input, config, &output),
                None => {
                    eprintln!(
                        "error: --model dac requires --config <config.json> (from \
                         tools/parity/dac_prepare_checkpoint.py)\n\n{USAGE}"
                    );
                    return ExitCode::from(2);
                }
            }
        }
        ModelKind::Utmos => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            match &config {
                Some(config) => convert_utmos_file(&input, config, &output),
                None => {
                    eprintln!(
                        "error: --model utmos requires --config <config.json> (from \
                         tools/parity/utmos_prepare_checkpoint.py)\n\n{USAGE}"
                    );
                    return ExitCode::from(2);
                }
            }
        }
        ModelKind::Csm => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // --config carries the raw Llama-3.2 tokenizer file (optional —
            // the repo is gated, T29; without it the runtime text path
            // fails loudly, M4-05-T05).
            convert_csm_file(&input, config.as_deref(), &output)
        }
        ModelKind::Moshi => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // --config carries the raw SentencePiece tokenizer file
            // (tokenizer_spm_32k_3.model — public in the kyutai repo;
            // without it the monologue decode fails loudly, M4-06-T22).
            convert_moshi_file(&input, config.as_deref(), &output)
        }
        ModelKind::CosyVoice2 => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // --config carries the upstream HF config.json (Qwen2 schema).
            // Optional: without it only the shape-derived hparams are
            // written and the runtime refuses the LLM bind (loud note).
            convert_cosyvoice2_file(&input, config.as_deref(), &output)
        }
        _ => match quant {
            Some(q) => convert_file_quantized(model, &input, &output, q),
            None => convert_file_licensed(model, &input, &output, license.as_deref()),
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
    license: Option<String>,
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
    let mut license: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let v = args.get(i + 1).ok_or("--model requires a value")?;
                model = Some(ModelKind::from_arg(v).ok_or_else(|| {
                    format!(
                        "unknown model `{v}` (whisper [alias: whisper-base] | silero-vad | \
                         piper-plus | campplus | kokoro | cosyvoice2 | voxtral | mimi | \
                         dac | csm | moshi | denoise)"
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
            "--license" => {
                license = Some(
                    args.get(i + 1)
                        .ok_or("--license requires an SPDX id")?
                        .clone(),
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
        license,
    })
}

/// Re-opens the produced GGUF through the runtime loader and prints a
/// verification line. Returns `Err(code)` if the output does not load.
///
/// Opens through the true-mmap loader (`vokra_mmap::open_gguf`) so the
/// verify pass touches only the header/metadata pages — verifying a
/// multi-GiB output (the 14 GiB Moshi full-7B GGUF) stays within the
/// streaming converter's bounded-memory contract instead of re-reading
/// the whole file into an owned buffer (M4 cc-06).
fn verify(model: ModelKind, output: &PathBuf) -> Result<(), ExitCode> {
    let file = match vokra_mmap::open_gguf(output) {
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
        ModelKind::Utmos => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let variant = file
                .get("vokra.utmos.arch.variant")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.utmos.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("; arch={arch} variant={variant} sample_rate={sr}");
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
        ModelKind::Voxtral => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let ae_n_layer = file
                .get("vokra.voxtral.audio_encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let td_n_layer = file
                .get("vokra.voxtral.text_decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.voxtral.text_decoder.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let mode = file
                .get("vokra.voxtral.mode")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} audio_layers={ae_n_layer} text_layers={td_n_layer} vocab={vocab} mode={mode}"
            );
        }
        ModelKind::Mimi => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_cb = file
                .get("vokra.mimi.n_codebooks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cb_size = file
                .get("vokra.mimi.codebook_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.mimi.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("; arch={arch} n_codebooks={n_cb} codebook_size={cb_size} d_model={d_model}");
        }
        ModelKind::Csm => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let bb_layers = file
                .get("vokra.csm.arch.backbone.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dt_layers = file
                .get("vokra.csm.arch.depth.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_cb = file
                .get("vokra.csm.audio.n_codebooks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let audio_vocab = file
                .get("vokra.csm.audio.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} backbone_layers={bb_layers} depth_layers={dt_layers} \
                 n_codebooks={n_cb} audio_vocab={audio_vocab}"
            );
        }
        ModelKind::Moshi => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let tm_layers = file
                .get("vokra.moshi.arch.temporal.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dt_layers = file
                .get("vokra.moshi.arch.depth.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_q_in = file
                .get("vokra.moshi.audio.n_q_in")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dep_q = file
                .get("vokra.moshi.audio.dep_q")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let attribution = file
                .get("vokra.provenance.attribution")
                .and_then(|v| v.as_str())
                .map(|_| "present")
                .unwrap_or("ABSENT");
            println!(
                "; arch={arch} temporal_layers={tm_layers} depth_layers={dt_layers} \
                 n_q_in={n_q_in} dep_q={dep_q} attribution={attribution}"
            );
        }
        ModelKind::Denoise => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_fft = file
                .get("vokra.denoise.n_fft")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_erb = file
                .get("vokra.denoise.n_erb")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let df_bins = file
                .get("vokra.denoise.df_bins")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let df_order = file
                .get("vokra.denoise.df_order")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} n_fft={n_fft} n_erb={n_erb} df_bins={df_bins} df_order={df_order}"
            );
        }
        ModelKind::Dac => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_cb = file
                .get("vokra.dac.n_codebooks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cb_dim = file
                .get("vokra.dac.codebook_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.dac.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.dac.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} n_codebooks={n_cb} codebook_dim={cb_dim} d_model={d_model} \
                 sample_rate={sr}"
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

    /// Campaign-1 P3 #11 (campaign-2 cli-enablers Fix B): every kind
    /// `ModelKind::from_arg` accepts parses through the standalone binary,
    /// and the help text lists each one. No new kinds are added.
    #[test]
    fn parses_every_model_kind_and_help_lists_them() {
        let kinds: &[(&str, ModelKind)] = &[
            ("whisper", ModelKind::Whisper),
            ("whisper-base", ModelKind::Whisper),
            ("silero-vad", ModelKind::SileroVad),
            ("piper-plus", ModelKind::PiperPlus),
            ("campplus", ModelKind::CamPlus),
            ("kokoro", ModelKind::Kokoro),
            ("cosyvoice2", ModelKind::CosyVoice2),
            ("voxtral", ModelKind::Voxtral),
            ("mimi", ModelKind::Mimi),
            ("dac", ModelKind::Dac),
            ("csm", ModelKind::Csm),
            ("moshi", ModelKind::Moshi),
        ];
        for (name, kind) in kinds {
            let parsed = parse_args(&args(&["--model", name, "--input", "i", "--output", "o"]))
                .unwrap_or_else(|e| panic!("--model {name} should parse: {e}"));
            assert_eq!(parsed.model, *kind, "--model {name}");
            assert!(USAGE.contains(name), "USAGE lists `{name}`");
        }
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
