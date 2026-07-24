//! `vokra-convert` command-line entry point (M0-03, FR-TL-01).
//!
//! ```text
//! vokra-convert --model <whisper|silero-vad|piper-plus|campplus|kokoro|cosyvoice2|voxtral|mimi|dac|csm|moshi|denoise|dia|zonos|kyutai-stt>
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
    ModelKind, convert_cosyvoice2_file, convert_cosyvoice3_file, convert_csm_file,
    convert_dac_file, convert_file_licensed, convert_file_quantized, convert_moshi_file,
    convert_piper_plus_file, convert_utmos_file,
};
use vokra_core::gguf::{FrontendSpec, GgmlType};

const USAGE: &str = "\
vokra-convert — convert an upstream checkpoint to Vokra GGUF (M0-03, FR-TL-01)

USAGE:
    vokra-convert --model <whisper|silero-vad|campplus|kokoro|voxtral|mimi|denoise|dia|zonos|kyutai-stt|parakeet-tdt|parakeet-ctc|canary|omniasr-ctc|distil-whisper|kotoba-whisper> --input <checkpoint> --output <out.gguf>
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
                       config.json), csm (Sesame CSM-1B safetensors),
                       moshi (Kyutai Moshi safetensors), dia (nari-labs
                       Dia-1.6B safetensors — SoTA plan Phase 1-4),
                       zonos (Zyphra Zonos-v0.1-transformer safetensors —
                       SoTA plan Phase 1-5), kyutai-stt (Kyutai
                       STT-2.6B-EN decoder-only English streaming ASR
                       over Mimi tokens — SoTA plan Phase 2; weight
                       license = CC-BY 4.0 attribution required), or
                       parakeet-tdt (NVIDIA Parakeet-TDT-0.6B-v3 English
                       ASR — FastConformer encoder + TDT decoder — SoTA
                       plan Phase 2; weight license = CC-BY 4.0
                       attribution required), or
                       parakeet-ctc (NVIDIA Parakeet-CTC-1.1B English ASR
                       — FastConformer encoder + CTC head, no RNN-T
                       prediction network — SoTA plan Phase 2; ships BF16
                       — pre-widen to F32 offline for now; weight license
                       = CC-BY 4.0 attribution required), or
                       canary (NVIDIA Canary-1B-v2 multilingual multi-task
                       ASR / AST across 25 European languages —
                       FastConformer encoder (32 layers) + Transformer AED
                       decoder (8 layers) — SoTA plan Phase 2; distributed
                       as a .nemo tarball, flatten to safetensors with a
                       prepare-checkpoint script first; upstream is BF16 —
                       pre-widen to F32 offline for now; weight license =
                       CC-BY 4.0 attribution required), or
                       omniasr-ctc (Meta omniASR-CTC-1B multilingual ASR
                       across 1600+ languages — wav2vec 2.0 waveform-in
                       encoder (48 layers) + single-Linear CTC head —
                       SoTA plan Phase 2; distributed as a fairseq2 .pt +
                       SentencePiece tokenizer, flatten to safetensors
                       with a prepare-checkpoint script first; upstream
                       is F32; weight license = Apache-2.0 permissive —
                       no runtime-side attribution obligation), or
                       distil-whisper (HuggingFace distil-large-v3.5 —
                       Whisper large-v3 encoder + 2-layer decoder; same
                       op inventory as vanilla Whisper, only n_text_layer
                       differs — SoTA plan Phase 2; ships F32
                       safetensors directly; weight license = MIT
                       permissive — no runtime-side attribution
                       obligation), or
                       kotoba-whisper (Kotoba Technologies
                       kotoba-whisper-v1.x / v2.x / bilingual family —
                       Japanese-distilled Whisper: large-v3 encoder +
                       shrunk 2-layer decoder; same tensor topology as
                       distil-large-v3.5 but distinct upstream release
                       — SoTA plan Phase 5 JA-ASR-2; ships F32/F16
                       safetensors directly; weight license = Apache-2.0
                       permissive — no runtime-side attribution
                       obligation. **JA-ASR-2 axis**: n_text_layer=2 is
                       read from checkpoint tensor names, never
                       hard-coded).
                       `whisper-base` is accepted as a backward-compatible
                       alias for `whisper` (size is still derived from the
                       checkpoint, not the flag).
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

    // `restamp` is a distinct subcommand (metadata-only provenance rewrite of an
    // existing GGUF — no `--model`), so it is dispatched before the converter
    // arg parser, which requires `--model`.
    if args.first().map(String::as_str) == Some("restamp") {
        return run_restamp(&args[1..]);
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
        ModelKind::CosyVoice3 => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // SoTA plan Phase 3 (2026-07-24): Fun-CosyVoice3 shares the
            // CosyVoice2 topology (Qwen2 LLM + chunk-aware CFM + HiFTNet
            // vocoder), so the shape-derivation walk delegates verbatim.
            // Same `--config` requirement — the upstream HF config.json
            // (Qwen2 schema) is optional; without it only the
            // shape-derived hparams are written and the runtime refuses
            // the LLM bind (loud note per FR-EX-08).
            convert_cosyvoice3_file(&input, config.as_deref(), &output)
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
                         dac | csm | moshi | denoise | dia | zonos | kyutai-stt | \
                         parakeet-tdt | parakeet-ctc | canary | omniasr-ctc | \
                         distil-whisper | kotoba-whisper)"
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
        ModelKind::CosyVoice3 => {
            // SoTA plan Phase 3: shape-parallel to CosyVoice2 but reads
            // the `vokra.cosyvoice3.*` chunk group (byte-parallel to
            // CosyVoice2's) so the verify surface reflects the arch
            // label the operator invoked.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.cosyvoice3.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.cosyvoice3.arch.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.cosyvoice3.arch.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden_dim = file
                .get("vokra.cosyvoice3.arch.hidden_dim")
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
        ModelKind::Dia => {
            // SoTA plan Phase 1-4 (2026-07-24). The `vokra.dia.*` chunk group
            // is written entirely from primary-source-transcribed constants —
            // the summary reads back the anchoring shape triples.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let enc_layers = file
                .get("vokra.dia.arch.encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_layers = file
                .get("vokra.dia.arch.decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let channels = file
                .get("vokra.dia.channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.dia.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} encoder_layers={enc_layers} decoder_layers={dec_layers} \
                 channels={channels} sample_rate={sr}"
            );
        }
        ModelKind::Zonos => {
            // SoTA plan Phase 1-5 (2026-07-24). The `vokra.zonos.*` chunk
            // group is written entirely from primary-source-transcribed
            // constants — the summary reads back the anchoring shape triples
            // (single uniform GQA backbone, 9 codebook channels, 44.1 kHz).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let bb_layers = file
                .get("vokra.zonos.arch.backbone.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.zonos.arch.backbone.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let num_cb = file
                .get("vokra.zonos.num_codebooks")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.zonos.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let conds = file
                .get("vokra.zonos.prefix_conditioner.count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} backbone_layers={bb_layers} d_model={d_model} \
                 num_codebooks={num_cb} conditioners={conds} sample_rate={sr}"
            );
        }
        ModelKind::KyutaiStt => {
            // SoTA plan Phase 2 (2026-07-24). The `vokra.kyutai_stt.*` chunk
            // group is written entirely from primary-source-transcribed
            // constants — the summary reads back the anchoring shape triples
            // (48-layer MHA backbone, 32 Mimi codebook channels, 24 kHz).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let bb_layers = file
                .get("vokra.kyutai_stt.arch.backbone.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.kyutai_stt.arch.backbone.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_q = file
                .get("vokra.kyutai_stt.audio.n_q")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_card = file
                .get("vokra.kyutai_stt.text.card")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.kyutai_stt.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} backbone_layers={bb_layers} d_model={d_model} \
                 n_q={n_q} text_card={text_card} sample_rate={sr}"
            );
        }
        ModelKind::Parakeet => {
            // SoTA plan Phase 2 (2026-07-24). The `vokra.parakeet.*` chunk
            // group is written entirely from primary-source-transcribed
            // constants — the summary reads back the anchoring shape triples
            // (24-layer FastConformer encoder, MHA 8-head, 2-layer 640-d
            // RNN-T prediction net, 8193 vocab, 5 duration bins, 16 kHz).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let enc_layers = file
                .get("vokra.parakeet.arch.encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.parakeet.arch.encoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.parakeet.arch.encoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_layers = file
                .get("vokra.parakeet.arch.decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_d_model = file
                .get("vokra.parakeet.arch.decoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.parakeet.joint.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_dur = file
                .get("vokra.parakeet.joint.n_durations")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.parakeet.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} encoder_layers={enc_layers} d_model={d_model} \
                 n_head={n_head} decoder_layers={dec_layers} \
                 decoder_d_model={dec_d_model} vocab={vocab} \
                 n_durations={n_dur} sample_rate={sr}"
            );
        }
        ModelKind::ParakeetCtc => {
            // SoTA plan Phase 2 (2026-07-24). The `vokra.parakeet_ctc.*`
            // chunk group is written entirely from primary-source-transcribed
            // constants — the summary reads back the anchoring shape triples
            // (42-layer FastConformer encoder, MHA 8-head, 80-bin log-mel,
            // attention_bias=true, scale_input=true, 1025 vocab with blank
            // at pad_token_id=1024, 16 kHz). No decoder / joint / duration
            // group exists — CTC has no RNN-T prediction network.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let enc_layers = file
                .get("vokra.parakeet_ctc.arch.encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.parakeet_ctc.arch.encoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.parakeet_ctc.arch.encoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let in_dim = file
                .get("vokra.parakeet_ctc.arch.encoder.in_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let attn_bias = file
                .get("vokra.parakeet_ctc.arch.encoder.attention_bias")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let scale_input = file
                .get("vokra.parakeet_ctc.arch.encoder.scale_input")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.parakeet_ctc.head.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let pad_id = file
                .get("vokra.parakeet_ctc.head.pad_token_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.parakeet_ctc.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} encoder_layers={enc_layers} d_model={d_model} \
                 n_head={n_head} in_dim={in_dim} attention_bias={attn_bias} \
                 scale_input={scale_input} vocab={vocab} \
                 pad_token_id={pad_id} sample_rate={sr}"
            );
        }
        ModelKind::Canary => {
            // SoTA plan Phase 2 (2026-07-24). The `vokra.canary.*` chunk
            // group is written entirely from primary-source-transcribed
            // constants — the summary reads back the anchoring shape
            // triples (32-layer FastConformer encoder, MHA 8-head,
            // 128-bin log-mel, attention_bias=true, 8-layer Transformer
            // decoder, 16 384 vocab, 16 kHz).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let enc_layers = file
                .get("vokra.canary.arch.encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let enc_d_model = file
                .get("vokra.canary.arch.encoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let enc_n_head = file
                .get("vokra.canary.arch.encoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let in_dim = file
                .get("vokra.canary.arch.encoder.in_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_layers = file
                .get("vokra.canary.arch.decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_d_model = file
                .get("vokra.canary.arch.decoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_head = file
                .get("vokra.canary.arch.decoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_max_seq = file
                .get("vokra.canary.arch.decoder.max_sequence_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.canary.head.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.canary.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} encoder_layers={enc_layers} enc_d_model={enc_d_model} \
                 enc_n_head={enc_n_head} in_dim={in_dim} decoder_layers={dec_layers} \
                 dec_d_model={dec_d_model} dec_n_head={dec_n_head} \
                 dec_max_seq={dec_max_seq} vocab={vocab} sample_rate={sr}"
            );
        }
        ModelKind::OmniasrCtc => {
            // SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B — 1600+
            // language wav2vec 2.0 CTC ASR (encoder + single-Linear CTC head
            // — no decoder / joint / duration bins). Verify the loaded
            // GGUF carries the key hparam chunk group.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let model_dim = file
                .get("vokra.omniasr_ctc.arch.encoder.model_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.omniasr_ctc.arch.encoder.num_encoder_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.omniasr_ctc.arch.encoder.num_encoder_attn_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let feature_dim = file
                .get("vokra.omniasr_ctc.arch.encoder.feature_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.omniasr_ctc.head.target_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let blank = file
                .get("vokra.omniasr_ctc.head.blank_id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.omniasr_ctc.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} model_dim={model_dim} n_layer={n_layer} n_head={n_head} \
                 feature_dim={feature_dim} target_vocab={vocab} blank_id={blank} \
                 sample_rate={sr}"
            );
        }
        ModelKind::DistilWhisper => {
            // SoTA plan Phase 2 (2026-07-24): HuggingFace distil-whisper /
            // distil-large-v3.5 — Whisper large-v3 encoder + 2-layer decoder.
            // Reuses the `vokra.whisper.*` chunk schema (schema shared with
            // vanilla Whisper) so the verify surface here is the same shape
            // as Whisper's: n_audio_layer / n_text_layer are the interesting
            // pair (the distil axis).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let d_model = file
                .get("vokra.whisper.n_audio_state")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_audio_layer = file
                .get("vokra.whisper.n_audio_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_text_layer = file
                .get("vokra.whisper.n_text_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_mels = file
                .get("vokra.whisper.n_mels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_vocab = file
                .get("vokra.whisper.n_vocab")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} d_model={d_model} n_audio_layer={n_audio_layer} \
                 n_text_layer={n_text_layer} n_mels={n_mels} n_vocab={n_vocab}"
            );
        }
        ModelKind::KotobaWhisper => {
            // SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba
            // Technologies kotoba-whisper family — Japanese-distilled
            // Whisper (large-v3 encoder + 2-layer decoder). Reuses the
            // `vokra.whisper.*` chunk schema (schema shared with
            // vanilla Whisper) so the verify surface here is the same
            // shape as Whisper's: n_audio_layer / n_text_layer are
            // the interesting pair (the JA-ASR-2 data-driven decoder
            // axis).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let d_model = file
                .get("vokra.whisper.n_audio_state")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_audio_layer = file
                .get("vokra.whisper.n_audio_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_text_layer = file
                .get("vokra.whisper.n_text_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_mels = file
                .get("vokra.whisper.n_mels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_vocab = file
                .get("vokra.whisper.n_vocab")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} d_model={d_model} n_audio_layer={n_audio_layer} \
                 n_text_layer={n_text_layer} n_mels={n_mels} n_vocab={n_vocab}"
            );
        }
        ModelKind::Chatterbox => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox verify surface —
            // arch/name plus the T3 axes that identify the multilingual vs
            // English-only variant and pin the Llama_520M backbone shape.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let variant = file
                .get("vokra.chatterbox.variant")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let text_vocab = file
                .get("vokra.chatterbox.arch.text_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speech_vocab = file
                .get("vokra.chatterbox.arch.speech_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.chatterbox.arch.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.chatterbox.arch.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.chatterbox.arch.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_dim = file
                .get("vokra.chatterbox.arch.head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let ffn_dim = file
                .get("vokra.chatterbox.arch.ffn_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.chatterbox.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} variant={variant} text_vocab={text_vocab} \
                 speech_vocab={speech_vocab} hidden={hidden} n_layer={n_layer} \
                 n_head={n_head} head_dim={head_dim} ffn={ffn_dim} sample_rate={sr}"
            );
        }
        ModelKind::ChatterboxTurbo => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox-Turbo verify surface —
            // arch/name plus the GPT-2-medium backbone axes + STFT frontend
            // + paralinguistic tag count that identify the Turbo variant vs
            // base Chatterbox (backbone family swap + 32 kHz vs 24 kHz +
            // 50 276 vs 2454/704 text vocab).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let backbone = file
                .get("vokra.chatterbox_turbo.backbone_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let text_vocab = file
                .get("vokra.chatterbox_turbo.arch.text_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speech_vocab = file
                .get("vokra.chatterbox_turbo.arch.speech_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.chatterbox_turbo.arch.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.chatterbox_turbo.arch.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.chatterbox_turbo.arch.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_dim = file
                .get("vokra.chatterbox_turbo.arch.head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let paraling = file
                .get("vokra.chatterbox_turbo.arch.paralinguistic_tag_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.chatterbox_turbo.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} backbone={backbone} text_vocab={text_vocab} \
                 speech_vocab={speech_vocab} hidden={hidden} n_layer={n_layer} \
                 n_head={n_head} head_dim={head_dim} paralinguistic_tags={paraling} \
                 sample_rate={sr}"
            );
        }
        ModelKind::ChatterboxNano => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox-Nano verify surface —
            // arch/name plus the Llama_520M backbone axes + STFT frontend +
            // paralinguistic tag count + the distinguishing GPT-2 EOT
            // stop_text_token that identify the Nano variant vs base
            // Chatterbox (sample rate + text vocab swap) and vs Turbo
            // (Llama_520M backbone family instead of gpt2-medium).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let backbone = file
                .get("vokra.chatterbox_nano.backbone_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let text_vocab = file
                .get("vokra.chatterbox_nano.arch.text_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speech_vocab = file
                .get("vokra.chatterbox_nano.arch.speech_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.chatterbox_nano.arch.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.chatterbox_nano.arch.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.chatterbox_nano.arch.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_dim = file
                .get("vokra.chatterbox_nano.arch.head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let paraling = file
                .get("vokra.chatterbox_nano.arch.paralinguistic_tag_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let stop_text = file
                .get("vokra.chatterbox_nano.token.stop_text")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.chatterbox_nano.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} backbone={backbone} text_vocab={text_vocab} \
                 speech_vocab={speech_vocab} hidden={hidden} n_layer={n_layer} \
                 n_head={n_head} head_dim={head_dim} paralinguistic_tags={paraling} \
                 stop_text_token={stop_text} sample_rate={sr}"
            );
        }
        ModelKind::Qwen3Tts => {
            // SoTA plan Phase 3 (2026-07-24): Qwen3-TTS verify surface —
            // arch/name plus the Qwen3 talker + code-predictor axes + the
            // codec handshake (`num_code_groups`) that identifies the
            // discrete multi-codebook LM topology distinct from
            // CosyVoice2/3's vocoder-LM topology.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.qwen3_tts.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.qwen3_tts.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let spk = file
                .get("vokra.qwen3_tts.speaker_embed_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.qwen3_tts.talker.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_layer = file
                .get("vokra.qwen3_tts.talker.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head = file
                .get("vokra.qwen3_tts.talker.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_head_kv = file
                .get("vokra.qwen3_tts.talker.n_head_kv")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_dim = file
                .get("vokra.qwen3_tts.talker.head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_vocab = file
                .get("vokra.qwen3_tts.talker.text_vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speech_vocab = file
                .get("vokra.qwen3_tts.talker.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let groups = file
                .get("vokra.qwen3_tts.talker.num_code_groups")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cp_layers = file
                .get("vokra.qwen3_tts.code_predictor.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cp_vocab = file
                .get("vokra.qwen3_tts.code_predictor.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} sample_rate={sr} \
                 speaker_embed_dim={spk} talker.hidden={hidden} talker.n_layer={n_layer} \
                 talker.n_head={n_head} talker.n_head_kv={n_head_kv} talker.head_dim={head_dim} \
                 talker.text_vocab={text_vocab} talker.speech_vocab={speech_vocab} \
                 num_code_groups={groups} code_predictor.n_layer={cp_layers} \
                 code_predictor.vocab={cp_vocab}"
            );
        }
        ModelKind::VoxCpm2 => {
            // SoTA plan Phase 4 (2026-07-24): VoxCPM-0.5B verify surface —
            // arch / name plus the MiniCPM-4 LM axes + AudioVAE V2
            // axes + the VAE handshake (`feat_dim == vae.latent_dim`)
            // that identifies the continuous VAE + diffusion-decoder
            // topology distinct from CosyVoice2/3's vocoder-LM and
            // Qwen3-TTS's codec-LM.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.voxcpm2.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let feat = file
                .get("vokra.voxcpm2.feat_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let patch = file
                .get("vokra.voxcpm2.patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let lm_hidden = file
                .get("vokra.voxcpm2.lm.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let lm_n_layer = file
                .get("vokra.voxcpm2.lm.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let lm_n_head = file
                .get("vokra.voxcpm2.lm.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let lm_n_head_kv = file
                .get("vokra.voxcpm2.lm.n_head_kv")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let lm_vocab = file
                .get("vokra.voxcpm2.lm.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_n_layer = file
                .get("vokra.voxcpm2.dit.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr_in = file
                .get("vokra.vae_continuous.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr_out = file
                .get("vokra.vae_continuous.out_sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vae_latent = file
                .get("vokra.vae_continuous.latent_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} feat_dim={feat} patch_size={patch} \
                 lm.hidden={lm_hidden} lm.n_layer={lm_n_layer} lm.n_head={lm_n_head} \
                 lm.n_head_kv={lm_n_head_kv} lm.vocab={lm_vocab} dit.n_layer={dit_n_layer} \
                 vae.sr_in={sr_in} vae.sr_out={sr_out} vae.latent_dim={vae_latent}"
            );
        }
        ModelKind::VibeVoice => {
            // SoTA plan Phase 4 (2026-07-24): VibeVoice-1.5B verify surface —
            // arch / name plus the Qwen2 decoder LM axes + acoustic + semantic
            // tokenizer VAE dims + diffusion-head axes (prediction_type,
            // beta_schedule, num_inference_steps) that identify the DDPM
            // v-prediction path distinct from VoxCPM's UnifiedCFM flow-
            // matching sampler.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.vibevoice.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let dec_hidden = file
                .get("vokra.vibevoice.decoder.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_layer = file
                .get("vokra.vibevoice.decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_head = file
                .get("vokra.vibevoice.decoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_head_kv = file
                .get("vokra.vibevoice.decoder.n_head_kv")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_vocab = file
                .get("vokra.vibevoice.decoder.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let acoustic_vae = file
                .get("vokra.vibevoice.acoustic_vae_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let semantic_vae = file
                .get("vokra.vibevoice.semantic_vae_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let acoustic_sr = file
                .get("vokra.vibevoice.acoustic.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_layers = file
                .get("vokra.vibevoice.diffusion_head.head_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let pred_type = file
                .get("vokra.vibevoice.diffusion_head.prediction_type")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let beta_sched = file
                .get("vokra.vibevoice.diffusion_head.ddpm_beta_schedule")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_inf_steps = file
                .get("vokra.vibevoice.diffusion_head.ddpm_num_inference_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} decoder.hidden={dec_hidden} \
                 decoder.n_layer={dec_n_layer} decoder.n_head={dec_n_head} \
                 decoder.n_head_kv={dec_n_head_kv} decoder.vocab={dec_vocab} \
                 acoustic.vae_dim={acoustic_vae} semantic.vae_dim={semantic_vae} \
                 acoustic.sr={acoustic_sr} head.layers={head_layers} \
                 head.prediction_type={pred_type} head.beta_schedule={beta_sched} \
                 head.num_inference_steps={n_inf_steps}"
            );
        }
        ModelKind::Irodori => {
            // SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Irodori-TTS-500M-v3
            // verify surface — arch / name plus the RF-DiT body axes +
            // text-encoder + speaker-encoder axes + duration-predictor
            // enable flag that identify the Rectified-Flow / Linear-or-Sway
            // schedule path distinct from VoxCPM's EpsS-schedule
            // flow-matching sampler and VibeVoice's DDPM sampler.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.irodori.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.irodori.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_latent = file
                .get("vokra.irodori.dit.latent_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_model = file
                .get("vokra.irodori.dit.model_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_layers = file
                .get("vokra.irodori.dit.num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_heads = file
                .get("vokra.irodori.dit.num_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dit_adaln_rank = file
                .get("vokra.irodori.dit.adaln_rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_vocab = file
                .get("vokra.irodori.text.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_dim = file
                .get("vokra.irodori.text.dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_layers = file
                .get("vokra.irodori.text.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speaker_dim = file
                .get("vokra.irodori.speaker.dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let speaker_layers = file
                .get("vokra.irodori.speaker.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let duration_enabled = file
                .get("vokra.irodori.duration.enabled")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_tok = file
                .get("vokra.irodori.text_tokenizer_repo")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} family={family} sample_rate={sr} \
                 dit.latent_dim={dit_latent} dit.model_dim={dit_model} \
                 dit.num_layers={dit_layers} dit.num_heads={dit_heads} \
                 dit.adaln_rank={dit_adaln_rank} text.vocab={text_vocab} text.dim={text_dim} \
                 text.n_layer={text_layers} speaker.dim={speaker_dim} \
                 speaker.n_layer={speaker_layers} duration.enabled={duration_enabled} \
                 text_tokenizer={text_tok}"
            );
        }
    }
    Ok(())
}

/// `vokra-convert restamp` — rewrite an existing GGUF's provenance metadata
/// without re-materialising tensors (the low-memory publish path).
fn run_restamp(args: &[String]) -> ExitCode {
    const USAGE: &str = "\
USAGE:
    vokra-convert restamp --input <in.gguf> --output <out.gguf> \\
        --license <spdx> [--model-id <id>] [--source <text>] [--attribution <text>]

Rewrites vokra.provenance.* on an existing GGUF, copying tensors verbatim (peak
memory = one tensor). For a large artifact that was converted before provenance
stamping, or is too big to re-convert on this machine.
";
    let mut input = None;
    let mut output = None;
    let mut license = None;
    let mut model_id = None;
    let mut source = None;
    let mut attribution = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--input" => {
                input = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--output" => {
                output = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--license" => {
                license = args.get(i + 1).cloned();
                i += 2;
            }
            "--model-id" => {
                model_id = args.get(i + 1).cloned();
                i += 2;
            }
            "--source" => {
                source = args.get(i + 1).cloned();
                i += 2;
            }
            "--attribution" => {
                attribution = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("error: unexpected argument `{other}`\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }
    let (Some(input), Some(output), Some(license)) = (input, output, license) else {
        eprintln!("error: --input, --output and --license are required\n\n{USAGE}");
        return ExitCode::from(2);
    };
    // model_id / source default to advisory placeholders if omitted.
    let model_id = model_id.unwrap_or_else(|| "restamped".to_owned());
    let source = source.unwrap_or_else(|| format!("restamped GGUF (licence {license})"));

    match vokra_convert::restamp_provenance(
        &input,
        &output,
        &license,
        &model_id,
        &source,
        attribution.as_deref(),
    ) {
        Ok(summary) => {
            println!(
                "restamped: {} tensors, {} metadata keys -> {}",
                summary.tensor_count,
                summary.metadata_count,
                output.display()
            );
            for note in &summary.notes {
                println!("  note: {note}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
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
            ("dia", ModelKind::Dia),
            ("zonos", ModelKind::Zonos),
            ("kyutai-stt", ModelKind::KyutaiStt),
            ("parakeet-tdt", ModelKind::Parakeet),
            ("parakeet-ctc", ModelKind::ParakeetCtc),
            ("canary", ModelKind::Canary),
            ("omniasr-ctc", ModelKind::OmniasrCtc),
            ("distil-whisper", ModelKind::DistilWhisper),
            ("kotoba-whisper", ModelKind::KotobaWhisper),
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
