//! `vokra-convert` command-line entry point (M0-03, FR-TL-01).
//!
//! ```text
//! vokra-convert --model <whisper|silero-vad|piper-plus|campplus|kokoro|cosyvoice2|voxtral|mimi|dac|csm|moshi|denoise|dia|zonos|kyutai-stt|charsiu>
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

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use vokra_convert::{
    ConvertError, ConvertSummary, ModelKind, convert_beat_this_with_config,
    convert_cosyvoice2_file, convert_cosyvoice3_file, convert_csm_file, convert_dac_file,
    convert_file_licensed, convert_file_quantized, convert_moshi_file, convert_piper_plus_file,
    convert_sbv2_file, convert_utmos_file,
};
use vokra_core::gguf::{FrontendSpec, GgmlType};

const USAGE: &str = "\
vokra-convert — convert an upstream checkpoint to Vokra GGUF (M0-03, FR-TL-01)

USAGE:
    vokra-convert --model <whisper|silero-vad|fsmn-vad|campplus|kokoro|voxtral|mimi|denoise|dia|zonos|kyutai-stt|parakeet-tdt|parakeet-ctc|canary|canary-qwen|omniasr-ctc|distil-whisper|kotoba-whisper|vits-ja|styletts2|charsiu> --input <checkpoint> --output <out.gguf>
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
                       charsiu (`charsiu/en_w2v2_fc_10ms` safetensors —
                       English Wav2Vec2 frame classifier for phone-level
                       forced alignment; converter verifies the canonical
                       tensor manifest and folds positional-conv weight norm),
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
                       hard-coded), or
                       vits-ja (ESPnet-family Japanese plain VITS —
                       Kim et al. 2021 VITS + HiFi-GAN generator, as
                       shipped by ESPnet's
                       egs2/jsut/tts1/conf/tuning/train_vits.yaml +
                       egs2/jvs/tts1/conf/tuning/finetune_vits.yaml
                       + COEIROINK deployments — SoTA plan Phase 5
                       JA-TTS-2; ships F32/F16 safetensors directly.
                       Distinct arch tag from piper-plus because plain
                       VITS decodes through a HiFi-GAN generator
                       directly while piper-plus (MB-iSTFT-VITS2)
                       decodes through a sub-band iSTFT + PQMF post-net.
                       **⚠️  Weight redistribution default is
                       `RedistributionForbidden`**: JSUT / JVS /
                       COEIROINK corpus terms forbid trained-weight
                       redistribution; override with --license <spdx>
                       if you trained on a permissive corpus), or
                       styletts2 (yl4579 StyleTTS 2 — Li et al. 2023
                       arXiv:2306.07691 — config-only scaffold. LJSpeech
                       / LibriTTS release axes are transcribed verbatim;
                       every F32 / F16 / BF16 tensor passes through
                       verbatim under its upstream safetensors name.
                       **⚠️  Weight redistribution default is `Unknown`
                       (fail-closed)**: the yl4579 pretrained models
                       ride a voice-consent / disclosure usage agreement
                       — NOT a standard SPDX permissive license. The
                       runtime `StyleTts2Tts::from_gguf` returns
                       NotImplemented naming the licence blocker;
                       override with --license <spdx> if you trained on
                       a permissive corpus).
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
        ModelKind::SbV2 => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // --config carries the JSON side-car with every vokra.sbv2.*
            // hparam (Task 25 review fast-follow, 2026-07-26): the generic
            // `_ =>` arm below funnels through `convert_file_licensed`,
            // whose own `SbV2` branch hard-codes `config_side_car = None`
            // (its signature has no `--config` parameter to forward) --
            // silently dropping this flag. Call `convert_sbv2_file`
            // directly instead, mirroring the `Csm` / `Moshi` /
            // `CosyVoice2` / `CosyVoice3` arms above.
            convert_sbv2(&input, config.as_deref(), &output, license.as_deref())
        }
        ModelKind::BeatThis => {
            if quant.is_some() {
                eprintln!("error: --quantize is only supported for whisper\n\n{USAGE}");
                return ExitCode::from(2);
            }
            // Same reason as the `SbV2` arm above: `convert_file_licensed`
            // has no `--config` parameter to forward, so routing through
            // the generic `_ =>` arm would drop this flag on the floor.
            // (Its `BeatThis` branch refuses loudly rather than silently
            // defaulting, so nothing would be *mis-stamped* — but the
            // command would be unusable from this binary.)
            //
            // The upstream `CPJKU/beat_this` `.pt` release ships no
            // config.yaml, so the six `vokra.beat_this.*` axes have to
            // come from the caller; `convert_beat_this_with_config`
            // documents the schema and refuses every missing key by name.
            convert_beat_this_with_config(&input, &output, config.as_deref(), license.as_deref())
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

/// Converts an SBV2 v2 checkpoint via [`convert_sbv2_file`], forwarding
/// `--config` through directly, and re-derives a [`ConvertSummary`] from its
/// [`vokra_convert::SbV2ConvertReport`] return value.
///
/// The generic `_ =>` dispatch arm in [`main`] routes every other model kind
/// through [`convert_file_licensed`], whose own `ModelKind::SbV2` branch
/// hard-codes `config_side_car = None` (that function's signature has no
/// `--config` parameter to forward) — so a bare `--model sbv2 --config
/// <side-car>` invocation would silently drop the flag and always emit a
/// hparam-empty GGUF. This helper is called from a dedicated `ModelKind::
/// SbV2` arm instead (mirroring the `Csm` / `Moshi` / `CosyVoice2` /
/// `CosyVoice3` arms, which each call their own model-specific converter
/// directly). The summary-building logic below duplicates
/// `convert_file_licensed`'s own `SbV2` branch on purpose: that branch
/// returns `ConvertReport`, not `ConvertSummary` like every sibling
/// wrapper, so it cannot be called from here without widening
/// `convert_file_licensed`'s own signature (out of scope for this fix).
fn convert_sbv2(
    input: &Path,
    config: Option<&Path>,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    let report = convert_sbv2_file(input, output, config, license)?;
    let mut notes = vec![format!(
        "sbv2: {} float weights written verbatim ({} read, {} non-float skipped), \
         vokra.sbv2.* hparam chunk written: {}",
        report.written, report.read, report.skipped_non_float, report.hparams_written,
    )];
    if !report.hparams_written {
        notes.push(
            "no --config side-car: vokra.sbv2.* metadata was not written -- pass \
             --config <side-car.json> for a hparam-complete GGUF"
                .to_owned(),
        );
    }
    Ok(ConvertSummary {
        model: ModelKind::SbV2,
        tensor_count: report.written,
        metadata_count: 0, // Populated by convert_sbv2_file's builder.
        output_bytes: std::fs::metadata(output)?.len(),
        notes,
    })
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
                         parakeet-tdt | parakeet-ctc | canary | canary-qwen | omniasr-ctc | \
                         distil-whisper | kotoba-whisper | vits-ja | styletts2 | fsmn-vad)"
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
        ModelKind::CanaryQwen => {
            // SoTA plan reuse bundle (2026-07-30): NVIDIA Canary-Qwen-2.5B —
            // FastConformer encoder (reused Canary-1B-v2 axes) + Qwen LLM
            // decoder (canonical Qwen-family axes with `0`-placeholder
            // dims pending .nemo extraction). Verify the loaded GGUF
            // carries the key hparam chunk group and the arch tag is
            // distinct from base Canary.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let enc_layers = file
                .get("vokra.canary_qwen.arch.encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let enc_d_model = file
                .get("vokra.canary_qwen.arch.encoder.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let enc_n_head = file
                .get("vokra.canary_qwen.arch.encoder.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let in_dim = file
                .get("vokra.canary_qwen.arch.encoder.in_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_layers = file
                .get("vokra.canary_qwen.arch.decoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_hidden_dim = file
                .get("vokra.canary_qwen.arch.decoder.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_head_q = file
                .get("vokra.canary_qwen.arch.decoder.n_head_q")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_n_head_kv = file
                .get("vokra.canary_qwen.arch.decoder.n_head_kv")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_head_dim = file
                .get("vokra.canary_qwen.arch.decoder.head_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.canary_qwen.arch.decoder.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cross_attn_hidden_dim = file
                .get("vokra.canary_qwen.arch.cross_attn.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sr = file
                .get("vokra.canary_qwen.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} encoder_layers={enc_layers} enc_d_model={enc_d_model} \
                 enc_n_head={enc_n_head} in_dim={in_dim} decoder_layers={dec_layers} \
                 dec_hidden_dim={dec_hidden_dim} dec_gqa={dec_n_head_q}q/{dec_n_head_kv}kv \
                 dec_head_dim={dec_head_dim} vocab={vocab} \
                 cross_attn_hidden_dim={cross_attn_hidden_dim} sample_rate={sr}"
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
        ModelKind::Crisperwhisper => {
            // residual wave 4 (2026-08-02): CrisperWhisper —
            // Whisper-large-v3 verbatim-word-timestamps fine-tune under
            // cc-by-nc-4.0. Reuses the vanilla Whisper converter via
            // the `WhisperVariant::CrisperWhisper` arm; every
            // architectural axis is byte-identical to whisper-large-v3
            // (32/32 layers, d_model=1280, n_mels=128, vocab=51866).
            // Verify surface mirrors distil-whisper / kotoba-whisper —
            // arch stamp + name + the four data-driven axes.
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
        ModelKind::VibeVoiceRealtime => {
            // 2026-08-01 add: VibeVoice-Realtime-0.5B (streaming sibling)
            // verify surface -- mirror of VibeVoice above, but the
            // semantic tokenizer axes are absent (streaming variant
            // carries only the acoustic tokenizer) and the
            // streaming-only `tts_backbone_num_hidden_layers` axis is
            // surfaced. The diffusion-head hidden dim is also
            // reported since it differs from the 1.5B value.
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
            let acoustic_sr = file
                .get("vokra.vibevoice.acoustic.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let head_hidden = file
                .get("vokra.vibevoice.diffusion_head.hidden_size")
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
            let tts_backbone = file
                .get("vokra.vibevoice.tts_backbone_num_hidden_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} \
                 decoder.hidden={dec_hidden} decoder.n_layer={dec_n_layer} \
                 decoder.n_head={dec_n_head} decoder.n_head_kv={dec_n_head_kv} \
                 decoder.vocab={dec_vocab} acoustic.vae_dim={acoustic_vae} \
                 acoustic.sr={acoustic_sr} head.hidden={head_hidden} \
                 head.layers={head_layers} head.prediction_type={pred_type} \
                 head.beta_schedule={beta_sched} \
                 head.num_inference_steps={n_inf_steps} \
                 tts_backbone_num_hidden_layers={tts_backbone}"
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
        ModelKind::VitsJa => {
            // SoTA plan Phase 5 JA-TTS-2 (2026-07-24): plain VITS JA
            // verify surface — arch / name plus the text-encoder +
            // flow + SDP + HiFi-GAN decoder axes that identify the
            // Kim et al. 2021 VITS + plain HiFi-GAN generator path
            // distinct from piper-plus's MB-iSTFT-VITS2 (sub-band
            // iSTFT + PQMF post-net).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.vits_ja.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.vits_ja.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.vits_ja.hidden_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_mels = file
                .get("vokra.vits_ja.n_mels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let aux_channels = file
                .get("vokra.vits_ja.aux_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_layer = file
                .get("vokra.vits_ja.text.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_head = file
                .get("vokra.vits_ja.text.n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let flow_flows = file
                .get("vokra.vits_ja.flow.n_flow")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sdp_flows = file
                .get("vokra.vits_ja.sdp.n_flow")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_init = file
                .get("vokra.vits_ja.decoder.initial_channel")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let dec_kernel = file
                .get("vokra.vits_ja.decoder.kernel_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} sample_rate={sr} \
                 hidden_channels={hidden} n_mels={n_mels} aux_channels={aux_channels} \
                 text.n_layer={text_layer} text.n_head={text_head} \
                 flow.n_flow={flow_flows} sdp.n_flow={sdp_flows} \
                 decoder.initial_channel={dec_init} decoder.kernel_size={dec_kernel}"
            );
        }
        ModelKind::StyleTts2 => {
            // StyleTTS 2 (yl4579, 2026-07-30) — config-only scaffold
            // verify surface. Print arch / name plus the transcribed
            // LJSpeech / LibriTTS axes (24 kHz, hidden_dim=512,
            // style_dim=128, iSTFTNet decoder).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let family = file
                .get("vokra.styletts2.model_family")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let sr = file
                .get("vokra.styletts2.sample_rate_hz")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let style_dim = file
                .get("vokra.styletts2.style_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.styletts2.hidden_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_mels = file
                .get("vokra.styletts2.n_mels")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let text_layer = file
                .get("vokra.styletts2.text_encoder.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let diffusion_steps = file
                .get("vokra.styletts2.diffusion.steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let uses_diffusion = file
                .get("vokra.styletts2.diffusion.uses_style_diffusion")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let dec_dim_in = file
                .get("vokra.styletts2.decoder.dim_in")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} family={family} sample_rate={sr} \
                 style_dim={style_dim} hidden_dim={hidden} n_mels={n_mels} \
                 text_encoder.n_layer={text_layer} \
                 diffusion.steps={diffusion_steps} uses_style_diffusion={uses_diffusion} \
                 decoder.dim_in={dec_dim_in}"
            );
        }
        ModelKind::DebertaV2 => {
            // SBV2 v2 plan Task 11 (2026-07-26): DeBERTa v2 BERT encoder
            // verify surface — arch / name / category plus the encoder
            // axes (n_layers / vocab_size / d_model) from the
            // `vokra.bert.deberta_v2.*` chunk group.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_layers = file
                .get("vokra.bert.deberta_v2.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.bert.deberta_v2.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab_size = file
                .get("vokra.bert.deberta_v2.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} category={category} n_layer={n_layers} d_model={d_model} vocab_size={vocab_size}"
            );
        }
        ModelKind::BertBase => {
            // WP-14 (2026-08-10): plain BERT verify surface — arch /
            // name / category plus the encoder axes (n_layers /
            // hidden / heads / vocab / max_pos) from the
            // `vokra.bert_base.*` chunk group. Mirrors the DeBERTa v2
            // / v3 arms above; the key prefix change reflects the
            // arch-different rename table this converter uses.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_layers = file
                .get("vokra.bert_base.n_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let hidden = file
                .get("vokra.bert_base.hidden")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let heads = file
                .get("vokra.bert_base.heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab = file
                .get("vokra.bert_base.vocab")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let max_pos = file
                .get("vokra.bert_base.max_pos")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} category={category} n_layers={n_layers} hidden={hidden} heads={heads} vocab={vocab} max_pos={max_pos}"
            );
        }
        ModelKind::DebertaV3 => {
            // SBV2 v2 plan Task 11 (2026-07-26): DeBERTa v3 BERT encoder
            // verify surface — arch / name / category plus the encoder
            // axes (n_layers / vocab_size / d_model) from the
            // `vokra.bert.deberta_v3.*` chunk group.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let n_layers = file
                .get("vokra.bert.deberta_v3.n_layer")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let d_model = file
                .get("vokra.bert.deberta_v3.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let vocab_size = file
                .get("vokra.bert.deberta_v3.vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} category={category} n_layer={n_layers} d_model={d_model} vocab_size={vocab_size}"
            );
        }
        ModelKind::SbV2 => {
            // SBV2 v2 plan Task 25 (2026-07-26): SBV2 verify surface -- arch
            // / name plus a few vokra.sbv2.* dims when a config side-car
            // produced them. `convert_sbv2_file` omits the whole
            // `vokra.sbv2.*` chunk (rather than stamping placeholders) when
            // no config was supplied, so these read back as 0 in that case
            // -- an honest "not written" signal, not a real dimension.
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let d_model = file
                .get("vokra.sbv2.d_model")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let n_vocab = file
                .get("vokra.sbv2.n_vocab")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let sample_rate = file
                .get("vokra.sbv2.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} d_model={d_model} n_vocab={n_vocab} sample_rate={sample_rate}"
            );
        }
        ModelKind::XCodec2 => {
            // SoTA plan Phase 5 codec (2026-07-28): X-Codec 2 verify surface —
            // arch/name/category + upstream_hf + the licence class + SPDX id.
            // The M4-16 op-only landing means there is no per-model
            // `vokra.xcodec2.*` hparam chunk yet — the interesting readback
            // here is the license triple (the whole point of this converter
            // + the license_class flip).
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_hf")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_hf={upstream} \
                 license={license} weight_license={class}"
            );
        }
        // SoTA plan Phase 5 fleet (2026-07-28): 12 BF16 pass-through
        // skeleton verify surfaces. Every module writes the same
        // `vokra.model.{arch,name,category}` + `vokra.provenance.{upstream_hf,
        // license,weight_license}` chunks — no per-model hparam chunk yet
        // (the skeletons are pass-through only). Group them into a single
        // arm to keep the verify surface a shape-lookup, not a per-model
        // switch we would have to keep in step with 12 real converters.
        ModelKind::KimiAudio
        | ModelKind::StepAudio2Mini
        | ModelKind::BaichuanAudio
        | ModelKind::Speechtokenizer
        | ModelKind::Funcodec
        | ModelKind::XyTokenizer
        | ModelKind::Bicodec
        | ModelKind::Neucodec
        | ModelKind::EcapaTdnn
        | ModelKind::Wespeaker
        | ModelKind::Speaker3d
        | ModelKind::TitaNet
        | ModelKind::Emotion2vec
        | ModelKind::Rmvpe
        // M5 gap follow-up (2026-07-30): CREPE emits the same
        // `vokra.model.{arch,name,category}` + `vokra.provenance.*`
        // triple as the Phase 5 fleet + RMVPE sibling; grouped here so
        // the verify surface stays a shape-lookup.
        | ModelKind::Crepe
        | ModelKind::Charsiu
        // 2026-07-30 license half unblock: pyannote/segmentation-3.0
        // (VAD backbone) emits the same `vokra.model.*` +
        // `vokra.provenance.*` triple + additional `vokra.pyannote.*`
        // hparam chunk group; grouped with the Phase 5 fleet since the
        // verify surface is identical (arch / name / category /
        // upstream / license lookup).
        | ModelKind::PyannoteSegmentation
        // 2026-08-01 Wave 5 pipeline orchestration add:
        // pyannote/speaker-diarization-3.1 emits the same
        // `vokra.model.*` + `vokra.provenance.*` triple plus a
        // `vokra.pyannote_pipeline.*` orchestration chunk group
        // (sub-model references + clustering knobs). Weightless
        // pipeline GGUF (zero tensors by upstream design — the
        // pipeline composes sibling weight repos, does not embed
        // their weights); grouped here since the verify surface
        // stays a uniform arch/name/category/upstream/license
        // triple lookup regardless of tensor count.
        | ModelKind::PyannoteSpeakerDiarization31
        // TIER 1+2 audio-gap 41 new variants (2026-07-30 ultracode
        // `wf_022575ce-077`) all emit the same standard
        // `vokra.model.*` + `vokra.provenance.*` chunk triple
        // (plus arch-specific hparam chunks); grouped here since
        // the verify surface is a uniform arch/name/category/
        // upstream/license lookup.
        | ModelKind::Qwen3TtsBase17B
        | ModelKind::Qwen3TtsCustomVoice17B
        | ModelKind::Qwen3TtsVoiceDesign17B
        | ModelKind::Qwen3Asr
        | ModelKind::Wav2Vec2Ctc
        // 2026-08-02 wave: Facebook data2vec-audio-base-960h
        // (`facebook/data2vec-audio-base-960h`, apache-2.0). Baevski et
        // al. 2022 — wav2vec 2.0 base topology + data2vec pretraining
        // objective + LibriSpeech 960h English char CTC head. Emits the
        // same standard `vokra.model.{arch,name,category}` +
        // `vokra.provenance.*` triple as the sibling Wav2Vec2Ctc arm
        // (in fact routes through the same `wav2vec2_ctc` converter
        // module — tensor names identical, only `name` +
        // `upstream_hf` differ so a stamped GGUF faithfully reports
        // the data2vec-audio release). Grouped here for the uniform
        // arch/name/category/upstream/license triple readback.
        | ModelKind::Data2vecAudioBase
        // 2026-08-02 wave: Meta MMS-1B-All (`facebook/mms-1b-all`,
        // cc-by-nc-4.0). Massively Multilingual Speech (Pratap et al.
        // 2023, arXiv:2305.13516). Base tensor path reuses the
        // wav2vec2_ctc converter (parent workflow REUSE HINT) with
        // placeholder axes (LargeXlsr53Base sibling) and faithful
        // `name` + `upstream_hf` for the placeholder-axis refusal
        // guardrail. Emits the same standard
        // `vokra.model.{arch,name,category}` + `vokra.provenance.*`
        // triple as the sibling Wav2Vec2Ctc arm — grouped here for the
        // uniform arch/name/category/upstream/license triple readback.
        | ModelKind::Mms1bAll
        | ModelKind::MossTts
        | ModelKind::MossTtsV15
        | ModelKind::MossTtsNano
        | ModelKind::MossTtsLocal
        // 2026-08-02 wave: MOSS-Audio-4B-Instruct
        // (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0). Reuses
        // the sibling MossTts converter per the parent workflow's
        // REUSE HINT via a new `MossTtsVariant::AudioInstruct4b`
        // arm. Same verify shape as the 4 sibling tts variants — the
        // triple lookup (arch / name / upstream_hf + license) is the
        // invariant surface even though this sibling stamps category
        // = `s2s` (audio-LLM) rather than `tts`.
        | ModelKind::MossAudio4bInstruct
        // 2026-08-02 wave: MOSS-Audio-8B-Instruct
        // (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`, apache-2.0). Reuses
        // the sibling MossTts converter per the parent workflow's
        // REUSE HINT via a new `MossTtsVariant::AudioInstruct8b`
        // arm — same verify shape as the 4B sibling above (arch /
        // name / upstream_hf + license triple is the invariant
        // surface even though this sibling stamps category = `s2s`
        // audio-LLM rather than `tts`).
        | ModelKind::MossAudio8bInstruct
        | ModelKind::MeloTtsEnglish
        | ModelKind::MeloTtsChinese
        | ModelKind::MeloTtsKorean
        | ModelKind::MeloTtsSpanish
        | ModelKind::MeloTtsJapanese
        | ModelKind::SpeechT5Tts
        | ModelKind::ParlerTtsMiniMultilingual
        | ModelKind::IndicParlerTts
        | ModelKind::ParlerTtsMiniV1English
        | ModelKind::VieNeuTts
        | ModelKind::Bark
        | ModelKind::BarkSmall
        | ModelKind::HifiganVocoder
        | ModelKind::Speecht5Hifigan
        | ModelKind::Vocos
        | ModelKind::BigVGan
        | ModelKind::Focalcodec
        | ModelKind::TigerSeparator
        | ModelKind::TigerSpeech
        | ModelKind::MpSenet
        | ModelKind::MpSenetDns
        | ModelKind::MetricganPlus
        | ModelKind::SepFormer
        | ModelKind::SepformerWham16kEnh
        | ModelKind::SepformerWhamr16k
        | ModelKind::SepformerLibri2Mix
        | ModelKind::SepformerLibri3Mix
        | ModelKind::SepformerWhamr8k
        | ModelKind::SepformerDns4Enh
        | ModelKind::FsmnVad
        | ModelKind::FireredVad
        | ModelKind::SmartTurn
        | ModelKind::Clap
        | ModelKind::Ast
        | ModelKind::LangIdVoxlingua107
        | ModelKind::LangIdCommonLanguage
        | ModelKind::XVector
        | ModelKind::DeepfakeDetection
        | ModelKind::KyutaiTts
        | ModelKind::AudioboxAesthetics
        | ModelKind::VoxtralMiniRealtime
        | ModelKind::CohereTranscribe
        | ModelKind::NemotronAsrStreaming
        // 2026-08-01 wave: IBM Granite Speech 4.1-2B — Conformer CTC
        // encoder + Granite-4.0-1b-base LLM decoder + BLIP-2 q-former
        // projector (audio-LLM ASR, apache-2.0). Emits the same
        // arch / name / category / upstream_hf / license triple as the
        // Phase 5 fleet plus a `vokra.granite_speech.*` hparam chunk
        // group (which this verify path deliberately does not readback
        // — the triple lookup is the invariant surface). Grouped here
        // so the verify surface stays a shape-lookup, not a per-model
        // switch we would have to keep in step with 40+ real
        // converters.
        | ModelKind::GraniteSpeech
        // M5-16 / FR-OP-83: FCPE — pass-through BF16 skeleton verify shares
        // the same shape as the Phase 5 fleet (arch/name/category +
        // upstream_hf + license triple), plus a `vokra.f0.fcpe.*` config
        // chunk this uniform verify arm does not read back.
        //
        // Until 2026-08-15 this comment claimed the chunk was "written by the
        // model, not the converter". Nothing wrote it: the converter stamped
        // none of the fourteen axes and the runtime silently defaulted every
        // one of them to the released v001 shape. The converter now derives
        // seven from the checkpoint's tensor shapes and asserts the other
        // seven only when those shapes prove the artifact is v001; its own
        // unit tests read the chunk back, so this shape-lookup arm stays a
        // shape lookup rather than growing a per-model switch.
        | ModelKind::Fcpe
        // 2026-08-01 Wave 3: SNAC — Multi-Scale Neural Audio Codec
        // (hubertsiuzdak/snac_{24khz,44khz}, MIT). Same verify shape as
        // focalcodec / bigvgan (arch / name / category + upstream_hf +
        // license triple) plus an arch-specific `vokra.snac.variant`
        // chunk not read back by this uniform verify arm.
        | ModelKind::Snac
        // 2026-08-01 Wave 3: OpenMOSS MOSS-Audio-Tokenizer
        // (`OpenMOSS-Team/MOSS-Audio-Tokenizer` + `-Nano`, apache-2.0).
        // The codec half of the MOSS-TTS pipeline. Same verify shape
        // as snac / focalcodec / bigvgan (arch / name / category +
        // upstream_hf + license triple) plus an arch-specific
        // `vokra.moss_audio_tokenizer.variant` chunk not read back
        // by this uniform verify arm.
        | ModelKind::MossAudioTokenizer
        // 2026-08-01 Wave 3: Amphion NaturalSpeech 3 FACodec —
        // factorized VQ (FVQ) codec (`amphion/naturalspeech3_facodec`,
        // apache-2.0). Same verify shape as snac /
        // moss_audio_tokenizer / focalcodec (arch / name / category
        // + upstream_hf + license triple) plus an arch-specific
        // `vokra.facodec.variant` chunk + a `vokra.facodec.*` hparam
        // chunk group not read back by this uniform verify arm.
        | ModelKind::Facodec
        // 2026-08-01 Wave 3 sibling-pair add: YuE bundle
        // (`m-a-p/YuE-upsampler` = vocoder / `m-a-p/xcodec_mini_infer`
        // = codec, both apache-2.0). Two distinct ModelKind entries
        // share this uniform verify arm (arch / name / category +
        // upstream_hf + license triple lookup) plus an arch-specific
        // `vokra.yue_bundle.variant` chunk not read back here.
        | ModelKind::YueUpsampler
        | ModelKind::YueXcodecMini
        // 2026-08-01 Wave 5 music-generation add: Meta AudioCraft
        // MusicGen-Medium (`facebook/musicgen-medium`, cc-by-nc-4.0).
        // First music-generation converter — emits the same standard
        // `vokra.model.{arch,name,category}` + `vokra.provenance.*`
        // triple as every sibling BF16 pass-through skeleton, with
        // `category = "music"` (first music-tree entry, distinct from
        // the speech-tree tags). Grouped here so the verify surface
        // stays a shape-lookup, not a per-model switch we would have
        // to keep in step with 40+ real converters.
        | ModelKind::MusicGenMedium
        // 2026-08-01 Wave 5 music-generation add: Meta AudioCraft
        // MusicGen-Large (`facebook/musicgen-large`, cc-by-nc-4.0).
        // Sibling to MusicGen-Medium (top rung of the family, 3.3B vs
        // 1.5B). Emits the same standard `vokra.model.{arch,name,
        // category}` + `vokra.provenance.*` triple as every sibling
        // BF16 pass-through skeleton, with `category = "music"`
        // (shared with sibling MusicGen-Medium — first music-tree
        // family). Grouped here so the verify surface stays a
        // shape-lookup, not a per-model switch we would have to keep
        // in step with 40+ real converters.
        | ModelKind::MusicGenLarge
        // 2026-08-02 Wave 5 sibling: Meta AudioCraft MusicGen-Melody
        // (`facebook/musicgen-melody`, cc-by-nc-4.0). Medium 1.5B LM
        // + 12-bin chromagram conditioning frontend. Reuses the medium
        // BF16 pass-through converter arm (sibling wrapper — no new
        // models/*.rs module). Grouped here for the uniform arch /
        // name / category / upstream_hf / license triple verify shape.
        | ModelKind::MusicGenMelody
        // 2026-08-01 Wave 5 residual: Meta AudioCraft AudioGen-Medium
        // (`facebook/audiogen-medium`, cc-by-nc-4.0). MusicGen sibling
        // — identical `musicgen` arch tag, tuned on environmental
        // sounds / SFX. `category = "music"` shared with MusicGen
        // family. Grouped here for the uniform verify shape.
        | ModelKind::AudioGenMedium
        // 2026-08-01 Wave 6 residual: MusicGen-Small (sibling of
        // MusicGen-Medium/Large + AudioGen-Medium — shared `musicgen`
        // arch, `music` category. cc-by-nc-4.0 T4).
        | ModelKind::MusicGenSmall
        // 2026-08-01 Wave 6 residual: Qwen2-Audio-7B-Instruct
        // (audio-LLM omni, apache-2.0, distinct arch `qwen2_audio`).
        | ModelKind::Qwen2Audio
        // 2026-08-02 Wave residual: Qwen2.5-Omni-7B (Thinker + Talker
        // unified any-to-any omni multimodal LLM over Qwen2.5-7B
        // backbone, apache-2.0, distinct arch tag `qwen2-omni` from
        // audio-only sibling `qwen2_audio`).
        | ModelKind::Qwen25Omni7b
        // 2026-08-01 Wave 6 residual: VibeVoice-ASR (VibeVoice sibling
        // with ASR head, MIT, distinct arch `vibevoice_asr`).
        | ModelKind::VibeVoiceAsr
        // 2026-08-01 Wave 6 residual: ACE-Step 1.5 (multi-file MIT
        // music-gen bundle, distinct arch `ace_step`, category `music`).
        | ModelKind::AceStep
        // 2026-08-01 Wave 7 residual: HuBERT-Large-LS960
        // (`facebook/hubert-large-ls960-ft`, apache-2.0). 317M BERT-
        // style masked-feature-prediction self-supervised speech
        // encoder + CTC head fine-tuned on LibriSpeech 960h. Distinct
        // arch tag `hubert` from sibling wav2vec2 (different
        // pretraining objective — HuBERT masks features and predicts
        // k-means-clustered targets, wav2vec2 uses contrastive
        // masked-convnet with Gumbel-softmax quantised negatives).
        // Category `asr`. Grouped here for the uniform verify shape.
        | ModelKind::HubertLargeLs960
        // 2026-08-04 hf-audio-gap SSL residual: w2v-BERT 2.0
        // (`facebook/w2v-bert-2.0`, MIT). ~580M-parameter self-
        // supervised speech encoder = Conformer body + dual
        // (wav2vec2-style contrastive + BERT-style MLM) SSL branches
        // (Chung et al. 2021 arXiv:2108.06209). Standalone binder
        // vs INTERNAL subgraph inside `vieneu` / `seamless_m4t_v2_
        // large` composites. Distinct arch tag `w2v-bert-2` from
        // sibling SSL encoders (hubert / wav2vec2_ctc / data2vec-
        // audio) — Conformer vs vanilla Transformer body + combined
        // SSL objectives = silently sharing would mis-route runtime
        // dispatch (FR-EX-08). Category `asr`. Grouped here for the
        // uniform verify shape (BF16 pass-through skeleton mirror of
        // hubert_large_ls960 / moonshine_base / musicgen_small /
        // openwakeword).
        | ModelKind::W2vBert2
        // 2026-08-01 Wave 5 music-generation add: AudioLDM 2
        // (`cvssp/audioldm2`, cc-by-nc-sa-4.0). Text-to-audio latent-
        // diffusion generator (Liu et al. 2024 arXiv:2308.05734).
        // Emits the same standard `vokra.model.{arch,name,category}`
        // + `vokra.provenance.*` triple as every sibling BF16 pass-
        // through skeleton, with `category = "music"` (shared with
        // sibling MusicGen family — music-tree taxonomy per
        // 2026-07-30 scope expansion). **Publish blocked** by the
        // doubly-restrictive NonCommercialShareAlike default (NC
        // gate + SA cascade); the uniform verify shape still applies
        // because the runtime-side arch/name/category/upstream/license
        // triple lookup does not depend on the publish gate.
        | ModelKind::AudioLdm2
        // 2026-08-02 Wave 8 sibling: AudioLDM 2 Large
        // (`cvssp/audioldm2-large`, cc-by-nc-sa-4.0). Wider/deeper
        // sibling of the base AudioLDM 2 variant — same multi-encoder
        // bundle topology (VAE + U-Net + HiFi-GAN + T5 + CLAP + GPT-2
        // audio-caption LM), only model dims differ. Reuses the base
        // BF16 pass-through converter arm (sibling wrapper — no new
        // models/*.rs module). Emits the same standard `vokra.model.
        // {arch,name,category}` + `vokra.provenance.*` triple as
        // sibling base, with `category = "music"` shared. **Publish
        // blocked** by the doubly-restrictive NonCommercialShareAlike
        // default (NC gate + SA cascade); the uniform verify shape
        // still applies because the runtime-side lookup does not
        // depend on the publish gate.
        | ModelKind::AudioLdm2Large
        // Wave D 2026-08-15: AudioSR (`haoheliu/audiosr_basic` /
        // `haoheliu/audiosr_speech`, MIT — arXiv:2309.07314). Audio
        // super-resolution / bandwidth extension, a **brand-new
        // capability category** for Vokra: `category =
        // "super-resolution"`, deliberately NOT the `enhancement`
        // cohort. Both checkpoints stamp
        // `vokra.provenance.upstream_hf` (the weights live on HF) so
        // they join this group rather than the GitHub-only
        // `upstream_url` arm — note the converter ALSO stamps
        // `vokra.provenance.upstream_url` with the MIT code repo,
        // because the GitHub tree is the config authority; this arm
        // simply reports the HF anchor, which is the one that
        // identifies the weights.
        | ModelKind::AudioSr
        | ModelKind::AudioSrSpeech
        // 2026-08-01 Wave 5 music-separation add: BS-Roformer /
        // Mel-Band Roformer (`chenmozhijin/BSRoformer-GGUF`, **weight
        // provenance unclear**). First music-source-separation
        // converter (Lu et al. 2023 arXiv:2310.01809 dual-path
        // frequency-band Transformer over STFT). Emits the same
        // standard `vokra.model.{arch,name,category}` +
        // `vokra.provenance.*` triple as every sibling BF16 pass-
        // through skeleton, with `category = "separation"` (shared
        // with the SepFormer speech-separation family — BS-Roformer
        // is the music-vocals analogue). **Publish blocked** by the
        // fail-closed `LicenseClass::RedistributionForbidden` default
        // (a converter cannot know which SPDX id covers the caller's
        // checkpoint; owner ADR selecting a specific checkpoint +
        // license required). The uniform verify shape still applies
        // because the runtime-side arch/name/category/upstream/license
        // triple lookup does not depend on the publish gate.
        | ModelKind::BsRoformer
        // 2026-08-02 Wave residual: openWakeWord (dscripka,
        // apache-2.0). Small custom-KWS MLP/CNN family over
        // precomputed melspec — audio-dialect `kws` op entry
        // (FR-OP `kws`). Distinct arch tag `openwakeword`,
        // category `kws`. BF16 pass-through skeleton — verified
        // via the uniform arch/name/category/upstream_hf/license
        // triple lookup shape below.
        | ModelKind::Openwakeword
        // 2026-08-02 Wave residual: Moonshine-Tiny (UsefulSensors,
        // MIT). 27M raw-audio transformer enc-dec ASR (arXiv:
        // 2410.15608) — distinct arch tag `moonshine` from sibling
        // Whisper (raw-audio Conv1D front-end + rotary + SwiGLU vs
        // mel + sinusoidal + GELU). Grouped here for the uniform
        // arch/name/category/upstream_hf/license triple verify shape.
        | ModelKind::MoonshineTiny
        // 2026-08-02 Wave residual: Moonshine-Base (UsefulSensors,
        // MIT). 61.5M raw-audio transformer enc-dec ASR (arXiv:
        // 2410.15608) — sibling to Moonshine-Tiny (same arch family,
        // wider/deeper backbone). Grouped here for the uniform
        // arch/name/category/upstream_hf/license triple verify shape.
        | ModelKind::MoonshineBase
        // 2026-08-02 Wave residual: Demucs (HT-Demucs) (facebook/demucs,
        // MIT). Hybrid transformer Demucs (Rouard et al. 2023, arXiv:
        // 2211.08553) — U-Net waveform branch + spectrogram branch +
        // cross-domain self-attention, 4-source music separation.
        // Distinct arch tag `demucs` from sibling SepFormer / TIGER
        // separators (different internal domain + different output
        // branching — FR-EX-08 forbids silent misroute across separator
        // families). Grouped here for the uniform arch/name/category/
        // upstream_hf/license triple verify shape.
        | ModelKind::DemucsHtdemucs
        // 2026-08-02 Wave residual: Ultravox v0.5 (Llama-3.2-1B)
        // (`fixie-ai/ultravox-v0_5-llama-3_2-1b`, MIT). Audio-text-to-
        // text multimodal = Llama-3.2-1B decoder + Whisper encoder +
        // projection adapter. Distinct arch tag `ultravox` from sibling
        // Voxtral (Mistral decoder) / Qwen2-Audio (Qwen2 decoder) — the
        // decoder backbone fixes tensor layout + tokenizer + rope base,
        // so FR-EX-08 forbids silent shape misroute across the three.
        // Grouped here for the uniform arch/name/category/upstream_hf/
        // license triple verify shape.
        | ModelKind::UltravoxV05Llama321b
        // 2026-08-02 Wave residual: Coqui XTTS-v2 (`coqui/XTTS-v2`,
        // coqui-public-model-license). Multilingual zero-shot voice-cloning
        // TTS = GPT-2 backbone + DVAE + HiFi-GAN vocoder head. Distinct
        // arch tag `xtts` from sibling piper-plus (VITS2) / Kokoro
        // (iSTFTNet) / CosyVoice2 (FSQ + HiFTNet) — FR-EX-08 forbids silent
        // shape misroute across TTS families. Grouped here for the uniform
        // arch/name/category/upstream_hf/license triple verify shape.
        | ModelKind::XttsV2
        // 2026-08-02 Wave residual: ConvTasNet Libri1Mix Enhancement
        // (`JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`, cc-by-sa-4.0).
        // First Copyleft-tier separator entry — Asteroid recipe fine-tune,
        // single-speaker enhancement head on Libri1Mix 16 kHz. Distinct
        // arch tag `conv_tasnet` from sibling separator families (sepformer
        // / demucs / tiger_separator / bs_roformer / mp_senet) — FR-EX-08
        // forbids silent shape misroute across separator families. Category
        // `enhancement` (single-output enhancement head — mirrors the
        // SepFormer WHAM / WHAMR / DNS-4 sibling posture). Grouped here for
        // the uniform arch/name/category/upstream_hf/license triple verify
        // shape.
        | ModelKind::ConvTasnetLibri1mix
        // 2026-08-02 Wave residual: Meta Seamless-M4T-v2-Large
        // (`facebook/seamless-m4t-v2-large`, cc-by-nc-4.0). 2.3B unified
        // any-to-any speech-and-text translation model — ASR + T2TT +
        // S2TT + T2ST + S2ST across ~100 source / ~35 target speech
        // languages. Ships 2 safetensors shards + `.pt` duplicates +
        // `vocoder_v2.pt`. Distinct arch tag `unity-2` (Meta's fairseq2
        // dispatch name) covering the 4 subgraphs (w2v-BERT enc + text
        // dec + T2U + HiFi-GAN vocoder) — FR-EX-08 forbids silent shape
        // misroute across sibling M4T v1 / MMS / Whisper families.
        // Category `s2s` (shared with baichuan_audio / step_audio2_mini).
        // T4 tier NonCommercial default per X-Codec 2 / MusicGen family
        // precedent. Grouped here for the uniform arch/name/category/
        // upstream_hf/license triple verify shape.
        | ModelKind::SeamlessM4tV2Large
        // coverage-audit wave-a (2026-08-03): FRCRN speech-enhancement
        // skeleton — same generic `vokra.model.{arch,name,category}` +
        // `vokra.provenance.{upstream_hf,license,weight_license}` chunk
        // set as the sibling BF16 pass-through models, so it verifies
        // through the same shared-arm surface.
        | ModelKind::Frcrn
        // ---- coverage-audit 2026-08-03 Wave B fast-track (13 variants) ----
        // All share the BF16 pass-through skeleton contract with FRCRN /
        // Wavtokenizer / the SoTA Phase 5 fleet: the converter stamps
        // vokra.model.{arch,name,category} + vokra.provenance.{upstream_hf,
        // license,weight_license}, so verify() surfaces the same 5-slot line.
        | ModelKind::Hibiki
        // coverage-audit-2026-08-03 Wave B fast-track (post-audit
        // 2026-08-13): BosonAI Higgs-Audio v3 TTS 4B — same BF16
        // pass-through skeleton contract as sibling wave-B models
        // (vokra.model.{arch,name,category} + vokra.provenance.
        // {upstream_hf,license,weight_license}), verifies through the
        // same shared 5-slot line.
        | ModelKind::HiggsAudioV3Tts4b
        | ModelKind::SberGigaamV3
        | ModelKind::SberGigaamMultilingual
        | ModelKind::ReazonspeechNemoV2
        | ModelKind::MagpiettsV2602
        | ModelKind::ParakeetUnified
        | ModelKind::Canary1bFlash
        | ModelKind::OwsmV4Medium1b
        | ModelKind::ParakeetTdt11b
        | ModelKind::FireredAsrAedL
        // coverage-audit-2026-08-03 Wave B fast-track (post-audit
        // 2026-08-13): FireRedTeam/FireRedASR-LLM-L — same BF16
        // pass-through skeleton contract (vokra.model.{arch,name,
        // category} + vokra.provenance.{upstream_hf,license,
        // weight_license}) as sibling wave-B models, verifies through
        // the same shared 5-slot line. Distinct arch tag from the
        // sibling FireredAsrAedL (Whisper-topology AED) because the
        // LLM release is Conformer + audio-text adapter + Qwen2 LM
        // decoder.
        | ModelKind::FireredAsrLlmL
        | ModelKind::SortformerDiar4spkV1
        | ModelKind::SenseVoiceSmall
        | ModelKind::WhisperMedusaV1
        | ModelKind::NemotronSpeechStreamingV2603
        // ---- coverage-audit-2026-08-03 Wave D T4 (non-commercial batch,
        // 2026-08-04): 3 HF-hosted BF16 pass-through skeletons on the same
        // shared verify surface. ChatTTS / Stable Audio Open Small / JASCO
        // 400M Chords+Drums all stamp `vokra.provenance.upstream_hf` (the
        // HF-hosted flavor). The 2 GitHub-only siblings (facebook_denoiser
        // + nisqa_v2_weight) live in dedicated arms below because they
        // stamp `vokra.provenance.upstream_url` — reading the wrong key
        // would surface `<none>` and hide the URL.
        | ModelKind::ChatTts
        | ModelKind::StableAudioOpenSmall
        | ModelKind::Jasco400mChordsDrums
        // ---- coverage-audit-2026-08-03 Wave A permissive continuation ----
        // (2026-08-04): 5 HF-hosted BF16 pass-through skeletons on the same
        // shared verify surface. UTMOSv2 / HT-Demucs Multi / openWakeWord op /
        // MossFormer2-SS-16K / AudioSeal real weight all stamp
        // `vokra.provenance.upstream_hf`. The 2 GitHub-only siblings
        // (torchaudio_squim + ten_vad) live in the dedicated arm below
        // (grouped with facebook_denoiser + nisqa_v2_weight) because they
        // stamp `vokra.provenance.upstream_url` — reading the wrong key
        // would surface `<none>` and hide the URL.
        | ModelKind::Utmosv2
        | ModelKind::HtdemucsMulti
        | ModelKind::OpenwakewordOp
        | ModelKind::Mossformer2Ss16k
        | ModelKind::AudiosealRealWeight
        // hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder
        // complement wave (2026-08-04): Aratako/MioCodec-25Hz-44.1kHz-v2
        // (MIT). Emits the same standard `vokra.model.{arch,name,
        // category}` + `vokra.provenance.upstream_hf` + `vokra.provenance
        // .{weight_license,license,model_id,source}` triple as the sibling
        // bicodec / neucodec / focalcodec / xcodec2 codec arms; grouped
        // here since the verify surface is a uniform arch/name/category/
        // upstream/license triple readback and the artifact is HF-hosted
        // (upstream_hf stamped, not upstream_url).
        | ModelKind::MioCodec
        // SoTA plan candidate wave (2026-08-04): Neuphonic NeuTTS Air
        // (apache-2.0). Emits the same standard `vokra.model.{arch,
        // name,category}` + `vokra.provenance.upstream_hf` +
        // `vokra.provenance.{weight_license,license,model_id,source}`
        // triple as the sibling BF16 pass-through skeletons (miocodec
        // / neucodec / bicodec / focalcodec / xcodec2); grouped here
        // since the verify surface is a uniform arch/name/category/
        // upstream/license triple readback and the artifact is
        // HF-hosted (upstream_hf stamped, not upstream_url).
        | ModelKind::NeuTtsAir
        // SoTA plan candidate wave (2026-08-04): SpeechBrain
        // SGMSE-VoiceBank (apache-2.0, score-based diffusion speech
        // enhancement — first real-weight consumer of the M3-05
        // flow_sampler + ODE solver op family). Emits the same
        // standard `vokra.model.{arch,name,category}` +
        // `vokra.provenance.upstream_hf` +
        // `vokra.provenance.{weight_license,license,model_id,source}`
        // triple as the sibling BF16 pass-through skeletons
        // (metricgan_plus / mp_senet / sepformer / neutts_air); grouped
        // here since the verify surface is a uniform arch/name/category/
        // upstream/license triple readback and the artifact is
        // HF-hosted (upstream_hf stamped, not upstream_url).
        | ModelKind::Sgmse
        // Music-understanding wave (2026-08-13, post-audit CC-gap): YAMNet
        // — Google Research 521-class AudioSet audio-event classifier
        // (MobileNetV1 depthwise-separable Conv2D backbone, ~15 MB edge
        // model, apache-2.0 default). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = thelou1s/yamnet` — grouped
        // here since the verify surface is a uniform arch/name/category/
        // upstream/license triple readback.
        | ModelKind::Yamnet
        // Music-understanding wave (2026-08-13): MERT — Music undERstanding
        // self-supervised encoder (HuBERT-derived Conv1D + 24-layer
        // Transformer, ~330M params, cc-by-nc-4.0 default). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = m-a-p/MERT-v1-330M`.
        | ModelKind::Mert
        // Music-understanding wave (2026-08-13): MuQ — Mel-RVQ + BEATs
        // teacher self-supervised music encoder (~500M params, license
        // unknown fail-closed default). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = OpenMuQ/MuQ-large-msd-iter`.
        | ModelKind::Muq
        // Music-understanding wave (2026-08-13): Dasheng — Universal audio
        // encoder (speech + music + environmental, MAE ViT/ConvNeXt
        // backbone, ~86M params base, apache-2.0 default). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = mispeech/dasheng-base`.
        | ModelKind::Dasheng
        // Music-understanding wave (2026-08-13): PANNs Cnn14 — 527-class
        // AudioSet audio-tagging backbone (VGG-flavour 14-layer 2D-CNN,
        // ~80M params, license unknown fail-closed default). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = nicofarr/panns_Cnn14`.
        | ModelKind::Panns
        // Music-understanding wave (2026-08-13): Basic-Pitch — Spotify
        // polyphonic audio-to-MIDI pitch-detection (~6 MB CNN over CQT,
        // 3-head posteriorgram output, apache-2.0 default). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = spotify/basic-pitch`.
        | ModelKind::BasicPitch
        // SSL audio-encoder wave (2026-08-13): MAEST — Music AEST
        // (Discogs-pretrained AST SSL music-tagger, ~87M F32 params
        // 30s-pw-129e, cc-by-nc-sa-4.0 default). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = mtg-upf/discogs-maest-30s-pw-129e`.
        | ModelKind::Maest
        // Meta music-gen post-audit CC-gap wave (2026-08-13): MAGNeT
        // Small 10secs — Meta AudioCraft non-autoregressive masked-LM
        // 10-second music generator (~500M params ~2 GB,
        // cc-by-nc-4.0 default = T4 fail-closed). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = facebook/magnet-small-10secs`.
        | ModelKind::MagnetSmall10secs
        // Meta music-gen post-audit CC-gap wave (2026-08-13, Wave D
        // remaining WF7): MAGNeT Medium 30secs — Meta AudioCraft
        // non-autoregressive masked-LM 30-second music generator sibling
        // of `MagnetSmall10secs` (~1.5B params ~5.7 GB = wider hidden +
        // more layers than small + 30 sec span matching MusicGen family
        // max, cc-by-nc-4.0 default = T4 fail-closed). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = facebook/magnet-medium-30secs`.
        | ModelKind::MagnetMedium30secs
        // Meta music-gen post-audit CC-gap wave (2026-08-13, Wave D
        // remaining WF8): MelodyFlow T24 30secs — Meta AudioCraft
        // flow-matching music **editing** model (DiT backbone, 24
        // timesteps, 30 sec / 48 kHz, ~1 B params ~4.0 GB bundle =
        // flow-matching transformer + 48 kHz RVQ codec + T5-base text
        // encoder, cc-by-nc-4.0 default = T4 fail-closed). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = facebook/melodyflow-t24-30secs`.
        | ModelKind::MelodyflowT2430secs
        // Wave 7 2026-08-14 audit follow-up: WavLM-Base-Plus-SV speaker
        // verification (Microsoft, apache-2.0 code, weight license
        // fail-closed BLANK for Copyleft/Unknown sign-off). HF-hosted,
        // stamps `vokra.provenance.upstream_hf = microsoft/wavlm-base-plus-sv`.
        // Speaker-fleet extension over campplus / wespeaker / ecapa_tdnn /
        // titanet / speaker_3d / redimnet.
        | ModelKind::WavlmSv
        // Wave D 2026-08-15: CT-Transformer punctuation restoration
        // (`funasr/ct-punc`, apache-2.0) — the first `punctuation`
        // category entry. HF-hosted, stamps
        // `vokra.provenance.upstream_hf = funasr/ct-punc`, so it belongs
        // on the `upstream_hf` arm rather than the GitHub-only
        // `upstream_url` arm below. The arch-specific `vokra.ct_punc.*`
        // chunk group (including the `punc_list` Array<String>) is
        // deliberately NOT read back here — this arm is the uniform
        // provenance-triple lookup; `vokra-models::ct_punc::CtPunc::from_gguf`
        // is what validates the full topology.
        | ModelKind::CtPunc
        // 2026-08-15 audit follow-up: ReDimNet2 B6-LM speaker backbone
        // (`Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`, apache-2.0).
        // HF-hosted, stamps `vokra.provenance.upstream_hf`, so it belongs
        // on this arm rather than the GitHub-only `upstream_url` arm
        // below. Speaker-fleet sibling of campplus / wespeaker /
        // ecapa_tdnn / titanet / speaker_3d / wavlm_sv. The twelve
        // `vokra.redimnet.*` topology axes are deliberately NOT read back
        // here — that is `vokra-models::redimnet`'s strict-loader job;
        // this arm is the uniform provenance-triple lookup.
        | ModelKind::Redimnet
        // 2026-08-15 audit follow-up: LLaMA-Omni2 streaming S2S
        // (`ICTNLP/LLaMA-Omni2-*`, apache-2.0). HF-hosted, stamps
        // `vokra.provenance.upstream_hf = ICTNLP/LLaMA-Omni2-<variant>`
        // — the repo id is per-release, which is exactly why printing it
        // back matters: it is how an operator confirms that `--model
        // llama-omni2-32b` stamped the 32B identity and not the 7B
        // default. The `vokra.llama_omni2.variant` chunk is validated by
        // `vokra-models::llama_omni2`, not here.
        | ModelKind::LlamaOmni2
        | ModelKind::Wavtokenizer => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_hf")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_hf={upstream} \
                 license={license} weight_license={class}"
            );
        }
        // coverage-audit-2026-08-03 Wave A: NKF-AEC — like the BF16
        // pass-through fleet above, but the upstream is GitHub-only so
        // the provenance key is `vokra.provenance.upstream_url` (rather
        // than `upstream_hf`). Kept as its own arm to surface the
        // correct key in the CLI verify output without silently
        // falling back to `<none>` for a key that will never exist.
        ModelKind::NkfAec => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream} \
                 license={license} weight_license={class}"
            );
        }
        // Coverage-audit 2026-08-03 Wave A: RNNoise v0.2 shares the
        // BF16 pass-through skeleton shape with the Phase 5 fleet
        // above, but stamps `vokra.provenance.upstream_url` (GitHub
        // Release) instead of `vokra.provenance.upstream_hf` — RNNoise
        // does not ship from Hugging Face. Separate verify arm so the
        // URL slot is surfaced instead of a `<none>` fallback that
        // would misleadingly suggest the artifact carries no upstream
        // record.
        ModelKind::Rnnoise => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream_url = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream_url} \
                 license={license} weight_license={class}"
            );
        }
        // Coverage-audit 2026-08-03 Wave A: Microsoft NSNet2 — same
        // `vokra.model.{arch,name,category}` chunks as the Phase 5 fleet,
        // but the upstream is on GitHub (not HuggingFace), so the
        // provenance URL rides `vokra.provenance.upstream_url` instead of
        // `upstream_hf`. Reading the wrong key would silently print
        // "<none>" and hide the URL — dedicated arm keeps the verify
        // output honest.
        ModelKind::Nsnet2 => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream} \
                 license={license} weight_license={class}"
            );
        }
        // coverage-audit-2026-08-03 Wave D T4 (non-commercial batch,
        // 2026-08-04): Facebook Denoiser + NISQA v2 weight are the two
        // GitHub-only entries in this wave (the other three ChatTTS /
        // Stable Audio Open Small / JASCO ride the shared upstream_hf
        // arm above). Both stamp `vokra.provenance.upstream_url` per the
        // NKF-AEC / RNNoise / NSNet2 / DNSMOS precedent — dedicated arms
        // keep the URL slot visible instead of falling through to a
        // `<none>` in an upstream_hf-only readback.
        ModelKind::FacebookDenoiser
        | ModelKind::NisqaV2Weight
        // coverage-audit-2026-08-03 Wave A continuation (TEN-VAD's license
        // classification was corrected to RedistributionForbidden on
        // 2026-08-22): 2 GitHub-only entries (torchaudio_squim + ten_vad) stamp
        // `vokra.provenance.upstream_url` per the NKF-AEC / RNNoise /
        // NSNet2 / facebook_denoiser precedent.
        | ModelKind::TorchaudioSquim
        | ModelKind::TenVad
        // SSL audio-encoder wave (2026-08-13): BEATs — no first-party
        // HF mirror as of 2026-08-13, stamps
        // `vokra.provenance.upstream_url = github.com/microsoft/unilm/
        // tree/master/beats` per the NKF-AEC / RNNoise / NSNet2 /
        // facebook_denoiser precedent. Reading the wrong key would
        // print `<none>` for the URL and hide the actual upstream in
        // the verify output, which is exactly what FR-EX-08 forbids
        // for provenance.
        | ModelKind::Beats
        // SSL audio-encoder wave (2026-08-13): EAT — no HF mirror as
        // of 2026-08-13, stamps
        // `vokra.provenance.upstream_url = github.com/cwx-worst-one/EAT`
        // per the sibling-BEATs precedent.
        | ModelKind::Eat
        // SSL audio-encoder wave (2026-08-13): ATST — no HF mirror as
        // of 2026-08-13, stamps
        // `vokra.provenance.upstream_url = github.com/Audio-WestlakeU/
        // audiossl/tree/main/audiossl/methods/atst` per the
        // sibling-BEATs precedent.
        | ModelKind::Atst
        // SSL audio-encoder wave (2026-08-13): M2D — no HF mirror as
        // of 2026-08-13, upstream LICENSE.pdf non-machine-readable.
        // Stamps `vokra.provenance.upstream_url =
        // github.com/nttcslab/m2d` per the sibling-BEATs precedent.
        | ModelKind::M2d
        // Post-audit CC-gap 2026-08-15: Voila (`maitrix-org/Voila`, mit)
        // — full-duplex S2S dialog family. Stamps
        // `vokra.provenance.upstream_url = github.com/maitrix-org/Voila`
        // (the MIT reference-code tree) rather than `upstream_hf`,
        // because the per-release HF weight repo ids are owner-verified
        // at bind time and the converter asserts none of them. Reading
        // the wrong key here would print `<none>` for the URL and hide
        // the only upstream anchor the artifact carries, which is exactly
        // what FR-EX-08 forbids for provenance.
        | ModelKind::Voila => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream} \
                 license={license} weight_license={class}"
            );
        }
        // Coverage-audit Wave A (2026-08-03): DNSMOS bundle — prints the
        // bundle inventory plus the two upstream checkpoint filenames so
        // an operator can distinguish a full (P.808 + P.835) bundle from
        // a single-variant partial. Reads `vokra.provenance.upstream_url`
        // rather than the BF16-fleet arm's `upstream_hf` because DNSMOS
        // is GitHub-native (no HF mirror).
        ModelKind::Dnsmos => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let bundle = file
                .get("vokra.dnsmos.bundle")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.values
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let sr = file
                .get("vokra.dnsmos.sample_rate")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream} \
                 license={license} bundle=[{bundle}] sample_rate={sr}"
            );
        }
        // Wave 6 2026-08-14 audit follow-up: DTLN-AEC + GTCRN. Both
        // ship with `vokra.provenance.upstream_url` (GitHub, not
        // HuggingFace) same as sibling `NkfAec` / `Rnnoise` / `Nsnet2`
        // fleet — dedicated arms keep the verify output honest instead
        // of falling into a wildcard that would print `<none>` for the
        // upstream URL. Wave 7 2026-08-14 RETRY: StoRM ships identical
        // provenance/upstream_url/category surface (MIT, GitHub-only
        // upstream via Google Drive distribution, category
        // `enhancement`), so it joins the shared arm.
        // Wave D 2026-08-15: DiffSinger joins the same arm — it also ships
        // `vokra.provenance.upstream_url` (GitHub, `openvpi/DiffSinger`;
        // the framework has no canonical HF mirror, individual voicebanks
        // are scattered across hosts). Printing the upstream URL and the
        // weight-license class is doubly load-bearing here: the apache-2.0
        // default is the FRAMEWORK grant, so an operator verifying a
        // converted singer voicebank needs to see at a glance whether the
        // artifact inherited Permissive or carries the voicebank's own
        // (often non-commercial) class from a `--license` override.
        // WeTextProcessing joins this group for its provenance shape only
        // (GitHub `upstream_url`, no HF mirror). It is the one weightless
        // entry in the catalogue: the GGUF holds the compiled OpenFST
        // `tagger.fst` / `verbalizer.fst` grammars as `U8` METADATA ARRAYS
        // (the `vokra.tokenizer.model` precedent — `GgmlType` has no byte
        // dtype), so its tensor count is legitimately 0 and the grammar
        // detail lives in the `vokra.itn.*` chunk group.
        // 2026-08-15 audit follow-up: MT3 and Beat This! join this arm.
        // Both are GitHub-native releases with no HF mirror, so both
        // stamp `vokra.provenance.upstream_url` (`github.com/magenta/mt3`
        // and `github.com/CPJKU/beat_this`) and would print `<none>` for
        // the upstream if they fell onto the `upstream_hf` arm above.
        //
        // Printing `weight_license` back is doubly load-bearing for MT3:
        // its converter hard-maps the class to `Unknown` no matter what
        // `--license` says, because the `gs://mt3/checkpoints/` bucket
        // ships no LICENSE. An operator who passed `--license apache-2.0`
        // and saw only the raw SPDX echoed would reasonably conclude the
        // artifact is cleared for redistribution; the class column is
        // where the fail-closed verdict is visible.
        ModelKind::DtlnAec
        | ModelKind::Gtcrn
        | ModelKind::Storm
        | ModelKind::DiffSinger
        | ModelKind::Mt3
        | ModelKind::BeatThis
        | ModelKind::WeTextProcessing => {
            let arch = file
                .get("vokra.model.arch")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let name = file
                .get("vokra.model.name")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let category = file
                .get("vokra.model.category")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let upstream = file
                .get("vokra.provenance.upstream_url")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let license = file
                .get("vokra.provenance.license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            let class = file
                .get("vokra.provenance.weight_license")
                .and_then(|v| v.as_str())
                .unwrap_or("<none>");
            println!(
                "; arch={arch} name={name} category={category} upstream_url={upstream} \
                 license={license} weight_license={class}"
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
            ("canary-qwen", ModelKind::CanaryQwen),
            ("omniasr-ctc", ModelKind::OmniasrCtc),
            ("distil-whisper", ModelKind::DistilWhisper),
            ("kotoba-whisper", ModelKind::KotobaWhisper),
            ("vits-ja", ModelKind::VitsJa),
            ("styletts2", ModelKind::StyleTts2),
            // SoTA plan Phase 5 VAD-2 (2026-07-30): FunASR FSMN-VAD.
            ("fsmn-vad", ModelKind::FsmnVad),
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

    /// Regression test for Task 25's review finding: the CLI dispatch match
    /// must forward `--config` to `convert_sbv2_file` for `--model sbv2`
    /// instead of silently dropping it through the generic wildcard arm
    /// (which always calls with `config_side_car = None`). Exercises the
    /// `convert_sbv2` helper directly -- the same one the `ModelKind::SbV2`
    /// dispatch arm calls -- with a real config side-car, and confirms the
    /// emitted GGUF actually carries the `vokra.sbv2.*` hparam chunk the
    /// side-car describes. Before the fix, `--model sbv2 --config <x>`
    /// always produced a hparam-empty GGUF because the dispatch match had
    /// no dedicated `SbV2` arm.
    #[test]
    fn convert_sbv2_forwards_config_and_writes_hparams() {
        use vokra_core::gguf::GgufFile;

        let pid = std::process::id();
        let input =
            std::env::temp_dir().join(format!("vokra-convert-main-sbv2-in-{pid}.safetensors"));
        let config =
            std::env::temp_dir().join(format!("vokra-convert-main-sbv2-config-{pid}.json"));
        let output = std::env::temp_dir().join(format!("vokra-convert-main-sbv2-out-{pid}.gguf"));

        // One minimal F32 tensor -- enough to exercise the pass-through
        // path; the point of this test is the hparam chunk, not tensor
        // coverage (mirrors `tests/roundtrip.rs`'s `synthetic_safetensors`).
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let header = r#"{"enc_p.emb.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&payload);
        std::fs::write(&input, &blob).expect("write synthetic safetensors input");

        // Minimal but internally-consistent config JSON covering every
        // field `SbV2Config::parse` requires (`crates/vokra-convert/src/
        // models/sbv2.rs`) -- two upsample stages / one resblock branch
        // with two dilations, so the array-length cross-checks pass.
        let config_json = br#"{
            "d_model": 192, "d_bert": 1024, "d_speaker": 512, "n_speakers": 3,
            "d_style": 256, "d_z": 192, "n_vocab": 178, "n_tones": 3, "d_ff": 768,
            "n_text_layers": 6, "n_flow_layers": 4, "n_sdp_layers": 4,
            "sample_rate": 44100,
            "decoder_initial_channel": 512,
            "decoder_conv_pre_kernel": 7, "decoder_conv_post_kernel": 7,
            "decoder_upsample_rates": [8, 8],
            "decoder_upsample_kernel_sizes": [16, 16],
            "decoder_upsample_out_channels": [256, 128],
            "decoder_resblock_kernel_sizes": [3],
            "decoder_resblock_dilation_counts": [2],
            "decoder_resblock_dilations_flat": [1, 3],
            "decoder_leaky_relu_slope": 0.2
        }"#;
        std::fs::write(&config, config_json).expect("write config side-car");

        let summary = convert_sbv2(&input, Some(config.as_path()), &output, None)
            .expect("convert_sbv2 must succeed with a valid config side-car");
        // Wave-4 CONVERTER-EMIT-EXPLICIT-ZEROS (2026-08-09): converter now
        // emits 4 fabricated zero-slot placeholders for optional real-ckpt
        // tensors the loader still requires (conv_post.bias +
        // style_injector.{proj_scale,proj_bias} + speaker.table). So the
        // total = 1 real synthetic input tensor + 4 fabricated zeros = 5.
        // Before the Wave-4 refactor this was 1; the loader fabricated the
        // zeros. The invariant this test still pins is "converter runs to
        // completion with a valid config side-car" — the exact count is a
        // sibling assertion, updated to match the new converter contract.
        assert_eq!(summary.tensor_count, 5);
        assert!(
            summary
                .notes
                .iter()
                .all(|n| !n.contains("no --config side-car")),
            "a supplied --config must not produce the missing-hparams warning: {:?}",
            summary.notes
        );

        let file = GgufFile::open(&output).expect("reopen the emitted GGUF");
        assert_eq!(
            file.get("vokra.sbv2.d_model").and_then(|v| v.as_u64()),
            Some(192),
            "vokra.sbv2.d_model must round-trip the --config value once the \
             CLI forwards --config through to convert_sbv2_file"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&config).ok();
        std::fs::remove_file(&output).ok();
    }
}
