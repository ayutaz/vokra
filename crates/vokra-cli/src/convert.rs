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
    ModelKind, PolicyPreset, VoxtralConfig, convert_chatterbox_file, convert_chatterbox_nano_file,
    convert_chatterbox_turbo_file, convert_cosyvoice2_file, convert_cosyvoice3_file,
    convert_dac_file, convert_file, convert_file_licensed, convert_file_quantized,
    convert_file_with_policy, convert_irodori_file, convert_kokoro_file, convert_piper_plus_file,
    convert_qwen3_tts_file, convert_vibevoice_file, convert_vits_ja_file, convert_voxcpm2_file,
    convert_voxtral_file_quantized, convert_voxtral_file_with_adapter_config_quantized,
    parse_voxtral_hf_config,
};
use vokra_core::gguf::GgmlType;

pub(crate) const USAGE: &str = "\
vokra-cli convert — convert an upstream checkpoint to Vokra GGUF (offline tool)

USAGE:
    vokra-cli convert --model <whisper|silero-vad|campplus|mimi|csm|moshi|denoise|dia|zonos|kyutai-stt|parakeet-tdt|parakeet-ctc|canary|omniasr-ctc|distil-whisper|kotoba-whisper|chatterbox|chatterbox-turbo|chatterbox-nano|qwen3-tts|vits-ja> --input <ckpt> --output <out.gguf>
    vokra-cli convert --model piper-plus --input <voice.onnx> --config <config.json> --output <out.gguf>
    vokra-cli convert --model kokoro --input <ckpt.safetensors> [--config <config.json>] --output <out.gguf>
    vokra-cli convert --model cosyvoice2 --input <llm.safetensors> [--config <config.json>] --output <out.gguf>
    vokra-cli convert --model cosyvoice3 --input <llm.safetensors> [--config <config.json>] --output <out.gguf>
    vokra-cli convert --model chatterbox --input <t3.safetensors> --output <out.gguf>
    vokra-cli convert --model chatterbox-turbo --input <t3_turbo_v1.safetensors> --output <out.gguf>
    vokra-cli convert --model chatterbox-nano --input <t3_nano_v1.safetensors> --output <out.gguf>
    vokra-cli convert --model qwen3-tts --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model voxcpm --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model vibevoice --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model irodori --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model dac --input <prepared.safetensors> --config <config.json> --output <out.gguf>
    vokra-cli convert --model voxtral --input <ckpt.safetensors | model.safetensors.index.json> \
                      [--config <config.json>] [--adapter-config <adapter.json>] \
                      [--tokenizer <tekken-vocab.bin>] --output <out.gguf>
    vokra-cli convert --model sbv2 --input <voice.safetensors> --output <out.gguf>
    vokra-cli convert --model deberta-v2 --input <bert_ja.safetensors> --output <out.gguf>
    vokra-cli convert --model deberta-v3 --input <bert_en.safetensors> --output <out.gguf>
    vokra-cli convert --model kimi-audio --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model step-audio2-mini --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model baichuan-audio --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model speechtokenizer --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model funcodec --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model xy-tokenizer --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model bicodec --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model neucodec --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model ecapa-tdnn --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model wespeaker --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model speaker-3d --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model emotion2vec --input <model.safetensors> --output <out.gguf>
    vokra-cli convert --model rmvpe --input <model.safetensors> --output <out.gguf>

OPTIONS:
    --model <kind>            whisper (alias: whisper-base) | silero-vad | piper-plus |
                              campplus | kokoro | cosyvoice2 | cosyvoice3 | voxtral | mimi | dac |
                              csm | moshi | denoise | dia | zonos | kyutai-stt |
                              parakeet-tdt | parakeet-ctc | canary | omniasr-ctc |
                              distil-whisper | kotoba-whisper |
                              chatterbox | chatterbox-turbo | chatterbox-nano |
                              qwen3-tts | voxcpm | vibevoice | irodori | vits-ja |
                              sbv2 | deberta-v2 | deberta-v3 | xcodec2 |
                              kimi-audio | step-audio2-mini | baichuan-audio |
                              speechtokenizer | funcodec | xy-tokenizer |
                              bicodec | neucodec | ecapa-tdnn | wespeaker |
                              speaker-3d | emotion2vec | rmvpe
                              (denoise: DeepFilterNet3 — a prepared safetensors
                              from tools/parity/dfn3_prepare_checkpoint.py)
                              (csm / moshi: this delegate runs the plain checkpoint
                              conversion; to embed the tokenizer side-car use the
                              standalone `vokra-convert` binary's --config)
                              (dia: nari-labs Dia-1.6B — a prepared safetensors
                              from the upstream torch .pth; every hparam is
                              transcribed from the primary-source config.json)
                              (zonos: Zyphra Zonos-v0.1-transformer — ships
                              safetensors directly; every hparam is transcribed
                              from the primary-source config.json)
                              (kyutai-stt: Kyutai STT-2.6B-EN — decoder-only
                              English streaming ASR over Mimi tokens; every
                              hparam is transcribed from config.json;
                              weight license = CC-BY 4.0 attribution required)
                              (parakeet-tdt: NVIDIA Parakeet-TDT-0.6B-v3 —
                              English ASR (FastConformer encoder + TDT
                              decoder); ships safetensors directly; every
                              hparam is transcribed from config.json;
                              weight license = CC-BY 4.0 attribution required)
                              (parakeet-ctc: NVIDIA Parakeet-CTC-1.1B —
                              English ASR (FastConformer encoder + CTC
                              head, no RNN-T prediction network); ships
                              BF16 safetensors — pre-widen to F32 offline
                              or wait for the streaming BF16 path; every
                              hparam is transcribed from config.json;
                              weight license = CC-BY 4.0 attribution required)
                              (canary: NVIDIA Canary-1B-v2 — multilingual
                              multi-task ASR / AST (25 European languages;
                              FastConformer encoder + Transformer AED
                              decoder); distributed as a .nemo tarball —
                              use a prepare-checkpoint script to flatten
                              to safetensors first; encoder / decoder
                              hparams stated on the model card are
                              transcribed verbatim, others from the shared
                              FastConformer-Transformer AED reference
                              config; weight license = CC-BY 4.0
                              attribution required)
                              (omniasr-ctc: Meta omniASR-CTC-1B —
                              1600+ language multilingual ASR
                              (wav2vec 2.0 waveform-in encoder + single-
                              Linear CTC head, no RNN-T prediction
                              network); distributed as a fairseq2 .pt +
                              SentencePiece tokenizer — use a
                              prepare-checkpoint script to flatten to
                              safetensors first; every hparam is
                              transcribed verbatim from the fairseq2
                              registry walk (the HF release carries no
                              config.json); weight license = Apache-2.0
                              permissive — no runtime-side attribution
                              obligation, unlike NVIDIA's CC-BY 4.0
                              Parakeet-CTC / Canary)
                              (distil-whisper: HuggingFace distil-large-v3.5
                              — Whisper large-v3 encoder + 2-layer decoder
                              (same op inventory as vanilla Whisper, only
                              n_text_layer differs); ships safetensors
                              directly; every hparam is transcribed
                              verbatim from config.json; weight license =
                              MIT permissive)
                              (kotoba-whisper: Kotoba Technologies
                              kotoba-whisper-v1.x / v2.x / bilingual
                              family — Japanese-distilled Whisper
                              (large-v3 encoder + shrunk 2-layer decoder,
                              distilled on ReazonSpeech Japanese audio);
                              same tensor topology as distil-large-v3.5
                              but distinct upstream release (Kotoba
                              Technologies vs HuggingFace) with Apache-2.0
                              weights (distinct from distil-whisper's
                              MIT); ships safetensors directly; every
                              hparam is transcribed verbatim from
                              config.json — SoTA plan Phase 5 JA-ASR-2;
                              **JA-ASR-2 axis**: n_text_layer=2 is read
                              from checkpoint tensor names via
                              count_layers, never hard-coded to 32)
                              (cosyvoice3: FunAudioLLM Fun-CosyVoice3-0.5B-2512
                              — same architecture as CosyVoice2 (Qwen2 LLM
                              backbone + chunk-aware Flow Matching CFM +
                              HiFTNet vocoder — arXiv:2505.17589 + SoTA
                              plan §1(a) 訂正 2026-07-22); Phase 3
                              refinements (DRSR + Core-Cocktail) are
                              training-side and leave the runtime operators
                              byte-identical to CosyVoice2; `--config`
                              accepts the upstream HF config.json (Qwen2
                              schema — head split + rope / eps / n_ctx
                              are not shape-derivable; without it the
                              runtime refuses the LLM bind loudly per
                              FR-EX-08); weight license = apache-2.0
                              permissive)
                              (chatterbox: Resemble AI Chatterbox-Multilingual
                              — T3 (Llama_520M backbone: hidden=1024 /
                              n_layer=30 / MHA n_head=16 n_head_kv=16 /
                              head_dim=64 / SwiGLU ffn=4096 / RoPE θ=500000
                              llama3-scaled) driving speech-token AR
                              sampling; terminal vocoder = HiFT-GAN (S3Gen)
                              wired through the shared HiFTChain seam
                              (SoTA plan §1(a) 訂正 2026-07-22, same seam
                              as CosyVoice2 / CosyVoice3); multilingual
                              variant covers 23 languages
                              (mtl_tts.py::SUPPORTED_LANGUAGES) — the T3
                              text-token vocabulary of 2454 pins the
                              multilingual identity vs the English-only
                              baseline at 704; no `config.json` on HF
                              (the release stores hparams in Python code),
                              so no --config side-car — every hparam is
                              transcribed verbatim from the upstream
                              source tree; weight license = MIT permissive
                              — no attribution obligation)
                              (chatterbox-turbo: Resemble AI Chatterbox-Turbo
                              — 350M distilled Turbo variant of Chatterbox;
                              backbone family swaps Llama_520M → gpt2-medium
                              (LayerNorm-with-bias + fused-QKV-with-bias +
                              GELU FFN — same 30 × 16 × 1024 shape as base);
                              sample rate 24 kHz → 32 kHz; text vocabulary
                              2454/704 → 50 276 (GPT-2 base 50 257 + 19
                              paralinguistic tags [angry]/[fear]/[surprised]/
                              [whispering]/[cough]/[laugh]/[chuckle]/…);
                              speech vocabulary 8194 → 6563; max
                              text/speech tokens 2048/4096 → 402/604;
                              speech-token-to-mel decoder distilled from
                              10 sampling steps to 1; terminal vocoder =
                              S3Gen HiFT-GAN (same shared HiFTChain seam
                              as base + CosyVoice2/3); every hparam is
                              transcribed verbatim from t3_turbo_v1.yaml
                              (huggingface.co/ResembleAI/chatterbox-turbo)
                              — no --config side-car; weight license =
                              MIT permissive — no attribution obligation)
                              (chatterbox-nano: Resemble AI Chatterbox-Nano
                              — compact 110M-parameter architecture
                              advertised at ~3x realtime on an 8-core CPU;
                              keeps base's Llama_520M backbone (SwiGLU +
                              RMSNorm + RoPE — 30 layers × 16 heads ×
                              1024 hidden, head_dim=64, ffn=4096,
                              rope_theta=500000, rms_norm_eps=1e-5) —
                              distinct from Turbo which swaps the
                              backbone to gpt2-medium; adopts Turbo's
                              low-latency profile: sample rate 24 kHz
                              → 32 kHz, text vocabulary 2454/704 →
                              50 276 (GPT-2 base 50 257 + 19
                              paralinguistic tags), speech vocabulary
                              8194 → 6563, max text/speech tokens
                              2048/4096 → 402/604, 1-step distilled
                              mel decoder; distinguishing sentinel:
                              stop_text_token = 50256 (GPT-2
                              <|endoftext|>) — distinct from both base
                              and Turbo which use 0; terminal vocoder
                              = S3Gen HiFT-GAN (same shared HiFTChain
                              seam as base + Turbo + CosyVoice2/3);
                              every hparam is transcribed verbatim
                              from t3_nano_v1.yaml
                              (huggingface.co/ResembleAI/chatterbox-nano)
                              — no --config side-car; weight license =
                              MIT permissive — no attribution obligation)
                              (qwen3-tts: Alibaba Qwen3-TTS-12Hz-0.6B-Base
                              — discrete multi-codebook LM (Qwen3-flavour
                              28-layer talker + 5-layer parallel
                              code-predictor + shared Qwen3-TTS-Codec seam
                              via vokra_ops::qwen3_tts_codec, 16-quantizer
                              semantic + acoustic split RVQ at 12.5 Hz);
                              talker axes: hidden=1024 / n_layer=28 /
                              GQA n_head=16 n_head_kv=8 / head_dim=128 /
                              SwiGLU ffn=3072 / RoPE θ=1000000 /
                              RMSNorm ε=1e-6 / speech_vocab=3072 /
                              text_vocab=151936 / max_positions=32768;
                              code predictor axes: n_layer=5 /
                              acoustic_vocab=2048; speaker encoder
                              24 kHz / 1024-dim embedding; distinct arch
                              tag from CosyVoice2/3 because Qwen3-TTS is
                              codec-LM not vocoder-LM (terminal step =
                              qwen3_tts_codec, NOT HiFTChain); upstream
                              ships BF16 (~0.9 GB) — pre-widen to F32
                              offline or wait for the streaming BF16
                              pass-through path; every hparam is
                              transcribed verbatim from config.json
                              (huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
                              — no --config side-car; weight license =
                              apache-2.0 permissive end-to-end — no
                              attribution obligation on the runtime side)
                              (vits-ja: ESPnet-family Japanese plain
                              VITS — Kim et al. 2021 VITS + plain
                              HiFi-GAN generator, as shipped by
                              ESPnet's egs2/jsut/tts1/conf/tuning/
                              train_vits.yaml + jvs/tts1/finetune_vits
                              + COEIROINK deployments; distinct arch
                              tag from piper-plus because plain VITS
                              decodes through a HiFi-GAN generator
                              directly while piper-plus (MB-iSTFT-VITS2)
                              decodes through a sub-band iSTFT + PQMF
                              post-net; every hparam is transcribed
                              verbatim from the ESPnet primary sources
                              — SoTA plan Phase 5 JA-TTS-2; **⚠️
                              weight redistribution default is
                              `RedistributionForbidden`**: JSUT / JVS /
                              COEIROINK corpus terms forbid
                              trained-weight redistribution — architecture
                              rides Apache-2.0 (ESPnet) + MIT
                              (jaywalnut310/vits) and is always
                              independently implementable; override
                              with --license <spdx> at conversion time
                              if trained on a permissive corpus)
                              (sbv2: Style-Bert-VITS2 v2 — a litagin02/
                              style_bert_vits2-family multilingual (JA+EN)
                              base safetensors checkpoint; this delegate
                              performs the byte-exact F32/F16/BF16 tensor
                              pass-through only, under upstream safetensors
                              names — it does NOT accept a --config side-car
                              here, so the vokra.sbv2.* hparam chunk
                              SbV2Model::from_gguf needs is omitted (use the
                              standalone `vokra-convert` binary's own --model
                              sbv2 --config <config.json> for a
                              hparam-complete GGUF); weight license defaults
                              to agpl-3.0 (LicenseClass::Copyleft —
                              redistribution permitted, original license
                              text must be preserved); tensor-name mapping
                              to the sbv2.* hierarchy from_gguf reads is a
                              follow-up (Task 30) — today's output is a
                              provenance-correct, byte-faithful staging
                              artifact, not yet loadable by from_gguf)
                              (deberta-v2 / deberta-v3: ku-nlp DeBERTa v2 /
                              v3 Japanese-character BERT checkpoints — the
                              JA / EN text encoders SBV2's SbV2Model wires
                              in as --bert-ja / --bert-en (`vokra-cli run`);
                              a HF transformers deberta_v2 / deberta_v3
                              safetensors checkpoint. F32/F16/BF16 tensors
                              pass through verbatim under upstream HF names
                              with a best-effort vokra.bert.deberta_v{2,3}.*
                              hparam chunk (shape-derived where possible, no
                              --config needed); weight license defaults to
                              cc-by-sa-4.0 for deberta-v2 and mit for
                              deberta-v3 (each model's own upstream HF
                              model-card license, not necessarily the same
                              as the deberta_v2/deberta_v3 *code* in HF
                              transformers, which is Apache-2.0); same Task
                              30 tensor-name-mapping caveat as sbv2 above)
    --input <path>            upstream checkpoint file. For voxtral, a
                              `*.index.json` path reads every shard listed in
                              its weight_map (the raw sharded BF16 release)
    --config <path>           piper-plus config.json (piper-plus only) OR Kokoro
                              config.json (misaki phoneme symbols + voice names;
                              omit to emit the p0..p_{n-1} placeholder table) OR
                              the upstream HF config.json for cosyvoice2 (Qwen2
                              schema: attention head split + rope_theta/
                              rms_norm_eps/n_ctx — not shape-derivable) OR the
                              upstream Voxtral/Mistral config.json (RoPE base,
                              RMSNorm eps, GQA head split incl. head_dim,
                              vocab, max positions — cross-validated against
                              the checkpoint shapes) OR the DAC prepare-script
                              config.json (required for dac — from
                              tools/parity/dac_prepare_checkpoint.py)
    --adapter-config <path>   Voxtral audio-adapter side-car JSON (M3-10 Wave 8):
                              writes `vokra.voxtral.adapter.*` metadata so the
                              runtime binds the checkpoint's adapter tensors
                              and routes ASR through the audio-conditioned
                              path (see docs/tickets/m3/M3-10*.md). Omit for
                              the honest LM-continuation path.
    --tokenizer <path>        Voxtral only: raw tokenizer bytes embedded
                              verbatim into `vokra.tokenizer.model` (the
                              tekken compact-vocab blob). REQUIRED for a
                              usable ASR GGUF — without it the runtime can
                              neither detokenize nor build the trained
                              transcription prompt (both are explicit
                              errors, never silent).
    --output <path>           GGUF file to write
    --quantize <kind>         K-quantize weight matrices: q4_k | q5_k | q6_k
                              (whisper and voxtral). For whisper it is an alias
                              for --policy-preset whisper_q4_k (when kind=q4_k).
                              For voxtral it REQUIRES --config: without it the
                              GGUF carries `0` hparam sentinels the runtime
                              refuses (FR-EX-08). Biases, norms and any tensor
                              that is not a whole number of 256-element
                              super-blocks stay full precision.
    --policy-preset <preset>  M2-08 quantization policy preset (whisper only):
                              vocoder_safe (default) | whisper_q4_k | fp16
    --license <spdx>          Override the converter's built-in weight-license
                              stamp with the caller-supplied SPDX id (e.g.
                              `cc-by-nc-4.0` or `apache-2.0`). Honored on the
                              generic fallthrough dispatch only (whisper /
                              piper-plus / voxtral / kokoro / dac / chatterbox
                              family paths ignore this flag today). Mutually
                              exclusive with --quantize / --policy-preset —
                              use `vokra-convert restamp` to change the
                              license on a quantized GGUF after the fact.
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
    /// P2 cc-10 — Voxtral only. Raw tokenizer bytes to embed verbatim into
    /// `vokra.tokenizer.model`. Without it the emitted GGUF carries no
    /// tokenizer, and the runtime can neither detokenize nor build the
    /// trained transcription prompt (both surface explicit errors) — so a
    /// CLI-only conversion was previously unusable through `vokra-cli run`.
    tokenizer: Option<PathBuf>,
    output: PathBuf,
    quant: Option<GgmlType>,
    policy: Option<PolicyPreset>,
    /// SoTA plan Phase 5 codec (2026-07-28) — mirror of `vokra-convert`'s
    /// `--license` flag. Overrides the converter's built-in weight-license
    /// stamp with the caller-supplied SPDX id (e.g. a caller who obtained
    /// the weight under a distinct license from the module default). Only
    /// honored on the generic fallthrough dispatch — the whisper /
    /// piper-plus / voxtral / kokoro / dac / chatterbox family paths take
    /// their own tailored routes and ignore this flag (loudly, if it is
    /// passed alongside them).
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
    let mut adapter_config: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut quant: Option<GgmlType> = None;
    let mut policy: Option<PolicyPreset> = None;
    let mut license: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                let v = args.get(i + 1).ok_or("--model requires a value")?;
                model = Some(ModelKind::from_arg(v).ok_or_else(|| {
                    format!(
                        "unknown model `{v}` \
                         (whisper [alias: whisper-base] | silero-vad | piper-plus | \
                         campplus | kokoro | cosyvoice2 | cosyvoice3 | voxtral | mimi | dac | \
                         csm | moshi | denoise | dia | zonos | kyutai-stt | \
                         parakeet-tdt | parakeet-ctc | canary | omniasr-ctc | \
                         distil-whisper | kotoba-whisper | \
                         chatterbox | chatterbox-turbo | chatterbox-nano | \
                         qwen3-tts | voxcpm | vibevoice | irodori | vits-ja | \
                         sbv2 | deberta-v2 | deberta-v3 | xcodec2 | \
                         kimi-audio | step-audio2-mini | baichuan-audio | \
                         speechtokenizer | funcodec | xy-tokenizer | \
                         bicodec | neucodec | ecapa-tdnn | wespeaker | \
                         speaker-3d | emotion2vec | rmvpe)"
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
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--tokenizer requires a value")?,
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
            "--license" => {
                let v = args.get(i + 1).ok_or("--license requires an SPDX id")?;
                license = Some(v.clone());
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
        tokenizer,
        output: output.ok_or("--output is required")?,
        quant,
        policy,
        license,
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

    // `--adapter-config` / `--tokenizer` are Voxtral-only side-cars. Passing
    // one on another model would previously be dropped without a word;
    // reject instead (FR-EX-08 — never silently ignore a user flag).
    if !matches!(model, ModelKind::Voxtral) {
        if p.adapter_config.is_some() {
            return Err(
                "--adapter-config is only supported for --model voxtral (it writes the \
                 `vokra.voxtral.adapter.*` metadata chunk)"
                    .to_owned(),
            );
        }
        if p.tokenizer.is_some() {
            return Err(
                "--tokenizer is only supported for --model voxtral. Other archs embed their \
                 tokenizer through their own path (whisper: the converter bakes the vocab; \
                 csm / moshi: the standalone `vokra-convert` binary's --config side-car)"
                    .to_owned(),
            );
        }
    }

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
            // M5-15-T37: Voxtral accepts `--quantize` (the second model to do
            // so, after whisper). `--policy-preset` is still whisper-only —
            // the per-tensor policy machinery (`MinDtypeRegistry`, the
            // `vokra.quant.*` chunk) is a whisper path in M2-08 and pretending
            // otherwise would silently ignore the flag.
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            // The base config is the upstream HF config.json (RoPE base,
            // RMSNorm eps, GQA head split incl. the decoupled head_dim,
            // vocab size, max positions). Omitting it leaves the shape-only
            // sentinels the runtime rejects at forward (FR-EX-08) — the raw
            // sharded release always ships one, so real conversions pass it.
            let mut base_cfg = match &p.config {
                Some(cfg_path) => {
                    let bytes = std::fs::read(cfg_path)
                        .map_err(|e| format!("--config {}: {e}", cfg_path.display()))?;
                    parse_voxtral_hf_config(&bytes).map_err(|e| e.to_string())?
                }
                None => VoxtralConfig::default(),
            };
            // P2 cc-10: embed the tokenizer verbatim when supplied. Without
            // it the GGUF carries no `vokra.tokenizer.model`, and the
            // runtime can neither detokenize nor build the trained
            // transcription prompt.
            if let Some(tok_path) = &p.tokenizer {
                let bytes = std::fs::read(tok_path)
                    .map_err(|e| format!("--tokenizer {}: {e}", tok_path.display()))?;
                if bytes.is_empty() {
                    return Err(format!(
                        "--tokenizer {}: file is empty — refusing to embed a zero-length \
                         tokenizer chunk",
                        tok_path.display()
                    ));
                }
                base_cfg.tokenizer_bytes = Some(bytes);
            }
            // `--quantize` requires the base config: the shape-only path
            // writes `0` hparam sentinels the runtime refuses at forward
            // (FR-EX-08), so quantizing it would produce an unloadable file
            // that *looks* like a success. Refuse loudly instead.
            if p.quant.is_some() && p.config.is_none() {
                return Err(
                    "--model voxtral --quantize requires --config <config.json>: the shape-only \
                     path writes `0` hparam sentinels (RoPE base, RMSNorm eps, GQA head split) \
                     that the runtime rejects, so the quantized GGUF would not load"
                        .to_owned(),
                );
            }
            match (&p.config, &p.adapter_config, &p.tokenizer) {
                // M3-10 Wave 8: adapter-conditioned convert (with or without
                // the base config / tokenizer side-cars).
                (_, Some(adapter_json), _) => convert_voxtral_file_with_adapter_config_quantized(
                    &p.input,
                    &base_cfg,
                    adapter_json,
                    &p.output,
                    p.quant,
                ),
                // Config and/or tokenizer without adapter → full hparam
                // chunk, honest LM-continuation posture (no adapter
                // metadata). The tokenizer-only case still needs the
                // cfg-carrying entry point to reach `tokenizer_bytes`.
                (Some(_), None, _) | (None, None, Some(_)) => {
                    convert_voxtral_file_quantized(&p.input, &base_cfg, &p.output, p.quant)
                }
                // Nothing → shape-only conversion (pre-Wave-8 behaviour;
                // `--quantize` was rejected above).
                (None, None, None) => convert_file(model, &p.input, &p.output),
            }
        }
        ModelKind::CosyVoice2 => {
            // Quantization surface is whisper-only; reject rather than
            // silently ignoring.
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            // --config = upstream HF config.json (Qwen2 schema). Optional:
            // without it only the shape-derived hparams are written and the
            // runtime refuses the LLM bind (loud converter note).
            convert_cosyvoice2_file(&p.input, p.config.as_deref(), &p.output)
        }
        ModelKind::CosyVoice3 => {
            // Quantization surface is whisper-only; reject rather than
            // silently ignoring (same posture as CosyVoice2).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            // --config = upstream HF config.json (Qwen2 schema). Same
            // requirement as CosyVoice2: without it only the shape-derived
            // hparams are written and the runtime refuses the LLM bind
            // (loud converter note). SoTA plan Phase 3: the Fun-CosyVoice3
            // pipeline (Qwen2 LLM → chunk-aware CFM → HiFTNet) is
            // topologically identical to CosyVoice2, so the CosyVoice2
            // shape-derivation walk is reused verbatim under the covers.
            convert_cosyvoice3_file(&p.input, p.config.as_deref(), &p.output)
        }
        ModelKind::Dac => {
            // M4-04 T11: DAC needs the prepare-script config side-car (the
            // shape facts live in the upstream .pth metadata the safetensors
            // flattening cannot carry). Quantization is whisper-only.
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            match &p.config {
                Some(config) => convert_dac_file(&p.input, config, &p.output),
                None => {
                    return Err("--model dac requires --config <config.json> (from \
                                tools/parity/dac_prepare_checkpoint.py)"
                        .to_owned());
                }
            }
        }
        ModelKind::Chatterbox => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox has no `config.json`
            // on HF (the release stores every hparam in Python code), so the
            // CLI takes no --config side-car — the transcribed constants in
            // `models::chatterbox` are authoritative. Quantization surface is
            // whisper-only (same posture as CosyVoice3 / dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_chatterbox_file(&p.input, &p.output)
        }
        ModelKind::ChatterboxTurbo => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox-Turbo ships a real
            // `t3_turbo_v1.yaml` alongside the safetensors, but every field
            // on that side-car is fixed for the Turbo release and byte-parallel
            // to the transcribed constants in `models::chatterbox_turbo` — so
            // the CLI takes no --config side-car today. Quantization surface
            // is whisper-only (same posture as base Chatterbox / CosyVoice3 /
            // dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_chatterbox_turbo_file(&p.input, &p.output)
        }
        ModelKind::ChatterboxNano => {
            // SoTA plan Phase 3 (2026-07-24): Chatterbox-Nano ships a real
            // `t3_nano_v1.yaml` alongside the safetensors, but every field
            // on that side-car is fixed for the Nano release and byte-parallel
            // to the transcribed constants in `models::chatterbox_nano` — so
            // the CLI takes no --config side-car today. Quantization surface
            // is whisper-only (same posture as base Chatterbox / Chatterbox-
            // Turbo / CosyVoice3 / dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_chatterbox_nano_file(&p.input, &p.output)
        }
        ModelKind::Qwen3Tts => {
            // SoTA plan Phase 3 (2026-07-24): Qwen3-TTS-12Hz-0.6B-Base ships
            // a real `config.json`, but every field is fixed for the 0.6B
            // release and byte-parallel to the transcribed constants in
            // `models::qwen3_tts` — so the CLI takes no --config side-car
            // today (a future 0.6B-CustomVoice / 0.6B-VoiceDesign / 1.7B
            // variant that reshapes the backbone would demand one).
            // Quantization surface is whisper-only (same posture as
            // Chatterbox family / CosyVoice3 / dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_qwen3_tts_file(&p.input, &p.output)
        }
        ModelKind::VoxCpm2 => {
            // SoTA plan Phase 4 (2026-07-24): VoxCPM-0.5B ships a real
            // `config.json`, but every field is fixed for the 0.5B release
            // and byte-parallel to the transcribed constants in
            // `models::voxcpm2` — so the CLI takes no --config side-car
            // today (a future 0.5B-CustomVoice / 1.5B variant that reshapes
            // the LM backbone or the AudioVAE would demand one).
            // Quantization surface is whisper-only (same posture as
            // Qwen3-TTS / Chatterbox family / CosyVoice3 / dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_voxcpm2_file(&p.input, &p.output)
        }
        ModelKind::VibeVoice => {
            // SoTA plan Phase 4 (2026-07-24): VibeVoice-1.5B ships a real
            // `config.json`, but every field is fixed for the 1.5B release
            // and byte-parallel to the transcribed constants in
            // `models::vibevoice` — so the CLI takes no --config side-car
            // today (a future 7B variant that reshapes the Qwen2 backbone
            // would demand one).
            // Quantization surface is whisper-only (same posture as
            // VoxCPM / Qwen3-TTS / Chatterbox family / CosyVoice3 / dia /
            // zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_vibevoice_file(&p.input, &p.output)
        }
        ModelKind::Irodori => {
            // SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Irodori-TTS-500M-v3
            // has no upstream `config.json` — every hparam lives in the
            // `train_500m_v3_phase{1,2}_*.yaml` files at
            // `github.com/Aratako/Irodori-TTS` and is fixed for the 500M-v3
            // release, so it is transcribed as compile-time constants in
            // `models::irodori` and the CLI takes no --config side-car
            // today (a future 600M-v3-VoiceDesign / 2.5B variant that
            // reshapes the DiT or adds caption conditioning would demand
            // one). Quantization surface is whisper-only (same posture as
            // VibeVoice / VoxCPM / Qwen3-TTS / Chatterbox family /
            // CosyVoice3 / dia / zonos).
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_irodori_file(&p.input, &p.output)
        }
        ModelKind::VitsJa => {
            // SoTA plan Phase 5 JA-TTS-2 (2026-07-24): plain VITS JA
            // (ESPnet-family Kim et al. 2021 VITS + HiFi-GAN generator)
            // has no upstream `config.json` — every hparam lives in the
            // ESPnet training yamls (`egs2/jsut/tts1/conf/tuning/
            // train_vits.yaml` + `egs2/jvs/tts1/conf/tuning/
            // finetune_vits.yaml`) and is transcribed as compile-time
            // constants in `models::vits_ja` (JSUT 22 kHz single-speaker
            // recipe defaults). No --config side-car today; JVS
            // multi-speaker + full-band 44 kHz + downstream re-training
            // variants share the same tensor topology and will land as
            // a follow-up `--config` axis. Quantization surface is
            // whisper-only (same posture as Irodori / VibeVoice /
            // VoxCPM / Qwen3-TTS / Chatterbox family / CosyVoice3 /
            // dia / zonos).
            //
            // **⚠️  Weight redistribution default is
            // `RedistributionForbidden`**: the JSUT / JVS / COEIROINK
            // corpus terms forbid trained-weight redistribution. A user
            // who trained on a permissive corpus overrides via
            // `vokra-convert --license <spdx>` (the standalone binary)
            // or the shared `convert_file_licensed` path.
            if p.quant.is_some() {
                return Err("--quantize is only supported for whisper".to_owned());
            }
            if p.policy.is_some() {
                return Err("--policy-preset is only supported for whisper".to_owned());
            }
            convert_vits_ja_file(&p.input, &p.output)
        }
        _ => {
            // Ticket precedence: an explicit --policy-preset wins; else the
            // legacy --quantize q4_k alias maps to the whisper_q4_k preset;
            // else fall through to convert_file_quantized (Q5/Q6 legacy
            // shapes), convert_file_licensed (when --license is set — the
            // SoTA-Phase-5 xcodec2 + generic license-override path) or the
            // plain byte-exact path.
            //
            // --license is mutually exclusive with --quantize / --policy-preset
            // for now: the tailored quant paths do not thread the license
            // override, and silently ignoring a user flag is a bug (FR-EX-08).
            if let Some(preset) = p.policy {
                if p.license.is_some() {
                    return Err(
                        "--license and --policy-preset are mutually exclusive (the policy \
                         preset takes its own tailored path that does not thread the license \
                         override; drop --license or use `vokra-convert restamp` after the \
                         quantized convert)"
                            .to_owned(),
                    );
                }
                convert_file_with_policy(model, &p.input, &p.output, preset)
            } else if let Some(q) = p.quant {
                if p.license.is_some() {
                    return Err(
                        "--license and --quantize are mutually exclusive on this dispatch \
                         path (the quant path does not thread the license override; drop \
                         --license or use `vokra-convert restamp` after the quantized convert)"
                            .to_owned(),
                    );
                }
                if q == GgmlType::Q4K {
                    // Backward-compat alias per T06 spec.
                    convert_file_with_policy(model, &p.input, &p.output, PolicyPreset::WhisperQ4K)
                } else {
                    convert_file_quantized(model, &p.input, &p.output, q)
                }
            } else if p.license.is_some() {
                convert_file_licensed(model, &p.input, &p.output, p.license.as_deref())
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
    fn parses_cosyvoice3_with_config() {
        // Config-driven Fun-CosyVoice3 path: parallel to CosyVoice2 (Qwen2
        // schema config), only the arch label differs. The plain
        // `--input`-only path still converts with shape-derived hparams
        // only (and the runtime refuses the LLM bind).
        let p = parse_args(&args(&[
            "--model",
            "cosyvoice3",
            "--input",
            "llm.safetensors",
            "--config",
            "config.json",
            "--output",
            "o.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::CosyVoice3);
        assert_eq!(p.config, Some(PathBuf::from("config.json")));
    }

    /// Every accepted spelling from `ModelKind::from_arg` parses via the CLI
    /// front-end for the Chatterbox family — the family, both HF variant
    /// tags, and the raw `t3_mtl23ls_v{2,3}` checkpoint stems.
    #[test]
    fn parses_chatterbox_variant_ids() {
        for spelling in [
            "chatterbox",
            "chatterbox-multilingual",
            "chatterbox-multilingual-v2",
            "chatterbox-multilingual-v3",
            "chatterbox-mtl23ls-v2",
            "chatterbox-mtl23ls-v3",
            "chatterbox-english",
            "chatterbox_en",
        ] {
            let p = parse_args(&args(&[
                "--model", spelling, "--input", "i", "--output", "o",
            ]))
            .unwrap_or_else(|e| panic!("--model {spelling} should parse: {e}"));
            assert_eq!(p.model, ModelKind::Chatterbox, "--model {spelling}");
            assert!(p.config.is_none(), "chatterbox takes no --config side-car");
        }
    }

    /// Every accepted spelling from `ModelKind::from_arg` parses via the CLI
    /// front-end for the Chatterbox-Turbo family — the canonical HF release
    /// id, the underscore spelling (== the arch tag), the v1 checkpoint
    /// stem, and the sibling ONNX release id (which still routes to the
    /// safetensors converter because the runtime never loads ONNX,
    /// FR-LD-05).
    #[test]
    fn parses_chatterbox_turbo_variant_ids() {
        for spelling in [
            "chatterbox-turbo",
            "chatterbox_turbo",
            "chatterbox-turbo-v1",
            "chatterbox-turbo-onnx",
        ] {
            let p = parse_args(&args(&[
                "--model", spelling, "--input", "i", "--output", "o",
            ]))
            .unwrap_or_else(|e| panic!("--model {spelling} should parse: {e}"));
            assert_eq!(p.model, ModelKind::ChatterboxTurbo, "--model {spelling}");
            assert!(
                p.config.is_none(),
                "chatterbox-turbo takes no --config side-car"
            );
        }
    }

    /// Every accepted spelling from `ModelKind::from_arg` parses via the CLI
    /// front-end for the Chatterbox-Nano family — the canonical HF release
    /// id, the underscore spelling (== the arch tag), and the v1 checkpoint
    /// stem. Nano does not ship an ONNX sibling release, so no `-onnx`
    /// alias here (distinct from Turbo).
    #[test]
    fn parses_chatterbox_nano_variant_ids() {
        for spelling in ["chatterbox-nano", "chatterbox_nano", "chatterbox-nano-v1"] {
            let p = parse_args(&args(&[
                "--model", spelling, "--input", "i", "--output", "o",
            ]))
            .unwrap_or_else(|e| panic!("--model {spelling} should parse: {e}"));
            assert_eq!(p.model, ModelKind::ChatterboxNano, "--model {spelling}");
            assert!(
                p.config.is_none(),
                "chatterbox-nano takes no --config side-car"
            );
        }
    }

    /// Every accepted spelling from `ModelKind::from_arg` parses via the
    /// CLI front-end for the Qwen3-TTS family — the canonical HF release
    /// id, the arch-tag underscore spelling, and the common short forms.
    /// Qwen3-TTS takes no --config side-car today (every hparam is fixed
    /// for the 0.6B-Base release and transcribed as compile-time
    /// constants).
    #[test]
    fn parses_qwen3_tts_variant_ids() {
        for spelling in [
            "qwen3-tts",
            "qwen3_tts",
            "qwen3-tts-0.6b",
            "qwen3-tts-0_6b",
            "qwen3-tts-12hz-0.6b-base",
            "qwen3-tts-12hz-0_6b-base",
            "qwen3-tts-12hz-0.6b",
        ] {
            let p = parse_args(&args(&[
                "--model", spelling, "--input", "i", "--output", "o",
            ]))
            .unwrap_or_else(|e| panic!("--model {spelling} should parse: {e}"));
            assert_eq!(p.model, ModelKind::Qwen3Tts, "--model {spelling}");
            assert!(p.config.is_none(), "qwen3-tts takes no --config side-car");
        }
    }

    #[test]
    fn parses_fun_cosyvoice3_variant_ids() {
        // Every accepted spelling from `ModelKind::from_arg` parses via
        // the CLI front-end (aliases the HF release + fairseq / modelscope
        // variants).
        for spelling in [
            "cosyvoice3",
            "cosyvoice-3",
            "fun-cosyvoice3",
            "fun-cosyvoice-3",
            "fun-cosyvoice3-0.5b",
            "fun-cosyvoice3-0.5b-2512",
            "fun-cosyvoice3-0_5b",
            "fun-cosyvoice3-0_5b-2512",
        ] {
            let p = parse_args(&args(&[
                "--model", spelling, "--input", "i", "--output", "o",
            ]))
            .unwrap_or_else(|e| panic!("--model {spelling} should parse: {e}"));
            assert_eq!(p.model, ModelKind::CosyVoice3, "--model {spelling}");
        }
    }

    #[test]
    fn parses_cosyvoice2_with_config() {
        // Config-driven CosyVoice2 path (P1 #4 / P2 #7 fix): `--config`
        // carries the upstream HF config.json (Qwen2 schema) so the
        // attention head split + rope/eps/n_ctx get written; the plain
        // `--input`-only path still converts with shape-derived hparams
        // only (and the runtime refuses the LLM bind).
        let p = parse_args(&args(&[
            "--model",
            "cosyvoice2",
            "--input",
            "llm.safetensors",
            "--config",
            "config.json",
            "--output",
            "o.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::CosyVoice2);
        assert_eq!(p.config, Some(PathBuf::from("config.json")));
    }

    /// Campaign-1 P3 #11 (campaign-2 cli-enablers Fix B): every kind
    /// `ModelKind::from_arg` accepts parses through the CLI front-end, and
    /// the help text lists each one. No new kinds are added — this pins the
    /// existing loader surface only.
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
            ("cosyvoice3", ModelKind::CosyVoice3),
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
            ("chatterbox", ModelKind::Chatterbox),
            ("chatterbox-turbo", ModelKind::ChatterboxTurbo),
            ("chatterbox-nano", ModelKind::ChatterboxNano),
            ("qwen3-tts", ModelKind::Qwen3Tts),
            ("voxcpm", ModelKind::VoxCpm2),
            ("vibevoice", ModelKind::VibeVoice),
            ("irodori", ModelKind::Irodori),
            ("vits-ja", ModelKind::VitsJa),
            ("sbv2", ModelKind::SbV2),
            ("deberta-v2", ModelKind::DebertaV2),
            ("deberta-v3", ModelKind::DebertaV3),
            ("xcodec2", ModelKind::XCodec2),
            // SoTA plan Phase 5 fleet (2026-07-28): 12 BF16 pass-through
            // skeleton wire-ups. Each entry pins the canonical hyphenated
            // CLI spelling ↔ `ModelKind` variant + confirms the USAGE
            // header lists the name literally (assertion below).
            ("kimi-audio", ModelKind::KimiAudio),
            ("step-audio2-mini", ModelKind::StepAudio2Mini),
            ("baichuan-audio", ModelKind::BaichuanAudio),
            ("speechtokenizer", ModelKind::Speechtokenizer),
            ("funcodec", ModelKind::Funcodec),
            ("xy-tokenizer", ModelKind::XyTokenizer),
            ("bicodec", ModelKind::Bicodec),
            ("neucodec", ModelKind::Neucodec),
            ("ecapa-tdnn", ModelKind::EcapaTdnn),
            ("wespeaker", ModelKind::Wespeaker),
            ("speaker-3d", ModelKind::Speaker3d),
            ("emotion2vec", ModelKind::Emotion2vec),
            // F0 pitch-extractor tier (2026-07-30): RMVPE — the first
            // `category = "f0"` binder in the converter tree.
            ("rmvpe", ModelKind::Rmvpe),
        ];
        for (name, kind) in kinds {
            let p = parse_args(&args(&["--model", name, "--input", "i", "--output", "o"]))
                .unwrap_or_else(|e| panic!("--model {name} should parse: {e}"));
            assert_eq!(p.model, *kind, "--model {name}");
            // `whisper-base` is the documented alias; every canonical
            // spelling appears verbatim in the help text.
            if *name != "whisper-base" {
                assert!(USAGE.contains(name), "USAGE lists `{name}`");
            }
        }
        assert!(
            USAGE.contains("whisper-base"),
            "USAGE documents the whisper-base alias"
        );
    }

    /// M5-15-T37: `--quantize` on voxtral **without** `--config` is a loud
    /// usage error, because the shape-only path writes `0` hparam sentinels
    /// and the resulting GGUF would not load (FR-EX-08). The guard fires
    /// before any file I/O, so no fixture checkpoint is needed.
    #[test]
    fn voxtral_quantize_without_config_is_a_loud_usage_error() {
        let e = main(&args(&[
            "--model",
            "voxtral",
            "--input",
            "/nonexistent/ckpt.safetensors",
            "--output",
            "/nonexistent/out.gguf",
            "--quantize",
            "q6_k",
        ]))
        .unwrap_err();
        assert!(e.contains("requires --config"), "message: {e}");
        assert!(e.contains("sentinels"), "message must say why: {e}");
    }

    /// The quantization surface widened to voxtral only — every other model
    /// that has a dedicated CLI arm keeps its explicit refusal (regression net
    /// for M5-15-T36/T37, which deliberately did **not** open the flag up
    /// wholesale).
    ///
    /// Models with no dedicated arm here (silero-vad / campplus / mimi / csm /
    /// moshi / denoise) fall through to `convert_file_quantized`, which reads
    /// the checkpoint before matching — so their refusal is a **library**-level
    /// contract, pinned by
    /// `vokra_convert`'s `quantization_is_still_refused_for_non_whisper_models`.
    #[test]
    fn quantize_is_still_rejected_for_models_with_a_dedicated_cli_arm() {
        for m in [
            "kokoro",
            "cosyvoice2",
            "cosyvoice3",
            "chatterbox",
            "chatterbox-turbo",
            "chatterbox-nano",
            "qwen3-tts",
            "voxcpm",
            "vibevoice",
            "irodori",
            "piper-plus",
            "dac",
        ] {
            let e = main(&args(&[
                "--model",
                m,
                "--input",
                "/nonexistent/ckpt",
                "--output",
                "/nonexistent/out.gguf",
                "--quantize",
                "q4_k",
            ]))
            .unwrap_err();
            assert!(
                e.contains("--quantize is only supported for whisper"),
                "{m}: expected the whisper-only refusal, got: {e}"
            );
        }
    }

    /// `--policy-preset` stays whisper-only even for voxtral: the M2-08
    /// per-tensor policy machinery is a whisper path, so accepting the flag
    /// would silently ignore it.
    #[test]
    fn policy_preset_is_still_whisper_only_for_voxtral() {
        let e = main(&args(&[
            "--model",
            "voxtral",
            "--input",
            "/nonexistent/ckpt",
            "--output",
            "/nonexistent/out.gguf",
            "--policy-preset",
            "whisper_q4_k",
        ]))
        .unwrap_err();
        assert!(
            e.contains("--policy-preset is only supported for whisper"),
            "message: {e}"
        );
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
    fn parses_voxtral_with_config_and_adapter_config() {
        // P1 fix (2026-07-16): `--config` carries the upstream HF config.json
        // (head_dim / GQA split / RoPE / eps / vocab) alongside the adapter
        // side-car; `--input` may be the sharded `*.index.json`.
        let p = parse_args(&args(&[
            "--model",
            "voxtral",
            "--input",
            "model.safetensors.index.json",
            "--config",
            "config.json",
            "--adapter-config",
            "adapter.json",
            "--output",
            "voxtral.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.model, ModelKind::Voxtral);
        assert_eq!(p.input, PathBuf::from("model.safetensors.index.json"));
        assert_eq!(p.config, Some(PathBuf::from("config.json")));
        assert_eq!(p.adapter_config, Some(PathBuf::from("adapter.json")));
    }

    // ---- P2 cc-10: --tokenizer side-car + Voxtral-only flag scoping ------

    #[test]
    fn parses_voxtral_tokenizer_side_car() {
        let p = parse_args(&args(&[
            "--model",
            "voxtral",
            "--input",
            "model.safetensors.index.json",
            "--config",
            "config.json",
            "--adapter-config",
            "adapter.json",
            "--tokenizer",
            "tekken-compact-vocab.bin",
            "--output",
            "voxtral.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.tokenizer, Some(PathBuf::from("tekken-compact-vocab.bin")));
        // Absent by default.
        let p = parse_args(&args(&[
            "--model",
            "voxtral",
            "--input",
            "v.safetensors",
            "--output",
            "v.gguf",
        ]))
        .expect("valid");
        assert_eq!(p.tokenizer, None);
    }

    #[test]
    fn tokenizer_requires_value() {
        assert_eq!(
            parse_args(&args(&[
                "--model",
                "voxtral",
                "--input",
                "v.safetensors",
                "--output",
                "v.gguf",
                "--tokenizer",
            ]))
            .err()
            .unwrap(),
            "--tokenizer requires a value"
        );
    }

    /// The Voxtral-only side-cars are rejected on other models rather than
    /// silently dropped (FR-EX-08).
    #[test]
    fn voxtral_only_side_cars_are_rejected_on_other_models() {
        let err = main(&args(&[
            "--model",
            "whisper",
            "--input",
            "w.safetensors",
            "--output",
            "w.gguf",
            "--adapter-config",
            "adapter.json",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--adapter-config is only supported for --model voxtral"),
            "got: {err}"
        );
        let err = main(&args(&[
            "--model",
            "whisper",
            "--input",
            "w.safetensors",
            "--output",
            "w.gguf",
            "--tokenizer",
            "tok.bin",
        ]))
        .unwrap_err();
        assert!(
            err.contains("--tokenizer is only supported for --model voxtral"),
            "got: {err}"
        );
    }

    /// An empty `--tokenizer` file is refused before any conversion work —
    /// embedding a zero-length chunk would produce a GGUF whose tokenizer
    /// "exists" but decodes nothing.
    #[test]
    fn empty_tokenizer_file_is_rejected() {
        let dir = std::env::temp_dir();
        let tok = dir.join(format!("vokra-cli-empty-tok-{}.bin", std::process::id()));
        std::fs::write(&tok, b"").unwrap();
        let missing_input = dir.join("definitely-not-here.safetensors");
        let err = main(&args(&[
            "--model",
            "voxtral",
            "--input",
            missing_input.to_str().unwrap(),
            "--output",
            "out.gguf",
            "--tokenizer",
            tok.to_str().unwrap(),
        ]))
        .unwrap_err();
        let _ = std::fs::remove_file(&tok);
        assert!(err.contains("file is empty"), "got: {err}");
    }

    /// The help documents the tokenizer side-car (a Voxtral GGUF without it
    /// cannot detokenize — the CLI must say so).
    #[test]
    fn usage_documents_tokenizer_side_car() {
        assert!(USAGE.contains("--tokenizer"), "USAGE lists --tokenizer");
        assert!(
            USAGE.contains("vokra.tokenizer.model"),
            "USAGE names the chunk the flag writes"
        );
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
