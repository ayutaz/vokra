//! # vokra-convert
//!
//! Vokra's **offline model conversion tool** (FR-TL-01, M0-03): it reads an
//! upstream checkpoint (safetensors for Whisper, ONNX for Silero VAD) and
//! writes a GGUF carrying the model's tensors plus the `vokra.*` metadata
//! chunks that Vokra's runtime understands.
//!
//! # Why this is a separate crate
//!
//! This is the *only* place ONNX / protobuf handling is allowed to live. The
//! runtime crates never load ONNX and never depend on protobuf/abseil/onnx
//! (FR-LD-05, NFR-DS-02). To keep that boundary airtight, `vokra-convert`
//! depends on nothing but `vokra-core` (for its GGUF writer): the safetensors
//! reader, the JSON parser and the ONNX protobuf decoder are all hand-written
//! here with the standard library only — no external crates — so no ONNX
//! dependency can leak toward the runtime. The dependency direction is
//! strictly one-way (`vokra-convert` -> `vokra-core`).
//!
//! # Scope (M0 minimal tool)
//!
//! Independent binary, F32/F16 tensors only. Integration into a richer
//! `vokra-cli` (FR-TL-02) is a v0.1 MVP / M1 concern.
//!
//! # Weight-license provenance (M2-13, FR-CP-05 conduit)
//!
//! A converter can stamp the produced GGUF with its **weight** license class so
//! the runtime's research-flag gate (FR-CP-03) can enforce it, by calling
//! [`vokra_core::stamp_provenance`] on the [`GgufBuilder`](vokra_core::gguf::GgufBuilder)
//! before serializing — it writes the `vokra.provenance.*` chunk. The class is
//! taken from `docs/license-audit.md` §3 (e.g. Whisper / piper-plus = permissive
//! MIT, a future F5-TTS / EnCodec voice = non-commercial). Only the `vokra.*`
//! metadata namespace is touched — no ONNX/protobuf enters the runtime
//! (NFR-DS-02). Per-model stamping in the existing `convert*` functions is a
//! deliberate follow-up (it shifts each model's metadata-key count); the conduit
//! and its round-trip through the runtime classifier are exercised in this
//! crate's tests.

mod json;
mod models;
mod onnx;
mod quantize;
mod safetensors;

use std::fmt;
use std::path::Path;

pub use quantize::{QuantizeError, quantize};
use vokra_core::gguf::GgmlType;

/// Which model's conversion routine to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// An OpenAI Whisper safetensors checkpoint (M2-06-T06). The specific size
    /// (base / small / medium / large-v3 / turbo) is **auto-detected from the
    /// checkpoint tensor shapes** (see `models::whisper` — `d_model`,
    /// `n_audio_layer`, `n_text_layer`, `n_mels` uniquely identify a size); the
    /// caller passes a single `whisper` label. The CLI keeps `whisper-base` as
    /// a backward-compatible alias for pre-M2-06 invocations, and both dispatch
    /// to the same size-detecting path.
    Whisper,
    /// `snakers4/silero-vad` v5 ONNX checkpoint.
    SileroVad,
    /// SaruLab **UTMOS22-strong** neural MOS predictor (M5-15 T14): a
    /// wav2vec2-base SSL encoder + listener/domain conditioning + BLSTM +
    /// regression head, used by `vokra-eval` for the NFR-QL-02 5 % quality
    /// gate. Convert with [`convert_utmos_file`] — it needs the config
    /// side-car that `tools/parity/utmos_prepare_checkpoint.py` emits
    /// alongside the flattened safetensors, so it is not a plain
    /// single-input [`convert_file`] model.
    Utmos,
    /// A piper-plus (MB-iSTFT-VITS2) voice: ONNX graph + `config.json`
    /// (M0-07). Convert with [`convert_piper_plus_file`] — it needs the extra
    /// `config.json` input, so it is not a plain single-input [`convert_file`]
    /// model.
    PiperPlus,
    /// `iic/speech_campplus` (3D-Speaker CAM++) speaker-encoder ONNX checkpoint
    /// (M0-08): 80-d fbank → 192-d speaker embedding for zero-shot voice
    /// conditioning.
    CamPlus,
    /// `hexgrad/Kokoro-82M` safetensors checkpoint (M2-07 foundation): a
    /// StyleTTS 2 派生 iSTFTNet TTS model with a per-voice style-vector
    /// voicepack. Weights are bound verbatim; hparams are shape-driven with
    /// `0` placeholders on the iSTFT triple pending T02 upstream inspection.
    Kokoro,
    /// `iic/CosyVoice2-0.5B` safetensors checkpoint (M3-09 scaffold): a
    /// Text tokenizer + LLM backbone + Flow Matching CFM + Mimi codec +
    /// chunk-aware streaming TTS / S2S model (Apache 2.0 code + weight,
    /// docs/license-audit.md). Weights are bound verbatim; numeric hparams
    /// (`n_layer` / `n_head` / `hidden_dim` / `ffn_dim` / streaming chunk
    /// sizes) are `0`-placeholders pending T02 upstream inspection — the
    /// runtime rejects `0` at load per `CosyVoice2Config::from_gguf`.
    CosyVoice2,
    /// `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` safetensors checkpoint (SoTA
    /// plan Phase 3, 2026-07-24). Same architecture as CosyVoice2 — Qwen2
    /// LLM backbone + chunk-aware Flow Matching CFM + **HiFTNet** vocoder
    /// (arXiv:2505.17589 + `cosyvoice/hifigan/generator.py` `HiFTGenerator`
    /// per SoTA plan §1(a) 訂正 2026-07-22). Phase 3 refinements
    /// (Dual-Resolution Speech Representations + Core-Cocktail Training)
    /// are training-side and leave the runtime forward operators
    /// byte-identical to CosyVoice2, so this converter delegates the tensor
    /// walk + shape derivation to `models::cosyvoice2` and rewrites the
    /// arch label + model name + provenance + metadata chunk prefix so the
    /// runtime dispatches to `vokra-models::cosyvoice3`. Weights are bound
    /// verbatim; the same `--config` (upstream HF Qwen2 `config.json`)
    /// requirement applies — without it the head split / rope / eps / n_ctx
    /// stay `0`-absent and the runtime refuses the LLM bind (FR-EX-08).
    /// Weight license = **apache-2.0** (permissive — no runtime-side
    /// attribution obligation), transcribed from
    /// `huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512` model-card
    /// header. Convert with [`convert_cosyvoice3_file`] to embed the
    /// Qwen2 tokenizer side-car (`vocab.json` + `merges.txt` picked up
    /// from the config's directory, mirroring the CosyVoice2 flow).
    CosyVoice3,
    /// Mistral **Voxtral** safetensors checkpoint (M3-10 foundation): a
    /// Whisper-derived audio encoder plus a Mistral (GQA/RoPE/SwiGLU/RMSNorm)
    /// text decoder for ASR and S2S. The tokenizer and optional side-car
    /// hparams (RoPE base, RMSNorm ε, GQA head split, vocab size, S2S codec
    /// type) are supplied through the config-aware
    /// [`convert_voxtral_file`] path — the shape-only [`convert_file`] path
    /// writes `0` sentinels for those fields (which the runtime loader
    /// rejects at forward time per FR-EX-08).
    Voxtral,
    /// Standalone Mimi (Kyutai) codec checkpoint (M4-04 T10): the moshi-native
    /// safetensors (`kyutai/moshiko-pytorch-bf16`
    /// `tokenizer-e351c8d8-checkpoint125.safetensors`, CC-BY 4.0 weights —
    /// attribution discharged by NOTICE §5). All tensors pass through; the
    /// converter additionally derives the effective (pre-projected) RVQ
    /// codebook tables the runtime decode consumes, and emits
    /// `vokra.mimi.{n_codebooks,codebook_size,d_model}` from the checkpoint
    /// shapes (ADR M4-04 §D-f/§D-k).
    Mimi,
    /// Standalone DAC (Descript Audio Codec) checkpoint (M4-04 T11): a
    /// **prepared** safetensors (from `tools/parity/dac_prepare_checkpoint.py`
    /// — the upstream release is a `.pth`) plus a JSON config side-car.
    /// Convert with [`convert_dac_file`] — the config is required, so this is
    /// not a plain single-input [`convert_file`] model. MIT weights.
    Dac,
    /// `sesame/csm-1b` safetensors checkpoint (M4-05): Sesame CSM-1B, the
    /// S2S speech-generation model (Llama-3.2-1B-flavor backbone +
    /// llama-100M-flavor depth transformer over Mimi RVQ frames; Apache 2.0
    /// code + weight, docs/license-audit.md — the HF repo is gated, T29
    /// owner hand-off). Weights are bound verbatim; flavor dims / RoPE
    /// scaling / rates are transcribed primary-source constants and the two
    /// vocab axes are `0`-placeholders the runtime rejects at load
    /// (FR-EX-08). The Llama-3.2 tokenizer blob is embedded through
    /// [`convert_csm_file`].
    Csm,
    /// `kyutai/moshiko-pytorch-bf16` safetensors checkpoint (M4-06):
    /// Moshi (Helium temporal transformer + depformer), full-duplex S2S
    /// with inner monologue. Weights are CC-BY 4.0 (`AttributionRequired`
    /// — the converter stamps the FR-MD-09 attribution text). The raw
    /// SentencePiece tokenizer embeds through [`convert_moshi_file`].
    Moshi,
    /// Rikorose/DeepFilterNet **DeepFilterNet3** denoiser checkpoint (M4-20
    /// T17): a **prepared** safetensors (from
    /// `tools/parity/dfn3_prepare_checkpoint.py` — the upstream release is a
    /// torch-pickle `.ckpt.best` inside `models/DeepFilterNet3.zip`). Every
    /// inference tensor binds verbatim under its upstream name; the
    /// published DFN3 hyper-parameters ride the `vokra.denoise.*` chunk.
    /// Dual MIT / Apache-2.0 code + weights (docs/license-audit.md).
    Denoise,
    /// nari-labs **Dia-1.6B** safetensors checkpoint (SoTA plan Phase 1-4,
    /// 2026-07-24). Text encoder (12L / 1024d / 16h × 128 head_dim / 4096
    /// FFN) + delayed-AR decoder (18L / 2048d GQA 16Q ÷ 4KV × 128 head_dim /
    /// cross-attn 16Q × 128 / 8192 FFN) over 9 DAC 44.1 kHz codebook
    /// channels with `delay_pattern=[0,8..15]`. Apache 2.0 code + weight.
    /// All hparams transcribed verbatim from `huggingface.co/nari-labs/
    /// Dia-1.6B/config.json`; every F32 / F16 tensor passes through
    /// verbatim. The upstream release ships torch `.pth`, so callers pre-
    /// flatten it to safetensors offline (the CSM / DAC pattern).
    Dia,
    /// Zyphra **Zonos-v0.1-transformer** safetensors checkpoint (SoTA plan
    /// Phase 1-5, 2026-07-24). Single-stack GQA transformer (26L / 2048d /
    /// 16Q ÷ 4KV × 128 head_dim / SwiGLU 8192 inner) with a typed prefix
    /// conditioner (espeak / speaker / Fourier / integer) over 9 DAC 44.1
    /// kHz codebook channels with `delay_pattern=[1..9]`. Apache 2.0 code
    /// plus weight. All hparams (including the 7 conditioner descriptors)
    /// transcribed verbatim from `huggingface.co/Zyphra/Zonos-v0.1-transformer/config.json`;
    /// every F32 / F16 tensor passes through verbatim. Ships safetensors
    /// directly — no `.pth` prepare step (unlike Dia).
    Zonos,
    /// Kyutai **STT-2.6B-EN** safetensors checkpoint (SoTA plan Phase 2,
    /// 2026-07-24). Decoder-only English streaming ASR: a 48-layer /
    /// dim=2048 / MHA transformer (RoPE max_period=100000, RMSNorm
    /// ε=1e-8, SiLU gating, sliding causal attention context=375) that
    /// consumes 32 Mimi audio codebooks per 12.5 Hz frame and emits text
    /// (`text_card=4000`). The depformer is structurally present but
    /// `dep_q=0` so its per-step weights are unused. CC-BY 4.0 weight
    /// (`AttributionRequired` — the converter stamps the FR-MD-09
    /// attribution text). Every hparam is transcribed verbatim from
    /// `huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json`. The
    /// upstream release is BF16 (~5.2 GB) and the streaming-BF16
    /// pass-through path is a follow-up (T29-equivalent — the Moshi
    /// pattern); this M2-13-preserving path handles F32 / F16 checkpoints
    /// today and skips BF16 with the loud "no float tensors" note.
    KyutaiStt,
    /// NVIDIA **Parakeet-TDT-0.6B-v3** safetensors checkpoint (SoTA
    /// plan Phase 2, 2026-07-24). English ASR: a FastConformer encoder
    /// (`num_hidden_layers=24`, `hidden_size=1024`, MHA
    /// `num_attention_heads=num_key_value_heads=8`, `intermediate_size
    /// =4096`, `subsampling_factor=8`, `conv_kernel_size=9`,
    /// `num_mel_bins=128`, `max_position_embeddings=5000`) + a 2-layer
    /// 640-d RNN-T prediction network + a joint / TDT head with
    /// `durations=[0,1,2,3,4]`, `vocab_size=8193` (blank at 8192),
    /// `max_symbols_per_step=10`. CC-BY 4.0 weight
    /// (`AttributionRequired` — the converter stamps the FR-MD-09
    /// attribution text). Every hparam is transcribed verbatim from
    /// `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`.
    /// Reuses the shared `vokra_ops::conformer` (FastConformer encoder
    /// body via `Stacking { factor: 8 }`) and `vokra_ops::rnnt_decode`
    /// (`Tdt` variant) primitives — no per-model op duplication.
    Parakeet,
    /// NVIDIA **Parakeet-CTC-1.1B** safetensors checkpoint (SoTA plan
    /// Phase 2, 2026-07-24). English ASR: a FastConformer encoder
    /// (`num_hidden_layers=42`, `hidden_size=1024`, MHA
    /// `num_attention_heads=num_key_value_heads=8`, `intermediate_size
    /// =4096`, `subsampling_factor=8`, `conv_kernel_size=9`,
    /// **`num_mel_bins=80`** (differs from TDT-0.6B-v3 = 128),
    /// **`attention_bias=true`** (differs from TDT-0.6B-v3 = false),
    /// **`scale_input=true`** (differs from TDT-0.6B-v3 = false),
    /// `max_position_embeddings=5000`) + a single-Linear CTC head with
    /// `vocab_size=1025` (1024 SentencePiece pieces + 1 blank at
    /// `pad_token_id=1024`). **No RNN-T prediction network, no joint /
    /// duration head** — CTC decoding is a host-side runtime function
    /// (`vokra_ops::ctc_decode`). CC-BY 4.0 weight
    /// (`AttributionRequired` — the converter stamps the FR-MD-09
    /// attribution text). Every hparam is transcribed verbatim from
    /// `huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`.
    /// Reuses the shared `vokra_ops::conformer` (FastConformer encoder
    /// body via `Stacking { factor: 8 }`) and `vokra_ops::ctc_decode`
    /// (greedy / beam CTC) primitives — no per-model op duplication.
    ParakeetCtc,
    /// NVIDIA **Canary-1B-v2** safetensors checkpoint (SoTA plan Phase 2,
    /// 2026-07-24). Multilingual multi-task ASR / AST across 25 European
    /// languages: a FastConformer encoder (`num_hidden_layers=32`,
    /// `hidden_size=1024`, MHA `num_attention_heads=num_key_value_heads=8`,
    /// `intermediate_size=4096`, `subsampling_factor=8`,
    /// `conv_kernel_size=9`, `num_mel_bins=128`, `attention_bias=true`,
    /// `scale_input=false`, `max_position_embeddings=5000`) + a
    /// Transformer decoder (`num_layers=8`, `hidden_size=1024`, MHA
    /// `num_attention_heads=8`, `inner_size=4096`,
    /// `max_sequence_length=1024`, `pre_ln=true`, `hidden_act="relu"`)
    /// with cross-attention from the encoder output. Vocab: a unified
    /// SentencePiece with `vocab_size=16 384`, including inline task
    /// tokens `<source_lang>`, `<target_lang>`, `<taskname>`, `<pnc>`,
    /// `<itn>`, `<timestamp>`, `<diarize>`, `<emotion>`. CC-BY 4.0
    /// weight (`AttributionRequired` — the converter stamps the
    /// FR-MD-09 attribution text). Every hparam stated on the model
    /// card is transcribed verbatim; every remaining hparam is
    /// transcribed from the shared FastConformer-Transformer AED
    /// reference config
    /// (`github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`).
    /// Reuses the shared `vokra_ops::conformer` (FastConformer encoder
    /// body via `Stacking { factor: 8 }`) and `vokra_ops::beam_search`
    /// (attention-decoder search — OP-3) primitives — no per-model op
    /// duplication.
    Canary,
    /// HuggingFace **distil-whisper / distil-large-v3.5** safetensors
    /// checkpoint (SoTA plan Phase 2, 2026-07-24). Whisper large-v3
    /// encoder + a 2-layer decoder — same op inventory as vanilla
    /// Whisper, only `n_text_layer` differs. Every hparam
    /// (`d_model=1280`, `n_audio_layer=32`, `n_text_layer=2`,
    /// `n_mels=128`, `vocab_size=51866`, `ffn_dim=5120`,
    /// `n_audio_ctx=1500`, `n_text_ctx=448`) is transcribed verbatim
    /// from `huggingface.co/distil-whisper/distil-large-v3.5/raw/main/
    /// config.json`. MIT weight (`Permissive` — no runtime-side
    /// attribution obligation). The GGUF carries `vokra.model.arch =
    /// "distil-whisper"` (distinct from vanilla Whisper's `"whisper"`)
    /// but the same `vokra.whisper.*` hparam chunk schema — the
    /// "very cheap follow-on" contract in the task. Reuses the shared
    /// Whisper op inventory (STFT / mel filterbank / GEMM / GEMV /
    /// softmax / layer-norm / GELU / conv1d) — no new op is added.
    DistilWhisper,
    /// Kotoba Technologies **kotoba-whisper** family safetensors
    /// checkpoint (SoTA plan Phase 5 JA-ASR-2, 2026-07-24).
    /// Japanese-distilled Whisper: large-v3 encoder (32 layers,
    /// d_model=1280, n_mels=128) + shrunk 2-layer decoder — same
    /// tensor topology as distil-large-v3.5, but distilled on
    /// ReazonSpeech Japanese audio and released under **apache-2.0**
    /// (distil-whisper is MIT). Every hparam (`d_model=1280`,
    /// `n_audio_layer=32`, `n_text_layer=2`, `n_mels=128`,
    /// `vocab_size=51866`, `ffn_dim=5120`, `n_audio_ctx=1500`,
    /// `n_text_ctx=448`) is transcribed verbatim from
    /// `huggingface.co/kotoba-tech/kotoba-whisper-v2.0/raw/main/
    /// config.json`. Apache-2.0 weight (`Permissive` — no runtime-side
    /// attribution obligation). The GGUF carries
    /// `vokra.model.arch = "kotoba-whisper"` (distinct from vanilla
    /// Whisper's `"whisper"` and distil-whisper's `"distil-whisper"`)
    /// but the same `vokra.whisper.*` hparam chunk schema — the
    /// "very cheap follow-on" contract in the task. **JA-ASR-2 axis**:
    /// the converter reads `n_text_layer` from the checkpoint's
    /// tensor names via `count_layers`, never hard-coding to 32.
    /// Reuses the shared Whisper op inventory (STFT / mel filterbank /
    /// GEMM / GEMV / softmax / layer-norm / GELU / conv1d) — no new
    /// op is added.
    KotobaWhisper,
    /// Meta **omniASR-CTC-1B** — the Omnilingual ASR family's 1B
    /// wav2vec 2.0 CTC checkpoint (SoTA plan Phase 2, 2026-07-24).
    /// Multilingual ASR across **1600+ languages** (`facebook/omniASR-CTC-1B`):
    /// a wav2vec 2.0 **waveform-in** encoder (7-layer Conv1D feature
    /// extractor at 320× total downsampling, grouped-Conv1D positional
    /// encoder, 48-layer pre-norm Transformer with
    /// `model_dim=1280`, MHA `num_encoder_attn_heads=16`,
    /// `ffn_inner_dim=5120`, `max_seq_len=4096`) + a single-Linear CTC
    /// head with `target_vocab_size=9812` (v1 SentencePiece char
    /// tokenizer, blank at index 0 per the fairseq2 convention).
    /// **No RNN-T prediction network, no joint / duration head** —
    /// CTC decoding is a host-side runtime function
    /// (`vokra_ops::ctc_decode`). Apache-2.0 weight (`Permissive` —
    /// no runtime-side attribution obligation, unlike NVIDIA's CC-BY 4.0
    /// Parakeet-CTC / Canary). Every hparam is transcribed verbatim
    /// from the fairseq2 registry walk
    /// (`omnilingual_asr/models/wav2vec2_asr/config.py::_1b_asr` →
    /// `wav2vec2_ssl/config.py::_1b_ssl` →
    /// `fairseq2/models/wav2vec2/config.py::large_lv60k`); the HF
    /// release carries no `config.json`, only the `.pt` + a
    /// SentencePiece tokenizer. Reuses `vokra_ops::ctc_decode`
    /// (greedy / beam CTC decoding); the wav2vec 2.0 encoder body is a
    /// distinct topology from FastConformer — no shared
    /// `vokra_ops::wav2vec2_encoder` op today (the "may need new op"
    /// note is deliberately deferred; the scaffold stops at shape /
    /// weight-store flow).
    OmniasrCtc,
    /// Resemble AI **Chatterbox-Multilingual** T3 safetensors checkpoint
    /// (SoTA plan Phase 3, 2026-07-24). MIT weight + code. T3 =
    /// **Llama_520M** backbone (`hidden_size=1024`, `num_hidden_layers=30`,
    /// MHA `num_attention_heads=num_key_value_heads=16`, `head_dim=64`,
    /// SwiGLU `intermediate_size=4096`, `rope_theta=500_000`,
    /// `rms_norm_eps=1e-5`) driving speech-token AR sampling; the
    /// terminal vocoder is HiFT-GAN (S3Gen) — the same `HiFTGenerator`
    /// topology CosyVoice2 / CosyVoice3 use, wired through the shared
    /// `vokra-models::cosyvoice2::hift_chain::HiFTChain` seam per SoTA
    /// plan §1(a) 訂正 2026-07-22 (no new op or backend kernel added).
    /// The multilingual variant is identified by
    /// `text_tokens_dict_size = 2454` (English-only baseline = 704) and
    /// ships 23 languages
    /// (`src/chatterbox/mtl_tts.py::SUPPORTED_LANGUAGES`). Every hparam
    /// is transcribed **verbatim** from `github.com/resemble-ai/chatterbox`
    /// (`src/chatterbox/models/t3/`) — the release ships safetensors +
    /// Python code, no `config.json` on HF, so the primary source is the
    /// code. Convert with [`convert_chatterbox_file`] — the converter
    /// takes no config side-car (every hparam is a compile-time
    /// constant); a variant tag (multilingual vs english-only) is a
    /// caller argument and defaults to multilingual.
    Chatterbox,
    /// Resemble AI **Chatterbox-Turbo** safetensors checkpoint
    /// (SoTA plan Phase 3, 2026-07-24). MIT weight + code. 350M-parameter
    /// distilled Turbo variant of Chatterbox. Backbone family swaps
    /// from **Llama_520M** (base Chatterbox) to **gpt2-medium**
    /// (`hidden_size=1024`, `num_hidden_layers=30`, `num_attention_heads=16`,
    /// `head_dim=64` — MHA with GPT-2's LayerNorm-with-bias + fused-QKV-
    /// with-bias + GELU FFN topology, not Llama's RMSNorm + SwiGLU);
    /// sample rate swaps from 24 kHz to **32 kHz**; text-token vocabulary
    /// swaps from 2454 (multilingual) / 704 (English-only) to **50 276**
    /// (GPT-2 base 50 257 + 19 paralinguistic tags [angry]/[fear]/
    /// [surprised]/[whispering]/[cough]/[laugh]/[chuckle]/… from
    /// `added_tokens.json`); speech-token vocabulary shrinks 8194 → 6563;
    /// max text/speech tokens shrink 2048 → 402 / 4096 → 604 for
    /// low-latency serving; the speech-token-to-mel decoder is distilled
    /// from 10 sampling steps to a single step. Terminal vocoder =
    /// S3Gen HiFT-GAN — same shared `HiFTChain` seam as CosyVoice2 /
    /// CosyVoice3 / base Chatterbox (SoTA plan §1(a) 訂正 2026-07-22, no
    /// new op or backend kernel added). Every hparam transcribed
    /// **verbatim** from `t3_turbo_v1.yaml` at
    /// `huggingface.co/ResembleAI/chatterbox-turbo` (fetched
    /// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). Convert with
    /// [`convert_chatterbox_turbo_file`] — the converter takes no config
    /// side-car (every hparam is fixed for the Turbo release and
    /// transcribed as compile-time constants).
    ChatterboxTurbo,
    /// Resemble AI **Chatterbox-Nano** safetensors checkpoint
    /// (SoTA plan Phase 3, 2026-07-24). MIT weight + code. Compact
    /// 110M-parameter architecture advertised at ~3× realtime on an
    /// 8-core CPU. Keeps base Chatterbox's **Llama_520M** backbone
    /// (SwiGLU + RMSNorm + RoPE — MHA `hidden_size=1024`,
    /// `num_hidden_layers=30`, `num_attention_heads=num_key_value_heads=16`,
    /// `head_dim=64`, `intermediate_size=4096`, `rope_theta=500_000`,
    /// `rms_norm_eps=1e-5`; `t3_nano_v1.yaml::llama_config_name = Llama_520M`
    /// is authoritative over the stale `gpt_transformer_type: gpt2`
    /// training-side legacy flag) — **distinct from Turbo which swaps
    /// the backbone to gpt2-medium**. Adopts Turbo's low-latency
    /// serving profile: sample rate 24 kHz → **32 kHz**; text-token
    /// vocabulary 2454/704 → **50 276** (GPT-2 base 50 257 + 19
    /// paralinguistic tags from `added_tokens.json`); speech-token
    /// vocabulary 8194 → **6563**; max text/speech tokens 2048/4096 →
    /// **402/604**; speech-token-to-mel decoder distilled from 10
    /// sampling steps to a single step. **Distinguishing sentinel**:
    /// `stop_text_token = 50256` (GPT-2 `<|endoftext|>`) — distinct
    /// from both base and Turbo which use `0`. Terminal vocoder =
    /// S3Gen HiFT-GAN (same shared `HiFTChain` seam as CosyVoice2 /
    /// CosyVoice3 / base Chatterbox / Chatterbox-Turbo per SoTA plan
    /// §1(a) 訂正 2026-07-22, no new op or backend kernel added).
    /// Every hparam transcribed **verbatim** from `t3_nano_v1.yaml`
    /// at `huggingface.co/ResembleAI/chatterbox-nano` (fetched
    /// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). Convert with
    /// [`convert_chatterbox_nano_file`] — the converter takes no
    /// config side-car (every hparam is fixed for the Nano release
    /// and transcribed as compile-time constants).
    ChatterboxNano,
    /// Alibaba **Qwen3-TTS-12Hz-0.6B-Base** safetensors checkpoint
    /// (SoTA plan Phase 3, 2026-07-24). **Apache-2.0 end-to-end** —
    /// LM + codec + tokenizer + speaker encoder all under a single
    /// apache-2.0 grant (`huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base`
    /// model-card `license: apache-2.0`, fetched 2026-07-24 — CLAUDE.md
    /// 「ハルシネーション厳禁」). Discrete multi-codebook LM: a
    /// **Qwen3-flavour talker** (decoder-only transformer,
    /// `hidden_size=1024`, `num_hidden_layers=28`, GQA
    /// `num_attention_heads=16` / `num_key_value_heads=8`,
    /// `head_dim=128`, SwiGLU `intermediate_size=3072`,
    /// `rope_theta=1_000_000`, `rms_norm_eps=1e-6`, 3072-per-codebook
    /// speech vocab + 151 936-token Qwen3 shared text vocab,
    /// `max_position_embeddings=32_768`, `position_id_per_seconds=13`,
    /// `text_hidden_size=2048`) plus a **5-layer code-predictor
    /// parallel head** (same GQA / RoPE / RMSNorm axes,
    /// 2048-per-codebook acoustic vocab, emits **16 codebook rows per
    /// step**) plus the shared **Qwen3-TTS-Codec** seam
    /// (`vokra_ops::qwen3_tts_codec` — 16-quantizer semantic +
    /// acoustic split RVQ at 12.5 Hz output rate, 24 kHz PCM).
    /// Speaker encoder: 24 kHz sample rate, 1024-dim embedding.
    /// Every hparam transcribed verbatim from
    /// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`
    /// (`talker.*` / `code_predictor.*`) plus `README.md` (speaker
    /// encoder axes). Distinct arch tag from CosyVoice2/3 because
    /// Qwen3-TTS is **codec-LM**, not vocoder-LM — the terminal step
    /// is `qwen3_tts_codec`, NOT `HiFTChain`; silently sharing either
    /// sibling's arch tag would mis-route the runtime dispatch. Reuses
    /// the existing `qwen3_tts_codec` primitive (SoTA plan Phase 3 TTS
    /// codec op) — no new op or backend kernel is added by this model.
    /// The upstream release is BF16 (~0.9 GB); today's F32/F16
    /// pass-through hits the `skipped_non_float` counter on the BF16
    /// tensors and the converter surfaces the loud "no float tensors"
    /// note. Convert with [`convert_qwen3_tts_file`] — the converter
    /// takes no config side-car (every hparam is fixed for the 0.6B
    /// release and transcribed as compile-time constants).
    Qwen3Tts,
    /// OpenBMB **VoxCPM-0.5B** safetensors checkpoint (SoTA plan Phase 4,
    /// 2026-07-24). Apache-2.0 end-to-end — LM + AudioVAE + Python
    /// pipeline all under a single apache-2.0 grant
    /// (`huggingface.co/openbmb/VoxCPM-0.5B` model-card + LICENSE,
    /// fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). NEW
    /// **class** of TTS vs every earlier target: end-to-end diffusion-
    /// autoregressive with a **continuous VAE + diffusion decoder** —
    /// the terminal decoding hop is neither vocoder-LM (HiFTChain) nor
    /// codec-LM (any RVQ / FSQ codec) but a continuous VAE decoder
    /// consuming flow-matching sampler output.
    ///
    /// Topology chains a MiniCPM-4 LM backbone (`hidden_size=1024`,
    /// `num_hidden_layers=24`, GQA `num_attention_heads=16` /
    /// `num_key_value_heads=2` for a very wide group ratio 8, SwiGLU
    /// `intermediate_size=4096`, RoPE `theta=10000` with longrope
    /// scaling / 32-entry `long_factor` and `short_factor` tables /
    /// `rms_norm_eps=1e-5` / `vocab_size=73448` /
    /// `max_position_embeddings=32_768`; MiniCPM-specific
    /// `scale_emb=12` / `dim_model_base=256` / `scale_depth=1.4`)
    /// through a 6-layer residual acoustic LM (same backbone family,
    /// `vocab_size=0`), a 4-layer local encoder (`hidden_dim=1024`,
    /// `ffn_dim=4096`, `num_heads=16`), a 4-layer local DiT (same axes),
    /// a UnifiedCFM flow-matching sampler (`sigma_min=1e-6`,
    /// `solver=euler`, `inference_cfg_rate=2.0`,
    /// `t_scheduler=log-norm` — the latter is training-side; inference
    /// walks a linear `t_span`), an AudioVAE V2 continuous encoder /
    /// decoder (`sample_rate=16_000`, `out_sample_rate=48_000`,
    /// `encoder_dim=128`, `encoder_rates=[2,5,8,8]` → 25 Hz continuous
    /// latents, `latent_dim=64`, `decoder_dim=2048`,
    /// `decoder_rates=[8,6,5,2,2,2]` → 48 kHz PCM out, `depthwise=true`),
    /// and an inline scalar-quantization bottleneck
    /// (`scalar_quantization_latent_dim=256`,
    /// `scalar_quantization_scale=9` — inside the LM hidden stream,
    /// distinct from the FSQ *codec* family).
    ///
    /// `patch_size=2` (LM slots two VAE frames per step), `feat_dim=64`
    /// (equals `vae.latent_dim` — the runtime enforces this handshake
    /// loudly per FR-EX-08), `max_length=4096`. Introduces the shared
    /// new SoTA plan Phase 4 primitive `vokra_ops::vae_continuous`
    /// (shared with the planned VibeVoice consumer); reuses the
    /// existing `vokra_ops::flow_sampler` for the CFM sampler. Every
    /// hparam transcribed verbatim from
    /// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` and
    /// `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
    /// (`AudioVAEConfig(BaseModel)` defaults).
    ///
    /// Distinct arch tag from CosyVoice2/3 / Qwen3-TTS / Chatterbox
    /// family / Dia / Zonos / CSM / Voxtral / Kyutai STT / Moshi —
    /// silently sharing would mis-route the runtime dispatch. The
    /// upstream release is BF16 (~1 GB); today's F32/F16 pass-through
    /// hits the `skipped_non_float` counter on BF16 tensors and the
    /// converter surfaces the loud "no float tensors" note. Convert
    /// with [`convert_voxcpm2_file`] — the converter takes no config
    /// side-car (every hparam is fixed for the 0.5B release and
    /// transcribed as compile-time constants).
    VoxCpm2,
    /// Microsoft **VibeVoice-1.5B** safetensors checkpoint (SoTA plan
    /// Phase 4, 2026-07-24). MIT end-to-end — code + weight under a
    /// single MIT grant (`huggingface.co/microsoft/VibeVoice-1.5B`
    /// model-card + `github.com/microsoft/VibeVoice/blob/main/LICENSE`,
    /// fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). SECOND
    /// consumer of the **continuous VAE + diffusion decoder** class
    /// (after VoxCPM-0.5B) — but where VoxCPM uses a UnifiedCFM
    /// flow-matching sampler, VibeVoice uses a **DDPM** sampler
    /// (v-prediction + cosine β schedule + 20 reduced-step inference on
    /// 1000 training steps). This axis introduces the SoTA plan Phase 4
    /// **new** primitive `vokra_ops::ddpm_sampler`; the acoustic VAE
    /// half shares the existing `vokra_ops::vae_continuous` primitive
    /// (introduced with VoxCPM and shared with this VibeVoice consumer
    /// per the vae_continuous rustdoc).
    ///
    /// Topology chains a **Qwen2 decoder LM** (`decoder_config`,
    /// Qwen2.5-1.5B flavour — `hidden_size=1536`,
    /// `num_hidden_layers=28`, MHA `num_attention_heads=12`, GQA
    /// `num_key_value_heads=2` (group ratio 6), SwiGLU
    /// `intermediate_size=8960`, RoPE `theta=1_000_000` no scaling,
    /// `rms_norm_eps=1e-6`, `vocab_size=151_936`,
    /// `max_position_embeddings=65_536`,
    /// `tie_word_embeddings=true`, `sliding_window=null`,
    /// `use_sliding_window=false`) through an **acoustic tokenizer**
    /// (`acoustic_tokenizer_config`, σ-VAE with mirror-symmetric
    /// encoder / decoder — `vae_dim=64`, `std_dist_type="gaussian"`
    /// with `fix_std=0.5`, `encoder_ratios=decoder_ratios=[8,5,5,4,2,2]`
    /// (product 3200 → **7.5 Hz** frame rate at 24 kHz input),
    /// `encoder_n_filters=decoder_n_filters=32`,
    /// `encoder_depths="3-3-3-3-3-3-8"`,
    /// `mixer_layer="depthwise_conv"`, `layernorm="RMSNorm"`,
    /// `layernorm_eps=1e-5`, `causal=true`, `channels=1`,
    /// `conv_bias=true`, `disable_last_norm=true`,
    /// `layer_scale_init_value=1e-6`), a **semantic tokenizer**
    /// (`semantic_tokenizer_config`, encoder-**only** variant of the
    /// same chain — `vae_dim=128`, `std_dist_type="none"` deterministic
    /// with `fix_std=0`, same `encoder_ratios=[8,5,5,4,2,2]`),
    /// and a **diffusion head** (`diffusion_head_config`, 4-layer
    /// AdaLN-modulated MLP with SwiGLU FFN — `hidden_size=1536`,
    /// `head_layers=4`, `head_ffn_ratio=3.0` (SwiGLU inner dim
    /// `int(1536·3)=4608`), `rms_norm_eps=1e-5`, `latent_size=64`,
    /// `speech_vae_dim=64`, `prediction_type="v_prediction"`,
    /// `diffusion_type="ddpm"`, `ddpm_num_steps=1000`,
    /// `ddpm_num_inference_steps=20`, `ddpm_beta_schedule="cosine"`,
    /// `ddpm_batch_mul=4`).
    ///
    /// Every hparam transcribed verbatim from
    /// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json`
    /// and `github.com/microsoft/VibeVoice/blob/main/vibevoice/
    /// modular/configuration_vibevoice.py` (fetched 2026-07-24 —
    /// CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// Distinct arch tag from VoxCPM / CosyVoice2/3 / Qwen3-TTS /
    /// Chatterbox family / Dia / Zonos / CSM / Voxtral / Kyutai STT /
    /// Moshi — silently sharing would misroute the runtime dispatch
    /// (VoxCPM → `flow_sample`, VibeVoice → `ddpm_sample`; the two
    /// samplers are irreconcilable, see the
    /// `vokra_ops::ddpm_sampler` rustdoc for the argument).
    ///
    /// The upstream release is BF16 (`torch_dtype = "bfloat16"`);
    /// today's F32/F16 pass-through hits the `skipped_non_float`
    /// counter on the BF16 tensors and the converter surfaces the
    /// loud "no float tensors" note. Convert with
    /// [`convert_vibevoice_file`] — the converter takes no config
    /// side-car (every hparam is fixed for the 1.5B release and
    /// transcribed as compile-time constants; a future 7B variant
    /// would demand `--config`).
    VibeVoice,
    /// Aratako **Irodori-TTS-500M-v3** safetensors checkpoint (SoTA plan
    /// Phase 5 JA-TTS-1, 2026-07-24). MIT end-to-end — code + weight
    /// under a single MIT LICENSE at `github.com/Aratako/Irodori-TTS/blob/main/LICENSE`
    /// (verified via `gh api /repos/Aratako/Irodori-TTS/license` →
    /// `MIT`, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    /// A **Rectified-Flow Diffusion Transformer (RF-DiT)** over the
    /// paired `Aratako/Semantic-DACVAE-Japanese-32dim` codec (32-d
    /// continuous latent → 48 kHz PCM). Topology chains a **prompt-text
    /// encoder** (Llama-family self-attention with RoPE + a sigmoid gate
    /// on the output projection, initialized from the LLM-JP-3 150M
    /// checkpoint — `text_vocab_size=99_574`, `text_dim=512`,
    /// `text_layers=10`, `text_heads=8` (`head_dim=64`),
    /// `text_mlp_ratio=2.6`, `text_add_bos=true`), a **reference-latent
    /// (speaker) encoder** (self-attention transformer over patched
    /// reference DACVAE latents — `speaker_dim=768`, `speaker_layers=8`,
    /// `speaker_heads=12` (`head_dim=64`), `speaker_mlp_ratio=2.6`,
    /// `speaker_patch_size=1`), a **RF-DiT body** (joint-attention DiT
    /// blocks with Low-Rank AdaLN modulation, SwiGLU FFN + RoPE —
    /// `latent_dim=32`, `latent_patch_size=1`, `model_dim=1280`,
    /// `num_layers=12`, `num_heads=20` (`head_dim=64`),
    /// `mlp_ratio=2.875`, `timestep_embed_dim=512`, `adaln_rank=192`,
    /// `norm_eps=1e-5`), and an **integrated duration predictor** (v3
    /// phase-2: `duration_aux_dim=14`, `duration_hidden_dim=1024`,
    /// `duration_layers=3`, `duration_attention_heads=8`,
    /// `duration_dropout=0.1`,
    /// `duration_architecture="token_sum_adarn_zero_no_aux"`,
    /// `duration_token_init_frames=9.0`,
    /// `duration_speaker_fusion="adarn_zero"`).
    ///
    /// Sampling integrates the rectified-flow ODE
    /// (`x_t = (1-t) x_0 + t z`, `v = z - x_0`) with an **Euler** step
    /// from `t=1` to `t=0` in 40 default steps under a `Schedule::Linear`
    /// or `Schedule::Sway` (F5-TTS toggle) — both directly supported by
    /// the existing `vokra_ops::flow_sampler` primitive (M3-05), with
    /// independent split-batch CFG on three axes (text / caption /
    /// speaker; per-axis scales 3.0 / 3.0 / 5.0; cfg window t ∈
    /// `[0.5, 1.0]`).
    ///
    /// Every hparam transcribed verbatim from
    /// `github.com/Aratako/Irodori-TTS` (`configs/train_500m_v3_phase1_body.yaml`
    /// plus `configs/train_500m_v3_phase2_duration.yaml` plus
    /// `irodori_tts/config.py::ModelConfig`, fetched 2026-07-24 —
    /// CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// Distinct arch tag from every Phase-4 continuous-VAE sibling
    /// (VoxCPM / VibeVoice) and every earlier sibling (CosyVoice2/3 /
    /// Qwen3-TTS / Chatterbox family / Dia / Zonos / CSM / Voxtral /
    /// Kyutai STT / Moshi) — silently sharing an arch tag would
    /// misroute the runtime dispatch (VibeVoice → `ddpm_sample`,
    /// VoxCPM → `flow_sample` with EpsS schedule, Irodori →
    /// `flow_sample` with Linear / Sway schedule and a distinct latent
    /// width of 32 vs the Phase-4 siblings' 64).
    ///
    /// Convert with [`convert_irodori_file`] — the converter takes no
    /// side-car config today (every hparam is fixed for the 500M-v3
    /// release and transcribed as compile-time constants; a future
    /// 600M VoiceDesign / 2.5B variant that reshapes the DiT or adds
    /// caption conditioning would demand `--config`).
    Irodori,
    /// ESPnet-family Japanese **plain VITS** safetensors checkpoint
    /// (SoTA plan Phase 5 JA-TTS-2, 2026-07-24). Architecture is
    /// Apache-2.0 (ESPnet `espnet2/gan_tts/vits/`) + MIT
    /// (`jaywalnut310/vits` reference).
    ///
    /// This is Kim et al. 2021 VITS (arXiv:2106.06103) — a text
    /// encoder, stochastic duration predictor, normalising flow, and
    /// **plain HiFi-GAN generator** (Kong et al. 2020,
    /// arXiv:2010.05646). The architecture is shared with piper-plus
    /// (MB-iSTFT-VITS2) up to the flow output, but the decoder is a
    /// HiFi-GAN generator directly (**no sub-band iSTFT, no PQMF**),
    /// which is why the two arch tags cannot alias. Every
    /// architectural axis is transcribed verbatim from
    /// `egs2/jsut/tts1/conf/tuning/train_vits.yaml`, from
    /// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`, and from
    /// `espnet2/gan_tts/vits/{vits,generator}.py` (fetched 2026-07-24
    /// — CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// **⚠️  Weight redistribution default is `RedistributionForbidden`**.
    /// The publicly distributed ESPnet-JSUT / ESPnet-JVS / COEIROINK
    /// checkpoints ride on corpus terms that forbid trained-weight
    /// redistribution (JSUT: *"Re-distribution is not permitted"*, JVS:
    /// same, COEIROINK: per-character licence terms a converter cannot
    /// machine-check). The provenance stamp therefore defaults to
    /// `RedistributionForbidden`; a user who trained on a permissive
    /// corpus overrides via `vokra-convert --license <spdx>`. Architecture
    /// is always independently implementable — the block runs (whisper.cpp
    /// 型 self re-imp, CLAUDE.md 設計判断 4). See
    /// `docs/tickets/sota-coverage-plan-2026-07-22.md` §2.4 for the
    /// "support the architecture, refuse the weights" rationale.
    ///
    /// Convert with [`convert_vits_ja_file`]. The converter takes no
    /// side-car config today (the JSUT 22 kHz single-speaker recipe
    /// axes are byte-parallel to the transcribed constants in
    /// `models::vits_ja`; a `--config` axis for the JVS multi-speaker
    /// / full-band 44 kHz / downstream re-training variants is a
    /// follow-up).
    VitsJa,
    /// ku-nlp **DeBERTa v2** Japanese-character BERT checkpoint (SBV2 v2
    /// plan Task 11, 2026-07-26): a Hugging Face `transformers`
    /// `deberta_v2` safetensors checkpoint for Japanese text
    /// (`ku-nlp/deberta-v2-large-japanese-char-wwm`, Apache-2.0
    /// model-card header). F32 / F16 / BF16 tensors pass through verbatim
    /// under upstream HF names; the runtime's `DebertaV2Encoder::from_gguf`
    /// will be written to map those names to the encoder's internal tensor
    /// access pattern (Task 30 — today the tensor-to-schema mapping is a
    /// deferred follow-up; every tensor is emitted verbatim so the mapping
    /// can be validated once a real checkpoint arrives). Every hparam
    /// required by the encoder is transcribed verbatim from the checkpoint's
    /// `config.json` and written to the `vokra.bert.deberta_v2.*` metadata
    /// chunk group. Convert with [`convert_deberta_v2_file`] with a
    /// safetensors checkpoint.
    DebertaV2,
    /// ku-nlp **DeBERTa v3** Japanese-character BERT checkpoint (SBV2 v2
    /// plan Task 11, 2026-07-26): a Hugging Face `transformers`
    /// `deberta_v3` safetensors checkpoint for Japanese text
    /// (`ku-nlp/deberta-v3-large-japanese-char-wwm`, Apache-2.0
    /// model-card header). F32 / F16 / BF16 tensors pass through verbatim
    /// under upstream HF names; the runtime's `DebertaV3Encoder::from_gguf`
    /// will be written to map those names to the encoder's internal tensor
    /// access pattern (Task 30 — today the tensor-to-schema mapping is a
    /// deferred follow-up; every tensor is emitted verbatim so the mapping
    /// can be validated once a real checkpoint arrives). Every hparam
    /// required by the encoder is transcribed verbatim from the checkpoint's
    /// `config.json` and written to the `vokra.bert.deberta_v3.*` metadata
    /// chunk group. Convert with [`convert_deberta_v3_file`] with a
    /// safetensors checkpoint.
    DebertaV3,
    /// Style-Bert-VITS2 v2 (SBV2) official checkpoint (SBV2 v2 plan Task 25,
    /// 2026-07-26): a `litagin02/style_bert_vits2`-family safetensors
    /// checkpoint for the multilingual (JA + EN) base model
    /// (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`). F32 / F16 /
    /// BF16 tensors pass through verbatim under upstream safetensors names;
    /// the runtime's `SbV2Model::from_gguf`
    /// (`crates/vokra-models/src/sbv2/mod.rs`, Task 24) will be written to
    /// map those names to the `sbv2.*` tensor hierarchy it reads (Task 30
    /// — today the tensor-to-schema mapping is a deferred follow-up; every
    /// tensor is emitted verbatim so the mapping can be validated once a
    /// real checkpoint arrives). Every one of the 22 required (+ 1
    /// optional) `vokra.sbv2.*` hparam keys is written **only** when a JSON
    /// config side-car is supplied — see [`convert_sbv2_file`]'s doc.
    /// **Weight license defaults to `agpl-3.0`** (→
    /// [`LicenseClass::Copyleft`](vokra_core::LicenseClass::Copyleft) —
    /// redistribution is permitted with the original licence preserved,
    /// never relabelled; design doc §9). Convert with [`convert_sbv2_file`].
    SbV2,
}

impl ModelKind {
    /// Parses the `--model` argument value.
    ///
    /// `whisper` is the canonical spelling (size is auto-detected from the
    /// checkpoint shapes); `whisper-base` is kept as a backward-compatible
    /// alias for pre-M2-06 invocations — both dispatch to the same
    /// size-detecting path (M2-06-T06).
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            // Canonical M2-06+ spelling: size auto-detected from checkpoint.
            "whisper" => Some(Self::Whisper),
            // Backward-compatible alias for pre-M2-06 invocations.
            "whisper-base" => Some(Self::Whisper),
            "silero-vad" => Some(Self::SileroVad),
            "utmos" => Some(Self::Utmos),
            "piper-plus" => Some(Self::PiperPlus),
            "campplus" => Some(Self::CamPlus),
            "kokoro" => Some(Self::Kokoro),
            "cosyvoice2" => Some(Self::CosyVoice2),
            "cosyvoice3"
            | "cosyvoice-3"
            | "fun-cosyvoice3"
            | "fun-cosyvoice-3"
            | "fun-cosyvoice3-0.5b"
            | "fun-cosyvoice3-0.5b-2512"
            | "fun-cosyvoice3-0_5b"
            | "fun-cosyvoice3-0_5b-2512" => Some(Self::CosyVoice3),
            "voxtral" => Some(Self::Voxtral),
            "mimi" => Some(Self::Mimi),
            "dac" => Some(Self::Dac),
            "csm" => Some(Self::Csm),
            "moshi" => Some(Self::Moshi),
            "denoise" => Some(Self::Denoise),
            "dia" | "dia-1.6b" | "dia-1_6b" => Some(Self::Dia),
            "zonos" | "zonos-v0.1" | "zonos-v0_1" | "zonos-v0.1-transformer" => Some(Self::Zonos),
            "kyutai-stt" | "kyutai-stt-2.6b-en" | "kyutai-stt-2.6b" | "stt-2.6b-en" => {
                Some(Self::KyutaiStt)
            }
            "parakeet"
            | "parakeet-tdt"
            | "parakeet-tdt-0.6b-v3"
            | "parakeet-tdt-0.6b"
            | "parakeet-tdt-0_6b-v3"
            | "parakeet-tdt-0_6b" => Some(Self::Parakeet),
            "parakeet-ctc" | "parakeet-ctc-1.1b" | "parakeet-ctc-1.1B" | "parakeet-ctc-1_1b" => {
                Some(Self::ParakeetCtc)
            }
            "canary" | "canary-1b-v2" | "canary-1b-v2-en" | "canary-1b_v2" => Some(Self::Canary),
            "omniasr-ctc" | "omniasr-ctc-1b" | "omniasr-ctc-1_1b" | "omniasr_ctc"
            | "omniasr_ctc_1b" => Some(Self::OmniasrCtc),
            "distil-whisper"
            | "distil_whisper"
            | "distil-whisper-large-v3"
            | "distil-whisper-large-v3.5"
            | "distil-whisper-large-v3_5"
            | "distil-large-v3"
            | "distil-large-v3.5"
            | "distil-large-v3_5" => Some(Self::DistilWhisper),
            // Kotoba Technologies kotoba-whisper family (SoTA plan
            // Phase 5 JA-ASR-2, 2026-07-24). Accept the canonical arch
            // spelling, the underscore variant, and every currently-
            // shipped HF release id (v1.0 / v1.1 / v2.0 / v2.1 /
            // bilingual-v1.0). All spellings resolve to the same
            // apache-2.0 Japanese-distilled checkpoint family; only
            // the distilled weights and training corpus differ, which
            // the shape-driven converter cannot distinguish.
            "kotoba-whisper"
            | "kotoba_whisper"
            | "kotoba-whisper-v1.0"
            | "kotoba-whisper-v1_0"
            | "kotoba-whisper-v1.1"
            | "kotoba-whisper-v1_1"
            | "kotoba-whisper-v2.0"
            | "kotoba-whisper-v2_0"
            | "kotoba-whisper-v2.1"
            | "kotoba-whisper-v2_1"
            | "kotoba-whisper-bilingual"
            | "kotoba-whisper-bilingual-v1.0"
            | "kotoba-whisper-bilingual-v1_0" => Some(Self::KotobaWhisper),
            // Resemble AI Chatterbox family — the multilingual variant is
            // the canonical Phase 3 landing. Accept the family, the two HF
            // variant tags, and the raw `t3_mtl23ls_v{2,3}` checkpoint
            // filenames.
            "chatterbox"
            | "chatterbox-multilingual"
            | "chatterbox-multilingual-v2"
            | "chatterbox-multilingual-v3"
            | "chatterbox-mtl23ls-v2"
            | "chatterbox-mtl23ls-v3"
            | "chatterbox-english"
            | "chatterbox_en" => Some(Self::Chatterbox),
            // Resemble AI Chatterbox-Turbo family — 350M distilled Turbo
            // variant. Accept the canonical release id, the underscore
            // spelling used by the arch tag, the v1 stem
            // (`t3_turbo_v1.safetensors`), and the sibling ONNX release id
            // (which still routes here because the runtime never loads
            // the ONNX graph — the safetensors path is the only real
            // conversion input, FR-LD-05).
            "chatterbox-turbo"
            | "chatterbox_turbo"
            | "chatterbox-turbo-v1"
            | "chatterbox-turbo-onnx" => Some(Self::ChatterboxTurbo),
            // Resemble AI Chatterbox-Nano family — compact 110M variant.
            // Accept the canonical HF release id, the underscore spelling
            // (== the arch tag), and the v1 stem
            // (`t3_nano_v1.safetensors`). Chatterbox-Nano does not ship
            // an ONNX sibling release, so no `-onnx` alias here.
            "chatterbox-nano" | "chatterbox_nano" | "chatterbox-nano-v1" => {
                Some(Self::ChatterboxNano)
            }
            // Alibaba Qwen3-TTS family — canonical HF release + `qwen3_tts`
            // arch-tag underscore spelling + common short forms. All spellings
            // resolve to the same 0.6B-Base checkpoint today; a future variant
            // (0.6B-CustomVoice / 0.6B-VoiceDesign / 1.7B) would be a distinct
            // `ModelKind` when it lands.
            "qwen3-tts"
            | "qwen3_tts"
            | "qwen3-tts-0.6b"
            | "qwen3-tts-0_6b"
            | "qwen3-tts-12hz-0.6b-base"
            | "qwen3-tts-12hz-0_6b-base"
            | "qwen3-tts-12hz-0.6b" => Some(Self::Qwen3Tts),
            // OpenBMB VoxCPM family — canonical HF release + arch-tag
            // underscore spelling + common short forms. All spellings
            // resolve to the same 0.5B release today; a future variant
            // (0.5B-CustomVoice / 1.5B) would be a distinct `ModelKind`.
            "voxcpm" | "voxcpm2" | "voxcpm-0.5b" | "voxcpm-0_5b" | "voxcpm-0.5b-base"
            | "voxcpm-0_5b-base" => Some(Self::VoxCpm2),
            // Microsoft VibeVoice family — canonical HF release + arch-tag
            // spelling + common short forms. Every spelling resolves to the
            // 1.5B release today; a future 7B variant would be a distinct
            // `ModelKind` when it lands (its config axes reshape the Qwen2
            // backbone).
            "vibevoice"
            | "vibevoice-1.5b"
            | "vibevoice-1_5b"
            | "vibevoice-1.5b-base"
            | "vibevoice-1_5b-base" => Some(Self::VibeVoice),
            // Aratako Irodori-TTS family (SoTA plan Phase 5 JA-TTS-1,
            // 2026-07-24). Accept the canonical arch spelling, the
            // underscore variant, and every currently-shipped HF
            // release id (500M / 500M-v2 / 500M-v2-VoiceDesign /
            // 500M-v3 / 600M-v3-VoiceDesign). All spellings resolve to
            // the same 500M-v3-shape converter today; the VoiceDesign
            // 3-branch variant reshapes the DiT (adds a caption
            // encoder) and would be a distinct `ModelKind` when its
            // config axes land as first-class constants (a future
            // `IrodoriVoiceDesign` variant).
            "irodori"
            | "irodori-tts"
            | "irodori_tts"
            | "irodori-tts-500m"
            | "irodori-tts-500m-v2"
            | "irodori-tts-500m-v2-voicedesign"
            | "irodori-tts-500m-v3"
            | "irodori-tts-500m-v3-base"
            | "irodori-tts-600m-v3-voicedesign" => Some(Self::Irodori),
            // ESPnet-family Japanese plain VITS (SoTA plan Phase 5
            // JA-TTS-2, 2026-07-24). Accept the canonical arch tag +
            // the underscore variant + the three upstream deployment
            // ids (ESPnet-JSUT / ESPnet-JVS / COEIROINK). All
            // spellings resolve to the same JSUT 22 kHz single-speaker
            // converter path today; the JVS multi-speaker + full-band
            // 44 kHz + downstream re-training variants share the same
            // tensor topology and are follow-up `--config` axes.
            "vits-ja" | "vits_ja" | "vits-jp" | "vits_jp" | "espnet-vits-ja" | "espnet-vits-jp"
            | "espnet-jsut-vits" | "espnet-jvs-vits" | "coeiroink-vits" => Some(Self::VitsJa),
            // ku-nlp DeBERTa family (SBV2 v2 plan Task 11, 2026-07-26).
            // Accept the canonical HF release id + underscore / hyphen
            // variants. All spellings resolve to the same Japanese-character
            // converter today; a future full-width hiragana / kanji
            // normalization variant would be distinct.
            "deberta-v2" | "deberta_v2" | "ku-nlp/deberta-v2-large-japanese-char-wwm" => {
                Some(Self::DebertaV2)
            }
            "deberta-v3" | "deberta_v3" | "ku-nlp/deberta-v3-large-japanese-char-wwm" => {
                Some(Self::DebertaV3)
            }
            // Style-Bert-VITS2 v2 (SBV2 v2 plan Task 25, 2026-07-26). Accept
            // the canonical arch spelling, the design doc's SKU id, and the
            // common project-name spellings (with/without hyphen, with/
            // without an explicit "v2"). All spellings resolve to the same
            // multilingual base converter path today.
            "sbv2"
            | "sbv2-v2"
            | "sbv2-v2-multilingual-base"
            | "style-bert-vits2"
            | "style_bert_vits2"
            | "style-bert-vits2-v2" => Some(Self::SbV2),
            _ => None,
        }
    }

    /// The canonical `--model` argument value for this kind.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::SileroVad => "silero-vad",
            Self::Utmos => "utmos",
            Self::PiperPlus => "piper-plus",
            Self::CamPlus => "campplus",
            Self::Kokoro => "kokoro",
            Self::CosyVoice2 => "cosyvoice2",
            Self::CosyVoice3 => "cosyvoice3",
            Self::Voxtral => "voxtral",
            Self::Mimi => "mimi",
            Self::Dac => "dac",
            Self::Csm => "csm",
            Self::Moshi => "moshi",
            Self::Denoise => "denoise",
            Self::Dia => "dia",
            Self::Zonos => "zonos",
            Self::KyutaiStt => "kyutai-stt",
            Self::Parakeet => "parakeet-tdt",
            Self::ParakeetCtc => "parakeet-ctc",
            Self::Canary => "canary",
            Self::OmniasrCtc => "omniasr-ctc",
            Self::DistilWhisper => "distil-whisper",
            Self::KotobaWhisper => "kotoba-whisper",
            Self::Chatterbox => "chatterbox",
            Self::ChatterboxTurbo => "chatterbox-turbo",
            Self::ChatterboxNano => "chatterbox-nano",
            Self::Qwen3Tts => "qwen3-tts",
            Self::VoxCpm2 => "voxcpm",
            Self::VibeVoice => "vibevoice",
            Self::Irodori => "irodori",
            Self::VitsJa => "vits-ja",
            Self::DebertaV2 => "deberta-v2",
            Self::DebertaV3 => "deberta-v3",
            Self::SbV2 => "sbv2",
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_arg())
    }
}

/// Summary of a successful conversion.
#[derive(Debug)]
pub struct ConvertSummary {
    /// The model that was converted.
    pub model: ModelKind,
    /// Number of tensors written to the GGUF.
    pub tensor_count: usize,
    /// Number of metadata entries written (including `general.alignment` if
    /// injected).
    pub metadata_count: usize,
    /// Size of the output GGUF in bytes.
    pub output_bytes: u64,
    /// Human-readable notes (e.g. skipped non-float initializers).
    pub notes: Vec<String>,
}

/// Errors that can occur while converting a checkpoint.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConvertError {
    /// Reading the input or writing the output failed.
    Io(std::io::Error),
    /// The input checkpoint could not be parsed (safetensors / JSON / ONNX).
    Parse(String),
    /// The GGUF could not be assembled (from `vokra-core`'s writer).
    Gguf(String),
    /// A command-line / usage problem.
    Usage(String),
    /// A [`QuantPolicy`](models::whisper) rule resolved to a K-quant target for
    /// a tensor that cannot be K-quantized (rank < 2 or element count not a
    /// whole number of `QK_K` super-blocks). Emitted instead of silently
    /// widening the tensor's dtype (FR-EX-08, M2-08 T06).
    QuantPolicyInapplicable {
        /// The offending tensor's upstream name.
        tensor: String,
        /// The scheme alias the policy resolved to (e.g. `"w4a16-q4k"`).
        scheme: &'static str,
        /// Human-readable reason (rank, element count, etc.).
        reason: String,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(m) => write!(f, "parse error: {m}"),
            Self::Gguf(m) => write!(f, "GGUF write error: {m}"),
            Self::Usage(m) => write!(f, "usage error: {m}"),
            Self::QuantPolicyInapplicable {
                tensor,
                scheme,
                reason,
            } => write!(
                f,
                "quant policy inapplicable for tensor `{tensor}` (scheme `{scheme}`): {reason}"
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<safetensors::SafetensorsError> for ConvertError {
    fn from(e: safetensors::SafetensorsError) -> Self {
        Self::Parse(e.to_string())
    }
}

impl From<onnx::OnnxError> for ConvertError {
    fn from(e: onnx::OnnxError) -> Self {
        Self::Parse(e.to_string())
    }
}

impl From<vokra_core::gguf::GgufError> for ConvertError {
    fn from(e: vokra_core::gguf::GgufError) -> Self {
        Self::Gguf(e.to_string())
    }
}

impl From<QuantizeError> for ConvertError {
    fn from(e: QuantizeError) -> Self {
        Self::Gguf(e.to_string())
    }
}

/// Converts `input` into a GGUF written to `output`, returning a summary.
///
/// This is the single entry point used by both the `vokra-convert` binary and
/// the integration tests.
pub fn convert_file(
    model: ModelKind,
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file_licensed(model, input, output, None)
}

/// [`convert_file`] with an explicit weight-licence override.
///
/// Each converter stamps the licence it knows for its model. That is right when
/// the model has one canonical licence, but wrong when the *actual distribution
/// source* declares a different one — e.g. OpenAI's Whisper is MIT on GitHub,
/// yet the Hugging Face weight repos this checkpoint may have come from tag
/// `base`/`small`/`medium` as `apache-2.0`. Publishing must state the licence
/// of the artifact being redistributed, so when the two disagree the caller
/// passes the source's SPDX id here and it overrides the stamped
/// `vokra.provenance.{weight_license,license}` — keeping the GGUF the single
/// source of truth the model card is generated from (no card/artifact drift).
///
/// `license` is the raw SPDX string (e.g. `"apache-2.0"`); the class is
/// re-derived from it. `None` keeps the converter's built-in stamp.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_file_licensed(
    model: ModelKind,
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    // Moshi streams tensor-by-tensor (the 14 GiB full-7B checkpoint must
    // never be materialized whole — bounded-memory contract); it routes
    // through `convert_moshi_file` BEFORE the whole-file read below.
    if matches!(model, ModelKind::Moshi) {
        return convert_moshi_file(input, None, output);
    }
    let bytes = std::fs::read(input)?;

    let (mut builder, notes) = match model {
        ModelKind::Whisper => (models::whisper::convert(bytes, None)?, Vec::new()),
        ModelKind::SileroVad => {
            let (builder, report) = models::silero::convert(bytes)?;
            let notes = vec![format!(
                "silero: {} float weights written (both rates, sr8k.*/sr16k.*), {} non-float constants skipped, {} op-scope float strays skipped",
                report.written, report.skipped_non_float, report.skipped_stray
            )];
            (builder, notes)
        }
        ModelKind::PiperPlus => {
            return Err(ConvertError::Usage(
                "piper-plus needs a --config config.json; use convert_piper_plus_file".to_owned(),
            ));
        }
        ModelKind::Utmos => {
            return Err(ConvertError::Usage(
                "utmos needs a --config config.json (emitted by                  tools/parity/utmos_prepare_checkpoint.py alongside the flattened safetensors);                  use convert_utmos_file"
                    .to_owned(),
            ));
        }
        ModelKind::CamPlus => {
            let (builder, report) = models::campplus::convert(&bytes)?;
            let notes = vec![format!(
                "campplus: {} weights written ({} onnx:: names recovered, {} affine-free BN params synthesized, {} unmapped, {} non-float skipped), block_config {:?}",
                report.written,
                report.renamed,
                report.synthesized,
                report.unmapped,
                report.skipped_non_float,
                report.block_config
            )];
            (builder, notes)
        }
        ModelKind::Kokoro => {
            let (builder, report) = models::kokoro::convert(bytes)?;
            // Backward-compat: the placeholder path (no --config) emits the
            // same 3-field summary M2-07 T06 shipped, plus any diagnostic
            // notes the model routine surfaced. When the caller has a
            // `config.json`, use `convert_kokoro_file` instead — it enriches
            // the summary with the phoneme-symbol count.
            let mut notes = vec![format!(
                "kokoro: {} float weights written, {} non-float skipped, style_dim {}, {} voices",
                report.written,
                report.skipped_non_float,
                report.style_dim,
                report.voices.len(),
            )];
            notes.extend(report.notes.iter().map(|n| format!("kokoro warning: {n}")));
            (builder, notes)
        }
        ModelKind::CosyVoice2 => {
            // Shape-derived hparams only; the attention head split needs
            // the upstream config.json — use `convert_cosyvoice2_file`
            // with a `--config` for the full hparam chunk.
            let (builder, report) = models::cosyvoice2::convert(bytes)?;
            (builder, cosyvoice2_notes(&report))
        }
        ModelKind::CosyVoice3 => {
            // SoTA plan Phase 3: same shape-driven walk as CosyVoice2, but
            // the emitted GGUF carries the CosyVoice3 arch label + hparam
            // prefix so the runtime dispatches to `vokra-models::cosyvoice3`.
            // Same `--config` requirement — use `convert_cosyvoice3_file`
            // for the full hparam chunk (Qwen2 head split + rope / eps /
            // n_ctx aren't shape-derivable).
            let (builder, report) = models::cosyvoice3::convert_with_config(bytes, None)?;
            (builder, cosyvoice3_notes(&report))
        }
        ModelKind::Voxtral => {
            // Foundation path (M3-10): shape-only conversion writes `0`
            // sentinels for the RoPE / RMSNorm / GQA / vocab side-car values
            // the runtime cannot recover from tensor shapes alone. Real
            // conversions call `convert_voxtral_file` with a `VoxtralConfig`.
            let (builder, report) = models::voxtral::convert(bytes, None)?;
            let notes = vec![format!(
                "voxtral: {} float weights written, {} non-float skipped, name {}, tokenizer embedded: {} (shape-only path — pass a --config for the full hparam chunk)",
                report.written, report.skipped_non_float, report.name, report.tokenizer_embedded
            )];
            (builder, notes)
        }
        ModelKind::Mimi => {
            let (builder, report) = models::mimi::convert(bytes)?;
            let notes = vec![format!(
                "mimi: {} tensors passed through ({} non-float skipped), derived effective \
                 codebook tables [{} x {} x {}] emitted as `{}`, neural-chain adapter wrote \
                 {} structural mimi.enc.*/mimi.dec.* tensors + the vokra.mimi.* config chunk \
                 group ({})",
                report.written,
                report.skipped_non_float,
                report.n_codebooks,
                report.codebook_size,
                report.d_model,
                models::mimi::DERIVED_TABLES_TENSOR,
                report.structural_written,
                if report.structural_written > 0 {
                    "PCM encode/decode bindable"
                } else {
                    "checkpoint carries no SEANet chain — quantizer-only"
                },
            )];
            (builder, notes)
        }
        ModelKind::Dac => {
            return Err(ConvertError::Usage(
                "dac needs a --config side-car (from tools/parity/dac_prepare_checkpoint.py); \
                 use convert_dac_file"
                    .to_owned(),
            ));
        }
        ModelKind::Moshi => {
            // Handled by the streaming early-return above (bounded memory);
            // reaching this arm would mean the whole checkpoint was read.
            unreachable!("ModelKind::Moshi routes through convert_moshi_file")
        }
        ModelKind::Csm => {
            // Tokenizer-less path (M4-05-T03/T04): every float tensor
            // verbatim + the vokra.csm.* / vokra.mimi.* chunk groups. The
            // Llama-3.2 tokenizer blob (gated repo — T29) travels through
            // `convert_csm_file`.
            let (builder, report) = models::csm::convert(bytes, None)?;
            let mut notes = vec![format!(
                "csm: {} float weights written, {} non-float skipped, tokenizer \
                 embedded: {} (vocab axes are `0`-placeholders pending the T29 \
                 checkpoint; the runtime rejects the load until then)",
                report.written, report.skipped_non_float, report.tokenizer_embedded
            )];
            notes.extend(report.notes.iter().map(|n| format!("csm warning: {n}")));
            (builder, notes)
        }
        ModelKind::Denoise => {
            // M4-20 T17: prepared DFN3 safetensors → verbatim upstream-named
            // tensors + the `vokra.denoise.*` chunk. The routine hard-errors
            // on any missing / mis-shaped / unknown tensor and re-binds its
            // own output through the runtime loader before returning.
            let (builder, written) = models::denoise::convert_builder(bytes)
                .map_err(|e| ConvertError::Parse(e.to_string()))?;
            let notes = vec![format!(
                "denoise: {written} DeepFilterNet3 tensors written verbatim (dead \
                 checkpoint tensors skipped by policy: erb_fb, df_dec.df_fc_a.*), \
                 loadability re-checked via DenoiseModel::from_gguf"
            )];
            (builder, notes)
        }
        ModelKind::Dia => {
            // SoTA plan Phase 1-4: pass every F32/F16 tensor through verbatim
            // and stamp the `vokra.dia.*` chunk group from the primary-source
            // constants transcribed in `models::dia`.
            let (builder, report) = models::dia::convert(bytes)?;
            let mut notes = vec![format!(
                "dia: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("dia warning: {n}")));
            (builder, notes)
        }
        ModelKind::Zonos => {
            // SoTA plan Phase 1-5: pass every F32/F16 tensor through verbatim
            // and stamp the `vokra.zonos.*` chunk group (backbone hparams +
            // vocab + delay pattern + 7 typed prefix-conditioner descriptors)
            // from the primary-source constants transcribed in `models::zonos`.
            let (builder, report) = models::zonos::convert(bytes)?;
            let mut notes = vec![format!(
                "zonos: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("zonos warning: {n}")));
            (builder, notes)
        }
        ModelKind::KyutaiStt => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through verbatim
            // and stamp the `vokra.kyutai_stt.*` chunk group (backbone +
            // depformer + audio + text + streaming + delays) from the
            // primary-source constants transcribed in `models::kyutai_stt`.
            // Provenance = CC-BY 4.0 (AttributionRequired) + FR-MD-09
            // attribution text.
            let (builder, report) = models::kyutai_stt::convert(bytes)?;
            let mut notes = vec![format!(
                "kyutai-stt: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("kyutai-stt warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Parakeet => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through
            // verbatim and stamp the `vokra.parakeet.*` chunk group
            // (encoder / decoder / joint / duration bins) from the
            // primary-source constants transcribed in `models::parakeet`.
            // Provenance = CC-BY 4.0 (AttributionRequired) + FR-MD-09
            // attribution text.
            let (builder, report) = models::parakeet::convert(bytes)?;
            let mut notes = vec![format!(
                "parakeet-tdt: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("parakeet-tdt warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::ParakeetCtc => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through
            // verbatim and stamp the `vokra.parakeet_ctc.*` chunk group
            // (encoder + CTC head — no decoder / joint / duration bins,
            // since CTC has no RNN-T prediction network) from the
            // primary-source constants transcribed in
            // `models::parakeet_ctc`. Provenance = CC-BY 4.0
            // (AttributionRequired) + FR-MD-09 attribution text.
            let (builder, report) = models::parakeet_ctc::convert(bytes)?;
            let mut notes = vec![format!(
                "parakeet-ctc: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("parakeet-ctc warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Canary => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through
            // verbatim and stamp the `vokra.canary.*` chunk group
            // (FastConformer encoder + Transformer AED decoder + head)
            // from the primary-source constants transcribed in
            // `models::canary`. Provenance = CC-BY 4.0
            // (AttributionRequired) + FR-MD-09 attribution text.
            let (builder, report) = models::canary::convert(bytes)?;
            let mut notes = vec![format!(
                "canary: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("canary warning: {n}")));
            (builder, notes)
        }
        ModelKind::OmniasrCtc => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through
            // verbatim and stamp the `vokra.omniasr_ctc.*` chunk group
            // (wav2vec 2.0 encoder + CTC head — no decoder or joint
            // section, since CTC has no RNN-T prediction network) from
            // the primary-source constants transcribed in
            // `models::omniasr_ctc`. Provenance = Apache-2.0
            // (Permissive) — no runtime-side attribution obligation,
            // unlike NVIDIA's CC-BY 4.0 Parakeet-CTC / Canary.
            let (builder, report) = models::omniasr_ctc::convert(bytes)?;
            let mut notes = vec![format!(
                "omniasr-ctc: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("omniasr-ctc warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::DistilWhisper => {
            // SoTA plan Phase 2: pass every F32/F16 tensor through
            // verbatim and stamp the `vokra.whisper.*` chunk group
            // (schema shared with vanilla Whisper — distil-whisper
            // differs only in `n_text_layer`, not in the schema) from
            // the checkpoint's tensor shapes. Provenance = MIT
            // (Permissive) — no runtime-side attribution obligation.
            // The arch stamp `vokra.model.arch = "distil-whisper"`
            // is distinct from vanilla Whisper's `"whisper"` so the
            // runtime can label the loaded model correctly.
            let (builder, report) = models::distil_whisper::convert(bytes)?;
            let mut notes = vec![format!(
                "distil-whisper: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("distil-whisper warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::KotobaWhisper => {
            // SoTA plan Phase 5 JA-ASR-2 (2026-07-24): pass every
            // F32/F16 tensor through verbatim and stamp the
            // `vokra.whisper.*` chunk group (schema shared with
            // vanilla Whisper — kotoba-whisper differs only in
            // `n_text_layer`, not in the schema) from the checkpoint's
            // tensor shapes. Provenance = **Apache-2.0** (Permissive)
            // — no runtime-side attribution obligation, distinct from
            // distil-whisper's MIT stamp. The arch stamp
            // `vokra.model.arch = "kotoba-whisper"` is distinct from
            // vanilla Whisper's `"whisper"` and distil-whisper's
            // `"distil-whisper"` so the runtime can label the loaded
            // model correctly. **JA-ASR-2 axis**: `n_text_layer` is
            // read from the checkpoint's `model.decoder.layers.*`
            // prefix count via `count_layers`, never hard-coded to
            // 32 — the runtime's shared `WhisperConfig::from_gguf`
            // (data-driven since M0) honors whatever value lands
            // here.
            let (builder, report) = models::kotoba_whisper::convert(bytes)?;
            let mut notes = vec![format!(
                "kotoba-whisper: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("kotoba-whisper warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Chatterbox => {
            // SoTA plan Phase 3 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.chatterbox.*` chunk
            // group (T3 axes + Llama_520M backbone + norm/RoPE) from the
            // transcribed constants in `models::chatterbox`. Provenance =
            // MIT (Permissive — no runtime-side attribution obligation).
            // The default dispatch path tags the GGUF as the multilingual
            // variant (`t3_mtl23ls_v3.safetensors`); the English-only path
            // is reachable through the variant-taking internal converter.
            let (builder, report) = models::chatterbox::convert(bytes)?;
            let mut notes = vec![format!(
                "chatterbox: {} float weights written verbatim, {} non-float skipped, \
                 variant {:?}",
                report.written, report.skipped_non_float, report.variant,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("chatterbox warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::ChatterboxTurbo => {
            // SoTA plan Phase 3 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.chatterbox_turbo.*`
            // chunk group (GPT-2-medium backbone axes + STFT frontend +
            // sentinel tokens + paralinguistic tag count) from the
            // transcribed constants in `models::chatterbox_turbo`. The
            // arch tag is intentionally distinct from base Chatterbox
            // because Turbo swaps backbone family + sample rate + text
            // vocabulary — silently sharing the base arch tag would
            // misrepresent the loaded model. Provenance = MIT
            // (Permissive — no runtime-side attribution obligation; the
            // whole Chatterbox family ships under a single MIT LICENSE).
            let (builder, report) = models::chatterbox_turbo::convert(bytes)?;
            let mut notes = vec![format!(
                "chatterbox-turbo: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("chatterbox-turbo warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::ChatterboxNano => {
            // SoTA plan Phase 3 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.chatterbox_nano.*`
            // chunk group (Llama_520M backbone axes + STFT frontend +
            // GPT-2 sentinel tokens + paralinguistic tag count) from
            // the transcribed constants in `models::chatterbox_nano`.
            // The arch tag is intentionally distinct from both base
            // Chatterbox and Turbo because Nano keeps base's Llama_520M
            // backbone but swaps sample rate + text vocab + stop-text
            // sentinel — silently sharing either sibling's arch tag
            // would misrepresent the loaded model. Provenance = MIT
            // (Permissive — no runtime-side attribution obligation; the
            // whole Chatterbox family ships under a single MIT LICENSE).
            let (builder, report) = models::chatterbox_nano::convert(bytes)?;
            let mut notes = vec![format!(
                "chatterbox-nano: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("chatterbox-nano warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Qwen3Tts => {
            // SoTA plan Phase 3 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.qwen3_tts.*` chunk
            // group (talker + code-predictor Qwen3 axes + codec
            // handshake) from the transcribed constants in
            // `models::qwen3_tts`. The arch tag is intentionally distinct
            // from the Qwen-family siblings CosyVoice2/3 because
            // Qwen3-TTS is codec-LM (terminal step = qwen3_tts_codec),
            // NOT vocoder-LM (HiFTChain) — silently sharing either
            // sibling's arch tag would mis-route the runtime dispatch.
            // Provenance = apache-2.0 end-to-end (Permissive — no
            // runtime-side attribution obligation; LM + codec +
            // tokenizer + speaker encoder all under a single apache-2.0
            // grant).
            let (builder, report) = models::qwen3_tts::convert(bytes)?;
            let mut notes = vec![format!(
                "qwen3-tts: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("qwen3-tts warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::VoxCpm2 => {
            // SoTA plan Phase 4 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.voxcpm2.*` +
            // `vokra.vae_continuous.*` chunk groups (MiniCPM-4 LM
            // backbone + 6-layer residual acoustic LM + 4-layer local
            // encoder + 4-layer local DiT + UnifiedCFM sampler +
            // AudioVAE V2 continuous encoder / decoder + inline
            // scalar-quantization bottleneck) from the transcribed
            // constants in `models::voxcpm2`. NEW CLASS of TTS vs every
            // sibling: the terminal decoding hop is a continuous VAE
            // decoder consuming flow-matching sampler output (not
            // vocoder-LM HiFTChain, not codec-LM RVQ / FSQ) — silently
            // sharing an arch tag would mis-route the runtime dispatch.
            // Provenance = apache-2.0 end-to-end (Permissive — no
            // runtime-side attribution obligation; code + weight all
            // under a single apache-2.0 grant).
            let (builder, report) = models::voxcpm2::convert(bytes)?;
            let mut notes = vec![format!(
                "voxcpm2: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("voxcpm2 warning: {n}")));
            (builder, notes)
        }
        ModelKind::VibeVoice => {
            // SoTA plan Phase 4 (2026-07-24): pass every F32/F16 tensor
            // through verbatim and stamp the `vokra.vibevoice.*` chunk
            // group (Qwen2 decoder LM + acoustic σ-VAE tokenizer +
            // semantic encoder-only deterministic tokenizer + 4-layer
            // AdaLN-modulated diffusion head with DDPM v-prediction +
            // cosine β schedule) from the transcribed constants in
            // `models::vibevoice`. SECOND consumer of the continuous
            // VAE + diffusion decoder class (after VoxCPM); the sampler
            // differs (VibeVoice → `ddpm_sample`, VoxCPM →
            // `flow_sample`) so silently sharing an arch tag would
            // mis-route the runtime dispatch. Provenance = **MIT**
            // end-to-end (Permissive — no runtime-side attribution
            // obligation; code + weight all under a single MIT grant,
            // huggingface.co/microsoft/VibeVoice-1.5B model card + the
            // repo's LICENSE).
            let (builder, report) = models::vibevoice::convert(bytes)?;
            let mut notes = vec![format!(
                "vibevoice: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("vibevoice warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Irodori => {
            // SoTA plan Phase 5 JA-TTS-1 (2026-07-24): pass every F32/F16
            // tensor through verbatim and stamp the `vokra.irodori.*`
            // chunk group (RF-DiT body + LLM-JP-3 prompt-text encoder +
            // reference-latent speaker encoder + v3 phase-2 duration
            // predictor) from the transcribed constants in
            // `models::irodori`. THIRD consumer of the continuous-latent
            // + DiT class (after VoxCPM + VibeVoice); the sampler is
            // Rectified-Flow Euler with a Linear or Sway schedule (F5-TTS
            // toggle), NOT VibeVoice's DDPM and NOT VoxCPM's EpsS-
            // schedule flow-matching — silently sharing an arch tag
            // would misroute the runtime dispatch. Provenance = **MIT**
            // end-to-end (Permissive — no runtime-side attribution
            // obligation; code + weight all under a single MIT LICENSE
            // at `github.com/Aratako/Irodori-TTS`, verified via
            // `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`).
            let (builder, report) = models::irodori::convert(bytes)?;
            let mut notes = vec![format!(
                "irodori: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("irodori warning: {n}")));
            (builder, notes)
        }
        ModelKind::VitsJa => {
            // SoTA plan Phase 5 JA-TTS-2 (2026-07-24): pass every F32/F16
            // tensor through verbatim and stamp the `vokra.vits_ja.*`
            // chunk group (Kim et al. 2021 VITS text encoder / SDP /
            // residual affine coupling flow / plain HiFi-GAN generator)
            // from the transcribed constants in `models::vits_ja`.
            // Distinct arch tag from piper-plus (MB-iSTFT-VITS2) —
            // silently sharing would misroute the runtime dispatch (the
            // piper-plus decoder consumes a different tensor topology:
            // sub-band iSTFT + PQMF). **⚠️  Provenance defaults to
            // `RedistributionForbidden`**: the ESPnet-JSUT / ESPnet-JVS /
            // COEIROINK corpus terms forbid trained-weight
            // redistribution. A user who trained on a permissive corpus
            // overrides at the outer `--license <spdx>` boundary below.
            let (builder, report) = models::vits_ja::convert(bytes)?;
            let mut notes = vec![format!(
                "vits-ja: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("vits-ja warning: {n}")));
            (builder, notes)
        }
        ModelKind::DebertaV2 => {
            // SBV2 v2 plan Task 11 (2026-07-26): pass every F32/F16/BF16
            // tensor through verbatim under upstream HF names and stamp the
            // `vokra.bert.deberta_v2.*` chunk group (DeBERTa v2 transformer
            // encoder + hparams) from the transcribed constants in
            // `models::deberta_v2`. Provenance = Apache-2.0 (Permissive —
            // no runtime-side attribution obligation, per HF model card
            // `ku-nlp/deberta-v2-large-japanese-char-wwm`). Tensor-to-schema
            // mapping (Task 30) is deferred; every tensor is emitted verbatim
            // so the mapping can be validated once a real checkpoint arrives.
            let report = convert_deberta_v2_file(input, output, license)?;
            let notes = vec![format!(
                "deberta-v2: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::DebertaV2,
                tensor_count: report.written,
                metadata_count: 0, // Populated by convert_deberta_v2_file's builder
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::DebertaV3 => {
            // SBV2 v2 plan Task 11 (2026-07-26): pass every F32/F16/BF16
            // tensor through verbatim under upstream HF names and stamp the
            // `vokra.bert.deberta_v3.*` chunk group (DeBERTa v3 transformer
            // encoder + hparams) from the transcribed constants in
            // `models::deberta_v3`. Provenance = Apache-2.0 (Permissive —
            // no runtime-side attribution obligation, per HF model card
            // `ku-nlp/deberta-v3-large-japanese-char-wwm`). Tensor-to-schema
            // mapping (Task 30) is deferred; every tensor is emitted verbatim
            // so the mapping can be validated once a real checkpoint arrives.
            let report = convert_deberta_v3_file(input, output, license)?;
            let notes = vec![format!(
                "deberta-v3: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::DebertaV3,
                tensor_count: report.written,
                metadata_count: 0, // Populated by convert_deberta_v3_file's builder
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::SbV2 => {
            // SBV2 v2 plan Task 25 (2026-07-26): pass every F32/F16/BF16
            // tensor through verbatim under its upstream safetensors name.
            // This generic dispatch path has no config-side-car parameter
            // in its own signature, so it always converts with
            // config_side_car = None -- the vokra.sbv2.* hparam chunk is
            // then omitted entirely rather than filled with invented
            // placeholders (see convert_sbv2_file's doc). Call
            // convert_sbv2_file(..., Some(config_path), ...) directly for a
            // hparam-complete GGUF. Tensor-to-schema mapping (Task 30) is
            // deferred; every tensor is emitted verbatim so the mapping can
            // be validated once a real checkpoint arrives.
            let report = convert_sbv2_file(input, output, None, license)?;
            let mut notes = vec![format!(
                "sbv2: {} float weights written verbatim ({} read, {} non-float skipped), \
                 vokra.sbv2.* hparam chunk written: {}",
                report.written, report.read, report.skipped_non_float, report.hparams_written,
            )];
            if !report.hparams_written {
                notes.push(
                    "no --config side-car: vokra.sbv2.* metadata was not written -- call \
                     convert_sbv2_file directly with a config path for a hparam-complete GGUF"
                        .to_owned(),
                );
            }
            return Ok(ConvertSummary {
                model: ModelKind::SbV2,
                tensor_count: report.written,
                metadata_count: 0, // Populated by convert_sbv2_file's builder
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
    };

    // Override the stamped licence when the caller supplies the distribution
    // source's SPDX id (add_string overwrites the key in place, so the model's
    // model_id / source / attribution stamps are preserved — only the licence
    // and its class change).
    if let Some(lic) = license {
        let class = vokra_core::LicenseClass::from_license_str(lic);
        builder.add_string(
            vokra_core::gguf::chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            class.as_str(),
        );
        builder.add_string(vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE, lic);
        // The built-in `source` string names the converter's default licence
        // (e.g. "openai/whisper (MIT)"); once the licence is overridden that
        // parenthetical would contradict it, so restate the source neutrally.
        builder.add_string(
            vokra_core::gguf::chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {lic} per source)"),
        );
    }

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Like [`convert_file`], but K-quantizes the model's large weight matrices to
/// `quant` (`Q4_K` / `Q5_K` / `Q6_K`) on the way out (M1-02, FR-QT-01).
///
/// Only `whisper` (all Whisper sizes) supports quantization in M1-02; other
/// models return a [`ConvertError::Usage`]. Biases, norms and non-block-aligned
/// tensors stay in full precision, and the emitted metadata is identical to
/// the plain path — only the quantized tensors' dtype and bytes differ, so the
/// runtime loads the result through the same GGUF path (dequantizing via
/// `vokra_core::gguf::quant`).
pub fn convert_file_quantized(
    model: ModelKind,
    input: &Path,
    output: &Path,
    quant: GgmlType,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let builder = match model {
        ModelKind::Whisper => models::whisper::convert(bytes, Some(quant))?,
        // Voxtral has a quantization path (M5-15-T36), but it needs the
        // side-car config this signature cannot carry: without it the GGUF
        // gets `0` sentinels for RoPE base / RMSNorm eps / GQA split and the
        // runtime refuses the forward (FR-EX-08). Point the caller at the
        // config-aware entry rather than emitting an unloadable file.
        ModelKind::Voxtral => {
            return Err(ConvertError::Usage(
                "voxtral quantization needs the side-car config: use \
                 `vokra-cli convert --model voxtral --config <config.json> --quantize <kind>` \
                 (or `convert_voxtral_file_quantized`). Quantizing without it would emit a GGUF \
                 with `0` hparam sentinels that the runtime refuses to run."
                    .to_owned(),
            ));
        }
        other => {
            return Err(ConvertError::Usage(format!(
                "quantization (--quantize) is only supported for whisper and voxtral, not {other}"
            )));
        }
    };

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes: vec![format!("quantized weight matrices to {quant:?}")],
    })
}

/// The named quantization presets accepted by `--policy-preset` (M2-08 T06).
///
/// Presets map to a [`QuantPolicy`](models::whisper) with the shape documented
/// in `docs/design/quantization-policy.md`:
///
/// - [`PolicyPreset::VocoderSafe`] — default whole-model widen to `F16`
///   (activation-safe, matches Vocos/BigVGAN's fp16-minimum registry).
/// - [`PolicyPreset::WhisperQ4K`] — default `Q4_K` with `.bias` / `.weight_norm`
///   pinned to `F32`. Backward-compatible alias for `--quantize q4_k`.
/// - [`PolicyPreset::Fp16`] — whole-model widen to `F16` with no rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyPreset {
    /// Whole-model widen to `F16`. CLI default when `--policy-preset` is not
    /// passed.
    VocoderSafe,
    /// `Q4_K` default; `.bias` / `.weight_norm` pinned to `F32`.
    WhisperQ4K,
    /// Whole-model widen to `F16`.
    Fp16,
}

impl PolicyPreset {
    /// Parses a `--policy-preset` argument value.
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "vocoder_safe" => Some(Self::VocoderSafe),
            "whisper_q4_k" => Some(Self::WhisperQ4K),
            "fp16" => Some(Self::Fp16),
            _ => None,
        }
    }
}

/// Runs a whisper conversion with an explicit [`PolicyPreset`] (M2-08 T06).
///
/// This is the T06 successor to [`convert_file_quantized`]: the offline
/// converter now resolves each tensor's target dtype through a first-match
/// policy rather than a hardcoded `is_quantizable()` filter, and stamps the
/// resolved policy into `vokra.quant.*` metadata for the runtime to read back.
/// Piper / CAM++ / Silero are unchanged in T06 and reject the flag.
pub fn convert_file_with_policy(
    model: ModelKind,
    input: &Path,
    output: &Path,
    preset: PolicyPreset,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let builder = match model {
        ModelKind::Whisper => {
            let policy = match preset {
                PolicyPreset::VocoderSafe => models::whisper::QuantPolicy::default_vocoder_safe(),
                PolicyPreset::WhisperQ4K => models::whisper::QuantPolicy::whisper_q4_k(),
                PolicyPreset::Fp16 => models::whisper::QuantPolicy::fp16(),
            };
            models::whisper::convert_with_policy(bytes, Some(policy))?
        }
        other => {
            return Err(ConvertError::Usage(format!(
                "--policy-preset is only supported for whisper in M2-08, not {other}"
            )));
        }
    };

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes: vec![format!("applied quantization policy preset {preset:?}")],
    })
}

/// Converts a piper-plus voice (`onnx` graph + `config` JSON) into a GGUF
/// written to `output`, returning a summary (M0-07-T07).
///
/// piper-plus voices are distributed as an FP16 ONNX graph plus a `config.json`
/// (phoneme table, sample rate, inference defaults), so unlike the single-input
/// [`convert_file`] models this one takes both. See
/// [`models::piper_plus`](crate) for the naming / metadata contract.
pub fn convert_piper_plus_file(
    onnx: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let onnx_bytes = std::fs::read(onnx)?;
    let config_bytes = std::fs::read(config)?;
    let (builder, report) = models::piper_plus::convert(&onnx_bytes, &config_bytes)?;

    let notes = vec![format!(
        "piper-plus: {} float weights written ({} onnx:: names recovered), {} non-float skipped, {} phoneme ids over num_symbols",
        report.written, report.renamed, report.skipped_non_float, report.phoneme_ids_over_range
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::PiperPlus,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Converts a Kokoro-82M safetensors checkpoint plus a Kokoro `config.json`
/// (misaki phoneme symbol table + voice-name list) into a GGUF written to
/// `output`, returning a summary (M2-07-T17-fixup #3).
///
/// This is the config-aware sibling of the plain [`convert_file`] path for
/// Kokoro. The safetensors bytes are converted exactly as with
/// `convert_file(ModelKind::Kokoro, …)`; the additional config JSON supplies
/// the real `vokra.kokoro.phoneme_symbols` (misaki phoneme table) and
/// `vokra.kokoro.voice_names` arrays (canonical release ships voices as
/// separate `voices/*.pt` files, so a config is authoritative for the voice
/// list). Callers who do not yet have the misaki phoneme table can still use
/// [`convert_file`] and get the `p0..p_{n_vocab-1}` placeholder for the same
/// legacy round-trip contract.
///
/// The accepted `config.json` schema is documented on
/// [`models::kokoro`](crate) — briefly: at least one of `{vocab: {symbol:id},
/// phoneme_symbols: [str], symbols: [str]}` plus at least one of `{voices:
/// [str], voice_names: [str]}` must be present; first-match wins per family.
pub fn convert_kokoro_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let (builder, report) = models::kokoro::convert_with_config(bytes, Some(&config_bytes))?;

    let mut notes = vec![format!(
        "kokoro: {} float weights written, {} non-float skipped, style_dim {}, \
         {} voices, {} phoneme symbols",
        report.written,
        report.skipped_non_float,
        report.style_dim,
        report.voices.len(),
        report.phoneme_symbol_count,
    )];
    // Surface any per-tensor-vs-config mismatch diagnostics recorded by the
    // model routine. The converter never fails on these — the runtime is the
    // authoritative gate (FR-EX-08) — but the operator gets a loud warning.
    notes.extend(report.notes.iter().map(|n| format!("kokoro warning: {n}")));

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Kokoro,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a **prepared** DAC safetensors checkpoint together with its JSON
/// config side-car into a Vokra GGUF (M4-04 T11).
///
/// The upstream DAC release is a torch-pickle `.pth`; run
/// `tools/parity/dac_prepare_checkpoint.py` first to flatten it into a
/// safetensors + config-JSON pair (no `.pth` parser enters the converter —
/// zero-dep, NFR-DS-02). The config supplies the shape facts the checkpoint
/// metadata carried (`n_codebooks` / `codebook_size` / `codebook_dim` /
/// `d_model` / `sample_rate` / `hop_length`); the converter cross-checks them
/// against the tensor shapes and fails explicitly on any mismatch (FR-EX-08).
///
/// All upstream tensors pass through; per-quantizer decode-ready tensors
/// (`vokra.dac.quantizer.{i}.{codebook,out_proj_weight,out_proj_bias}`, with
/// the weight norm folded offline) are emitted next to them — see
/// `models/dac.rs` module docs / ADR M4-04 §D-f.
pub fn convert_dac_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let cfg = models::dac::DacConfig::parse(&config_bytes)?;
    let (builder, report) = models::dac::convert(bytes, &cfg)?;

    let notes = vec![format!(
        "dac: {} tensors passed through ({} non-float skipped), {} quantizers folded \
         (weight-norm) into vokra.dac.quantizer.* decode tensors, sample_rate {}, hop {}",
        report.written, report.skipped_non_float, cfg.n_codebooks, cfg.sample_rate, cfg.hop_length,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Dac,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a prepared SaruLab UTMOS22-strong checkpoint into a Vokra GGUF
/// (M5-15 T14).
///
/// `input` is the flat safetensors and `config` the JSON side-car that
/// `tools/parity/utmos_prepare_checkpoint.py` writes from the upstream
/// `.ckpt` (the Lightning checkpoint is a torch pickle, which the zero-dep
/// Rust converter deliberately does not parse — the same offline-prepare
/// split as DAC and Kokoro).
///
/// The mapping is total: every upstream tensor must be consumed, and any
/// left over is a hard error rather than a silent drop (FR-EX-08).
pub fn convert_utmos_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let cfg = models::utmos::UtmosConvertConfig::parse(&config_bytes)?;
    let (builder, report) = models::utmos::convert(bytes, &cfg)?;

    let notes = vec![format!(
        "utmos: {} tensor(s) emitted from {} upstream tensor(s) (all consumed), variant \
         wav2vec2_regression.v1, {} transformer layer(s) d={}, pos_conv k={} groups={} \
         (weight-norm folded), BLSTM hidden {}, judge_id {} / domain_id {}",
        report.written,
        report.consumed,
        cfg.n_layer,
        cfg.hidden_dim,
        cfg.pos_conv_kernel,
        cfg.pos_conv_groups,
        cfg.blstm_hidden,
        cfg.judge_id,
        cfg.domain_id,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Utmos,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a Sesame CSM-1B safetensors checkpoint into a Vokra GGUF,
/// optionally embedding the raw `meta-llama/Llama-3.2-1B` tokenizer file
/// as `vokra.tokenizer.model` (M4-05-T03/T04/T05).
///
/// The tokenizer repo is gated (T29 owner hand-off); passing
/// `tokenizer = None` converts without the blob and the runtime text path
/// fails loudly until a tokenizer-carrying GGUF exists (FR-EX-08 — never a
/// silent byte-level fallback).
pub fn convert_csm_file(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let tokenizer_bytes = match tokenizer {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let (builder, report) = models::csm::convert(bytes, tokenizer_bytes)?;

    let mut notes = vec![format!(
        "csm: {} float weights written, {} non-float skipped, tokenizer embedded: {}",
        report.written, report.skipped_non_float, report.tokenizer_embedded
    )];
    notes.extend(report.notes.iter().map(|n| format!("csm warning: {n}")));

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Csm,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Formats the operator-facing notes for a CosyVoice2 conversion (shared
/// by [`convert_file`] and [`convert_cosyvoice2_file`]).
fn cosyvoice2_notes(report: &models::cosyvoice2::CosyVoice2Report) -> Vec<String> {
    let mut notes = vec![match report.derived {
        Some(d) => format!(
            "cosyvoice2: {} float weights written, {} non-float skipped; derived \
             hparams: vocab={} hidden={} n_layer={} ffn={} n_head={} n_head_kv={} \
             n_ctx={} attn_bias={}",
            report.written,
            report.skipped_non_float,
            d.vocab_size,
            d.hidden_dim,
            d.n_layer,
            d.ffn_dim,
            d.n_head,
            d.n_head_kv,
            d.n_ctx,
            d.has_attn_bias,
        ),
        None => format!(
            "cosyvoice2: {} float weights written, {} non-float skipped (no LLM \
             backbone tensors — numeric hparams are 0-placeholders and the runtime \
             rejects the LLM bind at load)",
            report.written, report.skipped_non_float,
        ),
    }];
    notes.push(format!(
        "cosyvoice2: text tokenizer embedded: {}",
        report.tokenizer_embedded
    ));
    notes.extend(
        report
            .notes
            .iter()
            .map(|n| format!("cosyvoice2 warning: {n}")),
    );
    notes
}

/// Converts a CosyVoice2 LLM safetensors checkpoint (the upstream
/// `FunAudioLLM/CosyVoice2-0.5B` `llm.pt` exported with verbatim names)
/// into a Vokra GGUF, optionally consuming the upstream HF `config.json`
/// (Qwen2 schema) via `config`.
///
/// The config supplies the attention head split
/// (`num_attention_heads` / `num_key_value_heads`) plus `rope_theta` /
/// `rms_norm_eps` / `max_position_embeddings` — none of which are
/// derivable from tensor shapes (`q_out == hidden` leaves `head_dim`
/// free). Without it the GGUF carries the shape-derived hparams only and
/// the runtime refuses the LLM bind (loud, FR-EX-08). Config values are
/// cross-checked against the tensor shapes and any disagreement fails the
/// conversion.
pub fn convert_cosyvoice2_file(
    input: &Path,
    config: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = match config {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    // Qwen2 text-tokenizer side-car (T06): the upstream `vocab.json` +
    // `merges.txt` live in the same directory as `config.json`
    // (`CosyVoice-BlankEN/`). When a `--config` is given, pick them up from
    // that directory and embed both (no second CLI flag needed). A partial or
    // absent pair is a loud note in the report, not a hard error — the
    // conversion still succeeds; the runtime text path fails loudly instead.
    let tokenizer_bytes: Option<(Vec<u8>, Vec<u8>)> = config.and_then(|p| {
        let dir = p.parent().unwrap_or_else(|| Path::new("."));
        match (
            std::fs::read(dir.join("vocab.json")),
            std::fs::read(dir.join("merges.txt")),
        ) {
            (Ok(vocab), Ok(merges)) => Some((vocab, merges)),
            _ => None,
        }
    });
    let tokenizer =
        tokenizer_bytes
            .as_ref()
            .map(|(vocab, merges)| models::cosyvoice2::TokenizerFiles {
                vocab_json: vocab,
                merges_txt: merges,
            });
    let (builder, report) = models::cosyvoice2::convert_with_config_and_tokenizer(
        bytes,
        config_bytes.as_deref(),
        tokenizer,
    )?;
    let notes = cosyvoice2_notes(&report);

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::CosyVoice2,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Formats the operator-facing notes for a Fun-CosyVoice3 conversion (shared
/// by [`convert_file`] and [`convert_cosyvoice3_file`]).
///
/// The shape-derivation report is the CosyVoice2 report (delegated),
/// so the format mirrors [`cosyvoice2_notes`] with the arch label
/// swapped — an operator reading the log sees the arch they invoked.
fn cosyvoice3_notes(report: &models::cosyvoice3::CosyVoice3Report) -> Vec<String> {
    let mut notes = vec![match report.derived {
        Some(d) => format!(
            "cosyvoice3: {} float weights written, {} non-float skipped; derived \
             hparams: vocab={} hidden={} n_layer={} ffn={} n_head={} n_head_kv={} \
             n_ctx={} attn_bias={}",
            report.written,
            report.skipped_non_float,
            d.vocab_size,
            d.hidden_dim,
            d.n_layer,
            d.ffn_dim,
            d.n_head,
            d.n_head_kv,
            d.n_ctx,
            d.has_attn_bias,
        ),
        None => format!(
            "cosyvoice3: {} float weights written, {} non-float skipped (no LLM \
             backbone tensors — numeric hparams are 0-placeholders and the runtime \
             rejects the LLM bind at load)",
            report.written, report.skipped_non_float,
        ),
    }];
    notes.push(format!(
        "cosyvoice3: text tokenizer embedded: {}",
        report.tokenizer_embedded
    ));
    // The delegated CosyVoice2 walk surfaces its notes verbatim under a
    // `cosyvoice2 warning:` prefix; rewrite the prefix so operators see
    // the arch label they invoked (the rewrite is the same one the
    // converter's error paths use).
    notes.extend(report.notes.iter().map(|n| {
        format!(
            "cosyvoice3 warning: {}",
            n.replace("cosyvoice2", "cosyvoice3")
        )
    }));
    notes
}

/// Converts a Fun-CosyVoice3 LLM safetensors checkpoint (the upstream
/// `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` `llm.pt` exported with
/// verbatim names) into a Vokra GGUF, optionally consuming the upstream
/// HF `config.json` (Qwen2 schema) via `config`.
///
/// Very-cheap follow-on to [`convert_cosyvoice2_file`]: the tensor
/// walk, shape derivation, Q/K/V bias uniformity check, and Qwen2
/// tokenizer pick-up (from the config's directory) are all delegated
/// to the CosyVoice2 converter. This entry point rewrites the arch
/// label + model name + provenance + metadata chunk prefix so the
/// runtime dispatches to `vokra-models::cosyvoice3` (SoTA plan
/// §1(a) 訂正 2026-07-22: CosyVoice3's terminal vocoder is HiFTNet,
/// same as CosyVoice2 — no runtime op / kernel is duplicated). The
/// same `--config` requirement applies: without it the head split /
/// rope / eps / n_ctx stay `0`-absent and the runtime refuses the LLM
/// bind loudly (FR-EX-08).
pub fn convert_cosyvoice3_file(
    input: &Path,
    config: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = match config {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    // Qwen2 text-tokenizer side-car: the upstream `vocab.json` +
    // `merges.txt` live in the same directory as `config.json`
    // (the CosyVoice2 pick-up pattern). When a `--config` is given, pick
    // them up from that directory and embed both (no second CLI flag
    // needed). A partial or absent pair is a loud note in the report,
    // not a hard error — the conversion still succeeds; the runtime
    // text path fails loudly instead.
    let tokenizer_bytes: Option<(Vec<u8>, Vec<u8>)> = config.and_then(|p| {
        let dir = p.parent().unwrap_or_else(|| Path::new("."));
        match (
            std::fs::read(dir.join("vocab.json")),
            std::fs::read(dir.join("merges.txt")),
        ) {
            (Ok(vocab), Ok(merges)) => Some((vocab, merges)),
            _ => None,
        }
    });
    let tokenizer =
        tokenizer_bytes
            .as_ref()
            .map(|(vocab, merges)| models::cosyvoice2::TokenizerFiles {
                vocab_json: vocab,
                merges_txt: merges,
            });
    let (builder, report) = models::cosyvoice3::convert_with_config_and_tokenizer(
        bytes,
        config_bytes.as_deref(),
        tokenizer,
    )?;
    let notes = cosyvoice3_notes(&report);

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::CosyVoice3,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a Moshi (`kyutai/moshiko-pytorch-bf16`) safetensors checkpoint
/// into a Vokra GGUF, optionally embedding the raw
/// `tokenizer_spm_32k_3.model` SentencePiece file as
/// `vokra.tokenizer.model` (M4-06-T22).
///
/// **Streaming / bounded memory**: the checkpoint is opened header-only
/// and every tensor payload is copied one at a time through a reused
/// buffer ([`vokra_core::gguf::GgufStreamWriter`]), so converting the
/// 14 GiB full-7B file peaks at roughly one tensor (~0.26 GiB) — the old
/// materialize-everything path peaked ≈ 97 GiB and could not run on a
/// 16 GB machine.
///
/// **BF16 passes through verbatim** (GGUF `BF16`, ggml type 30 — the
/// Voxtral converter posture): no convert-time widening; the runtime's
/// single `tensor_f32` decode path widens BF16 → f32 **exactly** at load
/// (BF16 is the top half of the f32 pattern). The `vokra.provenance.*`
/// chunks stamp the CC-BY 4.0 `AttributionRequired` class plus the
/// FR-MD-09 attribution text the runtime surfaces
/// (`Session::attribution` / C ABI / CLI banner).
pub fn convert_moshi_file(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let tokenizer_bytes = match tokenizer {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let outcome = models::moshi::convert_streaming(input, output, tokenizer_bytes)?;
    let report = &outcome.report;

    let mut notes = vec![format!(
        "moshi: {} float weights written ({} BF16 passthrough — runtime widens to \
         f32 exactly at load), {} non-float skipped, tokenizer embedded: {}",
        report.written,
        report.bf16_passthrough,
        report.skipped_non_float,
        report.tokenizer_embedded
    )];
    notes.extend(report.notes.iter().map(|n| format!("moshi warning: {n}")));

    Ok(ConvertSummary {
        model: ModelKind::Moshi,
        tensor_count: outcome.tensor_count,
        metadata_count: outcome.metadata_count,
        output_bytes: outcome.output_bytes,
        notes,
    })
}

/// Voxtral (Mistral) side-car hparams supplied by the caller (M3-10-T04). Same
/// shape as the module-private [`models::voxtral::VoxtralConfig`], re-exported
/// here so external callers can build one without pulling in the private
/// module.
// M4-20 T12/T17: DeepFilterNet3 `denoise` offline GGUF path (real checkpoint
// parse from the prepared safetensors + synthetic round-trip writer).
pub use models::denoise::{convert_denoise_bytes, convert_denoise_file, convert_denoise_synthetic};
// SoTA plan Phase 3 (2026-07-25): StepFun Step-Audio-2-mini (apache-2.0)
// skeleton converter — every F32 / F16 / BF16 tensor passes through
// verbatim under its upstream name. Re-exported so external callers
// (vokra-cli / integration tests / model-zoo publish) can drive it
// without pulling in the private `models::step_audio2_mini` module.
pub use models::step_audio2_mini::{StepAudio2MiniReport, convert_step_audio2_mini_file};
// SoTA plan Phase 3 (2026-07-25): SparkAudio Spark-TTS **bicodec** codec
// (apache-2.0 permissive) — Vokra-native GGUF builder that emits BF16 / F16 /
// F32 tensors verbatim under their upstream safetensors names alongside the
// `vokra.provenance.upstream_hf` / `vokra.provenance.license` /
// `vokra.model.category = "codec"` chunks the future `bicodec::from_gguf`
// runtime side will read. Real-weight parity is deferred to owner
// (`docs/license-audit.md` §3.1 sign-off).
pub use models::bicodec::{BicodecReport, convert_bicodec_file};
// bshall/knn-vc (mit, category: vc) — WavLM + k-NN + HiFi-GAN, few-shot VC.
// The `_file` entry lives inside the module (SoTA plan Phase 3 pattern —
// `models::qwen3_tts` / `models::vibevoice` / `models::voxcpm2` —
// generalised with a `license: Option<&str>` override for the
// `convert_file --license <spdx>` boundary).
pub use models::knn_vc::{KnnVcReport, convert_knn_vc_file};
// 3D-Speaker ERes2Net speaker encoder (iic/speech_eres2net_sv_zh-cn_16k-common,
// apache-2.0). File-based converter with per-call license override — the model
// module is `pub mod speaker_3d` in models/mod.rs; re-exporting the surface
// here makes it reachable from external callers (the `pub fn` in the module
// alone is dead code because `mod models` itself is private).
pub use models::speaker_3d::{Speaker3dReport, convert_speaker_3d_file};
// SoTA plan Phase 5 emotion tier (2026-07-25): emotion2vec+ Large — the
// first `category = "emotion"` model in the converter tree. Standalone
// file-based entry point (not routed through `ModelKind` dispatch)
// exposes its `pub` API to external callers.
pub use models::emotion2vec::{Emotion2vecReport, convert_emotion2vec_file};
pub use models::voxtral::VoxtralConfig;
// SoTA plan Phase 5 codec (2026-07-25): fnlp XY_Tokenizer_TTSD_V0
// (apache-2.0) — self-contained file-based entry point with an SPDX
// override argument (mirror of the `denoise` re-export pattern; the
// `models::xy_tokenizer` module is private otherwise).
pub use models::xy_tokenizer::{XyTokenizerReport, convert_xy_tokenizer_file};
// SBV2 v2 plan Task 11 (2026-07-26): DeBERTa v2 / v3 (category `bert`) —
// self-contained file-based entry points, re-exported so external callers
// (integration tests / a future `ModelKind::DebertaV2`/`DebertaV3` wiring
// pass, Task 12) can reach them (the `pub fn`s alone are unreachable —
// and thus dead code — because `mod models` itself is private, same
// reasoning as the `speaker_3d` re-export above). Not yet routed through
// `ModelKind` / `convert_file` dispatch — that wiring is Task 12's job.
pub use models::deberta_v2::{ConvertReport, convert_deberta_v2_file};
pub use models::deberta_v3::convert_deberta_v3_file;
// `sbv2::ConvertReport` is re-exported under an alias, not the bare name:
// `deberta_v2::ConvertReport` already claims `vokra_convert::ConvertReport`
// above, and the two are distinct types (SBV2's carries `hparams_written`,
// DeBERTa's does not), so re-exporting both under the same crate-root name
// would collide (E0252).
pub use models::sbv2::{ConvertReport as SbV2ConvertReport, convert_sbv2_file};

/// Voxtral audio-adapter side-car (M3-10 Wave 8). Callers supply this through
/// [`convert_voxtral_file_with_adapter_config`] (a JSON path) or by
/// constructing an [`AdapterSpec`] directly and attaching it to a
/// [`VoxtralConfig::adapter`] field.
pub use models::voxtral::AdapterSpec;

/// Parses an upstream HuggingFace-style Voxtral `config.json` into a
/// [`VoxtralConfig`] (the `vokra-cli convert --model voxtral --config` path).
/// See [`models::voxtral::parse_hf_config`] for the accepted schema; a JSON
/// with no recognized Voxtral hparams is a hard error (FR-EX-08).
pub fn parse_voxtral_hf_config(bytes: &[u8]) -> Result<VoxtralConfig, ConvertError> {
    models::voxtral::parse_hf_config(bytes)
}

/// Reads a (possibly sharded) Voxtral safetensors checkpoint into one buffer
/// per shard.
///
/// A path whose file name ends in `.index.json` (the HF
/// `model.safetensors.index.json` convention the sharded Voxtral release
/// ships) is parsed for its `weight_map`; every referenced shard file is read
/// from the index's directory, in sorted order (deterministic). Any other
/// path is read verbatim as a single-file checkpoint.
fn read_voxtral_checkpoint(input: &Path) -> Result<Vec<Vec<u8>>, ConvertError> {
    use vokra_core::json::JsonValue;

    let file_name = input.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if !file_name.ends_with(".index.json") {
        return Ok(vec![std::fs::read(input)?]);
    }
    let index_bytes = std::fs::read(input)?;
    let root = vokra_core::json::parse(&index_bytes)
        .map_err(|e| ConvertError::Parse(format!("voxtral index {}: {e}", input.display())))?;
    let weight_map = root
        .get("weight_map")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            ConvertError::Parse(format!(
                "voxtral index {}: missing `weight_map` object",
                input.display()
            ))
        })?;
    // De-duplicate + sort the shard names (many tensors map to each shard).
    let mut shard_names = std::collections::BTreeSet::new();
    for (tensor, file) in weight_map {
        let f = file.as_str().ok_or_else(|| {
            ConvertError::Parse(format!(
                "voxtral index {}: weight_map[{tensor}] is not a file-name string",
                input.display()
            ))
        })?;
        shard_names.insert(f.to_owned());
    }
    if shard_names.is_empty() {
        return Err(ConvertError::Parse(format!(
            "voxtral index {}: empty weight_map — no shards to read",
            input.display()
        )));
    }
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    shard_names
        .into_iter()
        .map(|f| Ok(std::fs::read(dir.join(f))?))
        .collect()
}

/// Convert a Voxtral safetensors checkpoint together with a Mistral-format
/// side-car config into a Vokra GGUF (M3-10).
///
/// This is the config-aware sibling of the plain [`convert_file`] path for
/// Voxtral. The safetensors bytes are converted the same way as with
/// `convert_file(ModelKind::Voxtral, …)`; the additional [`VoxtralConfig`]
/// supplies the values shapes cannot recover (RoPE base, RMSNorm ε, GQA head
/// split, vocab size, max sequence length, S2S codec identifier) plus the
/// raw Mistral tokenizer bytes for `vokra.tokenizer.model`.
///
/// The shape-only [`convert_file`] path writes `0` sentinels for the missing
/// side-car values; the runtime loader will still reject a forward attempt
/// that needs them (FR-EX-08).
pub fn convert_voxtral_file(
    input: &Path,
    config: &VoxtralConfig,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_voxtral_file_quantized(input, config, output, None)
}

/// [`convert_voxtral_file`] with an optional K-quant target (M5-15-T36,
/// FR-QT-01).
///
/// `quant` is `Q4_K` / `Q5_K` / `Q6_K`; `None` reproduces
/// [`convert_voxtral_file`] byte for byte. Applicability follows the same rule
/// as the Whisper converter — rank >= 2 and a whole number of 256-element
/// super-blocks — so biases, norms and 1-D tables stay full precision. The
/// upstream release is BF16, which is read through the exact
/// `SafetensorsFile::tensor_f32` widen before quantizing.
///
/// Voxtral is the **only** model besides Whisper with a quantization path:
/// [`convert_file_quantized`]'s hard error for every other model is deliberate
/// (FR-EX-08) and unchanged.
///
/// # Errors
///
/// As [`convert_voxtral_file`], plus [`ConvertError`] from the quantizer when
/// a target dtype is not a K-quant.
pub fn convert_voxtral_file_quantized(
    input: &Path,
    config: &VoxtralConfig,
    output: &Path,
    quant: Option<GgmlType>,
) -> Result<ConvertSummary, ConvertError> {
    let shards = read_voxtral_checkpoint(input)?;
    let (builder, report) = models::voxtral::convert_shards(shards, Some(config), quant)?;

    let notes = vec![format!(
        "voxtral: {} float weights written ({} BF16 passthrough — exact, {} K-quantized to {:?}, \
         {} left full precision as quant-inapplicable), {} non-float skipped, name {}, \
         tokenizer embedded: {}",
        report.written,
        report.bf16_passthrough,
        report.quantized,
        quant,
        report.quant_inapplicable,
        report.skipped_non_float,
        report.name,
        report.tokenizer_embedded
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Voxtral,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a Voxtral safetensors checkpoint plus a [`VoxtralConfig`] plus a
/// caller-supplied **adapter config JSON** into a Vokra GGUF (M3-10 Wave 8).
///
/// This is the audio-conditioning-aware sibling of
/// [`convert_voxtral_file`]. In addition to the base config's tokenizer /
/// RoPE / RMSNorm / GQA / vocab side-car, this path also writes the
/// `vokra.voxtral.adapter.*` metadata chunk parsed from
/// `adapter_config`. The chunk tells the runtime
/// [`AudioAdapter::from_gguf`](../vokra_models/voxtral/adapter/struct.AudioAdapter.html)
/// loader where to find the checkpoint's adapter weight tensors (kind, tensor
/// prefix, in / out dims, activation, LayerNorm flags…). Tensor bytes
/// themselves are carried through by the shared safetensors-copy loop —
/// nothing invents upstream tensor names (FR-EX-08 / FR-LD-02 / FR-MD-02).
///
/// # Accepted schema
///
/// See [`AdapterSpec`] and the module docs on
/// [`models::voxtral::parse_adapter_config`](self) for the JSON schema.
///
/// The shape-only [`convert_file`] path and the tokenizer-only
/// [`convert_voxtral_file`] path stay adapter-less; the runtime then treats
/// the model as `AdapterKind::None` and keeps the honest Wave 7
/// LM-continuation posture.
pub fn convert_voxtral_file_with_adapter_config(
    input: &Path,
    config: &VoxtralConfig,
    adapter_config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_voxtral_file_with_adapter_config_quantized(input, config, adapter_config, output, None)
}

/// [`convert_voxtral_file_with_adapter_config`] with an optional K-quant
/// target (M5-15-T36). See [`convert_voxtral_file_quantized`] for the
/// applicability rule.
///
/// # Errors
///
/// As [`convert_voxtral_file_with_adapter_config`], plus quantizer errors.
pub fn convert_voxtral_file_with_adapter_config_quantized(
    input: &Path,
    config: &VoxtralConfig,
    adapter_config: &Path,
    output: &Path,
    quant: Option<GgmlType>,
) -> Result<ConvertSummary, ConvertError> {
    let adapter_bytes = std::fs::read(adapter_config)?;
    let spec = models::voxtral::parse_adapter_config(&adapter_bytes)?;
    let mut cfg = config.clone();
    cfg.adapter = Some(spec);
    let shards = read_voxtral_checkpoint(input)?;
    let (builder, report) = models::voxtral::convert_shards(shards, Some(&cfg), quant)?;

    let adapter_kind = cfg
        .adapter
        .as_ref()
        .map(|a| a.kind.as_str())
        .unwrap_or("none");
    let notes = vec![format!(
        "voxtral: {} float weights written ({} BF16 passthrough — exact, {} K-quantized to {:?}, \
         {} left full precision as quant-inapplicable), {} non-float skipped, name {}, \
         tokenizer embedded: {}, adapter kind: {}",
        report.written,
        report.bf16_passthrough,
        report.quantized,
        quant,
        report.quant_inapplicable,
        report.skipped_non_float,
        report.name,
        report.tokenizer_embedded,
        adapter_kind,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Voxtral,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a nari-labs **Dia-1.6B** safetensors checkpoint into a Vokra GGUF
/// (SoTA plan Phase 1-4, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_csm_file` /
/// `convert_dac_file` / `convert_kokoro_file`. It is functionally identical
/// to `convert_file(ModelKind::Dia, input, output)` — Dia has no side-car
/// config or tokenizer to embed (the source vocab is byte-level and the
/// hparams are transcribed as constants in `models::dia`) — but the named
/// entry keeps the `convert_*_file` naming symmetry with the other
/// TTS / codec models.
///
/// The upstream Dia release ships torch `.pth`; run a prepare-checkpoint
/// script (CSM / DAC pattern) to flatten it to safetensors first.
pub fn convert_dia_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Dia, input, output)
}

/// Convert a Resemble AI **Chatterbox-Multilingual** T3 safetensors
/// checkpoint into a Vokra GGUF (SoTA plan Phase 3, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_dia_file` /
/// `convert_zonos_file` / `convert_csm_file` / `convert_kokoro_file`. It is
/// functionally identical to `convert_file(ModelKind::Chatterbox, input,
/// output)` — Chatterbox has no side-car config or tokenizer to embed
/// (every hparam is transcribed as constants in `models::chatterbox`; the
/// release stores hparams in Python code and ships no `config.json` on HF)
/// — but the named entry keeps the `convert_*_file` naming symmetry with
/// the other TTS models.
///
/// The upstream release is `ResembleAI/chatterbox` on HuggingFace; the
/// multilingual variant weight is `t3_mtl23ls_v3.safetensors` (v2 also
/// shipped). Weight license = **MIT** (`github.com/resemble-ai/chatterbox/LICENSE`
/// — Copyright (c) 2025 Resemble AI, fetched 2026-07-24) — the M2-13 gate
/// passes commercially without any attribution obligation.
pub fn convert_chatterbox_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Chatterbox, input, output)
}

/// Convert a Resemble AI **Chatterbox-Turbo** T3 safetensors
/// checkpoint into a Vokra GGUF (SoTA plan Phase 3, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_chatterbox_file`
/// / `convert_dia_file` / `convert_zonos_file` / `convert_csm_file` /
/// `convert_kokoro_file`. It is functionally identical to
/// `convert_file(ModelKind::ChatterboxTurbo, input, output)` —
/// Chatterbox-Turbo takes no side-car config on this conversion path
/// (every hparam of the `vokra.chatterbox_turbo.*` chunk group is
/// transcribed as compile-time constants in `models::chatterbox_turbo`
/// from `t3_turbo_v1.yaml`, primary source
/// `huggingface.co/ResembleAI/chatterbox-turbo`) — but the named entry
/// keeps the `convert_*_file` naming symmetry with the other TTS
/// models.
///
/// The Turbo variant is 350M parameters (vs base's 500M) and differs
/// from base Chatterbox on three architectural axes:
/// - Backbone family: **gpt2-medium** (LayerNorm-with-bias +
///   fused-QKV-with-bias + GELU FFN) instead of Llama_520M
///   (RMSNorm + SwiGLU) — same 30 × 16 × 1024 shape.
/// - Sample rate: **32 kHz** instead of 24 kHz.
/// - Text-token vocabulary: **50 276** (GPT-2 base 50 257 + 19 native
///   paralinguistic tags: `[angry]` / `[fear]` / `[surprised]` /
///   `[whispering]` / `[cough]` / `[laugh]` / `[chuckle]` / …) instead
///   of the base's 2454 (multilingual) / 704 (English-only).
///
/// The upstream release is `ResembleAI/chatterbox-turbo` on HuggingFace;
/// the backbone weight is `t3_turbo_v1.safetensors` (~1.92 GB). Weight
/// license = **MIT** (`github.com/resemble-ai/chatterbox/LICENSE` —
/// Copyright (c) 2025 Resemble AI, fetched 2026-07-24) — the whole
/// Chatterbox family (base + Turbo + multilingual variants) ships under
/// a single MIT LICENSE. The M2-13 gate passes commercially without
/// any attribution obligation.
pub fn convert_chatterbox_turbo_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::ChatterboxTurbo, input, output)
}

/// Convert a Resemble AI **Chatterbox-Nano** T3 safetensors checkpoint
/// into a Vokra GGUF (SoTA plan Phase 3, 2026-07-24).
///
/// This is the named entry point that mirrors
/// `convert_chatterbox_file` / `convert_chatterbox_turbo_file` /
/// `convert_dia_file` / `convert_zonos_file` / `convert_csm_file` /
/// `convert_kokoro_file`. It is functionally identical to
/// `convert_file(ModelKind::ChatterboxNano, input, output)` —
/// Chatterbox-Nano takes no side-car config on this conversion path
/// (every hparam of the `vokra.chatterbox_nano.*` chunk group is
/// transcribed as compile-time constants in `models::chatterbox_nano`
/// from `t3_nano_v1.yaml`, primary source
/// `huggingface.co/ResembleAI/chatterbox-nano`) — but the named entry
/// keeps the `convert_*_file` naming symmetry with the other TTS
/// models.
///
/// The Nano variant is a compact 110M-parameter Chatterbox
/// advertised at ~3× realtime on an 8-core CPU. It **keeps base
/// Chatterbox's Llama_520M backbone** (SwiGLU + RMSNorm + RoPE — 30 ×
/// 16 × 1024 with `head_dim=64`, `ffn=4096`, `rope_theta=500000`,
/// `rms_norm_eps=1e-5`) — distinct from Turbo which swaps the
/// backbone to gpt2-medium. It **adopts Turbo's low-latency serving
/// profile**:
/// - Sample rate: **32 kHz** instead of base's 24 kHz.
/// - Text-token vocabulary: **50 276** (GPT-2 base 50 257 + 19 native
///   paralinguistic tags: `[angry]` / `[fear]` / `[surprised]` /
///   `[whispering]` / `[cough]` / `[laugh]` / `[chuckle]` / …)
///   instead of the base's 2454 (multilingual) / 704 (English-only).
/// - Speech-token vocabulary: **6563** instead of the base's 8194.
/// - Max text/speech tokens: **402/604** instead of the base's
///   2048/4096.
/// - Speech-token-to-mel decoder distilled from 10 sampling steps to
///   a single step.
///
/// **Distinguishing sentinel**: `stop_text_token = 50256` (the GPT-2
/// `<|endoftext|>` token id) — distinct from both base Chatterbox and
/// Turbo which use `0`. Nano is the only member of the family whose
/// T3 stop-text sentinel is the GPT-2 EOT id.
///
/// The upstream release is `ResembleAI/chatterbox-nano` on HuggingFace;
/// the backbone weight is `t3_nano_v1.safetensors`. Weight license =
/// **MIT** (`github.com/resemble-ai/chatterbox/LICENSE` — Copyright
/// (c) 2025 Resemble AI, fetched 2026-07-24) — the whole Chatterbox
/// family (base + Turbo + Nano + multilingual variants) ships under a
/// single MIT LICENSE. The M2-13 gate passes commercially without
/// any attribution obligation.
pub fn convert_chatterbox_nano_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::ChatterboxNano, input, output)
}

/// Convert an Alibaba **Qwen3-TTS-12Hz-0.6B-Base** safetensors
/// checkpoint into a Vokra GGUF (SoTA plan Phase 3, 2026-07-24).
///
/// This is the named entry point that mirrors
/// `convert_chatterbox_nano_file` / `convert_dia_file` /
/// `convert_zonos_file` / `convert_csm_file` / `convert_kokoro_file`. It
/// is functionally identical to
/// `convert_file(ModelKind::Qwen3Tts, input, output)` — Qwen3-TTS
/// takes no side-car config on this conversion path (every hparam of
/// the `vokra.qwen3_tts.*` chunk group is transcribed as compile-time
/// constants in `models::qwen3_tts` from
/// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json`)
/// — but the named entry keeps the `convert_*_file` naming symmetry
/// with the other TTS models.
///
/// Qwen3-TTS-0.6B-Base is Alibaba's discrete multi-codebook LM
/// speech synthesizer with a **Qwen3-flavour talker** (28-layer /
/// hidden=1024 / GQA 16 Q ÷ 8 KV / head_dim=128 / SwiGLU
/// ffn=3072 / RoPE θ=1 000 000 / RMSNorm ε=1e-6, 3072-per-codebook
/// speech vocab + 151 936-token Qwen3 shared text vocab,
/// max_position_embeddings=32 768) plus a **5-layer code-predictor
/// parallel head** (same GQA / RoPE / RMSNorm axes, 2048-per-codebook
/// acoustic vocab, emits 16 codebook rows per step) plus the shared
/// **Qwen3-TTS-Codec** seam (`vokra_ops::qwen3_tts_codec` — 16-quantizer
/// semantic + acoustic split RVQ at 12.5 Hz output rate, 24 kHz PCM).
/// Speaker encoder: 24 kHz sample rate, 1024-dim embedding.
///
/// Distinct arch tag from CosyVoice2/3 because Qwen3-TTS is
/// **codec-LM**, not vocoder-LM — the terminal step is
/// `qwen3_tts_codec`, NOT `HiFTChain`. Silently sharing either
/// sibling's arch tag would mis-route the runtime dispatch. Reuses
/// the existing `qwen3_tts_codec` primitive (SoTA plan Phase 3 TTS
/// codec op) — no new op or backend kernel is added by this model.
///
/// # BF16 posture
///
/// The upstream Qwen3-TTS-0.6B release ships **BF16** safetensors
/// (README-declared "0.9B parameters in BF16"); today's pass-through
/// arm handles only F32 / F16, so BF16 tensors reach the
/// `skipped_non_float` counter and the converter surfaces the loud
/// "no float tensors" note. Pre-widen offline (float32) or wait for
/// the streaming BF16 pass-through path (T29-equivalent — the Moshi /
/// Kyutai STT pattern) to convert the release build directly.
///
/// Weight license = **apache-2.0** **end-to-end**
/// (`huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base` model-card
/// `license: apache-2.0`, fetched 2026-07-24) — LM + codec +
/// tokenizer + speaker encoder all under a single apache-2.0 grant.
/// The M2-13 gate passes commercially without any attribution
/// obligation on the runtime side.
pub fn convert_qwen3_tts_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Qwen3Tts, input, output)
}

/// Convert an OpenBMB **VoxCPM-0.5B** safetensors checkpoint into a Vokra
/// GGUF (SoTA plan Phase 4, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_qwen3_tts_file`
/// / `convert_chatterbox_nano_file` / `convert_dia_file` /
/// `convert_zonos_file`. It is functionally identical to
/// `convert_file(ModelKind::VoxCpm2, input, output)` — VoxCPM takes no
/// side-car config on this conversion path (every hparam of the
/// `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups is
/// transcribed as compile-time constants in `models::voxcpm2` from the
/// primary sources
/// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` and
/// `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`) — but
/// the named entry keeps the `convert_*_file` naming symmetry with the
/// other TTS models.
///
/// VoxCPM-0.5B is a **new class** of TTS vs every earlier target: an
/// end-to-end **diffusion-autoregressive** speech synthesizer whose
/// terminal decoding hop is a **continuous VAE decoder** consuming
/// flow-matching sampler output — not vocoder-LM (HiFTChain) and not
/// codec-LM (any RVQ / FSQ codec). Topology:
///
/// - **MiniCPM-4 LM backbone** — decoder-only transformer,
///   `hidden_size=1024`, `num_hidden_layers=24`, GQA
///   `num_attention_heads=16` / `num_key_value_heads=2` (group ratio 8,
///   very wide compared to Qwen2/3's 2/8), SwiGLU
///   `intermediate_size=4096`, RoPE `theta=10000` with **longrope
///   scaling** (32-entry `long_factor` / `short_factor` tables — the
///   long-context extension does not widen `rope_theta`), `rms_norm_eps
///   =1e-5`, `vocab_size=73_448`, `max_position_embeddings=32_768`,
///   MiniCPM-specific scale knobs (`scale_emb=12`, `dim_model_base=256`,
///   `scale_depth=1.4`, `use_mup=false`).
/// - **Residual acoustic LM** — 6 layers, same backbone family, `vocab_size=0`.
/// - **Local encoder** — 4-layer transformer (`hidden_dim=1024`,
///   `ffn_dim=4096`, `num_heads=16`) — consumes the continuous VAE
///   feature stream, lifts it to LM width.
/// - **Local DiT + UnifiedCFM** — the **diffusion decoder**: 4-layer
///   transformer that predicts velocity in the VAE latent space, driven
///   by a conditional flow-matching sampler (`sigma_min=1e-6`,
///   `solver=euler`, `inference_cfg_rate=2.0`,
///   `t_scheduler=log-norm` — training-side; inference walks a linear
///   `t_span`). Reuses [`vokra_ops::flow_sampler`] (Euler / linear /
///   SplitBatch CFG).
/// - **Scalar-quantization bottleneck** — inline FSQ projection on the
///   LM hidden stream (`scalar_quantization_latent_dim=256`,
///   `scalar_quantization_scale=9`). Distinct from the FSQ *codec*
///   family (`wavtokenizer_vq`, `xcodec2_fsq`) — the projection stays
///   continuous.
/// - **AudioVAE V2** continuous encoder / decoder — the SoTA plan Phase
///   4 **new op** `vokra_ops::vae_continuous` introduced with this model
///   and shared with the planned VibeVoice consumer. Encoder consumes
///   16 kHz mono PCM (`sample_rate=16_000`) with `encoder_dim=128`,
///   `encoder_rates=[2,5,8,8]` (hop 640 → 25 Hz frames) → `latent_dim=64`
///   continuous latents; decoder upsamples with `decoder_dim=2048`,
///   `decoder_rates=[8,6,5,2,2,2]` (hop 1920 → `out_sample_rate=48_000`
///   PCM out), `depthwise=true`. **VAE handshake**: the LM step feature
///   width (`feat_dim=64`) MUST equal the VAE latent width
///   (`latent_dim=64`); the runtime rejects a mismatch loudly at load
///   per FR-EX-08.
///
/// # BF16 posture
///
/// The upstream VoxCPM-0.5B release ships **BF16** safetensors
/// (`config.json.dtype = "bfloat16"`); today's F32/F16 pass-through arm
/// hits the `skipped_non_float` counter on BF16 tensors and the
/// converter surfaces the loud "no float tensors" note. Pre-widen
/// offline (F32) or wait for the streaming BF16 pass-through path
/// (T29-equivalent — the Moshi / Kyutai STT pattern) to convert the
/// release build directly.
///
/// Weight license = **apache-2.0** **end-to-end**
/// (`huggingface.co/openbmb/VoxCPM-0.5B` model-card + `LICENSE`,
/// fetched 2026-07-24) — code + weight all under a single apache-2.0
/// grant. The M2-13 gate passes commercially without any attribution
/// obligation on the runtime side.
pub fn convert_voxcpm2_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::VoxCpm2, input, output)
}

/// Convert a Microsoft **VibeVoice-1.5B** safetensors checkpoint into a
/// Vokra GGUF (SoTA plan Phase 4, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_voxcpm2_file` /
/// `convert_qwen3_tts_file` / `convert_chatterbox_nano_file` /
/// `convert_dia_file` / `convert_zonos_file` / `convert_csm_file` /
/// `convert_kokoro_file`. It is functionally identical to
/// `convert_file(ModelKind::VibeVoice, input, output)` — VibeVoice
/// takes no side-car config on this conversion path (every hparam of
/// the `vokra.vibevoice.*` chunk group is transcribed as compile-time
/// constants in `models::vibevoice` from the primary sources
/// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
/// `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
/// configuration_vibevoice.py`) — but the named entry keeps the
/// `convert_*_file` naming symmetry with the other TTS models.
///
/// VibeVoice-1.5B is the **second** consumer of the continuous VAE +
/// diffusion decoder class (after VoxCPM-0.5B) — but where VoxCPM uses
/// a UnifiedCFM flow-matching sampler, VibeVoice uses a **DDPM**
/// sampler (`v-prediction` + `cosine` β schedule + 20 reduced-step
/// inference on 1000 training steps). This axis introduces the SoTA
/// plan Phase 4 **new** primitive `vokra_ops::ddpm_sampler`; the
/// acoustic VAE half shares the existing `vokra_ops::vae_continuous`
/// primitive (introduced with VoxCPM per that module's rustdoc).
///
/// Topology:
///
/// - **Qwen2 decoder LM** — decoder-only transformer,
///   `hidden_size=1536`, `num_hidden_layers=28`, GQA
///   `num_attention_heads=12` / `num_key_value_heads=2` (group ratio
///   6), SwiGLU `intermediate_size=8960`, RoPE `theta=1_000_000` no
///   scaling, `rms_norm_eps=1e-6`, `vocab_size=151_936`,
///   `max_position_embeddings=65_536`, `tie_word_embeddings=true`,
///   `sliding_window=null`, `use_sliding_window=false`.
/// - **Acoustic σ-VAE tokenizer** — mirror-symmetric encoder /
///   decoder, `vae_dim=64`, `std_dist_type="gaussian"` with
///   `fix_std=0.5`, `encoder_ratios=decoder_ratios=[8,5,5,4,2,2]`
///   (product 3200 → 7.5 Hz frame rate at 24 kHz input),
///   `encoder_n_filters=decoder_n_filters=32`,
///   `encoder_depths="3-3-3-3-3-3-8"`,
///   `mixer_layer="depthwise_conv"`, `layernorm="RMSNorm"`,
///   `layernorm_eps=1e-5`, `causal=true`.
/// - **Semantic tokenizer** — encoder-**only** deterministic variant
///   of the same causal-Conv1d chain, `vae_dim=128`,
///   `std_dist_type="none"` with `fix_std=0`, same
///   `encoder_ratios=[8,5,5,4,2,2]`. VibeVoice does NOT decode the
///   semantic latents back to audio.
/// - **Diffusion head** — 4-layer AdaLN-modulated MLP with SwiGLU
///   FFN, `hidden_size=1536` (= LM hidden — square `cond_proj`),
///   `head_layers=4`, `head_ffn_ratio=3.0` → SwiGLU inner dim 4608,
///   `rms_norm_eps=1e-5`, `latent_size=64` (= acoustic
///   `vae_dim` — the VAE handshake),
///   `prediction_type="v_prediction"`, `diffusion_type="ddpm"`,
///   `ddpm_num_steps=1000`, `ddpm_num_inference_steps=20`,
///   `ddpm_beta_schedule="cosine"`, `ddpm_batch_mul=4`.
///
/// # BF16 posture
///
/// The upstream VibeVoice-1.5B release ships **BF16** safetensors
/// (`config.json.torch_dtype = "bfloat16"`); today's F32/F16
/// pass-through arm hits the `skipped_non_float` counter on BF16
/// tensors and the converter surfaces the loud "no float tensors"
/// note. Pre-widen offline (F32) or wait for the streaming BF16
/// pass-through path (T29-equivalent — the Moshi / Kyutai STT /
/// VoxCPM pattern) to convert the release build directly.
///
/// Weight license = **MIT** end-to-end
/// (`huggingface.co/microsoft/VibeVoice-1.5B` model-card `license: MIT`
/// alongside `github.com/microsoft/VibeVoice/blob/main/LICENSE`,
/// fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). The M2-13
/// gate passes commercially without any attribution obligation on the
/// runtime side (MIT is a `Permissive` license class, same commercial
/// verdict as apache-2.0). Note: the task description's "Apache-2.0"
/// hint is superseded by the primary-source verdict of MIT — VibeVoice
/// is MIT end-to-end.
pub fn convert_vibevoice_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::VibeVoice, input, output)
}

/// Convert an Aratako **Irodori-TTS-500M-v3** safetensors checkpoint into
/// a Vokra GGUF (SoTA plan Phase 5 JA-TTS-1, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_vibevoice_file` /
/// `convert_voxcpm2_file` / `convert_qwen3_tts_file` /
/// `convert_chatterbox_nano_file` / `convert_dia_file` /
/// `convert_zonos_file` / `convert_csm_file` / `convert_kokoro_file`. It
/// is functionally identical to
/// `convert_file(ModelKind::Irodori, input, output)` — Irodori-TTS-500M-v3
/// takes no side-car config on this conversion path (every hparam of
/// the `vokra.irodori.*` chunk group is transcribed as compile-time
/// constants in `models::irodori` from the primary sources
/// `github.com/Aratako/Irodori-TTS/blob/main/configs/train_500m_v3_phase1_body.yaml`
/// plus `..._phase2_duration.yaml` plus
/// `github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/config.py::ModelConfig`)
/// — but the named entry keeps the `convert_*_file` naming symmetry with
/// the other TTS models.
///
/// Irodori-TTS-500M-v3 is the **third** consumer of the continuous-latent
/// plus DiT class (after VoxCPM-0.5B and VibeVoice-1.5B) — but this time
/// the DiT is trained with **Rectified Flow** (Liu et al. 2022,
/// arxiv 2209.03003) instead of DDPM (VibeVoice) or the UnifiedCFM
/// flow-matching sampler with an EpsS-style schedule (VoxCPM). Sampling
/// integrates the RF ODE (`x_t = (1-t) x_0 + t z`, `v = z - x_0`) with
/// an **Euler** step over a `Schedule::Linear` or `Schedule::Sway`
/// schedule — both directly supported by the existing
/// `vokra_ops::flow_sampler` primitive (M3-05), so no new sampler is
/// added by this model.
///
/// Topology:
///
/// - **Prompt-text encoder** — Llama-family self-attention with RoPE +
///   a sigmoid gate on the output projection, initialized from the
///   LLM-JP-3 150M checkpoint (`text_tokenizer_repo = "llm-jp/llm-jp-3-150m"`,
///   Apache-2.0); `text_vocab_size=99_574`, `text_dim=512`,
///   `text_layers=10`, `text_heads=8` (`head_dim=64`),
///   `text_mlp_ratio=2.6`, `text_add_bos=true`.
/// - **Reference-latent (speaker) encoder** — self-attention transformer
///   over patched reference DACVAE latents driving speaker / style
///   conditioning; `speaker_dim=768`, `speaker_layers=8`,
///   `speaker_heads=12` (`head_dim=64`), `speaker_mlp_ratio=2.6`,
///   `speaker_patch_size=1`.
/// - **RF-DiT body** — joint-attention DiT blocks with Low-Rank AdaLN
///   modulation, SwiGLU FFN + RoPE, RMSNorm ε=1e-5; `latent_dim=32`
///   (matches the paired `Semantic-DACVAE-Japanese-32dim` codec),
///   `latent_patch_size=1`, `model_dim=1280`, `num_layers=12`,
///   `num_heads=20` (`head_dim=64`), `mlp_ratio=2.875` (SwiGLU inner
///   dim 3680), `timestep_embed_dim=512`, `adaln_rank=192`.
/// - **Duration predictor (v3 phase-2)** — integrated automatic length
///   estimation: `duration_aux_dim=14`, `duration_hidden_dim=1024`,
///   `duration_layers=3`, `duration_attention_heads=8`,
///   `duration_dropout=0.1`,
///   `duration_architecture="token_sum_adarn_zero_no_aux"`,
///   `duration_token_init_frames=9.0`,
///   `duration_speaker_fusion="adarn_zero"`.
///
/// Terminal decode: the paired `Aratako/Semantic-DACVAE-Japanese-32dim`
/// codec (a `dacvae.DACVAE` variant of the Meta open-source
/// `facebookresearch/dacvae` codec, Apache 2.0) — 32-d continuous latent
/// → 48 kHz mono PCM. Callers inject the codec through
/// `IrodoriTts::with_codec` once the paired GGUF is prepared (the same
/// `DacCodecGguf`-shaped seam Dia + Zonos use with vanilla DAC).
///
/// # BF16 posture
///
/// The upstream Irodori-TTS release trains in bf16
/// (`TrainConfig.precision = "bf16"`) but the released
/// `model.safetensors` blob is typically served in F32 / F16 (the
/// `save_pretrained` default). If a downstream ships BF16, today's
/// F32/F16 pass-through arm hits the `skipped_non_float` counter and
/// the "no float tensors" loud note fires. Pre-widen offline to F32
/// (the CSM / Kokoro / VoxCPM pattern) to convert a BF16 checkpoint
/// directly.
///
/// Weight license = **MIT** end-to-end (`github.com/Aratako/Irodori-TTS/blob/main/LICENSE`
/// verified via `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`,
/// fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). The M2-13
/// gate passes commercially without any attribution obligation on the
/// runtime side (MIT is a `Permissive` license class, same commercial
/// verdict as apache-2.0).
pub fn convert_irodori_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Irodori, input, output)
}

/// Convert an ESPnet-family Japanese **plain VITS** safetensors
/// checkpoint into a Vokra GGUF (SoTA plan Phase 5 JA-TTS-2,
/// 2026-07-24).
///
/// This is the named entry point that mirrors `convert_irodori_file` /
/// `convert_vibevoice_file` / `convert_voxcpm2_file` / etc. It is
/// functionally identical to
/// `convert_file(ModelKind::VitsJa, input, output)` — plain VITS JA
/// takes no side-car config on this conversion path today (every
/// hparam of the `vokra.vits_ja.*` chunk group is transcribed as
/// compile-time constants in `models::vits_ja` from the primary
/// sources `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
/// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` +
/// `espnet2/gan_tts/vits/{vits,generator}.py`) — but the named entry
/// keeps the `convert_*_file` naming symmetry with the other TTS
/// models.
///
/// # Architecture (from primary sources)
///
/// plain VITS JA is Kim et al. 2021 VITS (arXiv:2106.06103) — a text
/// encoder + stochastic duration predictor + normalising flow +
/// **plain HiFi-GAN generator** (Kong et al. 2020, arXiv:2010.05646).
/// Topology axes (all transcribed verbatim, fetched 2026-07-24 —
/// CLAUDE.md「ハルシネーション厳禁」):
///
/// - **Text encoder** — Conformer-style (`use_conformer_conv=false`
///   on the JA recipes), `n_layer=6`, `n_head=2`, `ffn_expand=4`,
///   `positionwise_conv_kernel=3`, `dropout=0.1`,
///   `attention_dropout=0.1`, `use_macaron_style=true`.
/// - **Stochastic duration predictor** — `kernel_size=3`,
///   `dropout=0.5`, `n_flow=4`, `dds_conv_layers=3`.
/// - **Residual affine coupling flow** — `n_flow=4`, `kernel_size=5`,
///   `base_dilation=1`, `n_layer=4`, `use_only_mean=true`.
/// - **HiFi-GAN decoder (22 kHz JA recipe)** — `kernel_size=7`,
///   `initial_channel=512`, `upsample_scales=[8, 8, 2, 2]`,
///   `upsample_kernel_sizes=[16, 16, 4, 4]`,
///   `resblock_kernel_sizes=[3, 7, 11]`,
///   `resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.
///   Distinct from piper-plus (MB-iSTFT-VITS2), which decodes
///   through a sub-band iSTFT + PQMF post-net.
/// - **Global axes** — `hidden_channels=192`, `segment_size=32`,
///   `aux_channels=513` (`n_fft/2 + 1` for the 22 kHz recipe's
///   `n_fft=1024`), `n_mels=80`, `sample_rate=22050`.
///
/// # ⚠️  Weight redistribution default — `RedistributionForbidden`
///
/// The publicly distributed ESPnet-JSUT / ESPnet-JVS / COEIROINK JA
/// VITS checkpoints ride on **corpus terms that forbid trained-weight
/// redistribution**:
///
/// - **JSUT** — *"Re-distribution is not permitted"*
///   (`sites.google.com/site/shinnosuketakamichi/publication/jsut`).
/// - **JVS** — same re-distribution ban
///   (`sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`).
/// - **COEIROINK** — per-character licence terms that a converter
///   cannot machine-check.
///
/// The converter therefore default-stamps the artifact as
/// [`vokra_core::LicenseClass::RedistributionForbidden`]. A user who
/// trained their own permissive-corpus VITS overrides at the outer
/// `--license <spdx>` boundary of `convert_file`, which rewrites the
/// provenance chunk to the correct SPDX id.
///
/// Architecture rides Apache-2.0 (ESPnet) + MIT (jaywalnut310/vits)
/// and is *always* independently implementable — the block runs
/// (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
/// See `docs/tickets/sota-coverage-plan-2026-07-22.md` §2.4 for the
/// "support the architecture, refuse the weights" rationale.
pub fn convert_vits_ja_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::VitsJa, input, output)
}

/// Convert a Zyphra **Zonos-v0.1-transformer** safetensors checkpoint into a
/// Vokra GGUF (SoTA plan Phase 1-5, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_dia_file` /
/// `convert_csm_file` / `convert_kokoro_file`. It is functionally identical
/// to `convert_file(ModelKind::Zonos, input, output)` — Zonos has no
/// side-car config or tokenizer to embed (every hparam is transcribed as
/// constants in `models::zonos`, and the eSpeak-NG phoneme conditioner keeps
/// its tokenizer state inside the tensor manifest) — but the named entry
/// keeps the `convert_*_file` naming symmetry with the other TTS / codec
/// models.
///
/// The upstream Zonos-v0.1-transformer release ships safetensors directly;
/// no `.pth` prepare step is required (unlike Dia).
pub fn convert_zonos_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Zonos, input, output)
}

/// Convert a Kyutai **STT-2.6B-EN** safetensors checkpoint into a Vokra
/// GGUF (SoTA plan Phase 2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_dia_file` /
/// `convert_zonos_file` / `convert_csm_file` / `convert_kokoro_file`. It
/// is functionally identical to
/// `convert_file(ModelKind::KyutaiStt, input, output)` — Kyutai STT has
/// no side-car config or tokenizer to embed at this scaffold stage (every
/// hparam is transcribed as constants in `models::kyutai_stt`; the
/// SentencePiece tokenizer + Mimi codec ride separate GGUFs) — but the
/// named entry keeps the `convert_*_file` naming symmetry with the other
/// ASR / TTS models.
///
/// The upstream Kyutai STT release ships raw safetensors (all BF16, ~5.2
/// GB); BF16 currently reaches the `skipped_non_float` counter and the
/// converter surfaces the "no float tensors" loud note — the
/// streaming-BF16 pass-through path is a follow-up wave (T29-equivalent,
/// the Moshi pattern). Provenance is stamped **CC-BY 4.0**
/// (`AttributionRequired`) and the FR-MD-09 attribution surface
/// activates so a downstream must show the Kyutai attribution.
pub fn convert_kyutai_stt_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::KyutaiStt, input, output)
}

/// Convert an NVIDIA **Parakeet-TDT-0.6B-v3** safetensors checkpoint
/// into a Vokra GGUF (SoTA plan Phase 2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_kyutai_stt_file`
/// / `convert_dia_file` / `convert_zonos_file`. It is functionally
/// identical to `convert_file(ModelKind::Parakeet, input, output)` —
/// Parakeet has no side-car config or tokenizer to embed at this
/// scaffold stage (every hparam is transcribed as constants in
/// `models::parakeet`; the SentencePiece tokenizer follows in a
/// follow-up wave via the `--config` side-car pattern) — but the named
/// entry keeps the `convert_*_file` naming symmetry with the other ASR
/// / TTS models.
///
/// The upstream Parakeet release ships raw safetensors (F32 per
/// `config.json` `dtype: "float32"`); BF16-converted variants currently
/// reach the `skipped_non_float` counter and the converter surfaces the
/// "no float tensors" loud note. Provenance is stamped **CC-BY 4.0**
/// (`AttributionRequired`) and the FR-MD-09 attribution surface
/// activates so a downstream must show the NVIDIA attribution.
pub fn convert_parakeet_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Parakeet, input, output)
}

/// Convert an NVIDIA **Parakeet-CTC-1.1B** safetensors checkpoint into a
/// Vokra GGUF (SoTA plan Phase 2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_parakeet_file` /
/// `convert_kyutai_stt_file`. It is functionally identical to
/// `convert_file(ModelKind::ParakeetCtc, input, output)` — Parakeet-CTC
/// has no side-car config or tokenizer to embed at this scaffold stage
/// (every hparam is transcribed as constants in `models::parakeet_ctc`;
/// the SentencePiece tokenizer follows in a follow-up wave via the
/// `--config` side-car pattern) — but the named entry keeps the
/// `convert_*_file` naming symmetry with the other ASR / TTS models.
///
/// # Architecture differences vs. Parakeet-TDT-0.6B-v3
///
/// - `num_hidden_layers` = **42** (not 24)
/// - `num_mel_bins` = **80** (not 128)
/// - `attention_bias` = **true** (not false)
/// - `scale_input` = **true** (not false)
/// - **No RNN-T prediction network, no joint / duration head** — the
///   CTC head is a single Linear from `d_model` to `vocab_size=1025`
///   (1024 SentencePiece pieces + 1 blank at `pad_token_id=1024`),
///   and decoding is a host-side runtime function
///   (`vokra_ops::ctc_decode`).
///
/// # BF16 posture
///
/// The upstream Parakeet-CTC release ships **BF16** safetensors (per
/// `config.json` `dtype: "bfloat16"`); today's pass-through arm handles
/// only F32 / F16, so BF16 tensors reach the `skipped_non_float` counter
/// and the converter surfaces the "no float tensors" loud note. Pre-widen
/// offline (float32) or wait for the streaming BF16 pass-through path
/// (T29-equivalent — the Moshi pattern) to convert the release build
/// directly. Provenance is stamped **CC-BY 4.0** (`AttributionRequired`)
/// and the FR-MD-09 attribution surface activates so a downstream must
/// show the NVIDIA attribution.
pub fn convert_parakeet_ctc_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::ParakeetCtc, input, output)
}

/// Convert an NVIDIA **Canary-1B-v2** safetensors checkpoint into a Vokra
/// GGUF (SoTA plan Phase 2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_parakeet_ctc_file` /
/// `convert_parakeet_file` / `convert_kyutai_stt_file`. It is functionally
/// identical to `convert_file(ModelKind::Canary, input, output)` — Canary
/// has no side-car config or tokenizer to embed at this scaffold stage
/// (every hparam is transcribed as constants in `models::canary`; the
/// unified SentencePiece tokenizer follows in a follow-up wave via the
/// `--config` side-car pattern) — but the named entry keeps the
/// `convert_*_file` naming symmetry with the other ASR / TTS models.
///
/// # Architecture summary
///
/// - Encoder: FastConformer, **32 layers** (model card), `d_model=1024`,
///   `n_heads=8`, `ff_expansion_factor=4` → `ffn_dim=4096`,
///   `conv_kernel_size=9`, `num_mel_bins=128`, `subsampling_factor=8`,
///   `attention_bias=true`, `scale_input=false`,
///   `max_position_embeddings=5000` (family reference defaults from
///   `fast-conformer_aed.yaml`).
/// - Decoder: Transformer, **8 layers** (model card), `d_model=1024`,
///   `n_heads=8`, `inner_size=4096`, `max_sequence_length=1024`,
///   `pre_ln=true`, `hidden_act="relu"` (family reference / flash
///   convention).
/// - Vocab: unified SentencePiece, **16 384 tokens** (model card),
///   inline task tokens (`<source_lang>`, `<target_lang>`, `<taskname>`,
///   `<pnc>`, `<itn>`, `<timestamp>`, `<diarize>`, `<emotion>`).
/// - Sample rate: **16 kHz** (model card, mono .wav / .flac).
///
/// # BF16 posture
///
/// The upstream Canary-1B-v2 `.nemo` tarball's PyTorch checkpoint is
/// typically **BF16**; today's pass-through arm handles only F32 / F16,
/// so BF16 tensors reach the `skipped_non_float` counter and the
/// converter surfaces the "no float tensors" loud note. Pre-widen offline
/// during the `.nemo` prepare step (F32) or wait for the streaming BF16
/// pass-through path (T29-equivalent — the Moshi pattern) to convert the
/// release build directly. Provenance is stamped **CC-BY 4.0**
/// (`AttributionRequired`) and the FR-MD-09 attribution surface activates
/// so a downstream must show the NVIDIA attribution.
pub fn convert_canary_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Canary, input, output)
}

/// Convert a Meta **omniASR-CTC-1B** safetensors checkpoint into a Vokra
/// GGUF (SoTA plan Phase 2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_parakeet_ctc_file` /
/// `convert_canary_file` / `convert_kyutai_stt_file`. It is functionally
/// identical to `convert_file(ModelKind::OmniasrCtc, input, output)` —
/// omniASR-CTC has no side-car config or tokenizer to embed at this
/// scaffold stage (every hparam is transcribed as constants in
/// `models::omniasr_ctc`; the fairseq2 registry walk fixes every axis,
/// and the SentencePiece char tokenizer ships separately on the HF
/// release) — but the named entry keeps the `convert_*_file` naming
/// symmetry with the other ASR / TTS models.
///
/// # Architecture summary
///
/// - Encoder: **wav2vec 2.0 waveform-in**, `model_dim=1280`, 48-layer
///   pre-norm Transformer, `num_encoder_attn_heads=16`,
///   `ffn_inner_dim=5120`. Waveform features come from a fixed 7-layer
///   Conv1D stem `[(512,10,5), (512,3,2)×4, (512,2,2)×2]` with per-layer
///   Layer Normalization and bias (large_lv60k axes). The positional
///   encoder is a single grouped Conv1D (`pos_conv_kernel_size=128`,
///   `num_pos_conv_groups=16`). The wav2vec 2.0 encoder is a distinct
///   topology from the FastConformer used by Parakeet-CTC — no shared
///   `vokra_ops::wav2vec2_encoder` op today (the task note's "may need
///   new op" is deliberately deferred).
/// - CTC head: single Linear from `model_dim=1280` to
///   `target_vocab_size=9812`, with bias (fairseq2 default
///   `final_proj_bias=True`). **`blank_id = 0`** — the fairseq2 wav2vec
///   2.0 convention (`torch.nn.functional.ctc_loss` called without an
///   explicit `blank=` argument uses `blank=0`). This is different
///   from the NeMo convention that Parakeet-CTC follows (blank at
///   `pad_token_id = vocab_size - 1`).
/// - Sample rate: **16 kHz** (model card + wav2vec 2.0 convention;
///   the HF release carries no `config.json`).
///
/// # Architecture differences vs. Parakeet-CTC-1.1B
///
/// - **Waveform input**, not log-mel bins — the encoder's Conv1D stem
///   produces the features (Parakeet-CTC takes `num_mel_bins=80` in).
/// - `num_encoder_layers` = **48** (Parakeet-CTC: 42).
/// - `model_dim` = **1280** (Parakeet-CTC: `d_model=1024`).
/// - `ffn_inner_dim` = **5120** (Parakeet-CTC: `intermediate_size=4096`).
/// - **Plain Transformer**, not FastConformer — no depthwise conv, no
///   macaron FFN, no 8× stacking subsample.
/// - `blank_id` = **0**, not vocab tail (fairseq2 vs NeMo convention).
/// - `target_vocab_size` = **9812** (Parakeet-CTC: `vocab_size=1025`) —
///   the 1600-language SentencePiece char tokenizer is much larger.
/// - **1600+ languages**, not English-only.
/// - License = **Apache-2.0** (`Permissive`), not CC-BY 4.0
///   (`AttributionRequired`) — no runtime-side attribution obligation.
///
/// # BF16 posture
///
/// The `facebook/omniASR-CTC-1B.pt` checkpoint is `torch.float32` per
/// the fairseq2 release; no BF16 pass-through is required to convert
/// the release build. A downstream that pre-widens to F16 offline
/// lands on the F16 arm (also pass-through); BF16 tensors reach the
/// `skipped_non_float` counter — never a silent widen (T29-equivalent
/// — the Moshi pattern). Provenance is stamped **Apache-2.0**
/// (`Permissive`) so the M2-13 gate passes commercially without an
/// attribution obligation on the runtime side.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_omniasr_ctc_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::OmniasrCtc, input, output)
}

/// Convert a HuggingFace **distil-whisper / distil-large-v3.5**
/// safetensors checkpoint into a Vokra GGUF (SoTA plan Phase 2,
/// 2026-07-24).
///
/// This is the named entry point that mirrors `convert_omniasr_ctc_file`
/// / `convert_canary_file` / `convert_parakeet_ctc_file` /
/// `convert_kyutai_stt_file`. It is functionally identical to
/// `convert_file(ModelKind::DistilWhisper, input, output)` —
/// distil-whisper has no side-car config or tokenizer to embed at this
/// scaffold stage (every hparam is shape-derived from the checkpoint's
/// tensors and the Whisper multilingual tokenizer boundary constants
/// are the same invariants the vanilla Whisper converter uses) — but
/// the named entry keeps the `convert_*_file` naming symmetry with the
/// other ASR / TTS models.
///
/// # Architecture summary
///
/// - **Encoder** (identical to Whisper `large-v3`): `d_model=1280`,
///   `n_audio_layer=32`, `n_audio_head=20` (head_dim=64), `ffn_dim=5120`,
///   `n_mels=128`, `n_audio_ctx=1500`.
/// - **Decoder** (the distil axis): `n_text_layer=2` (large-v3 has 32),
///   `n_text_head=20`, `n_text_ctx=448`.
/// - **Tokenizer**: large-v3 multilingual byte-level BPE, `vocab_size=51866`
///   (`eos_token_id=50257`, `decoder_start_token_id=50258`).
/// - **Sample rate**: 16 kHz (Whisper convention).
///
/// # Architecture differences vs. vanilla Whisper large-v3
///
/// - `n_text_layer` = **2** (large-v3: 32). This is the entire distil
///   difference; every other axis matches large-v3 exactly. The GGUF
///   converter enforces `n_text_layer < n_audio_layer` (FR-EX-08) so a
///   mislabelled vanilla Whisper checkpoint (32/32) cannot slip through
///   as distil-whisper.
///
/// # BF16 posture
///
/// The `distil-whisper/distil-large-v3.5` release is `torch_dtype:
/// float32` per its `config.json`, so no BF16 pass-through is required
/// to convert the release build. A downstream that pre-widens to F16
/// offline lands on the F16 arm (also pass-through); BF16 tensors
/// reach the `skipped_non_float` counter — never a silent widen. The
/// weight license stamped is **MIT** (`Permissive`) so the M2-13 gate
/// passes commercially without an attribution obligation on the
/// runtime side.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_distil_whisper_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::DistilWhisper, input, output)
}

/// Convert a Kotoba Technologies **kotoba-whisper** family
/// safetensors checkpoint into a Vokra GGUF (SoTA plan Phase 5
/// JA-ASR-2, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_distil_whisper_file`
/// / `convert_omniasr_ctc_file` / `convert_canary_file` /
/// `convert_parakeet_ctc_file` / `convert_kyutai_stt_file`. It is
/// functionally identical to `convert_file(ModelKind::KotobaWhisper,
/// input, output)` — kotoba-whisper has no side-car config or
/// tokenizer to embed at this scaffold stage (every hparam is
/// shape-derived from the checkpoint's tensors and the Whisper
/// multilingual tokenizer boundary constants are the same invariants
/// the vanilla Whisper / distil-whisper converters use) — but the
/// named entry keeps the `convert_*_file` naming symmetry with the
/// other ASR / TTS models.
///
/// # Architecture summary
///
/// - **Encoder** (identical to Whisper `large-v3`): `d_model=1280`,
///   `n_audio_layer=32`, `n_audio_head=20` (head_dim=64), `ffn_dim=5120`,
///   `n_mels=128`, `n_audio_ctx=1500`.
/// - **Decoder** (the JA-ASR-2 axis): `n_text_layer=2` (large-v3 has
///   32), `n_text_head=20`, `n_text_ctx=448`.
/// - **Tokenizer**: large-v3 multilingual byte-level BPE,
///   `vocab_size=51866` (`eos_token_id=50257`,
///   `decoder_start_token_id=50258`).
/// - **Sample rate**: 16 kHz (Whisper convention).
///
/// # Architecture differences vs. vanilla Whisper large-v3
///
/// - `n_text_layer` = **2** (large-v3: 32). This is the JA-ASR-2
///   axis — the converter reads it from the checkpoint's tensor
///   names via `count_layers`, never hard-coded to 32; the runtime's
///   shared `WhisperConfig::from_gguf` (data-driven since M0)
///   honors whatever value lands here. The converter enforces
///   `n_text_layer < n_audio_layer` (FR-EX-08) so a mislabelled
///   vanilla Whisper checkpoint (32/32) cannot slip through as
///   kotoba-whisper.
///
/// # Distinct from distil-whisper (same shape, different upstream)
///
/// kotoba-whisper and `distil-whisper/distil-large-v3.5` share the
/// exact same architectural shape, but kotoba-whisper is Apache-2.0
/// (distinct from distil-whisper's MIT) and is Japanese-specialized
/// (distilled on ReazonSpeech). The compliance registry resolves
/// both to Permissive, but the GGUF provenance stamp differs
/// (`weight_license = "Apache-2.0"` here vs `"MIT"` in the
/// distil-whisper converter) and the arch tag is distinct
/// (`"kotoba-whisper"` vs `"distil-whisper"`).
///
/// # BF16 posture
///
/// The kotoba-whisper releases are `torch_dtype: float32` /
/// `float16` per their `config.json`, so no BF16 pass-through is
/// required to convert the release builds. A downstream that
/// pre-widens to F16 offline lands on the F16 arm (also
/// pass-through); BF16 tensors reach the `skipped_non_float`
/// counter — never a silent widen. The weight license stamped is
/// **Apache-2.0** (`Permissive`) so the M2-13 gate passes
/// commercially without an attribution obligation on the runtime
/// side.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_kotoba_whisper_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::KotobaWhisper, input, output)
}

/// Rewrite an existing GGUF's provenance metadata without re-materialising its
/// tensor payloads.
///
/// This is the low-memory publish path used when a converted artifact was
/// stamped with an incomplete provenance group (or none), and re-running the
/// full converter is impractical because the checkpoint no longer fits in this
/// host's RAM. The input is opened via [`vokra_mmap`] so tensor bytes are
/// fault-in-only, and every payload is streamed straight into a new file via
/// [`GgufStreamWriter`] — peak footprint stays at roughly one tensor plus
/// mapped-page cost, not the whole file.
///
/// `license` is the raw SPDX id (class re-derived from it); `model_id` and
/// `source` are advisory provenance strings; `attribution`, when `Some`, sets
/// the CC-BY display text a downstream must show.
///
/// # Errors
///
/// [`ConvertError`] if the input cannot be opened/parsed, a tensor payload is
/// malformed, or the output cannot be written.
#[allow(clippy::too_many_arguments)]
pub fn restamp_provenance(
    input: &Path,
    output: &Path,
    license: &str,
    model_id: &str,
    source: &str,
    attribution: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    use vokra_core::gguf::chunks;
    use vokra_core::gguf::{GgufBuilder, GgufStreamWriter, GgufTensorDecl};

    // mmap the input so tensor payloads fault in lazily (never a whole-file
    // read) — this is what keeps the 8.7 GiB Voxtral case within memory.
    let file = vokra_mmap::open_gguf(input)
        .map_err(|e| ConvertError::Parse(format!("restamp: opening {input:?}: {e}")))?;

    // Carry every existing metadata key EXCEPT the ones we set ourselves: the
    // provenance group (replaced below) and the schema stamps (the writer
    // re-emits them universally, so passing them in would duplicate).
    let mut b = GgufBuilder::new();
    for (k, v) in file.metadata() {
        if k == chunks::KEY_PROVENANCE_WEIGHT_LICENSE
            || k == chunks::KEY_PROVENANCE_LICENSE
            || k == chunks::KEY_PROVENANCE_MODEL_ID
            || k == chunks::KEY_PROVENANCE_SOURCE
            || k == chunks::KEY_PROVENANCE_ATTRIBUTION
            || k == chunks::KEY_SCHEMA_VERSION
            || k == chunks::KEY_SCHEMA_PRODUCER
        {
            continue;
        }
        b.add_metadata(k, v.clone());
    }

    // Inject provenance (same conduit the converters use).
    let class = vokra_core::LicenseClass::from_license_str(license);
    b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
    b.add_string(chunks::KEY_PROVENANCE_LICENSE, license);
    b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, model_id);
    b.add_string(chunks::KEY_PROVENANCE_SOURCE, source);
    if let Some(text) = attribution {
        b.add_string(chunks::KEY_PROVENANCE_ATTRIBUTION, text);
    }

    // Declare tensors in the input's order; the stream writer wants only
    // declarations up front, then payloads one at a time.
    let decls: Vec<GgufTensorDecl> = file
        .tensors()
        .iter()
        .map(|t| GgufTensorDecl {
            name: t.name.clone(),
            dtype: t.dtype,
            dimensions: t.dimensions.clone(),
        })
        .collect();
    let tensor_count = decls.len();

    let out_file = std::fs::File::create(output)?;
    let mut w = GgufStreamWriter::begin(std::io::BufWriter::new(out_file), &b, &decls)?;
    // Copy each payload straight from the mapping — no widening, no owned copy
    // beyond the single tensor being written.
    let infos: Vec<_> = file.tensors().to_vec();
    for info in &infos {
        let bytes = file.tensor_bytes(info);
        w.write_tensor(&info.name, bytes)?;
    }
    let out_writer = w.finish()?;
    let out_file = out_writer
        .into_inner()
        .map_err(|e| ConvertError::Io(e.into_error()))?;
    out_file.sync_all().map_err(ConvertError::Io)?;
    let output_bytes = out_file.metadata().map_err(ConvertError::Io)?.len();

    Ok(ConvertSummary {
        model: ModelKind::Voxtral, // placeholder; restamp is model-agnostic
        tensor_count,
        metadata_count: b.metadata_count(),
        output_bytes,
        notes: vec![format!(
            "restamp: {tensor_count} tensors copied verbatim from {input:?}; \
             provenance set to license={license} class={} (tensors unchanged)",
            class.as_str()
        )],
    })
}

#[cfg(test)]
mod compliance_conduit_tests {
    //! The minimal M2-13 conduit (FR-CP-05): a converter stamps a GGUF's weight
    //! license class via [`vokra_core::stamp_provenance`], and the runtime's
    //! research-flag gate reads it back. Exercised at the `GgufBuilder` level —
    //! exactly what the `convert*` routines assemble internally — so no existing
    //! converter output (and its metadata-count assertions) is disturbed.
    use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};
    use vokra_core::{CompliancePolicy, LicenseClass, check_weight_license, resolve_license_class};

    #[test]
    fn converter_stamps_permissive_and_runtime_admits_it() {
        // What a Whisper/piper converter would do (MIT = permissive).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        vokra_core::stamp_provenance(
            &mut b,
            LicenseClass::Permissive,
            "MIT",
            Some("whisper-base"),
            Some("openai/whisper-base"),
        );
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert_eq!(resolve_license_class(&file).class, LicenseClass::Permissive);
        assert!(check_weight_license(&file, &CompliancePolicy::strict()).is_ok());
    }

    #[test]
    fn converter_stamps_noncommercial_and_runtime_gates_it() {
        // A future F5-TTS / EnCodec converter stamping CC-BY-NC makes the
        // runtime refuse the weight without a research flag.
        let mut b = GgufBuilder::new();
        vokra_core::stamp_provenance(
            &mut b,
            LicenseClass::NonCommercial,
            "CC-BY-NC-4.0",
            Some("encodec"),
            None,
        );
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert!(check_weight_license(&file, &CompliancePolicy::strict()).is_err());
    }
}

/// SoTA plan Phase 2-5 (2026-07-24): every ModelKind variant the campaign
/// added — plus every alias spelling the CLI accepts for it — must be
/// exercised by unit tests. Prior coverage only touched the *canonical*
/// spelling per family via `parses_every_model_kind_and_help_lists_them`
/// (in `main.rs`); this module exhaustively pins the alias walk (so a
/// future contributor who breaks the `--config` free-form spelling gets a
/// loud compile-time / test-time signal) and the `as_arg → from_arg`
/// round-trip (so a dropped alias in `as_arg` can be caught).
#[cfg(test)]
mod modelkind_alias_and_roundtrip_tests {
    use super::ModelKind;

    /// Every `ModelKind` variant round-trips through its canonical
    /// `--model` argument value: `as_arg` yields a stable string, and
    /// `from_arg` returns the same variant back. A missing arm in
    /// `as_arg` (a `_ => "..."` catch-all would be a silent misroute)
    /// falls out immediately.
    #[test]
    fn every_variant_round_trips_through_as_arg_and_from_arg() {
        use ModelKind::*;
        for kind in [
            Whisper,
            SileroVad,
            Utmos,
            PiperPlus,
            CamPlus,
            Kokoro,
            CosyVoice2,
            CosyVoice3,
            Voxtral,
            Mimi,
            Dac,
            Csm,
            Moshi,
            Denoise,
            Dia,
            Zonos,
            KyutaiStt,
            Parakeet,
            ParakeetCtc,
            Canary,
            OmniasrCtc,
            DistilWhisper,
            KotobaWhisper,
            Chatterbox,
            ChatterboxTurbo,
            ChatterboxNano,
            Qwen3Tts,
            VoxCpm2,
            VibeVoice,
            Irodori,
            VitsJa,
        ] {
            let arg = kind.as_arg();
            assert!(
                !arg.is_empty(),
                "as_arg for {kind:?} must be a non-empty stable string"
            );
            let parsed = ModelKind::from_arg(arg)
                .unwrap_or_else(|| panic!("from_arg({arg:?}) must round-trip {kind:?}"));
            assert_eq!(
                parsed, kind,
                "from_arg({arg:?}) = {parsed:?} but round-trip target was {kind:?}"
            );
        }
    }

    /// Every SoTA-plan alias spelling the CLI accepts must dispatch to
    /// the same variant the canonical form does. A silent mis-dispatch
    /// (e.g. `chatterbox-turbo-onnx` landing on `ModelKind::Chatterbox`)
    /// would map the wrong converter path onto a caller's ckpt and
    /// silently produce a wrong-shape GGUF — this test guards against
    /// that class of drift.
    #[test]
    fn sota_plan_aliases_dispatch_to_the_intended_variant() {
        // Order: (kind, [alias spellings...])
        let cases: &[(ModelKind, &[&str])] = &[
            // Phase 3 — CosyVoice3
            (
                ModelKind::CosyVoice3,
                &[
                    "cosyvoice3",
                    "cosyvoice-3",
                    "fun-cosyvoice3",
                    "fun-cosyvoice-3",
                    "fun-cosyvoice3-0.5b",
                    "fun-cosyvoice3-0.5b-2512",
                    "fun-cosyvoice3-0_5b",
                    "fun-cosyvoice3-0_5b-2512",
                ],
            ),
            // Phase 1-4 / 1-5 aliases still present
            (ModelKind::Dia, &["dia", "dia-1.6b", "dia-1_6b"]),
            (
                ModelKind::Zonos,
                &[
                    "zonos",
                    "zonos-v0.1",
                    "zonos-v0_1",
                    "zonos-v0.1-transformer",
                ],
            ),
            // Phase 2
            (
                ModelKind::KyutaiStt,
                &[
                    "kyutai-stt",
                    "kyutai-stt-2.6b-en",
                    "kyutai-stt-2.6b",
                    "stt-2.6b-en",
                ],
            ),
            (
                ModelKind::Parakeet,
                &[
                    "parakeet",
                    "parakeet-tdt",
                    "parakeet-tdt-0.6b-v3",
                    "parakeet-tdt-0.6b",
                    "parakeet-tdt-0_6b-v3",
                    "parakeet-tdt-0_6b",
                ],
            ),
            (
                ModelKind::ParakeetCtc,
                &[
                    "parakeet-ctc",
                    "parakeet-ctc-1.1b",
                    "parakeet-ctc-1.1B",
                    "parakeet-ctc-1_1b",
                ],
            ),
            (
                ModelKind::Canary,
                &["canary", "canary-1b-v2", "canary-1b-v2-en", "canary-1b_v2"],
            ),
            (
                ModelKind::OmniasrCtc,
                &[
                    "omniasr-ctc",
                    "omniasr-ctc-1b",
                    "omniasr-ctc-1_1b",
                    "omniasr_ctc",
                    "omniasr_ctc_1b",
                ],
            ),
            (
                ModelKind::DistilWhisper,
                &[
                    "distil-whisper",
                    "distil_whisper",
                    "distil-whisper-large-v3",
                    "distil-whisper-large-v3.5",
                    "distil-whisper-large-v3_5",
                    "distil-large-v3",
                    "distil-large-v3.5",
                    "distil-large-v3_5",
                ],
            ),
            // Phase 5 JA-ASR-2 — kotoba-whisper
            (
                ModelKind::KotobaWhisper,
                &[
                    "kotoba-whisper",
                    "kotoba_whisper",
                    "kotoba-whisper-v1.0",
                    "kotoba-whisper-v1_0",
                    "kotoba-whisper-v1.1",
                    "kotoba-whisper-v1_1",
                    "kotoba-whisper-v2.0",
                    "kotoba-whisper-v2_0",
                    "kotoba-whisper-v2.1",
                    "kotoba-whisper-v2_1",
                    "kotoba-whisper-bilingual",
                    "kotoba-whisper-bilingual-v1.0",
                    "kotoba-whisper-bilingual-v1_0",
                ],
            ),
            // Phase 3 — chatterbox family (base / turbo / nano)
            (
                ModelKind::Chatterbox,
                &[
                    "chatterbox",
                    "chatterbox-multilingual",
                    "chatterbox-multilingual-v2",
                    "chatterbox-multilingual-v3",
                    "chatterbox-mtl23ls-v2",
                    "chatterbox-mtl23ls-v3",
                    "chatterbox-english",
                    "chatterbox_en",
                ],
            ),
            (
                ModelKind::ChatterboxTurbo,
                &[
                    "chatterbox-turbo",
                    "chatterbox_turbo",
                    "chatterbox-turbo-v1",
                    "chatterbox-turbo-onnx",
                ],
            ),
            (
                ModelKind::ChatterboxNano,
                &["chatterbox-nano", "chatterbox_nano", "chatterbox-nano-v1"],
            ),
            // Phase 3 — Qwen3-TTS
            (
                ModelKind::Qwen3Tts,
                &[
                    "qwen3-tts",
                    "qwen3_tts",
                    "qwen3-tts-0.6b",
                    "qwen3-tts-0_6b",
                    "qwen3-tts-12hz-0.6b-base",
                    "qwen3-tts-12hz-0_6b-base",
                    "qwen3-tts-12hz-0.6b",
                ],
            ),
            // Phase 4 — VoxCPM
            (
                ModelKind::VoxCpm2,
                &[
                    "voxcpm",
                    "voxcpm2",
                    "voxcpm-0.5b",
                    "voxcpm-0_5b",
                    "voxcpm-0.5b-base",
                    "voxcpm-0_5b-base",
                ],
            ),
            // Phase 4 — VibeVoice
            (
                ModelKind::VibeVoice,
                &[
                    "vibevoice",
                    "vibevoice-1.5b",
                    "vibevoice-1_5b",
                    "vibevoice-1.5b-base",
                    "vibevoice-1_5b-base",
                ],
            ),
            // Phase 5 JA-TTS-1 — Irodori
            (
                ModelKind::Irodori,
                &[
                    "irodori",
                    "irodori-tts",
                    "irodori_tts",
                    "irodori-tts-500m",
                    "irodori-tts-500m-v2",
                    "irodori-tts-500m-v2-voicedesign",
                    "irodori-tts-500m-v3",
                    "irodori-tts-500m-v3-base",
                    "irodori-tts-600m-v3-voicedesign",
                ],
            ),
            // Phase 5 JA-TTS-2 — VITS-JA (RedistributionForbidden per license
            // registry; the converter still recognises every alias so a
            // developer who legitimately holds the weight can convert).
            (
                ModelKind::VitsJa,
                &[
                    "vits-ja",
                    "vits_ja",
                    "vits-jp",
                    "vits_jp",
                    "espnet-vits-ja",
                    "espnet-vits-jp",
                    "espnet-jsut-vits",
                    "espnet-jvs-vits",
                    "coeiroink-vits",
                ],
            ),
            // Whisper — the historical alias.
            (ModelKind::Whisper, &["whisper", "whisper-base"]),
        ];
        for (kind, aliases) in cases {
            for a in aliases.iter() {
                let parsed = ModelKind::from_arg(a)
                    .unwrap_or_else(|| panic!("--model {a} must dispatch to {kind:?}"));
                assert_eq!(
                    parsed, *kind,
                    "--model {a} routed to {parsed:?} but the alias table says {kind:?}"
                );
            }
        }
    }

    /// A `--model` argument the alias table does not know MUST return None;
    /// the CLI reports "unknown model" and refuses to run (FR-EX-08 — no
    /// silent default onto e.g. Whisper).
    #[test]
    fn unknown_model_arg_returns_none() {
        for s in [
            "",
            "bogus",
            "whisper-large-v4",         // typo for whisper (unknown size)
            "chatterbox-hyper-v99",     // future release not yet mapped
            "irodori-something-random", // uncovered alias
            "distil-huge-v3",           // no distil-huge- prefix
            "kotoba-japanese-whisper",  // no kotoba-japanese- prefix
            "voxcpm3",                  // future major bump not aliased today
            "vits-en",                  // vits-* only covers the JA arm here
            "whisper-base.en",          // registry-only alias, not CLI arg
        ] {
            assert!(
                ModelKind::from_arg(s).is_none(),
                "{s:?} must NOT resolve to any ModelKind (unknown model)"
            );
        }
    }

    /// `Denoise` and `Utmos` are canonical single-spelling variants; there
    /// is no alias walk for them in `from_arg`, but the round-trip test
    /// covers the canonical spelling. This test pins their canonical spellings
    /// explicitly so a rename would fail loudly here rather than silently
    /// break the pre-M2-06 CLI invocations that use them.
    #[test]
    fn denoise_and_utmos_canonical_spellings_are_stable() {
        assert_eq!(ModelKind::from_arg("denoise"), Some(ModelKind::Denoise));
        assert_eq!(ModelKind::from_arg("utmos"), Some(ModelKind::Utmos));
        assert_eq!(ModelKind::Denoise.as_arg(), "denoise");
        assert_eq!(ModelKind::Utmos.as_arg(), "utmos");
    }
}
