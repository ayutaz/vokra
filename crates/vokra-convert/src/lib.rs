// clippy::doc_lazy_continuation false-positives on multi-paragraph enum-variant
// docstrings that reference [[memory-slug]] items; the linter can't tell the
// double-bracket link from a markdown list bullet. Silence file-wide so Wave 4
// entries stay readable without artificial indentation.
#![allow(clippy::doc_lazy_continuation)]

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
pub mod spm_proto;

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
    /// NVIDIA **Canary-Qwen-2.5B** safetensors checkpoint (SoTA plan
    /// reuse bundle, 2026-07-30). Multimodal ASR + LLM head-swap on top
    /// of Canary's FastConformer encoder: a `32`-layer FastConformer
    /// encoder (same axes as Canary-1B-v2 — `d_model=1024`, MHA
    /// `n_head=8`, `ffn_dim=4096`, `num_mel_bins=128`,
    /// `subsampling_factor=8`, `conv_kernel_size=9`,
    /// `max_position_embeddings=5000`, `attention_bias=true`) feeding a
    /// Qwen LLM decoder (Voxtral-style soft-prompt prefix — GQA
    /// `n_head_q=16`, `n_head_kv=8`, `head_dim=128`,
    /// `rope_theta=1_000_000`, `rms_norm_eps=1e-6`, SwiGLU) whose exact
    /// dims (`n_layer` / `hidden_dim` / `ffn_dim` / `vocab_size` /
    /// `n_ctx`) ride as `0`-placeholder sentinels pending the `.nemo`
    /// tarball's `model_config.yaml` extraction. CC-BY 4.0 weight
    /// (`AttributionRequired` via the `canary-` family prefix walk in
    /// `vokra_core::compliance::license_class`). Every F32 / F16 /
    /// BF16 tensor passes through verbatim (mirror of the Canary /
    /// qwen3_tts / vibevoice / voxcpm2 BF16 pass-through pattern).
    /// Reuses the shared `canary::CanaryEncoderConfig` (FastConformer
    /// via `vokra_ops::conformer`) and `voxtral::TextDecoderConfig`
    /// (Qwen LLM primitive) — no per-model op duplication. Distinct
    /// arch tag `"canary-qwen"` from base `"canary"` because the LM
    /// head-swap changes the decoder topology from Transformer AED to
    /// Qwen LLM soft-prompt prefix.
    CanaryQwen,
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
    /// **nyrahealth/CrisperWhisper** safetensors checkpoint (residual
    /// wave 4, 2026-08-02). A **Whisper-large-v3 fine-tune emphasising
    /// verbatim word-level timestamps** — the architecture is
    /// byte-identical to whisper-large-v3 (`d_model=1280`,
    /// `n_audio_layer=32`, `n_text_layer=32`, `n_mels=128`,
    /// `vocab_size=51866`), only the trained weights + license differ.
    /// Primary source `huggingface.co/nyrahealth/CrisperWhisper` fetched
    /// 2026-08-02 (CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// Weight license = **cc-by-nc-4.0**
    /// ([`vokra_core::LicenseClass::NonCommercial`]) → **T4 tier
    /// (Research-only)** publish path per the X-Codec-2 (2026-07-28)
    /// precedent. The M2-13 runtime gate refuses to load the resulting
    /// GGUF in commercial mode (`requires_research_flag = true`) — an
    /// operator who never touched the license flag cannot silently bring
    /// up an NC weight in production. Publish requires
    /// `publish-one.sh --allow-noncommercial`.
    ///
    /// **Reuse posture**: the converter reuses `models::whisper` via
    /// the `WhisperVariant::CrisperWhisper` arm — every architectural
    /// axis (tensor topology, `vokra.whisper.*` hparam chunk, front-end
    /// spec, alignment-head table, detokenizer) is byte-identical to the
    /// vanilla Whisper path; only the `vokra.model.arch` stamp
    /// (`crisper-whisper`, distinct from `whisper`), license class,
    /// license SPDX id (`cc-by-nc-4.0`), and provenance source citation
    /// come from the variant. No new op or backend kernel is added.
    Crisperwhisper,
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
    /// (GPT-2 base 50 257 + 19 paralinguistic tags `[angry]`/`[fear]`/
    /// `[surprised]`/`[whispering]`/`[cough]`/`[laugh]`/`[chuckle]`/… from
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
    /// Alibaba **Qwen3-TTS-12Hz-1.7B-Base** safetensors checkpoint
    /// (extension of Phase 3, added 2026-08-01, Wave 4). **Apache-2.0
    /// end-to-end** — same license posture as every 1.7B sibling. The
    /// un-fine-tuned 1.7B backbone that the CustomVoice / VoiceDesign
    /// 1.7B siblings fine-tune from. Talker axes are byte-identical to
    /// the two 1.7B fine-tuned siblings (widened from the 0.6B baseline
    /// to `hidden_size=2048` / `intermediate_size=6144` /
    /// `text_hidden_size=2048`, same `num_hidden_layers=28` /
    /// `num_attention_heads=16` / GQA `num_key_value_heads=8` /
    /// `head_dim=128`); the code-predictor axes, RoPE / RMSNorm, codec
    /// handshake, sample rate + speaker embedding are all identical to
    /// the 0.6B / CustomVoice / VoiceDesign siblings — only the HF
    /// release id + `vokra.model.name` stamp differ. A distinct
    /// `Qwen3TtsVariant::_1_7B_Base` arm (rather than a slug-only
    /// registration on `_1_7B_CustomVoice`) is required so a downstream
    /// that ships all three 1.7B GGUFs side-by-side can tell them apart
    /// by `vokra.provenance.upstream_hf` / `vokra.model.name`. Primary
    /// source
    /// `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json`
    /// fetched 2026-08-01 (CLAUDE.md「ハルシネーション厳禁」). The
    /// upstream release is BF16 (~3679 MB single BF16 safetensors,
    /// same ~1.92 B params as the two 1.7B fine-tuned siblings —
    /// `hidden_size × num_hidden_layers` widen dominates the parameter
    /// count). Convert with the CLI alias `qwen3-tts-1.7b-base` — the
    /// converter dispatches through the shared
    /// `models::qwen3_tts::convert_variant` path with
    /// `Qwen3TtsVariant::_1_7B_Base`.
    Qwen3TtsBase17B,
    /// Alibaba **Qwen3-TTS-12Hz-1.7B-CustomVoice** safetensors checkpoint
    /// (extension of Phase 3, added 2026-07-30). **Apache-2.0 end-to-end** —
    /// same license posture as the 0.6B sibling. 1.7B backbone variant
    /// tuned for zero-shot voice cloning (`config.json.tts_model_type =
    /// "custom_voice"`). Talker axes widen from the 0.6B baseline:
    /// `hidden_size=2048` (vs 1024), `intermediate_size=6144` (vs 3072),
    /// `text_hidden_size=2048` (vs 2048 = now identity-sized); the
    /// code-predictor axes, GQA head split, RoPE / RMSNorm, codec
    /// handshake, sample rate + speaker embedding are all identical to
    /// the 0.6B sibling. Primary source
    /// `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/raw/main/config.json`
    /// fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」). The upstream
    /// release is BF16 (~3.83 GB / 1 916 676 352 params BF16). Convert
    /// with `convert_qwen3_tts_1_7b_customvoice_file` — the converter
    /// dispatches through the shared `models::qwen3_tts::convert_variant`
    /// path with `Qwen3TtsVariant::_1_7B_CustomVoice`.
    Qwen3TtsCustomVoice17B,
    /// Alibaba **Qwen3-TTS-12Hz-1.7B-VoiceDesign** safetensors checkpoint
    /// (extension of Phase 3, added 2026-07-30). **Apache-2.0 end-to-end** —
    /// same license posture as the CustomVoice sibling. 1.7B backbone
    /// variant tuned for text-prompt voice-design synthesis
    /// (`config.json.tts_model_type = "voice_design"`). Talker + code-
    /// predictor axes are byte-identical to
    /// [`Self::Qwen3TtsCustomVoice17B`]; only the HF release id (and
    /// therefore the `vokra.model.name` + provenance model_id stamp)
    /// differs. Primary source
    /// `huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/raw/main/config.json`
    /// fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」). The upstream
    /// release is BF16 (~3.83 GB / 1 916 676 352 params BF16). Convert
    /// with `convert_qwen3_tts_1_7b_voicedesign_file` — the converter
    /// dispatches through the shared `models::qwen3_tts::convert_variant`
    /// path with `Qwen3TtsVariant::_1_7B_VoiceDesign`.
    Qwen3TtsVoiceDesign17B,
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
    /// Microsoft **VibeVoice-Realtime-0.5B** safetensors checkpoint
    /// (2026-08-01). MIT end-to-end -- code + weight under a single
    /// MIT grant (`huggingface.co/microsoft/VibeVoice-Realtime-0.5B`
    /// model-card `license: mit`, fetched 2026-08-01 -- CLAUDE.md
    /// 「ハルシネーション厳禁」). Streaming sibling of the 2026-07-24
    /// Phase 4 [`Self::VibeVoice`] baseline: upstream `model_type` is
    /// `vibevoice_streaming` (distinct from 1.5B's `vibevoice`), the
    /// Qwen2 backbone is reshaped to 24-layer / 896-dim / 14-head /
    /// 2-kv / FFN-4864 / `tie_word_embeddings=false` /
    /// `max_positions=8192`, the semantic tokenizer is **absent**
    /// (acoustic-tokenizer only), and a new top-level axis
    /// `tts_backbone_num_hidden_layers=20` is added. Acoustic
    /// tokenizer + diffusion-head axes (except `hidden_size=896`) are
    /// byte-identical to 1.5B per primary source.
    ///
    /// Every hparam transcribed verbatim from
    /// `huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/main/
    /// config.json` (fetched 2026-08-01 -- CLAUDE.md 「ハルシネー
    /// ション厳禁」).
    ///
    /// Distinct `vokra.model.arch` tag (`vibevoice_streaming`) from
    /// [`Self::VibeVoice`] -- silently sharing the arch tag would
    /// misroute the runtime dispatch (the two variants ship
    /// wrong-shape KV caches for each other). Convert with the CLI
    /// alias `vibevoice-realtime` -- the converter takes no config
    /// side-car (every hparam is fixed for the 0.5B release and
    /// transcribed as compile-time constants).
    ///
    /// The upstream release is BF16 (`torch_dtype = "bfloat16"`);
    /// today's BF16 pass-through emits GGUF type 30 verbatim and the
    /// runtime widens on load via `decode_bf16` (exact, `bits << 16`).
    VibeVoiceRealtime,
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
    /// **StyleTTS 2** (yl4579, Li et al. 2023 arXiv:2306.07691)
    /// safetensors checkpoint — a **config-only scaffold** target.
    /// Every F32 / F16 / BF16 tensor passes through verbatim under its
    /// upstream safetensors name; every hparam of the `vokra.styletts2.*`
    /// chunk group is transcribed **verbatim** from
    /// `github.com/yl4579/StyleTTS2/blob/main/Models/LJSpeech/config.yml` +
    /// `Models/LibriTTS/config.yml` (fetched 2026-07-30 — CLAUDE.md
    /// 「ハルシネーション厳禁」).
    ///
    /// **⚠️  Weight distribution is fail-closed by default.** The
    /// upstream README (`github.com/yl4579/StyleTTS2/blob/main/README.md`
    /// §Pre-trained Models) conditions weight use on **voice consent /
    /// disclosure** — a usage agreement, NOT a standard SPDX permissive
    /// license. The Vokra registry
    /// (`vokra-core::LicenseClass::from_id("styletts2")` /
    /// `"styletts-2"`) resolves to [`vokra_core::LicenseClass::Unknown`], which fails
    /// closed under M2-13. The provenance stamp defaults to `unknown`;
    /// `docs/license-audit.md` §3.1 StyleTTS 2 sign-off is
    /// `☑ Rejected 2026-07-23 yousan` (weight redistribution declined).
    /// A user who trained their own StyleTTS 2 on a permissive corpus
    /// overrides at the outer `vokra-convert --license <spdx>` boundary
    /// — the same escape hatch vits-ja / kokoro / whisper use.
    ///
    /// The runtime `styletts2::StyleTts2Tts::from_gguf` is **also**
    /// deliberately unwired (returns `VokraError::NotImplemented`
    /// naming the licence blocker); a future wave binds real weights
    /// through it when a permissive-license StyleTTS 2 checkpoint
    /// arrives. Architecture rides MIT code
    /// (`github.com/yl4579/StyleTTS2/LICENSE`) and is *always*
    /// independently implementable (whisper.cpp 型 self re-implementation,
    /// CLAUDE.md 設計判断 4). See
    /// `docs/tickets/sota-coverage-plan-2026-07-22.md` §2.4 for the
    /// "support the architecture, refuse the weights" rationale.
    StyleTts2,
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
    /// microsoft **DeBERTa v3** English BERT checkpoint (SBV2 v2 plan Task
    /// 11, 2026-07-26; upstream/license label corrected 2026-07-27, Task 8):
    /// a Hugging Face `transformers` `deberta_v3` safetensors checkpoint
    /// for English text (`microsoft/deberta-v3-large`, MIT model-card
    /// header — v3 is the SBV2 `checkpoint.bert_en` counterpart to v2's
    /// `checkpoint.bert_ja`, see `crates/vokra-bert/src/lib.rs` and
    /// `tests/fixtures/sbv2/README.md`). F32 / F16 / BF16 tensors pass
    /// through verbatim under upstream HF names; the runtime's
    /// `DebertaV3Encoder::from_gguf` will be written to map those names to
    /// the encoder's internal tensor access pattern (Task 30 — today the
    /// tensor-to-schema mapping is a deferred follow-up; every tensor is
    /// emitted verbatim so the mapping can be validated once a real
    /// checkpoint arrives). Every hparam required by the encoder is
    /// transcribed verbatim from the checkpoint's `config.json` and
    /// written to the `vokra.bert.deberta_v3.*` metadata chunk group.
    /// Convert with [`convert_deberta_v3_file`] with a safetensors
    /// checkpoint.
    DebertaV3,
    /// **hfl/chinese-roberta-wwm-ext-large** plain-BERT checkpoint
    /// (WP-14, 2026-08-10). `BertForMaskedLM` Chinese BERT-large from
    /// HFL (whole-word masking, 21,128-piece WordPiece vocab, 1024
    /// hidden, 24 layers, 16 heads). **Apache-2.0** upstream weight
    /// (owner-side sign-off queued at `docs/license-audit.md` §3.1;
    /// primary source verified 2026-08-10 via HF cardData). Wired into
    /// the SBV2 v2 `language_id = 2` (ZH) BERT slot at runtime via
    /// [`SbV2Model::from_gguf_with_zh_bert`](https://docs.rs/vokra-models).
    /// Arch-different from [`Self::DebertaV2`] / [`Self::DebertaV3`]:
    /// post-norm, standard attention, no disentangled position path,
    /// no per-layer `wq_pos` / `wk_pos` / `pos_embed` duplication —
    /// see [`crate::models::bert_base`] for the rename table and
    /// [`crate::models::bert_base::convert_bert_base_file`] for the
    /// vocab-txt `--tokenizer` side-car channel. F32 / F16 / BF16
    /// tensors pass through verbatim under the `bert_base.*` names
    /// [`BertBaseEncoder::from_gguf`](https://docs.rs/vokra-bert)
    /// reads.
    BertBase,
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
    /// **RMVPE** (Robust Model for Vocal Pitch Estimation) safetensors
    /// checkpoint (F0 pitch-extractor tier, 2026-07-30). Neural pitch
    /// estimator required by RVC v2 and reused by GPT-SoVITS /
    /// retrieval-based VC pipelines: a U-Net encoder (5 down blocks) +
    /// intermediate GRU + U-Net decoder (5 up blocks) + 360-pitch-class
    /// head over a 128-mel spectrogram at 16 kHz PCM in. MIT weight +
    /// code (upstream `Dream-High/RMVPE` + `yxlllc/RMVPE` LICENSE both
    /// = MIT, fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」)
    /// → [`vokra_core::LicenseClass::Permissive`]. Every F32 / F16 / BF16 tensor
    /// passes through verbatim under upstream state_dict names; the
    /// `vokra.rmvpe.*` chunk group carries the primary-source hparams
    /// (hop=160, sr=16000, n_mels=128, win_length=1024, n_fft=2048,
    /// n_class=360, cents_per_class=20.0, base_hz=32.703). Distinct
    /// arch tag from every sibling (`ModelKind::Rmvpe` → `rmvpe`) —
    /// silently sharing would misroute the runtime dispatch (an ASR /
    /// TTS backbone would try to interpret the 360-class pitch head).
    /// Convert with [`convert_rmvpe_file`]; no side-car config today
    /// (every hparam is a fixed compile-time constant transcribed from
    /// the upstream release).
    Rmvpe,
    /// HKUSTAudio **X-Codec 2** safetensors checkpoint (SoTA plan Phase 5
    /// codec, 2026-07-28). Neural audio codec paired with the Llasa TTS
    /// family. Distinct arch tag from every sibling codec (Mimi / DAC /
    /// WavTokenizer / neucodec / step_audio2_mini) — X-Codec 2 is an
    /// **FSQ** codec (finite scalar quantization, single FSQ level bank
    /// per codebook), not RVQ, so silently sharing would mis-route the
    /// runtime dispatch (FSQ has no residual chain).
    ///
    /// The M4-16 op-only landing implemented the FSQ decode path
    /// (`xcodec2_fsq`, `crates/vokra-ops/src/fsq_codec.rs`, parity
    /// fixture is synthetic vector-quantize-pytorch 1.17.8 projection);
    /// this converter completes the "safetensors → GGUF" side. Every
    /// F32 / F16 / BF16 tensor passes through verbatim under its upstream
    /// safetensors name (the neucodec / step_audio2_mini contract).
    ///
    /// **Weight license default = `cc-by-nc-4.0`
    /// ([`vokra_core::LicenseClass::NonCommercial`])**: the HF model
    /// card at `huggingface.co/HKUSTAudio/xcodec2` carries `license:
    /// cc-by-nc-4.0` on its YAML front-matter (CC-verified 2026-07-15;
    /// sign-off 2026-07-23 yousan = ☑ Research-only,
    /// `docs/license-audit.md` §3.1). The runtime M2-13 gate refuses to
    /// load the resulting GGUF in commercial mode
    /// (`requires_research_flag = true`) — an operator who never touched
    /// the license flag cannot silently bring up an NC weight in
    /// production. A user who legitimately holds the weight under a
    /// distinct SPDX id overrides at the outer
    /// `convert_file --license <spdx>` boundary (the same pattern
    /// vits-ja / Whisper / kokoro use).
    XCodec2,
    /// Moonshot AI **Kimi-Audio-7B-Instruct** safetensors checkpoint
    /// (SoTA plan Phase 5 fleet, 2026-07-28). Category = `s2s`. BF16
    /// pass-through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim under its upstream safetensors name; the
    /// runtime binding + real-weight parity are deferred to owner
    /// (docs/license-audit.md §3.1 sign-off queue). Provenance =
    /// **MIT** (Permissive — no runtime-side attribution obligation).
    /// The `--license <spdx>` override at the outer boundary lets a
    /// caller ship the artifact under a distinct SPDX id.
    KimiAudio,
    /// StepFun **Step-Audio-2-mini** safetensors checkpoint (SoTA
    /// plan Phase 3, 2026-07-25). Category = `s2s`. 8B S2S with a
    /// dual codebook (semantic 1024 + acoustic 4096) and a
    /// flow-matching mel decoder. BF16 pass-through skeleton — every
    /// F32 / F16 / BF16 tensor passes through verbatim; real-weight
    /// parity is deferred to owner (docs/license-audit.md §3.1
    /// sign-off queue). Distinct arch tag from every sibling —
    /// silently sharing would mis-route the runtime dispatch.
    /// Provenance = **apache-2.0** (Permissive).
    StepAudio2Mini,
    /// Baichuan-Inc **Baichuan-Audio** safetensors checkpoint (SoTA
    /// plan follow-on, 2026-07-25). Category = `s2s`. Baichuan
    /// Omni-1.5 = Whisper-Large encoder + 8-layer RVQ 12.5 Hz +
    /// Flow Matching mel + CosyVoice2 HiFi-GAN. BF16 pass-through
    /// skeleton — every F32 / F16 / BF16 tensor passes through
    /// verbatim following the qwen3_tts / vibevoice / voxcpm2
    /// pattern; real-weight parity is deferred to owner
    /// (docs/license-audit.md §3.1 sign-off queue). Provenance =
    /// **apache-2.0** (Permissive).
    BaichuanAudio,
    /// fnlp **SpeechTokenizer** safetensors checkpoint (SoTA plan
    /// Phase 5 codec fleet, 2026-07-28). Category = `codec`. RVQ
    /// audio tokenizer paired with the SpeechGPT family. BF16
    /// pass-through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim under its upstream safetensors name; the
    /// runtime binding + real-weight parity are deferred to owner.
    /// Provenance = **apache-2.0** (Permissive).
    Speechtokenizer,
    /// Alibaba DAMO **FunCodec** safetensors checkpoint (SoTA plan
    /// Phase 5 codec fleet, 2026-07-28). Category = `codec`. MIT
    /// weight; the `funcodec-encodec-*` slug reuses the Meta EnCodec
    /// naming convention on the upstream side but the code is an
    /// independent MIT re-implementation (see the
    /// `check-encodec-exclusion.sh` allowlist for the M2-13 gate
    /// carve-out). BF16 pass-through skeleton — every F32 / F16 /
    /// BF16 tensor passes through verbatim. Provenance = **MIT**
    /// (Permissive).
    Funcodec,
    /// fnlp **XY_Tokenizer_TTSD_V0** safetensors checkpoint (SoTA
    /// plan Phase 5 codec, 2026-07-25). Category = `codec`. 1 kbps
    /// RVQ-8 @ 12.5 Hz — the codec half of MOSS-TTSD. BF16 pass-
    /// through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim following the qwen3_tts / vibevoice /
    /// voxcpm2 landed contract. Provenance = **apache-2.0**
    /// (Permissive).
    XyTokenizer,
    /// SparkAudio **Spark-TTS BiCodec** safetensors checkpoint (SoTA
    /// plan Phase 5 codec fleet, 2026-07-28). Category = `codec`.
    /// Dual-codebook (semantic + acoustic) codec paired with
    /// Spark-TTS. BF16 pass-through skeleton — every F32 / F16 /
    /// BF16 tensor passes through verbatim; the runtime binding is
    /// deferred to owner. Provenance = **apache-2.0** (Permissive).
    Bicodec,
    /// Neuphonic **NeuCodec** safetensors checkpoint (SoTA plan
    /// Phase 5 codec fleet, 2026-07-28). Category = `codec`. Neural
    /// audio codec (RVQ) for streaming TTS. BF16 pass-through
    /// skeleton — every F32 / F16 / BF16 tensor passes through
    /// verbatim under its upstream safetensors name. Provenance =
    /// **apache-2.0** (Permissive).
    Neucodec,
    /// **fjiang9/NKF-AEC** (Neural Kalman Filter AEC) safetensors
    /// checkpoint (coverage-audit-2026-08-03 Wave A). Category =
    /// `aec`. Yang et al. ICASSP 2023 arXiv:2207.11388 — a
    /// low-complexity neural echo canceller (5.3 KB `.pt`). The
    /// converter is a *neural* alternative sibling to the algorithmic
    /// M4-03 `vokra_aec_*` (SpeexDSP / WebRTC AEC3 Rust port), not a
    /// replacement — the audit places both AEC families side by side
    /// so downstream users can choose. Upstream is GitHub-only (no HF
    /// mirror) so provenance is stamped as
    /// `vokra.provenance.upstream_url = github.com/fjiang9/NKF-AEC`.
    /// The upstream release ships torch pickle `.pt`, so callers
    /// pre-flatten it to safetensors offline via
    /// `tools/parity/nkf_aec_prepare_checkpoint.py` (the DFN3 / DAC /
    /// Kokoro pickle-bridge pattern). BF16 pass-through skeleton —
    /// every F32 / F16 / BF16 tensor passes through verbatim under
    /// its upstream state-dict key; the runtime binding
    /// (native GEMV forward — the ticket's "5.3 KB ゆえ single-pass
    /// GEMV で完結" note) is deferred to owner sign-off in
    /// `docs/license-audit.md §3.1`. Provenance = **MIT**
    /// (Permissive — `Copyright (c) 2022 Fei Jiang`).
    NkfAec,
    /// **Xiph RNNoise v0.2** weight blob (coverage-audit 2026-08-03
    /// Wave A ticket). Category = `denoise`. Real-time noise
    /// reduction — a compact GRU stack (`input_dense` 42→24 →
    /// `vad_gru` / `noise_gru` / `denoise_gru` → `denoise_output`
    /// 96→22, plus a `vad_output` 24→1 auxiliary head) over 22-band
    /// Bark filterbank features (Valin 2018, arXiv:1709.08243).
    /// **Distinct arch tag from [`Self::Denoise`]** (DeepFilterNet3):
    /// DFN3 is complex-Conv + ERB deep-filtering + `vokra.denoise.*`,
    /// RNNoise is tiny-GRU + Bark + `vokra.rnnoise.*` (`ARCH =
    /// "rnnoise"`); silently sharing the arch tag would mis-route the
    /// runtime dispatch (a DFN3 loader would try to interpret
    /// RNNoise's `input_dense` as `enc_conv0`). Provenance =
    /// **BSD-3-Clause** (Permissive — `github.com/xiph/rnnoise/blob/
    /// main/COPYING`). Upstream ships as a ~90 KB C-array blob
    /// (`weights_blob_9.bin` in the v0.2 GitHub Release tarball) —
    /// `tools/parity/rnnoise_prepare_checkpoint.py` flattens it to
    /// safetensors before entering this converter (mirror of the
    /// DAC / DFN3 / CSM prepared-checkpoint contract; no C / Python
    /// enters the runtime, NFR-DS-02). BF16 pass-through skeleton —
    /// every F32 / F16 / BF16 tensor passes through verbatim; the
    /// runtime binding + real-weight parity are deferred to owner
    /// (docs/license-audit.md §3.1 sign-off queue). Convert with
    /// `convert_file(ModelKind::Rnnoise, …)`.
    Rnnoise,
    /// Microsoft **NSNet2** noise-suppression baseline (Coverage-audit
    /// 2026-08-03 Wave A ticket). Category = `enhancement`. Distinct
    /// arch tag from `denoise` (DeepFilterNet3) — NSNet2 is a 2-layer
    /// GRU + 3-Linear mask predictor over 257-bin STFT log-magnitude
    /// (`nsnet2-20ms-baseline.onnx`, ~2 MB), whereas DFN3 uses an ERB
    /// analysis / synthesis pair around a convolutional recurrent
    /// network. The upstream is ONNX-only and lives on GitHub
    /// (`github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline`,
    /// not HuggingFace); `tools/parity/nsnet2_prepare_checkpoint.py`
    /// bridges ONNX → safetensors so this converter never links ONNX
    /// (FR-LD-05, NFR-DS-02). BF16 pass-through skeleton — every
    /// F32 / F16 / BF16 tensor passes through verbatim under its
    /// upstream initializer name. Provenance = **MIT** (Permissive).
    Nsnet2,
    /// Microsoft **DNSMOS P.808 + P.835** MOS predictor bundle (coverage-
    /// audit Wave A ticket `dnsmos-p808-p835`, 2026-08-03): a **prepared**
    /// safetensors (from `tools/parity/dnsmos_prepare_checkpoint.py` — the
    /// upstream release is two ONNX checkpoints
    /// `github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/` =
    /// `model_v8.onnx` (P.808) + `sig_bak_ovr.onnx` (P.835)) whose tensor
    /// names carry the `p808.` / `p835.` prefix identifying the sub-model
    /// each tensor belongs to. Both sub-models land in a single GGUF —
    /// bundle option (a) in the ticket §Converter ("2 ONNX を単一 GGUF に
    /// merge、bundle metadata で variant tag"). Every F32 tensor binds
    /// verbatim under its prefixed name; the `vokra.dnsmos.*` chunk
    /// group carries the bundle inventory and the two upstream
    /// checkpoint filenames for auditability. MIT weight + code
    /// (`microsoft/DNS-Challenge/LICENSE`, verified 2026-08-03). The
    /// runtime binder lives in the `vokra-eval` crate
    /// (`vokra_eval::dnsmos::{p808_score, p835_score}` — follow-up CC
    /// ticket, not implemented by this converter).
    Dnsmos,
    /// **FRCRN** — Frequency Recurrent Convolutional Recurrent Network
    /// (Zhao et al. ICASSP 2022, `arXiv:2206.07293`) safetensors
    /// checkpoint (coverage-audit wave-a, 2026-08-03). Category =
    /// `denoise`. Monaural speech-enhancement model with a Complex-
    /// valued U-Net + frequency-recurrent LSTM, distributed from
    /// `github.com/alibabasglab/FRCRN` and mirrored in the
    /// ClearerVoice-Studio pipeline. BF16 pass-through skeleton —
    /// every F32 / F16 / BF16 tensor passes through verbatim under
    /// its upstream safetensors name. Provenance = **apache-2.0**
    /// (Permissive). Distinct arch tag from
    /// [`ModelKind::Denoise`] (DeepFilterNet3) because the topologies
    /// are unrelated (Complex U-Net + freq-recurrent LSTM vs the DFN3
    /// ERB / DF-net stack); silently aliasing would misroute the
    /// runtime dispatch. Convert with [`convert_frcrn_file`] (or the
    /// generic `convert_file` / `convert_file_licensed`).
    Frcrn,
    // ---- coverage-audit 2026-08-03 Wave B fast-track (13 variants) ----
    /// Kyutai **Hibiki-2B** (CC-BY-4.0 attribution) — Fr↔En simultaneous
    /// S2S translation, Moshi-family arch. Distinct arch tag `hibiki`
    /// from sibling Moshi (7B chat) — hparam mismatch would mis-route
    /// a Moshi loader onto a Hibiki checkpoint. Convert with
    /// `convert_hibiki_file`.
    Hibiki,
    /// Sber **GigaAM v3** (MIT, ~500 MB–1.2 GB) — Russian SoTA ASR,
    /// Conformer + CTC/RNN-T seam. Convert with `convert_sber_gigaam_v3_file`.
    SberGigaamV3,
    /// Sber **GigaAM multilingual** (MIT, ~600 MB–1.2 GB) — 70+ language
    /// ASR, char-wise CTC. Convert with `convert_sber_gigaam_multilingual_file`.
    SberGigaamMultilingual,
    /// Reazon **reazonspeech-nemo-v2** (Apache-2.0, ~1.2 GB) — Japanese
    /// long-form ASR, NeMo Conformer + Longformer local attention.
    /// Convert with `convert_reazonspeech_nemo_v2_file`.
    ReazonspeechNemoV2,
    /// NVIDIA **MagpieTTS v2602** (Apache-2.0, ~700 MB) — 9-language
    /// multilingual TTS (en/es/de/fr/vi/it/zh/hi/ja), NeMo primitive.
    /// Convert with `convert_magpietts_v2602_file`.
    MagpiettsV2602,
    /// NVIDIA **Parakeet-unified-EN-0.6B** (Apache-2.0, ~1.2 GB) —
    /// offline+streaming EN ASR, punc/capitalization inline. Convert
    /// with `convert_parakeet_unified_file`.
    ParakeetUnified,
    /// NVIDIA **Canary-1B-Flash** (CC-BY-4.0, ~1.8 GB) — ASR + AST 4
    /// lang, 1000+ RTFx. Convert with `convert_canary_1b_flash_file`.
    Canary1bFlash,
    /// ESPnet **OWSM v4 Medium 1B** (CC-BY-4.0, ~2 GB) — open-Whisper
    /// alternative, Conformer + CTC/attention decode, 320k hr training.
    /// Convert with `convert_owsm_v4_medium_1b_file`.
    OwsmV4Medium1b,
    /// NVIDIA **Parakeet-TDT-1.1B** (CC-BY-4.0, ~2.2 GB) — English ASR,
    /// hparam-extension of sibling parakeet-tdt-0.6b-v3. Convert with
    /// `convert_parakeet_tdt_1_1b_file`.
    ParakeetTdt11b,
    /// FireRedTeam **FireRedASR-AED-L** (Apache-2.0, ~2.2 GB) — Chinese
    /// SoTA ASR, AED (Whisper-like encoder-decoder). Convert with
    /// `convert_firered_asr_aed_l_file`.
    FireredAsrAedL,
    /// NVIDIA **Sortformer diar 4spk v1** (CC-BY-4.0, ~1 GB) — e2e
    /// speaker diarization with arrival-order sort loss. Convert with
    /// `convert_sortformer_diar_4spk_v1_file`.
    SortformerDiar4spkV1,
    /// FunAudioLLM **SenseVoiceSmall** (FunASR MODEL_LICENSE, ~470 MB)
    /// — Chinese SoTA ASR + LID + SER + AED multitask. License audit
    /// deferred to owner. Convert with `convert_sensevoicesmall_file`.
    SenseVoiceSmall,
    /// aiola **whisper-medusa-v1** (Apache-2.0, ~500 MB–2 GB) — Whisper
    /// + Medusa speculative decoding head, 20-80% latency reduction.
    /// Convert with `convert_whisper_medusa_v1_file`.
    WhisperMedusaV1,
    /// Meta **facebook-denoiser** (`facebookresearch/denoiser`,
    /// **cc-by-nc-4.0**, coverage-audit-2026-08-03 Wave D T4) —
    /// real-time speech-enhancement waveform U-Net + LSTM (Defossez
    /// et al. 2020 arXiv:2006.12847). GitHub-only upstream (no HF
    /// mirror), category = `enhancement`. Distinct arch tag from
    /// sibling denoise / rnnoise / nsnet2 / frcrn. Convert with
    /// `convert_facebook_denoiser_file` — publish requires
    /// `--allow-noncommercial`.
    FacebookDenoiser,
    /// **NISQA v2 weight** (`gabrielmittag/NISQA`, **cc-by-nc-sa-4.0**,
    /// coverage-audit-2026-08-03 Wave D T4) — non-intrusive speech
    /// quality assessment CNN + self-attention (Mittag et al. 2021
    /// arXiv:2104.09494). GitHub-only upstream (no HF mirror),
    /// category = `eval` (sibling of DNSMOS / UTMOS22-strong).
    /// Convert with `convert_nisqa_v2_weight_file` — publish requires
    /// `--allow-noncommercial` AND downstream artefacts inherit the
    /// SA cascade obligation.
    NisqaV2Weight,
    /// **2Noise ChatTTS** (`2Noise/ChatTTS`, **cc-by-nc-4.0**,
    /// coverage-audit-2026-08-03 Wave D T4) — GPT-style autoregressive
    /// decoder over discrete speech tokens, conversational TTS
    /// ZH/EN. Category = `tts`, distinct arch tag from piper-plus /
    /// Kokoro / CosyVoice2 / Chatterbox / xtts. Convert with
    /// `convert_chattts_file` — publish requires `--allow-noncommercial`.
    ChatTts,
    /// **Stability AI Stable Audio Open Small**
    /// (`stabilityai/stable-audio-open-small`, **Stability AI Community
    /// License**, coverage-audit-2026-08-03 Wave D T4) — compact
    /// latent-diffusion text-to-audio generator = DiT + audio VAE +
    /// T5 text conditioner. Category = `music`, distinct arch tag
    /// from musicgen / audioldm2 / jasco / ace_step / bs_roformer.
    /// Convert with `convert_stable_audio_open_small_file` — the raw
    /// SPDX string is NOT SPDX-registered so the converter hard-maps
    /// to [`vokra_core::LicenseClass::NonCommercial`] per the CPML precedent in
    /// `xtts_v2.rs`. Publish requires `--allow-noncommercial`.
    StableAudioOpenSmall,
    /// **Meta JASCO 400M Chords+Drums**
    /// (`facebook/jasco-400m-chords-drums`, **cc-by-nc-4.0**,
    /// coverage-audit-2026-08-03 Wave D T4) — 400M parameter joint
    /// audio-symbolic conditioning music generator conditioned on
    /// text + chord progression + drum tracks (Tal et al. 2024
    /// arXiv:2406.10970). Category = `music`, distinct arch tag from
    /// sibling musicgen / audioldm2 / stable_audio_open_small /
    /// ace_step / bs_roformer. Convert with
    /// `convert_jasco_400m_chords_drums_file` — publish requires
    /// `--allow-noncommercial` per MusicGen family T4 precedent.
    Jasco400mChordsDrums,
    // ---- coverage-audit-2026-08-03 Wave A permissive continuation ----
    // (2026-08-04): 7 BF16 pass-through skeletons all in the T1
    // (Permissive) tier — MIT / BSD-2-Clause / Apache-2.0 defaults
    // land as `LicenseClass::Permissive` and sign-off ☑ Commercial by
    // yousan at land time. Two flavors: HF-hosted (Utmosv2 /
    // HtdemucsMulti / OpenwakewordOp / Mossformer2Ss16k /
    // AudiosealRealWeight) and GitHub-only (TorchaudioSquim / TenVad).
    /// **UTMOSv2** (`sarulab-speech/UTMOSv2`, MIT,
    /// coverage-audit-2026-08-03 Wave A permissive continuation) —
    /// reference-free MOS-TTS quality estimator, direct successor of
    /// UTMOS22-strong (Baba et al. arXiv:2409.09305, VoiceMOS Challenge
    /// 2024 SoTA). wav2vec2-large SSL + listener/domain conditioning +
    /// improved Regressor head. Category = `eval`, distinct arch tag
    /// from `Utmos` (UTMOS22-strong = wav2vec2-base). Convert with
    /// `convert_utmosv2_file`.
    Utmosv2,
    /// **torchaudio Squim** (`pytorch/audio`, BSD-2-Clause,
    /// coverage-audit-2026-08-03 Wave A permissive continuation) —
    /// PyTorch torchaudio SQUIM reference-free speech quality
    /// estimator, single-pass STOI + PESQ + SI-SDR (`Objective`) + MOS
    /// (`Subjective`) prediction (Kumar et al. arXiv:2304.01448).
    /// Category = `eval`, GitHub-only upstream (torch.hub distributed,
    /// no HF mirror) — stamps `vokra.provenance.upstream_url`.
    /// Convert with `convert_torchaudio_squim_file`.
    TorchaudioSquim,
    /// **HT-Demucs Multi** (`facebook/htdemucs` +
    /// `facebook/htdemucs_6s`, MIT, coverage-audit-2026-08-03 Wave A
    /// permissive continuation) — Meta AudioCraft Hybrid Transformer
    /// Demucs music source separation covering both 4-source
    /// (`htdemucs_ft`) and 6-source (`htdemucs_6s`) variants under a
    /// single ModelKind (source count rides in tensor shapes verbatim).
    /// Category = `source-separation`, distinct arch tag from
    /// `DemucsHtdemucs` (base 4-stem `facebook/demucs`). Convert with
    /// `convert_htdemucs_multi_file`.
    HtdemucsMulti,
    /// **openWakeWord op-wiring** (`dscripka/openWakeWord`,
    /// Apache-2.0 code / CC-BY-NC-SA-4.0 official weights,
    /// coverage-audit-2026-08-03 Wave A permissive continuation) —
    /// runtime-op-wiring anchor for the first-class `kws` op family
    /// (Porcupine-compatible). Distinct arch tag from the sibling
    /// base `Openwakeword` ModelKind — the `_op` variant is where
    /// user-provided weights route (Vokra does NOT redistribute the
    /// upstream official CC-BY-NC-SA-4.0 weights). Category =
    /// `vad-kws`. Convert with `convert_openwakeword_op_file`.
    OpenwakewordOp,
    /// **MossFormer2-SS-16K** (`alibabasglab/MossFormer2_SS_16K`,
    /// Apache-2.0, coverage-audit-2026-08-03 Wave A permissive
    /// continuation) — Alibaba SGLab cocktail-party / multi-speaker
    /// speech separator at 16 kHz (Zhao et al. 2024 Interspeech,
    /// ClearerVoice-Studio project). FSMN + gated-attention topology,
    /// distinct arch tag from Tsinghua MOSS project (`MossTts` /
    /// `MossAudioTokenizer` — unrelated naming collision) and from
    /// `FsmnVad` (shared FSMN block, different task head). Category
    /// = `source-separation`. Convert with
    /// `convert_mossformer2_ss_16k_file`.
    Mossformer2Ss16k,
    /// **TEN-VAD** (`TEN-framework/ten-vad`, Apache-2.0 main +
    /// BSD-3-Clause LPCNet-derived front-end,
    /// coverage-audit-2026-08-03 Wave A permissive continuation) —
    /// compact ~306 KB LSTM/GRU VAD alternative to Silero VAD v5
    /// (upstream claim: ~5.5x lighter). Category = `vad-kws`,
    /// GitHub-only upstream (no HF mirror) — stamps
    /// `vokra.provenance.upstream_url`. NOTICE attribution required
    /// for the bundled LPCNet BSD-3-Clause front-end. Convert with
    /// `convert_ten_vad_file`.
    TenVad,
    /// **AudioSeal real weight** (`facebook/audioseal`, MIT,
    /// coverage-audit-2026-08-03 Wave A permissive continuation) —
    /// Meta paired Generator + Detector 16-bit-message audio
    /// watermark for EU AI Act Article 50 compliance
    /// (2026-08-02 applies, San Roman et al. arXiv:2401.17264).
    /// Replaces the M5-05 config-only scaffold with a real weight-
    /// loading path; runtime binder remains gated on M5-05 T04 ADR
    /// ratification. Category = `watermark`. Convert with
    /// `convert_audioseal_real_weight_file`.
    AudiosealRealWeight,
    /// **Aratako/MioCodec-25Hz-44.1kHz-v2** (MIT,
    /// hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder complement
    /// wave, 2026-08-04). Category = `codec`. Single-safetensors
    /// JA-focused 25 Hz / 44.1 kHz multilingual speech codec (~132M
    /// F32 params, ~528 MB, `pipeline: audio-to-audio`, 11-language
    /// coverage `en / ja / nl / fr / de / it / pl / pt / es / ko / zh`,
    /// arXiv:2507.21138). Fine-tuned from `Aratako/MioCodec-25Hz-24kHz`
    /// on `sarulab-speech/mls_sidon` + `mythicinfinity/Libriheavy-HQ`
    /// + `nvidia/hifitts-2`. **Simplest of the batch** — mirrors the
    /// bicodec / neucodec / focalcodec / xcodec2 BF16 pass-through
    /// skeleton (single-file `model.safetensors` + `config.yaml`,
    /// no torch-pickle prepare step, no ONNX mirror). Every F32 / F16
    /// / BF16 tensor passes through verbatim under its upstream
    /// safetensors name; the runtime binding + real-weight parity are
    /// deferred to owner (`docs/license-audit.md` §3.1 sign-off).
    /// Distinct arch tag `miocodec` from every sibling codec (Mimi /
    /// DAC / WavTokenizer / xcodec2 / neucodec / bicodec / funcodec /
    /// speechtokenizer / focalcodec / xy_tokenizer / snac /
    /// step_audio2_mini) — silently sharing would mis-route the
    /// runtime dispatch (Aratako's own codec design, not RVQ / FSQ /
    /// SoundStream / focal-modulation family). Provenance = **MIT**
    /// (Permissive — verified 2026-08-04 via HF cardData API primary
    /// source). Convert with `convert_miocodec_file`. Complements the
    /// existing Kokoro / piper-plus JA vocoder stack.
    MioCodec,
    /// Neuphonic **NeuTTS Air** (`neuphonic/neutts-air`, apache-2.0,
    /// SoTA plan candidate wave 2026-08-04). Category = `tts`.
    /// Qwen2-family 0.5B LLM backbone (hidden_size=896, 24 layers,
    /// 14 attention heads, 2 KV heads, RoPE θ=1e6, vocab extended
    /// from 151,936 → 217,652 to carry the NeuCodec audio-token
    /// space per upstream `config.json` primary source 2026-08-04)
    /// fine-tuned to emit NeuCodec audio tokens after text tokens
    /// for **on-device instant voice cloning**. Single-file BF16
    /// safetensors (~1.40 GB / 747.9 M BF16 params, HF API primary-
    /// source verified 2026-08-04). Sibling codec = the already-
    /// published [`Self::Neucodec`] (`neuphonic/neucodec`, base
    /// variant). The upstream repo also ships a foreign
    /// `neutss-air-BF16.gguf` (~1.40 GB) that this converter
    /// deliberately does NOT process — FR-LD-05 forbids Vokra
    /// runtime from loading foreign GGUFs, so only the safetensors
    /// path is walked. Distinct arch tag `neutts-air` from every
    /// sibling TTS module (`kokoro` / `piper-plus` / `cosyvoice2` /
    /// `cosyvoice3` / `csm` / `moshi` / `voxcpm2` / `dia` / `zonos` /
    /// `chatterbox` / `bark` / `styletts2` / `vibevoice` /
    /// `qwen3-tts` / `sbv2` / `chattts` / `irodori` / `melotts` /
    /// `vits-ja` / `vieneu` / `parler`) — silently sharing would
    /// mis-route the runtime dispatch (FR-EX-08). BF16 pass-through
    /// skeleton mirror of miocodec / neucodec / bicodec /
    /// focalcodec / xcodec2 — every F32 / F16 / BF16 tensor passes
    /// through verbatim under its upstream safetensors name; the
    /// runtime binding + real-weight parity are deferred to owner
    /// sign-off (`docs/license-audit.md` §3.1) — the RMVPE /
    /// Charsiu / MOSS-Audio-Tokenizer / MioCodec loud-partial
    /// precedent. Provenance = **apache-2.0** (Permissive —
    /// verified 2026-08-04 via HF cardData API primary source).
    /// Convert with `convert_neutts_air_file`.
    NeuTtsAir,
    /// NVIDIA **Nemotron-Speech-Streaming v2603** (Apache-2.0, ~1.2–2
    /// GB) — cache-aware FastConformer streaming ASR (40+ lang).
    /// Convert with `convert_nemotron_speech_streaming_v2603_file`.
    NemotronSpeechStreamingV2603,
    /// SpeechBrain **spkrec-ecapa-voxceleb** (ECAPA-TDNN) speaker
    /// verification checkpoint (SoTA plan Phase 5 speaker fleet,
    /// 2026-07-28). Category = `speaker`. TDNN-based speaker
    /// embedding extractor. BF16 pass-through skeleton — every F32 /
    /// F16 / BF16 tensor passes through verbatim under its upstream
    /// safetensors name. Provenance = **apache-2.0** (Permissive).
    EcapaTdnn,
    /// Wespeaker **wespeaker-voxceleb-resnet34-LM** speaker
    /// verification checkpoint (SoTA plan Phase 5 speaker fleet,
    /// 2026-07-28). Category = `speaker`. ResNet-34 speaker embedding
    /// extractor with large-margin fine-tuning. BF16 pass-through
    /// skeleton — every F32 / F16 / BF16 tensor passes through
    /// verbatim under its upstream safetensors name. Provenance =
    /// **apache-2.0** (Permissive).
    Wespeaker,
    /// Alibaba IIC **speech_eres2net_sv_zh-cn_16k-common** (3D-Speaker
    /// ERes2Net) speaker verification checkpoint (SoTA plan Phase 5
    /// speaker fleet, 2026-07-28). Category = `speaker`. Enhanced
    /// Res2Net variant tuned on the 3D-Speaker Zh corpus. BF16 pass-
    /// through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim. Provenance = **apache-2.0** (Permissive).
    Speaker3d,
    /// NVIDIA **TitaNet-Large** speaker verification checkpoint (SoTA
    /// follow-on, 2026-07-30). Category = `speaker`. Depth-wise-
    /// separable Conv1D speaker-embedding extractor, 16 kHz mono →
    /// 192-d embedding, ~23 M params. Provenance = **cc-by-4.0**
    /// (`AttributionRequired` — the converter stamps the FR-MD-09
    /// attribution text; NOTICE §11 covers the code-level NVIDIA
    /// credit). Every F32 / F16 / BF16 tensor passes through verbatim
    /// under its upstream safetensors name (mirror of wespeaker /
    /// ecapa_tdnn / speaker_3d skeleton). The `.nemo` tarball is
    /// bridged offline through `tools/parity/nemo_pt_to_safetensors.py`;
    /// this converter accepts safetensors only. Runtime port is
    /// out-of-scope (M5-residual `titanet_speaker_encode`, FR-OP-80
    /// variant); consumers today should use CAM++ (`speaker_encode`)
    /// under Apache-2.0. Convert with [`convert_titanet_file`].
    TitaNet,
    /// FunAudioLLM **emotion2vec_plus_large** speech emotion
    /// recognition checkpoint (SoTA plan Phase 5 emotion fleet,
    /// 2026-07-28). Category = `emotion`. Emotion embedding extractor
    /// paired with the FunASR family. BF16 pass-through skeleton —
    /// every F32 / F16 / BF16 tensor passes through verbatim.
    /// Provenance = **MIT** (Permissive).
    Emotion2vec,
    /// **CREPE** (Kim et al. 2018) — a monophonic F0 (fundamental-
    /// frequency) extractor. Convert with [`convert_crepe_file`] — it
    /// needs the `capacity` + `hop` + `fmin` + `fmax` JSON side-car that
    /// `tools/parity/keras_h5_to_safetensors.py` emits alongside the
    /// flattened safetensors (the upstream `.h5` release is Keras /
    /// TensorFlow, which the zero-dep Rust converter deliberately does
    /// not parse — the same offline-prepare split as DAC / Kokoro /
    /// UTMOS). Weight license = **MIT**.
    Crepe,
    /// **pyannote/segmentation-3.0** (Bredin, CNRS — 2026-07-30
    /// license half unblock, `docs/license-audit.md` §3.1 row 263).
    /// Category = `vad`. PyanNet voice-activity-detection /
    /// speaker-segmentation backbone (SincNet → BiLSTM x2 → Linear x2
    /// → powerset multiclass classifier, 7 classes for 3 speakers ×
    /// 2 overlap). BF16 pass-through skeleton + `vokra.pyannote.*`
    /// hparam chunk group (SINCNET_DEFAULTS + LSTM_DEFAULTS +
    /// LINEAR_DEFAULTS transcribed from PyanNet.py primary source).
    /// Provenance = **MIT** (Permissive — HF cardData primary source
    /// verified 2026-07-30, `gated: auto` is access control only, no
    /// additional obligations). Runtime binder is Wave 2 loud-partial
    /// (weights bind, forward returns `VokraError::UnsupportedOp`
    /// until SincNet primitive lands Wave 3) —
    /// `docs/handoff/pyannote-implementation-plan-2026-07-30.md`. The
    /// `.bin` (torch pickle) → safetensors bridge lives offline in
    /// `tools/parity/bin_to_safetensors.py`; this converter accepts
    /// safetensors only. Convert with
    /// `convert_pyannote_segmentation_file`.
    PyannoteSegmentation,
    /// **pyannote/speaker-diarization-3.1** pipeline orchestration
    /// (Bredin, CNRS — 2026-08-01 Wave 5, `docs/license-audit.md`
    /// §3.1 sign-off row). Category = `diarize`. **Weightless
    /// pipeline** — composes the sibling MIT weight repos
    /// `pyannote/segmentation-3.0` (VAD backbone) +
    /// `pyannote/wespeaker-voxceleb-resnet34-LM` (speaker encoder)
    /// via `AgglomerativeClustering` (centroid linkage, cosine
    /// distance cut = 0.7045654963945799). The emitted GGUF carries
    /// only the `vokra.pyannote_pipeline.*` orchestration hparams
    /// (pipeline type / sub-model references / batch sizes /
    /// clustering knobs) transcribed verbatim from the upstream
    /// `config.yaml` (primary source verified 2026-08-01 —
    /// CLAUDE.md「ハルシネーション厳禁」); no tensors, no YAML
    /// parser in the runtime tree (NFR-DS-02). Provenance = MIT
    /// (Permissive — HF cardData primary source verified 2026-07-30,
    /// `gated: auto` = access control only, no additional
    /// obligations; row 268 in §3.1 signed yousan). Runtime pipeline
    /// dispatch (`crates/vokra-models/src/pyannote/pipeline.rs`) is
    /// a separate WP — this variant only stamps orchestration
    /// metadata. Convert with
    /// `convert_pyannote_speaker_diarization_3_1_file`.
    PyannoteSpeakerDiarization31,
    // ---------------------------------------------------------------------------
    // TIER 1+2 audio-gap implementation (2026-07-30 ultracode `wf_022575ce-077`)
    // — 40 new ModelKind variants across 7 WT batches. Each is a BF16 pass-
    // through converter (models/*.rs); the dispatch wiring lives in the
    // `convert_file` match below + `from_arg` / `as_arg` above. License class
    // registrations live in `crates/vokra-core/src/compliance/license_class.rs`;
    // §3.1 sign-off rows in `docs/license-audit.md` (2026-07-30 yousan CC 判断).
    // Handoff: `docs/handoff/tier1-tier2-audio-impl-2026-07-30.md`.
    // ---------------------------------------------------------------------------
    /// Alibaba **Qwen3-ASR** family safetensors checkpoint (SoTA plan
    /// Phase 5 ASR fleet, 2026-07-30). Category = `asr`. Two sizes
    /// share arch / category / provenance stamps: `Qwen/Qwen3-ASR-0.6B`
    /// (audio encoder 18 × d=896 × 14h × ffn=3584 + Qwen3 text decoder
    /// 28 × d=1024 × 16Q ÷ 8KV × head_dim=128 × ffn=3072) and
    /// `Qwen/Qwen3-ASR-1.7B` (audio encoder 24 × d=1024 × 16h ×
    /// ffn=4096 + Qwen3 text decoder 28 × d=2048 × 16Q ÷ 8KV ×
    /// head_dim=128 × ffn=6144). Both are BF16 (`dtype=bfloat16` in
    /// `config.json`) — the pass-through arm handles the release
    /// checkpoints directly. Every hparam is transcribed verbatim
    /// from `huggingface.co/Qwen/Qwen3-ASR-{0.6B,1.7B}/raw/main/
    /// config.json` (CLAUDE.md「ハルシネーション厳禁」, fetched
    /// 2026-07-30). Provenance = **apache-2.0** (Permissive) per both
    /// HF model cards' `cardData.license` (CC-verified via HF API
    /// 2026-07-30). The `--model qwen3-asr-0.6b` / `-1.7b` slug picks
    /// the `models::qwen3_asr::Variant`; the bare `qwen3-asr` slug
    /// routes to the 1.7B (flagship) default.
    Qwen3Asr,
    /// **wav2vec 2.0 CTC** family safetensors checkpoint (SoTA plan
    /// Phase 5 ASR fleet, 2026-07-30). Category = `asr`. Four
    /// canonical variants share the 7-layer Conv1D feature-extractor
    /// topology (`conv_dim=[512×7]`, `conv_kernel=[10,3,3,3,3,2,2]`,
    /// `conv_stride=[5,2,2,2,2,2,2]`) at 320× total downsampling:
    /// - `facebook/wav2vec2-base-960h` (95M, base topology 12 × d=768
    ///   × 12h × ffn=3072, `feat_extract_norm="group"`,
    ///   `do_stable_layer_norm=false`, English LibriSpeech CTC head
    ///   `vocab_size=32`),
    /// - `facebook/wav2vec2-large-xlsr-53` (300M, large topology 24 ×
    ///   d=1024 × 16h × ffn=4096, `feat_extract_norm="layer"`,
    ///   `do_stable_layer_norm=true`, `Wav2Vec2ForPreTraining` — no
    ///   CTC head — reused as encoder base),
    /// - `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` (large
    ///   topology + CTC head `vocab_size=2341`),
    /// - `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` (large
    ///   topology + CTC head `vocab_size=3503`),
    /// - `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (large topology +
    ///   CTC head `vocab_size=392` — **eSpeak IPA phoneme** vocabulary,
    ///   arXiv:2109.11680, CommonVoice fine-tune; complementary to the
    ///   char / kana+kanji / hanzi rows above).
    ///
    /// Every hparam is transcribed verbatim from the primary-source
    /// `config.json` per variant (CLAUDE.md「ハルシネーション厳禁」,
    /// fetched 2026-07-30, espeak-cv-ft 2026-08-01). All five ship
    /// **apache-2.0** (Permissive) per the HF API `cardData.license`
    /// (CC-verified). The `--model wav2vec2-base-960h` /
    /// `wav2vec2-large-xlsr-53` / `wav2vec2-large-xlsr-53-japanese` /
    /// `wav2vec2-large-xlsr-53-chinese-zh-cn` /
    /// `wav2vec2-xlsr-53-espeak-cv-ft` slugs pick the
    /// `models::wav2vec2_ctc::Variant`; the bare `wav2vec2` slug
    /// routes to `base-960h` (the smallest / most widely-used release).
    Wav2Vec2Ctc,
    /// **data2vec-audio** (`facebook/data2vec-audio-base-960h`,
    /// apache-2.0) — Baevski et al. 2022, arXiv:2202.03555. A sibling
    /// release of the wav2vec 2.0 CTC family: data2vec-audio shares
    /// the wav2vec 2.0 **base** downstream inference topology
    /// (12 × d=768 × 12h × ffn=3072, `feat_extract_norm="group"`,
    /// `do_stable_layer_norm=false`), the same 7-layer Conv1D
    /// feature-extractor, and the LibriSpeech 960h English char CTC
    /// head (`vocab_size=32`). The tensor names are **identical** to
    /// `wav2vec2-base-960h`, so the [`Self::Wav2Vec2Ctc`] converter
    /// covers it verbatim; the only divergence is the **pretraining
    /// objective** (contextualised latent representation prediction
    /// with an EMA teacher), which does not affect downstream
    /// inference. A distinct `ModelKind` (rather than a slug-only
    /// alias of `Wav2Vec2Ctc`) is used so `vokra.model.name` +
    /// `vokra.provenance.upstream_hf` faithfully report the data2vec-
    /// audio upstream release instead of masquerading as
    /// `wav2vec2-base-960h`. Category `asr`, license Permissive
    /// (apache-2.0 per HF `cardData.license` CC-verified 2026-08-02).
    /// The `--model data2vec-audio-base` / `data2vec-audio-base-960h`
    /// slugs pick this arm; convert dispatch routes it through
    /// `models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant`
    /// with the sibling `Variant::Data2vecAudioBase960h` (correct
    /// provenance stamp on top of the shared topology axes).
    Data2vecAudioBase,
    /// Meta **MMS-1B-All** (`facebook/mms-1b-all`, cc-by-nc-4.0) —
    /// Massively Multilingual Speech (Pratap et al. 2023,
    /// arXiv:2305.13516). A 1B-parameter wav2vec 2.0 backbone
    /// (`Wav2Vec2ForCTC` head family) bundled with 1000+ per-language
    /// CTC adapters (~2000 sibling files in the repo); base tensor path
    /// reuses the [`Self::Wav2Vec2Ctc`] converter (parent workflow
    /// REUSE HINT) via a dedicated
    /// `models::wav2vec2_ctc::Variant::Mms1bAll` arm.
    ///
    /// **Placeholder axes** — the parent workflow's SIZE NOTE (4.00 GB
    /// checkpoint) forbids downloading `config.json` for
    /// primary-source transcription, so the axes route to the closest-
    /// family sibling (`LargeXlsr53Base`: 24 × d=1024 × 16h ×
    /// ffn=4096, `feat_extract_norm="layer"`,
    /// `do_stable_layer_norm=true`) as a placeholder while the
    /// discriminating `name = "mms-1b-all"` +
    /// `upstream_hf = "facebook/mms-1b-all"` stamps stay faithful for
    /// provenance readback + placeholder-axis refusal at load time
    /// (M2-13 runtime gate). A follow-up wave must transcribe the true
    /// MMS-1B topology + land a per-language adapter loader before any
    /// downstream loader can trust the emitted hparams.
    ///
    /// Weight-distribution licence = **cc-by-nc-4.0** (T4 tier /
    /// Research-only publish path per the X-Codec-2 (2026-07-28)
    /// precedent) — the M2-13 runtime gate refuses to load in
    /// commercial mode (`requires_research_flag = true`); publish
    /// requires `publish-one.sh --allow-noncommercial`. Category `asr`
    /// (recorded under `vokra.model.category`).
    Mms1bAll,
    /// OpenMOSS Team **MOSS-TTS** base checkpoint (SoTA follow-on,
    /// added 2026-07-30) — `OpenMOSS-Team/MOSS-TTS`. Category `tts`.
    /// LM-based multilingual TTS: `model_type = "moss_tts_delay"`,
    /// Qwen3-8B backbone (hidden=4096 / ffn=12288 / 36 layers / GQA
    /// 32Q ÷ 8KV / head_dim=128 / vocab=155_648 / RoPE θ=1e6 /
    /// RMSNorm ε=1e-6) + 32 parallel audio codebook streams
    /// (`n_vq=32`, `audio_vocab_size=1024`) at 24 kHz output.
    /// Primary source: `huggingface.co/OpenMOSS-Team/MOSS-TTS/raw/main/config.json`
    /// fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」.
    /// Provenance = **apache-2.0** (Permissive). Ships **~17 GB BF16**
    /// across 4 safetensors shards, so **vast.ai is required** (memory
    /// `[[feedback-large-models-on-vast-ai]]`). Convert with
    /// `convert_moss_tts_file` (variant selector
    /// `models::moss_tts::MossTtsVariant::Delay`).
    MossTts,
    /// OpenMOSS Team **MOSS-TTS-v1.5** sibling of [`Self::MossTts`]
    /// (`OpenMOSS-Team/MOSS-TTS-v1.5`, apache-2.0, added 2026-07-30).
    /// Shares identical Delay axes and BF16 posture with the base
    /// release; language coverage widens (adds Cantonese `yue` +
    /// Arabic `ar` + Czech `cs` + Danish `da` per the model-card tags
    /// fetched 2026-07-30). Every axis matches [`Self::MossTts`];
    /// only the `vokra.model.name` stamp and the
    /// `vokra.provenance.upstream_hf` slug differ. Ships **~17 GB
    /// BF16** across 4 safetensors shards, so **vast.ai is required**.
    MossTtsV15,
    /// OpenMOSS Team **MOSS-TTS-Nano-100M** checkpoint (added
    /// 2026-07-30) — `OpenMOSS-Team/MOSS-TTS-Nano-100M`. Category
    /// `tts`. Compact `model_type = "moss_tts_nano"` variant with a
    /// GPT-2 flavour backbone (hidden=768 / 12 layers / 12 MHA heads
    /// / head_dim=64 / vocab=16_384; `n_positions=32_768`) + 16
    /// parallel audio codebook streams (`n_vq=16`,
    /// `audio_vocab_size=1024`) at 48 kHz output
    /// (`audio_tokenizer_sample_rate`). Primary source:
    /// `huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M/raw/main/config.json`
    /// fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」.
    /// Provenance = **apache-2.0** (Permissive). Ships as a torch
    /// pickle `pytorch_model.bin` (not safetensors) — callers
    /// pre-bridge with `tools/parity/bin_to_safetensors.py` (the
    /// OpenBMB VoxCPM precedent, `docs/license-audit.md` row 281).
    /// The RoPE / RMSNorm hparam keys carry `0` sentinels (GPT-2 uses
    /// learned positional embeddings + LayerNorm) so the runtime
    /// binder can tell "not applicable" apart from a silent default
    /// (FR-EX-08).
    MossTtsNano,
    /// OpenMOSS Team **MOSS-TTS-Local-Transformer-v1.5** checkpoint
    /// (added 2026-07-30) —
    /// `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`. Category
    /// `tts`. Mid-scale `model_type = "moss_tts_local"` variant with a
    /// Qwen3-flavour 2.5B backbone (hidden=2560 / ffn=9728 / 36
    /// layers / GQA 32Q ÷ 8KV / head_dim=128 / vocab=151_936 / RoPE
    /// θ=1e6 / RMSNorm ε=1e-6) plus a GPT-2 local head +
    /// 12 parallel audio codebook streams (`n_vq=12`,
    /// `audio_vocab_size=1024`) at 48 kHz output. Primary source:
    /// `huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5/raw/main/config.json`
    /// fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」.
    /// Provenance = **apache-2.0** (Permissive). Ships **~9 GB BF16**
    /// as a single safetensors; **vast.ai is required** (borderline
    /// on M1 iMac 16 GB — too tight for the whole-file
    /// `std::fs::read` path).
    MossTtsLocal,
    /// OpenMOSS Team **MOSS-Audio-4B-Instruct** checkpoint
    /// (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0, added
    /// 2026-08-02). Category `s2s` (audio-LLM — matches the sibling
    /// `kimi-audio` / `baichuan-audio` / `step-audio2-mini`
    /// classification rather than the four `tts` `moss_tts_*`
    /// variants). Custom-code release
    /// (`configuration_moss_audio.py`, `trust_remote_code=True`)
    /// distinct from the four `moss_tts_{delay,nano,local}` sibling
    /// releases in size (4B vs 8B / 100M / 2.5B) and topology
    /// (audio-LLM `configuration_moss_audio.py` custom module).
    /// Ships as **3 shards ~8 GB BF16** per the parent task manifest
    /// (2026-08-02); large enough that a downloading conversion
    /// borderlines M1 iMac 16 GB — publish via vast.ai (memory
    /// `[[feedback-large-models-on-vast-ai]]`). Provenance =
    /// **apache-2.0** (Permissive).
    ///
    /// **Reuses the `models::moss_tts` converter** per the parent
    /// workflow's REUSE HINT rather than a fresh `models/*.rs` module,
    /// dispatching through the new `models::moss_tts::MossTtsVariant::AudioInstruct4b`
    /// arm. That arm inherits the sibling **Local** (Qwen3-flavour 2.5B)
    /// axes as a **placeholder** while the code-only task discipline
    /// forbids downloading `configuration_moss_audio.py` for primary-source
    /// hparam transcription; the emitted GGUF stamps a distinct
    /// `vokra.moss_tts.variant = "audio_4b"` sub-arch tag so a runtime
    /// dispatcher can recognise this artifact and refuse to bind the
    /// placeholder axes until the follow-up wave lands the true axes.
    /// The **provenance triple** (NAME +
    /// `vokra.provenance.upstream_hf` + license) is faithful.
    MossAudio4bInstruct,
    /// OpenMOSS Team **MOSS-Audio-8B-Instruct** checkpoint
    /// (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`, apache-2.0, added
    /// 2026-08-02). Category `s2s` (audio-LLM — same
    /// `configuration_moss_audio.py` custom module as
    /// [`Self::MossAudio4bInstruct`], larger 8 B backbone). Ships as
    /// **4 shards ~9.05 GB BF16** per the parent task manifest
    /// (2026-08-02); the whole-file `std::fs::read` conversion path
    /// exceeds M1 iMac 16 GB comfortably — publish via vast.ai (memory
    /// `[[feedback-large-models-on-vast-ai]]`). Provenance =
    /// **apache-2.0** (Permissive).
    ///
    /// **Reuses the `models::moss_tts` converter** per the parent
    /// workflow's REUSE HINT (mirroring the sibling 4B arm) rather
    /// than a fresh `models/*.rs` module, dispatching through the new
    /// `models::moss_tts::MossTtsVariant::AudioInstruct8b` arm. That
    /// arm inherits the sibling **Local** (Qwen3-flavour 2.5B) axes as
    /// a **placeholder** while the code-only task discipline forbids
    /// downloading `configuration_moss_audio.py` for primary-source
    /// hparam transcription; the emitted GGUF stamps a distinct
    /// `vokra.moss_tts.variant = "audio_8b"` sub-arch tag so a runtime
    /// dispatcher can recognise this artifact and refuse to bind the
    /// placeholder axes until the follow-up wave lands the true axes.
    /// The **provenance triple** (NAME +
    /// `vokra.provenance.upstream_hf` + license) is faithful.
    MossAudio8bInstruct,
    /// **MeloTTS-English** (`myshell-ai/MeloTTS-English`, MIT).
    /// Implementer C wave 2026-07-30. VITS2-family multilingual TTS
    /// with a modified duration predictor. Category = `tts`. See
    /// `convert_melotts_file` + `crate::models::melotts::MeloVariant`
    /// — one converter serves the 3 language variants; each pins its
    /// language-specific axes (`n_symbols` / `num_tones` /
    /// `num_languages` / `n_speakers_active`) as compile-time constants.
    MeloTtsEnglish,
    /// **MeloTTS-Chinese** (`myshell-ai/MeloTTS-Chinese`, MIT).
    /// Implementer C wave 2026-07-30. Same VITS2 backbone as
    /// [`Self::MeloTtsEnglish`]; distinct only in `n_symbols = 112`,
    /// `num_tones = 11`, `spk2id = {ZH:1}`.
    MeloTtsChinese,
    /// **MeloTTS-Korean** (`myshell-ai/MeloTTS-Korean`, MIT).
    /// Implementer C wave 2026-07-30. Same VITS2 backbone; distinct in
    /// `n_symbols = 219`, `num_tones = 16`, `num_languages = 10`,
    /// `spk2id = {KR:0}`.
    MeloTtsKorean,
    /// **MeloTTS-Spanish** (`myshell-ai/MeloTTS-Spanish`, MIT). Wave 8
    /// residual 2026-08-01. Same VITS2 backbone as sibling language
    /// variants; language-specific axes via `MeloVariant::Spanish`.
    MeloTtsSpanish,
    /// **MeloTTS-Japanese** (`myshell-ai/MeloTTS-Japanese`, MIT). Wave 8
    /// residual 2026-08-01. Same VITS2 backbone as sibling language
    /// variants; language-specific axes via `MeloVariant::Japanese`.
    MeloTtsJapanese,
    /// **SpeechT5 TTS** (`microsoft/speecht5_tts`, MIT). Implementer C
    /// wave 2026-07-30. Microsoft's unified encoder-decoder pre-training
    /// TTS head (12-layer encoder / 6-layer decoder × 12 heads × 768
    /// hidden × 3072 FFN) + speech-decoder prenet / postnet + 512-d
    /// speaker x-vector conditioning. Category = `tts`. See
    /// `convert_speecht5_file`.
    ///
    /// The sibling `microsoft/speecht5_vc` (voice-conversion) is
    /// deliberately out of scope — voice-cloning targets are
    /// `vokra-voiceclone-experimental` (CLAUDE.md 設計判断 8).
    SpeechT5Tts,
    /// **Parler-TTS mini-multilingual** (`parler-tts/parler-tts-mini-
    /// multilingual-v1.1`, apache-2.0). Implementer C wave 2026-07-30.
    /// Decoder-only Parler LM (24-layer × 1024d MHA over 9 DAC
    /// codebooks) + T5 text encoder (24-layer × 1024d × 16h × 2816
    /// FFN) conditioned on a natural-language voice description.
    /// Category = `tts`. See `convert_parler_file` +
    /// `crate::models::parler::ParlerVariant` — one converter serves
    /// both the multilingual base and the Indic fine-tune (they share
    /// the tensor topology).
    ParlerTtsMiniMultilingual,
    /// **Indic Parler-TTS** (`ai4bharat/indic-parler-tts`, apache-2.0,
    /// gated=auto). Implementer C wave 2026-07-30. Same architecture as
    /// [`Self::ParlerTtsMiniMultilingual`]; a fine-tune on 21 Indic
    /// languages. The `gated=auto` HF flag is access control — the
    /// license itself is apache-2.0 per the card front-matter.
    IndicParlerTts,
    /// **Parler-TTS mini-v1** (`parler-tts/parler-tts-mini-v1`,
    /// apache-2.0). Wave 4 land 2026-08-01. The original English-only
    /// Mini release (predecessor of the multilingual v1.1 variant).
    /// Same tensor topology as [`Self::ParlerTtsMiniMultilingual`]
    /// end-to-end except the top-level `vocab_size = 32128` (T5 text
    /// vocabulary only, no audio-code alphabet merged in) vs the
    /// multilingual's 90714. Every T5 / decoder / audio-encoder hparam
    /// is unchanged. Primary source verified 2026-08-01 from
    /// `huggingface.co/parler-tts/parler-tts-mini-v1/raw/main/config.json`
    /// — CLAUDE.md「ハルシネーション厳禁」. See `convert_parler_file`
    /// + `crate::models::parler::ParlerVariant::MiniV1English` — the
    /// single `crate::models::parler::convert_parler_file` converter
    /// dispatches the three variants; only `vocab_size` differs on this
    /// arm. Category = `tts`. ~3.5 GB single safetensors (M1 iMac 16 GB
    /// でローカル変換 safe per memory
    /// `[[feedback-large-models-on-vast-ai]]` ≥8 GB threshold 下、
    /// vast.ai 不要).
    ParlerTtsMiniV1English,
    /// **VieNeu-TTS-v3-Turbo** (`pnnbao-ump/VieNeu-TTS-v3-Turbo`,
    /// apache-2.0). Implementer C wave 2026-07-30. Novel hierarchical
    /// AR Vietnamese TTS (`architectures = ["VieNeuV3TurboForTTS"]`) —
    /// **NOT** a VITS / StyleTTS / Piper fork. LLM-family backbone
    /// (12L × 12H × 768 hidden, GQA 4 KV, RoPE θ=10000, RMSNorm
    /// ε=1e-6, max_pos=1024) + 2-layer local acoustic decoder (learned
    /// slot-position embedding, NOT RoPE) over external
    /// `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` codec (16 quantizers ×
    /// 1024 vocab @ 48 kHz). Category = `tts`. See
    /// `convert_vieneu_file`.
    VieNeuTts,
    /// **Bark (Suno)** family (`suno/bark` + `suno/bark-small`, MIT).
    /// Implementer C wave 2026-07-30. Hierarchical AR LM TTS in 3
    /// transformer stages (semantic → coarse EnCodec → fine EnCodec)
    /// over an external `facebook/encodec_24khz` vocoder
    /// (CC-BY-NC 4.0 — the M2-13 codec-side gate fires there, not on
    /// the Bark GGUF). Category = `tts`. See `convert_bark_file` +
    /// `crate::models::bark::BarkVariant` — one variant per
    /// per-stage `num_layers` (small=12, full=24).
    Bark,
    /// **Bark-small** (`suno/bark-small`, MIT) explicit variant.
    /// Implementer C wave 2026-07-30. Distinct dispatch arm from
    /// [`Self::Bark`] so the CLI slug + verify-time introspection
    /// surface the exact release. Runs the same
    /// `convert_bark_file` with `crate::models::bark::BarkVariant::Small`.
    BarkSmall,
    /// SpeechBrain **tts-hifigan-libritts-22050Hz** vocoder checkpoint
    /// (SoTA plan Phase D1, 2026-07-30). Category = `vocoder`. HiFi-GAN
    /// generator (Kong et al. 2020, arXiv:2010.05646) trained on
    /// LibriTTS at 22 050 Hz. Ships torch-pickle generator.ckpt +
    /// hyperparams.yaml; callers pre-flatten to safetensors offline via
    /// `tools/parity/hifigan_prepare_checkpoint.py`. BF16 pass-through
    /// skeleton — every F32 / F16 / BF16 tensor passes through
    /// verbatim under its upstream safetensors name; runtime binding +
    /// real-weight parity are deferred to owner
    /// (`docs/license-audit.md` §3.1 sign-off queue). Distinct arch
    /// tag from `bigvgan` (leaky_relu vs snake activation, no
    /// alias-free activation wrapper) and from `piper-plus`
    /// (standalone vocoder, not full TTS). Provenance = **apache-2.0**
    /// (Permissive — verified 2026-07-30 via HF API cardData).
    HifiganVocoder,
    /// Microsoft **SpeechT5 HiFi-GAN vocoder** (`microsoft/speecht5_hifigan`,
    /// MIT) — 2026-07-31 wave. Category = `vocoder`. HiFi-GAN vocoder
    /// companion to `microsoft/speecht5_tts` (Kong et al. 2020,
    /// arXiv:2010.05646 lineage), trained on LibriTTS at 16 kHz with
    /// 80-band mel input, upsample rates `[4, 4, 4, 4]` (total 256×),
    /// upsample kernels `[8, 8, 8, 8]`, MRF kernels `[3, 7, 11]` with
    /// dilations `[[1,3,5], [1,3,5], [1,3,5]]`, and
    /// `normalize_before: true` with learned scalar-per-mel-bin `mean` /
    /// `scale` tensors. Ships torch-pickle `pytorch_model.bin` +
    /// `config.json` only (no safetensors mirror on the primary release,
    /// verified 2026-07-31 via HF cardData API); callers pre-flatten to
    /// safetensors offline via
    /// `tools/parity/speecht5_hifigan_prepare_checkpoint.py` — a thin
    /// wrapper over the shared `bin_to_safetensors.py` bridge. BF16
    /// pass-through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim under its upstream HF-transformers
    /// `SpeechT5HifiGan` name (`conv_pre.*`, `upsampler.{i}.*`,
    /// `resblocks.{i}.convs1.{j}.*`, `resblocks.{i}.convs2.{j}.*`,
    /// `conv_post.*`, `mean`, `scale`); runtime binding + real-weight
    /// parity are deferred to owner (`docs/license-audit.md` §3.1
    /// sign-off queue). **Distinct arch tag** from `hifigan_vocoder`
    /// (SpeechBrain `tts-hifigan-libritts-22050Hz`, 22 050 Hz / no
    /// `normalize_before` / different tensor prefix); silently sharing
    /// a tag would mis-route runtime dispatch. Provenance = **mit**
    /// (Permissive — verified 2026-07-31 via HF cardData API
    /// `license: mit`). The sibling `microsoft/speecht5_vc`
    /// (voice-conversion) is deliberately out of scope — voice-cloning
    /// targets are `vokra-voiceclone-experimental` (CLAUDE.md 設計判断 8).
    Speecht5Hifigan,
    /// NVIDIA **BigVGAN** vocoder family (SoTA plan Phase D2-D5,
    /// 2026-07-30). Category = `vocoder`. AMPBlock1 with Snake or
    /// SnakeBeta plus transposed-conv upsample vocoder
    /// (arXiv:2206.04658). Four variants share this single
    /// [`ModelKind`], distinguished by a `bigvgan::BigVGanVariant`
    /// discriminator emitted under `vokra.bigvgan.variant`:
    /// `v2_22khz_80band_256x` (D2), `v2_44khz_128band_512x` (D3),
    /// `v2_24khz_100band_256x` (D4), `base_v1_24khz_100band` (D5,
    /// v1 base). All four ship torch-pickle
    /// (`bigvgan_generator.pt` alongside `config.json`); callers
    /// pre-flatten to safetensors offline via
    /// `tools/parity/bigvgan_prepare_checkpoint.py`. BF16 pass-through
    /// skeleton — every F32 / F16 / BF16 tensor passes through
    /// verbatim under its upstream safetensors name; runtime binding
    /// and real-weight parity are deferred to owner. Distinct arch
    /// tag from `hifigan_vocoder` (snake vs leaky_relu activation
    /// plus alias-free activation wrapper presence). Provenance =
    /// **MIT** (Permissive — verified 2026-07-30 via HF API
    /// cardData; GitHub NVIDIA/BigVGAN LICENSE is standard MIT
    /// `Copyright (c) 2024 NVIDIA CORPORATION`, CLAUDE.md 2026-07-22
    /// 訂正).
    BigVGan,
    /// **FocalCodec** (`lucadellalib/focalcodec_{50hz,25hz,12_5hz}`,
    /// apache-2.0) safetensors checkpoint (SoTA plan Phase D6,
    /// 2026-07-30; 25Hz / 12.5Hz variants added 2026-07-31).
    /// Category = `codec`. Focal-modulation-based single-codebook
    /// low-bitrate audio codec at 50 / 25 / 12.5 Hz (arXiv:2502.04465).
    /// **Unlike the sibling BigVGAN / HiFi-GAN vocoders**, FocalCodec
    /// ships `model.safetensors` + `config.json` directly (no
    /// torch-pickle prepare step). BF16 pass-through skeleton — every
    /// F32 / F16 / BF16 tensor passes through verbatim under its
    /// upstream safetensors name; runtime binding + real-weight parity
    /// are deferred to owner (`docs/license-audit.md` §3.1 sign-off
    /// queue). Distinct arch tag from every sibling codec
    /// (Mimi / DAC / WavTokenizer / neucodec / step_audio2_mini /
    /// X-Codec 2 / FunCodec / SpeechTokenizer / bicodec / XyTokenizer)
    /// because FocalCodec is neither RVQ nor FSQ nor SoundStream
    /// family. All three variants collapse into this single ModelKind;
    /// [`convert_file_with_slug`] picks the correct
    /// `models::focalcodec::FocalcodecVariant` from the raw `--model`
    /// slug (mirror of [`Self::BigVGan`]). Provenance = **apache-2.0**
    /// for every variant (Permissive — verified 2026-07-30 /
    /// 2026-07-31 via HF API cardData; base `microsoft/wavlm-large`
    /// is MIT, both compatible under `Permissive`).
    Focalcodec,
    /// **JusperLee/TIGER-DnR** (Implementer E TIER 1, 2026-07-30).
    /// Category = `enhancement`. TIGER = Time-frequency Interleaved
    /// Gain Extraction from a Restructured net — dialog / narration /
    /// SFX cinematic source separation trained on the DnR benchmark.
    /// Shares the `models::tiger::ARCH` tag `tiger_separator` +
    /// converter with the `TigerSpeech` sibling; the two differ only
    /// in training data + `vokra.tiger.variant` / `vokra.model.name` /
    /// `vokra.provenance.upstream_hf` stamps. Every F32 / F16 / BF16
    /// tensor passes through verbatim; the internal Time-Frequency
    /// dual-path body is a `loud-partial` follow-up (real-weight
    /// forward is deferred). Provenance = **apache-2.0** (Permissive
    /// — per HF model-card `cardData.license`).
    TigerSeparator,
    /// **JusperLee/TIGER-speech** (Implementer E TIER 1, 2026-07-30).
    /// Category = `enhancement`. Speaker separation on speech mixtures
    /// — same architecture as [`Self::TigerSeparator`], different
    /// training data + head count. Both variants route to
    /// `models::tiger::convert_tiger_file` with distinct
    /// `models::tiger::TigerVariant` arguments. Provenance =
    /// **apache-2.0** (Permissive).
    TigerSpeech,
    /// **JacobLinCool/MP-SENet-DNS** (Implementer E TIER 1, 2026-07-30).
    /// Category = `denoise`. MP-SENet = dual-branch (magnitude +
    /// phase) U-Net speech enhancement (arXiv:2305.13686 lineage —
    /// the JacobLinCool DNS-tuned re-release of `yxlu0057/MP-SENet`).
    /// Every F32 / F16 / BF16 tensor passes through verbatim; the
    /// internal dual-branch forward is a `loud-partial` follow-up.
    /// Provenance = **MIT** (Permissive — inherits the base
    /// `yxlu0057/MP-SENet` MIT LICENSE).
    MpSenet,
    /// **JacobLinCool/MP-SENet-DNS** (Implementer E TIER 1, 2026-08-02).
    /// Category = `denoise`. DNS-tuned re-release of the same MP-SENet
    /// architecture as [`Self::MpSenet`] (magnitude + phase parallel
    /// STFT enhancement network, arXiv:2305.13686 lineage). This is a
    /// distinct [`ModelKind`] arm so §3.1 sign-off + publish-side
    /// provenance stamp can be tracked per HF repo — the underlying
    /// converter is `models::mp_senet::convert_mp_senet_file`, shared
    /// verbatim with the sibling arm (the safetensors surface + arch
    /// tag are identical; only the `vokra.provenance.upstream_hf` slug
    /// and the sign-off row diverge). Every F32 / F16 / BF16 tensor
    /// passes through verbatim; the internal dual-branch forward is a
    /// `loud-partial` follow-up. Provenance = **MIT** (Permissive —
    /// inherits the base `yxlu0057/MP-SENet` MIT LICENSE).
    MpSenetDns,
    /// **speechbrain/metricgan-plus-voicebank** (Implementer E TIER 1,
    /// 2026-07-30). Category = `enhancement`. MetricGAN+ =
    /// generator-only speech-enhancement GAN optimising perceptual
    /// metrics (PESQ). Every F32 / F16 / BF16 tensor passes through
    /// verbatim; the internal LSTM-stack + spectral-mask head is a
    /// `loud-partial` follow-up. Provenance = **apache-2.0**
    /// (Permissive — SpeechBrain end-to-end Apache-2.0 LICENSE).
    MetricganPlus,
    /// **speechbrain/sepformer-wsj02mix** (Implementer E TIER 1,
    /// 2026-07-30). Category = `separation` (2-speaker source
    /// separation task — distinct from the enhancement-category
    /// SepFormer siblings). SepFormer = Transformer-based dual-path
    /// separator (Subakan et al. 2021). Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// the `SepformerWham16kEnh` / `SepformerWhamr16k` siblings; the
    /// three differ only in training data + head count +
    /// `vokra.sepformer.variant` / `vokra.model.category` /
    /// `vokra.model.name` / `vokra.provenance.upstream_hf` stamps.
    /// Every F32 / F16 / BF16 tensor passes through verbatim; the
    /// internal dual-path Transformer body is a `loud-partial`
    /// follow-up. Provenance = **apache-2.0** (Permissive).
    SepFormer,
    /// **speechbrain/sepformer-wham16k-enhancement** (Implementer E
    /// TIER 1, 2026-07-30). Category = `enhancement`. Single-speaker
    /// speech enhancement on WHAM! 16 kHz. Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// [`Self::SepFormer`] / [`Self::SepformerWhamr16k`]. Provenance =
    /// **apache-2.0** (Permissive).
    SepformerWham16kEnh,
    /// **speechbrain/sepformer-whamr16k** (Implementer E TIER 1,
    /// 2026-07-30). Category = `enhancement`. Joint dereverb +
    /// denoise on WHAMR! 16 kHz. Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// [`Self::SepFormer`] / [`Self::SepformerWham16kEnh`]. Provenance
    /// = **apache-2.0** (Permissive).
    SepformerWhamr16k,
    /// **speechbrain/sepformer-libri2mix** (Wave 4 candidate,
    /// 2026-08-01). Category = `separation` (2-speaker source
    /// separation — same head as [`Self::SepFormer`], differs only in
    /// the training corpus: LibriMix is a LibriSpeech-derived
    /// CC-BY-4.0 mixture set, WSJ0-2mix is proprietary WSJ0-derived).
    /// Shares the `models::sepformer::ARCH` tag `sepformer` +
    /// converter with [`Self::SepFormer`] / [`Self::SepformerWham16kEnh`]
    /// / [`Self::SepformerWhamr16k`]; the distinct ModelKind ensures
    /// the artifact does NOT silently inherit the Wsj02mix sibling's
    /// `vokra.model.name` / `vokra.provenance.upstream_hf` /
    /// `vokra.sepformer.variant` stamps. Every F32 / F16 / BF16 tensor
    /// passes through verbatim; the internal dual-path Transformer
    /// body is a `loud-partial` follow-up (same posture as the
    /// SepFormer siblings). Provenance = **apache-2.0** (Permissive
    /// — SpeechBrain end-to-end Apache-2.0; the LibriMix training
    /// corpus itself is CC-BY-4.0 but that is a corpus-level
    /// attribution obligation, not a weight-license restriction).
    SepformerLibri2Mix,
    /// **speechbrain/sepformer-libri3mix** (Wave 4 candidate,
    /// 2026-08-01). Category = `separation` (**3-speaker cocktail-
    /// party** source separation on LibriMix — same LibriSpeech-derived
    /// corpus family as [`Self::SepformerLibri2Mix`], and the same
    /// SepFormer topology as every sibling here; the sole difference
    /// is the masker output head branches into **3 parallel speaker
    /// streams instead of 2**). Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// [`Self::SepFormer`] / [`Self::SepformerWham16kEnh`] /
    /// [`Self::SepformerWhamr16k`] / [`Self::SepformerLibri2Mix`] /
    /// [`Self::SepformerWhamr8k`]; the distinct ModelKind ensures the
    /// artifact does NOT silently inherit the 2-speaker sibling's
    /// `vokra.model.name` / `vokra.provenance.upstream_hf` /
    /// `vokra.sepformer.variant` = wrong CDN attribution +
    /// `vokra.sepformer.n_out` = wrong binder output-stream axis (the
    /// new `vokra.sepformer.n_out = 3` chunk added the same wave makes
    /// this explicit at load time). Every F32 / F16 / BF16 tensor
    /// passes through verbatim; the internal dual-path Transformer
    /// body is a `loud-partial` follow-up (same posture as the
    /// SepFormer siblings). Provenance = **apache-2.0** (Permissive
    /// — SpeechBrain end-to-end Apache-2.0; the LibriMix training
    /// corpus itself is CC-BY-4.0 but that is a corpus-level
    /// attribution obligation, not a weight-license restriction —
    /// identical posture to the sibling `libri2mix` row).
    SepformerLibri3Mix,
    /// **speechbrain/sepformer-whamr** (Wave 4 candidate, 2026-08-01).
    /// Category = `enhancement`. Joint dereverb + denoise on WHAMR!
    /// **8 kHz** — the base-sample-rate sibling of
    /// [`Self::SepformerWhamr16k`]; same reverberant conditioning +
    /// masker head, only the sample rate differs (WHAMR paper Chen et
    /// al. 2022 originally released the 8 kHz variant; the 16 kHz
    /// sibling was published later for wider-band inputs). Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// [`Self::SepFormer`] / [`Self::SepformerWham16kEnh`] /
    /// [`Self::SepformerWhamr16k`] / [`Self::SepformerLibri2Mix`]; the
    /// distinct ModelKind ensures the artifact does NOT silently
    /// inherit the 16 kHz sibling's `vokra.provenance.upstream_hf` =
    /// wrong CDN attribution (both repos live under `speechbrain/` but
    /// with different HF slugs). Every F32 / F16 / BF16 tensor passes
    /// through verbatim; the internal dual-path Transformer body is a
    /// `loud-partial` follow-up (same posture as the sibling SepFormer
    /// variants). Provenance = **apache-2.0** (Permissive — SpeechBrain
    /// end-to-end Apache-2.0).
    SepformerWhamr8k,
    /// **speechbrain/sepformer-dns4-16k-enhancement** (Wave 4 candidate,
    /// 2026-08-01). Category = `enhancement`. Single-speaker speech
    /// enhancement trained on the **Microsoft DNS-4** (Deep Noise
    /// Suppression Challenge 4) corpus at 16 kHz. Shares the
    /// `models::sepformer::ARCH` tag `sepformer` + converter with
    /// [`Self::SepFormer`] / [`Self::SepformerWham16kEnh`] /
    /// [`Self::SepformerWhamr16k`] / [`Self::SepformerLibri2Mix`] /
    /// [`Self::SepformerLibri3Mix`] / [`Self::SepformerWhamr8k`]; the
    /// distinct ModelKind ensures the artifact does NOT silently
    /// inherit any WHAM! / WHAMR! enhancement sibling's
    /// `vokra.provenance.upstream_hf` = wrong CDN attribution (all
    /// four enhancement variants share `vokra.sepformer.n_out = 1`,
    /// so the distinct provenance stamp is the only surface that
    /// discriminates them at load time — silent misrouting would not
    /// fail loudly at the binder). Every F32 / F16 / BF16 tensor
    /// passes through verbatim; the internal dual-path Transformer
    /// body is a `loud-partial` follow-up (same posture as every
    /// sibling SepFormer variant). Provenance = **apache-2.0**
    /// (Permissive — SpeechBrain end-to-end Apache-2.0; the Microsoft
    /// DNS-4 corpus itself is corpus-level provenance separate from
    /// the fine-tuned weight license, identical posture to the
    /// sibling `wham16k-enhancement` / `whamr16k` rows).
    SepformerDns4Enh,
    /// **speechbrain/sgmse-voicebank** (SoTA plan candidate wave,
    /// 2026-08-04). Category = `enhancement`. SGMSE = Score-based
    /// Generative Model for Speech Enhancement (Welker et al. 2022 /
    /// Richter et al. 2023, arXiv:2212.11851 / arXiv:2208.05830) —
    /// NCSN++ v2 score network + OUVE SDE reverse sampler (predictor:
    /// reverse_diffusion, corrector: annealed Langevin dynamics, N=30
    /// per upstream `hyperparams.yaml`), fine-tuned on the VoiceBank-
    /// DEMAND corpus (Valentini-Botinhao 2016). Distinct from every
    /// sibling enhancement / denoise ModelKind (`Denoise` =
    /// DeepFilterNet3 masking, `MpSenet` / `MpSenetDns` = magnitude+
    /// phase U-Net, `MetricganPlus` = generator-only PESQ-tuned GAN,
    /// `SepFormer` family = dual-path Transformer masker+decoder,
    /// `FacebookDenoiser` = time-domain UNet) — SGMSE's diffusion
    /// posture warrants its own arch tag `sgmse` (silently sharing
    /// would mis-route runtime dispatch, FR-EX-08). **First real
    /// weight** in the Vokra catalog to exercise the M3-05
    /// `flow_sampler` + ODE solver op family (the sibling enhancement
    /// rows above are all masking or time-domain UNet). Single
    /// upstream `.ckpt` (`score_model_ema.ckpt` ~263 MB torch pickle) +
    /// `hyperparams.yaml`; bridged to safetensors by
    /// `tools/parity/sgmse_prepare_checkpoint.py` (`.ckpt` payload is
    /// a flat state_dict of the internal NCSN++ v2 network — SpeechBrain's
    /// `Pretrainer` adds the `score_model.` prefix at load time; the
    /// converter preserves the flat layout so a future `Sgmse::from_gguf`
    /// can walk the same names). Every F32 / F16 / BF16 tensor passes
    /// through verbatim; the internal NCSN++ v2 + OUVE SDE reverse
    /// sampler forward is a `loud-partial` follow-up. Provenance =
    /// **apache-2.0** (Permissive — HF cardData API primary source
    /// verified 2026-08-04). Convert with `convert_sgmse_file`.
    Sgmse,
    /// FunASR **fsmn-vad** VAD checkpoint (TIER 1 F wave, 2026-07-30).
    /// Category = `vad`. FSMN = Feedforward Sequential Memory Network
    /// (Zhang et al. 2015 arXiv:1512.08301), the classic FunASR
    /// streaming VAD. BF16 pass-through skeleton — every F32 / F16 /
    /// BF16 tensor passes through verbatim under its upstream
    /// safetensors name. `FunAudioLLM/fsmn-vad-GGUF` is a re-hosted
    /// GGUF sibling of the same weight (from_arg alias) — no separate
    /// ModelKind. Provenance = **apache-2.0** (Permissive).
    FsmnVad,
    /// FireRedTeam **FireRedVAD** checkpoint (TIER 1 F wave,
    /// 2026-07-30). Category = `vad`. Xiaohongshu's transformer-based
    /// streaming VAD (part of the FireRedTeam speech family alongside
    /// FireRedASR / FireRedTTS). BF16 pass-through skeleton —
    /// distinct arch tag from FSMN-VAD (transformer topology unlike
    /// FSMN's filter-window feed-forward). Provenance = **apache-2.0**
    /// (Permissive).
    FireredVad,
    /// pipecat-ai **smart-turn-v2** checkpoint (TIER 1 F wave,
    /// 2026-07-30). Category = `vad` (turn-taking = VAD variant for
    /// dialogue turn boundaries — Pipecat realtime pipelines).
    /// Small classifier that decides when a user has finished
    /// speaking, rather than raw voice activity. BF16 pass-through
    /// skeleton. Provenance = **bsd-2-clause** (Permissive).
    SmartTurn,
    /// LAION **CLAP** (Contrastive Language-Audio Pretraining)
    /// checkpoint (TIER 1 F wave, 2026-07-30). Category =
    /// `classification` (audio-text embedding — downstream users pick
    /// a text prompt vocabulary to get an N-way classifier). HTSAT
    /// audio encoder + text encoder + fused projection (Wu et al.
    /// 2023 arXiv:2211.06687). One of the highest-download HF audio
    /// releases (8.1M+). BF16 pass-through skeleton preserving both
    /// towers verbatim. Provenance = **apache-2.0** (Permissive).
    Clap,
    /// MIT (organization) **Audio Spectrogram Transformer** fine-
    /// tuned on AudioSet (TIER 1 F wave, 2026-07-30). Category =
    /// `classification`. Gong et al. 2021 (arXiv:2104.01778) — ViT
    /// over log-mel spectrogram, 527-class AudioSet classifier.
    /// BF16 pass-through skeleton. Note: `MIT` is the HF ORGANIZATION
    /// that published the model, NOT the SPDX license — the actual
    /// weight license is **bsd-3-clause** (Permissive).
    Ast,
    /// SpeechBrain **lang-id-voxlingua107-ecapa** checkpoint (TIER 1
    /// F wave, 2026-07-30). Category = `classification`. 107-language
    /// identification (Valk & Alumäe 2021 arXiv:2011.12998) with
    /// ECAPA-TDNN backbone. BF16 pass-through skeleton — shares the
    /// `models::speechbrain_lang_id` file with the CommonLanguage
    /// sibling ([`Self::LangIdCommonLanguage`]); both variants share
    /// the ECAPA-TDNN topology and differ only in the head vocab (a
    /// shape-derivable hparam). Provenance = **apache-2.0**
    /// (Permissive).
    LangIdVoxlingua107,
    /// SpeechBrain **lang-id-commonlanguage_ecapa** checkpoint (TIER
    /// 1 F wave, 2026-07-30, sibling of [`Self::LangIdVoxlingua107`]).
    /// Category = `classification`. CommonLanguage-trained variant
    /// (~45 languages) sharing the same ECAPA-TDNN topology; the
    /// distinct ModelKind ensures the correct `vokra.model.name` +
    /// `vokra.provenance.upstream_hf` land on the artifact. BF16 pass-
    /// through skeleton. Provenance = **apache-2.0** (Permissive).
    LangIdCommonLanguage,
    /// SpeechBrain **spkrec-xvect-voxceleb** X-vector checkpoint
    /// (TIER 1 F wave, 2026-07-30). Category = `speaker`. TDNN-based
    /// speaker embedding (Snyder et al. 2018 arXiv:1710.10467) — an
    /// alternative to CAM++ ([`Self::CamPlus`]) and ECAPA-TDNN
    /// ([`Self::EcapaTdnn`]) with distinct topology (plain TDNN stack
    /// with statistics pooling; no SE-Res2Blocks; no D-TDNN). BF16
    /// pass-through skeleton. Provenance = **apache-2.0** (Permissive).
    XVector,
    /// MelodyMachine **Deepfake-audio-detection-V2** checkpoint (TIER
    /// 1 F wave, 2026-07-30). Category = `classification`. WavLM-based
    /// binary classifier (real vs synthetic speech) for audio
    /// deepfake detection — sits in the EU AI Act Article 50 /
    /// SB-942 compliance surface. BF16 pass-through skeleton — the
    /// actual deployment decision (threshold, whether to expose to
    /// end-user) is downstream policy, not runtime. Provenance =
    /// **apache-2.0** (Permissive).
    DeepfakeDetection,
    /// Kyutai **TTS-1.6B-EN/FR** safetensors checkpoint (TIER 2 land,
    /// 2026-07-30). English + French text-to-speech: Moshi / Helium
    /// temporal transformer (`num_layers=16 / dim=2048 / num_heads=16`)
    /// with a 4-layer depformer over 32 Mimi audio codebooks
    /// (`dep_q=32`, `n_q=32`), cross-attention on a 512-d speaker
    /// reference embedding + LUT CFG scale (7-bin) + LUT unified
    /// control token, `depformer_multi_linear=true`,
    /// `demux_second_stream=true`. Distinct arch from
    /// `ModelKind::KyutaiStt` — STT and TTS are two directions of the
    /// same delayed-streams-modeling family (arXiv:2410.00037) but
    /// live at different `vokra.model.arch` tags because their
    /// runtime dispatch differs. CC-BY 4.0 weight
    /// (`AttributionRequired` — the converter stamps the FR-MD-09
    /// attribution text). Every F32 / F16 / BF16 tensor passes through
    /// verbatim; real-weight parity is deferred to owner
    /// (`docs/license-audit.md` §3.1 sign-off). Category = `tts`.
    KyutaiTts,
    /// Meta / Facebook **audiobox-aesthetics** safetensors checkpoint
    /// (TIER 2 land, 2026-07-30). Audio quality-rating classifier /
    /// regressor: wav2vec2-style SSL backbone + 5-layer projection MLP
    /// producing a 5-dim quality rating (BALANCED /
    /// CONTENT_ENJOYMENT / CONTENT_USEFULNESS / PRODUCTION_COMPLEXITY /
    /// PRODUCTION_QUALITY; arXiv:2502.05139 "Meta Audiobox Aesthetics:
    /// Unified Automatic Quality Assessment for Speech, Music, and
    /// Sound"). **First Vokra converter with category =
    /// `classification`** (sibling categories today: `asr` / `tts` /
    /// `codec` / `speaker` / `emotion` / `s2s` / `bert` / `vad`).
    /// CC-BY 4.0 weight (`AttributionRequired` — the converter stamps
    /// the FR-MD-09 attribution text). Every F32 / F16 / BF16 tensor
    /// passes through verbatim; upstream is F32 today (~104M
    /// parameters, ~415 MB on disk) but the BF16 arm accepts a
    /// future distilled fine-tune of the same architecture without a
    /// silent widen. Real-weight parity is deferred to owner
    /// (`docs/license-audit.md` §3.1 sign-off).
    AudioboxAesthetics,
    /// Mistral **Voxtral-Mini-4B-Realtime-2602** — apache-2.0 weight,
    /// ~8 GB, TIER 2 defer marker (2026-07-30). Ministral-3-3B-Base
    /// derived, streaming-optimised realtime ASR variant. This
    /// variant's size sits at the M1 iMac 16 GB local-convert ceiling
    /// (memory [[feedback-large-models-on-vast-ai]] = >8 GB safetensors
    /// preferred on vast.ai). CC dispatch **refuses local convert**
    /// and prints an owner-vast-ai flow message; owner runs the actual
    /// conversion + publish on vast.ai. from_arg + license_class both
    /// register this variant so an accidental local invocation
    /// fail-closed on a loud usage error rather than silently starting
    /// a mmap-heavy convert that may kill the host.
    VoxtralMiniRealtime,
    /// Cohere **cohere-transcribe-03-2026** — apache-2.0 weight,
    /// **HF gate = `auto`** — TIER 2 defer marker (2026-07-30). 14-lang
    /// ASR (`CohereAsrForConditionalGeneration`, ~1 GB safetensors),
    /// released 2026 by CohereLabs. HF gate acceptance requires an
    /// authenticated owner action (a token attached to a HF account that
    /// has clicked "Accept license" on the model card); CC cannot
    /// discharge that on its own, so this variant registers in from_arg
    /// plus license_class only and dispatch **refuses local convert**
    /// with a `defer-gated` owner message.
    CohereTranscribe,
    /// NVIDIA **nemotron-3.5-asr-streaming-0.6b** — `license: other`
    /// (NVIDIA custom licence text), ~1.2 GB — TIER 2 defer marker
    /// (2026-07-30). FastConformer cache-aware streaming ASR spanning
    /// 36 languages. HF cardData carries `license: other` (not a known
    /// SPDX id), so `LicenseClass::from_license_str` cannot classify
    /// without a primary-source read of the NVIDIA licence text —
    /// which is an owner decision (memory
    /// [[feedback-license-signoff-primary-source]] = fail-closed
    /// default). CC dispatch **refuses local convert** with a
    /// `defer-other-license` owner message. from_arg + license_class
    /// register the variant as `LicenseClass::Unknown` so an
    /// accidental commercial-mode load fails the M2-13 gate closed.
    NemotronAsrStreaming,
    /// **FCPE** (Fast Context-based Pitch Estimator, CNChTu/FCPE, MIT
    /// Permissive) — Conformer-based 360-bin log-frequency pitch
    /// classifier. Category = `f0`. Real forward: mel[T, 128] → Linear
    /// stem → `vokra_ops::conformer::ConformerEncoder` (6 blocks, d_model
    /// 512, n_heads 8, ffn_dim 2048, kernel_size 9) → LayerNorm → Linear
    /// head → softmax → cent-grid soft-argmax → Hz + V/UV. Every F32 /
    /// F16 / BF16 tensor passes through verbatim (the neucodec / xcodec2
    /// BF16 pass-through contract); upstream ships torch-pickle `.pt` so
    /// callers pre-flatten to safetensors via
    /// `tools/parity/fcpe_prepare_checkpoint.py` (the DFN3 / DAC / CSM
    /// bridge pattern — no pickle ever enters the runtime, FR-LD-05).
    Fcpe,
    /// **Vocos** (`charactr/vocos-mel-24khz`, `charactr/vocos-encodec-24khz`,
    /// MIT) safetensors checkpoint (2026-08-01 wave). Category = `vocoder`.
    /// **Highest-download HF vocoder audio release** as of 2026-08-01
    /// (2.85M mel-24khz downloads). Fourier-space vocoder
    /// (Siuzdak 2023 arXiv:2306.00814) = ConvNeXt V2 backbone +
    /// **iSTFT head** — a fundamentally different topology from every
    /// HiFi-GAN family sibling (`bigvgan`, `hifigan_vocoder`,
    /// `speecht5_hifigan`) which upsample time-domain waveforms
    /// through transposed-conv + MRF blocks. Distinct arch tag `vocos`
    /// silently mis-routing would misfire the runtime dispatch. Both
    /// upstream releases ship torch pickle `pytorch_model.bin` +
    /// `config.yaml` only (no `model.safetensors` mirror, verified
    /// 2026-08-01 via HF cardData API); callers pre-flatten to
    /// safetensors offline via
    /// `tools/parity/vocos_prepare_checkpoint.py` (thin bridge over
    /// `bin_to_safetensors.py` — the SpeechT5-HiFi-GAN pattern). BF16
    /// pass-through skeleton — every F32 / F16 / BF16 tensor passes
    /// through verbatim under its upstream state-dict name; runtime
    /// binding + real-weight parity are deferred to owner
    /// (`docs/license-audit.md` §3.1 sign-off queue). Provenance =
    /// **mit** for both variants (Permissive — verified 2026-08-01 via
    /// HF cardData API `license: mit`). Two variants collapse into
    /// this single ModelKind; [`convert_file_with_slug`] picks the
    /// correct `models::vocos::VocosVariant` from the raw `--model`
    /// slug (mirror of [`Self::Focalcodec`] / [`Self::BigVGan`]).
    /// **CLAUDE.md 設計判断 §Vocos**: INT8-fragile (「INT8 崩壊」→
    /// fp16 必須) — the converter never emits INT8 (K-quant is
    /// Whisper-only per `--quantize` guard); BF16 is pass-through safe.
    Vocos,
    /// **SNAC** (`hubertsiuzdak/snac_{24khz,44khz}`, MIT) safetensors
    /// checkpoint (2026-08-01 Wave 3). Category = `codec`. Multi-Scale
    /// Neural Audio Codec (Siuzdak et al. 2024, arXiv:2410.14411).
    /// Two variants share this single ModelKind; the slug picks the
    /// frame rate + RVQ depth via `convert_file_with_slug` (mirror of
    /// [`Self::Focalcodec`] / [`Self::BigVGan`] slug dispatch).
    Snac,
    /// Novateur **WavTokenizer-large-speech-75token** (2026-08-01 Wave 3).
    /// Single-codebook FSQ audio codec at 24 kHz with `hop_length = 320`
    /// → 75 tokens/sec (Ji et al. 2024, arXiv:2408.16532). MIT.
    /// Upstream ships `.ckpt` (Lightning); bridge required.
    Wavtokenizer,
    /// **IBM Granite Speech 4.1 2B** (`ibm-granite/granite-speech-4.1-2b`,
    /// apache-2.0, 4.8 GB 4-shard). HF Open ASR leaderboard top-tier
    /// (2026-08-01 Wave 3). FastConformer CTC encoder + Granite LLM
    /// decoder + LoRA adapter. Distinct arch tag `granite_speech`
    /// from Voxtral / Canary / omni-asr-ctc.
    GraniteSpeech,
    /// **MOSS-Audio-Tokenizer** (`OpenMOSS-Team/MOSS-Audio-Tokenizer`
    /// and `-Nano` sibling, apache-2.0). The codec half of the MOSS-TTS
    /// pipeline (waveform to discrete tokens fed into the sibling
    /// [`Self::MossTts`] / [`Self::MossTtsV15`] / [`Self::MossTtsNano`]
    /// / [`Self::MossTtsLocal`] LLM). Category = `codec`. Wave 3 codec
    /// add, 2026-08-01. Two variants collapse into this single
    /// ModelKind; [`convert_file_with_slug`] picks the correct
    /// `models::moss_audio_tokenizer::MossAudioTokenizerVariant`
    /// from the raw `--model` slug (mirror of [`Self::Snac`] and
    /// [`Self::Focalcodec`] slug dispatch). Full variant is
    /// `OpenMOSS-Team/MOSS-Audio-Tokenizer` (~1.77B F32 params, 6.6 GB
    /// effective across 2 sharded safetensors, M1 iMac tight-fit per
    /// memory `[[feedback-large-models-on-vast-ai]]`). Nano variant is
    /// `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (~22M F32 params, 88 MB
    /// single shard). Both variants ship sharded safetensors plus a
    /// `model.safetensors.index.json` weight-map; callers pre-merge to
    /// a single safetensors via
    /// `tools/parity/moss_audio_tokenizer_prepare_checkpoint.py` (the
    /// [`Self::GraniteSpeech`] posture). BF16 pass-through skeleton
    /// mirror of [`Self::Snac`] / [`Self::Neucodec`] /
    /// [`Self::Focalcodec`]; real-weight parity and runtime binder are
    /// deferred to owner sign-off (`docs/license-audit.md` §3.1).
    /// Distinct arch tag `moss_audio_tokenizer` from every sibling
    /// codec (silently sharing would mis-route runtime dispatch;
    /// `MossAudioTokenizerModel` class is OpenMOSS-specific).
    MossAudioTokenizer,
    /// **Amphion NaturalSpeech 3 FACodec** (`amphion/naturalspeech3_facodec`,
    /// apache-2.0). 2026-08-01 Wave 3 codec add. Category = `codec`.
    /// **First factorized VQ (FVQ) codec** in the tree — distinct from
    /// every sibling codec family (RVQ Mimi/DAC/SNAC / FSQ WavTokenizer/
    /// X-Codec 2 / focal-modulation Focalcodec). Runs 3 quantizer heads
    /// in **parallel** over disentangled subspaces: prosody (1
    /// codebook) + content (2 codebooks) + detail (3 codebooks) = 6
    /// total codebooks (paper §3.1 arXiv:2403.03100 Ju et al. 2024).
    /// 16 kHz, hop_size 200 → 80 tok/s per subspace. Four variants
    /// share this single ModelKind; the slug picks the encoder+decoder
    /// pair via `convert_file_with_slug` (mirror of
    /// [`Self::Snac`] / [`Self::MossAudioTokenizer`] slug dispatch):
    /// `v1` (encoder+decoder), `v2` (encoder_v2+decoder_v2, canonical
    /// default = highest-quality pair), `redecoder-v1` /
    /// `redecoder-v2` (adds redecoder for zero-shot voice conversion).
    /// Upstream ships 5 separate `torch.save()` pickle `.bin` files at
    /// repo root (no `model.safetensors` mirror, no `config.json`);
    /// callers pre-merge the variant subset to a single safetensors via
    /// `tools/parity/naturalspeech3_facodec_prepare_checkpoint.py`
    /// (uv-managed Python 3.12 multi-file bridge mirror of
    /// `sepformer_prepare_checkpoint.py`). BF16 pass-through skeleton
    /// (mirror of [`Self::Snac`] / [`Self::MossAudioTokenizer`]);
    /// real-weight parity + runtime binder deferred to owner
    /// (`docs/license-audit.md` §3.1 sign-off queue). **Voice-conversion
    /// policy note**: the `redecoder-v{1,2}` variants specifically
    /// enable zero-shot voice conversion (swap timbre while preserving
    /// prosody + content codes) — CLAUDE.md 設計判断 8 (ELVIS Act /
    /// NO FAKES Act) treats this as an owner routing decision (main
    /// zoo vs `vokra-voiceclone-experimental`); the base `v1` / `v2`
    /// variants are unambiguously codec-class and belong in the main
    /// zoo. All four variants fit comfortably on M1 iMac 16 GB
    /// (largest = redecoder-v2 ~601 MB peak resident, vast.ai not
    /// required per memory [[feedback-large-models-on-vast-ai]]).
    Facodec,
    /// **YuE-upsampler** (`m-a-p/YuE-upsampler`, apache-2.0) — the
    /// vocoder half of the YuE full-song music-generation system
    /// (Yuan et al. 2025, arXiv:2503.08638). 2026-08-01 Wave 3
    /// sibling-pair add. Category = `vocoder`. Distinct arch tag
    /// `yue_upsampler` from sibling Charactr AI `vocos` (different
    /// config axes: n_fft=3528, hop_length=882, 44.1 kHz output;
    /// trained on YuE codec latents not mel or EnCodec inputs) —
    /// silently sharing arch would mis-route the runtime dispatch.
    /// VocosBackbone (input_channels=1024, dim=512, intermediate_dim=1536,
    /// num_layers=8) + ISTFTHead (n_fft=3528, hop_length=882) yields
    /// 44.1 kHz PCM at 50 Hz frame rate. Upstream ships torch pickle
    /// `.pth` only (145 MB, no safetensors mirror) — callers pre-flatten
    /// via `tools/parity/yue_bundle_prepare_checkpoint.py` (mirror of
    /// `naturalspeech3_facodec_prepare_checkpoint.py` +
    /// `bin_to_safetensors.py`). BF16 pass-through skeleton mirror of
    /// vocos / snac / focalcodec / speecht5_hifigan; runtime binder +
    /// real-weight parity deferred to owner sign-off (§3.1). Both
    /// snapshots (`decoder_131000.pth` / `decoder_151000.pth`, byte-
    /// identical to the corresponding files inside `xcodec_mini_infer`)
    /// share this single ModelKind; the prep bridge picks one via
    /// `--snapshot 131000|151000` (default 151000 = later training
    /// step). Sibling [`Self::YueXcodecMini`] carries the codec half
    /// of the bundle (SoundStream RVQ + HuBERT semantic encoder).
    /// **CLAUDE.md 設計判断 §Vocos**: INT8-fragile (「INT8 崩壊」→ fp16
    /// 必須) inherited from the Vocos-family topology — the converter
    /// never emits INT8 (K-quant is Whisper-only per `--quantize`
    /// guard); BF16 is pass-through safe.
    YueUpsampler,
    /// **YuE xcodec-mini bundle** (`m-a-p/xcodec_mini_infer`,
    /// apache-2.0) — the codec half of the YuE full-song
    /// music-generation system. 2026-08-01 Wave 3 sibling-pair add.
    /// Category = `codec`. Distinct arch tag `yue_xcodec_mini` from
    /// every sibling codec (`mimi` / `dac` / `snac` / `wavtokenizer` /
    /// `neucodec` / `xcodec2` / `focalcodec` / `funcodec` /
    /// `speechtokenizer` / `bicodec` / `xy_tokenizer` /
    /// `step_audio2_mini` / `moss_audio_tokenizer` / `facodec`) — YuE
    /// xcodec-mini is a **multi-part bundle** (SoundStream RVQ codec
    /// at 16 kHz / 25 Hz + HuBERT-base semantic encoder + Vocos
    /// decoder head all in one repo), the semantic-encoder fusion is
    /// what distinguishes it from plain RVQ / FSQ codecs. Silently
    /// sharing an arch with `mimi` / `dac` / `snac` would mis-route
    /// to a codec-only decode path that has no semantic fusion input.
    /// SoundStream generator (n_filters=32, D=256, ratios=`[8,5,4,2]` →
    /// 640x downsample, sample_rate=16000, bins=1024,
    /// target_bandwidths=[0.5,1,1.5,2,4,6] kbps) — this is an RVQ
    /// codec, NOT FSQ. Upstream repo bundles three weight groups plus
    /// source-tree copies of RepCodec (MIT) and Descript-Audio-Codec
    /// (MIT) — the source trees are inference-tree artefacts, not
    /// loaded; only the three weight groups (codec 1.36 GB / HuBERT
    /// semantic encoder 377 MB / Vocos decoder head 145 MB byte-
    /// identical to [`Self::YueUpsampler`]) are consumed. All three
    /// upstream weight files are torch pickle (`.pth` / `.bin`) —
    /// callers pre-flatten and role-prefix (`codec.*` / `semantic.*` /
    /// `decoder.*` in the merged safetensors) via
    /// `tools/parity/yue_bundle_prepare_checkpoint.py` so a future
    /// `YueXcodecMini::from_gguf` can locate the three sub-modules.
    /// BF16 pass-through skeleton mirror of vocos / snac / focalcodec;
    /// runtime binder + real-weight parity deferred to owner sign-off
    /// (§3.1). Sibling [`Self::YueUpsampler`] carries only the Vocos
    /// vocoder head (145 MB standalone re-package).
    YueXcodecMini,
    /// **MusicGen-Medium** (`facebook/musicgen-medium`, **cc-by-nc-4.0**)
    /// safetensors checkpoint (Wave 5 candidate, 2026-08-01). First
    /// **music generation** target to land a converter (post-2026-07-30
    /// scope expansion `[[project-scope-expansion-2026-07-30]]`). 1.5B
    /// autoregressive transformer LM over EnCodec RVQ tokens conditioned
    /// on frozen T5 text encoder (Copet et al. 2023, arXiv:2306.05284
    /// "Simple and Controllable Music Generation"). Category = `music`
    /// (first music-tree entry, distinct from every speech-tree tag).
    /// BF16 pass-through skeleton mirror of xcodec2 / wavtokenizer;
    /// **NonCommercial default** — the M2-13 runtime gate refuses to
    /// load in commercial mode unless overridden via
    /// [`convert_file_licensed`] `license` (the Whisper / kokoro /
    /// vits-ja / xcodec2 override pattern). Distinct arch tag `musicgen`
    /// from every sibling — silently sharing with any speech-tree arch
    /// would mis-route runtime dispatch. Scale ~11.4 GB = **vast.ai
    /// handoff** per memory `[[feedback-large-models-on-vast-ai]]`
    /// (M1 iMac 16 GB unsafe for this class of publish). Real-weight
    /// parity + runtime binder deferred to owner sign-off
    /// (`docs/license-audit.md` §3.1 sign-off queue).
    MusicGenMedium,
    /// **MusicGen-Large** (`facebook/musicgen-large`, **cc-by-nc-4.0**)
    /// safetensors checkpoint (Wave 5 candidate, 2026-08-01). Second
    /// **music generation** target — top rung of the MusicGen family
    /// (300M `-small` / 1.5B `-medium` / **3.3B `-large`**). Post-2026-07-30
    /// scope expansion `[[project-scope-expansion-2026-07-30]]`. 3.3B
    /// autoregressive transformer LM over EnCodec RVQ tokens conditioned
    /// on frozen T5 text encoder (Copet et al. 2023, arXiv:2306.05284
    /// "Simple and Controllable Music Generation"). Category = `music`
    /// (shared with sibling MusicGen-Medium — first music-tree family).
    /// BF16 pass-through skeleton mirror of musicgen_medium / xcodec2 /
    /// wavtokenizer; **NonCommercial default** — the M2-13 runtime gate
    /// refuses to load in commercial mode unless overridden via
    /// [`convert_file_licensed`] `license` (the Whisper / kokoro /
    /// vits-ja / xcodec2 override pattern, same as sibling
    /// [`Self::MusicGenMedium`]). Sibling-file split (dedicated
    /// `musicgen_large.rs`) rather than a shared `musicgen.rs` variant
    /// enum — zero-churn on the medium landing + distinct upstream HF
    /// repo (`facebook/musicgen-large` vs `facebook/musicgen-medium`).
    /// Distinct arch tag `musicgen` (shared with sibling
    /// [`Self::MusicGenMedium`] — same family topology, only dims
    /// differ). Scale ~19.5 GB = **vast.ai handoff** per memory
    /// `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB unsafe;
    /// larger than sibling MusicGen-Medium ~11.4 GB). Real-weight
    /// parity + runtime binder deferred to owner sign-off
    /// (`docs/license-audit.md` §3.1 sign-off queue).
    MusicGenLarge,
    /// **MusicGen-Melody** (`facebook/musicgen-melody`, **cc-by-nc-4.0**)
    /// safetensors checkpoint (Wave 5 sibling, 2026-08-02). Third
    /// **music generation** target — the medium 1.5B autoregressive
    /// transformer LM + **chroma conditioning** (12-bin chromagram of a
    /// reference melody clip concatenated to the T5 text conditioning
    /// stream, Copet et al. 2023 arXiv:2306.05284 §3.3). The LM
    /// topology is byte-identical to sibling [`Self::MusicGenMedium`];
    /// only the conditioning frontend + projection differ, so the
    /// converter reuses the medium BF16 pass-through arm via the
    /// shared `models::musicgen_medium::convert_musicgen_family_file`
    /// helper (a wrapper `convert_musicgen_melody_file` flips only the
    /// `vokra.model.name` + `vokra.provenance.{model_id,source,
    /// upstream_hf}` stamps to the melody spellings). Category =
    /// `music` (shared with sibling MusicGen family — first music-tree
    /// family). Distinct arch tag `musicgen` (shared with siblings —
    /// same topology, only chroma frontend differs). Reusing the
    /// medium models/*.rs file rather than a new `musicgen_melody.rs`
    /// = the sibling-in-place pattern anticipated by the medium
    /// module docstring (option a/b decision — kept in one file for
    /// this variant since the tensor manifest is byte-identical, only
    /// four id chunks flip). **NonCommercial default** fail-closed —
    /// override via `--license <spdx>` (same Whisper / kokoro / vits-ja
    /// / xcodec2 pattern). Scale ~6 GB = **vast.ai handoff** per memory
    /// `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB unsafe
    /// per the ≥8 GB cutoff calibrated on Voxtral-Small-24B 48 GB
    /// swap-death). Real-weight parity + chroma-frontend runtime op
    /// deferred to owner sign-off (`docs/license-audit.md` §3.1
    /// sign-off queue).
    MusicGenMelody,
    /// **AudioGen-Medium** (`facebook/audiogen-medium`, **cc-by-nc-4.0**)
    /// safetensors checkpoint (Wave 5 residual, 2026-08-01). MusicGen
    /// sibling with identical arch (shared `musicgen` arch tag), tuned on
    /// environmental sounds / SFX rather than music. `category = "music"`
    /// shared with MusicGen family. `LicenseClass::NonCommercial` fail-
    /// closed default per X-Codec 2 / MusicGen T4 precedent. Scale
    /// ~3.7 GB = local convert safe on M1 iMac 16 GB.
    AudioGenMedium,
    /// **MusicGen-Small** (`facebook/musicgen-small`, **cc-by-nc-4.0**)
    /// safetensors (Wave 6 residual, 2026-08-01). 300M smallest of the
    /// MusicGen family. Shared `musicgen` arch + `music` category. T4
    /// NonCommercial default. Scale ~5.5 GB = vast.ai handoff per owner
    /// 「重いモデル vast.ai」directive.
    MusicGenSmall,
    /// **Qwen2-Audio-7B-Instruct** (`Qwen/Qwen2-Audio-7B-Instruct`,
    /// **apache-2.0**) safetensors (Wave 6 residual, 2026-08-01). Alibaba
    /// 7B audio-LLM = Whisper audio encoder + Qwen2-7B LM
    /// (arXiv:2407.10759). Distinct arch `qwen2_audio` + category
    /// `audio-llm`. Scale ~16 GB (5-shard) = vast.ai handoff.
    Qwen2Audio,
    /// **Qwen2.5-Omni-7B** (`Qwen/Qwen2.5-Omni-7B`, **apache-2.0**)
    /// safetensors (Wave residual, 2026-08-02). Alibaba 7B Thinker +
    /// Talker unified any-to-any omni multimodal LLM (audio + vision
    /// + text → audio + text) over a Qwen2.5-7B backbone. **Distinct
    /// arch tag `qwen2-omni`** from sibling [`Self::Qwen2Audio`]
    /// (audio-only Whisper + Qwen2-7B LM) — the fused Thinker + Talker
    /// pair changes tensor topology + tokenizer + modality-injection
    /// scheme, so FR-EX-08 forbids silent shape misroute across the two.
    /// Category `audio-llm`. Scale **22.37 GB (5-shard)** = vast.ai
    /// handoff per memory `[[feedback-large-models-on-vast-ai]]` (well
    /// above the ≥8 GB strict cutoff — local convert would exhaust the
    /// M1 iMac 16 GB budget). BF16 pass-through skeleton mirror of
    /// sibling `qwen2_audio.rs`. Runtime binder (Thinker + Talker
    /// forward + streaming speech head) deferred to owner sign-off
    /// (`docs/license-audit.md` §3.1).
    Qwen25Omni7b,
    /// **VibeVoice-ASR** (`microsoft/VibeVoice-ASR`, **MIT**) safetensors
    /// (Wave 6 residual, 2026-08-01). VibeVoice sibling with ASR head
    /// (VibeVoiceForASRTraining). Distinct arch `vibevoice_asr` (vs
    /// sibling TTS `vibevoice`). Scale ~16.5 GB (8-shard) = vast.ai.
    VibeVoiceAsr,
    /// **ACE-Step 1.5** (`ACE-Step/Ace-Step1.5`, **MIT**) multi-file
    /// bundle (Wave 6 residual, 2026-08-01). Flagship MIT music-gen =
    /// diffusion + VAE + Qwen3-Embedding + acestep-5Hz-LM + turbo.
    /// Distinct arch `ace_step`, category `music`. Scale ~9.6 GB =
    /// vast.ai handoff (multi-file merge via prep script).
    AceStep,
    /// **HuBERT-Large-LS960** (`facebook/hubert-large-ls960-ft`,
    /// **apache-2.0**) safetensors checkpoint (Wave 7 residual,
    /// 2026-08-01). HuBERT-Large (Hsu et al. 2021, arXiv:2106.07447)
    /// = 317M self-supervised speech encoder + CTC head fine-tuned on
    /// LibriSpeech 960h. **Distinct from sibling
    /// [`Self::Wav2Vec2Ctc`]**: HuBERT uses a BERT-style masked
    /// feature-prediction objective over k-means-clustered features,
    /// wav2vec 2.0 uses a contrastive masked convnet + Gumbel-softmax
    /// quantised negatives. Distinct arch `hubert` — the two share
    /// ops (7-layer Conv1D feature-extractor + Transformer encoder +
    /// CTC decode) but the arch tag stays distinct so runtime
    /// dispatch cannot silently misroute a HuBERT checkpoint into a
    /// wav2vec2 loader (FR-EX-08). Category `asr`, license Permissive
    /// (apache-2.0 per HF cardData). Scale ~1.26 GB = local convert
    /// safe on M1 iMac 16 GB (well below the vast.ai 8 GB cutoff).
    HubertLargeLs960,
    /// **w2v-BERT 2.0** (`facebook/w2v-bert-2.0`, **MIT**) safetensors
    /// → GGUF (hf-audio-gap SSL residual, 2026-08-04). Chung et al.
    /// 2021 arXiv:2108.06209 "w2v-BERT: Combining Contrastive Learning
    /// and Masked Language Modeling for Self-Supervised Speech Pre-
    /// Training" = ~580M-parameter self-supervised speech encoder =
    /// Conformer encoder body + dual (wav2vec2-style contrastive +
    /// BERT-style MLM) SSL branches over the shared representation.
    /// Released standalone by Meta alongside Seamless-M4T v2
    /// (Barrault et al. 2023 arXiv:2312.05187) as the "speech encoder
    /// half" of the Seamless stack.
    ///
    /// **Standalone vs internal-subgraph identity**: prior to this
    /// converter, w2v-BERT 2.0 tensors were present in the Vokra
    /// converter tree only as an INTERNAL subgraph inside two
    /// composite models — (a) [`Self::VieNeuTts`] uses w2v-BERT as its
    /// speaker encoder, (b) [`Self::SeamlessM4tV2Large`] uses w2v-BERT
    /// as its speech encoder. Neither exposes w2v-BERT for standalone
    /// use; a downstream who wants SSL feature extraction for a new
    /// per-language ASR / speaker / VAD head (143+ languages
    /// pretrained per the HF release manifest) requires a standalone
    /// binder that packs the encoder alone, without composite
    /// scaffolding. This variant lands precisely that standalone path.
    ///
    /// **Distinct arch tag `w2v-bert-2`** from siblings
    /// [`Self::HubertLargeLs960`] / [`Self::Wav2Vec2Ctc`] /
    /// [`Self::Data2vecAudioBase`] — the three SSL siblings share the
    /// general feature-extractor + Transformer-body + SSL-head shape
    /// but differ in pretraining objective and encoder topology
    /// (HuBERT = BERT-style masked-feature prediction over a vanilla
    /// Transformer body, wav2vec 2.0 = contrastive convnet with
    /// Gumbel-softmax quantised negatives over a vanilla Transformer
    /// body, data2vec-audio = contextualised latent representation
    /// prediction with an EMA teacher over a wav2vec2 body, w2v-BERT
    /// = combined contrastive + MLM branches over a **Conformer**
    /// encoder body). Silently sharing an arch tag would mis-route
    /// runtime dispatch at the encoder body (Conformer vs vanilla
    /// Transformer has different attention + convolution interleave +
    /// layer-norm placement), which FR-EX-08 (no silent op-shape
    /// misroute) forbids. Category `asr` shared with the SSL sibling
    /// family + Whisper. License **MIT** (HF cardData primary source
    /// `https://huggingface.co/api/models/facebook/w2v-bert-2.0` =
    /// `{"license":"mit","cardData":{"license":"mit"}}`, CC-verified
    /// 2026-08-04) → [`vokra_core::LicenseClass::Permissive`] default,
    /// sibling to Whisper / piper-plus / Silero / CAM++ / Moonshine /
    /// HuBERT / wav2vec2 first-party Permissive posture.
    ///
    /// **Scale ~2.16 GB single-file safetensors** (HF
    /// `/api/models/facebook/w2v-bert-2.0/tree/main` primary source
    /// `model.safetensors.size = 2_322_063_736`, CC-verified
    /// 2026-08-04) = **vast.ai handoff required** per memory
    /// `[[feedback-large-models-on-vast-ai]]` and
    /// `docs/handoff/vast-ai-large-model-publish.md` — **exceeds the
    /// 2 GB local-convert CC-workflow owner threshold**. Converter +
    /// §3.1 audit row + `signoff_match.py` entries land today so the
    /// future vast.ai owner publish is one command away
    /// (`bash scripts/publish/publish-one.sh w2v-bert-2-0 --push`),
    /// but the actual convert + upload happens on vast.ai (not
    /// M1 iMac).
    ///
    /// BF16 pass-through skeleton mirror of sibling
    /// `hubert_large_ls960` / `moonshine_base` / `musicgen_small` /
    /// `openwakeword` — no convert-time widening, runtime widens
    /// BF16 → f32 losslessly at load. Upstream ships F32 (per HF
    /// `safetensors.parameters.F32 = 580_493_120`); the BF16 arm is
    /// exercised only when a downstream re-quantises before
    /// conversion. Runtime binder (Conformer encoder forward + SSL
    /// feature extraction) is deferred to a follow-up (`docs/license-
    /// audit.md` §3.1 sign-off + owner ADR) — the future native
    /// forward will reuse the shared `vokra_ops::conformer` primitive
    /// (SoTA plan Phase 2 landed op, no per-model op duplication).
    W2vBert2,
    /// **AudioLDM 2** (`cvssp/audioldm2`, **cc-by-nc-sa-4.0**)
    /// safetensors checkpoint (Wave 5 candidate, 2026-08-01).
    /// Text-to-audio latent-diffusion generator (Liu et al. 2024 ICML,
    /// arXiv:2308.05734 "AudioLDM 2: Learning Holistic Audio Generation
    /// with Self-supervised Pretraining"). Multi-encoder bundle: VAE
    /// encoder/decoder + latent-diffusion U-Net + HiFi-GAN vocoder +
    /// frozen T5-base + CLAP text encoder + GPT-2 audio-caption LM
    /// (~8.5 GB total).
    ///
    /// **Distinct arch tag `audioldm2`** — silently sharing with sibling
    /// [`Self::MusicGenMedium`] / [`Self::MusicGenLarge`] would misroute
    /// runtime dispatch: MusicGen is an autoregressive transformer LM
    /// over EnCodec RVQ tokens, AudioLDM 2 is a *latent-diffusion*
    /// generator over a VAE latent (fundamentally different topology,
    /// different sampler surface — LDM sampler + VAE decode vs AR token
    /// decode + RVQ decode). `category = "music"` shared with the
    /// MusicGen family (per 2026-07-30 scope expansion
    /// `[[project-scope-expansion-2026-07-30]]`).
    ///
    /// **Doubly-restrictive `NonCommercialShareAlike` default** — the
    /// CVSSP primary source pins CC-BY-NC-SA-4.0 (the HF card's
    /// `-nc-4.0` tag is the looser form and would drop the SA cascade
    /// if defaulted to — Fish-Speech precedent for the same license).
    /// The M2-13 runtime gate refuses to load in commercial mode (NC
    /// gate = fail-closed) AND any downstream republish must carry the
    /// grant forward (SA cascade). Override via
    /// [`convert_file_licensed`] `license` (the Whisper / kokoro /
    /// vits-ja / xcodec2 / musicgen override pattern) only when the
    /// caller legitimately holds the weight under a different SPDX id.
    ///
    /// **Publish blocked (sa-cascade-defer)** — no entry in
    /// `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`, and no
    /// ☑ sign-off in `docs/license-audit.md` §3.1 (owner ADR required
    /// to resolve the SA cascade onto Vokra-added artifacts). The
    /// converter + prep-script land today so a future publish is one
    /// owner decision away, but nothing today routes to `publish-one.sh`.
    ///
    /// BF16 pass-through skeleton mirror of `musicgen_medium` /
    /// `xcodec2` / `wavtokenizer`. Scale ~8.5 GB = **vast.ai handoff**
    /// per memory `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB
    /// unsafe on the upper edge — the multi-encoder bundle doubles peak
    /// resident to ~17 GB on the pass). Real-weight parity + runtime
    /// binder (new op surface = latent-diffusion sampler + VAE +
    /// HiFi-GAN, distinct from `flow_sampler` which targets flow-
    /// matching) deferred to owner sign-off (`docs/license-audit.md`
    /// §3.1 sign-off queue).
    AudioLdm2,
    /// **AudioLDM 2 Large** (`cvssp/audioldm2-large`, **cc-by-nc-sa-4.0**)
    /// safetensors checkpoint (Wave 8 sibling, 2026-08-02). Wider/deeper
    /// sibling of the base AudioLDM 2 variant — the multi-encoder
    /// bundle topology (VAE + latent-diffusion U-Net + HiFi-GAN vocoder
    /// + T5-base + CLAP text encoder + GPT-2 audio-caption LM) is
    /// unchanged from sibling [`Self::AudioLdm2`], only model dims +
    /// optional variant-specific heads differ. Reuses the base BF16
    /// pass-through arm via the shared
    /// `models::audioldm2::convert_audioldm2_family_file` helper (a
    /// wrapper `convert_audioldm2_large_file` flips only the
    /// `vokra.model.name` + `vokra.provenance.{model_id,source,
    /// upstream_hf}` chunks to the large spellings) — no new
    /// `audioldm2_large.rs` module. This is the sibling-in-place
    /// landing pattern the base module docstring anticipated
    /// (musicgen_medium / musicgen_melody 2026-08-02 precedent).
    ///
    /// **Distinct arch tag `audioldm2`** (shared with sibling base —
    /// same family topology, only dims differ). `category = "music"`
    /// shared with the sibling AudioLDM 2 / MusicGen family.
    ///
    /// **Doubly-restrictive `NonCommercialShareAlike` default** — the
    /// CVSSP primary source pins CC-BY-NC-SA-4.0 for every family
    /// sibling (base + large + music + music-665k), so the large
    /// variant inherits the same doubly-restrictive class (NC gate +
    /// SA cascade). Override via [`convert_file_licensed`] `license`
    /// only when the caller legitimately holds the weight under a
    /// different SPDX id.
    ///
    /// **Publish blocked (sa-cascade-defer)** — no entry in
    /// `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS` for
    /// `vokra/audioldm2-large`, and no ☑ sign-off in
    /// `docs/license-audit.md` §3.1 (owner ADR required to resolve
    /// the SA cascade onto Vokra-added artifacts, same posture as
    /// sibling base [`Self::AudioLdm2`]).
    ///
    /// Scale ~7 GB = **vast.ai handoff** per memory
    /// `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB unsafe
    /// on the upper edge — the multi-encoder bundle roughly doubles
    /// peak resident on the pass). Real-weight parity + runtime
    /// binder deferred to owner sign-off (`docs/license-audit.md`
    /// §3.1 sign-off queue).
    AudioLdm2Large,
    /// **BS-Roformer / Mel-Band Roformer** (upstream `chenmozhijin/BSRoformer-
    /// GGUF` third-party mirror, **weight provenance unclear**) safetensors
    /// checkpoint (Wave 5 candidate, 2026-08-01). First **music source
    /// separation** target — Lu et al. 2023 arXiv:2310.01809 "Music Source
    /// Separation with Band-Split RoPE Transformer": dual-path frequency-band
    /// Transformer over an STFT spectrogram, alternating time-axis and
    /// band-axis self-attention (each with RoPE position encoding) to mask
    /// out a target stem (vocals is the most common publication target;
    /// drums / bass / other are also possible).
    ///
    /// **Distinct arch tag `bs_roformer`** — sibling separator arch tags
    /// (`sepformer` = dual-path time-domain, `tiger_separator` = time-frequency
    /// interleaved gain, `mp_senet` = magnitude-phase parallel) address
    /// different failure modes. Silently sharing would mis-route runtime
    /// dispatch to a wrong-shape forward. `category = "separation"` shared
    /// with the SepFormer speech-separation family — BS-Roformer is the
    /// music-vocals analogue.
    ///
    /// **License posture — weight provenance unclear (fail-closed default)**:
    /// weight redistribution default is
    /// [`vokra_core::LicenseClass::RedistributionForbidden`]. The architecture / reference
    /// code is MIT (`github.com/lucidrains/BS-RoFormer`, Phil Wang's
    /// clean-room implementation), but the paper's authors released no
    /// reference weights — every checkpoint in the wild is a downstream
    /// retraining under mixed licenses (some GPL-3.0, some CC-BY-NC-4.0, most
    /// unspecified). A converter cannot know which SPDX id covers the
    /// caller's checkpoint. Sibling [`Self::VitsJa`] uses the same
    /// fail-closed default (there for corpus-restriction reasons; here for
    /// unclear-provenance reasons).
    ///
    /// **Publish blocked** at
    /// `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS` — no entry
    /// for `bs-roformer` (unlisted slug fails closed as `UNKNOWN_REPO` at
    /// `publish-one.sh` gate time). An owner decision selecting a specific
    /// checkpoint (and thus a specific license) is the prerequisite to a
    /// first publish. A caller who knows the specific SPDX id for their
    /// checkpoint overrides at the outer `convert_file --license <spdx>`
    /// boundary (the same Whisper / kokoro / vits-ja / xcodec2 / musicgen
    /// override pattern).
    ///
    /// **Scale**: 150 MB (Mel-Band variants) to ~4-5 GB (top-of-range
    /// BS-Roformer). The 4.68 GB flagship class sits just under the M1 iMac
    /// 16 GB comfortable-local-convert threshold; vast.ai handoff per memory
    /// `[[feedback-large-models-on-vast-ai]]` is recommended for the
    /// top-of-range variants (a 16 GB box suffices).
    ///
    /// Convert with [`convert_bs_roformer_file`] (single-input pass-through,
    /// no config side-car needed). BF16 pass-through skeleton mirror of
    /// [`Self::VitsJa`] / [`Self::MusicGenLarge`]. Real-weight binder +
    /// runtime Lu-et-al-2023 parity is a follow-up wave gated on §3.1
    /// sign-off + owner routing decision (new op surface: band-split RoPE
    /// transformer with alternating time-axis / band-axis attention, mask
    /// estimator over STFT — distinct from every existing op).
    BsRoformer,
    /// **openWakeWord** (`dscripka/openWakeWord`, **apache-2.0**)
    /// safetensors → GGUF (Wave residual, 2026-08-02). Small
    /// custom-KWS MLP/CNN family (~1–5 MB per wake-word) over the
    /// shared Google speech_embedding melspec frontend. Audio-dialect
    /// `kws` op entry (FR-OP `kws`). Distinct arch tag `openwakeword`
    /// (not shared with any other model), category `kws`. HF API
    /// rate-limited (401) but upstream GitHub `dscripka/openWakeWord`
    /// primary source is Apache-2.0 (code + bundled checkpoints). BF16
    /// pass-through skeleton mirror of the sibling
    /// `musicgen_small.rs` / `hubert_large_ls960.rs` skeleton. Scale
    /// ~0.01 GB (~10 wake-words × 1–5 MB each + speech_embedding
    /// frontend) = local convert safe on M1 iMac 16 GB (well below
    /// the vast.ai ≥8 GB cutoff per memory
    /// `[[feedback-large-models-on-vast-ai]]`). Convert with
    /// `models::openwakeword::convert_openwakeword_file` — the
    /// converter takes no side-car config today (single-input
    /// [`convert_file`] path). Runtime port deferred (safetensors →
    /// GGUF bridge only; the audio-dialect `kws` op consumes the
    /// artifact in a future WP).
    Openwakeword,
    /// **Moonshine-Tiny** (`UsefulSensors/moonshine-tiny`, **MIT**)
    /// safetensors → GGUF (Wave residual, 2026-08-02). 27M-parameter
    /// transformer encoder-decoder ASR (Jeffries et al. 2024,
    /// arXiv:2410.15608). **Distinct from sibling [`Self::Whisper`]** in
    /// two significant ways: (1) **no mel front-end** — the model
    /// consumes raw 16 kHz audio directly via a learned Conv1D stack
    /// (bypassing STFT + Mel filterbank), (2) **rotary position encoding
    /// + SwiGLU** activations rather than Whisper's sinusoidal + GELU.
    /// Distinct arch tag `moonshine` — silently sharing with
    /// [`Self::Whisper`] would misroute runtime dispatch at the audio-
    /// input boundary (raw-audio Conv1D vs Mel encoder), which FR-EX-08
    /// (no silent op-shape misroute) forbids. Category `asr` shared
    /// with the Whisper family. License **MIT** →
    /// [`vokra_core::LicenseClass::Permissive`] default, sibling to the
    /// Whisper / piper-plus / Silero / CAM++ first-party Permissive
    /// posture. Scale ~0.11 GB = local convert safe on M1 iMac 16 GB
    /// (well below the vast.ai ≥8 GB cutoff per memory
    /// `[[feedback-large-models-on-vast-ai]]`). BF16 pass-through
    /// skeleton mirror of sibling `musicgen_small.rs` /
    /// `hubert_large_ls960.rs` / `openwakeword.rs`. Runtime binder (raw-
    /// audio Conv1D + rotary + SwiGLU encoder-decoder + greedy decode)
    /// deferred to owner sign-off (`docs/license-audit.md` §3.1).
    MoonshineTiny,
    /// **Moonshine-Base** (`UsefulSensors/moonshine-base`, **MIT**)
    /// safetensors → GGUF (Wave residual, 2026-08-02). 61.5M-parameter
    /// transformer encoder-decoder ASR (Jeffries et al. 2024,
    /// arXiv:2410.15608). Sibling to [`Self::MoonshineTiny`] with the
    /// same architecture family (raw-audio Conv1D front-end + rotary
    /// position encoding + SwiGLU activations) but a wider/deeper
    /// backbone (~2.3× parameter count vs the 27M Tiny variant per the
    /// upstream release manifest). **Distinct from sibling
    /// [`Self::Whisper`]** in two significant ways: (1) **no mel
    /// front-end** — the model consumes raw 16 kHz audio directly via a
    /// learned Conv1D stack (bypassing STFT + Mel filterbank), (2)
    /// **rotary position encoding + SwiGLU** activations rather than
    /// Whisper's sinusoidal + GELU. Shares arch tag `moonshine` with
    /// sibling [`Self::MoonshineTiny`] (Tiny and Base share the same
    /// architecture — only depth/width differ). Silently sharing with
    /// [`Self::Whisper`] would misroute runtime dispatch at the audio-
    /// input boundary (raw-audio Conv1D vs Mel encoder), which
    /// FR-EX-08 (no silent op-shape misroute) forbids. Category `asr`
    /// shared with the Whisper family. License **MIT** →
    /// [`vokra_core::LicenseClass::Permissive`] default, sibling to the
    /// Whisper / piper-plus / Silero / CAM++ / Moonshine-Tiny first-
    /// party Permissive posture. Scale ~0.25 GB = local convert safe on
    /// M1 iMac 16 GB (well below the vast.ai ≥8 GB cutoff per memory
    /// `[[feedback-large-models-on-vast-ai]]`). BF16 pass-through
    /// skeleton mirror of sibling `moonshine_tiny.rs` /
    /// `musicgen_small.rs` / `hubert_large_ls960.rs` / `openwakeword.rs`.
    /// Runtime binder (raw-audio Conv1D + rotary + SwiGLU encoder-
    /// decoder + greedy decode) deferred to owner sign-off
    /// (`docs/license-audit.md` §3.1).
    MoonshineBase,
    /// **Demucs (HT-Demucs)** (`facebook/demucs`, **MIT**) safetensors →
    /// GGUF (Wave residual, 2026-08-02). Meta's hybrid transformer Demucs
    /// (Rouard et al. 2023, arXiv:2211.08553) = U-Net waveform branch +
    /// spectrogram branch joined by cross-domain self-attention,
    /// **4-source music separation** (drums / bass / other / vocals —
    /// MUSDB18 stem taxonomy). **Distinct from siblings
    /// `Self::SepformerWsj02mix` et al. and [`Self::TigerSeparator`]** —
    /// SepFormer is waveform-only dual-path Transformer for speech
    /// mixtures, TIGER is a time-frequency dual-branch dialog/effects/music
    /// separator; HT-Demucs is a hybrid waveform+spectrogram U-Net +
    /// transformer targeting music. Silently sharing arch tags across
    /// these three would misroute runtime dispatch at the separator
    /// masker head (different output branching, different internal
    /// representation domain) — FR-EX-08 (no silent op-shape misroute)
    /// requires the distinct `demucs` arch tag. Category `separation`
    /// shared with the SepFormer / TIGER separator siblings. License
    /// **MIT** → [`vokra_core::LicenseClass::Permissive`] default, sibling
    /// to the Whisper / piper-plus / Silero / CAM++ / Moonshine first-
    /// party Permissive posture (upstream GitHub `LICENSE` primary source
    /// per memory `[[feedback-license-signoff-primary-source]]` — HF
    /// mirror returned 401 on the 2026-08-02 residual walk). Scale
    /// ~0.50 GB = local convert safe on M1 iMac 16 GB (well below the
    /// vast.ai ≥8 GB cutoff per memory
    /// `[[feedback-large-models-on-vast-ai]]`). BF16 pass-through
    /// skeleton mirror of sibling `moonshine_base.rs` / `musicgen_small.rs`
    /// / `hubert_large_ls960.rs` / `openwakeword.rs`. Runtime binder
    /// (hybrid U-Net waveform branch + spectrogram branch + cross-domain
    /// self-attention + `separate_masks` audio op emit) deferred to owner
    /// sign-off (`docs/license-audit.md` §3.1).
    DemucsHtdemucs,
    /// **Ultravox v0.5 (Llama-3.2-1B)** (`fixie-ai/ultravox-v0_5-llama-3_2-1b`,
    /// **MIT**) safetensors checkpoint (Wave residual, 2026-08-02). Ultravox
    /// v0.5 = audio-text-to-text multimodal model combining a Llama-3.2-1B
    /// decoder with a Whisper encoder + lightweight projection adapter.
    /// Both underlying arches (Llama + Whisper) already supported by sibling
    /// converters + runtime primitives; new wiring is the adapter projection
    /// + multimodal prompt template (runtime-side, not converter-side).
    /// Distinct arch tag `ultravox` from sibling [`Self::Voxtral`] (Mistral
    /// decoder) / [`Self::Qwen2Audio`] (Qwen2 decoder) — the decoder backbone
    /// fixes tensor layout + tokenizer + rope base, so FR-EX-08 forbids
    /// silent shape misroute across the three. Category `audio-llm`. Scale
    /// ~1.83 GB = local convert safe on M1 iMac 16 GB (well below the
    /// vast.ai ≥8 GB cutoff). BF16 pass-through skeleton mirror of sibling
    /// `demucs_htdemucs.rs` / `moonshine_base.rs` / `musicgen_small.rs` /
    /// `hubert_large_ls960.rs` / `openwakeword.rs`. Runtime binder deferred
    /// to owner sign-off (`docs/license-audit.md` §3.1).
    UltravoxV05Llama321b,
    /// **XTTS-v2** (`coqui/XTTS-v2`, **coqui-public-model-license**)
    /// safetensors → GGUF (Wave residual, 2026-08-02). Coqui's XTTS v2 =
    /// multilingual zero-shot voice-cloning TTS = GPT-2 backbone (~1.9 GB
    /// checkpoint) autoregressively generating discrete Mel VQ tokens (from
    /// a Discrete VAE / DVAE) conditioned on a speaker conditioning module
    /// (Perceiver-style latent) + language embedding, then decoding via a
    /// HiFi-GAN vocoder head. **Distinct from siblings** [`Self::PiperPlus`]
    /// (VITS2 flow), [`Self::Kokoro`] (StyleTTS2-derived iSTFTNet),
    /// [`Self::CosyVoice2`] (FSQ + Qwen2.5 AR + HiFTNet) — the GPT-2 +
    /// DVAE + HiFi-GAN triple is a distinct topology, so FR-EX-08 (no
    /// silent op-shape misroute) requires the distinct arch tag `xtts`.
    /// License = **coqui-public-model-license** (Coqui's bespoke research-
    /// only / non-commercial license, not SPDX-listed) →
    /// [`vokra_core::LicenseClass::NonCommercial`] fail-closed default per
    /// X-Codec-2 (2026-07-28) / MusicGen family (2026-08-01) T4 tier
    /// precedent. Publish requires `publish-one.sh --allow-noncommercial`.
    /// Note: Coqui shut down Jan 2024; upstream `coqui/XTTS-v2` on HF is
    /// primary source. Category `tts`. Scale ~1.90 GB = local convert safe
    /// on M1 iMac 16 GB (well below the vast.ai ≥8 GB cutoff per memory
    /// `[[feedback-large-models-on-vast-ai]]`). BF16 pass-through skeleton
    /// mirror of sibling `ultravox_v0_5_llama_3_2_1b.rs` / `moonshine_base.rs`
    /// / `musicgen_small.rs` / `hubert_large_ls960.rs`. Runtime binder
    /// (GPT-2 AR decoder + DVAE token generation + HiFi-GAN vocoder) deferred
    /// to owner sign-off (`docs/license-audit.md` §3.1).
    XttsV2,
    /// **ConvTasNet Libri1Mix Enhancement** (Asteroid,
    /// `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`, **cc-by-sa-4.0**)
    /// safetensors checkpoint (2026-08-02 Wave residual). ConvTasNet
    /// (Luo & Mesgarani 2019, arXiv:1809.07454) — fully convolutional
    /// TasNet: encoder + stacked dilated TCN mask estimator + decoder.
    /// This checkpoint is Asteroid's Libri1Mix `enhsingle` fine-tune
    /// (single-speaker enhancement, 16 kHz — one clean speaker +
    /// additive noise, one output stream).
    ///
    /// **Distinct arch tag `conv_tasnet`** — sibling separator families
    /// (`sepformer` = dual-path Transformer masker, `demucs` = hybrid
    /// U-Net + spectrogram + cross-domain attention, `tiger_separator`,
    /// `bs_roformer`, `mp_senet`) address different topologies. Silently
    /// sharing would mis-route runtime dispatch to a wrong-shape forward.
    /// Category `enhancement` (single-output enhancement head — mirrors
    /// the SepFormer WHAM / WHAMR / DNS-4 enhancement sibling posture).
    /// Future multi-speaker ConvTasNet variants would carry
    /// `category = "separation"` under a distinct `ModelKind` arm.
    ///
    /// **License posture — Copyleft (CC-BY-SA-4.0 SA cascade)**: first
    /// entry on the [`vokra_core::LicenseClass::Copyleft`] arm. A GGUF
    /// derived from a CC-BY-SA weight is itself CC-BY-SA; downstream
    /// re-labelling as Apache-2.0 is a misrepresentation, not a mere
    /// attribution drop. Publish is **redistributable with the original
    /// licence preserved** (T3 tier — `publish-one.sh` gate must ship
    /// the upstream LICENSE + NOTICE verbatim). No
    /// `--allow-noncommercial` required (Copyleft ≠ NonCommercial), but
    /// the SA cascade must carry forward on every derivative.
    ///
    /// **Upstream format**: single ~20 MB `pytorch_model.bin` (raw
    /// `torch.save` of the ConvTasNet state dict). Owners run the
    /// standard `bin_to_safetensors.py` prep step before pointing this
    /// converter at the resulting `.safetensors` (same workflow as the
    /// SepFormer `.ckpt` families). This converter deliberately never
    /// reads `pytorch_model.bin` directly — pickle deserialization
    /// inside the Rust runtime would violate the FR-LD-05 "no arbitrary
    /// code execution at load" rule.
    ///
    /// Scale ~0.02 GB = local convert safe on M1 iMac. BF16 pass-through
    /// skeleton mirror of `musicgen_small` / `sepformer` — real-weight
    /// parity + a native `ConvTasNet::from_gguf` forward path are
    /// deferred to owner sign-off (`docs/license-audit.md` §3.1).
    ConvTasnetLibri1mix,
    /// **Seamless-M4T-v2-Large** (`facebook/seamless-m4t-v2-large`,
    /// **cc-by-nc-4.0**) safetensors (Wave residual, 2026-08-02). Meta
    /// SeamlessM4T v2 flagship 2.3B parameter unified any-to-any speech-
    /// and-text translation model — ASR + T2TT + S2TT + T2ST + S2ST
    /// across ~100 source / ~35 target speech languages (Communication
    /// et al. 2023, arXiv:2312.05187). Ships **2 safetensors shards +
    /// `.pt` duplicates + `vocoder_v2.pt`** on HF; the converter walks
    /// whatever bytes the caller hands in (typical publish path pre-
    /// flattens shards + vocoder to a single safetensors offline —
    /// mirror of the CSM / DAC / SpeechT5-HiFi-GAN prepare-checkpoint
    /// pattern; NFR-DS-02 forbids pickle in the runtime, FR-LD-05
    /// forbids arbitrary code execution at load, so `.pt` files must
    /// be flattened outside this crate). Distinct arch tag **`unity-2`**
    /// (Meta's fairseq2 dispatch name) covering the 4 subgraphs —
    /// w2v-BERT 2.0 speech encoder + NLLB-derived text decoder + T2U
    /// (text-to-unit) decoder + HiFi-GAN vocoder (`vocoder_v2.pt`).
    /// FR-EX-08 forbids silent shape misroute across sibling M4T v1
    /// / MMS / Whisper families (different arch topology + different
    /// tokenizer + different vocab / unit head layout). Category
    /// `s2s` (shared with sibling baichuan_audio / step_audio2_mini).
    /// T4 tier `LicenseClass::NonCommercial` fail-closed default per
    /// X-Codec 2 (2026-07-28) / MusicGen family (2026-08-01)
    /// precedent — publish requires `publish-one.sh
    /// --allow-noncommercial`. Scale **~9.00 GB** = vast.ai handoff
    /// per memory `[[feedback-large-models-on-vast-ai]]` (>8 GB
    /// strict cutoff on M1 iMac 16 GB). BF16 pass-through skeleton
    /// mirror of `musicgen_small.rs` / `qwen2_audio.rs`. Runtime
    /// binder (4-subgraph forward + T2U dispatch + vocoder chain)
    /// deferred to owner sign-off (`docs/license-audit.md` §3.1).
    SeamlessM4tV2Large,
    /// **YAMNet** (`thelou1s/yamnet`, apache-2.0 default,
    /// music-understanding wave 2026-08-13) — Google Research's
    /// MobileNetV1 audio-event classifier (521-class AudioSet, ~15 MB
    /// edge model, 16 kHz mono log-mel input at 96 mel bins × 0.96 s
    /// frames). Category = `audio-tagging` (sibling of `panns` / `ast`
    /// / `clap`). Distinct arch tag `yamnet` because the MobileNetV1
    /// depthwise-separable Conv2D backbone differs from residual Cnn14
    /// (PANNs), patch-embed Transformer (AST), and contrastive
    /// text-audio (CLAP) — silently sharing would misroute the runtime
    /// dispatch (FR-EX-08). Scale = **local safe** (~15 MB, well below
    /// vast.ai threshold, memory `[[feedback-large-models-on-vast-ai]]`).
    /// Upstream HF mirror carries no `license:` tag as of 2026-08-13;
    /// reference implementation is Apache-2.0 (Google Research
    /// `tensorflow/models`), so default SPDX = `apache-2.0`. Convert
    /// with `convert_yamnet_file`.
    Yamnet,
    /// **MERT-v1-330M** (`m-a-p/MERT-v1-330M`, cc-by-nc-4.0 default,
    /// music-understanding wave 2026-08-13) — Music undERstanding
    /// model with large-scale self-supervised Training
    /// (HuBERT-derived Conv1D + 24-layer Transformer, ~330M params,
    /// 24 kHz mono waveform in, RVQ-VAE reconstruction target + CQT
    /// teacher). Li et al. 2023 arXiv:2306.00107 = MIREX-benchmark
    /// SoTA for music tagging / similarity / cover-song ID.
    /// Category = `music-embedding` (sibling of `muq` / `dasheng`).
    /// Distinct arch tag `mert` because the HuBERT-derived encoder +
    /// music-specific reconstruction heads differ from Dasheng
    /// (MAE ConvNeXt/ViT) and MuQ (Mel-RVQ + BEATs teacher) —
    /// silently sharing an arch tag would misroute the runtime
    /// dispatch and try to bind an MPM decoder over a MAE checkpoint
    /// (FR-EX-08). Scale = **local safe** (~0.3 GB, well below
    /// vast.ai threshold). License default = `cc-by-nc-4.0` = **T4
    /// tier (NonCommercial)** per X-Codec 2 (2026-07-28) / MusicGen
    /// family (2026-08-01) precedent — publish requires
    /// `publish-one.sh --allow-noncommercial`. Convert with
    /// `convert_mert_file`.
    Mert,
    /// **MuQ** (`OpenMuQ/MuQ-large-msd-iter`, license unknown default,
    /// music-understanding wave 2026-08-13) — Self-supervised music
    /// representation learner using **Mel-Residual Vector
    /// Quantization** targets and a **BEATs** acoustic teacher
    /// (Zhu et al. 2025 arXiv:2501.01108). Trained on the Million
    /// Song Dataset with iterative refinement, ~500M params.
    /// Positioned as a direct MERT alternative for music-tagging /
    /// genre / MIR downstream tasks. Category = `music-embedding`
    /// (sibling of `mert` / `dasheng`). Distinct arch tag `muq`
    /// because the Mel-RVQ + BEATs-teacher training target is a
    /// distinct topology from MERT (HuBERT-derived) and Dasheng
    /// (MAE ConvNeXt/ViT) — silently sharing an arch tag would
    /// misroute the runtime dispatch (FR-EX-08). Scale = **local
    /// safe** (~0.5 GB, well below vast.ai threshold). License
    /// default = `unknown` = **fail-closed** ([`vokra_core::LicenseClass::Unknown`],
    /// M2-13 runtime gate refuses to load without a research flag).
    /// Upstream HF cardData carries no `license:` tag as of
    /// 2026-08-13 — owner must complete primary-source
    /// confirmation before publish is unblocked. Convert with
    /// `convert_muq_file`.
    Muq,
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
            "dac" | "dac-24khz" | "dac-16khz" | "dac-44khz" | "dac-44_1khz"
            | "descript/dac_24khz" | "descript/dac_16khz" | "descript/dac_44khz" => {
                Some(Self::Dac)
            }
            "csm" => Some(Self::Csm),
            // Moshi + Moshika (RAG variant, 2026-07-30 WT7 addition) —
            // moshika-rag shares arch with moshiko-pytorch-bf16 per the RAG
            // fine-tune's HF `base_model` metadata. Category = "s2s".
            "moshi"
            | "moshika"
            | "moshiko"
            | "moshika-rag"
            | "moshika-rag-pytorch-bf16"
            | "moshiko-pytorch-bf16"
            | "moshika-pytorch-bf16"
            | "kyutai/moshika-rag-pytorch-bf16"
            | "kyutai/moshika-pytorch-bf16"
            | "kyutai/moshiko-pytorch-bf16" => Some(Self::Moshi),
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
            // NVIDIA Canary-Qwen family (SoTA plan reuse bundle,
            // 2026-07-30). Accept the canonical arch spelling, the
            // underscore variant, and the release id. The `canary-`
            // prefix walk still catches these for license classification
            // (attribution-required via CC-BY 4.0), so keeping distinct
            // ModelKind arms means the converter dispatches to the
            // Qwen-decoder-flavour arch chunk group rather than the
            // Transformer-AED-flavour Canary chunk group.
            "canary-qwen" | "canary_qwen" | "canary-qwen-2.5b" | "canary-qwen-2_5b"
            | "canary-qwen-2.5B" => Some(Self::CanaryQwen),
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
            // residual wave 4 (2026-08-02): CrisperWhisper —
            // Whisper-large-v3 verbatim-word-timestamps fine-tune under
            // cc-by-nc-4.0. Accept the canonical arch spelling, the
            // underscore variant, and the release id (both cased
            // versions — HF slugs are case-preserving).
            "crisperwhisper"
            | "crisper-whisper"
            | "crisper_whisper"
            | "nyrahealth/crisperwhisper"
            | "nyrahealth/CrisperWhisper" => Some(Self::Crisperwhisper),
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
            // in the block below resolve to the 0.6B checkpoint family; the
            // 1.7B siblings (CustomVoice / VoiceDesign) are distinct
            // `ModelKind` values (`Qwen3TtsCustomVoice17B` /
            // `Qwen3TtsVoiceDesign17B`) below because their talker axes widen
            // (hidden 1024 → 2048, ffn 3072 → 6144). The 0.6B-CustomVoice
            // spelling set added 2026-08-01 (Wave 4 slug-only add) also
            // routes here — the CustomVoice release is a fine-tune of
            // 0.6B-Base with byte-identical talker + code-predictor axes and
            // an identically-shaped CustomVoice head (`config.json.tts_model_type
            // = "custom_voice"`), so the existing 0.6B-Base converter branch
            // covers it verbatim (mirror of the wav2vec2-large-960h-lv60-self
            // slug-only precedent at rows above). The emitted GGUF stamps
            // `vokra.model.name = "qwen3-tts-12hz-0.6b-base"` /
            // `vokra.provenance.upstream_hf = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"`;
            // a future publish of the CustomVoice checkpoint that needs
            // faithful provenance either (a) adds a distinct
            // `Qwen3TtsVariant::_0_6B_CustomVoice` arm to
            // `crates/vokra-convert/src/models/qwen3_tts.rs` so the stamp
            // names this row's upstream repo, or (b) runs a `restamp` pass
            // to rewrite the `vokra.provenance.*` chunks (mirror of the
            // `restamp_provenance` low-memory rewrite path landed 2026-07-23,
            // `crates/vokra-convert/src/lib.rs::restamp_file`).
            "qwen3-tts"
            | "qwen3_tts"
            | "qwen3-tts-0.6b"
            | "qwen3-tts-0_6b"
            | "qwen3-tts-12hz-0.6b-base"
            | "qwen3-tts-12hz-0_6b-base"
            | "qwen3-tts-12hz-0.6b"
            // 2026-08-01 Wave 4 slug-only add: 0.6B-CustomVoice fine-tune
            // (identical axes to 0.6B-Base per approach directive).
            | "qwen3-tts-0.6b-customvoice"
            | "qwen3-tts-0_6b-customvoice"
            | "qwen3-tts-0.6b-custom-voice"
            | "qwen3-tts-12hz-0.6b-customvoice"
            | "qwen3-tts-12hz-0_6b-customvoice"
            | "qwen3-tts-12hz-0.6b-custom-voice"
            | "qwen/qwen3-tts-12hz-0.6b-customvoice" => Some(Self::Qwen3Tts),
            // OpenBMB VoxCPM family — canonical HF releases + arch-tag
            // spellings + common short forms. Both `openbmb/VoxCPM-0.5B`
            // and `openbmb/VoxCPM2` (2B scale-up, land 2026-07-30 —
            // spec `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
            // Option C hybrid) route to the same `ModelKind`; the
            // converter detects the variant from the safetensors payload
            // itself (see `models::voxcpm2::detect_variant`).
            "voxcpm" | "voxcpm2" | "voxcpm-0.5b" | "voxcpm-0_5b" | "voxcpm-0.5b-base"
            | "voxcpm-0_5b-base" | "voxcpm2-0.5b" | "voxcpm2-0_5b" | "voxcpm2-2b"
            | "voxcpm2-2_0b" | "voxcpm2-2b-base" => Some(Self::VoxCpm2),
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
            // Microsoft VibeVoice-Realtime family (streaming variant,
            // 2026-08-01 add). Every spelling routes to the 0.5B
            // release today; a future streaming variant with distinct
            // shape would be a new `ModelKind` when it lands.
            "vibevoice-realtime"
            | "vibevoice_realtime"
            | "vibevoice-realtime-0.5b"
            | "vibevoice-realtime-0_5b"
            | "vibevoice-streaming"
            | "vibevoice_streaming"
            | "microsoft/vibevoice-realtime-0.5b" => Some(Self::VibeVoiceRealtime),
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
            // StyleTTS 2 (yl4579, 2026-07-30) — config-only scaffold.
            // Accept the canonical arch tag, the underscore / space-2
            // variants, and the upstream GitHub / HF coordinates. The
            // registry `LicenseClass::from_id` matches the same
            // `styletts2` / `styletts-2` ids to `LicenseClass::Unknown`
            // (fail-closed under M2-13).
            "styletts2" | "styletts-2" | "styletts_2" | "yl4579/styletts2" | "yl4579/StyleTTS2" => {
                Some(Self::StyleTts2)
            }
            // DeBERTa family (SBV2 v2 plan Task 11, 2026-07-26; v3 alias
            // corrected 2026-07-27, Task 8). Accept the canonical short
            // arch spelling, the underscore variant, and the real HF
            // release id — different orgs for the two variants:
            //   * v2 = `ku-nlp` (Japanese-character `checkpoint.bert_ja`)
            //   * v3 = `microsoft` (English `checkpoint.bert_en`)
            // The `ku-nlp/deberta-v3-...` string is NOT a real HF repo; it
            // was a copy-paste from the v2 arm and is now covered as a
            // negative case in `unknown_model_arg_returns_none`.
            "deberta-v2" | "deberta_v2" | "ku-nlp/deberta-v2-large-japanese-char-wwm" => {
                Some(Self::DebertaV2)
            }
            "deberta-v3" | "deberta_v3" | "microsoft/deberta-v3-large" => Some(Self::DebertaV3),
            // WP-14 (2026-08-10): plain BERT (`hfl/chinese-roberta-wwm-ext-large`,
            // Apache-2.0). Accept the canonical arch tag, the underscore
            // variant, and the real HF release id. First consumer is the
            // SBV2 v2 ZH branch (`SbV2Model::from_gguf_with_zh_bert`);
            // future English WordPiece checkpoints
            // (`google-bert/bert-base-uncased`, ...) also route here — the
            // converter's `--tokenizer` + `do_lower_case` axes serve both.
            "bert-base"
            | "bert_base"
            | "chinese-roberta-wwm-ext-large"
            | "chinese_roberta_wwm_ext_large"
            | "hfl/chinese-roberta-wwm-ext-large" => Some(Self::BertBase),
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
            // HKUSTAudio X-Codec 2 (SoTA plan Phase 5 codec, 2026-07-28).
            // Accept the arch tag (`xcodec2`), the underscore + hyphen +
            // dot variants of the canonical `x-codec-2` name, and the
            // HF release id. Every spelling routes to the same FSQ
            // pass-through converter today; a hypothetical `xcodec3` /
            // `X-Codec-3` would be a distinct `ModelKind` when it lands.
            "xcodec2" | "x-codec-2" | "x_codec_2" | "xcodec-2" | "x-codec2"
            | "hkustaudio-xcodec2" => Some(Self::XCodec2),
            // SoTA plan Phase 5 fleet (2026-07-28): 12 BF16 pass-through
            // skeleton wire-ups. Each entry accepts the arch tag (== the
            // `vokra.model.arch` string the converter stamps, underscore),
            // the CLI-friendly hyphenated spelling, and the canonical HF
            // release id (or its underscore variant) so id lookups return
            // quickly without hitting a future prefix arm.
            "kimi-audio"
            | "kimi_audio"
            | "kimi-audio-7b-instruct"
            | "kimi-audio-7b"
            | "moonshotai/kimi-audio-7b-instruct" => Some(Self::KimiAudio),
            "step-audio2-mini"
            | "step_audio2_mini"
            | "step-audio-2-mini"
            | "stepfun-ai/step-audio-2-mini" => Some(Self::StepAudio2Mini),
            "baichuan-audio" | "baichuan_audio" | "baichuan-inc/baichuan-audio" => {
                Some(Self::BaichuanAudio)
            }
            "speechtokenizer"
            | "speech-tokenizer"
            | "speech_tokenizer"
            | "fnlp/speechtokenizer" => Some(Self::Speechtokenizer),
            "funcodec"
            | "fun-codec"
            | "fun_codec"
            | "funcodec-encodec-zh-en-16k-nq32-ds320"
            | "funcodec-encodec-zh_en" => Some(Self::Funcodec),
            "xy-tokenizer"
            | "xy_tokenizer"
            | "xy-tokenizer-ttsd-v0"
            | "xy_tokenizer_ttsd_v0"
            | "fnlp/xy_tokenizer_ttsd_v0" => Some(Self::XyTokenizer),
            "bicodec"
            | "bi-codec"
            | "bi_codec"
            | "spark-tts-bicodec"
            | "sparkaudio/spark-tts-0.5b" => Some(Self::Bicodec),
            "neucodec"
            | "neu-codec"
            | "neu_codec"
            | "neuphonic/neucodec"
            | "distill-neucodec"
            | "distill_neucodec"
            | "distill-neu-codec"
            | "neuphonic/distill-neucodec" => Some(Self::Neucodec),
            // coverage-audit-2026-08-03 Wave A: NKF-AEC family. Accept
            // the arch tag (underscore == the `vokra.model.arch` string
            // the converter stamps), the CLI-friendly hyphenated
            // spelling, and the canonical GitHub `<user>/<repo>`
            // release id (no HF mirror ships today).
            "nkf-aec" | "nkf_aec" | "fjiang9/nkf-aec" | "fjiang9/NKF-AEC" => Some(Self::NkfAec),
            // Coverage-audit 2026-08-03 Wave A: Xiph RNNoise v0.2.
            // Accept the canonical short arch tag, the versioned publish
            // slug (matches `huggingface.co/vokra/rnnoise-v0.2`), and the
            // upstream `xiph/rnnoise` GitHub-flavour id. Every spelling
            // resolves to the same `Rnnoise` variant today; a future
            // v0.3 would either bump the publish slug in place (still
            // this variant) or introduce a distinct `RnnoiseV03` when
            // the tensor topology reshapes.
            "rnnoise" | "rnnoise-v0.2" | "rnnoise-v0_2" | "xiph/rnnoise" => Some(Self::Rnnoise),
            // Microsoft NSNet2 NR baseline (Coverage-audit 2026-08-03 Wave
            // A). Accept the canonical arch tag, the release id
            // (with-baseline suffix + the on-disk ONNX filename stem), and
            // the "microsoft/nsnet2" org-prefixed alias the ticket lists.
            "nsnet2"
            | "nsnet2-baseline"
            | "nsnet2-20ms"
            | "nsnet2-20ms-baseline"
            | "microsoft/nsnet2"
            | "microsoft/nsnet2-baseline" => Some(Self::Nsnet2),
            // Microsoft DNSMOS bundle (coverage-audit Wave A,
            // 2026-08-03). Accept the short arch tag, the canonical
            // bundle SKU (P.808 + P.835), the underscore variant, and
            // the `microsoft/dnsmos` slug the CLI/orchestrator uses.
            "dnsmos" | "dnsmos-p808-p835" | "dnsmos_p808_p835" | "microsoft/dnsmos" => {
                Some(Self::Dnsmos)
            }
            // coverage-audit wave-a (2026-08-03): FRCRN speech
            // enhancement. Accept the arch tag, the ClearerVoice-Studio
            // release id (dnsmos-style path), the original GitHub
            // repo slug, and the model-scope path used by the
            // ClearerVoice-Studio download logic. Every spelling routes
            // to the same Apache-2.0 Complex U-Net + freq-recurrent
            // LSTM checkpoint.
            "frcrn"
            | "alibabasglab/frcrn"
            | "clearervoice-studio/frcrn"
            | "modelscope/clearervoice-studio-frcrn" => Some(Self::Frcrn),
            // coverage-audit 2026-08-03 Wave B fast-track (13 model)
            "hibiki" | "hibiki-2b" | "hibiki_2b" | "kyutai/hibiki-2b-pytorch-bf16" => {
                Some(Self::Hibiki)
            }
            "sber-gigaam-v3"
            | "sber_gigaam_v3"
            | "gigaam-v3"
            | "gigaam_v3"
            | "ai-sage/gigaam-v3"
            | "ai-sage/GigaAM-v3"
            | "salute-developers/gigaam-v3" => Some(Self::SberGigaamV3),
            "sber-gigaam-multilingual"
            | "sber_gigaam_multilingual"
            | "gigaam"
            | "gigaam-multilingual"
            | "gigaam_multilingual"
            | "ai-sage/gigaam-multilingual"
            | "ai-sage/GigaAM-Multilingual"
            | "salute-developers/gigaam-multilingual"
            | "salute-developers/GigaAM-Multilingual" => Some(Self::SberGigaamMultilingual),
            "reazonspeech-nemo-v2"
            | "reazonspeech_nemo_v2"
            | "reazon-research/reazonspeech-nemo-v2" => Some(Self::ReazonspeechNemoV2),
            "magpietts-v2602" | "magpietts_v2602" | "nvidia/magpietts-v2602" => {
                Some(Self::MagpiettsV2602)
            }
            "parakeet-unified-en-0.6b"
            | "parakeet_unified_en_0_6b"
            | "parakeet-unified"
            | "nvidia/parakeet-unified-en-0.6b" => Some(Self::ParakeetUnified),
            "canary-1b-flash"
            | "canary_1b_flash"
            | "canary-flash"
            | "canary_flash"
            | "canary-1b-flash-en"
            | "nvidia/canary-1b-flash" => Some(Self::Canary1bFlash),
            "owsm-v4-medium-1b"
            | "owsm_v4_medium_1b"
            | "owsm-v4-medium"
            | "espnet/owsm_v4_medium_1b" => Some(Self::OwsmV4Medium1b),
            "parakeet-tdt-1.1b"
            | "parakeet-tdt-1.1B"
            | "parakeet-tdt-1_1b"
            | "parakeet-tdt-1_1B"
            | "parakeet_tdt_1_1b"
            | "nvidia/parakeet-tdt-1.1b" => Some(Self::ParakeetTdt11b),
            "firered-asr-aed-l"
            | "firered_asr_aed_l"
            | "fireredasr-aed-l"
            | "fireredasr_aed_l"
            | "firered-asr-aed"
            | "firered_asr_aed"
            | "fireredteam/firered-asr-aed-l"
            | "fireredteam/firered_asr_aed_l"
            | "fireredteam/fireredasr-aed-l"
            | "FireRedTeam/FireRedASR-AED-L" => Some(Self::FireredAsrAedL),
            "sortformer-diar-4spk-v1"
            | "sortformer_diar_4spk_v1"
            | "nvidia/diar_sortformer_4spk-v1" => Some(Self::SortformerDiar4spkV1),
            "sensevoicesmall" | "sensevoice-small" | "funaudiollm/sensevoicesmall" => {
                Some(Self::SenseVoiceSmall)
            }
            "whisper-medusa-v1" | "whisper_medusa_v1" | "aiola/whisper-medusa-v1" => {
                Some(Self::WhisperMedusaV1)
            }
            // coverage-audit-2026-08-03 Wave D T4: facebook-denoiser
            // (GitHub-only, cc-by-nc-4.0). Distinct arch tag from
            // sibling denoise / rnnoise / nsnet2 / frcrn (each is a
            // distinct speech-enhancement topology).
            "facebook-denoiser"
            | "facebook_denoiser"
            | "denoiser"
            | "demucs-denoiser"
            | "facebookresearch/denoiser" => Some(Self::FacebookDenoiser),
            // coverage-audit-2026-08-03 Wave D T4: NISQA v2 weight
            // (GitHub-only, cc-by-nc-sa-4.0). Category = `eval`,
            // sibling of DNSMOS / UTMOS22-strong.
            "nisqa-v2"
            | "nisqa-v2-weight"
            | "nisqa_v2_weight"
            | "nisqa"
            | "gabrielmittag/nisqa"
            | "gabrielmittag/NISQA" => Some(Self::NisqaV2Weight),
            // coverage-audit-2026-08-03 Wave D T4: 2Noise ChatTTS
            // (HF, cc-by-nc-4.0). Distinct arch tag from sibling
            // TTS families (piper-plus / Kokoro / CosyVoice2 /
            // Chatterbox / xtts).
            "chattts" | "chat-tts" | "2noise/chattts" | "2Noise/ChatTTS" => Some(Self::ChatTts),
            // coverage-audit-2026-08-03 Wave D T4: Stable Audio Open
            // Small (HF, Stability AI Community License = SPDX
            // 未登録 hard-mapped to NonCommercial via CPML precedent).
            "stable-audio-open-small"
            | "stable_audio_open_small"
            | "sao-small"
            | "stabilityai/stable-audio-open-small" => Some(Self::StableAudioOpenSmall),
            // coverage-audit-2026-08-03 Wave D T4: Meta JASCO 400M
            // Chords+Drums (HF, cc-by-nc-4.0).
            "jasco-400m-chords-drums"
            | "jasco_400m_chords_drums"
            | "jasco-400m"
            | "jasco-chords-drums-400m"
            | "facebook/jasco-400m-chords-drums"
            | "facebook/jasco-chords-drums-400M" => Some(Self::Jasco400mChordsDrums),
            // ---- coverage-audit-2026-08-03 Wave A permissive continuation ----
            // (2026-08-04): 7 BF16 pass-through skeletons in the T1
            // (Permissive) tier. Aliases follow the canonical + common
            // spelling + upstream org path convention.
            "utmosv2"
            | "utmos-v2"
            | "utmos_v2"
            | "sarulab-speech/UTMOSv2"
            | "sarulab-speech/utmosv2" => Some(Self::Utmosv2),
            "torchaudio-squim"
            | "torchaudio_squim"
            | "squim"
            | "pytorch/audio" => Some(Self::TorchaudioSquim),
            // Note: bare `"htdemucs"` is claimed by the existing
            // `DemucsHtdemucs` variant (base 4-stem `facebook/demucs`);
            // the HtdemucsMulti (4-source-ft + 6-source-multi variants)
            // uses the more explicit `-ft` / `-6s` / `-4s-6s` / `-multi`
            // spellings to avoid overlap.
            "htdemucs-4s-6s"
            | "htdemucs-4s"
            | "htdemucs-6s"
            | "htdemucs-ft"
            | "htdemucs_4s_6s"
            | "htdemucs-multi"
            | "htdemucs_multi"
            | "facebook/htdemucs_ft"
            | "facebook/htdemucs_6s" => Some(Self::HtdemucsMulti),
            "openwakeword-op"
            | "openwakeword_op"
            | "owwakeword"
            | "dscripka/openWakeWord-op" => Some(Self::OpenwakewordOp),
            "mossformer2-ss-16k"
            | "mossformer2"
            | "mossformer2_ss_16k"
            | "alibabasglab/MossFormer2_SS_16K" => Some(Self::Mossformer2Ss16k),
            "ten-vad"
            | "ten_vad"
            | "TEN-framework/ten-vad"
            | "ten-framework/ten-vad" => Some(Self::TenVad),
            "audioseal-real-weight"
            | "audioseal_real_weight"
            | "audioseal"
            | "audio-seal"
            | "facebook/audioseal" => Some(Self::AudiosealRealWeight),
            // Music-understanding wave (2026-08-13). YAMNet — Google Research
            // 521-class AudioSet audio-event classifier (MobileNetV1 backbone,
            // ~15 MB edge model). apache-2.0 default (Permissive).
            "yamnet"
            | "google-yamnet"
            | "google/yamnet"
            | "thelou1s/yamnet" => Some(Self::Yamnet),
            // Music-understanding wave (2026-08-13). MERT-v1-330M — Music
            // undERstanding model, HuBERT-derived Conv1D + 24-layer
            // Transformer, ~330M params. cc-by-nc-4.0 default (NonCommercial).
            "mert"
            | "mert-v1-330m"
            | "mert-v1"
            | "mert-330m"
            | "m-a-p/mert-v1-330m"
            | "m-a-p/MERT-v1-330M" => Some(Self::Mert),
            // Music-understanding wave (2026-08-13). MuQ — Self-supervised
            // music representation learner with Mel-RVQ + BEATs teacher.
            // License unknown (fail-closed default).
            "muq"
            | "muq-large-msd-iter"
            | "muq-large"
            | "openmuq/muq-large-msd-iter"
            | "OpenMuQ/MuQ-large-msd-iter" => Some(Self::Muq),
            // hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder
            // complement wave (2026-08-04): Aratako/MioCodec-25Hz-44.1kHz-v2.
            // Accept the canonical arch tag, the underscore variant, the
            // versioned publish repo slug (matches `huggingface.co/vokra/
            // miocodec-25hz-44khz-v2`), the upstream HF slug (case-
            // insensitive here — from_arg is case-preserving so the
            // lowercase form is what a shell caller types), and the
            // hyphenated variant with the upstream `44.1khz` dot preserved.
            "miocodec"
            | "mio-codec"
            | "mio_codec"
            | "miocodec-25hz-44khz-v2"
            | "miocodec_25hz_44khz_v2"
            | "miocodec-25hz-44-1khz-v2"
            | "aratako/miocodec-25hz-44.1khz-v2"
            | "aratako/miocodec-25hz-44_1khz-v2" => Some(Self::MioCodec),
            // SoTA plan candidate wave (2026-08-04): Neuphonic NeuTTS Air
            // (apache-2.0). Accept the canonical arch tag / CLI slug (both
            // hyphenated — matches `huggingface.co/vokra/neutts-air`), the
            // underscore variant (== `models::neutts_air` module filename),
            // and the upstream `<org>/<name>` HF slug so a shell caller can
            // paste any spelling the model is discussed under.
            "neutts-air"
            | "neutts_air"
            | "neu-tts-air"
            | "neu_tts_air"
            | "neuphonic/neutts-air" => Some(Self::NeuTtsAir),
            "nemotron-speech-streaming-v2603"
            | "nemotron-speech-streaming"
            | "nemotron_speech_streaming_v2603"
            | "nemotron_speech_streaming"
            | "nvidia/nemotron-speech-streaming-v2603"
            | "nvidia/nemotron_speech_streaming_v2603"
            | "nvidia/nemo/nemotron_speech_streaming_v2603" => {
                Some(Self::NemotronSpeechStreamingV2603)
            }
            "ecapa-tdnn"
            | "ecapa_tdnn"
            | "spkrec-ecapa-voxceleb"
            | "speechbrain/spkrec-ecapa-voxceleb" => Some(Self::EcapaTdnn),
            "wespeaker"
            | "we-speaker"
            | "we_speaker"
            | "wespeaker-voxceleb-resnet34-lm"
            | "wespeaker/wespeaker-voxceleb-resnet34-lm" => Some(Self::Wespeaker),
            "speaker-3d"
            | "speaker_3d"
            | "3d-speaker"
            | "eres2net"
            | "speech_eres2net_sv_zh-cn_16k-common"
            | "iic/speech_eres2net_sv_zh-cn_16k-common" => Some(Self::Speaker3d),
            // NVIDIA TitaNet-Large speaker verification (SoTA follow-on,
            // 2026-07-30). Accept the arch tag underscore + hyphen
            // variants, the short form (family drops the "-large"
            // suffix for callers who match on family name), and the
            // canonical HF release id.
            "titanet-large"
            | "titanet_large"
            | "titanet"
            | "speakerverification_en_titanet_large"
            | "nvidia/speakerverification_en_titanet_large" => Some(Self::TitaNet),
            "emotion2vec"
            | "emotion-2vec"
            | "emotion2vec-plus-large"
            | "emotion2vec/emotion2vec_plus_large" => Some(Self::Emotion2vec),
            // RMVPE (F0 pitch-extractor tier, 2026-07-30). Accept the
            // arch tag, both underscore / hyphen variants of the
            // acronym, the two upstream GitHub coordinates, and the
            // RVC-Boss precursor spelling (the original upstream that
            // yxlllc / Dream-High forked; some checkpoints in the wild
            // still ship under it).
            "rmvpe" | "r-mvpe" | "r_mvpe" | "yxlllc/rmvpe" | "dream-high/rmvpe"
            | "rvc-boss/rmvpe" => Some(Self::Rmvpe),
            // CREPE (Kim et al. 2018, F0 pitch-extractor tier, 2026-07-30 —
            // sibling of RMVPE). Accept the arch tag, each capacity size
            // (upstream ships tiny/small/medium/large/full as separate
            // .h5 files), and the upstream GitHub coordinate.
            "crepe" | "crepe-tiny" | "crepe-small" | "crepe-medium" | "crepe-large"
            | "crepe-full" | "marl/crepe" => Some(Self::Crepe),
            // pyannote/segmentation-3.0 (VAD / speaker-segmentation
            // backbone, 2026-07-30 license half unblock). Accept the
            // arch tag underscore + hyphen variants, the short forms
            // (family drops the "-3.0" suffix for callers who match
            // on family name), and the canonical HF release id.
            "pyannote-segmentation"
            | "pyannote_segmentation"
            | "pyannote-segmentation-3.0"
            | "pyannote_segmentation_3_0"
            | "pyannote-segmentation-3_0"
            | "pyannote/segmentation-3.0"
            | "pyannote/segmentation" => Some(Self::PyannoteSegmentation),
            // pyannote/speaker-diarization-3.1 pipeline (2026-08-01 Wave 5
            // orchestration add). Accept the arch tag underscore / hyphen
            // variants, the short family form (drops the "-3.1" suffix),
            // and the canonical HF release id. Distinct from the
            // segmentation-3.0 backbone above (this arm routes to a
            // weightless pipeline GGUF, not to the PyanNet weights).
            "pyannote-speaker-diarization"
            | "pyannote_speaker_diarization"
            | "pyannote-speaker-diarization-3.1"
            | "pyannote_speaker_diarization_3_1"
            | "pyannote-speaker-diarization-3_1"
            | "pyannote/speaker-diarization-3.1"
            | "pyannote/speaker-diarization" => Some(Self::PyannoteSpeakerDiarization31),
            // 2026-08-01 Wave 4 variant-enum extension: 1.7B-Base — the
            // un-fine-tuned 1.7B backbone that the CustomVoice / VoiceDesign
            // 1.7B siblings fine-tune from. Distinct `Qwen3TtsVariant::_1_7B_Base`
            // arm (rather than slug-only on `_1_7B_CustomVoice`) so a downstream
            // that ships all three 1.7B GGUFs side-by-side can tell them apart
            // by `vokra.provenance.upstream_hf` / `vokra.model.name`. Talker +
            // code-predictor axes are byte-identical to the two 1.7B fine-tuned
            // siblings (hidden=2048, ffn=6144, n_layer=28).
            "qwen3-tts-1.7b-base"
            | "qwen3-tts-1_7b-base"
            | "qwen3-tts-12hz-1.7b-base"
            | "qwen3-tts-12hz-1_7b-base"
            | "qwen3-tts-12hz-1.7b"
            | "qwen3-tts-12hz-1_7b"
            | "qwen/qwen3-tts-12hz-1.7b-base" => Some(Self::Qwen3TtsBase17B),
            "qwen3-tts-1.7b-customvoice"
            | "qwen3-tts-1_7b-customvoice"
            | "qwen3-tts-1.7b-custom-voice"
            | "qwen3-tts-12hz-1.7b-customvoice"
            | "qwen3-tts-12hz-1_7b-customvoice"
            | "qwen3-tts-12hz-1.7b-custom-voice"
            | "qwen/qwen3-tts-12hz-1.7b-customvoice" => Some(Self::Qwen3TtsCustomVoice17B),
            "qwen3-tts-1.7b-voicedesign"
            | "qwen3-tts-1_7b-voicedesign"
            | "qwen3-tts-1.7b-voice-design"
            | "qwen3-tts-12hz-1.7b-voicedesign"
            | "qwen3-tts-12hz-1_7b-voicedesign"
            | "qwen3-tts-12hz-1.7b-voice-design"
            | "qwen/qwen3-tts-12hz-1.7b-voicedesign" => Some(Self::Qwen3TtsVoiceDesign17B),
            "qwen3-asr"
            | "qwen3_asr"
            | "qwen/qwen3-asr-0.6b"
            | "qwen/qwen3-asr-1.7b"
            | "qwen3-asr-0.6b"
            | "qwen3-asr-0_6b"
            | "qwen3_asr_0_6b"
            | "qwen3-asr-1.7b"
            | "qwen3-asr-1_7b"
            | "qwen3_asr_1_7b" => Some(Self::Qwen3Asr),
            "wav2vec2"
            | "wav2vec2_ctc"
            | "wav2vec2-base-960h"
            | "wav2vec2_base_960h"
            | "facebook/wav2vec2-base-960h"
            | "wav2vec2-large-xlsr-53"
            | "wav2vec2_large_xlsr_53"
            | "facebook/wav2vec2-large-xlsr-53"
            | "wav2vec2-large-xlsr-53-japanese"
            | "wav2vec2_large_xlsr_53_japanese"
            | "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
            | "wav2vec2-large-xlsr-53-chinese-zh-cn"
            | "wav2vec2_large_xlsr_53_chinese_zh_cn"
            | "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            // Wave 4 slug-only add (2026-08-01): Facebook wav2vec2 large
            // 960h with self-training on LV60 unlabelled audio
            // (`facebook/wav2vec2-large-960h-lv60-self`, apache-2.0). Same
            // Wav2Vec2ForCTC arch family as the base-960h / xlsr-53 rows
            // above — large topology (24 × d=1024 × 16h × ffn=4096,
            // `feat_extract_norm="layer"`, `do_stable_layer_norm=true`)
            // with an English char CTC head (`vocab_size=32`, same
            // LibriSpeech 960h tokenizer as base-960h). LV60-self is
            // trained with the self-training / pseudo-labelling procedure
            // from Xu et al. 2021 (arXiv:2010.11430) over the LibriVox
            // LV60 corpus, distinct upstream release from XLSR-53.
            //
            // Slug-only routes to the existing
            // `models::wav2vec2_ctc::Variant::LargeXlsr53Base` arm
            // below because that variant already pins the correct large
            // topology axes (24L / 1024h / 16h / 4096ffn +
            // `feat_extract_norm=layer` + `do_stable_layer_norm=true`)
            // and `vocab_size=32` matches LV60-self's English char CTC
            // head. `has_ctc_head` is stored `false` for the XLSR-53
            // base but LV60-self actually carries a CTC head — a future
            // real-weight publish therefore requires either
            // (a) a distinct `Wav2Vec2CtcVariant::Large960hLv60Self`
            // arm added to the converter so `has_ctc_head` +
            // `upstream_hf` faithfully name this row's upstream repo,
            // or (b) a `restamp` pass to rewrite the `vokra.provenance.*`
            // + `vokra.wav2vec2_ctc.has_ctc_head` chunks (mirror of the
            // `restamp_provenance` low-memory rewrite path landed
            // 2026-07-23, `crates/vokra-convert/src/lib.rs::restamp_file`).
            // Slug-only registration is landed together with the §3.1
            // row + `LicenseClass::Permissive` (already covered by the
            // `wav2vec2` prefix walk in
            // `crates/vokra-core/src/compliance/license_class.rs`) so
            // the future publish path only needs the converter arm to
            // close the loop.
            | "wav2vec2-large-960h-lv60-self"
            | "wav2vec2_large_960h_lv60_self"
            | "facebook/wav2vec2-large-960h-lv60-self"
            // Wave 4 variant-enum extension (2026-08-01):
            // `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (apache-2.0).
            // Same XLSR-53 large backbone as the four sibling variants
            // above; the discriminating axis is the CTC head, which
            // over the eSpeak-NG IPA **phoneme** inventory
            // (`vocab_size=392`, arXiv:2109.11680 — CommonVoice
            // fine-tune). vocab_size (392) differs from every existing
            // arm (32 char / 2341 kana+kanji / 3503 hanzi) by 12x+ and
            // the head is `Wav2Vec2ForCTC` (not the ForPreTraining
            // XLSR-53 base), so this cannot be routed slug-only through
            // an existing arm without stamping a demonstrably wrong
            // vocab_size and mis-representing has_ctc_head — a
            // dedicated `models::wav2vec2_ctc::Variant::LargeXlsr53EspeakCvFt`
            // arm carries the correct axes. The phoneme `vocab.json`
            // itself will be embedded as `vokra.tokenizer.model` U8
            // array (Whisper 手法, `include_bytes!` at compile time)
            // in a follow-up wave once the upstream file is
            // snapshotted; today the converter stamps only the axis
            // (`vokra.wav2vec2_ctc.vocab_size = 392`) so a future
            // `Wav2Vec2CtcWeights::from_gguf` reader can loudly reject
            // a mis-sized head (FR-EX-08).
            | "wav2vec2-xlsr-53-espeak-cv-ft"
            | "wav2vec2_xlsr_53_espeak_cv_ft"
            | "facebook/wav2vec2-xlsr-53-espeak-cv-ft" => Some(Self::Wav2Vec2Ctc),
            // 2026-08-02 wave: `facebook/data2vec-audio-base-960h`
            // (apache-2.0). Baevski et al. 2022 (arXiv:2202.03555):
            // wav2vec 2.0 base topology + data2vec pretraining
            // objective + LibriSpeech 960h English char CTC head. The
            // safetensors tensor names are identical to
            // `wav2vec2-base-960h` (data2vec differs in the pretraining
            // objective, not the downstream inference arch), so the
            // wav2vec2 CTC converter covers it verbatim — a distinct
            // `ModelKind` is used only so
            // `vokra.model.name` + `vokra.provenance.upstream_hf`
            // faithfully report the data2vec-audio release. The bare
            // `data2vec-audio-base` slug + the `-960h` variant + the
            // fully-qualified `facebook/data2vec-audio-base-960h` HF
            // repo id all route to the same arm.
            "data2vec-audio-base"
            | "data2vec_audio_base"
            | "data2vec-audio-base-960h"
            | "data2vec_audio_base_960h"
            | "facebook/data2vec-audio-base-960h" => Some(Self::Data2vecAudioBase),
            // 2026-08-02 wave: Meta MMS-1B-All (`facebook/mms-1b-all`,
            // **cc-by-nc-4.0**). Massively Multilingual Speech, Pratap
            // et al. 2023 (arXiv:2305.13516) — 1B wav2vec 2.0 backbone
            // + 1000+ per-language CTC adapters. Base tensor path
            // reuses the [`Self::Wav2Vec2Ctc`] converter via a
            // dedicated `models::wav2vec2_ctc::Variant::Mms1bAll`
            // arm (parent workflow REUSE HINT); the placeholder-axis
            // guardrail (distinct `name` + `upstream_hf`) lets a
            // future `Wav2Vec2CtcWeights::from_gguf` reader detect
            // this artifact and refuse to bind until the follow-up
            // wave transcribes the true MMS-1B topology + lands the
            // per-language adapter loader (~1000 sibling
            // `adapter.*.safetensors` files).
            "mms-1b-all"
            | "mms_1b_all"
            | "mms-1b"
            | "mms_1b"
            | "facebook/mms-1b-all" => Some(Self::Mms1bAll),
            "moss-tts" | "moss_tts" | "moss-tts-delay" | "openmoss-team/moss-tts" => {
                Some(Self::MossTts)
            }
            "moss-tts-v1.5"
            | "moss-tts-v1_5"
            | "moss_tts_v1.5"
            | "moss_tts_v1_5"
            | "openmoss-team/moss-tts-v1.5"
            | "openmoss-team/moss-tts-v1_5" => Some(Self::MossTtsV15),
            "moss-tts-nano"
            | "moss_tts_nano"
            | "moss-tts-nano-100m"
            | "moss_tts_nano_100m"
            | "openmoss-team/moss-tts-nano-100m"
            | "openmoss-team/moss-tts-nano" => Some(Self::MossTtsNano),
            "moss-tts-local"
            | "moss_tts_local"
            | "moss-tts-local-transformer"
            | "moss-tts-local-transformer-v1.5"
            | "moss_tts_local_transformer_v1.5"
            | "moss_tts_local_transformer_v1_5"
            | "openmoss-team/moss-tts-local-transformer-v1.5"
            | "openmoss-team/moss-tts-local-transformer-v1_5" => Some(Self::MossTtsLocal),
            // 2026-08-02 wave: OpenMOSS Team **MOSS-Audio-4B-Instruct**
            // (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0).
            // Distinct 4B audio-LLM sibling of the four `moss_tts_*`
            // tts variants (custom `configuration_moss_audio.py` +
            // `trust_remote_code=True`). Reuses the sibling MossTts
            // converter per the parent workflow's REUSE HINT via the
            // new `models::moss_tts::MossTtsVariant::AudioInstruct4b`
            // arm — see [`Self::MossAudio4bInstruct`] doc for the
            // placeholder-axis + faithful-provenance split.
            "moss-audio-4b-instruct"
            | "moss_audio_4b_instruct"
            | "moss-audio-4b"
            | "moss_audio_4b"
            | "openmoss-team/moss-audio-4b-instruct"
            | "openmoss-team/moss-audio-4b" => Some(Self::MossAudio4bInstruct),
            // 2026-08-02 wave: OpenMOSS Team **MOSS-Audio-8B-Instruct**
            // (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`, apache-2.0).
            // Larger 8B sibling of MOSS-Audio-4B-Instruct with the same
            // `configuration_moss_audio.py` custom-code audio-LLM
            // architecture (4 shards ~9.05 GB BF16 — vast.ai required).
            // Reuses the sibling MossTts converter per the parent
            // workflow's REUSE HINT via the new
            // `models::moss_tts::MossTtsVariant::AudioInstruct8b`
            // arm — see [`Self::MossAudio8bInstruct`] doc for the
            // placeholder-axis + faithful-provenance split.
            "moss-audio-8b-instruct"
            | "moss_audio_8b_instruct"
            | "moss-audio-8b"
            | "moss_audio_8b"
            | "openmoss-team/moss-audio-8b-instruct"
            | "openmoss-team/moss-audio-8b" => Some(Self::MossAudio8bInstruct),
            // 2026-08-01 Wave 4 slug-only add: OpenMOSS Team
            // **MOSS-VoiceGenerator** (`OpenMOSS-Team/MOSS-VoiceGenerator`,
            // apache-2.0). A distinct HF release under the same
            // `moss_tts_delay` internal `model_type` tag as
            // `OpenMOSS-Team/MOSS-TTS`, so the Delay-variant axes
            // (Qwen3-8B backbone, n_vq=32, 24 kHz) already cover it and
            // no new [`MossTtsVariant`] arm is required — the slug is
            // routed to the existing [`Self::MossTts`] dispatch. The
            // §3.1 sign-off row headed
            // `MOSS-VoiceGenerator (\`OpenMOSS-Team/MOSS-VoiceGenerator\`)`
            // is the publish gate that keeps this decision auditable
            // (`scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`
            // maps the `moss-voice-generator` slug to that row).
            //
            // NOTE — provenance stamp caveat: the underlying converter
            // arm writes `vokra.provenance.upstream_hf =
            // OpenMOSS-Team/MOSS-TTS` and `vokra.model.name = moss-tts`
            // from [`MossTtsVariant::Delay`]. A future publish of the
            // MOSS-VoiceGenerator checkpoint therefore requires either
            // (a) a distinct `MossTtsVariant::VoiceGenerator` arm added
            // to the converter so the provenance faithfully names the
            // upstream repo, or (b) a `restamp` pass to rewrite the
            // provenance chunk. Slug-only registration is the parent
            // decision recorded in this file's landing wave; the
            // §3.1 row + `check-catalog-reality.sh` slug alias +
            // `LicenseClass::Permissive` registration are landed together
            // so that the future publish path only needs the converter
            // arm to close the loop.
            "moss-voice-generator"
            | "moss_voice_generator"
            | "moss-voicegenerator"
            | "moss_voicegenerator"
            | "openmoss-team/moss-voice-generator"
            | "openmoss-team/moss-voicegenerator" => Some(Self::MossTts),
            "melotts-english"
            | "melotts_english"
            | "melo-tts-english"
            | "melo-english"
            | "myshell-ai/melotts-english" => Some(Self::MeloTtsEnglish),
            "melotts-chinese"
            | "melotts_chinese"
            | "melo-tts-chinese"
            | "melo-chinese"
            | "myshell-ai/melotts-chinese" => Some(Self::MeloTtsChinese),
            "melotts-korean"
            | "melotts_korean"
            | "melo-tts-korean"
            | "melo-korean"
            | "myshell-ai/melotts-korean" => Some(Self::MeloTtsKorean),
            "melotts-spanish"
            | "melotts_spanish"
            | "melo-spanish"
            | "myshell-ai/melotts-spanish" => Some(Self::MeloTtsSpanish),
            "melotts-japanese"
            | "melotts_japanese"
            | "melo-japanese"
            | "melo-ja"
            | "myshell-ai/melotts-japanese" => Some(Self::MeloTtsJapanese),
            "speecht5-tts" | "speecht5_tts" | "speecht5" | "microsoft/speecht5_tts" => {
                Some(Self::SpeechT5Tts)
            }
            "parler-tts"
            | "parler_tts"
            | "parler-tts-mini-multilingual"
            | "parler-tts-mini-multilingual-v1.1"
            | "parler-tts-mini-multilingual-v1_1"
            | "parler-tts/parler-tts-mini-multilingual-v1.1" => {
                Some(Self::ParlerTtsMiniMultilingual)
            }
            "indic-parler-tts"
            | "indic_parler_tts"
            | "indic-parler"
            | "ai4bharat/indic-parler-tts" => Some(Self::IndicParlerTts),
            // Wave 4 land 2026-08-01: English-only mini-v1 (predecessor of
            // the multilingual v1.1 variant). Same tensor topology, only
            // top-level vocab_size differs (32128 vs 90714 — verified
            // 2026-08-01 from huggingface.co/parler-tts/parler-tts-mini-v1/
            // raw/main/config.json). Distinct ModelKind because the
            // upstream_hf / provenance / vocab_size axis all differ, matching
            // the ParlerTtsMiniMultilingual + IndicParlerTts split pattern.
            "parler-tts-mini-v1"
            | "parler_tts_mini_v1"
            | "parler-mini-v1"
            | "parler-tts/parler-tts-mini-v1" => Some(Self::ParlerTtsMiniV1English),
            "vieneu-tts"
            | "vieneu-tts-v3-turbo"
            | "vieneu_v3_turbo"
            | "vieneu_tts_v3_turbo"
            | "pnnbao-ump/vieneu-tts-v3-turbo" => Some(Self::VieNeuTts),
            "bark" | "suno/bark" | "bark-full" => Some(Self::Bark),
            "bark-small" | "bark_small" | "suno/bark-small" => Some(Self::BarkSmall),
            "hifigan-vocoder"
            | "hifigan_vocoder"
            | "hifigan"
            | "tts-hifigan-libritts-22050hz"
            | "speechbrain/tts-hifigan-libritts-22050hz" => Some(Self::HifiganVocoder),
            "speecht5-hifigan"
            | "speecht5_hifigan"
            | "speecht5-vocoder"
            | "speecht5_vocoder"
            | "microsoft/speecht5_hifigan" => Some(Self::Speecht5Hifigan),
            "bigvgan"
            | "big-vgan"
            | "big_vgan"
            | "bigvgan-v2-22khz-80band-256x"
            | "bigvgan_v2_22khz_80band_256x"
            | "nvidia/bigvgan_v2_22khz_80band_256x"
            | "bigvgan-v2-44khz-128band-512x"
            | "bigvgan_v2_44khz_128band_512x"
            | "nvidia/bigvgan_v2_44khz_128band_512x"
            | "bigvgan-v2-24khz-100band-256x"
            | "bigvgan_v2_24khz_100band_256x"
            | "nvidia/bigvgan_v2_24khz_100band_256x"
            | "bigvgan-base-24khz-100band"
            | "bigvgan_base_24khz_100band"
            | "nvidia/bigvgan_base_24khz_100band" => Some(Self::BigVGan),
            "focalcodec"
            | "focal-codec"
            | "focal_codec"
            | "focalcodec-50hz"
            | "focalcodec_50hz"
            | "lucadellalib/focalcodec_50hz"
            | "focalcodec-25hz"
            | "focalcodec_25hz"
            | "lucadellalib/focalcodec_25hz"
            | "focalcodec-12-5hz"
            | "focalcodec-12_5hz"
            | "focalcodec_12_5hz"
            | "lucadellalib/focalcodec_12_5hz" => Some(Self::Focalcodec),
            // SNAC — Multi-Scale Neural Audio Codec (Siuzdak et al. 2024,
            // hubertsiuzdak/snac_{24khz,44khz}, MIT). Two variants share
            // this single ModelKind; the slug picks the frame rate + RVQ
            // depth via `convert_file_with_slug` (mirror of BigVGan /
            // Focalcodec's slug dispatch).
            "snac"
            | "snac-24khz"
            | "snac_24khz"
            | "hubertsiuzdak/snac_24khz"
            | "snac-44khz"
            | "snac_44khz"
            | "hubertsiuzdak/snac_44khz" => Some(Self::Snac),
            // WavTokenizer-large-speech-75token — MIT single-codebook FSQ codec.
            "wavtokenizer"
            | "wavtokenizer-large-speech-75token"
            | "wavtokenizer_large_speech_75token"
            | "novateur/wavtokenizer-large-speech-75token" => Some(Self::Wavtokenizer),
            "granite-speech"
            | "granite-speech-4.1-2b"
            | "granite_speech_4_1_2b"
            | "granite-speech-4-1-2b"
            | "ibm-granite/granite-speech-4.1-2b" => Some(Self::GraniteSpeech),
            // MOSS-Audio-Tokenizer — the codec half of the MOSS-TTS
            // pipeline (2026-08-01 Wave 3, OpenMOSS-Team, apache-2.0).
            // Two variants share this single ModelKind; the slug picks
            // Full vs Nano via `convert_file_with_slug` (mirror of
            // Snac / Focalcodec / BigVGan slug dispatch). Accept the
            // canonical short arch tag, both per-variant slugs (Full
            // canonical / Nano second-variant), the underscore
            // variants, and the raw HF org/name paths.
            "moss-audio-tokenizer"
            | "moss_audio_tokenizer"
            | "moss-audio-tokenizer-full"
            | "moss_audio_tokenizer_full"
            | "openmoss-team/moss-audio-tokenizer"
            | "moss-audio-tokenizer-nano"
            | "moss_audio_tokenizer_nano"
            | "openmoss-team/moss-audio-tokenizer-nano" => Some(Self::MossAudioTokenizer),
            // Amphion NaturalSpeech 3 FACodec — factorized VQ codec
            // (2026-08-01 Wave 3, apache-2.0). Four variants share
            // this single ModelKind; the slug picks the encoder+decoder
            // pair (+ optional redecoder) via `convert_file_with_slug`
            // (mirror of Snac / MossAudioTokenizer slug dispatch).
            // Accept the canonical short arch tag, the family name
            // spellings, and the raw HF repo id.
            "facodec"
            | "naturalspeech3-facodec"
            | "ns3-facodec"
            | "ns3_facodec"
            | "amphion-facodec"
            | "naturalspeech3_facodec"
            | "amphion/naturalspeech3_facodec"
            | "naturalspeech3-facodec-v1"
            | "naturalspeech3-facodec-v2"
            | "naturalspeech3-facodec-redecoder-v1"
            | "naturalspeech3-facodec-redecoder-v2"
            | "facodec-v1"
            | "facodec-v2"
            | "facodec-redecoder-v1"
            | "facodec-redecoder-v2" => Some(Self::Facodec),
            "tiger"
            | "tiger-dnr"
            | "tiger_dnr"
            | "tiger-separator"
            | "tiger_separator"
            | "jusperlee/tiger-dnr" => Some(Self::TigerSeparator),
            "tiger-speech" | "tiger_speech" | "jusperlee/tiger-speech" => Some(Self::TigerSpeech),
            "mp-senet" | "mp_senet" | "mpsenet" => Some(Self::MpSenet),
            "mp-senet-dns" | "mp_senet_dns" | "mpsenet-dns" | "jacoblincool/mp-senet-dns" => {
                Some(Self::MpSenetDns)
            }
            "metricgan-plus"
            | "metricgan_plus"
            | "metricganplus"
            | "metricgan-plus-voicebank"
            | "metricgan_plus_voicebank"
            | "speechbrain/metricgan-plus-voicebank" => Some(Self::MetricganPlus),
            "sepformer"
            | "sepformer-wsj02mix"
            | "sepformer_wsj02mix"
            | "sepformer-wsj0-2mix"
            | "speechbrain/sepformer-wsj02mix" => Some(Self::SepFormer),
            "sepformer-wham16k"
            | "sepformer-wham16k-enhancement"
            | "sepformer_wham16k_enhancement"
            | "speechbrain/sepformer-wham16k-enhancement" => Some(Self::SepformerWham16kEnh),
            "sepformer-whamr16k"
            | "sepformer_whamr16k"
            | "sepformer-whamr"
            | "speechbrain/sepformer-whamr16k" => Some(Self::SepformerWhamr16k),
            "sepformer-libri2mix"
            | "sepformer_libri2mix"
            | "sepformer-libri-2mix"
            | "speechbrain/sepformer-libri2mix" => Some(Self::SepformerLibri2Mix),
            "sepformer-libri3mix"
            | "sepformer_libri3mix"
            | "sepformer-libri-3mix"
            | "speechbrain/sepformer-libri3mix" => Some(Self::SepformerLibri3Mix),
            // NOTE: `sepformer-whamr` (bare short alias) historically
            // routes to `SepformerWhamr16k` above and is intentionally
            // preserved for backwards compatibility. The 8 kHz sibling
            // must be selected via one of the explicit -8khz suffix
            // aliases or the full upstream HF slug
            // `speechbrain/sepformer-whamr` (which is unambiguous
            // upstream = the 8 kHz repo). This avoids silently
            // flipping the semantics of `--model sepformer-whamr` for
            // existing callers.
            "sepformer-whamr-8khz"
            | "sepformer_whamr_8khz"
            | "sepformer-whamr-8k"
            | "sepformer_whamr_8k"
            | "sepformer-whamr8k"
            | "sepformer_whamr8k"
            | "speechbrain/sepformer-whamr" => Some(Self::SepformerWhamr8k),
            "sepformer-dns4-16k-enhancement"
            | "sepformer_dns4_16k_enhancement"
            | "sepformer-dns4-enhancement"
            | "sepformer_dns4_enhancement"
            | "sepformer-dns4"
            | "sepformer_dns4"
            | "speechbrain/sepformer-dns4-16k-enhancement" => Some(Self::SepformerDns4Enh),
            // SoTA plan candidate wave (2026-08-04): SpeechBrain
            // SGMSE-VoiceBank (apache-2.0, score-based diffusion speech
            // enhancement, first real-weight consumer of the M3-05
            // flow_sampler + ODE solver op family). Accept the canonical
            // CLI slug + arch tag `sgmse` (short), the VoiceBank-tuned
            // spelling `sgmse-voicebank` (matches
            // `huggingface.co/vokra/sgmse-voicebank` publish target),
            // underscore variants (mirroring the module filename), and
            // the upstream `<org>/<name>` HF slug.
            "sgmse"
            | "sgmse-voicebank"
            | "sgmse_voicebank"
            | "sgmse-voice-bank"
            | "sgmse_voice_bank"
            | "speechbrain/sgmse-voicebank" => Some(Self::Sgmse),
            "fsmn-vad"
            | "fsmn_vad"
            | "fsmnvad"
            | "fsmn-vad-zh-cn-16k-common"
            | "funasr/fsmn-vad"
            | "funaudiollm/fsmn-vad-gguf"
            | "fsmn-vad-gguf"
            | "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" => Some(Self::FsmnVad),
            "firered-vad" | "firered_vad" | "fireredvad" | "fireredteam/fireredvad" => {
                Some(Self::FireredVad)
            }
            "smart-turn"
            | "smart_turn"
            | "smart-turn-v2"
            | "smart_turn_v2"
            | "pipecat-ai/smart-turn-v2" => Some(Self::SmartTurn),
            "clap" | "clap-htsat-fused" | "clap_htsat_fused" | "laion/clap-htsat-fused" => {
                Some(Self::Clap)
            }
            "ast"
            | "audio-spectrogram-transformer"
            | "ast-finetuned-audioset"
            | "ast-finetuned-audioset-10-10-0.4593"
            | "mit/ast-finetuned-audioset-10-10-0.4593" => Some(Self::Ast),
            "lang-id-voxlingua107"
            | "lang_id_voxlingua107"
            | "lang-id-voxlingua107-ecapa"
            | "lang_id_voxlingua107_ecapa"
            | "speechbrain/lang-id-voxlingua107-ecapa" => Some(Self::LangIdVoxlingua107),
            "lang-id-commonlanguage"
            | "lang_id_commonlanguage"
            | "lang-id-commonlanguage-ecapa"
            | "lang_id_commonlanguage_ecapa"
            | "speechbrain/lang-id-commonlanguage_ecapa" => Some(Self::LangIdCommonLanguage),
            "xvector"
            | "x-vector"
            | "x_vector"
            | "spkrec-xvect-voxceleb"
            | "spkrec_xvect_voxceleb"
            | "speechbrain/spkrec-xvect-voxceleb" => Some(Self::XVector),
            "deepfake-detection"
            | "deepfake_detection"
            | "deepfake-audio-detection"
            | "deepfake-audio-detection-v2"
            | "melodymachine/deepfake-audio-detection-v2" => Some(Self::DeepfakeDetection),
            "kyutai-tts"
            | "kyutai_tts"
            | "kyutai-tts-1.6b"
            | "kyutai-tts-1.6b-en-fr"
            | "kyutai-tts-1.6b-en_fr"
            | "kyutai-tts-1_6b"
            | "kyutai-tts-1_6b-en-fr"
            | "kyutai-tts-1_6b-en_fr"
            | "kyutai/tts-1.6b-en_fr"
            | "tts-1.6b-en_fr" => Some(Self::KyutaiTts),
            "audiobox-aesthetics"
            | "audiobox_aesthetics"
            | "audiobox"
            | "facebook/audiobox-aesthetics" => Some(Self::AudioboxAesthetics),
            "voxtral-mini-4b-realtime-2602"
            | "voxtral-mini-4b-realtime"
            | "voxtral-realtime"
            | "voxtral-realtime-2602"
            | "voxtral_realtime"
            | "voxtral_realtime_2602"
            | "mistralai/voxtral-mini-4b-realtime-2602" => Some(Self::VoxtralMiniRealtime),
            "cohere-transcribe"
            | "cohere-transcribe-03-2026"
            | "cohere_transcribe"
            | "cohere_transcribe_03_2026"
            | "coherelabs/cohere-transcribe-03-2026" => Some(Self::CohereTranscribe),
            "nemotron-asr-streaming"
            | "nemotron-3.5-asr-streaming"
            | "nemotron-3.5-asr-streaming-0.6b"
            | "nemotron-3_5-asr-streaming-0_6b"
            | "nvidia/nemotron-3.5-asr-streaming-0.6b" => Some(Self::NemotronAsrStreaming),
            // CNChTu FCPE — Fast Context-based Pitch Estimator (MIT). Accept
            // the canonical short arch tag, the family name in both underscore
            // and hyphen spellings, and the full upstream GitHub slug. Every
            // spelling routes to the same 360-bin pitch classifier converter
            // today; a future FCPE_v002 would be a distinct `ModelKind` if its
            // config axes reshape the Conformer body (task hint: 4-6 layers so
            // the range is genuinely open).
            "fcpe"
            | "torchfcpe"
            | "fast-context-pitch-estimator"
            | "fast_context_pitch_estimator"
            | "cnchtu/fcpe" => Some(Self::Fcpe),
            // Charactr AI Vocos family (2026-08-01 wave). Accept the
            // canonical short arch tag, the underscore variant, both
            // per-variant slugs (mel-24khz canonical / encodec-24khz
            // second-variant), and the raw HF org/name paths. Every
            // spelling resolves to the same MIT ModelKind; the specific
            // VocosVariant is picked from the raw slug in
            // convert_file_with_slug (mirror of BigVGan / Focalcodec).
            "vocos"
            | "vocos-mel-24khz"
            | "vocos_mel_24khz"
            | "vocos-mel"
            | "charactr/vocos-mel-24khz"
            | "vocos-encodec-24khz"
            | "vocos_encodec_24khz"
            | "vocos-encodec"
            | "charactr/vocos-encodec-24khz" => Some(Self::Vocos),
            // 2026-08-01 Wave 3 sibling-pair add: YuE bundle
            // (`m-a-p/YuE-upsampler` + `m-a-p/xcodec_mini_infer`,
            // apache-2.0). Two distinct HF repos → two distinct
            // ModelKind entries (INTENTIONALLY not collapsed into
            // one ModelKind + variant slug dispatch: these are two
            // independent HF org publishes with different scopes,
            // not two frontends of a single release). Accept the
            // canonical short arch tag, hyphen / underscore
            // spellings, and the raw HF org/name paths for each.
            "yue-upsampler"
            | "yue_upsampler"
            | "map-yue-upsampler"
            | "m-a-p-yue-upsampler"
            | "m-a-p/yue-upsampler" => Some(Self::YueUpsampler),
            "yue-xcodec-mini"
            | "yue_xcodec_mini"
            | "yue-xcodec-mini-infer"
            | "yue_xcodec_mini_infer"
            | "xcodec-mini"
            | "xcodec_mini"
            | "xcodec-mini-infer"
            | "xcodec_mini_infer"
            | "map-xcodec-mini"
            | "m-a-p-xcodec-mini"
            | "yue-codec"
            | "m-a-p/xcodec_mini_infer" => Some(Self::YueXcodecMini),
            // Meta AudioCraft MusicGen-Medium (Wave 5 candidate,
            // 2026-08-01, `facebook/musicgen-medium`, cc-by-nc-4.0). Accept
            // the arch tag (`musicgen`), the arch+size spellings (hyphen
            // + underscore variants), and the canonical HF release id.
            // Every spelling routes to the same NonCommercial BF16
            // pass-through converter today; future family variants
            // (`musicgen-small` / `musicgen-large` / `musicgen-melody` /
            // `musicgen-stereo-*`) will be distinct `ModelKind` values
            // (sibling files per the chatterbox / chatterbox_turbo split,
            // or a shared `musicgen.rs` variant enum per the snac / vocos
            // split — decided when a second variant lands).
            "musicgen"
            | "musicgen-medium"
            | "musicgen_medium"
            | "facebook-musicgen-medium"
            | "facebook/musicgen-medium" => Some(Self::MusicGenMedium),
            // Meta AudioCraft MusicGen-Large (Wave 5 candidate,
            // 2026-08-01, `facebook/musicgen-large`, cc-by-nc-4.0).
            // Second MusicGen family member — top rung 3.3B. Bare
            // arch tag `musicgen` stays owned by
            // [`Self::MusicGenMedium`] (first-landed family default);
            // callers who want the -large variant must be explicit
            // via `-large` / `_large` / `facebook-musicgen-large` /
            // `facebook/musicgen-large`. Same family + same arch tag
            // as sibling MusicGen-Medium — silently sharing the bare
            // arch tag `musicgen` would create a routing ambiguity
            // between two distinct HF repos, so this arm requires an
            // explicit `-large` suffix.
            "musicgen-large"
            | "musicgen_large"
            | "facebook-musicgen-large"
            | "facebook/musicgen-large" => Some(Self::MusicGenLarge),
            // Meta AudioCraft MusicGen-Melody (Wave 5 sibling,
            // 2026-08-02, `facebook/musicgen-melody`, cc-by-nc-4.0).
            // Medium 1.5B LM + 12-bin chromagram conditioning frontend
            // (byte-identical LM topology to sibling MusicGenMedium,
            // only the conditioning frontend + projection differ).
            // Reuses the medium converter arm via the sibling wrapper
            // `convert_musicgen_melody_file` — no new
            // `musicgen_melody.rs` module. Same bare `musicgen` arch
            // tag stays owned by [`Self::MusicGenMedium`] (first-
            // landed family default); this arm requires the explicit
            // `-melody` suffix to disambiguate.
            "musicgen-melody"
            | "musicgen_melody"
            | "facebook-musicgen-melody"
            | "facebook/musicgen-melody" => Some(Self::MusicGenMelody),
            // AudioGen-Medium (Wave 5 residual, 2026-08-01,
            // `facebook/audiogen-medium`, cc-by-nc-4.0). MusicGen sibling
            // (identical `musicgen` arch tag, tuned on environmental
            // sounds / SFX). Slug alias set mirrors MusicGen-Medium.
            "audiogen"
            | "audiogen-medium"
            | "audiogen_medium"
            | "facebook-audiogen-medium"
            | "facebook/audiogen-medium" => Some(Self::AudioGenMedium),
            // MusicGen-Small (Wave 6 residual, 2026-08-01,
            // `facebook/musicgen-small`, cc-by-nc-4.0). 300M smallest of
            // MusicGen family. Shared `musicgen` arch + `music` category.
            "musicgen-small"
            | "musicgen_small"
            | "facebook-musicgen-small"
            | "facebook/musicgen-small" => Some(Self::MusicGenSmall),
            // Qwen2-Audio-7B-Instruct (Wave 6 residual, 2026-08-01).
            "qwen2-audio"
            | "qwen2-audio-7b"
            | "qwen2-audio-7b-instruct"
            | "qwen2_audio"
            | "qwen2_audio_7b_instruct"
            | "qwen/qwen2-audio-7b-instruct"
            | "Qwen/Qwen2-Audio-7B-Instruct" => Some(Self::Qwen2Audio),
            // Qwen2.5-Omni-7B (Wave residual, 2026-08-02). Thinker +
            // Talker unified any-to-any omni multimodal LLM over
            // Qwen2.5-7B backbone. Kebab-case arch tag `qwen2-omni`
            // (distinct from audio-only sibling `qwen2_audio`).
            "qwen2-5-omni-7b"
            | "qwen2_5_omni_7b"
            | "qwen2.5-omni-7b"
            | "qwen2-omni"
            | "qwen2-5-omni"
            | "qwen/qwen2.5-omni-7b"
            | "Qwen/Qwen2.5-Omni-7B" => Some(Self::Qwen25Omni7b),
            // VibeVoice-ASR (Wave 6 residual, 2026-08-01).
            "vibevoice-asr"
            | "vibevoice_asr"
            | "microsoft-vibevoice-asr"
            | "microsoft/VibeVoice-ASR" => Some(Self::VibeVoiceAsr),
            // ACE-Step 1.5 (Wave 6 residual, 2026-08-01).
            "ace-step"
            | "ace-step-1.5"
            | "ace-step-1_5"
            | "ace_step"
            | "ace_step_1_5"
            | "ACE-Step/Ace-Step1.5" => Some(Self::AceStep),
            // HuBERT-Large-LS960 (Wave 7 residual, 2026-08-01,
            // `facebook/hubert-large-ls960-ft`, apache-2.0). 317M
            // BERT-style masked-feature-prediction speech encoder + CTC
            // head fine-tuned on LibriSpeech 960h. Distinct arch tag
            // `hubert` — silently sharing with wav2vec2 would misroute
            // runtime dispatch (different pretraining objective / loader).
            "hubert-large-ls960"
            | "hubert_large_ls960"
            | "hubert-large-ls960-ft"
            | "hubert_large_ls960_ft"
            | "facebook-hubert-large-ls960-ft"
            | "facebook/hubert-large-ls960-ft" => Some(Self::HubertLargeLs960),
            // w2v-BERT 2.0 (hf-audio-gap SSL residual, 2026-08-04,
            // `facebook/w2v-bert-2.0`, MIT). ~580M-parameter SSL speech
            // encoder = Conformer body + dual contrastive + MLM branches.
            // Distinct arch tag `w2v-bert-2` from siblings hubert /
            // wav2vec2_ctc / data2vec-audio (Conformer vs vanilla
            // Transformer body + combined SSL objectives) — silently
            // sharing would mis-route runtime dispatch (FR-EX-08).
            // Accept the arch tag (hyphen / underscore / dot variants),
            // the family-name spellings, and the canonical HF release id.
            "w2v-bert-2"
            | "w2v_bert_2"
            | "w2v-bert-2-0"
            | "w2v_bert_2_0"
            | "w2v-bert-2.0"
            | "w2vbert2"
            | "w2vbert-2"
            | "w2v-bert"
            | "wav2vec2-bert"
            | "wav2vec2_bert"
            | "facebook-w2v-bert-2-0"
            | "facebook-w2v-bert-2.0"
            | "facebook/w2v-bert-2.0" => Some(Self::W2vBert2),
            // AudioLDM 2 (Wave 5 candidate, 2026-08-01,
            // `cvssp/audioldm2`, **cc-by-nc-sa-4.0**). Accept the arch
            // tag (`audioldm2` / `audio-ldm-2` / `audio_ldm_2`), the
            // underscore variants, and the canonical HF release id.
            // Every spelling routes to the same
            // NonCommercialShareAlike BF16 pass-through converter
            // today; future family variants (`audioldm2-music` /
            // `audioldm2-large` / `audioldm2-music-665k`) will be
            // distinct `ModelKind` values (sibling files per the
            // musicgen_medium / musicgen_large split, or a shared
            // `audioldm2.rs` variant enum per the snac / vocos split —
            // decided when a second variant lands).
            "audioldm2"
            | "audio-ldm-2"
            | "audio_ldm_2"
            | "audioldm-2"
            | "audioldm_2"
            | "cvssp-audioldm2"
            | "cvssp/audioldm2" => Some(Self::AudioLdm2),
            // AudioLDM 2 Large (Wave 8 sibling, 2026-08-02,
            // `cvssp/audioldm2-large`, cc-by-nc-sa-4.0). Distinct
            // ModelKind (mirrors musicgen_medium / musicgen_large
            // split posture) so the runtime-side lookup surface
            // stays one-model-per-ModelKind — future variants
            // (`audioldm2-music` / `audioldm2-music-665k`) will
            // land as distinct ModelKinds under the same shared
            // `audioldm2` arch tag.
            "audioldm2-large"
            | "audio-ldm-2-large"
            | "audio_ldm_2_large"
            | "audioldm-2-large"
            | "audioldm_2_large"
            | "cvssp-audioldm2-large"
            | "cvssp/audioldm2-large" => Some(Self::AudioLdm2Large),
            // BS-Roformer / Mel-Band Roformer (Wave 5 candidate,
            // 2026-08-01, `chenmozhijin/BSRoformer-GGUF`, weight
            // provenance unclear). Accept the arch tag (both
            // underscore and hyphen), the family-name spellings
            // (`bs-roformer` / `bsroformer` / `mel-band-roformer` /
            // `melband-roformer`), and the third-party HF mirror
            // slug. Every spelling routes to the same
            // `LicenseClass::RedistributionForbidden` fail-closed
            // converter today (single-variant standalone; a future
            // variant enum landing would extend from here). The
            // shorthand `mel-band-roformer` alias covers the Mel-Band
            // family sub-variant Lu et al. 2023 describes alongside
            // BS-Roformer — same arch tag `bs_roformer` because the
            // topology is identical (band-split module vs mel-filter-
            // bank module is a runtime hparam, not a distinct arch);
            // a downstream who needs to disambiguate can inspect
            // `vokra.model.name` or a future `vokra.bs_roformer.
            // variant` chunk.
            "bs-roformer"
            | "bs_roformer"
            | "bsroformer"
            | "mel-band-roformer"
            | "mel_band_roformer"
            | "melband-roformer"
            | "melband_roformer"
            | "chenmozhijin/bsroformer-gguf" => Some(Self::BsRoformer),
            // 2026-08-02 Wave residual: openWakeWord (dscripka,
            // apache-2.0). Small custom-KWS MLP/CNN family (~1–5 MB
            // per wake-word) — audio-dialect `kws` op entry.
            // Distinct arch tag `openwakeword`. Accept the arch tag
            // and canonical HF release id.
            "openwakeword"
            | "open-wakeword"
            | "open_wakeword"
            | "dscripka-openwakeword"
            | "dscripka/openwakeword"
            | "dscripka/openWakeWord" => Some(Self::Openwakeword),
            // 2026-08-02 Wave residual: Moonshine-Tiny (UsefulSensors,
            // MIT). 27M raw-audio transformer enc-dec ASR (arXiv:
            // 2410.15608). Distinct arch tag `moonshine`. Accept the
            // arch tag, the family-name spelling, hyphen / underscore
            // variants, and the canonical HF org/name path.
            "moonshine"
            | "moonshine-tiny"
            | "moonshine_tiny"
            | "usefulsensors-moonshine-tiny"
            | "usefulsensors/moonshine-tiny"
            | "UsefulSensors/moonshine-tiny" => Some(Self::MoonshineTiny),
            // 2026-08-02 Wave residual: Moonshine-Base (UsefulSensors,
            // MIT). 61.5M raw-audio transformer enc-dec ASR (arXiv:
            // 2410.15608). Sibling to Moonshine-Tiny — same arch family
            // (raw-audio Conv1D + rotary + SwiGLU), wider/deeper
            // backbone. Shared arch tag `moonshine` at the runtime side,
            // distinct ModelKind at the converter side (the two
            // checkpoints have different tensor shapes; the dispatch
            // must not silently pick Tiny for a Base checkpoint or vice
            // versa — FR-EX-08). Accept the family-name spelling with
            // the `-base` suffix, hyphen / underscore variants, and the
            // canonical HF org/name path.
            "moonshine-base"
            | "moonshine_base"
            | "usefulsensors-moonshine-base"
            | "usefulsensors/moonshine-base"
            | "UsefulSensors/moonshine-base" => Some(Self::MoonshineBase),
            // 2026-08-02 Wave residual: Demucs (HT-Demucs) (facebook/demucs,
            // MIT). Hybrid transformer Demucs (Rouard et al. 2023, arXiv:
            // 2211.08553) — U-Net waveform branch + spectrogram branch +
            // cross-domain self-attention, 4-source music separation
            // (drums / bass / other / vocals). Distinct arch tag `demucs`
            // from sibling SepFormer / TIGER separator siblings (different
            // internal domain + different output branching — FR-EX-08
            // forbids silent misroute across separator families). Accept
            // the canonical name spelling with hyphen / underscore
            // variants + the HF org/name path.
            "demucs-htdemucs"
            | "demucs_htdemucs"
            | "htdemucs"
            | "ht-demucs"
            | "facebook-demucs"
            | "facebook/demucs" => Some(Self::DemucsHtdemucs),
            // 2026-08-02 Wave residual: Ultravox v0.5 (Llama-3.2-1B)
            // (`fixie-ai/ultravox-v0_5-llama-3_2-1b`, MIT). Audio-text-to-
            // text multimodal = Llama-3.2-1B decoder + Whisper encoder +
            // projection adapter. Distinct arch tag `ultravox` from sibling
            // Voxtral / Qwen2-Audio (different decoder backbone fixes
            // tensor layout + tokenizer + rope base — FR-EX-08 forbids
            // silent shape misroute). Accept the arch tag (both underscore
            // and hyphen), the family-name spellings, and the canonical HF
            // release id.
            "ultravox"
            | "ultravox-v0-5-llama-3-2-1b"
            | "ultravox_v0_5_llama_3_2_1b"
            | "ultravox-v0_5-llama-3_2-1b"
            | "fixie-ai-ultravox-v0_5-llama-3_2-1b"
            | "fixie-ai/ultravox-v0_5-llama-3_2-1b" => Some(Self::UltravoxV05Llama321b),
            // 2026-08-02 Wave residual: XTTS-v2 (`coqui/XTTS-v2`,
            // coqui-public-model-license). Multilingual zero-shot voice-
            // cloning TTS = GPT-2 backbone + DVAE + HiFi-GAN. Distinct
            // arch tag `xtts` — silently sharing with sibling piper-plus /
            // Kokoro / CosyVoice2 would misroute runtime dispatch
            // (different tokenizer + different decoder head + different
            // vocoder), which FR-EX-08 forbids. Accept the arch tag,
            // family-name spellings, hyphen / underscore variants, and
            // the canonical HF org/name path.
            "xtts"
            | "xtts-v2"
            | "xtts_v2"
            | "xttsv2"
            | "coqui-xtts-v2"
            | "coqui/xtts-v2"
            | "coqui/XTTS-v2" => Some(Self::XttsV2),
            // ConvTasNet Libri1Mix Enhancement (Asteroid, 2026-08-02 Wave
            // residual, cc-by-sa-4.0). First Copyleft-tier separator arm.
            // Aliases cover the kebab-case and underscore spellings + the
            // full upstream HF slug (case-insensitive lookup handled by
            // whatever normalisation the caller applies before dispatch —
            // both lower-case and the upstream mixed-case form land the
            // same variant).
            "conv-tasnet-libri1mix"
            | "conv_tasnet_libri1mix"
            | "convtasnet-libri1mix"
            | "convtasnet_libri1mix"
            | "conv-tasnet-libri1mix-enhsingle-16k"
            | "conv_tasnet_libri1mix_enhsingle_16k"
            | "joriscos/convtasnet_libri1mix_enhsingle_16k"
            | "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k" => Some(Self::ConvTasnetLibri1mix),
            // Seamless-M4T-v2-Large (Wave residual, 2026-08-02,
            // `facebook/seamless-m4t-v2-large`, cc-by-nc-4.0). 2.3B unified
            // any-to-any speech-and-text translation, unity-2 arch (4
            // subgraphs: w2v-BERT enc + text dec + T2U + HiFi-GAN vocoder).
            // Accept the arch tag (`unity-2` / `unity_2` / `unity2`), the
            // model-id spellings (hyphen + underscore + compact variants),
            // the full upstream HF slug, and the Meta-repo alias.
            "seamless-m4t-v2-large"
            | "seamless_m4t_v2_large"
            | "seamlessm4t-v2-large"
            | "seamlessm4t_v2_large"
            | "seamless-m4t-v2"
            | "seamless_m4t_v2"
            | "unity-2"
            | "unity_2"
            | "unity2"
            | "facebook-seamless-m4t-v2-large"
            | "facebook/seamless-m4t-v2-large" => Some(Self::SeamlessM4tV2Large),
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
            Self::CanaryQwen => "canary-qwen",
            Self::OmniasrCtc => "omniasr-ctc",
            Self::DistilWhisper => "distil-whisper",
            Self::KotobaWhisper => "kotoba-whisper",
            Self::Crisperwhisper => "crisperwhisper",
            Self::Chatterbox => "chatterbox",
            Self::ChatterboxTurbo => "chatterbox-turbo",
            Self::ChatterboxNano => "chatterbox-nano",
            Self::Qwen3Tts => "qwen3-tts",
            Self::VoxCpm2 => "voxcpm",
            Self::VibeVoice => "vibevoice",
            Self::VibeVoiceRealtime => "vibevoice-realtime",
            Self::Irodori => "irodori",
            Self::VitsJa => "vits-ja",
            Self::StyleTts2 => "styletts2",
            Self::DebertaV2 => "deberta-v2",
            Self::DebertaV3 => "deberta-v3",
            Self::BertBase => "bert-base",
            Self::SbV2 => "sbv2",
            Self::XCodec2 => "xcodec2",
            // SoTA plan Phase 5 fleet (2026-07-28): 12 BF16 pass-through
            // skeleton wire-ups. Every canonical CLI slug is hyphenated
            // (matches `from_arg` above); the arch tag (underscore) rides
            // in the GGUF's `vokra.model.arch` chunk.
            Self::KimiAudio => "kimi-audio",
            Self::StepAudio2Mini => "step-audio2-mini",
            Self::BaichuanAudio => "baichuan-audio",
            Self::Speechtokenizer => "speechtokenizer",
            Self::Funcodec => "funcodec",
            Self::XyTokenizer => "xy-tokenizer",
            Self::Bicodec => "bicodec",
            Self::Neucodec => "neucodec",
            Self::NkfAec => "nkf-aec",
            // Coverage-audit 2026-08-03 Wave A: canonical CLI slug is
            // the versioned form (matches the `huggingface.co/vokra/
            // rnnoise-v0.2` publish repo). The arch tag stamped in
            // `vokra.model.arch` (`"rnnoise"` — see
            // `models::rnnoise::ARCH`) is the version-neutral short
            // form so a future v0.3 GGUF with the same topology stays
            // classifier-compatible.
            Self::Rnnoise => "rnnoise-v0.2",
            Self::Nsnet2 => "nsnet2",
            Self::Dnsmos => "dnsmos-p808-p835",
            // coverage-audit wave-a (2026-08-03): FRCRN speech enhancement.
            Self::Frcrn => "frcrn",
            // coverage-audit 2026-08-03 Wave B fast-track (13 variants).
            Self::Hibiki => "hibiki-2b",
            Self::SberGigaamV3 => "sber-gigaam-v3",
            Self::SberGigaamMultilingual => "sber-gigaam-multilingual",
            Self::ReazonspeechNemoV2 => "reazonspeech-nemo-v2",
            Self::MagpiettsV2602 => "magpietts-v2602",
            Self::ParakeetUnified => "parakeet-unified-en-0.6b",
            Self::Canary1bFlash => "canary-1b-flash",
            Self::OwsmV4Medium1b => "owsm-v4-medium-1b",
            Self::ParakeetTdt11b => "parakeet-tdt-1.1b",
            Self::FireredAsrAedL => "firered-asr-aed-l",
            Self::SortformerDiar4spkV1 => "sortformer-diar-4spk-v1",
            Self::SenseVoiceSmall => "sensevoicesmall",
            Self::WhisperMedusaV1 => "whisper-medusa-v1",
            Self::FacebookDenoiser => "facebook-denoiser",
            Self::NisqaV2Weight => "nisqa-v2-weight",
            Self::ChatTts => "chattts",
            Self::StableAudioOpenSmall => "stable-audio-open-small",
            Self::Jasco400mChordsDrums => "jasco-400m-chords-drums",
            // coverage-audit-2026-08-03 Wave A permissive continuation (2026-08-04).
            Self::Utmosv2 => "utmosv2",
            Self::TorchaudioSquim => "torchaudio-squim",
            Self::HtdemucsMulti => "htdemucs-4s-6s",
            Self::OpenwakewordOp => "openwakeword-op",
            Self::Mossformer2Ss16k => "mossformer2-ss-16k",
            Self::TenVad => "ten-vad",
            Self::AudiosealRealWeight => "audioseal-real-weight",
            // Music-understanding wave (2026-08-13). YAMNet canonical CLI slug
            // matches the shared arch tag stamped in `vokra.model.arch`.
            Self::Yamnet => "yamnet",
            // Music-understanding wave (2026-08-13). MERT canonical CLI slug
            // matches the shared arch tag stamped in `vokra.model.arch`.
            Self::Mert => "mert",
            // Music-understanding wave (2026-08-13). MuQ canonical CLI slug.
            Self::Muq => "muq",
            // hf-audio-gap-comprehensive-2026-07-30 §3.8 (2026-08-04):
            // canonical CLI slug matches the publish repo tail token
            // (`huggingface.co/vokra/miocodec-25hz-44khz-v2` — HF repo
            // naming = dashes only, lowercase, dots stripped from `44.1`
            // → `44khz`). The arch tag stamped in `vokra.model.arch`
            // (`"miocodec"` — see `models::miocodec::ARCH`) is the
            // version-neutral short form so a hypothetical future v3
            // GGUF with the same topology stays classifier-compatible.
            Self::MioCodec => "miocodec-25hz-44khz-v2",
            // SoTA plan candidate wave (2026-08-04): canonical CLI slug
            // matches the publish repo tail token
            // (`huggingface.co/vokra/neutts-air` — HF repo naming =
            // dashes only, lowercase) AND the arch tag stamped into
            // `vokra.model.arch` (`"neutts-air"` — see
            // `models::neutts_air::ARCH`), so the CLI id, the publish
            // repo id, and the on-disk arch tag are the same string.
            Self::NeuTtsAir => "neutts-air",
            Self::NemotronSpeechStreamingV2603 => "nemotron-speech-streaming-v2603",
            Self::EcapaTdnn => "ecapa-tdnn",
            Self::Wespeaker => "wespeaker",
            Self::Speaker3d => "speaker-3d",
            Self::TitaNet => "titanet-large",
            Self::Emotion2vec => "emotion2vec",
            Self::Rmvpe => "rmvpe",
            Self::Crepe => "crepe",
            Self::PyannoteSegmentation => "pyannote-segmentation",
            Self::PyannoteSpeakerDiarization31 => "pyannote-speaker-diarization-3.1",
            Self::Ast => "ast",
            Self::AudioboxAesthetics => "audiobox-aesthetics",
            Self::Bark => "bark",
            Self::BarkSmall => "bark-small",
            Self::BigVGan => "bigvgan",
            Self::Clap => "clap",
            Self::CohereTranscribe => "cohere-transcribe-03-2026",
            Self::DeepfakeDetection => "deepfake-detection",
            Self::FireredVad => "firered-vad",
            Self::Focalcodec => "focalcodec",
            Self::FsmnVad => "fsmn-vad",
            Self::HifiganVocoder => "hifigan-vocoder",
            Self::Speecht5Hifigan => "speecht5-hifigan",
            Self::IndicParlerTts => "indic-parler-tts",
            Self::ParlerTtsMiniV1English => "parler-tts-mini-v1",
            Self::KyutaiTts => "kyutai-tts",
            Self::LangIdCommonLanguage => "lang-id-commonlanguage",
            Self::LangIdVoxlingua107 => "lang-id-voxlingua107",
            Self::MeloTtsChinese => "melotts-chinese",
            Self::MeloTtsEnglish => "melotts-english",
            Self::MeloTtsKorean => "melotts-korean",
            Self::MeloTtsSpanish => "melotts-spanish",
            Self::MeloTtsJapanese => "melotts-japanese",
            Self::MetricganPlus => "metricgan-plus",
            Self::MossTts => "moss-tts",
            Self::MossTtsLocal => "moss-tts-local",
            Self::MossTtsNano => "moss-tts-nano",
            Self::MossTtsV15 => "moss-tts-v1.5",
            Self::MossAudio4bInstruct => "moss-audio-4b-instruct",
            Self::MossAudio8bInstruct => "moss-audio-8b-instruct",
            Self::MpSenet => "mp-senet",
            Self::MpSenetDns => "mp-senet-dns",
            Self::NemotronAsrStreaming => "nemotron-3.5-asr-streaming-0.6b",
            Self::ParlerTtsMiniMultilingual => "parler-tts",
            Self::Qwen3Asr => "qwen3-asr",
            Self::Qwen3TtsBase17B => "qwen3-tts-1.7b-base",
            Self::Qwen3TtsCustomVoice17B => "qwen3-tts-1.7b-customvoice",
            Self::Qwen3TtsVoiceDesign17B => "qwen3-tts-1.7b-voicedesign",
            Self::SepFormer => "sepformer-wsj02mix",
            Self::SepformerWham16kEnh => "sepformer-wham16k-enhancement",
            Self::SepformerWhamr16k => "sepformer-whamr16k",
            Self::SepformerLibri2Mix => "sepformer-libri2mix",
            Self::SepformerLibri3Mix => "sepformer-libri3mix",
            Self::SepformerWhamr8k => "sepformer-whamr-8khz",
            Self::SepformerDns4Enh => "sepformer-dns4-16k-enhancement",
            // SoTA plan candidate wave (2026-08-04): canonical CLI slug
            // matches the publish repo tail token
            // (`huggingface.co/vokra/sgmse-voicebank`) — HF repo naming
            // = dashes only, lowercase. The bare `sgmse` alias resolves
            // via `from_arg` for the short spelling, but the display
            // form keeps the corpus qualifier so the CLI id, the
            // publish repo id, and the on-disk model name (`vokra.
            // model.name = "sgmse-voicebank"`) are the same string.
            Self::Sgmse => "sgmse-voicebank",
            Self::SmartTurn => "smart-turn",
            Self::Snac => "snac",
            Self::Wavtokenizer => "wavtokenizer",
            Self::GraniteSpeech => "granite-speech-4.1-2b",
            Self::MossAudioTokenizer => "moss-audio-tokenizer",
            Self::Facodec => "naturalspeech3-facodec",
            Self::SpeechT5Tts => "speecht5-tts",
            Self::TigerSeparator => "tiger-dnr",
            Self::TigerSpeech => "tiger-speech",
            Self::VieNeuTts => "vieneu-tts",
            Self::VoxtralMiniRealtime => "voxtral-mini-4b-realtime-2602",
            Self::Wav2Vec2Ctc => "wav2vec2",
            Self::Data2vecAudioBase => "data2vec-audio-base",
            Self::Mms1bAll => "mms-1b-all",
            Self::XVector => "xvector",
            Self::Fcpe => "fcpe",
            Self::Vocos => "vocos-mel-24khz",
            Self::YueUpsampler => "yue-upsampler",
            Self::YueXcodecMini => "yue-xcodec-mini",
            Self::MusicGenMedium => "musicgen-medium",
            Self::MusicGenLarge => "musicgen-large",
            Self::MusicGenMelody => "musicgen-melody",
            Self::AudioGenMedium => "audiogen-medium",
            Self::MusicGenSmall => "musicgen-small",
            Self::Qwen2Audio => "qwen2-audio-7b-instruct",
            Self::Qwen25Omni7b => "qwen2-5-omni-7b",
            Self::VibeVoiceAsr => "vibevoice-asr",
            Self::AceStep => "ace-step-1.5",
            Self::HubertLargeLs960 => "hubert-large-ls960",
            Self::W2vBert2 => "w2v-bert-2-0",
            Self::AudioLdm2 => "audioldm2",
            Self::AudioLdm2Large => "audioldm2-large",
            Self::BsRoformer => "bs-roformer",
            Self::Openwakeword => "openwakeword",
            Self::MoonshineTiny => "moonshine-tiny",
            Self::MoonshineBase => "moonshine-base",
            Self::DemucsHtdemucs => "demucs-htdemucs",
            Self::UltravoxV05Llama321b => "ultravox-v0-5-llama-3-2-1b",
            Self::XttsV2 => "xtts-v2",
            Self::ConvTasnetLibri1mix => "conv-tasnet-libri1mix",
            Self::SeamlessM4tV2Large => "seamless-m4t-v2-large",
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
    /// A `QuantPolicy` rule resolved to a K-quant target for
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

/// [`convert_file`] with per-variant slug dispatch (2026-07-30 Task 3 add).
///
/// Some `ModelKind` variants ([`ModelKind::BigVGan`] / [`ModelKind::Qwen3Asr`]
/// / [`ModelKind::Wav2Vec2Ctc`]) collapse multiple upstream release variants
/// (e.g. wav2vec2-base-960h / -large-xlsr-53 / -large-xlsr-53-japanese) into
/// one enum arm — the plain [`convert_file`] path routes every alias to the
/// arm's default variant, silently losing the per-slug distinction. This
/// entry accepts the raw `--model` slug and picks the matching variant
/// enum before dispatch. For everything else it delegates to
/// [`convert_file`] verbatim.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_file_with_slug(
    model: ModelKind,
    slug: &str,
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    match model {
        ModelKind::BigVGan => {
            let variant = match slug.to_lowercase().as_str() {
                "bigvgan-v2-22khz-80band-256x"
                | "bigvgan_v2_22khz_80band_256x"
                | "nvidia/bigvgan_v2_22khz_80band_256x" => {
                    models::bigvgan::BigVGanVariant::V2_22khz80Band256x
                }
                "bigvgan-v2-44khz-128band-512x"
                | "bigvgan_v2_44khz_128band_512x"
                | "nvidia/bigvgan_v2_44khz_128band_512x" => {
                    models::bigvgan::BigVGanVariant::V2_44khz128Band512x
                }
                "bigvgan-base-24khz-100band"
                | "bigvgan_base_24khz_100band"
                | "nvidia/bigvgan_base_24khz_100band" => {
                    models::bigvgan::BigVGanVariant::BaseV1_24khz100Band
                }
                // Everything else (canonical "bigvgan" / "big-vgan" / v2-24khz)
                // → default v2_24khz100Band256x (the most common variant).
                _ => models::bigvgan::BigVGanVariant::V2_24khz100Band256x,
            };
            let report = models::bigvgan::convert_bigvgan_file(input, output, variant, license)?;
            let notes = vec![format!(
                "bigvgan ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Neucodec => {
            let variant = match slug.to_lowercase().as_str() {
                "distill-neucodec"
                | "distill_neucodec"
                | "distill-neu-codec"
                | "neuphonic/distill-neucodec" => models::neucodec::NeucodecVariant::Distill,
                // Canonical "neucodec" / neuphonic/neucodec → Base default.
                _ => models::neucodec::NeucodecVariant::Base,
            };
            let report =
                models::neucodec::convert_neucodec_variant_file(input, output, license, variant)?;
            let notes = vec![format!(
                "neucodec ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Snac => {
            let variant = match slug.to_lowercase().as_str() {
                "snac-44khz" | "snac_44khz" | "hubertsiuzdak/snac_44khz" => {
                    models::snac::SnacVariant::Hz44
                }
                // Canonical "snac" / -24khz → Hz24 (higher-download default).
                _ => models::snac::SnacVariant::Hz24,
            };
            let report = models::snac::convert_snac_file(input, output, license, variant)?;
            let notes = vec![format!(
                "snac ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::MossAudioTokenizer => {
            // MOSS-Audio-Tokenizer slug-based variant dispatch mirror
            // of the Snac / Focalcodec / BigVGan arms above — the
            // canonical / "moss-audio-tokenizer" / "-full" aliases
            // route to Full (higher-fidelity default the MOSS-TTS
            // consumer pairs with); explicit -nano slugs pick the
            // distilled variant.
            let variant = match slug.to_lowercase().as_str() {
                "moss-audio-tokenizer-nano"
                | "moss_audio_tokenizer_nano"
                | "openmoss-team/moss-audio-tokenizer-nano" => {
                    models::moss_audio_tokenizer::MossAudioTokenizerVariant::Nano
                }
                // Canonical "moss-audio-tokenizer" / -full →
                // Full default.
                _ => models::moss_audio_tokenizer::MossAudioTokenizerVariant::Full,
            };
            let report = models::moss_audio_tokenizer::convert_moss_audio_tokenizer_variant_file(
                input, output, variant, license,
            )?;
            let notes = vec![format!(
                "moss-audio-tokenizer ({}): {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Qwen3Asr => {
            let variant = match slug.to_lowercase().as_str() {
                "qwen3-asr-0.6b" | "qwen3_asr_0_6b" | "qwen/qwen3-asr-0.6b" => {
                    models::qwen3_asr::Variant::B06
                }
                // Canonical "qwen3-asr" / -1.7b → B17 (the flagship default).
                _ => models::qwen3_asr::Variant::B17,
            };
            let report = models::qwen3_asr::convert_qwen3_asr_file_with_variant(
                input, output, variant, license,
            )?;
            let notes = vec![format!(
                "qwen3-asr ({variant:?}): {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Data2vecAudioBase => {
            // Same converter as `Wav2Vec2Ctc` (tensor names identical —
            // data2vec-audio inherits the wav2vec 2.0 base downstream
            // inference topology + Conv1D feature-extractor + English
            // LibriSpeech 960h char CTC head); the dedicated
            // `Variant::Data2vecAudioBase960h` only overrides `name` +
            // `upstream_hf` so the stamped GGUF faithfully reports the
            // data2vec-audio upstream release rather than masquerading
            // as `wav2vec2-base-960h`.
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant(
                input,
                output,
                models::wav2vec2_ctc::Variant::Data2vecAudioBase960h,
                license,
            )?;
            let notes = vec![format!(
                "data2vec-audio-base-960h: {} float weights written verbatim ({} BF16 \
                 passthrough — runtime widens to f32 exactly at load), {} non-float skipped, \
                 {} tensors read (via wav2vec2_ctc converter — tensor names identical)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Mms1bAll => {
            // 2026-08-02 wave: Meta MMS-1B-All (`facebook/mms-1b-all`,
            // **cc-by-nc-4.0**). Massively Multilingual Speech (Pratap
            // et al. 2023, arXiv:2305.13516) — 1B wav2vec 2.0 backbone
            // + 1000+ per-language CTC adapters. Base tensor path
            // reuses the wav2vec2_ctc converter (parent workflow REUSE
            // HINT) via a dedicated `Variant::Mms1bAll` arm with
            // **placeholder axes** (routes to LargeXlsr53Base sibling)
            // and faithful `name = "mms-1b-all"` +
            // `upstream_hf = "facebook/mms-1b-all"` stamps for the
            // placeholder-axis refusal guardrail (M2-13 runtime gate).
            // A follow-up wave must transcribe the true MMS-1B
            // topology + land the per-language adapter loader
            // (~1000 sibling `adapter.*.safetensors`).
            //
            // Weight-distribution licence default = `cc-by-nc-4.0`
            // (T4 tier / Research-only publish path). Callers pass
            // `--license cc-by-nc-4.0` to override the arm's default
            // (`apache-2.0`) at stamp time so the M2-13 gate rejects
            // commercial loads.
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant(
                input,
                output,
                models::wav2vec2_ctc::Variant::Mms1bAll,
                license,
            )?;
            let notes = vec![format!(
                "mms-1b-all: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read (via \
                 wav2vec2_ctc converter — placeholder axes route to LargeXlsr53Base sibling, \
                 per-language adapter loader is a follow-up wave)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Wav2Vec2Ctc => {
            let variant = match slug.to_lowercase().as_str() {
                "wav2vec2-large-xlsr-53"
                | "wav2vec2_large_xlsr_53"
                | "facebook/wav2vec2-large-xlsr-53"
                // Wave 4 slug-only add (2026-08-01): route
                // `wav2vec2-large-960h-lv60-self` through the closest
                // existing large-topology variant. Both slugs share the
                // 24 × d=1024 × 16h × ffn=4096 backbone with
                // `feat_extract_norm=layer` + `do_stable_layer_norm=true`
                // + `vocab_size=32`; only `has_ctc_head` (LV60-self =
                // true vs XLSR-53-base = false) and the upstream_hf
                // provenance stamp differ. Following the slug-only
                // discipline (parent decision, no converter module
                // edits), the two axis mismatches are documented in
                // the from_arg() docstring above and are expected to
                // be fixed by (a) a dedicated `Large960hLv60Self`
                // variant arm or (b) a `restamp` pass when the row is
                // published — not now.
                | "wav2vec2-large-960h-lv60-self"
                | "wav2vec2_large_960h_lv60_self"
                | "facebook/wav2vec2-large-960h-lv60-self" => {
                    models::wav2vec2_ctc::Variant::LargeXlsr53Base
                }
                "wav2vec2-large-xlsr-53-japanese"
                | "wav2vec2_large_xlsr_53_japanese"
                | "jonatasgrosman/wav2vec2-large-xlsr-53-japanese" => {
                    models::wav2vec2_ctc::Variant::LargeXlsr53Japanese
                }
                "wav2vec2-large-xlsr-53-chinese-zh-cn"
                | "wav2vec2_large_xlsr_53_chinese_zh_cn"
                | "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn" => {
                    models::wav2vec2_ctc::Variant::LargeXlsr53ChineseZhCn
                }
                // Wave 4 variant-enum extension (2026-08-01): XLSR-53
                // large backbone with an eSpeak-NG IPA phoneme CTC head
                // (`vocab_size=392`, `Wav2Vec2ForCTC`,
                // arXiv:2109.11680). Distinct arm because vocab_size
                // and has_ctc_head both differ from LargeXlsr53Base
                // (the closest topology sibling) — routing slug-only
                // would stamp a wrong axis. `LicenseClass::Permissive`
                // already covered by the `wav2vec2` prefix walk in
                // `crates/vokra-core/src/compliance/license_class.rs`.
                "wav2vec2-xlsr-53-espeak-cv-ft"
                | "wav2vec2_xlsr_53_espeak_cv_ft"
                | "facebook/wav2vec2-xlsr-53-espeak-cv-ft" => {
                    models::wav2vec2_ctc::Variant::LargeXlsr53EspeakCvFt
                }
                // Canonical "wav2vec2" / -base-960h → Base960h default.
                _ => models::wav2vec2_ctc::Variant::Base960h,
            };
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant(
                input, output, variant, license,
            )?;
            let notes = vec![format!(
                "wav2vec2-ctc ({variant:?}): {} float weights written verbatim ({} BF16 \
                 passthrough — runtime widens to f32 exactly at load), {} non-float skipped, \
                 {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Focalcodec => {
            // Slug-based variant dispatch mirror of the BigVGan arm
            // above: the canonical / "focalcodec" / "focal-codec" alias
            // routes to Hz50 (backward compat with the 2026-07-30
            // publish); explicit -25hz / -12-5hz / -12_5hz slugs pick
            // the corresponding 2026-07-31 additions.
            let variant = match slug.to_lowercase().as_str() {
                "focalcodec-25hz" | "focalcodec_25hz" | "lucadellalib/focalcodec_25hz" => {
                    models::focalcodec::FocalcodecVariant::Hz25
                }
                "focalcodec-12-5hz"
                | "focalcodec-12_5hz"
                | "focalcodec_12_5hz"
                | "lucadellalib/focalcodec_12_5hz" => models::focalcodec::FocalcodecVariant::Hz12_5,
                // Everything else (canonical "focalcodec" /
                // "focal-codec" / -50hz aliases /
                // `lucadellalib/focalcodec_50hz`) → default Hz50
                // (backward compat with the 2026-07-30 publish).
                _ => models::focalcodec::FocalcodecVariant::Hz50,
            };
            let report =
                models::focalcodec::convert_focalcodec_file(input, output, license, variant)?;
            let notes = vec![format!(
                "focalcodec ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Facodec => {
            // 2026-08-01 Wave 3: Amphion NaturalSpeech 3 FACodec —
            // factorized VQ (FVQ) codec (apache-2.0). Slug-based
            // variant dispatch mirror of the Snac / MossAudioTokenizer
            // arms above — the canonical / "facodec" /
            // "naturalspeech3-facodec" / -v2 aliases route to V2
            // (default = highest-quality codec-only pair, per prep-
            // script default); explicit -v1 / -redecoder-v{1,2} slugs
            // pick the corresponding variants.
            //
            // **Voice-conversion policy note**: the redecoder-v{1,2}
            // variants enable zero-shot voice conversion (see
            // `models::naturalspeech3_facodec` module docstring for
            // the CLAUDE.md 設計判断 8 routing question — main zoo vs
            // `vokra-voiceclone-experimental`). The dispatch here
            // emits the artifact regardless; the publish target is
            // an owner routing decision.
            let variant = match slug.to_lowercase().as_str() {
                "facodec-v1" | "naturalspeech3-facodec-v1" | "ns3-facodec-v1" => {
                    models::naturalspeech3_facodec::FacodecVariant::V1
                }
                "facodec-redecoder-v1"
                | "naturalspeech3-facodec-redecoder-v1"
                | "ns3-facodec-redecoder-v1" => {
                    models::naturalspeech3_facodec::FacodecVariant::RedecoderV1
                }
                "facodec-redecoder-v2"
                | "naturalspeech3-facodec-redecoder-v2"
                | "ns3-facodec-redecoder-v2" => {
                    models::naturalspeech3_facodec::FacodecVariant::RedecoderV2
                }
                // Everything else (canonical "facodec" /
                // "naturalspeech3-facodec" / -v2 / raw HF slug) →
                // default V2 (highest-quality codec-only pair,
                // matches prep-script default).
                _ => models::naturalspeech3_facodec::FacodecVariant::V2,
            };
            let report =
                models::naturalspeech3_facodec::convert_naturalspeech3_facodec_variant_file(
                    input, output, variant, license,
                )?;
            let notes = vec![format!(
                "facodec ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        ModelKind::Vocos => {
            // 2026-08-01 wave: Charactr AI Vocos vocoder. Slug-based
            // variant dispatch mirror of the Focalcodec / BigVGan arms
            // above — the canonical / "vocos" / "vocos-mel-*" aliases
            // route to Mel24khz (default, HF-download top); explicit
            // -encodec-24khz slugs pick the encodec variant.
            let variant = match slug.to_lowercase().as_str() {
                "vocos-encodec-24khz"
                | "vocos_encodec_24khz"
                | "vocos-encodec"
                | "charactr/vocos-encodec-24khz" => models::vocos::VocosVariant::Encodec24khz,
                // Everything else (canonical "vocos" / "vocos-mel-24khz"
                // / "charactr/vocos-mel-24khz") → default Mel24khz.
                _ => models::vocos::VocosVariant::Mel24khz,
            };
            let report = models::vocos::convert_vocos_file(input, output, variant, license)?;
            let notes = vec![format!(
                "vocos ({}): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                variant.tag(),
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
            )];
            Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            })
        }
        _ => convert_file_licensed(model, input, output, license),
    }
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
    // pyannote/speaker-diarization-3.1 is a **weightless pipeline** — the
    // upstream repo ships only a ~2 KB config.yaml (no .bin / .safetensors),
    // the pipeline delegates every forward-pass computation to two sibling
    // weight repos (pyannote/segmentation-3.0 + wespeaker-voxceleb-
    // resnet34-LM). The converter reads the config.yaml as a raw sanity
    // buffer (no YAML parser — NFR-DS-02 zero-dep) and emits orchestration
    // metadata from primary-source Rust constants. Routes BEFORE the
    // whole-file read below so the sibling per-model arms can stay
    // safetensors-shaped.
    if matches!(model, ModelKind::PyannoteSpeakerDiarization31) {
        let report =
            models::pyannote_speaker_diarization_3_1::convert_pyannote_speaker_diarization_3_1_file(
                input, output, license,
            )?;
        let notes = vec![format!(
            "pyannote-speaker-diarization-3.1: weightless pipeline GGUF written \
             (input_recognized={}, sub-model refs pinned to pyannote/segmentation-3.0 \
             + pyannote/wespeaker-voxceleb-resnet34-LM); {} tensors emitted by design",
            report.input_recognized, report.written,
        )];
        return Ok(ConvertSummary {
            model: ModelKind::PyannoteSpeakerDiarization31,
            tensor_count: report.written,
            metadata_count: 0,
            output_bytes: std::fs::metadata(output)?.len(),
            notes,
        });
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
        ModelKind::PyannoteSpeakerDiarization31 => {
            // Handled by the weightless-pipeline early-return above
            // (config.yaml sanity buffer, no fs::read of a weight file);
            // reaching this arm would mean the outer bypass was removed.
            unreachable!(
                "ModelKind::PyannoteSpeakerDiarization31 routes through \
                 convert_pyannote_speaker_diarization_3_1_file"
            )
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
        ModelKind::CanaryQwen => {
            // SoTA plan reuse bundle (2026-07-30): pass every F32/F16/BF16
            // tensor through verbatim and stamp the `vokra.canary_qwen.*`
            // chunk group (FastConformer encoder axes reused from
            // Canary-1B-v2 + Qwen LLM decoder axes with `0`-placeholder
            // dims). Provenance = CC-BY 4.0 (AttributionRequired via
            // `canary-` prefix walk) + FR-MD-09 attribution text.
            // Distinct arch tag from base Canary so the runtime dispatches
            // to the Qwen-decoder path.
            let (builder, report) = models::canary_qwen::convert(bytes)?;
            let mut notes = vec![format!(
                "canary-qwen: {} float weights written verbatim, {} non-float skipped, \
                 {} BF16 pass-through",
                report.written, report.skipped_non_float, report.bf16_passthrough,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("canary-qwen warning: {n}")),
            );
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
        ModelKind::Crisperwhisper => {
            // residual wave 4 (2026-08-02): reuse the vanilla Whisper
            // converter via the `WhisperVariant::CrisperWhisper` arm
            // (same-shape / different-license case). Every architectural
            // axis (tensor topology, `vokra.whisper.*` hparam chunk,
            // front-end spec, alignment-head table, detokenizer) is
            // byte-identical to whisper-large-v3; only the arch stamp
            // (`crisper-whisper`, distinct from `whisper`), license
            // class (NonCommercial, cc-by-nc-4.0), and provenance
            // source citation come from the variant. The M2-13 runtime
            // gate refuses to load the resulting GGUF in commercial
            // mode; publish requires `publish-one.sh
            // --allow-noncommercial` per the X-Codec-2 T4 precedent
            // (2026-07-28).
            (
                models::whisper::convert_variant(
                    bytes,
                    models::whisper::WhisperVariant::CrisperWhisper,
                )?,
                Vec::new(),
            )
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
            // SoTA plan Phase 4 (2026-07-24): pass every F32/F16/BF16
            // tensor through verbatim and stamp the `vokra.voxcpm2.*` +
            // `vokra.vae_continuous.*` chunk groups (MiniCPM-4 LM
            // backbone + residual acoustic LM + local encoder + local
            // DiT + UnifiedCFM sampler + AudioVAE V2 continuous encoder /
            // decoder + inline scalar-quantization bottleneck) from the
            // transcribed constants in `models::voxcpm2`. NEW CLASS of
            // TTS vs every sibling: the terminal decoding hop is a
            // continuous VAE decoder consuming flow-matching sampler
            // output (not vocoder-LM HiFTChain, not codec-LM RVQ / FSQ)
            // — silently sharing an arch tag would mis-route the runtime
            // dispatch. Provenance = apache-2.0 end-to-end (Permissive —
            // no runtime-side attribution obligation; code + weight all
            // under a single apache-2.0 grant).
            //
            // 2026-07-30 variant support (Option C hybrid): the same
            // ModelKind serves both `openbmb/VoxCPM-0.5B` and
            // `openbmb/VoxCPM2` (the 2B scale-up). Variant selection
            // rides on the safetensors payload — see
            // `models::voxcpm2::detect_variant` — and the detected
            // variant is surfaced in this WP's notes so an operator
            // reading the CLI trailer sees which release was converted.
            let (builder, report) = models::voxcpm2::convert(bytes)?;
            let variant_label = match report.variant {
                Some(models::voxcpm2::VoxCpm2Variant::HalfB) => "voxcpm2-0.5b",
                Some(models::voxcpm2::VoxCpm2Variant::TwoB) => "voxcpm2-2b",
                None => "unknown-variant",
            };
            let mut notes = vec![format!(
                "voxcpm2: variant={variant_label}, {} float weights written verbatim, \
                 {} non-float skipped",
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
        ModelKind::VibeVoiceRealtime => {
            // Microsoft VibeVoice-Realtime-0.5B (2026-08-01): pass
            // every F32/F16/BF16 tensor through verbatim and stamp
            // the `vokra.vibevoice.*` chunk group (Qwen2 0.5B
            // decoder LM + acoustic σ-VAE tokenizer + 4-layer AdaLN
            // diffusion head with DDPM v-prediction +
            // streaming-only `tts_backbone_num_hidden_layers=20`)
            // from the transcribed constants in
            // `models::vibevoice`. Distinct arch tag
            // (`vibevoice_streaming`) from the 1.5B baseline so
            // runtime dispatch never conflates the two variants.
            // Semantic tokenizer keys are deliberately **not** written
            // (streaming variant is acoustic-tokenizer only).
            // Provenance = **MIT** end-to-end (Permissive --
            // huggingface.co/microsoft/VibeVoice-Realtime-0.5B model
            // card `license: mit`).
            let (builder, report) = models::vibevoice::convert_realtime_05b(bytes)?;
            let mut notes = vec![format!(
                "vibevoice-realtime: {} float weights written verbatim, \
                 {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("vibevoice-realtime warning: {n}")),
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
        ModelKind::StyleTts2 => {
            // Config-only scaffold (2026-07-30): pass every F32 / F16 /
            // BF16 tensor through verbatim and stamp the
            // `vokra.styletts2.*` chunk group (yl4579/StyleTTS2 config
            // axes) from the transcribed constants in
            // `models::styletts2`. **⚠️  Provenance defaults to
            // `Unknown`**: the yl4579 pretrained release rides a
            // voice-consent / disclosure usage agreement — NOT a
            // standard SPDX permissive license — so the M2-13 runtime
            // gate refuses to load in commercial mode. The runtime
            // `StyleTts2Tts::from_gguf` itself is deliberately unwired
            // (returns `NotImplemented` naming the licensing blocker);
            // a user who trained on a permissive corpus overrides at
            // the outer `--license <spdx>` boundary below.
            let (builder, report) = models::styletts2::convert(bytes)?;
            let mut notes = vec![format!(
                "styletts2: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("styletts2 warning: {n}")),
            );
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
            // Blocker 5 (2026-08-06): tokenizer side-car is a
            // vokra-cli-front-end concern (mirror of the Voxtral
            // `--tokenizer` boundary). The plain `convert_file_licensed`
            // dispatch path here has no side-car parameter to forward,
            // so pass `None` and let the CLI's own
            // `convert_deberta_v2_file(..., Some(bytes))` route stamp the
            // tokenizer when the flag is supplied.
            let report = convert_deberta_v2_file(input, output, license, None)?;
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
        ModelKind::BertBase => {
            // WP-14 (2026-08-10): plain BERT
            // (`hfl/chinese-roberta-wwm-ext-large`, apache-2.0) →
            // `bert_base.*` GGUF names + `vokra.bert_base.*` hparam
            // chunk group + optional `vokra.bert.wordpiece.*` tokenizer
            // side-car. The generic dispatch path has no `--tokenizer`
            // parameter to forward, so tokenizer emission is left off
            // here — the CLI's own `--tokenizer` flag routes directly
            // to `convert_bert_base_file(_, _, _, Some(bytes), _)`
            // (mirror of the deberta-v2 / deberta-v3 vocab-txt / spm.json
            // gates). `do_lower_case = false` is the ZH-first-consumer
            // default; English WordPiece checkpoints pass through
            // the direct-call path with `do_lower_case = true`.
            let report = convert_bert_base_file(input, output, license, None, false)?;
            let notes = vec![format!(
                "bert-base: {} float weights renamed + written verbatim, {} unmapped skipped, \
                 {} non-float skipped",
                report.written, report.skipped_unmapped, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::BertBase,
                tensor_count: report.written,
                metadata_count: 0, // Populated by convert_bert_base_file's builder
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::DebertaV3 => {
            // SBV2 v2 plan Task 11 (2026-07-26; upstream/license label
            // corrected 2026-07-27, Task 8): pass every F32/F16/BF16
            // tensor through verbatim under upstream HF names and stamp the
            // `vokra.bert.deberta_v3.*` chunk group (DeBERTa v3 transformer
            // encoder + hparams) from the transcribed constants in
            // `models::deberta_v3`. Provenance = MIT (Permissive — no
            // runtime-side attribution obligation, per HF model card
            // `microsoft/deberta-v3-large`; the real EN upstream, distinct
            // from v2's `ku-nlp` JA upstream). Tensor-to-schema mapping
            // (Task 30) is deferred; every tensor is emitted verbatim so
            // the mapping can be validated once a real checkpoint arrives.
            // Blocker 5 (2026-08-06): see the v2 arm above — tokenizer
            // side-car flows through the CLI `--tokenizer` route only.
            let report = convert_deberta_v3_file(input, output, license, None)?;
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
        ModelKind::XCodec2 => {
            // SoTA plan Phase 5 codec (2026-07-28): pass every F32 / F16 /
            // BF16 tensor through verbatim and stamp the
            // `vokra.model.arch = "xcodec2"` + `vokra.model.category =
            // "codec"` + `vokra.provenance.upstream_hf =
            // "HKUSTAudio/xcodec2"` chunk group. Provenance defaults to
            // **cc-by-nc-4.0 / NonCommercial** — the runtime M2-13 gate
            // refuses to load in commercial mode. A caller who trained
            // on a permissive corpus (or holds the weight under a
            // distinct SPDX id) overrides at the outer `--license
            // <spdx>` boundary below (same pattern as vits-ja /
            // Whisper / kokoro).
            let (builder, report) = models::xcodec2::convert(bytes)?;
            let notes = vec![format!(
                "xcodec2: {} float weights written verbatim ({} BF16 passthrough — runtime widens \
                 to f32 exactly at load), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            (builder, notes)
        }
        // ---- SoTA plan Phase 5 fleet (2026-07-28): 12 file-based BF16 -----
        // pass-through skeleton wire-ups. Each module exposes only a
        // `convert_<name>_file(input, output, license)` entry (no
        // bytes-based `convert()` helper), so we early-return with a
        // `ConvertSummary` following the DebertaV2 / DebertaV3 / SbV2
        // dispatch pattern. The outer `bytes = std::fs::read(input)?`
        // is unused on this path (the file-based converter does its own
        // I/O) — matches the DebertaV2 arm's cost profile.
        ModelKind::KimiAudio => {
            let report = models::kimi_audio::convert_kimi_audio_file(input, output, license)?;
            let notes = vec![format!(
                "kimi-audio: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::KimiAudio,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::StepAudio2Mini => {
            let report =
                models::step_audio2_mini::convert_step_audio2_mini_file(input, output, license)?;
            let notes = vec![format!(
                "step-audio2-mini: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::StepAudio2Mini,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::BaichuanAudio => {
            let report =
                models::baichuan_audio::convert_baichuan_audio_file(input, output, license)?;
            let notes = vec![format!(
                "baichuan-audio: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::BaichuanAudio,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Speechtokenizer => {
            let report =
                models::speechtokenizer::convert_speechtokenizer_file(input, output, license)?;
            let notes = vec![format!(
                "speechtokenizer: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Speechtokenizer,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Funcodec => {
            let report = models::funcodec::convert_funcodec_file(input, output, license)?;
            let notes = vec![format!(
                "funcodec: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Funcodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::XyTokenizer => {
            let report = models::xy_tokenizer::convert_xy_tokenizer_file(input, output, license)?;
            let notes = vec![format!(
                "xy-tokenizer: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::XyTokenizer,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Bicodec => {
            let report = models::bicodec::convert_bicodec_file(input, output, license)?;
            let notes = vec![format!(
                "bicodec: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Bicodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Neucodec => {
            let report = models::neucodec::convert_neucodec_file(input, output, license)?;
            let notes = vec![format!(
                "neucodec: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Neucodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::NkfAec => {
            // coverage-audit-2026-08-03 Wave A: pass every F32 / F16 /
            // BF16 tensor through verbatim (the flattened safetensors
            // from `tools/parity/nkf_aec_prepare_checkpoint.py` — the
            // upstream .pt is torch pickle) and stamp `vokra.model.*`
            // (arch = "nkf_aec", name = "nkf-aec", category = "aec") +
            // `vokra.provenance.upstream_url = github.com/fjiang9/NKF-AEC`
            // (GitHub-only release, no HF mirror) + the standard
            // `vokra.provenance.{weight_license,license,model_id,source}`
            // chunk via `stamp_provenance`. Provenance = **MIT**
            // (Permissive — `Copyright (c) 2022 Fei Jiang`).
            let report = models::nkf_aec::convert_nkf_aec_file(input, output, license)?;
            let notes = vec![format!(
                "nkf-aec: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::NkfAec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // Coverage-audit 2026-08-03 Wave A: Xiph RNNoise v0.2. BF16
        // pass-through skeleton (mirror of neucodec / emotion2vec) —
        // real-weight parity deferred to owner. The upstream weight
        // blob is Xiph's C-array `weights_blob_9.bin`; the operator
        // flattens it to safetensors with
        // `tools/parity/rnnoise_prepare_checkpoint.py` before pointing
        // this converter at the safetensors output. Uses the
        // file-based early-return pattern (mirror of the Phase 5
        // fleet) since `convert_rnnoise_file` does its own I/O — the
        // outer `bytes = std::fs::read(input)?` above is unused on this
        // path.
        ModelKind::Rnnoise => {
            let report = models::rnnoise::convert_rnnoise_file(input, output, license)?;
            let notes = vec![format!(
                "rnnoise: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Rnnoise,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Nsnet2 => {
            // Coverage-audit 2026-08-03 Wave A: MIT Permissive baseline —
            // pass every F32 / F16 / BF16 tensor through verbatim and stamp
            // the `vokra.model.*` + `vokra.provenance.*` chunks that identify
            // the artifact as the NSNet2 20 ms enhancement baseline. The
            // upstream ONNX → safetensors bridge lives in the offline
            // sidecar `tools/parity/nsnet2_prepare_checkpoint.py` so this
            // converter stays inside the zero-dep safetensors-only contract
            // (FR-LD-05, NFR-DS-02).
            let report = models::nsnet2::convert_nsnet2_file(input, output, license)?;
            let notes = vec![format!(
                "nsnet2: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Nsnet2,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Dnsmos => {
            // Coverage-audit Wave A (2026-08-03): Microsoft DNSMOS P.808 +
            // P.835 bundle. The file-based routine reads the prepared
            // safetensors from `tools/parity/dnsmos_prepare_checkpoint.py`
            // (which flattens the two upstream ONNX checkpoints with
            // `p808.` / `p835.` name prefixes), passes every F32 tensor
            // through verbatim under its prefixed name, and stamps the
            // `vokra.dnsmos.*` bundle inventory.
            let report = models::dnsmos::convert_dnsmos_file(input, output, license)?;
            let notes = vec![format!(
                "dnsmos: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped, {} bundle variant(s) detected",
                report.written,
                report.bf16_passthrough,
                report.skipped_non_float,
                report.bundle_variants,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Dnsmos,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Frcrn => {
            // coverage-audit wave-a (2026-08-03): FRCRN speech enhancement
            // (Apache-2.0 Permissive). File-based converter with the
            // standing per-call license override — same posture as the
            // wespeaker / emotion2vec / speaker_3d / ecapa_tdnn arms.
            let report = models::frcrn::convert_frcrn_file(input, output, license)?;
            let notes = vec![format!(
                "frcrn: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Frcrn,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // ---- coverage-audit 2026-08-03 Wave B fast-track (13 variants) ----
        // All are file-based BF16 pass-through skeletons; each module owns
        // its own convert_{snake}_file(input, output, license) and returns a
        // {Pascal}Report with (written, bf16_passthrough, skipped_non_float).
        ModelKind::Hibiki => {
            let report = models::hibiki::convert_hibiki_file(input, output, license)?;
            let notes = vec![format!(
                "hibiki: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Hibiki,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::SberGigaamV3 => {
            let report =
                models::sber_gigaam_v3::convert_sber_gigaam_v3_file(input, output, license)?;
            let notes = vec![format!(
                "sber-gigaam-v3: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SberGigaamV3,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::SberGigaamMultilingual => {
            let report = models::sber_gigaam_multilingual::convert_sber_gigaam_multilingual_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "sber-gigaam-multilingual: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SberGigaamMultilingual,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::ReazonspeechNemoV2 => {
            let report = models::reazonspeech_nemo_v2::convert_reazonspeech_nemo_v2_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "reazonspeech-nemo-v2: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ReazonspeechNemoV2,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::MagpiettsV2602 => {
            let report =
                models::magpietts_v2602::convert_magpietts_v2602_file(input, output, license)?;
            let notes = vec![format!(
                "magpietts-v2602: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MagpiettsV2602,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::ParakeetUnified => {
            let report =
                models::parakeet_unified::convert_parakeet_unified_file(input, output, license)?;
            let notes = vec![format!(
                "parakeet-unified-en-0.6b: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ParakeetUnified,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Canary1bFlash => {
            let report =
                models::canary_1b_flash::convert_canary_1b_flash_file(input, output, license)?;
            let notes = vec![format!(
                "canary-1b-flash: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Canary1bFlash,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::OwsmV4Medium1b => {
            let report =
                models::owsm_v4_medium_1b::convert_owsm_v4_medium_1b_file(input, output, license)?;
            let notes = vec![format!(
                "owsm-v4-medium-1b: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::OwsmV4Medium1b,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::ParakeetTdt11b => {
            let report =
                models::parakeet_tdt_1_1b::convert_parakeet_tdt_1_1b_file(input, output, license)?;
            let notes = vec![format!(
                "parakeet-tdt-1.1b: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ParakeetTdt11b,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::FireredAsrAedL => {
            let report =
                models::firered_asr_aed_l::convert_firered_asr_aed_l_file(input, output, license)?;
            let notes = vec![format!(
                "firered-asr-aed-l: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::FireredAsrAedL,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::SortformerDiar4spkV1 => {
            let report = models::sortformer_diar_4spk_v1::convert_sortformer_diar_4spk_v1_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "sortformer-diar-4spk-v1: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SortformerDiar4spkV1,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::SenseVoiceSmall => {
            let report =
                models::sensevoicesmall::convert_sensevoicesmall_file(input, output, license)?;
            let notes = vec![format!(
                "sensevoicesmall: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SenseVoiceSmall,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::WhisperMedusaV1 => {
            let report =
                models::whisper_medusa_v1::convert_whisper_medusa_v1_file(input, output, license)?;
            let notes = vec![format!(
                "whisper-medusa-v1: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::WhisperMedusaV1,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === coverage-audit-2026-08-03 Wave D T4 (non-commercial batch) ===
        ModelKind::FacebookDenoiser => {
            let report =
                models::facebook_denoiser::convert_facebook_denoiser_file(input, output, license)?;
            let notes = vec![format!(
                "facebook-denoiser: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (cc-by-nc-4.0 default, NonCommercial fail-closed — publish \
                 requires --allow-noncommercial per T4 precedent; runtime binder deferred to \
                 owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::FacebookDenoiser,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::NisqaV2Weight => {
            let report =
                models::nisqa_v2_weight::convert_nisqa_v2_weight_file(input, output, license)?;
            let notes = vec![format!(
                "nisqa-v2-weight: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (cc-by-nc-sa-4.0 default, NonCommercialShareAlike \
                 fail-closed — publish requires --allow-noncommercial + SA cascade obligation)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::NisqaV2Weight,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::ChatTts => {
            let report = models::chattts::convert_chattts_file(input, output, license)?;
            let notes = vec![format!(
                "chattts: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (cc-by-nc-4.0 default, NonCommercial fail-closed — publish requires \
                 --allow-noncommercial)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ChatTts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::StableAudioOpenSmall => {
            let report = models::stable_audio_open_small::convert_stable_audio_open_small_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "stable-audio-open-small: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped (stability-ai-community-license default, \
                 hard-mapped to NonCommercial per CPML precedent — publish requires \
                 --allow-noncommercial)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::StableAudioOpenSmall,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Jasco400mChordsDrums => {
            let report = models::jasco_400m_chords_drums::convert_jasco_400m_chords_drums_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "jasco-400m-chords-drums: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped (cc-by-nc-4.0 default, NonCommercial \
                 fail-closed — publish requires --allow-noncommercial per MusicGen family T4 \
                 precedent)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Jasco400mChordsDrums,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === coverage-audit-2026-08-03 Wave A permissive continuation ===
        // (2026-08-04): 7 BF16 pass-through skeletons all in the T1
        // (Permissive) tier — MIT / BSD-2-Clause / Apache-2.0 defaults
        // land as `LicenseClass::Permissive` and sign-off ☑ Commercial
        // by yousan at land time.
        ModelKind::Utmosv2 => {
            let report = models::utmosv2::convert_utmosv2_file(input, output, license)?;
            let notes = vec![format!(
                "utmosv2: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (mit default, Permissive)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Utmosv2,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::TorchaudioSquim => {
            let report =
                models::torchaudio_squim::convert_torchaudio_squim_file(input, output, license)?;
            let notes = vec![format!(
                "torchaudio-squim: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (bsd-2-clause default, Permissive; GitHub-only upstream, \
                 vokra.provenance.upstream_url stamped)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::TorchaudioSquim,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::HtdemucsMulti => {
            let report =
                models::htdemucs_multi::convert_htdemucs_multi_file(input, output, license)?;
            let notes = vec![format!(
                "htdemucs-multi: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (mit default, Permissive; source count 4-vs-6 rides in tensor \
                 shapes verbatim)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::HtdemucsMulti,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::OpenwakewordOp => {
            let report =
                models::openwakeword_op::convert_openwakeword_op_file(input, output, license)?;
            let notes = vec![format!(
                "openwakeword-op: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (apache-2.0 code default, Permissive; official upstream \
                 weights are CC-BY-NC-SA-4.0 — override via --license cc-by-nc-sa-4.0 to flip to \
                 NonCommercialShareAlike when distributing official weights)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::OpenwakewordOp,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Mossformer2Ss16k => {
            let report = models::mossformer2_ss_16k::convert_mossformer2_ss_16k_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "mossformer2-ss-16k: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (apache-2.0 default, Permissive)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Mossformer2Ss16k,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::TenVad => {
            let report = models::ten_vad::convert_ten_vad_file(input, output, license)?;
            let notes = vec![format!(
                "ten-vad: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (apache-2.0 default, Permissive; GitHub-only upstream, \
                 vokra.provenance.upstream_url stamped; NOTICE attribution required for bundled \
                 LPCNet BSD-3-Clause front-end)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::TenVad,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::AudiosealRealWeight => {
            let report = models::audioseal_real_weight::convert_audioseal_real_weight_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "audioseal-real-weight: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default, Permissive; runtime binder gated on M5-05 \
                 T04 ADR ratification)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::AudiosealRealWeight,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Music-understanding wave (2026-08-13) ===
        // YAMNet — Google Research MobileNetV1 audio-event classifier
        // (521-class AudioSet, ~15 MB edge model). apache-2.0 default
        // (Permissive). Sibling of `panns` / `ast` / `clap` under
        // `audio-tagging` category with a distinct arch tag so silent
        // runtime dispatch cannot misroute a depthwise-separable
        // checkpoint through a residual Cnn14 loader (FR-EX-08).
        ModelKind::Yamnet => {
            let report = models::yamnet::convert_yamnet_file(input, output, license)?;
            let notes = vec![format!(
                "yamnet: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (apache-2.0 default, Permissive; upstream HF mirror carries no license \
                 tag as of 2026-08-13 — reference impl `github.com/tensorflow/models/tree/master/\
                 research/audioset/yamnet` is Google Research Apache-2.0)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Yamnet,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // MERT-v1-330M — Music undERstanding self-supervised encoder.
        // HuBERT-derived Conv1D + 24-layer Transformer, ~330M params.
        // cc-by-nc-4.0 default (NonCommercial, T4 tier — publish requires
        // `--allow-noncommercial`). Sibling of `muq` / `dasheng` under
        // `music-embedding` category with a distinct arch tag so silent
        // runtime dispatch cannot misroute an MPM checkpoint through a
        // MAE loader (FR-EX-08).
        ModelKind::Mert => {
            let report = models::mert::convert_mert_file(input, output, license)?;
            let notes = vec![format!(
                "mert: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (cc-by-nc-4.0 default, NonCommercial fail-closed — publish requires \
                 `publish-one.sh --allow-noncommercial` per X-Codec 2 T4 precedent)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Mert,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // MuQ — Self-supervised music representation learner with Mel-RVQ
        // + BEATs teacher (MERT alternative). License unknown default =
        // fail-closed; runtime M2-13 gate refuses to load without a
        // research flag or explicit --license SPDX override.
        ModelKind::Muq => {
            let report = models::muq::convert_muq_file(input, output, license)?;
            let notes = vec![format!(
                "muq: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (unknown default, LicenseClass::Unknown fail-closed — upstream \
                 `OpenMuQ/MuQ-large-msd-iter` HF cardData carries no `license:` tag as of \
                 2026-08-13; owner must complete primary-source LICENSE confirmation before \
                 publish is unblocked)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Muq,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::MioCodec => {
            // hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder
            // complement wave (2026-08-04): pass every F32 / F16 / BF16
            // tensor through verbatim under its upstream safetensors name
            // and stamp `vokra.model.*` (arch = "miocodec",
            // name = "miocodec-25hz-44khz-v2", category = "codec") +
            // `vokra.provenance.upstream_hf =
            // Aratako/MioCodec-25Hz-44.1kHz-v2` + the standard
            // `vokra.provenance.{weight_license,license,model_id,source}`
            // chunk via `stamp_provenance`. Provenance = **MIT**
            // (Permissive — HF cardData API primary source verified
            // 2026-08-04).
            let report = models::miocodec::convert_miocodec_file(input, output, license)?;
            let notes = vec![format!(
                "miocodec: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped (mit default, Permissive; runtime binder + real-weight parity deferred \
                 to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MioCodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::NeuTtsAir => {
            // SoTA plan candidate wave (2026-08-04): pass every
            // F32 / F16 / BF16 tensor through verbatim under its
            // upstream safetensors name and stamp `vokra.model.*`
            // (arch = "neutts-air", name = "neutts-air",
            // category = "tts") + `vokra.provenance.upstream_hf =
            // neuphonic/neutts-air` + the standard
            // `vokra.provenance.{weight_license,license,model_id,source}`
            // chunk via `stamp_provenance`. Provenance = **apache-2.0**
            // (Permissive — HF cardData API primary source verified
            // 2026-08-04). Upstream ships BF16 end-to-end
            // (747.9 M BF16 params); the F32 / F16 arms are defensive
            // for future re-quantized derivatives. The upstream
            // foreign-GGUF sibling `neutss-air-BF16.gguf` is NOT
            // processed here — FR-LD-05 forbids Vokra runtime from
            // loading foreign GGUFs, and this converter's contract is
            // safetensors-in / Vokra-GGUF-out only.
            let report = models::neutts_air::convert_neutts_air_file(input, output, license)?;
            let notes = vec![format!(
                "neutts-air: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (apache-2.0 default, Permissive; runtime binder + \
                 real-weight parity deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::NeuTtsAir,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::NemotronSpeechStreamingV2603 => {
            let report =
                models::nemotron_speech_streaming_v2603::convert_nemotron_speech_streaming_v2603_file(
                    input, output, license,
                )?;
            let notes = vec![format!(
                "nemotron-speech-streaming-v2603: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::NemotronSpeechStreamingV2603,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::EcapaTdnn => {
            let report = models::ecapa_tdnn::convert_ecapa_tdnn_file(input, output, license)?;
            let notes = vec![format!(
                "ecapa-tdnn: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::EcapaTdnn,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Wespeaker => {
            let report = models::wespeaker::convert_wespeaker_file(input, output, license)?;
            let notes = vec![format!(
                "wespeaker: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Wespeaker,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Speaker3d => {
            let report = models::speaker_3d::convert_speaker_3d_file(input, output, license)?;
            let notes = vec![format!(
                "speaker-3d: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Speaker3d,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::TitaNet => {
            let report = models::titanet::convert_titanet_file(input, output, license)?;
            let notes = vec![format!(
                "titanet-large: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::TitaNet,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Emotion2vec => {
            let report = models::emotion2vec::convert_emotion2vec_file(input, output, license)?;
            let notes = vec![format!(
                "emotion2vec: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Emotion2vec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Rmvpe => {
            // F0 pitch-extractor tier (2026-07-30): every F32/F16/BF16
            // tensor passes through verbatim under upstream state_dict
            // names + the `vokra.rmvpe.*` chunk group carries the
            // primary-source hparams (hop / sr / n_mels / n_fft /
            // win_length / n_class / cents_per_class / base_hz).
            // Provenance = MIT (Permissive — no runtime-side
            // attribution obligation).
            let report = models::rmvpe::convert_rmvpe_file(input, output, license)?;
            let notes = vec![format!(
                "rmvpe: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Rmvpe,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        ModelKind::Crepe => {
            // M5 gap follow-up (2026-07-30): CREPE — a sibling of RMVPE
            // (F0 pitch-extractor tier) but with a Keras / TensorFlow
            // upstream that needs the offline
            // `tools/parity/keras_h5_to_safetensors.py` bridge, which
            // also emits the JSON side-car this converter requires
            // (capacity / hop / fmin / fmax). Route to
            // `convert_crepe_file` instead of the plain path.
            return Err(ConvertError::Usage(
                "crepe needs a --config config.json (emitted by \
                 tools/parity/keras_h5_to_safetensors.py alongside the flattened safetensors); \
                 use convert_crepe_file"
                    .to_owned(),
            ));
        }
        ModelKind::PyannoteSegmentation => {
            // 2026-07-30 license half unblock (docs/license-audit.md §3.1
            // row 263 = 2026-07-30 yousan ☑ Commercial, DIARIZE_OP
            // blocker text "trigger + license" → "trigger only"): every
            // F32 / F16 / BF16 tensor passes through verbatim under the
            // upstream state_dict names + the `vokra.pyannote.*` chunk
            // group carries the PyanNet primary-source hparams
            // (sample_rate / sincnet.stride / lstm.hidden_size /
            // lstm.num_layers / lstm.bidirectional / lstm.monolithic /
            // linear.hidden_size / linear.num_layers /
            // num_powerset_classes = 7 for segmentation-3.0).
            // Provenance = MIT (Permissive — HF cardData primary source
            // verified 2026-07-30, `gated: auto` is access control only).
            // Runtime binder is Wave 2 loud-partial per
            // `docs/handoff/pyannote-implementation-plan-2026-07-30.md`.
            let report = models::pyannote_segmentation::convert_pyannote_segmentation_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "pyannote-segmentation: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::PyannoteSegmentation,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Qwen3TtsBase17B (Wave 4 variant-enum extension, 2026-08-01) ===
        ModelKind::Qwen3TtsBase17B => {
            // Phase 3 extension (added 2026-08-01): dispatch through the
            // shared `models::qwen3_tts::convert_variant` path with the
            // 1.7B-Base variant selector. Talker axes are byte-identical
            // to the 1.7B-CustomVoice / VoiceDesign siblings
            // (hidden=2048 / ffn=6144 / text_hidden=2048 / n_layer=28,
            // primary source
            // `Qwen/Qwen3-TTS-12Hz-1.7B-Base/config.json.talker_config`,
            // fetched 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」);
            // only the HF release id + `vokra.model.name` /
            // `vokra.provenance.upstream_hf` stamps differ. Provenance
            // = apache-2.0 end-to-end (Permissive — same posture as
            // every other Qwen3-TTS release; LM + codec + tokenizer +
            // speaker encoder all under a single apache-2.0 grant).
            let (builder, report) = models::qwen3_tts::convert_variant(
                bytes,
                models::qwen3_tts::Qwen3TtsVariant::_1_7B_Base,
            )?;
            let mut notes = vec![format!(
                "qwen3-tts-1.7b-base: {} float weights written verbatim, {} non-float skipped ({} BF16 passthrough)",
                report.written, report.skipped_non_float, report.bf16_passthrough,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("qwen3-tts-1.7b-base warning: {n}")),
            );
            (builder, notes)
        }
        // === Qwen3TtsCustomVoice17B (from wf_022575ce-077-2) ===
        ModelKind::Qwen3TtsCustomVoice17B => {
            // Phase 3 extension (added 2026-07-30): dispatch through the
            // shared `models::qwen3_tts::convert_variant` path with the
            // 1.7B-CustomVoice variant selector. The talker axes widen
            // to hidden=2048 / ffn=6144 (primary source
            // `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/config.json.talker_config`,
            // fetched 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」);
            // every other constant matches the 0.6B sibling. Provenance
            // = apache-2.0 end-to-end (Permissive — same posture as the
            // 0.6B release).
            let (builder, report) = models::qwen3_tts::convert_variant(
                bytes,
                models::qwen3_tts::Qwen3TtsVariant::_1_7B_CustomVoice,
            )?;
            let mut notes = vec![format!(
                "qwen3-tts-1.7b-customvoice: {} float weights written verbatim, {} non-float skipped ({} BF16 passthrough)",
                report.written, report.skipped_non_float, report.bf16_passthrough,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("qwen3-tts-1.7b-customvoice warning: {n}")),
            );
            (builder, notes)
        }
        // === Qwen3TtsVoiceDesign17B (from wf_022575ce-077-2) ===
        ModelKind::Qwen3TtsVoiceDesign17B => {
            // Phase 3 extension (added 2026-07-30): identical talker /
            // code-predictor axes to the CustomVoice sibling; only the
            // `vokra.model.name` stamp + provenance model_id differ
            // (primary source
            // `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/config.json.tts_model_type
            // = "voice_design"` vs `"custom_voice"`, fetched 2026-07-30 —
            // CLAUDE.md「ハルシネーション厳禁」). Provenance = apache-2.0
            // end-to-end (Permissive — same posture as CustomVoice).
            let (builder, report) = models::qwen3_tts::convert_variant(
                bytes,
                models::qwen3_tts::Qwen3TtsVariant::_1_7B_VoiceDesign,
            )?;
            let mut notes = vec![format!(
                "qwen3-tts-1.7b-voicedesign: {} float weights written verbatim, {} non-float skipped ({} BF16 passthrough)",
                report.written, report.skipped_non_float, report.bf16_passthrough,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("qwen3-tts-1.7b-voicedesign warning: {n}")),
            );
            (builder, notes)
        }
        // === Qwen3Asr (from wf_022575ce-077-1) ===
        ModelKind::Qwen3Asr => {
            // SoTA plan Phase 5 ASR fleet (2026-07-30): Alibaba
            // Qwen3-ASR family. The generic `convert_file` dispatch
            // path routes to the flagship 1.7B variant default; the
            // CLI's `--model qwen3-asr-0.6b` slug picks the 0.6B
            // variant through `convert_qwen3_asr_file_with_variant`
            // (a `pub use` re-export lower in this file). Provenance
            // = apache-2.0 end-to-end (Permissive) per both HF
            // model cards' `cardData.license` (CC-verified 2026-07-30).
            let report = models::qwen3_asr::convert_qwen3_asr_file(input, output, license)?;
            let notes = vec![format!(
                "qwen3-asr: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Qwen3Asr,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Wav2Vec2Ctc (from wf_022575ce-077-1) ===
        ModelKind::Wav2Vec2Ctc => {
            // SoTA plan Phase 5 ASR fleet (2026-07-30): wav2vec 2.0
            // CTC family. The generic `convert_file` dispatch path
            // routes to `base-960h` (the smallest / most widely-used
            // release) default; the CLI's per-variant slugs
            // (`wav2vec2-large-xlsr-53-*` / `wav2vec2-large-xlsr-53-japanese`
            // / etc.) pick the specific variant through
            // `convert_wav2vec2_ctc_file_with_variant` (a `pub use`
            // re-export lower in this file). Provenance = apache-2.0
            // (Permissive) per each variant's HF `cardData.license`
            // (CC-verified 2026-07-30).
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file(input, output, license)?;
            let notes = vec![format!(
                "wav2vec2-ctc: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Wav2Vec2Ctc,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Mms1bAll (2026-08-02 wave, routes through wav2vec2_ctc
        // — parent workflow REUSE HINT) ===
        ModelKind::Mms1bAll => {
            // Meta MMS-1B-All (`facebook/mms-1b-all`, cc-by-nc-4.0).
            // 1B wav2vec 2.0 backbone + 1000+ per-language CTC
            // adapters (~2000 sibling files in the repo). Base tensor
            // path reuses the wav2vec2_ctc converter via the dedicated
            // `Variant::Mms1bAll` arm with **placeholder axes**
            // (routes to LargeXlsr53Base sibling — parent workflow
            // SIZE NOTE forbids downloading the 4.00 GB checkpoint /
            // config.json for real primary-source transcription).
            // Faithful `name = "mms-1b-all"` +
            // `upstream_hf = "facebook/mms-1b-all"` stamps for the
            // placeholder-axis refusal guardrail (M2-13 runtime gate).
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant(
                input,
                output,
                models::wav2vec2_ctc::Variant::Mms1bAll,
                license,
            )?;
            let notes = vec![format!(
                "mms-1b-all: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read (via \
                 wav2vec2_ctc converter — placeholder axes, per-language adapter loader is a \
                 follow-up wave)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Mms1bAll,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Data2vecAudioBase (2026-08-02 wave, routes through
        // wav2vec2_ctc — tensor names identical, only name +
        // upstream_hf differ) ===
        ModelKind::Data2vecAudioBase => {
            // Baevski et al. 2022 (arXiv:2202.03555):
            // `facebook/data2vec-audio-base-960h` (apache-2.0). Shares
            // the wav2vec 2.0 base downstream inference topology +
            // Conv1D feature-extractor + LibriSpeech 960h English char
            // CTC head with `wav2vec2-base-960h` — data2vec differs in
            // the pretraining objective (contextualised latent
            // representation prediction with an EMA teacher), not the
            // downstream inference arch. The dedicated
            // `Variant::Data2vecAudioBase960h` overrides only `name` +
            // `upstream_hf` so the stamped GGUF faithfully reports the
            // data2vec-audio release instead of masquerading as the
            // wav2vec2 sibling.
            let report = models::wav2vec2_ctc::convert_wav2vec2_ctc_file_with_variant(
                input,
                output,
                models::wav2vec2_ctc::Variant::Data2vecAudioBase960h,
                license,
            )?;
            let notes = vec![format!(
                "data2vec-audio-base-960h: {} float weights written verbatim ({} BF16 \
                 passthrough — runtime widens to f32 exactly at load), {} non-float skipped, \
                 {} tensors read (via wav2vec2_ctc converter — tensor names identical)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Data2vecAudioBase,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossTts (from wf_022575ce-077-2) ===
        ModelKind::MossTts => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::Delay,
                license,
            )?;
            let notes = vec![format!(
                "moss-tts: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (variant=delay, backbone=qwen3-8b, n_vq=32)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossTts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossTtsV15 (from wf_022575ce-077-2) ===
        ModelKind::MossTtsV15 => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::DelayV15,
                license,
            )?;
            let notes = vec![format!(
                "moss-tts-v1.5: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (variant=delay, backbone=qwen3-8b, n_vq=32)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossTtsV15,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossTtsNano (from wf_022575ce-077-2) ===
        ModelKind::MossTtsNano => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::Nano,
                license,
            )?;
            let notes = vec![format!(
                "moss-tts-nano: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (variant=nano, backbone=gpt2, n_vq=16, sr=48kHz)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossTtsNano,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossTtsLocal (from wf_022575ce-077-2) ===
        ModelKind::MossTtsLocal => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::Local,
                license,
            )?;
            let notes = vec![format!(
                "moss-tts-local: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (variant=local, backbone=qwen3-2.5b, n_vq=12, sr=48kHz)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossTtsLocal,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossAudio4bInstruct (2026-08-02 wave) ===
        // Reuses the MossTts converter (parent workflow REUSE HINT)
        // via the new AudioInstruct4b variant. Placeholder axes
        // inherited from Local family; distinct sub-arch tag
        // `vokra.moss_tts.variant = "audio_4b"` lets a runtime
        // dispatcher recognise the artifact and refuse to bind the
        // placeholder axes until the follow-up wave lands the true
        // primary-source hparam transcription. The provenance triple
        // (NAME + upstream_hf + license = apache-2.0 Permissive +
        // category = s2s) is faithful.
        ModelKind::MossAudio4bInstruct => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::AudioInstruct4b,
                license,
            )?;
            let notes = vec![format!(
                "moss-audio-4b-instruct: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped (variant=audio_4b, backbone=qwen3, \
                 axes=placeholder-from-local, category=s2s, ~8 GB BF16 across 3 shards — \
                 primary-source hparam transcription is a follow-up)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossAudio4bInstruct,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossAudio8bInstruct (2026-08-02 wave) ===
        // Reuses the MossTts converter (parent workflow REUSE HINT)
        // via the new AudioInstruct8b variant. Same custom-code
        // audio-LLM architecture as MossAudio4bInstruct, larger 8B
        // backbone (~9.05 GB BF16 across 4 shards — vast.ai required).
        // Placeholder axes inherited from Local family; distinct
        // sub-arch tag `vokra.moss_tts.variant = "audio_8b"` lets a
        // runtime dispatcher recognise the artifact and refuse to
        // bind the placeholder axes until the follow-up wave lands
        // the true primary-source hparam transcription.
        ModelKind::MossAudio8bInstruct => {
            let report = models::moss_tts::convert_moss_tts_file(
                input,
                output,
                models::moss_tts::MossTtsVariant::AudioInstruct8b,
                license,
            )?;
            let notes = vec![format!(
                "moss-audio-8b-instruct: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped (variant=audio_8b, backbone=qwen3, \
                 axes=placeholder-from-local, category=s2s, ~9.05 GB BF16 across 4 shards — \
                 primary-source hparam transcription is a follow-up)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossAudio8bInstruct,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MeloTtsEnglish (from wf_022575ce-077-3) ===
        ModelKind::MeloTtsEnglish => {
            let report = models::melotts::convert_melotts_file(
                input,
                output,
                models::melotts::MeloVariant::English,
                license,
            )?;
            let notes = vec![format!(
                "melotts-english: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MeloTtsEnglish,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MeloTtsChinese (from wf_022575ce-077-3) ===
        ModelKind::MeloTtsChinese => {
            let report = models::melotts::convert_melotts_file(
                input,
                output,
                models::melotts::MeloVariant::Chinese,
                license,
            )?;
            let notes = vec![format!(
                "melotts-chinese: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MeloTtsChinese,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MeloTtsKorean (from wf_022575ce-077-3) ===
        ModelKind::MeloTtsKorean => {
            let report = models::melotts::convert_melotts_file(
                input,
                output,
                models::melotts::MeloVariant::Korean,
                license,
            )?;
            let notes = vec![format!(
                "melotts-korean: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MeloTtsKorean,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MeloTtsSpanish (Wave 8 2026-08-01) ===
        ModelKind::MeloTtsSpanish => {
            let report = models::melotts::convert_melotts_file(
                input,
                output,
                models::melotts::MeloVariant::Spanish,
                license,
            )?;
            let notes = vec![format!(
                "melotts-spanish: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MeloTtsSpanish,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MeloTtsJapanese (Wave 8 2026-08-01) ===
        ModelKind::MeloTtsJapanese => {
            let report = models::melotts::convert_melotts_file(
                input,
                output,
                models::melotts::MeloVariant::Japanese,
                license,
            )?;
            let notes = vec![format!(
                "melotts-japanese: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MeloTtsJapanese,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SpeechT5Tts (from wf_022575ce-077-3) ===
        ModelKind::SpeechT5Tts => {
            let report = models::speecht5::convert_speecht5_file(input, output, license)?;
            let notes = vec![format!(
                "speecht5-tts: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SpeechT5Tts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === ParlerTtsMiniMultilingual (from wf_022575ce-077-3) ===
        ModelKind::ParlerTtsMiniMultilingual => {
            let report = models::parler::convert_parler_file(
                input,
                output,
                models::parler::ParlerVariant::MiniMultilingual,
                license,
            )?;
            let notes = vec![format!(
                "parler-tts (mini-multilingual): {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ParlerTtsMiniMultilingual,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === IndicParlerTts (from wf_022575ce-077-3) ===
        ModelKind::IndicParlerTts => {
            let report = models::parler::convert_parler_file(
                input,
                output,
                models::parler::ParlerVariant::IndicParler,
                license,
            )?;
            let notes = vec![format!(
                "indic-parler-tts: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::IndicParlerTts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === ParlerTtsMiniV1English (Wave 4 land 2026-08-01) ===
        // English-only mini-v1 (predecessor of the multilingual v1.1
        // variant). Same converter as its siblings; ParlerVariant dispatch
        // stamps the per-variant top-level vocab_size (32128 here vs the
        // multilingual's 90714) + the distinct upstream_hf / model_name /
        // variant_tag provenance chunks.
        ModelKind::ParlerTtsMiniV1English => {
            let report = models::parler::convert_parler_file(
                input,
                output,
                models::parler::ParlerVariant::MiniV1English,
                license,
            )?;
            let notes = vec![format!(
                "parler-tts-mini-v1 (English): {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::ParlerTtsMiniV1English,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === VieNeuTts (from wf_022575ce-077-3) ===
        ModelKind::VieNeuTts => {
            let report = models::vieneu::convert_vieneu_file(input, output, license)?;
            let notes = vec![format!(
                "vieneu-tts: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::VieNeuTts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Bark (from wf_022575ce-077-3) ===
        ModelKind::Bark => {
            let report = models::bark::convert_bark_file(
                input,
                output,
                models::bark::BarkVariant::Full,
                license,
            )?;
            let notes = vec![format!(
                "bark: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Bark,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === BarkSmall (from wf_022575ce-077-3) ===
        ModelKind::BarkSmall => {
            let report = models::bark::convert_bark_file(
                input,
                output,
                models::bark::BarkVariant::Small,
                license,
            )?;
            let notes = vec![format!(
                "bark-small: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::BarkSmall,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === HifiganVocoder (from wf_022575ce-077-4) ===
        ModelKind::HifiganVocoder => {
            // SoTA plan Phase D1 (2026-07-30): SpeechBrain HiFi-GAN vocoder
            // (LibriTTS 22050Hz, apache-2.0). BF16 pass-through skeleton
            // mirror of wespeaker / ecapa_tdnn; runtime forward primitive
            // already lives in `vokra_ops::hifigan` (M3-07). Real-weight
            // parity is deferred to owner (docs/license-audit.md §3.1
            // sign-off).
            let report =
                models::hifigan_vocoder::convert_hifigan_vocoder_file(input, output, license)?;
            let notes = vec![format!(
                "hifigan-vocoder: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::HifiganVocoder,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Speecht5Hifigan (2026-07-31 wave) ===
        ModelKind::Speecht5Hifigan => {
            // 2026-07-31 wave: Microsoft SpeechT5 HiFi-GAN vocoder
            // (`microsoft/speecht5_hifigan`, MIT). BF16 pass-through
            // skeleton mirror of hifigan_vocoder / wespeaker /
            // ecapa_tdnn. Distinct arch tag from `hifigan_vocoder`
            // (SpeechBrain sibling) — different sampling rate,
            // normalize_before layout, and HF-transformers naming.
            // Runtime binding + real-weight parity are deferred to
            // owner (docs/license-audit.md §3.1 sign-off). Upstream
            // ships pytorch_model.bin only — pre-flatten to
            // safetensors via
            // tools/parity/speecht5_hifigan_prepare_checkpoint.py.
            let report =
                models::speecht5_hifigan::convert_speecht5_hifigan_file(input, output, license)?;
            let notes = vec![format!(
                "speecht5-hifigan: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Speecht5Hifigan,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === BigVGan (from wf_022575ce-077-4) ===
        ModelKind::BigVGan => {
            // SoTA plan Phase D2-D5 (2026-07-30): NVIDIA BigVGAN vocoder
            // family (MIT). Default dispatch path tags the GGUF as the
            // `V2_24khz100Band256x` variant — the most-downloaded release
            // (per HF API 2026-07-30) and the same variant `bigvgan_v2_*`
            // shorthand aliases resolve to. Callers who want a different
            // variant use the standalone `convert_bigvgan_file` entry
            // point with an explicit `BigVGanVariant`. This mirrors the
            // Chatterbox default-multilingual dispatch pattern.
            let report = models::bigvgan::convert_bigvgan_file(
                input,
                output,
                models::bigvgan::BigVGanVariant::V2_24khz100Band256x,
                license,
            )?;
            let notes = vec![format!(
                "bigvgan: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped, variant v2_24khz_100band_256x (default; use \
                 convert_bigvgan_file for other variants)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::BigVGan,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Vocos (2026-08-01 wave, mel-24khz + encodec-24khz variants) ===
        ModelKind::Vocos => {
            // 2026-08-01 wave: Charactr AI Vocos vocoder
            // (`charactr/vocos-mel-24khz` = HF top vocoder by download
            // 2.85M dl; `charactr/vocos-encodec-24khz` = second
            // variant, both MIT). Distinct arch tag `vocos` from every
            // HiFi-GAN family sibling — Vocos is a Fourier-space
            // vocoder (ConvNeXt V2 backbone + iSTFT head), not
            // time-domain upsample + MRF. BF16 pass-through skeleton
            // mirror of speecht5_hifigan / bigvgan / focalcodec;
            // runtime binding is deferred to owner. Upstream ships
            // torch pickle only — pre-flatten via
            // tools/parity/vocos_prepare_checkpoint.py.
            //
            // This path is the enum-arm default (Mel24khz); the
            // encodec-24khz variant is picked from the raw `--model`
            // slug via `convert_file_with_slug` — this arm mirrors
            // the BigVGan / Focalcodec posture (single ModelKind +
            // slug dispatch, no ModelKind bloat for pure-metadata
            // variants).
            let report = models::vocos::convert_vocos_file(
                input,
                output,
                models::vocos::VocosVariant::Mel24khz,
                license,
            )?;
            let notes = vec![format!(
                "vocos (mel_24khz): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (use --model vocos-encodec-24khz or convert_vocos_file with an \
                 explicit VocosVariant for the encodec variant)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Vocos,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Focalcodec (from wf_022575ce-077-4; 25Hz + 12.5Hz variants added 2026-07-31) ===
        ModelKind::Focalcodec => {
            // SoTA plan Phase D6 (2026-07-30): lucadellalib FocalCodec
            // 50Hz (apache-2.0). Only member of the vocoder+codec fleet
            // that ships model.safetensors directly (no torch-pickle
            // prepare step). BF16 pass-through skeleton mirror of
            // funcodec / wespeaker; runtime binding is deferred to owner.
            //
            // This path is the enum-arm default (Hz50); the 25Hz /
            // 12.5Hz variants (2026-07-31) are picked from the raw
            // `--model` slug via `convert_file_with_slug` — this arm
            // mirrors the BigVGan / TigerSeparator posture (single
            // ModelKind + slug dispatch, no ModelKind bloat for
            // pure-metadata variants).
            let report = models::focalcodec::convert_focalcodec_file(
                input,
                output,
                license,
                models::focalcodec::FocalcodecVariant::Hz50,
            )?;
            let notes = vec![format!(
                "focalcodec (50hz): {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (use --model focalcodec-25hz / focalcodec-12-5hz or \
                 convert_focalcodec_file with an explicit FocalcodecVariant for other variants)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Focalcodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === GraniteSpeech (2026-08-01 Wave 3) ===
        ModelKind::GraniteSpeech => {
            let report = models::granite_speech::convert_granite_speech_file(
                input,
                output,
                models::granite_speech::GraniteSpeechVariant::V4_1_2B,
                license,
            )?;
            let notes = vec![format!(
                "granite-speech-4.1-2b: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Wavtokenizer (2026-08-01 Wave 3) ===
        ModelKind::Wavtokenizer => {
            let report = models::wavtokenizer::convert_wavtokenizer_file(input, output, license)?;
            let notes = vec![format!(
                "wavtokenizer-large-speech-75token: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MusicGenMedium (2026-08-01 Wave 5 music-generation add) ===
        ModelKind::MusicGenMedium => {
            // Meta AudioCraft MusicGen-Medium (`facebook/musicgen-medium`,
            // cc-by-nc-4.0). First music-generation target to land a
            // converter (post-2026-07-30 scope expansion). 1.5B
            // autoregressive transformer LM over EnCodec RVQ tokens
            // conditioned on frozen T5 text encoder (Copet et al. 2023,
            // arXiv:2306.05284). BF16 pass-through skeleton mirror of
            // xcodec2 / wavtokenizer with **NonCommercial default** — the
            // M2-13 runtime gate refuses to load in commercial mode unless
            // overridden via `--license <spdx>`. Scale ~11.4 GB =
            // vast.ai handoff per memory
            // `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB unsafe
            // for this class of publish). runtime binder + real-weight
            // parity deferred to owner sign-off (§3.1).
            let report =
                models::musicgen_medium::convert_musicgen_medium_file(input, output, license)?;
            let notes = vec![format!(
                "musicgen-medium: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-4.0 default, NonCommercial fail-closed unless --license overrides)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MusicGenLarge (2026-08-01 Wave 5 music-generation add) ===
        ModelKind::MusicGenLarge => {
            // Meta AudioCraft MusicGen-Large (`facebook/musicgen-large`,
            // cc-by-nc-4.0). Second music-generation target — top rung
            // of the MusicGen family (3.3B, vs 1.5B for sibling medium).
            // BF16 pass-through skeleton mirror of musicgen_medium /
            // xcodec2 / wavtokenizer with **NonCommercial default** —
            // the M2-13 runtime gate refuses to load in commercial
            // mode unless overridden via `--license <spdx>`. Scale
            // ~19.5 GB = vast.ai handoff per memory
            // `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB
            // unsafe for this class of publish; larger than sibling
            // MusicGen-Medium ~11.4 GB). runtime binder + real-weight
            // parity deferred to owner sign-off (§3.1) — shared with
            // sibling medium binder (identical topology, only dims
            // differ).
            let report =
                models::musicgen_large::convert_musicgen_large_file(input, output, license)?;
            let notes = vec![format!(
                "musicgen-large: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-4.0 default, NonCommercial fail-closed unless --license overrides)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MusicGenMelody (2026-08-02 Wave 5 sibling) ===
        ModelKind::MusicGenMelody => {
            // Meta AudioCraft MusicGen-Melody (`facebook/musicgen-melody`,
            // cc-by-nc-4.0). Third music-generation target — medium 1.5B
            // LM + 12-bin chromagram conditioning frontend concatenated
            // to the T5 text conditioning stream (Copet et al. 2023,
            // arXiv:2306.05284 §3.3). Reuses the medium BF16 pass-
            // through pipeline via the sibling wrapper
            // `convert_musicgen_melody_file` (which routes through the
            // shared `convert_musicgen_family_file` helper) — no new
            // `musicgen_melody.rs` module. Scale ~6 GB = vast.ai
            // handoff per memory `[[feedback-large-models-on-vast-ai]]`.
            let report =
                models::musicgen_medium::convert_musicgen_melody_file(input, output, license)?;
            let notes = vec![format!(
                "musicgen-melody: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-4.0 default, NonCommercial fail-closed unless --license overrides; \
                 reuses medium converter — chroma frontend runtime op deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === AudioGenMedium (2026-08-01 Wave 5 residual) ===
        ModelKind::AudioGenMedium => {
            // Meta AudioCraft AudioGen-Medium (`facebook/audiogen-medium`,
            // cc-by-nc-4.0). MusicGen sibling — identical topology
            // (shared `musicgen` arch tag), tuned on environmental sounds
            // / SFX (Kreuk et al. 2023, arXiv:2209.15352). Scale ~3.7 GB
            // = local convert safe. NonCommercial default fail-closed.
            let report =
                models::audiogen_medium::convert_audiogen_medium_file(input, output, license)?;
            let notes = vec![format!(
                "audiogen-medium: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-4.0 default, NonCommercial fail-closed unless --license overrides)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MusicGenSmall (2026-08-01 Wave 6 residual) ===
        ModelKind::MusicGenSmall => {
            let report =
                models::musicgen_small::convert_musicgen_small_file(input, output, license)?;
            let notes = vec![format!(
                "musicgen-small: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (cc-by-nc-4.0 default, NonCommercial fail-closed)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Qwen2Audio (2026-08-01 Wave 6 residual) ===
        ModelKind::Qwen2Audio => {
            let report = models::qwen2_audio::convert_qwen2_audio_file(input, output, license)?;
            let notes = vec![format!(
                "qwen2-audio-7b-instruct: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (apache-2.0 default)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Qwen25Omni7b (2026-08-02 Wave residual, Thinker + Talker unified omni) ===
        ModelKind::Qwen25Omni7b => {
            let report =
                models::qwen2_5_omni_7b::convert_qwen2_5_omni_7b_file(input, output, license)?;
            let notes = vec![format!(
                "qwen2-5-omni-7b: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (apache-2.0 default, Permissive — distinct arch tag \
                 `qwen2-omni` from sibling `qwen2_audio` audio-only; runtime binder deferred \
                 to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === VibeVoiceAsr (2026-08-01 Wave 6 residual) ===
        ModelKind::VibeVoiceAsr => {
            let report = models::vibevoice_asr::convert_vibevoice_asr_file(input, output, license)?;
            let notes = vec![format!(
                "vibevoice-asr: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === AceStep (2026-08-01 Wave 6 residual) ===
        ModelKind::AceStep => {
            let report = models::ace_step::convert_ace_step_file(input, output, license)?;
            let notes = vec![format!(
                "ace-step-1.5: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default — flagship MIT music-gen)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === HubertLargeLs960 (2026-08-01 Wave 7 residual) ===
        ModelKind::HubertLargeLs960 => {
            let report = models::hubert_large_ls960::convert_hubert_large_ls960_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "hubert-large-ls960: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (apache-2.0 default, Permissive — distinct arch tag `hubert` from sibling \
                 wav2vec2 despite shared ops)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === W2vBert2 (2026-08-04 hf-audio-gap SSL residual) ===
        ModelKind::W2vBert2 => {
            // w2v-BERT 2.0 (`facebook/w2v-bert-2.0`, MIT). ~580M-
            // parameter self-supervised speech encoder = Conformer body
            // + dual (wav2vec2-style contrastive + BERT-style MLM) SSL
            // branches (Chung et al. 2021 arXiv:2108.06209). Standalone
            // ModelKind = the encoder alone, distinct from composite
            // consumers `vieneu` / `seamless_m4t_v2_large` where it
            // rides as an INTERNAL subgraph. BF16 pass-through skeleton
            // mirror of sibling `hubert_large_ls960` / `moonshine_base`
            // / `musicgen_small` / `openwakeword`. Distinct arch tag
            // `w2v-bert-2` from siblings hubert / wav2vec2_ctc /
            // data2vec-audio (Conformer vs vanilla Transformer body +
            // combined SSL objectives) — FR-EX-08 (no silent op-shape
            // misroute) requires the distinct arch tag. Scale ~2.16 GB
            // = **vast.ai handoff** per memory `[[feedback-large-
            // models-on-vast-ai]]` (exceeds the 2 GB CC workflow
            // local-convert owner threshold); the converter + §3.1
            // audit row + signoff_match.py entries land today, actual
            // publish runs on vast.ai per `docs/handoff/vast-ai-large-
            // model-publish.md`.
            let report = models::w2v_bert_2::convert_w2v_bert_2_file(input, output, license)?;
            let notes = vec![format!(
                "w2v-bert-2.0: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (mit default, Permissive — distinct arch tag `w2v-bert-2` from sibling \
                 hubert / wav2vec2_ctc / data2vec-audio despite shared SSL family; ~2.16 GB \
                 single-file safetensors = vast.ai handoff per feedback-large-models-on-vast-ai)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === AudioLdm2 (2026-08-01 Wave 5 music-generation add) ===
        ModelKind::AudioLdm2 => {
            // AudioLDM 2 (`cvssp/audioldm2`, **cc-by-nc-sa-4.0**). Text-
            // to-audio latent-diffusion generator (Liu et al. 2024 ICML
            // arXiv:2308.05734). Multi-encoder bundle: VAE + latent-
            // diffusion U-Net + HiFi-GAN vocoder + T5-base + CLAP text
            // encoder + GPT-2 audio-caption LM (~8.5 GB total). BF16
            // pass-through skeleton mirror of musicgen_medium /
            // musicgen_large / xcodec2 with **NonCommercialShareAlike
            // default** — doubly restrictive: the M2-13 runtime gate
            // refuses to load in commercial mode (NC gate) AND any
            // downstream republish must carry the license forward (SA
            // cascade). Override via `--license <spdx>` only when the
            // caller legitimately holds the weight under a different
            // SPDX id. Scale ~8.5 GB = vast.ai handoff per memory
            // `[[feedback-large-models-on-vast-ai]]` (M1 iMac 16 GB
            // unsafe on the upper edge — multi-encoder bundle doubles
            // peak resident to ~17 GB). **Publish blocked (sa-cascade-
            // defer)**: no entry in `signoff_match.py::REPO_TO_SIGNOFF_
            // ROWS`, no ☑ sign-off in §3.1 — owner ADR required to
            // resolve the SA cascade onto Vokra-added artifacts.
            // runtime binder + real-weight parity deferred to owner
            // sign-off (§3.1) — new op surface (latent-diffusion
            // sampler + VAE + HiFi-GAN, distinct from `flow_sampler`
            // which targets flow-matching).
            let report = models::audioldm2::convert_audioldm2_file(input, output, license)?;
            let notes = vec![format!(
                "audioldm2: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-sa-4.0 default, NonCommercialShareAlike fail-closed — NC gate + SA \
                 cascade both in force unless --license overrides; publish blocked pending \
                 owner ADR per docs/license-audit.md §3.1)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === AudioLdm2Large (2026-08-02 Wave 8 sibling) ===
        ModelKind::AudioLdm2Large => {
            // AudioLDM 2 Large (`cvssp/audioldm2-large`, **cc-by-nc-sa-
            // 4.0**). Wider/deeper sibling of the base AudioLDM 2
            // variant — same multi-encoder bundle topology (VAE + U-Net
            // + HiFi-GAN + T5 + CLAP + GPT-2 audio-caption LM), only
            // model dims differ. Reuses the base BF16 pass-through arm
            // via the shared `convert_audioldm2_family_file` helper —
            // no new `audioldm2_large.rs` module (the sibling-in-place
            // pattern the base module docstring anticipated; mirror of
            // the musicgen_medium / musicgen_melody 2026-08-02 landing).
            // The wrapper flips only the `vokra.model.name` +
            // `vokra.provenance.{model_id,source,upstream_hf}` chunks
            // to the LARGE_* spellings. `LicenseClass::
            // NonCommercialShareAlike` default fail-closed (NC gate +
            // SA cascade); override via `--license <spdx>` only when
            // the caller legitimately holds the weight under a
            // different SPDX id. Scale ~7 GB = **vast.ai handoff** per
            // memory `[[feedback-large-models-on-vast-ai]]` (M1 iMac
            // 16 GB unsafe on the upper edge — the multi-encoder
            // bundle roughly doubles peak resident on the pass).
            // **Publish blocked (sa-cascade-defer)**: no entry in
            // `signoff_match.py::REPO_TO_SIGNOFF_ROWS` for the
            // `vokra/audioldm2-large` slug, no ☑ sign-off in §3.1
            // (owner ADR required to resolve the SA cascade onto
            // Vokra-added artifacts, same posture as sibling base).
            let report = models::audioldm2::convert_audioldm2_large_file(input, output, license)?;
            let notes = vec![format!(
                "audioldm2-large: {} float weights written verbatim ({} BF16 passthrough — \
                 runtime widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (cc-by-nc-sa-4.0 default, NonCommercialShareAlike fail-closed — NC gate + SA \
                 cascade both in force unless --license overrides; reuses base converter — \
                 publish blocked pending owner ADR per docs/license-audit.md §3.1)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === BsRoformer (2026-08-01 Wave 5 music-separation add) ===
        ModelKind::BsRoformer => {
            // BS-Roformer / Mel-Band Roformer (third-party mirror
            // `chenmozhijin/BSRoformer-GGUF`, **weight provenance
            // unclear**). First music-source-separation converter — Lu
            // et al. 2023 arXiv:2310.01809 dual-path frequency-band
            // Transformer over an STFT spectrogram. BF16 pass-through
            // skeleton mirror of vits_ja / musicgen_large / xcodec2 —
            // architecture code is MIT (`github.com/lucidrains/BS-
            // RoFormer`) but the paper released no weights, so every
            // checkpoint in the wild is a downstream retraining under
            // mixed licenses. **LicenseClass::RedistributionForbidden
            // fail-closed default** — a converter cannot know which
            // SPDX id covers the caller's checkpoint. The M2-13 runtime
            // gate does not block *loading* (unlike NonCommercial
            // which requires the research flag) — the fail-closed
            // publish gate at `LicenseClass::redistributable() = false`
            // is what blocks upload. `--license <spdx>` overrides at
            // the outer boundary (the vits-ja / Whisper / kokoro
            // pattern). **Publish blocked** at
            // `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`
            // (unlisted slug fails closed as `UNKNOWN_REPO`; owner ADR
            // selecting a specific checkpoint + license is the
            // prerequisite to a first publish).
            let report = models::bs_roformer::convert_bs_roformer_file(input, output, license)?;
            let notes = vec![format!(
                "bs-roformer: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read \
                 (weight-provenance-unclear default, RedistributionForbidden fail-closed — \
                 publish gate refuses upload unless --license overrides to a known SPDX id; \
                 publish blocked pending owner ADR per docs/license-audit.md §3.1)",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Openwakeword (2026-08-02 Wave residual, custom-KWS) ===
        ModelKind::Openwakeword => {
            // openWakeWord (dscripka, apache-2.0). Small custom-KWS
            // MLP/CNN family over precomputed melspec — audio-dialect
            // `kws` op entry (FR-OP `kws`). BF16 pass-through skeleton
            // mirror of sibling musicgen_small / hubert_large_ls960.
            // HF API rate-limited (401); upstream GitHub primary
            // source verified apache-2.0. Default license `apache-2.0`
            // + `LicenseClass::Permissive` (sibling Silero / CAM++ /
            // piper-plus first-party Permissive posture). Scale
            // ~0.01 GB = local convert safe.
            let report = models::openwakeword::convert_openwakeword_file(input, output, license)?;
            let notes = vec![format!(
                "openwakeword: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (apache-2.0 default, Permissive — audio-dialect \
                 `kws` op entry per FR-OP; distinct arch tag `openwakeword`, category `kws`)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MoonshineTiny (2026-08-02 Wave residual, raw-audio ASR) ===
        ModelKind::MoonshineTiny => {
            // Moonshine-Tiny (UsefulSensors, MIT). 27M transformer enc-
            // dec ASR with raw-audio Conv1D front-end (no mel) + rotary
            // + SwiGLU (Jeffries et al. 2024, arXiv:2410.15608). Distinct
            // arch tag `moonshine` from sibling Whisper (different audio
            // input path + different attention/MLP variants) — silently
            // sharing would misroute runtime dispatch at the audio-input
            // boundary (FR-EX-08). BF16 pass-through skeleton mirror of
            // sibling musicgen_small / hubert_large_ls960 / openwakeword.
            // Default license `mit` + Permissive (Whisper / piper-plus /
            // Silero / CAM++ first-party posture). Scale ~0.11 GB =
            // local convert safe on M1 iMac 16 GB.
            let report =
                models::moonshine_tiny::convert_moonshine_tiny_file(input, output, license)?;
            let notes = vec![format!(
                "moonshine-tiny: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default, Permissive — distinct arch tag `moonshine` \
                 from sibling Whisper: raw-audio Conv1D front-end + rotary + SwiGLU)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MoonshineBase (2026-08-02 Wave residual, raw-audio ASR) ===
        ModelKind::MoonshineBase => {
            // Moonshine-Base (UsefulSensors, MIT). 61.5M transformer
            // enc-dec ASR with raw-audio Conv1D front-end (no mel) +
            // rotary + SwiGLU (Jeffries et al. 2024, arXiv:2410.15608).
            // Sibling to Moonshine-Tiny — same arch family, wider/deeper
            // backbone (~2.3× parameter count per upstream release
            // manifest). Shares arch tag `moonshine` with Tiny at the
            // runtime side; distinct ModelKind at the converter side
            // (different tensor shapes — a Base checkpoint fed to a
            // Tiny loader would misroute at load, FR-EX-08). BF16 pass-
            // through skeleton mirror of sibling moonshine_tiny /
            // musicgen_small / hubert_large_ls960 / openwakeword.
            // Default license `mit` + Permissive (Whisper / piper-plus /
            // Silero / CAM++ / Moonshine-Tiny first-party posture).
            // Scale ~0.25 GB = local convert safe on M1 iMac 16 GB.
            let report =
                models::moonshine_base::convert_moonshine_base_file(input, output, license)?;
            let notes = vec![format!(
                "moonshine-base: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default, Permissive — distinct arch tag `moonshine` \
                 from sibling Whisper: raw-audio Conv1D front-end + rotary + SwiGLU; sibling \
                 Moonshine-Tiny same arch, wider/deeper backbone)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === DemucsHtdemucs (2026-08-02 Wave residual, music source separation) ===
        ModelKind::DemucsHtdemucs => {
            // HT-Demucs (facebook/demucs, MIT). Hybrid transformer Demucs
            // (Rouard et al. 2023, arXiv:2211.08553) — U-Net waveform
            // branch + spectrogram branch + cross-domain self-attention,
            // 4-source music separation. Distinct arch tag `demucs` from
            // sibling SepFormer / TIGER separators (different internal
            // domain + different output branching — FR-EX-08 forbids
            // silent misroute across separator families). BF16 pass-
            // through skeleton mirror of sibling moonshine_base /
            // musicgen_small / hubert_large_ls960 / openwakeword. Default
            // license `mit` + Permissive (Whisper / piper-plus / Silero /
            // CAM++ / Moonshine first-party posture). Scale ~0.50 GB =
            // local convert safe on M1 iMac 16 GB.
            let report =
                models::demucs_htdemucs::convert_demucs_htdemucs_file(input, output, license)?;
            let notes = vec![format!(
                "demucs-htdemucs: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (mit default, Permissive — distinct arch tag `demucs` \
                 from sibling SepFormer / TIGER separators: hybrid U-Net waveform + \
                 spectrogram branch + cross-domain self-attention, 4-source music separation)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === UltravoxV05Llama321b (2026-08-02 Wave residual, audio-text-to-text multimodal) ===
        ModelKind::UltravoxV05Llama321b => {
            // Ultravox v0.5 (fixie-ai/ultravox-v0_5-llama-3_2-1b, MIT).
            // Audio-text-to-text multimodal = Llama-3.2-1B decoder + Whisper
            // encoder + lightweight projection adapter. Both underlying
            // arches (Llama + Whisper) already supported by sibling
            // converters + runtime primitives; new wiring is the adapter
            // projection + multimodal prompt template (runtime-side, not
            // converter-side). Distinct arch tag `ultravox` from sibling
            // Voxtral (Mistral decoder) / Qwen2-Audio (Qwen2 decoder) — the
            // decoder backbone fixes tensor layout + tokenizer + rope base,
            // so FR-EX-08 forbids silent shape misroute across the three.
            // BF16 pass-through skeleton mirror of sibling demucs_htdemucs /
            // moonshine_base / musicgen_small / hubert_large_ls960 /
            // openwakeword. Default license `mit` + Permissive (Whisper /
            // piper-plus / Silero / CAM++ / Moonshine first-party posture).
            // Scale ~1.83 GB = local convert safe on M1 iMac 16 GB.
            let report =
                models::ultravox_v0_5_llama_3_2_1b::convert_ultravox_v0_5_llama_3_2_1b_file(
                    input, output, license,
                )?;
            let notes = vec![format!(
                "ultravox-v0-5-llama-3-2-1b: {} float weights written verbatim \
                 ({} BF16 passthrough), {} non-float skipped (mit default, Permissive — \
                 distinct arch tag `ultravox` from sibling Voxtral / Qwen2-Audio: \
                 Llama-3.2-1B decoder + Whisper encoder + projection adapter, \
                 audio-text-to-text multimodal — runtime binder deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === XttsV2 (2026-08-02 Wave residual, multilingual zero-shot voice-cloning TTS) ===
        ModelKind::XttsV2 => {
            // XTTS-v2 (coqui/XTTS-v2, coqui-public-model-license = NonCommercial
            // T4 tier). GPT-2 backbone + DVAE token decoder + HiFi-GAN vocoder
            // head. Distinct arch tag `xtts` from sibling TTS families
            // (piper-plus VITS2 / Kokoro iSTFTNet / CosyVoice2 FSQ+HiFTNet) —
            // FR-EX-08 requires the distinct arch tag because the tokenizer +
            // decoder head + vocoder triple would misroute silently otherwise.
            // BF16 pass-through skeleton mirror of sibling ultravox_v0_5_llama_3_2_1b
            // / musicgen_small / hubert_large_ls960. Default license
            // `coqui-public-model-license` + NonCommercial (X-Codec-2 /
            // MusicGen T4 precedent). Scale ~1.90 GB = local convert safe.
            let report = models::xtts_v2::convert_xtts_v2_file(input, output, license)?;
            let notes = vec![format!(
                "xtts-v2: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (coqui-public-model-license default, NonCommercial \
                 fail-closed — publish requires --allow-noncommercial per T4 precedent; \
                 runtime binder deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === ConvTasnetLibri1mix (2026-08-02 Wave residual, first Copyleft-tier separator) ===
        ModelKind::ConvTasnetLibri1mix => {
            // ConvTasNet Libri1Mix Enhancement (JorisCos/ConvTasNet_Libri1Mix_
            // enhsingle_16k, cc-by-sa-4.0). Asteroid recipe fine-tune —
            // single-speaker enhancement, 16 kHz, one output stream.
            // Distinct arch tag `conv_tasnet` from sibling separator
            // families (sepformer / demucs / tiger_separator / bs_roformer /
            // mp_senet) — FR-EX-08 forbids silent shape misroute across
            // separator families. First entry on the Copyleft weight-license
            // arm (SA cascade — a derived GGUF is itself CC-BY-SA), T3 tier
            // redistributable with original licence preserved. Owner runs
            // `bin_to_safetensors.py` before pointing this converter at the
            // resulting `.safetensors` (upstream ships raw pytorch_model.bin
            // pickle which FR-LD-05 forbids the runtime from reading
            // directly). Scale ~20 MB = local convert safe.
            let report = models::conv_tasnet_libri1mix::convert_conv_tasnet_libri1mix_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "conv-tasnet-libri1mix: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (cc-by-sa-4.0 default, Copyleft — SA cascade preserved on \
                 derivatives, T3 tier redistributable with original licence; runtime binder \
                 deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SeamlessM4tV2Large (2026-08-02 Wave residual, unity-2 4-subgraph any-to-any) ===
        ModelKind::SeamlessM4tV2Large => {
            // Meta SeamlessM4T v2 (facebook/seamless-m4t-v2-large,
            // cc-by-nc-4.0). 2.3B unified any-to-any speech-and-text
            // translation model — ASR + T2TT + S2TT + T2ST + S2ST across
            // ~100 source / ~35 target speech languages. Ships 2 safetensors
            // shards + `.pt` duplicates + `vocoder_v2.pt` (~9.00 GB total)
            // — the converter walks whatever bytes the caller hands in
            // (typical publish path pre-flattens shards + vocoder to a
            // single safetensors offline; NFR-DS-02 forbids pickle in the
            // runtime, FR-LD-05 forbids arbitrary code execution at load).
            // Distinct arch tag `unity-2` (Meta's fairseq2 dispatch name)
            // for the 4 subgraphs (w2v-BERT enc + text dec + T2U + HiFi-GAN
            // vocoder) — FR-EX-08 forbids silent shape misroute across
            // sibling M4T v1 / MMS / Whisper families. BF16 pass-through
            // skeleton mirror of sibling musicgen_small / qwen2_audio.
            // Default license cc-by-nc-4.0 + NonCommercial (X-Codec 2 /
            // MusicGen T4 precedent). Scale ~9.00 GB = vast.ai handoff per
            // memory `[[feedback-large-models-on-vast-ai]]`.
            let report = models::seamless_m4t_v2_large::convert_seamless_m4t_v2_large_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "seamless-m4t-v2-large: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (cc-by-nc-4.0 default, NonCommercial fail-closed — \
                 publish requires --allow-noncommercial per T4 precedent; runtime binder for \
                 the 4-subgraph unity-2 arch deferred to owner sign-off)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Snac (2026-08-01 Wave 3) ===
        ModelKind::Snac => {
            // 2026-08-01 Wave 3: SNAC — Multi-Scale Neural Audio
            // Codec (Siuzdak et al. 2024, MIT). Default dispatch
            // path tags the GGUF as the Hz24 variant — the
            // higher-download release (~452k dl/mo vs 44kHz ~1.3k
            // dl/mo per HF API 2026-08-01) and the primary consumer
            // of Orpheus-TTS + MOSS voice + CSM-1B-adjacent TTS
            // stacks. Callers who want the 44 kHz music-quality
            // variant (4 RVQ levels + 32-frame local attention)
            // use `--model snac-44khz` (routed via
            // `convert_file_with_slug`) or the standalone
            // `convert_snac_file` entry with an explicit
            // `SnacVariant`. This mirrors the Focalcodec /
            // BigVGan default-canonical dispatch pattern (single
            // ModelKind + slug dispatch, no ModelKind bloat for
            // pure-metadata variants).
            let report = models::snac::convert_snac_file(
                input,
                output,
                license,
                models::snac::SnacVariant::Hz24,
            )?;
            let notes = vec![format!(
                "snac (24khz): {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (use --model snac-44khz or convert_snac_file with an \
                 explicit SnacVariant for the music-quality variant)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Snac,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MossAudioTokenizer (2026-08-01 Wave 3) ===
        ModelKind::MossAudioTokenizer => {
            // MOSS-Audio-Tokenizer — the codec half of the MOSS-TTS
            // pipeline (OpenMOSS-Team, apache-2.0). Default dispatch
            // path tags the GGUF as the Full variant — the canonical
            // release the MOSS-TTS consumer pairs with. Callers who
            // want the Nano distilled variant (~22M params vs Full
            // ~1.77B) use `--model moss-audio-tokenizer-nano` (routed
            // via `convert_file_with_slug`) or the standalone
            // `convert_moss_audio_tokenizer_variant_file` entry with
            // an explicit `MossAudioTokenizerVariant`. Mirrors the
            // Snac / Focalcodec / BigVGan default-canonical dispatch
            // pattern (single ModelKind + slug dispatch, no
            // ModelKind bloat for pure-metadata variants).
            let report = models::moss_audio_tokenizer::convert_moss_audio_tokenizer_variant_file(
                input,
                output,
                models::moss_audio_tokenizer::MossAudioTokenizerVariant::Full,
                license,
            )?;
            let notes = vec![format!(
                "moss-audio-tokenizer (full): {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped (use --model moss-audio-tokenizer-nano or \
                 convert_moss_audio_tokenizer_variant_file with an explicit \
                 MossAudioTokenizerVariant for the distilled Nano variant)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MossAudioTokenizer,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === YueUpsampler (2026-08-01 Wave 3 sibling-pair) ===
        ModelKind::YueUpsampler => {
            // 2026-08-01 Wave 3 sibling-pair add: YuE bundle vocoder
            // half (`m-a-p/YuE-upsampler`, apache-2.0). Vocos backbone
            // + iSTFT head trained on YuE codec latents (44.1 kHz
            // output, 3528-point iSTFT). Distinct arch tag
            // `yue_upsampler` from sibling Charactr AI `vocos` because
            // the axes differ. BF16 pass-through skeleton mirror of
            // vocos / snac / focalcodec / speecht5_hifigan; runtime
            // binder + real-weight parity deferred to owner sign-off
            // (`docs/license-audit.md` §3.1 sign-off queue). Upstream
            // ships torch pickle only — pre-flatten via
            // `tools/parity/yue_bundle_prepare_checkpoint.py`.
            let report = models::yue_bundle::convert_yue_bundle_variant_file(
                input,
                output,
                models::yue_bundle::YueBundleVariant::Upsampler,
                license,
            )?;
            let notes = vec![format!(
                "yue-upsampler: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === YueXcodecMini (2026-08-01 Wave 3 sibling-pair) ===
        ModelKind::YueXcodecMini => {
            // 2026-08-01 Wave 3 sibling-pair add: YuE bundle codec
            // half (`m-a-p/xcodec_mini_infer`, apache-2.0). Multi-part
            // bundle = SoundStream RVQ codec (16 kHz / 25 Hz, 640x
            // downsample, 6 target bandwidths up to 6 kbps) + HuBERT-
            // base semantic encoder + Vocos decoder head (byte-
            // identical to the sibling `YueUpsampler` variant). The
            // semantic-encoder fusion is what distinguishes YuE
            // xcodec-mini from plain RVQ / FSQ codecs → distinct
            // arch tag `yue_xcodec_mini` from every sibling codec.
            // Prep bridge role-prefixes tensors under `codec.*` /
            // `semantic.*` / `decoder.*` in the merged safetensors so
            // a future `YueXcodecMini::from_gguf` can locate the three
            // sub-modules. BF16 pass-through skeleton mirror of
            // vocos / snac / focalcodec; runtime binder + real-weight
            // parity deferred to owner sign-off (§3.1).
            let report = models::yue_bundle::convert_yue_bundle_variant_file(
                input,
                output,
                models::yue_bundle::YueBundleVariant::XcodecMini,
                license,
            )?;
            let notes = vec![format!(
                "yue-xcodec-mini: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Facodec (2026-08-01 Wave 3) ===
        ModelKind::Facodec => {
            // Amphion NaturalSpeech 3 FACodec — factorized VQ (FVQ)
            // codec (apache-2.0). Default dispatch path tags the GGUF
            // as the V2 variant — the canonical highest-quality
            // codec-only pair (encoder_v2 + decoder_v2). Callers who
            // want a different variant (v1 base pair, or the
            // redecoder-v{1,2} zero-shot voice-conversion variants)
            // use `--model naturalspeech3-facodec-v1` /
            // `-redecoder-v1` / `-redecoder-v2` (routed via
            // `convert_file_with_slug`) or the standalone
            // `convert_naturalspeech3_facodec_variant_file` entry
            // with an explicit `FacodecVariant`. Mirrors the
            // Snac / MossAudioTokenizer / Focalcodec / BigVGan
            // default-canonical dispatch pattern (single ModelKind +
            // slug dispatch, no ModelKind bloat for pure-metadata
            // variants).
            //
            // **Voice-conversion policy note**: the redecoder-v{1,2}
            // variants enable zero-shot voice conversion — see
            // `models::naturalspeech3_facodec` module docstring for
            // the CLAUDE.md 設計判断 8 routing question. The default
            // V2 arm here is unambiguously codec-class (encoder +
            // decoder, no redecoder) and belongs in the main zoo.
            let report =
                models::naturalspeech3_facodec::convert_naturalspeech3_facodec_variant_file(
                    input,
                    output,
                    models::naturalspeech3_facodec::FacodecVariant::V2,
                    license,
                )?;
            let notes = vec![format!(
                "facodec (v2): {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped (use --model naturalspeech3-facodec-v1 / \
                 -redecoder-v1 / -redecoder-v2 or \
                 convert_naturalspeech3_facodec_variant_file with an explicit \
                 FacodecVariant for the other three variants)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Facodec,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === TigerSeparator (from wf_022575ce-077-5) ===
        ModelKind::TigerSeparator => {
            let report = models::tiger::convert_tiger_file(
                input,
                output,
                license,
                models::tiger::TigerVariant::Dnr,
            )?;
            let notes = vec![format!(
                "tiger-dnr: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::TigerSeparator,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === TigerSpeech (from wf_022575ce-077-5) ===
        ModelKind::TigerSpeech => {
            let report = models::tiger::convert_tiger_file(
                input,
                output,
                license,
                models::tiger::TigerVariant::Speech,
            )?;
            let notes = vec![format!(
                "tiger-speech: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::TigerSpeech,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MpSenet (from wf_022575ce-077-5) ===
        ModelKind::MpSenet => {
            let report = models::mp_senet::convert_mp_senet_file(input, output, license)?;
            let notes = vec![format!(
                "mp-senet: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MpSenet,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MpSenetDns (2026-08-02) — shared converter with MpSenet; a
        // distinct ModelKind arm keeps the §3.1 sign-off / publish-side
        // provenance stamp trackable per HF repo. The underlying
        // `convert_mp_senet_file` already stamps
        // `vokra.provenance.upstream_hf` = "JacobLinCool/MP-SENet-DNS"
        // (see `models::mp_senet::UPSTREAM_HF`), so the artifact is
        // byte-identical to the MpSenet arm — the split lives in
        // [`ModelKind`] + `signoff_match`, not in the converter body.
        ModelKind::MpSenetDns => {
            let report = models::mp_senet::convert_mp_senet_file(input, output, license)?;
            let notes = vec![format!(
                "mp-senet-dns: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped (shared converter with MpSenet)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MpSenetDns,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === MetricganPlus (from wf_022575ce-077-5) ===
        ModelKind::MetricganPlus => {
            let report =
                models::metricgan_plus::convert_metricgan_plus_file(input, output, license)?;
            let notes = vec![format!(
                "metricgan-plus: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::MetricganPlus,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepFormer (from wf_022575ce-077-5) ===
        ModelKind::SepFormer => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Wsj02mix,
            )?;
            let notes = vec![format!(
                "sepformer-wsj02mix: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepFormer,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerWham16kEnh (from wf_022575ce-077-5) ===
        ModelKind::SepformerWham16kEnh => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Wham16kEnhancement,
            )?;
            let notes = vec![format!(
                "sepformer-wham16k-enhancement: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerWham16kEnh,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerWhamr16k (from wf_022575ce-077-5) ===
        ModelKind::SepformerWhamr16k => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Whamr16k,
            )?;
            let notes = vec![format!(
                "sepformer-whamr16k: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerWhamr16k,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerLibri2Mix (Wave 4 candidate, 2026-08-01) ===
        // Shares the sepformer converter — LibriMix is a 2-speaker
        // separation head like Wsj02mix; the two differ only in the
        // training corpus (LibriMix = LibriSpeech-derived CC-BY-4.0
        // vs WSJ0-2mix = proprietary WSJ0). Distinct variant ensures
        // the correct vokra.model.name / vokra.provenance.upstream_hf
        // / vokra.sepformer.variant stamps land on the artifact.
        ModelKind::SepformerLibri2Mix => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Libri2Mix,
            )?;
            let notes = vec![format!(
                "sepformer-libri2mix: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerLibri2Mix,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerLibri3Mix (Wave 4 candidate, 2026-08-01) ===
        // Shares the sepformer converter — Libri3Mix is the 3-speaker
        // cocktail-party sibling of Libri2Mix; same LibriSpeech-derived
        // training corpus family and same SepFormer topology, only the
        // masker output head branches into 3 parallel speaker streams
        // instead of 2. Distinct variant ensures the correct
        // vokra.model.name / vokra.provenance.upstream_hf /
        // vokra.sepformer.variant / vokra.sepformer.n_out (=3) stamps
        // land on the artifact (NOT silently inherited from the
        // 2-speaker sibling = wrong CDN attribution + wrong binder
        // output-stream axis).
        ModelKind::SepformerLibri3Mix => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Libri3Mix,
            )?;
            let notes = vec![format!(
                "sepformer-libri3mix: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped; vokra.sepformer.n_out=3 stamped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerLibri3Mix,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerWhamr8k (Wave 4 candidate, 2026-08-01) ===
        // Shares the sepformer converter — WHAMR! 8 kHz is the base-
        // sample-rate sibling of Whamr16k; same reverberant conditioning
        // + masker head, only the sample rate differs. Distinct variant
        // ensures the correct vokra.model.name /
        // vokra.provenance.upstream_hf / vokra.sepformer.variant stamps
        // land on the artifact (NOT silently inherited from the 16 kHz
        // sibling = wrong CDN attribution).
        ModelKind::SepformerWhamr8k => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Whamr8k,
            )?;
            let notes = vec![format!(
                "sepformer-whamr (8 kHz): {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerWhamr8k,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SepformerDns4Enh (Wave 4 candidate, 2026-08-01) ===
        // Shares the sepformer converter — Microsoft DNS-4 is a
        // single-speaker enhancement task like the WHAM! / WHAMR!
        // enhancement siblings; the two differ only in the training
        // corpus (Microsoft DNS-4 vs WSJ0-derived WHAM! / WHAMR!).
        // Distinct variant ensures the correct vokra.model.name /
        // vokra.provenance.upstream_hf / vokra.sepformer.variant
        // stamps land on the artifact (NOT silently inherited from
        // any WHAM! / WHAMR! sibling = wrong CDN attribution; all
        // enhancement variants share n_out = 1 so provenance is the
        // only discriminator at load time).
        ModelKind::SepformerDns4Enh => {
            let report = models::sepformer::convert_sepformer_file(
                input,
                output,
                license,
                models::sepformer::SepformerVariant::Dns4Enhancement,
            )?;
            let notes = vec![format!(
                "sepformer-dns4-16k-enhancement: {} float weights written verbatim ({} BF16 \
                 passthrough), {} non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SepformerDns4Enh,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Sgmse (SoTA plan candidate wave, 2026-08-04) ===
        // SpeechBrain SGMSE-VoiceBank — score-based diffusion speech
        // enhancement (NCSN++ v2 + OUVE SDE reverse sampler, N=30
        // steps per upstream hyperparams.yaml). First real weight in
        // the Vokra catalog to exercise the M3-05 flow_sampler + ODE
        // solver op family — every existing enhancement row
        // (MetricGAN+ / MP-SENet / DeepFilterNet3 / FacebookDenoiser)
        // is masking or time-domain UNet, none exercises the flow
        // sampler path. BF16 pass-through skeleton (F32 / F16 / BF16
        // verbatim); the internal NCSN++ v2 + reverse sampler forward
        // is a `loud-partial` follow-up (RMVPE / Charsiu / MOSS-Audio-
        // Tokenizer / MioCodec landing precedent). apache-2.0 default
        // (Permissive — HF cardData API primary source verified
        // 2026-08-04). Upstream `.ckpt` payload is a flat state_dict
        // of the internal NCSN++ v2 network (SpeechBrain's Pretrainer
        // adds the `score_model.` prefix at load time); the offline
        // `.ckpt` → `.safetensors` bridge lives in
        // `tools/parity/sgmse_prepare_checkpoint.py`.
        ModelKind::Sgmse => {
            let report = models::sgmse::convert_sgmse_file(input, output, license)?;
            let notes = vec![format!(
                "sgmse-voicebank: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Sgmse,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === FsmnVad (SoTA plan Phase 5 VAD-2, 2026-07-30) ===
        ModelKind::FsmnVad => {
            // FSMN-VAD converter — full hparam chunk stamp + verbatim
            // float pass-through (F32 / F16 / BF16). Every hparam axis
            // is a compile-time constant transcribed from the released
            // FunASR checkpoint; a future non-default variant would
            // introduce a --config side-car (owner follow-up).
            let report = models::fsmn_vad::convert_fsmn_vad_file(input, output, license)?;
            let notes = vec![format!(
                "fsmn-vad: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped; vokra.fsmn_vad.* hparam chunk group stamped \
                 (n_blocks=4, input_dim=400, proj_dim=128, hidden_dim=128, lorder=20, \
                 rorder=0, n_class=2, n_mels=80, lfr_m=5, lfr_n=1, sample_rate=16000)",
                report.written, report.bf16_passthrough, report.skipped_non_float,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::FsmnVad,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === FireredVad (from wf_022575ce-077-6) ===
        ModelKind::FireredVad => {
            let report = models::firered_vad::convert_firered_vad_file(input, output, license)?;
            let notes = vec![format!(
                "firered-vad: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::FireredVad,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === SmartTurn (from wf_022575ce-077-6) ===
        ModelKind::SmartTurn => {
            let report = models::smart_turn::convert_smart_turn_file(input, output, license)?;
            let notes = vec![format!(
                "smart-turn: {} float weights written verbatim ({} BF16 passthrough), {} \
                 non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::SmartTurn,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Clap (from wf_022575ce-077-6) ===
        ModelKind::Clap => {
            let report = models::clap::convert_clap_file(input, output, license)?;
            let notes = vec![format!(
                "clap: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Clap,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === Ast (from wf_022575ce-077-6) ===
        ModelKind::Ast => {
            let report = models::ast::convert_ast_file(input, output, license)?;
            let notes = vec![format!(
                "ast: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Ast,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === LangIdVoxlingua107 (from wf_022575ce-077-6) ===
        ModelKind::LangIdVoxlingua107 => {
            // F7 variant — the module's default entry stamps the
            // VoxLingua107 name + upstream_hf slug.
            let report = models::speechbrain_lang_id::convert_speechbrain_lang_id_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "lang-id-voxlingua107: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::LangIdVoxlingua107,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === LangIdCommonLanguage (from wf_022575ce-077-6) ===
        ModelKind::LangIdCommonLanguage => {
            // F9 sibling — same ECAPA-TDNN topology, distinct
            // vokra.model.name + upstream_hf via the Variant enum.
            let report = models::speechbrain_lang_id::convert_speechbrain_lang_id_variant(
                input,
                output,
                license,
                models::speechbrain_lang_id::Variant::CommonLanguage,
            )?;
            let notes = vec![format!(
                "lang-id-commonlanguage: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::LangIdCommonLanguage,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === XVector (from wf_022575ce-077-6) ===
        ModelKind::XVector => {
            let report = models::xvector::convert_xvector_file(input, output, license)?;
            let notes = vec![format!(
                "xvector: {} float weights written verbatim ({} BF16 passthrough), {} non-float \
                 skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::XVector,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === DeepfakeDetection (from wf_022575ce-077-6) ===
        ModelKind::DeepfakeDetection => {
            let report = models::deepfake_detection::convert_deepfake_detection_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "deepfake-detection: {} float weights written verbatim ({} BF16 passthrough), \
                 {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::DeepfakeDetection,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === KyutaiTts (from wf_022575ce-077-7) ===
        ModelKind::KyutaiTts => {
            let report = models::kyutai_tts::convert_kyutai_tts_file(input, output, license)?;
            let notes = vec![format!(
                "kyutai-tts: {} float weights written verbatim ({} BF16 passthrough — runtime \
                 widens to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::KyutaiTts,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === AudioboxAesthetics (from wf_022575ce-077-7) ===
        ModelKind::AudioboxAesthetics => {
            let report = models::audiobox_aesthetics::convert_audiobox_aesthetics_file(
                input, output, license,
            )?;
            let notes = vec![format!(
                "audiobox-aesthetics: {} float weights written verbatim ({} BF16 passthrough — \
                 upstream is F32 today, the arm keeps future distilled BF16 fine-tunes \
                 verbatim), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::AudioboxAesthetics,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // === VoxtralMiniRealtime — routes to shared Voxtral streaming path ===
        // Voxtral-Mini-4B-Realtime-2602 is a Voxtral-family sibling (Mistral,
        // apache-2.0, ~8 GB single-file safetensors). The Voxtral converter
        // already supports streaming per-tensor (M5 gap A-3,
        // `convert_voxtral_file_streaming`), so an 8 GB checkpoint on a 16
        // GB host stays within footprint. Route through the shared Voxtral
        // path with a shape-only (no --config) invocation — the runtime
        // rejects `0`-placeholder hparams at forward per FR-EX-08 (Voxtral
        // family posture), so producing a Voxtral GGUF without the upstream
        // `config.json` is honest as a "converter output existence" artefact
        // and callers who want a runnable GGUF must pass `--config`.
        ModelKind::VoxtralMiniRealtime => {
            let cfg = VoxtralConfig::default();
            return convert_voxtral_file(input, &cfg, output);
        }
        // === CohereTranscribe (from wf_022575ce-077-7) ===
        ModelKind::CohereTranscribe => {
            return Err(ConvertError::Usage(
                "cohere-transcribe-03-2026 is a TIER 2 defer marker (defer-gated=true; HF \
                 cardData gated=`auto` requires an owner-authenticated HF token that has clicked \
                 \"Accept license\" on the model card at \
                 huggingface.co/CohereLabs/cohere-transcribe-03-2026 — CC cannot discharge that \
                 acceptance step). Owner runs the conversion after acceptance. This arm is \
                 fail-closed per FR-EX-08 (no silent fallback)."
                    .to_owned(),
            ));
        }
        // === NemotronAsrStreaming — owner ADR complete 2026-07-30 ===
        // License = OpenMDW-1.1 = Permissive (MIT-analog for ML weights,
        // CC 直接照合 2026-07-30 = commercial + redistribution 可、no
        // share-alike / no NC、attribution = notice 保持のみ)。
        // `LicenseClass::from_license_str("openmdw")` → `Permissive`。
        // BF16 pass-through converter (mirror of wespeaker / omniasr_ctc)。
        ModelKind::NemotronAsrStreaming => {
            let report = models::nemotron_asr::convert_nemotron_asr_file(input, output, license)?;
            let notes = vec![format!(
                "nemotron-asr-streaming: {} float weights written verbatim ({} BF16 \
                 passthrough — runtime widens to f32 exactly at load), {} non-float skipped, \
                 {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model,
                tensor_count: report.written,
                metadata_count: 0,
                output_bytes: std::fs::metadata(output)?.len(),
                notes,
            });
        }
        // M5-16 (FR-OP-83): FCPE — pass every F32 / F16 / BF16 tensor through
        // verbatim and stamp the `vokra.model.arch = "fcpe"` +
        // `vokra.model.category = "f0"` + `vokra.provenance.upstream_hf =
        // "CNChTu/FCPE"` chunk group. Provenance defaults to **mit /
        // Permissive** (CC-verified 2026-07-30). A caller who trained on a
        // different corpus (or holds the weight under a distinct SPDX id)
        // overrides at the outer `--license <spdx>` boundary below.
        ModelKind::Fcpe => {
            let report = models::fcpe::convert_fcpe_file(input, output, license)?;
            let notes = vec![format!(
                "fcpe: {} float weights written verbatim ({} BF16 passthrough — runtime widens \
                 to f32 exactly at load), {} non-float skipped, {} tensors read",
                report.written, report.bf16_passthrough, report.skipped_non_float, report.read,
            )];
            return Ok(ConvertSummary {
                model: ModelKind::Fcpe,
                tensor_count: report.written,
                metadata_count: 0,
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
/// Presets map to a `QuantPolicy` with the shape documented
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
/// `models::piper_plus` for the naming / metadata contract.
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

/// Converts a Silero VAD ONNX checkpoint into a GGUF written to `output`,
/// stamped with the caller-chosen upstream release variant.
///
/// This is the variant-aware sibling of the plain [`convert_file`] path for
/// Silero VAD. The ONNX weight extraction is identical (topology is
/// architecturally unchanged across upstream v5 and v6.2.1 per
/// `snakers4/silero-vad` `tinygrad_model.py` + `utils_vad.py`, verified
/// 2026-07-30); what this path adds is the `vokra.silero.version`
/// release-tag metadata and the variant-specific `vokra.model.name` /
/// `vokra.provenance.source` strings. The result is a self-describing
/// artifact whose provenance survives publication to a public model hub
/// (the pre-tagging default path in `convert_file` deliberately omits the
/// tag to stay byte-identical with the committed parity fixture — SPEC
/// "Conversion").
///
/// The runtime loader accepts either shape: an absent tag defaults to
/// [`vokra_core::gguf::silero::SileroVariant::V5`] (backward compat with
/// pre-tagging fixtures), while a present tag with an unknown value is a
/// fail-closed [`vokra_core::VokraError::ModelLoad`] (FR-EX-08).
pub fn convert_silero_file(
    input: &Path,
    output: &Path,
    variant: vokra_core::gguf::silero::SileroVariant,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (builder, report) = models::silero::convert_variant(bytes, variant)?;
    let notes = vec![format!(
        "silero {}: {} float weights written (both rates, sr8k.*/sr16k.*), \
         {} non-float constants skipped, {} op-scope float strays skipped",
        variant.tag(),
        report.written,
        report.skipped_non_float,
        report.skipped_stray,
    )];
    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::SileroVad,
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
/// `models::kokoro` — briefly: at least one of `{vocab: {symbol:id},
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

/// Convert a prepared marl/crepe checkpoint into a Vokra GGUF (M5 gap
/// follow-up, 2026-07-30).
///
/// `input` is the flat safetensors and `config` the JSON side-car that
/// `tools/parity/keras_h5_to_safetensors.py` writes from the upstream
/// `.h5` release (Keras / TensorFlow never enters the runtime — zero-dep
/// NFR-DS-02 / FR-LD-05, the same offline-prepare split as DAC / Kokoro
/// / UTMOS).
///
/// The 38-tensor mapping is total: every declared tensor must exist with
/// exactly the dims the capacity implies, and any upstream tensor left
/// over at the end is a hard `ConvertError::Parse` rather than a silent
/// drop (FR-EX-08 — same posture as `convert_utmos_file`).
pub fn convert_crepe_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let cfg = models::crepe::CrepeConvertConfig::parse(&config_bytes)?;
    let (builder, report) = models::crepe::convert(bytes, &cfg)?;

    let notes = vec![format!(
        "crepe: {} tensor(s) emitted from {} upstream tensor(s) (all consumed), capacity={}",
        report.written, report.read, report.capacity,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Crepe,
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
/// shape as the module-private `models::voxtral::VoxtralConfig`, re-exported
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
// NVIDIA TitaNet-Large speaker verification (nvidia/speakerverification_en_titanet_large,
// **cc-by-4.0** = AttributionRequired). File-based converter with per-call
// license override + FR-MD-09 attribution chunk stamped by default —
// the model module is `pub mod titanet` in models/mod.rs; re-exporting
// the surface here makes it reachable from external callers (same
// rationale as the speaker_3d / wespeaker / ecapa_tdnn re-exports:
// `pub fn` alone is dead code because `mod models` itself is private).
// Runtime port is out-of-scope — the M5-residual op
// `TITANET_SPEAKER_ENCODE_OP` (FR-OP-80 variant) is the anchor for a
// future landing; consumers today use CAM++ (`speaker_encode`)
// under Apache-2.0.
pub use models::titanet::{TitaNetReport, convert_titanet_file};
// coverage-audit-2026-08-03 Wave A: fjiang9/NKF-AEC (MIT). GitHub-only
// release (no HF mirror) so provenance stamps `upstream_url` rather
// than `upstream_hf`. File-based entry mirroring the speaker_3d /
// ecapa_tdnn re-export pattern.
pub use models::nkf_aec::{NkfAecReport, convert_nkf_aec_file};
// coverage-audit-2026-08-03 Wave A: Xiph RNNoise v0.2 (BSD-3-Clause).
// GitHub-only release (no HF mirror) so provenance stamps
// `upstream_url` rather than `upstream_hf`. File-based entry mirroring
// the nkf_aec / speaker_3d / ecapa_tdnn re-export pattern.
pub use models::rnnoise::{RnnoiseReport, convert_rnnoise_file};
// coverage-audit-2026-08-03 Wave A: Microsoft NSNet2 (MIT Permissive) —
// the DNS Challenge NR baseline. Standalone file-based entry point
// re-exported so external callers (integration tests / a future
// `vokra-cli` `--model nsnet2` invocation) can reach it without the
// private `models::nsnet2` module. Also routed through `ModelKind::Nsnet2`
// so `convert_file` / `convert_file_licensed` reach the same code path.
pub use models::nsnet2::{Nsnet2Report, convert_nsnet2_file};
// coverage-audit Wave A ticket `dnsmos-p808-p835` (2026-08-03): Microsoft
// DNSMOS P.808 + P.835 MOS predictor bundle — the first
// `category = "eval"` model in the converter tree. File-based entry
// point that reads the prepared safetensors from
// `tools/parity/dnsmos_prepare_checkpoint.py` (which flattens the two
// upstream ONNX checkpoints with `p808.` / `p835.` name prefixes).
// Wired through the `ModelKind::Dnsmos` dispatch arm in
// `convert_file_licensed` — the re-export here mirrors the
// `emotion2vec` / `speaker_3d` posture so external integration tests
// can reach `convert_dnsmos_file` + `DnsmosReport` without pulling in
// the private `models::dnsmos` module.
pub use models::dnsmos::{DnsmosReport, convert_dnsmos_file};
// coverage-audit wave-a (2026-08-03): FRCRN (alibabasglab/FRCRN,
// apache-2.0 Permissive). Category `denoise` — second `denoise` model
// in the converter tree after DeepFilterNet3, with a distinct arch tag
// (`frcrn`) so silently aliasing DFN3's manifest is impossible. Both a
// file-based entry point with an SPDX override argument AND the
// `ModelKind::Frcrn` dispatch arm above land the same bytes.
pub use models::frcrn::{FrcrnReport, convert_frcrn_file};
// ---- coverage-audit 2026-08-03 Wave B fast-track (13 variants) ----
pub use models::canary_1b_flash::{Canary1bFlashReport, convert_canary_1b_flash_file};
pub use models::firered_asr_aed_l::{FireredAsrAedLReport, convert_firered_asr_aed_l_file};
pub use models::hibiki::{HibikiReport, convert_hibiki_file};
pub use models::magpietts_v2602::{MagpiettsV2602Report, convert_magpietts_v2602_file};
pub use models::nemotron_speech_streaming_v2603::{
    NemotronSpeechStreamingV2603Report, convert_nemotron_speech_streaming_v2603_file,
};
pub use models::owsm_v4_medium_1b::{OwsmV4Medium1bReport, convert_owsm_v4_medium_1b_file};
pub use models::parakeet_tdt_1_1b::{ParakeetTdt11bReport, convert_parakeet_tdt_1_1b_file};
pub use models::parakeet_unified::{ParakeetUnifiedReport, convert_parakeet_unified_file};
pub use models::reazonspeech_nemo_v2::{
    ReazonspeechNemoV2Report, convert_reazonspeech_nemo_v2_file,
};
pub use models::sber_gigaam_multilingual::{
    SberGigaamMultilingualReport, convert_sber_gigaam_multilingual_file,
};
pub use models::sber_gigaam_v3::{SberGigaamV3Report, convert_sber_gigaam_v3_file};
pub use models::sensevoicesmall::{SenseVoiceSmallReport, convert_sensevoicesmall_file};
pub use models::sortformer_diar_4spk_v1::{
    SortformerDiar4spkV1Report, convert_sortformer_diar_4spk_v1_file,
};
pub use models::whisper_medusa_v1::{WhisperMedusaV1Report, convert_whisper_medusa_v1_file};
// coverage-audit-2026-08-03 Wave D T4 (non-commercial batch, 2026-08-04).
// 5 BF16 pass-through skeletons — all NonCommercial default (T4), publish
// requires `--allow-noncommercial`. stable_audio_open_small hard-maps the
// non-SPDX "stability-ai-community-license" to NonCommercial per the CPML
// precedent in `xtts_v2.rs`.
pub use models::chattts::{ChatTtsReport, convert_chattts_file};
pub use models::facebook_denoiser::{FacebookDenoiserReport, convert_facebook_denoiser_file};
pub use models::jasco_400m_chords_drums::{
    Jasco400mChordsDrumsReport, convert_jasco_400m_chords_drums_file,
};
pub use models::nisqa_v2_weight::{NisqaV2WeightReport, convert_nisqa_v2_weight_file};
pub use models::stable_audio_open_small::{
    StableAudioOpenSmallReport, convert_stable_audio_open_small_file,
};
// coverage-audit-2026-08-03 Wave A permissive continuation (2026-08-04).
// 7 BF16 pass-through skeletons all T1 Permissive tier.
pub use models::audioseal_real_weight::{
    AudiosealRealWeightReport, convert_audioseal_real_weight_file,
};
// hf-audio-gap-comprehensive-2026-07-30 §3.8 JA-vocoder complement wave
// (2026-08-04): Aratako/MioCodec-25Hz-44.1kHz-v2 (MIT). Standalone file-
// based entry point (mirror of the neucodec / bicodec / focalcodec re-
// export pattern); the runtime `ModelKind::MioCodec` dispatch arm above
// shares the same `models::miocodec::convert_miocodec_file` helper, so
// a caller who prefers `--model miocodec` via `convert_file_licensed`
// and a caller who calls `convert_miocodec_file` directly land the
// same bytes.
pub use models::htdemucs_multi::{HtdemucsMultiReport, convert_htdemucs_multi_file};
pub use models::miocodec::{MioCodecReport, convert_miocodec_file};
// SoTA plan candidate wave (2026-08-04): Neuphonic NeuTTS Air
// (apache-2.0) — Qwen2 0.5B LLM backbone emitting NeuCodec audio
// tokens. Standalone file-based entry point (mirror of the miocodec
// / neucodec / bicodec / focalcodec re-export pattern); the runtime
// `ModelKind::NeuTtsAir` dispatch arm above shares the same
// `models::neutts_air::convert_neutts_air_file` helper, so a caller
// who prefers `--model neutts-air` via `convert_file_licensed` and
// a caller who calls `convert_neutts_air_file` directly land the
// same bytes.
pub use models::mossformer2_ss_16k::{Mossformer2Ss16kReport, convert_mossformer2_ss_16k_file};
pub use models::neutts_air::{NeuTtsAirReport, convert_neutts_air_file};
pub use models::openwakeword_op::{OpenwakewordOpReport, convert_openwakeword_op_file};
pub use models::ten_vad::{TenVadReport, convert_ten_vad_file};
pub use models::torchaudio_squim::{TorchaudioSquimReport, convert_torchaudio_squim_file};
pub use models::utmosv2::{Utmosv2Report, convert_utmosv2_file};
// Music-understanding wave (2026-08-13): YAMNet — Google Research
// 521-class AudioSet audio-event classifier (MobileNetV1 backbone,
// ~15 MB edge model, apache-2.0 default). Standalone file-based entry
// point mirrors the utmosv2 / musicgen_medium re-export pattern for
// direct callers who prefer `convert_yamnet_file` over the
// `ModelKind::Yamnet` slug dispatch.
pub use models::yamnet::{YamnetReport, convert_yamnet_file};
// Music-understanding wave (2026-08-13): MERT — Music undERstanding
// self-supervised encoder (HuBERT-derived Conv1D + 24-layer Transformer,
// ~330M params, cc-by-nc-4.0 default). Standalone file-based entry
// point mirrors the yamnet / musicgen_medium re-export pattern.
pub use models::mert::{MertReport, convert_mert_file};
// Music-understanding wave (2026-08-13): MuQ — Self-supervised music
// representation learner (Mel-RVQ + BEATs teacher, MERT alternative,
// ~500M params, license unknown default). Standalone file-based entry
// point mirrors the mert / yamnet re-export pattern.
pub use models::muq::{MuqReport, convert_muq_file};
// SoTA plan Phase 5 emotion tier (2026-07-25): emotion2vec+ Large — the
// first `category = "emotion"` model in the converter tree. Standalone
// file-based entry point (not routed through `ModelKind` dispatch)
// exposes its `pub` API to external callers.
pub use models::emotion2vec::{Emotion2vecReport, convert_emotion2vec_file};
// SoTA plan Phase 5 VAD-2 (2026-07-30): FunASR FSMN-VAD — first-class
// audio-dialect op posture (distinct from Silero VAD v5's FR-LD-06
// 1:1 subgraph). Self-contained file-based entry point with SPDX
// override argument (mirror of the emotion2vec / xy_tokenizer /
// speaker_3d re-export pattern; `models::fsmn_vad` module is public
// for symmetry with the ModelKind dispatch above).
pub use models::fsmn_vad::{FsmnVadReport, convert_fsmn_vad_file};
pub use models::voxtral::VoxtralConfig;

/// The upstream Silero VAD release tag [`convert_silero_file`] accepts and
/// stamps into the emitted GGUF. Re-exported here so CLI / publisher /
/// integration-test call sites can name the enum without depending on
/// `vokra-core::gguf::silero` directly.
pub use vokra_core::gguf::silero::SileroVariant;
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
pub use models::bert_base::{BertBaseReport, convert_bert_base_file};
pub use models::deberta_v2::{ConvertReport, convert_deberta_v2_file};
pub use models::deberta_v3::convert_deberta_v3_file;
// `sbv2::ConvertReport` is re-exported under an alias, not the bare name:
// `deberta_v2::ConvertReport` already claims `vokra_convert::ConvertReport`
// above, and the two are distinct types (SBV2's carries `hparams_written`,
// DeBERTa's does not), so re-exporting both under the same crate-root name
// would collide (E0252).
pub use models::sbv2::{ConvertReport as SbV2ConvertReport, convert_sbv2_file};
// SoTA plan Phase 5 codec (2026-07-28): HKUSTAudio/xcodec2
// (**cc-by-nc-4.0** — HF card front-matter, `docs/license-audit.md` §3.1
// 2026-07-23 yousan = ☑ Research-only). Standalone file-based entry point
// with an SPDX override argument (mirror of the neucodec /
// step_audio2_mini pattern). The runtime `ModelKind::XCodec2` dispatch
// arm above shares the same `models::xcodec2::convert` internal helper,
// so a caller who prefers `--model xcodec2` via `convert_file_licensed`
// and a caller who calls `convert_xcodec2_file` directly land the same
// bytes.
pub use models::xcodec2::{XCodec2Report, convert_xcodec2_file};
// F0 pitch-extractor tier (2026-07-30): RMVPE — the first
// `category = "f0"` binder in the converter tree. Standalone file-based
// entry point (mirror of the emotion2vec re-export pattern; the
// `models::rmvpe` module is otherwise public so this re-export just
// preserves the canonical short-name spelling).
pub use models::rmvpe::{RmvpeReport, convert_rmvpe_file};
// Wave 5 music-separation add (2026-08-01): BS-Roformer / Mel-Band Roformer
// (**weight provenance unclear** — third-party mirror
// `chenmozhijin/BSRoformer-GGUF`, `docs/license-audit.md` §3.1 sign-off
// blank until owner ADR resolves which specific checkpoint + license the
// publish would target). Standalone file-based entry point with an SPDX
// override argument (mirror of the xcodec2 / vits_ja re-export pattern).
// The runtime `ModelKind::BsRoformer` dispatch arm above shares the same
// `models::bs_roformer::convert_bs_roformer_file` helper, so a caller who
// prefers `--model bs-roformer` via `convert_file_licensed` and a caller
// who calls `convert_bs_roformer_file` directly land the same bytes.
pub use models::bs_roformer::{BsRoformerReport, convert_bs_roformer_file};

/// Voxtral audio-adapter side-car (M3-10 Wave 8). Callers supply this through
/// [`convert_voxtral_file_with_adapter_config`] (a JSON path) or by
/// constructing an [`AdapterSpec`] directly and attaching it to a
/// [`VoxtralConfig::adapter`] field.
pub use models::voxtral::AdapterSpec;

/// Parses an upstream HuggingFace-style Voxtral `config.json` into a
/// [`VoxtralConfig`] (the `vokra-cli convert --model voxtral --config` path).
/// See `models::voxtral::parse_hf_config` for the accepted schema; a JSON
/// with no recognized Voxtral hparams is a hard error (FR-EX-08).
pub fn parse_voxtral_hf_config(bytes: &[u8]) -> Result<VoxtralConfig, ConvertError> {
    models::voxtral::parse_hf_config(bytes)
}

/// Resolves a (possibly sharded) Voxtral safetensors checkpoint to one path
/// per shard, WITHOUT reading any file bytes.
///
/// Same semantics as [`read_voxtral_checkpoint`] but returns paths only, so
/// callers who go through the streaming reader
/// (`models::voxtral::convert_shards_streaming`) can defer the file open
/// until after the header parse (and thus never mmap or read the whole
/// checkpoint at once). See [`convert_voxtral_file_streaming`] for the
/// user-facing path (M5 gap A-3, 2026-07-29).
fn resolve_voxtral_shard_paths(input: &Path) -> Result<Vec<std::path::PathBuf>, ConvertError> {
    use vokra_core::json::JsonValue;

    let file_name = input.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if !file_name.ends_with(".index.json") {
        return Ok(vec![input.to_path_buf()]);
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
    Ok(shard_names.into_iter().map(|f| dir.join(f)).collect())
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

/// Streaming counterpart of [`convert_voxtral_file`] (M5 gap A-3, 2026-07-29).
///
/// Header-only mmap per shard + one-tensor-at-a-time payload streaming, so a
/// Voxtral-Small-24B checkpoint (48 GB BF16, 11 shards) converts on a
/// 16 GB M1 iMac. The in-memory path
/// ([`convert_voxtral_file`] / [`convert_voxtral_file_quantized`]) is
/// unchanged and stays the default for callers who already fit — this
/// function does not change the existing bytes on disk, it just lifts the
/// mmap-everything memory cap.
///
/// The output GGUF is **byte-identical** to what [`convert_voxtral_file`]
/// would produce over the same checkpoint + config (pinned by a converter
/// test); the only observable difference is the process's peak RSS.
///
/// # Restrictions vs the in-memory path
///
/// - **No quantization**. K-quantizing needs `SafetensorsFile::tensor_f32`
///   to widen BF16 → f32 in memory before quantizing; a streaming
///   equivalent would need a chunked widen-then-quantize helper. Deferred
///   until a real need shows up (owner quantizes big models on vast.ai,
///   which fits the in-memory path).
/// - Adapter side-car support is a follow-up
///   ([`convert_voxtral_file_with_adapter_config`] does not have a
///   streaming variant yet — same in-memory `read_voxtral_checkpoint`
///   call).
///
/// # Errors
///
/// As [`convert_voxtral_file`], plus [`ConvertError::Io`] on shard open /
/// read failure at any point during the streaming pass.
pub fn convert_voxtral_file_streaming(
    input: &Path,
    config: &VoxtralConfig,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let paths = resolve_voxtral_shard_paths(input)?;
    let (tensor_count, metadata_count, output_bytes, report) =
        models::voxtral::convert_shards_streaming(&paths, output, Some(config))?;

    let notes = vec![format!(
        "voxtral (streaming): {} float weights written ({} BF16 passthrough — exact), \
         {} non-float skipped, name {}, tokenizer embedded: {}",
        report.written,
        report.bf16_passthrough,
        report.skipped_non_float,
        report.name,
        report.tokenizer_embedded
    )];

    Ok(ConvertSummary {
        model: ModelKind::Voxtral,
        tensor_count,
        metadata_count,
        output_bytes,
        notes,
    })
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
/// `models::voxtral::parse_adapter_config` for the JSON schema.
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

/// Convert a **StyleTTS 2** (yl4579) safetensors checkpoint into a
/// Vokra GGUF (config-only scaffold, 2026-07-30).
///
/// This is the named entry point that mirrors `convert_vits_ja_file`
/// (weight-restricted TTS) — functionally identical to
/// `convert_file(ModelKind::StyleTts2, input, output)` — kept for
/// `convert_*_file` naming symmetry with the other TTS models.
///
/// # ⚠️  Weight distribution — **fail-closed by default**
///
/// The upstream yl4579 release conditions weight use on **voice
/// consent + disclosure** (README §Pre-trained Models) — a usage
/// agreement, NOT a standard SPDX permissive license. The provenance
/// stamp defaults to
/// [`vokra_core::LicenseClass::Unknown`], which fails closed under
/// M2-13. The runtime `StyleTts2Tts::from_gguf`
/// (`crates/vokra-models/src/styletts2/`) is **also** deliberately
/// unwired (returns `NotImplemented` naming the licence blocker); a
/// future wave binds real weights through it when a permissive-license
/// StyleTTS 2 checkpoint arrives. A user who trained their own
/// StyleTTS 2 on a permissive corpus overrides at the outer `--license
/// <spdx>` boundary of `convert_file`, which rewrites the provenance
/// chunk to the correct SPDX id.
///
/// Architecture rides MIT code
/// (`github.com/yl4579/StyleTTS2/LICENSE`) and is *always*
/// independently implementable (whisper.cpp 型 self re-implementation,
/// CLAUDE.md 設計判断 4). See
/// `docs/tickets/sota-coverage-plan-2026-07-22.md` §2.4 for the
/// "support the architecture, refuse the weights" rationale.
pub fn convert_styletts2_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::StyleTts2, input, output)
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

/// Convert an NVIDIA **Canary-Qwen-2.5B** safetensors checkpoint into a
/// Vokra GGUF (SoTA plan reuse bundle, 2026-07-30).
///
/// Functionally identical to `convert_file(ModelKind::CanaryQwen, input,
/// output)` — Canary-Qwen has no side-car config or tokenizer to embed at
/// this scaffold stage (encoder hparams reuse the Canary-1B-v2 primary-
/// source defaults + decoder hparams carry canonical Qwen-family constants
/// with `0`-placeholder dims pending `.nemo` extraction) — but the named
/// entry keeps the `convert_*_file` naming symmetry with the other ASR /
/// TTS models.
///
/// # Architecture summary
///
/// - **Encoder**: FastConformer, **32 layers** (reused from Canary-1B-v2),
///   `d_model=1024`, `n_heads=8`, `ffn_dim=4096`, `num_mel_bins=128`,
///   `subsampling_factor=8`, `conv_kernel_size=9`,
///   `max_position_embeddings=5000`, `attention_bias=true`.
/// - **Decoder**: Qwen LLM (Voxtral-style soft-prompt prefix), GQA
///   `n_head_q=16`, `n_head_kv=8`, `head_dim=128`,
///   `rope_theta=1_000_000`, `rms_norm_eps=1e-6`, SwiGLU. Exact dims
///   (`n_layer`, `hidden_dim`, `ffn_dim`, `vocab_size`, `n_ctx`) ride as
///   `0`-placeholder sentinels — the runtime `CanaryQwenConfig::
///   validate_for_forward` rejects them loudly (FR-EX-08), so a real
///   `.nemo` extraction wave fills them.
/// - **Sample rate**: **16 kHz** (from the Canary FastConformer front-end).
///
/// # BF16 posture
///
/// The upstream Canary-Qwen-2.5B `.nemo` tarball's PyTorch checkpoint is
/// typically **BF16**. BF16 tensors pass through **verbatim** as GGUF
/// type 30 — the runtime widens BF16 → f32 losslessly at load. A
/// downstream that pre-widens to F16 or F32 offline also lands on the
/// pass-through arm. Provenance is stamped **CC-BY 4.0**
/// (`AttributionRequired` via the `canary-` family prefix walk) and the
/// FR-MD-09 attribution surface activates.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_canary_qwen_file(
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::CanaryQwen, input, output)
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
/// [`vokra_core::gguf::GgufStreamWriter`] — peak footprint stays at roughly one tensor plus
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
            CanaryQwen,
            OmniasrCtc,
            DistilWhisper,
            KotobaWhisper,
            Crisperwhisper,
            Chatterbox,
            ChatterboxTurbo,
            ChatterboxNano,
            Qwen3Tts,
            VoxCpm2,
            VibeVoice,
            VibeVoiceRealtime,
            Irodori,
            VitsJa,
            // SBV2 v2 plan Task 11 (2026-07-26) + Task 8 (2026-07-27):
            // pin every DeBERTa/SBV2 variant's `as_arg → from_arg`
            // round-trip so a dropped alias in either direction fails
            // loudly (mirror of the guard the rest of this list provides
            // for the Phase 2-5 additions above).
            DebertaV2,
            DebertaV3,
            SbV2,
            // StyleTTS 2 (2026-07-30): config-only scaffold with
            // fail-closed provenance (voice-consent gated weight).
            StyleTts2,
            // F0 pitch-extractor tier (2026-07-30): RMVPE — the first
            // `category = "f0"` binder in the converter tree.
            Rmvpe,
            // M5-16 / FR-OP-83: FCPE — pin the alias round-trip so a dropped
            // spelling in `as_arg` fails loudly (same rationale as the
            // Phase 2-5 additions above).
            Fcpe,
            // SoTA plan Phase 5 VAD-2 (2026-07-30): FunASR FSMN-VAD —
            // first-class audio-dialect op posture (distinct from Silero
            // VAD v5's FR-LD-06 1:1 subgraph). Every hparam axis is
            // stamped verbatim from the released FunASR checkpoint.
            FsmnVad,
            // Coverage-audit 2026-08-03 Wave A: RNNoise v0.2 — the
            // canonical `--model rnnoise-v0.2` spelling must round-trip
            // through `as_arg → from_arg` so a dropped alias fails
            // loudly here.
            Rnnoise,
            // Coverage-audit 2026-08-03 Wave A: NSNet2 — the canonical
            // `--model nsnet2` spelling must round-trip through
            // `as_arg → from_arg` so a dropped alias fails loudly here.
            Nsnet2,
            // Coverage-audit 2026-08-03 Wave A: DNSMOS bundle — the
            // canonical `--model dnsmos-p808-p835` spelling must
            // round-trip through `as_arg → from_arg` so a dropped alias
            // fails loudly here.
            Dnsmos,
            // Coverage-audit 2026-08-03 Wave A: FRCRN speech-enhancement
            // — the canonical `--model frcrn` spelling must round-trip
            // through `as_arg → from_arg` so a dropped alias fails
            // loudly here.
            Frcrn,
            // Coverage-audit 2026-08-03 Wave B fast-track (13 variants).
            Hibiki,
            SberGigaamV3,
            SberGigaamMultilingual,
            ReazonspeechNemoV2,
            MagpiettsV2602,
            ParakeetUnified,
            Canary1bFlash,
            OwsmV4Medium1b,
            ParakeetTdt11b,
            FireredAsrAedL,
            SortformerDiar4spkV1,
            SenseVoiceSmall,
            WhisperMedusaV1,
            NemotronSpeechStreamingV2603,
            // coverage-audit-2026-08-03 Wave D T4 (non-commercial batch, 2026-08-04).
            FacebookDenoiser,
            NisqaV2Weight,
            ChatTts,
            StableAudioOpenSmall,
            Jasco400mChordsDrums,
            // coverage-audit-2026-08-03 Wave A permissive continuation (2026-08-04).
            Utmosv2,
            TorchaudioSquim,
            HtdemucsMulti,
            OpenwakewordOp,
            Mossformer2Ss16k,
            TenVad,
            AudiosealRealWeight,
            // Music-understanding wave (2026-08-13): YAMNet — canonical
            // `--model yamnet` must round-trip through as_arg → from_arg
            // so a dropped alias fails loudly here.
            Yamnet,
            // Music-understanding wave (2026-08-13): MERT — canonical
            // `--model mert` must round-trip through as_arg → from_arg
            // so a dropped alias fails loudly here.
            Mert,
            // Music-understanding wave (2026-08-13): MuQ — canonical
            // `--model muq` must round-trip through as_arg → from_arg
            // so a dropped alias fails loudly here.
            Muq,
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
            // SoTA plan reuse bundle (2026-07-30) — canary-qwen aliases
            // must dispatch to CanaryQwen, not the base Canary variant
            // (silent mis-dispatch would run the Transformer-AED chunk
            // group writer instead of the Qwen-LLM chunk group writer).
            (
                ModelKind::CanaryQwen,
                &[
                    "canary-qwen",
                    "canary_qwen",
                    "canary-qwen-2.5b",
                    "canary-qwen-2_5b",
                    "canary-qwen-2.5B",
                ],
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
            // residual wave 4 (2026-08-02): CrisperWhisper —
            // Whisper-large-v3 verbatim-word-timestamps fine-tune,
            // cc-by-nc-4.0 (T4 Research-only publish path).
            (
                ModelKind::Crisperwhisper,
                &[
                    "crisperwhisper",
                    "crisper-whisper",
                    "crisper_whisper",
                    "nyrahealth/crisperwhisper",
                    "nyrahealth/CrisperWhisper",
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
            // Phase 3 — Qwen3-TTS (0.6B family; 1.7B siblings live in the
            // Qwen3TtsCustomVoice17B / Qwen3TtsVoiceDesign17B arms in the
            // subsequent aliases blocks). The 0.6B-CustomVoice slugs
            // (2026-08-01 Wave 4 slug-only add) also route here — the
            // fine-tune shares the 0.6B-Base topology per the parent
            // decision recorded in `from_arg` above.
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
                    // 2026-08-01 Wave 4 slug-only add: 0.6B-CustomVoice
                    // fine-tune (identical axes to 0.6B-Base — pin every
                    // spelling so a dropped alias in `from_arg` fails loudly
                    // rather than misrouting to `Unknown`).
                    "qwen3-tts-0.6b-customvoice",
                    "qwen3-tts-0_6b-customvoice",
                    "qwen3-tts-0.6b-custom-voice",
                    "qwen3-tts-12hz-0.6b-customvoice",
                    "qwen3-tts-12hz-0_6b-customvoice",
                    "qwen3-tts-12hz-0.6b-custom-voice",
                    "qwen/qwen3-tts-12hz-0.6b-customvoice",
                ],
            ),
            // Phase 4 — VoxCPM (0.5B + 2B — 2026-07-30 Option C hybrid)
            (
                ModelKind::VoxCpm2,
                &[
                    "voxcpm",
                    "voxcpm2",
                    "voxcpm-0.5b",
                    "voxcpm-0_5b",
                    "voxcpm-0.5b-base",
                    "voxcpm-0_5b-base",
                    "voxcpm2-0.5b",
                    "voxcpm2-0_5b",
                    "voxcpm2-2b",
                    "voxcpm2-2_0b",
                    "voxcpm2-2b-base",
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
            // 2026-08-01 add — VibeVoice-Realtime (streaming sibling)
            (
                ModelKind::VibeVoiceRealtime,
                &[
                    "vibevoice-realtime",
                    "vibevoice_realtime",
                    "vibevoice-realtime-0.5b",
                    "vibevoice-realtime-0_5b",
                    "vibevoice-streaming",
                    "vibevoice_streaming",
                    "microsoft/vibevoice-realtime-0.5b",
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
            // StyleTTS 2 (2026-07-30) — config-only scaffold. Weight
            // license is voice-consent gated (LicenseClass::Unknown =
            // fail-closed under M2-13); the converter still recognises
            // every alias so a developer who legitimately holds the
            // weight under a distinct SPDX id can convert via
            // `--license <spdx>`.
            (
                ModelKind::StyleTts2,
                &[
                    "styletts2",
                    "styletts-2",
                    "styletts_2",
                    "yl4579/styletts2",
                    "yl4579/StyleTTS2",
                ],
            ),
            // Whisper — the historical alias.
            (ModelKind::Whisper, &["whisper", "whisper-base"]),
            // SBV2 v2 plan Task 11 (2026-07-26) — DeBERTa v2 (JA) uses
            // the ku-nlp Japanese-character upstream; v3 (EN, Task 8
            // 2026-07-27 correction) uses the microsoft English
            // upstream. Note the two variants dispatch to distinct
            // orgs (`ku-nlp` vs `microsoft`); the nonexistent
            // `ku-nlp/deberta-v3-large-japanese-char-wwm` string is
            // pinned as a negative case in
            // `unknown_model_arg_returns_none`.
            (
                ModelKind::DebertaV2,
                &[
                    "deberta-v2",
                    "deberta_v2",
                    "ku-nlp/deberta-v2-large-japanese-char-wwm",
                ],
            ),
            (
                ModelKind::DebertaV3,
                &["deberta-v3", "deberta_v3", "microsoft/deberta-v3-large"],
            ),
            // SBV2 v2 plan Task 25 (2026-07-26) — Style-Bert-VITS2 v2.
            (
                ModelKind::SbV2,
                &[
                    "sbv2",
                    "sbv2-v2",
                    "sbv2-v2-multilingual-base",
                    "style-bert-vits2",
                    "style_bert_vits2",
                    "style-bert-vits2-v2",
                ],
            ),
            // SoTA plan Phase 5 VAD-2 (2026-07-30) — FSMN-VAD. Pins the
            // union alias set (HEAD's funasr/fsmn-vad + funaudiollm/
            // fsmn-vad-gguf + fsmn-vad-gguf plus a5763ce's fsmn-vad-
            // zh-cn-16k-common + fsmnvad + iic/speech_fsmn_vad_zh-cn-
            // 16k-common-pytorch).
            (
                ModelKind::FsmnVad,
                &[
                    "fsmn-vad",
                    "fsmn_vad",
                    "fsmnvad",
                    "fsmn-vad-zh-cn-16k-common",
                    "funasr/fsmn-vad",
                    "funaudiollm/fsmn-vad-gguf",
                    "fsmn-vad-gguf",
                    "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                ],
            ),
            // Coverage-audit 2026-08-03 Wave A — Xiph RNNoise v0.2.
            // Every alias must dispatch to the same variant so a caller
            // using the GitHub-flavour `xiph/rnnoise` id or the
            // versioned publish slug lands on the same converter.
            (
                ModelKind::Rnnoise,
                &["rnnoise", "rnnoise-v0.2", "rnnoise-v0_2", "xiph/rnnoise"],
            ),
            // Coverage-audit 2026-08-03 Wave A — Microsoft NSNet2 NR
            // baseline. Every alias resolves to the same `--allow-noncommercial`-
            // free MIT skeleton path.
            (
                ModelKind::Nsnet2,
                &[
                    "nsnet2",
                    "nsnet2-baseline",
                    "nsnet2-20ms",
                    "nsnet2-20ms-baseline",
                    "microsoft/nsnet2",
                    "microsoft/nsnet2-baseline",
                ],
            ),
            // coverage-audit wave-a (2026-08-03) — FRCRN speech
            // enhancement. Every alias resolves to the same Apache-2.0
            // Complex U-Net + freq-recurrent LSTM checkpoint.
            (
                ModelKind::Frcrn,
                &[
                    "frcrn",
                    "alibabasglab/frcrn",
                    "clearervoice-studio/frcrn",
                    "modelscope/clearervoice-studio-frcrn",
                ],
            ),
            // ---- coverage-audit 2026-08-03 Wave B fast-track (13 model) ----
            (
                ModelKind::Hibiki,
                &[
                    "hibiki",
                    "hibiki-2b",
                    "hibiki_2b",
                    "kyutai/hibiki-2b-pytorch-bf16",
                ],
            ),
            (
                ModelKind::SberGigaamV3,
                &[
                    "sber-gigaam-v3",
                    "sber_gigaam_v3",
                    "gigaam-v3",
                    "salute-developers/gigaam-v3",
                ],
            ),
            (
                ModelKind::SberGigaamMultilingual,
                &[
                    "sber-gigaam-multilingual",
                    "sber_gigaam_multilingual",
                    "gigaam-multilingual",
                    "salute-developers/gigaam-multilingual",
                ],
            ),
            (
                ModelKind::ReazonspeechNemoV2,
                &[
                    "reazonspeech-nemo-v2",
                    "reazonspeech_nemo_v2",
                    "reazon-research/reazonspeech-nemo-v2",
                ],
            ),
            (
                ModelKind::MagpiettsV2602,
                &[
                    "magpietts-v2602",
                    "magpietts_v2602",
                    "nvidia/magpietts-v2602",
                ],
            ),
            (
                ModelKind::ParakeetUnified,
                &[
                    "parakeet-unified-en-0.6b",
                    "parakeet_unified_en_0_6b",
                    "parakeet-unified",
                    "nvidia/parakeet-unified-en-0.6b",
                ],
            ),
            (
                ModelKind::Canary1bFlash,
                &[
                    "canary-1b-flash",
                    "canary_1b_flash",
                    "nvidia/canary-1b-flash",
                ],
            ),
            (
                ModelKind::OwsmV4Medium1b,
                &[
                    "owsm-v4-medium-1b",
                    "owsm_v4_medium_1b",
                    "owsm-v4-medium",
                    "espnet/owsm_v4_medium_1b",
                ],
            ),
            (
                ModelKind::ParakeetTdt11b,
                &[
                    "parakeet-tdt-1.1b",
                    "parakeet_tdt_1_1b",
                    "nvidia/parakeet-tdt-1.1b",
                ],
            ),
            (
                ModelKind::FireredAsrAedL,
                &[
                    "firered-asr-aed-l",
                    "firered_asr_aed_l",
                    "fireredteam/fireredasr-aed-l",
                ],
            ),
            (
                ModelKind::SortformerDiar4spkV1,
                &[
                    "sortformer-diar-4spk-v1",
                    "sortformer_diar_4spk_v1",
                    "nvidia/diar_sortformer_4spk-v1",
                ],
            ),
            (
                ModelKind::SenseVoiceSmall,
                &[
                    "sensevoicesmall",
                    "sensevoice-small",
                    "funaudiollm/sensevoicesmall",
                ],
            ),
            (
                ModelKind::WhisperMedusaV1,
                &[
                    "whisper-medusa-v1",
                    "whisper_medusa_v1",
                    "aiola/whisper-medusa-v1",
                ],
            ),
            (
                ModelKind::NemotronSpeechStreamingV2603,
                &[
                    "nemotron-speech-streaming-v2603",
                    "nemotron_speech_streaming_v2603",
                    "nvidia/nemotron-speech-streaming-v2603",
                ],
            ),
            // coverage-audit-2026-08-03 Wave D T4 (non-commercial batch, 2026-08-04).
            (
                ModelKind::FacebookDenoiser,
                &[
                    "facebook-denoiser",
                    "facebook_denoiser",
                    "denoiser",
                    "demucs-denoiser",
                    "facebookresearch/denoiser",
                ],
            ),
            (
                ModelKind::NisqaV2Weight,
                &[
                    "nisqa-v2",
                    "nisqa-v2-weight",
                    "nisqa_v2_weight",
                    "nisqa",
                    "gabrielmittag/nisqa",
                ],
            ),
            (
                ModelKind::ChatTts,
                &["chattts", "chat-tts", "2noise/chattts"],
            ),
            (
                ModelKind::StableAudioOpenSmall,
                &[
                    "stable-audio-open-small",
                    "stable_audio_open_small",
                    "sao-small",
                    "stabilityai/stable-audio-open-small",
                ],
            ),
            (
                ModelKind::Jasco400mChordsDrums,
                &[
                    "jasco-400m-chords-drums",
                    "jasco_400m_chords_drums",
                    "jasco-400m",
                    "facebook/jasco-400m-chords-drums",
                ],
            ),
            // coverage-audit-2026-08-03 Wave A permissive continuation (2026-08-04).
            (
                ModelKind::Utmosv2,
                &[
                    "utmosv2",
                    "utmos-v2",
                    "utmos_v2",
                    "sarulab-speech/UTMOSv2",
                    "sarulab-speech/utmosv2",
                ],
            ),
            (
                ModelKind::TorchaudioSquim,
                &[
                    "torchaudio-squim",
                    "torchaudio_squim",
                    "squim",
                    "pytorch/audio",
                ],
            ),
            (
                ModelKind::HtdemucsMulti,
                &[
                    "htdemucs-4s-6s",
                    "htdemucs-4s",
                    "htdemucs-6s",
                    "htdemucs-ft",
                    "htdemucs_4s_6s",
                    "htdemucs-multi",
                    "htdemucs_multi",
                    "facebook/htdemucs_ft",
                    "facebook/htdemucs_6s",
                ],
            ),
            (
                ModelKind::OpenwakewordOp,
                &[
                    "openwakeword-op",
                    "openwakeword_op",
                    "owwakeword",
                    "dscripka/openWakeWord-op",
                ],
            ),
            (
                ModelKind::Mossformer2Ss16k,
                &[
                    "mossformer2-ss-16k",
                    "mossformer2",
                    "mossformer2_ss_16k",
                    "alibabasglab/MossFormer2_SS_16K",
                ],
            ),
            (
                ModelKind::TenVad,
                &[
                    "ten-vad",
                    "ten_vad",
                    "TEN-framework/ten-vad",
                    "ten-framework/ten-vad",
                ],
            ),
            (
                ModelKind::AudiosealRealWeight,
                &[
                    "audioseal-real-weight",
                    "audioseal_real_weight",
                    "audioseal",
                    "audio-seal",
                    "facebook/audioseal",
                ],
            ),
            // hf-audio-gap-comprehensive-2026-07-30 §3.8 (2026-08-04).
            (
                ModelKind::MioCodec,
                &[
                    "miocodec",
                    "mio-codec",
                    "mio_codec",
                    "miocodec-25hz-44khz-v2",
                    "miocodec_25hz_44khz_v2",
                    "miocodec-25hz-44-1khz-v2",
                    "aratako/miocodec-25hz-44.1khz-v2",
                    "aratako/miocodec-25hz-44_1khz-v2",
                ],
            ),
            // SoTA plan candidate wave (2026-08-04): Neuphonic NeuTTS Air.
            (
                ModelKind::NeuTtsAir,
                &[
                    "neutts-air",
                    "neutts_air",
                    "neu-tts-air",
                    "neu_tts_air",
                    "neuphonic/neutts-air",
                ],
            ),
            // Music-understanding wave (2026-08-13): YAMNet.
            (
                ModelKind::Yamnet,
                &[
                    "yamnet",
                    "google-yamnet",
                    "google/yamnet",
                    "thelou1s/yamnet",
                ],
            ),
            // Music-understanding wave (2026-08-13): MERT.
            (
                ModelKind::Mert,
                &[
                    "mert",
                    "mert-v1-330m",
                    "mert-v1",
                    "mert-330m",
                    "m-a-p/mert-v1-330m",
                    "m-a-p/MERT-v1-330M",
                ],
            ),
            // Music-understanding wave (2026-08-13): MuQ.
            (
                ModelKind::Muq,
                &[
                    "muq",
                    "muq-large-msd-iter",
                    "muq-large",
                    "openmuq/muq-large-msd-iter",
                    "OpenMuQ/MuQ-large-msd-iter",
                ],
            ),
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
            // SBV2 v2 plan Task 8 (2026-07-27) regression pin: the
            // nonexistent copy-paste alias that used to silently resolve
            // to Some(DebertaV3). MUST return None so a future re-add of
            // the same drift is caught by this test rather than by an
            // owner discovering a wrong-shape GGUF at conversion time.
            "ku-nlp/deberta-v3-large-japanese-char-wwm",
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

    /// DNSMOS bundle CLI aliases (coverage-audit Wave A, 2026-08-03).
    /// Pins the four accepted spellings — arch tag, canonical SKU, its
    /// underscore variant, and the `microsoft/dnsmos` slug — plus the
    /// `as_arg()` canonical form (`dnsmos-p808-p835`) so a future
    /// rename would fail loudly here rather than silently break the
    /// orchestrator's dispatch.
    #[test]
    fn dnsmos_aliases_resolve_and_canonical_arg_is_stable() {
        for s in [
            "dnsmos",
            "dnsmos-p808-p835",
            "dnsmos_p808_p835",
            "microsoft/dnsmos",
        ] {
            assert_eq!(
                ModelKind::from_arg(s),
                Some(ModelKind::Dnsmos),
                "alias {s:?} must resolve to ModelKind::Dnsmos"
            );
        }
        assert_eq!(ModelKind::Dnsmos.as_arg(), "dnsmos-p808-p835");
    }

    /// **Task B-1 owner-blocked policy pin** (2026-07-29): the four
    /// voice-conversion (VC) converter modules
    /// (`crates/vokra-convert/src/models/openvoice_v2.rs` +
    /// `knn_vc.rs` + `freevc.rs` + `meanvc.rs`) are landed as
    /// `pub fn convert_*_file` skeletons but MUST NOT resolve through
    /// `ModelKind::from_arg` today — the CLI `--model` gate is the
    /// user-facing entry point and staying fail-closed here honours the
    /// existing policy chain.
    ///
    /// **Policy source 1 — CLAUDE.md 設計判断 8**: voice-cloning is
    /// intentionally excluded from the `ayutaz/vokra` public repo to
    /// avoid tool-distributor liability under Tennessee **ELVIS Act §3**
    /// (2024-07-01) + federal **NO FAKES Act**. The `vc` category is
    /// routed to a **separate** `vokra-voiceclone-experimental`
    /// repository per docs/license-audit.md §5.
    ///
    /// **Policy source 2 — docs/license-audit.md §3.1 rows 294-297**:
    /// the sign-off rows for `openvoice_v2` / `knn_vc` / `freevc` /
    /// `meanvc` are **blank per fail-closed default** (the `本欄の署名・
    /// 判定は owner 記入、CC は pre-fill しない` directive); no CC-side
    /// pre-fill is permitted, and `scripts/publish/publish-one.sh`
    /// refuses to distribute a GGUF whose §3.1 row is blank.
    ///
    /// **Policy source 3 — docs/m5-owner-verification-checklist.md §6.2
    /// (rows 146-149) + §6.4 "Voice-clone territory … ELVIS Act policy
    /// defer"**: the M5-05 T15 owner action is to choose destination
    /// among three candidates. Destination one:
    /// `staging/vokra-voiceclone-experimental` separate repo (existing
    /// scaffold per commit `6dc9f86`). Destination two: `ayutaz/vokra`
    /// core WITH the mandatory `--i-understand-risks --research-only`
    /// consent-machinery + M2-13 research-flag gate + a formal ADR
    /// overriding 設計判断 8. Destination three: explicit Rejected in
    /// §3.1.
    ///
    /// The module docstrings on `openvoice_v2.rs` (lines 60-75) and
    /// `freevc.rs` (lines 66-80) already flag their unwired state as
    /// **intentional** and gate the follow-up wave on the
    /// voiceclone-experimental hook. This test codifies that gate at
    /// the CLI boundary so a future change that wires any of the four
    /// slugs into `ModelKind::from_arg` MUST also delete or update this
    /// pin — surfacing the policy trade-off in code review rather than
    /// letting it land silently. A wiring PR that does not touch this
    /// test will fail here; a wiring PR that does touch this test will
    /// force the reviewer to acknowledge the ELVIS Act / NO FAKES Act
    /// consequences of the wiring.
    ///
    /// The alias set covers every spelling a caller might plausibly
    /// try — the upstream HF slug (`myshell-ai/OpenVoiceV2` etc.), the
    /// module-name `openvoice_v2` (underscore = arch tag = the
    /// `vokra.model.arch` value the runtime dispatch would look for),
    /// and the hyphen spelling `openvoice-v2` (the `vokra.model.name`
    /// value = the conventional CLI form used by sibling wired
    /// converters). If a future release adds a `v2.1` / `v3` variant,
    /// its spelling must also stay blocked here until the same owner
    /// ADR decision is recorded — a follow-up PR that adds the spelling
    /// belongs in the same wiring wave that lifts the policy defer.
    ///
    /// Not fixed by this task (preexisting inconsistency, owner-blocked
    /// scope):
    /// `crates/vokra-convert/src/lib.rs:2804`
    /// (`pub use models::knn_vc::{KnnVcReport, convert_knn_vc_file};`)
    /// already exposes the `knn_vc` converter fn on the Rust crate
    /// surface. That is a **preexisting** re-export from the BF16
    /// pass-through campaign (2026-07-25) and touching it now would
    /// itself be an owner-ADR-territory change (either it stays
    /// removed = a compat break for any external Rust caller reaching
    /// through the API, or it stays present = a partial-wire that
    /// still needs the M5-05 destination decision). This test pins the
    /// **CLI-facing** dispatch boundary only, which is the load-bearing
    /// user-visible gate; the Rust-surface question is deferred to the
    /// same owner ADR that resolves the four rows in §3.1.
    #[test]
    fn voice_clone_vc_slugs_are_owner_blocked_from_modelkind_dispatch() {
        // Every spelling below MUST return `None` today. The comment
        // beside each entry names the artefact that spelling corresponds
        // to inside the module (arch tag / model name / upstream slug)
        // so a future contributor lifting the policy defer can walk this
        // list back to the module they need to touch.
        for slug in [
            // openvoice_v2 — `pub const ARCH = "openvoice_v2"` /
            // `NAME = "openvoice-v2"` / upstream `myshell-ai/OpenVoiceV2`.
            "openvoice_v2",
            "openvoice-v2",
            "openvoicev2",
            "openvoice",
            "myshell-ai/OpenVoiceV2",
            // knn_vc — `pub(crate) const ARCH = "knn_vc"` /
            // `NAME = "knn-vc"` / upstream `bshall/knn-vc`. The `pub use`
            // at crates/vokra-convert/src/lib.rs:2804 exposes the
            // converter fn on the Rust surface (preexisting), but the
            // CLI-facing ModelKind dispatch stays fail-closed until the
            // owner ADR resolves the destination.
            "knn_vc",
            "knn-vc",
            "knnvc",
            "bshall/knn-vc",
            // freevc — `pub const ARCH = "freevc"` /
            // `NAME = "freevc"` / upstream `OlaWod/FreeVC`.
            "freevc",
            "free-vc",
            "free_vc",
            "OlaWod/FreeVC",
            // meanvc — module-scope `ARCH = "meanvc"` /
            // `NAME = "meanvc"` / upstream `ASLP-lab/MeanVC`.
            "meanvc",
            "mean-vc",
            "mean_vc",
            "ASLP-lab/MeanVC",
        ] {
            assert!(
                ModelKind::from_arg(slug).is_none(),
                "{slug:?} MUST NOT resolve to any ModelKind today — the four \
                 vc-category converters (openvoice_v2 / knn_vc / freevc / meanvc) \
                 are owner-blocked pending the M5-05 T15 destination decision \
                 (staging/vokra-voiceclone-experimental separate repo vs \
                 ayutaz/vokra core + consent machinery vs explicit Rejected in \
                 docs/license-audit.md §3.1). Wiring this spelling requires \
                 (a) an ADR overriding CLAUDE.md 設計判断 8 (ELVIS Act §3 + \
                 NO FAKES Act) AND (b) filling the owner sign-off in \
                 docs/license-audit.md §3.1 rows 294-297 — see the rustdoc \
                 above this test."
            );
        }
    }
}

/// 2026-08-02 wave: MOSS-Audio-4B-Instruct converter arm registration
/// (parent workflow REUSE HINT — dispatches through the sibling
/// `MossTts` converter via the new `MossTtsVariant::AudioInstruct4b`
/// arm). This module pins:
///
///   (a) the alias walk (`--model moss-audio-4b-instruct` +
///       underscore + `openmoss-team/…` fully-qualified spellings all
///       land on the same `ModelKind::MossAudio4bInstruct` variant),
///   (b) the `as_arg → from_arg` round-trip (a silently-dropped alias
///       in either direction fails loudly), and
///   (c) the end-to-end converter smoke: a synthetic BF16 safetensors
///       (a stand-in for the 3-shard ~8 GB upstream, which the
///       parent workflow forbids downloading) survives the dispatch,
///       lands on the pass-through arm, and emits the faithful
///       provenance triple (arch + name + upstream_hf + license =
///       apache-2.0 Permissive + category = `s2s` + distinct
///       `vokra.moss_tts.variant = "audio_4b"` sub-arch tag).
#[cfg(test)]
mod moss_audio_4b_instruct_arm_tests {
    use super::{ModelKind, models};
    use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};

    /// Every CLI alias spelling registered in `from_arg` for
    /// MOSS-Audio-4B-Instruct must dispatch to the same
    /// `ModelKind::MossAudio4bInstruct` variant.
    #[test]
    fn every_alias_dispatches_to_moss_audio_4b_instruct() {
        for slug in [
            "moss-audio-4b-instruct",
            "moss_audio_4b_instruct",
            "moss-audio-4b",
            "moss_audio_4b",
            "openmoss-team/moss-audio-4b-instruct",
            "openmoss-team/moss-audio-4b",
        ] {
            let parsed = ModelKind::from_arg(slug).unwrap_or_else(|| {
                panic!(
                    "{slug:?} must resolve to a ModelKind — dropping this alias would break \
                     the MOSS-Audio-4B-Instruct publish path"
                )
            });
            assert_eq!(
                parsed,
                ModelKind::MossAudio4bInstruct,
                "{slug:?} must dispatch to MossAudio4bInstruct, got {parsed:?}"
            );
        }
    }

    /// The canonical `as_arg` spelling round-trips through `from_arg`
    /// back to the same variant (guards against a silently-dropped
    /// arm in either direction).
    #[test]
    fn as_arg_round_trip_lands_on_the_same_variant() {
        let arg = ModelKind::MossAudio4bInstruct.as_arg();
        assert_eq!(
            arg, "moss-audio-4b-instruct",
            "canonical as_arg spelling must match the CLI default"
        );
        let parsed = ModelKind::from_arg(arg)
            .expect("as_arg → from_arg round-trip must land back on a variant");
        assert_eq!(parsed, ModelKind::MossAudio4bInstruct);
    }

    /// Smoke round-trip through the underlying MossTts converter's new
    /// `AudioInstruct4b` variant: a synthetic BF16 safetensors buffer
    /// (a stand-in for the ~8 GB upstream — parent workflow forbids
    /// downloading > 2 GB) survives the pass-through arm and the
    /// emitted GGUF carries the faithful provenance triple + `s2s`
    /// category + `audio_4b` sub-arch tag.
    #[test]
    fn smoke_dispatch_emits_faithful_provenance_and_audio_4b_sub_arch() {
        use models::moss_tts::{MossTtsVariant, convert_variant};
        // Minimal BF16 safetensors: single 2×3 tensor `embed.weight`.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"embed.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);

        let (builder, report) = convert_variant(buf, MossTtsVariant::AudioInstruct4b)
            .expect("MossTts AudioInstruct4b BF16 pass-through must succeed");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 must land on the pass-through counter"
        );

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse emitted GGUF");

        // Provenance triple: arch (moss_tts) + name (moss-audio-4b-instruct) +
        // upstream_hf (OpenMOSS-Team/MOSS-Audio-4B-Instruct) + license
        // (apache-2.0 Permissive) + category (s2s).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("moss_tts"),
            "arch must be moss_tts (shared with the sibling tts variants — REUSE HINT)"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("moss-audio-4b-instruct"),
            "name must reflect the audio-LLM sibling identity"
        );
        assert_eq!(
            file.get("vokra.model.category").and_then(|v| v.as_str()),
            Some("s2s"),
            "category must be s2s (audio-LLM), matching kimi_audio / baichuan_audio / \
             step_audio2_mini — NOT the sibling tts variants"
        );
        assert_eq!(
            file.get("vokra.provenance.upstream_hf")
                .and_then(|v| v.as_str()),
            Some("OpenMOSS-Team/MOSS-Audio-4B-Instruct"),
            "upstream_hf must preserve the HF slug verbatim (traceability)"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get("vokra.moss_tts.variant").and_then(|v| v.as_str()),
            Some("audio_4b"),
            "distinct sub-arch tag so a runtime dispatcher can refuse to bind placeholder axes"
        );
        // Backbone family = qwen3 (best-guess placeholder, documented in
        // the variant doc comment).
        assert_eq!(
            file.get("vokra.moss_tts.llm.family")
                .and_then(|v| v.as_str()),
            Some("qwen3")
        );
        // BF16 payload survives byte-for-byte.
        let info = file
            .tensor_info("embed.weight")
            .expect("emitted GGUF must carry the tensor");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
        // llm.hidden_dim must be a nonzero u32 (Local placeholder = 2560).
        // Guards against a silent regression where the selector routes
        // AudioInstruct4b to a zero-sentinel branch.
        match file.get("vokra.moss_tts.llm.hidden_dim") {
            Some(GgufMetadataValue::U32(v)) => assert!(
                *v > 0,
                "placeholder hidden_dim must be a positive u32 (Local family = 2560), got {v}"
            ),
            other => panic!("vokra.moss_tts.llm.hidden_dim must be a U32, got {other:?}"),
        }
    }
}

/// 2026-08-02 wave: MOSS-Audio-8B-Instruct converter arm registration
/// (parent workflow REUSE HINT — dispatches through the sibling
/// `MossTts` converter via the new `MossTtsVariant::AudioInstruct8b`
/// arm). Mirrors [`moss_audio_4b_instruct_arm_tests`] byte-for-byte on
/// the larger 8B sibling (4 shards ~9.05 GB BF16 per parent workflow
/// manifest 2026-08-02).
#[cfg(test)]
mod moss_audio_8b_instruct_arm_tests {
    use super::{ModelKind, models};
    use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};

    /// Every CLI alias spelling registered in `from_arg` for
    /// MOSS-Audio-8B-Instruct must dispatch to the same
    /// `ModelKind::MossAudio8bInstruct` variant.
    #[test]
    fn every_alias_dispatches_to_moss_audio_8b_instruct() {
        for slug in [
            "moss-audio-8b-instruct",
            "moss_audio_8b_instruct",
            "moss-audio-8b",
            "moss_audio_8b",
            "openmoss-team/moss-audio-8b-instruct",
            "openmoss-team/moss-audio-8b",
        ] {
            let parsed = ModelKind::from_arg(slug).unwrap_or_else(|| {
                panic!(
                    "{slug:?} must resolve to a ModelKind — dropping this alias would break \
                     the MOSS-Audio-8B-Instruct publish path"
                )
            });
            assert_eq!(
                parsed,
                ModelKind::MossAudio8bInstruct,
                "{slug:?} must dispatch to MossAudio8bInstruct, got {parsed:?}"
            );
        }
    }

    /// The canonical `as_arg` spelling round-trips through `from_arg`
    /// back to the same variant (guards against a silently-dropped
    /// arm in either direction).
    #[test]
    fn as_arg_round_trip_lands_on_the_same_variant() {
        let arg = ModelKind::MossAudio8bInstruct.as_arg();
        assert_eq!(
            arg, "moss-audio-8b-instruct",
            "canonical as_arg spelling must match the CLI default"
        );
        let parsed = ModelKind::from_arg(arg)
            .expect("as_arg → from_arg round-trip must land back on a variant");
        assert_eq!(parsed, ModelKind::MossAudio8bInstruct);
    }

    /// Smoke round-trip through the underlying MossTts converter's new
    /// `AudioInstruct8b` variant: a synthetic BF16 safetensors buffer
    /// (a stand-in for the ~9.05 GB upstream — parent workflow forbids
    /// downloading > 2 GB) survives the pass-through arm and the
    /// emitted GGUF carries the faithful provenance triple + `s2s`
    /// category + `audio_8b` sub-arch tag.
    #[test]
    fn smoke_dispatch_emits_faithful_provenance_and_audio_8b_sub_arch() {
        use models::moss_tts::{MossTtsVariant, convert_variant};
        // Minimal BF16 safetensors: single 2×3 tensor `embed.weight`.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let header = format!(
            r#"{{"embed.weight":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);

        let (builder, report) = convert_variant(buf, MossTtsVariant::AudioInstruct8b)
            .expect("MossTts AudioInstruct8b BF16 pass-through must succeed");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 must land on the pass-through counter"
        );

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse emitted GGUF");

        // Provenance triple: arch (moss_tts) + name (moss-audio-8b-instruct) +
        // upstream_hf (OpenMOSS-Team/MOSS-Audio-8B-Instruct) + license
        // (apache-2.0 Permissive) + category (s2s).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("moss_tts"),
            "arch must be moss_tts (shared with the sibling tts variants — REUSE HINT)"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("moss-audio-8b-instruct"),
            "name must reflect the audio-LLM sibling identity"
        );
        assert_eq!(
            file.get("vokra.model.category").and_then(|v| v.as_str()),
            Some("s2s"),
            "category must be s2s (audio-LLM), matching kimi_audio / baichuan_audio / \
             step_audio2_mini — NOT the sibling tts variants"
        );
        assert_eq!(
            file.get("vokra.provenance.upstream_hf")
                .and_then(|v| v.as_str()),
            Some("OpenMOSS-Team/MOSS-Audio-8B-Instruct"),
            "upstream_hf must preserve the HF slug verbatim (traceability)"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
        );
        assert_eq!(
            file.get("vokra.moss_tts.variant").and_then(|v| v.as_str()),
            Some("audio_8b"),
            "distinct sub-arch tag so a runtime dispatcher can refuse to bind placeholder axes"
        );
        // Backbone family = qwen3 (best-guess placeholder, documented in
        // the variant doc comment).
        assert_eq!(
            file.get("vokra.moss_tts.llm.family")
                .and_then(|v| v.as_str()),
            Some("qwen3")
        );
        // BF16 payload survives byte-for-byte.
        let info = file
            .tensor_info("embed.weight")
            .expect("emitted GGUF must carry the tensor");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());
        // llm.hidden_dim must be a nonzero u32 (Local placeholder = 2560).
        // Guards against a silent regression where the selector routes
        // AudioInstruct8b to a zero-sentinel branch.
        match file.get("vokra.moss_tts.llm.hidden_dim") {
            Some(GgufMetadataValue::U32(v)) => assert!(
                *v > 0,
                "placeholder hidden_dim must be a positive u32 (Local family = 2560), got {v}"
            ),
            other => panic!("vokra.moss_tts.llm.hidden_dim must be a U32, got {other:?}"),
        }
        // The 8B sub-arch tag must be distinct from the 4B sibling —
        // guards against a silent selector regression where both
        // variants collapse to the same `audio_4b` tag.
        assert_ne!(
            MossTtsVariant::AudioInstruct8b.sub_arch(),
            MossTtsVariant::AudioInstruct4b.sub_arch(),
            "the 8B and 4B audio-LLM siblings must carry distinct sub-arch tags"
        );
        assert_ne!(
            MossTtsVariant::AudioInstruct8b.name(),
            MossTtsVariant::AudioInstruct4b.name(),
            "the 8B and 4B audio-LLM siblings must carry distinct NAME stamps"
        );
        assert_ne!(
            MossTtsVariant::AudioInstruct8b.upstream_hf(),
            MossTtsVariant::AudioInstruct4b.upstream_hf(),
            "the 8B and 4B audio-LLM siblings must carry distinct upstream_hf slugs"
        );
    }
}
