//! # vokra-models
//!
//! Native model implementations for Vokra (SRS §1.3: "モデル自前実装。
//! piper-plus native TTS を含む" — self-implemented models, including the
//! piper-plus native TTS).
//!
//! Models are re-implemented in Rust in the whisper.cpp style: the model
//! *definition* lives here and only upstream **checkpoints** are consumed
//! (converted offline to GGUF). No ONNX graph is ever loaded at runtime
//! (FR-LD-05, permanent constraint).
//!
//! M0 content, one submodule per work package:
//!
//! - [`silero_vad`] — **M0-05**: Silero VAD v5 as a 1:1-preserved dedicated
//!   subgraph (LSTM state h/c kept intact);
//! - [`whisper`] — **M0-06**: Whisper base — encoder, decoder and beam search;
//! - [`speaker`] — **M0-08**: the native CAM++ (3D-Speaker) speaker encoder
//!   (reference fbank → 192-d embedding) for zero-shot voice cloning;
//! - [`piper_plus`] — **M0-07**: the piper-plus inference core (MB-iSTFT-VITS2
//!   text encoder / duration predictor / flow / MB-iSTFT decoder) as **Vokra's
//!   first native TTS** (FR-MD-03; client decision 2026-07-02 — the former wrap
//!   approach is abolished). G2P stays in `vokra-piper-plus` for now.
//!
//! Each submodule implements the matching engine trait from
//! [`vokra_core::engines`] (`VadEngine` / `AsrEngine` / `TtsEngine`) so it can
//! be injected into a `Session` without `vokra-core` knowing any model
//! specifics.

// M4-04 T10/T11: standalone codec GGUF binders (Mimi / DAC) — dumb bridges
// from the converter-derived tensors to the vokra-ops RVQ decode inputs.
// SoTA plan Phase 2 (2026-07-24): NVIDIA Canary-1B-v2 — multilingual
// multi-task ASR / AST (25 European languages). FastConformer encoder
// (32 layers / d_model=1024 / MHA / attention_bias=true / num_mel_bins=128)
// + Transformer decoder (8 layers / d_model=1024 / MHA / cross-attn to
// encoder + FFN, AED style) + unified SentencePiece vocab (16,384 tokens
// with task tokens `<source_lang>`, `<target_lang>`, `<taskname>`, `<pnc>`,
// `<itn>`, `<timestamp>`, `<diarize>`, `<emotion>` inline). Every hparam
// stated on the model card is transcribed verbatim; every hparam not on
// the card is transcribed from the shared FastConformer-Transformer AED
// reference config (fast-conformer_aed.yaml — the Canary family reference).
// Weights: CC-BY 4.0 (AttributionRequired — FR-MD-09 attribution surface).
// Reuses two existing ops (vokra_ops::conformer for the encoder body,
// vokra_core::decode::beam_search for the attention-decoder search) rather than
// duplicating.
// SoTA plan Phase X (2026-07-25): forced-alignment ops
// (CLAUDE.md 音声特化オペレータ §"Alignment / Duration / Prosody" —
// `force_align`). Two members:
//   * `ctc_segmentation` (Kürzinger et al., Interspeech 2020; reference
//     implementation github.com/lumaku/ctc-segmentation, Apache-2.0) is a
//     pure host-side algorithm — Viterbi over the standard CTC extended
//     sequence — with no external weights. Emits `Vec<AlignedToken>` for
//     word / sub-word / character granularity uniformly.
//   * `align::charsiu` — canonical `charsiu/en_w2v2_fc_10ms` GGUF loader
//     and real Wav2Vec2 frame-classification forward: waveform stem →
//     feature projection → grouped positional convolution → post-norm
//     Transformer → 42-phone head → upstream long-silence mask + monotone
//     DTW. The binder verifies pinned revision/checkpoint SHA, metadata,
//     vocabulary and every tensor shape. License = MIT (permissive).
pub mod aec;
pub mod align;
/// Native Audio Spectrogram Transformer AudioSet classifier (CPU / Metal).
pub mod ast;
/// Native Meta Audiobox Aesthetics WavLM quality scorer (CPU / Metal).
pub mod audiobox_aesthetics;
/// Shared bounded-memory AudioCraft autoregressive LM used by MusicGen and AudioGen.
pub mod audiocraft_lm;
/// Native AudioSeal watermark generator/detector (CPU / Metal).
pub mod audioseal;
pub mod canary;
/// Shared native T5 text encoder for MusicGen-family and related audio models.
pub mod t5_encoder;
// SoTA plan reuse bundle (2026-07-30): NVIDIA Canary-Qwen-2.5B —
// FastConformer encoder (reuse `canary::CanaryEncoderConfig` — Canary-1B-v2
// 32-layer × 1024 dim × 8 head × 128 mel bins, `vokra_ops::conformer` via
// `Stacking { factor: 8 }`) + Qwen LLM decoder (reuse `voxtral::TextDecoderConfig`
// — Qwen family GQA 16 Q ÷ 8 KV, `head_dim = 128`, `rope_base = 1_000_000`,
// RMSNorm ε = 1e-6, SwiGLU). Weight license = CC-BY 4.0 (attribution required
// via `canary-` prefix walk). Distinct arch tag `"canary-qwen"` from base
// `"canary"` because the LM head-swap changes the decoder topology from
// Transformer AED to Qwen LLM soft-prompt prefix (like Voxtral). No new op —
// both halves reuse existing Vokra primitives.
pub mod canary_qwen;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Multilingual TTS
// (MIT). T3 = Llama_520M backbone (hidden=1024 / n_layer=30 / MHA n_head=16
// n_head_kv=16 / head_dim=64 / SwiGLU ffn=4096 / RoPE θ=500000 llama3-scaled)
// + HiFT-GAN vocoder (S3Gen terminal — same `HiFTGenerator` topology
// CosyVoice2 / CosyVoice3 use, wired through the shared
// `cosyvoice2::hift_chain::HiFTChain` seam per SoTA plan §1(a) 訂正
// 2026-07-22). Every hparam transcribed verbatim from
// `github.com/resemble-ai/chatterbox` (`src/chatterbox/models/t3/`) —
// the release ships safetensors + Python code, no `config.json` on HF, so
// the primary source is the code. Multilingual variant identifier =
// `text_tokens_dict_size == 2454` (English-only = 704); 23 languages
// listed in `mtl_tts.py::SUPPORTED_LANGUAGES`. No new op or backend kernel
// — the Llama primitives and HiFTNet seam are shared with CosyVoice2 /
// CosyVoice3.
pub mod chatterbox;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Turbo TTS
// (MIT). 350M-parameter distilled Turbo variant of Chatterbox — swaps
// the backbone family from Llama_520M to **gpt2-medium** (30 layers ×
// 16 heads × 1024 hidden — same shape, different topology: LayerNorm
// with bias + fused QKV with bias + GELU FFN, not RMSNorm + SwiGLU),
// swaps sample rate 24 kHz → **32 kHz**, and swaps the text vocabulary
// 2454 (multilingual) / 704 (English-only) → **50 276** (GPT-2 base
// 50 257 + 19 paralinguistic tags [angry]/[fear]/[surprised]/
// [whispering]/[cough]/[laugh]/[chuckle]/… from `added_tokens.json`).
// Also shrinks speech vocabulary 8194 → 6563, max text tokens 2048 →
// 402, max speech tokens 4096 → 604, and distils the speech-token-to-
// mel decoder from 10 sampling steps to a single step. Terminal
// vocoder = S3Gen HiFT-GAN — same shared HiFTChain seam as CosyVoice2
// / CosyVoice3 / base Chatterbox (SoTA plan §1(a) 訂正 2026-07-22).
// Every hparam transcribed verbatim from `t3_turbo_v1.yaml` at
// `huggingface.co/ResembleAI/chatterbox-turbo` (fetched 2026-07-24 —
// CLAUDE.md「ハルシネーション厳禁」).
pub mod chatterbox_turbo;
// SoTA plan Phase 3 (2026-07-24): Resemble AI Chatterbox-Nano TTS
// (MIT). Compact 110M-parameter architecture advertised at
// ~3× realtime on an 8-core CPU. Keeps base Chatterbox's **Llama_520M**
// backbone (SwiGLU + RMSNorm + RoPE — MHA n_head=16 n_head_kv=16
// head_dim=64 hidden=1024 ffn=4096 layers=30, per
// `LLAMA_520M_CONFIG_DICT`; Nano's `t3_nano_v1.yaml` sets
// `llama_config_name: Llama_520M` which is authoritative over the
// stale `gpt_transformer_type: gpt2` training-side legacy flag) —
// distinct from Turbo which swaps the backbone to gpt2-medium. Adopts
// Turbo's low-latency serving profile: sample rate 24 kHz → 32 kHz;
// text vocabulary 2454 (base multilingual) / 704 (base English) →
// **50 276** (GPT-2 base 50 257 + 19 paralinguistic tags [angry] /
// [fear] / [surprised] / [whispering] / [cough] / [laugh] / [chuckle]
// / … from `added_tokens.json`); speech vocabulary 8194 → 6563; max
// text tokens 2048 → 402; max speech tokens 4096 → 604; speech-token-
// to-mel decoder distilled from 10 sampling steps to a single step.
// **Distinguishing sentinel**: `stop_text_token = 50256` (the GPT-2
// `<|endoftext|>` token id) — distinct from both base and Turbo which
// use 0. Terminal vocoder = S3Gen HiFT-GAN — same shared HiFTChain
// seam as CosyVoice2 / CosyVoice3 / base Chatterbox / Chatterbox-Turbo
// (SoTA plan §1(a) 訂正 2026-07-22). Every hparam transcribed verbatim
// from `t3_nano_v1.yaml` at `huggingface.co/ResembleAI/chatterbox-nano`
// (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
pub mod chatterbox_nano;
pub mod codec;
pub mod compute;
pub mod cosyvoice2;
pub mod dac;
/// Native DeepFilterNet3 enhancement wrapper with CPU / Metal dispatch.
pub mod deepfilternet3;
mod strict_checkpoint;
// SoTA plan Phase 3 (2026-07-24): FunAudioLLM Fun-CosyVoice3-0.5B TTS
// (apache-2.0). Same architecture as CosyVoice2 — Qwen2 LLM backbone +
// chunk-aware Flow Matching CFM + **HiFTNet** vocoder (arXiv:2505.17589
// + `cosyvoice/hifigan/generator.py` `HiFTGenerator` — SoTA plan §1(a)
// 訂正 2026-07-22). The Phase 3 refinements (Dual-Resolution Speech
// Representations + Core-Cocktail Training) are training-side and leave
// the runtime forward operators byte-identical to CosyVoice2. This
// module re-exports the CosyVoice2 HiFTChain seam directly and delegates
// its follow-up wave to the CosyVoice2 forward path — a very cheap
// follow-on that adds no new op / kernel.
pub mod cosyvoice3;
pub mod csm;
// SoTA plan Phase 1-4 (2026-07-24): nari-labs Dia-1.6B TTS
// (Apache 2.0). Text encoder + delayed-AR decoder over DAC 44.1 kHz RVQ
// frames. Config is transcribed verbatim from
// huggingface.co/nari-labs/Dia-1.6B/config.json (CLAUDE.md ハルシネー
// ション厳禁); real-checkpoint binding is a follow-up wave (T29-equivalent).
pub mod dia;
// DNSMOS P.808 / P.835 (Microsoft DNS-Challenge MOS predictors, MIT). The
// runtime strictly binds the audited 38-tensor public bundle and executes
// both released ONNX topologies natively. Learned convolutions and dense
// layers use the selected CPU/Metal GEMM backend; frontend DSP, pooling,
// chunking, and polynomial calibration remain deterministic host glue.
pub mod dnsmos_p808_p835;
// SoTA plan Phase 2 (2026-07-24): HuggingFace distil-whisper /
// distil-large-v3.5 — a distilled Whisper checkpoint that keeps the
// large-v3 encoder (32-layer / d_model=1280 / n_mels=128) intact and
// shrinks the decoder to 2 layers (same head width, same FFN dim, same
// large-v3 multilingual vocab at 51 866). No new op / kernel — the whole
// forward path is shared with the vanilla Whisper implementation, only
// `n_text_layer` differs. Config is transcribed verbatim from
// huggingface.co/distil-whisper/distil-large-v3.5/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁). **Real-checkpoint binding has landed**:
// `DistilWhisperAsr::from_gguf` loads a converted GGUF through
// `crate::whisper::WhisperAsr` (enforcing the distil invariant
// `n_text_layer < n_audio_layer` so a vanilla-Whisper GGUF cannot be
// mis-labelled), and `transcribe` — plus the `AsrEngine` impl the session
// facade consumes — delegates to that shared forward. A real
// distil-whisper GGUF therefore transcribes today, and `vokra-cli run`
// routes the `distil-whisper` arch to its ASR task. The only path that
// still hard-errors is the config-only `DistilWhisperAsr::new` scaffold
// (its standalone `DistilWhisperWeights` store was never wired to the
// shared engine — it exists for shape / invariant tests).
// Weights: MIT (Permissive — no runtime-side attribution obligation).
pub mod distil_whisper;
// SoTA plan Phase 5 JA-ASR-2 (2026-07-24): Kotoba Technologies
// **kotoba-whisper** — Whisper large-v3 encoder + a 2-layer decoder
// distilled on Japanese ReazonSpeech audio (multi-day Japanese ASR
// corpus). Same tensor topology as distil-large-v3.5 (identical shape
// quintuple `(1280, 32, 2, 128, 51866)`) but distinct upstream release
// (Kotoba Technologies) with **apache-2.0** weights (distil-whisper is
// MIT). Config is transcribed verbatim from
// huggingface.co/kotoba-tech/kotoba-whisper-v2.0/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁). **Real-checkpoint binding has landed**:
// `KotobaWhisperAsr::from_gguf` loads a converted GGUF through
// `crate::whisper::WhisperAsr` (enforcing the distil invariant
// `n_text_layer < n_audio_layer`), and `transcribe` — plus the `AsrEngine`
// impl the session facade consumes — delegates to that shared forward. A
// real kotoba-whisper GGUF therefore transcribes today, and `vokra-cli
// run` routes the `kotoba-whisper` arch to its ASR task. The only path
// that still hard-errors is the config-only `KotobaWhisperAsr::new` shell
// (no weights bound — it exists for shape / invariant tests).
// Weights: Apache-2.0 (Permissive
// — no runtime-side attribution obligation). The JA-ASR-2 axis
// (data-driven decoder depth) is honored by the shared WhisperConfig
// loader — this module rides on top of it with a distinct arch tag
// (`"kotoba-whisper"`) for correct provenance / telemetry.
pub mod kokoro;
pub mod kotoba_whisper;
// SoTA plan Phase 2 (2026-07-24): Kyutai STT-2.6B-EN — decoder-only
// English streaming ASR that consumes Mimi tokens (n_q=32, card=2048) and
// emits text tokens. Backbone is a 48-layer / dim=2048 / MHA transformer
// with RoPE (max_period=100000), RMSNorm ε=1e-8, SiLU gating, sliding
// causal attention (context=375). Depformer is present in the upstream
// config but STT sets dep_q=0 (text-only prediction). Config is
// transcribed verbatim from huggingface.co/kyutai/stt-2.6b-en/raw/main/
// config.json (CLAUDE.md ハルシネーション厳禁); real-checkpoint binding is
// a follow-up wave (T29-equivalent). Weights: CC-BY 4.0
// (AttributionRequired — FR-MD-09 attribution surface).
pub mod kyutai_stt;
// post-audit CC-gap 2026-08-13 Wave D: Meta AudioCraft MAGNeT runtime
// binder (**SCAFFOLD** — CPU-only, loud-partial forward per RMVPE /
// DNSMOS / openwakeword precedent). Consumes the two converter modules
// `magnet_small_10secs` (500 M params, 10 s horizon) +
// `magnet_medium_30secs` (1.5 B params, 30 s horizon). The runtime
// primitives `magnet_masked_decode` + `span_masking_scheduler` (the
// **FR-OP-85 anchor**) are deferred to a follow-up wave — see
// `docs/adr/M5-magnet-masked-ar-op.md` (Status: **Proposed**) for the
// proposed signatures and owner ratification queue; the codec decode
// integration (bundled 32 kHz EnCodec, FR-OP-32 CC-BY-NC-4.0
// distribution restriction) is owner-driven per the ADR §D-5. Weights:
// CC-BY-NC-4.0 (T4 tier — `--allow-noncommercial` required to publish;
// M2-13 compliance gate refuses commercial-mode load).
pub mod magnet;
// MyShell MeloTTS English / Chinese / Korean / Spanish / Japanese (MIT):
// strict 1,051-tensor public-checkpoint identity and official VITS2 config.
// Native feature-to-PCM execution is added in this module family without
// weakening the per-language raw-text frontend boundary.
pub mod melotts;
// post-audit CC-gap 2026-08-13 Wave D remaining WF8: Meta AudioCraft
// MelodyFlow T24 30secs runtime binder (**SCAFFOLD** — CPU-only,
// loud-partial forward per RMVPE / DNSMOS / openwakeword / sibling
// MAGNeT precedent). Consumes the converter module
// `melodyflow_t24_30secs` (~1 B params, DiT flow-matching editing
// backbone at 48 kHz with 24 default solver steps, Le Lan et al. 2024
// arXiv:2407.03648). The regeneration ODE integrator reuses
// `vokra_ops::flow_sampler::flow_sample` from M3-05 unchanged
// (`Schedule::Linear` + `OdeSolver::Euler` + `CfgMode::DualForward`
// matches Le Lan et al. Algorithm 1). The two runtime primitives
// still needed — `flow_editing_inversion` (reverse-ODE walk that maps
// ground-truth audio latent → noise latent under source text) and
// `t24_transformer` (the 24-layer DiT block stack with
// timestep-conditioned adaLN and dual text + audio prefix
// cross-attention) — are the **FR-OP-86 anchor** and are deferred to
// a follow-up wave per `docs/adr/M5-melodyflow-dit-sampler.md`
// (Status: **Proposed**) for the proposed signatures and owner
// ratification queue; the codec decode integration (bundled 48 kHz
// RVQ codec, FR-OP-32 CC-BY-NC-4.0 distribution restriction) is
// owner-driven per the ADR §D-5. Weights: CC-BY-NC-4.0 (T4 tier —
// `--allow-noncommercial` required to publish; M2-13 compliance gate
// refuses commercial-mode load).
pub mod melodyflow;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet TDT-0.6B-v3 — English
// streaming ASR built on a FastConformer encoder (8× subsampling,
// 24-layer / d_model=1024 / MHA) + a 2-layer 640-d RNN-T prediction
// network + a joint head with 5 TDT duration bins ([0, 1, 2, 3, 4]) and
// an 8193-piece vocab (8192 SentencePiece + 1 blank at 8192). Config is
// transcribed verbatim from
// huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁); real-checkpoint binding is a
// follow-up wave (T29-equivalent). Weights: CC-BY 4.0
// (AttributionRequired — FR-MD-09 attribution surface). Reuses two
// existing ops (vokra_ops::conformer for the encoder body,
// vokra_ops::rnnt_decode for the TDT decoder) rather than duplicating.
pub mod parakeet;
// SoTA plan Phase 2 (2026-07-24): NVIDIA Parakeet-CTC-1.1B — English ASR
// built on a FastConformer encoder (8× subsampling, 42-layer / d_model=1024
// / MHA / attention_bias=true / scale_input=true / num_mel_bins=80) + a
// single-Linear CTC head (vocab_size=1025 = 1024 SentencePiece + 1 blank
// at pad_token_id=1024). No RNN-T prediction network, no joint / duration
// head — CTC decoding is a host-side runtime function (vokra_ops::ctc_decode).
// Config is transcribed verbatim from
// huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json
// (CLAUDE.md ハルシネーション厳禁); real-checkpoint binding is a follow-up
// wave (T29-equivalent). Weights: CC-BY 4.0 (AttributionRequired — FR-MD-09
// attribution surface). Reuses two existing ops (vokra_ops::conformer for
// the encoder body, vokra_ops::ctc_decode for greedy / beam CTC decoding)
// rather than duplicating.
pub mod parakeet_ctc;
// SoTA plan Phase 2 (2026-07-24): Meta omniASR-CTC-1B — 1600+ language
// multilingual ASR built on a wav2vec 2.0 waveform-in encoder (7-layer
// Conv1D feature extractor, 320× downsampling; grouped-Conv1D positional
// encoder; 48-layer pre-norm Transformer, model_dim=1280, n_heads=16,
// ffn=5120) + a single-Linear CTC head (target_vocab_size=9812 v1 char
// tokenizer, blank at index 0 per the fairseq2 convention). No RNN-T
// prediction network, no joint / duration head — CTC decoding is a
// host-side runtime function (vokra_ops::ctc_decode). Every hparam is
// transcribed verbatim from the fairseq2 registry walk
// `omnilingual_asr/models/wav2vec2_asr/config.py::_1b_asr` →
// `wav2vec2_ssl/config.py::_1b_ssl` →
// `fairseq2/models/wav2vec2/config.py::large_lv60k` (CLAUDE.md ハルシネー
// ション厳禁); real-checkpoint binding is a follow-up wave
// (T29-equivalent). Weights: Apache-2.0 (Permissive — no runtime-side
// attribution obligation, unlike NVIDIA's CC-BY 4.0 Parakeet-CTC). The
// wav2vec 2.0 encoder body is a distinct topology from the FastConformer
// used by Parakeet-CTC (no shared vokra_ops::wav2vec2_encoder op today —
// that is a proposed name, not a landed module); the shared primitives
// reused today are vokra_ops::waveform_frontend for the 7-layer conv
// stem and vokra_ops::ctc_decode for the search.
pub mod omniasr_ctc;
// SoTA plan Phase 1-5 (2026-07-24): Zyphra Zonos-v0.1-transformer TTS
// (Apache 2.0). Single-stack GQA transformer with typed prefix conditioner
// (espeak / speaker / Fourier / integer) over DAC 44.1 kHz RVQ frames.
// Config is transcribed verbatim from huggingface.co/Zyphra/
// Zonos-v0.1-transformer/raw/main/config.json (CLAUDE.md ハルシネー
// ション厳禁); real-checkpoint binding is a follow-up wave (T29-equivalent).
pub(crate) mod mapped_weights;
pub mod mimi;
/// MioCodec 25 Hz / 44.1 kHz v2 native FSQ + AdaLN/iSTFT decoder.
pub mod miocodec;
pub mod moshi;
pub mod nanocodec;
pub mod neucodec;
pub mod xcodec2;
// Microsoft DNS-Challenge NSNet2-baseline (arXiv:2005.07551; code MIT,
// released model content CC-BY-4.0 — 2026-08-05 runtime binder). A
// denoise-family runtime binder alongside DFN3
// (`vokra_ops::denoise`) and the RNNoise v0.2 real-weight neural binder.
// NSNet2 itself is a deliberately-weaker
// industry-baseline reference for quantization-CI cross-checks
// (CLAUDE.md audio dialect §"Speech Enhancement / AGC / AEC"). REAL forward:
// STFT (n_fft=320, hop=160, win=320, causal / non-center) → log-power
// feature → fc_in + 2×GRU + fc_1/fc_2/mask + sigmoid → gated STFT → streaming
// iSTFT. Reuses the tested `vokra_ops::rnnoise_gru_forward` primitive with
// an ONNX `[Z;R;H]` → rnnoise `[R;Z;H]` load-time permutation. Env-gated
// real-weight parity harness: `crates/vokra-models/tests/parity_nsnet2.rs`
// (VOKRA_NSNET2_REAL_GGUF + VOKRA_NSNET2_REAL_WAV).
pub mod nsnet2;
pub mod piper_plus;
/// Xiph RNNoise v0.2 canonical real-weight network binder.
pub mod rnnoise;
// SoTA plan Phase 3 (2026-07-24): Alibaba **Qwen3-TTS-12Hz-0.6B-Base**
// TTS (apache-2.0 end-to-end — LM + codec + tokenizer + speaker
// encoder all under a single apache-2.0 grant, huggingface.co/Qwen/
// Qwen3-TTS-12Hz-0.6B-Base). Discrete multi-codebook LM topology:
// (a) Qwen3-flavour talker (decoder-only transformer, 28L / d=1024 /
// GQA 16Q ÷ 8KV / head_dim=128 / SwiGLU ffn=3072 / RoPE θ=1000000 /
// RMSNorm ε=1e-6, 3072-per-codebook speech vocab + 151936-token Qwen3
// text vocab, 32768 max positions) + (b) 5-layer code-predictor
// parallel head (same GQA/RoPE/RMSNorm axes, 2048 acoustic
// per-codebook vocab, emits 16 codebook rows per step) + (c) shared
// Qwen3-TTS-Codec seam (vokra_ops::qwen3_tts_codec — 16-quantizer
// semantic + acoustic split RVQ at 12.5 Hz output rate). Every hparam
// transcribed verbatim from config.json (talker.* / code_predictor.*)
// and README.md (speaker encoder 24 kHz / 1024-dim). No new op —
// consumes qwen3_tts_codec directly. Distinct arch tag from CosyVoice2
// / CosyVoice3 / Chatterbox because Qwen3-TTS is codec-LM not
// vocoder-LM — the terminal step is qwen3_tts_codec, NOT HiFTChain;
// silently sharing either sibling's arch tag would mis-route.
pub mod qwen3_tts;
// SBV2 v2 plan (2026-07-26): Style-Bert-VITS2 v2 native TTS. Clean-room
// Apache-2.0 implementation per `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
// — see `sbv2::mod` doc comment for the full reference list and the explicit
// NOT REFERENCED (AGPL-3.0) sources.
pub mod sbv2;
pub mod silero_vad;
// KWS (keyword-spotting / wake-word) family (SoTA plan KWS binder, 2026-08-05).
// First member: openWakeWord (dscripka/openWakeWord, Apache-2.0 code).
// Runtime binder for the `openwakeword_op` converter arch (2026-08-04).
// See `crates/vokra-models/src/kws/mod.rs` for the family charter.
pub mod kws;
// SoTA plan Phase 5 VAD-2 (2026-07-30): FunASR **FSMN-VAD**
// (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`, MIT). Feed-forward
// Sequential Memory Network for voice activity detection. Distinct
// posture from Silero VAD v5 (which is 1:1-preserved per FR-LD-06):
// FSMN's stateless feed-forward + memory blocks lower cleanly to a
// stack of graph-level ops (numeric core in `vokra-ops::fsmn_vad`).
// Real-weight parity deferred to owner (`docs/license-audit.md` §3.1
// sign-off recorded 2026-07-30 yousan); the model wrapper ships
// structural tests + a features-in streaming API today, PCM entry
// point returns loud `UnsupportedOp` until the fbank + LFR + CMVN
// pipeline is wired.
pub mod fsmn_vad;
pub mod speaker;
// SpeechBrain ECAPA-TDNN speaker encoder. The binder accepts only the exact
// 200-tensor public topology and runs its learned Conv1d/attention path through
// the explicit CPU/Metal Compute seam.
pub mod ecapa_tdnn;
// WeSpeaker ResNet34-LM speaker encoder. Strictly accepts the two public
// manifests and dispatches every learned Conv2D/projection GEMM through the
// selected CPU or Metal backend.
pub mod wespeaker;
// NVIDIA TitaNet-L speaker encoder. Strictly accepts the public 108-tensor
// NeMo inference manifest and dispatches its complete depthwise-separable
// Conv1D/SE/attention graph through the selected CPU or Metal backend.
pub mod titanet;
// SpeechBrain X-vector speaker encoder. Strictly accepts both existing public
// GGUF layouts (32 bare embedding tensors or 46 combined prefixed tensors),
// runs the exact 24-bin SpeechBrain frontend and five-layer TDNN, and routes
// every learned convolution through one explicit CPU/Metal Compute seam.
pub mod xvector;
// StyleTTS 2 (Li et al. 2023, arXiv:2306.07691). Config-only scaffold —
// the upstream `yl4579/StyleTTS2` release ships weights under a
// **voice-consent / disclosure usage agreement** (README §Pre-trained
// Model), NOT a standard SPDX permissive license, so the runtime is
// deliberately fail-closed: `StyleTts2Tts::from_gguf` and
// `StyleTts2Tts::synthesize` return `VokraError::NotImplemented` naming
// the licensing blocker. Architecture axes are re-implemented from the
// primary sources (upstream `models.py` + `Modules/*.py` + the paper
// §3) and are safe to depend on for downstream research callers who
// hold their own weight under a distinct SPDX id.
pub mod styletts2;
pub(crate) mod tls_scratch;
// SoTA plan Phase 4 (2026-07-24): OpenBMB **VoxCPM-0.5B** end-to-end
// diffusion-autoregressive TTS (apache-2.0 end-to-end — code + weight
// under a single grant, huggingface.co/openbmb/VoxCPM-0.5B). NEW CLASS
// vs every earlier target in this crate — the terminal decoding hop is
// neither a vocoder-LM (HiFTChain) nor a codec-LM (any RVQ / FSQ
// decoder) but a **continuous VAE decoder** consuming flow-matching
// sampler output. Topology: MiniCPM-4 LM backbone (24-layer / 1024d /
// GQA 16 Q ÷ 2 KV / SwiGLU 4096 / RoPE θ=10000 with longrope scaling /
// RMSNorm ε=1e-5 / vocab=73448) + 6-layer residual acoustic LM +
// 4-layer local encoder + 4-layer local DiT + UnifiedCFM flow-matching
// sampler (Euler solver, `inference_cfg_rate=2.0`) + AudioVAE V2
// continuous encoder / decoder (16 kHz PCM in → 25 Hz continuous
// latents → 48 kHz PCM out; `feat_dim=64` LM feature width matches VAE
// `latent_dim=64`) + inline scalar-quantization bottleneck
// (`scalar_quantization_latent_dim=256`, `scalar_quantization_scale=9`
// — inside the LM hidden stream, NOT a codec). Every hparam transcribed
// verbatim from `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json`
// and `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
// (`AudioVAEConfig` defaults). Reuses two ops: shared **new** SoTA plan
// Phase 4 primitive `vokra_ops::vae_continuous` (introduced with this
// model, shared with the planned VibeVoice consumer) and existing
// `vokra_ops::flow_sampler` (Euler / linear schedule / CFG SplitBatch).
// Distinct arch tag from every sibling — silently sharing would
// misroute the runtime dispatch.
pub mod voxcpm2;
// SoTA plan Phase 4 (2026-07-24): Microsoft **VibeVoice-1.5B** long-form
// multi-speaker end-to-end diffusion-autoregressive TTS (MIT — code +
// weight under a single grant, huggingface.co/microsoft/VibeVoice-1.5B).
// SECOND consumer of the continuous VAE + diffusion decoder class (after
// VoxCPM-0.5B) — but where VoxCPM uses a flow-matching sampler, VibeVoice
// uses a **DDPM** sampler (v-prediction + cosine β schedule). Topology:
// (a) Qwen2 decoder LM (28L / d=1536 / MHA n_head=12 n_head_kv=2 (GQA
// ratio 6) / SwiGLU ffn=8960 / RoPE θ=1_000_000 / RMSNorm ε=1e-6 /
// vocab=151_936 / max_positions=65_536 / tie_word_embeddings=true) +
// (b) acoustic σ-VAE tokenizer (vae_dim=64, mirror-symmetric
// encoder/decoder at 24 kHz, encoder_ratios=[8,5,5,4,2,2] → 7.5 Hz
// frame rate) + (c) semantic tokenizer (encoder-only deterministic,
// vae_dim=128, std_dist_type="none") + (d) diffusion head (4-layer
// AdaLN-modulated MLP, hidden=1536, head_ffn_ratio=3.0 → ffn_dim=4608,
// prediction_type="v_prediction", diffusion_type="ddpm",
// ddpm_num_steps=1000, ddpm_num_inference_steps=20,
// ddpm_beta_schedule="cosine"). Every hparam transcribed verbatim from
// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
// `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
// configuration_vibevoice.py`. Reuses two ops: shared Phase 4 primitive
// `vokra_ops::vae_continuous` (introduced with VoxCPM and shared with
// this VibeVoice consumer per the vae_continuous rustdoc) and the SoTA
// plan Phase 4 **new** primitive `vokra_ops::ddpm_sampler` introduced
// with this model. Distinct arch tag from VoxCPM / CosyVoice2/3 /
// Qwen3-TTS / Chatterbox family / Dia / Zonos / CSM / Voxtral /
// Kyutai STT / Moshi — silently sharing would misroute the runtime
// dispatch (VoxCPM → flow_sample, VibeVoice → ddpm_sample; the two
// samplers are irreconcilable).
pub mod vibevoice;
// SoTA plan Phase 5 JA-TTS-1 (2026-07-24): Aratako **Irodori-TTS-500M-v3**
// Japanese TTS (MIT). A Rectified-Flow Diffusion Transformer (RF-DiT)
// over the paired `Semantic-DACVAE-Japanese-32dim` codec (32-d continuous
// latent → 48 kHz PCM). Topology: (a) prompt-text encoder (`text_dim=512`
// / `text_layers=10` / `text_heads=8` / `text_mlp_ratio=2.6` — Llama-
// family self-attention with RoPE + a sigmoid gate on the output
// projection, initialized from the LLM-JP-3 150M checkpoint; text_vocab
// = 99574; add_bos=true), (b) reference-latent (speaker) encoder
// (`speaker_dim=768` / `speaker_layers=8` / `speaker_heads=12` /
// `speaker_mlp_ratio=2.6` / `speaker_patch_size=1`) driving speaker /
// style conditioning off a reference DACVAE latent, (c) RF-DiT body
// (`latent_dim=32` / `model_dim=1280` / `num_layers=12` / `num_heads=20`
// (`head_dim=64`) / `mlp_ratio=2.875` / `timestep_embed_dim=512` /
// `adaln_rank=192` with Low-Rank AdaLN modulation; SwiGLU FFN + RoPE;
// `norm_eps=1e-5`; joint-attention against text + speaker contexts), and
// (d) integrated duration predictor (v3 phase-2: `duration_aux_dim=14` /
// `duration_hidden_dim=1024` / `duration_layers=3` /
// `duration_attention_heads=8` / `duration_dropout=0.1` /
// `duration_architecture="token_sum_adarn_zero_no_aux"` /
// `duration_token_init_frames=9.0` /
// `duration_speaker_fusion="adarn_zero"`). Sampling: Euler ODE over the
// rectified-flow ODE (`x_t = (1-t) x_0 + t z`, `v = z - x_0`) integrated
// from t=1 to t=0 in 40 default steps under a Linear or Sway (F5-TTS)
// schedule with independent split-batch CFG on three axes (text /
// caption / speaker; scales 3.0 / 3.0 / 5.0; cfg window t ∈ [0.5, 1.0]).
// Reuses the shared `vokra_ops::flow_sampler` primitive (M3-05,
// `OdeSolver::Euler` + `Schedule::Linear` | `Schedule::Sway`) and the
// shared `crate::codec::DacCodecGguf` seam for the paired DACVAE decode.
// No new op is added — every RF-DiT / text-encoder / speaker-encoder
// building block is Linear + RMSNorm + SwiGLU + RoPE + softmax
// attention, all covered by the existing kernel inventory. Distinct
// arch tag from every sibling — silently sharing would misroute the
// runtime dispatch (VibeVoice → ddpm_sample, VoxCPM → EpsS flow_sample,
// Irodori → Linear/Sway flow_sample with a distinct latent width 32
// vs the Phase-4 siblings' 64). Every hparam transcribed verbatim from
// `github.com/Aratako/Irodori-TTS` (`configs/train_500m_v3_phase1_body.yaml`
// + `configs/train_500m_v3_phase2_duration.yaml` +
// `irodori_tts/config.py::ModelConfig`, fetched 2026-07-24 — CLAUDE.md
// 「ハルシネーション厳禁」). Weights: MIT (`Permissive` — no runtime-
// side attribution obligation; `gh api /repos/Aratako/Irodori-TTS/license`
// → `MIT`).
pub mod irodori;
// SoTA plan Phase 5 JA-TTS-2 (2026-07-24): **plain VITS** — the Kim et al.
// 2021 VITS (arXiv:2106.06103) architecture as shipped by ESPnet's
// `espnet2/gan_tts/vits/{vits,generator}.py` + the JA-family recipes
// (`egs2/jsut/tts1/conf/tuning/train_vits.yaml` and
// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml`) + COEIROINK deployments.
// Distinct from piper-plus (MB-iSTFT-VITS2): plain VITS decodes through
// a **HiFi-GAN generator directly** (no sub-band iSTFT, no PQMF), while
// piper-plus decodes through a sub-band iSTFT + PQMF post-net. Every
// architectural axis (text-encoder blocks / heads / FFN expand /
// dropout / macaron / conformer-conv, SDP kernel / dropout / flows,
// residual affine coupling flow kernel / layers / base_dilation /
// use_only_mean, HiFi-GAN decoder kernel / initial_channel / upsample
// scales+kernels / MRF resblock kernels+dilations, hidden_channels,
// segment_size, aux_channels, sample_rate) is transcribed verbatim from
// the ESPnet primary sources (fetched 2026-07-24 — CLAUDE.md
// 「ハルシネーション厳禁」). Reuses the shared `vokra_ops::hifigan_generator`
// (M3-07) primitive via `VitsJaConfig::to_hifigan_attrs`; no new op or
// backend kernel is added. **⚠️  Weight redistribution note**: the
// publicly distributed ESPnet-JSUT / ESPnet-JVS / COEIROINK JA VITS
// checkpoints ride on corpus terms that forbid re-distribution of the
// trained weight (JSUT: `Re-distribution is not permitted`, JVS: same);
// the converter default-stamps GGUFs produced from those checkpoints as
// `LicenseClass::RedistributionForbidden`. Users who trained their own
// permissive-corpus VITS override with `vokra-convert --license <spdx>`.
// Architecture is Apache-2.0 (ESPnet) + MIT (jaywalnut310/vits) and is
// always independently implementable (whisper.cpp 型 self re-imp,
// CLAUDE.md 設計判断 4).
pub mod vits_ja;
pub mod voxtral;
pub mod whisper;
pub mod zonos;

// F0 (fundamental-frequency) extractor family (FR-OP-83). Each member
// (`rmvpe` / `fcpe` / `crepe`, plus future PyIN / Harvest siblings) exposes a
// GGUF `from_gguf` loader, a fallible `extract` / `extract_real` pair carrying
// the real forward, and a `frame_times` accessor for the analysis timebase
// alone. No member answers a failure with a zero-filled track (FR-EX-08 —
// see the `f0` module doc). Kept in its own block so `rustfmt`'s alphabetical
// sort inside consecutive `pub mod` blocks does not hijack the doc-preceded
// siblings above.
pub mod f0;
// pyannote/segmentation-3.0 (Bredin, CNRS, MIT — 2026-07-30 Wave 2
// runtime scaffold with loud-partial forward). PyanNet VAD /
// speaker-segmentation backbone (SincNet → BiLSTM x2 → Linear x2 →
// powerset multiclass classifier). Config + weight-load are real; the
// inner forward returns `VokraError::UnsupportedOp` until Wave 3
// lands the SincNet primitive per
// `docs/handoff/pyannote-implementation-plan-2026-07-30.md`. Same
// posture as sibling RMVPE (`crate::f0::rmvpe`).
pub mod pyannote;
// Wave 1 2026-08-14 audit follow-up (vocoder + codec standalone binders):
// hifigan = standalone runtime binder for `hifigan_vocoder` /
// `speecht5_hifigan` converters (loud-partial `from_gguf`, real
// synthesized-weight `decode` via `vokra_ops::hifigan::hifigan_generator`).
// snac = standalone runtime binder for hubertsiuzdak/snac_{24khz,44khz}
// (loud-partial `encode`/`decode` — RMVPE / DNSMOS precedent; discovery /
// variant / config / license paths are real).
pub mod hifigan;
pub mod snac;
// FocalCodec 12.5/25/50 Hz native codec family.  The three public GGUFs
// share one WavLM + focal-modulation + BSQ + Vocos implementation; their
// compressor/decompressor time-scale factors are selected from a pinned
// release manifest rather than inferred from filenames.
pub mod focalcodec;
/// Alibaba DAMO FunCodec 16 kHz / 32-codebook native RVQ + SEANet decoder.
/// The exact public 230-tensor checkpoint supports frame-major token-to-PCM
/// decode on CPU and Metal; waveform encode remains an explicit unsupported
/// operation at the CLI boundary.
pub mod funcodec;
/// Strict Full/Nano MOSS Audio Tokenizer binder plus native CPU/Metal decode.
/// Full keeps its 7 GB artifact mapped and materialises one layer at a time to
/// 24 kHz mono; Nano owns its compact weights and emits 48 kHz stereo. The
/// public Nano artifact's historical Full metadata stamp is accepted only
/// behind its exact complete tensor manifest and remains visible as a repair
/// requirement.
pub mod moss_audio_tokenizer;
/// Native MOSS-TTS family. Nano is runnable end to end with its explicit
/// codec companion; Base/v1.5 additionally have a strict 463-tensor
/// mmap-backed Delay checkpoint binder while their generation graph lands.
pub mod moss_tts;
/// Fudan/OpenMOSS SpeechTokenizer 16 kHz / eight-codebook native RVQ +
/// weight-normalized SEANet decoder. The exact public 166-tensor checkpoint
/// supports frame-major token-to-PCM decode on CPU and Metal; waveform encode
/// remains an explicit unsupported operation at the CLI boundary.
pub mod speechtokenizer;
// Wave 2 2026-08-14 audit follow-up (vocoder recovery + music-und):
// vocos = standalone Vocos runtime binder with complete ConvNeXt-1D + iSTFT
// forward. CPU preserves the independent scalar reference; Metal dispatches
// dense/grouped Conv1D, LayerNorm and GELU through the whole-model backend
// seam without a silent CPU fallback.
// bigvgan = standalone binder for the 4 nvidia/bigvgan_* variants;
// decode delegates to existing `vokra_ops::bigvgan_generator`.
// beat_this = CPJKU Transformer beat/downbeat tracker (MIT, ISMIR
// 2024) — first MIR primitive per 2026-08-14 audit "music evaluation
// / analysis" gap.
pub mod beat_this;
pub mod bigvgan;
// Wave 3 2026-08-14 audit follow-up: MT3 audio→MIDI transcriber
// (Magenta, Apache-2.0 code; weight LicenseClass::Unknown fail-closed
// pending owner spec-review of gs://mt3/checkpoints/). T5-small
// encoder-decoder scaffold + MidiEvent enum, transcribe loud-partial
// pending the MT3-specific T5X binder/decoder composition + MIDI event
// codec Rust port; the shared relative-position and encoder pieces exist.
pub mod mt3;
pub mod vocos;
/// Native WavTokenizer large-speech 75 token/s codec decoder.
pub mod wavtokenizer;
/// Native m-a-p YuE 44.1 kHz Vocos upsampler (CPU / Metal).
pub mod yue_upsampler;
// Wave 4 2026-08-14 audit follow-up (LIB.RS RULE — append at end
// with Wave 4 comment marker): ReDimNet2 speaker encoder
// (Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM, apache-2.0). 2D
// basic_resnet stem + 1D conv+att blocks + ASTP pooling scaffold,
// encode loud-partial pending WeSpeaker Python source transcription
// (wespeaker/models/redimnet2.py + redimnet.py + arXiv:2402.01049).
pub mod redimnet;
// Wave 4 2026-08-14 audit follow-up: NVIDIA Sortformer diar-4spk-v1
// (CC-BY-NC-4.0 T4) — loud-partial runtime binder (FastConformer
// primitive exists in vokra_ops; 18-layer plain Transformer + per-
// frame 4-sigmoid head composition + tensor-name walk + region-
// merging port pending; §3.1 sign-off row 449 = ☑ Research-only
// 2026-08-04 yousan, X-Codec-2 T4 precedent).
pub mod sortformer_diar_4spk_v1;
// Wave 5 2026-08-14 audit follow-up: Meta HT-Demucs music source
// separation runtime binder (loud-partial per sortformer / mt3 /
// redimnet precedent — hybrid waveform-branch U-Net + spectrogram-
// branch U-Net + BiLSTM + Transformer bottleneck + cross-domain
// attention + stem-sum path deferred to follow-up wave. STFT/iSTFT
// primitives exist; a generic sequence LSTM does not (the one public
// LSTM, vokra_ops::hybrid_ctc_attention::LstmLmCell, is LM-shaped, and
// Silero's is a pub(crate) fixed-width cell in the separate
// vokra-vad-micro crate), so vokra_ops::lstm is a follow-up.
// `facebook/demucs` MIT — first
// music-source-separation Permissive land after BS-Roformer
// Rejected).
pub mod demucs;
// Native AudioCraft EnCodec token-to-waveform component used by the
// authenticated public MusicGen Small/Melody composite artifacts. This is a
// runtime binder only: FR-OP-32 still forbids a standalone EnCodec converter,
// model-zoo entry, or official weight publication. RVQ, Conv1D,
// ConvTranspose projection and LSTM projection execute through one selected
// CPU/Metal backend; unsupported coverage fails explicitly.
pub mod audiocraft_encodec;
// Wave 5 2026-08-14 audit follow-up (music generation — first runtime
// binder): musicgen = Meta MusicGen family runtime binder (small 300M
// + medium 1.5B, CC-BY-NC-4.0 T4). Autoregressive transformer LM over
// EnCodec RVQ tokens (4 codebooks, 50 Hz frame rate, 32 kHz output)
// conditioned on frozen T5-base text encoder with delay pattern across
// codebooks. Real config / variant / mapping-owned load / weight-license
// surfacing; the AudioCraft Medium/Large raw LM step and its
// delay/CFG/sampling route execute on CPU/Metal, while Small/Melody
// expose their embedded EnCodec RVQ + complete SEANet waveform decoder on
// CPU/Metal but retain a distinct composite LM-layout gap. Full generate
// remains loud-partial pending prompt tokenization/conditioner, Small/Melody
// LM composition, and explicit codec companions for LM-only files. Primary
// sources:
// huggingface.co/facebook/musicgen-{small,medium} +
// github.com/facebookresearch/audiocraft (MIT code) + arXiv:2306.05284
// (Copet et al. 2023). §3.1 row 399 = ☑ Research-only 2026-08-01
// yousan (X-Codec-2 T4 precedent). Loud-partial pattern per Wave 2-4
// precedent (vocos / bigvgan / snac / mt3 / kyutai_stt / parakeet_ctc
// / redimnet / sortformer).
pub mod musicgen;
// Conv-TasNet Libri1Mix native enhancement runtime. Strictly binds the pinned
// 345-tensor Asteroid checkpoint and all twelve topology/provenance axes, then
// runs the 512-filter encoder, 24-block dilated TCN masker and learned decoder.
// Conv1D, grouped Conv1D and Global LayerNorm share one explicit CPU/Metal
// backend route; scalar activations/layout/residual work stays host glue. The
// independent Asteroid 0.7.0 fixture pins encoder, bottleneck, mask and final
// waveform parity. Upstream license declarations conflict, so conversion and
// runtime support are complete while official redistribution remains
// fail-closed (`LicenseClass::Unknown`).
pub mod conv_tasnet;
// Wave 5 2026-08-14 audit follow-up (separation-runtime binder — LIB.RS
// RULE append at end with Wave 5 comment marker): SpeechBrain SepFormer
// family (Subakan et al. 2021 / arXiv:2010.13154 §3, apache-2.0) — 7
// variants (wsj02mix / libri2mix / libri3mix / wham16k-enhancement /
// whamr16k / whamr8k / dns4-16k-enhancement) share the `sepformer`
// converter arch + this runtime binder. Strict `from_gguf` validates the
// architecture, variant, category, output-stream count and all 417 learned
// tensors; `separate()` implements the encoder, dual-path Transformer masker
// (16 intra/inter Transformer layers), `n_out`-way head and decoder
// per `github.com/speechbrain/speechbrain/blob/develop/speechbrain/
// lobes/models/dual_path.py` + `resepformer.py` + arXiv:2010.13154.
// Sibling to the music-source-separation `demucs` (MIT Permissive) and
// the speech-enhancement `conv_tasnet` (runtime complete; redistribution
// unresolved) landed just above — SepFormer's §3.1 rows remain commercial
// (2026-07-30 / 2026-08-01 yousan) per HF cardData apache-2.0.
pub mod sepformer;
/// Native TIGER-DnR and TIGER-speech CPU / Metal source separators.
pub mod tiger;
// Standalone execution front door for the three public BERT-family sidecars
// used by SBV2. Their learned forwards remain in `vokra-bert`; this binder
// adds strict arch/licence/input gates and an explicit CPU-only backend
// contract so the released GGUFs can execute without a parent TTS model.
pub mod bert_runtime;
// Wave 6 2026-08-14 audit follow-up (denoise runtime binder — LIB.RS
// RULE append at end with Wave 6 comment marker): GTCRN
// (Xiaobin-Rong/gtcrn, MIT, ~23K params, arXiv:2211.02063) —
// Groupwise Temporal Convolutional Recurrent Network speech
// enhancement runtime binder for the `gtcrn` converter arch. Real
// from_gguf (arch check + 5-axis vokra.gtcrn.* chunk-group +
// non-empty tensor gate + weight-license class); `denoise()`
// loud-partial pending grouped Conv2D + PReLU + SB-TF-LSTM +
// ERB grouping primitives (Wave 3 blocker). §3.1 sign-off BLANK
// (row added Wave 6, fail-closed). Denoise alternative sibling to
// `denoise` (DFN3, ERB analysis/synthesis + CRN) / `nsnet2`
// (Microsoft DNS baseline, 2-layer GRU + 3-Linear mask) / `rnnoise`
// (Xiph GRU + BFCC) — arch tag `gtcrn` distinct from every sibling
// per FR-EX-08. First entry on the ultra-lightweight streaming
// enhancement arm from the Wave 6 audit follow-up.
pub mod gtcrn;
// Wave 6 2026-08-14 audit follow-up (music-generation runtime binder —
// LIB.RS RULE append at end with Wave 6 comment marker): CVSSP
// AudioLDM 2 family (`cvssp/audioldm2` base + `cvssp/audioldm2-large`
// sibling, CC-BY-NC-SA-4.0 T4 tier NonCommercialShareAlike doubly
// restrictive = NC gate + SA cascade both fail-closed at M2-13,
// `docs/license-audit.md` §3.1 row 400 = ☑ Research-only 2026-08-01
// yousan). Real `from_gguf` (arch check + `AudioLdm2Variant` Base/
// Large discrimination via `vokra.model.name` + `AudioLdm2Config`
// primary-source constant fallback for the yet-to-be-stamped
// `vokra.audioldm2.*` chunk group + non-empty tensor gate + weight-
// license class surfacing); `generate()` loud-partial pending T5-base
// text encoder + CLAP text encoder + latent-diffusion U-Net + VAE
// decoder + HiFi-GAN vocoder composition — `vokra_ops::flow_sampler`
// (M3-05) already covers the DDIM / DPM++ ODE integrator so the
// follow-up wave is composition + three greenfield forward bodies
// (T5-base + CLAP + audio-VAE-decoder), not five greenfield kernels.
// Primary sources: `huggingface.co/cvssp/audioldm2` +
// `github.com/haoheliu/AudioLDM2` + Liu et al. 2024 ICML
// `arXiv:2308.05734`. Distinct arch tag from every sibling music-
// generation family (`musicgen` AR-over-EnCodec / `magnet_*`
// non-autoregressive masked-LM / `melodyflow_t24_30secs` DiT
// flow-matching / `audiogen_medium` AR / `jasco_400m_chords_drums`
// AR / `stable_audio_open_small` different-conditioner LDM /
// `ace_step` chunked-AR / `bs_roformer` source-separation) — silently
// sharing arch would misroute the runtime dispatch to a wrong-shape
// forward (FR-EX-08). Mirror of the RMVPE / pyannote / MusicGen /
// Conv-TasNet loud-partial precedent (CLAUDE.md 教訓 (a) — 'loud-
// partial は fake-complete より honest').
pub mod audioldm2;
// Wave 6 2026-08-14 audit follow-up (music-generation family — LIB.RS
// RULE append at end with Wave 6 comment marker): audiogen = Meta
// AudioCraft AudioGen-Medium runtime binder (`facebook/audiogen-medium`,
// CC-BY-NC-4.0 T4). 1.5B autoregressive transformer LM over EnCodec RVQ
// tokens (4 codebooks, 50 Hz, 16 kHz) conditioned on frozen T5-large text
// encoder with delay pattern across codebooks — MusicGen sibling by
// topology (Kreuk et al. 2023 arXiv:2209.15352), distinct by training
// corpus (environmental sounds / SFX vs music) and by arch tag
// `audiogen` (FR-EX-08 dispatch safety — audit follow-up 2026-08-14
// retags the converter from the Wave 5 shared `musicgen` arch tag so a
// future modality-specific head (SFX-only conditioning stack, stereo
// output head, per-class embedding table) does not silent-mis-bind
// against MusicGen's music-only runtime path). Real config / mapping-owned
// load / weight-license surfacing; the raw AudioCraft LM step now executes
// on CPU/Metal with native delay/CFG/sampling. Full generate remains
// loud-partial pending T5-large companion tokenization/encoding and SEANet
// waveform decode (the EnCodec RVQ latent-fold primitive exists via
// vokra_ops::encodec_rvq_decode). §3.1 row 402 = ☑ Research-only 2026-08-01 yousan
// (X-Codec-2 T4 precedent inheritance). Loud-partial pattern per Wave 5
// musicgen precedent.
pub mod audiogen;
// Wave 6 2026-08-14 audit follow-up (music-generation runtime binder —
// LIB.RS RULE append at end with Wave 6 comment marker): Meta AudioCraft
// JASCO 400M Chords+Drums (`facebook/jasco-chords-drums-400M`,
// CC-BY-NC-4.0 T4 tier). Flow-matching music generation with joint
// text + chord + drum symbolic conditioning (Tal et al. 2024
// arXiv:2406.10970 "Joint Audio and Symbolic Conditioning for
// Temporally Controlled Text-to-Music Generation"). Real arch check +
// `JascoVariant::Chords400mDrums` bound (1B Chords+Drums / 400M Melody
// / 1B Melody sibling names emit "converter not yet in-tree" errors
// naming the paper as the primary source) + per-variant primary-source
// constant fallback config + tensor-manifest non-emptiness gate +
// weight-license class surfacing + argument-validation ordering
// (empty-symbolic-conditioning gate fires BEFORE loud-partial per
// FR-EX-08 — JASCO's key contribution is joint symbolic conditioning,
// silent fall-through to text-only would misrepresent the model);
// `generate()` loud-partial pending (a) joint symbolic conditioning
// encoder stack (frozen T5-base text encoder + JASCO chord encoder +
// drum encoder), (b) AudioCraft flow-matching transformer stack with
// joint conditioning cross-attention, and (c) the flow-matching sampler
// + EnCodec RVQ decode composition — TWO primitives LANDED via M3-05
// `vokra_ops::flow_sampler::flow_sample` + M4-04
// `vokra_ops::encodec_rvq_decode`. Primary sources:
// `huggingface.co/facebook/jasco-chords-drums-400M` +
// `github.com/facebookresearch/audiocraft` + `arxiv.org/abs/2406.10970`.
// §3.1 row 458 = ☑ Research-only 2026-08-04 yousan (X-Codec-2 T4
// precedent + MusicGen family T4 precedent, `--allow-noncommercial`
// publish path). Loud-partial pattern per Wave 2-5 precedent
// (vocos / bigvgan / snac / mt3 / musicgen / magnet / melodyflow /
// sortformer / audioldm2 / audiogen).
pub mod jasco;
// Wave 7 2026-08-14 audit follow-up (audio-tagging runtime binder —
// LIB.RS RULE append at end with Wave 7 comment marker): PANNs Cnn14
// (`nicofarr/panns_Cnn14`, license Unknown fail-closed, ~80M params,
// Kong et al. 2020 arXiv:1912.10211) — Pretrained Audio Neural Networks
// Cnn14 checkpoint runtime binder for the `panns` converter arch
// (§3.1 row 479 BLANK, upstream reference `qiuqiangkong/audioset_tagging_cnn`
// is MIT but the HF mirror LICENSE is un-verified so the converter
// defaults to `Unknown`, fail-closed at M2-13). Real from_gguf (arch
// check + PannsConfig primary-source constant fallback for the
// yet-to-be-stamped `vokra.panns.*` chunk group + non-empty tensor
// gate + weight-license class surfacing); classify() loud-partial
// pending (a) log-mel front-end binding against upstream
// `torchlibrosa.STFT` + `LogmelFilterBank` reference
// (`pytorch/pytorch_utils.py`), (b) 6-stage × 2-conv CNN14 backbone
// (Conv2D(3×3) + BatchNorm2D + ReLU + AvgPool2D(2×2) with channel
// plan 64→64→128→128→256→256→512→512→1024→1024→2048→2048 per
// `pytorch/models.py class Cnn14`), and (c) global attention pooling
// + fc1 Linear(2048, 2048) + fc_audioset Linear(2048, 527) + sigmoid
// 527-way head (Kong et al. §III-A). Primary sources:
// `github.com/qiuqiangkong/audioset_tagging_cnn` +
// `github.com/qiuqiangkong/panns_inference` +
// `arxiv.org/abs/1912.10211` + `research.google.com/audioset/ontology/`.
// Distinct arch tag `panns` from sibling audio-tagging / audio-
// embedding models (`yamnet` MobileNetV1 depthwise-separable / `ast`
// patch-embed Transformer / `clap` contrastive text-audio dual-encoder
// / `mert` HuBERT-derived masked prediction / `muq` Mel-RVQ + BEATs
// teacher / `dasheng` MAE ViT/ConvNeXt universal / `beats` iterative
// acoustic tokenizer) — silently sharing arch would misroute runtime
// dispatch to a wrong-topology loader (FR-EX-08 boundary). Loud-partial
// pattern per Wave 2-6 precedent (vocos / bigvgan / snac / mt3 /
// musicgen / magnet / melodyflow / sortformer / audioldm2 / audiogen /
// jasco / redimnet / gtcrn).
pub mod panns;
// Wave 7 2026-08-14 coverage-audit-2026-08-03 wave-b follow-up
// (streaming S2S runtime binder — LIB.RS RULE append at end with
// Wave 7 comment marker): ICTNLP LLaMA-Omni2 (apache-2.0 default,
// Qwen2.5 派生 chain, T1 tier Permissive; per memory
// `[[feedback-license-signoff-primary-source]]` §3.1 sign-off column
// stays BLANK until owner primary-source verifies HF card + Qwen2.5
// license-inheritance chain + speech-decoder training-corpus audit +
// ELVIS Act 精査). Four sibling HF releases (`ICTNLP/LLaMA-Omni2-{7B,
// 3B-Bilingual,1.5B,32B}`) share the three-stage streaming S2S
// topology (Whisper-family speech encoder + Qwen2.5-family text
// backbone + streaming AR speech decoder). Loud-partial pattern per
// Wave 4-6 precedent (kyutai_stt / canary_qwen / voxtral /
// firered_asr_llm_l / sepformer / demucs / conv_tasnet / musicgen /
// gtcrn / audiogen / audioldm2 / jasco): `converse()` returns
// `VokraError::UnsupportedOp(_)` naming the primary source URLs +
// missing shared primitives (Qwen2.5 backbone forward, Whisper-family
// speech encoder forward, streaming AR speech decoder, streaming
// session infrastructure) — never a silent noise stream (FR-EX-08).
// ELVIS Act 精査 = task-oriented S2S with fixed decoder voice (not
// target-speaker cloning), main-repo stay per CLAUDE.md 設計判断 8.
// Primary sources: `huggingface.co/ICTNLP/LLaMA-Omni2-7B` + sibling
// repos + `github.com/ictnlp/LLaMA-Omni2` (ACL 2025).
pub mod llama_omni2;
// Wave 7 2026-08-14 audit follow-up (denoise runtime binder — LIB.RS
// RULE append at end with Wave 7 comment marker; RETRY of Wave 6 lost
// item per WAVE 6 LESSON): StoRM (`sp-uhh/storm`, MIT,
// arXiv:2312.09386) — Stochastic Regeneration Model for Speech
// Enhancement and Dereverberation runtime binder for the `storm`
// converter arch. Real `from_gguf` (arch check + 6-axis `vokra.storm.*`
// chunk-group + non-empty tensor gate + weight-license class);
// `enhance()` loud-partial pending NCSN++ v2 U-Net score-network with
// sigma-conditional FiLM + OUVE-SDE (Ornstein-Uhlenbeck Variance-
// Exploding stochastic differential equation) predictor-corrector
// sampler primitives (~two greenfield ops). §3.1 sign-off BLANK (row
// added Wave 7, fail-closed, publish gated on owner ADR — Google
// Drive-only upstream, no HF mirror as of 2026-08-14 = need decision
// between T4 Research-only precedent vs new T1 Permissive
// GitHub-source precedent). Denoise + dereverb alternative sibling to
// `denoise` (DFN3), `nsnet2`, `rnnoise`, `gtcrn` — arch tag `storm`
// distinct from every sibling per FR-EX-08. First diffusion-based
// entry on the enhancement arm from the Wave 7 audit follow-up retry.
pub mod storm;
// Wave 7 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 7 comment marker): microsoft/wavlm-base-plus-sv speaker
// verification (CC-BY-SA-3.0 → Copyleft) — WavLM Transformer encoder
// with **gated relative position bias + convolutional position-bias
// fusion** (Chen et al. arXiv:2110.13900 "WavLM: Large-Scale
// Self-Supervised Pre-Training for Full Stack Speech Processing")
// + XVector head + Additive Margin Softmax loss (5-block TDNN →
// statistics pooling → 512-d embedding, VoxCeleb1 fine-tuned).
// Sibling speaker-fleet arch (never `campplus` / `ecapa_tdnn` /
// `wespeaker` / `titanet` / `speaker_3d` / `redimnet`). Loud-partial
// pattern per redimnet precedent — `from_gguf` real with strict
// `vokra.wavlm.*` scalar + axis-array chunk-group presence enforcement
// (13 scalar axes + 6 axis-arrays: conv_{dim,stride,kernel}_{0..6} +
// tdnn_{dim,kernel,dilation}_{0..4}) + tensor-manifest non-emptiness
// gate + weight-license class surfacing; `encode()` returns
// `UnsupportedOp` naming (i) 7-layer 1D conv feature-extractor stem
// (HuBERT/wav2vec2 lineage), (ii) WavLM Transformer encoder with
// gated relative position bias + convolutional position-bias fusion
// (WavLM-specific primitive that neither wav2vec2 nor HuBERT expose),
// (iii) XVector head + Additive Margin Softmax with primary-source
// URLs `huggingface.co/microsoft/wavlm-base-plus-sv` +
// `github.com/microsoft/UniSpeech` + arXiv:2110.13900. §3.1 sign-off
// BLANK (fail-closed default per feedback-license-signoff-primary-source
// memory — Copyleft class needs owner sign-off since redistribution
// obligations propagate to downstream consumers = an owner-scope
// legal decision, not a CC judgement).
pub mod wavlm;
// Native emotion2vec+ Large classifier. The strict binder accepts only the
// audited public 185-F32-tensor identity and its all-or-none 17-key topology
// contract: 7-layer waveform stem, 5 grouped positional convolutions, 10
// learned tokens, 4 context plus 8 global post-norm ALiBi blocks, mean pooling
// and the official bilingual 9-way head. Learned hot operations honor explicit
// CPU/Metal selection; every other backend is an error with no CPU fallback.
pub mod emotion2vec;
// Moonshine ASR family (`moonshine-ai/moonshine-{tiny,base}`, MIT) — real-time
// transformer encoder-decoder ASR alternative to Whisper for edge
// (Jeffries et al. 2024, arXiv:2410.15608 "Moonshine: Speech
// Recognition for Live Transcription and Voice Commands"). **Distinct
// from every Whisper-family sibling** (whisper / distil_whisper /
// kotoba_whisper) in two significant ways: (1) **no mel front-end** —
// the model consumes raw 16 kHz PCM directly via a learned Conv1D
// stem (strides = [64, 3, 2] → 384x downsampling); (2) RoPE plus a GELU
// encoder and SwiGLU decoder. The strict binder loads the complete official
// manifest and pinned HF BPE tokenizer; `transcribe` runs native Conv1D,
// encoder/decoder attention, tied logits and greedy decoding. Non-CPU
// backends fail explicitly until the composed attention path is wired. No new
// runtime dependency is introduced.
pub mod moonshine;
// Meta Facebook Denoiser DNS48 (`facebookresearch/denoiser`,
// CC-BY-NC-4.0 research-only): strict 48-F32-tensor binder and native
// utterance forward. The exact official sinc resampling, five Conv1d+GLU
// encoder blocks, two-layer causal LSTM, additive skips, and five
// Conv1d+GLU+ConvTranspose1d decoder blocks run on Mac CPU or Apple Metal.
// Every learned reduction goes through `Compute`; unsupported backends fail
// before PCM processing and never fall back silently. Runtime remains free of
// pickle, ONNX, protobuf, and external dependencies.
pub mod facebook_denoiser;
/// Native Alibaba FRCRN-SE-16K enhancement (CPU / Metal).
pub mod frcrn;
// Wave 9 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 9 comment marker): Voila (`maitrix-org/Voila`, MIT, 2025) —
// Maitrix's full-duplex speech-to-speech dialog family. Runtime binder
// loud-partial per llama_omni2 / moshi / csm full-duplex S2S precedent:
// real `from_gguf` (arch check + non-empty tensor gate + weight-license
// class surfacing; converter does NOT stamp `vokra.voila.*` topology
// chunks — variant discrimination across Voila-base / Voila-chat /
// Voila-audio-alpha / Voila-autonomous-preview deferred to a follow-up
// wave), `converse()` returns `UnsupportedOp` naming four pieces:
// (i) full-duplex session manager (concurrent input / output streams,
// barge-in / talk-over, ~195 ms end-to-end latency budget — mirror of
// moshi / csm session code architecturally, but Voila's session
// topology is distinct), (ii) Whisper-lineage speech encoder (raw PCM
// -> semantic features), (iii) Voila LLM backbone forward (transformer
// decoder producing speech tokens), (iv) speech decoder + vocoder head
// (speech tokens -> output PCM; integrated with the LLM backbone,
// plugs into the sibling hifigan / bigvgan / vocos vocoder-family
// neighbourhood at the follow-up wave). Primary source:
// github.com/maitrix-org/Voila. Distinct-arch discipline: sibling S2S
// arches enumerated (`moshi` Kyutai full-duplex + Mimi CC-BY 4.0,
// `csm` Sesame full-duplex + Mimi Apache 2.0, `llama_omni2` ICTNLP
// streaming half-duplex + Qwen2.5 + Whisper Apache 2.0) — voila is a
// FIRST full-duplex S2S with an integrated neural vocoder entry
// distinct from moshi/csm Mimi-based codec approach; sharing arch with
// any sibling would mis-route runtime dispatch (FR-EX-08). §3.1
// sign-off row for `maitrix-org/Voila` (MIT) BLANK per fail-closed rule
// (owner-side commercial-sign-off gate; feedback-license-signoff-
// primary-source memory — CC MUST NOT sign, row addition is a follow-up
// task for docs/license-audit.md).
//
// Post-audit CC-gap 2026-08-15: the converter this binder's error text
// already told operators to run — `vokra-cli convert --model voila` —
// did NOT exist at Wave 9 time; nothing in the tree could emit a GGUF
// this binder accepts. `crates/vokra-convert/src/models/voila.rs` now
// lands it as a BF16 pass-through skeleton, which is sufficient because
// this binder gates only on arch + a non-empty tensor manifest and walks
// no specific tensor name. It deliberately stamps neither `vokra.voila.*`
// axes (not transcribable, shift per release) nor
// `vokra.provenance.upstream_hf` (per-release HF weight repo ids are
// owner-verified at bind time) — provenance rides
// `vokra.provenance.upstream_url = github.com/maitrix-org/Voila`. The
// §3.1 row is STILL absent, so redistribution remains fail-closed and the
// row addition is still the owner follow-up recorded above.
pub mod voila;
// Wave 9 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 9 comment marker): **CLAP** (`laion/clap-htsat-fused`,
// **Apache-2.0** — permissive, T1 tier redistributable) — LAION
// Contrastive Language-Audio Pretraining (Wu et al. 2023 ICASSP
// arXiv:2211.06687 "Large-scale Contrastive Language-Audio Pretraining
// with Feature Fusion and Keyword-to-Caption Augmentation") runtime
// binder for the `clap` converter arch (Wave 9, 2026-08-14, converter
// side already landed at `crates/vokra-convert/src/models/clap.rs`,
// TIER 1 F wave 2026-07-30). Real `from_gguf` (arch check + non-empty
// tensor gate + weight-license class surfacing; the converter does
// NOT stamp `vokra.clap.*` topology chunks — plain BF16 pass-through
// per `wespeaker` / `neucodec` / `ecapa_tdnn` precedent, so this
// binder mirrors the arch-only-gate posture rather than the strict
// axis-array `wavlm_sv` / `storm` / `moonshine` posture);
// `encode_audio()` returns `UnsupportedOp` naming (i) HTSAT audio
// encoder walk (Hierarchical Token-Semantic Audio Transformer per
// Chen et al. 2022 — a Swin-Transformer variant over log-mel
// spectrogram patches with local/global feature fusion for the "fused"
// variant per Wu et al. §3.2), (ii) RoBERTa text encoder walk (paired
// text tower sharing the projection head, required at bind time so
// future `encode_text(caption)` can produce a vector in the same
// embedding space), (iii) shared projection head Linear(hidden, 512)
// after mean pooling over the HTSAT time-frequency tokens producing
// the final 512-dim paired language-audio embedding. Primary sources:
// `github.com/LAION-AI/CLAP` + `arXiv:2211.06687` +
// `huggingface.co/laion/clap-htsat-fused` (8.1M+ downloads at survey
// time, one of the highest-download HF audio releases). Distinct-arch
// discipline: sibling audio-embedding / classification / SSL-encoder
// arches enumerated (`panns` fixed 527-way AudioSet head,
// `emotion2vec` fixed 9-way emotion head, `wavlm_sv` XVector speaker
// verification, `ecapa_tdnn` / `wespeaker` / `campplus` speaker
// embedding, `audioldm2` audio generation, `musicgen` music
// generation, `wav2vec2_ctc` CTC ASR) — CLAP's defining trait is a
// two-tower contrastive paired language-audio embedding (Wu et al.
// 2023 ICASSP) with no fixed downstream classifier head; sharing arch
// with any sibling would mis-route runtime dispatch (FR-EX-08).
// §3.1 sign-off ☑ Commercial 2026-07-30 yousan (docs/license-audit.md,
// TIER 1 F wave 2026-07-30, apache-2.0 → Permissive T1 tier) — no
// additional license-audit action needed this wave (row already
// present). Cross-crate string handshake via duplicated
// `pub const ARCH = "clap"` (mirror of the converter's ARCH constant,
// preserving the layered convention `vokra-ops → nothing GGUF-aware`,
// `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
// `vokra-convert → GGUF writer`).
pub mod clap;
// Wave 9 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 9 comment marker): **3D-Speaker ERes2Net**
// (`iic/speech_eres2net_sv_zh-cn_16k-common`, **Apache-2.0** — permissive,
// T1 tier redistributable) — Alibaba DAMO's Enhanced Res2Net
// speaker-verification encoder (Chen et al. 2023 arXiv:2305.12838 "An
// Enhanced Res2Net with Local and Global Feature Fusion for Speaker
// Verification") runtime binder for the `speaker_3d` converter arch
// (Wave 9, 2026-08-14, converter already landed at
// `crates/vokra-convert/src/models/speaker_3d.rs`). Loud-partial per the
// emotion2vec / moonshine / panns / redimnet / wavlm / storm precedent
// (CLAUDE.md 教訓 (a): "loud-partial は fake-complete より honest") —
// real `from_gguf` (arch check + non-empty tensor gate + weight-license
// class surfacing — the converter stamps `apache-2.0` verified
// 2026-07-25 → `LicenseClass::Permissive`), `encode(fbank) ->
// Result<Vec<f32>>` returns `UnsupportedOp` naming three deferred
// pieces: (i) ERes2Net stem (initial 3x3 Conv2D + BN + ReLU followed by
// four Res2NetBlock stages with local + global feature fusion —
// upstream `speakerlab/models/eres2net/ERes2Net.py` in the 3D-Speaker
// toolkit), (ii) Attentive Statistics Pooling head (temporal
// attention-weighted mean + std concatenation, Chen et al. 2023
// section 3.3), (iii) Linear embedding projection (Linear(embed_in,
// EMBEDDING_DIM=192) + L2-normalize — the standard 192-d speaker
// embedding shared with sibling CAM++ so downstream `spk_proj` /
// cosine-similarity consumers see a compatible vector width). Primary
// source: github.com/alibaba-damo-academy/3D-Speaker. Distinct-arch
// discipline: sibling speaker-encoder arches enumerated (`campplus`
// CAM++ densely-connected TDNN, `xvector` Kaldi 5-layer TDNN + stats
// pool, `ecapa_tdnn` SpeechBrain SE-Res2Net + attentive stats pool,
// `titanet-large` NVIDIA ContextNet + attentive stats pool, `wavlm_sv`
// Microsoft WavLM base + XVector speaker head) — 3D-Speaker ERes2Net is
// distinct on the Res2Net-stem-with-local-and-global-fusion + ASP head
// axis; sharing arch with any sibling would mis-route runtime dispatch
// (FR-EX-08). §3.1 sign-off row for
// `iic/speech_eres2net_sv_zh-cn_16k-common` (Apache-2.0) BLANK per
// fail-closed rule (owner-side commercial-sign-off gate; feedback-
// license-signoff-primary-source memory — CC MUST NOT sign, row
// addition is a follow-up task for docs/license-audit.md). Cross-crate
// string handshake via duplicated `pub const ARCH = "speaker_3d"`
// (mirror of the converter's ARCH constant, preserving the layered
// convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
// reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`).
pub mod speaker_3d_eres2net;

// Wave 9 2026-08-14 audit follow-up (LIB.RS RULE append at end with
// Wave 9 comment marker): FunAudioLLM SenseVoiceSmall
// (`FunAudioLLM/SenseVoiceSmall`, **FunASR MODEL_LICENSE** — a custom
// upstream licence NOT in `LicenseClass::from_class_str`'s SPDX matcher,
// so classifier returns `Unknown` fail-closed per
// `[[feedback-license-signoff-primary-source]]`) — multi-task speech
// understanding runtime binder (multilingual ASR + LID + SER + AED, 50
// languages, ~15x lower latency than Whisper-Large per An et al. 2024
// arXiv:2407.04051) for the `sensevoicesmall` converter arch (converter
// side lives at `crates/vokra-convert/src/models/sensevoicesmall.rs`,
// coverage-audit-2026-08-03 Wave B). Module name deliberately
// `sensevoicesmall_runtime` per the task spec so the runtime-side
// `pub mod sensevoicesmall_runtime;` here cannot clash with any
// pre-existing crate-local `sensevoicesmall` symbol namespace inside
// `vokra-models`. Real `from_gguf` (arch check + sibling-ASR-family
// mis-route hint list + non-empty tensor gate + weight-license class
// surfacing that correctly fail-closes to `Unknown` for the FunASR
// MODEL_LICENSE default); `transcribe()` returns `UnsupportedOp`
// naming (i) SAN-M (Modified Multi-Head Attention with a parallel
// Memory-block Fully-Connected branch) enhanced Conformer encoder
// walk — distinct from vanilla Conformer used by Parakeet /
// Kotoba-Whisper / Reazonspeech-NeMo-v2, per An et al. 2024 §III-A +
// `funasr/models/sanm/`, (ii) four per-task heads on the shared
// encoder embedding in load-bearing tuple order [ASR, LID, SER, AED],
// 50-language coverage per §IV / TableI. Primary sources: HF release
// `huggingface.co/FunAudioLLM/SenseVoiceSmall`, reference code
// `github.com/FunAudioLLM/SenseVoice`, paper
// `arxiv.org/abs/2407.04051` — all three cited verbatim in the
// loud-partial diagnostic so a follow-up wave has exactly three
// anchors to walk. Distinct-arch discipline: sibling ASR-family arches
// enumerated (`whisper` / `distil_whisper` / `kotoba_whisper` /
// `moonshine` / `parakeet` / `parakeet_ctc` / `canary` /
// `canary_qwen` / `omniasr_ctc` / `kyutai_stt` /
// `reazonspeech_nemo_v2`) — SenseVoiceSmall is the FIRST multi-task
// entry (ASR + LID + SER + AED) on the ASR family arm, sharing arch
// with any sibling would mis-route runtime dispatch to a single-task
// loader missing the LID / SER / AED heads (FR-EX-08). §3.1 sign-off
// remains BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — the FunASR
// MODEL_LICENSE is a custom upstream licence requiring primary-source
// review; CC does NOT sign).
pub mod sensevoicesmall_runtime;

// Wave A (2026-08-15) — TorchAudio-SQUIM runtime binder (LIB.RS RULE: append
// at the END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit).
//
// Closes a real gap: `crates/vokra-convert/src/models/torchaudio_squim.rs`
// produced a GGUF stamped `vokra.model.arch = "torchaudio_squim"` that NOTHING
// in the workspace read, so every converted bundle was unloadable. This module
// is that consumer.
//
// TorchAudio-SQUIM (`pytorch/audio`, code BSD-2-Clause; Kumar et al. 2023
// ICASSP arXiv:2304.01448) ships TWO independently-trained bundles that answer
// DIFFERENT questions, and they stay two separate entry points here rather
// than being folded into one `score()`:
//   * SQUIM **Objective** (`squim_objective_dns2020.pth`) — genuinely
//     reference-FREE estimation of STOI + PESQ + SI-SDR from a degraded
//     waveform alone;
//   * SQUIM **Subjective** (`squim_subjective_bvcc_daps.pth`) — MOS estimation
//     against a NON-MATCHING REFERENCE (an unrelated clean utterance), i.e.
//     NOT reference-free in the same sense; it takes a second waveform.
//
// Complementary to (not a duplicate of) the `vokra-eval` SI-SNR / SI-SDR /
// STOI metrics landing in this same wave: SQUIM *estimates* those quantities
// with no reference, `vokra-eval` *computes* them from a paired clean
// reference. On a corpus where a reference exists, the computed metric is the
// ground truth this estimator's error is measured against.
//
// Real: strict `vokra.model.arch` verification (foreign GGUFs — including the
// sibling `category="eval"` fleet `utmos` / `utmosv2` / `dnsmos` /
// `nisqa_v2_weight` — are refused by name, FR-EX-08); bundle-prefix tensor
// routing (`objective.` / `subjective.`, the contract
// `tools/parity/torchaudio_squim_prepare_checkpoint.py` declares and the
// DNSMOS `p808.` / `p835.` convention mirrors) with empty-manifest,
// no-head and unroutable-tensor refusals that NAME the offender; optional
// `vokra.squim.sample_rate` validation carrying an explicit `ConfigSource` so
// a stamped 16 kHz is never confused with an assumed one; truthful partial-
// bundle advertisement; empty-PCM rejection; fail-closed weight-license
// surfacing.
//
// Loud-partial: `estimate_objective` / `estimate_subjective` each return
// `UnsupportedOp` naming their OWN deferred stages — the objective head's
// learnable 1-D encoder + DPRNN stack + three transformer metric heads; the
// subjective head's `wav2vec2_base` SSL encoder (the shared wav2vec2-lineage
// gap with `emotion2vec` / `wavlm`) + attentive pool + linear projector + the
// NMR pairing — plus the upstream file to transcribe and the reserved
// `vokra.squim.{objective,subjective}.topology` chunk to stamp. No score value
// is ever fabricated (FR-EX-08).
//
// The DPRNN stack additionally needs a recurrent cell, and that is a LIFT, not
// greenfield RNN work: four recurrent bodies already exist — the public
// shape-generic GRU `vokra_ops::rnnoise::gru_forward` (re-exported as
// `vokra_ops::rnnoise_gru_forward`), the public LSTM gate body
// `vokra_ops::hybrid_ctc_attention::LstmLmCell`, and the `pub(crate)`
// `kokoro::nn::BiLstm1d` / `pyannote::bilstm::BiLstmLayer`. What `vokra-ops`
// does not expose is a REUSABLE BARE CELL over a plain feature vector:
// `LstmLmCell::step` welds a token-embedding lookup onto its input and a vocab
// log-softmax onto its output, and the two BiLSTMs are unreachable from an op.
// `squim::missing_primitive_note` is the authoritative wording — pinned by
// `objective_head_loud_partials_naming_the_dprnn_and_its_missing_primitive`,
// which asserts the stale phrasing is absent as well as asserting the live
// blocker is present. This marker carried that stale phrasing ("the only
// recurrent kernels in the tree are the DFN3-specific GRU stack inside
// `vokra_ops::denoise`") until 2026-08-15: false from the moment
// `vokra-ops/src/rnnoise.rs` landed, and left behind by the same change that
// corrected every other copy.
//
// Deliberately does NOT implement `vokra_core::engines::MosScorerEngine`:
// `MosScore`'s fields are DNSMOS-shaped (`p808` / `sig` / `bak` / `ovrl`), each
// naming a specific ITU-T protocol, and SQUIM's BVCC+DAPS MOS is none of them —
// squatting `p808` would misrepresent which protocol produced the number, and
// the trait signature has nowhere to put the required non-matching reference.
//
// §3.1 sign-off remains BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]`). Noted for the owner, not
// decided here: the sidecar records the *code* as BSD-2-Clause but the
// *weights* as CC-BY-4.0 (objective) and CC-BY-NC-4.0 (subjective), so a
// bundle carrying the subjective head is NC-encumbered despite the stamped
// `bsd-2-clause` — `Squim::has_subjective()` surfaces that for a publish gate.
pub mod squim;

// Wave A (2026-08-15) — UTMOSv2 runtime binder (LIB.RS RULE: append at the
// END of the `pub mod` block with a Wave marker; do NOT alphabetize).
//
// Closes a write-only gap: `crates/vokra-convert/src/models/utmosv2.rs`
// (landed 2026-08-04) can PRODUCE a `utmosv2` GGUF, but nothing in the
// workspace read the `utmosv2` arch string back, so a converted checkpoint
// could never be loaded — which blocks the NFR-QL-02 5% quality gate (an
// M5 / v1.0 GA DoD item) that needs a loadable reference-free MOS
// instrument. Upstream `sarulab-speech/UTMOSv2` is **MIT** per the
// converter docstring's recorded verification against
// `github.com/sarulab-speech/UTMOSv2/blob/main/LICENSE`
// (`LicenseClass::Permissive`); this binder mirrors that recorded
// verification rather than re-deriving it.
//
// **Real**: strict `vokra.model.arch == "utmosv2"` verification that
// enumerates every sibling eval-family arch tag — `utmos`
// (UTMOS22-strong on wav2vec2-BASE with a different head layout, runtime
// skeleton in `vokra-eval::metrics::utmos`), `dnsmos` (ITU-T P.808 /
// P.835 CNN emitting 1 or 3 scalars over a fixed 9.01 s window),
// `nisqa_v2_weight`, `torchaudio_squim` — so a mis-routed GGUF fails with
// a specific message instead of a downstream missing-tensor error
// (FR-EX-08). Config parsed from EXACTLY the metadata the converter writes
// (`vokra.model.name` / `vokra.model.category` /
// `vokra.provenance.upstream_hf` + the `stamp_provenance` group), with a
// fail-closed `LicenseClass::Unknown` when the class stamp is absent
// (`[[feedback-license-signoff-primary-source]]`). Every emitted tensor is
// bound with real per-tensor shape checks: dtype restricted to the
// converter's F32 / F16 / BF16 pass-through arm (a K-quant means the file
// was re-quantised AFTER conversion, which would silently shift the
// calibration of the very instrument the quality gate trusts), rank >= 1,
// no zero-extent dimension, payload length, no duplicate names, and at
// least one rank >= 2 weight matrix (a Regressor head cannot exist without
// a Linear weight). Plus the `require` / `require_shape` / `load_f32`
// named-tensor accessors the follow-up forward wave binds against, and the
// terminal ACR clamp `clamp_to_mos_range` (real today — it refuses a
// non-finite regressor output rather than clamping NaN to a
// plausible-looking MOS).
//
// **Loud-partial**: `Utmosv2::predict_mos` returns `UnsupportedOp` naming
// the three deferred stages (spectrogram-domain branch / wav2vec2-large
// SSL encoder + listener-domain conditioning / Regressor head fusion), the
// concrete `vokra.utmosv2.*` axes the converter must start stamping, the
// absent sidecar `tools/parity/utmosv2_prepare_checkpoint.py`, the
// re-conversion command, and both primary sources
// (`github.com/sarulab-speech/UTMOSv2`; Baba et al.,
// `arxiv.org/abs/2409.09305`). The conversion contract is a verbatim float
// pass-through that stamps NO topology axes, so the stack is not
// primary-source-transcribable and would be silent-wrong if best-guessed;
// a fabricated MOS would silently corrupt the NFR-QL-02 gate it feeds
// (CLAUDE.md 教訓 (a) 「loud-partial は fake-complete より honest」).
//
// `vokra_core::engines::MosScorerEngine` is deliberately NOT implemented:
// its `MosScore` payload is DNSMOS-shaped (p808 / sig / bak / ovrl) and
// folding UTMOSv2's single utterance-level MOS into `p808` would attach an
// ITU-T P.808 claim the model does not make. §3.1 sign-off stays BLANK
// (owner-only).
pub mod utmosv2;

// Wave A (2026-08-15) — NISQA v2 runtime binder (LIB.RS RULE: append at the
// END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit).
//
// Closes a real gap: `crates/vokra-convert/src/models/nisqa_v2_weight.rs`
// (landed coverage-audit-2026-08-03 Wave D T4) produced a GGUF stamped
// `vokra.model.arch = "nisqa_v2_weight"` that NOTHING in the workspace read
// back, so every converted checkpoint was unloadable. This module is that
// consumer.
//
// NISQA v2 (`github.com/gabrielmittag/NISQA`; Mittag, Naderi, Chehadi,
// Möller 2021, `arxiv.org/abs/2104.09494`) is a non-intrusive
// MULTIDIMENSIONAL speech-quality predictor. Unlike the sibling
// `dnsmos_p808_p835` binder it emits FIVE scores per forward — overall MOS
// plus noisiness / discontinuity / coloration / loudness — surfaced as
// `NisqaScore` rather than collapsed to a scalar, because those sub-scores
// are the whole reason to run NISQA instead of DNSMOS. `HEAD_ORDER =
// ["mos","noi","dis","col","loud"]` is pinned verbatim from the
// `y_hat[:, i]` assignments in `nisqa/NISQA_lib.py`: the paper's prose
// lists coloration BEFORE discontinuity, the opposite of the tensor
// layout, so reading the prose order would silently swap two
// plausible-looking scores.
//
// NATIVE CPU/METAL: the exact public 94-tensor `NISQA_DIM` release is pinned
// by its complete name/shape manifest and provenance. The official
// checkpoint args are stamped by
// `tools/parity/nisqa_v2_weight_prepare_checkpoint.py`; the historical public
// GGUF is accepted only through the same exact manifest and receives those
// audited values in memory. STFT/mel, inference BatchNorm, ReLU, layout glue
// and exact PyTorch-bin adaptive max pooling are deterministic host work.
// Every learned Conv/Linear/attention reduction uses the selected Compute
// backend's GEMM/softmax/LayerNorm after a whole-model preflight, so Metal
// never falls back to CPU. `score_at_sample_rate` requires the real WAV rate
// because the release keeps native sample rate; the legacy rate-less entry
// fails explicitly rather than guessing 48 kHz.
//
// LICENSING: the upstream README states verbatim that the CODE is MIT but
// that the released WEIGHTS (`nisqa.tar` / `nisqa_mos_only.tar` /
// `nisqa_tts.tar`) are CC-BY-NC-SA-4.0 →
// `LicenseClass::NonCommercialShareAlike` = **T4 / research-only**: never
// publishable without `publish-one.sh --allow-noncommercial`, and the
// share-alike obligation cascades to any derived GGUF. The strict converter
// rejects conflicting `--license` overrides, so the canonical weights cannot
// be relabelled as permissive.
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
//
// `vokra_core::engines::MosScorerEngine` is deliberately NOT implemented,
// for the same reason as `utmosv2` but in the opposite direction: its
// `MosScore` payload is DNSMOS-shaped (p808 / sig / bak / ovrl) and has no
// slot for coloration, discontinuity or loudness, so the impl would
// silently drop three of the five dimensions this module exists to keep.
// Cross-crate string handshake via duplicated
// `pub const ARCH = "nisqa_v2_weight"` (mirror of the converter's ARCH
// constant, preserving the layered convention `vokra-ops → nothing
// GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
// `vokra-convert → GGUF writer`).
pub mod nisqa;

// Native TEN-VAD v1.0 runtime. The strict offline ONNX sidecar and converter
// pin the 19-tensor manifest, topology, revision, and source hash. Runtime is
// first-party Rust: LPCNet-derived streaming features feed the released
// separable-conv/two-LSTM graph through the common `VadEngine` API. Upstream's
// Apache text carries additional non-compete/application-only deployment
// terms, so canonical GGUFs are fail-closed for redistribution. The frontend
// preserves both BSD-2-Clause and BSD-3-Clause notices required by `NOTICES`.
pub mod ten_vad;

// Wave B (2026-08-15) — smart-turn v2 runtime binder for semantic
// turn-completion / endpointing (LIB.RS RULE: append at the END of the
// `pub mod` block with a Wave marker; do NOT alphabetize — rustfmt has
// reordered these before and broken a commit).
//
// Closes a real gap: `crates/vokra-convert/src/models/smart_turn.rs`
// produced a GGUF stamped `vokra.model.arch = "smart_turn"` that NOTHING in
// the workspace read back, so every converted checkpoint was unloadable.
// This module is that consumer.
//
// NOT A VAD, despite appearances — and this is the point most likely to be
// got wrong by a future reader. The converter stamps
// `vokra.model.category = "vad"` and the upstream HF card's pipeline tag is
// literally `voice-activity-detection` (HF cardData API, recorded in
// `docs/license-audit.md` §3.1 row "Smart-Turn v2", fetched 2026-07-30) —
// both are CATALOG labels, not architectural claims. A VAD answers "is
// there speech in this frame?" per frame; smart-turn
// (`pipecat-ai/smart-turn-v2`, BSD-2-Clause, Wav2Vec2-base backbone) answers
// "has this speaker finished their turn?" once per utterance-length
// segment. A mid-sentence pause is speech-present AND turn-incomplete at
// the same instant, which is exactly why a realtime pipeline runs both
// rather than one in place of the other. The API is therefore
// `SmartTurn::predict_endpoint(pcm, sample_rate) -> TurnPrediction` (ONE
// completion probability per segment), and
// `vokra_core::engines::VadEngine` is deliberately NOT implemented: its
// `VadStreamHandle::push_pcm` returns one probability PER FRAME, so the
// impl would have to either broadcast the single decision across every
// frame (fabricated per-frame signal) or return a one-element vector
// (silently wrong frame count) — both FR-EX-08 violations. Same posture
// `nisqa` / `utmosv2` take toward `MosScorerEngine`. Note this makes
// smart_turn distinct from the sibling `ten_vad` binder directly above,
// which DOES implement `VadEngine` because it really is a frame-level VAD.
//
// REAL: the strict converter and binder pin all 223 source tensors / 221
// canonical tensors, source revision and SHA-256 values, processor geometry,
// Wav2Vec2-base axes, folded positional-convolution weight norm, and the
// exact attention-pooling / classifier head. `predict_endpoint` implements
// the complete native CPU forward. The fixed 16-second processor contract is
// preserved efficiently: the first convolution's right-padded GroupNorm
// statistics are computed without convolving the constant zero tail, and
// only the encoder queries selected by Pipecat's ratio mask are evaluated.
// Independent Transformers parity is generated by
// `tools/parity/smart_turn_prepare_checkpoint.py` and gated by
// `tests/parity_smart_turn.rs`. Input validation rejects mismatched sample
// rates, empty or over-long utterances, and non-finite PCM before inference.
//
// LICENSING: `bsd-2-clause` → `LicenseClass::Permissive` (T1 Commercial).
// The `docs/license-audit.md` §3.1 row is owner-only and was already signed
// ☑ Commercial on 2026-07-30; this module does not touch it. Cross-crate
// string handshake via duplicated `pub const ARCH = "smart_turn"` (mirror
// of the converter's ARCH constant, preserving the layered convention
// `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
// `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`).
pub mod smart_turn;

// Wave B (2026-08-15) — FireRedVAD runtime binder (LIB.RS RULE: append at the
// END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit).
//
// Closes a real gap: `crates/vokra-convert/src/models/firered_vad.rs` (TIER 1
// F wave, 2026-07-30) produced a GGUF stamped `vokra.model.arch =
// "firered_vad"` / `vokra.model.name = "firered-vad"` / `vokra.model.category
// = "vad"` / `vokra.provenance.upstream_hf = "FireRedTeam/FireRedVAD"` that a
// workspace-wide grep proved NOTHING read back — every converted FireRedVAD
// checkpoint was unloadable. This module is that consumer.
//
// FireRedVAD (`huggingface.co/FireRedTeam/FireRedVAD`) is Xiaohongshu
// FireRedTeam's transformer-based streaming voice-activity detector, shipped
// alongside FireRedASR / FireRedTTS. It really IS a frame-level VAD, so —
// unlike the `smart_turn` binder directly above, which answers a different
// question once per segment — it DOES implement
// `vokra_core::engines::VadEngine`, exposing the same per-frame
// speech-probability API shape as `silero_vad` / `fsmn_vad` / `ten_vad`
// (one-shot `FireredVad::speech_probabilities` plus a streaming handle) so
// `vokra-core`'s `stream::open_stream` glue sees no asymmetry.
//
// REAL: strict `vokra.model.arch` verification that refuses foreign GGUFs
// loudly with the whole `category = "vad"` sibling fleet enumerated
// (`silero-vad` 1:1-preserved subgraph / `fsmn-vad` FunASR FSMN over Kaldi
// fbank+LFR+CMVN / `pyannote-segmentation` speaker segmentation reduced to a
// VAD signal / `smart_turn` end-of-turn prediction, plus the `vad-kws`
// neighbour `ten_vad`) — the shared category tag alone cannot disambiguate
// them, only the arch tag can; a non-empty tensor-manifest gate; a by-name
// tensor lookup that NAMES an absent tensor instead of returning `None` for a
// caller to swallow; an optional `KEY_REQUIRED_TENSORS` (`Array<String>`)
// declaration that turns a truncated / mis-merged GGUF into a LOAD-time
// failure naming the first missing tensor; the optional all-or-nothing
// `vokra.firered_vad.*` hyper-parameter group (absent → `config()` is `None`
// and the checkpoint still binds, because refusing it would re-open the very
// gap this module closes; partially stamped → loud, naming the missing key;
// `0` sentinel or indivisible `d_model % n_heads` → loud); a sample-rate guard
// that refuses a mismatched rate with `InvalidArgument` rather than resampling
// silently; and weight-license surfacing that fail-closes to
// `LicenseClass::Unknown`.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `FireredVad::speech_probabilities` / `FireredVadStream::push_pcm`
// return `VokraError::UnsupportedOp` naming three concrete blockers — (i) the
// MISSING TOPOLOGY TRANSCRIPTION: the converter describes the model only as a
// "transformer-based streaming VAD" and copies every tensor under its verbatim
// upstream safetensors name, so nothing in-repo transcribes the front-end,
// the encoder geometry or the output-head class ordering, and a best-guess
// topology would emit a SHAPE-VALID probability vector that is quietly wrong
// (a mis-guessed head count or a swapped speech column never crashes — note
// the blocker is GEOMETRY, not kernels: the encoder body composes from
// transformer primitives Vokra already carries once the geometry is known);
// (ii) the MISSING METADATA, i.e. the `vokra.firered_vad.*` group the
// converter does not stamp — head count in particular is invisible in the
// weight shapes whenever QKV is packed into one projection; (iii) the MISSING
// SIDECAR `tools/parity/firered_vad_prepare_checkpoint.py`, which has never
// been written. The stream deliberately never returns `Ok(vec![])`, because an
// empty return is indistinguishable from "no frame completed" and would let a
// caller loop forever believing the VAD runs. No fabricated speech
// probabilities are ever emitted (FR-EX-08). Deliberately NO
// `FireredVadConfig::upstream_default()`: unlike `FsmnVadConfig`, FireRedVAD's
// axes are not stated in any available primary source, so a default would be
// invented numbers wearing an authoritative face (CLAUDE.md ハルシネーション厳禁)
// — the same posture the sibling `ten_vad` binder takes.
//
// LICENSING: the converter stamps `apache-2.0` → `LicenseClass::Permissive` on
// the stated basis that the FireRedTeam family LICENSE pins Apache-2.0 across
// the team's releases and the FireRedVAD card inherits it — an
// INHERITED-FAMILY determination, not a transcription of a FireRedVAD-specific
// licence file. This binder only SURFACES whatever class the GGUF carries and
// fail-closes to `Unknown` when nothing is stamped.
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign, and does
// not treat the converter's default as a sign-off).
//
// Cross-crate string handshake via duplicated `pub const ARCH =
// "firered_vad"` / `NAME` / `CATEGORY` / `UPSTREAM_HF` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops →
// nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`). Note the arch uses `_` while the
// name uses `-`; both spellings are load-bearing on the wire and are pinned
// separately by a test.
pub mod firered_vad;

// Wave C1 (2026-08-15) — runtime binder for the `parakeet-tdt-1_1b` converter
// arch (NVIDIA Parakeet-TDT-1.1B, CC-BY 4.0). Closes a read-side gap: the
// converter (`crates/vokra-convert/src/models/parakeet_tdt_1_1b.rs`) has
// stamped `vokra.model.arch = "parakeet-tdt-1_1b"` since the 2026-08-03 Wave B
// coverage audit, but NO code in the workspace read that arch string — weights
// could be converted and then never loaded.
//
// SCOPE: the TDT DECODE leg is REAL and wired to `vokra_ops::rnnt_decode`'s
// `RnntDecoderKind::Tdt` mode (the primitive already implements TDT: per-frame
// vocab argmax over V+1 plus duration argmax over D, duration-driven frame
// skip, zero-duration multi-emit cap), reachable via
// `ParakeetTdt11b::decode_tdt` on a caller-materialized joint buffer. The full
// PCM -> text `transcribe` is a LOUD-PARTIAL (`VokraError::UnsupportedOp`)
// because the converter is a BF16 pass-through skeleton that stamps NO
// `vokra.parakeet_tdt_1_1b.*` hparam chunk group — its docstring defers the
// 1.1B axis transcription to the owner. Copying the `parakeet` (0.6B-v3) axes
// would be fabrication: the releases are known to differ (0.6B-v3 = 24 layers /
// 128 mel bins / attention_bias=false; the 1.1B CTC sibling = 42 layers / 80
// mel bins / attention_bias=true), so the 1.1B TDT axes are genuinely unknown.
// That is why this module carries no config constant, unlike `parakeet` /
// `parakeet_ctc` whose `config.json` files were fetched and transcribed
// verbatim (2026-07-24).
//
// Note the arch/name spelling split: the arch tag uses an UNDERSCORE
// (`parakeet-tdt-1_1b`) while the model name / publish slug / CLI argument use
// a DOT (`parakeet-tdt-1.1b`). Both are load-bearing on the wire and pinned
// separately by tests (mirror of the `firered_vad` split note above).
//
// LICENSING: the converter stamps `cc-by-4.0` -> `AttributionRequired`, so the
// FR-MD-09 attribution surface activates (the NVIDIA attribution must be
// displayed). This binder only SURFACES whatever class the GGUF carries and
// fail-closes to `Unknown` when nothing is stamped; the `--license` override is
// a supported convert-time path, so nothing is asserted.
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
pub mod parakeet_tdt_1_1b;

// Runtime-gap Wave 4 ASR (2026-08-22) — exact aiola Whisper-Medusa-v1
// binder. The canonical converter removes the checkpoint's outer
// `whisper_model.` wrapper, stamps the ordinary Whisper hparams/frontend/
// tokenizer contract, and preserves all eleven `medusa_heads.*` modules.
// This module owns the distinct arch so a vanilla Whisper loader cannot
// silently omit the official module-0 residual-SiLU output transform.
//
// `AsrEngine` executes that real module-0 forward. Accelerated future-token
// draft/verify/accept remains an explicit separate unsupported API until a
// tree-attention driver exists; it never aliases plain or module-0 decoding.
// The pinned checkpoint and upstream source revisions, config hash, complete
// 22-tensor head manifest, and MIT provenance are all validated fail-closed.
pub mod whisper_medusa;

// Wave C1 (2026-08-15) — runtime binder for the `firered_asr_aed_l` converter
// arch (FireRedTeam/FireRedASR-AED-L, Apache-2.0, ~1.1B params / ~2.2 GB BF16,
// Mandarin ASR). LIB.RS RULE: append at the END of the `pub mod` block with a
// Wave marker; do NOT alphabetize — rustfmt has reordered these before and
// broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/firered_asr_aed_l.rs`, coverage-audit
// 2026-08-03 Wave B) has stamped `vokra.model.arch = "firered_asr_aed_l"` /
// `vokra.model.name = "firered-asr-aed-l"` / `vokra.model.category = "asr"` /
// `vokra.provenance.upstream_hf = "FireRedTeam/FireRedASR-AED-L"`, but a
// workspace-wide grep proved NOTHING read that arch string back — every
// converted checkpoint was unloadable. This module is that consumer.
//
// REAL: strict `vokra.model.arch` verification that refuses a foreign GGUF
// loudly with the whole `category = "asr"` fleet enumerated — headed by the
// SAME TEAM's `firered_asr_llm_l` (Conformer encoder + audio-text adapter +
// Qwen2 LM decoder), then `whisper` / `distil-whisper` / `kotoba-whisper`,
// `canary` / `canary-1b-flash` / `canary-qwen`, `parakeet-tdt` /
// `parakeet-ctc` / `omniasr-ctc`, `kyutai-stt` / `voxtral` /
// `nemotron_asr_streaming`, `moonshine`; the shared `asr` category can never
// disambiguate them, only the arch tag can. Plus: a non-empty tensor-manifest
// gate; a by-name `dims()` lookup that NAMES an absent tensor instead of
// returning `None` for a caller to swallow; the optional `KEY_REQUIRED_TENSORS`
// (`Array<String>`) declaration that turns a truncated / mis-merged GGUF into a
// LOAD-time failure naming the first missing tensor; the optional
// all-or-nothing `vokra.firered_asr_aed_l.*` hyper-parameter group (absent ->
// `config()` is `None` and the checkpoint still binds, because refusing it
// would re-open the very gap this module closes; partially stamped -> loud,
// naming the missing key; `0` sentinel or indivisible `d_model % n_head` on
// either stack -> loud); a sample-rate guard that refuses a mismatched rate
// with `InvalidArgument` rather than resampling silently; tokenizer-presence
// surfacing; and weight-license surfacing that fail-closes to
// `LicenseClass::Unknown`. It implements `vokra_core::engines::AsrEngine` so
// the handle is reachable from the session glue exactly like Whisper /
// Voxtral / distil-Whisper.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `FireredAsrAed::transcribe_tokens` and the `AsrEngine` path return
// `VokraError::UnsupportedOp` naming three concrete blockers — (i) NO
// HYPER-PARAMETER TRANSCRIPTION: the converter stamps no
// `vokra.firered_asr_aed_l.*` group at all, and the audit ticket
// `docs/tickets/coverage-audit-2026-08-03/wave-b/firered-asr-aed-l.md` records
// 「FireRedTeam AED は Whisper と shape 互換ではない、独自 hparam」, so
// borrowing Whisper's axes would emit a SHAPE-VALID token sequence that is
// quietly wrong (head counts in particular are unrecoverable from the weight
// shapes when QKV is packed); (ii) NO TENSOR-NAME MANIFEST: the converter's own
// docstring defers real-weight binding to an upstream manifest fetch that has
// not happened, and the names in its test module are synthetic round-trip
// placeholders; (iii) NO TOKENIZER: an AED decoder emits token ids and no
// `vokra.tokenizer.model` blob rides on these GGUFs, so nothing can render
// Mandarin text. The blocker is GEOMETRY and VOCABULARY, not kernels — the same
// audit ticket records that the existing encoder / decoder / cross-attention /
// beam-search / STFT / mel-filterbank primitives cover this topology once the
// geometry is known. No fabricated token ids or transcript are ever emitted
// (FR-EX-08).
//
// STRUCTURED FOR THE LLM SIBLING: `FireredAsrAedEncoderConfig` and
// `FireredAsrAedDecoderConfig` are separate public types so that IF a real
// checkpoint later shows the two FireRedTeam releases share an acoustic
// encoder, the encoder half lifts into a shared type by a move rather than a
// rewrite. The module deliberately does NOT assert that they share one: the
// in-repo descriptions differ (the LLM converter calls the AED release's
// encoder a "Transformer encoder" and its own a "Conformer encoder") and no
// primary source in this repository settles it. The LLM variant
// (`firered_asr_llm_l`) still has NO binder of its own — a separate gap.
//
// LICENSING: the converter stamps `apache-2.0` -> `LicenseClass::Permissive`.
// This binder only SURFACES whatever class the GGUF carries and fail-closes to
// `Unknown` when nothing is stamped. `docs/license-audit.md` §3.1 already
// carries an owner-signed row for this release; this module neither reads nor
// writes that sign-off (owner-only per
// `[[feedback-license-signoff-primary-source]]`).
//
// Cross-crate string handshake via duplicated `pub const ARCH =
// "firered_asr_aed_l"` / `NAME` / `CATEGORY` / `UPSTREAM_HF` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops →
// nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`). Note the arch uses `_` while the
// name uses `-`; both spellings are load-bearing on the wire and are pinned
// separately by a test.
pub mod firered_asr_aed;

// Wave C1 (2026-08-15) — Sber GigaAM family runtime binder (LIB.RS RULE: append
// at the END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit).
//
// Closes a real gap, TWICE OVER. Two converters existed with no consumer
// anywhere in the workspace: `crates/vokra-convert/src/models/sber_gigaam_v3.rs`
// stamps `vokra.model.arch = "sber_gigaam_v3"` and
// `crates/vokra-convert/src/models/sber_gigaam_multilingual.rs` stamps
// `vokra.model.arch = "gigaam_multilingual"`. A workspace-wide grep proved
// NOTHING read either string back — every converted GigaAM checkpoint was
// unloadable. This module is that missing consumer, for both halves at once.
//
// ONE MODULE, TWO ARCH TAGS: both Wave B tickets
// (`docs/tickets/coverage-audit-2026-08-03/wave-b/sber-gigaam-{v3,multilingual}.md`)
// call for a single `ModelKind::GigaAm` carrying a variant enum, and the
// multilingual converter's own module doc records that splitting into two
// standalone converters was a deliberate WORKTREE-ISOLATION decision, not an
// architectural one. The runtime has no such constraint, so `gigaam` accepts
// `ACCEPTED_ARCHS` and distinguishes the halves with `GigaamVariant` — the
// `whisper` binder's existing ACCEPTED_ARCHS family precedent (`whisper` /
// `crisper-whisper` / `distil-whisper` / `kotoba-whisper` share one reader).
// What actually differs is the VOCABULARY (Russian / Central-Asian char space
// vs a 70+-language char space) and the provenance key: v3 stamps
// `vokra.provenance.upstream_hf = "ai-sage/GigaAM-v3"`, multilingual stamps
// `vokra.provenance.upstream_url = "github.com/salute-developers/GigaAM"`
// because its HF mirror is flagged 要 mirror URL 確認 in the ticket.
//
// REAL: strict `ACCEPTED_ARCHS` verification that refuses a foreign GGUF loudly,
// naming both the observed and expected tags and enumerating the sibling
// `category = "asr"` fleet (parakeet-ctc / canary / omniasr-ctc / whisper /
// distil-whisper / kotoba-whisper / sensevoicesmall) — the shared category tag
// alone cannot disambiguate them, only the arch tag can; a CROSSED-WIRES gate
// that refuses a GGUF whose arch names one variant while `vokra.model.name`
// names the other (they disagree about which vocabulary the CTC head emits
// over); a non-empty tensor-manifest gate; a by-name `dims()` lookup that NAMES
// an absent tensor instead of returning `None` for a caller to swallow; an
// optional `KEY_REQUIRED_TENSORS` (`Array<String>`) declaration that turns a
// truncated / mis-merged upload into a LOAD-time failure naming the first
// missing tensor; and `GigaamTopology` — a MEASURED structural probe that
// discovers every `<root>.layers.<i>.<leaf>` stack, counts its depth, and
// enforces contiguity (no index hole) plus uniformity (every layer carries the
// same leaf set as layer 0), refusing a violation by naming the EXACT absent
// tensor. Weight-license surfacing fail-closes to `LicenseClass::Unknown`.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より honest」):
// `Gigaam::transcribe` returns `VokraError::UnsupportedOp` naming three concrete
// blockers, all properties of the GGUF CONTRACT rather than of the kernel
// library — (i) the MISSING FRONT-END SPEC: neither converter stamps a
// `vokra.gigaam.*` / `vokra.frontend.*` chunk, so sample rate, mel-bin count,
// hop, window and normalisation convention are all unknown, and those differ
// silently between librosa / torchaudio / Kaldi; (ii) the MISSING ENCODER
// TENSOR-NAME MAPPING: both converters copy every tensor under its verbatim
// upstream state-dict key and both explicitly record real-weight binding as a
// follow-up "gated on the upstream tensor-name manifest fetch", so a best-guess
// mapping would emit a SHAPE-VALID but quietly wrong transcript rather than
// crash; (iii) the MISSING CTC VOCABULARY: no tokenizer / vocab chunk is
// embedded (contrast the Whisper converter's `vokra.tokenizer.model` U8 array),
// and GigaAM is char-wise CTC, so frame-argmax indices cannot be mapped to
// characters at all. The message states explicitly that the blockers are
// METADATA, not kernels — `vokra_ops::conformer`, `vokra_ops::ctc_decode_greedy`
// / `ctc_decode_beam` and `vokra_ops::mel` / `kaldi_fbank` all already exist —
// and points at the converter + sidecar to extend. No fabricated transcript is
// ever emitted (FR-EX-08). Deliberately NO `GigaamConfig::upstream_default()`:
// no in-repo primary source transcribes GigaAM's encoder geometry or front-end
// axes, so a default would be invented numbers wearing an authoritative face
// (CLAUDE.md ハルシネーション厳禁) — the same posture `ten_vad` / `firered_vad`
// take. `GigaamTopology` MEASURES the checkpoint instead of asserting anything.
//
// LICENSING: both converters stamp `mit` → `LicenseClass::Permissive` per the
// upstream `github.com/salute-developers/GigaAM/LICENSE`. This binder only
// SURFACES whatever class the GGUF carries. `docs/license-audit.md` §3.1
// sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign, and does not
// treat a converter default as a sign-off). Both tickets additionally flag open
// corpus-provenance questions (Sber-internal disclosure for v3; a Common Voice /
// MLS / VoxPopuli / FLEURS rights chain for the 70+-language variant) that are
// owner audit items, not runtime concerns.
//
// Cross-crate string handshake via duplicated `pub const ARCH_V3` /
// `ARCH_MULTILINGUAL` / `NAME_*` / `CATEGORY` / `UPSTREAM_*` (mirrors of the two
// converters' constants, preserving the layered convention `vokra-ops → nothing
// GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
// `vokra-convert → GGUF writer`). Note the arch tags use `_` while the names use
// `-`, AND that v3's arch carries the `sber_` vendor prefix while
// multilingual's does not; both asymmetries are load-bearing on the wire and are
// pinned separately by tests.
pub mod gigaam;

// Wave C1 (2026-08-15) — runtime binder for the `canary-1b-flash` converter
// arch (NVIDIA Canary-1B-Flash, CC-BY 4.0, 883M params, multitask ASR + AST +
// timestamps over English / German / French / Spanish). LIB.RS RULE: append at
// the END of the `pub mod` block with a Wave marker; do NOT alphabetize —
// rustfmt has reordered these before and broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/canary_1b_flash.rs`, coverage-audit
// 2026-08-03 Wave B) has stamped `vokra.model.arch = "canary-1b-flash"` /
// `vokra.model.name = "canary-1b-flash"` / `vokra.model.category = "asr"` /
// `vokra.provenance.upstream_hf = "nvidia/canary-1b-flash"`, but a
// workspace-wide grep proved NOTHING read that arch string back — a converted
// checkpoint was unloadable. This module is that consumer.
//
// STRUCTURE REUSE, NOT A THIRD SHAPE: the axes reuse `crate::canary`'s
// `CanaryEncoderConfig` / `CanaryDecoderConfig` / `CanaryHeadConfig` verbatim
// (re-exported), and `Canary1bFlashConfig::validate_for_forward` DELEGATES to
// the shared Canary-family validator instead of duplicating ~130 lines of
// shape algebra — the same posture `canary_qwen` takes when it re-exports the
// encoder types. Flash-specific state is only what genuinely differs.
//
// WHY A SEPARATE ARCH TAG: Canary-1B-Flash keeps Canary-1B-v2's 32-layer
// FastConformer encoder verbatim but distils the Transformer AED decoder from
// 8 layers to **4** (Canary-1B-v1: 24) — the axis behind the model card's
// "1000+ RTFx" claim. A loader that walks an 8-block decoder manifest against a
// 4-block payload does not crash, it silently mis-reads, so `canary` /
// `canary-1b-flash` / `canary-qwen` (Qwen LLM decoder, soft-prompt prefix) stay
// three distinct tags and the arch gate is strict (FR-EX-08).
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named and the Canary / NeMo-ASR neighbourhood enumerated
// (`canary` / `canary-qwen` / `parakeet-ctc` / `parakeet-tdt` / `whisper` /
// `voxtral` / `kyutai-stt`); primary-source axis transcription (32 encoder
// layers + 4 decoder layers from the model card; d_model / lm_dec_hidden /
// max_sequence_length = 1024 attested for `canary-1b-flash` BY NAME in the
// `fast-conformer_aed.yaml` variant table; the remaining axes from that same
// family reference); a forward-compatible OPTIONAL `vokra.canary_1b_flash.*`
// axis-override group (the current converter stamps NONE of it, so a real
// artifact resolves to `ConfigSource::FamilyAnchored` — absence is normal,
// but a PRESENT key of the wrong dtype fails loud rather than being silently
// ignored); a tensor manifest over the verbatim upstream safetensors names the
// converter passes through, with a non-empty gate plus `require_tensor` /
// `require_tensor_dims` lookups that NAME the missing tensor or BOTH the
// expected and actual dims; and weight-license + FR-MD-09 attribution
// surfacing that fail-closes to `LicenseClass::Unknown`.
//
// DELIBERATELY NOT TRANSCRIBED: `head.vocab_size` / `pad` / `bos` / `eos` stay
// `0` sentinels that `validate_for_forward` REFUSES. No primary source states
// them (the card says only "the 4-language subset of the unified Canary
// SentencePiece"), and copying Canary-1B-v2's 25-language 16384 would be
// fabrication across a different tokenizer.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `transcribe` / `transcribe_with_task` return
// `VokraError::UnsupportedOp` naming four blockers — (i) NO TENSOR-NAME
// MANIFEST: the converter copies every float tensor under its verbatim
// upstream name and nothing in-repo transcribes NeMo's `EncDecMultiTaskModel`
// state_dict naming, so walking guessed names into typed slots would bind
// shape-valid garbage; (ii) NO TOKENIZER: the SentencePiece model, its width
// and the pad/bos/eos/`<taskname>` ids live inside the `.nemo` tarball;
// (iii) the `0` head sentinel that follows from (ii), so no logits array can
// even be shaped; (iv) the 4-layer AED decoder STEP is unwired — a gap SHARED
// with `crate::canary`, not specific to Flash. The message names the
// primitives that DO exist and that the follow-up wave composes
// (`vokra_ops::waveform_frontend`, `vokra_ops::conformer` with
// `Stacking { factor: 8 }`, `vokra_core::decode::beam_search`). No fabricated
// token ids are ever emitted (FR-EX-08).
//
// LICENSING: the converter stamps `cc-by-4.0` -> `AttributionRequired`, so the
// FR-MD-09 attribution surface activates and `Canary1bFlashAsr::attribution`
// returns the stamped text a downstream must display. This binder only
// SURFACES whatever class the artifact carries and fail-closes to `Unknown`.
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
//
// Cross-crate string handshake via duplicated `pub const ARCH =
// "canary-1b-flash"` / `NAME` / `CATEGORY` / `UPSTREAM_HF` / `DEFAULT_LICENSE`
// (mirror of the converter's constants, preserving the layered convention
// `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models
// → GGUF binder`, `vokra-convert → GGUF writer`).
pub mod canary_1b_flash;

// Wave C2 (2026-08-15) — runtime binder for the `eat` converter arch (EAT,
// `cwx-worst-one/EAT`, MIT — a self-supervised audio encoder trained with a
// bootstrap / self-distillation objective and inverse block masking over an
// utterance-level Transformer, AudioSet-2M pre-training, ~86M params at the
// `eat-base` size point, Chen et al. 2024 arXiv:2401.03497). LIB.RS RULE:
// append at the END of the `pub mod` block with a Wave marker; do NOT
// alphabetize — rustfmt has reordered these before and broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/eat.rs`, SSL audio-encoder wave
// 2026-08-13) stamps `vokra.model.arch = "eat"` / `vokra.model.name =
// "eat-base"` / `vokra.model.category = "audio-embedding"` /
// `vokra.provenance.upstream_url = "github.com/cwx-worst-one/EAT"`, but a
// workspace-wide grep proved NOTHING read that arch string back — a converted
// checkpoint was unloadable. This module is that consumer.
//
// FEATURE EXTRACTOR, NOT AN END-TASK MODEL: EAT emits representations, not
// labels. The upstream release ships its downstream task heads (AudioSet
// tagging, ESC-50, SPC-2) as SEPARATE fine-tunes, so this binder exposes only
// `Eat::encode` (per-patch hidden states) and `Eat::embed_utterance`
// (utterance-level embedding). No classification head is invented, because the
// pre-training checkpoint this converter targets does not contain one.
//
// WHY A SEPARATE ARCH TAG: EAT's utterance-level Transformer trained with
// inverse block masking is a distinct topology axis from every sibling SSL
// audio encoder — `beats` (iterative acoustic-tokenizer SSL), `dasheng`
// (universal MAE), `atst` (teacher-student patchout), `m2d` (masked-modeling
// duo), `mert` / `muq` (music-domain SSL), `ast` (supervised, not
// self-supervised), `hubert` (masked cluster prediction over raw waveform).
// Silently aliasing would let dispatch bind e.g. an MAE decoder over an
// utterance-level checkpoint and produce shape-valid garbage instead of a loud
// error, so the arch gate is strict (FR-EX-08).
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named and the whole SSL-encoder neighbourhood enumerated; a
// `vokra.model.category` cross-check where PRESENT-but-wrong fails loud while
// absent is tolerated (hand-assembled fixtures); a tensor manifest over the
// verbatim upstream state-dict names the converter passes through, with a
// non-empty gate plus `require_tensor` / `require_tensor_dims` lookups that
// NAME the missing tensor or BOTH the expected and actual dims; and
// pure-observation structure discovery (`observed_block_count` /
// `has_patch_embed` / `count_with_prefix`) derived from what is actually on
// disk rather than from a transcribed constant — PLUS the full `EatConfig`
// axis group and the real ViT weight binding, both described in the two
// paragraphs below (this list is not exhaustive on its own).
//
// THE `vokra.eat.*` TOPOLOGY GROUP IS STAMPED AND IS READ STRICTLY: an
// earlier landing of this header recorded that the converter "stamps NO axes
// at all". That was true when written and is NO LONGER TRUE — the converter
// now stamps a 38-key `vokra.eat.*` group covering the ViT-B backbone, the
// patch grid, the pre-training decoder and the complete Kaldi-fbank argument
// set. `EatConfig::from_gguf` reads every one of them in the `vokra.wavlm.*`
// posture: a missing key is a loud `ModelLoad` naming it, never a fallback to
// a primary-source constant, because a silent default would let a mismatched
// artifact bind a topology it does not carry (FR-EX-08). `EatConfig` also
// maps those axes onto `vokra_ops::vit::ViTAttrs`, DERIVING the patch stride
// and checking it against the independently stamped grid rather than assuming
// it. `observed_block_count` still returns `None` (meaning "unknown", never
// "zero layers") for a checkpoint whose state_dict uses a prefix this repo has
// not transcribed.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): the PCM-in surfaces `encode` / `embed_utterance` return
// `VokraError::UnsupportedOp` naming three blockers — (i) NO VERIFIED
// TENSOR-NAME MANIFEST: the converter passes upstream names through verbatim
// and nothing in-repo transcribes them (EAT descends from fairseq data2vec2,
// whose modality-specific parameters live under a `modality_encoders.*` tree),
// so this PCM-in surface has no manifest to reach for and will not guess one;
// (ii) THE KALDI-FBANK WINDOW: the front-end arguments ARE stamped in full and
// `vokra.eat.fbank_window_type` is `hanning`, but the checked
// `KaldiFbankWindow` selector currently exposes Povey and Hamming, not
// Hanning, so every feature would desync — the stamp makes this detectable;
// (iii) NORM ORDER UNRECONCILED: `vokra.eat.layer_norm_first` is stamped as a
// transcribed config value, explicitly NOT as an assertion about where the
// norms sit, while `vokra_ops::vit::ViTEncoder` is pre-norm BY CONSTRUCTION.
// `embed_utterance` adds (iv) the utterance-level READ-OUT convention —
// `vokra_ops::vit::ViTPooling` can express either form and the width IS
// stamped, but WHICH form EAT trained, and at which index the CLS token sits,
// are not transcribed in-repo. The message also reports the manifest facts
// actually observed on disk. No fabricated hidden states or embeddings are
// ever emitted (FR-EX-08).
//
// THE ViT PRIMITIVES ARE REAL AND ARE BOUND: the 2-D patch embedding and the
// pre-norm Transformer encoder this header used to list as missing now exist
// as `vokra_ops::vit::vit_patch_embed` / `vokra_ops::vit::ViTEncoder`, and
// `Eat::bind_vit_weights` / `Eat::bind_vit_encoder` decode a real
// `ViTWeights` out of the GGUF through shape-gated `require_tensor` /
// `require_tensor_dims` lookups. A caller that can DEFEND a tensor naming
// supplies it as `EatVitTensorNames` and gets a real forward on that path.
//
// LICENSING: upstream reports `spdx_id: MIT` via the GitHub license API
// (converter task input 2026-08-13), so the converter stamps `mit` ->
// `LicenseClass::Permissive`. This binder only SURFACES whatever class the
// artifact carries and fail-closes to `Unknown` when the stamp is absent.
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "eat"` /
// `NAME` / `CATEGORY` / `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops →
// nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`).
pub mod eat;

// Wave C2 (2026-08-15) — runtime binder for the `atst` converter arch (ATST,
// "Audio Teacher-Student Transformer", `Audio-WestlakeU/audiossl/tree/main/
// audiossl/methods/atst`, code MIT / **weight CC-BY-4.0**; a self-supervised
// audio encoder trained with a BYOL-style EMA teacher + student-patchout
// objective over a log-mel patch grid, ~86M params at the `atst-base` size
// point). LIB.RS RULE: append at the END of the `pub mod` block with a Wave
// marker; do NOT alphabetize — rustfmt has reordered these before and broken a
// commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/atst.rs`, SSL-encoder wave 2026-08-13) has
// been stamping `vokra.model.arch = "atst"` / `vokra.model.name = "atst-base"`
// / `vokra.model.category = "audio-embedding"` / `vokra.provenance.upstream_url`
// (ATST is not on HuggingFace, so there is no `upstream_hf` counterpart), but a
// workspace-wide grep proved NOTHING read that arch string back — a converted
// checkpoint was unloadable. This module is that consumer.
//
// FEATURE EXTRACTOR, NOT AN END-TASK MODEL: the checkpoint carries NO task
// head. Downstream sound-event-detection / audio-tagging / speaker heads are
// trained separately by consumers, so this binder exposes hidden states
// (`encode`) and the utterance-level pooled embedding (`embed`) and invents no
// classifier and no class-label list.
//
// WHY A SEPARATE ARCH TAG: every sibling SSL audio/music-embedding encoder
// differs in the pre-training objective that shapes the topology — `beats`
// (iterative acoustic tokenizer + MAM), `eat` (utterance-level MAE + inverse
// block masking), `dasheng` (universal MAE), `m2d` (masked modelling duo),
// `maest` (AST backbone / Discogs tagger), `mert` (HuBERT-derived MPM), `muq`
// (Mel-RVQ + BEATs teacher), `yamnet` (supervised CNN, not SSL) — and the
// wav2vec2 lineage (`hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec`)
// sits on a raw-waveform 1-D conv stem rather than a log-mel patch grid.
// Binding any of them over an ATST payload does not crash, it silently
// mis-reads, so the arch gate is strict (FR-EX-08).
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named and the whole neighbourhood enumerated; a tensor
// manifest over the verbatim upstream `state_dict` names the converter passes
// through, with a non-empty gate plus `require_tensor` / `require_tensor_dims`
// lookups that NAME the missing tensor or BOTH the expected and actual dims;
// `name` / `category` / `upstream_url` metadata surfacing (`vokra.model.name`
// is SURFACED, not gated — the frame-level `atst-frame` TASLP-2023 sibling
// shares this arch under its own name); the BYOL duo diagnostic `AtstBranch` +
// `branch_tensor_count`; weight-licence + FR-MD-09 attribution surfacing that
// fail-closes to `LicenseClass::Unknown`; and the compliance-gated
// `from_gguf_with_policy` / `from_path` entry points.
//
// TWO BLOCKERS CLOSED ON 2026-08-15: this header previously named FOUR
// blockers. Two of them are now facts about the world rather than about this
// repository. (a) The `vokra.atst.*` AXIS CHUNK GROUP now EXISTS — the
// converter stamps the full topology group (Transformer width / depth /
// heads, the patch grid and position table, and the whole log-mel front-end),
// each value transcribed from the upstream source tree with its file and line
// recorded, and `AtstConfig` is the strict consumer. (b) The ViT-STYLE
// ENCODER PRIMITIVE now EXISTS — `vokra_ops::vit` landed with 2-D patch
// embedding over a mel plane, learned prepended tokens, an additive
// positional table, a pre-norm Transformer stack, a final norm and pooling;
// `AtstConfig::vit_tensor_shapes` derives the exact dims every ViT weight
// must carry and `Atst::verify_vit_tensor_shapes` walks a caller-supplied
// `AtstVitTensorNames` through `require_tensor_dims`.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `encode` / `embed` return `VokraError::UnsupportedOp` naming the
// TWO blockers that remain, both facts about a real checkpoint that no file in
// this repository records — (i) TEACHER/STUDENT BRANCH SELECTION IS
// UNRESOLVED: a BYOL-style EMA checkpoint carries BOTH branches and picking
// the wrong one yields a shape-valid but numerically different embedding, so
// the branch upstream's own inference entry point uses must be read off the
// upstream tree first (`AtstBranch` / `Atst::branch_tensor_count` exist today
// only as diagnostics over what is on disk; they gate nothing); (ii) NO
// VERIFIED TENSOR-NAME MANIFEST: the converter copies every float tensor under
// its verbatim upstream `state_dict` name and nothing in-repo transcribes
// ATST's naming — the real key chain is recorded as running
// `ATSTLightningModule.model` -> `ATST.student` / `.teacher` ->
// `MultiCropWrapper` -> the `AST` encoder, so the prefix is at least
// `model.student.`, but no checkpoint key listing has been read. This is why
// `AtstVitTensorNames` deliberately has NO `Default` and no `atst_base()`
// constructor. Reading one real key listing resolves BOTH remaining blockers
// at once, since the branch prefix and the tensor names are the same listing.
// No fabricated hidden states or embeddings are ever emitted (FR-EX-08).
//
// LICENSING: the upstream LICENSE file SPLITS the tiers — code `mit`,
// pretrained checkpoints `cc-by-4.0` — and `vokra.provenance.weight_license`
// tracks the WEIGHT, so the converter stamps `cc-by-4.0` ->
// `AttributionRequired`. That is commercially permitted (loads under
// `CompliancePolicy::strict`) but carries a display obligation, so
// `Atst::attribution` surfaces the FR-MD-09 text rather than burying it. This
// binder only SURFACES whatever class the artifact carries and fail-closes to
// `Unknown`. `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign, and does
// not treat a converter default as a sign-off).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "atst"` /
// `NAME` / `CATEGORY` / `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops →
// nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`).
pub mod atst;

// Wave C2 (2026-08-15) — runtime binder for the `m2d` converter arch (M2D,
// "Masked Modeling Duo", `nttcslab/m2d`, license Unknown / fail-closed — a
// self-supervised general-audio encoder trained by a DUO of networks, an
// `online` and a `target`, whose objective encourages BOTH to model the input
// rather than only reconstructing masked patches; Niizumi et al., ICASSP 2023,
// arXiv:2210.14648, plus the TASLP 2024 sound-event-detection / speech
// extension; ~86M-parameter class base variant). LIB.RS RULE: append at the END
// of the `pub mod` block with a Wave marker; do NOT alphabetize — rustfmt has
// reordered these before and broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/m2d.rs`, SSL audio-encoder wave 2026-08-13)
// stamps `vokra.model.arch = "m2d"` / `vokra.model.name = "m2d-base"` /
// `vokra.model.category = "audio-embedding"` / `vokra.provenance.upstream_url =
// "github.com/nttcslab/m2d"`, but a workspace-wide grep proved NOTHING read
// that arch string back — a converted checkpoint was unloadable. This module is
// that consumer.
//
// FEATURE EXTRACTOR, NOT AN END-TASK MODEL: the checkpoint carries an encoder,
// not a classifier (upstream ships sound-event-detection / audio-tagging /
// speaker heads separately, as fine-tuning recipes). The surface is therefore
// `encode` (a `[num_frames, hidden_size]` hidden-state block) and `embed` (an
// utterance-level pooled embedding). No classification head is invented.
//
// WHY A SEPARATE ARCH TAG: M2D's masked-modeling-DUO objective leaves TWO
// parallel branches in the state dict, where every sibling SSL encoder
// (`beats` iterative tokenizer / `eat` inverse block masking / `atst`
// teacher-student patchout / `dasheng` single-branch MAE / `mert` / `muq`) has
// one. A single-branch loader pointed at an M2D checkpoint does not crash — it
// silently binds one branch and returns a plausible-but-wrong embedding, so the
// arch gate is strict and its error enumerates the whole neighbourhood
// (FR-EX-08).
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named; the REQUIRED eight-key `vokra.m2d.*` axis group (the
// converter stamps all eight, each transcribed from a primary source it cites
// line by line, so `M2dConfig::from_gguf` demands every one and fails with a
// loud `ModelLoad` NAMING the absent key — deliberately NO primary-source
// constant fallback, since the producer stamps these and a silent default
// would let a mismatched artifact, e.g. the separate 32 kHz M2D identity, bind
// as the canonical one); `M2dConfig::vit_attrs`, the mapping onto
// `vokra_ops::vit::ViTAttrs`; a tensor manifest over the verbatim upstream
// `state_dict` names the converter passes through, with a non-empty gate plus
// `require_tensor` / `require_tensor_dims` lookups that NAME the missing tensor
// or BOTH the expected and actual dims, and `branch_triage` which OBSERVES how
// the bound manifest is prefixed; and weight-license surfacing that fail-closes
// to `LicenseClass::Unknown`.
//
// NOT CARRIED BY THE ARTIFACT: `ViTAttrs` has 12 axes and the stamped group
// supplies 5 (embed_dim, depth, n_heads, patch_h, patch_w). The other 7
// (`UNSTAMPED_VIT_AXES`: stride_h/w, n_prepended_tokens, mlp_ratio,
// layer_norm_eps, gelu, pos_embed_policy) are carried by NO `vokra.m2d.*` key,
// so `vit_attrs` takes them from the CALLER as an `M2dUnstampedAxes` rather
// than hard-coding a constant the artifact cannot contradict (CLAUDE.md
// 「ハルシネーション厳禁」).
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `encode` / `embed` return `VokraError::UnsupportedOp` naming THREE
// remaining blockers — (1) UNVERIFIED TENSOR-NAME MANIFEST (nothing in-repo
// transcribes M2D's `state_dict` naming; a real checkpoint also settles whether
// the release keeps the `online.`/`target.` prefixes at all and whether qkv is
// fused); (2) UNSTAMPED ViT AXES; (3) NO MEL FRONT-END BINDING (the ViT forward
// consumes a log-mel plane, and n_fft/hop/window/f_min/f_max are not stamped) —
// plus, for `embed`, the unresolved POOLING RECIPE. The message states
// OUTRIGHT which blockers are already RESOLVED (the axis group IS stamped,
// branch selection rides `vokra.m2d.inference_branch`, and `vokra_ops::vit` now
// supplies the ViT encoder that `conformer`/`zipformer`/`ebranchformer` — 1-D
// ASR encoders — could not stand in for) so nobody re-reports them. All three
// primary sources are cited. No fabricated hidden states or embeddings are ever
// emitted (FR-EX-08).
//
// LICENSING: upstream's LICENSE is a PDF that GitHub's classifier cannot read
// (`spdx_id: NOASSERTION`), so the converter default is `unknown` ->
// `LicenseClass::Unknown` — `requires_research_flag() == true` and
// `redistributable() == false`, i.e. fail-closed at the M2-13 gate and refused
// at publish. Clearing that is an OWNER action (read `LICENSE.pdf`, confirm the
// SPDX tier, re-convert with `--license <spdx>`); `docs/license-audit.md` §3.1
// sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]` — CC does NOT sign).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "m2d"` / `NAME`
// / `CATEGORY` / `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops → nothing
// GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
// `vokra-convert → GGUF writer`).
pub mod m2d;

// Wave C2 (2026-08-15) — runtime binder for the `w2v-bert-2` converter arch
// (Meta / SeamlessM4T-v2 w2v-BERT 2.0 speech encoder, `facebook/w2v-bert-2.0`,
// MIT, ~580M params). LIB.RS RULE: append at the END of the `pub mod` block
// with a Wave marker; do NOT alphabetize — rustfmt has reordered these before
// and broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/w2v_bert_2.rs`) stamps
// `vokra.model.arch = "w2v-bert-2"` / `vokra.model.name = "w2v-bert-2.0"` /
// `vokra.model.category = "asr"` /
// `vokra.provenance.upstream_hf = "facebook/w2v-bert-2.0"`, but a
// workspace-wide grep proved NOTHING read that arch string back — weights
// converted and then nothing could load them. This module is that consumer.
//
// FEATURE EXTRACTOR, NOT AN END-TASK MODEL: upstream ships
// `architectures: ["Wav2Vec2BertModel"]` — no task head — so the surface is
// `encode` (a hidden-state sequence) plus the stem, and NO classification head
// is invented on top of a checkpoint that does not contain one.
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named and two confusable neighbourhoods enumerated — the SSL
// siblings (`hubert` / `wav2vec2_ctc` / `wavlm_sv` / `emotion2vec`, all
// vanilla-Transformer bodies where w2v-BERT alone has a CONFORMER body) and
// the two composites that EMBED w2v-BERT as an internal subgraph (`unity-2`
// SeamlessM4T-v2 speech encoder, `vieneu-tts` speaker encoder), whose
// artifacts nest these tensors under a composite prefix; topology recovered
// from the TENSOR SHAPES ON DISK rather than from metadata, because this
// converter is a BF16 pass-through skeleton that stamps no
// `vokra.w2v_bert_2.*` chunk group at all — including `num_attention_heads`,
// recoverable only because `self_attn.distance_embedding.weight` is
// `[num_positions, head_size]` (Q/K/V/out are all `[hidden, hidden]` and carry
// no head geometry, so without that table the binder reports `None` instead of
// guessing); a per-layer required-tensor sweep over the verbatim upstream
// `state_dict` names that NAMES a missing tensor, deliberately NOT requiring
// the three `conv_module` conv biases (upstream builds them `bias=False`) and
// treating `distance_embedding` as optional (relative_key only); a contiguity
// gate on the `encoder.layers.{i}` index range so a hole cannot silently
// shorten the stack; and a genuinely REAL `feature_projection` forward
// (`project_features` = LayerNorm(160) -> Linear(160, 1024)) pinned by a
// hand-computed numeric unit test rather than by the implementation.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): `encode` returns `VokraError::UnsupportedOp` naming four concrete
// divergences between upstream `Wav2Vec2BertEncoderLayer` and the SHARED
// `vokra_ops::conformer` primitive — which was read first per the wiring
// brief, and which does cover the macaron layer skeleton exactly: (i)
// `position_embeddings_type = "relative_key"` Shaw-style `distance_embedding`
// attention bias vs a `PositionEncoding` exposing only `None` / `Rope` (that
// module's own docstring records the relative path as omitted); (ii)
// upstream's CAUSAL left-only depthwise padding `(kernel_size - 1, 0)` vs the
// primitive's symmetric same-padding; (iii) upstream's bias-FREE conv module
// vs `ConformerConvWeights` requiring three biases; (iv) upstream's
// PRE-projection stem LayerNorm vs `StackingNorm`'s post-projection one.
// Composing the stack with the primitive as-is would emit shape-valid but
// numerically WRONG hidden states — precisely the silent misroute FR-EX-08
// exists to prevent. Closing this is additive work on `vokra_ops::conformer`
// (shared with the parakeet / canary fleet), which is why it is deliberately
// NOT done from inside this binder. The four gaps are pinned as data in
// `CONFORMER_PRIMITIVE_GAPS` so the message and the tests must move together.
//
// LICENSING: the converter stamps `mit` -> `Permissive` (T1 Commercial). This
// binder only SURFACES whatever class the artifact carries and fail-closes to
// `LicenseClass::Unknown`. `docs/license-audit.md` §3.1 sign-off stays BLANK
// (owner-only per `[[feedback-license-signoff-primary-source]]`).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "w2v-bert-2"`
// / `NAME` / `CATEGORY` / `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` (mirror of
// the converter's constants, preserving the layered convention `vokra-ops →
// nothing GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
// binder`, `vokra-convert → GGUF writer`).
pub mod w2v_bert2;

// Wave C2 (2026-08-15) — MAEST music-tagging SSL binder: the runtime consumer
// for the `maest` converter arch ("Music Audio Efficient Spectrogram
// Transformer", `mtg-upf/discogs-maest-30s-pw-129e`, **cc-by-nc-sa-4.0**;
// Alonso-Jiménez et al. ISMIR 2023, arXiv:2309.16418; ~87M F32 params).
// LIB.RS RULE: append at the END of the `pub mod` block with a Wave marker; do
// NOT alphabetize — rustfmt has reordered these before and broken a commit.
//
// Closes a real read-side gap: the converter
// (`crates/vokra-convert/src/models/maest.rs`, SSL audio-encoder wave
// 2026-08-13) stamps `vokra.model.arch = "maest"` / `vokra.model.name =
// "maest-30s-pw-129e"` / `vokra.model.category = "music-embedding"` /
// `vokra.provenance.upstream_hf = "mtg-upf/discogs-maest-30s-pw-129e"`, but a
// workspace-wide grep proved NOTHING read that arch string back — weights
// converted and then nothing could load them. This module is that consumer.
//
// THE MUSIC-DOMAIN MEMBER OF THE SSL FLEET: where `atst` / `eat` / `m2d` /
// `dasheng` are general-audio encoders that ship no task head, MAEST is
// pretrained on the MTG Discogs4All MUSIC-tagger dataset and upstream's HF
// `config` declares `architectures: ["ASTForAudioClassification"]` — i.e. the
// release DOES carry a tagging head. The surface is therefore `encode`
// (per-patch hidden states) + `embed` (pooled clip embedding) + `tag` (Discogs
// tag logits).
//
// WHY A SEPARATE ARCH TAG: MAEST is built on the very same Audio Spectrogram
// Transformer backbone as the sibling `ast` arch — but `ast` is SUPERVISED,
// fine-tuned on AudioSet, general-audio, and carries a different taxonomy.
// Backbone identity is not topology identity, and a silent bind between the two
// would look plausible right up until the numbers and the label set are wrong
// (FR-EX-08), so the arch gate is strict and its error enumerates the whole
// neighbourhood including that pair.
//
// LABEL TAXONOMY — COUNT HAS TWO WITNESSES, NAMES HAVE NONE: the converter
// stamps the label COUNT (`vokra.maest.num_labels`, from `config.json`'s
// `id2label` cardinality) but NO label list. This module still hardcodes no
// taxonomy constant of its own: `Maest::label_count` scans the tensors actually
// on disk under `TAG_HEAD_PREFIX` and reports the head projection's leading dim
// (`nn.Linear` weight is `[out_features, in_features]`) — `None` for a
// bare-encoder export or an ambiguous layout, never a fallback number. Keeping
// the stamp and the payload independent is what lets the head binding
// cross-check them and refuse an artifact where they disagree. A unit test pins
// the read-not-guessed property by asserting two synthetic artifacts with
// different head widths report different counts, which a hardcoded taxonomy
// size could not satisfy (CLAUDE.md「ハルシネーション厳禁」). The label NAMES
// remain unrecoverable from the artifact; that does not block `tag_mel`, whose
// return type is logits, but it does mean logit index `i` cannot be mapped onto
// a Discogs genre / mood / instrument / era string from the GGUF alone.
//
// REAL: strict `vokra.model.arch` verification refusing a foreign GGUF loudly
// with BOTH tags named; strict `MaestConfig` reading of the `vokra.maest.*`
// topology + front-end axis group (every stamped key required, a missing one a
// loud `ModelLoad` naming it, no primary-source constant fallback); the mapping
// of those axes onto `vokra_ops::vit::ViTAttrs`; a tensor manifest over the
// verbatim upstream `state_dict` names the converter passes through, with a
// non-empty gate plus `require_tensor` / `require_tensor_dims` lookups that NAME
// the missing tensor or BOTH the expected and actual dims, and
// `detect_tensor_prefix` probing which `state_dict` prefix the artifact actually
// uses; `Maest::encoder` weight binding and the `MaestEncoder` FORWARD over a
// log-mel plane (`encode_mel` / `embed_mel` / `tag_mel`); tag-head discovery
// from disk; metadata surfacing (`name` / `category` / `upstream_hf` /
// `model_id` / `source`); and weight-license + FR-MD-09 attribution surfacing
// that fail-closes to `LicenseClass::Unknown`.
//
// The forward is transcribed from the HuggingFace AST modelling file at
// `v4.34.0` — the tag the checkpoint's own config names
// (`transformers_version: "4.34.0.dev0"`) — which supplies the `state_dict`
// names, the pre-norm block ordering, the `(cls, distillation, patches)`
// concatenation order, the `[num_mel_bins, max_length]` plane orientation, and
// the `(sequence_output[:, 0] + sequence_output[:, 1]) / 2` pooling rule.
// NOTE: no numerical parity run exists — the weights are gated CC-BY-NC-SA 4.0
// and no fixture is committed — so the tests assert shape, finiteness and
// determinism only, never an expected numeric value.
//
// LOUD-PARTIAL (CLAUDE.md 教訓 (a)「loud-partial は fake-complete より
// honest」): the PCM-in surfaces `encode` / `embed` / `tag` return
// `VokraError::UnsupportedOp` for ONE remaining gap — the STFT FRAMING /
// CENTERING convention of the log-mel front end. The converter deliberately
// stamps no `center` / `pad_mode` and writes no `vokra.frontend.*` group
// because no primary source it reached states them, and choosing wrongly shifts
// every frame by half a window: shape-valid, numerically wrong, silent. Every
// OTHER front-end axis IS stamped and is echoed in the error message. No
// fabricated hidden states, embeddings or logits are ever emitted (FR-EX-08).
//
// LICENSING: the converter stamps `cc-by-nc-sa-4.0` ->
// `LicenseClass::NonCommercialShareAlike` = T4 tier + ShareAlike cascade, whose
// `requires_research_flag()` is true — so a correctly stamped artifact is
// REFUSED under `CompliancePolicy::strict()` and loads only with an explicit
// research opt-in. That refusal is the fail-closed default working as intended.
// Three obligations cascade: non-commercial, share-alike on any downstream
// distribution, and attribution. `docs/license-audit.md` §3.1 sign-off stays
// BLANK (owner-only per `[[feedback-license-signoff-primary-source]]` — CC does
// NOT sign).
//
// Cross-crate string handshake via duplicated `pub const ARCH = "maest"` /
// `NAME` / `CATEGORY` / `UPSTREAM_HF` / `DEFAULT_LICENSE_SPDX` (mirror of the
// converter's constants, preserving the layered convention `vokra-ops → nothing
// GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
// `vokra-convert → GGUF writer`).
pub mod maest;
// Wave D (2026-08-15) — DiffSinger singing voice synthesis (SVS) runtime
// binder: the first singing-voice entry in the catalogue. Real config +
// `from_gguf` (strict 19-axis `vokra.diffsinger.*` parse + arch verify +
// tensor gate + license surface) + a real `Score` input type (phonemes +
// per-note MIDI pitch + durations) whose score-to-frame expansion runs
// through the landed `vokra_ops::length_conditioning`; the acoustic
// forward is loud-partial pending the FFT-block encoder + LynxNet2
// shallow-diffusion denoiser. Emits a mel and hands off to the landed
// `hifigan` / `bigvgan` / `vocos` vocoder binders rather than embedding
// one. **SVS != SVC**: no source recording in the signal path, so this is
// not an ELVIS Act voice-clone trigger and must not be relocated to
// `vokra-voiceclone-experimental` (see the module docstring).
pub mod diffsinger;
// Wave D (2026-08-15) — AudioSR audio super-resolution / bandwidth-extension
// runtime binder (`haoheliu/versatile_audio_super_resolution`, MIT,
// arXiv:2309.07314) for the `audiosr` converter arch. **Opens a brand-new
// capability category**: Vokra had no audio super-resolution model before
// this landing (category tag `super-resolution`, deliberately distinct from
// the `enhancement` cohort which removes additive noise within a fixed
// bandwidth rather than synthesising new spectral content above the cutoff).
//
// REAL in this landing: strict `vokra.model.arch` verification naming both
// the expected and actual tag; basic/speech variant discrimination; a STRICT
// `vokra.audiosr.*` topology read that names the first missing axis (no
// silent defaults); a loud tensor accessor that names a missing tensor; the
// 256-band mel filterbank built through `vokra_ops::mel::MelFilterbank` from
// the transcribed axes; and the cosine-schedule cumulative-alpha diffusion
// table built through `vokra_ops::ddpm_sampler` from the transcribed
// `timesteps: 1000` + `beta_schedule: "cosine"`.
//
// LOUD-PARTIAL: `super_resolve()` returns `UnsupportedOp` naming (a) the 2-D
// latent-diffusion U-Net forward (GREENFIELD — no equivalent primitive in
// `vokra_ops`), (b) the mel-VAE encode/decode walk (`vokra_ops::vae_continuous`
// is an ANCHOR, walk not pinned), (c) the vocoder walk (`vokra_ops::hifigan`
// is an ANCHOR; upstream `audiosr/utils.py` carries no vocoder config block,
// verified absent), and (d) the absent tensor-name manifest (upstream ships
// `pytorch_model.bin` pickle, not downloaded). No fabricated waveform is ever
// emitted (FR-EX-08). Arch tag `audiosr` is distinct from `audioldm2` — same
// author and latent-diffusion family, opposite task, incompatible shapes.
pub mod audiosr;
// Wave D (2026-08-15) — CT-Transformer punctuation restoration
// (`funasr/ct-punc`): the FIRST `punctuation`-category model in the tree.
// Pairs with the inverse-text-normalization stage: every ASR binder in this
// crate emits unpunctuated text, and those two together are what turn it
// into a readable transcript.
//
// REAL FORWARD, not a loud-partial — every primitive it needs already
// exists, so `CtPunc::logits` actually runs the model: embedding lookup ->
// `* sqrt(att_unit)` -> `SinusoidalPositionEncoder` (1-BASED positions, and
// a `[sin block | cos block]` layout rather than interleaved pairs) -> N
// pre-norm `EncoderLayerSANM` blocks -> `encoder.after_norm` -> a linear
// head emitting one punctuation label per token. Each block's attention is
// `MultiHeadedAttentionSANM`: ONE fused `linear_q_k_v` projection plus a
// PARALLEL depthwise-Conv1d FSMN memory branch over `v` whose output is
// ADDED to the attention output — which is exactly why this cannot reuse
// `vokra-bert`'s `BertBaseEncoder`, and why `ARCH = "ct_punc"` must not
// alias `bert_base` / `deberta_v2` / `deberta_v3` / `sensevoicesmall` /
// `fsmn-vad` (FR-EX-08).
//
// The punctuation label inventory is READ from the artifact's
// `vokra.ct_punc.punc_list` Array<String> chunk, never hardcoded — a
// checkpoint with a different label set means different head columns.
//
// DELIBERATELY OUT OF SCOPE (documented, not faked): tokenisation (upstream
// fronts a 471067-entry `CharTokenizer` with jieba word segmentation, so the
// forward takes token IDS), upstream's cross-window `split_size = 20` cache
// policy, and padded batches. Real-checkpoint numeric parity needs the
// 1.05 GiB `model.pt` and is an owner task; the in-module tests pin the
// structure analytically instead (zeroing the block LayerNorms provably
// collapses the encoder to the identity, making the expected logits
// hand-computable).
//
// LICENSING — note the FunASR code / weight split. FunASR's CODE is MIT
// (`github.com/modelscope/FunASR/LICENSE` = `MIT License / Copyright (c)
// 2025 FunASR`), but its weight releases are not uniformly MIT: the sibling
// `sensevoicesmall_runtime` binder correctly fail-closes to
// `LicenseClass::Unknown` because `FunAudioLLM/SenseVoiceSmall` ships the
// bespoke `MODEL_LICENSE` instead of an SPDX id. THIS checkpoint's weights
// are `apache-2.0` per the `funasr/ct-punc` model-card front-matter AND the
// HF model API `cardData.license` (both read 2026-08-15), and the repo
// carries no `MODEL_LICENSE` file. `docs/license-audit.md` §3.1 sign-off
// stays BLANK (owner-only per `[[feedback-license-signoff-primary-source]]`).
pub mod ct_punc;
// Wave D (2026-08-15) — WeTextProcessing (`wenet-e2e/WeTextProcessing`,
// Apache-2.0) inverse text normalization / text normalization grammar bundles:
// the GGUF binder for the FIRST `text-normalization` category entry, and the
// first weightless binder in the tree.
//
// Every ASR model in this crate emits normalized, unpunctuated text — an
// utterance spoken as "one hundred fourteen thousand five" comes back as those
// words, while a production transcript needs "114005". ITN is the missing back
// half of the ASR pipeline; the same machine over the other grammar pair is TN,
// the front half of a TTS pipeline.
//
// The binder is deliberately thin, because the layering says so: `vokra-core`
// reads GGUF, `vokra-models` binds it, `vokra-ops` owns the GGUF-unaware
// operator. So `WeTextProcessing::from_gguf` verifies `vokra.model.arch`
// strictly, validates the `vokra.itn.*` chunk group (language / direction /
// both grammar blobs / their size stamps), and hands the bytes to
// `vokra_ops::itn::ItnGrammarSet`. The pipeline itself reuses the M5-06
// `vokra_core::decode::wfst` port (tropical semiring + `Fst` +
// `read_openfst_vector`).
//
// LOUD-PARTIAL: `pipeline()` / `normalize()` return `VokraError::UnsupportedOp`
// when `vokra-ops` was built without the `vokra-wfst` feature, or when the
// stored grammars fall outside the byte-verified OpenFST shape (pynini attaches
// byte symbol tables by default, so non-zero header flags are the likely real
// case). The message names the field, its value, and the developer-side fix.
// `from_gguf` / `reorder_tagged` / `reader_gap` are fully real, and no
// fabricated normalised text is ever returned (FR-EX-08).
//
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]`).
pub mod wetextprocessing;

// Wave G (2026-08-15) `lang_id_ecapa` binder.
// Completed natively for Mac CPU/Metal on 2026-08-26. The prepared-v2
// contract keeps the real topology differences: VoxLingua107 uses 60 mel
// bins, a 256-d embedding and XVector MLP/log-softmax; CommonLanguage uses 80
// mel bins, a 192-d embedding and cosine classification. Both consume the
// reusable strict ECAPA backbone.
//
// The runtime requires all 200 embedding tensors, the exact canonical
// classifier manifest and the ordered official label array. Historical public
// embedding-only files remain explicit load errors. Conv1d, attentive-pooling
// Softmax and classifier GEMV dispatch through Compute, with no Metal→CPU
// fallback.
//
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]`).
pub mod lang_id;

// Complete native runtime for the canonical `deepfake_detection` converter
// arch (`MelodyMachine/Deepfake-audio-detection-V2`, apache-2.0). The pinned
// official config identifies a Wav2Vec2ForSequenceClassification checkpoint,
// correcting the historical WavLM scaffold.
//
// DESIGN: a detector emits a SCORE, not a verdict. There is deliberately no
// `is_fake() -> bool` — `score()` returns `DeepfakeScore` (raw logits plus a
// real, tested, numerically stable softmax) and the caller picks the
// operating point, because the right threshold depends on the base rate and
// on the relative cost of the two error directions, and both directions do
// real harm. Burying it in this crate would hide the choice from the person
// accountable for it, so `DeepfakeScore::exceeds` takes the threshold as an
// explicit argument and it shows up at the call site.
//
// Binding pins the complete 215-F32-tensor manifest, immutable provenance,
// 16 kHz normalized waveform contract, 12-layer base encoder, 256-wide
// projector and official `0=fake / 1=real` class order. The full encoder and
// task head run through the shared Compute seam on CPU or Metal. Any other
// backend fails explicitly; no learned operation silently falls back to CPU.
pub mod deepfake_detection;

// Wave G (2026-08-15) — runtime binder for the `chattts` converter arch
// (`2Noise/ChatTTS`, cc-by-nc-4.0): a dialogue-oriented TTS whose GGUFs have
// been stamped `vokra.model.arch = "chattts"` since the coverage-audit
// Wave D T4 landing with nothing in the workspace reading the tag back.
//
// LICENCE POSTURE (the point of this one): the converter's default SPDX is
// `cc-by-nc-4.0` → `LicenseClass::NonCommercial`, which requires the research
// flag, so a CORRECTLY stamped artifact is REFUSED by
// `CompliancePolicy::strict()` and loads only under an explicit research
// opt-in. That refusal is the fail-closed default working, not a defect, and
// both halves are tested — as is the unstamped case, which fails closed to
// `Unknown` and is refused for the same reason. In-tree precedent: `maest`
// (cc-by-nc-sa-4.0); publish-side precedent: X-Codec-2, the first T4 release.
//
// ELVIS ACT: the audit ticket flags ChatTTS as borderline — the 30-d `spk_emb`
// is seed-derived in the official release, but an arbitrary vector can
// technically be substituted, and the owner ADR has not been made. So this
// module exposes NO speaker-embedding injection entry point and will not grow
// one before that ADR lands. The module census only REPORTS whether a
// `spk_stat` group is on disk, which is the input the ADR needs; reporting
// presence is not the trigger, providing the injection path is.
//
// REAL: strict arch verification naming both tags and enumerating the TTS
// neighbourhood (`vocos` sharpest — ChatTTS's vocoder head IS Vocos, so tensor
// shapes genuinely overlap while loader identity does not); zero-tensor
// refusal; `require_tensor` / `require_tensor_dims` loud lookups; metadata and
// weight-licence surfacing; and the on-disk module census. The census matters
// because the PUBLISHED `vokra/chattts` repository was built from
// `asset/gpt/model.safetensors` ALONE — so the artifact a caller most likely
// holds carries the GPT backbone and neither the DVAE nor the Vocos head, and
// naming that beats dying in a missing-tensor trail.
//
// LOUD-PARTIAL: `synthesize()` returns `VokraError::UnsupportedOp`. The
// converter stamps NO `vokra.chattts.*` axis group at all — layer count,
// hidden width, head count, vocab size, DVAE codebook layout and even the
// output sample rate are unrecoverable — so every axis would have to be
// guessed, and a guessed axis is shape-valid, numerically wrong and silent.
// The error names both blockers (the un-written prep script that would pin the
// module-namespace convention, and the absent axis group), the three deferred
// modules, the on-disk census and all three primary sources. No fabricated
// waveform is ever emitted (FR-EX-08).
//
// `docs/license-audit.md` §3.1 sign-off stays BLANK (owner-only per
// `[[feedback-license-signoff-primary-source]]`).
pub mod chattts;

// Mac CPU/Metal coverage wave (2026-08-24): strict native reader and complete
// Wav2Vec2 CTC forward for the seven public `vokra/wav2vec2-*` GGUF repos.
// The module keeps the encoder-only XLSR checkpoint usable through its feature
// API and refuses the adapter-only MMS artifact explicitly.
pub mod wav2vec2_ctc;
// Mac CPU/Metal coverage wave (2026-08-24): strict public
// `facebook/hubert-large-ls960-ft` CTC binder. The arch/tensor namespace
// remains distinct; learned Wav2Vec2-family ops are shared after exact bind.
pub mod hubert;
// Mac CPU/Metal coverage wave (2026-08-24): dedicated Data2Vec Audio
// binder for its distinct names and five-layer positional-conv stack.
pub mod data2vec_audio;
// Mac CPU/Metal coverage wave (2026-08-26): strict native runtime for the
// public SpeechBrain MetricGAN+ VoiceBank GGUF. The 21-tensor generator runs
// the official Hamming-STFT / two-layer BiLSTM spectral-mask pipeline; all
// learned projections use one explicit CPU/Metal backend.
pub mod metricgan_plus;
// Mac CPU/Metal coverage wave (2026-08-26): exact 247-tensor MP-SENet DNS
// binder and native magnitude/phase enhancement graph. The released
// batch_first=false attention-axis quirk is preserved for checkpoint parity.
pub mod mp_senet;

pub use compute::{Compute, DecoderStepDims, DecoderStepSession, HotOp, make_backend};

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_vokra_core_ir() {
        // Smoke test for the crate wiring (M0-02-T02): vokra-models builds
        // model graphs on top of the vokra-core IR (and, from M0-04 on, the
        // vokra-ops operators).
        let desc = vokra_core::TensorDesc::new("logits", vokra_core::DType::F32, [1, 51_865]);
        assert_eq!(desc.num_elements(), Some(51_865));
    }
}
